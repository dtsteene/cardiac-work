# # Complete Multiscale Simulation with Prestressing (Hybrid Version)
#
# This script combines the improved active tension implementation (scifem/spatially varying)
# with robust MPI handling and logging suitable for compute clusters.

import json
import os
import csv
import time as time_module
import logging
import shutil
from pathlib import Path

# Scientific and FEniCSx imports
import scifem
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import dolfinx
import ufl
import ldrb
import adios4dolfinx

# Cardiac specific libraries
import cardiac_geometries
import cardiac_geometries.geometry
import circulation
from circulation.regazzoni2020 import Regazzoni2020
import pulse

# ============================================================================
# CONFIGURATION: Adjustable Parameters
# ============================================================================
# BPM (Beats Per Minute) Configuration - Can be set via command line or environment
import sys
import argparse

import geometry_generator

# Parse command line arguments
parser = argparse.ArgumentParser(description='Complete cardiac cycle simulation')
parser.add_argument('bpm', type=int, nargs='?', default=None, help='Heart rate in BPM (default: 75)')
parser.add_argument('--beats', type=int, default=1, help='Number of beats to run (default: 1)')
parser.add_argument('--ci', action='store_true', help='Enable CI mode (2 timesteps only for quick testing)')
parser.add_argument('--mesh', type=str, default=None, help='Path to custom XDMF mesh (optional)')
parser.add_argument('--char_length', type=float, default=5.0, help='Mesh characteristic length (default: 5.0)')
parser.add_argument('--metrics_space', type=str, default="DG0", help='Function space for metrics (e.g., DG0, DG1, Quadrature4, Quadrature6)')
parser.add_argument('--circulation_params', type=str, default=None, help='Path to JSON file with circulation parameters')
parser.add_argument('--alpha_epi', type=float, default=1e5, help='Epicardial spring stiffness (Pa/m) (default: 1e5)')
parser.add_argument('--alpha_base', type=float, default=1e6, help='Basal spring stiffness (Pa/m) (default: 1e6)')
parser.add_argument(
    '--base-dirichlet',
    choices=('x', 'none', 'full'),
    default='x',
    help='Basal Dirichlet constraint: x fixes the base-normal displacement, '
         'none uses Robin support only, full clamps all displacement components '
         '(default: x).',
)
parser.add_argument('--one-sided-robin', action='store_true', help='Use one-sided Robin BC (only resists outward displacement)')
parser.add_argument('--incompressible', action='store_true', help='Use incompressible formulation')
parser.add_argument('--geometry-dir', type=str, default=None,
                    help='Path to a pre-built geometry directory (containing geometry.bp). '
                         'Skips all geometry generation and loads directly from this path.')
parser.add_argument('--restart-from', type=str, default=None,
                    help='Path to previous results directory to continue from. '
                         'Loads displacement checkpoint + 0D state and continues for --beats more beats. '
                         'Automatically uses geometry from the restart dir.')
args = parser.parse_args()

# --- Restart Setup ---
RESTART_MODE = args.restart_from is not None
RESTART_DIR = Path(args.restart_from) if RESTART_MODE else None
if RESTART_MODE:
    # Copy geometry from restart dir into output dir (don't modify the source!)
    # The geometry_generator scaling code re-saves geometry.bp in-place,
    # which would corrupt the original results if we point at them directly.
    if args.geometry_dir is None:
        import shutil as _shutil_restart
        _restart_geo_src = RESTART_DIR / "geometry"
        # Use a unique temp name to avoid collisions with stale dirs
        _restart_geo_dst = Path(f"_restart_geo_{os.getpid()}")
        if _restart_geo_dst.exists():
            _shutil_restart.rmtree(_restart_geo_dst, ignore_errors=True)
        _shutil_restart.copytree(_restart_geo_src, _restart_geo_dst)
        args.geometry_dir = str(_restart_geo_dst)
    # Load BPM from old simulation params
    with open(RESTART_DIR / "simulation_params.json") as _f:
        _restart_sp = json.load(_f)

# Determine BPM
if args.bpm is not None:
    BPM = args.bpm
else:
    BPM = int(os.getenv("BPM", 75))

# CI Mode flag (OFF by default, must be explicitly enabled)
CI_MODE = args.ci or bool(os.getenv("CI"))

if CI_MODE:
    print(" CI MODE ENABLED - Short circuit test (2 timesteps only)")
else:
    print("✓ PRODUCTION MODE - Full simulation")

# Heart rate in Hz (BPM / 60)
HR_HZ = BPM / 60.0
# RR interval (seconds) = 1 / HR
RR_INTERVAL = 1.0 / HR_HZ

# Activation timing parameters scaled to the cardiac cycle
if BPM == 60:
    # 60 BPM: RR = 1.0s
    TC_ACTIVATION = 0.15  # Contraction duration
    TR_ACTIVATION = 0.35  # Relaxation duration
elif BPM == 75:
    # 75 BPM: RR = 0.8s
    scale_factor = RR_INTERVAL / 0.8
    TC_ACTIVATION = 0.25 * scale_factor  # ≈ 0.25s
    TR_ACTIVATION = 0.4 * scale_factor   # ≈ 0.4s
else:
    raise ValueError(f"BPM={BPM} not supported. Use 60 or 75.")

# CRITICAL FIX: Align Cardiac Cycle with End Diastole (ED)
# The FEM mesh is generated in the ED configuration (Maximum Volume).
# Therefore, we must align t=0 of the simulation with the onset of contraction.
# We set tC (start of contraction) to 0.0.
tC_ACTIVATION = 0.0

# ============================================================================
# Setup Logging and Directories (From V2 - Robust)
# ============================================================================

# Helper function to convert units
def mmHg_to_kPa(x):
    return x * 0.133322

# JSON serializer for numpy types
def custom_json(obj):
    if isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)

# Setup logging to print only from rank 0
class MPIFilter(logging.Filter):
    def __init__(self, comm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm = comm

    def filter(self, record):
        return 1 if self.comm.rank == 0 else 0

if args.mesh:
    mesh_name = Path(args.mesh).stem
    outdir = Path(f"results_{mesh_name}_hybrid_{BPM}bpm")
else:
    # Include knob settings in output directory name for sweeps
    metrics_str = args.metrics_space.lower()
    mesh_res_str = f"L{int(args.char_length)}" # L5 or L10
    outdir = Path(f"results_biv_{metrics_str}_{mesh_res_str}_{BPM}bpm")

comm = MPI.COMM_WORLD

# Rank 0 handles directory creation
if comm.rank == 0:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "solver").mkdir(exist_ok=True)
    (outdir / "visualization").mkdir(exist_ok=True)
comm.barrier()
geodir = Path(args.geometry_dir) if args.geometry_dir else None

# --- Ensure the run directory owns a geometry handle for postprocess_metrics.py ---
# postprocess_metrics.py needs <outdir>/geometry to exist (it only reads marker
# name→tag mapping from markers_geometry.json). When --geometry-dir points at an
# external shared-mesh folder, symlink it into outdir so postprocess works with
# no extra flags. Restart mode already copies its own geo.
if comm.rank == 0 and geodir is not None and not RESTART_MODE:
    _outdir_geo = outdir / "geometry"
    if not _outdir_geo.exists():
        try:
            _outdir_geo.symlink_to(geodir.resolve())
        except OSError:
            import shutil as _shutil_geo
            _shutil_geo.copytree(geodir.resolve(), _outdir_geo)
comm.barrier()

circulation.log.setup_logging(logging.INFO)
logger = logging.getLogger("pulse")
scifem_logger = logging.getLogger("scifem")
scifem_logger.setLevel(logging.WARNING)

mpi_filter = MPIFilter(comm)
logger.addFilter(mpi_filter)

if comm.rank == 0:
    logger.info("=" * 80)
    logger.info(f"CONFIGURATION: BPM = {BPM}, HR = {HR_HZ} Hz, RR = {RR_INTERVAL:.4f} s")
    logger.info(f"Activation parameters: tC={tC_ACTIVATION:.4f}s, TC={TC_ACTIVATION:.4f}s, TR={TR_ACTIVATION:.4f}s")
    logger.info("=" * 80)


# --- Geometry Generation (Hybrid: V1 Logic + V2 MPI Safety) ---

geo = geometry_generator.generate_and_load(comm, outdir, args, logger, geodir=geodir)
geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

# Store Target Volumes (ED)
volume2ml = 1e6
mesh_unit = "m"

# Helper to assist with parser_ds above which isn't defined
def parser_ds(ds_measure, marker_id):
    return ds_measure(marker_id)

if RESTART_MODE:
    # Load target volumes from old simulation params
    lvv_target = _restart_sp["lvv_target_m3"]
    rvv_target = _restart_sp["rvv_target_m3"]
    logger.info(f"RESTART: ED Volumes from old sim: LV={lvv_target * volume2ml:.2f} mL, RV={rvv_target * volume2ml:.2f} mL")
else:
    lvv_target = 0.0
    rvv_target = 0.0

    # Determine correct markers for volume calculation
    lv_vol_marker = "LV" if "LV" in geometry.markers else "ENDO_LV"
    rv_vol_marker = "RV" if "RV" in geometry.markers else "ENDO_RV"

    lvv_target = comm.allreduce(geometry.volume(lv_vol_marker), op=MPI.SUM)
    rvv_target = comm.allreduce(geometry.volume(rv_vol_marker), op=MPI.SUM)

    logger.info(
        f"ED Volumes: LV={lvv_target * volume2ml:.2f} mL, RV={rvv_target * volume2ml:.2f} mL",
    )

# --- 0D Circulation Model (From V2) ---

# Helper to update parameters containing units
def update_parameters_from_json(params, json_params):
    """
    Recursively update parameters from JSON dict, preserving units if they exist in params.
    """
    ureg = circulation.units.ureg
    
    for key, value in json_params.items():
        if key in params:
            if isinstance(value, dict) and isinstance(params[key], dict):
                update_parameters_from_json(params[key], value)
            else:
                # Update value, preserving unit if target has one
                if hasattr(params[key], "units"):
                    original_unit = params[key].units
                    # JSON value is number, attach original unit
                    params[key] = value * original_unit
                else:
                    params[key] = value
        else:
             # Key in JSON but not in defaults. Add it.
             params[key] = value

_json_initial_state = None  # Set by get_updated_parameters() if JSON has initial_state

def get_updated_parameters():
    """
    Returns Regazzoni2020 parameters consistent with the configured BPM.
    Reference defaults are for 75 BPM (RR=0.8s).

    CRITICAL ALIGNMENT:
    We shift the 0D model phase so that Ventricular Contraction starts at t=0.
    This ensures alignment with the FEM mesh which is at End Diastole.
    """
    params = Regazzoni2020.default_parameters()

    # Load from JSON if provided
    if args.circulation_params:
        p_path = Path(args.circulation_params)
        if p_path.exists():
            if comm.rank == 0:
                logger.info(f"Loading circulation parameters from {p_path}")

            with open(p_path) as f:
                data = json.load(f)

            # JSON file structure is {"parameters": {...}} or just {...}
            # We look for "parameters" key first
            json_params = data.get("parameters", data)

            update_parameters_from_json(params, json_params)

            # Store initial_state from JSON for 0D pre-run (if converged ICs available)
            global _json_initial_state
            _json_initial_state = data.get("initial_state", None)
            if _json_initial_state and comm.rank == 0:
                converged = data.get("initial_state_converged", False)
                logger.info(f"  Loaded initial_state from JSON (converged={converged})")

    # Scale factor relative to reference RR=0.8s (75 BPM)
    factor = RR_INTERVAL / 0.8

    # Default Phases (at 75 BPM):
    # LV tC = 0.1s
    # LA tC = 0.9s (Previous beat relative to 0? No, this is wrapped.
    #              In Regazzoni code, tC is just a parameter.
    #              Modulo arithmetic places 0.9 near the end of the 0.8s cycle -> 0.1s into next?
    #              Wait. 0.9 mod 0.8 = 0.1. So LA contracts at 0.1?? same as LV?
    #              Let's check Regazzoni defaults carefully.
    #              default_parameters has "tC": 0.9 * s for LA.
    #              If RR=0.8, then 0.9 is 0.1s into the cycle.
    #              If LV tC=0.1. Then LA and LV contract simultaneously? That's wrong.
    #              Let's assume the Regazzoni defaults imply a specific relative timing.
    #              Usually Atrial contraction is BEFORE Ventricular.
    #              If LA tC = 0.9 and RR = 1.0 (hypothetically), then it's late diastole.
    #              Let's use the AV delay (time between LA and LV start).
    #              We will simply FORCE LV tC to 0.0, and shift others relative to it.

    # Reference timing to shift FROM:
    ref_LV_tC = 0.1 * factor # Scale the reference start time too

    # Shift needed to move LV tC to 0.0
    time_shift = -ref_LV_tC

    # Apply Scaling and Shifting
    for chamber in ["LA", "RA", "LV", "RV"]:
        # Scale original tC
        original_tC = params["chambers"][chamber]["tC"].magnitude # Removing unit for calc
        scaled_tC = original_tC * factor

        # Shift
        new_tC = scaled_tC + time_shift

        # Wrap to [0, RR) to be safe (though formula handles negatives usually)
        # Using modulo manually to ensure parametrics are clean
        # new_tC = new_tC % RR_INTERVAL
        # Actually, let's keep it linear, the model handles modulo.

        params["chambers"][chamber]["tC"] = new_tC * circulation.units.ureg("s")

        # Scale durations
        params["chambers"][chamber]["TC"] *= factor
        params["chambers"][chamber]["TR"] *= factor

    # Explicit Overrides for Ventricles to match FEM configuration strictly
    for chamber in ["LV", "RV"]:
        params["chambers"][chamber]["tC"] = 0.0 * circulation.units.ureg("s")
        params["chambers"][chamber]["TC"] = TC_ACTIVATION * circulation.units.ureg("s")
        params["chambers"][chamber]["TR"] = TR_ACTIVATION * circulation.units.ureg("s")

    # Update HR
    params["HR"] = circulation.units.ureg(f"{HR_HZ} Hz")

    return params

def run_0D(init_state, nbeats=10):
    logger.info("Running 0D circulation model to steady state...")
    # Use parameters consistent with BPM
    params = get_updated_parameters()
    model = Regazzoni2020(parameters=params)

    history = model.solve(num_beats=nbeats, initial_state=init_state)
    state = dict(zip(model.state_names(), model.state))
    return history, state

init_state_circ = {
   "V_LV": lvv_target * volume2ml * circulation.units.ureg("mL"),
    "V_RV": rvv_target * volume2ml * circulation.units.ureg("mL"),
}

if not RESTART_MODE:
    # --- FRESH RUN: run standalone 0D to get initial circulation state ---
    if comm.rank == 0:
        # Use converged ICs from JSON if available, otherwise model defaults
        ic_0d = _json_initial_state
        if ic_0d is not None:
            logger.info(f"Using initial_state from JSON for 0D pre-run")
        else:
            logger.info(f"No initial_state in JSON, using model defaults for 0D pre-run")
        history, circ_state = run_0D(init_state=ic_0d)
        np.save(outdir / "state.npy", circ_state, allow_pickle=True)
        np.save(outdir / "history.npy", history, allow_pickle=True)
    comm.Barrier()

    history = np.load(outdir / "history.npy", allow_pickle=True).item()
    circ_state = np.load(outdir / "state.npy", allow_pickle=True).item()
else:
    # --- RESTART: extract 0D state and ratios from old simulation ---
    logger.info("RESTART: Skipping 0D pre-run (will use state from previous coupled simulation)")
    history = None  # Will be replaced by coupled_history

    # Extract final 0D state from old coupled history
    restart_history_tmp = np.load(RESTART_DIR / "circulation" / "history.npy", allow_pickle=True).item()
    circ_state_names = ["V_LA", "V_LV", "V_RA", "V_RV", "p_AR_SYS", "p_VEN_SYS",
                        "p_AR_PUL", "p_VEN_PUL", "Q_AR_SYS", "Q_VEN_SYS", "Q_AR_PUL", "Q_VEN_PUL"]
    circ_state = {k: float(restart_history_tmp[k][-1]) for k in circ_state_names if k in restart_history_tmp}
    logger.info(f"RESTART: Extracted 0D state from old history (V_LV={circ_state['V_LV']:.2f}, V_RV={circ_state['V_RV']:.2f})")

    # Store restart targets for the inflation ramp (used after problem setup)
    last_V_LV_0D = circ_state["V_LV"]
    last_V_RV_0D = circ_state["V_RV"]
    old_Ta_solver_history = np.load(RESTART_DIR / "solver" / "Ta_solver_history.npy")

error_LV = 0.0 # Deprecated: Offset removed in favor of Ratio Coupling
error_RV = 0.0 # Deprecated: Offset removed in favor of Ratio Coupling

# Scaling Ratios for Multiplicative Coupling (Ratio = Mesh_ED / Circ_ED)
if RESTART_MODE:
    # Load ratios from old simulation to ensure consistency
    ratio_LV = _restart_sp["ratio_LV"]
    ratio_RV = _restart_sp["ratio_RV"]
else:
    ratio_LV = init_state_circ["V_LV"].magnitude / circ_state["V_LV"]
    ratio_RV = init_state_circ["V_RV"].magnitude / circ_state["V_RV"]

if comm.rank == 0:
    logger.info(f"Coupling Ratios (Mesh/Circ): LV={ratio_LV:.4f}, RV={ratio_RV:.4f}")

# Plotting 0D results (Rank 0 only, skip for restart)
if comm.rank == 0 and history is not None:
    fig, ax = plt.subplots(2, 3, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1, 0.6]})
    fig.suptitle(f"0D Circulation Model — {BPM} BPM", fontsize=14, fontweight="bold")

    # PV loops
    ax[0, 0].plot(history["V_LV"], history["p_LV"], "tab:blue")
    ax[0, 0].set_title("LV — All beats"); ax[0, 0].set_ylabel("Pressure (mmHg)")
    ax[0, 1].plot(history["V_LV"][-1000:], history["p_LV"][-1000:], "tab:blue")
    ax[0, 1].set_title("LV — Last beat")
    ax[1, 0].plot(history["V_RV"], history["p_RV"], "tab:red")
    ax[1, 0].set_title("RV — All beats"); ax[1, 0].set_ylabel("Pressure (mmHg)")
    ax[1, 0].set_xlabel("Volume (mL)")
    ax[1, 1].plot(history["V_RV"][-1000:], history["p_RV"][-1000:], "tab:red")
    ax[1, 1].set_title("RV — Last beat"); ax[1, 1].set_xlabel("Volume (mL)")
    for a in ax[:, :2].flat:
        a.grid(True, alpha=0.3)

    # Summary stats table (last beat)
    def _stats(v, p):
        v_lb, p_lb = np.array(v[-1000:]), np.array(p[-1000:])
        edv, esv = float(np.max(v_lb)), float(np.min(v_lb))
        sv = edv - esv
        ef = sv / edv * 100 if edv > 0 else 0
        esp = float(np.max(p_lb))
        edp = float(p_lb[np.argmax(v_lb)])
        return edv, esv, sv, ef, esp, edp

    lv = _stats(history["V_LV"], history["p_LV"])
    rv = _stats(history["V_RV"], history["p_RV"])
    co = lv[2] * BPM / 1000  # SV * HR in L/min

    rows = [
        ["", "LV", "RV"],
        ["EDV (mL)", f"{lv[0]:.1f}", f"{rv[0]:.1f}"],
        ["ESV (mL)", f"{lv[1]:.1f}", f"{rv[1]:.1f}"],
        ["SV (mL)",  f"{lv[2]:.1f}", f"{rv[2]:.1f}"],
        ["EF (%)",   f"{lv[3]:.1f}", f"{rv[3]:.1f}"],
        ["ESP (mmHg)", f"{lv[4]:.1f}", f"{rv[4]:.1f}"],
        ["EDP (mmHg)", f"{lv[5]:.1f}", f"{rv[5]:.1f}"],
        ["CO (L/min)", f"{co:.2f}", ""],
        ["HR (bpm)", f"{BPM}", ""],
    ]
    for a in ax[:, 2]:
        a.axis("off")
    tbl = ax[0, 2].table(cellText=rows[1:], colLabels=rows[0],
                          loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.4)
    ax[0, 2].set_title("Hemodynamic Summary", fontweight="bold", fontsize=11)

    fig.tight_layout()
    fig.savefig(outdir / "0D_circulation_pv.png", dpi=150)
    plt.close(fig)

# --- Activation Model (From V1: Vectorized for Scifem) ---

# CRITICAL FIX: The cycle is now aligned so t=0 is End Diastole (tC=0).
# We do NOT use the previous shifting logic.
tc_shifted = tC_ACTIVATION # Should be 0.0

if comm.rank == 0:
    logger.info(f"Cardiac cycle timing (t=0 aligned to Contraction Onset/ED):")
    logger.info(f"  Contraction:     {tc_shifted:.4f} → {tc_shifted + TC_ACTIVATION:.4f} s")
    logger.info(f"  Relaxation:      {tc_shifted + TC_ACTIVATION:.4f} → {tc_shifted + TC_ACTIVATION + TR_ACTIVATION:.4f} s")
    logger.info(f"  Rest (Filling):  {tc_shifted + TC_ACTIVATION + TR_ACTIVATION:.4f} → {RR_INTERVAL:.4f} s")

def get_activation(t):
    """
    Returns spatially-varying active tension [LV, Septum, RV] scaled appropriately.
    """
    # Use the configured activation parameters based on BPM
    value = circulation.time_varying_elastance.blanco_ventricle(
        EA=1.0,
        EB=0.0,
        tC=tc_shifted,          # 0.0
        TC=TC_ACTIVATION,
        TR=TR_ACTIVATION,
        RR=RR_INTERVAL,
    )(t)
    # V1 Logic: Return array for spatially varying tension
    return np.array([100 * value, 100 * value, 100 * value])

if comm.rank == 0:
    fig, ax = plt.subplots(figsize=(12, 5))
    # Plot one full cardiac cycle starting from ED
    t = np.linspace(0, RR_INTERVAL, 200)
    activation_curve = get_activation(t)
    ax.plot(t, activation_curve.T, label=['LV', 'RV', 'Septum'], linewidth=2)

    # Mark cardiac cycle phases
    contraction_end = tc_shifted + TC_ACTIVATION
    relaxation_end = tc_shifted + TC_ACTIVATION + TR_ACTIVATION

    ax.axvspan(tc_shifted, contraction_end, alpha=0.1, color='red', label='Contraction')
    ax.axvspan(contraction_end, relaxation_end, alpha=0.1, color='blue', label='Relaxation')
    ax.axvspan(relaxation_end, RR_INTERVAL, alpha=0.1, color='green', label='Rest/Filling')

    ax.set_xlabel(f"Time (s) - {BPM} BPM", fontsize=12)
    ax.set_ylabel("Activation (kPa)", fontsize=12)
    ax.set_title(f"Activation Curve ({BPM} BPM) - Aligned to ED at t=0", fontsize=13, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 110])

    fig.savefig(outdir / "visualization" / "activation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


# --- Setup Problem (From V1: Uses Scifem and markers_mt) ---

def setup_problem(
    geometry,
    f0,
    s0,
    material_params,
    alpha_epi_val=1e5,
    alpha_base_val=1e6,
    incompressible=False,
    base_dirichlet="x",
):
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)

    # Use scifem to create simple function space based on markers_mt
    # markers_mt was saved in additional_data in geometry step
    markers_mt = geo.additional_data["markers_mt"]

    # Create function space for active tension
    V_Ta = scifem.create_space_of_simple_functions(
        mesh=geo.mesh,
        cell_tag=markers_mt,
        tags=[1, 2, 3] # Tags for LV, RV, Septum (check LDRB output for specific indices)
    )

    Ta = pulse.Variable(dolfinx.fem.Function(V_Ta), "kPa")
    active_model = pulse.ActiveStress(f0, activation=Ta)
    
    if incompressible:
        comp_model = pulse.compressibility.Incompressible()
        if comm.rank == 0:
            print("Using Incompressible formulation")
    else:
        comp_model = pulse.compressibility.Compressible2()
        if comm.rank == 0:
            print("Using Compressible formulation")

    model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    alpha_epi = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(alpha_epi_val)), "Pa / m",
    )
    robin_epi = pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])  # one_sided removed for pulse 0.6 compat

    alpha_base = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(alpha_base_val)), "Pa / m",
    )
    robin_base = pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])  # one_sided removed for pulse 0.6 compat
    robin = [robin_epi, robin_base]

    def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
        facets = geometry.facet_tags.find(geometry.markers["BASE"][0])

        if base_dirichlet == "full":
            dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
            u_zero = dolfinx.fem.Function(V)
            u_zero.x.array[:] = 0.0
            return [dolfinx.fem.dirichletbc(u_zero, dofs)]
        
        try:
            # --- CASE 1: Standard / Compressible ---
            # Try to access the X-component subspace (.sub(0))
            # This works for standard VectorFunctionSpaces
            V_x = V.sub(0)
            dofs = dolfinx.fem.locate_dofs_topological(V_x, 2, facets)
            return [dolfinx.fem.dirichletbc(0.0, dofs, V_x)]
            
        except AssertionError:
            # --- CASE 2: Incompressible (Mixed Space) ---
            # The AssertionError "num_sub_elements > i" confirms we are in a 
            # nested subspace where .sub(0) is not accessible.
            
            # We cannot isolate X, so we apply a Full Clamp (0,0,0) to the base.
            # This prevents sliding and rigid body motion.
            
            if comm.rank == 0:
                print("Warning: Incompressible Mode detected - Applying Full Base Clamp (0,0,0)")
            
            # Locate DOFs for the FULL vector on the base
            dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
            
            # Create a vector zero function (0,0,0)
            u_zero = dolfinx.fem.Function(V)
            u_zero.x.array[:] = 0.0
            
            # Apply BC to the full vector space V
            return [dolfinx.fem.dirichletbc(u_zero, dofs)]

    if base_dirichlet == "none":
        dirichlet_bcs = ()
    else:
        dirichlet_bcs = (dirichlet_bc,)

    return model, robin, dirichlet_bcs, Ta


material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
# Use Compressible for Prestressing always (Hybrid Strategy)
model, robin, dirichlet_bcs, Ta = setup_problem(
    geometry=geometry, f0=geo.f0, s0=geo.s0, material_params=material_params,
    alpha_epi_val=args.alpha_epi, alpha_base_val=args.alpha_base,
    incompressible=False, base_dirichlet=args.base_dirichlet,
)

# --- Prestressing (Inverse Elasticity) ---

if RESTART_MODE:
    # Use dummy pressures — prestress is cached and won't be recomputed
    p_LV_ED = 0.0
    p_RV_ED = 0.0
else:
    p_LV_ED = mmHg_to_kPa(history["p_LV"][-1])
    p_RV_ED = mmHg_to_kPa(history["p_RV"][-1])

logger.info(f"Target ED Pressures: p_LV={p_LV_ED:.2f} kPa, p_RV={p_RV_ED:.2f} kPa")

pressure_lv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa")
pressure_rv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa")

# FIX: Use ENDO_LV/RV markers directly for surface traction if "LV"/"RV" missing
lv_marker_name = "LV" if "LV" in geometry.markers else "ENDO_LV"
rv_marker_name = "RV" if "RV" in geometry.markers else "ENDO_RV"

lv_marker_id = geometry.markers[lv_marker_name][0] 
rv_marker_id = geometry.markers[rv_marker_name][0]

neumann_lv = pulse.NeumannBC(traction=pressure_lv, marker=lv_marker_id)
neumann_rv = pulse.NeumannBC(traction=pressure_rv, marker=rv_marker_id)

bcs_prestress = pulse.BoundaryConditions(
    robin=robin, dirichlet=dirichlet_bcs, neumann=(neumann_lv, neumann_rv),
)

solver_dir = outdir / "solver"
viz_dir = outdir / "visualization"

# For restart: copy cached prestress files from old results dir (rank 0 only to avoid race)
if RESTART_MODE and comm.rank == 0:
    import shutil as _shutil
    for _pf in ["prestress_inverse.bp", "prestress_backward.bp"]:
        _src = RESTART_DIR / "solver" / _pf
        _dst = solver_dir / _pf
        if _src.exists() and not _dst.exists():
            if _src.is_dir():
                _shutil.copytree(_src, _dst)
            else:
                _shutil.copy2(_src, _dst)
            logger.info(f"RESTART: Copied {_pf} from old results")
if RESTART_MODE:
    comm.barrier()

prestress_fname = solver_dir / "prestress_inverse.bp"
if not prestress_fname.exists():
    logger.info("Start prestressing (Using Compressible Formulation for Stability)...")
    prestress_problem = pulse.unloading.PrestressProblem(
        geometry=geometry,
        model=model,
        bcs=bcs_prestress,
        parameters={"u_space": "P_2", "mesh_unit": mesh_unit},
        targets=[
            pulse.unloading.TargetPressure(traction=pressure_lv, target=p_LV_ED, name="LV"),
            pulse.unloading.TargetPressure(traction=pressure_rv, target=p_RV_ED, name="RV"),
        ],
        ramp_steps=20,
    )
    u_pre = prestress_problem.unload()
    adios4dolfinx.write_function_on_input_mesh(prestress_fname, u_pre, time=0.0, name="u_pre")
    with dolfinx.io.VTXWriter(
        comm, solver_dir / "prestress_backward.bp", [u_pre], engine="BP4",
    ) as vtx:
        vtx.write(0.0)

# --- Forward Problem Setup ---

V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 2, (3,)))
u_pre = dolfinx.fem.Function(V)
adios4dolfinx.read_function(prestress_fname, u_pre, time=0.0, name="u_pre")

logger.info("Deforming mesh to Reference Configuration...")
geometry.deform(u_pre)

logger.info("Mapping fibers to Reference Configuration...")
f0_quad = pulse.utils.map_vector_field(f=geo.f0, u=u_pre, normalize=True, name="f0_unloaded")
s0_quad = pulse.utils.map_vector_field(f=geo.s0, u=u_pre, normalize=True, name="s0_unloaded")

# Map n0 and l0 for saving with checkpoint (same file = same DOF ordering)
n0_quad = None
if geo.n0 is not None:
    n0_quad = pulse.utils.map_vector_field(f=geo.n0, u=u_pre, normalize=True, name="n0_unloaded")
l0_field = geo.additional_data.get("apex_gradient", None)
f0_map = pulse.utils.map_vector_field(
    geo.additional_data["f0_DG_1"], u=u_pre, normalize=True, name="f0",
)
s0_map = pulse.utils.map_vector_field(
    geo.additional_data.get("s0_DG_1", geo.s0), u=u_pre, normalize=True, name="s0",
)

# Robust Volume Calculation (Handle missing 'LV' tags)
x = ufl.SpatialCoordinate(geometry.mesh)
n = ufl.FacetNormal(geometry.mesh)

# Determine correct markers for volume calculation
lv_marker = "LV" if "LV" in geometry.markers else "ENDO_LV"
rv_marker = "RV" if "RV" in geometry.markers else "ENDO_RV"

lvv_unloaded = comm.allreduce(geometry.volume(lv_marker), op=MPI.SUM)
rvv_unloaded = comm.allreduce(geometry.volume(rv_marker), op=MPI.SUM)

logger.info(f"Unloaded volumes: LV={lvv_unloaded * volume2ml:.2f} mL, RV={rvv_unloaded * volume2ml:.2f} mL")

if args.incompressible:
    logger.info("Warning: Hybrid Prestressing Strategy Active (Compressible Unloading -> Incompressible Forward)")

model, robin, dirichlet_bcs, Ta = setup_problem(
    geometry=geometry, f0=f0_quad, s0=s0_quad, material_params=material_params,
    alpha_epi_val=args.alpha_epi, alpha_base_val=args.alpha_base,
    incompressible=args.incompressible, base_dirichlet=args.base_dirichlet,
)

lv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(lvv_unloaded))
rv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(rvv_unloaded))

# Note on Cavity for Incompressible:
# pulse.problem.Cavity usually handles u from Mixed Space automatically if passed the right function
cavities = [
    pulse.problem.Cavity(marker=lv_marker, volume=lv_volume),
    pulse.problem.Cavity(marker=rv_marker, volume=rv_volume),
]

bcs_forward = pulse.BoundaryConditions(robin=robin, dirichlet=dirichlet_bcs)

problem = pulse.problem.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs_forward,
    cavities=cavities,
    parameters={"mesh_unit": mesh_unit, "u_space": "P_2"},
)

# Extract Displacement for Post-Processing
if args.incompressible:
    # In mixed space, problem.u returns the displacement sub-function
    u_disp = problem.u 
else:
    u_disp = problem.u

# Setup Stress/Strain Post-processing - kinematics
# FIXED: Use full CardiacModel (material + compressibility) instead of material only
# This ensures stresses include pressure contribution for proper boundary work calculation

W = dolfinx.fem.functionspace(geometry.mesh, ("DG", 1))
I = ufl.Identity(3)
F = ufl.variable(ufl.grad(u_disp) + I)
C = F.T * F
E = 0.5 * (C - I)
f_map = (F * f0_map) / ufl.sqrt(ufl.inner(F * f0_map, F * f0_map))

# For fiber stress visualization only: Create a simple standalone material
# (We use problem.model in MetricsCalculator for actual work calculations)
material_viz = pulse.HolzapfelOgden(f0=f0_map, s0=f0_map, **material_params)
T_viz = material_viz.sigma(F)  # Simple passive stress for visualization only

fiber_stress = dolfinx.fem.Function(W, name="fiber_stress")
fiber_stress_expr = dolfinx.fem.Expression(ufl.inner(T_viz * f_map, f_map), W.element.interpolation_points)
fiber_strain = dolfinx.fem.Function(W, name="fiber_strain")
fiber_strain_expr = dolfinx.fem.Expression(ufl.inner(E * f0_map, f0_map), W.element.interpolation_points)

# Writers
# Visualization writers (ParaView)
vtx_u = dolfinx.io.VTXWriter(geometry.mesh.comm, viz_dir / "displacement.bp", [u_disp], engine="BP4")
vtx_p = None
if args.incompressible:
    vtx_p = dolfinx.io.VTXWriter(geometry.mesh.comm, viz_dir / "pressure.bp", [problem.p], engine="BP4")
vtx_stress = dolfinx.io.VTXWriter(geometry.mesh.comm, viz_dir / "stress_strain.bp", [fiber_stress, fiber_strain, Ta.value], engine="BP4")

# --- Inflation (Reference -> End-Diastole) ---

if not RESTART_MODE:
    logger.info("Inflating to End-Diastolic Target...")
    ramp_steps = 20
    for i in range(ramp_steps):
        factor = (i + 1) / ramp_steps
        current_lvv = lvv_unloaded + factor * (lvv_target - lvv_unloaded)
        current_rvv = rvv_unloaded + factor * (rvv_target - rvv_unloaded)
        lv_volume.value = current_lvv
        rv_volume.value = current_rvv
        problem.solve()

        plv = problem.cavity_pressures[0].x.array[0] * 1e-3
        prv = problem.cavity_pressures[1].x.array[0] * 1e-3
        if comm.rank == 0:
            logger.info(f"Inflation Step {i + 1}/{ramp_steps}: pLV={plv:.2f} kPa, pRV={prv:.2f} kPa")

    vtx_u.write(0.0)
    if vtx_p:
        vtx_p.write(0.0)
    vtx_stress.write(0.0)

    # Store old values (handling Array for Ta due to scifem/V1)
    problem.old_Ta = Ta.value.x.array.copy()
    problem.old_lv_volume = lv_volume.value.copy()
    problem.old_rv_volume = rv_volume.value.copy()
else:
    # --- RESTART: Inflate normally, then ramp from ED to the restart state ---
    # We can't skip inflation because the Lagrange multiplier (cavity pressure)
    # needs to be established by the solver. After inflation, we ramp volumes
    # and activation from ED to the restart state so the solver converges smoothly.
    logger.info("RESTART: Inflating to ED (same as fresh run)...")
    ramp_steps = 20
    for i in range(ramp_steps):
        factor = (i + 1) / ramp_steps
        current_lvv = lvv_unloaded + factor * (lvv_target - lvv_unloaded)
        current_rvv = rvv_unloaded + factor * (rvv_target - rvv_unloaded)
        lv_volume.value = current_lvv
        rv_volume.value = current_rvv
        problem.solve()

        plv = problem.cavity_pressures[0].x.array[0] * 1e-3
        prv = problem.cavity_pressures[1].x.array[0] * 1e-3
        if comm.rank == 0:
            logger.info(f"Inflation Step {i + 1}/{ramp_steps}: pLV={plv:.2f} kPa, pRV={prv:.2f} kPa")

    problem.old_Ta = Ta.value.x.array.copy()
    problem.old_lv_volume = lv_volume.value.copy()
    problem.old_rv_volume = rv_volume.value.copy()

    # Now ramp from ED to the restart state (volumes + activation)
    restart_lv_target = (last_V_LV_0D * ratio_LV) / volume2ml
    restart_rv_target = (last_V_RV_0D * ratio_RV) / volume2ml
    restart_Ta_target = old_Ta_solver_history[-1]
    ed_lv = lv_volume.value.copy()
    ed_rv = rv_volume.value.copy()
    ed_Ta = Ta.value.x.array.copy()

    # Only ramp if the restart state differs from ED
    vol_diff = abs(restart_lv_target - ed_lv) * volume2ml
    ta_diff = np.max(np.abs(restart_Ta_target - ed_Ta))
    if vol_diff > 0.01 or ta_diff > 0.01:  # > 0.01 mL or 0.01 kPa
        n_ramp = 20
        logger.info(f"RESTART: Ramping from ED to restart state ({n_ramp} steps, "
                     f"dV_LV={vol_diff:.2f}mL, dTa_max={ta_diff:.1f}kPa)...")
        for ri in range(n_ramp):
            frac = (ri + 1) / n_ramp
            lv_volume.value = ed_lv + frac * (restart_lv_target - ed_lv)
            rv_volume.value = ed_rv + frac * (restart_rv_target - ed_rv)
            Ta.assign(ed_Ta + frac * (restart_Ta_target - ed_Ta))
            problem.solve()
        problem.old_lv_volume = lv_volume.value.copy()
        problem.old_rv_volume = rv_volume.value.copy()
        problem.old_Ta = Ta.value.x.array.copy()
        logger.info(f"RESTART: Ramp complete. V_LV={lv_volume.value*volume2ml:.2f}mL, "
                     f"Ta_max={np.max(Ta.value.x.array):.1f}")
    else:
        logger.info("RESTART: State at ED matches restart state, no ramp needed")

# --- Multiscale Coupling Loop (Hybrid Logic) ---

def p_BiV_func(V_LV, V_RV, t):
    """
    Coupling function: receives volumes (mL) and time from the 0D model,
    drives the FEM to the corresponding state, returns pressures (mmHg).
    """
    logger.info(f"Coupling Time {t:.4f}: Target V_LV={V_LV:.2f}, V_RV={V_RV:.2f}")

    # --- Compute targets ---
    # Convert 0D volumes (mL) to FEM volumes (m³) with ratio scaling
    new_value_LV = (V_LV * ratio_LV) * (1.0 / volume2ml)
    new_value_RV = (V_RV * ratio_RV) * (1.0 / volume2ml)
    new_value_Ta = get_activation(t)

    # --- Load previous converged state ---
    old_lv_volume = problem.old_lv_volume
    old_rv_volume = problem.old_rv_volume
    old_Ta = problem.old_Ta

    dLV = new_value_LV - old_lv_volume
    dRV = new_value_RV - old_rv_volume
    dTa = new_value_Ta - old_Ta

    tol = 1e-12

    # Skip FEM solve entirely if nothing changed (e.g. rest phase)
    if abs(dLV) > tol or abs(dRV) > tol or np.max(np.abs(dTa)) > tol:

        # Iteration tracking: these advance as sub-steps succeed,
        # giving us a "checkpoint" to reset to on failure
        old_lv_it = old_lv_volume.copy()
        old_rv_it = old_rv_volume.copy()
        old_Ta_it = old_Ta.copy()

        converged = False
        num_failures = 0
        num_steps = 1  # Start optimistic: try one full step
        max_failures = 20

        while not converged and num_failures < max_failures:

            # Snapshot before this attempt so we can detect progress
            snapshot_lv = old_lv_it.copy()

            for i in range(num_steps):
                frac = (i + 1) / num_steps
                lv_volume.value = old_lv_it + frac * (new_value_LV - old_lv_it)
                rv_volume.value = old_rv_it + frac * (new_value_RV - old_rv_it)
                Ta.assign(old_Ta_it + frac * (new_value_Ta - old_Ta_it))

                try:
                    problem.solve()
                except RuntimeError as e:
                    print(f"Error during solve: {e}")

                    # Reset FEM to last known good state so Newton
                    # starts from a converged configuration next attempt
                    lv_volume.value = old_lv_it.copy()
                    rv_volume.value = old_rv_it.copy()
                    Ta.assign(old_Ta_it)
                    problem.reset()

                    num_failures += 1
                    converged = False
                    break
                else:
                    # Sub-step succeeded: advance the checkpoint
                    converged = True
                    old_lv_it = lv_volume.value.copy()
                    old_rv_it = rv_volume.value.copy()
                    old_Ta_it = Ta.value.x.array.copy()

            # All sub-steps succeeded, we're done
            if converged:
                break

            # Decide step size for next attempt
            made_progress = not np.isclose(old_lv_it, snapshot_lv)
            if made_progress:
                # Some sub-steps worked before we failed.
                # The remaining range is already smaller, just bisect it.
                num_steps = 2
            else:
                # Failed on the very first sub-step, no progress at all.
                # The step size itself is too large, must halve it.
                num_steps *= 2

        if not converged:
            msg = (
                f"Failed to converge after {num_failures} attempts. "
                f"LV: {new_value_LV}, RV: {new_value_RV}, "
                f"Ta remaining: {np.max(np.abs(new_value_Ta - old_Ta_it)):.4f}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

    # --- Save state for next coupling call ---
    problem.old_Ta = Ta.value.x.array.copy()
    problem.old_lv_volume = lv_volume.value.copy()
    problem.old_rv_volume = rv_volume.value.copy()

    # --- Extract pressures and return ---
    # Lagrange multipliers come out in Pa, convert to mmHg for 0D model
    lv_p_kPa = problem.cavity_pressures[0].x.array[0] * 1e-3
    rv_p_kPa = problem.cavity_pressures[1].x.array[0] * 1e-3

    return circulation.units.kPa_to_mmHg(lv_p_kPa), circulation.units.kPa_to_mmHg(rv_p_kPa)


# --- Checkpointing and Callback ---

checkpoint_file = solver_dir / "checkpoint.bp"
restart_time_offset = 0.0

if RESTART_MODE:
    # --- RESTART: Load state from previous simulation ---
    restart_checkpoint = RESTART_DIR / "solver" / "checkpoint.bp"
    restart_history = np.load(RESTART_DIR / "circulation" / "history.npy", allow_pickle=True).item()

    # 1. Determine time offset from old checkpoint
    old_timestamps = adios4dolfinx.read_timestamps(restart_checkpoint, comm, "displacement")
    restart_time_offset = float(old_timestamps[-1])
    logger.info(f"RESTART: Continuing from t={restart_time_offset:.4f}s ({len(old_timestamps)} old steps)")

    # 2. State restoration already done via inflation + ramp (above).
    #    The solver found its own path to the restart state, so displacement
    #    and Lagrange multipliers are internally consistent.

    # 3. Load previous histories to prepend
    Ta_history = np.load(RESTART_DIR / "solver" / "Ta_history.npy").tolist()
    Ta_solver_history = np.load(RESTART_DIR / "solver" / "Ta_solver_history.npy").tolist()
    pressure_history = np.load(RESTART_DIR / "solver" / "solver_cavity_pressure_mmHg.npy").tolist()
    logger.info(f"RESTART: Loaded {len(Ta_history)} previous history steps")

    # 4. Copy old checkpoint directory so we can append new steps to it
    if comm.rank == 0:
        shutil.rmtree(checkpoint_file, ignore_errors=True)
        shutil.copytree(restart_checkpoint, checkpoint_file)
        logger.info(f"RESTART: Copied checkpoint directory ({len(old_timestamps)} existing steps)")
    comm.barrier()

else:
    # --- FRESH RUN: Initialize checkpoint from scratch ---
    if comm.rank == 0:
        shutil.rmtree(checkpoint_file, ignore_errors=True)
    comm.barrier()

    adios4dolfinx.write_mesh(checkpoint_file, geometry.mesh)
    adios4dolfinx.write_meshtags(checkpoint_file, mesh=geometry.mesh, meshtags=geometry.facet_tags, meshtag_name="ffun")
    adios4dolfinx.write_meshtags(checkpoint_file, mesh=geometry.mesh, meshtags=geo.additional_data["markers_mt"], meshtag_name="cfun")

    adios4dolfinx.write_function(checkpoint_file, u=f0_quad, name="f0", time=0.0)
    adios4dolfinx.write_function(checkpoint_file, u=s0_quad, name="s0", time=0.0)
    if n0_quad is not None:
        adios4dolfinx.write_function(checkpoint_file, u=n0_quad, name="n0", time=0.0)
    if l0_field is not None:
        adios4dolfinx.write_function(checkpoint_file, u=l0_field, name="l0", time=0.0)
    logger.info("Checkpoint initialized: mesh + markers + fibers")

    Ta_history: list = []
    Ta_solver_history: list = []
    pressure_history: list = []  # Solver cavity pressures (mmHg) at each step

def callback(model, i: int, t: float, save=True):
    # Apply time offset for restart continuations
    t_abs = t + restart_time_offset

    # 1. Record activation for postprocessing
    raw_activation_vec = get_activation(t)
    Ta_history.append(raw_activation_vec)

    # Also record the actual solver Ta (may differ slightly due to substep convergence)
    solver_ta_array = Ta.value.x.array[:]
    Ta_solver_history.append(solver_ta_array.copy())

    # Record solver cavity pressures (the Lagrange multiplier — exact surface traction).
    # These are the pressures actually applied to the FEM boundary, NOT the 0D ODE
    # output (which is 1 timestep ahead due to staggered coupling).
    # Saved as solver_cavity_pressure_mmHg.npy (renamed from pressure_history.npy).
    lv_p_Pa = float(problem.cavity_pressures[0].x.array[0])
    rv_p_Pa = float(problem.cavity_pressures[1].x.array[0])
    pressure_history.append([lv_p_Pa * 1e-3 / 0.133322, rv_p_Pa * 1e-3 / 0.133322])  # Pa -> mmHg

    max_ta_solver = np.max(solver_ta_array)

    # 2. Console Feedback (lightweight — no metrics computation)
    if comm.rank == 0 and (i % 10 == 0 or CI_MODE):
        lv_p_kPa = problem.cavity_pressures[0].x.array[0] * 1e-3
        v_lv_ml = float(lv_volume.value * volume2ml)
        print(f"STEP {i:04d} | t={t_abs:.3f} | Ta={max_ta_solver:.1f} | V_LV={v_lv_ml:.1f}mL")

    # 3. Save checkpoint data (displacement only — all metrics computed offline)
    if save:
        fiber_stress.interpolate(fiber_stress_expr)
        fiber_strain.interpolate(fiber_strain_expr)
        vtx_u.write(t_abs)
        if vtx_p:
            vtx_p.write(t_abs)
        vtx_stress.write(t_abs)
        adios4dolfinx.write_function(checkpoint_file, u=problem.u, name="displacement", time=t_abs)

    # 4. Periodically flush Ta/pressure to disk so data survives SIGKILL.
    #    Without this, a SLURM timeout loses all Ta/pressure data (only in memory).
    steps_per_beat = int(round(RR_INTERVAL / dt))
    if comm.rank == 0 and i > 0 and i % steps_per_beat == 0:
        np.save(solver_dir / "Ta_solver_history.npy", np.array(Ta_solver_history))
        np.save(solver_dir / "solver_cavity_pressure_mmHg.npy", np.array(pressure_history))
        np.save(solver_dir / "Ta_history.npy", np.array(Ta_history))

# --- Save simulation_params.json EARLY (before solver loop) ---
# This ensures material/activation params survive even if SIGKILL hits during the loop.
if comm.rank == 0:
    sim_params = {
        "BPM": BPM,
        "HR_HZ": HR_HZ,
        "RR_INTERVAL": RR_INTERVAL,
        "dt": 0.001,  # hardcoded here because dt variable is defined later
        "mesh_unit": mesh_unit,
        "volume2ml": volume2ml,
        "incompressible": args.incompressible,
        "alpha_epi": args.alpha_epi,
        "alpha_base": args.alpha_base,
        "base_dirichlet": args.base_dirichlet,
        "one_sided_robin": args.one_sided_robin,
        "material_params": {k: {"value": float(v.value), "unit": str(v.original_unit)} for k, v in material_params.items()},
        "activation": {
            "TC": TC_ACTIVATION,
            "TR": TR_ACTIVATION,
            "tC": tC_ACTIVATION,
            "peak_kPa": 100.0,
        },
        "ratio_LV": ratio_LV,
        "ratio_RV": ratio_RV,
        "lvv_unloaded_m3": float(lvv_unloaded),
        "rvv_unloaded_m3": float(rvv_unloaded),
        "lvv_target_m3": float(lvv_target),
        "rvv_target_m3": float(rvv_target),
        "geo_scale": getattr(geo, '_geo_scale', 1.0),
        "geometry_dir": str(Path(args.geometry_dir).resolve()) if args.geometry_dir else None,
    }
    with open(outdir / "simulation_params.json", "w") as f:
        json.dump(sim_params, f, indent=2, default=custom_json)
    logger.info("Saved simulation_params.json (early, before solver loop)")

# --- Run Simulation ---

logger.info("Initializing coupled circulation model with consistent parameters...")
# Use consistency helper to get correct parameters for Atria and Ventricles
coupled_params = get_updated_parameters()

circulation_model = circulation.regazzoni2020.Regazzoni2020(
    parameters=coupled_params,
    add_units=False,
    callback=callback,
    p_BiV=p_BiV_func,
    verbose=True,
    comm=comm,
    outdir=outdir,
)

logger.info(f"Starting coupled simulation at {BPM} BPM (HR={HR_HZ} Hz, RR={RR_INTERVAL:.3f}s)...")
num_beats = args.beats  # Number of beats to simulate
dt = 0.001

# CI Mode: Only 2 timesteps for quick testing; Production: Full beat
if CI_MODE:
    end_time = 2 * dt  # ~0.002s for quick validation
    logger.info(f"⚠️  CI MODE: Running only {end_time}s ({int(end_time/dt)} timesteps)")
else:
    end_time = None  # Full beat
    logger.info(f"✓ PRODUCTION MODE: Running full beat ({RR_INTERVAL:.3f}s)")

try:
    coupled_history = circulation_model.solve(num_beats=num_beats, initial_state=circ_state, dt=dt, T=end_time)
    logger.info("Simulation complete.")
    # Merge old + new circulation history for restart continuations
    if RESTART_MODE and comm.rank == 0:
        old_hist = np.load(RESTART_DIR / "circulation" / "history.npy", allow_pickle=True).item()
        merged = {}
        for k in coupled_history:
            old_arr = np.array(old_hist[k])
            new_arr = np.array(coupled_history[k])
            if k == "time":
                new_arr = new_arr + restart_time_offset
            merged[k] = np.concatenate([old_arr, new_arr])
        coupled_history = merged
        logger.info(f"RESTART: Merged circulation history ({len(merged['time'])} total points)")
    if comm.rank == 0:
        np.save(outdir / "ode_state_history.npy", coupled_history, allow_pickle=True)
        # Backwards compat: also save as history.npy for any scripts that expect it
        np.save(outdir / "history.npy", coupled_history, allow_pickle=True)
        logger.info(f"Saved circulation ODE state history: {len(coupled_history['time'])} points")
finally:
    # --- Save Checkpoint Data (ALWAYS, even if simulation crashes) ---
    if comm.rank == 0:
        logger.info("Saving simulation checkpoint data for offline postprocessing...")
        try:
            # Save Ta history: [N_timesteps, 3] array with [LV, Septum, RV] activation (kPa)
            np.save(solver_dir / "Ta_history.npy", np.array(Ta_history))
            np.save(solver_dir / "Ta_solver_history.npy", np.array(Ta_solver_history))
            np.save(solver_dir / "solver_cavity_pressure_mmHg.npy", np.array(pressure_history))
            logger.info(f"  Ta history saved: {len(Ta_history)} timesteps")
            logger.info(f"  Pressure history saved: {len(pressure_history)} timesteps")

            # Save simulation parameters needed for offline reconstruction
            sim_params = {
                "BPM": BPM,
                "HR_HZ": HR_HZ,
                "RR_INTERVAL": RR_INTERVAL,
                "dt": dt,
                "mesh_unit": mesh_unit,
                "volume2ml": volume2ml,
                "incompressible": args.incompressible,
                "alpha_epi": args.alpha_epi,
                "alpha_base": args.alpha_base,
                "base_dirichlet": args.base_dirichlet,
                "one_sided_robin": args.one_sided_robin,
                "material_params": {k: {"value": float(v.value), "unit": str(v.original_unit)} for k, v in material_params.items()},
                "activation": {
                    "TC": TC_ACTIVATION,
                    "TR": TR_ACTIVATION,
                    "tC": tC_ACTIVATION,
                    "peak_kPa": 100.0,
                },
                "ratio_LV": ratio_LV,
                "ratio_RV": ratio_RV,
                "lvv_unloaded_m3": float(lvv_unloaded),
                "rvv_unloaded_m3": float(rvv_unloaded),
                "lvv_target_m3": float(lvv_target),
                "rvv_target_m3": float(rvv_target),
                "geo_scale": getattr(geo, '_geo_scale', 1.0),
                "geometry_dir": str(Path(args.geometry_dir).resolve()) if args.geometry_dir else None,
            }
            with open(outdir / "simulation_params.json", "w") as f:
                json.dump(sim_params, f, indent=2, default=custom_json)
            logger.info("  Simulation parameters saved")
            logger.info("  Run postprocess_metrics.py on this directory to compute all metrics")
        except Exception as e:
            logger.error(f"Failed to save checkpoint data: {e}")
