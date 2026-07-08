# # Complete Multiscale Simulation with Prestressing
#
# Uniform spatial active tension (single Constant) combined with
# pulse.FrankStarlingActiveStress so the active force ramps with local
# fiber stretch — the Frank-Starling mechanism. Robust MPI handling and
# logging suitable for compute clusters.

import json
import os
import csv
import time as time_module
import logging
import shutil
from pathlib import Path

# Scientific and FEniCSx imports
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
parser.add_argument('--load-unloaded-from', type=str, default=os.getenv("LOAD_UNLOADED_FROM") or None,
                    help='Path to an existing solver/prestress_inverse.bp to use as the unloaded '
                         'reference. When set, inverse unloading is skipped and u_pre is loaded '
                         'from this file.')
parser.add_argument('--restart-pre-circ', action='store_true',
                    default=os.getenv("RESTART_PRE_CIRC", "0") == "1",
                    help='In restart mode, run a standalone 0D warm-up with the target circulation '
                         'parameters before coupled FEM. This avoids shocking the old 0D state with '
                         'new PAH parameters at the first coupled timestep while keeping the restarted '
                         'unloaded/reference geometry.')
parser.add_argument('--restart-ramp-steps', type=int,
                    default=int(os.getenv("RESTART_RAMP_STEPS", 20)),
                    help='Number of mechanics substeps used to ramp from ED to the restart target state '
                         '(default: 20). Increase for fixed-reference acute loading tests.')
parser.add_argument('--pre-circ-beats', type=int, default=int(os.getenv("PRE_CIRC_BEATS", 10)),
                    help='Number of standalone 0D warm-up beats before unloading (default: 10).')
parser.add_argument('--pre-circ-max-beats', type=int, default=int(os.getenv("PRE_CIRC_MAX_BEATS", 10)),
                    help='Maximum standalone 0D warm-up beats if convergence checking is enabled.')
parser.add_argument('--pre-circ-convergence-tol', type=float, default=float(os.getenv("PRE_CIRC_CONVERGENCE_TOL", 0.0)),
                    help='If >0, rerun the standalone 0D warm-up until final-cycle LV/RV volume drift is below this fraction.')
parser.add_argument('--stop-after-unloading', action='store_true',
                    default=os.getenv("STOP_AFTER_UNLOADING", "0") == "1",
                    help='Stop after prestress/unloading diagnostics, before the coupled FEM cycle.')
parser.add_argument('--lv-material-scale', type=float, default=float(os.getenv("LV_MATERIAL_SCALE", 1.0)),
                    help='Scale passive HO a-like parameters in LDRB LV cells.')
parser.add_argument('--rv-material-scale', type=float, default=float(os.getenv("RV_MATERIAL_SCALE", 1.0)),
                    help='Scale passive HO a-like parameters in LDRB RV cells.')
parser.add_argument('--septum-material-scale', type=float, default=float(os.getenv("SEPTUM_MATERIAL_SCALE", 1.0)),
                    help='Scale passive HO a-like parameters in LDRB septum cells.')
parser.add_argument('--rv-edp-scale', type=float, default=float(os.getenv("RV_EDP_SCALE", 1.0)),
                    help='Scale the RV ED pressure target used by unloading.')
parser.add_argument('--rv-edp-max-mmhg', type=float, default=None if os.getenv("RV_EDP_MAX_MMHG") in (None, "") else float(os.getenv("RV_EDP_MAX_MMHG")),
                    help='Optional cap on RV ED pressure target in mmHg.')
parser.add_argument('--rv-edp-override-mmhg', type=float, default=None if os.getenv("RV_EDP_OVERRIDE_MMHG") in (None, "") else float(os.getenv("RV_EDP_OVERRIDE_MMHG")),
                    help='Optional absolute RV ED pressure target in mmHg.')
args = parser.parse_args()

# --- Restart Setup ---
RESTART_MODE = args.restart_from is not None
RESTART_DIR = Path(args.restart_from) if RESTART_MODE else None
LOAD_UNLOADED_FROM = Path(args.load_unloaded_from).expanduser().resolve() if args.load_unloaded_from else None
if RESTART_MODE and LOAD_UNLOADED_FROM is not None:
    raise ValueError("--restart-from and --load-unloaded-from cannot be combined")
if LOAD_UNLOADED_FROM is not None and not LOAD_UNLOADED_FROM.exists():
    raise FileNotFoundError(f"LOAD_UNLOADED_FROM does not exist: {LOAD_UNLOADED_FROM}")
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


def summarize_history(history, cycle_length, keys=("V_LV", "V_RV", "p_LV", "p_RV")):
    """Small JSON-safe summary of a circulation history."""
    diagnostics = {
        "n_points": int(len(history.get("time", []))),
        "cycle_length_s": float(cycle_length),
    }

    time = np.asarray(history.get("time", []), dtype=float)
    prev_idx = None
    if len(time) > 1:
        target_t = time[-1] - cycle_length
        prev_idx = int(np.argmin(np.abs(time - target_t)))
        diagnostics["final_time_s"] = float(time[-1])
        diagnostics["previous_cycle_index"] = prev_idx
        diagnostics["previous_cycle_time_s"] = float(time[prev_idx])

    max_volume_cycle_rel_change = 0.0
    for key in keys:
        if key not in history:
            continue
        arr = np.asarray(history[key], dtype=float)
        item = {
            "shape": list(arr.shape),
            "first": float(arr[0]),
            "last": float(arr[-1]),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
        if prev_idx is not None and 0 <= prev_idx < len(arr):
            delta = float(arr[-1] - arr[prev_idx])
            denom = max(abs(float(arr[prev_idx])), 1e-12)
            rel = delta / denom
            item["final_cycle_delta"] = delta
            item["final_cycle_rel_change"] = float(rel)
            if key in ("V_LV", "V_RV"):
                max_volume_cycle_rel_change = max(max_volume_cycle_rel_change, abs(rel))
        diagnostics[key] = item

    diagnostics["max_volume_cycle_rel_change"] = float(max_volume_cycle_rel_change)
    return diagnostics


def summarize_variable(var):
    """Serialize pulse.Variable values, including regional Function-valued parameters."""
    value = getattr(var, "value", var)
    unit = str(getattr(var, "original_unit", ""))
    if isinstance(value, dolfinx.fem.Function):
        arr = np.asarray(value.x.array, dtype=float)
        local_count = int(arr.size)
        return {
            "kind": "Function",
            "unit": unit,
            "local_count": local_count,
            "local_min": float(np.min(arr)) if local_count else None,
            "local_max": float(np.max(arr)) if local_count else None,
            "local_mean": float(np.mean(arr)) if local_count else None,
        }
    if isinstance(value, dolfinx.fem.Constant):
        arr = np.asarray(value.value, dtype=float)
        return {"kind": "Constant", "unit": unit, "value": arr.tolist()}
    return {"kind": "scalar", "unit": unit, "value": float(value)}

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

# --- Ensure the run directory owns a geometry handle for offline postprocessing ---
# postprocess_metrics.py and compute_per_cell.py expect <outdir>/geometry to
# exist. In restart mode the mechanics uses a scratch copy of the restart
# geometry, but offline postprocessing should point at the durable source.
if comm.rank == 0:
    _outdir_geo = outdir / "geometry"
    if RESTART_MODE and RESTART_DIR is not None:
        _geo_source = RESTART_DIR / "geometry"
    else:
        _geo_source = geodir
    if _geo_source is not None and not _outdir_geo.exists():
        try:
            _outdir_geo.symlink_to(_geo_source.resolve())
        except OSError:
            import shutil as _shutil_geo
            _shutil_geo.copytree(_geo_source.resolve(), _outdir_geo)
comm.barrier()

circulation.log.setup_logging(logging.INFO)
logger = logging.getLogger("pulse")

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

    # NOTE on atrial timing: the uniform shift above leaves atrial tC ≈ RR
    # (≡0 mod RR), so the atrial a-wave lands during early ventricular systole
    # rather than late diastole. We deliberately do NOT re-time it: an
    # explicitly late-diastolic kick adds ~16 mL of extra preload that over-
    # fills the LV past its imaged end-diastolic geometry (EDV 112→128 vs mesh
    # ~113). Since passive filling already reaches the imaged ED, the imaged ED
    # already embodies the in-vivo atrial contribution. The mis-timed a-wave is
    # therefore a cosmetic LA-pressure-trace artifact only; EDV/SV/EF and the
    # ventricular work (the proxy target) are unaffected. Documented caveat.

    # Update HR
    params["HR"] = circulation.units.ureg(f"{HR_HZ} Hz")

    return params

def run_0D(init_state, nbeats=None):
    logger.info("Running 0D circulation model to steady state...")
    # Use parameters consistent with BPM
    params = get_updated_parameters()
    requested_beats = int(nbeats if nbeats is not None else args.pre_circ_beats)
    max_beats = max(int(args.pre_circ_max_beats), requested_beats)
    tol = float(args.pre_circ_convergence_tol)
    attempted = []

    while True:
        logger.info(f"0D pre-run attempt: {requested_beats} beats")
        model = Regazzoni2020(parameters=params)
        history = model.solve(num_beats=requested_beats, initial_state=init_state)
        state = dict(zip(model.state_names(), model.state))
        diagnostics = summarize_history(history, RR_INTERVAL)
        diagnostics["requested_beats"] = int(requested_beats)
        diagnostics["convergence_tol"] = tol
        diagnostics["max_beats"] = int(max_beats)
        attempted.append(
            {
                "beats": int(requested_beats),
                "max_volume_cycle_rel_change": diagnostics["max_volume_cycle_rel_change"],
                "V_LV_final_cycle_delta": diagnostics.get("V_LV", {}).get("final_cycle_delta"),
                "V_RV_final_cycle_delta": diagnostics.get("V_RV", {}).get("final_cycle_delta"),
            }
        )

        if tol <= 0.0 or diagnostics["max_volume_cycle_rel_change"] <= tol or requested_beats >= max_beats:
            diagnostics["attempts"] = attempted
            diagnostics["converged"] = bool(tol <= 0.0 or diagnostics["max_volume_cycle_rel_change"] <= tol)
            return history, state, diagnostics

        next_beats = min(max_beats, max(requested_beats + 5, int(np.ceil(requested_beats * 1.5))))
        if next_beats == requested_beats:
            diagnostics["attempts"] = attempted
            diagnostics["converged"] = False
            return history, state, diagnostics
        requested_beats = next_beats

init_state_circ = {
   "V_LV": lvv_target * volume2ml * circulation.units.ureg("mL"),
    "V_RV": rvv_target * volume2ml * circulation.units.ureg("mL"),
}

if not RESTART_MODE:
    # --- FRESH RUN: run standalone 0D to get initial circulation state ---
    if comm.rank == 0:
        preload_circ_dir = outdir / "circulation"
        preload_circ_dir.mkdir(exist_ok=True)
        # Use converged ICs from JSON if available, otherwise model defaults
        ic_0d = _json_initial_state
        if ic_0d is not None:
            logger.info(f"Using initial_state from JSON for 0D pre-run")
        else:
            logger.info(f"No initial_state in JSON, using model defaults for 0D pre-run")
        history, circ_state, pre_circ_diagnostics = run_0D(init_state=ic_0d)
        if pre_circ_diagnostics["max_volume_cycle_rel_change"] > 0.01:
            logger.warning(
                "0D pre-run final-cycle volume drift is %.2f%% "
                "(V_LV d=%.3f mL, V_RV d=%.3f mL).",
                100.0 * pre_circ_diagnostics["max_volume_cycle_rel_change"],
                pre_circ_diagnostics.get("V_LV", {}).get("final_cycle_delta", 0.0),
                pre_circ_diagnostics.get("V_RV", {}).get("final_cycle_delta", 0.0),
            )
        np.save(outdir / "state.npy", circ_state, allow_pickle=True)
        np.save(outdir / "history.npy", history, allow_pickle=True)
        np.save(preload_circ_dir / "preload_state.npy", circ_state, allow_pickle=True)
        np.save(preload_circ_dir / "preload_history.npy", history, allow_pickle=True)
        with open(preload_circ_dir / "preload_diagnostics.json", "w") as f:
            json.dump(pre_circ_diagnostics, f, indent=2, default=custom_json)
    comm.Barrier()

    history = np.load(outdir / "history.npy", allow_pickle=True).item()
    circ_state = np.load(outdir / "state.npy", allow_pickle=True).item()
    if comm.rank == 0:
        with open(outdir / "circulation" / "preload_diagnostics.json") as f:
            pre_circ_diagnostics = json.load(f)
    else:
        pre_circ_diagnostics = None
    pre_circ_diagnostics = comm.bcast(pre_circ_diagnostics, root=0)
else:
    # --- RESTART: extract 0D state and ratios from old simulation ---
    logger.info("RESTART: Extracting 0D state from previous coupled simulation")
    history = None  # Will be replaced by coupled_history unless restart-pre-circ is enabled
    pre_circ_diagnostics = {
        "restart_mode": True,
        "restart_pre_circ": bool(args.restart_pre_circ),
    }

    # Extract final 0D state from old coupled history
    restart_history_tmp = np.load(RESTART_DIR / "circulation" / "history.npy", allow_pickle=True).item()
    circ_state_names = ["V_LA", "V_LV", "V_RA", "V_RV", "p_AR_SYS", "p_VEN_SYS",
                        "p_AR_PUL", "p_VEN_PUL", "Q_AR_SYS", "Q_VEN_SYS", "Q_AR_PUL", "Q_VEN_PUL"]
    circ_state = {k: float(restart_history_tmp[k][-1]) for k in circ_state_names if k in restart_history_tmp}
    logger.info(f"RESTART: Extracted 0D state from old history (V_LV={circ_state['V_LV']:.2f}, V_RV={circ_state['V_RV']:.2f})")

    if args.restart_pre_circ:
        if comm.rank == 0:
            preload_circ_dir = outdir / "circulation"
            preload_circ_dir.mkdir(exist_ok=True)
            ic_0d = _json_initial_state if _json_initial_state is not None else circ_state
            source = "target JSON initial_state" if _json_initial_state is not None else "previous coupled state"
            logger.info(f"RESTART: Running target-parameter 0D pre-run from {source}")
            history, circ_state, pre_circ_diagnostics = run_0D(init_state=ic_0d)
            pre_circ_diagnostics["restart_mode"] = True
            pre_circ_diagnostics["restart_pre_circ"] = True
            np.save(outdir / "state.npy", circ_state, allow_pickle=True)
            np.save(outdir / "history.npy", history, allow_pickle=True)
            np.save(preload_circ_dir / "restart_preload_state.npy", circ_state, allow_pickle=True)
            np.save(preload_circ_dir / "restart_preload_history.npy", history, allow_pickle=True)
            with open(preload_circ_dir / "restart_preload_diagnostics.json", "w") as f:
                json.dump(pre_circ_diagnostics, f, indent=2, default=custom_json)
        comm.Barrier()

        history = np.load(outdir / "history.npy", allow_pickle=True).item()
        circ_state = np.load(outdir / "state.npy", allow_pickle=True).item()
        if comm.rank == 0:
            with open(outdir / "circulation" / "restart_preload_diagnostics.json") as f:
                pre_circ_diagnostics = json.load(f)
        else:
            pre_circ_diagnostics = None
        pre_circ_diagnostics = comm.bcast(pre_circ_diagnostics, root=0)
        logger.info(
            f"RESTART: Target 0D pre-run state selected "
            f"(V_LV={circ_state['V_LV']:.2f}, V_RV={circ_state['V_RV']:.2f})"
        )

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
    # Fixed-ratio coupling (2026-06-22): the per-case ratio above (Mesh_ED / this case's
    # 0D warm-up ED) re-normalises every case back to the mesh ED, which clamps the FEM
    # preload and discards the across-case dilation. Setting FIXED_RATIO_{LV,RV} pins a
    # single reference ratio (case0 baseline) for ALL cases so the FEM feels the real
    # preload spread (the imaged mesh = baseline; higher afterload dilates from there).
    _fr_lv = os.environ.get("FIXED_RATIO_LV", "").strip()
    _fr_rv = os.environ.get("FIXED_RATIO_RV", "").strip()
    if _fr_lv and _fr_rv:
        ratio_LV = float(_fr_lv)
        ratio_RV = float(_fr_rv)
        if comm.rank == 0:
            logger.info(f"FIXED coupling ratios from env (overriding per-case): "
                        f"LV={ratio_LV:.5f}, RV={ratio_RV:.5f}")

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

# Peak active tension (kPa) at the activation curve's plateau. The default
# 100 kPa was tuned for the no-Frank-Starling sweep; with the static F-L
# multiplier active the cycle-average effective Ta is ~half, so set
# TA_PEAK_KPA=200 to recover the original effective amplitude.
TA_PEAK_KPA = float(os.environ.get("TA_PEAK_KPA", 100.0))

# Frank-Starling mode for the forward problem:
#   FS_PRELOAD_ONLY=1  →  freeze g(λ) at end-diastole; multiplier held fixed
#                         through the rest of the beat (preload-dependent F-S,
#                         the operational definition at the whole-heart level)
#   FS_PRELOAD_ONLY=0  →  live g(λ) recomputed on every Newton step
#                         (instantaneous static length-tension)
FS_PRELOAD_ONLY = os.environ.get("FS_PRELOAD_ONLY", "0") == "1"

# Master switch for the active-contraction model in the FORWARD solve:
#   USE_FRANK_STARLING=1 (default) -> pulse.FrankStarlingActiveStress, live g(λ)
#                                     (set TA_PEAK_KPA=200 to compensate the ~0.5
#                                      cycle-average multiplier)
#   USE_FRANK_STARLING=0           -> plain pulse.ActiveStress, constant Ta
#                                     (the thesis / no-Frank-Starling bundle)
# Inverse unloading is always passive (Ta=0), so this never affects the prestress.
USE_FRANK_STARLING = os.environ.get("USE_FRANK_STARLING", "1") == "1"
if not USE_FRANK_STARLING:
    # preload-only freezing only makes sense for the F-S multiplier
    FS_PRELOAD_ONLY = False

# Activation-lag (relaxation) Frank-Starling: if FS_RELAX_TAU_MS>0, g(λ) is a
# stored field that relaxes toward the instantaneous target with time constant
# tau, advanced once per timestep. Keeps force from collapsing as fibers
# shorten (the "momentum" the instantaneous mode lacks). Takes precedence over
# FS_PRELOAD_ONLY. tau->0 ≈ instantaneous, tau->inf ≈ preload-frozen.
FS_RELAX_TAU_S = float(os.environ.get("FS_RELAX_TAU_MS", "0")) / 1000.0
if not USE_FRANK_STARLING:
    FS_RELAX_TAU_S = 0.0

def _fs_mode_kwargs():
    """Active-model kwargs selecting the Frank-Starling mode (relaxation >
    preload-only > instantaneous)."""
    if FS_RELAX_TAU_S > 0:
        return {"relaxation_tau": FS_RELAX_TAU_S}
    if FS_PRELOAD_ONLY:
        return {"preload_only": True}
    return {}

# Frank-Starling g(lambda) curve shape. Defaults reproduce the original curve
# (g=0.5 at lambda=1.0), which throttles force to ~50-70% at the operating
# stretch and forces unphysiologically high Ta. Re-centre via env (e.g.
# FS_STRETCH_OPTIMAL≈1.05) so peak force sits near the end-diastolic stretch →
# physiological force at literature Ta, while preserving preload-dependence at
# shorter (under-filled) lengths.
FS_AMP_MIN = float(os.environ.get("FS_AMP_MIN", "0.0"))
FS_AMP_MAX = float(os.environ.get("FS_AMP_MAX", "1.0"))
FS_STRETCH_THRESHOLD = float(os.environ.get("FS_STRETCH_THRESHOLD", "0.85"))
FS_STRETCH_OPTIMAL = float(os.environ.get("FS_STRETCH_OPTIMAL", "1.15"))

def _fs_curve_kwargs():
    """g(lambda) shape kwargs (only meaningful for Frank-Starling modes)."""
    return {"amp_min": FS_AMP_MIN, "amp_max": FS_AMP_MAX,
            "stretch_threshold": FS_STRETCH_THRESHOLD, "stretch_optimal": FS_STRETCH_OPTIMAL}


def get_activation(t):
    """
    Returns the uniform scalar active tension in kPa at time t.

    Spatial uniformity is intentional: any per-region modulation comes from
    the Frank-Starling stretch factor in pulse.FrankStarlingActiveStress,
    not from a region-dependent Ta. Peak amplitude is set by the
    TA_PEAK_KPA environment variable (default 100).
    """
    value = circulation.time_varying_elastance.blanco_ventricle(
        EA=1.0,
        EB=0.0,
        tC=tc_shifted,          # 0.0
        TC=TC_ACTIVATION,
        TR=TR_ACTIVATION,
        RR=RR_INTERVAL,
    )(t)
    return TA_PEAK_KPA * value

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


# --- Setup Problem (uniform scalar Ta + Frank-Starling) ---

def setup_problem(
    geometry,
    f0,
    s0,
    material_params,
    alpha_epi_val=1e5,
    alpha_base_val=1e6,
    incompressible=False,
    base_dirichlet="x",
    use_frank_starling=True,
):
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)

    # Single uniform scalar activation. Per-region modulation comes from the
    # Frank-Starling stretch factor below, not from a region-dependent Ta.
    Ta = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)),
        "kPa",
    )
    if use_frank_starling:
        # Frank-Starling: multiplies Ta by a piecewise-linear g(lambda) of
        # fiber stretch. .register(u) must be called once problem.u exists.
        # In preload-only mode the multiplier is a Function set once by
        # freeze_at(u_ED) and held constant through the rest of the beat.
        active_model = pulse.FrankStarlingActiveStress(
            f0=f0, activation=Ta,
            **_fs_mode_kwargs(), **_fs_curve_kwargs(),
        )
    else:
        # Plain active stress for inverse unloading: Ta is identically 0 there
        # so the active term contributes nothing regardless of FS multiplier,
        # but FS would raise ValueError without a registered displacement.
        active_model = pulse.ActiveStress(f0=f0, activation=Ta)

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

    return model, robin, dirichlet_bcs, Ta, active_model


def apply_region_material_scales(material_params, markers_mt, lv_scale=1.0, rv_scale=1.0, septum_scale=1.0):
    """Return HO params with selected a-like parameters represented as DG0 fields."""
    scales = {
        1: float(lv_scale),       # LDRB LV
        2: float(rv_scale),       # LDRB RV
        3: float(septum_scale),   # LDRB septum
    }
    if all(np.isclose(scale, 1.0) for scale in scales.values()):
        return material_params

    V0 = dolfinx.fem.functionspace(geometry.mesh, ("DG", 0))
    tag_values = np.asarray(markers_mt.values, dtype=np.int32)
    scaled_params = dict(material_params)

    for name in ("a", "a_f", "a_s", "a_fs"):
        base_var = material_params[name]
        base_value = float(base_var.value)
        field = dolfinx.fem.Function(V0, name=f"material_{name}_regional")
        n = min(len(field.x.array), len(tag_values))
        field.x.array[:n] = base_value
        values = field.x.array[:n]
        for tag, scale in scales.items():
            values[tag_values[:n] == tag] = base_value * scale
        field.x.scatter_forward()
        scaled_params[name] = pulse.Variable(field, base_var.original_unit)

    return scaled_params


material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
material_region_scales = {
    "LV": float(args.lv_material_scale),
    "RV": float(args.rv_material_scale),
    "Septum": float(args.septum_material_scale),
}
material_params = apply_region_material_scales(
    material_params,
    geo.additional_data["markers_mt"],
    lv_scale=material_region_scales["LV"],
    rv_scale=material_region_scales["RV"],
    septum_scale=material_region_scales["Septum"],
)
if comm.rank == 0 and any(not np.isclose(v, 1.0) for v in material_region_scales.values()):
    logger.info(f"Regional material scaling enabled: {material_region_scales}")
# Use Compressible for Prestressing always (Hybrid Strategy). Skip Frank-
# Starling here: PrestressProblem doesn't expose a forward displacement to
# register, and Ta is zero throughout unloading so the active term is zero
# regardless.
model, robin, dirichlet_bcs, Ta, _active_unused = setup_problem(
    geometry=geometry, f0=geo.f0, s0=geo.s0, material_params=material_params,
    alpha_epi_val=args.alpha_epi, alpha_base_val=args.alpha_base,
    incompressible=False, base_dirichlet=args.base_dirichlet,
    use_frank_starling=False,
)

# --- Prestressing (Inverse Elasticity) ---

if RESTART_MODE:
    # Use dummy pressures — prestress is cached and won't be recomputed
    p_LV_ED = 0.0
    p_RV_ED = 0.0
else:
    p_LV_ED = mmHg_to_kPa(history["p_LV"][-1])
    p_RV_ED = mmHg_to_kPa(history["p_RV"][-1])

p_LV_ED_raw = float(p_LV_ED)
p_RV_ED_raw = float(p_RV_ED)
rv_edp_adjustments = {
    "scale": float(args.rv_edp_scale),
    "max_mmhg": args.rv_edp_max_mmhg,
    "override_mmhg": args.rv_edp_override_mmhg,
}
if not RESTART_MODE:
    p_RV_ED *= float(args.rv_edp_scale)
    if args.rv_edp_max_mmhg is not None:
        p_RV_ED = min(p_RV_ED, mmHg_to_kPa(args.rv_edp_max_mmhg))
    if args.rv_edp_override_mmhg is not None:
        p_RV_ED = mmHg_to_kPa(args.rv_edp_override_mmhg)

logger.info(
    f"Target ED Pressures: p_LV={p_LV_ED:.2f} kPa, p_RV={p_RV_ED:.2f} kPa "
    f"(raw p_LV={p_LV_ED_raw:.2f} kPa, raw p_RV={p_RV_ED_raw:.2f} kPa)"
)

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
if LOAD_UNLOADED_FROM is not None and comm.rank == 0:
    logger.info(f"Using pre-computed unloaded reference from {LOAD_UNLOADED_FROM}")
    _src = LOAD_UNLOADED_FROM
    _dst = solver_dir / "prestress_inverse.bp"
    if _dst.exists():
        if _dst.is_dir():
            shutil.rmtree(_dst)
        else:
            _dst.unlink()
    if _src.is_dir():
        shutil.copytree(_src, _dst)
    else:
        shutil.copy2(_src, _dst)

    _back_src = _src.parent / "prestress_backward.bp"
    _back_dst = solver_dir / "prestress_backward.bp"
    if _back_src.exists():
        if _back_dst.exists():
            if _back_dst.is_dir():
                shutil.rmtree(_back_dst)
            else:
                _back_dst.unlink()
        if _back_src.is_dir():
            shutil.copytree(_back_src, _back_dst)
        else:
            shutil.copy2(_back_src, _back_dst)
        logger.info(f"Copied prestress_backward.bp from {_back_src}")
elif RESTART_MODE and comm.rank == 0:
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
if RESTART_MODE or LOAD_UNLOADED_FROM is not None:
    comm.barrier()

prestress_fname = solver_dir / "prestress_inverse.bp"
if LOAD_UNLOADED_FROM is not None and not prestress_fname.exists():
    raise FileNotFoundError(f"Failed to stage shared unloaded reference at {prestress_fname}")
if LOAD_UNLOADED_FROM is None and not prestress_fname.exists():
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

unloading_diagnostics = {
    "p_LV_ED_kPa": float(p_LV_ED),
    "p_RV_ED_kPa": float(p_RV_ED),
    "p_LV_ED_raw_kPa": float(p_LV_ED_raw),
    "p_RV_ED_raw_kPa": float(p_RV_ED_raw),
    "p_LV_ED_mmhg": float(p_LV_ED / 0.133322) if p_LV_ED else 0.0,
    "p_RV_ED_mmhg": float(p_RV_ED / 0.133322) if p_RV_ED else 0.0,
    "p_LV_ED_raw_mmhg": float(p_LV_ED_raw / 0.133322) if p_LV_ED_raw else 0.0,
    "p_RV_ED_raw_mmhg": float(p_RV_ED_raw / 0.133322) if p_RV_ED_raw else 0.0,
    "rv_edp_adjustments": rv_edp_adjustments,
    "lvv_target_mL": float(lvv_target * volume2ml),
    "rvv_target_mL": float(rvv_target * volume2ml),
    "lvv_unloaded_mL": float(lvv_unloaded * volume2ml),
    "rvv_unloaded_mL": float(rvv_unloaded * volume2ml),
    "lv_unloaded_fraction_of_ED": float(lvv_unloaded / lvv_target) if lvv_target else None,
    "rv_unloaded_fraction_of_ED": float(rvv_unloaded / rvv_target) if rvv_target else None,
    "lv_shrink_percent": float(100.0 * (1.0 - lvv_unloaded / lvv_target)) if lvv_target else None,
    "rv_shrink_percent": float(100.0 * (1.0 - rvv_unloaded / rvv_target)) if rvv_target else None,
    "ratio_LV": float(ratio_LV),
    "ratio_RV": float(ratio_RV),
    "material_region_scales": material_region_scales,
    "load_unloaded_from": str(LOAD_UNLOADED_FROM) if LOAD_UNLOADED_FROM is not None else None,
    "used_precomputed_unloaded_reference": LOAD_UNLOADED_FROM is not None,
}


def build_simulation_params(stage, dt_value=0.001):
    return {
        "stage": stage,
        "BPM": BPM,
        "HR_HZ": HR_HZ,
        "RR_INTERVAL": RR_INTERVAL,
        "dt": dt_value,
        "mesh_unit": mesh_unit,
        "volume2ml": volume2ml,
        "incompressible": args.incompressible,
        "alpha_epi": args.alpha_epi,
        "alpha_base": args.alpha_base,
        "base_dirichlet": args.base_dirichlet,
        "one_sided_robin": args.one_sided_robin,
        "stop_after_unloading": args.stop_after_unloading,
        "restart_pre_circ": bool(args.restart_pre_circ),
        "restart_ramp_steps": int(args.restart_ramp_steps),
        "pre_circ": pre_circ_diagnostics,
        "unloading": unloading_diagnostics,
        "load_unloaded_from": str(LOAD_UNLOADED_FROM) if LOAD_UNLOADED_FROM is not None else None,
        "used_precomputed_unloaded_reference": LOAD_UNLOADED_FROM is not None,
        "material_region_scales": material_region_scales,
        "rv_edp_adjustments": rv_edp_adjustments,
        "material_params": {k: summarize_variable(v) for k, v in material_params.items()},
        "activation": {
            "TC": TC_ACTIVATION,
            "TR": TR_ACTIVATION,
            "tC": tC_ACTIVATION,
            "peak_kPa": TA_PEAK_KPA,
        },
        "frank_starling": {
            "enabled": USE_FRANK_STARLING,
            "mode": (
                "off_constant_Ta" if not USE_FRANK_STARLING
                else "relaxation" if FS_RELAX_TAU_S > 0
                else "preload_only" if FS_PRELOAD_ONLY
                else "instantaneous"
            ),
            "relaxation_tau_s": FS_RELAX_TAU_S,
            "amp_min": FS_AMP_MIN, "amp_max": FS_AMP_MAX,
            "stretch_threshold": FS_STRETCH_THRESHOLD, "stretch_optimal": FS_STRETCH_OPTIMAL,
            "amp_min": 0.0,
            "amp_max": 1.0,
            "stretch_threshold": 0.85,
            "stretch_optimal": 1.15,
        },
        "ratio_LV": ratio_LV,
        "ratio_RV": ratio_RV,
        "lvv_unloaded_m3": float(lvv_unloaded),
        "rvv_unloaded_m3": float(rvv_unloaded),
        "lvv_target_m3": float(lvv_target),
        "rvv_target_m3": float(rvv_target),
        "p_LV_ED_kPa": float(p_LV_ED),
        "p_RV_ED_kPa": float(p_RV_ED),
        "p_LV_ED_raw_kPa": float(p_LV_ED_raw),
        "p_RV_ED_raw_kPa": float(p_RV_ED_raw),
        "geo_scale": getattr(geo, '_geo_scale', 1.0),
        "geometry_dir": str((outdir / "geometry").resolve()) if (outdir / "geometry").exists() else (
            str(Path(args.geometry_dir).resolve()) if args.geometry_dir else None
        ),
    }


def write_simulation_params(stage, dt_value=0.001):
    if comm.rank == 0:
        sim_params = build_simulation_params(stage=stage, dt_value=dt_value)
        with open(outdir / "simulation_params.json", "w") as f:
            json.dump(sim_params, f, indent=2, default=custom_json)
        circ_dir = outdir / "circulation"
        circ_dir.mkdir(exist_ok=True)
        with open(circ_dir / "unloading_diagnostics.json", "w") as f:
            json.dump(unloading_diagnostics, f, indent=2, default=custom_json)
        logger.info(f"Saved simulation_params.json ({stage})")


write_simulation_params(stage="post_unloading")

if args.stop_after_unloading:
    logger.info("STOP_AFTER_UNLOADING requested; exiting before forward coupled cycle.")
    sys.exit(0)

if args.incompressible:
    logger.info("Warning: Hybrid Prestressing Strategy Active (Compressible Unloading -> Incompressible Forward)")

model, robin, dirichlet_bcs, Ta, active_model = setup_problem(
    geometry=geometry, f0=f0_quad, s0=s0_quad, material_params=material_params,
    alpha_epi_val=args.alpha_epi, alpha_base_val=args.alpha_base,
    incompressible=args.incompressible, base_dirichlet=args.base_dirichlet,
    use_frank_starling=USE_FRANK_STARLING,
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

# Hand FrankStarlingActiveStress the displacement so g(lambda) can evaluate.
# Plain ActiveStress (no-FS) has no register() and needs no displacement.
if USE_FRANK_STARLING:
    active_model.register(u_disp)

# Setup Stress/Strain Post-processing - kinematics
# FIXED: Use full CardiacModel (material + compressibility) instead of material only
# This ensures stresses include pressure contribution for proper boundary work calculation

W = dolfinx.fem.functionspace(geometry.mesh, ("DG", 1))
I = ufl.Identity(3)
F = ufl.variable(ufl.grad(u_disp) + I)
C = ufl.variable(F.T * F)
E = 0.5 * (C - I)
f_map = (F * f0_map) / ufl.sqrt(ufl.inner(F * f0_map, F * f0_map))

# For live visualization only, use DG1 fiber fields so the expression can be
# sampled into the DG1 output space. Offline work/stress metrics still use the
# quadrature-fiber CardiacModel reconstructed from the checkpoint.
material_viz = pulse.HolzapfelOgden(f0=f0_map, s0=s0_map, **material_params)
if USE_FRANK_STARLING:
    active_viz = pulse.FrankStarlingActiveStress(
        f0=f0_map, activation=Ta,
        **_fs_mode_kwargs(), **_fs_curve_kwargs(),
    )
    active_viz.register(u_disp)
else:
    active_viz = pulse.ActiveStress(f0_map, activation=Ta)
if args.incompressible:
    comp_viz = pulse.compressibility.Incompressible()
    comp_viz.register(problem.p)
else:
    comp_viz = pulse.compressibility.Compressible2()
model_viz = pulse.CardiacModel(
    material=material_viz,
    active=active_viz,
    compressibility=comp_viz,
)
S_viz = model_viz.S(C)
T_viz = (1.0 / ufl.det(F)) * F * S_viz * F.T

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
vtx_stress = dolfinx.io.VTXWriter(geometry.mesh.comm, viz_dir / "stress_strain.bp", [fiber_stress, fiber_strain], engine="BP4")

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

    # In preload-only Frank-Starling mode, freeze g(λ) at the end-diastolic
    # configuration we just inflated to. Both the forward solver's active
    # model and the visualization active model share Ta but are separate
    # FS instances and must each be frozen.
    if FS_PRELOAD_ONLY:
        active_model.freeze_at(u_disp)
        active_viz.freeze_at(u_disp)
        if comm.rank == 0:
            logger.info("Frank-Starling multiplier frozen at end-diastole (preload-only mode)")

    # Store old values (scalar Ta is just a float).
    problem.old_Ta = float(Ta.value.value)
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

    # Restart path also needs freeze_at(u_ED) before any active-stress ramp.
    if FS_PRELOAD_ONLY:
        active_model.freeze_at(u_disp)
        active_viz.freeze_at(u_disp)
        if comm.rank == 0:
            logger.info("RESTART: Frank-Starling multiplier frozen at end-diastole (preload-only mode)")

    problem.old_Ta = float(Ta.value.value)
    problem.old_lv_volume = lv_volume.value.copy()
    problem.old_rv_volume = rv_volume.value.copy()

    # Now ramp from ED to the restart state (volumes + activation)
    restart_lv_target = (last_V_LV_0D * ratio_LV) / volume2ml
    restart_rv_target = (last_V_RV_0D * ratio_RV) / volume2ml
    # Old solver Ta history may be either [N,3] (legacy region-aware) or [N]
    # (uniform); collapse to a scalar either way.
    restart_Ta_target = float(np.mean(np.atleast_1d(old_Ta_solver_history[-1])))
    ed_lv = lv_volume.value.copy()
    ed_rv = rv_volume.value.copy()
    ed_Ta = float(Ta.value.value)

    # Only ramp if the restart state differs from ED
    vol_diff = abs(restart_lv_target - ed_lv) * volume2ml
    ta_diff = abs(restart_Ta_target - ed_Ta)
    if vol_diff > 0.01 or ta_diff > 0.01:  # > 0.01 mL or 0.01 kPa
        n_ramp = int(args.restart_ramp_steps)
        logger.info(f"RESTART: Ramping from ED to restart state ({n_ramp} steps, "
                     f"dV_LV={vol_diff:.2f}mL, dTa={ta_diff:.1f}kPa)...")
        for ri in range(n_ramp):
            frac = (ri + 1) / n_ramp
            lv_volume.value = ed_lv + frac * (restart_lv_target - ed_lv)
            rv_volume.value = ed_rv + frac * (restart_rv_target - ed_rv)
            Ta.assign(ed_Ta + frac * (restart_Ta_target - ed_Ta))
            problem.solve()
        problem.old_lv_volume = lv_volume.value.copy()
        problem.old_rv_volume = rv_volume.value.copy()
        problem.old_Ta = float(Ta.value.value)
        logger.info(f"RESTART: Ramp complete. V_LV={lv_volume.value*volume2ml:.2f}mL, "
                     f"Ta={float(Ta.value.value):.1f}")
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
    if abs(dLV) > tol or abs(dRV) > tol or abs(dTa) > tol:

        # Iteration tracking: these advance as sub-steps succeed,
        # giving us a "checkpoint" to reset to on failure
        old_lv_it = old_lv_volume.copy()
        old_rv_it = old_rv_volume.copy()
        old_Ta_it = float(old_Ta)

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
                    old_Ta_it = float(Ta.value.value)

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
                f"Ta remaining: {abs(new_value_Ta - old_Ta_it):.4f}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

    # --- Save state for next coupling call ---
    problem.old_Ta = float(Ta.value.value)
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

    # 1. Record activation for postprocessing (scalar Ta).
    Ta_history.append(float(get_activation(t)))

    # Also record the actual solver Ta (may differ from get_activation by the
    # substep convergence residual).
    solver_ta_value = float(Ta.value.value)
    Ta_solver_history.append(solver_ta_value)

    # Record solver cavity pressures (the Lagrange multiplier — exact surface traction).
    # These are the pressures actually applied to the FEM boundary, NOT the 0D ODE
    # output (which is 1 timestep ahead due to staggered coupling).
    # Saved as solver_cavity_pressure_mmHg.npy (renamed from pressure_history.npy).
    lv_p_Pa = float(problem.cavity_pressures[0].x.array[0])
    rv_p_Pa = float(problem.cavity_pressures[1].x.array[0])
    pressure_history.append([lv_p_Pa * 1e-3 / 0.133322, rv_p_Pa * 1e-3 / 0.133322])  # Pa -> mmHg

    # 1b. Advance the relaxation (activation-lag) Frank-Starling multiplier by
    # one timestep using the just-converged displacement (explicit update).
    # Once per accepted step; no-op unless relaxation mode is active.
    if FS_RELAX_TAU_S > 0:
        active_model.advance(dt)
        active_viz.advance(dt)

    # 2. Console Feedback (lightweight — no metrics computation)
    if comm.rank == 0 and (i % 10 == 0 or CI_MODE):
        lv_p_kPa = problem.cavity_pressures[0].x.array[0] * 1e-3
        v_lv_ml = float(lv_volume.value * volume2ml)
        print(f"STEP {i:04d} | t={t_abs:.3f} | Ta={solver_ta_value:.1f} | V_LV={v_lv_ml:.1f}mL")

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
write_simulation_params(stage="pre_solver", dt_value=0.001)

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
            # Save Ta history: [N_timesteps] scalar array (kPa).
            # Spatial uniformity is intentional — see get_activation() docstring.
            np.save(solver_dir / "Ta_history.npy", np.array(Ta_history))
            np.save(solver_dir / "Ta_solver_history.npy", np.array(Ta_solver_history))
            np.save(solver_dir / "solver_cavity_pressure_mmHg.npy", np.array(pressure_history))
            logger.info(f"  Ta history saved: {len(Ta_history)} timesteps")
            logger.info(f"  Pressure history saved: {len(pressure_history)} timesteps")

            # Save simulation parameters needed for offline reconstruction
            write_simulation_params(stage="final", dt_value=dt)
            logger.info("  Simulation parameters saved")
            logger.info("  Run postprocess_metrics.py on this directory to compute all metrics")
        except Exception as e:
            logger.error(f"Failed to save checkpoint data: {e}")
