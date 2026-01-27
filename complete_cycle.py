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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Complete cardiac cycle simulation')
parser.add_argument('bpm', type=int, nargs='?', default=None, help='Heart rate in BPM (default: 75)')
parser.add_argument('--ci', action='store_true', help='Enable CI mode (2 timesteps only for quick testing)')
parser.add_argument('--mesh', type=str, default=None, help='Path to custom XDMF mesh (optional)')
parser.add_argument('--char_length', type=float, default=5.0, help='Mesh characteristic length (default: 5.0)')
parser.add_argument('--metrics_space', type=str, default="DG0", choices=["DG0", "DG1"], help='Function space for metrics (DG0 or DG1)')
parser.add_argument('--circulation_params', type=str, default=None, help='Path to JSON file with circulation parameters')
parser.add_argument('--alpha_epi', type=float, default=1e5, help='Epicardial spring stiffness (Pa/m) (default: 1e5)')
parser.add_argument('--alpha_base', type=float, default=1e6, help='Basal spring stiffness (Pa/m) (default: 1e6)')
args = parser.parse_args()

# Determine BPM
if args.bpm is not None:
    BPM = args.bpm
else:
    BPM = int(os.getenv("BPM", 75))

# CI Mode flag (OFF by default, must be explicitly enabled)
CI_MODE = args.ci or bool(os.getenv("CI"))

if CI_MODE:
    print("⚠️  CI MODE ENABLED - Short circuit test (2 timesteps only)")
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
comm.barrier()
geodir = outdir / "geometry"

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

if comm.rank == 0 and not (geodir / "geometry.bp").exists():
    logger.info("Generating and processing geometry (Rank 0)...")
    mode = -1
    std = 0
    # Configurable fidelity mesh (default 5.0)
    char_length = args.char_length

    if args.mesh:
        # --- CUSTOM MESH PATH ---
        logger.info(f"Loading CUSTOM MESH from: {args.mesh}")
        
        # 1. Read Mesh
        with dolfinx.io.XDMFFile(MPI.COMM_SELF, args.mesh, "r") as xdmf:
             mesh_in = xdmf.read_mesh(name="mesh")
             
             # Create connectivity to read facet tags (Requires IndexMap for facets)
             mesh_in.topology.create_connectivity(mesh_in.topology.dim - 1, mesh_in.topology.dim)

             # Custom meshes typically have facet_tags
             try:
                 ft_in = xdmf.read_meshtags(mesh_in, name="facet_tags")
             except:
                 logger.warning("Could not read 'facet_tags', trying 'mesh_tags'...")
                 ft_in = xdmf.read_meshtags(mesh_in, name="mesh_tags")
        
        # 2. Define Markers (Based on Paraview Inspection)
        # 40: Epicardium, 30: LV Endo, 20: RV Endo, 10: Base
        markers = {
            "BASE": (10, 2),
            "ENDO_LV": (30, 2),
            "ENDO_RV": (20, 2),
            "EPI": (40, 2)
        }
        
        # 3. Create Geometry Object
        # Note: We must create the object structure expected by the rest of the code
        # We assume standard LDRB fiber creation is needed since XDMF lacked fibers.
        
        # Create minimal Geometry object wrapper or use cardiac_geometries class
        # Ideally we use cardiac_geometries if possible to get helper methods
        geo = cardiac_geometries.geometry.Geometry(
            mesh=mesh_in,
            markers=markers,
            ffun=ft_in,
            f0=None, s0=None, n0=None # Fibers generated below
        )
        
        # NOTE: Custom meshes might NOT require rotation if they are already aligned.
        # Assuming NO ROTATION for now unless proven otherwise.
        
    else:
        # --- DEFAULT UKB GENERATION ---
        # Generate base mesh
        geo = cardiac_geometries.mesh.ukb(
            outdir=geodir,
            comm=MPI.COMM_SELF, # Generation happens on rank 0 only initially
            mode=mode,
            std=std,
            case="ED",
            char_length_max=char_length,
            char_length_min=char_length,
            clipped=True,
        )

        # Rotate Mesh (Base Normal -> X-axis)
        geo = geo.rotate(target_normal=[1.0, 0.0, 0.0], base_marker="BASE")

    fiber_angles = dict(
        alpha_endo_lv=60,
        alpha_epi_lv=-60,
        alpha_endo_rv=90,
        alpha_epi_rv=-25,
        beta_endo_lv=-20,
        beta_epi_lv=20,
        beta_endo_rv=0,
        beta_epi_rv=20,
    )

    # Generate Fibers (LDRB)
    # Note: LDRB needs transforming markers to its expected keys
    ldrb_markers = cardiac_geometries.mesh.transform_markers(geo.markers, clipped=True)
    
    # If using custom mesh, clipped=True might not be relevant but transform_markers filters by standard keys which is good.
    # However, for our custom map, we constructed it with standard keys "ENDO_LV" etc, so it should pass.
    
    system = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=ldrb_markers,
        **fiber_angles,
        fiber_space="Quadrature_6",
    )
    #could change fiber_space to DG_1 or CG_1?
    # Generate DG0 system for Active Tension Markers (From V1)
    # This creates markers scalar which helps define LV/RV/Septum regions
    system_dg0 = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=ldrb_markers,
        **fiber_angles,
        fiber_space="DG_0",
    )

    # --- MPI FIX for V1 ---
    # V1 used comm.allgather here which crashes on clusters with distributed meshes.
    # Since we are currently in a "Rank 0 generation" block (MPI.COMM_SELF),
    # we can just extract the array directly.
    # (Note: If generation were distributed, we would need a purely local extraction).

    markers_scalar = system_dg0.markers_scalar

    # Get the index map to find total cells (Owned + Ghosts) on this process
    imap = geo.mesh.topology.index_map(3)
    num_cells_local = imap.size_local
    num_ghosts = imap.num_ghosts
    total_cells = num_cells_local + num_ghosts

    # Create entities for ALL cells on this processor (including ghosts)
    entities = np.arange(total_cells, dtype=np.int32)

    # Get values for ALL cells (do not slice off the end)
    values = markers_scalar.x.array[:total_cells].astype(np.int32)

    markers_mt = dolfinx.mesh.meshtags(geo.mesh, 3, entities, values)
    # ----------------------

    # Additional Vectors for Analysis in DG 1 Space
    fiber_space = "DG_1"
    system_fibers = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=cardiac_geometries.mesh.transform_markers(geo.markers, clipped=True),
        **fiber_angles,
        fiber_space=fiber_space,
    )

    # Save Everything
    additional_data = {
        "f0_DG_1": system_fibers.f0,
        "s0_DG_1": system_fibers.s0,
        "n0_DG_1": system_fibers.n0,
        "markers_mt": markers_mt, # Saved for V1 active tension logic
    }

    if (geodir / "geometry.bp").exists():
        shutil.rmtree(geodir / "geometry.bp")

    cardiac_geometries.geometry.save_geometry(
        path=geodir / "geometry.bp",
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=geo.markers,
        info=geo.info,
        f0=system.f0,
        s0=system.s0,
        n0=system.n0,
        additional_data=additional_data,
    )

comm.barrier()

# Load the generated geometry
logger.info("Loading geometry...")
geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)

if args.mesh:
    # Custom mesh assumption: Units in cm (typical for clinical scans)
    scale = 1e-2
else:
    # UKB mesh assumption: Units in mm
    scale = 1e-3

geo.mesh.geometry.x[:] *= scale

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

# Store Target Volumes (ED)
volume2ml = 1e6
mesh_unit = "m"

# Helper to assist with parser_ds above which isn't defined
def parser_ds(ds_measure, marker_id):
    return ds_measure(marker_id)

lvv_target = 0.0
rvv_target = 0.0

try:
    lvv_target = comm.allreduce(geometry.volume("LV"), op=MPI.SUM)
except Exception:
    # Fallback for meshes without volume tags (Surface Integral)
    # V = 1/3 * int(x . n) ds
    logger.info("Volume tag 'LV' not found. Calculating cavity volume from surface 'ENDO_LV'...")
    x = ufl.SpatialCoordinate(geometry.mesh)
    n = ufl.FacetNormal(geometry.mesh)
    # Note: Normal points OUT of the domain (into the cavity).
    # So V_cavity = -1/3 * int(x.n) ds(endo)? 
    # Or is it positive? Divergence theorem on the HOLE?
    # Usually V = -1/3 int(x.n) ds_endo if n points OUT of wall (INTO cavity).
    # Let's verify sign convention.
    try:
        ds_lv = parser_ds(geometry.ds, geometry.markers["ENDO_LV"][0])
        val = dolfinx.fem.assemble_scalar(dolfinx.fem.form(-1.0/3.0 * ufl.dot(x, n) * ds_lv))
        lvv_target = comm.allreduce(val, op=MPI.SUM)
    # RECOMMENDED FIX
    except Exception as e:
        logger.error(f"CRITICAL: Could not calculate LV Volume via tag OR surface integral: {e}")
        # Stop the script immediately
        raise RuntimeError("Cannot determine LV target volume. Check mesh markers.")

try:
    rvv_target = comm.allreduce(geometry.volume("RV"), op=MPI.SUM)
except Exception:
    logger.info("Volume tag 'RV' not found. Calculating cavity volume from surface 'ENDO_RV'...")
    x = ufl.SpatialCoordinate(geometry.mesh)
    n = ufl.FacetNormal(geometry.mesh)
    try:
        ds_rv = parser_ds(geometry.ds, geometry.markers["ENDO_RV"][0])
        val = dolfinx.fem.assemble_scalar(dolfinx.fem.form(-1.0/3.0 * ufl.dot(x, n) * ds_rv))
        rvv_target = comm.allreduce(val, op=MPI.SUM)
    except Exception as e:
        logger.warning(f"Failed to calc RV volume from surface: {e}")

# Helper to assist with parser_ds above which isn't defined
def parser_ds(ds_measure, marker_id):
    return ds_measure(marker_id)
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

if comm.rank == 0:
    history, circ_state = run_0D(init_state=None)
    np.save(outdir / "state.npy", circ_state, allow_pickle=True)
    np.save(outdir / "history.npy", history, allow_pickle=True)
comm.Barrier()

history = np.load(outdir / "history.npy", allow_pickle=True).item()
circ_state = np.load(outdir / "state.npy", allow_pickle=True).item()

error_LV = circ_state["V_LV"] - init_state_circ["V_LV"].magnitude
error_RV = circ_state["V_RV"] - init_state_circ["V_RV"].magnitude

# Plotting 0D results (Rank 0 only)
if comm.rank == 0:
    fig, ax = plt.subplots(2, 2, sharex=True, sharey="row", figsize=(10, 5))
    ax[0, 0].plot(history["V_LV"], history["p_LV"])
    ax[0, 0].set_title("All beats")
    ax[0, 1].plot(history["V_LV"][-1000:], history["p_LV"][-1000:])
    ax[0, 1].set_title("Last beat")
    ax[1, 0].plot(history["V_RV"], history["p_RV"])
    ax[1, 1].plot(history["V_RV"][-1000:], history["p_RV"][-1000:])
    fig.savefig(outdir / "0D_circulation_pv.png")
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
    return np.array([100 * value, 70 * value, 100 * value])

if comm.rank == 0:
    fig, ax = plt.subplots(figsize=(12, 5))
    # Plot one full cardiac cycle starting from ED
    t = np.linspace(0, RR_INTERVAL, 200)
    activation_curve = get_activation(t)
    ax.plot(t, activation_curve.T, label=['LV', 'Septum', 'RV'], linewidth=2)

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

    fig.savefig(outdir / "activation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


# --- Setup Problem (From V1: Uses Scifem and markers_mt) ---

def setup_problem(geometry, f0, s0, material_params, alpha_epi_val=1e5, alpha_base_val=1e6):
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
    comp_model = pulse.compressibility.Compressible2()

    model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    alpha_epi = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(alpha_epi_val)), "Pa / m",
    )
    robin_epi = pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])

    alpha_base = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(alpha_base_val)), "Pa / m",
    )
    robin_base = pulse.RobinBC(value=alpha_base, marker=geometry.markers["BASE"][0])
    robin = [robin_epi, robin_base]

    def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
        facets = geometry.facet_tags.find(geometry.markers["BASE"][0])
        dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), 2, facets)
        return [dolfinx.fem.dirichletbc(0.0, dofs, V.sub(0))]

    return model, robin, dirichlet_bc, Ta


material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
model, robin, dirichlet_bc, Ta = setup_problem(
    geometry=geometry, f0=geo.f0, s0=geo.s0, material_params=material_params,
    alpha_epi_val=args.alpha_epi, alpha_base_val=args.alpha_base
)

# --- Prestressing (Inverse Elasticity) ---

p_LV_ED = mmHg_to_kPa(history["p_LV"][-1])
p_RV_ED = mmHg_to_kPa(history["p_RV"][-1])

logger.info(f"Target ED Pressures: p_LV={p_LV_ED:.2f} kPa, p_RV={p_RV_ED:.2f} kPa")

pressure_lv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa")
pressure_rv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa")

# FIX: Use ENDO_LV/RV markers directly for surface traction if "LV"/"RV" missing
lv_marker_id = geometry.markers["LV"][0] if "LV" in geometry.markers else geometry.markers["ENDO_LV"][0]
rv_marker_id = geometry.markers["RV"][0] if "RV" in geometry.markers else geometry.markers["ENDO_RV"][0]

neumann_lv = pulse.NeumannBC(traction=pressure_lv, marker=lv_marker_id)
neumann_rv = pulse.NeumannBC(traction=pressure_rv, marker=rv_marker_id)

bcs_prestress = pulse.BoundaryConditions(
    robin=robin, dirichlet=(dirichlet_bc,), neumann=(neumann_lv, neumann_rv),
)

prestress_fname = outdir / "prestress_biv_inverse.bp"
if not prestress_fname.exists():
    logger.info("Start prestressing...")
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
        comm, outdir / "prestress_biv_backward.bp", [u_pre], engine="BP4",
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
f0_map = pulse.utils.map_vector_field(
    geo.additional_data["f0_DG_1"], u=u_pre, normalize=True, name="f0",
)
s0_map = pulse.utils.map_vector_field(
    geo.additional_data.get("s0_DG_1", geo.s0), u=u_pre, normalize=True, name="s0",
)

# Robust Volume Calculation (Handle missing 'LV' tags)
x = ufl.SpatialCoordinate(geometry.mesh)
n = ufl.FacetNormal(geometry.mesh)

try:
    lvv_unloaded = comm.allreduce(geometry.volume("LV"), op=MPI.SUM)
except:
    ds_lv = parser_ds(geometry.ds, geometry.markers["ENDO_LV"][0])
    val = dolfinx.fem.assemble_scalar(dolfinx.fem.form(-1.0/3.0 * ufl.dot(x, n) * ds_lv))
    lvv_unloaded = comm.allreduce(val, op=MPI.SUM)

try:
    rvv_unloaded = comm.allreduce(geometry.volume("RV"), op=MPI.SUM)
except:
    ds_rv = parser_ds(geometry.ds, geometry.markers["ENDO_RV"][0])
    val = dolfinx.fem.assemble_scalar(dolfinx.fem.form(-1.0/3.0 * ufl.dot(x, n) * ds_rv))
    rvv_unloaded = comm.allreduce(val, op=MPI.SUM)

logger.info(f"Unloaded volumes: LV={lvv_unloaded * volume2ml:.2f} mL, RV={rvv_unloaded * volume2ml:.2f} mL")

model, robin, dirichlet_bc, Ta = setup_problem(
    geometry=geometry, f0=f0_quad, s0=s0_quad, material_params=material_params,
)

lv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(lvv_unloaded))
rv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(rvv_unloaded))

# Use correct markers for Cavity definitions
lv_marker = "LV" if "LV" in geometry.markers else "ENDO_LV"
rv_marker = "RV" if "RV" in geometry.markers else "ENDO_RV"

cavities = [
    pulse.problem.Cavity(marker=lv_marker, volume=lv_volume),
    pulse.problem.Cavity(marker=rv_marker, volume=rv_volume),
]

bcs_forward = pulse.BoundaryConditions(robin=robin, dirichlet=(dirichlet_bc,))

problem = pulse.problem.StaticProblem(
    model=model,
    geometry=geometry,
    bcs=bcs_forward,
    cavities=cavities,
    parameters={"mesh_unit": mesh_unit, "u_space": "P_2"},
)

# Setup Stress/Strain Post-processing
# FIXED: Use full CardiacModel (material + compressibility) instead of material only
# This ensures stresses include pressure contribution for proper boundary work calculation

W = dolfinx.fem.functionspace(geometry.mesh, ("DG", 1))
I = ufl.Identity(3)
F = ufl.variable(ufl.grad(problem.u) + I)
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
vtx = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "displacement.bp", [problem.u], engine="BP4")
# Edited to include Ta
vtx_stress = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "stress_strain.bp", [fiber_stress, fiber_strain, Ta.value], engine="BP4")

# --- Inflation (Reference -> End-Diastole) ---

logger.info("Inflating to End-Diastolic Target...")
ramp_steps = 10
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

vtx.write(0.0)
vtx_stress.write(0.0)

# Store old values (handling Array for Ta due to scifem/V1)
problem.old_Ta = Ta.value.x.array.copy()
problem.old_lv_volume = lv_volume.value.copy()
problem.old_rv_volume = rv_volume.value.copy()

# --- Multiscale Coupling Loop (Hybrid Logic) ---

def p_BiV_func(V_LV, V_RV, t):
    logger.info(f"Coupling Time {t:.4f}: Target V_LV={V_LV:.2f}, V_RV={V_RV:.2f}")

    # Logic from V1: Get array value
    value = get_activation(t)
    old_Ta = problem.old_Ta
    dTa = value - old_Ta # Vector subtraction

    new_value_LV = (V_LV - error_LV) * (1.0 / volume2ml)
    new_value_RV = (V_RV - error_RV) * (1.0 / volume2ml)

    old_lv_volume = problem.old_lv_volume
    old_rv_volume = problem.old_rv_volume

    dLV = new_value_LV - old_lv_volume
    dRV = new_value_RV - old_rv_volume

    converged = False
    num_failures = 0
    num_steps = 1
    tol = 1e-12

    old_lv_it = old_lv_volume.copy()
    old_rv_it = old_rv_volume.copy()
    old_Ta_it = Ta.value.x.array.copy()

    # Hybrid check: abs for volumes, max(abs) for Ta array
    if abs(dLV) > tol or abs(dRV) > tol or np.max(np.abs(dTa)) > tol:

        while not converged and num_failures < 20:
            for i in range(num_steps):
                lv_volume.value = old_lv_volume + (i + 1) * (new_value_LV - old_lv_it) / num_steps
                rv_volume.value = old_rv_volume + (i + 1) * (new_value_RV - old_rv_it) / num_steps

                # V1 Logic: Update the array
                Ta.assign(old_Ta + (i + 1) * dTa / num_steps)

                try:
                    problem.solve()
                except RuntimeError as e:
                    # V2 Logic: Robust logging and reset
                    print(f"Error during solve: {e}")
                    lv_volume.value = old_lv_volume.copy()
                    rv_volume.value = old_rv_volume.copy()
                    Ta.assign(old_Ta)
                    problem.reset()
                    num_failures += 1
                    num_steps *= 2
                    converged = False
                else:
                    converged = True
                    old_lv_it = lv_volume.value.copy()
                    old_rv_it = rv_volume.value.copy()
                    old_Ta_it = Ta.value.x.array.copy()

            if not converged:
                msg = f"Failed to converge. LV: {new_value_LV}, RV: {new_value_RV}, Ta max: {np.max(value)}"
                logger.error(msg)
                raise RuntimeError("Failed to converge on pressure calculation.")

    problem.old_Ta = Ta.value.x.array.copy()
    problem.old_lv_volume = lv_volume.value.copy()
    problem.old_rv_volume = rv_volume.value.copy()

    lv_p_kPa = problem.cavity_pressures[0].x.array[0] * 1e-3
    rv_p_kPa = problem.cavity_pressures[1].x.array[0] * 1e-3

    return circulation.units.kPa_to_mmHg(lv_p_kPa), circulation.units.kPa_to_mmHg(rv_p_kPa)

# --- Import Metrics Calculator ---
from metrics_calculator import MetricsCalculator

# --- Checkpointing and Callback ---

filename = outdir / Path("function_checkpoint.bp")
if comm.rank == 0:
    shutil.rmtree(filename, ignore_errors=True)
comm.barrier()

adios4dolfinx.write_mesh(filename, geometry.mesh)
adios4dolfinx.write_meshtags(filename, mesh=geometry.mesh, meshtags=geometry.facet_tags, meshtag_name="ffun")
# Write the markers_mt from V1 logic as well
adios4dolfinx.write_meshtags(filename, mesh=geometry.mesh, meshtags=geo.additional_data["markers_mt"], meshtag_name="cfun")

# --- Setup "Sublime" Logging ---
# This file records the heartbeat of the simulation step-by-step
trace_log_path = outdir / "active_mechanics_trace.csv"

if comm.rank == 0:
    with open(trace_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Step", "Time", 
            "Ta_Input_Func",    # The theoretical value from get_activation(t)
            "Ta_Solver_Max",    # What the solver actually used
            "Ta_Metrics_Max",   # What the calculator saw (Must match Solver!)
            "S_Active_Max",     # The resulting physical stress (kPa)
            "Work_Active_Inc",  # Incremental work done this step
            "Sync_Error"        # Check if Solver and Metrics are out of sync
        ])
    logger.info(f"Trace log initialized: {trace_log_path}")

output_file = outdir / "output.json"
Ta_history: list[float] = []

# --- Initialize Metrics Calculator ---
# Prepare fiber field dictionary in current configuration
fiber_fields_map = {
    'f0': f0_quad,
    's0': s0_quad,
    'n0': geo.n0,  # Normal (sheet normal)
    'l0': None,    # Longitudinal (will be computed if needed)
    'c0': None,    # Circumferential (will be computed if needed)
}

# Supervisor suggestion: build a metrics-only CardiacModel with DG-space fibers
material_metrics = pulse.HolzapfelOgden(f0=f0_map, s0=s0_map, **material_params)
active_metrics = pulse.ActiveStress(f0_map, activation=Ta)
comp_metrics = pulse.compressibility.Compressible2()
metrics_model = pulse.CardiacModel(
    material=material_metrics,
    active=active_metrics,
    compressibility=comp_metrics,
)

metrics_type_arg = ("DG", 0) if args.metrics_space == "DG0" else ("DG", 1)

metrics_calc = MetricsCalculator(
    geometry=geometry,
    geo=geo,
    fiber_field_map=fiber_fields_map,
    problem=problem,
    comm=comm,
    cardiac_model=metrics_model,  # Use DG fiber model for metrics stress
    metrics_space_type=metrics_type_arg
)

if comm.rank == 0:
    logger.info("Metrics calculator initialized for True Work vs Clinical Proxies")

def callback(model, i: int, t: float, save=True):
    # 1. Update Inputs
    raw_activation_vec = get_activation(t)
    Ta_history.append(raw_activation_vec)
    fiber_stress.interpolate(fiber_stress_expr)
    fiber_strain.interpolate(fiber_strain_expr)
    
    # 2. FORCE SYNC
    solver_ta_array = Ta.value.x.array[:]
    metrics_calc.cardiac_model.active.activation.value.x.array[:] = solver_ta_array  
    max_ta_solver = np.max(solver_ta_array)

    # 3. METRICS CALCULATION
    if i == 0:
        metrics_calc.update_state()
        metrics_calc_skip_work = True
    else:
        metrics_calc_skip_work = False

    # Get current state
    lv_p_kPa = problem.cavity_pressures[0].x.array[0] * 1e-3
    rv_p_kPa = problem.cavity_pressures[1].x.array[0] * 1e-3
    current_state = {
        "p_LV": circulation.units.kPa_to_mmHg(lv_p_kPa),
        "p_RV": circulation.units.kPa_to_mmHg(rv_p_kPa),
        "V_LV": float(lv_volume.value * volume2ml),
        "V_RV": float(rv_volume.value * volume2ml)
    }
    
    if hasattr(model, "V_LV"):
        current_state.setdefault("V_LV_0D", model.V_LV)
    if hasattr(model, "V_RV"):
        current_state.setdefault("V_RV_0D", model.V_RV)

    region_metrics = metrics_calc.compute_regional_metrics(
        timestep_idx=i, t=t,
        model_history=model.history,
        skip_work_calc=metrics_calc_skip_work,
        current_state=current_state
    )

    # 4. SUBLIME LOGGING (Expanded)
    if comm.rank == 0:
        s_act_max = region_metrics.get("debug_S_active_max", 0.0)
        
        # Energy Breakdown
        w_total = region_metrics.get("work_true_LV", 0.0)
        w_fiber = region_metrics.get("work_fiber_LV", 0.0)
        w_sheet = region_metrics.get("work_sheet_LV", 0.0)
        w_normal = region_metrics.get("work_normal_LV", 0.0)
        w_shear = region_metrics.get("work_shear_LV", 0.0)
        w_passive = region_metrics.get("work_passive_LV", 0.0)
        w_robin = region_metrics.get("work_robin_epi", 0.0)
        
        # NEW: Pressure-Strain Index
        w_ps_idx = region_metrics.get("work_ps_index_LV", 0.0)

        ta_internal = region_metrics.get("debug_Ta_internal_max", 0.0)
        sync_error = abs(max_ta_solver - ta_internal)

        if i == 0 or not trace_log_path.exists():
            with open(trace_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Step", "Time", "Ta_Solver", "S_Act_Max",
                    "W_Total", "W_Fiber", "W_Sheet", "W_Normal", "W_Shear",
                    "W_Passive", "W_Robin_Epi", "W_PS_Index", # <--- Added
                    "Sync_Error"
                ])

        with open(trace_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i, f"{t:.5f}", 
                f"{max_ta_solver:.5f}", 
                f"{s_act_max:.5f}",         
                f"{w_total:.6e}",
                f"{w_fiber:.6e}",
                f"{w_sheet:.6e}",
                f"{w_normal:.6e}",
                f"{w_shear:.6e}",
                f"{w_passive:.6e}",
                f"{w_robin:.6e}",
                f"{w_ps_idx:.6e}", # <--- Added
                f"{sync_error:.1e}"
            ])
            
        if i % 10 == 0:
            print(f"STEP {i:04d} | t={t:.3f} | W_Tot={w_total:.1e} | W_Fib={w_fiber:.1e} | W_PS_Idx={w_ps_idx:.1e}")

    region_metrics["Ta"] = max_ta_solver
    # Store metrics
    metrics_calc.store_metrics(region_metrics, i, t, downsample_factor=1)
    metrics_calc.update_state()

    # 5. Save Files
    if save:
        vtx.write(t)
        vtx_stress.write(t)
        adios4dolfinx.write_function(filename, u=problem.u, name="displacement", time=t)
        adios4dolfinx.write_function(filename, u=fiber_stress, name="fiber_stress", time=t)
        adios4dolfinx.write_function(filename, u=fiber_strain, name="fiber_strain", time=t)
        
        out = {k: v[: i + 1] for k, v in model.history.items()}
        out["Ta"] = Ta_history
        V_LV = model.history["V_LV"][: i + 1] - error_LV
        V_RV = model.history["V_RV"][: i + 1] - error_RV
        out["V_LV"] = V_LV
        out["V_RV"] = V_RV

        if comm.rank == 0:
            output_file.write_text(json.dumps(out, indent=4, default=custom_json))

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
num_beats = 1  # Single beat simulation
dt = 0.001

# CI Mode: Only 2 timesteps for quick testing; Production: Full beat
if CI_MODE:
    end_time = 2 * dt  # ~0.002s for quick validation
    logger.info(f"⚠️  CI MODE: Running only {end_time}s ({int(end_time/dt)} timesteps)")
else:
    end_time = None  # Full beat
    logger.info(f"✓ PRODUCTION MODE: Running full beat ({RR_INTERVAL:.3f}s)")

try:
    circulation_model.solve(num_beats=num_beats, initial_state=circ_state, dt=dt, T=end_time)
    logger.info("Simulation complete.")
finally:
    # --- Save Metrics (ALWAYS, even if simulation crashes) ---
    if comm.rank == 0:
        logger.info("Saving mechanics metrics (true work vs clinical proxies)...")
        try:
            # Save with downsampling options: full resolution, every 5th step, every 10th step
            metrics_calc.save_metrics(outdir, downsample_factors=[1, 5, 10])
            logger.info("✓ Metrics saved to results directory")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")