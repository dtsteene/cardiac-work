# # Complete Multiscale Simulation with Prestressing (Hybrid Version)
#
# This script combines the improved active tension implementation (scifem/spatially varying)
# with robust MPI handling and logging suitable for compute clusters.

import json
import os
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

outdir = Path(f"results_biv_complete_cycle_hybrid_{BPM}bpm")
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
    # Higher fidelity mesh (char_length 10 -> 5.0) for better septum accuracy
    char_length = 5.0

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
    system = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=cardiac_geometries.mesh.transform_markers(geo.markers, clipped=True),
        **fiber_angles,
        fiber_space="Quadrature_6",
    )
    #could change fiber_space to DG_1 or CG_1?
    # Generate DG0 system for Active Tension Markers (From V1)
    # This creates markers scalar which helps define LV/RV/Septum regions
    system_dg0 = ldrb.dolfinx_ldrb(
        mesh=geo.mesh,
        ffun=geo.ffun,
        markers=cardiac_geometries.mesh.transform_markers(geo.markers, clipped=True),
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

# Scale to meters
scale = 1e-3
geo.mesh.geometry.x[:] *= scale

geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

# Store Target Volumes (ED)
volume2ml = 1e6
mesh_unit = "m"

lvv_target = comm.allreduce(geometry.volume("LV"), op=MPI.SUM)
rvv_target = comm.allreduce(geometry.volume("RV"), op=MPI.SUM)
logger.info(
    f"ED Volumes: LV={lvv_target * volume2ml:.2f} mL, RV={rvv_target * volume2ml:.2f} mL",
)

# --- 0D Circulation Model (From V2) ---

def get_updated_parameters():
    """
    Returns Regazzoni2020 parameters consistent with the configured BPM.
    Reference defaults are for 75 BPM (RR=0.8s).

    CRITICAL ALIGNMENT:
    We shift the 0D model phase so that Ventricular Contraction starts at t=0.
    This ensures alignment with the FEM mesh which is at End Diastole.
    """
    params = Regazzoni2020.default_parameters()

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

def setup_problem(geometry, f0, s0, material_params):
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
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e5)), "Pa / m",
    )
    robin_epi = pulse.RobinBC(value=alpha_epi, marker=geometry.markers["EPI"][0])

    alpha_base = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e6)), "Pa / m",
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
)

# --- Prestressing (Inverse Elasticity) ---

p_LV_ED = mmHg_to_kPa(history["p_LV"][-1])
p_RV_ED = mmHg_to_kPa(history["p_RV"][-1])

logger.info(f"Target ED Pressures: p_LV={p_LV_ED:.2f} kPa, p_RV={p_RV_ED:.2f} kPa")

pressure_lv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa")
pressure_rv = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 0.0), "kPa")
neumann_lv = pulse.NeumannBC(traction=pressure_lv, marker=geometry.markers["LV"][0])
neumann_rv = pulse.NeumannBC(traction=pressure_rv, marker=geometry.markers["RV"][0])

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

lvv_unloaded = comm.allreduce(geometry.volume("LV"), op=MPI.SUM)
rvv_unloaded = comm.allreduce(geometry.volume("RV"), op=MPI.SUM)
logger.info(f"Unloaded volumes: LV={lvv_unloaded * volume2ml:.2f} mL, RV={rvv_unloaded * volume2ml:.2f} mL")

model, robin, dirichlet_bc, Ta = setup_problem(
    geometry=geometry, f0=f0_quad, s0=s0_quad, material_params=material_params,
)

lv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(lvv_unloaded))
rv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(rvv_unloaded))
cavities = [
    pulse.problem.Cavity(marker="LV", volume=lv_volume),
    pulse.problem.Cavity(marker="RV", volume=rv_volume),
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
vtx_stress = dolfinx.io.VTXWriter(geometry.mesh.comm, outdir / "stress_strain.bp", [fiber_stress, fiber_strain], engine="BP4")

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

metrics_calc = MetricsCalculator(
    geometry=geometry,
    geo=geo,
    fiber_field_map=fiber_fields_map,
    problem=problem,
    comm=comm,
    cardiac_model=metrics_model  # Use DG fiber model for metrics stress
)

if comm.rank == 0:
    logger.info("Metrics calculator initialized for True Work vs Clinical Proxies")

def callback(model, i: int, t: float, save=True):
    """
    Callback executed at each timestep.

    Computes and stores:
    - True work (stress-based) per region
    - Clinical work proxies (pressure-based) per region
    - Stress/strain quantities

    Skips plotting (wasteful on compute nodes).
    Saves everything at every timestep but allows downsampling later.
    """

    # Interpolate stress/strain fields for checkpointing
    fiber_stress.interpolate(fiber_stress_expr)
    fiber_strain.interpolate(fiber_strain_expr)
    Ta_history.append(get_activation(t))

    # Update state tracking for work calculation
    if i == 0:
        # First step: initialize previous state
        metrics_calc.update_state()
        metrics_calc_skip_work = True
    else:
        metrics_calc_skip_work = False

    # Explicitly get current state (P and V) to ensure metrics calculator has them
    # irrespective of history update timing
    lv_p_kPa = problem.cavity_pressures[0].x.array[0] * 1e-3
    rv_p_kPa = problem.cavity_pressures[1].x.array[0] * 1e-3

    current_state = {
        "p_LV": circulation.units.kPa_to_mmHg(lv_p_kPa),
        "p_RV": circulation.units.kPa_to_mmHg(rv_p_kPa),
    }

    # Prefer directly computed cavity volumes (mechanics side) to avoid empty 0D history timing.
    # lv_volume/rv_volume are in m^3; convert to mL for consistency with circulation volumes.
    V_LV_ml = float(lv_volume.value * volume2ml)
    V_RV_ml = float(rv_volume.value * volume2ml)
    current_state["V_LV"] = V_LV_ml
    current_state["V_RV"] = V_RV_ml

    if comm.rank == 0 and i < 5:
        print(f"DEBUG VOLUMES i={i}: lv_volume.value={float(lv_volume.value):.3e} m^3 → {V_LV_ml:.3f} mL, rv_volume.value={float(rv_volume.value):.3e} m^3 → {V_RV_ml:.3f} mL")

    # If the 0D model exposes volumes, keep them as an alternate sanity check (not primary).
    if hasattr(model, "V_LV"):
        current_state.setdefault("V_LV_0D", model.V_LV)
    if hasattr(model, "V_RV"):
        current_state.setdefault("V_RV_0D", model.V_RV)

    # Compute all metrics for this timestep
    region_metrics = metrics_calc.compute_regional_metrics(
        timestep_idx=i,
        t=t,
        model_history=model.history,
        skip_work_calc=metrics_calc_skip_work,
        current_state=current_state
    )

    # Store metrics (all timesteps; downsampling happens at save time)
    metrics_calc.store_metrics(region_metrics, i, t, downsample_factor=1)

    # Update state for next iteration (after metrics computed)
    metrics_calc.update_state()

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

            # No incremental plotting (wasteful on compute nodes)
            # Plots will be generated in post-processing if needed

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