#!/usr/bin/env python3
"""
postprocess_metrics.py — Offline Metrics Computation from Simulation Checkpoints

Reconstructs the FEM problem from saved checkpoint data and replays the displacement
history to compute all work/stress/strain metrics without re-running the solver.

Required files in results_dir:
  - solver/checkpoint.bp         (displacement at each timestep)
  - solver/Ta_solver_history.npy (active tension per timestep: [N, 3] for [LV, Sep, RV])
  - simulation_params.json       (material params, BCs, activation settings)
  - geometry/geometry.bp         (mesh, facet tags, region markers)
  - circulation/history.npy      (pressure/volume from 0D model)

Usage:
  python3 postprocess_metrics.py <results_directory>
  mpirun -n 4 python3 postprocess_metrics.py <results_directory>
"""

import json
import sys
import logging
from pathlib import Path

import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
import scifem
import adios4dolfinx
import pulse
import cardiac_geometries
import cardiac_geometries.geometry

# ─── Parse Arguments ──────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python3 postprocess_metrics.py <results_directory>")
    sys.exit(1)

results_dir = Path(sys.argv[1]).resolve()
comm = MPI.COMM_WORLD
rank = comm.rank

logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
logger = logging.getLogger("postprocess")

# ─── 1. Load Saved Parameters ────────────────────────────────────────────────

sim_params_path = results_dir / "simulation_params.json"
if not sim_params_path.exists():
    logger.error(f"Missing {sim_params_path} — was this simulation run with the updated complete_cycle.py?")
    sys.exit(1)

with open(sim_params_path) as f:
    sim_params = json.load(f)

if rank == 0:
    logger.info(f"Loaded simulation parameters from {sim_params_path}")
    logger.info(f"  BPM={sim_params['BPM']}, dt={sim_params['dt']}, "
                f"incompressible={sim_params['incompressible']}")

# ─── 2. Load Ta History ──────────────────────────────────────────────────────

solver_dir = results_dir / "solver"
ta_path = solver_dir / "Ta_solver_history.npy"
if not ta_path.exists():
    logger.error(f"Missing {ta_path} — cannot reconstruct active stress")
    sys.exit(1)

Ta_history = np.load(ta_path)
if rank == 0:
    logger.info(f"Loaded Ta history: {Ta_history.shape} (timesteps x regions)")

# ─── 3. Load Circulation History (Pressures/Volumes) ─────────────────────────

circ_path = results_dir / "circulation" / "history.npy"
if not circ_path.exists():
    circ_path = results_dir / "history.npy"
if not circ_path.exists():
    logger.error("Missing circulation history.npy")
    sys.exit(1)

circ_history = np.load(circ_path, allow_pickle=True).item()
if rank == 0:
    logger.info(f"Loaded circulation history: {len(circ_history['time'])} points")

# ─── 4. Load Geometry ────────────────────────────────────────────────────────

geo_dir = results_dir / "geometry"
if not geo_dir.exists():
    logger.error(f"Missing geometry directory: {geo_dir}")
    sys.exit(1)

geo = cardiac_geometries.geometry.Geometry.from_folder(geo_dir, comm=comm)
geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})

if rank == 0:
    logger.info("Geometry loaded successfully")

# ─── 5. Apply Prestress (Deform to Reference Configuration) ──────────────────

prestress_fname = solver_dir / "prestress_inverse.bp"
if not prestress_fname.exists():
    logger.error(f"Missing prestress file: {prestress_fname}")
    sys.exit(1)

V = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 2, (3,)))
u_pre = dolfinx.fem.Function(V)
adios4dolfinx.read_function(prestress_fname, u_pre, time=0.0, name="u_pre")

if rank == 0:
    logger.info("Deforming mesh to Reference Configuration...")
geometry.deform(u_pre)

# Map fibers to reference configuration
f0_quad = pulse.utils.map_vector_field(f=geo.f0, u=u_pre, normalize=True, name="f0_unloaded")
s0_quad = pulse.utils.map_vector_field(f=geo.s0, u=u_pre, normalize=True, name="s0_unloaded")

if rank == 0:
    logger.info("Fibers mapped to Reference Configuration")

# ─── 6. Reconstruct Problem (Material + BCs + Cavities) ─────────────────────

# Material parameters — reconstruct pulse.Variable objects with saved units
mat_params_raw = sim_params["material_params"]
material_params = {}
for k, entry in mat_params_raw.items():
    material_params[k] = pulse.Variable(entry["value"], entry["unit"])

material = pulse.HolzapfelOgden(f0=f0_quad, s0=s0_quad, **material_params)

# Active stress model
markers_mt = geo.additional_data["markers_mt"]
V_Ta = scifem.create_space_of_simple_functions(
    mesh=geo.mesh,
    cell_tag=markers_mt,
    tags=[1, 2, 3]
)
Ta = pulse.Variable(dolfinx.fem.Function(V_Ta), "kPa")
active_model = pulse.ActiveStress(f0_quad, activation=Ta)

# Compressibility
if sim_params["incompressible"]:
    comp_model = pulse.compressibility.Incompressible()
else:
    comp_model = pulse.compressibility.Compressible2()

cardiac_model = pulse.CardiacModel(
    material=material,
    active=active_model,
    compressibility=comp_model,
)

# Boundary conditions
alpha_epi = sim_params["alpha_epi"]
alpha_base = sim_params["alpha_base"]

robin_epi = pulse.RobinBC(
    value=pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(alpha_epi)), "Pa / m"),
    marker=geometry.markers["EPI"][0]
)
robin_base = pulse.RobinBC(
    value=pulse.Variable(dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(alpha_base)), "Pa / m"),
    marker=geometry.markers["BASE"][0]
)

# Dirichlet BC
def dirichlet_bc(V_space):
    facets = geometry.facet_tags.find(geometry.markers["BASE"][0])
    if sim_params["incompressible"]:
        dofs = dolfinx.fem.locate_dofs_topological(V_space, 2, facets)
        u_zero = dolfinx.fem.Function(V_space)
        u_zero.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_zero, dofs)]
    else:
        V_x = V_space.sub(0)
        dofs = dolfinx.fem.locate_dofs_topological(V_x, 2, facets)
        return [dolfinx.fem.dirichletbc(0.0, dofs, V_x)]

# Cavities
volume2ml = sim_params["volume2ml"]
lvv_unloaded = sim_params["lvv_unloaded_m3"]
rvv_unloaded = sim_params["rvv_unloaded_m3"]

lv_marker = "LV" if "LV" in geometry.markers else "ENDO_LV"
rv_marker = "RV" if "RV" in geometry.markers else "ENDO_RV"

lv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(lvv_unloaded))
rv_volume = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(rvv_unloaded))

cavities = [
    pulse.problem.Cavity(marker=lv_marker, volume=lv_volume),
    pulse.problem.Cavity(marker=rv_marker, volume=rv_volume),
]

bcs = pulse.BoundaryConditions(robin=[robin_epi, robin_base], dirichlet=(dirichlet_bc,))

problem = pulse.problem.StaticProblem(
    model=cardiac_model,
    geometry=geometry,
    bcs=bcs,
    cavities=cavities,
    parameters={"mesh_unit": sim_params["mesh_unit"], "u_space": "P_2"},
)

if rank == 0:
    logger.info("Problem reconstructed (material + BCs + cavities)")

# ─── 7. Initialize Metrics Calculator ────────────────────────────────────────

from metrics_calculator import MetricsCalculator

# Fiber field map
l0_field = geo.additional_data.get("apex_gradient", None)
if l0_field is not None and rank == 0:
    logger.info("Loaded apex_gradient — longitudinal strain (E_ll) available")

fiber_fields_map = {
    'f0': f0_quad,
    's0': s0_quad,
    'n0': geo.n0,
    'l0': l0_field,
    'c0': None,
}

# Use the same model for metrics (shares material/compressibility instances)
active_metrics = pulse.ActiveStress(f0_quad, activation=Ta)
metrics_model = pulse.CardiacModel(
    material=cardiac_model.material,
    active=active_metrics,
    compressibility=cardiac_model.compressibility,
)

metrics_calc = MetricsCalculator(
    geometry=geometry,
    geo=geo,
    fiber_field_map=fiber_fields_map,
    problem=problem,
    comm=comm,
    cardiac_model=metrics_model,
    metrics_space_type=("Quadrature", 6),
    alpha_epi=alpha_epi,
    alpha_base=alpha_base,
    hydro_pressure=problem.p if sim_params["incompressible"] else None,
)

if rank == 0:
    logger.info("MetricsCalculator initialized")

# ─── 8. Read Displacement Timestamps ─────────────────────────────────────────

checkpoint_path = solver_dir / "checkpoint.bp"
if not checkpoint_path.exists():
    logger.error(f"Missing {checkpoint_path}")
    sys.exit(1)

timestamps = adios4dolfinx.read_timestamps(checkpoint_path, comm, "displacement")
if rank == 0:
    logger.info(f"Found {len(timestamps)} displacement timesteps "
                f"(t={timestamps[0]:.4f} to {timestamps[-1]:.4f})")

# ─── 9. Build Pressure/Volume Interpolator ───────────────────────────────────

# Circulation history has finer time resolution (0.1ms) than FEM checkpoints (1ms).
# We need to interpolate pressures at exact checkpoint times.
circ_time = np.array(circ_history["time"])
circ_p_LV = np.array(circ_history["p_LV"])   # in mmHg (from 0D model)
circ_p_RV = np.array(circ_history["p_RV"])
circ_V_LV = np.array(circ_history["V_LV"])   # in mL (0D units)
circ_V_RV = np.array(circ_history["V_RV"])

ratio_LV = sim_params["ratio_LV"]
ratio_RV = sim_params["ratio_RV"]

def get_state_at_time(t_fem):
    """Interpolate circulation state at FEM checkpoint time."""
    p_lv = float(np.interp(t_fem, circ_time, circ_p_LV))
    p_rv = float(np.interp(t_fem, circ_time, circ_p_RV))
    v_lv = float(np.interp(t_fem, circ_time, circ_V_LV))
    v_rv = float(np.interp(t_fem, circ_time, circ_V_RV))
    return {
        "p_LV": p_lv,
        "p_RV": p_rv,
        "V_LV": v_lv * ratio_LV,     # Scale to mesh volume
        "V_RV": v_rv * ratio_RV,
        "V_LV_Clinical": v_lv,
        "V_RV_Clinical": v_rv,
    }

# ─── 10. Replay Loop ─────────────────────────────────────────────────────────

if rank == 0:
    logger.info("=" * 60)
    logger.info("  REPLAYING DISPLACEMENT HISTORY")
    logger.info("=" * 60)

n_steps = len(timestamps)

# Verify Ta history length matches
if len(Ta_history) != n_steps:
    if rank == 0:
        logger.warning(f"Ta history length ({len(Ta_history)}) != checkpoint steps ({n_steps}). "
                       f"Using min of both.")
    n_steps = min(n_steps, len(Ta_history))

for i in range(n_steps):
    t = timestamps[i]

    # 1. Load displacement from checkpoint into problem.u
    adios4dolfinx.read_function(checkpoint_path, problem.u, time=t, name="displacement")

    # 2. Set active tension from saved history
    Ta.assign(Ta_history[i])
    # Sync to metrics model
    metrics_model.active.activation.value.x.array[:] = Ta.value.x.array[:]

    # 3. Get circulation state at this time
    current_state = get_state_at_time(t)

    # 4. Compute metrics
    if i == 0:
        metrics_calc.update_state()
        skip_work = True
    else:
        skip_work = False

    region_metrics = metrics_calc.compute_regional_metrics(
        timestep_idx=i, t=t,
        model_history=circ_history,
        skip_work_calc=skip_work,
        current_state=current_state,
    )

    # Enrich with state info
    region_metrics.update(current_state)
    region_metrics["Ta_Solver"] = float(np.max(Ta_history[i]))

    # Store
    metrics_calc.store_metrics(region_metrics, i, t, downsample_factor=1)
    metrics_calc.update_state()

    # Progress
    if rank == 0 and (i % 50 == 0 or i == n_steps - 1):
        w_lv = region_metrics.get("work_true_LV", 0.0)
        e_ff = region_metrics.get("mean_E_ff_LV", 0.0)
        p_lv = current_state["p_LV"]
        ta_max = float(np.max(Ta_history[i]))
        logger.info(f"  Step {i:04d}/{n_steps} | t={t:.3f} | Ta={ta_max:.1f} | "
                    f"p_LV={p_lv:.1f}mmHg | E_ff={e_ff:.3f} | W_LV={w_lv:.2e}")

# ─── 11. Save Metrics ────────────────────────────────────────────────────────

metrics_dir = results_dir / "metrics"
if rank == 0:
    metrics_dir.mkdir(exist_ok=True)
    logger.info("Saving metrics...")
    metrics_calc.save_metrics(metrics_dir, downsample_factors=[1, 10])
    logger.info(f"Metrics saved to {metrics_dir}")
    logger.info("Done. Run run_postprocessing.py for plots and analysis.")
