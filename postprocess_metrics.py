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

  # With manually edited septum tags (from septum_editor.py):
  python3 postprocess_metrics.py <results_directory> --edited-tags edited_tags.npz
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
import basix.ufl
import pulse
import cardiac_geometries
import cardiac_geometries.geometry

# ─── Parse Arguments ──────────────────────────────────────────────────────────

import argparse

_parser = argparse.ArgumentParser(description="Offline metrics from simulation checkpoint.")
_parser.add_argument("results_dir", type=Path, help="Path to simulation results directory")
_parser.add_argument(
    "--edited-tags", type=Path, default=None, metavar="NPZ",
    help="Path to edited_tags.npz from septum_editor. Overrides checkpoint "
         "region markers (matched by cell centroid, no DOF-ordering dependency).",
)
_parser.add_argument(
    "--last-beat", action="store_true", default=False,
    help="Only process the last cardiac cycle (saves time for converged multi-beat sims).",
)
_parser.add_argument(
    "--retag-septum", action="store_true", default=False,
    help="Recompute LV/RV/Septum region tags using the geometric distance-based method "
         "(ldrb geometric-septum-tagging). Overrides markers from checkpoint.",
)
_args = _parser.parse_args()

results_dir = _args.results_dir.resolve()
edited_tags_path = _args.edited_tags
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

# FIX: The history.npy may contain multi-beat 0D pre-run data (e.g. 10 beats, 0-8s)
# while the FEM checkpoint is only 1 coupled beat (0-0.8s). If we interpolate FEM times
# against this multi-beat history, we read from beat 1 (transient) instead of the last
# (quasi-steady) beat. Detect this and use only the last beat.
circ_time_arr = np.array(circ_history["time"])
HR_Hz = sim_params["BPM"] / 60.0
cycle_length = 1.0 / HR_Hz
circ_duration = circ_time_arr[-1] - circ_time_arr[0]

if circ_duration > cycle_length * 1.5:
    # Multi-beat history detected — extract last beat
    last_beat_start = circ_time_arr[-1] - cycle_length
    mask = circ_time_arr >= last_beat_start - 1e-10
    if rank == 0:
        logger.warning(
            f"⚠️  Multi-beat circulation history detected ({circ_duration:.1f}s = "
            f"{circ_duration/cycle_length:.0f} beats). Extracting last beat from "
            f"t={last_beat_start:.2f}s onward ({np.sum(mask)} points)."
        )
    # Shift time so last beat starts at 0
    for key in circ_history:
        arr = np.array(circ_history[key])
        if len(arr) == len(circ_time_arr):
            circ_history[key] = arr[mask]
    circ_history["time"] = np.array(circ_history["time"]) - last_beat_start

if rank == 0:
    logger.info(f"Loaded circulation history: {len(circ_history['time'])} points, "
                f"time range: {circ_history['time'][0]:.4f} - {circ_history['time'][-1]:.4f}s")

# ─── 4. Load Geometry from Checkpoint ─────────────────────────────────────────
#
# CRITICAL: Load the mesh from checkpoint.bp — the SAME file where displacement
# functions are stored. adios4dolfinx DOF ordering is tied to the specific file
# where the mesh was written. Loading from a different file (geometry.bp, or even
# a separate reference_mesh.bp) creates different DOF numbering, corrupting data.

checkpoint_path = solver_dir / "checkpoint.bp"
if not checkpoint_path.exists():
    logger.error(f"Missing checkpoint: {checkpoint_path}")
    sys.exit(1)

# Load mesh (already in reference configuration, in meters)
mesh = adios4dolfinx.read_mesh(checkpoint_path, comm)
if rank == 0:
    logger.info(f"Reference mesh loaded (ndofs={mesh.geometry.x.shape[0]}, "
                f"coords range=[{mesh.geometry.x.min():.6f}, {mesh.geometry.x.max():.6f}])")

# Load markers from checkpoint (same file = same DOF ordering)
ffun = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="ffun")
markers_mt = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="cfun")

# ─── Optional: Override region markers from edited_tags.npz ─────────────────
# The septum_editor exports cell centroids + edited tags in a .npz file.
# We match cells between the XDMF (editor) and .bp (checkpoint) meshes by
# nearest centroid — no DOF-ordering dependency, fully safe.
if edited_tags_path is not None:
    from scipy.spatial import cKDTree

    if rank == 0:
        logger.info(f"Loading edited region tags from {edited_tags_path}")

    edited = np.load(edited_tags_path)
    edit_centroids = edited["centroids"]  # (n_edit, 3)
    edit_tags = edited["tags"]            # (n_edit,)

    # Compute centroids of the checkpoint mesh (local cells only)
    mesh.topology.create_connectivity(3, 0)
    imap_mt = mesh.topology.index_map(3)
    n_local = imap_mt.size_local
    local_cells = np.arange(n_local, dtype=np.int32)
    bp_centroids = dolfinx.mesh.compute_midpoints(mesh, 3, local_cells)  # (n_local, 3)

    # Build KDTree on the editor centroids, query with checkpoint centroids
    tree = cKDTree(edit_centroids)
    dists, indices = tree.query(bp_centroids)

    max_dist = dists.max()
    if rank == 0:
        logger.info(f"  Centroid matching: max distance = {max_dist:.2e} m "
                     f"({n_local} local cells matched)")
    if max_dist > 1e-6:
        logger.warning(f"  ⚠️  Large centroid mismatch ({max_dist:.2e} m) — "
                        f"edited tags may not correspond to this mesh!")

    # Build new MeshTags with edited values on the checkpoint mesh
    new_values = edit_tags[indices].astype(np.int32)
    old_values = markers_mt.values[:n_local]
    n_changed = int((new_values != old_values).sum())

    markers_mt = dolfinx.mesh.meshtags(
        mesh, 3,
        np.arange(imap_mt.size_local + imap_mt.num_ghosts, dtype=np.int32),
        np.concatenate([new_values, markers_mt.values[n_local:]]),
    )
    markers_mt.name = "cfun"

    if rank == 0:
        n_lv = int((new_values == 1).sum())
        n_rv = int((new_values == 2).sum())
        n_sept = int((new_values == 3).sum())
        logger.info(f"  Overridden: {n_changed} cells changed "
                     f"(LV={n_lv}, RV={n_rv}, Septum={n_sept})")

# ─── Optional: Geometric septum retagging ────────────────────────────────────
# Recomputes LV/RV/Septum cell tags using distance-based geometry:
#   Septum = max(dist_to_LV_endo, dist_to_RV_endo) < dist_to_EPI
# All data (mesh, ffun, centroids) comes from checkpoint.bp — no DOF mismatch risk.
if _args.retag_septum:
    from scipy.spatial import cKDTree as _cKDTree

    if rank == 0:
        logger.info("Recomputing region tags with geometric distance-based septum tagging...")

    # Log OLD tags for comparison
    _imap = mesh.topology.index_map(3)
    _n_local = _imap.size_local
    _old_tags = markers_mt.values[:_n_local].copy()
    _old_lv = comm.allreduce(int((_old_tags == 1).sum()))
    _old_rv = comm.allreduce(int((_old_tags == 2).sum()))
    _old_sept = comm.allreduce(int((_old_tags == 3).sum()))
    if rank == 0:
        logger.info(f"  OLD tags (global): LV={_old_lv}, RV={_old_rv}, Septum={_old_sept}")

    # Read marker name→tag mapping from geometry folder (just a dict, no DOF dependency)
    _geo_dir_tmp = results_dir / "geometry"
    if rank == 0:
        _geo_tmp = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, _geo_dir_tmp)
        _raw_markers = _geo_tmp.markers
    else:
        _raw_markers = None
    _raw_markers = comm.bcast(_raw_markers, root=0)

    # Build surface coordinate trees from facet tags (all from checkpoint.bp)
    # IMPORTANT: gather surface coords from ALL ranks to build complete trees
    mesh.topology.create_connectivity(2, 0)
    _f2v = mesh.topology.connectivity(2, 0)

    def _surface_coords_global(marker_tag_ids):
        """Get surface vertex coordinates gathered across all MPI ranks."""
        facets = np.hstack([ffun.find(t) for t in marker_tag_ids])
        verts = set()
        for f in facets:
            verts.update(_f2v.links(f))
        local_coords = mesh.geometry.x[np.array(sorted(verts), dtype=np.int64)] if verts else np.empty((0, 3))
        # Gather all local coords to all ranks
        all_coords = comm.allgather(local_coords)
        return np.vstack(all_coords) if any(len(c) > 0 for c in all_coords) else np.empty((0, 3))

    # Determine facet tag IDs (handle both naming conventions)
    _lv_tags = [_raw_markers.get("LV", _raw_markers.get("ENDO_LV", [None]))[0]]
    _rv_tags = [_raw_markers.get("RV", _raw_markers.get("ENDO_RV", [None]))[0]]
    _epi_tags = [_raw_markers.get("EPI", [None])[0]]

    _lv_coords = _surface_coords_global(_lv_tags)
    _rv_coords = _surface_coords_global(_rv_tags)
    _epi_coords = _surface_coords_global(_epi_tags)

    if rank == 0:
        logger.info(f"  Surface vertices (global): LV_endo={len(_lv_coords)}, "
                     f"RV_endo={len(_rv_coords)}, EPI={len(_epi_coords)}")

    _tree_lv = _cKDTree(_lv_coords)
    _tree_rv = _cKDTree(_rv_coords)
    _tree_epi = _cKDTree(_epi_coords)

    # Compute distances at cell centroids (from the checkpoint mesh)
    mesh.topology.create_connectivity(3, 0)
    _centroids = dolfinx.mesh.compute_midpoints(mesh, 3, np.arange(_n_local, dtype=np.int32))

    _d_lv, _ = _tree_lv.query(_centroids)
    _d_rv, _ = _tree_rv.query(_centroids)
    _d_epi, _ = _tree_epi.query(_centroids)

    # Septum = closer to both endo surfaces than to epi
    _is_septum = np.maximum(_d_lv, _d_rv) < _d_epi
    _new_tags = np.where(_is_septum, 3,
                np.where(_d_lv <= _d_rv, 1, 2)).astype(np.int32)

    _n_changed = int((_new_tags != _old_tags).sum())

    markers_mt = dolfinx.mesh.meshtags(
        mesh, 3,
        np.arange(_imap.size_local + _imap.num_ghosts, dtype=np.int32),
        np.concatenate([_new_tags, markers_mt.values[_n_local:]]),
    )
    markers_mt.name = "cfun"

    _new_lv = comm.allreduce(int((_new_tags == 1).sum()))
    _new_rv = comm.allreduce(int((_new_tags == 2).sum()))
    _new_sept = comm.allreduce(int((_new_tags == 3).sum()))
    _n_changed = comm.allreduce(_n_changed)
    if rank == 0:
        logger.info(f"  NEW tags (global): LV={_new_lv}, RV={_new_rv}, Septum={_new_sept}")
        logger.info(f"  Changed:  {_n_changed} cells "
                     f"(LV: {_old_lv}→{_new_lv}, RV: {_old_rv}→{_new_rv}, "
                     f"Septum: {_old_sept}→{_new_sept})")

# Load geometry folder for marker name mapping only
geo_dir = results_dir / "geometry"
geo = cardiac_geometries.geometry.Geometry.from_folder(comm, geo_dir)

# Build HeartGeometry (mesh is already in reference config — NO prestress needed)
geometry = pulse.HeartGeometry(
    mesh=mesh,
    facet_tags=ffun,
    markers=geo.markers,
    metadata={"quadrature_degree": 6},
)

if rank == 0:
    logger.info("Geometry loaded successfully")

# ─── 4b. Compute AHA Regions ─────────────────────────────────────────────────
# Uses LDRB apex-to-base Laplace solution to split LV/RV/Septum into
# Basal (1-3), Mid (4-6) sub-regions for reliable patch-level metrics.

import ldrb.aha
from cardiac_geometries.mesh import transform_markers

ldrb_markers = transform_markers(geo.markers, clipped=True)
aha_func = ldrb.aha.gernerate_aha_biv(
    mesh=mesh, ffun=ffun, markers=ldrb_markers,
    function_space="DG_0",
)

# Convert DG0 function to MeshTags
imap = mesh.topology.index_map(3)
_total_cells = imap.size_local + imap.num_ghosts
aha_values = aha_func.x.array[:_total_cells].astype(np.int32)
aha_mt = dolfinx.mesh.meshtags(mesh, 3, np.arange(_total_cells, dtype=np.int32), aha_values)

if rank == 0:
    aha_labels = {0: "Apical", 1: "Basal_LV", 2: "Basal_RV", 3: "Basal_Septum",
                  4: "Mid_LV", 5: "Mid_RV", 6: "Mid_Septum"}
    logger.info("AHA region counts:")
    local_counts = np.bincount(aha_values, minlength=7)
    for v in range(7):
        logger.info(f"  {aha_labels[v]:>14s} (tag {v}): {local_counts[v]} cells")

# ─── 5. Load Fiber Fields ────────────────────────────────────────────────────
# Fibers were saved into checkpoint.bp by complete_cycle.py (same file = correct DOFs)

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))

q_el = basix.ufl.quadrature_element(mesh.topology.cell_name(), value_shape=(3,), degree=6)
Q_vec = dolfinx.fem.functionspace(mesh, q_el)
f0_quad = dolfinx.fem.Function(Q_vec)
s0_quad = dolfinx.fem.Function(Q_vec)
adios4dolfinx.read_function(checkpoint_path, f0_quad, time=0.0, name="f0")
adios4dolfinx.read_function(checkpoint_path, s0_quad, time=0.0, name="s0")

if rank == 0:
    logger.info("Fiber fields loaded from checkpoint")

# ─── 6. Reconstruct Problem (Material + BCs + Cavities) ─────────────────────

# Material parameters — reconstruct pulse.Variable objects with saved units
mat_params_raw = sim_params["material_params"]
material_params = {}
for k, entry in mat_params_raw.items():
    material_params[k] = pulse.Variable(entry["value"], entry["unit"])

material = pulse.HolzapfelOgden(f0=f0_quad, s0=s0_quad, **material_params)

# Active stress model (markers_mt loaded from checkpoint.bp above)
V_Ta = scifem.create_space_of_simple_functions(
    mesh=mesh,
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

# Load n0 and l0 from checkpoint.bp (if available)
# Q_vec already defined above
n0_quad = None
try:
    n0_quad = dolfinx.fem.Function(Q_vec)
    adios4dolfinx.read_function(checkpoint_path, n0_quad, time=0.0, name="n0")
    if rank == 0:
        logger.info("Loaded n0 from checkpoint")
except Exception:
    if rank == 0:
        logger.warning("n0 not in checkpoint — sheet-normal strain unavailable")

l0_field = None
try:
    l0_field = dolfinx.fem.Function(Q_vec)
    adios4dolfinx.read_function(checkpoint_path, l0_field, time=0.0, name="l0")
    if rank == 0:
        logger.info("Loaded apex_gradient (l0) — longitudinal strain (E_ll) available")
except Exception:
    if rank == 0:
        logger.warning("l0 not in checkpoint — longitudinal strain unavailable")

fiber_fields_map = {
    'f0': f0_quad,
    's0': s0_quad,
    'n0': n0_quad,
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

# Create a geo-like object with markers_mt on the checkpoint mesh
class GeoProxy:
    def __init__(self, mesh, additional_data):
        self.mesh = mesh
        self.additional_data = additional_data

geo_proxy = GeoProxy(mesh, {"markers_mt": markers_mt})

metrics_calc = MetricsCalculator(
    geometry=geometry,
    geo=geo_proxy,
    fiber_field_map=fiber_fields_map,
    problem=problem,
    comm=comm,
    cardiac_model=metrics_model,
    metrics_space_type=("Quadrature", 6),
    alpha_epi=alpha_epi,
    alpha_base=alpha_base,
    hydro_pressure=problem.p if sim_params["incompressible"] else None,
    one_sided_robin=sim_params.get("one_sided_robin", False),
    aha_tags=aha_mt,
)

if rank == 0:
    logger.info("MetricsCalculator initialized")

# ─── 8. Read Displacement Timestamps ─────────────────────────────────────────

timestamps = adios4dolfinx.read_timestamps(checkpoint_path, comm, "displacement")
if rank == 0:
    logger.info(f"Found {len(timestamps)} displacement timesteps "
                f"(t={timestamps[0]:.4f} to {timestamps[-1]:.4f})")

# ─── 9. Build Pressure/Volume Interpolator ───────────────────────────────────

# Circulation history has finer time resolution (0.1ms) than FEM checkpoints (1ms).
# We need to interpolate pressures at exact checkpoint times.
# FIX: If the circulation history was shifted to start at 0 (multi-beat extraction),
# we must also shift the checkpoint timestamps to match, otherwise np.interp
# extrapolates to a constant (the last value).
circ_time = np.array(circ_history["time"])
if len(timestamps) > 0 and timestamps[0] > circ_time[-1]:
    # Checkpoint times are absolute (e.g. 7.2-8.0) but circ was shifted to 0-0.8
    _time_offset = timestamps[0] - circ_time[0]
    if rank == 0:
        logger.info(f"Aligning checkpoint times to circulation: offset={_time_offset:.4f}s")
    timestamps = timestamps - _time_offset
circ_p_LV = np.array(circ_history["p_LV"])   # in mmHg (from 0D model)
circ_p_RV = np.array(circ_history["p_RV"])
circ_V_LV = np.array(circ_history["V_LV"])   # in mL (0D units)
circ_V_RV = np.array(circ_history["V_RV"])

ratio_LV = sim_params["ratio_LV"]
ratio_RV = sim_params["ratio_RV"]

# Load solver cavity pressures if available (exact Lagrange multiplier — much more
# accurate than 0D model pressures for boundary work computation)
solver_pressure_path = solver_dir / "solver_cavity_pressure_mmHg.npy"
if not solver_pressure_path.exists():
    solver_pressure_path = solver_dir / "pressure_history.npy"  # backwards compat
solver_pressures = None
if solver_pressure_path.exists():
    solver_pressures = np.load(solver_pressure_path)  # [N, 2] = [p_LV_mmHg, p_RV_mmHg]
    if rank == 0:
        logger.info(f"Loaded solver cavity pressures: {solver_pressures.shape}")

def get_state_at_time(t_fem, step_idx=None):
    """Interpolate circulation state at FEM checkpoint time.

    Uses solver cavity pressures (Lagrange multiplier) for p_LV/p_RV when available,
    falling back to 0D model pressures otherwise. The solver pressures are the exact
    surface traction and give perfect energy conservation.
    """
    # Pressures: prefer solver cavity pressures over 0D model
    if solver_pressures is not None and step_idx is not None and step_idx < len(solver_pressures):
        p_lv = float(solver_pressures[step_idx, 0])
        p_rv = float(solver_pressures[step_idx, 1])
    else:
        p_lv = float(np.interp(t_fem, circ_time, circ_p_LV))
        p_rv = float(np.interp(t_fem, circ_time, circ_p_RV))

    # 0D model pressures (always available, for comparison)
    p_lv_0d = float(np.interp(t_fem, circ_time, circ_p_LV))
    p_rv_0d = float(np.interp(t_fem, circ_time, circ_p_RV))

    # Volumes: always from 0D model (scaled to mesh)
    v_lv = float(np.interp(t_fem, circ_time, circ_V_LV))
    v_rv = float(np.interp(t_fem, circ_time, circ_V_RV))
    return {
        "p_LV": p_lv,                   # Solver (Lagrange multiplier) pressure
        "p_RV": p_rv,
        "p_LV_0D": p_lv_0d,             # 0D model (elastance) pressure
        "p_RV_0D": p_rv_0d,
        "V_LV": v_lv * ratio_LV,        # 0D volume scaled to mesh frame
        "V_RV": v_rv * ratio_RV,
        "V_LV_Clinical": v_lv,          # Unscaled 0D volume
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

# --last-beat: skip to the last cardiac cycle
start_step = 0
if _args.last_beat:
    cycle_length = sim_params.get("RR_INTERVAL", 60.0 / sim_params.get("BPM", 75))
    last_beat_start_time = timestamps[-1] - cycle_length
    # Find first timestep at or after last_beat_start_time
    start_step = int(np.searchsorted(timestamps, last_beat_start_time - 1e-10))
    if rank == 0:
        logger.info(f"--last-beat: skipping to step {start_step}/{n_steps} "
                    f"(t={timestamps[start_step]:.4f}s, last beat of {timestamps[-1]/cycle_length:.0f})")

for i in range(start_step, n_steps):
    t = timestamps[i]

    # 1. Load displacement from checkpoint into problem.u
    if rank == 0:
        logger.debug(f"  [{i}] Reading displacement at t={t:.4f}...")
        sys.stderr.flush()
    adios4dolfinx.read_function(checkpoint_path, problem.u, time=t, name="displacement")
    if rank == 0:
        logger.debug(f"  [{i}] Displacement loaded. Setting Ta...")
        sys.stderr.flush()

    # 2. Set active tension from saved history
    Ta.assign(Ta_history[i])
    # Sync to metrics model
    metrics_model.active.activation.value.x.array[:] = Ta.value.x.array[:]

    # 3. Get circulation state at this time
    current_state = get_state_at_time(t, step_idx=i)

    if rank == 0:
        logger.debug(f"  [{i}] Computing metrics (skip_work={i==0})...")
        sys.stderr.flush()

    # 4. Compute metrics
    # Note: do NOT call update_state() before compute_regional_metrics at step 0.
    # That would set has_previous_state=True, causing V_LV_FEM (and other init keys)
    # to be skipped — resulting in 799 vs 800 length mismatch.
    skip_work = (i == start_step)

    region_metrics = metrics_calc.compute_regional_metrics(
        timestep_idx=i, t=t,
        model_history=circ_history,
        skip_work_calc=skip_work,
        current_state=current_state,
    )

    # Enrich with state info
    region_metrics.update(current_state)
    region_metrics["Ta_Solver"] = float(np.max(Ta_history[i]))

    # Store using relative index so arrays start at 0
    metrics_calc.store_metrics(region_metrics, i - start_step, t, downsample_factor=1)
    metrics_calc.update_state()

    # Progress
    if rank == 0 and (i % 10 == 0 or i == n_steps - 1):
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
