#!/usr/bin/env python3
"""
compute_per_cell.py — Per-Cell Work Density from Simulation Checkpoints

Replays the last cardiac cycle from checkpoint and computes work density
per mesh cell using DG0 test function assembly. Also computes the Euclidean
transventricular coordinate (tau) and study region definitions.

Output: per_cell_data.npz containing per-cell work arrays, tau, and region masks.

This script reuses the same problem setup as postprocess_metrics.py but adds
per-cell integration via the DG0 test function trick:
    W_per_cell = assemble_vector(form(work_density * v_dg0 * dx))
where v_dg0 is a DG0 TestFunction acting as a spatial partition of unity.

Usage:
    python compute_per_cell.py <results_directory>
    mpirun -n 4 python compute_per_cell.py <results_directory>
"""

import json
import sys
import logging
from pathlib import Path

import numpy as np
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import scifem
import adios4dolfinx
import basix.ufl
import pulse
import cardiac_geometries
import cardiac_geometries.geometry
from scipy.spatial import cKDTree
from petsc4py import PETSc

# pyvista is used for point-to-facet distance (more accurate than
# nearest-vertex on coarse meshes, see verify_sweep_envelope.py)
try:
    import pyvista as pv
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False

# ── Parse Arguments ──────────────────────────────────────────────────────────

import argparse
_parser = argparse.ArgumentParser(description="Per-cell work from checkpoint.")
_parser.add_argument("results_dir", type=Path)
_parser.add_argument("--retag-septum", action="store_true", default=False)
_parser.add_argument("--geometry-fields", type=Path, default=None,
                     help="Path to precomputed geometry_fields.npz (from "
                          "precompute_geometry_fields.py). If provided, tau/distances/"
                          "envelope/entry_t are loaded from this file instead of "
                          "recomputed — guaranteeing identical values across all cases "
                          "that share the same mesh.")
_parser.add_argument("--d-sum-max-mm", type=float, default=22.0,
                     help="Envelope upper bound on d_sum (mm). Only used if "
                          "--geometry-fields is not provided.")
_parser.add_argument("--d-sum-min-mm", type=float, default=4.0,
                     help="Envelope lower bound on d_sum (mm).")
_parser.add_argument("--d-epi-min-mm", type=float, default=2.0,
                     help="Envelope minimum d_epi (mm).")
_args = _parser.parse_args()

results_dir = _args.results_dir.resolve()
comm = MPI.COMM_WORLD
rank = comm.rank

logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
logger = logging.getLogger("per_cell")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: Problem Setup (mirrors postprocess_metrics.py)
# ════════════════════════════════════════════════════════════════════════════

# ── Load Parameters ──────────────────────────────────────────────────────────
sim_params_path = results_dir / "simulation_params.json"
with open(sim_params_path) as f:
    sim_params = json.load(f)
if rank == 0:
    logger.info(f"Loaded sim params: BPM={sim_params['BPM']}, dt={sim_params['dt']}")

solver_dir = results_dir / "solver"
checkpoint_path = solver_dir / "checkpoint.bp"

# ── Load Ta History ──────────────────────────────────────────────────────────
Ta_history = np.load(solver_dir / "Ta_solver_history.npy")
if rank == 0:
    logger.info(f"Ta history: {Ta_history.shape}")

# ── Load Circulation History ─────────────────────────────────────────────────
circ_path = results_dir / "circulation" / "history.npy"
circ_history = np.load(circ_path, allow_pickle=True).item()

circ_time_arr = np.array(circ_history["time"])
HR_Hz = sim_params["BPM"] / 60.0
cycle_length = 1.0 / HR_Hz
circ_duration = circ_time_arr[-1] - circ_time_arr[0]

if circ_duration > cycle_length * 1.5:
    last_beat_start = circ_time_arr[-1] - cycle_length
    mask = circ_time_arr >= last_beat_start - 1e-10
    for key in circ_history:
        arr = np.array(circ_history[key])
        if len(arr) == len(circ_time_arr):
            circ_history[key] = arr[mask]
    circ_history["time"] = np.array(circ_history["time"]) - last_beat_start

# ── Load Mesh from Checkpoint ────────────────────────────────────────────────
mesh = adios4dolfinx.read_mesh(checkpoint_path, comm)
ffun = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="ffun")
markers_mt = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="cfun")

if rank == 0:
    logger.info(f"Mesh: {mesh.geometry.x.shape[0]} nodes")

# Optional geometric septum retagging
if _args.retag_septum:
    mesh.topology.create_connectivity(2, 0)
    f2v = mesh.topology.connectivity(2, 0)
    imap = mesh.topology.index_map(3)
    n_local = imap.size_local

    geo_dir = results_dir / "geometry"
    if rank == 0:
        geo_tmp = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, geo_dir)
        raw_markers = geo_tmp.markers
    else:
        raw_markers = None
    raw_markers = comm.bcast(raw_markers, root=0)

    def surface_coords_global(tag_ids):
        facets = np.hstack([ffun.find(t) for t in tag_ids])
        verts = set()
        for f in facets:
            verts.update(f2v.links(f))
        local_coords = mesh.geometry.x[np.array(sorted(verts), dtype=np.int64)] if verts else np.empty((0, 3))
        all_coords = comm.allgather(local_coords)
        return np.vstack(all_coords) if any(len(c) > 0 for c in all_coords) else np.empty((0, 3))

    lv_tags = [raw_markers.get("LV", raw_markers.get("ENDO_LV", [None]))[0]]
    rv_tags = [raw_markers.get("RV", raw_markers.get("ENDO_RV", [None]))[0]]
    epi_tags = [raw_markers.get("EPI", [None])[0]]

    lv_coords = surface_coords_global(lv_tags)
    rv_coords = surface_coords_global(rv_tags)
    epi_coords = surface_coords_global(epi_tags)

    centroids = dolfinx.mesh.compute_midpoints(mesh, 3, np.arange(n_local, dtype=np.int32))
    d_lv_arr = cKDTree(lv_coords).query(centroids)[0]
    d_rv_arr = cKDTree(rv_coords).query(centroids)[0]
    d_epi_arr = cKDTree(epi_coords).query(centroids)[0]

    is_sept = np.maximum(d_lv_arr, d_rv_arr) < d_epi_arr
    new_tags = np.where(is_sept, 3, np.where(d_lv_arr <= d_rv_arr, 1, 2)).astype(np.int32)
    markers_mt = dolfinx.mesh.meshtags(
        mesh, 3,
        np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32),
        np.concatenate([new_tags, markers_mt.values[n_local:]]),
    )
    markers_mt.name = "cfun"
    if rank == 0:
        logger.info(f"Retagged: LV={int((new_tags==1).sum())}, RV={int((new_tags==2).sum())}, Sept={int((new_tags==3).sum())}")

# ── Load Geometry ────────────────────────────────────────────────────────────
geo_dir = results_dir / "geometry"
geo = cardiac_geometries.geometry.Geometry.from_folder(comm, geo_dir)
geometry = pulse.HeartGeometry(
    mesh=mesh, facet_tags=ffun, markers=geo.markers,
    metadata={"quadrature_degree": 6},
)

# ── Fiber Fields ─────────────────────────────────────────────────────────────
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
q_el = basix.ufl.quadrature_element(mesh.topology.cell_name(), value_shape=(3,), degree=6)
Q_vec = dolfinx.fem.functionspace(mesh, q_el)

f0_quad = dolfinx.fem.Function(Q_vec)
s0_quad = dolfinx.fem.Function(Q_vec)
adios4dolfinx.read_function(checkpoint_path, f0_quad, time=0.0, name="f0")
adios4dolfinx.read_function(checkpoint_path, s0_quad, time=0.0, name="s0")

n0_quad = None
try:
    n0_quad = dolfinx.fem.Function(Q_vec)
    adios4dolfinx.read_function(checkpoint_path, n0_quad, time=0.0, name="n0")
except Exception:
    pass

l0_field = None
try:
    l0_field = dolfinx.fem.Function(Q_vec)
    adios4dolfinx.read_function(checkpoint_path, l0_field, time=0.0, name="l0")
except Exception:
    pass

if rank == 0:
    logger.info(f"Fibers loaded: f0, s0, n0={'yes' if n0_quad else 'no'}, l0={'yes' if l0_field else 'no'}")

# ── Reconstruct Problem ──────────────────────────────────────────────────────
mat_params_raw = sim_params["material_params"]
material_params = {}
for k, entry in mat_params_raw.items():
    material_params[k] = pulse.Variable(entry["value"], entry["unit"])

material = pulse.HolzapfelOgden(f0=f0_quad, s0=s0_quad, **material_params)
Ta_space = scifem.create_space_of_simple_functions(mesh=mesh, cell_tag=markers_mt, tags=[1, 2, 3])
Ta = pulse.Variable(dolfinx.fem.Function(Ta_space), "kPa")
active_model = pulse.ActiveStress(f0_quad, activation=Ta)

if sim_params["incompressible"]:
    comp_model = pulse.compressibility.Incompressible()
else:
    comp_model = pulse.compressibility.Compressible2()

cardiac_model = pulse.CardiacModel(material=material, active=active_model, compressibility=comp_model)

alpha_epi = sim_params["alpha_epi"]
alpha_base = sim_params["alpha_base"]

robin_epi = pulse.RobinBC(
    value=pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha_epi)), "Pa / m"),
    marker=geometry.markers["EPI"][0],
)
robin_base = pulse.RobinBC(
    value=pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha_base)), "Pa / m"),
    marker=geometry.markers["BASE"][0],
)

def dirichlet_bc(V_space):
    facets = geometry.facet_tags.find(geometry.markers["BASE"][0])
    if sim_params["incompressible"]:
        dofs = dolfinx.fem.locate_dofs_topological(V_space, 2, facets)
        u_zero = dolfinx.fem.Function(V_space)
        return [dolfinx.fem.dirichletbc(u_zero, dofs)]
    else:
        V_x = V_space.sub(0)
        dofs = dolfinx.fem.locate_dofs_topological(V_x, 2, facets)
        return [dolfinx.fem.dirichletbc(0.0, dofs, V_x)]

volume2ml = sim_params["volume2ml"]
lv_marker = "LV" if "LV" in geometry.markers else "ENDO_LV"
rv_marker = "RV" if "RV" in geometry.markers else "ENDO_RV"

lv_volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(sim_params["lvv_unloaded_m3"]))
rv_volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(sim_params["rvv_unloaded_m3"]))

cavities = [
    pulse.problem.Cavity(marker=lv_marker, volume=lv_volume),
    pulse.problem.Cavity(marker=rv_marker, volume=rv_volume),
]
bcs = pulse.BoundaryConditions(robin=[robin_epi, robin_base], dirichlet=(dirichlet_bc,))
problem = pulse.problem.StaticProblem(
    model=cardiac_model, geometry=geometry, bcs=bcs, cavities=cavities,
    parameters={"mesh_unit": sim_params["mesh_unit"], "u_space": "P_2"},
)

if rank == 0:
    logger.info("Problem reconstructed")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: Per-Cell DG0 Forms (the new part)
# ════════════════════════════════════════════════════════════════════════════

u = problem.u
I = ufl.Identity(3)
F = ufl.variable(ufl.grad(u) + I)
C = ufl.variable(F.T * F)
E_ufl = 0.5 * (C - I)

S_tot_ufl = cardiac_model.S(C)
S_act_ufl = cardiac_model.active.S(C, dev=True)
S_pas_ufl = cardiac_model.material.S(C, dev=True)
S_cmp_ufl = cardiac_model.compressibility.S(C)

f0 = f0_quad
s0 = s0_quad
n0 = n0_quad
l0 = l0_field
if l0 is not None:
    l0_norm = ufl.sqrt(ufl.inner(l0, l0))
    l0 = l0 / l0_norm

def proj(T, v):
    return ufl.inner(ufl.dot(T, v), v)

# Quadrature-space functions for state tracking (same as MetricsCalculator)
q_el_tensor = basix.ufl.quadrature_element(mesh.topology.cell_name(), value_shape=(3, 3), degree=6)
W_tensor = dolfinx.fem.functionspace(mesh, q_el_tensor)
q_el_scalar = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=6)
W_scalar = dolfinx.fem.functionspace(mesh, q_el_scalar)

E_cur = dolfinx.fem.Function(W_tensor, name="E_cur")
E_prev = dolfinx.fem.Function(W_tensor, name="E_prev")
S_prev = dolfinx.fem.Function(W_tensor, name="S_prev")

# Expressions for interpolation
points_tensor = W_tensor.element.interpolation_points
expr_E = dolfinx.fem.Expression(E_ufl, points_tensor)
expr_S_total = dolfinx.fem.Expression(S_tot_ufl, points_tensor)

# DG0 function space for per-cell extraction
V_DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
v_dg0 = ufl.TestFunction(V_DG0)

# dx with same quadrature as the tensor space
dx_q = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})

# Work density expressions (trapezoidal: 0.5 * (S_new + S_old) : dE)
dE = E_cur - E_prev

wd_total = 0.5 * ufl.inner(S_tot_ufl + S_prev, dE)
wd_fiber = 0.5 * (proj(S_tot_ufl, f0) + proj(S_prev, f0)) * proj(dE, f0)
wd_sheet = 0.5 * (proj(S_tot_ufl, s0) + proj(S_prev, s0)) * proj(dE, s0)

wd_normal = None
if n0 is not None:
    wd_normal = 0.5 * (proj(S_tot_ufl, n0) + proj(S_prev, n0)) * proj(dE, n0)

# Strain increment projections (for proxy work: P * d_eps_ff per cell)
deps_ff = proj(dE, f0)  # scalar: fiber strain increment
deps_ll = None
if l0 is not None:
    deps_ll = proj(dE, l0)

# Compile per-cell forms
# Each form: assemble_vector gives a vector of length n_cells
form_w_total = dolfinx.fem.form(wd_total * v_dg0 * dx_q)
form_w_ff = dolfinx.fem.form(wd_fiber * v_dg0 * dx_q)
form_w_ss = dolfinx.fem.form(wd_sheet * v_dg0 * dx_q)
form_w_nn = dolfinx.fem.form(wd_normal * v_dg0 * dx_q) if wd_normal is not None else None
form_deps_ff = dolfinx.fem.form(deps_ff * v_dg0 * dx_q)
form_deps_ll = dolfinx.fem.form(deps_ll * v_dg0 * dx_q) if deps_ll is not None else None

# Cell volume form (assemble once)
form_vol = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * v_dg0 * dx_q)

if rank == 0:
    logger.info("Per-cell DG0 forms compiled")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: Tau + Study Region
# ════════════════════════════════════════════════════════════════════════════

mesh.topology.create_connectivity(2, 0)
mesh.topology.create_connectivity(3, 0)
f2v_conn = mesh.topology.connectivity(2, 0)
imap_3 = mesh.topology.index_map(3)
n_local_cells = imap_3.size_local


local_cells = np.arange(n_local_cells, dtype=np.int32)

# Surface coordinates (gathered across MPI ranks)
geo_dir_path = results_dir / "geometry"
if rank == 0:
    geo_for_markers = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, geo_dir_path)
    raw_markers_dict = geo_for_markers.markers
else:
    raw_markers_dict = None
raw_markers_dict = comm.bcast(raw_markers_dict, root=0)

def _surface_coords_global(tag_ids):
    facets = np.hstack([ffun.find(t) for t in tag_ids])
    verts = set()
    for f_idx in facets:
        verts.update(f2v_conn.links(f_idx))
    local_coords = mesh.geometry.x[np.array(sorted(verts), dtype=np.int64)] if verts else np.empty((0, 3))
    all_coords = comm.allgather(local_coords)
    return np.vstack(all_coords) if any(len(c) > 0 for c in all_coords) else np.empty((0, 3))

_lv_tags = [raw_markers_dict.get("LV", raw_markers_dict.get("ENDO_LV", [None]))[0]]
_rv_tags = [raw_markers_dict.get("RV", raw_markers_dict.get("ENDO_RV", [None]))[0]]
_epi_tags = [raw_markers_dict.get("EPI", [None])[0]]

lv_surf = _surface_coords_global(_lv_tags)
rv_surf = _surface_coords_global(_rv_tags)
epi_surf = _surface_coords_global(_epi_tags)

centroids = dolfinx.mesh.compute_midpoints(mesh, 3, local_cells)

# Per-cell distances. Prefer point-to-facet distance (via pyvista/VTK) which
# is mesh-resolution-independent. Fall back to nearest-vertex (cKDTree) if
# pyvista is unavailable. On coarse meshes (char_length ~ 10 mm) the two
# methods can disagree by several mm near surfaces, and facet distance is the
# correct answer — see verify_sweep_envelope.py for the design discussion.
def _build_surface_polydata_global(tag_ids):
    """Build a GLOBAL PolyData by gathering surface triangles across all MPI ranks.

    Each rank extracts its local facets for the given tag IDs, then all ranks
    allgather the triangle vertex coordinates. The resulting PolyData contains
    the COMPLETE surface, so compute_implicit_distance gives identical results
    on every rank regardless of how the mesh is partitioned.
    """
    # Local: extract triangles and their vertex coordinates
    facets = np.hstack([ffun.find(t) for t in tag_ids])
    local_triangles = []  # list of (3, 3) arrays: 3 vertex coords per triangle
    for f_idx in facets:
        vert_ids = f2v_conn.links(f_idx)
        if len(vert_ids) != 3:
            continue
        local_triangles.append(mesh.geometry.x[vert_ids])  # shape (3, 3)

    if local_triangles:
        local_tri_array = np.array(local_triangles)  # (n_local_tri, 3, 3)
    else:
        local_tri_array = np.empty((0, 3, 3))

    # Global: gather all triangles to all ranks
    all_tri_arrays = comm.allgather(local_tri_array)
    global_tris = np.concatenate([a for a in all_tri_arrays if len(a) > 0])

    if len(global_tris) == 0:
        return None

    # Build PolyData from the global triangle set
    n_tri = len(global_tris)
    points = global_tris.reshape(-1, 3)  # (n_tri * 3, 3)
    faces_flat = np.zeros(n_tri * 4, dtype=np.int64)
    for i in range(n_tri):
        faces_flat[i * 4] = 3
        faces_flat[i * 4 + 1] = i * 3
        faces_flat[i * 4 + 2] = i * 3 + 1
        faces_flat[i * 4 + 3] = i * 3 + 2

    return pv.PolyData(points, faces=faces_flat)

if _HAS_PYVISTA:
    if rank == 0:
        logger.info("Computing d_lv, d_rv, d_epi via point-to-facet distance "
                     "(pyvista, GLOBAL surface)")
    lv_poly = _build_surface_polydata_global(_lv_tags)
    rv_poly = _build_surface_polydata_global(_rv_tags)
    epi_poly = _build_surface_polydata_global(_epi_tags)
    if lv_poly is None or rv_poly is None or epi_poly is None:
        if rank == 0:
            logger.warning("No facets found for one of LV/RV/EPI — "
                           "falling back to nearest-vertex.")
        d_lv = cKDTree(lv_surf).query(centroids)[0] if len(lv_surf) else np.full(len(centroids), np.inf)
        d_rv = cKDTree(rv_surf).query(centroids)[0] if len(rv_surf) else np.full(len(centroids), np.inf)
        d_epi = cKDTree(epi_surf).query(centroids)[0] if len(epi_surf) else np.full(len(centroids), np.inf)
    else:
        if rank == 0:
            logger.info(f"  Global surfaces: LV={lv_poly.n_faces_strict} tri, "
                         f"RV={rv_poly.n_faces_strict} tri, EPI={epi_poly.n_faces_strict} tri")
        centroids_poly = pv.PolyData(centroids.astype(np.float64))
        d_lv = np.abs(centroids_poly.compute_implicit_distance(lv_poly)["implicit_distance"])
        d_rv = np.abs(centroids_poly.compute_implicit_distance(rv_poly)["implicit_distance"])
        d_epi = np.abs(centroids_poly.compute_implicit_distance(epi_poly)["implicit_distance"])
else:
    if rank == 0:
        logger.warning("pyvista not available — falling back to nearest-vertex distance. "
                       "Envelope d_epi threshold may not reliably exclude surface cells.")
    d_lv = cKDTree(lv_surf).query(centroids)[0]
    d_rv = cKDTree(rv_surf).query(centroids)[0]
    d_epi = cKDTree(epi_surf).query(centroids)[0]

d_sum = d_lv + d_rv

# Euclidean tau
tau = d_lv / (d_lv + d_rv)

# Geometric septum (lower bound)
is_geometric_septum = np.maximum(d_lv, d_rv) < d_epi

# LDRB scalar-field septum (upper bound) — solve 2 Laplace equations
if rank == 0:
    logger.info("Solving Laplace equations for LDRB septum definition...")

V_CG1 = dolfinx.fem.functionspace(mesh, ("CG", 1))
u_trial = ufl.TrialFunction(V_CG1)
v_test = ufl.TestFunction(V_CG1)
a_laplace = ufl.dot(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
L_zero = dolfinx.fem.Constant(mesh, 0.0) * v_test * ufl.dx

# lv_rv_scalar: u=1 on LV, u=0 on RV
lv_facets = ffun.find(_lv_tags[0])
rv_facets = ffun.find(_rv_tags[0])
epi_facets = ffun.find(_epi_tags[0])

lv_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, 2, lv_facets)
rv_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, 2, rv_facets)
epi_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, 2, epi_facets)

prob_lvrv = dolfinx.fem.petsc.LinearProblem(
    a_laplace, L_zero,
    bcs=[dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), lv_dofs, V_CG1),
         dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_dofs, V_CG1)],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="percell_lvrv",
)
lv_rv_scalar = prob_lvrv.solve()

# epi_scalar: u=1 on EPI, u=0 on LV+RV
prob_epi = dolfinx.fem.petsc.LinearProblem(
    a_laplace, L_zero,
    bcs=[dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), epi_dofs, V_CG1),
         dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), lv_dofs, V_CG1),
         dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_dofs, V_CG1)],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="percell_epi",
)
epi_scalar = prob_epi.solve()

# Interpolate CG1 → DG0 for per-cell values
lvrv_dg0 = dolfinx.fem.Function(V_DG0)
lvrv_dg0.interpolate(lv_rv_scalar)
lvrv_vals = lvrv_dg0.x.array[:n_local_cells].copy()

epi_dg0 = dolfinx.fem.Function(V_DG0)
epi_dg0.interpolate(epi_scalar)
epi_vals = epi_dg0.x.array[:n_local_cells].copy()

# LDRB septum: epi_scalar <= 0.5 AND 0.1 < lv_rv_scalar < 0.9
is_ldrb_septum = (epi_vals <= 0.5) & (lvrv_vals > 0.1) & (lvrv_vals < 0.9)

# Auto-detect mesh units for envelope thresholds. The mesh might be in mm
# (UKB synthetic) or m (patient-specific). Detect via d_sum magnitude.
all_d_sums = d_sum[is_geometric_septum | is_ldrb_septum]
if len(all_d_sums) > 0 and all_d_sums.max() < 0.1:
    mesh_scale_to_mm = 1000.0  # mesh is in m
else:
    mesh_scale_to_mm = 1.0     # mesh is in mm

d_sum_max = _args.d_sum_max_mm / mesh_scale_to_mm
d_sum_min = _args.d_sum_min_mm / mesh_scale_to_mm
d_epi_min = _args.d_epi_min_mm / mesh_scale_to_mm

# Study region = union of geometric and LDRB, with a generous d_sum safety cut
study_region = (is_geometric_septum | is_ldrb_septum) & (d_sum < d_sum_max)

# ── Topological epi exclusion ────────────────────────────────────────────────
# A cell that has any face in the epi facet set is "touching the epi". Those
# cells must never be in the envelope (the sweep would pull them in as the
# threshold widens, which is visually and anatomically wrong). This uses mesh
# topology so it's exact — no distance threshold tuning needed.
mesh.topology.create_connectivity(3, 2)
c2f = mesh.topology.connectivity(3, 2)
epi_facets_local_set = set(ffun.find(_epi_tags[0]).tolist())
touches_epi = np.zeros(n_local_cells, dtype=bool)
for cell_i in range(n_local_cells):
    for facet in c2f.links(cell_i):
        if facet in epi_facets_local_set:
            touches_epi[cell_i] = True
            break

# ── Sweep scalar: entry_t (t at which a cell first joins the sweep set) ──────
#   cell ∈ septum(t)  iff  entry_t(cell) < t
#   t = 0: exactly the geometric septum (by construction, entry_t < 0 ⇔ geometric)
#   t > 0: widens outward (cells with d_epi close to max(d_lv,d_rv))
#   t < 0: tightens inward (only the deepest septum cells)
#
# Measured in mesh units (same as d_lv etc.). Users convert to mm if needed.
entry_t = np.maximum(d_lv, d_rv) - d_epi

# ── Envelope: anatomical bound the sweep must stay within ───────────────────
# Combines distance thresholds + topological epi exclusion.
envelope = ((d_epi >= d_epi_min)
            & (d_sum >= d_sum_min)
            & (d_sum <= d_sum_max)
            & ~touches_epi)

# Region tags from marker
region_tags = markers_mt.values[:n_local_cells]

if rank == 0:
    n_geo = comm.allreduce(int(is_geometric_septum.sum()))
    n_ldrb = comm.allreduce(int(is_ldrb_septum.sum()))
    n_study = comm.allreduce(int(study_region.sum()))
    logger.info(f"Study region: geometric={n_geo}, LDRB={n_ldrb}, union={n_study}")
    logger.info(f"Tau range in study region: [{tau[study_region].min():.3f}, {tau[study_region].max():.3f}]")
else:
    comm.allreduce(int(is_geometric_septum.sum()))
    comm.allreduce(int(is_ldrb_septum.sum()))
    comm.allreduce(int(study_region.sum()))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: Replay Last Beat + Accumulate Per-Cell Work
# ════════════════════════════════════════════════════════════════════════════

timestamps = adios4dolfinx.read_timestamps(checkpoint_path, comm, "displacement")
n_steps = len(timestamps)
steps_per_beat = int(round(cycle_length / sim_params["dt"]))
start_step = max(0, n_steps - steps_per_beat)

if rank == 0:
    logger.info(f"Replaying last beat: steps {start_step}–{n_steps-1} "
                f"({n_steps - start_step} steps, cycle={cycle_length:.3f}s)")

# Pressure interpolation from circulation history
circ_time = np.array(circ_history["time"])

# Conversion factor: 1 mmHg = 133.322 Pa
# Circulation pressures are in mmHg; we convert to Pa so the proxy work
# (P × dE × vol) is in the same units as the true work (S × dE × vol) which is in Pa.
MMHG_TO_PA = 133.322

def get_pressure_at_time(t):
    """Get LV and RV pressure (Pa) at time t by interpolation."""
    # Align checkpoint time to circulation time
    t_circ = t - timestamps[start_step]
    if t_circ < circ_time[0]:
        t_circ = circ_time[0]
    if t_circ > circ_time[-1]:
        t_circ = circ_time[-1]
    p_lv_mmHg = np.interp(t_circ, circ_time, circ_history["p_LV"])
    p_rv_mmHg = np.interp(t_circ, circ_time, circ_history["p_RV"])
    return p_lv_mmHg * MMHG_TO_PA, p_rv_mmHg * MMHG_TO_PA

# Accumulation arrays (local cells only)
n = n_local_cells
cum_w_total = np.zeros(n)
cum_w_ff = np.zeros(n)
cum_w_ss = np.zeros(n)
cum_w_nn = np.zeros(n)
# Proxy: accumulate P * deps_ff per cell (pressure in Pa, strain dimensionless,
# volume in m³ → units of Pa·m³ = J, same as w_true)
cum_proxy_PLV_ff = np.zeros(n)
cum_proxy_PRV_ff = np.zeros(n)
cum_proxy_Trans_ff = np.zeros(n)
cum_proxy_PLV_ll = np.zeros(n)
cum_proxy_PRV_ll = np.zeros(n)
cum_proxy_Trans_ll = np.zeros(n)

# Cell volumes (reuse the form_vol compiled above)
cell_vols_vec = dolfinx.fem.assemble_vector(form_vol)
cell_volumes = cell_vols_vec.array[:n].copy()

has_previous = False
p_LV_prev = 0.0
p_RV_prev = 0.0

for i in range(start_step, n_steps):
    t = timestamps[i]

    # Load displacement
    adios4dolfinx.read_function(checkpoint_path, problem.u, time=t, name="displacement")

    # Set active tension
    Ta.assign(Ta_history[i])
    cardiac_model.active.activation.value.x.array[:] = Ta.value.x.array[:]

    # Interpolate state variables
    E_cur.interpolate(expr_E)
    # S fields are evaluated symbolically through S_tot_ufl in the forms

    if has_previous:
        # Assemble per-cell work increments
        w_total_vec = dolfinx.fem.assemble_vector(form_w_total)
        w_ff_vec = dolfinx.fem.assemble_vector(form_w_ff)
        w_ss_vec = dolfinx.fem.assemble_vector(form_w_ss)

        cum_w_total += w_total_vec.array[:n]
        cum_w_ff += w_ff_vec.array[:n]
        cum_w_ss += w_ss_vec.array[:n]

        if form_w_nn is not None:
            w_nn_vec = dolfinx.fem.assemble_vector(form_w_nn)
            cum_w_nn += w_nn_vec.array[:n]

        # Proxy work: P_avg * deps_ff per cell
        p_LV, p_RV = get_pressure_at_time(t)
        p_LV_avg = 0.5 * (p_LV + p_LV_prev)
        p_RV_avg = 0.5 * (p_RV + p_RV_prev)

        deps_ff_vec = dolfinx.fem.assemble_vector(form_deps_ff)
        deps_ff_arr = deps_ff_vec.array[:n]

        cum_proxy_PLV_ff += p_LV_avg * deps_ff_arr
        cum_proxy_PRV_ff += p_RV_avg * deps_ff_arr
        cum_proxy_Trans_ff += (p_LV_avg - p_RV_avg) * deps_ff_arr

        if form_deps_ll is not None:
            deps_ll_vec = dolfinx.fem.assemble_vector(form_deps_ll)
            deps_ll_arr = deps_ll_vec.array[:n]
            cum_proxy_PLV_ll += p_LV_avg * deps_ll_arr
            cum_proxy_PRV_ll += p_RV_avg * deps_ll_arr
            cum_proxy_Trans_ll += (p_LV_avg - p_RV_avg) * deps_ll_arr

        p_LV_prev, p_RV_prev = p_LV, p_RV
    else:
        p_LV_prev, p_RV_prev = get_pressure_at_time(t)

    # Shift state: current → previous (for next step's trapezoidal average)
    E_prev.x.array[:] = E_cur.x.array[:]
    S_prev.interpolate(expr_S_total)
    has_previous = True

    if rank == 0 and (i % 50 == 0 or i == n_steps - 1):
        logger.info(f"  Step {i:04d}/{n_steps} | t={t:.4f}s | cum_W_total={cum_w_total.sum():.4e}")

# Cross-fiber = total - (ff + ss + nn); computed once after accumulation.
# If n0 is missing, cum_w_nn stays zero, so cross absorbs the nn contribution
# (flagged via the warning below).
cum_w_cross = cum_w_total - (cum_w_ff + cum_w_ss + cum_w_nn)
if form_w_nn is None and rank == 0:
    logger.warning("n0 not loaded — cum_w_cross includes the sheet-normal component "
                   "in addition to true cross-fiber work.")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: Validation
# ════════════════════════════════════════════════════════════════════════════

if rank == 0:
    logger.info("\n=== VALIDATION ===")

# Load regional metrics for comparison
metrics_dir = results_dir / "metrics"
metrics_file = metrics_dir / "metrics_downsample_1.npy"
if metrics_file.exists():
    regional = np.load(metrics_file, allow_pickle=True).item()

    # Sum per-cell work over last beat and compare to regional totals
    # Regional metrics store per-timestep increments for ALL beats.
    # We only computed per-cell for the LAST beat, so sum only the last beat's
    # worth of regional increments.
    n_regional_steps = len(regional.get("work_true_Whole", [0]))
    # steps_per_beat was computed earlier from cycle_length / dt
    last_beat_regional = max(0, n_regional_steps - steps_per_beat)
    work_true_LV_regional = np.sum(regional.get("work_true_LV", [0])[last_beat_regional:])
    work_true_RV_regional = np.sum(regional.get("work_true_RV", [0])[last_beat_regional:])
    work_true_Sept_regional = np.sum(regional.get("work_true_Septum", [0])[last_beat_regional:])
    work_true_Whole_regional = np.sum(regional.get("work_true_Whole", [0])[last_beat_regional:])

    # Sum per-cell by region tag
    lv_mask = region_tags == 1
    rv_mask = region_tags == 2
    sept_mask = region_tags == 3

    w_lv_percell = comm.allreduce(cum_w_total[lv_mask].sum())
    w_rv_percell = comm.allreduce(cum_w_total[rv_mask].sum())
    w_sept_percell = comm.allreduce(cum_w_total[sept_mask].sum())
    w_whole_percell = comm.allreduce(cum_w_total.sum())

    if rank == 0:
        logger.info(f"  Per-cell sum:  LV={w_lv_percell:.6e}  RV={w_rv_percell:.6e}  "
                     f"Sept={w_sept_percell:.6e}  Whole={w_whole_percell:.6e}")
        logger.info(f"  Regional sum:  LV={work_true_LV_regional:.6e}  RV={work_true_RV_regional:.6e}  "
                     f"Sept={work_true_Sept_regional:.6e}  Whole={work_true_Whole_regional:.6e}")

        for name, pc, reg in [("LV", w_lv_percell, work_true_LV_regional),
                               ("RV", w_rv_percell, work_true_RV_regional),
                               ("Septum", w_sept_percell, work_true_Sept_regional),
                               ("Whole", w_whole_percell, work_true_Whole_regional)]:
            if abs(reg) > 1e-15:
                rel_err = abs(pc - reg) / abs(reg) * 100
                status = "✓" if rel_err < 1.0 else "✗"
                logger.info(f"    {name}: {status} rel_error={rel_err:.4f}%")
            else:
                logger.info(f"    {name}: regional=0 (skip check)")
else:
    if rank == 0:
        logger.warning("No regional metrics found — skipping validation")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: Gather across MPI ranks and save
# ════════════════════════════════════════════════════════════════════════════
#
# Each rank holds a local slice of the mesh (n_local_cells cells). To write a
# complete per_cell_data.npz we must gather all local arrays to rank 0 and
# concatenate. The global ordering is arbitrary (rank0 cells, then rank1, ...)
# since downstream analysis indexes cells as a flat set, not by spatial order.

def gather_to_root(local_arr):
    """Gather a local per-cell array to rank 0 and concatenate."""
    gathered = comm.gather(np.ascontiguousarray(local_arr), root=0)
    if rank == 0:
        return np.concatenate(gathered)
    return None

# Gather all per-cell arrays
g_tau = gather_to_root(tau)
g_d_lv = gather_to_root(d_lv)
g_d_rv = gather_to_root(d_rv)
g_d_epi = gather_to_root(d_epi)
g_d_sum = gather_to_root(d_sum)
g_is_geometric_septum = gather_to_root(is_geometric_septum)
g_is_ldrb_septum = gather_to_root(is_ldrb_septum)
g_study_region = gather_to_root(study_region)
g_region_tags = gather_to_root(region_tags)
g_w_total = gather_to_root(cum_w_total)
g_w_ff = gather_to_root(cum_w_ff)
g_w_ss = gather_to_root(cum_w_ss)
g_w_nn = gather_to_root(cum_w_nn)
g_w_cross = gather_to_root(cum_w_cross)
g_proxy_PLV_ff = gather_to_root(cum_proxy_PLV_ff)
g_proxy_PRV_ff = gather_to_root(cum_proxy_PRV_ff)
g_proxy_Trans_ff = gather_to_root(cum_proxy_Trans_ff)
g_proxy_PLV_ll = gather_to_root(cum_proxy_PLV_ll)
g_proxy_PRV_ll = gather_to_root(cum_proxy_PRV_ll)
g_proxy_Trans_ll = gather_to_root(cum_proxy_Trans_ll)
g_cell_volumes = gather_to_root(cell_volumes)
g_centroids = gather_to_root(centroids)
g_lvrv_vals = gather_to_root(lvrv_vals)
g_epi_vals = gather_to_root(epi_vals)
# New: envelope and entry_t for the threshold-relaxation sweep
g_envelope = gather_to_root(envelope)
g_entry_t = gather_to_root(entry_t)
g_touches_epi = gather_to_root(touches_epi)

if rank == 0:
    output_path = results_dir / "per_cell_data.npz"
    np.savez(output_path,
             tau=g_tau,
             d_lv=g_d_lv,
             d_rv=g_d_rv,
             d_epi=g_d_epi,
             d_sum=g_d_sum,
             is_geometric_septum=g_is_geometric_septum,
             is_ldrb_septum=g_is_ldrb_septum,
             study_region=g_study_region,
             region_tags=g_region_tags,
             # Threshold-relaxation sweep fields (Question A)
             envelope=g_envelope,
             entry_t=g_entry_t,
             touches_epi=g_touches_epi,
             # Per-cell work
             w_total=g_w_total,
             w_ff=g_w_ff,
             w_ss=g_w_ss,
             w_nn=g_w_nn,
             w_cross=g_w_cross,
             proxy_PLV_ff=g_proxy_PLV_ff,
             proxy_PRV_ff=g_proxy_PRV_ff,
             proxy_Trans_ff=g_proxy_Trans_ff,
             proxy_PLV_ll=g_proxy_PLV_ll,
             proxy_PRV_ll=g_proxy_PRV_ll,
             proxy_Trans_ll=g_proxy_Trans_ll,
             cell_volumes=g_cell_volumes,
             centroids=g_centroids,
             lv_rv_scalar=g_lvrv_vals,
             epi_scalar_dg0=g_epi_vals,
             # Envelope parameters used (for provenance)
             envelope_d_epi_min_mm=_args.d_epi_min_mm,
             envelope_d_sum_min_mm=_args.d_sum_min_mm,
             envelope_d_sum_max_mm=_args.d_sum_max_mm,
    )
    logger.info(f"Saved per-cell data to {output_path}")
    logger.info(f"  Global cells: {len(g_tau)}")
    logger.info(f"  Envelope cells: {int(g_envelope.sum())}")
    logger.info(f"  Geometric septum: {int(g_is_geometric_septum.sum())}")
    logger.info(f"  entry_t range in envelope: [{g_entry_t[g_envelope].min():.4f}, "
                f"{g_entry_t[g_envelope].max():.4f}]")
    logger.info("Done.")
