#!/usr/bin/env python3
"""
precompute_geometry_fields.py — Compute tau, distances, envelope, entry_t ONCE
for a shared mesh and save to a reference file.

All downstream per-cell computations (compute_per_cell.py) load this reference
instead of recomputing distances, guaranteeing bit-identical geometric fields
across all cases that use the same mesh.

Run this ONCE after generating the shared mesh, BEFORE submitting any sims
or per-cell jobs.

Usage:
    python3 precompute_geometry_fields.py data/shared_ukb_mesh/ukb/geometry
"""
import argparse
from pathlib import Path
import numpy as np
import dolfinx
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
import pyvista as pv
import cardiac_geometries
import cardiac_geometries.geometry

parser = argparse.ArgumentParser()
parser.add_argument("geodir", type=Path, help="Geometry directory (with mesh.xdmf)")
parser.add_argument("--d-epi-min-mm", type=float, default=2.0)
parser.add_argument("--d-sum-min-mm", type=float, default=4.0)
parser.add_argument("--d-sum-max-mm", type=float, default=22.0)
args = parser.parse_args()

geodir = args.geodir.resolve()
print(f"Loading mesh from {geodir}/mesh.xdmf")

# ── Load mesh serially ───────────────────────────────────────────────────────
with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(geodir / "mesh.xdmf"), "r") as xdmf:
    mesh = xdmf.read_mesh(name="Mesh")
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(2, 0)
    mesh.topology.create_connectivity(3, 2)
    ffun = xdmf.read_meshtags(mesh, name="Facet tags")

LV_MARKER, RV_MARKER, EPI_MARKER = 1, 2, 3
f2v = mesh.topology.connectivity(2, 0)
c2f = mesh.topology.connectivity(3, 2)

cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
centroids = dolfinx.mesh.compute_midpoints(mesh, 3, cells)
n_cells = len(cells)
print(f"Cells: {n_cells}")

# ── Surface PolyData (for facet-distance) ────────────────────────────────────
def build_surface_poly(marker):
    facets = ffun.find(marker)
    tris = []
    for f_idx in facets:
        verts = f2v.links(f_idx)
        if len(verts) == 3:
            tris.append(mesh.geometry.x[verts])
    if not tris:
        return None
    tris = np.array(tris)
    n = len(tris)
    points = tris.reshape(-1, 3)
    faces = np.zeros(n * 4, dtype=np.int64)
    for i in range(n):
        faces[i*4] = 3
        faces[i*4+1:i*4+4] = [i*3, i*3+1, i*3+2]
    return pv.PolyData(points, faces=faces)

lv_poly = build_surface_poly(LV_MARKER)
rv_poly = build_surface_poly(RV_MARKER)
epi_poly = build_surface_poly(EPI_MARKER)
print(f"Surface triangles: LV={lv_poly.n_faces_strict}, RV={rv_poly.n_faces_strict}, EPI={epi_poly.n_faces_strict}")

# ── Distances (facet-based, deterministic serial computation) ────────────────
centroids_poly = pv.PolyData(centroids.astype(np.float64))
d_lv = np.abs(centroids_poly.compute_implicit_distance(lv_poly)["implicit_distance"])
d_rv = np.abs(centroids_poly.compute_implicit_distance(rv_poly)["implicit_distance"])
d_epi = np.abs(centroids_poly.compute_implicit_distance(epi_poly)["implicit_distance"])
d_sum = d_lv + d_rv

# ── Detect mesh units ────────────────────────────────────────────────────────
bbox = mesh.geometry.x.max(axis=0) - mesh.geometry.x.min(axis=0)
if bbox.max() > 10:
    mesh_to_mm = 1.0
    print("Mesh unit: mm")
else:
    mesh_to_mm = 1000.0
    print("Mesh unit: m")

d_epi_min = args.d_epi_min_mm / mesh_to_mm
d_sum_min = args.d_sum_min_mm / mesh_to_mm
d_sum_max = args.d_sum_max_mm / mesh_to_mm

# ── Tau ──────────────────────────────────────────────────────────────────────
tau = d_lv / (d_lv + d_rv)

# ── Geometric septum ─────────────────────────────────────────────────────────
is_geometric_septum = np.maximum(d_lv, d_rv) < d_epi

# ── LDRB septum (Laplace solves) ─────────────────────────────────────────────
print("Solving Laplace equations for LDRB definition...")
V_CG1 = dolfinx.fem.functionspace(mesh, ("CG", 1))
u_t = ufl.TrialFunction(V_CG1)
v_t = ufl.TestFunction(V_CG1)
a = ufl.dot(ufl.grad(u_t), ufl.grad(v_t)) * ufl.dx
L = dolfinx.fem.Constant(mesh, 0.0) * v_t * ufl.dx

lv_f = ffun.find(LV_MARKER)
rv_f = ffun.find(RV_MARKER)
epi_f = ffun.find(EPI_MARKER)
lv_d = dolfinx.fem.locate_dofs_topological(V_CG1, 2, lv_f)
rv_d = dolfinx.fem.locate_dofs_topological(V_CG1, 2, rv_f)
epi_d = dolfinx.fem.locate_dofs_topological(V_CG1, 2, epi_f)

lvrv = dolfinx.fem.petsc.LinearProblem(
    a, L,
    bcs=[dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), lv_d, V_CG1),
         dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_d, V_CG1)],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="geo_lvrv",
).solve()

epi_s = dolfinx.fem.petsc.LinearProblem(
    a, L,
    bcs=[dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), epi_d, V_CG1),
         dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), lv_d, V_CG1),
         dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_d, V_CG1)],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="geo_epi",
).solve()

V_DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
lvrv_dg0 = dolfinx.fem.Function(V_DG0)
lvrv_dg0.interpolate(lvrv)
lvrv_vals = lvrv_dg0.x.array[:n_cells].copy()

epi_dg0 = dolfinx.fem.Function(V_DG0)
epi_dg0.interpolate(epi_s)
epi_vals = epi_dg0.x.array[:n_cells].copy()

is_ldrb_septum = (epi_vals <= 0.5) & (lvrv_vals > 0.1) & (lvrv_vals < 0.9)

# ── Topological epi exclusion ────────────────────────────────────────────────
epi_facets_set = set(ffun.find(EPI_MARKER).tolist())
touches_epi = np.zeros(n_cells, dtype=bool)
for ci in range(n_cells):
    for fi in c2f.links(ci):
        if fi in epi_facets_set:
            touches_epi[ci] = True
            break

# ── Entry_t and envelope ─────────────────────────────────────────────────────
entry_t = np.maximum(d_lv, d_rv) - d_epi
envelope = (d_epi >= d_epi_min) & (d_sum >= d_sum_min) & (d_sum <= d_sum_max) & ~touches_epi
study_region = (is_geometric_septum | is_ldrb_septum) & (d_sum < d_sum_max)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = geodir / "geometry_fields.npz"
np.savez(out_path,
         centroids=centroids,
         tau=tau,
         d_lv=d_lv, d_rv=d_rv, d_epi=d_epi, d_sum=d_sum,
         is_geometric_septum=is_geometric_septum,
         is_ldrb_septum=is_ldrb_septum,
         study_region=study_region,
         envelope=envelope,
         entry_t=entry_t,
         touches_epi=touches_epi,
         lv_rv_scalar=lvrv_vals,
         epi_scalar_dg0=epi_vals,
         # Parameters used
         d_epi_min_mm=args.d_epi_min_mm,
         d_sum_min_mm=args.d_sum_min_mm,
         d_sum_max_mm=args.d_sum_max_mm,
)

print(f"\n=== REFERENCE GEOMETRY FIELDS ===")
print(f"  n_cells:    {n_cells}")
print(f"  geometric:  {int(is_geometric_septum.sum())}")
print(f"  ldrb:       {int(is_ldrb_septum.sum())}")
print(f"  envelope:   {int(envelope.sum())}")
print(f"  touches_epi:{int(touches_epi.sum())}")
print(f"  entry_t:    [{entry_t[envelope].min():.6f}, {entry_t[envelope].max():.6f}]")
print(f"\nSaved to {out_path}")
print(f"compute_per_cell.py will load this instead of recomputing distances.")
