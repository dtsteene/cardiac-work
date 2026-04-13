#!/usr/bin/env python3
"""
precompute_geometry_fields.py — Compute canonical septum tags ONCE on the
zero-strain atlas mesh, save as a reference file.

All downstream per-cell computations (compute_per_cell.py --geometry-fields)
load this reference and propagate it to each case via the u_pre permutation.
This guarantees anatomically-identical tagging across every case that uses
the same shared mesh, independent of case-specific prestress deformation.

INPUT: a geometry folder containing geometry.bp (the shared atlas mesh,
       BEFORE any prestress is applied — the "zero-strain canonical mesh").

Run this ONCE after generating the shared mesh, BEFORE submitting any sims.

Usage:
    python3 precompute_geometry_fields.py data/shared_ukb_mesh/ukb/geometry

The output geometry_fields.npz is saved next to the input geometry.bp.
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
import adios4dolfinx

parser = argparse.ArgumentParser()
parser.add_argument("geometry_dir", type=Path,
                    help="Geometry folder containing geometry.bp (the shared "
                         "atlas mesh, zero-strain). This is the same folder "
                         "that cardiac_geometries.Geometry.from_folder reads. "
                         "Tags are computed on this mesh and then propagated "
                         "to each sim's prestressed mesh via u_pre permutation "
                         "at per-cell computation time.")
parser.add_argument("--d-sum-max-mm", type=float, default=22.0,
                    help="Upper bound on d_lv + d_rv (mm) for the envelope. "
                         "Prevents the sweep from reaching free-wall cells.")
args = parser.parse_args()

geom_dir = args.geometry_dir.resolve()
ckpt_path = geom_dir / "geometry.bp"
if not ckpt_path.exists():
    raise FileNotFoundError(f"Expected {ckpt_path} — pass the geometry folder, not a checkpoint")

print(f"Loading canonical mesh from {geom_dir}")
print("  (zero-strain atlas, before any prestress)")

# ── Load mesh SERIALLY via cardiac_geometries ───────────────────────────────
# Use cardiac_geometries.Geometry.from_folder (same code path compute_per_cell.py
# uses for its markers/geometry object), so the cell ordering is guaranteed to
# match what compute_per_cell.py will see during the u_pre permutation step.
import cardiac_geometries
geo_loader = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, geom_dir)
mesh = geo_loader.mesh
ffun = geo_loader.ffun
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
mesh.topology.create_connectivity(2, 0)
mesh.topology.create_connectivity(3, 2)

# Marker values: checkpoint.bp uses the standard convention from complete_cycle.py
# (LV=1, RV=2, EPI=3, BASE=4). Verify by checking what's in ffun.
unique_markers = sorted(set(int(v) for v in ffun.values))
print(f"  Facet markers found: {unique_markers}")
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

# ── Entry_t and corrected envelope ───────────────────────────────────────────
# entry_t(c) = max(d_lv, d_rv) - d_epi is the sweep scalar: t=0 recovers the
# geometric septum (by construction), t<0 tightens the core, t>0 widens toward
# the epicardium.
entry_t = np.maximum(d_lv, d_rv) - d_epi

# Envelope: minimal definition. Only the bounds we need to prevent the sweep
# from spilling into free-wall territory. We intentionally drop d_epi_min and
# d_sum_min — they excluded 50-60% of legitimate apical septum cells and broke
# the sweep(t=0) == geometric invariant. See session update 2026-04-12 in
# results/docs/transmural_work_profiles.md.
envelope = (d_sum <= d_sum_max) & ~touches_epi

study_region = (is_geometric_septum | is_ldrb_septum) & (d_sum < d_sum_max)

# ── Save ─────────────────────────────────────────────────────────────────────
# Save next to the input geometry.bp
out_path = geom_dir / "geometry_fields.npz"
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
         # Envelope parameter (only one now)
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
