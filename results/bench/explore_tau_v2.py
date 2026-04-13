"""
Compare septum definitions and transventricular coordinates.

Produces XDMF files for ParaView showing:
1. Euclidean tau vs Laplace tau (lv_rv_scalar) side by side
2. Geometric septum (lower bound) vs LDRB scalar-field septum (upper bound)
3. The union study region with transition cells highlighted
4. Zone classification: core geometric, LDRB-only extension, free wall

Also solves the transventricular Laplace equation (u=1 on LV endo, u=0 on RV endo)
directly — this is the same PDE that LDRB's scalar_laplacians() solves for lv_rv_scalar.

Usage:
    python explore_tau_v2.py [geometry_dir]
    python explore_tau_v2.py /path/to/results/sims/some_run/geometry
"""

import argparse
from pathlib import Path

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "geodir",
    nargs="?",
    default="/home/dtsteene/D1/cardiac-work/results/sims/ukb_10beats_spectrum/UKB_10beats_healthy/geometry",
)
parser.add_argument("--nbins", type=int, default=15, help="Number of tau bins for profiles")
parser.add_argument("--d-sum-max-mm", type=float, default=25.0,
                    help="Safety cutoff: exclude cells with d_sum > this (mm)")
args = parser.parse_args()

geodir = Path(args.geodir)
print(f"Loading geometry from {geodir}")

# ── Load mesh and facet tags ─────────────────────────────────────────
with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(geodir / "mesh.xdmf"), "r") as xdmf:
    mesh = xdmf.read_mesh(name="Mesh")
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(2, 0)  # facets → vertices
    ffun = xdmf.read_meshtags(mesh, name="Facet tags")

# Marker values (from markers.json)
LV_MARKER = 1
RV_MARKER = 2
EPI_MARKER = 3

# ── Extract surface vertex coordinates ───────────────────────────────
f2v = mesh.topology.connectivity(2, 0)


def surface_coords(marker_value):
    facets = ffun.find(marker_value)
    vertices = set()
    for f in facets:
        vertices.update(f2v.links(f))
    verts = np.array(sorted(vertices), dtype=np.int64)
    return mesh.geometry.x[verts]


lv_coords = surface_coords(LV_MARKER)
rv_coords = surface_coords(RV_MARKER)
epi_coords = surface_coords(EPI_MARKER)
print(f"Surface vertices: LV={len(lv_coords)}, RV={len(rv_coords)}, EPI={len(epi_coords)}")

# ── Build KDTrees and compute Euclidean distances ────────────────────
tree_lv = cKDTree(lv_coords)
tree_rv = cKDTree(rv_coords)
tree_epi = cKDTree(epi_coords)

cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
centroids = dolfinx.mesh.compute_midpoints(mesh, 3, cells)

d_lv = tree_lv.query(centroids)[0]
d_rv = tree_rv.query(centroids)[0]
d_epi = tree_epi.query(centroids)[0]
d_sum = d_lv + d_rv

n_cells = len(cells)
print(f"Number of cells: {n_cells}")

# ── Euclidean tau (old approach) ─────────────────────────────────────
tau_euclidean = d_lv / (d_lv + d_rv)

# ── Solve transventricular Laplace equation (= LDRB's lv_rv_scalar) ──
# ∇²u = 0,  u(LV endo) = 1,  u(RV endo) = 0
# Zero Neumann on epi and base (natural BC)
print("\nSolving transventricular Laplace equation...")

V_CG1 = dolfinx.fem.functionspace(mesh, ("CG", 1))
u = ufl.TrialFunction(V_CG1)
v = ufl.TestFunction(V_CG1)

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = dolfinx.fem.Constant(mesh, 0.0) * v * ufl.dx

# Dirichlet BCs: u=1 on LV endo, u=0 on RV endo
lv_facets = ffun.find(LV_MARKER)
rv_facets = ffun.find(RV_MARKER)

lv_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, mesh.topology.dim - 1, lv_facets)
rv_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, mesh.topology.dim - 1, rv_facets)

bc_lv = dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), lv_dofs, V_CG1)
bc_rv = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_dofs, V_CG1)

problem = dolfinx.fem.petsc.LinearProblem(
    a, L, bcs=[bc_lv, bc_rv],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="tau_lv_rv",
)
lv_rv_scalar_fn = problem.solve()
lv_rv_scalar_fn.name = "lv_rv_scalar"
print(f"  lv_rv_scalar range: [{lv_rv_scalar_fn.x.array.min():.4f}, {lv_rv_scalar_fn.x.array.max():.4f}]")

# Also solve epi scalar (for LDRB-style septum definition)
# ∇²u = 0,  u(EPI) = 1,  u(LV endo) = 0,  u(RV endo) = 0
print("Solving epicardial Laplace equation...")
epi_facets = ffun.find(EPI_MARKER)
epi_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, mesh.topology.dim - 1, epi_facets)

bc_epi_1 = dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), epi_dofs, V_CG1)
bc_lv_0 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), lv_dofs, V_CG1)
bc_rv_0 = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_dofs, V_CG1)

problem_epi = dolfinx.fem.petsc.LinearProblem(
    a, L, bcs=[bc_epi_1, bc_lv_0, bc_rv_0],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="tau_epi",
)
epi_scalar_fn = problem_epi.solve()
epi_scalar_fn.name = "epi_scalar"
print(f"  epi_scalar range: [{epi_scalar_fn.x.array.min():.4f}, {epi_scalar_fn.x.array.max():.4f}]")

# ── Project CG1 scalars to DG0 (per-cell values) ────────────────────
V_DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))

# Interpolate CG1 → DG0 (evaluates CG1 at cell midpoint)
tau_laplace_fn = dolfinx.fem.Function(V_DG0, name="tau_laplace")
tau_laplace_fn.interpolate(lv_rv_scalar_fn)
tau_laplace = tau_laplace_fn.x.array[:n_cells].copy()

epi_scalar_dg0_fn = dolfinx.fem.Function(V_DG0, name="epi_scalar_dg0")
epi_scalar_dg0_fn.interpolate(epi_scalar_fn)
epi_scalar_dg0 = epi_scalar_dg0_fn.x.array[:n_cells].copy()

# ── Define septum regions ────────────────────────────────────────────
print("\n" + "=" * 70)
print("SEPTUM DEFINITIONS")
print("=" * 70)

# 1. Geometric septum (lower bound): max(d_LV, d_RV) < d_epi
is_geometric_septum = np.maximum(d_lv, d_rv) < d_epi
n_geo = np.sum(is_geometric_septum)
print(f"\nGeometric septum: {n_geo} cells ({100*n_geo/n_cells:.1f}%)")
print(f"  max(d_LV, d_RV) < d_epi")

# 2. LDRB scalar-field septum (upper bound): epi_scalar <= 0.5 AND 0.1 < lv_rv < 0.9
is_ldrb_septum = (epi_scalar_dg0 <= 0.5) & (tau_laplace > 0.1) & (tau_laplace < 0.9)
n_ldrb = np.sum(is_ldrb_septum)
print(f"\nLDRB scalar septum: {n_ldrb} cells ({100*n_ldrb/n_cells:.1f}%)")
print(f"  epi_scalar <= 0.5 AND 0.1 < lv_rv_scalar < 0.9")

# 3. Union (study region) with d_sum safety filter
# Determine mesh units from d_sum range
d_sum_sept = d_sum[is_geometric_septum | is_ldrb_septum]
if d_sum_sept.max() < 0.1:  # mesh in meters
    mesh_scale = 1000.0  # m → mm for display
    d_sum_max = args.d_sum_max_mm / 1000.0
    print(f"  Mesh appears to be in meters (d_sum range: {d_sum_sept.min()*1000:.1f}-{d_sum_sept.max()*1000:.1f}mm)")
else:  # mesh in mm
    mesh_scale = 1.0
    d_sum_max = args.d_sum_max_mm
    print(f"  Mesh appears to be in mm (d_sum range: {d_sum_sept.min():.1f}-{d_sum_sept.max():.1f}mm)")
is_union = (is_geometric_septum | is_ldrb_septum) & (d_sum < d_sum_max)
n_union = np.sum(is_union)
print(f"\nUnion (study region): {n_union} cells ({100*n_union/n_cells:.1f}%)")
print(f"  (geometric | LDRB) AND d_sum < {args.d_sum_max_mm}mm")

# 4. Cells only in LDRB but not geometric (the "extension")
is_ldrb_only = is_ldrb_septum & ~is_geometric_septum & (d_sum < d_sum_max)
n_ldrb_only = np.sum(is_ldrb_only)
print(f"\nLDRB-only extension: {n_ldrb_only} cells ({100*n_ldrb_only/n_cells:.1f}%)")

# 5. Cells only in geometric but not LDRB (if any — shouldn't be many)
is_geo_only = is_geometric_septum & ~is_ldrb_septum
n_geo_only = np.sum(is_geo_only)
print(f"Geometric-only (not in LDRB): {n_geo_only} cells")

# ── Zone classification ──────────────────────────────────────────────
# 0 = free wall (outside study region)
# 1 = geometric septum core (in both definitions)
# 2 = LDRB-only extension (in LDRB but not geometric)
# 3 = geometric-only (in geometric but not LDRB — should be rare)
zone = np.zeros(n_cells, dtype=np.float64)
zone[is_geometric_septum & is_ldrb_septum] = 1.0  # Core: both agree
zone[is_ldrb_only] = 2.0                           # Extension: LDRB adds
zone[is_geo_only] = 3.0                            # Rare: geometric adds

# Sub-classify free wall
is_fw = ~is_union
zone_detailed = zone.copy()
zone_detailed[is_fw & (d_lv <= d_rv)] = -1.0  # LV free wall
zone_detailed[is_fw & (d_lv > d_rv)] = -2.0   # RV free wall

# ── Tau statistics comparison ────────────────────────────────────────
print("\n" + "=" * 70)
print("TAU COMPARISON: EUCLIDEAN vs LAPLACE")
print("=" * 70)

for name, mask in [("Geometric septum", is_geometric_septum),
                   ("LDRB septum", is_ldrb_septum),
                   ("Union", is_union),
                   ("LDRB-only extension", is_ldrb_only)]:
    if np.sum(mask) == 0:
        continue
    te = tau_euclidean[mask]
    tl = tau_laplace[mask]
    print(f"\n{name} ({np.sum(mask)} cells):")
    print(f"  Euclidean tau: [{te.min():.3f}, {te.max():.3f}] mean={te.mean():.3f} median={np.median(te):.3f}")
    print(f"  Laplace tau:   [{tl.min():.3f}, {tl.max():.3f}] mean={tl.mean():.3f} median={np.median(tl):.3f}")
    diff = np.abs(te - tl)
    print(f"  |Euclidean - Laplace|: mean={diff.mean():.4f} max={diff.max():.4f}")

# ── Linearity check for both tau coordinates ─────────────────────────
print("\n" + "=" * 70)
print("LINEARITY CHECK (within study region)")
print("=" * 70)

study_mask = is_union
study_centroids = centroids[study_mask]

for tau_name, tau_vals in [("Euclidean", tau_euclidean[study_mask]),
                            ("Laplace", tau_laplace[study_mask])]:
    # Linearity check: for evenly-spaced tau values, measure what fraction
    # of the wall thickness each bin spans using d_sum as a proxy for
    # through-wall distance. In a linear coordinate, each bin should span
    # an equal fraction of d_sum.
    #
    # Method: for each cell, its "through-wall position" is approximated by
    # d_lv (distance from LV endo). In a perfectly linear coordinate,
    # the mean d_lv should increase proportionally with tau.
    n_iso = 10
    tau_min, tau_max = tau_vals.min(), tau_vals.max()
    bin_edges = np.linspace(tau_min, tau_max, n_iso + 1)

    mean_dlv_per_bin = []
    n_per_bin = []
    for i in range(n_iso):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (tau_vals >= lo) & (tau_vals < hi) if i < n_iso - 1 else (tau_vals >= lo)
        n = np.sum(mask)
        n_per_bin.append(n)
        if n > 0:
            mean_dlv_per_bin.append(d_lv[study_mask][mask].mean() * mesh_scale)
        else:
            mean_dlv_per_bin.append(np.nan)

    mean_dlv_per_bin = np.array(mean_dlv_per_bin)
    valid = ~np.isnan(mean_dlv_per_bin)

    if np.sum(valid) >= 3:
        # Compute incremental d_lv change between consecutive bins
        valid_vals = mean_dlv_per_bin[valid]
        increments = np.diff(valid_vals)
        # For a linear coordinate, increments should be roughly equal
        cv = np.std(np.abs(increments)) / np.mean(np.abs(increments)) if np.mean(np.abs(increments)) > 0 else 0
        print(f"\n{tau_name} tau ({n_iso} bins, {np.sum(study_mask)} cells):")
        print(f"  Mean d_LV per bin (mm): {', '.join(f'{d:.2f}' for d in mean_dlv_per_bin if not np.isnan(d))}")
        print(f"  Cells per bin: {', '.join(str(n) for n in n_per_bin)}")
        print(f"  Inter-bin d_LV increments (mm): {', '.join(f'{d:.2f}' for d in increments)}")
        print(f"  Increment CV={cv:.3f}", end="")
        if cv < 0.30:
            print(f"  ✓ Acceptable — bins span roughly equal wall thickness")
        else:
            print(f"  ✗ Non-uniform — bins span unequal wall thickness")

# ── Distance statistics for transition cells ─────────────────────────
print("\n" + "=" * 70)
print("TRANSITION CELL ANALYSIS")
print("=" * 70)

scale = mesh_scale  # to mm
for name, mask in [("Geometric core", is_geometric_septum & is_ldrb_septum),
                   ("LDRB-only extension", is_ldrb_only),
                   ("Geometric-only", is_geo_only)]:
    n = np.sum(mask)
    if n == 0:
        print(f"\n{name}: 0 cells")
        continue
    print(f"\n{name}: {n} cells")
    print(f"  d_LV (mm): [{d_lv[mask].min()*scale:.1f}, {d_lv[mask].max()*scale:.1f}]")
    print(f"  d_RV (mm): [{d_rv[mask].min()*scale:.1f}, {d_rv[mask].max()*scale:.1f}]")
    print(f"  d_epi (mm): [{d_epi[mask].min()*scale:.1f}, {d_epi[mask].max()*scale:.1f}]")
    print(f"  d_sum (mm): [{d_sum[mask].min()*scale:.1f}, {d_sum[mask].max()*scale:.1f}]")
    print(f"  epi_scalar: [{epi_scalar_dg0[mask].min():.3f}, {epi_scalar_dg0[mask].max():.3f}]")
    print(f"  lv_rv_scalar: [{tau_laplace[mask].min():.3f}, {tau_laplace[mask].max():.3f}]")

# ── Export everything to XDMF ────────────────────────────────────────
outdir = geodir.parent / "tau_exploration_v2"
outdir.mkdir(exist_ok=True)

# DG0 functions for visualization
def make_dg0(name, values):
    fn = dolfinx.fem.Function(V_DG0, name=name)
    fn.x.array[:n_cells] = values.astype(np.float64)
    return fn

fields = [
    # Tau coordinates (the key comparison)
    make_dg0("tau_euclidean", tau_euclidean),
    make_dg0("tau_laplace", tau_laplace),
    make_dg0("tau_difference", tau_laplace - tau_euclidean),
    # Laplace scalar fields
    epi_scalar_dg0_fn,
    # Septum definitions
    make_dg0("is_geometric_septum", is_geometric_septum.astype(np.float64)),
    make_dg0("is_ldrb_septum", is_ldrb_septum.astype(np.float64)),
    make_dg0("is_union", is_union.astype(np.float64)),
    # Zone classification (0=FW, 1=core, 2=LDRB extension, 3=geo-only)
    make_dg0("zone", zone),
    make_dg0("zone_detailed", zone_detailed),
    # Distances
    make_dg0("d_LV", d_lv),
    make_dg0("d_RV", d_rv),
    make_dg0("d_EPI", d_epi),
    make_dg0("d_sum", d_sum),
]

# Main XDMF with all fields
outfile = outdir / "comparison.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(outfile), "w") as xdmf:
    xdmf.write_mesh(mesh)
    for fn in fields:
        xdmf.write_function(fn)

# Also export the CG1 Laplace solutions (smoother visualization)
outfile_cg1 = outdir / "laplace_fields_cg1.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(outfile_cg1), "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(lv_rv_scalar_fn)
    xdmf.write_function(epi_scalar_fn)

print(f"\n{'=' * 70}")
print(f"EXPORTED FILES")
print(f"{'=' * 70}")
print(f"\n  {outfile}")
print(f"    DG0 fields: tau_euclidean, tau_laplace, tau_difference,")
print(f"    epi_scalar_dg0, is_geometric_septum, is_ldrb_septum, is_union,")
print(f"    zone, zone_detailed, d_LV, d_RV, d_EPI, d_sum")
print(f"\n  {outfile_cg1}")
print(f"    CG1 fields: lv_rv_scalar, epi_scalar (smoother, for visual comparison)")
print(f"\nParaView tips:")
print(f"  - Compare tau_euclidean vs tau_laplace side by side")
print(f"  - Use 'zone' to see: 1=core (both agree), 2=LDRB extension, 3=geo-only")
print(f"  - Use 'zone_detailed' for: -2=RV FW, -1=LV FW, 0=outside, 1=core, 2=ext, 3=geo-only")
print(f"  - Threshold on 'is_union' >= 0.5 to isolate the study region")
print(f"  - Check 'tau_difference' for where Euclidean and Laplace disagree most")
