#!/usr/bin/env python3
"""
verify_sweep_envelope.py — Visual verification of the t-threshold septum sweep

For one UKB sim's mesh, computes:
  - Per-cell Euclidean distances d_LV, d_RV, d_epi (via cKDTree on surface vertices)
  - Per-cell entry_t = max(d_LV, d_RV) - d_epi
      Interpretation: a cell is in septum(t) iff entry_t(cell) < t
      t=0 corresponds exactly to the geometric definition
      t<0 is tighter than geometric (only the deepest cells)
      t>0 is wider than geometric (adds cells approaching the epicardium)
  - Per-cell envelope = (d_epi >= d_epi_min) AND (d_sum <= d_sum_max)
      Anatomical bound: excludes cells near the epicardial surface and cells
      far from both endos (so the sweep stays inside the septum region even
      as t grows)
  - entry_t_for_sweep: entry_t inside the envelope, a large sentinel (+1e6)
                      outside — use this in ParaView to step through the sweep

Exports an XDMF that you open in ParaView:
  1. Select "entry_t_for_sweep" as the color field
  2. Apply a Threshold filter: keep cells with entry_t_for_sweep <= <slider>
  3. Drag the slider from a negative value (e.g., -5) up to +10 or so
  4. Watch the septum grow

Pass criteria:
  - At slider = 0, the visible cells EXACTLY match the geometric septum
  - The growth is spatially coherent (no distant cells popping in)
  - No cells on the epicardial surface ever appear (envelope guard works)

Usage:
  python verify_sweep_envelope.py [geometry_dir]
  python verify_sweep_envelope.py --d-epi-min 2.0 --d-sum-max 25.0
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


parser = argparse.ArgumentParser()
parser.add_argument("geodir", nargs="?",
                    default="/home/dtsteene/D1/cardiac-work/results/sims/2026-04-08/UKB_6beats_run_1017516/geometry",
                    help="Geometry directory (must contain mesh.xdmf)")
parser.add_argument("--d-epi-min", type=float, default=2.0,
                    help="Minimum d_epi (distance to epi surface) for a cell to be in the "
                         "envelope (mm). Distance is computed as point-to-triangle "
                         "(facet) distance, which is mesh-resolution-independent.")
parser.add_argument("--n-epi-layers-exclude", type=int, default=1,
                    help="Number of cell layers adjacent to the epi surface to exclude from "
                         "the envelope, using mesh topology (cell-to-facet adjacency). "
                         "1 = exclude cells with ANY face on the epi (cells literally touching "
                         "the outer surface). 2 = also exclude cells that share a face with an "
                         "epi-touching cell (i.e., cells 2 layers deep). Set to 0 to disable.")
parser.add_argument("--distance-method", choices=["vertex", "facet"], default="facet",
                    help="How to compute per-cell distances to surfaces: 'vertex' uses "
                         "cKDTree on surface vertex coordinates (fast but coarse on low-res "
                         "meshes); 'facet' uses VTK implicit distance to the surface triangles "
                         "(slightly slower but correct).")
parser.add_argument("--d-sum-min", type=float, default=4.0,
                    help="Minimum d_sum = d_LV + d_RV for a cell to be in the envelope (mm). "
                         "Rejects degenerate cells where both endos are very close. "
                         "Keep below the thinnest wall region of the septum "
                         "(typically 5-8 mm total thickness in PAH).")
parser.add_argument("--d-sum-max", type=float, default=22.0,
                    help="Maximum d_sum for a cell to be in the envelope (mm). "
                         "Rejects free-wall cells where one endo is far.")
parser.add_argument("--output-name", type=str, default="sweep_envelope",
                    help="Base name for the output XDMF (written next to geodir)")
args = parser.parse_args()

geodir = Path(args.geodir).resolve()
print(f"Loading mesh from {geodir}/mesh.xdmf")

# ── Load mesh + facet tags ───────────────────────────────────────────────────
with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(geodir / "mesh.xdmf"), "r") as xdmf:
    mesh = xdmf.read_mesh(name="Mesh")
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(2, 0)
    ffun = xdmf.read_meshtags(mesh, name="Facet tags")

# Determine mesh unit from coordinate range
coords_bb = mesh.geometry.x.max(axis=0) - mesh.geometry.x.min(axis=0)
print(f"Mesh bbox: {coords_bb[0]:.2f} x {coords_bb[1]:.2f} x {coords_bb[2]:.2f}")
if coords_bb.max() > 10:
    unit = "mm"
    mesh_to_mm = 1.0
else:
    unit = "m"
    mesh_to_mm = 1000.0
print(f"Mesh unit: {unit} (multiply by {mesh_to_mm} for mm)")

# Marker values (from markers.json convention used elsewhere)
LV_MARKER = 1
RV_MARKER = 2
EPI_MARKER = 3

# ── Extract surface mesh triangles via facet-tag connectivity ───────────────
f2v = mesh.topology.connectivity(2, 0)

# Build cell-to-facet connectivity so we can find cells that share faces
# with tagged epi facets (used for topological epi-adjacency exclusion)
mesh.topology.create_connectivity(3, 2)
c2f = mesh.topology.connectivity(3, 2)
mesh.topology.create_connectivity(2, 3)
f2c = mesh.topology.connectivity(2, 3)
epi_facets_set = set(ffun.find(EPI_MARKER).tolist())

def surface_triangles(marker_value):
    """Build a PolyData with triangle faces for cells with the given marker."""
    facets = ffun.find(marker_value)
    tri_verts = []
    vertex_set = set()
    for f in facets:
        verts = f2v.links(f)
        if len(verts) != 3:
            continue  # skip non-triangle facets (shouldn't happen for tet mesh)
        tri_verts.append(verts)
        vertex_set.update(verts)
    if not tri_verts:
        return None
    # Build PolyData
    unique_verts = sorted(vertex_set)
    vertex_map = {v: i for i, v in enumerate(unique_verts)}
    points = mesh.geometry.x[unique_verts]
    faces_flat = []
    for v0, v1, v2 in tri_verts:
        faces_flat.extend([3, vertex_map[v0], vertex_map[v1], vertex_map[v2]])
    poly = pv.PolyData(points, faces=np.array(faces_flat, dtype=np.int64))
    return poly

lv_surface = surface_triangles(LV_MARKER)
rv_surface = surface_triangles(RV_MARKER)
epi_surface = surface_triangles(EPI_MARKER)
print(f"Surface facets: LV={lv_surface.n_faces_strict}, "
      f"RV={rv_surface.n_faces_strict}, EPI={epi_surface.n_faces_strict}")

# ── Per-cell distances ───────────────────────────────────────────────────────
cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
centroids = dolfinx.mesh.compute_midpoints(mesh, 3, cells)
n_cells = len(cells)
print(f"Cells: {n_cells}")

if args.distance_method == "facet":
    # VTK implicit distance: nearest point on any triangle of the surface.
    # Unsigned (always positive). This is mesh-resolution-independent.
    print("Computing point-to-facet distances (facet method)...")
    centroids_poly = pv.PolyData(centroids.astype(np.float64))
    d_lv = np.abs(centroids_poly.compute_implicit_distance(lv_surface)["implicit_distance"]) * mesh_to_mm
    d_rv = np.abs(centroids_poly.compute_implicit_distance(rv_surface)["implicit_distance"]) * mesh_to_mm
    d_epi = np.abs(centroids_poly.compute_implicit_distance(epi_surface)["implicit_distance"]) * mesh_to_mm
else:
    # Fallback: nearest-vertex distance via cKDTree (fast but coarse).
    print("Computing nearest-vertex distances (vertex method)...")
    tree_lv = cKDTree(lv_surface.points)
    tree_rv = cKDTree(rv_surface.points)
    tree_epi = cKDTree(epi_surface.points)
    d_lv = tree_lv.query(centroids)[0] * mesh_to_mm
    d_rv = tree_rv.query(centroids)[0] * mesh_to_mm
    d_epi = tree_epi.query(centroids)[0] * mesh_to_mm

d_sum = d_lv + d_rv

# ── LDRB reference markers via Laplace scalar fields ────────────────────────
# Solve the standard LDRB Laplace problems (what the fenicsx-ldrb main branch
# uses) so we can show the LDRB septum definition as a reference overlay.
#   lv_rv_scalar: u=1 on LV endo, u=0 on RV endo
#   epi_scalar:   u=1 on epi,     u=0 on both endos
# LDRB septum rule: (epi_scalar <= 0.5) AND (0.1 < lv_rv_scalar < 0.9)
print("Solving Laplace equations for LDRB septum reference...")
V_CG1 = dolfinx.fem.functionspace(mesh, ("CG", 1))
u_trial = ufl.TrialFunction(V_CG1)
v_test = ufl.TestFunction(V_CG1)
a_laplace = ufl.dot(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
L_zero = dolfinx.fem.Constant(mesh, 0.0) * v_test * ufl.dx

lv_facets = ffun.find(LV_MARKER)
rv_facets = ffun.find(RV_MARKER)
epi_facets = ffun.find(EPI_MARKER)
lv_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, 2, lv_facets)
rv_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, 2, rv_facets)
epi_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, 2, epi_facets)

problem_lvrv = dolfinx.fem.petsc.LinearProblem(
    a_laplace, L_zero,
    bcs=[dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), lv_dofs, V_CG1),
         dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_dofs, V_CG1)],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="verify_lvrv",
)
lv_rv_scalar = problem_lvrv.solve()

problem_epi = dolfinx.fem.petsc.LinearProblem(
    a_laplace, L_zero,
    bcs=[dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), epi_dofs, V_CG1),
         dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), lv_dofs, V_CG1),
         dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_dofs, V_CG1)],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="verify_epi",
)
epi_scalar = problem_epi.solve()

# Interpolate CG1 → DG0 per cell
V_DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
lvrv_dg0 = dolfinx.fem.Function(V_DG0)
lvrv_dg0.interpolate(lv_rv_scalar)
lvrv_vals = lvrv_dg0.x.array[:n_cells].copy()

epi_dg0 = dolfinx.fem.Function(V_DG0)
epi_dg0.interpolate(epi_scalar)
epi_vals = epi_dg0.x.array[:n_cells].copy()

# LDRB septum rule
is_ldrb_septum = (epi_vals <= 0.5) & (lvrv_vals > 0.1) & (lvrv_vals < 0.9)
print(f"LDRB septum cells: {int(is_ldrb_septum.sum())}")

# ── Topological epi-adjacency exclusion ─────────────────────────────────────
# Mark cells that have any face in the epi facet set. Then optionally expand
# the exclusion by n_layers (each expansion adds neighbors of the current set).
touches_epi = np.zeros(n_cells, dtype=bool)
for cell_i in range(n_cells):
    for facet in c2f.links(cell_i):
        if facet in epi_facets_set:
            touches_epi[cell_i] = True
            break

# Expand exclusion by additional layers if requested
epi_adjacent = touches_epi.copy()
for layer in range(1, args.n_epi_layers_exclude):
    new_layer = np.zeros(n_cells, dtype=bool)
    # For each cell currently marked, mark its face-neighbors too
    for cell_i in np.where(epi_adjacent)[0]:
        for facet in c2f.links(cell_i):
            for neighbor in f2c.links(facet):
                if neighbor != cell_i:
                    new_layer[neighbor] = True
    epi_adjacent |= new_layer

# Final mask: cells to exclude from the envelope because they're too close
# (topologically) to the epi surface
exclude_by_topology = epi_adjacent if args.n_epi_layers_exclude > 0 else np.zeros(n_cells, dtype=bool)

print(f"\nCells touching epi surface (n_epi_layers_exclude=1):  {int(touches_epi.sum())}")
print(f"Cells within {args.n_epi_layers_exclude} layer(s) of epi (to exclude): {int(exclude_by_topology.sum())}")

# ── Sweep fields ─────────────────────────────────────────────────────────────
# entry_t is the t at which this cell is first included by the sweep rule
#   rule:  cell in septum(t)  iff  max(d_LV, d_RV) < d_epi + t
#          iff  entry_t < t
entry_t = np.maximum(d_lv, d_rv) - d_epi  # in mm

# Envelope: anatomical bound that the sweep must stay within
envelope = ((d_epi >= args.d_epi_min)
            & (d_sum >= args.d_sum_min)
            & (d_sum <= args.d_sum_max)
            & ~exclude_by_topology)
# breakdown for diagnostic printing
env_epi_ok = (d_epi >= args.d_epi_min)
env_sum_min_ok = (d_sum >= args.d_sum_min)
env_sum_max_ok = (d_sum <= args.d_sum_max)
env_not_touching_epi = ~exclude_by_topology

# entry_t_for_sweep: large sentinel outside envelope so those cells never
# appear when the user slides a Threshold filter up
LARGE_SENTINEL = 1e6
entry_t_for_sweep = np.where(envelope, entry_t, LARGE_SENTINEL)

# Normalized gradient scalar: 0 at the deepest (tightest) cells,
# 1 at the widest (highest entry_t) cells, within the envelope.
# Outside envelope: NaN (so ParaView can mask it out cleanly).
# Use this to SEE how the sweep progresses through the septum at a glance.
if envelope.any():
    et_in = entry_t[envelope]
    et_lo, et_hi = float(et_in.min()), float(et_in.max())
    if et_hi - et_lo > 1e-12:
        frac = (entry_t - et_lo) / (et_hi - et_lo)
    else:
        frac = np.zeros_like(entry_t)
    septum_depth_fraction = np.where(envelope, frac, np.nan)
else:
    septum_depth_fraction = np.full(n_cells, np.nan)

# Geometric septum mask for reference
is_geometric = (np.maximum(d_lv, d_rv) < d_epi)  # same as entry_t < 0

# ── Sanity checks ────────────────────────────────────────────────────────────
print("\n=== SANITY CHECKS ===\n")

n_geo = int(is_geometric.sum())
n_env = int(envelope.sum())
n_geo_in_env = int((is_geometric & envelope).sum())
n_geo_out_env = int((is_geometric & ~envelope).sum())
print(f"Geometric septum cells:                 {n_geo}")
print(f"Envelope cells:                         {n_env}")
print(f"Geometric ∩ envelope:                   {n_geo_in_env}")
print(f"Geometric cells EXCLUDED by envelope:   {n_geo_out_env}  "
      f"{'<-- BAD if > 0' if n_geo_out_env > 0 else '(good, envelope keeps all geometric)'}")

# Per-constraint breakdown
print(f"\nEnvelope constraint breakdown (using current parameters):")
print(f"  d_epi >= {args.d_epi_min} mm            passes: {int(env_epi_ok.sum()):>4}")
print(f"  d_sum >= {args.d_sum_min} mm            passes: {int(env_sum_min_ok.sum()):>4}")
print(f"  d_sum <= {args.d_sum_max} mm           passes: {int(env_sum_max_ok.sum()):>4}")
print(f"  not touching epi (topology): passes: {int(env_not_touching_epi.sum()):>4}")
print(f"  All four (envelope):         passes: {int(envelope.sum()):>4}")

# Show how many cells each constraint REJECTS
n_rej_epi = int((~env_epi_ok).sum())
n_rej_sum_min = int((~env_sum_min_ok).sum())
n_rej_sum_max = int((~env_sum_max_ok).sum())
print(f"\nCells rejected by each constraint:")
print(f"  too close to epi (d_epi < {args.d_epi_min}):  {n_rej_epi:>4}")
print(f"  thin wall (d_sum < {args.d_sum_min}):         {n_rej_sum_min:>4}")
print(f"  too deep in free wall (d_sum > {args.d_sum_max}): {n_rej_sum_max:>4}")

if n_geo_out_env > 0:
    excluded_mask = is_geometric & ~envelope
    exc_d_epi = d_epi[excluded_mask]
    exc_d_sum = d_sum[excluded_mask]
    print(f"\n  Excluded-geo cells: d_epi range [{exc_d_epi.min():.2f}, {exc_d_epi.max():.2f}] mm")
    print(f"  Excluded-geo cells: d_sum range [{exc_d_sum.min():.2f}, {exc_d_sum.max():.2f}] mm")
    print(f"  → may need to relax envelope to keep these (e.g. lower d_epi_min or d_sum_min)")

# Check that entry_t < 0 ⇔ geometric (by construction should be exact)
from_rule = (entry_t < 0)
mismatch = int((from_rule != is_geometric).sum())
print(f"\nConsistency check (entry_t<0 vs geometric rule): {mismatch} mismatches "
      f"{'<-- BAD' if mismatch else '(ok)'}")

# entry_t statistics within envelope
env_entries = entry_t[envelope]
print(f"\nentry_t within envelope:")
print(f"  min  = {env_entries.min():>8.3f} mm  (tightest cells — most deeply septal)")
print(f"  max  = {env_entries.max():>8.3f} mm  (loosest cells — widest point of the sweep)")
print(f"  geometric threshold: entry_t < 0")
print(f"  {int((env_entries < 0).sum())} cells have entry_t < 0  (these are the geometric septum)")
print(f"  {int((env_entries >= 0).sum())} cells have entry_t >= 0 (add as t grows)")

# Suggest some threshold values for the slider
print(f"\nSuggested slider positions for ParaView Threshold:")
for t in [-5, -3, -1, 0, 1, 2, 3, 5, 10]:
    n_in = int((env_entries < t).sum())
    print(f"  t = {t:>+3} mm  →  {n_in:>4} cells in septum(t)")

# ── Reference markers: encode named septum definitions as an integer label ──
# ref_marker values:
#   0  → not in any reference definition (outside envelope or free wall)
#   1  → LDRB-only (in LDRB septum but NOT in geometric septum)
#   2  → geometric-only (in geometric septum but NOT in LDRB)
#   3  → BOTH (intersection: in geometric AND in LDRB)
# Use this in ParaView with a categorical color map to see at a glance:
#   - the geometric reference = cells with ref_marker ∈ {2, 3}
#   - the LDRB reference      = cells with ref_marker ∈ {1, 3}
#   - the intersection        = cells with ref_marker == 3
ref_marker = np.zeros(n_cells, dtype=np.float64)
ref_marker[is_ldrb_septum & ~is_geometric] = 1.0
ref_marker[is_geometric & ~is_ldrb_septum] = 2.0
ref_marker[is_geometric & is_ldrb_septum] = 3.0

# Compute the t value where sweep set cell count most closely matches |LDRB|.
# This tells us: "at what t does our t-threshold sweep best approximate the
# cell count of the LDRB definition?" (approximate because the SETS differ —
# see note about the basal lip mismatch)
n_ldrb = int(is_ldrb_septum.sum())
# Sweep cell count at each t: count of cells in envelope with entry_t < t
# Binary search along t for the matching cell count
sorted_entries = np.sort(entry_t[envelope])
if n_ldrb <= len(sorted_entries):
    t_match_ldrb_by_count = float(sorted_entries[min(n_ldrb - 1, len(sorted_entries) - 1)])
else:
    t_match_ldrb_by_count = float(sorted_entries[-1])

# Similarly for geometric (should give t ≈ 0 since by construction)
n_geometric_in_env = int((is_geometric & envelope).sum())
if n_geometric_in_env <= len(sorted_entries):
    t_match_geo_by_count = float(sorted_entries[min(n_geometric_in_env - 1, len(sorted_entries) - 1)])
else:
    t_match_geo_by_count = 0.0

print(f"\nReference point t values (by cell-count match):")
print(f"  geometric ({n_geometric_in_env} cells in envelope) → t ≈ {t_match_geo_by_count:>+6.2f} mm  "
      f"(should be ≈ 0 by construction)")
print(f"  LDRB      ({n_ldrb} cells total)         → t ≈ {t_match_ldrb_by_count:>+6.2f} mm  "
      f"(approximate — LDRB set and sweep set differ by ~40-100 cells at the basal lip)")

# Jaccard similarity between sweep(t_match) and LDRB
def jaccard(a, b):
    i = int((a & b).sum())
    u = int((a | b).sum())
    return i / u if u else 0.0

sweep_at_ldrb_match = envelope & (entry_t < t_match_ldrb_by_count)
j_ldrb = jaccard(sweep_at_ldrb_match, is_ldrb_septum)
j_geo = jaccard(envelope & (entry_t < 0), is_geometric)
print(f"\nJaccard similarity (set overlap, 1.0 = identical, 0.0 = disjoint):")
print(f"  geometric ↔ sweep(t=0):        {j_geo:.3f}")
print(f"  LDRB      ↔ sweep(t={t_match_ldrb_by_count:.2f}): {j_ldrb:.3f}")

# ── Write XDMF with all fields ───────────────────────────────────────────────
# Note: V_DG0 was already created earlier (for the CG1→DG0 interpolation)

def make_field(name, arr):
    f = dolfinx.fem.Function(V_DG0, name=name)
    f.x.array[:n_cells] = arr.astype(np.float64)
    return f

fields = [
    make_field("entry_t_for_sweep", entry_t_for_sweep),   # ← main slider field
    make_field("septum_depth_fraction", septum_depth_fraction),  # 0=deepest, 1=widest, NaN outside
    make_field("ref_marker", ref_marker),                  # 0=none, 1=LDRB-only, 2=geo-only, 3=both
    make_field("is_geometric_septum", is_geometric.astype(np.float64)),
    make_field("is_ldrb_septum", is_ldrb_septum.astype(np.float64)),
    make_field("entry_t_raw", entry_t),
    make_field("envelope", envelope.astype(np.float64)),
    make_field("lv_rv_scalar", lvrv_vals),                 # raw LDRB Laplace
    make_field("epi_scalar", epi_vals),                    # raw LDRB Laplace
    make_field("d_lv_mm", d_lv),
    make_field("d_rv_mm", d_rv),
    make_field("d_epi_mm", d_epi),
    make_field("d_sum_mm", d_sum),
]

outdir = geodir.parent / "sweep_envelope_check"
outdir.mkdir(exist_ok=True)
outfile = outdir / f"{args.output_name}.xdmf"

with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(outfile), "w") as xdmf:
    xdmf.write_mesh(mesh)
    for f in fields:
        xdmf.write_function(f)

print(f"\n=== OUTPUT ===")
print(f"Wrote {outfile}")
print()
print("In ParaView:")
print(f"  1. File → Open → {outfile}  →  Apply")
print("  2. (Optional) Clip to see the inside of the heart")
print()
print("Views you can look at (each just by changing the 'Color by' dropdown):")
print()
print("  A) Gradient view — smooth sweep progression")
print("     Color by 'septum_depth_fraction'")
print("     Dark blue = tightest (deepest septum)")
print("     Dark red  = widest (envelope bound)")
print("     NaN outside envelope (transparent or background)")
print()
print("  B) Named definitions — where do geometric and LDRB stand?")
print("     Color by 'ref_marker' with a categorical colormap")
print("     0 = outside both    (grey/transparent)")
print("     1 = LDRB only       (red)")
print("     2 = geometric only  (blue)")
print("     3 = both (intersection) (purple)")
print("     You'll see how the two definitions disagree at the boundaries.")
print()
print("  C) Sliding threshold sweep — see cells enter as t grows")
print("     Filters → Threshold")
print("     Array: 'entry_t_for_sweep', Lower: -100")
print("     Upper threshold: drag from -5 up through 0 to ~+10")
print("     At upper=0 you get exactly the geometric septum.")
print("     At upper>0 the sweep widens smoothly outward.")
print()
print("  D) Raw LDRB Laplace fields (for debugging the LDRB definition)")
print("     Color by 'lv_rv_scalar' or 'epi_scalar'")
