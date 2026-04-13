"""
Explore the transmural coordinate tau and its relationship to geometric septum tagging.

Computes:
1. tau = d_LV / (d_LV + d_RV) per cell
2. Geometric septum tags (max(d_LV, d_RV) < d_epi)
3. Histogram of tau values, colored by region tag
4. Exports tau as XDMF for ParaView visualization

Usage:
    python explore_tau.py [geometry_dir]

    geometry_dir defaults to the UKB healthy spectrum case.
"""

import argparse
from pathlib import Path

import dolfinx
import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "geodir",
    nargs="?",
    default="/home/dtsteene/D1/cardiac-work/results/sims/ukb_10beats_spectrum/UKB_10beats_healthy/geometry",
)
parser.add_argument("--nbins", type=int, default=20, help="Number of tau bins")
args = parser.parse_args()

geodir = Path(args.geodir)
print(f"Loading geometry from {geodir}")

# ── Load mesh and facet tags ─────────────────────────────────────────
with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(geodir / "mesh.xdmf"), "r") as xdmf:
    mesh = xdmf.read_mesh(name="Mesh")
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(2, 0)  # facets → vertices
    ffun = xdmf.read_meshtags(mesh, name="Facet tags")

# Marker values (from markers.json: LV=1, RV=2, EPI=3, BASE=4)
LV_MARKER = 1
RV_MARKER = 2
EPI_MARKER = 3

# ── Extract surface vertex coordinates ───────────────────────────────
f2v = mesh.topology.connectivity(2, 0)


def surface_coords(marker_value):
    """Get coordinates of all vertices belonging to facets with given marker."""
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

# ── Build KDTrees ────────────────────────────────────────────────────
tree_lv = cKDTree(lv_coords)
tree_rv = cKDTree(rv_coords)
tree_epi = cKDTree(epi_coords)

# ── Compute distances from cell centroids ────────────────────────────
cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
centroids = dolfinx.mesh.compute_midpoints(mesh, 3, cells)

d_lv = tree_lv.query(centroids)[0]
d_rv = tree_rv.query(centroids)[0]
d_epi = tree_epi.query(centroids)[0]

print(f"Number of cells: {len(cells)}")

# ── Compute tau ──────────────────────────────────────────────────────
tau = d_lv / (d_lv + d_rv)

# ── Compute geometric septum tags ────────────────────────────────────
is_septum = np.maximum(d_lv, d_rv) < d_epi
region = np.zeros(len(cells), dtype=np.int32)
region[is_septum] = 3  # Septum
region[~is_septum & (d_lv <= d_rv)] = 1  # LV free wall
region[~is_septum & (d_lv > d_rv)] = 2  # RV free wall

n_lv = np.sum(region == 1)
n_rv = np.sum(region == 2)
n_sept = np.sum(region == 3)
print(f"\nGeometric tags: LV={n_lv} ({100*n_lv/len(cells):.1f}%), "
      f"RV={n_rv} ({100*n_rv/len(cells):.1f}%), "
      f"Septum={n_sept} ({100*n_sept/len(cells):.1f}%)")

# ── Tau statistics by region ─────────────────────────────────────────
print("\nTau statistics by geometric region:")
for name, val in [("LV free wall", 1), ("RV free wall", 2), ("Septum", 3)]:
    mask = region == val
    t = tau[mask]
    print(f"  {name:15s}: tau = {t.min():.3f} – {t.max():.3f}  "
          f"(mean={t.mean():.3f}, median={np.median(t):.3f})")

# ── Histogram of tau by region ───────────────────────────────────────
bin_edges = np.linspace(0, 1, args.nbins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

print(f"\nTau histogram ({args.nbins} bins):")
print(f"{'Bin':>8s} {'tau range':>14s} {'Total':>6s} {'LV':>5s} {'RV':>5s} {'Sept':>5s}")
print("-" * 55)
for i in range(args.nbins):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask = (tau >= lo) & (tau < hi) if i < args.nbins - 1 else (tau >= lo) & (tau <= hi)
    n_total = np.sum(mask)
    n_lv_bin = np.sum(mask & (region == 1))
    n_rv_bin = np.sum(mask & (region == 2))
    n_sept_bin = np.sum(mask & (region == 3))
    print(f"  {i:4d}    [{lo:.2f}, {hi:.2f})  {n_total:5d}  {n_lv_bin:5d} {n_rv_bin:5d} {n_sept_bin:5d}")

# ── Distance statistics ──────────────────────────────────────────────
print("\nDistance statistics (mm):")
scale = 1000  # m → mm
print(f"  d_LV:  min={d_lv.min()*scale:.1f}, max={d_lv.max()*scale:.1f}, mean={d_lv.mean()*scale:.1f}")
print(f"  d_RV:  min={d_rv.min()*scale:.1f}, max={d_rv.max()*scale:.1f}, mean={d_rv.mean()*scale:.1f}")
print(f"  d_EPI: min={d_epi.min()*scale:.1f}, max={d_epi.max()*scale:.1f}, mean={d_epi.mean()*scale:.1f}")

# ── Export to XDMF for ParaView ──────────────────────────────────────
outdir = geodir.parent / "tau_exploration"
outdir.mkdir(exist_ok=True)

V_DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))

# Tau field
tau_fn = dolfinx.fem.Function(V_DG0, name="tau")
tau_fn.x.array[:len(cells)] = tau

# Region field
region_fn = dolfinx.fem.Function(V_DG0, name="geometric_region")
region_fn.x.array[:len(cells)] = region.astype(np.float64)

# d_LV, d_RV, d_EPI fields
dlv_fn = dolfinx.fem.Function(V_DG0, name="d_LV")
dlv_fn.x.array[:len(cells)] = d_lv

drv_fn = dolfinx.fem.Function(V_DG0, name="d_RV")
drv_fn.x.array[:len(cells)] = d_rv

depi_fn = dolfinx.fem.Function(V_DG0, name="d_EPI")
depi_fn.x.array[:len(cells)] = d_epi

outfile = outdir / "tau_exploration.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(outfile), "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(tau_fn)
    xdmf.write_function(region_fn)
    xdmf.write_function(dlv_fn)
    xdmf.write_function(drv_fn)
    xdmf.write_function(depi_fn)

print(f"\nExported to {outfile}")
print("Open in ParaView to visualize tau as a continuous field on the mesh.")
print("Compare with geometric_region (discrete: 1=LV, 2=RV, 3=Septum).")

# ── Investigate d_epi as epicardial junction filter ──────────────────
print("\n" + "=" * 70)
print("EPICARDIAL JUNCTION ANALYSIS")
print("=" * 70)

# Normalize d_epi relative to local wall thickness
# "depth ratio" = d_epi / (d_epi + min(d_lv, d_rv))
# High depth_ratio = deep in the wall (endocardial side)
# Low depth_ratio = near epicardial surface
d_endo_min = np.minimum(d_lv, d_rv)
depth_ratio = d_epi / (d_epi + d_endo_min)

print(f"\nDepth ratio = d_epi / (d_epi + min(d_LV, d_RV))")
print(f"  High = deep in wall (far from epi), Low = near epi surface")
print(f"\nDepth ratio by geometric region:")
for name, val in [("LV free wall", 1), ("RV free wall", 2), ("Septum", 3)]:
    mask = region == val
    dr = depth_ratio[mask]
    print(f"  {name:15s}: {dr.min():.3f} – {dr.max():.3f}  "
          f"(mean={dr.mean():.3f}, median={np.median(dr):.3f})")

# Show cells in the mid-tau range (0.3-0.7) split by depth
mid_tau = (tau >= 0.3) & (tau <= 0.7)
print(f"\nCells with tau in [0.3, 0.7] (potential septum): {np.sum(mid_tau)}")
print(f"  Of which geometric-tagged septum: {np.sum(mid_tau & is_septum)}")
print(f"  Of which geometric-tagged free wall: {np.sum(mid_tau & ~is_septum)}")

# These free-wall cells in mid-tau range — are they epicardial junction?
mid_tau_freewall = mid_tau & ~is_septum
if np.sum(mid_tau_freewall) > 0:
    print(f"\n  Free wall cells in mid-tau range (the contamination):")
    dr_contam = depth_ratio[mid_tau_freewall]
    depi_contam = d_epi[mid_tau_freewall] * scale
    print(f"    depth_ratio: {dr_contam.min():.3f} – {dr_contam.max():.3f} (mean={dr_contam.mean():.3f})")
    print(f"    d_epi (mm):  {depi_contam.min():.1f} – {depi_contam.max():.1f} (mean={depi_contam.mean():.1f})")
    print(f"    regions: LV={np.sum(mid_tau_freewall & (region==1))}, RV={np.sum(mid_tau_freewall & (region==2))}")

    # Compare with septum cells in same tau range
    mid_tau_septum = mid_tau & is_septum
    dr_sept = depth_ratio[mid_tau_septum]
    depi_sept = d_epi[mid_tau_septum] * scale
    print(f"\n  Septum cells in same tau range (the real septum):")
    print(f"    depth_ratio: {dr_sept.min():.3f} – {dr_sept.max():.3f} (mean={dr_sept.mean():.3f})")
    print(f"    d_epi (mm):  {depi_sept.min():.1f} – {depi_sept.max():.1f} (mean={depi_sept.mean():.1f})")

# Can we find a d_epi threshold that separates them?
print(f"\n--- d_epi filter analysis ---")
print(f"If we filter by d_epi > threshold, how many mid-tau cells survive?")
for thresh_mm in [2, 3, 4, 5, 6, 7, 8]:
    thresh = thresh_mm / scale  # mm to mesh units
    deep_enough = d_epi > thresh
    n_survive = np.sum(mid_tau & deep_enough)
    n_sept_survive = np.sum(mid_tau & deep_enough & is_septum)
    n_fw_survive = np.sum(mid_tau & deep_enough & ~is_septum)
    print(f"  d_epi > {thresh_mm}mm: {n_survive} cells survive "
          f"(septum={n_sept_survive}, free wall={n_fw_survive})")

# Also export depth_ratio to XDMF
depth_fn = dolfinx.fem.Function(V_DG0, name="depth_ratio")
depth_fn.x.array[:len(cells)] = depth_ratio

# Re-export with depth_ratio included
with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(outfile), "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(tau_fn)
    xdmf.write_function(region_fn)
    xdmf.write_function(dlv_fn)
    xdmf.write_function(drv_fn)
    xdmf.write_function(depi_fn)
    xdmf.write_function(depth_fn)

print(f"\nRe-exported with depth_ratio to {outfile}")

# ── Export analysis regions to separate XDMF for visual inspection ───
# Define 4 zones:
#   0 = excluded (epicardial junction: mid-tau but low depth_ratio)
#   1 = LV free wall (tau < 0.3, clearly LV-dominated)
#   2 = RV free wall (tau > 0.7, clearly RV-dominated)
#   3 = geometric septum (max(d_LV, d_RV) < d_epi)
#   4 = transition zone (mid-tau, high depth_ratio, but NOT geometric septum)
#       These are the cells a "too generous" segmentation would include.

# d_sum = d_LV + d_RV: total distance to both endo surfaces.
# Small d_sum = sandwiched between two nearby endo surfaces (septal/near-septal).
# Large d_sum = far from both endo (epicardial junction, deep free wall).
# Septum cells have d_sum = 7.8–14.5mm on this mesh.
# Using 18mm captures ~54% expansion beyond geometric septum while
# excluding epicardial junction cells (d_sum > 20mm).
D_SUM_THRESH = 18.0  # mm — adjust to control transition zone width
d_sum = d_lv + d_rv

zone = np.zeros(len(cells), dtype=np.float64)

# Geometric septum is the primary classifier — always zone 3
zone[is_septum] = 3.0

# Non-septum cells: classify by distance to nearest endo
zone[~is_septum & (d_lv <= d_rv)] = 1.0   # LV free wall
zone[~is_septum & (d_lv > d_rv)] = 2.0    # RV free wall

# Transition zone: non-septum cells with small d_sum (near-septal tissue)
# and intermediate tau (in the septum's tau range).
sept_tau_min = tau[is_septum].min() - 0.02  # small margin
sept_tau_max = tau[is_septum].max() + 0.02
near_septum_tau = (tau >= sept_tau_min) & (tau <= sept_tau_max)
zone[~is_septum & near_septum_tau & (d_sum < D_SUM_THRESH)] = 4.0

zone_fn = dolfinx.fem.Function(V_DG0, name="analysis_zone")
zone_fn.x.array[:len(cells)] = zone

# Also make a "study region" mask: geometric septum + transition zone
# This is the maximal region we'd study with transmural profiles
study_region = np.zeros(len(cells), dtype=np.float64)
study_region[zone == 3.0] = 1.0  # Core septum
study_region[zone == 4.0] = 0.5  # Transition zone
study_fn = dolfinx.fem.Function(V_DG0, name="study_region")
study_fn.x.array[:len(cells)] = study_region

# Count zones
for zval, zname in [(0, "Epicardial junction (excluded)"),
                     (1, "LV free wall"),
                     (2, "RV free wall"),
                     (3, "Geometric septum (core)"),
                     (4, "Transition zone (extra)")]:
    n = np.sum(zone == zval)
    print(f"  Zone {zval} ({zname}): {n} cells ({100*n/len(cells):.1f}%)")

# Tau range in each zone
for zval, zname in [(3, "Geometric septum"), (4, "Transition zone")]:
    mask = zone == zval
    if np.sum(mask) > 0:
        t = tau[mask]
        dr = depth_ratio[mask]
        de = d_epi[mask] * scale
        print(f"\n  {zname}:")
        print(f"    tau:         {t.min():.3f} – {t.max():.3f}")
        print(f"    depth_ratio: {dr.min():.3f} – {dr.max():.3f}")
        print(f"    d_epi (mm):  {de.min():.1f} – {de.max():.1f}")

# Export zones XDMF
zone_outfile = outdir / "tau_zones.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(zone_outfile), "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(tau_fn)
    xdmf.write_function(zone_fn)
    xdmf.write_function(study_fn)
    xdmf.write_function(region_fn)
    xdmf.write_function(depth_fn)

print(f"\nZone visualization exported to {zone_outfile}")
print("  analysis_zone: 0=excluded, 1=LV FW, 2=RV FW, 3=septum, 4=transition")
print("  study_region:  1.0=core septum, 0.5=transition, 0.0=outside")

# ── Extended tau histogram: beyond geometric septum ──────────────────
print(f"\n" + "=" * 70)
print("EXTENDED TAU ANALYSIS: BEYOND GEOMETRIC SEPTUM")
print("=" * 70)
print(f"\nWhat if we expand the 'septum' beyond the geometric boundary?")
print(f"Sweep: use geometric septum + free wall cells with tau in expanding range")
print(f"\n{'tau range':>16s} {'N cells':>8s} {'N sept':>7s} {'N FW':>6s} {'FW fraction':>12s}")
print("-" * 55)
for tau_lo, tau_hi in [(0.40, 0.60), (0.35, 0.65), (0.30, 0.70),
                        (0.25, 0.75), (0.20, 0.80), (0.15, 0.85),
                        (0.10, 0.90)]:
    mask = (tau >= tau_lo) & (tau <= tau_hi)
    n_tot = np.sum(mask)
    n_s = np.sum(mask & is_septum)
    n_fw = np.sum(mask & ~is_septum)
    fw_frac = n_fw / n_tot * 100 if n_tot > 0 else 0
    print(f"  [{tau_lo:.2f}, {tau_hi:.2f}]  {n_tot:7d}  {n_s:6d}  {n_fw:5d}  {fw_frac:10.1f}%")
