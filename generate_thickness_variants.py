"""
Generate UKB mesh variants at different wall thickness levels.
Outputs STL surfaces for ParaView viewing.

How it works:
  - The heart wall sits between two surfaces: epicardium (outer) and endocardium (inner)
  - Wall thickness = distance between them
  - To THICKEN: we move each endocardial point AWAY from its nearest epicardial point
    (i.e., deeper into the blood cavity), making the cavity smaller and the wall thicker
  - To THIN: we move each endocardial point TOWARD its nearest epicardial point
  - The epicardial surface is always kept fixed

Displacement convention:
  For each endo point, we compute the unit vector pointing from endo → nearest epi point.
  displacement_mm > 0: move endo toward epi = THINNING
  displacement_mm < 0: move endo away from epi (into cavity) = THICKENING
"""

import numpy as np
import h5py
import json
from pathlib import Path
from scipy.spatial import KDTree

import ukb.surface as ukb_surf
import ukb.atlas as ukb_atlas

ATLAS_PATH = Path.home() / ".ukb" / "UKBRVLV.h5"
OUT_ROOT = Path(__file__).parent / "thickness_variants"
OUT_ROOT.mkdir(exist_ok=True)

# Vertex ranges (from ukb/surface.py)
LV_ENDO = (0, 1500)
RV_ENDO_RANGES = [(1500, 2165), (2165, 3224)]
RVFW_RANGE = (5729, 5808)
EPI_RANGE = (3224, 5582)

VARIANTS = [
    ("thinned",       +1.5, "Wall thinned by 1.5 mm"),
    ("baseline",       0.0, "Mean shape, no modification"),
    ("thick_1mm",     -1.0, "Wall thickened by 1.0 mm"),
    ("thick_2mm",     -2.0, "Wall thickened by 2.0 mm"),
    ("extra_thick",   -3.0, "Wall thickened by 3.0 mm"),
]


def modify_endocardial_points(points, displacement_mm):
    """Move endo points along through-wall direction. Negative = thickening."""
    if displacement_mm == 0.0:
        return points.copy()

    pts = points.copy()
    epi = pts[EPI_RANGE[0]:EPI_RANGE[1]]
    epi_tree = KDTree(epi)

    # Helper to displace a slice of endo points
    def displace(sl):
        endo = pts[sl]
        _, nearest_idx = epi_tree.query(endo)
        direction = epi[nearest_idx] - endo  # endo → epi
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        pts[sl] = endo + displacement_mm * (direction / norms)

    displace(slice(LV_ENDO[0], LV_ENDO[1]))
    for s, e in RV_ENDO_RANGES:
        displace(slice(s, e))
    displace(slice(RVFW_RANGE[0], RVFW_RANGE[1]))
    return pts


def write_surfaces(points, folder, case="ED"):
    """Write STL surface files."""
    folder.mkdir(parents=True, exist_ok=True)
    ukb_surf.get_epi_mesh(points).write(str(folder / f"EPI_{case}.stl"))
    for ch in ["LV", "RV", "RVFW"]:
        ukb_surf.get_chamber_mesh(ch, points).write(str(folder / f"{ch}_{case}.stl"))
    for v in ["MV", "AV", "TV", "PV"]:
        ukb_surf.get_valve_mesh(v, points).write(str(folder / f"{v}_{case}.stl"))


def measure_thickness(points):
    """Wall thickness stats."""
    lv = points[LV_ENDO[0]:LV_ENDO[1]]
    rv = np.vstack([points[s:e] for s, e in RV_ENDO_RANGES])
    epi = points[EPI_RANGE[0]:EPI_RANGE[1]]
    tree = KDTree(epi)
    lv_t = tree.query(lv)[0]
    rv_t = tree.query(rv)[0]
    n_sept = 2165 - 1500
    return {
        "LV": float(np.mean(lv_t)),
        "RV_septum": float(np.mean(rv_t[:n_sept])),
        "RV_freewall": float(np.mean(rv_t[n_sept:])),
    }


# ── MAIN ──────────────────────────────────────────────────────
print("Loading UKB atlas...")
pc = h5py.File(ATLAS_PATH, "r")
ed_mean = ukb_atlas.compute_S(pc, mode=-1)
N = ed_mean.shape[1] // 2
ed_points = np.reshape(ed_mean[0, :N], (-1, 3))
pc.close()

print(f"Point cloud: {len(ed_points)} points\n")

all_results = []
for name, disp, desc in VARIANTS:
    print(f"--- {name} (disp = {disp:+.1f} mm): {desc}")

    pts_mod = modify_endocardial_points(ed_points, disp)
    thick = measure_thickness(pts_mod)
    print(f"    LV={thick['LV']:.2f}  Sept={thick['RV_septum']:.2f}  "
          f"RVFW={thick['RV_freewall']:.2f} mm")

    folder = OUT_ROOT / name
    write_surfaces(pts_mod, folder, case="ED")
    print(f"    Surfaces → {folder}/")

    params = {"variant": name, "displacement_mm": disp, "description": desc, **thick}
    (folder / "parameters.json").write_text(json.dumps(params, indent=2))
    all_results.append((name, disp, thick))

# Also try clipping each variant (needed for meshing later)
print("\nClipping surfaces...")
for name, _, _ in VARIANTS:
    folder = OUT_ROOT / name
    try:
        import ukb.clip
        ukb.clip.main(folder=folder, case="ED")
        print(f"  {name}: clipped OK")
    except Exception as e:
        print(f"  {name}: clip failed ({e})")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"\n{'Variant':<15s} | {'Disp':>6s} | {'LV':>7s} | {'Sept':>7s} | {'RV FW':>7s}")
print("-" * 55)
for name, disp, thick in all_results:
    print(f"{name:<15s} | {disp:+5.1f} | {thick['LV']:7.2f} | "
          f"{thick['RV_septum']:7.2f} | {thick['RV_freewall']:7.2f}")

print(f"\nOpen in ParaView:")
print(f"  {OUT_ROOT}/{{variant}}/LV_ED.stl   — endocardial LV surface")
print(f"  {OUT_ROOT}/{{variant}}/RV_ED.stl   — endocardial RV surface")
print(f"  {OUT_ROOT}/{{variant}}/EPI_ED.stl  — epicardial surface (same for all)")
print(f"\nClipped surfaces (for meshing):")
print(f"  {OUT_ROOT}/{{variant}}/lv_clipped.ply")
print(f"  {OUT_ROOT}/{{variant}}/rv_clipped.ply")
print(f"  {OUT_ROOT}/{{variant}}/epi_clipped.ply")
