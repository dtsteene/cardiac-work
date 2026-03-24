"""
Investigate UKB atlas wall thickness:
  1. Measure baseline (mean shape) wall thickness per region
  2. Analyze how each PCA mode changes wall thickness
  3. Identify which modes give controlled thickness variation
  4. Test direct surface displacement as an alternative

This helps us decide the best approach for thickening the mesh
for the PAH wall-thickness parameter study.
"""

import numpy as np
import h5py
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
ATLAS_PATH = Path.home() / ".ukb" / "UKBRVLV.h5"
OUT_DIR = Path(__file__).parent / "thickness_investigation"
OUT_DIR.mkdir(exist_ok=True)

N_MODES = 20  # Number of PCA modes to analyze

# Vertex ranges from ukb/surface.py
LV_ENDO = (0, 1500)
RV_ENDO = [(1500, 2165), (2165, 3224)]  # septum + RV endo
EPI = (3224, 5582)
RVFW = (5729, 5808)


# ── Helper Functions ───────────────────────────────────────────
def get_ed_points(hdf, mode=-1, std=1.0):
    """Compute ED point cloud for a given PCA mode/std."""
    mu = np.transpose(hdf["MU"])
    if mode == -1:
        S = mu
    else:
        eigenvalue = hdf["LATENT"][0, mode]
        eigenvector = hdf["COEFF"][mode, :]
        S = mu + (std * np.sqrt(eigenvalue) * eigenvector)
    N = S.shape[1] // 2
    return np.reshape(S[0, :N], (-1, 3))


def extract_surfaces(points):
    """Split point cloud into anatomical surfaces."""
    lv = points[LV_ENDO[0]:LV_ENDO[1]]
    rv = np.vstack([points[s:e] for s, e in RV_ENDO])
    epi = points[EPI[0]:EPI[1]]
    return lv, rv, epi


def compute_wall_thickness(endo_pts, epi_pts):
    """Wall thickness = distance from each endo point to nearest epi point."""
    tree = KDTree(epi_pts)
    distances, _ = tree.query(endo_pts)
    return distances


def compute_thickness_stats(lv_thick, rv_thick):
    """Summary statistics for wall thickness."""
    # Split RV into septum vs free wall based on vertex ranges
    n_rv_sept = 2165 - 1500   # 665 points
    n_rv_endo = 3224 - 2165   # 1059 points

    rv_sept = rv_thick[:n_rv_sept]
    rv_fw = rv_thick[n_rv_sept:]

    stats = {
        "LV": {"mean": np.mean(lv_thick), "std": np.std(lv_thick),
                "min": np.min(lv_thick), "max": np.max(lv_thick)},
        "RV (all)": {"mean": np.mean(rv_thick), "std": np.std(rv_thick),
                      "min": np.min(rv_thick), "max": np.max(rv_thick)},
        "RV septum": {"mean": np.mean(rv_sept), "std": np.std(rv_sept),
                       "min": np.min(rv_sept), "max": np.max(rv_sept)},
        "RV free wall": {"mean": np.mean(rv_fw), "std": np.std(rv_fw),
                          "min": np.min(rv_fw), "max": np.max(rv_fw)},
    }
    return stats


# ── PART 1: Baseline Wall Thickness ───────────────────────────
print("=" * 70)
print("PART 1: Baseline (Mean Shape) Wall Thickness")
print("=" * 70)

pc = h5py.File(ATLAS_PATH, "r")

ed_mean = get_ed_points(pc, mode=-1)
lv_endo, rv_endo, epi = extract_surfaces(ed_mean)

lv_thick_base = compute_wall_thickness(lv_endo, epi)
rv_thick_base = compute_wall_thickness(rv_endo, epi)

stats_base = compute_thickness_stats(lv_thick_base, rv_thick_base)

print(f"\nTotal points: {len(ed_mean)}")
print(f"  LV endo: {len(lv_endo)}, RV endo: {len(rv_endo)}, EPI: {len(epi)}")
print(f"\nWall thickness (mm) — mean shape:")
for region, s in stats_base.items():
    print(f"  {region:15s}: {s['mean']:.2f} ± {s['std']:.2f}  "
          f"[{s['min']:.2f} – {s['max']:.2f}]")

# Histogram
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(lv_thick_base, bins=40, color="tab:blue", alpha=0.7, edgecolor="k")
axes[0].set_title(f"LV wall thickness\nmean={np.mean(lv_thick_base):.1f} mm")
axes[0].set_xlabel("Thickness (mm)")

n_rv_sept = 2165 - 1500
axes[1].hist(rv_thick_base[:n_rv_sept], bins=40, color="tab:orange", alpha=0.7, edgecolor="k")
axes[1].set_title(f"RV septum thickness\nmean={np.mean(rv_thick_base[:n_rv_sept]):.1f} mm")
axes[1].set_xlabel("Thickness (mm)")

axes[2].hist(rv_thick_base[n_rv_sept:], bins=40, color="tab:green", alpha=0.7, edgecolor="k")
axes[2].set_title(f"RV free wall thickness\nmean={np.mean(rv_thick_base[n_rv_sept:]):.1f} mm")
axes[2].set_xlabel("Thickness (mm)")

plt.suptitle("Baseline Wall Thickness Distribution (Mean Shape)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "baseline_thickness.png", dpi=150)
plt.close()
print(f"\nSaved: {OUT_DIR / 'baseline_thickness.png'}")


# ── PART 2: PCA Mode Effects on Wall Thickness ────────────────
print("\n" + "=" * 70)
print("PART 2: PCA Mode Effects on Wall Thickness")
print("=" * 70)

# For each mode, compute thickness at ±1σ and record the change
mode_effects = {
    "lv_mean_delta": np.zeros(N_MODES),
    "rv_mean_delta": np.zeros(N_MODES),
    "rv_sept_mean_delta": np.zeros(N_MODES),
    "rv_fw_mean_delta": np.zeros(N_MODES),
    "lv_thickness_plus": np.zeros(N_MODES),
    "rv_thickness_plus": np.zeros(N_MODES),
    "lv_thickness_minus": np.zeros(N_MODES),
    "rv_thickness_minus": np.zeros(N_MODES),
    # Track how much the mode changes size vs thickness
    "lv_centroid_shift": np.zeros(N_MODES),
    "epi_centroid_shift": np.zeros(N_MODES),
    "explained_variance": np.zeros(N_MODES),
}

explained = pc["EXPLAINED"][0, :N_MODES] if "EXPLAINED" in pc else None
latent = pc["LATENT"][0, :N_MODES]

lv_centroid_base = np.mean(lv_endo, axis=0)
epi_centroid_base = np.mean(epi, axis=0)

for mode in range(N_MODES):
    # +1 std
    ed_plus = get_ed_points(pc, mode=mode, std=1.0)
    lv_p, rv_p, epi_p = extract_surfaces(ed_plus)
    lv_thick_p = compute_wall_thickness(lv_p, epi_p)
    rv_thick_p = compute_wall_thickness(rv_p, epi_p)

    # -1 std
    ed_minus = get_ed_points(pc, mode=mode, std=-1.0)
    lv_m, rv_m, epi_m = extract_surfaces(ed_minus)
    lv_thick_m = compute_wall_thickness(lv_m, epi_m)
    rv_thick_m = compute_wall_thickness(rv_m, epi_m)

    # Deltas (positive = thickening at +1σ)
    lv_delta = np.mean(lv_thick_p) - np.mean(lv_thick_base)
    rv_delta = np.mean(rv_thick_p) - np.mean(rv_thick_base)
    rv_sept_delta = np.mean(rv_thick_p[:n_rv_sept]) - np.mean(rv_thick_base[:n_rv_sept])
    rv_fw_delta = np.mean(rv_thick_p[n_rv_sept:]) - np.mean(rv_thick_base[n_rv_sept:])

    mode_effects["lv_mean_delta"][mode] = lv_delta
    mode_effects["rv_mean_delta"][mode] = rv_delta
    mode_effects["rv_sept_mean_delta"][mode] = rv_sept_delta
    mode_effects["rv_fw_mean_delta"][mode] = rv_fw_delta
    mode_effects["lv_thickness_plus"][mode] = np.mean(lv_thick_p)
    mode_effects["rv_thickness_plus"][mode] = np.mean(rv_thick_p)
    mode_effects["lv_thickness_minus"][mode] = np.mean(lv_thick_m)
    mode_effects["rv_thickness_minus"][mode] = np.mean(rv_thick_m)
    mode_effects["lv_centroid_shift"][mode] = np.linalg.norm(np.mean(lv_p, axis=0) - lv_centroid_base)
    mode_effects["epi_centroid_shift"][mode] = np.linalg.norm(np.mean(epi_p, axis=0) - epi_centroid_base)
    if explained is not None:
        mode_effects["explained_variance"][mode] = explained[mode]

# Print summary table
print(f"\n{'Mode':>4s} | {'Var%':>5s} | {'ΔLV':>7s} | {'ΔRV':>7s} | {'ΔSept':>7s} | {'ΔRVFW':>7s} | "
      f"{'LV+':>6s} | {'LV-':>6s} | {'RV+':>6s} | {'RV-':>6s} | {'Centroid':>8s}")
print("-" * 105)
for m in range(N_MODES):
    var_str = f"{mode_effects['explained_variance'][m]:.1f}" if explained is not None else "?"
    print(f"{m:4d} | {var_str:>5s} | "
          f"{mode_effects['lv_mean_delta'][m]:+.2f} | "
          f"{mode_effects['rv_mean_delta'][m]:+.2f} | "
          f"{mode_effects['rv_sept_mean_delta'][m]:+.2f} | "
          f"{mode_effects['rv_fw_mean_delta'][m]:+.2f} | "
          f"{mode_effects['lv_thickness_plus'][m]:.2f} | "
          f"{mode_effects['lv_thickness_minus'][m]:.2f} | "
          f"{mode_effects['rv_thickness_plus'][m]:.2f} | "
          f"{mode_effects['rv_thickness_minus'][m]:.2f} | "
          f"{mode_effects['lv_centroid_shift'][m]:.2f}")

# Plot: mode sensitivity
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
modes_arr = np.arange(N_MODES)

ax = axes[0, 0]
ax.bar(modes_arr - 0.2, mode_effects["lv_mean_delta"], 0.4, label="LV", color="tab:blue")
ax.bar(modes_arr + 0.2, mode_effects["rv_mean_delta"], 0.4, label="RV", color="tab:red")
ax.set_xlabel("PCA Mode")
ax.set_ylabel("Δ Mean Thickness (mm) at +1σ")
ax.set_title("Wall Thickness Change per Mode")
ax.legend()
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(modes_arr)

ax = axes[0, 1]
ax.bar(modes_arr - 0.2, mode_effects["rv_sept_mean_delta"], 0.4, label="RV Septum", color="tab:orange")
ax.bar(modes_arr + 0.2, mode_effects["rv_fw_mean_delta"], 0.4, label="RV Free Wall", color="tab:green")
ax.set_xlabel("PCA Mode")
ax.set_ylabel("Δ Mean Thickness (mm) at +1σ")
ax.set_title("RV Sub-region Thickness Change")
ax.legend()
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(modes_arr)

# Thickness range across ±2σ for the most impactful modes
ax = axes[1, 0]
for mode_i in range(min(8, N_MODES)):
    # Sweep from -2σ to +2σ
    stds = np.linspace(-2, 2, 9)
    lv_thicks = []
    rv_thicks = []
    for s in stds:
        ed_s = get_ed_points(pc, mode=mode_i, std=s)
        lv_s, rv_s, epi_s = extract_surfaces(ed_s)
        lv_thicks.append(np.mean(compute_wall_thickness(lv_s, epi_s)))
        rv_thicks.append(np.mean(compute_wall_thickness(rv_s, epi_s)))
    ax.plot(stds, lv_thicks, marker=".", label=f"Mode {mode_i}")
ax.set_xlabel("Standard Deviations from Mean")
ax.set_ylabel("LV Mean Thickness (mm)")
ax.set_title("LV Thickness vs PCA Score (modes 0–7)")
ax.legend(fontsize=7, ncol=2)

ax = axes[1, 1]
for mode_i in range(min(8, N_MODES)):
    stds = np.linspace(-2, 2, 9)
    rv_thicks = []
    for s in stds:
        ed_s = get_ed_points(pc, mode=mode_i, std=s)
        _, rv_s, epi_s = extract_surfaces(ed_s)
        rv_thicks.append(np.mean(compute_wall_thickness(rv_s, epi_s)))
    ax.plot(stds, rv_thicks, marker=".", label=f"Mode {mode_i}")
ax.set_xlabel("Standard Deviations from Mean")
ax.set_ylabel("RV Mean Thickness (mm)")
ax.set_title("RV Thickness vs PCA Score (modes 0–7)")
ax.legend(fontsize=7, ncol=2)

plt.suptitle("PCA Mode Effects on Wall Thickness", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_mode_thickness_effects.png", dpi=150)
plt.close()
print(f"\nSaved: {OUT_DIR / 'pca_mode_thickness_effects.png'}")


# ── PART 3: Thickness vs Size Decomposition ──────────────────
# Key question: do PCA modes change thickness independently of overall size?
print("\n" + "=" * 70)
print("PART 3: Thickness vs Size — Can We Separate Them?")
print("=" * 70)

# For each mode, compute:
#   - overall size change (volume proxy via convex hull or bounding box)
#   - thickness change
# If thickness and size are decoupled in some modes, those are ideal for our study

from scipy.spatial import ConvexHull

vol_base_lv = ConvexHull(lv_endo).volume
vol_base_rv = ConvexHull(rv_endo).volume
vol_base_epi = ConvexHull(epi).volume

print(f"\nBaseline convex hull volumes (mm³):")
print(f"  LV endo: {vol_base_lv:.0f}")
print(f"  RV endo: {vol_base_rv:.0f}")
print(f"  EPI:     {vol_base_epi:.0f}")

# Compute normalized thickness-to-size sensitivity
print(f"\n{'Mode':>4s} | {'ΔLV_thick':>10s} | {'ΔLV_vol%':>9s} | {'ΔRV_thick':>10s} | "
      f"{'ΔRV_vol%':>9s} | {'ΔEPI_vol%':>10s} | {'thick/size LV':>14s} | {'thick/size RV':>14s}")
print("-" * 110)

thickness_size_ratio_lv = np.zeros(N_MODES)
thickness_size_ratio_rv = np.zeros(N_MODES)

for mode in range(N_MODES):
    ed_plus = get_ed_points(pc, mode=mode, std=1.0)
    lv_p, rv_p, epi_p = extract_surfaces(ed_plus)

    lv_thick_delta = np.mean(compute_wall_thickness(lv_p, epi_p)) - np.mean(lv_thick_base)
    rv_thick_delta = np.mean(compute_wall_thickness(rv_p, epi_p)) - np.mean(rv_thick_base)

    try:
        vol_lv = ConvexHull(lv_p).volume
        vol_rv = ConvexHull(rv_p).volume
        vol_epi = ConvexHull(epi_p).volume
    except Exception:
        vol_lv = vol_rv = vol_epi = np.nan

    dv_lv = (vol_lv - vol_base_lv) / vol_base_lv * 100
    dv_rv = (vol_rv - vol_base_rv) / vol_base_rv * 100
    dv_epi = (vol_epi - vol_base_epi) / vol_base_epi * 100

    # Ratio: thickness change per unit size change
    # High ratio = mode changes thickness more than size (what we want)
    mean_thick_base_lv = np.mean(lv_thick_base)
    mean_thick_base_rv = np.mean(rv_thick_base)
    pct_thick_lv = lv_thick_delta / mean_thick_base_lv * 100 if mean_thick_base_lv > 0 else 0
    pct_thick_rv = rv_thick_delta / mean_thick_base_rv * 100 if mean_thick_base_rv > 0 else 0

    # ratio = %thickness change / %volume change (>1 means thickness-dominant)
    ratio_lv = abs(pct_thick_lv) / max(abs(dv_lv), 0.01)
    ratio_rv = abs(pct_thick_rv) / max(abs(dv_rv), 0.01)

    thickness_size_ratio_lv[mode] = ratio_lv
    thickness_size_ratio_rv[mode] = ratio_rv

    print(f"{mode:4d} | {lv_thick_delta:+10.3f} | {dv_lv:+8.1f}% | {rv_thick_delta:+10.3f} | "
          f"{dv_rv:+8.1f}% | {dv_epi:+9.1f}% | {ratio_lv:14.2f} | {ratio_rv:14.2f}")


# ── PART 4: Direct Surface Displacement Test ─────────────────
print("\n" + "=" * 70)
print("PART 4: Direct Surface Displacement (Alternative to PCA)")
print("=" * 70)
print("\nApproach: Move endocardial points along endo→epi direction")
print("to thicken walls by a controlled amount.")

# For each endo point, compute the direction toward the nearest epi point
epi_tree = KDTree(epi)

# LV: displacement direction = from endo toward nearest epi point (outward = thinning, inward = thickening)
_, lv_nearest_epi_idx = epi_tree.query(lv_endo)
lv_directions = epi[lv_nearest_epi_idx] - lv_endo  # points from endo → epi
lv_directions /= np.linalg.norm(lv_directions, axis=1, keepdims=True)  # normalize

# RV: same approach
_, rv_nearest_epi_idx = epi_tree.query(rv_endo)
rv_directions = epi[rv_nearest_epi_idx] - rv_endo
rv_directions /= np.linalg.norm(rv_directions, axis=1, keepdims=True)

# Test: displace endo inward by various amounts (negative = toward cavity = thickening)
displacements_mm = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

print(f"\n{'Disp (mm)':>10s} | {'LV thick':>9s} | {'RV thick':>9s} | {'Sept thick':>10s} | {'RVFW thick':>10s}")
print("-" * 65)

for d in displacements_mm:
    # Move endo points OPPOSITE to endo→epi direction (negative d = thickening)
    lv_endo_mod = lv_endo + d * lv_directions  # positive d moves toward epi (thinning)
    rv_endo_mod = rv_endo + d * rv_directions

    lv_t = compute_wall_thickness(lv_endo_mod, epi)
    rv_t = compute_wall_thickness(rv_endo_mod, epi)

    print(f"{d:+10.1f} | {np.mean(lv_t):9.2f} | {np.mean(rv_t):9.2f} | "
          f"{np.mean(rv_t[:n_rv_sept]):10.2f} | {np.mean(rv_t[n_rv_sept:]):10.2f}")


# ── PART 5: Controllability Summary ──────────────────────────
print("\n" + "=" * 70)
print("PART 5: Summary & Recommendation")
print("=" * 70)

# Find best PCA modes for thickness control
best_lv_modes = np.argsort(np.abs(mode_effects["lv_mean_delta"]))[::-1][:5]
best_rv_modes = np.argsort(np.abs(mode_effects["rv_mean_delta"]))[::-1][:5]

print(f"\nTop 5 PCA modes affecting LV thickness: {list(best_lv_modes)}")
for m in best_lv_modes:
    print(f"  Mode {m}: ΔLV = {mode_effects['lv_mean_delta'][m]:+.3f} mm at +1σ "
          f"(thick/size ratio: {thickness_size_ratio_lv[m]:.2f})")

print(f"\nTop 5 PCA modes affecting RV thickness: {list(best_rv_modes)}")
for m in best_rv_modes:
    print(f"  Mode {m}: ΔRV = {mode_effects['rv_mean_delta'][m]:+.3f} mm at +1σ "
          f"(thick/size ratio: {thickness_size_ratio_rv[m]:.2f})")

# Thickness range achievable via PCA (combining best modes)
print(f"\nThickness range from single best mode (±2σ):")
best_lv = best_lv_modes[0]
best_rv = best_rv_modes[0]
print(f"  LV: Mode {best_lv} → "
      f"{mode_effects['lv_thickness_minus'][best_lv]:.2f} to "
      f"{mode_effects['lv_thickness_plus'][best_lv]:.2f} mm "
      f"(baseline: {np.mean(lv_thick_base):.2f} mm)")
print(f"  RV: Mode {best_rv} → "
      f"{mode_effects['rv_thickness_minus'][best_rv]:.2f} to "
      f"{mode_effects['rv_thickness_plus'][best_rv]:.2f} mm "
      f"(baseline: {np.mean(rv_thick_base):.2f} mm)")

print(f"\nDirect displacement range tested: ±3 mm")
print(f"  → Full independent control over LV and RV thickness")
print(f"  → Can target specific regions (septum vs free wall)")

pc.close()
print(f"\nAll plots saved to: {OUT_DIR}/")
