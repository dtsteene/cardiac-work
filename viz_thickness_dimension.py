#!/usr/bin/env python3
"""
viz_thickness_dimension.py — Show the listener what the thickness sweep
is actually perturbing. 4 mesh cross-sections side by side (rvfw +3 mm
through +12 mm), each annotated with its wall volume and the fractional
increase, with cells coloured by side (LV half vs RV half via the
Euclidean tau split) so the RV wall visibly thickens across the row.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/analysis/thickness_v2")
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("rvfw_03mm", "+3 mm", "1029754"),
    ("rvfw_06mm", "+6 mm", "1029756"),
    ("rvfw_09mm", "+9 mm", "1029758"),
    ("rvfw_12mm", "+12 mm", "1029760"),
]
ROOT = Path("results/sims/2026-04-14")


def load(run_id):
    return np.load(ROOT / f"UKB_1beats_run_{run_id}" / "per_cell_data.npz",
                    allow_pickle=True)


# Pick the apex-base axis once (longest extent) using the first variant
pc0 = load(VARIANTS[0][2])
ext = pc0["centroids"].max(axis=0) - pc0["centroids"].min(axis=0)
ax_long = int(np.argmax(ext))
xy_axes = [a for a in (0, 1, 2) if a != ax_long]
print(f"Apex-base axis = {ax_long}, short-axis coords are axes {xy_axes}")

# Short-axis slab at mid-ventricle, same Z interval for all variants
z_mids = []
for _v, _l, rid in VARIANTS:
    c = load(rid)["centroids"]
    z_mids.append(np.median(c[:, ax_long]))
Z_MID = float(np.mean(z_mids))
SLAB_HALF = 0.006  # ±6 mm

# Figure setup
fig, axes = plt.subplots(1, len(VARIANTS), figsize=(15, 4.2),
                          sharex=True, sharey=True, constrained_layout=True)

# Collect wall volumes for the delta annotation
wall_volumes = []
for v_i, (variant, label, run_id) in enumerate(VARIANTS):
    pc = load(run_id)
    cv = pc["cell_volumes"]
    V = float(cv.sum() * 1e6)  # mL
    wall_volumes.append(V)

V0 = wall_volumes[0]

# Find global axis limits for consistent framing across panels
all_x, all_y = [], []
for _v, _l, rid in VARIANTS:
    c = load(rid)["centroids"]
    slab = np.abs(c[:, ax_long] - Z_MID) < SLAB_HALF
    all_x.append(c[slab, xy_axes[0]] * 1000.0)
    all_y.append(c[slab, xy_axes[1]] * 1000.0)
xmin = min(a.min() for a in all_x) - 4
xmax = max(a.max() for a in all_x) + 4
ymin = min(a.min() for a in all_y) - 4
ymax = max(a.max() for a in all_y) + 4

for v_i, (variant, label, run_id) in enumerate(VARIANTS):
    ax = axes[v_i]
    pc = load(run_id)
    c = pc["centroids"]
    slab = np.abs(c[:, ax_long] - Z_MID) < SLAB_HALF
    tau = pc["tau"]              # Euclidean d_LV / (d_LV + d_RV)

    xs = c[slab, xy_axes[0]] * 1000.0
    ys = c[slab, xy_axes[1]] * 1000.0
    tau_slab = tau[slab]
    lv_side = tau_slab < 0.5
    rv_side = ~lv_side

    ax.scatter(xs[lv_side], ys[lv_side], s=35, c="#1f77b4",
                edgecolors="none", alpha=0.80, label="LV side")
    ax.scatter(xs[rv_side], ys[rv_side], s=35, c="#d62728",
                edgecolors="none", alpha=0.80, label="RV side")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    V_mL = wall_volumes[v_i]
    delta_pct = 100 * (V_mL - V0) / V0
    ax.set_title(
        f"RVfw {label}\n"
        f"$V_{{wall}}$ = {V_mL:.0f} mL  ({delta_pct:+.0f} % vs +3)",
        fontsize=11, fontweight="bold")

axes[0].legend(loc="upper right", fontsize=9, framealpha=0.95)

fig.suptitle("What the wall thickness sweep actually varies "
              "(mid-ventricle short-axis slice, same scale)",
              fontsize=12, fontweight="bold")
fig.savefig(OUT / "fig_thickness_dimension.png", dpi=160,
             bbox_inches="tight")
fig.savefig(OUT / "fig_thickness_dimension.pdf", bbox_inches="tight")
print(f"Saved {OUT / 'fig_thickness_dimension.png'}")
print(f"Wall volumes: {wall_volumes}")
