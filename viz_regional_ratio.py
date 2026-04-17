#!/usr/bin/env python3
"""
viz_regional_ratio.py — One-figure slide on the regional LV/RV ratio
finding: pressure-strain proxies cannot recover the true regional balance
in any pressure convention.

Loads the cascade_raw.npz written by analyze_cascade.py, computes the
proxy ratios under three conventions, and produces a single focused
panel ready for a slide or a discussion-chapter insert.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results/analysis/cascade")
d = np.load(OUT / "cascade_raw.npz")

# Wall volumes per tau-split territory (m^3); same shared mesh as cascade.
WALL_VOL_M3 = json.load(open("/tmp/ukb_wall_volumes.json"))
V_LV = WALL_VOL_M3["LV_tau_lap"]
V_RV = WALL_VOL_M3["RV_tau_lap"]
KPA  = 1e-3  # J/m^3 -> kPa = mJ/mL

# Work DENSITY at Level 0 (true tensor) per territory — Laplace tau split.
# Using density makes the LV/RV ratio a dimensionless work-per-muscle-volume
# contrast, which is the clinically meaningful quantity.
W_LV_true = d["LV_tau_lap_W0_per_step"].sum() / V_LV * KPA  # kPa
W_RV_true = d["RV_tau_lap_W0_per_step"].sum() / V_RV * KPA

# Proxy densities at Level 3 (P × dE_ff) under each convention
W_LV_pLV = d["LV_tau_lap_W3_per_step"].sum()       / V_LV * KPA  # correct: P_LV
W_LV_pRV = d["LV_tau_lap_W3_wrong_per_step"].sum() / V_LV * KPA  # wrong: P_RV
W_RV_pLV = d["RV_tau_lap_W3_wrong_per_step"].sum() / V_RV * KPA  # wrong: P_LV
W_RV_pRV = d["RV_tau_lap_W3_per_step"].sum()       / V_RV * KPA  # correct: P_RV

true_ratio       = W_LV_true / W_RV_true
pLV_mode_ratio   = W_LV_pLV / W_RV_pLV  # both territories use P_LV
pRV_mode_ratio   = W_LV_pRV / W_RV_pRV  # both territories use P_RV
correct_ratio    = W_LV_pLV / W_RV_pRV  # each territory uses its adjacent pressure

print(f"W_true density  LV = {W_LV_true:+.2f} kPa    RV = {W_RV_true:+.2f} kPa    "
      f"LV/RV = {true_ratio:.2f}")
print(f"P_LV mode    LV/RV = {pLV_mode_ratio:.2f}")
print(f"P_RV mode    LV/RV = {pRV_mode_ratio:.2f}")
print(f"Correct P    LV/RV = {correct_ratio:.2f}")

ratios = {
    "$P_{LV}$ everywhere":               pLV_mode_ratio,
    "$P_{RV}$ everywhere":               pRV_mode_ratio,
    "adjacent cavity":                    correct_ratio,
}

TRUE_RATIO = true_ratio

fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)

xs = np.arange(len(ratios))
values = list(ratios.values())
labels = list(ratios.keys())

colors = ["#1f77b4", "#9467bd", "#2ca02c"]
ax.bar(xs, values, width=0.56, color=colors,
       edgecolor="white", linewidth=1.2, zorder=3)

ax.axhline(TRUE_RATIO, color="black", lw=2.0, zorder=2,
           label=f"$\\mathbf{{S}}{{:}}\\mathbf{{E}}$ LV/RV = {TRUE_RATIO:.2f}")

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel(r"$W_{LV}/W_{RV}$", fontsize=12)
ax.set_ylim(0, max(max(values), TRUE_RATIO) * 1.20)
ax.grid(alpha=0.25, axis="y", zorder=1)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)

fig.savefig(OUT / "fig_regional_ratio.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "fig_regional_ratio.pdf", bbox_inches="tight")
print(f"Saved {OUT / 'fig_regional_ratio.png'}")
