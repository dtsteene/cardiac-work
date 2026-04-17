#!/usr/bin/env python3
"""
plot_efficiency_spectrum.py — η = W_true / W_proxy across the PAH spectrum.

Shows whether the proxy captures a stable fraction of true work as disease
severity increases. A flat η means proportional change-tracking; a drifting
η means the proxy's "calibration" changes with the thing you're measuring.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/analysis/efficiency")
OUT.mkdir(parents=True, exist_ok=True)

SPECTRUM = [
    ("healthy",         "Borderline PH", "1020849",  30.6),
    ("mild",            "Mild",          "1020851",  38.2),
    ("moderate",        "Moderate",      "1020852",  55.4),
    ("moderate_severe", "Mod–severe",    "1020853",  62.5),
    ("severe",          "Severe",        "1020854",  70.8),
    ("very_severe",     "Very severe",   "1020855",  85.0),
    ("end_stage",       "End-stage",     "1020856",  88.4),
]
ROOT = Path("results/sims/2026-04-12")

PROXIES = [
    ("PLV",   "#1f77b4", r"$P_{LV}$",         "o"),
    ("PRV",   "#d62728", r"$P_{RV}$",         "s"),
    ("Trans", "#2ca02c", r"$P_{LV}-P_{RV}$",  "^"),
]
REGIONS = [
    ("LV",  "region_tags", 1,    "LV free wall"),
    ("RV",  "region_tags", 2,    "RV free wall"),
    ("Sep", "is_ldrb_septum", True, "Septum (LDRB)"),
]


def load_densities(pc_path):
    pc = np.load(pc_path, allow_pickle=True)
    cv = pc["cell_volumes"]
    KPA = 1e-3
    out = {}
    for tag, key, val, title in REGIONS:
        if key == "region_tags":
            mask = pc[key] == val
        else:
            mask = pc[key].astype(bool)
        V = float(cv[mask].sum())
        out[tag] = {
            "W":     float(pc["w_total"][mask].sum()) / V * KPA,
            "PLV":   float(pc["proxy_PLV_ll"][mask].sum()) / V * KPA,
            "PRV":   float(pc["proxy_PRV_ll"][mask].sum()) / V * KPA,
            "Trans": float((pc["proxy_PLV_ll"][mask].sum()
                            - pc["proxy_PRV_ll"][mask].sum())) / V * KPA,
        }
    return out


# Load all 7 cases
cases = []
for sev, label, rid, rvesp in SPECTRUM:
    d = load_densities(ROOT / f"UKB_6beats_run_{rid}" / "per_cell_data.npz")
    cases.append({"label": label, "rvesp": rvesp, "d": d})

x = np.arange(len(cases))
xlabels = [f"{c['label']}\n({c['rvesp']:.0f} mmHg)" for c in cases]

# ── Figure 1: η across spectrum, one panel per region ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)

for i, (tag, _, _, title) in enumerate(REGIONS):
    ax = axes[i]
    for pk, color, label, marker in PROXIES:
        etas = []
        for c in cases:
            wp = c["d"][tag][pk]
            wt = c["d"][tag]["W"]
            etas.append(wt / wp if abs(wp) > 0.001 else np.nan)
        etas = np.array(etas)
        valid = etas[np.isfinite(etas)]
        cv = float(np.std(valid) / abs(np.mean(valid)) * 100) if len(valid) >= 2 else np.nan
        ax.plot(x, etas, "-", marker=marker, color=color, lw=2.0, ms=8,
                label=f"{label}  (cv={cv:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    if i == 0:
        ax.set_ylabel(r"$\eta = W_{true}\;/\;W_{proxy}$", fontsize=11)
    ax.legend(fontsize=9, loc="best")

fig.suptitle(
    r"Proxy efficiency $\eta$ across PAH severity spectrum"
    "\nFlat = proxy tracks proportional changes faithfully    "
    "Drifting = proxy calibration shifts with disease",
    fontsize=12, fontweight="bold")
fig.savefig(OUT / "fig_efficiency_spectrum.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "fig_efficiency_spectrum.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'fig_efficiency_spectrum.png'}")

# ── Figure 2: cv summary bar chart ─────────────────────────────────────────
# Compact view: one bar per (region, proxy), colored by cv quality.
fig2, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
bar_data = []
for tag, _, _, title in REGIONS:
    for pk, color, label, _ in PROXIES:
        etas = []
        for c in cases:
            wp = c["d"][tag][pk]
            wt = c["d"][tag]["W"]
            etas.append(wt / wp if abs(wp) > 0.001 else np.nan)
        valid = np.array(etas)[np.isfinite(etas)]
        cv = float(np.std(valid) / abs(np.mean(valid)) * 100) if len(valid) >= 2 else 100.0
        bar_data.append((f"{title}\n{label}", cv, color))

xs = np.arange(len(bar_data))
colors = [b[2] for b in bar_data]
cvs = [b[1] for b in bar_data]
bars = ax.bar(xs, cvs, color=colors, edgecolor="white", linewidth=1.2,
              alpha=0.85)
ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in bar_data], fontsize=8, ha="center")
ax.set_ylabel("Coefficient of variation of η (%)", fontsize=11)
ax.axhline(10, color="gray", ls="--", lw=1.0, alpha=0.5)
ax.text(len(bar_data) - 0.5, 11, "10% threshold", fontsize=8,
        color="gray", ha="right")
for i, v in enumerate(cvs):
    ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8, fontweight="bold")
ax.set_ylim(0, max(cvs) * 1.15)
ax.grid(alpha=0.25, axis="y")
ax.set_title("η stability across PAH spectrum — lower is better\n"
             "(cv < 10% = proxy tracks proportional changes faithfully)",
             fontsize=11, fontweight="bold")
fig2.savefig(OUT / "fig_cv_summary_spectrum.png", dpi=160, bbox_inches="tight")
fig2.savefig(OUT / "fig_cv_summary_spectrum.pdf", bbox_inches="tight")
plt.close(fig2)
print(f"Saved {OUT / 'fig_cv_summary_spectrum.png'}")

print("Done.")
