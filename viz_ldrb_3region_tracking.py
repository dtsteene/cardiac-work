#!/usr/bin/env python3
"""
viz_ldrb_3region_tracking.py — Section B supplementary plot:
Proxy tracking in the LDRB three-region split (LV free wall, Septum,
RV free wall).

Point of this figure: once we carve the septum out as its own region,
the LV and RV free walls become *cleaner* than they were in the binary
tau split. Specifically, the RV free wall should show higher r(W_true,
P_RV) because the septal cells that were previously in the "RV
territory" under the tau split — and dragging its correlation down —
are now in their own septum region.

Output: fig_ldrb_3region_tracking.png — 1×3 dual-axis panels
(LV free wall | Septum | RV free wall), each with W_true on the left
axis and the three canonical proxies on the right.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

OUT = Path("results/analysis/spectrum")
OUT.mkdir(parents=True, exist_ok=True)

SPECTRUM = [
    ("healthy",         "Borderline PH", "1020849", 30.6),
    ("mild",            "Mild",          "1020851", 38.2),
    ("moderate",        "Moderate",      "1020852", 55.4),
    ("moderate_severe", "Mod–severe",    "1020853", 62.5),
    ("severe",          "Severe",        "1020854", 70.8),
    ("very_severe",     "Very severe",   "1020855", 85.0),
    ("end_stage",       "End-stage",     "1020856", 88.4),
]
ROOT = Path("results/sims/2026-04-12")

REGION_MASK = {
    "LV free wall": lambda pc: pc["region_tags"] == 1,
    "Septum":       lambda pc: pc["region_tags"] == 3,
    "RV free wall": lambda pc: pc["region_tags"] == 2,
}


def safe_r(x, y):
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 3:
        return float("nan")
    return pearsonr(x, y)[0]


# Collect per-region densities (kPa) across the spectrum
rows = {reg: {"W_true": [], "W_PLV": [], "W_PRV": [], "W_Trans": []}
        for reg in REGION_MASK}
KPA = 1e-3  # J/m^3 -> kPa = mJ/mL
for _sev, _lbl, run_id, _rve in SPECTRUM:
    pc = np.load(ROOT / f"UKB_6beats_run_{run_id}" / "per_cell_data.npz",
                  allow_pickle=True)
    cv = pc["cell_volumes"]
    for reg, mask_fn in REGION_MASK.items():
        m = mask_fn(pc)
        V = float(cv[m].sum())
        rows[reg]["W_true"].append(float(pc["w_total"][m].sum()) / V * KPA)
        rows[reg]["W_PLV"].append(float(pc["proxy_PLV_ll"][m].sum()) / V * KPA)
        rows[reg]["W_PRV"].append(float(pc["proxy_PRV_ll"][m].sum()) / V * KPA)
        rows[reg]["W_Trans"].append(float(pc["proxy_Trans_ll"][m].sum()) / V * KPA)

# Pearson r per region per proxy
print(f"{'region':<14} {'r(W_true,P_LV)':>18} {'r(W_true,P_RV)':>18} "
      f"{'r(W_true,Trans)':>18}")
r_values = {}
for reg in REGION_MASK:
    Wt = np.array(rows[reg]["W_true"])
    for p in ("PLV", "PRV", "Trans"):
        r_values[f"{reg}_{p}"] = safe_r(Wt, np.array(rows[reg][f"W_{p}"]))
    print(f"{reg:<14} {r_values[f'{reg}_PLV']:>+18.4f} "
          f"{r_values[f'{reg}_PRV']:>+18.4f} "
          f"{r_values[f'{reg}_Trans']:>+18.4f}")

# ── Figure ──────────────────────────────────────────────────────────────────
PROXY_STYLE = {
    "PLV":   dict(color="#1f77b4", marker="o",  label=r"$P_{LV}$"),
    "PRV":   dict(color="#d62728", marker="s",  label=r"$P_{RV}$"),
    "Trans": dict(color="#2ca02c", marker="^",  label=r"$P_{LV}-P_{RV}$"),
}
X = np.arange(len(SPECTRUM))
xtick_labels = [f"{label}\n({rve:.0f} mmHg)"
                for _sev, label, _id, rve in SPECTRUM]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
for j, reg in enumerate(REGION_MASK):
    ax = axes[j]
    Wt = np.array(rows[reg]["W_true"])

    # Left axis: W_true density (black)
    ax.plot(X, Wt, "-D", color="black", lw=2.6, ms=8,
            label=r"$W_{true}$", zorder=5)
    ax.set_ylabel(r"$W_{true}$ density (kPa = mJ/mL)",
                  color="black", fontsize=11)
    ax.tick_params(axis="y", labelcolor="black")
    ax.set_xticks(X)
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title(reg, fontsize=13, fontweight="bold")

    # Right axis: proxies
    ax2 = ax.twinx()
    for proxy, style in PROXY_STYLE.items():
        vals = np.array(rows[reg][f"W_{proxy}"])
        r_val = r_values[f"{reg}_{proxy}"]
        ax2.plot(X, vals, "-", lw=1.7, ms=6, alpha=0.95,
                 color=style["color"], marker=style["marker"],
                 label=f"{style['label']}  (r={r_val:+.2f})")
    ax2.set_ylabel(r"Proxy density (kPa)",
                   color="dimgray", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="dimgray")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=9, framealpha=0.95,
               handlelength=1.8)

fig.suptitle("Proxy tracking in LDRB three-region split",
              fontsize=12, fontweight="bold")
fig.savefig(OUT / "fig_ldrb_3region_tracking.png", dpi=160,
             bbox_inches="tight")
fig.savefig(OUT / "fig_ldrb_3region_tracking.pdf", bbox_inches="tight")
print(f"\nSaved {OUT / 'fig_ldrb_3region_tracking.png'}")

np.savez(OUT / "ldrb_3region_raw.npz",
          **{f"{reg}_{k}": np.array(v)
             for reg, d in rows.items() for k, v in d.items()},
          **{f"r_{k}": v for k, v in r_values.items()})
