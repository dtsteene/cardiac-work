#!/usr/bin/env python3
"""
viz_free_wall_tracking.py — Section B, plot 2: proxy tracking in the LV
and RV "territories" (Laplace tau split, no septum carved out).

The opening argument before the septum gets introduced: at a region that
faces ONE cavity with an unambiguous pressure choice, does the clinical
pressure-strain proxy track W_true across the severity spectrum? If yes,
the proxy is a usable method — and then (next slide) the septum is a
legitimate question.

For each of the 7 spectrum cases (2026-04-12 borderline-PH through
end-stage, canonical-tagged), cells are split by the Laplace lv_rv_scalar
at 0.5 into:
    LV_tau_lap  = {c : lv_rv_scalar(c) > 0.5}      ← closer to LV endo
    RV_tau_lap  = {c : lv_rv_scalar(c) <= 0.5}     ← closer to RV endo
Note: compute_per_cell.py stores lv_rv_scalar with LV=1, RV=0, so LV_tau
is the > 0.5 side. (In analyze_spectrum.py we flip to match
metrics_calculator's LV=0 convention; here we use the compute_per_cell
convention directly on the per_cell files.)

Each region's W_true and W_proxy (PLV, PRV, Trans) are the regional sums
of the per-cell fields over the 7 spectrum cases → 7-point Pearson r per
proxy per region.

Output: fig_free_wall_tracking.png — 1×2 dual-axis panels (LV | RV),
black diamond W_true on the left axis, coloured proxies on the right.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

OUT = Path("results/analysis/spectrum")
OUT.mkdir(parents=True, exist_ok=True)

# Ordered spectrum with achieved RV ESP (mmHg) for secondary ticks
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


def safe_r(x, y):
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 3:
        return float("nan")
    return pearsonr(x, y)[0]


# Per-case regional densities (kPa = mJ/mL)
rows = {"W_true_LV":  [], "W_PLV_LV":  [], "W_PRV_LV":  [], "W_Trans_LV":  [],
        "W_true_RV":  [], "W_PLV_RV":  [], "W_PRV_RV":  [], "W_Trans_RV":  []}
KPA = 1e-3  # J/m^3 -> kPa
for sev, label, run_id, _rve in SPECTRUM:
    pc = np.load(ROOT / f"UKB_6beats_run_{run_id}" / "per_cell_data.npz",
                  allow_pickle=True)
    cv = pc["cell_volumes"]
    # Laplace LV side = lv_rv_scalar > 0.5 (compute_per_cell convention: LV=1)
    lv_mask = pc["lv_rv_scalar"] > 0.5
    rv_mask = ~lv_mask
    for mask, suffix in [(lv_mask, "LV"), (rv_mask, "RV")]:
        V = float(cv[mask].sum())
        rows[f"W_true_{suffix}"].append(float(pc["w_total"][mask].sum()) / V * KPA)
        rows[f"W_PLV_{suffix}"].append(float(pc["proxy_PLV_ll"][mask].sum()) / V * KPA)
        rows[f"W_PRV_{suffix}"].append(float(pc["proxy_PRV_ll"][mask].sum()) / V * KPA)
        rows[f"W_Trans_{suffix}"].append(float(pc["proxy_Trans_ll"][mask].sum()) / V * KPA)

arrs = {k: np.array(v) for k, v in rows.items()}

# Pearson r per (region, proxy) across the 7 severities
r_values = {}
for region in ("LV", "RV"):
    Wt = arrs[f"W_true_{region}"]
    for p in ("PLV", "PRV", "Trans"):
        r_values[f"{region}_{p}"] = safe_r(Wt, arrs[f"W_{p}_{region}"])

print("Pearson r (7 severities, Laplace tau split, no septum carved out):")
print(f"{'region':<10} {'r(W_true,W_PLV)':>18} {'r(W_true,W_PRV)':>18} "
      f"{'r(W_true,W_Trans)':>20}")
for reg in ("LV", "RV"):
    print(f"{reg:<10} {r_values[f'{reg}_PLV']:>+18.4f} "
          f"{r_values[f'{reg}_PRV']:>+18.4f} {r_values[f'{reg}_Trans']:>+20.4f}")

# ── Figure ──────────────────────────────────────────────────────────────────
PROXY_STYLE = {
    "PLV":   dict(color="#1f77b4", marker="o",  label=r"$P_{LV}$"),
    "PRV":   dict(color="#d62728", marker="s",  label=r"$P_{RV}$"),
    "Trans": dict(color="#2ca02c", marker="^",  label=r"$P_{LV}-P_{RV}$"),
}
X = np.arange(len(SPECTRUM))
xtick_labels = [f"{label}\n({rve:.0f} mmHg)"
                for _sev, label, _id, rve in SPECTRUM]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
for j, region in enumerate(("LV", "RV")):
    ax = axes[j]
    Wt = arrs[f"W_true_{region}"]

    # Left axis: W_true density (black)
    ax.plot(X, Wt, "-D", color="black", lw=2.6, ms=8,
            label=r"$W_{true}$", zorder=5)
    ax.set_ylabel(r"$W_{true}$ density (kPa = mJ/mL)",
                  color="black", fontsize=11)
    ax.tick_params(axis="y", labelcolor="black")
    ax.set_xticks(X)
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title(region, fontsize=13, fontweight="bold")

    # Right axis: proxies
    ax2 = ax.twinx()
    for proxy, style in PROXY_STYLE.items():
        vals = arrs[f"W_{proxy}_{region}"]
        r_val = r_values[f"{region}_{proxy}"]
        ax2.plot(X, vals, "-", lw=1.7, ms=6, alpha=0.95,
                 color=style["color"], marker=style["marker"],
                 label=f"{style['label']}  (r={r_val:+.2f})")
    ax2.set_ylabel(r"Proxy density (kPa)",
                   color="dimgray", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="dimgray")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=9, framealpha=0.95,
               handlelength=1.8)

fig.suptitle("LV / RV territory tracking  (binary tau split, no septum)",
              fontsize=12, fontweight="bold")
fig.savefig(OUT / "fig_free_wall_tracking.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "fig_free_wall_tracking.pdf", bbox_inches="tight")
print(f"\nSaved {OUT / 'fig_free_wall_tracking.png'}")

# Also save raw for downstream reuse
np.savez(OUT / "free_wall_raw.npz", **arrs, **{f"r_{k}": v for k, v in r_values.items()})
print(f"Saved {OUT / 'free_wall_raw.npz'}")
