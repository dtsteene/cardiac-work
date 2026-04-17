#!/usr/bin/env python3
"""
plot_efficiency_thickness.py — η = W_true / W_proxy across thickness sweeps.

Same format as plot_efficiency_spectrum.py but with wall thickness on x-axis.
One figure per sweep (rvfw_healthy, rvfw_severe, lvfw_severe), each with
3 panels (LV, RV, Septum).
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/analysis/efficiency")
OUT.mkdir(parents=True, exist_ok=True)

THICKNESS_RUNS = [
    ("1029754", "healthy",  3.0, "rvfw", "2026-04-14"),
    ("1029755", "severe",   3.0, "rvfw", "2026-04-14"),
    ("1029756", "healthy",  6.0, "rvfw", "2026-04-14"),
    ("1029757", "severe",   6.0, "rvfw", "2026-04-14"),
    ("1029758", "healthy",  9.0, "rvfw", "2026-04-14"),
    ("1029759", "severe",   9.0, "rvfw", "2026-04-14"),
    ("1029760", "healthy", 12.0, "rvfw", "2026-04-14"),
    ("1029761", "severe",  12.0, "rvfw", "2026-04-14"),
    ("1033857", "severe",   4.5, "rvfw", "2026-04-15"),
    ("1033858", "severe",   7.5, "rvfw", "2026-04-15"),
    ("1033859", "severe",  10.5, "rvfw", "2026-04-15"),
    ("1033860", "severe",   3.0, "lvfw", "2026-04-15"),
    ("1033861", "severe",   6.0, "lvfw", "2026-04-15"),
    ("1033862", "severe",   9.0, "lvfw", "2026-04-15"),
    ("1033863", "severe",  12.0, "lvfw", "2026-04-15"),
]

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

SWEEP_TITLES = {
    "rvfw_healthy": "RVFW thickness, healthy circulation (n=4)",
    "rvfw_severe":  "RVFW thickness, severe PAH circulation (n=7)",
    "lvfw_severe":  "LVFW thickness, severe PAH circulation (n=4)",
}


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


# Group by sweep
groups = {}
for rid, sev, mm, fam, dt in THICKNESS_RUNS:
    key = f"{fam}_{sev}"
    root = Path(f"results/sims/{dt}")
    d = load_densities(root / f"UKB_1beats_run_{rid}" / "per_cell_data.npz")
    groups.setdefault(key, []).append({"mm": mm, "d": d})
for k in groups:
    groups[k].sort(key=lambda c: c["mm"])

# One figure per sweep
all_cv = []  # for summary bar chart
for sweep_key, cases in groups.items():
    xs = [c["mm"] for c in cases]
    xlabels = [f"+{mm:g}" for mm in xs]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for i, (tag, _, _, title) in enumerate(REGIONS):
        ax = axes[i]
        for pk, color, label, marker in PROXIES:
            etas = []
            for c in cases:
                wp = c["d"][tag][pk]
                wt = c["d"][tag]["W"]
                etas.append(wt / wp if abs(wp) > 0.001 else np.nan)
            etas_arr = np.array(etas)
            valid = etas_arr[np.isfinite(etas_arr)]
            cv = float(np.std(valid) / abs(np.mean(valid)) * 100) if len(valid) >= 2 else np.nan
            all_cv.append((sweep_key, tag, title, pk, label, color, cv))
            ax.plot(range(len(xs)), etas, "-", marker=marker, color=color,
                    lw=2.0, ms=8, label=f"{label}  (cv={cv:.1f}%)")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_xlabel("Added wall thickness (mm)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel(r"$\eta = W_{true}\;/\;W_{proxy}$", fontsize=11)
        # Cap y-axis if Trans blows up
        yvals = [e for e in etas if np.isfinite(e) and abs(e) < 50]
        if yvals:
            ymax = max(abs(v) for v in yvals) * 1.3
            # Only cap if there are outliers beyond this
            all_finite = [e for e in etas if np.isfinite(e)]
            if all_finite and max(abs(v) for v in all_finite) > ymax * 2:
                ax.set_ylim(-ymax * 0.1, ymax)
                ax.text(0.95, 0.95, f"(Trans off scale)", transform=ax.transAxes,
                        fontsize=8, ha="right", va="top", color="#2ca02c",
                        style="italic")
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"Proxy efficiency η — {SWEEP_TITLES.get(sweep_key, sweep_key)}\n"
        "Flat = proxy tracks proportional changes    "
        "Drifting = proxy calibration shifts with geometry",
        fontsize=12, fontweight="bold")
    fig.savefig(OUT / f"fig_efficiency_{sweep_key}.png", dpi=160,
                bbox_inches="tight")
    fig.savefig(OUT / f"fig_efficiency_{sweep_key}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_efficiency_{sweep_key}.png")

# ── Summary bar chart: cv per (sweep, region, proxy) for septum only ────────
# Focus on septum since that's the thesis region
fig2, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
sep_data = [(sw, pk, lbl, col, cv)
            for sw, tag, title, pk, lbl, col, cv in all_cv if tag == "Sep"]

# Group by sweep, then by proxy within sweep
sweep_order = ["rvfw_healthy", "rvfw_severe", "lvfw_severe"]
sweep_labels = {
    "rvfw_healthy": "RVFW\nhealthy",
    "rvfw_severe":  "RVFW\nsevere",
    "lvfw_severe":  "LVFW\nsevere",
}
proxy_order = ["PLV", "PRV", "Trans"]
bar_width = 0.25
x_base = 0
xtick_pos, xtick_lbl = [], []

for si, sw in enumerate(sweep_order):
    sw_items = [(pk, lbl, col, cv) for s, pk, lbl, col, cv in sep_data if s == sw]
    sw_dict = {pk: (lbl, col, cv) for pk, lbl, col, cv in sw_items}
    center = x_base + bar_width
    xtick_pos.append(center)
    xtick_lbl.append(sweep_labels[sw])
    for pi, pk in enumerate(proxy_order):
        if pk in sw_dict:
            lbl, col, cv = sw_dict[pk]
            xpos = x_base + pi * bar_width
            bar = ax.bar(xpos, min(cv, 100), bar_width * 0.85, color=col,
                         edgecolor="white", linewidth=1.2, alpha=0.85)
            txt = f"{cv:.0f}%" if cv < 100 else f"{cv:.0f}%↑"
            ax.text(xpos, min(cv, 100) + 1.5, txt, ha="center", fontsize=8,
                    fontweight="bold")
    x_base += len(proxy_order) * bar_width + 0.5

ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lbl, fontsize=10)
ax.axhline(10, color="gray", ls="--", lw=1.0, alpha=0.5)
ax.text(x_base - 0.3, 11, "10% threshold", fontsize=8, color="gray", ha="right")
ax.set_ylabel("Coefficient of variation of η (%)", fontsize=11)
ax.set_ylim(0, 105)
ax.grid(alpha=0.25, axis="y")

# Legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=col, edgecolor="white", label=lbl)
           for _, col, lbl, _ in PROXIES]
ax.legend(handles=handles, fontsize=10, loc="upper left")

ax.set_title("Septum: η stability across thickness sweeps — lower is better\n"
             "(cv < 10% = proxy tracks proportional changes faithfully)",
             fontsize=11, fontweight="bold")
fig2.savefig(OUT / "fig_cv_summary_thickness.png", dpi=160, bbox_inches="tight")
fig2.savefig(OUT / "fig_cv_summary_thickness.pdf", bbox_inches="tight")
plt.close(fig2)
print(f"Saved fig_cv_summary_thickness.png")

print("Done.")
