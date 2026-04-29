#!/usr/bin/env python3
"""
run_handover_plots_v12.py — patch old-style handover plot scripts for the
v12 8-case spectrum and produce figures for both variants.

Usage:
  python run_handover_plots_v12.py lin
  python run_handover_plots_v12.py exp

Produces, under results/handover/handover_{lin|exp}/spectrum_results/:
  loops_plv.{png,pdf}
  loops_regional.{png,pdf}
  loop_comparison.{png,pdf}
  pv_loops_0d.{png,pdf}
  pv_loops_solver.{png,pdf}
  scatter_spectrum.{png,pdf}
  work_traces.{png,pdf}

Plot style matches handover_old:
  • Pearson r and r^2 (= squared Pearson), with-intercept OLS fit.
  • No Q metric.
  • Strain traces on loop plots are shifted so that E_ff(ED)=0 and
    E_ll(ED)=0, matching the clinical convention (ED as reference).
    The proxy/internal work integrals are reference-invariant over a
    closed cycle, so this change is visual only — correlations and Q
    remain unchanged.
"""
import sys, csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import pearsonr

if len(sys.argv) != 2 or sys.argv[1] not in ("lin","exp"):
    sys.exit("usage: run_handover_plots_v12.py {lin|exp}")
VARIANT = sys.argv[1]

ROOT = Path("/global/D1/homes/dtsteene/cardiac-work/results/handover")
BASE = ROOT / f"handover_{VARIANT}"
DATA = BASE / "data"
OUT  = BASE / "spectrum_results"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [f"C{i+1}" for i in range(8)]
REGIONS = ["LV", "RV", "Septum"]
PA_TO_KPA = 1e-3
MMHG_TO_KPA = 0.133322
KPA = 1e-3

# Load per-case achieved RV_ESP from the summary CSV for the colormap.
summary = pd.read_csv(BASE / "hemodynamic_summary.csv")
summary = summary.set_index("case_id")
RV_ESP = {cid: float(summary.loc[cid, "RV_ESP_mmHg"]) for cid in CASES}

cmap = plt.cm.coolwarm
norm = Normalize(vmin=25, vmax=95)

PROXIES = [
    ("PLV",   "#1f77b4", "$P_{LV}$",         "o"),
    ("PRV",   "#d62728", "$P_{RV}$",         "s"),
    ("Trans", "#2ca02c", "$P_{LV}-P_{RV}$",  "^"),
]


def clean_spines(ax, right=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right" if not right else "left"].set_visible(False)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)


def load_case(cid):
    met = pd.read_csv(DATA / "metrics" / f"{cid}_metrics.csv")
    sp  = pd.read_csv(DATA / "solver_pressures" / f"{cid}_solver_pressure_mmHg.csv")
    return met, sp


# ─── loops_plv + loops_regional ────────────────────────────────────────────
for proxy_mode, fname in [("PLV","loops_plv"), ("Trans","loops_regional")]:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for cid in CASES:
        color = cmap(norm(RV_ESP[cid]))
        met, sp = load_case(cid)
        P_LV = np.interp(met["time_s"], sp["time_s"], sp["P_LV_mmHg"]) * MMHG_TO_KPA
        P_RV = np.interp(met["time_s"], sp["time_s"], sp["P_RV_mmHg"]) * MMHG_TO_KPA
        for col, reg in enumerate(REGIONS):
            E_ff = met[f"mean_E_ff_{reg}"].values
            E_ll = met[f"mean_E_ll_{reg}"].values
            S_ff = met[f"mean_S_ff_Pa_{reg}"].values * PA_TO_KPA
            # ED-reference: subtract the value at t=0 of the (last) beat
            E_ff = E_ff - E_ff[0]
            E_ll = E_ll - E_ll[0]
            axes[0, col].plot(E_ff, S_ff, color=color, lw=1.3, alpha=0.85)
            if reg == "LV":
                P = P_LV
            elif reg == "RV":
                P = P_RV if proxy_mode == "Trans" else P_LV
            else:
                P = (P_LV - P_RV) if proxy_mode == "Trans" else P_LV
            axes[1, col].plot(E_ll, P, color=color, lw=1.3, alpha=0.85)
    for col, reg in enumerate(REGIONS):
        axes[0, col].set_title(reg, fontsize=13, fontweight="bold", pad=8)
        axes[0, col].set_xlabel(r"$E_{ff} - E_{ff}^{ED}$", fontsize=10)
        axes[0, col].set_ylabel(r"$S_{ff}$ (kPa)", fontsize=10)
        axes[0, col].grid(alpha=0.15, linewidth=0.6); clean_spines(axes[0, col])
        axes[1, col].set_xlabel(r"$\varepsilon_{ll} - \varepsilon_{ll}^{ED}$", fontsize=10)
        axes[1, col].grid(alpha=0.15, linewidth=0.6); clean_spines(axes[1, col])
    if proxy_mode == "PLV":
        for col in range(3):
            axes[1, col].set_ylabel("$P_{LV}$ (kPa)", fontsize=10)
    else:
        axes[1, 0].set_ylabel("$P_{LV}$ (kPa)", fontsize=10)
        axes[1, 1].set_ylabel("$P_{RV}$ (kPa)", fontsize=10)
        axes[1, 2].set_ylabel("$P_{LV}-P_{RV}$ (kPa)", fontsize=10)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.02, aspect=30)
    cbar.set_label("RV$_{ESP}$ (mmHg)", fontsize=10); cbar.outline.set_linewidth(0.5)
    fig.suptitle(f"top: stress–strain   ·   bottom: pressure × longitudinal strain "
                 f"(ED-referenced)   —   v12 {VARIANT.upper()}",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / f"{fname}.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig); print(f"[{VARIANT}] Saved {fname}")

# ─── loop_comparison (3x3) ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
column_titles = ["$S_{ff}$ vs $E_{ff}$  (truth)",
                 "uniform $P_{LV}$ × $\\varepsilon_{ll}$",
                 "regional  $P$ × $\\varepsilon_{ll}$"]
for cid in CASES:
    color = cmap(norm(RV_ESP[cid]))
    met, sp = load_case(cid)
    P_LV = np.interp(met["time_s"], sp["time_s"], sp["P_LV_mmHg"]) * MMHG_TO_KPA
    P_RV = np.interp(met["time_s"], sp["time_s"], sp["P_RV_mmHg"]) * MMHG_TO_KPA
    for row, reg in enumerate(REGIONS):
        E_ff = met[f"mean_E_ff_{reg}"].values
        E_ll = met[f"mean_E_ll_{reg}"].values
        S_ff = met[f"mean_S_ff_Pa_{reg}"].values * PA_TO_KPA
        E_ff = E_ff - E_ff[0]; E_ll = E_ll - E_ll[0]
        axes[row, 0].plot(E_ff, S_ff, color=color, lw=1.3, alpha=0.85)
        axes[row, 1].plot(E_ll, P_LV, color=color, lw=1.3, alpha=0.85)
        if reg == "LV":    P_reg = P_LV
        elif reg == "RV":  P_reg = P_RV
        else:              P_reg = P_LV - P_RV
        axes[row, 2].plot(E_ll, P_reg, color=color, lw=1.3, alpha=0.85)
for col in range(3):
    axes[0, col].set_title(column_titles[col], fontsize=12, fontweight="bold", pad=8)
for row, reg in enumerate(REGIONS):
    axes[row, 0].set_ylabel(f"{reg}\n$S_{{ff}}$ (kPa)", fontsize=10)
    axes[row, 1].set_ylabel("$P_{LV}$ (kPa)", fontsize=10)
    reg_p_label = {"LV":"$P_{LV}$","RV":"$P_{RV}$","Septum":"$P_{LV}-P_{RV}$"}[reg]
    axes[row, 2].set_ylabel(f"{reg_p_label} (kPa)", fontsize=10)
for row in range(3):
    for col in range(3):
        axes[row, col].grid(alpha=0.15, linewidth=0.6); clean_spines(axes[row, col])
        if row == 2:
            if col == 0:
                axes[row, col].set_xlabel(r"$E_{ff} - E_{ff}^{ED}$", fontsize=10)
            else:
                axes[row, col].set_xlabel(r"$\varepsilon_{ll} - \varepsilon_{ll}^{ED}$", fontsize=10)
sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02, aspect=30)
cbar.set_label("RV$_{ESP}$ (mmHg)", fontsize=10); cbar.outline.set_linewidth(0.5)
fig.suptitle(f"Stress–strain truth vs pressure–strain proxies — v12 {VARIANT.upper()}",
             fontsize=13, fontweight="bold")
fig.savefig(OUT / "loop_comparison.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "loop_comparison.pdf", bbox_inches="tight")
plt.close(fig); print(f"[{VARIANT}] Saved loop_comparison")

# ─── pv_loops_0d + pv_loops_solver ─────────────────────────────────────────
for src, fname, title in [
    ("circulation_timeseries", "pv_loops_0d",    "PV loops — 0D Windkessel pressure"),
    ("solver_pressures",       "pv_loops_solver","PV loops — FEM cavity pressure"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for cid in CASES:
        color = cmap(norm(RV_ESP[cid]))
        if src == "circulation_timeseries":
            circ = pd.read_csv(DATA / src / f"{cid}_circulation.csv")
            V_LV, P_LV = circ["V_LV"], circ["p_LV"]
            V_RV, P_RV = circ["V_RV"], circ["p_RV"]
        else:
            circ = pd.read_csv(DATA / "circulation_timeseries" / f"{cid}_circulation.csv")
            sp   = pd.read_csv(DATA / "solver_pressures" / f"{cid}_solver_pressure_mmHg.csv")
            V_LV, V_RV = circ["V_LV"], circ["V_RV"]
            P_LV = np.interp(circ["time_s"], sp["time_s"], sp["P_LV_mmHg"])
            P_RV = np.interp(circ["time_s"], sp["time_s"], sp["P_RV_mmHg"])
        axes[0].plot(V_LV, P_LV, color=color, lw=1.2, alpha=0.85)
        axes[1].plot(V_RV, P_RV, color=color, lw=1.2, alpha=0.85)
    for ax, name in zip(axes, ["LV","RV"]):
        ax.set_xlabel("Volume (mL)", fontsize=10)
        ax.set_ylabel(f"$P_{{{name}}}$ (mmHg)", fontsize=10)
        ax.set_title(f"{name} PV loop", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.15, linewidth=0.6); clean_spines(ax)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("RV$_{ESP}$ (mmHg)", fontsize=10); cbar.outline.set_linewidth(0.5)
    fig.suptitle(f"{title} — v12 {VARIANT.upper()}", fontsize=12, fontweight="bold")
    fig.savefig(OUT / f"{fname}.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig); print(f"[{VARIANT}] Saved {fname}")


# ─── scatter_spectrum (3x3, Pearson r / r^2) ───────────────────────────────
MASK_FNS = [
    lambda pc: pc["region_tags"] == 1,                 # LV
    lambda pc: pc["region_tags"] == 2,                 # RV
    lambda pc: pc["is_geometric_septum"].astype(bool), # Septum
]

def load_densities(cid, mask_fn):
    pc = np.load(DATA / "per_cell_data" / f"{cid}_per_cell_data.npz", allow_pickle=True)
    cv = pc["cell_volumes"]; mask = mask_fn(pc); V = cv[mask].sum()
    return {
        "W":     -pc["w_total"][mask].sum() / V * KPA,
        "PLV":   -pc["proxy_PLV_ll"][mask].sum() / V * KPA,
        "PRV":   -pc["proxy_PRV_ll"][mask].sum() / V * KPA,
        "Trans": -(pc["proxy_PLV_ll"][mask].sum() - pc["proxy_PRV_ll"][mask].sum()) / V * KPA,
    }

def ols(x, y):
    slope = np.sum((x - x.mean())*(y - y.mean())) / np.sum((x - x.mean())**2)
    intercept = y.mean() - slope * x.mean()
    return slope, intercept

fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
DIAGONAL = {(0,0),(1,1),(2,2)}
corr_cache = {}  # used by work_traces too
for row, (rname, mask_fn) in enumerate(zip(REGIONS, MASK_FNS)):
    rows = [load_densities(cid, mask_fn) for cid in CASES]
    for col, (pk, color, label, marker) in enumerate(PROXIES):
        ax = axes[row, col]
        xs = np.array([r[pk] for r in rows])
        ys = np.array([r["W"] for r in rows])
        ax.scatter(xs, ys, c=color, marker=marker, s=70,
                   edgecolors="white", linewidth=0.7, zorder=3)
        for cid, xv, yv in zip(CASES, xs, ys):
            ax.annotate(cid, (xv, yv), fontsize=7, ha="left",
                        xytext=(5, 3), textcoords="offset points", color="#888888")
        r_val = pearsonr(xs, ys)[0] if len(xs) >= 3 else float("nan")
        r2 = r_val ** 2 if not np.isnan(r_val) else float("nan")
        slope, intercept = ols(xs, ys)
        pad = 0.1 * (xs.max() - xs.min()) if xs.max() != xs.min() else 1.0
        xr = np.linspace(xs.min() - pad, xs.max() + pad, 20)
        ax.plot(xr, slope * xr + intercept,
                color="#888888", lw=1.2, alpha=0.9, zorder=2)
        on_diag = (row, col) in DIAGONAL
        ax.set_title(f"r = {r_val:+.2f}    R² = {r2:.2f}",
                     fontsize=13 if on_diag else 10,
                     fontweight="bold" if on_diag else "normal",
                     color=color if on_diag else "#444444", pad=6)
        ax.grid(alpha=0.15, linewidth=0.6); clean_spines(ax)
        if col == 0:
            ax.set_ylabel(f"{rname}\n$\\int S:dE$ density (kPa)", fontsize=10)
        if row == 2:
            ax.set_xlabel(f"{label} density (kPa)", fontsize=10)
        if row == 0:
            ax.text(0.5, 1.18, label, transform=ax.transAxes, fontsize=13,
                    fontweight="bold", ha="center", color=color)
        corr_cache[(rname, pk)] = (r_val, r2, xs, ys)

fig.suptitle(
    f"Pressure-strain proxy density vs $\\int S:dE$ (kPa) — v12 {VARIANT.upper()}",
    fontsize=14, fontweight="bold", y=1.02,
)
fig.savefig(OUT / "scatter_spectrum.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "scatter_spectrum.pdf", bbox_inches="tight")
plt.close(fig); print(f"[{VARIANT}] Saved scatter_spectrum")


# ─── work_traces (2x2 with correlation table) ──────────────────────────────
fig = plt.figure(figsize=(16, 11))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

# Top-left: LV panel. Top-right: RV panel. Bottom-left: Septum.
# Bottom-right: correlation table.
BEST = {"LV":"PLV", "RV":"PRV", "Septum":"Trans"}

x_idx = np.arange(8)
x_labels = [f"{cid}\n(sPAP={RV_ESP[cid]:.0f})" for cid in CASES]

def plot_trace(ax, region, best_only=True):
    ys_w = np.array([corr_cache[(region, "PLV")][3][i] for i in range(8)])
    ax.plot(x_idx, ys_w, "o-", color="#1a1a1a", lw=1.8, ms=6, label=r"$\int S:dE$")
    ax.set_ylabel(r"$\int S:dE$ density (kPa)", fontsize=10)
    ax.tick_params(axis='y', labelcolor="#1a1a1a")
    ax2 = ax.twinx()
    for pk, color, label, _ in PROXIES:
        if best_only and pk != BEST[region]:
            continue
        xs_p = np.array([corr_cache[(region, pk)][2][i] for i in range(8)])
        alpha = 0.25 + 0.75 * abs(corr_cache[(region, pk)][0])
        ax2.plot(x_idx, xs_p, "s--", color=color, lw=1.5, ms=5,
                 alpha=alpha, label=label)
    ax2.set_ylabel(f"{ {'PLV':'$P_{LV}$','PRV':'$P_{RV}$','Trans':'$P_{LV}-P_{RV}$'}[BEST[region]] } "
                   f"proxy density (kPa)", fontsize=10)
    ax2.tick_params(axis='y', labelcolor=dict(PLV="#1f77b4",PRV="#d62728",Trans="#2ca02c")[BEST[region]])
    ax.set_xticks(x_idx); ax.set_xticklabels(x_labels, fontsize=8, rotation=30, ha="right")
    ax.grid(alpha=0.15, linewidth=0.6); clean_spines(ax)
    clean_spines(ax2, right=True)
    ax.set_title(f"{region}  ·  best proxy: {BEST[region]}", fontsize=12, fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")


def plot_septum(ax):
    ys_w = np.array([corr_cache[("Septum", "PLV")][3][i] for i in range(8)])
    ax.plot(x_idx, ys_w, "o-", color="#1a1a1a", lw=1.8, ms=6, label=r"$\int S:dE$")
    ax.set_ylabel(r"$\int S:dE$ density (kPa)", fontsize=10)
    ax2 = ax.twinx()
    for pk, color, label, _ in PROXIES:
        xs_p = np.array([corr_cache[("Septum", pk)][2][i] for i in range(8)])
        alpha = 0.25 + 0.75 * abs(corr_cache[("Septum", pk)][0])
        ax2.plot(x_idx, xs_p, "s--", color=color, lw=1.4, ms=4,
                 alpha=alpha, label=label)
    ax2.set_ylabel("proxy density (kPa)", fontsize=10)
    ax.set_xticks(x_idx); ax.set_xticklabels(x_labels, fontsize=8, rotation=30, ha="right")
    ax.grid(alpha=0.15, linewidth=0.6); clean_spines(ax); clean_spines(ax2, right=True)
    ax.set_title("Septum  ·  all three proxies overlaid (opacity ∝ |r|)",
                 fontsize=12, fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")

ax_lv = fig.add_subplot(gs[0, 0]); plot_trace(ax_lv, "LV")
ax_rv = fig.add_subplot(gs[0, 1]); plot_trace(ax_rv, "RV")
ax_sp = fig.add_subplot(gs[1, 0]); plot_septum(ax_sp)
ax_tb = fig.add_subplot(gs[1, 1]); ax_tb.axis("off")

# Correlation table
table_rows = []
for region in REGIONS:
    for pk, _, label, _ in PROXIES:
        r_val, r2, _, _ = corr_cache[(region, pk)]
        table_rows.append([region, label, f"{r_val:+.3f}", f"{r2:.3f}"])

tbl = ax_tb.table(cellText=table_rows,
                  colLabels=["region","proxy","r","r²"],
                  loc="center", cellLoc="center", colWidths=[0.2,0.3,0.2,0.2])
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.4)

# Emphasise best-per-region rows
for i, (region, label, rv, r2v) in enumerate(table_rows):
    if label == {"LV":"$P_{LV}$","RV":"$P_{RV}$","Septum":"$P_{LV}-P_{RV}$"}[region]:
        for col in range(4):
            tbl[(i+1, col)].set_facecolor("#fff3b0")
ax_tb.set_title("Correlation summary (Pearson r and r²)",
                fontsize=12, fontweight="bold", pad=8)

fig.suptitle(f"Spectrum work traces and correlations — v12 {VARIANT.upper()}",
             fontsize=14, fontweight="bold")
fig.savefig(OUT / "work_traces.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "work_traces.pdf", bbox_inches="tight")
plt.close(fig); print(f"[{VARIANT}] Saved work_traces")

# ─── Print tight summary ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"v12 {VARIANT.upper()} — correlation summary (Pearson r, r² = r²)")
print("=" * 70)
for region in REGIONS:
    print(f"  {region}:")
    for pk, _, label, _ in PROXIES:
        r_val, r2, _, _ = corr_cache[(region, pk)]
        print(f"    {label:<18} r = {r_val:+.3f}   r² = {r2:.3f}")
print(f"\nFigures in {OUT}/")
