#!/usr/bin/env python3
"""
analyze_thickness_efficiency.py — Reframing the thickness sims as a
proxy-efficiency / ratio-stability test instead of a Pearson-r test.

Why:
    Pearson r only measures rank agreement, which is trivially high on
    a monotone geometric sweep at fixed circulation — it cannot tell us
    whether the proxy captures a stable fraction of the true work, and
    it deliberately throws away the scale information that we need.

What this script asks:
    1. η = W_true / W_proxy — is the efficiency ratio stable under
       thickness variation? A flat line means the proxy captures a
       fixed fraction of true work regardless of wall geometry, i.e.
       a single calibration could turn the proxy into a real estimator.
       A drifting line means the proxy is confounded by geometry and
       cannot be corrected out.

    2. Pooled proxy-vs-truth scatter across spectrum + thickness.
       Does a single through-origin line fit all 22 points? If yes,
       the proxy is simultaneously robust to disease and to geometry.
       If the sweeps form separate clouds, the proxy has sweep-
       dependent bias.

    3. η drift magnitude: spectrum vs thickness. Which sweep causes a
       bigger fractional change in efficiency? If thickness drift is
       smaller than spectrum drift, then geometry is a smaller confound
       than the physiological axis we actually care about.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path("results/analysis/thickness_efficiency")
OUT.mkdir(parents=True, exist_ok=True)

# ── Runs ────────────────────────────────────────────────────────────────────
SPECTRUM_RUNS = [
    ("healthy",         "Borderline PH", "1020849"),
    ("mild",            "Mild",          "1020851"),
    ("moderate",        "Moderate",      "1020852"),
    ("moderate_severe", "Mod–severe",    "1020853"),
    ("severe",          "Severe",        "1020854"),
    ("very_severe",     "Very severe",   "1020855"),
    ("end_stage",       "End-stage",     "1020856"),
]
SPECTRUM_ROOT = Path("results/sims/2026-04-12")

# (run_id, sev, thickness_mm, family)
THICKNESS_RUNS = [
    # 2026-04-14 base set: rvfw {3, 6, 9, 12} × {healthy, severe}
    ("1029754", "healthy",  3.0, "rvfw", "2026-04-14"),
    ("1029755", "severe",   3.0, "rvfw", "2026-04-14"),
    ("1029756", "healthy",  6.0, "rvfw", "2026-04-14"),
    ("1029757", "severe",   6.0, "rvfw", "2026-04-14"),
    ("1029758", "healthy",  9.0, "rvfw", "2026-04-14"),
    ("1029759", "severe",   9.0, "rvfw", "2026-04-14"),
    ("1029760", "healthy", 12.0, "rvfw", "2026-04-14"),
    ("1029761", "severe",  12.0, "rvfw", "2026-04-14"),
    # 2026-04-15 round1+2 densification (severe)
    ("1033857", "severe",   4.5, "rvfw", "2026-04-15"),
    ("1033858", "severe",   7.5, "rvfw", "2026-04-15"),
    ("1033859", "severe",  10.5, "rvfw", "2026-04-15"),
    # 2026-04-15 LV control (severe)
    ("1033860", "severe",   3.0, "lvfw", "2026-04-15"),
    ("1033861", "severe",   6.0, "lvfw", "2026-04-15"),
    ("1033862", "severe",   9.0, "lvfw", "2026-04-15"),
    ("1033863", "severe",  12.0, "lvfw", "2026-04-15"),
]

# ── Helpers ─────────────────────────────────────────────────────────────────
def density(pc, mask):
    V = float(pc["cell_volumes"][mask].sum())
    if V <= 0 or not mask.any():
        return None
    KPA = 1e-3  # J/m³ → kPa = mJ/mL
    return {
        "V_m3":    V,
        "W_true":  float(pc["w_total"][mask].sum())        / V * KPA,
        "PLV":     float(pc["proxy_PLV_ll"][mask].sum())   / V * KPA,
        "PRV":     float(pc["proxy_PRV_ll"][mask].sum())   / V * KPA,
        "Trans":   float(pc["proxy_Trans_ll"][mask].sum()) / V * KPA,
    }


def region_data(pc):
    return {
        "LV":     density(pc, pc["region_tags"] == 1),
        "RV":     density(pc, pc["region_tags"] == 2),
        "Septum": density(pc, pc["is_ldrb_septum"].astype(bool)),
    }


def load_pc(path):
    if not path.exists():
        print(f"  MISSING {path}")
        return None
    return np.load(path, allow_pickle=True)


# ── Load data ───────────────────────────────────────────────────────────────
spectrum = []
for sev_key, label, run_id in SPECTRUM_RUNS:
    pc = load_pc(SPECTRUM_ROOT / f"UKB_6beats_run_{run_id}" / "per_cell_data.npz")
    if pc is None:
        continue
    spectrum.append({
        "sweep": "spectrum",
        "label": label,
        "regions": region_data(pc),
    })
print(f"Loaded {len(spectrum)} spectrum cases")

thickness = []
for run_id, sev, mm, family, date in THICKNESS_RUNS:
    root = Path(f"results/sims/{date}")
    pc = load_pc(root / f"UKB_1beats_run_{run_id}" / "per_cell_data.npz")
    if pc is None:
        continue
    thickness.append({
        "sweep":  f"{family}_{sev}",
        "family": family,
        "sev":    sev,
        "mm":     mm,
        "label":  f"+{mm:g}",
        "regions": region_data(pc),
    })
print(f"Loaded {len(thickness)} thickness cases")

# Group thickness by sweep
sweeps = {}
for c in thickness:
    sweeps.setdefault(c["sweep"], []).append(c)
for k in sweeps:
    sweeps[k].sort(key=lambda c: c["mm"])
print("Thickness sweeps:", {k: len(v) for k, v in sweeps.items()})

# ── Plot 1: η across each thickness sweep ───────────────────────────────────
REGIONS = ("LV", "RV", "Septum")
PROXIES = [("PLV",   "#1f77b4", r"$P_{LV}$"),
           ("PRV",   "#d62728", r"$P_{RV}$"),
           ("Trans", "#2ca02c", r"$P_{LV}-P_{RV}$")]


def efficiency(case, region, proxy_key):
    d = case["regions"][region]
    wp = d[proxy_key]
    return d["W_true"] / wp if wp != 0 else np.nan


def plot_eta_sweep(cases, xs, xlabel, title, out_name):
    """Plot η vs sweep axis for each region × proxy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for i, region in enumerate(REGIONS):
        ax = axes[i]
        for proxy_key, color, label in PROXIES:
            eta = [efficiency(c, region, proxy_key) for c in cases]
            cv = np.std(eta) / abs(np.mean(eta)) * 100.0
            ax.plot(xs, eta, "-o", color=color, lw=1.8, ms=7,
                    label=f"{label}  (cv={cv:.1f}%)")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(region, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel(r"$\eta = W_{true}/W_{proxy}$", fontsize=11)
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(OUT / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_name}")


# η across each thickness sweep
for sweep_key, cases in sweeps.items():
    xs = [c["mm"] for c in cases]
    fam, sev = sweep_key.split("_")
    title = (f"Proxy efficiency across {fam.upper()} thickness sweep "
             f"({sev} circulation, n={len(cases)})")
    plot_eta_sweep(cases, xs, f"{fam.upper()} added thickness (mm)",
                   title, f"fig_efficiency_{sweep_key}.png")

# η across spectrum
xs_spec = list(range(len(spectrum)))
plot_eta_sweep(spectrum, xs_spec,
               "PAH severity (increasing RV afterload)",
               f"Proxy efficiency across PAH spectrum (n={len(spectrum)})",
               "fig_efficiency_spectrum.png")

# Custom x tick labels for spectrum
# (quick patch: replot with labels)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
for i, region in enumerate(REGIONS):
    ax = axes[i]
    for proxy_key, color, label in PROXIES:
        eta = [efficiency(c, region, proxy_key) for c in spectrum]
        cv = np.std(eta) / abs(np.mean(eta)) * 100.0
        ax.plot(xs_spec, eta, "-o", color=color, lw=1.8, ms=7,
                label=f"{label}  (cv={cv:.1f}%)")
    ax.set_xticks(xs_spec)
    ax.set_xticklabels([c["label"] for c in spectrum],
                       rotation=35, ha="right", fontsize=8)
    ax.set_title(region, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    if i == 0:
        ax.set_ylabel(r"$\eta = W_{true}/W_{proxy}$", fontsize=11)
    ax.legend(fontsize=8, loc="best")
fig.suptitle(f"Proxy efficiency across PAH spectrum (n={len(spectrum)})",
             fontsize=12, fontweight="bold")
fig.savefig(OUT / "fig_efficiency_spectrum.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Plot 2: pooled proxy-vs-truth scatter, one panel per (region, proxy) ────
all_cases = spectrum + thickness

SWEEP_STYLE = {
    "spectrum":     ("#000000", "D", "spectrum (7 PAH severities)"),
    "rvfw_severe":  ("#d62728", "s", "RVFW thickness, severe (n=7)"),
    "rvfw_healthy": ("#ff7f0e", "^", "RVFW thickness, healthy (n=4)"),
    "lvfw_severe":  ("#1f77b4", "o", "LVFW thickness, severe (n=4)"),
}


def through_origin_fit(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 2 or np.sum(x**2) == 0:
        return np.nan, np.nan
    a = np.sum(x * y) / np.sum(x**2)
    y_pred = a * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(a), float(r2)


fig, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
for row, region in enumerate(REGIONS):
    for col, (proxy_key, color, proxy_label) in enumerate(PROXIES):
        ax = axes[row, col]
        xs_all, ys_all = [], []
        for c in all_cases:
            d = c["regions"][region]
            x_val = d[proxy_key]
            y_val = d["W_true"]
            sc, mk, _ = SWEEP_STYLE[c["sweep"]]
            ax.scatter(x_val, y_val, c=sc, marker=mk, s=55,
                       edgecolors="white", linewidth=0.6, zorder=3)
            xs_all.append(x_val)
            ys_all.append(y_val)
        a, r2 = through_origin_fit(xs_all, ys_all)
        xs_arr = np.array(xs_all)
        xr = np.linspace(xs_arr.min(), xs_arr.max(), 20)
        ax.plot(xr, a * xr, "k--", lw=1.5, alpha=0.7, zorder=2)
        # Also fit separately per sweep to show dispersion
        for sweep_key, (sc, mk, _) in SWEEP_STYLE.items():
            sub = [c for c in all_cases if c["sweep"] == sweep_key]
            if len(sub) < 2:
                continue
            xs_s = np.array([c["regions"][region][proxy_key] for c in sub])
            ys_s = np.array([c["regions"][region]["W_true"] for c in sub])
            a_s, _ = through_origin_fit(xs_s, ys_s)
            xr_s = np.linspace(xs_s.min(), xs_s.max(), 10)
            ax.plot(xr_s, a_s * xr_s, color=sc, lw=1.0, alpha=0.55, zorder=1)
        ax.set_xlabel(f"{proxy_label} density (kPa)", fontsize=9)
        if col == 0:
            ax.set_ylabel(r"$W_{true}$ density (kPa)", fontsize=10)
        ax.set_title(f"{region}    slope={a:+.2f}   $R^2$={r2:.3f}",
                     fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)

handles = [Line2D([0], [0], marker=mk, color="white",
                  markerfacecolor=sc, markeredgecolor="white",
                  markersize=9, label=lbl)
           for sc, mk, lbl in SWEEP_STYLE.values()]
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, 1.01), frameon=False)
fig.suptitle(
    "Pooled proxy-vs-truth: does a single line fit both disease and geometry?\n"
    "dashed black = fit across all 22 points   coloured = per-sweep fit",
    fontsize=12, fontweight="bold", y=1.05)
fig.savefig(OUT / "fig_pooled_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig_pooled_scatter.png")

# ── Print η stability summary ───────────────────────────────────────────────
print("\n" + "=" * 78)
print("η = W_true/W_proxy STABILITY  (coefficient of variation = std/|mean|)")
print("=" * 78)
sweeps_all = {"spectrum": spectrum, **sweeps}
for region in REGIONS:
    print(f"\n--- {region} ---")
    header = f"{'sweep':<16}"
    for _, _, lbl in PROXIES:
        header += f"  {lbl:>22}"
    print(header)
    print(f"{'':<16}"
          + "  " + "  ".join(["  mean    cv       "] * len(PROXIES)))
    for sweep_name, cases in sweeps_all.items():
        row = f"{sweep_name:<16}"
        for proxy_key, _, _ in PROXIES:
            etas = np.array([efficiency(c, region, proxy_key) for c in cases])
            mean = float(np.mean(etas))
            cv = float(np.std(etas) / abs(mean) * 100.0) if mean != 0 else float("nan")
            row += f"  {mean:>+7.2f}  {cv:>6.1f}%  "
        print(row)

print("\n" + "=" * 78)
print("POOLED THROUGH-ORIGIN FITS   (spectrum + all 15 thickness = 22 points)")
print("=" * 78)
print(f"{'region':<10} {'proxy':<10} {'slope':>9} {'R^2':>10}")
for region in REGIONS:
    for proxy_key, _, _ in PROXIES:
        xs = [c["regions"][region][proxy_key] for c in all_cases]
        ys = [c["regions"][region]["W_true"]  for c in all_cases]
        a, r2 = through_origin_fit(xs, ys)
        print(f"{region:<10} {proxy_key:<10} {a:>+9.3f} {r2:>10.4f}")

print("\nDone.")
