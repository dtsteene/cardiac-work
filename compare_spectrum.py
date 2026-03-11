#!/usr/bin/env python3
"""
compare_spectrum.py

Compare N cardiac simulation cases across a disease severity spectrum.
Designed for the PH severity gradient (Healthy → Mild → Moderate → Moderate-Severe → Severe)
but works for any ordered set of cases.

Generates publication-quality figures showing how cardiac work, stress-strain loops,
and proxy accuracy evolve across the severity spectrum.

Usage:
  python3 compare_spectrum.py dir1/ dir2/ dir3/ dir4/ dir5/
  python3 compare_spectrum.py dir1/ dir2/ ... --labels "Healthy" "Mild" "Moderate" "Mod-Severe" "Severe"
  python3 compare_spectrum.py dir1/ dir2/ ... --outdir figures/spectrum/
"""

import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path


# ─── Thesis Figure Style ─────────────────────────────────────────────────────

def setup_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
    })


# ─── Color Palette ───────────────────────────────────────────────────────────

REGION_COLORS = {"LV": "#4878A8", "RV": "#C25454", "Septum": "#5EA55E"}


def spectrum_colors(n):
    """Generate a sequential colormap from blue (healthy) to red (severe)."""
    cmap = mpl.colormaps.get_cmap("RdYlBu_r")
    # Sample from 0.15 to 0.85 to avoid washed-out extremes
    return [cmap(0.15 + 0.7 * i / max(n - 1, 1)) for i in range(n)]


def _save(fig, outdir, name):
    out = Path(outdir) / name
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_metrics(folder):
    path = Path(folder)
    search_dirs = [
        path / "analysis" / "last_beat",
        path / "analysis_last_beat",
        path / "metrics",
        path,
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        files = sorted(d.glob("metrics_downsample_*.npy"), key=lambda p: len(str(p)))
        if files:
            print(f"  Loading: {files[0]}")
            m = np.load(files[0], allow_pickle=True).item()
            return _normalize_keys(m)
    print(f"  No metrics found in {folder}")
    return None


def _normalize_keys(m):
    renames = {}
    regions = ["LV", "RV", "Septum", "Whole"]
    for reg in regions:
        renames[f"work_fiber_{reg}"]  = f"work_ff_{reg}"
        renames[f"work_sheet_{reg}"]  = f"work_ss_{reg}"
        renames[f"work_normal_{reg}"] = f"work_nn_{reg}"
        renames[f"work_shear_{reg}"]  = f"work_cross_{reg}"
    for old_k in list(m.keys()):
        if old_k.startswith("work_ps_index_"):
            new_k = old_k.replace("work_ps_index_", "work_ps_ff_")
            renames[old_k] = new_k
    for old, new in renames.items():
        if old in m and new not in m:
            m[new] = m[old]
    return m


def total_work(m, key):
    if key not in m:
        return 0.0
    return float(np.sum(m[key]))


def get_array(m, key):
    return np.array(m[key]) if key in m else np.array([])


def load_hemodynamics(folder):
    """Extract key hemodynamic parameters from circulation history or params."""
    path = Path(folder)

    hemo = {}

    # Try simulation_params.json for HR
    params_file = path / "simulation_params.json"
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
        if "heart_rate_bpm" in params:
            hemo["HR"] = params["heart_rate_bpm"]

    # Try circulation parameters
    circ_params = path / "circulation" / "parameters.json"
    if circ_params.exists():
        with open(circ_params) as f:
            cp = json.load(f)
        if "description" in cp:
            hemo["description"] = cp["description"]

    return hemo


# ─── Hemodynamic Summary Table ───────────────────────────────────────────────

def extract_hemodynamics(results, labels):
    """Extract key hemodynamic indices from the metrics arrays."""
    hemo_data = []
    for r, lbl in zip(results, labels):
        p_lv = get_array(r, "p_LV")
        p_rv = get_array(r, "p_RV")
        v_lv = get_array(r, "V_LV_FEM") if "V_LV_FEM" in r else get_array(r, "V_LV")
        v_rv = get_array(r, "V_RV_FEM") if "V_RV_FEM" in r else get_array(r, "V_RV")

        entry = {"label": lbl}
        if len(p_lv) > 0:
            entry["LV_ESP"] = float(np.max(p_lv))
            entry["LV_EDP"] = float(p_lv[0])
        if len(p_rv) > 0:
            entry["RV_ESP"] = float(np.max(p_rv))
            entry["RV_EDP"] = float(p_rv[0])
        if len(v_lv) > 1:
            entry["LV_EDV"] = float(np.max(v_lv))
            entry["LV_ESV"] = float(np.min(v_lv))
            entry["LV_SV"] = entry["LV_EDV"] - entry["LV_ESV"]
            entry["LV_EF"] = entry["LV_SV"] / entry["LV_EDV"] * 100
        if len(v_rv) > 1:
            entry["RV_EDV"] = float(np.max(v_rv))
            entry["RV_ESV"] = float(np.min(v_rv))
            entry["RV_SV"] = entry["RV_EDV"] - entry["RV_ESV"]
            entry["RV_EF"] = entry["RV_SV"] / entry["RV_EDV"] * 100

        hemo_data.append(entry)
    return hemo_data


def print_hemodynamic_table(hemo_data):
    W = 110
    print("\n" + "=" * W)
    print(f"{'HEMODYNAMIC SUMMARY ACROSS PH SEVERITY':^{W}}")
    print("=" * W)

    keys = [
        ("LV_ESP", "LV ESP", "mmHg", 1),
        ("RV_ESP", "RV ESP", "mmHg", 1),
        ("LV_EDP", "LV EDP", "mmHg", 1),
        ("RV_EDP", "RV EDP", "mmHg", 1),
        ("LV_EDV", "LV EDV", "mL", 1),
        ("RV_EDV", "RV EDV", "mL", 1),
        ("LV_SV",  "LV SV",  "mL", 1),
        ("RV_SV",  "RV SV",  "mL", 1),
        ("LV_EF",  "LV EF",  "%",  1),
        ("RV_EF",  "RV EF",  "%",  1),
    ]

    header = f"  {'Metric':<12} {'Unit':<6}"
    for h in hemo_data:
        header += f" {h['label']:>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for key, name, unit, _ in keys:
        row = f"  {name:<12} {unit:<6}"
        for h in hemo_data:
            val = h.get(key, float('nan'))
            row += f" {val:>12.1f}"
        print(row)

    print("=" * W)


# ─── Figure 1: Hemodynamic Trends ────────────────────────────────────────────

def plot_hemodynamic_trends(results, labels, colors, outdir):
    """Line plots showing how key hemodynamic indices change across severity."""
    hemo = extract_hemodynamics(results, labels)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.0))

    panels = [
        (axes[0, 0], [("RV_ESP", "RV ESP")], "Pressure (mmHg)", "RV Systolic Pressure"),
        (axes[0, 1], [("LV_ESP", "LV ESP")], "Pressure (mmHg)", "LV Systolic Pressure"),
        (axes[0, 2], [("RV_EDP", "RV EDP"), ("LV_EDP", "LV EDP")], "Pressure (mmHg)", "End-Diastolic Pressure"),
        (axes[1, 0], [("RV_EDV", "RV EDV"), ("LV_EDV", "LV EDV")], "Volume (mL)", "End-Diastolic Volume"),
        (axes[1, 1], [("RV_SV", "RV SV"), ("LV_SV", "LV SV")], "Volume (mL)", "Stroke Volume"),
        (axes[1, 2], [("RV_EF", "RV EF"), ("LV_EF", "LV EF")], "EF (%)", "Ejection Fraction"),
    ]

    chamber_colors = {"RV": REGION_COLORS["RV"], "LV": REGION_COLORS["LV"]}

    for ax, metrics, ylabel, title in panels:
        for key, mlbl in metrics:
            vals = [h.get(key, np.nan) for h in hemo]
            chamber = "RV" if "RV" in key else "LV"
            ax.plot(x, vals, 'o-', color=chamber_colors[chamber], lw=1.5, ms=6, label=mlbl)
            # Annotate values
            for xi, v in zip(x, vals):
                if not np.isnan(v):
                    ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                                xytext=(0, 7), fontsize=6, ha='center', color='0.4')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        if len(metrics) > 1:
            ax.legend(fontsize=7, loc='best')

    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    _save(fig, outdir, "hemodynamic_trends.png")


# ─── Figure 2: PV Loops Overlay ──────────────────────────────────────────────

def plot_pv_loops(results, labels, colors, outdir):
    """Overlay PV loops for all cases, one panel per chamber."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    for ax, chamber, p_key in zip(axes, ["LV", "RV"], ["p_LV", "p_RV"]):
        for i, (r, lbl, c) in enumerate(zip(results, labels, colors)):
            v_key = f"V_{chamber}_FEM" if f"V_{chamber}_FEM" in r else f"V_{chamber}"
            v = get_array(r, v_key)
            p = get_array(r, p_key)
            if len(v) == 0 or len(p) == 0:
                continue
            ax.plot(v, p, color=c, lw=1.5, label=lbl)
            ax.plot(v[0], p[0], 'o', color=c, ms=4)

        ax.set_xlabel("Volume (mL)")
        ax.set_ylabel("Pressure (mmHg)")
        ax.set_title(f"{chamber} PV Loop", fontweight='bold')
        if chamber == "LV":
            ax.legend(fontsize=6, loc='upper right')

    fig.tight_layout(w_pad=1.5)
    _save(fig, outdir, "pv_loops_spectrum.png")


# ─── Figure 3: Stress-Strain Loops ───────────────────────────────────────────

def plot_stress_strain_spectrum(results, labels, colors, outdir):
    """Overlay fiber stress-strain loops, colored by severity."""
    regions = ["LV", "RV", "Septum"]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.0))

    for ax, region in zip(axes, regions):
        for r, lbl, c in zip(results, labels, colors):
            E = get_array(r, f"mean_E_ff_{region}") * 100
            S = get_array(r, f"mean_S_ff_{region}") / 1e3
            if len(E) == 0:
                continue
            ax.plot(E, S, color=c, lw=1.5, label=lbl)
            ax.plot(E[0], S[0], 'o', color=c, ms=3)

        ax.axhline(0, color='0.5', lw=0.4)
        ax.axvline(0, color='0.5', lw=0.4)
        ax.set_title(region, fontweight='bold')
        ax.set_xlabel(r"$E_{ff}$ (%)")
        if region == "LV":
            ax.set_ylabel(r"$S_{ff}$ (kPa)")
            ax.legend(fontsize=6, loc='best')

    fig.tight_layout(w_pad=1.0)
    _save(fig, outdir, "stress_strain_spectrum.png")


# ─── Figure 4: Pressure-Strain Loops ─────────────────────────────────────────

def plot_pressure_strain_spectrum(results, labels, colors, outdir):
    """Overlay pressure-strain loops (clinical proxy) across severity."""
    regions = [
        ("LV",     "mean_E_ff_LV",     "mean_E_ll_LV",     "p_LV"),
        ("RV",     "mean_E_ff_RV",     "mean_E_ll_RV",     "p_RV"),
        ("Septum", "mean_E_ff_Septum", "mean_E_ll_Septum", "p_LV"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.5))
    row_titles = [r"$P$ vs $E_{ff}$ (fiber)", r"$P$ vs $E_{ll}$ (longitudinal)"]

    for col, (region, eff_key, ell_key, p_key) in enumerate(regions):
        for row, (strain_key, row_title) in enumerate([(eff_key, row_titles[0]),
                                                        (ell_key, row_titles[1])]):
            ax = axes[row, col]
            for r, lbl, c in zip(results, labels, colors):
                strain = get_array(r, strain_key) * 100
                pressure = get_array(r, p_key)
                if len(strain) == 0 or len(pressure) == 0:
                    continue
                ax.plot(strain, pressure, color=c, lw=1.5, label=lbl)
                ax.plot(strain[0], pressure[0], 'o', color=c, ms=3)

            ax.axhline(0, color='0.5', lw=0.4)
            ax.axvline(0, color='0.5', lw=0.4)
            if row == 0:
                ax.set_title(region, fontweight='bold')
            strain_lbl = r"$E_{ff}$ (%)" if row == 0 else r"$E_{ll}$ (%)"
            ax.set_xlabel(strain_lbl)
            if col == 0:
                ax.set_ylabel("$P$ (mmHg)")
                if row == 0:
                    ax.legend(fontsize=5.5, loc='best')

    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    _save(fig, outdir, "pressure_strain_spectrum.png")


# ─── Figure 5: Work Trends ──────────────────────────────────────────────────

def plot_work_trends(results, labels, colors, outdir):
    """
    Line plots: how total and fiber work evolve across severity, per region.
    Also shows proxy work to assess tracking across the spectrum.
    """
    regions = ["LV", "RV", "Septum"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.5))

    for col, region in enumerate(regions):
        # Row 0: Absolute work values
        ax = axes[0, col]
        for wkey, wlbl, ls in [
            (f"work_true_{region}", "Total", "-"),
            (f"work_ff_{region}", "Fiber", "--"),
        ]:
            vals = [total_work(r, wkey) * 1e3 for r in results]
            ax.plot(x, vals, f'o{ls}', color=REGION_COLORS[region], lw=1.5, ms=6, label=wlbl)
            if ls == "--":
                ax.plot(x, vals, f'o{ls}', color=REGION_COLORS[region], lw=1.5, ms=6, alpha=0.6)

        # Add proxy line
        if region in ("LV", "RV"):
            proxy_key = f"work_ps_ll_{region}"
            proxy_vals = [total_work(r, proxy_key) * 1e3 for r in results]
            if any(abs(v) > 1e-10 for v in proxy_vals):
                ax.plot(x, proxy_vals, 's:', color=REGION_COLORS[region],
                        lw=1.0, ms=5, alpha=0.5, label="PS proxy")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
        ax.set_title(region, fontweight='bold')
        ax.axhline(0, color='k', lw=0.5)
        if col == 0:
            ax.set_ylabel("Work (mJ)")
            ax.legend(fontsize=6.5, loc='best')

        # Row 1: Normalized to baseline
        ax = axes[1, col]
        baseline_total = total_work(results[0], f"work_true_{region}")
        baseline_fiber = total_work(results[0], f"work_ff_{region}")

        if abs(baseline_total) > 1e-12:
            vals_norm = [total_work(r, f"work_true_{region}") / baseline_total for r in results]
            ax.plot(x, vals_norm, 'o-', color=REGION_COLORS[region], lw=1.5, ms=6, label="Total")
        if abs(baseline_fiber) > 1e-12:
            vals_norm = [total_work(r, f"work_ff_{region}") / baseline_fiber for r in results]
            ax.plot(x, vals_norm, 'o--', color=REGION_COLORS[region], lw=1.5, ms=6,
                    alpha=0.6, label="Fiber")

        ax.axhline(1.0, color='k', ls=':', lw=0.8, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
        if col == 0:
            ax.set_ylabel("Normalized to baseline")
            ax.legend(fontsize=6.5, loc='best')

    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    _save(fig, outdir, "work_trends.png")


# ─── Figure 6: Work Decomposition Spectrum ───────────────────────────────────

def plot_work_decomposition_spectrum(results, labels, colors, outdir):
    """Grouped bar chart: ff/ss/nn/cross work per region, all cases side by side."""
    components = [
        ("work_ff", "Fiber"),
        ("work_ss", "Sheet"),
        ("work_nn", "Normal"),
        ("work_cross", "Cross"),
    ]
    regions = ["LV", "RV", "Septum"]
    n = len(results)

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.5))

    for ax, region in zip(axes, regions):
        x = np.arange(len(components))
        width = 0.8 / n

        for j, (r, lbl, c) in enumerate(zip(results, labels, colors)):
            vals = [total_work(r, f"{ckey}_{region}") * 1e3 for ckey, _ in components]
            offset = (j - (n - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=lbl, color=c,
                   alpha=0.85, edgecolor='none')

        ax.set_xticks(x)
        ax.set_xticklabels([clbl for _, clbl in components], fontsize=8)
        ax.set_title(region, fontweight='bold')
        ax.axhline(0, color='k', lw=0.5)
        if region == "LV":
            ax.set_ylabel("Work (mJ)")
            ax.legend(fontsize=5.5, loc='best', ncol=1)

    fig.tight_layout(w_pad=0.8)
    _save(fig, outdir, "work_decomposition_spectrum.png")


# ─── Figure 7: Work Redistribution ──────────────────────────────────────────

def plot_work_redistribution_spectrum(results, labels, colors, outdir):
    """
    Stacked bar chart showing how work redistributes across regions.
    Also: work share trends as line plot.
    """
    regions = ["LV", "RV", "Septum"]
    n = len(labels)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.5))

    # Panel 1: Stacked bar — total work by region
    ax = axes[0]
    x = np.arange(n)
    bottoms = np.zeros(n)
    for region in regions:
        vals = [total_work(r, f"work_true_{region}") * 1e3 for r in results]
        ax.bar(x, vals, bottom=bottoms, color=REGION_COLORS[region],
               label=region, edgecolor='none', alpha=0.85)
        # Annotate
        for xi, v, b in zip(x, vals, bottoms):
            if abs(v) > 0.5:
                ax.text(xi, b + v / 2, f"{v:.1f}", ha='center', va='center',
                        fontsize=6, color='white', fontweight='bold')
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel("Total work (mJ)")
    ax.set_title("Regional total work", fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')

    # Panel 2: Work share (%) trends
    ax = axes[1]
    for region in regions:
        shares = []
        for r in results:
            w_reg = total_work(r, f"work_true_{region}")
            w_tot = sum(total_work(r, f"work_true_{rr}") for rr in regions)
            shares.append(w_reg / w_tot * 100 if abs(w_tot) > 1e-15 else 0)
        ax.plot(x, shares, 'o-', color=REGION_COLORS[region], lw=1.5, ms=6, label=region)
        for xi, s in zip(x, shares):
            ax.annotate(f"{s:.0f}%", (xi, s), textcoords="offset points",
                        xytext=(0, 7), fontsize=6, ha='center', color='0.4')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel("Work share (%)")
    ax.set_title("Work distribution", fontweight='bold')
    ax.legend(fontsize=7, loc='best')

    # Panel 3: Fiber work fraction trends
    ax = axes[2]
    for region in regions:
        effs = []
        for r in results:
            w_ff = total_work(r, f"work_ff_{region}")
            w_tot = total_work(r, f"work_true_{region}")
            effs.append(w_ff / w_tot * 100 if abs(w_tot) > 1e-15 else 0)
        ax.plot(x, effs, 'o-', color=REGION_COLORS[region], lw=1.5, ms=6, label=region)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel(r"$W_{ff}/W_{total}$ (%)")
    ax.set_title("Fiber work fraction", fontweight='bold')
    ax.legend(fontsize=7, loc='best')

    fig.tight_layout(w_pad=1.0)
    _save(fig, outdir, "work_redistribution_spectrum.png")


# ─── Figure 8: Proxy Accuracy Across Spectrum ───────────────────────────────

def plot_proxy_accuracy_spectrum(results, labels, colors, outdir):
    """
    How well the PS proxy tracks FEM work across the severity spectrum.
    Normalized to baseline: ideal tracking means points on y=x diagonal.
    """
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.5))

    strain_comp = "ll"
    truth_configs = [
        (0, "Total work", "work_true"),
        (1, "Fiber work", "work_ff"),
    ]

    # LV and RV freewall
    for row, row_title, truth_prefix in truth_configs:
        for col, region in enumerate(["LV", "RV"]):
            ax = axes[row, col]
            truth_key = f"{truth_prefix}_{region}"
            if region == "Septum":
                proxy_key = f"work_ps_{strain_comp}_Septum_PLV"
            else:
                proxy_key = f"work_ps_{strain_comp}_{region}"

            truth_vals = [total_work(r, truth_key) for r in results]
            proxy_vals = [total_work(r, proxy_key) for r in results]

            bt, bp = truth_vals[0], proxy_vals[0]
            if abs(bt) < 1e-12 or abs(bp) < 1e-12:
                continue

            tn = [v / bt for v in truth_vals]
            pn = [v / bp for v in proxy_vals]

            # Diagonal
            lo = min(min(pn), min(tn), 0.5) - 0.1
            hi = max(max(pn), max(tn), 1.1) + 0.1
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.15, lw=1)

            # Points colored by severity
            ax.plot(pn, tn, '-', color='0.7', lw=0.8, zorder=3)
            for i, (px, ty) in enumerate(zip(pn, tn)):
                ax.plot(px, ty, 'o', color=colors[i], ms=8, zorder=5)
                ax.annotate(labels[i], (px, ty), xytext=(6, 6),
                            textcoords='offset points', fontsize=6, color='0.4')

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.grid(True, alpha=0.08)
            if row == 0:
                ax.set_title(f"{region} free wall", fontweight='bold')
            if row == 1:
                ax.set_xlabel("PS proxy (norm.)")
            if col == 0:
                ax.set_ylabel(f"{row_title}\n(normalized)")

    # Septum: multiple pressure variants
    septum_variants = [
        ("PLV",  r"$P_{LV}$",  REGION_COLORS["LV"],     "o"),
        ("Trans", "Trans.",     REGION_COLORS["Septum"],  "s"),
        ("PRV",  r"$P_{RV}$",  REGION_COLORS["RV"],      "^"),
    ]

    for row, row_title, truth_prefix in truth_configs:
        ax = axes[row, 2]
        truth_key = f"{truth_prefix}_Septum"
        truth_vals = [total_work(r, truth_key) for r in results]
        bt = truth_vals[0]
        if abs(bt) < 1e-12:
            continue
        tn = [v / bt for v in truth_vals]

        all_x = []
        for variant, vlbl, vcol, vmk in septum_variants:
            proxy_key = f"work_ps_{strain_comp}_Septum_{variant}"
            proxy_vals = [total_work(r, proxy_key) for r in results]
            bp = proxy_vals[0]
            if abs(bp) < 1e-12:
                continue
            pn = [v / bp for v in proxy_vals]
            all_x.extend(pn)
            ax.plot(pn, tn, f'{vmk}-', color=vcol, lw=1.2, ms=6,
                    label=vlbl if row == 0 else None, zorder=5)

        if all_x:
            lo = min(min(all_x), min(tn), 0.5) - 0.1
            hi = max(max(all_x), max(tn), 1.1) + 0.1
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.15, lw=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        ax.grid(True, alpha=0.08)
        if row == 0:
            ax.set_title("Septum", fontweight='bold')
            ax.legend(fontsize=7, loc='upper left')
        if row == 1:
            ax.set_xlabel("PS proxy (norm.)")

    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    _save(fig, outdir, "proxy_accuracy_spectrum.png")


# ─── Figure 9: Dyssynchrony Across Spectrum ─────────────────────────────────

def plot_dyssynchrony_spectrum(results, labels, colors, outdir):
    """Regional work rate over the cardiac cycle, one panel per case."""
    n = len(results)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows),
                              squeeze=False)

    for j, (r, lbl) in enumerate(zip(results, labels)):
        row, col = divmod(j, ncols)
        ax = axes[row, col]
        t = get_array(r, 'time') * 1000

        for region, reg_c in REGION_COLORS.items():
            w = get_array(r, f"work_true_{region}") * 1e3
            if len(w) == 0 or len(t) == 0:
                continue
            if len(w) == len(t) - 1:
                t_mid = 0.5 * (t[:-1] + t[1:])
                dt = np.diff(t)
                rate = w / dt
            elif len(w) == len(t):
                t_mid = t
                rate = w
            else:
                continue
            ax.plot(t_mid, rate, color=reg_c, lw=1.2, label=region)

        ax.set_title(lbl, fontweight='bold', color=colors[j])
        ax.set_xlabel("Time (ms)")
        ax.axhline(0, color='k', lw=0.5)
        if col == 0:
            ax.set_ylabel("Work rate (mJ/ms)")
        if j == 0:
            ax.legend(fontsize=6)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    _save(fig, outdir, "dyssynchrony_spectrum.png")


# ─── Figure 10: Multi-component Stress-Strain ───────────────────────────────

def plot_stress_components_spectrum(results, labels, colors, outdir):
    """3x3 grid of stress-strain loops (ff, ss, nn) x (LV, RV, Septum)."""
    components = [
        ("ff", r"$S_{ff}$ (kPa)"),
        ("ss", r"$S_{ss}$ (kPa)"),
        ("nn", r"$S_{nn}$ (kPa)"),
    ]
    regions = ["LV", "RV", "Septum"]

    fig, axes = plt.subplots(len(components), 3, figsize=(8.0, 7.5))

    for row, (comp, ylabel) in enumerate(components):
        for col, region in enumerate(regions):
            ax = axes[row, col]
            for r, lbl, c in zip(results, labels, colors):
                E = get_array(r, f"mean_E_{comp}_{region}") * 100
                S = get_array(r, f"mean_S_{comp}_{region}") / 1e3
                if len(E) == 0:
                    continue
                ax.plot(E, S, color=c, lw=1.5, label=lbl)
                ax.plot(E[0], S[0], 'o', color=c, ms=3)

            ax.axhline(0, color='0.5', lw=0.4)
            ax.axvline(0, color='0.5', lw=0.4)
            if row == 0:
                ax.set_title(region, fontweight='bold')
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(components) - 1:
                ax.set_xlabel("Strain (%)")
            if row == 0 and col == 0:
                ax.legend(fontsize=5, loc='best')

    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    _save(fig, outdir, "stress_components_spectrum.png")


# ─── Console Report ──────────────────────────────────────────────────────────

def print_work_report(results, labels):
    W = 110
    regions = ["LV", "RV", "Septum"]

    print("\n" + "=" * W)
    print(f"{'WORK METRICS ACROSS PH SEVERITY SPECTRUM':^{W}}")
    print("=" * W)

    for wtype, wkey in [("Total", "work_true"), ("Fiber", "work_ff")]:
        print(f"\n  -- {wtype} Work (mJ) --")
        header = f"  {'Case':<15}"
        for reg in regions:
            header += f" {reg:>10}"
        header += f" {'Sum':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        baseline_vals = None
        for j, (r, lbl) in enumerate(zip(results, labels)):
            vals = [total_work(r, f"{wkey}_{reg}") * 1e3 for reg in regions]
            tot = sum(vals)
            row = f"  {lbl:<15}"
            for v in vals:
                row += f" {v:>10.2f}"
            row += f" {tot:>10.2f}"

            if baseline_vals is not None:
                # Show ratio to baseline
                ratios = [v / b if abs(b) > 1e-10 else float('nan')
                          for v, b in zip(vals, baseline_vals)]
                row += "  |"
                for ratio in ratios:
                    row += f" {ratio:>5.2f}x"
            else:
                baseline_vals = vals

            print(row)

    # Proxy tracking report
    print(f"\n  -- Proxy Tracking (PS_ll / FEM fiber) Ratio --")
    header = f"  {'Case':<15}"
    for reg in ["LV", "RV", "Sep(PLV)", "Sep(Trans)"]:
        header += f" {reg:>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r, lbl in zip(results, labels):
        ratios = []
        for region in ["LV", "RV"]:
            fem = total_work(r, f"work_ff_{region}")
            proxy = total_work(r, f"work_ps_ll_{region}")
            ratios.append(proxy / fem if abs(fem) > 1e-12 else float('nan'))

        for variant in ["PLV", "Trans"]:
            fem = total_work(r, "work_ff_Septum")
            proxy = total_work(r, f"work_ps_ll_Septum_{variant}")
            ratios.append(proxy / fem if abs(fem) > 1e-12 else float('nan'))

        row = f"  {lbl:<15}"
        for ratio in ratios:
            row += f" {ratio:>10.3f}"
        print(row)

    print("=" * W)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare cardiac simulations across a disease severity spectrum.")
    parser.add_argument("case_dirs", nargs="+",
                        help="Paths to result directories (ordered by severity)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for cases (default: directory basenames)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: results/sims/compare_spectrum)")
    parser.add_argument("--primary", action="store_true",
                        help="Only generate primary figures (skip extras)")
    args = parser.parse_args()

    setup_style()

    folders = args.case_dirs
    n = len(folders)

    if args.labels:
        if len(args.labels) != n:
            print(f"ERROR: {len(args.labels)} labels for {n} cases")
            sys.exit(1)
        labels = args.labels
    else:
        labels = [Path(f).name for f in folders]

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path(__file__).resolve().parent / "results" / "sims" / "compare_spectrum"
    outdir.mkdir(parents=True, exist_ok=True)

    colors = spectrum_colors(n)

    print("=" * 60)
    print("  compare_spectrum.py")
    print(f"  {n} cases across severity spectrum")
    print("=" * 60)

    results = []
    for folder, lbl in zip(folders, labels):
        print(f"\n[{lbl}]  {folder}")
        m = load_metrics(folder)
        if m is None:
            print(f"  ERROR: could not load metrics from {folder}")
            sys.exit(1)
        results.append(m)

    # Reports
    hemo = extract_hemodynamics(results, labels)
    print_hemodynamic_table(hemo)
    print_work_report(results, labels)

    # Figures
    primary_figures = [
        ("Hemodynamic trends",      plot_hemodynamic_trends),
        ("PV loops",                plot_pv_loops),
        ("Stress-strain loops",     plot_stress_strain_spectrum),
        ("Pressure-strain loops",   plot_pressure_strain_spectrum),
        ("Work trends",             plot_work_trends),
        ("Work decomposition",      plot_work_decomposition_spectrum),
        ("Work redistribution",     plot_work_redistribution_spectrum),
        ("Proxy accuracy",          plot_proxy_accuracy_spectrum),
    ]

    extra_figures = [
        ("Dyssynchrony",            plot_dyssynchrony_spectrum),
        ("Stress components",       plot_stress_components_spectrum),
    ]

    figure_set = primary_figures if args.primary else primary_figures + extra_figures

    print(f"\nGenerating {len(figure_set)} figures...")
    for name, fig_fn in figure_set:
        print(f"  {name}...")
        fig_fn(results, labels, colors, outdir)

    print(f"\nDone. Output: {outdir}")


if __name__ == "__main__":
    main()
