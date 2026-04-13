#!/usr/bin/env python3
"""
eval_proxies.py

Per-simulation proxy validation: how well do clinical pressure-strain proxies
track the FEM ground truth (total internal work S:dE)?

Two proxy strain types:
  - Longitudinal (ll): the clinical proxy — measurable via speckle tracking (GLS)
  - Fiber (ff): computational reference — requires known fiber orientation

For the septum, each strain type is evaluated with three pressure definitions:
  Trans (P_LV - P_RV), P_LV only, P_RV only.

Outputs:
  proxy_validation.png  — regression scatter (true vs proxy power)
  proxy_stats.json      — R², slope, ratio per region/proxy

Usage:
  python3 eval_proxies.py <results_folder>
"""

import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

from plot_utils import setup_style, load_metrics, REGION_COLORS, save_fig


# ─── Statistics ───────────────────────────────────────────────────────────────

def _stats(truth, proxy):
    """R², linear slope, and work ratio between two timeseries."""
    if truth is None or proxy is None:
        return None
    n = min(len(truth), len(proxy))
    t, p = truth[:n], proxy[:n]
    W_true, W_proxy = float(np.sum(t)), float(np.sum(p))
    ratio = W_proxy / W_true if abs(W_true) > 1e-12 else 0.0
    r, _ = pearsonr(t, p)
    slope = np.polyfit(t, p, 1)[0]
    return {
        "W_true": W_true, "W_proxy": W_proxy,
        "Ratio": ratio, "R_squared": r**2, "Slope": slope,
    }


def _get(m, key):
    return np.array(m[key]) if key in m else None


# ─── Figure ───────────────────────────────────────────────────────────────────

def _plot_regression(ax, truth, proxy, label, color):
    """Scatter + linear fit. Returns slope."""
    if truth is None or proxy is None:
        return None
    n = min(len(truth), len(proxy))
    x, y = truth[:n], proxy[:n]
    ax.scatter(x, y, alpha=0.25, s=8, color=color, label=label)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, color=color, lw=1.5)
    return m


def analyze_proxies(metrics, outdir):
    setup_style()
    outdir = Path(outdir)

    # Ground truth
    tru = {r: _get(metrics, f"work_true_{r}") for r in ["LV", "RV", "Septum"]}

    # Proxies: ll (clinical) and ff (computational reference)
    proxy = {}
    for strain in ["ll", "ff"]:
        proxy[(strain, "LV")] = _get(metrics, f"work_ps_{strain}_LV")
        proxy[(strain, "RV")] = _get(metrics, f"work_ps_{strain}_RV")
        for pdef in ["Trans", "PLV", "PRV"]:
            proxy[(strain, "Septum", pdef)] = _get(metrics, f"work_ps_{strain}_Septum_{pdef}")

    # ── Figure: 2 rows (ll, ff) × 3 cols (LV, Septum, RV) ──
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    strain_labels = {"ll": "Longitudinal (clinical)", "ff": "Fiber (computational)"}
    sep_colors = {"Trans": "tab:green", "PLV": REGION_COLORS["LV"], "PRV": REGION_COLORS["RV"]}

    all_stats = {}

    for row, strain in enumerate(["ll", "ff"]):
        all_stats[strain] = {}

        # LV
        ax = axes[row, 0]
        t = tru["LV"]
        p = proxy[(strain, "LV")]
        if t is not None:
            ax.plot([t.min(), t.max()], [t.min(), t.max()], "k--", alpha=0.4, lw=0.8)
        s = _stats(t, p)
        _plot_regression(ax, t, p, "PS proxy", REGION_COLORS["LV"])
        all_stats[strain]["LV"] = s
        ax.set_title(f"LV — {strain_labels[strain]}")
        if s:
            ax.text(0.05, 0.92, f"R²={s['R_squared']:.3f}  slope={s['Slope']:.3f}",
                    transform=ax.transAxes, fontsize=8, va="top")
        if row == 1:
            ax.set_xlabel("True work rate (S:dE)")
        ax.set_ylabel("Proxy work rate")

        # Septum
        ax = axes[row, 1]
        t = tru["Septum"]
        if t is not None:
            ax.plot([t.min(), t.max()], [t.min(), t.max()], "k--", alpha=0.4, lw=0.8)
        all_stats[strain]["Septum"] = {}
        y_offset = 0.92
        for pdef in ["Trans", "PLV", "PRV"]:
            p = proxy[(strain, "Septum", pdef)]
            label = {"Trans": "ΔP (LV−RV)", "PLV": "P_LV", "PRV": "P_RV"}[pdef]
            _plot_regression(ax, t, p, label, sep_colors[pdef])
            s = _stats(t, p)
            all_stats[strain]["Septum"][pdef] = s
            if s:
                ax.text(0.05, y_offset, f"{label}: R²={s['R_squared']:.3f}  slope={s['Slope']:.3f}",
                        transform=ax.transAxes, fontsize=7, va="top", color=sep_colors[pdef])
                y_offset -= 0.09
        ax.set_title(f"Septum — {strain_labels[strain]}")
        ax.legend(fontsize=7, loc="lower right")
        if row == 1:
            ax.set_xlabel("True work rate (S:dE)")

        # RV
        ax = axes[row, 2]
        t = tru["RV"]
        p = proxy[(strain, "RV")]
        if t is not None:
            ax.plot([t.min(), t.max()], [t.min(), t.max()], "k--", alpha=0.4, lw=0.8)
        s = _stats(t, p)
        _plot_regression(ax, t, p, "PS proxy", REGION_COLORS["RV"])
        all_stats[strain]["RV"] = s
        ax.set_title(f"RV — {strain_labels[strain]}")
        if s:
            ax.text(0.05, 0.92, f"R²={s['R_squared']:.3f}  slope={s['Slope']:.3f}",
                    transform=ax.transAxes, fontsize=8, va="top")
        if row == 1:
            ax.set_xlabel("True work rate (S:dE)")

    fig.suptitle("Proxy vs True Internal Work (S:dE)", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, outdir, "proxy_validation.png")

    # ── Console report ──
    print("\n" + "=" * 72)
    print(f"{'PROXY EVALUATION REPORT':^72}")
    print("=" * 72)
    print(f"{'Strain':<8} {'Region / Pressure':<25} {'Ratio':>8} {'R²':>8} {'Slope':>8}")
    print("-" * 72)
    for strain in ["ll", "ff"]:
        tag = "ε_ll" if strain == "ll" else "ε_ff"
        for region in ["LV", "RV"]:
            s = all_stats[strain].get(region)
            if s:
                print(f"{tag:<8} {region:<25} {s['Ratio']:>8.3f} {s['R_squared']:>8.3f} {s['Slope']:>8.3f}")
        for pdef in ["Trans", "PLV", "PRV"]:
            s = all_stats[strain]["Septum"].get(pdef)
            plabel = {"Trans": "Septum ΔP(LV-RV)", "PLV": "Septum P_LV", "PRV": "Septum P_RV"}[pdef]
            if s:
                print(f"{tag:<8} {plabel:<25} {s['Ratio']:>8.3f} {s['R_squared']:>8.3f} {s['Slope']:>8.3f}")
        print("-" * 72)
    print("=" * 72)

    # ── JSON ──
    json_path = outdir / "proxy_stats.json"
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"  Saved: {json_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pressure-strain proxies against FEM ground truth")
    parser.add_argument("results_folder", type=str)
    args = parser.parse_args()

    res_dir = Path(args.results_folder)
    if not res_dir.exists():
        print(f"Error: {res_dir} not found")
        sys.exit(1)

    metrics = load_metrics(res_dir)
    if metrics:
        analyze_proxies(metrics, res_dir)
