#!/usr/bin/env python3
"""
eval_proxies.py

Per-simulation proxy check: how big is each clinical pressure-strain proxy's
work loop compared with the FEM ground truth (total internal work S:dE)?

What this reports per region/pressure is deliberately scalar:
  - W_true  : loop area = total integral of the true work (∫ S:dE)
  - W_proxy : loop area of the proxy (∫ P dε)
  - Ratio   : W_proxy / W_true

We do NOT report a through-beat R² here. Point-by-point correlation of the
work *rate* within a single beat mostly reflects shared cardiac timing and is
not the scientific question. Whether the proxy *follows* the truth is answered
across simulations by sweep_analysis.py (cross-simulation correlation); within
a single sim only the loop area and its ratio are meaningful.

Two proxy strain types:
  - Longitudinal (ll): the clinical proxy — measurable via speckle tracking (GLS)
  - Fiber (ff): computational reference — requires known fiber orientation
For the septum each strain is evaluated with three pressures: Trans (P_LV−P_RV),
P_LV only, P_RV only.

Outputs:
  proxy_validation.png  — true-vs-proxy work-rate scatter against the identity line
  proxy_stats.json      — W_true, W_proxy, Ratio per region/proxy

Usage:
  python3 eval_proxies.py <results_folder>
"""

import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from plot_utils import setup_style, load_metrics, REGION_COLORS, save_fig


# ─── Statistics ───────────────────────────────────────────────────────────────

def _stats(truth, proxy):
    """Loop area (total integral) of truth and proxy, and their ratio."""
    if truth is None or proxy is None:
        return None
    n = min(len(truth), len(proxy))
    t, p = truth[:n], proxy[:n]
    W_true, W_proxy = float(np.sum(t)), float(np.sum(p))
    # NaN, not 0.0: with no true work to compare against the ratio is undefined,
    # and 0.0 would read as "the proxy captures none of the work".
    ratio = W_proxy / W_true if abs(W_true) > 1e-12 else float("nan")
    return {"W_true": W_true, "W_proxy": W_proxy, "Ratio": ratio}


def _get(m, key):
    return np.array(m[key]) if key in m else None


# ─── Figure ───────────────────────────────────────────────────────────────────

def _plot_scatter(ax, truth, proxy, label, color):
    """Scatter of proxy vs true work rate. Deviation from the identity line
    (drawn separately) shows the area ratio; no through-beat fit is drawn."""
    if truth is None or proxy is None:
        return
    n = min(len(truth), len(proxy))
    ax.scatter(truth[:n], proxy[:n], alpha=0.25, s=8, color=color, label=label)


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
        _plot_scatter(ax, t, p, "PS proxy", REGION_COLORS["LV"])
        all_stats[strain]["LV"] = s
        ax.set_title(f"LV — {strain_labels[strain]}")
        if s:
            ax.text(0.05, 0.92, f"ratio={s['Ratio']:.3f}",
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
            _plot_scatter(ax, t, p, label, sep_colors[pdef])
            s = _stats(t, p)
            all_stats[strain]["Septum"][pdef] = s
            if s:
                ax.text(0.05, y_offset, f"{label}: ratio={s['Ratio']:.3f}",
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
        _plot_scatter(ax, t, p, "PS proxy", REGION_COLORS["RV"])
        all_stats[strain]["RV"] = s
        ax.set_title(f"RV — {strain_labels[strain]}")
        if s:
            ax.text(0.05, 0.92, f"ratio={s['Ratio']:.3f}",
                    transform=ax.transAxes, fontsize=8, va="top")
        if row == 1:
            ax.set_xlabel("True work rate (S:dE)")

    fig.suptitle("Proxy vs True Internal Work (S:dE) — loop areas", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, outdir, "proxy_validation.png")

    # ── Console report ──
    print("\n" + "=" * 64)
    print(f"{'PROXY LOOP-AREA REPORT':^64}")
    print("=" * 64)
    print(f"{'Strain':<8} {'Region / Pressure':<25} {'W_true':>10} {'Ratio':>8}")
    print("-" * 64)
    for strain in ["ll", "ff"]:
        tag = "ε_ll" if strain == "ll" else "ε_ff"
        for region in ["LV", "RV"]:
            s = all_stats[strain].get(region)
            if s:
                print(f"{tag:<8} {region:<25} {s['W_true']:>10.4f} {s['Ratio']:>8.3f}")
        for pdef in ["Trans", "PLV", "PRV"]:
            s = all_stats[strain]["Septum"].get(pdef)
            plabel = {"Trans": "Septum ΔP(LV-RV)", "PLV": "Septum P_LV", "PRV": "Septum P_RV"}[pdef]
            if s:
                print(f"{tag:<8} {plabel:<25} {s['W_true']:>10.4f} {s['Ratio']:>8.3f}")
        print("-" * 64)
    print("=" * 64)

    # ── JSON ──
    json_path = outdir / "proxy_stats.json"
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"  Saved: {json_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-sim proxy loop-area check vs FEM ground truth")
    parser.add_argument("results_folder", type=str)
    args = parser.parse_args()

    res_dir = Path(args.results_folder)
    if not res_dir.exists():
        print(f"Error: {res_dir} not found")
        sys.exit(1)

    metrics = load_metrics(res_dir)
    analyze_proxies(metrics, res_dir)
