#!/usr/bin/env python3
"""
compare_mesh_geometry.py

Side-by-side geometry comparison for any number of simulation cases.
Designed for the three-mesh comparison:
  - Healthy patient-specific
  - PAH patient-specific
  - UKB atlas (synthetic)

Usage:
  python3 compare_mesh_geometry.py dir1/ dir2/ dir3/
  python3 compare_mesh_geometry.py dir1/ dir2/ dir3/ \
      --labels "Healthy" "PAH" "UKB" \
      --outdir results/comparisons/mesh_geometry/
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


# ─── Style ───────────────────────────────────────────────────────────────────

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
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
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


REGION_COLORS = {"LV": "#4878A8", "RV": "#C25454", "Septum": "#5EA55E"}

# Default case colors — extended palette for up to 5 cases
CASE_COLORS = ["#4878A8", "#C25454", "#7B68AA", "#D4A030", "#5EA55E"]


def _save(fig, outdir, name):
    out = Path(outdir) / name
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_metrics(folder):
    path = Path(folder)
    for d in [path / "analysis" / "last_beat", path / "analysis_last_beat",
              path / "metrics", path]:
        if not d.exists():
            continue
        files = sorted(d.glob("metrics_downsample_*.npy"), key=lambda p: len(str(p)))
        if files:
            print(f"  Loading: {files[0]}")
            return np.load(files[0], allow_pickle=True).item()
    print(f"  No metrics found in {folder}")
    return None


def total_work(m, key):
    return float(np.sum(m[key])) if key in m else 0.0


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_geometry_context(results, labels, colors, outdir):
    """
    Three-panel geometry context plot for N cases.
      Panel 1: Grouped bar — wall volume per region per case (mL)
      Panel 2: Grouped bar — work density per region per case (mJ/mL)
      Panel 3: Stacked bar — volume composition (%) per case
    """
    regions = ["LV", "RV", "Septum"]
    vol_m3_to_mL = 1e6
    n = len(results)

    volumes = {}
    for j, r in enumerate(results):
        for region in regions:
            key = f"region_volume_{region}"
            volumes[(j, region)] = float(r.get(key, 0.0)) * vol_m3_to_mL

    # Print table
    print(f"\n  {'Case':<18} {'LV (mL)':>10} {'RV (mL)':>10} {'Sep (mL)':>10} {'Total':>10}")
    print("  " + "-" * 56)
    for j, lbl in enumerate(labels):
        vols = [volumes[(j, reg)] for reg in regions]
        print(f"  {lbl:<18} {vols[0]:>10.1f} {vols[1]:>10.1f} {vols[2]:>10.1f} {sum(vols):>10.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2))

    bar_w = 0.6 / n  # total group width = 0.6; distribute evenly
    x = np.arange(len(regions))

    # Panel 1: wall volumes
    ax = axes[0]
    for j, (lbl, c) in enumerate(zip(labels, colors)):
        offsets = (j - (n - 1) / 2) * bar_w
        vals = [volumes[(j, reg)] for reg in regions]
        bars = ax.bar(x + offsets, vals, bar_w * 0.92,
                      label=lbl, color=c, alpha=0.85, edgecolor='none')
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                        f"{v:.0f}", ha='center', va='bottom',
                        fontsize=6, color='0.3')
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Wall volume (mL)")
    ax.set_title("Region volumes", fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')

    # Panel 2: work density (mJ / mL)
    ax = axes[1]
    for j, (lbl, c) in enumerate(zip(labels, colors)):
        offsets = (j - (n - 1) / 2) * bar_w
        densities = []
        for region in regions:
            w = total_work(results[j], f"work_true_{region}") * 1e3
            v = volumes[(j, region)]
            densities.append(w / v if v > 0 else 0.0)
        ax.bar(x + offsets, densities, bar_w * 0.92,
               label=lbl, color=c, alpha=0.85, edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Work density (mJ/mL)")
    ax.set_title("Work density", fontweight='bold')
    ax.axhline(0, color='k', lw=0.5)

    # Panel 3: volume composition (stacked, one bar per case)
    ax = axes[2]
    x3 = np.arange(n)
    for j, (lbl, c) in enumerate(zip(labels, colors)):
        vols = [volumes[(j, reg)] for reg in regions]
        tot = sum(vols) or 1.0
        bottom = 0.0
        for reg, v in zip(regions, vols):
            pct = v / tot * 100
            ax.bar(j, pct, bottom=bottom, color=REGION_COLORS[reg],
                   alpha=0.85, edgecolor='white', lw=0.5,
                   label=reg if j == 0 else None)
            if pct > 5:
                ax.text(j, bottom + pct / 2, f"{pct:.0f}%",
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
            bottom += pct
    ax.set_xticks(x3)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Volume fraction (%)")
    ax.set_title("Composition", fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])

    fig.tight_layout(w_pad=1.0)
    _save(fig, outdir, "geometry_context.png")


def plot_geometry_context_stacked(results, labels, colors, outdir):
    """
    Compact stacked-bar version: one bar per case, stacked by region.
    Annotates absolute volumes and percentages.
    """
    regions = ["LV", "RV", "Septum"]
    vol_m3_to_mL = 1e6
    n = len(results)

    volumes = {}
    for j, r in enumerate(results):
        for region in regions:
            key = f"region_volume_{region}"
            volumes[(j, region)] = float(r.get(key, 0.0)) * vol_m3_to_mL

    fig, ax = plt.subplots(figsize=(max(3.2, n * 1.4), 3.5))

    x = np.arange(n)
    bar_w = min(0.55, 0.9 / n * n)  # stays reasonable for 2-5 cases

    for j in range(n):
        vols = [volumes[(j, reg)] for reg in regions]
        tot = sum(vols) or 1.0
        bottom = 0.0
        for reg, v in zip(regions, vols):
            pct = v / tot * 100
            ax.bar(j, v, bar_w, bottom=bottom, color=REGION_COLORS[reg],
                   alpha=0.85, edgecolor='white', lw=0.5,
                   label=reg if j == 0 else None)
            if pct > 6:
                ax.text(j, bottom + v / 2,
                        f"{v:.0f} mL\n({pct:.0f}%)",
                        ha='center', va='center', fontsize=7.5,
                        color='white', fontweight='bold')
            bottom += v
        # Total label above bar
        ax.text(j, tot * vol_m3_to_mL / vol_m3_to_mL + 1,
                f"{tot:.0f} mL",
                ha='center', va='bottom', fontsize=7, color='0.35')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Wall volume (mL)")
    ax.set_title("Region wall volumes — mesh comparison", fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)

    fig.tight_layout()
    _save(fig, outdir, "geometry_context_stacked.png")


def plot_geometry_context_combined(results, labels, colors, outdir):
    """
    2×2 panel combining all geometry information in a single presentation figure.
      [0,0] Stacked absolute volumes  [0,1] Volume fractions (%)
      [1,0] Work density by region    [1,1] Total work by region
    """
    regions = ["LV", "RV", "Septum"]
    vol_m3_to_mL = 1e6
    n = len(results)
    x = np.arange(n)
    bar_w = min(0.55, 0.85)

    volumes = {}
    for j, r in enumerate(results):
        for region in regions:
            volumes[(j, region)] = float(r.get(f"region_volume_{region}", 0.0)) * vol_m3_to_mL

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))

    # [0,0] Stacked absolute volumes
    ax = axes[0, 0]
    for j in range(n):
        bot = 0.0
        for reg in regions:
            v = volumes[(j, reg)]
            ax.bar(j, v, bar_w, bottom=bot, color=REGION_COLORS[reg],
                   edgecolor='white', lw=0.5, label=reg if j == 0 else None)
            if v / (sum(volumes[(j, r)] for r in regions) or 1) > 0.06:
                ax.text(j, bot + v / 2, f"{v:.0f}",
                        ha='center', va='center', fontsize=7.5,
                        color='white', fontweight='bold')
            bot += v
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Wall volume (mL)")
    ax.set_title("Absolute wall volumes", fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')

    # [0,1] Volume fractions
    ax = axes[0, 1]
    group_w = 0.6 / len(regions)
    xr = np.arange(len(regions))
    for j, (lbl, c) in enumerate(zip(labels, colors)):
        offsets = (j - (n - 1) / 2) * (0.6 / n)
        vols = [volumes[(j, reg)] for reg in regions]
        tot = sum(vols) or 1.0
        fracs = [v / tot * 100 for v in vols]
        ax.bar(xr + offsets, fracs, 0.6 / n * 0.9, label=lbl, color=c,
               alpha=0.85, edgecolor='none')
    ax.set_xticks(xr)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Volume fraction (%)")
    ax.set_title("Composition by region", fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, 75)

    # [1,0] Work density
    ax = axes[1, 0]
    for j, (lbl, c) in enumerate(zip(labels, colors)):
        offsets = (j - (n - 1) / 2) * (0.6 / n)
        dens = []
        for region in regions:
            w = total_work(results[j], f"work_true_{region}") * 1e3
            v = volumes[(j, region)]
            dens.append(w / v if v > 0 else 0.0)
        ax.bar(xr + offsets, dens, 0.6 / n * 0.9, label=lbl, color=c,
               alpha=0.85, edgecolor='none')
    ax.set_xticks(xr)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Work density (mJ/mL)")
    ax.set_title("Work density", fontweight='bold')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=7, loc='upper right')

    # [1,1] Total work (mJ) per region
    ax = axes[1, 1]
    for j, (lbl, c) in enumerate(zip(labels, colors)):
        offsets = (j - (n - 1) / 2) * (0.6 / n)
        works = [total_work(results[j], f"work_true_{reg}") * 1e3 for reg in regions]
        ax.bar(xr + offsets, works, 0.6 / n * 0.9, label=lbl, color=c,
               alpha=0.85, edgecolor='none')
    ax.set_xticks(xr)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Work (mJ)")
    ax.set_title("Total work per region", fontweight='bold')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    _save(fig, outdir, "geometry_context_combined.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Geometry context comparison across multiple meshes.")
    parser.add_argument("case_dirs", nargs="+",
                        help="Paths to simulation result directories")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    setup_style()

    n = len(args.case_dirs)
    labels = args.labels if args.labels else [Path(f).name for f in args.case_dirs]
    if len(labels) != n:
        print(f"ERROR: {len(labels)} labels for {n} cases"); sys.exit(1)

    colors = CASE_COLORS[:n]

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path(__file__).resolve().parent / "results" / "comparisons" / "mesh_geometry"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  compare_mesh_geometry.py")
    print(f"  {n} meshes")
    print("=" * 60)

    results = []
    for folder, lbl in zip(args.case_dirs, labels):
        print(f"\n[{lbl}]  {folder}")
        m = load_metrics(folder)
        if m is None:
            print(f"  ERROR: could not load metrics"); sys.exit(1)
        results.append(m)

    print("\nGenerating figures...")
    plot_geometry_context(results, labels, colors, outdir)
    plot_geometry_context_stacked(results, labels, colors, outdir)
    plot_geometry_context_combined(results, labels, colors, outdir)

    print(f"\nDone. Output: {outdir}")


if __name__ == "__main__":
    main()
