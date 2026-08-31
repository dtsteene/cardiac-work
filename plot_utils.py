"""
Shared utilities for visualization and analysis scripts.

Provides:
- Publication-quality matplotlib style
- Metrics loading from a results folder
- Common color palettes and helper functions
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


# ─── Color Palettes ──────────────────────────────────────────────────────────

REGION_COLORS = {"LV": "#4878A8", "RV": "#C25454", "Septum": "#5EA55E"}
CASE_COLORS = ["#4878A8", "#C25454"]  # Healthy = steel blue, PAH = muted red


def spectrum_colors(n):
    """Sequential colormap from blue (healthy) to red (severe)."""
    cmap = mpl.colormaps.get_cmap("RdYlBu_r")
    return [cmap(0.15 + 0.7 * i / max(n - 1, 1)) for i in range(n)]


# ─── Thesis Figure Style ─────────────────────────────────────────────────────

def setup_style():
    """Configure matplotlib for clean, publication-quality figures."""
    mpl.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        # Axes
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Grid (when enabled explicitly)
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
    })


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_metrics(folder):
    """Load the metrics dict written by postprocess_metrics.py.

    Reads `<folder>/metrics/` (the run-root layout), falling back to `<folder>`
    itself (the per-beat layout: run_postprocessing.py passes analysis/last_beat
    and analysis/all_beats, where the file sits flat). Prefers the finest
    downsampling available (metrics_downsample_1.npy over _10.npy).

    Raises FileNotFoundError if the run has no metrics — an unpostprocessed or
    mistyped run should stop the caller, not yield an empty figure.
    """
    root = Path(folder)
    candidates = [root / "metrics", root]
    files = []
    for d in candidates:
        files = sorted(
            d.glob("metrics_downsample_*.npy"),
            key=lambda p: int(p.stem.rsplit("_", 1)[1]),
        )
        if files:
            break
    if not files:
        raise FileNotFoundError(
            f"No metrics_downsample_*.npy in {candidates[0]} or {candidates[1]}. "
            f"Run postprocess_metrics.py on this results folder first."
        )
    print(f"  Loading: {files[0]}")
    return np.load(files[0], allow_pickle=True).item()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def save_fig(fig, outdir, name):
    """Save figure and close."""
    out = Path(outdir) / name
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)
