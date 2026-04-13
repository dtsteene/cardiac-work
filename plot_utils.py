"""
Shared utilities for visualization and analysis scripts.

Provides:
- Publication-quality matplotlib style
- Metrics loading with multi-layout search and legacy key normalization
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
    """Load metrics dict, searching multiple directory layout conventions.

    Searches (in order): analysis/last_beat/, analysis_last_beat/, metrics/, root.
    Applies legacy key normalization so all downstream code uses current names.
    """
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
    """Map legacy metric key names to current convention."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_array(m, key):
    """Safely extract a numpy array from metrics dict."""
    return np.array(m[key]) if key in m else np.array([])


def total_work(m, key):
    """Sum a work timeseries to get total work (J)."""
    if key not in m:
        return 0.0
    return float(np.sum(m[key]))


def save_fig(fig, outdir, name):
    """Save figure and close."""
    out = Path(outdir) / name
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)
