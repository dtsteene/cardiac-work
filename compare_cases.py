#!/usr/bin/env python3
"""
compare_cases.py

Compare two cardiac simulation cases (e.g. Healthy vs PAH).
Generates publication-quality figures for the thesis.

Usage:
  python3 compare_cases.py <CASE_A_DIR> <CASE_B_DIR> [--labels A B]
  python3 compare_cases.py healthy_dir/ pah_dir/
  python3 compare_cases.py healthy/ pah/ --labels "Healthy" "PAH" --outdir figures/
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


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


# Two-case color palette: Healthy = steel blue, PAH = muted red
COLORS = ["#4878A8", "#C25454"]
REGION_COLORS = {"LV": "#4878A8", "RV": "#C25454", "Septum": "#5EA55E"}


def _save(fig, outdir, name):
    out = Path(outdir) / name
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_metrics(folder):
    """Load metrics dict from folder, searching multiple layout conventions."""
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
    """Map legacy metric key names to current convention so all code works uniformly."""
    # Old name -> new name (only copy if new name doesn't already exist)
    renames = {}
    regions = ["LV", "RV", "Septum", "Whole"]
    for reg in regions:
        renames[f"work_fiber_{reg}"]  = f"work_ff_{reg}"
        renames[f"work_sheet_{reg}"]  = f"work_ss_{reg}"
        renames[f"work_normal_{reg}"] = f"work_nn_{reg}"
        renames[f"work_shear_{reg}"]  = f"work_cross_{reg}"
    # PS proxy keys: work_ps_index_* -> work_ps_ff_*
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


def compute_ps_work(m, strain_key, p_key):
    """Pressure-strain work index (unscaled) in Joules."""
    strain = get_array(m, strain_key)
    pressure = get_array(m, p_key) * 133.322  # mmHg -> Pa
    if len(strain) < 2 or len(pressure) < 2:
        return 0.0
    dE = np.diff(strain)
    p_avg = 0.5 * (pressure[:-1] + pressure[1:])
    return float(np.sum(p_avg * dE))


def compute_ps_work_scaled(m, strain_key, p_key, region):
    """PS work with volume scaling derived from precomputed fiber-strain proxy."""
    if region == "Septum":
        precomputed_key = "work_ps_ff_Septum_PLV"
        ff_strain_key = "mean_E_ff_Septum"
        ff_p_key = "p_LV"
    else:
        precomputed_key = f"work_ps_ff_{region}"
        ff_strain_key = f"mean_E_ff_{region}"
        ff_p_key = "p_LV" if region == "LV" else "p_RV"

    precomputed_total = total_work(m, precomputed_key)
    raw_ff = compute_ps_work(m, ff_strain_key, ff_p_key)
    if abs(raw_ff) < 1e-20:
        return 0.0
    vol_factor = precomputed_total / raw_ff
    raw = compute_ps_work(m, strain_key, p_key)
    return raw * vol_factor


# ─── Figure 1: Simplification Cascade ────────────────────────────────────────

def plot_simplification_cascade(results, labels, outdir):
    """
    3x3 grid: the cascade from FEM ground truth to clinical measurement.
      Row 0: S_ff vs E_ff  (FEM stress-strain)
      Row 1: P   vs E_ff   (replace stress with pressure)
      Row 2: P   vs E_ll   (replace fiber strain with longitudinal ~ GLS)
    Columns: LV, RV, Septum.
    """
    regions = [
        ("LV",     "mean_E_ff_LV",     "mean_S_ff_LV",     "mean_E_ll_LV",     "p_LV"),
        ("RV",     "mean_E_ff_RV",     "mean_S_ff_RV",     "mean_E_ll_RV",     "p_RV"),
        ("Septum", "mean_E_ff_Septum", "mean_S_ff_Septum", "mean_E_ll_Septum", "p_LV"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(7.0, 7.5))

    row_info = [
        (r"$S_{ff}$ vs $E_{ff}$  (FEM ground truth)",
         "Fiber strain $E_{ff}$ (%)", r"Fiber stress $S_{ff}$ (kPa)"),
        (r"$P$ vs $E_{ff}$  (pressure substitution)",
         "Fiber strain $E_{ff}$ (%)", "Pressure (mmHg)"),
        (r"$P$ vs $E_{ll}$  (clinical measurement)",
         "Longitudinal strain $E_{ll}$ (%)", "Pressure (mmHg)"),
    ]

    for col, (region, eff_key, sff_key, ell_key, p_key) in enumerate(regions):
        for row in range(3):
            ax = axes[row, col]

            for i, (r, lbl, c) in enumerate(zip(results, labels, COLORS)):
                if row == 0:
                    x = get_array(r, eff_key) * 100
                    y = get_array(r, sff_key) / 1e3
                elif row == 1:
                    x = get_array(r, eff_key) * 100
                    y = get_array(r, p_key)
                else:
                    x = get_array(r, ell_key) * 100
                    y = get_array(r, p_key)

                if len(x) == 0:
                    continue
                ax.plot(x, y, color=c, lw=1.5, label=lbl)
                ax.plot(x[0], y[0], 'o', color=c, ms=4, zorder=5)

                # Septum: add transmural pressure as dashed
                if region == "Septum" and row >= 1:
                    p_rv = get_array(r, 'p_RV')
                    p_trans = get_array(r, p_key) - p_rv
                    ax.plot(x, p_trans, color=c, lw=1.2, ls='--', alpha=0.7,
                            label=f"{lbl} (trans.)" if row == 1 else None)

            ax.axhline(0, color='0.5', lw=0.4)
            ax.axvline(0, color='0.5', lw=0.4)

            if row == 0:
                ax.set_title(region, fontweight="bold")
            if col == 0:
                ax.set_ylabel(row_info[row][2])
            if row == 2:
                ax.set_xlabel(row_info[row][1])

            if row == 0 and col == 0:
                ax.legend(fontsize=6, loc='upper right', handlelength=1.0)
            if row == 1 and col == 2:
                ax.legend(fontsize=5, loc='upper right', handlelength=1.0)

    # Row labels on the left margin
    for row, (rlabel, _, _) in enumerate(row_info):
        axes[row, 0].annotate(
            rlabel, xy=(-0.45, 0.5), xycoords='axes fraction',
            fontsize=8, ha='right', va='center', rotation=90, style='italic')

    fig.tight_layout(rect=[0.06, 0, 1, 1], h_pad=1.0, w_pad=0.8)
    _save(fig, outdir, "simplification_cascade.png")


# ─── Figure 2: Pressure-Strain Loops ─────────────────────────────────────────

# ─── Figure 3: Fiber Stress-Strain Loops ─────────────────────────────────────

def plot_stress_loops(results, labels, outdir):
    """
    2x3 grid:
      Row 0: fiber stress-strain loops (S_ff vs E_ff)
      Row 1: fiber strain and pressure vs time
    """
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5))

    regions = [
        ("LV",     "mean_E_ff_LV",     "mean_S_ff_LV",     "p_LV"),
        ("RV",     "mean_E_ff_RV",     "mean_S_ff_RV",     "p_RV"),
        ("Septum", "mean_E_ff_Septum", "mean_S_ff_Septum", "p_LV"),
    ]

    for col, (region, strain_key, stress_key, p_key) in enumerate(regions):
        ax_loop = axes[0, col]
        ax_time = axes[1, col]

        for r, lbl, c in zip(results, labels, COLORS):
            t    = get_array(r, 'time')
            Eff  = get_array(r, strain_key) * 100
            Sff  = get_array(r, stress_key) / 1e3
            pres = get_array(r, p_key)
            if len(Eff) == 0:
                continue

            # Row 0: stress-strain loop
            ax_loop.plot(Eff, Sff, color=c, lw=1.5, label=lbl)
            ax_loop.plot(Eff[0], Sff[0], 'o', color=c, ms=3)

            # Row 1: strain vs time
            ax_time.plot(t * 1000, Eff, color=c, lw=1.5, label=lbl)
            # pressure on twin axis
            ax2 = ax_time.twinx() if col == 2 else ax_time.twinx()
            ax2.plot(t * 1000, pres, color=c, lw=0.8, ls=':', alpha=0.5)
            ax2.tick_params(axis='y', labelsize=7, colors='0.5')
            if col == 2:
                ax2.set_ylabel("$P$ (mmHg)", fontsize=8, color='0.5')
            else:
                ax2.set_yticklabels([])
            ax2.spines['right'].set_visible(True)
            ax2.spines['right'].set_color('0.8')

        ax_loop.axhline(0, color='0.5', lw=0.4)
        ax_loop.axvline(0, color='0.5', lw=0.4)
        ax_loop.set_title(region, fontweight="bold")
        if col == 0:
            ax_loop.set_ylabel(r"$S_{ff}$ (kPa)")
        if col == 2:
            ax_loop.legend(fontsize=7, loc='upper right')
        ax_time.set_xlabel("Time (ms)")
        if col == 0:
            ax_time.set_ylabel(r"$E_{ff}$ (%)")

    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    _save(fig, outdir, "stress_loops.png")


# ─── Figure 4: Work Decomposition ────────────────────────────────────────────

def plot_work_decomposition(results, labels, outdir):
    """
    1x3 grouped bar chart: fiber/sheet/normal/cross work for LV, RV, Septum.
    PS proxy shown as grouped bars (same style as components) after a separator.
    """
    components = [
        ("work_ff",    "Fiber"),
        ("work_ss",    "Sheet"),
        ("work_nn",    "Normal"),
        ("work_cross", "Cross"),
    ]

    def _ps_ll(r, region, p_variant="default"):
        """Get precomputed PS proxy work (mJ) using longitudinal strain."""
        if region in ("LV", "RV"):
            key = f"work_ps_ll_{region}"
        elif p_variant == "default":
            key = "work_ps_ll_Septum_PLV"
        elif p_variant == "trans":
            key = "work_ps_ll_Septum_Trans"
        else:
            raise ValueError(f"Unknown p_variant: {p_variant}")
        if key not in r:
            raise KeyError(f"Missing '{key}' — rerun postprocessing")
        return total_work(r, key) * 1e3

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0))

    for ax, region in zip(axes, ["LV", "RV", "Septum"]):
        n = len(results)
        width = 0.35

        # Build per-region component + proxy list
        if region in ("LV", "RV"):
            all_comps = components + [("ps_ll", "PS")]
        else:
            all_comps = components + [("ps_ll_plv", r"PS $P_{LV}$"),
                                      ("ps_ll_trans", "PS Trans.")]

        def _get_val(r, ckey, region):
            if ckey == "ps_ll":
                return _ps_ll(r, region)
            elif ckey == "ps_ll_plv":
                return _ps_ll(r, region, "default")
            elif ckey == "ps_ll_trans":
                return _ps_ll(r, region, "trans")
            else:
                return total_work(r, f"{ckey}_{region}") * 1e3

        x = np.arange(len(all_comps))

        for j, (r, lbl) in enumerate(zip(results, labels)):
            vals = [_get_val(r, ckey, region) for ckey, _ in all_comps]
            offset = (j - (n - 1) / 2) * width
            ax.bar(x + offset, vals, width,
                   label=lbl, color=COLORS[j],
                   alpha=0.85 if j == 0 else 0.65,
                   edgecolor='none')

        # Separator between decomposition components and PS proxy
        sep_x = len(components) - 0.5
        ax.axvline(sep_x, color='gray', lw=0.7, ls='--', alpha=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels([clbl for _, clbl in all_comps], fontsize=7)
        ax.set_title(region, fontweight='bold')
        ax.axhline(0, color='k', lw=0.5)
        if region == "LV":
            ax.set_ylabel("Work (mJ)")
            ax.legend(fontsize=6.5, loc='best')

    fig.tight_layout(w_pad=0.8)
    _save(fig, outdir, "work_decomposition.png")


def plot_work_decomposition_stacked(results, labels, outdir):
    """
    1x3 stacked bar chart: per-case bars for each region, stacked by component.
    Each segment labelled with its % of |total work|.
    PS proxy shown as grouped bars (case-colored) after a separator.
    """
    components = [
        ("work_ff",    "Fiber"),
        ("work_ss",    "Sheet"),
        ("work_nn",    "Normal"),
        ("work_cross", "Cross"),
    ]
    # Distinct colors per component (colorblind-friendly)
    COMP_COLORS = ["#4878A8", "#6AB187", "#C25454", "#9B59B6"]

    def _ps_ll_val(r, region, p_variant="default"):
        if region in ("LV", "RV"):
            key = f"work_ps_ll_{region}"
        elif p_variant == "default":
            key = "work_ps_ll_Septum_PLV"
        elif p_variant == "trans":
            key = "work_ps_ll_Septum_Trans"
        else:
            return 0.0
        return total_work(r, key) * 1e3 if key in r else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5))
    n = len(results)

    for ax, region in zip(axes, ["LV", "RV", "Septum"]):
        width_stacked = 0.5
        width_ps = 0.35

        # --- Stacked bars at x = 0, 1, ..., n-1 ---
        for j, r in enumerate(results):
            vals = [total_work(r, f"{ckey}_{region}") * 1e3 for ckey, _ in components]
            total_abs = sum(abs(v) for v in vals)

            pos_bot = 0.0
            neg_bot = 0.0
            for k, (v, (_, clbl)) in enumerate(zip(vals, components)):
                color = COMP_COLORS[k]
                bot = pos_bot if v >= 0 else neg_bot
                ax.bar(j, v, width_stacked, bottom=bot, color=color, alpha=0.85,
                       edgecolor='white', lw=0.5,
                       label=clbl if j == 0 else None)
                pct = abs(v) / total_abs * 100 if total_abs > 0 else 0
                if pct > 5:
                    mid = bot + v / 2
                    ax.text(j, mid, f"{pct:.0f}%",
                            ha='center', va='center', fontsize=7,
                            color='white', fontweight='bold')
                if v >= 0:
                    pos_bot += v
                else:
                    neg_bot += v

        # --- PS proxy bars after separator ---
        sep_x = n - 0.5 + 0.3
        ax.axvline(sep_x, color='gray', lw=0.7, ls='--', alpha=0.4)

        if region in ("LV", "RV"):
            ps_center = sep_x + 0.9
            for j, (r, lbl) in enumerate(zip(results, labels)):
                ps_val = _ps_ll_val(r, region)
                offset = (j - (n - 1) / 2) * width_ps
                ax.bar(ps_center + offset, ps_val, width_ps,
                       color=COLORS[j], alpha=0.85 if j == 0 else 0.65,
                       edgecolor='none')
            all_ticks = list(range(n)) + [ps_center]
            all_ticklabels = list(labels) + ["PS"]
        else:
            # Septum: PLV and Trans variants
            ps_center1 = sep_x + 0.9
            ps_center2 = ps_center1 + n * width_ps + 0.35
            for j, (r, lbl) in enumerate(zip(results, labels)):
                ps_plv = _ps_ll_val(r, region, "default")
                ps_trans = _ps_ll_val(r, region, "trans")
                offset = (j - (n - 1) / 2) * width_ps
                ax.bar(ps_center1 + offset, ps_plv, width_ps,
                       color=COLORS[j], alpha=0.85 if j == 0 else 0.65,
                       edgecolor='none')
                ax.bar(ps_center2 + offset, ps_trans, width_ps,
                       color=COLORS[j], alpha=0.85 if j == 0 else 0.65,
                       edgecolor='none')
            all_ticks = list(range(n)) + [ps_center1, ps_center2]
            all_ticklabels = list(labels) + [r"PS $P_{LV}$", "PS Trans."]

        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_ticklabels, fontsize=7)
        ax.set_title(region, fontweight='bold')
        ax.axhline(0, color='k', lw=0.5)
        if region == "LV":
            ax.set_ylabel("Work (mJ)")

    # Component legend on RV panel (bottom right — least overlap)
    axes[1].legend(fontsize=6.5, loc='lower right')

    fig.tight_layout(w_pad=0.8)
    _save(fig, outdir, "work_decomposition_stacked.png")


# ─── Figure 5: Pressure vs Transmural Stress ─────────────────────────────────

def plot_pressure_vs_radial_stress(results, labels, outdir):
    """
    2x3 panel: why cavity pressure tracks wall stress.
      Row 0: P(t) and -sigma_nn(t) overlaid
      Row 1: scatter of -sigma_nn vs P with regression
    """
    regions = [
        ("LV",     "mean_sigma_nn_LV",     "p_LV",  None),
        ("RV",     "mean_sigma_nn_RV",     "p_RV",  None),
        ("Septum", "mean_sigma_nn_Septum", "p_LV",  "p_RV"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5))

    for col, (region, sigma_key, p_key, _) in enumerate(regions):
        ax_time = axes[0, col]
        ax_scatter = axes[1, col]

        for i, (r, lbl, c) in enumerate(zip(results, labels, COLORS)):
            t = get_array(r, 'time')
            sigma_nn = get_array(r, sigma_key)
            pres_Pa = get_array(r, p_key) * 133.322

            if len(t) == 0 or len(sigma_nn) == 0:
                continue

            neg_sigma = -sigma_nn

            # Row 0: time traces
            ax_time.plot(t * 1000, pres_Pa / 1e3, color=c, lw=1.5,
                         label=f"{lbl} $P$")
            ax_time.plot(t * 1000, neg_sigma / 1e3, color=c, lw=1.2, ls='--',
                         label=r"{} $-\bar{{\sigma}}_{{nn}}$".format(lbl))

            # Row 1: scatter
            ax_scatter.scatter(pres_Pa / 1e3, neg_sigma / 1e3, color=c, s=8,
                               alpha=0.4, label=lbl, rasterized=True)

            # Regression through origin
            if len(pres_Pa) > 2:
                slope = np.sum(neg_sigma * pres_Pa) / np.sum(pres_Pa**2)
                p_fit = np.linspace(0, pres_Pa.max(), 50)
                ax_scatter.plot(p_fit / 1e3, slope * p_fit / 1e3, color=c, lw=1.0,
                                ls=':', alpha=0.8)
                # Annotate slope
                ax_scatter.text(0.95, 0.15 - i * 0.12,
                                f"slope = {slope:.2f}",
                                transform=ax_scatter.transAxes,
                                fontsize=7, color=c, ha='right')

        # Reference: y = 0.5x
        xlim = ax_scatter.get_xlim()
        x_ref = np.linspace(0, max(xlim[1], 1) * 1.1, 50)
        ax_scatter.plot(x_ref, 0.5 * x_ref, 'k--', lw=0.8, alpha=0.3,
                        label=r"$\sigma_{nn} = P/2$")

        ax_time.set_title(region, fontweight='bold')
        if col == 0:
            ax_time.set_ylabel("Stress / Pressure (kPa)")
            ax_time.legend(fontsize=5, loc='upper right', ncol=1, handlelength=1.0)
            ax_scatter.set_ylabel(r"$-\bar{\sigma}_{nn}$ (kPa)")
        ax_scatter.set_xlabel("$P$ (kPa)")
        if col == 0:
            ax_scatter.legend(fontsize=5, loc='upper left', handlelength=1.0)

    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    _save(fig, outdir, "pressure_vs_transmural_stress.png")


# ─── Figure 6: Work Redistribution ───────────────────────────────────────────

def plot_work_redistribution(results, labels, outdir):
    """
    Bar chart: absolute regional work (LV, RV, Septum) with percentage annotations.
    2 panels: total work, fiber work.  Optional 3rd: work density if volumes available.
    """
    regions = ["LV", "RV", "Septum"]

    W = 90
    print("\n" + "=" * W)
    print(f"{'BIVENTRICULAR WORK REDISTRIBUTION':^{W}}")
    print("=" * W)

    shares = {}
    for wtype, wkey in [("Total", "work_true"), ("Fiber", "work_ff")]:
        print(f"\n  -- {wtype} Work --")
        print(f"  {'Case':<10} {'LV (mJ)':>10} {'RV (mJ)':>10} {'Sep (mJ)':>10} "
              f"{'Total (mJ)':>11} {'LV%':>6} {'RV%':>6} {'Sep%':>6}")
        print("  " + "-" * 73)
        for j, (r, lbl) in enumerate(zip(results, labels)):
            vals = [total_work(r, f"{wkey}_{reg}") for reg in regions]
            tot = sum(vals)
            pcts = [v / tot * 100 if abs(tot) > 1e-15 else 0.0 for v in vals]
            shares[(wtype, j)] = pcts
            print(f"  {lbl:<10} {vals[0]*1e3:>10.2f} {vals[1]*1e3:>10.2f} {vals[2]*1e3:>10.2f} "
                  f"{tot*1e3:>11.2f} {pcts[0]:>5.1f}% {pcts[1]:>5.1f}% {pcts[2]:>5.1f}%")
    print("=" * W)

    # Check for volume data
    vol_m3_to_mL = 1e6
    has_volumes = all(f"region_volume_{reg}" in results[0] for reg in regions)
    n_panels = 3 if has_volumes else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(3.3 * n_panels, 3.0))
    if n_panels == 1:
        axes = [axes]

    panel_configs = [("Total work", "work_true"), ("Fiber work", "work_ff")]

    for ax, (panel_title, wkey) in zip(axes[:2], panel_configs):
        x = np.arange(len(regions))
        width = 0.35

        for j, (r, lbl) in enumerate(zip(results, labels)):
            reg_vals = [total_work(r, f"{wkey}_{reg}") for reg in regions]
            tot = sum(reg_vals)
            vals_mJ = [v * 1e3 for v in reg_vals]
            pcts = [v / tot * 100 if abs(tot) > 1e-15 else 0.0 for v in reg_vals]

            offset = (j - 0.5) * width
            bars = ax.bar(x + offset, vals_mJ, width,
                          label=lbl, color=COLORS[j],
                          alpha=0.85 if j == 0 else 0.65,
                          edgecolor='none')
            for bar, pct in zip(bars, pcts):
                h = bar.get_height()
                va = 'bottom' if h >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f"{pct:.0f}%", ha='center', va=va, fontsize=7, color='0.3')

        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.set_title(panel_title, fontweight='bold')
        ax.axhline(0, color='k', lw=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Work (mJ)")

    # Panel 3: work density if available
    if has_volumes:
        ax = axes[2]
        x = np.arange(len(regions))
        width = 0.35
        for j, (r, lbl) in enumerate(zip(results, labels)):
            densities = []
            for reg in regions:
                w_mJ = total_work(r, f"work_true_{reg}") * 1e3
                v_mL = float(r[f"region_volume_{reg}"]) * vol_m3_to_mL
                densities.append(w_mJ / v_mL if v_mL > 0 else 0.0)
            offset = (j - 0.5) * width
            ax.bar(x + offset, densities, width,
                   label=lbl, color=COLORS[j],
                   alpha=0.85 if j == 0 else 0.65,
                   edgecolor='none')
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.set_ylabel("Work density (mJ/mL)")
        ax.set_title("Work density", fontweight='bold')
        ax.axhline(0, color='k', lw=0.5)
        ax.legend(fontsize=6, loc='lower right')

    fig.tight_layout(w_pad=0.8)
    _save(fig, outdir, "work_redistribution.png")


def plot_work_density(results, labels, outdir):
    """
    Standalone work density figure: total and fiber work density per region.
    Shows how work per unit volume shifts across LV, RV, Septum.
    """
    regions = ["LV", "RV", "Septum"]
    vol_m3_to_mL = 1e6

    has_volumes = all(f"region_volume_{reg}" in results[0] for reg in regions)
    if not has_volumes:
        print("  No region_volume_* keys — skipping work density plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.2))

    for ax, (panel_title, wkey) in zip(axes, [("Total work density", "work_true"),
                                               ("Fiber work density", "work_ff")]):
        x = np.arange(len(regions))
        width = 0.35

        for j, (r, lbl) in enumerate(zip(results, labels)):
            densities = []
            for reg in regions:
                w_mJ = total_work(r, f"{wkey}_{reg}") * 1e3
                v_mL = float(r[f"region_volume_{reg}"]) * vol_m3_to_mL
                densities.append(w_mJ / v_mL if v_mL > 0 else 0.0)
            offset = (j - 0.5) * width
            ax.bar(x + offset, densities, width,
                   label=lbl, color=COLORS[j],
                   alpha=0.85 if j == 0 else 0.65,
                   edgecolor='none')

        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.set_title(panel_title, fontweight='bold')
        ax.axhline(0, color='k', lw=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Work density (mJ/mL)")

    axes[1].legend(fontsize=6.5, loc='lower right')
    fig.tight_layout(w_pad=1.0)
    _save(fig, outdir, "work_density.png")


def plot_work_redistribution_stacked(results, labels, outdir):
    """
    Stacked bar version of work redistribution.
    One bar per case for Total work and Fiber work, stacked by region (LV/RV/Septum).
    Work density is not stackable and is omitted.
    """
    regions = ["LV", "RV", "Septum"]
    panel_configs = [("Total work", "work_true"), ("Fiber work", "work_ff")]

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 3.5))

    for ax, (panel_title, wkey) in zip(axes, panel_configs):
        x = np.arange(len(results))
        width = 0.5

        for j, r in enumerate(results):
            reg_vals = [total_work(r, f"{wkey}_{reg}") * 1e3 for reg in regions]
            total = sum(reg_vals)
            bot = 0.0
            for k, (reg, v) in enumerate(zip(regions, reg_vals)):
                ax.bar(j, v, width, bottom=bot, color=REGION_COLORS[reg],
                       alpha=0.85, edgecolor='white', lw=0.5,
                       label=reg if j == 0 else None)
                pct = abs(v / total) * 100 if abs(total) > 1e-15 else 0.0
                if pct > 5:
                    ax.text(j, bot + v / 2, f"{pct:.0f}%",
                            ha='center', va='center', fontsize=8,
                            color='white', fontweight='bold')
                bot += v

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(panel_title, fontweight='bold')
        ax.axhline(0, color='k', lw=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Work (mJ)")
            ax.legend(fontsize=7, loc='upper left')

    fig.tight_layout(w_pad=0.8)
    _save(fig, outdir, "work_redistribution_stacked.png")


# ─── Figure 7: Sensitivity / Proxy Tracking ──────────────────────────────────

def _plot_sensitivity_one(results, labels, outdir, strain_comp, strain_label, suffix):
    """
    2x3 grid: [Total truth, Fiber truth] x [LV, RV, Septum].
    Healthy normalized to (1,1). Ideal tracking: points on y=x.
    Uses independent x/y ranges so axes aren't squashed.
    """
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.5))

    def get_proxy_total(r, region):
        if region == "Septum":
            key = f"work_ps_{strain_comp}_Septum_PLV"
        else:
            key = f"work_ps_{strain_comp}_{region}"
        if key not in r:
            raise KeyError(f"Missing '{key}'")
        return total_work(r, key)

    row_labels = ["Total work\n(normalized)", "Fiber work\n(normalized)"]

    fw_configs = [
        (0, 0, "LV", "work_true_LV"),
        (0, 1, "RV", "work_true_RV"),
        (1, 0, "LV", "work_ff_LV"),
        (1, 1, "RV", "work_ff_RV"),
    ]

    for row, col, region, truth_key in fw_configs:
        ax = axes[row, col]
        truth_vals = [total_work(r, truth_key) for r in results]
        proxy_vals = [get_proxy_total(r, region) for r in results]

        bt, bp = truth_vals[0], proxy_vals[0]
        if abs(bt) < 1e-12 or abs(bp) < 1e-12:
            continue

        tn = [v / bt for v in truth_vals]
        pn = [v / bp for v in proxy_vals]

        # Independent x/y with generous padding
        x_lo = min(min(pn), 0.5) - 0.1
        x_hi = max(max(pn), 1.1) + 0.1
        y_lo = min(min(tn), 0.5) - 0.1
        y_hi = max(max(tn), 1.1) + 0.1
        diag_lo, diag_hi = min(x_lo, y_lo), max(x_hi, y_hi)

        ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], 'k--', alpha=0.15, lw=1)
        ax.plot(pn, tn, 'o-', color=COLORS[0], lw=1.5, ms=8, zorder=5)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (pn[i], tn[i]), xytext=(8, 8),
                        textcoords='offset points', fontsize=7, color='0.4')

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.08)
        if row == 0:
            ax.set_title(f"{region} free wall", fontweight='bold')
        if row == 1:
            ax.set_xlabel("PS proxy (norm.)")
        if col == 0:
            ax.set_ylabel(row_labels[row])

    # Septum: multiple pressure variants
    septum_variants = [
        ("p_LV",  r"$P_{LV}$",    COLORS[0],  "o"),
        ("trans",  "Trans.",        REGION_COLORS["Septum"], "s"),
        ("p_RV",  r"$P_{RV}$",    COLORS[1],  "^"),
    ]

    def get_septum_proxy(r, p_variant):
        variant_map = {"p_LV": "PLV", "trans": "Trans", "p_RV": "PRV"}
        key = f"work_ps_{strain_comp}_Septum_{variant_map[p_variant]}"
        if key not in r:
            raise KeyError(f"Missing '{key}'")
        return total_work(r, key)

    for row, truth_key in [
        (0, "work_true_Septum"),
        (1, "work_ff_Septum"),
    ]:
        ax = axes[row, 2]
        truth_vals = [total_work(r, truth_key) for r in results]
        bt = truth_vals[0]
        if abs(bt) < 1e-12:
            continue
        tn = [v / bt for v in truth_vals]

        all_x = []
        for p_variant, plbl, pcol, pmk in septum_variants:
            pvals = [get_septum_proxy(r, p_variant) for r in results]
            bp = pvals[0]
            if abs(bp) < 1e-12:
                continue
            pn = [v / bp for v in pvals]
            all_x.extend(pn)
            ax.plot(pn, tn, f'{pmk}-', color=pcol, lw=1.5, ms=7,
                    label=plbl, zorder=5)

        # Independent x/y with generous padding
        x_lo = min(min(all_x), 0.5) - 0.1
        x_hi = max(max(all_x), 1.1) + 0.1
        y_lo = min(min(tn), 0.5) - 0.1
        y_hi = max(max(tn), 1.1) + 0.1
        diag_lo, diag_hi = min(x_lo, y_lo), max(x_hi, y_hi)
        ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], 'k--', alpha=0.15, lw=1)

        ax.annotate(labels[0], (1.0, 1.0), xytext=(8, 8),
                    textcoords='offset points', fontsize=7, color='0.4')
        if len(labels) > 1:
            ax.annotate(labels[1], (all_x[1] if len(all_x) > 1 else 1.0, tn[1]),
                        xytext=(8, -14), textcoords='offset points',
                        fontsize=7, color='0.4')

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.08)
        if row == 0:
            ax.set_title("Septum", fontweight='bold')
            ax.legend(fontsize=7, loc='upper left', handlelength=1.5,
                      framealpha=0.9)
        if row == 1:
            ax.set_xlabel("PS proxy (norm.)")

    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    _save(fig, outdir, f"sensitivity_{suffix}.png")


def plot_sensitivity(results, labels, outdir):
    _plot_sensitivity_one(results, labels, outdir,
                          "ll", "Longitudinal Strain E_ll", "longitudinal")


# ─── Figure 8: Geometry Context ───────────────────────────────────────────────

def plot_geometry_context(results, labels, outdir):
    """Region volumes, work density, and volume fractions."""
    regions = ["LV", "RV", "Septum"]

    vol_m3_to_mL = 1e6
    volumes = {}
    has_volumes = True

    W = 90
    print("\n" + "=" * W)
    print(f"{'GEOMETRY CONTEXT':^{W}}")
    print("=" * W)

    for j, (r, lbl) in enumerate(zip(results, labels)):
        for region in regions:
            key = f"region_volume_{region}"
            if key in r:
                volumes[(j, region)] = float(r[key]) * vol_m3_to_mL
            else:
                has_volumes = False
                volumes[(j, region)] = 0.0

    if not has_volumes:
        print("  No region_volume_* keys — skipping geometry context plot.")
        print("=" * W)
        return

    # Print table
    print(f"\n  {'Case':<15} {'LV (mL)':>10} {'RV (mL)':>10} {'Sep (mL)':>10}")
    print("  " + "-" * 50)
    for j, lbl in enumerate(labels):
        vols = [volumes[(j, reg)] for reg in regions]
        print(f"  {lbl:<15} {vols[0]:>10.2f} {vols[1]:>10.2f} {vols[2]:>10.2f}")
    print("=" * W)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

    # Panel 1: volumes
    ax = axes[0]
    x = np.arange(len(regions))
    width = 0.35
    for j, lbl in enumerate(labels):
        vals = [volumes[(j, reg)] for reg in regions]
        offset = (j - 0.5) * width
        ax.bar(x + offset, vals, width, label=lbl, color=COLORS[j],
               alpha=0.85 if j == 0 else 0.65, edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Wall volume (mL)")
    ax.set_title("Region volumes", fontweight='bold')
    ax.legend(fontsize=6, loc='upper left')

    # Panel 2: work density
    ax = axes[1]
    densities = {}
    for j, r in enumerate(results):
        for region in regions:
            w = total_work(r, f"work_true_{region}") * 1e3
            v = volumes[(j, region)]
            densities[(j, region)] = w / v if v > 0 else 0.0
    for j, lbl in enumerate(labels):
        vals = [densities[(j, reg)] for reg in regions]
        offset = (j - 0.5) * width
        ax.bar(x + offset, vals, width, label=lbl, color=COLORS[j],
               alpha=0.85 if j == 0 else 0.65, edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Work density (mJ/mL)")
    ax.set_title("Work density", fontweight='bold')
    ax.axhline(0, color='k', lw=0.5)

    # Panel 3: volume fractions
    ax = axes[2]
    for j, lbl in enumerate(labels):
        vols = [volumes[(j, reg)] for reg in regions]
        tot = sum(vols)
        fracs = [v / tot * 100 if tot > 0 else 0 for v in vols]
        bottom = 0
        for k, (reg, frac) in enumerate(zip(regions, fracs)):
            ax.bar(j, frac, bottom=bottom, color=REGION_COLORS[reg],
                   label=reg if j == 0 else None, edgecolor='none')
            if frac > 8:
                ax.text(j, bottom + frac / 2, f"{frac:.0f}%",
                        ha='center', va='center', fontsize=8, color='white',
                        fontweight='bold')
            bottom += frac
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Volume fraction (%)")
    ax.set_title("Composition", fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 105)

    fig.tight_layout(w_pad=0.8)
    _save(fig, outdir, "geometry_context.png")


def plot_geometry_context_stacked(results, labels, outdir):
    """
    Stacked bar version of region volumes: one bar per case, stacked by region.
    Work density is not stackable and is omitted.
    """
    regions = ["LV", "RV", "Septum"]
    vol_m3_to_mL = 1e6

    volumes = {}
    for j, r in enumerate(results):
        for region in regions:
            key = f"region_volume_{region}"
            if key not in r:
                return  # silently skip if no volume data
            volumes[(j, region)] = float(r[key]) * vol_m3_to_mL

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))

    x = np.arange(len(results))
    width = 0.5

    for j in range(len(results)):
        vols = [volumes[(j, reg)] for reg in regions]
        total = sum(vols)
        bot = 0.0
        for k, (reg, v) in enumerate(zip(regions, vols)):
            ax.bar(j, v, width, bottom=bot, color=REGION_COLORS[reg],
                   alpha=0.85, edgecolor='white', lw=0.5,
                   label=reg if j == 0 else None)
            pct = v / total * 100 if total > 0 else 0.0
            if pct > 5:
                ax.text(j, bot + v / 2, f"{pct:.0f}%",
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
            bot += v

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Wall volume (mL)")
    ax.set_title("Region volumes", fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')

    fig.tight_layout()
    _save(fig, outdir, "geometry_context_stacked.png")


# ─── Extra Figures (--all mode) ──────────────────────────────────────────────

def plot_stress_loops_components(results, labels, outdir):
    """Multi-component stress-strain loops: ff, ss, nn for each region."""
    components = [
        ("ff", r"$S_{ff}$ vs $E_{ff}$"),
        ("ss", r"$S_{ss}$ vs $E_{ss}$"),
        ("nn", r"$S_{nn}$ vs $E_{nn}$"),
    ]
    regions = ["LV", "RV", "Septum"]

    fig, axes = plt.subplots(len(components), 3, figsize=(7.0, 7.0))

    for row, (comp, comp_label) in enumerate(components):
        for col, region in enumerate(regions):
            ax = axes[row, col]
            strain_key = f"mean_E_{comp}_{region}"
            stress_key = f"mean_S_{comp}_{region}"

            for r, lbl, c in zip(results, labels, COLORS):
                E = get_array(r, strain_key) * 100
                S = get_array(r, stress_key) / 1e3
                if len(E) == 0:
                    continue
                ax.plot(E, S, color=c, lw=1.5, label=lbl)
                ax.plot(E[0], S[0], 'o', color=c, ms=3)

            ax.axhline(0, color='0.5', lw=0.4)
            ax.axvline(0, color='0.5', lw=0.4)
            if row == 0:
                ax.set_title(region, fontweight='bold')
            if col == 0:
                ax.set_ylabel(comp_label)
            if row == len(components) - 1:
                ax.set_xlabel("Strain (%)")
            if row == 0 and col == 0:
                ax.legend(fontsize=6, loc='best', handlelength=1.0)

    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    _save(fig, outdir, "stress_loops_components.png")


def plot_fiber_efficiency(results, labels, outdir):
    """Fiber work fraction: W_ff / W_total for each region."""
    regions = ["LV", "RV", "Septum"]

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))

    x = np.arange(len(regions))
    width = 0.35

    for j, (r, lbl) in enumerate(zip(results, labels)):
        ratios = []
        for reg in regions:
            w_ff = total_work(r, f"work_ff_{reg}")
            w_tot = total_work(r, f"work_true_{reg}")
            ratios.append(w_ff / w_tot * 100 if abs(w_tot) > 1e-15 else 0.0)
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, ratios, width, label=lbl, color=COLORS[j],
                      alpha=0.85 if j == 0 else 0.65, edgecolor='none')
        for bar, v in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}%", ha='center', va='bottom', fontsize=7, color='0.3')

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel(r"$W_{ff} / W_{\mathrm{total}}$ (%)")
    ax.set_title("Fiber work fraction", fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, 110)

    fig.tight_layout()
    _save(fig, outdir, "fiber_efficiency.png")


def plot_proxy_directional_match(results, labels, outdir):
    """
    Scatter: proxy vs ground-truth instantaneous power for LV, RV, Septum.
    Shows how well the proxy tracks the FEM work rate over the cardiac cycle.
    """
    regions_cfg = [
        ("LV",     "work_ff_LV",     "work_ps_ff_LV",     None),
        ("RV",     "work_ff_RV",     "work_ps_ff_RV",     None),
        ("Septum", "work_ff_Septum", "work_ps_ff_Septum_PLV", "work_ps_ff_Septum_Trans"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.0))

    for ax, (region, truth_key, proxy_key, proxy_trans_key) in zip(axes, regions_cfg):
        all_r2 = []  # collect for annotation box

        for i, (r, lbl, c) in enumerate(zip(results, labels, COLORS)):
            truth = get_array(r, truth_key) * 1e3
            proxy = get_array(r, proxy_key) * 1e3
            if len(truth) == 0 or len(proxy) == 0:
                continue

            ax.scatter(proxy, truth, color=c, s=8, alpha=0.4,
                       label=lbl, rasterized=True, zorder=3)

            # R^2
            if len(truth) > 2 and np.std(truth) > 0:
                corr = np.corrcoef(proxy, truth)[0, 1]
                all_r2.append((lbl, corr**2, c, None))

            # Transmural variant for septum
            if proxy_trans_key and proxy_trans_key in r:
                proxy_t = get_array(r, proxy_trans_key) * 1e3
                if len(proxy_t) > 0:
                    # Align lengths (trans proxy may be 1 shorter due to diff)
                    n_common = min(len(proxy_t), len(truth))
                    pt, tt = proxy_t[:n_common], truth[:n_common]
                    ax.scatter(pt, tt, color=c, s=8, alpha=0.2,
                               marker='s', rasterized=True, zorder=2)
                    if np.std(tt) > 0:
                        corr_t = np.corrcoef(pt, tt)[0, 1]
                        all_r2.append((f"{lbl} trans.", corr_t**2, c, 'italic'))

        # Clip axes to 2nd-98th percentile to remove outliers
        all_proxy = np.concatenate([get_array(r, proxy_key) * 1e3 for r in results if proxy_key in r])
        all_truth = np.concatenate([get_array(r, truth_key) * 1e3 for r in results if truth_key in r])
        if len(all_proxy) > 10:
            p2x, p98x = np.percentile(all_proxy, [1, 99])
            p2y, p98y = np.percentile(all_truth, [1, 99])
            pad_x = (p98x - p2x) * 0.15
            pad_y = (p98y - p2y) * 0.15
            ax.set_xlim(p2x - pad_x, p98x + pad_x)
            ax.set_ylim(p2y - pad_y, p98y + pad_y)

        # y=x reference
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.15, zorder=1)

        # R² annotation box (top-left, stacked)
        r2_text = "\n".join(
            f"{'$R^2$' if style is None else '$R^2_{trans}$'} = {r2:.2f}"
            for lbl, r2, c, style in all_r2
        )
        if all_r2:
            # Color-coded lines
            y_pos = 0.97
            for lbl, r2, c, style in all_r2:
                tag = "$R^2$" if style is None else "$R^2_{\\mathrm{trans}}$"
                ax.text(0.04, y_pos, f"{tag} = {r2:.2f}  ({lbl})",
                        transform=ax.transAxes, fontsize=6, color=c,
                        ha='left', va='top',
                        fontstyle='italic' if style else 'normal')
                y_pos -= 0.09

        ax.set_title(region, fontweight='bold')
        ax.set_xlabel("Proxy (mJ)")
        ax.grid(True, alpha=0.08)
        if ax == axes[0]:
            ax.set_ylabel("FEM fiber work (mJ)")

    fig.tight_layout(w_pad=1.0)
    _save(fig, outdir, "proxy_directional_match.png")


def plot_dyssynchrony(results, labels, outdir):
    """
    Time-resolved regional work rate: LV, RV, Septum on same axes.
    Shows timing differences in energy generation.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(3.5 * len(results), 3.0),
                              squeeze=False)

    for j, (r, lbl) in enumerate(zip(results, labels)):
        ax = axes[0, j]
        t = get_array(r, 'time') * 1000  # ms

        for region, reg_c in REGION_COLORS.items():
            w = get_array(r, f"work_true_{region}") * 1e3  # mJ
            if len(w) == 0 or len(t) == 0:
                continue
            # Work rate: dW/dt (finite differences)
            if len(w) == len(t) - 1:
                t_mid = 0.5 * (t[:-1] + t[1:])
                dt = np.diff(t)
                rate = w / dt  # mJ/ms = W
            elif len(w) == len(t):
                t_mid = t
                rate = w
            else:
                continue
            ax.plot(t_mid, rate, color=reg_c, lw=1.2, label=region)

        ax.set_title(lbl, fontweight='bold')
        ax.set_xlabel("Time (ms)")
        if j == 0:
            ax.set_ylabel("Work rate (mJ/ms)")
            ax.legend(fontsize=7)
        ax.axhline(0, color='k', lw=0.5)

    fig.tight_layout(w_pad=0.8)
    _save(fig, outdir, "dyssynchrony.png")


# ─── Console Report ──────────────────────────────────────────────────────────

def print_report(results, labels):
    W = 100
    print("\n" + "=" * W)
    print(f"{'PROXY SENSITIVITY REPORT':^{W}}")
    print("=" * W)
    print(f"  Normalized to {labels[0]} = 1.0\n")

    regions = ["LV", "RV", "Septum"]

    for region in regions:
        print(f"  -- {region} {'Freewall' if region != 'Septum' else ''} --")

        truth_keys = [
            (f"work_true_{region}",  "Total Work"),
            (f"work_ff_{region}",    "Fiber Work"),
        ]

        if region == "Septum":
            proxy_keys = [
                (f"work_ps_ff_Septum_PLV",   "PS (P_LV)  "),
                (f"work_ps_ff_Septum_Trans", "PS (Trans)  "),
                (f"work_ps_ff_Septum_PRV",   "PS (P_RV)  "),
            ]
        else:
            proxy_keys = [
                (f"work_ps_ff_{region}", "PS Proxy    "),
            ]

        all_keys = truth_keys + proxy_keys

        for r, lbl in zip(results, labels):
            rv_peak = float(np.max(get_array(r, 'p_RV'))) if 'p_RV' in r else 0.0
            vals = "  ".join(f"{kl}: {total_work(r, kk):>10.3e}" for kk, kl in all_keys)
            print(f"    {lbl:<10} (RV peak {rv_peak:.1f} mmHg)  {vals}")

        if len(results) >= 2:
            r_base, r_comp = results[0], results[1]
            for kk, kl in all_keys:
                b = total_work(r_base, kk)
                p = total_work(r_comp,  kk)
                ratio = p / b if abs(b) > 1e-12 else float('nan')
                print(f"      {kl:<14}: {labels[1]}/{labels[0]} = {ratio:.3f}   ({b:.3e} -> {p:.3e})")
        print()
    print("=" * W)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare two cardiac simulation cases.")
    parser.add_argument("case_a_dir", help="Path to first (baseline) result directory")
    parser.add_argument("case_b_dir", help="Path to second (comparison) result directory")
    parser.add_argument("--labels", nargs=2, default=["Healthy", "PAH"],
                       help="Labels for the two cases (default: Healthy PAH)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--primary", action="store_true",
                       help="Run only the primary figure set")
    group.add_argument("--all", dest="run_all", action="store_true", default=True,
                       help="Run all figures (default)")
    parser.add_argument("--outdir", type=str, default=None,
                       help="Output directory (default: results/sims/compare_cases)")
    args = parser.parse_args()

    if args.primary:
        args.run_all = False

    setup_style()

    folders = [args.case_a_dir, args.case_b_dir]
    labels  = args.labels
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path(__file__).resolve().parent / "results" / "sims" / "compare_cases"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  compare_cases.py")
    print("=" * 60)

    results = []
    for folder, lbl in zip(folders, labels):
        print(f"\n[{lbl}]  {folder}")
        m = load_metrics(folder)
        if m is None:
            print(f"  ERROR: could not load metrics from {folder}")
            sys.exit(1)
        results.append(m)

    print_report(results, labels)

    primary_figures = [
        plot_geometry_context,
        plot_geometry_context_stacked,
        plot_simplification_cascade,
        plot_pressure_vs_radial_stress,
        plot_work_redistribution,
        plot_work_redistribution_stacked,
        plot_work_decomposition,
        plot_work_decomposition_stacked,
        plot_work_density,
        plot_stress_loops,
        plot_sensitivity,
    ]

    extra_figures = [
        plot_proxy_directional_match,
        plot_fiber_efficiency,
        plot_stress_loops_components,
        plot_dyssynchrony,
    ]

    figure_set = primary_figures if args.primary else primary_figures + extra_figures

    mode = "primary" if args.primary else "all"
    print(f"\nGenerating figures ({mode}: {len(figure_set)} plots)...")
    for fig_fn in figure_set:
        fig_fn(results, labels, outdir)

    print(f"\nDone. Output: {outdir}")


if __name__ == "__main__":
    main()
