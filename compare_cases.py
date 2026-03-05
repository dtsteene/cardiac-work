#!/usr/bin/env python3
"""
compare_cases.py

Sensitivity analysis for LV Freewall, RV Freewall, and Septum mechanical work proxies.

Compares pressure-strain (PS) proxies against FEM ground truths:
  1. Total Work  (work_true  = full tensor S:dE)
  2. Fiber Work  (work_fiber = S_ff * dE_ff)

For the Septum, multiple PS proxy variants are compared:
  - PLV:   P_LV × dε
  - PRV:   P_RV × dε
  - Trans: (P_LV − P_RV) × dε   (transmural pressure)
  - Mean:  ½(P_LV + P_RV) × dε

Produces:
  sensitivity_comparison.png   — normalized proxy vs truth
  stress_loops.png             — fiber stress-strain loops healthy vs PAH
  work_decomposition.png       — fiber/sheet/normal/shear breakdown
  pressure_strain.png          — pressure-strain loops

Usage:
  python3 compare_cases.py <HEALTHY_RESULT_DIR> <PAH_RESULT_DIR>
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_metrics(folder):
    """
    Load metrics from folder. Prefers analysis_last_beat/ subdirectory if present,
    then falls back to the folder itself.
    """
    path = Path(folder)

    last_beat = path / "analysis_last_beat"
    if last_beat.exists():
        files = sorted(last_beat.glob("metrics_downsample_*.npy"),
                       key=lambda p: len(str(p)))
        if files:
            print(f"  Loading (last beat): {files[0]}")
            return np.load(files[0], allow_pickle=True).item()

    files = sorted(path.glob("metrics_downsample_*.npy"), key=lambda p: len(str(p)))
    if not files:
        print(f"  No metrics found in {folder}")
        return None
    print(f"  Loading (direct): {files[0]}")
    return np.load(files[0], allow_pickle=True).item()


def total_work(m, key):
    """Sum incremental work array → total cycle work (Joules)."""
    if key not in m:
        return 0.0
    return float(np.sum(m[key]))


def get_array(m, key):
    return np.array(m[key]) if key in m else np.array([])


# ─── Figure 1: Sensitivity Comparison ────────────────────────────────────────

def plot_sensitivity(results, labels, outdir):
    """
    2×3 grid: [Total truth, Fiber truth] × [LV, RV, Septum].
    Healthy = (1, 1). Ideal tracking → points on y = x.
    For Septum, multiple proxy methods are shown.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Pressure-Strain Proxy Sensitivity: LV & RV Freewalls + Septum\n"
        "(Healthy = 1.0  |  Ideal tracking: points lie on y = x)",
        fontsize=13, fontweight='bold'
    )

    # Freewall configs: single proxy per region
    fw_configs = [
        # (row, col, region, truth_key,     truth_label,              proxy_key,           color)
        (0, 0, "LV", "work_true_LV",  "Total Work (S:dE)",      "work_ps_index_LV",  "tab:blue"),
        (0, 1, "RV", "work_true_RV",  "Total Work (S:dE)",      "work_ps_index_RV",  "tab:red"),
        (1, 0, "LV", "work_fiber_LV", "Fiber Work (S_ff·dE_ff)","work_ps_index_LV",  "tab:blue"),
        (1, 1, "RV", "work_fiber_RV", "Fiber Work (S_ff·dE_ff)","work_ps_index_RV",  "tab:red"),
    ]

    for row, col, region, truth_key, truth_label, proxy_key, color in fw_configs:
        ax = axes[row, col]
        truth_vals = [total_work(r, truth_key) for r in results]
        proxy_vals = [total_work(r, proxy_key) for r in results]

        bt, bp = truth_vals[0], proxy_vals[0]
        if abs(bt) < 1e-12 or abs(bp) < 1e-12:
            ax.set_title(f"{region} Freewall — no data")
            continue

        tn = [v / bt for v in truth_vals]
        pn = [v / bp for v in proxy_vals]

        all_v = tn + pn
        lo, hi = min(all_v) - 0.15, max(all_v) + 0.15
        lo, hi = min(lo, 0.5), max(hi, 1.1)

        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, lw=1.5, label="Ideal (y = x)")
        ax.plot(pn, tn, 'o-', color=color, lw=2, ms=10, label="PS Proxy")
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (pn[i], tn[i]), xytext=(8, 8),
                        textcoords='offset points', fontsize=9, color=color)
        if len(results) > 1:
            dx = pn[1] - tn[1]
            ax.annotate(f"Δ(proxy−truth) = {dx:+.2f}",
                        xy=(pn[1], tn[1]), xytext=(-10, -25),
                        textcoords='offset points', fontsize=8, color='dimgray',
                        arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel("PS Proxy Work (Normalized)", fontsize=10)
        ax.set_ylabel(f"{truth_label}\n(Normalized)", fontsize=10)
        ax.set_title(f"{region} Freewall  |  {truth_label}", fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Septum configs: multiple proxy methods
    septum_proxies = [
        ("work_ps_index_Septum_PLV",   "PS (P_LV)",    "tab:blue",  "o"),
        ("work_ps_index_Septum_Trans", "PS (Trans)",    "tab:green", "s"),
        ("work_ps_index_Septum_PRV",   "PS (P_RV)",    "tab:red",   "^"),
        ("work_ps_index_Septum_Mean",  "PS (Mean)",     "tab:purple","D"),
    ]

    for row, truth_key, truth_label in [
        (0, "work_true_Septum",  "Total Work (S:dE)"),
        (1, "work_fiber_Septum", "Fiber Work (S_ff·dE_ff)"),
    ]:
        ax = axes[row, 2]
        truth_vals = [total_work(r, truth_key) for r in results]
        bt = truth_vals[0]
        if abs(bt) < 1e-12:
            ax.set_title("Septum — no data")
            continue
        tn = [v / bt for v in truth_vals]

        all_v = list(tn)
        for proxy_key, plbl, pcol, pmk in septum_proxies:
            pvals = [total_work(r, proxy_key) for r in results]
            bp = pvals[0]
            if abs(bp) < 1e-12:
                continue
            pn = [v / bp for v in pvals]
            all_v.extend(pn)
            ax.plot(pn, tn, f'{pmk}-', color=pcol, lw=2, ms=9, label=plbl)

        lo, hi = min(all_v) - 0.15, max(all_v) + 0.15
        lo, hi = min(lo, 0.5), max(hi, 1.1)
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, lw=1.5, label="Ideal (y = x)")

        # Annotate labels for the PLV proxy only (avoid clutter)
        pvals_plv = [total_work(r, "work_ps_index_Septum_PLV") for r in results]
        bp_plv = pvals_plv[0]
        if abs(bp_plv) > 1e-12:
            pn_plv = [v / bp_plv for v in pvals_plv]
            for i, lbl in enumerate(labels):
                ax.annotate(lbl, (pn_plv[i], tn[i]), xytext=(8, 8),
                            textcoords='offset points', fontsize=9, color='dimgray')

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel("PS Proxy Work (Normalized)", fontsize=10)
        ax.set_ylabel(f"{truth_label}\n(Normalized)", fontsize=10)
        ax.set_title(f"Septum  |  {truth_label}", fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(outdir) / "sensitivity_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─── Figure 2: Fiber Stress-Strain Loops ─────────────────────────────────────

def plot_stress_loops(results, labels, outdir):
    """
    2×3 grid:
      Row 0: fiber stress-strain loops (S_ff vs E_ff) for LV, RV, Septum
      Row 1: fiber strain + pressure vs time         for LV, RV, Septum
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Fiber Stress-Strain Loops: LV & RV Freewalls + Septum  (Healthy vs PAH)",
                 fontsize=13, fontweight='bold')

    colors = ['tab:blue', 'tab:red']

    regions = [
        ("LV",     "mean_E_ff_LV",     "mean_S_ff_LV",     "p_LV"),
        ("RV",     "mean_E_ff_RV",     "mean_S_ff_RV",     "p_RV"),
        ("Septum", "mean_E_ff_Septum", "mean_S_ff_Septum", "p_LV"),
    ]

    for col, (region, strain_key, stress_key, p_key) in enumerate(regions):
        ax_loop = axes[0, col]
        ax_time = axes[1, col]
        ax2 = ax_time.twinx()
        ax2.set_ylabel("Pressure (mmHg)", fontsize=8, color='dimgray')
        ax2.tick_params(axis='y', labelcolor='dimgray')

        for i, (r, lbl, c) in enumerate(zip(results, labels, colors)):
            t    = get_array(r, 'time')
            Eff  = get_array(r, strain_key) * 100
            Sff  = get_array(r, stress_key) / 1e3     # Pa → kPa
            pres = get_array(r, p_key)

            # Stress-strain loop
            ax_loop.plot(Eff, Sff, color=c, lw=1.8, label=lbl)
            if len(Eff) > 0:
                ax_loop.plot(Eff[0], Sff[0], 'o', color=c, ms=6)

            # Time traces
            ax_time.plot(t, Eff, color=c, lw=1.5, label=lbl)
            ax2.plot(t, pres, color=c, lw=1, ls='--', alpha=0.6)

        # Format
        ax_loop.set_xlabel(f"Fiber Strain E_ff (%)", fontsize=10)
        ax_loop.set_ylabel(f"2nd-PK Fiber Stress S_ff (kPa)", fontsize=10)
        ax_loop.set_title(f"{region} — Fiber Stress-Strain Loop\n"
                          "(area ∝ net mechanical work)", fontweight='bold')
        ax_loop.legend(fontsize=9)
        ax_loop.grid(True, alpha=0.3)
        ax_loop.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax_loop.axvline(0, color='k', lw=0.5, alpha=0.4)

        ax_time.set_xlabel("Time (s)", fontsize=10)
        ax_time.set_ylabel(f"Fiber Strain E_ff (%)", fontsize=10)
        ax_time.set_title(f"{region} — Fiber Strain vs Time\n"
                          "(dashed = pressure)", fontweight='bold')
        ax_time.legend(fontsize=9)
        ax_time.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(outdir) / "stress_loops.png"
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─── Figure 3: Work Decomposition ────────────────────────────────────────────

def plot_work_decomposition(results, labels, outdir):
    """
    1×3 grouped bar chart: fiber/sheet/normal/shear for LV, RV, Septum.
    PS proxy shown as horizontal reference lines.
    """
    components = [
        ("work_fiber",  "Fiber\n(S_ff·dE_ff)"),
        ("work_sheet",  "Sheet\n(S_ss·dE_ss)"),
        ("work_normal", "Normal\n(S_nn·dE_nn)"),
        ("work_shear",  "Shear\n(cross-terms)"),
    ]

    case_colors = ['tab:blue', 'tab:red']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Work Decomposition: Fiber / Sheet / Normal / Shear\n"
        "(negative = work done BY myocardium | PS proxy shown as dashed lines)",
        fontsize=13, fontweight='bold'
    )

    # For septum, show multiple proxy lines
    septum_proxy_keys = [
        ("work_ps_index_Septum_PLV",   "PS (P_LV)"),
        ("work_ps_index_Septum_Trans", "PS (Trans)"),
    ]

    for ax, region in zip(axes, ["LV", "RV", "Septum"]):
        x = np.arange(len(components))
        n = len(results)
        width = 0.35

        for j, (r, lbl) in enumerate(zip(results, labels)):
            vals = [total_work(r, f"{ckey}_{region}") * 1e3 for ckey, _ in components]
            offset = (j - (n - 1) / 2) * width
            ax.bar(x + offset, vals, width,
                   label=lbl, color=case_colors[j],
                   alpha=0.75 if j == 0 else 0.55,
                   edgecolor='k', linewidth=0.5)

        # PS proxy reference lines
        if region in ("LV", "RV"):
            for j, (r, lbl) in enumerate(zip(results, labels)):
                ps_val = total_work(r, f"work_ps_index_{region}") * 1e3
                ax.axhline(ps_val, ls=':', lw=1.5, color=case_colors[j],
                           label=f"PS Proxy ({lbl})")
        else:  # Septum
            for pkey, plbl in septum_proxy_keys:
                for j, (r, lbl) in enumerate(zip(results, labels)):
                    ps_val = total_work(r, pkey) * 1e3
                    ls = ':' if 'PLV' in pkey else '-.'
                    ax.axhline(ps_val, ls=ls, lw=1.5, color=case_colors[j],
                               alpha=0.7, label=f"{plbl} ({lbl})")

        ax.set_xticks(x)
        ax.set_xticklabels([clbl for _, clbl in components], fontsize=9)
        ax.set_ylabel("Work (mJ)", fontsize=11)
        ax.set_title(f"{region}", fontweight='bold', fontsize=12)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='k', lw=0.8)

    plt.tight_layout()
    out = Path(outdir) / "work_decomposition.png"
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─── Figure 4: Pressure-Strain Loops ─────────────────────────────────────────

def plot_pressure_strain(results, labels, outdir):
    """
    1×3 panel: pressure-strain loops for LV, RV, Septum.
    For Septum, both P_LV and transmural (P_LV−P_RV) loops are shown.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Pressure-Strain Loops: LV & RV Freewalls + Septum  (Healthy vs PAH)\n"
        "(loop area ∝ pressure-strain work index)",
        fontsize=13, fontweight='bold'
    )

    colors = ['tab:blue', 'tab:red']

    # Freewalls
    for ax, region, p_key, strain_key in [
        (axes[0], "LV", "p_LV", "mean_E_ff_LV"),
        (axes[1], "RV", "p_RV", "mean_E_ff_RV"),
    ]:
        for i, (r, lbl, c) in enumerate(zip(results, labels, colors)):
            strain = get_array(r, strain_key) * 100
            pressure = get_array(r, p_key)
            if len(strain) == 0 or len(pressure) == 0:
                continue
            ax.plot(strain, pressure, color=c, lw=1.8, label=lbl)
            ax.plot(strain[0], pressure[0], 'o', color=c, ms=6)

        ax.set_xlabel(f"{region} Fiber Strain E_ff (%)", fontsize=10)
        ax.set_ylabel(f"{region} Cavity Pressure (mmHg)", fontsize=10)
        ax.set_title(f"{region} Freewall — Pressure-Strain Loop", fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.axvline(0, color='k', lw=0.5, alpha=0.4)

    # Septum: show P_LV and transmural loops
    ax = axes[2]
    for i, (r, lbl, c) in enumerate(zip(results, labels, colors)):
        strain = get_array(r, 'mean_E_ff_Septum') * 100
        p_lv   = get_array(r, 'p_LV')
        p_rv   = get_array(r, 'p_RV')
        if len(strain) == 0:
            continue

        # P_LV loop (solid)
        ax.plot(strain, p_lv, color=c, lw=1.8, label=f"{lbl} (P_LV)")
        ax.plot(strain[0], p_lv[0], 'o', color=c, ms=6)

        # Transmural loop (dashed)
        p_trans = p_lv - p_rv
        ax.plot(strain, p_trans, color=c, lw=1.8, ls='--',
                label=f"{lbl} (P_LV−P_RV)")
        ax.plot(strain[0], p_trans[0], 's', color=c, ms=5)

    ax.set_xlabel("Septum Fiber Strain E_ff (%)", fontsize=10)
    ax.set_ylabel("Pressure (mmHg)", fontsize=10)
    ax.set_title("Septum — Pressure-Strain Loops\n"
                 "(solid = P_LV, dashed = transmural)", fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.axvline(0, color='k', lw=0.5, alpha=0.4)

    plt.tight_layout()
    out = Path(outdir) / "pressure_strain.png"
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─── Figure 5: Longitudinal Strain Pressure-Strain Loops ─────────────────────

def plot_pressure_strain_longitudinal(results, labels, outdir):
    """
    1×3 panel: pressure vs longitudinal (sheet-normal) strain for LV, RV, Septum.
    Longitudinal direction (n0 from LDRB) is the clinical echo/CMR GLS analogue.
    Falls back to fiber strain if longitudinal keys are unavailable.
    """
    # Check available keys and report
    sample = results[0]
    print("  [pressure_strain_longitudinal] Available strain keys:")
    for k in sorted(sample.keys()):
        if k.startswith("mean_E_"):
            print(f"    {k}")

    # Determine which strain key to use (prefer nn = longitudinal)
    if "mean_E_nn_LV" in sample:
        strain_suffix = "mean_E_nn"
        strain_label = "Longitudinal Strain E_nn"
    else:
        print("  WARNING: mean_E_nn_* not found, falling back to fiber strain E_ff")
        strain_suffix = "mean_E_ff"
        strain_label = "Fiber Strain E_ff (longitudinal proxy)"

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Pressure-Strain Loops — Clinical Perspective ({strain_label})\n"
        "(longitudinal strain is what echo/CMR measures; loop area ~ myocardial work index)",
        fontsize=13, fontweight='bold'
    )

    colors = ['tab:blue', 'tab:red']

    # Freewalls
    for ax, region, p_key in [
        (axes[0], "LV", "p_LV"),
        (axes[1], "RV", "p_RV"),
    ]:
        strain_key = f"{strain_suffix}_{region}"
        for i, (r, lbl, c) in enumerate(zip(results, labels, colors)):
            strain = get_array(r, strain_key) * 100
            pressure = get_array(r, p_key)
            if len(strain) == 0 or len(pressure) == 0:
                continue
            ax.plot(strain, pressure, color=c, lw=1.8, label=lbl)
            ax.plot(strain[0], pressure[0], 'o', color=c, ms=6)

        ax.set_xlabel(f"{region} {strain_label} (%)", fontsize=10)
        ax.set_ylabel(f"{region} Cavity Pressure (mmHg)", fontsize=10)
        ax.set_title(f"{region} Freewall — Pressure-Strain Loop", fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.axvline(0, color='k', lw=0.5, alpha=0.4)

    # Septum: P_LV and transmural loops
    ax = axes[2]
    strain_key = f"{strain_suffix}_Septum"
    for i, (r, lbl, c) in enumerate(zip(results, labels, colors)):
        strain = get_array(r, strain_key) * 100
        p_lv = get_array(r, 'p_LV')
        p_rv = get_array(r, 'p_RV')
        if len(strain) == 0:
            continue

        ax.plot(strain, p_lv, color=c, lw=1.8, label=f"{lbl} (P_LV)")
        ax.plot(strain[0], p_lv[0], 'o', color=c, ms=6)

        p_trans = p_lv - p_rv
        ax.plot(strain, p_trans, color=c, lw=1.8, ls='--',
                label=f"{lbl} (P_LV-P_RV)")
        ax.plot(strain[0], p_trans[0], 's', color=c, ms=5)

    ax.set_xlabel(f"Septum {strain_label} (%)", fontsize=10)
    ax.set_ylabel("Pressure (mmHg)", fontsize=10)
    ax.set_title("Septum — Pressure-Strain Loops\n"
                 "(solid = P_LV, dashed = transmural)", fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.axvline(0, color='k', lw=0.5, alpha=0.4)

    plt.tight_layout()
    out = Path(outdir) / "pressure_strain_longitudinal.png"
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─── Figure 6: Multi-Component Stress-Strain Loops ──────────────────────────

def plot_stress_loops_components(results, labels, outdir):
    """
    2×3 grid (rows: cases, columns: LV/RV/Septum).
    Each subplot overlays fiber, sheet, and normal stress-strain loops.
    """
    # Check available keys and report
    sample = results[0]
    print("  [stress_loops_components] Available stress/strain keys:")
    for k in sorted(sample.keys()):
        if k.startswith("mean_S_") or k.startswith("mean_E_"):
            print(f"    {k}")

    components = [
        ("ff", "Fiber",  "tab:blue"),
        ("ss", "Sheet",  "tab:orange"),
        ("nn", "Normal", "tab:green"),
    ]
    regions = ["LV", "RV", "Septum"]

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), squeeze=False)
    fig.suptitle(
        "Multi-Component Stress-Strain Loops: Fiber / Sheet / Normal\n"
        "(enclosed area proportional to work contribution in each direction)",
        fontsize=13, fontweight='bold'
    )

    for row, (r, lbl) in enumerate(zip(results, labels)):
        for col, region in enumerate(regions):
            ax = axes[row, col]
            has_data = False

            for comp, comp_label, color in components:
                strain_key = f"mean_E_{comp}_{region}"
                stress_key = f"mean_S_{comp}_{region}"

                strain = get_array(r, strain_key)
                stress = get_array(r, stress_key)

                if len(strain) == 0 or len(stress) == 0:
                    ax.text(0.5, 0.5 - 0.1 * components.index((comp, comp_label, color)),
                            f"{comp_label}: no data ({strain_key})",
                            transform=ax.transAxes, ha='center', fontsize=8,
                            color='gray')
                    continue

                has_data = True
                strain_pct = strain * 100
                stress_kpa = stress / 1e3

                ax.plot(strain_pct, stress_kpa, color=color, lw=1.8, label=comp_label)
                ax.plot(strain_pct[0], stress_kpa[0], 'o', color=color, ms=5)

            ax.set_xlabel("Strain (%)", fontsize=10)
            ax.set_ylabel("2nd-PK Stress (kPa)", fontsize=10)
            ax.set_title(f"{lbl} — {region}", fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='k', lw=0.5, alpha=0.4)
            ax.axvline(0, color='k', lw=0.5, alpha=0.4)

    plt.tight_layout()
    out = Path(outdir) / "stress_loops_components.png"
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─── Console Report ──────────────────────────────────────────────────────────

def print_report(results, labels):
    W = 100
    print("\n" + "=" * W)
    print(f"{'PROXY SENSITIVITY REPORT':^{W}}")
    print("=" * W)
    print(f"  Normalized to {labels[0]} = 1.0\n")

    regions = ["LV", "RV", "Septum"]

    for region in regions:
        print(f"  ── {region} {'Freewall' if region != 'Septum' else ''} ──")

        truth_keys = [
            (f"work_true_{region}",  "Total Work"),
            (f"work_fiber_{region}", "Fiber Work"),
        ]

        if region == "Septum":
            proxy_keys = [
                (f"work_ps_index_Septum_PLV",   "PS (P_LV)  "),
                (f"work_ps_index_Septum_Trans", "PS (Trans)  "),
                (f"work_ps_index_Septum_PRV",   "PS (P_RV)  "),
                (f"work_ps_index_Septum_Mean",  "PS (Mean)   "),
            ]
        else:
            proxy_keys = [
                (f"work_ps_index_{region}", "PS Proxy    "),
            ]

        all_keys = truth_keys + proxy_keys

        # Raw values
        for r, lbl in zip(results, labels):
            rv_peak = float(np.max(get_array(r, 'p_RV'))) if 'p_RV' in r else 0.0
            vals = "  ".join(f"{kl}: {total_work(r, kk):>10.3e}" for kk, kl in all_keys)
            print(f"    {lbl:<10} (RV peak {rv_peak:.1f} mmHg)  {vals}")

        # Normalized
        if len(results) >= 2:
            base, pah = results[0], results[1]
            for kk, kl in all_keys:
                b = total_work(base, kk)
                p = total_work(pah,  kk)
                ratio = p / b if abs(b) > 1e-12 else float('nan')
                print(f"      {kl:<14}: PAH/Healthy = {ratio:.3f}   ({b:.3e} → {p:.3e})")
        print()

    print("  Interpretation:")
    print("  * PS ratio ≈ True Work ratio  → proxy tracks well")
    print("  * PS ratio >> True Work ratio  → proxy over-estimates in PAH")
    print("  * Septum Trans proxy uses (P_LV − P_RV) to capture transmural loading")
    print("=" * W)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compare_cases.py <HEALTHY_RESULT_DIR> <PAH_RESULT_DIR>")
        sys.exit(1)

    folders = [sys.argv[1], sys.argv[2]]
    labels  = ["Healthy", "PAH"]
    outdir  = Path(__file__).resolve().parent

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

    print("\nGenerating figures...")
    plot_sensitivity(results, labels, outdir)
    plot_stress_loops(results, labels, outdir)
    plot_work_decomposition(results, labels, outdir)
    plot_pressure_strain(results, labels, outdir)
    plot_pressure_strain_longitudinal(results, labels, outdir)
    plot_stress_loops_components(results, labels, outdir)

    print("\nDone. Output files written to:", outdir)


if __name__ == "__main__":
    main()
