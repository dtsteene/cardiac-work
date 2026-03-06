#!/usr/bin/env python3
"""
compare_cases.py

Sensitivity analysis for LV Freewall, RV Freewall, and Septum mechanical work proxies.

Compares pressure-strain (PS) proxies against FEM ground truths:
  1. Total Work  (work_true  = full tensor S:dE)
  2. Fiber Work  (work_ff = S_ff * dE_ff)

For the Septum, multiple PS proxy variants are compared:
  - PLV:   P_LV × dε
  - PRV:   P_RV × dε
  - Trans: (P_LV − P_RV) × dε   (transmural pressure)

Produces:
  simplification_cascade.png         — S_ff×E_ff → P×E_ff → P×E_ll cascade
  pressure_vs_transmural_stress.png  — P ≈ -2σ̄_nn validation
  work_redistribution.png            — regional work shares
  work_decomposition.png             — ff/ss/nn/cross work breakdown
  stress_loops.png                   — fiber stress-strain loops
  pressure_strain.png                — pressure-strain loops
  sensitivity_longitudinal.png       — proxy tracking accuracy

Usage:
  python3 compare_cases.py <HEALTHY_RESULT_DIR> <PAH_RESULT_DIR>
"""

import sys
import argparse
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



def compute_ps_work(m, strain_key, p_key):
    """
    Compute pressure-strain work index from timeseries with any strain component.
    Returns total work in Joules (with region volume scaling derived from the
    precomputed fiber-strain PS proxy).
    """
    strain = get_array(m, strain_key)
    pressure = get_array(m, p_key) * 133.322  # mmHg → Pa
    if len(strain) < 2 or len(pressure) < 2:
        return 0.0
    dE = np.diff(strain)
    p_avg = 0.5 * (pressure[:-1] + pressure[1:])
    return float(np.sum(p_avg * dE))


def compute_ps_work_scaled(m, strain_key, p_key, region):
    """
    Compute PS work with volume scaling, consistent with the precomputed proxy.
    Uses the precomputed fiber-strain PS proxy to derive the region volume factor,
    then applies it to the new strain direction.
    """
    # Get volume scale factor from precomputed fiber PS proxy
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


# ─── Figure 1: Sensitivity Comparison ────────────────────────────────────────

def _plot_sensitivity_one(results, labels, outdir, strain_comp, strain_label, suffix):
    """
    2×3 grid: [Total truth, Fiber truth] × [LV, RV, Septum].
    Healthy = (1, 1). Ideal tracking → points on y = x.

    strain_comp: 'ff', 'nn', or 'ss'
    strain_label: e.g. "Fiber Strain E_ff" or "Longitudinal Strain E_nn"
    suffix: filename suffix, e.g. "fiber" or "longitudinal"
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Pressure-Strain Proxy Sensitivity — PS proxy uses {strain_label}\n"
        "(Healthy = 1.0  |  Ideal tracking: points lie on y = x)",
        fontsize=13, fontweight='bold'
    )

    def get_proxy_total(r, region):
        """Read precomputed PS work for this strain direction."""
        if region == "Septum":
            key = f"work_ps_{strain_comp}_Septum_PLV"
        else:
            key = f"work_ps_{strain_comp}_{region}"
        if key not in r:
            raise KeyError(f"Missing precomputed key '{key}' — rerun simulation with updated metrics_calculator")
        return total_work(r, key)

    # Freewall configs
    fw_configs = [
        (0, 0, "LV", "work_true_LV",  "Total Work (S:dE)",       "tab:blue"),
        (0, 1, "RV", "work_true_RV",  "Total Work (S:dE)",       "tab:red"),
        (1, 0, "LV", "work_ff_LV", "Fiber Work (S_ff·dE_ff, circ.)", "tab:blue"),
        (1, 1, "RV", "work_ff_RV", "Fiber Work (S_ff·dE_ff, circ.)", "tab:red"),
    ]

    for row, col, region, truth_key, truth_label, color in fw_configs:
        ax = axes[row, col]
        truth_vals = [total_work(r, truth_key) for r in results]
        proxy_vals = [get_proxy_total(r, region) for r in results]

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
        ax.plot(pn, tn, 'o-', color=color, lw=2, ms=10, label=f"PS({strain_label.split()[0]})")
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (pn[i], tn[i]), xytext=(8, 8),
                        textcoords='offset points', fontsize=9, color=color)
        if len(results) > 1:
            dx = pn[1] - tn[1]
            ax.annotate(f"Δ(proxy-truth) = {dx:+.2f}",
                        xy=(pn[1], tn[1]), xytext=(-10, -25),
                        textcoords='offset points', fontsize=8, color='dimgray',
                        arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel(f"PS Proxy Work [{strain_label}] (Normalized)", fontsize=10)
        ax.set_ylabel(f"{truth_label}\n(Normalized)", fontsize=10)
        ax.set_title(f"{region} Freewall  |  {truth_label}", fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Septum: multiple pressure variants (P_LV, Trans, P_RV)
    septum_pressure_variants = [
        ("p_LV",  "PS (P_LV)",    "tab:blue",  "o"),
        ("trans",  "PS (Trans)",    "tab:green", "s"),
        ("p_RV",  "PS (P_RV)",    "tab:red",   "^"),
    ]

    def get_septum_proxy(r, p_variant):
        """Read precomputed septum PS work for given pressure variant."""
        variant_map = {"p_LV": "PLV", "trans": "Trans", "p_RV": "PRV"}
        suffix = variant_map.get(p_variant)
        if suffix is None:
            raise ValueError(f"Unknown p_variant: {p_variant}")
        key = f"work_ps_{strain_comp}_Septum_{suffix}"
        if key not in r:
            raise KeyError(f"Missing precomputed key '{key}' — rerun simulation with updated metrics_calculator")
        return total_work(r, key)

    for row, truth_key, truth_label in [
        (0, "work_true_Septum",  "Total Work (S:dE)"),
        (1, "work_ff_Septum", "Fiber Work (S_ff·dE_ff, circ.)"),
    ]:
        ax = axes[row, 2]
        truth_vals = [total_work(r, truth_key) for r in results]
        bt = truth_vals[0]
        if abs(bt) < 1e-12:
            ax.set_title("Septum — no data")
            continue
        tn = [v / bt for v in truth_vals]

        all_v = list(tn)
        first_proxy_pn = None
        for p_variant, plbl, pcol, pmk in septum_pressure_variants:
            pvals = [get_septum_proxy(r, p_variant) for r in results]
            bp = pvals[0]
            if abs(bp) < 1e-12:
                continue
            pn = [v / bp for v in pvals]
            all_v.extend(pn)
            ax.plot(pn, tn, f'{pmk}-', color=pcol, lw=2, ms=9, label=plbl)
            if first_proxy_pn is None:
                first_proxy_pn = pn

        lo, hi = min(all_v) - 0.15, max(all_v) + 0.15
        lo, hi = min(lo, 0.5), max(hi, 1.1)
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, lw=1.5, label="Ideal (y = x)")

        if first_proxy_pn is not None:
            for i, lbl in enumerate(labels):
                ax.annotate(lbl, (first_proxy_pn[i], tn[i]), xytext=(8, 8),
                            textcoords='offset points', fontsize=9, color='dimgray')

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel(f"PS Proxy Work [{strain_label}] (Normalized)", fontsize=10)
        ax.set_ylabel(f"{truth_label}\n(Normalized)", fontsize=10)
        ax.set_title(f"Septum  |  {truth_label}", fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(outdir) / f"sensitivity_{suffix}.png"
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


def plot_sensitivity(results, labels, outdir):
    """Generate sensitivity plot for longitudinal strain PS proxy (clinically relevant)."""
    # Require longitudinal strain — no fallback
    _plot_sensitivity_one(results, labels, outdir,
                          "ll", "Longitudinal Strain E_ll (apex-to-base)", "longitudinal")


# ─── Figure 2: Fiber Stress-Strain Loops ─────────────────────────────────────

def plot_stress_loops(results, labels, outdir):
    """
    2×3 grid:
      Row 0: fiber stress-strain loops (S_ff vs E_ff) for LV, RV, Septum
      Row 1: fiber strain + pressure vs time         for LV, RV, Septum
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Fiber Stress-Strain Loops (S_ff vs E_ff, circumferential): LV & RV + Septum  (Healthy vs PAH)",
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
        ("work_ff",    "Fiber\n(S_ff·dE_ff)\ncircumferential"),
        ("work_ss",    "Sheet\n(S_ss·dE_ss)\napicobasal"),
        ("work_nn",    "Transmural\n(S_nn·dE_nn)\nsheet-normal"),
        ("work_cross", "Cross\n(off-diagonal)"),
    ]

    case_colors = ['tab:blue', 'tab:red']

    # PS proxy uses precomputed longitudinal strain (E_ll)
    # Fail hard if not available — requires new simulation with work_ps_ll_* keys
    ps_strain_tag = "E_ll"

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Work Decomposition: Fiber (circ.) / Sheet (apico.) / Transmural / Cross\n"
        f"(negative = work done BY myocardium | PS proxy uses P × {ps_strain_tag}, shown as dashed lines)",
        fontsize=13, fontweight='bold'
    )

    def _ps_ll(r, region, p_variant="default"):
        """Read precomputed PS work using longitudinal strain (work_ps_ll_*)."""
        if region in ("LV", "RV"):
            key = f"work_ps_ll_{region}"
        elif p_variant == "default":
            key = "work_ps_ll_Septum_PLV"
        elif p_variant == "trans":
            key = "work_ps_ll_Septum_Trans"
        else:
            raise ValueError(f"Unknown p_variant: {p_variant}")
        if key not in r:
            raise KeyError(f"Missing precomputed key '{key}' — rerun simulation with updated metrics_calculator")
        return total_work(r, key) * 1e3

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

        # PS proxy reference lines (using longitudinal strain)
        if region in ("LV", "RV"):
            for j, (r, lbl) in enumerate(zip(results, labels)):
                ps_val = _ps_ll(r, region)
                ax.axhline(ps_val, ls=':', lw=1.5, color=case_colors[j],
                           label=f"PS({ps_strain_tag}) ({lbl})")
        else:  # Septum: P_LV and Trans only
            for p_variant, plbl, ls_style in [
                ("default", f"PS(P_LV, {ps_strain_tag})", ':'),
                ("trans",   f"PS(Trans, {ps_strain_tag})", '-.'),
            ]:
                for j, (r, lbl) in enumerate(zip(results, labels)):
                    ps_val = _ps_ll(r, region, p_variant)
                    ax.axhline(ps_val, ls=ls_style, lw=1.5, color=case_colors[j],
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
    2×3 panel: pressure-strain loops for LV, RV, Septum.
    Row 0: Fiber strain (E_ff — circumferential direction)
    Row 1: Longitudinal strain (E_ll preferred, E_nn fallback)
    For Septum, both P_LV and transmural (P_LV-P_RV) loops are shown.
    """
    strain_rows = [
        ("mean_E_ff", "Fiber Strain E_ff (circumferential)"),
        ("mean_E_ll", "Longitudinal Strain E_ll (apex-to-base)"),
    ]

    n_rows = len(strain_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 6 * n_rows), squeeze=False)
    fig.suptitle(
        "Pressure-Strain Loops: LV & RV Freewalls + Septum  (Healthy vs PAH)\n"
        "(loop area ~ pressure-strain work index)",
        fontsize=13, fontweight='bold'
    )

    colors = ['tab:blue', 'tab:red']

    for row, (strain_prefix, strain_label) in enumerate(strain_rows):
        # Freewalls
        for col, (region, p_key) in enumerate([("LV", "p_LV"), ("RV", "p_RV")]):
            ax = axes[row, col]
            strain_key = f"{strain_prefix}_{region}"
            for r, lbl, c in zip(results, labels, colors):
                strain = get_array(r, strain_key) * 100
                pressure = get_array(r, p_key)
                if len(strain) == 0 or len(pressure) == 0:
                    continue
                ax.plot(strain, pressure, color=c, lw=1.8, label=lbl)
                ax.plot(strain[0], pressure[0], 'o', color=c, ms=6)

            short_label = strain_label.split("(")[0].strip()
            ax.set_xlabel(f"{region} {strain_label} (%)", fontsize=10)
            ax.set_ylabel(f"{region} Cavity Pressure (mmHg)", fontsize=10)
            ax.set_title(f"{region} Freewall — {short_label}", fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='k', lw=0.5, alpha=0.4)
            ax.axvline(0, color='k', lw=0.5, alpha=0.4)

        # Septum: show P_LV and transmural loops
        ax = axes[row, 2]
        strain_key = f"{strain_prefix}_Septum"
        for r, lbl, c in zip(results, labels, colors):
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

        short_label = strain_label.split("(")[0].strip()
        ax.set_xlabel(f"Septum {strain_label} (%)", fontsize=10)
        ax.set_ylabel("Pressure (mmHg)", fontsize=10)
        ax.set_title(f"Septum — {short_label}\n"
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


# ─── Figure 7: Fiber Work Efficiency Ratio ───────────────────────────────────

def plot_fiber_efficiency(results, labels, outdir):
    """
    1×2 grouped bar chart: fiber efficiency and normal fraction per region.
    efficiency = work_ff / work_true
    normal_fraction = work_nn / work_true
    """
    regions = ["LV", "RV", "Septum"]
    case_colors = ['tab:blue', 'tab:red']

    # Compute ratios and print table
    W = 90
    print("\n" + "=" * W)
    print(f"{'FIBER EFFICIENCY & TRANSMURAL FRACTION':^{W}}")
    print("=" * W)
    print(f"  {'Region':<10} {'Case':<10} {'W_true (mJ)':>12} {'W_ff (mJ)':>13} "
          f"{'W_nn (mJ)':>14} {'Fiber Eff':>10} {'Trans Frac':>12}")
    print("-" * W)

    table = {}  # (region, case_idx) -> (fiber_eff, normal_frac)
    for j, (r, lbl) in enumerate(zip(results, labels)):
        for region in regions:
            wt = total_work(r, f"work_true_{region}")
            wf = total_work(r, f"work_ff_{region}")
            wn = total_work(r, f"work_nn_{region}")
            fe = wf / wt if abs(wt) > 1e-15 else float('nan')
            nf = wn / wt if abs(wt) > 1e-15 else float('nan')
            table[(region, j)] = (fe, nf)
            print(f"  {region:<10} {lbl:<10} {wt*1e3:>12.3f} {wf*1e3:>13.3f} "
                  f"{wn*1e3:>14.3f} {fe:>10.3f} {nf:>12.3f}")

    if len(results) >= 2:
        print("-" * W)
        print("  Change Healthy → PAH:")
        for region in regions:
            fe_h, nf_h = table[(region, 0)]
            fe_p, nf_p = table[(region, 1)]
            print(f"    {region:<10} Fiber Eff: {fe_h:.3f} → {fe_p:.3f} (Δ{fe_p-fe_h:+.3f})   "
                  f"Trans Frac: {nf_h:.3f} → {nf_p:.3f} (Δ{nf_p-nf_h:+.3f})")
    print("=" * W)

    # Plot (Septum excluded — work_true near zero in PAH makes ratios undefined)
    plot_regions = ["LV", "RV"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Fiber Work Efficiency & Transmural Work Fraction by Region\n"
        "(efficiency = W_component / W_true;  key question: does RV fiber efficiency collapse in PAH?)",
        fontsize=12, fontweight='bold'
    )

    x = np.arange(len(plot_regions))
    n = len(results)
    width = 0.35

    for panel, (ax, metric_label, metric_idx) in enumerate(zip(
        axes,
        ["Fiber Efficiency\n(W_ff / W_true)", "Transmural Fraction\n(W_nn / W_true)"],
        [0, 1],
    )):
        for j, (r, lbl) in enumerate(zip(results, labels)):
            vals = [table[(reg, j)][metric_idx] for reg in plot_regions]
            offset = (j - (n - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width,
                          label=lbl, color=case_colors[j],
                          alpha=0.75 if j == 0 else 0.55,
                          edgecolor='k', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.2f}", ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(plot_regions, fontsize=11)
        ax.set_ylabel("Fraction of Total Work", fontsize=11)
        ax.set_ylim(-0.1, 1.0)
        ax.set_title(metric_label, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='k', lw=0.8)
        ax.axhline(1, color='k', lw=0.5, ls=':', alpha=0.3)

    fig.text(0.5, -0.02,
             "Septum excluded: work_true near zero in PAH makes efficiency ratios undefined.",
             ha='center', fontsize=9, style='italic', color='dimgray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = Path(outdir) / "fiber_efficiency.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


# ─── Figure 8: PS Proxy Directional Correlation ─────────────────────────────

def plot_proxy_directional_match(results, labels, outdir):
    """
    1×3 grouped bar chart: PS proxy (with 3 strain directions) vs true work components.
    Shows PS(E_ff), PS(E_nn), PS(E_ss) alongside fiber/sheet/normal/shear true work.
    """
    regions = ["LV", "RV", "Septum"]
    case_colors = ['tab:blue', 'tab:red']

    def p_key_for(region):
        return "p_LV" if region in ("LV", "Septum") else "p_RV"

    long_short = "E_ll"

    # Bar categories: 3 PS proxy variants + 4 true work components
    components = [
        (f"PS({long_short})\nlongitud.", "ps_long"),
        ("PS(E_ff)\nfiber",   "ps_ff"),
        ("PS(E_ss)\nsheet",   "ps_ss"),
        ("W_ff\n(circ.)",     "work_ff"),
        ("W_ss\n(apico.)",    "work_ss"),
        ("W_nn\n(transm.)",   "work_nn"),
        ("W_cross\n(off-diag)", "work_cross"),
    ]

    # Print table
    W = 120
    print("\n" + "=" * W)
    print(f"{'PS PROXY (3 strain dirs) vs DIRECTIONAL WORK COMPONENTS (mJ)':^{W}}")
    print("=" * W)
    print(f"  Longitudinal strain: E_ll (apex-to-base)")

    def _ps_precomputed(r, strain_comp, region):
        """Read precomputed PS proxy for given strain component and region."""
        if region == "Septum":
            key = f"work_ps_{strain_comp}_Septum_PLV"
        else:
            key = f"work_ps_{strain_comp}_{region}"
        if key not in r:
            raise KeyError(f"Missing precomputed key '{key}' — rerun simulation with updated metrics_calculator")
        return total_work(r, key)

    for region in regions:
        print(f"\n  ── {region} (pressure: {'P_LV' if region != 'RV' else 'P_RV'}) ──")
        print(f"  {'Case':<10} {'PS('+long_short+')':>10} {'PS(E_ff)':>10} {'PS(E_ss)':>10} "
              f"{'W_ff':>10} {'W_ss':>10} {'W_nn':>10} {'W_cross':>10}")
        print("  " + "-" * 100)
        for r, lbl in zip(results, labels):
            ps_long = _ps_precomputed(r, "ll", region) * 1e3
            ps_ff = _ps_precomputed(r, "ff", region) * 1e3
            ps_ss = compute_ps_work_scaled(r, f"mean_E_ss_{region}", p_key_for(region), region) * 1e3
            wf  = total_work(r, f"work_ff_{region}") * 1e3
            ws  = total_work(r, f"work_ss_{region}") * 1e3
            wn  = total_work(r, f"work_nn_{region}") * 1e3
            wsh = total_work(r, f"work_cross_{region}") * 1e3
            print(f"  {lbl:<10} {ps_long:>10.3f} {ps_ff:>10.3f} {ps_ss:>10.3f} "
                  f"{wf:>10.3f} {ws:>10.3f} {wn:>10.3f} {wsh:>10.3f}")

    print("=" * W)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        f"PS Proxy (3 strain directions) vs True Work Components\n"
        f"PS({long_short}) = longitudinal (clinical GLS analogue)  |  "
        f"PS(E_ff) = fiber (circumferential)",
        fontsize=12, fontweight='bold'
    )

    for ax, region in zip(axes, regions):
        x = np.arange(len(components))
        n = len(results)
        width = 0.35

        for j, (r, lbl) in enumerate(zip(results, labels)):
            vals = []
            for _, comp_id in components:
                if comp_id == "ps_long":
                    vals.append(_ps_precomputed(r, "ll", region) * 1e3)
                elif comp_id == "ps_ff":
                    vals.append(_ps_precomputed(r, "ff", region) * 1e3)
                elif comp_id == "ps_ss":
                    vals.append(compute_ps_work_scaled(r, f"mean_E_ss_{region}", p_key_for(region), region) * 1e3)
                else:
                    vals.append(total_work(r, f"{comp_id}_{region}") * 1e3)
            offset = (j - (n - 1) / 2) * width
            ax.bar(x + offset, vals, width,
                   label=lbl, color=case_colors[j],
                   alpha=0.75 if j == 0 else 0.55,
                   edgecolor='k', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in components], fontsize=8)
        ax.set_ylabel("Work (mJ)", fontsize=11)
        p_label = "P_LV" if region != "RV" else "P_RV"
        ax.set_title(f"{region}  (pressure = {p_label})", fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='k', lw=0.8)

    plt.tight_layout()
    out = Path(outdir) / "proxy_directional_match.png"
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─── Figure 9: Biventricular Work Redistribution ────────────────────────────

def plot_work_redistribution(results, labels, outdir):
    """
    1×2 stacked bar chart: regional share of total biventricular work.
    Panel 1: total work (work_true). Panel 2: fiber work only.
    """
    regions = ["LV", "RV", "Septum"]
    region_colors = ['tab:blue', 'tab:red', 'tab:green']

    W = 90
    print("\n" + "=" * W)
    print(f"{'BIVENTRICULAR WORK REDISTRIBUTION':^{W}}")
    print("=" * W)

    shares = {}  # (work_type, case_idx) -> [lv%, rv%, septum%]
    for wtype, wkey in [("Total", "work_true"), ("Fiber", "work_ff")]:
        print(f"\n  ── {wtype} Work ──")
        print(f"  {'Case':<10} {'LV (mJ)':>10} {'RV (mJ)':>10} {'Sep (mJ)':>10} "
              f"{'Total (mJ)':>11} {'LV%':>6} {'RV%':>6} {'Sep%':>6}")
        print("  " + "-" * 73)
        for j, (r, lbl) in enumerate(zip(results, labels)):
            vals = [total_work(r, f"{wkey}_{reg}") for reg in regions]
            tot = sum(vals)
            pcts = [v / tot * 100 if abs(tot) > 1e-15 else 0.0 for v in vals]
            shares[(wtype, j)] = pcts
            print(f"  {lbl:<10} {vals[0]*1e3:>10.3f} {vals[1]*1e3:>10.3f} {vals[2]*1e3:>10.3f} "
                  f"{tot*1e3:>11.3f} {pcts[0]:>5.1f}% {pcts[1]:>5.1f}% {pcts[2]:>5.1f}%")

        if len(results) >= 2:
            pct_h = shares[(wtype, 0)]
            pct_p = shares[(wtype, 1)]
            print(f"  {'Δ(PAH-H)':<10} "
                  f"{'':>10} {'':>10} {'':>10} {'':>11} "
                  f"{pct_p[0]-pct_h[0]:>+5.1f}% {pct_p[1]-pct_h[1]:>+5.1f}% {pct_p[2]-pct_h[2]:>+5.1f}%")
    print("=" * W)

    # Plot: grouped bars showing absolute work (mJ) with percentage annotations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Biventricular Work Redistribution: Absolute Work by Region\n"
        "(key question: does RV share of total work rise while fiber work share falls in PAH?)",
        fontsize=12, fontweight='bold'
    )

    case_colors = ['tab:blue', 'tab:red']

    for ax, (wtype, wkey) in zip(axes, [("Total Work", "work_true"), ("Fiber Work", "work_ff")]):
        x = np.arange(len(regions))
        n = len(results)
        width = 0.35

        for j, (r, lbl) in enumerate(zip(results, labels)):
            reg_vals = [total_work(r, f"{wkey}_{reg}") for reg in regions]
            tot = sum(reg_vals)
            vals_mJ = [v * 1e3 for v in reg_vals]
            pcts = [v / tot * 100 if abs(tot) > 1e-15 else 0.0 for v in reg_vals]

            offset = (j - (n - 1) / 2) * width
            bars = ax.bar(x + offset, vals_mJ, width,
                          label=lbl, color=case_colors[j],
                          alpha=0.75 if j == 0 else 0.55,
                          edgecolor='k', linewidth=0.5)
            for bar, v_mJ, pct in zip(bars, vals_mJ, pcts):
                va = 'bottom' if v_mJ >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v_mJ:.2f}\n({pct:.0f}%)", ha='center', va=va, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(regions, fontsize=11)
        ax.set_ylabel("Work (mJ)", fontsize=11)
        ax.set_title(wtype, fontweight='bold', fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='k', lw=0.8)

    plt.tight_layout()
    out = Path(outdir) / "work_redistribution.png"
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─── Figure 10: Pressure-Strain Dyssynchrony Index ──────────────────────────

def plot_dyssynchrony(results, labels, outdir):
    """
    Grouped bar chart: time delay (ms) between peak cavity pressure and
    peak fiber shortening (most negative E_ff) for each region and case.
    """
    regions_cfg = [
        ("LV",     "mean_E_ff_LV",     [("P_LV", "p_LV")]),
        ("RV",     "mean_E_ff_RV",     [("P_RV", "p_RV")]),
        ("Septum", "mean_E_ff_Septum", [("P_LV", "p_LV"), ("P_RV", "p_RV")]),
    ]

    W = 90
    print("\n" + "=" * W)
    print(f"{'PRESSURE-STRAIN DYSSYNCHRONY INDEX':^{W}}")
    print("=" * W)
    print(f"  Delay = t(peak pressure) - t(peak fiber shortening, E_ff)  [ms]")
    print(f"  Positive delay = pressure peaks AFTER max shortening")
    print("-" * W)

    # Collect all bar data: list of (bar_label, {case: delay_ms})
    bar_data = []

    for region, strain_key, pressure_defs in regions_cfg:
        for p_label, p_key in pressure_defs:
            bar_label = f"{region}\nvs {p_label}" if len(pressure_defs) > 1 else region
            delays = {}
            for j, (r, lbl) in enumerate(zip(results, labels)):
                t = get_array(r, 'time')
                strain = get_array(r, strain_key)
                pressure = get_array(r, p_key)
                if len(t) == 0 or len(strain) == 0 or len(pressure) == 0:
                    delays[lbl] = float('nan')
                    continue
                t_peak_p = t[np.argmax(pressure)]
                t_peak_short = t[np.argmin(strain)]  # most negative = peak shortening
                delay_ms = (t_peak_p - t_peak_short) * 1000
                delays[lbl] = delay_ms
                print(f"  {region:<8} vs {p_label:<4}  {lbl:<10}  "
                      f"t_peak_P={t_peak_p*1000:.0f}ms  t_peak_short={t_peak_short*1000:.0f}ms  "
                      f"delay={delay_ms:+.1f}ms")
            bar_data.append((bar_label, delays))

    print("=" * W)

    # Plot
    case_colors = ['tab:blue', 'tab:red']
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "Pressure-Strain Dyssynchrony Index (using fiber strain E_ff)\n"
        "delay = t(peak pressure) - t(peak fiber shortening);  "
        "positive = pressure peaks after max shortening",
        fontsize=12, fontweight='bold'
    )

    x = np.arange(len(bar_data))
    n = len(results)
    width = 0.35

    for j, (lbl, c) in enumerate(zip(labels, case_colors)):
        vals = [bd[1].get(lbl, 0.0) for bd in bar_data]
        offset = (j - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=lbl, color=c,
                      alpha=0.75 if j == 0 else 0.55,
                      edgecolor='k', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                va = 'bottom' if v >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(), f"{v:+.0f}",
                        ha='center', va=va, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([bd[0] for bd in bar_data], fontsize=10)
    ax.set_ylabel("Delay (ms)", fontsize=11)
    ax.set_title("Peak Pressure − Peak Shortening Delay per Region", fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='k', lw=0.8)

    plt.tight_layout()
    out = Path(outdir) / "dyssynchrony.png"
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─── Figure: Simplification Cascade ──────────────────────────────────────────

def plot_simplification_cascade(results, labels, outdir):
    """
    3×3 grid showing the cascade of clinical simplifications:
      Row 0: S_ff vs E_ff  (FEM fiber stress-strain — gold standard)
      Row 1: P   vs E_ff   (replace stress with cavity pressure)
      Row 2: P   vs E_ll   (replace fiber strain with longitudinal/GLS)
    Columns: LV, RV, Septum.
    """
    colors = ['tab:blue', 'tab:red']
    regions = [
        ("LV",     "mean_E_ff_LV",     "mean_S_ff_LV",     "mean_E_ll_LV",     "p_LV"),
        ("RV",     "mean_E_ff_RV",     "mean_S_ff_RV",     "mean_E_ll_RV",     "p_RV"),
        ("Septum", "mean_E_ff_Septum", "mean_S_ff_Septum", "mean_E_ll_Septum", "p_LV"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle(
        "Cascade of Clinical Simplifications: S:dE → P·dE_ff → P·dE_ll\n"
        "(each row is a further approximation toward what clinicians can measure)",
        fontsize=14, fontweight='bold'
    )

    row_labels = [
        "Level 1: S_ff vs E_ff\n(FEM fiber stress-strain)",
        "Level 2: P vs E_ff\n(replace stress with pressure)",
        "Level 3: P vs E_ll\n(replace fiber with longitudinal = GLS)",
    ]

    for col, (region, eff_key, sff_key, ell_key, p_key) in enumerate(regions):
        # Row 0: S_ff vs E_ff
        ax = axes[0, col]
        for r, lbl, c in zip(results, labels, colors):
            Eff = get_array(r, eff_key) * 100
            Sff = get_array(r, sff_key) / 1e3  # Pa → kPa
            if len(Eff) > 0:
                ax.plot(Eff, Sff, color=c, lw=1.8, label=lbl)
                ax.plot(Eff[0], Sff[0], 'o', color=c, ms=6)
        ax.set_xlabel("Fiber Strain E_ff (%)")
        ax.set_ylabel("Fiber Stress S_ff (kPa)")
        ax.set_title(f"{region} — S_ff vs E_ff  (ground truth)", fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.axvline(0, color='k', lw=0.5, alpha=0.4)

        # Row 1: P vs E_ff
        ax = axes[1, col]
        for r, lbl, c in zip(results, labels, colors):
            Eff = get_array(r, eff_key) * 100
            pres = get_array(r, p_key)
            if len(Eff) > 0:
                ax.plot(Eff, pres, color=c, lw=1.8, label=lbl)
                ax.plot(Eff[0], pres[0], 'o', color=c, ms=6)
                if region == "Septum":
                    p_rv = get_array(r, 'p_RV')
                    p_trans = pres - p_rv
                    ax.plot(Eff, p_trans, color=c, lw=1.5, ls='--', alpha=0.7,
                            label=f"{lbl} (Trans)")
        ax.set_xlabel("Fiber Strain E_ff (%)")
        ax.set_ylabel("Cavity Pressure (mmHg)")
        ax.set_title(f"{region} — P vs E_ff  (replace stress with pressure)", fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.axvline(0, color='k', lw=0.5, alpha=0.4)

        # Row 2: P vs E_ll (GLS analogue)
        ax = axes[2, col]
        for r, lbl, c in zip(results, labels, colors):
            Ell = get_array(r, ell_key) * 100
            pres = get_array(r, p_key)
            if len(Ell) > 0:
                ax.plot(Ell, pres, color=c, lw=1.8, label=lbl)
                ax.plot(Ell[0], pres[0], 'o', color=c, ms=6)
                if region == "Septum":
                    p_rv = get_array(r, 'p_RV')
                    p_trans = pres - p_rv
                    ax.plot(Ell, p_trans, color=c, lw=1.5, ls='--', alpha=0.7,
                            label=f"{lbl} (Trans)")
        ax.set_xlabel("Longitudinal Strain E_ll (%, GLS analogue)")
        ax.set_ylabel("Cavity Pressure (mmHg)")
        ax.set_title(f"{region} — P vs E_ll  (clinical measurement)", fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.axvline(0, color='k', lw=0.5, alpha=0.4)

    # Row labels on the left
    for row, rlabel in enumerate(row_labels):
        axes[row, 0].annotate(rlabel, xy=(-0.35, 0.5), xycoords='axes fraction',
                              fontsize=10, fontweight='bold', ha='right', va='center',
                              rotation=90)

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    out = Path(outdir) / "simplification_cascade.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


# ─── Figure: Pressure vs Transmural Stress ──────────────────────────────────

def plot_pressure_vs_radial_stress(results, labels, outdir):
    """
    2×3 panel showing why P ≈ -2σ̄_nn (pressure is proportional to transmural stress).
      Row 0: Time traces of P(t) and -σ̄_nn(t) overlaid for each region
      Row 1: Scatter of -σ̄_nn vs P at each timestep, with regression line
    Columns: LV, RV, Septum.
    """
    colors = ['tab:blue', 'tab:red']
    regions = [
        ("LV",     "mean_sigma_nn_LV",     "p_LV",  None),
        ("RV",     "mean_sigma_nn_RV",     "p_RV",  None),
        ("Septum", "mean_sigma_nn_Septum", "p_LV",  "p_RV"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "Pressure vs Volume-Averaged Transmural (Cauchy) Stress\n"
        "Theory: σ̄_nn ≈ -P/2 (thin-wall approximation)  |  "
        "Confirms that cavity pressure measures transmural stress",
        fontsize=13, fontweight='bold'
    )

    for col, (region, sigma_key, p_key, p_key2) in enumerate(regions):
        ax_time = axes[0, col]
        ax_scatter = axes[1, col]

        for i, (r, lbl, c) in enumerate(zip(results, labels, colors)):
            t = get_array(r, 'time')
            sigma_nn = get_array(r, sigma_key)  # Pa, negative (compressive)
            pres_mmHg = get_array(r, p_key)
            pres_Pa = pres_mmHg * 133.322

            if len(t) == 0 or len(sigma_nn) == 0:
                continue

            neg_sigma = -sigma_nn  # flip sign: compressive → positive

            # Row 0: time traces — both on same axis in kPa for direct comparison
            ax_time.plot(t * 1000, pres_Pa / 1e3, color=c, lw=2,
                         label=f"{lbl} — P")
            ax_time.plot(t * 1000, neg_sigma / 1e3, color=c, lw=1.5, ls='--',
                         label=f"{lbl} — -σ̄_nn")
            # Also plot P/2 as thin dotted line (thin-wall prediction)
            ax_time.plot(t * 1000, pres_Pa / 2e3, color=c, lw=1, ls=':',
                         alpha=0.5, label=f"{lbl} — P/2" if i == 0 else None)

            # Row 1: scatter -σ̄_nn vs P
            ax_scatter.scatter(pres_Pa / 1e3, neg_sigma / 1e3, color=c, s=15,
                               alpha=0.5, label=f"{lbl}")

            # Regression line (through origin)
            if len(pres_Pa) > 2:
                slope = np.sum(neg_sigma * pres_Pa) / np.sum(pres_Pa**2)
                p_fit = np.linspace(0, pres_Pa.max(), 50)
                ax_scatter.plot(p_fit / 1e3, slope * p_fit / 1e3, color=c, lw=1.5,
                                ls=':', alpha=0.8,
                                label=f"slope={slope:.3f} (theory=0.5)")

            # Peak ratio annotation
            peak_p = np.max(pres_Pa)
            peak_sigma = np.max(neg_sigma)
            ratio = peak_sigma / peak_p if peak_p > 0 else 0
            ax_scatter.annotate(
                f"{lbl}: |σ̄_nn|/P = {ratio:.3f}",
                xy=(0.05, 0.92 - i * 0.08), xycoords='axes fraction',
                fontsize=9, color=c, fontweight='bold'
            )

        # Format time axis
        ax_time.set_xlabel("Time (ms)")
        ax_time.set_ylabel("Stress / Pressure (kPa)", fontsize=10)
        p_label = p_key.replace("p_", "P_")
        ax_time.set_title(f"{region} — P(t) vs -σ̄_nn(t)\n"
                          f"(solid = {p_label}, dashed = -σ̄_nn, dotted = P/2)",
                          fontweight='bold')
        ax_time.legend(fontsize=7, loc='upper left')
        ax_time.grid(True, alpha=0.3)

        # Format scatter
        ax_scatter.set_xlabel("Cavity Pressure P (kPa)")
        ax_scatter.set_ylabel("-σ̄_nn  Transmural Stress (kPa)")
        ax_scatter.set_title(f"{region} — -σ̄_nn vs P  (slope should be ≈ 0.5)",
                             fontweight='bold')

        # Reference line: y = 0.5x (thin-wall prediction)
        xlim = ax_scatter.get_xlim()
        x_ref = np.linspace(0, xlim[1] * 1.1, 50)
        ax_scatter.plot(x_ref, 0.5 * x_ref, 'k--', lw=1, alpha=0.4,
                        label="Theory: σ̄_nn = P/2")
        ax_scatter.legend(fontsize=7, loc='upper left')
        ax_scatter.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(outdir) / "pressure_vs_transmural_stress.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
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
            (f"work_ff_{region}", "Fiber Work (circ.)"),
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
    print("  * PS proxy uses FIBER STRAIN (E_ff, circumferential) — not longitudinal/GLS")
    print("  * PS ratio ≈ True Work ratio  → proxy tracks well")
    print("  * PS ratio >> True Work ratio  → proxy over-estimates in PAH")
    print("  * Septum Trans proxy uses (P_LV − P_RV) to capture transmural loading")
    print("=" * W)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare Healthy vs PAH cardiac work metrics.")
    parser.add_argument("healthy_dir", help="Path to Healthy result directory")
    parser.add_argument("pah_dir", help="Path to PAH result directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--primary", action="store_true",
                       help="Run only the primary figure set")
    group.add_argument("--all", dest="run_all", action="store_true", default=True,
                       help="Run all figures (default)")
    parser.add_argument("--outdir", type=str, default=None,
                       help="Output directory (default: results/sims/compare_cases)")
    args = parser.parse_args()

    # --primary implies not --all
    if args.primary:
        args.run_all = False

    folders = [args.healthy_dir, args.pah_dir]
    labels  = ["Healthy", "PAH"]
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

    # Primary figure set (7 core figures — the cascade story)
    primary_figures = [
        plot_simplification_cascade,       # THE key figure: S_ff×E_ff → P×E_ff → P×E_ll
        plot_pressure_vs_radial_stress,    # Why P works: P ≈ -2σ̄_nn
        plot_work_redistribution,
        plot_work_decomposition,
        plot_stress_loops,
        plot_pressure_strain,
        plot_sensitivity,
    ]

    # Additional figures (--all only)
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

    print("\nDone. Output files written to:", outdir)


if __name__ == "__main__":
    main()
