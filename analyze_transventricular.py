#!/usr/bin/env python3
"""
analyze_transventricular.py — Transventricular Work Profiles and Sensitivity Curves

Loads per_cell_data.npz from one or more completed simulations and produces:

1. **Transventricular profile plots**: work density vs tau (LV→RV), showing where
   each pressure proxy tracks the true work and where it diverges.

2. **Sensitivity curves**: proxy accuracy integrated over [tau_cutoff, 1-tau_cutoff]
   as the septum boundary shifts from tight (geometric) to wide (LDRB union).
   Reference points marked: geometric septum, LDRB septum, TriSeg volume.

3. **Clinical summary**: single-number proxy ratios with uncertainty bounds from
   the sensitivity curve.

Usage:
    python analyze_transventricular.py <results_dir> [<results_dir2> ...]
    python analyze_transventricular.py results/sims/2026-04-06/UKB_6beats_run_1017000
    # Multi-case comparison:
    python analyze_transventricular.py results/sims/2026-04-0?/UKB_6beats_run_101700*
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Transventricular work profile analysis")
parser.add_argument("result_dirs", nargs="+", type=Path,
                    help="One or more simulation result directories")
parser.add_argument("--nbins", type=int, default=12,
                    help="Number of tau bins for profiles (default 12)")
parser.add_argument("--output-dir", type=Path, default=None,
                    help="Output directory for figures (default: first result dir / transventricular_analysis)")
parser.add_argument("--label-from", choices=["description", "dirname"], default="description",
                    help="How to label each case in plots")
args = parser.parse_args()

# ── Load all cases ───────────────────────────────────────────────────────────

def load_case(result_dir: Path) -> dict:
    """Load per_cell_data.npz + metadata for a single sim."""
    pc_path = result_dir / "per_cell_data.npz"
    if not pc_path.exists():
        print(f"ERROR: {pc_path} not found. Run compute_per_cell.py first.")
        sys.exit(1)

    data = dict(np.load(pc_path))

    # Get label
    desc_path = result_dir / "run_description.txt"
    if desc_path.exists() and args.label_from == "description":
        raw = desc_path.read_text().strip()
        # Try to extract a meaningful identifier from various description formats
        if raw.startswith("Phase4 thickness"):
            # "Phase4 thickness global_1mm severe 6beats" -> "global_1mm"
            parts = raw.split()
            label = parts[2] if len(parts) > 2 else result_dir.name
        elif raw.startswith("Phase1 v2circ"):
            # "Phase1 v2circ healthy 6beats new_circ_lib" -> "healthy"
            parts = raw.split()
            label = parts[2] if len(parts) > 2 else result_dir.name
        elif raw.startswith("Phase3"):
            # "Phase3 mesh convergence char_length=5 mild 6beats..." -> "fine_mild"
            label = "fine_" + (raw.split()[-3] if len(raw.split()) >= 3 else "mesh")
        else:
            # Fall back to first token after stripping known prefixes
            for prefix in ["Phase1 v2circ ", "v2circ ", "Phase1 ", "Phase4 ", "Phase3 "]:
                if raw.startswith(prefix):
                    raw = raw[len(prefix):]
                    break
            label = raw.split()[0] if raw else result_dir.name
    else:
        label = result_dir.name

    # Try to get achieved RV pressure for disease ordering
    # Use the pressure_history.npy (solver pressures) — last beat only
    rv_esp_achieved = None
    pres_path = result_dir / "solver" / "pressure_history.npy"
    sim_params_path = result_dir / "simulation_params.json"
    if pres_path.exists() and sim_params_path.exists():
        try:
            with open(sim_params_path) as f:
                sp = json.load(f)
            bpm = sp.get("BPM", 75)
            dt = sp.get("dt", 0.001)
            steps_per_beat = int(round((60.0 / bpm) / dt))
            pres = np.load(pres_path)  # shape (n_steps, 2) = [LV, RV]
            if pres.ndim == 2 and pres.shape[1] >= 2:
                last_beat_pres = pres[-steps_per_beat:]
                rv_esp_achieved = float(np.max(last_beat_pres[:, 1]))
        except Exception:
            pass

    data["__label"] = label
    data["__dir"] = result_dir
    data["__rv_esp"] = rv_esp_achieved
    return data


cases = [load_case(d.resolve()) for d in args.result_dirs]

# Sort by achieved RV ESP if available (disease severity ordering)
if all(c["__rv_esp"] is not None for c in cases):
    cases.sort(key=lambda c: c["__rv_esp"])
    print("Cases sorted by achieved RV_ESP:")
    for c in cases:
        print(f"  {c['__label']}: RV_ESP={c['__rv_esp']:.1f} mmHg")
else:
    print("Cases (no RV_ESP ordering):")
    for c in cases:
        print(f"  {c['__label']}")

# ── Output directory ─────────────────────────────────────────────────────────

if args.output_dir is None:
    if len(cases) == 1:
        out_dir = cases[0]["__dir"] / "transventricular_analysis"
    else:
        # Multi-case: put results in a top-level analysis dir
        out_dir = Path.cwd() / "results" / "analysis" / "transventricular"
else:
    out_dir = args.output_dir.resolve()
out_dir.mkdir(parents=True, exist_ok=True)
print(f"\nOutput directory: {out_dir}")

# ── Profile binning ──────────────────────────────────────────────────────────

def bin_profiles(data: dict, nbins: int) -> dict:
    """Bin study-region cells by tau and compute volume-weighted mean work."""
    study = data["study_region"]
    tau = data["tau"][study]
    vol = data["cell_volumes"][study]

    # Work arrays
    w_true = data["w_total"][study]
    w_ff = data["w_ff"][study]
    w_ss = data["w_ss"][study]
    w_nn = data["w_nn"][study]
    w_cross = data["w_cross"][study]

    # Proxy work (P * deps_ff, summed over time already)
    # As of 2026-04-08, pressures are saved in Pa so proxy work has same units as w_true.
    p_PLV_ff = data["proxy_PLV_ff"][study]
    p_PRV_ff = data["proxy_PRV_ff"][study]
    p_Trans_ff = data["proxy_Trans_ff"][study]
    p_PLV_ll = data["proxy_PLV_ll"][study]
    p_PRV_ll = data["proxy_PRV_ll"][study]
    p_Trans_ll = data["proxy_Trans_ll"][study]

    # Bin edges from min to max tau in study region
    bin_edges = np.linspace(tau.min(), tau.max(), nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Volume-weighted mean per bin
    def vol_mean(arr):
        result = np.zeros(nbins)
        for i in range(nbins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == nbins - 1:
                m = (tau >= lo) & (tau <= hi)
            else:
                m = (tau >= lo) & (tau < hi)
            if vol[m].sum() > 0:
                # Work density = sum(w) / sum(vol) (total work per unit volume in bin)
                result[i] = arr[m].sum() / vol[m].sum()
            else:
                result[i] = np.nan
        return result

    def bin_sum(arr):
        result = np.zeros(nbins)
        for i in range(nbins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == nbins - 1:
                m = (tau >= lo) & (tau <= hi)
            else:
                m = (tau >= lo) & (tau < hi)
            result[i] = arr[m].sum()
        return result

    # Count cells per bin with consistent edge semantics (matches vol_mean/bin_sum).
    # The inline expression previously had an operator-precedence bug that dropped
    # the lower bound for the last bin.
    n_per_bin = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == nbins - 1:
            m = (tau >= lo) & (tau <= hi)
        else:
            m = (tau >= lo) & (tau < hi)
        n_per_bin[i] = m.sum()

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "n_per_bin": n_per_bin,
        # Densities (work per unit volume)
        "w_true": vol_mean(w_true),
        "w_ff": vol_mean(w_ff),
        "w_ss": vol_mean(w_ss),
        "w_nn": vol_mean(w_nn),
        "w_cross": vol_mean(w_cross),
        "p_PLV_ff": vol_mean(p_PLV_ff),
        "p_PRV_ff": vol_mean(p_PRV_ff),
        "p_Trans_ff": vol_mean(p_Trans_ff),
        "p_PLV_ll": vol_mean(p_PLV_ll),
        "p_PRV_ll": vol_mean(p_PRV_ll),
        "p_Trans_ll": vol_mean(p_Trans_ll),
        # Totals (work integrated over volume in bin)
        "W_true_sum": bin_sum(w_true),
        "W_ff_sum": bin_sum(w_ff),
        "W_PLV_sum": bin_sum(p_PLV_ff),
        "W_PRV_sum": bin_sum(p_PRV_ff),
        "W_Trans_sum": bin_sum(p_Trans_ff),
    }


print("\nComputing profiles...")
for c in cases:
    c["__profile"] = bin_profiles(c, args.nbins)

# ── Figure 1: Transventricular profiles ─────────────────────────────────────

print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Transventricular Work Density Profiles\n(volume-weighted mean across tau bins)",
             fontsize=13)

n_cases = len(cases)
colors = plt.cm.viridis(np.linspace(0, 0.9, n_cases))

# Panel (0,0): True work density
ax = axes[0, 0]
for c, col in zip(cases, colors):
    p = c["__profile"]
    ax.plot(p["bin_centers"], p["w_true"], "-o", color=col, label=c["__label"], lw=1.5, ms=4)
ax.axhline(0, color="k", lw=0.5, alpha=0.5)
ax.set_xlabel("tau (LV→RV)")
ax.set_ylabel("Work density (Pa = J/m³)")
ax.set_title("True internal work S:dE")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.3)

# Panel (0,1): Fiber work (ff) component only
ax = axes[0, 1]
for c, col in zip(cases, colors):
    p = c["__profile"]
    ax.plot(p["bin_centers"], p["w_ff"], "-o", color=col, label=c["__label"], lw=1.5, ms=4)
ax.axhline(0, color="k", lw=0.5, alpha=0.5)
ax.set_xlabel("tau (LV→RV)")
ax.set_ylabel("Mean fiber work density")
ax.set_title("Fiber-direction work (w_ff)")
ax.grid(alpha=0.3)

# Panel (1,0): Proxy comparison for severe/representative case
# (show all proxies for the middle case)
ax = axes[1, 0]
mid_idx = len(cases) // 2
c = cases[mid_idx]
p = c["__profile"]
ax.plot(p["bin_centers"], p["w_true"], "k-o", label="True (S:dE)", lw=2, ms=5)
ax.plot(p["bin_centers"], p["p_PLV_ff"], "b--s", label="P_LV × dE_ff", lw=1.5, ms=4)
ax.plot(p["bin_centers"], p["p_PRV_ff"], "r--^", label="P_RV × dE_ff", lw=1.5, ms=4)
ax.plot(p["bin_centers"], p["p_Trans_ff"], "g--d", label="(P_LV-P_RV) × dE_ff", lw=1.5, ms=4)
ax.axhline(0, color="k", lw=0.5, alpha=0.5)
ax.set_xlabel("tau (LV→RV)")
ax.set_ylabel("Work density (arbitrary)")
ax.set_title(f"Proxy vs True work — {c['__label']}")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel (1,1): Proxy ratio R(tau) = w_proxy / w_true for Trans proxy across all cases
ax = axes[1, 1]
for c, col in zip(cases, colors):
    p = c["__profile"]
    # Avoid division by zero or small values
    w_true = p["w_true"]
    w_proxy = p["p_Trans_ff"]
    ratio = np.where(np.abs(w_true) > 1e-10, w_proxy / w_true, np.nan)
    ax.plot(p["bin_centers"], ratio, "-o", color=col, label=c["__label"], lw=1.5, ms=4)
ax.axhline(0, color="k", lw=0.5, alpha=0.5)
ax.axhline(1, color="k", lw=0.5, ls="--", alpha=0.5)
ax.set_xlabel("tau (LV→RV)")
ax.set_ylabel("R = w_proxy / w_true")
ax.set_title("Transmural proxy ratio: (P_LV-P_RV)×dE_ff / S:dE")
ax.legend(fontsize=8, loc="best")
ax.set_ylim(-2, 2)
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = out_dir / "profile_comparison.pdf"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {fig_path}")
plt.close(fig)

# ── Figure 2: Per-case proxy comparison (one panel per case) ─────────────────
# Dynamic grid: at most 4 columns, enough rows to fit all cases

n_cols = min(4, n_cases) if n_cases > 0 else 1
n_rows = (n_cases + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows),
                         sharex=True, sharey=True, squeeze=False)
fig.suptitle("Per-case proxy profiles — bin means in the study region", fontsize=12)
axes = axes.flatten()

for ax, c in zip(axes, cases):
    p = c["__profile"]
    ax.plot(p["bin_centers"], p["w_true"], "k-", label="True", lw=2)
    ax.plot(p["bin_centers"], p["p_PLV_ff"], "b--", label="P_LV × dE_ff", lw=1.2)
    ax.plot(p["bin_centers"], p["p_PRV_ff"], "r--", label="P_RV × dE_ff", lw=1.2)
    ax.plot(p["bin_centers"], p["p_Trans_ff"], "g--", label="Trans × dE_ff", lw=1.2)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    title = c['__label']
    if c.get("__rv_esp"):
        title += f" (RV={c['__rv_esp']:.0f})"
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlabel("tau")
    if ax is axes[0]:
        ax.legend(fontsize=8, loc="best")

# Hide unused subplots
for ax in axes[len(cases):]:
    ax.set_visible(False)

plt.tight_layout()
fig_path = out_dir / "profile_per_case.pdf"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {fig_path}")
plt.close(fig)

# ── Figure 3: Sensitivity curve ──────────────────────────────────────────────
#
# Sweep symmetric tau cutoffs: define septum = cells with tau ∈ [tc, 1-tc]
# For each cutoff, integrate true and proxy work over the cells in that range,
# compute R = W_proxy / W_true. As tc shrinks, more cells are included.

def sensitivity_curve(data: dict, n_cutoffs: int = 30) -> dict:
    study = data["study_region"]
    tau = data["tau"][study]
    w_true = data["w_total"][study]
    w_ff = data["w_ff"][study]
    p_PLV_ff = data["proxy_PLV_ff"][study]
    p_PRV_ff = data["proxy_PRV_ff"][study]
    p_Trans_ff = data["proxy_Trans_ff"][study]

    tau_min = tau.min()
    # Sweep from tight (near 0.5) to wide (tau_min)
    # Center-to-edge: tc goes from 0.50 down to tau_min
    cutoffs = np.linspace(tau_min, 0.49, n_cutoffs)

    R_PLV = np.full(n_cutoffs, np.nan)
    R_PRV = np.full(n_cutoffs, np.nan)
    R_Trans = np.full(n_cutoffs, np.nan)
    R_PLV_vs_ff = np.full(n_cutoffs, np.nan)
    R_PRV_vs_ff = np.full(n_cutoffs, np.nan)
    R_Trans_vs_ff = np.full(n_cutoffs, np.nan)
    n_cells = np.zeros(n_cutoffs, dtype=int)
    W_true_vals = np.full(n_cutoffs, np.nan)

    for i, tc in enumerate(cutoffs):
        m = (tau >= tc) & (tau <= 1 - tc)
        n_cells[i] = m.sum()
        if n_cells[i] == 0:
            continue
        W_true = w_true[m].sum()
        W_ff = w_ff[m].sum()
        W_PLV = p_PLV_ff[m].sum()
        W_PRV = p_PRV_ff[m].sum()
        W_Trans = p_Trans_ff[m].sum()
        W_true_vals[i] = W_true
        if abs(W_true) > 1e-15:
            R_PLV[i] = W_PLV / W_true
            R_PRV[i] = W_PRV / W_true
            R_Trans[i] = W_Trans / W_true
        if abs(W_ff) > 1e-15:
            R_PLV_vs_ff[i] = W_PLV / W_ff
            R_PRV_vs_ff[i] = W_PRV / W_ff
            R_Trans_vs_ff[i] = W_Trans / W_ff

    # Reference points: map septum definitions to tau cutoffs
    # Geometric septum bound: narrowest symmetric range that contains all geometric cells
    is_geo = data["is_geometric_septum"]
    is_ldrb = data["is_ldrb_septum"]
    tau_all = data["tau"]

    if is_geo.sum() > 0:
        tau_geo = tau_all[is_geo]
        geo_tc = max(0.0, min(tau_geo.min(), 1 - tau_geo.max()))
    else:
        geo_tc = None
    if is_ldrb.sum() > 0:
        tau_ldrb = tau_all[is_ldrb]
        ldrb_tc = max(0.0, min(tau_ldrb.min(), 1 - tau_ldrb.max()))
    else:
        ldrb_tc = None

    return {
        "cutoffs": cutoffs,
        "n_cells": n_cells,
        "R_PLV": R_PLV,
        "R_PRV": R_PRV,
        "R_Trans": R_Trans,
        "R_PLV_vs_ff": R_PLV_vs_ff,
        "R_PRV_vs_ff": R_PRV_vs_ff,
        "R_Trans_vs_ff": R_Trans_vs_ff,
        "W_true": W_true_vals,
        "geo_tc": geo_tc,
        "ldrb_tc": ldrb_tc,
    }


print("\nComputing sensitivity curves...")
for c in cases:
    c["__sens"] = sensitivity_curve(c)

# Plot sensitivity curves — one figure with 3 panels (one per proxy), all cases overlaid
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Sensitivity Curves: Proxy Ratio vs Septum Boundary Width", fontsize=12)

proxy_names = [("R_PLV", "P_LV × dE_ff"),
               ("R_PRV", "P_RV × dE_ff"),
               ("R_Trans", "(P_LV - P_RV) × dE_ff")]

for ax, (key, title) in zip(axes, proxy_names):
    for c, col in zip(cases, colors):
        s = c["__sens"]
        ax.plot(s["cutoffs"], s[key], "-", color=col, label=c["__label"], lw=1.5)
    # Mark reference points — use first case's values (they should all be similar since same mesh)
    s0 = cases[0]["__sens"]
    if s0["geo_tc"] is not None:
        ax.axvline(s0["geo_tc"], color="blue", ls=":", lw=1.5, alpha=0.7)
        ax.text(s0["geo_tc"], ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else -0.9,
                " Geometric", fontsize=8, color="blue", rotation=90, va="top")
    if s0["ldrb_tc"] is not None:
        ax.axvline(s0["ldrb_tc"], color="red", ls=":", lw=1.5, alpha=0.7)
        ax.text(s0["ldrb_tc"], ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else -0.9,
                " LDRB", fontsize=8, color="red", rotation=90, va="top")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.axhline(1, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("tau cutoff (septum = [tc, 1-tc])")
    ax.set_title(f"R = W_proxy / W_true\n{title}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-3, 3)
    if ax is axes[0]:
        ax.set_ylabel("Proxy ratio R")
        ax.legend(fontsize=8, loc="best")

plt.tight_layout()
fig_path = out_dir / "sensitivity_curves.pdf"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {fig_path}")
plt.close(fig)

# ── Figure 4: Alternative denominator — R vs w_ff instead of w_true ──────────
# Separates "wrong pressure" from "missing directions" effects

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle("Proxy Ratios with Different Denominators\nR_true = W_proxy/W_true (total), R_ff = W_proxy/W_ff (fiber only)",
             fontsize=11)

for c, col in zip(cases, colors):
    s = c["__sens"]
    axes[0].plot(s["cutoffs"], s["R_Trans"], "-", color=col, label=c["__label"], lw=1.5)
    axes[1].plot(s["cutoffs"], s["R_Trans_vs_ff"], "-", color=col, label=c["__label"], lw=1.5)

for ax, title in zip(axes, ["R = W_Trans / W_true (total)", "R = W_Trans / W_ff (fiber only)"]):
    ax.axhline(1, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    s0 = cases[0]["__sens"]
    if s0["geo_tc"] is not None:
        ax.axvline(s0["geo_tc"], color="blue", ls=":", lw=1.5, alpha=0.7)
    if s0["ldrb_tc"] is not None:
        ax.axvline(s0["ldrb_tc"], color="red", ls=":", lw=1.5, alpha=0.7)
    ax.set_xlabel("tau cutoff")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-3, 3)

axes[0].set_ylabel("Proxy ratio")
axes[0].legend(fontsize=8, loc="best")

plt.tight_layout()
fig_path = out_dir / "sensitivity_denom_comparison.pdf"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {fig_path}")
plt.close(fig)

# ── Summary table (CSV + JSON) ───────────────────────────────────────────────

print("\nWriting summary table...")
summary_rows = []
for c in cases:
    s = c["__sens"]
    # Extract proxy ratios at the geometric and LDRB reference points
    def interp_at(tc, arr):
        if tc is None:
            return None
        return float(np.interp(tc, s["cutoffs"], arr))

    row = {
        "case": c["__label"],
        "rv_esp_achieved": c.get("__rv_esp"),
        "n_study_cells": int(c["study_region"].sum()),
        "n_geometric": int(c["is_geometric_septum"].sum()),
        "n_ldrb": int(c["is_ldrb_septum"].sum()),
        "tau_range_study": [float(c["tau"][c["study_region"]].min()),
                            float(c["tau"][c["study_region"]].max())],
        # At geometric boundary
        "R_PLV_at_geometric": interp_at(s["geo_tc"], s["R_PLV"]),
        "R_PRV_at_geometric": interp_at(s["geo_tc"], s["R_PRV"]),
        "R_Trans_at_geometric": interp_at(s["geo_tc"], s["R_Trans"]),
        # At LDRB boundary
        "R_PLV_at_ldrb": interp_at(s["ldrb_tc"], s["R_PLV"]),
        "R_PRV_at_ldrb": interp_at(s["ldrb_tc"], s["R_PRV"]),
        "R_Trans_at_ldrb": interp_at(s["ldrb_tc"], s["R_Trans"]),
        # Range across the whole sweep (uncertainty from segmentation)
        "R_Trans_min": float(np.nanmin(s["R_Trans"])),
        "R_Trans_max": float(np.nanmax(s["R_Trans"])),
    }
    summary_rows.append(row)

# Write JSON
json_path = out_dir / "summary.json"
with open(json_path, "w") as f:
    json.dump(summary_rows, f, indent=2, default=float)
print(f"  Saved: {json_path}")

# Write CSV
csv_path = out_dir / "summary.csv"
csv_fields = [
    "case", "rv_esp_achieved", "n_study_cells", "n_geometric", "n_ldrb",
    "R_PLV_at_geometric", "R_PRV_at_geometric", "R_Trans_at_geometric",
    "R_PLV_at_ldrb", "R_PRV_at_ldrb", "R_Trans_at_ldrb",
    "R_Trans_min", "R_Trans_max",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    for r in summary_rows:
        writer.writerow({k: r.get(k) if r.get(k) is not None else "" for k in csv_fields})
print(f"  Saved: {csv_path}")

def fmt(v, spec):
    """Format a value, returning '-' for None or NaN."""
    if v is None:
        return "-"
    try:
        if np.isnan(v):
            return "-"
    except (TypeError, ValueError):
        pass
    return format(v, spec)

# Print summary to console
print("\n=== SUMMARY ===")
print(f"{'Case':<18} {'RV_ESP':>8} {'n_study':>8} {'R_PLV@geo':>11} {'R_Trans@geo':>12} {'R_Trans@ldrb':>13}")
for r in summary_rows:
    print(f"{r['case']:<18} "
          f"{fmt(r.get('rv_esp_achieved'), '.0f'):>8} "
          f"{r['n_study_cells']:>8} "
          f"{fmt(r.get('R_PLV_at_geometric'), '.3f'):>11} "
          f"{fmt(r.get('R_Trans_at_geometric'), '.3f'):>12} "
          f"{fmt(r.get('R_Trans_at_ldrb'), '.3f'):>13}")

print(f"\nDone. All outputs in {out_dir}")
