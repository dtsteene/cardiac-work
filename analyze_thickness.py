#!/usr/bin/env python3
"""
analyze_thickness.py — Thickness variant proxy validation

Loads per_cell_data.npz from thickness variant simulations and computes
Pearson r across the 6 thickness variants, for each severity level. Produces
a summary figure showing whether the proxy ranking (Trans > PLV > PRV) is
invariant under wall thickness variation.

Expects a 2×6 matrix of simulations:
    severities: {healthy, severe}
    thickness:  {global_1mm, global_2mm, rvfw_2mm, rvfw_3mm, rvfw_5mm, rvfw_7mm}

Each sim is identified via its run_description.txt containing "Phase4 thickness".

Usage:
    python3 analyze_thickness.py [--output-dir results/analysis/thickness]
    python3 analyze_thickness.py --results-dir results/sims/2026-04-13

Auto-discovers thickness sim directories if not given.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=Path, default=None,
                    help="Dated results dir (e.g. results/sims/2026-04-13). "
                         "If omitted, auto-discovers the most recent dir with "
                         "thick_*/Phase4 thickness runs.")
parser.add_argument("--output-dir", type=Path,
                    default=Path("results/analysis/thickness"))
args = parser.parse_args()

out_dir = args.output_dir
out_dir.mkdir(parents=True, exist_ok=True)

# ── Discover thickness sim directories ───────────────────────────────────────
if args.results_dir:
    search_dirs = [args.results_dir]
else:
    # Find recent results/sims/<date>/ directories
    search_dirs = sorted(Path("results/sims").glob("2026-*"))
    if not search_dirs:
        print("ERROR: no results/sims/2026-* directories found")
        raise SystemExit(1)

THICKNESS_VARIANTS = [
    # v1 (thickness_cl10, saturated-warp): 1–3 mm effective range
    "global_1mm", "global_2mm", "rvfw_2mm", "rvfw_3mm", "rvfw_5mm", "rvfw_7mm",
    # v2 (thickness_v2, epi-outward warp): 3–12 mm effective range
    "rvfw_03mm", "rvfw_06mm", "rvfw_09mm", "rvfw_12mm",
]
SEVERITIES = ["healthy", "severe"]  # extend to moderate etc. if needed

# Map (severity, variant) → sim directory
sims = {}  # (severity, variant) -> Path
for base_dir in search_dirs:
    for sim_dir in sorted(base_dir.glob("UKB_*beats_run_*")):
        desc_path = sim_dir / "run_description.txt"
        if not desc_path.exists():
            continue
        desc = desc_path.read_text().strip()
        if "Phase4 thickness" not in desc and "Phase4v2 thickness" not in desc:
            continue
        # Parse: "Phase4 thickness global_1mm severe 6beats"
        parts = desc.split()
        try:
            variant = next(p for p in parts if p in THICKNESS_VARIANTS)
            severity = next(p for p in parts if p in SEVERITIES)
        except StopIteration:
            continue
        sims[(severity, variant)] = sim_dir

print(f"Discovered {len(sims)} thickness sim directories")
for key in sorted(sims):
    print(f"  {key[0]:<8} {key[1]:<12} → {sims[key].name}")

if len(sims) < 12:
    missing = [(s, v) for s in SEVERITIES for v in THICKNESS_VARIANTS
               if (s, v) not in sims]
    print(f"\nWARNING: missing {len(missing)} sims:")
    for m in missing:
        print(f"  {m}")

# ── Load per-cell data ───────────────────────────────────────────────────────
def safe_r(x, y):
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 3:
        return np.nan
    return pearsonr(x, y)[0]

def load_case(sim_dir):
    # Prefer the last-beat per-cell file (produced by ALL_BEATS=1 runs).
    # Fall back to the legacy single per_cell_data.npz if present.
    for name in ("per_cell_data_beat5.npz", "per_cell_data.npz"):
        pc_path = sim_dir / name
        if pc_path.exists():
            return np.load(pc_path, allow_pickle=True)
    return None


def compute_region_sums(pc, region_key):
    """Compute all proxy region sums for a single case.
    Returns a dict with W_true, W_PLV, W_PRV, W_Trans, W_mean, W_dom.
    """
    mask = pc[region_key]
    plv_cell = pc["proxy_PLV_ll"]
    prv_cell = pc["proxy_PRV_ll"]
    tau = pc["tau"]
    W_PLV = plv_cell[mask].sum()
    W_PRV = prv_cell[mask].sum()
    # W_dom: per-cell dominant cavity rule
    tau_lv = mask & (tau < 0.5)
    tau_rv = mask & (tau >= 0.5)
    W_dom = plv_cell[tau_lv].sum() + prv_cell[tau_rv].sum()
    return {
        "W_true": pc["w_total"][mask].sum(),
        "W_PLV": W_PLV,
        "W_PRV": W_PRV,
        "W_Trans": W_PLV - W_PRV,  # identical to proxy_Trans_ll.sum() by construction
        "W_mean": 0.5 * (W_PLV + W_PRV),
        "W_dom": W_dom,
    }


def optimal_angle(W_plv_arr, W_prv_arr, W_true_arr):
    """Find θ maximizing r(cos(θ)*W_PLV + sin(θ)*W_PRV, W_true).

    Grid-searches the unit circle at 0.1° resolution. Robust to any
    correlation structure of the inputs (an earlier "closed-form" version
    maximized the numerator of r rather than r itself and under-reported
    the optimum whenever W_PLV and W_PRV were correlated).
    """
    thetas = np.linspace(-np.pi, np.pi, 3601)
    best_theta, best_r = 0.0, -np.inf
    for th in thetas:
        W = np.cos(th) * W_plv_arr + np.sin(th) * W_prv_arr
        r = safe_r(W, W_true_arr)
        if not np.isnan(r) and r > best_r:
            best_r = r
            best_theta = th
    return float(best_theta), float(best_r)

# ── Compute per-(severity, region_def) sums and correlations ─────────────────
# For each (severity, region_def) row, collect W_true + all five candidate
# proxies across the 6 thickness variants, then compute Pearson r for each
# canonical proxy (PLV, PRV, Trans, Mean, Dom) plus the closed-form optimal
# linear blend of PLV and PRV.

PROXIES = ["PLV", "PRV", "Trans", "Mean", "Dom"]
REGION_DEFS = ["geometric", "ldrb"]
MASK_KEY = {"geometric": "is_geometric_septum", "ldrb": "is_ldrb_septum"}

results = {}  # (severity, region_def) -> dict of arrays + r values

for severity in SEVERITIES:
    for region_def in REGION_DEFS:
        rows = {k: [] for k in ["W_true", "W_PLV", "W_PRV", "W_Trans",
                                 "W_mean", "W_dom"]}
        labels = []
        for variant in THICKNESS_VARIANTS:
            key = (severity, variant)
            if key not in sims:
                continue
            pc = load_case(sims[key])
            if pc is None:
                print(f"  SKIP {key}: no per_cell_data.npz")
                continue
            sums = compute_region_sums(pc, MASK_KEY[region_def])
            for k, v in sums.items():
                rows[k].append(float(v))
            labels.append(variant)

        if len(labels) < 2:
            continue

        arrs = {k: np.array(v) for k, v in rows.items()}
        W_true = arrs["W_true"]
        r_values = {
            "r_PLV":   safe_r(W_true, arrs["W_PLV"]),
            "r_PRV":   safe_r(W_true, arrs["W_PRV"]),
            "r_Trans": safe_r(W_true, arrs["W_Trans"]),
            "r_Mean":  safe_r(W_true, arrs["W_mean"]),
            "r_Dom":   safe_r(W_true, arrs["W_dom"]),
        }
        theta_star, r_opt = optimal_angle(arrs["W_PLV"], arrs["W_PRV"], W_true)
        r_values["r_Opt"] = r_opt
        r_values["theta_star_deg"] = float(np.degrees(theta_star))

        results[(severity, region_def)] = {
            "labels": labels,
            **arrs,
            **r_values,
        }

# ── Console summary ──────────────────────────────────────────────────────────
print()
print("=" * 100)
print("THICKNESS-VARIATION PROXY CORRELATIONS (Pearson r across 6 wall-thickness variants)")
print("=" * 100)
print(f"\n{'severity':<10} {'region':<10} {'n':>3}  "
      f"{'r_PLV':>8} {'r_PRV':>8} {'r_Trans':>8} {'r_Mean':>8} {'r_Dom':>8} "
      f"{'r_Opt':>8}  {'θ*':>8}  winner")
print("-" * 100)
for severity in SEVERITIES:
    for region_def in REGION_DEFS:
        r = results.get((severity, region_def))
        if r is None:
            continue
        cand = {k: r[k] for k in ["r_PLV", "r_PRV", "r_Trans", "r_Mean", "r_Dom"]}
        winner = max(cand.items(), key=lambda kv: abs(kv[1]))[0].replace("r_", "")
        print(f"{severity:<10} {region_def:<10} {len(r['labels']):>3}  "
              f"{r['r_PLV']:>+8.4f} {r['r_PRV']:>+8.4f} {r['r_Trans']:>+8.4f} "
              f"{r['r_Mean']:>+8.4f} {r['r_Dom']:>+8.4f} {r['r_Opt']:>+8.4f}  "
              f"{r['theta_star_deg']:>+6.1f}°  {winner}")

# ── Figure 1: Core visual — z-scored tracking across thickness variants ──────
# Plotting z-scores makes Pearson r visually obvious: a proxy with r≈1 overlays
# W_true exactly, and the geometric distance from W_true at each x is literally
# the residual that drives r.
def zscore(a):
    a = np.asarray(a, dtype=float)
    s = np.std(a)
    return (a - np.mean(a)) / (s if s > 0 else 1.0)

ROW_LABEL = {"healthy": "Healthy circulation", "severe": "Severe PAH"}
VARIANT_LABEL = {
    # v1 saturated-warp
    "global_1mm":  "global\n1 mm",
    "global_2mm":  "global\n2 mm",
    "rvfw_2mm":    "RVfw\n2 mm",
    "rvfw_3mm":    "RVfw\n3 mm",
    "rvfw_5mm":    "RVfw\n5 mm",
    "rvfw_7mm":    "RVfw\n7 mm",
    # v2 epi-outward warp
    "rvfw_03mm":   "RVfw\n+3 mm",
    "rvfw_06mm":   "RVfw\n+6 mm",
    "rvfw_09mm":   "RVfw\n+9 mm",
    "rvfw_12mm":   "RVfw\n+12 mm",
}
PROXY_STYLE = {
    "W_PLV":   dict(color="#1f77b4", marker="o", label="$P_{LV}$"),
    "W_PRV":   dict(color="#d62728", marker="s", label="$P_{RV}$"),
    "W_Trans": dict(color="#2ca02c", marker="^", label="$P_{LV}-P_{RV}$"),
}

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True,
                          sharey=True, constrained_layout=True)
for i, severity in enumerate(SEVERITIES):
    for j, region_def in enumerate(REGION_DEFS):
        ax = axes[i, j]
        r = results.get((severity, region_def))
        if r is None:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_axis_off()
            continue
        x = np.arange(len(r["labels"]))
        ax.axhline(0.0, color="lightgray", lw=0.8, zorder=0)
        ax.plot(x, zscore(r["W_true"]), "-", color="black", lw=2.8,
                marker="D", ms=7, label="$W_{true}$", zorder=5)
        for key, style in PROXY_STYLE.items():
            r_key = "r_" + key.replace("W_", "")
            ax.plot(x, zscore(r[key]), "-", lw=1.5, ms=6, alpha=0.9,
                    label=f"{style['label']} (r={r[r_key]:+.2f})",
                    color=style["color"], marker=style["marker"])
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABEL[v] for v in r["labels"]], fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_title(f"{ROW_LABEL[severity]} — {region_def} septum",
                     fontsize=11, fontweight="bold")
        if j == 0:
            ax.set_ylabel("z-score")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9,
                  handlelength=1.5)

fig.suptitle("Proxy tracking across wall thickness",
             fontsize=12, fontweight="bold")
fig.savefig(out_dir / "fig_thickness_core.png", dpi=160, bbox_inches="tight")
fig.savefig(out_dir / "fig_thickness_core.pdf", bbox_inches="tight")
print(f"\nSaved {out_dir / 'fig_thickness_core.png'}")

# ── Figure 2: Stats heatmap — 3 canonical proxies only ──────────────────────
# r_Mean, r_Dom, and r_Opt are deliberately omitted from the heatmap:
# r_Opt is tautological (sup over θ ⇒ always ≥ the canonicals), r_Mean and
# r_Dom are minor alternatives that aren't part of the headline message.
# The optimal angle is a different quantity (how far from the transmural
# direction) and lives in Figure 3 on its own 1D axis.
configs = [(s, rd) for s in SEVERITIES for rd in REGION_DEFS
           if (s, rd) in results]
R_KEYS = ["r_PLV", "r_PRV", "r_Trans"]
R_LABELS = [r"$P_{LV}$", r"$P_{RV}$", r"$P_{LV}-P_{RV}$"]

mat = np.array([[results[c][k] for k in R_KEYS] for c in configs])
row_labels = [f"{ROW_LABEL[s]} / {rd}" for s, rd in configs]

fig2, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(R_LABELS)))
ax.set_xticklabels(R_LABELS, fontsize=11)
ax.set_yticks(range(len(configs)))
ax.set_yticklabels(row_labels, fontsize=10)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat[i, j]
        col = "white" if abs(val) > 0.55 else "black"
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                color=col, fontsize=11, fontweight="bold")
cbar = fig2.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Pearson r")
fig2.savefig(out_dir / "fig_thickness_stats.png", dpi=160, bbox_inches="tight")
fig2.savefig(out_dir / "fig_thickness_stats.pdf", bbox_inches="tight")
print(f"Saved {out_dir / 'fig_thickness_stats.png'}")

# ── Figure 3: Optimal-blend angle on a 1D axis ──────────────────────────────
# The optimal angle θ* is a different quantity than r_Opt and tells us:
# "what linear combination of P_LV and P_RV does the data actually pick?".
# Canonical reference angles:
#   0°   = pure P_LV
#   +90° = pure P_RV
#   -45° = transmural (P_LV − P_RV, normalized)
# A data-driven θ* close to −45° is direct evidence that the transmural
# pressure is the natural combination the data chooses, not something we
# imposed a priori.
CANON = [(0, r"$P_{LV}$", "#1f77b4"),
         (90, r"$P_{RV}$", "#d62728"),
         (-45, r"$P_{LV}-P_{RV}$", "#2ca02c")]

fig3, ax3 = plt.subplots(figsize=(9, 2.6), constrained_layout=True)
for ang, lab, col in CANON:
    ax3.axvline(ang, color=col, lw=2.0, alpha=0.6, zorder=1)
    ax3.text(ang, 1.05, lab, ha="center", va="bottom",
             color=col, fontsize=10, fontweight="bold")
y_positions = np.arange(len(configs))
for i, (s, rd) in enumerate(configs):
    theta = results[(s, rd)]["theta_star_deg"]
    ax3.plot(theta, i, "o", ms=12, color="black", zorder=5)
    ax3.text(theta, i, f"  {theta:+.0f}°",
              ha="left", va="center", fontsize=9)
ax3.set_yticks(y_positions)
ax3.set_yticklabels([f"{ROW_LABEL[s]} / {rd}" for s, rd in configs], fontsize=10)
ax3.set_xlim(-100, 100)
ax3.set_ylim(-0.5, len(configs) - 0.5 + 0.4)
ax3.set_xlabel("Optimal blend angle θ*   (cos(θ)·$P_{LV}$ + sin(θ)·$P_{RV}$)", fontsize=10)
ax3.grid(alpha=0.25, axis="x")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
fig3.savefig(out_dir / "fig_thickness_angle.png", dpi=160, bbox_inches="tight")
fig3.savefig(out_dir / "fig_thickness_angle.pdf", bbox_inches="tight")
print(f"Saved {out_dir / 'fig_thickness_angle.png'}")

# ── Save raw results for downstream ──────────────────────────────────────────
save_dict = {
    "severities": np.array(SEVERITIES),
    "variants":   np.array(THICKNESS_VARIANTS),
}
for (s, rd), r in results.items():
    for k, v in r.items():
        if k == "labels":
            continue
        save_dict[f"{s}_{rd}_{k}"] = v
np.savez(out_dir / "thickness_raw.npz", **save_dict)
print(f"Saved {out_dir / 'thickness_raw.npz'}")
print("\nDone.")
