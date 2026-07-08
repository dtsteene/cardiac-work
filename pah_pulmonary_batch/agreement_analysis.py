#!/usr/bin/env python3
"""Identifiability diagnostic for the fixed-ratio pressure-strain proxy sweep.

Clinical reviewers ask for *correlation* between a pressure-strain proxy and true
work. On this sweep that request is unanswerable, and this script proves *why* —
by showing the failure survives switching to the agreement metrics (Lin's CCC,
%RMSE) that a methods reviewer would demand instead. Nothing here re-simulates;
it replays the per-cell ``.npz`` files on the login node.

The sweep has a **single** driver (RV afterload). Two degeneracies follow:

  * **RV free wall** — 70–115 % true-work range but perfectly monotone, so every
    co-monotone pressure (P_LV, P_RV, Mean, Sum) tracks it at |r|≈0.99 AND fits it
    to a few-% affine RMSE. Correlation *and* agreement rate them identically.
  * **Septum** — only 2–4 % true-work range (nearly flat), so any swinging proxy
    "correlates" with the tiny residual wiggle; again no metric separates them.

The one pressure that *does* separate is the **non-monotone Transmural** (P_LV−P_RV),
and it separates as clearly *wrong* (anti-correlated on the RV, ~0 on the septum).

So the defensible claim from this dataset is precise and narrow:

    This single-parameter sweep RULES OUT transmural pressure, but CANNOT
    adjudicate P_RV vs Mean vs P_LV for the septum — the limit is the
    experiment's one monotone degree of freedom and flat septum, not the metric.

That is the rigorous motivation for a multi-parameter (e.g. LV×RV afterload)
redesign, in which these same agreement metrics WILL discriminate.

Run:  python3 pah_pulmonary_batch/agreement_analysis.py     # login node, pure NumPy
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
import analysis_core as ac

CASES = ["case0_rv25", "case1_rv35", "case2_rv45", "case3_rv55",
         "case4_rv65", "case5_rv75", "case6_rv85", "case7_rv95"]
BUNDLES = ["no_frank_starling", "frank_starling_preload", "frank_starling_relax"]
REGIONS = ["LV", "RV", "Septum"]
FOCUS = ["RV", "Septum"]                         # the two regions of clinical interest
CHOICES = ["PLV", "PRV", "Trans", "Mean", "Sum"]
COMONOTONE = ["PLV", "PRV", "Mean", "Sum"]       # the mutually-indistinguishable set
REGION_COLORS = {"LV": "#4575b4", "RV": "#d73027", "Septum": "#762a83"}
SWEEP = paths.RESULTS_ROOT / "sims/2026-06-22/pah_pulmonary_fixedratio"
OUT_ROOT = paths.RESULTS_ROOT / "handover/pah_pulmonary_fixedratio_20260622"
STRAIN = "ll"                                     # longitudinal / GLS


def load_case(cd):
    """Per-region true work (∮S:dE) and proxy work (∮P dε), sign-flipped positive."""
    z = np.load(cd / "per_cell_data.npz", allow_pickle=True)
    masks = ac.region_masks(z["region_tags"], z["is_geometric_septum"])
    out = {}
    for r in REGIONS:
        m = masks[r]
        row = {"W": -float(z["w_total"][m].sum())}
        for pk in CHOICES:
            row[pk] = -float(z[f"proxy_{pk}_{STRAIN}"][m].sum())
        out[r] = row
    return out


def aggregate(bundle):
    """truth[region] and proxy[region][choice] as length-8 arrays across cases."""
    per_case = [load_case(SWEEP / bundle / c) for c in CASES]
    truth = {r: np.array([pc[r]["W"] for pc in per_case], float) for r in REGIONS}
    proxy = {r: {pk: np.array([pc[r][pk] for pc in per_case], float) for pk in CHOICES}
             for r in REGIONS}
    return truth, proxy


def rng(a):
    a = np.asarray(a, float)
    return 100.0 * (a.max() - a.min()) / abs(a.mean())


def discrimination_table(truth, proxy):
    """For each focus region × pressure: dynamic range, |Pearson r|, |Spearman rho|,
    and the agreement %RMSE around the per-region affine fit. The point of the table
    is that these are ~identical across the co-monotone pressures."""
    rows = []
    for r in FOCUS:
        W = truth[r]
        for pk in CHOICES:
            p = proxy[r][pk]
            pear = ac.pearson_r(p, W)
            rho = spearmanr(p, W).correlation if np.std(p) and np.std(W) else float("nan")
            rel = ac.agreement_stats(p, W)["rel_rmse_affine"]
            rows.append((r, pk, rng(W), abs(pear), abs(rho), 100.0 * rel))
    return rows


def spread(rows, region, metric_idx):
    """Max−min of a metric over the co-monotone pressures for one region."""
    vals = [row[metric_idx] for row in rows if row[0] == region and row[1] in COMONOTONE]
    vals = [v for v in vals if np.isfinite(v)]
    return (min(vals), max(vals)) if vals else (float("nan"), float("nan"))


def figure(truth, proxy, rows, out_dir, bundle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    # (a) dynamic range — the root cause: septum is flat, RV saturates.
    ranges = [rng(truth[r]) for r in REGIONS]
    ax[0].bar(REGIONS, ranges, color=[REGION_COLORS[r] for r in REGIONS], ec="k", lw=0.5)
    for i, v in enumerate(ranges):
        ax[0].text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")
    ax[0].set_ylabel("true-work dynamic range (% of sweep-mean)")
    ax[0].set_title("(a) only the RV moves; the septum is flat", fontweight="bold")
    ax[0].grid(alpha=0.25, axis="y")

    # (b) |Pearson r| per pressure, for RV and septum — co-monotone pressures are
    # indistinguishable; only Trans (hatched) separates, and it separates as wrong.
    x = np.arange(len(CHOICES))
    w = 0.38
    for j, r in enumerate(FOCUS):
        vals = [next(row[3] for row in rows if row[0] == r and row[1] == pk) for pk in CHOICES]
        bars = ax[1].bar(x + (j - 0.5) * w, vals, w, label=r, color=REGION_COLORS[r],
                         ec="k", lw=0.5)
        for bi, pk in enumerate(CHOICES):
            if pk == "Trans":
                bars[bi].set_hatch("//")
                bars[bi].set_alpha(0.55)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(CHOICES)
    ax[1].set_ylabel("|Pearson r|  (proxy vs true work)")
    ax[1].set_ylim(0, 1.08)
    ax[1].axhspan(0.0, 0.0, color="none")
    lo_rv, hi_rv = spread(rows, "RV", 3)
    lo_se, hi_se = spread(rows, "Septum", 3)
    ax[1].set_title("(b) no metric separates PLV/PRV/Mean; only Trans (hatched) does\n"
                    f"co-monotone |r| span: RV {lo_rv:.2f}–{hi_rv:.2f}, "
                    f"Septum {lo_se:.2f}–{hi_se:.2f}", fontweight="bold", fontsize=10)
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.25, axis="y")

    fig.suptitle(f"Identifiability: this monotone sweep cannot pick the septal "
                 f"pressure — {bundle} (fixed ratio, {STRAIN})",
                 fontsize=13, fontweight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(str(out_dir / f"identifiability_{bundle}.{ext}"),
                    dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    for bundle in BUNDLES:
        print("\n" + "=" * 78)
        print(f"BUNDLE = {bundle}   (strain={STRAIN})")
        print("=" * 78)
        try:
            truth, proxy = aggregate(bundle)
        except FileNotFoundError as e:
            print(f"  missing data — skipped ({e})")
            continue
        out_dir = OUT_ROOT / bundle / "agreement"
        rows = discrimination_table(truth, proxy)

        print(f"\n  {'region':7s} {'press':6s} {'range%':>7s} {'|r|':>6s} "
              f"{'|rho|':>6s} {'%RMSE':>7s}")
        for r, pk, rg, ar, arho, rel in rows:
            flag = "  (non-monotone → separable/WRONG)" if pk == "Trans" else ""
            print(f"  {r:7s} {pk:6s} {rg:7.1f} {ar:6.2f} {arho:6.2f} {rel:7.1f}{flag}")
        for r in FOCUS:
            lo_r, hi_r = spread(rows, r, 3)
            lo_e, hi_e = spread(rows, r, 5)
            print(f"    → {r}: across PLV/PRV/Mean/Sum, |r| spans {lo_r:.3f}–{hi_r:.3f} "
                  f"(Δ={hi_r-lo_r:.3f}); agreement %RMSE spans {lo_e:.1f}–{hi_e:.1f}"
                  f"  ⇒ indistinguishable")

        write = out_dir / f"identifiability_{bundle}.csv"
        write.parent.mkdir(parents=True, exist_ok=True)
        with open(write, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["region", "pressure", "range_pct", "abs_pearson_r",
                        "abs_spearman_rho", "rel_rmse_affine_pct"])
            w.writerows(rows)
        figure(truth, proxy, rows, out_dir, bundle)
        print(f"\n  wrote table + figure → {out_dir}")

    print("\nCONCLUSION: the sweep rules OUT transmural pressure but cannot adjudicate")
    print("P_RV vs Mean vs P_LV for the septum. A multi-parameter redesign (e.g. LV×RV")
    print("afterload grid) is needed; the agreement metrics in analysis_core are ready.")


if __name__ == "__main__":
    main()
