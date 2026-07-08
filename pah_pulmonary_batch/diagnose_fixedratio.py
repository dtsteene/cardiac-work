#!/usr/bin/env python3
"""Fixed-ratio sweep diagnostics: dynamic range, monotonicity, and loop totals.

Answers three questions on the new fixed-ratio sweep, compared to the old per-case
(volume-capped) sweep, for full LDRB regions and the AHA mid ring:

  1. Is true work dynamic only in the RV? (range as % of sweep-mean per region)
  2. Is *everything* monotonic in afterload? If true work AND the proxy are both
     monotonic across the 8 cases, Pearson r is ~±1 by construction — so a high |r|
     does NOT mean the proxy is a good quantitative tracker. We report Spearman |rho|
     (=1 iff strictly monotonic) and the count of sign flips in consecutive diffs
     alongside Pearson r to expose this.
  3. Loop totals (∮S:dE true work, ∮P dε proxy) per region — magnitude + range.

Run:  python3 pah_pulmonary_batch/diagnose_fixedratio.py
"""
import os, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

CASES = ["case0_rv25","case1_rv35","case2_rv45","case3_rv55",
         "case4_rv65","case5_rv75","case6_rv85","case7_rv95"]
BUNDLES = ["no_frank_starling", "frank_starling_preload", "frank_starling_relax"]
REG = {"LV": 1, "RV": 2, "Septum": 3}
AHA_MID = (4, 5, 6)
NB = 6
SEV = "peak RV systolic pressure (mmHg)"
SWEEPS = {
    "old (per-case ratio)": paths.RESULTS_ROOT / "sims/2026-06-09/pah_pulmonary_20260609_prodsweep",
    "new (fixed ratio)":    paths.RESULTS_ROOT / "sims/2026-06-22/pah_pulmonary_fixedratio",
}


def rv_systolic(cd):
    p = np.load(cd / "solver/solver_cavity_pressure_mmHg.npy").astype(float)
    n = len(p); return float(p[:, 1][n - n // NB:].max())


def load_aha(cd, ncells):
    side = cd / "aha_tags.npy"
    if side.exists():
        a = np.load(side).astype(np.int32)
        if len(a) == ncells:
            return a
    return None


def aggregate(sweep, bundle, band="full"):
    """Per-region true work total + ll proxies, per case (severity attached)."""
    rows = []
    for c in CASES:
        cd = sweep / bundle / c
        z = np.load(cd / "per_cell_data.npz", allow_pickle=True)
        tags = z["region_tags"]
        aha = load_aha(cd, len(tags))
        mid = np.isin(aha, AHA_MID) if (band == "mid" and aha is not None) else np.ones(len(tags), bool)
        row = {"sev": rv_systolic(cd)}
        for r, tag in REG.items():
            m = (tags == tag) & mid
            row[f"{r}_W"] = -float(z["w_total"][m].sum())
            for pk in ("PLV", "PRV", "Trans", "Mean", "Sum"):
                row[f"{r}_{pk}"] = -float(z[f"proxy_{pk}_ll"][m].sum())
        rows.append(row)
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], float) for k in keys}


def monotonic_diag(x, y):
    """Return (pearson_r, spearman_rho, n_sign_flips) for y ordered by x."""
    o = np.argsort(x); ys = np.asarray(y)[o]
    if np.std(ys) == 0:
        return float("nan"), float("nan"), 0
    pear = float(np.corrcoef(x, y)[0, 1])
    rho = float(spearmanr(x, y).correlation)
    d = np.diff(ys)
    s = np.sign(d[d != 0])
    flips = int((np.diff(s) != 0).sum()) if len(s) > 1 else 0
    return pear, rho, flips


def rng(a):
    a = np.asarray(a, float); return 100 * (a.max() - a.min()) / abs(a.mean())


def supervisor_figure(out_dir, bundle="no_frank_starling"):
    """One figure for the supervisor making the methodological point:
    (a) only the RV has true-work dynamic range; LV/septum are flat;
    (b) on the RV every monotonic pressure proxy collapses onto the true-work trend,
        so Pearson r is ~+1 for all of them — correlation cannot pick a winner; only
        the NON-monotonic transmural pressure breaks the pattern;
    (c) on the (flat) septum the proxies still swing with rising pressure while true
        work barely moves — a high proxy r there is the monotonic confound, not tracking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sweep = SWEEPS["new (fixed ratio)"]
    df = aggregate(sweep, bundle, "full")
    x = df["sev"]; o = np.argsort(x); xs = x[o]
    def nrm(a):
        a = np.asarray(a, float)[o]; return (a - a.min()) / (a.max() - a.min() + 1e-30)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    # (a) dynamic range
    regs = list(REG)
    ranges = [rng(df[f"{r}_W"]) for r in regs]
    ax[0].bar(regs, ranges, color=["#4575b4", "#d73027", "#762a83"], ec="k", lw=0.5)
    for i, v in enumerate(ranges):
        ax[0].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax[0].set_ylabel("true-work range (% of sweep-mean)")
    ax[0].set_title("(a) only the RV has dynamic range", fontweight="bold")
    ax[0].grid(alpha=0.25, axis="y")
    # (b) RV monotonic collapse
    cols = {"W": "#222", "PLV": "#4575b4", "PRV": "#d73027", "Mean": "#1b7837", "Trans": "#e6ab02"}
    for k, lab in [("W", "true work"), ("PLV", "P_LV·ε"), ("PRV", "P_RV·ε"),
                   ("Mean", "Mean·ε"), ("Trans", "(P_LV−P_RV)·ε")]:
        key = "RV_W" if k == "W" else f"RV_{k}"
        r = np.corrcoef(x, df[key])[0, 1]
        ax[1].plot(xs, nrm(df[key]), "-o", color=cols[k], lw=2.2 if k == "W" else 1.6,
                   ms=5 if k == "W" else 4, label=f"{lab}  (r={r:+.2f})")
    ax[1].set_xlabel(SEV); ax[1].set_ylabel("normalised to [0,1]")
    ax[1].set_title("(b) RV: every monotonic proxy → r≈+1", fontweight="bold")
    ax[1].legend(frameon=False, fontsize=8.5); ax[1].grid(alpha=0.25)
    # (c) septum flat truth vs swinging proxy
    axc = ax[2]
    axc.plot(xs, df["Septum_W"][o], "-o", color="#222", lw=2.4, ms=5,
             label=f"true work (range {rng(df['Septum_W']):.0f}%)")
    axc.set_ylabel("true SS work (a.u.)")
    axr = axc.twinx()
    axr.plot(xs, df["Septum_PRV"][o], "--s", color="#d73027", lw=2.0, ms=4,
             label=f"P_RV·ε (range {rng(df['Septum_PRV']):.0f}%)")
    axr.set_ylabel("proxy work (a.u.)"); axr.spines["top"].set_visible(False)
    axc.set_xlabel(SEV); axc.set_title("(c) septum: truth flat, proxy swings", fontweight="bold")
    axc.grid(alpha=0.25)
    h1, l1 = axc.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    axc.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(f"Why correlation is degenerate on this monotonic sweep — {bundle} (fixed ratio)",
                 fontsize=13, fontweight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_dir / f"monotonic_confound_{bundle}.png"), dpi=170, bbox_inches="tight")
    fig.savefig(str(out_dir / f"monotonic_confound_{bundle}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote monotonic_confound_{bundle}.png")


def main():
    for band in ("full", "mid"):
        print("\n" + "=" * 78)
        print(f"BAND = {band.upper()}")
        print("=" * 78)
        for bundle in BUNDLES:
            print(f"\n--- {bundle} ---")
            for label, sweep in SWEEPS.items():
                try:
                    df = aggregate(sweep, bundle, band)
                except FileNotFoundError:
                    print(f"  {label}: (no AHA tags / missing) — skipped")
                    continue
                x = df["sev"]
                print(f"  {label}:  RV sys {x.min():.0f}-{x.max():.0f} mmHg")
                for r in REG:
                    W = df[f"{r}_W"]
                    pr, rho, flips = monotonic_diag(x, W)
                    # best non-transmural proxy pearson for reference
                    prox = {pk: monotonic_diag(x, df[f"{r}_{pk}"])[0] for pk in
                            ("PLV", "PRV", "Trans", "Mean", "Sum")}
                    mono = "MONOTONIC" if flips == 0 else f"{flips} flip(s)"
                    print(f"     {r:7s} true-work range {rng(W):4.0f}%  "
                          f"|rho|={abs(rho):.2f} ({mono})  "
                          f"pearson r(W,P): PLV{prox['PLV']:+.2f} PRV{prox['PRV']:+.2f} "
                          f"Trans{prox['Trans']:+.2f} Mean{prox['Mean']:+.2f}")


if __name__ == "__main__":
    main()
    out = paths.RESULTS_ROOT / "handover/pah_pulmonary_fixedratio_20260622/supervisor"
    for b in BUNDLES:
        supervisor_figure(out, b)
