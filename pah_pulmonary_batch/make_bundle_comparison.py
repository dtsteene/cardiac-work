#!/usr/bin/env python3
"""
Cross-bundle comparison: no-FS vs FS-preload vs FS-relaxation.

The three bundles differ ONLY in the active-tension law:
  no_frank_starling      constant Ta peak 100 kPa, no length feedback
  frank_starling_preload Ta peak 220 kPa scaled by a Frank-Starling gain from the
                         END-DIASTOLIC fiber stretch, frozen for the beat (preload_only)
  frank_starling_relax   same gain, applied dynamically with relaxation tau = 250 ms

The FS gain and the relaxation lag are applied PER CELL inside the solver, so the global
Ta_solver_history curve is byte-identical for preload and relax (both 220 kPa). The realized,
FS-modulated, relaxation-lagged tension is only visible in the regional fibre stress S_ff(t).

All pressures are the REALIZED solver cavity pressures (solver_cavity_pressure_mmHg.npy),
never the 0D values. Volumes are the coupled realized chamber volumes.

Story order:  (1) PV loops qualitatively  ->  (2) hemodynamics heatmap quantitatively
              ->  (3) fibre stress over the beat  ->  (4) regional energetics heatmap
              ->  (5) septum proxy robustness.

Output: <RESULTS_ROOT>/handover/pah_pulmonary_paper_20260611/comparison/

Run:  python3 pah_pulmonary_batch/make_bundle_comparison.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pah_handover as H

REGS = ["LV", "RV", "Septum"]
# bundle display: short label, colour, line style
BUN = [
    ("no_frank_starling",      "no FS (Ta=100)",                "#555555", "-"),
    ("frank_starling_preload", "FS preload (Ta=220)",           "#1f78b4", "-"),
    ("frank_starling_relax",   "FS relax (Ta=220, tau=250 ms)", "#e31a1c", "--"),
]
CH = [("LV", "V_LV", 0), ("RV", "V_RV", 1)]   # chamber, volume key, realized-pressure column


def beat(arr):
    a = np.asarray(arr); return a[H.last_beat_slice(len(a))]


def loop_area(V, P):
    """Enclosed PV-loop area (shoelace) = stroke work, mmHg*mL."""
    V = np.asarray(V); P = np.asarray(P)
    return 0.5 * abs(np.dot(V, np.roll(P, -1)) - np.dot(P, np.roll(V, -1)))


def _rv_loop(o, sp, vk, pcol):
    n = sp.shape[0]; lb = H.last_beat_slice(n)
    mn = min(len(o[vk]), n); lo = H.last_beat_slice(mn)
    V = np.asarray(o[vk])[:mn][lo]; P = sp[:, pcol][lb][:len(V)]
    return V, P


# --------------------------------------------------------------------------- #
def _delta_heatmap(rows, base, pre, rel, out, name, title, figw=6.4, rowlab=None):
    """Generic Delta% (vs no-FS) heatmap with FS-preload | FS-relax columns."""
    M = np.array([[100 * (pre[k] - base[k]) / abs(base[k]),
                   100 * (rel[k] - base[k]) / abs(base[k])] for k in rows])
    fig, ax = plt.subplots(figsize=(figw, 0.5 * len(rows) + 1.3))
    vmax = max(np.nanpercentile(np.abs(M), 95), 1.0)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["FS preload", "FS relax"], fontsize=11)
    labels = rowlab if rowlab else [f"{a} · {b}" for a, b in rows]
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=9.5)
    for i in range(len(rows)):
        for j in range(2):
            ax.text(j, i, f"{M[i,j]:+.0f}%", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    grp = [r[0] for r in rows]
    for i in range(1, len(rows)):
        if grp[i] != grp[i - 1]:
            ax.axhline(i - 0.5, color="k", lw=1.2)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04); cb.set_label("% change vs no-FS")
    ax.set_title(title, fontsize=11)
    H.savefig(fig, out / name)


# (1) qualitative -------------------------------------------------------------
def fig_pv_loops_grid(out, drop_relax=False):
    """Grid of realized PV loops: rows = activation model, cols = LV | RV; each cell
    shows the 8-case severity fan. drop_relax=True omits the FS-relax row (lite version)."""
    bundles = BUN[:2] if drop_relax else BUN
    sev = [H.rv_systolic(H.SWEEP / BUN[0][0] / c) for c in H.CASES]
    norm = H.Normalize(min(sev), max(sev))
    nrow = len(bundles)
    fig, axes = plt.subplots(nrow, 2, figsize=(10.0, 3.6 * nrow),
                             constrained_layout=True, squeeze=False)
    for ri, (b, lab, *_), in enumerate(bundles):
        for ci, (ch, vk, pcol) in enumerate(CH):
            ax = axes[ri, ci]
            for c in H.CASES:
                m, o, sp = H.load_loop(H.SWEEP / b / c)
                V, P = _rv_loop(o, sp, vk, pcol)
                ax.plot(V, P, color=H.CMAP(norm(H.rv_systolic(H.SWEEP / b / c))),
                        lw=1.3, alpha=0.85)
            ax.set_xlabel(f"{ch} FEM cavity volume (mL)"); H.style(ax)
            ax.set_ylabel((f"{lab}\n" if ci == 0 else "") + f"{ch} pressure (mmHg)",
                          fontsize=9 if ci == 0 else 10)
            if ri == 0:
                ax.set_title(ch, fontweight="bold")
    sm = H.ScalarMappable(cmap=H.CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=40); cb.set_label(H.SEV, fontsize=9)
    fig.suptitle("Pressure-volume loops by activation model", fontsize=12)
    H.savefig(fig, out / ("pv_loops_grid_no_relax" if drop_relax else "pv_loops_grid"))


# (2) quantitative hemodynamics ----------------------------------------------
def _hemo(bundle):
    acc = {}
    for c in H.CASES:
        m, o, sp = H.load_loop(H.SWEEP / bundle / c)
        for ch, vk, pcol in CH:
            V, P = _rv_loop(o, sp, vk, pcol)
            edv, esv = V.max(), V.min(); sv = edv - esv
            vals = {"EDV": edv, "ESV": esv, "stroke vol": sv, "EF": 100 * sv / edv,
                    "peak P": P.max(), "stroke work": loop_area(V, P)}
            for q, val in vals.items():
                acc.setdefault((ch, q), []).append(val)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def fig_hemodynamics_delta(out):
    base, pre, rel = (_hemo(b) for b in ("no_frank_starling",
                                         "frank_starling_preload", "frank_starling_relax"))
    rows = [(ch, q) for ch, *_ in CH
            for q in ["EDV", "ESV", "stroke vol", "EF", "peak P", "stroke work"]]
    _delta_heatmap(rows, base, pre, rel, out, "hemodynamics_delta",
                   "Hemodynamic change when FS is added (mean over sweep)", figw=6.6)


# (3) fibre stress over the beat ---------------------------------------------
def fig_activation_stress_curves(out, case="case4_rv65"):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True, sharex=True)
    for ax, reg in zip(axes, REGS):
        for b, lab, col, ls in BUN:
            cd = H.SWEEP / b / case
            m = np.load(cd / "metrics/metrics_downsample_1.npy", allow_pickle=True).item()
            s = beat(m[f"mean_S_ff_{reg}"]) * H.PA_TO_KPA
            ph = np.linspace(0, 1, len(s))
            leg = f"{lab} — RV {H.rv_systolic(cd):.0f} mmHg"   # realized peak RV pressure
            ax.plot(ph, s, ls, color=col, lw=2.0, alpha=0.9, label=leg)
        ax.set_title(reg, fontweight="bold"); ax.set_xlabel("cardiac cycle (fraction)")
        ax.set_ylabel("mean fibre stress $S_{ff}$ (kPa)"); H.style(ax)
    axes[0].legend(frameon=False, fontsize=8.0, loc="upper right")
    fig.suptitle(f"Fibre stress over the beat by activation model ({case})", fontsize=12)
    H.savefig(fig, out / "activation_stress_curves")


# (4) regional energetics (areas) --------------------------------------------
def _energetics(bundle):
    df = H.aggregate(bundle)
    matched = {"LV": "PLV", "RV": "PRV", "Septum": "PRV"}   # wall-matched pressure proxy
    q = {}
    for reg in REGS:
        q[(reg, r"true work $\oint S{:}dE$")] = float(df[f"{reg}_W"].mean())
        q[(reg, r"proxy work $\oint P\,d\varepsilon$")] = float(df[f"{reg}_{matched[reg]}"].mean())
    return q


def fig_energetics_delta(out):
    base, pre, rel = (_energetics(b) for b in ("no_frank_starling",
                                               "frank_starling_preload", "frank_starling_relax"))
    rows = list(base.keys())   # region-major: true work then proxy work per region
    _delta_heatmap(rows, base, pre, rel, out, "energetics_delta",
                   "Regional work change when FS is added (mean over sweep)", figw=7.0)


# (5) the monotonic confound -------------------------------------------------
def fig_rv_proxy_confound(out, bundle="no_frank_starling"):
    """Pairwise correlation matrix of RV true work and every candidate proxy across the
    sweep. Across this monotonic sweep all non-transmural pressures are mutually collinear
    (r ~ 1), so correlation cannot single out RV pressure — only transmural breaks the
    pattern. This is the figure that must NOT be glossed over: it shows the correlation
    argument for RV pressure is confounded by monotonicity, n=8."""
    df = H.aggregate(bundle)
    def beat(a):
        a = np.asarray(a); return a[H.last_beat_slice(len(a))]
    def loop(x, y):
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    eps_amp, normP = [], []   # bare strain amplitude; pressure-shape-only proxy
    for c in H.CASES:
        m, o, sp = H.load_loop(H.SWEEP / bundle / c)
        e = beat(m["mean_E_ll_RV"]) * 100; pR = beat(sp[:, 1]); L = min(len(e), len(pR))
        e, pR = e[:L], pR[:L]
        eps_amp.append(e.max() - e.min())
        normP.append(loop(e, pR / pR.max()))
    names = ["true work", "$P_{LV}$", "$P_{RV}$", "mean", "sum", "trans",
             r"$\varepsilon_{ll}$", "$P_{RV}$/pk"]
    keys = ["RV_W", "RV_PLV", "RV_PRV", "RV_Mean", "RV_Sum", "RV_Trans"]
    cols = [np.asarray(df[k], float) for k in keys] + [np.array(eps_amp), np.array(normP)]
    M = np.array([[np.corrcoef(a, b)[0, 1] for b in cols] for a in cols])
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, fontsize=10)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(M[i, j]) > 0.55 else "black")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03); cb.set_label("Pearson r")
    blab = {"no_frank_starling": "no FS", "frank_starling_preload": "FS preload (ED)",
            "frank_starling_relax": "FS relax"}.get(bundle, bundle)
    ax.set_title(f"RV pressure proxies — pairwise correlation across the sweep ({blab})", fontsize=11)
    H.savefig(fig, out / f"rv_proxy_confound_{bundle}")


def fig_transmural_diagnostic(out, case="case7_rv95", bundle="no_frank_starling"):
    """Why the transmural (P_LV-P_RV) proxy is jagged. It is a small difference of two large,
    smooth pressures. Activation timing is identical across cases (shared envelope, LV pressure
    peaks at the same phase every case), but under high pulmonary afterload RV systole is
    prolonged (its pressure peak shifts later), so late in systole P_RV briefly exceeds P_LV
    and the transmural difference snaps negative. This is a loading effect, not a timing bug,
    and it is the hemodynamic basis of paradoxical septal motion.
    (a) pressure traces over the beat (most severe case);
    (b) across the sweep: RV pressure-peak timing and transmural minimum vs afterload."""
    def beat(a):
        a = np.asarray(a); return a[H.last_beat_slice(len(a))]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    m, o, sp = H.load_loop(H.SWEEP / bundle / case)
    pL, pR = beat(sp[:, 0]), beat(sp[:, 1]); tm = pL - pR; ph = np.linspace(0, 1, len(tm))
    axa.plot(ph, pL, color="#377eb8", label="$P_{LV}$")
    axa.plot(ph, pR, color="#984ea3", label="$P_{RV}$")
    axa.plot(ph, tm, color="#e31a1c", lw=2, label="$P_{LV}-P_{RV}$")
    axa.axhline(0, color="0.6", lw=0.8)
    axa.set_xlabel("cardiac cycle (fraction)"); axa.set_ylabel("pressure (mmHg)")
    axa.legend(frameon=False, fontsize=9); H.style(axa)
    axa.set_title(f"(a) pressure traces ({case})", fontweight="bold")
    # (b) RV pressure curves over the beat, all cases overlaid, coloured by peak RV
    # pressure -> the peak shifts right (RV systole prolongs) as afterload rises.
    sev = [H.rv_systolic(H.SWEEP / bundle / c) for c in H.CASES]
    norm = H.Normalize(min(sev), max(sev))
    for c, s in zip(H.CASES, sev):
        _, _, sp2 = H.load_loop(H.SWEEP / bundle / c)
        pR2 = beat(sp2[:, 1]); ph2 = np.linspace(0, 1, len(pR2))
        axb.plot(ph2, pR2, color=H.CMAP(norm(s)), lw=1.7, alpha=0.9)
    axb.axvline(np.argmax(beat(sp[:, 0])) / len(tm), color="0.5", ls=":", lw=1.0)
    axb.text(np.argmax(beat(sp[:, 0])) / len(tm), axb.get_ylim()[1], " LV peak",
             color="0.4", fontsize=8, va="top")
    axb.set_xlabel("cardiac cycle (fraction)"); axb.set_ylabel("RV pressure (mmHg)")
    H.style(axb); axb.set_title("(b) RV pressure curves across the sweep", fontweight="bold")
    sm = H.ScalarMappable(cmap=H.CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axb, shrink=0.85, pad=0.02)
    cb.set_label("peak RV systolic (mmHg)", fontsize=9)
    fig.suptitle("Transmural pressure diagnostic", fontsize=12)
    H.savefig(fig, out / "transmural_diagnostic")


def main():
    out = H.OUT / "comparison"; out.mkdir(parents=True, exist_ok=True)
    fig_pv_loops_grid(out)                 # full: no-FS / preload / relax
    fig_pv_loops_grid(out, drop_relax=True)  # lite: drops the FS-relax row
    fig_hemodynamics_delta(out)
    fig_activation_stress_curves(out)
    fig_energetics_delta(out)
    fig_rv_proxy_confound(out, "no_frank_starling")
    fig_rv_proxy_confound(out, "frank_starling_preload")
    fig_transmural_diagnostic(out)
    print(f"wrote comparison figures -> {out}")


if __name__ == "__main__":
    main()
