#!/usr/bin/env python3
"""Supervisor-report figures for the pulmonary-sweep study (RV-led framing).

All computed from the fixed-ratio no-FS sweep (+ FS bundles and the softmat
pilot). Login-safe: numpy + matplotlib only, no FEniCSx. Okabe-Ito colours for
identity, single-hue viridis ramp for the ordered severity axis. Regions use the
canonical region_tags mask (LV=1 / RV=2 / Septum=3).

Fig 1  LEAD        : the RV carries the dynamic range; P_RV tracks its work
Fig 2  the metric  : per-region proxy correlations + why the clean sweep needs noise
Fig 3  experiments : softening saturates, Frank-Starling doesn't open the LV gap
Fig 4  supporting  : volume->strain collapse (collaborator ED-overlap explainer)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

FR = paths.RESULTS_ROOT / "sims/2026-06-22/pah_pulmonary_fixedratio"
SOFT = paths.RESULTS_ROOT / "sims/2026-07-08/softmat_pilot_L10"
OUT = paths.RESULTS_ROOT / "handover/supervisor_2026-08"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [f"case{i}_rv{r}" for i, r in enumerate([25, 35, 45, 55, 65, 75, 85, 95])]
REG = {"LV": 1, "RV": 2, "Septum": 3}
PROXIES = ["PLV", "PRV", "Mean", "Trans"]
BLACK, ORANGE, SKY, GREEN, BLUE, VERM, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
REG_COL = {"RV": VERM, "LV": BLUE, "Septum": GREEN}
PROXY_COL = {"PLV": BLUE, "PRV": VERM, "Mean": GREEN, "Trans": ORANGE}

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "figure.dpi": 120,
})


def find_metrics(cd):
    for p in ["metrics_downsample_1.npy", "metrics_downsample_2.npy"]:
        if (cd / "metrics" / p).exists():
            return cd / "metrics" / p
    g = list((cd / "metrics").glob("metrics*.npy")) if (cd / "metrics").exists() else []
    return g[0] if g else None


def loop_area(V, P):
    V, P = np.asarray(V, float), np.asarray(P, float)
    return abs(0.5 * np.sum((V - np.roll(V, 1)) * (P + np.roll(P, 1))))


def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def load_sweep():
    rvsys = []
    W = {r: [] for r in REG}
    PX = {r: {p: [] for p in PROXIES} for r in REG}
    eff_sep, eff_lv, v_lv, p_lv = [], [], [], []
    for c in CASES:
        cd = FR / "no_frank_starling" / c
        d = np.load(cd / "per_cell_data.npz", allow_pickle=True)
        rt = np.asarray(d["region_tags"]); vol = np.asarray(d["cell_volumes"])
        m = np.load(find_metrics(cd), allow_pickle=True).item()
        rvsys.append(float(np.asarray(m["p_RV"], float).max()))
        for r, tag in REG.items():
            msk = rt == tag
            W[r].append(float(np.sum(np.asarray(d["w_total"])[msk] * vol[msk])))
            for p in PROXIES:
                PX[r][p].append(float(np.sum(np.asarray(d[f"proxy_{p}_ff"])[msk] * vol[msk])))
        eff_sep.append(np.asarray(m["mean_E_ff_Septum"], float))
        eff_lv.append(np.asarray(m["mean_E_ff_LV"], float))
        v_lv.append(np.asarray(m["V_LV_FEM"], float))
        p_lv.append(np.asarray(m["p_LV"], float))
    return dict(rvsys=np.array(rvsys),
                W={r: np.array(v) for r, v in W.items()},
                PX={r: {p: np.array(v) for p, v in d.items()} for r, d in PX.items()},
                eff_sep=eff_sep, eff_lv=eff_lv, v_lv=v_lv, p_lv=p_lv)


def _idx(v):
    """Index to the mild case, magnitude-wise (works are negative)."""
    return np.abs(v) / np.abs(v[0])


def fig1(S):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    x = S["rvsys"]
    # panel A: internal-work dynamic range by region
    for r in ["RV", "LV", "Septum"]:
        y = _idx(S["W"][r])
        a.plot(x, y, "-o", color=REG_COL[r], lw=2.4, ms=5)
        a.annotate(f"{r}  (×{y[-1]:.1f})", (x[-1], y[-1]), xytext=(6, 0),
                   textcoords="offset points", color=REG_COL[r], fontsize=10,
                   va="center", weight="bold")
    a.axhline(1.0, color="0.6", lw=0.8, ls=":")
    a.set_xlabel("RV systolic pressure [mmHg]")
    a.set_ylabel("true internal work, indexed to mildest case")
    a.set_title("The RV carries the dynamic range", fontsize=12)
    a.set_xlim(x[0] - 2, x[-1] + 20)
    # panel B: RV proxy tracking — indexed overlay against true RV work
    yt = _idx(S["W"]["RV"])
    b.plot(x, yt, "-o", color=BLACK, lw=3.0, ms=6, zorder=5)
    b.annotate("true RV work", (x[-1], yt[-1]), xytext=(6, 0),
               textcoords="offset points", color=BLACK, fontsize=10, va="center", weight="bold")
    for p in ["PRV", "Mean", "Trans"]:
        yp = _idx(S["PX"]["RV"][p])
        r = np.corrcoef(S["W"]["RV"], S["PX"]["RV"][p])[0, 1]
        b.plot(x, yp, "-o", color=PROXY_COL[p], lw=2.0, ms=4)
        b.annotate(f"{p}  (r={r:+.2f})", (x[-1], yp[-1]), xytext=(6, 0),
                   textcoords="offset points", color=PROXY_COL[p], fontsize=9,
                   va="center", weight="bold")
    b.axhline(1.0, color="0.6", lw=0.8, ls=":")
    b.set_xlabel("RV systolic pressure [mmHg]")
    b.set_ylabel("indexed to mildest case")
    b.set_title("P_RV tracks true RV work; transmural anti-tracks", fontsize=12)
    b.set_xlim(x[0] - 2, x[-1] + 22)
    out = OUT / "fig1_rv_leads_dynamic_range.png"
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf")); plt.close(fig)
    return out


def fig2(S):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    regions = ["RV", "LV", "Septum"]
    xpos = np.arange(len(regions)); w = 0.2
    # panel A: Pearson r(proxy, true work) grouped by region
    for j, p in enumerate(PROXIES):
        vals = [np.corrcoef(S["W"][r], S["PX"][r][p])[0, 1] for r in regions]
        a.bar(xpos + (j - 1.5) * w, vals, w, color=PROXY_COL[p], label=p)
    a.axhline(0, color="0.5", lw=0.8)
    a.set_xticks(xpos); a.set_xticklabels(regions)
    a.set_ylabel("Pearson r(proxy, true internal work)")
    a.set_title("Each free wall has a clear ipsilateral winner", fontsize=12)
    a.legend(fontsize=9, ncol=2, loc="lower center")
    a.set_ylim(-1.05, 1.15)
    # panel B: the degeneracy — Spearman saturates for good proxies on the clean sweep
    for j, p in enumerate(PROXIES):
        vals = [spear(S["W"][r], S["PX"][r][p]) for r in regions]
        b.bar(xpos + (j - 1.5) * w, vals, w, color=PROXY_COL[p], label=p)
    b.axhline(1.0, color="0.5", lw=0.8, ls="--")
    b.text(len(regions) - 1, 1.02, "monotone ceiling — good proxies tie here",
           ha="right", fontsize=8.5, color="0.35")
    b.axhline(0, color="0.5", lw=0.8)
    b.set_xticks(xpos); b.set_xticklabels(regions)
    b.set_ylabel("Spearman rank r")
    b.set_title("Clean sweep → good proxies tie at 1.0 → needs noise to rank", fontsize=11.5)
    b.set_ylim(-1.05, 1.2)
    out = OUT / "fig2_proxy_correlations.png"
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf")); plt.close(fig)
    return out


def _lv_sw(cd):
    m = np.load(find_metrics(cd), allow_pickle=True).item()
    return loop_area(m["V_LV_FEM"], m["p_LV"])


def fig3():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    scales, sp_soft = [], []
    for tag, sv in [("100", 1.0), ("050", 0.5), ("033", 0.33)]:
        wb = _lv_sw(SOFT / f"scale{tag}" / "case0_rv25")
        ws = _lv_sw(SOFT / f"scale{tag}" / "case7_rv95")
        scales.append(sv); sp_soft.append(100 * (wb - ws) / wb)
    a.bar([f"{s:.2f}×" for s in scales], sp_soft, color=SKY, width=0.6)
    for i, v in enumerate(sp_soft):
        a.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9)
    a.set_ylabel("LV case0→case7 stroke-work spread [%]")
    a.set_xlabel("passive stiffness scale")
    a.set_title("Softening inflates loops but saturates the gap", fontsize=11)
    a.set_ylim(0, max(sp_soft) * 1.35)
    modes = [("no-FS", "no_frank_starling"), ("FS preload", "frank_starling_preload"),
             ("FS relax", "frank_starling_relax")]
    sp_fs = []
    for _, d in modes:
        sw = np.array([_lv_sw(FR / d / c) for c in CASES])
        sp_fs.append(100 * (sw.max() - sw.min()) / sw.max())
    b.bar([m[0] for m in modes], sp_fs, color=GREEN, width=0.6)
    for i, v in enumerate(sp_fs):
        b.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9)
    b.set_ylabel("LV 8-case stroke-work spread [%]")
    b.set_xlabel("active-contraction model")
    b.set_title("Frank-Starling doesn't open the LV gap either", fontsize=11)
    b.set_ylim(0, max(sp_fs) * 1.35)
    out = OUT / "fig3_experiments_summary.png"
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf")); plt.close(fig)
    return out


def _last_beat_slice(e_ref):
    e = np.asarray(e_ref); hi = e > 0.035
    starts = np.where((~hi[:-1]) & (hi[1:]))[0] + 1
    return slice(int(starts[-2]), int(starts[-1]) + 1) if len(starts) >= 2 else slice(0, len(e))


def fig4(S):
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.9), constrained_layout=True)
    a, b, c = axs
    ramp = cm.viridis(np.linspace(0.05, 0.9, len(CASES)))
    sl = _last_beat_slice(S["eff_sep"][0])
    for i, e in enumerate(S["eff_sep"]):
        seg = e[sl]; a.plot(np.linspace(0, 1, len(seg)), seg, color=ramp[i], lw=1.7)
    a.axhline(-0.124, color="0.5", lw=0.8, ls="--")
    a.annotate("peak pinned ≈ −0.124\n(set by contraction, not load)", (0.55, -0.124),
               xytext=(0.30, -0.095), fontsize=9, arrowprops=dict(arrowstyle="->", color="0.4"))
    a.annotate("ED offset drifts\n+0.056 → +0.044", (0.02, 0.050), xytext=(0.22, 0.028),
               fontsize=9, arrowprops=dict(arrowstyle="->", color="0.4"))
    a.set_xlabel("beat fraction (one beat)"); a.set_ylabel("septal fiber strain $E_{ff}$")
    a.set_title("Septum contracts to the same peak", fontsize=11.5)
    ed_v = np.array([S["v_lv"][i][sl][0] for i in range(len(CASES))])
    ed_e = np.array([S["eff_lv"][i][sl][0] for i in range(len(CASES))])
    for i in range(len(CASES)):
        V, P, E = S["v_lv"][i][sl], S["p_lv"][i][sl], S["eff_lv"][i][sl]
        b.plot(V, P, color=ramp[i], lw=0.8, alpha=0.30)
        b.plot(V[0], P[0], "o", color=ramp[i], ms=8, mec="white", mew=0.6, zorder=5)
        c.plot(E, P, color=ramp[i], lw=0.8, alpha=0.30)
        c.plot(E[0], P[0], "o", color=ramp[i], ms=8, mec="white", mew=0.6, zorder=5)
    b.set_xlabel("LV volume [mL]"); b.set_ylabel("LV pressure [mmHg]")
    b.set_title(f"ED spreads {np.ptp(ed_v):.0f} mL in VOLUME", fontsize=11.5)
    c.set_xlabel("LV fiber strain $E_{ff}$")
    c.set_title(f"… but only {np.ptp(ed_e):.3f} in STRAIN", fontsize=11.5)
    b.axvspan(ed_v.min(), ed_v.max(), color="0.5", alpha=0.10)
    c.axvspan(ed_e.min(), ed_e.max(), color="0.5", alpha=0.10)
    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(S["rvsys"][0], S["rvsys"][-1]))
    cb = fig.colorbar(sm, ax=axs, pad=0.01, fraction=0.03); cb.set_label("RV systolic [mmHg]", fontsize=9)
    fig.suptitle("Supporting: why the loops overlap at ED (volume spread → cube-rooted into strain)",
                 fontsize=12, weight="bold")
    out = OUT / "fig4_ed_strain_volume_explainer.png"
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf")); plt.close(fig)
    return out


def main():
    S = load_sweep()
    outs = [fig1(S), fig2(S), fig3(), fig4(S)]
    print("wrote:")
    for o in outs:
        print(" ", o)
    print("\nInternal-work dynamic range (|max/min|):")
    for r in ["RV", "LV", "Septum"]:
        W = S["W"][r]
        print(f"  {r:7} {np.abs(W).max()/np.abs(W).min():.2f}×   best proxy Pearson: " +
              max(PROXIES, key=lambda p: np.corrcoef(W, S['PX'][r][p])[0, 1]) +
              f" ({max(np.corrcoef(W, S['PX'][r][p])[0,1] for p in PROXIES):+.3f})")


if __name__ == "__main__":
    main()
