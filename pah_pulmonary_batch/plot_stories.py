#!/usr/bin/env python3
"""Two self-contained supervisor notes, kept strictly separate.

STORY 1 (story1_softening/) — the softening-parameters investigation, 3 plots:
  p1  the problem   : LV / RV / Septum fiber stress-strain loops (baseline sweep)
  p2  the attempt   : softened material — regions x stiffness, baseline vs severe
  p3  the outcome   : the baseline->severe gap is not recovered

STORY 2 (story2_rv_proxy/) — where the answerable signal is:
  s2a dynamic range : RV internal work 2.7x, LV modest, septum flat
  s2b work ratio    : how hard the LV works relative to the RV
  s2c proxy tracking: indexed to mildest case, incl. P_LV — direction vs magnitude

ED is the tip of every loop = maximum fiber strain (end of filling); it is marked
with argmax(E_ff), never index 0. Login-safe: numpy + matplotlib only.
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

FR = paths.RESULTS_ROOT / "sims/2026-06-22/pah_pulmonary_fixedratio/no_frank_starling"
SOFT = paths.RESULTS_ROOT / "sims/2026-07-08/softmat_pilot_L10"
OUT1 = paths.RESULTS_ROOT / "handover/supervisor_2026-08/story1_softening"
OUT2 = paths.RESULTS_ROOT / "handover/supervisor_2026-08/story2_rv_proxy"
OUT1.mkdir(parents=True, exist_ok=True); OUT2.mkdir(parents=True, exist_ok=True)

CASES = [f"case{i}_rv{r}" for i, r in enumerate([25, 35, 45, 55, 65, 75, 85, 95])]
REG = {"LV": 1, "RV": 2, "Septum": 3}
BLACK, ORANGE, SKY, GREEN, BLUE, VERM, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
REG_COL = {"RV": VERM, "LV": BLUE, "Septum": GREEN}
PROXY_COL = {"PLV": BLUE, "PRV": VERM, "Mean": GREEN, "Trans": ORANGE}
RAMP = cm.viridis(np.linspace(0.05, 0.9, len(CASES)))

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "figure.dpi": 120,
})


def metrics(cd):
    for p in ["metrics_downsample_1.npy", "metrics_downsample_2.npy"]:
        if (cd / "metrics" / p).exists():
            return np.load(cd / "metrics" / p, allow_pickle=True).item()
    g = list((cd / "metrics").glob("metrics*.npy"))
    return np.load(g[0], allow_pickle=True).item()


def ed_slice(e):
    """One ED-to-ED beat, sliced between the last two strain maxima (ED = max stretch)."""
    e = np.asarray(e)
    thr = e.min() + 0.6 * (e.max() - e.min())
    idx = np.where(e >= thr)[0]
    if len(idx) < 2:
        return slice(0, len(e))
    eds, grp = [], [idx[0]]
    for i in idx[1:]:
        if i - grp[-1] <= 30:
            grp.append(i)
        else:
            eds.append(grp[int(np.argmax(e[grp]))]); grp = [i]
    eds.append(grp[int(np.argmax(e[grp]))])
    return slice(eds[-2], eds[-1] + 1) if len(eds) >= 2 else slice(0, len(e))


def ed_mark(E):
    """Index of ED within a one-beat loop = the tip (maximum fiber strain)."""
    return int(np.argmax(np.asarray(E)))


def loops_of(m, sl):
    return {r: (np.asarray(m[f"mean_E_ff_{r}"], float)[sl],
                np.asarray(m[f"mean_S_ff_{r}"], float)[sl] / 1e3) for r in REG}  # kPa


def rv_sys(m):
    return float(np.asarray(m["p_RV"], float).max())


def stroke_work(m):
    V, P = np.asarray(m["V_LV_FEM"], float), np.asarray(m["p_LV"], float)
    return abs(0.5 * np.sum((V - np.roll(V, 1)) * (P + np.roll(P, 1))))


def region_work(cd):
    d = np.load(cd / "per_cell_data.npz", allow_pickle=True)
    rt = np.asarray(d["region_tags"]); vol = np.asarray(d["cell_volumes"])
    out = {}
    for r, tag in REG.items():
        msk = rt == tag
        out[r] = {"W": float(np.sum(np.asarray(d["w_total"])[msk] * vol[msk]))}
        for p in ["PLV", "PRV", "Mean", "Trans"]:
            out[r][p] = float(np.sum(np.asarray(d[f"proxy_{p}_ff"])[msk] * vol[msk]))
    return out


def sweep():
    rvsys = []
    S = {r: {k: [] for k in ["W", "PLV", "PRV", "Mean", "Trans"]} for r in REG}
    loops = {r: [] for r in REG}
    for c in CASES:
        cd = FR / c
        m = metrics(cd)
        rvsys.append(rv_sys(m))
        sl = ed_slice(m["mean_E_ff_LV"])
        L = loops_of(m, sl)
        for r in REG:
            loops[r].append(L[r])
        w = region_work(cd)
        for r in REG:
            for k in ["W", "PLV", "PRV", "Mean", "Trans"]:
                S[r][k].append(w[r][k])
    return dict(rvsys=np.array(rvsys),
                S={r: {k: np.array(v) for k, v in d.items()} for r, d in S.items()},
                loops=loops)


def idx(v):
    return np.abs(v) / np.abs(v[0])


# ============================================================ STORY 1 (3 plots)
def p1_problem(D):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)
    for ax, r in zip(axs, ["LV", "RV", "Septum"]):
        for i, (E, Sf) in enumerate(D["loops"][r]):
            ax.plot(E, Sf, color=RAMP[i], lw=1.4)
            j = ed_mark(E)
            ax.plot(E[j], Sf[j], "o", color=RAMP[i], ms=8, mec="white", mew=0.6, zorder=5)
        ax.set_xlabel("fiber strain $E_{ff}$"); ax.set_title(r)
    axs[0].set_ylabel("fiber stress $S_{ff}$ [kPa]")
    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(D["rvsys"][0], D["rvsys"][-1]))
    cb = fig.colorbar(sm, ax=axs, pad=0.01, fraction=0.03); cb.set_label("RV systolic [mmHg]")
    fig.suptitle("Fiber stress–strain loops across the afterload sweep  (dots = end-diastole)",
                 fontsize=12.5, weight="bold")
    out = OUT1 / "p1_problem_stress_strain.png"
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf")); plt.close(fig)
    return out


def p2_softening_grid():
    scales = [("100", "1.00×"), ("050", "0.50×"), ("033", "0.33×")]
    cases = [("case0_rv25", BLUE, "--"), ("case7_rv95", VERM, "-")]
    # cache loops + RV systolic
    data = {}
    labels = {}
    for tag, _ in scales:
        for cs, _, _ in cases:
            m = metrics(SOFT / f"scale{tag}" / cs)
            data[(tag, cs)] = loops_of(m, ed_slice(m["mean_E_ff_LV"]))
            labels[cs] = rv_sys(m)
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 12), constrained_layout=True)
    for ri, r in enumerate(["LV", "RV", "Septum"]):
        for ci, (tag, stitle) in enumerate(scales):
            ax = axs[ri, ci]
            for cs, col, ls in cases:
                E, Sf = data[(tag, cs)][r]
                ax.plot(E, Sf, ls, color=col, lw=1.7)
                j = ed_mark(E)
                ax.plot(E[j], Sf[j], "o", color=col, ms=7, mec="white", mew=0.6, zorder=5)
            if ri == 0:
                ax.set_title(f"stiffness {stitle}", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{r}\n$S_{{ff}}$ [kPa]", fontsize=10)
            if ri == 2:
                ax.set_xlabel("fiber strain $E_{ff}$")
    handles = [plt.Line2D([], [], color=BLUE, ls="--", lw=2,
                          label=f"baseline  (RV {labels['case0_rv25']:.0f} mmHg)"),
               plt.Line2D([], [], color=VERM, ls="-", lw=2,
                          label=f"severe  (RV {labels['case7_rv95']:.0f} mmHg)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10.5,
               bbox_to_anchor=(0.5, 1.035))
    fig.suptitle("Softened material: stress–strain by region and stiffness  (dots = ED)",
                 fontsize=12.5, weight="bold", y=1.06)
    out = OUT1 / "p2_softening_grid.png"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def p3_outcome():
    scales, tags = [1.00, 0.50, 0.33], ["100", "050", "033"]
    ed0, ed7, w0, w7 = [], [], [], []
    for tag in tags:
        m0, m7 = metrics(SOFT / f"scale{tag}" / "case0_rv25"), metrics(SOFT / f"scale{tag}" / "case7_rv95")
        ed0.append(float(np.max(m0["mean_E_ff_LV"])))   # ED = max fiber strain
        ed7.append(float(np.max(m7["mean_E_ff_LV"])))
        w0.append(stroke_work(m0)); w7.append(stroke_work(m7))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    a.plot(scales, ed0, "-o", color=BLUE, lw=2, label="baseline (RV≈24)")
    a.plot(scales, ed7, "-o", color=VERM, lw=2, label="severe (RV≈80)")
    a.set_xlabel("passive stiffness scale"); a.set_ylabel("LV end-diastolic strain $E_{ff}$")
    a.set_title("ED strain: baseline vs severe stay together")
    a.set_xticks(scales); a.invert_xaxis(); a.legend(fontsize=9)
    b.plot(scales, w0, "-o", color=BLUE, lw=2, label="baseline (RV≈24)")
    b.plot(scales, w7, "-o", color=VERM, lw=2, label="severe (RV≈80)")
    b.set_xlabel("passive stiffness scale"); b.set_ylabel("LV stroke work [mmHg·mL]")
    b.set_title("LV stroke work: the gap does not open")
    b.set_xticks(scales); b.invert_xaxis(); b.legend(fontsize=9)
    out = OUT1 / "p3_outcome_gap_not_recovered.png"
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf")); plt.close(fig)
    return out


# ============================================================ STORY 2 (unchanged)
def s2a_dynamic_range(D):
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    x = D["rvsys"]
    for r in ["RV", "LV", "Septum"]:
        y = idx(D["S"][r]["W"])
        ax.plot(x, y, "-o", color=REG_COL[r], lw=2.6, ms=5)
        ax.annotate(f"{r}  (×{y[-1]:.1f})", (x[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", color=REG_COL[r], fontsize=10, va="center", weight="bold")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("RV systolic pressure [mmHg]")
    ax.set_ylabel("internal work ÷ its mildest-case value")
    ax.set_title("Internal-work change across the sweep, by region")
    ax.set_xlim(x[0] - 2, x[-1] + 20)
    out = OUT2 / "s2a_dynamic_range_by_region.png"
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf")); plt.close(fig)
    return out


def s2b_work_ratio(D):
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    x = D["rvsys"]
    ratio_true = np.abs(D["S"]["LV"]["W"]) / np.abs(D["S"]["RV"]["W"])
    ratio_prox = np.abs(D["S"]["LV"]["PLV"]) / np.abs(D["S"]["RV"]["PRV"])
    ax.plot(x, ratio_true, "-o", color=BLACK, lw=2.8, ms=6, label="true  W_LV / W_RV")
    ax.plot(x, ratio_prox, "--o", color=VERM, lw=2.0, ms=4,
            label="proxy  (P_LV·ε_LV) / (P_RV·ε_RV)")
    for xi, yi in zip(x, ratio_true):
        ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 7), textcoords="offset points",
                    fontsize=8, ha="center", color="0.3")
    ax.set_xlabel("RV systolic pressure [mmHg]")
    ax.set_ylabel("LV : RV internal-work ratio")
    ax.set_title("How hard the LV works relative to the RV")
    ax.legend(fontsize=9)
    out = OUT2 / "s2b_lv_rv_work_ratio.png"
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf")); plt.close(fig)
    return out


def s2c_proxy_tracking(D):
    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    x = D["rvsys"]
    yt = idx(D["S"]["RV"]["W"])
    ax.plot(x, yt, "-o", color=BLACK, lw=3.0, ms=6, zorder=5)
    ax.annotate("true RV work", (x[-1], yt[-1]), xytext=(6, 0), textcoords="offset points",
                color=BLACK, fontsize=10, va="center", weight="bold")
    for p in ["PRV", "PLV", "Mean", "Trans"]:
        yp = idx(D["S"]["RV"][p])
        r = np.corrcoef(D["S"]["RV"]["W"], D["S"]["RV"][p])[0, 1]
        ax.plot(x, yp, "-o", color=PROXY_COL[p], lw=2.0, ms=4)
        ax.annotate(f"{p}  (r={r:+.2f})", (x[-1], yp[-1]), xytext=(6, 0),
                    textcoords="offset points", color=PROXY_COL[p], fontsize=9, va="center", weight="bold")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("RV systolic pressure [mmHg]")
    ax.set_ylabel("value ÷ its mildest-case value")
    ax.set_title("RV: true work vs pressure-strain proxies")
    ax.set_xlim(x[0] - 2, x[-1] + 24)
    out = OUT2 / "s2c_proxy_tracking_indexed.png"
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf")); plt.close(fig)
    return out


def main():
    D = sweep()
    o = [p1_problem(D), p2_softening_grid(), p3_outcome(),
         s2a_dynamic_range(D), s2b_work_ratio(D), s2c_proxy_tracking(D)]
    print("wrote:")
    for f in o:
        print(" ", f)


if __name__ == "__main__":
    main()
