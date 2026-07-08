#!/usr/bin/env python3
"""
Handover/figure generator for the PAH pulmonary-windkessel sweep.

Three bundles (no-FS / FS-preload / FS-relaxation), 8 cases each (RV systolic
25->95 mmHg), one shared 8/5 inverse-unloaded mesh, canonical cell tagging.

Per bundle the figures are split into three categories:
  <bundle>/correlation/   HEADLINE: SS work vs PS proxy (longitudinal strain),
                          per region (LV / RV / Septum), over pressure choices
                          P_LV / P_RV / Trans / Mean / Sum / Affine(lambda).
  <bundle>/ratio/         LV/RV free-wall ratio spectrum + scatter; septum ratio.
  <bundle>/loops/         standalone-0D + coupled PV loops; per-frame (ED/unloaded)
                          fibre stress-strain & pressure-strain loops; septum
                          candidate-pressure grid.
  <bundle>/data/          scalar csv + correlation tables.

Run:  python3 pah_pulmonary_batch/make_pah_handover.py
"""
import os, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path
import paths
SWEEP = paths.RESULTS_ROOT / os.environ.get(
    "PAH_SWEEP", "sims/2026-06-09/pah_pulmonary_20260609_prodsweep")
OUT = paths.RESULTS_ROOT / os.environ.get(
    "PAH_OUT", "handover/pah_pulmonary_paper_20260611")
CASES = ["case0_rv25","case1_rv35","case2_rv45","case3_rv55",
         "case4_rv65","case5_rv75","case6_rv85","case7_rv95"]
BUNDLES = {
    "no_frank_starling":       "Constant active tension Ta=100 kPa (thesis model)",
    "frank_starling_preload":  "Frank-Starling frozen at ED stretch, Ta=220 kPa",
    "frank_starling_relax":    "Frank-Starling activation-lag, Ta=220 kPa, tau=250 ms",
}
REG = {"LV": 1, "RV": 2, "Septum": 3}
NBEATS = 6
PA_TO_KPA = 1e-3

CMAP = plt.cm.viridis
C_MODEL  = "#222222"
C_WITH   = "#1b7837"   # adjacent / wall-matched pressure
C_WITHOUT= "#d6604d"   # P_LV everywhere
HILITE   = "#fff3d4"
SEV = "peak RV systolic pressure (mmHg)"

# pressure choices for the correlation grids (longitudinal strain).
# Affine(lambda) is still computed in aggregate() and kept in the raw CSV, but is
# left out of the figures/tables for now to cut clutter (easy to re-add here).
PCHOICES = [
    ("PLV",   r"$P_{LV}\times\varepsilon_{ll}$"),
    ("PRV",   r"$P_{RV}\times\varepsilon_{ll}$"),
    ("Trans", r"$(P_{LV}{-}P_{RV})\times\varepsilon_{ll}$"),
    ("Mean",  r"$\frac{1}{2}(P_{LV}{+}P_{RV})\times\varepsilon_{ll}$"),
    ("Sum",   r"$(P_{LV}{+}P_{RV})\times\varepsilon_{ll}$"),
]

# short pressure labels for compact panels (heatmap / story scatters)
SHORT = {"PLV": "$P_{LV}$", "PRV": "$P_{RV}$", "Trans": "$P_{LV}{-}P_{RV}$",
         "Mean": "mean $P$", "Sum": "sum $P$", "Affine": "affine $P(\\lambda)$"}

def frame_strain(E, frame):
    """Reference-shift a Green-Lagrange strain trace.
    'unloaded' = raw E (relative to the stress-free reference);
    'ED' = re-zeroed at end-diastole (most-stretched instant; what speckle-tracking sees)."""
    E = np.asarray(E, float)
    return E - E.max() if frame == "ED" else E

def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)

def r_value(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def savefig(fig, path):
    fig.savefig(str(path) + ".png", dpi=170, bbox_inches="tight")
    fig.savefig(str(path) + ".pdf", bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def rv_systolic(case_dir):
    p = np.load(case_dir / "solver/solver_cavity_pressure_mmHg.npy").astype(float)
    n = len(p); lb = slice(n - n // NBEATS, n)
    return float(p[:, 1][lb].max())

AHA_MID = (4, 5, 6)   # Mid_LV / Mid_RV / Mid_Septum — the canonical AHA mid ring

def load_aha(cd, ncells):
    """Per-cell AHA biventricular tag (0=Apical,1-3=Basal,4-6=Mid LV/RV/Sep).
    Uses the `aha_tags.npy` sidecar from compute_aha_band.py (the proven path —
    gernerate_aha_biv on the geometry mesh mapped via ckpt_to_cg_idx). Returns None
    if absent, so callers skip the mid band cleanly."""
    side = cd / "aha_tags.npy"
    if side.exists():
        a = np.load(side).astype(np.int32)
        if len(a) == ncells:
            return a
    return None

def aggregate(bundle, band="full"):
    """Per-case, per-region true work + the 6 ll pressure-strain proxies.
    Work reported positive (negate the dW-convention sum).
    band="full" uses the whole LDRB region; band="mid" restricts to the AHA mid ring
    (tags 4/5/6) intersected with the region — the mid-ventricular slab away from base
    and apex."""
    rows = []
    for c in CASES:
        cd = SWEEP / bundle / c
        z = np.load(cd / "per_cell_data.npz", allow_pickle=True)
        tags = z["region_tags"]
        aha = load_aha(cd, len(tags))
        if band == "mid" and aha is None:
            raise FileNotFoundError(
                f"{cd}: no AHA tags (run compute_aha_band.py) — cannot aggregate band='mid'")
        midmask = np.isin(aha, AHA_MID) if aha is not None else np.ones(len(tags), bool)
        dlv, drv = z["d_lv"], z["d_rv"]
        lam = dlv / (dlv + drv + 1e-30)                       # 0 at LV face -> 1 at RV face
        affine = (1 - lam) * z["proxy_PLV_ll"] + lam * z["proxy_PRV_ll"]
        row = {"case": c, "sev": rv_systolic(cd)}
        for r, tag in REG.items():
            m = (tags == tag) & midmask if band == "mid" else (tags == tag)
            row[f"{r}_W"] = -float(z["w_total"][m].sum())      # true SS work (positive)
            row[f"{r}_Wff"] = -float(z["w_ff"][m].sum())
            row[f"{r}_PLV"]   = -float(z["proxy_PLV_ll"][m].sum())
            row[f"{r}_PRV"]   = -float(z["proxy_PRV_ll"][m].sum())
            row[f"{r}_Trans"] = -float(z["proxy_Trans_ll"][m].sum())
            row[f"{r}_Mean"]  = -float(z["proxy_Mean_ll"][m].sum())
            row[f"{r}_Sum"]   = -float(z["proxy_Sum_ll"][m].sum())
            row[f"{r}_Affine"]= -float(affine[m].sum())
        rows.append(row)
    # to simple column dict
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], dtype=object if k == "case" else float) for k in keys}

def scatter_fit(ax, x, y, sev, norm, title=None):
    """Severity-coloured scatter + least-squares line; r shown as a corner badge."""
    ax.scatter(x, y, c=sev, cmap=CMAP, norm=norm, s=70, ec="k", lw=0.5, zorder=3)
    if np.std(x) > 0:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, a + b * xs, "-", color="0.4", lw=1.2, zorder=2)
    r = r_value(x, y)
    ax.text(0.05, 0.94, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    if title is not None:
        ax.set_title(title, fontsize=11)
    ax.locator_params(axis="x", nbins=5); ax.locator_params(axis="y", nbins=6)
    style(ax)
    return r

# --------------------------------------------------------------------------- #
# Correlation (headline)
# --------------------------------------------------------------------------- #
def fig_region_correlation(df, region, out):
    """One region: grid of true SS work vs each ll pressure-strain proxy.
    Proxy formula is the panel title; r is a corner badge; one shared y-axis label.
    No 'best' is crowned: across this monotonic sweep every non-transmural pressure is
    collinear (r within ~0.02), so correlation cannot distinguish them — only transmural
    differs. Crowning a winner would over-claim; see fig_rv_proxy_confound."""
    sev = df["sev"]; norm = Normalize(sev.min(), sev.max())
    y = df[f"{region}_W"]
    rs = {k: r_value(df[f"{region}_{k}"], y) for k, _ in PCHOICES}
    nontrans = [abs(rs[k]) for k, _ in PCHOICES if k != "Trans" and np.isfinite(rs[k])]
    spread = max(nontrans) - min(nontrans) if nontrans else float("nan")
    n = len(PCHOICES); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.9 * nrow),
                             constrained_layout=True, sharey=True)
    axf = axes.ravel()
    for ax, (key, lab) in zip(axf, PCHOICES):
        scatter_fit(ax, df[f"{region}_{key}"], y, sev, norm, title=lab)
        ax.set_xlabel("proxy work (a.u.)")
    for ax in axf[n:]:
        ax.set_visible(False)
    fig.supylabel(f"true SS work  $\\oint S{{:}}dE$   ({region}, a.u.)", fontsize=12)
    sm = ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30); cb.set_label(SEV, fontsize=9)
    fig.suptitle(f"{region}: true work vs each pressure-strain proxy   "
                 f"(non-transmural r spread {spread:.2f} — indistinguishable)",
                 fontsize=12.5, fontweight="bold")
    savefig(fig, out / f"correlation_{region}")
    return rs

def fig_true_vs_proxy_range(df, out):
    """Per region across the sweep: true SS work (left axis) vs the matched pressure-strain
    proxy (right axis, different scale). Shows that for the septum/LV the proxy is NOT flat
    like the truth — it varies a lot because it tracks the rising pressure, so it would
    falsely report work changes that the true work does not have."""
    o = np.argsort(df["sev"]); x = df["sev"][o]
    matched = {"LV": "PLV", "RV": "PRV", "Septum": "PLV"}     # clinical adjacent-wall pressure
    plab = {"PLV": "$P_{LV}$", "PRV": "$P_{RV}$"}
    pcol = {"LV": "#4575b4", "RV": "#d73027", "Septum": "#762a83"}

    def rng(a):
        a = np.asarray(a, float); return 100 * (a.max() - a.min()) / abs(a.mean())

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    for ax, reg in zip(axes, REG):
        W = df[f"{reg}_W"][o]; pk = matched[reg]; Pw = df[f"{reg}_{pk}"][o]
        ax.plot(x, W, "-o", color="#222", lw=2.4, ms=5,
                label=f"true work  (range {rng(W):.0f}%)")
        ax.set_ylabel("true SS work (a.u.)")
        axr = ax.twinx()
        axr.plot(x, Pw, "--s", color=pcol[reg], lw=2.0, ms=5,
                 label=f"proxy {plab[pk]}  (range {rng(Pw):.0f}%)")
        axr.set_ylabel("proxy work (a.u.)")
        axr.spines["top"].set_visible(False)
        ax.set_title(reg, fontweight="bold"); ax.set_xlabel(SEV); style(ax)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("True work vs matched pressure-strain proxy across the sweep", fontsize=12)
    savefig(fig, out / "true_vs_proxy_range")

def fig_direction_recovers(out, bundle="no_frank_starling"):
    """Both clinically-measurable strain directions cancel in the septum: longitudinal (GLS)
    partially, circumferential almost completely. Only the local fibre (helical) direction —
    which follows the through-wall fibre rotation and is NOT measurable in vivo — is coherent.
    (a) directional coherence |net|/gross per region (1 = no cancellation);
    (b) septal signed coherence net/gross across the sweep (fibre pinned at -1, GLS wobbles,
        circumferential ~0). signed because septal P*deps is net-negative (shortening)."""
    DIRS = [("ll", "longitudinal (GLS)", "#762a83"),
            ("circ", "circumferential", "#e6ab02"),
            ("ff", "fibre (helical)", "#1b7837")]
    regs = list(REG)
    cohacc = {d: {r: [] for r in regs} for d, _, _ in DIRS}
    septsig = {d: [] for d, _, _ in DIRS}
    sev = []
    for c in CASES:
        z = np.load(SWEEP / bundle / c / "per_cell_data.npz")
        tags = z["region_tags"]; sev.append(rv_systolic(SWEEP / bundle / c))
        for d, _, _ in DIRS:
            va = z[f"proxy_PLV_{d}"]
            for r in regs:
                v = va[tags == REG[r]]; g = v[v > 0].sum() - v[v < 0].sum()
                cohacc[d][r].append(abs(v.sum()) / g if g > 0 else np.nan)
            vs = va[tags == REG["Septum"]]; gs = vs[vs > 0].sum() - vs[vs < 0].sum()
            septsig[d].append(vs.sum() / gs if gs > 0 else np.nan)
    o = np.argsort(sev); x = np.array(sev)[o]
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    xp = np.arange(len(regs)); w = 0.27
    for i, (d, lab, col) in enumerate(DIRS):
        axb.bar(xp + (i - 1) * w, [np.nanmean(cohacc[d][r]) for r in regs], w, label=lab, color=col)
    axb.set_xticks(xp); axb.set_xticklabels(regs); axb.set_ylim(0, 1.05)
    axb.set_ylabel("directional coherence   |net| / gross")
    axb.legend(frameon=False, fontsize=8.5); style(axb)
    axb.set_title("(a) cancellation by strain direction", fontweight="bold")
    for d, lab, col in DIRS:
        axc.plot(x, np.array(septsig[d])[o], "-o", color=col, lw=2.0, ms=5, label=lab)
    axc.axhline(0, color="0.6", lw=0.8); axc.set_ylim(-1.05, 0.25)
    axc.set_xlabel(SEV); axc.set_ylabel("septal signed coherence   net / gross")
    axc.legend(frameon=False, fontsize=8.5); style(axc)
    axc.set_title("(b) septal coherence across the sweep", fontweight="bold")
    fig.suptitle("Septal proxy: longitudinal vs circumferential vs fibre strain", fontsize=12)
    savefig(fig, out / "direction_recovers")

def fig_work_dynamic_range(df, out):
    """True work as % deviation from each region's sweep-mean. The pulmonary sweep
    loads the RV, so only the RV has dynamic range; LV/septum are nearly flat, which
    is why their proxy correlations are noise (and flip sign across activation models)."""
    o = np.argsort(df["sev"]); x = df["sev"][o]
    colors = {"LV": "#4575b4", "RV": "#d73027", "Septum": "#762a83"}
    fig, ax = plt.subplots(figsize=(8.0, 4.9), constrained_layout=True)
    ax.axhline(0, color="0.6", lw=0.8)
    for reg in REG:
        W = df[f"{reg}_W"][o]; dev = 100 * (W - W.mean()) / abs(W.mean())
        rng = 100 * (W.max() - W.min()) / abs(W.mean())
        ax.plot(x, dev, "-o", color=colors[reg], lw=2.2, ms=5,
                label=f"{reg}  (range {rng:.0f}% of mean)")
    ax.set_xlabel(SEV); ax.set_ylabel("true work deviation from sweep-mean (%)")
    ax.legend(frameon=False, fontsize=9.5); style(ax)
    ax.set_title("Dynamic range of true work by region", fontsize=12, fontweight="bold")
    savefig(fig, out / "work_dynamic_range")

# --------------------------------------------------------------------------- #
# Full region vs AHA mid ring
# --------------------------------------------------------------------------- #
def fig_band_compare(df_full, df_mid, out):
    """Full LDRB region vs AHA mid ring (tags 4/5/6), per region.
    (top) proxy correlation r(true work, P x eps_ll) for each pressure choice;
    (bottom) true-work dynamic range (% of sweep-mean). Restricting to the mid slab
    removes the base/apex, where fibre orientation and tethering differ — this shows
    whether the proxy conclusions and the RV-only dynamic range survive that cut."""
    regs = list(REG)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)
    xp = np.arange(len(PCHOICES)); w = 0.38
    for j, reg in enumerate(regs):
        ax = axes[0, j]
        rf = [r_value(df_full[f"{reg}_{k}"], df_full[f"{reg}_W"]) for k, _ in PCHOICES]
        rm = [r_value(df_mid[f"{reg}_{k}"], df_mid[f"{reg}_W"]) for k, _ in PCHOICES]
        ax.bar(xp - w/2, rf, w, label="full region", color="#9ecae1", ec="k", lw=0.4)
        ax.bar(xp + w/2, rm, w, label="AHA mid ring", color="#08519c", ec="k", lw=0.4)
        ax.axhline(0, color="0.5", lw=0.8); ax.set_ylim(-1.05, 1.05)
        ax.set_xticks(xp); ax.set_xticklabels([SHORT[k] for k, _ in PCHOICES], fontsize=8)
        ax.set_title(reg, fontweight="bold")
        if j == 0:
            ax.set_ylabel("r(true work, proxy)"); ax.legend(frameon=False, fontsize=9)
        style(ax)
    for j, reg in enumerate(regs):
        ax = axes[1, j]
        for df_, lab, col in [(df_full, "full region", "#9ecae1"),
                              (df_mid, "AHA mid ring", "#08519c")]:
            o = np.argsort(df_["sev"]); x = df_["sev"][o]; W = df_[f"{reg}_W"][o]
            rng = 100 * (W.max() - W.min()) / abs(W.mean())
            ax.plot(x, 100 * (W - W.mean()) / abs(W.mean()), "-o", color=col, lw=2.0, ms=4,
                    label=f"{lab} (range {rng:.0f}%)")
        ax.axhline(0, color="0.6", lw=0.8); ax.set_xlabel(SEV)
        if j == 0:
            ax.set_ylabel("true work dev. from mean (%)")
        ax.legend(frameon=False, fontsize=8.5); style(ax)
    fig.suptitle("Full LDRB region vs AHA mid ring", fontsize=13, fontweight="bold")
    savefig(fig, out / "band_compare")

# --------------------------------------------------------------------------- #
# Ratio
# --------------------------------------------------------------------------- #
def fig_ratio_spectrum(df, out):
    o = np.argsort(df["sev"]); x = df["sev"][o]
    ss     = (df["LV_W"]/df["RV_W"])[o]
    adj    = (df["LV_PLV"]/df["RV_PRV"])[o]            # wall-matched: LV uses PLV, RV uses PRV
    plvall = (df["LV_PLV"]/df["RV_PLV"])[o]            # PLV everywhere (no RV catheter)
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.plot(x, ss,     "-o", color=C_MODEL,  lw=2.4, ms=6,   label="SS work (model truth)")
    ax.plot(x, adj,    "-s", color=C_WITH,   lw=2.0, ms=5.5, label="PS proxy, adjacent (P_LV|LV, P_RV|RV)")
    ax.plot(x, plvall, "--^",color=C_WITHOUT,lw=2.0, ms=5.5, label="PS proxy, P_LV everywhere")
    ax.set_xlabel(SEV); ax.set_ylabel("LV / RV free-wall work ratio")
    ax.legend(frameon=False, fontsize=9); style(ax)
    ax.set_title("LV/RV free-wall work ratio across the loading sweep")
    savefig(fig, out / "ratio_spectrum")

def fig_ratio_scatter(df, out):
    model  = df["LV_W"]/df["RV_W"]
    adj    = df["LV_PLV"]/df["RV_PRV"]
    plvall = df["LV_PLV"]/df["RV_PLV"]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = [min(model.min(), adj.min(), plvall.min())*0.9, max(model.max(), adj.max(), plvall.max())*1.1]
    ax.plot(lim, lim, "-", color="0.6", lw=1)
    mae_adj = np.mean(np.abs(adj-model)); mae_plv = np.mean(np.abs(plvall-model))
    ax.scatter(model, adj,    c=C_WITH,    s=70, ec="k", lw=0.5, label=f"adjacent (MAE {mae_adj:.2f})")
    ax.scatter(model, plvall, c=C_WITHOUT, s=70, ec="k", lw=0.5, marker="^", label=f"P_LV everywhere (MAE {mae_plv:.2f})")
    ax.set_xlabel("SS work ratio (truth)"); ax.set_ylabel("PS proxy ratio")
    ax.legend(frameon=False, fontsize=9); style(ax); ax.set_aspect("equal")
    ax.set_title("Free-wall ratio: proxy vs truth")
    savefig(fig, out / "ratio_scatter")

def fig_septum_ratio(df, out):
    """Septum / mean-free-wall work ratio vs severity: SS truth vs each septum proxy."""
    o = np.argsort(df["sev"]); x = df["sev"][o]
    fw_mean = 0.5*(df["LV_W"]+df["RV_W"])
    fw_adj  = 0.5*(df["LV_PLV"]+df["RV_PRV"])
    ss = (df["Septum_W"]/fw_mean)[o]
    proxies = [("PLV","#377eb8","o","$P_{LV}$"),("PRV","#984ea3","s","$P_{RV}$"),
               ("Trans","#ff7f00","^","$P_{LV}{-}P_{RV}$"),("Mean","#1b7837","D","mean"),
               ("Sum","#999999","v","sum"),("Affine","#e7298a","*","affine($\\lambda$)")]
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    ax.plot(x, ss, "-o", color=C_MODEL, lw=2.4, ms=6, zorder=4, label="SS work")
    for key,col,mk,lab in proxies:
        ax.plot(x, (df[f"Septum_{key}"]/fw_adj)[o], "--", marker=mk, color=col, lw=1.6, ms=5, alpha=0.9, label=lab)
    ax.set_xlabel(SEV); ax.set_ylabel("septum / free-wall work ratio")
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.01,0.5)); style(ax)
    ax.set_title("Septum share of work: truth vs proxies")
    savefig(fig, out / "septum_ratio")

# --------------------------------------------------------------------------- #
# Circulation / qualitative loops
# --------------------------------------------------------------------------- #
def load_loop(cd):
    m = np.load(cd / "metrics/metrics_downsample_1.npy", allow_pickle=True).item()
    o = np.load(cd / "ode_state_history.npy", allow_pickle=True).item()
    sp = np.load(cd / "solver/solver_cavity_pressure_mmHg.npy").astype(float)
    # Return the FEM cavity volumes the model actually sees (V_FEM = V_0D * ratio), NOT the
    # raw coupled 0D volumes. Plotting the 0D volume against FEM pressure is misleading: the
    # per-case coupling ratio can clamp the FEM preload (0D EDV varies far more than the FEM
    # cavity does). Coupled plots must only show what the FEM sees.
    _sp = json.load(open(cd / "simulation_params.json"))
    o["V_LV"] = np.asarray(o["V_LV"]) * float(_sp.get("ratio_LV", 1.0))
    o["V_RV"] = np.asarray(o["V_RV"]) * float(_sp.get("ratio_RV", 1.0))
    return m, o, sp

def last_beat_slice(n):
    return slice(n - n // NBEATS, n)

def fig_pv_coupled(df, bundle, out):
    sev = df["sev"]; norm = Normalize(sev.min(), sev.max())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=True)
    for i, c in enumerate(CASES):
        m, o, sp = load_loop(SWEEP / bundle / c)
        col = CMAP(norm(df["sev"][i]))
        n = sp.shape[0]; lb = last_beat_slice(n)
        mn = min(len(o["V_LV"]), n); lo = last_beat_slice(mn)
        axes[0].plot(np.asarray(o["V_LV"])[:mn][lo], sp[:, 0][lb][:len(np.asarray(o["V_LV"])[:mn][lo])], color=col, lw=1.4, alpha=0.85)
        axes[1].plot(np.asarray(o["V_RV"])[:mn][lo], sp[:, 1][lb][:len(np.asarray(o["V_RV"])[:mn][lo])], color=col, lw=1.4, alpha=0.85)
    for ax, t in zip(axes, ["LV", "RV"]):
        ax.set_title(t, fontweight="bold"); ax.set_xlabel("FEM cavity volume (mL)"); ax.set_ylabel("cavity pressure (mmHg)"); style(ax)
    sm = ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02, aspect=30); cb.set_label(SEV, fontsize=9)
    fig.suptitle("Coupled FEM cavity pressure-volume loops", fontsize=12)
    savefig(fig, out / "loops_pv_coupled")

def fig_pv_0d(df, bundle, out):
    """STANDALONE 0D PV loops from the pre-coupling warm-up (circulation/preload_history.npy),
    NOT the coupled ode_state — the 0D model run on its own, warmed up and converged.
    Last converged beat (RR = 0.8 s @ 75 bpm)."""
    sev = df["sev"]; norm = Normalize(sev.min(), sev.max())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=True)
    for i, c in enumerate(CASES):
        ph = SWEEP / bundle / c / "circulation" / "preload_history.npy"
        if not ph.exists():
            continue
        o = np.load(ph, allow_pickle=True).item()
        col = CMAP(norm(df["sev"][i]))
        t = np.asarray(o["time"]); m = t >= (t[-1] - 0.8)   # last converged warm-up beat
        axes[0].plot(np.asarray(o["V_LV"])[m], np.asarray(o["p_LV"])[m], color=col, lw=1.4, alpha=0.85)
        axes[1].plot(np.asarray(o["V_RV"])[m], np.asarray(o["p_RV"])[m], color=col, lw=1.4, alpha=0.85)
    for ax, lab in zip(axes, ["LV", "RV"]):
        ax.set_title(lab, fontweight="bold"); ax.set_xlabel("Volume (mL)"); ax.set_ylabel("0D pressure (mmHg)"); style(ax)
    sm = ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02, aspect=30); cb.set_label(SEV, fontsize=9)
    fig.suptitle("Standalone 0D circulation PV loops (pre-coupling)", fontsize=12)
    savefig(fig, out / "loops_pv_0d")

def fig_stress_pressure_strain(df, bundle, out, frame, band="full"):
    """3 rows (LV/RV/Septum) x 2 cols: fibre stress-strain | pressure-strain.
    band='mid' uses the AHA mid-ring region keys (Mid_LV/Mid_RV/Mid_Septum)."""
    sev = df["sev"]; norm = Normalize(sev.min(), sev.max())
    regs = ["LV", "RV", "Septum"]
    keyreg = {r: (f"Mid_{r}" if band == "mid" else r) for r in regs}
    pkey = {"LV": 0, "RV": 1, "Septum": 0}   # septum uses LV pressure axis for the PS panel
    fig, axes = plt.subplots(3, 2, figsize=(10.6, 12.5), constrained_layout=True)
    miss = False
    for i, c in enumerate(CASES):
        m, o, sp = load_loop(SWEEP / bundle / c)
        col = CMAP(norm(df["sev"][i]))
        nm = len(np.asarray(m["mean_E_ff_LV"])); lbm = last_beat_slice(nm)   # last beat only
        nsp = sp.shape[0]; lbp = last_beat_slice(nsp)
        for ri, reg in enumerate(regs):
            kr = keyreg[reg]
            if f"mean_E_ff_{kr}" not in m or np.allclose(np.asarray(m[f"mean_E_ff_{kr}"]), 0):
                miss = True; continue
            Eff = frame_strain(np.asarray(m[f"mean_E_ff_{kr}"])[lbm], frame)
            Sff = np.asarray(m[f"mean_S_ff_{kr}"])[lbm] * PA_TO_KPA
            Ell = frame_strain(np.asarray(m[f"mean_E_ll_{kr}"])[lbm], frame) * 100.0
            P = sp[:, pkey[reg]][lbp]
            L = min(len(Eff), len(P))
            axes[ri, 0].plot(Eff[:L], Sff[:L], color=col, lw=1.4, alpha=0.85)
            axes[ri, 1].plot(Ell[:L], P[:L], color=col, lw=1.4, alpha=0.85)
    if miss:
        print(f"   [fig_stress_pressure_strain band={band}] missing/zero Mid_* traces — "
              f"re-run postprocess_metrics with the AHA sidecar present")
    plab = {0: "$P_{LV}$", 1: "$P_{RV}$"}   # which cavity pressure drives each PS panel
    bl = "  (AHA mid ring)" if band == "mid" else ""
    axes[0, 0].set_title("Fibre stress-strain (model)" + bl, fontweight="bold")
    axes[0, 1].set_title("Pressure-longitudinal-strain (clinical proxy)" + bl, fontweight="bold")
    for ri, reg in enumerate(regs):
        pl = plab[pkey[reg]]
        axes[ri, 0].set_ylabel("$S_{ff}$ (kPa)"); axes[ri, 0].set_xlabel("$E_{ff}$")
        # the cavity pressure for this region is communicated by the y-axis label alone
        axes[ri, 1].set_ylabel(f"{pl} (mmHg)", fontsize=14, fontweight="bold")
        axes[ri, 1].set_xlabel(r"$\varepsilon_{ll}$ (%)")
        axes[ri, 0].annotate(keyreg[reg], xy=(-0.25, 0.5), xycoords="axes fraction", fontsize=13,
                             fontweight="bold", rotation=90, ha="center", va="center")
        for col_i in range(2): style(axes[ri, col_i])
    sm = ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02, aspect=40); cb.set_label(SEV, fontsize=9)
    savefig(fig, out / ("loops_stress_pressure_strain_mid" if band == "mid" else "loops_stress_pressure_strain"))

def fig_septum_candidate_loops(df, bundle, out, frame):
    """Septum only: fibre stress-strain loop + every candidate cavity-pressure
    pressure-strain loop (P_LV, P_RV, transmural, Mean, Sum) against the SAME
    septal longitudinal strain, so loop shape/area is comparable across choices."""
    sev = df["sev"]; norm = Normalize(sev.min(), sev.max())
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.6), constrained_layout=True)
    ax = axes.ravel()
    for i, c in enumerate(CASES):
        m, o, sp = load_loop(SWEEP / bundle / c)
        col = CMAP(norm(df["sev"][i]))
        nm = len(np.asarray(m["mean_E_ff_Septum"])); lbm = last_beat_slice(nm)
        lbp = last_beat_slice(sp.shape[0])
        Eff = frame_strain(np.asarray(m["mean_E_ff_Septum"])[lbm], frame)
        Sff = np.asarray(m["mean_S_ff_Septum"])[lbm] * PA_TO_KPA
        Ell = frame_strain(np.asarray(m["mean_E_ll_Septum"])[lbm], frame) * 100.0
        plv = sp[:, 0][lbp]; prv = sp[:, 1][lbp]
        L = min(len(Eff), len(plv))
        ax[0].plot(Eff[:L], Sff[:L], color=col, lw=1.4, alpha=0.85)
        series = [plv, prv, plv - prv, 0.5 * (plv + prv), plv + prv]
        for k, P in enumerate(series, start=1):
            ax[k].plot(Ell[:L], P[:L], color=col, lw=1.4, alpha=0.85)
    ax[0].set_title("Fibre stress-strain (model)", fontweight="bold")
    ax[0].set_xlabel("$E_{ff}$"); ax[0].set_ylabel("$S_{ff}$ (kPa)")
    titles = ["$P_{LV}$", "$P_{RV}$", "$P_{LV}-P_{RV}$ (transmural)", "Mean $\\frac{1}{2}(P_{LV}+P_{RV})$", "Sum $P_{LV}+P_{RV}$"]
    for k, t in enumerate(titles, start=1):
        ax[k].set_title(t, fontweight="bold")
        ax[k].set_xlabel(r"$\varepsilon_{ll}$ (%)"); ax[k].set_ylabel("pressure (mmHg)")
        ax[k].axhline(0, color="0.7", lw=0.7, ls=":")
    for a in ax:
        style(a)
    sm = ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02, aspect=40); cb.set_label(SEV, fontsize=9)
    savefig(fig, out / "loops_septum_candidates")

# --------------------------------------------------------------------------- #
def write_tables(df, allr, ddir):
    import csv
    # scalar per-case csv
    with open(ddir / "sweep_case_values.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["case", "sev"] + [f"{r}_{k}" for r in REG for k in
                ["W","Wff","PLV","PRV","Trans","Mean","Sum","Affine"]]
        w.writerow(cols)
        for i in range(len(df["case"])):
            w.writerow([df[c][i] for c in cols])
    # correlation table
    with open(ddir / "correlations.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["region","pressure","r"])
        for r in REG:
            for k, _ in PCHOICES:
                w.writerow([r, k, f"{allr[r][k]:+.4f}"])
    with open(ddir / "correlations.md", "w") as f:
        f.write("| region | " + " | ".join(k for k,_ in PCHOICES) + " |\n")
        f.write("|" + "---|"*(len(PCHOICES)+1) + "\n")
        for r in REG:
            f.write(f"| {r} | " + " | ".join(f"{allr[r][k]:+.3f}" for k,_ in PCHOICES) + " |\n")

def main():
    summary = {}
    for bundle, desc in BUNDLES.items():
        base = OUT / bundle
        cdir = base / "correlation"; rdir = base / "ratio"; qdir = base / "loops"; ddir = base / "data"
        bdir = base / "band_compare"
        for d in (cdir, rdir, qdir, ddir, bdir): d.mkdir(parents=True, exist_ok=True)
        df = aggregate(bundle)
        print(f"[{bundle}] n={len(df['case'])}  RV sys {df['sev'].min():.0f}-{df['sev'].max():.0f} mmHg")
        # Full LDRB region vs AHA mid ring (tags 4/5/6). Skip gracefully if the AHA
        # tags have not been backfilled yet (compute_aha_band.py / current per_cell run).
        try:
            df_mid = aggregate(bundle, band="mid")
            fig_band_compare(df, df_mid, bdir)
            # Mid-ring headline figures. Only the RV has real dynamic range here too
            # (work range ~65-95% of mean); LV and septum stay flat (~5-14%), so their
            # proxy r is the monotonic confound, not tracking — show RV correlation
            # only, and let work_dynamic_range / true_vs_proxy_range make the
            # "LV+septum are flat" point honestly.
            fig_region_correlation(df_mid, "RV", bdir)
            fig_true_vs_proxy_range(df_mid, bdir)
            fig_work_dynamic_range(df_mid, bdir)
            allr_mid = {region: {k: r_value(df_mid[f"{region}_{k}"], df_mid[f"{region}_W"])
                                 for k, _ in PCHOICES} for region in REG}
            write_tables(df_mid, allr_mid, bdir)
            print("  mid-ring r:")
            for r in REG:
                print("    %-7s "%r + "  ".join(f"{k}={allr_mid[r][k]:+.2f}" for k,_ in PCHOICES))
        except FileNotFoundError as e:
            print(f"  (skipping mid-ring band: {e})")
        # Correlations are only meaningful where true work has dynamic range. In this
        # pulmonary sweep only the RV does (range ~70-95% of mean); LV and septum vary
        # ~5-13% (CV ~2-4%) so their proxy r is noise and flips sign across bundles.
        # -> RV-focused correlation grid + one plot that shows LV/septum are flat.
        allr = {region: {k: r_value(df[f"{region}_{k}"], df[f"{region}_W"])
                         for k, _ in PCHOICES} for region in REG}
        fig_region_correlation(df, "RV", cdir)
        fig_work_dynamic_range(df, cdir)
        fig_true_vs_proxy_range(df, cdir)
        fig_direction_recovers(cdir, bundle)
        fig_ratio_spectrum(df, rdir); fig_ratio_scatter(df, rdir); fig_septum_ratio(df, rdir)
        fig_pv_coupled(df, bundle, qdir); fig_pv_0d(df, bundle, qdir)
        for frame in ("ED", "unloaded"):
            fdir2 = qdir / frame; fdir2.mkdir(parents=True, exist_ok=True)
            fig_stress_pressure_strain(df, bundle, fdir2, frame)
            fig_stress_pressure_strain(df, bundle, fdir2, frame, band="mid")
            fig_septum_candidate_loops(df, bundle, fdir2, frame)
        write_tables(df, allr, ddir)
        summary[bundle] = allr
        print(f"   correlation: {', '.join(f'{r}:best={max(PCHOICES,key=lambda kc: abs(allr[r][kc[0]]))[0]}' for r in REG)}")
    print("\n=== r summary (SS work vs P x eps_ll) ===")
    for b in BUNDLES:
        print(f"\n{b}")
        for r in REG:
            print("  %-7s "%r + "  ".join(f"{k}={summary[b][r][k]:+.2f}" for k,_ in PCHOICES))

if __name__ == "__main__":
    main()
