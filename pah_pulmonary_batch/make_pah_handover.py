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
  <bundle>/circulation/   Qualitative: 0D PV loops, coupled PV loops, coupled
                          fibre stress-strain and pressure-strain loops.
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
SWEEP = paths.RESULTS_ROOT / "sims/2026-06-09/pah_pulmonary_20260609_prodsweep"
OUT = paths.RESULTS_ROOT / "handover/pah_pulmonary_paper_20260611"
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

# pressure choices for the correlation grids (longitudinal strain)
PCHOICES = [
    ("PLV",   r"$P_{LV}\times\varepsilon_{ll}$"),
    ("PRV",   r"$P_{RV}\times\varepsilon_{ll}$"),
    ("Trans", r"$(P_{LV}{-}P_{RV})\times\varepsilon_{ll}$"),
    ("Mean",  r"$\frac{1}{2}(P_{LV}{+}P_{RV})\times\varepsilon_{ll}$"),
    ("Sum",   r"$(P_{LV}{+}P_{RV})\times\varepsilon_{ll}$"),
    ("Affine",r"affine $P(\lambda)\times\varepsilon_{ll}$"),
]

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

def aggregate(bundle):
    """Per-case, per-region true work + the 6 ll pressure-strain proxies.
    Work reported positive (negate the dW-convention sum)."""
    rows = []
    for c in CASES:
        cd = SWEEP / bundle / c
        z = np.load(cd / "per_cell_data.npz", allow_pickle=True)
        tags = z["region_tags"]
        dlv, drv = z["d_lv"], z["d_rv"]
        lam = dlv / (dlv + drv + 1e-30)                       # 0 at LV face -> 1 at RV face
        affine = (1 - lam) * z["proxy_PLV_ll"] + lam * z["proxy_PRV_ll"]
        row = {"case": c, "sev": rv_systolic(cd)}
        for r, tag in REG.items():
            m = tags == tag
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

def scatter_fit(ax, x, y, sev, norm):
    ax.scatter(x, y, c=sev, cmap=CMAP, norm=norm, s=80, ec="k", lw=0.6, zorder=3)
    if np.std(x) > 0:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, a + b * xs, "-", color="0.35", lw=1.3, zorder=2)
    r = r_value(x, y)
    ax.set_title(f"r = {r:+.3f}", fontsize=11, fontweight="bold")

# --------------------------------------------------------------------------- #
# Correlation (headline)
# --------------------------------------------------------------------------- #
def fig_region_correlation(df, region, out):
    """One region: 2x3 grid of true SS work vs each ll pressure-strain proxy."""
    sev = df["sev"]; norm = Normalize(sev.min(), sev.max())
    y = df[f"{region}_W"]
    rs = {k: r_value(df[f"{region}_{k}"], y) for k, _ in PCHOICES}
    best = max(PCHOICES, key=lambda kc: abs(rs[kc[0]]) if not np.isnan(rs[kc[0]]) else -1)[0]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4), constrained_layout=True)
    for ax, (key, lab) in zip(axes.ravel(), PCHOICES):
        if key == best:
            ax.set_facecolor(HILITE)
        scatter_fit(ax, df[f"{region}_{key}"], y, sev, norm)
        ax.set_xlabel(lab + "  (proxy work, a.u.)")
        ax.set_ylabel(f"SS work  $\\oint S{{:}}dE$  ({region})")
        style(ax)
    sm = ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.02, aspect=30); cb.set_label(SEV, fontsize=9)
    fig.suptitle(f"{region}: stress-strain work vs longitudinal pressure-strain proxy "
                 f"(n=8 sweep)   best: {best}", fontsize=13)
    savefig(fig, out / f"correlation_{region}")
    return rs

def fig_r_summary(allr, out):
    """Bar chart: Pearson r per pressure choice, grouped by region."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    keys = [k for k, _ in PCHOICES]
    width = 0.26
    xpos = np.arange(len(keys))
    colors = {"LV": "#4575b4", "RV": "#d73027", "Septum": "#762a83"}
    for i, region in enumerate(REG):
        vals = [allr[region][k] for k in keys]
        ax.bar(xpos + (i - 1) * width, vals, width, label=region, color=colors[region], alpha=0.9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(keys)
    ax.set_ylabel("Pearson r  (SS work vs $P\\times\\varepsilon_{ll}$)")
    ax.set_ylim(-1, 1.05); ax.legend(frameon=False)
    ax.set_title("Correlation of true work with each pressure choice, by region")
    style(fig.axes[0])
    savefig(fig, out / "r_summary")

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
        ax.set_title(t, fontweight="bold"); ax.set_xlabel("Volume (mL)"); ax.set_ylabel("cavity pressure (mmHg)"); style(ax)
    sm = ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02, aspect=30); cb.set_label(SEV, fontsize=9)
    fig.suptitle("Coupled FEM cavity pressure-volume loops", fontsize=12)
    savefig(fig, out / "loops_pv_coupled")

def fig_pv_0d(df, bundle, out):
    sev = df["sev"]; norm = Normalize(sev.min(), sev.max())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=True)
    for i, c in enumerate(CASES):
        _, o, _ = load_loop(SWEEP / bundle / c)
        col = CMAP(norm(df["sev"][i]))
        n = len(o["V_LV"]); lb = last_beat_slice(n)
        axes[0].plot(np.asarray(o["V_LV"])[lb], np.asarray(o["p_LV"])[lb], color=col, lw=1.4, alpha=0.85)
        axes[1].plot(np.asarray(o["V_RV"])[lb], np.asarray(o["p_RV"])[lb], color=col, lw=1.4, alpha=0.85)
    for ax, t in zip(axes, ["LV", "RV"]):
        ax.set_title(t, fontweight="bold"); ax.set_xlabel("Volume (mL)"); ax.set_ylabel("0D pressure (mmHg)"); style(ax)
    sm = ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02, aspect=30); cb.set_label(SEV, fontsize=9)
    fig.suptitle("0D circulation-model pressure-volume loops", fontsize=12)
    savefig(fig, out / "loops_pv_0d")

def fig_stress_pressure_strain(df, bundle, out):
    """3 rows (LV/RV/Septum) x 2 cols: fibre stress-strain | pressure-strain. Strain zeroed at ED."""
    sev = df["sev"]; norm = Normalize(sev.min(), sev.max())
    regs = ["LV", "RV", "Septum"]
    pkey = {"LV": 0, "RV": 1, "Septum": 0}   # septum uses LV pressure axis for the PS panel
    fig, axes = plt.subplots(3, 2, figsize=(10.6, 12.5), constrained_layout=True)
    for i, c in enumerate(CASES):
        m, o, sp = load_loop(SWEEP / bundle / c)
        col = CMAP(norm(df["sev"][i]))
        nm = len(np.asarray(m["mean_E_ff_LV"])); lbm = last_beat_slice(nm)   # last beat only
        nsp = sp.shape[0]; lbp = last_beat_slice(nsp)
        for ri, reg in enumerate(regs):
            Eff = np.asarray(m[f"mean_E_ff_{reg}"])[lbm]; Eff = Eff - Eff.max()
            Sff = np.asarray(m[f"mean_S_ff_{reg}"])[lbm] * PA_TO_KPA
            Ell = np.asarray(m[f"mean_E_ll_{reg}"])[lbm]; Ell = (Ell - Ell.max()) * 100.0
            P = sp[:, pkey[reg]][lbp]
            L = min(len(Eff), len(P))
            axes[ri, 0].plot(Eff[:L], Sff[:L], color=col, lw=1.4, alpha=0.85)
            axes[ri, 1].plot(Ell[:L], P[:L], color=col, lw=1.4, alpha=0.85)
    axes[0, 0].set_title("Fibre stress-strain (model)", fontweight="bold")
    axes[0, 1].set_title("Pressure-longitudinal-strain (clinical proxy)", fontweight="bold")
    for ri, reg in enumerate(regs):
        axes[ri, 0].set_ylabel("$S_{ff}$ (kPa)"); axes[ri, 0].set_xlabel("$E_{ff}$")
        axes[ri, 1].set_ylabel("cavity P (mmHg)"); axes[ri, 1].set_xlabel("$\\varepsilon_{ll}$ (%)")
        axes[ri, 0].annotate(reg, xy=(-0.25, 0.5), xycoords="axes fraction", fontsize=13,
                             fontweight="bold", rotation=90, ha="center", va="center")
        for col_i in range(2): style(axes[ri, col_i])
    sm = ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02, aspect=40); cb.set_label(SEV, fontsize=9)
    fig.suptitle("Coupled fibre stress-strain and pressure-strain loops (strain zeroed at ED)", fontsize=12)
    savefig(fig, out / "loops_stress_pressure_strain")

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
        cdir = base / "correlation"; rdir = base / "ratio"; qdir = base / "circulation"; ddir = base / "data"
        for d in (cdir, rdir, qdir, ddir): d.mkdir(parents=True, exist_ok=True)
        df = aggregate(bundle)
        print(f"[{bundle}] n={len(df['case'])}  RV sys {df['sev'].min():.0f}-{df['sev'].max():.0f} mmHg")
        allr = {}
        for region in REG:
            allr[region] = fig_region_correlation(df, region, cdir)
        fig_r_summary(allr, cdir)
        fig_ratio_spectrum(df, rdir); fig_ratio_scatter(df, rdir); fig_septum_ratio(df, rdir)
        fig_pv_coupled(df, bundle, qdir); fig_pv_0d(df, bundle, qdir); fig_stress_pressure_strain(df, bundle, qdir)
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
