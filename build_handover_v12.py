#!/usr/bin/env python3
"""
build_handover_v12.py — build handover_lin or handover_exp from v12 FEM sims.

Usage:
  python build_handover_v12.py lin    # uses v12 linear params + FEM jobs 1046216-1046223
  python build_handover_v12.py exp    # uses v12 exp params + FEM jobs 1047450-1047457

Produces, under results/handover/handover_{lin,exp}/:
  hemodynamic_summary.csv
  data/
    circulation_params/     — 8 JSONs (renamed cN_optimized_*.json)
    circulation_timeseries/ — 8 CSVs (last-beat 0D state)
    solver_pressures/       — 8 CSVs (last-beat FEM cavity pressures)
    metrics/                — 8 metrics CSVs (mean fiber/longitudinal E,S)
    per_cell_data/          — 8 NPZ symlinks
  geometry/                 — symlinks to handover_old/geometry/ (same mesh)
  scripts/                  — copied from handover_old/scripts/
  spectrum_results/         — PV loops, scatter, sweep, loop overlays
  README.md
"""
import sys, json, csv, shutil
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

if len(sys.argv) != 2 or sys.argv[1] not in ("lin", "exp"):
    sys.exit("usage: build_handover_v12.py {lin|exp}")
VARIANT = sys.argv[1]

CWD = Path("/global/D1/homes/dtsteene/cardiac-work")
SIMS = CWD / "results/sims/2026-04-23"
OLD = CWD / "results/handover/handover_old"
OUT = CWD / f"results/handover/handover_{VARIANT}"

if VARIANT == "lin":
    PARAM_DIR = CWD / "data/ukb_circ_v12_linear"
    PARAM_FNAME = lambda sev: f"optimized_regazzoni_ukb_linear_{sev}.json"
    JOBS = dict(zip(
        ["sPAP22","sPAP30","sPAP45","sPAP55","sPAP65","sPAP75","sPAP85","sPAP95"],
        ["1046216","1046217","1046218","1046219","1046220","1046221","1046222","1046223"],
    ))
    EDPVR_LABEL = "linear (kE = 0)"
else:
    PARAM_DIR = CWD / "data/ukb_circ_v12_exp"
    PARAM_FNAME = lambda sev: f"optimized_regazzoni_ukb_{sev}.json"
    JOBS = dict(zip(
        ["sPAP22","sPAP30","sPAP45","sPAP55","sPAP65","sPAP75","sPAP85","sPAP95"],
        ["1047450","1047451","1047452","1047453","1047454","1047455","1047456","1047457"],
    ))
    EDPVR_LABEL = "exponential Klotz (kE free)"

# Ordered (sPAP22 → sPAP95). CID, archival_key, run_id, circ_json_path.
CASES = [(f"C{i+1}", sev, JOBS[sev], PARAM_DIR / PARAM_FNAME(sev))
         for i, sev in enumerate(JOBS.keys())]

# ── Prepare output tree ─────────────────────────────────────────────────────
for sub in ["circulation_params","circulation_timeseries","solver_pressures",
            "metrics","per_cell_data"]:
    (OUT / "data" / sub).mkdir(parents=True, exist_ok=True)
(OUT / "spectrum_results").mkdir(parents=True, exist_ok=True)
(OUT / "geometry").mkdir(parents=True, exist_ok=True)
(OUT / "scripts").mkdir(parents=True, exist_ok=True)

# ── Copy shared assets from handover_old ────────────────────────────────────
# scripts/ (same plotting code)
for p in (OLD / "scripts").iterdir():
    if p.is_file():
        shutil.copy2(p, OUT / "scripts" / p.name)

# geometry/ (same UKB mesh across both variants)
for p in (OLD / "geometry").iterdir():
    dst = OUT / "geometry" / p.name
    if not dst.exists():
        if p.is_file():
            shutil.copy2(p, dst)

# ── Load FEM run artefacts + compute summary ────────────────────────────────
KPA, PA_TO_KPA, MMHG_TO_KPA = 1e-3, 1e-3, 0.133322

pcs, metrics, solver_p, circ_hist, rvesp = {}, {}, {}, {}, {}
summary_rows = []

for cid, sev, rid, circ_path in CASES:
    rundir = SIMS / f"UKB_6beats_run_{rid}"
    pcs[cid] = np.load(rundir / "per_cell_data.npz", allow_pickle=True)
    metrics[cid] = np.load(rundir / "metrics" / "metrics_downsample_1.npy",
                            allow_pickle=True).item()
    solver_p[cid] = np.load(rundir / "solver" / "solver_cavity_pressure_mmHg.npy")
    circ_hist[cid] = np.load(rundir / "circulation" / "history.npy",
                              allow_pickle=True).item()

    sp = solver_p[cid]; h = circ_hist[cid]
    n_sp = sp.shape[0]; beat_sp = n_sp // 6; sp_last = sp[5*beat_sp:]
    n_0d = len(h["V_LV"]); beat_0d = n_0d // 6
    sl_0d = slice(5*beat_0d, 6*beat_0d)
    V_LV = np.array(h["V_LV"])[sl_0d]; V_RV = np.array(h["V_RV"])[sl_0d]
    rv_esp_val = float(sp_last[:, 1].max())
    rvesp[cid] = rv_esp_val

    summary_rows.append({
        "case_id": cid, "archival_key": sev, "run_id": rid,
        "RV_ESP_mmHg": round(rv_esp_val, 1),
        "RV_EDP_mmHg": round(float(sp_last[:, 1].min()), 1),
        "RV_EDV_mL":   round(float(V_RV.max()), 1),
        "RV_ESV_mL":   round(float(V_RV.min()), 1),
        "RV_SV_mL":    round(float(V_RV.max()-V_RV.min()), 1),
        "RV_EF_pct":   round(float((V_RV.max()-V_RV.min())/V_RV.max()*100), 1),
        "LV_ESP_mmHg": round(float(sp_last[:, 0].max()), 1),
        "LV_EDP_mmHg": round(float(sp_last[:, 0].min()), 1),
        "LV_EDV_mL":   round(float(V_LV.max()), 1),
        "LV_ESV_mL":   round(float(V_LV.min()), 1),
        "LV_SV_mL":    round(float(V_LV.max()-V_LV.min()), 1),
        "LV_EF_pct":   round(float((V_LV.max()-V_LV.min())/V_LV.max()*100), 1),
        "mPAP_mmHg":   round(float(np.array(h["p_AR_PUL"])[sl_0d].mean()), 1),
        "Ao_SBP_mmHg": round(float(np.array(h["p_AR_SYS"])[sl_0d].max()), 1),
        "Ao_DBP_mmHg": round(float(np.array(h["p_AR_SYS"])[sl_0d].min()), 1),
        "CO_Lmin":     round(float(V_RV.max()-V_RV.min())*75/1000, 2),
        "HR_bpm":      75,
    })

    # Write circulation time-series (last beat, 0D state)
    t_0d = np.arange(beat_0d) * 0.001
    circ_keys = [k for k in
        ["V_LA","V_LV","V_RA","V_RV","p_LA","p_LV","p_RA","p_RV",
         "p_AR_SYS","p_VEN_SYS","p_AR_PUL","p_VEN_PUL",
         "Q_MV","Q_AV","Q_TV","Q_PV","Q_AR_SYS","Q_VEN_SYS",
         "Q_AR_PUL","Q_VEN_PUL"] if k in h]
    with open(OUT / "data/circulation_timeseries" / f"{cid}_circulation.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["time_s"] + circ_keys)
        for i in range(beat_0d):
            w.writerow([f"{t_0d[i]:.4f}"] +
                       [f"{np.array(h[k])[5*beat_0d+i]:.4f}" for k in circ_keys])

    # Solver pressures (last beat)
    with open(OUT / "data/solver_pressures" / f"{cid}_solver_pressure_mmHg.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["time_s","P_LV_mmHg","P_RV_mmHg"])
        t_sp = np.arange(sp_last.shape[0]) * 0.001
        for i in range(sp_last.shape[0]):
            w.writerow([f"{t_sp[i]:.4f}", f"{sp_last[i,0]:.2f}", f"{sp_last[i,1]:.2f}"])

    # Regional metrics CSV (mean E_ff, E_ll, S_ff per region, last beat)
    m = metrics[cid]; t_m = np.array(m["time"]); n_m = len(t_m)
    regions = ["LV","RV","Septum"]
    cols = ["time_s"] + [f"{q}_{r}" for r in regions for q in
                         ("mean_E_ff","mean_E_ll","mean_S_ff_Pa")]
    with open(OUT / "data/metrics" / f"{cid}_metrics.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        beat_m = n_m // 6
        sl_m = slice(5*beat_m, 6*beat_m)
        tslice = t_m[sl_m]
        for i, ti in enumerate(tslice):
            row = [f"{ti:.4f}"]
            for r in regions:
                row += [f"{np.array(m.get(f'mean_E_ff_{r}', np.zeros(n_m)))[sl_m][i]:.6f}",
                        f"{np.array(m.get(f'mean_E_ll_{r}', np.zeros(n_m)))[sl_m][i]:.6f}",
                        f"{np.array(m.get(f'mean_S_ff_{r}', np.zeros(n_m)))[sl_m][i]:.2f}"]
            w.writerow(row)

    # Circulation params JSON
    if Path(circ_path).exists():
        shutil.copy2(circ_path, OUT / "data/circulation_params" / f"{cid}_{Path(circ_path).name}")

    # per_cell_data symlink (relative)
    pc_src = (rundir / "per_cell_data.npz").resolve()
    pc_dst = OUT / "data/per_cell_data" / f"{cid}_per_cell_data.npz"
    if pc_dst.is_symlink() or pc_dst.exists():
        pc_dst.unlink()
    pc_dst.symlink_to(pc_src)

# Sort summary by RV ESP
summary_rows.sort(key=lambda r: r["RV_ESP_mmHg"])
with open(OUT / "hemodynamic_summary.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    w.writeheader(); w.writerows(summary_rows)
print(f"[{VARIANT}] hemodynamic_summary.csv written ({len(summary_rows)} cases)")

# ── Plot helpers ────────────────────────────────────────────────────────────
REG_NAMES = ["LV","Septum","RV"]
PROXIES = [("PLV","#1f77b4","$P_{LV}$","o"),
           ("PRV","#d62728","$P_{RV}$","s"),
           ("Trans","#2ca02c","$P_{LV}-P_{RV}$","^")]

def densities_for_mask(mask_per_case):
    rows = []
    for cid, _, _, _ in CASES:
        pc = pcs[cid]; cv = pc["cell_volumes"]; mask = mask_per_case[cid]
        if not mask.any(): return None
        V = cv[mask].sum()
        rows.append({
            "W":     pc["w_total"][mask].sum() / V * KPA,
            "PLV":   pc["proxy_PLV_ll"][mask].sum() / V * KPA,
            "PRV":   pc["proxy_PRV_ll"][mask].sum() / V * KPA,
            "Trans": (pc["proxy_PLV_ll"][mask].sum() - pc["proxy_PRV_ll"][mask].sum()) / V * KPA,
        })
    return rows

def compute_metrics(rows, pk):
    xs = np.array([r[pk] for r in rows]); ys = np.array([r["W"] for r in rows])
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan"), float("nan"), float("nan")
    r_val = pearsonr(xs, ys)[0]
    a = np.sum(xs*ys) / np.sum(xs**2) if np.sum(xs**2) > 0 else 0
    ss_res = np.sum((ys - a*xs)**2); ss_tot = np.sum((ys - ys.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    return r_val, r2, abs(r_val) * max(r2, 0)

# ── Loop overlays ──────────────────────────────────────────────────────────
cmap = plt.cm.coolwarm
norm = Normalize(vmin=25, vmax=95)

for proxy_mode, fname in [("PLV","loops_plv.png"), ("Trans","loops_regional.png")]:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for cid, _, _, _ in CASES:
        color = cmap(norm(rvesp[cid]))
        m = metrics[cid]; t = np.array(m["time"]); n = len(t)
        sp = solver_p[cid]
        t_pres = np.linspace(0, (sp.shape[0]-1)*0.001, sp.shape[0])
        P_LV = np.interp(t, t_pres, sp[:, 0]) * MMHG_TO_KPA
        P_RV = np.interp(t, t_pres, sp[:, 1]) * MMHG_TO_KPA
        for col, rmet in enumerate(REG_NAMES):
            E_ff = np.array(m.get(f"mean_E_ff_{rmet}", np.zeros(n)))
            E_ll = np.array(m.get(f"mean_E_ll_{rmet}", np.zeros(n)))
            S_ff = np.array(m.get(f"mean_S_ff_{rmet}", np.zeros(n))) * PA_TO_KPA
            axes[0, col].plot(E_ff, S_ff, color=color, lw=1.5, alpha=0.85)
            if rmet == "LV": P = P_LV
            elif rmet == "RV": P = P_RV if proxy_mode == "Trans" else P_LV
            else: P = (P_LV - P_RV) if proxy_mode == "Trans" else P_LV
            axes[1, col].plot(E_ll, P, color=color, lw=1.5, alpha=0.85)
    for col, rmet in enumerate(REG_NAMES):
        axes[0, col].set_title(rmet, fontsize=12, fontweight="bold")
        axes[0, col].set_xlabel(r"$E_{ff}$"); axes[0, col].set_ylabel(r"$S_{ff}$ (kPa)")
        axes[0, col].grid(alpha=0.25); axes[0, col].axhline(0, color="gray", lw=0.5)
        axes[0, col].axvline(0, color="gray", lw=0.5)
        axes[1, col].set_xlabel(r"$\varepsilon_{ll}$"); axes[1, col].grid(alpha=0.25)
        axes[1, col].axhline(0, color="gray", lw=0.5); axes[1, col].axvline(0, color="gray", lw=0.5)
    if proxy_mode == "PLV":
        for col in range(3): axes[1, col].set_ylabel(r"$P_{LV}$ (kPa)")
    else:
        axes[1, 0].set_ylabel(r"$P_{LV}$ (kPa)")
        axes[1, 1].set_ylabel(r"$P_{LV}-P_{RV}$ (kPa)")
        axes[1, 2].set_ylabel(r"$P_{RV}$ (kPa)")
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("RV$_{ESP}$ (mmHg)")
    fig.suptitle(f"v12 {VARIANT.upper()} — top: S:E  |  bottom: pressure × longitudinal strain",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / "spectrum_results" / fname, dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "spectrum_results" / fname.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig); print(f"[{VARIANT}] {fname}")

# ── 3x3 scatter ────────────────────────────────────────────────────────────
geo_masks = {cid: pcs[cid]["is_geometric_septum"].astype(bool) for cid,_,_,_ in CASES}
lv_masks  = {cid: pcs[cid]["region_tags"] == 1 for cid,_,_,_ in CASES}
rv_masks  = {cid: pcs[cid]["region_tags"] == 2 for cid,_,_,_ in CASES}
all_region_masks = [lv_masks, geo_masks, rv_masks]

fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
for row, (rname, rmasks) in enumerate(zip(REG_NAMES, all_region_masks)):
    rows = densities_for_mask(rmasks)
    for col, (pk, color, label, marker) in enumerate(PROXIES):
        ax = axes[row, col]
        xs = np.array([r[pk] for r in rows]); ys = np.array([r["W"] for r in rows])
        ax.scatter(xs, ys, c=color, marker=marker, s=80, edgecolors="white",
                   linewidth=0.8, zorder=3)
        for c, xv, yv in zip(CASES, xs, ys):
            ax.annotate(c[0], (xv, yv), fontsize=7, ha="left",
                        xytext=(5,3), textcoords="offset points", color="gray")
        r_val, r2, Q = compute_metrics(rows, pk)
        a = np.sum(xs*ys) / np.sum(xs**2) if np.sum(xs**2) > 0 else 0
        xr = np.linspace(min(xs.min(), 0), max(xs.max(), 0), 20)
        ax.plot(xr, a*xr, "k--", lw=1.5, alpha=0.6)
        ax.set_title(f"r={r_val:+.2f}  R²={r2:.2f}  Q={Q:.2f}", fontsize=10)
        ax.grid(alpha=0.25); ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        if col == 0: ax.set_ylabel(f"{rname}\n$W_{{true}}$ density (kPa)")
        if row == 2: ax.set_xlabel(f"{label} density (kPa)")
        if row == 0: ax.text(0.5, 1.15, label, transform=ax.transAxes,
                             fontsize=12, fontweight="bold", ha="center", color=color)
fig.suptitle(f"v12 {VARIANT.upper()} — proxy vs truth (geometric septum, 8 cases)",
             fontsize=13, fontweight="bold")
fig.savefig(OUT / "spectrum_results" / "scatter_spectrum.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "spectrum_results" / "scatter_spectrum.pdf", bbox_inches="tight")
plt.close(fig); print(f"[{VARIANT}] scatter_spectrum.png")

# ── PV loops (0D and solver) ────────────────────────────────────────────────
for src_label, src_file, fname in [
    ("0D", "circulation_timeseries", "pv_loops_0d.png"),
    ("Solver", "solver_pressures", "pv_loops_solver.png"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for cid, _, _, _ in CASES:
        color = cmap(norm(rvesp[cid]))
        if src_label == "0D":
            h = circ_hist[cid]
            n_0d = len(h["V_LV"]); beat = n_0d // 6; sl = slice(5*beat, 6*beat)
            V_LV = np.array(h["V_LV"])[sl]; V_RV = np.array(h["V_RV"])[sl]
            P_LV = np.array(h["p_LV"])[sl]; P_RV = np.array(h["p_RV"])[sl]
        else:
            sp = solver_p[cid]; beat = sp.shape[0] // 6; sp_last = sp[5*beat:]
            h = circ_hist[cid]; n_0d = len(h["V_LV"]); beat_0d = n_0d // 6
            sl_0d = slice(5*beat_0d, 6*beat_0d)
            V_LV = np.array(h["V_LV"])[sl_0d]; V_RV = np.array(h["V_RV"])[sl_0d]
            t_pres = np.linspace(0, (sp_last.shape[0]-1)*0.001, sp_last.shape[0])
            t_0d = np.arange(beat_0d) * 0.001
            P_LV = np.interp(t_0d, t_pres, sp_last[:, 0])
            P_RV = np.interp(t_0d, t_pres, sp_last[:, 1])
        axes[0].plot(V_LV, P_LV, color=color, lw=1.3, alpha=0.85)
        axes[1].plot(V_RV, P_RV, color=color, lw=1.3, alpha=0.85)
    for ax, name in zip(axes, ["LV","RV"]):
        ax.set_xlabel("Volume (mL)"); ax.set_ylabel(f"P_{name} (mmHg)")
        ax.set_title(f"{name} PV loop", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("RV$_{ESP}$ (mmHg)")
    fig.suptitle(f"v12 {VARIANT.upper()} — PV loops ({src_label})",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / "spectrum_results" / fname, dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "spectrum_results" / fname.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig); print(f"[{VARIANT}] {fname}")

# ── README.md ──────────────────────────────────────────────────────────────
readme = f"""# Handover — v12 {VARIANT.upper()} EDPVR

Eight biventricular FEM simulations with v12 circulation parameters, using
**{EDPVR_LABEL}** end-diastolic pressure–volume relations.

## Cases

Cases are positional identifiers (`sPAP22` … `sPAP95`) — the number is the
0D-target systolic pulmonary artery pressure in mmHg. See `../docs/circulation_pah_targets.md`
for full target provenance and a side-by-side LIN/EXP achieved-hemodynamics
comparison.

## Directory layout

    hemodynamic_summary.csv    — one row per case, ordered by achieved RV_ESP
    data/
      circulation_params/      — Regazzoni 2020 parameter JSON per case
      circulation_timeseries/  — 0D state per case, last beat (1 ms sampling)
      solver_pressures/        — FEM cavity Lagrange multiplier P_LV, P_RV
      metrics/                 — regional mean E_ff, E_ll, S_ff per case
      per_cell_data/           — per-cell w_total + proxy values (symlinks)
    geometry/                  — canonical UKB mesh + septum definition (shared)
    scripts/                   — plotting scripts (shared)
    spectrum_results/          — loop overlays, scatter, PV loops

Both handover_lin and handover_exp share the same mesh, same fibers, same
active tension waveform; they differ only in the 0D circulation parameter
sets tuned with or without the exponential EDPVR term.
"""
with open(OUT / "README.md","w") as f: f.write(readme)

# ── Summary print ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"Handover build — v12 {VARIANT.upper()} — SUMMARY")
print("=" * 70)
for rname, rmasks in zip(REG_NAMES, all_region_masks):
    rws = densities_for_mask(rmasks)
    print(f"  {rname}:")
    for pk, _, label, _ in PROXIES:
        r_val, r2, Q = compute_metrics(rws, pk)
        print(f"    {label:<18} r={r_val:+.2f}  R²={r2:+.2f}  Q={Q:.3f}")
print(f"\nHandover saved to {OUT}/")
