#!/usr/bin/env python3
"""
build_handover_geometric.py — Full handover with geometric septum definition.

Produces:
  results/handover_geometric/
    hemodynamic_summary.csv
    circulation_params/
    circulation_timeseries/
    solver_pressures/
    per_cell_data/
    figures/
      loops_plv.png / loops_trans.png
      sweep_r.png        — Pearson r across septum boundary
      sweep_r2.png       — through-origin R² across septum boundary
      sweep_Q.png        — combined Q = |r| × max(R², 0) across boundary
      scatter_spectrum.png — 3×3 proxy vs truth (geometric septum)
    README.md
"""
import numpy as np
import json
import csv
import shutil
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path("results/sims/2026-04-12")
OUT = Path("results/handover_geometric")
OUT.mkdir(parents=True, exist_ok=True)
for sub in ["circulation_params", "circulation_timeseries", "solver_pressures",
            "per_cell_data", "figures"]:
    (OUT / sub).mkdir(exist_ok=True)

CASES = [
    ("C1", "healthy",         "1020849", "data/ukb_circ_v2/optimized_regazzoni_ukb_healthy.json"),
    ("C2", "mild",            "1020851", "data/ukb_circ_v2/optimized_regazzoni_ukb_mild.json"),
    ("C3", "moderate",        "1020852", "data/ukb_circ_v2/optimized_regazzoni_ukb_moderate.json"),
    ("C4", "severe",          "1020854", "data/ukb_circ_v2/optimized_regazzoni_ukb_severe.json"),
    ("C5", "moderate_severe", "1020853", "data/ukb_circ_v2/optimized_regazzoni_ukb_moderate_severe.json"),
    ("C6", "very_severe",     "1020855", "data/ukb_circ_v2/optimized_regazzoni_ukb_very_severe.json"),
    ("C7", "end_stage",       "1020856", "data/ukb_circ_v2/optimized_regazzoni_ukb_end_stage.json"),
]

KPA = 1e-3
PA_TO_KPA = 1e-3
MMHG_TO_KPA = 0.133322

REG_NAMES = ["LV", "Septum", "RV"]
PROXIES = [
    ("PLV",   "#1f77b4", "$P_{LV}$",         "o"),
    ("PRV",   "#d62728", "$P_{RV}$",         "s"),
    ("Trans", "#2ca02c", "$P_{LV}-P_{RV}$",  "^"),
]

# ── Load everything ─────────────────────────────────────────────────────────
pcs = {}
metrics = {}
solver_p = {}
circ_hist = {}
rvesp = {}

summary_rows = []

for cid, sev, rid, circ_path in CASES:
    d = ROOT / f"UKB_6beats_run_{rid}"
    pcs[cid] = np.load(d / "per_cell_data.npz", allow_pickle=True)
    metrics[cid] = np.load(d / "metrics" / "metrics_downsample_1.npy", allow_pickle=True).item()
    solver_p[cid] = np.load(d / "solver" / "solver_cavity_pressure_mmHg.npy")
    circ_hist[cid] = np.load(d / "circulation" / "history.npy", allow_pickle=True).item()
    params = json.load(open(d / "simulation_params.json"))

    sp = solver_p[cid]
    h = circ_hist[cid]
    n_sp = sp.shape[0]; beat_sp = n_sp // 6; sp_last = sp[5*beat_sp:]
    n_0d = len(h["V_LV"]); beat_0d = n_0d // 6; sl_0d = slice(5*beat_0d, 6*beat_0d)

    V_LV = np.array(h["V_LV"])[sl_0d]; V_RV = np.array(h["V_RV"])[sl_0d]
    rv_esp_val = float(sp_last[:, 1].max())
    rvesp[cid] = rv_esp_val

    summary_rows.append({
        "case_id": cid, "archival_key": sev, "run_id": rid,
        "RV_ESP_mmHg": round(rv_esp_val, 1),
        "RV_EDP_mmHg": round(float(sp_last[:, 1].min()), 1),
        "RV_EDV_mL": round(float(V_RV.max()), 1),
        "RV_ESV_mL": round(float(V_RV.min()), 1),
        "RV_SV_mL": round(float(V_RV.max() - V_RV.min()), 1),
        "RV_EF_pct": round(float((V_RV.max()-V_RV.min())/V_RV.max()*100), 1),
        "LV_ESP_mmHg": round(float(sp_last[:, 0].max()), 1),
        "LV_EDP_mmHg": round(float(sp_last[:, 0].min()), 1),
        "LV_EDV_mL": round(float(V_LV.max()), 1),
        "LV_ESV_mL": round(float(V_LV.min()), 1),
        "LV_SV_mL": round(float(V_LV.max() - V_LV.min()), 1),
        "LV_EF_pct": round(float((V_LV.max()-V_LV.min())/V_LV.max()*100), 1),
        "mPAP_mmHg": round(float(np.array(h["p_AR_PUL"])[sl_0d].mean()), 1),
        "Ao_SBP_mmHg": round(float(np.array(h["p_AR_SYS"])[sl_0d].max()), 1),
        "Ao_DBP_mmHg": round(float(np.array(h["p_AR_SYS"])[sl_0d].min()), 1),
        "CO_Lmin": round(float(V_RV.max()-V_RV.min()) * 75 / 1000, 2),
        "HR_bpm": 75,
    })

    # Export circulation time-series (last beat)
    beat_0d_len = beat_0d
    t_0d = np.arange(beat_0d_len) * 0.001
    circ_keys = [k for k in ["V_LA","V_LV","V_RA","V_RV","p_LA","p_LV","p_RA","p_RV",
                              "p_AR_SYS","p_VEN_SYS","p_AR_PUL","p_VEN_PUL",
                              "Q_MV","Q_AV","Q_TV","Q_PV","Q_AR_SYS","Q_VEN_SYS",
                              "Q_AR_PUL","Q_VEN_PUL"] if k in h]
    with open(OUT / "circulation_timeseries" / f"{cid}_circulation.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time_s"] + circ_keys)
        for i in range(beat_0d_len):
            w.writerow([f"{t_0d[i]:.4f}"] + [f"{np.array(h[k])[5*beat_0d+i]:.4f}" for k in circ_keys])

    # Solver pressures
    with open(OUT / "solver_pressures" / f"{cid}_solver_pressure_mmHg.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time_s", "P_LV_mmHg", "P_RV_mmHg"])
        t_sp = np.arange(sp_last.shape[0]) * 0.001
        for i in range(sp_last.shape[0]):
            w.writerow([f"{t_sp[i]:.4f}", f"{sp_last[i,0]:.2f}", f"{sp_last[i,1]:.2f}"])

    # Copy circ params
    src = Path(circ_path)
    if src.exists():
        shutil.copy2(src, OUT / "circulation_params" / f"{cid}_{src.name}")

    # Symlink per_cell_data
    pc_dst = OUT / "per_cell_data" / f"{cid}_per_cell_data.npz"
    if not pc_dst.exists():
        pc_dst.symlink_to((d / "per_cell_data.npz").resolve())

# Sort summary by RV ESP
summary_rows.sort(key=lambda r: r["RV_ESP_mmHg"])
with open(OUT / "hemodynamic_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_rows[0].keys()); w.writeheader(); w.writerows(summary_rows)
print("Hemodynamic summary saved")

# ── Helpers ─────────────────────────────────────────────────────────────────
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
    Q = abs(r_val) * max(r2, 0)
    return r_val, r2, Q

# ── Septum boundary sweep (80-120% of geometric) ───────────────────────────
ref = pcs["C1"]
entry_t = ref["entry_t"]; envelope = ref["envelope"]
n_geo = ref["is_geometric_septum"].astype(bool).sum()
entry_t_env = entry_t[envelope]
sorted_t = np.sort(entry_t_env)

fractions = np.linspace(0.80, 1.20, 25)
sweep_data = {pk: {"r": [], "r2": [], "Q": []} for pk in ["PLV", "PRV", "Trans"]}
sweep_frac = []
sweep_n = []

for frac in fractions:
    target = int(n_geo * frac)
    idx = min(max(target - 1, 0), len(sorted_t) - 1)
    t_val = sorted_t[idx]
    mask = (entry_t <= t_val) & envelope
    n_cells = int(mask.sum())
    sweep_n.append(n_cells)
    sweep_frac.append(frac)

    masks = {cid: mask for cid, _, _, _ in CASES}
    rows = densities_for_mask(masks)
    if rows is None:
        for pk in ["PLV", "PRV", "Trans"]:
            for k in ["r", "r2", "Q"]: sweep_data[pk][k].append(float("nan"))
        continue
    for pk in ["PLV", "PRV", "Trans"]:
        r_val, r2, Q = compute_metrics(rows, pk)
        sweep_data[pk]["r"].append(r_val)
        sweep_data[pk]["r2"].append(r2)
        sweep_data[pk]["Q"].append(Q)

# Also compute LV and RV fixed Q for reference
lv_rows = densities_for_mask({cid: pcs[cid]["region_tags"] == 1 for cid, _, _, _ in CASES})
rv_rows = densities_for_mask({cid: pcs[cid]["region_tags"] == 2 for cid, _, _, _ in CASES})
_, _, q_lv_plv = compute_metrics(lv_rows, "PLV")
_, _, q_rv_prv = compute_metrics(rv_rows, "PRV")

# ── Sweep figures ───────────────────────────────────────────────────────────
frac_pct = [f*100 for f in sweep_frac]

for metric_key, ylabel, title_metric, fname in [
    ("r",  "Pearson r", "Directional tracking", "sweep_r.png"),
    ("r2", "$R^2_{origin}$", "Proportional fit (through-origin)", "sweep_r2.png"),
    ("Q",  "$Q = |r| \\times \\max(R^2, 0)$", "Combined quality", "sweep_Q.png"),
]:
    fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(11, 6.5),
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True, sharex=True)

    for pk, color, label, _ in PROXIES:
        ax.plot(frac_pct, sweep_data[pk][metric_key], "-o", color=color,
                lw=2.2, ms=5, label=label)

    ax.axvline(100, color="gray", ls="--", lw=1.0, alpha=0.5)
    ax.text(100.5, ax.get_ylim()[1]*0.95, "geometric\nbaseline", fontsize=8,
            color="gray", va="top")

    if metric_key == "Q":
        ax.axhline(q_lv_plv, color="#1f77b4", ls=":", lw=1.0, alpha=0.5)
        ax.text(121, q_lv_plv, f" LV×$P_{{LV}}$={q_lv_plv:.2f}", fontsize=8, color="#1f77b4", va="center")
        ax.axhline(q_rv_prv, color="#d62728", ls=":", lw=1.0, alpha=0.5)
        ax.text(121, q_rv_prv, f" RV×$P_{{RV}}$={q_rv_prv:.2f}", fontsize=8, color="#d62728", va="center")
        ax.set_ylim(-0.05, 1.05)
    elif metric_key == "r":
        ax.set_ylim(-1.05, 1.1)
        ax.axhline(0, color="lightgray", lw=0.5)
    elif metric_key == "r2":
        ax.axhline(0, color="lightgray", lw=0.5)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, loc="best")
    ax.set_title(f"Septum: {title_metric} across boundary definition\n"
                 f"x-axis = septum size as % of geometric baseline ({n_geo} cells)",
                 fontsize=12, fontweight="bold")

    ax_n.fill_between(frac_pct, sweep_n, color="lightgray", alpha=0.7)
    ax_n.set_ylabel("Cells", fontsize=10)
    ax_n.set_xlabel("Septum size (% of geometric baseline)", fontsize=10)
    ax_n.grid(alpha=0.25)

    fig.savefig(OUT / "figures" / fname, dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "figures" / fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

# ── Loop overlays (geometric septum, RV_ESP colorbar) ───────────────────────
cmap = plt.cm.coolwarm
norm = Normalize(vmin=30, vmax=90)

for proxy_mode, fname in [("PLV", "loops_plv.png"), ("Trans", "loops_trans.png")]:
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
        axes[0, col].set_xlabel(r"$E_{ff}$", fontsize=10)
        axes[0, col].set_ylabel(r"$S_{ff}$ (kPa)", fontsize=10)
        axes[0, col].grid(alpha=0.25); axes[0, col].axhline(0, color="gray", lw=0.5); axes[0, col].axvline(0, color="gray", lw=0.5)
        axes[1, col].set_xlabel(r"$\varepsilon_{ll}$", fontsize=10)
        axes[1, col].grid(alpha=0.25); axes[1, col].axhline(0, color="gray", lw=0.5); axes[1, col].axvline(0, color="gray", lw=0.5)
    if proxy_mode == "PLV":
        for col in range(3): axes[1, col].set_ylabel(r"$P_{LV}$ (kPa)", fontsize=10)
    else:
        axes[1, 0].set_ylabel(r"$P_{LV}$ (kPa)", fontsize=10)
        axes[1, 1].set_ylabel(r"$P_{LV}-P_{RV}$ (kPa)", fontsize=10)
        axes[1, 2].set_ylabel(r"$P_{RV}$ (kPa)", fontsize=10)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label(r"RV$_{ESP}$ (mmHg)", fontsize=10)
    fig.suptitle("Top: fiber stress-strain    Bottom: pressure × longitudinal strain", fontsize=12, fontweight="bold")
    fig.savefig(OUT / "figures" / fname, dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "figures" / fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig); print(f"Saved {fname}")

# ── 3x3 scatter (geometric septum) ─────────────────────────────────────────
geo_masks = {cid: pcs[cid]["is_geometric_septum"].astype(bool) for cid, _, _, _ in CASES}
lv_masks = {cid: pcs[cid]["region_tags"] == 1 for cid, _, _, _ in CASES}
rv_masks = {cid: pcs[cid]["region_tags"] == 2 for cid, _, _, _ in CASES}
all_region_masks = [lv_masks, geo_masks, rv_masks]

fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
for row, (rname, rmasks) in enumerate(zip(REG_NAMES, all_region_masks)):
    rows = densities_for_mask(rmasks)
    for col, (pk, color, label, marker) in enumerate(PROXIES):
        ax = axes[row, col]
        xs = np.array([r[pk] for r in rows]); ys = np.array([r["W"] for r in rows])
        ax.scatter(xs, ys, c=color, marker=marker, s=80, edgecolors="white", linewidth=0.8, zorder=3)
        for c, xv, yv in zip(CASES, xs, ys):
            ax.annotate(c[0], (xv, yv), fontsize=7, ha="left", xytext=(5,3), textcoords="offset points", color="gray")
        r_val, r2, Q = compute_metrics(rows, pk)
        a = np.sum(xs*ys) / np.sum(xs**2) if np.sum(xs**2) > 0 else 0
        xr = np.linspace(min(xs.min(), 0), max(xs.max(), 0), 20)
        ax.plot(xr, a * xr, "k--", lw=1.5, alpha=0.6)
        ax.set_title(f"r={r_val:+.2f}  R²={r2:.2f}  Q={Q:.2f}", fontsize=10)
        ax.grid(alpha=0.25); ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        if col == 0: ax.set_ylabel(f"{rname}\n$W_{{true}}$ density (kPa)", fontsize=10)
        if row == 2: ax.set_xlabel(f"{label} density (kPa)", fontsize=10)
        if row == 0: ax.text(0.5, 1.15, label, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="center", color=color)
fig.suptitle("Proxy vs truth — geometric septum definition\nr = directional tracking    R² = proportional fit    Q = |r| × max(R², 0)", fontsize=13, fontweight="bold")
fig.savefig(OUT / "figures" / "scatter_spectrum.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "figures" / "scatter_spectrum.pdf", bbox_inches="tight")
plt.close(fig); print("Saved scatter_spectrum.png")

# ── Print summary ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY — Q values (geometric septum)")
print("="*70)
for rname, rmasks in zip(REG_NAMES, all_region_masks):
    rows = densities_for_mask(rmasks)
    print(f"\n  {rname}:")
    for pk, _, label, _ in PROXIES:
        r_val, r2, Q = compute_metrics(rows, pk)
        print(f"    {label:<16} r={r_val:+.2f}  R²={r2:+.2f}  Q={Q:.3f}")

print(f"\nSweep stability (Q range across 80-120% septum size):")
for pk, _, label, _ in PROXIES:
    qs = [q for q in sweep_data[pk]["Q"] if not np.isnan(q)]
    if qs:
        print(f"  {label:<16} Q = [{min(qs):.3f}, {max(qs):.3f}]")

print(f"\nHandover saved to {OUT}/")
print("Done.")
