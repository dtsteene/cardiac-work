#!/usr/bin/env python3
"""
build_spectrum_handover.py — Package the 7 spectrum simulations into a
clean handover for collaborators.

Outputs (all in results/handover_spectrum/):
  hemodynamic_summary.csv    — one row per case, achieved hemodynamics
  circulation_timeseries/    — per-case CSV with all 0D time-series (last beat)
  solver_pressures/          — per-case CSV with FEM cavity pressures (last beat)
  circulation_params/        — the JSON parameter files used
  per_cell_data/             — symlinks to per_cell_data.npz for work analysis
"""
import numpy as np
import json
import csv
import shutil
from pathlib import Path

ROOT = Path("results/sims/2026-04-12")
OUT = Path("results/handover_spectrum")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "circulation_timeseries").mkdir(exist_ok=True)
(OUT / "solver_pressures").mkdir(exist_ok=True)
(OUT / "circulation_params").mkdir(exist_ok=True)
(OUT / "per_cell_data").mkdir(exist_ok=True)

# The 7 cases, ordered by achieved RV ESP
SIMS = [
    ("healthy",         "1020849", "data/ukb_circ_v2/optimized_regazzoni_ukb_healthy.json"),
    ("mild",            "1020851", "data/ukb_circ_v2/optimized_regazzoni_ukb_mild.json"),
    ("moderate",        "1020852", "data/ukb_circ_v2/optimized_regazzoni_ukb_moderate.json"),
    ("severe",          "1020854", "data/ukb_circ_v2/optimized_regazzoni_ukb_severe.json"),
    ("moderate_severe", "1020853", "data/ukb_circ_v2/optimized_regazzoni_ukb_moderate_severe.json"),
    ("very_severe",     "1020855", "data/ukb_circ_v2/optimized_regazzoni_ukb_very_severe.json"),
    ("end_stage",       "1020856", "data/ukb_circ_v2/optimized_regazzoni_ukb_end_stage.json"),
]

summary_rows = []

for archival_key, rid, circ_json_path in SIMS:
    d = ROOT / f"UKB_6beats_run_{rid}"

    # Load data
    h = np.load(d / "circulation" / "history.npy", allow_pickle=True).item()
    sp = np.load(d / "solver" / "solver_cavity_pressure_mmHg.npy")
    params = json.load(open(d / "simulation_params.json"))

    n_0d = len(h["V_LV"])
    beat_0d = n_0d // 6
    sl_0d = slice(5 * beat_0d, 6 * beat_0d)

    n_sp = sp.shape[0]
    beat_sp = n_sp // 6
    sl_sp = slice(5 * beat_sp, 6 * beat_sp)

    # Hemodynamic characterization (last beat)
    V_LV = np.array(h["V_LV"])[sl_0d]
    V_RV = np.array(h["V_RV"])[sl_0d]
    V_LA = np.array(h["V_LA"])[sl_0d]
    V_RA = np.array(h["V_RA"])[sl_0d]
    p_LV_0d = np.array(h["p_LV"])[sl_0d]
    p_RV_0d = np.array(h["p_RV"])[sl_0d]
    p_AR_PUL = np.array(h["p_AR_PUL"])[sl_0d]
    p_AR_SYS = np.array(h["p_AR_SYS"])[sl_0d]
    p_VEN_PUL = np.array(h["p_VEN_PUL"])[sl_0d]

    sp_last = sp[sl_sp]

    rv_edv = float(V_RV.max()); rv_esv = float(V_RV.min())
    rv_sv = rv_edv - rv_esv; rv_ef = rv_sv / rv_edv * 100
    lv_edv = float(V_LV.max()); lv_esv = float(V_LV.min())
    lv_sv = lv_edv - lv_esv; lv_ef = lv_sv / lv_edv * 100

    rv_esp = float(sp_last[:, 1].max())
    rv_edp = float(sp_last[:, 1].min())
    lv_esp = float(sp_last[:, 0].max())
    lv_edp = float(sp_last[:, 0].min())

    mpap = float(p_AR_PUL.mean())
    ao_dbp = float(p_AR_SYS.min())
    ao_sbp = float(p_AR_SYS.max())
    la_mean = float(np.array(h["p_LA"])[sl_0d].mean()) if "p_LA" in h else float("nan")
    co = rv_sv * 75 / 1000  # L/min at 75 bpm

    # Neutral case ID ordered by RV ESP
    case_id = f"C{len(summary_rows)+1}"

    summary_rows.append({
        "case_id": case_id,
        "archival_key": archival_key,
        "run_id": rid,
        "RV_ESP_mmHg": round(rv_esp, 1),
        "RV_EDP_mmHg": round(rv_edp, 1),
        "RV_EDV_mL": round(rv_edv, 1),
        "RV_ESV_mL": round(rv_esv, 1),
        "RV_SV_mL": round(rv_sv, 1),
        "RV_EF_pct": round(rv_ef, 1),
        "LV_ESP_mmHg": round(lv_esp, 1),
        "LV_EDP_mmHg": round(lv_edp, 1),
        "LV_EDV_mL": round(lv_edv, 1),
        "LV_ESV_mL": round(lv_esv, 1),
        "LV_SV_mL": round(lv_sv, 1),
        "LV_EF_pct": round(lv_ef, 1),
        "mPAP_mmHg": round(mpap, 1),
        "Ao_SBP_mmHg": round(ao_sbp, 1),
        "Ao_DBP_mmHg": round(ao_dbp, 1),
        "LA_mean_mmHg": round(la_mean, 1),
        "CO_Lmin": round(co, 2),
        "HR_bpm": 75,
    })

    # Export 0D circulation time-series (last beat, time in seconds)
    dt_0d = 0.001  # 1 ms
    t_0d = np.arange(beat_0d) * dt_0d
    circ_keys = ["V_LA", "V_LV", "V_RA", "V_RV",
                 "p_LA", "p_LV", "p_RA", "p_RV",
                 "p_AR_SYS", "p_VEN_SYS", "p_AR_PUL", "p_VEN_PUL",
                 "Q_MV", "Q_AV", "Q_TV", "Q_PV",
                 "Q_AR_SYS", "Q_VEN_SYS", "Q_AR_PUL", "Q_VEN_PUL"]

    with open(OUT / "circulation_timeseries" / f"{case_id}_circulation.csv", "w", newline="") as f:
        available = [k for k in circ_keys if k in h]
        writer = csv.writer(f)
        writer.writerow(["time_s"] + available)
        for i in range(beat_0d):
            row = [f"{t_0d[i]:.4f}"]
            for k in available:
                row.append(f"{np.array(h[k])[5*beat_0d + i]:.4f}")
            writer.writerow(row)

    # Export solver pressures (last beat)
    dt_sp = 0.001
    t_sp = np.arange(sp_last.shape[0]) * dt_sp
    with open(OUT / "solver_pressures" / f"{case_id}_solver_pressure_mmHg.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "P_LV_mmHg", "P_RV_mmHg"])
        for i in range(sp_last.shape[0]):
            writer.writerow([f"{t_sp[i]:.4f}", f"{sp_last[i,0]:.2f}", f"{sp_last[i,1]:.2f}"])

    # Copy circulation parameter JSON
    src = Path(circ_json_path)
    if src.exists():
        shutil.copy2(src, OUT / "circulation_params" / f"{case_id}_{src.name}")

    # Symlink per_cell_data
    pc_src = d / "per_cell_data.npz"
    pc_dst = OUT / "per_cell_data" / f"{case_id}_per_cell_data.npz"
    if pc_src.exists() and not pc_dst.exists():
        pc_dst.symlink_to(pc_src.resolve())

    print(f"{case_id} ({archival_key}, RV_ESP={rv_esp:.0f}): exported")

# Sort by RV ESP and reassign case IDs
summary_rows.sort(key=lambda r: r["RV_ESP_mmHg"])
for i, row in enumerate(summary_rows):
    row["case_id"] = f"C{i+1}"

# Write summary CSV
with open(OUT / "hemodynamic_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

# Also write a nice readable version
print("\n" + "="*120)
print("HEMODYNAMIC SUMMARY — ordered by RV ESP")
print("="*120)
print(f"{'ID':<4} {'key':<18} {'RV_ESP':>7} {'RV_EDP':>7} {'RV_EDV':>7} {'RV_SV':>6} {'RV_EF':>6} | "
      f"{'LV_ESP':>7} {'LV_EDP':>7} {'LV_EDV':>7} {'LV_SV':>6} {'LV_EF':>6} | "
      f"{'mPAP':>6} {'CO':>5}")
print("-"*120)
for r in summary_rows:
    print(f"{r['case_id']:<4} {r['archival_key']:<18} {r['RV_ESP_mmHg']:>7.0f} {r['RV_EDP_mmHg']:>7.0f} "
          f"{r['RV_EDV_mL']:>7.0f} {r['RV_SV_mL']:>6.0f} {r['RV_EF_pct']:>5.0f}% | "
          f"{r['LV_ESP_mmHg']:>7.0f} {r['LV_EDP_mmHg']:>7.0f} {r['LV_EDV_mL']:>7.0f} "
          f"{r['LV_SV_mL']:>6.0f} {r['LV_EF_pct']:>5.0f}% | "
          f"{r['mPAP_mmHg']:>6.0f} {r['CO_Lmin']:>5.1f}")

# Print proposed x-axis labels
print("\nProposed x-axis labels (RV ESP in mmHg):")
for r in summary_rows:
    print(f"  {r['case_id']}: RV$_{{ESP}}$ = {r['RV_ESP_mmHg']:.0f} mmHg")

print(f"\nHandover package saved to {OUT}/")
print("Done.")
