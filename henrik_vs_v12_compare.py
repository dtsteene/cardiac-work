#!/usr/bin/env python3
"""
henrik_vs_v12_compare.py — head-to-head between the Henrik-simple PUL
sweep and the v12 EXP n=16 spectrum, focusing on:

  1. 0D ↔ FEM coupling consistency
       - 0D vs FEM cavity pressure drift, last beat
       - LV/RV SV balance (mass conservation)
       - 0D target sPAP vs FEM-achieved sPAP
  2. Comparison to clinical targets
       - Kovacs 2009: sPAP 20.8±4.4, PAWP 8±3, CI 4.1±1.3 (healthy)
       - Tello 2019:  sPAP 75±24,  PAWP 9±3, CI 2.8±0.7, RVEF 37±13
       - Humbert T16: RAP <8/8-14/>14, CI ≥2.5/2.0-2.4/<2.0, RVEF >54/37-54/<37
  3. Spectrum behaviour
       - sPAP, SBP, CI, RVEF as a function of severity
"""
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

ROOT_25 = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-25")
ROOT_23 = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-23")
ROOT_24 = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-24")

HENRIK = [("pul_f100", 1049979, 1.00),
          ("pul_f075", 1049980, 0.75),
          ("pul_f050", 1049981, 0.50),
          ("pul_f025", 1049982, 0.25),
          ("pul_f010", 1049983, 0.10)]

V12 = [("sPAP22", 1047450), ("sPAP25", 1048194), ("sPAP30", 1047451), ("sPAP35", 1048195),
       ("sPAP45", 1047452), ("sPAP50", 1048196), ("sPAP55", 1047453), ("sPAP60", 1048197),
       ("sPAP65", 1047454), ("sPAP70", 1048198), ("sPAP75", 1047455), ("sPAP80", 1048199),
       ("sPAP85", 1047456), ("sPAP87", 1048200), ("sPAP92", 1048201), ("sPAP95", 1047457)]

def find(jid, candidates):
    for r in candidates:
        p = r / f"UKB_6beats_run_{jid}"
        if p.exists(): return p

def per_case_metrics(rundir):
    """Pull last-beat hemodynamics + a coupling-quality measure."""
    sp = np.load(rundir/"solver/solver_cavity_pressure_mmHg.npy")
    h  = np.load(rundir/"circulation/history.npy", allow_pickle=True).item()

    # last beat of solver
    beat_p = sp.shape[0] // 6
    last_p = sp[5*beat_p:]
    P_LV_FEM = last_p[:, 0]
    P_RV_FEM = last_p[:, 1]

    # last beat of 0D
    n_0d = len(h["V_LV"]); beat_0d = n_0d // 6
    sl = slice(5*beat_0d, 6*beat_0d)
    V_LV = np.array(h["V_LV"])[sl]
    V_RV = np.array(h["V_RV"])[sl]
    P_LV_0D = np.array(h["p_LV"])[sl]
    P_RV_0D = np.array(h["p_RV"])[sl]
    p_AR_PUL = np.array(h["p_AR_PUL"])[sl]
    p_AR_SYS = np.array(h["p_AR_SYS"])[sl]

    # Resample FEM to 0D time grid
    nt = len(V_LV)
    if len(P_LV_FEM) != nt:
        t_fem = np.linspace(0, 1, len(P_LV_FEM))
        t_0d  = np.linspace(0, 1, nt)
        P_LV_FEM = np.interp(t_0d, t_fem, P_LV_FEM)
        P_RV_FEM = np.interp(t_0d, t_fem, P_RV_FEM)

    # --- Hemodynamics
    SBP = float(P_LV_FEM.max())
    DBP = float(P_LV_FEM.min())
    sPAP = float(P_RV_FEM.max())
    RAP  = float(P_RV_FEM.min())
    PAWP = float(np.array(h["p_LA"])[sl].mean())
    mPAP = float(p_AR_PUL.mean())
    LV_EDV = float(V_LV.max()); LV_ESV = float(V_LV.min())
    RV_EDV = float(V_RV.max()); RV_ESV = float(V_RV.min())
    LV_SV = LV_EDV - LV_ESV
    RV_SV = RV_EDV - RV_ESV
    LVEF = LV_SV / LV_EDV * 100
    RVEF = RV_SV / RV_EDV * 100
    CO = LV_SV * 75 / 1000.0
    CI = CO / 1.75

    # --- Coupling consistency
    # 0D vs FEM cavity-pressure drift over the last beat
    drift_LV_rms = float(np.sqrt(np.mean((P_LV_FEM - P_LV_0D)**2)))
    drift_RV_rms = float(np.sqrt(np.mean((P_RV_FEM - P_RV_0D)**2)))
    drift_LV_max = float(np.max(np.abs(P_LV_FEM - P_LV_0D)))
    drift_RV_max = float(np.max(np.abs(P_RV_FEM - P_RV_0D)))
    sv_imbalance = abs(LV_SV - RV_SV) / max(LV_SV, 1.0) * 100  # in %

    return dict(SBP=SBP, DBP=DBP, sPAP=sPAP, RAP=RAP, PAWP=PAWP, mPAP=mPAP,
                LV_EDV=LV_EDV, RV_EDV=RV_EDV, LVEF=LVEF, RVEF=RVEF, CI=CI,
                drift_LV_rms=drift_LV_rms, drift_RV_rms=drift_RV_rms,
                drift_LV_max=drift_LV_max, drift_RV_max=drift_RV_max,
                sv_imbalance_pct=sv_imbalance,
                LV_SV=LV_SV, RV_SV=RV_SV)


# ─── Pull data
print("=" * 100)
print("HENRIK simple PUL sweep — coupled FEM achieved hemodynamics + coupling drift (n=5)")
print("=" * 100)
print(f"{'case':<10} {'f':>5} {'sPAP':>6} {'SBP':>6} {'mPAP':>5} {'PAWP':>5} {'RAP':>5} "
      f"{'CI':>5} {'LVEF':>5} {'RVEF':>5} | {'driftLV':>7} {'driftRV':>7} {'SVΔ%':>5}")
print("-"*100)
hen_data = []
for label, jid, f in HENRIK:
    d = find(jid, [ROOT_25])
    if d is None: continue
    m = per_case_metrics(d); m["label"] = label; m["factor"] = f
    hen_data.append(m)
    print(f"{label:<10} {f:>5.2f} "
          f"{m['sPAP']:>6.1f} {m['SBP']:>6.1f} {m['mPAP']:>5.1f} "
          f"{m['PAWP']:>5.1f} {m['RAP']:>5.1f} {m['CI']:>5.2f} "
          f"{m['LVEF']:>5.1f} {m['RVEF']:>5.1f} | "
          f"{m['drift_LV_rms']:>7.1f} {m['drift_RV_rms']:>7.1f} "
          f"{m['sv_imbalance_pct']:>5.1f}")

print()
print("=" * 100)
print("v12 EXP n=16 — coupled FEM achieved hemodynamics + coupling drift")
print("=" * 100)
print(f"{'case':<10} {'sPAP':>6} {'SBP':>6} {'mPAP':>5} {'PAWP':>5} {'RAP':>5} "
      f"{'CI':>5} {'LVEF':>5} {'RVEF':>5} | {'driftLV':>7} {'driftRV':>7} {'SVΔ%':>5}")
print("-"*100)
v12_data = []
for label, jid in V12:
    d = find(jid, [ROOT_23, ROOT_24])
    if d is None: continue
    m = per_case_metrics(d); m["label"] = label
    v12_data.append(m)
    print(f"{label:<10} "
          f"{m['sPAP']:>6.1f} {m['SBP']:>6.1f} {m['mPAP']:>5.1f} "
          f"{m['PAWP']:>5.1f} {m['RAP']:>5.1f} {m['CI']:>5.2f} "
          f"{m['LVEF']:>5.1f} {m['RVEF']:>5.1f} | "
          f"{m['drift_LV_rms']:>7.1f} {m['drift_RV_rms']:>7.1f} "
          f"{m['sv_imbalance_pct']:>5.1f}")

# ─── Compare coupling quality
def stats(rows, k):
    arr = np.array([r[k] for r in rows])
    return arr.mean(), arr.max(), arr.std()

print()
print("=" * 80)
print("COUPLING-DRIFT SUMMARY (RMS over last beat, mmHg). Lower = tighter coupling.")
print("=" * 80)
for setname, rows in [("Henrik-simple n=5", hen_data), ("v12 EXP n=16", v12_data)]:
    m_lv, mx_lv, std_lv = stats(rows, "drift_LV_rms")
    m_rv, mx_rv, std_rv = stats(rows, "drift_RV_rms")
    print(f"  {setname}:")
    print(f"    LV drift  mean = {m_lv:5.2f}  max = {mx_lv:5.2f}  std = {std_lv:.2f}")
    print(f"    RV drift  mean = {m_rv:5.2f}  max = {mx_rv:5.2f}  std = {std_rv:.2f}")
    sv_m = np.mean([r["sv_imbalance_pct"] for r in rows])
    sv_x = np.max([r["sv_imbalance_pct"] for r in rows])
    print(f"    SV imbalance mean = {sv_m:5.2f}%  max = {sv_x:5.2f}%")

# ─── Clinical targets summary
print()
print("=" * 80)
print("VS CLINICAL TARGETS")
print("=" * 80)
print(f"  Kovacs 2009 healthy:  sPAP 20.8±4.4, PAWP 8.0±2.9, CI 4.1±1.3")
print(f"  Tello 2019 severe:    sPAP 75±24,    PAWP 9±3,     CI 2.8±0.7,  RVEF 37±13")
print(f"  Humbert T16 high-risk: RAP >14, CI <2.0, RVEF <37%")
print()

print("Henrik healthy end (pul_f100):")
m = hen_data[0]
print(f"  sPAP={m['sPAP']:.1f}  PAWP={m['PAWP']:.1f}  CI={m['CI']:.2f}  RVEF={m['RVEF']:.1f}%")
print(f"  matches Kovacs sPAP within ±SD? {abs(m['sPAP']-20.8) < 4.4}")
print()
print("Henrik severe end (pul_f010):")
m = hen_data[-1]
print(f"  sPAP={m['sPAP']:.1f}  PAWP={m['PAWP']:.1f}  CI={m['CI']:.2f}  RVEF={m['RVEF']:.1f}%")
print(f"  matches Tello sPAP within ±SD? {abs(m['sPAP']-75) < 24}")
print(f"  in Humbert high-risk? CI<2.0: {m['CI']<2.0}, RVEF<37: {m['RVEF']<37}")
print()
print("v12 EXP healthy (sPAP22):")
m = v12_data[0]
print(f"  sPAP={m['sPAP']:.1f}  PAWP={m['PAWP']:.1f}  CI={m['CI']:.2f}  RVEF={m['RVEF']:.1f}%")
print()
print("v12 EXP severe (sPAP95):")
m = v12_data[-1]
print(f"  sPAP={m['sPAP']:.1f}  PAWP={m['PAWP']:.1f}  CI={m['CI']:.2f}  RVEF={m['RVEF']:.1f}%")

# ─── Ranges
print()
print("=" * 80)
print("SPECTRUM RANGES (FEM achieved)")
print("=" * 80)
def rng(rows, k): return f"{min(r[k] for r in rows):>6.1f} → {max(r[k] for r in rows):>6.1f}"
print(f"{'metric':<10} {'Henrik (5)':<22} {'v12 EXP (16)':<22}")
for k in ["sPAP","SBP","mPAP","PAWP","RAP","CI","LVEF","RVEF","RV_EDV","LV_EDV"]:
    print(f"{k:<10} {rng(hen_data,k):<22} {rng(v12_data,k):<22}")
