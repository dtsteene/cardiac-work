#!/usr/bin/env python3
"""
pressure_stress_check.py — peak pressures vs peak septal stress, old vs new.

Skip strain entirely. Ask: does peak septal fiber stress track peak P_LV,
peak P_RV, or peak transmural?

S_ff reported in metrics is the 2nd Piola-Kirchhoff fiber stress (reference
config). That's fine for case-to-case comparison — it differs from Cauchy
by roughly det(F)*F^-1 factors that don't swing wildly across cases.
"""
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

def peak_P_and_S(rundir):
    sp = np.load(rundir / "solver" / "solver_cavity_pressure_mmHg.npy")
    beat = sp.shape[0] // 6
    last = sp[5*beat:]
    P_LV_peak = float(last[:, 0].max())
    P_RV_peak = float(last[:, 1].max())
    Trans_peak = float((last[:, 0] - last[:, 1]).max())

    metrics = np.load(rundir / "metrics" / "metrics_downsample_1.npy",
                       allow_pickle=True).item()
    t = np.array(metrics["time"]); n = len(t); bm = n // 6
    S_ff = np.array(metrics.get("mean_S_ff_Septum", np.zeros(n)))[5*bm:6*bm]
    S_ff_peak = float(np.abs(S_ff).max()) * 1e-3  # Pa → kPa
    return P_LV_peak, P_RV_peak, Trans_peak, S_ff_peak

# OLD: 7 cases from handover_old (2026-04-12)
OLD_ROOT = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-12")
OLD_JIDS = ["1020849","1020851","1020852","1020854","1020853","1020855","1020856"]

# NEW v12 EXP: 16 cases
NEW_A = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-23")
NEW_B = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-24")
NEW_JIDS = [1047450, 1048194, 1047451, 1048195, 1047452, 1048196, 1047453, 1048197,
            1047454, 1048198, 1047455, 1048199, 1047456, 1048200, 1048201, 1047457]

def find_new(jid):
    for r in [NEW_A, NEW_B]:
        p = r / f"UKB_6beats_run_{jid}"
        if p.exists(): return p

for name, rundirs in [("OLD 7-case", [OLD_ROOT / f"UKB_6beats_run_{j}" for j in OLD_JIDS]),
                      ("NEW 16-case v12 EXP", [find_new(j) for j in NEW_JIDS])]:
    data = [peak_P_and_S(d) for d in rundirs]
    P_LV, P_RV, Trans, S_ff = zip(*data)
    P_LV = np.array(P_LV); P_RV = np.array(P_RV); Trans = np.array(Trans)
    S_ff = np.array(S_ff)
    print("=" * 70)
    print(f"{name}")
    print("=" * 70)
    print(f"  P_LV peak: mean {P_LV.mean():.1f}, std {P_LV.std():.2f}, range [{P_LV.min():.1f}, {P_LV.max():.1f}]")
    print(f"  P_RV peak: mean {P_RV.mean():.1f}, std {P_RV.std():.2f}, range [{P_RV.min():.1f}, {P_RV.max():.1f}]")
    print(f"  Trans peak: mean {Trans.mean():.1f}, std {Trans.std():.2f}, range [{Trans.min():.1f}, {Trans.max():.1f}]")
    print(f"  S_ff sept peak: mean {S_ff.mean():.2f} kPa, std {S_ff.std():.2f}, range [{S_ff.min():.2f}, {S_ff.max():.2f}]")
    print()
    for label, x in [("P_LV peak", P_LV), ("P_RV peak", P_RV), ("Trans peak", Trans)]:
        r = pearsonr(x, S_ff)[0]
        print(f"  corr(peak {label:<12}, peak S_ff septum) = r = {r:+.3f}   r² = {r**2:.3f}")
    print()
