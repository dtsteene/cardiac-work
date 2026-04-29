#!/usr/bin/env python3
"""
directional_work_check.py — decompose septal work by direction (ff, ss,
nn, cross) and see which direction dominates, and which pressure each
direction correlates with.

User's hypothesis: in the septum, the sheet (ss) and sheet-normal (nn)
directions might carry more of the work than the fiber (ff) direction,
and these might correlate better with LV pressure alone than with
transmural, because the septum is mechanically wedged between two
cavities — so the stress across the wall responds to each boundary
independently, not just to their difference.
"""
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

ROOT_A = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-23")
ROOT_B = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-24")
CASES = [
    ("sPAP22", 1047450), ("sPAP25", 1048194), ("sPAP30", 1047451), ("sPAP35", 1048195),
    ("sPAP45", 1047452), ("sPAP50", 1048196), ("sPAP55", 1047453), ("sPAP60", 1048197),
    ("sPAP65", 1047454), ("sPAP70", 1048198), ("sPAP75", 1047455), ("sPAP80", 1048199),
    ("sPAP85", 1047456), ("sPAP87", 1048200), ("sPAP92", 1048201), ("sPAP95", 1047457),
]

def find(jid):
    for r in [ROOT_A, ROOT_B]:
        p = r / f"UKB_6beats_run_{jid}"
        if p.exists(): return p

# per-case beat-integrated septal work, and peak pressures.
#
# Important: metrics_downsample_1.npy stores work as per-step increments, not as
# cumulative traces. The safest source here is per_cell_data.npz, which already
# contains the last-beat integrated per-cell work. Use the same geometric septum
# mask as the proxy plots.
rows = []
for label, jid in CASES:
    d = find(jid)
    pc = np.load(d / "per_cell_data.npz", allow_pickle=True)
    mask = pc["is_geometric_septum"].astype(bool)
    vol = pc["cell_volumes"][mask].sum()

    def dens(key):
        if key not in pc.files:
            return 0.0
        return float(-pc[key][mask].sum() / vol * 1e-3)

    sp = np.load(d / "solver" / "solver_cavity_pressure_mmHg.npy")
    beat_p = sp.shape[0] // 6
    last_p = sp[5*beat_p:]

    rows.append({
        "case": label,
        "P_LV_peak": float(last_p[:, 0].max()),
        "P_RV_peak": float(last_p[:, 1].max()),
        "Trans_peak": float((last_p[:, 0] - last_p[:, 1]).max()),
        "P_mean":  0.5 * (float(last_p[:, 0].mean()) + float(last_p[:, 1].mean())),
        "W_true":    dens("w_total"),
        "W_ff":      dens("w_ff"),
        "W_ss":      dens("w_ss"),
        "W_nn":      dens("w_nn"),
        "W_cross":   dens("w_cross"),
    })

# Print per-case breakdown
print("=" * 98)
print("SEPTUM work per direction per case (last beat)")
print("=" * 98)
print(f"{'case':<8} {'W_true':>10} {'W_ff':>9} {'W_ss':>9} {'W_nn':>9} "
      f"{'W_cross':>9}")
for r in rows:
    print(f"{r['case']:<8} {r['W_true']:>10.2e} {r['W_ff']:>9.2e} "
          f"{r['W_ss']:>9.2e} {r['W_nn']:>9.2e} {r['W_cross']:>9.2e}")

# Sum of components — what fraction of W_true does each carry on average?
print()
print("Fractional contribution of each directional component to |W_true| (case-mean)")
print("-" * 60)
abs_W_true = np.array([abs(r["W_true"]) for r in rows])
for comp in ["W_ff","W_ss","W_nn","W_cross"]:
    vals = np.array([abs(r[comp]) for r in rows])
    frac = (vals / abs_W_true).mean()
    print(f"  |{comp}| / |W_true|  mean fraction = {frac:.2%}")

# Correlations of each directional work with pressures
print()
print("=" * 70)
print("Correlation of each directional septal work with case-level pressure")
print("=" * 70)
print(f"{'component':<10} {'vs P_LV':>9} {'vs P_RV':>9} {'vs Trans':>10} "
      f"{'vs mean_P':>11}")
print("-" * 60)
for comp in ["W_true", "W_ff", "W_ss", "W_nn", "W_cross"]:
    xs = np.array([r[comp] for r in rows])
    for pk, lbl in []:
        pass
    r_LV = pearsonr(xs, [r["P_LV_peak"] for r in rows])[0]
    r_RV = pearsonr(xs, [r["P_RV_peak"] for r in rows])[0]
    r_Tr = pearsonr(xs, [r["Trans_peak"] for r in rows])[0]
    r_Pm = pearsonr(xs, [r["P_mean"] for r in rows])[0]
    print(f"{comp:<10} {r_LV:>+9.3f} {r_RV:>+9.3f} {r_Tr:>+10.3f} "
          f"{r_Pm:>+11.3f}")
