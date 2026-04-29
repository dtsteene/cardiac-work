#!/usr/bin/env python3
"""
loop_convergence_check.py — beat-to-beat convergence of the FEM-coupled
PV loops for Henrik-simple (n=5) and v12 EXP (n=16).

For each beat, extract:
  EDV, ESV, SV, ESP, EDP for both LV and RV
Then report the relative change between beat 5 and beat 6 — that's the
"converged?" metric. Also flag any case whose last-beat change exceeds
a threshold.
"""
from pathlib import Path
import numpy as np

ROOTS = [Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-23"),
         Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-24"),
         Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-25")]

HENRIK = [("pul_f100", 1049979), ("pul_f075", 1049980),
          ("pul_f050", 1049981), ("pul_f025", 1049982),
          ("pul_f010", 1049983)]

V12 = [("sPAP22", 1047450), ("sPAP25", 1048194), ("sPAP30", 1047451),
       ("sPAP35", 1048195), ("sPAP45", 1047452), ("sPAP50", 1048196),
       ("sPAP55", 1047453), ("sPAP60", 1048197), ("sPAP65", 1047454),
       ("sPAP70", 1048198), ("sPAP75", 1047455), ("sPAP80", 1048199),
       ("sPAP85", 1047456), ("sPAP87", 1048200), ("sPAP92", 1048201),
       ("sPAP95", 1047457)]

def find(jid):
    for r in ROOTS:
        p = r / f"UKB_6beats_run_{jid}"
        if p.exists(): return p

def per_beat(rundir, n_beats=6):
    """Return per-beat dict of EDV/ESV/SV/ESP/EDP for LV and RV (FEM
    cavity pressure for ESP/EDP, 0D for volume). Length n_beats."""
    sp = np.load(rundir/"solver/solver_cavity_pressure_mmHg.npy")
    h  = np.load(rundir/"circulation/history.npy", allow_pickle=True).item()

    # Each side has integer * (samples per beat)
    n_p = sp.shape[0]; bp = n_p // n_beats
    n_v = len(h["V_LV"]); bv = n_v // n_beats

    out = {}
    for b in range(n_beats):
        # FEM pressures
        P_LV_b = sp[b*bp:(b+1)*bp, 0]
        P_RV_b = sp[b*bp:(b+1)*bp, 1]
        # 0D volumes
        V_LV_b = np.array(h["V_LV"])[b*bv:(b+1)*bv]
        V_RV_b = np.array(h["V_RV"])[b*bv:(b+1)*bv]
        out[b+1] = {
            "LV_EDV": float(V_LV_b.max()),
            "LV_ESV": float(V_LV_b.min()),
            "LV_SV":  float(V_LV_b.max() - V_LV_b.min()),
            "LV_ESP": float(P_LV_b.max()),
            "LV_EDP": float(P_LV_b.min()),
            "RV_EDV": float(V_RV_b.max()),
            "RV_ESV": float(V_RV_b.min()),
            "RV_SV":  float(V_RV_b.max() - V_RV_b.min()),
            "RV_ESP": float(P_RV_b.max()),
            "RV_EDP": float(P_RV_b.min()),
        }
    return out

def convergence_summary(beat_data, keys=("LV_EDV","LV_ESV","LV_ESP",
                                          "RV_EDV","RV_ESV","RV_ESP")):
    """Relative change |b6 - b5| / |b5| in percent for each key."""
    return {k: abs(beat_data[6][k] - beat_data[5][k]) /
            max(abs(beat_data[5][k]), 1e-9) * 100 for k in keys}


# ─── Pull and tabulate ──────────────────────────────────────────────────
def report(setname, cases):
    print("=" * 110)
    print(f"{setname} — beat 5 → beat 6 relative change (%) — converged when all values are small")
    print("=" * 110)
    print(f"{'case':<10} | {'LV_EDV':>7} {'LV_ESV':>7} {'LV_ESP':>7} | "
          f"{'RV_EDV':>7} {'RV_ESV':>7} {'RV_ESP':>7} | {'WORST':>5}")
    print("-" * 110)
    all_worst = []
    for label, jid in cases:
        d = find(jid)
        if d is None: continue
        bd = per_beat(d)
        cs = convergence_summary(bd)
        worst = max(cs.values())
        all_worst.append((label, worst, cs))
        flag = " ⚠" if worst > 1.0 else ""
        print(f"{label:<10} | "
              f"{cs['LV_EDV']:>7.2f} {cs['LV_ESV']:>7.2f} {cs['LV_ESP']:>7.2f} | "
              f"{cs['RV_EDV']:>7.2f} {cs['RV_ESV']:>7.2f} {cs['RV_ESP']:>7.2f} | "
              f"{worst:>5.2f}{flag}")
    worst_overall = max(w for _, w, _ in all_worst)
    mean_worst = np.mean([w for _, w, _ in all_worst])
    print(f"\n  Worst change anywhere: {worst_overall:.2f}%  (mean across cases: {mean_worst:.2f}%)")
    return all_worst

# ─── Beat-by-beat trajectory for one Henrik and one v12 case ────────────
def trajectory(setname, label, jid):
    d = find(jid)
    if d is None: return
    bd = per_beat(d)
    print(f"\n{setname}: {label}  (per-beat trajectory)")
    print(f"  beat | LV_EDV  LV_ESV  LV_ESP   RV_EDV  RV_ESV  RV_ESP   LV_SV  RV_SV  ΔSV")
    for b in range(1, 7):
        m = bd[b]
        print(f"   {b}   | {m['LV_EDV']:>6.1f}  {m['LV_ESV']:>6.1f}  "
              f"{m['LV_ESP']:>6.1f}   {m['RV_EDV']:>6.1f}  {m['RV_ESV']:>6.1f}  "
              f"{m['RV_ESP']:>6.1f}   {m['LV_SV']:>5.1f}  {m['RV_SV']:>5.1f}  "
              f"{m['LV_SV']-m['RV_SV']:+.1f}")


hen = report("HENRIK simple n=5", HENRIK)
print()
v12 = report("v12 EXP n=16", V12)

# show trajectories for the most extreme cases
print()
print("=" * 110)
print("BEAT-BY-BEAT TRAJECTORY for the two most-drifted cases of each set")
print("=" * 110)
hen.sort(key=lambda x: x[1], reverse=True)
v12.sort(key=lambda x: x[1], reverse=True)
for label, _, _ in hen[:1]:
    jid = dict(HENRIK)[label]
    trajectory("Henrik (worst)", label, jid)
for label, _, _ in v12[:2]:
    jid = dict(V12)[label]
    trajectory("v12 EXP (worst)", label, jid)
# best of each as reference
for label, _, _ in hen[-1:]:
    jid = dict(HENRIK)[label]
    trajectory("Henrik (best)", label, jid)
for label, _, _ in v12[-1:]:
    jid = dict(V12)[label]
    trajectory("v12 EXP (best)", label, jid)
