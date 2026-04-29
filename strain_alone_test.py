#!/usr/bin/env python3
"""
strain_alone_test.py — does a pressure-free strain-only metric track
septal internal work as well as the P_LV × ε proxy?

The motivation: in the narrow-SBP regime of pre-capillary PAH, P_LV is
nearly constant across cases, so P_LV × ε_ll ≈ (constant) × ε_ll. If
this is true, a proxy built from strain alone (no pressure measurement
needed) should track septal work about as well.

Test: for each of 16 EXP cases, compute several pressure-free
strain-only metrics on the septum, correlate each against ground-truth
internal work ∫ S : dE. Compare with the P_LV and transmural proxy
correlations.

Strain metrics tested:
  - peak shortening        max(|ε_ll|) across the cycle (single scalar)
  - peak-to-peak           max(ε_ll) - min(ε_ll)
  - integral of |dε|       total strain traversed (loop length on ε axis)
  - integral of (dε/dt)²   strain-rate squared — a pseudo strain energy
  - area of strain-time    ∫ ε²(t) dt over cycle
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT_A = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-23")
ROOT_B = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-24")

CASES = [
    ("sPAP22", 1047450), ("sPAP25", 1048194), ("sPAP30", 1047451), ("sPAP35", 1048195),
    ("sPAP45", 1047452), ("sPAP50", 1048196), ("sPAP55", 1047453), ("sPAP60", 1048197),
    ("sPAP65", 1047454), ("sPAP70", 1048198), ("sPAP75", 1047455), ("sPAP80", 1048199),
    ("sPAP85", 1047456), ("sPAP87", 1048200), ("sPAP92", 1048201), ("sPAP95", 1047457),
]
KPA = 1e-3

def find(jid):
    for r in [ROOT_A, ROOT_B]:
        p = r / f"UKB_6beats_run_{jid}"
        if p.exists(): return p
    raise FileNotFoundError(str(jid))

W_sept = []
peak_short = []
pp = []
int_abs_de = []
int_derate_sq = []
int_eps_sq = []
# for reference:
proxy_PLV = []
proxy_Trans = []

for label, jid in CASES:
    d = find(jid)
    pc = np.load(d / "per_cell_data.npz", allow_pickle=True)
    m = pc["is_geometric_septum"].astype(bool)
    V = pc["cell_volumes"][m].sum()
    W_sept.append(-pc["w_total"][m].sum() / V * KPA)
    proxy_PLV.append(-pc["proxy_PLV_ll"][m].sum() / V * KPA)
    proxy_Trans.append(-(pc["proxy_PLV_ll"][m].sum()
                         - pc["proxy_PRV_ll"][m].sum()) / V * KPA)

    # time-resolved septal longitudinal strain from metrics file
    metrics = np.load(d / "metrics" / "metrics_downsample_1.npy",
                       allow_pickle=True).item()
    # take the last beat
    t = np.array(metrics["time"]); n = len(t); beat = n // 6
    sl = slice(5*beat, 6*beat)
    e_ll = np.array(metrics["mean_E_ll_Septum"])[sl]
    e_ll -= e_ll[0]  # reference to ED
    tt = t[sl] - t[sl][0]

    peak_short.append(abs(e_ll.min()))
    pp.append(e_ll.max() - e_ll.min())
    de = np.diff(e_ll)
    dt = np.diff(tt)
    int_abs_de.append(np.abs(de).sum())
    int_derate_sq.append(np.sum((de/np.maximum(dt, 1e-9))**2 * dt))
    int_eps_sq.append(np.trapz(e_ll**2, tt))

W_sept = np.array(W_sept)
proxy_PLV = np.array(proxy_PLV)
proxy_Trans = np.array(proxy_Trans)

def corr(x):
    x = np.asarray(x, float)
    r = pearsonr(x, W_sept)[0]
    return r, r**2

print("n=16 EXP — septum region — proxy / metric vs. internal work")
print("=" * 70)

rows = [
    ("P_LV  × ε (proxy)",            proxy_PLV),
    ("P_LV - P_RV × ε (proxy)",      proxy_Trans),
    ("peak shortening  |min(ε_ll)|", peak_short),
    ("peak-to-peak ε_ll",            pp),
    ("∫ |dε_ll| dt  (path length)", int_abs_de),
    ("∫ (dε/dt)²  dt (strain rate)", int_derate_sq),
    ("∫ ε_ll² dt  (pseudo energy)", int_eps_sq),
]

print(f"{'metric':<32} {'r':>8} {'r²':>7} {'note':<30}")
print("-" * 80)
for label, vals in rows:
    r, r2 = corr(vals)
    note = "pressure × strain" if "(proxy)" in label else "STRAIN ONLY"
    print(f"{label:<32} {r:>+8.3f} {r2:>7.3f}  {note}")
