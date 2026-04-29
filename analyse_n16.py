#!/usr/bin/env python3
"""
analyse_n16.py — n=16 EXP spectrum analysis.

Loads all 16 FEM runs (original 8 + extra 8), builds the hemodynamic
summary, computes Pearson r / r^2, and runs leave-one-out for each
region × proxy pair.
"""
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

SIMS = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-23")
# Some of the extra jobs finished on 04-24, check both dates
CANDIDATE_ROOTS = [
    SIMS,
    Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-24"),
]

# All 16 cases, ordered by sPAP target
CASES_ORDERED = [
    ("sPAP22", 1047450), ("sPAP25", 1048194), ("sPAP30", 1047451), ("sPAP35", 1048195),
    ("sPAP45", 1047452), ("sPAP50", 1048196), ("sPAP55", 1047453), ("sPAP60", 1048197),
    ("sPAP65", 1047454), ("sPAP70", 1048198), ("sPAP75", 1047455), ("sPAP80", 1048199),
    ("sPAP85", 1047456), ("sPAP87", 1048200), ("sPAP92", 1048201), ("sPAP95", 1047457),
]
KPA = 1e-3

def find_rundir(jid):
    for root in CANDIDATE_ROOTS:
        d = root / f"UKB_6beats_run_{jid}"
        if d.exists():
            return d
    raise FileNotFoundError(f"run dir for job {jid} not found")

summary_rows = []
pcs = {}

for label, jid in CASES_ORDERED:
    d = find_rundir(jid)
    pc = np.load(d / "per_cell_data.npz", allow_pickle=True)
    sp = np.load(d / "solver" / "solver_cavity_pressure_mmHg.npy")
    # last beat
    beat_len = sp.shape[0] // 6
    last = sp[5 * beat_len:]
    summary_rows.append({
        "case": label,
        "jid": jid,
        "sPAP_FEM": float(last[:, 1].max()),
        "RAP_FEM":  float(last[:, 1].min()),
        "SBP_FEM":  float(last[:, 0].max()),
        "PAWP_FEM": float(last[:, 0].min()),
    })
    pcs[label] = pc

print("=" * 78)
print("v12 EXP n=16 — FEM achieved hemodynamics (last beat)")
print("=" * 78)
print(f"{'case':<8} {'sPAP':>6} {'RAP':>6} {'SBP':>6} {'PAWP':>6}")
print("-" * 40)
for r in summary_rows:
    print(f"{r['case']:<8} {r['sPAP_FEM']:>6.1f} {r['RAP_FEM']:>6.1f} "
          f"{r['SBP_FEM']:>6.1f} {r['PAWP_FEM']:>6.1f}")

# Correlations + LOO
MASKS = {
    "LV":     lambda pc: pc["region_tags"] == 1,
    "RV":     lambda pc: pc["region_tags"] == 2,
    "Septum": lambda pc: pc["is_geometric_septum"].astype(bool),
}

def densities():
    out = {reg: {"W": [], "PLV": [], "PRV": [], "Trans": []} for reg in MASKS}
    for label, _ in CASES_ORDERED:
        pc = pcs[label]; cv = pc["cell_volumes"]
        for reg, mfn in MASKS.items():
            m = mfn(pc); V = cv[m].sum()
            out[reg]["W"].append(-pc["w_total"][m].sum() / V * KPA)
            out[reg]["PLV"].append(-pc["proxy_PLV_ll"][m].sum() / V * KPA)
            out[reg]["PRV"].append(-pc["proxy_PRV_ll"][m].sum() / V * KPA)
            out[reg]["Trans"].append(
                -(pc["proxy_PLV_ll"][m].sum() - pc["proxy_PRV_ll"][m].sum()) / V * KPA
            )
    for reg in out:
        for k in out[reg]:
            out[reg][k] = np.array(out[reg][k])
    return out

def loo_r(xs, ys):
    full = pearsonr(xs, ys)[0]
    loos = np.array([pearsonr(np.delete(xs, i), np.delete(ys, i))[0]
                     for i in range(len(xs))])
    return full, loos

d = densities()

print("\n" + "=" * 78)
print("v12 EXP n=16 — Pearson r, r², and leave-one-out stability")
print("=" * 78)
print(f"{'region':<8} {'proxy':<16} {'r':>7} {'r²':>6} "
      f"{'LOO min':>9} {'LOO max':>9} {'max|Δr|':>9}")
print("-" * 78)

for reg in ["LV", "RV", "Septum"]:
    for pk, lbl in [("PLV", "P_LV"), ("PRV", "P_RV"), ("Trans", "P_LV−P_RV")]:
        xs = d[reg][pk]; ys = d[reg]["W"]
        full, loos = loo_r(xs, ys)
        r2 = full ** 2
        max_d = float(np.abs(loos - full).max())
        print(f"{reg:<8} {lbl:<16} {full:+7.3f} {r2:6.3f} "
              f"{loos.min():+9.3f} {loos.max():+9.3f} {max_d:9.3f}")

# Side-by-side summary of what changed
print("\n" + "=" * 78)
print("SEPTUM: n=8 vs n=16 for v12 EXP")
print("=" * 78)
n8 = {"PLV": (0.944, [0.909, 0.993], 0.049),
      "Trans": (0.818, [0.751, 0.915], 0.098)}
sept_full = {"PLV": pearsonr(d["Septum"]["PLV"], d["Septum"]["W"])[0],
             "Trans": pearsonr(d["Septum"]["Trans"], d["Septum"]["W"])[0]}
sept_loos = {}
for pk in ["PLV", "Trans"]:
    xs = d["Septum"][pk]; ys = d["Septum"]["W"]
    sept_loos[pk] = loo_r(xs, ys)[1]

print(f"{'proxy':<16} {'n=8 r':>7} {'n=16 r':>8} {'n=8 max|Δr|':>13} {'n=16 max|Δr|':>14}")
print("-" * 63)
for pk, lbl in [("PLV", "P_LV"), ("Trans", "P_LV−P_RV")]:
    n8r, n8rng, n8d = n8[pk]
    n16r = sept_full[pk]
    n16d = float(np.abs(sept_loos[pk] - n16r).max())
    print(f"{lbl:<16} {n8r:>+7.3f} {n16r:>+8.3f} {n8d:>13.3f} {n16d:>14.3f}")
