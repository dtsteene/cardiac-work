#!/usr/bin/env python3
"""
loo_v12.py — leave-one-out r-stability for the v12 LIN and EXP handovers.

For each region × proxy pair, recomputes Pearson r with each of the 8 cases
removed in turn and reports:
  - full-sample r
  - LOO mean, min, max, stdev
  - which dropped case moves r the most (max |Δr|)

Tells us whether the correlation is driven by one outlier case or is stable.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

ROOT = Path("/global/D1/homes/dtsteene/cardiac-work/results/handover")
CASES = [f"C{i+1}" for i in range(8)]
CASE_LABEL = dict(zip(CASES, ["sPAP22","sPAP30","sPAP45","sPAP55",
                              "sPAP65","sPAP75","sPAP85","sPAP95"]))
KPA = 1e-3

MASKS = {
    "LV":     lambda pc: pc["region_tags"] == 1,
    "RV":     lambda pc: pc["region_tags"] == 2,
    "Septum": lambda pc: pc["is_geometric_septum"].astype(bool),
}


def densities(variant):
    base = ROOT / f"handover_{variant}" / "data" / "per_cell_data"
    out = {reg: {"W": [], "PLV": [], "PRV": [], "Trans": []} for reg in MASKS}
    for cid in CASES:
        pc = np.load(base / f"{cid}_per_cell_data.npz", allow_pickle=True)
        cv = pc["cell_volumes"]
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
    """Return (full_r, loo_rs, dropped_indices)."""
    full_r = pearsonr(xs, ys)[0]
    loos = []
    for i in range(len(xs)):
        idx = np.array([j for j in range(len(xs)) if j != i])
        loos.append(pearsonr(xs[idx], ys[idx])[0])
    return full_r, np.array(loos)


for variant in ["lin", "exp"]:
    print("=" * 78)
    print(f"v12 {variant.upper()} — leave-one-out Pearson r stability (n = 8 → 7)")
    print("=" * 78)
    d = densities(variant)
    header = (f"{'region':<8} {'proxy':<16} {'full r':>7} "
              f"{'LOO mean':>9} {'LOO min':>9} {'LOO max':>9} "
              f"{'max|Δr|':>9} {'worst drop':>12}")
    print(header); print("-" * len(header))
    for reg in ["LV", "RV", "Septum"]:
        for pk, label in [("PLV","P_LV"), ("PRV","P_RV"), ("Trans","P_LV−P_RV")]:
            xs = d[reg][pk]; ys = d[reg]["W"]
            full, loos = loo_r(xs, ys)
            deltas = np.abs(loos - full)
            max_d = float(deltas.max())
            worst = CASE_LABEL[CASES[int(np.argmax(deltas))]]
            print(f"{reg:<8} {label:<16} "
                  f"{full:+7.3f} "
                  f"{loos.mean():+9.3f} {loos.min():+9.3f} {loos.max():+9.3f} "
                  f"{max_d:9.3f} {worst:>12}")
    print()
