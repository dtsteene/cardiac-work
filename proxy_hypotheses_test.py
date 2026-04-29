#!/usr/bin/env python3
"""
proxy_hypotheses_test.py — pressure-strain hypotheses using the correct
per-cell proxy integrals and the saved directional work components.

H1. mean(P_LV, P_RV) × ε_ll should beat P_LV × ε_ll for total septal W
    if W_nn dominates the work and W_nn correlates with mean pressure.

H2. Directional tensor-work components (W_ff, W_ss, W_nn, W_cross) let us ask
    which part of the mechanical work each pressure-strain proxy is really
    tracking. If newer per_cell_data files contain geometric radial/circumferential
    proxies, those are included automatically.
"""
import os
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

ROOT_A = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-23")
ROOT_B = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-24")
CASES = [
    ("sPAP22",1047450),("sPAP25",1048194),("sPAP30",1047451),("sPAP35",1048195),
    ("sPAP45",1047452),("sPAP50",1048196),("sPAP55",1047453),("sPAP60",1048197),
    ("sPAP65",1047454),("sPAP70",1048198),("sPAP75",1047455),("sPAP80",1048199),
    ("sPAP85",1047456),("sPAP87",1048200),("sPAP92",1048201),("sPAP95",1047457),
]
KPA = 1e-3
PER_CELL_BASENAME = os.environ.get("PER_CELL_BASENAME", "per_cell_data.npz")

def find(jid):
    for r in [ROOT_A, ROOT_B]:
        p = r / f"UKB_6beats_run_{jid}"
        if p.exists(): return p

# Build per-case arrays
rows = []
for label, jid in CASES:
    d = find(jid)
    pc = np.load(d / PER_CELL_BASENAME, allow_pickle=True)
    m  = np.load(d / "metrics" / "metrics_downsample_1.npy",
                 allow_pickle=True).item()

    mask = pc["is_geometric_septum"].astype(bool)
    V = pc["cell_volumes"][mask].sum()
    # per-cell integrated proxies (sign convention: negate so positive tracks work)
    pPLV = -pc["proxy_PLV_ll"][mask].sum() / V * KPA
    pPRV = -pc["proxy_PRV_ll"][mask].sum() / V * KPA
    W_tot = -pc["w_total"][mask].sum() / V * KPA

    # directional components, integrated over the same geometric septum.
    # Do not use last-minus-first on metrics work arrays: those are per-step
    # increments, not cumulative traces.
    def dens(key):
        return -pc[key][mask].sum() / V * KPA if key in pc.files else 0.0
    W_nn   = dens("w_nn")
    W_ff   = dens("w_ff")
    W_ss   = dens("w_ss")
    W_crs  = dens("w_cross")
    W_true = dens("w_total")

    row = {
        "case": label,
        "proxy_PLV":   pPLV,
        "proxy_PRV":   pPRV,
        "proxy_Trans": pPLV - pPRV,                      # (P_LV - P_RV) × ε_ll
        "proxy_Mean":  0.5 * (pPLV + pPRV),              # mean(P_LV,P_RV) × ε_ll
        "proxy_Sum":   pPLV + pPRV,                      # (P_LV + P_RV) × ε_ll
        "W_total":     W_tot,       # from per_cell ∫ S:dE
        "W_true_ms":   W_true,      # same, from metrics (should ≈ W_tot)
        "W_nn":        W_nn,
        "W_ff":        W_ff,
        "W_ss":        W_ss,
        "W_cross":     W_crs,
    }

    for suffix in ["ff", "radial", "circ"]:
        plv_key = f"proxy_PLV_{suffix}"
        prv_key = f"proxy_PRV_{suffix}"
        trans_key = f"proxy_Trans_{suffix}"
        if plv_key in pc.files and prv_key in pc.files:
            plv = dens(plv_key)
            prv = dens(prv_key)
            row[f"{suffix}_PLV"] = plv
            row[f"{suffix}_PRV"] = prv
            row[f"{suffix}_Trans"] = dens(trans_key) if trans_key in pc.files else plv - prv
            row[f"{suffix}_Mean"] = 0.5 * (plv + prv)

    rows.append(row)

# ─── H1: which pressure weighting best tracks total septal W? ─────────────
print("=" * 72)
print("H1: which pressure weighting on ε_ll tracks total septal W best?")
print("=" * 72)
print(f"{'proxy (per_cell integrated)':<32} {'r vs W_total':>14} {'r²':>7}")
print("-" * 56)
for pk, lbl in [("proxy_PLV",   "P_LV × ε_ll"),
                ("proxy_PRV",   "P_RV × ε_ll"),
                ("proxy_Trans", "(P_LV - P_RV) × ε_ll"),
                ("proxy_Mean",  "mean(P_LV,P_RV) × ε_ll"),
                ("proxy_Sum",   "(P_LV + P_RV) × ε_ll")]:
    xs = np.array([r[pk] for r in rows])
    ys = np.array([r["W_total"] for r in rows])
    rv = pearsonr(xs, ys)[0]
    print(f"{lbl:<32} {rv:>+14.3f} {rv**2:>7.3f}")

# ─── H2: which pressure weighting best tracks each directional component?
print()
print("=" * 78)
print("H2: which pressure weighting tracks each DIRECTIONAL component?")
print("=" * 78)

for comp in ["W_total","W_nn","W_ff","W_ss","W_cross"]:
    ys = np.array([r[comp] for r in rows])
    if ys.std() < 1e-30:
        continue
    print(f"\nComponent: {comp}")
    print(f"  {'proxy':<28} {'r':>8} {'r²':>7}")
    for pk, lbl in [("proxy_PLV",   "P_LV × ε_ll"),
                    ("proxy_Trans", "(P_LV - P_RV) × ε_ll"),
                    ("proxy_Mean",  "mean(P_LV,P_RV) × ε_ll"),
                    ("proxy_PRV",   "P_RV × ε_ll")]:
        xs = np.array([r[pk] for r in rows])
        rv = pearsonr(xs, ys)[0]
        print(f"  {lbl:<28} {rv:>+8.3f} {rv**2:>7.3f}")

optional_proxy_sets = [
    ("ff", "fibre strain"),
    ("radial", "geometric radial strain"),
    ("circ", "geometric circumferential strain"),
]
available = [(suffix, label) for suffix, label in optional_proxy_sets if f"{suffix}_PLV" in rows[0]]
if available:
    print()
    print("=" * 86)
    print("H3: do alternative strain directions improve the pressure-strain proxy?")
    print("=" * 86)
    for suffix, label in available:
        print(f"\nStrain direction: {label}")
        print(f"  {'proxy':<30} {'r vs W_total':>12} {'r vs W_ff':>10} {'r vs W_nn':>10}")
        for pk, lbl in [
            (f"{suffix}_PLV", "P_LV x strain"),
            (f"{suffix}_PRV", "P_RV x strain"),
            (f"{suffix}_Trans", "(P_LV-P_RV) x strain"),
            (f"{suffix}_Mean", "mean(P_LV,P_RV) x strain"),
        ]:
            xs = np.array([r[pk] for r in rows])
            r_total = pearsonr(xs, np.array([r["W_total"] for r in rows]))[0]
            r_ff = pearsonr(xs, np.array([r["W_ff"] for r in rows]))[0]
            r_nn = pearsonr(xs, np.array([r["W_nn"] for r in rows]))[0]
            print(f"  {lbl:<30} {r_total:>+12.3f} {r_ff:>+10.3f} {r_nn:>+10.3f}")
else:
    print()
    print("H3 skipped: per_cell_data files do not yet contain ff/radial/circ proxy keys.")
