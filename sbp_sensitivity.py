#!/usr/bin/env python3
"""
sbp_sensitivity.py — synthetic SBP-sweep sensitivity test for the septum
proxy question.

Question: the n=16 EXP result says P_LV beats P_LV-P_RV for septum work
(r=0.93 vs 0.82). We hypothesise this is because SBP is ~constant
(std≈2.5 mmHg) across our cases, so P_LV × ε_ll is essentially
rescaled strain and tracks work through strain. If SBP varied more,
the two proxies should re-separate.

Test: take the 16 per_cell datasets, synthetically scale per-case P_LV
by factor k_i chosen to span a target SBP range, recompute proxies,
look at correlations vs ground truth. Repeat with increasing SBP stdev
and plot the P_LV r and transmural r as functions of SBP stdev.

Key property we use: proxy_PLV[case] = ∫ P_LV dε_ll × vol is already
saved per cell. Scaling P_LV(t) by a constant k scales this integral
by k. Scaling only P_LV (not P_RV) and not re-solving strain is a
synthetic "what if SBP had been different while everything else the
same" test. Not a real physics simulation but a clean linear-algebra
test of whether the correlation ordering is driven by SBP variance.
"""
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-23")
ROOT2 = Path("/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-04-24")

# v12 EXP n=16 job IDs
CASES = [
    ("sPAP22", 1047450), ("sPAP25", 1048194), ("sPAP30", 1047451), ("sPAP35", 1048195),
    ("sPAP45", 1047452), ("sPAP50", 1048196), ("sPAP55", 1047453), ("sPAP60", 1048197),
    ("sPAP65", 1047454), ("sPAP70", 1048198), ("sPAP75", 1047455), ("sPAP80", 1048199),
    ("sPAP85", 1047456), ("sPAP87", 1048200), ("sPAP92", 1048201), ("sPAP95", 1047457),
]
KPA = 1e-3

def find(jid):
    for r in [ROOT, ROOT2]:
        p = r / f"UKB_6beats_run_{jid}"
        if p.exists(): return p
    raise FileNotFoundError(str(jid))

# Load per-case: septum proxy totals and the achieved mean P_LV (= peak/cycle avg, rough SBP)
pc_data = []
achieved_sbp = []
for label, jid in CASES:
    pc = np.load(find(jid) / "per_cell_data.npz", allow_pickle=True)
    sp = np.load(find(jid) / "solver" / "solver_cavity_pressure_mmHg.npy")
    beat_len = sp.shape[0] // 6
    last = sp[5*beat_len:]
    achieved_sbp.append(float(last[:, 0].max()))
    m = pc["is_geometric_septum"].astype(bool)
    V = pc["cell_volumes"][m].sum()
    pc_data.append({
        "label": label,
        "sbp": float(last[:, 0].max()),
        "int_P_LV":  -pc["proxy_PLV_ll"][m].sum() / V * KPA,
        "int_P_RV":  -pc["proxy_PRV_ll"][m].sum() / V * KPA,
        "W_truth":   -pc["w_total"][m].sum() / V * KPA,
    })
achieved_sbp = np.array(achieved_sbp)

int_PLV = np.array([d["int_P_LV"]  for d in pc_data])
int_PRV = np.array([d["int_P_RV"]  for d in pc_data])
W       = np.array([d["W_truth"]   for d in pc_data])

base_sbp_mean = achieved_sbp.mean()
base_sbp_std  = achieved_sbp.std()

print(f"Observed SBP across 16 cases: {base_sbp_mean:.1f} ± {base_sbp_std:.1f} mmHg")

# Baseline proxies, no scaling (reproduces n=16 EXP result)
proxy_PLV_base  = int_PLV
proxy_Trans_base = int_PLV - int_PRV
r_PLV_base   = pearsonr(proxy_PLV_base, W)[0]
r_Trans_base = pearsonr(proxy_Trans_base, W)[0]
print(f"Baseline (SBP std≈{base_sbp_std:.1f}): "
      f"r(P_LV) = {r_PLV_base:+.3f}, r(Trans) = {r_Trans_base:+.3f}")

# Sweep: simulate wider SBP variance by scaling each case's P_LV by k_i,
# where k_i = SBP_synthetic_i / SBP_base_i. Do this many times with random
# assignments of SBP_synthetic drawn from Normal(mean=base_sbp_mean,
# std=sweep_std), for a range of sweep_std values. Collect the distribution
# of r(P_LV) and r(Trans) at each sweep_std.

sweep_stds = np.arange(0, 45, 2.5)  # 0 to 42 mmHg SBP std
n_mc = 400
rng = np.random.default_rng(42)

median_r_PLV = []
median_r_Trans = []
p10_r_PLV = []; p90_r_PLV = []
p10_r_Trans = []; p90_r_Trans = []

for s in sweep_stds:
    r_PLV_samples = []; r_Trans_samples = []
    for _ in range(n_mc):
        if s == 0:
            sbp_synth = np.full(16, base_sbp_mean)
        else:
            sbp_synth = rng.normal(base_sbp_mean, s, 16)
        # don't allow unphysical SBP < 60 or > 200
        sbp_synth = np.clip(sbp_synth, 60, 200)
        k = sbp_synth / achieved_sbp
        proxy_PLV  = k * int_PLV
        proxy_Trans = k * int_PLV - int_PRV
        r_PLV_samples.append(pearsonr(proxy_PLV, W)[0])
        r_Trans_samples.append(pearsonr(proxy_Trans, W)[0])
    rP = np.array(r_PLV_samples); rT = np.array(r_Trans_samples)
    median_r_PLV.append(np.median(rP)); p10_r_PLV.append(np.percentile(rP, 10)); p90_r_PLV.append(np.percentile(rP, 90))
    median_r_Trans.append(np.median(rT)); p10_r_Trans.append(np.percentile(rT, 10)); p90_r_Trans.append(np.percentile(rT, 90))

# Plot
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax.fill_between(sweep_stds, p10_r_PLV, p90_r_PLV, color="#1f77b4", alpha=0.18,
                label="P_LV r, 10–90 %")
ax.plot(sweep_stds, median_r_PLV, color="#1f77b4", lw=2.2, label="P_LV r (median)")
ax.fill_between(sweep_stds, p10_r_Trans, p90_r_Trans, color="#2ca02c", alpha=0.18,
                label=r"$P_{LV}-P_{RV}$ r, 10–90 %")
ax.plot(sweep_stds, median_r_Trans, color="#2ca02c", lw=2.2, label=r"$P_{LV}-P_{RV}$ r (median)")
ax.axvline(base_sbp_std, color="black", ls="--", lw=1, alpha=0.6)
ax.text(base_sbp_std + 0.5, 0.4,
        f"observed\nSBP std\n= {base_sbp_std:.1f}",
        fontsize=9, va="center", color="black")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("synthetic SBP standard deviation across cases (mmHg)", fontsize=11)
ax.set_ylabel("Pearson r (septum proxy vs ground truth)", fontsize=11)
ax.set_title("Septum proxy correlation vs synthetic SBP variance (v12 EXP, n=16)",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.25); ax.legend(fontsize=10, loc="lower left")
ax.set_ylim(-0.2, 1.02); ax.set_xlim(sweep_stds.min(), sweep_stds.max())
fig.savefig("sbp_sensitivity_septum.png", dpi=160, bbox_inches="tight")
fig.savefig("sbp_sensitivity_septum.pdf", bbox_inches="tight")
print("Saved sbp_sensitivity_septum.{png,pdf}")

# Also print a summary table at some sweep_stds of interest
print()
print(f"{'SBP std':>8} | {'r(P_LV)':>11} | {'r(Trans)':>11} | winner")
print("-" * 50)
for s, rP, rT in zip(sweep_stds, median_r_PLV, median_r_Trans):
    winner = "P_LV" if abs(rP) > abs(rT) else "Trans"
    print(f"{s:>6.1f}   | {rP:>+8.3f}    | {rT:>+8.3f}    | {winner}")
