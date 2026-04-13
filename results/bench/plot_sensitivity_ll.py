#!/usr/bin/env python3
"""
plot_sensitivity_ll.py — Sensitivity curve for longitudinal-strain proxies.

Following the specification in results/docs/transmural_work_profiles.md
(Phase 4, "Sensitivity Analysis + Reference Points", lines 358-379):

  For each tau_cutoff in linspace(tau_range_geo[0], tau_range_ldrb[0], N):
      sept_mask(case) = study_region(case) & (tau(case) >= tc) & (tau(case) <= 1 - tc)
      W_true[case]  = sum(w_total[sept_mask])
      W_proxy[case] = sum(proxy_*_ll[sept_mask])
  For each proxy, compute Pearson R² between W_true[case] and W_proxy[case]
  across the 8 spectrum cases.

Outcome metric: inter-case Pearson R² (tracks "does proxy follow disease progression").
Proxy variants: P_LV, P_RV, P_LV-P_RV, (P_LV+P_RV)/2, per-cell dominant pressure.
Strain component: dε_ll (longitudinal, clinical GLS analogue).

No conclusions drawn — the script produces one figure and a numeric table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

ROOT = Path('/home/dtsteene/D1/cardiac-work')

# 8 spectrum cases, ordered by intended severity
SPECTRUM = [
    ('healthy', 1017516),
    ('borderline', 1017517),
    ('mild', 1017525),
    ('moderate', 1017519),
    ('moderate_severe', 1017520),
    ('severe', 1017521),
    ('very_severe', 1017522),
    ('end_stage', 1017523),
]

N_CUTOFFS = 30  # sweep resolution


# ── Load all cases ──────────────────────────────────────────────────────────

def load_case(label, jid):
    d = ROOT / 'results' / 'sims' / '2026-04-08' / f'UKB_6beats_run_{jid}'
    pc = np.load(d / 'per_cell_data.npz')
    pres = np.load(d / 'solver' / 'pressure_history.npy')
    return {
        'label': label,
        'rv_esp': float(pres[-800:, 1].max()),
        'tau': pc['tau'],
        'study_region': pc['study_region'],
        'is_geometric_septum': pc['is_geometric_septum'],
        'is_ldrb_septum': pc['is_ldrb_septum'],
        'w_total': pc['w_total'],
        'proxy_PLV_ll': pc['proxy_PLV_ll'],
        'proxy_PRV_ll': pc['proxy_PRV_ll'],
        'proxy_Trans_ll': pc['proxy_Trans_ll'],
    }


print("Loading 8 spectrum cases...")
cases = [load_case(label, jid) for label, jid in SPECTRUM]

# ── Determine the sweep range ───────────────────────────────────────────────
# The mesh is nominally the same for all 8 cases but sim-to-sim mesh generation
# produces small tau differences across cases. To define a sweep that is
# non-degenerate for ALL cases, use:
#   tc_narrow = MAX across cases of tau[is_geometric_septum].min()
#               (tightest symmetric window that still encloses the LV-side edge
#                of every case's geometric septum)
#   tc_wide   = MIN across cases of tau[is_ldrb_septum].min()
#               (widest symmetric window that doesn't exceed any case's LDRB
#                LV-side edge)

geo_mins = [c['tau'][c['is_geometric_septum']].min() for c in cases]
ldrb_mins = [c['tau'][c['is_ldrb_septum']].min() for c in cases]
tc_narrow = max(geo_mins)
tc_wide = min(ldrb_mins)

print(f"\nTau range consistency check (per case):")
print(f"  {'Case':<18} {'tau_geo_min':>12} {'tau_ldrb_min':>13}")
for c, gm, lm in zip(cases, geo_mins, ldrb_mins):
    print(f"  {c['label']:<18} {gm:>12.4f} {lm:>13.4f}")

print(f"\nSweep bounds (common across all cases):")
print(f"  tc_narrow (geometric end) = max(tau_geo_min) = {tc_narrow:.4f}")
print(f"  tc_wide   (LDRB end)      = min(tau_ldrb_min) = {tc_wide:.4f}")
print(f"  Sweep: {N_CUTOFFS} values from {tc_narrow:.4f} (narrow) to {tc_wide:.4f} (wide)")

# Per the doc Phase 4, sweep from tight to wide (tc decreases)
cutoffs = np.linspace(tc_narrow, tc_wide, N_CUTOFFS)

# ── Compute per-case sums at each cutoff ────────────────────────────────────

n_cases = len(cases)
W_true = np.zeros((N_CUTOFFS, n_cases))
W_PLV = np.zeros((N_CUTOFFS, n_cases))
W_PRV = np.zeros((N_CUTOFFS, n_cases))
W_Trans = np.zeros((N_CUTOFFS, n_cases))
W_mean = np.zeros((N_CUTOFFS, n_cases))
W_dom = np.zeros((N_CUTOFFS, n_cases))
n_cells_in_mask = np.zeros((N_CUTOFFS, n_cases), dtype=int)

for i, tc in enumerate(cutoffs):
    upper = 1.0 - tc
    for j, c in enumerate(cases):
        tau = c['tau']
        mask = c['study_region'] & (tau >= tc) & (tau <= upper)
        n_cells_in_mask[i, j] = int(mask.sum())
        if mask.sum() == 0:
            continue
        plv = c['proxy_PLV_ll'][mask]
        prv = c['proxy_PRV_ll'][mask]
        trans = c['proxy_Trans_ll'][mask]
        tau_m = tau[mask]

        W_true[i, j] = c['w_total'][mask].sum()
        W_PLV[i, j] = plv.sum()
        W_PRV[i, j] = prv.sum()
        W_Trans[i, j] = trans.sum()
        W_mean[i, j] = 0.5 * (plv.sum() + prv.sum())
        # Per-cell dominant: use P_LV for tau<0.5, P_RV for tau>=0.5
        is_lv = tau_m < 0.5
        W_dom[i, j] = np.where(is_lv, plv, prv).sum()

# ── Compute Pearson R² across cases at each cutoff ──────────────────────────

def inter_case_r2(truth_row, proxy_row):
    """Pearson R² between per-case totals, at a single cutoff."""
    # Need at least 2 distinct truth values
    if np.std(truth_row) == 0 or np.std(proxy_row) == 0:
        return np.nan, np.nan
    r, _ = pearsonr(truth_row, proxy_row)
    return r, r ** 2


r_PLV = np.zeros(N_CUTOFFS)
r_PRV = np.zeros(N_CUTOFFS)
r_Trans = np.zeros(N_CUTOFFS)
r_mean = np.zeros(N_CUTOFFS)
r_dom = np.zeros(N_CUTOFFS)
r2_PLV = np.zeros(N_CUTOFFS)
r2_PRV = np.zeros(N_CUTOFFS)
r2_Trans = np.zeros(N_CUTOFFS)
r2_mean = np.zeros(N_CUTOFFS)
r2_dom = np.zeros(N_CUTOFFS)

for i in range(N_CUTOFFS):
    r_PLV[i], r2_PLV[i] = inter_case_r2(W_true[i], W_PLV[i])
    r_PRV[i], r2_PRV[i] = inter_case_r2(W_true[i], W_PRV[i])
    r_Trans[i], r2_Trans[i] = inter_case_r2(W_true[i], W_Trans[i])
    r_mean[i], r2_mean[i] = inter_case_r2(W_true[i], W_mean[i])
    r_dom[i], r2_dom[i] = inter_case_r2(W_true[i], W_dom[i])

# ── Print numeric table (first, middle, last cutoffs) ──────────────────────

print(f"\n{'='*90}")
print("Per-case totals at sweep endpoints and midpoint")
print('='*90)
for i_label, i in [('narrow (geometric end)', 0),
                    ('middle', N_CUTOFFS // 2),
                    ('wide (LDRB end)', N_CUTOFFS - 1)]:
    tc = cutoffs[i]
    print(f"\n[tau_cutoff = {tc:.4f}]  septum = [{tc:.4f}, {1-tc:.4f}]   ({i_label})")
    print(f"  {'Case':<18} {'n_cells':>8} {'W_true':>13} {'W_PLV_ll':>13} "
          f"{'W_PRV_ll':>13} {'W_Trans_ll':>13} {'W_mean_ll':>13} {'W_dom_ll':>13}")
    for j, c in enumerate(cases):
        print(f"  {c['label']:<18} {n_cells_in_mask[i, j]:>8} "
              f"{W_true[i, j]:>+13.4e} {W_PLV[i, j]:>+13.4e} "
              f"{W_PRV[i, j]:>+13.4e} {W_Trans[i, j]:>+13.4e} "
              f"{W_mean[i, j]:>+13.4e} {W_dom[i, j]:>+13.4e}")
    print(f"  {'INTER-CASE R (signed)':<32} "
          f"r_PLV={r_PLV[i]:+.4f}  r_PRV={r_PRV[i]:+.4f}  "
          f"r_Trans={r_Trans[i]:+.4f}  r_mean={r_mean[i]:+.4f}  r_dom={r_dom[i]:+.4f}")
    print(f"  {'INTER-CASE R² (unsigned)':<32} "
          f"R²_PLV={r2_PLV[i]:.4f}  R²_PRV={r2_PRV[i]:.4f}  "
          f"R²_Trans={r2_Trans[i]:.4f}  R²_mean={r2_mean[i]:.4f}  R²_dom={r2_dom[i]:.4f}")

# ── Plot ─────────────────────────────────────────────────────────────────────

# Pre-compute reference point positions (per-case tau_geo_min, tau_ldrb_min)
geo_ref = np.mean(geo_mins)
ldrb_ref = np.mean(ldrb_mins)
geo_range = (min(geo_mins), max(geo_mins))
ldrb_range = (min(ldrb_mins), max(ldrb_mins))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)

# Left panel: signed Pearson r (shows sign of correlation)
ax = axes[0]
ax.plot(cutoffs, r_PLV, '-o', color='C0', label='P_LV × dε_ll', lw=1.8, ms=5)
ax.plot(cutoffs, r_PRV, '-o', color='C3', label='P_RV × dε_ll', lw=1.8, ms=5)
ax.plot(cutoffs, r_Trans, '-o', color='C2', label='(P_LV - P_RV) × dε_ll', lw=2.2, ms=6)
ax.plot(cutoffs, r_mean, '-o', color='m', label='(P_LV + P_RV)/2 × dε_ll', lw=1.8, ms=5)
ax.plot(cutoffs, r_dom, '-o', color='k', label='P_dom × dε_ll  (per cell)', lw=1.8, ms=5)

# Reference point bands
ax.axvspan(geo_range[0], geo_range[1], alpha=0.12, color='blue')
ax.axvspan(ldrb_range[0], ldrb_range[1], alpha=0.12, color='red')
ax.axvline(geo_ref, color='blue', ls='--', lw=1, alpha=0.6)
ax.axvline(ldrb_ref, color='red', ls='--', lw=1, alpha=0.6)

ax.axhline(0, color='k', lw=0.5, alpha=0.5)
ax.axhline(1, color='k', lw=0.3, ls=':', alpha=0.5)
ax.axhline(-1, color='k', lw=0.3, ls=':', alpha=0.5)
ax.set_xlabel(r'tau cutoff  (septum = [$t_c$, $1 - t_c$])', fontsize=10)
ax.set_ylabel('Pearson r (across 8 spectrum cases)', fontsize=10)
ax.set_title('Signed correlation of per-case W_proxy with per-case W_true', fontsize=11)
ax.set_ylim(-1.1, 1.1)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc='lower left')
# Annotate the reference point bands
ylim = ax.get_ylim()
ax.text(geo_ref, ylim[0] + 0.08, ' geometric', fontsize=9, color='blue', ha='left', va='bottom')
ax.text(ldrb_ref, ylim[0] + 0.08, ' LDRB', fontsize=9, color='red', ha='left', va='bottom')
# Annotate tight/wide
ax.annotate('tight\n(narrow septum)', xy=(tc_narrow, ylim[1] - 0.1),
            xytext=(-5, 0), textcoords='offset points', fontsize=8,
            ha='right', va='top', color='gray')
ax.annotate('wide\n(broad septum)', xy=(tc_wide, ylim[1] - 0.1),
            xytext=(5, 0), textcoords='offset points', fontsize=8,
            ha='left', va='top', color='gray')

# Right panel: R² (unsigned — how much variance in W_true is explained)
ax = axes[1]
ax.plot(cutoffs, r2_PLV, '-o', color='C0', label='P_LV × dε_ll', lw=1.8, ms=5)
ax.plot(cutoffs, r2_PRV, '-o', color='C3', label='P_RV × dε_ll', lw=1.8, ms=5)
ax.plot(cutoffs, r2_Trans, '-o', color='C2', label='(P_LV - P_RV) × dε_ll', lw=2.2, ms=6)
ax.plot(cutoffs, r2_mean, '-o', color='m', label='(P_LV + P_RV)/2 × dε_ll', lw=1.8, ms=5)
ax.plot(cutoffs, r2_dom, '-o', color='k', label='P_dom × dε_ll  (per cell)', lw=1.8, ms=5)

ax.axvspan(geo_range[0], geo_range[1], alpha=0.12, color='blue')
ax.axvspan(ldrb_range[0], ldrb_range[1], alpha=0.12, color='red')
ax.axvline(geo_ref, color='blue', ls='--', lw=1, alpha=0.6)
ax.axvline(ldrb_ref, color='red', ls='--', lw=1, alpha=0.6)

ax.set_xlabel(r'tau cutoff  (septum = [$t_c$, $1 - t_c$])', fontsize=10)
ax.set_ylabel('Pearson R² (across 8 spectrum cases)', fontsize=10)
ax.set_title('Variance of W_true explained by each proxy (per case totals)', fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc='lower left')
ylim = ax.get_ylim()
ax.text(geo_ref, ylim[0] + 0.04, ' geometric', fontsize=9, color='blue', ha='left', va='bottom')
ax.text(ldrb_ref, ylim[0] + 0.04, ' LDRB', fontsize=9, color='red', ha='left', va='bottom')

fig.suptitle('Inter-case sensitivity curve: '
             'does each proxy track W_true across the 8-case disease spectrum?\n'
             'Longitudinal strain (ε_ll, clinical GLS analogue). '
             'Sweep width = how narrow/wide you draw the septum (symmetric τ window).',
             fontsize=11)
plt.tight_layout()

out_dir = ROOT / 'results' / 'analysis' / 'transventricular'
out_dir.mkdir(parents=True, exist_ok=True)
fig_path = out_dir / 'sensitivity_ll_inter_case.pdf'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
fig.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved {fig_path}")
plt.close(fig)

# ── Also save raw data as npz so you can replot without rerunning ───────────
np.savez(out_dir / 'sensitivity_ll_inter_case_raw.npz',
         cutoffs=cutoffs,
         W_true=W_true, W_PLV=W_PLV, W_PRV=W_PRV,
         W_Trans=W_Trans, W_mean=W_mean, W_dom=W_dom,
         n_cells_in_mask=n_cells_in_mask,
         r_PLV=r_PLV, r_PRV=r_PRV, r_Trans=r_Trans, r_mean=r_mean, r_dom=r_dom,
         r2_PLV=r2_PLV, r2_PRV=r2_PRV, r2_Trans=r2_Trans, r2_mean=r2_mean, r2_dom=r2_dom,
         case_labels=np.array([c['label'] for c in cases]),
         case_rv_esp=np.array([c['rv_esp'] for c in cases]),
         geo_mins=np.array(geo_mins), ldrb_mins=np.array(ldrb_mins),
         geo_ref=geo_ref, ldrb_ref=ldrb_ref)
print(f"Saved raw data to {out_dir / 'sensitivity_ll_inter_case_raw.npz'}")
