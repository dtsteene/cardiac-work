#!/usr/bin/env python3
"""
proxy_accuracy_breakdown.py

Separates three distinct questions about pressure-strain proxy accuracy:

1. INTRA-BEAT SHAPE — during one beat of one patient, does W_proxy(t) track W_true(t)?
   Metric: Pearson R² on per-timestep work increments.
   Answers: "can the proxy replace the true work curve within a cycle?"

2. INTRA-CASE TOTAL — does the beat-integrated proxy match the beat-integrated work?
   Metric: R = W_proxy_total / W_true_total (the amplitude ratio).
   Answers: "for a single patient, does the proxy give the right total magnitude?"

3. INTER-CASE TREND — across patients, does the proxy track the change in work?
   Metric: Pearson R² between beat-total W_proxy and beat-total W_true ACROSS cases.
   Also: monotonicity (Spearman rho), and linear regression slope.
   Answers: "can the proxy tell a clinician if the patient is getting better or worse?"

A proxy can be bad at (1)-(2) but good at (3), and vice versa. For clinical use,
(3) is probably the most important: what clinicians need is a reliable TREND.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/home/dtsteene/D1/cardiac-work')

SPECTRUM = [
    ('healthy', 1017516, '2026-04-08'),
    ('borderline', 1017517, '2026-04-08'),
    ('mild', 1017525, '2026-04-08'),
    ('moderate', 1017519, '2026-04-08'),
    ('moderate_severe', 1017520, '2026-04-08'),
    ('severe', 1017521, '2026-04-08'),
    ('very_severe', 1017522, '2026-04-08'),
    ('end_stage', 1017523, '2026-04-08'),
]


def load_case(label, jid, date):
    d = ROOT / 'results' / 'sims' / date / f'UKB_6beats_run_{jid}'
    pc = np.load(d / 'per_cell_data.npz')
    m = np.load(d / 'metrics' / 'metrics_downsample_1.npy', allow_pickle=True).item()
    pres = np.load(d / 'solver' / 'pressure_history.npy')

    study = pc['study_region']
    tau = pc['tau'][study]
    plv = pc['proxy_PLV_ff'][study]
    prv = pc['proxy_PRV_ff'][study]
    trans = pc['proxy_Trans_ff'][study]
    w_true = pc['w_total'][study]

    # Dominant pressure per cell
    is_lv_side = tau < 0.5
    dom = np.where(is_lv_side, plv, prv)

    # Beat totals
    totals = {
        'W_true': w_true.sum(),
        'W_PLV': plv.sum(),
        'W_PRV': prv.sum(),
        'W_Trans': trans.sum(),
        'W_mean': 0.5 * (plv.sum() + prv.sum()),
        'W_dom': dom.sum(),
    }

    # Per-timestep regional timeseries for INTRA-BEAT R² (use metrics_calculator output)
    n = len(m.get('work_true_Septum', []))
    spb = 800
    lb = max(0, n - spb)
    # Sum LV+RV+Septum for a fair "within-study-region" comparison of shape
    # (since per-cell has its own study region while metrics uses LDRB tags)
    ts_true = np.array(m['work_true_Septum'][lb:])
    ts_plv = np.array(m['work_ps_ff_Septum_PLV'][lb:])
    ts_prv = np.array(m['work_ps_ff_Septum_PRV'][lb:])
    ts_trans = np.array(m['work_ps_ff_Septum_Trans'][lb:])

    intra_r2 = {}
    if ts_true.std() > 0:
        intra_r2['R2_PLV'] = pearsonr(ts_true, ts_plv)[0] ** 2
        intra_r2['R2_PRV'] = pearsonr(ts_true, ts_prv)[0] ** 2
        intra_r2['R2_Trans'] = pearsonr(ts_true, ts_trans)[0] ** 2
        intra_r2['R2_mean'] = pearsonr(ts_true, 0.5 * (ts_plv + ts_prv))[0] ** 2

    rv_esp = float(pres[-spb:, 1].max())

    return {
        'label': label,
        'rv_esp': rv_esp,
        'totals': totals,
        'intra_r2': intra_r2,
    }


print("Loading cases...")
cases = [load_case(label, jid, date) for (label, jid, date) in SPECTRUM]
cases.sort(key=lambda c: c['rv_esp'])

# Build arrays for the three analyses
labels = [c['label'] for c in cases]
rv = np.array([c['rv_esp'] for c in cases])
W_true = np.array([c['totals']['W_true'] for c in cases])
W_PLV = np.array([c['totals']['W_PLV'] for c in cases])
W_PRV = np.array([c['totals']['W_PRV'] for c in cases])
W_Trans = np.array([c['totals']['W_Trans'] for c in cases])
W_mean = np.array([c['totals']['W_mean'] for c in cases])
W_dom = np.array([c['totals']['W_dom'] for c in cases])

# =========================================================================
# Question 1: INTRA-BEAT SHAPE (Pearson R² on per-timestep increments)
# =========================================================================
print("\n" + "=" * 75)
print("Q1: INTRA-BEAT SHAPE — per-timestep R² within each case (higher = better)")
print("=" * 75)
print(f"{'Case':<18} {'RV':>4} {'R²_PLV':>9} {'R²_PRV':>9} {'R²_Trans':>10} {'R²_mean':>9} {'winner':>10}")
for c in cases:
    ir = c['intra_r2']
    row = [ir.get('R2_PLV', 0), ir.get('R2_PRV', 0), ir.get('R2_Trans', 0), ir.get('R2_mean', 0)]
    names = ['PLV', 'PRV', 'Trans', 'mean']
    winner = names[int(np.argmax(row))]
    print(f"{c['label']:<18} {c['rv_esp']:>4.0f} "
          f"{ir.get('R2_PLV', 0):>9.4f} {ir.get('R2_PRV', 0):>9.4f} "
          f"{ir.get('R2_Trans', 0):>10.4f} {ir.get('R2_mean', 0):>9.4f} "
          f"{winner:>10}")

# =========================================================================
# Question 2: INTRA-CASE TOTAL (amplitude ratio per case)
# =========================================================================
print("\n" + "=" * 75)
print("Q2: INTRA-CASE TOTAL — beat-integrated ratio W_proxy / W_true")
print("=" * 75)
print("    A proxy is 'stable' if R doesn't change across cases.")
print(f"{'Case':<18} {'RV':>4} {'R_PLV':>7} {'R_PRV':>7} {'R_Trans':>8} {'R_mean':>8} {'R_dom':>7}")
for c in cases:
    t = c['totals']
    wt = t['W_true']
    print(f"{c['label']:<18} {c['rv_esp']:>4.0f} "
          f"{t['W_PLV']/wt:>7.3f} {t['W_PRV']/wt:>7.3f} "
          f"{t['W_Trans']/wt:>8.3f} {t['W_mean']/wt:>8.3f} {t['W_dom']/wt:>7.3f}")

# Standard deviation of ratios = stability measure
print("\nStability (std of R across cases, lower = more stable):")
for name, arr in [('R_PLV', W_PLV/W_true), ('R_PRV', W_PRV/W_true),
                  ('R_Trans', W_Trans/W_true), ('R_mean', W_mean/W_true),
                  ('R_dom', W_dom/W_true)]:
    print(f"  {name:<10}: mean={arr.mean():+.3f}, std={arr.std():.3f}, "
          f"range=[{arr.min():+.3f}, {arr.max():+.3f}]")

# =========================================================================
# Question 3: INTER-CASE TREND (proxy tracks work as disease changes)
# =========================================================================
print("\n" + "=" * 75)
print("Q3: INTER-CASE TREND — does beat-total W_proxy track beat-total W_true")
print("    across the disease spectrum?")
print("=" * 75)
print("    Pearson R² = linearity, Spearman ρ = monotonicity, slope = scaling.")
print(f"{'Proxy':<12} {'Pearson R²':>12} {'Spearman ρ':>13} {'slope':>10} {'intercept':>12}")
for name, arr in [('W_PLV', W_PLV), ('W_PRV', W_PRV), ('W_Trans', W_Trans),
                  ('W_mean', W_mean), ('W_dom', W_dom)]:
    r_pearson, _ = pearsonr(W_true, arr)
    rho_spearman, _ = spearmanr(W_true, arr)
    slope, intercept = np.polyfit(W_true, arr, 1)
    print(f"{name:<12} {r_pearson**2:>12.4f} {rho_spearman:>13.4f} "
          f"{slope:>10.4f} {intercept:>+12.4e}")

print("\nNote: for a perfect trend proxy, we want:")
print("  - Pearson R² close to 1.0 (linear scaling)")
print("  - Spearman ρ close to 1.0 (monotonic)")
print("  - Reasonable slope (doesn't need to be 1.0, just consistent)")

# =========================================================================
# Save the aggregated data
# =========================================================================
out_dir = ROOT / 'results' / 'analysis' / 'transventricular'
out_dir.mkdir(parents=True, exist_ok=True)

# =========================================================================
# Figure: all three questions in one
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: Intra-beat R² per case (Q1)
ax = axes[0]
proxies = ['R2_PLV', 'R2_PRV', 'R2_Trans', 'R2_mean']
markers = ['o', 's', 'D', '^']
colors = ['C0', 'C3', 'C2', 'C4']
for name, mark, col in zip(proxies, markers, colors):
    vals = [c['intra_r2'].get(name, 0) for c in cases]
    ax.plot(rv, vals, '-' + mark, color=col, label=name, lw=1.5, ms=7)
ax.set_xlabel('RV peak pressure (mmHg)')
ax.set_ylabel('Pearson R² (within beat)')
ax.set_title('Q1: Intra-beat shape matching\n(can proxy track work during cycle?)')
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)

# Panel 2: Intra-case ratio (Q2)
ax = axes[1]
ratios = {
    'R_PLV': W_PLV / W_true,
    'R_PRV': W_PRV / W_true,
    'R_Trans': W_Trans / W_true,
    'R_mean': W_mean / W_true,
    'R_dom': W_dom / W_true,
}
markers = {'R_PLV': 'o', 'R_PRV': 's', 'R_Trans': 'D', 'R_mean': '^', 'R_dom': 'v'}
colors = {'R_PLV': 'C0', 'R_PRV': 'C3', 'R_Trans': 'C2', 'R_mean': 'm', 'R_dom': 'k'}
for name, vals in ratios.items():
    ax.plot(rv, vals, '-' + markers[name], color=colors[name], label=name, lw=1.5, ms=7)
ax.set_xlabel('RV peak pressure (mmHg)')
ax.set_ylabel('R = W_proxy / W_true (per case)')
ax.set_title('Q2: Intra-case total matching\n(does proxy give right magnitude per patient?)')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_ylim(-0.02, 0.30)

# Panel 3: Inter-case trend (Q3) - scatter plot W_true vs W_proxy
ax = axes[2]
for name, arr, col, mk in [('W_PLV', W_PLV, 'C0', 'o'),
                             ('W_Trans', W_Trans, 'C2', 'D'),
                             ('W_mean', W_mean, 'm', '^'),
                             ('W_dom', W_dom, 'k', 'v')]:
    # Scatter + best-fit line
    r2 = pearsonr(W_true, arr)[0] ** 2
    ax.scatter(W_true, arr, color=col, marker=mk, s=70, label=f"{name} (R²={r2:.3f})")
    slope, intercept = np.polyfit(W_true, arr, 1)
    x_fit = np.linspace(W_true.min(), W_true.max(), 20)
    ax.plot(x_fit, slope * x_fit + intercept, '--', color=col, lw=1, alpha=0.5)
ax.set_xlabel('W_true (beat total, J)')
ax.set_ylabel('W_proxy (beat total, J)')
ax.set_title('Q3: Inter-case trend\n(does proxy track work across patients?)')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
# Annotate cases
for i, c in enumerate(cases):
    ax.annotate(c['label'][:4], (W_true[i], W_PLV[i]),
                fontsize=7, xytext=(4, -2), textcoords='offset points', color='C0')

fig.suptitle('Three Distinct Questions about Proxy Accuracy', fontsize=13)
plt.tight_layout()
fig_path = out_dir / 'proxy_accuracy_breakdown.pdf'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
fig.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved {fig_path}")
plt.close(fig)
