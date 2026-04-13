#!/usr/bin/env python3
"""
plot_old_vs_new_r2.py — compare R² (Pearson) for septum proxies between
the old presentation sims (linear EDPVR) and the new sims (kE nonlinear EDPVR).

Shows how the conclusion shifted because of the circulation library update.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

ROOT = Path('/home/dtsteene/D1/cardiac-work')


def compute_r2(metrics_file):
    if not metrics_file.exists():
        return None
    m = np.load(metrics_file, allow_pickle=True).item()
    n = len(m.get('work_true_Septum', []))
    spb = 800
    last_beat = max(0, n - spb)
    t = np.array(m['work_true_Septum'][last_beat:])
    plv = np.array(m['work_ps_ff_Septum_PLV'][last_beat:])
    prv = np.array(m['work_ps_ff_Septum_PRV'][last_beat:])
    trans = np.array(m['work_ps_ff_Septum_Trans'][last_beat:])
    if t.std() == 0:
        return None
    return {
        'R2_PLV': pearsonr(t, plv)[0] ** 2,
        'R2_PRV': pearsonr(t, prv)[0] ** 2,
        'R2_Trans': pearsonr(t, trans)[0] ** 2,
    }


def get_rv_esp(pres_file):
    if not pres_file.exists():
        return None
    p = np.load(pres_file)
    if p.ndim != 2:
        return None
    return float(p[-800:, 1].max())


# Cases that exist in BOTH old and new
old_new_pairs = [
    ('healthy', 'UKB_10beats_healthy', 1017516),
    ('mild', 'UKB_10beats_mild', 1017525),
    ('moderate', 'UKB_10beats_moderate', 1017519),
    ('moderate_severe', 'UKB_10beats_moderate_severe', 1017520),
    ('severe', 'UKB_10beats_severe', 1017521),
]

old_data = []
new_data = []
for label, old_dir, new_jid in old_new_pairs:
    old_metrics = ROOT / 'results/sims/ukb_10beats_spectrum' / old_dir / 'metrics' / 'metrics_downsample_1.npy'
    old_pres = ROOT / 'results/sims/ukb_10beats_spectrum' / old_dir / 'solver' / 'pressure_history.npy'
    new_metrics = ROOT / f'results/sims/2026-04-08/UKB_6beats_run_{new_jid}' / 'metrics' / 'metrics_downsample_1.npy'
    new_pres = ROOT / f'results/sims/2026-04-08/UKB_6beats_run_{new_jid}' / 'solver' / 'pressure_history.npy'

    o = compute_r2(old_metrics)
    n = compute_r2(new_metrics)
    o_rv = get_rv_esp(old_pres)
    n_rv = get_rv_esp(new_pres)

    if o and n:
        o['label'] = label
        o['rv_esp'] = o_rv
        n['label'] = label
        n['rv_esp'] = n_rv
        old_data.append(o)
        new_data.append(n)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

ax = axes[0]
labels = [c['label'] for c in old_data]
x = np.arange(len(labels))
width = 0.25
ax.bar(x - width, [c['R2_PLV'] for c in old_data], width, label='R²_PLV', color='C0')
ax.bar(x, [c['R2_PRV'] for c in old_data], width, label='R²_PRV', color='C3')
ax.bar(x + width, [c['R2_Trans'] for c in old_data], width, label='R²_Trans', color='C2')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('R² (Pearson, septum)')
ax.set_title('OLD sims (linear EDPVR)\n— "transmural slightly best in healthy-moderate"')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.3, axis='y')
# Annotate which is best per case
for i, c in enumerate(old_data):
    best = max(['R2_PLV', 'R2_PRV', 'R2_Trans'], key=lambda k: c[k])
    bestv = c[best]
    ax.annotate(f'{bestv:.3f}', xy=(x[i] + (-width if best == 'R2_PLV' else 0 if best == 'R2_PRV' else width), bestv),
                fontsize=7, ha='center', xytext=(0, 2), textcoords='offset points')

ax = axes[1]
ax.bar(x - width, [c['R2_PLV'] for c in new_data], width, label='R²_PLV', color='C0')
ax.bar(x, [c['R2_PRV'] for c in new_data], width, label='R²_PRV', color='C3')
ax.bar(x + width, [c['R2_Trans'] for c in new_data], width, label='R²_Trans', color='C2')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax.set_title('NEW sims (kE nonlinear EDPVR)\n— "P_LV consistently ≥ Trans, severe collapse"')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.3, axis='y')
for i, c in enumerate(new_data):
    best = max(['R2_PLV', 'R2_PRV', 'R2_Trans'], key=lambda k: c[k])
    bestv = c[best]
    ax.annotate(f'{bestv:.3f}', xy=(x[i] + (-width if best == 'R2_PLV' else 0 if best == 'R2_PRV' else width), bestv),
                fontsize=7, ha='center', xytext=(0, 2), textcoords='offset points')

fig.suptitle('OLD vs NEW: How the kE nonlinear EDPVR library update changed the proxy R² conclusion',
             fontsize=12)
plt.tight_layout()

out_dir = ROOT / 'results' / 'analysis' / 'transventricular'
fig_path = out_dir / 'old_vs_new_r2.pdf'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
fig.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"Saved {fig_path}")
plt.close(fig)

# Print numeric comparison
print("\n=== Numeric comparison ===")
print(f"{'Case':<18} {'OLD R²_PLV':>12} {'OLD R²_Tr':>12} {'NEW R²_PLV':>12} {'NEW R²_Tr':>12} {'OLD diff':>10} {'NEW diff':>10}")
print(f"{'':<18} {'':>12} {'':>12} {'':>12} {'':>12} {'(Tr-PLV)':>10} {'(Tr-PLV)':>10}")
print('-' * 100)
for o, n in zip(old_data, new_data):
    diff_old = o['R2_Trans'] - o['R2_PLV']
    diff_new = n['R2_Trans'] - n['R2_PLV']
    print(f"{o['label']:<18} {o['R2_PLV']:>12.4f} {o['R2_Trans']:>12.4f} {n['R2_PLV']:>12.4f} {n['R2_Trans']:>12.4f} {diff_old:>+10.4f} {diff_new:>+10.4f}")
