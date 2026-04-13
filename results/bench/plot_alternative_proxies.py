#!/usr/bin/env python3
"""
plot_alternative_proxies.py — Compare R_PLV, R_PRV, R_Trans, R_mean, R_dom

Shows that the conventional transmural proxy collapses with PAH severity
while alternative proxies (mean and dominant-per-cell) stay stable.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/home/dtsteene/D1/cardiac-work')

SPECTRUM_CASES = [
    ('healthy', 1017516, '2026-04-08'),
    ('borderline', 1017517, '2026-04-08'),
    ('mild', 1017525, '2026-04-08'),
    ('moderate', 1017519, '2026-04-08'),
    ('moderate_severe', 1017520, '2026-04-08'),
    ('severe', 1017521, '2026-04-08'),
    ('very_severe', 1017522, '2026-04-08'),
    ('end_stage', 1017523, '2026-04-08'),
]

THICKNESS_CASES = [
    ('global_1mm', 1017843),
    ('global_2mm', 1017844),
    ('rvfw_2mm', 1017845),
    ('rvfw_3mm', 1017846),
    ('rvfw_5mm', 1017847),
    ('rvfw_7mm', 1017848),
]


def load_proxy_ratios(jid, date):
    d = ROOT / 'results' / 'sims' / date / f'UKB_6beats_run_{jid}'
    pc = np.load(d / 'per_cell_data.npz')
    pres = np.load(d / 'solver' / 'pressure_history.npy')
    rv_esp = float(pres[-800:, 1].max())

    study = pc['study_region']
    tau = pc['tau'][study]
    plv = pc['proxy_PLV_ff'][study]
    prv = pc['proxy_PRV_ff'][study]
    w_true = pc['w_total'][study].sum()

    is_lv_side = tau < 0.5
    w_dom = np.where(is_lv_side, plv, prv).sum()

    return {
        'rv_esp': rv_esp,
        'R_PLV': plv.sum() / w_true,
        'R_PRV': prv.sum() / w_true,
        'R_Trans': pc['proxy_Trans_ff'][study].sum() / w_true,
        'R_mean': 0.5 * (plv.sum() + prv.sum()) / w_true,
        'R_dom': w_dom / w_true,
    }


# ── Spectrum data ────────────────────────────────────────────────────────────
spectrum = []
for label, jid, date in SPECTRUM_CASES:
    d = load_proxy_ratios(jid, date)
    d['label'] = label
    spectrum.append(d)
spectrum.sort(key=lambda x: x['rv_esp'])

# ── Thickness data ───────────────────────────────────────────────────────────
thickness = []
for label, jid in THICKNESS_CASES:
    d = load_proxy_ratios(jid, '2026-04-09')
    d['label'] = label
    thickness.append(d)

# Order thickness sensibly
order = {'global_1mm': 0, 'global_2mm': 1, 'rvfw_2mm': 2, 'rvfw_3mm': 3, 'rvfw_5mm': 4, 'rvfw_7mm': 5}
thickness.sort(key=lambda x: order[x['label']])

# ── Figure: Alternative proxies vs conventional ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: spectrum
ax = axes[0]
rv = [c['rv_esp'] for c in spectrum]
ax.plot(rv, [c['R_PLV'] for c in spectrum], 'b-o', label='R_PLV', lw=1.5, ms=6, alpha=0.7)
ax.plot(rv, [c['R_PRV'] for c in spectrum], 'r-o', label='R_PRV', lw=1.5, ms=6, alpha=0.7)
ax.plot(rv, [c['R_Trans'] for c in spectrum], 'g-s', label='R_Trans (conventional)', lw=2.5, ms=8)
ax.plot(rv, [c['R_mean'] for c in spectrum], 'm-D', label='R_mean (alternative)', lw=2.5, ms=8)
ax.plot(rv, [c['R_dom'] for c in spectrum], 'k-^', label='R_dom (alternative)', lw=2.5, ms=8)
ax.axhline(0, color='k', lw=0.5, alpha=0.5)
ax.set_xlabel('RV peak pressure (mmHg)')
ax.set_ylabel('Proxy ratio R')
ax.set_title('Disease Spectrum\n(R_Trans collapses, R_mean and R_dom stay stable)')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_ylim(-0.02, 0.30)

# Right: thickness
ax = axes[1]
labels = [c['label'] for c in thickness]
x = np.arange(len(labels))
ax.plot(x, [c['R_PLV'] for c in thickness], 'b-o', label='R_PLV', lw=1.5, ms=6, alpha=0.7)
ax.plot(x, [c['R_PRV'] for c in thickness], 'r-o', label='R_PRV', lw=1.5, ms=6, alpha=0.7)
ax.plot(x, [c['R_Trans'] for c in thickness], 'g-s', label='R_Trans (conventional)', lw=2.5, ms=8)
ax.plot(x, [c['R_mean'] for c in thickness], 'm-D', label='R_mean (alternative)', lw=2.5, ms=8)
ax.plot(x, [c['R_dom'] for c in thickness], 'k-^', label='R_dom (alternative)', lw=2.5, ms=8)
ax.axhline(0, color='k', lw=0.5, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Proxy ratio R')
ax.set_title('Thickness Variants (severe PAH)\n(rvfw_5mm: R_Trans ≈ 0 but R_mean, R_dom ≈ 0.25)')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_ylim(-0.02, 0.30)

fig.suptitle('Alternative Proxies for Septal Work\n'
             'R_mean = (W_PLV + W_PRV)/2 / W_true     '
             'R_dom = use P_LV for tau<0.5, P_RV for tau≥0.5',
             fontsize=12)
plt.tight_layout()

out_dir = ROOT / 'results' / 'analysis' / 'transventricular'
fig_path = out_dir / 'alternative_proxies.pdf'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
fig.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"Saved {fig_path}")

# Also save a copy in thickness dir
out_dir2 = ROOT / 'results' / 'analysis' / 'transventricular_thickness'
fig.savefig(out_dir2 / 'alternative_proxies.pdf', dpi=150, bbox_inches='tight')
fig.savefig(out_dir2 / 'alternative_proxies.png', dpi=150, bbox_inches='tight')
print(f"Saved {out_dir2 / 'alternative_proxies.pdf'}")
plt.close(fig)
