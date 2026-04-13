#!/usr/bin/env python3
"""
plot_spectrum_summary.py — Summary figures for the disease spectrum

Key finding: as PAH worsens, the transmural (P_LV - P_RV) proxy collapses
because W_PLV and W_PRV become similar in magnitude. This contradicts the
initial hypothesis that transmural pressure is the preferred septal proxy.

Generates:
1. Absolute work magnitudes vs RV_ESP (W_true, W_PLV, W_PRV, W_Trans)
2. Proxy ratio R(severity) curves for all three proxies
3. Side-by-side spectrum + thickness comparison
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Hard-coded case lists since this is a one-shot summary script
SPECTRUM_CASES = [
    ('healthy', 1017516),
    ('borderline', 1017517),
    ('mild', 1017525),
    ('moderate', 1017519),
    ('moderate_severe', 1017520),
    ('severe', 1017521),
    ('very_severe', 1017522),
    ('end_stage', 1017523),
]

THICKNESS_CASES = [
    ('global_1mm', 1017843),
    ('global_2mm', 1017844),
    ('rvfw_2mm', 1017845),
    ('rvfw_3mm', 1017846),
    ('rvfw_5mm', 1017847),
    ('rvfw_7mm', 1017848),
]

ROOT = Path('/home/dtsteene/D1/cardiac-work')


def load_case(jid, date):
    d = ROOT / 'results' / 'sims' / date / f'UKB_6beats_run_{jid}'
    pc = np.load(d / 'per_cell_data.npz')
    pres = np.load(d / 'solver' / 'pressure_history.npy')
    rv_esp = float(pres[-800:, 1].max())
    lv_esp = float(pres[-800:, 0].max())
    study = pc['study_region']
    return {
        'rv_esp': rv_esp,
        'lv_esp': lv_esp,
        'w_true': pc['w_total'][study].sum(),
        'w_ff': pc['w_ff'][study].sum(),
        'w_PLV': pc['proxy_PLV_ff'][study].sum(),
        'w_PRV': pc['proxy_PRV_ff'][study].sum(),
        'w_Trans': pc['proxy_Trans_ff'][study].sum(),
        'n_study': int(study.sum()),
    }


# ── Load spectrum ────────────────────────────────────────────────────────────
print("Loading spectrum cases...")
spectrum = []
for label, jid in SPECTRUM_CASES:
    data = load_case(jid, '2026-04-08')
    data['label'] = label
    spectrum.append(data)
spectrum.sort(key=lambda x: x['rv_esp'])

# ── Load thickness ───────────────────────────────────────────────────────────
print("Loading thickness cases...")
thickness = []
for label, jid in THICKNESS_CASES:
    data = load_case(jid, '2026-04-09')
    data['label'] = label
    thickness.append(data)

# ── Figure 1: Absolute work vs disease severity ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
rv = [c['rv_esp'] for c in spectrum]
ax.plot(rv, [c['w_true'] for c in spectrum], 'k-o', label='W_true (S:dE)', lw=2.5, ms=7)
ax.plot(rv, [c['w_ff'] for c in spectrum], 'k--s', label='W_ff (fiber only)', lw=1.5, ms=6, alpha=0.6)
ax.plot(rv, [c['w_PLV'] for c in spectrum], 'b-o', label='W_PLV proxy', lw=1.8, ms=6)
ax.plot(rv, [c['w_PRV'] for c in spectrum], 'r-o', label='W_PRV proxy', lw=1.8, ms=6)
ax.plot(rv, [c['w_Trans'] for c in spectrum], 'g-o', label='W_Trans = W_PLV - W_PRV', lw=1.8, ms=6)
ax.axhline(0, color='k', lw=0.5, alpha=0.5)
ax.set_xlabel('RV peak pressure (mmHg)')
ax.set_ylabel('Total septum work (J, study region)')
ax.set_title('Absolute work magnitudes vs disease severity')
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3)
# Annotate cases on top
for c in spectrum:
    ax.annotate(c['label'][:4], (c['rv_esp'], c['w_PLV']),
                fontsize=7, xytext=(0, -12), textcoords='offset points',
                ha='center', color='blue', alpha=0.7)

ax = axes[1]
# Proxy ratios W_proxy / W_true
ratios_PLV = [c['w_PLV'] / c['w_true'] for c in spectrum]
ratios_PRV = [c['w_PRV'] / c['w_true'] for c in spectrum]
ratios_Trans = [c['w_Trans'] / c['w_true'] for c in spectrum]
ax.plot(rv, ratios_PLV, 'b-o', label='R_PLV = W_PLV / W_true', lw=2, ms=7)
ax.plot(rv, ratios_PRV, 'r-o', label='R_PRV = W_PRV / W_true', lw=2, ms=7)
ax.plot(rv, ratios_Trans, 'g-o', label='R_Trans = W_Trans / W_true', lw=2, ms=7)
ax.axhline(0, color='k', lw=0.5, alpha=0.5)
ax.set_xlabel('RV peak pressure (mmHg)')
ax.set_ylabel('Proxy ratio R')
ax.set_title('Fraction of true work captured by each proxy')
ax.legend(fontsize=9, loc='center right')
ax.grid(alpha=0.3)
ax.set_ylim(0, 0.30)

fig.suptitle('Disease Spectrum: Pressure-strain proxies for septal work\n'
             'Key finding: as PAH worsens, the transmural proxy collapses '
             'because W_PLV and W_PRV cancel each other',
             fontsize=12)
plt.tight_layout()
out_dir = ROOT / 'results' / 'analysis' / 'transventricular'
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / 'spectrum_summary.pdf', dpi=150, bbox_inches='tight')
fig.savefig(out_dir / 'spectrum_summary.png', dpi=150, bbox_inches='tight')
print(f"Saved {out_dir / 'spectrum_summary.pdf'}")
plt.close(fig)

# ── Figure 2: Thickness comparison ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Order thickness by displacement amount
thickness_order = ['global_1mm', 'global_2mm', 'rvfw_2mm', 'rvfw_3mm', 'rvfw_5mm', 'rvfw_7mm']
thick_sorted = sorted(thickness, key=lambda c: thickness_order.index(c['label']))
labels = [c['label'] for c in thick_sorted]
x = np.arange(len(labels))

ax = axes[0]
ax.bar(x - 0.3, [c['w_PLV'] for c in thick_sorted], width=0.2, label='W_PLV', color='C0')
ax.bar(x - 0.1, [c['w_PRV'] for c in thick_sorted], width=0.2, label='W_PRV', color='C3')
ax.bar(x + 0.1, [c['w_Trans'] for c in thick_sorted], width=0.2, label='W_Trans', color='C2')
ax.bar(x + 0.3, [c['w_true'] for c in thick_sorted], width=0.2, label='W_true', color='gray', alpha=0.7)
ax.axhline(0, color='k', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Total septum work (J)')
ax.set_title('Thickness variants — absolute work magnitudes (severe PAH circ)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

ax = axes[1]
ratios_PLV_t = [c['w_PLV']/c['w_true'] for c in thick_sorted]
ratios_PRV_t = [c['w_PRV']/c['w_true'] for c in thick_sorted]
ratios_Trans_t = [c['w_Trans']/c['w_true'] for c in thick_sorted]
ax.plot(x, ratios_PLV_t, 'b-o', label='R_PLV', lw=2, ms=7)
ax.plot(x, ratios_PRV_t, 'r-o', label='R_PRV', lw=2, ms=7)
ax.plot(x, ratios_Trans_t, 'g-o', label='R_Trans', lw=2, ms=7)
ax.axhline(0, color='k', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Proxy ratio R')
ax.set_title('Proxy accuracy across thickness variants')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 0.30)

fig.suptitle('Thickness Variants: Effect of wall thickening on septal proxy accuracy '
             '(severe PAH circulation, RV_ESP ~70 mmHg)',
             fontsize=12)
plt.tight_layout()
out_dir2 = ROOT / 'results' / 'analysis' / 'transventricular_thickness'
out_dir2.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir2 / 'thickness_summary.pdf', dpi=150, bbox_inches='tight')
fig.savefig(out_dir2 / 'thickness_summary.png', dpi=150, bbox_inches='tight')
print(f"Saved {out_dir2 / 'thickness_summary.pdf'}")
plt.close(fig)

# ── Print summary table ──────────────────────────────────────────────────────
print("\n=== SPECTRUM SUMMARY ===")
print(f"{'Case':<18} {'RV':>4} {'LV':>4} {'W_true':>11} {'W_PLV':>11} {'W_PRV':>11} {'W_Trans':>11} {'R_PLV':>7} {'R_PRV':>7} {'R_Trans':>8}")
for c in spectrum:
    print(f"{c['label']:<18} {c['rv_esp']:>4.0f} {c['lv_esp']:>4.0f} "
          f"{c['w_true']:>11.4e} {c['w_PLV']:>11.4e} {c['w_PRV']:>11.4e} {c['w_Trans']:>11.4e} "
          f"{c['w_PLV']/c['w_true']:>7.3f} {c['w_PRV']/c['w_true']:>7.3f} {c['w_Trans']/c['w_true']:>8.3f}")

print("\n=== THICKNESS SUMMARY (severe PAH) ===")
print(f"{'Variant':<18} {'RV':>4} {'W_true':>11} {'W_PLV':>11} {'W_PRV':>11} {'W_Trans':>11} {'R_PLV':>7} {'R_PRV':>7} {'R_Trans':>8}")
for c in thick_sorted:
    print(f"{c['label']:<18} {c['rv_esp']:>4.0f} "
          f"{c['w_true']:>11.4e} {c['w_PLV']:>11.4e} {c['w_PRV']:>11.4e} {c['w_Trans']:>11.4e} "
          f"{c['w_PLV']/c['w_true']:>7.3f} {c['w_PRV']/c['w_true']:>7.3f} {c['w_Trans']/c['w_true']:>8.3f}")
