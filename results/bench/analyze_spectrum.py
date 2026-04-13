#!/usr/bin/env python3
"""
Spectrum Analysis: Disease severity gradient for UKB biventricular simulations.

Analyzes how internal work, proxy accuracy, and septum mechanics change
across healthy → severe PAH using 10-beat converged simulations.

Usage:
    python3 analyze_spectrum.py [results_dir]
    python3 analyze_spectrum.py results/sims/ukb_10beats_spectrum
"""

import sys
import numpy as np
from pathlib import Path
from scipy import stats

# ── Configuration ─────────────────────────────────────────────
RBASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/sims/ukb_10beats_spectrum")
CASES = ["healthy", "healthy_low", "mild", "moderate", "moderate_severe", "severe"]
T_CYCLE = 0.8

# Disease severity index (for correlation analysis)
# Based on target RV ESP from circ params
SEVERITY_INDEX = {
    "healthy": 0, "healthy_low": 1, "mild": 2,
    "moderate": 3, "moderate_severe": 4, "severe": 5,
}


def load_metrics(name):
    p = RBASE / f"UKB_10beats_{name}" / "metrics" / "metrics_downsample_1.npy"
    return np.load(p, allow_pickle=True).item()


def beat_work(m, key):
    """Net work over the stored beat (handles full or last-beat-only)."""
    arr = np.array(m[key])
    return arr[-1] - arr[0]


def fmt(val, width=10, decimals=3):
    return f"{val:>{width}.{decimals}f}"


# ── Load Data ─────────────────────────────────────────────────
data = {name: load_metrics(name) for name in CASES}

print("=" * 120)
print("DISEASE SEVERITY SPECTRUM — UKB Biventricular, 10 Beats, Geometric Septum Tagging")
print("=" * 120)

for name in CASES:
    t = np.array(data[name]['time'])
    print(f"  {name}: {len(t)} pts, t=[{t[0]:.3f}, {t[-1]:.3f}]s")

# ══════════════════════════════════════════════════════════════
# 1. HEMODYNAMICS
# ══════════════════════════════════════════════════════════════
print("\n\n" + "═" * 100)
print("  1. HEMODYNAMICS")
print("═" * 100)
header = f"{'Case':<18}"
for col in ['ESP_LV', 'EDP_LV', 'ESP_RV', 'EDP_RV', 'SV_LV', 'EF_LV', 'SV_RV', 'EF_RV', 'P_trans']:
    header += f"{col:>10}"
print(header)
print("-" * 108)

hemo = {}
for name in CASES:
    m = data[name]
    plv, prv = np.array(m['p_LV']), np.array(m['p_RV'])
    vlv = np.array(m.get('V_LV_FEM', m.get('V_LV')))
    vrv = np.array(m.get('V_RV_FEM', m.get('V_RV')))

    h = {
        'ESP_LV': np.max(plv), 'EDP_LV': plv[0],
        'ESP_RV': np.max(prv), 'EDP_RV': prv[0],
        'EDV_LV': np.max(vlv), 'ESV_LV': np.min(vlv),
        'EDV_RV': np.max(vrv), 'ESV_RV': np.min(vrv),
    }
    h['SV_LV'] = h['EDV_LV'] - h['ESV_LV']
    h['EF_LV'] = h['SV_LV'] / h['EDV_LV'] * 100 if h['EDV_LV'] > 0 else 0
    h['SV_RV'] = h['EDV_RV'] - h['ESV_RV']
    h['EF_RV'] = h['SV_RV'] / h['EDV_RV'] * 100 if h['EDV_RV'] > 0 else 0
    h['P_trans'] = h['ESP_LV'] - h['ESP_RV']
    h['RV_LV_ratio'] = h['ESP_RV'] / h['ESP_LV'] if h['ESP_LV'] > 0 else 0
    hemo[name] = h

    line = f"{name:<18}"
    for col in ['ESP_LV', 'EDP_LV', 'ESP_RV', 'EDP_RV']:
        line += f"{h[col]:>10.1f}"
    line += f"{h['SV_LV']:>10.1f}{h['EF_LV']:>9.1f}%"
    line += f"{h['SV_RV']:>10.1f}{h['EF_RV']:>9.1f}%"
    line += f"{h['P_trans']:>10.1f}"
    print(line)


# ══════════════════════════════════════════════════════════════
# 2. INTERNAL WORK BY REGION
# ══════════════════════════════════════════════════════════════
print("\n\n" + "═" * 100)
print("  2. INTERNAL WORK BY REGION (last beat)")
print("═" * 100)
print(f"{'Case':<18} {'W_LV':>12} {'W_RV':>12} {'W_Sept':>12} {'W_Total':>12} {'%LV':>7} {'%RV':>7} {'%Sept':>7} {'RV/LV':>8}")
print("-" * 100)

work_by_region = {}
for name in CASES:
    m = data[name]
    w = {r: beat_work(m, f'work_true_{r}') for r in ['LV', 'RV', 'Septum']}
    w['Total'] = sum(w.values())
    work_by_region[name] = w

    pcts = {r: w[r] / w['Total'] * 100 if w['Total'] != 0 else 0 for r in ['LV', 'RV', 'Septum']}
    rv_lv = w['RV'] / w['LV'] if w['LV'] != 0 else 0

    print(f"{name:<18} {w['LV']:>12.2e} {w['RV']:>12.2e} {w['Septum']:>12.2e} {w['Total']:>12.2e} "
          f"{pcts['LV']:>6.1f}% {pcts['RV']:>6.1f}% {pcts['Septum']:>6.1f}% {rv_lv:>8.3f}")


# ══════════════════════════════════════════════════════════════
# 3. WORK BY DIRECTION (% of regional total)
# ══════════════════════════════════════════════════════════════
print("\n\n" + "═" * 100)
print("  3. WORK DECOMPOSITION BY DIRECTION (% of regional total)")
print("═" * 100)
print(f"{'Case':<18} {'Region':<10} {'ff':>8} {'ss':>8} {'nn':>8} {'cross':>8} {'ff+cross':>10}")
print("-" * 70)

for name in CASES:
    m = data[name]
    for region in ['LV', 'RV', 'Septum']:
        w_true = beat_work(m, f'work_true_{region}')
        if abs(w_true) < 1e-15:
            continue
        pcts = {}
        for comp in ['ff', 'ss', 'nn', 'cross']:
            k = f'work_{comp}_{region}'
            pcts[comp] = beat_work(m, k) / w_true * 100 if k in m else 0
        pcts['ff_cross'] = pcts['ff'] + pcts['cross']
        print(f"{name:<18} {region:<10} {pcts['ff']:>7.1f}% {pcts['ss']:>7.1f}% "
              f"{pcts['nn']:>7.1f}% {pcts['cross']:>7.1f}% {pcts['ff_cross']:>9.1f}%")


# ══════════════════════════════════════════════════════════════
# 4. SEPTUM PROXY VALIDATION — ABSOLUTE & DIRECTIONAL
# ══════════════════════════════════════════════════════════════
print("\n\n" + "═" * 100)
print("  4. SEPTUM PROXY VALIDATION")
print("═" * 100)

# 4a. Absolute ratios
print("\n  4a. Proxy-to-truth ratio (W_proxy / W_true):")
print(f"{'Case':<18} {'W_true':>12} {'PLV_ff':>10} {'PRV_ff':>10} {'Trans_ff':>10} {'PLV_ll':>10} {'Trans_ll':>10}")
print("-" * 80)

proxy_data = {}
for name in CASES:
    m = data[name]
    w_true = beat_work(m, 'work_true_Septum')
    proxies = {}
    for pk, label in [
        ('work_ps_ff_Septum_PLV', 'PLV_ff'), ('work_ps_ff_Septum_PRV', 'PRV_ff'),
        ('work_ps_ff_Septum_Trans', 'Trans_ff'),
        ('work_ps_ll_Septum_PLV', 'PLV_ll'), ('work_ps_ll_Septum_Trans', 'Trans_ll'),
    ]:
        if pk in m:
            proxies[label] = beat_work(m, pk) / w_true if abs(w_true) > 1e-15 else 0
    proxy_data[name] = {'w_true': w_true, **proxies}

    line = f"{name:<18} {w_true:>12.2e}"
    for label in ['PLV_ff', 'PRV_ff', 'Trans_ff', 'PLV_ll', 'Trans_ll']:
        line += f"{proxies.get(label, 0):>10.3f}"
    print(line)

# 4b. Directional tracking — does the proxy change in the same direction as truth?
print("\n  4b. Directional tracking across disease spectrum:")
print("      (Does the proxy trend match the truth trend as RV pressure rises?)")
print()

severity = np.array([SEVERITY_INDEX[n] for n in CASES])
w_true_arr = np.array([proxy_data[n]['w_true'] for n in CASES])

print(f"  {'Proxy':<15} {'Pearson r':>10} {'p-value':>10} {'Spearman ρ':>12} {'p-value':>10} {'Tracks?':>10}")
print("  " + "-" * 70)

for label in ['PLV_ff', 'PRV_ff', 'Trans_ff', 'PLV_ll', 'Trans_ll']:
    proxy_arr = np.array([proxy_data[n].get(label, 0) * proxy_data[n]['w_true'] for n in CASES])

    # Correlation between proxy work and true work across cases
    if np.std(proxy_arr) > 1e-20 and np.std(w_true_arr) > 1e-20:
        r_pearson, p_pearson = stats.pearsonr(w_true_arr, proxy_arr)
        r_spearman, p_spearman = stats.spearmanr(w_true_arr, proxy_arr)
    else:
        r_pearson, p_pearson, r_spearman, p_spearman = 0, 1, 0, 1

    tracks = "YES" if r_pearson > 0.8 and p_pearson < 0.05 else ("WEAK" if r_pearson > 0.5 else "NO")
    print(f"  {label:<15} {r_pearson:>10.4f} {p_pearson:>10.4f} {r_spearman:>12.4f} {p_spearman:>10.4f} {tracks:>10}")

# 4c. Ratio stability — does the proxy/truth ratio stay constant across severity?
print("\n  4c. Ratio stability across disease spectrum:")
print("      (Constant ratio = proxy is a reliable scaled estimator)")
print()
print(f"  {'Proxy':<15} {'Mean ratio':>12} {'Std ratio':>12} {'CV%':>8} {'Min':>8} {'Max':>8} {'Stable?':>10}")
print("  " + "-" * 75)

for label in ['PLV_ff', 'PRV_ff', 'Trans_ff', 'PLV_ll', 'Trans_ll']:
    ratios = np.array([proxy_data[n].get(label, 0) for n in CASES])
    mean_r = np.mean(ratios)
    std_r = np.std(ratios)
    cv = abs(std_r / mean_r * 100) if mean_r != 0 else 999
    stable = "YES" if cv < 20 else ("FAIR" if cv < 50 else "NO")
    print(f"  {label:<15} {mean_r:>12.4f} {std_r:>12.4f} {cv:>7.1f}% {np.min(ratios):>8.3f} {np.max(ratios):>8.3f} {stable:>10}")

# 4d. Which pressure is BEST for septum at each severity?
print("\n  4d. Best septum pressure proxy at each severity level:")
print(f"  {'Case':<18} {'RV/LV':>8} {'Best_ff':>10} {'Best_ll':>10} {'|err_ff|':>10} {'|err_ll|':>10}")
print("  " + "-" * 70)

for name in CASES:
    rv_lv = hemo[name]['RV_LV_ratio']
    # Best = closest ratio to 1 (or closest to the mean ratio)
    ff_proxies = {k: proxy_data[name].get(k, 0) for k in ['PLV_ff', 'PRV_ff', 'Trans_ff']}
    ll_proxies = {k: proxy_data[name].get(k, 0) for k in ['PLV_ll', 'Trans_ll']}

    best_ff = min(ff_proxies, key=lambda k: abs(ff_proxies[k] - 1)) if ff_proxies else '?'
    best_ll = min(ll_proxies, key=lambda k: abs(ll_proxies[k] - 1)) if ll_proxies else '?'
    err_ff = abs(ff_proxies[best_ff] - 1) * 100
    err_ll = abs(ll_proxies[best_ll] - 1) * 100

    print(f"  {name:<18} {rv_lv:>8.3f} {best_ff:>10} {best_ll:>10} {err_ff:>9.1f}% {err_ll:>9.1f}%")


# ══════════════════════════════════════════════════════════════
# 5. FREE WALL PROXY ACCURACY
# ══════════════════════════════════════════════════════════════
print("\n\n" + "═" * 100)
print("  5. FREE WALL PROXY ACCURACY (proxy / true work ratio)")
print("═" * 100)
print(f"{'Case':<18} {'LV_P×ε_ff':>10} {'LV_P×ε_ll':>10} {'|':>3} {'RV_P×ε_ff':>10} {'RV_P×ε_ll':>10}")
print("-" * 60)

for name in CASES:
    m = data[name]
    results = {}
    for chamber in ['LV', 'RV']:
        w_true = beat_work(m, f'work_true_{chamber}')
        for suffix, key in [('ff', f'work_ps_ff_{chamber}'), ('ll', f'work_ps_ll_{chamber}')]:
            results[f'{chamber}_{suffix}'] = beat_work(m, key) / w_true if key in m and abs(w_true) > 1e-15 else 0
    print(f"{name:<18} {results['LV_ff']:>10.3f} {results['LV_ll']:>10.3f} {'|':>3} "
          f"{results['RV_ff']:>10.3f} {results['RV_ll']:>10.3f}")


# ══════════════════════════════════════════════════════════════
# 6. TRANSMURAL PRESSURE & SEPTUM MECHANICS
# ══════════════════════════════════════════════════════════════
print("\n\n" + "═" * 100)
print("  6. TRANSMURAL PRESSURE GRADIENT & SEPTUM MECHANICS")
print("═" * 100)
print(f"{'Case':<18} {'P_LV':>8} {'P_RV':>8} {'P_trans':>8} {'RV/LV':>8} {'E_ff':>8} {'E_nn':>8} {'W_ff%':>8} {'W_nn%':>8}")
print("-" * 85)

for name in CASES:
    m = data[name]
    h = hemo[name]
    w_sept = beat_work(m, 'work_true_Septum')
    w_ff = beat_work(m, 'work_ff_Septum') / w_sept * 100 if 'work_ff_Septum' in m and abs(w_sept) > 1e-15 else 0
    w_nn = beat_work(m, 'work_nn_Septum') / w_sept * 100 if 'work_nn_Septum' in m and abs(w_sept) > 1e-15 else 0
    eff = np.min(m.get('mean_E_ff_Septum', [0]))
    enn = np.min(m.get('mean_E_nn_Septum', [0]))

    print(f"{name:<18} {h['ESP_LV']:>8.1f} {h['ESP_RV']:>8.1f} {h['P_trans']:>8.1f} "
          f"{h['RV_LV_ratio']:>8.3f} {eff:>8.4f} {enn:>8.4f} {w_ff:>7.1f}% {w_nn:>7.1f}%")

# Correlation: P_trans vs septum fiber strain
print("\n  Correlation: transmural pressure vs septum mechanics")
p_trans = np.array([hemo[n]['P_trans'] for n in CASES])
eff_sept = np.array([np.min(data[n].get('mean_E_ff_Septum', [0])) for n in CASES])
w_ff_sept = np.array([beat_work(data[n], 'work_ff_Septum') for n in CASES])
w_nn_sept = np.array([beat_work(data[n], 'work_nn_Septum') for n in CASES])

r, p = stats.pearsonr(p_trans, eff_sept)
print(f"    P_trans vs E_ff_septum: r={r:.4f}, p={p:.4f}")
r, p = stats.pearsonr(p_trans, w_ff_sept)
print(f"    P_trans vs W_ff_septum: r={r:.4f}, p={p:.4f}")
r, p = stats.pearsonr(p_trans, w_nn_sept)
print(f"    P_trans vs W_nn_septum: r={r:.4f}, p={p:.4f}")

rv_lv = np.array([hemo[n]['RV_LV_ratio'] for n in CASES])
r, p = stats.pearsonr(rv_lv, w_ff_sept / (w_ff_sept + w_nn_sept))
print(f"    RV/LV ratio vs W_ff/(W_ff+W_nn): r={r:.4f}, p={p:.4f}")


# ══════════════════════════════════════════════════════════════
# 7. ENERGY BALANCE CHECK
# ══════════════════════════════════════════════════════════════
print("\n\n" + "═" * 100)
print("  7. ENERGY BALANCE (sanity check)")
print("═" * 100)
print(f"{'Case':<18} {'W_internal':>12} {'W_boundary':>12} {'W_robin':>12} {'Balance':>10}")
print("-" * 70)

for name in CASES:
    m = data[name]
    w_int = beat_work(m, 'work_true_Whole')
    w_bnd = sum(beat_work(m, k) for k in ['work_boundary_exact_LV', 'work_boundary_exact_RV'] if k in m)
    w_rob = sum(beat_work(m, k) for k in ['work_robin_epi', 'work_robin_base'] if k in m)
    w_ext = w_bnd + w_rob
    bal = (w_int - w_ext) / abs(w_int) * 100 if abs(w_int) > 1e-15 else 0
    print(f"{name:<18} {w_int:>12.2e} {w_bnd:>12.2e} {w_rob:>12.2e} {bal:>9.1f}%")


print("\n" + "=" * 120)
print("Analysis complete.")
