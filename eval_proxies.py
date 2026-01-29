#!/usr/bin/env python3
"""
eval_proxies.py

Quantitatively compares True Internal Work vs. Clinical Proxies.
Generates:
1. 'proxy_validation.png': Bar charts of Total Work and Regression plots of Power.
2. Console Report: Exact Error % and Correlation Coefficients.

Usage:
  python3 eval_proxies.py <results_folder>
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr

def load_metrics(results_dir):
    path = Path(results_dir)
    if path.suffix == ".npy": fpath = path
    else:
        # Prefer full resolution or downsample 1 for accuracy
        candidates = sorted(list(path.glob("metrics_downsample_*.npy")), key=lambda p: len(p.name))
        if not candidates: return None
        fpath = candidates[0]
    
    print(f"📂 Loading: {fpath.name}")
    return np.load(fpath, allow_pickle=True).item()

def get_data(metrics, key):
    return np.array(metrics.get(key, [])) if key in metrics else None

def calculate_stats(true_power, proxy_power):
    """Calculates quantitative agreement metrics."""
    if true_power is None or proxy_power is None: return None
    if len(true_power) != len(proxy_power):
        min_len = min(len(true_power), len(proxy_power))
        true_power = true_power[:min_len]
        proxy_power = proxy_power[:min_len]

    # 1. Total Work (Integral) - Joules
    # Assuming uniform dt (sum is proportional to integral)
    W_true = np.sum(true_power)
    W_proxy = np.sum(proxy_power)
    
    # Avoid div by zero
    if abs(W_true) < 1e-9: ratio = 0.0
    else: ratio = W_proxy / W_true
    
    pct_error = (W_proxy - W_true) / W_true * 100.0

    # 2. Temporal Correlation (Pearson R)
    r_val, _ = pearsonr(true_power, proxy_power)
    
    return {
        "W_true": W_true,
        "W_proxy": W_proxy,
        "Ratio": ratio,
        "Error_Pct": pct_error,
        "R_squared": r_val**2
    }

def analyze_proxies(metrics, outdir):
    # --- DATA PREP ---
    # True Work Powers (Watts/step)
    tru_lv = get_data(metrics, "work_true_LV")
    tru_sep = get_data(metrics, "work_true_Septum")
    tru_rv = get_data(metrics, "work_true_RV")

    # Proxies - LV
    # 1. PV Loop Proxy (P*dV)
    prx_pv_lv = get_data(metrics, "work_proxy_pv_LV")
    # 2. PS Loop Proxy (P*dE*V)
    prx_ps_lv = get_data(metrics, "work_ps_index_LV")

    # Proxies - Septum (The Investigation)
    prx_sep_trans = get_data(metrics, "work_ps_index_Septum_Trans") # P_LV - P_RV
    prx_sep_plv   = get_data(metrics, "work_ps_index_Septum_PLV")   # P_LV
    prx_sep_prv   = get_data(metrics, "work_ps_index_Septum_PRV")   # P_RV
    prx_sep_mean  = get_data(metrics, "work_ps_index_Septum_Mean")  # Mean P

    # Proxies - RV
    prx_pv_rv = get_data(metrics, "work_proxy_pv_RV")
    prx_ps_rv = get_data(metrics, "work_ps_index_RV")

    # --- FIGURE SETUP ---
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2])
    fig.suptitle("Quantitative Validation: True Work vs. Proxies", fontsize=16, fontweight='bold')

    # --- ROW 1: TOTAL CYCLE ENERGY COMPARISON (Bar Charts) ---
    
    # 1. LV Comparison
    ax_lv = fig.add_subplot(gs[0, 0])
    stats_lv_pv = calculate_stats(tru_lv, prx_pv_lv)
    stats_lv_ps = calculate_stats(tru_lv, prx_ps_lv)
    
    vals = [stats_lv_pv["W_true"], stats_lv_pv["W_proxy"], stats_lv_ps["W_proxy"]]
    labels = ["True (S:E)", "PV (P·dV)", "PS (P·dE)"]
    colors = ['black', 'tab:blue', 'tab:cyan']
    
    bars = ax_lv.bar(labels, vals, color=colors, alpha=0.7)
    ax_lv.bar_label(bars, fmt='%.1e', padding=3)
    ax_lv.set_title("Left Ventricle: Total Work", fontweight='bold')
    ax_lv.set_ylabel("Work (Arbitrary Units / Joules)")
    
    # 2. Septum Comparison (The Big Question)
    ax_sep = fig.add_subplot(gs[0, 1])
    
    # Stats for Septum variants
    s_trans = calculate_stats(tru_sep, prx_sep_trans)
    s_plv   = calculate_stats(tru_sep, prx_sep_plv)
    s_mean  = calculate_stats(tru_sep, prx_sep_mean)
    
    sep_vals = [s_trans["W_true"], s_trans["W_proxy"], s_plv["W_proxy"], s_mean["W_proxy"]]
    sep_lbls = ["True", "Trans\n(LV-RV)", "P_LV", "Mean P"]
    sep_cols = ['black', 'tab:green', 'tab:blue', 'tab:purple']
    
    bars_sep = ax_sep.bar(sep_lbls, sep_vals, color=sep_cols, alpha=0.7)
    ax_sep.bar_label(bars_sep, fmt='%.1e', padding=3)
    ax_sep.set_title("Septum: Which Pressure Fits?", fontweight='bold')

    # 3. RV Comparison
    ax_rv = fig.add_subplot(gs[0, 2])
    stats_rv_pv = calculate_stats(tru_rv, prx_pv_rv)
    stats_rv_ps = calculate_stats(tru_rv, prx_ps_rv)
    
    rv_vals = [stats_rv_pv["W_true"], stats_rv_pv["W_proxy"], stats_rv_ps["W_proxy"]]
    rv_lbls = ["True", "PV", "PS"]
    rv_cols = ['black', 'tab:red', 'tab:orange']
    
    bars_rv = ax_rv.bar(rv_lbls, rv_vals, color=rv_cols, alpha=0.7)
    ax_rv.bar_label(bars_rv, fmt='%.1e', padding=3)
    ax_rv.set_title("Right Ventricle: Total Work", fontweight='bold')

    # --- ROW 2: INSTANTANEOUS POWER REGRESSION (Scatter) ---
    # Plot True Power (X) vs Proxy Power (Y). Ideal is y=x line.
    
    def plot_regression(ax, true_p, proxy_p, label, color):
        if true_p is None or proxy_p is None: return
        min_l = min(len(true_p), len(proxy_p))
        x = true_p[:min_l]
        y = proxy_p[:min_l]
        
        # Scatter
        ax.scatter(x, y, alpha=0.3, s=10, color=color, label=label)
        
        # Fit Line
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color=color, linewidth=1.5, linestyle='-')
        
        return m, b

    # 4. LV Power Regression
    ax_reg_lv = fig.add_subplot(gs[1, 0])
    ax_reg_lv.plot([min(tru_lv), max(tru_lv)], [min(tru_lv), max(tru_lv)], 'k--', alpha=0.5, label="Ideal (y=x)")
    
    m_pv, _ = plot_regression(ax_reg_lv, tru_lv, prx_pv_lv, "PV Proxy", "tab:blue")
    m_ps, _ = plot_regression(ax_reg_lv, tru_lv, prx_ps_lv, "PS Proxy", "tab:cyan")
    
    ax_reg_lv.set_title(f"LV Power Correlation\nSlope PV={m_pv:.2f}, PS={m_ps:.2f}")
    ax_reg_lv.set_xlabel("True Power (S:E)")
    ax_reg_lv.set_ylabel("Proxy Power")
    ax_reg_lv.legend()
    ax_reg_lv.grid(True, alpha=0.3)

    # 5. Septum Power Regression
    ax_reg_sep = fig.add_subplot(gs[1, 1])
    ax_reg_sep.plot([min(tru_sep), max(tru_sep)], [min(tru_sep), max(tru_sep)], 'k--', alpha=0.5)
    
    m_trans, _ = plot_regression(ax_reg_sep, tru_sep, prx_sep_trans, "Trans (LV-RV)", "tab:green")
    m_plv, _   = plot_regression(ax_reg_sep, tru_sep, prx_sep_plv, "P_LV Only", "tab:blue")
    
    ax_reg_sep.set_title(f"Septum Correlation\nSlope Trans={m_trans:.2f}, LV={m_plv:.2f}")
    ax_reg_sep.set_xlabel("True Power (S:E)")
    ax_reg_sep.legend()
    ax_reg_sep.grid(True, alpha=0.3)

    # 6. RV Power Regression
    ax_reg_rv = fig.add_subplot(gs[1, 2])
    ax_reg_rv.plot([min(tru_rv), max(tru_rv)], [min(tru_rv), max(tru_rv)], 'k--', alpha=0.5)
    
    m_rv_pv, _ = plot_regression(ax_reg_rv, tru_rv, prx_pv_rv, "PV Proxy", "tab:red")
    
    ax_reg_rv.set_title(f"RV Correlation\nSlope PV={m_rv_pv:.2f}")
    ax_reg_rv.set_xlabel("True Power (S:E)")
    ax_reg_rv.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = Path(outdir) / "proxy_validation.png"
    plt.savefig(outpath, dpi=150)
    print(f"✅ Saved plot: {outpath}")
    
    # --- PRINT TEXT REPORT ---
    print("\n" + "="*60)
    print(f"{'PROXY EVALUATION REPORT':^60}")
    print("="*60)
    print(f"{'Region / Proxy':<25} | {'Error %':>10} | {'Ratio':>8} | {'R² (Timing)':>10}")
    print("-" * 60)
    
    def pr(name, s):
        print(f"{name:<25} | {s['Error_Pct']:>10.1f}% | {s['Ratio']:>8.2f} | {s['R_squared']:>10.3f}")

    pr("LV: PV (P*dV)", stats_lv_pv)
    pr("LV: PS (P*dE)", stats_lv_ps)
    print("-" * 60)
    pr("Septum: Trans (LV-RV)", s_trans)
    pr("Septum: P_LV Only", s_plv)
    pr("Septum: Mean P", s_mean)
    print("-" * 60)
    pr("RV: PV (P*dV)", stats_rv_pv)
    pr("RV: PS (P*dE)", stats_rv_ps)
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 eval_proxies.py <results_folder>")
        sys.exit(1)
    
    metrics = load_metrics(sys.argv[1])
    if metrics:
        analyze_proxies(metrics, sys.argv[1])
