#!/usr/bin/env python3
"""
plot_loops.py

Generates two key figures for the meeting:
1. loops.png: 
   - Top: PV Loops (LV & RV)
   - Mid: Pressure-Strain Loops (LV, Septum, RV)
   - Bot: Stress-Strain Loops (LV, Septum, RV)

2. engineering_debug.png:
   - Cumulative Energy Balance (Internal vs External)
   - Instantaneous Work Power
   - Septum Pressure Proxy Comparison

Usage:
  python3 plot_loops.py <path_to_results_folder>
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# --- 1. Data Loading ---
def load_metrics(results_dir):
    path = Path(results_dir)
    # Smart find for the .npy file
    if path.suffix == ".npy":
        fpath = path
    else:
        # Try downsample 1 first, then others
        candidates = sorted(list(path.glob("metrics_downsample_*.npy")), key=lambda p: len(p.name))
        if not candidates:
            print(f"❌ No metrics_downsample_*.npy found in {path}")
            sys.exit(1)
        fpath = candidates[0]

    print(f"📂 Loading: {fpath.name}")
    return np.load(fpath, allow_pickle=True).item()

def get_arr(metrics, keys, min_len=None):
    """Helper to safely get array from list of possible keys"""
    for k in keys:
        if k in metrics:
            arr = np.array(metrics[k])
            if min_len:
                return arr[:min_len]
            return arr
    return None

# --- 2. Plotting Functions ---

def plot_clinical_dashboard(metrics, outdir):
    """Creates the 3-row grid for the cardiologist."""
    
    # Setup Data
    p_LV = get_arr(metrics, ["p_LV"])
    v_LV = get_arr(metrics, ["V_LV"])
    p_RV = get_arr(metrics, ["p_RV"])
    v_RV = get_arr(metrics, ["V_RV"])
    
    # Determine safe length
    if p_LV is None: return
    N = len(p_LV)
    
    # Strains (E_ff)
    e_LV = get_arr(metrics, ["mean_E_ff_LV"], N)
    e_Sep = get_arr(metrics, ["mean_E_ff_Septum"], N)
    e_RV = get_arr(metrics, ["mean_E_ff_RV"], N)
    
    # Stresses (S_ff)
    s_LV = get_arr(metrics, ["mean_S_ff_LV"], N)
    s_Sep = get_arr(metrics, ["mean_S_ff_Septum"], N)
    s_RV = get_arr(metrics, ["mean_S_ff_RV"], N)

    # --- FIGURE SETUP ---
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 6, figure=fig)
    fig.suptitle("Loops: Hemodynamics & Mechanics", fontsize=18, fontweight='bold')

    # ROW 1: PV LOOPS (Spans 3 columns each)
    ax_pv_lv = fig.add_subplot(gs[0, 0:3])
    ax_pv_rv = fig.add_subplot(gs[0, 3:6])
    
    # ROW 2: PRESSURE-STRAIN (Spans 2 cols each)
    ax_ps_lv = fig.add_subplot(gs[1, 0:2])
    ax_ps_sep = fig.add_subplot(gs[1, 2:4])
    ax_ps_rv = fig.add_subplot(gs[1, 4:6])
    
    # ROW 3: STRESS-STRAIN (Spans 2 cols each)
    ax_ss_lv = fig.add_subplot(gs[2, 0:2])
    ax_ss_sep = fig.add_subplot(gs[2, 2:4])
    ax_ss_rv = fig.add_subplot(gs[2, 4:6])

    # --- PLOTTING ROW 1 (PV) ---
    def plot_cycle(ax, x, y, color, title, xlabel, ylabel):
        if x is None or y is None: return
        ax.plot(x, y, color=color, linewidth=2.5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        # Arrow
        mid = len(x)//2
        ax.arrow(x[mid], y[mid], x[mid+1]-x[mid], y[mid+1]-y[mid], 
                 color='k', head_width=0.02*(max(x)-min(x)), length_includes_head=True)

    plot_cycle(ax_pv_lv, v_LV, p_LV, 'tab:blue', "LV PV Loop", "Volume (mL)", "Pressure (mmHg)")
    plot_cycle(ax_pv_rv, v_RV, p_RV, 'tab:red', "RV PV Loop", "Volume (mL)", "Pressure (mmHg)")
    
    # Add Stroke Volume Text
    if v_LV is not None:
        sv = v_LV.max() - v_LV.min()
        ax_pv_lv.text(0.5, 0.5, f"SV: {sv:.1f} mL", transform=ax_pv_lv.transAxes, 
                      ha='center', bbox=dict(facecolor='white', alpha=0.8))

    # --- PLOTTING ROW 2 (Pressure-Strain) ---
    # Note: Septum usually plotted against LV Pressure in clinical echo
    plot_cycle(ax_ps_lv, e_LV, p_LV, 'tab:blue', "LV Pressure-Strain", "Strain (%)", "P_LV (mmHg)")
    plot_cycle(ax_ps_sep, e_Sep, p_LV, 'tab:green', "Septal Pressure-Strain (vs LVP)", "Strain (%)", "P_LV (mmHg)")
    plot_cycle(ax_ps_rv, e_RV, p_RV, 'tab:red', "RV Pressure-Strain", "Strain (%)", "P_RV (mmHg)")

    # --- PLOTTING ROW 3 (Stress-Strain) ---
    # Convert Pa to kPa for readability
    def to_kpa(arr): return arr * 1e-3 if arr is not None else None
    
    plot_cycle(ax_ss_lv, e_LV, to_kpa(s_LV), 'tab:blue', "LV Stress-Strain", "Strain (%)", "Stress (kPa)")
    plot_cycle(ax_ss_sep, e_Sep, to_kpa(s_Sep), 'tab:green', "Septal Stress-Strain", "Strain (%)", "Stress (kPa)")
    plot_cycle(ax_ss_rv, e_RV, to_kpa(s_RV), 'tab:red', "RV Stress-Strain", "Strain (%)", "Stress (kPa)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = outdir / "loops.png"
    plt.savefig(outpath, dpi=150)
    print(f"✅ Saved: {outpath}")
    plt.close()

def plot_engineering_debug(metrics, outdir):
    """Creates the energy balance analysis for the engineering team."""
    
    # 1. Get Instantaneous Power (Work per step)
    # True Internal Work (Sum of regions)
    w_true_lv = get_arr(metrics, ["work_true_LV"])
    w_true_rv = get_arr(metrics, ["work_true_RV"])
    w_true_sep = get_arr(metrics, ["work_true_Septum"])
    
    if w_true_lv is None:
        print("⚠ Missing work metrics. Skipping energy debug.")
        return

    # Total Internal Power
    # Note: If 'work_true_Whole' exists, use it. Else sum regions.
    w_int_total_step = get_arr(metrics, ["work_true_Whole"])
    if w_int_total_step is None:
        w_int_total_step = w_true_lv + w_true_rv + w_true_sep

    # External Power (PV Proxies)
    w_pv_lv_step = get_arr(metrics, ["work_proxy_pv_LV"], len(w_int_total_step))
    w_pv_rv_step = get_arr(metrics, ["work_proxy_pv_RV"], len(w_int_total_step))
    w_ext_total_step = w_pv_lv_step + w_pv_rv_step

    # 2. Integrate to get Cumulative Energy (Joules)
    E_int_cum = np.cumsum(w_int_total_step)
    E_ext_cum = np.cumsum(w_ext_total_step)
    
    # Time array
    time = get_arr(metrics, ["time"], len(w_int_total_step))

    # --- FIGURE ---
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("Engineering Debug: Energy Balance & Consistency", fontsize=16, fontweight='bold')

    # PLOT 1: Cumulative Energy (The "Devil's Advocate" Plot)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, E_ext_cum, 'k--', linewidth=2, label='External Work (PV Area)')
    ax1.plot(time, E_int_cum, 'b-', linewidth=2, alpha=0.8, label='Internal Work (Strain Energy)')
    
    # Highlight the discrepancy
    final_err = (E_ext_cum[-1] - E_int_cum[-1]) / E_ext_cum[-1] * 100
    ax1.fill_between(time, E_int_cum, E_ext_cum, color='red', alpha=0.1, label='Energy Error')
    
    ax1.set_title(f"Global Energy Balance (Error: {final_err:.1f}%)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Cumulative Work (Joules)")
    ax1.set_xlabel("Time (s)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PLOT 2: Septum Pressure Proxy Investigation
    # Compare True Septum Work vs the Proxies
    ax2 = fig.add_subplot(gs[0, 1])
    
    w_sep_true_cum = np.cumsum(w_true_sep)
    ax2.plot(time, w_sep_true_cum, 'k-', linewidth=2.5, label='True Septal Work')
    
    # Try to find the proxies we added
    proxies = {
        "work_ps_index_Septum_Trans": ("Trans-Septal (P_LV - P_RV)", "green"),
        "work_ps_index_Septum_PLV":   ("LV Pressure Only", "blue"),
        "work_ps_index_Septum_PRV":   ("RV Pressure Only", "red"),
        "work_ps_index_Septum":       ("Standard (Old)", "gray")
    }
    
    for key, (label, color) in proxies.items():
        arr = get_arr(metrics, [key], len(time))
        if arr is not None:
            ax2.plot(time, np.cumsum(arr), linestyle='--', color=color, label=f'Proxy: {label}')

    ax2.set_title("Septum Work: True vs Proxies", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Cumulative Work (Joules)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # PLOT 3: Instantaneous Power (Where does the error happen?)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time, w_int_total_step, 'b-', label='Internal Power (S:dE)')
    ax3.plot(time, w_ext_total_step, 'k--', label='External Power (P*dV)')
    ax3.set_title("Instantaneous Power Input", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Power (Watts/Step)")
    ax3.set_xlabel("Time (s)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = outdir / "engineering_debug.png"
    plt.savefig(outpath, dpi=150)
    print(f"✅ Saved: {outpath}")
    plt.close()

# --- Main ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_loops.py <results_folder>")
        sys.exit(1)
        
    res_dir = Path(sys.argv[1])
    metrics = load_metrics(res_dir)
    
    plot_clinical_dashboard(metrics, res_dir)
    plot_engineering_debug(metrics, res_dir)
