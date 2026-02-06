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

3. clinical_hemodynamics.png:
    - Reconstructed LV & RV PV Loops (Clinical)
    
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

    if v_RV is not None:
        sv_rv = v_RV.max() - v_RV.min()
        ax_pv_rv.text(0.5, 0.5, f"SV: {sv_rv:.1f} mL", transform=ax_pv_rv.transAxes, 
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
    
    # Calculate Energy Gap (Missing Energy)
    W_int_end = E_int_cum[-1]
    W_ext_end = E_ext_cum[-1]
    
    # Dissipation Ratio: (Int - Ext) / Int 
    # If Ext > Int, this is negative (energy creation? or spring work?)
    # If Int > Ext, this is positive (energy dissipation)
    if abs(W_int_end) > 1e-9:
        dissipation_ratio = (W_int_end - W_ext_end) / W_int_end * 100.0
    else:
        dissipation_ratio = 0.0

    # Old metric for comparison: relative error
    final_err = (W_ext_end - W_int_end) / W_ext_end * 100
    
    ax1.fill_between(time, E_int_cum, E_ext_cum, color='red', alpha=0.1, label='Energy Gap')
    
    # Annotation on the plot
    text_str = f"Dissipation: {dissipation_ratio:.1f}%\n(Int: {W_int_end:.2f}J, Ext: {W_ext_end:.2f}J)"
    ax1.text(0.05, 0.85, text_str, transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), fontsize=10)

    ax1.set_title(f"Global Energy Balance", fontsize=12, fontweight='bold')
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

def plot_full_hemodynamics(metrics, outdir):
    """Creates the Clinical Hemodynamics dashboard (Reconstructed Volumes)."""
    
    # 1. Retrieve Data
    v_LV_clin = get_arr(metrics, ["V_LV_Clinical"])
    v_RV_clin = get_arr(metrics, ["V_RV_Clinical"])
    p_LV = get_arr(metrics, ["p_LV"])
    p_RV = get_arr(metrics, ["p_RV"])
    
    # Get Time (safely)
    if p_LV is not None:
        time = get_arr(metrics, ["time"], len(p_LV))
    else:
        time = None

    # Check existence
    if v_LV_clin is None or v_RV_clin is None:
        print("⚠ Missing 'V_LV_Clinical' or 'V_RV_Clinical'. Skipping full hemodynamics plot.")
        return

    # --- FIGURE SETUP ---
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("Estimated Full Organ Hemodynamics (Reconstructed)", fontsize=18, fontweight='bold')

    ax_pv_lv = fig.add_subplot(gs[0, 0])
    ax_pv_rv = fig.add_subplot(gs[0, 1])
    ax_vol = fig.add_subplot(gs[1, 0])
    ax_pres = fig.add_subplot(gs[1, 1])

    # --- PLOT 1: LV PV Loop (Clinical) ---
    ax_pv_lv.plot(v_LV_clin, p_LV, 'tab:blue', linewidth=2.5)
    ax_pv_lv.set_title("LV PV Loop (Clinical)", fontweight='bold')
    ax_pv_lv.set_xlabel("Volume (mL)")
    ax_pv_lv.set_ylabel("Pressure (mmHg)")
    ax_pv_lv.grid(True, alpha=0.3)
    
    # Annotate SV and EF
    sv_lv = v_LV_clin.max() - v_LV_clin.min()
    edv_lv = v_LV_clin.max()
    ef_lv = (sv_lv / edv_lv) * 100.0 if edv_lv > 0 else 0
    text_lv = f"SV: {sv_lv:.1f} mL\nEF: {ef_lv:.1f}%"
    ax_pv_lv.text(0.5, 0.5, text_lv, transform=ax_pv_lv.transAxes, 
                  ha='center', va='center', bbox=dict(facecolor='white', alpha=0.9))

    # --- PLOT 2: RV PV Loop (Clinical) ---
    ax_pv_rv.plot(v_RV_clin, p_RV, 'tab:red', linewidth=2.5)
    ax_pv_rv.set_title("RV PV Loop (Clinical)", fontweight='bold')
    ax_pv_rv.set_xlabel("Volume (mL)")
    ax_pv_rv.set_ylabel("Pressure (mmHg)")
    ax_pv_rv.grid(True, alpha=0.3)
    
    # Annotate SV and EF
    sv_rv = v_RV_clin.max() - v_RV_clin.min()
    edv_rv = v_RV_clin.max()
    ef_rv = (sv_rv / edv_rv) * 100.0 if edv_rv > 0 else 0
    text_rv = f"SV: {sv_rv:.1f} mL\nEF: {ef_rv:.1f}%"
    ax_pv_rv.text(0.5, 0.5, text_rv, transform=ax_pv_rv.transAxes, 
                  ha='center', va='center', bbox=dict(facecolor='white', alpha=0.9))

    # --- PLOT 3: Volume Traces ---
    if time is not None:
        ax_vol.plot(time, v_LV_clin, 'tab:blue', linewidth=2, label='LV Volume')
        ax_vol.plot(time, v_RV_clin, 'tab:red', linewidth=2, label='RV Volume')
        ax_vol.set_xlabel("Time (s)")
    else:
        ax_vol.plot(v_LV_clin, 'tab:blue', linewidth=2, label='LV Volume')
        ax_vol.plot(v_RV_clin, 'tab:red', linewidth=2, label='RV Volume')
        ax_vol.set_xlabel("Step")

    ax_vol.set_title("Clinical Volumes over Time", fontweight='bold')
    ax_vol.set_ylabel("Volume (mL)")
    ax_vol.legend()
    ax_vol.grid(True, alpha=0.3)

    # --- PLOT 4: Pressure Traces ---
    if time is not None:
        ax_pres.plot(time, p_LV, 'tab:blue', linewidth=2, label='LV Pressure')
        ax_pres.plot(time, p_RV, 'tab:red', linewidth=2, label='RV Pressure')
        ax_pres.set_xlabel("Time (s)")
    else:
        ax_pres.plot(p_LV, 'tab:blue', linewidth=2, label='LV Pressure')
        ax_pres.plot(p_RV, 'tab:red', linewidth=2, label='RV Pressure')
        ax_pres.set_xlabel("Step")

    ax_pres.set_title("Pressures over Time", fontweight='bold')
    ax_pres.set_ylabel("Pressure (mmHg)")
    ax_pres.legend()
    ax_pres.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = outdir / "clinical_hemodynamics.png"
    plt.savefig(outpath, dpi=150)
    print(f"✅ Saved: {outpath}")
    plt.close()

def plot_stress_decomposition(metrics, outdir):
    """Creates stress_analysis.png with component breakdown."""
    
    # Get Time and Pressures
    p_LV = get_arr(metrics, ["p_LV"])
    p_RV = get_arr(metrics, ["p_RV"])
    if p_LV is None: return
    time = get_arr(metrics, ["time"], len(p_LV))

    regions = ["LV", "Septum", "RV"]
    # Decluttered: Fiber (Directional) and Total Magnitude (Scalar)
    directions = ["ff", "mag"] 
    titles = ["Fiber Direction", "Total Magnitude"]
    
    fig = plt.figure(figsize=(12, 12)) # Thinner since fewer columns
    gs = gridspec.GridSpec(3, 2, figure=fig)
    fig.suptitle("Cauchy Stress Decomposition (Current Config)", fontsize=18, fontweight='bold')

    for r_idx, region in enumerate(regions):
        # Determine Ref Pressure for overlay
        # LV/Sep -> LV Pressure, RV -> RV Pressure
        p_ref = p_LV if region in ["LV", "Septum"] else p_RV
        p_color = 'gray'
        
        for d_idx, direction in enumerate(directions):
            ax = fig.add_subplot(gs[r_idx, d_idx])
            
            # Get Data
            sigma_tot = get_arr(metrics, [f"mean_sigma_{direction}_{region}"], len(time))
            sigma_act = get_arr(metrics, [f"mean_sigma_{direction}_active_{region}"], len(time))
            sigma_pas = get_arr(metrics, [f"mean_sigma_{direction}_passive_{region}"], len(time))
            sigma_cmp = get_arr(metrics, [f"mean_sigma_{direction}_comp_{region}"], len(time))
            
            # Combine Passive + Comp for total mechanics
            if sigma_pas is not None and sigma_cmp is not None:
                # Note: sigma_tot essentially equals Act + Pas + Comp
                pass # We largely use Tot and Act for the plot logic below

            # Convert to kPa
            if sigma_tot is not None: sigma_tot *= 1e-3
            if sigma_act is not None: sigma_act *= 1e-3
            
            # PLOT STRESS (Simplified Shading Logic)
            if sigma_tot is not None and sigma_act is not None:
                # 1. Active Stress (Input Potential)
                ax.plot(time, sigma_act, color='orange', linestyle='--', linewidth=1.5, label='Active (Input)')
                
                # 2. Total Stress (Actual Output)
                ax.plot(time, sigma_tot, 'k-', linewidth=2.5, label='Total (Output)')
                
                # 3. "Useful Stress" = Area under Total
                ax.fill_between(time, 0, sigma_tot, color='blue', alpha=0.1, label='Useful Stress')
                
                # 4. "Elastic Loss" = Difference between Active and Total
                # Only shade where Active > Total (typical shortening loss)
                ax.fill_between(time, sigma_tot, sigma_act, 
                                where=(sigma_act > sigma_tot),
                                color='orange', alpha=0.2, hatch='///', label='Elastic Loss')

            ax.set_ylabel("Stress (kPa)")
            if r_idx == 2: ax.set_xlabel("Time (s)")
            
            if r_idx == 0: ax.set_title(f"{titles[d_idx]} Direction", fontweight='bold')
            if d_idx == 0: ax.text(-0.25, 0.5, region, transform=ax.transAxes, fontsize=14, fontweight='bold', va='center', rotation=90)
            
            # Legends on specific plots only to avoid clutter
            if r_idx == 0 and d_idx == 0:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

            # PRESSURE OVERLAY (Right Axis)
            ax2 = ax.twinx()
            ax2.plot(time, p_ref, color=p_color, linestyle=':', linewidth=1.5, alpha=0.7)
            ax2.set_ylabel("Pressure (mmHg)", color=p_color)
            ax2.tick_params(axis='y', labelcolor=p_color)
            ax2.grid(False) # Turn off grid for second axis

            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = outdir / "stress_analysis.png"
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
    plot_full_hemodynamics(metrics, res_dir)
    plot_stress_decomposition(metrics, res_dir)
