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
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from plot_utils import load_metrics

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
    
    # Setup Data — prefer FEM cavity volumes (divergence theorem) over 0D-scaled volumes
    p_LV = get_arr(metrics, ["p_LV"])
    v_LV = get_arr(metrics, ["V_LV_FEM", "V_LV"])
    p_RV = get_arr(metrics, ["p_RV"])
    v_RV = get_arr(metrics, ["V_RV_FEM", "V_RV"])
    
    # Determine safe length
    if p_LV is None: return
    N = len(p_LV)
    
    # Strains for pressure-strain (prefer longitudinal E_ll, fallback E_nn, then E_ff)
    if "mean_E_ll_LV" in metrics:
        ps_e_LV = get_arr(metrics, ["mean_E_ll_LV"], N)
        ps_e_Sep = get_arr(metrics, ["mean_E_ll_Septum"], N)
        ps_e_RV = get_arr(metrics, ["mean_E_ll_RV"], N)
        ps_strain_label = "Longitudinal Strain E_ll (%)"
        ps_strain_tag = "(E_ll, GLS analogue)"
    elif "mean_E_nn_LV" in metrics:
        ps_e_LV = get_arr(metrics, ["mean_E_nn_LV"], N)
        ps_e_Sep = get_arr(metrics, ["mean_E_nn_Septum"], N)
        ps_e_RV = get_arr(metrics, ["mean_E_nn_RV"], N)
        ps_strain_label = "Sheet-Normal Strain E_nn (%)"
        ps_strain_tag = "(E_nn, NOT true longitudinal)"
    else:
        ps_e_LV = get_arr(metrics, ["mean_E_ff_LV"], N)
        ps_e_Sep = get_arr(metrics, ["mean_E_ff_Septum"], N)
        ps_e_RV = get_arr(metrics, ["mean_E_ff_RV"], N)
        ps_strain_label = "Fiber Strain E_ff (%)"
        ps_strain_tag = "(E_ff, fiber)"

    # Strains for stress-strain (always fiber E_ff)
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
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
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
    plot_cycle(ax_ps_lv, ps_e_LV, p_LV, 'tab:blue', f"LV Pressure-Strain {ps_strain_tag}", ps_strain_label, "P_LV (mmHg)")
    plot_cycle(ax_ps_sep, ps_e_Sep, p_LV, 'tab:green', f"Septal P-S {ps_strain_tag} (vs LVP)", ps_strain_label, "P_LV (mmHg)")
    plot_cycle(ax_ps_rv, ps_e_RV, p_RV, 'tab:red', f"RV Pressure-Strain {ps_strain_tag}", ps_strain_label, "P_RV (mmHg)")

    # --- PLOTTING ROW 3 (Stress-Strain) ---
    # Convert Pa to kPa for readability
    def to_kpa(arr): return arr * 1e-3 if arr is not None else None
    
    plot_cycle(ax_ss_lv, e_LV, to_kpa(s_LV), 'tab:blue', "LV Stress-Strain", "Strain (%)", "Stress (kPa)")
    plot_cycle(ax_ss_sep, e_Sep, to_kpa(s_Sep), 'tab:green', "Septal Stress-Strain", "Strain (%)", "Stress (kPa)")
    plot_cycle(ax_ss_rv, e_RV, to_kpa(s_RV), 'tab:red', "RV Stress-Strain", "Strain (%)", "Stress (kPa)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = outdir / "loops.png"
    plt.savefig(outpath, dpi=150)
    print(f" Saved: {outpath}")
    plt.close()

def plot_engineering_debug(metrics, outdir):
    """
    Revised Energy Accounting Dashboard (V2).
    Validates: 
    1. Numerical Conservation (Solver Health)
    2. Geometric Consistency (Tag Health)
    3. Physics Sanity (Active vs Passive behavior)
    """
    
    # --- 1. GATHER DATA ---
    # Helper to get data safely
    N = len(metrics["time"])
    def get_safe(keys):
        # Logic to find key in dictionary
        for k in keys:
            if k in metrics:
                # If it's a list, convert to numpy
                return np.array(metrics[k])
        return np.zeros(N)

    time = np.array(metrics["time"])

    # --- A. The Sinks (External Outputs) ---
    # 1. PV Loop Work — pure 0D clinical proxy: P_0D * dV_0D
    #    Recompute from saved arrays so it works with old and new metrics files.
    MMHG_TO_PA = 133.322
    ML_TO_M3 = 1e-6
    p_lv_0d = get_safe(["p_LV_0D", "p_LV"]) * MMHG_TO_PA
    p_rv_0d = get_safe(["p_RV_0D", "p_RV"]) * MMHG_TO_PA
    v_lv_0d = get_safe(["V_LV_Clinical", "V_LV"]) * ML_TO_M3
    v_rv_0d = get_safe(["V_RV_Clinical", "V_RV"]) * ML_TO_M3
    dV_lv = np.diff(v_lv_0d, prepend=v_lv_0d[0])
    dV_rv = np.diff(v_rv_0d, prepend=v_rv_0d[0])
    p_lv_avg = 0.5 * (p_lv_0d + np.roll(p_lv_0d, 1)); p_lv_avg[0] = p_lv_0d[0]
    p_rv_avg = 0.5 * (p_rv_0d + np.roll(p_rv_0d, 1)); p_rv_avg[0] = p_rv_0d[0]
    w_pv_proxy = p_lv_avg * dV_lv + p_rv_avg * dV_rv
    
    # 2. Robin Springs (The Elastic Support)
    w_robin = get_safe(["work_robin_epi"]) + get_safe(["work_robin_base"])
    
    # 3. Exact Boundary Work (for validation only)
    w_boundary_exact = get_safe(["work_boundary_exact_LV"]) + get_safe(["work_boundary_exact_RV"])

    # --- B. The Sources (Internal Mechanics) ---
    # We break the "Total" work into its physical components
    
    # Active Work (The Engine)
    w_active = get_safe(["work_active_Whole"]) 
    if np.sum(np.abs(w_active)) == 0: # Fallback if 'Whole' is missing
        w_active = get_safe(["work_active_LV"]) + get_safe(["work_active_RV"]) + get_safe(["work_active_Septum"])

    # Passive Work (The Elastic Storage)
    w_passive = get_safe(["work_passive_Whole"])
    if np.sum(np.abs(w_passive)) == 0:
        w_passive = get_safe(["work_passive_LV"]) + get_safe(["work_passive_RV"]) + get_safe(["work_passive_Septum"])

    # Compressible Work (The Error - Should be zero for incompressible)
    w_comp = get_safe(["work_comp_Whole"])
    if np.sum(np.abs(w_comp)) == 0:
        w_comp = get_safe(["work_comp_LV"]) + get_safe(["work_comp_RV"]) + get_safe(["work_comp_Septum"])

    # Total Internal (Sum of parts)
    w_int_total = w_active + w_passive + w_comp

    # Fix: Ensure all arrays match the minimum length (e.g. if time is longer than data)
    arrays_to_sync = [w_int_total, w_pv_proxy, w_robin, w_boundary_exact, w_active, w_passive, w_comp]
    min_len = len(time)
    for arr in arrays_to_sync:
        if len(arr) < min_len:
            min_len = len(arr)
    
    if min_len < len(time) or any(len(arr) > min_len for arr in arrays_to_sync):
        print(f"⚠️  Alignment: Truncating arrays to {min_len} steps")
        time = time[:min_len]
        w_int_total = w_int_total[:min_len]
        w_pv_proxy = w_pv_proxy[:min_len]
        w_robin = w_robin[:min_len]
        w_boundary_exact = w_boundary_exact[:min_len]
        w_active = w_active[:min_len]
        w_passive = w_passive[:min_len]
        w_comp = w_comp[:min_len]


    # --- 2. FIGURE SETUP ---
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig) # 2x3 Grid
    fig.suptitle("Engineering Debug: Physics & Numerics", fontsize=18, fontweight='bold')

    # ==========================================================
    # ROW 1: NUMERICAL VALIDATION (Is the math correct?)
    # ==========================================================

    # PLOT 1: Conservation of Energy (Total In vs Total Out)
    ax1 = fig.add_subplot(gs[0, 0])
    # Note: Internal work is usually negative (generation), External is negative (outflow).
    # We plot magnitude or consistent signs to compare.
    # Here assuming standard sign convention where Internal = External.
    
    ax1.plot(time, np.cumsum(w_int_total), 'k-', lw=3, label='Total Internal (Act+Pas)')
    ax1.plot(time, np.cumsum(w_boundary_exact + w_robin), 'r--', lw=2.5, label='Total External (Bnd+Spr)')
    
    ax1.set_title("1. Solver Health (Conservation)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Cumulative Energy (J)")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.05, "Must Overlap Perfectly", transform=ax1.transAxes, fontsize=8, color='red')

    # PLOT 2: Proxy vs Exact Boundary Work
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, np.cumsum(w_pv_proxy), 'b-', label=r'Clinical PV ($P_{0D} \cdot dV_{0D}$)')
    ax2.plot(time, np.cumsum(w_boundary_exact), 'g--', label=r'Exact Boundary ($P_{solv} \cdot dV_{FEM}$)')
    ax2.plot(time, np.cumsum(w_int_total), 'k:', lw=1, alpha=0.5, label=r'Internal ($\mathbf{S}:\mathrm{d}\mathbf{E}$)')
    ax2.set_title("2. Clinical PV vs Exact Boundary Work", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # PLOT 3: Incompressibility Check
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time, np.cumsum(w_comp), 'm-', lw=2)
    ax3.set_title("3. Incompressibility Error", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Compressible Work (J)")
    ax3.text(0.5, 0.9, "Should be ~Zero", transform=ax3.transAxes, ha='center', color='gray')
    ax3.grid(True, alpha=0.3)

    # ==========================================================
    # ROW 2: PHYSICS VALIDATION (Is it a heart?)
    # ==========================================================

    # PLOT 4: The Components of Internal Work
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Active Work (Should be Monotonic/Large)
    ax4.plot(time, np.cumsum(w_active), 'r-', lw=2, label='Active (Engine)')
    
    # Passive Work (Should Oscillate)
    ax4.plot(time, np.cumsum(w_passive), 'b-', lw=2, label='Passive (Spring)')
    
    ax4.set_title("4. Internal Source Split", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Cumulative Energy (J)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Check Passive Return
    passive_residual = np.cumsum(w_passive)[-1]
    ax4.text(0.95, 0.05, f"Net Passive: {passive_residual:.2e} J\n(Target: 0.0)", 
             transform=ax4.transAxes, ha='right', bbox=dict(facecolor='white', alpha=0.8))

    # PLOT 5: Full Physics Balance
    # W_active + W_passive + W_comp = W_PV + W_robin  (exact identity)
    ax5 = fig.add_subplot(gs[1, 1:]) # Spans 2 cols

    E_active = np.cumsum(w_active)
    E_passive = np.cumsum(w_passive)
    E_comp = np.cumsum(w_comp)
    E_boundary = np.cumsum(w_boundary_exact)
    E_Robin = np.cumsum(w_robin)

    # LHS: Active + Passive + Compressible (= Total Internal)
    E_LHS = E_active + E_passive + E_comp
    # RHS: Exact Boundary + Robin (= Total External)
    E_RHS = E_boundary + E_Robin

    ax5.plot(time, E_LHS, 'r-', lw=3, label='Internal: Active + Passive + Comp')
    ax5.plot(time, E_RHS, 'k--', lw=2, label='External: Boundary + Robin')
    ax5.plot(time, E_active, 'r:', lw=1, alpha=0.5, label='Active Only')
    ax5.plot(time, E_boundary, 'b:', lw=1, alpha=0.5, label='Boundary Only')

    ax5.set_title("5. Physics Balance: W_Internal = W_Boundary + W_Robin", fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    balance_err = E_LHS[-1] - E_RHS[-1]
    ax5.text(0.95, 0.05, f"Balance error: {balance_err:.2e} J",
             transform=ax5.transAxes, ha='right', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(outdir / "engineering_debug.png")
    print("Saved V2 debug dashboard.")
    plt.close()


def plot_full_hemodynamics(metrics, outdir):
    """Creates the Clinical Hemodynamics dashboard (Reconstructed Volumes)."""
    
    # 1. Retrieve Data — use 0D model pressures with 0D volumes for consistency
    v_LV_clin = get_arr(metrics, ["V_LV_Clinical"])
    v_RV_clin = get_arr(metrics, ["V_RV_Clinical"])
    p_LV = get_arr(metrics, ["p_LV_0D", "p_LV"])  # Prefer 0D pressures, fallback to solver
    p_RV = get_arr(metrics, ["p_RV_0D", "p_RV"])
    
    # Get Time (safely)
    if p_LV is not None:
        time = get_arr(metrics, ["time"], len(p_LV))
    else:
        time = None

    # Check existence
    if v_LV_clin is None or v_RV_clin is None:
        print(" Missing 'V_LV_Clinical' or 'V_RV_Clinical'. Skipping full hemodynamics plot.")
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
    print(f"Saved: {outpath}")
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
            if sigma_pas is not None: sigma_pas *= 1e-3
            if sigma_cmp is not None: sigma_cmp *= 1e-3
            
            # PLOT STRESS (Simplified Shading Logic)
            if sigma_tot is not None and sigma_act is not None:
                # 1. Active Stress (Input Potential)
                ax.plot(time, sigma_act, color='orange', linestyle='--', linewidth=1.5, label='Active (Input)')
                
                # 2. Total Stress (Actual Output)
                ax.plot(time, sigma_tot, 'k-', linewidth=2.5, label='Total (Output)')

                # 3. Hydrostatic/Bulk Stress (New for Incompressible Debug)
                if sigma_cmp is not None:
                    # In Incompressible: sigma_cmp = -pI. 
                    # If this is large negative, it means high hydrostatic pressure support.
                    ax.plot(time, sigma_cmp, color='cyan', linestyle='-.', linewidth=1, alpha=0.8, label='Hydrostatic (-pI)')
                
                # 4. "Useful Stress" = Area under Total
                ax.fill_between(time, 0, sigma_tot, color='blue', alpha=0.1, label='Useful Stress')
                
                # 5. "Elastic Loss" = Difference between Active and Total
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
    print(f" Saved: {outpath}")
    plt.close()

def save_hemodynamics_json(metrics, outdir):
    data = {}
    
    def get_max(key):
        arr = get_arr(metrics, [key])
        return float(np.max(arr)) if arr is not None else None
    
    data["Pressures"] = {
        "LV_Peak": get_max("p_LV"),
        "RV_Peak": get_max("p_RV"),
        "LA_Mean": float(np.mean(get_arr(metrics, ["p_LA"]))) if get_arr(metrics, ["p_LA"]) is not None else None
    }
    
    # Volumes (Try Clinical first, then raw)
    v_lv = get_arr(metrics, ["V_LV_Clinical"])
    if v_lv is None: v_lv = get_arr(metrics, ["V_LV"])
    
    v_rv = get_arr(metrics, ["V_RV_Clinical"])
    if v_rv is None: v_rv = get_arr(metrics, ["V_RV"])
    
    if v_lv is not None:
        edv_lv = float(np.max(v_lv))
        esv_lv = float(np.min(v_lv))
        sv_lv = edv_lv - esv_lv
        ef_lv = (sv_lv / edv_lv * 100) if edv_lv > 0 else 0
        data["LV"] = {"EDV_mL": edv_lv, "ESV_mL": esv_lv, "SV_mL": sv_lv, "EF_pct": ef_lv}
    
    if v_rv is not None:
        edv_rv = float(np.max(v_rv))
        esv_rv = float(np.min(v_rv))
        sv_rv = edv_rv - esv_rv
        ef_rv = (sv_rv / edv_rv * 100) if edv_rv > 0 else 0
        data["RV"] = {"EDV_mL": edv_rv, "ESV_mL": esv_rv, "SV_mL": sv_rv, "EF_pct": ef_rv}
        
    outpath = Path(outdir) / "hemodynamics.json"
    with open(outpath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ Saved stats: {outpath.name}")

def save_detailed_stats(metrics, outdir):
    """Calculates hard engineering and mechanical statistics and saves to JSON."""
    stats = {}
    
    # --- 1. ENGINEERING DEBUG STATS ---
    # Recreating logic from plot_engineering_debug to ensure identical numbers
    N = len(metrics["time"])
    def get_safe(keys):
        for k in keys:
            if k in metrics: return np.array(metrics[k])
        return np.zeros(N)

    time = np.array(metrics["time"])
    
    # Sinks
    w_pv_proxy = get_safe(["work_proxy_pv_LV"]) + get_safe(["work_proxy_pv_RV"])
    w_robin = get_safe(["work_robin_epi"]) + get_safe(["work_robin_base"])
    w_boundary_exact = get_safe(["work_boundary_exact_LV"]) + get_safe(["work_boundary_exact_RV"])

    # Sources
    w_active = get_safe(["work_active_Whole"]) 
    if np.sum(np.abs(w_active)) == 0:
        w_active = get_safe(["work_active_LV"]) + get_safe(["work_active_RV"]) + get_safe(["work_active_Septum"])

    w_passive = get_safe(["work_passive_Whole"])
    if np.sum(np.abs(w_passive)) == 0:
        w_passive = get_safe(["work_passive_LV"]) + get_safe(["work_passive_RV"]) + get_safe(["work_passive_Septum"])

    w_comp = get_safe(["work_comp_Whole"])
    if np.sum(np.abs(w_comp)) == 0:
        w_comp = get_safe(["work_comp_LV"]) + get_safe(["work_comp_RV"]) + get_safe(["work_comp_Septum"])

    w_int_total = w_active + w_passive + w_comp

    # Sync lengths
    arrays_map = {
        "w_int_total": w_int_total, "w_pv_proxy": w_pv_proxy, "w_robin": w_robin, 
        "w_boundary_exact": w_boundary_exact, "w_active": w_active, 
        "w_passive": w_passive, "w_comp": w_comp
    }
    
    min_len = len(time)
    for k, arr in arrays_map.items():
        if len(arr) < min_len: min_len = len(arr)
        
    # Truncate if needed
    if min_len < len(time):
        for k in arrays_map:
            arrays_map[k] = arrays_map[k][:min_len]

    # Calculate Integrals (Joules) - final cumulative sum
    def total(arr): return float(np.sum(arr)) # Assuming arrays are effectively dW. cumsum[-1] == sum

    E_int_total = total(arrays_map["w_int_total"])
    E_ext_total = total(arrays_map["w_boundary_exact"] + arrays_map["w_robin"])
    E_pv = total(arrays_map["w_pv_proxy"])
    E_boundary = total(arrays_map["w_boundary_exact"])
    E_active = total(arrays_map["w_active"])
    E_passive = total(arrays_map["w_passive"])
    E_comp = total(arrays_map["w_comp"])
    E_robin = total(arrays_map["w_robin"])

    stats["Engineering_Integrals"] = {
        "Solver_Conservation_Error_J": E_int_total - E_ext_total,
        "Solver_Conservation_Error_Pct": (E_int_total - E_ext_total)/E_int_total * 100 if abs(E_int_total) > 1e-6 else 0,
        "Geometric_Consistency_Error_J": E_pv - E_boundary,
        "Incompressibility_Error_J": E_comp,
        "Net_Passive_Work_J": E_passive,
        "Physics_Balance_Error_J": (E_active + E_passive + E_comp) - (E_pv + E_robin) # Internal - External
    }

    # --- 2. STRESS ANALYSIS STATS ---
    stats["Stress_Analysis_kPa"] = {}
    regions = ["LV", "Septum", "RV"]
    directions = ["ff", "mag"]
    
    for region in regions:
        r_stats = {}
        for direction in directions:
            # Keys like "mean_sigma_ff_LV"
            k_base = f"mean_sigma_{direction}"
            
            # Fetch arrays (converted to kPa)
            def get_stress_kpa(suffix):
                key = f"{k_base}_{suffix}_{region}"
                # Handle base case (total) which has no suffix in key pattern used in plot
                if suffix == "total": key = f"{k_base}_{region}"
                
                arr = get_arr(metrics, [key], min_len)
                return arr * 1e-3 if arr is not None else None

            s_tot = get_stress_kpa("total")
            s_act = get_stress_kpa("active")
            s_pas = get_stress_kpa("passive")
            
            d_stats = {}
            if s_tot is not None:
                d_stats["Max_Total"] = float(np.max(s_tot))
                d_stats["Mean_Total"] = float(np.mean(s_tot))
            if s_act is not None:
                d_stats["Max_Active"] = float(np.max(s_act))
                d_stats["Mean_Active"] = float(np.mean(s_act))
            if s_pas is not None:
                d_stats["Max_Passive"] = float(np.max(s_pas))
                d_stats["Mean_Passive"] = float(np.mean(s_pas))
            
            if s_act is not None and s_tot is not None:
                diff = s_act - s_tot
                loss = np.where(diff > 0, diff, 0)
                d_stats["Mean_Elastic_Loss"] = float(np.mean(loss))

            r_stats[direction] = d_stats
        stats["Stress_Analysis_kPa"][region] = r_stats

    # Save
    outpath = Path(outdir) / "detailed_stats.json"
    with open(outpath, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"✅ Saved detailed stats: {outpath.name}")

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
    save_hemodynamics_json(metrics, res_dir)
    save_detailed_stats(metrics, res_dir)
