#!/usr/bin/env python3
"""
Comprehensive Post-Processing & Analysis Script

This unified script handles ALL analysis tasks:
1. Load and validate simulation data (metrics + hemodynamics)
2. Print diagnostic summary (pressure/volume ranges, work magnitudes)
3. Generate hemodynamic plots (PV loops, time series)
4. Generate work comparison plots (True vs Proxy vs Boundary)
5. Phase-windowed correlation analysis
6. Boundary work validation
7. Statistics export (JSON)

Replaces: postprocess.py, diagnose_work.py, analyze_metrics.py (legacy)

Usage:
  python3 analyze_metrics.py <results_dir> [downsample_factor]

Example:
  python3 analyze_metrics.py results/sims/run_943768 1
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

# Unit conversion: 1 mmHg*mL = 1.33322e-4 Joules
MMHG_ML_TO_J = 1.33322e-4


def print_diagnostics(metrics, results_dir):
    """
    Print comprehensive diagnostics summary (from diagnose_work.py).
    Shows hemodynamics, activation, work magnitudes, and correlations.
    """
    print("\n" + "="*80)
    print(f"DIAGNOSTIC SUMMARY: {Path(results_dir).name}")
    print("="*80)
    
    # --- 1. Hemodynamics ---
    print("\n### HEMODYNAMICS ###")
    V_LV = np.array(metrics.get("V_LV", []))
    V_RV = np.array(metrics.get("V_RV", []))
    p_LV = np.array(metrics.get("p_LV", []))
    p_RV = np.array(metrics.get("p_RV", []))
    
    if len(V_LV) > 0:
        print(f"LV Volume:   {V_LV.min():>6.1f} - {V_LV.max():<6.1f} mL  (Δ{V_LV.max()-V_LV.min():.1f})")
        print(f"RV Volume:   {V_RV.min():>6.1f} - {V_RV.max():<6.1f} mL  (Δ{V_RV.max()-V_RV.min():.1f})")
        print(f"LV Pressure: {p_LV.min():>6.1f} - {p_LV.max():<6.1f} mmHg (Δ{p_LV.max()-p_LV.min():.1f})")
        print(f"RV Pressure: {p_RV.min():>6.1f} - {p_RV.max():<6.1f} mmHg (Δ{p_RV.max()-p_RV.min():.1f})")
        
        # Warnings
        if V_LV.max() - V_LV.min() < 10:
            print("⚠ WARNING: LV volume swing < 10 mL")
        if p_LV.max() - p_LV.min() < 50:
            print("⚠ WARNING: LV pressure swing < 50 mmHg")
    
    # --- 2. Work Magnitudes ---
    print("\n### WORK MAGNITUDES ###")
    work_true_LV = np.array(metrics.get("work_true_LV", []))
    work_proxy_LV = np.array(metrics.get("work_proxy_pv_LV", []))
    work_true_RV = np.array(metrics.get("work_true_RV", []))
    work_proxy_RV = np.array(metrics.get("work_proxy_pv_RV", []))
    
    if len(work_true_LV) > 0:
        # Convert proxy to Joules
        work_proxy_LV_J = work_proxy_LV * MMHG_ML_TO_J
        work_proxy_RV_J = work_proxy_RV * MMHG_ML_TO_J
        
        cumul_true_LV = np.cumsum(work_true_LV)[-1] if len(work_true_LV) > 0 else 0
        cumul_proxy_LV = np.cumsum(work_proxy_LV_J)[-1] if len(work_proxy_LV_J) > 0 else 0
        cumul_true_RV = np.cumsum(work_true_RV)[-1] if len(work_true_RV) > 0 else 0
        cumul_proxy_RV = np.cumsum(work_proxy_RV_J)[-1] if len(work_proxy_RV_J) > 0 else 0
        
        print(f"LV True Work:   {cumul_true_LV:>10.4e} J  (Range: {work_true_LV.min():.2e} to {work_true_LV.max():.2e})")
        print(f"LV Proxy Work:  {cumul_proxy_LV:>10.4e} J  (Range: {work_proxy_LV_J.min():.2e} to {work_proxy_LV_J.max():.2e})")
        print(f"RV True Work:   {cumul_true_RV:>10.4e} J")
        print(f"RV Proxy Work:  {cumul_proxy_RV:>10.4e} J")
        
        # Ratio
        ratio_LV = cumul_true_LV / cumul_proxy_LV if cumul_proxy_LV != 0 else 0
        ratio_RV = cumul_true_RV / cumul_proxy_RV if cumul_proxy_RV != 0 else 0
        print(f"\nLV True/Proxy Ratio: {ratio_LV:.3f}")
        print(f"RV True/Proxy Ratio: {ratio_RV:.3f}")
        
        # Interpretation
        if 0.5 < ratio_LV < 2.0:
            print("✓ LV magnitudes similar (good!)")
        elif ratio_LV < 0.1:
            print("❌ True Work << Proxy (stress integration issue?)")
        elif ratio_LV > 10:
            print("❌ True Work >> Proxy (unit conversion error?)")
        
        # Correlation
        if len(work_true_LV) > 2:
            corr_LV = np.corrcoef(work_true_LV[1:], work_proxy_LV_J[1:])[0, 1]
            corr_RV = np.corrcoef(work_true_RV[1:], work_proxy_RV_J[1:])[0, 1]
            print(f"\n📊 LV Correlation: {corr_LV:.3f}")
            print(f"📊 RV Correlation: {corr_RV:.3f}")
            
            if corr_LV > 0.7:
                print("✓ Strong correlation (excellent!)")
            elif corr_LV > 0.4:
                print("⚠ Moderate correlation")
            else:
                print("❌ Weak correlation (physics mismatch?)")
    
    print("\n" + "="*80)


def plot_hemodynamics(metrics, output_dir):
    """
    Create hemodynamic plots (from postprocess.py):
    1. PV loop with metrics overlay
    2. Time series (pressure + volume for all chambers)
    3. GridSpec complete cycle visualization
    """
    times = np.array(metrics.get("time", []))
    p_LV = np.array(metrics.get("p_LV", []))
    p_RV = np.array(metrics.get("p_RV", []))
    p_LA = np.array(metrics.get("p_LA", []))
    p_RA = np.array(metrics.get("p_RA", []))
    V_LV = np.array(metrics.get("V_LV", []))
    V_RV = np.array(metrics.get("V_RV", []))
    
    if len(V_LV) == 0:
        print("⚠ No hemodynamic data for plotting")
        return
    
    # Align arrays
    min_len = min(len(times), len(p_LV), len(V_LV))
    times = times[-min_len:]
    p_LV = p_LV[-min_len:]
    p_RV = p_RV[-min_len:]
    V_LV = V_LV[-min_len:]
    V_RV = V_RV[-min_len:]
    p_LA = p_LA[-min_len:] if len(p_LA) >= min_len else p_LA
    p_RA = p_RA[-min_len:] if len(p_RA) >= min_len else p_RA
    
    # --- 1. PV Loop with Metrics ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(V_LV, p_LV, 'b-', linewidth=2.5, label='LV PV Loop')
    ax1.scatter(V_LV[0], p_LV[0], color='green', s=100, zorder=5, label='ED')
    ax1.scatter(V_LV[-1], p_LV[-1], color='red', s=100, zorder=5, label='ES')
    
    # Direction arrows
    for i in range(0, len(V_LV)-1, max(1, len(V_LV)//10)):
        ax1.arrow(V_LV[i], p_LV[i], V_LV[i+1]-V_LV[i], p_LV[i+1]-p_LV[i],
                 head_width=2, head_length=3, fc='blue', ec='blue', alpha=0.5)
    
    ax1.set_xlabel('LV Volume (mL)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('LV Pressure (mmHg)', fontsize=12, fontweight='bold')
    ax1.set_title('Left Ventricular PV Loop', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Metrics panel
    lv_ef = (V_LV.max() - V_LV.min()) / V_LV.max() * 100
    sv = V_LV.max() - V_LV.min()
    dp_dt = np.gradient(p_LV)
    max_dp_dt = np.max(np.abs(dp_dt))
    
    ax2.axis('off')
    metrics_text = f"""
HEMODYNAMIC PARAMETERS

Left Ventricle:
  • Peak Systolic Pressure: {p_LV.max():.1f} mmHg
  • End Diastolic Pressure: {p_LV.min():.1f} mmHg
  • End Systolic Volume:    {V_LV.min():.1f} mL
  • End Diastolic Volume:   {V_LV.max():.1f} mL
  • Stroke Volume:          {sv:.1f} mL
  • Ejection Fraction:      {lv_ef:.1f}%
  • dP/dt max:              {max_dp_dt:.1f} mmHg/s

Right Ventricle:
  • Peak Systolic Pressure: {p_RV.max():.1f} mmHg
  • End Diastolic Pressure: {p_RV.min():.1f} mmHg
  • End Systolic Volume:    {V_RV.min():.1f} mL
  • End Diastolic Volume:   {V_RV.max():.1f} mL

Simulation: ✓ SUCCESS
  • Time Points: {len(times)}
"""
    
    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / "pv_loop_analysis.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: pv_loop_analysis.png")
    plt.close()
    
    # --- 2. Time Series Grid ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].plot(times, p_LV, 'b-', linewidth=2)
    axes[0,0].set_ylabel('LV Pressure (mmHg)', fontweight='bold')
    axes[0,0].set_title('Left Ventricle Pressure', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(times, p_RV, 'r-', linewidth=2)
    axes[0,1].set_ylabel('RV Pressure (mmHg)', fontweight='bold')
    axes[0,1].set_title('Right Ventricle Pressure', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(times, V_LV, 'b-', linewidth=2, label='LV')
    axes[1,0].plot(times, V_RV, 'r-', linewidth=2, label='RV')
    axes[1,0].set_xlabel('Time (s)', fontweight='bold')
    axes[1,0].set_ylabel('Volume (mL)', fontweight='bold')
    axes[1,0].set_title('Ventricular Volumes', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    if len(p_LA) > 0:
        axes[1,1].plot(times, p_LA, 'b--', linewidth=2, label='LA')
        axes[1,1].plot(times, p_RA, 'r--', linewidth=2, label='RA')
        axes[1,1].set_xlabel('Time (s)', fontweight='bold')
        axes[1,1].set_ylabel('Atrial Pressure (mmHg)', fontweight='bold')
        axes[1,1].set_title('Atrial Pressures', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "hemodynamics_timeseries.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: hemodynamics_timeseries.png")
    plt.close()
    
    # --- 3. Complete Cycle GridSpec Visualization ---
    try:
        fig = plt.figure(layout="constrained", figsize=(14, 8))
        gs = GridSpec(3, 4, figure=fig)
        
        # Left: LV PV Loop
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.plot(V_LV, p_LV, 'b-', linewidth=2)
        ax1.set_xlabel("LVV [mL]", fontweight='bold')
        ax1.set_ylabel("LVP [mmHg]", fontweight='bold')
        ax1.set_title("LV PV Loop", fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Second column: RV PV Loop
        ax2 = fig.add_subplot(gs[:, 1])
        ax2.plot(V_RV, p_RV, 'r-', linewidth=2)
        ax2.set_xlabel("RVV [mL]", fontweight='bold')
        ax2.set_ylabel("RVP [mmHg]", fontweight='bold')
        ax2.set_title("RV PV Loop", fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Right side: Time series
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(times, p_LV, 'b-', linewidth=1.5)
        ax3.set_ylabel("LVP [mmHg]", fontweight='bold')
        ax3.set_xticklabels([])
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Pressures", fontweight='bold', fontsize=10)
        
        ax5 = fig.add_subplot(gs[0, 3])
        ax5.plot(times, p_RV, 'r-', linewidth=1.5)
        ax5.set_ylabel("RVP [mmHg]", fontweight='bold')
        ax5.set_xticklabels([])
        ax5.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(times, V_LV, 'b-', linewidth=1.5)
        ax4.set_ylabel("LVV [mL]", fontweight='bold')
        ax4.set_xticklabels([])
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Volumes", fontweight='bold', fontsize=10)
        
        ax6 = fig.add_subplot(gs[1, 3])
        ax6.plot(times, V_RV, 'r-', linewidth=1.5)
        ax6.set_ylabel("RVV [mL]", fontweight='bold')
        ax6.set_xticklabels([])
        ax6.grid(True, alpha=0.3)
        
        # Active tension (if available)
        Ta = np.array(metrics.get("Ta", []))
        if len(Ta) > 0:
            if Ta.ndim > 1:
                Ta = Ta[:, 0]
            Ta = Ta[-min_len:] if len(Ta) >= min_len else Ta
            
            ax7 = fig.add_subplot(gs[2, 2:])
            ax7.plot(times[:len(Ta)], Ta, 'purple', linewidth=1.5)
            ax7.set_ylabel("Ta (LV) [kPa]", fontweight='bold')
            ax7.set_xlabel("Time [s]", fontweight='bold')
            ax7.grid(True, alpha=0.3)
            ax7.set_title("Active Tension", fontweight='bold', fontsize=10)
        
        plt.savefig(output_dir / "pv_loop_complete_cycle.png", dpi=150, bbox_inches='tight')
        print(f"✓ Saved: pv_loop_complete_cycle.png")
        plt.close()
    except Exception as e:
        print(f"⚠ Complete cycle visualization skipped: {e}")


def load_metrics(results_dir, downsample_factor=1):
    """Load metrics from saved file."""
    results_dir = Path(results_dir)
    metrics_file = results_dir / f"metrics_downsample_{downsample_factor}.npy"

    if not metrics_file.exists():
        print(f"ERROR: {metrics_file} not found")
        return None

    metrics = np.load(metrics_file, allow_pickle=True).item()

    # Also try to load history.npy for raw pressure/volume data
    history_file = results_dir / "history.npy"
    if history_file.exists():
        try:
            print(f"Loading history from {history_file}...")
            history = np.load(history_file, allow_pickle=True).item()

            # Use metrics time length to estimate target size
            target_len = len(metrics["time"])

            # Merge history into metrics
            for key, val in history.items():
                if key not in metrics:
                    metrics[key] = val
        except Exception as e:
            print(f"Warning: Could not load history.npy: {e}")

    return metrics


def extract_regional_data(metrics, regions=["LV", "RV", "Septum"]):
    """Extract work data for specified regions."""
    data = {}

    # Extract raw history needed for re-calculations
    full_times = np.array(metrics["time"])
    p_LV = np.array(metrics.get("p_LV", []))
    p_RV = np.array(metrics.get("p_RV", []))
    V_LV = np.array(metrics.get("V_LV", []))
    V_RV = np.array(metrics.get("V_RV", []))

    # Ensure all arrays are aligned in length
    min_len_history = min(len(p_LV), len(p_RV), len(V_LV), len(V_RV), len(full_times))

    # Recalculate dV (backward difference)
    dV_LV = np.zeros(min_len_history)
    dV_RV = np.zeros(min_len_history)
    dV_LV[1:] = V_LV[1:min_len_history] - V_LV[0:min_len_history-1]
    dV_RV[1:] = V_RV[1:min_len_history] - V_RV[0:min_len_history-1]

    # Process regions
    for region in regions:
        true_work = np.array(metrics.get(f"work_true_{region}", []))
        proxy_work = np.array(metrics.get(f"work_proxy_pv_{region}", []))

        # Align time with work data
        current_times = full_times
        min_len = min(len(true_work), len(proxy_work))

        # Truncate to match work length
        if len(true_work) > min_len: true_work = true_work[-min_len:]
        if len(proxy_work) > min_len: proxy_work = proxy_work[-min_len:]

        # Convert proxy work from mmHg*mL to Joules
        if len(proxy_work) > 0:
            proxy_work = proxy_work * MMHG_ML_TO_J

        # Determine current times for this region
        region_times = current_times[-min_len:] if min_len > 0 else []

        data[region] = {
            "true_work": true_work,
            "proxy_work": proxy_work,
            "time": region_times if min_len > 0 else [],
        }

    return data


def compute_statistics(data, metrics=None):
    """Compute statistics for work comparison."""
    stats = {}

    for region, values in data.items():
        true_w = values["true_work"]
        proxy_w = values["proxy_work"]

        if len(true_w) > 0 and len(proxy_w) > 0:
            correlation = np.corrcoef(true_w, proxy_w)[0, 1] if len(true_w) > 1 else 0.0
            
            stats[region] = {
                "true_work_mean": float(np.mean(true_w)),
                "true_work_max": float(np.max(true_w)),
                "true_work_min": float(np.min(true_w)),
                "proxy_work_mean": float(np.mean(proxy_w)),
                "proxy_work_max": float(np.max(proxy_w)),
                "proxy_work_min": float(np.min(proxy_w)),
                "correlation": float(correlation),
            }

    return stats


def plot_validation_boundary_work(data, metrics, output_file=None):
    """Create validation plot: Boundary Work vs Proxy Work."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Extract necessary data
    times = np.array(metrics.get("time", []))
    boundary_lv = np.array(metrics.get("work_boundary_LV", []))
    proxy_lv = np.array(metrics.get("work_proxy_pv_LV", []))
    boundary_rv = np.array(metrics.get("work_boundary_RV", []))
    proxy_rv = np.array(metrics.get("work_proxy_pv_RV", []))

    # Convert proxy to Joules
    proxy_lv = proxy_lv * MMHG_ML_TO_J
    proxy_rv = proxy_rv * MMHG_ML_TO_J

    # Ensure all arrays are aligned
    min_len = min(len(times), len(boundary_lv), len(proxy_lv))
    if min_len > 0:
        times = times[-min_len:]
        boundary_lv = boundary_lv[-min_len:]
        proxy_lv = proxy_lv[-min_len:]
        boundary_rv = boundary_rv[-min_len:]
        proxy_rv = proxy_rv[-min_len:]

        # LV Validation
        ax = axes[0]
        ax.plot(times, proxy_lv, 'b-', linewidth=2.0, label='Proxy Work (P·ΔV)')
        ax.plot(times, boundary_lv, 'r--', linewidth=1.5, label='Boundary Work (∫p·n·Δu·dA)')
        ax.fill_between(times, proxy_lv, boundary_lv, alpha=0.2, color='gray')
        ax.set_ylabel('Work (J)', fontsize=11, fontweight='bold')
        ax.set_title('LV: Boundary Work vs Proxy Work (Validation)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # Calculate validation error
        proxy_max = np.max(np.abs(proxy_lv))
        if proxy_max > 1e-12:
            error_lv = np.mean(np.abs(boundary_lv - proxy_lv)) / proxy_max * 100
        else:
            error_lv = 0.0
        ax.text(0.02, 0.95, f'Mean Error: {error_lv:.2f}%', transform=ax.transAxes,
                verticalalignment='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # RV Validation
        ax = axes[1]
        ax.plot(times, proxy_rv, 'b-', linewidth=2.0, label='Proxy Work (P·ΔV)')
        ax.plot(times, boundary_rv, 'r--', linewidth=1.5, label='Boundary Work (∫p·n·Δu·dA)')
        ax.fill_between(times, proxy_rv, boundary_rv, alpha=0.2, color='gray')
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Work (J)', fontsize=11, fontweight='bold')
        ax.set_title('RV: Boundary Work vs Proxy Work (Validation)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # Calculate validation error
        proxy_max = np.max(np.abs(proxy_rv))
        if proxy_max > 1e-12:
            error_rv = np.mean(np.abs(boundary_rv - proxy_rv)) / proxy_max * 100
        else:
            error_rv = 0.0
        ax.text(0.02, 0.95, f'Mean Error: {error_rv:.2f}%', transform=ax.transAxes,
                verticalalignment='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {Path(output_file).name}")

        plt.close()
        return {"lv_error": error_lv, "rv_error": error_rv}
    else:
        print("⚠ Insufficient data for boundary work validation plot")
        return None


def plot_phase_windowed_analysis(data, metrics, output_file=None):
    """Create phase-windowed analysis plots: Full cycle vs Ejection phase."""
    # Extract volume data to define ejection phase
    V_LV = np.array(metrics.get("V_LV", []))
    times = np.array(metrics.get("time", []))

    if len(V_LV) < 2:
        print("⚠ Insufficient volume data for phase analysis")
        return None

    # Define ejection phase (dV < 0)
    dV_LV = np.diff(V_LV, prepend=V_LV[0])
    ejection_mask = dV_LV < -1e-6

    # Extract LV data
    lv_data = data.get("LV", None)
    if lv_data is None or len(lv_data["true_work"]) == 0:
        print("⚠ No LV data for phase analysis")
        return None

    true_w = np.array(lv_data["true_work"])
    proxy_w = np.array(lv_data["proxy_work"])

    # Align ejection mask to work arrays
    min_len = min(len(ejection_mask), len(true_w))
    if min_len > 0:
        ejection_mask = ejection_mask[-min_len:]
        true_w = true_w[-min_len:]
        proxy_w = proxy_w[-min_len:]
        times = times[-min_len:]

    # Split into ejection and non-ejection phases
    ejection_true = true_w[ejection_mask]
    ejection_proxy = proxy_w[ejection_mask]

    # Compute correlations
    global_corr = np.corrcoef(true_w, proxy_w)[0, 1] if len(true_w) > 1 else 0.0
    global_rmse = np.sqrt(np.mean((true_w - proxy_w)**2))

    ejection_corr = np.corrcoef(ejection_true, ejection_proxy)[0, 1] if len(ejection_true) > 1 else 0.0
    ejection_rmse = np.sqrt(np.mean((ejection_true - ejection_proxy)**2))

    # Create phase visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Full cycle time series with phase markers
    ax = axes[0, 0]
    ax.plot(times, true_w, 'b-', linewidth=1.5, label='True Work', alpha=0.8)
    ax.plot(times, proxy_w, 'r-', linewidth=1.5, label='Proxy Work', alpha=0.8)
    ax.fill_between(times, -0.5e-3, 0.5e-3, where=ejection_mask, alpha=0.2, color='green', label='Ejection Phase')
    ax.set_ylabel('Work (J)', fontweight='bold')
    ax.set_title('Full Cycle: True vs Proxy (Ejection Phase Highlighted)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Global correlation scatter
    ax = axes[0, 1]
    ax.scatter(proxy_w, true_w, alpha=0.6, s=30, c='gray', label='All points')
    ax.scatter(ejection_proxy, ejection_true, alpha=0.8, s=50, c='green', label='Ejection', marker='o')
    z = np.polyfit(proxy_w, true_w, 1)
    p = np.poly1d(z)
    px = np.linspace(np.min(proxy_w), np.max(proxy_w), 100)
    ax.plot(px, p(px), 'b--', linewidth=2, label=f'Global R={global_corr:.3f}')
    ax.set_xlabel('Proxy Work', fontweight='bold')
    ax.set_ylabel('True Work', fontweight='bold')
    ax.set_title('Global Correlation (All Phases)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Ejection-only correlation scatter
    ax = axes[1, 0]
    ax.scatter(ejection_proxy, ejection_true, alpha=0.8, s=50, c='green', marker='o')
    if len(ejection_proxy) > 2:
        z_ej = np.polyfit(ejection_proxy, ejection_true, 1)
        p_ej = np.poly1d(z_ej)
        px_ej = np.linspace(np.min(ejection_proxy), np.max(ejection_proxy), 100)
        ax.plot(px_ej, p_ej(px_ej), 'g--', linewidth=2, label=f'Ejection R={ejection_corr:.3f}')
    ax.set_xlabel('Proxy Work', fontweight='bold')
    ax.set_ylabel('True Work', fontweight='bold')
    ax.set_title('Ejection Phase Only Correlation', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: Statistics comparison
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
PHASE-WINDOWED CORRELATION ANALYSIS (LV)

Global (All Phases):
  • Correlation: {global_corr:>8.4f}
  • RMSE:        {global_rmse:>12.4e} J
  • N points:    {len(true_w):>8d}

Ejection Phase Only:
  • Correlation: {ejection_corr:>8.4f}
  • RMSE:        {ejection_rmse:>12.4e} J
  • N points:    {len(ejection_true):>8d}

Interpretation:
  If ejection corr >> global corr:
    ✓ Proxy captures systolic work
    ✓ Diastolic elastic energy dominates
  If both low:
    ✗ Different physics (stress vs PV)
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {Path(output_file).name}")

    plt.close()

    return {
        "global_correlation": float(global_corr),
        "global_rmse": float(global_rmse),
        "ejection_correlation": float(ejection_corr),
        "ejection_rmse": float(ejection_rmse),
        "num_ejection_points": int(np.sum(ejection_mask)),
    }


def plot_comparison(data, output_file=None):
    """Create comprehensive comparison plots."""
    regions = list(data.keys())
    ordered_regions = ["LV", "RV", "Septum"]
    other_regions = [r for r in regions if r not in ordered_regions]
    plot_regions = ordered_regions + sorted(other_regions)

    # Filter out empty data
    plot_regions = [r for r in plot_regions if r in data and len(data[r]["true_work"]) > 0]

    # Calculate grid size
    n_plots = len(plot_regions)
    if n_plots == 0:
        print("⚠ No valid data to plot. Skipping plot_comparison.")
        return None
    
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig = plt.figure(figsize=(5*cols, 4*rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

    colors = {
        "LV": "blue", "RV": "red", "Septum": "green"
    }

    for idx, region in enumerate(plot_regions):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])

        true_w = np.array(data[region]["true_work"]).ravel()
        proxy_w = np.array(data[region]["proxy_work"]).ravel()

        # Ensure lengths match
        min_p_len = min(len(true_w), len(proxy_w))
        if min_p_len > 0:
            true_w = true_w[-min_p_len:]
            proxy_w = proxy_w[-min_p_len:]

        color = colors.get(region, "black")

        # Plot Scatter
        ax.scatter(proxy_w, true_w, alpha=0.5, s=20, color=color, label="Data")

        # Calculate stats
        if len(true_w) > 1:
            corr = np.corrcoef(true_w, proxy_w)[0, 1]
            z = np.polyfit(proxy_w, true_w, 1)
            p = np.poly1d(z)
            px = np.linspace(np.min(proxy_w), np.max(proxy_w), 100)
            ax.plot(px, p(px), 'k--', linewidth=1.5, label=f"R={corr:.3f}")

        ax.set_xlabel("Proxy Work (P·V)")
        ax.set_ylabel("True Work (S·E)")
        ax.set_title(f"{region} Comparison", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {Path(output_file).name}")

    plt.close()


def print_statistics(stats):
    """Print formatted statistics."""
    print("\n" + "="*80)
    print("WORK METRICS STATISTICS")
    print("="*80)

    for region, values in stats.items():
        print(f"\n{region}:")
        print(f"  True Work:  mean={values['true_work_mean']:>12.4e}, max={values['true_work_max']:>12.4e}")
        print(f"  Proxy Work: mean={values['proxy_work_mean']:>12.4e}, max={values['proxy_work_max']:>12.4e}")
        print(f"  Correlation: {values['correlation']:>6.3f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_metrics.py <results_dir> [downsample_factor]")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    downsample_factor = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print("\n" + "="*80)
    print("COMPREHENSIVE POST-PROCESSING & ANALYSIS")
    print("="*80)
    print(f"Results: {results_dir}")
    print(f"Downsample: {downsample_factor}")

    # Load metrics
    print(f"\nLoading metrics from {results_dir}...")
    metrics = load_metrics(results_dir, downsample_factor)

    if metrics is None:
        sys.exit(1)

    # --- 1. Print Diagnostics ---
    print_diagnostics(metrics, results_dir)

    # --- 2. Hemodynamic Plots ---
    print("\n" + "="*80)
    print("HEMODYNAMIC VISUALIZATION")
    print("="*80)
    plot_hemodynamics(metrics, results_dir)

    # --- 3. Work Analysis ---
    print("\n" + "="*80)
    print("WORK ANALYSIS")
    print("="*80)
    
    # Extract regional data
    regions = ["LV", "RV", "Septum"]
    data = extract_regional_data(metrics, regions=regions)

    # Compute statistics
    stats = compute_statistics(data, metrics=metrics)
    print_statistics(stats)

    # Create comparison plots
    plot_file = results_dir / f"work_comparison_downsample_{downsample_factor}.png"
    plot_comparison(data, output_file=str(plot_file))

    # --- 4. Boundary Work Validation ---
    print("\n" + "="*80)
    print("BOUNDARY WORK VALIDATION")
    print("="*80)
    boundary_validation = None
    if "work_boundary_LV" in metrics:
        boundary_plot_file = results_dir / f"boundary_work_validation_downsample_{downsample_factor}.png"
        boundary_validation = plot_validation_boundary_work(data, metrics, output_file=str(boundary_plot_file))
    else:
        print("⚠ No boundary work data found")

    # --- 5. Phase-Windowed Analysis ---
    print("\n" + "="*80)
    print("PHASE-WINDOWED ANALYSIS")
    print("="*80)
    phase_analysis = None
    if "V_LV" in metrics:
        phase_plot_file = results_dir / f"phase_windowed_analysis_downsample_{downsample_factor}.png"
        phase_analysis = plot_phase_windowed_analysis(data, metrics, output_file=str(phase_plot_file))
    else:
        print("⚠ No volume data for phase analysis")

    # --- 6. Save Statistics ---
    stats_json = {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                      for kk, vv in v.items()} for k, v in stats.items()}
    
    if phase_analysis:
        stats_json["phase_analysis_LV"] = phase_analysis
    if boundary_validation:
        stats_json["boundary_validation"] = boundary_validation

    stats_file = results_dir / f"work_statistics_downsample_{downsample_factor}.json"
    with open(stats_file, "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"\n✓ Saved statistics: {stats_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE ✓")
    print("="*80)
    print("\nGenerated files:")
    print("  1. pv_loop_analysis.png")
    print("  2. hemodynamics_timeseries.png")
    print("  3. pv_loop_complete_cycle.png")
    print("  4. work_comparison_downsample_*.png")
    print("  5. boundary_work_validation_downsample_*.png")
    print("  6. phase_windowed_analysis_downsample_*.png")
    print("  7. work_statistics_downsample_*.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
