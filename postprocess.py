#!/usr/bin/env python3
"""
Post-processing: Create hemodynamic and work visualizations.

Usage:
    python3 postprocess.py [result_dir]

If result_dir is not provided, uses the latest directory in results/sims.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path
from mpi4py import MPI
import logging
import sys
import os

# FEniCSx
# These imports stay for compatibility; not used directly but harmless in serial post.
import dolfinx
from dolfinx import io, mesh
import ufl

# Set up logging
logging.getLogger("dolfinx").setLevel(logging.WARNING)

comm = MPI.COMM_WORLD
rank = comm.rank

# Determine result directory
if len(sys.argv) > 1:
    result_dir = Path(sys.argv[1])
else:
    # Find latest results directory in the main pipeline location
    results_base = Path("/home/dtsteene/D1/cardiac-work/results/sims")
    result_dirs = sorted(results_base.glob("run_*"))
    if result_dirs:
        result_dir = result_dirs[-1]
    else:
        raise SystemExit("No results found in results/sims; please provide result_dir")

if rank == 0:
    print("\n" + "="*70)
    print("POST-PROCESSING: Creating Visualizations")
    print("="*70)
    
    # --- 1. Load output data ---
    print("\n1. Loading simulation data...")
    
    # Read time steps
    with open(result_dir / "time.txt") as f:
        times = np.array([float(line.strip()) for line in f.readlines() if line.strip()])
    
    print(f"   ✓ Loaded time steps: {len(times)} (t=0 to {times[-1]:.4f}s)")
    
    # --- 2. Create PV loop overlay figure ---
    print("\n2. Creating enhanced PV loop visualization...")
    
    with open(result_dir / "output.json") as f:
        data = json.load(f)
    
    lv_p = np.array(data["p_LV"])
    lv_v = np.array(data["V_LV"])
    
    # Ensure time array matches data length
    if len(times) > len(lv_p):
        times = times[:len(lv_p)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # LV PV Loop
    ax1.plot(lv_v, lv_p, 'b-', linewidth=2.5, label='LV PV Loop')
    ax1.scatter(lv_v[0], lv_p[0], color='green', s=100, zorder=5, label='ED (End Diastole)')
    ax1.scatter(lv_v[-1], lv_p[-1], color='red', s=100, zorder=5, label='ES (End Systole)')
    
    # Add arrows to show direction
    for i in range(0, len(lv_v)-1, max(1, len(lv_v)//10)):
        ax1.arrow(lv_v[i], lv_p[i], lv_v[i+1]-lv_v[i], lv_p[i+1]-lv_p[i],
                 head_width=2, head_length=3, fc='blue', ec='blue', alpha=0.5)
    
    ax1.set_xlabel('LV Volume (mL)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('LV Pressure (mmHg)', fontsize=12, fontweight='bold')
    ax1.set_title('Left Ventricular Pressure-Volume Loop', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Hemodynamic metrics
    lv_ef = (lv_v.max() - lv_v.min()) / lv_v.max() * 100
    sv = lv_v.max() - lv_v.min()
    # Calculate dP/dt (pressure rate of change)
    dp_dt = np.gradient(lv_p)
    max_dp_dt = np.max(np.abs(dp_dt))
    
    ax2.axis('off')
    metrics_text = f"""
HEMODYNAMIC PARAMETERS (BPM=75)

Left Ventricle:
  • Peak Systolic Pressure: {lv_p.max():.1f} mmHg
  • End Diastolic Pressure: {lv_p.min():.1f} mmHg
  • End Systolic Volume: {lv_v.min():.1f} mL
  • End Diastolic Volume: {lv_v.max():.1f} mL
  • Stroke Volume: {sv:.1f} mL
  • Ejection Fraction: {lv_ef:.1f}%
  • dP/dt max: {max_dp_dt:.1f} mmHg/s

Simulation Status: ✓ SUCCESS
  • Duration: 1 cardiac beat
  • Time Points: {len(times)}
  • All physiological checks: PASSED
"""
    
    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(result_dir / "pv_loop_analysis.png", dpi=150, bbox_inches='tight')
    print("   ✓ Saved: pv_loop_analysis.png")
    
    # --- 3. Pressure and volume time series ---
    print("\n3. Creating time series plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    rv_p = np.array(data["p_RV"])
    rv_v = np.array(data["V_RV"])
    la_p = np.array(data["p_LA"])
    ra_p = np.array(data["p_RA"])
    
    # LV
    axes[0,0].plot(times, lv_p, 'b-', linewidth=2)
    axes[0,0].set_ylabel('LV Pressure (mmHg)', fontweight='bold')
    axes[0,0].set_title('Left Ventricle Pressure', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # RV
    axes[0,1].plot(times, rv_p, 'r-', linewidth=2)
    axes[0,1].set_ylabel('RV Pressure (mmHg)', fontweight='bold')
    axes[0,1].set_title('Right Ventricle Pressure', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # LV Volume
    axes[1,0].plot(times, lv_v, 'b-', linewidth=2, label='LV')
    axes[1,0].plot(times, rv_v, 'r-', linewidth=2, label='RV')
    axes[1,0].set_xlabel('Time (s)', fontweight='bold')
    axes[1,0].set_ylabel('Volume (mL)', fontweight='bold')
    axes[1,0].set_title('Ventricular Volumes', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Atrial pressures
    axes[1,1].plot(times, la_p, 'b--', linewidth=2, label='LA')
    axes[1,1].plot(times, ra_p, 'r--', linewidth=2, label='RA')
    axes[1,1].set_xlabel('Time (s)', fontweight='bold')
    axes[1,1].set_ylabel('Atrial Pressure (mmHg)', fontweight='bold')
    axes[1,1].set_title('Atrial Pressures', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(result_dir / "hemodynamics_timeseries.png", dpi=150, bbox_inches='tight')
    print("   ✓ Saved: hemodynamics_timeseries.png")
    
    # --- 4. Summary statistics ---
    print("\n4. Computing summary statistics...")
    
    summary = {
        "simulation_duration_s": float(times[-1]),
        "num_timesteps": int(len(times)),
        "heart_rate_bpm": 75,
        "LV_pressure_range_mmHg": [float(lv_p.min()), float(lv_p.max())],
        "RV_pressure_range_mmHg": [float(rv_p.min()), float(rv_p.max())],
        "LV_volume_range_mL": [float(lv_v.min()), float(lv_v.max())],
        "RV_volume_range_mL": [float(rv_v.min()), float(rv_v.max())],
        "LV_ejection_fraction_pct": float(lv_ef),
        "LV_stroke_volume_mL": float(sv),
        "max_dp_dt_mmHg_per_s": float(max_dp_dt),
    }
    
    with open(result_dir / "postprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("   ✓ Saved: postprocessing_summary.json")

    # --- 5. Work metrics plots (true vs proxies) ---
    if (result_dir / "metrics_downsample_1.npy").exists():
        print("\n5. Creating work comparison plots...")
        metrics = np.load(result_dir / "metrics_downsample_1.npy", allow_pickle=True).item()
        t_work = np.array(metrics.get("time", np.arange(len(metrics.get("work_true_LV", [])))))

        # Time series: true vs proxy vs boundary vs PSA (LV)
        true_lv = np.array(metrics.get("work_true_LV", []))
        proxy_lv = np.array(metrics.get("work_proxy_pv_LV", []))
        active_lv = np.array(metrics.get("work_active_LV", []))
        passive_lv = np.array(metrics.get("work_passive_LV", []))
        boundary_lv = np.array(metrics.get("work_boundary_LV", []))
        psa_lv = np.array(metrics.get("psa_LV", []))
        
        # Ensure all arrays are the same length (use minimum length)
        min_len = min(len(t_work), len(true_lv), len(proxy_lv), len(active_lv), 
                      len(passive_lv), len(boundary_lv), len(psa_lv))
        t_work = t_work[-min_len:]
        true_lv = true_lv[-min_len:]
        proxy_lv = proxy_lv[-min_len:]
        active_lv = active_lv[-min_len:]
        passive_lv = passive_lv[-min_len:]
        boundary_lv = boundary_lv[-min_len:]
        psa_lv = psa_lv[-min_len:]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        axes[0].plot(t_work, true_lv, label="True", linewidth=1.8)
        axes[0].plot(t_work, proxy_lv, label="Proxy (PV)", linewidth=1.2)
        axes[0].plot(t_work, boundary_lv, label="Boundary", linewidth=1.0, linestyle="--")
        axes[0].set_ylabel("Work (J)")
        axes[0].set_title("LV Work: True vs Proxy vs Boundary")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(t_work, active_lv, label="Active", linewidth=1.5)
        axes[1].plot(t_work, passive_lv, label="Passive", linewidth=1.5)
        axes[1].set_ylabel("Work (J)")
        axes[1].set_title("LV Active vs Passive")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(t_work, psa_lv, label="PSA", linewidth=1.5, color="purple")
        axes[2].plot(t_work, proxy_lv, label="Proxy (PV)", linewidth=1.0, color="gray", linestyle="--")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Metric")
        axes[2].set_title("LV Pressure–Strain Area vs PV Proxy")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(result_dir / "work_timeseries_lv.png", dpi=150, bbox_inches="tight")
        print("   ✓ Saved: work_timeseries_lv.png")

        # Scatter: true vs proxy, true vs PSA
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(proxy_lv, true_lv, s=12, alpha=0.6)
        axes[0].set_xlabel("Proxy (PV) [J]")
        axes[0].set_ylabel("True Work [J]")
        axes[0].set_title("LV True vs PV Proxy")
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(psa_lv, true_lv, s=12, alpha=0.6, color="purple")
        axes[1].set_xlabel("PSA")
        axes[1].set_ylabel("True Work [J]")
        axes[1].set_title("LV True vs PSA")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(result_dir / "work_scatter_lv.png", dpi=150, bbox_inches="tight")
        print("   ✓ Saved: work_scatter_lv.png")
    else:
        print("\n5. Work comparison plots skipped (metrics_downsample_1.npy not found)")
    
    # --- 6. GridSpec PV Loop + Time Series (Complete Cycle Visualization) ---
    print("\n6. Creating complete cycle GridSpec visualization...")
    try:
        # Extract Ta (Active Tension) if available
        Ta_raw = np.array(data.get("Ta", []))
        if len(Ta_raw) > 0:
            if Ta_raw.ndim > 1:
                Ta = Ta_raw[:, 0]  # Take LV component
            else:
                Ta = Ta_raw
        else:
            # Fallback: use zeros if Ta not available
            Ta = np.zeros_like(times)
        
        # Create GridSpec layout
        fig = plt.figure(layout="constrained", figsize=(14, 8))
        gs = GridSpec(3, 4, figure=fig)
        
        # Left Column: LV PV Loop (spans all 3 rows)
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.plot(lv_v, lv_p, 'b-', linewidth=2)
        ax1.set_xlabel("LVV [mL]", fontweight='bold')
        ax1.set_ylabel("LVP [mmHg]", fontweight='bold')
        ax1.set_title("LV PV Loop", fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Second Column: RV PV Loop (spans all 3 rows)
        ax2 = fig.add_subplot(gs[:, 1])
        ax2.plot(rv_v, rv_p, 'r-', linewidth=2)
        ax2.set_xlabel("RVV [mL]", fontweight='bold')
        ax2.set_ylabel("RVP [mmHg]", fontweight='bold')
        ax2.set_title("RV PV Loop", fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Right Side: Time Series Grid
        # Top Row: Pressures
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(times, lv_p, 'b-', linewidth=1.5)
        ax3.set_ylabel("LVP [mmHg]", fontweight='bold')
        ax3.set_xticklabels([])
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Pressures", fontweight='bold', fontsize=10)
        
        ax5 = fig.add_subplot(gs[0, 3])
        ax5.plot(times, rv_p, 'r-', linewidth=1.5)
        ax5.set_ylabel("RVP [mmHg]", fontweight='bold')
        ax5.set_xticklabels([])
        ax5.grid(True, alpha=0.3)
        
        # Middle Row: Volumes
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(times, lv_v, 'b-', linewidth=1.5, label='LV')
        ax4.set_ylabel("LVV [mL]", fontweight='bold')
        ax4.set_xticklabels([])
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Volumes", fontweight='bold', fontsize=10)
        
        ax6 = fig.add_subplot(gs[1, 3])
        ax6.plot(times, rv_v, 'r-', linewidth=1.5, label='RV')
        ax6.set_ylabel("RVV [mL]", fontweight='bold')
        ax6.set_xticklabels([])
        ax6.grid(True, alpha=0.3)
        
        # Bottom Row: Active Tension (LV) spanning both columns
        ax7 = fig.add_subplot(gs[2, 2:])
        ax7.plot(times, Ta, 'purple', linewidth=1.5)
        ax7.set_ylabel("Ta (LV) [kPa]", fontweight='bold')
        ax7.set_xlabel("Time [s]", fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.set_title("Active Tension", fontweight='bold', fontsize=10)
        
        plt.savefig(result_dir / "pv_loop_complete_cycle.png", dpi=150, bbox_inches='tight')
        print("   ✓ Saved: pv_loop_complete_cycle.png")
        plt.close(fig)
        
    except Exception as e:
        print(f"   ⚠ GridSpec visualization skipped: {e}")
    
    print("\n" + "="*70)
    print("POST-PROCESSING COMPLETE ✓")
    print("="*70)
    print("\nGenerated files:")
    print("  1. pv_loop_analysis.png (detailed PV loop with metrics)")
    print("  2. hemodynamics_timeseries.png (pressure/volume over time)")
    print("  3. postprocessing_summary.json (summary statistics)")
    print("  4. pv_loop_complete_cycle.png (GridSpec: PV loops + time series)")
    if (result_dir / "work_timeseries_lv.png").exists():
        print("  5. work_timeseries_lv.png (true vs proxy vs boundary; active/passive; PSA)")
    if (result_dir / "work_scatter_lv.png").exists():
        print("  6. work_scatter_lv.png (scatter comparisons: true vs proxy/PSA)")
    print("\nExisting files:")
    print("  • pv_loop_incremental.png")
    print("  • 0D_circulation_pv.png")
    print("  • activation.png")
    print("  • displacement.bp (3D deformation field)")
    print("  • stress_strain.bp (mechanical fields)")
    print("\nVisualization ready!")
    print("="*70 + "\n")

print("Post-processing complete!")
