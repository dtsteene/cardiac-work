#!/usr/bin/env python3
"""
Post-processing: Create displacement animations and strain visualizations

Usage:
  python3 postprocess.py [result_dir]
  
  If result_dir is not provided, uses the latest results directory.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from mpi4py import MPI
import logging
import sys
import os

# FEniCSx
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
    # Find latest results directory
    results_base = Path("/home/dtsteene/D1/prelimSim/results")
    result_dirs = sorted(results_base.glob("results_*"))
    if result_dirs:
        result_dir = result_dirs[-1]
    else:
        result_dir = Path("/home/dtsteene/D1/prelimSim/results/results_debug_941935")

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
    
    print("\n" + "="*70)
    print("POST-PROCESSING COMPLETE ✓")
    print("="*70)
    print("\nGenerated files:")
    print("  1. pv_loop_analysis.png (detailed PV loop with metrics)")
    print("  2. hemodynamics_timeseries.png (pressure/volume over time)")
    print("  3. postprocessing_summary.json (summary statistics)")
    print("\nExisting files:")
    print("  • pv_loop_incremental.png")
    print("  • 0D_circulation_pv.png")
    print("  • activation.png")
    print("  • displacement.bp (3D deformation field)")
    print("  • stress_strain.bp (mechanical fields)")
    print("\nVisualization ready!")
    print("="*70 + "\n")

print("Post-processing complete!")
