#!/usr/bin/env python3
"""
Comprehensive Work Calculation Diagnostics

This script checks BOTH proxy work and true work calculations
to identify any remaining bugs.

Usage:
    python3 diagnose_work.py <run_dir>

Example:
    python3 diagnose_work.py results/sims/run_942413
"""

import numpy as np
import json
import sys
from pathlib import Path

def diagnose_work(run_dir):
    """Comprehensive diagnostics for work calculations."""
    run_path = Path(run_dir)

    if not run_path.exists():
        print(f"Error: Directory {run_dir} does not exist")
        return 1

    print("="*80)
    print(f"WORK CALCULATION DIAGNOSTICS: {run_path.name}")
    print("="*80)

    # Load data
    history_file = run_path / "history.npy"
    metrics_file = run_path / "metrics_downsample_1.npy"
    output_file = run_path / "output.json"

    if not history_file.exists():
        print(f"Error: {history_file} not found")
        return 1

    history = np.load(history_file, allow_pickle=True).item()

    # --- 1. Check Volume and Pressure Swings ---
    print("\n### 1. HEMODYNAMICS CHECK ###")
    V_LV = history["V_LV"]
    V_RV = history["V_RV"]
    p_LV = history["p_LV"]
    p_RV = history["p_RV"]

    print(f"LV Volume:  {V_LV.min():.1f} - {V_LV.max():.1f} mL (Δ{V_LV.max()-V_LV.min():.1f})")
    print(f"RV Volume:  {V_RV.min():.1f} - {V_RV.max():.1f} mL (Δ{V_RV.max()-V_RV.min():.1f})")
    print(f"LV Pressure: {p_LV.min():.1f} - {p_LV.max():.1f} mmHg (Δ{p_LV.max()-p_LV.min():.1f})")
    print(f"RV Pressure: {p_RV.min():.1f} - {p_RV.max():.1f} mmHg (Δ{p_RV.max()-p_RV.min():.1f})")

    if V_LV.max() - V_LV.min() < 10:
        print("⚠ WARNING: LV volume swing is too small!")
    if p_LV.max() - p_LV.min() < 50:
        print("⚠ WARNING: LV pressure swing is too small!")

    # --- 2. Check Activation (Ta) ---
    print("\n### 2. ACTIVATION CHECK ###")
    if output_file.exists():
        output = json.load(open(output_file))
        if "Ta" in output:
            Ta = np.array(output["Ta"])
            if Ta.ndim == 2:
                Ta_mean = Ta.mean(axis=1)
            else:
                Ta_mean = Ta
            print(f"Ta shape: {Ta.shape}")
            print(f"Ta range: {Ta_mean.min():.2f} - {Ta_mean.max():.2f} kPa")
            print(f"Ta nonzero count: {np.count_nonzero(Ta_mean)} / {len(Ta_mean)}")

            if Ta_mean.max() < 10:
                print("⚠ WARNING: Activation is very weak!")
        else:
            print("❌ Ta NOT in output.json")
    else:
        print("❌ output.json not found")

    # --- 3. Check Proxy Work in Metrics File ---
    print("\n### 3. PROXY WORK (Stored During Simulation) ###")
    if metrics_file.exists():
        metrics = np.load(metrics_file, allow_pickle=True).item()

        work_proxy_LV = metrics.get("work_proxy_pv_LV", [])
        work_proxy_RV = metrics.get("work_proxy_pv_RV", [])
        work_proxy_Septum = metrics.get("work_proxy_pv_Septum", [])

        print(f"LV Proxy:  {len(work_proxy_LV)} timesteps")
        print(f"  Range: {np.min(work_proxy_LV):.3e} to {np.max(work_proxy_LV):.3e} J")
        print(f"  Nonzero: {np.count_nonzero(work_proxy_LV)} / {len(work_proxy_LV)}")
        print(f"  First 10: {work_proxy_LV[:10]}")

        print(f"\nRV Proxy:  {len(work_proxy_RV)} timesteps")
        print(f"  Range: {np.min(work_proxy_RV):.3e} to {np.max(work_proxy_RV):.3e} J")
        print(f"  Nonzero: {np.count_nonzero(work_proxy_RV)} / {len(work_proxy_RV)}")

        print(f"\nSeptum Proxy: {len(work_proxy_Septum)} timesteps")
        print(f"  Range: {np.min(work_proxy_Septum):.3e} to {np.max(work_proxy_Septum):.3e} J")
        print(f"  Nonzero: {np.count_nonzero(work_proxy_Septum)} / {len(work_proxy_Septum)}")

        # Check if first timestep is zero (expected)
        if work_proxy_LV[0] == 0.0:
            print("✓ First timestep is zero (expected)")
        else:
            print(f"⚠ First timestep is {work_proxy_LV[0]:.3e} (should be zero)")

        # Check if subsequent timesteps are nonzero
        if np.count_nonzero(work_proxy_LV[1:]) < len(work_proxy_LV) * 0.8:
            print("❌ FAIL: Many timesteps have zero proxy work!")
        else:
            print("✓ Most timesteps have nonzero proxy work")

        # Cumulative work
        cumul_LV = np.cumsum(work_proxy_LV)
        cumul_RV = np.cumsum(work_proxy_RV)
        print(f"\n💚 Cumulative LV Proxy Work: {cumul_LV[-1]:.3f} J")
        print(f"💚 Cumulative RV Proxy Work: {cumul_RV[-1]:.3f} J")

    else:
        print(f"❌ {metrics_file} not found")

    # --- 4. Check True Work ---
    print("\n### 4. TRUE WORK (Stress-Based) ###")
    if metrics_file.exists():
        work_true_LV = metrics.get("work_true_LV", [])
        work_true_RV = metrics.get("work_true_RV", [])
        work_true_Septum = metrics.get("work_true_Septum", [])

        print(f"LV True:   {len(work_true_LV)} timesteps")
        print(f"  Range: {np.min(work_true_LV):.3e} to {np.max(work_true_LV):.3e} J")
        print(f"  Nonzero: {np.count_nonzero(work_true_LV)} / {len(work_true_LV)}")
        print(f"  First 10: {work_true_LV[:10]}")

        print(f"\nRV True:   {len(work_true_RV)} timesteps")
        print(f"  Range: {np.min(work_true_RV):.3e} to {np.max(work_true_RV):.3e} J")
        print(f"  Nonzero: {np.count_nonzero(work_true_RV)} / {len(work_true_RV)}")

        print(f"\nSeptum True: {len(work_true_Septum)} timesteps")
        print(f"  Range: {np.min(work_true_Septum):.3e} to {np.max(work_true_Septum):.3e} J")
        print(f"  Nonzero: {np.count_nonzero(work_true_Septum)} / {len(work_true_Septum)}")

        # Cumulative work
        cumul_true_LV = np.cumsum(work_true_LV)
        cumul_true_RV = np.cumsum(work_true_RV)
        print(f"\n💙 Cumulative LV True Work: {cumul_true_LV[-1]:.3f} J")
        print(f"💙 Cumulative RV True Work: {cumul_true_RV[-1]:.3f} J")

        # Compare magnitudes
        print("\n### 5. WORK COMPARISON ###")
        ratio_LV = cumul_true_LV[-1] / cumul_LV[-1] if cumul_LV[-1] != 0 else 0
        ratio_RV = cumul_true_RV[-1] / cumul_RV[-1] if cumul_RV[-1] != 0 else 0

        print(f"LV: True/Proxy Ratio = {ratio_LV:.2f}")
        print(f"RV: True/Proxy Ratio = {ratio_RV:.2f}")

        if 0.5 < ratio_LV < 2.0:
            print("✓ LV work magnitudes are similar (good!)")
        elif ratio_LV < 0.1:
            print("❌ True Work is much smaller than Proxy (stress integration issue?)")
        elif ratio_LV > 10:
            print("❌ True Work is much larger than Proxy (unit issue?)")

        # Correlation
        if len(work_true_LV) > 2:
            corr_LV = np.corrcoef(work_true_LV[1:], work_proxy_LV[1:])[0, 1]
            corr_RV = np.corrcoef(work_true_RV[1:], work_proxy_RV[1:])[0, 1]
            print(f"\n📊 Correlation LV: {corr_LV:.3f}")
            print(f"📊 Correlation RV: {corr_RV:.3f}")

            if corr_LV > 0.7:
                print("✓ Strong correlation (excellent!)")
            elif corr_LV > 0.4:
                print("⚠ Moderate correlation")
            else:
                print("❌ Weak correlation (physics mismatch?)")

    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)

    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 diagnose_work.py <run_dir>")
        print("Example: python3 diagnose_work.py results/sims/run_942413")
        sys.exit(1)

    sys.exit(diagnose_work(sys.argv[1]))
