#!/usr/bin/env python3
"""
run_postprocessing.py

Automates the post-processing workflow:
1. Reads simulation results.
2. Determines cycle length from parameters.json (or defaults).
3. Slices data into 'analysis_all_beats' and 'analysis_last_beat'.
4. Runs 'eval_proxies.py' and 'plot_loops.py' for both sets.

Usage:
  python3 run_postprocessing.py <path_to_results_folder>
"""

import os
import sys
import json
import numpy as np
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_postprocessing.py <results_directory>")
        sys.exit(1)

    results_dir = Path(sys.argv[1]).resolve()
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist.")
        sys.exit(1)

    print(f"🚀 Starting Analysis on: {results_dir}")

    # --- 1. Determine Cycle Length ---
    params_file = results_dir / "parameters.json"
    cycle_length = 0.8 # Default
    
    if params_file.exists():
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
            if 'HR' in params:
                hr = params['HR']
                cycle_length = 1.0 / float(hr)
                print(f"ℹ️  Read HR={hr} Hz -> Cycle Length = {cycle_length:.4f} s")
            else:
                 print("⚠️  'HR' not found in parameters.json. Defaulting to 0.8s.")
        except Exception as e:
            print(f"⚠️  Error reading parameters.json: {e}. Defaulting to 0.8s.")
    else:
        print("⚠️  parameters.json not found. Defaulting to 0.8s.")

    # --- 2. Load Metrics ---
    # Find the metrics file (prefer downsample_1)
    metrics_files = sorted(list(results_dir.glob("metrics_downsample_*.npy")), key=lambda p: len(p.name))
    if not metrics_files:
        print("❌ Error: No metrics_downsample_*.npy files found.")
        sys.exit(1)
    
    src_metrics_path = metrics_files[0]
    print(f"📂 Using data file: {src_metrics_path.name}")
    
    try:
        metrics = np.load(src_metrics_path, allow_pickle=True).item()
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")
        sys.exit(1)

    if 'time' not in metrics:
        print("❌ Error: 'time' array not found in metrics.")
        sys.exit(1)

    time = np.array(metrics['time'])
    if len(time) == 0:
        print("❌ Error: Time array is empty.")
        sys.exit(1)

    final_time = time[-1]
    last_beat_start = final_time - cycle_length
    
    # --- 3. Create Analysis Directories ---
    all_beats_dir = results_dir / "analysis_all_beats"
    last_beat_dir = results_dir / "analysis_last_beat"
    
    all_beats_dir.mkdir(exist_ok=True)
    last_beat_dir.mkdir(exist_ok=True)

    # --- 4. Save 'All Beats' ---
    # Copy/Save the full dataset
    np.save(all_beats_dir / src_metrics_path.name, metrics)
    print(f"💾 Prepared: {all_beats_dir.name}")

    # --- 5. Slice for 'Last Beat' ---
    mask_last = time >= last_beat_start
    print(f"✂️  Slicing last beat: t >= {last_beat_start:.4f} s ... {last_beat_start+cycle_length:.4f} s")
    print(f"   Points found: {np.sum(mask_last)}")
    
    if np.sum(mask_last) < 10:
        print("⚠️  Warning: Very few points found for the last beat. Check simulation time vs cycle length.")

    metrics_last = {}
    
    # Handle N and N-1 length arrays
    n_time = len(time)
    n_work = n_time - 1
    
    mask_last_work = mask_last[1:] if len(mask_last) > 1 else mask_last # Adjust if necessary, usually work is dt steps
    # Actually, simpler: work[i] is step i->i+1. If we keep time[i], we keep work[i] IF time[i+1] is also kept.
    # But let's just slice by length.
    
    for k, v in metrics.items():
        if isinstance(v, (list, np.ndarray)):
            arr = np.array(v)
            if len(arr) == n_time:
                metrics_last[k] = arr[mask_last]
            elif len(arr) == n_work:
                # Slice work arrays: we need mask of length N-1
                # If mask_last matches time, work array corresponds to Intervals.
                # We want intervals that are "in" the last beat.
                # If we keep time indices [A, ..., B], we normally want work indices [A, ..., B-1].
                # This corresponds to mask_last[:-1]
                metrics_last[k] = arr[mask_last[:-1]]
            else:
                metrics_last[k] = v
        else:
            metrics_last[k] = v
            
    np.save(last_beat_dir / src_metrics_path.name, metrics_last)
    print(f"💾 Prepared: {last_beat_dir.name}")

    # --- 6. Run Analysis Scripts ---
    # We find the scripts relative to THIS script location
    script_dir = Path(__file__).resolve().parent
    eval_proxies_script = script_dir / "eval_proxies.py"
    plot_loops_script = script_dir / "plot_loops.py"

    if not eval_proxies_script.exists():
        print(f"❌ Could not find {eval_proxies_script}")
        sys.exit(1)

    env = os.environ.copy()

    for label, target_dir in [("ALL BEATS", all_beats_dir), ("LAST BEAT", last_beat_dir)]:
        print(f"\n--- 📊 Generating Reports for {label} ---")
        
        # 1. Quantify Proxies
        print(f"   Running eval_proxies.py...")
        cmd1 = [sys.executable, str(eval_proxies_script), str(target_dir)]
        subprocess.run(cmd1, env=env, check=False)
        
        # 2. Generate Meeting Report
        print(f"   Running plot_loops.py...")
        cmd2 = [sys.executable, str(plot_loops_script), str(target_dir)]
        subprocess.run(cmd2, env=env, check=False)

    print(f"\n✅ Analysis pipeline finished for: {results_dir.name}")

if __name__ == "__main__":
    main()
