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

    print(f"Starting Analysis on: {results_dir}")

    # --- 0. Run Metrics Computation if Needed ---
    metrics_subdir = results_dir / "metrics"
    # Look in metrics/ first, then root (legacy)
    metrics_files = sorted(list(metrics_subdir.glob("metrics_downsample_*.npy")))
    if not metrics_files:
        metrics_files = sorted(list(results_dir.glob("metrics_downsample_*.npy")))
    sim_params_file = results_dir / "simulation_params.json"
    checkpoint_file = results_dir / "solver" / "checkpoint.bp"

    if not metrics_files and sim_params_file.exists() and checkpoint_file.exists():
        print("No metrics files found — running postprocess_metrics.py to compute them...")
        script_dir = Path(__file__).resolve().parent
        postprocess_script = script_dir / "postprocess_metrics.py"
        if postprocess_script.exists():
            # Use mpirun if available, otherwise single process
            import shutil
            mpirun = shutil.which("mpirun")
            if mpirun:
                cmd = [mpirun, "-n", "1", sys.executable, str(postprocess_script), str(results_dir)]
            else:
                cmd = [sys.executable, str(postprocess_script), str(results_dir)]
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print("ERROR: postprocess_metrics.py failed")
                sys.exit(1)
            # Refresh metrics files list (check metrics/ subdir first)
            metrics_files = sorted(list(metrics_subdir.glob("metrics_downsample_*.npy")))
            if not metrics_files:
                metrics_files = sorted(list(results_dir.glob("metrics_downsample_*.npy")))
        else:
            print(f"ERROR: {postprocess_script} not found")
            sys.exit(1)
    elif not metrics_files:
        print("ERROR: No metrics files and no checkpoint data for recomputation")
        sys.exit(1)

    # --- 1. Determine Cycle Length ---
    params_file = results_dir / "parameters.json"
    if not params_file.exists():
        params_file = results_dir / "circulation" / "parameters.json"
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
    # Find the metrics file (prefer downsample_1); check metrics/ subdir, then root (legacy)
    metrics_files = sorted(list(metrics_subdir.glob("metrics_downsample_*.npy")), key=lambda p: len(p.name))
    if not metrics_files:
        metrics_files = sorted(list(results_dir.glob("metrics_downsample_*.npy")), key=lambda p: len(p.name))
    if not metrics_files:
        print("Error: No metrics_downsample_*.npy files found.")
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
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    all_beats_dir = analysis_dir / "all_beats"
    last_beat_dir = analysis_dir / "last_beat"
    
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

    for label, base_dir in [("ALL BEATS", all_beats_dir), ("LAST BEAT", last_beat_dir)]:
        print(f"\n--- 📊 Generating Reports for {label} ---")
        
        # The metrics file inside the base directory
        npy_file = base_dir / src_metrics_path.name
        if not npy_file.exists():
            print(f"⚠️  Missing metrics file in {base_dir}, skipping...")
            continue

        # Define Configurations
        configs = [
            ("total", "truth_total"), 
            ("fiber", "truth_fiber")
        ]

        for truth_mode, subfolder_name in configs:
            print(f"   👉 Mode: {truth_mode.upper()} -> {subfolder_name}/")
            
            # 1. Create Sub-directory
            work_dir = base_dir / subfolder_name
            work_dir.mkdir(exist_ok=True)
            
            # 2. Symlink Data (Avoid duplication)
            # Create a link inside 'work_dir' pointing to '../metrics.npy'
            link_target = work_dir / src_metrics_path.name
            if not link_target.exists():
                # Use relative path for portability if moved, or absolute? 
                # Absolute is safer for script execution.
                try:
                    link_target.symlink_to(npy_file)
                except FileExistsError:
                    pass
            
            # 3. Run Eval Proxies
            cmd_eval = [sys.executable, str(eval_proxies_script), str(work_dir), "--truth", truth_mode]
            subprocess.run(cmd_eval, env=env, check=False)
            
            # 4. Run Plot Loops (Generates standard clinical dashboard + engineering debug)
            # Use 'total' engineering debug for consistency, but saves plots in this folder
            # so the user has a self-contained set of images for this "truth mode".
            cmd_plot = [sys.executable, str(plot_loops_script), str(work_dir)]
            subprocess.run(cmd_plot, env=env, check=False)

    # --- 7. Organize Results into Subdirectories ---
    print("\n--- 📁 Organizing results ---")
    organize_results(results_dir)

    print(f"\n✅ Analysis pipeline finished for: {results_dir.name}")


def organize_results(results_dir):
    """
    Organize simulation outputs into a clean directory structure:
      solver/        — FEM checkpoint data (displacement, Ta, prestress)
      geometry/      — mesh, fibers, markers
      circulation/   — 0D model data
      visualization/ — ParaView-ready files
      metrics/       — computed offline by postprocess_metrics.py
      analysis/      — plots and reports
    """
    import shutil

    # --- 1. Circulation files (written to root by circulation model) ---
    circ_dir = results_dir / "circulation"
    circ_dir.mkdir(exist_ok=True)
    circ_files = [
        "history.npy", "state.npy", "initial_conditions.json",
        "parameters.json", "0D_circulation_pv.png",
    ]
    for fname in circ_files:
        src = results_dir / fname
        dst = circ_dir / fname
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  Moved {fname} -> circulation/")

    # Symlink parameters.json back to root (needed by this script for HR)
    params_link = results_dir / "parameters.json"
    params_real = circ_dir / "parameters.json"
    if params_real.exists() and not params_link.exists():
        params_link.symlink_to(params_real)

    # --- 2. Visualization files (written to root by geometry_generator) ---
    viz_dir = results_dir / "visualization"
    viz_dir.mkdir(exist_ok=True)
    viz_files = [
        "markers_scalar.xdmf", "markers_scalar.h5",
        "debug_surfaces.xdmf", "debug_surfaces.h5",
        "fiber_directions.bp", "fiber_directions.h5", "fiber_directions.xdmf",
        "activation.png",
    ]
    for fname in viz_files:
        src = results_dir / fname
        dst = viz_dir / fname
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.move(str(src), str(dst))
            else:
                shutil.move(str(src), str(dst))
            print(f"  Moved {fname} -> visualization/")

    # --- 3. Legacy solver files (from old directory layout) ---
    solver_dir = results_dir / "solver"
    solver_dir.mkdir(exist_ok=True)
    legacy_solver = {
        "function_checkpoint.bp": "checkpoint.bp",
        "prestress_biv_backward.bp": "prestress_backward.bp",
        "prestress_biv_inverse.bp": "prestress_inverse.bp",
        "Ta_history.npy": "Ta_history.npy",
        "Ta_solver_history.npy": "Ta_solver_history.npy",
    }
    for old_name, new_name in legacy_solver.items():
        src = results_dir / old_name
        dst = solver_dir / new_name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  Moved {old_name} -> solver/{new_name}")

    # --- 4. Legacy metrics files (move to metrics/) ---
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    for f in results_dir.glob("metrics_downsample_*.npy"):
        dst = metrics_dir / f.name
        if not dst.exists():
            shutil.move(str(f), str(dst))
            print(f"  Moved {f.name} -> metrics/")

    # --- 5. Legacy analysis directories ---
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    legacy_analysis = {
        "analysis_all_beats": "all_beats",
        "analysis_last_beat": "last_beat",
    }
    for old_name, new_name in legacy_analysis.items():
        src = results_dir / old_name
        dst = analysis_dir / new_name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  Moved {old_name} -> analysis/{new_name}")

    # --- 6. Legacy VTX visualization (from old layout) ---
    for vtx_name in ["displacement.bp", "stress_strain.bp", "pressure.bp"]:
        src = results_dir / vtx_name
        dst = viz_dir / vtx_name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  Moved {vtx_name} -> visualization/")

    # --- 7. Clean up redundant files ---
    redundant = [
        "output.json", "results_state.txt", "results_var.txt",
        "time.txt", "state.txt", "state_names.txt", "var_names.txt",
        "active_mechanics_trace.csv", "metrics_downsample_5.npy",
    ]
    for fname in redundant:
        fpath = results_dir / fname
        if fpath.exists():
            fpath.unlink()
            print(f"  Removed {fname} (redundant)")


if __name__ == "__main__":
    main()
