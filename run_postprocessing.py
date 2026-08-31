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
    metrics_files = sorted(metrics_subdir.glob("metrics_downsample_*.npy"))
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
    
    # Cycle length sets the beat segmentation, so a wrong value silently
    # corrupts every per-beat metric. If the 0D model ran, its HR must be
    # readable; only a run with no circulation output may use the default.
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
        hr = params["HR"]
        cycle_length = 1.0 / float(hr)
        print(f"ℹ️  Read HR={hr} Hz -> Cycle Length = {cycle_length:.4f} s")
    else:
        print("⚠️  No circulation parameters.json — defaulting to 0.8 s cycle length.")

    # --- 2. Load Metrics (finest downsampling available) ---
    metrics_files = sorted(
        metrics_subdir.glob("metrics_downsample_*.npy"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    if not metrics_files:
        print("Error: No metrics_downsample_*.npy files found.")
        sys.exit(1)
    
    src_metrics_path = metrics_files[0]
    print(f"📂 Using data file: {src_metrics_path.name}")
    
    metrics = np.load(src_metrics_path, allow_pickle=True).item()

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
    report_failures = []

    for label, work_dir in [("ALL BEATS", all_beats_dir), ("LAST BEAT", last_beat_dir)]:
        print(f"\n--- 📊 Generating Reports for {label} ---")

        if not (work_dir / src_metrics_path.name).exists():
            print(f"⚠️  Missing metrics file in {work_dir}, skipping...")
            continue

        # Run eval proxies (total S:dE ground truth), then plot loops
        # (clinical dashboard + engineering debug). Neither aborts the other,
        # but a non-zero exit is recorded: these used to fail silently, so a
        # run could report success having produced no figures at all.
        for name, script in [("eval_proxies", eval_proxies_script),
                             ("plot_loops", plot_loops_script)]:
            rc = subprocess.run([sys.executable, str(script), str(work_dir)],
                                env=env, check=False).returncode
            if rc != 0:
                print(f"❌ {name}.py failed on {label} (exit {rc})")
                report_failures.append(f"{name} [{label}] exit {rc}")

    # --- 7. Organize Results into Subdirectories ---
    print("\n--- 📁 Organizing results ---")
    organize_results(results_dir)

    if report_failures:
        print(f"\n❌ Analysis pipeline finished WITH FAILURES for: {results_dir.name}")
        for f in report_failures:
            print(f"   - {f}")
        sys.exit(1)

    print(f"\n✅ Analysis pipeline finished for: {results_dir.name}")


def organize_results(results_dir):
    """Tidy the files the solver and 0D model write to the results root.

    complete_cycle.py already writes into solver/, visualization/, and
    geometry/, and postprocess_metrics.py writes into metrics/. What still
    lands in the root is the 0D circulation output (history/state/parameters)
    and the geometry debug XDMF, so only those are relocated here.
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
            shutil.move(str(src), str(dst))
            print(f"  Moved {fname} -> visualization/")

    # --- 3. Clean up the 0D solver's plain-text dumps (superseded by history.npy) ---
    redundant = [
        "output.json", "results_state.txt", "results_var.txt",
        "time.txt", "state.txt", "state_names.txt", "var_names.txt",
        "active_mechanics_trace.csv",
    ]
    for fname in redundant:
        fpath = results_dir / fname
        if fpath.exists():
            fpath.unlink()
            print(f"  Removed {fname} (redundant)")


if __name__ == "__main__":
    main()
