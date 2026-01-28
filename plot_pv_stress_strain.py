import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main(results_dir_str):
    results_dir = Path(results_dir_str)
    print(f"\nANALYZE: {results_dir_str}")
    metrics_path = results_dir / "metrics_downsample_1.npy"
    
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found")
        return
        
    data = np.load(metrics_path, allow_pickle=True).item()
    
    # --- 1. Work Analysis ---
    # Sum of incremental work
    w_tensor = np.sum(data["work_true_LV"])
    w_fiber = np.sum(data["work_fiber_LV"])
    
    # Component Breakdown
    w_sheet = np.sum(data.get("work_sheet_LV", [0]))
    w_normal = np.sum(data.get("work_normal_LV", [0]))
    w_shear = np.sum(data.get("work_shear_LV", [0]))
    
    w_pv = np.sum(data.get("work_proxy_pv_LV", [0]))
    
    print("-" * 60)
    print("ENERGY BALANCE RESULTS (LV)")
    print("-" * 60)
    print(f"Total Internal Work (Tensor): {w_tensor:.4f} J")
    print(f"   - Fiber Work:              {w_fiber:.4f} J")
    print(f"   - Sheet Work:              {w_sheet:.4f} J")
    print(f"   - Normal Work:             {w_normal:.4f} J")
    print(f"   - Shear/Cross Work:        {w_shear:.4f} J")
    print(f"Total PV Area Proxy:          {w_pv:.4f} J")
    
    # Check sum
    sum_components = w_fiber + w_sheet + w_normal + w_shear
    diff = w_tensor - sum_components
    print(f"Sum Check Error:              {diff:.2e} J")
    
    if abs(w_tensor) > 1e-6:
        ratio = w_fiber / w_tensor
        print(f"Fiber/Tensor Ratio:           {ratio:.4f}")
    else:
        print("Work too small to calculate ratio")
        
    print("-" * 60)
    
    # --- 2. Plot Stress-Strain Loops ---
    # Use mean fiber stress/strain
    time = np.array(data["time"])
    s_ff = np.array(data["mean_S_ff_LV"]) / 1000.0 # Convert Pa -> kPa
    e_ff = np.array(data["mean_E_ff_LV"]) * 100.0 # Percent
    
    # Ensure lengths match
    min_len = min(len(time), len(s_ff), len(e_ff))
    time = time[:min_len]
    s_ff = s_ff[:min_len]
    e_ff = e_ff[:min_len]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(e_ff, s_ff, c=time, cmap='viridis', s=10, label='Cycle Path')
    ax.plot(e_ff, s_ff, 'k-', alpha=0.3)
    
    # Mark Start/End
    ax.plot(e_ff[0], s_ff[0], 'go', label='Start (ED)')
    ax.plot(e_ff[-1], s_ff[-1], 'rx', label='End')
    
    ax.set_xlabel("Fiber Strain (%)")
    ax.set_ylabel("Fiber Stress (kPa)")
    ax.set_title("LV Averaged Fiber Stress-Strain Loop")
    plt.colorbar(sc, label="Time (s)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    out_file = results_dir / "stress_strain_loop.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot: {out_file}")
    plt.close(fig)
    
    # --- 3. Work Time Series ---
    # Cumulative work over time
    cum_tensor = np.cumsum(data["work_true_LV"])
    
    # helper for robust length
    def safer_cumsum(key, target_len):
        arr = data.get(key, np.zeros(target_len))
        if len(arr) == 0:
            return np.zeros(target_len)
        return np.cumsum(arr)

    cum_fiber = np.cumsum(data["work_fiber_LV"])
    cum_normal = safer_cumsum("work_normal_LV", len(cum_tensor))
    cum_shear = safer_cumsum("work_shear_LV", len(cum_tensor))

    # Align Time and Work lengths
    n_plot = min(len(time), len(cum_tensor))
    t_plot = time[:n_plot]
    cum_tensor = cum_tensor[:n_plot]
    cum_fiber = cum_fiber[:n_plot]
    cum_normal = cum_normal[:n_plot]
    cum_shear = cum_shear[:n_plot]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t_plot, cum_tensor, 'k-', label='Total Internal Work', linewidth=2)
    ax2.plot(t_plot, cum_fiber, 'r--', label='Fiber Work', linewidth=2)
    ax2.plot(t_plot, cum_normal, 'g:', label='Normal Work', linewidth=1.5)
    ax2.plot(t_plot, cum_shear, 'b:', label='Shear Work', linewidth=1.5)
    
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cumulative Work (J)")
    ax2.set_title(f"Work Decomposition")
    ax2.legend()
    ax2.grid(True)
    
    out_file2 = results_dir / "work_accumulation.png"
    plt.savefig(out_file2, dpi=150)
    print(f"Saved plot: {out_file2}")
    plt.close(fig2)


def diagnose_active(results_dir):
    path = Path(results_dir) / "metrics_downsample_1.npy"
    if not path.exists():
        print(f"File not found: {path}")
        return

    data = np.load(path, allow_pickle=True).item()
    t = np.array(data["time"])
    
    # 1. Check Input: Active Tension (Ta)
    if "Ta" in data:
        Ta = np.array(data["Ta"])
        if Ta.ndim > 1: Ta = Ta.flatten()
    else:
        print("WARNING: 'Ta' (Active Tension scalar) not found in metrics.")
        Ta = np.zeros_like(t)

    # 2. Check Output: Active Stress Tensor Magnitude
    S_active_mag = np.array(data.get("mean_S_active_LV", np.zeros_like(t)))
    
    # 3. Check Integral: Active Work
    # Work is cumulative sum, sometimes length matches t, sometimes t-1
    W_active_raw = np.array(data.get("work_active_LV", np.zeros_like(t)))
    W_active_cum = np.cumsum(W_active_raw)

    # --- FIX: ALIGN LENGTHS ---
    # Find the minimum common length to avoid shape mismatch errors
    min_len = min(len(t), len(S_active_mag), len(W_active_cum), len(Ta))
    
    t = t[:min_len]
    S_active_mag = S_active_mag[:min_len]
    W_active_cum = W_active_cum[:min_len]
    Ta = Ta[:min_len]

    print(f"\n--- DIAGNOSTICS: {results_dir} ---")
    print(f"Max Input Ta:        {np.max(Ta):.4e}")
    print(f"Max Output S_active: {np.max(S_active_mag):.4e} (Should be ~ Ta)")
    print(f"Total Active Work:   {W_active_cum[-1]:.4e}")

    # PLOT
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Active Stress Magnitude (Pa)', color=color)
    l1, = ax1.plot(t, S_active_mag, color=color, label='Calc. Active Stress (Mean)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Cumulative Active Work (J)', color=color)  
    l2, = ax2.plot(t, W_active_cum, color=color, linestyle='--', label='Cumul. Active Work')
    ax2.tick_params(axis='y', labelcolor=color)

    # Plot Input Ta if available (normalized to fit plot 1)
    if np.max(Ta) > 1e-6:
        # Scale Ta to match S_active range for visual comparison
        scaling_factor = np.max(S_active_mag) / np.max(Ta)
        l3, = ax1.plot(t, Ta * scaling_factor, 'k:', alpha=0.5, label='Input Ta (Scaled)')
        lines = [l1, l2, l3]
    else:
        lines = [l1, l2]

    # Combine legends
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')
    plt.title(f"Active Mechanics Audit\n(Is S_active responding to Time?)")
    
    out_path = Path(results_dir) / "debug_active_mechanics.png"
    plt.savefig(out_path)
    print(f"Saved diagnostic plot: {out_path}")
    plt.close(fig)
    
    # HEURISTIC CHECK
    if np.max(S_active_mag) < 100.0: 
        print("\n[CRITICAL ERROR] Active Stress is near ZERO.")
        print("Possible causes:")
        print("1. The 'Ta' variable in cardiac_model is not being updated.")
        print("2. The fiber orientation f0 is zero.")
        print("3. The metrics calculator is using a COPY of the model, not the live one.")
    elif abs(W_active_cum[-1]) < 1e-6:
        print("\n[CRITICAL ERROR] Active Stress exists, but Work is ZERO.")
        print("Possible cause: S_active is orthogonal to Strain Rate (dE), or dE is zero.")

if __name__ == "__main__":
    # Analyze the latest run
    
    
    #run both main and diagnose active on /home/dtsteene/D1/cardiac-work/results/sims/run_946891, /home/dtsteene/D1/cardiac-work/results/sims/run_946895, /home/dtsteene/D1/cardiac-work/results/sims/run_946897
    #/home/dtsteene/D1/cardiac-work/results/sims/run_946898
    for run_id in ["run_947252"]:
        results_path = f"/home/dtsteene/D1/cardiac-work/results/sims/{run_id}"
        main(results_path)
        diagnose_active(results_path)
    