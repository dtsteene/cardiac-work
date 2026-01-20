#!/usr/bin/env python3
"""
Post-Processing Script: Analyze True Work vs Clinical Proxy Results

This script loads the metrics saved by complete_cycle.py and generates
publication-quality comparison plots and statistics.

Usage:
  python3 analyze_metrics.py <results_dir>

Example:
  python3 analyze_metrics.py results_biv_complete_cycle_hybrid_75bpm
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

# Unit conversion: 1 mmHg*mL = 1.33322e-4 Joules
MMHG_ML_TO_J = 1.33322e-4


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
                    val_array = np.array(val)
                    # Downsample if history is significantly longer
                    if len(val_array) > target_len:
                         step = len(val_array) // target_len
                         step = max(1, step)
                         metrics[key] = val_array[::step][:target_len]
                    else:
                         metrics[key] = val_array[:target_len]
        except Exception as e:
            print(f"Warning: Could not load history.npy: {e}")

    return metrics



def extract_regional_data(metrics, regions=["LV", "RV", "Septum"]):
    """
    Extract work data for specified regions.

    Includes advanced proxy calculations for the Septum to compare:
    - Standard Proxy: (pLV+pRV)/2 * (dV_LV+dV_RV)
    - LV-focused Proxy: pLV * (dV_LV+dV_RV)
    - RV-focused Proxy: pRV * (dV_LV+dV_RV)
    - Weighted Blend: (alpha*pLV + beta*pRV) * (dV_LV+dV_RV)
    """
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
    # dV[i] = V[i] - V[i-1]. First element is 0.
    dV_LV = np.zeros(min_len_history)
    dV_RV = np.zeros(min_len_history)
    dV_LV[1:] = V_LV[1:min_len_history] - V_LV[0:min_len_history-1]
    dV_RV[1:] = V_RV[1:min_len_history] - V_RV[0:min_len_history-1]

    dV_Total = (dV_LV + dV_RV) / 2.0  # Septum volume change (split contribution)
    # Note: In metrics_calculator.py, proxy_Septum uses dV_total / 2.0
    # Let's align with the definition used: Work ~ P * dV

    # Process standard regions
    for region in regions:
        true_work = np.array(metrics.get(f"work_true_{region}", []))

        # Default proxy from file
        proxy_work = np.array(metrics.get(f"work_proxy_pv_{region}", []))

        # Align time with work data
        current_times = full_times
        min_len = min(len(true_work), len(proxy_work))

        # Truncate to match work length (sometimes Work has one less point if step 0 skipped)
        if len(true_work) > min_len: true_work = true_work[-min_len:]
        if len(proxy_work) > min_len: proxy_work = proxy_work[-min_len:]

        # Convert proxy work from mmHg*mL to Joules (True work is Joules)
        if len(proxy_work) > 0:
            proxy_work = proxy_work * MMHG_ML_TO_J


        # Determine time slice (usually aligned to end)
        time_slice = slice(-min_len, None) if min_len > 0 else slice(None)

        # Determine current times for this region - EXPLICIT TRUNCATION
        region_times = current_times[-min_len:] if min_len > 0 else []

        # Recalculate proxy if history is available (to fix potential zero-proxy bug in saved metrics)
        if region == "LV" and min_len > 0:
             p_val = p_LV[-min_len:]
             dV_val = dV_LV[-min_len:]
             if len(p_val) == len(proxy_work):
                 proxy_work = p_val * dV_val * MMHG_ML_TO_J
        elif region == "RV" and min_len > 0:
             p_val = p_RV[-min_len:]
             dV_val = dV_RV[-min_len:]
             if len(p_val) == len(proxy_work):
                 proxy_work = p_val * dV_val * MMHG_ML_TO_J

        # If this is the Septum, we generate extra comparisons
        # We need raw P and dV arrays matched to this time slice
        if region == "Septum" and min_len > 0:
            # Match lengths
            # CAUTION: The raw arrays (p_LV, dV_LV) might be slightly longer than truncated work arrays
            # We must truncate them to exactly match min_len

            p_L_val = p_LV[-min_len:]
            p_R_val = p_RV[-min_len:]
            dV_L_val = dV_LV[-min_len:]
            dV_R_val = dV_RV[-min_len:]

            # Septum dV is typically defined as the change in the total volume that it affects
            # Or geometrically, it moves between LV and RV.
            # MetricsCalculator uses: proxies["work_proxy_pv_Septum"] = p_avg * dV_total / 2.0

            dV_Septum = (dV_L_val + dV_R_val) / 2.0

            # 1. Standard Proxy (Already loaded as proxy_work): Average Pressure
            # proxy_std = (p_L_val + p_R_val) / 2.0 * dV_Septum

            # 2. LV Pressure Proxy
            proxy_lv = p_L_val * dV_Septum

            # 3. RV Pressure Proxy
            proxy_rv = p_R_val * dV_Septum

            # 4. Weighted Blend (e.g., 2/3 LV + 1/3 RV) - often Septum follows LV more
            proxy_blend = (0.67 * p_L_val + 0.33 * p_R_val) * dV_Septum

            # Convert to Joules
            proxy_lv *= MMHG_ML_TO_J
            proxy_rv *= MMHG_ML_TO_J
            proxy_blend *= MMHG_ML_TO_J

            # Store these variants for Septum
            data["Septum_LV_Proxy"] = {
                "true_work": true_work, # Septum true work
                "proxy_work": proxy_lv,
                "time": region_times
            }
            data["Septum_RV_Proxy"] = {
                "true_work": true_work,
                "proxy_work": proxy_rv,
                "time": region_times
            }
            data["Septum_Blend_Proxy"] = {
                "true_work": true_work,
                "proxy_work": proxy_blend,
                "time": region_times
            }


        data[region] = {
            "true_work": true_work,
            "proxy_work": proxy_work,
            "time": region_times if min_len > 0 else [],
        }

    return data


def compute_statistics(data, metrics=None):
    """Compute statistics for work comparison, including phase-windowed analysis."""
    stats = {}

    # Extract ejection phase mask if possible (V_LV decreasing)
    ejection_mask = None
    if metrics is not None:
        V_LV = np.array(metrics.get("V_LV", []))
        if len(V_LV) > 1:
            dV_LV = np.diff(V_LV, prepend=V_LV[0])  # Volume decrease during ejection
            ejection_mask = dV_LV < -1e-6  # Negative dV indicates ejection
            print(f"  Ejection phase detected: {np.sum(ejection_mask)} / {len(ejection_mask)} timesteps")

    for region, values in data.items():
        true_w = values["true_work"]
        proxy_w = values["proxy_work"]

        # Avoid division by zero
        proxy_w_safe = np.clip(np.abs(proxy_w), 1e-12, None) * np.sign(proxy_w + 1e-15)

        if len(true_w) > 0 and len(proxy_w) > 0:
            # Compute full-cycle metrics
            stats[region] = {
                "true_work_mean": np.mean(true_w),
                "true_work_max": np.max(true_w),
                "true_work_min": np.min(true_w),
                "proxy_work_mean": np.mean(proxy_w),
                "proxy_work_max": np.max(proxy_w),
                "proxy_work_min": np.min(proxy_w),
                "correlation": np.corrcoef(true_w, proxy_w)[0, 1] if len(true_w) > 1 else 0,
                "rmse": np.sqrt(np.mean((true_w - proxy_w)**2)),
            }

            # Compute ejection-phase metrics if mask available
            if ejection_mask is not None and len(ejection_mask) == len(true_w):
                ejection_true = true_w[ejection_mask]
                ejection_proxy = proxy_w[ejection_mask]

                if len(ejection_true) > 1:
                    stats[region]["correlation_ejection"] = (
                        np.corrcoef(ejection_true, ejection_proxy)[0, 1]
                    )
                    stats[region]["rmse_ejection"] = np.sqrt(
                        np.mean((ejection_true - ejection_proxy)**2)
                    )
                    stats[region]["num_ejection_points"] = int(np.sum(ejection_mask))
                else:
                    stats[region]["correlation_ejection"] = 0.0
                    stats[region]["rmse_ejection"] = 0.0
                    stats[region]["num_ejection_points"] = 0

    return stats


def plot_validation_boundary_work(data, metrics, output_file=None):
    """
    Create validation plot: Boundary Work vs Proxy Work.
    Boundary work should match proxy work if physics are correct.
    """
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
            print(f"✓ Saved boundary validation plot: {output_file}")
            print(f"  LV Validation Error: {error_lv:.2f}%")
            print(f"  RV Validation Error: {error_rv:.2f}%")
        else:
            plt.show()

        plt.close()
        return {"lv_error": error_lv, "rv_error": error_rv}
    else:
        print("⚠ Insufficient data for boundary work validation plot")
        return None


def plot_phase_windowed_analysis(data, metrics, output_file=None):
    """
    Create phase-windowed analysis plots: Full cycle vs Ejection phase.
    Compare correlation and RMSE between global and ejection-only.
    """
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

    # Align ejection mask to work arrays (they might be different lengths)
    min_len = min(len(ejection_mask), len(true_w))
    if min_len > 0:
        ejection_mask = ejection_mask[-min_len:]
        true_w = true_w[-min_len:]
        proxy_w = proxy_w[-min_len:]
        times = times[-min_len:]

    # Split into ejection and non-ejection phases
    ejection_true = true_w[ejection_mask]
    ejection_proxy = proxy_w[ejection_mask]
    non_ejection_true = true_w[~ejection_mask]
    non_ejection_proxy = proxy_w[~ejection_mask]

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

Non-Ejection Phase:
  • N points:    {len(non_ejection_true):>8d}

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
        print(f"✓ Saved phase-windowed analysis plot: {output_file}")
    else:
        plt.show()

    plt.close()

    return {
        "global_correlation": float(global_corr),
        "global_rmse": float(global_rmse),
        "ejection_correlation": float(ejection_corr),
        "ejection_rmse": float(ejection_rmse),
        "num_ejection_points": int(np.sum(ejection_mask)),
        "num_non_ejection_points": int(np.sum(~ejection_mask)),
    }


def plot_comparison(data, output_file=None):
    """Create comprehensive comparison plots."""

    regions = list(data.keys())
    # Sort regions to keep main ones first
    ordered_regions = ["LV", "RV", "Septum"]
    other_regions = [r for r in regions if r not in ordered_regions]
    plot_regions = ordered_regions + sorted(other_regions)

    # Filter out empty data
    plot_regions = [r for r in plot_regions if r in data and len(data[r]["true_work"]) > 0]

    # Calculate grid size roughly
    n_plots = len(plot_regions)
    cols = 3
    rows = (n_plots + cols - 1) // cols

    # Re-setup figure based on dynamic number of regions
    fig = plt.figure(figsize=(5*cols, 4*rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

    colors = {
        "LV": "blue", "RV": "red", "Septum": "green",
        "Septum_LV_Proxy": "purple", "Septum_RV_Proxy": "orange", "Septum_Blend_Proxy": "teal"
    }

    # === Comparison Plots (Scatter + Regression) for all variants ===
    for idx, region in enumerate(plot_regions):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])

        true_w = np.array(data[region]["true_work"]).ravel()
        proxy_w = np.array(data[region]["proxy_work"]).ravel()

        # Ensure lengths match for plotting (safety check)
        min_p_len = min(len(true_w), len(proxy_w))
        if min_p_len > 0:
            true_w = true_w[:min_p_len]
            proxy_w = proxy_w[:min_p_len]
        else:
            print(f"Warning: Empty data for {region}")
            continue

        color = colors.get(region, "black")

        # Plot Scatter
        ax.scatter(proxy_w, true_w, alpha=0.5, s=20, color=color, label="Data")

        # Calculate stats for title
        if len(true_w) > 1:
            corr = np.corrcoef(true_w, proxy_w)[0, 1]
            rmse = np.sqrt(np.mean((true_w - proxy_w)**2))

            # Regression Line
            mask = ~(np.isnan(proxy_w) | np.isnan(true_w))
            if np.sum(mask) > 2 and np.std(proxy_w[mask]) > 1e-9:
                z = np.polyfit(proxy_w[mask], true_w[mask], 1)
                p = np.poly1d(z)
                px = np.linspace(np.min(proxy_w[mask]), np.max(proxy_w[mask]), 100)
                ax.plot(px, p(px), "k--", alpha=0.7, linewidth=1.5, label=f"R={corr:.3f}")

        ax.set_xlabel("Proxy Work (P·V)")
        ax.set_ylabel("True Work (S·E)")
        ax.set_title(f"{region} Comparison", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2)

        # Add text box with correlation
        # ax.text(0.05, 0.95, f"R = {corr:.3f}", transform=ax.transAxes, verticalalignment='top')

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Saved comparison plot: {output_file}")
    else:
        plt.show()

    plt.close()


def print_statistics(stats):
    """Print formatted statistics including phase-windowed metrics."""
    print("\n" + "="*80)
    print("WORK METRICS STATISTICS")
    print("="*80)

    for region, values in stats.items():
        print(f"\n{region}:")
        print(f"  True Work:  mean={values['true_work_mean']:>12.4e}, max={values['true_work_max']:>12.4e}, min={values['true_work_min']:>12.4e}")
        print(f"  Proxy Work: mean={values['proxy_work_mean']:>12.4e}, max={values['proxy_work_max']:>12.4e}, min={values['proxy_work_min']:>12.4e}")
        print(f"  Correlation (Global): {values['correlation']:>6.3f}")
        
        # Print ejection phase stats if available
        if "correlation_ejection" in values:
            print(f"  Correlation (Ejection): {values['correlation_ejection']:>6.3f}")
            print(f"  RMSE (Global): {values['rmse']:>12.4e}")
            print(f"  RMSE (Ejection): {values['rmse_ejection']:>12.4e}")
            print(f"  Ejection points: {values.get('num_ejection_points', 0):>6d}")
        else:
            print(f"  RMSE: {values['rmse']:>12.4e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_metrics.py <results_dir> [downsample_factor]")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    downsample_factor = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Load metrics
    print(f"Loading metrics from {results_dir} (downsample_factor={downsample_factor})...")
    metrics = load_metrics(results_dir, downsample_factor)

    if metrics is None:
        sys.exit(1)

    # Extract regional data
    regions = ["LV", "RV", "Septum"]
    data = extract_regional_data(metrics, regions=regions)

    # Compute statistics (pass metrics for phase-windowed analysis)
    stats = compute_statistics(data, metrics=metrics)
    print_statistics(stats)

    # Create comparison plots
    plot_file = results_dir / f"work_comparison_downsample_{downsample_factor}.png"
    plot_comparison(data, output_file=str(plot_file))

    # NEW: Create boundary work validation plot (if data available)
    print("\n" + "="*80)
    print("BOUNDARY WORK VALIDATION")
    print("="*80)
    boundary_validation = None
    if "work_boundary_LV" in metrics:
        boundary_plot_file = results_dir / f"boundary_work_validation_downsample_{downsample_factor}.png"
        boundary_validation = plot_validation_boundary_work(data, metrics, output_file=str(boundary_plot_file))
    else:
        print("⚠ No boundary work data found (geometry may lack facet tags)")

    # NEW: Create phase-windowed analysis plot
    print("\n" + "="*80)
    print("PHASE-WINDOWED ANALYSIS")
    print("="*80)
    phase_analysis = None
    if "V_LV" in metrics:
        phase_plot_file = results_dir / f"phase_windowed_analysis_downsample_{downsample_factor}.png"
        phase_analysis = plot_phase_windowed_analysis(data, metrics, output_file=str(phase_plot_file))
    else:
        print("⚠ No volume data for phase analysis")

    # Save statistics to JSON (including phase-windowed)
    stats_json = {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                      for kk, vv in v.items()} for k, v in stats.items()}
    
    # Add phase analysis results
    if phase_analysis:
        stats_json["phase_analysis_LV"] = phase_analysis
    if boundary_validation:
        stats_json["boundary_validation"] = boundary_validation

    stats_file = results_dir / f"work_statistics_downsample_{downsample_factor}.json"
    with open(stats_file, "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"\n✓ Saved statistics: {stats_file}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
