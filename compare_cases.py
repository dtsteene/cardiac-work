import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(folder):
    path = Path(folder)
    # Find metrics file (prefer downsample_1)
    files = list(path.glob("metrics_downsample_*.npy"))
    if not files: return None
    files.sort(key=lambda p: len(str(p))) 
    return np.load(files[0], allow_pickle=True).item()

def get_total_work(metrics, key):
    # Sum Incremental Work (Joules) -> Total Work (Joules)
    if key not in metrics: return 0.0
    d_work = np.array(metrics[key])
    
    # Robustness: Truncate to match if needed (though sum doesn't need time)
    return np.sum(d_work)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compare_cases.py <HEALTHY_RESULT_DIR> <PAH_RESULT_DIR>")
        sys.exit(1)
        
    folders = [sys.argv[1], sys.argv[2]]
    labels = ["Healthy", "PAH"]
    
    # Data Storage
    true_work = []
    proxy_lv = []    # Standard Method (PLV * Strain)
    proxy_trans = [] # Partner Method ((PLV-PRV) * Strain)
    rv_peaks = []
    
    print(f"{'Case':<10} | {'RV Peak':<10} | {'True Work':<12} | {'LV Proxy':<12} | {'Trans Proxy':<12}")
    print("-" * 75)
    
    for i, f in enumerate(folders):
        m = load_metrics(f)
        if m is None:
            print(f"Skipping {f} (No data)")
            continue
            
        # Get RV Peak Pressure for validation
        rv_p = np.max(m['p_RV'])
        
        # Calculate Total Work (Joules)
        w_true = get_total_work(m, 'work_true_Septum')
        w_lv = get_total_work(m, 'work_ps_index_Septum_PLV')
        w_trans = get_total_work(m, 'work_ps_index_Septum_Trans')
        
        true_work.append(w_true)
        proxy_lv.append(w_lv)
        proxy_trans.append(w_trans)
        rv_peaks.append(rv_p)
        
        print(f"{labels[i]:<10} | {rv_p:>6.1f} mmHg | {w_true:>12.2e} | {w_lv:>12.2e} | {w_trans:>12.2e}")

    # --- NORMALIZATION ---
    # Normalize to the first case (Healthy = 1.0)
    base_true = true_work[0]
    base_lv = proxy_lv[0]
    base_trans = proxy_trans[0]
    
    # base_true = 1
    # base_lv = 1
    # base_trans = 1
    
    
    true_norm = [x / base_true for x in true_work]
    lv_norm = [x / base_lv for x in proxy_lv]
    trans_norm = [x / base_trans for x in proxy_trans]

    # --- THE PLOT ---
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Identity Line (Target)
    ax.plot([0.5, 1.1], [0.5, 1.1], 'k--', alpha=0.3, label="Ideal Accuracy (y=x)")
    
    # 2. Standard Method (Blue)
    ax.plot(lv_norm, true_norm, 'o-', color='tab:blue', lw=2, markersize=10, label="Standard (LV Only)")
    
    # 3. Partner Method (Green)
    ax.plot(trans_norm, true_norm, 's-', color='tab:green', lw=2, markersize=10, label="Partner (Transmural)")
    
    # Annotations
    for i, txt in enumerate(labels):
        # Standard
        ax.annotate(f"{txt}", (lv_norm[i], true_norm[i]), 
                   xytext=(-10, 10), textcoords='offset points', color='tab:blue', fontsize=9, ha='right')
                   
        # Partner
        ax.annotate(f"{txt}", (trans_norm[i], true_norm[i]), 
                   xytext=(10, -15), textcoords='offset points', color='tab:green', fontsize=9, ha='left')

    ax.set_title("Normalized Validation: Sensitivity to Disease", fontsize=14, fontweight='bold')
    ax.set_xlabel("Clinical Work Index (Internal Normalized)", fontsize=12)
    ax.set_ylabel("True Mechanical Work (Normalized)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("proxy_sensitivity_comparison.png", dpi=150)
    print("\n Plot saved: proxy_sensitivity_comparison.png")

if __name__ == "__main__":
    main()
