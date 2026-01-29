import pandas as pd
import numpy as np
import glob
import os
import re

# Constants
VOL_LV_WALL = 5.83e-05 # m3
VOL_RV_WALL = 4.29e-05 # m3
VOL_SEPTUM =  3.32e-05 # m3
MMHG_TO_PA = 133.322
ML_TO_M3 = 1e-6

def get_run_id(path):
    match = re.search(r'run_(\d+)', path)
    if match:
        return int(match.group(1))
    return 0

def analyze_file(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        
        # Pre-process Units
        df['p_LV_Pa'] = df['p_LV'] * MMHG_TO_PA
        df['p_RV_Pa'] = df['p_RV'] * MMHG_TO_PA
        df['V_LV_m3'] = df['V_LV'] * ML_TO_M3
        df['V_RV_m3'] = df['V_RV'] * ML_TO_M3
        
        # Diffs
        df['dV_LV'] = df['V_LV_m3'].diff().fillna(0)
        df['dV_RV'] = df['V_RV_m3'].diff().fillna(0)
        df['dE_ff_LV'] = df['mean_E_ff_LV'].diff().fillna(0)
        df['dE_ff_RV'] = df['mean_E_ff_RV'].diff().fillna(0)
        df['dE_ff_Septum'] = df['mean_E_ff_Septum'].diff().fillna(0)
        
        # Work Rates
        W_ext_LV_rate = - df['p_LV_Pa'] * df['dV_LV']
        W_ext_RV_rate = - df['p_RV_Pa'] * df['dV_RV']
        
        W_fiber_LV_rate = df['mean_S_ff_LV'] * df['dE_ff_LV'] * VOL_LV_WALL
        W_fiber_RV_rate = df['mean_S_ff_RV'] * df['dE_ff_RV'] * VOL_RV_WALL
        W_fiber_Septum_rate = df['mean_S_ff_Septum'] * df['dE_ff_Septum'] * VOL_SEPTUM
        
        W_active_LV_rate = - df['mean_S_active_LV'] * df['dE_ff_LV'] * VOL_LV_WALL
        W_active_RV_rate = - df['mean_S_active_RV'] * df['dE_ff_RV'] * VOL_RV_WALL
        W_active_Septum_rate = - df['mean_S_active_Septum'] * df['dE_ff_Septum'] * VOL_SEPTUM
        
        # Totals (Cumsum at end)
        w_ext = (W_ext_LV_rate.sum() + W_ext_RV_rate.sum())
        w_fiber = (W_fiber_LV_rate.sum() + W_fiber_RV_rate.sum() + W_fiber_Septum_rate.sum())
        w_active = (W_active_LV_rate.sum() + W_active_RV_rate.sum() + W_active_Septum_rate.sum())
        
        time_final = df['time'].iloc[-1]
        
        return {
            'time': time_final,
            'W_ext': w_ext,
            'W_fiber': w_fiber,
            'W_active': w_active
        }
    except Exception as e:
        # print(f"Error processing {csv_path}: {e}")
        return None

def main():
    root_dir = "/home/dtsteene/D1/cardiac-work/results/sims"
    files = glob.glob(os.path.join(root_dir, "run_*", "active_mechanics_trace.csv"))
    
    files.sort(key=get_run_id)
    
    results = []
    print(f"{'Run ID':<10} {'Time(s)':<8} {'W_ext (J)':<12} {'W_active (J)':<12} {'W_fiber (J)':<12} {'Balance (A-E)':<12}")
    print("-" * 70)
    
    for f in files:
        run_id = get_run_id(f)
        data = analyze_file(f)
        if data:
            balance = data['W_active'] - data['W_ext']
            print(f"{run_id:<10} {data['time']:<8.3f} {data['W_ext']:<12.4f} {data['W_active']:<12.4f} {data['W_fiber']:<12.4f} {balance:<12.4f}")

if __name__ == "__main__":
    main()
