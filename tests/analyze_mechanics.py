import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants from simulation.log
VOL_LV_WALL = 5.83e-05 # m3
VOL_RV_WALL = 4.29e-05 # m3
VOL_SEPTUM =  3.32e-05 # m3
VOL_WHOLE = VOL_LV_WALL + VOL_RV_WALL + VOL_SEPTUM

# Conversion factors
MMHG_TO_PA = 133.322
ML_TO_M3 = 1e-6

def analyze_mechanics(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Inspect Columns
    print("\n--- Columns in CSV ---")
    for col in df.columns:
        print(f"  {col}")
        
    # 2. Pre-process Units
    # Pressure: mmHg -> Pa
    df['p_LV_Pa'] = df['p_LV'] * MMHG_TO_PA
    df['p_RV_Pa'] = df['p_RV'] * MMHG_TO_PA
    
    # Volume: mL -> m3
    df['V_LV_m3'] = df['V_LV'] * ML_TO_M3
    df['V_RV_m3'] = df['V_RV'] * ML_TO_M3
    
    # Calculate differentials (Delta)
    # We use difference between current and previous step
    # fillna(0) for the first step
    df['dV_LV'] = df['V_LV_m3'].diff().fillna(0)
    df['dV_RV'] = df['V_RV_m3'].diff().fillna(0)
    
    # Strains (dimensionless)
    df['dE_ff_LV'] = df['mean_E_ff_LV'].diff().fillna(0)
    df['dE_ff_RV'] = df['mean_E_ff_RV'].diff().fillna(0)
    df['dE_ff_Septum'] = df['mean_E_ff_Septum'].diff().fillna(0)
    df['dE_ff_Whole'] = df['mean_E_ff_Whole'].diff().fillna(0) # Approximate
    
    # 3. Calculate Work Terms (Cumulative Sums)
    
    # --- A. External Work (PV Work) ---
    # Work done BY the fluid = Integral(P dV)
    # Usually Ejection is Work Output.
    # Total External Work over cycle = Loop Area.
    # W_ext = Sum(P * dV)
    # If dV is positive (filling), Work is positive (fluid does work on wall? or wall does work on fluid?)
    # Thermodynamic work done BY system (Ventricle) on surroundings (Blood):
    # dW = P_internal * dV_chamber. 
    # But V_chamber decreases during systole (dV < 0). So P*dV is negative.
    # So Work Done By Heart = - Sum(P * dV).
    
    df['W_ext_LV_rate'] = - df['p_LV_Pa'] * df['dV_LV']
    df['W_ext_RV_rate'] = - df['p_RV_Pa'] * df['dV_RV']
    
    df['W_ext_LV_cum'] = df['W_ext_LV_rate'].cumsum()
    df['W_ext_RV_cum'] = df['W_ext_RV_rate'].cumsum()
    
    # --- B. Internal Work (Fiber Stress-Strain Work) ---
    # Work done by the fibers = Integral(Stress * dStrain) * Volume
    # W_int = Sum(S * dE) * V_wall
    # Note: S_ff and dE_ff.
    
    # LV Free Wall
    df['W_fiber_LV_rate'] = df['mean_S_ff_LV'] * df['dE_ff_LV'] * VOL_LV_WALL
    df['W_fiber_LV_cum'] = df['W_fiber_LV_rate'].cumsum()
    
    # RV Free Wall
    df['W_fiber_RV_rate'] = df['mean_S_ff_RV'] * df['dE_ff_RV'] * VOL_RV_WALL
    df['W_fiber_RV_cum'] = df['W_fiber_RV_rate'].cumsum()
    
    # Septum
    df['W_fiber_Septum_rate'] = df['mean_S_ff_Septum'] * df['dE_ff_Septum'] * VOL_SEPTUM
    df['W_fiber_Septum_cum'] = df['W_fiber_Septum_rate'].cumsum()
    
    # Total Fiber Work
    df['W_fiber_Total_cum'] = df['W_fiber_LV_cum'] + df['W_fiber_RV_cum'] + df['W_fiber_Septum_cum']

    # --- C. Active Work Proxy ---
    # Using Active Stress * dStrain
    # S_active is usually positive.
    # During shortening, dE is negative.
    # So Active Work done BY fiber = - S_active * dE ?
    # Or is S_active the tension?
    # If fiber shortens (dE < 0) and pulls (S > 0), it does positive work?
    # Mechanical Work Output: Force * Distance.
    # If muscle contracts (shortens) against load, it does work.
    # So if dE < 0 and S > 0, Work should be positive.
    # So W_active = - S_active * dE * Volume
    
    df['W_active_LV_rate'] = - df['mean_S_active_LV'] * df['dE_ff_LV'] * VOL_LV_WALL
    df['W_active_LV_cum'] = df['W_active_LV_rate'].cumsum()
    
    df['W_active_RV_rate'] = - df['mean_S_active_RV'] * df['dE_ff_RV'] * VOL_RV_WALL
    df['W_active_RV_cum'] = df['W_active_RV_rate'].cumsum()
    
    df['W_active_Septum_rate'] = - df['mean_S_active_Septum'] * df['dE_ff_Septum'] * VOL_SEPTUM
    df['W_active_Septum_cum'] = df['W_active_Septum_rate'].cumsum()
    
    df['W_active_Total_cum'] = df['W_active_LV_cum'] + df['W_active_RV_cum'] + df['W_active_Septum_cum']

    
    # 4. Print Summary at the end
    print("\n--- Energy Accounting (End of Simulation) ---")
    print(f"Time: {df['time'].iloc[-1]:.3f} s")
    
    print("\nExternal Work (PV Work) [Joules]:")
    print(f"  LV: {df['W_ext_LV_cum'].iloc[-1]:.4f}")
    print(f"  RV: {df['W_ext_RV_cum'].iloc[-1]:.4f}")
    print(f"  Total Ext: {df['W_ext_LV_cum'].iloc[-1] + df['W_ext_RV_cum'].iloc[-1]:.4f}")
    
    print("\nFiber Work (Stress-Strain Area * Volume) [Joules]:")
    # This includes passive + active elastic energy + active work
    print(f"  LV: {df['W_fiber_LV_cum'].iloc[-1]:.4f}")
    print(f"  RV: {df['W_fiber_RV_cum'].iloc[-1]:.4f}")
    print(f"  Septum: {df['W_fiber_Septum_cum'].iloc[-1]:.4f}")
    print(f"  Total Fiber: {df['W_fiber_Total_cum'].iloc[-1]:.4f}")
    
    print("\nActive Work Proxy (- S_active * dE * Vol) [Joules]:")
    print(f"  LV: {df['W_active_LV_cum'].iloc[-1]:.4f}")
    print(f"  RV: {df['W_active_RV_cum'].iloc[-1]:.4f}")
    print(f"  Septum: {df['W_active_Septum_cum'].iloc[-1]:.4f}")
    print(f"  Total Active: {df['W_active_Total_cum'].iloc[-1]:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['W_ext_LV_cum'], label='Ext Work LV (PV)')
    plt.plot(df['time'], df['W_fiber_Total_cum'], label='Total Fiber Work (S*dE)')
    plt.plot(df['time'], df['W_active_Total_cum'], label='Total Active Work Proxy (-Sa*dE)')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Work (J)')
    plt.title('Energy Accounting Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('energy_accounting.png')
    print("\nSaved plot to energy_accounting.png")

if __name__ == "__main__":
    csv_path = "/home/dtsteene/D1/cardiac-work/results/sims/run_947321/active_mechanics_trace.csv"
    analyze_mechanics(csv_path)
