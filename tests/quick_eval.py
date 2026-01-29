import numpy as np
import pathlib

def find_peaks(x):
    """Simple peak finding for clean simulation data."""
    # Find local maxima
    peaks = []
    # Use a window or simple comparison
    for i in range(1, len(x) - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(i)
    
    # Filter very close peaks (e.g. noise)
    # Assuming cycle > 0.5s and dt is small. 
    # With 4000 steps for 4s, dt=0.001. 500 steps is 0.5s.
    filtered_peaks = []
    if peaks:
        filtered_peaks.append(peaks[0])
        for p in peaks[1:]:
            if p - filtered_peaks[-1] > 200: # 0.2s buffer
                filtered_peaks.append(p)
    
    return filtered_peaks

def main():
    # 1. Load Data
    path = pathlib.Path('/home/dtsteene/D1/cardiac-work/results/sims/run_947565/metrics_downsample_1.npy')
    if not path.exists():
        print(f"Error: File {path} not found.")
        return

    try:
        data = np.load(path, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Extract required arrays
    time = np.array(data['time'])
    V_LV = np.array(data['V_LV'])
    p_LV = np.array(data['p_LV'])
    work_septum = np.array(data['work_true_Septum'])
    work_lv = np.array(data['work_true_LV'])
    stress_septum = np.array(data['mean_S_ff_Septum'])

    # Validate lengths
    min_len = min(len(time), len(V_LV), len(p_LV), len(work_septum), len(work_lv), len(stress_septum))
    if len(time) != min_len:
        # print(f"Warning: Array length mismatch. Time: {len(time)}, Work: {len(work_septum)}. Truncating to {min_len}.")
        time = time[:min_len]
        V_LV = V_LV[:min_len]
        p_LV = p_LV[:min_len]
        work_septum = work_septum[:min_len]
        work_lv = work_lv[:min_len]
        stress_septum = stress_septum[:min_len]


    # 2. Identify Last Beat
    # We assume the beat starts at End Diastole (Max Volume)
    # Note: t=0 is usually ED.
    
    peaks = find_peaks(V_LV)
    
    # If t=0 is a peak (starts at max volume), include index 0
    if V_LV[0] > V_LV[1]:
        peaks.insert(0, 0)

    # Filter peaks to only keep prominent ones (roughly cycle length apart)
    # In 4s simulation, expect ~5 peaks.
    real_peaks = []
    if len(peaks) > 0:
        real_peaks.append(peaks[0])
        for p in peaks[1:]:
            if time[p] - time[real_peaks[-1]] > 0.5:
                real_peaks.append(p)

    if not real_peaks:
        print("Could not identify cardiac cycles. Using full duration.")
        start_idx = 0
        end_idx = len(time) - 1
    else:
        # Last beat starts at the last ED peak
        start_idx = real_peaks[-1]
        end_idx = len(time) - 1
    
    # Check if we have enough data for a "last beat" (at least 0.1s?)
    if time[end_idx] - time[start_idx] < 0.1:
         # Maybe the last peak is the very end? Use the previous peak
         if len(real_peaks) >= 2:
             start_idx = real_peaks[-2]
             end_idx = real_peaks[-1]
         else:
             print("Warning: Last beat detection ambiguous.")

    dt_last_beat = time[end_idx] - time[start_idx]
    
    # 3. Calculate Work Totals (Joules)
    # The 'work_*' arrays appear to be incremental work (dWork) or Power*dt,
    # given their small magnitude and length (N-1).
    # We sum them to get total work over the interval.
    
    # Adjust slicing for arrays of length N-1
    # work[i] corresponds to interval i (time[i] to time[i+1])?
    # We just sum the subset corresponding to the beat.
    
    w_start = start_idx
    w_end = end_idx
    if w_end > len(work_septum):
        w_end = len(work_septum)
        
    septum_work_j = np.sum(work_septum[w_start:w_end])
    lv_work_j = np.sum(work_lv[w_start:w_end])

    # 4. Peak Stress at t=0
    # User asked for mean_S_ff_Septum at t=0
    # Assuming index 0 is t=0
    stress_septum_t0 = stress_septum[0]
    
    # 5. Physiological Checks
    # Metrics over the last beat
    beat_V_LV = V_LV[start_idx:end_idx+1]
    beat_p_LV = p_LV[start_idx:end_idx+1]
    
    min_v_lv = np.min(beat_V_LV)
    max_v_lv = np.max(beat_V_LV)
    max_p_lv = np.max(beat_p_LV)
    
    # EF = (EDV - ESV) / EDV
    # EDV is usually the start of the beat (Max V)
    edv = max_v_lv
    esv = min_v_lv
    ef = (edv - esv) / edv * 100

    # 6. Pass/Fail Logic
    pass_septum_work = septum_work_j < 10.0
    pass_stress_t0 = stress_septum_t0 > -200.0
    pass_ef = 40.0 <= ef <= 70.0

    # 7. Print Report
    print("-" * 40)
    print("QUICK EVALUATION REPORT: run_947565")
    print("-" * 40)
    print(f"Data Interval: {time[start_idx]:.2f}s - {time[end_idx]:.2f}s (Duration: {dt_last_beat:.2f}s)")
    print("-" * 40)
    
    print(f"1. Work Totals (Last Beat)")
    print(f"   LV Work:     {lv_work_j:.4f} J")
    print(f"   Septum Work: {septum_work_j:.4f} J")
    print(f"   -> Check:    {'PASS' if pass_septum_work else 'FAIL'} (< 10.0 J)")
    
    print(f"\n2. Peak Stress Analysis (t=0)")
    print(f"   Septum Stress: {stress_septum_t0:.2f} kPa")
    print(f"   -> Check:      {'PASS' if pass_stress_t0 else 'FAIL'} (> -200 kPa)")

    print(f"\n3. Physiological Check")
    print(f"   Min LV Volume: {min_v_lv:.1f} mL")
    print(f"   Max LV Volume: {max_v_lv:.1f} mL")
    print(f"   Max LV Pressure: {max_p_lv:.1f} mmHg")
    print(f"   EF:            {ef:.1f} %")
    print(f"   -> Check:      {'PASS' if pass_ef else 'FAIL'} (40-70%)")
    
    print("-" * 40)
    
    if pass_septum_work and pass_stress_t0 and pass_ef:
        print("OVERALL STATUS: PASS")
    else:
        print("OVERALL STATUS: FAIL")
    print("-" * 40)

if __name__ == "__main__":
    main()
