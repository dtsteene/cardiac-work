"""
Beat-to-beat convergence analysis for cardiac simulations.

Quantifies how close the last beat is to steady state by:
1. PV loop overlap between consecutive beats (L2 norm)
2. Beat-to-beat change in key hemodynamic metrics
3. Richardson extrapolation to estimate converged values
4. RC time constants of the Windkessel model

Usage:
    python analyze_convergence.py /path/to/results/sims/2026-03-20
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# ── Configuration ────────────────────────────────────────────
CIRC_DIR = Path(__file__).parent / "data" / "ukb_circ"

# Map job folders to case names and circ param files
CASES = {
    "healthy":         ("optimized_regazzoni_ukb_healthy.json", None),
    "healthy_low":     ("optimized_regazzoni_ukb_healthy_low.json", None),
    "mild":            ("optimized_regazzoni_ukb_mild.json", None),
    "moderate":        ("optimized_regazzoni_ukb_moderate.json", None),
    "moderate_severe": ("optimized_regazzoni_ukb_moderate_severe.json", None),
    "severe":          ("optimized_regazzoni_ukb_severe.json", None),
}


def find_case_folders(results_dir):
    """Auto-detect case folders by matching run descriptions."""
    results_dir = Path(results_dir)
    case_folders = {}
    # Sort case names longest first so "moderate_severe" matches before "moderate"
    sorted_cases = sorted(CASES.keys(), key=len, reverse=True)
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        desc_file = d / "run_description.txt"
        if desc_file.exists():
            desc = desc_file.read_text().strip().lower()
            # Extract the param name from description like "...ukb_circ healthy_low params"
            for case_name in sorted_cases:
                # Match exact token: "healthy" should not match "healthy_low"
                # Look for "case_name params" or " case_name " pattern
                pattern_underscore = f"{case_name} params"
                pattern_space = f"{case_name.replace('_', ' ')} params"
                if pattern_underscore in desc or pattern_space in desc:
                    case_folders[case_name] = d
                    break  # first (longest) match wins
    return case_folders


def load_case_data(folder):
    """Load metrics and circulation data for a case."""
    folder = Path(folder)
    m = np.load(folder / "metrics" / "metrics_downsample_1.npy", allow_pickle=True).item()
    circ = np.load(folder / "circulation" / "history.npy", allow_pickle=True).item()
    with open(folder / "simulation_params.json") as f:
        sp = json.load(f)
    return m, circ, sp


def compute_convergence_metrics(m, circ, sp):
    """Compute convergence metrics for a simulation."""
    bpm = sp["BPM"]
    T_cycle = 60.0 / bpm
    dt = m["time"][1] - m["time"][0]
    npts = int(round(T_cycle / dt))
    nbeats = int(len(m["time"]) / npts)

    results = {"nbeats": nbeats, "T_cycle": T_cycle, "npts_per_beat": npts}

    # ── 1. Per-beat hemodynamics ──
    hemo = {}
    for key, data_key in [
        ("EDV_LV", "V_LV_FEM"), ("EDV_RV", "V_RV_FEM"),
        ("ESV_LV", "V_LV_FEM"), ("ESV_RV", "V_RV_FEM"),
        ("peak_pLV", "p_LV"), ("peak_pRV", "p_RV"),
        ("min_pLV", "p_LV"), ("min_pRV", "p_RV"),
    ]:
        if data_key not in m:
            continue
        v = np.array(m[data_key])
        beat_vals = []
        for b in range(nbeats):
            s, e = b * npts, min((b + 1) * npts, len(v))
            beat = v[s:e]
            if "EDV" in key:
                beat_vals.append(np.max(beat))
            elif "ESV" in key:
                beat_vals.append(np.min(beat))
            elif "peak" in key:
                beat_vals.append(np.max(beat))
            elif "min" in key:
                beat_vals.append(np.min(beat))
        hemo[key] = np.array(beat_vals)

    # Stroke volumes
    if "EDV_LV" in hemo and "ESV_LV" in hemo:
        hemo["SV_LV"] = hemo["EDV_LV"] - hemo["ESV_LV"]
    if "EDV_RV" in hemo and "ESV_RV" in hemo:
        hemo["SV_RV"] = hemo["EDV_RV"] - hemo["ESV_RV"]

    results["hemodynamics"] = hemo

    # ── 2. Per-beat work ──
    work = {}
    for wkey in ["work_true_LV", "work_true_RV", "work_true_Septum",
                 "work_ff_LV", "work_ff_RV", "work_ff_Septum"]:
        if wkey not in m:
            continue
        w = np.array(m[wkey])
        beat_work = []
        for b in range(nbeats):
            s, e = b * npts, min((b + 1) * npts, len(w))
            beat_work.append(w[e - 1] - w[s])
        work[wkey] = np.array(beat_work)
    results["work"] = work

    # ── 3. PV loop L2 distance (normalized) ──
    pv_l2 = {}
    for chamber in ["LV", "RV"]:
        vk = f"V_{chamber}_FEM"
        pk = f"p_{chamber}"
        if vk not in m or pk not in m:
            continue
        v = np.array(m[vk])
        p = np.array(m[pk])

        beats_v, beats_p = [], []
        for b in range(nbeats):
            s, e = b * npts, min((b + 1) * npts, len(v))
            beats_v.append(v[s:e])
            beats_p.append(p[s:e])

        # Normalize by last beat range
        v_scale = np.ptp(beats_v[-1]) or 1.0
        p_scale = np.ptp(beats_p[-1]) or 1.0

        l2s = []
        for b in range(nbeats - 1):
            n = min(len(beats_v[b]), len(beats_v[b + 1]))
            l2 = np.sqrt(np.mean(
                ((beats_v[b][:n] - beats_v[b + 1][:n]) / v_scale) ** 2 +
                ((beats_p[b][:n] - beats_p[b + 1][:n]) / p_scale) ** 2
            ))
            l2s.append(l2)
        pv_l2[chamber] = np.array(l2s)
    results["pv_l2"] = pv_l2

    # ── 4. 0D state convergence ──
    state_conv = {}
    for k in ["V_LA", "V_LV", "V_RA", "V_RV", "p_AR_SYS", "p_VEN_SYS", "p_AR_PUL", "p_VEN_PUL"]:
        if k not in circ:
            continue
        v = np.array(circ[k])
        end_vals = []
        for b in range(nbeats):
            idx = min((b + 1) * npts - 1, len(v) - 1)
            end_vals.append(v[idx])
        state_conv[k] = np.array(end_vals)
    results["state_convergence"] = state_conv

    return results


def richardson_extrapolation(beat_values):
    """Estimate converged value using Aitken/Richardson extrapolation.

    Given beat values w1, w2, w3 assumed to follow w_n = w_inf + C*r^n,
    estimates w_inf. Returns (w_extrapolated, error_last_beat_pct).
    """
    if len(beat_values) < 3:
        return beat_values[-1], 0.0

    w1, w2, w3 = beat_values[-3], beat_values[-2], beat_values[-1]
    denom = w1 - 2 * w2 + w3

    if abs(denom) < 1e-15:
        return w3, 0.0

    w_inf = w3 - (w3 - w2) ** 2 / denom
    err = abs(w3 - w_inf) / abs(w_inf) * 100 if abs(w_inf) > 1e-15 else 0.0
    return w_inf, min(err, 999.0)  # cap at 999%


def compute_time_constants(circ_json_path, T_cycle):
    """Compute RC time constants from circulation parameters."""
    with open(circ_json_path) as f:
        data = json.load(f)
    params = data.get("parameters", data)
    circ = params.get("circulation", {})

    taus = {}
    for sys_name, sys_params in circ.items():
        if sys_name == "external":
            continue
        R_AR = sys_params.get("R_AR", 0)
        C_AR = sys_params.get("C_AR", 0)
        R_VEN = sys_params.get("R_VEN", 0)
        C_VEN = sys_params.get("C_VEN", 0)
        taus[f"tau_AR_{sys_name}"] = R_AR * C_AR / T_cycle
        taus[f"tau_VEN_{sys_name}"] = R_VEN * C_VEN / T_cycle
    taus["max_tau"] = max(taus.values()) if taus else 0
    return taus


def plot_convergence(all_results, outdir):
    """Create convergence visualization."""
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    case_names = list(all_results.keys())
    n_cases = len(case_names)

    # ── Figure 1: PV loop convergence + hemodynamics ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Beat-to-Beat Convergence Analysis", fontsize=14, fontweight="bold")

    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_cases))

    # Row 1: PV loop L2 norms
    for i, chamber in enumerate(["LV", "RV"]):
        ax = axes[0, i]
        for j, name in enumerate(case_names):
            r = all_results[name]
            l2 = r["convergence"]["pv_l2"].get(chamber, [])
            if len(l2) > 0:
                beats = np.arange(1, len(l2) + 1)
                ax.semilogy(beats, l2, "o-", color=colors[j], label=name, linewidth=2, markersize=6)
        ax.axhline(0.01, color="green", linestyle="--", alpha=0.5, label="Excellent (<0.01)")
        ax.axhline(0.05, color="orange", linestyle="--", alpha=0.5, label="Acceptable (<0.05)")
        ax.set_xlabel("Beat Transition (n → n+1)")
        ax.set_ylabel("Normalized PV Loop L2 Distance")
        ax.set_title(f"{chamber} PV Loop Convergence")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 1, Col 3: Time constants
    ax = axes[0, 2]
    tau_data = {}
    for name in case_names:
        if "time_constants" in all_results[name]:
            tau_data[name] = all_results[name]["time_constants"].get("max_tau", 0)
    if tau_data:
        bars = ax.barh(list(tau_data.keys()), list(tau_data.values()), color=colors[:len(tau_data)])
        ax.axvline(3, color="red", linestyle="--", label="3 beats (current)")
        ax.set_xlabel("Max RC Time Constant (beats)")
        ax.set_title("Slowest 0D Compartment")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")

    # Row 2: SV and peak pressure convergence
    for i, (metric, ylabel) in enumerate([
        ("SV_LV", "LV Stroke Volume (mL)"),
        ("SV_RV", "RV Stroke Volume (mL)"),
        ("peak_pLV", "LV Peak Pressure (mmHg)"),
    ]):
        ax = axes[1, i]
        for j, name in enumerate(case_names):
            r = all_results[name]
            vals = r["convergence"]["hemodynamics"].get(metric, [])
            if len(vals) > 0:
                beats = np.arange(1, len(vals) + 1)
                ax.plot(beats, vals, "o-", color=colors[j], label=name, linewidth=2, markersize=6)
        ax.set_xlabel("Beat Number")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} Per Beat")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(outdir / "convergence_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Richardson extrapolation error ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Estimated Error in Last-Beat Metrics (Richardson Extrapolation)", fontsize=14, fontweight="bold")

    # Hemodynamic errors
    ax = axes[0]
    hemo_metrics = ["SV_LV", "SV_RV", "peak_pLV", "peak_pRV"]
    x = np.arange(len(hemo_metrics))
    width = 0.8 / n_cases
    for j, name in enumerate(case_names):
        errs = []
        for metric in hemo_metrics:
            vals = all_results[name]["convergence"]["hemodynamics"].get(metric, [])
            if len(vals) >= 3:
                _, err = richardson_extrapolation(vals)
            else:
                err = 0
            errs.append(err)
        ax.bar(x + j * width, errs, width, label=name, color=colors[j])
    ax.set_xticks(x + width * (n_cases - 1) / 2)
    ax.set_xticklabels(hemo_metrics, fontsize=9)
    ax.set_ylabel("Estimated Error (%)")
    ax.set_title("Hemodynamic Metrics")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(2, color="orange", linestyle="--", alpha=0.7, label="2% threshold")
    ax.axhline(5, color="red", linestyle="--", alpha=0.7, label="5% threshold")

    # Work errors
    ax = axes[1]
    work_metrics = ["work_true_LV", "work_true_RV", "work_true_Septum"]
    work_labels = ["W_LV", "W_RV", "W_Septum"]
    x = np.arange(len(work_metrics))
    for j, name in enumerate(case_names):
        errs = []
        for wk in work_metrics:
            vals = all_results[name]["convergence"]["work"].get(wk, [])
            if len(vals) >= 3:
                _, err = richardson_extrapolation(vals)
            else:
                err = 0
            errs.append(err)
        ax.bar(x + j * width, errs, width, label=name, color=colors[j])
    ax.set_xticks(x + width * (n_cases - 1) / 2)
    ax.set_xticklabels(work_labels, fontsize=9)
    ax.set_ylabel("Estimated Error (%)")
    ax.set_title("Myocardial Work")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(2, color="orange", linestyle="--", alpha=0.7)
    ax.axhline(5, color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(outdir / "convergence_richardson_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: 0D state drift ──
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("0D Windkessel State: End-of-Beat Values", fontsize=14, fontweight="bold")
    state_keys = ["V_LA", "V_LV", "V_RA", "V_RV", "p_AR_SYS", "p_VEN_SYS", "p_AR_PUL", "p_VEN_PUL"]
    state_labels = ["V_LA (mL)", "V_LV (mL)", "V_RA (mL)", "V_RV (mL)",
                    "p_AR_SYS (mmHg)", "p_VEN_SYS (mmHg)", "p_AR_PUL (mmHg)", "p_VEN_PUL (mmHg)"]

    for i, (sk, sl) in enumerate(zip(state_keys, state_labels)):
        ax = axes[i // 4, i % 4]
        for j, name in enumerate(case_names):
            vals = all_results[name]["convergence"]["state_convergence"].get(sk, [])
            if len(vals) > 0:
                beats = np.arange(1, len(vals) + 1)
                ax.plot(beats, vals, "o-", color=colors[j], label=name, linewidth=2, markersize=5)
        ax.set_xlabel("Beat")
        ax.set_ylabel(sl)
        ax.set_title(sk)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(outdir / "convergence_0D_states.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved convergence plots to {outdir}/")


def print_summary(all_results):
    """Print convergence summary table."""
    print("\n" + "=" * 100)
    print("CONVERGENCE SUMMARY")
    print("=" * 100)

    # Header
    print(f"\n{'Case':<18} {'PV_L2_LV':>10} {'PV_L2_RV':>10} {'dSV_LV%':>10} {'dSV_RV%':>10} "
          f"{'dp_LV%':>10} {'dp_RV%':>10} {'max_tau':>10}")
    print("-" * 100)

    for name, data in all_results.items():
        conv = data["convergence"]
        tau = data.get("time_constants", {}).get("max_tau", 0)

        # Last PV L2
        lv_l2 = conv["pv_l2"].get("LV", [0])[-1]
        rv_l2 = conv["pv_l2"].get("RV", [0])[-1]

        # Last beat-to-beat change in SV
        sv_lv = conv["hemodynamics"].get("SV_LV", [])
        sv_rv = conv["hemodynamics"].get("SV_RV", [])
        dsv_lv = abs(sv_lv[-1] - sv_lv[-2]) / abs(sv_lv[-2]) * 100 if len(sv_lv) >= 2 and sv_lv[-2] != 0 else 0
        dsv_rv = abs(sv_rv[-1] - sv_rv[-2]) / abs(sv_rv[-2]) * 100 if len(sv_rv) >= 2 and sv_rv[-2] != 0 else 0

        # Peak pressure change
        plv = conv["hemodynamics"].get("peak_pLV", [])
        prv = conv["hemodynamics"].get("peak_pRV", [])
        dplv = abs(plv[-1] - plv[-2]) / abs(plv[-2]) * 100 if len(plv) >= 2 else 0
        dprv = abs(prv[-1] - prv[-2]) / abs(prv[-2]) * 100 if len(prv) >= 2 else 0

        print(f"{name:<18} {lv_l2:>10.4f} {rv_l2:>10.4f} {dsv_lv:>9.2f}% {dsv_rv:>9.2f}% "
              f"{dplv:>9.2f}% {dprv:>9.2f}% {tau:>10.1f}")

    # Richardson extrapolation summary
    print(f"\n{'--- Richardson Extrapolation: Estimated error in last beat ---':^100}")
    print(f"{'Case':<18} {'SV_LV%':>10} {'SV_RV%':>10} {'pLV%':>10} {'pRV%':>10} "
          f"{'W_LV%':>10} {'W_RV%':>10} {'W_Sept%':>10} {'CONVERGED?':>12}")
    print("-" * 100)

    for name, data in all_results.items():
        conv = data["convergence"]
        errs = {}

        for metric in ["SV_LV", "SV_RV", "peak_pLV", "peak_pRV"]:
            vals = conv["hemodynamics"].get(metric, [])
            _, err = richardson_extrapolation(vals)
            errs[metric] = err

        for wk, label in [("work_true_LV", "W_LV"), ("work_true_RV", "W_RV"),
                          ("work_true_Septum", "W_Sept")]:
            vals = conv["work"].get(wk, [])
            _, err = richardson_extrapolation(vals)
            errs[label] = err

        max_err = max(errs.values())
        converged = "YES" if max_err < 2.0 else ("MARGINAL" if max_err < 5.0 else "NO")

        print(f"{name:<18} {errs.get('SV_LV', 0):>9.2f}% {errs.get('SV_RV', 0):>9.2f}% "
              f"{errs.get('peak_pLV', 0):>9.2f}% {errs.get('peak_pRV', 0):>9.2f}% "
              f"{errs.get('W_LV', 0):>9.2f}% {errs.get('W_RV', 0):>9.2f}% "
              f"{errs.get('W_Sept', 0):>9.2f}% {converged:>12}")

    print()
    print("Thresholds: <2% = converged, 2-5% = marginal, >5% = NOT converged")
    print("PV L2 norm: <0.01 = excellent, 0.01-0.05 = acceptable, >0.05 = poor")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_convergence.py /path/to/results/sims/DATE")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    # Find case folders
    case_folders = find_case_folders(results_dir)
    if not case_folders:
        print("No case folders found. Check run_description.txt files.")
        sys.exit(1)

    print(f"Found {len(case_folders)} cases:")
    for name, folder in case_folders.items():
        print(f"  {name}: {folder.name}")

    # Analyze each case
    all_results = {}
    for name, folder in sorted(case_folders.items()):
        m, circ, sp = load_case_data(folder)
        conv = compute_convergence_metrics(m, circ, sp)

        # Time constants
        circ_json = CIRC_DIR / CASES[name][0]
        taus = compute_time_constants(circ_json, conv["T_cycle"]) if circ_json.exists() else {}

        all_results[name] = {
            "folder": folder,
            "convergence": conv,
            "time_constants": taus,
        }

    # Print summary
    print_summary(all_results)

    # Plot
    plot_convergence(all_results, results_dir / "convergence_analysis")


if __name__ == "__main__":
    main()
