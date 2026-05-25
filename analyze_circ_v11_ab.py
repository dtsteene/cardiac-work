#!/usr/bin/env python3
"""
A/B analysis of v11 circulation optimization: linear vs exponential EDPVR.

Reads the 18 JSON outputs (9 severities × 2 variants) from
    data/ukb_circ_v11/          (exponential, kE free)
    data/ukb_circ_v11_linear/   (linear, kE=0)
and produces:
  1. Side-by-side achieved-hemodynamics table
  2. Per-target pass rate per variant (ESC/Kovacs tolerances)
  3. PV-loop spectrum plot (both variants, 2x2 grid)
  4. Transmural-pressure collapse plot (both variants)
  5. EDPVR family curves (both variants' Klotz passive curves at healthy)

Usage: python analyze_circ_v11_ab.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Configuration
# ==============================================================================
WORK = Path("/global/D1/homes/dtsteene/cardiac-work")
EXP_DIR = WORK / "data" / "ukb_circ_v11"
LIN_DIR = WORK / "data" / "ukb_circ_v11_linear"
OUT_DIR = WORK / "results" / "v11_ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEVERITIES = [
    "healthy_low", "healthy", "borderline", "mild",
    "moderate", "moderate_severe", "severe", "very_severe", "end_stage",
]

# ESC Table 16 tolerances (Humbert 2022) for pass/fail
PRESSURE_TOL_PCT = {"RV_ESP": 15, "RV_EDP": 25, "LV_ESP": 20,
                    "LV_EDP": 25, "Ao_DBP": 20, "LA_P_MEAN": 15}
VOLUME_TOL_PCT = 10


# ==============================================================================
# Load
# ==============================================================================
def load_variant(base, name_suffix):
    out = {}
    for sev in SEVERITIES:
        p = base / f"optimized_regazzoni_{name_suffix}_{sev}.json"
        if not p.exists():
            out[sev] = None
            continue
        out[sev] = json.loads(p.read_text())
    return out


EXP = load_variant(EXP_DIR, "ukb")
LIN = load_variant(LIN_DIR, "ukb_linear")


def _ok(vals, target, tol_pct):
    """True if (vals - target) / target is within +/- tol_pct."""
    if target is None or abs(target) < 1e-6:
        return True
    return abs(vals - target) / abs(target) * 100 <= tol_pct


# ==============================================================================
# Achieved-hemodynamics table (printed + CSV)
# ==============================================================================
print("=" * 110)
print(f"{'SEVERITY':<18}| {'VAR':>6} | {'RV_ESP':>7}|{'RV_EDP':>7}|{'LV_ESP':>7}|"
      f"{'LV_EDP':>7}|{'Ao_DBP':>7}|{'LA_P':>6}|{'CI':>5}|{'LV_EF':>6}|{'RV_EF':>6}")
print("-" * 110)

rows_csv = ["severity,variant,RV_ESP,RV_EDP,LV_ESP,LV_EDP,Ao_DBP,LA_P_MEAN,"
            "LV_EDV,RV_EDV,CI,LV_EF_pct,RV_EF_pct"]

for sev in SEVERITIES:
    for label, data in [("EXP", EXP.get(sev)), ("LIN", LIN.get(sev))]:
        if data is None:
            print(f"{sev:<18}| {label:>6} |  (missing)")
            continue
        m = data.get("metrics_achieved", {})
        h = data.get("derived_hemodynamics", {})
        print(f"{sev:<18}| {label:>6} | "
              f"{m.get('RV_ESP', 0):>7.1f}|{m.get('RV_EDP', 0):>7.1f}|"
              f"{m.get('LV_ESP', 0):>7.1f}|{m.get('LV_EDP', 0):>7.1f}|"
              f"{m.get('Ao_DBP', 0):>7.1f}|{m.get('LA_P_MEAN', 0):>6.1f}|"
              f"{h.get('CI_Lpm_m2', 0):>5.2f}|"
              f"{h.get('LV_EF_pct', 0):>5.1f}%|{h.get('RV_EF_pct', 0):>5.1f}%")
        rows_csv.append(
            f"{sev},{label},{m.get('RV_ESP',0):.2f},{m.get('RV_EDP',0):.2f},"
            f"{m.get('LV_ESP',0):.2f},{m.get('LV_EDP',0):.2f},"
            f"{m.get('Ao_DBP',0):.2f},{m.get('LA_P_MEAN',0):.2f},"
            f"{m.get('LV_EDV',0):.2f},{m.get('RV_EDV',0):.2f},"
            f"{h.get('CI_Lpm_m2',0):.3f},{h.get('LV_EF_pct',0):.2f},"
            f"{h.get('RV_EF_pct',0):.2f}")

(OUT_DIR / "achieved_hemodynamics.csv").write_text("\n".join(rows_csv) + "\n")
print("=" * 110)
print(f"Saved: {OUT_DIR / 'achieved_hemodynamics.csv'}")


# ==============================================================================
# Per-target pass-rate summary
# ==============================================================================
def pass_count(data_dict):
    counts = {k: 0 for k in list(PRESSURE_TOL_PCT) + ["LV_EDV", "RV_EDV", "SV_BAL"]}
    total = 0
    for sev in SEVERITIES:
        d = data_dict.get(sev)
        if d is None:
            continue
        total += 1
        m = d["metrics_achieved"]
        t = d["pressure_targets"]
        v = d["mesh_volumes_mL"]
        for key, tol in PRESSURE_TOL_PCT.items():
            if _ok(m[key], t[key], tol):
                counts[key] += 1
        for edv, mesh_val in [("LV_EDV", v["LV_EDV"]), ("RV_EDV", v["RV_EDV"])]:
            if _ok(m[edv], mesh_val, VOLUME_TOL_PCT):
                counts[edv] += 1
        lv_sv = m["SV"]
        rv_sv = (d["derived_hemodynamics"].get("RV_SV")
                 or m.get("RV_EDV", 0) * d["derived_hemodynamics"].get("RV_EF_pct", 0) / 100)
        if abs(lv_sv - rv_sv) < 2.0:
            counts["SV_BAL"] += 1
    return counts, total


exp_pass, exp_n = pass_count(EXP)
lin_pass, lin_n = pass_count(LIN)

print()
print("=" * 65)
print(f"{'TARGET':<12} | EXP pass (n={exp_n}) | LIN pass (n={lin_n})")
print("-" * 65)
all_targets = list(PRESSURE_TOL_PCT) + ["LV_EDV", "RV_EDV", "SV_BAL"]
for tgt in all_targets:
    e = exp_pass.get(tgt, 0)
    l = lin_pass.get(tgt, 0)
    print(f"{tgt:<12} | {e}/{exp_n} ({e/max(exp_n,1)*100:4.0f}%) | "
          f"{l}/{lin_n} ({l/max(lin_n,1)*100:4.0f}%)")
print("=" * 65)


# ==============================================================================
# PV-loop spectrum plot (rerun the calibrated parameters, plot loops)
# ==============================================================================
def replay_pv_loops(data, n_beats=50, dt=1e-3):
    """Re-solve the calibrated parameters to steady state and return last-beat
    LV/RV volumes and pressures."""
    from circulation.regazzoni2020 import Regazzoni2020
    params = data["parameters"]
    init = data["initial_state"]
    model = Regazzoni2020(parameters=params, initial_state=init,
                          add_units=False, verbose=False)
    history = model.solve(num_beats=n_beats, dt=dt)
    hr = params["HR"]
    samples = int((1 / hr) / dt)
    slc = slice(-samples, None)
    return {
        "V_LV": history["V_LV"][slc], "p_LV": history["p_LV"][slc],
        "V_RV": history["V_RV"][slc], "p_RV": history["p_RV"][slc],
    }


def plot_pv_loops():
    cmap = plt.cm.coolwarm
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    variants = [("Exponential EDPVR (kE free)", EXP), ("Linear EDPVR (kE=0)", LIN)]
    for col, (title, data_dict) in enumerate(variants):
        ax_lv, ax_rv = axes[0, col], axes[1, col]
        for i, sev in enumerate(SEVERITIES):
            d = data_dict.get(sev)
            if d is None:
                continue
            try:
                loops = replay_pv_loops(d)
            except Exception as e:
                print(f"  [skip] {title} / {sev}: {e}")
                continue
            color = cmap(i / (len(SEVERITIES) - 1))
            ax_lv.plot(loops["V_LV"], loops["p_LV"], color=color, lw=1.5, label=sev)
            ax_rv.plot(loops["V_RV"], loops["p_RV"], color=color, lw=1.5, label=sev)
        ax_lv.set(xlabel="V_LV [mL]", ylabel="P_LV [mmHg]", title=f"LV — {title}")
        ax_rv.set(xlabel="V_RV [mL]", ylabel="P_RV [mmHg]", title=f"RV — {title}")
        ax_lv.grid(alpha=0.3); ax_rv.grid(alpha=0.3)
        ax_lv.legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle("v11 A/B comparison — PV loops across spectrum", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "pv_loops_ab.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ==============================================================================
# Transmural-pressure collapse plot
# ==============================================================================
def plot_transmural_collapse():
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for label, data_dict, marker in [("Exponential", EXP, "o"),
                                      ("Linear",      LIN, "s")]:
        xs, ys = [], []
        for sev in SEVERITIES:
            d = data_dict.get(sev)
            if d is None:
                continue
            m = d["metrics_achieved"]
            rv = m["RV_ESP"]
            trans = m["LV_ESP"] - m["RV_ESP"]
            xs.append(rv); ys.append(trans)
        if xs:
            ax.plot(xs, ys, marker=marker, lw=1.5, ms=8, label=label)
    ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.5)
    ax.set(xlabel="Achieved RV_ESP [mmHg]",
           ylabel=r"Transmural pressure $P_\mathrm{LV,ES} - P_\mathrm{RV,ES}$ [mmHg]",
           title="v11 A/B — Transmural pressure collapse vs RV_ESP")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "transmural_collapse_ab.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ==============================================================================
# EDPVR family comparison
# ==============================================================================
def plot_edpvr_family():
    """Plot the passive (diastolic) P-V curves for LV and RV under each calibrated
    variant at the healthy operating point, to show how the Klotz exponential
    differs from the linear EDPVR."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, chamber in zip(axes, ["LV", "RV"]):
        for label, data_dict in [("Exponential", EXP), ("Linear", LIN)]:
            d = data_dict.get("healthy")
            if d is None:
                continue
            p = d["parameters"]["chambers"][chamber]
            EA, EB, V0 = p["EA"], p["EB"], p["V0"]
            kE = p.get("kE", 0.0)
            V = np.linspace(V0, V0 + 140, 400)
            if kE > 1e-9:
                P_pas = (EB / kE) * (np.exp(kE * (V - V0)) - 1)
            else:
                P_pas = EB * (V - V0)
            ax.plot(V, P_pas, lw=2, label=f"{label} (kE={kE:.3g})")
        ax.set(xlabel=f"V_{chamber} [mL]", ylabel="P_passive [mmHg]",
               title=f"Healthy {chamber} — passive EDPVR family")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "edpvr_family_ab.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    try:
        plot_pv_loops()
    except Exception as e:
        print(f"pv_loops plot skipped: {e}")
    try:
        plot_transmural_collapse()
    except Exception as e:
        print(f"transmural_collapse plot skipped: {e}")
    try:
        plot_edpvr_family()
    except Exception as e:
        print(f"edpvr_family plot skipped: {e}")

    print()
    print(f"All outputs in: {OUT_DIR}/")
