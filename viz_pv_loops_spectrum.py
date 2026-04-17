#!/usr/bin/env python3
"""
viz_pv_loops_spectrum.py — 2x2 grid of PV loops across the severity spectrum.

Rows: ODE (circulation model) vs Solver (FEM Lagrange multiplier)
Cols: LV cavity  vs  RV cavity

ODE pressures come from circulation/history.npy (p_LV, p_RV).
Solver pressures come from solver/solver_cavity_pressure_mmHg.npy.
Cavity volumes are shared (coupling is volume-driven), taken from history.npy.

Last-beat loops only, coloured by severity (perceptually ordered).
Opens Section B of the presentation: "here is the spectrum of hemodynamic
states we are testing against".
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

OUT = Path("results/analysis/spectrum")
OUT.mkdir(parents=True, exist_ok=True)

# Ordered spectrum (severity_key, display_label, sim_dir_suffix)
SPECTRUM = [
    ("healthy",         "Borderline PH", "1020849"),
    ("mild",            "Mild",          "1020851"),
    ("moderate",        "Moderate",      "1020852"),
    ("moderate_severe", "Mod–severe",    "1020853"),
    ("severe",          "Severe",        "1020854"),
    ("very_severe",     "Very severe",   "1020855"),
    ("end_stage",       "End-stage",     "1020856"),
]

ROOT = Path("results/sims/2026-04-12")
BPM = 75
DT = 0.001
CYCLE_LEN = 60.0 / BPM
SAMPLES_PER_BEAT = int(round(CYCLE_LEN / DT))  # 800

# Plasma palette sampled at 7 points, same convention as the thesis
cmap = cm.plasma
colors = [cmap(0.10 + 0.75 * i / (len(SPECTRUM) - 1)) for i in range(len(SPECTRUM))]

fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True,
                          sharex="col")
# Axis roles
ax_ode_lv   = axes[0, 0]
ax_ode_rv   = axes[0, 1]
ax_solv_lv  = axes[1, 0]
ax_solv_rv  = axes[1, 1]

for i, (key, label, run_id) in enumerate(SPECTRUM):
    sim_dir = ROOT / f"UKB_6beats_run_{run_id}"
    hist = np.load(sim_dir / "history.npy", allow_pickle=True).item()
    V_LV = np.asarray(hist["V_LV"])    # mL
    V_RV = np.asarray(hist["V_RV"])
    p_LV_ode = np.asarray(hist["p_LV"])  # mmHg (ODE side)
    p_RV_ode = np.asarray(hist["p_RV"])

    solver_p = np.load(sim_dir / "solver" / "solver_cavity_pressure_mmHg.npy",
                        allow_pickle=True)
    p_LV_sol = solver_p[:, 0]
    p_RV_sol = solver_p[:, 1]

    # Last beat slice; ODE arrays have length 4801, solver has 4800
    last = slice(-SAMPLES_PER_BEAT, None)

    color = colors[i]
    ax_ode_lv.plot(V_LV[last], p_LV_ode[last], color=color, lw=2.0,
                    label=label, alpha=0.95)
    ax_ode_rv.plot(V_RV[last], p_RV_ode[last], color=color, lw=2.0,
                    alpha=0.95)
    # Solver side: V from history, P from solver. Align lengths.
    V_LV_for_solver = V_LV[-len(p_LV_sol):]
    V_RV_for_solver = V_RV[-len(p_RV_sol):]
    last_solver = slice(-SAMPLES_PER_BEAT, None)
    ax_solv_lv.plot(V_LV_for_solver[last_solver], p_LV_sol[last_solver],
                     color=color, lw=2.0, alpha=0.95)
    ax_solv_rv.plot(V_RV_for_solver[last_solver], p_RV_sol[last_solver],
                     color=color, lw=2.0, alpha=0.95)

# Cosmetic
for ax in axes.flat:
    ax.grid(alpha=0.25)
    ax.axhline(0, color="lightgray", lw=0.5)
ax_ode_lv.set_ylabel(r"$P_{LV}$ (mmHg)")
ax_ode_rv.set_ylabel(r"$P_{RV}$ (mmHg)")
ax_solv_lv.set_ylabel(r"$P_{LV}$ (mmHg)")
ax_solv_rv.set_ylabel(r"$P_{RV}$ (mmHg)")
ax_solv_lv.set_xlabel(r"$V_{LV}$ (mL)")
ax_solv_rv.set_xlabel(r"$V_{RV}$ (mL)")

# Row labels on the left axis
ax_ode_lv.text(-0.22, 0.5, "ODE\n(circulation model)", transform=ax_ode_lv.transAxes,
                ha="center", va="center", fontsize=12, fontweight="bold", rotation=90)
ax_solv_lv.text(-0.22, 0.5, "Solver\n(FEM cavity $P$)", transform=ax_solv_lv.transAxes,
                 ha="center", va="center", fontsize=12, fontweight="bold", rotation=90)

ax_ode_lv.set_title("LV", fontsize=13, fontweight="bold")
ax_ode_rv.set_title("RV", fontsize=13, fontweight="bold")

# Legend on the top-left panel (labels were attached there)
ax_ode_lv.legend(loc="lower right", fontsize=9, framealpha=0.95, ncol=1)

fig.suptitle("Spectrum PV loops  (last beat, 7 calibrated severities)",
              fontsize=13, fontweight="bold")
fig.savefig(OUT / "fig_spectrum_pv_loops.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "fig_spectrum_pv_loops.pdf", bbox_inches="tight")
print(f"Saved {OUT / 'fig_spectrum_pv_loops.png'}")
