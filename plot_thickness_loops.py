#!/usr/bin/env python3
"""
plot_thickness_loops.py — Overlay fiber stress-strain and pressure-strain
loops across thickness variants, so we can see what's mechanically changing.

For each sweep (rvfw_severe, lvfw_severe, uniform_severe):
  Row 1: S_ff vs E_ff  (fiber stress-strain — what the tissue actually does)
  Row 2: P_cav vs ε_ll (pressure-longitudinal strain — what the proxy sees)
  Columns: LV | Septum | RV

Loops are coloured by thickness (light = thin, dark = thick).
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

OUT = Path("results/analysis/thickness_loops")
OUT.mkdir(parents=True, exist_ok=True)

# ── Run definitions ─────────────────────────────────────────────────────────
SWEEPS = {
    "rvfw_severe": {
        "title": "RV free wall thickening — severe PAH",
        "runs": [
            (3.0,  "results/sims/2026-04-14/UKB_1beats_run_1029755"),
            (4.5,  "results/sims/2026-04-15/UKB_1beats_run_1033857"),
            (6.0,  "results/sims/2026-04-14/UKB_1beats_run_1029757"),
            (7.5,  "results/sims/2026-04-15/UKB_1beats_run_1033858"),
            (9.0,  "results/sims/2026-04-14/UKB_1beats_run_1029759"),
            (10.5, "results/sims/2026-04-15/UKB_1beats_run_1033859"),
            (12.0, "results/sims/2026-04-14/UKB_1beats_run_1029761"),
        ],
    },
    "lvfw_severe": {
        "title": "LV free wall thickening — severe PAH",
        "runs": [
            (3.0,  "results/sims/2026-04-15/UKB_1beats_run_1033860"),
            (6.0,  "results/sims/2026-04-15/UKB_1beats_run_1033861"),
            (9.0,  "results/sims/2026-04-15/UKB_1beats_run_1033862"),
            (12.0, "results/sims/2026-04-15/UKB_1beats_run_1033863"),
        ],
    },
    "uniform_severe": {
        "title": "Uniform thickening — severe PAH",
        "runs": [
            (3.0,  "results/sims/2026-04-16/UKB_1beats_run_1034606"),
            (6.0,  "results/sims/2026-04-16/UKB_1beats_run_1034607"),
            (9.0,  "results/sims/2026-04-16/UKB_1beats_run_1034608"),
            (12.0, "results/sims/2026-04-16/UKB_1beats_run_1034609"),
        ],
    },
}

# Regions: use LDRB 3-region tags (LV=1, RV=2, Septum=3) for metric keys
REGIONS = [
    ("LV",     "LV",     "LV free wall"),
    ("Septum", "Septum", "Septum"),
    ("RV",     "RV",     "RV free wall"),
]

PA_TO_KPA = 1e-3
MMHG_TO_KPA = 0.133322


def load_case(sim_dir):
    sim_dir = Path(sim_dir)
    m = np.load(sim_dir / "metrics" / "metrics_downsample_1.npy",
                allow_pickle=True).item()
    sp = np.load(sim_dir / "solver" / "solver_cavity_pressure_mmHg.npy")
    return m, sp


for sweep_key, sweep in SWEEPS.items():
    runs = sweep["runs"]
    n = len(runs)
    mm_values = [r[0] for r in runs]
    mm_min, mm_max = min(mm_values), max(mm_values)

    # Color map: light (thin) → dark (thick)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=mm_min, vmax=mm_max)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    for mm, sim_dir in runs:
        color = cmap(norm(mm))
        m, sp = load_case(sim_dir)
        t = np.array(m["time"])
        P_LV_kpa = sp[:len(t), 0] * MMHG_TO_KPA
        P_RV_kpa = sp[:len(t), 1] * MMHG_TO_KPA

        for col, (reg_key, met_key, title) in enumerate(REGIONS):
            # Extract regional means
            E_ff = np.array(m.get(f"mean_E_ff_{met_key}", np.zeros(len(t))))
            E_ll = np.array(m.get(f"mean_E_ll_{met_key}", np.zeros(len(t))))
            S_ff = np.array(m.get(f"mean_S_ff_{met_key}", np.zeros(len(t)))) * PA_TO_KPA

            # Choose cavity pressure for this region
            if reg_key == "LV":
                P = P_LV_kpa
            elif reg_key == "RV":
                P = P_RV_kpa
            else:  # Septum — show both
                P = P_LV_kpa  # will also overlay P_RV below

            # Row 0: S_ff vs E_ff (fiber stress-strain)
            ax = axes[0, col]
            ax.plot(E_ff, S_ff, color=color, lw=1.4, alpha=0.85)

            # Row 1: P vs ε_ll (pressure-longitudinal strain = proxy loop)
            ax = axes[1, col]
            ax.plot(E_ll, P, color=color, lw=1.4, alpha=0.85)

            # For septum, also show P_RV as dashed
            if reg_key == "Septum":
                ax.plot(E_ll, P_RV_kpa, color=color, lw=1.0, ls="--",
                        alpha=0.6)

    # Labels and formatting
    for col, (_, _, title) in enumerate(REGIONS):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")
        axes[0, col].set_xlabel(r"$E_{ff}$ (fiber strain)", fontsize=10)
        axes[0, col].set_ylabel(r"$S_{ff}$ (kPa)", fontsize=10)
        axes[0, col].grid(alpha=0.25)
        axes[0, col].axhline(0, color="gray", lw=0.5)
        axes[0, col].axvline(0, color="gray", lw=0.5)

        axes[1, col].set_xlabel(r"$\varepsilon_{ll}$ (longitudinal strain)",
                                fontsize=10)
        axes[1, col].set_ylabel(r"$P_{cav}$ (kPa)", fontsize=10)
        axes[1, col].grid(alpha=0.25)
        axes[1, col].axhline(0, color="gray", lw=0.5)
        axes[1, col].axvline(0, color="gray", lw=0.5)

    # Septum note
    axes[1, 1].text(0.98, 0.02, "solid = $P_{LV}$\ndashed = $P_{RV}$",
                    transform=axes[1, 1].transAxes, fontsize=8,
                    ha="right", va="bottom", style="italic", color="gray")

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("Added wall thickness (mm)", fontsize=10)

    fig.suptitle(f"{sweep['title']}\n"
                 f"Top: fiber stress-strain (what the tissue does)    "
                 f"Bottom: pressure × longitudinal strain (what the proxy sees)",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / f"fig_loops_{sweep_key}.png", dpi=160,
                bbox_inches="tight")
    fig.savefig(OUT / f"fig_loops_{sweep_key}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_loops_{sweep_key}.png")

print("Done.")
