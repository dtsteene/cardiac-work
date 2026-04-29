"""Minimal LV-only cascade figure for the thesis.

Reads cached cascade data from analyze_cascade.py and produces a single-panel
figure intended for the conceptual cascade explanation in the question chapter.
The four curves are annotated with right-side brackets naming each
simplification step.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("results/analysis/cascade/cascade_raw.npz")
OUT = Path("results/analysis/cascade/fig_cascade_cumulative_minimal")
WALL_VOL_M3 = json.load(open("/tmp/ukb_wall_volumes.json"))

d = np.load(DATA)
t = d["time"]
R = "LV_tau_lap"
V = WALL_VOL_M3[R]


def cum_kPa(key):
    return np.cumsum(d[f"{R}_{key}"]) / V * 1e-3


curves = {
    "W0_per_step": (r"$\mathbf{S}:\dot{\mathbf{E}}$",         "#111111", 2.4),
    "W2_per_step": (r"$S_{ff}\,\dot E_{ff}$",                  "#1f4e79", 1.8),
    "W3_per_step": (r"$P_{cav}\,\dot E_{ff}$",                 "#5b8fc9", 1.8),
    "W4_per_step": (r"$P_{cav}\,\dot \varepsilon_{ll}$",       "#b04a3a", 1.8),
}

cum = {k: cum_kPa(k) for k in curves}

fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)

for key, (label, color, lw) in curves.items():
    ax.plot(t, cum[key], color=color, lw=lw, label=label)

ax.axhline(0, color="lightgray", lw=0.6, zorder=0)
ax.set_xlabel(r"$t$ (s)", fontsize=11)
ax.set_ylabel("Cumulative work density (kPa)", fontsize=11)
ax.grid(alpha=0.20)
ax.set_xlim(0, t[-1])
ax.legend(loc="lower left", fontsize=10, framealpha=0.95, handlelength=2.2,
          frameon=False)

fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
print(f"Saved {OUT.with_suffix('.png')}")
print("Plateaus (kPa): "
      f"W0={cum['W0_per_step'][-1]:.2f}  "
      f"W2={cum['W2_per_step'][-1]:.2f}  "
      f"W3={cum['W3_per_step'][-1]:.2f}  "
      f"W4={cum['W4_per_step'][-1]:.2f}")
