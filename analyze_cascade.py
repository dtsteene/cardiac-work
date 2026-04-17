#!/usr/bin/env python3
"""
analyze_cascade.py — The simplification cascade from gold-standard tensor
work down to the clinical pressure-strain proxy, for the LV and RV
territories of one representative case.

Level 0  full tensor   W = ∫_T ∫_Ω    S : dE              dV dt    (gold std)
Level 1  fiber         W = ∫_T ∫_Ω    S_ff dE_ff          dV dt    (best single direction)
Level 2  P × E_ff      W = ∫_T ∫_Ω    P_cav(t) dE_ff      dV dt    (stress → cavity P)
Level 3  P × E_ll      W = ∫_T ∫_Ω    P_cav(t) dE_ll      dV dt    (clinical proxy)

Each level is a strict simplification of the previous and each step has a
named physical interpretation. This script:

  1. confirms the energy balance W_int ≈ W_boundary on the whole heart
     (foundational claim: tensor work is the conserved quantity that moves
     blood, regional decomposition is a partition of a physically meaningful
     scalar);
  2. builds a power-time-series plot showing, for each region, the five
     levels overlaid against the gold-standard integrand;
  3. builds a cumulative-work plot showing how the simplification accumulates
     over one cycle;
  4. builds a bar chart of total integrated work per level per region.

Loops grid is built by analyze_cascade_loops.py (separate to keep this file
under one responsibility).

Source case: a healthy circulation run with both metrics_calculator output
and saved solver pressures. Currently 2026-04-13/UKB_1beats_run_1023580
(robin-only base BC test, but otherwise standard healthy hemodynamics —
P_LV peak 121 mmHg, P_RV peak 29 mmHg, 75 bpm). The cascade message is
methodological so the BC variant doesn't change the conclusions.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
SIM_DIR = Path("results/sims/2026-04-13/UKB_1beats_run_1023580")
OUT_DIR = Path("results/analysis/cascade")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Wall volumes per region — pulled from the canonical UKB shared mesh.
# Same for any 6/1-beat sim that uses the shared geometry.
WALL_VOL_M3 = json.load(open("/tmp/ukb_wall_volumes.json"))
print(f"Wall volumes (mL): {{k: round(v*1e6,2) for k,v in WALL_VOL_M3.items()}}")

# All work figures below are displayed as INTENSIVE WORK DENSITY (J/m^3 = Pa,
# shown as kPa = mJ/mL) rather than extensive work (J) — the clinical
# myocardial-work index is intensive, and intensive form removes the
# volume-scaling confound when comparing across meshes with different wall
# volumes. Energy balance (fig 1) and loops (fig 5) are left as-is because
# they are about conservation / stress-strain geometry, not regional work.
def to_kPa(joules, region):
    """J per region  ->  kPa (= mJ/mL) of work density in that region."""
    return joules / WALL_VOL_M3[region] * 1e-3

# ── Load metrics + pressures ─────────────────────────────────────────────────
metrics = np.load(SIM_DIR / "metrics" / "metrics_downsample_1.npy",
                  allow_pickle=True).item()
solver_pressures = np.load(SIM_DIR / "solver" /
                            "solver_cavity_pressure_mmHg.npy",
                            allow_pickle=True)  # shape (n_steps, 2) mmHg
P_LV_MMHG = solver_pressures[:, 0]
P_RV_MMHG = solver_pressures[:, 1]
MMHG_TO_PA = 133.322

t = np.array(metrics["time"])  # seconds
dt = float(t[1] - t[0])
n_steps = len(t)
print(f"Time series: {n_steps} steps, dt={dt*1000:.2f} ms, "
      f"total duration {t[-1]:.3f} s")
print(f"Solver pressures: P_LV peak {P_LV_MMHG.max():.1f} mmHg, "
      f"P_RV peak {P_RV_MMHG.max():.1f} mmHg")

# Match pressure length to metrics length (pressures may have been saved at
# every solver substep; metrics are downsampled to physical timesteps)
if len(P_LV_MMHG) != n_steps:
    print(f"  ! pressure length {len(P_LV_MMHG)} != metrics length {n_steps}; "
          f"interpolating to metrics time grid")
    src_t = np.linspace(t[0], t[-1], len(P_LV_MMHG))
    P_LV_MMHG = np.interp(t, src_t, P_LV_MMHG)
    P_RV_MMHG = np.interp(t, src_t, P_RV_MMHG)


def arr(key):
    return np.array(metrics[key])


# ── Helpers ──────────────────────────────────────────────────────────────────
def cumulative(per_step):
    """Cumulative sum starting at 0 of a per-step incremental quantity."""
    return np.concatenate(([0.0], np.cumsum(per_step)))[:-1]


def trapz_increment(y):
    """Per-step increment using centered differences (for d/dt of a state).
    Returns dy with same length as y, dy[0] = y[1]-y[0], dy[-1] = y[-1]-y[-2].
    """
    return np.gradient(y)  # centered difference; uses first/last differences at edges


# ── Cascade levels per region ────────────────────────────────────────────────
def cascade_for_region(region_name):
    """Compute the per-step work increments at each cascade level for one
    region. Returns a dict of arrays each of length n_steps in joules.
    Region must be one of LV_tau_{eu,lap} / RV_tau_{eu,lap} — the tau-split
    territories where every wall cell belongs to exactly one cavity.
    """
    R = region_name
    V = WALL_VOL_M3[R]

    # Level 0 — full tensor, exact (sum of stress-type decomposition)
    W0_per_step = (arr(f"work_active_{R}")
                   + arr(f"work_passive_{R}")
                   + arr(f"work_comp_{R}"))

    # Level 2 — fiber projection, exact
    W2_per_step = arr(f"work_ff_{R}")

    # Mean stress and strain components (regional means)
    S_ff = arr(f"mean_S_ff_{R}")
    S_ss = arr(f"mean_S_ss_{R}")
    S_nn = arr(f"mean_S_nn_{R}")
    E_ff = arr(f"mean_E_ff_{R}")
    E_ss = arr(f"mean_E_ss_{R}")
    E_nn = arr(f"mean_E_nn_{R}")
    E_ll = arr(f"mean_E_ll_{R}")

    # Level 1 — deviatoric tensor work using diagonal components only.
    tr_S = S_ff + S_ss + S_nn
    tr_E = E_ff + E_ss + E_nn
    s_ff = S_ff - tr_S / 3.0
    s_ss = S_ss - tr_S / 3.0
    s_nn = S_nn - tr_S / 3.0
    e_ff = E_ff - tr_E / 3.0
    e_ss = E_ss - tr_E / 3.0
    e_nn = E_nn - tr_E / 3.0
    de_ff = trapz_increment(e_ff)
    de_ss = trapz_increment(e_ss)
    de_nn = trapz_increment(e_nn)
    W1_per_step = (s_ff * de_ff + s_ss * de_ss + s_nn * de_nn) * V

    # Level 3 — replace stress with cavity pressure. Each region has a
    # "correct" cavity (adjacent to its wall) and a "wrong" cavity
    # (whichever is left). The clinical convention applies P_LV to
    # everything, so for the RV territory the "wrong" label corresponds to
    # the default clinical behaviour — computing it exposes how misleading
    # the LV-only convention is when applied to RV wall tissue.
    if R.startswith("LV_tau"):
        P_pa          = P_LV_MMHG * MMHG_TO_PA  # correct: adjacent cavity
        P_pa_wrong    = P_RV_MMHG * MMHG_TO_PA  # wrong:   opposite cavity
    elif R.startswith("RV_tau"):
        P_pa          = P_RV_MMHG * MMHG_TO_PA  # correct: adjacent cavity
        P_pa_wrong    = P_LV_MMHG * MMHG_TO_PA  # wrong:   clinical default
    else:
        raise ValueError(R)

    dE_ff_full = trapz_increment(E_ff)
    dE_ll_full = trapz_increment(E_ll)

    W3_per_step         = P_pa       * dE_ff_full * V
    W3_wrong_per_step   = P_pa_wrong * dE_ff_full * V
    W4_per_step         = P_pa       * dE_ll_full * V
    W4_wrong_per_step   = P_pa_wrong * dE_ll_full * V

    return {
        "W0_per_step":        W0_per_step,
        "W1_per_step":        W1_per_step,
        "W2_per_step":        W2_per_step,
        "W3_per_step":        W3_per_step,
        "W4_per_step":        W4_per_step,
        "W3_wrong_per_step":  W3_wrong_per_step,
        "W4_wrong_per_step":  W4_wrong_per_step,
        "S_ff": S_ff, "S_ss": S_ss, "S_nn": S_nn,
        "E_ff": E_ff, "E_ss": E_ss, "E_nn": E_nn, "E_ll": E_ll,
        "P_pa": P_pa, "P_pa_wrong": P_pa_wrong,
    }


cascade = {R: cascade_for_region(R)
           for R in ("LV_tau_eu", "RV_tau_eu", "LV_tau_lap", "RV_tau_lap")}

# ── Energy balance check on the whole heart ──────────────────────────────────
W_int_whole = (arr("work_active_Whole")
               + arr("work_passive_Whole")
               + arr("work_comp_Whole"))
W_bnd_lv = arr("work_boundary_exact_LV")
W_bnd_rv = arr("work_boundary_exact_RV")
W_bnd_total = W_bnd_lv + W_bnd_rv

# Cumulative
cum_int = np.cumsum(W_int_whole)
cum_bnd = np.cumsum(W_bnd_total)
residual = cum_int - cum_bnd
final_int = cum_int[-1]
final_bnd = cum_bnd[-1]
final_residual = residual[-1]
rel_residual = abs(final_residual) / max(abs(final_int), 1e-12)
print(f"\nEnergy balance (whole heart, one beat):")
print(f"  ∫ S:dE dV dt              = {final_int*1000:+.4f} mJ")
print(f"  ∫ (P_LV dV_LV + P_RV dV_RV)= {final_bnd*1000:+.4f} mJ")
print(f"  residual                   = {final_residual*1000:+.4f} mJ "
      f"({rel_residual*100:.2f}% of internal)")

# ── Figure 1: energy balance ─────────────────────────────────────────────────
fig, axA = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
axA.plot(t, cum_int * 1000.0, color="#2c3e50", lw=2.4,
         label=r"$\int_0^t \int_\Omega \mathbf{S}:\dot{\mathbf{E}}\,dV\,dt'$")
axA.plot(t, cum_bnd * 1000.0, color="#c0392b", lw=2.0, ls="--",
         label=r"$\int_0^t (P_{LV}\dot V_{LV} + P_{RV}\dot V_{RV})\,dt'$")
axA.axhline(0, color="lightgray", lw=0.6)
axA.set_xlabel("time (s)", fontsize=11)
axA.set_ylabel("Cumulative work (mJ)", fontsize=11)
axA.legend(loc="upper right", fontsize=11, framealpha=0.95)
axA.grid(alpha=0.25)
axA.set_title("Energy balance", fontsize=12, fontweight="bold")
fig.savefig(OUT_DIR / "fig_energy_balance.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig_energy_balance.pdf", bbox_inches="tight")
print(f"\nSaved {OUT_DIR / 'fig_energy_balance.png'}")

# ── Figure 2: cascade power time series ──────────────────────────────────────
# Four-level cascade (deviatoric step dropped — redundant for
# near-incompressible material). Internal keys preserved so the saved
# npz stays backwards-compatible.
LEVEL_STYLE = {
    "W0_per_step": dict(color="black",   lw=2.6, label=r"$\mathbf{S}:\dot{\mathbf{E}}$"),
    "W2_per_step": dict(color="#2ca02c", lw=1.8, label=r"$S_{ff}\,\dot E_{ff}$"),
    "W3_per_step": dict(color="#1f77b4", lw=1.8, label=r"$P_{cav}\,\dot E_{ff}$"),
    "W4_per_step": dict(color="#d62728", lw=1.8, label=r"$P_{cav}\,\dot \varepsilon_{ll}$"),
}

# The four main cascade figures use the Laplace tau split as default
# (smoother, internally consistent with LDRB tagging). A companion
# comparison figure at the end overlays Eu vs Lap totals so the listener
# sees that the cascade story is robust across the split definition.
PRIMARY = "lap"  # "eu" or "lap"
LV_R = f"LV_tau_{PRIMARY}"
RV_R = f"RV_tau_{PRIMARY}"

fig2, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=False,
                           constrained_layout=True)
for j, R in enumerate((LV_R, RV_R)):
    ax = axes[j]
    c = cascade[R]
    for key, style in LEVEL_STYLE.items():
        power_kpas = to_kPa(c[key] / dt, R)  # kPa/s = (mJ/mL)/s
        ax.plot(t, power_kpas, **style)
    ax.axhline(0, color="lightgray", lw=0.6)
    ax.set_xlabel(r"$t$ (s)", fontsize=11)
    if j == 0:
        ax.set_ylabel(r"$\dot W / V$ (kPa/s)", fontsize=12)
    ax.grid(alpha=0.25)
    ax.set_title("LV" if j == 0 else "RV", fontsize=13, fontweight="bold")
    lines, labels = ax.get_legend_handles_labels()
    new_labels = []
    for line in lines:
        for key, style in LEVEL_STYLE.items():
            if style["color"] == line.get_color():
                tot = to_kPa(cascade[R][key].sum(), R)
                new_labels.append(f"{style['label']}  ({tot:+.2f} kPa)")
                break
    ax.legend(lines, new_labels, loc="lower right", fontsize=9,
              framealpha=0.95, handlelength=1.6)

fig2.suptitle(f"Instantaneous power per region  ({PRIMARY} tau split)",
              fontsize=12, fontweight="bold")
fig2.savefig(OUT_DIR / "fig_cascade_power.png", dpi=160, bbox_inches="tight")
fig2.savefig(OUT_DIR / "fig_cascade_power.pdf", bbox_inches="tight")
print(f"Saved {OUT_DIR / 'fig_cascade_power.png'}")

# ── Figure 3: cascade cumulative work ────────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=False,
                           constrained_layout=True)
for j, R in enumerate((LV_R, RV_R)):
    ax = axes[j]
    c = cascade[R]
    for key, style in LEVEL_STYLE.items():
        cum = to_kPa(np.cumsum(c[key]), R)
        ax.plot(t, cum, **style)
    ax.axhline(0, color="lightgray", lw=0.6)
    ax.set_xlabel(r"$t$ (s)", fontsize=11)
    if j == 0:
        ax.set_ylabel(r"$\int \dot W\,dt / V$ (kPa)", fontsize=12)
    ax.grid(alpha=0.25)
    ax.set_title("LV" if j == 0 else "RV", fontsize=13, fontweight="bold")
    lines, labels = ax.get_legend_handles_labels()
    new_labels = []
    for line in lines:
        for key, style in LEVEL_STYLE.items():
            if style["color"] == line.get_color():
                tot = to_kPa(cascade[R][key].sum(), R)
                new_labels.append(f"{style['label']}  ({tot:+.2f} kPa)")
                break
    ax.legend(lines, new_labels, loc="center right", fontsize=9,
              framealpha=0.95, handlelength=1.6)

fig3.suptitle(f"Cumulative work per region  ({PRIMARY} tau split)",
              fontsize=12, fontweight="bold")
fig3.savefig(OUT_DIR / "fig_cascade_cumulative.png", dpi=160,
             bbox_inches="tight")
fig3.savefig(OUT_DIR / "fig_cascade_cumulative.pdf", bbox_inches="tight")
print(f"Saved {OUT_DIR / 'fig_cascade_cumulative.png'}")

# ── Figure 4: bar chart of total work per level per region ───────────────────
levels = ["W0_per_step", "W2_per_step", "W3_per_step", "W4_per_step"]
level_labels = ["$\\mathbf{S}{:}\\dot{\\mathbf{E}}$",
                "$S_{ff}\\,\\dot E_{ff}$",
                "$P\\,\\dot E_{ff}$",
                "$P\\,\\dot\\varepsilon_{ll}$"]
totals = {R: np.array([to_kPa(cascade[R][k].sum(), R) for k in levels])
          for R in (LV_R, RV_R)}

fig4, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
x = np.arange(len(levels))
w = 0.36
ax.bar(x - w / 2, totals[LV_R], w, color="#1f77b4", label="LV territory",
       edgecolor="white")
ax.bar(x + w / 2, totals[RV_R], w, color="#d62728", label="RV territory",
       edgecolor="white")
ax.axhline(0, color="lightgray", lw=0.6)
for i, key in enumerate(levels):
    ax.text(x[i] - w / 2, totals[LV_R][i],
            f"{totals[LV_R][i]:+.2f}", ha="center",
            va="bottom" if totals[LV_R][i] >= 0 else "top",
            fontsize=8, color="#1f77b4")
    ax.text(x[i] + w / 2, totals[RV_R][i],
            f"{totals[RV_R][i]:+.2f}", ha="center",
            va="bottom" if totals[RV_R][i] >= 0 else "top",
            fontsize=8, color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(level_labels, fontsize=11)
ax.set_ylabel("Net work density per cycle (kPa = mJ/mL)", fontsize=11)
ax.set_title("Cascade totals", fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
ax.grid(alpha=0.25, axis="y")
fig4.savefig(OUT_DIR / "fig_cascade_totals.png", dpi=160, bbox_inches="tight")
fig4.savefig(OUT_DIR / "fig_cascade_totals.pdf", bbox_inches="tight")
print(f"Saved {OUT_DIR / 'fig_cascade_totals.png'}")

# ── Figure 5: cascade loops grid ─────────────────────────────────────────────
# 2 rows (LV, RV) × 3 cols (L1..L3 after dropping the deviatoric step).
# Level 0 has no loop by design — it's the time series we already showed in
# Figure 2. Each loop traces the (strain, stress) trajectory through one
# cardiac cycle. LV rows coloured blue, RV rows red.
PA_TO_KPA = 1.0 / 1000.0

LOOP_LEVELS = [
    ("L1", r"$E_{ff}$",          r"$S_{ff}$ (kPa)"),
    ("L2", r"$E_{ff}$",          r"$P_{cav}$ (kPa)"),
    ("L3", r"$\varepsilon_{ll}$", r"$P_{cav}$ (kPa)"),
]

ROW_COLORS = ("#1f77b4", "#d62728")  # LV blue, RV red

fig5, axes = plt.subplots(2, 3, figsize=(12, 7.0), constrained_layout=True)
for row_i, R in enumerate((LV_R, RV_R)):
    c = cascade[R]
    color = ROW_COLORS[row_i]
    series = {
        "L1": (c["E_ff"],          c["S_ff"] * PA_TO_KPA),
        "L2": (c["E_ff"],          c["P_pa"] * PA_TO_KPA),
        "L3": (c["E_ll"],          c["P_pa"] * PA_TO_KPA),
    }
    for col_i, (level, xlabel, ylabel) in enumerate(LOOP_LEVELS):
        ax = axes[row_i, col_i]
        x, y = series[level]
        ax.plot(x, y, color=color, lw=1.8, alpha=0.95)
        ax.set_xlabel(xlabel, fontsize=11)
        if col_i == 0:
            ax.set_ylabel(f"{'LV' if row_i == 0 else 'RV'}\n{ylabel}",
                          fontsize=11)
        else:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.25)
        ax.axhline(0, color="lightgray", lw=0.5, zorder=0)
        ax.axvline(0, color="lightgray", lw=0.5, zorder=0)
fig5.savefig(OUT_DIR / "fig_cascade_loops.png", dpi=160, bbox_inches="tight")
fig5.savefig(OUT_DIR / "fig_cascade_loops.pdf", bbox_inches="tight")
print(f"Saved {OUT_DIR / 'fig_cascade_loops.png'}")

# ── Figure 6: robustness of the cascade to the tau-split definition ──────────
# Totals at each level under both tau flavours (Euclidean and Laplace).
# If the bars match to within a few percent, the cascade story is robust
# to how we split LV and RV territories — exactly the robustness argument
# we used for the septum sweep, but now for the LV/RV partition.
fig6, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
flavour_labels = {"eu": "Euclidean", "lap": "Laplace"}
bar_positions = []
bar_values = []
bar_colors = []
bar_labels = []
x_ticks = []
x_tick_labels = []

group_width = 0.38
cluster_width = 4 * group_width + 0.6  # 4 bars per cluster (2 sides × 2 flavours), spacer
cluster_x = 0.0
for lvl_idx, key in enumerate(levels):
    x_ticks.append(cluster_x + 1.5 * group_width)
    x_tick_labels.append(level_labels[lvl_idx])
    for side_i, (side, color) in enumerate([("LV", "#1f77b4"), ("RV", "#d62728")]):
        for fl_i, (fl, _) in enumerate([("eu", "o"), ("lap", "/")]):
            R = f"{side}_tau_{fl}"
            val = to_kPa(cascade[R][key].sum(), R)
            bx = cluster_x + (side_i * 2 + fl_i) * group_width
            hatch = "" if fl == "eu" else "//"
            ax.bar(bx, val, group_width * 0.92, color=color,
                   edgecolor="white", hatch=hatch, alpha=0.88)
    cluster_x += cluster_width

ax.axhline(0, color="lightgray", lw=0.6)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels, fontsize=10)
ax.set_ylabel("Net work density per cycle (kPa)", fontsize=11)
ax.set_title("Cascade robustness — Eu vs Lap tau-split comparison  "
             "(solid = Euclidean, hatched = Laplace)", fontsize=11,
             fontweight="bold")

# Custom legend: one entry per (side, flavour)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#1f77b4", edgecolor="white", label="LV territory (Eu)"),
    Patch(facecolor="#1f77b4", edgecolor="white", hatch="//", label="LV territory (Lap)"),
    Patch(facecolor="#d62728", edgecolor="white", label="RV territory (Eu)"),
    Patch(facecolor="#d62728", edgecolor="white", hatch="//", label="RV territory (Lap)"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
          framealpha=0.95)
ax.grid(alpha=0.25, axis="y")

# Print quantitative agreement at the bottom
print("\nCascade totals — Eu vs Lap comparison (kPa density):")
print(f"{'level':<10} {'LV_eu':>8} {'LV_lap':>8} {'Δ(%)':>6}   "
      f"{'RV_eu':>8} {'RV_lap':>8} {'Δ(%)':>6}")
for key, lbl in zip(levels, ["L0","L1","L2","L3","L4"]):
    lv_eu  = to_kPa(cascade["LV_tau_eu"][key].sum(),  "LV_tau_eu")
    lv_lap = to_kPa(cascade["LV_tau_lap"][key].sum(), "LV_tau_lap")
    rv_eu  = to_kPa(cascade["RV_tau_eu"][key].sum(),  "RV_tau_eu")
    rv_lap = to_kPa(cascade["RV_tau_lap"][key].sum(), "RV_tau_lap")
    lv_d = 100 * (lv_eu - lv_lap) / abs(lv_lap) if abs(lv_lap) > 1e-9 else 0
    rv_d = 100 * (rv_eu - rv_lap) / abs(rv_lap) if abs(rv_lap) > 1e-9 else 0
    print(f"{lbl:<10} {lv_eu:>+8.2f} {lv_lap:>+8.2f} {lv_d:>+5.1f}%   "
          f"{rv_eu:>+8.2f} {rv_lap:>+8.2f} {rv_d:>+5.1f}%")

fig6.savefig(OUT_DIR / "fig_cascade_tau_comparison.png", dpi=160,
             bbox_inches="tight")
fig6.savefig(OUT_DIR / "fig_cascade_tau_comparison.pdf", bbox_inches="tight")
print(f"Saved {OUT_DIR / 'fig_cascade_tau_comparison.png'}")

# ── Figure 7: pressure-convention comparison (correct vs wrong cavity) ───────
# For each territory, plot L3 and L4 using the "correct" cavity pressure
# (the one adjacent to the wall) vs the "wrong" one (the opposite cavity).
# For the RV territory the "wrong" row IS the clinical default (everyone
# uses P_LV by habit) — so this figure explicitly measures how misleading
# that convention is.
fig7, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2),
                                 constrained_layout=True, sharey=False)

LV_R_lap = "LV_tau_lap"
RV_R_lap = "RV_tau_lap"

def total_kPa(R, key):
    return to_kPa(cascade[R][key].sum(), R)

def pct_of_L0(R, key):
    return 100.0 * cascade[R][key].sum() / cascade[R]["W0_per_step"].sum()

# Panel A: bars comparing correct vs wrong for L3 and L4
width = 0.32
xs = np.arange(4)  # LV/L3, LV/L4, RV/L3, RV/L4
correct_vals = [
    total_kPa(LV_R_lap, "W3_per_step"),
    total_kPa(LV_R_lap, "W4_per_step"),
    total_kPa(RV_R_lap, "W3_per_step"),
    total_kPa(RV_R_lap, "W4_per_step"),
]
wrong_vals = [
    total_kPa(LV_R_lap, "W3_wrong_per_step"),
    total_kPa(LV_R_lap, "W4_wrong_per_step"),
    total_kPa(RV_R_lap, "W3_wrong_per_step"),
    total_kPa(RV_R_lap, "W4_wrong_per_step"),
]
l0_vals = [
    total_kPa(LV_R_lap, "W0_per_step"),
    total_kPa(LV_R_lap, "W0_per_step"),
    total_kPa(RV_R_lap, "W0_per_step"),
    total_kPa(RV_R_lap, "W0_per_step"),
]

labels = [
    r"LV $\cdot$ L3" + "\n$P\\cdot\\dot E_{ff}$",
    r"LV $\cdot$ L4" + "\n$P\\cdot\\dot\\varepsilon_{ll}$",
    r"RV $\cdot$ L3" + "\n$P\\cdot\\dot E_{ff}$",
    r"RV $\cdot$ L4" + "\n$P\\cdot\\dot\\varepsilon_{ll}$",
]

axA.bar(xs - width, l0_vals,      width, color="black",    alpha=0.35,
        edgecolor="white", label=r"$W_{true}$ (gold std, both)")
axA.bar(xs,         correct_vals, width, color="#2ca02c",
        edgecolor="white",
        label=r"correct $P$ (LV→$P_{LV}$, RV→$P_{RV}$)")
axA.bar(xs + width, wrong_vals,   width, color="#d62728",
        edgecolor="white",
        label=r"wrong $P$ (swap)  →  RV case = clinical default")
axA.axhline(0, color="lightgray", lw=0.6)
axA.set_xticks(xs)
axA.set_xticklabels(labels, fontsize=9)
axA.set_ylabel("Net work density per cycle (kPa)", fontsize=11)
axA.set_title("Absolute magnitudes", fontsize=11, fontweight="bold")
axA.legend(loc="lower right", fontsize=8, framealpha=0.95)
axA.grid(alpha=0.25, axis="y")

# Panel B: same data as fraction of W_true
pct_correct = [100 * c / l for c, l in zip(correct_vals, l0_vals)]
pct_wrong   = [100 * w / l for w, l in zip(wrong_vals, l0_vals)]
axB.bar(xs - width/2, pct_correct, width, color="#2ca02c",
        edgecolor="white", label="correct cavity")
axB.bar(xs + width/2, pct_wrong,   width, color="#d62728",
        edgecolor="white", label="wrong cavity")
axB.axhline(100, color="black", lw=1.0, ls="--", alpha=0.5)
axB.axhline(0,   color="lightgray", lw=0.6)
axB.set_xticks(xs)
axB.set_xticklabels(labels, fontsize=9)
axB.set_ylabel("Proxy captures X % of $W_{true}$", fontsize=11)
axB.set_title("Fraction of true work captured", fontsize=11, fontweight="bold")
axB.legend(loc="upper left", fontsize=8, framealpha=0.95)
axB.grid(alpha=0.25, axis="y")

# Annotate each bar with its value
for ax, vals, bars, color in [(axA, l0_vals, xs - width, "black"),
                               (axA, correct_vals, xs, "#2ca02c"),
                               (axA, wrong_vals, xs + width, "#d62728")]:
    for i, v in enumerate(vals):
        ax.text(bars[i], v, f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, color=color)
for i, (p, w) in enumerate(zip(pct_correct, pct_wrong)):
    axB.text(xs[i] - width/2, p, f"{p:.0f}%",
             ha="center", va="bottom", fontsize=8, color="#2ca02c")
    axB.text(xs[i] + width/2, w, f"{w:.0f}%",
             ha="center", va="bottom", fontsize=8, color="#d62728")

fig7.suptitle("Does the pressure choice matter? Clinical default (P_LV everywhere) "
              "vs cavity-adjacent convention",
              fontsize=12, fontweight="bold")
fig7.savefig(OUT_DIR / "fig_cascade_pressure_convention.png", dpi=160,
             bbox_inches="tight")
fig7.savefig(OUT_DIR / "fig_cascade_pressure_convention.pdf", bbox_inches="tight")
print(f"Saved {OUT_DIR / 'fig_cascade_pressure_convention.png'}")

# Console summary
print("\nPressure-convention comparison (Laplace tau split, % of gold standard):")
print(f"{'case':<30} {'correct':>10} {'wrong':>10}   {'notes'}")
for i, lbl in enumerate([
    "LV territory  · L3 (P × dE_ff)",
    "LV territory  · L4 (P × dE_ll)",
    "RV territory  · L3 (P × dE_ff)  ← default clinical applies here",
    "RV territory  · L4 (P × dE_ll)  ← default clinical applies here",
]):
    note = ("   clinical default=WRONG" if "clinical" in lbl else "")
    print(f"{lbl[:30]:<30} {pct_correct[i]:>+9.1f}% {pct_wrong[i]:>+9.1f}%{note}")

# ── Save raw cascade data for downstream ─────────────────────────────────────
save = {"time": t, "dt": dt}
for R in ("LV_tau_eu", "RV_tau_eu", "LV_tau_lap", "RV_tau_lap"):
    for k in ["W0_per_step", "W1_per_step", "W2_per_step",
              "W3_per_step", "W4_per_step",
              "W3_wrong_per_step", "W4_wrong_per_step",
              "S_ff", "S_ss", "S_nn", "E_ff", "E_ss", "E_nn", "E_ll",
              "P_pa", "P_pa_wrong"]:
        save[f"{R}_{k}"] = cascade[R][k]
save["energy_int_whole"] = W_int_whole
save["energy_bnd_whole"] = W_bnd_total
np.savez(OUT_DIR / "cascade_raw.npz", **save)
print(f"Saved {OUT_DIR / 'cascade_raw.npz'}")
print("\nDone.")
