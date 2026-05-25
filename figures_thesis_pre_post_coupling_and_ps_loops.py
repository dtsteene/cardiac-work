#!/usr/bin/env python3
"""Thesis figures: pre/post-coupling PV loops, and PS-loop sweep.

Figure A (fig_4_6_pv_loops_pre_post_coupling.png):
    2x2 panel grid. Columns = LV | RV. Rows = baseline + severe case.
    Each panel overlays the standalone-0D PV loop (from circulation/preload_history.npy)
    against the FEM-coupled PV loop. The coupled curve uses the volumes from the
    coupled 0D state history (ode_state_history.npy at the case root, or
    circulation/history.npy when present) and pairs them with the cavity Lagrange
    multiplier pressures from solver/solver_cavity_pressure_mmHg.npy, which are
    the formally correct cavity pressures for the FEM-coupled system. Pressure in
    mmHg, volume in mL. Title shows the achieved peak RV systolic pressure (FEM,
    coupled) computed directly from solver_cavity_pressure_mmHg.npy.

Figure B (fig_5_0c_ps_loops_sweep.png): DEPRECATED 2026-05-14 — superseded
    by Figure C (fig_5_0c_cascade_loops_sweep.png). The old three-panel proxy-only
    figure is preserved here only as figure_B(); it is no longer wired into main().

Figure C (fig_5_0c_cascade_loops_sweep.png):
    2x3 panel grid showing the FE-vs-proxy cascade side by side across all 16
    sweep cases. Top row: fibre-direction stress-strain loops S_ff vs E_ff for
    LV free wall, RV free wall, and septum (FE reference). Bottom row: clinical
    pressure-strain proxy p_cav vs epsilon_ll for the same three regions
    (p_LV for LV and septum, p_RV for RV). All 16 traces overlaid per panel,
    coloured by achieved peak RV systolic pressure with the coolwarm colormap
    (vmin~32 mmHg, vmax~100 mmHg so white sits near the mid-pressure case).
    A single shared colorbar on the right.

Canonical sweep: 5 mmHg RV-EDP capped per-case-unloading production run at
/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-05-10/capped_shared_l5_20260510_141015/
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


SWEEP_ROOT = Path(
    "/global/D1/homes/dtsteene/cardiac-work/results/sims/2026-05-10/"
    "capped_shared_l5_20260510_141015"
)
THESIS_FIG = Path("/home/dtsteene/D1/RV/figures")
CASE_NAMES = [
    "sPAP22", "sPAP25", "sPAP30", "sPAP35", "sPAP45", "sPAP50",
    "sPAP55", "sPAP60", "sPAP65", "sPAP70", "sPAP75", "sPAP80",
    "sPAP85", "sPAP87", "sPAP92", "sPAP95",
]
BEAT_S = 0.8  # 75 BPM -> 0.8 s period


# ---------- I/O helpers ----------

def load_coupled_history(case_dir: Path) -> dict:
    """Coupled FEM-0D state. Prefer circulation/history.npy when present, else
    fall back to the case-root ode_state_history.npy (same dict structure)."""
    hp = case_dir / "circulation" / "history.npy"
    if not hp.exists():
        hp = case_dir / "ode_state_history.npy"
    return np.load(hp, allow_pickle=True).item()


def load_standalone_preload(case_dir: Path) -> dict:
    return np.load(case_dir / "circulation" / "preload_history.npy",
                   allow_pickle=True).item()


def load_solver_cavity_pressure(case_dir: Path) -> np.ndarray:
    """Returns array of shape (N, 2) in mmHg: columns [p_LV, p_RV]."""
    return np.load(case_dir / "solver" / "solver_cavity_pressure_mmHg.npy")


def load_last_beat_metrics(case_dir: Path) -> dict:
    return np.load(case_dir / "analysis" / "last_beat" / "metrics_downsample_1.npy",
                   allow_pickle=True).item()


def case_peak_rv_systolic_mmHg(case_dir: Path) -> float:
    """Peak RV systolic pressure (mmHg) from the FEM cavity LM in the last beat."""
    sp = load_solver_cavity_pressure(case_dir)  # (N, 2) [p_LV, p_RV]
    # last beat = last 0.8 s of samples; assume dt=0.001 -> 800 samples
    n_last = min(int(round(BEAT_S / 0.001)), sp.shape[0])
    return float(sp[-n_last:, 1].max())


def case_peak_lv_systolic_mmHg(case_dir: Path) -> float:
    sp = load_solver_cavity_pressure(case_dir)
    n_last = min(int(round(BEAT_S / 0.001)), sp.shape[0])
    return float(sp[-n_last:, 0].max())


# ---------- Figure A: pre/post coupling PV loops ----------

def _close_loop(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.append(x, x[0]), np.append(y, y[0])


def _last_beat_idx(time: np.ndarray, period_s: float = BEAT_S) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    mask = t >= t[-1] - period_s
    if mask.sum() < 50:
        mask = np.ones_like(t, dtype=bool)
    return mask


def plot_pv_panel(
    ax: plt.Axes,
    case_dir: Path,
    side: str,  # "LV" or "RV"
    title: str,
) -> None:
    v_key = f"V_{side}"

    coupled = load_coupled_history(case_dir)
    standalone = load_standalone_preload(case_dir)
    solver_p = load_solver_cavity_pressure(case_dir)  # mmHg, (N,2) [p_LV,p_RV]

    # --- coupled (FEM cavity LM pressure paired with 0D volumes) ---
    t_c = np.asarray(coupled["time"], dtype=float)
    v_c = np.asarray(coupled[v_key], dtype=float)
    # Solver pressure has shape (N,2). Number of samples may be N-1 vs time N.
    # We treat solver_p as aligned with t_c[1:] (post-step values).
    p_col = 0 if side == "LV" else 1
    p_c_full = solver_p[:, p_col]
    # Trim time/volume to match solver_p length
    n = min(len(t_c), len(p_c_full))
    t_c_use = t_c[-n:]
    v_c_use = v_c[-n:]
    p_c_use = p_c_full[-n:]
    mask_c = _last_beat_idx(t_c_use, BEAT_S)
    vc, pc = _close_loop(v_c_use[mask_c], p_c_use[mask_c])

    # --- standalone (preload) ---
    t_s = np.asarray(standalone["time"], dtype=float)
    v_s = np.asarray(standalone[v_key], dtype=float)
    p_s = np.asarray(standalone[f"p_{side}"], dtype=float)  # already mmHg in 0D
    mask_s = _last_beat_idx(t_s, BEAT_S)
    vs, ps = _close_loop(v_s[mask_s], p_s[mask_s])

    ax.plot(
        vs, ps,
        color="#888888", linewidth=1.8, linestyle="--",
        label="standalone 0D (preload)",
    )
    ax.plot(
        vc, pc,
        color=("#2B6CB0" if side == "LV" else "#C53030"),
        linewidth=2.4, label="FEM-coupled",
    )
    ax.set_xlabel(f"V_{side} (mL)")
    ax.set_ylabel(f"p_{side} (mmHg)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_A(baseline_case: str, severe_case: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))

    for row_idx, name in enumerate([baseline_case, severe_case]):
        cdir = SWEEP_ROOT / name
        rv_peak = case_peak_rv_systolic_mmHg(cdir)
        label = "baseline" if name == baseline_case else "severe"

        plot_pv_panel(
            axes[row_idx, 0], cdir, "LV",
            f"{label} LV: peak RV systolic = {rv_peak:.0f} mmHg",
        )
        plot_pv_panel(
            axes[row_idx, 1], cdir, "RV",
            f"{label} RV: peak RV systolic = {rv_peak:.0f} mmHg",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncols=2, frameon=False, fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])

    out_png = THESIS_FIG / "fig_4_6_pv_loops_pre_post_coupling.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------- Spectrum PV-loop family figures (standalone + coupled) ----------

def _pv_loops_spectrum_panel(
    ax_lv,
    ax_rv,
    get_lv_loop,
    get_rv_loop,
    rvsp_by_case,
    cmap,
    norm,
):
    """Render one 1x2 spectrum figure given case-to-loop callables."""
    for name in CASE_NAMES:
        color = cmap(norm(rvsp_by_case[name]))
        v_lv, p_lv = get_lv_loop(name)
        v_rv, p_rv = get_rv_loop(name)
        v_lv_c, p_lv_c = _close_loop(v_lv, p_lv)
        v_rv_c, p_rv_c = _close_loop(v_rv, p_rv)
        ax_lv.plot(v_lv_c, p_lv_c, color=color, linewidth=1.35, alpha=0.95)
        ax_rv.plot(v_rv_c, p_rv_c, color=color, linewidth=1.35, alpha=0.95)
    for ax, side in [(ax_lv, "LV"), (ax_rv, "RV")]:
        ax.set_xlabel(f"{side} volume (mL)")
        ax.set_ylabel(f"{side} pressure (mmHg)")
        ax.set_title(side)
        ax.grid(alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def figure_pv_loops_standalone_spectrum() -> Path:
    """fig 4.2: standalone 0D PV-loop family across the 16-case sweep.

    Volumes and pressures both come from the standalone 0D pre-run's last
    beat (circulation/preload_history.npy). This is what the optimizer's
    calibrated parameters produce *before* the FEM-coupled run touches the
    cavity. Coloured by achieved peak RV systolic pressure (coupled) so each
    case has the same colour across all chapter figures.
    """
    rvsp = {n: case_peak_rv_systolic_mmHg(SWEEP_ROOT / n) for n in CASE_NAMES}
    rvsp_vals = np.array(list(rvsp.values()))
    norm = mcolors.Normalize(rvsp_vals.min(), rvsp_vals.max())
    cmap = cm.coolwarm

    def get_lv(name):
        s = load_standalone_preload(SWEEP_ROOT / name)
        t = np.asarray(s["time"], dtype=float)
        last = _segment_beats(t)[-1]
        return (np.asarray(s["V_LV"], dtype=float)[last],
                np.asarray(s["p_LV"], dtype=float)[last])

    def get_rv(name):
        s = load_standalone_preload(SWEEP_ROOT / name)
        t = np.asarray(s["time"], dtype=float)
        last = _segment_beats(t)[-1]
        return (np.asarray(s["V_RV"], dtype=float)[last],
                np.asarray(s["p_RV"], dtype=float)[last])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), constrained_layout=True)
    _pv_loops_spectrum_panel(axes[0], axes[1], get_lv, get_rv, rvsp, cmap, norm)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.86)
    cbar.set_label("achieved RV systolic pressure (mmHg)")

    out_png = THESIS_FIG / "fig_4_2_pv_loops_spectrum.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def figure_pv_loops_standalone_vs_coupled_spectrum() -> Path:
    """fig 4.2 (combined 2x2): standalone 0D and FEM-coupled PV-loop families
    side by side, all 16 cases, same colour mapping.

    Top row: standalone 0D (preload_history.npy last beat).
    Bottom row: FEM-coupled (FEM cavity Lagrange multiplier paired with
    coupled circulation history last beat).
    """
    rvsp = {n: case_peak_rv_systolic_mmHg(SWEEP_ROOT / n) for n in CASE_NAMES}
    rvsp_vals = np.array(list(rvsp.values()))
    norm = mcolors.Normalize(rvsp_vals.min(), rvsp_vals.max())
    cmap = cm.coolwarm

    def get_standalone(name, side):
        s = load_standalone_preload(SWEEP_ROOT / name)
        t = np.asarray(s["time"], dtype=float)
        last = _segment_beats(t)[-1]
        return (np.asarray(s[f"V_{side}"], dtype=float)[last],
                np.asarray(s[f"p_{side}"], dtype=float)[last])

    def get_coupled(name, side, p_col):
        cdir = SWEEP_ROOT / name
        c = load_coupled_history(cdir)
        sp = load_solver_cavity_pressure(cdir)
        t = np.asarray(c["time"], dtype=float)
        last = _segment_beats(t)[-1]
        last_p = last[last < sp.shape[0]]
        return (np.asarray(c[f"V_{side}"], dtype=float)[last_p],
                sp[last_p, p_col])

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4),
                             constrained_layout=True)
    _pv_loops_spectrum_panel(
        axes[0, 0], axes[0, 1],
        lambda n: get_standalone(n, "LV"),
        lambda n: get_standalone(n, "RV"),
        rvsp, cmap, norm,
    )
    _pv_loops_spectrum_panel(
        axes[1, 0], axes[1, 1],
        lambda n: get_coupled(n, "LV", 0),
        lambda n: get_coupled(n, "RV", 1),
        rvsp, cmap, norm,
    )
    axes[0, 0].set_title("LV — standalone 0D")
    axes[0, 1].set_title("RV — standalone 0D")
    axes[1, 0].set_title("LV — FEM-coupled")
    axes[1, 1].set_title("RV — FEM-coupled")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.78, location="right")
    cbar.set_label("achieved RV systolic pressure (mmHg)")

    out_png = THESIS_FIG / "fig_4_2_pv_loops_spectrum.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def figure_pv_loops_coupled_spectrum() -> Path:
    """fig 4.2b: FEM-coupled PV-loop family across the 16-case sweep.

    Volumes from circulation/history.npy last beat; pressures from the FEM
    cavity Lagrange multiplier (solver/solver_cavity_pressure_mmHg.npy) last
    beat. Same colour mapping and visual style as fig 4.2 for direct
    standalone-vs-coupled comparison.
    """
    rvsp = {n: case_peak_rv_systolic_mmHg(SWEEP_ROOT / n) for n in CASE_NAMES}
    rvsp_vals = np.array(list(rvsp.values()))
    norm = mcolors.Normalize(rvsp_vals.min(), rvsp_vals.max())
    cmap = cm.coolwarm

    def _last_beat_idx_in_coupled(cdir):
        c = load_coupled_history(cdir)
        t = np.asarray(c["time"], dtype=float)
        last = _segment_beats(t)[-1]
        sp = load_solver_cavity_pressure(cdir)
        last_p = last[last < sp.shape[0]]
        return c, sp, last_p

    def get_lv(name):
        cdir = SWEEP_ROOT / name
        c, sp, last_p = _last_beat_idx_in_coupled(cdir)
        return (np.asarray(c["V_LV"], dtype=float)[last_p], sp[last_p, 0])

    def get_rv(name):
        cdir = SWEEP_ROOT / name
        c, sp, last_p = _last_beat_idx_in_coupled(cdir)
        return (np.asarray(c["V_RV"], dtype=float)[last_p], sp[last_p, 1])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), constrained_layout=True)
    _pv_loops_spectrum_panel(axes[0], axes[1], get_lv, get_rv, rvsp, cmap, norm)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.86)
    cbar.set_label("achieved RV systolic pressure (mmHg)")

    out_png = THESIS_FIG / "fig_4_2b_pv_loops_coupled_spectrum.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------- Coupling-residual visual walkthrough (4 new figures) ----------

def _segment_beats(t: np.ndarray, period_s: float = BEAT_S):
    """Return list of integer-index arrays, one per complete beat."""
    t = np.asarray(t, dtype=float)
    n_beats = int(round((t[-1] - t[0]) / period_s))
    beats = []
    for i in range(n_beats):
        t_start = t[0] + i * period_s
        t_end = t_start + period_s
        mask = (t >= t_start - 1e-9) & (t < t_end - 1e-9)
        idx = np.where(mask)[0]
        if idx.size >= 50:
            beats.append(idx)
    return beats


def _plot_beat_pv(ax, v, p, color, alpha, linewidth=1.4, linestyle="-"):
    vv, pp = _close_loop(v, p)
    ax.plot(vv, pp, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)


def figure_standalone_convergence(case_name: str, skip_first: int = 0) -> Path:
    """Fig 4.6: standalone 0D PV-loop convergence over preload beats.

    Single case. LV | RV panels. Each completed beat plotted as one PV loop,
    coloured by beat order using a sequential colormap (early beats light,
    later beats dark), with the final beat drawn bold in the side colour.

    The first ``skip_first`` beats are dropped as optimizer warm-up — they
    are pure startup wobble before the calibrated parameters lock the model
    into its limit cycle and are not informative for the convergence story.
    """
    cdir = SWEEP_ROOT / case_name
    standalone = load_standalone_preload(cdir)
    t_s = np.asarray(standalone["time"], dtype=float)
    beats_all = _segment_beats(t_s)
    beats = beats_all[skip_first:]
    n_beats = len(beats)
    rv_peak = case_peak_rv_systolic_mmHg(cdir)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    sides = [("LV", "#1B4F8B", cm.Blues), ("RV", "#9B2A2A", cm.Reds)]

    for ax, (side, color_last, cmap_side) in zip(axes, sides):
        v = np.asarray(standalone[f"V_{side}"], dtype=float)
        p = np.asarray(standalone[f"p_{side}"], dtype=float)
        # Y-axis range from the converged tail (avoids early-transient overshoot)
        tail_p = np.concatenate([p[idx] for idx in beats[-5:]])
        tail_v = np.concatenate([v[idx] for idx in beats[-5:]])
        for j, idx in enumerate(beats):
            is_last = j == n_beats - 1
            if is_last:
                _plot_beat_pv(ax, v[idx], p[idx], color_last, alpha=1.0,
                              linewidth=2.6)
            else:
                # Map j in [0, n_beats-2] to colormap range [0.25, 0.78]
                cmap_pos = 0.25 + 0.53 * (j / max(1, n_beats - 2))
                _plot_beat_pv(ax, v[idx], p[idx], cmap_side(cmap_pos),
                              alpha=0.85, linewidth=1.0)
        ax.set_xlabel(f"V_{side} (mL)")
        ax.set_ylabel(f"p_{side} (mmHg)")
        if skip_first > 0:
            title = (
                f"{side}: standalone 0D, beats "
                f"{len(beats_all) - n_beats + 1}–{len(beats_all)}"
            )
        else:
            title = f"{side}: standalone 0D, {n_beats} beats"
        ax.set_title(title, fontsize=11)
        ax.set_ylim(min(-2, tail_p.min() - 5), tail_p.max() + 12)
        ax.set_xlim(tail_v.min() - 3, tail_v.max() + 5)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Standalone 0D convergence — {case_name}, "
        f"achieved peak RV systolic = {rv_peak:.0f} mmHg (coupled)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out_png = THESIS_FIG / "fig_4_6_standalone_convergence.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def figure_coupling_jump(case_name: str) -> Path:
    """Fig 4.7: operating-point shift between converged standalone and
    coupled limit cycles.

    Standalone-0D last beat (grey dashed) vs FEM-coupled last beat (bold).
    LV | RV panels. Pure steady-state comparison; the transient that takes
    the system from one to the other is shown in figure 4.8.
    """
    cdir = SWEEP_ROOT / case_name
    standalone = load_standalone_preload(cdir)
    coupled = load_coupled_history(cdir)
    solver_p = load_solver_cavity_pressure(cdir)
    rv_peak = case_peak_rv_systolic_mmHg(cdir)

    t_s = np.asarray(standalone["time"], dtype=float)
    s_last = _segment_beats(t_s)[-1]

    t_c = np.asarray(coupled["time"], dtype=float)
    c_last = _segment_beats(t_c)[-1]
    sp_last = c_last[c_last < solver_p.shape[0]]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    sides = [("LV", "#1B4F8B", 0), ("RV", "#9B2A2A", 1)]

    for ax, (side, color, p_col) in zip(axes, sides):
        v_s = np.asarray(standalone[f"V_{side}"], dtype=float)
        p_s = np.asarray(standalone[f"p_{side}"], dtype=float)
        _plot_beat_pv(ax, v_s[s_last], p_s[s_last], "#888888", alpha=0.95,
                      linewidth=1.8, linestyle="--")

        v_c = np.asarray(coupled[f"V_{side}"], dtype=float)
        p_c = solver_p[:, p_col]
        _plot_beat_pv(ax, v_c[sp_last], p_c[sp_last], color, alpha=1.0,
                      linewidth=2.4)

        ax.set_xlabel(f"V_{side} (mL)")
        ax.set_ylabel(f"p_{side} (mmHg)")
        ax.set_title(f"{side}: standalone vs coupled, converged",
                     fontsize=11)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], color="#888888", linestyle="--", linewidth=1.8,
               label="standalone 0D (converged)"),
        Line2D([], [], color="#444444", linewidth=2.4,
               label="FEM-coupled (converged)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncols=2, frameon=False,
               fontsize=10)
    fig.suptitle(
        f"Operating-point shift at FEM coupling — {case_name}, "
        f"achieved peak RV systolic = {rv_peak:.0f} mmHg",
        fontsize=11, y=1.02,
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    out_png = THESIS_FIG / "fig_4_7_coupling_jump.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def figure_coupled_convergence(case_name: str) -> Path:
    """Fig 4.8: FEM-coupled PV-loop convergence over the simulated beats.

    Single case. LV | RV panels. Each completed coupled beat plotted using the
    FEM cavity Lagrange-multiplier pressure paired with the 0D-side volume,
    coloured by beat order using a sequential colormap (early beats light,
    later beats dark), with the final beat drawn bold in the side colour.
    """
    cdir = SWEEP_ROOT / case_name
    coupled = load_coupled_history(cdir)
    solver_p = load_solver_cavity_pressure(cdir)
    t_c = np.asarray(coupled["time"], dtype=float)
    beats = _segment_beats(t_c)
    n_beats = len(beats)
    rv_peak = case_peak_rv_systolic_mmHg(cdir)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    sides = [("LV", "#1B4F8B", 0, cm.Blues), ("RV", "#9B2A2A", 1, cm.Reds)]

    for ax, (side, color_last, p_col, cmap_side) in zip(axes, sides):
        v = np.asarray(coupled[f"V_{side}"], dtype=float)
        p = solver_p[:, p_col]
        for j, idx in enumerate(beats):
            idx_p = idx[idx < solver_p.shape[0]]
            is_last = j == n_beats - 1
            if is_last:
                _plot_beat_pv(ax, v[idx_p], p[idx_p], color_last, alpha=1.0,
                              linewidth=2.6)
            else:
                cmap_pos = 0.25 + 0.53 * (j / max(1, n_beats - 2))
                _plot_beat_pv(ax, v[idx_p], p[idx_p], cmap_side(cmap_pos),
                              alpha=0.9, linewidth=1.3)
        ax.set_xlabel(f"V_{side} (mL)")
        ax.set_ylabel(f"p_{side} (mmHg)")
        ax.set_title(f"{side}: FEM-coupled, {n_beats} beats", fontsize=11)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"FEM-coupled convergence — {case_name}, "
        f"achieved peak RV systolic = {rv_peak:.0f} mmHg",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out_png = THESIS_FIG / "fig_4_8_coupled_convergence.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def figure_all_16_pre_post() -> Path:
    """Fig 4.9: all 16 cases — standalone 0D vs FEM-coupled PV loops.

    4x4 panel grid. Each panel overlays LV (blue) and RV (red) PV loops:
      - standalone 0D last beat: light tone, dashed
      - FEM-coupled last beat: solid bold
    Panels ordered by achieved peak RV systolic pressure (FEM, coupled),
    low → high, row-major.
    """
    rvsp = {n: case_peak_rv_systolic_mmHg(SWEEP_ROOT / n) for n in CASE_NAMES}
    ordered = sorted(CASE_NAMES, key=lambda n: rvsp[n])

    fig, axes = plt.subplots(4, 4, figsize=(15.0, 13.0),
                             sharex=False, sharey=False)
    for ax, name in zip(axes.flat, ordered):
        cdir = SWEEP_ROOT / name
        standalone = load_standalone_preload(cdir)
        coupled = load_coupled_history(cdir)
        solver_p = load_solver_cavity_pressure(cdir)

        # standalone last beats
        t_s = np.asarray(standalone["time"], dtype=float)
        s_last = _segment_beats(t_s)[-1]
        # coupled last beat
        t_c = np.asarray(coupled["time"], dtype=float)
        c_last = _segment_beats(t_c)[-1]
        c_last_p = c_last[c_last < solver_p.shape[0]]

        for side, color_solid, color_light, p_col in (
            ("LV", "#2B6CB0", "#9CB7D4", 0),
            ("RV", "#C53030", "#E0A6A6", 1),
        ):
            v_s = np.asarray(standalone[f"V_{side}"], dtype=float)
            p_s = np.asarray(standalone[f"p_{side}"], dtype=float)
            _plot_beat_pv(ax, v_s[s_last], p_s[s_last], color_light,
                          alpha=0.95, linewidth=1.2, linestyle="--")

            v_c = np.asarray(coupled[f"V_{side}"], dtype=float)
            p_c = solver_p[:, p_col]
            _plot_beat_pv(ax, v_c[c_last_p], p_c[c_last_p], color_solid,
                          alpha=1.0, linewidth=1.8)

        ax.set_title(f"{name}  —  peak RV sys = {rvsp[name]:.0f} mmHg",
                     fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel("V (mL)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("p (mmHg)", fontsize=9)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], color="#2B6CB0", linewidth=1.8, label="LV coupled"),
        Line2D([], [], color="#9CB7D4", linestyle="--", linewidth=1.2,
               label="LV standalone"),
        Line2D([], [], color="#C53030", linewidth=1.8, label="RV coupled"),
        Line2D([], [], color="#E0A6A6", linestyle="--", linewidth=1.2,
               label="RV standalone"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), ncols=4, frameon=False,
               fontsize=10)
    fig.suptitle(
        "Standalone 0D vs FEM-coupled PV loops, all 16 cases "
        "(ordered by achieved peak RV systolic)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.015, 1.0, 0.985])
    out_png = THESIS_FIG / "fig_4_9_all_16_pre_post.png"
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------- Figure B: PS-loop sweep ----------

def figure_B() -> Path:
    """Three-panel pressure-strain loop figure across all 16 cases.

    LV uses p_LV vs mean_E_ll_LV;  RV uses p_RV vs mean_E_ll_RV;
    Septum uses p_LV vs mean_E_ll_Septum (best single-pressure choice per thesis).
    Colored by achieved peak RV systolic pressure (mmHg).
    """
    # Gather per-case peak RVSP for colormap
    peak_rvsp = {}
    for name in CASE_NAMES:
        peak_rvsp[name] = case_peak_rv_systolic_mmHg(SWEEP_ROOT / name)

    rvsp_values = np.array([peak_rvsp[c] for c in CASE_NAMES])
    norm = mcolors.Normalize(vmin=float(rvsp_values.min()),
                             vmax=float(rvsp_values.max()))
    cmap = cm.coolwarm

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), sharey=True)
    panels = [
        ("LV free wall", "LV", "p_LV", 0),
        ("RV free wall", "RV", "p_RV", 1),
        ("Septum (p_LV)", "Septum", "p_LV", 0),
    ]

    for ax, (title, region, p_label, p_col) in zip(axes, panels):
        for name in CASE_NAMES:
            cdir = SWEEP_ROOT / name
            metrics = load_last_beat_metrics(cdir)
            t_m = np.asarray(metrics["time"], dtype=float)
            strain = np.asarray(metrics[f"mean_E_ll_{region}"], dtype=float)

            # Solver pressure sampled at dt=0.001; metrics also at dt=0.001
            # but starting at the same final-beat window. Reconstruct pressure
            # samples aligned to metrics time using the case's full solver_p.
            sp = load_solver_cavity_pressure(cdir)
            # solver_p length = total_steps; assume dt=0.001 with t[0]=dt
            # and total duration covers entire simulation.
            n_total = sp.shape[0]
            # time axis for sp: 0.001..n_total*0.001
            t_sp = (np.arange(1, n_total + 1)) * 0.001
            # Interpolate solver pressure onto metrics time
            p_use = np.interp(t_m, t_sp, sp[:, p_col])

            x, y = _close_loop(strain, p_use)
            ax.plot(
                x, y,
                color=cmap(norm(peak_rvsp[name])),
                linewidth=1.4, alpha=0.85,
            )

        ax.set_xlabel(r"longitudinal strain $\varepsilon_{\ell\ell}$")
        ax.set_title(f"{title}", fontsize=11)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("pressure (mmHg)")

    # Single colorbar to the right
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.025, pad=0.02)
    cbar.set_label("peak RV systolic (mmHg)")

    fig.suptitle(
        "Pressure-strain loops across the RVSP sweep (last simulated beat)",
        fontsize=12, y=1.02,
    )

    out_png = THESIS_FIG / "fig_5_0c_ps_loops_sweep.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------- Figure C: Cascade FE-vs-proxy loops (2x3) ----------

def figure_C() -> Path:
    """Combined cascade-loops figure: 2x3 panels.

    Top row (FE side): S_ff vs E_ff for LV free wall, RV free wall, septum.
    Bottom row (proxy side): cavity pressure vs epsilon_ll for the same regions.
        LV uses p_LV, RV uses p_RV, septum uses p_LV.

    All 16 cases overlaid per panel; colour = achieved peak RV systolic pressure
    (coolwarm). Single shared colorbar on the right.
    """
    # Per-case peak RVSP for the shared colormap
    peak_rvsp = {n: case_peak_rv_systolic_mmHg(SWEEP_ROOT / n) for n in CASE_NAMES}
    rvsp_values = np.array([peak_rvsp[c] for c in CASE_NAMES])

    # Symmetric-ish range so the mid-pressure case lands near coolwarm's white.
    # The sweep covers roughly 32-100 mmHg peak RVSP; pick a range so the midpoint
    # ~66 mmHg falls at the centre.
    vmin = float(np.floor(rvsp_values.min()))
    vmax = float(np.ceil(rvsp_values.max()))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.coolwarm

    fig, axes = plt.subplots(
        2, 3, figsize=(13.5, 8.4),
        sharex="row", sharey="row",
        constrained_layout=False,
    )

    regions = [
        ("LV free wall", "LV", 0),  # p_col for proxy row: 0 = p_LV
        ("RV free wall", "RV", 1),  # 1 = p_RV
        ("Septum", "Septum", 0),    # septum uses p_LV
    ]

    for col, (title, region, p_col) in enumerate(regions):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        for name in CASE_NAMES:
            cdir = SWEEP_ROOT / name
            m = load_last_beat_metrics(cdir)
            color = cmap(norm(peak_rvsp[name]))

            # --- FE row: S_ff (kPa) vs E_ff ---
            S_ff = np.asarray(m[f"mean_S_ff_{region}"], dtype=float) / 1000.0  # Pa -> kPa
            E_ff = np.asarray(m[f"mean_E_ff_{region}"], dtype=float)
            x_top, y_top = _close_loop(E_ff, S_ff)
            ax_top.plot(x_top, y_top, color=color, linewidth=1.2, alpha=0.7)

            # --- Proxy row: p_cav (mmHg) vs E_ll ---
            t_m = np.asarray(m["time"], dtype=float)
            E_ll = np.asarray(m[f"mean_E_ll_{region}"], dtype=float)
            sp = load_solver_cavity_pressure(cdir)
            n_total = sp.shape[0]
            t_sp = (np.arange(1, n_total + 1)) * 0.001
            p_use = np.interp(t_m, t_sp, sp[:, p_col])
            x_bot, y_bot = _close_loop(E_ll, p_use)
            ax_bot.plot(x_bot, y_bot, color=color, linewidth=1.2, alpha=0.7)

        ax_top.set_title(title, fontsize=12)
        ax_top.grid(alpha=0.25)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        ax_top.axhline(0.0, color="#888888", linewidth=0.6, alpha=0.5)
        ax_top.axvline(0.0, color="#888888", linewidth=0.6, alpha=0.5)

        ax_bot.grid(alpha=0.25)
        ax_bot.spines["top"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)
        ax_bot.axhline(0.0, color="#888888", linewidth=0.6, alpha=0.5)
        ax_bot.axvline(0.0, color="#888888", linewidth=0.6, alpha=0.5)

    # Axis labels
    axes[0, 0].set_ylabel(r"$S_{ff}$ (kPa)")
    axes[1, 0].set_ylabel("cavity pressure (mmHg)")
    for ax in axes[0, :]:
        ax.set_xlabel(r"$E_{ff}$")
    # Bottom row x-label
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\varepsilon_{\ell\ell}$")

    # Row labels on the leftmost panels (small text inside the axes area)
    axes[0, 0].text(
        0.02, 0.97, r"$S_{ff}$ vs $E_{ff}$  (FE)",
        transform=axes[0, 0].transAxes,
        fontsize=9, color="#444444",
        va="top", ha="left",
    )
    axes[1, 0].text(
        0.02, 0.97, r"$p_\mathrm{cav}$ vs $\varepsilon_{\ell\ell}$  (proxy)",
        transform=axes[1, 0].transAxes,
        fontsize=9, color="#444444",
        va="top", ha="left",
    )
    # Pressure-source annotations on the proxy row
    for ax, label in zip(axes[1, :], [r"$p_\mathrm{LV}$", r"$p_\mathrm{RV}$", r"$p_\mathrm{LV}$"]):
        ax.text(
            0.98, 0.97, label,
            transform=ax.transAxes,
            fontsize=10, color="#222222",
            va="top", ha="right",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                  "edgecolor": "#cccccc", "alpha": 0.85},
        )

    # Layout: leave room on the right for the colorbar
    fig.subplots_adjust(left=0.08, right=0.90, top=0.93, bottom=0.08,
                        wspace=0.16, hspace=0.22)

    # Shared colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.10, 0.018, 0.80])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Peak RV systolic pressure (mmHg)", fontsize=10)

    out_png = THESIS_FIG / "fig_5_0c_cascade_loops_sweep.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------- Figure D: Septum pressure-choices PS loops (1x4) ----------

def figure_D() -> Path:
    """Septum pressure-strain loops under four candidate y-axis pressures.

    1 row x 4 columns: p_LV, p_RV, mean (p_LV+p_RV)/2, transmural (p_LV-p_RV).
    All 16 sweep cases overlaid per panel; x-axis is the septum longitudinal
    strain mean_E_ll_Septum. Colour = achieved peak RV systolic pressure
    (coolwarm). Single shared colorbar on the right.
    """
    peak_rvsp = {n: case_peak_rv_systolic_mmHg(SWEEP_ROOT / n) for n in CASE_NAMES}
    rvsp_values = np.array([peak_rvsp[c] for c in CASE_NAMES])

    vmin = float(np.floor(rvsp_values.min()))
    vmax = float(np.ceil(rvsp_values.max()))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.coolwarm

    fig, axes = plt.subplots(
        1, 4, figsize=(16.0, 4.6),
        sharex=True,
        constrained_layout=False,
    )

    # (title, pressure-builder)
    def p_lv(p_LV, p_RV):
        return p_LV

    def p_rv(p_LV, p_RV):
        return p_RV

    def p_mean(p_LV, p_RV):
        return 0.5 * (p_LV + p_RV)

    def p_trans(p_LV, p_RV):
        return p_LV - p_RV

    panels = [
        (r"$p_\mathrm{LV}$", p_lv),
        (r"$p_\mathrm{RV}$", p_rv),
        (r"mean $(p_\mathrm{LV}+p_\mathrm{RV})/2$", p_mean),
        (r"transmural $p_\mathrm{LV}-p_\mathrm{RV}$", p_trans),
    ]

    for col, (title, builder) in enumerate(panels):
        ax = axes[col]
        for name in CASE_NAMES:
            cdir = SWEEP_ROOT / name
            m = load_last_beat_metrics(cdir)
            color = cmap(norm(peak_rvsp[name]))

            t_m = np.asarray(m["time"], dtype=float)
            E_ll = np.asarray(m["mean_E_ll_Septum"], dtype=float)

            sp = load_solver_cavity_pressure(cdir)  # (N,2) [p_LV, p_RV] mmHg
            n_total = sp.shape[0]
            t_sp = (np.arange(1, n_total + 1)) * 0.001
            p_LV = np.interp(t_m, t_sp, sp[:, 0])
            p_RV = np.interp(t_m, t_sp, sp[:, 1])
            p_use = builder(p_LV, p_RV)

            x, y = _close_loop(E_ll, p_use)
            ax.plot(x, y, color=color, linewidth=1.2, alpha=0.7)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"$\varepsilon_{\ell\ell}$ (septum)")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.0, color="#888888", linewidth=0.6, alpha=0.5)
        ax.axvline(0.0, color="#888888", linewidth=0.6, alpha=0.5)

    axes[0].set_ylabel("pressure (mmHg)")

    # Layout: leave room on the right for the colorbar
    fig.subplots_adjust(left=0.06, right=0.91, top=0.90, bottom=0.14,
                        wspace=0.22)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.925, 0.16, 0.014, 0.72])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Peak RV systolic pressure (mmHg)", fontsize=10)

    out_png = THESIS_FIG / "fig_5_0d_septum_pressure_choices.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------- Driver ----------

def main() -> None:
    # Compute peak RV systolic mmHg from FEM cavity LM (the canonical "achieved")
    rvsp = {n: case_peak_rv_systolic_mmHg(SWEEP_ROOT / n) for n in CASE_NAMES}
    # Lowest RVSP -> baseline, highest -> severe
    sorted_cases = sorted(CASE_NAMES, key=lambda n: rvsp[n])
    baseline = sorted_cases[0]
    severe = sorted_cases[-1]

    print(f"Baseline:  {baseline}  peak RVSP = {rvsp[baseline]:.1f} mmHg")
    print(f"Severe:    {severe}  peak RVSP = {rvsp[severe]:.1f} mmHg")

    # Coupling-residual walkthrough (mid-pressure case)
    mid_case = "sPAP70"
    print(f"Mid-case for coupling residual figures: {mid_case}  "
          f"peak RVSP = {rvsp[mid_case]:.1f} mmHg")

    out_46 = figure_standalone_convergence(mid_case)
    print(f"Wrote 4.6: {out_46}")

    out_47 = figure_coupling_jump(mid_case)
    print(f"Wrote 4.7: {out_47}")

    out_48 = figure_coupled_convergence(mid_case)
    print(f"Wrote 4.8: {out_48}")

    out_49 = figure_all_16_pre_post()
    print(f"Wrote 4.9: {out_49}")

    out_c = figure_C()
    print(f"Wrote Figure C: {out_c}")

    out_d = figure_D()
    print(f"Wrote Figure D: {out_d}")


if __name__ == "__main__":
    main()
