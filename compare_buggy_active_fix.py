#!/usr/bin/env python3
"""compare_buggy_active_fix.py

Side-by-side comparison of the pre-99e78f0 active-stress counterfactual
against the production active-stress formulation on the same 10mm UKB
sPAP22 case. Reads metrics_downsample_1.npy from each run (both must have
been re-postprocessed with --no-skip-regional-internal so that
mean_sigma_ff_LV/RV/Septum and work_active/passive/comp arrays are present),
extracts the cycle-end stress magnitudes and energy closure, and writes a
JSON summary plus a bar-chart figure for the thesis.

Outputs (under --out, default results/analysis/buggy_active_audit):
  buggy_vs_fixed_summary.json         small numerical summary
  fig_buggy_active_audit.{png,pdf}    figure used by the implementation chapter
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGIONS = ("LV", "RV", "Septum")


def load_metrics(case_dir: Path) -> dict:
    p = case_dir / "metrics" / "metrics_downsample_1.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run the re-postprocess sbatch first.")
    return np.load(p, allow_pickle=True).item()


def peak_sigma_ff(m: dict) -> dict[str, float]:
    out = {}
    for r in REGIONS:
        key = f"mean_sigma_ff_{r}"
        if key not in m:
            raise KeyError(f"{key} missing — re-postprocess with --no-skip-regional-internal")
        out[r] = float(np.max(np.abs(np.asarray(m[key], dtype=float))) / 1e3)  # → kPa
    return out


def cycle_closure(m: dict) -> dict[str, float]:
    """Cycle-end energy closure (last-beat-restricted if metrics holds all beats).

    Production postprocessing stores per-step increments for the last beat only
    (per the audit script comment). Sum the available arrays for a cycle-end
    residual.
    """
    work_int_inc = (np.asarray(m["work_active_Whole"], dtype=float)
                    + np.asarray(m["work_passive_Whole"], dtype=float)
                    + np.asarray(m["work_comp_Whole"], dtype=float))
    work_cav_inc = (np.asarray(m["work_boundary_exact_LV"], dtype=float)
                    + np.asarray(m["work_boundary_exact_RV"], dtype=float))
    work_rob_inc = (np.asarray(m["work_robin_epi"], dtype=float)
                    + np.asarray(m["work_robin_base"], dtype=float))
    w_int = work_int_inc.sum()
    w_cav = work_cav_inc.sum()
    w_rob = work_rob_inc.sum()
    residual = w_int - (w_cav + w_rob)
    rel = abs(residual) / max(abs(w_cav), 1e-12)
    return {
        "W_int_J": float(w_int),
        "W_cav_J": float(w_cav),
        "W_robin_J": float(w_rob),
        "R_J": float(residual),
        "R_rel": float(rel),
    }


def make_figure(buggy: dict, fixed: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    width = 0.35
    x = np.arange(len(REGIONS))
    buggy_vals = [buggy["sigma_ff_kPa"][r] for r in REGIONS]
    fixed_vals = [fixed["sigma_ff_kPa"][r] for r in REGIONS]

    ax.bar(x - width / 2, buggy_vals, width,
           label=r"active stress on $\mathbf{C}$ (pre-fix)",
           color="#cc4444")
    ax.bar(x + width / 2, fixed_vals, width,
           label=r"active stress on $\bar{\mathbf{C}}=J^{-2/3}\mathbf{C}$ (production)",
           color="#4477aa")

    # Literature envelope band (kPa) — same as fig_stress_magnitudes
    ax.axhspan(20, 80, color="#bbbbbb", alpha=0.25, zorder=0,
               label="reported envelope (20–80 kPa)")

    for xi, v in zip(x - width / 2, buggy_vals):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + width / 2, fixed_vals):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["LV free wall", "RV free wall", "Septum"])
    ax.set_ylabel(r"peak $|\sigma_{ff}|$ (kPa)")
    ax.set_ylim(0, max(max(buggy_vals + fixed_vals) + 12, 90))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"Wrote {out_path.with_suffix('.png')}")
    print(f"Wrote {out_path.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buggy", required=True, type=Path)
    parser.add_argument("--fixed", required=True, type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path("results/analysis/buggy_active_audit"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading buggy: {args.buggy}")
    m_buggy = load_metrics(args.buggy)
    print(f"Loading fixed: {args.fixed}")
    m_fixed = load_metrics(args.fixed)

    summary = {
        "buggy": {
            "case": str(args.buggy),
            "label": "pre-99e78f0 active stress on C",
            "sigma_ff_kPa": peak_sigma_ff(m_buggy),
            "closure": cycle_closure(m_buggy),
        },
        "fixed": {
            "case": str(args.fixed),
            "label": "production active stress on Cdev",
            "sigma_ff_kPa": peak_sigma_ff(m_fixed),
            "closure": cycle_closure(m_fixed),
        },
    }

    out_json = args.out / "buggy_vs_fixed_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    print("\nPeak |sigma_ff| (kPa):")
    print(f"{'region':12s}  {'buggy':>8s}  {'fixed':>8s}  {'ratio':>6s}")
    for r in REGIONS:
        b = summary["buggy"]["sigma_ff_kPa"][r]
        f = summary["fixed"]["sigma_ff_kPa"][r]
        print(f"{r:12s}  {b:8.2f}  {f:8.2f}  {f / b:6.2f}")

    print("\nCycle-end closure:")
    for k in ("buggy", "fixed"):
        c = summary[k]["closure"]
        print(f"  {k:6s}  W_int={c['W_int_J']:+.4f} J  W_cav={c['W_cav_J']:+.4f} J  "
              f"W_rob={c['W_robin_J']:+.4f} J  R={c['R_J']:+.4e} J  R/|W_cav|={c['R_rel']:.3e}")

    make_figure(summary["buggy"], summary["fixed"], args.out / "fig_buggy_active_audit")


if __name__ == "__main__":
    main()
