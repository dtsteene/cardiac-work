#!/usr/bin/env python3
"""
audit_three_bugs_energy_budget.py

Reconstruct the cumulative energy-budget closure under four postprocessing
variants on the production sPAP22 case:

  V0 (all 3 bugs):     internal work without compressibility stress,
                       cavity work using 0D pressure,
                       Robin work using reference-config inner(u, du) ds_ref
  V1 (+ deviatoric):   add compressibility stress back (full strain)
  V2 (+ pressure):     also switch cavity pressure to solver Lagrange multiplier
  V3 (production):     also switch Robin to Nanson normal-projection formula

Bugs 1 and 2 are derived from the existing metrics_downsample_1.npy
(active + passive + comp work decomposition, both LM and 0D pressures, the
production cavity-work increments). Bug 3 needs a Robin-only replay that
assembles the buggy reference-config form at each timestep against the saved
displacement; that part lives in robin_reference_replay.py and writes
robin_reference_work_per_step.npy alongside the case metrics. This script
picks the buggy Robin file up if present; otherwise reports only V1, V2, V3
(using production Robin in all of them, treating V0 == V1).

Output: closure_variants.npz with cumulative-work traces for each variant
and final-time residuals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_CASE = (
    "results/sims/2026-05-10/capped_shared_l5_20260510_141015/sPAP22"
)
DEFAULT_OUT = "results/analysis/three_bugs_audit"


def cumulative_from_increments(per_step: np.ndarray) -> np.ndarray:
    """Cumulative sum of per-step increments, starting from 0 at step 0."""
    return np.concatenate(([0.0], np.cumsum(per_step)))[: len(per_step)]


def load_case(case_dir: Path) -> dict:
    """Load the per-step quantities we need from the production metrics."""
    metrics = np.load(case_dir / "metrics" / "metrics_downsample_1.npy",
                      allow_pickle=True).item()

    def arr(key: str) -> np.ndarray:
        v = metrics[key]
        return np.asarray(v, dtype=float)

    out = {
        "time": arr("time"),
        # internal work per step, decomposed
        "work_active": arr("work_active_Whole"),
        "work_passive": arr("work_passive_Whole"),
        "work_comp": arr("work_comp_Whole"),
        "work_true": arr("work_true_Whole"),  # = active + passive + comp
        # cavity boundary work per step (uses LM pressure, divergence-theorem ΔV)
        "work_cav_LV_lm": arr("work_boundary_exact_LV"),
        "work_cav_RV_lm": arr("work_boundary_exact_RV"),
        # pressures per step
        "p_LV_lm": arr("p_LV"),
        "p_RV_lm": arr("p_RV"),
        "p_LV_0d": arr("p_LV_0D"),
        "p_RV_0d": arr("p_RV_0D"),
        # Robin work per step (production, Nanson)
        "work_robin_epi": arr("work_robin_epi"),
        "work_robin_base": arr("work_robin_base"),
    }
    out["work_robin_total"] = out["work_robin_epi"] + out["work_robin_base"]
    return out


def step_average(p: np.ndarray) -> np.ndarray:
    """Trapezoidal-rule mid-step average: 0.5*(p[i] + p[i-1]) for i>=1, p[0] at 0."""
    avg = 0.5 * (p[1:] + p[:-1])
    return np.concatenate(([p[0]], avg))


def cavity_work_0d_variant(
    work_cav_lm: np.ndarray, p_lm: np.ndarray, p_0d: np.ndarray
) -> np.ndarray:
    """
    Reconstruct cavity-pressure-work increments as if the 0D pressure had been
    used in place of the LM (solver) pressure.

    The production increment is bar_p_lm[i] * ΔV[i]. Replacing bar_p_lm by
    bar_p_0d gives:
        bug2_work[i] = (bar_p_0d[i] / bar_p_lm[i]) * work_cav_lm[i]
    During isovolumic phases ΔV ≈ 0 so the increment is ≈ 0 regardless of
    pressure; we guard against division-by-near-zero by zeroing out steps
    where the LM pressure is tiny.
    """
    bar_p_lm = step_average(p_lm)
    bar_p_0d = step_average(p_0d)
    out = np.zeros_like(work_cav_lm)
    mask = np.abs(bar_p_lm) > 1e-9
    out[mask] = (bar_p_0d[mask] / bar_p_lm[mask]) * work_cav_lm[mask]
    return out


def reference_robin_per_step(case_dir: Path, n_last_beat: int) -> np.ndarray | None:
    """
    Load the buggy reference-config Robin work per step (last beat only) if
    a replay output file is present, else return None.

    The replay writes one row per checkpoint timestep (4800 for a 6-beat run);
    the metrics file holds only the last beat (802 entries). We slice the
    replay output to the trailing n_last_beat rows to align.
    """
    cand = case_dir / "metrics" / "robin_reference_work_per_step.npy"
    if not cand.exists():
        return None
    full = np.load(cand)  # shape (n_total, 2) = [epi_inc, base_inc]
    if full.shape[0] < n_last_beat:
        raise ValueError(f"Robin replay has {full.shape[0]} rows but last beat "
                         f"requires {n_last_beat}.")
    last_beat = full[-n_last_beat:]
    # Sum epi + base per step
    return last_beat[:, 0] + last_beat[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path(DEFAULT_CASE))
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    args = parser.parse_args()

    case_dir = args.case
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading case: {case_dir}")
    d = load_case(case_dir)
    n = len(d["time"])
    print(f"  {n} timesteps, t = {d['time'][0]:.3f} → {d['time'][-1]:.3f} s")

    # ── Internal work variants ────────────────────────────────────────────────
    # production: active + passive + comp (full strain)
    w_int_full = d["work_active"] + d["work_passive"] + d["work_comp"]
    # bug1 mimic: drop compressibility stress contribution
    w_int_no_comp = d["work_active"] + d["work_passive"]
    # Cross-check w_int_full ≈ work_true
    diff_true = np.abs(w_int_full - d["work_true"]).sum()
    print(f"  internal work sanity: |full - true| sum = {diff_true:.3e}")

    # ── Cavity work variants ──────────────────────────────────────────────────
    w_cav_lm_total = d["work_cav_LV_lm"] + d["work_cav_RV_lm"]
    bug2_lv = cavity_work_0d_variant(d["work_cav_LV_lm"], d["p_LV_lm"], d["p_LV_0d"])
    bug2_rv = cavity_work_0d_variant(d["work_cav_RV_lm"], d["p_RV_lm"], d["p_RV_0d"])
    w_cav_0d_total = bug2_lv + bug2_rv

    # Sanity print
    print(f"  cavity work cycle end: LM = {w_cav_lm_total.sum():.4f} J, "
          f"0D = {w_cav_0d_total.sum():.4f} J, "
          f"diff = {w_cav_0d_total.sum() - w_cav_lm_total.sum():+.4f} J")

    # ── Robin work variants ──────────────────────────────────────────────────
    w_robin_correct = d["work_robin_total"]
    w_robin_buggy = reference_robin_per_step(case_dir, n_last_beat=n)
    if w_robin_buggy is None:
        print("  Robin reference-config replay not found at "
              "metrics/robin_reference_work_per_step.npy")
        print("  -> V0 will use production Robin (== V1 result for bug 3 column)")
        w_robin_buggy = w_robin_correct.copy()
        bug3_available = False
    else:
        print(f"  Robin reference-config replay loaded "
              f"(cycle end: buggy = {w_robin_buggy.sum():.4f} J, "
              f"correct = {w_robin_correct.sum():.4f} J)")
        bug3_available = True

    # ── Assemble variants ────────────────────────────────────────────────────
    # closure residual at each step = W_int_cum - (W_cav_cum + W_robin_cum)
    # Want absolute closure error at cycle end (smaller = better closure).

    variants = {
        "V0_all_bugs": {
            "w_int_inc":   w_int_no_comp,
            "w_cav_inc":   w_cav_0d_total,
            "w_robin_inc": w_robin_buggy,
            "label": "all three bugs (no-comp + 0D pressure + ref Robin)",
        },
        "V1_fix_dev": {
            "w_int_inc":   w_int_full,
            "w_cav_inc":   w_cav_0d_total,
            "w_robin_inc": w_robin_buggy,
            "label": "+ deviatoric/comp fix",
        },
        "V2_fix_press": {
            "w_int_inc":   w_int_full,
            "w_cav_inc":   w_cav_lm_total,
            "w_robin_inc": w_robin_buggy,
            "label": "+ cavity pressure (LM) fix",
        },
        "V3_production": {
            "w_int_inc":   w_int_full,
            "w_cav_inc":   w_cav_lm_total,
            "w_robin_inc": w_robin_correct,
            "label": "production (all 3 fixes)",
        },
    }

    save_blob: dict = {
        "time": d["time"],
        "bug3_available": bug3_available,
    }
    print("\nCycle-end closure summary:")
    print(f"{'variant':16}  {'W_int (J)':>12}  {'W_cav (J)':>12}  "
          f"{'W_rob (J)':>10}  {'residual (J)':>14}  {'rel resid':>10}")
    for name, v in variants.items():
        wi = np.cumsum(v["w_int_inc"])
        wc = np.cumsum(v["w_cav_inc"])
        wr = np.cumsum(v["w_robin_inc"])
        residual = wi - (wc + wr)
        # Final-time residual: we use the cycle-spanning final tick.
        final_int = wi[-1]
        final_cav = wc[-1]
        final_rob = wr[-1]
        final_res = residual[-1]
        rel = abs(final_res) / max(abs(final_cav), 1e-12)
        save_blob[name] = {
            "w_int_cum": wi,
            "w_cav_cum": wc,
            "w_robin_cum": wr,
            "residual_cum": residual,
            "final_internal_J": float(final_int),
            "final_cavity_J": float(final_cav),
            "final_robin_J": float(final_rob),
            "final_residual_J": float(final_res),
            "final_residual_rel": float(rel),
            "label": v["label"],
        }
        print(f"{name:16}  {final_int:12.4f}  {final_cav:12.4f}  "
              f"{final_rob:10.4f}  {final_res:14.4e}  {rel:10.3e}")

    np.save(out_dir / f"{case_dir.name}_closure_variants.npy", save_blob,
            allow_pickle=True)
    # Also dump a small JSON summary for inspection
    summary = {
        "case": str(case_dir),
        "bug3_available": bug3_available,
        "variants": {
            k: {kk: vv for kk, vv in v.items()
                if not kk.endswith("_cum")}
            for k, v in save_blob.items()
            if isinstance(v, dict)
        },
    }
    with open(out_dir / f"{case_dir.name}_closure_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {out_dir / (case_dir.name + '_closure_variants.npy')}")
    print(f"Wrote {out_dir / (case_dir.name + '_closure_summary.json')}")


if __name__ == "__main__":
    main()
