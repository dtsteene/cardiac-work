#!/usr/bin/env python3
"""Analyze the softer-material pilot (2026-07-08).

Question: does reducing passive stiffness open up LV (and RV/septum) dynamic
range in the pure-PAH sweep? We compare the baseline->severe (case0->case7)
spread of fiber strain across three whole-heart stiffness scales.

Two things are measured per region, per scale:
  * lambda_ED  = sqrt(1 + 2*E_ff_ED)   (ED = most-stretched instant) — the PRELOAD
    stretch the Frank-Starling gain reads. Whether softening widens the case0->case7
    spread of lambda_ED tells us if softer material grows the FS "signal".
  * systolic E_ff excursion = E_ff_ED - E_ff_ES (peak stretch to peak shortening) —
    the contraction strain amplitude. Whether softening widens its case0->case7
    spread tells us if softer material opens up systolic strain range even when
    ED volume is coupling-pinned.

Login-safe: pure NumPy replay of the metrics sidecars. Run after the 6 pilot
jobs finish:  python3 pah_pulmonary_batch/analyze_softmat_pilot.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

ROOT = paths.RESULTS_ROOT / "sims/2026-07-08/softmat_pilot_L10"
SCALES = [("100", 1.00), ("050", 0.50), ("033", 0.33)]
CASES = ["case0_rv25", "case7_rv95"]
REGIONS = ["LV", "RV", "Septum"]


def strain_metrics(case_dir):
    """Per-region (E_ff at ED, E_ff at ES) from the metrics sidecar."""
    m = np.load(case_dir / "metrics/metrics_downsample_1.npy", allow_pickle=True).item()
    out = {}
    for r in REGIONS:
        eff = np.asarray(m[f"mean_E_ff_{r}"], float)
        eff = eff[np.isfinite(eff)]
        out[r] = (float(eff.max()), float(eff.min()))   # ED (most stretched), ES
    return out


def main():
    # data[scale_tag][case][region] = (E_ff_ED, E_ff_ES)
    data = {}
    missing = []
    for tag, _ in SCALES:
        data[tag] = {}
        for case in CASES:
            cd = ROOT / f"scale{tag}" / case
            f = cd / "metrics/metrics_downsample_1.npy"
            if not f.exists():
                missing.append(str(cd)); continue
            data[tag][case] = strain_metrics(cd)

    if missing:
        print("Not all runs are ready yet — missing metrics for:")
        for m in missing:
            print(f"  {m}")
        print("\n(Re-run once the jobs finish.)")
        if len(missing) == len(SCALES) * len(CASES):
            return

    lam = lambda e: np.sqrt(1 + 2 * e)

    for r in REGIONS:
        print("\n" + "=" * 72)
        print(f"REGION = {r}")
        print("=" * 72)
        print(f"  {'scale':6s} | {'lamED c0':>9s} {'lamED c7':>9s} {'Δ(c7-c0)':>9s} "
              f"| {'exc c0':>8s} {'exc c7':>8s} {'Δexc':>8s}")
        base_dlam = base_dexc = None
        for tag, sval in SCALES:
            if tag not in data or not all(c in data[tag] for c in CASES):
                print(f"  {sval:<6.2f} | (incomplete)")
                continue
            ed0, es0 = data[tag]["case0_rv25"][r]
            ed7, es7 = data[tag]["case7_rv95"][r]
            l0, l7 = lam(ed0), lam(ed7)
            dlam = l7 - l0
            exc0, exc7 = ed0 - es0, ed7 - es7      # systolic shortening excursion
            dexc = exc7 - exc0
            if base_dlam is None:
                base_dlam, base_dexc = dlam, dexc
            print(f"  {sval:<6.2f} | {l0:9.4f} {l7:9.4f} {dlam:+9.4f} "
                  f"| {exc0:8.4f} {exc7:8.4f} {dexc:+8.4f}")
        # verdict
        if base_dlam is not None and "033" in data and all(c in data["033"] for c in CASES):
            ed0, es0 = data["033"]["case0_rv25"][r]; ed7, es7 = data["033"]["case7_rv95"][r]
            soft_dlam = lam(ed7) - lam(ed0)
            soft_dexc = (ed7 - es7) - (ed0 - es0)
            fl = soft_dlam / base_dlam if base_dlam else float("nan")
            fe = soft_dexc / base_dexc if base_dexc else float("nan")
            print(f"\n  softening 1.0->0.33x  changes case0->case7 spread by:")
            print(f"    ED-stretch spread   x{fl:5.2f}   (preload / FS signal)")
            print(f"    systolic excursion  x{fe:5.2f}   (contraction strain range)")


if __name__ == "__main__":
    main()
