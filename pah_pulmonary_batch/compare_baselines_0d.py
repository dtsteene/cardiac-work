#!/usr/bin/env python3
"""Standalone 0D comparison of candidate linear-EDPVR baselines.

Purpose
-------
We are building a new PAH batch in which the *only* knobs that change across
cases are the pulmonary windkessel resistance/compliance (R_AR_PUL up,
C_AR_PUL down).  Everything else -- mesh, fibres, the shared inverse-unloaded
reference, and the baseline circulation parameters -- is held fixed, and the
ventricular EDPVR must be LINEAR (no Klotz/exponential kE term).

Before committing a baseline we compare candidates with the standalone 0D
Regazzoni model:

  * sPAP22  (lowest-pressure v12_exp case)  -- as-is, exponential EDPVR (reference)
  * sPAP22  relinearised (kE -> 0, EB kept) -- the "strip the Klotz" candidate
  * healthy_circulation_params.json         -- already linear, no kE

This is *throwaway analysis*, not production.  It is pure numpy/scipy ODE
integration (no FEniCSx), so it is safe to run on the login node.

CAVEAT (see project memory): the standalone 0D library has drifted and tends
to INFLATE systolic RV pressure relative to the coupled FEM sim.  Therefore
treat the systolic numbers below as relative-only.  The reliable comparison
for picking a baseline is the DIASTOLIC limb (EDV / EDP) and the overall loop
shape, which is exactly what the linear-vs-exponential EDPVR choice controls.

The 0D model also uses its own elastance ventricles here; in the real batch
the LV/RV are replaced by the FEM cavities, so these chamber elastances only
set the warm-up / initial state and the unloading ED target.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import circulation
from circulation.regazzoni2020 import Regazzoni2020

# The 0D solver logs every beat; silence it for clean, fast batch runs.
logging.getLogger("circulation").setLevel(logging.ERROR)
logging.getLogger("circulation.base").setLevel(logging.ERROR)

ureg = circulation.units.ureg
HERE = Path(__file__).resolve().parent
WORK = HERE.parent

# ---- BPM=75 activation timing, copied verbatim from complete_cycle.py -------
BPM = 75
HR_HZ = BPM / 60.0          # 1.25 Hz
RR_INTERVAL = 1.0 / HR_HZ   # 0.8 s
_SCALE = RR_INTERVAL / 0.8  # 1.0
TC_ACTIVATION = 0.25 * _SCALE
TR_ACTIVATION = 0.40 * _SCALE


def _update_from_json(params, json_params):
    """Recursively merge JSON params into a unit-carrying defaults dict.

    Mirrors complete_cycle.update_parameters_from_json: preserve the unit of an
    existing default; add unit-less keys (e.g. kE) verbatim.
    """
    for key, value in json_params.items():
        if key in params and isinstance(value, dict) and isinstance(params[key], dict):
            _update_from_json(params[key], value)
        elif key in params and hasattr(params[key], "units"):
            params[key] = value * params[key].units
        else:
            params[key] = value


def build_params(json_path: Path, relinearize: bool):
    """Build Regazzoni params + initial_state exactly as complete_cycle does
    for BPM=75, optionally forcing a linear ventricular EDPVR (kE -> 0)."""
    params = Regazzoni2020.default_parameters()
    data = json.load(open(json_path))
    _update_from_json(params, data.get("parameters", data))

    # --- timing: shift so LV contraction onset sits at t=0 (see complete_cycle)
    factor = RR_INTERVAL / 0.8
    time_shift = -(0.1 * factor)
    for ch in ["LA", "RA", "LV", "RV"]:
        original_tC = params["chambers"][ch]["tC"].magnitude
        params["chambers"][ch]["tC"] = (original_tC * factor + time_shift) * ureg("s")
        params["chambers"][ch]["TC"] *= factor
        params["chambers"][ch]["TR"] *= factor
    for ch in ["LV", "RV"]:
        params["chambers"][ch]["tC"] = 0.0 * ureg("s")
        params["chambers"][ch]["TC"] = TC_ACTIVATION * ureg("s")
        params["chambers"][ch]["TR"] = TR_ACTIVATION * ureg("s")
    params["HR"] = ureg(f"{HR_HZ} Hz")

    if relinearize:
        for ch in ["LV", "RV"]:
            params["chambers"][ch].pop("kE", None)

    return params, data.get("initial_state", None)


def run_0d(params, initial_state, num_beats, tag):
    outdir = Path("/tmp/regz_compare") / tag
    outdir.mkdir(parents=True, exist_ok=True)
    model = Regazzoni2020(parameters=params, outdir=outdir)
    history = model.solve(num_beats=num_beats, initial_state=initial_state)
    return history


def last_cycle_mask(history):
    t = np.asarray(history["time"], dtype=float)
    return t >= (t[-1] - RR_INTERVAL)


def chamber_stats(history, chamber):
    m = last_cycle_mask(history)
    V = np.asarray(history[f"V_{chamber}"], dtype=float)[m]
    p = np.asarray(history[f"p_{chamber}"], dtype=float)[m]
    edv = float(V.max())
    esv = float(V.min())
    sv = edv - esv
    ef = 100.0 * sv / edv if edv else float("nan")
    edp = float(p[np.argmax(V)])          # pressure at end-diastole (max V)
    esp = float(p[np.argmin(V)])          # pressure at end-systole (min V)
    p_peak = float(p.max())               # peak systolic pressure
    return dict(EDV=edv, ESV=esv, SV=sv, EF=ef, EDP=edp, ESP=esp, p_peak=p_peak)


def pul_stats(history, params):
    m = last_cycle_mask(history)
    pa = np.asarray(history["p_AR_PUL"], dtype=float)[m]
    pul = params["circulation"]["PUL"]
    R = pul["R_AR"]; C = pul["C_AR"]
    R = R.magnitude if hasattr(R, "magnitude") else R
    C = C.magnitude if hasattr(C, "magnitude") else C
    return dict(PA_sys=float(pa.max()), PA_dia=float(pa.min()),
                PA_mean=float(pa.mean()), R_AR=float(R), C_AR=float(C))


def fmt(d, keys):
    return "  ".join(f"{k}={d[k]:7.2f}" for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats", type=int, default=60,
                    help="0D beats to steady state (default 60)")
    args = ap.parse_args()

    candidates = [
        ("sPAP22_exp",
         WORK / "data/ukb_circ_v12_exp/optimized_regazzoni_ukb_sPAP22.json", False),
        ("sPAP22_linear",
         WORK / "data/ukb_circ_v12_exp/optimized_regazzoni_ukb_sPAP22.json", True),
        ("healthy_linear",
         WORK / "data/healthy_circulation_params.json", False),
    ]

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for tag, path, relin in candidates:
        params, ic = build_params(path, relin)
        kE_LV = params["chambers"]["LV"].get("kE", 0.0)
        kE_RV = params["chambers"]["RV"].get("kE", 0.0)
        hist = run_0d(params, ic, args.beats, tag)
        lv = chamber_stats(hist, "LV")
        rv = chamber_stats(hist, "RV")
        pul = pul_stats(hist, params)
        results[tag] = dict(LV=lv, RV=rv, PUL=pul, kE_LV=kE_LV, kE_RV=kE_RV)

        m = last_cycle_mask(hist)
        axes[0].plot(np.asarray(hist["V_LV"])[m], np.asarray(hist["p_LV"])[m], label=tag)
        axes[1].plot(np.asarray(hist["V_RV"])[m], np.asarray(hist["p_RV"])[m], label=tag)

    axes[0].set(title="LV PV loop (last beat)", xlabel="V_LV [mL]", ylabel="p_LV [mmHg]")
    axes[1].set(title="RV PV loop (last beat)", xlabel="V_RV [mL]", ylabel="p_RV [mmHg]")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    figpath = HERE / "baseline_compare_0d.png"
    fig.tight_layout(); fig.savefig(figpath, dpi=140)

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"0D baseline comparison  ({args.beats} beats, BPM={BPM})")
    print("  systolic numbers are 0D-only (drifted lib inflates them) -> relative use")
    print("  reliable picks come from the diastolic limb: EDV / EDP / loop shape")
    print("=" * 78)
    for tag in results:
        r = results[tag]
        print(f"\n### {tag}   (kE_LV={r['kE_LV']:.4g}, kE_RV={r['kE_RV']:.4g})")
        print(f"  LV  {fmt(r['LV'], ['EDV','ESV','SV','EF','EDP','ESP','p_peak'])}")
        print(f"  RV  {fmt(r['RV'], ['EDV','ESV','SV','EF','EDP','ESP','p_peak'])}")
        print(f"  PUL {fmt(r['PUL'], ['R_AR','C_AR','PA_sys','PA_dia','PA_mean'])}")

    print("\nPhysiological sanity reference (rough, healthy adult):")
    print("  LV: EDV 120-150, EF 55-70%, EDP 6-12, ESP ~120")
    print("  RV: EDV 120-160, EF 45-65%, EDP 2-6,  ESP/peak ~20-30")
    print(f"\nfigure: {figpath}")
    (HERE / "baseline_compare_0d.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
