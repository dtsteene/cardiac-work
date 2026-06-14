#!/usr/bin/env python3
"""Manual pulmonary-windkessel sweep, visualised as a PV-loop spectrum (0D).

We hold an entire baseline circulation fixed and vary ONLY the pulmonary
arterial windkessel along a smooth PAH locus: R_AR_PUL up, C_AR_PUL down.
For each point we run the standalone 0D Regazzoni model and plot the LV/RV
PV loops and the pulmonary-artery pressure across the spectrum, plus trend
curves (RV peak pressure, RV stroke work, PA systolic) vs severity.

Use it to (a) pick which baseline to start manual tuning from, and (b) choose
the 8 (R_AR, C_AR) pairs for the real batch.

CAVEATS (read before trusting a number)
  * Coupled sims replace LV/RV with the FEM cavities, so these 0D ventricular
    PV loops reflect the JSON elastance, NOT the FEM mechanics. They are an
    afterload-exploration aid, not the final loops.
  * The standalone lib inflates systolic RV pressure (project memory). Treat
    RV/LV systolic magnitudes as relative; the PA pressure trend and the
    windkessel values themselves are the transferable outputs.

Pure numpy/scipy ODE integration -> login-node safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from compare_baselines_0d import (
    build_params, run_0d, last_cycle_mask, RR_INTERVAL, ureg, WORK, HERE,
)

# Named baselines we might start tuning from. Each: (json, relinearize).
BASELINES = {
    "healthy_linear": (WORK / "data/healthy_circulation_params.json", False),
    "sPAP22_linear": (WORK / "data/ukb_circ_v12_exp/optimized_regazzoni_ukb_sPAP22.json", True),
    "sPAP22_exp": (WORK / "data/ukb_circ_v12_exp/optimized_regazzoni_ukb_sPAP22.json", False),
}


def pul_gradient(n, r_lo, r_hi, c_hi, c_lo):
    """n points along a geometric PAH locus: R_AR r_lo->r_hi, C_AR c_hi->c_lo."""
    s = np.linspace(0.0, 1.0, n)
    R = r_lo * (r_hi / r_lo) ** s
    C = c_hi * (c_lo / c_hi) ** s
    return list(zip(R, C))


def loop_area(V, p):
    """Signed PV-loop area (stroke work proxy) via the shoelace formula [mmHg*mL]."""
    return 0.5 * float(np.abs(np.sum(V * np.roll(p, -1) - np.roll(V, -1) * p)))


def run_spectrum(baseline, n, grad, beats):
    json_path, relin = BASELINES[baseline]
    rows = []
    loops = []  # (R, C, V_LV, p_LV, V_RV, p_RV, p_AR_PUL)
    for k, (R, C) in enumerate(grad):
        params, ic = build_params(json_path, relin)
        params["circulation"]["PUL"]["R_AR"] = R * ureg("mmHg*s/mL")
        params["circulation"]["PUL"]["C_AR"] = C * ureg("mL/mmHg")
        hist = run_0d(params, ic, beats, f"sweep_{baseline}_{k}")
        m = last_cycle_mask(hist)
        Vlv = np.asarray(hist["V_LV"])[m]; plv = np.asarray(hist["p_LV"])[m]
        Vrv = np.asarray(hist["V_RV"])[m]; prv = np.asarray(hist["p_RV"])[m]
        pap = np.asarray(hist["p_AR_PUL"])[m]
        loops.append((R, C, Vlv, plv, Vrv, prv, pap))
        rows.append(dict(
            k=k, R_AR=R, C_AR=C, RC=R * C,
            RV_peak=float(prv.max()), RV_EDV=float(Vrv.max()),
            RV_ESV=float(Vrv.min()), RV_SV=float(Vrv.max() - Vrv.min()),
            RV_SW=loop_area(Vrv, prv),
            LV_peak=float(plv.max()), LV_SV=float(Vlv.max() - Vlv.min()),
            PA_sys=float(pap.max()), PA_dia=float(pap.min()), PA_mean=float(pap.mean()),
        ))
    return rows, loops


def plot_spectrum(baseline, rows, loops, outpath):
    n = len(loops)
    colors = cm.viridis(np.linspace(0, 1, n))
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    for (R, C, Vlv, plv, Vrv, prv, pap), col in zip(loops, colors):
        ax[0, 0].plot(Vlv, plv, color=col, lw=1.4)
        ax[0, 1].plot(Vrv, prv, color=col, lw=1.4,
                      label=f"R={R:.3f} C={C:.2f}")
    ax[0, 0].set(title="LV PV loops", xlabel="V_LV [mL]", ylabel="p_LV [mmHg]")
    ax[0, 1].set(title="RV PV loops (0D, elastance — not FEM)",
                 xlabel="V_RV [mL]", ylabel="p_RV [mmHg]")
    ax[0, 1].legend(fontsize=7, ncol=1)

    sev = [r["k"] for r in rows]
    ax[1, 0].plot(sev, [r["RV_peak"] for r in rows], "o-", label="RV peak p")
    ax[1, 0].plot(sev, [r["PA_sys"] for r in rows], "s-", label="PA systolic")
    ax[1, 0].plot(sev, [r["PA_mean"] for r in rows], "^-", label="PA mean")
    ax[1, 0].set(title="Afterload vs severity", xlabel="case #", ylabel="mmHg")
    ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(sev, [r["RV_SW"] for r in rows], "o-", color="firebrick", label="RV stroke work")
    ax[1, 1].set(title="RV stroke work (loop area)", xlabel="case #",
                 ylabel="mmHg*mL")
    ax[1, 1].grid(alpha=0.3); ax[1, 1].legend()

    twin = ax[1, 1].twinx()
    twin.plot(sev, [r["RC"] for r in rows], "x--", color="gray", label="RC time")
    twin.set_ylabel("RC time [s]")

    fig.suptitle(f"Pulmonary windkessel sweep — baseline: {baseline}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["healthy_linear", "sPAP22_linear"],
                    choices=list(BASELINES))
    ap.add_argument("--n", type=int, default=8, help="number of sweep points")
    ap.add_argument("--beats", type=int, default=50)
    ap.add_argument("--r-lo", type=float, default=0.05)
    ap.add_argument("--r-hi", type=float, default=0.55)
    ap.add_argument("--c-hi", type=float, default=0.90)
    ap.add_argument("--c-lo", type=float, default=0.15)
    args = ap.parse_args()

    grad = pul_gradient(args.n, args.r_lo, args.r_hi, args.c_hi, args.c_lo)
    summary = {}
    for baseline in args.baselines:
        rows, loops = run_spectrum(baseline, args.n, grad, args.beats)
        outpath = HERE / f"sweep_{baseline}.png"
        plot_spectrum(baseline, rows, loops, outpath)
        summary[baseline] = rows
        print(f"\n{'='*86}\nBASELINE: {baseline}   ->  {outpath.name}\n{'='*86}")
        print(f"{'#':>2} {'R_AR':>7} {'C_AR':>6} {'RC':>6} | "
              f"{'RVpeak':>7} {'RV_EDV':>7} {'RV_SV':>6} {'RV_SW':>8} | "
              f"{'PAsys':>6} {'PAdia':>6} {'PAmean':>6} | {'LVpeak':>7}")
        for r in rows:
            print(f"{r['k']:>2} {r['R_AR']:>7.3f} {r['C_AR']:>6.2f} {r['RC']:>6.3f} | "
                  f"{r['RV_peak']:>7.1f} {r['RV_EDV']:>7.1f} {r['RV_SV']:>6.1f} {r['RV_SW']:>8.0f} | "
                  f"{r['PA_sys']:>6.1f} {r['PA_dia']:>6.1f} {r['PA_mean']:>6.1f} | {r['LV_peak']:>7.1f}")

    (HERE / "sweep_summary_0d.json").write_text(json.dumps(summary, indent=2))
    print("\nReminder: 0D RV/LV systolic is inflated and elastance-based (FEM replaces "
          "it in coupling). PA pressures + windkessel values are the transferable picks.")


if __name__ == "__main__":
    main()
