#!/usr/bin/env python3
"""Place 8 pulmonary-windkessel cases evenly in 0D PA-systolic (~25->90 mmHg).

Both knobs co-vary along one geometric PAH locus (R_AR up, C_AR down). We:
  1. densely sample severity s in [0,1] along the locus on the LINEAR baseline,
     recording 0D PA-systolic (the windkessel-determined afterload proxy),
  2. invert PA_sys(s) to find the s (hence R_AR, C_AR) that hit
     PA-systolic = linspace(25, 90, 8),
  3. write 8 circulation JSONs identical to the baseline except for
     PUL.R_AR / PUL.C_AR, and verify each by re-running 0D.

Caveat kept in mind: coupled RV systolic != 0D PA systolic exactly (FEM
ventricle + drifted lib). PA is the best 0D proxy; final magnitudes get
confirmed once the coupled batch lands. Pure 0D -> login-safe.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from compare_baselines_0d import build_params, run_0d, last_cycle_mask, ureg, HERE, OUT

BASELINE = HERE / "circ_params" / "baseline_linear.json"
OUTDIR = HERE / "circ_params"

# geometric locus endpoints. s=0 is a genuinely NORMAL pulmonary circulation
# (PA-sys ~25, mean ~14); s=1 is severe PAH. Both knobs co-vary (R up, C down).
R_LO, R_HI = 0.030, 0.45
C_HI, C_LO = 1.20, 0.16
PA_TARGETS = np.linspace(25.0, 90.0, 8)


def locus(s):
    R = R_LO * (R_HI / R_LO) ** s
    C = C_HI * (C_LO / C_HI) ** s
    return R, C


def pa_systolic(R, C, tag, beats=40):
    params, ic = build_params(BASELINE, relinearize=False)
    params["circulation"]["PUL"]["R_AR"] = R * ureg("mmHg*s/mL")
    params["circulation"]["PUL"]["C_AR"] = C * ureg("mL/mmHg")
    hist = run_0d(params, ic, beats, tag)
    m = last_cycle_mask(hist)
    pap = np.asarray(hist["p_AR_PUL"])[m]
    prv = np.asarray(hist["p_RV"])[m]
    Vrv = np.asarray(hist["V_RV"])[m]
    return dict(PA_sys=float(pap.max()), PA_mean=float(pap.mean()),
                RV_peak=float(prv.max()), RV_EDV=float(Vrv.max()),
                RV_SV=float(Vrv.max() - Vrv.min()), hist=hist)


def main():
    base = json.load(open(BASELINE))

    # 1. dense map s -> PA_sys (smooth monotonic -> 12 points suffice for interp)
    s_dense = np.linspace(0.0, 1.0, 12)
    pa_dense = []
    for i, s in enumerate(s_dense):
        R, C = locus(s)
        pa_dense.append(pa_systolic(R, C, f"dense_{i}", beats=30)["PA_sys"])
    pa_dense = np.array(pa_dense)

    # 2. invert for the 8 PA targets (PA_sys monotonic increasing in s)
    s_targets = np.interp(PA_TARGETS, pa_dense, s_dense)

    # 3. write + verify the 8 cases
    rows, loops = [], []
    for k, (paT, s) in enumerate(zip(PA_TARGETS, s_targets)):
        R, C = locus(s)
        case = copy.deepcopy(base)
        case["parameters"]["circulation"]["PUL"]["R_AR"] = R
        case["parameters"]["circulation"]["PUL"]["C_AR"] = C
        case["case"] = dict(index=k, target_PA_sys=float(paT),
                            severity_s=float(s), R_AR_PUL=R, C_AR_PUL=C)
        case["description"] = (base.get("description", "") +
                               f"  [case {k}: target PA-sys {paT:.0f} mmHg, "
                               f"R_AR={R:.4f}, C_AR={C:.4f}]")
        name = f"case{k}_pa{round(paT):02d}.json"
        (OUTDIR / name).write_text(json.dumps(case, indent=2))

        v = pa_systolic(R, C, f"verify_{k}", beats=30)
        rows.append(dict(k=k, name=name, paT=float(paT), R=R, C=C, RC=R * C,
                         PA_sys=v["PA_sys"], PA_mean=v["PA_mean"],
                         RV_peak=v["RV_peak"], RV_EDV=v["RV_EDV"], RV_SV=v["RV_SV"]))
        m = last_cycle_mask(v["hist"])
        loops.append((np.asarray(v["hist"]["V_RV"])[m], np.asarray(v["hist"]["p_RV"])[m]))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = cm.viridis(np.linspace(0, 1, 8))
    for (Vrv, prv), col, r in zip(loops, colors, rows):
        ax[0].plot(Vrv, prv, color=col, lw=1.5, label=f"pa{round(r['paT'])} R={r['R']:.3f}")
    ax[0].set(title="RV PV loops, 8 cases (0D)", xlabel="V_RV [mL]", ylabel="p_RV [mmHg]")
    ax[0].legend(fontsize=7)
    ax[1].plot([r["paT"] for r in rows], [r["PA_sys"] for r in rows], "o-", label="achieved PA-sys")
    ax[1].plot([r["paT"] for r in rows], [r["paT"] for r in rows], "k--", alpha=0.4, label="target")
    ax[1].set(title="PA-systolic: target vs achieved", xlabel="target [mmHg]", ylabel="0D PA-sys [mmHg]")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "sweep_params_8cases.png", dpi=140)

    print("\n" + "=" * 92)
    print("8 pulmonary windkessel cases written to circ_params/  (even in 0D PA-systolic)")
    print("=" * 92)
    print(f"{'#':>2} {'file':22} {'PAtgt':>6} {'R_AR':>7} {'C_AR':>6} {'RC':>6} "
          f"{'PAsys':>6} {'PAmean':>6} {'RVpeak':>7} {'RV_EDV':>7}")
    for r in rows:
        print(f"{r['k']:>2} {r['name']:22} {r['paT']:>6.0f} {r['R']:>7.4f} {r['C']:>6.3f} "
              f"{r['RC']:>6.3f} {r['PA_sys']:>6.1f} {r['PA_mean']:>6.1f} "
              f"{r['RV_peak']:>7.1f} {r['RV_EDV']:>7.1f}")
    print(f"\nfigure: {HERE / 'sweep_params_8cases.png'}")
    print("RV_peak/RV_EDV are 0D-elastance (FEM replaces in coupling); PA cols are the proxy.")
    (OUTDIR / "sweep_manifest.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
