#!/usr/bin/env python3
"""Arm 2 — the systemic mirror of the pulmonary windkessel sweep.

Place 8 cases evenly in 0D LV-systolic (~100->160 mmHg) by sweeping ONLY the
systemic windkessel (SYS.R_AR up, SYS.C_AR down at conserved systemic RC), with
the pulmonary side pinned at the arm-1 anchor node (case0_rv25). Everything else
- geometry, material, activation, chamber elastances, valves - is untouched.

Together with the pulmonary arm this gives two mirror-image one-parameter
experiments that cross at a shared baseline: vary one circuit at a time and ask
which wall follows which pressure. Arm 2 is the specificity control - it is what
turns "P_RV correlates with RV work" into "each wall follows its own pressure".

Method is identical to make_sweep_params.py: densely sample the locus in 0D,
record LV-systolic, invert P_sys(s) with np.interp onto the targets, write the
JSONs, then re-run 0D to verify each. Pure 0D -> login-safe.
"""
from __future__ import annotations
import copy, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from compare_baselines_0d import build_params, run_0d, last_cycle_mask, ureg, HERE

ANCHOR  = HERE / "circ_params" / "case0_rv25.json"   # arm-1 node: healthy pulmonary side
OUTDIR  = HERE / "circ_params"
LV_TARGETS = np.linspace(100.0, 160.0, 8)

_base = json.load(open(ANCHOR))["parameters"]["circulation"]["SYS"]
R0, C0 = float(_base["R_AR"]), float(_base["C_AR"])
RC0 = R0 * C0                       # conserved systemic RC (s)
R_LO, R_HI = 0.55, 1.60             # brackets the 100-160 mmHg LV-systolic band


def locus(s):
    """R up along a geometric locus; C follows so that R*C stays at RC0."""
    R = R_LO * (R_HI / R_LO) ** s
    return R, RC0 / R


def run_case(R, C, tag, beats=40):
    params, ic = build_params(ANCHOR, relinearize=False)
    params["circulation"]["SYS"]["R_AR"] = R * ureg("mmHg*s/mL")
    params["circulation"]["SYS"]["C_AR"] = C * ureg("mL/mmHg")
    hist = run_0d(params, ic, beats, tag)
    m = last_cycle_mask(hist)
    plv = np.asarray(hist["p_LV"])[m]
    prv = np.asarray(hist["p_RV"])[m]
    pao = np.asarray(hist["p_AR_SYS"])[m]
    Vlv = np.asarray(hist["V_LV"])[m]
    sv = float(Vlv.max() - Vlv.min())
    return dict(LV_sys=float(plv.max()), RV_sys=float(prv.max()),
                Ao_dia=float(pao.min()), SV=sv, CO=sv * 0.075,
                LV_EDV=float(Vlv.max()), hist=hist)


def main():
    base = json.load(open(ANCHOR))
    print(f"anchor: {ANCHOR.name}   SYS R={R0:.4f} C={C0:.4f}  (RC={RC0:.4f} s, conserved)")

    s_dense = np.linspace(0.0, 1.0, 12)
    lv_dense = []
    for i, s in enumerate(s_dense):
        R, C = locus(s)
        v = run_case(R, C, f"sysdense_{i}", beats=30)
        lv_dense.append(v["LV_sys"])
        print(f"  dense s={s:4.2f}  R={R:6.4f} C={C:6.4f} -> LV-sys {v['LV_sys']:6.1f} mmHg, CO {v['CO']:.2f}")
    lv_dense = np.array(lv_dense)
    if not np.all(np.diff(lv_dense) > 0):
        print("WARNING: LV-systolic is not monotone in s; inversion may be unreliable")

    s_targets = np.interp(LV_TARGETS, lv_dense, s_dense)

    rows, loops = [], []
    for k, (lvT, s) in enumerate(zip(LV_TARGETS, s_targets)):
        R, C = locus(s)
        case = copy.deepcopy(base)
        case["parameters"]["circulation"]["SYS"]["R_AR"] = R
        case["parameters"]["circulation"]["SYS"]["C_AR"] = C
        case["case"] = dict(index=k, arm="systemic", target_LV_sys=float(lvT),
                            severity_s=float(s), R_AR_SYS=R, C_AR_SYS=C)
        case["description"] = (f"Systemic-windkessel arm case {k}: target LV-sys {lvT:.0f} mmHg, "
                               f"SYS R_AR={R:.4f}, C_AR={C:.4f} (RC conserved at {RC0:.4f}s); "
                               f"pulmonary side pinned at the case0_rv25 anchor.")
        name = f"sys{k}_lv{round(lvT):03d}.json"
        (OUTDIR / name).write_text(json.dumps(case, indent=2))

        v = run_case(R, C, f"sysverify_{k}", beats=30)
        rows.append(dict(k=k, name=name, lvT=float(lvT), R=R, C=C,
                         LV_sys=v["LV_sys"], RV_sys=v["RV_sys"], Ao_dia=v["Ao_dia"],
                         SV=v["SV"], CO=v["CO"], LV_EDV=v["LV_EDV"]))
        m = last_cycle_mask(v["hist"])
        loops.append((np.asarray(v["hist"]["V_LV"])[m], np.asarray(v["hist"]["p_LV"])[m]))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for (V, p), col, r in zip(loops, cm.viridis(np.linspace(0, 1, len(rows))), rows):
        ax[0].plot(V, p, color=col, lw=1.5, label=f"lv{round(r['lvT'])} R={r['R']:.3f}")
    ax[0].set(title="LV PV loops, 8 systemic cases (0D)", xlabel="V_LV [mL]", ylabel="p_LV [mmHg]")
    ax[0].legend(fontsize=7)
    ax[1].plot([r["lvT"] for r in rows], [r["LV_sys"] for r in rows], "o-", label="achieved")
    ax[1].plot([r["lvT"] for r in rows], [r["lvT"] for r in rows], "k--", alpha=.4, label="target")
    ax[1].set(title="LV-systolic: target vs achieved", xlabel="target [mmHg]", ylabel="0D LV-sys [mmHg]")
    ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot([r["LV_sys"] for r in rows], [r["CO"] for r in rows], "o-", color="C3", label="CO")
    ax[2].axhline(rows[0]["CO"] * 0.85, ls=":", color="k", alpha=.6, label="-15% of first case")
    ax[2].set(title="Cardiac-output drift across the arm", xlabel="LV-sys [mmHg]", ylabel="CO [L/min]")
    ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE / "systemic_sweep_params_8cases.png", dpi=140)

    co = np.array([r["CO"] for r in rows])
    print("\n" + "=" * 96)
    print("8 systemic windkessel cases written to circ_params/  (even in 0D LV-systolic)")
    print("=" * 96)
    print(f"{'#':>2} {'file':20} {'LVtgt':>6} {'R_AR':>7} {'C_AR':>6} {'LVsys':>6} "
          f"{'RVsys':>6} {'AoDia':>6} {'SV':>6} {'CO':>5} {'LV_EDV':>7}")
    for r in rows:
        print(f"{r['k']:>2} {r['name']:20} {r['lvT']:>6.0f} {r['R']:>7.4f} {r['C']:>6.3f} "
              f"{r['LV_sys']:>6.1f} {r['RV_sys']:>6.1f} {r['Ao_dia']:>6.1f} "
              f"{r['SV']:>6.1f} {r['CO']:>5.2f} {r['LV_EDV']:>7.1f}")
    print(f"\ncardiac-output drift across the arm: {100*(co.max()-co.min())/co.mean():.1f}% of mean "
          f"(pulmonary arm is 11.9%; keep under ~15%)")
    print(f"RV-systolic drift (should be small - pulmonary side is pinned): "
          f"{max(r['RV_sys'] for r in rows) - min(r['RV_sys'] for r in rows):.1f} mmHg")
    print(f"figure: {HERE / 'systemic_sweep_params_8cases.png'}")
    (OUTDIR / "systemic_sweep_manifest.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
