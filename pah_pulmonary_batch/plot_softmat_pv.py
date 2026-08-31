#!/usr/bin/env python3
"""PV loops for the softer-material pilot: LV and RV, all 6 runs on two panels.

x = FEM cavity volume (V_{LV,RV}_FEM, the volume the FEM actually sees),
y = cavity pressure (p_{LV,RV}, mmHg). Color = stiffness scale, style = case
(baseline rv25 solid / severe rv95 dashed). Login-safe (numpy + matplotlib).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

ROOT = paths.RESULTS_ROOT / "sims/2026-07-08/softmat_pilot_L10"
SCALES = [("100", 1.00, "#1a9850"), ("050", 0.50, "#fd8d3c"), ("033", 0.33, "#d73027")]
CASES = [("case0_rv25", "baseline rv25", "-"), ("case7_rv95", "severe rv95", "--")]


def load(cd):
    m = np.load(cd / "metrics/metrics_downsample_1.npy", allow_pickle=True).item()
    return {"V_LV": np.asarray(m["V_LV_FEM"], float), "p_LV": np.asarray(m["p_LV"], float),
            "V_RV": np.asarray(m["V_RV_FEM"], float), "p_RV": np.asarray(m["p_RV"], float)}


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    rows = []
    for tag, sval, col in SCALES:
        for case, clabel, ls in CASES:
            cd = ROOT / f"scale{tag}" / case
            if not (cd / "metrics/metrics_downsample_1.npy").exists():
                print(f"missing: {cd}"); continue
            d = load(cd)
            lab = f"{sval:.2f}x, {clabel}"
            axL.plot(d["V_LV"], d["p_LV"], ls, color=col, lw=1.8, label=lab)
            axR.plot(d["V_RV"], d["p_RV"], ls, color=col, lw=1.8, label=lab)
            for reg in ("LV", "RV"):
                V, P = d[f"V_{reg}"], d[f"p_{reg}"]
                rows.append((f"{sval:.2f}x", case, reg, V.max(), V.min(),
                             V.max() - V.min(), P.max()))
    for ax, reg in ((axL, "LV"), (axR, "RV")):
        ax.set_title(f"{reg} PV loops — softer-material pilot (L10, 1 beat)", fontsize=12)
        ax.set_xlabel(f"$V_{{{reg}}}$ (FEM)  [mL]"); ax.set_ylabel(f"$p_{{{reg}}}$  [mmHg]")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    out = ROOT / "pv_loops_lv_rv.png"
    fig.savefig(str(out), dpi=160); fig.savefig(str(out).replace(".png", ".pdf"))
    print(f"\nwrote {out}")
    print(f"\n{'scale':6s} {'case':11s} {'reg':4s} {'EDV':>7s} {'ESV':>7s} {'SV':>6s} {'Ppk':>7s}")
    for r in rows:
        print(f"{r[0]:6s} {r[1]:11s} {r[2]:4s} {r[3]:7.1f} {r[4]:7.1f} {r[5]:6.1f} {r[6]:7.1f}")


if __name__ == "__main__":
    main()
