#!/usr/bin/env python3
"""Generate (and verify) the linear-EDPVR baseline circulation params.

Baseline = sPAP22 (UKB-matched) with the ventricular EDPVR made LINEAR:
  * kE removed for LV and RV (no Klotz/exponential)
  * diastolic slope EB re-fit so the linear law gives the physiological ED
    pressure at the shared UKB L5 mesh ED volume:
        EB = EDP_target / (mesh_EDV - V0)
Everything else (systemic + pulmonary windkessel, atria, valves) is kept from
sPAP22, which was optimised for this mesh. The 8 PAH cases will differ from
this baseline ONLY in the pulmonary windkessel R_AR_PUL / C_AR_PUL.

Writes pah_pulmonary_batch/circ_params/baseline_linear.json and prints a
warm-up verification (0D, login-safe).
"""
from __future__ import annotations
import json
from pathlib import Path

from compare_baselines_0d import build_params, run_0d, last_cycle_mask, WORK, HERE
import numpy as np

SRC = WORK / "data/ukb_circ_v12_exp/optimized_regazzoni_ukb_sPAP22.json"
OUTDIR = HERE / "circ_params"
OUTDIR.mkdir(exist_ok=True)

# shared UKB L5 mesh ED cavity volumes [mL] + physiological ED pressure targets
MESH = {"LV": dict(EDV=111.5, EDP=8.0), "RV": dict(EDV=76.9, EDP=5.0)}


def main():
    data = json.load(open(SRC))
    ch = data["parameters"]["chambers"]

    refit = {}
    for c in ("LV", "RV"):
        ch[c].pop("kE", None)                       # -> linear EDPVR
        V0 = ch[c]["V0"]
        EB_new = MESH[c]["EDP"] / (MESH[c]["EDV"] - V0)
        ch[c]["EB"] = EB_new
        refit[c] = EB_new

    data["description"] = (
        "PAH pulmonary-windkessel batch baseline. sPAP22 (UKB L5) with LINEAR "
        "ventricular EDPVR: kE removed, EB re-fit to give EDP~8/5 mmHg at the "
        "shared mesh ED volumes (LV 111.5, RV 76.9 mL). Only R_AR_PUL/C_AR_PUL "
        "vary across the 8 PAH cases; everything else is fixed."
    )
    data["edpvr"] = "linear"
    data["refit_EB_mmHg_per_mL"] = refit
    data["mesh_ed_targets"] = MESH

    out = OUTDIR / "baseline_linear.json"
    out.write_text(json.dumps(data, indent=2))

    # ---- verify: warm-up lands on the mesh ED state ----------------------
    params, ic = build_params(out, relinearize=False)  # JSON already linear
    hist = run_0d(params, ic, 40, "baseline_verify")
    m = last_cycle_mask(hist)
    print("\n" + "=" * 70)
    print(f"baseline_linear.json written -> {out}")
    print("=" * 70)
    print(f"re-fit EB: LV {refit['LV']:.4f}, RV {refit['RV']:.4f} mmHg/mL")
    print(f"pulmonary baseline: R_AR={ch['LV'] and data['parameters']['circulation']['PUL']['R_AR']:.4f}, "
          f"C_AR={data['parameters']['circulation']['PUL']['C_AR']:.4f}")
    print(f"\n{'ch':3} {'EDV':>7} {'target':>7} {'Δ':>6} {'EDP_law':>8} {'EF':>6} {'peak':>6}")
    for c in ("LV", "RV"):
        V = np.asarray(hist[f"V_{c}"])[m]
        p = np.asarray(hist[f"p_{c}"])[m]
        edv = float(V.max())
        EB = data["parameters"]["chambers"][c]["EB"]
        V0 = data["parameters"]["chambers"][c]["V0"]
        edp = EB * (edv - V0)
        ef = 100 * (V.max() - V.min()) / V.max()
        print(f"{c:3} {edv:>7.1f} {MESH[c]['EDV']:>7.1f} {edv-MESH[c]['EDV']:>+6.1f} "
              f"{edp:>8.2f} {ef:>6.1f} {float(p.max()):>6.1f}")


if __name__ == "__main__":
    main()
