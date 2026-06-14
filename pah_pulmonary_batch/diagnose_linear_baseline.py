#!/usr/bin/env python3
"""Is the linear-EDPVR baseline sound enough to build the batch on?

The naive plan is "take sPAP22, set kE=0". This script shows, with numbers,
what that does to the 0D warm-up relative to the SHARED UKB L5 mesh it must
seed -- and what a proper linear re-fit would need instead.

Shared UKB L5 mesh ED cavity volumes (from a prior sim's simulation_params):
    LV EDV = 111.5 mL   (unloaded 83.1)
    RV EDV =  76.9 mL   (unloaded 50.3)
The 0D warm-up sets the coupled initial state + the ED unloading target, so
its EDV should land near these and its end-diastolic pressure should be
physiological (LV ~8, RV ~5 mmHg).

We compare three ventricular EDPVRs on the SAME sPAP22 circulation:
  exp      : original Klotz/exponential (reference)
  linear   : kE -> 0, EB unchanged           (the naive strip)
  linfit   : kE -> 0, EB re-fit to hit target EDP at the mesh EDV

Reminder: in the coupled run FEM replaces the ventricle, so this only governs
warm-up quality / the ED target. But a baseline that seeds 129 mL into an
111 mL cavity at 2 mmHg is sloppy provenance for the paper.

Pure 0D -> login-node safe.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_baselines_0d import build_params, run_0d, last_cycle_mask, ureg, WORK, HERE

SPAP22 = WORK / "data/ukb_circ_v12_exp/optimized_regazzoni_ukb_sPAP22.json"

# Shared UKB L5 mesh ED cavity volumes [mL] and physiological ED pressure targets [mmHg]
MESH = {"LV": dict(EDV=111.5, EDP_target=8.0), "RV": dict(EDV=76.9, EDP_target=5.0)}


def edpvr_pressure(EB, V0, kE, V):
    """Diastolic pressure of the Regazzoni EDPVR at volume V (EB diastolic slope)."""
    dV = V - V0
    if kE and kE > 0:
        return (EB / kE) * (np.exp(kE * dV) - 1.0)
    return EB * dV


def warmup_stats(params, ic, tag):
    hist = run_0d(params, ic, 40, tag)
    m = last_cycle_mask(hist)
    out = {}
    for ch in ("LV", "RV"):
        V = np.asarray(hist[f"V_{ch}"])[m]
        p = np.asarray(hist[f"p_{ch}"])[m]
        EB = params["chambers"][ch]["EB"]; EB = EB.magnitude if hasattr(EB, "magnitude") else EB
        V0 = params["chambers"][ch]["V0"]; V0 = V0.magnitude if hasattr(V0, "magnitude") else V0
        kE = params["chambers"][ch].get("kE", 0.0)
        edv = float(V.max())
        out[ch] = dict(
            EDV=edv, ESV=float(V.min()), SV=float(V.max() - V.min()),
            EF=100 * (V.max() - V.min()) / V.max(),
            EDP_law=float(edpvr_pressure(EB, V0, kE, edv)),  # honest diastolic p at EDV
            p_peak=float(p.max()), EB=EB, V0=V0, kE=float(kE),
        )
    return out, hist


def main():
    # ---- base params (exp) and the two linear variants -------------------
    p_exp, ic = build_params(SPAP22, relinearize=False)
    p_lin, _ = build_params(SPAP22, relinearize=True)

    # re-fit EB so the LINEAR EDPVR gives target EDP at the mesh EDV
    p_fit, _ = build_params(SPAP22, relinearize=True)
    refit = {}
    for ch in ("LV", "RV"):
        V0 = p_fit["chambers"][ch]["V0"]; V0 = V0.magnitude if hasattr(V0, "magnitude") else V0
        EB_new = MESH[ch]["EDP_target"] / (MESH[ch]["EDV"] - V0)
        p_fit["chambers"][ch]["EB"] = EB_new * ureg("mmHg/mL")
        refit[ch] = EB_new

    variants = [("exp", p_exp), ("linear", p_lin), ("linfit", p_fit)]
    results, hists = {}, {}
    for tag, params in variants:
        results[tag], hists[tag] = warmup_stats(params, ic, f"diag_{tag}")

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for tag in ("exp", "linear", "linfit"):
        m = last_cycle_mask(hists[tag])
        ax[0].plot(np.asarray(hists[tag]["V_LV"])[m], np.asarray(hists[tag]["p_LV"])[m], label=tag)
        ax[1].plot(np.asarray(hists[tag]["V_RV"])[m], np.asarray(hists[tag]["p_RV"])[m], label=tag)
    for a, ch in zip(ax, ("LV", "RV")):
        a.axvline(MESH[ch]["EDV"], ls="--", color="k", alpha=0.5, label=f"mesh EDV={MESH[ch]['EDV']}")
        a.set(title=f"{ch} PV loop (sPAP22 warm-up)", xlabel=f"V_{ch} [mL]", ylabel=f"p_{ch} [mmHg]")
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "linear_baseline_diag.png", dpi=140)

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 76)
    print("Linear-EDPVR baseline readiness  (sPAP22 circulation, 0D warm-up)")
    print(f"  shared UKB mesh ED targets:  LV EDV={MESH['LV']['EDV']} (EDP~8),  "
          f"RV EDV={MESH['RV']['EDV']} (EDP~5)")
    print("=" * 76)
    for ch in ("LV", "RV"):
        tgt = MESH[ch]
        print(f"\n{ch}  (target EDV={tgt['EDV']} mL, target EDP={tgt['EDP_target']} mmHg)")
        print(f"  {'variant':8} {'EB':>8} {'kE':>8} {'EDV':>7} {'dV_mesh':>8} "
              f"{'EDP_law':>8} {'EF':>6} {'p_peak':>7}")
        for tag in ("exp", "linear", "linfit"):
            r = results[tag][ch]
            print(f"  {tag:8} {r['EB']:>8.4f} {r['kE']:>8.4f} {r['EDV']:>7.1f} "
                  f"{r['EDV']-tgt['EDV']:>+8.1f} {r['EDP_law']:>8.2f} {r['EF']:>6.1f} {r['p_peak']:>7.1f}")
    print(f"\nre-fit EB to hit target EDP at mesh EDV:  "
          f"LV {refit['LV']:.4f}  RV {refit['RV']:.4f}  mmHg/mL")
    print(f"figure: {HERE / 'linear_baseline_diag.png'}")
    (HERE / "linear_baseline_diag.json").write_text(json.dumps(
        {"results": results, "refit_EB": refit, "mesh": MESH}, indent=2))


if __name__ == "__main__":
    main()
