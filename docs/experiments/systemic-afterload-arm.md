# Systemic-afterload arm — arm 2, the specificity control

**Status:** submitted 2026-09-01, jobs 1377593–1377600 (8 cases, no-FS, L5, 6 beats).
`sims/2026-09-01/pah_pulmonary_20260901_systemic/`
**Parameters:** `pah_pulmonary_batch/make_systemic_sweep_params.py` →
`circ_params/sys{0..7}_lv{100..160}.json`

## What it is

The mirror image of the [pulmonary afterload sweep](pulmonary-afterload-sweep.md).
Eight cases sweeping **only** `SYS.R_AR` up and `SYS.C_AR` down along one
geometric locus at conserved systemic RC, with the pulmonary side pinned at the
pulmonary arm's own anchor node (`case0_rv25`). Targets are placed by inverting
0D LV-systolic onto an even spread from 100 to 160 mmHg, the same
interpolation-inversion the pulmonary arm uses. Same mesh, same shared
inverse-unloaded reference, same fixed coupling ratios (LV 1.02479 / RV 0.88262),
same activation.

## Why it is the experiment that matters

The pulmonary arm alone supports "P_RV tracks RV work". That is a weaker claim
than it looks, because on a single monotone sweep almost anything tracks
anything — the RV free wall scores r = +0.998 against the *LV* pressure there.

Two mirror-image one-parameter experiments support something much stronger:
**each wall follows its own pressure, demonstrated by varying one circuit at a
time.** That is a control, it is symmetric, and it takes one sentence to state in
a methods section. The systemic arm is where the LV and septal claims come from,
and it is what makes the RV claim specific rather than merely correlational.

It also settles the question the [design scan](../findings/proxy-identifiability.md)
left open. The interdependence coefficient ∂W_RV/∂P_LV is currently estimable
only from the [shared-unloaded spectrum](spectrum-shared-unloaded.md), where
contractility and cardiac output move too. This arm measures it at fixed
contractility, and its value decides whether any 2-D design can crown P_RV by
correlation.

## 0D verification, before any FEM

| | |
|---|---|
| LV-systolic achieved | 99.9 → 160.0 mmHg against targets 100 → 160 |
| RV-systolic drift | **0.4 mmHg** across the whole arm — the pulmonary side really is pinned |
| cardiac-output drift | **4.5%** of mean (pulmonary arm 11.9%; the gate is 15%) |
| systemic RC | conserved at 1.113 s by construction |

The cardiac-output number matters. Sweep C's output falls 46% across its cases,
which is why its RV numbers cannot be trusted — work falls when flow falls
whatever the pressures do. This arm changes LV afterload by 60 mmHg while moving
output by 4.5%, so it is a cleaner loading experiment than either predecessor.

## What to check when it lands

1. **Dynamic range.** LV and septal true work density should both move
   materially; the pooled linear model predicts roughly 33% and 29%. If the
   septum stays flat here it is flat for a reason other than fixed LV pressure,
   and the septal question needs rethinking rather than more loading.
2. **∂W_RV/∂P_LV**, from the RV free wall's slope against LV systolic. This is
   the number the grid design turns on.
3. **Specificity.** P_LV should win the LV free wall and the septum; P_RV should
   *not* win the RV free wall here, since RV loading barely moves. That
   asymmetry between the two arms is the result.
4. **Preload not clamped** — confirm FEM LV EDV spreads across the arm rather
   than being re-normalised per case.

## Housekeeping

Each case writes ~19 GB, of which ~17 GB is `visualization/` (ParaView output
that no analysis reads). `/global/D1` is at 99%. Delete `visualization/` per case
once postprocessing has produced `per_cell_data.npz` and `metrics/`; that takes
the arm from ~145 GB to ~16 GB.
