# Spectrum re-run with shared unloading — sweep C

**Status:** complete, 2026-08-31. `sims/2026-08-31/spectrum_shared_unloaded_production_20260831_184436/`
**Pilot:** `sims/2026-08-31/spectrum_shared_unloaded_pilot_20260831_163747/` (L10, 1 beat)
**Bundle:** `no_frank_starling` only. Jobs 1377341–1377347 (L5, 6 beats).

## What it is

The seven optimiser-tuned severity cases (`ukb_circ_linear`, healthy → end_stage)
re-run with the two 2026-06-22 fixes applied: a **single shared inverse-unloaded
reference** for all cases, and a **fixed coupling ratio** anchored to the
baseline node. It exists to answer one question — does the dynamic range that
made the original spectrum useful survive the corrections, or was it an artifact
of per-case unloading?

## What it showed

**The dynamic range survives, and the septum is not dead.** True work density
across the seven cases:

| region | range (range/mean) | for comparison, pulmonary sweep |
|---|---|---|
| LV | 44.5% | 14% |
| RV | 88.0% | 80% |
| **Septum** | **29.2%** | 4% |

The septum was flat on the pulmonary sweep because LV pressure was held fixed,
not because septal work is insensitive. Given an LV-pressure axis it moves, and
by a margin far above the ≥15% gate the [afterload grid
design](../superpowers/specs/2026-07-08-rv-lv-afterload-grid-design.md) set for
itself.

**The collinearity is partly broken for free.** LV systolic spans 92.9–119.5 mmHg
(26.6 mmHg, against 7.8 on the pulmonary sweep). `corr(P_LV, P_RV)` falls from
−0.982 to −0.454 and the condition number of the centred (P_LV, P_RV) design
matrix from 38.1 to 2.95.

**But P_RV loses on the RV free wall.** Pearson r against true RV work density:
P_LV +0.985, Mean/Sum +0.977, P_RV +0.936, transmural −0.471. On the magnitude
lens (through-origin calibration, relative RMSE) Mean wins at 7.8% against P_RV
at 13.1%. This contradicts the pulmonary sweep, where P_RV won magnitude
decisively (4.4% against Mean's 7.1%).

**The septum tracks LV pressure.** P_LV is best on both lenses — r = +0.871,
relative RMSE 14.5% — against P_RV's +0.718 / 25.9% and transmural's +0.282 /
22.5%. This is the first sweep with enough septal signal for the question to
mean anything, and the answer is the LV pressure.

## Why the RV result disagrees with the pulmonary sweep

Sweep C is not a pure loading experiment. The optimiser moved contractility as
well: LV `EA` ends at 74% of its baseline value while RV `EA` rises 70%, and
cardiac output falls 4.5 → 2.8 L/min, a 46% drop. Work falls when flow falls
regardless of what the pressures do. The pulmonary sweep drifts only 11.9% in
stroke volume over its eight cases, so it does not carry that confound.

The practical consequence is that sweep C cannot adjudicate the RV question, and
the disagreement between the two sweeps is a confound, not a contradiction. What
sweep C *does* establish is that the septum has recoverable signal once LV
pressure varies — which is what the grid design needed to know.

## Caveats

Single bundle (no-FS, Ta = 100 kPa); the Frank-Starling variants have not been
re-run with the fixes. Cardiac-output drift and contractility drift are
unresolved confounds by construction — they are properties of how these
circulation parameters were generated, not of the re-run.
