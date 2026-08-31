# The septal pressure proxy

## The question

The septum is loaded by `P_LV` on one side and `P_RV` on the other. In health
`P_RV` is small and `P_LV` is a serviceable approximation. In pulmonary arterial
hypertension `P_RV` rises toward `P_LV`, the septum flattens (the D-sign), and
the choice stops being obvious. Six candidates are evaluated:

| Candidate | Definition |
|---|---|
| `PLV` | `P_LV` |
| `PRV` | `P_RV` |
| `Trans` | `P_LV − P_RV` (transmural) |
| `Mean` | `½(P_LV + P_RV)` |
| `NearestSide` | `P_LV` where τ < 0.5, else `P_RV` |
| `TauWeighted` | `(1−τ)·P_LV + τ·P_RV` |

`τ` is the transmural coordinate, 0 on the LV side and 1 on the RV side. A
seventh key, `Sum = P_LV + P_RV`, is emitted by `metrics_calculator` but equals
exactly `2 × Mean` — verified bit-for-bit — so it is affine in an existing
candidate and gives an identical correlation. It differs only in the
scale-sensitive ratio view, which is why it is kept.

## Transmural is the worst candidate, and the earlier claim was an artifact

This is the most important thing to know about the septal result, because the
project's **early** conclusion was the opposite.

The original "transmural is best" finding came from runs that used a *per-case*
inverse unloading step. That step gave each case its own unloaded reference
geometry, and since every strain is measured against the reference, the
per-case references introduced a case-dependent bias that happened to flatter
transmural pressure. Switching to a **single shared inverse-unloaded reference**
removes the artifact — and transmural's apparent advantage disappears with it.

Every current analysis agrees. On the pulmonary-loading sweep with the shared
reference, `P_RV` tracks septal work best (r ≈ 0.75 in the no-Frank-Starling
bundle), with `Mean` and `Sum` just behind, while transmural is worst
(r ≈ 0.43, dropping further in the Frank-Starling bundles). On the capped
thesis sweep, transmural is the only candidate that *anti*-correlates. And in
the August analysis transmural anti-tracks the RV free wall too (r = −0.77).

## The capped thesis sweep, in full

All 16 cases, `results/analysis/capped_shared_l5_sweep_20260510_141015/`:

| Candidate | r (fibre strain) | r (longitudinal strain) | Ratio preservation (mean abs log error) |
|---|---|---|---|
| TauWeighted | **+0.979** | +0.540 | 0.970 |
| Mean | +0.976 | +0.535 | 0.969 |
| NearestSide | +0.944 | +0.547 | 0.999 |
| PLV | +0.842 | +0.540 | **0.805** |
| PRV | +0.803 | +0.527 | 1.171 |
| Trans | **−0.276** | **−0.331** | **2.075** |

Two things beyond the transmural result stand out.

**The strain definition matters more than the pressure choice.** Against fibre
strain the good candidates reach r ≈ 0.98; against longitudinal strain
*everything* collapses to r ≈ 0.54. Longitudinal strain is what clinical
imaging actually measures (GLS), so that is the more clinically relevant column
— and in it no pressure choice rescues the proxy. The ceiling is set by the
strain direction, not by the pressure. This is why the defense-stage work
concluded that a better pressure choice cannot fix the correlation.

**In the clinically relevant column, no candidate is good.** An r of 0.54,
r² ≈ 0.29, means the proxy explains under a third of the variance in true
septal work.

## Why the two sweeps rank the leaders differently

The capped thesis sweep puts `TauWeighted`/`Mean` on top against fibre strain;
the pulmonary sweep puts `P_RV` on top. Both are consistent about transmural
being worst, and the difference in the leader is expected: the sweeps differ in
design, and on the pulmonary sweep — where LV loading is held fixed — the
non-transmural candidates are affine in `P_RV` and therefore
[not separable by correlation at all](../open-questions.md#rv--lv-afterload-grid).
Read the leader ordering as weakly determined; read "transmural is worst" as
robust.

## What is still open

The septal correlation is mask-sensitive in a way the free-wall correlations are
not — 0.75 by the thesis convention against 0.93 with the canonical region mask
— and that reconciliation against `sweep_analysis.py` is
[still open](../open-questions.md#septal-r-is-mask-sensitive-and-unreconciled).
Quote septal numbers with that caveat attached.

Restricting to the [AHA mid-ventricular ring](../experiments/supporting-studies.md#aha-mid-ventricular-ring)
raises the septal `P_RV` correlation from 0.75 to 0.95, which suggests much of
the whole-wall degradation is basal and apical geometry being averaged in.
Transmural remains worst under that banding as well.
