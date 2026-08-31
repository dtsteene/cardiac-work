# What correlation can and cannot decide

**Date:** 2026-08-31. Supersedes the correlation half of the [afterload grid
design](../superpowers/specs/2026-07-08-rv-lv-afterload-grid-design.md).

The grid design assumed that breaking the (P_LV, P_RV) collinearity would make
Pearson correlation discriminating again. Analysis across the two completed
sweeps says that assumption is wrong for the RV free wall, and says why.

## The scan

A linear model `y = c₀ + a·P_LV + b·P_RV` was fitted to the pooled fifteen case
points of the [pulmonary sweep](../experiments/pulmonary-afterload-sweep.md) and
the [shared-unloaded spectrum](../experiments/spectrum-shared-unloaded.md), for
true work density and for each candidate proxy, per region (R² 0.87–0.90). Any
proposed 2-D loading design can then be evaluated before it is run.

Eighteen designs were scanned, varying the LV systolic span from fixed to
80–180 mmHg against three RV spans. Pearson r on the RV free wall:

| LV span (mmHg) | r(P_LV) | r(P_RV) |
|---|---|---|
| fixed at 120 | +1.000 | +1.000 |
| 110–130 | +1.000 | +0.981 |
| 100–140 | +0.999 | +0.932 |
| 90–160 | +0.998 | +0.845 |
| 80–180 | +0.998 | +0.776 |

**Widening the LV axis makes the RV result worse, not better.** No design in the
scan crowns P_RV on the RV free wall by correlation.

## Why

The fitted sensitivity of true RV work is 0.051 kPa per mmHg of LV pressure
against 0.066 per mmHg of RV pressure — a ratio of 0.8. RV work is nearly as
sensitive to LV pressure as to its own, which is ventricular interdependence
through the shared septum doing exactly what it should. A proxy built on P_LV
therefore carries real information about RV work, and once the LV axis is wide
enough to matter, it carries more variance than P_RV does.

This is not a design flaw to be engineered away. It is the physics of a
two-chamber pump sharing a wall.

For the other two regions the same fit gives sensitivity ratios |a/b| of 6.6
(LV) and 2.5 (septum) — both LV-pressure-dominated, consistent with P_LV winning
both lenses in every measured sweep.

## What actually governs a design's separating power

Every candidate is a linear weighting of the two pressures: P_LV is (1,0), P_RV
is (0,1), Mean is (½,½), Sum is (1,1), transmural is (1,−1). Two candidates
produce perfectly correlated series whenever the case points lie on a **line** in
(P_LV, P_RV) space — on a line both are affine in one parameter, so Pearson
cannot tell them apart. This is a property of the design alone, before any
simulation is run.

Scoring designs by `1 − max|r|` between any two different candidates (Mean and
Sum excluded, since Sum = 2·Mean can never be separated by correlation at all):

| design | n | P_LV/P_RV range | separability |
|---|---|---|---|
| pulmonary-only arm | 8 | 3.25 | **0.000** |
| systemic-only arm | 8 | 1.45 | **0.000** |
| anti-diagonal arm | 8 | 5.58 | **0.000** |
| pulmonary + systemic cross | 15 | 3.83 | 0.078 |
| cross + 4 anti-diagonal corners | 19 | 5.40 | 0.129 |
| full 4×4 grid | 16 | 5.22 | **0.178** |

Two things follow. **Any single one-dimensional arm is degenerate whatever its
direction** — steering it along the anti-diagonal to maximise the P_LV/P_RV
ratio range does not help, because it is still a line. And **even the best design
here scores 0.178**: the candidates remain mutually correlated at |r| ≈ 0.82,
because they are all linear combinations of only two pressures. The grid is a
large relative improvement over nothing and a modest absolute one.

Condition number is the wrong summary to optimise. The cross has a respectable
condition number of 2.2 but under half the grid's separating power.

## The one number everything depends on

The interdependence coefficient ∂W_RV/∂P_LV is estimated **only** from sweep C,
where contractility and cardiac output vary simultaneously. It is therefore
confounded and may be overstated several-fold. The pulmonary sweep cannot help:
its LV span is 7.8 mmHg.

Re-running the design scan with that one coefficient reduced four-fold flips the
verdict completely:

| ∂W_RV/∂P_LV | correlation crowns, RV free wall |
|---|---|
| as fitted (0.051) | P_LV, in every design |
| four times smaller (0.013) | **P_RV**, in both the cross and the 4×4 grid |

**Consequence for the pilot.** The grid design's first gate was septal dynamic
range. Sweep C has already cleared that gate at 29.2%. The gate that now matters
is measuring ∂W_RV/∂P_LV cleanly — at fixed contractility and fixed pulmonary
loading — because it decides whether a 2-D design can answer the RV question at
all. A systemic-afterload-only arm measures it directly.

## Noise does not rescue correlation

Four thousand Monte-Carlo draws per level, Gaussian noise on each proxy from 0
to 35% of its own spread, on the predicted 4×4 grid: P_RV ranks first on the RV
free wall in **0%** of draws at every noise level. Noise degrades the candidates
roughly equally; it blurs a ranking rather than reversing one.

What noise is genuinely for is narrower and still worth doing. It puts error
bars on a comparison that is currently two bare numbers, turning "P_RV beats
Mean" into a testable claim with a win rate. It exposes spurious winners on a
flat truth — transmural pressure "wins" the septum on the pulmonary sweep at
3.4% relative RMSE purely because septal work is flat there and the flattest
proxy wins by default. And it is what a clinical measurement actually looks like.
Put it on the inputs — strain and pressure — not on the finished proxy work.

## What this means for the claims

Correlation should not be asked to crown a single pressure for the RV free wall.
The magnitude lens — one-constant calibration, relative RMSE — is not a fallback
for a degenerate design; it is the correct tool, because it is the lens on which
the pressure choice actually shows up.

The defensible per-region reading across everything measured so far:

- **LV free wall — P_LV.** Wins both lenses in every sweep and every activation
  bundle. Robust.
- **RV free wall — P_RV on magnitude, when LV loading is fixed.** Once LV loading
  varies, additive combinations do at least as well, and that is a finding about
  interdependence rather than a failure of the proxy.
- **Septum — P_LV.** Best on both lenses on the only sweep with real septal
  signal. Transmural pressure remains the worst candidate throughout.

## Reproducing

The scan, the fit and the noise study are pure post-processing over
`per_cell_data.npz` and the solver pressures of the two sweeps. Region
quantities use `analysis_core.region_density` with the geometric septum mask.
