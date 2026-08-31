# Supporting studies

Smaller campaigns whose job is to close off objections rather than produce a
headline. Grouped here because none needs a page of its own.

## Mesh convergence

`results/analysis/mesh_convergence/`, resolutions h = 10, 7.5, 5, 3.75 mm.

Establishes that the production resolution (h = 5) is converged enough for the
claims made. Across the 16 paired cases, h10 → h5 moves LV ESP by at most 8.98%
and RV ESP by at most 14.11%. Derived work *ratios* move considerably more — the
free-wall tensor ratio by up to 20.71%, the adjacent-`ll` ratio by up to 36.92%
— so the honest statement is that the direction of the free-wall result is
resolution-robust while its absolute magnitude is not.

Septal quantities are deliberately excluded from the h10→h5 comparison: at
h = 10 the geometric septum mask does not match the tag-3 septum volume, so the
two resolutions are not measuring the same set of cells and a difference between
them would be a mask artifact, not a convergence result. At h = 5 the geometric
and tag-3 septum volumes do agree, which is what licenses the canonical mask.

## AHA mid-ventricular ring

LDRB tags 4/5/6, wired through `compute_per_cell` (`aha_tag`), the backfill
path, and `make_pah_handover.py` as `band="mid"`.

Standard clinical work imaging reports on AHA segments, not on whole walls, so
this exists to make the simulation comparable to what a clinician actually
measures. Restricting to the mid-ventricular ring sharpens the septal `P_RV`
proxy substantially — r rises from 0.75 to 0.95 in the no-Frank-Starling bundle
— because the basal and apical thirds contribute geometry that the whole-wall
average smears together.

Transmural pressure remains the worst-performing septal candidate under this
banding too, consistent with the whole-wall result.

## Base Dirichlet and metric-space sensitivity

`results/analysis/base_dirichlet_sensitivity/`,
`results/analysis/metric_space_sensitivity/`.

Two numerical-robustness appendices. The first varies the basal boundary
condition, the second the finite-element space in which stress and strain are
stored and integrated.

The metric-space study is the source of a standing rule: **integrate work in
DG0, not DG1**. Projecting stress or strain onto DG1 produces spurious
oscillations at the thin septum, where the element size approaches the wall
thickness. Those oscillations do not average out of a work integral, so DG1
work densities in the septum are not trustworthy. The production path uses
Quadrature6 for metric storage and DG0 for work integration.

## Septum mask sensitivity

`analysis/h5_sweep_septum_sensitivity_epi_excluded/`,
`..._epi_inclusive/`, `..._compare/`.

Tests whether including or excluding the epicardial layer changes the septal
conclusions. This is the ancestor of the still-open
[septal mask question](../open-questions.md#septal-r-is-mask-sensitive-and-unreconciled):
the septum is the one region where a defensible change of mask definition moves
the correlation materially (0.75 vs 0.93), which is why septal numbers carry a
caveat that free-wall numbers do not.

## Circulation parameter tuning

`pah_pulmonary_batch/` (0D sweeps), with Optuna used for Bayesian parameter
search in earlier rounds.

Not a physics result, but the reason the hemodynamics are plausible. One
practical rule came out of it and is worth repeating: for any pre-coupling 0D
curve, read `preload_history.npy` from the run rather than re-running the
Regazzoni model. The library has drifted since the sweeps were generated and a
re-run inflates RV pressure — the real spectrum is 32–93 mmHg, whereas a fresh
re-run reports up to 122 mmHg.
