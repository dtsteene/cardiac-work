# Agreement-Metric Re-analysis of the Pressure-Strain Proxy Sweep

**Date:** 2026-07-08
**Status:** Implemented — pivoted to an identifiability finding (see Findings below)
**Author:** Daniel (with Claude)

## Findings (2026-07-08, after running the metrics)

Running the agreement metrics on the fixedratio sweep **inverted the expected
deliverable**, honestly and usefully. Key numbers (all three bundles, both
bandings):

- **True-work dynamic range: RV 70–115 %, LV 10–15 %, Septum only 2–4 %.** The
  septum is essentially flat.
- **RV free wall** is large but perfectly monotone, so the co-monotone pressures
  (PLV, PRV, Mean, Sum) track it at |r| ≈ 0.99–1.00 *and* fit it to a few-% affine
  RMSE — correlation and agreement rate them identically. No discrimination.
- **Septum** is flat, so every swinging proxy "correlates" with the tiny residual;
  the septal ranking is unstable across bundles (Trans |r| on septum = 0.03 in
  no_FS, 0.16 in preload, 0.86 in relax) — i.e. **noise**, not signal.
- The **all-region pooled single-k test is conceptually invalid** here (LV and RV
  free walls have *different* correct pressures, so no region-invariant k exists;
  `k_spread ≈ 4.5` for every choice). Its apparent "PLV wins" was an artifact of
  PLV fitting the large LV free wall. **Removed** from the deliverable.
- **Stale docs:** CLAUDE.md's cited septal "PRV r≈0.75 / Trans r≈0.43" does **not**
  reproduce on the fixedratio data (septal truth is flat there). Those are from the
  older per-case-ratio sweep. Flag for the handover-hygiene track.

**Defensible conclusion:** this single-parameter sweep *rules out* transmural
pressure (clearly wrong on the RV, where there is real range) but **cannot
adjudicate P_RV vs Mean vs P_LV for the septum** — the limit is the experiment's
one monotone degree of freedom and flat septum, not the metric. This is the
rigorous motivation for a multi-parameter (e.g. LV×RV afterload) redesign, in which
the same agreement metrics *will* discriminate.

**Delivered:** the four metric functions in `analysis_core.py` (kept — correct,
unit-tested 17/17, and the right tool for the redesign) and a reworked
`pah_pulmonary_batch/agreement_analysis.py` that produces, per bundle, the
identifiability table (`identifiability_<bundle>.csv`) and figure
(`identifiability_<bundle>.png/pdf`) under
`results/handover/pah_pulmonary_fixedratio_20260622/<bundle>/agreement/`.

The original design (below) is retained for provenance.

---

## Motivation

The PAH pulmonary-windkessel sweep varies a **single** driver (RV afterload,
25 → 95 mmHg). Every quantity in the RV is therefore a monotone function of that
one driver, so the sweep lives on a 1-D manifold and *all* pairwise Pearson /
Spearman correlations are ≈ ±1 **by construction**. This is a property of the
experimental design, not of the physics or the proxy. Consequences:

- A high `|r|` between a proxy and true work does **not** mean the proxy is a good
  quantitative tracker — it only means both are monotone in afterload.
- Correlation is invariant to scale and offset, but scale/offset is *exactly* what
  distinguishes the candidate pressures (`P_LV`, `P_RV`, `Trans`, `Mean`): they
  rank-order the 8 cases identically and differ only in magnitude and slope.
  Correlation therefore **cannot** pick a winner, and that is guaranteed a priori.

We already have the honest result stated as *ratio preservation* (log-MAE). This
work restates it as a **clinical-standard agreement analysis** that is immune to the
monotonicity critique, using data already on disk (no new simulations).

## Scope

- **Data:** existing fixed-ratio sweep,
  `results/sims/2026-06-22/pah_pulmonary_fixedratio/<bundle>/<case>/per_cell_data.npz`,
  3 bundles × 8 cases. Login-node only, pure NumPy/SciPy.
- **Pressure choices available in the npz** (as integrated `proxy_{*}_ll` fields):
  `PLV, PRV, Trans, Mean, Sum`. Note `Sum ≡ 2·Mean`, so it collapses onto `Mean`
  once a global scale `k` is fitted — report it but flag the equivalence.
  `TauWeighted`/`NearestSide` are **not** recoverable post-hoc (they require
  per-timestep re-integration), so they are out of scope for this re-analysis.
- **Regions:** LV free wall (tag 1), RV free wall (tag 2), geometric Septum
  (`is_geometric_septum`), matching `analysis_core.region_masks`.
- **Strain:** `ll` (longitudinal / GLS) — the primary clinical strain. `ff` as a
  secondary check.

## Key idea: why agreement discriminates when correlation cannot

Cavity pressure `P` is **not** wall stress `σ` (Laplace: σ is several × larger), so
no proxy will ever equal true work `∮ S:dE` in absolute magnitude. The discriminating
question is therefore not "does the proxy equal truth" but:

> **Does a single proportionality constant `k` map `proxy → true work`
> consistently across regions and cases?**

If `P_RV` is the right integrand for the RV and septum, *one* global `k` should map
its proxy onto true work everywhere. A wrong pressure (`P_LV`, `Trans`) needs a
*different* `k` per region, or maps with more scatter. This is the ratio-preservation
result recast as a single-line agreement test that penalizes exactly the scale/slope
differences Pearson discards.

**Calibration convention (decided): single global `k`, pooled per bundle.** Fit one
`k` across all regions × cases; the winning pressure is the one whose single line has
the lowest %RMSE and the most region-invariant per-region `k`.

## Metrics to add to `analysis_core.py`

All pure-NumPy, unit-tested in `tests/test_analysis_core.py` alongside the existing
`correlation_stats` / `ratio_preservation`.

1. `concordance_ccc(x, y) -> float` — Lin's concordance correlation coefficient:
   `2·cov / (var_x + var_y + (mean_x − mean_y)²)`. NaN-guarded like `pearson_r`.
2. `proportional_fit(proxy, truth) -> {k, resid_rmse, rel_rmse}` — least-squares
   through-origin slope `k = Σ(proxy·truth)/Σ(proxy²)`, plus absolute and relative
   RMSE of `truth − k·proxy`.
3. `agreement_stats(proxy, truth) -> {n, slope, intercept, ccc_raw, rel_rmse_affine, bias}`
   — the per-region summary. `slope`/`intercept` reuse `correlation_stats` (this is the
   *per-region* calibration); `ccc_raw` is Lin's CCC on the **uncalibrated** proxy vs
   truth (scale-sensitive — low for all choices because `P ≠ σ`, reported only as a
   pre-calibration diagnostic, **not** the discriminator); `rel_rmse_affine` is %RMSE
   around the per-region affine fit. The discriminator is the *cross-region* stability
   of `slope`, evaluated by the pooled test below — **not** any per-region CCC (which,
   if calibrated per region, would collapse back to correlation and re-introduce the
   monotonicity degeneracy).
4. `pooled_proportional(proxy_by_region, truth_by_region) -> {k_global, rel_rmse,
   ccc_pooled, k_by_region, k_spread}` — the headline cross-region test: one global
   `k` over the pooled points, plus per-region `k` and their spread (max/min).

No changes to simulation, `compute_per_cell.py`, or `postprocess_metrics.py`.

## Deliverables (one diagnostic script + figures)

New script `pah_pulmonary_batch/agreement_analysis.py` (mirrors `diagnose_fixedratio.py`
structure and reuses its `aggregate()` loader), producing per bundle:

1. **Per-region agreement table** (stdout + CSV): for each pressure choice, in each
   region — slope, intercept, CCC, %RMSE, bias. Expected to confirm PRV/Mean win on
   RV & septum, Trans worst, but as *agreement* not correlation.
2. **Pooled proportionality table** (the headline): global `k`, pooled CCC, pooled
   %RMSE, and per-region `k` spread, per pressure choice. Winner = smallest %RMSE +
   most region-invariant `k`.
3. **Figures** (`results/handover/.../agreement/`):
   - proxy-vs-truth scatter with the single global `k` line, points colored by region,
     one panel per pressure choice — visually shows PRV collapsing all regions onto one
     line vs PLV/Trans splaying by region.
   - Bland–Altman of the calibrated proxy (`k·proxy`) vs truth (bias + 95% limits),
     per pressure choice.

Output CSVs land next to the existing sweep tables so they are handover-discoverable.

## Success criteria

- New metric functions are unit-tested against hand-computed values (CCC identity,
  through-origin `k`, RMSE) and pass on the login node without FEniCSx.
- The pooled proportionality table ranks the pressures and the ranking is consistent
  with the existing ratio-preservation (log-MAE) result — demonstrating the two agree
  and that the conclusion does not depend on the degenerate correlation.
- One figure per bundle that makes the point to a clinician in a glance: the right
  pressure maps all regions onto a single line; the wrong ones do not.

## Out of scope (explicitly)

- New simulations, the 2-D LV×RV afterload experiment (separate design).
- `TauWeighted`/`NearestSide` agreement (needs re-integration in `compute_per_cell`).
- Any change to the environment/handover-packaging track.
