# RV × LV Afterload Grid — De-collinearizing the Septal Proxy Test

**Date:** 2026-07-08
**Status:** Design approved, pending implementation plan
**Bundle:** `frank_starling_preload` (Frank–Starling frozen at ED stretch, Ta = 220 kPa)

## Motivation

The current pulmonary-windkessel study is a **one-dimensional monotone loading sweep**:
RV afterload rises across 8 cases while LV loading is held fixed. Because every
hemodynamic quantity rises monotonically together, the per-case correlation between
true septal work and the pressure-strain proxies is **degenerate** — the code itself
records this in `make_pah_handover.py:fig_region_correlation`:

> "across this monotonic sweep every non-transmural pressure is collinear
> (r within ~0.02), so correlation cannot distinguish them — only transmural differs."

The precise cause: with LV loading fixed, P_LV is ~constant across cases, so the
case-vector (P_LV, P_RV) traces a horizontal line. On that line P_RV, Mean =
½(P_LV+P_RV), Sum = P_LV+P_RV, and transmural P_LV−P_RV are all **affine functions of
P_RV alone**, hence mathematically indistinguishable by Pearson correlation. The
correlation is computed across cases (each case contributes one scalar QoI per region;
Pearson over the case points — see `sweep_analysis.py:correlation_rows` and
`make_pah_handover.py:fig_region_correlation`).

The fix is to make the loading **two-dimensional** so the (P_LV, P_RV) case cloud spans
a plane, restoring the discriminating power of correlation and enabling honest
magnitude/agreement analysis.

### Two degeneracies, not one (2026-07-08 finding)

A re-analysis of the fixed-ratio sweep showed the degeneracy has **two distinct forms**,
and collinearity-breaking only addresses the first:

- **RV — saturated/monotone degeneracy.** True work spans a large range (70–115% of
  mean) but perfectly monotone in the single driver, so every pressure fits equally well
  (P_LV r≈+0.998 vs P_RV r≈+0.990). This is the collinearity the 2-D grid targets.
- **Septum — flat-truth degeneracy.** True septal work barely moves across the entire
  sweep (**2–4% of mean**, all three bundles), so *every* pressure "correlates" ~0.99
  trivially. Breaking collinearity does **nothing** here — there is no signal to
  discriminate against, and swapping correlation for agreement metrics does not rescue it
  (verified: neither can pick a septal winner on a single-parameter sweep).

**Consequence for this design.** The 2-D grid must defeat *both*: break the RV
collinearity **and inject genuine dynamic range into septal (and LV) true work**. The
latter is the new make-or-break requirement. Physically it should work — septal work is
flat in the 1-D sweep precisely because it is LV-pressure-dominated and LV pressure was
held constant; directly sweeping LV/systemic afterload is the lever that should wake the
septum up. But this is now the **first thing the pilot must verify**, before any proxy
analysis is trusted.

## Objective & Hypotheses

Cross RV pulmonary afterload with an independent **systemic (LV) afterload** axis so the
proxies de-collinearize.

- **H1 — RV free wall (confirmatory).** P_RV remains the best single-ventricle proxy and
  transmural fails; expected to survive the richer design (currently r≈0.98).
- **H2 — Septum (the real test).** Across a 2-D loading cloud, which pressure best
  proxies true septal work — P_RV alone, Sum, Mean, or an affine blend of P_LV and P_RV?
  The 2-D design lets us answer this on **both** shape (correlation) and magnitude
  (ratio/agreement) axes rather than guessing. No method is crowned a priori.
- **H3 — Methods.** Demonstrate that Pearson r is degenerate under 1-D monotone loading
  and becomes discriminating only in 2-D — the structural reason the 2-metric
  clinical/canine-occlusion literature cannot observe this.

## Experimental Design

### Grid
Full **4 (RV) × 4 (LV) factorial = 16 cases**, one bundle (`frank_starling_preload`).

**RV / pulmonary axis** — reuse 4 pulmonary-afterload levels from the existing sweep
recipe: RV systolic ≈ **25, 45, 65, 90 mmHg**, produced as now by raising `PUL.R_AR`
and lowering `PUL.C_AR` at conserved pulmonary RC ≈ 0.33 s.

**LV / systemic axis** — target LV systolic ≈ **100, 120, 140, 160 mmHg** by raising
`SYS.R_AR` and lowering `SYS.C_AR` at **conserved systemic RC** (methodologically
symmetric with the pulmonary axis; keeps the change mostly systolic and bounds mean
arterial pressure drift). C-held-constant is the fallback if conserved-RC over-perturbs
diastole.

All 16 cases load the **same shared inverse-unloaded reference** and use the
**fixed single ED-ratio** (the 2026-06-22 fix) so ED strain varies correctly across
cases rather than being clamped per-case.

### Parameterization (no Optuna)

The hand-tuned `baseline_linear.json` (the baseline the collaborators preferred) is
**left untouched except for the four windkessel knobs**. No optimizer is used — the
Optuna pipeline is intentionally excluded from the repo. Pressures are set by the same
deterministic 0-D interpolation-inversion already used in `make_sweep_params.py`:

- **Per-axis loci.** Define a geometric locus in (`PUL.R_AR`, `PUL.C_AR`) (R↑/C↓ at
  conserved pulmonary RC) and an analogous locus in (`SYS.R_AR`, `SYS.C_AR`) (R↑/C↓ at
  conserved systemic RC). For each axis, densely sample the locus in 0-D, record the
  systolic pressure, and invert `P_sys(s)` with `np.interp` to find the R/C that hit the
  target systolic values.
- **Factorial product.** Take the 4×4 product of the two inverted axes → 16 circulation
  JSONs, each = baseline with only `PUL.R_AR/C_AR` and `SYS.R_AR/C_AR` changed.
- **Accept the drift.** The two axes are not perfectly independent in the coupled 0-D
  (systemic afterload nudges RV loading via venous return and vice versa), so the product
  will not land *exactly* on the (LV_sys, RV_sys) target grid. This is accepted: the
  analysis only needs the (P_LV, P_RV) case cloud to **span 2-D**, not to hit exact grid
  points. Each case records its **achieved** (LV_sys, RV_sys); targets are nominal.

### Preload coupling (critical — do not normalize the preload away)

The FEM must feel the **true across-case preload spread**; this is the physiology the
experiment depends on. `complete_cycle.py` couples FEM to 0-D with a multiplicative
volume ratio. Its **default** sets that ratio **per case** as
`Mesh_ED / (this case's 0-D ED volume)`, which re-normalizes every case back to the mesh
ED and **clamps the FEM preload** — destroying the across-case dilation and, with it, the
Frank–Starling signal (the 2026-06-22 diagnosis: FEM EDV spread collapsed to +9% while
0-D was +41%).

Requirements for the grid:

- **Single fixed ratio for all 16 cases.** Pass the same `FIXED_RATIO_LV` /
  `FIXED_RATIO_RV`, anchored to the one physiological **baseline node** (imaged mesh =
  lowest-RV / nominal-LV corner), computed once from that node's 0-D ED volumes vs the
  mesh ED. Never recompute per case.
- **Launcher must set them explicitly.** The env default is empty, and empty silently
  falls back to the per-case normalization bug — so the launcher sets both for every job.
- **Doubly critical for `frank_starling_preload`.** FS gain is frozen at ED stretch; a
  clamped preload yields zero across-case FS signal.
- **Verify, don't assume.** Collect each case's FEM ED volume post-run and confirm LV and
  RV EDV genuinely spread across the grid (a preload-clamp check) before trusting any
  proxy analysis.

### Bundle
`frank_starling_preload`: `USE_FRANK_STARLING=1`, `TA_PEAK_KPA=220.0`,
`FS_PRELOAD_ONLY=1` (Frank–Starling gain frozen at ED stretch). This is the intended
active model for the experiment; the no-FS and FS-relax bundles are deferred.

### Two-stage execution
1. **Pilot** — L10 mesh (`CHAR_LENGTH=10.0`,
   `pah_pulmonary_batch/shared_unloaded_L10/ref/solver/prestress_inverse.bp`),
   **`BEATS=1`**, all 16 cases. Validates the systemic-afterload knob, the circulation
   convergence at each grid node, the pipeline end-to-end, and gives a first look at
   whether the proxies de-collinearize. Cheap (16 short runs).

   **Pilot gate (do this first, before any proxy analysis).** Compute the dynamic range
   of **true septal work** (and LV work) across the 16 cases. If septal true-work range
   is not materially larger than the ~2–4% of the 1-D sweep (target ≳ 15%), the LV axis
   is not injecting septal signal — **stop and rethink the axis** (widen the LV span,
   consider a contractility component, or reconsider the septal sub-region/metric) before
   spending compute on production. A flat septum in 2-D means the experiment still cannot
   answer the septal question, and no downstream metric fixes that.
2. **Production** — L5 mesh (`CHAR_LENGTH=5.0`, existing L5 unloaded ref),
   `BEATS=6`, same 16 cases. Optionally replicated on the other FS bundles once the
   L5 grid is validated.

## Analysis Plan

Per region (LV / RV / Septum), across the 16 case-points. **No single method is
privileged**; the candidate pressure family — P_LV, P_RV, Mean = ½(P_LV+P_RV),
Sum = P_LV+P_RV, Trans = P_LV−P_RV, Affine(λ) — is evaluated even-handedly on two
complementary axes.

1. **Shape — discriminating correlation.** True SS work vs each proxy across the 16
   cases. In 2-D the |r| values separate (contrast the ~0.02 spread of the 1-D sweep).
   Report the r-spread per region as the headline evidence that the design is
   non-degenerate.

2. **Magnitude — ratio / agreement.** Reuse the **already-built, unit-tested agreement
   module** in `analysis_core.py` (`concordance_ccc`, `proportional_fit`,
   `agreement_stats`) — per region, per pressure choice: through-origin k, Lin's CCC,
   %RMSE, bias, and a Bland–Altman of the calibrated proxy. This axis is scale-sensitive,
   so it is the **only** lens that separates **Sum from Mean** (identical under
   correlation because Sum = 2·Mean → same |r|, but distinct in magnitude). Sum has been
   performing well here; make that an explicit finding.

   **Do NOT pool k across walls.** A single global k fitted over LV+RV+Septum together is
   conceptually invalid — the free walls have *different* correct pressures (P_LV for LV,
   P_RV for RV), so no single k exists (empirically k_spread≈4.5, an artifact not a
   result). All agreement stats are computed **within a region**. The septal verdict
   comes from the septum's own agreement + correlation, not a cross-wall pool.

3. **Supporting — free affine blend.** Regress true septal work on the two
   single-ventricle proxies to recover a data-driven λ\* (with CI) and check whether any
   free blend beats the fixed candidates. One lens among several, **not** the headline.

4. **Degeneracy diagnostics (H3).** Condition number / VIF of the (P_LV, P_RV) design
   matrix and the (P_LV, P_RV) case-cloud scatter, shown against the near-singular 1-D
   sweep, to demonstrate the design actually broke the collinearity.

5. **RV confirmatory (H1).** Verify P_RV still wins and transmural fails on the RV free
   wall.

## Scope

**In:** loading-only 2-D grid (RV pulmonary × LV systemic afterload); one bundle
(`frank_starling_preload`); pilot (L10, 1-beat) → production (L5, 6-beat); the analysis
above reusing the existing per-cell / correlation / ratio machinery.

**Out (deferred):**
- Biological-scatter axis (per-case material/fiber/geometry perturbation) for the
  "is biology messy enough" K9 question — future overlay, not v1.
- No new material model, no mesh/fiber perturbation, no geometry change.
- Other FS bundles (no-FS, FS-relax) — only after the L5 grid validates.

## Reuse / Touch Points

- **New circulation JSONs** — extend `pah_pulmonary_batch/make_sweep_params.py` (or a
  sibling) to emit a 4×4 grid of `SYS.R_AR/C_AR` × `PUL.R_AR/C_AR` params, verified in
  0-D (`sweep_pulmonary_0d.py`) before any FEM.
- **Launcher** — generalize `submit_pah_pulmonary_sweep.sh` to iterate the 16 grid JSONs
  and accept `CHAR_LENGTH` / `BEATS` / `LOAD_UNLOADED_FROM` overrides (L10/1-beat pilot
  vs L5/6-beat production), with the `frank_starling_preload` env block.
- **Fields & metrics** — `compute_per_cell.py` / `postprocess_metrics.py` unchanged
  (per-cell proxies already emitted).
- **Agreement metrics** — reuse the existing `analysis_core.py` module
  (`concordance_ccc`, `proportional_fit`, `agreement_stats`; unit-tested) and the
  `pah_pulmonary_batch/agreement_analysis.py` script, generalized from the 1-D sweep to
  the 2-D grid (per-region only; drop the invalid cross-wall pooled-k table).
- **Figures** — extend `make_pah_handover.py` (and/or `sweep_analysis.py`) so the
  correlation/ratio figures handle a 2-D grid: severity coloring becomes a 2-D
  (RV, LV) encoding, and add the degeneracy-diagnostic and ratio-agreement panels.
- **Stale docs (separate handover fix, flag not block):** CLAUDE.md's cited septal
  "P_RV r≈0.75 / Trans r≈0.43" is from the older per-case-ratio sweep and does not
  reproduce on fixed-ratio data — correct it before handover.

## Success Criteria

- 16-case 0-D grid converges at every node; achieved (LV_sys, RV_sys) span the intended
  box (exact grid points not required — drift accepted).
- Pilot (L10, 1-beat) completes all 16 cases and produces per-cell proxies.
- **Preload not clamped:** all 16 cases share the single baseline-anchored
  `FIXED_RATIO_LV/RV`, and post-run FEM LV/RV EDV genuinely spread across the grid
  (not re-normalized to mesh ED).
- **Septal signal exists (the pilot gate):** true septal work range across the 16 cases
  is materially larger than the 1-D sweep's ~2–4% (target ≳ 15%). Without this, the
  experiment cannot answer the septal question regardless of metric.
- Per-region (not pooled) agreement + correlation together yield a defensible per-region
  verdict on H1/H2.
- The (P_LV, P_RV) case cloud spans 2-D (condition number materially below the 1-D
  sweep's), and the per-region non-transmural |r| spread is materially larger than the
  ~0.02 of the 1-D sweep — i.e. the proxies are now distinguishable.
- A clear per-region verdict on H1/H2 emerges from the combined shape + magnitude
  analysis.
