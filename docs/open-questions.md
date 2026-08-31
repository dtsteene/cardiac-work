# Open questions and known gaps

Everything here is genuinely unresolved as of 2026-08-31. Nothing in this page
is rhetorical: each item would change a number, a figure, or a claim.

## A stale copy of the superseded "transmural is best" claim

Not a scientific open question — a documentation hazard, recorded because the
stale claim is the kind that gets quoted by accident.

The project's early conclusion was that transmural pressure `P_LV − P_RV` is the
better septal proxy. **That result was an artifact** of per-case inverse
unloading, and it disappears once a single shared inverse-unloaded reference is
used. Every current analysis agrees transmural is the *worst* septal candidate;
see [septal proxy](findings/septal-proxy.md). The repo's own
[CLAUDE.md](../CLAUDE.md) states the corrected version and names the artifact.

A stale copy of the superseded claim still exists at **`/home/dtsteene/CLAUDE.md`**
(dated 2026-03-24), which is a user-level instructions file outside this repo:

> Our simulations show that **transmural pressure (P_LV - P_RV)** is a better
> proxy for septal work in PAH.

It should be updated or deleted so it cannot be mistaken for current guidance.
The thesis manuscript (`/home/dtsteene/D1/RV/`) has not been checked against
the corrected numbers either, and is worth a pass for the same reason.

## Septal r is mask-sensitive and unreconciled

Both August supervisor notes flag this and neither resolves it. The septal
correlation is 0.75 by the thesis convention and 0.93 with the canonical region
mask. The RV and LV numbers are stable across the same change (RV `P_RV`
r ≈ 0.99 vs pipeline 0.98), so this is specific to the septum, where the mask
definition genuinely changes which cells are counted.

Until this is reconciled against `sweep_analysis.py`, **no septal figure should
be quoted**. Note this interacts with the item above: a mask that changes r from
0.75 to 0.93 could plausibly change a proxy ranking too.

## RV × LV afterload grid

**Designed, never run.**
`docs/superpowers/specs/2026-07-08-rv-lv-afterload-grid-design.md`, status
"Design approved, pending implementation plan".

The existing pulmonary sweep raises RV afterload while holding LV loading fixed.
With `P_LV` nearly constant, the case-vector `(P_LV, P_RV)` traces a horizontal
line, and on that line `P_RV`, `Mean = ½(P_LV+P_RV)`, `Sum = P_LV+P_RV` and
transmural `P_LV−P_RV` are all affine in `P_RV` alone. Pearson correlation
cannot separate them; the code says so itself, in
`make_pah_handover.py:fig_region_correlation`. The grid varies both afterloads
to break that collinearity, which is what would make the correlation test
actually discriminating.

**Revised 2026-08-31.** Two of the design's premises did not survive contact
with the completed [shared-unloaded spectrum](experiments/spectrum-shared-unloaded.md):
the septal dynamic-range gate is already cleared (29.2%), and correlation will
not crown `P_RV` on the RV free wall in *any* 2-D design — widening the LV axis
makes it worse, because true RV work is nearly as sensitive to LV pressure as to
its own. See [what correlation can and cannot decide](findings/proxy-identifiability.md).

The grid is still the right destination, but it should be **staged**. Its
lowest-RV row is a systemic-afterload-only arm mirroring the pulmonary sweep;
those four cases measure the one coefficient the whole question turns on —
∂W_RV/∂P_LV at fixed contractility — and decide whether the remaining twelve are
worth running. Note that no single one-dimensional arm can separate the
candidates at all, so the arm is a diagnostic, not a substitute for the grid.

## Adding measurement noise and a small cohort

**Partly answered 2026-08-31:** noise does not rescue correlation — P_RV ranks
first on the RV free wall in 0% of 4000 draws at every noise level from 2% to
35%. Noise blurs a ranking rather than reversing one. Its real value is error
bars, exposing spurious winners on a flat truth, and realism; put it on strain
and pressure, not on the finished proxy work. See
[proxy identifiability](findings/proxy-identifiability.md).

Proposed in the August RV note. The clean monotone sweep can rank proxies by
direction but not robustly by magnitude. Adding controlled measurement noise
plus a small cohort varying contractility, wall thickness and fibre angle would
turn correlation into a real ranking test and let us say how far ahead `P_RV` is
of `Mean`/`P_LV`. Here noise *discriminates* a real signal rather than
fabricating one.

Distinct from this: a heterogeneous virtual population with imaging noise would
answer the separate clinical-utility question of whether work imaging tracks
true work across patients.

## Canonical raw simulation data no longer exists

See [provenance](provenance.md). The thesis-canonical sweep's raw checkpoints
were deleted; the derived analysis survives in full. Thesis numbers are
defensible as analysis but not re-derivable as simulation without re-running.

## Smaller loose ends

The `SWEEPS_INDEX.md` and `README_WHAT_MATTERS.md` under `results/` were
accurate when written (May 2026) but predate all June–July work and point at
directories that no longer exist. They are superseded by this tree.

`results/sims/2026-06-22/pah_pulmonary_fixedratio/no_frank_starling/case_NONEXISTENT/`
is a 3.5 KB artifact from a deliberate canary test of the submit script's error
handling. Harmless; safe to delete.

Four near-duplicate `pah_pulmonary_sweep_fixedratio*` directories exist under
`results/analysis/` (`_moved`, `_moved2`, `_canary`), each holding only a
`job_ids.txt` and a `pah_pulmonary_cases.tsv`. They should be collapsed to the
canonical one plus the canary.
