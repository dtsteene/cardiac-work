# Capped shared-unloaded L5 sweep — the thesis sweep

**Status:** canonical for the thesis. Raw simulation data deleted; derived
analysis intact. See [provenance](../provenance.md).

**Analysis:** `results/analysis/capped_shared_l5_sweep_20260510_141015/`
**Raw:** `sims/2026-05-10/capped_shared_l5_20260510_141015/` — *gone*

## What it was

Sixteen cases spanning a pulmonary-hypertension severity spectrum, labelled by
systolic pulmonary artery pressure from sPAP22 to sPAP95. Mesh resolution
h = 5 mm. Each case used a per-case unloaded reference derived from a shared
unloading step, with the RV end-diastolic pressure capped at 5.0 mmHg.

The cap is the reason this sweep supersedes its predecessor. Without it, the
unloading step drove the RV toward unphysiological end-diastolic pressures and
distorted the unloaded reference geometry — which changes the reference state
that every strain is measured against. Capping RV-EDP fixed the reference, not
the systolic ceiling: peak systolic pressure actually *rose* between the pre-cap
and capped versions of the same case (sPAP70: 75.21 → 85.83 mmHg), because a
different unloaded geometry inflates differently.

Achieved pressure ranges across the 16 completed cases: LV ESP 105.3–120.2 mmHg,
RV ESP 32.2–100.4 mmHg.

## What it showed

**Free walls — the proxy works.** For the LV and RV free walls the obvious
pressure choice tracks true work closely: LV r = 0.994, RV r = 0.967. This is
the reassuring half of the result, and it matters because it establishes that
the pipeline can recover a proxy relationship when one exists.

**Septum — no candidate is good, and transmural is the worst.** Against fibre
strain, `TauWeighted` (r = +0.979) and `Mean` (r = +0.976) track best, `PLV`
and `PRV` are mid-pack (+0.84, +0.80), and transmural *anti*-correlates
(r = −0.276). Against longitudinal strain every candidate collapses to
r ≈ +0.53–0.57, with transmural again negative (−0.331). On ratio preservation
— which asks whether the proxy gets the *magnitude* right, not just the
direction — `PLV` is best (mean abs log error 0.805) and transmural is worst
(2.075).

This is consistent with the corrected project conclusion: the early
"transmural is best" result was an artifact of per-case inverse unloading and
does not survive a shared reference. See
[septal proxy](../findings/septal-proxy.md). Note separately that the septal r
is [mask-sensitive](../open-questions.md#septal-r-is-mask-sensitive-and-unreconciled)
(0.75 vs 0.93), so septal numbers carry a caveat that free-wall numbers do not.

**RV free-wall clinical bridge.** RV pressure × longitudinal strain correlates
with true RV work at r = +0.967, and RV systolic pressure alone at r = +0.909.
Peak absolute longitudinal strain on its own is uninformative (r = −0.092) —
strain without pressure carries almost no work information, which is the
central argument for pressure-strain over strain-only imaging.

## Numerical robustness

The h = 10 → h = 5 refinement moves LV ESP by at most 8.98% and RV ESP by at
most 14.11% across the 16 paired cases. Free-wall work ratios move more
(tensor ratio up to 20.71%, adjacent-`ll` ratio up to 36.92%), which is worth
stating plainly: the *direction* of the free-wall result is robust to
resolution, the absolute ratio is not.

Septal h10→h5 differences are deliberately **not** summarised, because the 10 mm
geometric septum mask does not match the tag-3 / canonical septum volume — the
masks are not comparable, so a difference between them would not mean what it
appears to mean. At h = 5 the geometric septum volume does match tag-3, which is
why the canonical mask is defined there.

## Provenance caution

An earlier inconsistency was resolved on 2026-05-14 and is worth knowing about,
because stale copies of the affected figures may still circulate: chapter-5
figures had been sourced from the pre-cap CSVs while the prose quoted capped
numbers. The canonical analysis directory was fixed to
`capped_shared_l5_sweep_20260510_141015/`, a byte-identical duplicate directory
was archived, pre-cap CSVs were renamed with a `.precap_thesis_2026-05-04.csv`
suffix, and all chapter-5 figures were regenerated.
