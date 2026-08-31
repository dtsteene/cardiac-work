# Pulmonary afterload sweep

**Status:** canonical (current). `sims/2026-06-22/pah_pulmonary_fixedratio/`
**Tooling:** `pah_pulmonary_batch/`

## What it is

Eight cases sweeping **only** the pulmonary arterial resistance and compliance
(`R_AR`, `C_AR`) of the 0D circulation, labelled by target RV systolic pressure:
`case0_rv25` through `case7_rv95` in 10 mmHg steps. Everything else — geometry,
material, activation, systemic circulation — is held fixed.

That deliberate narrowness is the point. The thesis sweep varied a whole
severity spectrum at once; this one isolates afterload so that any change in
work is attributable to it alone. Baseline is a linear-EDPVR refit
(sPAP22 + EB).

Three activation bundles run the same eight cases:

| Bundle | Activation |
|---|---|
| `no_frank_starling` | prescribed constant Ta, no length dependence |
| `frank_starling_preload` | Frank-Starling frozen at ED stretch |
| `frank_starling_relax` | Frank-Starling with activation-lag relaxation |

Frank-Starling is a **simulation-time** choice (`USE_FRANK_STARLING`), not an
analysis one. The analysis layer is deliberately agnostic to it.

## Why it took four generations

The first three attempts were superseded for a reason worth recording, because
it is the kind of error that does not announce itself.

The FEM mesh and the 0D circulation model do not share a volume scale, so they
are coupled through per-case ratios (`ratio_LV`, `ratio_RV` = mesh ED volume /
circulation ED volume). Letting those ratios be derived per case allowed the 0D
warm-up to inflate the preload delivered to the mesh by roughly **41%** relative
to the intended operating point — the coupling was absorbing a discrepancy
instead of exposing it.

Clamping the ratio to a fixed per-case anchor (case0: LV 1.02479, RV 0.88262,
via `FIXED_RATIO_LV` / `FIXED_RATIO_RV`) brought that to about **9%**. That is
the `fixedratio` sweep, and it is the one to use.

Generations, oldest first: `2026-06-07/pah_pulmonary_20260607_224702` →
`2026-06-08` (two) → `2026-06-09/pah_pulmonary_20260609_prodsweep` →
`2026-06-22/pah_pulmonary_fixedratio` (canonical, jobs 1310381–1310404).

The earlier `frankstarling_l5_*` sweeps of 2026-05-25 through 05-29 are the
ancestors of the FS bundles and are likewise superseded.

## What it showed

The scientific content is in [findings](../findings/README.md); in short, this
sweep is where the RV proves to be the region with real dynamic range (true
work nearly triples, ×2.7, while the LV changes ~15% downward and the septum is
flat), and where `P_RV` shows itself the only proxy that recovers RV work
*magnitude* as well as direction.

## The correlation caveat, stated plainly

Because LV loading is fixed, `P_LV` is nearly constant across the eight cases.
The case-vector `(P_LV, P_RV)` therefore traces a horizontal line, and on that
line `P_RV`, `Mean`, `Sum` and transmural `P_LV − P_RV` are all affine functions
of `P_RV` alone — so Pearson correlation **cannot distinguish them**. The code
records this itself in `make_pah_handover.py:fig_region_correlation`.

This is a property of the experimental design, not a bug, and it is exactly what
the [RV × LV afterload grid](../open-questions.md#rv--lv-afterload-grid) was
designed to fix. Until that grid runs, rank proxies by magnitude-preserving
measures, not by correlation.

## Note on ventricular interdependence

A recurring misreading is that the coupling is "series-only" and therefore has
no shared septum. It is not. The coupled solve drives a genuine biventricular
mesh with a shared septal wall and returns both cavity pressures
(`p_BiV_func`, `complete_cycle.py`), so interdependence — the D-sign — is fully
in the loop. Only the 0D *warm-up* treats the chambers independently, and that
only seeds the initial state.

The reason LV end-diastolic volume barely shifts across the sweep is not a
missing septum but that end-diastole is a low-pressure state; see
[ED overlap](../findings/ed-overlap.md).
