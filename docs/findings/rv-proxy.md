# The RV proxy — direction versus magnitude

Source: `results/handover/supervisor_2026-08/story2_rv_proxy/`, computed on the
fixed-ratio no-Frank-Starling sweep.

## The RV is where the mechanics actually move

Indexing each region's true internal work `∫∫ S : dE dV` to its own mildest
case, across the eight-case afterload sweep:

- **RV: ×2.7** — nearly triples
- **LV: ~15%, downward** — it underfills as the RV obstructs
- **Septum: flat**

This is the precondition for everything else on this page. A proxy can only be
tested where the underlying quantity varies, and the RV is the only region with
real dynamic range in this design. It is also, conveniently, the region PAH
clinicians care about.

## Correlation cannot tell the candidates apart

Because the sweep is clean and monotone, `P_RV`, `Mean` and even `P_LV` all sit
at r ≈ +1.0 (`P_LV` at 0.97). By correlation alone they look equally good. They
are not — and the reason correlation cannot see it is structural, not
statistical: with LV loading fixed, every non-transmural candidate is an affine
function of `P_RV` alone, and Pearson r is invariant under affine
transformation.

Transmural is the exception, and it fails: r = −0.77, anti-tracking the RV.

## Magnitude separates them

Indexing true RV work and each proxy to its own mildest-case value puts them on
one axis and makes the difference visible:

| | Indexed rise across the sweep |
|---|---|
| True RV work | ×2.7 |
| `P_RV` proxy | **×2.4** |
| `Mean` proxy | ×1.7 |
| `P_LV` proxy | ×1.4 |

`P_RV` is the only candidate that recovers the magnitude as well as the
direction. `Mean` captures under two-thirds of the true rise and `P_LV` barely
half — while both report r ≈ 1.0. This is the clearest demonstration in the
project that a near-perfect correlation can hide a badly wrong magnitude, and
it is the strongest current argument for `P_RV` as the RV proxy.

## The LV:RV work ratio is recoverable

True LV:RV internal-work ratio falls monotonically from **8.5 to 2.7** across
the sweep — in mild disease the LV does roughly eight times the RV's work; in
severe disease only about three times. A purely proxy-based estimate
(`P_LV·ε_LV` over `P_RV·ε_RV`) follows the same curve with a modest upward bias,
10 → 3.4.

This matters because it is a clinically meaningful quantity — "how hard is the
RV working relative to the LV" — that survives the translation from tensor
truth to pressures and strains. It is arguably the most directly usable result
the project has.

## Caveats

The RV and LV numbers are solid and thesis-consistent: RV `P_RV` r ≈ 0.99
against a pipeline value of 0.98. The **septal** number from the same analysis
is mask-sensitive and should not be quoted until
[reconciled](../open-questions.md#septal-r-is-mask-sensitive-and-unreconciled).

The magnitude ranking above rests on a single monotone sweep. It shows `P_RV`
ahead, but it cannot say by how much with confidence, because there is no noise
model and no cohort variation to test robustness against. Turning this into a
defensible ranking is what the
[noise-plus-cohort proposal](../open-questions.md#adding-measurement-noise-and-a-small-cohort)
is for.
