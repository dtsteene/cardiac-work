# Does Frank-Starling change the answer?

**Date:** 2026-08-31. Measured on the three activation bundles of the
[pulmonary afterload sweep](../experiments/pulmonary-afterload-sweep.md) —
same eight cases, same mesh, same reference, three activation models.

Frank-Starling has been treated as an optional simulation-time toggle
(`USE_FRANK_STARLING`), and the thesis model does without it. A collaborator
with a clinical background has argued it is needed. The question is whether it
changes any conclusion, and the answer is that it changes one, in our favour.

## What the three bundles give

| bundle | RV true-work range | RV fold change | best r, RV | best magnitude, RV |
|---|---|---|---|---|
| no FS, Ta = 100 kPa | 80% | ×2.54 | Mean | P_RV |
| FS frozen at ED, Ta = 220 | **115%** | **×3.67** | **P_RV** | **P_RV** |
| FS + activation lag, Ta = 220 | 88% | ×2.76 | Mean | P_RV |

**Frank-Starling amplifies the RV signal and sharpens the RV verdict.** With FS
frozen at end-diastolic stretch the RV free wall's true work range grows by half
again, and it becomes the only bundle in which P_RV wins the RV free wall on
*both* lenses — correlation as well as magnitude. Without FS, correlation
crowns Mean and only the magnitude lens recovers P_RV.

That is physiologically what one expects. As pulmonary afterload rises the RV
dilates; length-dependent activation recruits more force at the longer
end-diastolic length, so the work rises further than a prescribed constant
tension allows. Suppressing that is suppressing the mechanism by which a real
ventricle answers a rising load.

**It changes no other verdict.** P_LV wins the LV free wall on both lenses in
all three bundles. The septum stays flat in all three (2–4% range on this sweep)
and its nominal "winner" flips between bundles, which is noise on a flat truth
rather than a result.

## The confound to fix, and why matched Ta is not enough

The FS bundles run at Ta = 220 kPa against the no-FS bundle's 100 kPa, so the
comparison above conflates length-dependent activation with a doubling of peak
tension. But the doubling is not arbitrary, and understanding why changes what
the clean experiment should be.

The gain curve `g(λ)` in use rises from zero at λ = 0.85 to its peak at
λ = 1.15 (`FS_STRETCH_THRESHOLD`, `FS_STRETCH_OPTIMAL`). Measured on the
pulmonary sweep, the fibre stretch relative to the unloaded reference is:

| region | λ at end-diastole | λ at end-systole |
|---|---|---|
| LV | 1.063 | 0.890 |
| RV | 1.054 | 0.878 |
| septum | 1.049 | 0.867 |

**The heart never reaches the curve's optimum.** It operates entirely on the
ascending limb, between g ≈ 0.7 at its most stretched and g ≈ 0 at end-systole.
That is why Ta had to be doubled — FS delivers at best about 70% of the peak
tension, and `complete_cycle.py` says so in its own comment: the default curve
"throttles force to ~50-70% at the operating stretch and forces
unphysiologically high Ta".

Two consequences. The first is that Ta = 220 with this curve is roughly
equivalent in delivered force to Ta = 100 without it, so the bundles are not as
mismatched as the raw numbers suggest. The second is less comfortable: sitting
on the ascending limb is where `dg/dλ` is largest, so the FS effect is at its
most sensitive possible setting, and some of the extra dynamic range may be an
artifact of where the curve is centred rather than physiology.

**The clean experiment is therefore not FS on/off at matched Ta.** It is FS with
the curve re-centred so its optimum sits near the operating end-diastolic
stretch — `FS_STRETCH_OPTIMAL ≈ 1.05`, which the code comment already
recommends — run at literature Ta ≈ 100 kPa, against no-FS at the same Ta. Then
both arms deliver physiological force and the only difference is
length-dependence.

## Recommendation

Keep it, and make `frank_starling_preload` the default for new sweeps. The
argument is not merely that it is better physiology, though it is: it measurably
widens the dynamic range the whole study depends on, and it is the one setting
under which the headline RV claim survives both lenses. The activation-lag
variant buys nothing here and adds a parameter, so prefer the preload-frozen
form.

Two things should accompany that. Run the re-centred comparison above once, so
the effect is attributable to length-dependence rather than to peak tension or
to curve placement. And note that FS makes the preload-coupling fix
load-bearing rather than merely correct — FS gain is read at end-diastolic
stretch, so a clamped preload yields no across-case FS signal at all. The
2026-06-22 fixed-ratio anchoring is a precondition for using FS, not an
independent choice.
