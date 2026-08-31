# Findings

What the simulations showed, region by region. Each page states the evidence and
its limits; where a claim is contested, it says so rather than picking a side.

| Page | In one line |
|---|---|
| [Septal proxy](septal-proxy.md) | No pressure choice tracks septal work well; the early "transmural is best" result was an unloading artifact |
| [RV proxy](rv-proxy.md) | `P_RV` is the only proxy that recovers RV work magnitude, not just direction |
| [ED overlap](ed-overlap.md) | The flat LV end-diastolic point is diastolic physics, not a modelling error |
| [Proxy identifiability](proxy-identifiability.md) | Correlation cannot crown `P_RV` on the RV free wall in *any* 2-D loading design; the magnitude lens is the correct tool, not a fallback |
| [Frank-Starling](frank-starling.md) | FS widens the RV dynamic range by half again and is the only bundle where `P_RV` wins the RV on both lenses |

## The one thing to carry away

The regions behave differently enough that a single "best proxy" answer does not
exist, and most confusion in this project has come from stating a region-specific
result as though it were general.

For the **free walls** the obvious pressure works well — LV r = 0.994,
RV r = 0.967 against true internal work. For the **RV specifically**, `P_RV`
additionally recovers magnitude, which no other candidate does. For the
**septum**, no candidate is good: correlations against longitudinal strain sit
around +0.54 for every non-transmural choice, and transmural anti-correlates.

So the honest summary is that pressure-strain proxies work where the pressure
unambiguously belongs to the wall, and degrade exactly where the clinical
question is hardest — the shared wall. Note that the project's early
"transmural is best" septal conclusion was an artifact of per-case inverse
unloading and has been superseded; transmural is in fact the worst septal
candidate.

## Two methodological cautions

**Correlation is a weak discriminator on a monotone sweep.** When every
hemodynamic quantity rises together, proxies that are affine functions of one
another are mathematically indistinguishable by Pearson r. Several will report
r ≈ 1 while disagreeing badly on magnitude. Prefer indexed-tracking and
ratio-preservation views. See
[the afterload grid](../open-questions.md#rv--lv-afterload-grid) for the
experiment designed to fix this.

**Proxies miss most of the work by construction.** The clinical assumption is
`work ≈ pressure × fibre strain`, but ground-truth internal work `∫∫ S : dE dV`
integrates all stress directions. Cross-fibre and sheet contributions account
for roughly half to two-thirds of total internal work, so even a perfectly
tracking proxy is measuring a fraction of the energy expended. Tracking and
completeness are different questions and should not be conflated.
