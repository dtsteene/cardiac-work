# Transventricular Work Profiles: Boundary-Free Proxy Validation

## Motivation

We discovered that the septum proxy comparison (which pressure P best tracks septal work?) is **highly sensitive to the septum boundary definition**. Changing from the LDRB scalar-field heuristic to a geometric distance-based septum tag:

- Shrinks the septum volume by ~40%
- Changes the P_RV proxy correlation from r = -0.91 (inverse!) to r = +0.97 (strong)
- Flips the conclusion about which pressure is "best"

Meanwhile, boundary-independent quantities are stable:
- Whole-heart work: 0.0% change
- Free wall proxy ratios (P_LV x eps_ff / W_true): stable at ~0.25 +/- 0.03

This means the 3-way LV/RV/Septum split is scientifically fragile for proxy validation. We need an approach that avoids arbitrary boundaries.

## Key Observation

The septum is not a discrete region -- it's a **transition zone** where LV and RV loading overlap. Any hard boundary is arbitrary. Instead of asking "what is the septal work?", we should ask:

> At each point in the myocardium, how well does each pressure proxy predict the local work density?

This is a continuous, spatial question with no boundary ambiguity.

## Literature Context: The Septum Definition Problem

This boundary sensitivity is not unique to our work. The computational cardiology literature documents multiple approaches to septum definition, each with known limitations:

- **Euclidean distance heuristics** (`max(d_LV, d_RV) < d_epi`): computationally cheap, but overtagging occurs at high-curvature regions (base, apex) where d_epi becomes disproportionately small.
- **LDRB thresholding** (nested if/else on Laplacian scalar fields): the standard approach, but Laplace solutions are diffusive -- equipotential lines "bow" outward at the junctions to satisfy Neumann BCs on the epicardium, causing systematic overtagging.
- **Universal Ventricular Coordinates (UVC)** (Bayer et al. 2018): pins the septum boundaries via Dirichlet BCs at the anterior/posterior junctions. Defines the septum as an angular sector in a rotational coordinate. More principled, but still produces a discrete boundary. Limitations: asymmetric LV/RV coordinate definitions, discontinuities at the transventricular boundary, apicobasal coordinate non-linearity due to Laplace normalization.
- **Cobiveco** (Schuler et al. 2021): state-of-the-art biventricular coordinate system. Defines four coordinates: transventricular v (binary, LV=0/RV=1), transmural m (endo-to-epi depth), rotational r (circumferential position, septum = r in [2/3, 1]), and apicobasal a. Key innovations over UVC: symmetric coordinate directions in both ventricles, normalized distances along bijective trajectories (solving the trajectory distance equation, not raw Laplace values), implicit domain remeshing at the septal surface, and ridge surface extraction for principled septum/free-wall boundary identification. Achieves >4x reduction in transfer and linearity errors vs UVC across 36 patient geometries. **Important**: Cobiveco explicitly evaluated and rejected the simple Eikonal quotient g1/(g1+g2) of distances to two boundaries (their Fig. 2), showing it produces cusps and non-uniform spacing in regions of non-uniform thickness. The trajectory distance equation (their Eqs. 8-9) resolves this. See "Relationship to our tau coordinate" section below.
- **AHA 17-segment model**: the clinical standard. Septum = segments 2, 3, 8, 9, 14 (specific angular sectors in the rotational/circumferential direction). Note: AHA segments span the full wall thickness -- they define circumferential regions, not transventricular positions. Mapping AHA to our tau framework requires a rotational coordinate, which we do not currently compute (see Open Questions).
- **TriSeg model** (lumped-parameter): defines septal volume as 1/3 of total LV wall volume. Provides a volumetric benchmark.

All of these produce discrete boundaries, and the field has documented sensitivity to each. Our transventricular profile approach transcends all of them by showing the continuous picture. The sensitivity curve then lets any reader find their preferred definition and read off the corresponding proxy accuracy.

### Coordinate Choice: Euclidean tau vs Laplace tau (Empirical Comparison)

Cobiveco (Schuler et al. 2021, Sec. 2.3, Fig. 2) ranks transventricular coordinate approaches from worst to best:
1. **Eikonal quotient** g1/(g1+g2) — cusps, non-uniform spacing
2. **Laplace solution** u12 — smooth but non-linear with varying thickness
3. **Poisson-normalized** p12 — better but trajectories lose bijectivity
4. **Trajectory distances** d12 — best linearity, requires additional PDE solves

Based on this, Laplace (level 2) seemed preferable to Euclidean (level 1). We tested both on our UKB mesh using `explore_tau_v2.py`:

**Euclidean tau** = d_LV / (d_LV + d_RV):
- Septum range: [0.30, 0.74] — compact, centered on 0.5
- **Monotonic in d_LV**: cells near LV endo always get lower tau, cells near RV endo always get higher tau
- Shows DG0 marbling at coarse resolution — cosmetic artifact that disappears with binning or finer mesh
- Cell distribution heavily clustered in middle bins (114 cells near tau=0.5, only 4 at edges)

**Laplace tau** = lv_rv_scalar (u=1 on LV endo, u=0 on RV endo):
- Septum range: [0.11, 0.91] — spread across nearly full [0,1] range
- Visually smoother (CG1 continuity enforced at nodes)
- **Non-monotonic in d_LV**: cells at d_LV=9.9mm appear in the same tau bin as cells at d_LV=5mm, because the Laplace bowing artifact at the junctions leaks into the septum interior
- At the base, the Laplace field assigns LV-side values to RV-facing cells and vice versa — visible in ParaView as a thin basal lip with inverted LV/RV assignment
- Mean |Euclidean - Laplace| = 0.23 across geometric septum cells — these are fundamentally different coordinates, not small corrections

**Both fail the linearity check** (CV > 0.9), but for different reasons. The Euclidean tau compresses the mid-septum; the Laplace tau scrambles the through-wall ordering.

**Decision: use Euclidean tau.** The marbling is cosmetic (bins smooth it), while the Laplace through-wall scrambling is a systematic error that binning cannot fix. The Euclidean coordinate is simpler, monotonic, and physically interpretable ("relative distance from LV endo"). Its main weakness — non-uniform contour spacing with varying wall thickness (Cobiveco Fig. 2) — is less severe in the thin, roughly planar septum than in the curved free wall where Cobiveco demonstrated it.

**Note on Cobiveco's recommendation**: Cobiveco's evaluation focused on the *entire* biventricular domain including the free wall, where thickness variation is large and the Eikonal quotient performs worst. For the septum-restricted analysis we perform here, the Euclidean approach avoids the Laplace bowing artifact that proves more harmful than the non-uniform spacing it was meant to fix. Cobiveco's trajectory distance approach (level 4) would likely be the best of both worlds, but requires additional PDE solves not justified for this thesis scope.

For future work involving cross-patient comparison or whole-heart analysis, Cobiveco's full coordinate system would be the appropriate framework.

## The Two-Boundary Problem

The 3-way LV/RV/Septum split involves **two** distinct boundaries:

1. **Septum vs free wall** -- "is this cell part of the shared wall?"
2. **LV-side vs RV-side within the septum** -- "which ventricle dominates here?"

Our sensitivity problem is entirely about boundary #2. Boundary #1 is anatomically unambiguous (the septum is tissue sandwiched between two cavities, not facing the epicardium) and is well-handled by geometric tagging. Nobody disputes what the septum *is*. The dispute is where, within the septum, LV-dominated behavior gives way to RV-dominated behavior.

Our approach: bound boundary #1 using the geometric definition (lower bound) and LDRB scalar-field definition (upper bound), then study boundary #2 as a continuous function (tau profiles) within this region. The sensitivity curve shows how the conclusion changes as boundary #1 moves between these bounds.

## Proposed Method: Transventricular Work Density Profiles

### Coordinate System

**Terminology note**: in Cobiveco's framework (Schuler et al. 2021), "transmural" (m) means endo-to-epi depth, while "transventricular" (v) means LV-to-RV position. Our tau sweeps from LV endocardium to RV endocardium across the septum -- it is a **transventricular position coordinate**, analogous to a continuous version of Cobiveco's binary v. We use "transventricular" throughout to avoid confusion with the endo-to-epi transmural direction.

For each cell in the mesh, compute:

**Transventricular coordinate** tau in [0, 1]:
```
tau(x) = d_LV(x) / (d_LV(x) + d_RV(x))
```
where d_LV, d_RV are Euclidean distances from cell centroid to nearest LV/RV endocardial surface vertex (computed via cKDTree). This is the Eikonal quotient g1/(g1+g2) from Cobiveco's Eq. 4.

- tau ~ 0: near LV endocardium
- tau ~ 0.5: equidistant from both endo surfaces (mid-septum)
- tau ~ 1: near RV endocardium

See "Coordinate Choice" section above for why Euclidean tau was chosen over Laplace-based alternatives.

**Euclidean distances** d_LV, d_RV, d_epi (via cKDTree to surface vertices) serve double duty: defining tau AND identifying the study region (geometric septum tagging).

**Endo proximity** d_sum = d_LV + d_RV:
- Small d_sum: cell is sandwiched between two nearby endocardial surfaces (septal or near-septal tissue)
- Large d_sum: cell is far from both endocardial surfaces (epicardial junction or deep free wall)

### Study Region Definition

The study region is bounded by two septum definitions that serve as **lower and upper bounds** on the sensitivity sweep:

**Lower bound — Geometric septum** (tight, anatomically conservative):
```
is_geometric_septum = max(d_LV, d_RV) < d_epi
```
A cell is septal iff it is closer to both endocardial surfaces than to the epicardium. This is the definition from the `geometric-septum-tagging` branch of fenicsx-ldrb. It produces a compact septum that excludes ambiguous junction cells.

**Upper bound — LDRB scalar-field septum** (wide, known to over-tag):
```
is_ldrb_septum = (epi_scalar <= 0.5) AND (0.1 < lv_rv_scalar < 0.9)
```
This is the original LDRB heuristic from the main branch of fenicsx-ldrb. It uses the Laplace epi-scalar to identify cells "far from the epicardium" and the lv_rv_scalar to identify cells "between the two ventricles." It is known to over-tag at the junctions because the epi-scalar bows outward there, but it represents a reasonable upper limit on how generously the septum could be defined using standard scalar-field approaches.

**Study region** = union of both definitions:
```
study_region = is_geometric_septum | is_ldrb_septum
```

This union captures all cells that *either* definition considers septal. The sensitivity curve then sweeps from narrow (geometric only) to wide (full union), showing how proxy accuracy changes. Key reference points on the curve:

| Definition | Description | Role on sensitivity curve |
|------------|-------------|--------------------------|
| Geometric septum | `max(d_LV, d_RV) < d_epi` | Lower bound (tightest) |
| LDRB septum | `epi ≤ 0.5 AND 0.1 < lv_rv < 0.9` | Upper bound (widest) |
| TriSeg 1/3 volume | Volume-matched tau range | External reference |

This design has two advantages:
1. **No arbitrary d_sum threshold**: the study region is defined by the union of two independently-motivated definitions, not a hand-tuned distance parameter.
2. **Interpretable bounds**: a reviewer familiar with LDRB can recognize the upper bound as "the standard approach" and the lower bound as "the conservative geometric approach." The sensitivity curve shows exactly how much the conclusion depends on this choice.

### Epicardial Junction Safety Filter

The LDRB scalar-field definition can include cells at the epicardial junction (where LV and RV free walls meet at the outer surface) because the epi-scalar bows outward there. These cells have intermediate lv_rv values but are mechanically free-wall tissue.

As a safety check, we exclude any cell in the study region with d_sum > D_SUM_MAX (a generous threshold, e.g., 25mm). Epicardial junction cells have large d_sum because they are far from both endocardial surfaces. Genuine septal cells have small d_sum (8-15mm on typical meshes). This is not a tuning parameter — it is a hard cutoff to remove obvious misclassifications, set well above the septal d_sum range.

### Hybrid Approach: Euclidean Distances for Coordinate, LDRB Scalars for Upper Bound

We use Euclidean distances and LDRB Laplace scalar fields for **different purposes**:

| Purpose | Method | Rationale |
|---------|--------|-----------|
| **Transventricular coordinate** (tau) | `d_LV/(d_LV+d_RV)` (Euclidean) | Monotonic in through-wall position, no bowing artifact, physically interpretable |
| **Geometric septum lower bound** | `max(d_LV, d_RV) < d_epi` (Euclidean) | Conservative septum boundary, mesh-independent |
| **LDRB septum upper bound** | `epi ≤ 0.5 AND 0.1 < lv_rv < 0.9` (Laplace) | Standard LDRB approach — generous septum boundary, known to over-tag at junctions |
| **Junction safety filter** | `d_sum < D_SUM_MAX` (Euclidean) | Removes obvious epicardial junction misclassifications from the LDRB upper bound |

**Why Euclidean for the coordinate?** See "Coordinate Choice" section — empirical comparison showed Laplace tau scrambles the through-wall ordering at the base due to bowing, while Euclidean tau is monotonic despite cosmetic marbling.

**Why Laplace for the upper bound?** The LDRB scalar-field septum definition (`epi_scalar ≤ 0.5 AND 0.1 < lv_rv_scalar < 0.9`) is the standard approach in the fenicsx-ldrb main branch. It over-tags at junctions (the bowing artifact adds ~37% more cells than geometric tagging), making it a natural generous upper bound. The 119 LDRB-only extension cells (on the UKB mesh) are concentrated at the basal junction — anatomically ambiguous but not unreasonable for a "wide" septum definition.

### Empirical Validation (explore_tau.py, explore_tau_v2.py)

On the UKB healthy mesh (char_length=10, 2120 cells):

**Study region statistics (from explore_tau_v2.py):**
- Geometric septum: 318 cells (15.0%)
- LDRB scalar septum: 436 cells (20.6%) — 37% larger
- Union (study region): 437 cells (20.6%)
- LDRB-only extension: 119 cells at basal junction (d_epi 3-9mm, d_sum 11-22mm)
- Geometric-only: 1 cell (negligible — the two definitions almost fully overlap where geometric is tighter)

**Euclidean tau distribution by geometric region:**
- LV free wall: tau in [0.06, 0.50], mean=0.17, median=0.13
- RV free wall: tau in [0.50, 0.93], mean=0.83, median=0.86
- Geometric septum: tau in [0.30, 0.74], mean=0.50, median=0.50
- LDRB-only extension: tau in [0.20, 0.78], mean=0.49, median=0.48

**Key findings:**
- Free wall cells cluster at the extremes (LV < 0.20, RV > 0.80)
- The mid-range tau in [0.20, 0.80] is dominated by septal and near-septal cells
- tau = 0.5 sits at mid-septum (mean and median = 0.50)
- DG0 marbling visible at coarse resolution — cosmetic, disappears with binning (10-15 bins)
- LDRB extension cells extend the tau range by ~0.08 on each side (from [0.30, 0.74] to [0.20, 0.78])

### Per-Cell Work Density

For each cell in the study region, accumulate over the last cardiac cycle:

- **True work density**: w_true(x) = integral_beat S(x):dE(x) dt
- **Proxy work densities**:
  - w_PLV(x) = integral_beat P_LV(t) * d_eps_ff(x,t)
  - w_PRV(x) = integral_beat P_RV(t) * d_eps_ff(x,t)
  - w_Trans(x) = integral_beat (P_LV - P_RV)(t) * d_eps_ff(x,t)
  - Same for eps_ll (longitudinal strain, clinically = GLS)

The pressure is a global scalar (from the cavity Lagrange multiplier). The strain is per-cell.

## Two distinct questions this framework can answer

After iterating on the design, we realised that the per-cell + tau + pressure-proxy infrastructure we built can answer two **independent** questions. They use the same per-cell arrays but different sweep axes and aggregation rules. The primary question (A) is the one we're attacking first; the secondary question (B) is queued and has different clinical relevance.

### Question A — primary clinical question

**"Given a septum definition, how reliably does each pressure proxy track changes in septal work across disease severity?"**

- **Inner aggregation**: sum proxy and true work over all cells in the current septum definition, independently for each of the 8 spectrum cases. Produces one (W_true, W_proxy) pair per case.
- **Sweep axis**: septum definition width. We need the sweep endpoints to correspond to real, named septum definitions that a clinician or reviewer might use.
- **Outer metric**: Pearson `r` (signed) across the 8 spectrum cases between per-case `W_true` and per-case `W_proxy`, computed at each sweep step.
- **Interpretation**: "If a clinician draws the septum THIS way, how well does each proxy track disease progression?"
- **Clinically answerable**: yes — a reader picks their preferred septum definition between the sweep endpoints and reads off the proxy's across-patient tracking quality.

### Question B — queued mechanics question

**"Does the best proxy depend on where you are within the septum wall? Does P_LV track best on the LV side and P_RV track best on the RV side?"**

- **Inner aggregation**: sum proxy and true work over all cells in a narrow band of tau values (a "slice" of the septum at a given transventricular position), independently for each of the 8 cases.
- **Sweep axis**: tau (position coordinate), from LV-side (low tau) to RV-side (high tau).
- **Outer metric**: Pearson `r` across the 8 spectrum cases, per tau band.
- **Interpretation**: "At each position through the septum wall, which proxy tracks disease progression best?"
- **Clinically answerable**: **no**. A standard clinical echo cannot measure strain on only one half of the septum — GLS is measured from the full wall, and cavity pressure is a single scalar per ventricle. Question B answers a mechanics question, not a clinical proxy-design question.
- **Connection to P_dom**: `P_dom` is a specific bet on what question B's answer will be — "use the closer cavity's pressure per cell". If question B confirms a position-dependent regime where P_LV is best at low tau and P_RV at high tau, then P_dom (aggregated over the full septum) is the corresponding clinical-friendly proxy that implements that local rule. Question B could also reveal that the crossover isn't exactly at tau=0.5, that neither pressure works at some intermediate tau, or that the transition is smooth rather than sharp.

**Decision**: question A is the primary research focus. Question B is queued for later investigation using the same per-cell data infrastructure.

## Sensitivity Curve — Question A (primary)

### Design: additive sweep anchored to the geometric definition

The sweep is parameterised by an integer `k` counting cells included in the septum definition. The ordering is designed so that:

1. **At k = |geometric|**: the septum is **exactly** the geometric definition (`max(d_LV, d_RV) < d_epi`).
2. **At k = |geometric ∪ LDRB|**: the septum is the full study region (geometric ∪ LDRB = all cells that either definition considers septal).
3. **Intermediate k**: start with all geometric cells, then append LDRB-only cells one at a time in order of increasing `d_sum = d_LV + d_RV` (smallest d_sum = most "sandwiched between both endocardia" = most septal).
4. **Optionally below k = |geometric|**: remove geometric cells in order of **decreasing** d_sum (least septal first), allowing the sweep to explore septum definitions tighter than geometric.

This produces a strictly nested sequence of septum definitions: every cell in a narrower definition is also in every wider definition. No cell is ever added and then removed.

### Sweep algorithm

```python
# 1. Base: all geometric septum cells
geo_cells = np.where(is_geometric_septum)[0]

# 2. LDRB-only extras: cells in LDRB but NOT in geometric
ldrb_only = np.where(is_ldrb_septum & ~is_geometric_septum)[0]

# 3. Sort LDRB-only extras by ascending d_sum (most septal first)
ldrb_only_sorted = ldrb_only[np.argsort(d_sum[ldrb_only])]

# 4. Optionally sort geometric cells by d_sum (ascending) so
#    "removing from the right" = removing least septal first
geo_sorted = geo_cells[np.argsort(d_sum[geo_cells])]

# 5. The full ordered sweep = tight → wide
#    (read the sweep as: at step k, include the first k cells of this list)
full_order = np.concatenate([geo_sorted, ldrb_only_sorted])

# 6. At each step k, the septum = set of first k cells
for k in range(1, len(full_order) + 1):
    mask = np.zeros(n_cells, dtype=bool)
    mask[full_order[:k]] = True
    W_true[case, k] = w_total[case][mask].sum()
    W_PLV[case, k]  = proxy_PLV_ll[case][mask].sum()
    # ... etc. for each proxy
```

### Reference points on the sweep

| Sweep position | Meaning |
|----------------|---------|
| `k = 0` | empty septum (or skipped) |
| `k = |geometric|` | **geometric septum exactly** — the anatomically conservative definition |
| `k = |geometric ∪ LDRB|` | **full study region (geometric ∪ LDRB)** — includes all cells either definition tags as septal |

### Plot semantics

- **x-axis**: k (number of cells in septum definition), or equivalently a fraction `k / |geometric ∪ LDRB|`, with vertical reference lines marking `k = |geometric|` and `k = |geometric ∪ LDRB|`.
- **y-axis**: Pearson `r` across the 8 spectrum cases, between per-case W_true and per-case W_proxy, computed at each sweep step.
- **One curve per proxy**: P_LV × dε_ll, P_RV × dε_ll, (P_LV − P_RV) × dε_ll, (P_LV + P_RV)/2 × dε_ll, P_dom × dε_ll.

The sweep is additive and nested by construction — cells are only added (or only removed if extending below geometric). This eliminates the ambiguity of the earlier symmetric-tau-window approach, which was not the geometric definition at any sweep point.

### Historical note — why we abandoned the symmetric tau-window sweep

An earlier version of this section (Phase 4 code pre-2026-04-10) used a symmetric tau window:

```
for tau_cutoff in linspace(tau_geo.min(), tau_ldrb.min(), N):
    mask = study_region & (tau >= tau_cutoff) & (tau <= 1 - tau_cutoff)
```

This was found to be **not what we actually wanted for Question A**. At the tightest sweep point, the mask was "cells in study_region with tau in the symmetric window `[max(tau_geo_min_across_cases), 1 − max(tau_geo_min)]`" — which is neither the geometric septum nor any other named definition. The tau-window sweep turned out to be a **position sweep** (LV-side vs RV-side coverage), which is closer in spirit to Question B than Question A. See the Queued section below for why a tau-position sweep is still interesting as a separate analysis.

## Transventricular Profiles — Question B (queued)

> **CRITICAL UPDATE 2026-04-14 — elevate Question B from "queued" to "required for defense".**
>
> Re-examining the role of Question B in the thesis argument: **Question A alone (scalar r across disease cases, stable under septum-boundary sweep) is defensible but not explanatory**. It tells a supervisor that P_LV's correlation is not a tagging artifact, but it does not tell them *why* the correlation holds, and it cannot distinguish "P_LV tracks truth across the whole wall" from "P_LV only tracks the LV-side of the septum and the LV-side happens to dominate the integral." Those two explanations predict identical Question-A curves but diverge in severe PAH, where RV hypertrophy can shift the integral weight toward the RV side — exactly the regime the thesis is about.
>
> The transmural density profile (Question B) is the **mechanistic lens that resolves this ambiguity**. Three concrete claims it enables that Question A cannot:
>
> 1. **Spatial validity of each proxy.** Plotting `rho_true(tau)`, `rho_PLV(tau)`, `rho_PRV(tau)`, `rho_Trans(tau)` as continuous curves over the wall reveals *where* each pressure is a valid local predictor. Expected pattern: P_LV matches near tau≈0.3 (LV side), P_RV matches near tau≈0.7 (RV side), P_Trans matches across the whole wall. That is a pointwise statement, which is what clinical myocardial-work imaging actually needs — clinicians apply the proxy pointwise to strain maps, not to a transmurally-averaged septum.
>
> 2. **Testable artifact hypothesis for Question A.** If the profile shows P_LV only tracks near the LV side but its aggregate r is still 0.99, then the LV side must dominate the integral — the RV-side error is small, not small-and-tracked. This generates a **falsifiable prediction**: P_LV should degrade in cases where RV-side work grows enough to shift the integral weight. Check this against the 8-case spectrum. If the prediction holds, the aggregate-r result is reframed honestly as "P_LV is a good aggregate proxy *because* the LV side dominates in our cohort". If it fails, the aggregate result stands on its own.
>
> 3. **Second thesis finding: transmural reorganization with disease.** Overlay `rho_true(tau)` curves for all 8 cases. The profile shape itself may migrate across the disease spectrum (RV-side work growing in PAH as the RV hypertrophies into the septum). That is a **physics** statement about PAH septal mechanics, independent of proxy validation, and gives the thesis a second story that does not depend on the proxy ranking at all.
>
> **Status change**: Question B was queued on the argument that it is not clinically answerable by standard echo (true — it remains a mechanics question) and therefore not a proxy-design result (also true). But it is now required as the **mechanistic explanation** for Question A. Without it, the Question-A r≈0.99 is vulnerable to the "curve fit" critique: a supervisor can ask "how do you know this isn't an integration-weight coincidence?" and the scalar sweep alone cannot answer. The transmural profile can.
>
> **Re-framing of the thesis structure**:
> - Question A (scalar sweep) = **robustness section** — establishes that the proxy ranking is real, not a boundary-definition artifact.
> - Question B (transmural profile) = **mechanism section** — explains *where* in the wall each proxy is valid and *why* the aggregate result holds. Also delivers the independent physics finding about transmural work redistribution under PAH.
> - Both together = a defensible story: the ranking is robust (A) *and* mechanistically grounded (B).
>
> **Minimum viable experiment before committing**: one afternoon's work on the healthy UKB case. Take the existing `per_cell_data.npz`, bin cells by tau (10 strata), compute `rho_true`, `rho_PLV`, `rho_PRV`, `rho_Trans` within the fixed geometric septum, plot all four curves on one axis. If they look like the expected LV-side/RV-side/both pattern → the mechanism story is there, commit to Question B as a primary result. If they are flat and parallel → the mechanism lens adds nothing, stay with Question A only. This experiment uses already-saved data, no new simulations, no new per-cell computation.

**Status**: **ELEVATED from queued to required-for-defense as of 2026-04-14** — see update block above. The data infrastructure (`per_cell_data.npz` with tau, per-cell w_total, and per-cell proxies) already supports this analysis — it requires only a new analysis script, no new simulations or new per-cell computation.

### Design: tau-band sweep at fixed septum definition

Instead of sweeping the septum definition width (Question A), fix the septum to the full study region and sweep a narrow **position band** across tau:

```python
# Fix the septum
septum_mask = study_region  # use the full study region, no width sweep

# Sweep a narrow window across tau
band_width = 0.05  # ~5% of the tau range
tau_centers = np.linspace(tau[septum_mask].min() + band_width/2,
                          tau[septum_mask].max() - band_width/2, N)
for tau_c in tau_centers:
    band_mask = septum_mask & (tau >= tau_c - band_width/2) & (tau < tau_c + band_width/2)
    # Sum over this thin slice in each case, compute Pearson r across cases
```

### Plot semantics

- **x-axis**: tau position (from LV-side ≈ 0.2 to RV-side ≈ 0.8)
- **y-axis**: Pearson `r` across the 8 spectrum cases, per proxy, per band
- **One curve per proxy**, same 5 proxies as Question A

### What the plot would show

- If `r_PLV(tau)` is high at low tau and drops at high tau, and `r_PRV(tau)` does the opposite, then the septum has position-dependent mechanics and the "correct" proxy depends on where you are
- If all curves are flat at the same level, the septum behaves uniformly and no position dependence exists
- If `r_Trans(tau)` is high everywhere but `r_PLV(tau)` and `r_PRV(tau)` are low, the difference matters at every position
- If `r_dom(tau)` is high everywhere and matches whichever single pressure is highest at each position, the hard tau=0.5 split of P_dom is a valid approximation

### Relation to P_dom

`P_dom` (defined in compute_per_cell.py / plot_sensitivity_ll.py) assigns `P_LV` to cells with `tau < 0.5` and `P_RV` to cells with `tau ≥ 0.5`. If Question B's answer is "LV-side cells are best tracked by P_LV and RV-side cells by P_RV", then P_dom is the correct septum-level aggregation of that local rule. If Question B's answer is more nuanced (smooth transition, non-0.5 crossover, intermediate tau regime), then P_dom is a coarse approximation and a refined version would be informative.

### Why queued and not primary

1. **Not clinically answerable by standard echo**. GLS is measured from the full septal wall; cavity pressures are single scalars. Echo cannot assign different pressures to different halves of the septum in vivo.
2. **Mechanics question, not proxy-design question**. Question B asks whether the septum behaves non-uniformly through its wall. Interesting for understanding, but not directly actionable for a clinical proxy recommendation.
3. **Primary question A answers the clinical motivation first**. Once we have question A's answer, we can optionally return to B for deeper mechanistic insight.

**When we do return to B**, the data is already in place — we only need a new analysis script that bins the same per-cell arrays by tau instead of summing over the septum.

## Critical Design Decisions

### Quadrature Points, Not DG0 Projection

Previous experience showed that projecting stress/strain to DG0 breaks the energy balance: the S:dE integral no longer matches the PdV boundary work (can be off by ~50%). This happens because DG0 averaging within a cell loses the correlation between S and dE at individual quadrature points.

For per-cell work, we MUST integrate at the quadrature points within each cell. The key technique: use a DG0 test function to extract per-cell integrals in a single global assembly:

```python
# One-shot per-cell integration (no cell loop needed):
V_DG0 = FunctionSpace(mesh, ("DG", 0))
v = TestFunction(V_DG0)
work_per_cell_form = form(S_inner_dE * v * dx)
work_per_cell_vec = assemble_vector(work_per_cell_form)
# work_per_cell_vec[i] = integral_cell_i S:dE dx  (exact at quadrature points)
```

This uses the SAME quadrature rule as the regional integration. No projection, no accuracy loss. The DG0 test function acts as a spatial filter, not a projection target.

**Do NOT use**: project S and E to DG0 separately, then multiply. That destroys the pointwise S*dE correlation and breaks energy balance.

### Validation: Energy Balance Must Pass

Before trusting any per-cell result:

1. `sum(w_true_per_cell)` must equal `W_true_Whole` (regional integral) to machine precision
2. `sum(w_true_per_cell[cells_in_LV])` must equal `W_true_LV` (with current tagging)
3. Same checks for proxy work values

If these fail, the implementation is wrong. This is non-negotiable -- we've been burned before by projection artifacts.

### Proxy Ratio Interpretation

The proxy ratio R(tau) = w_proxy / w_true will NOT be 1.0 even where the "right" pressure is used. The proxy uses one strain component (eps_ff or eps_ll), while w_true is the full tensor contraction S:dE across all directions (fiber, sheet, sheet-normal, cross-fiber). Fiber work is ~25-35% of total internal work.

Options for the denominator:
- R = w_proxy / w_true: measures what fraction of TOTAL work the proxy captures. Clinically relevant (clinicians want to estimate total work).
- R = w_proxy / w_ff: measures how well the proxy tracks the FIBER component. Mechanistically cleaner.

Decision: present both. R vs w_true is the primary clinical metric. R vs w_ff as supplementary, to separate the "wrong pressure" effect from the "missing directions" effect.

### Handling Near-Zero and Negative Work

In PAH, some septal cells may have near-zero or negative true work (septum being pushed by elevated RV pressure). The ratio R = w_proxy/w_true diverges or flips sign at these cells.

Solution: plot w_proxy(tau) and w_true(tau) as **separate curves** rather than only as a ratio. The divergence of the curves is well-defined even when w_true crosses zero. The ratio plot is supplementary, with near-zero bins flagged or excluded.

## Layered Presentation

**Layer 1 -- The transventricular profile** (for us and detailed reviewers):
Full transventricular work density w(tau) and proxy curves for each severity level. This is the complete, boundary-free picture. Shows where each proxy tracks the true work and where it diverges.

**Layer 2 -- The sensitivity curve** (bridging continuous to discrete):
Proxy accuracy integrated over [tau_cutoff, 1-tau_cutoff] for a sweep of tau_cutoff values. Marked with reference points for geometric tagging, LDRB tagging, TriSeg volume. Shows how stable/unstable the conclusion is as you shift the boundary. Directly explains why the old tagging and new tagging gave different answers -- they are different points on this curve.

**Layer 3 -- Clinical summary** (for the skimming reader):
"P_LV x eps_ff captures X +/- Y% of septal work, where Y quantifies segmentation uncertainty." The +/-Y comes directly from the sensitivity curve. This is an honest, clinically useful statement that acknowledges measurement reality.

## Implementation Plan

### Phase 1: Per-Cell Work Accumulation (metrics_calculator.py)

Modify `MetricsCalculator` to accumulate per-cell work via DG0 test function assembly:

```python
# At init: compute Euclidean distances and tau
centroids = compute_midpoints(mesh, 3, local_cells)
d_lv = cKDTree(lv_surface_coords).query(centroids)[0]
d_rv = cKDTree(rv_surface_coords).query(centroids)[0]
d_epi = cKDTree(epi_surface_coords).query(centroids)[0]
d_sum = d_lv + d_rv
tau = d_lv / (d_lv + d_rv)

# Study region: union of geometric and LDRB septum definitions
# Requires LDRB scalar fields (epi_scalar, lv_rv_scalar) — solve Laplace or load from LDRB
is_geometric_septum = np.maximum(d_lv, d_rv) < d_epi
is_ldrb_septum = (epi_scalar <= 0.5) & (lv_rv_scalar > 0.1) & (lv_rv_scalar < 0.9)
study_region = (is_geometric_septum | is_ldrb_septum) & (d_sum < D_SUM_MAX)

# Set up DG0 test function for per-cell integration
V_DG0 = FunctionSpace(mesh, ("DG", 0))
v_dg0 = TestFunction(V_DG0)

# At each timestep: assemble per-cell work increment
dW_cells = assemble_vector(form(S_inner_dE * v_dg0 * dx))
cumulative_work_cells += dW_cells  # trapezoidal accumulation

# Same for proxy: P(t) * d_eps_ff(x) * v_dg0 * dx
```

**Storage**: ~2000 cells x 12 values = 24K floats per simulation. Negligible.
Per-cell arrays to store (net over last beat):
- w_true[cell]: total internal work density
- w_ff[cell], w_ss[cell], w_nn[cell], w_cross[cell]: directional decomposition
- w_proxy_{PLV,PRV,Trans}_{ff,ll}[cell]: 6 proxy variants
- tau[cell]: Euclidean transventricular coordinate (computed once)
- is_geometric_septum[cell]: geometric tag (computed once)
- is_ldrb_septum[cell]: LDRB scalar-field tag (computed once)
- d_sum[cell]: endo proximity for junction filtering (computed once)

### Phase 2: Validation

1. Energy balance: `sum(w_true_per_cell) == W_true_Whole` (must match to <1e-10)
2. Regional consistency: `sum(w_true_per_cell[region]) == W_true_region` (exact match)
3. Proxy consistency: `sum(w_proxy_per_cell) == W_proxy_regional` (exact match)
4. **Tau linearity check**: bin study-region cells by tau into 10 bins, compute mean d_LV per bin, measure incremental d_LV change between bins. For a linear coordinate, increments should be roughly equal. Report CV of increments. On the UKB mesh, both Euclidean and Laplace tau fail (CV ~0.9) — Euclidean due to mid-septum compression, Laplace due to through-wall scrambling. Euclidean was chosen despite this because its failure mode (non-uniform bin density) is less harmful than Laplace's (wrong ordering). Verify on patient-specific and thickness variant meshes.

### Phase 3: Transventricular Profile Analysis (new script: `analyze_transventricular.py`)

```python
# Study region = union of geometric + LDRB, with d_sum safety filter
study_region = (is_geometric_septum | is_ldrb_septum) & (d_sum < D_SUM_MAX)

# Bin by Euclidean tau within study region
bins = np.linspace(tau[study_region].min(), tau[study_region].max(), N_bins + 1)
for b in range(N_bins):
    mask = study_region & (tau >= bins[b]) & (tau < bins[b+1])
    V_bin = sum(V_cell[mask])
    profile_true[b] = sum(w_true[mask]) / V_bin
    profile_PLV[b] = sum(w_proxy_PLV[mask]) / V_bin
    ratio_PLV[b] = profile_PLV[b] / profile_true[b]
```

### Phase 4: Sensitivity Analysis — Additive cell-count sweep (Question A)

```python
# For each case in the spectrum, build the nested sweep:
# start with all geometric cells, then add LDRB-only cells in order of d_sum
# (ascending), so each step adds the most septal cell remaining.

geo_cells      = np.where(is_geometric_septum)[0]
ldrb_only      = np.where(is_ldrb_septum & ~is_geometric_septum)[0]
ldrb_sorted    = ldrb_only[np.argsort(d_sum[ldrb_only])]
# Optionally also sort geo_cells by ascending d_sum so the "below geometric"
# region of the sweep can remove cells least-septal-first.
geo_sorted     = geo_cells[np.argsort(d_sum[geo_cells])]
full_order     = np.concatenate([geo_sorted, ldrb_sorted])

# Outer loop: sweep step k (number of cells in the definition)
for k in range(1, len(full_order) + 1):
    mask = np.zeros(n_cells, dtype=bool)
    mask[full_order[:k]] = True
    # Inner loop: per-case sums
    for case in cases:
        W_true[case, k]  = case.w_total[mask].sum()
        W_PLV[case, k]   = case.proxy_PLV_ll[mask].sum()
        W_PRV[case, k]   = case.proxy_PRV_ll[mask].sum()
        W_Trans[case, k] = case.proxy_Trans_ll[mask].sum()
        W_mean[case, k]  = 0.5 * (W_PLV[case, k] + W_PRV[case, k])
        # W_dom uses each cell's tau value to pick P_LV or P_RV per cell
        W_dom[case, k]   = dominant_proxy(case, mask)

# Across-case Pearson r at each sweep step
for k in range(N_steps):
    r_PLV[k]   = pearsonr(W_true[:, k], W_PLV[:, k])[0]
    r_PRV[k]   = pearsonr(W_true[:, k], W_PRV[:, k])[0]
    r_Trans[k] = pearsonr(W_true[:, k], W_Trans[:, k])[0]
    r_mean[k]  = pearsonr(W_true[:, k], W_mean[:, k])[0]
    r_dom[k]   = pearsonr(W_true[:, k], W_dom[:, k])[0]

# Reference points:
#   k = |geometric|            → geometric septum (anatomically conservative)
#   k = |geometric ∪ LDRB|     → full study region (generous)
```

**Per-case consistency note**: because the current 8 sims used slightly different gmsh-generated meshes (known mesh-variation issue, to be fixed by generating one shared mesh and reusing via `--geometry-dir`), each case has slightly different `|geometric|` and `|geometric ∪ LDRB|` counts. For the across-case Pearson correlation we need one common sweep axis. Options:

1. **Fraction f = k / |geometric ∪ LDRB|**: normalize per case, sweep f ∈ [0, 1]. Each case contributes its own sweep interpolated onto a common f grid.
2. **Min-cell-count**: use the smallest |geometric ∪ LDRB| across cases as the upper limit on k; drop the last few cells of cases with more.
3. **Best option — fix the mesh first**: regenerate a single shared UKB mesh, rerun the 8 sims, then each case has identical cell counts and the sweep uses the same k for every case.

Plan: fix the mesh (Option 3) before running the final version of the Phase 4 analysis. The mesh fix is independent of all analysis code and can be queued as a sim rerun.

### Phase 5: Mesh Convergence

The current UKB mesh (char_length=10, 2120 cells) has only 318 septal cells. This gives ~30 cells per tau bin (at 10 bins), which is marginal. Before drawing conclusions:

- Rerun on a finer mesh (char_length=5 or smaller) to get ~2500+ septal cells
- Verify that profiles are qualitatively unchanged (same crossover location, same shape)
- If profiles change with mesh refinement, the coarse mesh is insufficient

## Open Questions

1. **Apicobasal variation**: tau captures transventricular position but not apicobasal level. The septum is thicker at the base than the apex, fiber angles vary apicobasally, and PAH remodeling is non-uniform along this axis. This is not merely a nice-to-have check -- averaging over apicobasal level could wash out real gradients. At minimum, split the profiles into 2-3 apicobasal slices (basal, mid, apical) and verify the 1D profiles are qualitatively stable. If they diverge, 2D profiles (tau vs apicobasal coordinate) are needed. Cobiveco's apicobasal coordinate a could serve this purpose if implemented; alternatively, a simple apex-to-base distance normalized by heart length is a pragmatic substitute.

2. **AHA segment mapping**: the AHA 17-segment model defines the septum via angular sectors in the rotational/circumferential direction (segments 2, 3, 8, 9, 14 span specific angular ranges). This is fundamentally a *circumferential* segmentation, not a transventricular one -- AHA segments span the full wall thickness. Mapping AHA to our tau framework would require computing a rotational coordinate (Cobiveco's r, or a simplified circumferential angle). Without this, we cannot place AHA reference points on our sensitivity curve. This is an acknowledged limitation -- the TriSeg volume definition remains as the primary external reference.

3. **Coordinate limitations (Cobiveco comparison)**: our Euclidean tau is the Eikonal quotient that Cobiveco found suboptimal for general use (their Fig. 2). We tested the Laplace alternative (lv_rv_scalar) and found it worse for the septum specifically — the bowing artifact at junctions produces non-monotonic through-wall ordering even within the filtered study region (see "Coordinate Choice" section). Cobiveco's trajectory distances (Eqs. 8-9) would likely avoid both problems but require additional PDE solves. For our coarsely-binned septum-restricted analysis, the Euclidean approach is adequate — its weakness (non-uniform bin density) is handled by volume-weighted averaging within bins.

## Exploration Scripts

- `explore_tau.py`: original script — Euclidean tau, geometric tags, zone classification. Exports XDMF.
- `explore_tau_v2.py`: comparison script — solves Laplace equations, compares Euclidean vs Laplace tau, defines geometric + LDRB study region, exports all fields to XDMF for ParaView inspection. Includes linearity check.

## Status

### Infrastructure (done)
- [x] Define transventricular coordinate (Euclidean tau) and validate on UKB healthy mesh
- [x] Identify and resolve epicardial junction contamination (d_sum filter)
- [x] Validate zone classification visually in ParaView
- [x] Literature review: position relative to UVC, Cobiveco, AHA, TriSeg definitions
- [x] Decision: study region = union of geometric septum + LDRB scalar-field septum
- [x] Compare Euclidean vs Laplace tau (explore_tau_v2.py): Laplace has bowing artifact at base → **use Euclidean tau**
- [x] Implement DG0 test function per-cell integration (compute_per_cell.py)
- [x] Validation: per-cell sum matches regional integral (whole-mesh: 0.0% error, per-region: <1.5% modulo retag differences)
- [x] Per-cell data generated for 8 spectrum cases and 6 thickness variants
- [x] Separate Question A (primary, clinical) from Question B (queued, mechanics)

### Known problems to fix before final results
- [ ] **Mesh variation between spectrum sims**: each sim currently regenerates the UKB mesh via gmsh, producing slightly different meshes (cell counts 2143-2183). Fix by generating one shared mesh and passing via `--geometry-dir` to all 8 sims. This is a sim-rerun, independent of any analysis code change.
- [ ] **Sweep design**: replace symmetric tau-window sweep with additive cell-count sweep anchored to geometric definition (see "Sensitivity Curve — Question A" above)

### Primary analysis — Question A
- [ ] Generate one shared UKB char_length=10 mesh via `geometry_generator.py --single ukb -c 10 --output-dir data/shared_ukb_mesh`
- [ ] Rerun 8 spectrum sims with `GEOMETRY_DIR` pointing to shared mesh (all 8 use identical cells)
- [ ] Run compute_per_cell.py on new results
- [ ] Implement additive cell-count sweep with d_sum-ordered insertion
- [ ] Generate primary sensitivity curve figure (inter-case Pearson r vs sweep step, one curve per proxy, `_ll` proxies)
- [ ] Mark reference points: `k = |geometric|`, `k = |geometric ∪ LDRB|`

### Secondary analyses (maybe)
- [ ] Same sweep on thickness variants (after deciding whether thickness variants should also share a common mesh baseline)
- [ ] Apicobasal stability check: split profiles into basal/mid/apical slices
- [ ] Map TriSeg volume definition to a sweep reference point (by cell count matching 1/3 of LV wall volume)
- [ ] Mesh convergence check (single finer-mesh sim to confirm the char_length=10 result is stable)

### Queued — Question B (mechanics, through-wall variation)
- [ ] Write tau-band sweep analysis script (same per-cell data, different aggregation)
- [ ] Produce `r(tau)` plots for each proxy across the spectrum
- [ ] Compare to P_dom hard-split at tau=0.5 — does the data support a discrete split or a smooth transition?
- [ ] Document findings as a supplementary mechanics analysis (not the primary clinical story)
- [ ] Run on all 6 UKB spectrum cases
- [ ] Run on patient-specific meshes (healthy + PAH) — note: mesh units differ (UKB=mm, patient=cm)
- [ ] Run on thickness variants
- [ ] Generate layered presentation (profiles -> sensitivity -> summary)
- [ ] Write thesis section

---

# Session Update 2026-04-12 — Major Discoveries and Design Revisions

This section captures the outcome of a long working session on 2026-04-12 that
produced several new findings and forced a rethink of the septum definition
pipeline. Everything above this line is preserved as the historical record; the
content below supersedes it where they conflict.

## Summary of new findings

1. **The proxy depends on which cells you call septum.** The geometric septum
   gives P_LV as the best proxy (r = +0.961 across 8 spectrum cases). The LDRB
   septum gives P_Trans = P_LV − P_RV as best (r = +0.976). Neither definition is
   wrong; they describe different cell sets.

2. **The algebraic identity W_Trans = W_PLV − W_PRV holds exactly**, because the
   cavity pressures are spatial scalars. When the LDRB definition shifts the
   septum toward the RV (median tau 0.57 vs 0.50 for geometric), the P_RV term
   becomes non-negligible and the Trans proxy improves. This is not a new proxy
   — it is a specific reweighting of two old ones that happens to work when the
   cell set is RV-shifted.

3. **The sweep curves reveal that P_LV is robust to septum-definition choice
   while P_Trans is not.** Across t ∈ [−10, +15] mm of the sweep, P_LV's Pearson
   r stays between 0.944 and 0.996 (range 0.052). P_Trans swings from 0.481 to
   0.898 (range 0.416). The LDRB-direct result (r_Trans = 0.976) is an isolated
   outlier; at the same cell count along the sweep, Trans is only ~0.60. The
   high LDRB Trans score is about the *specific cells* LDRB picks, not about
   wider definitions in general.

4. **The envelope was too aggressive.** The original envelope
   `(d_epi ≥ 2mm) ∧ (4mm ≤ d_sum ≤ 22mm) ∧ ¬touches_epi` excluded 44–63% of
   geometric septum cells (apical region where d_sum < 4mm). This broke the
   invariant that `sweep(t=0) == geometric`. We relaxed the envelope to
   `(d_sum ≤ 22mm) ∧ ¬touches_epi` and verified that the sweep at t=0 now
   captures 100% of geometric cells (minus the ~10 cells that legitimately
   touch the epi surface in the thin apex). After the fix, CHECK 6 passes on
   all 8 cases.

5. **Cross-mesh stability argument (3 meshes).** We computed both definitions
   on three meshes: UKB synthetic, patient healthy, patient PAH.

   | Mesh            | n_cells | Geo %   | LDRB %  | Jaccard |
   |-----------------|---------|---------|---------|---------|
   | UKB synthetic   | 2153    | 15.6%   | 21.1%   | 0.731   |
   | Patient healthy | 4742    | 15.6%   | 20.6%   | 0.747   |
   | Patient PAH     | 7686    | 14.7%   | 17.2%   | 0.619   |
   | mean ± std      |         | 15.3 ± 0.4% | 19.6 ± 1.7% |     |

   Geometric fraction varies by only 0.4pp across three very different meshes;
   LDRB varies 4× more. The Jaccard drops to 0.619 on the PAH geometry — the
   LDRB definition becomes less coherent precisely on the pathological case
   that matters clinically.

6. **Regional growth argument (healthy vs PAH patient).** Computed regional
   volumes on the healthy and PAH patient meshes and looked at proportionality
   to total growth:

   | Region            | Healthy (mL) | PAH (mL) | Growth | Proportionality |
   |-------------------|--------------|----------|--------|------------------|
   | Total myocardium  | 127.5        | 164.0    | 1.29x  | 1.00 (reference) |
   | LV free wall      | 60.9         | 69.6     | 1.14x  | 0.89 (undergrows)|
   | RV free wall      | 32.7         | 56.2     | **1.72x** | **1.34 (overgrows, PAH hypertrophy signature)** |
   | Septum (geo)      | 23.1         | 26.6     | 1.15x  | **0.90 (tracks LV)** |
   | Septum (LDRB)     | 34.7         | 37.0     | 1.06x  | 0.83 (undergrows) |

   The RV hypertrophy (1.72×) is the expected PAH response. The geometric
   septum grows proportionally to the LV (0.90 vs 0.89), which is
   biomechanically plausible — the septum is LV-dominated. The LDRB septum
   grows less than anything else (0.83), which doesn't match any plausible
   tissue response to pressure overload.

   **Caveat**: this compares two different patients, not the same patient
   longitudinally. It's suggestive, not definitive.

## The prestress variance problem — CRITICAL DISCOVERY

While validating the sweep on 8 spectrum cases (all using the shared UKB
`geometry.bp`), we observed that the geometric septum cell count varied wildly
per case, while LDRB was essentially constant:

| Case            | RV_ESP (mmHg) | Geometric cells | LDRB cells |
|-----------------|---------------|-----------------|------------|
| healthy         | 27.7          | 164             | 455        |
| borderline      | 29.0          | 328             | 454        |
| mild            | 37.7          | 330             | 455        |
| moderate        | 51.2          | 356             | 455        |
| severe          | 67.3          | 372             | 452        |
| moderate_severe | 69.5          | 361             | 453        |
| very_severe     | 81.9          | 373             | 453        |
| end_stage       | 82.9          | 263             | 454        |
| **std**         |               | **67.3**        | **1.1**    |

All 8 cases use the same shared UKB mesh. The variance comes from **prestress**:
each case applies different end-diastolic pressure targets, which backward-
displaces the reference mesh differently. `compute_per_cell.py` loads from
`checkpoint.bp` (the prestressed reference), so the Euclidean distances
`d_LV`, `d_RV`, `d_epi` are computed on slightly deformed meshes. Small
coordinate changes → direct linear change in distances → different cells
satisfy `max(d_LV, d_RV) < d_epi` → very different cell counts.

**LDRB is invariant** because Laplace solutions with Dirichlet BCs are
smoothing operators: the equipotential lines barely move when the mesh is
slightly perturbed.

### What this means

The inter-case Pearson r computations we have been running for the geometric
septum are partially comparing apples to oranges: for each case we sum work
over a different cell set. Some of the r = 0.961 signal is noise from tagging
drift, not a real measurement of proxy accuracy.

We have two kinds of septum variability, and they have opposite robustness
properties:

| Variability type         | What varies                   | Geometric | LDRB         |
|--------------------------|-------------------------------|-----------|--------------|
| **Inter-mesh** (different hearts) | anatomy / shape       | stable (0.4pp) | variable (1.7pp) |
| **Intra-mesh** (same mesh, different prestress) | small deformation | unstable (std 67) | essentially constant (std 1) |

Neither definition is uniformly better. The Euclidean approach handles shape
differences well but is fragile to small deformations. The Laplace approach is
the opposite.

## Design decision: septum as an anatomical concept

We resolved the ambiguity by taking a principled stance: **the septum is an
anatomical region, not a mechanical state**. A cell is "septum" because of its
location in the tissue, not because of how stretched it is at any moment.
Echocardiographers do not say "this cell became septal at end-systole".

Consequences:

- Tagging is computed **once on a canonical reference geometry**, and every
  case in a spectrum inherits the same cell tag arrays.
- The canonical reference for the UKB spectrum is the **zero-strain shared
  mesh** (`data/shared_ukb_mesh/ukb/geometry/geometry.bp`), loaded before any
  prestress.
- Work integration remains per-case on the actual deformed mesh at each
  timestep. Only the *membership* of the septum set is fixed, not the
  deformation, strain, stress, or volume element.

This kills the intra-case variance entirely by construction. The inter-case
Pearson r becomes a clean measurement of how the proxies track disease across
a shared anatomical region.

### Why zero-strain reference is defensible

The zero-strain mesh is the only reference guaranteed to be consistent across
cases in a spectrum. Alternatives (end-diastolic per case, mid-cycle, etc.)
all introduce case-dependent variance in the tagging. The zero-strain state
does not correspond to any in-vivo configuration, but it is the neutral
reference — the atlas. Using it for tagging is analogous to drawing AHA
segments on an anatomical template and applying them to all patients.

**We are not lying about physics.** The work density is still computed on the
case-specific deformed mesh at each timestep. We are just saying "the set of
cells we sum over is a fixed anatomical region". That is what clinicians do
with segment-based analysis.

## Sweep design after the discovery

### Corrected envelope

The envelope is now minimal: only `d_sum ≤ 22 mm` (to prevent the sweep from
reaching free-wall cells at large t) and `~touches_epi` (topological exclusion
of cells with any epi-face, which guarantees epi-free septum). We removed
`d_epi_min` and `d_sum_min` because they excluded legitimate apical septum
cells and broke the `sweep(t=0) == geometric` invariant. See CHECK 6 in the
comprehensive eval — it now passes at 100% coverage.

### Sweep parameterization (unchanged in form, changed in interpretation)

The entry_t sweep is retained:

```
entry_t(cell) = max(d_LV, d_RV) - d_epi   (all distances in mm on reference mesh)
septum(t)     = envelope ∩ {entry_t < t}
```

But with the canonical-reference decision, `d_LV`, `d_RV`, `d_epi`, and
therefore `entry_t` are computed **once** on the zero-strain shared mesh. All 8
spectrum cases receive identical sweep masks at every t. The sweep is now a
property of the reference geometry, not of each case.

- t = 0 exactly recovers `max(d_LV, d_RV) < d_epi` (geometric septum)
- t < 0 selects the deeper core
- t > 0 adds cells toward the epicardium

### Why keep Euclidean and not switch to Laplace-based sweep

Euclidean `entry_t` is simple, monotonic (in the sense that sweep(t1) ⊆
sweep(t2) for t1 < t2), and one-dimensional. On the canonical reference mesh
(no prestress), the only residual weakness is the cosmetic non-uniform bin
density that Cobiveco flagged — but that weakness is tolerable here because
we're aggregating work over the whole sweep mask, not resolving it per tau
slice.

Laplace-based sweeps (e.g., sweeping an epi_scalar or lv_rv_scalar threshold)
would also be stable, but inherit the curvature-bowing artifact LDRB suffers
from: they systematically undertag curved septum regions (the bulge toward the
LV in PAH-shaped hearts) and systematically overtag at the basal lip. Pure
Euclidean on a canonical reference avoids both by construction.

#### The critical argument: growth direction (added 2026-04-14)

After actually implementing a Laplace-based sweep scalar

    entry_Lap(c) = max(u_epi(c), 2 * |u_LVRV(c) - 0.5|)

and comparing it directly against the Euclidean `entry_t` in ParaView on the
UKB mesh (`verify_sweep_envelope.py` now writes both to the same XDMF), the
**real** reason to prefer Euclidean became clear, and it is not about
smoothness or bowing artifacts. It is about the **direction the sweep grows
in as the threshold expands**. The two methods grow in topologically
orthogonal directions, and only one of them corresponds to the scientific
question this thesis asks.

**What the sweep is supposed to do.** The proxy-validation question is:
*"At what lateral extent of the septum does each pressure proxy stop
tracking the true work?"* Every step of the sweep should add or remove cells
at the **lateral boundary** (toward the anterior/posterior septal
junctions), while the cells that remain should always span the **full wall
thickness** from LV endo to RV endo. The sweep is asking how wide a septum
you can draw, not how deep inside the wall you reach. "Transmural integrity"
is a precondition, not something the sweep should be varying.

**What Euclidean `entry_t` does.** `max(d_LV, d_RV) - d_epi` grows the sweep
laterally:

- The core (t very negative) is cells with both `d_LV` and `d_RV` small
  **and** `d_epi` large — i.e. cells that fully span the LV-to-RV wall and
  sit deep away from the epicardium. These are full-thickness cells at the
  narrow middle of the septum.
- As t increases, the admission rule relaxes `max(d_LV, d_RV) < d_epi + t`.
  New cells enter where the *sideways margin* to the epicardium shrinks — at
  the anterior/posterior junctions where `d_epi` drops. The sweep slides
  outward toward the junctions.
- Every cell ever admitted has small `d_LV` AND small `d_RV`, so transmural
  integrity is maintained throughout the sweep. The sweep grows
  **septum → free wall**, laterally.

**What Laplace `entry_Lap` does.** `max(u_epi, 2|u_LVRV - 0.5|)` grows the
sweep transmurally:

- Including `u_epi` in the max means "a cell is in the core only if it is
  deep inside the wall" — but `u_epi` is specifically a **transmural**
  depth field (0 on both endos, 1 on epi), so this locks the core into the
  mid-wall layer, not the narrow central septum.
- Including `2|u_LVRV - 0.5|` pins the core to the LV-RV bisector, a
  thin sheet running through the middle of the septum along the transmural
  direction.
- The intersection is a *slab* running roughly parallel to the endos,
  sitting in the middle of the wall. As s grows, the slab inflates outward
  **toward the endos**, not outward toward the junctions.
- At low s the sweep does not span LV to RV at all — it is a ribbon in the
  mid-wall that does not touch either endocardium. Transmural integrity is
  violated for most of the sweep range.
- The sweep grows **inside → outside**, transmurally.

**This is a property of the equation, not a bad parameterization.** You
cannot fix it by choosing different thresholds, because `u_epi` is itself a
transmural coordinate by construction. Removing `u_epi` from the max
destroys the sweep's ability to distinguish septum from free wall (both have
cells near the LV-RV bisector). The only way to get a Laplace sweep that
grows laterally would be to solve a new Laplace problem with Dirichlet BCs
on the anterior and posterior *septal junctions* — but you do not know where
those are until after you have tagged the septum, which is what the sweep
is for. Catch-22.

Meanwhile, the Euclidean sweep gets lateral growth for free, because
subtracting `d_epi` creates a "margin from the outer surface" that shrinks
toward the junctions by pure geometry — no PDE needed.

**How this shows up in the cell-count plateau.** The 198-cell jump between
s=0.5 and s=0.6 (~49% of the entire Laplace sweep entering in one step) is
the transmural slab *reaching the endocardia simultaneously* on both LV and
RV sides. Before s=0.5 the slab is still mid-wall; at s≈0.55 it touches
both endos at once across a large area of the septum. This is not a
numerical artifact — it is the Laplace sweep fulfilling its growth
direction, which happens to be the wrong direction for our question. The
Euclidean sweep has no equivalent jump because its growth is continuous
along the lateral surface of the wall, where there is no geometric
coincidence to trigger a sudden enrolment.

**One-sentence framing for the thesis.** *"The Euclidean sweep varies the
lateral extent of the septum while holding transmural integrity fixed,
which is exactly the quantity the proxy-validation question varies. The
Laplace sweep varies transmural depth, which is a different question — it
asks how deep into the wall the proxy is valid, not how wide the septum
can be drawn. The two sweeps are not two versions of the same test; they
test two different things, and only the Euclidean one matches the
proxy-validation framing."*

This argument subsumes the earlier bowing / non-linearity observations.
The 4× variation in per-step cell count and the 49% worst-step fraction
are symptoms; the underlying cause is that the Laplace sweep is measuring
and growing in the wrong dimension for this study.

### Ridge-bounded sweep — future work

Cobiveco (Schuler et al. 2021) bounds the septum using **ridge curves**: the
anterior and posterior septal junctions where the RV wall meets the LV wall
on the epicardium. Their pipeline solves a Laplace equation, extracts the u
= 0.5 isosurface as the "septal surface", then clips it to the region
bounded by the two ridge curves. Critically, their septum is a 2D surface,
not a 3D region — they never volumetrically tag cells as septum.

The ridge concept is anatomically strong: it identifies the physical
"corners" where the shared wall ends, independently of any distance field or
Laplace artifact. A ridge-bounded volumetric sweep would:

1. Detect anterior and posterior ridges on the epicardium of the canonical
   reference mesh (as geodesic curves of maximum curvature, or via an
   auxiliary Laplace solve localized to the junctions)
2. Define the "anatomical septum envelope" as cells whose epicardial
   projection falls between the two ridges
3. Sweep transmural depth (endocardium → epicardium) within this envelope,
   holding the lateral extent fixed by the ridges

This would give us a sweep that is invariant to both Laplace bowing and
Euclidean prestress drift (the latter is already handled by the canonical
reference). **This is a valuable methodological extension but is deferred as
future work.** Implementing ridge extraction is non-trivial (geodesic paths on
irregular surfaces, robust to mesh quality), and the current Euclidean
entry_t sweep on the canonical reference is sufficient for the proxy-ranking
result we need for the thesis.

### What Cobiveco does NOT give us

Reading Cobiveco carefully (see
`results/docs/coupled_circulation_justification.md` and the Cobiveco reading
in our session notes):

- Cobiveco's septum is a **surface**, not a volumetric region. Their u = 0.5
  isosurface has the same bowing artifact our LDRB definition has. They
  sidestep the problem by using the surface as a boundary rather than a
  volumetric tag, and by clipping it at the ridges.
- They do not address the Laplace bowing problem per se — they just use the
  Laplace field for coordinate generation, not for volumetric septum tagging.
- They do not address loading / prestress invariance (implicit zero-strain).
- They do not have a continuous sweep — the septum is a single binary region
  bounded by ridges.
- Their validation is comparative against UVC, not against any anatomical
  ground truth.

Cobiveco is genuinely useful for the ridge idea, but it is not a complete
solution to the septum-tagging problem. The ridge concept is the one piece
worth borrowing, and even that is deferred.

## Pressure source fix

We discovered that `compute_per_cell.py` was using the 0D circulation model's
pressure history (`circulation/history.npy → p_LV, p_RV`) instead of the FEM
Lagrange multiplier (`solver/solver_cavity_pressure_mmHg.npy`, formerly
`pressure_history.npy`). The two are the same signal offset by exactly 1
timestep due to staggered coupling (solver_pressure[i] == circ_pressure[i+1]
to machine precision). The numerical impact on the inter-case Pearson r is
negligible (<0.001 change in r), but the proxy should use the pressure that
physically loaded the tissue.

Fixed in `compute_per_cell.py`: now loads from `solver_cavity_pressure_mmHg.npy`
with a fallback to `pressure_history.npy` for backwards compatibility with
older runs. Also renamed the solver pressure file from `pressure_history.npy`
to `solver_cavity_pressure_mmHg.npy` in `complete_cycle.py` for self-
documentation. See `results/docs/coupled_circulation_justification.md` for the
full coupling analysis.

## Stage-based implementation roadmap

**Stage 1 — canonical reference mesh fix (immediate, required)**
- Modify `compute_per_cell.py` to load the reference mesh for distance
  computation from a canonical location (`data/shared_ukb_mesh/ukb/geometry/
  geometry.bp`) instead of the per-case `checkpoint.bp`.
- Keep all work / deformation / strain / stress computation on the per-case
  deformed mesh (from checkpoint.bp). Only `d_LV`, `d_RV`, `d_epi`, the
  geometric and LDRB tags, and `entry_t` come from the canonical reference.
- Verify that all 8 spectrum cases now have identical `n_geo`, `n_ldrb`, and
  identical sweep masks at every t.
- Rerun compute_per_cell.py on the 6-beat production sims once they finish.

**Stage 2 — scientific framing (thesis chapter)**
- Frame the contribution as: "For any reasonable anatomical tagging of the
  septum, P_LV is a more stable proxy for longitudinal septal work than
  P_Trans or P_RV, with r > 0.94 across a continuous sweep of definitions."
- Argue that no single septum definition is "correct", and that reporting
  proxy accuracy as a function of definition is more honest than committing
  to one.
- Include the three quantitative arguments (cross-mesh stability, regional
  growth, sweep stability) with appropriate caveats.
- Discuss ridge-based sweep as future work, citing Cobiveco.

**Stage 3 — cross-mesh validation (optional but strong)**
- Rerun the same analysis on patient healthy and PAH meshes.
- If the proxy ranking survives the mesh change, that is strong evidence the
  result is not an artifact of any specific geometry.
- Note that patient meshes have only single cases each (not a severity
  spectrum), so the inter-case Pearson r is not directly computable on them —
  but per-cell work profiles can still be compared visually.

**Stage 4 — ridge-bounded sweep (future, not thesis)**
- Implement anterior/posterior ridge extraction on the canonical reference.
- Build a ridge-bounded envelope that fixes the lateral septum extent
  anatomically.
- Sweep transmural depth within the ridge envelope.
- Compare proxy ranking to the Euclidean entry_t sweep result. If the
  ranking survives, that is the strongest possible methodological robustness
  claim.
- Contribution: a cleanly anatomical volumetric septum definition that does
  not depend on Laplace artifacts, Euclidean prestress drift, or ad-hoc
  distance thresholds.

## Honest reframing of the primary result

The old claim "the geometric definition is more stable than LDRB" is
**partially false**: geometric is more stable across different meshes but
less stable to prestress on the same mesh. With the canonical-reference fix,
this ambiguity is resolved: both definitions become stable because the tagging
is done once on a fixed geometry. The result for the thesis is:

> **Across a continuous family of anatomical septum definitions** (bounded by
> a tight core and a wide envelope, parameterized by a Euclidean threshold on
> depth below the geometric boundary), **P_LV is the most stable and highest-
> correlating proxy for longitudinal septal work in this mesh family**. The
> Trans = P_LV − P_RV proxy performs comparably only at the LDRB-specific cell
> set, which has known curvature artifacts and does not generalize to other
> reasonable definitions.

This is a stronger, more robust, and more honest claim than the previous
"geometric septum → PLV wins" or "LDRB septum → Trans wins" framings.

## Updated implementation status

### Fixed in this session
- [x] Removed d_epi_min and d_sum_min from envelope; verified 100% geometric coverage at sweep t<0 in new CHECK 6
- [x] compute_per_cell.py now loads solver cavity pressure instead of 0D ODE pressure (solver_cavity_pressure_mmHg.npy, with backwards-compat fallback)
- [x] Renamed pressure_history.npy → solver_cavity_pressure_mmHg.npy in complete_cycle.py for self-documentation
- [x] analyze_sweep.py reconstructs envelope from raw fields (backwards compatible with old per_cell_data.npz)
- [x] Cross-mesh septum stability computed on 3 meshes (UKB, healthy patient, PAH patient)
- [x] Regional growth analysis computed on healthy/PAH patient meshes
- [x] Sweep sensitivity plotted with correct envelope and reference points
- [x] Proxy stability argument quantified: P_LV range 0.052, P_Trans range 0.416 across the sweep
- [x] Work breakdown figures regenerated for presentation (LV/RV/Septum with dual axes and correlation table)
- [x] PV loops plotted (ODE and solver — validated <1% agreement, 1-step lag)
- [x] Coupled circulation justification doc written (`results/docs/coupled_circulation_justification.md`)
- [x] Septum definition quantitative arguments doc written (`results/docs/septum_definition_quantitative_arguments.md`)
- [x] 8 six-beat production sims submitted to habanaq with 8h walltime, jobs 1020849–1020856 (started running 2026-04-12 ~16:00)

### Pending for Stage 1 completion
- [ ] Modify `compute_per_cell.py` to load distance reference from canonical `data/shared_ukb_mesh/ukb/geometry/geometry.bp`
- [ ] Verify per-case cell counts become identical after the fix (std → 0 by construction)
- [ ] Rerun `compute_per_cell.py` on the 6-beat outputs once jobs 1020849–1020856 finish
- [ ] Rerun `analyze_sweep.py` on the updated per_cell_data.npz files
- [ ] Verify the proxy stability argument still holds (P_LV range ~0.05, P_Trans range ~0.4) on the converged 6-beat data
- [ ] Regenerate presentation plots with the canonical-reference, 6-beat, corrected-pressure data

### Pending for Stage 2 (thesis framing)
- [ ] Draft the "septum definition" methods subsection with the anatomical framing
- [ ] Draft the proxy stability results subsection, citing the sweep analysis
- [ ] Add appropriate caveats (cross-mesh comparison is suggestive not longitudinal; Laplace artifacts vs Euclidean prestress drift; ridge-based refinement as future work)

### Pending for Stage 3 (cross-mesh validation, optional but strong)
- [ ] Run compute_per_cell.py on a patient healthy case and a patient PAH case
- [ ] Compare proxy rankings across UKB spectrum vs individual patient meshes
- [ ] If the ranking survives, add as a robustness result in the thesis

### Deferred to future work (Stage 4+)
- [ ] Ridge-based anatomical sweep (Cobiveco-inspired)
- [ ] Mesh convergence study (the 2153-cell UKB mesh may be too coarse)
- [ ] Patient-specific longitudinal study (same patient, multiple timepoints)
- [ ] Mechanics Question B (tau-band sweep, mechanistic story rather than clinical proxy)
