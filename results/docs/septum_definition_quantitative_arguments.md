# Septum Definition: Quantitative Arguments

Three quantitative arguments for preferring the geometric septum definition over
the LDRB Laplace-based definition, and for using a continuous sweep rather than
committing to either definition.

## Argument 1: Cross-mesh stability (coherence)

The geometric fraction of the total myocardium is remarkably stable across
three different meshes:

| Mesh            | n_cells | Geometric %  | LDRB %  | Jaccard(geo, LDRB) |
|-----------------|---------|--------------|---------|---------------------|
| UKB synthetic   | 2153    | 15.6%        | 21.1%   | 0.731               |
| Patient healthy | 4742    | 15.6%        | 20.6%   | 0.747               |
| Patient PAH     | 7686    | 14.7%        | 17.2%   | **0.619**           |
| **mean ± std**  |         | 15.3 ± 0.4%  | 19.6 ± 1.7% |                  |

Two observations:

1. The **geometric fraction varies by only 0.4 percentage points across three very
   different meshes** (two synthetic-healthy, one pathological). The LDRB
   fraction varies 4x more.

2. The **Jaccard similarity between geometric and LDRB drops to 0.619 on the PAH
   mesh** from ~0.74 on the other two. The LDRB definition becomes less coherent
   with the geometric definition precisely on the pathological geometry where it
   matters most.

This suggests the LDRB Laplace solution is sensitive to shape changes (RV dilation,
septal flattening) that the Euclidean distance definition handles naturally.

Plot: `cross_mesh_septum_stability.png`

## Argument 2: Regional growth proportionality

A well-behaved septum definition should grow in step with the myocardium it
anchors to. Comparing the same patient's healthy and PAH meshes:

| Region              | Healthy (mL) | PAH (mL) | Growth | Proportion to total |
|---------------------|--------------|----------|--------|----------------------|
| Total myocardium    | 127.5        | 164.0    | 1.29x  | 1.00 (reference)     |
| LV free wall        | 60.9         | 69.6     | 1.14x  | 0.89 (undergrows)    |
| RV free wall        | 32.7         | 56.2     | 1.72x  | **1.34 (overgrows)** |
| Septum (geometric)  | 23.1         | 26.6     | 1.15x  | **0.90 (tracks LV)** |
| Septum (LDRB)       | 34.7         | 37.0     | 1.06x  | 0.83                 |

The physiological expectation for PAH is clear: the RV hypertrophies under
pressure overload, growing disproportionately. The LV stays relatively stable.
The septum should follow the LV (it is anatomically continuous with both free
walls, but predominantly oriented along the LV circumferential fibers).

Observations:

- **RV grows 1.72x**, proportionality 1.34 — the expected PAH hypertrophy signature
- **LV grows 1.14x**, proportionality 0.89 — largely unchanged, diluted by RV growth
- **Geometric septum grows 1.15x, proportionality 0.90** — nearly identical to LV

  This is biomechanically plausible: the septum is part of the LV-dominated
  myocardium and should track it.

- **LDRB septum grows only 1.06x, proportionality 0.83** — undergrows even more
  than the LV

  This is biomechanically implausible: if the RV is ballooning out by 72% in
  volume, the septum interface should shift and bulge toward the LV, and the
  Laplace equipotentials should follow the deformed geometry. Instead, LDRB tags
  fewer cells in relative terms. This is a Laplace artifact of the shape change,
  not a real tissue response.

### Caveats

Comparing two separate patient meshes is NOT a true longitudinal measurement — we
are not tracking the same heart over time. The "healthy" and "PAH" meshes come
from different patients. The comparison is only meaningful insofar as both meshes
capture the characteristic shapes of healthy and pathological hearts. A proper
longitudinal measurement (same patient scanned twice) would strengthen this
argument considerably; we acknowledge this and report the cross-patient comparison
as suggestive rather than definitive. The UKB synthetic-healthy comparison
(Argument 1) is a cleaner test of cross-mesh consistency since both use the same
atlas.

Plot: `regional_growth.png`

## Argument 3: Proxy stability across the sweep

Rather than commit to a single definition, we sweep the septum width continuously
via `entry_t(c) = max(d_LV, d_RV) - d_epi`. The threshold `t` controls how far
the sweep extends beyond the geometric boundary:

    septum(t) = envelope ∩ {entry_t < t}

At `t = 0`, the sweep is exactly the geometric septum. At `t > 0`, cells toward
the epicardium are added. At `t < 0`, only the deepest core remains.

At each `t`, we compute the Pearson correlation between the true internal work
`∫ S:dE dV` and the three candidate pressure proxies (P_LV, P_RV, P_LV−P_RV), across
the 8 disease severity cases.

### Stability of each proxy

| Proxy    | min r    | max r   | range | stable? |
|----------|----------|---------|-------|---------|
| P_LV     | +0.944   | +0.996  | 0.052 | **yes** |
| P_RV     | +0.276   | +0.770  | 0.494 | no      |
| P_Trans  | +0.481   | +0.898  | 0.416 | no      |

Across t ∈ [−10, +15] mm:

- **P_LV stays between 0.94 and 0.99** — its rank as "best proxy" is independent
  of where you draw the septum boundary. The choice of definition simply doesn't
  affect P_LV's ability to track disease progression.

- **P_Trans swings from 0.48 to 0.90** — its performance depends strongly on
  exactly which cells you include.

### The LDRB-direct anomaly

At the cell count matching the direct LDRB definition (453 cells), the sweep
gives r_Trans ≈ 0.55–0.65. But direct LDRB gives r_Trans = 0.976. The high LDRB
correlation is NOT a property of "wider definitions" in general — it depends on
the specific cells that the Laplace solution happens to include.

Those specific cells sit at a higher mean tau (≈ 0.57 vs 0.50 for geometric) —
the LDRB definition is RV-shifted. When you include more RV-side cells, P_RV's
contribution to net loading is non-negligible, and P_Trans = P_LV − P_RV correctly
captures the net transmural driving pressure. But this shift comes from the
Laplace artifacts discussed in arguments 1 and 2 — it's not a physiologically
principled choice of where the septum ends.

### The honest conclusion

If we had to commit to a single septum definition, P_LV is the safest proxy: it
works well across every reasonable definition we tried. P_Trans only wins at the
specific (and artifact-prone) LDRB cell set.

But committing to a single definition is unnecessary. The sweep shows the proxy
performance as a function of definition width, so the reader can pick their
preferred definition and read off the correlation. This is more honest than
cherry-picking a definition that favors a particular proxy.

Plots:
- `sweep_definition_explainer.png` — the math and geometry of entry_t
- `sweep_sensitivity_clean.png` — r(t) curves with direct-definition reference points
- `sweep_proxy_stability.png` — stability comparison, showing P_LV is robust while P_Trans is not

## Summary for the supervisor presentation

The story has three quantitative pieces:

1. **Cross-mesh stability**: the geometric definition holds its fraction of total
   myocardium across three meshes to within 0.4 percentage points. LDRB varies 4x
   more and becomes less coherent with geometric on the PAH mesh (Jaccard 0.62).

2. **Biomechanical plausibility**: the geometric septum grows proportionally to
   the LV (0.90x total), which is physiologically expected. LDRB grows less than
   any other region (0.83x), which is implausible given the massive RV
   hypertrophy.

3. **Proxy robustness**: sweeping the septum width continuously shows that P_LV
   correlates with true work to r > 0.94 across all reasonable definitions, while
   P_Trans swings wildly. The apparent P_Trans win at the LDRB-direct point is an
   artifact of the specific RV-shifted cell set.

The sweep approach lets us make a claim that doesn't depend on an arbitrary
boundary choice: "P_LV is a reliable proxy for septal work in this mesh family
regardless of how you define the septum's edge."
