# Implementation Worksheet: Transventricular Profile Analysis

**Created**: 2026-04-08
**Goal**: Implement per-cell work computation, transventricular profiles, and sensitivity curves.
**Context**: 6/8 disease spectrum sims complete (moderate running, healthy pending). Core analysis code NOT yet implemented.

---

## Current State

### What EXISTS:
- `metrics_calculator.py`: computes **regional** work integrals (LV/RV/Septum/Whole totals). Uses volume-integrated UFL forms with integer subdomain markers. ~900 lines, well-tested, energy-balanced.
- `postprocess_metrics.py`: replays displacement checkpoints, calls MetricsCalculator, saves `metrics_downsample_*.npy`. Has cKDTree septum re-tagging code (d_lv, d_rv, d_epi computed but NOT saved).
- `explore_tau.py` / `explore_tau_v2.py`: standalone exploration scripts. Compute Euclidean and Laplace tau, define study regions, export XDMF. NOT connected to the metrics pipeline.
- 6 completed sims with full regional metrics, perfect energy balance.

### What DOES NOT exist:
- Per-cell work arrays (w_true[cell], w_proxy[cell], etc.)
- DG0 test function per-cell integration in metrics_calculator.py
- Tau coordinate in the metrics output
- Study region definition (geometric + LDRB union) in the pipeline
- `analyze_transventricular.py` script
- Sensitivity curve computation
- Any transventricular profile figures

---

## Implementation Phases

### Phase A: Per-Cell Work in MetricsCalculator
**What**: Add DG0 test function assembly to `metrics_calculator.py` so that at each timestep, we get work per cell, not just per region.
**Why**: The regional integrals (work_true_LV, etc.) are sums over all cells in a region. We need the individual cell contributions to build profiles.

**The technique** (from design doc, validated approach):
```python
# Instead of:  W_region = assemble_scalar(form(S:dE * dx(region_tag)))
# We do:       W_per_cell = assemble_vector(form(S:dE * v_dg0 * dx))
# where v_dg0 is a DG0 TestFunction. This gives W_per_cell[i] = integral over cell i.
```

Key point: this uses the SAME quadrature points as the existing regional integration. No projection, no accuracy loss. The DG0 test function acts as a spatial partition of unity — sum of all cell contributions equals the regional total exactly.

**What to accumulate per cell (trapezoidal rule, same as existing regional):**
- `w_true[cell]`: total internal work density (S:dE)
- `w_ff[cell]`, `w_ss[cell]`, `w_nn[cell]`, `w_cross[cell]`: directional decomposition
- `w_proxy_PLV_ff[cell]`: P_LV(t) * d_eps_ff per cell
- `w_proxy_PRV_ff[cell]`: P_RV(t) * d_eps_ff per cell  
- `w_proxy_Trans_ff[cell]`: (P_LV - P_RV)(t) * d_eps_ff per cell
- Same 3 for eps_ll (longitudinal strain)

That's 4 + 6 = 10 per-cell time-integrated quantities. At ~2000 cells, storage is trivial.

**Files to modify:**
- `metrics_calculator.py`: add DG0 function space, test function, per-cell forms, per-cell accumulation arrays, per-cell output
- `postprocess_metrics.py`: save per-cell arrays alongside regional metrics

**Validation (MUST PASS before proceeding):**
- `sum(w_true_per_cell)` == `work_true_Whole` to machine precision
- `sum(w_true_per_cell[LV_cells])` == `work_true_LV` (exact match)
- Same for RV, Septum, and proxy work

---

### Phase B: Tau + Study Region in PostProcessing
**What**: Compute Euclidean tau and study region definition during postprocessing, save with per-cell results.
**Why**: The profile analysis needs to know each cell's transventricular position and whether it's in the study region.

**What to compute (once per mesh, not per timestep):**
- d_lv, d_rv, d_epi via cKDTree (already done in postprocess_metrics.py for septum re-tagging — just need to SAVE these)
- tau = d_lv / (d_lv + d_rv) (Euclidean, decided over Laplace after empirical comparison)
- d_sum = d_lv + d_rv
- is_geometric_septum = max(d_lv, d_rv) < d_epi
- is_ldrb_septum = (epi_scalar <= 0.5) AND (0.1 < lv_rv_scalar < 0.9) — requires solving 2 Laplace equations (code exists in explore_tau_v2.py)
- study_region = (is_geometric_septum | is_ldrb_septum) AND (d_sum < D_SUM_MAX)

**Output**: save as .npz file alongside metrics:
```
per_cell_data.npz:
  tau, d_lv, d_rv, d_epi, d_sum,
  is_geometric_septum, is_ldrb_septum, study_region,
  w_true, w_ff, w_ss, w_nn, w_cross,
  w_proxy_PLV_ff, w_proxy_PRV_ff, w_proxy_Trans_ff,
  w_proxy_PLV_ll, w_proxy_PRV_ll, w_proxy_Trans_ll,
  cell_volumes
```

**Files to modify:**
- `postprocess_metrics.py`: add tau/study region computation, save per_cell_data.npz
- OR: create a new lightweight script `compute_per_cell.py` that runs AFTER existing postprocessing, reading checkpoint + existing metrics, computing per-cell work independently. This avoids touching the validated postprocess_metrics.py.

**Decision needed**: modify postprocess_metrics.py or create separate script?
- Pro separate: doesn't risk breaking existing pipeline, can iterate faster
- Pro modify: single pipeline, guaranteed same forms and quadrature
- **Recommendation**: separate script initially, merge into pipeline once validated

---

### Phase C: Transventricular Profile Analysis Script
**What**: New script `analyze_transventricular.py` that loads per_cell_data.npz and produces profiles + sensitivity curves.
**Why**: This is the actual thesis analysis — the figures that answer "which pressure proxy works for the septum?"

**What it produces:**

1. **Transventricular profiles** (Layer 1 in design doc):
   - Bin study-region cells by tau (10-15 bins)
   - In each bin: volume-weighted mean of w_true, w_proxy_PLV, w_proxy_Trans, etc.
   - Plot w(tau) for each severity level — shows where proxies track true work
   - Plot R(tau) = w_proxy/w_true ratio — shows proxy accuracy across septum

2. **Sensitivity curve** (Layer 2):
   - Sweep septum boundary from geometric (tight) to LDRB union (wide)
   - At each boundary: compute integrated proxy ratio
   - Mark reference points (geometric def, LDRB def, TriSeg volume)
   - This directly explains why different septum definitions give different answers

3. **Clinical summary** (Layer 3):
   - "P_trans captures X +/- Y% of septal work" where Y = segmentation uncertainty from sensitivity curve

**Input**: per_cell_data.npz from each sim
**Output**: figures (PDF/PNG) + summary table (JSON/CSV)

---

### Phase D: Run Analysis on All Cases  
**What**: Rerun per-cell postprocessing on all completed sims, generate profiles.
**Why**: We have the data, we just need to run the new code on it.

**Steps:**
1. Run compute_per_cell.py on all 6 (then 7, then 8) completed sims
2. Validate energy balance on each
3. Run analyze_transventricular.py to generate profiles
4. Compare across disease severity spectrum
5. Generate thesis-ready figures

---

### Phase E: Thickness Variants (collaborator request)
**What**: Mesh 5 thickness variant STL surfaces, run sims with severe circ params, analyze.
**Why**: Shows how RV wall thickening (PAH hallmark) affects work distribution and proxy accuracy.

**Steps:**
1. Mesh the 5 STL surface sets (baseline, thinned, thick_1mm, thick_2mm, extra_thick) via geometry_generator.py
2. Run 5 sims (6 beats each, severe circ params)
3. Run per-cell postprocessing on each
4. Compare transventricular profiles across thickness variants

---

## Progress Tracking

### Phase A: Per-Cell Work in MetricsCalculator
- [ ] Design per-cell forms (S:dE * v_dg0 * dx, P * deps_ff * v_dg0 * dx)
- [ ] Implement in metrics_calculator.py or new compute_per_cell.py
- [ ] Test on one sim (borderline)
- [ ] Validate: sum(per_cell) == regional total (must be exact)
- [ ] Validate: per-cell directional decomposition sums to per-cell total

### Phase B: Tau + Study Region
- [ ] Port tau computation from explore_tau_v2.py into pipeline
- [ ] Port Laplace solve (epi_scalar, lv_rv_scalar) for LDRB septum definition
- [ ] Compute study region = geometric | LDRB, with d_sum safety filter
- [ ] Save per_cell_data.npz
- [ ] Visual validation: export tau + study_region to XDMF, check in ParaView

### Phase C: Analysis Script
- [ ] Create analyze_transventricular.py
- [ ] Implement binning by tau (volume-weighted)
- [ ] Implement transventricular profile plots (w vs tau, R vs tau)
- [ ] Implement sensitivity curve (proxy ratio vs septum definition width)
- [ ] Mark reference points on sensitivity curve
- [ ] Test on one case, then run on full spectrum

### Phase D: Full Spectrum Analysis
- [ ] Run per-cell postprocessing on all completed sims
- [ ] Generate spectrum comparison figures
- [ ] Verify profiles make physical sense (proxy ratio should differ LV-side vs RV-side)
- [ ] Write thesis section

### Phase E: Thickness Variants
- [ ] Mesh 5 thickness variant surfaces
- [ ] Submit 5 sims (severe circ params, 6 beats)
- [ ] Run per-cell postprocessing
- [ ] Compare profiles across thickness

---

## Key Design Decisions (settled)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tau coordinate | Euclidean d_LV/(d_LV+d_RV) | Monotonic, no Laplace bowing artifact (empirically verified) |
| Study region lower bound | Geometric: max(d_LV,d_RV) < d_epi | 318 cells, conservative, no artifacts |
| Study region upper bound | LDRB: epi_scalar<=0.5 AND 0.1<lv_rv<0.9 | 436 cells, standard approach, 37% wider |
| Per-cell integration | DG0 test function assembly | Same quadrature as regional, no projection loss |
| Number of beats | 6 | <0.5% beat-to-beat convergence verified |
| Disease ordering | By achieved RV_ESP, not label | moderate_severe/severe overlap at ~68-69 mmHg |

## Sims Status

| Severity | RV_ESP target | RV_ESP achieved | Job | Status |
|----------|--------------|-----------------|-----|--------|
| healthy | 25 | — | — | Waiting for v9 circ param |
| borderline | 30 | 33.6 | 1017000 | DONE |
| mild | 38 | 39.6 | 1017001 | DONE |
| moderate | 55 | — | 1017231 | RUNNING |
| moderate_severe | 63 | 69.1 | 1017002 | DONE |
| severe | 72 | 67.9 | 1017003 | DONE |
| very_severe | 85 | 82.8 | 1017004 | DONE |
| end_stage | 95 | 84.0 | 1017005 | DONE |
| mild (fine mesh) | 38 | — | 1017233 | RUNNING (mesh convergence test) |
