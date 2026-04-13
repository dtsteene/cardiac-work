# Postprocessing Architecture (2026-04-13 refactor)

## The split

The offline postprocessing pipeline is now two scripts with non-overlapping
responsibilities:

| Script | Owns | Runs by default |
|---|---|---|
| `compute_per_cell.py` | Per-cell and regional internal work (S:dE), transventricular coordinate, region masks | Yes |
| `postprocess_metrics.py` | Boundary work, Robin work, PV/PS/SS loop time-series, energy balance diagnostic | Yes, in slim mode |

Both are invoked by `run_sim_and_post.sbatch` after every simulation.

`compute_per_cell.py` is now the **canonical** source for any quantity of
the form "work integrated over a region of the mesh." Its output
(`per_cell_data.npz`) contains DG0-cell-wise work densities that can be
summed over any region: LDRB markers, study regions, AHA segments, or
masks you design after the fact. Regional sums from this file are
machine-precision equal to the scalar-form assembly of the same integral
(verified every invocation by the built-in cross-check).

`postprocess_metrics.py` is now responsible only for things
`compute_per_cell.py` cannot produce:

- Boundary work via Nanson's formula on the LV/RV cavity surfaces.
- Robin spring work on epicardium and base.
- PV, PS, and SS loop time-series (pressure/volume/strain at every
  timestep, not just an integral over the last beat).
- The energy-balance diagnostic that compares boundary work to the
  sum of internal work.

## The flags

`postprocess_metrics.py` exposes three `--skip-*` flags that use
`argparse.BooleanOptionalAction`, so every flag has a negative form:

```bash
# Production default: slim mode
python3 postprocess_metrics.py .

# Equivalent explicit form
python3 postprocess_metrics.py . \
    --skip-decomp --skip-research --skip-regional-internal

# Restore the legacy path (all three buckets computed in-place)
python3 postprocess_metrics.py . \
    --no-skip-decomp --no-skip-research --no-skip-regional-internal
```

Inside `MetricsCalculator`, the three flags map to constructor kwargs:

- `enable_regional_internal` — per-region `work_true_*`,
  `work_active/passive/comp_*`, and `work_ff/ss/nn/cross_*`. When off,
  `_calculate_incremental_work` early-returns `{}`, the matching
  `wd_*` forms are skipped at JIT time in `_precompile_forms`, and
  the `S_active/S_passive/S_comp` tensor-function interpolations in
  `_calculate_state_variables` are skipped too.

- `enable_decomp` — directional decomposition of mean state variables:
  `mean_S/E_ss/nn` and the full `mean_sigma_*` Cauchy stress projection.
  When off, the `S_ss/S_nn/E_ss/E_nn` and all Cauchy projection forms
  are skipped at JIT time, and the matching `sigma_total/active/passive/comp`
  tensor-function interpolations are skipped per timestep.

- `enable_research` — **reserved placeholder.** No research metrics
  currently live in `metrics_calculator.py`; fiber efficiency,
  dyssynchrony, and work redistribution are all computed downstream in
  `compare_cases.py` and `eval_proxies.py`. The gate is kept in the
  constructor signature as a stable landing spot for future migrations.

## What stays always-on

These are unconditional because compute_per_cell does not produce them
and they are part of the clinical proxy story the thesis relies on:

- `vol_*` regional volume forms.
- `S_ff_*`, `E_ff_*`, `S_ll_*`, `E_ll_*` — the PS-loop essentials.
  `mean_E_ff`, `mean_S_ff`, `mean_E_ll`, `mean_S_ll` per region drive
  the pressure-strain loops used in proxy validation.
- `robin_epi` / `robin_base` Robin work forms.
- `vol_cur_LV` / `vol_prev_LV` / `vol_cur_RV` / `vol_prev_RV` Nanson
  cavity volume forms used by `_calculate_boundary_work_exact`.
- `E_cur` and `S_total` tensor-function interpolations (needed by
  every path).
- `_u_prev` displacement tracking.

## When to enable the legacy path

Reach for `--no-skip-regional-internal` when:

- You are running an energy-balance cross-check between boundary work and
  sum-of-internal-work. The in-place version computes both from identical
  solver state.
- You are debugging a new proxy and want the legacy `work_ps_ff_*` and
  `work_true_*` columns in the same `metrics.npy` file for direct
  subtraction.

Reach for `--no-skip-decomp` when:

- You want `mean_sigma_*` or `mean_E_ss/nn` plotted alongside a PS loop
  for a figure.
- You are investigating a directional-work question and want to pipe
  `mean_S_ss_Septum(t)` into a loop plot.

Otherwise leave them off — `compute_per_cell.py` is faster and writes a
richer per-cell dataset.

## Contract with downstream consumers

Scripts that consume `metrics.npy`:

- Must NOT require `work_true_*`, `work_active/passive/comp_*`,
  `work_ff/ss/nn/cross_*`, `mean_S/E_ss/nn_*`, or `mean_sigma_*_*` to
  be present by default. If they need any of those, they should either
  (a) read from `per_cell_data.npz` instead, or (b) re-run
  `postprocess_metrics.py --no-skip-regional-internal` / `--no-skip-decomp`
  explicitly.

- Can rely on `mean_E_ff_*`, `mean_S_ff_*`, `mean_E_ll_*`, `mean_S_ll_*`,
  `V_LV_FEM`, `V_RV_FEM`, `p_LV`, `p_RV`, `work_boundary_exact_*`,
  `work_robin_epi/base`, `work_proxy_pv_*`, and `work_ps_ff/ll_*` being
  present.

## Why it's faster

When all three flags default to skip:

- ~40 UFL forms that used to compile at `MetricsCalculator.__init__`
  no longer compile (per-region × per-marker multiplication of the
  ss/nn projections, Cauchy projections, magnitudes, and work
  decomposition forms).
- Four tensor-function interpolations per timestep become zero
  (`sigma_total/active/passive/comp`), plus three more when
  regional_internal is also off (`S_active/S_passive/S_comp`). Tensor
  interpolation on the Quadrature element space dominates the
  replay-loop wall-clock in the old pipeline.
- The ~30 `assemble_scalar` calls per region per timestep that used to
  compute mean stresses / Cauchy projections / work decomposition
  disappear.

Expected speedup for the replay loop: 3–10x depending on mesh size and
number of timesteps. Actual numbers are in `results/bench/` once the
verification runs complete.

## Historical note

Before 2026-04-13, `metrics_calculator.py` computed everything above in
a single monolithic path. The motivation for the split was the
prestress-variance discovery and the `u_pre` permutation fix (see
`transmural_work_profiles.md`, session update 2026-04-12), which made
per-cell tagging tractable and let `compute_per_cell.py` take over the
regional-work responsibility with better provenance (identical cell
labels across a spectrum of cases, not just identical scalar totals).
