# cardiac-work

Biventricular FEM simulations that evaluate clinical pressure-strain
proxies against the ground-truth tensor work `W = ∫ S:dE dV`, with
emphasis on the interventricular septum in pulmonary arterial
hypertension (PAH). The central question: **which pressure (P_LV,
P_RV, or transmural P_LV−P_RV) best tracks septal work as disease
severity rises?**

## Context

- **Project overview + architecture notes:** [CLAUDE.md](CLAUDE.md) — always current, read this first.
- **Thesis manuscript (MyST Jupyter Book):** [`/home/dtsteene/D1/RV/`](/home/dtsteene/D1/RV/) — the written argument built from the simulations in this repo.
- **Side experiments parked on branches:**
  - `thickness` — wall-thickness sensitivity exploration.
  - `transmural` — transmural τ-coordinate analysis.

## Pipeline

Simulation and analysis are **decoupled**. Change a metric, rerun
postprocessing — no re-simulation needed.

```
complete_cycle.py  →  compute_per_cell.py  →  postprocess_metrics.py  →  plots / analyses
  (FEM + 0D solver)    (per-cell work/proxies)   (regional metrics)
```

### Story beats → entry-point scripts

**1–2. Single-sim simplification cascade** — S:dE down to P·dε_ff, and what happens to regional ratios.

- [`analyze_cascade.py`](analyze_cascade.py) — the cascade itself
- [`viz_regional_ratio.py`](viz_regional_ratio.py) — LV/RV work ratios under each pressure convention

**3. Spectrum tracking** — how well proxies track true work across PAH severity.

- [`analyze_spectrum.py`](analyze_spectrum.py) — spectrum-axis orchestrator (sweep, tracking, stats, angle)
- [`plot_work_density.py`](plot_work_density.py) — intensive work density per region
- [`plot_efficiency_spectrum.py`](plot_efficiency_spectrum.py) — η = W_true / W_proxy across severity
- [`viz_pv_loops_spectrum.py`](viz_pv_loops_spectrum.py) — PV loops grid
- [`viz_lvrv_tracking.py`](viz_lvrv_tracking.py) — binary tau split, LV/RV territories
- [`viz_best_proxy_per_cell.py`](viz_best_proxy_per_cell.py) — per-cell "which proxy wins" ParaView XDMF
- [`analyze_sweep.py`](analyze_sweep.py) — sensitivity of proxy correlations to septum-boundary choice
- [`table_septum_tagging_sensitivity.py`](table_septum_tagging_sensitivity.py) — cell-count / CV table for septum tagging

**Methods / convergence**

- [`analyze_per_beat.py`](analyze_per_beat.py) — per-beat convergence of proxy correlations
- [`analyze_convergence.py`](analyze_convergence.py) — beat-to-beat PV-loop + hemodynamic convergence
- [`eval_proxies.py`](eval_proxies.py) — quantitative R², error %

## Running a sim

All heavy compute goes through SLURM (never the login node).

```bash
# Single sim + postprocessing
sbatch run_sim_and_post.sbatch

# Postprocess only (replays from saved checkpoint)
sbatch --export=RESULTS_DIR=results/sims/<dir> run_postprocess_only.sbatch

# Spectrum sweep (7 severities on shared UKB mesh)
bash launch_phase1_shared_mesh.sh

# Per-cell work in canonical-tagging mode
sbatch --export=RESULTS_DIR=results/sims/<dir> run_per_cell_canonical.sbatch
```

## Geometry

[`geometry_generator.py`](geometry_generator.py) owns the mesh, fibers,
region tags, and the per-cell geometric fields (distances, Laplace
scalars, septum masks). One command writes everything:

- `geometry.bp` — solver artifact
- `geometry_fields.npz` — consumer-facing per-cell arrays
- `geometry_fields.xdmf` — ParaView-toggleable viz of all fields

```bash
python3 geometry_generator.py --single ukb -c 5 --output-dir data/shared_ukb_mesh
```

## Tests

```bash
python3 tests/test_syntax.py                # file-level syntax sanity
python3 tests/test_canonical_tagging.py <sim_dirs>   # invariants across a spectrum
```
