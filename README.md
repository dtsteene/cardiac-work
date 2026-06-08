# cardiac-work

Biventricular finite-element simulations evaluating clinical
pressure-strain proxies against the ground-truth tensor work
`W = ∫ S:dE dV`, with emphasis on the interventricular septum in
pulmonary arterial hypertension (PAH). The central question is
**which pressure (P_LV, P_RV, or transmural P_LV − P_RV) best tracks
septal work as RV pressure rises?**

This is the simulation code for the MSc thesis
*Pressure Proxies for Biventricular Myocardial Work: A Finite
Element Study* (Daniel Steeneveldt, University of Oslo). The thesis
itself lives at [github.com/dtsteene/RV](https://github.com/dtsteene/RV).

## Layout

```
cardiac-work/
├── README.md
├── tests/                              unit / smoke tests
├── data/                               pre-tuned 0D circulation JSONs + reference meshes (gitignored)
├── sbatch/
│   ├── jobs/                           atomic SLURM job templates (one sim or one analysis)
│   └── sweeps/                         multi-job launchers (submit dozens of jobs for a sweep)
│
│   ── pipeline (FEM + 0D solver + postprocessing) ──
├── complete_cycle.py                   coupled FEM + 0D Windkessel simulation driver
├── compute_per_cell.py                 replays checkpoint → per-cell stress / strain / proxy fields
├── postprocess_metrics.py              per-cell → regional metrics (LV / RV / septum)
├── metrics_calculator.py               stress / strain / work decomposition library
├── geometry_generator.py               UKB mesh + LDRB fibers + region tags + per-cell geometric fields
├── clinical_frame.py                   clinical-frame direction helpers (longitudinal projection)
├── geometry_utils.py                   pure point-to-surface geometry primitives (shared, no FEniCSx)
│
│   ── analysis (read precomputed metrics; pure NumPy, no FEniCSx) ──
├── analysis_core.py                    THE stats core: per-region work, swappable-pressure proxy,
│                                       ratios, correlations, log-MAE (see tests/test_analysis_core.py)
├── sweep_analysis.py                   cross-sim sweep: proxy↔truth correlation + ratio across sims
├── run_postprocessing.py               beat-slice orchestrator (calls eval_proxies + plot_loops)
├── eval_proxies.py                     proxy R² / error tables per sim
├── plot_loops.py                       PV / PS / SS debug loops
├── plot_utils.py                       shared matplotlib utilities
│
│   ── ParaView field exporters ──
├── export_static_geometry_tags.py      ch. 2 mesh / fiber / tag panels
├── export_unloading_cap_paraview.py    fig_unloaded_cap_grid, fig_ed_cap_grid
└── export_production_sweep_for_animation.py   time-series field export for animations
```

## Pipeline

Simulation and analysis are decoupled — change a metric, rerun
postprocessing, no re-simulation needed.

```
complete_cycle.py  →  compute_per_cell.py  →  postprocess_metrics.py  →  analysis_core.py
  (FEM + 0D solver)    (per-cell work/proxies)   (regional metrics)        (stats: ratios,
                                                                            correlations, proxy)
```

## Analysis core

`analysis_core.py` is the one home for the per-region statistics the
project cares about. It is pure NumPy/SciPy (no FEniCSx), reads the
precomputed `per_cell_data.npz` / `metrics_downsample_*.npy`, and
provides:

- `region_masks`, `region_density` — ground-truth S:dE work per LV / RV / septum
- `pressure_candidates` — the swappable septal pressure (P_LV, P_RV,
  transmural P_LV−P_RV, mean, nearest-side, tau-weighted)
- `pearson_r`, `correlation_stats` — proxy↔truth correlation (r, R², slope)
- `ratio_preservation`, `log_mae` — proxy-vs-truth ratio error

These definitions are lifted verbatim from the thesis sweep harness and
pinned by `tests/test_analysis_core.py` (run on the login node —
pure NumPy, no FEniCSx needed). Every analysis script delegates here.

## Reproducing thesis figures

The standalone scripts that produced the (now frozen) thesis figures
were removed after the defense to slim the working set; their reusable
math lives in `analysis_core.py`. The original figure producers remain
in git history — recover any of them with:

```bash
git show a0af112:freewall_ratio_proxy_test.py        # view
git checkout a0af112 -- analyze_h5_septum_mechanism.py  # restore
```

(`a0af112` is the last commit before the post-defense cleanup; see its
README for the full figure→script map.) Chapter 1, 2, and 4 pure
matplotlib illustrations are generated from the thesis repo
(`RV/scripts/`), not here.

## Running on SLURM

All heavy compute goes through SLURM — never the login node. Job
templates live in [`sbatch/jobs/`](sbatch/jobs/); multi-job launchers
that submit a whole sweep live in [`sbatch/sweeps/`](sbatch/sweeps/).

### Single jobs

```bash
# Single sim + postprocessing
sbatch sbatch/jobs/run_sim_and_post.sbatch

# Postprocess only (replays from a saved checkpoint)
sbatch --export=RESULTS_DIR=results/sims/<dir> sbatch/jobs/run_postprocess_only.sbatch

# Per-cell work (ED tagging, then canonical reference-config tagging)
sbatch --export=RESULTS_DIR=results/sims/<dir> sbatch/jobs/run_per_cell.sbatch
sbatch --export=RESULTS_DIR=results/sims/<dir> sbatch/jobs/run_per_cell_canonical.sbatch

# Cross-simulation sweep analysis (proxy↔truth correlation + ratio tables)
sbatch sbatch/jobs/run_sweep_analysis.sbatch

# Postprocess recovery if outputs are missing for a finished sim
sbatch --export=RESULTS_DIR=results/sims/<dir> sbatch/jobs/run_repost_if_missing.sbatch

# Mesh-convergence geometry generation (one shared UKB mesh per level)
sbatch sbatch/jobs/run_mesh_convergence_geometry.sbatch
```

### Sweeps

```bash
# Capped RV-EDP shared-L5 production sweep (16 cases — the main thesis sweep)
bash sbatch/sweeps/submit_capped_shared_l5_sweep.sh

# Shared-unloaded-reference sweep (Design B — only the 0D circulation JSON varies)
bash sbatch/sweeps/submit_shared_unloaded_l5_sweep.sh

# v12 EXP-extra spectrum dispatcher (uses pre-tuned 0D JSONs in data/)
sbatch sbatch/sweeps/submit_spectrum_v12_exp_extra.sbatch

# Patient-mesh FEM sweep
sbatch sbatch/sweeps/submit_patient_mesh_fem.sbatch

# Unloading-only sensitivity sweeps (appendix C reference-state)
bash sbatch/sweeps/submit_unloading_ab.sh
bash sbatch/sweeps/submit_unloading_stiffness_sweep.sh
```

## Geometry

[`geometry_generator.py`](geometry_generator.py) writes the mesh,
fibers, region tags, and per-cell geometric fields (distances,
Laplace scalars, septum masks) in a single command:

```bash
python3 geometry_generator.py --single ukb -c 5 --output-dir data/shared_ukb_mesh
```

Outputs: `geometry.bp` (solver), `geometry_fields.npz` (per-cell
arrays), `geometry_fields.xdmf` (ParaView).

## Dependencies

- [FEniCSx](https://fenicsproject.org) (`dolfinx`, `ufl`, `basix`,
  `ffcx`) for the finite-element kernel.
- [fenicsx-pulse](https://github.com/finsberg/fenicsx-pulse) for the
  cardiac constitutive models (Holzapfel-Ogden passive,
  active-stress contraction).
- [cardiac-geometries](https://github.com/ComputationalPhysiology/cardiac-geometries),
  [fenicsx-ldrb](https://github.com/finsberg/fenicsx-ldrb) for
  meshing and fiber generation.
- [circulation](https://github.com/ComputationalPhysiology/circulation) for the
  closed-loop 0D Windkessel model.
- [adios4dolfinx](https://github.com/jorgensd/adios4dolfinx) for
  checkpoint I/O, GMSH for meshing.

Install via the upstream FEniCSx conda channel; this repo carries
no pinned environment file. Pre-tuned 0D circulation JSONs are
checked in under `data/ukb_circ_v12_exp/` and
`data/patient_mesh_circ_v12_exp/`, so Optuna is not required to
reproduce any of the thesis sweeps.

## Data

Raw simulation outputs (~2 TB of checkpoints, per-beat metrics, and
ParaView bundles) are not in the repository. The full 16-case
production sweep is reproducible from
[`sbatch/sweeps/submit_capped_shared_l5_sweep.sh`](sbatch/sweeps/submit_capped_shared_l5_sweep.sh)
— it lists every case name and the exact env-var settings used.

## Tests

```bash
python3 tests/test_syntax.py                          # file-level syntax sanity
python3 tests/test_canonical_tagging.py <sim_dirs>    # invariants across a spectrum
```
