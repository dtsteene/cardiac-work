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
├── run_postprocessing.py               beat-slice orchestrator (calls eval_proxies + plot_loops)
├── eval_proxies.py                     proxy R² / error tables per sim
├── plot_loops.py                       PV / PS / SS debug loops
├── plot_utils.py                       shared matplotlib utilities
│
│   ── figure producers (one or more published thesis figures each) ──
├── figures_thesis_pre_post_coupling_and_ps_loops.py
├── generate_headline_figures.py
├── plot_h5_thesis_updates.py
├── analyze_h5_septum_mechanism.py
├── analyze_h5_sweep_core.py
├── septum_proxy_robustness_old_new.py
├── septum_mechanics_proxy_test.py
├── freewall_ratio_proxy_test.py
├── regional_ratio_waveform_test.py
├── analyze_cascade.py
├── plot_stress_magnitudes.py
├── audit_three_bugs_energy_budget.py   ─┐ together produce fig_closure_audit
├── robin_reference_replay.py            │
├── plot_three_bugs_audit.py            ─┘
│
│   ── ParaView field exporters ──
├── export_static_geometry_tags.py      ch. 2 mesh / fiber / tag panels
└── export_unloading_cap_paraview.py    fig_unloaded_cap_grid, fig_ed_cap_grid
```

## Pipeline

Simulation and analysis are decoupled — change a metric, rerun
postprocessing, no re-simulation needed.

```
complete_cycle.py  →  compute_per_cell.py  →  postprocess_metrics.py  →  figure scripts
  (FEM + 0D solver)    (per-cell work/proxies)   (regional metrics)
```

## Reproducing thesis figures

Every figure-producing script at the root is here because at least
one published figure depends on it. Most read saved metrics from a
results directory and need no FEniCSx itself — only `numpy`,
`matplotlib`, and the `.npz` / `.npy` outputs.

| Thesis figure(s)                                          | Script                                                                                                  |
|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| 4.2, 4.6, 4.7, 4.8, 4.9, 5.0c, 5.0d, `pv_standalone_vs_coupled` | [`figures_thesis_pre_post_coupling_and_ps_loops.py`](figures_thesis_pre_post_coupling_and_ps_loops.py)  |
| 5.0 freewall / septum headlines                           | [`generate_headline_figures.py`](generate_headline_figures.py)                                          |
| 5.0b, 5.1, 5.2, 5.2b, 5.2c, 5.3a, 5.4, 5.5                | [`plot_h5_thesis_updates.py`](plot_h5_thesis_updates.py)                                                |
| 5.3 septum old/new ratio error, 5.3b old/new pressure path | [`septum_proxy_robustness_old_new.py`](septum_proxy_robustness_old_new.py)                              |
| 5.3b septum strain-direction, 5.3c septum layer mechanics | [`analyze_h5_septum_mechanism.py`](analyze_h5_septum_mechanism.py)                                      |
| `fig_septum_lambda_scan`, `fig_septum_layer_work`         | [`septum_mechanics_proxy_test.py`](septum_mechanics_proxy_test.py)                                      |
| `fig_freewall_ratio_spectrum`, `fig_freewall_single_case_ratio` | [`freewall_ratio_proxy_test.py`](freewall_ratio_proxy_test.py)                                          |
| `fig_regional_ratio_waveform`                             | [`regional_ratio_waveform_test.py`](regional_ratio_waveform_test.py)                                    |
| `fig_cascade_loops`, `fig_cascade_cumulative`             | [`analyze_cascade.py`](analyze_cascade.py)                                                              |
| `fig_closure_audit`                                       | [`audit_three_bugs_energy_budget.py`](audit_three_bugs_energy_budget.py) (+ [`robin_reference_replay.py`](robin_reference_replay.py)) → [`plot_three_bugs_audit.py`](plot_three_bugs_audit.py) |
| `fig_stress_magnitudes`                                   | [`plot_stress_magnitudes.py`](plot_stress_magnitudes.py)                                                |
| Appendix B tables (numerical robustness)                  | [`analyze_h5_sweep_core.py`](analyze_h5_sweep_core.py)                                                  |
| `fig_unloaded_cap_grid`, `fig_ed_cap_grid` (ParaView)     | [`export_unloading_cap_paraview.py`](export_unloading_cap_paraview.py)                                  |
| Chapter 2 mesh / fiber / tag panels (ParaView)            | [`export_static_geometry_tags.py`](export_static_geometry_tags.py)                                      |

Chapter 1, 2, and 4 figures that are pure matplotlib illustrations
(Regazzoni PV loops, Klotz EDPVR, Holzapfel-Ogden, circulation-
network diagrams) are generated from inside the thesis repository
(`RV/scripts/`) — not produced here.

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

# Robin reference-config replay (writes the V0 input the closure-audit figure needs)
sbatch --export=RESULTS_DIR=results/sims/<dir> sbatch/jobs/run_robin_reference_replay.sbatch

# Sweep-level h=5 core analysis (writes appendix-B tables)
sbatch sbatch/jobs/run_h5_sweep_core_analysis.sbatch

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
