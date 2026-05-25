# cardiac-work

Biventricular finite-element simulations evaluating clinical
pressure-strain proxies against the ground-truth tensor work
`W = ∫ S:dE dV`, with emphasis on the interventricular septum in
pulmonary arterial hypertension (PAH). Central question: **which
pressure (P_LV, P_RV, or transmural P_LV − P_RV) best tracks septal
work as RV pressure rises?**

This is the simulation code for the MSc thesis
*Pressure Proxies for Biventricular Myocardial Work: A Finite Element
Study* (Daniel Steeneveldt, University of Oslo). The thesis itself
lives at [github.com/dtsteene/RV](https://github.com/dtsteene/RV).

## Pipeline

Simulation and analysis are decoupled. Change a metric, rerun
postprocessing — no re-simulation needed.

```
complete_cycle.py  →  compute_per_cell.py  →  postprocess_metrics.py  →  figure scripts
  (FEM + 0D solver)    (per-cell work/proxies)   (regional metrics)
```

- [`complete_cycle.py`](complete_cycle.py) runs the coupled biventricular FEM
  + 0D circulation simulation and writes a displacement checkpoint,
  active-tension history, and cavity-pressure history.
- [`compute_per_cell.py`](compute_per_cell.py) replays the checkpoint and
  writes per-cell stress, strain, and proxy fields without re-solving.
- [`postprocess_metrics.py`](postprocess_metrics.py) aggregates per-cell
  data into regional metrics (LV free wall, RV free wall, septum).
- [`metrics_calculator.py`](metrics_calculator.py) is the underlying
  metric library used by both per-cell and regional postprocessing.
- [`geometry_generator.py`](geometry_generator.py) builds the mesh,
  LDRB fibers, region tags, and per-cell geometric fields.

## Reproducing thesis figures

Every script at the project root produces (or is required to produce)
a figure or table in the published thesis. Most read saved metrics
from a results directory and need no FEniCSx itself — only `numpy`,
`matplotlib`, and the `.npz`/`.npy` outputs.

| Thesis figure(s)                          | Script                                                                                 |
|-------------------------------------------|----------------------------------------------------------------------------------------|
| 4.2, 4.6, 4.7, 4.8, 4.9, 5.0c, 5.0d, pv_standalone_vs_coupled | [`figures_thesis_pre_post_coupling_and_ps_loops.py`](figures_thesis_pre_post_coupling_and_ps_loops.py) |
| 5.0 freewall headline, 5.0 septum headline | [`generate_headline_figures.py`](generate_headline_figures.py)                         |
| 5.0b, 5.1, 5.2, 5.2b, 5.2c, 5.3a, 5.4, 5.5 | [`plot_h5_thesis_updates.py`](plot_h5_thesis_updates.py)                               |
| 5.3 (septum old/new ratio error), 5.3b (old/new pressure path) | [`septum_proxy_robustness_old_new.py`](septum_proxy_robustness_old_new.py) |
| 5.3b (septum strain-direction diagnostic), 5.3c (septum layer mechanics) | [`analyze_h5_septum_mechanism.py`](analyze_h5_septum_mechanism.py) |
| `fig_septum_lambda_scan`, `fig_septum_layer_work` | [`septum_mechanics_proxy_test.py`](septum_mechanics_proxy_test.py)                     |
| `fig_freewall_ratio_spectrum`, `fig_freewall_single_case_ratio` | [`freewall_ratio_proxy_test.py`](freewall_ratio_proxy_test.py)                         |
| `fig_regional_ratio_waveform`             | [`regional_ratio_waveform_test.py`](regional_ratio_waveform_test.py)                   |
| `fig_cascade_loops`, `fig_cascade_cumulative` | [`analyze_cascade.py`](analyze_cascade.py)                                             |
| `fig_closure_audit`                       | [`audit_three_bugs_energy_budget.py`](audit_three_bugs_energy_budget.py) (+ [`robin_reference_replay.py`](robin_reference_replay.py)) → [`plot_three_bugs_audit.py`](plot_three_bugs_audit.py) |
| `fig_stress_magnitudes`                   | [`plot_stress_magnitudes.py`](plot_stress_magnitudes.py)                               |
| Appendix B tables (numerical robustness)  | [`analyze_h5_sweep_core.py`](analyze_h5_sweep_core.py)                                 |
| `fig_unloaded_cap_grid`, `fig_ed_cap_grid` (ParaView) | [`export_unloading_cap_paraview.py`](export_unloading_cap_paraview.py)             |
| Chapter 2 mesh / fiber / tag panels (ParaView) | [`export_static_geometry_tags.py`](export_static_geometry_tags.py)                |

Chapter 1, 2, and 4 figures that are pure matplotlib illustrations
(regazzoni PV loops, Klotz EDPVR, Holzapfel-Ogden, circulation-
network diagrams) are generated from inside the thesis repository
(`RV/scripts/`) and are not produced here.

## Running simulations

All SLURM job templates live under [`sbatch/`](sbatch/) and all heavy
compute goes through SLURM — never the login node.

```bash
# Single sim + postprocessing
sbatch sbatch/run_sim_and_post.sbatch

# Postprocess only (replays from saved checkpoint)
sbatch --export=RESULTS_DIR=results/sims/<dir> sbatch/run_postprocess_only.sbatch

# Per-cell work (ED tagging mode, then canonical/reference mode)
sbatch --export=RESULTS_DIR=results/sims/<dir> sbatch/run_per_cell.sbatch
sbatch --export=RESULTS_DIR=results/sims/<dir> sbatch/run_per_cell_canonical.sbatch

# Capped RV-EDP shared-L5 production sweep (16 cases — the main thesis sweep)
bash sbatch/submit_capped_shared_l5_sweep.sh

# Shared-unloaded-reference sweep (Design B — only 0D circulation varies)
bash sbatch/submit_shared_unloaded_l5_sweep.sh

# v12 EXP-extra spectrum dispatcher (uses pre-tuned 0D JSONs in data/)
sbatch sbatch/submit_spectrum_v12_exp_extra.sbatch

# Patient-mesh FEM sweep
sbatch sbatch/submit_patient_mesh_fem.sbatch

# Unloading-only sensitivity sweeps (appendix C reference-state)
bash sbatch/submit_unloading_ab.sh
bash sbatch/submit_unloading_stiffness_sweep.sh

# Robin reference-config replay (input for the closure-audit figure)
sbatch --export=RESULTS_DIR=results/sims/<dir> sbatch/run_robin_reference_replay.sbatch

# Postprocess recovery (rerun postprocessing for a finished sim if outputs are missing)
sbatch --export=RESULTS_DIR=results/sims/<dir> sbatch/run_repost_if_missing.sbatch
```

## Geometry

[`geometry_generator.py`](geometry_generator.py) writes the mesh,
fibers, region tags, and per-cell geometric fields (distances,
Laplace scalars, septum masks) in one command:

```bash
python3 geometry_generator.py --single ukb -c 5 --output-dir data/shared_ukb_mesh
```

Outputs: `geometry.bp` (solver), `geometry_fields.npz` (per-cell
arrays), `geometry_fields.xdmf` (ParaView).

## Dependencies

- [FEniCSx](https://fenicsproject.org) (`dolfinx`, `ufl`,
  `basix`, `ffcx`) for the finite-element kernel.
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
no pinned environment file. Pre-tuned 0D circulation parameter
JSONs are checked in under `data/ukb_circ_v12_exp/` and
`data/patient_mesh_circ_v12_exp/` — Optuna is not required to
reproduce the thesis sweeps.

## Data

Simulation outputs (~2 TB of checkpoints, per-beat metrics, and
ParaView bundles) are not in the repository. The thesis figures
are reproducible from a 16-case capped shared-L5 sweep — see
[`sbatch/submit_capped_shared_l5_sweep.sh`](sbatch/submit_capped_shared_l5_sweep.sh)
for the exact case list.

## Tests

```bash
python3 tests/test_syntax.py                           # file-level syntax sanity
python3 tests/test_canonical_tagging.py <sim_dirs>     # invariants across a spectrum
```
