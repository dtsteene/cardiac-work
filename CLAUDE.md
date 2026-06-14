# Cardiac Work — Proxy Validation via Biventricular FEM

> **Results live in a shared directory; the repo's `results/` is a symlink — see WORKFLOW.md
> for the layout and collaborator setup.**

## What This Project Is

A master thesis project investigating whether clinical pressure-strain proxies accurately
track the mechanical work done by the myocardium, with emphasis on the interventricular
septum in pulmonary arterial hypertension (PAH). The code runs biventricular finite element
simulations (FEniCSx + fenicsx-pulse), computes ground-truth internal work from the
stress-strain tensor integral, and compares it against clinical proxies derived from
pressure-strain loops.

The thesis manuscript lives in `/home/dtsteene/D1/RV/` as a MyST Markdown Jupyter Book.
Writing instructions are in `/home/dtsteene/D1/RV/THESIS_INSTRUCTIONS.md`.

## The Scientific Question

Clinicians estimate myocardial work from pressure-volume loops: W = ∮ P dV. For individual
wall segments they use pressure-strain loops: W_PS ~ ∮ P d(ε_ff). The question is **which
pressure P should be used for the septum?**

The septum is a shared wall loaded by P_LV from one side and P_RV from the other. Across
this pulmonary-loading sweep (single shared inverse-unloaded reference, which removes the
per-case unloading artifact), **P_RV tracks septal work best** (r≈0.75 in the no-FS
bundle), with Mean = ½(P_LV+P_RV), Sum = P_LV+P_RV (≡ 2·Mean, hence identical correlation),
and the affine λ-blend near-equivalent just behind (r≈0.70). **Transmural pressure
(P_LV − P_RV) is the worst choice** (r≈0.43, dropping further in the FS bundles); the
earlier "transmural is best" result was an artifact of per-case inverse unloading that is
eliminated here. On the RV free wall, P_RV (and Mean/Sum/affine) track work at r≈0.98
while transmural fails.

Ground-truth internal work: W_int = ∫₀ᵀ ∫_Ω S : dE dV (second Piola-Kirchhoff stress,
Green-Lagrange strain). This is what the code computes and what the proxies are evaluated
against.

## Why Internal Work Matters

The tensor integral S:dE captures the actual mechanical energy expenditure of the
myocardium — it accounts for all stress directions (fiber, sheet, sheet-normal, cross-fiber),
not just fiber shortening. Clinical proxies assume work = pressure × strain_ff, which misses
~50-65% of total internal work. Understanding what the proxy captures vs what it misses is
essential for interpreting clinical myocardial work imaging in disease states.

We decompose work by:
- **Direction**: fiber (ff), sheet (ss), sheet-normal (nn), cross-fiber
- **Stress type**: active, passive, compressible
- **Region**: LV free wall (marker=1), RV free wall (marker=2), Septum (marker=3)

## Current Objective: PAH Pulmonary-Windkessel Pressure-Proxy Study

A controlled pressure-loading sweep on **one fixed heart** — one shared 8/5 mmHg
inverse-unloaded mesh with identical unloaded volumes (LV 82.94 / RV 51.33 mL) and
canonical cell tagging (LV 3465 / RV 3339 / Septum 1266 cells) across all 24 simulations.

The pulmonary arterial resistance R is swept up and compliance C swept down together at
conserved RC ≈ 0.33 s, spanning RV systolic 25 → 95 mmHg (mPAP 17 → 55 mmHg,
PVR 1.5 → 11 WU) in 8 evenly spaced cases (`case0_rv25` … `case7_rv95`).

Three active-contraction bundles are run over these 8 cases:
- **`no_frank_starling`** — constant active tension Ta = 100 kPa (thesis model)
- **`frank_starling_preload`** — Frank-Starling frozen at ED stretch, Ta = 220 kPa
- **`frank_starling_relax`** — Frank-Starling with activation lag τ = 250 ms, Ta = 220 kPa

The production sweep lives under
`results/sims/2026-06-09/pah_pulmonary_20260609_prodsweep/<bundle>/<case>/`.

Each case's `per_cell_data.npz` carries true work (`w_total`, `w_ff/ss/nn/cross`) and the
ll/ff pressure-strain proxies `proxy_{PLV,PRV,Trans,Mean,Sum}_*` (Affine is derivable via
λ = d_lv / (d_lv + d_rv)).

## Code Architecture

### Simulation Pipeline
```
complete_cycle.py ─▶ compute_per_cell.py ─▶ postprocess_metrics.py ─▶ analysis_core.py ─▶ sweep_analysis.py
  FEM + 0D solver     per-cell S, E, work      regional metrics          per-region stats     cross-sim
  → checkpoint.bp     → per_cell_data.npz      → metrics_*.npy           (proxy, ratio,        correlation
  + Ta/pressure hist                                                        correlation)
```

**Key paradigm**: Simulation and postprocessing are fully decoupled. `complete_cycle.py`
saves displacement checkpoints + Ta + pressures. `postprocess_metrics.py` replays from
checkpoint to compute region metrics. `compute_per_cell.py` writes per-cell fields to
`per_cell_data.npz`. Changing `metrics_calculator.py` only requires rerunning
postprocessing, not re-simulation.

### Core Files
| File | Role |
|------|------|
| `complete_cycle.py` | FEM solver driver — geometry, fibers, solver loop, checkpoint saving |
| `postprocess_metrics.py` | Offline regional metrics from checkpoint — no re-solve needed |
| `compute_per_cell.py` | Per-cell S, E, work density and pressure-strain proxies → `per_cell_data.npz` |
| `metrics_calculator.py` | MetricsCalculator class — S, E, work decomposition per region |
| `analysis_core.py` | Pure-NumPy stats home — proxy work, ratios, correlations; no FEniCSx dep |
| `sweep_analysis.py` | Cross-simulation sweep aggregation — scalar QoIs, correlation/ratio tables |
| `clinical_frame.py` | Longitudinal-direction projection: projects LDRB `l0` into wall tangent plane |
| `paths.py` | Results-root indirection — `$CARDIAC_RESULTS_ROOT` → repo `results/` symlink |
| `plot_loops.py` | PV/PS/SS loop figures, energy balance debug plots |
| `eval_proxies.py` | Quantitative proxy validation (R², error %) |
| `run_postprocessing.py` | Orchestrator — auto-detects what needs running |
| `geometry_generator.py` | UKB atlas mesh generation (kept; legacy thickness/PCA paths not in active use) |
| `geometry_utils.py` | Mesh utility helpers |
| `plot_utils.py` | Shared matplotlib styling |
| `export_production_sweep_for_animation.py` | ED-static PVD export per bundle — work/pressure-strain density fields at ED for all 8 cases |
| `export_beat_animation.py` | Through-beat PVD animation — baseline + severe, cumulative density fields over one heartbeat |
| `export_static_geometry_tags.py` | Export cell-region tags for static geometry visualisation |
| `export_unloading_cap_paraview.py` | Unloading-cap export (kept; legacy, not in active sweep pipeline) |

### Study Home: `pah_pulmonary_batch/`
| File | Role |
|------|------|
| `pah_pulmonary_batch/make_baseline.py` | Linear-EDPVR baseline circulation params (sPAP22, EB re-fit for UKB L5 mesh) |
| `pah_pulmonary_batch/make_sweep_params.py` | Places 8 cases evenly in RV systolic 25 → 90 mmHg; writes circulation JSONs |
| `pah_pulmonary_batch/sweep_pulmonary_0d.py` | 0D standalone circulation sweep for quick convergence checks |
| `pah_pulmonary_batch/compare_baselines_0d.py` | Compare candidate baseline circulation params in 0D |
| `pah_pulmonary_batch/diagnose_linear_baseline.py` | Diagnose 0D convergence of the linear-EDPVR baseline |
| `pah_pulmonary_batch/submit_pah_pulmonary_sweep.sh` | Slurm launcher for the full 3-bundle × 8-case sweep |
| `pah_pulmonary_batch/make_pah_handover.py` | Figure generator: per-bundle correlation / ratio / circulation panels; dual-frame (clinical + unloaded) stress/pressure-strain loops |

### Batch Submission (Slurm, under `sbatch/jobs/`)
- `run_sim_and_post.sbatch` — full sim + postprocessing (mi50q, 8 MPI)
- `run_postprocess_only.sbatch` — standalone postprocessing
- `run_per_cell.sbatch` — per-cell field computation
- `run_per_cell_canonical.sbatch` — per-cell on canonical (atlas) mesh
- `run_sweep_analysis.sbatch` — sweep aggregation
- `export_production_sweep_for_animation.sbatch` — ED-static PVD export
- `export_beat_animation.sbatch` — through-beat PVD export
- `run_repost_if_missing.sbatch`, `recover_postprocess_run.sbatch` — recovery helpers
- `run_mesh_convergence_geometry.sbatch` — mesh convergence geometry runs

## Data Layout

```
data/
  healthy.{h5,xdmf}                 # Patient-specific healthy mesh
  pah.{h5,xdmf}                     # Patient-specific PAH mesh
  healthy_circulation_params.json    # 0D boundary conditions (healthy)
  ph_circulation_params.json         # 0D boundary conditions (PAH)
  shared_ukb_mesh/                   # Shared UKB atlas mesh used across all sweep cases

results/                             # Symlink → shared dir (see WORKFLOW.md)
  sims/                              # Simulation outputs (date-organised)
  handover/                          # Handover figure outputs

paraview_exports/                    # PVD/VTU output for PyVista animation
  pah_pulmonary_ed/<bundle>/         # ED-static sweep (one VTU per case + sweep.pvd)
  pah_pulmonary_beat/<bundle>/<case>/# Through-beat animation (step_NNN.vtu + beat.pvd)
```

## Simulation Output Structure

Each simulation produces a `results_*_hybrid_*bpm/` directory containing:
- `solver/checkpoint.bp` — displacement field u(t) for all timesteps
- `Ta_solver_history.npy` — active tension per region per timestep
- `pressure_history.npy` — Lagrange multiplier cavity pressures (NOT 0D Windkessel)
- `simulation_params.json` — material params, BPM, dt, mesh info
- `circulation/history.npy` — 0D model state history
- `per_cell_data.npz` — per-cell work density and proxy fields (from `compute_per_cell.py`)
- `geometry/` — mesh + LDRB fiber fields
- `visualization/` — XDMF for ParaView

## LDRB Fiber Coordinate System
- `f0` = fiber = circumferential (helical, rotated by alpha)
- `s0` = sheet = apicobasal (rotated by beta)
- `n0` = sheet-normal = transmural (rotated by beta, endo to epi)
- `l0` = longitudinal = apex-to-base (Laplace gradient, no rotation)

LV angles: alpha_endo=+60, alpha_epi=-60. RV angles: alpha_endo=+90, alpha_epi=-25.

## Metric Naming Convention
- Work keys: `work_ff`, `work_ss`, `work_nn`, `work_cross`
- PS proxy keys: `work_ps_ff_*` (fiber strain), `work_ps_ll_*` (longitudinal/GLS strain)
- Per-cell proxy keys: `proxy_{PLV,PRV,Trans,Mean,Sum}_{ff,ll}` (in `per_cell_data.npz`)
- Strain: `mean_E_ff`, `mean_E_ss`, `mean_E_nn`, `mean_E_ll`
- Regions in key names: `_LV`, `_RV`, `_Septum`

## Dual-Frame Strain Convention
Strain loops are available in two reference frames from `pah_pulmonary_batch/make_pah_handover.py`:
- **Unloaded frame** — raw Green-Lagrange E (ED sits at ~+5-10% stretch)
- **Clinical frame** — E − E[ED], re-zeroed at end-diastole (shortening reads negative;
  matches speckle-tracking convention)

The proxy work ∮ P dε is offset-invariant, so frame choice does not change r-values
or ratios. Correlation and ratio figures use a single figure set.

## Critical Bugs (Fixed, But Be Aware)

1. **pulse.Variable unit serialization**: `str(v.unit)` returns SI decomposition (Pa),
   not original unit (kPa). Use `str(v.original_unit)`. Documented in
   `docs/bug_report_unit_serialization.md`.

2. **Solver vs 0D pressure mismatch**: Cavity Lagrange multiplier pressure differs 30-60%
   from 0D Windkessel. Boundary work must use solver pressure from `pressure_history.npy`.
   Runs before 2026-03-08 lack this file.

3. **DG1 projection oscillations**: Projecting stress/strain to DG1 causes spurious
   oscillations at thin septum. Use DG0 for work integration.

## Software Stack
- FEniCSx (finite elements), fenicsx-pulse (cardiac mechanics), cardiac-geometries
- LDRB (fiber fields), circulation (0D lumped-parameter model)
- adios4dolfinx (checkpointing), GMSH (meshing)
- Slurm HPC (mi50q partition)

## Three Simulation Cases (Thesis)
1. **UKB synthetic** — UK Biobank statistical shape model mean, healthy hemodynamics
2. **Patient-specific healthy** — CMR-derived geometry, healthy circulation
3. **Patient-specific PAH** — CMR-derived, RV hypertrophy, elevated RV pressure (~60 mmHg)

The pulmonary-windkessel sweep (current objective) uses the UKB synthetic mesh with
programmatically varied pulmonary afterload.

## Thesis Connection
The thesis (`/home/dtsteene/D1/RV/`) is fully written (6 chapters + intro) as MyST
Markdown. When working on thesis content, read `THESIS_INSTRUCTIONS.md` first — it
enforces prose-first scientific writing (no bullets for content, equations in sentences,
sparse subheadings). The thesis title is "Pressure Proxies for Septal Mechanical Work:
A Biventricular Finite Element Study."
