# End-to-End Workflow: PAH Pulmonary-Windkessel Pressure-Proxy Study

This document traces the canonical pipeline from circulation tuning to ParaView
animation export.  A new reader should be able to reproduce all results from it.

## Overview

The study asks which ventricular pressure (P_LV, P_RV, or transmural P_LV−P_RV)
best proxies the true internal myocardial work of the septum as RV systolic
pressure rises from 25 to 95 mmHg.  Three Frank-Starling bundles are compared:
`no_frank_starling` (constant Ta=100 kPa, the thesis model), `frank_starling_preload`
(g frozen at ED stretch, Ta=220 kPa), and `frank_starling_relax` (activation-lag
g, Ta=220 kPa, tau=250 ms).  Each bundle runs 8 pulmonary-windkessel cases on
one shared UKB L5 mesh.

```
pah_pulmonary_batch/make_baseline.py        tune circulation
pah_pulmonary_batch/make_sweep_params.py
pah_pulmonary_batch/submit_pah_pulmonary_sweep.sh  -->  complete_cycle.py  (FEM, sbatch)
                                                          ↓
                                              compute_per_cell.py   (fields, sbatch)
                                              postprocess_metrics.py (loops, sbatch)
                                                          ↓
                                         pah_pulmonary_batch/make_pah_handover.py
                                                          ↓
                                   export_production_sweep_for_animation.py  (sbatch)
                                   export_beat_animation.py                   (sbatch)
```

---

## Step 1 — Tune circulation

All circulation work is **login-safe** (pure 0D, no FEM).

**1a. Build the linear-EDPVR baseline**

```bash
cd cardiac-work
python pah_pulmonary_batch/make_baseline.py
```

Writes `pah_pulmonary_batch/circ_params/baseline_linear_v2.json`.  The baseline
is sPAP22 (UKB-matched) with the ventricular EDPVR linearised: the exponential
Klotz term is removed and a diastolic slope EB is re-fit so the linear law
recovers the physiological ED pressure at the shared mesh ED volumes
(LV 111.5 mL → EDP 8.0 mmHg, RV 76.9 mL → EDP 5.0 mmHg).  Everything else
(systemic windkessel, pulmonary windkessel, atria, valves) is kept from the
Optuna-optimised sPAP22 starting point.

> **Note:** The Optuna optimiser (`optimize_mesh_circ.py`) is **not** in this
> repo.  It lives in the separate `circulation/examples/` directory.
> Circulation tuning here is manual R/C adjustment on top of the linear
> baseline — the optimiser is not needed to reproduce the sweep.

**1b. Place 8 PAH cases along the pulmonary locus**

```bash
python pah_pulmonary_batch/make_sweep_params.py
```

Writes `pah_pulmonary_batch/circ_params/case{0..7}_rv{25,35,45,55,65,75,85,95}.json`.
Each JSON is identical to the baseline except for `R_AR_PUL` (up) and
`C_AR_PUL` (down) so that 0D PA systolic hits 25→90 mmHg in 8 even steps.

**1c. Inspect convergence and PV loops before committing to FEM runs**

```bash
python pah_pulmonary_batch/sweep_pulmonary_0d.py        # 0D sweep across cases
python pah_pulmonary_batch/compare_baselines_0d.py      # compare baseline variants
python pah_pulmonary_batch/diagnose_linear_baseline.py  # sanity-check the linear baseline
```

These are all pure-0D and run on the login node.  Use them to confirm each
case reaches a steady hemodynamic cycle before submitting expensive FEM jobs.

---

## Step 2 — 0D warm-up inside the coupled solver

`complete_cycle.py` runs a 0D warm-up before coupling: the circulation model
spins up for `PRE_CIRC_BEATS=30` beats (up to `PRE_CIRC_MAX_BEATS=80`) with a
convergence tolerance of `PRE_CIRC_CONVERGENCE_TOL=0.002`.  These knobs are set
via env in the sbatch export (see Step 3).  The standalone tools above let you
verify 0D convergence and PV-loop shape independently so you can catch
hemodynamic problems before committing to a 6-beat FEM run.

---

## Step 3 — Coupled FEM simulation (sbatch)

> All FEM work must go through sbatch or an interactive `salloc`.  Never run
> FEniCSx or the replay scripts on the login node.

**Launch the full sweep (3 bundles × 8 cases = 24 jobs):**

```bash
bash pah_pulmonary_batch/submit_pah_pulmonary_sweep.sh
```

Each job calls `sbatch/jobs/run_sim_and_post.sbatch` → `complete_cycle.py`.

**Bundle selection** is controlled by env vars passed via `--export`:

| Bundle | Key env vars |
|--------|-------------|
| `no_frank_starling` | `USE_FRANK_STARLING=0`, `TA_PEAK_KPA=100.0` |
| `frank_starling_preload` | `USE_FRANK_STARLING=1`, `TA_PEAK_KPA=220.0`, `FS_PRELOAD_ONLY=1` |
| `frank_starling_relax` | `USE_FRANK_STARLING=1`, `TA_PEAK_KPA=220.0`, `FS_PRELOAD_ONLY=0`, `FS_RELAX_TAU_MS=250` |

To fire a subset of bundles:

```bash
BUNDLES_OVERRIDE="no_frank_starling" bash pah_pulmonary_batch/submit_pah_pulmonary_sweep.sh
```

**Shared unloaded reference** — all 24 jobs load the same inverse-unloaded
mesh (`pah_pulmonary_batch/shared_unloaded_L5/ref/solver/prestress_inverse.bp`,
fixed ED target LV 7.77/RV 5.00 mmHg) via `LOAD_UNLOADED_FROM`.  This ensures
identical unloaded volumes (LV 82.94/RV 51.33 mL) and cell tagging across all
cases so that results are directly comparable.

**What `complete_cycle.py` saves per case** (under `results/sims/<date>/pah_pulmonary_<stamp>/<bundle>/<case>/`):

| File | Contents |
|------|----------|
| `solver/checkpoint.bp` | Displacement field u(t) for all timesteps |
| `Ta_solver_history.npy` | Active tension per region per timestep |
| `pressure_history.npy` | Lagrange multiplier cavity pressures (use these, not 0D Windkessel) |
| `simulation_params.json` | Material params, BPM, dt, mesh info |
| `circulation/history.npy` | 0D model state history |

A submission manifest is written to `results/analysis/pah_pulmonary_sweep_<stamp>/pah_pulmonary_cases.tsv`.

---

## Step 4 — Fields: per-cell work and loop metrics (sbatch)

These scripts replay from checkpoint — they never re-solve the FEM problem.
Changing the metric definition only requires rerunning postprocessing.

**Canonical per-cell fields** (`compute_per_cell.py`):

```bash
sbatch sbatch/jobs/run_per_cell_canonical.sbatch   # or sbatch/jobs/run_per_cell.sbatch for a single case
```

Reads `solver/checkpoint.bp` + `Ta_solver_history.npy` + `pressure_history.npy`,
replays the last beat with DG0 assembly, and writes `per_cell_data.npz` into
the case directory.  Fields in the NPZ:

- **True work density**: `w_total`, `w_ff`, `w_ss`, `w_nn`, `w_cross` (J/m³, S:dE)
- **Directional strain**: `E_ll`, `E_ff`, `E_circ`, `E_radial` (Green-Lagrange)
- **Pressure-strain proxy density** per pressure choice:
  `proxy_{PLV,PRV,Trans,Mean,Sum}_ll` and `_ff` variants
- **Region tags** and per-cell volumes (for normalisation)
- **Transventricular coordinate** tau (Euclidean)

**Loop and boundary metrics** (`postprocess_metrics.py`):

```bash
sbatch sbatch/jobs/run_postprocess_only.sbatch
```

Writes regional aggregates (mean stress/strain, PV loop areas, Robin work) to
`metrics/` inside the case directory.  Used by `plot_loops.py` and
`run_postprocessing.py`.

---

## Step 5 — Sweep figures (login-safe)

```bash
python pah_pulmonary_batch/make_pah_handover.py
```

Reads `per_cell_data.npz` for all 24 cases.  Output under
`results/handover/pah_pulmonary_paper_20260611/` (path from `paths.RESULTS_ROOT`).

Per bundle, three figure sets are produced:

| Subdirectory | Contents |
|---|---|
| `<bundle>/correlation/` | **Headline**: SS true work vs P×ε_ll proxy, per region (LV/RV/Septum), over all pressure choices (P_LV, P_RV, Trans, Mean, Sum, Affine(λ)) |
| `<bundle>/ratio/` | LV/RV free-wall work ratio spectrum + scatter; septum ratio |
| `<bundle>/circulation/clinical/` | 0D PV loops, coupled PV loops, fibre stress-strain and pressure-strain loops — **clinical frame** (ε re-zeroed at ED) |
| `<bundle>/circulation/unloaded/` | Same loops in **unloaded frame** (raw Green-Lagrange; ED sits at positive stretch) |
| `<bundle>/data/` | Scalar CSV + correlation tables |

The correlation and ratio figures are frame-invariant (the proxy work ∮P dε is
offset-invariant), so they appear only once per bundle.

---

## Step 6 — Animation export (sbatch)

### ED-static sweep PVD (all 8 cases, per bundle)

```bash
BUNDLE=no_frank_starling sbatch sbatch/jobs/export_production_sweep_for_animation.sbatch
BUNDLE=frank_starling_preload sbatch sbatch/jobs/export_production_sweep_for_animation.sbatch
BUNDLE=frank_starling_relax sbatch sbatch/jobs/export_production_sweep_for_animation.sbatch
```

Calls `export_production_sweep_for_animation.py --bundle <BUNDLE>`.  For each
case, writes an ED-deformed `.vtu` with per-cell work density
(`w_{total,ff,ss,nn,cross}` ÷ cell volume, J/m³) and pressure-strain proxy
density, plus a `sweep.pvd` using severity as the PVD "time" axis.

Output: `paraview_exports/pah_pulmonary_ed/<bundle>/`.

### Through-beat PVD (baseline + severe, per bundle)

```bash
BUNDLE=no_frank_starling sbatch sbatch/jobs/export_beat_animation.sbatch
```

Override cases with `CASES="case0_rv25 case7_rv95"` (the default).  Calls
`export_beat_animation.py --bundle <BUNDLE>`.  Replays the last beat
timestep-by-timestep from `solver/checkpoint.bp` and writes per-step VTU with
**cumulative-from-ED** work density and PS density, so the animation shows work
building up over the beat.  Runs serially by design (single rank keeps DOLFINx
cell ordering consistent with PyVista).

Output: `paraview_exports/pah_pulmonary_beat/<bundle>/<case>/beat.pvd`.

---

## Collaborator setup (henriknf)

```bash
# 1. Clone the repo (needs collaborator access on github.com/dtsteene/cardiac-work)
git clone git@github.com:dtsteene/cardiac-work.git
cd cardiac-work

# 2. Point the repo at the shared results dir.
#    Option A — symlink (matches the production layout):
ln -s /global/D1/cardiac_rv_shared/results results

#    Option B — env var (no symlink needed):
export CARDIAC_RESULTS_ROOT=/global/D1/cardiac_rv_shared/results

# 3. Sanity check
python -c "import paths; print(paths.RESULTS_ROOT)"
```

The shared results directory lives at `/global/D1/cardiac_rv_shared/results`
(created by moving `results/` from the home directory — a pending step if it
does not exist yet; the `mv` on beegfs is atomic and needs no extra space).
`henriknf` is in the `cppm_via_users` group (gid 5009), which has `rwx` on the
shared directory via setgid bit — no additional ACL setup is needed.

> **Note on POSIX ACLs:** beegfs on ex3 does not support `setfacl`.  Sharing is
> via group ownership (`cppm_via_users`) + setgid on all directories.
> `/global/D1/homes/dtsteene` is not world-traversable, so the shared dir must
> remain directly under `/global/D1/` (not under the home directory).

---

## Path resolution

`paths.py` exposes `RESULTS_ROOT` with the following resolution order:

1. `$CARDIAC_RESULTS_ROOT` if set
2. `<repo>/results` (expected to be a symlink to the shared dir on the production system)

All scripts that read or write results import `paths.RESULTS_ROOT` — there are
no hardcoded absolute paths to home directories.

---

## Kept-but-legacy scripts (not part of the canonical sweep)

The following scripts remain in the repo but are not part of the pulmonary-
windkessel study pipeline.  Do not delete them — they may be useful for future
mesh work.

| Script | Original purpose |
|--------|-----------------|
| `geometry_generator.py` | UKB atlas mesh generation with thickness/PCA control |
| `export_unloading_cap_paraview.py` | ParaView export of the unloading cap geometry |
| `export_static_geometry_tags.py` | Static geometry tag export for visualisation |
