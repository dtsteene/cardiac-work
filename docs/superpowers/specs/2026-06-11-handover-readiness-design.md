# cardiac-work handover-readiness — design

**Date:** 2026-06-11
**Goal:** make the `cardiac-work` repo handover-ready for the supervisor, organised
around the workflow we actually use now (the PAH pulmonary-windkessel pressure-proxy
study), with two functional additions (dual-frame strain; PVD/animation export matching
the exact deliverables) and a documentation rewrite. Legacy slimming and true
crash-resume are explicitly **out of scope** for this effort.

## Context / current state

- Repo is already pruned (16 root `.py`, organised `sbatch/jobs/`, `tests/`). Core
  pipeline is lean: `complete_cycle.py` (sim + checkpoint + Ta/pressure history) →
  `compute_per_cell.py` / `postprocess_metrics.py` / `metrics_calculator.py` (fields,
  work, proxies) → `analysis_core.py` / `sweep_analysis.py` (sweep aggregation).
- The study workflow lives in `pah_pulmonary_batch/`: manual circulation tuning
  (`make_baseline.py`, `make_sweep_params.py`, `sweep_pulmonary_0d.py`,
  `compare_baselines_0d.py`, `diagnose_linear_baseline.py`), the launcher
  (`submit_pah_pulmonary_sweep.sh`), and the figure generator (`make_pah_handover.py`).
- The Optuna optimiser (`optimize_mesh_circ.py`) is **not** in this repo (it lives in
  `circulation/examples/`) — already out, as desired.
- `clinical_frame.py` handles the longitudinal *direction* projection, **not** the
  strain *reference frame*; dual-frame is a genuine addition.
- `CLAUDE.md` is stale: references 6 deleted files (`compare_cases`, `compare_spectrum`,
  `generate_thickness_variants`, `investigate_wall_thickness`, `septum_editor`,
  `compare_mesh_geometry`) and the obsolete "wall-thickness variants" objective.

### Verified data facts (the study these tools serve)
The production sweep `results/sims/2026-06-09/pah_pulmonary_20260609_prodsweep/` —
3 bundles (`no_frank_starling`, `frank_starling_preload`, `frank_starling_relax`) ×
8 cases (RV systolic 25→95 mmHg) — was verified: one shared 8/5 inverse-unloaded mesh
(`shared_unloaded_L5/ref`, identical unloaded volumes LV 82.94 / RV 51.33 mL across all
24), canonical cell tagging (identical counts LV 3465 / RV 3339 / Septum 1266), no NaNs.
Each case's `per_cell_data.npz` carries true work (`w_total`, `w_ff/ss/nn/cross`) and the
ll/ff pressure-strain proxies `proxy_{PLV,PRV,Trans,Mean,Sum}_*` (+ derivable Affine via
λ = d_lv/(d_lv+d_rv)).

## Workstream A — dual-frame strain

Strain loops must be available in **two reference frames**, as separate figure sets:

- **Unloaded frame** = raw Green–Lagrange strain (the metrics' `mean_E_{ff,ll}` are
  already relative to the unloaded reference, since sims load the unloaded mesh — so raw
  `E` is the unloaded frame: ED sits at ~+5–10 % stretch, systole moves toward unloaded).
- **Clinical frame** = `E − E[ED]` (re-zeroed at last-beat end-diastole; shortening reads
  negative; what speckle-tracking measures — current behaviour).

**Design:**
- Parameterize the loop plotter in `pah_pulmonary_batch/make_pah_handover.py`
  (`fig_stress_pressure_strain`) by `frame ∈ {"clinical","unloaded"}`.
- Emit two figure sets per bundle:
  `circulation/clinical/loops_stress_pressure_strain.{png,pdf}` and
  `circulation/unloaded/loops_stress_pressure_strain.{png,pdf}`.
  Pressure–strain panels follow the same frame on the ε axis.
- Correlation and ratio figures are unchanged (single set): the proxy work ∮P dε is
  offset-invariant, so frame choice does not change r-values or ratios. The README states
  this explicitly.
- `clinical_frame.py` (longitudinal direction projection) is untouched.

**Acceptance:** for every bundle, `circulation/clinical/` and `circulation/unloaded/`
each contain the stress–strain + pressure–strain loop figure; the unloaded-frame ED point
sits at positive stretch, the clinical-frame ED point sits at strain 0.

## Workstream B — PVD / animation export (all 3 bundles, bundle-agnostic)

Two exporters under `paraview_exports/`, each taking `(sweep_root, bundle, case…)` args so
they run for any bundle. Both run via sbatch (FEM checkpoint reads / replay — not login).

### B1. ED-static export (adapt `export_production_sweep_for_animation.py`)
- Repoint `SWEEP_ROOT`/severity list from the dead capped sweep to the pulmonary sweep,
  per bundle: `results/sims/2026-06-09/pah_pulmonary_20260609_prodsweep/<bundle>/<case>`.
- ED time = start of the last beat (6-beat × 75 bpm → derive from `simulation_params`/
  history rather than hardcoding 4.0 s).
- Per case: ED-deformed `.vtu` + per-bundle `sweep.pvd` (severity as the pvd "time").
- Cell fields: **work density** = `w_{total,ff,ss,nn,cross}` ÷ `cell_volumes` (J/m³), and
  **pressure-strain density** = `proxy_{PLV,PRV,Trans,Mean,Sum}_ll` ÷ `cell_volumes`,
  plus region tags / coordinates already supported.
- Output: `paraview_exports/pah_pulmonary_ed/<bundle>/`.

### B2. Through-beat export (new)
- For **baseline (`case0_rv25`) and severe (`case7_rv95`)** per bundle.
- Replay the last beat timestep-by-timestep from `solver/checkpoint.bp` (reuse the
  `compute_per_cell.py` replay/forms machinery), writing a **time-series PVD** of the
  deforming mesh with **cumulative-from-ED** work density and pressure-strain density
  (accumulated over the beat), so the animation shows work building up.
- Output: `paraview_exports/pah_pulmonary_beat/<bundle>/<case>/` (PVD + per-step VTU,
  beat phase as time).
- An sbatch wrapper analogous to `run_per_cell.sbatch`.

**Acceptance:** ED-static PVD opens in PyVista with work-density + PS-density fields for
all 8 cases of each bundle; through-beat PVD animates the cumulative fields over one beat
for baseline & severe in each bundle.

## Workstream C — docs + structure

- **Rewrite `CLAUDE.md`**: replace the "Current Objective: Wall Thickness Variants"
  section with the pulmonary-windkessel pressure-proxy study; correct the file table
  (drop the 6 dead refs; add `pah_pulmonary_batch/`, `analysis_core.py`,
  `clinical_frame.py`, `sweep_analysis.py`, `make_pah_handover.py`, the exporters);
  keep the accurate bug/convention sections.
- **Add `WORKFLOW.md`** at repo root tracing the canonical pipeline end-to-end:
  1. tune circulation (`pah_pulmonary_batch/` — manual R/C, linear EDPVR baseline),
  2. inspect 0D convergence + pre-coupling PV loops,
  3. `complete_cycle.py` — coupled sim, saves displacement checkpoint + Ta + pressures,
  4. `compute_per_cell.py` / `postprocess_metrics.py` — fields: S, E, work density S:dE,
     directional strain (ff/ll/circ/radial) and stress (ff/ss/nn), regional proxies,
  5. `make_pah_handover.py` — sweep aggregation: correlations (primary) + ratios,
     dual-frame loops,
  6. exporters — ED-static + through-beat PVD for PyVista animation.
- **Light structure only**: keep `pah_pulmonary_batch/` as the canonical study home;
  make entry points obvious in the docs. Note (do not delete) the kept-but-legacy bits
  (`geometry_generator.py` thickness/PCA paths, `export_unloading_cap_paraview.py`).

**Acceptance:** `CLAUDE.md` references no nonexistent files; `WORKFLOW.md` lets a new
reader run the pipeline end-to-end; every script named in the docs exists.

## Workstream D — shared results directory (collaboration with henriknf)

**Goal:** move results out of the locked home dir into a shared location that the
supervisor (`henriknf`, uid 5370) can read *and* write, and make the code path-agnostic
so both users run from the same data.

### Constraints discovered on ex3 (beegfs `/global/D1`)
- `henriknf` and `dtsteene` share exactly one group: **`cppm_via_users` (gid 5009)** —
  the only viable sharing mechanism.
- **POSIX ACLs are unsupported** (`setfacl` → "Operation not supported") → must use
  group ownership + **setgid**, not per-user ACLs.
- `/global/D1/homes/dtsteene` is `drwxr-x---` group `uio` (henriknf not in `uio`) →
  the shared dir **cannot** live under the home dir.
- `/global/D1/projects/` is not writable by `dtsteene` (group `domain users`).
- `/global/D1/` itself is world-writable and already hosts peer dirs
  (`swarm_results`, `dspy_cachedir`) → the shared dir lives directly under it.
- `results/` is **3.4 TB** on beegfs; the FS is ~100 % full (732 G free) → a copy is
  impossible, but a same-filesystem **`mv` is instant and needs no space**.
- Hardcoded paths are minimal: 0 literal absolute results paths; 4 files use
  repo-relative `results/...`; a handful build `ROOT/"results"` from a hardcoded
  `ROOT="/home/dtsteene/..."` (e.g. `make_pah_handover.py`, the exporters).

### Design
1. **Create** `/global/D1/cardiac_rv_shared/` (name configurable): owner `dtsteene`,
   `chgrp cppm_via_users`, `chmod 2770` (setgid + `rwxrwx---`) so the group has rwx and
   new files inherit `cppm_via_users`; no world access. Parent `/global/D1` is 0777 so
   henriknf can traverse to it.
2. **Move** the whole tree: `mv .../cardiac-work/results /global/D1/cardiac_rv_shared/results`
   — one atomic rename on the same FS. Precondition: **no queued/running jobs** (verify
   `squeue` empty first).
3. **Symlink back**: `ln -s /global/D1/cardiac_rv_shared/results <repo>/results`, so every
   repo-relative `results/...` path keeps working unchanged.
4. **Open permissions** on the moved tree: `chgrp -R cppm_via_users`,
   `chmod -R g+rwX`, and setgid on directories (`find -type d -exec chmod g+s`). This is
   recursive over 3.4 TB / many files (slow beegfs metadata pass) — run it deliberately
   and **verify henriknf can read+write** a probe file afterward.
5. **Path indirection**: add `paths.py` exposing `RESULTS_ROOT` resolved as
   `os.environ.get("CARDIAC_RESULTS_ROOT")` → else `<repo>/results` (the symlink). Replace
   hardcoded `ROOT="/home/dtsteene/..."`/`ROOT/"results"` builders (in `make_pah_handover.py`,
   the exporters, `pah_pulmonary_batch/` tools) with `paths.RESULTS_ROOT`. Repo root itself
   derived via `Path(__file__).resolve().parents[...]`, never a hardcoded home path.
6. **Document for henriknf** (in `WORKFLOW.md`/README): clone the repo, then either
   `ln -s /global/D1/cardiac_rv_shared/results results` or `export CARDIAC_RESULTS_ROOT=...`;
   he is in `cppm_via_users` so he has rwx automatically.

### Safety / reversibility
- The `mv` is atomic — no partial state. Rollback = `mv` back + remove symlink.
- `chgrp`/`chmod` are reversible (`chgrp -R uio`, restore modes).
- Owner access (`dtsteene`) is never reduced. The FS-full condition is pre-existing and
  unaffected by the move (rename frees no space, needs none).

**Acceptance:** `results/` is a symlink to `/global/D1/cardiac_rv_shared/results`; all
existing scripts still find their data via the symlink / `paths.RESULTS_ROOT`; a probe
confirms `henriknf` can create and read a file under the shared dir; `make_pah_handover.py`
runs unchanged against the symlinked path.

## Out of scope (deferred by user)
- Deleting legacy files (thickness/PCA mesh gen, unloading-cap exporter, etc.).
- True partial-checkpoint crash-resume of an interrupted sim.

## Notes / constraints
- No heavy compute on the login node — exporters and any replay go through sbatch.
- Commits: no `Co-Authored-By` trailer; commit only when the user asks.
- Figure-generation reruns are cheap and idempotent (`make_pah_handover.py`).
