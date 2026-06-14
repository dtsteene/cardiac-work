# cardiac-work Handover-Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `cardiac-work` repo handover-ready: path-agnostic shared results dir for collaboration with `henriknf`, dual-frame (clinical + unloaded) strain figures, PVD/animation exporters for the PAH pulmonary sweep, and refreshed docs.

**Architecture:** A small `paths.py` resolves the results root (env override → repo `results` symlink) so the repo is user-agnostic; the 3.4 TB `results/` is moved (atomic same-FS `mv`) to a group-shared dir and symlinked back. Figure/export code reads through `paths.RESULTS_ROOT`. Dual-frame strain is a one-line reference-shift toggle in the loop plotter. Exporters reuse the existing `compute_per_cell.py` replay machinery.

**Tech Stack:** Python 3.11 (numpy, matplotlib, pyvista, dolfinx/adios4dolfinx), beegfs/`/global/D1`, SLURM, pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-11-handover-readiness-design.md`

**Conventions:** No `Co-Authored-By` trailer on commits. No heavy compute on the login node (FEM replay → sbatch). All paths below are relative to the repo root `/global/D1/homes/dtsteene/cardiac-work` unless absolute.

---

## Task 1: `paths.py` — results-root indirection

**Files:**
- Create: `paths.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
import importlib, os
from pathlib import Path

def test_results_root_defaults_to_repo_results(monkeypatch):
    monkeypatch.delenv("CARDIAC_RESULTS_ROOT", raising=False)
    import paths; importlib.reload(paths)
    assert paths.results_root() == paths.REPO_ROOT / "results"

def test_results_root_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CARDIAC_RESULTS_ROOT", str(tmp_path))
    import paths; importlib.reload(paths)
    assert paths.results_root() == tmp_path

def test_repo_root_is_this_repo():
    import paths; importlib.reload(paths)
    assert (paths.REPO_ROOT / "complete_cycle.py").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /global/D1/homes/dtsteene/cardiac-work && python -m pytest tests/test_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paths'`

- [ ] **Step 3: Write minimal implementation**

```python
# paths.py
"""Single source of truth for the results root, so the repo is user-agnostic.

Resolution order:
  1. $CARDIAC_RESULTS_ROOT if set (lets a collaborator point at the shared dir)
  2. <repo>/results  (a symlink to the shared dir on the production system)
"""
from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def results_root() -> Path:
    env = os.environ.get("CARDIAC_RESULTS_ROOT")
    return Path(env).resolve() if env else (REPO_ROOT / "results")


RESULTS_ROOT = results_root()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_paths.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add paths.py tests/test_paths.py
git commit -m "feat: paths.py results-root indirection (env override -> repo symlink)"
```

---

## Task 2: Repoint hardcoded result roots to `paths.RESULTS_ROOT`

Hardcoded `ROOT="/home/dtsteene/..."` / `ROOT/"results"` builders break for any other user. Replace them with `paths.RESULTS_ROOT`. The four repo-relative `results/...` users keep working via the symlink (Task 3) and need no change.

**Files:**
- Modify: `pah_pulmonary_batch/make_pah_handover.py` (the `ROOT`/`SWEEP`/`OUT` constants near the top)
- Modify: `export_production_sweep_for_animation.py` (the `REPO`/`SWEEP_ROOT`/`DEFAULT_OUT` constants)

- [ ] **Step 1: Repoint `make_pah_handover.py`**

Replace the hardcoded root block:

```python
ROOT = Path("/home/dtsteene/D1/cardiac-work")
SWEEP = ROOT / "results/sims/2026-06-09/pah_pulmonary_20260609_prodsweep"
OUT = ROOT / "results/handover/pah_pulmonary_paper_20260611"
```

with:

```python
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path
import paths
SWEEP = paths.RESULTS_ROOT / "sims/2026-06-09/pah_pulmonary_20260609_prodsweep"
OUT = paths.RESULTS_ROOT / "handover/pah_pulmonary_paper_20260611"
```

- [ ] **Step 2: Repoint `export_production_sweep_for_animation.py`**

Replace:

```python
REPO = Path(__file__).resolve().parent
SWEEP_ROOT = REPO / "results/sims/2026-05-10/capped_shared_l5_20260510_141015"
DEFAULT_OUT = REPO / "paraview_exports/production_capped_sweep_ed"
```

with:

```python
import paths
SWEEP_ROOT = paths.RESULTS_ROOT / "sims/2026-06-09/pah_pulmonary_20260609_prodsweep"
DEFAULT_OUT = paths.REPO_ROOT / "paraview_exports/pah_pulmonary_ed"
```

(Task 5 finishes the rest of this file's pulmonary repointing; this step only fixes the root constants so imports resolve.)

- [ ] **Step 3: Verify imports resolve**

Run: `python -c "import ast; ast.parse(open('pah_pulmonary_batch/make_pah_handover.py').read()); ast.parse(open('export_production_sweep_for_animation.py').read()); print('parse OK')"`
Expected: `parse OK`

Run: `python -c "import paths; print(paths.RESULTS_ROOT)"`
Expected: prints `<repo>/results`

- [ ] **Step 4: Re-run the handover generator end-to-end (still pre-move, symlink not yet made — uses repo/results directly)**

Run: `python pah_pulmonary_batch/make_pah_handover.py 2>&1 | tail -5`
Expected: prints the `r summary` block, no traceback; figures regenerate under `results/handover/pah_pulmonary_paper_20260611/`.

- [ ] **Step 5: Commit**

```bash
git add pah_pulmonary_batch/make_pah_handover.py export_production_sweep_for_animation.py
git commit -m "refactor: resolve results root via paths.RESULTS_ROOT (user-agnostic)"
```

---

## Task 3: Move `results/` to the shared dir + symlink (the delicate one)

This is a one-time operational runbook, not unit-testable. Execute deliberately; each step is verified before the next. **Rollback** at any point: `mv /global/D1/cardiac_rv_shared/results <repo>/results && rm -f <repo>/results` (if symlink already made, remove it first).

**Files:** none in the repo (filesystem operation). Shared dir name: `/global/D1/cardiac_rv_shared` (override by editing the `SHARED` var below if you chose a different name).

- [ ] **Step 1: Preconditions — verify nothing is running and `results/` is a real dir**

Run:
```bash
squeue -u "$USER" -h | wc -l          # MUST be 0
cd /global/D1/homes/dtsteene/cardiac-work
[ -d results ] && [ ! -L results ] && echo "results is a real dir: OK" || echo "ABORT: results missing or already a symlink"
```
Expected: queue count `0`, and `results is a real dir: OK`. If not, STOP.

- [ ] **Step 2: Create the shared dir with the sharing group + setgid**

Run:
```bash
SHARED=/global/D1/cardiac_rv_shared
mkdir -p "$SHARED"
chgrp cppm_via_users "$SHARED"
chmod 2770 "$SHARED"                    # setgid + rwxrwx--- ; group cppm_via_users gets rwx
ls -ld "$SHARED"
```
Expected: `drwxrws--- ... dtsteene cppm_via_users ... /global/D1/cardiac_rv_shared`

- [ ] **Step 3: Atomic move (same FS rename — instant, no copy)**

Run:
```bash
mv /global/D1/homes/dtsteene/cardiac-work/results "$SHARED/results"
ls -ld "$SHARED/results" && ls /global/D1/homes/dtsteene/cardiac-work/results 2>&1 | head -1
```
Expected: `$SHARED/results` exists; the old path now errors `No such file or directory` (it's gone — about to be symlinked).

- [ ] **Step 4: Symlink back so repo-relative paths keep working**

Run:
```bash
cd /global/D1/homes/dtsteene/cardiac-work
ln -s "$SHARED/results" results
readlink results
ls results/sims | head -3                # reads through the symlink
```
Expected: `readlink` prints `/global/D1/cardiac_rv_shared/results`; `ls` shows sim dirs.

- [ ] **Step 5: Open group permissions on the moved tree (slow recursive metadata pass — run in background)**

Run:
```bash
SHARED=/global/D1/cardiac_rv_shared
nohup bash -c "
  chgrp -R cppm_via_users '$SHARED/results' &&
  chmod -R g+rwX '$SHARED/results' &&
  find '$SHARED/results' -type d -exec chmod g+s {} + &&
  echo DONE_PERMS
" > /tmp/share_perms.log 2>&1 &
echo "perms running in background; tail /tmp/share_perms.log"
```
Expected: backgrounded; later `tail /tmp/share_perms.log` ends with `DONE_PERMS`.

- [ ] **Step 6: Verify sharing — group, setgid, and a fresh file inherits the group**

Run (after `DONE_PERMS`):
```bash
SHARED=/global/D1/cardiac_rv_shared
stat -c '%A %U %G' "$SHARED/results"                 # expect drwxrws--- dtsteene cppm_via_users
touch "$SHARED/results/.share_probe" && stat -c '%G' "$SHARED/results/.share_probe"
rm -f "$SHARED/results/.share_probe"
```
Expected: dir is `drwxrws--- dtsteene cppm_via_users`; probe file group is `cppm_via_users` (setgid inheritance works).

- [ ] **Step 7: Confirm the repo still finds data via the symlink**

Run: `python -c "import paths; p=paths.RESULTS_ROOT/'sims/2026-06-09/pah_pulmonary_20260609_prodsweep'; print(p.exists(), len(list(p.glob('*/case*'))))"`
Expected: `True` and a nonzero case count.

- [ ] **Step 8: Document for henriknf (handled in Task 8 WORKFLOW.md) — no commit here**

Nothing is committed in this task (filesystem only). The symlink itself is inside `results/` which is gitignored, so git sees no change. Note completion in the worktree log / tell the user.

---

## Task 4: Dual-frame strain in the loop figures

Add a clinical/unloaded reference toggle and emit two figure sets. The metrics' `mean_E_*` are raw Green–Lagrange strain relative to the unloaded reference, so "unloaded" = raw, "clinical" = subtract the last-beat ED (most-stretched) value.

**Files:**
- Modify: `pah_pulmonary_batch/make_pah_handover.py` (`fig_stress_pressure_strain`, and its call in `main`)
- Test: `tests/test_strain_frame.py`

- [ ] **Step 1: Write the failing test for the frame helper**

```python
# tests/test_strain_frame.py
import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pah_pulmonary_batch"))
import numpy as np
import make_pah_handover as mph

def test_unloaded_frame_is_raw():
    E = np.array([0.10, 0.06, 0.02, 0.07])
    np.testing.assert_allclose(mph.frame_strain(E, "unloaded"), E)

def test_clinical_frame_zeros_at_ed_max():
    E = np.array([0.10, 0.06, 0.02, 0.07])   # ED = most stretched = 0.10
    out = mph.frame_strain(E, "clinical")
    assert out.max() == 0.0
    np.testing.assert_allclose(out, E - 0.10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_strain_frame.py -v`
Expected: FAIL — `AttributeError: module 'make_pah_handover' has no attribute 'frame_strain'`

- [ ] **Step 3: Add the helper and parameterize the loop figure**

Add near the top of `make_pah_handover.py` (after imports):

```python
def frame_strain(E, frame):
    """Reference-shift a Green-Lagrange strain trace.
    'unloaded' = raw E (relative to the stress-free reference);
    'clinical' = re-zeroed at end-diastole (most-stretched instant)."""
    E = np.asarray(E, float)
    return E - E.max() if frame == "clinical" else E
```

Change `fig_stress_pressure_strain(df, bundle, out)` to take a `frame` argument and use the helper. Replace the two strain lines:

```python
            Eff = np.asarray(m[f"mean_E_ff_{reg}"])[lbm]; Eff = Eff - Eff.max()
            ...
            Ell = np.asarray(m[f"mean_E_ll_{reg}"])[lbm]; Ell = (Ell - Ell.max()) * 100.0
```

with:

```python
            Eff = frame_strain(np.asarray(m[f"mean_E_ff_{reg}"])[lbm], frame)
            ...
            Ell = frame_strain(np.asarray(m[f"mean_E_ll_{reg}"])[lbm], frame) * 100.0
```

Update the signature, the suptitle to name the frame, and the savefig path:

```python
def fig_stress_pressure_strain(df, bundle, out, frame):
    ...
    fig.suptitle(f"Coupled fibre stress-strain and pressure-strain loops "
                 f"({frame} frame: strain {'zeroed at ED' if frame=='clinical' else 'from unloaded reference'})",
                 fontsize=12)
    savefig(fig, out / "loops_stress_pressure_strain")
```

- [ ] **Step 4: Wire both frames into `main`**

In `main`, replace the single call

```python
        fig_pv_coupled(df, bundle, qdir); fig_pv_0d(df, bundle, qdir); fig_stress_pressure_strain(df, bundle, qdir)
```

with:

```python
        fig_pv_coupled(df, bundle, qdir); fig_pv_0d(df, bundle, qdir)
        for frame in ("clinical", "unloaded"):
            fdir2 = qdir / frame; fdir2.mkdir(parents=True, exist_ok=True)
            fig_stress_pressure_strain(df, bundle, fdir2, frame)
```

- [ ] **Step 5: Run unit test + the generator**

Run: `python -m pytest tests/test_strain_frame.py -v`
Expected: PASS (2 passed)

Run: `python pah_pulmonary_batch/make_pah_handover.py 2>&1 | tail -3`
Expected: no traceback.

Run: `for b in no_frank_starling frank_starling_preload frank_starling_relax; do ls results/handover/pah_pulmonary_paper_20260611/$b/circulation/clinical/loops_stress_pressure_strain.png results/handover/pah_pulmonary_paper_20260611/$b/circulation/unloaded/loops_stress_pressure_strain.png; done`
Expected: all 6 files listed (clinical + unloaded for each bundle).

- [ ] **Step 6: Eyeball one unloaded-frame figure**

Read `results/handover/pah_pulmonary_paper_20260611/no_frank_starling/circulation/unloaded/loops_stress_pressure_strain.png`.
Expected: the ED point sits at **positive** strain (~+5–10 %), systole moves toward 0 — confirming the unloaded reference (vs clinical where ED is at 0).

- [ ] **Step 7: Commit**

```bash
git add pah_pulmonary_batch/make_pah_handover.py tests/test_strain_frame.py
git commit -m "feat: dual-frame (clinical + unloaded) stress/pressure-strain loops"
```

---

## Task 5: ED-static PVD exporter — repoint to the pulmonary sweep, per bundle, with density fields

Finish adapting `export_production_sweep_for_animation.py`: bundle-agnostic, derive ED time from the run instead of hardcoding 4.0 s, and add work-density / pressure-strain-density cell fields (per-cell J ÷ cell volume).

**Files:**
- Modify: `export_production_sweep_for_animation.py`
- Modify: `sbatch/jobs/export_production_sweep_for_animation.sbatch`

- [ ] **Step 1: Make the case list + ED time bundle-driven**

Replace the hardcoded `SEVERITIES`/`ED_TIME_S` with CLI/derived values:

```python
import argparse, json
CASES = ["case0_rv25","case1_rv35","case2_rv45","case3_rv55",
         "case4_rv65","case5_rv75","case6_rv85","case7_rv95"]

def ed_time_for(case_dir: Path) -> float:
    """Start of the final beat = (n_beats-1)/HR, read from simulation_params.json."""
    sp = json.loads((case_dir / "simulation_params.json").read_text())
    bpm = float(sp.get("BPM", sp.get("bpm", 75)))
    beats = int(sp.get("BEATS", sp.get("beats", 6)))
    return (beats - 1) * 60.0 / bpm
```

Add `--bundle` (default `no_frank_starling`) and iterate `CASES` for `SWEEP_ROOT/<bundle>/<case>`, writing to `DEFAULT_OUT/<bundle>/`.

- [ ] **Step 2: Add density fields**

Where per-cell J fields are attached to `grid.cell_data`, also attach densities (J/m³) using `cell_volumes` already loaded from the npz:

```python
        vol = cell_volumes  # m^3, from per_cell_data.npz
        for npz_key, vtu_name, kind in PER_CELL_J_FIELDS:
            data = pcd[npz_key][perm]              # existing J field, permuted to mesh order
            grid.cell_data[vtu_name] = data
            grid.cell_data[vtu_name.replace("_J", "_density_Jm3")] = data / np.where(vol > 0, vol, np.nan)
        for proxy_key in ("proxy_PLV_ll","proxy_PRV_ll","proxy_Trans_ll","proxy_Mean_ll","proxy_Sum_ll"):
            if proxy_key in pcd.files:
                grid.cell_data[f"{proxy_key}_density_Jm3"] = pcd[proxy_key][perm] / np.where(vol > 0, vol, np.nan)
```

(`perm` is the existing centroid-KDTree permutation from npz order to serial mesh order.)

- [ ] **Step 3: Make the sbatch wrapper bundle-aware**

Edit `sbatch/jobs/export_production_sweep_for_animation.sbatch` so it passes `--bundle "$BUNDLE"` (default `no_frank_starling`) to the script and sets a per-job `XDG_CACHE_HOME` (mirror `run_per_cell.sbatch`).

- [ ] **Step 4: Run the exporter for all three bundles (login is OK: single-timestep mesh read, serial pyvista — but if it imports dolfinx MPI, submit via sbatch)**

Run:
```bash
for b in no_frank_starling frank_starling_preload frank_starling_relax; do
  sbatch --partition=mi50q --export=ALL,BUNDLE=$b sbatch/jobs/export_production_sweep_for_animation.sbatch
done
squeue -u "$USER"
```
Expected: 3 jobs queued.

- [ ] **Step 5: Verify output**

Run (after completion): `find paraview_exports/pah_pulmonary_ed -name "*.vtu" | wc -l; ls paraview_exports/pah_pulmonary_ed/*/sweep.pvd`
Expected: 24 VTUs (8 × 3 bundles); a `sweep.pvd` per bundle.

Run: `python -c "import pyvista as pv; g=pv.read(sorted(__import__('glob').glob('paraview_exports/pah_pulmonary_ed/no_frank_starling/ed_meshes/*.vtu'))[0]); print([k for k in g.cell_data.keys() if 'density' in k][:6])"`
Expected: lists `w_total_density_Jm3`, `proxy_*_ll_density_Jm3`, etc.

- [ ] **Step 6: Commit**

```bash
git add export_production_sweep_for_animation.py sbatch/jobs/export_production_sweep_for_animation.sbatch
git commit -m "feat: ED-static PVD export for the pulmonary sweep (per bundle, density fields)"
```

---

## Task 6: Through-beat PVD exporter — baseline + severe, cumulative densities

New exporter that replays the last beat from `solver/checkpoint.bp`, accumulating per-cell work and pressure-strain, and writes a time-series PVD of the deforming mesh. Reuses `compute_per_cell.py` forms/replay; runs only for `case0_rv25` and `case7_rv95` per bundle.

**Files:**
- Create: `export_beat_animation.py`
- Create: `sbatch/jobs/export_beat_animation.sbatch`

- [ ] **Step 1: Write the exporter skeleton (reusing compute_per_cell forms)**

```python
#!/usr/bin/env python3
"""Through-beat PVD animation for two cases (baseline + severe) of a bundle.

Replays the last beat from the displacement checkpoint, accumulating per-cell
work density (S:dE) and pressure-strain density (P*deps_ll) from ED, and writes
a deforming-mesh time series: paraview_exports/pah_pulmonary_beat/<bundle>/<case>/.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pyvista as pv
import paths
# reuse the replay scaffolding compute_per_cell already builds
import compute_per_cell as pc   # exposes form builders / pressure reader (refactor target below)

# CLI: --bundle, --cases (default the two), --sweep (default the prodsweep)
```

If `compute_per_cell.py` does not already expose its per-step assembly as importable
functions, add a thin module-level function `assemble_step(problem, forms, step_idx)`
returning `(dw_total_percell, dw_ff_percell, dproxy_percell_dict, u_vec)` and call it
from both `compute_per_cell.main` and this exporter (DRY — do not copy the loop body).

- [ ] **Step 2: Implement the accumulation + PVD writing**

```python
def export_case(sweep_root, bundle, case, out_root):
    cd = sweep_root / bundle / case
    forms, problem, mesh, cell_vols, timestamps, lb = pc.prepare_replay(cd)  # last-beat slice
    out = out_root / bundle / case; out.mkdir(parents=True, exist_ok=True)
    cum_w = np.zeros(mesh.topology.index_map(mesh.topology.dim).size_local)
    cum_ps = np.zeros_like(cum_w)
    collection = []
    for k, step in enumerate(range(lb.start, lb.stop)):
        dW, dPS, u_vec = pc.assemble_step(problem, forms, step)
        cum_w += dW; cum_ps += dPS
        grid = pc.deformed_grid(mesh, u_vec)
        grid.cell_data["cum_work_density_Jm3"] = cum_w / np.where(cell_vols>0, cell_vols, np.nan)
        grid.cell_data["cum_ps_density_Jm3"]   = cum_ps / np.where(cell_vols>0, cell_vols, np.nan)
        fn = out / f"step_{k:03d}.vtu"; grid.save(fn)
        collection.append((k, fn.name))
    pc.write_pvd(out / "beat.pvd", collection)  # phase as time
```

- [ ] **Step 3: Add the sbatch wrapper**

`sbatch/jobs/export_beat_animation.sbatch` mirrors `run_per_cell.sbatch` (8 ranks, per-job `XDG_CACHE_HOME`, `--partition` overridable) and runs:
`mpirun -n $SLURM_NTASKS python $SIM_DIR/export_beat_animation.py --bundle $BUNDLE`

- [ ] **Step 4: Smoke-test on one bundle/case via sbatch**

Run: `sbatch --partition=mi50q --export=ALL,BUNDLE=no_frank_starling sbatch/jobs/export_beat_animation.sbatch`
Expected: 1 job queued; on completion `paraview_exports/pah_pulmonary_beat/no_frank_starling/case0_rv25/beat.pvd` exists with `step_*.vtu` files.

- [ ] **Step 5: Verify the animation fields + monotone accumulation**

Run:
```bash
python -c "
import pyvista as pv, glob, numpy as np
fs=sorted(glob.glob('paraview_exports/pah_pulmonary_beat/no_frank_starling/case0_rv25/step_*.vtu'))
tot=[float(np.nansum(pv.read(f).cell_data['cum_work_density_Jm3'])) for f in (fs[0],fs[len(fs)//2],fs[-1])]
print('frames',len(fs),'cum work (start,mid,end)',[round(t,3) for t in tot])
"
```
Expected: ≥ ~150 frames; cumulative work grows start→end (monotone-ish).

- [ ] **Step 6: Run the remaining bundles**

Run: `for b in frank_starling_preload frank_starling_relax; do sbatch --partition=mi50q --export=ALL,BUNDLE=$b sbatch/jobs/export_beat_animation.sbatch; done`
Expected: 2 jobs; outputs under each bundle for `case0_rv25` and `case7_rv95`.

- [ ] **Step 7: Commit**

```bash
git add export_beat_animation.py sbatch/jobs/export_beat_animation.sbatch compute_per_cell.py
git commit -m "feat: through-beat PVD animation export (baseline + severe, cumulative densities)"
```

---

## Task 7: Rewrite `CLAUDE.md` around the actual workflow

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace the stale objective + file table**

Edit `CLAUDE.md`:
- Replace the "Current Objective: Wall Thickness Variants" section with a "Current Objective: PAH pulmonary-windkessel pressure-proxy study" section (sweep `R↑/C↓` at conserved RC over RV systolic 25→95; 3 active-model bundles; one shared 8/5 unloaded mesh; canonical tagging).
- In the file tables: **remove** the six dead rows (`compare_cases.py`, `compare_spectrum.py`, `generate_thickness_variants.py`, `investigate_wall_thickness.py`, `septum_editor.py`, `compare_mesh_geometry.py`); **add** rows for `pah_pulmonary_batch/` (study home: `make_baseline.py`, `make_sweep_params.py`, `sweep_pulmonary_0d.py`, `compare_baselines_0d.py`, `diagnose_linear_baseline.py`, `submit_pah_pulmonary_sweep.sh`, `make_pah_handover.py`), `analysis_core.py`, `clinical_frame.py`, `sweep_analysis.py`, `paths.py`, `export_production_sweep_for_animation.py`, `export_beat_animation.py`.
- Keep the accurate "Critical Bugs", "LDRB", "Metric Naming", and "Simulation→Postprocessing paradigm" sections.
- Add a one-line "Results live in a shared dir; `results/` is a symlink — see WORKFLOW.md" note.

- [ ] **Step 2: Verify no dead file references remain**

Run:
```bash
for f in compare_cases.py compare_spectrum.py generate_thickness_variants.py investigate_wall_thickness.py septum_editor.py compare_mesh_geometry.py; do
  grep -q "$f" CLAUDE.md && echo "STILL REFERENCED: $f" || true
done
echo "check done"
# every backticked .py named in CLAUDE.md must exist:
grep -oE '[A-Za-z0-9_/]+\.py' CLAUDE.md | sort -u | while read p; do [ -e "$p" ] || [ -e "$(basename $p)" ] || echo "MISSING: $p"; done
```
Expected: no `STILL REFERENCED` lines; no `MISSING` lines.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: rewrite CLAUDE.md around the pulmonary-sweep workflow; drop dead refs"
```

---

## Task 8: Add `WORKFLOW.md` (end-to-end pipeline + collaborator setup)

**Files:**
- Create: `WORKFLOW.md`

- [ ] **Step 1: Write the pipeline walkthrough**

Create `WORKFLOW.md` covering, in order, with the exact entry-point commands:
1. **Tune circulation** — `pah_pulmonary_batch/make_baseline.py` (linear-EDPVR baseline), `make_sweep_params.py` (RV-even cases), inspect with `sweep_pulmonary_0d.py` / `compare_baselines_0d.py`. State that the Optuna optimiser is intentionally **not** in this repo.
2. **0D convergence + pre-coupling PV loops** — where the 0D warm-up + PV check live.
3. **Coupled simulation** — `submit_pah_pulmonary_sweep.sh` → `complete_cycle.py` (saves displacement checkpoint + Ta + pressure history).
4. **Fields** — `compute_per_cell.py` (work density S:dE, ff/ss/nn decomposition, ll/ff/circ/radial strain, regional PLV/PRV/Trans/Mean/Sum proxies) / `postprocess_metrics.py`.
5. **Sweep metrics** — `make_pah_handover.py` → per-bundle `correlation/` (primary), `ratio/`, dual-frame `circulation/{clinical,unloaded}/`.
6. **Animation export** — `export_production_sweep_for_animation.py` (ED static, all cases) and `export_beat_animation.py` (through-beat, baseline+severe).

- [ ] **Step 2: Add the collaborator (henriknf) setup section**

Document:
```bash
# Shared data lives at /global/D1/cardiac_rv_shared/results (group cppm_via_users).
git clone <repo>            # or copy the repo
cd cardiac-work
ln -s /global/D1/cardiac_rv_shared/results results   # OR: export CARDIAC_RESULTS_ROOT=/global/D1/cardiac_rv_shared/results
python -c "import paths; print(paths.RESULTS_ROOT)"  # sanity check
```
Note: henriknf is in `cppm_via_users`, so read+write is automatic; ACLs are unsupported here, sharing is via that group + setgid.

- [ ] **Step 3: Verify every command/script named exists**

Run: `grep -oE '[A-Za-z0-9_./-]+\.(py|sh)' WORKFLOW.md | sort -u | while read p; do [ -e "$p" ] || echo "MISSING: $p"; done; echo done`
Expected: no `MISSING` lines.

- [ ] **Step 4: Commit**

```bash
git add WORKFLOW.md
git commit -m "docs: add WORKFLOW.md (end-to-end pipeline + collaborator setup)"
```

---

## Final verification (run after all tasks)

- [ ] **All tests pass:** `python -m pytest tests/ -q` → no failures.
- [ ] **Handover regenerates clean:** `python pah_pulmonary_batch/make_pah_handover.py` → no traceback; `correlation/`, `ratio/`, `circulation/clinical/`, `circulation/unloaded/` all populated for 3 bundles.
- [ ] **Results symlink intact:** `readlink results` → `/global/D1/cardiac_rv_shared/results`; `stat -c '%G' results/` → `cppm_via_users`.
- [ ] **No dead doc refs:** the Task 7 / Task 8 grep checks report nothing missing.
- [ ] **Working tree clean of junk:** `git status --short` shows only intended files; large data + diagnostics remain gitignored.
