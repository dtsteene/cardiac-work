# Handover — Cardiac RV Proxy Project

Everything a collaborator needs to run this project. Written for Henrik, but
applies to anyone in the `cppm_via_users` group on the Simula cluster.

Read this first, then [WORKFLOW.md](WORKFLOW.md) for the end-to-end pipeline and
[CLAUDE.md](CLAUDE.md) for the scientific context and code map.

---

## TL;DR — jump straight in

Everything (data, results, package sources, and a ready-to-activate conda env)
lives under one shared directory you already have group access to:

```
/global/D1/cardiac_rv_shared/
├── data/      # meshes + 0D circulation params      (repo `data`  → here)
├── results/   # simulation outputs + figures        (repo `results` → here)
├── src/       # the 4 editable-installed packages    (see below)
└── env/       # ready-to-activate conda env          (see below)
```

```bash
# 1. Activate the shared env (no build needed)
source /global/D1/cardiac_rv_shared/env/bin/activate
conda-unpack        # ONE TIME only, first activation, fixes hard-coded paths

# 2. Get the analysis repo
git clone git@github.com:dtsteene/cardiac-work.git
cd cardiac-work
# `data` and `results` are symlinks that already resolve to the shared dir.

# 3. Sanity check
python -c "import fenicsx_pulse, ldrb, circulation, warp; print('env OK')"
python tests/test_paths.py
```

If `source .../env/bin/activate` isn't your style, the env is a standard
conda-pack tree — `conda activate /global/D1/cardiac_rv_shared/env` works too.

---

## The conda environment

The real working env is **Python 3.11.14** with the full FEniCSx stack. It was
built as `~/.conda/envs/RV` on Daniel's account. You get it two ways:

### A) Shared prebuilt env (recommended — no FEniCSx build)
`/global/D1/cardiac_rv_shared/env/` is a `conda-pack` of the RV env. Activate it
(see TL;DR) and run `conda-unpack` once. The 4 editable packages are already
wired to `/global/D1/cardiac_rv_shared/src/*`, so imports resolve with no extra
steps for everyone in the group.

### B) Rebuild from spec (portable / off-cluster)
In the repo root:
- `environment.yml` — full conda export (exact builds; this cluster / linux-64)
- `environment.nobuild.yml` — no build strings (more portable across platforms)
- `pip-freeze.txt` — pip view for reference

```bash
conda env create -f environment.nobuild.yml -n RV
conda activate RV
# then reinstall the 4 editable packages (see next section):
for p in circulation fenicsx-ldrb fenicsx-pulse fenicsx-warp; do
  pip install -e /global/D1/cardiac_rv_shared/src/$p
done
```

FEniCSx builds are fragile — prefer (A) unless you're moving off this cluster.

### Env quirks worth knowing
- **`pip list` is misleading.** It reports stale editable paths
  (`/home/dtsteene/D1/...`, `/global/D1/homes/...`) that don't exist, and an old
  `fenicsx-pulse 0.5.1` (the real source is 0.6.1). The **`.pth` files** in
  `site-packages` are authoritative and point at `/global/D1/cardiac_rv_shared/src/*`.
- **ldrb conda/pip clobber.** `fenicsx-ldrb` was first conda-installed, then
  re-installed editable over the top, so conda thinks some files were deleted.
  Harmless for running; it only means `conda-pack` needs
  `--ignore-missing-files` (already used to build the shared env).

---

## The four editable-installed packages

These are the moving parts — **local edits here silently change results.** All
four are git clones under `/global/D1/cardiac_rv_shared/src/`, and that shared
clone *is* the editable install (import resolves straight to it). All previously
uncommitted working-tree edits have now been committed so nothing is lost.

| Package | Source of truth | Branch @ commit | Owner |
|---|---|---|---|
| `circulation` (0.2.0) | `github.com/dtsteene/circulation` | `fix/update-steady-state` @ `d21b0f2` | Daniel's fork |
| `fenicsx-ldrb` (0.1.18) | `github.com/dtsteene/fenicsx-ldrb` | `geometric-septum-tagging` @ `b016efc` | Daniel's fork |
| `fenicsx-pulse` (0.6.1) | `github.com/finsberg/fenicsx-pulse` | `rv-handover-pin` @ `e9694f6` | **Henrik's upstream** |
| `fenicsx-warp` (0.2.0) | `github.com/ComputationalPhysiology/fenicsx-warp` | `main` @ `a85180a` | Org (see note) |

What was captured in the handover commits:
- **circulation** — v11 target-framework rewrite of `optimize_mesh_circ.py`
  (RV_ESP-driven spectrum, Humbert 2022 risk bands) + `regazzoni2020.py` edits.
  Pushed to the fork.
- **fenicsx-ldrb** — int64 dtype fix in the geometric septum-tagging routine.
  Pushed to the fork.
- **fenicsx-pulse** — clean, at the commit that adds the **Frank-Starling
  activation-lag (relaxation) mode** to `FrankStarlingActiveStress` — this is
  the "Frank-Starling greiene" from Slack. Pinned locally to branch
  `rv-handover-pin` (Daniel has no push rights to `finsberg/fenicsx-pulse`; pull
  it upstream or cherry-pick if you want it there).
- **fenicsx-warp** — incremental-loading (`n_steps`) staged BC in
  `solve_hyperelastic` for large-deformation warping. Committed to the shared
  clone's local `main` + `handover/local-edits`; **not** on a personal remote
  (no write access to the ComputationalPhysiology org, `gh` not available here).
  The shared clone is the source of truth.

---

## Science / repo state (what Henrik asked about)

- The **Frank-Starling fixed-ratio fix** and the **AHA midwall band** work
  (described in Slack) is now committed to `cardiac-work` — previously it was all
  uncommitted, which is why the 3-week-old tree "looked fine": the fix wasn't in
  it. See the git log around this handover.
- **ED-frame vs unloaded-frame strain loops** both generated by
  `pah_pulmonary_batch/make_pah_handover.py` → `results/.../loops/ED/` and
  `loops/unloaded/` (Henrik's unloaded-reference request).
- **Open question (parallel work in progress):** a second, *independent* loading
  axis to break the monotone-collinearity that makes Pearson r saturate at ±0.99
  for everything. The design lives in
  `docs/superpowers/specs/2026-07-08-rv-lv-afterload-grid-design.md` and
  `2026-07-08-agreement-metrics-design.md` (RV×LV afterload grid + agreement
  metrics). That thread is being developed separately and is intentionally left
  uncommitted here.

---

## Housekeeping done in this handover
- 392 `slurm-*.out/.err` files swept from the repo root into `logs/slurm/` and
  gitignored.
- `environment.yml`, `environment.nobuild.yml`, `pip-freeze.txt` added.
- All package working-tree edits committed (see table above).
