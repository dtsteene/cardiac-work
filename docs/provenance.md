# Data provenance

Where the results live, what survives, and what does not.

## Where everything is

All data lives under one group-readable shared root, `/global/D1/cardiac_rv_shared/`
(group `cppm_via_users`, setgid so new files inherit it):

| Path | Contents |
|---|---|
| `data/` | meshes and 0D circulation parameter JSONs |
| `results/` | simulation outputs, analysis products, figures (~2.5 TB) |
| `src/` | the four editable-installed packages (fenicsx-pulse, fenicsx-ldrb, fenicsx-warp, circulation) |
| `env/` | ready-to-activate conda-pack of the working environment |
| `cardiac-work/` | this repository |

Inside the repo, `data` and `results` are symlinks into that shared root, so
paths resolve identically for every group member. `paths.py` resolves the
results root, honouring `$CARDIAC_RESULTS_ROOT` if set.

**`/global/D1` is at 100% capacity** (~1.1 TB free of 515 TB). Plan any new
sweep around that.

## The canonical thesis raw data was deleted

The sweep that `results/sims/SWEEPS_INDEX.md` calls *"CANONICAL — primary thesis
sweep"* — `sims/2026-05-10/capped_shared_l5_20260510_141015/`, 16 cases with a
5.0 mmHg RV-EDP cap — **no longer exists on disk**. Neither does
`sims/2026-05-12/shared_unloaded_l5_20260512_175915/`. These are the only two
May sweep directories missing, and they are the two that mattered most.

`results/DELETED_pre_may_manifest.txt` records roughly 890 GB of *April*
deletions and says nothing about these two, so they went in a later cleanup
without a manifest entry. The `sims/_CURRENT_H5_PRODUCTION/` index is likewise
dead: all 16 of its symlinks dangle, pointing through the old
`/home/dtsteene/D1/...` path at deleted April directories.

### What survives, and why it is still enough

The derived analysis survived intact:
`results/analysis/capped_shared_l5_sweep_20260510_141015/`, 47 files and 30 MB,
containing every headline CSV (`h5_sweep_case_values.csv`,
`h5_septum_proxy_correlations.csv`, `h5_septum_ratio_preservation.csv`,
`h5_freewall_ratio_summary.csv`), the summary markdown, all figures, and
critically `capped_shared_l5_cases.tsv` — the case definitions.

So the thesis numbers are **defensible as analysis** and every published figure
can be traced to a surviving CSV. What is not possible without re-simulating is
recomputing a *new* metric from the original displacement fields, because the
checkpoints are gone. Because the case definitions survive, re-running is
possible in principle; it is a compute and disk-space question, not a lost-input
one.

Verified reference values for that sweep, for cross-checking any re-run:
sPAP70 RV_ESP = 85.83 mmHg; free-wall LV r = 0.994, RV r = 0.967; septum ratio
preservation 0.805 / 1.171 / 2.075 / 0.969 for PLV / PRV / Trans / Mean.

## Reading older runs

The cleanup of 2026-08-31 removed the compatibility code that read pre-June
metric key names (`work_fiber_*` → `work_ff_*`) and the older results-directory
layouts. To read a run in the old on-disk layout:

```bash
git checkout pre-cleanup-2026-08-31
```

That tag holds the full pipeline exactly as it was before the cleanup,
including every compatibility path that was removed.

Runs from before 2026-03-08 additionally lack `pressure_history.npy` and so
cannot have boundary work recomputed correctly — the 3D cavity Lagrange
multiplier pressure differs from the 0D Windkessel pressure by 30–60%, and the
solver pressure is the correct one for boundary work.

## Frank-Starling metadata

Runs predating 2026-06-07 have no `frank_starling.enabled` key in
`simulation_params.json`. `postprocess_metrics.py` assumes Frank-Starling was
**on** for those, which is correct: every surviving pre-toggle run belongs to
the frankstarling sweeps. It now warns when it makes that assumption rather than
making it silently, because guessing wrong changes the replayed physics.

## per_cell_data.npz and the AHA tags

Existing `per_cell_data.npz` files written before 2026-07-08 contain an
`aha_tag` array. Re-running `compute_per_cell.py` today will **not** reproduce
it, and that is deliberate rather than a regression.

`gernerate_aha_biv` mis-segments on the adios checkpoint mesh + ffun pair — it
returns almost-all-apical garbage — and running it on the geometry-dir mesh
instead would land in a different cell ordering than the per-cell arrays. So
commit `7780397` removed the in-line computation. AHA tags now come from
`compute_aha_band.py`, which runs the segmentation on the geometry mesh and maps
cells onto the npz ordering via the canonical `ckpt_to_cg_idx` permutation,
writing an `aha_tags.npy` sidecar next to the npz. `sbatch/jobs/run_aha_band.sbatch`
does the backfill.

Practical consequence: if you regenerate a `per_cell_data.npz`, regenerate its
`aha_tags.npy` too, or any band-restricted analysis (`make_pah_handover.py`
`band="mid"`) will be reading a stale sidecar.
