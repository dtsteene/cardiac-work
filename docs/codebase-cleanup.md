# Codebase cleanup, 2026-08-31

A record of what changed in the handover cleanup, why, and what is still
outstanding. Relevant if you are reading code that looks different from an
older note, or wondering whether a behaviour change was deliberate.

**Escape hatch:** everything removed below is preserved at the tag
`pre-cleanup-2026-08-31`. To read a run in a pre-June on-disk layout, or to see
any deleted compatibility path:

```bash
git checkout pre-cleanup-2026-08-31
```

## Verification

The cleanup was proven behaviour-preserving by re-running the pipeline on a
canonical case (`2026-06-22/pah_pulmonary_fixedratio/no_frank_starling/case5_rv75`)
with the cleaned code and diffing every output against the production results.
All three runs came back **bitwise identical — largest relative difference
0.000e+00**, not merely within tolerance:

| Job | Covered | Result |
|---|---|---|
| 1377179 | `postprocess_metrics.py` after the legacy demolition | 132/132 metric keys identical |
| 1377183 | `postprocess_metrics.py` at the final state (sim_params, flag removal, `__init__` split) | 132/132 metric keys identical |
| 1377184 | `compute_per_cell.py` in canonical tagging mode | 46/46 numeric arrays identical |

Exact equality is the expected outcome if the deleted branches really were dead
and the extractions really were pure, so it is the result that carries the
claim. The only structural difference found was an `aha_tag` array present in
the production `per_cell_data.npz` and absent from a fresh run — traced to
commit `7780397` (2026-07-08), unrelated to this cleanup, and documented in
[provenance](provenance.md#per_cell_datanpz-and-the-aha-tags).

Verification outputs were written to separate subdirectories
(`--metrics-subdir`, `--output-tag`) so production results were never touched.

## Two latent bugs, both in regional wall volumes

Neither fired in production — verified: every canonical run carries
`region_volume_*_tau_lap` and physical ~1e-4 m³ volumes, not 1.0. But both would
have fired for a collaborator on a freshly built environment, and both would
have corrupted results *silently*.

**The volumes could be replaced by 1.0 m³.** Regional wall volumes scale the
pressure-strain indices, so a volume of 1.0 instead of ~1e-4 shifts work density
by about four orders of magnitude. Two independent paths could do it:

1. An `else` branch commented "Fallback if no tags" was attached to the
   `tau_tags_lap` check rather than to `region_tags`. Whenever the Laplace tau
   diagnostic was unavailable — no `petsc4py`, or missing LV/RV endocardial
   markers — it overwrote the LV/RV/Septum/Whole volumes that had just been
   integrated correctly fifteen lines above.
2. The enclosing `except Exception` replaced `region_volumes` with
   `defaultdict(lambda: 1.0)`. Being a defaultdict, it also removed the
   `KeyError` that would otherwise have exposed a missing region.

Both are gone; failures there now propagate. This is the clearest argument in
the codebase for why broad exception handlers around numerical setup are worse
than no handler at all: the fallback did not make the code robust, it made a
wrong answer look like a right one.

## Other behaviour changes worth knowing

`plot_utils.load_metrics` now raises `FileNotFoundError` instead of returning
`None`. Both callers mishandled the `None`: `plot_loops` passed it straight into
plotting, and `eval_proxies` exited 0 having silently done nothing.

`run_postprocessing` no longer falls back to a 0.8 s cycle length when
`circulation/parameters.json` is malformed. Cycle length drives beat
segmentation, so a wrong value silently corrupts every per-beat metric; a
missing file (a run with no 0D model) still uses the default, with a warning.

`eval_proxies` reports a proxy/truth ratio of `NaN`, not `0.0`, when there is no
true work to divide by. Zero read as "the proxy captures none of the work"; the
ratio is in fact undefined.

`MetricsCalculator` no longer emits `debug_Ta_internal_max = 0.0` when the
activation read fails. Zero activation is a physically meaningful value, so the
key is suppressed instead — the convention the surrounding code already used
for `debug_S_active_max`.

`postprocess_metrics` still assumes Frank-Starling was **on** for runs predating
the 2026-06-07 toggle, which is correct (every surviving pre-toggle run is from
the frankstarling sweeps), but now warns rather than assuming silently.

## Removed

The metric-key normaliser (`work_fiber_*` → `work_ff_*`) and the
results-directory auto-migration in `run_postprocessing` — canonical runs use
only current key names, verified against the data.

The four-layout metrics search was narrowed to `<folder>/metrics/`, and **that
went too far**. It was verified against the run *root*, but `run_postprocessing`
passes `plot_loops` and `eval_proxies` the per-beat directories
(`analysis/last_beat`, `analysis/all_beats`) where the metrics file sits flat.
Both raised `FileNotFoundError` on every run, and because the subprocess calls
used `check=False` the jobs still reported COMPLETED while producing no figures
and no proxy stats — the 2026-08-31 spectrum pilot lost all eight per-beat
outputs. Fixed in `9718cc0`: `load_metrics` now tries `<folder>/metrics/` then
`<folder>` itself, `run_postprocessing` propagates subprocess exit codes so a
broken plot step fails the Slurm job, and `tests/test_plot_utils.py` covers both
layouts. The lesson worth keeping: the verification runs exercised
`postprocess_metrics.py` directly and never `run_postprocessing.py`, so a whole
branch of the pipeline was unverified while the metric comparison looked
perfect.

`--tag-at-unloaded`, which tagged the septum on the unloaded reference mesh.
Its own help text called this "anatomically meaningless"; it existed only to
reproduce old runs whose raw data no longer exists.

`--skip-research`, plumbed from the CLI through the `MetricsCalculator`
constructor into an attribute that nothing read. It gated no code in any file.

The `[N,3]` region-aware Ta history branch. Ta histories are uniform scalars,
shape `[N]`, in every surviving run.

`plot_utils.get_array` and `total_work`, which nothing called.

## Restructured

`sim_params.py` now owns the `simulation_params.json` material-parameter
reconstruction, which `postprocess_metrics` and `compute_per_cell` each carried
a verbatim copy of — a duplication that had already forced the same
regional-scaling bug to be fixed twice. Verified byte-equivalent to the previous
inline logic across six runs from three campaigns.

`MetricsCalculator.__init__` went from 226 lines to 35 by extracting four named
setup steps. Verified a pure extraction: all 64 `self.<attr>` assignments occur
in identical order, and a statement-level diff shows nothing lost.

`tests/test_syntax.py` was rewritten. It had hardcoded four filenames, two of
them deleted in the June cleanup, so it had been exiting 1 since then while
covering two of twenty-two modules. It now parses every module in the repo.

## Still outstanding

Two files remain hard to read and were deliberately left alone, because
restructuring them is the riskiest change available and each verification cycle
costs a cluster run:

- `compute_per_cell.py` is a single 1414-line `_main`.
- `postprocess_metrics.py` is 955 lines of module-level script with no
  functions at all.

Nine broad `except Exception` handlers remain. The ones in `metrics_calculator`
were reviewed and kept deliberately: they degrade an optional *diagnostic*
(scipy/petsc4py imports, the Laplace solve) with a clear message, and are
harmless now that the volume clobber is gone. The rest are in peripheral export
scripts.
