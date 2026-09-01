# cardiac-work

Biventricular finite-element simulations evaluating clinical
pressure-strain proxies against the ground-truth tensor work
`W = ∫ S:dE dV`, with emphasis on the interventricular septum in
pulmonary arterial hypertension (PAH).

Simulation code for the MSc thesis *Pressure Proxies for Biventricular
Myocardial Work: A Finite Element Study* (Daniel Steeneveldt, University of
Oslo). The thesis itself lives at [github.com/dtsteene/RV](https://github.com/dtsteene/RV).

## Start here

Four documents, four jobs. This one is the map.

| I want to… | Go to |
|---|---|
| **Understand what was run and what it showed** | [`docs/`](docs/README.md) — the knowledge base |
| **Set up the environment / cluster access** | [HANDOVER.md](HANDOVER.md) |
| **Run the pipeline end to end** | [WORKFLOW.md](WORKFLOW.md) |
| **Give an AI assistant project context** | [CLAUDE.md](CLAUDE.md) |

New to the project? Read [docs/open-questions.md](docs/open-questions.md) first —
it says what is unresolved, including which numbers should not yet be quoted.

## The one idea worth knowing

**Simulation and analysis are decoupled.** `complete_cycle.py` saves
displacement checkpoints; everything downstream replays them. Changing a metric
means re-running postprocessing, never re-simulating.

```
complete_cycle.py ─▶ compute_per_cell.py ─▶ postprocess_metrics.py ─▶ analysis_core.py ─▶ sweep_analysis.py
  FEM + 0D solver     per-cell S, E, work      regional metrics          per-region stats     cross-sim
  → checkpoint.bp     → per_cell_data.npz      → metrics_*.npy           (proxy, ratio,        correlation
                                                                          correlation, area)   + ratio
```

**If you read three files:** `complete_cycle.py` (the solver),
`analysis_core.py` (the reusable per-region statistics — pure NumPy, no
FEniCSx, fully tested, and the most readable file here), and
`sweep_analysis.py` (the headline result — does the proxy follow the truth
*across* simulations).

## How do I…?

| I want to… | Where |
|---|---|
| Run one simulation + postprocessing | `sbatch sbatch/jobs/run_sim_and_post.sbatch` (details in [WORKFLOW.md](WORKFLOW.md)) |
| Add or change a work / strain metric | `metrics_calculator.py` (FEM), then rerun postprocessing |
| Add or change a proxy statistic | `analysis_core.py` (+ `tests/test_analysis_core.py`) |
| Swap which pressure the septum proxy uses | `analysis_core.pressure_candidates` |
| Get the headline cross-simulation result | `sweep_analysis.py` → `results/analysis/sweep/sweep_*.csv` |
| Sanity-check one sim's proxy loop areas | `eval_proxies.py` |
| Toggle Frank-Starling activation | `USE_FRANK_STARLING` env var (`1`=on default, `0`=constant Ta) |
| Tune the 0D circulation parameters | `pah_pulmonary_batch/` |
| Generate a mesh + fibers + tags | `python3 geometry_generator.py --single ukb -c 5 --output-dir data/shared_ukb_mesh` |

## Layout

```
cardiac-work/
├── docs/            KNOWLEDGE BASE — experiments, findings, open questions, provenance
├── tests/           unit / smoke tests (most run on the login node)
├── sbatch/          jobs/ = one sim or analysis; sweeps/ = multi-job launchers
├── pah_pulmonary_batch/   0D circulation tuning + the canonical sweep tooling
├── data → /global/D1/cardiac_rv_shared/data        (symlink, gitignored)
├── results → /global/D1/cardiac_rv_shared/results  (symlink, gitignored)
│
│   ── pipeline ──
├── complete_cycle.py        coupled FEM + 0D Windkessel simulation driver
├── compute_per_cell.py      checkpoint → per-cell stress / strain / proxy fields
├── postprocess_metrics.py   per-cell → regional metrics (LV / RV / septum)
├── metrics_calculator.py    stress / strain / work decomposition library
├── geometry_generator.py    UKB mesh + LDRB fibers + region tags
├── sim_params.py            simulation_params.json → pulse objects (both replay paths)
├── clinical_frame.py        clinical-frame direction helpers
├── geometry_utils.py        pure point-to-surface geometry primitives
│
│   ── analysis (pure NumPy, no FEniCSx) ──
├── analysis_core.py         THE stats core: per-region work, swappable-pressure
│                            proxy, ratios, correlations, log-MAE
├── sweep_analysis.py        cross-sim proxy↔truth correlation + ratio
├── run_postprocessing.py    beat-slice orchestrator
├── eval_proxies.py          per-sim proxy loop-area + ratio check
├── plot_loops.py            PV / PS / SS debug loops
└── export_*.py              ParaView field exporters → results/paraview_exports/
```

## Tests

Fast, pure-NumPy, login-node safe (no FEniCSx):

```bash
python3 tests/test_analysis_core.py     # per-region stats: correlation, ratio, proxy, density
python3 tests/test_geometry_utils.py    # point-to-surface geometry primitives
python3 tests/test_plot_utils.py        # metrics loading: both layouts, precedence, raising
python3 tests/test_syntax.py            # every module in the repo parses
python3 tests/test_sim_params.py        # material-param deserialisation (skips without pulse)
```

FEM-dependent, needs the env / an allocation:

```bash
python3 tests/test_canonical_tagging.py <sim_dirs>    # invariants across a spectrum
```

## Reproducing thesis figures

The standalone thesis-figure scripts were removed after the defense; their
reusable math lives in `analysis_core.py`. Recover any of them from git:

```bash
git show a0af112:freewall_ratio_proxy_test.py          # view
git checkout a0af112 -- analyze_h5_septum_mechanism.py  # restore
```

`a0af112` is the last commit before the post-defense cleanup. Note that the raw
simulation data for the thesis sweep no longer exists — see
[docs/provenance.md](docs/provenance.md) — though its derived CSVs and figures
do. Chapter 1, 2 and 4 illustrations are generated from the thesis repo
(`RV/scripts/`), not here.

## Environment

`environment.yml` (exact builds, this cluster) and `environment.nobuild.yml`
(portable) pin the full stack; `pip-freeze.txt` is a pip-side reference. On the
Simula cluster there is a ready-to-activate shared env needing no FEniCSx
build — see [HANDOVER.md](HANDOVER.md), which owns all environment setup.

The stack is FEniCSx (`dolfinx`/`ufl`/`basix`/`ffcx`) with
[fenicsx-pulse](https://github.com/finsberg/fenicsx-pulse) for the constitutive
models, [cardiac-geometries](https://github.com/ComputationalPhysiology/cardiac-geometries)
and [fenicsx-ldrb](https://github.com/finsberg/fenicsx-ldrb) for meshing and
fibers, [circulation](https://github.com/ComputationalPhysiology/circulation)
for the closed-loop 0D model, and adios4dolfinx for checkpoint I/O.
