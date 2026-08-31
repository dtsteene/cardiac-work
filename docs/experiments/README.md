# Experiment registry

Every simulation campaign worth knowing about, and its standing. Campaigns not
listed here were exploratory and are not load-bearing for any result.

Status means:

- **Canonical** — a live result. Quote it.
- **Superseded** — correct when run, replaced by something later. Kept for
  provenance; do not quote.
- **Dead** — failed, partial, or a test artifact.

## Registry

| Campaign | Path | Cases | Status | Page |
|---|---|---|---|---|
| Capped shared-unloaded L5 sweep | `analysis/capped_shared_l5_sweep_20260510_141015/` (raw data deleted) | 16 | **Canonical (thesis)** | [thesis-capped-sweep](thesis-capped-sweep.md) |
| Pulmonary afterload, fixed-ratio coupling | `sims/2026-06-22/pah_pulmonary_fixedratio/` | 3 bundles × 8 | **Canonical (current)** | [pulmonary-afterload-sweep](pulmonary-afterload-sweep.md) |
| Passive-softening pilot | `sims/2026-07-08/softmat_pilot_L10/` | 3 stiffnesses × 2 | **Canonical (current)** | [softening-pilot](softening-pilot.md) |
| Mesh convergence h10/h7.5/h5/h3.75 | `analysis/mesh_convergence/` | 4 resolutions | Canonical (appendix) | [supporting-studies](supporting-studies.md) |
| AHA mid-ventricular ring | tags 4/5/6 in `compute_per_cell` | — | Canonical (method) | [supporting-studies](supporting-studies.md) |
| Base-Dirichlet / metric-space sensitivity | `analysis/base_dirichlet_sensitivity/`, `analysis/metric_space_sensitivity/` | — | Canonical (appendix) | [supporting-studies](supporting-studies.md) |
| Pre-cap thesis sweep | `sims/_CURRENT_H5_PRODUCTION/` | 16 | Superseded (and links are dead) | [thesis-capped-sweep](thesis-capped-sweep.md) |
| Pulmonary sweep, generations 1–4 | `sims/2026-06-07`, `2026-06-09` | 8 each | Superseded | [pulmonary-afterload-sweep](pulmonary-afterload-sweep.md) |
| PAH severity spectrum, shared unloaded ref | `sims/2026-08-31/` | 7 severities | **Canonical (newest)** | see the directory README |
| Frank-Starling L5 sweeps | *deleted 2026-08-31* | varied | Superseded, raw data removed | [pulmonary-afterload-sweep](pulmonary-afterload-sweep.md) |
| Cap-sensitivity (3 mmHg) | `analysis/cap_sensitivity_*_20260511_*` (raw deleted) | 6 | Superseded | — |
| `case_NONEXISTENT` | in the fixed-ratio sweep | — | Dead (canary artifact) | — |

## The through-line

The campaigns are not independent; each answered a problem the previous one
exposed.

The **thesis capped sweep** established the core result on 16 cases spanning
sPAP 22–95 mmHg, after an RV-EDP cap fixed an unloading artifact in the
pre-cap runs. It remains the thesis's evidence base.

The **pulmonary afterload sweep** then narrowed to a cleaner question — vary
*only* pulmonary resistance and compliance — and went through four generations
before the coupling was right. The problem was that the 0D warm-up inflated FEM
preload by ~41%; clamping the coupling ratio per case brought that to ~9%, which
is the fixed-ratio sweep that is canonical now.

The **softening pilot** was a response to a specific supervisor challenge
(Espen's, on end-diastolic overlap) rather than an exploration: it tested
whether softer passive material would separate the case-to-case ED points. It
did not, and the [ED-overlap finding](../findings/ed-overlap.md) explains why
that is physics rather than a bug.

The **supporting studies** exist to close off numerical objections — resolution,
boundary conditions, metric storage space — so that the main result cannot be
attributed to a discretisation choice.
