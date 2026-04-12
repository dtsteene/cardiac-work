#!/usr/bin/env python3
"""
analyze_sweep.py — t-threshold septum sweep with inter-case Pearson r

The primary analysis script for Question A:
  "Given a septum definition width, how reliably does each pressure proxy
   track changes in septal work across the disease spectrum?"

Sweep: the t-threshold relaxation of the geometric septum definition.
  septum(t) = envelope AND (entry_t < t)
  t = 0: exactly the geometric septum
  t < 0: tighter than geometric (only deepest cells)
  t > 0: wider than geometric (adds cells toward epi)

At each t, sum W_true and each W_proxy over all cells in septum(t), per case.
Compute Pearson r across the 8 spectrum cases.

Proxies (all using longitudinal strain dε_ll, the clinical GLS analogue):
  P_LV × dε_ll,  P_RV × dε_ll,  (P_LV - P_RV) × dε_ll,
  (P_LV + P_RV)/2 × dε_ll,  P_dom × dε_ll (per-cell dominant pressure)

Usage:
  python3 analyze_sweep.py <result_dir1> <result_dir2> ... [--output-dir DIR]
  python3 analyze_sweep.py results/sims/2026-04-12/UKB_1beats_run_*
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("result_dirs", nargs="+", type=Path)
parser.add_argument("--geometry-fields", type=Path, required=True,
                    help="Path to precomputed geometry_fields.npz (from "
                         "precompute_geometry_fields.py). Provides tau, envelope, "
                         "entry_t — guaranteed identical across all cases.")
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--t-min", type=float, default=-10.0, help="Sweep start (mm, tight end)")
parser.add_argument("--t-max", type=float, default=15.0, help="Sweep end (mm, wide end)")
parser.add_argument("--n-steps", type=int, default=50, help="Number of sweep steps")
args = parser.parse_args()

# ── Load precomputed geometry reference ──────────────────────────────────────

print(f"Loading geometry reference from {args.geometry_fields}")
geo_ref = np.load(args.geometry_fields)
ref_centroids = geo_ref["centroids"]  # (n_cells_ref, 3)
ref_tau = geo_ref["tau"]
ref_envelope = geo_ref["envelope"]
ref_entry_t = geo_ref["entry_t"]
ref_is_geo = geo_ref["is_geometric_septum"]
ref_is_ldrb = geo_ref["is_ldrb_septum"]
n_ref = len(ref_tau)
print(f"  Reference: {n_ref} cells, geometric={int(ref_is_geo.sum())}, "
      f"envelope={int(ref_envelope.sum())}")

# Build a KDTree on the reference centroids for matching
from scipy.spatial import cKDTree
ref_tree = cKDTree(ref_centroids)

# ── Load cases ───────────────────────────────────────────────────────────────

def load_case(d):
    d = d.resolve()
    pc = np.load(d / "per_cell_data.npz")

    # Label
    desc = (d / "run_description.txt").read_text().strip() if (d / "run_description.txt").exists() else d.name
    for pfx in ["Phase1 shared-mesh v2circ ", "Phase1 v2circ ", "v2circ "]:
        desc = desc.replace(pfx, "")
    label = desc.split()[0]

    # RV ESP
    rv_esp = None
    pres_path = d / "solver" / "pressure_history.npy"
    if pres_path.exists():
        pres = np.load(pres_path)
        if pres.ndim == 2 and pres.shape[1] >= 2:
            rv_esp = float(pres[:, 1].max())

    # Match per-cell data to the geometry reference by centroid position.
    # The per-cell npz has work arrays in MPI-gather order; the geometry
    # reference was computed serially. We match by nearest centroid.
    pc_centroids = pc["centroids"]
    dists, indices = ref_tree.query(pc_centroids)
    max_dist = dists.max()
    if max_dist > 1e-6:
        print(f"  WARNING: {label} centroid match max dist = {max_dist:.2e} "
              f"(expected ~0 for shared mesh)")

    # Geometry fields from the REFERENCE (identical across all cases)
    # Work fields from the PER-CELL data (case-specific)
    return {
        "label": label,
        "rv_esp": rv_esp,
        "dir": d,
        # Geometry from reference (via index matching)
        "tau": ref_tau[indices],
        "envelope": ref_envelope[indices],
        "entry_t": ref_entry_t[indices],
        "is_geometric_septum": ref_is_geo[indices],
        "is_ldrb_septum": ref_is_ldrb[indices],
        # Work from per-cell computation (case-specific)
        "w_total": pc["w_total"],
        "proxy_PLV_ll": pc["proxy_PLV_ll"],
        "proxy_PRV_ll": pc["proxy_PRV_ll"],
        "proxy_Trans_ll": pc["proxy_Trans_ll"],
    }

print("Loading cases...")
cases = []
for d in args.result_dirs:
    d = d.resolve()
    if not (d / "per_cell_data.npz").exists():
        print(f"  SKIP {d.name}: no per_cell_data.npz")
        continue
    cases.append(load_case(d))

if len(cases) < 2:
    print("ERROR: need at least 2 cases for inter-case correlation")
    sys.exit(1)

# Sort by RV ESP
cases.sort(key=lambda c: c["rv_esp"] or 0)
print(f"Loaded {len(cases)} cases:")
for c in cases:
    n_env = int(c["envelope"].sum())
    n_geo = int(c["is_geometric_septum"].sum())
    print(f"  {c['label']:<18} RV_ESP={c['rv_esp'] or '?':>5}  "
          f"n_cells={len(c['tau'])}  envelope={n_env}  geometric={n_geo}")

# Verify all cases have same cell count (shared mesh)
cell_counts = [len(c["tau"]) for c in cases]
if len(set(cell_counts)) != 1:
    print(f"WARNING: cell counts differ across cases: {cell_counts}")
    print("  This means the sims used different meshes. Results may be noisy.")
else:
    print(f"All cases have {cell_counts[0]} cells (shared mesh ✓)")

# ── Output directory ─────────────────────────────────────────────────────────
if args.output_dir:
    out_dir = args.output_dir.resolve()
else:
    out_dir = Path.cwd() / "results" / "analysis" / "sweep"
out_dir.mkdir(parents=True, exist_ok=True)

# ── Detect mesh units (mm vs m) ─────────────────────────────────────────────
# entry_t is in mesh units. If entry_t values are ~0.01 range, mesh is in m.
et_sample = cases[0]["entry_t"][cases[0]["envelope"]]
if abs(et_sample.max()) < 0.1:
    mesh_to_mm = 1000.0
    unit = "m"
else:
    mesh_to_mm = 1.0
    unit = "mm"
print(f"Mesh unit: {unit} (entry_t max in envelope = {et_sample.max():.4f})")

# ── Sweep ────────────────────────────────────────────────────────────────────

t_values = np.linspace(args.t_min / mesh_to_mm, args.t_max / mesh_to_mm, args.n_steps)
n_cases = len(cases)

W_true = np.zeros((args.n_steps, n_cases))
W_PLV = np.zeros((args.n_steps, n_cases))
W_PRV = np.zeros((args.n_steps, n_cases))
W_Trans = np.zeros((args.n_steps, n_cases))
W_mean = np.zeros((args.n_steps, n_cases))
W_dom = np.zeros((args.n_steps, n_cases))
n_cells_sweep = np.zeros((args.n_steps, n_cases), dtype=int)

for i, t in enumerate(t_values):
    for j, c in enumerate(cases):
        mask = c["envelope"] & (c["entry_t"] < t)
        n_cells_sweep[i, j] = int(mask.sum())
        if mask.sum() == 0:
            continue
        plv = c["proxy_PLV_ll"][mask]
        prv = c["proxy_PRV_ll"][mask]
        tau_m = c["tau"][mask]

        W_true[i, j] = c["w_total"][mask].sum()
        W_PLV[i, j] = plv.sum()
        W_PRV[i, j] = prv.sum()
        W_Trans[i, j] = c["proxy_Trans_ll"][mask].sum()
        W_mean[i, j] = 0.5 * (plv.sum() + prv.sum())
        W_dom[i, j] = np.where(tau_m < 0.5, plv, prv).sum()

# ── Pearson r across cases at each sweep step ────────────────────────────────

def safe_pearsonr(x, y):
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 3:
        return np.nan
    return pearsonr(x, y)[0]

r_PLV = np.array([safe_pearsonr(W_true[i], W_PLV[i]) for i in range(args.n_steps)])
r_PRV = np.array([safe_pearsonr(W_true[i], W_PRV[i]) for i in range(args.n_steps)])
r_Trans = np.array([safe_pearsonr(W_true[i], W_Trans[i]) for i in range(args.n_steps)])
r_mean = np.array([safe_pearsonr(W_true[i], W_mean[i]) for i in range(args.n_steps)])
r_dom = np.array([safe_pearsonr(W_true[i], W_dom[i]) for i in range(args.n_steps)])

# ── Reference points ─────────────────────────────────────────────────────────
# t = 0 is geometric by construction. Find LDRB-equivalent by cell count match.
geo_n = int(np.mean([c["is_geometric_septum"].sum() for c in cases]))
ldrb_n = int(np.mean([c["is_ldrb_septum"].sum() for c in cases]))
mean_cells_at_t = n_cells_sweep.mean(axis=1)

# ── Print summary at key t values ────────────────────────────────────────────
t_mm = t_values * mesh_to_mm
print(f"\n{'t (mm)':>8} {'n_cells':>8} {'r_PLV':>8} {'r_PRV':>8} {'r_Trans':>9} {'r_mean':>8} {'r_dom':>8}")
print("-" * 65)
for t_target in [-5, -3, -1, 0, 1, 3, 5, 10]:
    idx = np.argmin(np.abs(t_mm - t_target))
    print(f"{t_mm[idx]:>+8.1f} {mean_cells_at_t[idx]:>8.0f} "
          f"{r_PLV[idx]:>8.3f} {r_PRV[idx]:>8.3f} {r_Trans[idx]:>9.3f} "
          f"{r_mean[idx]:>8.3f} {r_dom[idx]:>8.3f}")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 1]})

# Left: main sensitivity curve
ax = axes[0]
ax.plot(t_mm, r_PLV, '-o', color='C0', label='P_LV × dε_ll', lw=1.8, ms=3)
ax.plot(t_mm, r_PRV, '-o', color='C3', label='P_RV × dε_ll', lw=1.8, ms=3)
ax.plot(t_mm, r_Trans, '-s', color='C2', label='(P_LV − P_RV) × dε_ll', lw=2.2, ms=4)
ax.plot(t_mm, r_mean, '-D', color='m', label='(P_LV + P_RV)/2 × dε_ll', lw=1.8, ms=3)
ax.plot(t_mm, r_dom, '-^', color='k', label='P_dom × dε_ll', lw=1.8, ms=3)

ax.axvline(0, color='blue', ls='--', lw=1.5, alpha=0.7, label='t=0 (geometric)')
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
ax.set_xlabel('t (mm) — septum width parameter\n← tighter   |   geometric at 0   |   wider →')
ax.set_ylabel(f'Pearson r (across {n_cases} spectrum cases)')
ax.set_title('Sensitivity curve: does the proxy track disease progression?')
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3)
ax.set_ylim(-1.1, 1.1)

# Right: cell count vs t (shows how many cells are in the sweep at each step)
ax2 = axes[1]
ax2.plot(t_mm, mean_cells_at_t, 'k-', lw=1.5)
ax2.axvline(0, color='blue', ls='--', lw=1.5, alpha=0.7)
ax2.axhline(geo_n, color='blue', ls=':', lw=1, alpha=0.5)
ax2.axhline(ldrb_n, color='red', ls=':', lw=1, alpha=0.5)
ax2.text(t_mm.max() * 0.95, geo_n, f' geo={geo_n}', fontsize=8, color='blue', va='bottom', ha='right')
ax2.text(t_mm.max() * 0.95, ldrb_n, f' ldrb={ldrb_n}', fontsize=8, color='red', va='bottom', ha='right')
ax2.set_xlabel('t (mm)')
ax2.set_ylabel('Mean cells in septum(t)')
ax2.set_title('Septum size')
ax2.grid(alpha=0.3)

fig.suptitle(f'Question A: inter-case sensitivity curve ({n_cases} cases, '
             f'{cell_counts[0]} cells/case, longitudinal strain proxy)',
             fontsize=11)
plt.tight_layout()

fig_path = out_dir / "sweep_sensitivity.pdf"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"\nSaved {fig_path}")

# Save raw data
np.savez(out_dir / "sweep_raw.npz",
         t_values=t_values, t_mm=t_mm,
         W_true=W_true, W_PLV=W_PLV, W_PRV=W_PRV,
         W_Trans=W_Trans, W_mean=W_mean, W_dom=W_dom,
         n_cells_sweep=n_cells_sweep,
         r_PLV=r_PLV, r_PRV=r_PRV, r_Trans=r_Trans, r_mean=r_mean, r_dom=r_dom,
         case_labels=np.array([c["label"] for c in cases]),
         case_rv_esp=np.array([c["rv_esp"] or 0 for c in cases]))
print(f"Saved raw data to {out_dir / 'sweep_raw.npz'}")
print("Done.")
