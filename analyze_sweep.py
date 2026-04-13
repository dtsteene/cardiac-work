#!/usr/bin/env python3
"""
analyze_sweep.py — t-threshold septum sweep with inter-case Pearson r

Sweep: the t-threshold relaxation of the geometric septum definition.
  septum(t) = envelope AND (entry_t < t)
  t = 0: exactly the geometric septum
  t < 0: tighter than geometric (only deepest cells)
  t > 0: wider than geometric (adds cells toward epi)

At each t, sum W_true and each W_proxy over all cells in septum(t), per case.
Compute Pearson r across the spectrum cases.

Geometry fields (tau, envelope, entry_t) are loaded from per_cell_data.npz or
(if --percase is given) per_cell_data_percase.npz. The canonical pipeline uses
the u_pre permutation so tagging is identical across all cases; the per-case
pipeline recomputes distances on each case's prestressed mesh and produces
drift. Compare the two to measure how much the proxy correlations depend on
the tagging variance.

Usage:
  python3 analyze_sweep.py <result_dir1> <result_dir2> ...
  python3 analyze_sweep.py results/sims/2026-04-12/UKB_6beats_run_*
  python3 analyze_sweep.py --percase results/sims/2026-04-12/UKB_6beats_run_*
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
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--t-min", type=float, default=-10.0, help="Sweep start (mm)")
parser.add_argument("--t-max", type=float, default=15.0, help="Sweep end (mm)")
parser.add_argument("--n-steps", type=int, default=50, help="Number of sweep steps")
parser.add_argument("--percase", action="store_true",
                    help="Load per_cell_data_percase.npz instead of per_cell_data.npz "
                         "(uses per-case prestressed-mesh tagging instead of the "
                         "canonical u_pre permutation)")
args = parser.parse_args()
PC_FILE = "per_cell_data_percase.npz" if args.percase else "per_cell_data.npz"
MODE = "per-case (prestressed tagging)" if args.percase else "canonical (u_pre permutation)"

# ── Load cases ───────────────────────────────────────────────────────────────

def load_case(d):
    d = d.resolve()
    pc = np.load(d / PC_FILE)

    desc = (d / "run_description.txt").read_text().strip() if (d / "run_description.txt").exists() else d.name
    for pfx in ["Phase1 shared-mesh v2circ ", "Phase1 v2circ ", "v2circ "]:
        desc = desc.replace(pfx, "")
    label = desc.split()[0]

    rv_esp = None
    pres_path = d / "solver" / "solver_cavity_pressure_mmHg.npy"
    if not pres_path.exists():
        pres_path = d / "solver" / "pressure_history.npy"  # backwards compat
    if pres_path.exists():
        pres = np.load(pres_path)
        if pres.ndim == 2 and pres.shape[1] >= 2:
            rv_esp = float(pres[:, 1].max())

    # Reconstruct envelope from raw fields: only d_sum_max + ~touches_epi
    # (the saved "envelope" field may use the old overly-restrictive definition)
    d_sum_max_mm = float(pc.get("envelope_d_sum_max_mm", 22.0))
    et_sample = pc["entry_t"][pc["envelope"]] if pc["envelope"].sum() > 0 else pc["entry_t"]
    mesh_to_mm_local = 1000.0 if (len(et_sample) > 0 and abs(et_sample.max()) < 0.1) else 1.0
    d_sum_max = d_sum_max_mm / mesh_to_mm_local
    envelope = (pc["d_sum"] <= d_sum_max) & ~pc["touches_epi"]

    return {
        "label": label,
        "rv_esp": rv_esp,
        "dir": d,
        "tau": pc["tau"],
        "envelope": envelope,
        "entry_t": pc["entry_t"],
        "is_geometric_septum": pc["is_geometric_septum"],
        "is_ldrb_septum": pc.get("is_ldrb_septum", np.zeros_like(pc["tau"], dtype=bool)),
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

cases.sort(key=lambda c: c["rv_esp"] or 0)
print(f"Loaded {len(cases)} cases:")
for c in cases:
    n_env = int(c["envelope"].sum())
    n_geo = int(c["is_geometric_septum"].sum())
    print(f"  {c['label']:<18} RV_ESP={c['rv_esp'] or '?':>5}  "
          f"n_cells={len(c['tau'])}  envelope={n_env}  geometric={n_geo}")

# ── Output directory ─────────────────────────────────────────────────────────
if args.output_dir:
    out_dir = args.output_dir.resolve()
else:
    out_dir = Path.cwd() / "results" / "analysis" / "sweep"
out_dir.mkdir(parents=True, exist_ok=True)

# ── Detect mesh units ────────────────────────────────────────────────────────
et_sample = cases[0]["entry_t"][cases[0]["envelope"]]
if len(et_sample) > 0 and abs(et_sample.max()) < 0.1:
    mesh_to_mm = 1000.0
    unit = "m"
else:
    mesh_to_mm = 1.0
    unit = "mm"
print(f"Mesh unit: {unit}")

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

# ── Pearson r ────────────────────────────────────────────────────────────────
def safe_pearsonr(x, y):
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 3:
        return np.nan
    return pearsonr(x, y)[0]

r_PLV = np.array([safe_pearsonr(W_true[i], W_PLV[i]) for i in range(args.n_steps)])
r_PRV = np.array([safe_pearsonr(W_true[i], W_PRV[i]) for i in range(args.n_steps)])
r_Trans = np.array([safe_pearsonr(W_true[i], W_Trans[i]) for i in range(args.n_steps)])
r_mean = np.array([safe_pearsonr(W_true[i], W_mean[i]) for i in range(args.n_steps)])
r_dom = np.array([safe_pearsonr(W_true[i], W_dom[i]) for i in range(args.n_steps)])

# ── Reference definitions (direct, no sweep) ────────────────────────────────
geo_n = int(np.mean([c["is_geometric_septum"].sum() for c in cases]))
ldrb_n = int(np.mean([c["is_ldrb_septum"].sum() for c in cases]))
mean_cells_at_t = n_cells_sweep.mean(axis=1)

# Direct correlations over named definitions (no envelope filtering)
W_true_geo = np.array([c["w_total"][c["is_geometric_septum"]].sum() for c in cases])
W_PLV_geo = np.array([c["proxy_PLV_ll"][c["is_geometric_septum"]].sum() for c in cases])
W_PRV_geo = np.array([c["proxy_PRV_ll"][c["is_geometric_septum"]].sum() for c in cases])
W_Trans_geo = np.array([c["proxy_Trans_ll"][c["is_geometric_septum"]].sum() for c in cases])

W_true_ldrb = np.array([c["w_total"][c["is_ldrb_septum"]].sum() for c in cases])
W_PLV_ldrb = np.array([c["proxy_PLV_ll"][c["is_ldrb_septum"]].sum() for c in cases])
W_PRV_ldrb = np.array([c["proxy_PRV_ll"][c["is_ldrb_septum"]].sum() for c in cases])
W_Trans_ldrb = np.array([c["proxy_Trans_ll"][c["is_ldrb_septum"]].sum() for c in cases])

ref_geo = {"PLV": safe_pearsonr(W_true_geo, W_PLV_geo),
            "PRV": safe_pearsonr(W_true_geo, W_PRV_geo),
            "Trans": safe_pearsonr(W_true_geo, W_Trans_geo)}
ref_ldrb = {"PLV": safe_pearsonr(W_true_ldrb, W_PLV_ldrb),
             "PRV": safe_pearsonr(W_true_ldrb, W_PRV_ldrb),
             "Trans": safe_pearsonr(W_true_ldrb, W_Trans_ldrb)}

print(f"\n=== Direct reference definitions (no sweep) ===")
print(f"  Geometric ({geo_n} cells): r_PLV={ref_geo['PLV']:+.4f}  r_PRV={ref_geo['PRV']:+.4f}  r_Trans={ref_geo['Trans']:+.4f}")
print(f"  LDRB      ({ldrb_n} cells): r_PLV={ref_ldrb['PLV']:+.4f}  r_PRV={ref_ldrb['PRV']:+.4f}  r_Trans={ref_ldrb['Trans']:+.4f}")

# ── Print summary ────────────────────────────────────────────────────────────
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

ax = axes[0]
ax.plot(t_mm, r_PLV, '-o', color='C0', label='P_LV × dε_ll', lw=1.8, ms=3)
ax.plot(t_mm, r_PRV, '-o', color='C3', label='P_RV × dε_ll', lw=1.8, ms=3)
ax.plot(t_mm, r_Trans, '-s', color='C2', label='(P_LV − P_RV) × dε_ll', lw=2.2, ms=4)
ax.plot(t_mm, r_mean, '-D', color='m', label='(P_LV + P_RV)/2 × dε_ll', lw=1.8, ms=3)
ax.plot(t_mm, r_dom, '-^', color='k', label='P_dom × dε_ll', lw=1.8, ms=3)
ax.axvline(0, color='blue', ls='--', lw=1.5, alpha=0.7, label='t=0 (geometric)')
ax.axhline(0, color='k', lw=0.5, alpha=0.3)

# Reference markers: direct definitions (plotted at their mean cell count on right panel)
# On main panel, show as horizontal dashes at the right edge
ref_x = t_mm.max() * 0.98
ax.plot(ref_x, ref_geo["PLV"], 's', color='C0', ms=8, mew=2, mfc='none', zorder=5)
ax.plot(ref_x, ref_geo["Trans"], 's', color='C2', ms=8, mew=2, mfc='none', zorder=5)
ax.plot(ref_x*0.94, ref_ldrb["PLV"], 'D', color='C0', ms=7, mew=2, mfc='none', zorder=5)
ax.plot(ref_x*0.94, ref_ldrb["Trans"], 'D', color='C2', ms=7, mew=2, mfc='none', zorder=5)
ax.annotate('geo', (ref_x, ref_geo["PLV"]), fontsize=7, ha='left', va='bottom',
            xytext=(3, 2), textcoords='offset points')
ax.annotate('ldrb', (ref_x*0.94, ref_ldrb["Trans"]), fontsize=7, ha='right', va='bottom',
            xytext=(-3, 2), textcoords='offset points')

ax.set_xlabel('t (mm) — septum width\n← tighter | geometric at 0 | wider →')
ax.set_ylabel(f'Pearson r (across {n_cases} cases)')
ax.set_title('Sensitivity: does the proxy track disease progression?')
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3)
ax.set_ylim(-1.1, 1.1)

ax2 = axes[1]
ax2.plot(t_mm, mean_cells_at_t, 'k-', lw=1.5)
ax2.axvline(0, color='blue', ls='--', lw=1.5, alpha=0.7)
ax2.axhline(geo_n, color='blue', ls=':', lw=1, alpha=0.5)
ax2.axhline(ldrb_n, color='red', ls=':', lw=1, alpha=0.5)
ax2.text(t_mm.max()*0.95, geo_n, f' geo≈{geo_n}', fontsize=8, color='blue', va='bottom', ha='right')
ax2.text(t_mm.max()*0.95, ldrb_n, f' ldrb≈{ldrb_n}', fontsize=8, color='red', va='bottom', ha='right')
ax2.set_xlabel('t (mm)')
ax2.set_ylabel('Mean cells in septum(t)')
ax2.set_title('Septum size')
ax2.grid(alpha=0.3)

fig.suptitle(f'Inter-case sensitivity ({n_cases} cases, longitudinal strain proxy)', fontsize=11)
plt.tight_layout()
fig_path = out_dir / "sweep_sensitivity.pdf"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"\nSaved {fig_path}")

np.savez(out_dir / "sweep_raw.npz",
         t_values=t_values, t_mm=t_mm,
         W_true=W_true, W_PLV=W_PLV, W_PRV=W_PRV,
         W_Trans=W_Trans, W_mean=W_mean, W_dom=W_dom,
         n_cells_sweep=n_cells_sweep,
         r_PLV=r_PLV, r_PRV=r_PRV, r_Trans=r_Trans, r_mean=r_mean, r_dom=r_dom,
         # Direct reference definitions
         ref_geo_r_PLV=ref_geo["PLV"], ref_geo_r_Trans=ref_geo["Trans"],
         ref_ldrb_r_PLV=ref_ldrb["PLV"], ref_ldrb_r_Trans=ref_ldrb["Trans"],
         geo_n=geo_n, ldrb_n=ldrb_n,
         case_labels=np.array([c["label"] for c in cases]),
         case_rv_esp=np.array([c["rv_esp"] or 0 for c in cases]))
print(f"Saved raw data to {out_dir / 'sweep_raw.npz'}")
print("Done.")
