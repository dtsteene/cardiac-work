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
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

KPA = 1e-3
PROXY_NAMES = ["PLV", "PRV", "Trans", "Mean", "Dominant"]
SELECTED_T_MM = [-10, -5, -2, 0, 2, 5, 10, 20]

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
parser.add_argument("--include-epi", action="store_true",
                    help="Use an epi-inclusive sweep envelope: d_sum <= d_sum_max. "
                         "The default keeps the historical envelope "
                         "d_sum <= d_sum_max AND NOT touches_epi.")
args = parser.parse_args()
PC_FILE = "per_cell_data_percase.npz" if args.percase else "per_cell_data.npz"
TAG_MODE = "per-case (prestressed tagging)" if args.percase else "canonical (u_pre permutation)"
ENVELOPE_MODE = "epi-inclusive" if args.include_epi else "epi-excluded"
MODE = f"{TAG_MODE}, {ENVELOPE_MODE} envelope"

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

    # Reconstruct the sweep envelope from raw fields. The historical/default
    # version excludes cells topologically touching the epicardium; --include-epi
    # keeps those cells eligible so the relaxed septum can grow all the way to
    # the epicardial side where the geometric parametrization permits it.
    d_sum_max_mm = float(pc.get("envelope_d_sum_max_mm", 22.0))
    et_sample = pc["entry_t"][pc["envelope"]] if pc["envelope"].sum() > 0 else pc["entry_t"]
    mesh_to_mm_local = 1000.0 if (len(et_sample) > 0 and abs(et_sample.max()) < 0.1) else 1.0
    d_sum_max = d_sum_max_mm / mesh_to_mm_local
    touches_epi = pc["touches_epi"].astype(bool)
    envelope_base = pc["d_sum"] <= d_sum_max
    envelope = envelope_base if args.include_epi else (envelope_base & ~touches_epi)

    return {
        "label": label,
        "rv_esp": rv_esp,
        "dir": d,
        "tau": pc["tau"],
        "cell_volumes": pc["cell_volumes"],
        "region_tags": pc["region_tags"],
        "envelope": envelope,
        "n_epi_eligible": int((envelope_base & touches_epi).sum()),
        "entry_t": pc["entry_t"],
        "is_geometric_septum": pc["is_geometric_septum"].astype(bool),
        "is_ldrb_septum": pc.get("is_ldrb_septum", np.zeros_like(pc["tau"], dtype=bool)).astype(bool),
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
          f"n_cells={len(c['tau'])}  envelope={n_env}  "
          f"epi_eligible={c['n_epi_eligible']}  geometric={n_geo}")

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


def positive_density(case, mask, values):
    arr = case[values] if isinstance(values, str) else values
    volume = float(case["cell_volumes"][mask].sum())
    if volume <= 0:
        return np.nan
    return float(-arr[mask].sum() / volume * KPA)


def finite_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def slope_through_origin(x, y):
    x, y = finite_pair(x, y)
    denom = float(np.dot(x, x))
    if len(x) < 2 or denom == 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def ratio_error_stats(true_ratio, proxy_ratio):
    true_ratio, proxy_ratio = finite_pair(true_ratio, proxy_ratio)
    keep = (true_ratio != 0) & (proxy_ratio != 0)
    true_ratio = true_ratio[keep]
    proxy_ratio = proxy_ratio[keep]
    if len(true_ratio) == 0:
        return {
            "mean_abs_ratio_error": np.nan,
            "median_abs_ratio_error": np.nan,
            "max_abs_ratio_error": np.nan,
            "mean_signed_relative_ratio_error": np.nan,
            "mean_abs_relative_ratio_error": np.nan,
            "median_abs_relative_ratio_error": np.nan,
            "mean_abs_log_ratio_error": np.nan,
            "median_abs_log_ratio_error": np.nan,
            "max_abs_log_ratio_error": np.nan,
            "proxy_over_true_ratio_mean": np.nan,
            "proxy_over_true_ratio_median": np.nan,
        }

    raw_err = proxy_ratio - true_ratio
    rel_err = proxy_ratio / true_ratio - 1.0
    log_err = np.abs(np.log(np.abs(proxy_ratio / true_ratio)))
    ratio = proxy_ratio / true_ratio
    return {
        "mean_abs_ratio_error": float(np.mean(np.abs(raw_err))),
        "median_abs_ratio_error": float(np.median(np.abs(raw_err))),
        "max_abs_ratio_error": float(np.max(np.abs(raw_err))),
        "mean_signed_relative_ratio_error": float(np.mean(rel_err)),
        "mean_abs_relative_ratio_error": float(np.mean(np.abs(rel_err))),
        "median_abs_relative_ratio_error": float(np.median(np.abs(rel_err))),
        "mean_abs_log_ratio_error": float(np.mean(log_err)),
        "median_abs_log_ratio_error": float(np.median(log_err)),
        "max_abs_log_ratio_error": float(np.max(log_err)),
        "proxy_over_true_ratio_mean": float(np.mean(ratio)),
        "proxy_over_true_ratio_median": float(np.median(ratio)),
    }


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

r_PLV = np.array([safe_pearsonr(W_true[i], W_PLV[i]) for i in range(args.n_steps)])
r_PRV = np.array([safe_pearsonr(W_true[i], W_PRV[i]) for i in range(args.n_steps)])
r_Trans = np.array([safe_pearsonr(W_true[i], W_Trans[i]) for i in range(args.n_steps)])
r_mean = np.array([safe_pearsonr(W_true[i], W_mean[i]) for i in range(args.n_steps)])
r_dom = np.array([safe_pearsonr(W_true[i], W_dom[i]) for i in range(args.n_steps)])

# ── Magnitude/ratio metrics ─────────────────────────────────────────────────
# The correlation curve answers whether a proxy ranks the severity cases like
# tensor work. For the thesis story we also need to know whether the proxy keeps
# the septum/free-wall magnitude relation. Use the same positive-density
# convention as analyze_h5_sweep_core.py: -integral / volume, reported in kPa.
D_true = np.full((args.n_steps, n_cases), np.nan)
D_PLV = np.full((args.n_steps, n_cases), np.nan)
D_PRV = np.full((args.n_steps, n_cases), np.nan)
D_Trans = np.full((args.n_steps, n_cases), np.nan)
D_mean = np.full((args.n_steps, n_cases), np.nan)
D_dom = np.full((args.n_steps, n_cases), np.nan)
R_true = np.full((args.n_steps, n_cases), np.nan)
R_PLV = np.full((args.n_steps, n_cases), np.nan)
R_PRV = np.full((args.n_steps, n_cases), np.nan)
R_Trans = np.full((args.n_steps, n_cases), np.nan)
R_mean = np.full((args.n_steps, n_cases), np.nan)
R_dom = np.full((args.n_steps, n_cases), np.nan)
fw_tensor_mean_density = np.full(n_cases, np.nan)
fw_adjacent_ll_mean_density = np.full(n_cases, np.nan)

for j, c in enumerate(cases):
    lv_mask = c["region_tags"] == 1
    rv_mask = c["region_tags"] == 2
    fw_tensor_mean_density[j] = 0.5 * (
        positive_density(c, lv_mask, "w_total") + positive_density(c, rv_mask, "w_total")
    )
    fw_adjacent_ll_mean_density[j] = 0.5 * (
        positive_density(c, lv_mask, "proxy_PLV_ll") + positive_density(c, rv_mask, "proxy_PRV_ll")
    )

for i, t in enumerate(t_values):
    for j, c in enumerate(cases):
        mask = c["envelope"] & (c["entry_t"] < t)
        if mask.sum() == 0:
            continue

        plv = c["proxy_PLV_ll"]
        prv = c["proxy_PRV_ll"]
        trans = c["proxy_Trans_ll"]
        mean_proxy = 0.5 * (plv + prv)
        dom_proxy = np.where(c["tau"] < 0.5, plv, prv)

        D_true[i, j] = positive_density(c, mask, "w_total")
        D_PLV[i, j] = positive_density(c, mask, plv)
        D_PRV[i, j] = positive_density(c, mask, prv)
        D_Trans[i, j] = positive_density(c, mask, trans)
        D_mean[i, j] = positive_density(c, mask, mean_proxy)
        D_dom[i, j] = positive_density(c, mask, dom_proxy)

        if fw_tensor_mean_density[j] != 0:
            R_true[i, j] = D_true[i, j] / fw_tensor_mean_density[j]
        if fw_adjacent_ll_mean_density[j] != 0:
            R_PLV[i, j] = D_PLV[i, j] / fw_adjacent_ll_mean_density[j]
            R_PRV[i, j] = D_PRV[i, j] / fw_adjacent_ll_mean_density[j]
            R_Trans[i, j] = D_Trans[i, j] / fw_adjacent_ll_mean_density[j]
            R_mean[i, j] = D_mean[i, j] / fw_adjacent_ll_mean_density[j]
            R_dom[i, j] = D_dom[i, j] / fw_adjacent_ll_mean_density[j]

proxy_density = {
    "PLV": D_PLV,
    "PRV": D_PRV,
    "Trans": D_Trans,
    "Mean": D_mean,
    "Dominant": D_dom,
}
proxy_ratio = {
    "PLV": R_PLV,
    "PRV": R_PRV,
    "Trans": R_Trans,
    "Mean": R_mean,
    "Dominant": R_dom,
}
proxy_r = {
    "PLV": r_PLV,
    "PRV": r_PRV,
    "Trans": r_Trans,
    "Mean": r_mean,
    "Dominant": r_dom,
}

boundary_rows = []
for i, t in enumerate(t_values):
    for proxy in PROXY_NAMES:
        stats = ratio_error_stats(R_true[i], proxy_ratio[proxy][i])
        d_true, d_proxy = finite_pair(D_true[i], proxy_density[proxy][i])
        r_value = float(proxy_r[proxy][i])
        row = {
            "envelope_mode": ENVELOPE_MODE,
            "tag_mode": TAG_MODE,
            "t_mm": float(t * mesh_to_mm),
            "n_cases": int(len(d_true)),
            "n_cells_mean": float(np.mean(n_cells_sweep[i])),
            "n_cells_min": int(np.min(n_cells_sweep[i])),
            "n_cells_max": int(np.max(n_cells_sweep[i])),
            "proxy": proxy,
            "pearson_r": r_value,
            "pearson_r2": float(r_value * r_value) if np.isfinite(r_value) else np.nan,
            "true_density_mean_kPa": float(np.nanmean(D_true[i])),
            "proxy_density_mean_kPa": float(np.nanmean(proxy_density[proxy][i])),
            "density_slope_proxy_per_true": slope_through_origin(d_true, d_proxy),
            "true_septum_to_fwmean_ratio_mean": float(np.nanmean(R_true[i])),
            "proxy_septum_to_fw_adjacent_ratio_mean": float(np.nanmean(proxy_ratio[proxy][i])),
            **stats,
        }
        boundary_rows.append(row)

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
for t_target in SELECTED_T_MM:
    if t_target < t_mm[0] - 1e-9 or t_target > t_mm[-1] + 1e-9:
        continue
    idx = np.argmin(np.abs(t_mm - t_target))
    print(f"{t_mm[idx]:>+8.1f} {mean_cells_at_t[idx]:>8.0f} "
          f"{r_PLV[idx]:>8.3f} {r_PRV[idx]:>8.3f} {r_Trans[idx]:>9.3f} "
          f"{r_mean[idx]:>8.3f} {r_dom[idx]:>8.3f}")

selected_rows = []
for t_target in SELECTED_T_MM:
    if t_target < t_mm[0] - 1e-9 or t_target > t_mm[-1] + 1e-9:
        continue
    idx = int(np.argmin(np.abs(t_mm - t_target)))
    for row in boundary_rows:
        if abs(row["t_mm"] - float(t_mm[idx])) < 1e-9:
            selected_rows.append({**row, "requested_t_mm": float(t_target)})

write_csv(out_dir / "boundary_metrics_by_t.csv", boundary_rows)
write_csv(out_dir / "boundary_metrics_selected_thresholds.csv", selected_rows)
print(f"Saved boundary metrics to {out_dir / 'boundary_metrics_by_t.csv'}")
print(f"Saved selected thresholds to {out_dir / 'boundary_metrics_selected_thresholds.csv'}")

# ── Plot (matches analyze_spectrum.py fig_spectrum_sweep style) ──────────────
SWEEP_STYLE = {
    "r_PLV":   dict(color="#1f77b4", label=r"$P_{LV}$",              values=r_PLV),
    "r_PRV":   dict(color="#d62728", label=r"$P_{RV}$",              values=r_PRV),
    "r_Trans": dict(color="#2ca02c", label=r"$P_{LV}-P_{RV}$",       values=r_Trans),
    "r_Mean":  dict(color="#ff7f0e", label=r"$(P_{LV}+P_{RV})/2$",
                    values=r_mean, ls="--"),
}

fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
ax.axhline(0.0, color="lightgray", lw=0.8, zorder=0)
ax.axhline(1.0, color="lightgray", lw=0.5, ls=":")
for key, style in SWEEP_STYLE.items():
    ax.plot(t_mm, style["values"],
            color=style["color"], lw=2.4 if key == "r_Trans" else 1.5,
            ls=style.get("ls", "-"), label=style["label"],
            alpha=0.95 if key == "r_Trans" else 0.75)

ax.axvline(0.0, color="gray", lw=1.0, ls="--", alpha=0.7)
ax.text(0.0, -0.96, "  geometric cutoff (t=0)",
        ha="left", va="bottom", fontsize=9, color="gray")

ax.set_xlim(t_mm[0] - 0.5, t_mm[-1] + 0.5)
ax.set_ylim(-1.05, 1.1)
ax.set_xlabel(r"Boundary relaxation threshold $t$ (mm)   "
              r"$\;\;\mathrm{mask}(t) = \{c : \mathrm{entry}_t(c) \leq t\} "
              rf"\cap \mathrm{{envelope}}_{{\mathrm{{{ENVELOPE_MODE}}}}}$",
              fontsize=10)
ax.set_ylabel(f"Pearson r with $W_{{true}}$ across {n_cases} severities",
              fontsize=11)
ax.grid(alpha=0.25)
ax.legend(loc="lower left", fontsize=9, ncol=3, framealpha=0.95)

ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
sel = np.linspace(0, len(t_mm) - 1, 6).astype(int)
ax_top.set_xticks(t_mm[sel])
ax_top.set_xticklabels([f"{int(mean_cells_at_t[i])}" for i in sel], fontsize=8)
ax_top.set_xlabel("cells in sweep region", fontsize=9, color="dimgray")
ax_top.tick_params(axis="x", labelcolor="dimgray")

ax.set_title(f"Proxy tracking across septum sweep  ({MODE})",
             fontsize=12, fontweight="bold")

fig_path = out_dir / "sweep_sensitivity.pdf"
fig.savefig(fig_path, bbox_inches="tight")
fig.savefig(fig_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
print(f"\nSaved {fig_path}")

fig2, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True, constrained_layout=True)
for proxy, color, lw in [
    ("PLV", "#1f77b4", 1.5),
    ("PRV", "#d62728", 1.5),
    ("Trans", "#2ca02c", 2.4),
    ("Mean", "#ff7f0e", 1.8),
    ("Dominant", "#9467bd", 1.4),
]:
    sub = [row for row in boundary_rows if row["proxy"] == proxy]
    x = np.array([row["t_mm"] for row in sub], dtype=float)
    pearson = np.array([row["pearson_r"] for row in sub], dtype=float)
    ratio_err = np.array([row["mean_abs_log_ratio_error"] for row in sub], dtype=float)
    axes[0].plot(x, pearson, color=color, lw=lw, label=proxy, alpha=0.9)
    axes[1].plot(x, ratio_err, color=color, lw=lw, label=proxy, alpha=0.9)

for axis in axes:
    axis.axvline(0.0, color="gray", lw=1.0, ls="--", alpha=0.7)
    axis.grid(alpha=0.25)

axes[0].axhline(0.0, color="lightgray", lw=0.8, zorder=0)
axes[0].set_ylabel("Pearson r")
axes[0].set_ylim(-1.05, 1.1)
axes[0].legend(loc="lower left", fontsize=8, ncol=5, framealpha=0.95)
axes[1].set_ylabel("Mean absolute log ratio error")
axes[1].set_xlabel("Boundary threshold t (mm); negative is tighter/deeper septum")
axes[1].set_ylim(bottom=0.0)
fig2.suptitle(f"Septum boundary sweep with ratio preservation ({MODE})", fontsize=12, fontweight="bold")
fig2_path = out_dir / "boundary_metrics_summary.pdf"
fig2.savefig(fig2_path, bbox_inches="tight")
fig2.savefig(fig2_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
print(f"Saved {fig2_path}")

np.savez(out_dir / "sweep_raw.npz",
         t_values=t_values, t_mm=t_mm,
         W_true=W_true, W_PLV=W_PLV, W_PRV=W_PRV,
         W_Trans=W_Trans, W_mean=W_mean, W_dom=W_dom,
         D_true=D_true, D_PLV=D_PLV, D_PRV=D_PRV,
         D_Trans=D_Trans, D_mean=D_mean, D_dom=D_dom,
         R_true=R_true, R_PLV=R_PLV, R_PRV=R_PRV,
         R_Trans=R_Trans, R_mean=R_mean, R_dom=R_dom,
         fw_tensor_mean_density=fw_tensor_mean_density,
         fw_adjacent_ll_mean_density=fw_adjacent_ll_mean_density,
         n_cells_sweep=n_cells_sweep,
         r_PLV=r_PLV, r_PRV=r_PRV, r_Trans=r_Trans, r_mean=r_mean, r_dom=r_dom,
         # Direct reference definitions
         ref_geo_r_PLV=ref_geo["PLV"], ref_geo_r_Trans=ref_geo["Trans"],
         ref_ldrb_r_PLV=ref_ldrb["PLV"], ref_ldrb_r_Trans=ref_ldrb["Trans"],
         geo_n=geo_n, ldrb_n=ldrb_n,
         include_epi=args.include_epi,
         envelope_mode=ENVELOPE_MODE,
         n_epi_eligible=np.array([c["n_epi_eligible"] for c in cases]),
         case_labels=np.array([c["label"] for c in cases]),
         case_rv_esp=np.array([c["rv_esp"] or 0 for c in cases]))
print(f"Saved raw data to {out_dir / 'sweep_raw.npz'}")
print("Done.")
