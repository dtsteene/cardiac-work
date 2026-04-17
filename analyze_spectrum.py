#!/usr/bin/env python3
"""
analyze_spectrum.py — Septal proxy correlation across the PAH severity spectrum.

Mirrors the structure of analyze_thickness.py but with severity as the sweep
axis instead of wall thickness. Produces two figures:

  fig_spectrum_core.png   — W_true + 3 canonical proxies vs severity, absolute
                             units, one panel per septum definition.
  fig_spectrum_stats.png  — Pearson r heatmap across {PLV, PRV, Trans, Mean,
                             Dom, optimal blend} × {geometric, LDRB}.

Usage:
    python3 analyze_spectrum.py [--results-dir results/sims/2026-04-12]
                                [--output-dir results/analysis/spectrum]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=Path,
                    default=Path("results/sims/2026-04-12"),
                    help="Dated results dir containing the canonical-tagged "
                         "spectrum sims.")
parser.add_argument("--output-dir", type=Path,
                    default=Path("results/analysis/spectrum"))
args = parser.parse_args()

out_dir = args.output_dir
out_dir.mkdir(parents=True, exist_ok=True)

# ── Spectrum configuration ───────────────────────────────────────────────────
# Archival severity keys (match the JSON filenames on disk). Display labels are
# remapped per the calibration-honesty rename documented in tuning.md.
SPECTRUM = [
    # (archival_key, display_label, achieved_RV_ESP_mmHg)
    ("healthy",         "Borderline PH",     30.6),
    ("mild",            "Mild",              38.2),
    ("moderate",        "Moderate",          55.4),
    ("moderate_severe", "Mod\u2013severe",   62.5),
    ("severe",          "Severe",            70.8),
    ("very_severe",     "Very severe",       85.0),
    ("end_stage",       "End-stage",         88.4),
]
KEYS = [k for k, _, _ in SPECTRUM]
DISPLAY = {k: d for k, d, _ in SPECTRUM}
RVESP = {k: r for k, _, r in SPECTRUM}

# ── Discovery ────────────────────────────────────────────────────────────────
sims = {}  # severity_key -> Path
for d in sorted(args.results_dir.glob("UKB_6beats_run_*")):
    desc_path = d / "run_description.txt"
    if not desc_path.exists():
        continue
    if not (d / "per_cell_data.npz").exists():
        continue
    desc = desc_path.read_text().strip()
    # Match longest keys first so moderate_severe isn't shadowed by moderate
    for k in sorted(KEYS, key=len, reverse=True):
        if f" {k} " in desc + " ":
            if k not in sims:  # first match wins
                sims[k] = d
            break

print(f"Discovered {len(sims)}/{len(KEYS)} spectrum sims in {args.results_dir}")
for k in KEYS:
    if k in sims:
        print(f"  {DISPLAY[k]:<16} (RV_ESP~{RVESP[k]:.0f})  <- {sims[k].name}")
    else:
        print(f"  {DISPLAY[k]:<16} MISSING")

missing = [k for k in KEYS if k not in sims]
if missing:
    print(f"\nWARNING: missing {len(missing)} cases: {missing}")

# ── Shared helpers (mirror analyze_thickness.py) ─────────────────────────────
def safe_r(x, y):
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 3:
        return np.nan
    return pearsonr(x, y)[0]


def compute_region_sums(pc, region_key):
    mask = pc[region_key]
    plv_cell = pc["proxy_PLV_ll"]
    prv_cell = pc["proxy_PRV_ll"]
    tau = pc["tau"]
    W_PLV = float(plv_cell[mask].sum())
    W_PRV = float(prv_cell[mask].sum())
    tau_lv = mask & (tau < 0.5)
    tau_rv = mask & (tau >= 0.5)
    W_dom = float(plv_cell[tau_lv].sum() + prv_cell[tau_rv].sum())
    return {
        "W_true":  float(pc["w_total"][mask].sum()),
        "W_PLV":   W_PLV,
        "W_PRV":   W_PRV,
        "W_Trans": W_PLV - W_PRV,
        "W_mean":  0.5 * (W_PLV + W_PRV),
        "W_dom":   W_dom,
    }


def optimal_angle(W_plv, W_prv, W_true):
    """Maximize r(cos(θ)*W_plv + sin(θ)*W_prv, W_true) over θ.

    Pearson r is scale-invariant in its first argument, so a (cos, sin)
    parameterisation covers every non-zero linear combination of the two
    proxies. We grid-search the circle at 0.1-degree resolution and return
    the argmax — correct under any correlation structure of u and v.
    Grid search is simpler and more robust than the closed-form generalized
    eigenproblem here, and 3600 correlations is a few milliseconds.
    """
    thetas = np.linspace(-np.pi, np.pi, 3601)
    best_theta, best_r = 0.0, -np.inf
    for th in thetas:
        W = np.cos(th) * W_plv + np.sin(th) * W_prv
        r = safe_r(W, W_true)
        if not np.isnan(r) and r > best_r:
            best_r = r
            best_theta = th
    return float(best_theta), float(best_r)


# ── Load all per_cell payloads once (used by both nominal-cutoff and sweep) ──
# Dict: severity_key -> pc (the npz object)
pcs = {}
for k in KEYS:
    if k not in sims:
        continue
    pcs[k] = np.load(sims[k] / "per_cell_data.npz", allow_pickle=True)

if not pcs:
    raise SystemExit("No per_cell_data.npz files found — aborting.")

# ── Reference entry_t for the threshold-relaxation sweep ─────────────────────
# entry_t is the threshold at which each cell enters the relaxed-septum
# region. Values differ slightly between cases because they are computed on
# each case's prestressed mesh; using one reference case makes the sweep
# definition uniform across the spectrum (a "canonical sweep"). Pick the
# earliest case in the spectrum so the reference is the healthiest geometry.
ref_key = next(k for k in KEYS if k in pcs)
ref_entry_t = pcs[ref_key]["entry_t"]
ref_envelope = pcs[ref_key]["envelope"]
print(f"\nSweep reference: {DISPLAY[ref_key]}  "
      f"(entry_t range [{ref_entry_t.min():+.4f}, {ref_entry_t.max():+.4f}])")

# Nominal septum-definition masks are case-specific (LDRB varies with the
# per-case mesh), which is part of the point — we're asking "does Trans win
# at either convention", not "does Trans win for one fixed mask set".
REGION_DEFS = ["geometric", "ldrb"]
MASK_KEY = {"geometric": "is_geometric_septum", "ldrb": "is_ldrb_septum"}


def sums_for_mask(pc, mask):
    """Intensive (density) region values for one case and mask.

    All values are in kPa (= mJ/mL = J/m^3 * 1e-3) — W per unit wall volume
    within the mask, so correlations across cases compare like-for-like
    even when the mask's cell count varies slightly between cases.
    """
    V = float(pc["cell_volumes"][mask].sum())
    if V <= 0:
        V = 1.0  # guard; downstream will see zero densities
    KPA = 1e-3  # J/m^3 -> kPa
    plv_cell = pc["proxy_PLV_ll"]
    prv_cell = pc["proxy_PRV_ll"]
    tau = pc["tau"]
    W_PLV = float(plv_cell[mask].sum()) / V * KPA
    W_PRV = float(prv_cell[mask].sum()) / V * KPA
    tau_lv = mask & (tau < 0.5)
    tau_rv = mask & (tau >= 0.5)
    W_dom = (float(plv_cell[tau_lv].sum())
             + float(prv_cell[tau_rv].sum())) / V * KPA
    return {
        "W_true":  float(pc["w_total"][mask].sum()) / V * KPA,
        "W_PLV":   W_PLV,
        "W_PRV":   W_PRV,
        "W_Trans": W_PLV - W_PRV,
        "W_mean":  0.5 * (W_PLV + W_PRV),
        "W_dom":   W_dom,
        "V_m3":    V,
    }


def collect_rvalues(case_masks):
    """Given {case_key: mask}, compute per-case density sums and r across cases."""
    rows = {k: [] for k in ["W_true", "W_PLV", "W_PRV", "W_Trans",
                             "W_mean", "W_dom", "V_m3"]}
    case_keys_used = []
    for k in KEYS:
        if k not in pcs or k not in case_masks:
            continue
        m = case_masks[k]
        if not m.any():
            continue
        s = sums_for_mask(pcs[k], m)
        for col, v in s.items():
            rows[col].append(v)
        case_keys_used.append(k)
    arrs = {col: np.array(v) for col, v in rows.items()}
    Wt = arrs["W_true"]
    r_vals = {
        "r_PLV":   safe_r(Wt, arrs["W_PLV"]),
        "r_PRV":   safe_r(Wt, arrs["W_PRV"]),
        "r_Trans": safe_r(Wt, arrs["W_Trans"]),
        "r_Mean":  safe_r(Wt, arrs["W_mean"]),
        "r_Dom":   safe_r(Wt, arrs["W_dom"]),
    }
    theta, r_opt = optimal_angle(arrs["W_PLV"], arrs["W_PRV"], Wt)
    r_vals["r_Opt"] = r_opt
    r_vals["theta_deg"] = float(np.degrees(theta))
    return case_keys_used, arrs, r_vals


# ── (1) Nominal cutoffs: geometric and LDRB septum per case ──────────────────
results = {}
for region_def in REGION_DEFS:
    case_masks = {k: pcs[k][MASK_KEY[region_def]] for k in pcs}
    case_keys_used, arrs, r_vals = collect_rvalues(case_masks)
    results[region_def] = {
        "keys": case_keys_used,
        "rv_esp": np.array([RVESP[k] for k in case_keys_used]),
        **arrs,
        **r_vals,
    }

# ── (2) Threshold-relaxation sweep: r(t) across a grid of cutoffs ────────────
# Sweep the relaxation parameter t from the geometric-core threshold (where
# the mask is smallest) out to the envelope limit. Use ref_entry_t to define
# the mask uniformly — at each t, a cell is included iff ref_entry_t <= t AND
# it belongs to the reference envelope (prevents runaway growth into LV/RV
# free wall).
t_lo = float(ref_entry_t[ref_envelope].min())
t_hi = float(ref_entry_t[ref_envelope].max())
T_GRID = np.linspace(t_lo, t_hi, 40)

sweep = {
    "t": T_GRID,
    "n_cells": np.zeros_like(T_GRID, dtype=int),
    "r_PLV":   np.full_like(T_GRID, np.nan, dtype=float),
    "r_PRV":   np.full_like(T_GRID, np.nan, dtype=float),
    "r_Trans": np.full_like(T_GRID, np.nan, dtype=float),
    "r_Mean":  np.full_like(T_GRID, np.nan, dtype=float),
    "r_Dom":   np.full_like(T_GRID, np.nan, dtype=float),
    "r_Opt":   np.full_like(T_GRID, np.nan, dtype=float),
    "theta_deg": np.full_like(T_GRID, np.nan, dtype=float),
}
for i, t in enumerate(T_GRID):
    mask = (ref_entry_t <= t) & ref_envelope
    if mask.sum() < 5:
        continue
    sweep["n_cells"][i] = int(mask.sum())
    case_masks = {k: mask for k in pcs}
    _, _, rv = collect_rvalues(case_masks)
    for key in ["r_PLV", "r_PRV", "r_Trans", "r_Mean", "r_Dom", "r_Opt",
                "theta_deg"]:
        sweep[key][i] = rv[key]

# ── Console summary ──────────────────────────────────────────────────────────
print()
print("=" * 98)
print("SPECTRUM PROXY CORRELATIONS (Pearson r across PAH severity spectrum)")
print("=" * 98)
print(f"\n{'region':<10} {'n':>3}   {'r_PLV':>8} {'r_PRV':>8} {'r_Trans':>8} "
      f"{'r_Mean':>8} {'r_Dom':>8} {'r_Opt':>8}   {'θ*':>7}  winner")
print("-" * 98)
for rd in REGION_DEFS:
    r = results[rd]
    cand = {k: r[k] for k in ["r_PLV", "r_PRV", "r_Trans", "r_Mean", "r_Dom"]}
    winner = max(cand.items(), key=lambda kv: abs(kv[1]))[0].replace("r_", "")
    print(f"{rd:<10} {len(r['keys']):>3}   "
          f"{r['r_PLV']:>+8.4f} {r['r_PRV']:>+8.4f} {r['r_Trans']:>+8.4f} "
          f"{r['r_Mean']:>+8.4f} {r['r_Dom']:>+8.4f} {r['r_Opt']:>+8.4f}   "
          f"{r['theta_deg']:>+6.1f}°  {winner}")

# ── Figure 1: Tracking — dual-axis W_true (left) vs proxies (right) ──────────
# Dual axes because the absolute magnitudes differ: the proxies capture only
# fiber-direction work (~40% of total), the true work captures all stress/
# strain components. Putting them on independent axes lets the eye read
# "do the curves track each other?" without being confused by the offset.
PROXY_STYLE = {
    "W_PLV":   dict(color="#1f77b4", marker="o",  label=r"$P_{LV}$"),
    "W_PRV":   dict(color="#d62728", marker="s",  label=r"$P_{RV}$"),
    "W_Trans": dict(color="#2ca02c", marker="^",  label=r"$P_{LV}-P_{RV}$"),
}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4),
                          constrained_layout=True)
for j, rd in enumerate(REGION_DEFS):
    ax = axes[j]
    r = results[rd]
    x = np.arange(len(r["keys"]))

    # Left axis: W_true density (ground truth)
    ax.plot(x, r["W_true"], "-D", color="black", lw=2.6, ms=8,
            label=r"$W_{true}$ (left axis)", zorder=5)
    ax.set_ylabel(r"$W_{true}$ density (kPa = mJ/mL)",
                  color="black", fontsize=11)
    ax.tick_params(axis="y", labelcolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{DISPLAY[k]}\n({RVESP[k]:.0f} mmHg)" for k in r["keys"]],
        fontsize=8,
    )
    ax.grid(alpha=0.25)
    ax.set_title(f"{rd.capitalize()} septum  (n={len(r['keys'])} cases)",
                 fontsize=12, fontweight="bold")

    # Right axis: proxies (density)
    ax2 = ax.twinx()
    for key, style in PROXY_STYLE.items():
        r_key = "r_" + key.replace("W_", "")
        ax2.plot(x, r[key], "-", lw=1.7, ms=6, alpha=0.95,
                 color=style["color"], marker=style["marker"],
                 label=f"{style['label']} (r={r[r_key]:+.2f})")
    ax2.set_ylabel(r"Proxy density (kPa)",
                   color="dimgray", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="dimgray")

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=9, framealpha=0.95,
               handlelength=1.8)

fig.suptitle("Septal proxy tracking across PAH",
             fontsize=12, fontweight="bold")
fig.savefig(out_dir / "fig_spectrum_tracking.png", dpi=160, bbox_inches="tight")
fig.savefig(out_dir / "fig_spectrum_tracking.pdf", bbox_inches="tight")
print(f"\nSaved {out_dir / 'fig_spectrum_tracking.png'}")

# ── Figure 2: Robustness sweep — r(t) across relaxed-septum thresholds ───────
# Strongest argument for the transmural proxy: it dominates across the ENTIRE
# sweep of septum boundaries, not just at the two nominal cutoffs. The two
# nominal cutoffs (geometric and LDRB) are annotated as vertical reference
# lines so the reader can see where they sit inside the sweep.
# r_Opt is intentionally NOT plotted — it is the supremum over the optimal
# blend angle at each sweep step, so it's always ≥ every canonical proxy
# by construction. Showing it implies a comparison that doesn't exist.
# The angle plot (fig_spectrum_angle.png) carries the optimal-blend info
# via θ* instead, which is a meaningful quantity rather than a tautology.
SWEEP_STYLE = {
    "r_PLV":   dict(color="#1f77b4", label=r"$P_{LV}$"),
    "r_PRV":   dict(color="#d62728", label=r"$P_{RV}$"),
    "r_Trans": dict(color="#2ca02c", label=r"$P_{LV}-P_{RV}$"),
    "r_Mean":  dict(color="#ff7f0e", label=r"$(P_{LV}+P_{RV})/2$", ls="--"),
}

fig3, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
ax.axhline(0.0, color="lightgray", lw=0.8, zorder=0)
ax.axhline(1.0, color="lightgray", lw=0.5, ls=":")
for key, style in SWEEP_STYLE.items():
    ax.plot(sweep["t"] * 1000.0, sweep[key],
            color=style["color"], lw=2.4 if key == "r_Trans" else 1.5,
            ls=style.get("ls", "-"), label=style["label"],
            alpha=0.95 if key == "r_Trans" else 0.75)

# Mark where the geometric-cutoff sits in the sweep (entry_t = 0 by definition)
ax.axvline(0.0, color="gray", lw=1.0, ls="--", alpha=0.7)
ax.text(0.0, -0.96, "  geometric cutoff (t=0)",
        ha="left", va="bottom", fontsize=9, color="gray")

ax.set_xlim((sweep["t"][0] - 0.001) * 1000.0,
            (sweep["t"][-1] + 0.001) * 1000.0)
ax.set_ylim(-1.05, 1.1)
ax.set_xlabel(r"Boundary relaxation threshold $t$ (mm)   "
              r""
              r"$\;\;\mathrm{mask}(t) = \{c : \max(d_{LV}, d_{RV}) - d_{epi} \leq t\} \cap \mathrm{envelope}$",
              fontsize=10)
ax.set_ylabel("Pearson r with $W_{true}$ across 7 severities", fontsize=11)
ax.grid(alpha=0.25)
ax.legend(loc="lower left", fontsize=9, ncol=3, framealpha=0.95)

# Secondary x-axis with cell counts
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
sel = np.linspace(0, len(sweep["t"]) - 1, 6).astype(int)
ax_top.set_xticks(sweep["t"][sel] * 1000.0)
ax_top.set_xticklabels([f"{int(sweep['n_cells'][i])}" for i in sel],
                        fontsize=8)
ax_top.set_xlabel("cells in sweep region", fontsize=9, color="dimgray")
ax_top.tick_params(axis="x", labelcolor="dimgray")

ax.set_title("Proxy tracking across septum sweep",
             fontsize=12, fontweight="bold")
fig3.savefig(out_dir / "fig_spectrum_sweep.png", dpi=160, bbox_inches="tight")
fig3.savefig(out_dir / "fig_spectrum_sweep.pdf", bbox_inches="tight")
print(f"Saved {out_dir / 'fig_spectrum_sweep.png'}")

# ── Figure 4: spatial filmstrip — the sweeping septum, anatomy + distance ────
# Two-row figure showing the same sweep snapshots in two complementary views:
#   Row 1: short-axis spatial slice (anatomy — viewer sees the septum grow)
#   Row 2: (d_LV, d_RV) distance space (math — region is exactly the triangle
#          {(a,b) : max(a,b) <= d_epi + t}, symmetric about the diagonal).
# Six columns at evenly-spaced t values across the sweep range.
ref_pc = pcs[ref_key]
centroids = ref_pc["centroids"]
d_lv = ref_pc["d_lv"]
d_rv = ref_pc["d_rv"]
d_epi = ref_pc["d_epi"]
ent = ref_pc["entry_t"]
env = ref_pc["envelope"]

# Identify the apex–base axis as the longest extent
extents = centroids.max(axis=0) - centroids.min(axis=0)
ax_long = int(np.argmax(extents))
xy_axes = [a for a in (0, 1, 2) if a != ax_long]
mid = float(np.median(centroids[:, ax_long]))
# Slab thickness ~6 mm (0.006 m)
slab_half = 0.006
slab_mask = np.abs(centroids[:, ax_long] - mid) < slab_half

# Choose 6 t snapshots spanning the sweep
T_SNAPS_idx = np.linspace(2, len(T_GRID) - 2, 6).astype(int)
T_SNAPS = T_GRID[T_SNAPS_idx]

fig4, axes = plt.subplots(2, len(T_SNAPS), figsize=(15.5, 6.5),
                          constrained_layout=True)

# Pre-compute spatial axis limits (so all panels share scale)
xs_all = centroids[slab_mask, xy_axes[0]] * 1000.0  # mm
ys_all = centroids[slab_mask, xy_axes[1]] * 1000.0
x_lim = (xs_all.min() - 4, xs_all.max() + 4)
y_lim = (ys_all.min() - 4, ys_all.max() + 4)

# Distance-space limits in mm
ds_max = max(d_lv.max(), d_rv.max()) * 1000.0 * 1.05
de_med = float(np.median(d_epi[env]))  # for reference triangle drawing

for col, t in enumerate(T_SNAPS):
    mask = (ent <= t) & env
    n_in = int(mask.sum())

    # ── Row 1: short-axis spatial slice ───────────────────────────────────
    ax_s = axes[0, col]
    in_slab_out = slab_mask & ~mask
    in_slab_in = slab_mask & mask
    ax_s.scatter(centroids[in_slab_out, xy_axes[0]] * 1000.0,
                 centroids[in_slab_out, xy_axes[1]] * 1000.0,
                 s=20, c="lightgray", edgecolors="none", alpha=0.65)
    ax_s.scatter(centroids[in_slab_in, xy_axes[0]] * 1000.0,
                 centroids[in_slab_in, xy_axes[1]] * 1000.0,
                 s=22, c="#c0392b", edgecolors="none", alpha=0.95)
    ax_s.set_xlim(x_lim)
    ax_s.set_ylim(y_lim)
    ax_s.set_aspect("equal")
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    ax_s.set_title(f"$t={t*1000:+.1f}$ mm   ({n_in} cells)",
                   fontsize=10, fontweight="bold")
    if col == 0:
        ax_s.set_ylabel("short-axis slice\n(anatomy view)", fontsize=10)

    # ── Row 2: distance-space view ────────────────────────────────────────
    ax_d = axes[1, col]
    # All cells in the envelope, plotted as (d_LV, d_RV) in mm
    ax_d.scatter(d_lv[env] * 1000.0, d_rv[env] * 1000.0,
                 s=14, c="lightgray", edgecolors="none", alpha=0.6)
    ax_d.scatter(d_lv[mask] * 1000.0, d_rv[mask] * 1000.0,
                 s=15, c="#c0392b", edgecolors="none", alpha=0.95)
    # Reference: the boundary line max(d_LV, d_RV) = d_epi + t.
    # Drawn at the median d_epi in the envelope as a representative slope.
    boundary = (de_med + t) * 1000.0
    ax_d.axvline(boundary, color="#2c3e50", lw=1.0, alpha=0.7, ls="--")
    ax_d.axhline(boundary, color="#2c3e50", lw=1.0, alpha=0.7, ls="--")
    ax_d.plot([0, ds_max], [0, ds_max], color="#7f8c8d", lw=0.8, ls=":")
    ax_d.set_xlim(0, ds_max)
    ax_d.set_ylim(0, ds_max)
    ax_d.set_aspect("equal")
    ax_d.set_xticks([0, 10, 20, 30, 40])
    ax_d.set_yticks([0, 10, 20, 30, 40])
    ax_d.tick_params(labelsize=7)
    if col == 0:
        ax_d.set_ylabel("$d_{RV}$ (mm)", fontsize=10)
        ax_d.text(-0.45, 1.15,
                  "distance space\n(math view)",
                  transform=ax_d.transAxes, fontsize=10, ha="left",
                  va="bottom")
    if col == len(T_SNAPS) - 1:
        ax_d.set_xlabel("$d_{LV}$ (mm)", fontsize=10)

# Math definition banner above the figure
fig4.suptitle(
    "Sweeping septum boundary  —  "
    r"$\mathrm{mask}(t) = \{c \;:\; \max(d_{LV}(c), d_{RV}(c)) - d_{epi}(c) \leq t\} \cap \mathrm{envelope}$"
    "\n(symmetric in $d_{LV}\\leftrightarrow d_{RV}$ — the sweep cannot favour LV or RV)",
    fontsize=11, fontweight="bold")
fig4.savefig(out_dir / "fig_spectrum_sweep_anatomy.png", dpi=160,
             bbox_inches="tight")
fig4.savefig(out_dir / "fig_spectrum_sweep_anatomy.pdf", bbox_inches="tight")
print(f"Saved {out_dir / 'fig_spectrum_sweep_anatomy.png'}")

# ── Figure 2: Stats heatmap — 3 canonical proxies only ──────────────────────
# r_Opt is tautological (supremum over θ, always ≥ the canonical r values),
# r_Mean / r_Dom are minor alternatives. They go into the separate angle
# figure below, not the heatmap.
R_KEYS = ["r_PLV", "r_PRV", "r_Trans"]
R_LABELS = [r"$P_{LV}$", r"$P_{RV}$", r"$P_{LV}-P_{RV}$"]
rows_labels = [f"{rd.capitalize()} septum" for rd in REGION_DEFS]
mat = np.array([[results[rd][k] for k in R_KEYS] for rd in REGION_DEFS])

fig2, ax = plt.subplots(figsize=(5.5, 2.2), constrained_layout=True)
im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(R_LABELS)))
ax.set_xticklabels(R_LABELS, fontsize=11)
ax.set_yticks(range(len(rows_labels)))
ax.set_yticklabels(rows_labels, fontsize=10)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat[i, j]
        col = "white" if abs(val) > 0.55 else "black"
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                color=col, fontsize=11, fontweight="bold")
cbar = fig2.colorbar(im, ax=ax, shrink=0.9)
cbar.set_label("Pearson r")
fig2.savefig(out_dir / "fig_spectrum_stats.png", dpi=160, bbox_inches="tight")
fig2.savefig(out_dir / "fig_spectrum_stats.pdf", bbox_inches="tight")
print(f"Saved {out_dir / 'fig_spectrum_stats.png'}")

# ── Figure 2b: Optimal-blend angle on a 1D axis ─────────────────────────────
# What linear combination of P_LV and P_RV does the data actually choose?
# Canonical reference angles:
#   0°   = pure P_LV       +90° = pure P_RV       -45° = transmural
# θ* close to -45° → the data picks the transmural direction.
CANON_ANG = [(0, r"$P_{LV}$", "#1f77b4"),
             (90, r"$P_{RV}$", "#d62728"),
             (-45, r"$P_{LV}-P_{RV}$", "#2ca02c")]

fig2b, ax2b = plt.subplots(figsize=(8.5, 2.2), constrained_layout=True)
for ang, lab, col in CANON_ANG:
    ax2b.axvline(ang, color=col, lw=2.0, alpha=0.6)
    ax2b.text(ang, 1.08, lab, ha="center", va="bottom",
               color=col, fontsize=10, fontweight="bold")
for i, rd in enumerate(REGION_DEFS):
    theta = results[rd]["theta_deg"]
    ax2b.plot(theta, i, "o", ms=12, color="black", zorder=5)
    ax2b.text(theta, i, f"  {theta:+.0f}°", ha="left", va="center", fontsize=9)
ax2b.set_yticks(range(len(REGION_DEFS)))
ax2b.set_yticklabels([f"{rd.capitalize()} septum" for rd in REGION_DEFS], fontsize=10)
ax2b.set_xlim(-100, 100)
ax2b.set_ylim(-0.5, len(REGION_DEFS) - 0.5 + 0.4)
ax2b.set_xlabel("Optimal blend angle θ*   (cos(θ)·$P_{LV}$ + sin(θ)·$P_{RV}$)", fontsize=10)
ax2b.grid(alpha=0.25, axis="x")
ax2b.spines["top"].set_visible(False)
ax2b.spines["right"].set_visible(False)
fig2b.savefig(out_dir / "fig_spectrum_angle.png", dpi=160, bbox_inches="tight")
fig2b.savefig(out_dir / "fig_spectrum_angle.pdf", bbox_inches="tight")
print(f"Saved {out_dir / 'fig_spectrum_angle.png'}")

# ── Save raw ──────────────────────────────────────────────────────────────────
save_dict = {"severity_keys": np.array(KEYS)}
for rd, r in results.items():
    for k, v in r.items():
        if k == "keys":
            save_dict[f"{rd}_keys"] = np.array(v)
        else:
            save_dict[f"{rd}_{k}"] = v
for k, v in sweep.items():
    save_dict[f"sweep_{k}"] = v
np.savez(out_dir / "spectrum_raw.npz", **save_dict)
print(f"Saved {out_dir / 'spectrum_raw.npz'}")
print("\nDone.")
