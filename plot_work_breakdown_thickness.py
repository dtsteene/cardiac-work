#!/usr/bin/env python3
"""
plot_work_breakdown_thickness.py — Work breakdown across thickness variants

Mirrors plot_work_breakdown.py (which sweeps disease severity) but
instead sweeps the RV free wall thickness at fixed circulation. Produces
one figure per (severity, septum_definition) combination:

  work_breakdown_thickness_severe_ldrb.png
  work_breakdown_thickness_severe_geometric.png
  work_breakdown_thickness_healthy_ldrb.png
  work_breakdown_thickness_healthy_geometric.png

Each figure has the same 2×2 layout as the spectrum version:
  - LV free wall panel    (W_true + P_LV proxy)
  - RV free wall panel    (W_true + P_RV proxy)   ← cardiologists care most
  - Septum panel          (W_true + all 3 proxies, opacity ∝ |r|)
  - 9-cell correlation table (3 regions × 3 proxies)

Usage:
    python3 plot_work_breakdown_thickness.py results/sims/2026-04-14/UKB_1beats_run_*
    python3 plot_work_breakdown_thickness.py --output-dir results/analysis/work_breakdown_thickness results/sims/...
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("result_dirs", nargs="+", type=Path)
parser.add_argument("--output-dir", type=Path,
                    default=Path("results/analysis/work_breakdown_thickness"))
args = parser.parse_args()
out_dir = args.output_dir
out_dir.mkdir(parents=True, exist_ok=True)

# Canonical variant order (thinnest → thickest) for each family
RVFW_ORDER = ["rvfw_03mm", "rvfw_4p5mm", "rvfw_06mm", "rvfw_7p5mm",
              "rvfw_09mm", "rvfw_10p5mm", "rvfw_12mm"]
LVFW_ORDER = ["lvfw_03mm", "lvfw_06mm", "lvfw_09mm", "lvfw_12mm"]
VARIANT_ORDER = RVFW_ORDER + LVFW_ORDER
VARIANT_LABEL = {
    "rvfw_03mm":  "+3",   "rvfw_4p5mm": "+4.5", "rvfw_06mm":  "+6",
    "rvfw_7p5mm": "+7.5", "rvfw_09mm":  "+9",   "rvfw_10p5mm": "+10.5",
    "rvfw_12mm":  "+12",
    "lvfw_03mm":  "+3",   "lvfw_06mm":  "+6",
    "lvfw_09mm":  "+9",   "lvfw_12mm":  "+12",
}

# ── Load cases (each: severity, variant, per_cell) ─────────────────────────
cases = []
for d in args.result_dirs:
    d = d.resolve()
    pc_path = d / "per_cell_data.npz"
    if not pc_path.exists():
        print(f"  SKIP {d.name}: no per_cell_data.npz")
        continue
    desc_path = d / "run_description.txt"
    if not desc_path.exists():
        print(f"  SKIP {d.name}: no run_description.txt")
        continue
    desc = desc_path.read_text().strip()
    # Parse "Phase4v2 thickness rvfw_03mm severe 1beats ..." or similar
    tokens = desc.split()
    variant = next((t for t in tokens if t in VARIANT_ORDER), None)
    severity = next((t for t in tokens if t in ("healthy", "severe")), None)
    if variant is None or severity is None:
        print(f"  SKIP {d.name}: cannot parse variant/severity from: {desc}")
        continue

    pc = np.load(pc_path)
    cases.append({
        "variant": variant, "severity": severity,
        "label": VARIANT_LABEL[variant],
        "pc": {k: pc[k] for k in pc.files},
    })

print(f"Loaded {len(cases)} cases")
for c in cases:
    print(f"  {c['severity']:<8} {c['variant']:<10} → {c['label']}")


def safe_r(x, y):
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 3:
        return np.nan
    return pearsonr(x, y)[0]


def r_to_alpha(r):
    ar = abs(r) if not np.isnan(r) else 0
    return 0.25 + 0.75 * ar


def make_plot(cases_subset, septum_key, title_septum, title_sev, out_name,
              order=VARIANT_ORDER, family_label="thickness"):
    """Identical layout to plot_work_breakdown.py.make_plot, but for a
    thickness sweep at a fixed severity."""
    # Sort by variant order within the supplied family
    cases_sorted = sorted(cases_subset, key=lambda c: order.index(c["variant"]))
    n = len(cases_sorted)
    if n < 2:
        print(f"  SKIP {out_name}: only {n} case(s), need ≥2")
        return
    labels = [c["label"] for c in cases_sorted]
    n_sep = int(np.mean([c["pc"][septum_key].sum() for c in cases_sorted]))
    x = np.arange(n)
    ms, lw = 8, 2.2

    regions_masks = {
        "LV":     lambda c: c["pc"]["region_tags"] == 1,
        "RV":     lambda c: c["pc"]["region_tags"] == 2,
        "Septum": lambda c: c["pc"][septum_key],
    }
    proxy_keys = {"PLV": "proxy_PLV_ll",
                  "PRV": "proxy_PRV_ll",
                  "Trans": "proxy_Trans_ll"}

    table_r = {}
    work = {}
    for rname, mask_fn in regions_masks.items():
        W_true = np.array([c["pc"]["w_total"][mask_fn(c)].sum() for c in cases_sorted])
        work[(rname, "true")] = W_true
        table_r[rname] = {}
        for pname, pkey in proxy_keys.items():
            W_proxy = np.array([c["pc"][pkey][mask_fn(c)].sum() for c in cases_sorted])
            work[(rname, pname)] = W_proxy
            table_r[rname][pname] = safe_r(W_true, W_proxy)

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1],
                            hspace=0.35, wspace=0.3)

    # ── LV ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, work[("LV", "true")], "-D", color="k", lw=lw + 0.5, ms=ms + 1, zorder=5)
    ax.set_ylabel(r"$W_{true}$ (J)", fontsize=10)
    ax.grid(alpha=0.2)
    ax_r = ax.twinx()
    ax_r.plot(x, work[("LV", "PLV")], "-o", color="C0", lw=lw, ms=ms - 1, alpha=0.85)
    ax_r.set_ylabel(r"$W_{proxy}$ (J)", fontsize=10, color="C0")
    ax_r.tick_params(axis="y", labelcolor="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    ax.set_title("LV free wall", fontsize=12, fontweight="bold")
    r_lv = table_r["LV"]["PLV"]
    ax.legend([Line2D([0], [0], color="k", marker="D", lw=lw + 0.5, ms=ms),
                Line2D([0], [0], color="C0", marker="o", lw=lw, ms=ms - 1)],
               [r"$W_{true}$ (left)",
                fr"$P_{{LV}}$ proxy (right)  r={r_lv:+.2f}"],
               fontsize=8, loc="lower left")

    # ── RV (cardiologist focus) ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, work[("RV", "true")], "-D", color="k", lw=lw + 0.5, ms=ms + 1, zorder=5)
    ax.set_ylabel(r"$W_{true}$ (J)", fontsize=10)
    ax.grid(alpha=0.2)
    ax_r = ax.twinx()
    ax_r.plot(x, work[("RV", "PRV")], "-s", color="C3", lw=lw, ms=ms - 1, alpha=0.85)
    ax_r.set_ylabel(r"$W_{proxy}$ (J)", fontsize=10, color="C3")
    ax_r.tick_params(axis="y", labelcolor="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    ax.set_title("RV free wall", fontsize=12, fontweight="bold")
    r_rv = table_r["RV"]["PRV"]
    ax.legend([Line2D([0], [0], color="k", marker="D", lw=lw + 0.5, ms=ms),
                Line2D([0], [0], color="C3", marker="s", lw=lw, ms=ms - 1)],
               [r"$W_{true}$ (left)",
                fr"$P_{{RV}}$ proxy (right)  r={r_rv:+.2f}"],
               fontsize=8, loc="lower left")

    # ── Septum ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, work[("Septum", "true")], "-D", color="k", lw=lw + 0.5, ms=ms + 1, zorder=5)
    ax.set_ylabel(r"$W_{true}$ (J)", fontsize=10)
    ax.grid(alpha=0.2)
    ax_r = ax.twinx()
    r_plv = table_r["Septum"]["PLV"]
    r_prv = table_r["Septum"]["PRV"]
    r_trans = table_r["Septum"]["Trans"]
    ax_r.plot(x, work[("Septum", "PLV")],  "-o", color="C0", lw=lw, ms=ms - 1, alpha=r_to_alpha(r_plv))
    ax_r.plot(x, work[("Septum", "PRV")],  "-s", color="C3", lw=lw, ms=ms - 1, alpha=r_to_alpha(r_prv))
    ax_r.plot(x, work[("Septum", "Trans")], "-^", color="C2", lw=lw, ms=ms - 1, alpha=r_to_alpha(r_trans))
    ax_r.set_ylabel(r"$W_{proxy}$ (J)", fontsize=10, color="gray")
    ax_r.tick_params(axis="y", labelcolor="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    ax.set_title(f"Septum ({title_septum}, {n_sep} cells)", fontsize=12, fontweight="bold")
    ax.legend([Line2D([0], [0], color="k", marker="D", lw=lw + 0.5, ms=ms),
                Line2D([0], [0], color="C0", marker="o", lw=lw, ms=ms - 1),
                Line2D([0], [0], color="C3", marker="s", lw=lw, ms=ms - 1),
                Line2D([0], [0], color="C2", marker="^", lw=lw, ms=ms - 1)],
               [r"$W_{true}$ (left)", r"$P_{LV}$", r"$P_{RV}$", r"$P_{LV}-P_{RV}$"],
               fontsize=8, loc="lower left", title="right axis:", title_fontsize=7)

    # ── Correlation table ─────────────────────────────────────────
    ax_tab = fig.add_subplot(gs[1, 1])
    ax_tab.axis("off")
    ax_tab.set_title(r"Pearson r  ($W_{true}$ vs $W_{proxy}$)",
                      fontsize=12, fontweight="bold", pad=15)
    col_labels = [r"$P_{LV}$", r"$P_{RV}$", r"$P_{Trans}$"]
    row_labels = ["LV", "RV", "Septum"]
    cell_text, cell_colors = [], []
    for rname in row_labels:
        rt, rc = [], []
        for pname in ["PLV", "PRV", "Trans"]:
            r_val = table_r[rname][pname]
            rt.append(f"{r_val:+.3f}")
            ar = abs(r_val)
            if r_val > 0:
                g = min(1.0, 0.6 + 0.4 * ar)
                rc.append((1 - 0.35 * ar, g, 1 - 0.35 * ar))
            else:
                rc.append((1, 1 - 0.35 * ar, 1 - 0.35 * ar))
        cell_text.append(rt)
        cell_colors.append(rc)

    table = ax_tab.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                          cellColours=cell_colors, cellLoc="center",
                          bbox=[0.15, 0.25, 0.8, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for i, rname in enumerate(row_labels):
        vals = [abs(table_r[rname][p]) for p in ["PLV", "PRV", "Trans"]]
        best_col = int(np.argmax(vals))
        table[i + 1, best_col].set_text_props(fontweight="bold", fontsize=14)

    ax_tab.text(0.55, 0.08,
                 f"Septum: {title_septum} ({n_sep} cells)  |  "
                 f"Strain: longitudinal  |  n={n} thickness variants\n"
                 f"Circulation: {title_sev}  |  Family: {family_label}",
                 transform=ax_tab.transAxes, fontsize=9, ha="center", va="bottom",
                 style="italic", color="gray")

    fig.suptitle(f"Work across {family_label} — {title_septum} septum, "
                  f"{title_sev} circulation\n"
                  f"Black = true work (left axis)     Color = proxy (right axis)     "
                  f"Opacity $\\propto$ |r|", fontsize=12)
    fig.savefig(out_dir / out_name, dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / out_name}")


# ── Produce the 2x2x2 = 8 combinations (severity × septum_def × family) ────
for sev_key, sev_title in [("severe", "severe PAH"),
                            ("healthy", "healthy")]:
    subset = [c for c in cases if c["severity"] == sev_key]
    for sep_key, sep_name in [("is_ldrb_septum",      "LDRB"),
                               ("is_geometric_septum", "Geometric")]:
        # RVFW family
        rvfw_subset = [c for c in subset if c["variant"] in RVFW_ORDER]
        if rvfw_subset:
            make_plot(rvfw_subset, sep_key, sep_name, sev_title,
                       f"work_breakdown_thickness_{sev_key}_{sep_name.lower()}_rvfw.png",
                       order=RVFW_ORDER, family_label="RV free wall thickness (mm)")
        # LVFW family
        lvfw_subset = [c for c in subset if c["variant"] in LVFW_ORDER]
        if lvfw_subset:
            make_plot(lvfw_subset, sep_key, sep_name, sev_title,
                       f"work_breakdown_thickness_{sev_key}_{sep_name.lower()}_lvfw.png",
                       order=LVFW_ORDER, family_label="LV free wall thickness (mm)")

print("Done.")
