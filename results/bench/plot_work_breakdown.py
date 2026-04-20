#!/usr/bin/env python3
"""
plot_work_breakdown.py — Work breakdown across disease spectrum

Generates one PNG per septum definition showing:
  - LV panel: true work + P_LV proxy
  - RV panel: true work + P_RV proxy
  - Septum panel: true work + all 3 proxies (opacity ∝ |r|)
  - Correlation table (all 9 region×proxy combinations)

Usage:
    python3 plot_work_breakdown.py results/sims/2026-04-12/UKB_1beats_run_*
    python3 plot_work_breakdown.py --output-dir results/analysis/presentation results/sims/...
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
parser.add_argument("--output-dir", type=Path, default=None)
args = parser.parse_args()

out_dir = args.output_dir or Path("results/analysis/presentation")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load cases ──────────────────────────────────────────────────────────────
cases = []
for d in args.result_dirs:
    d = d.resolve()
    pc_path = d / "per_cell_data.npz"
    if not pc_path.exists():
        print(f"  SKIP {d.name}: no per_cell_data.npz")
        continue
    pc = np.load(pc_path)
    # Solver pressure (prefer new name, fall back to old)
    sp_path = d / "solver" / "solver_cavity_pressure_mmHg.npy"
    if not sp_path.exists():
        sp_path = d / "solver" / "pressure_history.npy"
    sp = np.load(sp_path)

    desc = (d / "run_description.txt").read_text().strip() if (d / "run_description.txt").exists() else d.name
    for pfx in ["Phase1 shared-mesh v2circ ", "Phase1 v2circ ", "v2circ "]:
        desc = desc.replace(pfx, "")
    label = desc.split()[0]
    cases.append({"label": label, "rv_esp": float(sp[:, 1].max()),
                  "pc": {k: pc[k] for k in pc.files}})

cases.sort(key=lambda c: c["rv_esp"])
n = len(cases)
labels = [c["label"] for c in cases]
print(f"Loaded {n} cases")

def safe_r(x, y):
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 3:
        return np.nan
    return pearsonr(x, y)[0]

def r_to_alpha(r):
    ar = abs(r) if not np.isnan(r) else 0
    return 0.25 + 0.75 * ar

# ── Plot function ───────────────────────────────────────────────────────────
def make_plot(cases, septum_key, title_label, out_name):
    n_sep = int(np.mean([c["pc"][septum_key].sum() for c in cases]))
    x = np.arange(n)
    ms, lw = 8, 2.2

    regions_masks = {
        "LV": lambda c: c["pc"]["region_tags"] == 1,
        "RV": lambda c: c["pc"]["region_tags"] == 2,
        "Septum": lambda c: c["pc"][septum_key],
    }
    proxy_keys = {"PLV": "proxy_PLV_ll", "PRV": "proxy_PRV_ll", "Trans": "proxy_Trans_ll"}

    table_r = {}
    work = {}
    for rname, mask_fn in regions_masks.items():
        W_true = np.array([c["pc"]["w_total"][mask_fn(c)].sum() for c in cases])
        work[(rname, "true")] = W_true
        table_r[rname] = {}
        for pname, pkey in proxy_keys.items():
            W_proxy = np.array([c["pc"][pkey][mask_fn(c)].sum() for c in cases])
            work[(rname, pname)] = W_proxy
            table_r[rname][pname] = safe_r(W_true, W_proxy)

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1],
                           hspace=0.35, wspace=0.3)

    # ── LV ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, work[("LV","true")], "-D", color="k", lw=lw+0.5, ms=ms+1, zorder=5)
    ax.set_ylabel("W$_{true}$ (J)", fontsize=10); ax.grid(alpha=0.2)
    ax_r = ax.twinx()
    ax_r.plot(x, work[("LV","PLV")], "-o", color="C0", lw=lw, ms=ms-1, alpha=0.85)
    ax_r.set_ylabel("W$_{proxy}$ (J)", fontsize=10, color="C0")
    ax_r.tick_params(axis="y", labelcolor="C0")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_title("LV", fontsize=12, fontweight="bold")
    ax.legend([Line2D([0],[0], color="k", marker="D", lw=lw+0.5, ms=ms),
               Line2D([0],[0], color="C0", marker="o", lw=lw, ms=ms-1)],
              ["W$_{true}$ (left)", "$P_{LV}$ proxy (right)"], fontsize=8, loc="lower left")

    # ── RV ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, work[("RV","true")], "-D", color="k", lw=lw+0.5, ms=ms+1, zorder=5)
    ax.set_ylabel("W$_{true}$ (J)", fontsize=10); ax.grid(alpha=0.2)
    ax_r = ax.twinx()
    ax_r.plot(x, work[("RV","PRV")], "-s", color="C3", lw=lw, ms=ms-1, alpha=0.85)
    ax_r.set_ylabel("W$_{proxy}$ (J)", fontsize=10, color="C3")
    ax_r.tick_params(axis="y", labelcolor="C3")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_title("RV", fontsize=12, fontweight="bold")
    ax.legend([Line2D([0],[0], color="k", marker="D", lw=lw+0.5, ms=ms),
               Line2D([0],[0], color="C3", marker="s", lw=lw, ms=ms-1)],
              ["W$_{true}$ (left)", "$P_{RV}$ proxy (right)"], fontsize=8, loc="lower left")

    # ── Septum ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, work[("Septum","true")], "-D", color="k", lw=lw+0.5, ms=ms+1, zorder=5)
    ax.set_ylabel("W$_{true}$ (J)", fontsize=10); ax.grid(alpha=0.2)
    ax_r = ax.twinx()
    r_plv = table_r["Septum"]["PLV"]
    r_prv = table_r["Septum"]["PRV"]
    r_trans = table_r["Septum"]["Trans"]
    ax_r.plot(x, work[("Septum","PLV")],  "-o", color="C0", lw=lw, ms=ms-1, alpha=r_to_alpha(r_plv))
    ax_r.plot(x, work[("Septum","PRV")],  "-s", color="C3", lw=lw, ms=ms-1, alpha=r_to_alpha(r_prv))
    ax_r.plot(x, work[("Septum","Trans")], "-^", color="C2", lw=lw, ms=ms-1, alpha=r_to_alpha(r_trans))
    ax_r.set_ylabel("W$_{proxy}$ (J)", fontsize=10, color="gray")
    ax_r.tick_params(axis="y", labelcolor="gray")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Septum ({title_label}, {n_sep} cells)", fontsize=12, fontweight="bold")
    ax.legend([Line2D([0],[0], color="k", marker="D", lw=lw+0.5, ms=ms),
               Line2D([0],[0], color="C0", marker="o", lw=lw, ms=ms-1),
               Line2D([0],[0], color="C3", marker="s", lw=lw, ms=ms-1),
               Line2D([0],[0], color="C2", marker="^", lw=lw, ms=ms-1)],
              ["W$_{true}$ (left)", "$P_{LV}$", "$P_{RV}$", "$P_{LV}-P_{RV}$"],
              fontsize=8, loc="lower left", title="right axis:", title_fontsize=7)

    # ── Correlation table ─────────────────────────────────────────
    ax_tab = fig.add_subplot(gs[1, 1])
    ax_tab.axis("off")
    ax_tab.set_title("Pearson r  (W$_{true}$ vs W$_{proxy}$)",
                     fontsize=12, fontweight="bold", pad=15)
    col_labels = ["$P_{LV}$", "$P_{RV}$", "$P_{Trans}$"]
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
                rc.append((1 - 0.35*ar, g, 1 - 0.35*ar))
            else:
                rc.append((1, 1 - 0.35*ar, 1 - 0.35*ar))
        cell_text.append(rt); cell_colors.append(rc)

    table = ax_tab.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                         cellColours=cell_colors, cellLoc="center",
                         bbox=[0.15, 0.25, 0.8, 0.55])
    table.auto_set_font_size(False); table.set_fontsize(13)
    for i, rname in enumerate(row_labels):
        vals = [abs(table_r[rname][p]) for p in ["PLV", "PRV", "Trans"]]
        best_col = np.argmax(vals)
        table[i+1, best_col].set_text_props(fontweight="bold", fontsize=14)

    ax_tab.text(0.55, 0.1, f"Septum: {title_label} ({n_sep} cells)  |  "
                f"Strain: longitudinal  |  n={n}",
                transform=ax_tab.transAxes, fontsize=9, ha="center", va="bottom",
                style="italic", color="gray")

    fig.suptitle(f"Work across disease spectrum — {title_label} septum\n"
                 f"Black = true work (left axis)     Color = proxy (right axis)     "
                 f"Opacity $\\propto$ |r|", fontsize=12)
    fig.savefig(out_dir / out_name, dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / out_name}")

make_plot(cases, "is_geometric_septum", "Geometric", "work_breakdown_geometric.png")
make_plot(cases, "is_ldrb_septum", "LDRB", "work_breakdown_ldrb.png")
print("Done.")
