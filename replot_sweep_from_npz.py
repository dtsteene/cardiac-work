#!/usr/bin/env python3
"""
replot_sweep_from_npz.py — Regenerate sweep_sensitivity.png from a cached
sweep_raw.npz in the analyze_spectrum.py fig_spectrum_sweep style. Use when
the original sim dirs aren't available to rerun analyze_sweep.py from scratch.

Usage:
    python3 replot_sweep_from_npz.py results/analysis/sweep_6beat_canonical \
                                     results/analysis/sweep_6beat_ed
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def replot(sweep_dir: Path, title_suffix: str = ""):
    sweep_dir = sweep_dir.resolve()
    raw = np.load(sweep_dir / "sweep_raw.npz", allow_pickle=True)
    t_mm = raw["t_mm"]
    n_cases = int(len(raw["case_labels"])) if "case_labels" in raw.files else 0
    mean_cells_at_t = raw["n_cells_sweep"].mean(axis=1)

    SWEEP_STYLE = {
        "r_PLV":   dict(color="#1f77b4", label=r"$P_{LV}$",
                        values=raw["r_PLV"]),
        "r_PRV":   dict(color="#d62728", label=r"$P_{RV}$",
                        values=raw["r_PRV"]),
        "r_Trans": dict(color="#2ca02c", label=r"$P_{LV}-P_{RV}$",
                        values=raw["r_Trans"]),
        "r_Mean":  dict(color="#ff7f0e", label=r"$(P_{LV}+P_{RV})/2$",
                        values=raw["r_mean"], ls="--"),
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    ax.axhline(0.0, color="lightgray", lw=0.8, zorder=0)
    ax.axhline(1.0, color="lightgray", lw=0.5, ls=":")
    for key, style in SWEEP_STYLE.items():
        ax.plot(t_mm, style["values"],
                color=style["color"],
                lw=2.4 if key == "r_Trans" else 1.5,
                ls=style.get("ls", "-"),
                label=style["label"],
                alpha=0.95 if key == "r_Trans" else 0.75)

    ax.axvline(0.0, color="gray", lw=1.0, ls="--", alpha=0.7)
    ax.text(0.0, -0.96, "  geometric cutoff (t=0)",
            ha="left", va="bottom", fontsize=9, color="gray")

    ax.set_xlim(t_mm[0] - 0.5, t_mm[-1] + 0.5)
    ax.set_ylim(-1.05, 1.1)
    ax.set_xlabel(r"Boundary relaxation threshold $t$ (mm)   "
                  r"$\;\;\mathrm{mask}(t) = \{c : \mathrm{entry}_t(c) \leq t\} "
                  r"\cap \mathrm{envelope}$",
                  fontsize=10)
    ax.set_ylabel(f"Pearson r with $W_{{true}}$ across {n_cases} severities",
                  fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, ncol=3, framealpha=0.95)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    sel = np.linspace(0, len(t_mm) - 1, 6).astype(int)
    ax_top.set_xticks(t_mm[sel])
    ax_top.set_xticklabels([f"{int(mean_cells_at_t[i])}" for i in sel],
                           fontsize=8)
    ax_top.set_xlabel("cells in sweep region", fontsize=9, color="dimgray")
    ax_top.tick_params(axis="x", labelcolor="dimgray")

    title = "Proxy tracking across septum sweep"
    if title_suffix:
        title += f"  ({title_suffix})"
    ax.set_title(title, fontsize=12, fontweight="bold")

    out_png = sweep_dir / "sweep_sensitivity.png"
    out_pdf = sweep_dir / "sweep_sensitivity.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    label_map = {
        "sweep_6beat_canonical": "canonical tagging (unloaded geometry)",
        "sweep_6beat_ed":        "ED tagging (per-case loaded geometry)",
    }
    for d in sys.argv[1:]:
        p = Path(d)
        replot(p, title_suffix=label_map.get(p.name, ""))
