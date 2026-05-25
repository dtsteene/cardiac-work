#!/usr/bin/env python3
"""
plot_three_bugs_audit.py

Build the chapter-3 figure that shows the cumulative effect of the two
boundary fixes on the energy-budget closure, on the sPAP22 case.

Reads results/analysis/three_bugs_audit/sPAP22_closure_variants.npy, written
by audit_three_bugs_energy_budget.py.

Layout: single-panel bar chart of |R(T)|, the cycle-end closure residual,
on a log y-axis, with each bar annotated by its value in mJ and the
reduction factor against the previous bar.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(
    "results/analysis/three_bugs_audit/sPAP22_closure_variants.npy"
)
OUT_DIR = Path("results/analysis/three_bugs_audit")
THESIS_FIGURES_DIR = Path("/home/dtsteene/D1/RV/figures")


# Two-bug story: V1 is the pre-fix state (with both boundary bugs), V2 adds
# the cavity-pressure fix, V3 adds the Robin fix. V0 (no comp-stress work)
# would add an additional bar that differs from V1 by ~1.2e-5 J — invisible
# on the log axis and not what the audit is meant to test, so we drop it.
VARIANT_ORDER = ["V1_fix_dev", "V2_fix_press", "V3_production"]
VARIANT_LABELS = {
    "V1_fix_dev":    "buggy boundary\npostprocessors",
    "V2_fix_press":  "+ cavity-\npressure fix",
    "V3_production": "+ Robin fix\n(production)",
}
VARIANT_COLORS = {
    "V1_fix_dev":    "#a83232",
    "V2_fix_press":  "#1d4ed8",
    "V3_production": "#15803d",
}


def fmt_residual(j: float) -> str:
    """Format a residual value in J as a short string in mJ or µJ."""
    mj = abs(j) * 1e3
    if mj >= 1.0:
        return f"{mj:.2f} mJ"
    if mj >= 1e-3:
        return f"{mj * 1e3:.2f} µJ"
    return f"{j:.2e} J"


def main() -> None:
    blob = np.load(DEFAULT_INPUT, allow_pickle=True).item()
    bug3_available = bool(blob.get("bug3_available", False))

    res_abs = np.array([abs(blob[n]["final_residual_J"]) for n in VARIANT_ORDER])
    res_rel = np.array([abs(blob[n]["final_residual_rel"]) for n in VARIANT_ORDER])
    labels = [VARIANT_LABELS[n] for n in VARIANT_ORDER]
    colors = [VARIANT_COLORS[n] for n in VARIANT_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    xs = np.arange(len(VARIANT_ORDER))
    bars = ax.bar(xs, res_abs * 1e3,  # J → mJ
                  color=colors, edgecolor="white", linewidth=1.5,
                  width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"cycle-end closure residual $|R(T)|$ (mJ)", fontsize=11)
    ax.set_title("Closure residual after each successive fix",
                 fontsize=11)

    # Value labels on top of each bar
    for bar, j_value, rel in zip(bars, res_abs, res_rel):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h * 1.45,
                f"{fmt_residual(j_value)}\n({rel:.0e})",
                ha="center", va="bottom",
                fontsize=9, color="black", linespacing=1.3)

    # Clean up the axes — no grid, lighter spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Padded y-limits so annotations don't get clipped
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 3.5)

    if not bug3_available:
        fig.text(0.5, 0.02,
                 "Bug-3 (Robin) replay missing — last two bars share production Robin.",
                 ha="center", fontsize=8, color="#b91c1c", style="italic")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig_closure_audit.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.pdf')}")

    if THESIS_FIGURES_DIR.exists():
        thesis_out = THESIS_FIGURES_DIR / "fig_closure_audit.png"
        fig.savefig(thesis_out, dpi=180, bbox_inches="tight")
        print(f"Wrote {thesis_out}")


if __name__ == "__main__":
    main()
