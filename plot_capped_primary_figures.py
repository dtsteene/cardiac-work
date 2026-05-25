#!/usr/bin/env python3
"""Figures needed when the capped RV-EDP sweep is the primary thesis sweep."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/dtsteene/D1")
ANALYSIS = ROOT / "cardiac-work" / "results" / "analysis" / "capped_rvedp_sweep"
THESIS_FIG = ROOT / "RV" / "figures"


def case_number(label: str) -> int:
    match = re.search(r"(\d+)", label)
    return int(match.group(1)) if match else 0


def save(fig: plt.Figure, stem: str) -> None:
    THESIS_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(THESIS_FIG / f"{stem}.{ext}", dpi=260 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def last_beat(metrics: dict[str, object]) -> slice:
    time = np.asarray(metrics["time"], dtype=float)
    n = len(time) // 6
    return slice(5 * n, 6 * n)


def pv_loop_figure() -> None:
    rows = pd.read_csv(ANALYSIS / "h5_sweep_case_values.csv")
    rows["case_num"] = rows["case"].map(case_number)
    rows = rows.sort_values("case_num")
    rvsp = rows["RV_ESP_mmHg"].to_numpy(float)

    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(rvsp.min(), rvsp.max())

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), constrained_layout=True)
    for _, row in rows.iterrows():
        metrics = np.load(Path(row["result_dir"]) / "metrics" / "metrics_downsample_1.npy", allow_pickle=True).item()
        beat = last_beat(metrics)
        color = cmap(norm(float(row["RV_ESP_mmHg"])))
        axes[0].plot(
            np.asarray(metrics["V_LV_FEM"], dtype=float)[beat],
            np.asarray(metrics["p_LV"], dtype=float)[beat],
            color=color,
            linewidth=1.35,
            alpha=0.95,
        )
        axes[1].plot(
            np.asarray(metrics["V_RV_FEM"], dtype=float)[beat],
            np.asarray(metrics["p_RV"], dtype=float)[beat],
            color=color,
            linewidth=1.35,
            alpha=0.95,
        )

    axes[0].set_title("LV")
    axes[1].set_title("RV")
    axes[0].set_xlabel("LV volume (mL)")
    axes[1].set_xlabel("RV volume (mL)")
    axes[0].set_ylabel("LV pressure (mmHg)")
    axes[1].set_ylabel("RV pressure (mmHg)")
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.86)
    cbar.set_label("achieved RV systolic pressure (mmHg)")
    save(fig, "fig_4_2_pv_loops_spectrum")


def main() -> None:
    pv_loop_figure()
    print(f"Wrote capped primary thesis figures to {THESIS_FIG}")


if __name__ == "__main__":
    main()
