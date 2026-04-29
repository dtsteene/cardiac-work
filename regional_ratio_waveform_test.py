#!/usr/bin/env python3
"""
Test whether pressure-normalized waveform loops preserve the LV/RV work ratio.

The existing fig_regional_ratio plot uses the one-beat cascade data and asks
whether simple pressure choices preserve the LV/RV ratio of model-resolved
tensor work density. This script adds one extra comparison:

    raw pressure proxy:       integral P(t) dE
    waveform-only proxy:      integral [P(t) / peak(P)] dE

The second version keeps the timing and shape of the pressure waveform, but
removes the pressure magnitude. For the adjacent-cavity convention this tests
whether the large LV/RV pressure scale difference is what breaks the ratio.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "analysis" / "cascade"
RAW = OUT / "cascade_raw.npz"
VOL_JSON = Path("/tmp/ukb_wall_volumes.json")
KPA = 1e-3  # J/m^3 -> kPa = mJ/mL


def load_wall_volumes() -> dict[str, float]:
    if not VOL_JSON.exists():
        raise FileNotFoundError(
            f"Missing {VOL_JSON}. Re-run analyze_cascade.py, or recreate the "
            "wall-volume json used by viz_regional_ratio.py."
        )
    with VOL_JSON.open() as f:
        return json.load(f)


def density(d: np.lib.npyio.NpzFile, region: str, key: str, volume: float) -> float:
    return float(d[f"{region}_{key}"].sum() / volume * KPA)


def pressure_peak(d: np.lib.npyio.NpzFile, region: str) -> float:
    return float(np.max(d[f"{region}_P_pa"]))


def ratio_error(ratio: float, reference: float) -> float:
    return abs(ratio - reference)


def build_rows() -> list[dict[str, object]]:
    d = np.load(RAW)
    volumes = load_wall_volumes()
    rows: list[dict[str, object]] = []

    for split in ("tau_lap", "tau_eu"):
        lv = f"LV_{split}"
        rv = f"RV_{split}"
        v_lv = float(volumes[lv])
        v_rv = float(volumes[rv])

        w_lv = density(d, lv, "W0_per_step", v_lv)
        w_rv = density(d, rv, "W0_per_step", v_rv)
        tensor_ratio = w_lv / w_rv

        p_lv_peak = pressure_peak(d, lv)
        p_rv_peak = pressure_peak(d, rv)
        pressure_ratio = p_lv_peak / p_rv_peak

        rows.append(
            {
                "split": split,
                "strain": "tensor",
                "mode": "model-resolved tensor work",
                "lv_density": w_lv,
                "rv_density": w_rv,
                "ratio": tensor_ratio,
                "abs_error_vs_tensor_ratio": 0.0,
                "pressure_ratio_used": "",
                "pressure_normalized": False,
            }
        )

        for level, strain in (("W3", "fiber strain"), ("W4", "longitudinal strain")):
            lv_p_lv = density(d, lv, f"{level}_per_step", v_lv)
            lv_p_rv = density(d, lv, f"{level}_wrong_per_step", v_lv)
            rv_p_rv = density(d, rv, f"{level}_per_step", v_rv)
            rv_p_lv = density(d, rv, f"{level}_wrong_per_step", v_rv)

            comparisons = [
                ("P_LV everywhere", lv_p_lv, rv_p_lv, None, None, 1.0, False),
                ("P_RV everywhere", lv_p_rv, rv_p_rv, None, None, 1.0, False),
                (
                    "adjacent pressure",
                    lv_p_lv,
                    rv_p_rv,
                    None,
                    None,
                    pressure_ratio,
                    False,
                ),
                (
                    "adjacent waveform only",
                    lv_p_lv,
                    rv_p_rv,
                    p_lv_peak,
                    p_rv_peak,
                    pressure_ratio,
                    True,
                ),
            ]

            for (
                mode,
                lv_value,
                rv_value,
                lv_scale,
                rv_scale,
                pressure_ratio_used,
                pressure_norm,
            ) in comparisons:
                if lv_scale is None:
                    ratio = lv_value / rv_value
                    lv_report = lv_value
                    rv_report = rv_value
                else:
                    ratio = (lv_value / lv_scale) / (rv_value / rv_scale)
                    lv_report = lv_value / lv_scale
                    rv_report = rv_value / rv_scale

                rows.append(
                    {
                        "split": split,
                        "strain": strain,
                        "mode": mode,
                        "lv_density": lv_report,
                        "rv_density": rv_report,
                        "ratio": ratio,
                        "abs_error_vs_tensor_ratio": ratio_error(ratio, tensor_ratio),
                        "pressure_ratio_used": pressure_ratio_used,
                        "pressure_normalized": pressure_norm,
                    }
                )

    return rows


def write_csv(rows: list[dict[str, object]]) -> Path:
    path = OUT / "regional_ratio_waveform_test.csv"
    fieldnames = [
        "split",
        "strain",
        "mode",
        "lv_density",
        "rv_density",
        "ratio",
        "abs_error_vs_tensor_ratio",
        "pressure_ratio_used",
        "pressure_normalized",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_summary(rows: list[dict[str, object]]) -> None:
    for split in ("tau_lap", "tau_eu"):
        ref = next(
            r for r in rows if r["split"] == split and r["strain"] == "tensor"
        )
        print("=" * 78)
        print(
            f"{split}: tensor work density LV/RV = {ref['ratio']:.3f} "
            f"(LV={ref['lv_density']:+.3f} kPa, RV={ref['rv_density']:+.3f} kPa)"
        )
        print("=" * 78)
        print(
            f"{'strain':<20} {'mode':<24} {'LV/RV':>8} "
            f"{'abs error':>10} {'P ratio':>9}"
        )
        for r in rows:
            if r["split"] != split or r["strain"] == "tensor":
                continue
            p_ratio = r["pressure_ratio_used"]
            p_text = "" if p_ratio == "" else f"{float(p_ratio):.3f}"
            print(
                f"{r['strain']:<20} {r['mode']:<24} "
                f"{float(r['ratio']):>8.3f} "
                f"{float(r['abs_error_vs_tensor_ratio']):>10.3f} "
                f"{p_text:>9}"
            )
        print()


def make_figure(rows: list[dict[str, object]]) -> Path:
    split = "tau_lap"
    ref = next(r for r in rows if r["split"] == split and r["strain"] == "tensor")
    tensor_ratio = float(ref["ratio"])

    modes = [
        "P_LV everywhere",
        "P_RV everywhere",
        "adjacent pressure",
        "adjacent waveform only",
    ]
    strains = ["fiber strain", "longitudinal strain"]
    colors = {
        "P_LV everywhere": "#4C78A8",
        "P_RV everywhere": "#B279A2",
        "adjacent pressure": "#59A14F",
        "adjacent waveform only": "#F28E2B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, constrained_layout=True)
    for ax, strain in zip(axes, strains):
        values = []
        for mode in modes:
            row = next(
                r
                for r in rows
                if r["split"] == split and r["strain"] == strain and r["mode"] == mode
            )
            values.append(float(row["ratio"]))
        xs = np.arange(len(modes))
        ax.bar(xs, values, color=[colors[m] for m in modes], width=0.62)
        ax.axhline(tensor_ratio, color="black", linewidth=2.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            ["LV P", "RV P", "adjacent P", "waveform only"],
            rotation=22,
            ha="right",
        )
        ax.set_title(strain)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for x, value in zip(xs, values):
            ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    axes[0].set_ylabel("LV/RV ratio")
    axes[1].text(
        0.98,
        0.96,
        f"tensor work ratio = {tensor_ratio:.2f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    y_max = max(
        tensor_ratio,
        *[
            float(r["ratio"])
            for r in rows
            if r["split"] == split and r["strain"] in strains
        ],
    )
    axes[0].set_ylim(0, y_max * 1.22)

    path = OUT / "fig_regional_ratio_waveform.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "fig_regional_ratio_waveform.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    rows = build_rows()
    print_summary(rows)
    csv_path = write_csv(rows)
    fig_path = make_figure(rows)
    print(f"Saved {csv_path}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
