#!/usr/bin/env python3
"""Septum mechanism diagnostics for the h=5 corrected pressure sweep.

This script deliberately reuses completed simulations only. It asks whether the
high-resolution septal ambiguity is mainly a pressure-choice issue, a
strain-direction issue, or a through-wall tissue-mechanics issue.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent / "RV"
OUT = Path(os.environ.get("H5_SWEEP_ANALYSIS", ROOT / "results" / "analysis" / "h5_sweep"))
FIG_OUT = OUT
THESIS_FIG = THESIS / "figures"
STATUS = OUT / "h5_sweep_case_status.csv"
KPA = 1e-3

PROXIES = ["PLV", "PRV", "Trans", "Mean", "NearestSide", "TauWeighted"]
PROXY_LABELS = {
    "PLV": "LV pressure",
    "PRV": "RV pressure",
    "Trans": "transmural",
    "Mean": "mean",
    "NearestSide": "nearest side",
    "TauWeighted": "through-wall weighted",
}
STRAINS = ["ll", "ff"]
STRAIN_LABELS = {"ll": "longitudinal", "ff": "fibre"}
COMPONENTS = ["ff", "ss", "nn", "cross"]
LAYERS = [
    ("LV-side third", 0.0, 1.0 / 3.0),
    ("middle third", 1.0 / 3.0, 2.0 / 3.0),
    ("RV-side third", 2.0 / 3.0, 1.0),
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_status() -> list[dict[str, str]]:
    with STATUS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["status"] == "complete_post"]


def pressure_peaks(run: Path) -> tuple[float, float]:
    pressure = np.load(run / "solver" / "solver_cavity_pressure_mmHg.npy")
    beat_len = pressure.shape[0] // 6
    last = pressure[5 * beat_len :]
    return float(last[:, 0].max()), float(last[:, 1].max())


def density(pc: np.lib.npyio.NpzFile, mask: np.ndarray, values: str | np.ndarray) -> float:
    arr = pc[values] if isinstance(values, str) else values
    volume = float(pc["cell_volumes"][mask].sum())
    if volume <= 0:
        return float("nan")
    return float(-arr[mask].sum() / volume * KPA)


def candidate_arrays(pc: np.lib.npyio.NpzFile, suffix: str) -> dict[str, np.ndarray]:
    # Canonical convention for pressure choices: tau=0 on the LV side and
    # tau=1 on the RV side. The saved Laplace scalar has the opposite orientation.
    tau = 1.0 - pc["lv_rv_scalar"] if "lv_rv_scalar" in pc.files else pc["tau"]
    plv = pc[f"proxy_PLV_{suffix}"]
    prv = pc[f"proxy_PRV_{suffix}"]
    return {
        "PLV": plv,
        "PRV": prv,
        "Trans": plv - prv,
        "Mean": 0.5 * (plv + prv),
        "NearestSide": np.where(tau < 0.5, plv, prv),
        "TauWeighted": (1.0 - tau) * plv + tau * prv,
    }


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(pearsonr(x[ok], y[ok]).statistic)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def leave_one_out_range(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    vals = []
    for idx in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[idx] = False
        r = safe_pearson(x[keep], y[keep])
        if math.isfinite(r):
            vals.append(r)
    if not vals:
        return float("nan"), float("nan")
    return float(min(vals)), float(max(vals))


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []

    for item in read_status():
        run = Path(item["result_dir"])
        pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
        lvsp, rvsp = pressure_peaks(run)
        septum = pc["is_geometric_septum"].astype(bool)
        tau = 1.0 - pc["lv_rv_scalar"] if "lv_rv_scalar" in pc.files else pc["tau"]

        base: dict[str, object] = {
            "case": item["case"],
            "job_id": item["job_id"],
            "result_dir": str(run),
            "LV_ESP_mmHg": lvsp,
            "RV_ESP_mmHg": rvsp,
            "Trans_ESP_mmHg": lvsp - rvsp,
            "septum_n_cells": int(septum.sum()),
            "septum_volume_mL": float(pc["cell_volumes"][septum].sum() * 1e6),
        }
        for key in ["total", *COMPONENTS]:
            base[f"W_{key}_kPa"] = density(pc, septum, f"w_{key}")
        for comp in COMPONENTS:
            base[f"frac_{comp}"] = float(base[f"W_{comp}_kPa"]) / float(base["W_total_kPa"])
        for suffix in STRAINS:
            arrays = candidate_arrays(pc, suffix)
            for proxy, values in arrays.items():
                base[f"{proxy}_{suffix}_kPa"] = density(pc, septum, values)
        case_rows.append(base)

        for label, lo, hi in LAYERS:
            if hi >= 1.0:
                mask = septum & (tau >= lo) & (tau <= hi)
            else:
                mask = septum & (tau >= lo) & (tau < hi)
            row: dict[str, object] = {
                "case": item["case"],
                "job_id": item["job_id"],
                "layer": label,
                "tau_min": lo,
                "tau_max": hi,
                "RV_ESP_mmHg": rvsp,
                "n_cells": int(mask.sum()),
                "volume_mL": float(pc["cell_volumes"][mask].sum() * 1e6),
            }
            for key in ["total", *COMPONENTS]:
                row[f"W_{key}_kPa"] = density(pc, mask, f"w_{key}")
            for comp in COMPONENTS:
                row[f"frac_{comp}"] = float(row[f"W_{comp}_kPa"]) / float(row["W_total_kPa"])
            for suffix in STRAINS:
                arrays = candidate_arrays(pc, suffix)
                for proxy, values in arrays.items():
                    row[f"{proxy}_{suffix}_kPa"] = density(pc, mask, values)
            layer_rows.append(row)

    cases = pd.DataFrame(case_rows).sort_values("RV_ESP_mmHg")
    layers = pd.DataFrame(layer_rows).sort_values(["RV_ESP_mmHg", "layer"])

    corr_rows: list[dict[str, object]] = []
    for target_key, target_label in [
        ("W_total_kPa", "septum total tensor work"),
        ("W_ff_kPa", "septum fibre work"),
    ]:
        y = cases[target_key].to_numpy(float)
        for suffix in STRAINS:
            for proxy in PROXIES:
                x = cases[f"{proxy}_{suffix}_kPa"].to_numpy(float)
                r = safe_pearson(x, y)
                lo, hi = leave_one_out_range(x, y)
                corr_rows.append(
                    {
                        "n": len(cases),
                        "target": target_label,
                        "target_key": target_key,
                        "strain": suffix,
                        "proxy": proxy,
                        "r": r,
                        "r2": r * r if math.isfinite(r) else float("nan"),
                        "spearman_rho": safe_spearman(x, y),
                        "leave_one_out_r_min": lo,
                        "leave_one_out_r_max": hi,
                    }
                )

    ratio_rows: list[dict[str, object]] = []
    h5_values = pd.read_csv(OUT / "h5_sweep_case_values.csv")
    h5_values = h5_values[h5_values["case"].isin(cases["case"])].copy()
    h5_values = h5_values.sort_values("RV_ESP_mmHg")
    fw_tensor_mean = 0.5 * (
        h5_values["LV_W_total_kPa"].to_numpy(float)
        + h5_values["RV_W_total_kPa"].to_numpy(float)
    )
    target = h5_values["Septum_W_total_kPa"].to_numpy(float) / fw_tensor_mean
    for suffix in STRAINS:
        fw_adjacent_mean = 0.5 * (
            h5_values[f"LV_PLV_{suffix}_kPa"].to_numpy(float)
            + h5_values[f"RV_PRV_{suffix}_kPa"].to_numpy(float)
        )
        for proxy in PROXIES:
            prox = h5_values[f"Septum_{proxy}_{suffix}_kPa"].to_numpy(float) / fw_adjacent_mean
            errs = np.abs(np.log(np.abs(prox / target)))
            raw = np.abs(prox - target)
            ratio_rows.append(
                {
                    "n": len(h5_values),
                    "strain": suffix,
                    "proxy": proxy,
                    "mean_abs_raw_error": float(np.mean(raw[np.isfinite(raw)])),
                    "max_abs_raw_error": float(np.max(raw[np.isfinite(raw)])),
                    "mean_abs_log_error": float(np.mean(errs[np.isfinite(errs)])),
                    "max_abs_log_error": float(np.max(errs[np.isfinite(errs)])),
                }
            )

    return cases, layers, pd.DataFrame(corr_rows), pd.DataFrame(ratio_rows)


def save_figure(fig: plt.Figure, analysis_name: str, thesis_name: str) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    THESIS_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        analysis_path = FIG_OUT / f"{analysis_name}.{ext}"
        thesis_path = THESIS_FIG / f"{thesis_name}.{ext}"
        fig.savefig(analysis_path, dpi=220 if ext == "png" else None, bbox_inches="tight")
        fig.savefig(thesis_path, dpi=220 if ext == "png" else None, bbox_inches="tight")


def plot_strain_direction(corrs: pd.DataFrame, ratios: pd.DataFrame) -> None:
    subset_corr = corrs[corrs["target_key"] == "W_total_kPa"].copy()
    subset_corr["label"] = subset_corr["proxy"].map(PROXY_LABELS)
    ratios = ratios.copy()
    ratios["label"] = ratios["proxy"].map(PROXY_LABELS)

    selected = ["PLV", "Trans", "Mean", "NearestSide", "TauWeighted"]
    labels = [PROXY_LABELS[p] for p in selected]
    x = np.arange(len(selected))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 8.4), constrained_layout=True, sharex=True)
    colors = {"ll": "#3B6EA8", "ff": "#C95D3F"}

    for offset, suffix in [(-width / 2, "ll"), (width / 2, "ff")]:
        values = [
            float(subset_corr[(subset_corr["proxy"] == p) & (subset_corr["strain"] == suffix)]["r"].iloc[0])
            for p in selected
        ]
        axes[0].bar(x + offset, values, width=width, color=colors[suffix], label=STRAIN_LABELS[suffix])

    axes[0].axhline(0, color="0.25", linewidth=0.8)
    axes[0].set_ylim(-0.15, 1.05)
    axes[0].set_ylabel("Pearson r with septal tensor work")
    axes[0].set_title("Case ranking")
    axes[0].legend(frameon=False)

    for offset, suffix in [(-width / 2, "ll"), (width / 2, "ff")]:
        values = [
            float(ratios[(ratios["proxy"] == p) & (ratios["strain"] == suffix)]["mean_abs_log_error"].iloc[0])
            for p in selected
        ]
        axes[1].bar(x + offset, values, width=width, color=colors[suffix], label=STRAIN_LABELS[suffix])

    axes[1].set_ylabel("Mean absolute log ratio error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_title("Magnitude preservation")

    fig.suptitle("Septum diagnostic: fibre-aligned strain separates strain-direction error from pressure-choice error")
    save_figure(fig, "fig_h5_septum_strain_direction_diagnostic", "fig_5_3b_septum_strain_direction_diagnostic")
    plt.close(fig)


def plot_layer_mechanics(layers: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), constrained_layout=True, sharex=True)
    layer_colors = {
        "LV-side third": "#3B6EA8",
        "middle third": "#6A737D",
        "RV-side third": "#C95D3F",
    }
    layer_order = [label for label, _, _ in LAYERS]

    for layer in layer_order:
        rr = layers[layers["layer"] == layer].sort_values("RV_ESP_mmHg")
        color = layer_colors[layer]
        axes[0].plot(rr["RV_ESP_mmHg"], rr["W_total_kPa"], marker="o", linewidth=1.8, color=color, label=layer)
        axes[1].plot(rr["RV_ESP_mmHg"], rr["frac_ff"], marker="o", linewidth=1.8, color=color)
        axes[2].plot(rr["RV_ESP_mmHg"], rr["frac_nn"], marker="o", linewidth=1.8, color=color)

    axes[0].set_ylabel("Tensor work density (kPa)")
    axes[1].set_ylabel("Fibre fraction of net work")
    axes[2].set_ylabel("Sheet-normal fraction of net work")
    for ax in axes:
        ax.set_xlabel("RV systolic pressure (mmHg)")
        ax.grid(True, alpha=0.18)
    axes[0].set_title("Through-wall work")
    axes[1].set_title("Fibre contribution")
    axes[2].set_title("Normal contribution")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Septum layer diagnostic across the pressure sweep")
    save_figure(fig, "fig_h5_septum_layer_mechanics", "fig_5_3c_septum_layer_mechanics")
    plt.close(fig)


def write_summary(cases: pd.DataFrame, layers: pd.DataFrame, corrs: pd.DataFrame, ratios: pd.DataFrame) -> None:
    total_corr = corrs[corrs["target_key"] == "W_total_kPa"]
    def val(frame: pd.DataFrame, proxy: str, strain: str, col: str) -> float:
        return float(frame[(frame["proxy"] == proxy) & (frame["strain"] == strain)][col].iloc[0])

    lines = [
        "# H5 Septum Mechanism Diagnostic",
        "",
        f"Completed cases included: {len(cases)}.",
        "",
        "## Strain Direction",
        "",
        (
            "For septal total tensor work, longitudinal pressure-strain and fibre-aligned "
            "pressure-strain answer different questions. With longitudinal strain, "
            f"LV pressure gives r={val(total_corr, 'PLV', 'll', 'r'):+.3f}, "
            f"mean pressure r={val(total_corr, 'Mean', 'll', 'r'):+.3f}, and "
            f"transmural pressure r={val(total_corr, 'Trans', 'll', 'r'):+.3f}."
        ),
        (
            "With model-side fibre strain, the correlations are recomputed for the same "
            f"pressure choices: LV r={val(total_corr, 'PLV', 'ff', 'r'):+.3f}, "
            f"mean r={val(total_corr, 'Mean', 'ff', 'r'):+.3f}, nearest-side "
            f"r={val(total_corr, 'NearestSide', 'ff', 'r'):+.3f}, and through-wall weighted "
            f"r={val(total_corr, 'TauWeighted', 'ff', 'r'):+.3f}."
        ),
        "",
        "## Magnitude Preservation",
        "",
        (
            "The fibre-aligned diagnostic also improves septum/free-wall magnitude preservation "
            "for two-sided pressure choices. Mean absolute log ratio error for mean pressure "
            f"falls from {val(ratios, 'Mean', 'll', 'mean_abs_log_error'):.3f} with longitudinal "
            f"strain to {val(ratios, 'Mean', 'ff', 'mean_abs_log_error'):.3f} with fibre strain. "
            "Nearest-side and through-wall weighted pressure show the same pattern."
        ),
        "",
        "## Through-Wall Pattern",
        "",
        (
            "The layer table separates the septum into LV-side, middle, and RV-side thirds by "
            "the LV-to-RV scalar. This is diagnostic rather than a fully resolved transmural "
            "stress measurement. It shows that septal work and the non-fibre contribution are "
            "not distributed uniformly through the wall."
        ),
        "",
        "## Thesis Interpretation",
        "",
        (
            "A septal pressure choice should not be selected from sweep correlation alone. "
            "Magnitude preservation still favours pressures inside or near the physical "
            "LV-RV loading interval, while transmural pressure remains a poor density-ratio "
            "proxy. The fibre diagnostic suggests that some of the septal difficulty is a "
            "strain-direction limitation of longitudinal strain, not only a pressure-choice "
            "limitation."
        ),
        "",
        "## Outputs",
        "",
        "- `h5_septum_mechanism_cases.csv`",
        "- `h5_septum_mechanism_layers.csv`",
        "- `h5_septum_mechanism_correlations.csv`",
        "- `h5_septum_mechanism_ratio_errors.csv`",
        "- `fig_h5_septum_strain_direction_diagnostic.png/pdf`",
        "- `fig_h5_septum_layer_mechanics.png/pdf`",
    ]
    (OUT / "h5_septum_mechanism_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, layers, corrs, ratios = build_tables()
    cases.to_csv(OUT / "h5_septum_mechanism_cases.csv", index=False)
    layers.to_csv(OUT / "h5_septum_mechanism_layers.csv", index=False)
    corrs.to_csv(OUT / "h5_septum_mechanism_correlations.csv", index=False)
    ratios.to_csv(OUT / "h5_septum_mechanism_ratio_errors.csv", index=False)
    plot_strain_direction(corrs, ratios)
    plot_layer_mechanics(layers)
    write_summary(cases, layers, corrs, ratios)
    print(f"Wrote septum mechanism diagnostics for {len(cases)} completed cases to {OUT}")


if __name__ == "__main__":
    main()
