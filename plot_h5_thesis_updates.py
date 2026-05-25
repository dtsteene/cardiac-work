#!/usr/bin/env python3
"""Create thesis figures from the h=5 corrected-sweep analysis outputs."""

from __future__ import annotations

import re
import shutil
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path("/home/dtsteene/D1")
# H5_SWEEP_ANALYSIS = ROOT / "cardiac-work" / "results" / "analysis" / "h5_sweep"  # pre-cap thesis (deprecated 2026-05-14)
# H5_SWEEP_ANALYSIS = ROOT / "cardiac-work" / "results" / "analysis" / "capped_canonical_h5_sweep"  # duplicate of canonical, archived 2026-05-14
ANALYSIS = Path(os.environ.get("H5_SWEEP_ANALYSIS", ROOT / "cardiac-work" / "results" / "analysis" / "capped_shared_l5_sweep_20260510_141015"))
THESIS_FIG = ROOT / "RV" / "figures"
SKIP_RESOLUTION = os.environ.get("H5_SWEEP_SKIP_RESOLUTION", "0") == "1"


def case_number(label: str) -> int:
    match = re.search(r"(\d+)", label)
    return int(match.group(1)) if match else 0


def save_and_copy(fig: plt.Figure, stem: str, thesis_stem: str | None = None) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        out = ANALYSIS / f"{stem}.{ext}"
        fig.savefig(out, dpi=260 if ext == "png" else None, bbox_inches="tight")
        if thesis_stem is not None:
            shutil.copy2(out, THESIS_FIG / f"{thesis_stem}.{ext}")
    plt.close(fig)


def add_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = "#555555") -> None:
    if len(x) < 2:
        return
    order = np.argsort(x)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(x[order], slope * x[order] + intercept, color=color, linewidth=1.5)


def last_beat(metrics: dict[str, object]) -> slice:
    time = np.asarray(metrics["time"], dtype=float)
    n = len(time) // 6
    return slice(5 * n, 6 * n)


def peak_longitudinal_shortening(result_dir: str | Path, region: str) -> float:
    metrics = np.load(Path(result_dir) / "metrics" / "metrics_downsample_1.npy", allow_pickle=True).item()
    beat = last_beat(metrics)
    eps = np.asarray(metrics[f"mean_E_ll_{region}"], dtype=float)[beat]
    eps = eps - eps[0]
    return float(abs(np.min(eps)))


def plot_freewall(rows: pd.DataFrame) -> None:
    # Order cases by achieved RV systolic pressure (not by sPAP target).
    rows = rows.sort_values("RV_ESP_mmHg")
    rvsp = rows["RV_ESP_mmHg"].to_numpy(float)
    xs = np.arange(len(rows))
    p_ratio = rows["LV_ESP_mmHg"].to_numpy(float) / rvsp
    tensor = rows["FW_tensor_LV_RV_ratio"].to_numpy(float)
    adjacent = rows["FW_adjacent_ll_LV_RV_ratio"].to_numpy(float)
    lv_everywhere = rows["LV_PLV_ll_kPa"].to_numpy(float) / rows["RV_PLV_ll_kPa"].to_numpy(float)
    rv_everywhere = rows["LV_PRV_ll_kPa"].to_numpy(float) / rows["RV_PRV_ll_kPa"].to_numpy(float)
    waveform = adjacent / p_ratio

    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    ax.plot(xs, tensor, color="black", linewidth=2.6, marker="o", label="tensor work")
    ax.plot(xs, adjacent, color="#2F855A", linewidth=2.0, marker="o", label="adjacent pressure")
    ax.plot(xs, lv_everywhere, color="#3B6EA8", linewidth=1.8, marker="o", label="LV pressure everywhere")
    ax.plot(xs, rv_everywhere, color="#9B5CA8", linewidth=1.8, marker="o", label="RV pressure everywhere")
    ax.plot(xs, waveform, color="#D67A28", linewidth=1.8, marker="o", label="waveform only")
    ax.set_xticks(xs)
    rvsp_labels = [f"{int(round(v))}" for v in rvsp]
    ax.set_xticklabels(rvsp_labels, rotation=0, ha="center")
    ax.set_xlabel("Peak RV systolic pressure (mmHg)")
    ax.set_ylabel("free-wall LV/RV work-density ratio")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(framealpha=0.95, ncols=2)
    save_and_copy(fig, "fig_h5_freewall_ratio_spectrum", "fig_5_2_freewall_ratio_spectrum")


def plot_freewall_single(rows: pd.DataFrame) -> None:
    rows = rows.sort_values("case_num")
    row = rows.iloc[0]
    p_ratio = float(row["LV_ESP_mmHg"]) / float(row["RV_ESP_mmHg"])
    values = [
        ("tensor work", float(row["FW_tensor_LV_RV_ratio"]), "#222222"),
        ("adjacent pressure", float(row["FW_adjacent_ll_LV_RV_ratio"]), "#2F855A"),
        ("LV pressure everywhere", float(row["LV_PLV_ll_kPa"]) / float(row["RV_PLV_ll_kPa"]), "#3B6EA8"),
        ("RV pressure everywhere", float(row["LV_PRV_ll_kPa"]) / float(row["RV_PRV_ll_kPa"]), "#9B5CA8"),
        ("waveform only", float(row["FW_adjacent_ll_LV_RV_ratio"]) / p_ratio, "#D67A28"),
    ]

    labels = [v[0] for v in values]
    heights = [v[1] for v in values]
    colors = [v[2] for v in values]
    xs = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(8.7, 4.2), constrained_layout=True)
    ax.bar(xs, heights, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(heights[0], color="black", linewidth=1.2, linestyle="--", alpha=0.75)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("free-wall LV/RV work-density ratio")
    ax.set_title(f"Single baseline case ({row['case']})")
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_and_copy(fig, "fig_h5_freewall_single_case_ratio", "fig_5_1_freewall_single_case_ratio")


def plot_freewall_scatter(rows: pd.DataFrame) -> None:
    rows = rows.sort_values("case_num")
    rvsp = rows["RV_ESP_mmHg"].to_numpy(float)
    tensor = rows["FW_tensor_LV_RV_ratio"].to_numpy(float)
    p_ratio = rows["LV_ESP_mmHg"].to_numpy(float) / rows["RV_ESP_mmHg"].to_numpy(float)
    candidates = [
        ("LV pressure everywhere", rows["LV_PLV_ll_kPa"].to_numpy(float) / rows["RV_PLV_ll_kPa"].to_numpy(float)),
        ("RV pressure everywhere", rows["LV_PRV_ll_kPa"].to_numpy(float) / rows["RV_PRV_ll_kPa"].to_numpy(float)),
        ("adjacent pressure", rows["FW_adjacent_ll_LV_RV_ratio"].to_numpy(float)),
        ("waveform only", rows["FW_adjacent_ll_LV_RV_ratio"].to_numpy(float) / p_ratio),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 7.2), constrained_layout=True, sharex=True, sharey=True)
    axes_flat = axes.ravel()
    vmin = min([tensor.min(), *(vals.min() for _, vals in candidates)])
    vmax = max([tensor.max(), *(vals.max() for _, vals in candidates)])
    pad = 0.08 * (vmax - vmin)
    lims = (vmin - pad, vmax + pad)

    for ax, (label, proxy) in zip(axes_flat, candidates):
        sc = ax.scatter(proxy, tensor, c=rvsp, cmap="coolwarm", s=58, edgecolor="black", linewidth=0.4)
        add_fit(ax, proxy, tensor)
        ax.plot(lims, lims, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
        r = pearsonr(proxy, tensor)[0]
        mae = float(np.mean(np.abs(proxy - tensor)))
        ax.set_title(f"{label}\nr = {r:+.3f}, MAE = {mae:.2f}")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    fig.supxlabel("proxy LV/RV ratio")
    fig.supylabel("tensor LV/RV ratio")
    cbar = fig.colorbar(sc, ax=axes_flat, shrink=0.88)
    cbar.set_label("RV systolic pressure (mmHg)")
    save_and_copy(fig, "fig_h5_freewall_ratio_scatter", "fig_5_2c_freewall_ratio_scatter")


def plot_components(rows: pd.DataFrame) -> None:
    rows = rows.sort_values("RV_ESP_mmHg")
    components = [
        ("ff", "fibre", "#2F6F9F"),
        ("ss", "sheet", "#8C6BB1"),
        ("nn", "sheet-normal", "#B65A36"),
        ("cross", "cross", "#5A8F5B"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1), constrained_layout=True, sharex=True)
    for ax, region in zip(axes, ["LV", "RV", "Septum"]):
        total = rows[f"{region}_W_total_kPa"].to_numpy(float)
        rvsp = rows["RV_ESP_mmHg"].to_numpy(float)
        for comp, label, color in components:
            frac = rows[f"{region}_W_{comp}_kPa"].to_numpy(float) / total
            ax.plot(rvsp, 100.0 * frac, marker="o", linewidth=1.8, color=color, label=label)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_title(region if region != "Septum" else "septum")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("RV systolic pressure (mmHg)")
    axes[0].set_ylabel("signed component / net work (%)")
    axes[0].legend(framealpha=0.95, fontsize=9)
    save_and_copy(fig, "fig_h5_work_components_vs_rvsp", "fig_5_0b_work_components_vs_rvsp")


def plot_freewall_bridge(rows: pd.DataFrame) -> None:
    rows = rows.sort_values("case_num").copy()
    rv_bridge = pd.read_csv(ANALYSIS / "h5_rv_bridge_case_values.csv").sort_values("case")
    rv_corr = pd.read_csv(ANALYSIS / "h5_rv_bridge_correlations.csv")

    lv_tensor = rows["LV_W_total_kPa"].to_numpy(float)
    lv_proxy = rows["LV_PLV_ll_kPa"].to_numpy(float)
    lv_pressure = rows["LV_ESP_mmHg"].to_numpy(float)
    lv_peak = 100.0 * np.array(
        [peak_longitudinal_shortening(path, "LV") for path in rows["result_dir"]],
        dtype=float,
    )

    rv_tensor = rv_bridge["rv_tensor_work_density_kPa"].to_numpy(float)
    rv_proxy = rv_bridge["rv_ps_ll_density_kPa"].to_numpy(float)
    rv_peak = 100.0 * rv_bridge["peak_abs_E_ll"].to_numpy(float)
    rv_pressure = rv_bridge["rvsp_mmHg"].to_numpy(float)
    rv_ef = rv_bridge["rv_ef_fem_percent"].to_numpy(float)
    rv_lv_pressure_proxy = rv_bridge["rv_wrong_plv_ps_ll_density_kPa"].to_numpy(float)

    rvsp = rows["RV_ESP_mmHg"].to_numpy(float)
    r_lv_proxy = pearsonr(lv_proxy, lv_tensor)[0]
    r_lv_peak = pearsonr(lv_peak, lv_tensor)[0]
    r_lv_pressure = pearsonr(lv_pressure, lv_tensor)[0]
    r_rv_proxy = float(rv_corr.loc[rv_corr["metric_key"] == "rv_ps_ll_density_kPa", "r"].iloc[0])
    r_rv_peak = float(rv_corr.loc[rv_corr["metric_key"] == "peak_abs_E_ll", "r"].iloc[0])
    r_rv_ef = float(rv_corr.loc[rv_corr["metric_key"] == "rv_ef_fem_percent", "r"].iloc[0])
    r_rv_pressure = float(rv_corr.loc[rv_corr["metric_key"] == "rvsp_mmHg", "r"].iloc[0])
    r_rv_lv_pressure = pearsonr(rv_lv_pressure_proxy, rv_tensor)[0]

    def padded_limits(values: np.ndarray) -> tuple[float, float]:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        span = hi - lo
        pad = 0.08 * span if span > 0 else 0.1 * abs(hi) if hi else 0.1
        return lo - pad, hi + pad

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35), constrained_layout=True)
    for ax, x, y, title, annotation in [
        (
            axes[0],
            lv_proxy,
            lv_tensor,
            "LV",
            f"r: PS {r_lv_proxy:+.3f} | strain {r_lv_peak:+.3f} | pressure {r_lv_pressure:+.3f}",
        ),
        (
            axes[1],
            rv_proxy,
            rv_tensor,
            "RV",
            (
                f"r: PS {r_rv_proxy:+.3f} | strain {r_rv_peak:+.3f} | EF {r_rv_ef:+.3f}\n"
                f"pressure {r_rv_pressure:+.3f} | LVp x RV strain {r_rv_lv_pressure:+.3f}"
            ),
        ),
    ]:
        sc = ax.scatter(x, y, c=rvsp, cmap="coolwarm", s=62, edgecolor="black", linewidth=0.4)
        add_fit(ax, x, y)
        ax.set_title(title)
        ax.set_xlim(padded_limits(x))
        ax.set_ylim(padded_limits(y))
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            va="top",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
    fig.supxlabel(r"$p\,\varepsilon_{ll}$ proxy (kPa)")
    fig.supylabel("FE stress-strain work density (kPa)")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.88)
    cbar.set_label("RV systolic pressure (mmHg)")
    save_and_copy(fig, "fig_h5_freewall_bridge", "fig_5_2b_rv_lakatos_bridge")


def plot_septum_correlation_scatter(rows: pd.DataFrame) -> None:
    rows = rows.sort_values("case_num")
    rvsp = rows["RV_ESP_mmHg"].to_numpy(float)
    tensor = rows["Septum_W_total_kPa"].to_numpy(float)
    candidates = [
        ("LV pressure", rows["Septum_PLV_ll_kPa"].to_numpy(float)),
        ("RV pressure", rows["Septum_PRV_ll_kPa"].to_numpy(float)),
        ("transmural pressure", rows["Septum_Trans_ll_kPa"].to_numpy(float)),
        ("mean pressure", rows["Septum_Mean_ll_kPa"].to_numpy(float)),
        ("nearest-side pressure", rows["Septum_NearestSide_ll_kPa"].to_numpy(float)),
        ("through-wall weighted", rows["Septum_TauWeighted_ll_kPa"].to_numpy(float)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.1), constrained_layout=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, (label, proxy) in zip(axes_flat, candidates):
        sc = ax.scatter(proxy, tensor, c=rvsp, cmap="coolwarm", s=56, edgecolor="black", linewidth=0.4)
        add_fit(ax, proxy, tensor)
        r = pearsonr(proxy, tensor)[0]
        ax.set_title(f"{label}\nr = {r:+.3f}")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.supxlabel("septal proxy work density (kPa)")
    fig.supylabel("septal tensor work density (kPa)")
    cbar = fig.colorbar(sc, ax=axes_flat, shrink=0.88)
    cbar.set_label("RV systolic pressure (mmHg)")
    save_and_copy(fig, "fig_h5_septum_proxy_correlation_scatter", "fig_5_3a_septum_proxy_correlation_scatter")


def plot_lambda(rows: pd.DataFrame) -> None:
    rows = rows.sort_values("case_num")
    target = rows["Septum_W_total_kPa"].to_numpy(float)
    fibre = rows["Septum_W_ff_kPa"].to_numpy(float)
    plv = rows["Septum_PLV_ll_kPa"].to_numpy(float)
    prv = rows["Septum_PRV_ll_kPa"].to_numpy(float)
    tensor_ratio = rows["Septum_to_FWmean_tensor_ratio"].to_numpy(float)
    fw_adj = 0.5 * (rows["LV_PLV_ll_kPa"].to_numpy(float) + rows["RV_PRV_ll_kPa"].to_numpy(float))

    scan = []
    for lam in np.linspace(-1.0, 5.0, 1201):
        vals = lam * plv + (1.0 - lam) * prv
        ratio = vals / fw_adj
        scan.append(
            {
                "lambda": lam,
                "r_total": pearsonr(vals, target)[0],
                "r_fibre": pearsonr(vals, fibre)[0],
                "mean_abs_log_ratio_error": float(np.mean(np.abs(np.log(np.abs(ratio / tensor_ratio))))),
            }
        )
    scan_df = pd.DataFrame(scan)
    scan_df.to_csv(ANALYSIS / "h5_septum_lambda_scan.csv", index=False)
    best_rows = [
        {
            "objective": "best correlation with total tensor work",
            **scan_df.loc[scan_df["r_total"].idxmax()].to_dict(),
        },
        {
            "objective": "best correlation with fibre work",
            **scan_df.loc[scan_df["r_fibre"].idxmax()].to_dict(),
        },
        {
            "objective": "best septum/free-wall ratio preservation",
            **scan_df.loc[scan_df["mean_abs_log_ratio_error"].idxmin()].to_dict(),
        },
    ]
    pd.DataFrame(best_rows).to_csv(ANALYSIS / "h5_septum_best_lambda.csv", index=False)

    best_corr = scan_df.loc[scan_df["r_total"].idxmax()]
    best_ratio = scan_df.loc[scan_df["mean_abs_log_ratio_error"].idxmin()]

    fig, ax1 = plt.subplots(figsize=(8.6, 4.7), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.plot(scan_df["lambda"], scan_df["r_total"], color="#2F6F9F", linewidth=2.2, label="correlation")
    ax2.plot(
        scan_df["lambda"],
        scan_df["mean_abs_log_ratio_error"],
        color="#B65A36",
        linewidth=2.2,
        label="ratio error",
    )
    for x, color in [(0.0, "#888888"), (0.5, "#888888"), (1.0, "#888888")]:
        ax1.axvline(x, color=color, linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.axvline(float(best_corr["lambda"]), color="#2F6F9F", linewidth=1.2, linestyle=":")
    ax2.axvline(float(best_ratio["lambda"]), color="#B65A36", linewidth=1.2, linestyle=":")
    ax1.set_xlabel(r"$\lambda$ in $p_\lambda=\lambda p_\mathrm{LV}+(1-\lambda)p_\mathrm{RV}$")
    ax1.set_ylabel("correlation with septal tensor work", color="#2F6F9F")
    ax2.set_ylabel("mean absolute log-ratio error", color="#B65A36")
    ax1.tick_params(axis="y", labelcolor="#2F6F9F")
    ax2.tick_params(axis="y", labelcolor="#B65A36")
    ax1.grid(alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.text(0.02, 0.05, "RV", transform=ax1.transAxes, color="#555555")
    ax1.text(0.25, 0.05, "mean", transform=ax1.transAxes, color="#555555")
    ax1.text(0.49, 0.05, "LV", transform=ax1.transAxes, color="#555555")
    save_and_copy(fig, "fig_h5_septum_lambda_scan", "fig_5_4_septum_lambda_scan")


def plot_resolution_shift() -> None:
    diffs = pd.read_csv(ANALYSIS / "h10_to_h5_key_qoi_differences.csv")
    labels = {
        "LV_ESP_mmHg": "LV ESP",
        "RV_ESP_mmHg": "RV ESP",
        "FW_tensor_LV_RV_ratio": "free-wall tensor ratio",
        "FW_adjacent_ll_LV_RV_ratio": "free-wall proxy ratio",
    }
    rows = []
    for qoi, label in labels.items():
        values = diffs.loc[diffs["qoi"] == qoi, "percent_difference_vs_h5"].to_numpy(float)
        rows.append({"quantity": label, "mean_abs_percent": values.mean(), "max_abs_percent": values.max()})
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "h10_to_h5_nonseptal_shift_summary.csv", index=False)

    xs = np.arange(len(out))
    fig, ax = plt.subplots(figsize=(8.3, 4.4), constrained_layout=True)
    width = 0.36
    ax.bar(xs - width / 2, out["mean_abs_percent"], width=width, color="#5A7FA8", label="mean")
    ax.bar(xs + width / 2, out["max_abs_percent"], width=width, color="#C46A4A", label="max")
    ax.set_xticks(xs)
    ax.set_xticklabels(out["quantity"], rotation=20, ha="right")
    ax.set_ylabel("|h=10 - h=5| / h=5 (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(framealpha=0.95)
    save_and_copy(fig, "fig_h10_to_h5_nonseptal_shift", "fig_5_5_h10_to_h5_nonseptal_shift")


def main() -> None:
    rows = pd.read_csv(ANALYSIS / "h5_sweep_case_values.csv")
    rows["case_num"] = rows["case"].map(case_number)
    plot_components(rows)
    plot_freewall_single(rows)
    plot_freewall(rows)
    plot_freewall_scatter(rows)
    plot_freewall_bridge(rows)
    plot_septum_correlation_scatter(rows)
    plot_lambda(rows)
    if not SKIP_RESOLUTION:
        plot_resolution_shift()
    print(f"Wrote h=5 thesis figures to {ANALYSIS} and copied thesis figures to {THESIS_FIG}")


if __name__ == "__main__":
    main()
