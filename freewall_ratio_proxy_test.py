#!/usr/bin/env python3
"""Free-wall LV/RV ratio test for pressure-strain proxies.

The tau-split LV/RV ratio includes septal tissue, which makes the regional
comparison mechanically muddy.  This script repeats the magnitude-ratio test
using only the anatomical free-wall regions:

    LV free wall: region_tags == 1 / metrics region "LV"
    RV free wall: region_tags == 2 / metrics region "RV"

It asks whether each proxy preserves the LV/RV ratio of model-resolved tensor
work density.  The main comparison is intentionally simple:

    W_LV / W_RV  from tensor work
    versus
    proxy_LV / proxy_RV

The pressure-normalized "waveform only" version is included as a diagnostic,
not as a work proxy.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/dtsteene/D1/cardiac-work")
OUT = ROOT / "results" / "analysis" / "freewall_ratio"
CASCADE_SIM = ROOT / "results" / "sims" / "2026-04-13" / "UKB_1beats_run_1023580"
KPA = 1e-3
MMHG_TO_PA = 133.322


CORRECTED_16 = [
    ("sPAP22", 1047450),
    ("sPAP25", 1048194),
    ("sPAP30", 1047451),
    ("sPAP35", 1048195),
    ("sPAP45", 1047452),
    ("sPAP50", 1048196),
    ("sPAP55", 1047453),
    ("sPAP60", 1048197),
    ("sPAP65", 1047454),
    ("sPAP70", 1048198),
    ("sPAP75", 1047455),
    ("sPAP80", 1048199),
    ("sPAP85", 1047456),
    ("sPAP87", 1048200),
    ("sPAP92", 1048201),
    ("sPAP95", 1047457),
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def last_beat_slice(time: np.ndarray) -> slice:
    if len(time) < 2:
        return slice(None)
    dt = float(np.median(np.diff(time)))
    duration = float(time[-1] - time[0] + dt)
    n_beats = max(1, int(round(duration / 0.8)))
    if n_beats <= 1:
        return slice(None)
    beat = len(time) // n_beats
    return slice((n_beats - 1) * beat, n_beats * beat)


def edv_m3_from_metrics(metrics: dict[str, object]) -> tuple[float, float]:
    time = np.asarray(metrics["time"], dtype=float)
    sl = last_beat_slice(time)
    lv = float(np.asarray(metrics["V_LV_FEM"], dtype=float)[sl].max() * 1e-6)
    rv = float(np.asarray(metrics["V_RV_FEM"], dtype=float)[sl].max() * 1e-6)
    return lv, rv


def integrate_pressure_strain(pressure_pa: np.ndarray, strain: np.ndarray) -> float:
    return float(np.sum(pressure_pa * np.gradient(np.asarray(strain, dtype=float))) * KPA)


def single_case_rows() -> list[dict[str, object]]:
    metrics = np.load(
        CASCADE_SIM / "metrics" / "metrics_downsample_1.npy", allow_pickle=True
    ).item()
    pressure = np.load(CASCADE_SIM / "solver" / "solver_cavity_pressure_mmHg.npy")
    n = len(metrics["time"])
    pressure = pressure[:n]
    p_lv = pressure[:, 0] * MMHG_TO_PA
    p_rv = pressure[:, 1] * MMHG_TO_PA
    p_ratio = float(p_lv.max() / p_rv.max())

    v_lv_wall = float(metrics["region_volume_LV"])
    v_rv_wall = float(metrics["region_volume_RV"])
    lv_edv, rv_edv = edv_m3_from_metrics(metrics)
    g_lv = lv_edv / v_lv_wall
    g_rv = rv_edv / v_rv_wall
    geometry_ratio = g_lv / g_rv

    w_lv = float(
        (
            np.asarray(metrics["work_active_LV"])
            + np.asarray(metrics["work_passive_LV"])
            + np.asarray(metrics["work_comp_LV"])
        ).sum()
        / v_lv_wall
        * KPA
    )
    w_rv = float(
        (
            np.asarray(metrics["work_active_RV"])
            + np.asarray(metrics["work_passive_RV"])
            + np.asarray(metrics["work_comp_RV"])
        ).sum()
        / v_rv_wall
        * KPA
    )
    tensor_ratio = w_lv / w_rv

    rows: list[dict[str, object]] = [
        {
            "case": "cascade_healthy",
            "strain": "tensor",
            "proxy": "model-resolved tensor work",
            "ratio": tensor_ratio,
            "abs_error": 0.0,
            "LV_density": w_lv,
            "RV_density": w_rv,
            "pressure_peak_ratio": p_ratio,
            "geometry_ratio": geometry_ratio,
        }
    ]

    for strain_label, key in (("fiber strain", "mean_E_ff"), ("longitudinal strain", "mean_E_ll")):
        e_lv = np.asarray(metrics[f"{key}_LV"], dtype=float)
        e_rv = np.asarray(metrics[f"{key}_RV"], dtype=float)
        lv_p_lv = integrate_pressure_strain(p_lv, e_lv)
        lv_p_rv = integrate_pressure_strain(p_rv, e_lv)
        rv_p_lv = integrate_pressure_strain(p_lv, e_rv)
        rv_p_rv = integrate_pressure_strain(p_rv, e_rv)

        comparisons = [
            ("P_LV everywhere", lv_p_lv / rv_p_lv, lv_p_lv, rv_p_lv),
            ("P_RV everywhere", lv_p_rv / rv_p_rv, lv_p_rv, rv_p_rv),
            ("adjacent pressure", lv_p_lv / rv_p_rv, lv_p_lv, rv_p_rv),
            ("waveform only", (lv_p_lv / rv_p_rv) / p_ratio, lv_p_lv / p_lv.max(), rv_p_rv / p_rv.max()),
            ("P x geometry", (lv_p_lv / rv_p_rv) * geometry_ratio, lv_p_lv * g_lv, rv_p_rv * g_rv),
            (
                "geometry x waveform",
                (lv_p_lv / rv_p_rv) / p_ratio * geometry_ratio,
                lv_p_lv / p_lv.max() * g_lv,
                rv_p_rv / p_rv.max() * g_rv,
            ),
        ]
        for proxy, ratio, lv_value, rv_value in comparisons:
            rows.append(
                {
                    "case": "cascade_healthy",
                    "strain": strain_label,
                    "proxy": proxy,
                    "ratio": ratio,
                    "abs_error": abs(ratio - tensor_ratio),
                    "LV_density": lv_value,
                    "RV_density": rv_value,
                    "pressure_peak_ratio": p_ratio,
                    "geometry_ratio": geometry_ratio,
                }
            )
    return rows


def find_corrected_run(job_id: int) -> Path:
    for root in [
        ROOT / "results" / "sims" / "2026-04-23",
        ROOT / "results" / "sims" / "2026-04-24",
    ]:
        run = root / f"UKB_6beats_run_{job_id}"
        if run.exists():
            return run
    raise FileNotFoundError(job_id)


def cell_density(pc: np.lib.npyio.NpzFile, mask: np.ndarray, key: str) -> float:
    volume = float(pc["cell_volumes"][mask].sum())
    return float(-pc[key][mask].sum() / volume * KPA)


def ratio_row(
    case: str,
    strain: str,
    proxy: str,
    true_ratio: float,
    ratio: float,
    p_ratio: float,
    geometry_ratio: float,
) -> dict[str, object]:
    return {
        "case": case,
        "strain": strain,
        "proxy": proxy,
        "tensor_ratio": true_ratio,
        "proxy_ratio": ratio,
        "abs_error": abs(ratio - true_ratio),
        "pressure_peak_ratio": p_ratio,
        "geometry_ratio": geometry_ratio,
    }


def spectrum_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, job_id in CORRECTED_16:
        run = find_corrected_run(job_id)
        pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
        metrics = np.load(
            run / "metrics" / "metrics_downsample_1.npy", allow_pickle=True
        ).item()
        pressure = np.load(run / "solver" / "solver_cavity_pressure_mmHg.npy")
        p_ratio = float(pressure[:, 0].max() / pressure[:, 1].max())

        lv = pc["region_tags"] == 1
        rv = pc["region_tags"] == 2
        true_ratio = cell_density(pc, lv, "w_total") / cell_density(pc, rv, "w_total")
        ff_tensor_ratio = cell_density(pc, lv, "w_ff") / cell_density(pc, rv, "w_ff")

        lv_edv, rv_edv = edv_m3_from_metrics(metrics)
        g_lv = lv_edv / float(pc["cell_volumes"][lv].sum())
        g_rv = rv_edv / float(pc["cell_volumes"][rv].sum())
        geometry_ratio = g_lv / g_rv

        rows.append(
            ratio_row(
                label,
                "tensor",
                "fibre tensor work",
                true_ratio,
                ff_tensor_ratio,
                p_ratio,
                geometry_ratio,
            )
        )

        for strain, suffix in (("fiber strain", "ff"), ("longitudinal strain", "ll")):
            lv_plv = cell_density(pc, lv, f"proxy_PLV_{suffix}")
            lv_prv = cell_density(pc, lv, f"proxy_PRV_{suffix}")
            rv_plv = cell_density(pc, rv, f"proxy_PLV_{suffix}")
            rv_prv = cell_density(pc, rv, f"proxy_PRV_{suffix}")
            adjacent = lv_plv / rv_prv
            comparisons = [
                ("P_LV everywhere", lv_plv / rv_plv),
                ("P_RV everywhere", lv_prv / rv_prv),
                ("adjacent pressure", adjacent),
                ("waveform only", adjacent / p_ratio),
                ("P x geometry", adjacent * geometry_ratio),
                ("geometry x waveform", adjacent / p_ratio * geometry_ratio),
            ]
            for proxy, ratio in comparisons:
                rows.append(
                    ratio_row(label, strain, proxy, true_ratio, ratio, p_ratio, geometry_ratio)
                )
    return rows


def summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for strain in ["fiber strain", "longitudinal strain"]:
        for proxy in [
            "P_LV everywhere",
            "P_RV everywhere",
            "adjacent pressure",
            "waveform only",
            "P x geometry",
            "geometry x waveform",
        ]:
            matches = [
                r for r in rows if r["strain"] == strain and r["proxy"] == proxy
            ]
            errors = np.array([float(r["abs_error"]) for r in matches])
            true = np.array([float(r["tensor_ratio"]) for r in matches])
            pred = np.array([float(r["proxy_ratio"]) for r in matches])
            out.append(
                {
                    "strain": strain,
                    "proxy": proxy,
                    "mean_abs_error": float(errors.mean()),
                    "median_abs_error": float(np.median(errors)),
                    "max_abs_error": float(errors.max()),
                    "r_vs_tensor_ratio": float(np.corrcoef(true, pred)[0, 1]),
                }
            )
    return out


def print_single(rows: list[dict[str, object]]) -> None:
    ref = next(r for r in rows if r["strain"] == "tensor")
    print("=" * 86)
    print(
        f"Single healthy free-wall case: tensor LV/RV={float(ref['ratio']):.3f}, "
        f"P_peak ratio={float(ref['pressure_peak_ratio']):.3f}, "
        f"geometry ratio={float(ref['geometry_ratio']):.3f}"
    )
    print("=" * 86)
    print(f"{'strain':<20} {'proxy':<24} {'LV/RV':>8} {'abs error':>10}")
    for row in rows:
        if row["strain"] == "tensor":
            continue
        print(
            f"{row['strain']:<20} {row['proxy']:<24} "
            f"{float(row['ratio']):>8.3f} {float(row['abs_error']):>10.3f}"
        )
    print()


def print_spectrum_summary(rows: list[dict[str, object]]) -> None:
    print("=" * 86)
    print("Corrected n=16 free-wall LV/RV ratio errors")
    print("=" * 86)
    print(f"{'strain':<20} {'proxy':<24} {'mean abs err':>13} {'r':>8}")
    for row in summary_rows(rows):
        print(
            f"{row['strain']:<20} {row['proxy']:<24} "
            f"{float(row['mean_abs_error']):>13.3f} {float(row['r_vs_tensor_ratio']):>+8.3f}"
        )


def make_single_figure(rows: list[dict[str, object]]) -> Path:
    ref = next(r for r in rows if r["strain"] == "tensor")
    tensor_ratio = float(ref["ratio"])
    strains = ["fiber strain", "longitudinal strain"]
    proxies = ["P_LV everywhere", "P_RV everywhere", "adjacent pressure", "waveform only"]
    colors = ["#4C78A8", "#B279A2", "#59A14F", "#F28E2B"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True, constrained_layout=True)
    for ax, strain in zip(axes, strains):
        values = [
            float(next(r for r in rows if r["strain"] == strain and r["proxy"] == proxy)["ratio"])
            for proxy in proxies
        ]
        xs = np.arange(len(proxies))
        ax.bar(xs, values, color=colors, width=0.62)
        ax.axhline(tensor_ratio, color="black", linewidth=2.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(["LV P", "RV P", "adjacent P", "waveform"], rotation=25, ha="right")
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
    y_max = max(float(r["ratio"]) for r in rows if r["strain"] != "tensor")
    axes[0].set_ylim(0, max(y_max, tensor_ratio) * 1.22)
    path = OUT / "fig_freewall_single_case_ratio.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "fig_freewall_single_case_ratio.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def make_spectrum_figure(rows: list[dict[str, object]]) -> Path:
    cases = [case for case, _ in CORRECTED_16]
    proxies = [
        ("tensor", "model-resolved tensor work", "tensor"),
        ("longitudinal strain", "P_LV everywhere", "LV P everywhere"),
        ("longitudinal strain", "adjacent pressure", "adjacent P"),
        ("longitudinal strain", "waveform only", "waveform only"),
    ]
    colors = ["black", "#4C78A8", "#59A14F", "#F28E2B"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    xs = np.arange(len(cases))
    tensor = [
        float(next(r for r in rows if r["case"] == case)["tensor_ratio"])
        for case in cases
    ]
    ax.plot(xs, tensor, color=colors[0], linewidth=2.7, marker="o", label="tensor work")
    for (strain, proxy, label), color in zip(proxies[1:], colors[1:]):
        vals = [
            float(
                next(
                    r
                    for r in rows
                    if r["case"] == case and r["strain"] == strain and r["proxy"] == proxy
                )["proxy_ratio"]
            )
            for case in cases
        ]
        ax.plot(xs, vals, color=color, linewidth=2.0, marker="o", label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels(cases, rotation=35, ha="right")
    ax.set_ylabel("free-wall LV/RV ratio")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(framealpha=0.95, ncols=2)
    path = OUT / "fig_freewall_ratio_spectrum.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "fig_freewall_ratio_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    single = single_case_rows()
    spectrum = spectrum_rows()
    summary = summary_rows(spectrum)

    print_single(single)
    print_spectrum_summary(spectrum)

    write_csv(OUT / "single_case_freewall_ratios.csv", single)
    write_csv(OUT / "spectrum_freewall_ratios.csv", spectrum)
    write_csv(OUT / "spectrum_freewall_ratio_summary.csv", summary)
    fig_single = make_single_figure(single)
    fig_spectrum = make_spectrum_figure(spectrum)

    print(f"\nSaved {OUT / 'single_case_freewall_ratios.csv'}")
    print(f"Saved {OUT / 'spectrum_freewall_ratios.csv'}")
    print(f"Saved {OUT / 'spectrum_freewall_ratio_summary.csv'}")
    print(f"Saved {fig_single}")
    print(f"Saved {fig_spectrum}")


if __name__ == "__main__":
    main()
