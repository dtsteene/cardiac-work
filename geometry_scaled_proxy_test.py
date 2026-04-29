#!/usr/bin/env python3
"""Test a simple geometry-scaled pressure-strain proxy.

Candidate proxy:

    geometry-scaled proxy = (cavity volume / chamber wall volume) * integral P(t) dE

This is a crude bulk version of Laplace thinking.  It asks whether cavity
pressure becomes a better regional stress scale if it is corrected by a
clinically measurable geometry term.  The test is intentionally simple:

1. Single one-beat cascade: does it preserve the LV/RV tensor-work density
   ratio better than pressure alone?
2. Corrected pressure spectrum: does it improve cross-case correlations with
   model-resolved total tensor work and fibre work?

This is a hypothesis check, not a proposed final clinical method.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


ROOT = Path("/home/dtsteene/D1/cardiac-work")
OUT = ROOT / "results" / "analysis" / "geometry_scaled_proxy"
CASCADE_OUT = ROOT / "results" / "analysis" / "cascade"
CASCADE_RAW = CASCADE_OUT / "cascade_raw.npz"
CASCADE_SIM = ROOT / "results" / "sims" / "2026-04-13" / "UKB_1beats_run_1023580"
WALL_VOL_JSON = Path("/tmp/ukb_wall_volumes.json")
KPA = 1e-3


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


def corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


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
    if beat > 0 and len(time) >= n_beats * beat:
        return slice((n_beats - 1) * beat, n_beats * beat)
    return slice(None)


def edv_m3_from_metrics(metrics: dict[str, object]) -> tuple[float, float]:
    time = np.asarray(metrics["time"], dtype=float)
    sl = last_beat_slice(time)
    lv = np.asarray(metrics["V_LV_FEM"], dtype=float)[sl].max() * 1e-6
    rv = np.asarray(metrics["V_RV_FEM"], dtype=float)[sl].max() * 1e-6
    return float(lv), float(rv)


def cascade_single_case_rows() -> list[dict[str, object]]:
    d = np.load(CASCADE_RAW)
    with WALL_VOL_JSON.open() as f:
        wall_volumes = json.load(f)

    metrics = np.load(
        CASCADE_SIM / "metrics" / "metrics_downsample_1.npy", allow_pickle=True
    ).item()
    lv_edv, rv_edv = edv_m3_from_metrics(metrics)

    rows: list[dict[str, object]] = []
    for split in ("tau_lap", "tau_eu"):
        lv = f"LV_{split}"
        rv = f"RV_{split}"
        v_lv = float(wall_volumes[lv])
        v_rv = float(wall_volumes[rv])
        g_lv = lv_edv / v_lv
        g_rv = rv_edv / v_rv
        geom_ratio = g_lv / g_rv
        p_ratio = float(np.max(d[f"{lv}_P_pa"]) / np.max(d[f"{rv}_P_pa"]))

        def density(region: str, key: str, volume: float) -> float:
            return float(d[f"{region}_{key}"].sum() / volume * KPA)

        tensor_ratio = density(lv, "W0_per_step", v_lv) / density(
            rv, "W0_per_step", v_rv
        )
        rows.append(
            {
                "split": split,
                "strain": "tensor",
                "proxy": "model-resolved tensor work",
                "ratio": tensor_ratio,
                "abs_error": 0.0,
                "pressure_peak_ratio": p_ratio,
                "geometry_ratio": geom_ratio,
            }
        )

        for level, strain in (("W3", "fiber strain"), ("W4", "longitudinal strain")):
            adj = density(lv, f"{level}_per_step", v_lv) / density(
                rv, f"{level}_per_step", v_rv
            )
            waveform_only = adj / p_ratio
            geom_pressure = adj * geom_ratio
            geom_waveform = waveform_only * geom_ratio
            comparisons = [
                ("adjacent pressure", adj),
                ("waveform only", waveform_only),
                ("P x geometry", geom_pressure),
                ("geometry x waveform", geom_waveform),
            ]
            for proxy, ratio in comparisons:
                rows.append(
                    {
                        "split": split,
                        "strain": strain,
                        "proxy": proxy,
                        "ratio": ratio,
                        "abs_error": abs(ratio - tensor_ratio),
                        "pressure_peak_ratio": p_ratio,
                        "geometry_ratio": geom_ratio,
                    }
                )
    return rows


def masks(pc: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    tags = pc["region_tags"]
    return {
        "Whole": np.ones_like(tags, dtype=bool),
        "LV": tags == 1,
        "RV": tags == 2,
        "Septum": pc["is_geometric_septum"].astype(bool),
    }


def density_from_cells(pc: np.lib.npyio.NpzFile, mask: np.ndarray, key: str) -> float:
    volume = float(pc["cell_volumes"][mask].sum())
    return float(-pc[key][mask].sum() / volume * KPA)


def find_corrected_run(job_id: int) -> Path:
    for root in [
        ROOT / "results" / "sims" / "2026-04-23",
        ROOT / "results" / "sims" / "2026-04-24",
    ]:
        run = root / f"UKB_6beats_run_{job_id}"
        if run.exists():
            return run
    raise FileNotFoundError(job_id)


def spectrum_case_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, job_id in CORRECTED_16:
        run = find_corrected_run(job_id)
        pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
        metrics = np.load(
            run / "metrics" / "metrics_downsample_1.npy", allow_pickle=True
        ).item()
        lv_edv, rv_edv = edv_m3_from_metrics(metrics)
        region_masks = masks(pc)
        lv_wall = float(pc["cell_volumes"][region_masks["LV"]].sum())
        rv_wall = float(pc["cell_volumes"][region_masks["RV"]].sum())
        lv_geom = lv_edv / lv_wall
        rv_geom = rv_edv / rv_wall

        for region, mask in region_masks.items():
            plv = density_from_cells(pc, mask, "proxy_PLV_ll")
            prv = density_from_cells(pc, mask, "proxy_PRV_ll")
            plv_geo = plv * lv_geom
            prv_geo = prv * rv_geom
            rows.append(
                {
                    "case": label,
                    "job_id": job_id,
                    "region": region,
                    "LV_EDV_mL": lv_edv * 1e6,
                    "RV_EDV_mL": rv_edv * 1e6,
                    "LV_geom": lv_geom,
                    "RV_geom": rv_geom,
                    "geom_ratio_LV_RV": lv_geom / rv_geom,
                    "W_total": density_from_cells(pc, mask, "w_total"),
                    "W_ff": density_from_cells(pc, mask, "w_ff"),
                    "PLV": plv,
                    "PRV": prv,
                    "Trans": plv - prv,
                    "Mean": 0.5 * (plv + prv),
                    "PLV_geo": plv_geo,
                    "PRV_geo": prv_geo,
                    "Trans_geo": plv_geo - prv_geo,
                    "Mean_geo": 0.5 * (plv_geo + prv_geo),
                }
            )
    return rows


def spectrum_correlation_rows(case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    proxies = [
        ("PLV", "P_LV x eps_ll"),
        ("PRV", "P_RV x eps_ll"),
        ("Trans", "(P_LV-P_RV) x eps_ll"),
        ("Mean", "mean(P_LV,P_RV) x eps_ll"),
        ("PLV_geo", "P_LV x geometry x eps_ll"),
        ("PRV_geo", "P_RV x geometry x eps_ll"),
        ("Trans_geo", "geometry-scaled transmural x eps_ll"),
        ("Mean_geo", "geometry-scaled mean x eps_ll"),
    ]
    targets = [("W_total", "total tensor work"), ("W_ff", "fibre work")]
    regions = ["Whole", "LV", "RV", "Septum"]
    out: list[dict[str, object]] = []
    for region in regions:
        region_rows = [r for r in case_rows if r["region"] == region]
        for target_key, target_label in targets:
            y = np.array([float(r[target_key]) for r in region_rows])
            for proxy_key, proxy_label in proxies:
                x = np.array([float(r[proxy_key]) for r in region_rows])
                rv = corr(x, y)
                out.append(
                    {
                        "dataset": "corrected n=16",
                        "region": region,
                        "target": target_label,
                        "target_key": target_key,
                        "proxy": proxy_label,
                        "proxy_key": proxy_key,
                        "r": rv,
                        "r2": rv * rv,
                    }
                )
    return out


def print_single_summary(rows: list[dict[str, object]]) -> None:
    for split in ("tau_lap", "tau_eu"):
        ref = next(r for r in rows if r["split"] == split and r["strain"] == "tensor")
        print("=" * 84)
        print(
            f"{split}: tensor LV/RV={float(ref['ratio']):.3f}, "
            f"P_peak ratio={float(ref['pressure_peak_ratio']):.3f}, "
            f"geometry ratio={float(ref['geometry_ratio']):.3f}"
        )
        print("=" * 84)
        print(f"{'strain':<20} {'proxy':<24} {'LV/RV':>8} {'abs error':>10}")
        for row in rows:
            if row["split"] != split or row["strain"] == "tensor":
                continue
            print(
                f"{row['strain']:<20} {row['proxy']:<24} "
                f"{float(row['ratio']):>8.3f} {float(row['abs_error']):>10.3f}"
            )
        print()


def print_spectrum_summary(rows: list[dict[str, object]]) -> None:
    print("=" * 84)
    print("Corrected n=16 spectrum: selected correlations with total tensor work")
    print("=" * 84)
    for region in ("LV", "RV", "Septum"):
        print(f"\n{region}")
        print(f"{'proxy':<38} {'r':>8} {'R2':>8}")
        selected = [
            r
            for r in rows
            if r["region"] == region and r["target_key"] == "W_total"
        ]
        for key in ("PLV", "PRV", "Trans", "Mean", "PLV_geo", "PRV_geo", "Trans_geo", "Mean_geo"):
            row = next(r for r in selected if r["proxy_key"] == key)
            print(f"{row['proxy']:<38} {float(row['r']):>+8.3f} {float(row['r2']):>8.3f}")


def make_single_figure(rows: list[dict[str, object]]) -> Path:
    split = "tau_lap"
    ref = next(r for r in rows if r["split"] == split and r["strain"] == "tensor")
    tensor_ratio = float(ref["ratio"])
    strains = ["fiber strain", "longitudinal strain"]
    proxies = ["adjacent pressure", "waveform only", "P x geometry", "geometry x waveform"]
    colors = ["#59A14F", "#4C78A8", "#E15759", "#F28E2B"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, constrained_layout=True)
    for ax, strain in zip(axes, strains):
        values = [
            float(
                next(
                    r
                    for r in rows
                    if r["split"] == split and r["strain"] == strain and r["proxy"] == proxy
                )["ratio"]
            )
            for proxy in proxies
        ]
        xs = np.arange(len(proxies))
        ax.bar(xs, values, color=colors, width=0.62)
        ax.axhline(tensor_ratio, color="black", linewidth=2.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(["adjacent P", "waveform", "P x geom", "geom x waveform"], rotation=25, ha="right")
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
    y_max = max(float(r["ratio"]) for r in rows if r["split"] == split and r["strain"] != "tensor")
    axes[0].set_ylim(0, max(y_max, tensor_ratio) * 1.22)
    path = OUT / "fig_geometry_scaled_single_case_ratio.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "fig_geometry_scaled_single_case_ratio.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    single_rows = cascade_single_case_rows()
    spectrum_cases = spectrum_case_rows()
    spectrum_corrs = spectrum_correlation_rows(spectrum_cases)

    print_single_summary(single_rows)
    print_spectrum_summary(spectrum_corrs)

    write_csv(OUT / "single_case_geometry_scaled_ratios.csv", single_rows)
    write_csv(OUT / "spectrum_geometry_scaled_case_values.csv", spectrum_cases)
    write_csv(OUT / "spectrum_geometry_scaled_correlations.csv", spectrum_corrs)
    fig = make_single_figure(single_rows)

    print(f"\nSaved {OUT / 'single_case_geometry_scaled_ratios.csv'}")
    print(f"Saved {OUT / 'spectrum_geometry_scaled_case_values.csv'}")
    print(f"Saved {OUT / 'spectrum_geometry_scaled_correlations.csv'}")
    print(f"Saved {fig}")


if __name__ == "__main__":
    main()
