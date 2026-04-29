#!/usr/bin/env python3
"""RV free-wall pressure-strain bridge to Lakatos et al. 2024.

Lakatos et al. show clinically that RV pressure-strain work correlates with
invasive RV contractility better than RV GLS alone. This script asks the FEM
analogue available in this thesis: across the corrected pressure-loading sweep,
does the RV free-wall pressure-longitudinal-strain proxy track model-resolved
RV free-wall tensor work better than strain-only quantities?
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


ROOT = Path("/home/dtsteene/D1/cardiac-work")
RUN_ROOTS = [
    ROOT / "results" / "sims" / "2026-04-23",
    ROOT / "results" / "sims" / "2026-04-24",
]
OUT = ROOT / "results" / "analysis" / "rv_lakatos_bridge"
KPA = 1e-3

CASES = [
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


def find_run(job_id: int) -> Path:
    for root in RUN_ROOTS:
        run_dir = root / f"UKB_6beats_run_{job_id}"
        if run_dir.exists():
            return run_dir
    raise FileNotFoundError(job_id)


def density(pc: np.lib.npyio.NpzFile, mask: np.ndarray, key: str) -> float:
    volume = float(pc["cell_volumes"][mask].sum())
    return float(-pc[key][mask].sum() / volume * KPA)


def corr(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(pearsonr(x_arr, y_arr)[0])


def last_beat_slice(metrics: dict[str, object]) -> slice:
    time = np.asarray(metrics["time"], dtype=float)
    samples_per_beat = len(time) // 6
    return slice(5 * samples_per_beat, 6 * samples_per_beat)


def case_row(label: str, job_id: int) -> dict[str, float | str | int]:
    run_dir = find_run(job_id)
    pc = np.load(run_dir / "per_cell_data.npz", allow_pickle=True)
    metrics = np.load(run_dir / "metrics" / "metrics_downsample_1.npy", allow_pickle=True).item()

    rv_mask = pc["region_tags"] == 2
    beat = last_beat_slice(metrics)
    time = np.asarray(metrics["time"], dtype=float)[beat]
    time = time - time[0]

    eps_ll = np.asarray(metrics["mean_E_ll_RV"], dtype=float)[beat]
    eps_ll = eps_ll - eps_ll[0]
    deps = np.diff(eps_ll)
    dt = np.diff(time)

    rv_pressure = np.asarray(metrics["p_RV"], dtype=float)[beat]
    rv_volume = np.asarray(metrics["V_RV_FEM"], dtype=float)[beat]

    return {
        "case": label,
        "job_id": job_id,
        "rvsp_mmHg": float(np.max(rv_pressure)),
        "rv_ef_fem_percent": float((np.max(rv_volume) - np.min(rv_volume)) / np.max(rv_volume) * 100.0),
        "peak_abs_E_ll": float(abs(np.min(eps_ll))),
        "peak_to_peak_E_ll": float(np.max(eps_ll) - np.min(eps_ll)),
        "strain_path_E_ll": float(np.sum(np.abs(deps))),
        "strain_rate_energy_E_ll": float(np.sum((deps / np.maximum(dt, 1e-9)) ** 2 * dt)),
        "strain_time_energy_E_ll": float(np.trapz(eps_ll**2, time)),
        "rv_tensor_work_density_kPa": density(pc, rv_mask, "w_total"),
        "rv_fibre_work_density_kPa": density(pc, rv_mask, "w_ff"),
        "rv_ps_ll_density_kPa": density(pc, rv_mask, "proxy_PRV_ll"),
        "rv_ps_ff_density_kPa": density(pc, rv_mask, "proxy_PRV_ff"),
        "rv_wrong_plv_ps_ll_density_kPa": density(pc, rv_mask, "proxy_PLV_ll"),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_correlations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    target = [float(row["rv_tensor_work_density_kPa"]) for row in rows]
    candidates = [
        ("rv_ps_ll_density_kPa", "RV pressure x longitudinal strain"),
        ("rv_ps_ff_density_kPa", "RV pressure x fibre strain"),
        ("peak_abs_E_ll", "peak absolute longitudinal strain"),
        ("peak_to_peak_E_ll", "peak-to-peak longitudinal strain"),
        ("strain_path_E_ll", "longitudinal strain path length"),
        ("strain_rate_energy_E_ll", "integral strain-rate squared"),
        ("strain_time_energy_E_ll", "integral longitudinal strain squared"),
        ("rv_ef_fem_percent", "FEM RV ejection fraction analogue"),
        ("rvsp_mmHg", "RV systolic pressure"),
        ("rv_wrong_plv_ps_ll_density_kPa", "LV pressure x RV longitudinal strain"),
    ]
    out = []
    for key, label in candidates:
        r = corr([float(row[key]) for row in rows], target)
        out.append(
            {
                "target": "RV free-wall total tensor work density",
                "metric_key": key,
                "metric": label,
                "r": r,
                "r2": r * r,
            }
        )
    return out


def add_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    order = np.argsort(x)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(x[order], slope * x[order] + intercept, color=color, linewidth=1.7)


def make_figure(rows: list[dict[str, object]], corr_rows: list[dict[str, object]]) -> None:
    tensor = np.array([float(row["rv_tensor_work_density_kPa"]) for row in rows])
    ps_ll = np.array([float(row["rv_ps_ll_density_kPa"]) for row in rows])
    peak = 100.0 * np.array([float(row["peak_abs_E_ll"]) for row in rows])
    rvsp = np.array([float(row["rvsp_mmHg"]) for row in rows])

    r_ps = next(float(row["r"]) for row in corr_rows if row["metric_key"] == "rv_ps_ll_density_kPa")
    r_peak = next(float(row["r"]) for row in corr_rows if row["metric_key"] == "peak_abs_E_ll")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True, sharey=True)
    cmap = "viridis"

    sc0 = axes[0].scatter(peak, tensor, c=rvsp, cmap=cmap, s=56, edgecolor="black", linewidth=0.4)
    add_fit(axes[0], peak, tensor, "#666666")
    axes[0].set_xlabel("peak RV longitudinal shortening (%)")
    axes[0].set_ylabel("RV free-wall tensor work density (kPa)")
    axes[0].set_title(f"strain alone: r = {r_peak:+.3f}")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(ps_ll, tensor, c=rvsp, cmap=cmap, s=56, edgecolor="black", linewidth=0.4)
    add_fit(axes[1], ps_ll, tensor, "#666666")
    axes[1].set_xlabel("RV pressure-strain work density (kPa)")
    axes[1].set_title(f"pressure x strain: r = {r_ps:+.3f}")
    axes[1].grid(True, alpha=0.25)

    cbar = fig.colorbar(sc0, ax=axes, shrink=0.88)
    cbar.set_label("RV systolic pressure (mmHg)")

    fig.suptitle("RV free wall: FEM analogue of the Lakatos pressure-strain argument")
    fig.savefig(OUT / "fig_rv_lakatos_bridge.png", dpi=300)
    fig.savefig(OUT / "fig_rv_lakatos_bridge.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [case_row(label, job_id) for label, job_id in CASES]
    corr_rows = make_correlations(rows)
    write_csv(OUT / "rv_lakatos_bridge_case_values.csv", rows)
    write_csv(OUT / "rv_lakatos_bridge_correlations.csv", corr_rows)
    make_figure(rows, corr_rows)

    print("RV free-wall bridge to Lakatos et al. 2024")
    print("=" * 58)
    for row in corr_rows:
        print(f"{row['metric']:<42} r={float(row['r']):+0.3f}  R2={float(row['r2']):0.3f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
