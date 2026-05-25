#!/usr/bin/env python3
"""Core h=5 corrected-sweep analysis harness.

This is intentionally small and rerunnable. It reads the h=5 sweep manifest,
detects which jobs have completed postprocessing, and writes the key tables
needed to decide whether the h=5 sweep preserves the thesis story.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr


ROOT = Path("/home/dtsteene/D1/cardiac-work")
SIM_ROOT = ROOT / "results" / "sims"
OUT = Path(os.environ.get("H5_SWEEP_OUT", ROOT / "results" / "analysis" / "h5_sweep"))
MANIFEST = Path(
    os.environ.get(
        "H5_SWEEP_MANIFEST",
        ROOT / "results" / "analysis" / "h5_sweep_submission" / "h5_corrected_sweep_cases.tsv",
    )
)
H10_MANIFEST = ROOT / "results" / "important_sims" / "manifests" / "new_sweep_v12_exp_n16.csv"
KPA = 1e-3

PROXIES = ["PLV", "PRV", "Trans", "Mean", "NearestSide", "TauWeighted"]
STRAINS = ["ll", "ff"]
COMPONENTS = ["ff", "ss", "nn", "cross"]
RV_BRIDGE_CANDIDATES = [
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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str] | None = None) -> None:
    if not rows:
        return
    names = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_run(job_id: str) -> Path | None:
    matches = sorted(SIM_ROOT.glob(f"*/UKB_6beats_run_{job_id}"))
    return matches[-1] if matches else None


def row_run(item: dict[str, str]) -> Path | None:
    """Resolve a manifest row to a result directory.

    Older manifests only carried Slurm job ids and used the default
    UKB_6beats_run_<job_id> layout. Newer controlled-design sweeps write
    explicit per-case result directories, so prefer result_dir when present.
    """
    explicit = item.get("result_dir", "")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
    return find_run(item["job_id"])


def run_status(run: Path | None) -> str:
    if run is None:
        return "no_result_dir"
    if (run / "per_cell_data.npz").exists() and (run / "solver" / "solver_cavity_pressure_mmHg.npy").exists():
        return "complete_post"
    if (run / "simulation_stdout.log").exists() or (run / "simulation.log").exists():
        return "partial_or_running"
    return "result_dir_only"


def has_canonical_per_cell(run: Path) -> bool:
    """True when per_cell_data.npz was made with canonical atlas fields."""
    try:
        pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
        mode = str(pc["tagging_mode"].item()) if "tagging_mode" in pc.files else ""
        if mode != "canonical":
            return False
        tag3 = pc["region_tags"] == 3
        geom = pc["is_geometric_septum"].astype(bool)
        tag3_volume = float(pc["cell_volumes"][tag3].sum())
        geom_volume = float(pc["cell_volumes"][geom].sum())
        ratio = geom_volume / tag3_volume if tag3_volume else 0.0
        return 0.8 <= ratio <= 1.25
    except Exception:
        return False


def density(pc: np.lib.npyio.NpzFile, mask: np.ndarray, values: str | np.ndarray) -> float:
    arr = pc[values] if isinstance(values, str) else values
    volume = float(pc["cell_volumes"][mask].sum())
    return float(-arr[mask].sum() / volume * KPA)


def masks(pc: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    tags = pc["region_tags"]
    return {
        "LV": tags == 1,
        "RV": tags == 2,
        "Septum": pc["is_geometric_septum"].astype(bool),
    }


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


def corr(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(pearsonr(x_arr, y_arr)[0])


def last_beat_pressure(run: Path) -> tuple[float, float]:
    pressure = np.load(run / "solver" / "solver_cavity_pressure_mmHg.npy")
    beat_len = pressure.shape[0] // 6
    last = pressure[5 * beat_len :]
    return float(last[:, 0].max()), float(last[:, 1].max())


def last_beat_metrics_slice(metrics: dict[str, object]) -> slice:
    time = np.asarray(metrics["time"], dtype=float)
    samples_per_beat = len(time) // 6
    return slice(5 * samples_per_beat, 6 * samples_per_beat)


def case_values(case: str, job_id: str, run: Path) -> dict[str, object]:
    pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
    lvsp, rvsp = last_beat_pressure(run)
    region_masks = masks(pc)
    tag3 = pc["region_tags"] == 3
    geom_sept = region_masks["Septum"]
    tag3_volume = float(pc["cell_volumes"][tag3].sum() * 1e6)
    geom_volume = float(pc["cell_volumes"][geom_sept].sum() * 1e6)
    row: dict[str, object] = {
        "case": case,
        "job_id": job_id,
        "result_dir": str(run),
        "LV_ESP_mmHg": lvsp,
        "RV_ESP_mmHg": rvsp,
        "Trans_ESP_mmHg": lvsp - rvsp,
        "tag3_septum_n_cells": int(tag3.sum()),
        "tag3_septum_volume_mL": tag3_volume,
        "geometric_septum_n_cells": int(geom_sept.sum()),
        "geometric_septum_volume_mL": geom_volume,
        "geometric_to_tag3_volume_ratio": geom_volume / tag3_volume if tag3_volume else float("nan"),
    }
    for region, mask in region_masks.items():
        row[f"{region}_volume_mL"] = float(pc["cell_volumes"][mask].sum() * 1e6)
        row[f"{region}_W_total_kPa"] = density(pc, mask, "w_total")
        for component in COMPONENTS:
            row[f"{region}_W_{component}_kPa"] = density(pc, mask, f"w_{component}")
        for suffix in STRAINS:
            arrays = candidate_arrays(pc, suffix)
            for proxy, values in arrays.items():
                row[f"{region}_{proxy}_{suffix}_kPa"] = density(pc, mask, values)

    row["FW_tensor_LV_RV_ratio"] = float(row["LV_W_total_kPa"]) / float(row["RV_W_total_kPa"])
    row["FW_adjacent_ll_LV_RV_ratio"] = float(row["LV_PLV_ll_kPa"]) / float(row["RV_PRV_ll_kPa"])
    row["FW_adjacent_ff_LV_RV_ratio"] = float(row["LV_PLV_ff_kPa"]) / float(row["RV_PRV_ff_kPa"])
    fw_tensor_mean = 0.5 * (float(row["LV_W_total_kPa"]) + float(row["RV_W_total_kPa"]))
    fw_adj_ll_mean = 0.5 * (float(row["LV_PLV_ll_kPa"]) + float(row["RV_PRV_ll_kPa"]))
    row["Septum_to_FWmean_tensor_ratio"] = float(row["Septum_W_total_kPa"]) / fw_tensor_mean
    for proxy in PROXIES:
        row[f"Septum_{proxy}_ll_to_FW_adjacent_ratio"] = float(row[f"Septum_{proxy}_ll_kPa"]) / fw_adj_ll_mean
    return row


def status_rows(manifest: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for item in manifest:
        run = row_run(item)
        status = run_status(run)
        if status == "complete_post" and run is not None and not has_canonical_per_cell(run):
            status = "complete_post_noncanonical"
        rows.append(
            {
                **item,
                "status": status,
                "result_dir": str(run) if run else "",
            }
        )
    return rows


def correlation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for target_key, target_label in [
        ("Septum_W_total_kPa", "septum total tensor work"),
        ("Septum_W_ff_kPa", "septum fibre work"),
    ]:
        target = [float(row[target_key]) for row in rows]
        for suffix in STRAINS:
            for proxy in PROXIES:
                key = f"Septum_{proxy}_{suffix}_kPa"
                r = corr([float(row[key]) for row in rows], target)
                out.append(
                    {
                        "n": len(rows),
                        "target": target_label,
                        "target_key": target_key,
                        "strain": suffix,
                        "proxy": proxy,
                        "r": r,
                        "r2": r * r if not math.isnan(r) else float("nan"),
                    }
                )
    return out


def ratio_preservation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    tensor = np.asarray([float(row["Septum_to_FWmean_tensor_ratio"]) for row in rows])
    for proxy in PROXIES:
        vals = np.asarray([float(row[f"Septum_{proxy}_ll_to_FW_adjacent_ratio"]) for row in rows])
        log_err = np.abs(np.log(np.abs(vals / tensor)))
        raw_err = np.abs(vals - tensor)
        out.append(
            {
                "n": len(rows),
                "strain": "ll",
                "proxy": proxy,
                "mean_abs_raw_error": float(np.mean(raw_err)),
                "max_abs_raw_error": float(np.max(raw_err)),
                "mean_abs_log_error": float(np.mean(log_err)),
                "max_abs_log_error": float(np.max(log_err)),
            }
        )
    return out


def freewall_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        tensor = float(row["FW_tensor_LV_RV_ratio"])
        out.append(
            {
                "case": row["case"],
                "job_id": row["job_id"],
                "RV_ESP_mmHg": row["RV_ESP_mmHg"],
                "tensor_LV_RV_ratio": tensor,
                "adjacent_ll_LV_RV_ratio": row["FW_adjacent_ll_LV_RV_ratio"],
                "adjacent_ll_abs_error": abs(float(row["FW_adjacent_ll_LV_RV_ratio"]) - tensor),
                "adjacent_ff_LV_RV_ratio": row["FW_adjacent_ff_LV_RV_ratio"],
                "adjacent_ff_abs_error": abs(float(row["FW_adjacent_ff_LV_RV_ratio"]) - tensor),
            }
        )
    return out


def component_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for region in ["LV", "RV", "Septum"]:
        total = np.asarray([float(row[f"{region}_W_total_kPa"]) for row in rows])
        for comp in COMPONENTS:
            vals = np.asarray([float(row[f"{region}_W_{comp}_kPa"]) for row in rows])
            frac = vals / total
            out.append(
                {
                    "n": len(rows),
                    "region": region,
                    "component": comp,
                    "mean_kPa": float(vals.mean()),
                    "min_kPa": float(vals.min()),
                    "max_kPa": float(vals.max()),
                    "mean_signed_fraction": float(frac.mean()),
                    "min_signed_fraction": float(frac.min()),
                    "max_signed_fraction": float(frac.max()),
                }
            )
    return out


def rv_bridge_case_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        run = Path(str(row["result_dir"]))
        metrics_path = run / "metrics" / "metrics_downsample_1.npy"
        if not metrics_path.exists():
            continue
        metrics = np.load(metrics_path, allow_pickle=True).item()
        beat = last_beat_metrics_slice(metrics)
        time = np.asarray(metrics["time"], dtype=float)[beat]
        time = time - time[0]
        eps_ll = np.asarray(metrics["mean_E_ll_RV"], dtype=float)[beat]
        eps_ll = eps_ll - eps_ll[0]
        deps = np.diff(eps_ll)
        dt = np.diff(time)
        rv_pressure = np.asarray(metrics["p_RV"], dtype=float)[beat]
        rv_volume = np.asarray(metrics["V_RV_FEM"], dtype=float)[beat]
        out.append(
            {
                "case": row["case"],
                "job_id": row["job_id"],
                "rvsp_mmHg": float(np.max(rv_pressure)),
                "rv_ef_fem_percent": float((np.max(rv_volume) - np.min(rv_volume)) / np.max(rv_volume) * 100.0),
                "peak_abs_E_ll": float(abs(np.min(eps_ll))),
                "peak_to_peak_E_ll": float(np.max(eps_ll) - np.min(eps_ll)),
                "strain_path_E_ll": float(np.sum(np.abs(deps))),
                "strain_rate_energy_E_ll": float(np.sum((deps / np.maximum(dt, 1e-9)) ** 2 * dt)),
                "strain_time_energy_E_ll": float(np.trapz(eps_ll**2, time)),
                "rv_tensor_work_density_kPa": row["RV_W_total_kPa"],
                "rv_fibre_work_density_kPa": row["RV_W_ff_kPa"],
                "rv_ps_ll_density_kPa": row["RV_PRV_ll_kPa"],
                "rv_ps_ff_density_kPa": row["RV_PRV_ff_kPa"],
                "rv_wrong_plv_ps_ll_density_kPa": row["RV_PLV_ll_kPa"],
            }
        )
    return out


def rv_bridge_correlation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    target = [float(row["rv_tensor_work_density_kPa"]) for row in rows]
    out: list[dict[str, object]] = []
    for key, label in RV_BRIDGE_CANDIDATES:
        r = corr([float(row[key]) for row in rows], target)
        out.append(
            {
                "n": len(rows),
                "target": "RV free-wall total tensor work density",
                "metric_key": key,
                "metric": label,
                "r": r,
                "r2": r * r if not math.isnan(r) else float("nan"),
            }
        )
    return out


def h10_cases() -> dict[str, str]:
    if not H10_MANIFEST.exists():
        return {}
    out = {}
    for row in read_csv(H10_MANIFEST):
        out[row["label"]] = row["run_id"]
    return out


def h10_comparison_rows(h5_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    h10 = h10_cases()
    out = []
    qois = [
        "LV_ESP_mmHg",
        "RV_ESP_mmHg",
        "FW_tensor_LV_RV_ratio",
        "FW_adjacent_ll_LV_RV_ratio",
        "Septum_W_total_kPa",
        "Septum_to_FWmean_tensor_ratio",
        "Septum_Mean_ll_to_FW_adjacent_ratio",
    ]
    for h5 in h5_rows:
        case = str(h5["case"])
        job_id = h10.get(case)
        if not job_id:
            continue
        run = find_run(job_id)
        if run is None or run_status(run) != "complete_post":
            continue
        old = case_values(case, job_id, run)
        h10_mask_ratio = float(old["geometric_to_tag3_volume_ratio"])
        h5_mask_ratio = float(h5["geometric_to_tag3_volume_ratio"])
        mask_comparable = 0.8 <= h10_mask_ratio <= 1.25 and 0.8 <= h5_mask_ratio <= 1.25
        for qoi in qois:
            h5_value = float(h5[qoi])
            h10_value = float(old[qoi])
            out.append(
                {
                    "case": case,
                    "qoi": qoi,
                    "h10_job": job_id,
                    "h5_job": h5["job_id"],
                    "h10_value": h10_value,
                    "h5_value": h5_value,
                    "absolute_difference": h5_value - h10_value,
                    "percent_difference_vs_h5": abs(h5_value - h10_value) / max(abs(h5_value), 1e-12) * 100.0,
                    "h10_geometric_to_tag3_volume_ratio": h10_mask_ratio,
                    "h5_geometric_to_tag3_volume_ratio": h5_mask_ratio,
                    "septum_mask_comparable": mask_comparable,
                }
            )
    return out


def mask_diagnostic_rows(h5_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in h5_rows:
        rows.append(
            {
                "dataset": "h5",
                "case": row["case"],
                "job_id": row["job_id"],
                "tag3_septum_n_cells": row["tag3_septum_n_cells"],
                "tag3_septum_volume_mL": row["tag3_septum_volume_mL"],
                "geometric_septum_n_cells": row["geometric_septum_n_cells"],
                "geometric_septum_volume_mL": row["geometric_septum_volume_mL"],
                "geometric_to_tag3_volume_ratio": row["geometric_to_tag3_volume_ratio"],
            }
        )
    for case, job_id in h10_cases().items():
        run = find_run(job_id)
        if run is None or run_status(run) != "complete_post":
            continue
        row = case_values(case, job_id, run)
        rows.append(
            {
                "dataset": "h10_n16",
                "case": row["case"],
                "job_id": row["job_id"],
                "tag3_septum_n_cells": row["tag3_septum_n_cells"],
                "tag3_septum_volume_mL": row["tag3_septum_volume_mL"],
                "geometric_septum_n_cells": row["geometric_septum_n_cells"],
                "geometric_septum_volume_mL": row["geometric_septum_volume_mL"],
                "geometric_to_tag3_volume_ratio": row["geometric_to_tag3_volume_ratio"],
            }
        )
    return rows


def markdown_summary(
    status: list[dict[str, object]],
    h5_rows: list[dict[str, object]],
    septum_corr: list[dict[str, object]],
    septum_ratio: list[dict[str, object]],
    rv_bridge_corr: list[dict[str, object]],
    h10_diff: list[dict[str, object]],
) -> str:
    complete = [row for row in status if row["status"] == "complete_post"]
    pending = [row for row in status if row["status"] != "complete_post"]
    lines = [
        "# h=5 Corrected Sweep Core Status",
        "",
        f"Complete postprocessed cases: {len(complete)} / {len(status)}",
        "",
    ]
    if pending:
        lines.append("## Not Yet Complete")
        lines.append("")
        for row in pending:
            lines.append(f"- {row['case']} job {row['job_id']}: {row['status']}")
        lines.append("")
    if h5_rows:
        rv_range = (min(float(row["RV_ESP_mmHg"]) for row in h5_rows), max(float(row["RV_ESP_mmHg"]) for row in h5_rows))
        lv_range = (min(float(row["LV_ESP_mmHg"]) for row in h5_rows), max(float(row["LV_ESP_mmHg"]) for row in h5_rows))
        lines += [
            "## Current Completed Range",
            "",
            f"- LV ESP range: {lv_range[0]:.1f}-{lv_range[1]:.1f} mmHg",
            f"- RV ESP range: {rv_range[0]:.1f}-{rv_range[1]:.1f} mmHg",
            "",
        ]
        if len(h5_rows) < 5:
            lines += [
                "Only the endpoint/anchor cases are complete, so correlation rankings below are a smoke test rather than the final sweep result.",
                "",
            ]
        lines += [
            "## Septum Proxy Correlations",
            "",
        ]
        for target in ["septum total tensor work", "septum fibre work"]:
            rr = [row for row in septum_corr if row["target"] == target and row["strain"] == "ll"]
            vals = ", ".join(f"{row['proxy']} r={float(row['r']):+.3f}" for row in rr)
            lines.append(f"- {target}, longitudinal: {vals}")
        lines += ["", "## Septum Ratio Preservation", ""]
        for row in septum_ratio:
            lines.append(
                f"- {row['proxy']}: mean log error {float(row['mean_abs_log_error']):.3f}, "
                f"max {float(row['max_abs_log_error']):.3f}"
            )
        lines.append("")
    if rv_bridge_corr:
        lines += ["## RV Free-Wall Bridge", ""]
        for key in ["rv_ps_ll_density_kPa", "peak_abs_E_ll", "rvsp_mmHg", "rv_wrong_plv_ps_ll_density_kPa"]:
            match = next((row for row in rv_bridge_corr if row["metric_key"] == key), None)
            if match is None:
                continue
            lines.append(f"- {match['metric']}: r={float(match['r']):+.3f}")
        lines.append("")
    if h10_diff:
        lines += ["## h=10 to h=5 Drift", ""]
        by_qoi: dict[str, list[float]] = {}
        for row in h10_diff:
            qoi = str(row["qoi"])
            if qoi.startswith("Septum") and not bool(row.get("septum_mask_comparable", True)):
                continue
            by_qoi.setdefault(str(row["qoi"]), []).append(float(row["percent_difference_vs_h5"]))
        skipped = [
            row
            for row in h10_diff
            if str(row["qoi"]).startswith("Septum") and not bool(row.get("septum_mask_comparable", True))
        ]
        for qoi, vals in by_qoi.items():
            lines.append(f"- {qoi}: max {max(vals):.2f}% over {len(vals)} paired completed cases")
        if skipped:
            cases = sorted({str(row["case"]) for row in skipped})
            lines.append(
                "- Septal h=10-to-h=5 differences are not summarized for "
                f"{', '.join(cases)} because the 10 mm geometric septum mask "
                "does not match the tag-3/canonical septum volume."
            )
        lines.append("")
    if h5_rows:
        bad_masks = [
            row
            for row in h5_rows
            if not (0.8 <= float(row["geometric_to_tag3_volume_ratio"]) <= 1.25)
        ]
        lines += ["## Septum Mask Check", ""]
        if bad_masks:
            for row in bad_masks:
                lines.append(
                    f"- WARNING {row['case']}: geometric/tag-3 septum volume ratio "
                    f"{float(row['geometric_to_tag3_volume_ratio']):.3f}"
                )
        else:
            lines.append("- Completed h=5 cases have geometric septum volumes matching the tag-3 septum.")
        lines.append("")
    lines.append("Generated by `analyze_h5_sweep_core.py`.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_tsv(MANIFEST)
    status = status_rows(manifest)
    write_csv(OUT / "h5_sweep_case_status.csv", status)

    h5_rows = []
    for row in status:
        if row["status"] != "complete_post":
            continue
        run = Path(str(row["result_dir"]))
        h5_rows.append(case_values(str(row["case"]), str(row["job_id"]), run))

    if h5_rows:
        write_csv(OUT / "h5_sweep_case_values.csv", h5_rows)
        freewall = freewall_summary_rows(h5_rows)
        septum_corr = correlation_rows(h5_rows)
        septum_ratio = ratio_preservation_rows(h5_rows)
        components = component_summary_rows(h5_rows)
        rv_bridge = rv_bridge_case_rows(h5_rows)
        rv_bridge_corr = rv_bridge_correlation_rows(rv_bridge)
        h10_diff = h10_comparison_rows(h5_rows)
        masks_out = mask_diagnostic_rows(h5_rows)
        write_csv(OUT / "h5_freewall_ratio_summary.csv", freewall)
        write_csv(OUT / "h5_septum_proxy_correlations.csv", septum_corr)
        write_csv(OUT / "h5_septum_ratio_preservation.csv", septum_ratio)
        write_csv(OUT / "h5_work_component_summary.csv", components)
        write_csv(OUT / "h5_rv_bridge_case_values.csv", rv_bridge)
        write_csv(OUT / "h5_rv_bridge_correlations.csv", rv_bridge_corr)
        write_csv(OUT / "h10_to_h5_key_qoi_differences.csv", h10_diff)
        write_csv(OUT / "septum_mask_diagnostics.csv", masks_out)
    else:
        septum_corr = []
        septum_ratio = []
        rv_bridge_corr = []
        h10_diff = []

    (OUT / "h5_sweep_summary.md").write_text(
        markdown_summary(status, h5_rows, septum_corr, septum_ratio, rv_bridge_corr, h10_diff)
    )
    print(f"Wrote {OUT}")
    print(f"Complete postprocessed cases: {sum(row['status'] == 'complete_post' for row in status)} / {len(status)}")


if __name__ == "__main__":
    main()
