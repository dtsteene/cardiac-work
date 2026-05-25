#!/usr/bin/env python3
"""Exploratory analysis for the patient-mesh pressure sweeps.

The patient-specific meshes are useful as a geometry-sensitivity check, but
they are not the same controlled experiment as the canonical h=5 sweep.  This
script therefore keeps the analysis deliberately local to each mesh:

1. report the healthy and PAH mesh geometry,
2. report which mesh-pressure jobs completed and which run was selected,
3. compute within-mesh pressure/proxy diagnostics, and
4. compare same nominal pressure cases between meshes as an exploratory
   geometry contrast.

The outputs are written to results/analysis/patient_geometry_exploratory.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from patient_mesh_volume_check import read_mesh


ROOT = Path("/home/dtsteene/D1/cardiac-work")
MANIFEST_ROOT = ROOT / "results" / "patient_mesh_sweep"
SIM_ROOT = ROOT / "results" / "sims"
OUT = ROOT / "results" / "analysis" / "patient_geometry_exploratory"
KPA = 1e-3

CASE_ORDER = ["sPAP22", "sPAP30", "sPAP45", "sPAP55", "sPAP65", "sPAP75", "sPAP85", "sPAP95"]
MESHES = ["healthy", "pah"]
REGIONS = ["LV", "RV", "Septum"]
STRAINS = {
    "ll": "longitudinal",
    "ff": "fibre",
    "radial": "geometric radial",
    "circ": "geometric circumferential",
}
PRESSURES = ["PLV", "PRV", "Trans", "Mean", "NearestSide", "TauWeighted"]


def case_index(case: str) -> int:
    return CASE_ORDER.index(case) if case in CASE_ORDER else 999


def manifest_sort_key(path: Path) -> tuple[int, str]:
    """Prefer later resubmission manifests when duplicate jobs exist."""
    name = path.name
    if "resubmit" in name:
        return (1, name)
    return (0, name)


def read_manifests() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not MANIFEST_ROOT.exists():
        return rows
    for path in sorted(MANIFEST_ROOT.glob("*.tsv"), key=manifest_sort_key):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if not row.get("mesh_key") or not row.get("case") or not row.get("job_id"):
                    continue
                row["manifest"] = str(path)
                rows.append(row)
    return rows


def find_result_dir(job_id: str) -> Path | None:
    matches = sorted(SIM_ROOT.glob(f"*/*_run_{job_id}"))
    return matches[-1] if matches else None


def pressure_path(run_dir: Path) -> Path:
    return run_dir / "solver" / "solver_cavity_pressure_mmHg.npy"


def row_status(row: dict[str, str]) -> dict[str, Any]:
    run_dir = find_result_dir(row["job_id"])
    pc_path = run_dir / "per_cell_data.npz" if run_dir else None
    p_path = pressure_path(run_dir) if run_dir else None
    complete = bool(run_dir and pc_path and pc_path.exists() and p_path.exists())
    return {
        "mesh_key": row["mesh_key"],
        "case": row["case"],
        "job_id": row["job_id"],
        "manifest": row["manifest"],
        "run_dir": str(run_dir) if run_dir else "",
        "has_run_dir": run_dir is not None,
        "has_per_cell_data": bool(pc_path and pc_path.exists()),
        "has_pressure": bool(p_path and p_path.exists()),
        "complete": complete,
        "selected": False,
    }


def select_completed_rows(manifest_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Return one completed row per mesh/case, preferring newer/larger job ids."""
    statuses = [row_status(row) for row in manifest_rows]
    selected_by_case: dict[tuple[str, str], dict[str, str]] = {}
    for row, status in zip(manifest_rows, statuses):
        if not status["complete"]:
            continue
        key = (row["mesh_key"], row["case"])
        old = selected_by_case.get(key)
        if old is None or int(row["job_id"]) > int(old["job_id"]):
            selected_by_case[key] = row

    selected_keys = {(row["mesh_key"], row["case"], row["job_id"]) for row in selected_by_case.values()}
    for status in statuses:
        status["selected"] = (status["mesh_key"], status["case"], status["job_id"]) in selected_keys

    selected = sorted(selected_by_case.values(), key=lambda row: (row["mesh_key"], case_index(row["case"])))
    return selected, statuses


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def corr(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(pearsonr(x_arr, y_arr)[0])


def density(pc: np.lib.npyio.NpzFile, mask: np.ndarray, values: str | np.ndarray) -> float:
    arr = pc[values] if isinstance(values, str) else values
    volume = float(pc["cell_volumes"][mask].sum())
    if volume <= 0:
        return float("nan")
    return float(-arr[mask].sum() / volume * KPA)


def region_masks(pc: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    tags = pc["region_tags"]
    return {
        "LV": tags == 1,
        "RV": tags == 2,
        "Septum": pc["is_geometric_septum"].astype(bool),
    }


def has_strain(pc: np.lib.npyio.NpzFile, suffix: str) -> bool:
    return f"proxy_PLV_{suffix}" in pc.files and f"proxy_PRV_{suffix}" in pc.files


def proxy_arrays(pc: np.lib.npyio.NpzFile, suffix: str) -> dict[str, np.ndarray]:
    # Canonical convention for pressure choices: tau=0 on the LV side and
    # tau=1 on the RV side. The saved Laplace scalar has the opposite orientation.
    tau = 1.0 - pc["lv_rv_scalar"] if "lv_rv_scalar" in pc.files else pc["tau"]
    plv = pc[f"proxy_PLV_{suffix}"]
    prv = pc[f"proxy_PRV_{suffix}"]
    trans_key = f"proxy_Trans_{suffix}"
    trans = pc[trans_key] if trans_key in pc.files else plv - prv
    return {
        "PLV": plv,
        "PRV": prv,
        "Trans": trans,
        "Mean": 0.5 * (plv + prv),
        "NearestSide": np.where(tau < 0.5, plv, prv),
        "TauWeighted": (1.0 - tau) * plv + tau * prv,
    }


def last_beat_pressures(run_dir: Path) -> tuple[float, float]:
    pressure = np.load(pressure_path(run_dir))
    start = 5 * (pressure.shape[0] // 6) if pressure.shape[0] >= 6 else 0
    last = pressure[start:]
    return float(last[:, 0].max()), float(last[:, 1].max())


def scalar_value(value: np.ndarray) -> str:
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return ""


def load_case(row: dict[str, str]) -> dict[str, Any]:
    run_dir = find_result_dir(row["job_id"])
    if run_dir is None:
        raise FileNotFoundError(row["job_id"])
    pc = np.load(run_dir / "per_cell_data.npz", allow_pickle=True)
    lv_sp, rv_sp = last_beat_pressures(run_dir)
    masks = region_masks(pc)

    out: dict[str, Any] = {
        "mesh_key": row["mesh_key"],
        "case": row["case"],
        "case_index": case_index(row["case"]),
        "job_id": row["job_id"],
        "manifest": row["manifest"],
        "run_dir": str(run_dir),
        "LVSP_mmHg": lv_sp,
        "RVSP_mmHg": rv_sp,
        "TransSP_mmHg": lv_sp - rv_sp,
        "tagging_mode": scalar_value(pc["tagging_mode"]) if "tagging_mode" in pc.files else "",
    }
    for suffix in STRAINS:
        out[f"has_{suffix}"] = has_strain(pc, suffix)

    for region, mask in masks.items():
        out[f"{region}_volume_mL"] = float(pc["cell_volumes"][mask].sum() * 1e6)
        out[f"{region}_W_total_kPa"] = density(pc, mask, "w_total")
        out[f"{region}_W_ff_kPa"] = density(pc, mask, "w_ff")
        out[f"{region}_W_ss_kPa"] = density(pc, mask, "w_ss")
        out[f"{region}_W_nn_kPa"] = density(pc, mask, "w_nn")
        out[f"{region}_W_cross_kPa"] = density(pc, mask, "w_cross")
        for suffix in STRAINS:
            if not has_strain(pc, suffix):
                continue
            for pressure, values in proxy_arrays(pc, suffix).items():
                out[f"{region}_{pressure}_{suffix}_kPa"] = density(pc, mask, values)

    out["FW_tensor_LV_RV_ratio"] = out["LV_W_total_kPa"] / out["RV_W_total_kPa"]
    out["Septum_to_FWmean_tensor_ratio"] = out["Septum_W_total_kPa"] / (
        0.5 * (out["LV_W_total_kPa"] + out["RV_W_total_kPa"])
    )
    return out


def finite_ratio_error(values: list[float], targets: list[float]) -> tuple[float, float, int]:
    raw: list[float] = []
    log: list[float] = []
    for value, target in zip(values, targets):
        if not np.isfinite(value) or not np.isfinite(target):
            continue
        raw.append(abs(value - target))
        if value > 0 and target > 0:
            log.append(abs(math.log(value / target)))
    return (
        float(np.mean(raw)) if raw else float("nan"),
        float(np.mean(log)) if log else float("nan"),
        len(log),
    )


def summarize_mesh(rows: list[dict[str, Any]], mesh_key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mesh_rows = sorted([row for row in rows if row["mesh_key"] == mesh_key], key=lambda row: row["case_index"])
    freewall_rows: list[dict[str, Any]] = []
    septum_rows: list[dict[str, Any]] = []
    if len(mesh_rows) < 3:
        return freewall_rows, septum_rows

    for suffix, strain_label in STRAINS.items():
        available = [row for row in mesh_rows if row.get(f"has_{suffix}")]
        if len(available) < 3:
            continue

        fw_pred = [
            float(row[f"LV_PLV_{suffix}_kPa"]) / float(row[f"RV_PRV_{suffix}_kPa"])
            for row in available
        ]
        fw_target = [float(row["FW_tensor_LV_RV_ratio"]) for row in available]
        fw_raw, fw_log, fw_log_n = finite_ratio_error(fw_pred, fw_target)

        freewall_rows.append(
            {
                "mesh_key": mesh_key,
                "n": len(available),
                "strain_suffix": suffix,
                "strain": strain_label,
                "freewall_adjacent_mean_abs_ratio_error": fw_raw,
                "freewall_adjacent_mean_abs_log_ratio_error": fw_log,
                "freewall_adjacent_log_n": fw_log_n,
                "freewall_ratio_r": corr(fw_pred, fw_target),
                "rv_freewall_adjacent_r": corr(
                    [float(row[f"RV_PRV_{suffix}_kPa"]) for row in available],
                    [float(row["RV_W_total_kPa"]) for row in available],
                ),
                "rvsp_only_r": corr(
                    [float(row["RVSP_mmHg"]) for row in available],
                    [float(row["RV_W_total_kPa"]) for row in available],
                ),
            }
        )

        fw_mean_proxy = [
            0.5 * (float(row[f"LV_PLV_{suffix}_kPa"]) + float(row[f"RV_PRV_{suffix}_kPa"]))
            for row in available
        ]
        sept_target_density = [float(row["Septum_W_total_kPa"]) for row in available]
        sept_ratio_target = [float(row["Septum_to_FWmean_tensor_ratio"]) for row in available]
        for pressure in PRESSURES:
            key = f"Septum_{pressure}_{suffix}_kPa"
            if key not in available[0]:
                continue
            sept_proxy = [float(row[key]) for row in available]
            sept_ratio_pred = [
                value / fw if abs(fw) > 1e-30 else float("nan")
                for value, fw in zip(sept_proxy, fw_mean_proxy)
            ]
            sept_raw, sept_log, sept_log_n = finite_ratio_error(sept_ratio_pred, sept_ratio_target)
            septum_rows.append(
                {
                    "mesh_key": mesh_key,
                    "n": len(available),
                    "strain_suffix": suffix,
                    "strain": strain_label,
                    "pressure": pressure,
                    "septum_r_vs_tensor": corr(sept_proxy, sept_target_density),
                    "septum_mean_abs_ratio_error": sept_raw,
                    "septum_mean_abs_log_ratio_error": sept_log,
                    "septum_log_n": sept_log_n,
                    "septum_proxy_mean_kPa": float(np.mean(sept_proxy)),
                    "septum_proxy_min_kPa": float(np.min(sept_proxy)),
                    "septum_proxy_max_kPa": float(np.max(sept_proxy)),
                    "septum_ratio_pred_mean": float(np.mean(sept_ratio_pred)),
                    "septum_ratio_pred_min": float(np.min(sept_ratio_pred)),
                    "septum_ratio_pred_max": float(np.max(sept_ratio_pred)),
                }
            )
    return freewall_rows, septum_rows


def best_rows(freewall_rows: list[dict[str, Any]], septum_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    strain_order = {suffix: i for i, suffix in enumerate(STRAINS)}
    keys = sorted(
        {(row["mesh_key"], row["strain_suffix"]) for row in freewall_rows},
        key=lambda item: (item[0], strain_order.get(item[1], 999)),
    )
    for mesh_key, suffix in keys:
        fw = next(row for row in freewall_rows if row["mesh_key"] == mesh_key and row["strain_suffix"] == suffix)
        sept = [row for row in septum_rows if row["mesh_key"] == mesh_key and row["strain_suffix"] == suffix]
        if not sept:
            continue
        best_corr = max(
            sept,
            key=lambda row: -np.inf if not np.isfinite(float(row["septum_r_vs_tensor"])) else float(row["septum_r_vs_tensor"]),
        )
        valid_mag = [row for row in sept if np.isfinite(float(row["septum_mean_abs_log_ratio_error"]))]
        best_mag = min(valid_mag, key=lambda row: float(row["septum_mean_abs_log_ratio_error"])) if valid_mag else None
        best_raw = min(sept, key=lambda row: float(row["septum_mean_abs_ratio_error"]))
        out.append(
            {
                "mesh_key": mesh_key,
                "n": fw["n"],
                "strain_suffix": suffix,
                "strain": fw["strain"],
                "freewall_adjacent_mean_abs_ratio_error": fw["freewall_adjacent_mean_abs_ratio_error"],
                "freewall_adjacent_mean_abs_log_ratio_error": fw["freewall_adjacent_mean_abs_log_ratio_error"],
                "rv_freewall_adjacent_r": fw["rv_freewall_adjacent_r"],
                "rvsp_only_r": fw["rvsp_only_r"],
                "best_septum_correlation_pressure": best_corr["pressure"],
                "best_septum_correlation_r": best_corr["septum_r_vs_tensor"],
                "best_septum_magnitude_pressure": best_mag["pressure"] if best_mag else "",
                "best_septum_magnitude_log_error": best_mag["septum_mean_abs_log_ratio_error"] if best_mag else float("nan"),
                "best_septum_magnitude_log_n": best_mag["septum_log_n"] if best_mag else 0,
                "best_septum_raw_magnitude_pressure": best_raw["pressure"],
                "best_septum_raw_magnitude_error": best_raw["septum_mean_abs_ratio_error"],
            }
        )
    return out


def geometry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mesh_key in MESHES:
        row = read_mesh(mesh_key)
        rows.append(
            {
                "mesh_key": mesh_key,
                "LV_EDV_mL": row["LV_EDV_mL"],
                "RV_EDV_mL": row["RV_EDV_mL"],
                "RV_to_LV": row["RV_to_LV"],
                "wall_volume_mL": row["wall_volume_mL"],
                "nodes": int(row["nodes"]),
                "tets": int(row["tets"]),
            }
        )
    if len(rows) == 2:
        healthy = next(row for row in rows if row["mesh_key"] == "healthy")
        pah = next(row for row in rows if row["mesh_key"] == "pah")
        for key in ["LV_EDV_mL", "RV_EDV_mL", "RV_to_LV", "wall_volume_mL"]:
            pah[f"{key}_relative_to_healthy"] = pah[key] / healthy[key]
    return rows


def contrast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mesh_case = {(row["mesh_key"], row["case"]): row for row in rows}
    out: list[dict[str, Any]] = []
    for case in CASE_ORDER:
        healthy = by_mesh_case.get(("healthy", case))
        pah = by_mesh_case.get(("pah", case))
        if healthy is None or pah is None:
            continue
        out.append(
            {
                "case": case,
                "healthy_job_id": healthy["job_id"],
                "pah_job_id": pah["job_id"],
                "healthy_RVSP_mmHg": healthy["RVSP_mmHg"],
                "pah_RVSP_mmHg": pah["RVSP_mmHg"],
                "healthy_RV_W_total_kPa": healthy["RV_W_total_kPa"],
                "pah_RV_W_total_kPa": pah["RV_W_total_kPa"],
                "pah_to_healthy_RV_W_total": pah["RV_W_total_kPa"] / healthy["RV_W_total_kPa"],
                "healthy_FW_tensor_LV_RV_ratio": healthy["FW_tensor_LV_RV_ratio"],
                "pah_FW_tensor_LV_RV_ratio": pah["FW_tensor_LV_RV_ratio"],
                "healthy_Septum_to_FWmean_tensor_ratio": healthy["Septum_to_FWmean_tensor_ratio"],
                "pah_Septum_to_FWmean_tensor_ratio": pah["Septum_to_FWmean_tensor_ratio"],
            }
        )
    return out


def fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}"


def write_markdown(
    geometry: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    best: list[dict[str, Any]],
    contrast: list[dict[str, Any]],
) -> None:
    selected_counts = {
        mesh: sum(1 for row in cases if row["mesh_key"] == mesh)
        for mesh in MESHES
    }
    incomplete = [row for row in statuses if not row["complete"]]
    with (OUT / "patient_geometry_exploratory_summary.md").open("w") as handle:
        handle.write("# Patient Geometry Exploratory Analysis\n\n")
        handle.write(
            "This analysis treats the patient-specific meshes as a geometry-sensitivity "
            "check, not as a main validation result.  Each metric below is computed "
            "within a mesh first, because pooling the healthy and PAH meshes would mix "
            "pressure effects with geometry, tagging, and convergence effects.\n\n"
        )

        handle.write("## Mesh Geometry\n\n")
        handle.write("| mesh | LV EDV mL | RV EDV mL | RV/LV | wall mL | nodes | tets |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in geometry:
            handle.write(
                f"| {row['mesh_key']} | {fmt(row['LV_EDV_mL'], 1)} | "
                f"{fmt(row['RV_EDV_mL'], 1)} | {fmt(row['RV_to_LV'], 3)} | "
                f"{fmt(row['wall_volume_mL'], 1)} | {row['nodes']} | {row['tets']} |\n"
            )
        handle.write("\n")
        if len(geometry) == 2:
            pah = next((row for row in geometry if row["mesh_key"] == "pah"), None)
            if pah is not None:
                handle.write(
                    "PAH relative to healthy: "
                    f"LV EDV {fmt(pah.get('LV_EDV_mL_relative_to_healthy'), 3)}x, "
                    f"RV EDV {fmt(pah.get('RV_EDV_mL_relative_to_healthy'), 3)}x, "
                    f"RV/LV {fmt(pah.get('RV_to_LV_relative_to_healthy'), 3)}x, "
                    f"wall volume {fmt(pah.get('wall_volume_mL_relative_to_healthy'), 3)}x.\n\n"
                )

        handle.write("## Completed Cases\n\n")
        selected_modes = sorted({str(row["tagging_mode"]) for row in cases if row.get("tagging_mode")})
        handle.write(
            f"Selected completed cases: healthy {selected_counts.get('healthy', 0)}/8, "
            f"PAH {selected_counts.get('pah', 0)}/8.  "
            f"Manifest rows without complete postprocessing outputs: {len(incomplete)} "
            "(including duplicate, resubmitted, and failed rows)."
        )
        if selected_modes:
            handle.write(f"  Selected case tagging mode: {', '.join(selected_modes)}.")
        handle.write("\n\n")

        handle.write("## Best Proxy Diagnostics\n\n")
        handle.write(
            "| mesh | strain | n | FW LV/RV error | RV FW r | RVSP-only r | "
            "best septal r | best septal magnitude error |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in best:
            mag = (
                f"{row['best_septum_magnitude_pressure']} {fmt(row['best_septum_magnitude_log_error'], 3)}"
                if np.isfinite(float(row["best_septum_magnitude_log_error"]))
                else f"raw {row['best_septum_raw_magnitude_pressure']} {fmt(row['best_septum_raw_magnitude_error'], 3)}"
            )
            handle.write(
                f"| {row['mesh_key']} | {row['strain']} | {row['n']} | "
                f"{fmt(row['freewall_adjacent_mean_abs_ratio_error'], 3)} | "
                f"{fmt(row['rv_freewall_adjacent_r'], 3, signed=True)} | "
                f"{fmt(row['rvsp_only_r'], 3, signed=True)} | "
                f"{row['best_septum_correlation_pressure']} {fmt(row['best_septum_correlation_r'], 3, signed=True)} | "
                f"{mag} |\n"
            )
        handle.write("\n")

        if contrast:
            handle.write("## Same Nominal Case Geometry Contrast\n\n")
            handle.write(
                "| case | RVSP healthy/PAH | RV work healthy/PAH | "
                "LV/RV tensor ratio healthy/PAH | septum/FW ratio healthy/PAH |\n"
            )
            handle.write("|---|---:|---:|---:|---:|\n")
            for row in contrast:
                handle.write(
                    f"| {row['case']} | {fmt(row['healthy_RVSP_mmHg'], 1)}/{fmt(row['pah_RVSP_mmHg'], 1)} | "
                    f"{fmt(row['healthy_RV_W_total_kPa'], 2)}/{fmt(row['pah_RV_W_total_kPa'], 2)} | "
                    f"{fmt(row['healthy_FW_tensor_LV_RV_ratio'], 2)}/{fmt(row['pah_FW_tensor_LV_RV_ratio'], 2)} | "
                    f"{fmt(row['healthy_Septum_to_FWmean_tensor_ratio'], 2)}/{fmt(row['pah_Septum_to_FWmean_tensor_ratio'], 2)} |\n"
                )
            handle.write("\n")

        handle.write("## Reading\n\n")
        handle.write(
            "The PAH mesh is not simply a pressure-shifted version of the healthy mesh: "
            "it has smaller cavities and a larger wall volume.  In the completed PAH "
            "sequence, RV free-wall pressure-strain ranking is very high, while septal "
            "magnitude is best captured by mean or two-sided pressure choices rather "
            "than the transventricular pressure difference.  The healthy sequence is "
            "shorter because high-pressure cases did not complete, so it should be "
            "treated as suggestive rather than conclusive.  The clean thesis use is "
            "therefore a future-work or geometry-sensitivity note, not a main result.\n"
        )


def make_figure(best: list[dict[str, Any]]) -> None:
    rows = [row for row in best if row["strain_suffix"] in {"ll", "ff"}]
    if not rows:
        return
    labels = [f"{row['mesh_key']}\n{row['strain'].replace('geometric ', '')}" for row in rows]
    fw = [float(row["freewall_adjacent_mean_abs_ratio_error"]) for row in rows]
    rv = [float(row["rv_freewall_adjacent_r"]) for row in rows]
    sept = [float(row["best_septum_magnitude_log_error"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
    x = np.arange(len(rows))
    colors = ["#4C78A8" if row["mesh_key"] == "healthy" else "#E15759" for row in rows]
    axes[0].bar(x - 0.18, fw, width=0.36, color=colors, alpha=0.9, label="FW LV/RV")
    axes[0].bar(x + 0.18, sept, width=0.36, color=colors, alpha=0.45, label="best septum")
    axes[0].set_ylabel("error")
    axes[0].set_title("Magnitude preservation")
    axes[0].legend(framealpha=0.95)
    axes[1].bar(x, rv, width=0.55, color=colors)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("correlation")
    axes[1].set_title("RV free-wall ranking")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(OUT / "fig_patient_geometry_proxy_diagnostic.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "fig_patient_geometry_proxy_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_manifests()
    selected_rows, statuses = select_completed_rows(manifest_rows)
    cases = [load_case(row) for row in selected_rows]

    geometry = geometry_rows()
    freewall_rows: list[dict[str, Any]] = []
    septum_rows: list[dict[str, Any]] = []
    for mesh_key in sorted({row["mesh_key"] for row in cases}):
        fw, sept = summarize_mesh(cases, mesh_key)
        freewall_rows += fw
        septum_rows += sept
    best = best_rows(freewall_rows, septum_rows)
    contrast = contrast_rows(cases)

    write_csv(OUT / "patient_mesh_geometry.csv", geometry)
    write_csv(OUT / "patient_mesh_manifest_status.csv", statuses)
    write_csv(OUT / "patient_geometry_case_values.csv", cases)
    write_csv(OUT / "patient_geometry_freewall_summary.csv", freewall_rows)
    write_csv(OUT / "patient_geometry_septum_summary.csv", septum_rows)
    write_csv(OUT / "patient_geometry_best_summary.csv", best)
    write_csv(OUT / "patient_geometry_same_case_contrast.csv", contrast)
    write_markdown(geometry, statuses, cases, best, contrast)
    make_figure(best)

    print("Selected completed cases:")
    for mesh_key in sorted({row["mesh_key"] for row in cases}):
        mesh_cases = [row["case"] for row in sorted(cases, key=lambda row: row["case_index"]) if row["mesh_key"] == mesh_key]
        print(f"  {mesh_key}: {len(mesh_cases)}/8 {', '.join(mesh_cases)}")

    print("\nBest proxy diagnostics:")
    for row in best:
        mag_text = (
            f"{row['best_septum_magnitude_pressure']} "
            f"{float(row['best_septum_magnitude_log_error']):.3f}"
            if np.isfinite(float(row["best_septum_magnitude_log_error"]))
            else f"raw {row['best_septum_raw_magnitude_pressure']} "
            f"{float(row['best_septum_raw_magnitude_error']):.3f}"
        )
        print(
            f"{row['mesh_key']:<8} {row['strain']:<24} n={row['n']} "
            f"FW err={float(row['freewall_adjacent_mean_abs_ratio_error']):.3f} "
            f"RV r={float(row['rv_freewall_adjacent_r']):+.3f} "
            f"sept mag={mag_text}"
        )
    print(f"\nWrote {OUT / 'patient_geometry_exploratory_summary.md'}")


if __name__ == "__main__":
    main()
