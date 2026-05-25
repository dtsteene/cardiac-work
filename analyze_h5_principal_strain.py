#!/usr/bin/env python3
"""Replay h=5 cases and compare clinical directions with principal strain.

This is an exploratory strain-only diagnostic layered on top of the existing
h=5 pressure-strain direction analysis.  It asks two questions:

1. If the strain tensor is collapsed to one scalar, what does the most
   compressive principal strain do as a theoretical upper-bound direction?
2. How closely do longitudinal, fibre, radial, and circumferential directions
   align with the dominant principal deformation directions?

The replay uses the saved displacement checkpoints and computes a DG0
cell-centre Green-Lagrange strain tensor.  By default the replay samples every
10 saved time steps to keep this exploratory diagnostic cheap.  Clinical strain
is zeroed at the start of the final beat by subtracting the first tensor in
that beat before principal strains are evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/dtsteene/D1/cardiac-work")
SIM_ROOT = ROOT / "results" / "sims"
MANIFEST = Path(
    os.environ.get(
        "H5_PRINCIPAL_MANIFEST",
        ROOT / "results" / "analysis" / "h5_sweep_submission" / "h5_corrected_sweep_cases.tsv",
    )
)
STRAIN_BEST = Path(
    os.environ.get(
        "H5_STRAIN_BEST",
        ROOT / "results" / "analysis" / "h5_strain_directions" / "h5_strain_direction_best.csv",
    )
)
OUT = Path(os.environ.get("H5_PRINCIPAL_OUT", ROOT / "results" / "analysis" / "h5_principal_strain"))
CACHE = OUT / "cache"
KPA = 1e-3
MMHG_TO_PA = 133.322

PRINCIPAL_STRAINS = {
    "pmin": "principal shortening",
    "pabs": "dominant absolute principal",
}
PRESSURES = ["PLV", "PRV", "Trans", "Mean", "NearestSide", "TauWeighted"]
ALIGNMENT_CANDIDATES = ["longitudinal", "fibre", "radial", "circumferential"]
REGIONS = ["LV", "RV", "Septum"]


def ensure_fem_imports() -> None:
    """Import the FEniCSx stack only for checkpoint replay.

    This keeps the login-node guard useful even in shells where the RV conda
    environment is not active, and allows summary-only table regeneration from
    cached NPZ files without loading the FEM stack.
    """

    if "dolfinx" in globals():
        return

    import adios4dolfinx as _adios4dolfinx
    import basix as _basix
    import basix.ufl  # noqa: F401 - attaches the ufl submodule to basix
    import dolfinx as _dolfinx
    import ufl as _ufl
    from mpi4py import MPI as _MPI

    globals()["adios4dolfinx"] = _adios4dolfinx
    globals()["basix"] = _basix
    globals()["dolfinx"] = _dolfinx
    globals()["ufl"] = _ufl
    globals()["MPI"] = _MPI


def cache_path(case: str, stride: int) -> Path:
    return CACHE / f"{case}_principal_replay_stride{stride}.npz"


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def find_run(job_id: str) -> Path:
    matches = sorted(SIM_ROOT.glob(f"*/UKB_6beats_run_{job_id}"))
    if not matches:
        raise FileNotFoundError(job_id)
    return matches[-1]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def unit_rows(values: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    out = np.zeros_like(values, dtype=float)
    good = norms > eps
    out[good] = values[good] / norms[good, None]
    return out


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return float("nan")
    return float(np.average(values[good], weights=weights[good]))


def corr(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(pearsonr(x_arr, y_arr)[0])


def ratio_error(values: list[float], targets: list[float]) -> tuple[float, float, int]:
    raw = [abs(v - t) for v, t in zip(values, targets)]
    log = [
        abs(np.log(v / t))
        for v, t in zip(values, targets)
        if np.isfinite(v) and np.isfinite(t) and v > 0 and t > 0
    ]
    return float(np.mean(raw)), float(np.mean(log)) if log else float("nan"), len(log)


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


def pressure_arrays(plv: np.ndarray, prv: np.ndarray, pc: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    # Canonical convention for pressure choices: tau=0 on the LV side and
    # tau=1 on the RV side. The saved Laplace scalar has the opposite orientation.
    tau = 1.0 - pc["lv_rv_scalar"] if "lv_rv_scalar" in pc.files else pc["tau"]
    trans = plv - prv
    return {
        "PLV": plv,
        "PRV": prv,
        "Trans": trans,
        "Mean": 0.5 * (plv + prv),
        "NearestSide": np.where(tau < 0.5, plv, prv),
        "TauWeighted": (1.0 - tau) * plv + tau * prv,
    }


def avg_unit_vector(field: dolfinx.fem.Function, mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    """Volume-average a quadrature vector field after normalizing it pointwise."""

    v_space = dolfinx.fem.functionspace(mesh, ("DG", 0))
    v = ufl.TestFunction(v_space)
    dx_q = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})
    volume_form = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * v * dx_q)
    volumes = dolfinx.fem.assemble_vector(volume_form).array.copy()

    unit = field / ufl.sqrt(ufl.inner(field, field) + 1e-30)
    comps = []
    for i in range(3):
        form = dolfinx.fem.form(unit[i] * v * dx_q)
        comps.append(dolfinx.fem.assemble_vector(form).array / volumes)
    return unit_rows(np.vstack(comps).T)


def load_reference_directions(
    checkpoint_path: Path,
    mesh: dolfinx.mesh.Mesh,
    pc: np.lib.npyio.NpzFile,
) -> dict[str, np.ndarray]:
    q_el = basix.ufl.quadrature_element(mesh.topology.cell_name(), value_shape=(3,), degree=6)
    q_vec = dolfinx.fem.functionspace(mesh, q_el)

    f0 = dolfinx.fem.Function(q_vec)
    l0 = dolfinx.fem.Function(q_vec)
    adios4dolfinx.read_function(checkpoint_path, f0, time=0.0, name="f0")
    adios4dolfinx.read_function(checkpoint_path, l0, time=0.0, name="l0")

    dirs = {
        "fibre": avg_unit_vector(f0, mesh),
        "longitudinal": avg_unit_vector(l0, mesh),
    }

    if "radial_endo_to_epi" in pc.files:
        radial_raw = unit_rows(np.asarray(pc["radial_endo_to_epi"], dtype=float))
        longitudinal = dirs["longitudinal"]
        longitudinal = longitudinal - np.sum(longitudinal * radial_raw, axis=1)[:, None] * radial_raw
        longitudinal = unit_rows(longitudinal)
        dirs["longitudinal"] = longitudinal
        radial = radial_raw - np.sum(radial_raw * longitudinal, axis=1)[:, None] * longitudinal
        radial = unit_rows(radial)
        circ = unit_rows(np.cross(longitudinal, radial))
        dirs["radial"] = radial
        dirs["circumferential"] = circ

    return dirs


def replay_case(item: dict[str, str], stride: int, force: bool = False) -> Path:
    ensure_fem_imports()

    case = item["case"]
    run = find_run(item["job_id"])
    out_path = cache_path(case, stride)
    if out_path.exists() and not force:
        return out_path

    logger = logging.getLogger("principal_replay")
    logger.info("Processing %s (%s), stride=%d", case, run, stride)

    sim_params = json.loads((run / "simulation_params.json").read_text())
    pressures = np.load(run / "solver" / "solver_cavity_pressure_mmHg.npy")
    pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
    volumes = np.asarray(pc["cell_volumes"], dtype=float)
    n_cells = len(volumes)

    checkpoint_path = run / "solver" / "checkpoint.bp"
    mesh = adios4dolfinx.read_mesh(checkpoint_path, MPI.COMM_WORLD)
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This exploratory script currently expects a single MPI rank.")
    if mesh.topology.index_map(3).size_local != n_cells:
        raise RuntimeError(f"{case}: checkpoint cell count does not match per_cell_data.npz")

    dirs = load_reference_directions(checkpoint_path, mesh, pc)

    u_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
    u = dolfinx.fem.Function(u_space)
    identity = ufl.Identity(3)
    deformation_gradient = ufl.grad(u) + identity
    strain = 0.5 * (deformation_gradient.T * deformation_gradient - identity)
    e_space = dolfinx.fem.functionspace(mesh, ("DG", 0, (3, 3)))
    e_dg0 = dolfinx.fem.Function(e_space)
    e_expr = dolfinx.fem.Expression(strain, e_space.element.interpolation_points)

    timestamps = adios4dolfinx.read_timestamps(checkpoint_path, MPI.COMM_WORLD, "displacement")
    hr_hz = sim_params["BPM"] / 60.0
    cycle_length = 1.0 / hr_hz
    steps_per_beat = int(round(cycle_length / sim_params["dt"]))
    n_beats = len(timestamps) // steps_per_beat
    start_step = (n_beats - 1) * steps_per_beat
    end_step = n_beats * steps_per_beat
    sample_steps = list(range(start_step, end_step, stride))
    if sample_steps[-1] != end_step - 1:
        sample_steps.append(end_step - 1)

    proxy_plv_pmin = np.zeros(n_cells)
    proxy_prv_pmin = np.zeros(n_cells)
    proxy_plv_pabs = np.zeros(n_cells)
    proxy_prv_pabs = np.zeros(n_cells)
    mean_lambda_min_trace = []
    mean_lambda_abs_trace = []
    beat_times = []

    e0 = None
    prev_lam_min = None
    prev_lam_abs = None
    p_lv_prev = None
    p_rv_prev = None

    best_min_mean = np.inf
    best_abs_mean = -np.inf
    ref_min_step = start_step
    ref_abs_step = start_step
    ref_min_time = float(timestamps[start_step])
    ref_abs_time = float(timestamps[start_step])
    ref_lam_min = np.zeros(n_cells)
    ref_lam_abs = np.zeros(n_cells)
    ref_vec_min = np.zeros((n_cells, 3))
    ref_vec_abs = np.zeros((n_cells, 3))

    for sample_i, step in enumerate(sample_steps):
        time = float(timestamps[step])
        adios4dolfinx.read_function(checkpoint_path, u, time=time, name="displacement")
        e_dg0.interpolate(e_expr)
        e_cur = e_dg0.x.array.reshape(-1, 3, 3)[:n_cells]
        e_cur = 0.5 * (e_cur + np.transpose(e_cur, (0, 2, 1)))
        if e0 is None:
            e0 = e_cur.copy()
        e_rel = e_cur - e0

        eigvals, eigvecs = np.linalg.eigh(e_rel)
        lam_min = eigvals[:, 0]
        vec_min = eigvecs[:, :, 0]
        abs_idx = np.argmax(np.abs(eigvals), axis=1)
        row_idx = np.arange(n_cells)
        lam_abs = eigvals[row_idx, abs_idx]
        vec_abs = eigvecs[row_idx, :, abs_idx]

        mean_lam_min = weighted_mean(lam_min, volumes)
        mean_lam_abs = weighted_mean(np.abs(lam_abs), volumes)
        mean_lambda_min_trace.append(mean_lam_min)
        mean_lambda_abs_trace.append(mean_lam_abs)
        beat_times.append(time - float(timestamps[start_step]))

        if mean_lam_min < best_min_mean:
            best_min_mean = mean_lam_min
            ref_min_step = step
            ref_min_time = time
            ref_lam_min = lam_min.copy()
            ref_vec_min = vec_min.copy()
        if mean_lam_abs > best_abs_mean:
            best_abs_mean = mean_lam_abs
            ref_abs_step = step
            ref_abs_time = time
            ref_lam_abs = lam_abs.copy()
            ref_vec_abs = vec_abs.copy()

        p_lv = float(pressures[step, 0]) * MMHG_TO_PA
        p_rv = float(pressures[step, 1]) * MMHG_TO_PA
        if prev_lam_min is not None:
            p_lv_avg = 0.5 * (p_lv + p_lv_prev)
            p_rv_avg = 0.5 * (p_rv + p_rv_prev)
            d_lam_min = lam_min - prev_lam_min
            d_lam_abs = lam_abs - prev_lam_abs
            proxy_plv_pmin += p_lv_avg * d_lam_min * volumes
            proxy_prv_pmin += p_rv_avg * d_lam_min * volumes
            proxy_plv_pabs += p_lv_avg * d_lam_abs * volumes
            proxy_prv_pabs += p_rv_avg * d_lam_abs * volumes

        prev_lam_min = lam_min
        prev_lam_abs = lam_abs
        p_lv_prev = p_lv
        p_rv_prev = p_rv

        if sample_i % 50 == 0 or sample_i == len(sample_steps) - 1:
            logger.info("  %s sample %d/%d (step %d)", case, sample_i + 1, len(sample_steps), step)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_items = {
        "case": np.array(case),
        "job_id": np.array(item["job_id"]),
        "run_dir": np.array(str(run)),
        "start_step": np.array(start_step),
        "end_step": np.array(end_step),
        "stride": np.array(stride),
        "sample_steps": np.asarray(sample_steps),
        "ref_min_step": np.array(ref_min_step),
        "ref_abs_step": np.array(ref_abs_step),
        "ref_min_time": np.array(ref_min_time),
        "ref_abs_time": np.array(ref_abs_time),
        "beat_times": np.asarray(beat_times),
        "mean_lambda_min_trace": np.asarray(mean_lambda_min_trace),
        "mean_lambda_abs_trace": np.asarray(mean_lambda_abs_trace),
        "lambda_min_ref": ref_lam_min,
        "lambda_abs_ref": ref_lam_abs,
        "vec_min_ref": ref_vec_min,
        "vec_abs_ref": ref_vec_abs,
        "proxy_PLV_pmin": proxy_plv_pmin,
        "proxy_PRV_pmin": proxy_prv_pmin,
        "proxy_Trans_pmin": proxy_plv_pmin - proxy_prv_pmin,
        "proxy_PLV_pabs": proxy_plv_pabs,
        "proxy_PRV_pabs": proxy_prv_pabs,
        "proxy_Trans_pabs": proxy_plv_pabs - proxy_prv_pabs,
    }
    for name, values in dirs.items():
        save_items[f"dir_{name}"] = values
    np.savez(out_path, **save_items)
    logger.info("Wrote %s", out_path)
    return out_path


def case_values(manifest: list[dict[str, str]], stride: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in manifest:
        case = item["case"]
        run = find_run(item["job_id"])
        pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
        principal = np.load(cache_path(case, stride), allow_pickle=True)
        region_masks = masks(pc)

        row: dict[str, object] = {
            "case": case,
            "job_id": item["job_id"],
            "run_dir": str(run),
            "stride": stride,
            "ref_min_time_s": float(principal["ref_min_time"]),
            "ref_abs_time_s": float(principal["ref_abs_time"]),
            "has_radial_circ": "dir_radial" in principal.files and "dir_circumferential" in principal.files,
        }
        for region, mask in region_masks.items():
            row[f"{region}_volume_mL"] = float(pc["cell_volumes"][mask].sum() * 1e6)
            row[f"{region}_W_total_kPa"] = density(pc, mask, "w_total")
            row[f"{region}_W_ff_kPa"] = density(pc, mask, "w_ff")
            for suffix in PRINCIPAL_STRAINS:
                arrays = pressure_arrays(
                    np.asarray(principal[f"proxy_PLV_{suffix}"]),
                    np.asarray(principal[f"proxy_PRV_{suffix}"]),
                    pc,
                )
                for pressure, values in arrays.items():
                    row[f"{region}_{pressure}_{suffix}_kPa"] = density(pc, mask, values)

        row["FW_tensor_LV_RV_ratio"] = float(row["LV_W_total_kPa"]) / float(row["RV_W_total_kPa"])
        row["Septum_to_FWmean_tensor_ratio"] = float(row["Septum_W_total_kPa"]) / (
            0.5 * (float(row["LV_W_total_kPa"]) + float(row["RV_W_total_kPa"]))
        )
        rows.append(row)
    return rows


def summarize_principal(rows: list[dict[str, object]], cohort: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for suffix, strain_label in PRINCIPAL_STRAINS.items():
        fw_pred = [
            float(r[f"LV_PLV_{suffix}_kPa"]) / float(r[f"RV_PRV_{suffix}_kPa"])
            for r in rows
        ]
        fw_target = [float(r["FW_tensor_LV_RV_ratio"]) for r in rows]
        fw_raw, fw_log, fw_log_n = ratio_error(fw_pred, fw_target)

        rv_proxy = [float(r[f"RV_PRV_{suffix}_kPa"]) for r in rows]
        rv_target = [float(r["RV_W_total_kPa"]) for r in rows]
        sept_tensor = [float(r["Septum_W_total_kPa"]) for r in rows]

        for pressure in PRESSURES:
            sept_proxy = [float(r[f"Septum_{pressure}_{suffix}_kPa"]) for r in rows]
            fw_mean_proxy = [
                0.5 * (float(r[f"LV_PLV_{suffix}_kPa"]) + float(r[f"RV_PRV_{suffix}_kPa"]))
                for r in rows
            ]
            sept_ratio_pred = [s / f if abs(f) > 1e-30 else float("nan") for s, f in zip(sept_proxy, fw_mean_proxy)]
            sept_ratio_target = [float(r["Septum_to_FWmean_tensor_ratio"]) for r in rows]
            sept_raw, sept_log, sept_log_n = ratio_error(sept_ratio_pred, sept_ratio_target)
            out.append(
                {
                    "cohort": cohort,
                    "n": len(rows),
                    "strain_suffix": suffix,
                    "strain": strain_label,
                    "pressure": pressure,
                    "freewall_adjacent_mean_abs_ratio_error": fw_raw,
                    "freewall_adjacent_mean_abs_log_ratio_error": fw_log,
                    "freewall_adjacent_log_n": fw_log_n,
                    "freewall_ratio_r": corr(fw_pred, fw_target),
                    "rv_freewall_adjacent_r": corr(rv_proxy, rv_target),
                    "septum_r_vs_tensor": corr(sept_proxy, sept_tensor),
                    "septum_mean_abs_ratio_error": sept_raw,
                    "septum_mean_abs_log_ratio_error": sept_log,
                    "septum_log_n": sept_log_n,
                    "septum_proxy_mean_kPa": float(np.mean(sept_proxy)),
                    "septum_proxy_min_kPa": float(np.min(sept_proxy)),
                    "septum_proxy_max_kPa": float(np.max(sept_proxy)),
                }
            )
    return out


def best_by_strain(summary: list[dict[str, object]], cohort: str) -> list[dict[str, object]]:
    rows = [r for r in summary if r["cohort"] == cohort]
    out: list[dict[str, object]] = []
    for suffix, strain_label in PRINCIPAL_STRAINS.items():
        matches = [r for r in rows if r["strain_suffix"] == suffix]
        if not matches:
            continue
        best_corr = max(
            matches,
            key=lambda r: -np.inf if not np.isfinite(float(r["septum_r_vs_tensor"])) else float(r["septum_r_vs_tensor"]),
        )
        valid_mag = [r for r in matches if np.isfinite(float(r["septum_mean_abs_log_ratio_error"]))]
        best_mag = min(valid_mag, key=lambda r: float(r["septum_mean_abs_log_ratio_error"])) if valid_mag else None
        best_raw_mag = min(matches, key=lambda r: float(r["septum_mean_abs_ratio_error"]))
        out.append(
            {
                "cohort": cohort,
                "n": matches[0]["n"],
                "strain_suffix": suffix,
                "strain": strain_label,
                "freewall_adjacent_mean_abs_ratio_error": matches[0]["freewall_adjacent_mean_abs_ratio_error"],
                "freewall_ratio_r": matches[0]["freewall_ratio_r"],
                "rv_freewall_adjacent_r": matches[0]["rv_freewall_adjacent_r"],
                "best_septum_correlation_pressure": best_corr["pressure"],
                "best_septum_correlation_r": best_corr["septum_r_vs_tensor"],
                "best_septum_magnitude_pressure": best_mag["pressure"] if best_mag else "",
                "best_septum_magnitude_log_error": best_mag["septum_mean_abs_log_ratio_error"] if best_mag else float("nan"),
                "best_septum_magnitude_log_n": best_mag["septum_log_n"] if best_mag else 0,
                "best_septum_raw_magnitude_pressure": best_raw_mag["pressure"],
                "best_septum_raw_magnitude_error": best_raw_mag["septum_mean_abs_ratio_error"],
            }
        )
    return out


def alignment_rows(manifest: list[dict[str, str]], stride: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in manifest:
        case = item["case"]
        run = find_run(item["job_id"])
        pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
        principal = np.load(cache_path(case, stride), allow_pickle=True)
        region_masks = masks(pc)
        volumes = np.asarray(pc["cell_volumes"], dtype=float)

        target_dirs = {
            "principal_shortening": np.asarray(principal["vec_min_ref"], dtype=float),
            "dominant_absolute_principal": np.asarray(principal["vec_abs_ref"], dtype=float),
        }
        candidate_dirs = {}
        for candidate in ALIGNMENT_CANDIDATES:
            key = f"dir_{candidate}"
            if key in principal.files:
                candidate_dirs[candidate] = np.asarray(principal[key], dtype=float)

        for target_name, target in target_dirs.items():
            dots_by_candidate = {
                candidate: np.clip(np.abs(np.sum(direction * target, axis=1)), 0.0, 1.0)
                for candidate, direction in candidate_dirs.items()
            }
            candidate_names = list(dots_by_candidate)
            dot_stack = np.vstack([dots_by_candidate[name] for name in candidate_names])
            best_idx = np.argmax(dot_stack, axis=0)
            for region, mask in region_masks.items():
                weights = volumes[mask]
                for i, candidate in enumerate(candidate_names):
                    dots = dots_by_candidate[candidate][mask]
                    angles = np.degrees(np.arccos(dots))
                    rows.append(
                        {
                            "case": case,
                            "job_id": item["job_id"],
                            "target": target_name,
                            "region": region,
                            "candidate": candidate,
                            "n_cells": int(mask.sum()),
                            "volume_mL": float(weights.sum() * 1e6),
                            "mean_abs_cos": weighted_mean(dots, weights),
                            "mean_angle_deg": weighted_mean(angles, weights),
                            "volume_fraction_best_candidate": float(weights[best_idx[mask] == i].sum() / weights.sum()),
                        }
                    )
    return rows


def summarize_alignment(rows: list[dict[str, object]], cohort: str) -> list[dict[str, object]]:
    filtered = rows
    if cohort == "common13_all_directions":
        complete_cases = {
            r["case"]
            for r in rows
            if r["candidate"] == "circumferential"
        }
        filtered = [r for r in rows if r["case"] in complete_cases]

    out: list[dict[str, object]] = []
    keys = sorted({(r["target"], r["region"], r["candidate"]) for r in filtered})
    for target, region, candidate in keys:
        matches = [
            r for r in filtered
            if r["target"] == target and r["region"] == region and r["candidate"] == candidate
        ]
        if not matches:
            continue
        out.append(
            {
                "cohort": cohort,
                "n_cases": len({r["case"] for r in matches}),
                "target": target,
                "region": region,
                "candidate": candidate,
                "mean_abs_cos": float(np.mean([float(r["mean_abs_cos"]) for r in matches])),
                "mean_angle_deg": float(np.mean([float(r["mean_angle_deg"]) for r in matches])),
                "mean_volume_fraction_best_candidate": float(
                    np.mean([float(r["volume_fraction_best_candidate"]) for r in matches])
                ),
            }
        )
    return out


def read_existing_best() -> list[dict[str, object]]:
    if not STRAIN_BEST.exists():
        return []
    with STRAIN_BEST.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_markdown(best: list[dict[str, object]], combined: list[dict[str, object]], alignment: list[dict[str, object]]) -> None:
    md = OUT / "h5_principal_strain_summary.md"
    common_combined = [r for r in combined if r["cohort"] == "common13_all_directions"]
    common_align = [
        r for r in alignment
        if r["cohort"] == "common13_all_directions"
        and r["target"] == "principal_shortening"
        and r["region"] == "Septum"
    ]
    order = {
        "longitudinal": 0,
        "fibre": 1,
        "radial": 2,
        "circumferential": 3,
    }
    common_align = sorted(common_align, key=lambda r: order.get(str(r["candidate"]), 99))

    with md.open("w") as handle:
        handle.write("# H5 Principal-Strain Diagnostic\n\n")
        handle.write(
            "This replay diagnostic zeroes the Green-Lagrange strain tensor at the "
            "start of the final beat, then evaluates principal strains from the "
            "cell-centre tensor through the beat. `principal shortening` is the "
            "most compressive principal strain and is the natural theoretical "
            "single-component strain for a pressure-shortening loop. `dominant "
            "absolute principal` is the signed eigenvalue with largest absolute "
            "magnitude and often represents wall thickening rather than shortening.\n\n"
        )
        handle.write("## Pressure-Strain Performance, Common 13-Case Cohort\n\n")
        handle.write(
            "| strain | free-wall LV/RV error | RV free-wall r | best septal r | best septal magnitude error |\n"
        )
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in common_combined:
            mag = row.get("best_septum_magnitude_log_error", "nan")
            if mag == "" or not np.isfinite(float(mag)):
                mag_text = "n/a"
            else:
                mag_text = f"{row['best_septum_magnitude_pressure']} {float(mag):.3f}"
            handle.write(
                f"| {row['strain']} | "
                f"{float(row['freewall_adjacent_mean_abs_ratio_error']):.3f} | "
                f"{float(row['rv_freewall_adjacent_r']):+.3f} | "
                f"{row['best_septum_correlation_pressure']} {float(row['best_septum_correlation_r']):+.3f} | "
                f"{mag_text} |\n"
            )

        handle.write("\n## Septal Alignment With Principal Shortening\n\n")
        handle.write("| candidate direction | mean angle | mean |dot| | best-aligned volume fraction |\n")
        handle.write("|---|---:|---:|---:|\n")
        for row in common_align:
            handle.write(
                f"| {row['candidate']} | "
                f"{float(row['mean_angle_deg']):.1f} deg | "
                f"{float(row['mean_abs_cos']):.3f} | "
                f"{float(row['mean_volume_fraction_best_candidate']):.2f} |\n"
            )

        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The principal-shortening scalar is a useful upper-bound diagnostic for "
            "the strain-direction reduction. It is not a clinical replacement, "
            "because it requires the local three-dimensional strain tensor and "
            "allows the measured direction to vary cell by cell. In this replay it "
            "does improve the RV free-wall trend and septal ranking relative to "
            "some anatomical directions, but it does not remove the pressure-choice "
            "problem by itself. The dominant-absolute principal strain is even less "
            "suited to a pressure-shortening loop because its largest magnitude can "
            "come from positive wall thickening, which has the opposite loop sign "
            "under the present convention.\n\n"
        )
        handle.write(
            "Directionally, the septal principal-shortening axis is usually closer "
            "to the fibre direction than to the clinical longitudinal axis. Radial "
            "and circumferential directions add geometric information, but neither "
            "is consistently the dominant shortening direction. This supports the "
            "cautious thesis statement: the omitted strain-direction reduction is "
            "real, but switching from longitudinal to one fixed anatomical component "
            "does not create a clean clinical substitute for the tensor strain field.\n"
        )


def make_figures(combined: list[dict[str, object]], alignment_summary: list[dict[str, object]]) -> None:
    common = [r for r in combined if r["cohort"] == "common13_all_directions"]
    labels = [str(r["strain"]).replace("geometric ", "") for r in common]
    fw = [float(r["freewall_adjacent_mean_abs_ratio_error"]) for r in common]
    rv_r = [float(r["rv_freewall_adjacent_r"]) for r in common]
    sept_r = [float(r["best_septum_correlation_r"]) for r in common]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    x = np.arange(len(labels))
    colors = ["#4C78A8", "#E15759", "#F28E2B", "#59A14F", "#7F7F7F", "#B07AA1"]
    axes[0].bar(x, fw, color=colors[: len(labels)])
    axes[0].set_title("Free-wall magnitude")
    axes[0].set_ylabel("LV/RV ratio error")
    axes[1].bar(x, rv_r, color=colors[: len(labels)])
    axes[1].set_title("RV free-wall trend")
    axes[1].set_ylabel("Pearson r")
    axes[1].set_ylim(-1.05, 1.05)
    axes[2].bar(x, sept_r, color=colors[: len(labels)])
    axes[2].set_title("Best septal trend")
    axes[2].set_ylabel("Pearson r")
    axes[2].set_ylim(-1.05, 1.05)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(OUT / "fig_h5_principal_strain_performance.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "fig_h5_principal_strain_performance.pdf", bbox_inches="tight")
    plt.close(fig)

    align = [
        r for r in alignment_summary
        if r["cohort"] == "common13_all_directions"
        and r["target"] == "principal_shortening"
        and r["region"] in REGIONS
    ]
    candidates = ["longitudinal", "fibre", "radial", "circumferential"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True, constrained_layout=True)
    for ax, region in zip(axes, REGIONS):
        vals = []
        for candidate in candidates:
            match = [r for r in align if r["region"] == region and r["candidate"] == candidate]
            vals.append(float(match[0]["mean_angle_deg"]) if match else np.nan)
        ax.bar(np.arange(len(candidates)), vals, color=colors[: len(candidates)])
        ax.set_title(region)
        ax.set_xticks(np.arange(len(candidates)))
        ax.set_xticklabels(["long.", "fibre", "radial", "circ."], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("mean angle to principal shortening (deg)")
    fig.savefig(OUT / "fig_h5_principal_shortening_alignment.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "fig_h5_principal_shortening_alignment.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute cached principal-strain replay files.")
    parser.add_argument("--summary-only", action="store_true", help="Reuse cache and regenerate tables/figures only.")
    parser.add_argument("--cases", nargs="*", default=None, help="Optional case labels to process, e.g. sPAP55.")
    parser.add_argument("--stride", type=int, default=10, help="Replay every Nth checkpoint step; use 1 for full 1 ms replay.")
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help="Allow checkpoint replay outside SLURM. Summary-only mode is always allowed.",
    )
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if not args.summary_only and "SLURM_JOB_ID" not in os.environ and not args.allow_local:
        raise SystemExit(
            "Refusing checkpoint replay outside SLURM. Submit run_h5_principal_strain.sbatch "
            "or pass --allow-local only for a tiny interactive smoke test."
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest()
    if args.cases:
        requested = set(args.cases)
        manifest = [item for item in manifest if item["case"] in requested]
        missing = requested - {item["case"] for item in manifest}
        if missing:
            raise SystemExit(f"Unknown cases: {', '.join(sorted(missing))}")

    if not args.summary_only:
        for item in manifest:
            replay_case(item, stride=args.stride, force=args.force)

    rows = case_values(manifest, stride=args.stride)
    common = [r for r in rows if bool(r["has_radial_circ"])]
    summary = summarize_principal(rows, "available_per_direction")
    summary += summarize_principal(common, "common13_all_directions")
    best = best_by_strain(summary, "available_per_direction")
    best += best_by_strain(summary, "common13_all_directions")

    align_rows = alignment_rows(manifest, stride=args.stride)
    align_summary = summarize_alignment(align_rows, "available_per_direction")
    align_summary += summarize_alignment(align_rows, "common13_all_directions")

    existing = read_existing_best()
    combined = [r for r in existing if r.get("cohort") == "common13_all_directions"]
    combined += [r for r in best if r.get("cohort") == "common13_all_directions"]

    write_csv(OUT / "h5_principal_strain_case_values.csv", rows)
    write_csv(OUT / "h5_principal_strain_summary.csv", summary)
    write_csv(OUT / "h5_principal_strain_best.csv", best)
    write_csv(OUT / "h5_principal_alignment_case_values.csv", align_rows)
    write_csv(OUT / "h5_principal_alignment_summary.csv", align_summary)
    write_csv(OUT / "h5_strain_direction_best_with_principal.csv", combined)
    write_markdown(best, combined, align_summary)
    make_figures(combined, align_summary)

    print(f"Processed {len(rows)} cases; {len(common)} have radial/circumferential directions.")
    print("\nCommon 13-case principal-strain summary:")
    for row in [r for r in best if r["cohort"] == "common13_all_directions"]:
        print(
            f"{row['strain']:<29} "
            f"FW err={float(row['freewall_adjacent_mean_abs_ratio_error']):.3f} "
            f"RV r={float(row['rv_freewall_adjacent_r']):+.3f} "
            f"best sept r={row['best_septum_correlation_pressure']} "
            f"{float(row['best_septum_correlation_r']):+.3f} "
            f"best sept mag={row['best_septum_magnitude_pressure']} "
            f"{float(row['best_septum_magnitude_log_error']):.3f}"
        )
    print(f"\nWrote {OUT / 'h5_principal_strain_summary.md'}")


if __name__ == "__main__":
    main()
