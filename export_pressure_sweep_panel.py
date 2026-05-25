#!/usr/bin/env python3
"""Export a side-by-side ParaView panel for the capped RV-pressure sweep.

This is a figure-making export. It writes one translated mesh per selected
case, all merged into a single ParaView-readable scene. The geometry is sampled
at peak RV pressure in the final beat, while accumulated work/proxy fields come
from the final-beat per-cell postprocess.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import adios4dolfinx
import dolfinx
import dolfinx.fem
import dolfinx.plot
import numpy as np
import pyvista as pv
from mpi4py import MPI


REPO = Path(__file__).resolve().parent
DEFAULT_MANIFEST = REPO / "results/analysis/capped_shared_l5_sweep_20260510_141015/capped_shared_l5_cases.tsv"
DEFAULT_OUT = REPO / "paraview_exports/rv_pressure_work_panel"
KPA = 1e-3


@dataclass(frozen=True)
class CaseRow:
    case: str
    result_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["sPAP22", "sPAP35", "sPAP50", "sPAP65", "sPAP80", "sPAP95"],
        help="Cases to place left-to-right in the panel.",
    )
    parser.add_argument("--spacing-mm", type=float, default=95.0, help="Side-by-side spacing in mm.")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="y", help="Axis used for panel offsets.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args()


def read_manifest(path: Path, selected: list[str]) -> list[CaseRow]:
    by_case: dict[str, CaseRow] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            case = row["case"]
            result_dir = Path(row["result_dir"])
            if case in selected:
                by_case[case] = CaseRow(case=case, result_dir=result_dir)
    missing = [case for case in selected if case not in by_case]
    if missing:
        raise RuntimeError(f"Missing selected cases in manifest {path}: {missing}")
    return [by_case[case] for case in selected]


def write_pvd(path: Path, rel_file: str) -> None:
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
                "  <Collection>",
                f'    <DataSet timestep="0" group="" part="0" file="{rel_file}"/>',
                "  </Collection>",
                "</VTKFile>",
            ]
        )
        + "\n"
    )


def case_number(case: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", case)
    return float(match.group(1)) if match else float("nan")


def density_from_cell_integral(pc: np.lib.npyio.NpzFile, values: str | np.ndarray) -> np.ndarray:
    arr = pc[values] if isinstance(values, str) else np.asarray(values, dtype=float)
    return -np.asarray(arr, dtype=float) / np.maximum(np.asarray(pc["cell_volumes"], dtype=float), 1e-30) * KPA


def add_density_fields(grid: pv.UnstructuredGrid, pc: np.lib.npyio.NpzFile) -> list[str]:
    smooth_names: list[str] = []
    field_map: dict[str, np.ndarray] = {
        "W_total_density_kPa": density_from_cell_integral(pc, "w_total"),
        "W_fiber_density_kPa": density_from_cell_integral(pc, "w_ff"),
        "proxy_PLV_ll_density_kPa": density_from_cell_integral(pc, "proxy_PLV_ll"),
        "proxy_PRV_ll_density_kPa": density_from_cell_integral(pc, "proxy_PRV_ll"),
        "proxy_Trans_ll_density_kPa": density_from_cell_integral(pc, "proxy_Trans_ll"),
        "proxy_Mean_ll_density_kPa": density_from_cell_integral(
            pc, 0.5 * (np.asarray(pc["proxy_PLV_ll"]) + np.asarray(pc["proxy_PRV_ll"]))
        ),
        "proxy_PLV_ff_density_kPa": density_from_cell_integral(pc, "proxy_PLV_ff"),
        "proxy_PRV_ff_density_kPa": density_from_cell_integral(pc, "proxy_PRV_ff"),
        "proxy_Trans_ff_density_kPa": density_from_cell_integral(pc, "proxy_Trans_ff"),
        "proxy_Mean_ff_density_kPa": density_from_cell_integral(
            pc, 0.5 * (np.asarray(pc["proxy_PLV_ff"]) + np.asarray(pc["proxy_PRV_ff"]))
        ),
    }
    for name, values in field_map.items():
        grid.cell_data[name] = values
        grid.cell_data[f"{name}_pos"] = np.maximum(values, 0.0)
        grid.cell_data[f"{name}_abs"] = np.abs(values)
        smooth_names += [f"{name}_pos", f"{name}_abs"]

    return smooth_names


def add_static_cell_fields(grid: pv.UnstructuredGrid, pc: np.lib.npyio.NpzFile) -> None:
    tags = np.asarray(pc["region_tags"], dtype=float)
    grid.cell_data["region_tags"] = tags
    grid.cell_data["is_LV"] = (tags == 1).astype(float)
    grid.cell_data["is_RV"] = (tags == 2).astype(float)
    grid.cell_data["is_Septum"] = (tags == 3).astype(float)
    grid.cell_data["cell_volume_m3"] = np.asarray(pc["cell_volumes"], dtype=float)
    for name in ["tau", "lv_rv_scalar", "is_geometric_septum", "is_ldrb_septum", "study_region", "envelope", "touches_epi"]:
        if name in pc.files:
            grid.cell_data[name] = np.asarray(pc[name], dtype=float)
    for name in ["d_lv", "d_rv", "d_epi", "d_sum", "entry_t"]:
        if name in pc.files:
            grid.cell_data[f"{name}_mm"] = np.asarray(pc[name], dtype=float) * 1000.0


def smooth_cell_fields(grid: pv.UnstructuredGrid, names: list[str]) -> None:
    tmp = pv.UnstructuredGrid(grid.cells, grid.celltypes, grid.points)
    for name in names:
        if name in grid.cell_data:
            tmp.cell_data[name] = grid.cell_data[name]
    if not tmp.cell_data:
        return
    smoothed = tmp.cell_data_to_point_data(pass_cell_data=False)
    for name in names:
        if name in smoothed.point_data:
            grid.point_data[f"{name}_viz"] = smoothed.point_data[name]


def final_beat_indices(timestamps: np.ndarray, bpm: float) -> np.ndarray:
    cycle_length = 60.0 / bpm
    start = float(timestamps[-1]) - cycle_length
    return np.where((timestamps >= start - 1e-10) & (timestamps <= float(timestamps[-1]) + 1e-10))[0]


def read_case_grid(case: CaseRow, panel_index: int, offset: np.ndarray) -> tuple[pv.UnstructuredGrid, dict[str, float | str]]:
    comm = MPI.COMM_SELF
    result_dir = case.result_dir.resolve()
    solver_dir = result_dir / "solver"
    checkpoint = solver_dir / "checkpoint.bp"
    pc_path = result_dir / "per_cell_data.npz"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not pc_path.exists():
        raise FileNotFoundError(pc_path)

    sim_params = json.loads((result_dir / "simulation_params.json").read_text())
    pressures = np.load(solver_dir / "solver_cavity_pressure_mmHg.npy")
    timestamps = np.asarray(adios4dolfinx.read_timestamps(checkpoint, comm, "displacement"), dtype=float)
    beat = final_beat_indices(timestamps, float(sim_params["BPM"]))
    if len(beat) == 0:
        raise RuntimeError(f"No final-beat timestamps for {case.case}")
    peak_local = int(np.argmax(pressures[beat, 1]))
    peak_idx = int(beat[peak_local])
    ed_idx = int(beat[0])

    mesh = adios4dolfinx.read_mesh(checkpoint, comm)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    topology, cell_types, points_ref = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)
    cell_types = np.full_like(cell_types, pv.CellType.TETRA, dtype=np.uint8)

    V2 = dolfinx.fem.functionspace(mesh, ("P", 2, (3,)))
    V1 = dolfinx.fem.functionspace(mesh, ("P", 1, (3,)))
    u2 = dolfinx.fem.Function(V2, name="displacement")
    u1 = dolfinx.fem.Function(V1, name="u_p1")

    adios4dolfinx.read_function(checkpoint, u2, time=float(timestamps[ed_idx]), name="displacement")
    u1.interpolate(u2)
    u_ed = u1.x.array.reshape((-1, 3)).copy()

    adios4dolfinx.read_function(checkpoint, u2, time=float(timestamps[peak_idx]), name="displacement")
    u1.interpolate(u2)
    u_peak = u1.x.array.reshape((-1, 3)).copy()
    u_from_ed = u_peak - u_ed

    grid = pv.UnstructuredGrid(topology, cell_types, points_ref.copy() + u_peak + offset)
    grid.point_data["u_m"] = u_peak
    grid.point_data["u_from_ED_m"] = u_from_ed
    grid.point_data["u_mag_mm"] = np.linalg.norm(u_peak, axis=1) * 1000.0
    grid.point_data["u_mag_from_ED_mm"] = np.linalg.norm(u_from_ed, axis=1) * 1000.0

    pc = np.load(pc_path, allow_pickle=True)
    try:
        add_static_cell_fields(grid, pc)
        smooth_names = add_density_fields(grid, pc)
    finally:
        pc.close()

    rvsp = float(np.nanmax(pressures[beat, 1]))
    lvsp = float(np.nanmax(pressures[beat, 0]))
    p_lv = float(pressures[peak_idx, 0])
    p_rv = float(pressures[peak_idx, 1])
    phase = (float(timestamps[peak_idx]) - float(timestamps[beat[0]])) / max(float(timestamps[beat[-1]] - timestamps[beat[0]]), 1e-12)
    target = case_number(case.case)

    for name, value in {
        "case_index": float(panel_index),
        "case_sPAP_target_mmHg": target,
        "RVSP_mmHg": rvsp,
        "LVSP_mmHg": lvsp,
        "p_LV_at_frame_mmHg": p_lv,
        "p_RV_at_frame_mmHg": p_rv,
        "p_trans_at_frame_mmHg": p_lv - p_rv,
        "beat_phase_at_frame": phase,
        "checkpoint_step_index": float(peak_idx),
    }.items():
        grid.cell_data[name] = np.full(grid.n_cells, value, dtype=float)
        grid.point_data[name] = np.full(grid.n_points, value, dtype=float)

    smooth_cell_fields(grid, smooth_names)
    meta = {
        "case": case.case,
        "result_dir": str(result_dir),
        "panel_index": panel_index,
        "offset_m": " ".join(f"{x:.6g}" for x in offset),
        "RVSP_mmHg": rvsp,
        "LVSP_mmHg": lvsp,
        "p_LV_at_frame_mmHg": p_lv,
        "p_RV_at_frame_mmHg": p_rv,
        "p_trans_at_frame_mmHg": p_lv - p_rv,
        "beat_phase_at_frame": phase,
        "checkpoint_step_index": peak_idx,
        "time_s": float(timestamps[peak_idx]),
    }
    return grid, meta


def write_readme(out_dir: Path, cases: list[dict[str, float | str]]) -> None:
    lines = [
        "# RV Pressure Work Panel Export",
        "",
        "Open `volume_panel.pvd` for clipping/slicing or `surface_panel.pvd` for a quick outer-surface view.",
        "The meshes are translated side-by-side. Each case is shown at peak RV pressure in the final beat.",
        "Accumulated work-density fields are final-beat per-cell values from `per_cell_data.npz`.",
        "",
        "Recommended coloring fields:",
        "",
        "- `W_total_density_kPa_pos_viz`: smooth positive accumulated total stress-strain work density.",
        "- `W_fiber_density_kPa_pos_viz`: smooth positive accumulated fibre work density.",
        "- `u_mag_from_ED_mm`: displacement magnitude from final-beat ED to the peak-RV-pressure frame.",
        "- `RVSP_mmHg`: repeated scalar that documents achieved RV systolic pressure per case.",
        "",
        "Use the unsmoothed cell fields without `_viz` for traceability; the `_viz` fields are only for screenshots.",
        "",
        "## Cases",
        "",
        "| Case | RVSP (mmHg) | LVSP (mmHg) | Frame phase | Checkpoint step |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in cases:
        lines.append(
            f"| {row['case']} | {float(row['RVSP_mmHg']):.1f} | {float(row['LVSP_mmHg']):.1f} | "
            f"{float(row['beat_phase_at_frame']):.3f} | {int(row['checkpoint_step_index'])} |"
        )
    (out_dir / "README_RV_PRESSURE_WORK_PANEL.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"{out_dir} exists; pass --force")
        import shutil

        shutil.rmtree(out_dir)
    volume_dir = out_dir / "volume"
    surface_dir = out_dir / "surface"
    volume_dir.mkdir(parents=True, exist_ok=True)
    surface_dir.mkdir(parents=True, exist_ok=True)

    axis_index = {"x": 0, "y": 1, "z": 2}[args.axis]
    spacing = args.spacing_mm / 1000.0
    cases = read_manifest(args.manifest.resolve(), args.cases)

    grids: list[pv.UnstructuredGrid] = []
    rows: list[dict[str, float | str]] = []
    center = 0.5 * (len(cases) - 1)
    for i, case in enumerate(cases):
        offset = np.zeros(3, dtype=float)
        offset[axis_index] = (i - center) * spacing
        print(f"Exporting {case.case} at offset {offset}")
        grid, meta = read_case_grid(case, i, offset)
        grids.append(grid)
        rows.append(meta)

    merged = grids[0]
    for grid in grids[1:]:
        merged = merged.merge(grid, merge_points=False)

    volume_path = volume_dir / "rv_pressure_work_panel_peak_rv_pressure.vtu"
    surface_path = surface_dir / "rv_pressure_work_panel_peak_rv_pressure_surface.vtp"
    merged.save(volume_path)
    merged.extract_surface().save(surface_path)

    write_pvd(out_dir / "volume_panel.pvd", volume_path.relative_to(out_dir).as_posix())
    write_pvd(out_dir / "surface_panel.pvd", surface_path.relative_to(out_dir).as_posix())

    with (out_dir / "CASE_MANIFEST.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_readme(out_dir, rows)
    print(f"Wrote {out_dir / 'volume_panel.pvd'}")
    print(f"Wrote {out_dir / 'surface_panel.pvd'}")


if __name__ == "__main__":
    main()
