#!/usr/bin/env python3
"""Export static ParaView snapshots for uncapped-vs-capped unloading figures.

The legacy VTX files are useful for playback, but the first frame is not a
reliable comparison point across the old uncapped and new capped pipelines. This
script reads the solver checkpoint directly, selects one end-diastolic frame,
and writes small VTU/VTP files with a single, explicit convention:

    reference_unloaded_mesh: points are the unloaded reference coordinates
    ed_deformed_mesh:        points are reference + u(t_ED)

Both carry the vector field `u_to_ED_m` and scalar `u_to_ED_mag_cm`.
"""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_OUT = REPO / "paraview_exports" / "unloading_cap_comparison_20260511"


@dataclass(frozen=True)
class Case:
    label: str
    severity: int
    cap_label: str
    cap_mmhg: float
    result_dir: Path


DEFAULT_CASES = [
    Case("uncapped_sPAP22_run_1050300", 22, "uncapped", -1.0, REPO / "results/sims/2026-04-26/UKB_6beats_run_1050300"),
    Case("uncapped_sPAP60_run_1050301", 60, "uncapped", -1.0, REPO / "results/sims/2026-04-26/UKB_6beats_run_1050301"),
    Case("uncapped_sPAP95_run_1050302", 95, "uncapped", -1.0, REPO / "results/sims/2026-04-26/UKB_6beats_run_1050302"),
    Case("capped5_sPAP22", 22, "cap5", 5.0, REPO / "results/sims/2026-05-10/capped_shared_l5_20260510_141015/sPAP22"),
    Case("capped5_sPAP60", 60, "cap5", 5.0, REPO / "results/sims/2026-05-10/capped_shared_l5_20260510_141015/sPAP60"),
    Case("capped5_sPAP95", 95, "cap5", 5.0, REPO / "results/sims/2026-05-10/capped_shared_l5_20260510_141015/sPAP95"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--ed-time",
        type=float,
        default=4.0,
        help="Target ED checkpoint time in seconds. Default 4.0 = final-beat ED for 75 bpm, 6-beat runs.",
    )
    parser.add_argument("--force", action="store_true", help="Replace files in the output directory.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def first_float(*values, default=np.nan) -> float:
    for value in values:
        if value is not None:
            return float(value)
    return float(default)


def case_metadata(case: Case) -> dict[str, float | str]:
    params = load_json(case.result_dir / "simulation_params.json")
    unload = load_json(case.result_dir / "circulation" / "unloading_diagnostics.json")
    volume2ml = float(params.get("volume2ml", 1e6))
    rv_unloaded_ml = first_float(
        unload.get("rvv_unloaded_mL"),
        params.get("rvv_unloaded_mL"),
        params.get("rvv_unloaded_m3", np.nan) * volume2ml if "rvv_unloaded_m3" in params else None,
    )
    rv_ed_ml = first_float(
        unload.get("rvv_target_mL"),
        params.get("rvv_target_mL"),
        params.get("rvv_target_m3", np.nan) * volume2ml if "rvv_target_m3" in params else None,
    )
    lv_unloaded_ml = first_float(
        unload.get("lvv_unloaded_mL"),
        params.get("lvv_unloaded_mL"),
        params.get("lvv_unloaded_m3", np.nan) * volume2ml if "lvv_unloaded_m3" in params else None,
    )
    lv_ed_ml = first_float(
        unload.get("lvv_target_mL"),
        params.get("lvv_target_mL"),
        params.get("lvv_target_m3", np.nan) * volume2ml if "lvv_target_m3" in params else None,
    )
    return {
        "label": case.label,
        "severity_sPAP": case.severity,
        "cap_label": case.cap_label,
        "cap_mmhg": case.cap_mmhg,
        "rv_unloaded_mL": rv_unloaded_ml,
        "rv_ed_mL": rv_ed_ml,
        "rv_unloaded_fraction": rv_unloaded_ml / rv_ed_ml if rv_ed_ml and np.isfinite(rv_ed_ml) else np.nan,
        "lv_unloaded_mL": lv_unloaded_ml,
        "lv_ed_mL": lv_ed_ml,
        "lv_unloaded_fraction": lv_unloaded_ml / lv_ed_ml if lv_ed_ml and np.isfinite(lv_ed_ml) else np.nan,
        "p_RV_ED_raw_mmhg": first_float(unload.get("p_RV_ED_raw_mmhg"), default=np.nan),
        "p_RV_ED_mmhg": first_float(unload.get("p_RV_ED_mmhg"), default=np.nan),
        "result_dir": str(case.result_dir),
    }


def read_case_grid(case: Case, target_time: float) -> tuple[pv.UnstructuredGrid, dict[str, float | str]]:
    comm = MPI.COMM_WORLD
    checkpoint = case.result_dir / "solver" / "checkpoint.bp"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    mesh = adios4dolfinx.read_mesh(checkpoint, comm)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    topology, cell_types, points_ref = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)
    cell_types = np.full_like(cell_types, pv.CellType.TETRA, dtype=np.uint8)

    W2 = dolfinx.fem.functionspace(mesh, ("P", 2, (3,)))
    W1 = dolfinx.fem.functionspace(mesh, ("P", 1, (3,)))
    u2 = dolfinx.fem.Function(W2, name="u_to_ED_m")
    u1 = dolfinx.fem.Function(W1, name="u_to_ED_m")

    timestamps = np.asarray(adios4dolfinx.read_timestamps(checkpoint, comm, "displacement"), dtype=float)
    if timestamps.size == 0:
        raise RuntimeError(f"No displacement timestamps in {checkpoint}")
    idx = int(np.argmin(np.abs(timestamps - target_time)))
    actual_time = float(timestamps[idx])

    adios4dolfinx.read_function(checkpoint, u2, time=actual_time, name="displacement")
    u1.interpolate(u2)
    u_vec = u1.x.array.reshape((-1, 3)).copy()
    u_mag_m = np.linalg.norm(u_vec, axis=1)
    u2_mag_cm = np.linalg.norm(u2.x.array.reshape((-1, 3)), axis=1) * 100.0

    grid = pv.UnstructuredGrid(topology, cell_types, points_ref.copy())
    grid.point_data["u_to_ED_m"] = u_vec
    grid.point_data["u_to_ED_mag_cm"] = u_mag_m * 100.0
    grid.point_data["u_to_ED_mag_mm"] = u_mag_m * 1000.0
    grid.point_data["severity_sPAP"] = np.full(grid.n_points, case.severity, dtype=float)
    grid.point_data["cap_mmhg"] = np.full(grid.n_points, case.cap_mmhg, dtype=float)
    grid.point_data["is_cap5"] = np.full(grid.n_points, 1.0 if case.cap_mmhg == 5.0 else 0.0)
    grid.point_data["ed_time_s"] = np.full(grid.n_points, actual_time, dtype=float)

    meta = case_metadata(case)
    for key in ["rv_unloaded_fraction", "lv_unloaded_fraction", "rv_unloaded_mL", "lv_unloaded_mL"]:
        grid.point_data[key] = np.full(grid.n_points, float(meta[key]), dtype=float)

    try:
        cfun = adios4dolfinx.read_meshtags(checkpoint, mesh, meshtag_name="cfun")
        region = np.zeros(grid.n_cells, dtype=np.int32)
        region[np.asarray(cfun.indices, dtype=np.int32)] = np.asarray(cfun.values, dtype=np.int32)
        grid.cell_data["region_tag_cfun"] = region
        grid.cell_data["is_septum_tag3"] = (region == 3).astype(np.int8)
    except Exception:
        pass

    meta.update(
        {
            "selected_ed_time_s": actual_time,
            "target_ed_time_s": target_time,
            "p1_u_max_cm": float((u_mag_m * 100.0).max()),
            "p1_u_mean_cm": float((u_mag_m * 100.0).mean()),
            "p2_u_max_cm": float(u2_mag_cm.max()),
            "p2_u_mean_cm": float(u2_mag_cm.mean()),
        }
    )
    return grid, meta


def write_case(case: Case, out_dir: Path, target_time: float) -> dict[str, float | str]:
    grid_ref, meta = read_case_grid(case, target_time)
    grid_ed = grid_ref.copy(deep=True)
    grid_ed.points = grid_ref.points + grid_ref.point_data["u_to_ED_m"]

    ref_dir = out_dir / "reference_unloaded_mesh"
    ed_dir = out_dir / "ed_deformed_mesh"
    surf_dir = out_dir / "surfaces_for_screenshots"
    for directory in [ref_dir, ed_dir, surf_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    ref_vtu = ref_dir / f"{case.label}_reference_unloaded.vtu"
    ed_vtu = ed_dir / f"{case.label}_ed_deformed.vtu"
    ref_surf = surf_dir / f"{case.label}_reference_unloaded_surface.vtp"
    ed_surf = surf_dir / f"{case.label}_ed_deformed_surface.vtp"

    grid_ref.save(ref_vtu)
    grid_ed.save(ed_vtu)
    grid_ref.extract_surface().save(ref_surf)
    grid_ed.extract_surface().save(ed_surf)

    meta.update(
        {
            "reference_vtu": str(ref_vtu),
            "ed_vtu": str(ed_vtu),
            "reference_surface_vtp": str(ref_surf),
            "ed_surface_vtp": str(ed_surf),
        }
    )
    return meta


def write_collections(out_dir: Path, rows: list[dict[str, float | str]]) -> None:
    """Write tiny collection files so ParaView can open groups at once."""
    for stem, key in [
        ("ed_deformed_surfaces", "ed_surface_vtp"),
        ("reference_unloaded_surfaces", "reference_surface_vtp"),
        ("ed_deformed_volumes", "ed_vtu"),
        ("reference_unloaded_volumes", "reference_vtu"),
    ]:
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            "  <Collection>",
        ]
        for i, row in enumerate(rows):
            rel = Path(str(row[key])).relative_to(out_dir)
            lines.append(f'    <DataSet timestep="{i}" group="" part="0" file="{rel.as_posix()}"/>')
        lines += ["  </Collection>", "</VTKFile>"]
        (out_dir / f"{stem}.pvd").write_text("\n".join(lines) + "\n")

    # PVD opens these as a time series. VTM opens them as a multi-block dataset,
    # which is usually nicer for side-by-side visibility toggles in ParaView.
    for stem, key in [
        ("ed_deformed_surfaces", "ed_surface_vtp"),
        ("reference_unloaded_surfaces", "reference_surface_vtp"),
    ]:
        blocks = pv.MultiBlock()
        for row in rows:
            blocks[str(row["label"])] = pv.read(str(row[key]))
        blocks.save(out_dir / f"{stem}.vtm")


def write_readme(out_dir: Path, rows: list[dict[str, float | str]]) -> None:
    severe = [r for r in rows if int(r["severity_sPAP"]) == 95]
    medium = [r for r in rows if int(r["severity_sPAP"]) == 60]

    def bullet(row: dict[str, float | str]) -> str:
        return (
            f"- `{row['label']}`: RV unloaded {float(row['rv_unloaded_mL']):.1f} mL "
            f"({float(row['rv_unloaded_fraction']):.3f} of ED), "
            f"P1 max |u_ED| {float(row['p1_u_max_cm']):.2f} cm "
            f"(P2 checkpoint max {float(row['p2_u_max_cm']):.2f} cm)"
        )

    readme = f"""# Unloading cap comparison ParaView export

Purpose: compare old uncapped inverse unloading with the production 5 mmHg
RV-EDP-capped inverse unloading, without relying on the old time-resolved VTX
first-frame convention.

All files were exported from `solver/checkpoint.bp` at final-beat ED
`t = {rows[0]['selected_ed_time_s']:.3f} s` (target 4.000 s). The convention is
identical in every case:

- `reference_unloaded_mesh/`: mesh points are the unloaded/reference state.
- `ed_deformed_mesh/`: mesh points are already moved to ED, i.e. `x_ref + u_ED`.
- `surfaces_for_screenshots/`: surface-only versions for quick screenshots.
- field `u_to_ED_m`: vector displacement from unloaded reference to ED.
- field `u_to_ED_mag_cm`: displacement magnitude in cm for colouring.
- field `rv_unloaded_fraction`: RV unloaded volume divided by ED RV volume.

## Open this first

For screenshots, open:

- `ed_deformed_surfaces.vtm` for already-deformed ED surfaces.
- `reference_unloaded_surfaces.vtm` to compare the stress-free reference shapes.

The `.vtm` files open all cases as named blocks. The `.pvd` files are also
included, but ParaView treats them as a time selector over cases.

Colour by `u_to_ED_mag_cm`. For the sPAP95 cap-effect comparison, use the same
colour range on both panels. A sensible fixed range is 0 to 2.1 cm if you want
the checkpoint-scale maximum represented, or 0 to 1.5 cm if you want stronger
surface contrast.

## Why this replaces the deleted/old export

The old uncapped VTX playback and the newer capped VTX playback do not put the
same physical state in frame 0. The uncapped file is already essentially at ED
in frame 0, while the capped file starts during the numerical re-equilibration
after the preload/restart path. That is a visualization/export convention issue,
not evidence that the production result data are corrupt. This package avoids
that ambiguity by reading the solver checkpoint and selecting the same final-beat
ED time for all cases.

## Key cases

Medium pressure:
{chr(10).join(bullet(r) for r in medium)}

High pressure:
{chr(10).join(bullet(r) for r in severe)}

## ParaView recipe

1. Open `ed_deformed_surfaces.vtm`.
2. Split the view or duplicate the source and isolate `uncapped_sPAP95` and
   `capped5_sPAP95` by toggling named blocks, or open the individual VTP files
   in `surfaces_for_screenshots/`.
3. Colour both by `u_to_ED_mag_cm`, same colour range.
4. Use the same camera and save the two panels.
5. For the reference-shrinkage panel, use the corresponding
   `reference_unloaded_surface.vtp` files and colour by `rv_unloaded_fraction`
   or keep a neutral surface colour.

The full-volume VTU files are included if you want clipping, slices, or region
tag inspection. Cell field `region_tag_cfun` is copied from the checkpoint when
available.
"""
    (out_dir / "README_PARAVIEW.md").write_text(readme)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    if out_dir.exists() and args.force:
        for child in out_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    if MPI.COMM_WORLD.size != 1:
        raise SystemExit("Run this exporter with a single MPI rank.")

    rows = []
    for case in DEFAULT_CASES:
        print(f"Exporting {case.label} ...", flush=True)
        rows.append(write_case(case, out_dir, args.ed_time))

    manifest = out_dir / "CASE_MANIFEST.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_collections(out_dir, rows)
    write_readme(out_dir, rows)
    print(f"Wrote {out_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
