#!/usr/bin/env python3
"""Export an unclipped UKB mesh and quantify clipping volume loss.

The production UKB geometry is generated with ``clipped=True``. This helper
generates the same atlas case with ``clipped=False`` and writes a small
ParaView-ready export, then compares clipped production cavity volumes against
the full unclipped LV/RV cavities closed by valve surfaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cardiac_geometries.geometry
import cardiac_geometries.mesh
import dolfinx
from mpi4py import MPI
import numpy as np
import pulse


REPO = Path(__file__).resolve().parent
DEFAULT_CLIPPED = REPO / "data/mesh_convergence/ukb_L5/ukb/geometry"
DEFAULT_OUTPUT = REPO / "paraview_exports/production_h5_core/unclipped_ukb_mesh"
DEFAULT_RAW = REPO / "paraview_exports/_cache/unclipped_ukb_raw_L5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clipped-geometry", type=Path, default=DEFAULT_CLIPPED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--char-length", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def apply_reference_rotation(geo, reference) -> bool:
    rotation = reference.info.get("rotation_matrix") if reference.info else None
    if rotation is None:
        return False
    R = np.asarray(rotation, dtype=float)
    geo.mesh.geometry.x[:, :3] = geo.mesh.geometry.x[:, :3] @ R.T
    return True


def maybe_scale_to_meters(geo) -> float:
    coords = geo.mesh.geometry.x
    extent = float((coords.max(axis=0) - coords.min(axis=0)).max())
    if extent > 1.0:
        coords[:] *= 1e-3
        return 1e-3
    return 1.0


def volume_ml(geo, markers: str | list[str]) -> float:
    heart = pulse.HeartGeometry.from_cardiac_geometries(
        geo,
        metadata={"quadrature_degree": 6},
    )
    return float(heart.volume(markers) * 1e6)


def write_mesh_exports(geo, out_dir: Path) -> None:
    with dolfinx.io.XDMFFile(MPI.COMM_SELF, out_dir / "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(geo.mesh)

    with dolfinx.io.XDMFFile(MPI.COMM_SELF, out_dir / "surface_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(geo.mesh)
        xdmf.write_meshtags(geo.ffun, geo.mesh.geometry)

    if geo.cfun is not None:
        with dolfinx.io.XDMFFile(MPI.COMM_SELF, out_dir / "volume_tags.xdmf", "w") as xdmf:
            xdmf.write_mesh(geo.mesh)
            xdmf.write_meshtags(geo.cfun, geo.mesh.geometry)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    raw_dir = args.raw_dir.resolve()

    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clipped = cardiac_geometries.geometry.Geometry.from_folder(
        MPI.COMM_SELF,
        args.clipped_geometry.resolve(),
    )

    if not (raw_dir / "geometry.bp").exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        unclipped = cardiac_geometries.mesh.ukb(
            outdir=raw_dir,
            comm=MPI.COMM_SELF,
            case="ED",
            char_length_max=args.char_length,
            char_length_min=args.char_length,
            clipped=False,
            create_fibers=False,
        )
    else:
        unclipped = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, raw_dir)

    rotated = apply_reference_rotation(unclipped, clipped)
    scale = maybe_scale_to_meters(unclipped)
    write_mesh_exports(unclipped, out_dir)

    maybe_scale_to_meters(clipped)

    lv_full = volume_ml(unclipped, ["LV", "MV", "AV"])
    rv_full = volume_ml(unclipped, ["RV", "TV", "PV"])
    lv_endo_only = volume_ml(unclipped, "LV")
    rv_endo_only = volume_ml(unclipped, "RV")
    lv_clipped = volume_ml(clipped, "LV")
    rv_clipped = volume_ml(clipped, "RV")

    rows = []
    for chamber, full, clipped_ml, endo_only in [
        ("LV", lv_full, lv_clipped, lv_endo_only),
        ("RV", rv_full, rv_clipped, rv_endo_only),
    ]:
        lost = full - clipped_ml
        rows.append(
            {
                "chamber": chamber,
                "unclipped_closed_volume_mL": full,
                "unclipped_endo_only_surface_integral_mL": endo_only,
                "clipped_production_volume_mL": clipped_ml,
                "lost_by_clipping_mL": lost,
                "lost_by_clipping_percent_of_unclipped": 100.0 * lost / full,
                "retained_percent_of_unclipped": 100.0 * clipped_ml / full,
            }
        )

    csv_path = out_dir / "clipping_volume_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "definition": (
            "Full unclipped cavity volume is the surface integral over the "
            "endocardial surface plus valve surfaces: LV+MV+AV and RV+TV+PV. "
            "Clipped production volume is the volume used by the simulation: "
            "the clipped production LV or RV endocardial marker."
        ),
        "clipped_geometry": str(args.clipped_geometry.resolve()),
        "unclipped_raw_generation_cache": str(raw_dir),
        "char_length_mm": args.char_length,
        "unclipped_rotated_with_clipped_reference_matrix": rotated,
        "unclipped_scale_to_meters": scale,
        "unclipped_markers": unclipped.markers,
        "clipped_markers": clipped.markers,
    }
    (out_dir / "clipping_volume_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    lines = [
        "# UKB Clipping Volume Comparison",
        "",
        "The unclipped UKB atlas mesh was regenerated with `clipped=False` and the same",
        f"{args.char_length:g} mm characteristic length used for the production h=5 mesh.",
        "",
        "Definitions:",
        "",
        "- Full unclipped LV volume: `LV + MV + AV` surface integral.",
        "- Full unclipped RV volume: `RV + TV + PV` surface integral.",
        "- Clipped production volume: the `LV` or `RV` cavity volume used by the production simulations.",
        "",
        "| chamber | unclipped closed volume (mL) | clipped production volume (mL) | lost (mL) | lost (%) | retained (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {chamber} | {full:.2f} | {clipped:.2f} | {lost:.2f} | {lost_pct:.1f} | {retained:.1f} |".format(
                chamber=row["chamber"],
                full=row["unclipped_closed_volume_mL"],
                clipped=row["clipped_production_volume_mL"],
                lost=row["lost_by_clipping_mL"],
                lost_pct=row["lost_by_clipping_percent_of_unclipped"],
                retained=row["retained_percent_of_unclipped"],
            )
        )
    lines.extend(
        [
            "",
            "Open `mesh.xdmf` for the unclipped mesh and `surface_tags.xdmf` for the",
            "unclipped surface markers. The coordinates are written in meters so this",
            "can be compared directly with the production clipped mesh.",
        ]
    )
    (out_dir / "README_UNCLIPPED_UKB.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote {out_dir}")
    for row in rows:
        print(
            "{chamber}: unclipped={full:.2f} mL, clipped={clipped:.2f} mL, "
            "lost={lost:.2f} mL ({lost_pct:.1f}%)".format(
                chamber=row["chamber"],
                full=row["unclipped_closed_volume_mL"],
                clipped=row["clipped_production_volume_mL"],
                lost=row["lost_by_clipping_mL"],
                lost_pct=row["lost_by_clipping_percent_of_unclipped"],
            )
        )


if __name__ == "__main__":
    main()
