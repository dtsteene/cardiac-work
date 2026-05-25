#!/usr/bin/env python3
"""Export volume and surface mesh tags from a cardiac_geometries folder.

This is a small helper for ParaView. It writes tag XDMF files into an output
folder and does not modify the source geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cardiac_geometries.geometry
import dolfinx
from mpi4py import MPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry_dir", type=Path, help="Folder containing geometry.bp")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    geo = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, args.geometry_dir.resolve())

    with dolfinx.io.XDMFFile(MPI.COMM_SELF, out / "surface_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(geo.mesh)
        xdmf.write_meshtags(geo.ffun, geo.mesh.geometry)

    markers_mt = geo.additional_data.get("markers_mt")
    if markers_mt is not None:
        with dolfinx.io.XDMFFile(MPI.COMM_SELF, out / "volume_tags.xdmf", "w") as xdmf:
            xdmf.write_mesh(geo.mesh)
            xdmf.write_meshtags(markers_mt, geo.mesh.geometry)

    print(f"Wrote tag XDMF files to {out}")


if __name__ == "__main__":
    main()
