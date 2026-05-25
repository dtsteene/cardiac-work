#!/usr/bin/env python3
"""Export canonical septum sweep masks as a standalone XDMF/H5 pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

import cardiac_geometries
import dolfinx


SWEEP_THRESHOLDS_MM = [
    ("sweep_tm5mm", -5.0),
    ("sweep_tm3mm", -3.0),
    ("sweep_tm1mm", -1.0),
    ("sweep_tp0mm", 0.0),
    ("sweep_tp1mm", 1.0),
    ("sweep_tp3mm", 3.0),
    ("sweep_tp5mm", 5.0),
    ("sweep_tp10mm", 10.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write canonical_sweep_fields.xdmf on a selected geometry mesh."
    )
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        required=True,
        help="Geometry folder containing geometry.bp and geometry_fields.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output folder for canonical_sweep_fields.{xdmf,h5}.",
    )
    return parser.parse_args()


def function_from_array(V0, name: str, values: np.ndarray, n_cells: int):
    f = dolfinx.fem.Function(V0)
    f.name = name
    f.x.array[:n_cells] = np.asarray(values, dtype=np.float64)
    return f


def main() -> None:
    args = parse_args()
    geometry_dir = args.geometry_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = geometry_dir / "geometry_fields.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    geo = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, geometry_dir)
    mesh = geo.mesh
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local

    fields = np.load(npz_path)
    if len(fields["entry_t"]) != n_cells:
        raise RuntimeError(
            f"Geometry has {n_cells} cells but {npz_path} has "
            f"{len(fields['entry_t'])} cell values"
        )

    is_geo = fields["is_geometric_septum"].astype(bool)
    is_ldrb = fields["is_ldrb_septum"].astype(bool)
    envelope = fields["envelope"].astype(bool)
    touches_epi = fields["touches_epi"].astype(bool)
    entry_t = fields["entry_t"]

    mesh_to_mm = 1000.0 if max(float(fields["d_lv"].max()), float(fields["entry_t"].max())) < 0.1 else 1.0

    combined = np.zeros(n_cells, dtype=np.float64)
    combined[envelope] = 1.0
    combined[is_geo & ~is_ldrb] = 2.0
    combined[is_ldrb & ~is_geo] = 3.0
    combined[is_geo & is_ldrb] = 4.0
    combined[touches_epi] = 5.0

    scalar_fields = [
        ("is_geometric", is_geo.astype(float)),
        ("is_ldrb", is_ldrb.astype(float)),
        ("envelope", envelope.astype(float)),
        ("touches_epi", touches_epi.astype(float)),
        ("entry_t_mm", entry_t * mesh_to_mm),
        ("d_lv_mm", fields["d_lv"] * mesh_to_mm),
        ("d_rv_mm", fields["d_rv"] * mesh_to_mm),
        ("d_epi_mm", fields["d_epi"] * mesh_to_mm),
        ("tau", fields["tau"]),
        ("combined_definition", combined),
        ("lv_rv_scalar", fields["lv_rv_scalar"]),
        ("epi_scalar_dg0", fields["epi_scalar_dg0"]),
    ]

    for name, threshold_mm in SWEEP_THRESHOLDS_MM:
        scalar_fields.append(
            (name, (envelope & (entry_t < threshold_mm / mesh_to_mm)).astype(float))
        )

    V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    out_xdmf = out_dir / "canonical_sweep_fields.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(out_xdmf), "w") as xdmf:
        xdmf.write_mesh(mesh)
        for name, values in scalar_fields:
            xdmf.write_function(function_from_array(V0, name, values, n_cells), 0.0)

    print(f"Wrote {out_xdmf}")
    print(f"Wrote {out_xdmf.with_suffix('.h5')}")
    print(f"cells={n_cells} nodes={mesh.geometry.x.shape[0]}")
    for name, _ in SWEEP_THRESHOLDS_MM:
        values = dict(scalar_fields)[name]
        print(f"{name}: {int(values.sum())}")


if __name__ == "__main__":
    main()
