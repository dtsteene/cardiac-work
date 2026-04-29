#!/usr/bin/env python3
"""Check LV/RV cavity volumes in the two top-level patient meshes.

The top-level XDMF/HDF5 meshes use facet tags:
  20 = RV endocardium
  30 = LV endocardium
  40 = epicardium

The coordinates are in cm, so cm^3 is mL.  The cavity volume is computed
with the same divergence-theorem idea used by the solver/optimizer, but
implemented directly from the HDF5 mesh/facet topology so it can run without
dolfinx.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


ROOT = Path("/home/dtsteene/D1/cardiac-work")
MESHES = ["healthy", "pah"]


def face_to_opposite_vertex(cells: np.ndarray) -> dict[tuple[int, int, int], list[int]]:
    face_definitions = [
        (1, 2, 3, 0),
        (0, 3, 2, 1),
        (0, 1, 3, 2),
        (0, 2, 1, 3),
    ]
    out: dict[tuple[int, int, int], list[int]] = {}
    for tet in cells:
        for i, j, k, opposite in face_definitions:
            face = tuple(sorted((int(tet[i]), int(tet[j]), int(tet[k]))))
            out.setdefault(face, []).append(int(tet[opposite]))
    return out


def oriented_surface_volume(
    coords: np.ndarray,
    triangles: np.ndarray,
    opposite_vertices: dict[tuple[int, int, int], list[int]],
) -> float:
    volume = 0.0
    for triangle in triangles:
        ia, ib, ic = map(int, triangle)
        a, b, c = coords[[ia, ib, ic]]
        area_vector = 0.5 * np.cross(b - a, c - a)

        opposite = coords[opposite_vertices[tuple(sorted((ia, ib, ic)))][0]]
        if np.dot(area_vector, opposite - a) > 0:
            area_vector = -area_vector

        centroid = (a + b + c) / 3.0
        volume += float(np.dot(centroid, area_vector) / 3.0)
    return abs(volume)


def tet_wall_volume(coords: np.ndarray, cells: np.ndarray) -> float:
    a = coords[cells[:, 0]]
    b = coords[cells[:, 1]]
    c = coords[cells[:, 2]]
    d = coords[cells[:, 3]]
    vols = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0
    return float(vols.sum())


def read_mesh(name: str) -> dict[str, float]:
    path = ROOT / "data" / f"{name}.h5"
    with h5py.File(path, "r") as handle:
        coords = handle["Mesh/mesh/geometry"][:]
        cells = handle["Mesh/mesh/topology"][:]
        triangles = handle["MeshTags/facet_tags/topology"][:]
        tags = handle["MeshTags/facet_tags/Values"][:].reshape(-1)

    opposite_vertices = face_to_opposite_vertex(cells)
    lv_volume = oriented_surface_volume(coords, triangles[tags == 30], opposite_vertices)
    rv_volume = oriented_surface_volume(coords, triangles[tags == 20], opposite_vertices)
    wall_volume = tet_wall_volume(coords, cells)
    return {
        "nodes": float(len(coords)),
        "tets": float(len(cells)),
        "LV_EDV_mL": lv_volume,
        "RV_EDV_mL": rv_volume,
        "RV_to_LV": rv_volume / lv_volume,
        "wall_volume_mL": wall_volume,
    }


def main() -> None:
    rows = [(name, read_mesh(name)) for name in MESHES]
    print(f"{'mesh':<10} {'LV_EDV':>9} {'RV_EDV':>9} {'RV/LV':>8} {'wall':>9} {'nodes':>7} {'tets':>7}")
    for name, row in rows:
        print(
            f"{name:<10} {row['LV_EDV_mL']:9.1f} {row['RV_EDV_mL']:9.1f} "
            f"{row['RV_to_LV']:8.3f} {row['wall_volume_mL']:9.1f} "
            f"{row['nodes']:7.0f} {row['tets']:7.0f}"
        )

    healthy = rows[0][1]
    pah = rows[1][1]
    print("\nPAH relative to healthy")
    for key in ["LV_EDV_mL", "RV_EDV_mL", "RV_to_LV", "wall_volume_mL"]:
        ratio = pah[key] / healthy[key]
        print(f"  {key:<16} {ratio:7.3f}x")


if __name__ == "__main__":
    main()
