#!/usr/bin/env python3
"""Backfill AHA biventricular mid-ring tags onto existing per_cell_data.npz files.

For each case directory this computes the LDRB AHA biventricular segmentation
(`ldrb.aha.gernerate_aha_biv` — the same call postprocess_metrics.py /
export_aha_band.py use) on the case geometry, then maps the per-cell AHA tag onto
the per_cell_data.npz cell ordering by nearest-centroid match (the npz stores the
reference cell centroids). Writes a sidecar `aha_tags.npy` (int8, one tag per npz
cell) next to each npz.

AHA tags:  0=Apical, 1=Basal_LV, 2=Basal_RV, 3=Basal_Septum,
                      4=Mid_LV,   5=Mid_RV,   6=Mid_Septum
The canonical "mid ring" used for region-restricted metrics is tags {4, 5, 6}
(away from both the base and the apex), split into LV / RV / septum.

This is purely geometric (no checkpoint replay), so it is cheap and independent of
the work/strain fields — it can be backfilled onto any existing run. Run serially.

Usage:
    python compute_aha_band.py <case_dir> [<case_dir> ...]
    python compute_aha_band.py --sweep <sweep_root>            # all bundles/cases under it
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
import dolfinx
import cardiac_geometries.geometry
import ldrb.aha
from cardiac_geometries.mesh import transform_markers
from scipy.spatial import cKDTree

LAB = {0: "Apical", 1: "Basal_LV", 2: "Basal_RV", 3: "Basal_Septum",
       4: "Mid_LV", 5: "Mid_RV", 6: "Mid_Septum"}


def aha_per_cell(geometry_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (aha_tag, centroids) for the geometry's local cell ordering (serial)."""
    comm = MPI.COMM_SELF
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, geometry_dir.resolve())
    mesh, ffun = geo.mesh, geo.ffun
    ldrb_markers = transform_markers(geo.markers, clipped=True)
    aha_func = ldrb.aha.gernerate_aha_biv(mesh=mesh, ffun=ffun, markers=ldrb_markers,
                                          function_space="DG_0")
    ncell = mesh.topology.index_map(3).size_local
    aha = aha_func.x.array[:ncell].astype(np.int8)
    centroids = dolfinx.mesh.compute_midpoints(mesh, 3, np.arange(ncell, dtype=np.int32))
    return aha, centroids


def backfill_case(case_dir: Path) -> None:
    npz_path = case_dir / "per_cell_data.npz"
    geo_dir = case_dir / "geometry"
    if not npz_path.exists():
        print(f"  ! {case_dir.name}: no per_cell_data.npz — skipping")
        return
    if not geo_dir.exists():
        print(f"  ! {case_dir.name}: no geometry/ — skipping")
        return

    z = np.load(npz_path, allow_pickle=True)
    npz_centroids = np.asarray(z["centroids"], float)
    region_tags = np.asarray(z["region_tags"]).astype(np.int32)

    aha, geo_centroids = aha_per_cell(geo_dir)

    # Map AHA tags (cardiac_geometries / "cg" cell ordering) onto the npz cell
    # ordering. The per-case geometry/ mesh is the atlas/ED config while the npz
    # cells are the inverse-unloaded reference, so a direct centroid match is off
    # by the prestress u_pre (~7 mm). Canonical runs store the exact cg→ckpt
    # bijection in ckpt_to_cg_idx (npz_array[i] ↔ cg cell ckpt_to_cg_idx[i]);
    # use it. Otherwise fall back to a centroid match (valid when the geometry IS
    # the reference mesh, e.g. tag-at-ed/unloaded per-case runs).
    cg_idx = np.asarray(z["ckpt_to_cg_idx"]).astype(np.int64) if "ckpt_to_cg_idx" in z.files else np.array([])
    if cg_idx.size == len(npz_centroids) and cg_idx.max() < len(aha):
        aha_npz = aha[cg_idx].astype(np.int8)
        max_d = 0.0
        how = "ckpt_to_cg_idx permutation"
    else:
        dist, idx = cKDTree(geo_centroids).query(npz_centroids)
        max_d = float(dist.max())
        aha_npz = aha[idx].astype(np.int8)
        how = "centroid match"

    # Sanity: AHA septum tags {3,6} should overlap heavily with region_tags==3.
    mid = np.isin(aha_npz, [4, 5, 6])
    n = {LAB[v]: int((aha_npz == v).sum()) for v in range(7)}
    sep_overlap = float(
        (np.isin(aha_npz, [3, 6]) & (region_tags == 3)).sum()
        / max(1, (region_tags == 3).sum())
    )

    out = case_dir / "aha_tags.npy"
    np.save(out, aha_npz)
    print(f"  {case_dir.name}: via {how} (d_max={max_d:.2e} m)  mid={int(mid.sum())}/{len(aha_npz)} "
          f"(LV {n['Mid_LV']} / RV {n['Mid_RV']} / Sep {n['Mid_Septum']})  "
          f"sept->LDRB overlap {sep_overlap:.0%}  -> {out.name}")
    if sep_overlap < 0.80:
        print(f"  !! WARNING {case_dir.name}: AHA-septum↔LDRB-septum overlap {sep_overlap:.0%} is low — "
              f"cell correspondence may be wrong.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case_dirs", type=Path, nargs="*",
                    help="case directories containing per_cell_data.npz + geometry/")
    ap.add_argument("--sweep", type=Path, default=None,
                    help="sweep root; backfills every <bundle>/<case> beneath it that "
                         "has a per_cell_data.npz")
    args = ap.parse_args()

    cases: list[Path] = list(args.case_dirs)
    if args.sweep is not None:
        cases += sorted(args.sweep.glob("*/*/per_cell_data.npz"))
        cases = [p.parent if p.name == "per_cell_data.npz" else p for p in cases]
    # de-dup, normalise to directories
    seen, ordered = set(), []
    for c in cases:
        c = c if c.is_dir() else c.parent
        if c not in seen:
            seen.add(c); ordered.append(c)

    if not ordered:
        ap.error("no case directories given (positional args or --sweep)")

    print(f"Backfilling AHA tags onto {len(ordered)} case(s)")
    for c in ordered:
        backfill_case(c)
    print("Done.")


if __name__ == "__main__":
    main()
