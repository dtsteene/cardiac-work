#!/usr/bin/env python3
"""Export ED-deformed meshes for the production capped (L5, 5 mmHg cap)
unloading sweep, packaged for local PyVista animation.

For each of the 16 sPAP cases in the canonical sweep:
  results/sims/2026-05-10/capped_shared_l5_20260510_141015/sPAP*/

we read the displacement at t = 4.0 s (= start of the final beat = ED of the
last cardiac cycle for 75 bpm, 6-beat runs), apply it to the per-case unloaded
reference mesh, and attach every per-cell scalar already living in
``per_cell_data.npz`` (last-beat integrated true work, decomposition, and
P x epsilon proxies). Cells in ``per_cell_data.npz`` are in MPI-rank
concatenation order, so we map them onto the serial DOLFINx cell ordering via
KDTree on cell centroids (the centroids are also saved in the npz).

Output layout (under paraview_exports/production_capped_sweep_ed/):

    sweep.pvd                       ParaView/PyVista collection, severity as time
    ed_meshes/sPAPxx_ed.vtu         ED-deformed volumetric mesh, all fields
    ed_surfaces/sPAPxx_ed_surface.vtp   Outer surface extract (epi + endo), all fields
    manifest.json                   Per-case metadata (severity, ED time, peak pressures, volumes)
    global_ranges.json              Field min/max across the full sweep
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import adios4dolfinx
import dolfinx
import dolfinx.fem
import dolfinx.plot
import numpy as np
import pyvista as pv
from mpi4py import MPI
from scipy.spatial import cKDTree


import paths
SWEEP_ROOT = paths.RESULTS_ROOT / "sims/2026-06-09/pah_pulmonary_20260609_prodsweep"
DEFAULT_OUT = paths.REPO_ROOT / "paraview_exports/pah_pulmonary_ed"
SEVERITIES = [22, 25, 30, 35, 45, 50, 55, 60, 65, 70, 75, 80, 85, 87, 92, 95]
ED_TIME_S = 4.0  # start of 6th beat at 75 bpm

# Fields we copy verbatim from per_cell_data.npz onto each VTU. Each entry is
# (npz_key, vtu_name, kind) where kind = "work" (J), "proxy" (J), "geom" (other).
PER_CELL_J_FIELDS = [
    ("w_total", "w_total_J", "work"),
    ("w_ff", "w_ff_J", "work"),
    ("w_ss", "w_ss_J", "work"),
    ("w_nn", "w_nn_J", "work"),
    ("w_cross", "w_cross_J", "work"),
    ("proxy_PLV_ll", "proxy_PLV_ll_J", "proxy"),
    ("proxy_PRV_ll", "proxy_PRV_ll_J", "proxy"),
    ("proxy_Trans_ll", "proxy_Trans_ll_J", "proxy"),
    ("proxy_PLV_ff", "proxy_PLV_ff_J", "proxy"),
    ("proxy_PRV_ff", "proxy_PRV_ff_J", "proxy"),
    ("proxy_Trans_ff", "proxy_Trans_ff_J", "proxy"),
]

# Geometric / coordinate fields, useful for clipping locally
GEOM_FIELDS = [
    ("region_tags", "region_tag", "int32"),  # 1=LV, 2=RV, 3=Septum
    ("tau", "tau_lv_to_rv", "float32"),
    ("d_lv", "d_lv_m", "float32"),
    ("d_rv", "d_rv_m", "float32"),
    ("d_epi", "d_epi_m", "float32"),
    ("d_sum", "d_sum_m", "float32"),
    ("is_geometric_septum", "is_geometric_septum", "uint8"),
    ("is_ldrb_septum", "is_ldrb_septum", "uint8"),
    ("study_region", "study_region", "uint8"),
    ("envelope", "envelope", "uint8"),
]


@dataclass(frozen=True)
class Case:
    severity: int
    result_dir: Path
    label: str

    @classmethod
    def from_severity(cls, sev: int) -> "Case":
        return cls(severity=sev,
                   result_dir=SWEEP_ROOT / f"sPAP{sev}",
                   label=f"sPAP{sev}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ed-time", type=float, default=ED_TIME_S,
                        help="Target ED time in seconds (default 4.0 = start of final beat at 75 bpm).")
    parser.add_argument("--severities", type=int, nargs="*", default=None,
                        help="Subset of sPAP values to export. Default: all 16.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh)


def case_metadata(case: Case, ed_time_actual: float) -> dict:
    """Lightweight metadata for the per-case manifest entry."""
    params = load_json(case.result_dir / "simulation_params.json")
    unload = load_json(case.result_dir / "circulation" / "unloading_diagnostics.json")
    p_history = case.result_dir / "solver" / "solver_cavity_pressure_mmHg.npy"
    peak_lv = peak_rv = float("nan")
    if p_history.exists():
        ph = np.load(p_history)
        # last 800 steps = last beat at dt=0.001 / 75 bpm
        last_beat = ph[-800:]
        peak_lv = float(last_beat[:, 0].max())
        peak_rv = float(last_beat[:, 1].max())
    volume2ml = float(params.get("volume2ml", 1e6))

    def _vol_ml(key_ml: str, key_m3: str) -> float:
        for src in (unload, params):
            if key_ml in src and src[key_ml] is not None:
                return float(src[key_ml])
            if key_m3 in src and src[key_m3] is not None:
                return float(src[key_m3]) * volume2ml
        return float("nan")

    return {
        "label": case.label,
        "severity_sPAP": case.severity,
        "ed_time_s": ed_time_actual,
        "peak_p_LV_mmHg": peak_lv,
        "peak_p_RV_mmHg": peak_rv,
        "rv_unloaded_mL": _vol_ml("rvv_unloaded_mL", "rvv_unloaded_m3"),
        "rv_ed_mL": _vol_ml("rvv_target_mL", "rvv_target_m3"),
        "lv_unloaded_mL": _vol_ml("lvv_unloaded_mL", "lvv_unloaded_m3"),
        "lv_ed_mL": _vol_ml("lvv_target_mL", "lvv_target_m3"),
        "result_dir": str(case.result_dir),
    }


def _serial_cell_centroids(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    c2v = mesh.topology.connectivity(mesh.topology.dim, 0)
    n = mesh.topology.index_map(mesh.topology.dim).size_local
    verts = mesh.geometry.x
    centroids = np.zeros((n, 3), dtype=np.float64)
    for c in range(n):
        centroids[c] = verts[c2v.links(c)].mean(axis=0)
    return centroids


def read_case_grid(case: Case, target_time: float) -> tuple[pv.UnstructuredGrid, dict]:
    """Build an ED-deformed UnstructuredGrid with all per-cell fields attached."""
    comm = MPI.COMM_WORLD
    checkpoint = case.result_dir / "solver" / "checkpoint.bp"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    mesh = adios4dolfinx.read_mesh(checkpoint, comm)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    topology, cell_types, points_ref = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)
    cell_types = np.full_like(cell_types, pv.CellType.TETRA, dtype=np.uint8)

    # Read displacement and interpolate P2 -> P1 for clean VTU
    W2 = dolfinx.fem.functionspace(mesh, ("P", 2, (3,)))
    W1 = dolfinx.fem.functionspace(mesh, ("P", 1, (3,)))
    u2 = dolfinx.fem.Function(W2)
    u1 = dolfinx.fem.Function(W1)

    timestamps = np.asarray(adios4dolfinx.read_timestamps(checkpoint, comm, "displacement"))
    if timestamps.size == 0:
        raise RuntimeError(f"No displacement timestamps in {checkpoint}")
    idx = int(np.argmin(np.abs(timestamps - target_time)))
    actual_time = float(timestamps[idx])
    adios4dolfinx.read_function(checkpoint, u2, time=actual_time, name="displacement")
    u1.interpolate(u2)
    u_vec = u1.x.array.reshape((-1, 3)).copy()

    # ED-deformed grid (points = reference + displacement)
    grid = pv.UnstructuredGrid(topology, cell_types, points_ref + u_vec)
    grid.point_data["u_to_ED_m"] = u_vec
    grid.point_data["u_to_ED_mag_mm"] = np.linalg.norm(u_vec, axis=1) * 1000.0

    # Load per_cell_data and build centroid -> mesh cell index map
    pcd_path = case.result_dir / "per_cell_data.npz"
    if not pcd_path.exists():
        raise FileNotFoundError(pcd_path)
    pcd = np.load(pcd_path)
    serial_centroids = _serial_cell_centroids(mesh)
    tree = cKDTree(pcd["centroids"])
    dist, mesh_to_pcd = tree.query(serial_centroids, k=1)
    if dist.max() > 1e-6:
        raise RuntimeError(
            f"{case.label}: KDTree max distance {dist.max():.3e} m exceeds tolerance; "
            "cell ordering mismatch."
        )

    cell_volumes = pcd["cell_volumes"][mesh_to_pcd].astype(np.float64)
    cell_volumes_safe = np.where(cell_volumes > 0, cell_volumes, np.nan)
    grid.cell_data["cell_volume_m3"] = cell_volumes

    # Region tags + masks
    region_tags = pcd["region_tags"][mesh_to_pcd].astype(np.int32)
    grid.cell_data["region_tag"] = region_tags
    grid.cell_data["is_LV"] = (region_tags == 1).astype(np.uint8)
    grid.cell_data["is_RV"] = (region_tags == 2).astype(np.uint8)
    grid.cell_data["is_Septum"] = (region_tags == 3).astype(np.uint8)

    # Geometric / coordinate fields (best-effort: skip if missing in older runs)
    for npz_key, name, dtype in GEOM_FIELDS:
        if npz_key in pcd.files:
            arr = pcd[npz_key][mesh_to_pcd]
            if dtype == "uint8":
                arr = arr.astype(np.uint8)
            elif dtype == "int32":
                arr = arr.astype(np.int32)
            else:
                arr = arr.astype(np.float32)
            grid.cell_data[name] = arr

    # Per-cell J fields + density (J / m^3 = Pa)
    for npz_key, vtu_name, _kind in PER_CELL_J_FIELDS:
        if npz_key not in pcd.files:
            continue
        vals_J = pcd[npz_key][mesh_to_pcd].astype(np.float64)
        grid.cell_data[vtu_name] = vals_J.astype(np.float32)
        grid.cell_data[vtu_name.replace("_J", "_density_Pa")] = (
            vals_J / cell_volumes_safe
        ).astype(np.float32)

    # Smart "combined" proxy:
    #   LV freewall (tag=1)  -> PLV * epsilon_ll  (adjacent pressure)
    #   RV freewall (tag=2)  -> PRV * epsilon_ll  (adjacent pressure)
    #   Septum     (tag=3)  -> PLV * epsilon_ll  (standard clinical convention)
    plv_ll = pcd["proxy_PLV_ll"][mesh_to_pcd].astype(np.float64)
    prv_ll = pcd["proxy_PRV_ll"][mesh_to_pcd].astype(np.float64)
    combined_J = np.where(region_tags == 2, prv_ll, plv_ll)
    grid.cell_data["proxy_combined_ll_J"] = combined_J.astype(np.float32)
    grid.cell_data["proxy_combined_ll_density_Pa"] = (
        combined_J / cell_volumes_safe
    ).astype(np.float32)
    # Same construction for the fiber-strain proxy, for completeness
    plv_ff = pcd["proxy_PLV_ff"][mesh_to_pcd].astype(np.float64)
    prv_ff = pcd["proxy_PRV_ff"][mesh_to_pcd].astype(np.float64)
    combined_ff_J = np.where(region_tags == 2, prv_ff, plv_ff)
    grid.cell_data["proxy_combined_ff_J"] = combined_ff_J.astype(np.float32)
    grid.cell_data["proxy_combined_ff_density_Pa"] = (
        combined_ff_J / cell_volumes_safe
    ).astype(np.float32)

    # Field-data scalars (severity, peak pressures) -- baked into the file as
    # length-1 arrays, accessible as field_data in PyVista.
    meta = case_metadata(case, actual_time)
    for key in ("severity_sPAP", "ed_time_s", "peak_p_LV_mmHg", "peak_p_RV_mmHg",
                "rv_unloaded_mL", "rv_ed_mL", "lv_unloaded_mL", "lv_ed_mL"):
        val = meta.get(key, float("nan"))
        grid.field_data[key] = np.array([float(val)], dtype=np.float32)

    return grid, meta


def write_pvd(pvd_path: Path, entries: list[tuple[float, Path]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for timestep, file_path in entries:
        rel = file_path.relative_to(pvd_path.parent)
        lines.append(
            f'    <DataSet timestep="{timestep:.4f}" group="" part="0" '
            f'file="{escape(str(rel))}"/>'
        )
    lines.append("  </Collection>")
    lines.append("</VTKFile>")
    pvd_path.write_text("\n".join(lines) + "\n")


def update_global_ranges(global_ranges: dict, grid: pv.UnstructuredGrid) -> None:
    for name, arr in grid.cell_data.items():
        a = np.asarray(arr).ravel()
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            continue
        cur = global_ranges.setdefault(name, [float("inf"), float("-inf")])
        cur[0] = min(cur[0], float(finite.min()))
        cur[1] = max(cur[1], float(finite.max()))


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vol_dir = out_dir / "ed_meshes"
    surf_dir = out_dir / "ed_surfaces"
    vol_dir.mkdir(exist_ok=True)
    surf_dir.mkdir(exist_ok=True)

    severities = args.severities or SEVERITIES
    manifest: list[dict] = []
    pvd_entries: list[tuple[float, Path]] = []
    surf_entries: list[tuple[float, Path]] = []
    global_ranges: dict[str, list[float]] = {}

    for sev in severities:
        case = Case.from_severity(sev)
        print(f"[{case.label}] reading {case.result_dir}")
        grid, meta = read_case_grid(case, args.ed_time)

        vol_path = vol_dir / f"{case.label}_ed.vtu"
        grid.save(vol_path)
        print(f"  -> {vol_path}  ({grid.n_cells} cells, {grid.n_points} points)")
        pvd_entries.append((float(case.severity), vol_path))

        # Extract surface (epi + endo); inherits all cell/point data
        surface = grid.extract_surface(pass_pointid=False, pass_cellid=False)
        surf_path = surf_dir / f"{case.label}_ed_surface.vtp"
        surface.save(surf_path)
        surf_entries.append((float(case.severity), surf_path))

        update_global_ranges(global_ranges, grid)
        manifest.append(meta)

    sweep_pvd = out_dir / "sweep.pvd"
    write_pvd(sweep_pvd, pvd_entries)
    write_pvd(out_dir / "sweep_surface.pvd", surf_entries)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "global_ranges.json").write_text(
        json.dumps(global_ranges, indent=2, sort_keys=True)
    )

    print(f"\nWrote PVD: {sweep_pvd}")
    print(f"Wrote {len(manifest)} cases. Manifest: {out_dir/'manifest.json'}")
    print(f"Field ranges: {out_dir/'global_ranges.json'}")


if __name__ == "__main__":
    main()
