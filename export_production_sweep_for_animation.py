#!/usr/bin/env python3
"""Export ED-deformed meshes for the PAH pulmonary-windkessel sweep
(``pah_pulmonary_20260609_prodsweep``), packaged for local PyVista animation.

Sweep layout::

    SWEEP_ROOT/<bundle>/<case>/

where bundles are ``no_frank_starling``, ``frank_starling_preload``,
``frank_starling_relax`` and cases are ``case0_rv25`` … ``case7_rv95``.

For each case we read the displacement at the start of the final beat (ED of
the last cardiac cycle, derived from BPM / RR_INTERVAL saved in
``simulation_params.json``), apply it to the per-case unloaded reference mesh,
and attach every per-cell scalar already living in ``per_cell_data.npz``
(last-beat integrated true work, decomposition, and P x epsilon proxies),
together with J/m³ density variants.  Cells in ``per_cell_data.npz`` are in
MPI-rank concatenation order, so we map them onto the serial DOLFINx cell
ordering via KDTree on cell centroids (the centroids are also saved in the npz).

Output layout (under ``DEFAULT_OUT/<bundle>/``)::

    sweep.pvd                         ParaView/PyVista collection, RV sPAP as time
    ed_meshes/<case>_ed.vtu           ED-deformed volumetric mesh, all fields
    ed_surfaces/<case>_ed_surface.vtp Outer surface extract, all fields
    manifest.json                     Per-case metadata
    global_ranges.json                Field min/max across the full bundle
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

CASES = [
    "case0_rv25",
    "case1_rv35",
    "case2_rv45",
    "case3_rv55",
    "case4_rv65",
    "case5_rv75",
    "case6_rv85",
    "case7_rv95",
]

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


def ed_time_for(case_dir: Path) -> float:
    """Start of the final beat in seconds, derived from simulation_params.json.

    Derived as ``(n_beats - 1) * RR_INTERVAL`` where ``n_beats`` is inferred
    from the length of the pressure history file divided by RR_INTERVAL.
    Falls back to 75 BPM / 6 beats if keys are absent.
    """
    sp_path = case_dir / "simulation_params.json"
    sp: dict = {}
    if sp_path.exists():
        sp = json.loads(sp_path.read_text())

    bpm = float(sp.get("BPM", sp.get("bpm", 75)))
    rr = float(sp.get("RR_INTERVAL", 60.0 / bpm))
    dt = float(sp.get("dt", 0.001))

    # Infer n_beats from the pressure-history file length
    p_hist = case_dir / "solver" / "solver_cavity_pressure_mmHg.npy"
    if p_hist.exists():
        n_steps = np.load(p_hist, mmap_mode="r").shape[0]
        total_time = n_steps * dt
        n_beats = max(1, round(total_time / rr))
    else:
        n_beats = int(sp.get("BEATS", sp.get("beats", 6)))

    return (n_beats - 1) * rr


def _rv_peak_mmhg(case_dir: Path, n_beats: int = 6) -> float:
    """Peak RV pressure (mmHg) over the last beat, for use as PVD timestep."""
    p_hist = case_dir / "solver" / "solver_cavity_pressure_mmHg.npy"
    if not p_hist.exists():
        return float("nan")
    ph = np.load(p_hist)
    last_n = max(1, ph.shape[0] // n_beats)
    return float(ph[-last_n:, 1].max())


@dataclass(frozen=True)
class Case:
    name: str        # e.g. "case3_rv55"
    result_dir: Path
    label: str       # same as name

    @classmethod
    def from_bundle_case(cls, bundle: str, case: str) -> "Case":
        return cls(name=case,
                   result_dir=SWEEP_ROOT / bundle / case,
                   label=case)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", default="no_frank_starling",
                        choices=["no_frank_starling",
                                 "frank_starling_preload",
                                 "frank_starling_relax"],
                        help="Which bundle (sub-directory) to export (default: no_frank_starling).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                        help="Root output directory; results go into <output-dir>/<bundle>/.")
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
        # last 1/n_beats fraction of steps = last beat
        n_steps = ph.shape[0]
        dt = float(params.get("dt", 0.001))
        rr = float(params.get("RR_INTERVAL", 60.0 / float(params.get("BPM", 75))))
        total_time = n_steps * dt
        n_beats = max(1, round(total_time / rr))
        last_n = max(1, n_steps // n_beats)
        last_beat = ph[-last_n:]
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
        "case_name": case.name,
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

    # Extra proxy density fields not already covered by PER_CELL_J_FIELDS
    for proxy_key in ("proxy_Mean_ll", "proxy_Sum_ll"):
        if proxy_key in pcd.files:
            vals_J = pcd[proxy_key][mesh_to_pcd].astype(np.float64)
            grid.cell_data[f"{proxy_key}_J"] = vals_J.astype(np.float32)
            grid.cell_data[f"{proxy_key}_density_Pa"] = (
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

    # Field-data scalars (peak pressures etc.) -- baked in as length-1 arrays
    meta = case_metadata(case, actual_time)
    for key in ("ed_time_s", "peak_p_LV_mmHg", "peak_p_RV_mmHg",
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
    bundle = args.bundle
    out_dir = args.output_dir / bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    vol_dir = out_dir / "ed_meshes"
    surf_dir = out_dir / "ed_surfaces"
    vol_dir.mkdir(exist_ok=True)
    surf_dir.mkdir(exist_ok=True)

    manifest: list[dict] = []
    pvd_entries: list[tuple[float, Path]] = []
    surf_entries: list[tuple[float, Path]] = []
    global_ranges: dict[str, list[float]] = {}

    for case_name in CASES:
        case = Case.from_bundle_case(bundle, case_name)
        ed_time = ed_time_for(case.result_dir)
        print(f"[{bundle}/{case.label}] reading {case.result_dir}  (ED={ed_time:.3f}s)")
        grid, meta = read_case_grid(case, ed_time)

        vol_path = vol_dir / f"{case.label}_ed.vtu"
        grid.save(vol_path)
        print(f"  -> {vol_path}  ({grid.n_cells} cells, {grid.n_points} points)")

        # Use RV peak pressure as the PVD "time" axis so the animation sweeps sPAP
        pvd_time = float(meta.get("peak_p_RV_mmHg", float("nan")))
        pvd_entries.append((pvd_time, vol_path))

        # Extract surface (epi + endo); inherits all cell/point data
        surface = grid.extract_surface(pass_pointid=False, pass_cellid=False)
        surf_path = surf_dir / f"{case.label}_ed_surface.vtp"
        surface.save(surf_path)
        surf_entries.append((pvd_time, surf_path))

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
