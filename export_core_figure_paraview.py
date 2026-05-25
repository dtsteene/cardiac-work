#!/usr/bin/env python3
"""Export a lightweight ParaView package for thesis field screenshots.

This is a figure-making export, not a replacement for the quantitative
postprocessing pipeline. It replays a completed checkpoint over one beat, writes
only a small number of frames, and stores two copies of the important fields:

* signed/raw cell data: DG0 per-cell values, suitable for tracing back to the
  scientific postprocess definitions;
* positive presentation cell data (``*_abs`` or ``*_pos``): fields intended for
  colour maps where non-negative intensities are easier to compare;
* point data with ``_viz`` suffix: ParaView-style cell-to-point averages of the
  presentation fields for smooth screenshots.

The cumulative work and pressure-strain proxy fields are integrated over the
replayed beat at the requested accumulation stride, then sampled at the figure
frames. Use ``--accumulation-stride 1`` for checkpoint-resolution cumulative
fields. Instantaneous stress/strain fields are quadrature-integrated to DG0
cell averages. The raw signed fields remain available, but the smoothed fields
are deliberately positive presentation fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import adios4dolfinx
import basix.ufl
import cardiac_geometries.geometry
import dolfinx
import dolfinx.fem
import dolfinx.plot
import numpy as np
import pulse
import pyvista as pv
import scifem
import ufl
from mpi4py import MPI

from clinical_frame import build_radial_endo_to_epi_dg0, tangent_project_longitudinal


REPO = Path(__file__).resolve().parent
MMHG_TO_PA = 133.322


@dataclass(frozen=True)
class FieldInfo:
    name: str
    center: str
    viz_copy: bool
    description: str


FIELD_INFOS = [
    FieldInfo("region_tag_cfun", "cell", False, "LV/RV/septum cell tag from checkpoint cfun."),
    FieldInfo("is_LV", "cell", False, "Tag-derived LV free-wall indicator."),
    FieldInfo("is_RV", "cell", False, "Tag-derived RV free-wall indicator."),
    FieldInfo("is_Septum", "cell", False, "Tag-derived septum indicator."),
    FieldInfo("tau", "cell", False, "Canonical LV-RV coordinate from per_cell_data.npz, if available."),
    FieldInfo("is_geometric_septum", "cell", False, "Canonical geometric septum mask, if available."),
    FieldInfo("is_ldrb_septum", "cell", False, "Canonical LDRB septum mask, if available."),
    FieldInfo("study_region", "cell", False, "Union/envelope study region, if available."),
    FieldInfo("envelope", "cell", False, "Canonical septum-envelope mask, if available."),
    FieldInfo("touches_epi", "cell", False, "Cells touching the epicardial tail of the septum envelope."),
    FieldInfo("entry_t_mm", "cell", False, "Septum-envelope entry coordinate in mm, if available."),
    FieldInfo("d_lv_mm", "cell", False, "Distance to LV endocardium in mm, if available."),
    FieldInfo("d_rv_mm", "cell", False, "Distance to RV endocardium in mm, if available."),
    FieldInfo("d_epi_mm", "cell", False, "Distance to epicardium in mm, if available."),
    FieldInfo("d_sum_mm", "cell", False, "d_lv + d_rv in mm, if available."),
    FieldInfo("cell_volume_m3", "cell", False, "Cell volume used to convert cell-integrated fields to densities."),
    FieldInfo("u_m", "point", False, "P1-interpolated displacement vector in metres."),
    FieldInfo("u_mag_mm", "point", False, "P1-interpolated displacement magnitude in mm."),
    FieldInfo("u_from_ED_m", "point", False, "P1-interpolated displacement vector relative to ED."),
    FieldInfo("u_mag_from_ED_mm", "point", False, "P1-interpolated displacement magnitude relative to ED in mm."),
    FieldInfo("E_ff", "cell", False, "Signed instantaneous Green-Lagrange fibre strain, DG0 cell average."),
    FieldInfo("S_ff", "cell", False, "Signed instantaneous 2PK fibre stress projection, DG0 cell average."),
    FieldInfo("E_ll", "cell", False, "Signed instantaneous tangent-longitudinal strain, DG0 cell average."),
    FieldInfo("S_ll", "cell", False, "Signed instantaneous tangent-longitudinal 2PK stress, DG0 cell average."),
    FieldInfo("E_ss", "cell", False, "Signed instantaneous sheet strain, DG0 cell average."),
    FieldInfo("S_ss", "cell", False, "Signed instantaneous sheet 2PK stress, DG0 cell average."),
    FieldInfo("E_nn", "cell", False, "Signed instantaneous sheet-normal strain, DG0 cell average."),
    FieldInfo("S_nn", "cell", False, "Signed instantaneous sheet-normal 2PK stress, DG0 cell average."),
    FieldInfo("sigma_ff", "cell", False, "Signed instantaneous Cauchy fibre stress projection, DG0 cell average."),
    FieldInfo("sigma_mag", "cell", True, "Instantaneous Cauchy stress Frobenius magnitude, DG0 cell average."),
    FieldInfo("Ta", "cell", True, "Active tension field used by the replayed model."),
    FieldInfo("E_ff_abs", "cell", True, "Non-negative fibre strain magnitude for visualization."),
    FieldInfo("S_ff_abs", "cell", True, "Non-negative fibre 2PK stress magnitude for visualization."),
    FieldInfo("E_ll_abs", "cell", True, "Non-negative tangent-longitudinal strain magnitude for visualization."),
    FieldInfo("S_ll_abs", "cell", True, "Non-negative tangent-longitudinal 2PK stress magnitude for visualization."),
    FieldInfo("E_ss_abs", "cell", True, "Non-negative sheet strain magnitude for visualization."),
    FieldInfo("S_ss_abs", "cell", True, "Non-negative sheet 2PK stress magnitude for visualization."),
    FieldInfo("E_nn_abs", "cell", True, "Non-negative sheet-normal strain magnitude for visualization."),
    FieldInfo("S_nn_abs", "cell", True, "Non-negative sheet-normal 2PK stress magnitude for visualization."),
    FieldInfo("sigma_ff_abs", "cell", True, "Non-negative Cauchy fibre stress magnitude for visualization."),
    FieldInfo("dE_ff_step", "cell", False, "Signed fibre strain increment since previous replay step."),
    FieldInfo("dE_ll_step", "cell", False, "Signed tangent-longitudinal strain increment since previous replay step."),
    FieldInfo("dE_ff_step_abs", "cell", True, "Non-negative fibre strain-increment magnitude."),
    FieldInfo("dE_ll_step_abs", "cell", True, "Non-negative tangent-longitudinal strain-increment magnitude."),
    FieldInfo("W_total_cell", "cell", False, "Signed cumulative true stress-strain work per cell over the replayed beat."),
    FieldInfo("W_total_density", "cell", False, "Signed cumulative true stress-strain work per cell divided by cell volume."),
    FieldInfo("W_ff_cell", "cell", False, "Signed cumulative fibre component of stress-strain work per cell."),
    FieldInfo("W_ff_density", "cell", False, "Signed cumulative fibre work per cell divided by cell volume."),
    FieldInfo("dW_total_density", "cell", False, "Signed last-step true work increment divided by cell volume."),
    FieldInfo("power_density", "cell", False, "Signed last-step true work increment divided by cell volume and dt."),
    FieldInfo("W_total_cell_pos", "cell", True, "Positive-work-sign cumulative true work per cell for visualization."),
    FieldInfo("W_total_density_pos", "cell", True, "Positive-work-sign cumulative true work density for visualization."),
    FieldInfo("W_ff_cell_pos", "cell", True, "Positive-work-sign cumulative fibre work per cell for visualization."),
    FieldInfo("W_ff_density_pos", "cell", True, "Positive-work-sign cumulative fibre work density for visualization."),
    FieldInfo("dW_total_density_pos", "cell", True, "Positive-work-sign last-step work increment density."),
    FieldInfo("power_density_pos", "cell", True, "Positive-work-sign instantaneous true power density."),
    FieldInfo("W_total_density_abs", "cell", True, "Absolute cumulative true work density magnitude."),
    FieldInfo("W_ff_density_abs", "cell", True, "Absolute cumulative fibre work density magnitude."),
    FieldInfo("power_density_abs", "cell", True, "Absolute instantaneous true power-density magnitude."),
    FieldInfo("proxy_PLV_ff_cell", "cell", False, "Signed cumulative LV-pressure fibre proxy per cell."),
    FieldInfo("proxy_PRV_ff_cell", "cell", False, "Signed cumulative RV-pressure fibre proxy per cell."),
    FieldInfo("proxy_Trans_ff_cell", "cell", False, "Signed cumulative transmural-pressure fibre proxy per cell."),
    FieldInfo("proxy_PLV_ff_density", "cell", False, "Signed LV-pressure fibre proxy divided by cell volume."),
    FieldInfo("proxy_PRV_ff_density", "cell", False, "Signed RV-pressure fibre proxy divided by cell volume."),
    FieldInfo("proxy_Trans_ff_density", "cell", False, "Signed transmural-pressure fibre proxy divided by cell volume."),
    FieldInfo("proxy_PLV_ll_cell", "cell", False, "Signed cumulative LV-pressure longitudinal proxy per cell."),
    FieldInfo("proxy_PRV_ll_cell", "cell", False, "Signed cumulative RV-pressure longitudinal proxy per cell."),
    FieldInfo("proxy_Trans_ll_cell", "cell", False, "Signed cumulative transmural-pressure longitudinal proxy per cell."),
    FieldInfo("proxy_PLV_ll_density", "cell", False, "Signed LV-pressure longitudinal proxy divided by cell volume."),
    FieldInfo("proxy_PRV_ll_density", "cell", False, "Signed RV-pressure longitudinal proxy divided by cell volume."),
    FieldInfo("proxy_Trans_ll_density", "cell", False, "Signed transmural-pressure longitudinal proxy divided by cell volume."),
    FieldInfo("ratio_PLV_ll_to_W_total", "cell", False, "Signed clipped PLV longitudinal proxy / true work ratio."),
    FieldInfo("ratio_Trans_ll_to_W_total", "cell", False, "Signed clipped transmural longitudinal proxy / true work ratio."),
    FieldInfo("p_LV_mmHg", "cell", False, "Solver LV pressure at the frame, repeated as a cell field."),
    FieldInfo("p_RV_mmHg", "cell", False, "Solver RV pressure at the frame, repeated as a cell field."),
    FieldInfo("p_trans_mmHg", "cell", False, "LV-RV pressure difference at the frame, repeated as a cell field."),
    FieldInfo("beat_phase", "cell", False, "Phase in the exported beat, repeated as a cell field."),
]

PROXY_COMPONENTS = ("PLV_ff", "PRV_ff", "Trans_ff", "PLV_ll", "PRV_ll", "Trans_ll")

for proxy_component in PROXY_COMPONENTS:
    FIELD_INFOS.extend(
        [
            FieldInfo(
                f"dproxy_{proxy_component}_density",
                "cell",
                False,
                f"Signed last-step {proxy_component} pressure-strain proxy increment density.",
            ),
            FieldInfo(
                f"dproxy_{proxy_component}_power_density",
                "cell",
                False,
                f"Signed instantaneous {proxy_component} pressure-strain proxy power density.",
            ),
            FieldInfo(
                f"proxy_{proxy_component}_cell_pos",
                "cell",
                True,
                f"Positive-work-sign cumulative {proxy_component} pressure-strain proxy per cell.",
            ),
            FieldInfo(
                f"proxy_{proxy_component}_density_pos",
                "cell",
                True,
                f"Positive-work-sign cumulative {proxy_component} pressure-strain proxy density.",
            ),
            FieldInfo(
                f"dproxy_{proxy_component}_density_pos",
                "cell",
                True,
                f"Positive-work-sign last-step {proxy_component} proxy increment density.",
            ),
            FieldInfo(
                f"dproxy_{proxy_component}_power_density_pos",
                "cell",
                True,
                f"Positive-work-sign instantaneous {proxy_component} proxy power density.",
            ),
            FieldInfo(
                f"proxy_{proxy_component}_density_abs",
                "cell",
                True,
                f"Absolute cumulative {proxy_component} pressure-strain proxy density magnitude.",
            ),
            FieldInfo(
                f"dproxy_{proxy_component}_density_abs",
                "cell",
                True,
                f"Absolute last-step {proxy_component} proxy increment density magnitude.",
            ),
            FieldInfo(
                f"dproxy_{proxy_component}_power_density_abs",
                "cell",
                True,
                f"Absolute instantaneous {proxy_component} proxy power-density magnitude.",
            ),
        ]
    )

FIELD_INFOS.extend(
    [
        FieldInfo("f0_vec", "cell", True, "DG0 cell-average fibre direction vector."),
        FieldInfo("s0_vec", "cell", True, "DG0 cell-average sheet direction vector."),
        FieldInfo("n0_vec", "cell", True, "DG0 cell-average sheet-normal direction vector, if available."),
        FieldInfo("l0_raw_apex_base_vec", "cell", True, "DG0 cell-average raw apex-base LDRB direction vector."),
        FieldInfo("l0_tangent_vec", "cell", True, "DG0 cell-average tangent-projected longitudinal direction used by proxies."),
        FieldInfo("radial_endo_epi_vec", "cell", True, "DG0 endocardium-to-epicardium radial direction vector."),
        FieldInfo("f0_vec_viz", "point", False, "Point-smoothed fibre direction vector for glyphs."),
        FieldInfo("s0_vec_viz", "point", False, "Point-smoothed sheet direction vector for glyphs."),
        FieldInfo("n0_vec_viz", "point", False, "Point-smoothed sheet-normal direction vector for glyphs, if available."),
        FieldInfo("l0_raw_apex_base_vec_viz", "point", False, "Point-smoothed raw apex-base direction vector for glyphs."),
        FieldInfo("l0_tangent_vec_viz", "point", False, "Point-smoothed tangent-projected longitudinal direction vector for glyphs."),
        FieldInfo("radial_endo_epi_vec_viz", "point", False, "Point-smoothed radial direction vector for glyphs."),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Completed simulation result directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to paraview_exports/core_figure_fields/<case>.",
    )
    parser.add_argument("--frames", type=int, default=9, help="Number of frames to write across the beat.")
    parser.add_argument("--beat", type=int, default=None, help="0-indexed beat to export. Defaults to last beat.")
    parser.add_argument(
        "--accumulation-stride",
        type=int,
        default=10,
        help="Replay every Nth checkpoint step for cumulative work/proxies. Default 10 is lightweight; use 1 for exact checkpoint-resolution accumulation.",
    )
    parser.add_argument("--case-label", default=None, help="Label used in filenames and README.")
    parser.add_argument("--no-surface", action="store_true", help="Skip extracted surface VTP files.")
    parser.add_argument("--no-volume", action="store_true", help="Skip full volume VTU files.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args()


def projection(T, v):
    return ufl.inner(ufl.dot(T, v), v)


def safe_ratio(num: np.ndarray, den: np.ndarray, clip: float = 5.0) -> np.ndarray:
    out = np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=np.abs(den) > 1e-18)
    return np.clip(out, -clip, clip)


def positive_work_sign(values: np.ndarray) -> np.ndarray:
    """Return non-negative positive-work-sign intensity from signed work-like data."""

    return np.maximum(-np.asarray(values, dtype=np.float64), 0.0)


def magnitude(values: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(values, dtype=np.float64))


def write_pvd(path: Path, frame_rows: list[dict[str, str | float]], key: str) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for row in frame_rows:
        rel = Path(str(row[key])).relative_to(path.parent)
        lines.append(f'    <DataSet timestep="{float(row["time_s"]):.9g}" group="" part="0" file="{rel.as_posix()}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    path.write_text("\n".join(lines) + "\n")


def read_markers(result_dir: Path) -> dict:
    geom_dir = result_dir / "geometry"
    marker_path = geom_dir / "markers.json"
    if marker_path.exists():
        return json.loads(marker_path.read_text())
    sim_params = json.loads((result_dir / "simulation_params.json").read_text())
    fallback = Path(sim_params.get("geometry_dir", "")) / "markers.json"
    if fallback.exists():
        return json.loads(fallback.read_text())
    return {}


def cell_data_to_point_viz(grid: pv.UnstructuredGrid, names: list[str]) -> None:
    if not names:
        return
    tmp = pv.UnstructuredGrid(grid.cells, grid.celltypes, grid.points)
    for name in names:
        if name in grid.cell_data:
            tmp.cell_data[name] = grid.cell_data[name]
    smoothed = tmp.cell_data_to_point_data(pass_cell_data=False)
    for name in names:
        if name in smoothed.point_data:
            grid.point_data[f"{name}_viz"] = smoothed.point_data[name]


def write_readme(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, str | float]]) -> None:
    field_lines = []
    for info in FIELD_INFOS:
        suffix = " A point-smoothed copy is written as `{}_viz`.".format(info.name) if info.viz_copy else ""
        field_lines.append(f"- `{info.name}` ({info.center}): {info.description}{suffix}")

    frame_lines = [
        f"- frame {int(row['frame'])}: t={float(row['time_s']):.4f} s, phase={float(row['beat_phase']):.3f}, "
        f"step={int(row['step_index'])}"
        for row in rows
    ]

    text = f"""# Core Figure ParaView Export

Case: `{args.case_label}`

Source result directory:

`{args.results_dir.resolve()}`

Open `volume.pvd` for clipping/slices and `surface.pvd` for quick outer-surface
screenshots. Files are sparse in time: cumulative work/proxy fields were
integrated during replay with accumulation stride `{args.accumulation_stride}`,
but only the listed frames were written. Use `--accumulation-stride 1` if you
need checkpoint-resolution cumulative fields rather than lightweight figure
fields.

Signed unsuffixed cell fields are the per-cell export values. For screenshots,
prefer the non-negative presentation fields ending in `_pos_viz` or `_abs_viz`.
Fields ending in `_viz` are point-smoothed copies made only for screenshots; do
not use them for quantitative thesis numbers.

## Frames

{chr(10).join(frame_lines)}

## Fields

{chr(10).join(field_lines)}
"""
    (out_dir / "README_CORE_FIGURE_FIELDS.md").write_text(text)


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_SELF
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logger = logging.getLogger("core_figure_export")

    result_dir = args.results_dir.resolve()
    if not result_dir.exists():
        raise FileNotFoundError(result_dir)
    args.results_dir = result_dir
    case_label = args.case_label or result_dir.name
    args.case_label = case_label

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = REPO / "paraview_exports" / "core_figure_fields" / case_label
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"{out_dir} exists; pass --force to replace it")
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    volume_dir = out_dir / "volume"
    surface_dir = out_dir / "surface"
    if not args.no_volume:
        volume_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_surface:
        surface_dir.mkdir(parents=True, exist_ok=True)

    solver_dir = result_dir / "solver"
    checkpoint_path = solver_dir / "checkpoint.bp"
    sim_params = json.loads((result_dir / "simulation_params.json").read_text())
    markers = read_markers(result_dir)
    ta_history = np.load(solver_dir / "Ta_solver_history.npy")
    solver_pressures = np.load(solver_dir / "solver_cavity_pressure_mmHg.npy")

    logger.info("Reading checkpoint mesh: %s", checkpoint_path)
    mesh = adios4dolfinx.read_mesh(checkpoint_path, comm)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    topology, cell_types, points_ref = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)
    cell_types = np.full_like(cell_types, pv.CellType.TETRA, dtype=np.uint8)
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local

    ffun = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="ffun")
    markers_mt = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="cfun")

    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, result_dir / "geometry")
    geometry = pulse.HeartGeometry(
        mesh=mesh,
        facet_tags=ffun,
        markers=geo.markers,
        metadata={"quadrature_degree": 6},
    )

    q_vec = basix.ufl.quadrature_element(mesh.topology.cell_name(), value_shape=(3,), degree=6)
    Q_vec = dolfinx.fem.functionspace(mesh, q_vec)
    f0 = dolfinx.fem.Function(Q_vec, name="f0")
    s0 = dolfinx.fem.Function(Q_vec, name="s0")
    adios4dolfinx.read_function(checkpoint_path, f0, time=0.0, name="f0")
    adios4dolfinx.read_function(checkpoint_path, s0, time=0.0, name="s0")

    n0 = None
    try:
        n0 = dolfinx.fem.Function(Q_vec, name="n0")
        adios4dolfinx.read_function(checkpoint_path, n0, time=0.0, name="n0")
    except Exception:
        logger.warning("No n0 field found; sheet-normal fields will be skipped")

    radial_endo_epi_dg0 = None
    l0_raw = None
    l0 = None
    try:
        l0_raw = dolfinx.fem.Function(Q_vec, name="l0_raw")
        adios4dolfinx.read_function(checkpoint_path, l0_raw, time=0.0, name="l0")
        radial_endo_epi_dg0, _ = build_radial_endo_to_epi_dg0(
            mesh=mesh,
            ffun=ffun,
            markers=geo.markers,
            comm=comm,
        )
        l0 = tangent_project_longitudinal(l0_raw, radial_endo_epi_dg0)
    except Exception:
        logger.warning("No l0 field found; longitudinal proxy fields will be skipped")

    material_params = {
        key: pulse.Variable(entry["value"], entry["unit"])
        for key, entry in sim_params["material_params"].items()
    }
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)
    ta_space = scifem.create_space_of_simple_functions(mesh=mesh, cell_tag=markers_mt, tags=[1, 2, 3])
    Ta = pulse.Variable(dolfinx.fem.Function(ta_space), "kPa")
    active_model = pulse.ActiveStress(f0, activation=Ta)
    comp_model = pulse.compressibility.Incompressible() if sim_params["incompressible"] else pulse.compressibility.Compressible2()
    cardiac_model = pulse.CardiacModel(material=material, active=active_model, compressibility=comp_model)

    lv_volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(sim_params["lvv_unloaded_m3"]))
    rv_volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(sim_params["rvv_unloaded_m3"]))
    lv_marker = "LV" if "LV" in geometry.markers else "ENDO_LV"
    rv_marker = "RV" if "RV" in geometry.markers else "ENDO_RV"
    cavities = [
        pulse.problem.Cavity(marker=lv_marker, volume=lv_volume),
        pulse.problem.Cavity(marker=rv_marker, volume=rv_volume),
    ]

    robin_epi = pulse.RobinBC(
        value=pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(sim_params["alpha_epi"])), "Pa / m"),
        marker=geometry.markers["EPI"][0],
    )
    robin_base = pulse.RobinBC(
        value=pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(sim_params["alpha_base"])), "Pa / m"),
        marker=geometry.markers["BASE"][0],
    )

    def dirichlet_bc(V_space):
        facets = geometry.facet_tags.find(geometry.markers["BASE"][0])
        base_dirichlet = sim_params.get("base_dirichlet", "x")
        if base_dirichlet == "full" or sim_params["incompressible"]:
            dofs = dolfinx.fem.locate_dofs_topological(V_space, 2, facets)
            u_zero = dolfinx.fem.Function(V_space)
            u_zero.x.array[:] = 0.0
            return [dolfinx.fem.dirichletbc(u_zero, dofs)]
        V_x = V_space.sub(0)
        dofs = dolfinx.fem.locate_dofs_topological(V_x, 2, facets)
        return [dolfinx.fem.dirichletbc(0.0, dofs, V_x)]

    dirichlet = () if sim_params.get("base_dirichlet", "x") == "none" else (dirichlet_bc,)
    bcs = pulse.BoundaryConditions(robin=[robin_epi, robin_base], dirichlet=dirichlet)
    problem = pulse.problem.StaticProblem(
        model=cardiac_model,
        geometry=geometry,
        bcs=bcs,
        cavities=cavities,
        parameters={"mesh_unit": sim_params["mesh_unit"], "u_space": "P_2"},
    )

    u = problem.u
    I = ufl.Identity(3)
    F = ufl.variable(ufl.grad(u) + I)
    C = ufl.variable(F.T * F)
    E = 0.5 * (C - I)
    S = cardiac_model.S(C)
    J = ufl.det(F)
    sigma = (1.0 / J) * F * S * F.T
    f_cur = F * f0
    f_cur = f_cur / ufl.sqrt(ufl.inner(f_cur, f_cur) + 1e-16)

    q_tensor = basix.ufl.quadrature_element(mesh.topology.cell_name(), value_shape=(3, 3), degree=6)
    W_tensor = dolfinx.fem.functionspace(mesh, q_tensor)
    E_cur = dolfinx.fem.Function(W_tensor, name="E_cur")
    E_prev = dolfinx.fem.Function(W_tensor, name="E_prev")
    S_prev = dolfinx.fem.Function(W_tensor, name="S_prev")
    expr_E = dolfinx.fem.Expression(E, W_tensor.element.interpolation_points)
    expr_S = dolfinx.fem.Expression(S, W_tensor.element.interpolation_points)

    V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    v0 = ufl.TestFunction(V0)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})
    mass_form = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * v0 * dx)
    cell_volumes = np.maximum(dolfinx.fem.assemble_vector(mass_form).array[:n_local].copy(), 1e-30)

    dE = E_cur - E_prev
    wd_total = 0.5 * ufl.inner(S + S_prev, dE)
    wd_ff = 0.5 * (projection(S, f0) + projection(S_prev, f0)) * projection(dE, f0)
    deps_ff = projection(dE, f0)
    deps_ll = projection(dE, l0) if l0 is not None else None

    forms_avg = {
        "E_ff": dolfinx.fem.form(projection(E, f0) * v0 * dx),
        "S_ff": dolfinx.fem.form(projection(S, f0) * v0 * dx),
        "sigma_ff": dolfinx.fem.form(projection(sigma, f_cur) * v0 * dx),
        "sigma_mag": dolfinx.fem.form(ufl.sqrt(ufl.inner(sigma, sigma) + 1e-30) * v0 * dx),
        "Ta": dolfinx.fem.form(Ta.value * v0 * dx),
    }
    if l0 is not None:
        forms_avg["E_ll"] = dolfinx.fem.form(projection(E, l0) * v0 * dx)
        forms_avg["S_ll"] = dolfinx.fem.form(projection(S, l0) * v0 * dx)
    if s0 is not None:
        forms_avg["E_ss"] = dolfinx.fem.form(projection(E, s0) * v0 * dx)
        forms_avg["S_ss"] = dolfinx.fem.form(projection(S, s0) * v0 * dx)
    if n0 is not None:
        forms_avg["E_nn"] = dolfinx.fem.form(projection(E, n0) * v0 * dx)
        forms_avg["S_nn"] = dolfinx.fem.form(projection(S, n0) * v0 * dx)

    vector_exprs = {
        "f0_vec": f0,
        "s0_vec": s0,
    }
    if n0 is not None:
        vector_exprs["n0_vec"] = n0
    if l0_raw is not None:
        vector_exprs["l0_raw_apex_base_vec"] = l0_raw
    if l0 is not None:
        vector_exprs["l0_tangent_vec"] = l0
    if radial_endo_epi_dg0 is not None:
        vector_exprs["radial_endo_epi_vec"] = radial_endo_epi_dg0
    forms_vec_avg = {
        name: [dolfinx.fem.form(vec[i] * v0 * dx) for i in range(3)]
        for name, vec in vector_exprs.items()
    }

    form_w_total = dolfinx.fem.form(wd_total * v0 * dx)
    form_w_ff = dolfinx.fem.form(wd_ff * v0 * dx)
    form_deps_ff = dolfinx.fem.form(deps_ff * v0 * dx)
    form_deps_ll = dolfinx.fem.form(deps_ll * v0 * dx) if deps_ll is not None else None

    timestamps = np.asarray(adios4dolfinx.read_timestamps(checkpoint_path, comm, "displacement"), dtype=float)
    if len(timestamps) == 0:
        raise RuntimeError(f"No displacement timestamps in {checkpoint_path}")
    cycle_length = 60.0 / float(sim_params["BPM"])
    if args.beat is None:
        beat_start_t = timestamps[-1] - cycle_length
        beat_end_t = timestamps[-1]
    else:
        beat_start_t = args.beat * cycle_length
        beat_end_t = (args.beat + 1) * cycle_length
    beat_indices = np.where((timestamps >= beat_start_t - 1e-10) & (timestamps <= beat_end_t + 1e-10))[0]
    if beat_indices.size == 0:
        raise RuntimeError("Beat selection produced no checkpoint frames")

    target_times = np.linspace(float(timestamps[beat_indices[0]]), float(timestamps[beat_indices[-1]]), max(args.frames, 2))
    write_indices = sorted({int(beat_indices[np.argmin(np.abs(timestamps[beat_indices] - t))]) for t in target_times})
    write_set = set(write_indices)
    replay_indices = beat_indices[:: max(args.accumulation_stride, 1)]
    replay_indices = np.unique(np.concatenate([replay_indices, np.asarray(write_indices, dtype=int)]))
    replay_indices = replay_indices[(replay_indices >= beat_indices[0]) & (replay_indices <= beat_indices[-1])]

    logger.info(
        "Replaying %d checkpoint steps; writing %d frames to %s",
        len(replay_indices),
        len(write_indices),
        out_dir,
    )

    W1 = dolfinx.fem.functionspace(mesh, ("P", 1, (3,)))
    u1 = dolfinx.fem.Function(W1, name="u_p1")
    ed_time = float(timestamps[beat_indices[0]])
    adios4dolfinx.read_function(checkpoint_path, u, time=ed_time, name="displacement")
    u1.interpolate(u)
    u_ed_vec = u1.x.array.reshape((-1, 3)).copy()

    cfun = np.zeros(n_local, dtype=np.int32)
    cfun[np.asarray(markers_mt.indices[:n_local], dtype=np.int32)] = np.asarray(markers_mt.values[:n_local], dtype=np.int32)
    lv_tag = int(markers.get("LV", markers.get("ENDO_LV", [1]))[0])
    rv_tag = int(markers.get("RV", markers.get("ENDO_RV", [2]))[0])
    sept_tag = 3

    static_cell_fields: dict[str, np.ndarray] = {
        "region_tag_cfun": cfun.astype(float),
        "is_LV": (cfun == lv_tag).astype(float),
        "is_RV": (cfun == rv_tag).astype(float),
        "is_Septum": (cfun == sept_tag).astype(float),
        "cell_volume_m3": cell_volumes,
    }

    pc_path = result_dir / "per_cell_data.npz"
    if pc_path.exists():
        pc = np.load(pc_path, allow_pickle=True)
        if "tau" in pc and len(pc["tau"]) == n_local:
            optional_names = [
                "tau",
                "d_lv",
                "d_rv",
                "d_epi",
                "d_sum",
                "is_geometric_septum",
                "is_ldrb_septum",
                "study_region",
                "envelope",
                "entry_t",
                "touches_epi",
            ]
            for name in optional_names:
                if name in pc:
                    values = pc[name].astype(float)
                    static_cell_fields[f"{name}_mm" if name.startswith("d_") or name == "entry_t" else name] = (
                        values * 1000.0 if name.startswith("d_") or name == "entry_t" else values
                    )
        pc.close()

    cumulative = {
        "W_total_cell": np.zeros(n_local, dtype=float),
        "W_ff_cell": np.zeros(n_local, dtype=float),
        "proxy_PLV_ff_cell": np.zeros(n_local, dtype=float),
        "proxy_PRV_ff_cell": np.zeros(n_local, dtype=float),
        "proxy_Trans_ff_cell": np.zeros(n_local, dtype=float),
        "proxy_PLV_ll_cell": np.zeros(n_local, dtype=float),
        "proxy_PRV_ll_cell": np.zeros(n_local, dtype=float),
        "proxy_Trans_ll_cell": np.zeros(n_local, dtype=float),
    }
    proxy_step = {
        f"dproxy_{component}_cell": np.zeros(n_local, dtype=float)
        for component in PROXY_COMPONENTS
    }

    has_previous = False
    previous_t = None
    p_LV_prev = 0.0
    p_RV_prev = 0.0
    frame_rows: list[dict[str, str | float]] = []

    smooth_names = [info.name for info in FIELD_INFOS if info.viz_copy]

    def pressure_at(idx: int) -> tuple[float, float]:
        j = min(int(idx), solver_pressures.shape[0] - 1)
        return float(solver_pressures[j, 0]), float(solver_pressures[j, 1])

    for idx in replay_indices:
        t = float(timestamps[int(idx)])
        adios4dolfinx.read_function(checkpoint_path, u, time=t, name="displacement")
        ta_idx = min(int(idx), len(ta_history) - 1)
        Ta.assign(ta_history[ta_idx])
        cardiac_model.active.activation.value.x.array[:] = Ta.value.x.array[:]

        E_cur.interpolate(expr_E)
        if not has_previous:
            E_prev.x.array[:] = E_cur.x.array[:]
            S_prev.interpolate(expr_S)
            p_LV_prev, p_RV_prev = pressure_at(int(idx))
            previous_t = t
            has_previous = True

        dt = max(t - float(previous_t), 1e-12)
        dW_total = np.zeros(n_local, dtype=float)
        dW_ff = np.zeros(n_local, dtype=float)
        deps_ff_arr = np.zeros(n_local, dtype=float)
        deps_ll_arr = np.zeros(n_local, dtype=float)
        for key in proxy_step:
            proxy_step[key].fill(0.0)
        if t > float(previous_t) + 1e-15:
            dW_total = dolfinx.fem.assemble_vector(form_w_total).array[:n_local].copy()
            dW_ff = dolfinx.fem.assemble_vector(form_w_ff).array[:n_local].copy()
            cumulative["W_total_cell"] += dW_total
            cumulative["W_ff_cell"] += dW_ff

            p_LV, p_RV = pressure_at(int(idx))
            p_LV_avg = 0.5 * (p_LV + p_LV_prev) * MMHG_TO_PA
            p_RV_avg = 0.5 * (p_RV + p_RV_prev) * MMHG_TO_PA
            deps_ff_arr = dolfinx.fem.assemble_vector(form_deps_ff).array[:n_local].copy()
            proxy_step["dproxy_PLV_ff_cell"] = p_LV_avg * deps_ff_arr
            proxy_step["dproxy_PRV_ff_cell"] = p_RV_avg * deps_ff_arr
            proxy_step["dproxy_Trans_ff_cell"] = (p_LV_avg - p_RV_avg) * deps_ff_arr
            cumulative["proxy_PLV_ff_cell"] += proxy_step["dproxy_PLV_ff_cell"]
            cumulative["proxy_PRV_ff_cell"] += proxy_step["dproxy_PRV_ff_cell"]
            cumulative["proxy_Trans_ff_cell"] += proxy_step["dproxy_Trans_ff_cell"]
            if form_deps_ll is not None:
                deps_ll_arr = dolfinx.fem.assemble_vector(form_deps_ll).array[:n_local].copy()
                proxy_step["dproxy_PLV_ll_cell"] = p_LV_avg * deps_ll_arr
                proxy_step["dproxy_PRV_ll_cell"] = p_RV_avg * deps_ll_arr
                proxy_step["dproxy_Trans_ll_cell"] = (p_LV_avg - p_RV_avg) * deps_ll_arr
                cumulative["proxy_PLV_ll_cell"] += proxy_step["dproxy_PLV_ll_cell"]
                cumulative["proxy_PRV_ll_cell"] += proxy_step["dproxy_PRV_ll_cell"]
                cumulative["proxy_Trans_ll_cell"] += proxy_step["dproxy_Trans_ll_cell"]
            p_LV_prev, p_RV_prev = p_LV, p_RV

        if int(idx) in write_set:
            u1.interpolate(u)
            u_vec = u1.x.array.reshape((-1, 3)).copy()
            u_from_ed = u_vec - u_ed_vec
            grid = pv.UnstructuredGrid(topology, cell_types, points_ref.copy() + u_vec)
            grid.point_data["u_m"] = u_vec
            grid.point_data["u_mag_mm"] = np.linalg.norm(u_vec, axis=1) * 1000.0
            grid.point_data["u_from_ED_m"] = u_from_ed
            grid.point_data["u_mag_from_ED_mm"] = np.linalg.norm(u_from_ed, axis=1) * 1000.0

            cell_fields: dict[str, np.ndarray] = {name: values.copy() for name, values in static_cell_fields.items()}
            for name, form in forms_avg.items():
                cell_fields[name] = dolfinx.fem.assemble_vector(form).array[:n_local].copy() / cell_volumes
            for name, forms in forms_vec_avg.items():
                cell_fields[name] = np.column_stack(
                    [dolfinx.fem.assemble_vector(form).array[:n_local].copy() / cell_volumes for form in forms]
                )

            for name in ["E_ff", "S_ff", "E_ll", "S_ll", "E_ss", "S_ss", "E_nn", "S_nn", "sigma_ff"]:
                if name in cell_fields:
                    cell_fields[f"{name}_abs"] = magnitude(cell_fields[name])

            cell_fields["dE_ff_step"] = deps_ff_arr / cell_volumes
            cell_fields["dE_ff_step_abs"] = magnitude(cell_fields["dE_ff_step"])
            if form_deps_ll is not None:
                cell_fields["dE_ll_step"] = deps_ll_arr / cell_volumes
                cell_fields["dE_ll_step_abs"] = magnitude(cell_fields["dE_ll_step"])

            cell_fields.update({name: values.copy() for name, values in cumulative.items()})
            cell_fields["W_total_density"] = cumulative["W_total_cell"] / cell_volumes
            cell_fields["W_ff_density"] = cumulative["W_ff_cell"] / cell_volumes
            cell_fields["dW_total_density"] = dW_total / cell_volumes
            cell_fields["power_density"] = (dW_total / cell_volumes) / dt
            for proxy in ["PLV_ff", "PRV_ff", "Trans_ff", "PLV_ll", "PRV_ll", "Trans_ll"]:
                key = f"proxy_{proxy}_cell"
                if key in cumulative:
                    cell_fields[f"proxy_{proxy}_density"] = cumulative[key] / cell_volumes
                dproxy_key = f"dproxy_{proxy}_cell"
                if dproxy_key in proxy_step:
                    dproxy_density = proxy_step[dproxy_key] / cell_volumes
                    cell_fields[f"dproxy_{proxy}_density"] = dproxy_density
                    cell_fields[f"dproxy_{proxy}_power_density"] = dproxy_density / dt
            cell_fields["ratio_PLV_ll_to_W_total"] = safe_ratio(
                cumulative["proxy_PLV_ll_cell"], cumulative["W_total_cell"]
            )
            cell_fields["ratio_Trans_ll_to_W_total"] = safe_ratio(
                cumulative["proxy_Trans_ll_cell"], cumulative["W_total_cell"]
            )

            for name in ["W_total_cell", "W_total_density", "W_ff_cell", "W_ff_density", "dW_total_density", "power_density"]:
                if name in cell_fields:
                    cell_fields[f"{name}_pos"] = positive_work_sign(cell_fields[name])
            for name in ["W_total_density", "W_ff_density", "power_density"]:
                if name in cell_fields:
                    cell_fields[f"{name}_abs"] = magnitude(cell_fields[name])
            for proxy in PROXY_COMPONENTS:
                for suffix in ["density", "cell"]:
                    name = f"proxy_{proxy}_{suffix}"
                    if name in cell_fields:
                        cell_fields[f"{name}_pos"] = positive_work_sign(cell_fields[name])
                for name in [
                    f"proxy_{proxy}_density",
                    f"dproxy_{proxy}_density",
                    f"dproxy_{proxy}_power_density",
                ]:
                    if name in cell_fields:
                        cell_fields[f"{name}_pos"] = positive_work_sign(cell_fields[name])
                        cell_fields[f"{name}_abs"] = magnitude(cell_fields[name])

            p_LV_now, p_RV_now = pressure_at(int(idx))
            phase = (t - float(timestamps[beat_indices[0]])) / max(
                float(timestamps[beat_indices[-1]] - timestamps[beat_indices[0]]), 1e-12
            )
            cell_fields["p_LV_mmHg"] = np.full(n_local, p_LV_now)
            cell_fields["p_RV_mmHg"] = np.full(n_local, p_RV_now)
            cell_fields["p_trans_mmHg"] = np.full(n_local, p_LV_now - p_RV_now)
            cell_fields["beat_phase"] = np.full(n_local, phase)

            for name, values in cell_fields.items():
                grid.cell_data[name] = np.asarray(values, dtype=float)
            cell_data_to_point_viz(grid, [name for name in smooth_names if name in grid.cell_data])

            frame_no = len(frame_rows)
            row: dict[str, str | float] = {
                "frame": frame_no,
                "step_index": int(idx),
                "time_s": t,
                "beat_phase": float(phase),
            }
            if not args.no_volume:
                vtu = volume_dir / f"{case_label}_frame_{frame_no:03d}.vtu"
                grid.save(vtu)
                row["volume_file"] = str(vtu)
            if not args.no_surface:
                vtp = surface_dir / f"{case_label}_surface_frame_{frame_no:03d}.vtp"
                grid.extract_surface().save(vtp)
                row["surface_file"] = str(vtp)
            frame_rows.append(row)
            logger.info("Wrote frame %03d step=%d t=%.4f phase=%.3f", frame_no, int(idx), t, phase)

        E_prev.x.array[:] = E_cur.x.array[:]
        S_prev.interpolate(expr_S)
        previous_t = t

    if not args.no_volume:
        write_pvd(out_dir / "volume.pvd", frame_rows, "volume_file")
    if not args.no_surface:
        write_pvd(out_dir / "surface.pvd", frame_rows, "surface_file")

    with (out_dir / "FRAME_MANIFEST.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)

    with (out_dir / "FIELD_MANIFEST.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "center", "point_viz_copy", "description"])
        writer.writeheader()
        for info in FIELD_INFOS:
            writer.writerow(
                {
                    "name": info.name,
                    "center": info.center,
                    "point_viz_copy": bool(info.viz_copy),
                    "description": info.description,
                }
            )

    write_readme(out_dir, args, frame_rows)
    logger.info("Done. Open %s or %s", out_dir / "surface.pvd", out_dir / "volume.pvd")


if __name__ == "__main__":
    main()
