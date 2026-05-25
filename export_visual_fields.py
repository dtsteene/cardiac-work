#!/usr/bin/env python3
"""Export time-resolved visual fields from a completed simulation checkpoint.

This produces ParaView XDMF/HDF5 fields on the checkpoint mesh. It deliberately
does not permute cells or map between meshes: every field is assembled and
written on the same mesh read from solver/checkpoint.bp. That is the safe path
for dofmaps.

The output is for visualization and figure-making. The thesis-integrated work
numbers still come from compute_per_cell.py and the analysis CSV files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import adios4dolfinx
import basix.ufl
import cardiac_geometries.geometry
import dolfinx
import numpy as np
import pulse
import scifem
import ufl
from mpi4py import MPI
from clinical_frame import build_radial_endo_to_epi_dg0, tangent_project_longitudinal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=20, help="Write every Nth checkpoint frame.")
    parser.add_argument("--beat", type=int, default=None, help="0-indexed beat to export. Defaults to last beat.")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames by uniform subsampling after beat selection.")
    parser.add_argument("--name", default="visual_fields.xdmf")
    parser.add_argument(
        "--include-legacy-diff",
        action="store_true",
        help="Also write the old passive-only fiber-stress approximation and full-minus-old difference fields.",
    )
    return parser.parse_args()


def normalize(v):
    return v / ufl.sqrt(ufl.inner(v, v) + 1e-16)


def projection(T, v):
    return ufl.inner(ufl.dot(T, v), v)


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.rank
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    logger = logging.getLogger("visual_fields")

    results_dir = args.results_dir.resolve()
    solver_dir = results_dir / "solver"
    checkpoint_path = solver_dir / "checkpoint.bp"
    output_dir = args.output_dir.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    with (results_dir / "simulation_params.json").open() as handle:
        sim_params = json.load(handle)

    ta_history = np.load(solver_dir / "Ta_solver_history.npy")
    mesh = adios4dolfinx.read_mesh(checkpoint_path, comm)
    ffun = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="ffun")
    markers_mt = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="cfun")
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, results_dir / "geometry")
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
        n0 = None

    l0 = None
    try:
        l0 = dolfinx.fem.Function(Q_vec, name="l0")
        adios4dolfinx.read_function(checkpoint_path, l0, time=0.0, name="l0")
        radial_endo_epi_dg0, _ = build_radial_endo_to_epi_dg0(
            mesh=mesh,
            ffun=ffun,
            markers=geo.markers,
            comm=comm,
        )
        l0 = tangent_project_longitudinal(l0, radial_endo_epi_dg0)
    except Exception:
        l0 = None

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

    alpha_epi = sim_params["alpha_epi"]
    alpha_base = sim_params["alpha_base"]
    robin_epi = pulse.RobinBC(
        value=pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha_epi)), "Pa / m"),
        marker=geometry.markers["EPI"][0],
    )
    robin_base = pulse.RobinBC(
        value=pulse.Variable(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha_base)), "Pa / m"),
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
    sigma = (1.0 / ufl.det(F)) * F * S * F.T

    legacy_material = pulse.HolzapfelOgden(f0=f0, s0=f0, **material_params)
    sigma_legacy = legacy_material.sigma(F)

    f_cur = normalize(F * f0)
    s_cur = normalize(F * s0)
    n_cur = normalize(F * n0) if n0 is not None else None
    l_cur = normalize(F * l0) if l0 is not None else None

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
    n_local = mesh.topology.index_map(3).size_local

    mass_form = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * v0 * dx)
    volumes = dolfinx.fem.assemble_vector(mass_form).array[:n_local].copy()
    volumes = np.maximum(volumes, 1e-30)

    dE = E_cur - E_prev
    power_density = 0.5 * ufl.inner(S + S_prev, dE)

    forms = {
        "u_mag_m": dolfinx.fem.form(ufl.sqrt(ufl.inner(u, u) + 1e-30) * v0 * dx),
        "E_ff": dolfinx.fem.form(projection(E, f0) * v0 * dx),
        "S_ff": dolfinx.fem.form(projection(S, f0) * v0 * dx),
        "sigma_ff": dolfinx.fem.form(projection(sigma, f_cur) * v0 * dx),
        "Ta": dolfinx.fem.form(Ta.value * v0 * dx),
        "power_density": dolfinx.fem.form(power_density * v0 * dx),
    }
    if s0 is not None:
        forms["E_ss"] = dolfinx.fem.form(projection(E, s0) * v0 * dx)
        forms["S_ss"] = dolfinx.fem.form(projection(S, s0) * v0 * dx)
        forms["sigma_ss"] = dolfinx.fem.form(projection(sigma, s_cur) * v0 * dx)
    if n0 is not None:
        forms["E_nn"] = dolfinx.fem.form(projection(E, n0) * v0 * dx)
        forms["S_nn"] = dolfinx.fem.form(projection(S, n0) * v0 * dx)
        forms["sigma_nn"] = dolfinx.fem.form(projection(sigma, n_cur) * v0 * dx)
    if l0 is not None:
        forms["E_ll"] = dolfinx.fem.form(projection(E, l0) * v0 * dx)
        forms["S_ll"] = dolfinx.fem.form(projection(S, l0) * v0 * dx)
        forms["sigma_ll"] = dolfinx.fem.form(projection(sigma, l_cur) * v0 * dx)

    if args.include_legacy_diff:
        legacy_ff = projection(sigma_legacy, f_cur)
        full_ff = projection(sigma, f_cur)
        delta_ff = full_ff - legacy_ff
        forms["sigma_ff_legacy_passive_s0eqf0"] = dolfinx.fem.form(legacy_ff * v0 * dx)
        forms["sigma_ff_delta_full_minus_legacy"] = dolfinx.fem.form(delta_ff * v0 * dx)
        forms["sigma_ff_rel_delta_vs_legacy"] = dolfinx.fem.form(
            (ufl.sqrt(delta_ff * delta_ff) / ufl.sqrt(legacy_ff * legacy_ff + 1.0)) * v0 * dx
        )

    out_fields = {name: dolfinx.fem.Function(V0, name=name) for name in forms}

    timestamps = np.asarray(adios4dolfinx.read_timestamps(checkpoint_path, comm, "displacement"), dtype=float)
    if len(timestamps) == 0:
        raise RuntimeError(f"No displacement timestamps found in {checkpoint_path}")

    cycle_length = 60.0 / float(sim_params["BPM"])
    if args.beat is None:
        start_t = timestamps[-1] - cycle_length + 1e-12
        end_t = timestamps[-1] + 1e-12
    else:
        start_t = args.beat * cycle_length - 1e-12
        end_t = (args.beat + 1) * cycle_length + 1e-12

    selected = np.where((timestamps >= start_t) & (timestamps <= end_t))[0]
    if selected.size == 0:
        raise RuntimeError("Beat selection produced no frames")
    selected = selected[:: max(args.stride, 1)]
    if args.max_frames is not None and selected.size > args.max_frames:
        pick = np.unique(np.linspace(0, selected.size - 1, args.max_frames).round().astype(int))
        selected = selected[pick]
    if selected[-1] != np.where((timestamps >= start_t) & (timestamps <= end_t))[0][-1]:
        selected = np.append(selected, np.where((timestamps >= start_t) & (timestamps <= end_t))[0][-1])

    if rank == 0:
        logger.info("Exporting %d frames from %s", len(selected), results_dir)
        logger.info("Output: %s", output_dir / args.name)

    xdmf_path = output_dir / args.name
    has_previous = False
    previous_t = None
    summary_rows = []

    with dolfinx.io.XDMFFile(comm, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        for step_no, idx in enumerate(selected):
            t = float(timestamps[idx])
            adios4dolfinx.read_function(checkpoint_path, u, time=t, name="displacement")
            ta_idx = min(int(idx), len(ta_history) - 1)
            Ta.assign(ta_history[ta_idx])
            cardiac_model.active.activation.value.x.array[:] = Ta.value.x.array[:]

            E_cur.interpolate(expr_E)
            if not has_previous:
                E_prev.x.array[:] = E_cur.x.array[:]
                S_prev.interpolate(expr_S)
                has_previous = True
                previous_t = t

            dt = max(t - float(previous_t), 1e-12)
            for name, form in forms.items():
                values = dolfinx.fem.assemble_vector(form).array[:n_local].copy() / volumes
                if name == "power_density":
                    values = values / dt
                if args.include_legacy_diff and (
                    name.startswith("sigma_ff_") or name == "sigma_ff"
                ):
                    local_n = len(values)
                    local_min = float(np.min(values)) if local_n else np.inf
                    local_max = float(np.max(values)) if local_n else -np.inf
                    local_sum_abs = float(np.sum(np.abs(values)))
                    local_sum_sq = float(np.sum(values * values))
                    global_n = comm.allreduce(local_n, op=MPI.SUM)
                    global_min = comm.allreduce(local_min, op=MPI.MIN)
                    global_max = comm.allreduce(local_max, op=MPI.MAX)
                    global_sum_abs = comm.allreduce(local_sum_abs, op=MPI.SUM)
                    global_sum_sq = comm.allreduce(local_sum_sq, op=MPI.SUM)
                    if rank == 0 and global_n:
                        summary_rows.append(
                            {
                                "step": int(idx),
                                "time": t,
                                "field": name,
                                "min": global_min,
                                "max": global_max,
                                "mean_abs": global_sum_abs / global_n,
                                "rms": float(np.sqrt(global_sum_sq / global_n)),
                            }
                        )
                field = out_fields[name]
                field.x.array[:] = 0.0
                field.x.array[:n_local] = values
                field.x.scatter_forward()
                xdmf.write_function(field, t)

            E_prev.x.array[:] = E_cur.x.array[:]
            S_prev.interpolate(expr_S)
            previous_t = t
            if rank == 0:
                logger.info("  frame %03d/%03d step=%d t=%.4f", step_no + 1, len(selected), int(idx), t)

    if rank == 0:
        if summary_rows:
            import csv

            with (output_dir / "field_diff_summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["step", "time", "field", "min", "max", "mean_abs", "rms"],
                )
                writer.writeheader()
                writer.writerows(summary_rows)
        (output_dir / "README_VISUAL_FIELDS.md").write_text(
            "Open visual_fields.xdmf in ParaView.\n"
            "Fields are cell-centered on the checkpoint mesh.\n"
            "`u_mag_m` is displacement magnitude; `E_*` are Green-Lagrange strain projections; "
            "`S_*` are second Piola stress projections; `sigma_*` are Cauchy-stress projections "
            "onto pushed-forward current directions; `power_density` is an "
            "interval-averaged S:dE/dt visualization field over the written frame interval.\n"
            "`sigma_ff_legacy_passive_s0eqf0` reproduces the old passive-only visualization "
            "approximation; `sigma_ff_delta_full_minus_legacy` and "
            "`sigma_ff_rel_delta_vs_legacy` show the full-model change when requested.\n"
            "For the exact thesis work numbers, use compute_per_cell.py outputs and analysis CSVs.\n"
        )


if __name__ == "__main__":
    main()
