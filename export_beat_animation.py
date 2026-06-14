#!/usr/bin/env python3
"""export_beat_animation.py — Through-beat PVD animation for baseline + severe PAH cases.

For each requested case in a bundle, replays the last cardiac beat step-by-step
from the displacement checkpoint, accumulates per-cell cumulative work density
(true internal work S:dE) and per-cell cumulative pressure-strain proxy density
(P_LV * d_epsilon_ll) from end-diastole, and writes a time-series PVD so that
importing into PyVista or ParaView shows the work building up over one heartbeat.

Output layout::

    <output_dir>/<bundle>/<case>/step_NNN.vtu   one deformed mesh per timestep
    <output_dir>/<bundle>/<case>/beat.pvd        PVD collection (timestep = beat phase)

Usage (single case, development)::

    python export_beat_animation.py --bundle no_frank_starling --cases case0_rv25

Usage (MPI, production)::

    mpirun -n 8 -launcher fork python export_beat_animation.py \\
        --bundle no_frank_starling

Design
------
Per-cell DG0 assembly (the expensive part) runs in parallel across all MPI ranks.
VTU writing is handled entirely on rank 0 via a lightweight serial mesh loaded
once with ``COMM_SELF``; the per-cell arrays are gathered from all ranks before
each write. This avoids the complexity of parallel VTU DOF ownership.

compute_per_cell.py was refactored to guard its script body inside ``_main()``
so that ``import compute_per_cell`` is now instantaneous (no replay on import).
This exporter does NOT import compute_per_cell — it independently reconstructs
the minimal subset of forms needed for the animation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

import numpy as np
from mpi4py import MPI

import paths

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SWEEP = (
    paths.RESULTS_ROOT
    / "sims/2026-06-09/pah_pulmonary_20260609_prodsweep"
)
DEFAULT_OUT = paths.REPO_ROOT / "paraview_exports/pah_pulmonary_beat"

DEFAULT_CASES = ["case0_rv25", "case7_rv95"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bundle",
        default="no_frank_starling",
        help="Bundle sub-directory (default: no_frank_starling).",
    )
    p.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_CASES,
        metavar="CASE",
        help="Case names to export (default: case0_rv25 case7_rv95).",
    )
    p.add_argument(
        "--sweep",
        type=Path,
        default=DEFAULT_SWEEP,
        help="Sweep root directory.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Root output directory; per-case output goes under "
             "<output-dir>/<bundle>/<case>/.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# PVD writer
# ---------------------------------------------------------------------------

def write_pvd(pvd_path: Path, entries: list[tuple[float, Path]]) -> None:
    """Write a PVD collection file linking VTU files with timestep values."""
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for timestep, file_path in entries:
        rel = file_path.relative_to(pvd_path.parent)
        lines.append(
            f'    <DataSet timestep="{timestep:.6f}" group="" part="0" '
            f'file="{escape(str(rel))}"/>'
        )
    lines.append("  </Collection>")
    lines.append("</VTKFile>")
    pvd_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# VTU writer (rank 0 only, serial mesh)
# ---------------------------------------------------------------------------

def _build_serial_vtk_mesh(checkpoint_path: Path):
    """Load mesh serially on rank 0 for VTU construction.

    Returns ``(topology, cell_types, points_ref, W1_serial, u1_serial,
    mesh_serial)`` where ``W1_serial`` / ``u1_serial`` are a serial P1
    function space and function for reading displacement at each timestep.
    """
    import dolfinx
    import dolfinx.fem
    import dolfinx.plot
    import adios4dolfinx
    import pyvista as pv

    mesh_serial = adios4dolfinx.read_mesh(checkpoint_path, MPI.COMM_SELF)
    topology, cell_types, points_ref = dolfinx.plot.vtk_mesh(
        mesh_serial, mesh_serial.topology.dim
    )
    cell_types = np.full_like(cell_types, pv.CellType.TETRA, dtype=np.uint8)

    # P2 → P1 interpolation space (same as export_production_sweep_for_animation.py)
    W2_s = dolfinx.fem.functionspace(mesh_serial, ("P", 2, (3,)))
    W1_s = dolfinx.fem.functionspace(mesh_serial, ("P", 1, (3,)))
    u2_s = dolfinx.fem.Function(W2_s)
    u1_s = dolfinx.fem.Function(W1_s)

    return topology, cell_types, points_ref, W2_s, W1_s, u2_s, u1_s, mesh_serial


def _read_serial_displacement(
    checkpoint_path: Path,
    t: float,
    u2_s,
    u1_s,
) -> np.ndarray:
    """Read displacement at time ``t`` into serial P1 function; return (N,3) array."""
    import adios4dolfinx

    adios4dolfinx.read_function(
        checkpoint_path, u2_s, time=t, name="displacement"
    )
    u1_s.interpolate(u2_s)
    return u1_s.x.array.reshape((-1, 3)).copy()


# ---------------------------------------------------------------------------
# Per-case beat replay
# ---------------------------------------------------------------------------

def export_case_beat(
    results_dir: Path,
    case_out_dir: Path,
    comm: MPI.Comm,
    logger: logging.Logger,
) -> None:
    """Replay the last beat of one simulation case and write step_NNN.vtu + beat.pvd.

    Parameters
    ----------
    results_dir:
        Root of one simulation output (must contain solver/checkpoint.bp,
        solver/Ta_solver_history.npy, solver/solver_cavity_pressure_mmHg.npy,
        simulation_params.json, and geometry/).
    case_out_dir:
        Directory where step_NNN.vtu files and beat.pvd will be written.
    comm:
        MPI communicator.
    logger:
        Caller's logger instance.
    """
    rank = comm.rank

    # ── Heavy imports (deferred so CLI parsing is fast) ─────────────────────
    import dolfinx
    import dolfinx.fem
    import dolfinx.fem.petsc
    import dolfinx.plot
    import ufl
    import adios4dolfinx
    import basix.ufl
    import pulse
    import cardiac_geometries
    import cardiac_geometries.geometry
    import pyvista as pv

    MMHG_TO_PA = 133.322

    # ── Load parameters ─────────────────────────────────────────────────────
    with open(results_dir / "simulation_params.json") as fh:
        sim_params = json.load(fh)

    solver_dir = results_dir / "solver"
    checkpoint_path = solver_dir / "checkpoint.bp"

    Ta_history = np.load(solver_dir / "Ta_solver_history.npy")

    pres_path = solver_dir / "solver_cavity_pressure_mmHg.npy"
    if not pres_path.exists():
        pres_path = solver_dir / "pressure_history.npy"
    solver_cavity_pressure_mmHg = np.load(pres_path)

    bpm = float(sim_params["BPM"])
    dt = float(sim_params["dt"])
    cycle_length = 60.0 / bpm
    steps_per_beat = int(round(cycle_length / dt))

    if rank == 0:
        logger.info(f"  BPM={bpm}, dt={dt}, steps_per_beat={steps_per_beat}")

    # ── Load mesh + fibers from checkpoint (parallel) ───────────────────────
    mesh = adios4dolfinx.read_mesh(checkpoint_path, comm)
    ffun = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="ffun")

    q_el = basix.ufl.quadrature_element(
        mesh.topology.cell_name(), value_shape=(3,), degree=6
    )
    Q_vec = dolfinx.fem.functionspace(mesh, q_el)

    f0_quad = dolfinx.fem.Function(Q_vec)
    s0_quad = dolfinx.fem.Function(Q_vec)
    adios4dolfinx.read_function(checkpoint_path, f0_quad, time=0.0, name="f0")
    adios4dolfinx.read_function(checkpoint_path, s0_quad, time=0.0, name="s0")

    # longitudinal fiber for proxy strain (ll proxy)
    l0_quad: Optional[dolfinx.fem.Function] = None
    try:
        l0_quad = dolfinx.fem.Function(Q_vec)
        adios4dolfinx.read_function(checkpoint_path, l0_quad, time=0.0, name="l0")
        if rank == 0:
            logger.info("  l0 loaded for longitudinal proxy")
    except Exception:
        if rank == 0:
            logger.warning("  l0 not found — proxy_PLV_ll will be zero")

    # ── Reconstruct pulse problem (mirrors compute_per_cell.py) ─────────────
    geo_dir = results_dir / "geometry"
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, geo_dir)
    geometry_obj = pulse.HeartGeometry(
        mesh=mesh,
        facet_tags=ffun,
        markers=geo.markers,
        metadata={"quadrature_degree": 6},
    )

    mat_params_raw = sim_params["material_params"]
    material_params = {
        k: pulse.Variable(entry["value"], entry["unit"])
        for k, entry in mat_params_raw.items()
    }
    material = pulse.HolzapfelOgden(
        f0=f0_quad, s0=s0_quad, **material_params
    )

    # Active model (same Frank-Starling detection logic as compute_per_cell.py)
    _fs_meta = sim_params.get("frank_starling", {})
    _fs_mode = str(_fs_meta.get("mode", "")).lower()
    USE_FRANK_STARLING = bool(_fs_meta.get("enabled", True))
    FS_RELAX_TAU_S = float(_fs_meta.get("relaxation_tau_s", 0.0) or 0.0)
    if _fs_mode in ("preload_only", "instantaneous", "relaxation"):
        FS_PRELOAD_ONLY = _fs_mode == "preload_only"
    else:
        FS_PRELOAD_ONLY = os.environ.get("FS_PRELOAD_ONLY", "0") == "1"
    if not USE_FRANK_STARLING:
        FS_PRELOAD_ONLY = False
        FS_RELAX_TAU_S = 0.0

    Ta = pulse.Variable(
        dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)),
        "kPa",
    )
    if USE_FRANK_STARLING:
        fs_kwargs: dict = {}
        if FS_RELAX_TAU_S > 0:
            fs_kwargs["relaxation_tau"] = FS_RELAX_TAU_S
        elif FS_PRELOAD_ONLY:
            fs_kwargs["preload_only"] = True
        active_model = pulse.FrankStarlingActiveStress(
            f0=f0_quad, activation=Ta, **fs_kwargs
        )
    else:
        active_model = pulse.ActiveStress(f0_quad, activation=Ta)

    comp_model = (
        pulse.compressibility.Incompressible()
        if sim_params["incompressible"]
        else pulse.compressibility.Compressible2()
    )
    cardiac_model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    alpha_epi = sim_params["alpha_epi"]
    alpha_base = sim_params["alpha_base"]
    robin_epi = pulse.RobinBC(
        value=pulse.Variable(
            dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha_epi)),
            "Pa / m",
        ),
        marker=geometry_obj.markers["EPI"][0],
    )
    robin_base = pulse.RobinBC(
        value=pulse.Variable(
            dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha_base)),
            "Pa / m",
        ),
        marker=geometry_obj.markers["BASE"][0],
    )

    def dirichlet_bc(V_space):
        facets = geometry_obj.facet_tags.find(geometry_obj.markers["BASE"][0])
        base_mode = sim_params.get("base_dirichlet", "x")
        if base_mode == "full" or sim_params["incompressible"]:
            dofs = dolfinx.fem.locate_dofs_topological(V_space, 2, facets)
            u_zero = dolfinx.fem.Function(V_space)
            u_zero.x.array[:] = 0.0
            return [dolfinx.fem.dirichletbc(u_zero, dofs)]
        V_x = V_space.sub(0)
        dofs = dolfinx.fem.locate_dofs_topological(V_x, 2, facets)
        return [dolfinx.fem.dirichletbc(0.0, dofs, V_x)]

    base_dirichlet = sim_params.get("base_dirichlet", "x")
    dirichlet_bcs = () if base_dirichlet == "none" else (dirichlet_bc,)

    lv_marker = "LV" if "LV" in geometry_obj.markers else "ENDO_LV"
    rv_marker = "RV" if "RV" in geometry_obj.markers else "ENDO_RV"
    lv_volume = dolfinx.fem.Constant(
        mesh, dolfinx.default_scalar_type(sim_params["lvv_unloaded_m3"])
    )
    rv_volume = dolfinx.fem.Constant(
        mesh, dolfinx.default_scalar_type(sim_params["rvv_unloaded_m3"])
    )
    cavities = [
        pulse.problem.Cavity(marker=lv_marker, volume=lv_volume),
        pulse.problem.Cavity(marker=rv_marker, volume=rv_volume),
    ]
    bcs = pulse.BoundaryConditions(
        robin=[robin_epi, robin_base], dirichlet=dirichlet_bcs
    )
    problem = pulse.problem.StaticProblem(
        model=cardiac_model,
        geometry=geometry_obj,
        bcs=bcs,
        cavities=cavities,
        parameters={"mesh_unit": sim_params["mesh_unit"], "u_space": "P_2"},
    )
    if USE_FRANK_STARLING:
        active_model.register(problem.u)

    if rank == 0:
        logger.info("  Problem reconstructed")

    # ── DG0 forms for per-cell work and proxy ───────────────────────────────
    u = problem.u
    I3 = ufl.Identity(3)
    F = ufl.variable(ufl.grad(u) + I3)
    C = ufl.variable(F.T * F)
    E_ufl = 0.5 * (C - I3)

    S_tot_ufl = cardiac_model.S(C)

    q_el_tensor = basix.ufl.quadrature_element(
        mesh.topology.cell_name(), value_shape=(3, 3), degree=6
    )
    W_tensor = dolfinx.fem.functionspace(mesh, q_el_tensor)

    E_cur = dolfinx.fem.Function(W_tensor, name="E_cur")
    E_prev = dolfinx.fem.Function(W_tensor, name="E_prev")
    S_prev = dolfinx.fem.Function(W_tensor, name="S_prev")

    points_tensor = W_tensor.element.interpolation_points
    expr_E = dolfinx.fem.Expression(E_ufl, points_tensor)
    expr_S_total = dolfinx.fem.Expression(S_tot_ufl, points_tensor)

    V_DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    v_dg0 = ufl.TestFunction(V_DG0)

    dx_q = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})
    dE = E_cur - E_prev

    # Total work density: trapezoidal 0.5*(S_new + S_old):dE
    wd_total = 0.5 * ufl.inner(S_tot_ufl + S_prev, dE)
    form_w_total = dolfinx.fem.form(wd_total * v_dg0 * dx_q)

    # Proxy: P_LV * d(epsilon_ll) — longitudinal Green-Lagrange strain increment.
    # If l0 is unavailable, form_deps_ll stays None and cum_ps stays zero.
    form_deps_ll = None
    if l0_quad is not None:
        l0_norm = ufl.sqrt(ufl.inner(l0_quad, l0_quad))
        l0_unit = l0_quad / l0_norm
        deps_ll = ufl.inner(ufl.dot(dE, l0_unit), l0_unit)
        form_deps_ll = dolfinx.fem.form(deps_ll * v_dg0 * dx_q)

    # Cell volumes (assembled once; time-independent in Lagrangian frame)
    form_vol = dolfinx.fem.form(
        dolfinx.fem.Constant(mesh, 1.0) * v_dg0 * dx_q
    )
    cell_vols_vec = dolfinx.fem.assemble_vector(form_vol)
    imap_3 = mesh.topology.index_map(3)
    n_local = imap_3.size_local
    cell_volumes = cell_vols_vec.array[:n_local].copy()
    # guard divide-by-zero with np.nan (as specified)
    cell_volumes_safe = np.where(cell_volumes > 0, cell_volumes, np.nan)

    if rank == 0:
        logger.info("  DG0 forms compiled")

    # ── Determine last beat timestep slice ───────────────────────────────────
    timestamps = adios4dolfinx.read_timestamps(checkpoint_path, comm, "displacement")
    n_steps = len(timestamps)
    n_beats = n_steps // steps_per_beat
    start_step = (n_beats - 1) * steps_per_beat
    end_step = n_beats * steps_per_beat
    beat_steps = list(range(start_step, end_step))
    n_beat_steps = len(beat_steps)

    if rank == 0:
        logger.info(
            f"  Replaying beat {n_beats - 1}: steps {start_step}–{end_step - 1} "
            f"({n_beat_steps} steps, "
            f"t={timestamps[start_step]:.4f}–{timestamps[end_step - 1]:.4f}s)"
        )

    # ── Serial mesh on rank 0 for VTU deformed-mesh writing ─────────────────
    # Each step, rank 0 reads displacement via COMM_SELF to build the pyvista
    # grid. This avoids the complexity of gathering partitioned P1 DOF arrays
    # across MPI ranks (shared DOFs at partition boundaries).
    if rank == 0:
        (
            topology_vtk, cell_types_vtk, points_ref,
            _u2_s, _W1_s, _u2_s_fn, _u1_s_fn, _mesh_s,
        ) = _build_serial_vtk_mesh(checkpoint_path)
        logger.info(
            f"  Serial VTK mesh: {_mesh_s.geometry.x.shape[0]} nodes, "
            f"{topology_vtk.shape[0]} cells"
        )
    else:
        topology_vtk = cell_types_vtk = points_ref = None
        _u2_s_fn = _u1_s_fn = None

    # ── Preload Frank-Starling at ED if needed ───────────────────────────────
    if FS_PRELOAD_ONLY:
        adios4dolfinx.read_function(
            checkpoint_path, problem.u, time=timestamps[start_step],
            name="displacement",
        )
        active_model.freeze_at(problem.u)

    # ── Output directory ─────────────────────────────────────────────────────
    if rank == 0:
        case_out_dir.mkdir(parents=True, exist_ok=True)

    # ── Replay loop ──────────────────────────────────────────────────────────
    cum_w = np.zeros(n_local)   # J per cell (true internal work), local partition
    cum_ps = np.zeros(n_local)  # J per cell (P_LV * d_eps_ll proxy), local partition
    has_previous = False
    p_LV_prev = 0.0

    pvd_entries: list[tuple[float, Path]] = []

    for beat_idx, global_step in enumerate(beat_steps):
        t = float(timestamps[global_step])

        # Load displacement into the parallel problem (P2 field for mechanics)
        adios4dolfinx.read_function(
            checkpoint_path, problem.u, time=t, name="displacement"
        )
        Ta.assign(float(np.mean(np.atleast_1d(Ta_history[global_step]))))

        # Interpolate current strain tensor into quadrature space
        E_cur.interpolate(expr_E)

        if has_previous:
            # Accumulate true work increment (trapezoidal, via DG0 form)
            w_vec = dolfinx.fem.assemble_vector(form_w_total)
            cum_w += w_vec.array[:n_local]

            # Accumulate proxy increment: P_LV_avg * deps_ll per cell
            p_LV = float(solver_cavity_pressure_mmHg[global_step, 0]) * MMHG_TO_PA
            p_LV_avg = 0.5 * (p_LV + p_LV_prev)

            if form_deps_ll is not None:
                deps_ll_vec = dolfinx.fem.assemble_vector(form_deps_ll)
                cum_ps += p_LV_avg * deps_ll_vec.array[:n_local]

            p_LV_prev = p_LV
        else:
            p_LV_prev = (
                float(solver_cavity_pressure_mmHg[global_step, 0]) * MMHG_TO_PA
            )

        # Shift state: current → previous for next step's trapezoidal average
        E_prev.x.array[:] = E_cur.x.array[:]
        S_prev.interpolate(expr_S_total)
        has_previous = True

        if USE_FRANK_STARLING and FS_RELAX_TAU_S > 0:
            active_model.advance(dt)

        # ── Gather per-cell arrays to rank 0 ────────────────────────────────
        gathered_w = comm.gather(np.ascontiguousarray(cum_w), root=0)
        gathered_ps = comm.gather(np.ascontiguousarray(cum_ps), root=0)
        gathered_vols = comm.gather(np.ascontiguousarray(cell_volumes_safe), root=0)

        # ── Write VTU on rank 0 ──────────────────────────────────────────────
        if rank == 0:
            g_w = np.concatenate(gathered_w)
            g_ps = np.concatenate(gathered_ps)
            g_vols = np.concatenate(gathered_vols)

            # Per-cell densities (Pa = J / m³); guard with np.nan for zero-volume cells
            cum_work_density_Pa = g_w / g_vols
            cum_ps_density_Pa = g_ps / g_vols

            # Load displacement serially for deformed mesh vertices
            u_s = _read_serial_displacement(checkpoint_path, t, _u2_s_fn, _u1_s_fn)

            # Build deformed pyvista grid
            grid = pv.UnstructuredGrid(
                topology_vtk, cell_types_vtk, points_ref + u_s
            )
            grid.cell_data["cum_work_density_Pa"] = cum_work_density_Pa.astype(np.float32)
            grid.cell_data["cum_ps_density_Pa"] = cum_ps_density_Pa.astype(np.float32)
            grid.field_data["beat_step"] = np.array([beat_idx], dtype=np.int32)
            grid.field_data["beat_time_s"] = np.array([t], dtype=np.float64)

            vtu_name = f"step_{beat_idx:03d}.vtu"
            vtu_path = case_out_dir / vtu_name
            grid.save(vtu_path)

            # Beat phase ∈ [0, 1] as PVD timestep
            beat_phase = float(beat_idx) / max(1, n_beat_steps - 1)
            pvd_entries.append((beat_phase, vtu_path))

            if beat_idx % 50 == 0 or beat_idx == n_beat_steps - 1:
                logger.info(
                    f"  step {beat_idx:03d}/{n_beat_steps} | t={t:.4f}s | "
                    f"cum_W_sum={g_w.sum():.4e} J"
                )

    # Write PVD on rank 0
    if rank == 0:
        pvd_path = case_out_dir / "beat.pvd"
        write_pvd(pvd_path, pvd_entries)
        logger.info(f"  Wrote {len(pvd_entries)} VTU files + {pvd_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.rank

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("beat_anim")

    bundle = args.bundle
    cases = args.cases
    sweep_root = args.sweep
    out_dir = args.output_dir

    if rank == 0:
        logger.info(f"export_beat_animation: bundle={bundle}, cases={cases}")
        logger.info(f"  sweep root: {sweep_root}")
        logger.info(f"  output dir: {out_dir}")

    for case_name in cases:
        results_dir = sweep_root / bundle / case_name
        case_out_dir = out_dir / bundle / case_name

        if rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Case: {bundle}/{case_name}")
            logger.info(f"  results_dir: {results_dir}")
            logger.info(f"  case_out_dir: {case_out_dir}")

        if not results_dir.exists():
            if rank == 0:
                logger.error(f"  SKIP — results_dir does not exist: {results_dir}")
            continue

        export_case_beat(results_dir, case_out_dir, comm, logger)

    if rank == 0:
        logger.info("\nDone.")


if __name__ == "__main__":
    main()
