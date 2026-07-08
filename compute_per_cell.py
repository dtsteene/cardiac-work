#!/usr/bin/env python3
"""
compute_per_cell.py — Per-Cell Work Density from Simulation Checkpoints

Replays the last cardiac cycle from checkpoint and computes work density
per mesh cell using DG0 test function assembly. Also computes the Euclidean
transventricular coordinate (tau) and study region definitions.

Output: per_cell_data.npz containing per-cell work arrays, tau, and region masks.

This script reuses the same problem setup as postprocess_metrics.py but adds
per-cell integration via the DG0 test function trick:
    W_per_cell = assemble_vector(form(work_density * v_dg0 * dx))
where v_dg0 is a DG0 TestFunction acting as a spatial partition of unity.

Usage:
    python compute_per_cell.py <results_directory>
    mpirun -n 4 python compute_per_cell.py <results_directory>
"""

import json
import os
import sys
import logging
from pathlib import Path

import numpy as np
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import adios4dolfinx
import basix.ufl
import pulse
import cardiac_geometries
import cardiac_geometries.geometry
from scipy.spatial import cKDTree
from petsc4py import PETSc
from clinical_frame import tangent_project_longitudinal
from geometry_utils import closest_points_to_surface as _closest_points_to_surface

# pyvista is used for point-to-facet distance (more accurate than
# nearest-vertex on coarse meshes).
try:
    import pyvista as pv
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False


def _main():  # noqa: C901 — long but linear script body
    # ── Parse Arguments ──────────────────────────────────────────────────────────

    import argparse
    _parser = argparse.ArgumentParser(description="Per-cell work from checkpoint.")
    _parser.add_argument("results_dir", type=Path)
    _parser.add_argument("--retag-septum", action="store_true", default=False)
    _parser.add_argument("--geometry-fields", type=Path, default=None,
                         help="CANONICAL atlas mode: load canonical septum tags from "
                              "a precomputed geometry_fields.npz (emitted automatically "
                              "by geometry_generator.py alongside geometry.bp). Tags are "
                              "propagated to this case via the u_pre permutation, giving "
                              "identical cell labels across all cases in a spectrum. "
                              "RECOMMENDED for sweep analyses. Mutually exclusive with "
                              "--tag-at-unloaded.")
    _parser.add_argument("--tag-at-unloaded", action="store_true",
                         help="LEGACY mode: compute septum tags on this case's unloaded "
                              "stress-free reference mesh (checkpoint mesh at u=0). This "
                              "is anatomically meaningless — the unloaded state never "
                              "occurs in vivo. Kept only for reproducibility of old runs. "
                              "See session update 2026-04-13 in transmural_work_profiles.md.")
    _parser.add_argument("--d-sum-max-mm", type=float, default=22.0,
                         help="Envelope upper bound on d_sum (mm).")
    _parser.add_argument("--beat", type=int, default=None,
                         help="Replay a specific beat (0-indexed) instead of the last beat. "
                              "Useful for per-beat convergence analysis. Output is written "
                              "to per_cell_data_beat{N}.npz instead of per_cell_data.npz.")
    _parser.add_argument("--output-tag", type=str, default="",
                         help="Optional tag inserted into the output filename, e.g. "
                              "--output-tag ed → per_cell_data_ed.npz (or "
                              "per_cell_data_ed_beat{N}.npz with --all-beats). "
                              "Lets you run multiple tagging modes without overwriting.")
    _parser.add_argument("--all-beats", action="store_true",
                         help="Replay EVERY beat in the checkpoint and save one per_cell_data_beat{N}.npz "
                              "per beat. Amortizes the mesh/fiber/form setup cost across all beats. "
                              "Mutually exclusive with --beat.")
    _args = _parser.parse_args()

    if _args.beat is not None and _args.all_beats:
        _parser.error("--beat and --all-beats are mutually exclusive")

    # Determine tagging mode. Default is tag-at-ED: compute distances on the case's
    # deformed mesh at the start of the last beat (≈ end-diastole, the clinically
    # relevant reference frame). This is case-specific but anatomically meaningful.
    #
    # Three modes total:
    #   --geometry-fields PATH  → canonical atlas (same cells for every case)
    #   (default, no flags)     → per-case at end-diastole (deformed mesh at t_start_last_beat)
    #   --tag-at-unloaded       → per-case at unloaded (legacy, not clinically meaningful)
    if _args.geometry_fields is not None and _args.tag_at_unloaded:
        _parser.error("--geometry-fields and --tag-at-unloaded are mutually exclusive")
    _mode_canonical = _args.geometry_fields is not None
    _mode_unloaded = _args.tag_at_unloaded
    _mode_ed = not _mode_canonical and not _mode_unloaded

    results_dir = _args.results_dir.resolve()
    comm = MPI.COMM_WORLD
    rank = comm.rank

    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    logger = logging.getLogger("per_cell")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 1: Problem Setup (mirrors postprocess_metrics.py)
    # ════════════════════════════════════════════════════════════════════════════

    # ── Load Parameters ──────────────────────────────────────────────────────────
    sim_params_path = results_dir / "simulation_params.json"
    with open(sim_params_path) as f:
        sim_params = json.load(f)
    if rank == 0:
        logger.info(f"Loaded sim params: BPM={sim_params['BPM']}, dt={sim_params['dt']}")

    solver_dir = results_dir / "solver"
    checkpoint_path = solver_dir / "checkpoint.bp"

    # ── Load Ta History ──────────────────────────────────────────────────────────
    Ta_history = np.load(solver_dir / "Ta_solver_history.npy")
    if rank == 0:
        logger.info(f"Ta history: {Ta_history.shape}")

    # ── Load Solver Cavity Pressure (Lagrange multiplier, mmHg) ─────────────────
    # This is the pressure actually applied as the FEM boundary condition at each
    # timestep. NOT the 0D ODE output (which leads by 1 step due to staggered
    # coupling). Shape: [n_steps, 2] with columns [LV, RV].
    solver_pres_path = solver_dir / "solver_cavity_pressure_mmHg.npy"
    if not solver_pres_path.exists():
        # Backwards compat: old runs saved as pressure_history.npy
        solver_pres_path = solver_dir / "pressure_history.npy"
    solver_cavity_pressure_mmHg = np.load(solver_pres_path)
    if rank == 0:
        logger.info(f"Solver cavity pressure: {solver_cavity_pressure_mmHg.shape} from {solver_pres_path.name}")

    HR_Hz = sim_params["BPM"] / 60.0
    cycle_length = 1.0 / HR_Hz

    # ── Load Mesh from Checkpoint ────────────────────────────────────────────────
    mesh = adios4dolfinx.read_mesh(checkpoint_path, comm)
    ffun = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="ffun")
    markers_mt = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="cfun")

    if rank == 0:
        logger.info(f"Mesh: {mesh.geometry.x.shape[0]} nodes")

    # Optional geometric septum retagging
    if _args.retag_septum:
        mesh.topology.create_connectivity(2, 0)
        f2v = mesh.topology.connectivity(2, 0)
        imap = mesh.topology.index_map(3)
        n_local = imap.size_local

        geo_dir = results_dir / "geometry"
        if rank == 0:
            geo_tmp = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, geo_dir)
            raw_markers = geo_tmp.markers
        else:
            raw_markers = None
        raw_markers = comm.bcast(raw_markers, root=0)

        def surface_coords_global(tag_ids):
            facets = np.hstack([ffun.find(t) for t in tag_ids])
            verts = set()
            for f in facets:
                verts.update(f2v.links(f))
            local_coords = mesh.geometry.x[np.array(sorted(verts), dtype=np.int64)] if verts else np.empty((0, 3))
            all_coords = comm.allgather(local_coords)
            return np.vstack(all_coords) if any(len(c) > 0 for c in all_coords) else np.empty((0, 3))

        lv_tags = [raw_markers.get("LV", raw_markers.get("ENDO_LV", [None]))[0]]
        rv_tags = [raw_markers.get("RV", raw_markers.get("ENDO_RV", [None]))[0]]
        epi_tags = [raw_markers.get("EPI", [None])[0]]

        lv_coords = surface_coords_global(lv_tags)
        rv_coords = surface_coords_global(rv_tags)
        epi_coords = surface_coords_global(epi_tags)

        centroids = dolfinx.mesh.compute_midpoints(mesh, 3, np.arange(n_local, dtype=np.int32))
        d_lv_arr = cKDTree(lv_coords).query(centroids)[0]
        d_rv_arr = cKDTree(rv_coords).query(centroids)[0]
        d_epi_arr = cKDTree(epi_coords).query(centroids)[0]

        is_sept = np.maximum(d_lv_arr, d_rv_arr) < d_epi_arr
        new_tags = np.where(is_sept, 3, np.where(d_lv_arr <= d_rv_arr, 1, 2)).astype(np.int32)
        markers_mt = dolfinx.mesh.meshtags(
            mesh, 3,
            np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32),
            np.concatenate([new_tags, markers_mt.values[n_local:]]),
        )
        markers_mt.name = "cfun"
        if rank == 0:
            logger.info(f"Retagged: LV={int((new_tags==1).sum())}, RV={int((new_tags==2).sum())}, Sept={int((new_tags==3).sum())}")

    # ── Load Geometry ────────────────────────────────────────────────────────────
    geo_dir = results_dir / "geometry"
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, geo_dir)
    geometry = pulse.HeartGeometry(
        mesh=mesh, facet_tags=ffun, markers=geo.markers,
        metadata={"quadrature_degree": 6},
    )

    # ── Fiber Fields ─────────────────────────────────────────────────────────────
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
    q_el = basix.ufl.quadrature_element(mesh.topology.cell_name(), value_shape=(3,), degree=6)
    Q_vec = dolfinx.fem.functionspace(mesh, q_el)

    f0_quad = dolfinx.fem.Function(Q_vec)
    s0_quad = dolfinx.fem.Function(Q_vec)
    adios4dolfinx.read_function(checkpoint_path, f0_quad, time=0.0, name="f0")
    adios4dolfinx.read_function(checkpoint_path, s0_quad, time=0.0, name="s0")

    n0_quad = None
    try:
        n0_quad = dolfinx.fem.Function(Q_vec)
        adios4dolfinx.read_function(checkpoint_path, n0_quad, time=0.0, name="n0")
    except Exception:
        pass

    l0_field = None
    try:
        l0_field = dolfinx.fem.Function(Q_vec)
        adios4dolfinx.read_function(checkpoint_path, l0_field, time=0.0, name="l0")
    except Exception:
        pass

    if rank == 0:
        logger.info(f"Fibers loaded: f0, s0, n0={'yes' if n0_quad else 'no'}, l0={'yes' if l0_field else 'no'}")

    # ── Reconstruct Problem ──────────────────────────────────────────────────────
    mat_params_raw = sim_params["material_params"]
    material_params = {}
    for k, entry in mat_params_raw.items():
        material_params[k] = pulse.Variable(entry["value"], entry["unit"])

    material = pulse.HolzapfelOgden(f0=f0_quad, s0=s0_quad, **material_params)
    # Uniform scalar Ta + FrankStarlingActiveStress, matching the forward sim.
    # Pick up the mode from simulation_params.json if present (set by
    # complete_cycle.py when the forward sim runs), else from FS_PRELOAD_ONLY
    # env var, else default to instantaneous.
    _fs_meta = sim_params.get("frank_starling", {}) if isinstance(sim_params, dict) else {}
    _fs_meta_mode = str(_fs_meta.get("mode", "")).lower()
    USE_FRANK_STARLING = bool(_fs_meta.get("enabled", True))
    FS_RELAX_TAU_S = float(_fs_meta.get("relaxation_tau_s", 0.0) or 0.0)
    if _fs_meta_mode in ("preload_only", "instantaneous", "relaxation"):
        FS_PRELOAD_ONLY = (_fs_meta_mode == "preload_only")
    else:
        FS_PRELOAD_ONLY = os.environ.get("FS_PRELOAD_ONLY", "0") == "1"
    if not USE_FRANK_STARLING:
        FS_PRELOAD_ONLY = False
        FS_RELAX_TAU_S = 0.0

    def _fs_mode_kwargs():
        if FS_RELAX_TAU_S > 0:
            return {"relaxation_tau": FS_RELAX_TAU_S}
        if FS_PRELOAD_ONLY:
            return {"preload_only": True}
        return {}

    _relax_dt = float(sim_params.get("dt", 0.001))
    if rank == 0:
        _mode = ("off (constant Ta)" if not USE_FRANK_STARLING
                 else f"relaxation (tau={FS_RELAX_TAU_S*1000:.0f}ms)" if FS_RELAX_TAU_S > 0
                 else "preload_only" if FS_PRELOAD_ONLY else "instantaneous")
        logger.info(f"Active model for replay: {_mode}")
    Ta = pulse.Variable(
        dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)),
        "kPa",
    )
    if USE_FRANK_STARLING:
        active_model = pulse.FrankStarlingActiveStress(
            f0=f0_quad, activation=Ta,
            **_fs_mode_kwargs(),
        )
    else:
        active_model = pulse.ActiveStress(f0_quad, activation=Ta)

    if sim_params["incompressible"]:
        comp_model = pulse.compressibility.Incompressible()
    else:
        comp_model = pulse.compressibility.Compressible2()

    cardiac_model = pulse.CardiacModel(material=material, active=active_model, compressibility=comp_model)

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

    base_dirichlet = sim_params.get("base_dirichlet", "x")
    dirichlet_bcs = () if base_dirichlet == "none" else (dirichlet_bc,)
    if rank == 0:
        logger.info(f"Base Dirichlet mode: {base_dirichlet}")

    volume2ml = sim_params["volume2ml"]
    lv_marker = "LV" if "LV" in geometry.markers else "ENDO_LV"
    rv_marker = "RV" if "RV" in geometry.markers else "ENDO_RV"

    lv_volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(sim_params["lvv_unloaded_m3"]))
    rv_volume = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(sim_params["rvv_unloaded_m3"]))

    cavities = [
        pulse.problem.Cavity(marker=lv_marker, volume=lv_volume),
        pulse.problem.Cavity(marker=rv_marker, volume=rv_volume),
    ]
    bcs = pulse.BoundaryConditions(robin=[robin_epi, robin_base], dirichlet=dirichlet_bcs)
    problem = pulse.problem.StaticProblem(
        model=cardiac_model, geometry=geometry, bcs=bcs, cavities=cavities,
        parameters={"mesh_unit": sim_params["mesh_unit"], "u_space": "P_2"},
    )

    # Hand FrankStarlingActiveStress the displacement so g(lambda) matches
    # the forward sim on replay. (Plain ActiveStress has no register / needs none.)
    if USE_FRANK_STARLING:
        active_model.register(problem.u)

    if rank == 0:
        logger.info("Problem reconstructed")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 2: Per-Cell DG0 Forms (the new part)
    # ════════════════════════════════════════════════════════════════════════════

    u = problem.u
    I = ufl.Identity(3)
    F = ufl.variable(ufl.grad(u) + I)
    C = ufl.variable(F.T * F)
    E_ufl = 0.5 * (C - I)

    S_tot_ufl = cardiac_model.S(C)
    S_act_ufl = cardiac_model.active.S(C, dev=True)
    S_pas_ufl = cardiac_model.material.S(C, dev=True)
    S_cmp_ufl = cardiac_model.compressibility.S(C)

    f0 = f0_quad
    s0 = s0_quad
    n0 = n0_quad
    l0 = l0_field
    if l0 is not None:
        l0_norm = ufl.sqrt(ufl.inner(l0, l0))
        l0 = l0 / l0_norm

    def proj(T, v):
        return ufl.inner(ufl.dot(T, v), v)

    # Quadrature-space functions for state tracking (same as MetricsCalculator)
    q_el_tensor = basix.ufl.quadrature_element(mesh.topology.cell_name(), value_shape=(3, 3), degree=6)
    W_tensor = dolfinx.fem.functionspace(mesh, q_el_tensor)
    q_el_scalar = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=6)
    W_scalar = dolfinx.fem.functionspace(mesh, q_el_scalar)

    E_cur = dolfinx.fem.Function(W_tensor, name="E_cur")
    E_prev = dolfinx.fem.Function(W_tensor, name="E_prev")
    S_prev = dolfinx.fem.Function(W_tensor, name="S_prev")

    # Expressions for interpolation
    points_tensor = W_tensor.element.interpolation_points
    expr_E = dolfinx.fem.Expression(E_ufl, points_tensor)
    expr_S_total = dolfinx.fem.Expression(S_tot_ufl, points_tensor)

    # DG0 function space for per-cell extraction
    V_DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    v_dg0 = ufl.TestFunction(V_DG0)

    # Cellwise geometric frame for candidate clinical radial/circumferential
    # pressure-strain proxies. The radial direction is nearest endocardium -> nearest
    # epicardium; later, before projection, it is orthogonalized against the saved
    # longitudinal direction l0. This keeps the frame geometric/clinical rather than
    # tied to the material sheet-normal direction n0.
    V_DG0_vec = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
    mesh.topology.create_connectivity(2, 0)
    mesh.topology.create_connectivity(3, 0)
    f2v_conn = mesh.topology.connectivity(2, 0)
    imap_3 = mesh.topology.index_map(3)
    n_local_cells = imap_3.size_local
    local_cells = np.arange(n_local_cells, dtype=np.int32)
    centroids_ref = dolfinx.mesh.compute_midpoints(mesh, 3, local_cells)


    def _surface_triangles_global(tag_ids):
        local = []
        for tag in tag_ids:
            for facet in ffun.find(tag):
                vert_ids = f2v_conn.links(facet)
                if len(vert_ids) == 3:
                    local.append(mesh.geometry.x[vert_ids])
        local_arr = np.asarray(local, dtype=float) if local else np.empty((0, 3, 3), dtype=float)
        gathered = comm.allgather(local_arr)
        nonempty = [arr for arr in gathered if len(arr) > 0]
        if not nonempty:
            return np.empty((0, 3, 3), dtype=float)
        return np.concatenate(nonempty, axis=0)


    def _normalize_rows(values):
        norms = np.linalg.norm(values, axis=1)
        out = np.zeros_like(values)
        good = norms > 1e-12
        out[good] = values[good] / norms[good, None]
        if (~good).any() and rank == 0:
            logger.warning(f"Geometric frame: {int((~good).sum())} near-zero radial vectors")
        return out


    def _dg0_vector_function(values, name):
        fn = dolfinx.fem.Function(V_DG0_vec, name=name)
        arr = fn.x.array.reshape(-1, 3)
        arr[:] = 0.0
        arr[:len(values)] = values
        fn.x.scatter_forward()
        return fn


    _lv_tags_frame = [geometry.markers.get("LV", geometry.markers.get("ENDO_LV"))[0]]
    _rv_tags_frame = [geometry.markers.get("RV", geometry.markers.get("ENDO_RV"))[0]]
    _epi_tags_frame = [geometry.markers["EPI"][0]]

    _lv_tri = _surface_triangles_global(_lv_tags_frame)
    _rv_tri = _surface_triangles_global(_rv_tags_frame)
    _epi_tri = _surface_triangles_global(_epi_tags_frame)

    _cp_lv, _d_lv_frame = _closest_points_to_surface(centroids_ref, _lv_tri)
    _cp_rv, _d_rv_frame = _closest_points_to_surface(centroids_ref, _rv_tri)
    _cp_epi, _ = _closest_points_to_surface(centroids_ref, _epi_tri)
    _nearest_lv = _d_lv_frame <= _d_rv_frame
    _cp_endo = np.where(_nearest_lv[:, None], _cp_lv, _cp_rv)
    _r_endo_epi_values = _normalize_rows(_cp_epi - _cp_endo)
    radial_endo_epi_dg0 = _dg0_vector_function(_r_endo_epi_values, "radial_endo_to_epi")

    if rank == 0:
        logger.info("Built geometric radial candidate: nearest endocardium -> epicardium")

    if l0 is not None:
        l0 = tangent_project_longitudinal(l0, radial_endo_epi_dg0)
        if rank == 0:
            logger.info("Projected l0 into the wall tangent plane for longitudinal strain")

    # dx with same quadrature as the tensor space
    dx_q = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})

    # Work density expressions (trapezoidal: 0.5 * (S_new + S_old) : dE)
    dE = E_cur - E_prev

    wd_total = 0.5 * ufl.inner(S_tot_ufl + S_prev, dE)
    wd_fiber = 0.5 * (proj(S_tot_ufl, f0) + proj(S_prev, f0)) * proj(dE, f0)
    wd_sheet = 0.5 * (proj(S_tot_ufl, s0) + proj(S_prev, s0)) * proj(dE, s0)

    wd_normal = None
    if n0 is not None:
        wd_normal = 0.5 * (proj(S_tot_ufl, n0) + proj(S_prev, n0)) * proj(dE, n0)

    # Strain increment projections (for proxy work: P * d_eps_ff per cell)
    deps_ff = proj(dE, f0)  # scalar: fiber strain increment
    deps_ll = None
    if l0 is not None:
        deps_ll = proj(dE, l0)

    deps_radial = None
    deps_circ = None
    if l0 is not None:
        def unit(v):
            return v / ufl.sqrt(ufl.inner(v, v) + 1e-16)

        radial0 = unit(radial_endo_epi_dg0 - ufl.inner(radial_endo_epi_dg0, l0) * l0)
        circ0 = unit(ufl.cross(l0, radial0))
        deps_radial = proj(dE, radial0)
        deps_circ = proj(dE, circ0)
    elif rank == 0:
        logger.warning("l0 not loaded — radial/circumferential clinical-frame proxies unavailable")

    # Compile per-cell forms
    # Each form: assemble_vector gives a vector of length n_cells
    form_w_total = dolfinx.fem.form(wd_total * v_dg0 * dx_q)
    form_w_ff = dolfinx.fem.form(wd_fiber * v_dg0 * dx_q)
    form_w_ss = dolfinx.fem.form(wd_sheet * v_dg0 * dx_q)
    form_w_nn = dolfinx.fem.form(wd_normal * v_dg0 * dx_q) if wd_normal is not None else None
    form_deps_ff = dolfinx.fem.form(deps_ff * v_dg0 * dx_q)
    form_deps_ll = dolfinx.fem.form(deps_ll * v_dg0 * dx_q) if deps_ll is not None else None
    form_deps_radial = dolfinx.fem.form(deps_radial * v_dg0 * dx_q) if deps_radial is not None else None
    form_deps_circ = dolfinx.fem.form(deps_circ * v_dg0 * dx_q) if deps_circ is not None else None

    # Cell volume form (assemble once)
    form_vol = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * v_dg0 * dx_q)

    # ── Scalar cross-validation forms ───────────────────────────────────────────
    # These compute the same quantities as the DG0 forms above but via scalar
    # integration (assemble_scalar). At each timestep, we can verify that
    #   sum(assemble_vector(wd_total * v_dg0 * dx)) == assemble_scalar(wd_total * dx)
    # This is a machine-precision sanity check proving the DG0 test-function trick
    # is equivalent to direct integration. See validate_canonical_tagging TEST 6.
    form_w_total_scalar = dolfinx.fem.form(wd_total * dx_q)
    form_w_ff_scalar = dolfinx.fem.form(wd_fiber * dx_q)

    if rank == 0:
        logger.info("Per-cell DG0 forms compiled (with scalar cross-check forms)")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 3: Tau + Study Region
    # ════════════════════════════════════════════════════════════════════════════

    mesh.topology.create_connectivity(2, 0)
    mesh.topology.create_connectivity(3, 0)
    f2v_conn = mesh.topology.connectivity(2, 0)
    imap_3 = mesh.topology.index_map(3)
    n_local_cells = imap_3.size_local


    local_cells = np.arange(n_local_cells, dtype=np.int32)

    # Surface coordinates (gathered across MPI ranks)
    geo_dir_path = results_dir / "geometry"
    if rank == 0:
        geo_for_markers = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, geo_dir_path)
        raw_markers_dict = geo_for_markers.markers
    else:
        raw_markers_dict = None
    raw_markers_dict = comm.bcast(raw_markers_dict, root=0)

    def _surface_coords_global(tag_ids):
        facets = np.hstack([ffun.find(t) for t in tag_ids])
        verts = set()
        for f_idx in facets:
            verts.update(f2v_conn.links(f_idx))
        local_coords = mesh.geometry.x[np.array(sorted(verts), dtype=np.int64)] if verts else np.empty((0, 3))
        all_coords = comm.allgather(local_coords)
        return np.vstack(all_coords) if any(len(c) > 0 for c in all_coords) else np.empty((0, 3))

    _lv_tags = [raw_markers_dict.get("LV", raw_markers_dict.get("ENDO_LV", [None]))[0]]
    _rv_tags = [raw_markers_dict.get("RV", raw_markers_dict.get("ENDO_RV", [None]))[0]]
    _epi_tags = [raw_markers_dict.get("EPI", [None])[0]]

    lv_surf = _surface_coords_global(_lv_tags)
    rv_surf = _surface_coords_global(_rv_tags)
    epi_surf = _surface_coords_global(_epi_tags)

    centroids = dolfinx.mesh.compute_midpoints(mesh, 3, local_cells)

    # ── ED-deformation for tag-at-ED mode ────────────────────────────────────────
    # When the default mode (tag at end-diastole) is active, we shift the centroids
    # and surface vertex positions used for distance computation by u(t_ED), where
    # t_ED is the first timestep of the last beat. The FEM mesh object itself is
    # NOT modified — the deformation only affects the positions passed to pyvista's
    # distance queries. Work forms continue to compute on the Lagrangian reference
    # with u(t) as the deformation field, unaffected by this tag-time choice.
    #
    # We need:
    #   1. u at cell centroids  (for centroids_ED = centroids + u_at_centroids)
    #   2. u at mesh vertex positions  (for surface_ED = surface + u_at_vertices)
    #
    # Option (1) uses DG0 vector interpolation (same trick as canonical u_pre path).
    # Option (2) uses u.eval(points, cells) via the bounding box tree.
    if _mode_ed:
        if rank == 0:
            logger.info("Tag-at-ED mode: loading u at start of last beat")

        # Figure out the target time (start of last beat ≈ ED by periodicity)
        _timestamps = adios4dolfinx.read_timestamps(checkpoint_path, comm, "displacement")
        _n_steps = len(_timestamps)
        _steps_per_beat = int(round(cycle_length / sim_params["dt"]))
        _start_step = max(0, _n_steps - _steps_per_beat)
        _t_ED = float(_timestamps[_start_step])
        if rank == 0:
            logger.info(f"  ED time: {_t_ED:.4f} s (step {_start_step} of {_n_steps})")

        # Load u at t_ED into a temporary Function (DOES NOT corrupt problem.u —
        # the replay loop later re-loads at each timestep anyway)
        _V_u = problem.u.function_space
        _u_ED = dolfinx.fem.Function(_V_u)
        adios4dolfinx.read_function(checkpoint_path, _u_ED, time=_t_ED, name="displacement")

        # (1) u at cell centroids via DG0 vector interpolation
        _V_DG0v = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
        _u_ED_dg0 = dolfinx.fem.Function(_V_DG0v)
        _u_ED_dg0.interpolate(_u_ED)
        _u_at_centroids = _u_ED_dg0.x.array.reshape(-1, 3)[:n_local_cells]
        centroids = centroids + _u_at_centroids  # ED-deformed centroid positions

        # (2) u at mesh vertex positions via u.eval over the bounding-box tree.
        # _surface_coords_global gave us a GLOBAL array of vertex coords — we need
        # to add u(vertex) to each of these. Since all ranks allgather the surface
        # coordinates before building the polydata, each rank can evaluate u on its
        # OWN local vertices and contribute. Simpler approach: each rank evaluates
        # u on the entire GLOBAL surface via the local bounding box tree, accepting
        # that vertices outside the local partition will return NaN; we then reduce.
        #
        # Even simpler: deform the local mesh.geometry.x before surface extraction,
        # so _surface_coords_global picks up the already-deformed positions. This is
        # only safe if we DO NOT use mesh.geometry.x for any other purpose after.
        # To keep the mesh intact, we instead save a copy and restore after surface
        # extraction.
        import dolfinx.geometry as _geo
        _bb_tree = _geo.bb_tree(mesh, mesh.topology.dim)
        _local_vertex_x = mesh.geometry.x.copy()  # save unloaded positions
        _cands = _geo.compute_collisions_points(_bb_tree, _local_vertex_x)
        _colls = _geo.compute_colliding_cells(mesh, _cands, _local_vertex_x)
        # For each vertex, pick the first containing cell (there's always at least one)
        _first_cells = np.array(
            [_colls.links(i)[0] if len(_colls.links(i)) > 0 else -1
             for i in range(len(_local_vertex_x))],
            dtype=np.int32,
        )
        _valid_mask = _first_cells >= 0
        _u_at_verts = np.zeros_like(_local_vertex_x)
        if _valid_mask.any():
            _u_at_verts[_valid_mask] = _u_ED.eval(
                _local_vertex_x[_valid_mask], _first_cells[_valid_mask]
            )

        # Temporarily deform mesh.geometry.x to ED positions so surface extraction
        # picks up the deformed vertex coordinates. Restored immediately after.
        mesh.geometry.x[:] = _local_vertex_x + _u_at_verts
        lv_surf = _surface_coords_global(_lv_tags)
        rv_surf = _surface_coords_global(_rv_tags)
        epi_surf = _surface_coords_global(_epi_tags)
        mesh.geometry.x[:] = _local_vertex_x  # restore unloaded positions

        if rank == 0:
            max_disp = float(np.linalg.norm(_u_at_verts, axis=1).max())
            logger.info(f"  Max vertex displacement at t_ED: {max_disp*1000:.2f} mm")
            logger.info("  Centroids and surfaces deformed to ED for distance computation")
            logger.info("  (mesh.geometry.x restored to unloaded state; FEM forms unaffected)")

    elif _mode_unloaded:
        if rank == 0:
            logger.warning("=" * 70)
            logger.warning("LEGACY mode: tagging at unloaded reference (u=0).")
            logger.warning("This is anatomically meaningless — the unloaded state never occurs")
            logger.warning("in vivo. Kept only for reproducibility of old runs. Consider")
            logger.warning("omitting the flag to use the default tag-at-ED mode instead.")
            logger.warning("=" * 70)
    # If _mode_canonical, centroids and surfaces will be superseded by canonical
    # fields below. We keep centroids as computed so that if canonical loading
    # fails, we fall through to the per-case path.

    # ── Canonical anatomical tagging via u_pre permutation ──────────────────────
    # If --geometry-fields is provided, load canonical distances and tags from a
    # precomputed reference (computed on the zero-strain shared mesh, in cg cell
    # ordering). The cg → checkpoint cell permutation is found by:
    #   1. Load cg mesh via cardiac_geometries (same as geometry object)
    #   2. Read u_pre from this case's prestress_inverse.bp onto the cg mesh
    #      (written via write_function_on_input_mesh, so it round-trips cleanly)
    #   3. cg_centroids + u_pre ≈ checkpoint_centroids (to ~1 mm DG0 interp error)
    #   4. Nearest-neighbor match gives a bijection cg → ckpt
    #
    # Verified empirically on 8 spectrum cases: drift 0.3-1.1 mm (well below
    # inter-cell spacing ~3.5 mm), bijection holds on 6/8 cases and has at most
    # 1 duplicate on the other 2. Canonical geometric count stays 333 across all
    # cases, compared to 164-373 variance with per-case tagging.
    #
    # IMPORTANT: the nearest-centroid matching approach without u_pre (older
    # version) was BROKEN — prestress drift was same order as inter-cell spacing,
    # giving ambiguous matches. u_pre is the missing piece.
    _ckpt_to_cg_idx_global = None  # cg → ckpt cell index map
    if _mode_canonical and rank == 0:
        logger.info("=" * 70)
        logger.info("Canonical anatomical tagging mode (via u_pre permutation)")
        logger.info("Fields loaded once from canonical reference, applied to all cases")
        logger.info("via cg → checkpoint cell bijection. Guaranteed consistent across")
        logger.info("the spectrum by construction.")
        logger.info("=" * 70)
    if _mode_canonical:
        if rank == 0:
            logger.info(f"Loading canonical reference fields from {_args.geometry_fields}")
        _gf = np.load(_args.geometry_fields)

        # Step 1: Load the canonical shared geometry (same file used by cardiac_geometries
        # in line 168's geometry loading). This mesh has THE canonical cell ordering
        # that matches the centroids stored in _gf["centroids"].
        _canon_geo_dir = Path(_args.geometry_fields).parent
        import cardiac_geometries as _cg_mod
        _cg_geo = _cg_mod.geometry.Geometry.from_folder(MPI.COMM_SELF, _canon_geo_dir)
        _cg_mesh = _cg_geo.mesh
        _n_cg = _cg_mesh.topology.index_map(3).size_local
        _cg_cells = np.arange(_n_cg, dtype=np.int32)
        _cg_centroids = dolfinx.mesh.compute_midpoints(_cg_mesh, 3, _cg_cells)

        # Sanity: cg mesh centroids must match the npz centroids exactly
        _cg_drift = np.abs(_cg_centroids - _gf["centroids"]).max()
        if _cg_drift > 1e-10:
            raise RuntimeError(
                f"cg mesh centroids differ from npz by {_cg_drift:.3e} — "
                f"mesh does not match geometry_fields.npz source"
            )

        # Step 2: Read u_pre from this case's prestress_inverse.bp onto the cg mesh.
        # complete_cycle.py wrote it via adios4dolfinx.write_function_on_input_mesh,
        # so it round-trips cleanly to the cg mesh ordering.
        _pres_path = solver_dir / "prestress_inverse.bp"
        _V_cg_L2 = dolfinx.fem.functionspace(_cg_mesh, ("Lagrange", 2, (3,)))
        _u_pre_cg = dolfinx.fem.Function(_V_cg_L2)
        adios4dolfinx.read_function(_pres_path, _u_pre_cg, time=0.0, name="u_pre")

        # Interpolate u_pre to cg cell centroids via DG0
        _V_cg_DG0v = dolfinx.fem.functionspace(_cg_mesh, ("DG", 0, (3,)))
        _u_pre_cg_dg0 = dolfinx.fem.Function(_V_cg_DG0v)
        _u_pre_cg_dg0.interpolate(_u_pre_cg)
        _u_pre_per_cell = _u_pre_cg_dg0.x.array.reshape(-1, 3)[:_n_cg]

        # Step 3: Compute prestressed cg centroids. These should match the checkpoint
        # centroids to ~1 mm (DG0 interpolation error, << inter-cell spacing).
        _cg_centroids_prestressed = _cg_centroids + _u_pre_per_cell

        # Step 4: Match each LOCAL checkpoint cell to its canonical cg counterpart.
        #
        # We use a GLOBAL linear sum assignment (Hungarian algorithm) instead of
        # naive nearest-neighbor. This guarantees a true bijection: every cg cell
        # is used exactly once, no duplicates, no orphans. The cost is one square
        # assignment of ~n_global cells (~2153 for UKB) computed on each rank in
        # ~10-20s, which is negligible vs the rest of the computation.
        #
        # History: earlier versions used cKDTree nearest-neighbor per rank. This
        # failed on severe/end_stage (bijection 2152/2153 due to adjacent cells
        # tilting toward the same cg centroid under large prestress drift). The
        # linear_sum_assignment formulation eliminates this by construction.
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        # Gather all local ckpt centroids to every rank. The resulting array is in
        # MPI-rank order: [rank0's local cells, rank1's local cells, ...]. Total
        # length equals n_global_cells (sum of size_local across ranks, ignoring
        # ghosts because compute_midpoints was called with local_cells only).
        _all_centroids_list = comm.allgather(centroids)
        _all_ckpt_centroids = np.vstack(_all_centroids_list)
        _local_sizes = [len(c) for c in _all_centroids_list]
        _rank_offset = sum(_local_sizes[:rank])
        _n_total = sum(_local_sizes)

        if _n_total != _n_cg:
            raise RuntimeError(
                f"Total ckpt cells ({_n_total}) != n_cg ({_n_cg}). "
                f"Mesh correspondence is broken — check that the checkpoint was "
                f"generated from the same shared mesh as the canonical geometry."
            )

        # Build the square cost matrix and run linear_sum_assignment on every rank
        # (deterministic; result is identical across ranks). Each rank then pulls
        # its local slice of the global permutation.
        _cost = cdist(_all_ckpt_centroids, _cg_centroids_prestressed)
        _row_ind, _col_ind = linear_sum_assignment(_cost)
        # row_ind == arange(n_total) for square inputs; col_ind is the full
        # global permutation: global_ckpt_idx → cg_idx
        _global_ckpt_to_cg_idx = _col_ind  # shape (n_total,)
        _ckpt_to_cg_idx = _global_ckpt_to_cg_idx[_rank_offset:_rank_offset + n_local_cells]
        # Expose the global permutation so the save block can write it into the
        # per_cell_data.npz. Downstream scripts (especially visualization) can
        # then invert it without re-running the u_pre + linear_sum_assignment dance.
        _ckpt_to_cg_idx_global = _global_ckpt_to_cg_idx

        # Sanity / diagnostics: compute the assignment drift on this rank's slice
        _assigned_distances = _cost[np.arange(_rank_offset, _rank_offset + n_local_cells),
                                    _ckpt_to_cg_idx]
        _max_drift_local = float(_assigned_distances.max()) if len(_assigned_distances) > 0 else 0.0
        _max_drift_global = comm.allreduce(_max_drift_local, op=MPI.MAX)

        # Bijection check (should always pass with linear_sum_assignment on a square cost)
        _n_unique_global = len(np.unique(_global_ckpt_to_cg_idx))
        if rank == 0:
            _spacing_est = (_cg_centroids.max(0) - _cg_centroids.min(0)).max() / (_n_cg ** (1/3))
            logger.info(f"  u_pre permutation (linear_sum_assignment): "
                        f"max drift = {_max_drift_global:.3e} m")
            logger.info(f"  Drift is within {_max_drift_global/_spacing_est*100:.1f}% "
                        f"of inter-cell spacing (~{_spacing_est:.3e} m)")
            if _n_unique_global != _n_cg:
                logger.error(
                    f"  !! BIJECTION FAILED: {_n_unique_global}/{_n_cg} unique targets. "
                    f"This should not happen with linear_sum_assignment on a square cost."
                )
            else:
                logger.info(f"  ✓ Exact bijection: {_n_unique_global}/{_n_cg} cells mapped")

        # Step 5: Apply the permutation to pull canonical fields into ckpt ordering
        d_lv = _gf["d_lv"][_ckpt_to_cg_idx]
        d_rv = _gf["d_rv"][_ckpt_to_cg_idx]
        d_epi = _gf["d_epi"][_ckpt_to_cg_idx]
        d_sum = _gf["d_sum"][_ckpt_to_cg_idx]
        tau = _gf["tau"][_ckpt_to_cg_idx]
        entry_t = _gf["entry_t"][_ckpt_to_cg_idx]
        is_geometric_septum = _gf["is_geometric_septum"][_ckpt_to_cg_idx]
        is_ldrb_septum = _gf["is_ldrb_septum"][_ckpt_to_cg_idx]
        touches_epi = _gf["touches_epi"][_ckpt_to_cg_idx]
        lvrv_vals = _gf["lv_rv_scalar"][_ckpt_to_cg_idx] if "lv_rv_scalar" in _gf.files else np.zeros(len(_ckpt_to_cg_idx))
        epi_vals = _gf["epi_scalar_dg0"][_ckpt_to_cg_idx] if "epi_scalar_dg0" in _gf.files else np.zeros(len(_ckpt_to_cg_idx))

        # Step 6: Reconstruct the corrected envelope (d_sum_max only + ~touches_epi)
        _d_sum_max_mm_canonical = float(_gf.get("d_sum_max_mm", 22.0))
        if len(d_sum) > 0 and d_sum.max() < 0.1:
            mesh_scale_to_mm = 1000.0
        else:
            mesh_scale_to_mm = 1.0
        d_sum_max = _d_sum_max_mm_canonical / mesh_scale_to_mm
        envelope = (d_sum <= d_sum_max) & ~touches_epi
        study_region = (is_geometric_septum | is_ldrb_septum) & (d_sum < d_sum_max)

        # Consistency check: n_geo and n_ldrb should match the canonical counts
        # (modulo a ~1-cell non-bijection tolerance)
        if rank == 0:
            _n_geo_ckpt = comm.allreduce(int(is_geometric_septum.sum()))
            _n_ldrb_ckpt = comm.allreduce(int(is_ldrb_septum.sum()))
            _n_env_ckpt = comm.allreduce(int(envelope.sum()))
            _n_geo_canon = int(_gf["is_geometric_septum"].sum())
            _n_ldrb_canon = int(_gf["is_ldrb_septum"].sum())
            logger.info(f"  Canonical geometric: {_n_geo_canon} (ckpt gives {_n_geo_ckpt})")
            logger.info(f"  Canonical LDRB:      {_n_ldrb_canon} (ckpt gives {_n_ldrb_ckpt})")
            logger.info(f"  Corrected envelope:  {_n_env_ckpt} cells")
            if abs(_n_geo_ckpt - _n_geo_canon) > 3:
                logger.warning(f"  Geometric count mismatch > 3 cells — check permutation")
        else:
            comm.allreduce(int(is_geometric_septum.sum()))
            comm.allreduce(int(is_ldrb_septum.sum()))
            comm.allreduce(int(envelope.sum()))

    # Per-case computation path (original behavior — used if --geometry-fields is NOT given).
    # Per-cell distances. Prefer point-to-facet distance (via pyvista/VTK) which
    # is mesh-resolution-independent. Fall back to nearest-vertex (cKDTree) if
    # pyvista is unavailable. On coarse meshes (char_length ~ 10 mm) the two
    # methods can disagree by several mm near surfaces, and facet distance is the
    # correct answer.
    def _build_surface_polydata_global(tag_ids):
        """Build a GLOBAL PolyData by gathering surface triangles across all MPI ranks.

        Each rank extracts its local facets for the given tag IDs, then all ranks
        allgather the triangle vertex coordinates. The resulting PolyData contains
        the COMPLETE surface, so compute_implicit_distance gives identical results
        on every rank regardless of how the mesh is partitioned.
        """
        # Local: extract triangles and their vertex coordinates
        facets = np.hstack([ffun.find(t) for t in tag_ids])
        local_triangles = []  # list of (3, 3) arrays: 3 vertex coords per triangle
        for f_idx in facets:
            vert_ids = f2v_conn.links(f_idx)
            if len(vert_ids) != 3:
                continue
            local_triangles.append(mesh.geometry.x[vert_ids])  # shape (3, 3)

        if local_triangles:
            local_tri_array = np.array(local_triangles)  # (n_local_tri, 3, 3)
        else:
            local_tri_array = np.empty((0, 3, 3))

        # Global: gather all triangles to all ranks
        all_tri_arrays = comm.allgather(local_tri_array)
        global_tris = np.concatenate([a for a in all_tri_arrays if len(a) > 0])

        if len(global_tris) == 0:
            return None

        # Build PolyData from the global triangle set
        n_tri = len(global_tris)
        points = global_tris.reshape(-1, 3)  # (n_tri * 3, 3)
        faces_flat = np.zeros(n_tri * 4, dtype=np.int64)
        for i in range(n_tri):
            faces_flat[i * 4] = 3
            faces_flat[i * 4 + 1] = i * 3
            faces_flat[i * 4 + 2] = i * 3 + 1
            faces_flat[i * 4 + 3] = i * 3 + 2

        return pv.PolyData(points, faces=faces_flat)

    if not _mode_canonical:
        if _HAS_PYVISTA:
            if rank == 0:
                logger.info("Computing d_lv, d_rv, d_epi via point-to-facet distance "
                             "(pyvista, GLOBAL surface)")
            lv_poly = _build_surface_polydata_global(_lv_tags)
            rv_poly = _build_surface_polydata_global(_rv_tags)
            epi_poly = _build_surface_polydata_global(_epi_tags)
            if lv_poly is None or rv_poly is None or epi_poly is None:
                if rank == 0:
                    logger.warning("No facets found for one of LV/RV/EPI — "
                                   "falling back to nearest-vertex.")
                d_lv = cKDTree(lv_surf).query(centroids)[0] if len(lv_surf) else np.full(len(centroids), np.inf)
                d_rv = cKDTree(rv_surf).query(centroids)[0] if len(rv_surf) else np.full(len(centroids), np.inf)
                d_epi = cKDTree(epi_surf).query(centroids)[0] if len(epi_surf) else np.full(len(centroids), np.inf)
            else:
                if rank == 0:
                    logger.info(f"  Global surfaces: LV={lv_poly.n_faces_strict} tri, "
                                 f"RV={rv_poly.n_faces_strict} tri, EPI={epi_poly.n_faces_strict} tri")
                centroids_poly = pv.PolyData(centroids.astype(np.float64))
                d_lv = np.abs(centroids_poly.compute_implicit_distance(lv_poly)["implicit_distance"])
                d_rv = np.abs(centroids_poly.compute_implicit_distance(rv_poly)["implicit_distance"])
                d_epi = np.abs(centroids_poly.compute_implicit_distance(epi_poly)["implicit_distance"])
        else:
            if rank == 0:
                logger.warning("pyvista not available — falling back to nearest-vertex distance. "
                               "Envelope d_epi threshold may not reliably exclude surface cells.")
            d_lv = cKDTree(lv_surf).query(centroids)[0]
            d_rv = cKDTree(rv_surf).query(centroids)[0]
            d_epi = cKDTree(epi_surf).query(centroids)[0]

        d_sum = d_lv + d_rv

        # Euclidean tau
        tau = d_lv / (d_lv + d_rv)

        # Geometric septum (lower bound)
        is_geometric_septum = np.maximum(d_lv, d_rv) < d_epi

        # LDRB scalar-field septum (upper bound) — solve 2 Laplace equations
        if rank == 0:
            logger.info("Solving Laplace equations for LDRB septum definition...")

        V_CG1 = dolfinx.fem.functionspace(mesh, ("CG", 1))
        u_trial = ufl.TrialFunction(V_CG1)
        v_test = ufl.TestFunction(V_CG1)
        a_laplace = ufl.dot(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
        L_zero = dolfinx.fem.Constant(mesh, 0.0) * v_test * ufl.dx

        lv_facets = ffun.find(_lv_tags[0])
        rv_facets = ffun.find(_rv_tags[0])
        epi_facets = ffun.find(_epi_tags[0])

        lv_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, 2, lv_facets)
        rv_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, 2, rv_facets)
        epi_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, 2, epi_facets)

        prob_lvrv = dolfinx.fem.petsc.LinearProblem(
            a_laplace, L_zero,
            bcs=[dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), lv_dofs, V_CG1),
                 dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_dofs, V_CG1)],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="percell_lvrv",
        )
        lv_rv_scalar = prob_lvrv.solve()

        prob_epi = dolfinx.fem.petsc.LinearProblem(
            a_laplace, L_zero,
            bcs=[dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), epi_dofs, V_CG1),
                 dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), lv_dofs, V_CG1),
                 dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_dofs, V_CG1)],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="percell_epi",
        )
        epi_scalar = prob_epi.solve()

        lvrv_dg0 = dolfinx.fem.Function(V_DG0)
        lvrv_dg0.interpolate(lv_rv_scalar)
        lvrv_vals = lvrv_dg0.x.array[:n_local_cells].copy()

        epi_dg0 = dolfinx.fem.Function(V_DG0)
        epi_dg0.interpolate(epi_scalar)
        epi_vals = epi_dg0.x.array[:n_local_cells].copy()

        is_ldrb_septum = (epi_vals <= 0.5) & (lvrv_vals > 0.1) & (lvrv_vals < 0.9)

        # Auto-detect mesh units for envelope thresholds
        all_d_sums = d_sum[is_geometric_septum | is_ldrb_septum]
        if len(all_d_sums) > 0 and all_d_sums.max() < 0.1:
            mesh_scale_to_mm = 1000.0
        else:
            mesh_scale_to_mm = 1.0

        d_sum_max = _args.d_sum_max_mm / mesh_scale_to_mm

        study_region = (is_geometric_septum | is_ldrb_septum) & (d_sum < d_sum_max)

        # Topological epi exclusion
        mesh.topology.create_connectivity(3, 2)
        c2f = mesh.topology.connectivity(3, 2)
        epi_facets_local_set = set(ffun.find(_epi_tags[0]).tolist())
        touches_epi = np.zeros(n_local_cells, dtype=bool)
        for cell_i in range(n_local_cells):
            for facet in c2f.links(cell_i):
                if facet in epi_facets_local_set:
                    touches_epi[cell_i] = True
                    break

        # Sweep scalar: entry_t = max(d_lv, d_rv) - d_epi
        entry_t = np.maximum(d_lv, d_rv) - d_epi

        # Envelope: minimal (d_sum ≤ d_sum_max AND ~touches_epi).
        # We intentionally do NOT impose d_epi_min or d_sum_min — those excluded
        # 50-60% of geometric septum cells (apex/thin regions) and broke the
        # invariant that sweep(t=0) == geometric.
        envelope = ((d_sum <= d_sum_max) & ~touches_epi)

    # Region tags from marker (always needed, outside the _mode_canonical branch)
    region_tags = markers_mt.values[:n_local_cells]

    # NOTE: the AHA mid-ring tags (Mid_LV/RV/Septum) are NOT computed here.
    # gernerate_aha_biv mis-segments on the adios checkpoint mesh+ffun pair (it
    # produces almost-all-apical garbage), and computing it on the geometry-dir
    # mesh would land in a different cell ordering than these per-cell arrays.
    # AHA tags are instead added as an aha_tags.npy sidecar by compute_aha_band.py,
    # which runs gernerate_aha_biv on the geometry mesh and maps cells onto this
    # npz ordering via the canonical ckpt_to_cg_idx permutation. The sbatch tail
    # runs it automatically after this script.

    n_geo = comm.allreduce(int(is_geometric_septum.sum()))
    n_ldrb = comm.allreduce(int(is_ldrb_septum.sum()))
    n_study = comm.allreduce(int(study_region.sum()))
    # Gather tau values from all ranks' local study-region cells so the printed
    # range reflects the full study region rather than rank 0's local slice
    # (which may be empty after the canonical permutation remap).
    _local_tau_study = tau[study_region] if study_region.any() else np.empty(0, dtype=float)
    _gathered = comm.gather(_local_tau_study, root=0)
    if rank == 0:
        logger.info(f"Study region: geometric={n_geo}, LDRB={n_ldrb}, union={n_study}")
        _all = np.concatenate(_gathered) if _gathered else np.empty(0)
        if _all.size > 0:
            logger.info(f"Tau range in study region: [{_all.min():.3f}, {_all.max():.3f}]")
        else:
            logger.info("Tau range in study region: [empty — no cells]")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 4: Replay Last Beat + Accumulate Per-Cell Work
    # ════════════════════════════════════════════════════════════════════════════

    timestamps = adios4dolfinx.read_timestamps(checkpoint_path, comm, "displacement")
    n_steps = len(timestamps)
    steps_per_beat = int(round(cycle_length / sim_params["dt"]))
    n_beats_in_ckpt = n_steps // steps_per_beat

    # Decide which beat(s) to replay.
    #   --all-beats    → all 0..n_beats_in_ckpt-1
    #   --beat N       → just beat N
    #   (default)      → last beat
    if _args.all_beats:
        beats_to_process = list(range(n_beats_in_ckpt))
    elif _args.beat is not None:
        if _args.beat < 0 or _args.beat >= n_beats_in_ckpt:
            raise ValueError(
                f"--beat {_args.beat} out of range [0, {n_beats_in_ckpt-1}] "
                f"(checkpoint has {n_beats_in_ckpt} beats)"
            )
        beats_to_process = [_args.beat]
    else:
        beats_to_process = [n_beats_in_ckpt - 1]

    if rank == 0:
        logger.info(f"Beats to process: {beats_to_process} "
                    f"(checkpoint has {n_beats_in_ckpt} beats)")

    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ BEAT LOOP: everything below this is repeated once per beat.             │
    # │ Setup (mesh, fibers, forms, canonical tagging) was done ONCE above.     │
    # │ Each iteration re-initialises accumulators and saves its own npz.        │
    # └─────────────────────────────────────────────────────────────────────────┘
    for _current_beat_idx in beats_to_process:
        start_step = _current_beat_idx * steps_per_beat
        end_step = (_current_beat_idx + 1) * steps_per_beat
        _beat_label = f"beat {_current_beat_idx}"
        if rank == 0:
            logger.info(f"\n━━━ Replaying {_beat_label}: steps {start_step}–{end_step-1} "
                            f"({end_step - start_step} steps, cycle={cycle_length:.3f}s) ━━━")

        # Conversion factor: 1 mmHg = 133.322 Pa
        # Solver pressures are in mmHg; convert to Pa so proxy work (P × dE × vol)
        # has the same units as true work (S × dE × vol), both in Pa.
        MMHG_TO_PA = 133.322

        def get_pressure_at_step(step_idx):
            """Get LV and RV solver cavity pressure (Pa) at a given timestep index.

            Uses the Lagrange multiplier pressure (the actual FEM boundary traction),
            not the 0D ODE pressure.  Indexed directly — no interpolation needed.
            """
            p_lv_mmHg = float(solver_cavity_pressure_mmHg[step_idx, 0])
            p_rv_mmHg = float(solver_cavity_pressure_mmHg[step_idx, 1])
            return p_lv_mmHg * MMHG_TO_PA, p_rv_mmHg * MMHG_TO_PA

        # Accumulation arrays (local cells only)
        n = n_local_cells
        cum_w_total = np.zeros(n)
        cum_w_ff = np.zeros(n)
        cum_w_ss = np.zeros(n)
        cum_w_nn = np.zeros(n)
        # Scalar cross-check accumulators (sum over the whole domain via scalar
        # integration — should equal sum(cum_w_total) to machine precision, proving
        # the DG0 test-function trick is exact).
        cum_w_total_scalar = 0.0
        cum_w_ff_scalar = 0.0
        # Proxy: accumulate P * deps_ff per cell (pressure in Pa, strain dimensionless,
        # volume in m³ → units of Pa·m³ = J, same as w_true)
        cum_proxy_PLV_ff = np.zeros(n)
        cum_proxy_PRV_ff = np.zeros(n)
        cum_proxy_Trans_ff = np.zeros(n)
        cum_proxy_PLV_ll = np.zeros(n)
        cum_proxy_PRV_ll = np.zeros(n)
        cum_proxy_Trans_ll = np.zeros(n)
        cum_proxy_PLV_radial = np.zeros(n)
        cum_proxy_PRV_radial = np.zeros(n)
        cum_proxy_Trans_radial = np.zeros(n)
        cum_proxy_PLV_circ = np.zeros(n)
        cum_proxy_PRV_circ = np.zeros(n)
        cum_proxy_Trans_circ = np.zeros(n)
        # Mean = (P_LV+P_RV)/2 and Sum = P_LV+P_RV septum pressure variants
        # (Sum = 2*Mean, but both kept so loops/scatter can be plotted directly).
        cum_proxy_Mean_ff = np.zeros(n)
        cum_proxy_Sum_ff = np.zeros(n)
        cum_proxy_Mean_ll = np.zeros(n)
        cum_proxy_Sum_ll = np.zeros(n)
        cum_proxy_Mean_radial = np.zeros(n)
        cum_proxy_Sum_radial = np.zeros(n)
        cum_proxy_Mean_circ = np.zeros(n)
        cum_proxy_Sum_circ = np.zeros(n)

        # Cell volumes (reuse the form_vol compiled above)
        cell_vols_vec = dolfinx.fem.assemble_vector(form_vol)
        cell_volumes = cell_vols_vec.array[:n].copy()

        has_previous = False
        p_LV_prev = 0.0
        p_RV_prev = 0.0

        # In preload-only Frank-Starling mode, freeze g(λ) at ED before the loop.
        if FS_PRELOAD_ONLY:
            if rank == 0:
                logger.info(f"Freezing Frank-Starling multiplier at ED (t={timestamps[0]:.4f}s)...")
            adios4dolfinx.read_function(checkpoint_path, problem.u, time=timestamps[0], name="displacement")
            active_model.freeze_at(problem.u)

        for i in range(start_step, end_step):
            t = timestamps[i]

            # Load displacement
            adios4dolfinx.read_function(checkpoint_path, problem.u, time=t, name="displacement")

            # Set active tension (scalar; legacy [N,3] histories collapse to mean).
            Ta.assign(float(np.mean(np.atleast_1d(Ta_history[i]))))

            # Interpolate state variables
            E_cur.interpolate(expr_E)
            # S fields are evaluated symbolically through S_tot_ufl in the forms

            if has_previous:
                # Assemble per-cell work increments
                w_total_vec = dolfinx.fem.assemble_vector(form_w_total)
                w_ff_vec = dolfinx.fem.assemble_vector(form_w_ff)
                w_ss_vec = dolfinx.fem.assemble_vector(form_w_ss)

                cum_w_total += w_total_vec.array[:n]
                cum_w_ff += w_ff_vec.array[:n]
                cum_w_ss += w_ss_vec.array[:n]

                # Scalar cross-check: independent integration of the same integrand
                w_total_s = dolfinx.fem.assemble_scalar(form_w_total_scalar)
                w_ff_s = dolfinx.fem.assemble_scalar(form_w_ff_scalar)
                cum_w_total_scalar += w_total_s
                cum_w_ff_scalar += w_ff_s

                if form_w_nn is not None:
                    w_nn_vec = dolfinx.fem.assemble_vector(form_w_nn)
                    cum_w_nn += w_nn_vec.array[:n]

                # Proxy work: P_avg * deps_ff per cell (trapezoidal rule)
                p_LV, p_RV = get_pressure_at_step(i)
                p_LV_avg = 0.5 * (p_LV + p_LV_prev)
                p_RV_avg = 0.5 * (p_RV + p_RV_prev)
                p_Mean_avg = 0.5 * (p_LV_avg + p_RV_avg)   # (P_LV+P_RV)/2
                p_Sum_avg = p_LV_avg + p_RV_avg            # P_LV+P_RV

                deps_ff_vec = dolfinx.fem.assemble_vector(form_deps_ff)
                deps_ff_arr = deps_ff_vec.array[:n]

                cum_proxy_PLV_ff += p_LV_avg * deps_ff_arr
                cum_proxy_PRV_ff += p_RV_avg * deps_ff_arr
                cum_proxy_Trans_ff += (p_LV_avg - p_RV_avg) * deps_ff_arr
                cum_proxy_Mean_ff += p_Mean_avg * deps_ff_arr
                cum_proxy_Sum_ff += p_Sum_avg * deps_ff_arr

                if form_deps_ll is not None:
                    deps_ll_vec = dolfinx.fem.assemble_vector(form_deps_ll)
                    deps_ll_arr = deps_ll_vec.array[:n]
                    cum_proxy_PLV_ll += p_LV_avg * deps_ll_arr
                    cum_proxy_PRV_ll += p_RV_avg * deps_ll_arr
                    cum_proxy_Trans_ll += (p_LV_avg - p_RV_avg) * deps_ll_arr
                    cum_proxy_Mean_ll += p_Mean_avg * deps_ll_arr
                    cum_proxy_Sum_ll += p_Sum_avg * deps_ll_arr

                if form_deps_radial is not None:
                    deps_radial_vec = dolfinx.fem.assemble_vector(form_deps_radial)
                    deps_radial_arr = deps_radial_vec.array[:n]
                    cum_proxy_PLV_radial += p_LV_avg * deps_radial_arr
                    cum_proxy_PRV_radial += p_RV_avg * deps_radial_arr
                    cum_proxy_Trans_radial += (p_LV_avg - p_RV_avg) * deps_radial_arr
                    cum_proxy_Mean_radial += p_Mean_avg * deps_radial_arr
                    cum_proxy_Sum_radial += p_Sum_avg * deps_radial_arr

                if form_deps_circ is not None:
                    deps_circ_vec = dolfinx.fem.assemble_vector(form_deps_circ)
                    deps_circ_arr = deps_circ_vec.array[:n]
                    cum_proxy_PLV_circ += p_LV_avg * deps_circ_arr
                    cum_proxy_PRV_circ += p_RV_avg * deps_circ_arr
                    cum_proxy_Trans_circ += (p_LV_avg - p_RV_avg) * deps_circ_arr
                    cum_proxy_Mean_circ += p_Mean_avg * deps_circ_arr
                    cum_proxy_Sum_circ += p_Sum_avg * deps_circ_arr

                p_LV_prev, p_RV_prev = p_LV, p_RV
            else:
                p_LV_prev, p_RV_prev = get_pressure_at_step(i)

            # Shift state: current → previous (for next step's trapezoidal average)
            E_prev.x.array[:] = E_cur.x.array[:]
            S_prev.interpolate(expr_S_total)
            has_previous = True

            # Advance the relaxation (activation-lag) multiplier with the current
            # displacement, after S_prev has captured g_state at this step — mirrors
            # the forward sim so g_state follows the same trajectory. No-op unless
            # relaxation mode.
            if FS_RELAX_TAU_S > 0:
                active_model.advance(_relax_dt)

            if rank == 0 and (i % 50 == 0 or i == end_step - 1):
                logger.info(f"  Step {i:04d}/{end_step} | t={t:.4f}s | cum_W_total={cum_w_total.sum():.4e}")

        # Cross-fiber = total - (ff + ss + nn); computed once after accumulation.
        # If n0 is missing, cum_w_nn stays zero, so cross absorbs the nn contribution
        # (flagged via the warning below).
        cum_w_cross = cum_w_total - (cum_w_ff + cum_w_ss + cum_w_nn)
        if form_w_nn is None and rank == 0:
            logger.warning("n0 not loaded — cum_w_cross includes the sheet-normal component "
                           "in addition to true cross-fiber work.")

        finite_checks = {
            "w_total": cum_w_total,
            "w_ff": cum_w_ff,
            "w_ss": cum_w_ss,
            "w_nn": cum_w_nn,
            "w_cross": cum_w_cross,
            "proxy_PLV_ll": cum_proxy_PLV_ll,
            "proxy_PRV_ll": cum_proxy_PRV_ll,
            "proxy_Trans_ll": cum_proxy_Trans_ll,
        }
        bad_counts = {
            name: comm.allreduce(int(arr.size - np.isfinite(arr).sum()), op=MPI.SUM)
            for name, arr in finite_checks.items()
        }
        if any(count > 0 for count in bad_counts.values()):
            if rank == 0:
                logger.error("Non-finite per-cell values detected; refusing to save invalid work data.")
                for name, count in bad_counts.items():
                    if count > 0:
                        logger.error(f"  {name}: {count} non-finite entries")
            raise FloatingPointError("Non-finite per-cell work/proxy values")

        # ════════════════════════════════════════════════════════════════════════════
        # SECTION 5: Validation
        # ════════════════════════════════════════════════════════════════════════════

        if rank == 0:
            logger.info("\n=== VALIDATION ===")

        # ── DG0 vs scalar integration cross-check ──────────────────────────────────
        # The DG0 test-function trick says:
        #   sum_i ∫_{cell_i} f dx  ==  ∫_Ω f dx
        # where the LHS is assemble_vector(form(f * v_dg0 * dx)) summed cell-by-cell
        # and the RHS is assemble_scalar(form(f * dx)). Both use the same quadrature
        # points. This should hold to machine precision. Failure means the per-cell
        # accumulation has an indexing bug or the DG0 trick is being used incorrectly.
        w_total_dg0_sum_global = comm.allreduce(cum_w_total.sum())
        w_total_scalar_global = comm.allreduce(cum_w_total_scalar)
        w_ff_dg0_sum_global = comm.allreduce(cum_w_ff.sum())
        w_ff_scalar_global = comm.allreduce(cum_w_ff_scalar)
        if rank == 0:
            def _rel(a, b):
                if abs(b) < 1e-30: return float("nan")
                return abs(a - b) / abs(b)
            rel_total = _rel(w_total_dg0_sum_global, w_total_scalar_global)
            rel_ff = _rel(w_ff_dg0_sum_global, w_ff_scalar_global)
            logger.info("  DG0 vs scalar integration (machine-precision check):")
            logger.info(f"    w_total: DG0_sum={w_total_dg0_sum_global:.6e}  scalar={w_total_scalar_global:.6e}  rel={rel_total:.2e}")
            logger.info(f"    w_ff:    DG0_sum={w_ff_dg0_sum_global:.6e}  scalar={w_ff_scalar_global:.6e}  rel={rel_ff:.2e}")
            if not np.isfinite([w_total_dg0_sum_global, w_total_scalar_global, w_ff_dg0_sum_global, w_ff_scalar_global, rel_total, rel_ff]).all():
                logger.error("    !! DG0/scalar validation contains non-finite values. This output is invalid.")
                raise FloatingPointError("Non-finite DG0/scalar validation")
            if rel_total > 1e-10 or rel_ff > 1e-10:
                logger.error("    !! DG0/scalar mismatch exceeds 1e-10 — DG0 trick is NOT producing "
                             "the same result as direct integration. This is a BUG.")
            else:
                logger.info("    ✓ DG0 per-cell integration matches scalar domain integration to machine precision")

        # Load regional metrics for comparison
        metrics_dir = results_dir / "metrics"
        metrics_file = metrics_dir / "metrics_downsample_1.npy"
        if metrics_file.exists():
            regional = np.load(metrics_file, allow_pickle=True).item()

            # Sum per-cell work over last beat and compare to regional totals
            # Regional metrics store per-timestep increments for ALL beats.
            # We only computed per-cell for the LAST beat, so sum only the last beat's
            # worth of regional increments.
            n_regional_steps = len(regional.get("work_true_Whole", [0]))
            # steps_per_beat was computed earlier from cycle_length / dt
            last_beat_regional = max(0, n_regional_steps - steps_per_beat)
            work_true_LV_regional = np.sum(regional.get("work_true_LV", [0])[last_beat_regional:])
            work_true_RV_regional = np.sum(regional.get("work_true_RV", [0])[last_beat_regional:])
            work_true_Sept_regional = np.sum(regional.get("work_true_Septum", [0])[last_beat_regional:])
            work_true_Whole_regional = np.sum(regional.get("work_true_Whole", [0])[last_beat_regional:])

            # Sum per-cell by region tag
            lv_mask = region_tags == 1
            rv_mask = region_tags == 2
            sept_mask = region_tags == 3

            w_lv_percell = comm.allreduce(cum_w_total[lv_mask].sum())
            w_rv_percell = comm.allreduce(cum_w_total[rv_mask].sum())
            w_sept_percell = comm.allreduce(cum_w_total[sept_mask].sum())
            w_whole_percell = comm.allreduce(cum_w_total.sum())

            if rank == 0:
                logger.info(f"  Per-cell sum:  LV={w_lv_percell:.6e}  RV={w_rv_percell:.6e}  "
                             f"Sept={w_sept_percell:.6e}  Whole={w_whole_percell:.6e}")
                logger.info(f"  Regional sum:  LV={work_true_LV_regional:.6e}  RV={work_true_RV_regional:.6e}  "
                             f"Sept={work_true_Sept_regional:.6e}  Whole={work_true_Whole_regional:.6e}")

                for name, pc, reg in [("LV", w_lv_percell, work_true_LV_regional),
                                       ("RV", w_rv_percell, work_true_RV_regional),
                                       ("Septum", w_sept_percell, work_true_Sept_regional),
                                       ("Whole", w_whole_percell, work_true_Whole_regional)]:
                    if abs(reg) > 1e-15:
                        rel_err = abs(pc - reg) / abs(reg) * 100
                        status = "✓" if rel_err < 1.0 else "✗"
                        logger.info(f"    {name}: {status} rel_error={rel_err:.4f}%")
                    else:
                        logger.info(f"    {name}: regional=0 (skip check)")
        else:
            if rank == 0:
                logger.warning("No regional metrics found — skipping validation")

        # ════════════════════════════════════════════════════════════════════════════
        # SECTION 6: Gather across MPI ranks and save
        # ════════════════════════════════════════════════════════════════════════════
        #
        # Each rank holds a local slice of the mesh (n_local_cells cells). To write a
        # complete per_cell_data.npz we must gather all local arrays to rank 0 and
        # concatenate. The global ordering is arbitrary (rank0 cells, then rank1, ...)
        # since downstream analysis indexes cells as a flat set, not by spatial order.

        def gather_to_root(local_arr):
            """Gather a local per-cell array to rank 0 and concatenate."""
            gathered = comm.gather(np.ascontiguousarray(local_arr), root=0)
            if rank == 0:
                return np.concatenate(gathered)
            return None

        # Gather all per-cell arrays
        g_tau = gather_to_root(tau)
        g_d_lv = gather_to_root(d_lv)
        g_d_rv = gather_to_root(d_rv)
        g_d_epi = gather_to_root(d_epi)
        g_d_sum = gather_to_root(d_sum)
        g_is_geometric_septum = gather_to_root(is_geometric_septum)
        g_is_ldrb_septum = gather_to_root(is_ldrb_septum)
        g_study_region = gather_to_root(study_region)
        g_region_tags = gather_to_root(region_tags)
        g_w_total = gather_to_root(cum_w_total)
        g_w_ff = gather_to_root(cum_w_ff)
        g_w_ss = gather_to_root(cum_w_ss)
        g_w_nn = gather_to_root(cum_w_nn)
        g_w_cross = gather_to_root(cum_w_cross)
        g_proxy_PLV_ff = gather_to_root(cum_proxy_PLV_ff)
        g_proxy_PRV_ff = gather_to_root(cum_proxy_PRV_ff)
        g_proxy_Trans_ff = gather_to_root(cum_proxy_Trans_ff)
        g_proxy_PLV_ll = gather_to_root(cum_proxy_PLV_ll)
        g_proxy_PRV_ll = gather_to_root(cum_proxy_PRV_ll)
        g_proxy_Trans_ll = gather_to_root(cum_proxy_Trans_ll)
        g_proxy_PLV_radial = gather_to_root(cum_proxy_PLV_radial)
        g_proxy_PRV_radial = gather_to_root(cum_proxy_PRV_radial)
        g_proxy_Trans_radial = gather_to_root(cum_proxy_Trans_radial)
        g_proxy_PLV_circ = gather_to_root(cum_proxy_PLV_circ)
        g_proxy_PRV_circ = gather_to_root(cum_proxy_PRV_circ)
        g_proxy_Trans_circ = gather_to_root(cum_proxy_Trans_circ)
        g_proxy_Mean_ff = gather_to_root(cum_proxy_Mean_ff)
        g_proxy_Sum_ff = gather_to_root(cum_proxy_Sum_ff)
        g_proxy_Mean_ll = gather_to_root(cum_proxy_Mean_ll)
        g_proxy_Sum_ll = gather_to_root(cum_proxy_Sum_ll)
        g_proxy_Mean_radial = gather_to_root(cum_proxy_Mean_radial)
        g_proxy_Sum_radial = gather_to_root(cum_proxy_Sum_radial)
        g_proxy_Mean_circ = gather_to_root(cum_proxy_Mean_circ)
        g_proxy_Sum_circ = gather_to_root(cum_proxy_Sum_circ)
        g_radial_endo_to_epi = gather_to_root(_r_endo_epi_values)
        g_cell_volumes = gather_to_root(cell_volumes)
        g_centroids = gather_to_root(centroids)
        g_lvrv_vals = gather_to_root(lvrv_vals)
        g_epi_vals = gather_to_root(epi_vals)
        # New: envelope and entry_t for the threshold-relaxation sweep
        g_envelope = gather_to_root(envelope)
        g_entry_t = gather_to_root(entry_t)
        g_touches_epi = gather_to_root(touches_epi)

        if rank == 0:
            # Filename: if we're processing a specific beat (via --beat or --all-beats),
            # tag the output with the beat index. Otherwise (default: last beat only),
            # use the unversioned filename.
            _tag_part = f"_{_args.output_tag}" if _args.output_tag else ""
            if _args.all_beats or _args.beat is not None:
                output_path = results_dir / f"per_cell_data{_tag_part}_beat{_current_beat_idx}.npz"
            else:
                output_path = results_dir / f"per_cell_data{_tag_part}.npz"
            np.savez(output_path,
                     tau=g_tau,
                     d_lv=g_d_lv,
                     d_rv=g_d_rv,
                     d_epi=g_d_epi,
                     d_sum=g_d_sum,
                     is_geometric_septum=g_is_geometric_septum,
                     is_ldrb_septum=g_is_ldrb_septum,
                     study_region=g_study_region,
                     region_tags=g_region_tags,
                     # Threshold-relaxation sweep fields (Question A)
                     envelope=g_envelope,
                     entry_t=g_entry_t,
                     touches_epi=g_touches_epi,
                     # Per-cell work
                     w_total=g_w_total,
                     w_ff=g_w_ff,
                     w_ss=g_w_ss,
                     w_nn=g_w_nn,
                     w_cross=g_w_cross,
                     proxy_PLV_ff=g_proxy_PLV_ff,
                     proxy_PRV_ff=g_proxy_PRV_ff,
                     proxy_Trans_ff=g_proxy_Trans_ff,
                     proxy_PLV_ll=g_proxy_PLV_ll,
                     proxy_PRV_ll=g_proxy_PRV_ll,
                     proxy_Trans_ll=g_proxy_Trans_ll,
                     proxy_PLV_radial=g_proxy_PLV_radial,
                     proxy_PRV_radial=g_proxy_PRV_radial,
                     proxy_Trans_radial=g_proxy_Trans_radial,
                     proxy_PLV_circ=g_proxy_PLV_circ,
                     proxy_PRV_circ=g_proxy_PRV_circ,
                     proxy_Trans_circ=g_proxy_Trans_circ,
                     proxy_Mean_ff=g_proxy_Mean_ff,
                     proxy_Sum_ff=g_proxy_Sum_ff,
                     proxy_Mean_ll=g_proxy_Mean_ll,
                     proxy_Sum_ll=g_proxy_Sum_ll,
                     proxy_Mean_radial=g_proxy_Mean_radial,
                     proxy_Sum_radial=g_proxy_Sum_radial,
                     proxy_Mean_circ=g_proxy_Mean_circ,
                     proxy_Sum_circ=g_proxy_Sum_circ,
                     cell_volumes=g_cell_volumes,
                     centroids=g_centroids,
                     radial_endo_to_epi=g_radial_endo_to_epi,
                     lv_rv_scalar=g_lvrv_vals,
                     epi_scalar_dg0=g_epi_vals,
                     # DG0 vs scalar integration cross-check (machine-precision invariant)
                     w_total_scalar_integral=w_total_scalar_global,
                     w_ff_scalar_integral=w_ff_scalar_global,
                     # Tagging mode used (for provenance)
                     tagging_mode=np.array(
                         "canonical" if _mode_canonical else ("tag_at_ed" if _mode_ed else "tag_at_unloaded")
                     ),
                     # Canonical mode permutation: global_ckpt_idx → cg_idx.
                     # Only populated in --geometry-fields mode; empty otherwise.
                     # Downstream visualization uses this to map per-cell fields
                     # onto the cg atlas mesh without recomputing u_pre.
                     ckpt_to_cg_idx=(
                         _ckpt_to_cg_idx_global if _ckpt_to_cg_idx_global is not None
                         else np.array([], dtype=np.int64)
                     ),
                     # Envelope parameter used (for provenance)
                     envelope_d_sum_max_mm=_args.d_sum_max_mm,
            )
            logger.info(f"Saved per-cell data to {output_path}")
            logger.info(f"  Global cells: {len(g_tau)}")
            logger.info(f"  Envelope cells: {int(g_envelope.sum())}")
            logger.info(f"  Geometric septum: {int(g_is_geometric_septum.sum())}")
            logger.info(f"  entry_t range in envelope: [{g_entry_t[g_envelope].min():.4f}, "
                        f"{g_entry_t[g_envelope].max():.4f}]")
            logger.info("Done.")


if __name__ == "__main__":
    _main()
