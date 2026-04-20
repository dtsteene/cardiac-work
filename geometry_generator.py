import logging
import shutil
from pathlib import Path
import numpy as np
from mpi4py import MPI
import dolfinx
import ldrb
import cardiac_geometries
import cardiac_geometries.geometry
import cardiac_geometries.utils


def _compute_and_save_geometry_fields(geodir, outdir, d_sum_max_mm, logger):
    """Compute geometric fields (distances, Laplace scalars, region masks) for the
    saved geometry and write them alongside as:

        <geodir>/geometry_fields.npz   — per-cell numpy arrays, consumer-facing
                                          (read by compute_per_cell.py etc.)
        <outdir>/geometry_fields.xdmf  — same fields as DG0 functions for ParaView viz

    All fields are pure geometry (depend only on mesh + facet markers). Runs
    serially on rank 0 by loading the saved geometry with COMM_SELF so the
    cell ordering matches downstream consumers.

    Absorbs the logic that was previously split across precompute_geometry_fields.py,
    verify_sweep_envelope.py, and viz_region_split.py.
    """
    import ufl
    import pyvista as pv
    import dolfinx.fem.petsc
    from petsc4py import PETSc

    logger.info(f"Computing geometry fields (d_sum_max={d_sum_max_mm} mm)...")

    geo_ser = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, geodir)
    mesh = geo_ser.mesh
    ffun = geo_ser.ffun
    markers = geo_ser.markers

    # Look up marker integer values from the geometry's marker dict.
    # Two naming conventions coexist:
    #   - UKB mesh (cardiac_geometries.mesh.ukb):  "LV", "RV", "EPI"
    #   - Custom patient meshes (XDMF path):       "ENDO_LV", "ENDO_RV", "EPI"
    # Values may be stored as tuple or list: markers[name] = (int, dim).
    def _marker_value(names):
        for name in names:
            if name in markers:
                v = markers[name]
                return v[0] if isinstance(v, (tuple, list)) else v
        return None

    LV_MARKER = _marker_value(["ENDO_LV", "LV"])
    RV_MARKER = _marker_value(["ENDO_RV", "RV"])
    EPI_MARKER = _marker_value(["EPI"])
    if LV_MARKER is None or RV_MARKER is None or EPI_MARKER is None:
        raise RuntimeError(
            f"Missing LV/RV/EPI in geometry markers: {markers}")
    logger.info(f"  Using markers: LV={LV_MARKER}, RV={RV_MARKER}, EPI={EPI_MARKER}")

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(2, 0)
    mesh.topology.create_connectivity(3, 2)
    f2v = mesh.topology.connectivity(2, 0)
    c2f = mesh.topology.connectivity(3, 2)

    cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
    centroids = dolfinx.mesh.compute_midpoints(mesh, 3, cells)
    n_cells = len(cells)

    # Detect mesh units. After generate_and_load scaling, mesh is in meters
    # (bbox ~0.05-0.15 m). Fallback for unscaled meshes (mm).
    bbox = mesh.geometry.x.max(axis=0) - mesh.geometry.x.min(axis=0)
    mesh_to_mm = 1.0 if bbox.max() > 10 else 1000.0
    logger.info(f"  Cells: {n_cells}, mesh extent: {bbox.max():.4f} "
                f"(mesh_to_mm={mesh_to_mm})")

    # ── Surface PolyData for facet-based distance ───────────────────────────
    def _build_surface_poly(marker):
        facets = ffun.find(marker)
        tris = []
        for f_idx in facets:
            verts = f2v.links(f_idx)
            if len(verts) == 3:
                tris.append(mesh.geometry.x[verts])
        if not tris:
            return None
        tris = np.array(tris)
        n = len(tris)
        points = tris.reshape(-1, 3)
        faces = np.zeros(n * 4, dtype=np.int64)
        for i in range(n):
            faces[i*4] = 3
            faces[i*4+1:i*4+4] = [i*3, i*3+1, i*3+2]
        return pv.PolyData(points, faces=faces)

    lv_poly = _build_surface_poly(LV_MARKER)
    rv_poly = _build_surface_poly(RV_MARKER)
    epi_poly = _build_surface_poly(EPI_MARKER)

    # ── Distances ───────────────────────────────────────────────────────────
    centroids_poly = pv.PolyData(centroids.astype(np.float64))
    d_lv = np.abs(centroids_poly.compute_implicit_distance(lv_poly)["implicit_distance"])
    d_rv = np.abs(centroids_poly.compute_implicit_distance(rv_poly)["implicit_distance"])
    d_epi = np.abs(centroids_poly.compute_implicit_distance(epi_poly)["implicit_distance"])
    d_sum = d_lv + d_rv
    tau = d_lv / (d_lv + d_rv)

    # Geometric septum: max(d_lv, d_rv) < d_epi
    is_geometric_septum = np.maximum(d_lv, d_rv) < d_epi

    # ── LDRB septum via Laplace solves ──────────────────────────────────────
    V_CG1 = dolfinx.fem.functionspace(mesh, ("CG", 1))
    u_t = ufl.TrialFunction(V_CG1)
    v_t = ufl.TestFunction(V_CG1)
    a = ufl.dot(ufl.grad(u_t), ufl.grad(v_t)) * ufl.dx
    L = dolfinx.fem.Constant(mesh, 0.0) * v_t * ufl.dx

    lv_f = ffun.find(LV_MARKER)
    rv_f = ffun.find(RV_MARKER)
    epi_f = ffun.find(EPI_MARKER)
    lv_d = dolfinx.fem.locate_dofs_topological(V_CG1, 2, lv_f)
    rv_d = dolfinx.fem.locate_dofs_topological(V_CG1, 2, rv_f)
    epi_d = dolfinx.fem.locate_dofs_topological(V_CG1, 2, epi_f)

    lvrv = dolfinx.fem.petsc.LinearProblem(
        a, L,
        bcs=[dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), lv_d, V_CG1),
             dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_d, V_CG1)],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="geo_lvrv",
    ).solve()

    epi_s = dolfinx.fem.petsc.LinearProblem(
        a, L,
        bcs=[dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), epi_d, V_CG1),
             dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), lv_d, V_CG1),
             dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), rv_d, V_CG1)],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="geo_epi",
    ).solve()

    V_DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    lvrv_dg0 = dolfinx.fem.Function(V_DG0)
    lvrv_dg0.interpolate(lvrv)
    lvrv_vals = lvrv_dg0.x.array[:n_cells].copy()

    epi_dg0 = dolfinx.fem.Function(V_DG0)
    epi_dg0.interpolate(epi_s)
    epi_vals = epi_dg0.x.array[:n_cells].copy()

    is_ldrb_septum = (epi_vals <= 0.5) & (lvrv_vals > 0.1) & (lvrv_vals < 0.9)

    # ── Topological epi exclusion ───────────────────────────────────────────
    epi_facets_set = set(ffun.find(EPI_MARKER).tolist())
    touches_epi = np.zeros(n_cells, dtype=bool)
    for ci in range(n_cells):
        for fi in c2f.links(ci):
            if fi in epi_facets_set:
                touches_epi[ci] = True
                break

    # ── Entry_t, envelope, study_region ─────────────────────────────────────
    # entry_t = max(d_lv, d_rv) - d_epi: t=0 recovers the geometric septum.
    # Envelope: d_sum bound + exclude cells touching epi surface (drops
    # d_epi_min/d_sum_min; see results/docs/transmural_work_profiles.md).
    entry_t = np.maximum(d_lv, d_rv) - d_epi
    d_sum_max = d_sum_max_mm / mesh_to_mm
    envelope = (d_sum <= d_sum_max) & ~touches_epi
    study_region = (is_geometric_septum | is_ldrb_septum) & (d_sum < d_sum_max)

    # ── Save npz (consumer-facing) ──────────────────────────────────────────
    out_npz = geodir / "geometry_fields.npz"
    np.savez(out_npz,
             centroids=centroids,
             tau=tau,
             d_lv=d_lv, d_rv=d_rv, d_epi=d_epi, d_sum=d_sum,
             is_geometric_septum=is_geometric_septum,
             is_ldrb_septum=is_ldrb_septum,
             study_region=study_region,
             envelope=envelope,
             entry_t=entry_t,
             touches_epi=touches_epi,
             lv_rv_scalar=lvrv_vals,
             epi_scalar_dg0=epi_vals,
             d_sum_max_mm=d_sum_max_mm)
    logger.info(
        f"  Saved {out_npz}: geo={int(is_geometric_septum.sum())}, "
        f"ldrb={int(is_ldrb_septum.sum())}, envelope={int(envelope.sum())}, "
        f"touches_epi={int(touches_epi.sum())}")

    # ── Save XDMF (viz) ─────────────────────────────────────────────────────
    def _make_field(name, arr):
        f = dolfinx.fem.Function(V_DG0)
        f.name = name
        f.x.array[:n_cells] = arr.astype(np.float64)
        return f

    # Convenience scalars for ParaView exploration
    side_eu = np.where(tau < 0.5, 1.0, 2.0)
    side_lap = np.where(lvrv_vals > 0.5, 1.0, 2.0)
    # ref_marker: 0=none, 1=LDRB-only, 2=geo-only, 3=both
    ref_marker = np.zeros(n_cells, dtype=np.float64)
    ref_marker[is_ldrb_septum & ~is_geometric_septum] = 1.0
    ref_marker[is_geometric_septum & ~is_ldrb_septum] = 2.0
    ref_marker[is_geometric_septum & is_ldrb_septum] = 3.0

    xdmf_fields = [
        _make_field("is_geometric_septum", is_geometric_septum.astype(float)),
        _make_field("is_ldrb_septum",      is_ldrb_septum.astype(float)),
        _make_field("envelope",            envelope.astype(float)),
        _make_field("study_region",        study_region.astype(float)),
        _make_field("touches_epi",         touches_epi.astype(float)),
        _make_field("ref_marker",          ref_marker),
        _make_field("tau_eu",              tau),
        _make_field("tau_lap",             lvrv_vals),
        _make_field("epi_scalar",          epi_vals),
        _make_field("side_eu",             side_eu),
        _make_field("side_lap",            side_lap),
        _make_field("d_lv",                d_lv),
        _make_field("d_rv",                d_rv),
        _make_field("d_epi",               d_epi),
        _make_field("d_sum",               d_sum),
        _make_field("entry_t",             entry_t),
    ]

    out_xdmf = outdir / "geometry_fields.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_SELF, str(out_xdmf), "w") as xf:
        xf.write_mesh(mesh)
        for f in xdmf_fields:
            xf.write_function(f)
    logger.info(f"  Saved {out_xdmf}")


def generate_and_load(comm, outdir, args, logger, geodir=None):
    """
    Handles the generation (on Rank 0) and loading (on all Ranks) of the geometry.
    Returns the loaded cardiac_geometries.geometry.Geometry object.

    geodir: optional override for the geometry directory. If None, defaults to
            outdir / "geometry". Pass a pre-built geometry path to skip generation
            entirely and load a previously tagged geometry from disk.
    """
    if geodir is None:
        geodir = outdir / "geometry"

    # ========================================================================
    # PHASE 1: GENERATION (Rank 0 Only)
    # ========================================================================
    # We check if geometry exists. If not, Rank 0 generates it.
    if comm.rank == 0 and not (geodir / "geometry.bp").exists():
        logger.info("Generating and processing geometry (Rank 0)...")

        # Determine settings from args
        char_length = args.char_length

        if args.mesh:
            # --- CUSTOM MESH PATH ---
            logger.info(f"Loading CUSTOM MESH from: {args.mesh}")

            with dolfinx.io.XDMFFile(MPI.COMM_SELF, args.mesh, "r") as xdmf:
                mesh_in = xdmf.read_mesh(name="mesh")
                mesh_in.topology.create_connectivity(mesh_in.topology.dim - 1, mesh_in.topology.dim)
                try:
                    ft_in = xdmf.read_meshtags(mesh_in, name="facet_tags")
                except RuntimeError:
                    logger.warning("Could not read 'facet_tags', trying 'mesh_tags'...")
                    ft_in = xdmf.read_meshtags(mesh_in, name="mesh_tags")


            # Standard Marker Map
            markers = {
                "BASE": (10, 2),
                "ENDO_LV": (30, 2),
                "ENDO_RV": (20, 2),
                "EPI": (40, 2)
            }

            geo = cardiac_geometries.geometry.Geometry(
                mesh=mesh_in,
                markers=markers,
                ffun=ft_in,
                f0=None, s0=None, n0=None
            )
            # Assume custom meshes are pre-rotated

        else:
            # --- DEFAULT UKB GENERATION ---
            logger.info("Generating synthetic UKB mesh...")
            geo = cardiac_geometries.mesh.ukb(
                outdir=geodir,
                comm=MPI.COMM_SELF,
                case="ED",
                char_length_max=char_length,
                char_length_min=char_length,
                clipped=True,
            )
            geo = geo.rotate(target_normal=[1.0, 0.0, 0.0], base_marker="BASE")

        # --- FIBER GENERATION (LDRB) ---
        fiber_angles = dict(
            alpha_endo_lv=60, alpha_epi_lv=-60,
            alpha_endo_rv=90, alpha_epi_rv=-25,
            beta_endo_lv=-20, beta_epi_lv=20,
            beta_endo_rv=0, beta_epi_rv=20,
        )

        ldrb_markers = cardiac_geometries.mesh.transform_markers(geo.markers, clipped=True)

        # 1. System for Solver (Quadrature)
        system = ldrb.dolfinx_ldrb(
            mesh=geo.mesh, ffun=geo.ffun, markers=ldrb_markers,
            **fiber_angles, fiber_space="Quadrature_6",
        )

        # 2. System for Markers (DG0)
        system_dg0 = ldrb.dolfinx_ldrb(
            mesh=geo.mesh, ffun=geo.ffun, markers=ldrb_markers,
            **fiber_angles, fiber_space="DG_0",
        )

        # --- EXTRACT MARKERS (MPI SAFE FOR V1) ---
        markers_scalar = system_dg0.markers_scalar
        imap = geo.mesh.topology.index_map(3)
        total_cells = imap.size_local + imap.num_ghosts
        entities = np.arange(total_cells, dtype=np.int32)
        values = markers_scalar.x.array[:total_cells].astype(np.int32)
        markers_mt = dolfinx.mesh.meshtags(geo.mesh, 3, entities, values)

        # Write markers for debug / editor input
        with dolfinx.io.XDMFFile(MPI.COMM_SELF, outdir / "markers_scalar.xdmf", "w") as xdmf:
            xdmf.write_mesh(geo.mesh)
            xdmf.write_meshtags(markers_mt, geo.mesh.geometry)

        # 3. System for Viz (DG1)
        fiber_space = "DG_1"
        system_fibers = ldrb.dolfinx_ldrb(
            mesh=geo.mesh, ffun=geo.ffun, markers=ldrb_markers,
            **fiber_angles, fiber_space=fiber_space,
        )

        # --- SAVE ---
        # apex_gradient: gradient of the apex Laplace solution = true longitudinal
        # (base-to-apex) direction, independent of fiber architecture.
        # Needed for clinical GLS-analogue strain computation.
        additional_data = {
            "f0_DG_1": system_fibers.f0,
            "s0_DG_1": system_fibers.s0,
            "n0_DG_1": system_fibers.n0,
            "markers_mt": markers_mt,
        }
        if system_fibers.apex_gradient is not None:
            additional_data["apex_gradient_DG_1"] = system_fibers.apex_gradient
            logger.info("Saved apex_gradient_DG_1 (longitudinal direction) to geometry")
        if system.apex_gradient is not None:
            additional_data["apex_gradient"] = system.apex_gradient
            logger.info("Saved apex_gradient (quadrature, longitudinal direction) to geometry")

        if (geodir / "geometry.bp").exists():
            shutil.rmtree(geodir / "geometry.bp")

        cardiac_geometries.geometry.save_geometry(
            path=geodir / "geometry.bp",
            mesh=geo.mesh,
            ffun=geo.ffun,
            markers=geo.markers,
            info=geo.info,
            f0=system.f0,
            s0=system.s0,
            n0=system.n0,
            additional_data=additional_data,
        )

        # Export Debug Surfaces
        logger.info("Exporting surface tags for inspection...")
        with dolfinx.io.XDMFFile(MPI.COMM_SELF, outdir / "debug_surfaces.xdmf", "w") as xdmf:
            xdmf.write_mesh(geo.mesh)
            xdmf.write_meshtags(geo.ffun, geo.mesh.geometry)

        # Export fiber directions as VTX/BP for ParaView visualization
        logger.info("Exporting fiber directions (VTX) for ParaView...")
        fiber_viz_fields = []
        system_fibers.f0.name = "f0_fiber"
        fiber_viz_fields.append(system_fibers.f0)
        system_fibers.s0.name = "s0_sheet"
        fiber_viz_fields.append(system_fibers.s0)
        system_fibers.n0.name = "n0_sheet_normal"
        fiber_viz_fields.append(system_fibers.n0)
        if system_fibers.apex_gradient is not None:
            system_fibers.apex_gradient.name = "l0_longitudinal"
            fiber_viz_fields.append(system_fibers.apex_gradient)
            logger.info("  Wrote l0_longitudinal (apex_gradient) to fiber_directions.bp")
        with dolfinx.io.VTXWriter(MPI.COMM_SELF, outdir / "fiber_directions.bp", fiber_viz_fields) as vtx:
            vtx.write(0.0)

    # ========================================================================
    # PHASE 2: SYNCHRONIZATION & LOADING (All Ranks)
    # ========================================================================
    # Crucial: All ranks wait here until Rank 0 finishes generating/writing files
    comm.barrier()

    logger.info("Loading geometry...")
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)

    # --- SCALING ---
    # Apply scaling based on source assumption (Custom=cm, UKB=mm).
    # SKIP scaling if geometry was loaded from a pre-built directory that is
    # already in meters (extent ~0.05-0.15 m). This prevents the double-scaling
    # bug where re-running with GEOMETRY_DIR applies 1e-3 again and corrupts
    # the geometry.bp on re-save.
    coords = geo.mesh.geometry.x
    extent_max = float((coords.max(axis=0) - coords.min(axis=0)).max())
    if extent_max < 1.0 and extent_max > 0.01:
        # Already in meters — skip scaling
        logger.info(f"Geometry extent={extent_max:.4f} m — already in meters, skipping scaling")
        scale = 1.0
    elif args.mesh:
        scale = 1e-2  # cm -> m
    else:
        scale = 1e-3  # mm -> m

    if scale != 1.0:
        geo.mesh.geometry.x[:] *= scale

    # Re-save geometry in meters so postprocessing loads the exact same mesh state.
    # This ensures adios4dolfinx DOF ordering matches between simulation and replay.
    # SKIP re-save if no scaling was applied (geometry already in meters) — avoids
    # deleting geometry.bp while other MPI ranks are still reading it.
    if scale != 1.0:
        comm.barrier()
        if comm.rank == 0:
            if (geodir / "geometry.bp").exists():
                shutil.rmtree(geodir / "geometry.bp")
        comm.barrier()
        cardiac_geometries.geometry.save_geometry(
            path=geodir / "geometry.bp",
            mesh=geo.mesh,
            ffun=geo.ffun,
            markers=geo.markers,
            info=geo.info,
            f0=geo.f0,
            s0=geo.s0,
            n0=geo.n0,
            additional_data=geo.additional_data,
        )
        logger.info("Re-saved geometry in meters for offline postprocessing")
    else:
        logger.info("Geometry already in meters — skipping re-save")

    geo._geo_scale = 1.0  # Already in meters

    # ========================================================================
    # PHASE 3: GEOMETRY FIELDS (Rank 0 Only, cached)
    # ========================================================================
    # Compute & save the per-cell geometric fields (distances, Laplace scalars,
    # region masks) that downstream analysis consumes. Only done once per
    # geometry — if geometry_fields.npz already exists we skip.
    comm.barrier()
    if comm.rank == 0 and not (geodir / "geometry_fields.npz").exists():
        d_sum_max_mm = getattr(args, 'd_sum_max_mm', 22.0)
        _compute_and_save_geometry_fields(geodir, outdir, d_sum_max_mm, logger)
    comm.barrier()

    return geo


if __name__ == "__main__":
    import argparse

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate and inspect cardiac geometries"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./geometries",
        help="Output directory for geometry subdirectories (default: ./geometries)"
    )
    parser.add_argument(
        "--single",
        type=str,
        choices=["ukb", "pah", "healthy"],
        default=None,
        help="Generate only a single geometry type instead of all three"
    )
    parser.add_argument(
        "-m", "--mesh",
        type=str,
        default=None,
        help="Path to custom mesh XDMF file (only used with --single)"
    )
    parser.add_argument(
        "-c", "--char-length",
        type=float,
        default=5.0,
        help="Characteristic mesh length in mm (default: 5.0)"
    )
    parser.add_argument(
        "--d-sum-max-mm",
        type=float,
        default=22.0,
        help="Upper bound on d_lv + d_rv (mm) for the septum envelope field. "
             "Prevents the sweep from reaching free-wall cells. Default: 22.0"
    )

    cli_args = parser.parse_args()

    # Create MPI communicator
    comm = MPI.COMM_WORLD

    # Define geometry configurations
    geometry_configs = {
        "ukb": {
            "mesh": None,  # None means use synthetic UKB
            "outdir": Path(cli_args.output_dir) / "ukb"
        },
        "healthy": {
            "mesh": "/Users/daniel/Documents/master/data/meshes 2/healthy.xdmf",
            "outdir": Path(cli_args.output_dir) / "healthy"
        },
        "pah": {
            "mesh": "/Users/daniel/Documents/master/data/meshes 2/pah.xdmf",
            "outdir": Path(cli_args.output_dir) / "pah"
        }
    }

    # Determine which geometries to generate
    if cli_args.single:
        if cli_args.single == "ukb":
            to_generate = ["ukb"]
        elif cli_args.single in geometry_configs:
            to_generate = [cli_args.single]
            # Override mesh path if provided
            if cli_args.mesh:
                geometry_configs[cli_args.single]["mesh"] = cli_args.mesh
    else:
        to_generate = ["ukb", "healthy", "pah"]

    # Generate geometries
    for geo_type in to_generate:
        config = geometry_configs[geo_type]
        outdir = config["outdir"]
        mesh_path = config["mesh"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Generating {geo_type.upper()} geometry...")
        logger.info(f"{'='*60}")

        outdir.mkdir(parents=True, exist_ok=True)

        # Create arguments namespace for generate_and_load
        args = argparse.Namespace(
            char_length=cli_args.char_length,
            mesh=mesh_path,
            d_sum_max_mm=cli_args.d_sum_max_mm,
        )

        # Log info
        logger.info(f"Output directory: {outdir}")
        logger.info(f"Mesh: {mesh_path if mesh_path else 'UKB synthetic'}")
        logger.info(f"Characteristic length: {args.char_length} mm")

        # Generate and load geometry
        geo = generate_and_load(comm, outdir, args, logger)

        logger.info(f"✓ {geo_type.upper()} geometry generated successfully!")
        logger.info(f"  Mesh: {geo.mesh}")
        logger.info(f"  Number of cells: {geo.mesh.topology.index_map(3).size_local}")
        logger.info(f"  Output files saved to: {outdir}\n")

    logger.info("="*60)
    logger.info("All geometries completed!")
    logger.info("="*60)
