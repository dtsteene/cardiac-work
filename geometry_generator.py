import logging
import shutil
from pathlib import Path
import numpy as np
from mpi4py import MPI
import dolfinx
import ldrb
import cardiac_geometries
import cardiac_geometries.geometry

def generate_and_load(comm, outdir, args, logger, manual_refinement=False, geodir=None):
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

        # --- OPTIONAL MANUAL REFINEMENT ---
        # Open the interactive Septum Tag Editor so the user can correct the
        # LDRB-generated septum boundary before it flows into the FEM solver.
        # After the window is closed, editor.tags holds the final assignments
        # (1=LV, 2=RV, 3=Septum) which are used to rebuild markers_mt.
        # These tags end up in geometry.bp → additional_data["markers_mt"] →
        # scifem.create_space_of_simple_functions → correct DOF assignment in FEniCSx.
        if manual_refinement:
            from septum_editor import SeptumEditor
            xdmf_path = outdir / "markers_scalar.xdmf"
            logger.info("=" * 60)
            logger.info("MANUAL REFINEMENT: Launching Septum Tag Editor.")
            logger.info("Edit septum tags, then close the window to continue.")
            logger.info("(Press S inside the editor to also persist edits to disk.)")
            logger.info("=" * 60)
            editor = SeptumEditor(xdmf_path, output_path=None)
            editor.run()  # blocks until window is closed
            # Rebuild markers_mt from the editor's in-memory tags.
            # This captures all edits regardless of whether the user pressed S.
            updated_values = editor.tags[:total_cells].astype(np.int32)
            markers_mt = dolfinx.mesh.meshtags(geo.mesh, 3, entities, updated_values)
            n_lv   = int((updated_values == 1).sum())
            n_rv   = int((updated_values == 2).sum())
            n_sept = int((updated_values == 3).sum())
            logger.info(f"Manual refinement complete — LV={n_lv}, RV={n_rv}, Septum={n_sept} cells.")

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
    # Apply scaling based on source assumption (Custom=cm, UKB=mm)
    if args.mesh:
        scale = 1e-2 # cm -> m
    else:
        scale = 1e-3 # mm -> m

    geo.mesh.geometry.x[:] *= scale

    # Re-save geometry in meters so postprocessing loads the exact same mesh state.
    # This ensures adios4dolfinx DOF ordering matches between simulation and replay.
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

    geo._geo_scale = 1.0  # Already in meters

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
        "--manual-refinement",
        action="store_true",
        help="Launch interactive Septum Tag Editor after LDRB tagging to manually correct tags before saving"
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
        )
        
        # Log info
        logger.info(f"Output directory: {outdir}")
        logger.info(f"Mesh: {mesh_path if mesh_path else 'UKB synthetic'}")
        logger.info(f"Characteristic length: {args.char_length} mm")
        
        # Generate and load geometry
        geo = generate_and_load(comm, outdir, args, logger,
                                manual_refinement=cli_args.manual_refinement)
        
        logger.info(f"✓ {geo_type.upper()} geometry generated successfully!")
        logger.info(f"  Mesh: {geo.mesh}")
        logger.info(f"  Number of cells: {geo.mesh.topology.index_map(3).size_local}")
        logger.info(f"  Output files saved to: {outdir}\n")
    
    logger.info("="*60)
    logger.info("All geometries completed!")
    logger.info("="*60)

