import logging
import shutil
from pathlib import Path
import numpy as np
from mpi4py import MPI
import dolfinx
import ldrb
import cardiac_geometries
import cardiac_geometries.geometry

def generate_and_load(comm, outdir, args, logger):
    """
    Handles the generation (on Rank 0) and loading (on all Ranks) of the geometry.
    Returns the loaded cardiac_geometries.geometry.Geometry object.
    """
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

        # Write markers for debug
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
        additional_data = {
            "f0_DG_1": system_fibers.f0,
            "s0_DG_1": system_fibers.s0,
            "n0_DG_1": system_fibers.n0,
            "markers_mt": markers_mt,
        }

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

    return geo