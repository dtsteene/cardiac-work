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


# ── Wall Thickness Modification ───────────────────────────────
# Point cloud vertex ranges from the UKB atlas (ukb/surface.py)
_LV_ENDO = (0, 1500)
_RV_ENDO_RANGES = [(1500, 2165), (2165, 3224)]
_RVFW_RANGE = (5729, 5808)
_EPI_RANGE = (3224, 5582)


def modify_wall_thickness(points, displacement_mm, logger=None):
    """
    Modify wall thickness by displacing endocardial points.

    For each endocardial point, we find its nearest epicardial point.
    The vector from endo → epi is the through-wall direction.
    We move the endo point along this vector:
      displacement_mm < 0: move INTO cavity  → wall THICKENS
      displacement_mm > 0: move TOWARD epi   → wall THINS
      displacement_mm = 0: no change

    The epicardial surface is always kept fixed.

    Parameters
    ----------
    points : np.ndarray, shape (n_points, 3)
        Full UKB point cloud (before unwanted node removal).
    displacement_mm : float
        How far to move endocardial points (mm). Negative = thickening.
    logger : logging.Logger, optional
        Logger for status messages.

    Returns
    -------
    np.ndarray
        Modified point cloud (same shape as input).
    """
    from scipy.spatial import KDTree

    if displacement_mm == 0.0:
        return points.copy()

    pts = points.copy()
    epi = pts[_EPI_RANGE[0]:_EPI_RANGE[1]]
    epi_tree = KDTree(epi)

    def _displace(sl, region_name):
        endo = pts[sl]
        # Average direction over k nearest epi points → smooth displacement field
        # (single nearest-epi gives kinks that cause GMSH to hang/fail)
        k = min(5, len(epi))
        _, nearest_idxs = epi_tree.query(endo, k=k)  # (n, k)
        if k == 1:
            nearest_idxs = nearest_idxs[:, np.newaxis]
        neighbors = epi[nearest_idxs]          # (n, k, 3)
        raw_dir = neighbors - endo[:, np.newaxis, :]  # (n, k, 3)
        direction = raw_dir.mean(axis=1)       # (n, 3) — smoothed transmural vector
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        # Use single-nearest distance as local wall thickness for clamping
        thick = np.linalg.norm(epi[nearest_idxs[:, 0]] - endo, axis=1)
        max_abs = 0.40 * thick
        clamped = np.clip(displacement_mm, -max_abs, max_abs)
        pts[sl] = endo + clamped[:, np.newaxis] * (direction / norms)
        if logger:
            thick_after = np.mean(np.linalg.norm(epi[nearest_idxs[:, 0]] - pts[sl], axis=1))
            actual_disp = np.mean(np.abs(clamped))
            logger.info(f"  {region_name}: {np.mean(thick):.2f} → {thick_after:.2f} mm "
                        f"(mean |disp| = {actual_disp:.2f} mm)")

    if logger:
        logger.info(f"Modifying wall thickness (displacement = {displacement_mm:+.1f} mm):")

    _displace(slice(_LV_ENDO[0], _LV_ENDO[1]), "LV endo")
    for s, e in _RV_ENDO_RANGES:
        _displace(slice(s, e), f"RV endo [{s}:{e}]")
    # Note: RVFW (79 pts, indices 5729-5808) is NOT displaced — it sits outside
    # the EPI range and displacing it simultaneously with RV endo causes PLC
    # self-intersections in GMSH at ≥2mm displacement.

    return pts


def modify_wall_thickness_regional(points, displacement_mm, region="all",
                                    max_fraction=0.90, logger=None):
    """
    Modify wall thickness for a specific region only.

    Parameters
    ----------
    points : np.ndarray, shape (n_points, 3)
        Full UKB point cloud.
    displacement_mm : float
        How far to move endocardial points (mm). Negative = thickening.
    region : str
        Which region to thicken:
        - "all": all endocardial surfaces (same as modify_wall_thickness)
        - "rv_fw": RV freewall only (indices 2165-3224, the non-septal RV endo)
        - "rv": all RV endo (septal + freewall, indices 1500-3224)
        - "lv": LV endo only (indices 0-1500)
    max_fraction : float
        Maximum displacement as fraction of local wall thickness (default 0.90).
        Higher values allow more aggressive thickening. Safe for warp-based
        meshes where the PDE solver maintains element quality.
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Modified point cloud.
    """
    if region == "all":
        return modify_wall_thickness(points, displacement_mm, logger)

    from scipy.spatial import KDTree

    if displacement_mm == 0.0:
        return points.copy()

    pts = points.copy()
    epi = pts[_EPI_RANGE[0]:_EPI_RANGE[1]]
    epi_tree = KDTree(epi)

    def _displace(sl, region_name):
        endo = pts[sl]
        k = min(5, len(epi))
        _, nearest_idxs = epi_tree.query(endo, k=k)
        if k == 1:
            nearest_idxs = nearest_idxs[:, np.newaxis]
        neighbors = epi[nearest_idxs]
        raw_dir = neighbors - endo[:, np.newaxis, :]
        direction = raw_dir.mean(axis=1)
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        thick = np.linalg.norm(epi[nearest_idxs[:, 0]] - endo, axis=1)
        max_abs = max_fraction * thick
        clamped = np.clip(displacement_mm, -max_abs, max_abs)
        pts[sl] = endo + clamped[:, np.newaxis] * (direction / norms)
        if logger:
            thick_after = np.mean(np.linalg.norm(epi[nearest_idxs[:, 0]] - pts[sl], axis=1))
            actual_disp = np.mean(np.abs(clamped))
            logger.info(f"  {region_name}: {np.mean(thick):.2f} → {thick_after:.2f} mm "
                        f"(mean |disp| = {actual_disp:.2f} mm, "
                        f"clamp={max_fraction*100:.0f}%)")

    if logger:
        logger.info(f"Regional thickness modification: region={region}, "
                    f"displacement={displacement_mm:+.1f} mm")

    region_map = {
        "rv_fw": [(2165, 3224, "RV freewall endo")],
        "rv": [(1500, 2165, "RV septal endo"), (2165, 3224, "RV freewall endo")],
        "lv": [(0, 1500, "LV endo")],
    }

    if region not in region_map:
        raise ValueError(f"Unknown region '{region}'. Choose from: all, rv_fw, rv, lv")

    for start, end, name in region_map[region]:
        _displace(slice(start, end), name)

    return pts


def _get_atlas_points(pca_scores=None, logger=None):
    """Get UKB atlas point cloud, optionally deformed by PCA scores.

    Parameters
    ----------
    pca_scores : np.ndarray, optional
        Z-score per PCA mode. If None, returns the mean shape.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Full UKB point cloud (ED frame).
    """
    import ukb.atlas
    import h5py

    atlas_path = ukb.atlas.download_atlas(Path.home() / ".ukb", all=False)
    with h5py.File(atlas_path, "r") as hdf:
        mu = np.transpose(hdf["MU"])
        S = mu.copy()
        if pca_scores is not None:
            for i, z in enumerate(pca_scores):
                if z != 0:
                    eigenvalue = hdf["LATENT"][0, i]
                    eigenvector = hdf["COEFF"][i, :]
                    S = S + z * np.sqrt(eigenvalue) * eigenvector
    N = S.shape[1] // 2
    pts = np.reshape(S[0, :N], (-1, 3))
    if logger:
        from scipy.spatial import KDTree
        epi = pts[_EPI_RANGE[0]:_EPI_RANGE[1]]
        tree = KDTree(epi)
        lv = pts[_LV_ENDO[0]:_LV_ENDO[1]]
        rv = np.vstack([pts[s:e] for s, e in _RV_ENDO_RANGES])
        logger.info(f"  LV thickness: {np.mean(tree.query(lv)[0]):.2f} mm")
        logger.info(f"  RV sept thickness: {np.mean(tree.query(rv[:665])[0]):.2f} mm")
        logger.info(f"  RV FW thickness: {np.mean(tree.query(rv[665:])[0]):.2f} mm")
    return pts


# ── Default UKB Clipping Plane ───────────────────────────────
# Values from ukb.clip.main defaults — used for warp base constraint
_CLIP_ORIGIN = np.array([-13.612554383622273, 18.55767189380559, 15.135103714006394])
_CLIP_NORMAL = np.array([-0.7160843664428893, 0.544394641424108, 0.4368725838557541])


def _generate_baseline_mesh(baseline_dir, char_length, logger):
    """Generate a baseline (mean shape) clipped mesh for warping.

    If the mesh already exists in *baseline_dir*, it is reused.
    Returns the path to the .msh file.
    """
    import ukb.surface as ukb_surf
    import ukb.clip

    msh_file = baseline_dir / "ED_clipped.msh"
    if msh_file.exists():
        logger.info(f"Reusing existing baseline mesh: {msh_file}")
        return msh_file

    baseline_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path.home() / ".ukb"

    logger.info("Generating baseline (mean) surfaces...")
    ukb_surf.main(folder=baseline_dir, case="ED", mode=-1, std=0.0, cache_dir=cache_dir)

    logger.info("Clipping baseline surfaces...")
    ukb.clip.main(folder=baseline_dir, case="ED", smooth=True)

    logger.info("Meshing baseline volume (GMSH)...")
    _create_clipped_mesh_robust(
        folder=baseline_dir, case="ED",
        char_length_max=char_length, char_length_min=char_length,
    )

    logger.info(f"Baseline mesh ready: {msh_file}")
    return msh_file


def _generate_ukb_with_warp(geodir, char_length, target_points, logger,
                            baseline_dir=None, info=None):
    """Generate UKB geometry by warping a baseline mesh to target points.

    Instead of re-meshing from scratch, this warps the mean-shape baseline mesh
    using PDE-based deformation (fenicsx-warp).  The mesh topology and facet tags
    are preserved exactly, ensuring consistent region assignments across variants.

    Parameters
    ----------
    geodir : Path
        Output directory for warped geometry files.
    char_length : float
        Characteristic mesh length (only used if baseline needs generation).
    target_points : np.ndarray, shape (N, 3)
        Target UKB point cloud.
    logger : logging.Logger
    baseline_dir : Path, optional
        Directory containing the baseline mesh.  If None, uses
        ``geodir.parent / "warp_baseline"``.
    info : dict, optional
        Additional info to store in the Geometry object.
    """
    from warp import warp_mesh

    # 1. Ensure baseline mesh exists
    if baseline_dir is None:
        baseline_dir = geodir.parent / "warp_baseline"
    msh_file = _generate_baseline_mesh(baseline_dir, char_length, logger)

    # 2. Load baseline mesh into DOLFINx (facet tags preserved)
    logger.info(f"Loading baseline mesh from {msh_file}...")
    geometry = cardiac_geometries.utils.gmsh2dolfin(
        comm=MPI.COMM_SELF, msh_file=msh_file,
    )

    # 3. Get reference (mean) point cloud
    logger.info("Computing reference (mean) point cloud...")
    points_mean = _get_atlas_points(pca_scores=None)

    # 4. Log target thickness
    logger.info("Target shape:")
    from scipy.spatial import KDTree
    epi_t = target_points[_EPI_RANGE[0]:_EPI_RANGE[1]]
    tree_t = KDTree(epi_t)
    lv_t = target_points[_LV_ENDO[0]:_LV_ENDO[1]]
    rv_t = np.vstack([target_points[s:e] for s, e in _RV_ENDO_RANGES])
    logger.info(f"  LV thickness: {np.mean(tree_t.query(lv_t)[0]):.2f} mm")
    logger.info(f"  RV sept thickness: {np.mean(tree_t.query(rv_t[:665])[0]):.2f} mm")
    logger.info(f"  RV FW thickness: {np.mean(tree_t.query(rv_t[665:])[0]):.2f} mm")

    # 5. Warp mesh in-place
    # Auto-detect required incremental steps from max displacement magnitude.
    max_disp = float(np.max(np.linalg.norm(target_points - points_mean, axis=1)))
    n_steps = max(1, int(np.ceil(max_disp / 2.0)))  # 1 step per 2mm of displacement
    logger.info(f"Warping mesh (hyperelastic, RBF, max_disp={max_disp:.1f}mm, n_steps={n_steps})...")
    warp_mesh(
        domain=geometry.mesh,
        points_reference=points_mean,
        points_target=target_points,
        interpolation_method="rbf",
        solver_method="hyperelastic",
        clip_origin=tuple(_CLIP_ORIGIN),
        clip_normal=tuple(_CLIP_NORMAL),
        n_steps=n_steps,
    )
    logger.info("Warp complete.")

    # 6. Build Geometry (mesh warped in-place, facet tags preserved)
    geodir.mkdir(parents=True, exist_ok=True)
    geo = cardiac_geometries.geometry.Geometry(
        mesh=geometry.mesh,
        markers=geometry.markers,
        ffun=geometry.ffun,
        f0=None, s0=None, n0=None,
        info=info or {"mesh_type": "ukb_warp"},
    )
    return geo


def compute_pca_thickness_direction(atlas_path, strategy="global_no_size", n_modes=50, logger=None):
    """
    Find the optimal direction in PCA space for wall thickness control.

    Strategies:
      - "rv_fw_no_size": Maximize RV free wall thickness, orthogonal to overall size
      - "rv_all_no_size": Maximize all-RV thickness, orthogonal to size
      - "global_no_size": Maximize LV+RV thickness, orthogonal to size
      - "max_rv_fw": Maximize RV free wall (no size constraint)

    Returns: unit vector z of length n_modes
    """
    from scipy.spatial import KDTree, ConvexHull
    import h5py

    with h5py.File(atlas_path, "r") as hdf:
        n_avail = min(n_modes, hdf["COEFF"].shape[0])

        # Get baseline
        mu = np.transpose(hdf["MU"])
        N = mu.shape[1] // 2
        pts_base = np.reshape(mu[0, :N], (-1, 3))

        lv_base = pts_base[_LV_ENDO[0]:_LV_ENDO[1]]
        rv_base = np.vstack([pts_base[s:e] for s, e in _RV_ENDO_RANGES])
        epi_base = pts_base[_EPI_RANGE[0]:_EPI_RANGE[1]]
        tree_base = KDTree(epi_base)
        lv_thick_base = np.mean(tree_base.query(lv_base)[0])
        rv_thick_base = np.mean(tree_base.query(rv_base)[0])
        rv_fw_thick_base = np.mean(tree_base.query(rv_base[665:])[0])  # 665 = sept count
        epi_vol_base = ConvexHull(epi_base).volume
        lv_vol_base = ConvexHull(lv_base).volume

        # Per-mode sensitivities
        sens_lv = np.zeros(n_avail)
        sens_rv = np.zeros(n_avail)
        sens_rv_fw = np.zeros(n_avail)
        sens_epi_vol = np.zeros(n_avail)
        sens_lv_vol = np.zeros(n_avail)

        for m in range(n_avail):
            eigenvalue = hdf["LATENT"][0, m]
            eigenvector = hdf["COEFF"][m, :]
            S = mu + np.sqrt(eigenvalue) * eigenvector
            pts = np.reshape(S[0, :N], (-1, 3))

            lv = pts[_LV_ENDO[0]:_LV_ENDO[1]]
            rv = np.vstack([pts[s:e] for s, e in _RV_ENDO_RANGES])
            epi = pts[_EPI_RANGE[0]:_EPI_RANGE[1]]
            tree = KDTree(epi)

            sens_lv[m] = np.mean(tree.query(lv)[0]) - lv_thick_base
            sens_rv[m] = np.mean(tree.query(rv)[0]) - rv_thick_base
            sens_rv_fw[m] = np.mean(tree.query(rv[665:])[0]) - rv_fw_thick_base
            sens_epi_vol[m] = (ConvexHull(epi).volume - epi_vol_base) / epi_vol_base * 100
            sens_lv_vol[m] = (ConvexHull(lv).volume - lv_vol_base) / lv_vol_base * 100

    def _orthogonalize(z, constraints):
        for c in constraints:
            c = c[:len(z)]
            norm_c = np.linalg.norm(c)
            if norm_c > 1e-10:
                c_unit = c / norm_c
                z = z - np.dot(z, c_unit) * c_unit
        return z

    if strategy == "rv_fw_no_size":
        z = _orthogonalize(sens_rv_fw.copy(), [sens_epi_vol])
    elif strategy == "rv_all_no_size":
        z = _orthogonalize(sens_rv.copy(), [sens_epi_vol])
    elif strategy == "global_no_size":
        z = _orthogonalize(sens_lv + sens_rv, [sens_epi_vol, sens_lv_vol])
    elif strategy == "max_rv_fw":
        z = sens_rv_fw.copy()
    else:
        raise ValueError(f"Unknown PCA strategy: {strategy}")

    norm = np.linalg.norm(z)
    if norm < 1e-10:
        raise ValueError(f"Strategy {strategy} resulted in zero direction")
    z = z / norm

    if logger:
        # Report expected thickness change per σ
        drvfw = np.dot(z, sens_rv_fw[:n_avail])
        dlv = np.dot(z, sens_lv[:n_avail])
        dsize = np.dot(z, sens_epi_vol[:n_avail])
        logger.info(f"PCA strategy '{strategy}': per 1σ → "
                    f"ΔRVFW={drvfw:+.3f}mm, ΔLV={dlv:+.3f}mm, ΔEpiVol={dsize:+.1f}%")

    return z


def _create_clipped_mesh_robust(folder, case, char_length_max, char_length_min):
    """Like ukb.mesh.create_clipped_mesh but handles variable surface counts.

    The upstream code hardcodes addPlaneSurface(..., tag=4) which fails when
    the clipped PLY files produce 4+ surfaces. This version auto-detects
    the topology and assigns tags dynamically.
    """
    import gmsh

    _logger = logging.getLogger(__name__)

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.merge(f"{folder}/lv_clipped.ply")
    gmsh.merge(f"{folder}/rv_clipped.ply")
    gmsh.merge(f"{folder}/epi_clipped.ply")

    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.create_topology()
    gmsh.model.mesh.create_geometry()

    surfaces_before = gmsh.model.getEntities(2)
    curves = gmsh.model.getEntities(1)
    n_surf = len(surfaces_before)
    n_curves = len(curves)
    _logger.info(f"  Clipped mesh: {n_surf} surfaces, {n_curves} curves after topology")

    # Create base plane from boundary curves
    curve_tags = [c[1] for c in curves]
    base_ring = gmsh.model.geo.addCurveLoop(curve_tags)
    base_tag = gmsh.model.geo.addPlaneSurface([base_ring])
    gmsh.model.geo.synchronize()

    surfaces_after = gmsh.model.getEntities(2)
    surf_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces_after])
    vol = gmsh.model.geo.addVolume([surf_loop])
    gmsh.model.geo.synchronize()

    # Assign physical groups dynamically
    # PLY load order: LV, RV, EPI → typically surfaces 1, 2, 3
    # But topology classification may split them
    surf_tags = [s[1] for s in surfaces_before]
    if n_surf == 3:
        physical_groups = {"LV": [1], "RV": [2], "EPI": [3], "BASE": [base_tag]}
    elif n_surf == 4:
        # Most likely RV split into 2 parts
        physical_groups = {"LV": [1], "RV": [2, 3], "EPI": [4], "BASE": [base_tag]}
    elif n_surf == 5:
        physical_groups = {"LV": [1], "RV": [2, 3], "EPI": [4, 5], "BASE": [base_tag]}
    else:
        # Generic fallback
        physical_groups = {
            "LV": [surf_tags[0]],
            "RV": surf_tags[1:-1],
            "EPI": [surf_tags[-1]],
            "BASE": [base_tag],
        }
        _logger.warning(f"Unexpected {n_surf} surfaces — using fallback grouping")

    for name, tags in physical_groups.items():
        p = gmsh.model.addPhysicalGroup(2, tags)
        gmsh.model.setPhysicalName(2, p, name)

    p = gmsh.model.addPhysicalGroup(3, [vol])
    gmsh.model.setPhysicalName(3, p, "Wall")

    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length_max)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length_min)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.1)
    gmsh.model.geo.synchronize()

    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        _logger.warning(f"Delaunay meshing failed ({e}), trying HXT algorithm...")
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.model.mesh.generate(3)

    outfile = (folder / f"{case}_clipped").with_suffix(".msh")
    gmsh.write(str(outfile))
    _logger.info(f"Created mesh {outfile}")
    gmsh.finalize()


def _generate_ukb_with_pca(geodir, char_length, pca_scores, logger):
    """
    Generate UKB geometry using arbitrary PCA score vector (re-mesh approach).

    Parameters
    ----------
    pca_scores : np.ndarray
        Z-score for each PCA mode. Shape = mu + sum(z_i * sqrt(lambda_i) * v_i)
    """
    import ukb.surface as ukb_surf
    import ukb.clip

    case = "ED"

    logger.info("Step 1/6: Generating point cloud from PCA scores...")
    ed_full = _get_atlas_points(pca_scores=pca_scores, logger=logger)

    # Step 3: Write STL surfaces
    logger.info("Step 3/6: Writing STL surfaces...")
    geodir.mkdir(parents=True, exist_ok=True)
    ukb_surf.get_epi_mesh(ed_full).write(str(geodir / f"EPI_{case}.stl"))
    for ch in ["LV", "RV", "RVFW"]:
        ukb_surf.get_chamber_mesh(ch, ed_full).write(str(geodir / f"{ch}_{case}.stl"))
    for v in ["MV", "AV", "TV", "PV"]:
        ukb_surf.get_valve_mesh(v, ed_full).write(str(geodir / f"{v}_{case}.stl"))

    # Step 4: Clip
    logger.info("Step 4/6: Clipping surfaces...")
    ukb.clip.main(folder=geodir, case=case, smooth=True)

    # Step 5: GMSH (use robust version that handles variable surface count)
    logger.info("Step 5/6: Generating volumetric mesh (GMSH)...")
    _create_clipped_mesh_robust(
        folder=geodir, case=case,
        char_length_max=char_length, char_length_min=char_length,
    )

    # Step 6: Convert to DOLFINx
    logger.info("Step 6/6: Converting to DOLFINx Geometry...")
    mesh_name = geodir / f"{case}_clipped.msh"
    geometry = cardiac_geometries.utils.gmsh2dolfin(
        comm=MPI.COMM_SELF, msh_file=mesh_name,
    )

    geo = cardiac_geometries.geometry.Geometry(
        mesh=geometry.mesh,
        markers=geometry.markers,
        ffun=geometry.ffun,
        f0=None, s0=None, n0=None,
        info={"mesh_type": "ukb_pca", "pca_scores": pca_scores.tolist()},
    )
    return geo


def _generate_ukb_with_thickness(geodir, char_length, wall_displacement_mm, logger):
    """
    Replicate what cardiac_geometries.mesh.ukb() does, but with a
    wall thickness modification step injected between surface generation
    and clipping/meshing.

    Steps:
      1. Generate point cloud from PCA atlas (mean shape)
      2. ** Modify endocardial points to change wall thickness **
      3. Write modified STL surfaces
      4. Clip surfaces
      5. Generate volumetric mesh with GMSH
      6. Convert to DOLFINx Geometry

    Returns a Geometry object identical to what mesh.ukb() would return.
    """
    import ukb.surface as ukb_surf
    import ukb.clip

    case = "ED"

    # Step 1: Generate mean point cloud from atlas
    logger.info("Step 1/6: Generating point cloud from UKB atlas (mean shape)...")
    ed_full = _get_atlas_points(pca_scores=None)

    # Step 2: Modify wall thickness
    logger.info("Step 2/6: Modifying wall thickness...")
    ed_modified = modify_wall_thickness(ed_full, wall_displacement_mm, logger)

    # Step 3: Write STL surfaces from modified points
    logger.info("Step 3/6: Writing modified STL surfaces...")
    geodir.mkdir(parents=True, exist_ok=True)

    ukb_surf.get_epi_mesh(ed_modified).write(str(geodir / f"EPI_{case}.stl"))
    for ch in ["LV", "RV", "RVFW"]:
        ukb_surf.get_chamber_mesh(ch, ed_modified).write(str(geodir / f"{ch}_{case}.stl"))
    for v in ["MV", "AV", "TV", "PV"]:
        ukb_surf.get_valve_mesh(v, ed_modified).write(str(geodir / f"{v}_{case}.stl"))

    # Step 4: Clip surfaces
    logger.info("Step 4/6: Clipping surfaces...")
    ukb.clip.main(folder=geodir, case=case, smooth=True)

    # Step 5: Generate volumetric mesh with GMSH (robust version handles
    # variable surface counts that arise from displaced endo surfaces)
    logger.info("Step 5/6: Generating volumetric mesh (GMSH)...")
    _create_clipped_mesh_robust(
        folder=geodir, case=case,
        char_length_max=char_length, char_length_min=char_length,
    )

    # Step 6: Convert to DOLFINx
    logger.info("Step 6/6: Converting to DOLFINx Geometry...")
    mesh_name = geodir / f"{case}_clipped.msh"
    geometry = cardiac_geometries.utils.gmsh2dolfin(
        comm=MPI.COMM_SELF,
        msh_file=mesh_name,
    )

    geo = cardiac_geometries.geometry.Geometry(
        mesh=geometry.mesh,
        markers=geometry.markers,
        ffun=geometry.ffun,
        f0=None, s0=None, n0=None,
        info={"mesh_type": "ukb", "wall_displacement_mm": wall_displacement_mm},
    )
    return geo


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

    # Wall displacement: 0.0 = baseline, negative = thickening (mm)
    wall_displacement_mm = getattr(args, 'wall_displacement', 0.0)
    pca_strategy = getattr(args, 'pca_strategy', None)
    pca_magnitude = getattr(args, 'pca_magnitude', 0.0)
    use_warp = getattr(args, 'warp', False)
    warp_baseline = getattr(args, 'warp_baseline', None)
    baseline_dir = Path(warp_baseline) if warp_baseline else None

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

        elif pca_strategy is not None and pca_magnitude != 0.0:
            # --- UKB WITH PCA-BASED THICKNESS ---
            logger.info(f"Generating UKB mesh via PCA thickness control "
                        f"(strategy={pca_strategy}, magnitude={pca_magnitude:+.1f}σ)...")
            atlas_path = Path.home() / ".ukb" / "UKBRVLV.h5"
            z_dir = compute_pca_thickness_direction(atlas_path, strategy=pca_strategy, logger=logger)
            pca_scores = z_dir * pca_magnitude

            if use_warp:
                logger.info("Using WARP method (topology-preserving)")
                target_points = _get_atlas_points(pca_scores=pca_scores, logger=logger)
                geo = _generate_ukb_with_warp(
                    geodir, char_length, target_points, logger,
                    baseline_dir=baseline_dir,
                    info={"mesh_type": "ukb_pca_warp",
                          "pca_strategy": pca_strategy,
                          "pca_magnitude": pca_magnitude,
                          "pca_scores": pca_scores.tolist()},
                )
            else:
                logger.info("Using REMESH method")
                geo = _generate_ukb_with_pca(geodir, char_length, pca_scores, logger)
            geo = geo.rotate(target_normal=[1.0, 0.0, 0.0], base_marker="BASE")

        elif wall_displacement_mm != 0.0:
            # --- UKB WITH WALL THICKNESS MODIFICATION ---
            logger.info(f"Generating UKB mesh with wall thickness modification "
                        f"(displacement = {wall_displacement_mm:+.1f} mm)...")

            if use_warp:
                logger.info("Using WARP method (topology-preserving)")
                mean_points = _get_atlas_points(pca_scores=None)
                target_points = modify_wall_thickness(mean_points, wall_displacement_mm, logger)
                geo = _generate_ukb_with_warp(
                    geodir, char_length, target_points, logger,
                    baseline_dir=baseline_dir,
                    info={"mesh_type": "ukb_warp",
                          "wall_displacement_mm": wall_displacement_mm},
                )
            else:
                logger.info("Using REMESH method")
                geo = _generate_ukb_with_thickness(
                    geodir, char_length, wall_displacement_mm, logger
                )
            geo = geo.rotate(target_normal=[1.0, 0.0, 0.0], base_marker="BASE")

        else:
            # --- DEFAULT UKB GENERATION (no thickness change) ---
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
    parser.add_argument(
        "-w", "--wall-displacement",
        type=float,
        default=0.0,
        help=("Wall thickness displacement in mm. "
              "Negative = thicken (endo moves into cavity), "
              "positive = thin (endo moves toward epi). "
              "Default: 0.0 (no change)")
    )
    parser.add_argument(
        "--pca-strategy",
        type=str,
        choices=["rv_fw_no_size", "rv_all_no_size", "global_no_size", "max_rv_fw"],
        default=None,
        help=("PCA thickness strategy. Uses multi-mode combination for "
              "physiologically realistic thickness variation. "
              "Requires --pca-magnitude.")
    )
    parser.add_argument(
        "--pca-magnitude",
        type=float,
        default=0.0,
        help=("PCA score magnitude (in standard deviations). "
              "Positive = thicker, negative = thinner. "
              "E.g. --pca-magnitude 3.0 for +3σ thickening.")
    )
    parser.add_argument(
        "--warp",
        action="store_true",
        default=False,
        help=("Use mesh warping (fenicsx-warp) instead of re-meshing for "
              "thickness/PCA variants. Preserves baseline mesh topology and "
              "facet tags across variants for consistent comparisons.")
    )
    parser.add_argument(
        "--warp-baseline",
        type=str,
        default=None,
        help=("Path to pre-existing baseline mesh directory for warping "
              "(must contain ED_clipped.msh). If not provided, baseline is "
              "auto-generated alongside the output directory.")
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
            wall_displacement=cli_args.wall_displacement,
            pca_strategy=cli_args.pca_strategy,
            pca_magnitude=cli_args.pca_magnitude,
            warp=cli_args.warp,
            warp_baseline=cli_args.warp_baseline,
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

