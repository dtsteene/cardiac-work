"""Clinical-frame direction helpers for postprocessing.

The saved LDRB apex-gradient gives a base-to-apex direction, but it can have a
small through-wall component in curved ventricular geometry.  For the
clinical-style longitudinal strain analogue, project that direction into the
local wall tangent plane before evaluating ``l0 . E . l0``.
"""

from __future__ import annotations

import numpy as np
import dolfinx
import dolfinx.fem
import ufl
from scipy.spatial import cKDTree

from geometry_utils import closest_points_to_surface as _closest_points_to_surface


def _surface_triangles_global(mesh, ffun, tag_ids, comm):
    mesh.topology.create_connectivity(2, 0)
    f2v_conn = mesh.topology.connectivity(2, 0)
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
    return out


def build_radial_endo_to_epi_dg0(mesh, ffun, markers, comm, name="radial_endo_to_epi"):
    """Return a DG0 nearest-endocardium-to-epicardium direction field."""

    mesh.topology.create_connectivity(3, 0)
    imap_3 = mesh.topology.index_map(3)
    n_local_cells = imap_3.size_local
    local_cells = np.arange(n_local_cells, dtype=np.int32)
    centroids_ref = dolfinx.mesh.compute_midpoints(mesh, 3, local_cells)

    lv_marker = markers.get("LV", markers.get("ENDO_LV"))[0]
    rv_marker = markers.get("RV", markers.get("ENDO_RV"))[0]
    epi_marker = markers["EPI"][0]

    lv_tri = _surface_triangles_global(mesh, ffun, [lv_marker], comm)
    rv_tri = _surface_triangles_global(mesh, ffun, [rv_marker], comm)
    epi_tri = _surface_triangles_global(mesh, ffun, [epi_marker], comm)

    cp_lv, d_lv = _closest_points_to_surface(centroids_ref, lv_tri)
    cp_rv, d_rv = _closest_points_to_surface(centroids_ref, rv_tri)
    cp_epi, _ = _closest_points_to_surface(centroids_ref, epi_tri)
    nearest_lv = d_lv <= d_rv
    cp_endo = np.where(nearest_lv[:, None], cp_lv, cp_rv)
    radial_values = _normalize_rows(cp_epi - cp_endo)

    V_DG0_vec = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
    radial = dolfinx.fem.Function(V_DG0_vec, name=name)
    arr = radial.x.array.reshape(-1, 3)
    arr[:] = 0.0
    arr[: len(radial_values)] = radial_values
    radial.x.scatter_forward()
    return radial, radial_values


def _unit(v):
    return v / ufl.sqrt(ufl.inner(v, v) + 1e-16)


def tangent_project_longitudinal(longitudinal, radial):
    """Project a longitudinal direction into the plane orthogonal to radial."""

    l0 = _unit(longitudinal)
    r0 = _unit(radial)
    return _unit(l0 - ufl.inner(l0, r0) * r0)
