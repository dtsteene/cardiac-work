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


def _closest_point_on_triangle(point, tri):
    # Ericson, Real-Time Collision Detection, section 5.1.5.
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)

    denom = 1.0 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)


def _closest_points_to_surface(query_points, triangles, k=24):
    if len(triangles) == 0:
        raise RuntimeError("Cannot build clinical frame: requested surface has no triangles")
    centers = triangles.mean(axis=1)
    tree = cKDTree(centers)
    k_eff = min(k, len(triangles))
    _, nearest = tree.query(query_points, k=k_eff)
    if nearest.ndim == 1:
        nearest = nearest[:, None]

    closest = np.empty_like(query_points)
    distances = np.empty(len(query_points), dtype=float)
    for i, point in enumerate(query_points):
        best_point = None
        best_d2 = np.inf
        for tri_idx in nearest[i]:
            candidate = _closest_point_on_triangle(point, triangles[int(tri_idx)])
            d2 = float(np.dot(point - candidate, point - candidate))
            if d2 < best_d2:
                best_d2 = d2
                best_point = candidate
        closest[i] = best_point
        distances[i] = best_d2 ** 0.5
    return closest, distances


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
