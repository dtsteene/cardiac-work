"""Low-level geometric primitives shared across the pipeline.

Pure NumPy/SciPy helpers with no FEniCSx or module-global dependencies, so
they are cheap to import and trivially unit-testable. Previously these were
copy-pasted verbatim into both ``clinical_frame.py`` and
``compute_per_cell.py``; they live here once now.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def closest_point_on_triangle(point, tri):
    """Closest point on a single triangle to ``point``.

    Ericson, *Real-Time Collision Detection*, section 5.1.5. ``tri`` is a
    ``(3, 3)`` array of vertex coordinates; returns the closest point (which
    may lie on a vertex, edge, or face interior).
    """
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


def closest_points_to_surface(query_points, triangles, k=24):
    """For each query point, the closest point on a triangulated surface.

    ``triangles`` is an ``(N, 3, 3)`` array. Uses a KD-tree over triangle
    centroids to shortlist the ``k`` nearest triangles per query, then an exact
    point-to-triangle test on each candidate. Returns ``(closest, distances)``.
    """
    if len(triangles) == 0:
        raise RuntimeError("Cannot build surface frame: requested surface has no triangles")
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
            candidate = closest_point_on_triangle(point, triangles[int(tri_idx)])
            d2 = float(np.dot(point - candidate, point - candidate))
            if d2 < best_d2:
                best_d2 = d2
                best_point = candidate
        closest[i] = best_point
        distances[i] = best_d2 ** 0.5
    return closest, distances


__all__ = ["closest_point_on_triangle", "closest_points_to_surface"]
