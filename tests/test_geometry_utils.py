"""Unit tests for geometry_utils — pure NumPy, safe to run on the login node.

    python3 tests/test_geometry_utils.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry_utils import closest_point_on_triangle, closest_points_to_surface


# Reference triangle in the z=0 plane: a=(0,0,0), b=(1,0,0), c=(0,1,0)
TRI = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_point_above_interior_projects_down():
    p = np.array([0.25, 0.25, 1.0])
    cp = closest_point_on_triangle(p, TRI)
    assert np.allclose(cp, [0.25, 0.25, 0.0]), cp


def test_point_past_vertex_a_returns_a():
    p = np.array([-1.0, -1.0, 0.0])
    cp = closest_point_on_triangle(p, TRI)
    assert np.allclose(cp, [0.0, 0.0, 0.0]), cp


def test_point_past_vertex_b_returns_b():
    p = np.array([2.0, 0.0, 0.0])
    cp = closest_point_on_triangle(p, TRI)
    assert np.allclose(cp, [1.0, 0.0, 0.0]), cp


def test_point_off_edge_ab_projects_onto_edge():
    p = np.array([0.5, -1.0, 0.0])
    cp = closest_point_on_triangle(p, TRI)
    assert np.allclose(cp, [0.5, 0.0, 0.0]), cp


def test_closest_points_to_surface_distances():
    queries = np.array([
        [0.25, 0.25, 1.0],   # 1.0 above interior
        [2.0, 0.0, 0.0],     # 1.0 past vertex b
        [0.5, -1.0, 0.0],    # 1.0 off edge ab
    ])
    closest, distances = closest_points_to_surface(queries, TRI[None, :, :], k=1)
    assert np.allclose(distances, [1.0, 1.0, 1.0]), distances
    assert np.allclose(closest[0], [0.25, 0.25, 0.0]), closest[0]


def test_empty_surface_raises():
    try:
        closest_points_to_surface(np.zeros((1, 3)), np.empty((0, 3, 3)))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on empty surface")


def test_two_triangle_surface_picks_nearer():
    # Second triangle shifted up by z=5; nearest must be the z=0 one.
    tris = np.stack([TRI, TRI + np.array([0.0, 0.0, 5.0])])
    closest, distances = closest_points_to_surface(np.array([[0.25, 0.25, 1.0]]), tris, k=2)
    assert np.allclose(distances, [1.0]), distances
    assert np.allclose(closest[0], [0.25, 0.25, 0.0]), closest[0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
