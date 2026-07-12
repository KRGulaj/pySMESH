"""Tests for pysmesh.point_in_solid (A5 / D1).

Covers: analytical inside/outside/on-boundary classification on the unit box and sphere,
batch masks, dtype/shape, the strict-IN contract (boundary points are False), non-solid
input rejection, and parameter validation.

Reference spec:
  point_in_solid: BRepClass3d_SolidClassifier — mask[i] True iff points[i] is strictly inside
    the solid (TopAbs_IN). The box fixture spans [0, 2]^3, so (1, 1, 1) is inside, (5, 5, 5)
    is outside, and any face point (e.g. (0, 1, 1)) is ON -> False. The unit sphere (radius 1
    at origin) contains the origin and excludes (2, 0, 0).
"""

from __future__ import annotations

import numpy as np
import pytest

import pysmesh
from pysmesh import PysmeshError, point_in_solid


# ---------------------------------------------------------------------------
# Analytical classification — box
# ---------------------------------------------------------------------------


def test_point_in_solid_interior_point_is_true(box_brep: bytes) -> None:
    """The box centroid (1, 1, 1) is strictly inside."""
    mask = point_in_solid(box_brep, np.array([[1.0, 1.0, 1.0]]))
    assert bool(mask[0]) is True


def test_point_in_solid_exterior_point_is_false(box_brep: bytes) -> None:
    """A point well outside the box is False."""
    mask = point_in_solid(box_brep, np.array([[5.0, 5.0, 5.0]]))
    assert bool(mask[0]) is False


def test_point_in_solid_negative_octant_point_is_false(box_brep: bytes) -> None:
    """A point outside on the min side is False."""
    mask = point_in_solid(box_brep, np.array([[-1.0, -1.0, -1.0]]))
    assert bool(mask[0]) is False


def test_point_in_solid_boundary_point_is_false(box_brep: bytes) -> None:
    """A point on a face (x = 0) is ON, not IN, so False under the strict contract."""
    mask = point_in_solid(box_brep, np.array([[0.0, 1.0, 1.0]]), tol=1e-7)
    assert bool(mask[0]) is False


def test_point_in_solid_corner_point_is_false(box_brep: bytes) -> None:
    """A box corner is on the boundary -> False."""
    mask = point_in_solid(box_brep, np.array([[0.0, 0.0, 0.0]]), tol=1e-7)
    assert bool(mask[0]) is False


def test_point_in_solid_batch_mask(box_brep: bytes) -> None:
    """A batch of points yields the elementwise inside mask."""
    pts = np.array(
        [
            [1.0, 1.0, 1.0],   # inside
            [5.0, 5.0, 5.0],   # outside
            [0.5, 0.5, 0.5],   # inside
            [3.0, 1.0, 1.0],   # outside
        ]
    )
    mask = point_in_solid(box_brep, pts)
    assert list(mask) == [True, False, True, False]


def test_point_in_solid_result_dtype_and_shape(box_brep: bytes) -> None:
    """Result is a (N,) bool array."""
    pts = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    mask = point_in_solid(box_brep, pts)
    assert mask.dtype == np.bool_
    assert mask.shape == (2,)


def test_point_in_solid_accepts_list_input(box_brep: bytes) -> None:
    """A plain nested list is coerced to (N, 3) float64."""
    mask = point_in_solid(box_brep, [[1.0, 1.0, 1.0]])
    assert bool(mask[0]) is True


def test_point_in_solid_empty_points_returns_empty(box_brep: bytes) -> None:
    """An empty (0, 3) query returns an empty mask."""
    mask = point_in_solid(box_brep, np.empty((0, 3), dtype=np.float64))
    assert mask.shape == (0,)


# ---------------------------------------------------------------------------
# Analytical classification — sphere (curved)
# ---------------------------------------------------------------------------


def test_point_in_solid_sphere_centre_is_true(sphere_brep: bytes) -> None:
    """The origin is inside the unit sphere."""
    mask = point_in_solid(sphere_brep, np.array([[0.0, 0.0, 0.0]]))
    assert bool(mask[0]) is True


def test_point_in_solid_sphere_outside_is_false(sphere_brep: bytes) -> None:
    """A point beyond the unit radius is outside the sphere."""
    mask = point_in_solid(sphere_brep, np.array([[2.0, 0.0, 0.0]]))
    assert bool(mask[0]) is False


# ---------------------------------------------------------------------------
# Input / parameter validation
# ---------------------------------------------------------------------------


def test_point_in_solid_non_solid_shell_raises(open_box_shell_brep: bytes) -> None:
    """An open shell has no interior -> PysmeshError (C++ layer)."""
    with pytest.raises(PysmeshError):
        point_in_solid(open_box_shell_brep, np.array([[1.0, 1.0, 1.0]]))


def test_point_in_solid_zero_tol_raises(box_brep: bytes) -> None:
    """PysmeshError when tol <= 0 (Python layer)."""
    with pytest.raises(PysmeshError):
        point_in_solid(box_brep, np.array([[1.0, 1.0, 1.0]]), tol=0.0)


def test_point_in_solid_wrong_shape_raises(box_brep: bytes) -> None:
    """PysmeshError when points is not (N, 3)."""
    with pytest.raises(PysmeshError):
        point_in_solid(box_brep, np.array([[1.0, 1.0]]))


def test_point_in_solid_malformed_brep_raises() -> None:
    """PysmeshError on garbage BREP bytes."""
    with pytest.raises(PysmeshError):
        point_in_solid(b"garbage", np.array([[0.0, 0.0, 0.0]]))


# ---------------------------------------------------------------------------
# Public namespace
# ---------------------------------------------------------------------------


def test_public_namespace_exports() -> None:
    """point_in_solid is importable from pysmesh directly."""
    assert hasattr(pysmesh, "point_in_solid")
    assert pysmesh.point_in_solid is point_in_solid
