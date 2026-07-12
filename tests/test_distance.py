"""Tests for pysmesh.shape_distance and pysmesh.free_boundary_edges (A3 / D1).

Covers: exact analytical minimum distance between separated boxes, witness-point geometry,
the touching/overlap degenerate case, free-boundary detection on watertight solids (empty)
vs. an open shell (the four naked edges of a removed face), id alignment with Shape.edges(),
parameter validation, and public-namespace exports.

Reference specs:
  shape_distance: BRepExtrema_DistShapeShape — exact minimum distance between two shapes and
    the pair of witness points, one on each shape. Two axis-aligned boxes with a 3.0 gap
    along +x have distance exactly 3.0 (a purely translational analytical case).
  free_boundary_edges: an edge is free iff bordered by exactly one face. A watertight solid
    (box/cylinder/sphere) has none; removing one face of a box opens four naked edges.
"""

from __future__ import annotations

import numpy as np
import pytest

import pysmesh
from pysmesh import (
    PysmeshError,
    ShapeDistanceResult,
    free_boundary_edges,
    load_brep,
    shape_distance,
)
from pysmesh.distance import ShapeDistanceResult as _SDResultDirect


# ---------------------------------------------------------------------------
# shape_distance — analytical distance
# ---------------------------------------------------------------------------


def test_shape_distance_separated_boxes_returns_3(box_brep: bytes, box_far_brep: bytes) -> None:
    """Box (x in [0,2]) to box_far (x in [5,7]) has exact minimum distance 3.0."""
    r = shape_distance(box_brep, box_far_brep)
    assert r.distance == pytest.approx(3.0, abs=1e-9)


def test_shape_distance_witness_points_realise_the_distance(
    box_brep: bytes, box_far_brep: bytes
) -> None:
    """||point_b - point_a|| equals the reported distance (points realise the minimum)."""
    r = shape_distance(box_brep, box_far_brep)
    gap = float(np.linalg.norm(r.point_b - r.point_a))
    assert gap == pytest.approx(r.distance, abs=1e-9)


def test_shape_distance_witness_points_on_facing_planes(
    box_brep: bytes, box_far_brep: bytes
) -> None:
    """The minimum is perpendicular: witness x-coords sit on the facing planes (2 and 5),
    with identical y, z (the separation is purely along +x)."""
    r = shape_distance(box_brep, box_far_brep)
    assert r.point_a[0] == pytest.approx(2.0, abs=1e-9)
    assert r.point_b[0] == pytest.approx(5.0, abs=1e-9)
    assert r.point_a[1] == pytest.approx(r.point_b[1], abs=1e-9)
    assert r.point_a[2] == pytest.approx(r.point_b[2], abs=1e-9)


def test_shape_distance_identical_shapes_is_zero(box_brep: bytes) -> None:
    """Degenerate case: distance between a shape and itself is 0 (they overlap)."""
    r = shape_distance(box_brep, box_brep)
    assert r.distance == pytest.approx(0.0, abs=1e-9)


def test_shape_distance_result_types(box_brep: bytes, box_far_brep: bytes) -> None:
    """distance is a float; witness points are (3,) float64 arrays."""
    r = shape_distance(box_brep, box_far_brep)
    assert isinstance(r.distance, float)
    assert r.point_a.shape == (3,)
    assert r.point_b.shape == (3,)
    assert r.point_a.dtype == np.float64
    assert r.point_b.dtype == np.float64


def test_shape_distance_result_is_frozen() -> None:
    """ShapeDistanceResult is a frozen dataclass (immutable)."""
    r = ShapeDistanceResult(
        distance=1.0, point_a=np.zeros(3), point_b=np.ones(3)
    )
    with pytest.raises((AttributeError, TypeError)):
        r.distance = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# shape_distance — parameter validation
# ---------------------------------------------------------------------------


def test_shape_distance_malformed_first_brep_raises(box_brep: bytes) -> None:
    """PysmeshError when the first BREP is garbage."""
    with pytest.raises(PysmeshError):
        shape_distance(b"not-a-brep", box_brep)


def test_shape_distance_malformed_second_brep_raises(box_brep: bytes) -> None:
    """PysmeshError when the second BREP is garbage."""
    with pytest.raises(PysmeshError):
        shape_distance(box_brep, b"garbage")


# ---------------------------------------------------------------------------
# free_boundary_edges — watertight solids (no free boundary)
# ---------------------------------------------------------------------------


def test_free_boundary_edges_closed_box_is_empty(box_brep: bytes) -> None:
    """A watertight box has no naked edges — every edge is shared by two faces."""
    ids = free_boundary_edges(box_brep)
    assert ids.dtype == np.int32
    assert ids.size == 0


def test_free_boundary_edges_closed_cylinder_is_empty(cylinder_brep: bytes) -> None:
    """A watertight cylinder has no free boundary (the seam edge is shared, not naked)."""
    ids = free_boundary_edges(cylinder_brep)
    assert ids.size == 0


def test_free_boundary_edges_closed_sphere_is_empty(sphere_brep: bytes) -> None:
    """A watertight sphere has no free boundary (degenerate pole edges are excluded)."""
    ids = free_boundary_edges(sphere_brep)
    assert ids.size == 0


# ---------------------------------------------------------------------------
# free_boundary_edges — open shell (the hole)
# ---------------------------------------------------------------------------


def test_free_boundary_edges_open_shell_has_four(open_box_shell_brep: bytes) -> None:
    """Removing one face of the box opens exactly four naked edges (the opening rim)."""
    ids = free_boundary_edges(open_box_shell_brep)
    assert ids.size == 4


def test_free_boundary_edges_ids_are_valid_and_unique(open_box_shell_brep: bytes) -> None:
    """Free-edge ids are distinct 1-based ordinals within the shape's edge range."""
    shape = load_brep(open_box_shell_brep)
    n_edges = len(shape.edges())
    ids = free_boundary_edges(open_box_shell_brep)
    assert len(set(ids.tolist())) == ids.size
    for eid in ids:
        assert 1 <= eid <= n_edges


def test_free_boundary_edges_ids_ascending(open_box_shell_brep: bytes) -> None:
    """Returned ids are in ascending order (loop over the 1-based edge map)."""
    ids = free_boundary_edges(open_box_shell_brep)
    assert list(ids) == sorted(ids)


def test_free_boundary_edges_ids_index_shape_edges(open_box_shell_brep: bytes) -> None:
    """Each free-edge id resolves to a real EdgeInfo of positive length in Shape.edges()."""
    shape = load_brep(open_box_shell_brep)
    edges = {e.id: e for e in shape.edges()}
    ids = free_boundary_edges(open_box_shell_brep)
    for eid in ids:
        assert edges[int(eid)].length > 0.0


def test_free_boundary_edges_malformed_brep_raises() -> None:
    """PysmeshError on garbage BREP bytes."""
    with pytest.raises(PysmeshError):
        free_boundary_edges(b"garbage")


# ---------------------------------------------------------------------------
# Public namespace
# ---------------------------------------------------------------------------


def test_public_namespace_exports() -> None:
    """All A3 names are importable from pysmesh directly."""
    assert hasattr(pysmesh, "shape_distance")
    assert hasattr(pysmesh, "free_boundary_edges")
    assert hasattr(pysmesh, "ShapeDistanceResult")
    assert pysmesh.shape_distance is shape_distance
    assert pysmesh.free_boundary_edges is free_boundary_edges
    assert pysmesh.ShapeDistanceResult is _SDResultDirect
