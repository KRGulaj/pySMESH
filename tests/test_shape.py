"""Tier-1 geometry-query tests: load_brep, Shape queries, face_distance.

Fixtures are committed BREPs from ``tests/fixtures/generate_fixtures.cpp``; the box is an
axis-aligned cube of edge length 2.0 with its min corner at the origin, so analytically it
has 6 faces, 12 edges, 8 vertices, and total surface area 6 * 2^2 = 24.
"""

from __future__ import annotations

import numpy as np
import pytest

from pysmesh import PysmeshError, load_brep

BOX_EDGE = 2.0
BOX_CENTER = np.array([1.0, 1.0, 1.0])


def test_load_brep_box_face_edge_vertex_counts(box_brep: bytes) -> None:
    shape = load_brep(box_brep)

    assert len(shape.faces()) == 6
    assert len(shape.edges()) == 12
    assert len(shape.vertices()) == 8


def test_load_brep_box_total_area_equals_six_a_squared(box_brep: bytes) -> None:
    shape = load_brep(box_brep)

    total_area = sum(f.area for f in shape.faces())

    assert total_area == pytest.approx(6.0 * BOX_EDGE**2)  # 24.0


def test_face_ids_are_one_based_contiguous(box_brep: bytes) -> None:
    shape = load_brep(box_brep)

    ids = sorted(f.id for f in shape.faces())

    assert ids == list(range(1, 7))


def test_face_distance_on_surface_is_zero(box_brep: bytes) -> None:
    shape = load_brep(box_brep)

    for face in shape.faces():
        # Each face centroid lies on its face -> distance 0.
        point = np.asarray(face.centroid, dtype=np.float64).reshape(1, 3)
        dist = shape.face_distance(face.id, point)
        assert dist[0] == pytest.approx(0.0, abs=1e-9)


def test_face_distance_off_surface_equals_normal_offset(box_brep: bytes) -> None:
    shape = load_brep(box_brep)
    offset = 0.5

    for face in shape.faces():
        centroid = np.asarray(face.centroid, dtype=np.float64)
        # Outward normal of a box face = direction from box center to the face centroid.
        outward = centroid - BOX_CENTER
        outward /= np.linalg.norm(outward)
        point = (centroid + offset * outward).reshape(1, 3)

        dist = shape.face_distance(face.id, point)

        # Closest point is the face-interior centroid -> distance is exactly the offset.
        assert dist[0] == pytest.approx(offset, abs=1e-9)


def test_face_distance_bad_face_id_raises(box_brep: bytes) -> None:
    shape = load_brep(box_brep)
    point = np.zeros((1, 3), dtype=np.float64)

    with pytest.raises(PysmeshError, match="face_id"):
        shape.face_distance(999, point)


def test_load_brep_empty_bytes_raises() -> None:
    with pytest.raises(PysmeshError):
        load_brep(b"")
