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


# ---------------------------------------------------------------------------
# A4 — FaceInfo.surface_type (BRepAdaptor_Surface::GetType)
# ---------------------------------------------------------------------------


def test_surface_type_box_all_planes(box_brep: bytes) -> None:
    """Every face of an axis-aligned box is a plane."""
    shape = load_brep(box_brep)

    types = [f.surface_type for f in shape.faces()]

    assert types == ["Plane"] * 6


def test_surface_type_is_str(box_brep: bytes) -> None:
    """surface_type is a plain Python str."""
    shape = load_brep(box_brep)

    assert isinstance(shape.faces()[0].surface_type, str)


def test_surface_type_cylinder_one_wall_two_caps(cylinder_brep: bytes) -> None:
    """A cylinder is one lateral (Cylinder) surface plus two planar caps."""
    shape = load_brep(cylinder_brep)

    counts: dict[str, int] = {}
    for f in shape.faces():
        counts[f.surface_type] = counts.get(f.surface_type, 0) + 1

    assert counts == {"Cylinder": 1, "Plane": 2}


def test_surface_type_sphere_is_sphere(sphere_brep: bytes) -> None:
    """The unit sphere has a single spherical face."""
    shape = load_brep(sphere_brep)

    types = [f.surface_type for f in shape.faces()]

    assert types == ["Sphere"]


# ---------------------------------------------------------------------------
# A4 — Shape.face_adjacency (edge->face ancestor walk)
# ---------------------------------------------------------------------------


def test_face_adjacency_box_one_triple_per_edge(box_brep: bytes) -> None:
    """A closed box has 12 manifold edges -> 12 adjacency triples."""
    shape = load_brep(box_brep)

    adj = shape.face_adjacency()

    assert len(adj) == 12


def test_face_adjacency_triples_are_ordered_and_in_range(box_brep: bytes) -> None:
    """Each triple is (face_i<face_j, edge_id) with all ids inside the shape's ranges."""
    shape = load_brep(box_brep)

    adj = shape.face_adjacency()

    for fi, fj, eid in adj:
        assert 1 <= fi < fj <= 6
        assert 1 <= eid <= 12


def test_face_adjacency_edge_ids_are_unique(box_brep: bytes) -> None:
    """Every box edge borders exactly two faces, so each appears once."""
    shape = load_brep(box_brep)

    edge_ids = [eid for _, _, eid in shape.face_adjacency()]

    assert sorted(edge_ids) == list(range(1, 13))


def test_face_adjacency_box_each_face_has_degree_four(box_brep: bytes) -> None:
    """Every box face neighbours the four faces it does not oppose."""
    shape = load_brep(box_brep)

    degree: dict[int, int] = {i: 0 for i in range(1, 7)}
    for fi, fj, _ in shape.face_adjacency():
        degree[fi] += 1
        degree[fj] += 1

    assert all(d == 4 for d in degree.values())


def test_face_adjacency_sphere_single_face_is_empty(sphere_brep: bytes) -> None:
    """A one-face sphere: the seam edge lists the same face twice -> no distinct pair."""
    shape = load_brep(sphere_brep)

    assert shape.face_adjacency() == []


def test_face_adjacency_cylinder_wall_touches_both_caps(cylinder_brep: bytes) -> None:
    """The lateral (Cylinder) face is adjacent to exactly the two planar caps."""
    shape = load_brep(cylinder_brep)
    wall_id = next(f.id for f in shape.faces() if f.surface_type == "Cylinder")

    neighbours = {
        (fj if fi == wall_id else fi)
        for fi, fj, _ in shape.face_adjacency()
        if wall_id in (fi, fj)
    }

    assert len(neighbours) == 2


# ---------------------------------------------------------------------------
# A4 — Shape.match_faces (nearest face by centroid)
# ---------------------------------------------------------------------------


def test_match_faces_own_centroids_is_identity(box_brep: bytes) -> None:
    """Querying each face's own centroid returns that face's 1-based id."""
    shape = load_brep(box_brep)
    faces = shape.faces()
    centroids = np.array([f.centroid for f in faces], dtype=np.float64)

    ids = shape.match_faces(centroids, tol=1e-6)

    assert list(ids) == [f.id for f in faces]


def test_match_faces_result_dtype_int32(box_brep: bytes) -> None:
    """The result is a (Q,) int32 array."""
    shape = load_brep(box_brep)
    centroids = np.array([shape.faces()[0].centroid], dtype=np.float64)

    ids = shape.match_faces(centroids, tol=1e-6)

    assert ids.dtype == np.int32
    assert ids.shape == (1,)


def test_match_faces_within_tol_matches(box_brep: bytes) -> None:
    """A centroid perturbed by less than tol still matches its face."""
    shape = load_brep(box_brep)
    face = shape.faces()[2]
    perturbed = np.asarray(face.centroid, dtype=np.float64) + 1e-4
    perturbed = perturbed.reshape(1, 3)

    ids = shape.match_faces(perturbed, tol=1e-3)

    assert int(ids[0]) == face.id


def test_match_faces_beyond_tol_returns_minus_one(box_brep: bytes) -> None:
    """A point farther than tol from every face centroid returns the -1 sentinel."""
    shape = load_brep(box_brep)

    ids = shape.match_faces(np.array([[100.0, 100.0, 100.0]]), tol=1e-3)

    assert int(ids[0]) == -1


def test_match_faces_mixed_batch(box_brep: bytes) -> None:
    """Matched and unmatched queries resolve elementwise."""
    shape = load_brep(box_brep)
    c0 = shape.faces()[0].centroid
    pts = np.array([c0, [100.0, 100.0, 100.0]], dtype=np.float64)

    ids = shape.match_faces(pts, tol=1e-6)

    assert int(ids[0]) == 1
    assert int(ids[1]) == -1


def test_match_faces_empty_query_returns_empty(box_brep: bytes) -> None:
    """An empty (0, 3) query returns an empty int32 array."""
    shape = load_brep(box_brep)

    ids = shape.match_faces(np.empty((0, 3), dtype=np.float64), tol=1e-6)

    assert ids.shape == (0,)


def test_match_faces_zero_tol_raises(box_brep: bytes) -> None:
    """tol <= 0 is rejected."""
    shape = load_brep(box_brep)

    with pytest.raises(PysmeshError, match="tol"):
        shape.match_faces(np.array([[1.0, 1.0, 1.0]]), tol=0.0)


def test_match_faces_wrong_shape_raises(box_brep: bytes) -> None:
    """A query array that is not (N, 3) is rejected."""
    shape = load_brep(box_brep)

    with pytest.raises(PysmeshError):
        shape.match_faces(np.array([[1.0, 1.0]]), tol=1e-6)
