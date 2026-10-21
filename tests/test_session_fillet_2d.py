# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for fillet_2d.

The face is an ``a x b`` rectangle on z = 0 with a corner at the origin. Rounding a corner
with radius ``r`` cuts away the spandrel between the corner and the arc: the square
``r x r`` less a quarter disc. Measured from the corner along each side:

* spandrel area ``r^2 (1 - pi / 4)``, centroid ``r (10 - 3 pi) / (12 - 3 pi)`` along each
  side (the square's centroid ``r / 2`` less the quarter disc's ``r - 4 r / (3 pi)``,
  weighted by area);
* arc length ``pi r / 2``; the arc's centroid lies ``2 sqrt(2) r / pi`` from the arc's
  centre (at ``(r, r)``) along the bisector, so at ``r - 2 r / pi`` along each side;
* each side through the corner loses ``r`` at that end.

All edges are lines and circle arcs, which OCCT measures exactly: 1e-12 relative.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, Session

A: float = 3.0
B: float = 7.0
R: float = 1.0
RTOL: float = 1e-12
ATOL: float = 1e-12 * B
SPANDREL_AREA: float = R * R * (1.0 - math.pi / 4.0)
SPANDREL_CENTROID: float = R * (10.0 - 3.0 * math.pi) / (12.0 - 3.0 * math.pi)
ARC_LENGTH: float = 0.5 * math.pi * R
ARC_CENTROID: float = R - 2.0 * R / math.pi

# A free face: the rectangle [0, 3] x [0, 7] on z = 0 with a circular hole of radius
# 0.5 at (1.5, 3.5). BREP text written by OCCT 8.0.1 (BRepBuilderAPI_MakeFace of the
# rectangle, Add of the reversed circle wire; BRepCheck_Analyzer valid).
HOLED_FACE_BREP: bytes = (
    b"\nCASCADE Topology V3, (c) Open Cascade\nLocations 0\nCurve2ds 4\n"
    b"1 0 0 0 1 \n1 0 0 1 0 \n1 3 0 0 1 \n1 0 7 1 0 \nCurves 5\n"
    b"1 0 0 0 0 1 0 \n1 0 0 0 1 0 0 \n1 3 0 0 0 1 0 \n1 0 7 0 1 0 0 \n"
    b"2 1.5 3.5 0 0 0 1 1 0 -0 -0 1 0 0.5\nPolygon3D 0\n"
    b"PolygonOnTriangulations 0\nSurfaces 1\n1 0 0 0 0 0 1 1 0 0 0 1 0 \n"
    b"Triangulations 0\n\nTShapes 13\nVe\n1e-07\n0 0 0\n0 0\n\n0101101\n*\n"
    b"Ve\n1e-07\n0 7 0\n0 0\n\n0101101\n*\nEd\n 1e-07 1 1 0\n1  1 0 0 7\n"
    b"2  1 1 0 0 7\n0\n\n0101000\n+13 0 -12 0 *\nVe\n1e-07\n3 0 0\n0 0\n\n"
    b"0101101\n*\nEd\n 1e-07 1 1 0\n1  2 0 0 3\n2  2 1 0 0 3\n0\n\n0101000\n"
    b"+13 0 -10 0 *\nVe\n1e-07\n3 7 0\n0 0\n\n0101101\n*\nEd\n 1e-07 1 1 0\n"
    b"1  3 0 0 7\n2  3 1 0 0 7\n0\n\n0101000\n+10 0 -8 0 *\nEd\n"
    b" 1e-07 1 1 0\n1  4 0 0 3\n2  4 1 0 0 3\n0\n\n0101000\n+12 0 -8 0 *\n"
    b"Wi\n\n0101100\n-11 0 +9 0 +7 0 -6 0 *\nVe\n1e-07\n2 3.5 0\n0 0\n\n"
    b"0101101\n*\nEd\n 1e-07 1 1 0\n1  5 0 0 6.28318530717959\n0\n\n"
    b"0101100\n+4 0 -4 0 *\nWi\n\n0101100\n+3 0 *\nFa\n0  1e-07 1 0\n\n"
    b"1101000\n+5 0 -2 0 *\n\n+1 0 "
)

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, per kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def new_ids(s: Session, kind: EntityKind, before: set[int]) -> list[EntityId]:
    """The ids of ``kind`` that are alive now and were not in ``before``."""
    return [EntityId(int(i)) for i in s.entities(kind) if int(i) not in before]


def rectangle_session() -> tuple[Session, EntityId]:
    """A free a x b rectangle on z = 0, a corner at the origin; its face id."""
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), A, B)
    return s, EntityId(int(s.entities(EntityKind.FACE)[0]))


def vertex_at(s: Session, point: tuple[float, float, float]) -> EntityId:
    """The one live vertex at ``point``."""
    table = s.entity_table(EntityKind.VERTEX)
    hits = np.flatnonzero(np.all(np.isclose(table.centroid, point, atol=ATOL), axis=1))
    assert hits.size == 1
    return EntityId(int(table.ids[hits[0]]))


def table_rows(
    s: Session, kind: EntityKind
) -> dict[int, tuple[float, NDArray[np.float64]]]:
    """Id -> (measure, centroid) for every live entity of ``kind``."""
    table = s.entity_table(kind)
    return {
        int(i): (float(table.measure[k]), table.centroid[k])
        for k, i in enumerate(table.ids)
    }


def centre_normal(s: Session, face: EntityId) -> NDArray[np.float64]:
    """The outward normal at the middle of the face's parameter range."""
    u0, u1, v0, v1 = s.face_parameter_bounds([face])[0]
    return np.asarray(
        s.surface_at(face, [(0.5 * (u0 + u1), 0.5 * (v0 + v1))]).normals[0]
    )


# ---- analytic ---------------------------------------------------------------------- #


def test_one_rounded_corner_removes_the_spandrel_area_and_moves_the_centroid() -> None:
    s, face = rectangle_session()
    centre = s.mass_properties([face]).centroid[0]
    area = A * B - SPANDREL_AREA
    spandrel = np.array([SPANDREL_CENTROID, SPANDREL_CENTROID, 0.0], dtype=np.float64)

    s.fillet_2d(face, [vertex_at(s, (0.0, 0.0, 0.0))], R)

    after = s.mass_properties([face])
    assert after.measure[0] == pytest.approx(area, rel=RTOL)
    expected = (A * B * centre - SPANDREL_AREA * spandrel) / area
    np.testing.assert_allclose(after.centroid[0], expected, rtol=RTOL, atol=ATOL)


def test_four_rounded_corners_give_the_rounded_rectangle() -> None:
    """Area ``a b - (4 - pi) r^2``; perimeter ``2 (a + b) - 8 r + 2 pi r``."""
    s, face = rectangle_session()
    centre = s.mass_properties([face]).centroid[0]
    corners = [EntityId(int(i)) for i in s.entities(EntityKind.VERTEX)]

    s.fillet_2d(face, corners, R)

    after = s.mass_properties([face])
    assert after.measure[0] == pytest.approx(A * B - (4.0 - math.pi) * R * R, rel=RTOL)
    np.testing.assert_allclose(after.centroid[0], centre, rtol=RTOL, atol=ATOL)
    perimeter = s.entity_table(EntityKind.EDGE).measure.sum()
    expected = 2.0 * (A + B) - 8.0 * R + 2.0 * math.pi * R
    assert perimeter == pytest.approx(expected, rel=RTOL)
    assert len(s.entities(EntityKind.EDGE)) == 8
    assert len(s.entities(EntityKind.VERTEX)) == 8


# ---- history against geometry -------------------------------------------------------- #


def test_every_id_denotes_the_analytic_sub_shape_after_one_rounded_corner() -> None:
    """The face and the two sides keep their ids, the arc and its ends are new."""
    s, face = rectangle_session()
    corner = vertex_at(s, (0.0, 0.0, 0.0))
    edges = table_rows(s, EntityKind.EDGE)
    vertices = table_rows(s, EntityKind.VERTEX)
    # A side from the corner at the origin has its centroid at half its length from it.
    at_corner = {
        i
        for i, (length, c) in edges.items()
        if np.isclose(np.linalg.norm(c), 0.5 * length, rtol=RTOL)
    }
    assert len(at_corner) == 2

    delta = s.fillet_2d(face, [corner], R)

    assert sorted(delta.deleted.tolist()) == [int(corner)]
    assert not s.is_alive(corner)
    assert s.is_alive(face)
    assert {int(face)} | at_corner <= set(delta.modified.tolist())
    after_edges = table_rows(s, EntityKind.EDGE)
    for i, (length, c) in edges.items():
        if i in at_corner:
            assert after_edges[i][0] == pytest.approx(length - R, rel=RTOL)
            np.testing.assert_allclose(
                after_edges[i][1], c * (length + R) / length, rtol=RTOL, atol=ATOL
            )
        else:
            assert after_edges[i][0] == pytest.approx(length, rel=RTOL)
            np.testing.assert_allclose(after_edges[i][1], c, rtol=RTOL, atol=ATOL)
    (arc,) = new_ids(s, EntityKind.EDGE, set(edges))
    assert after_edges[int(arc)][0] == pytest.approx(ARC_LENGTH, rel=RTOL)
    np.testing.assert_allclose(
        after_edges[int(arc)][1],
        (ARC_CENTROID, ARC_CENTROID, 0.0),
        rtol=RTOL,
        atol=ATOL,
    )
    after_vertices = table_rows(s, EntityKind.VERTEX)
    for i, (_, c) in vertices.items():
        if i != int(corner):
            np.testing.assert_allclose(after_vertices[i][1], c, rtol=RTOL, atol=ATOL)
    ends = new_ids(s, EntityKind.VERTEX, set(vertices))
    points = sorted(tuple(np.round(after_vertices[int(v)][1], 12)) for v in ends)
    assert points == [(0.0, R, 0.0), (R, 0.0, 0.0)]
    assert sorted(delta.created.tolist()) == sorted(int(i) for i in [arc, *ends])
    for i in [arc, *ends]:
        assert s.name_of(i).role is NameRole.GENERATED
        assert s.origin(i).sources.tolist() == [int(corner)]


def test_a_reversed_face_keeps_its_outward_normal() -> None:
    s, face = rectangle_session()
    s.change_orientation([face])
    np.testing.assert_allclose(centre_normal(s, face), (0.0, 0.0, -1.0), atol=1e-15)

    s.fillet_2d(face, [vertex_at(s, (0.0, 0.0, 0.0))], R)

    np.testing.assert_allclose(centre_normal(s, face), (0.0, 0.0, -1.0), atol=1e-15)


def test_a_translated_face_is_rounded_where_it_lies() -> None:
    s, face = rectangle_session()
    edges_before = {int(i) for i in s.entities(EntityKind.EDGE)}
    s.translate((10.0, 0.0, 0.0), [face])

    s.fillet_2d(face, [vertex_at(s, (10.0, 0.0, 0.0))], R)

    (arc,) = new_ids(s, EntityKind.EDGE, edges_before)
    arc_centroid = s.mass_properties([arc]).centroid[0]
    np.testing.assert_allclose(
        arc_centroid, (10.0 + ARC_CENTROID, ARC_CENTROID, 0.0), rtol=RTOL, atol=ATOL
    )
    assert s.mass_properties([face]).measure[0] == pytest.approx(
        A * B - SPANDREL_AREA, rel=RTOL
    )


# ---- determinism --------------------------------------------------------------------- #


def _rounded() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
    s, face = rectangle_session()
    s.fillet_2d(face, [EntityId(int(i)) for i in s.entities(EntityKind.VERTEX)], R)
    return s.brep(), id_space(s)


def test_two_fresh_sessions_give_identical_bytes_and_ids() -> None:
    first = _rounded()

    second = _rounded()

    assert first == second


# ---- refusals ------------------------------------------------------------------------ #


def assert_refused(s: Session, pattern: str, *args: object) -> None:
    """The call raises a PysmeshError matching ``pattern`` and leaves the session as is."""
    before = (s.brep(), id_space(s), s.op_count, s.issued_id_count)

    with pytest.raises(ps.PysmeshError, match=pattern):
        s.fillet_2d(*args)  # type: ignore[arg-type]

    assert (s.brep(), id_space(s), s.op_count, s.issued_id_count) == before


def test_an_empty_vertex_list_is_refused() -> None:
    s, face = rectangle_session()

    assert_refused(
        s, rf"fillet_2d: no vertex of face {int(face)} is named", face, [], R
    )


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_a_radius_that_is_not_positive_is_refused_with_its_value(
    radius: float,
) -> None:
    s, face = rectangle_session()
    corner = vertex_at(s, (0.0, 0.0, 0.0))

    assert_refused(
        s,
        rf"fillet_2d: the radius {radius:g} of the fillets on face {int(face)} is not "
        "positive",
        face,
        [corner],
        radius,
    )


def test_a_non_finite_radius_is_refused() -> None:
    s, face = rectangle_session()

    assert_refused(
        s, r"fillet_2d.*radius", face, [vertex_at(s, (0.0, 0.0, 0.0))], math.inf
    )


def test_a_radius_too_large_for_the_sides_is_refused_naming_the_vertex() -> None:
    s, face = rectangle_session()
    corner = vertex_at(s, (0.0, 0.0, 0.0))

    assert_refused(
        s,
        rf"fillet_2d: at vertex {int(corner)} of face {int(face)} no fillet of radius 5 "
        "was built: no arc of that radius fits between the two edges that meet there",
        face,
        [corner],
        5.0,
    )


def test_a_vertex_of_another_body_is_refused() -> None:
    s, face = rectangle_session()
    s.add_rectangle((10.0, 0.0, 0.0), (0.0, 0.0, 1.0), A, B)
    other = vertex_at(s, (10.0, 0.0, 0.0))

    assert_refused(
        s,
        rf"fillet_2d: vertex {int(other)} is not a vertex of face {int(face)}",
        face,
        [other],
        R,
    )


def test_a_vertex_named_twice_is_refused() -> None:
    s, face = rectangle_session()
    corner = vertex_at(s, (0.0, 0.0, 0.0))

    assert_refused(
        s,
        rf"fillet_2d: vertex {int(corner)} of face {int(face)} is named twice",
        face,
        [corner, corner],
        R,
    )


def test_a_dead_face_is_refused() -> None:
    s, face = rectangle_session()
    corner = vertex_at(s, (0.0, 0.0, 0.0))
    s.remove([face])

    assert_refused(s, rf"fillet_2d: entity {int(face)} is dead", face, [corner], R)


def test_an_edge_named_as_the_face_is_refused() -> None:
    s, _ = rectangle_session()
    edge = EntityId(int(s.entities(EntityKind.EDGE)[0]))

    assert_refused(
        s,
        rf"fillet_2d: entity {int(edge)} is a EDGE, not a FACE",
        edge,
        [vertex_at(s, (0.0, 0.0, 0.0))],
        R,
    )


def test_a_face_of_a_solid_is_refused_naming_face_and_solid() -> None:
    s = Session()
    s.add_box(A, B, 1.0)
    box = int(s.entities(EntityKind.SOLID)[0])
    pairs = s.adjacency(EntityKind.FACE, EntityKind.VERTEX)
    face = EntityId(int(pairs.ids[0]))
    corner = EntityId(int(pairs.related[0]))

    assert_refused(
        s,
        rf"fillet_2d: face {int(face)} bounds solid {box}; a 2D fillet rounds the "
        "corners of a free face only",
        face,
        [corner],
        R,
    )


def test_a_face_of_an_open_shell_is_refused_naming_the_body_type(
    open_box_shell_brep: bytes,
) -> None:
    s = Session()
    s.add_brep(open_box_shell_brep)
    pairs = s.adjacency(EntityKind.FACE, EntityKind.VERTEX)
    face = EntityId(int(pairs.ids[0]))
    corner = EntityId(int(pairs.related[0]))

    assert_refused(
        s,
        rf"fillet_2d: face {int(face)} belongs to a SHELL body",
        face,
        [corner],
        R,
    )


def test_a_curved_face_is_refused_naming_its_surface_type() -> None:
    s = Session()
    s.add_line((1.0, 0.0, 0.0), (1.0, 0.0, 2.0))
    s.revolve(
        [EntityId(int(s.entities(EntityKind.EDGE)[0]))],
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        0.5 * math.pi,
    )
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))
    corner = vertex_at(s, (1.0, 0.0, 0.0))

    assert_refused(
        s,
        rf"fillet_2d: face {int(face)} is a \w+ face; a 2D fillet rounds the corners of "
        "a planar face",
        face,
        [corner],
        0.1,
    )


def test_a_face_with_a_hole_is_refused_with_its_wire_count() -> None:
    """OCCT's ChFi2d_Builder joins every edge of the face into one wire.

    Measured on OCCT 8.0.1: on this face the result has one wire and BRepCheck_Analyzer
    rejects it.
    """
    s = Session()
    s.add_brep(HOLED_FACE_BREP)
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))
    corner = vertex_at(s, (0.0, 0.0, 0.0))

    assert_refused(
        s,
        rf"fillet_2d: face {int(face)} is bounded by 2 wires",
        face,
        [corner],
        R,
    )


def test_a_vertex_between_two_collinear_edges_is_refused_naming_it() -> None:
    s = Session()
    s.add_polyline(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 7.0, 0.0],
                [0.0, 7.0, 0.0],
            ],
            dtype=np.float64,
        ),
        closed=True,
    )
    s.make_face([EntityId(int(i)) for i in s.entities(EntityKind.EDGE)])
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))
    middle = vertex_at(s, (1.5, 0.0, 0.0))

    assert_refused(
        s,
        rf"fillet_2d: at vertex {int(middle)} of face {int(face)} no fillet of radius 0.5 "
        "was built: the two edges that meet there are tangent",
        face,
        [middle],
        0.5,
    )
