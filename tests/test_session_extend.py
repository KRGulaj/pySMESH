# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for extend_face and extend_edge.

Each extension is checked against a closed form:

* A rectangle ``a x b`` on its plane frame ``(O, X, Y)`` extended by ``u_min``, ``u_max``,
  ``v_min``, ``v_max`` is the rectangle ``[-u_min, a + u_max] x [-v_min, b + v_max]``: area
  ``(a + u_min + u_max)(b + v_min + v_max)``, centroid
  ``O + (a + u_max - u_min) / 2 X + (b + v_max - v_min) / 2 Y``.
* A cylinder of radius ``r`` and height ``h`` extended by ``d`` along its axis has the side
  area ``2 pi r (h + d)``.
* A straight edge from ``p`` to ``q`` extended by ``d`` at ``q`` ends at
  ``q + d (q - p) / |q - p|`` and has the length ``|q - p| + d``.
* A quarter circle of radius 2 from (2, 0, 0) to (0, 2, 0), extended by 1 at its end, ends
  at (-1, 2, 0), one unit along the tangent (-1, 0, 0); its first ``pi`` of length is the
  arc, so the point there is (0, 2, 0) with the tangent (-1, 0, 0).

The extended surfaces and curves are B-splines that OCCT integrates adaptively, so measures
are compared at 1e-9 relative; points that GeomLib places exactly at 1e-12.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, Session

RECT_A: float = 3.0
RECT_B: float = 7.0
MEASURE_RTOL: float = 1e-9
POINT_ATOL: float = 1e-12

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)

# A face on the infinite plane z = 0, with no wire, as OCCT 8.0.1 writes it
# (BRepBuilderAPI_MakeFace(gp_Pln) then BRepTools::Write). BRepCheck_Analyzer accepts it.
INFINITE_PLANE_FACE_BREP: bytes = (
    b"\nCASCADE Topology V3, (c) Open Cascade\nLocations 0\nCurve2ds 0\nCurves 0\n"
    b"Polygon3D 0\nPolygonOnTriangulations 0\nSurfaces 1\n"
    b"1 0 0 0 0 0 1 1 0 -0 -0 1 0 \nTriangulations 0\n\nTShapes 1\nFa\n1  1e-07 1 0\n\n"
    b"1101000\n*\n\n+1 0 \n"
)


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, per kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def rectangle_session() -> tuple[Session, EntityId]:
    """A free rectangle 3 x 7 on z = 0 with its corner at the origin."""
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), RECT_A, RECT_B)
    return s, EntityId(int(s.entities(EntityKind.FACE)[0]))


def new_ids(s: Session, kind: EntityKind, before: set[int]) -> list[EntityId]:
    """The live ids of one kind that were not live before."""
    return [EntityId(int(i)) for i in s.entities(kind) if int(i) not in before]


# ---- extend_face --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("u_min", "u_max", "v_min", "v_max"),
    [(0.0, 1.0, 0.0, 0.0), (0.5, 0.0, 0.0, 2.0), (1.0, 1.5, 0.25, 0.75)],
)
def test_extend_face_of_a_rectangle_is_the_larger_rectangle(
    u_min: float, u_max: float, v_min: float, v_max: float
) -> None:
    s, face = rectangle_session()
    p = s.surface_parameters([face])
    x = np.asarray(p.ref_dir[0])
    y = np.cross(p.axis[0], x)
    faces_before = {int(i) for i in s.entities(EntityKind.FACE)}

    s.extend_face(face, u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)

    (new,) = new_ids(s, EntityKind.FACE, faces_before)
    m = s.mass_properties([new], precision=1e-12)
    a, b = RECT_A + u_min + u_max, RECT_B + v_min + v_max
    assert m.measure[0] == pytest.approx(a * b, rel=MEASURE_RTOL)
    want = (0.5 * (RECT_A + u_max - u_min)) * x + (0.5 * (RECT_B + v_max - v_min)) * y
    np.testing.assert_allclose(m.centroid[0], want, rtol=MEASURE_RTOL, atol=1e-9)


def test_extend_face_keeps_the_source_and_names_the_new_face_after_it() -> None:
    s, face = rectangle_session()
    before = {k: {int(i) for i in s.entities(k)} for k in _KINDS}
    area = s.mass_properties([face]).measure[0]

    delta = s.extend_face(face, u_max=1.0)

    assert s.is_alive(face)
    assert s.mass_properties([face]).measure[0] == pytest.approx(area, rel=1e-15)
    created = set(delta.created.tolist())
    for kind in (EntityKind.FACE, EntityKind.EDGE, EntityKind.VERTEX):
        fresh = new_ids(s, kind, before[kind])
        assert fresh
        for i in fresh:
            assert int(i) in created
            assert s.name_of(i).role is NameRole.GENERATED
            assert s.origin(i).sources.tolist() == [int(face)]
    assert delta.deleted.size == 0


def test_extend_face_keeps_the_outward_normal_of_the_source() -> None:
    s, face = rectangle_session()
    s.change_orientation([face])
    faces_before = {int(i) for i in s.entities(EntityKind.FACE)}

    s.extend_face(face, v_max=1.0)

    (new,) = new_ids(s, EntityKind.FACE, faces_before)
    u0, u1, v0, v1 = s.face_parameter_bounds([new])[0]
    normal = s.surface_at(new, [(0.5 * (u0 + u1), 0.5 * (v0 + v1))]).normals[0]
    np.testing.assert_allclose(normal, (0.0, 0.0, -1.0), atol=1e-12)


def test_extend_face_of_a_cylinder_side_along_its_axis_adds_the_analytic_area() -> None:
    radius, height, extra = 2.0, 5.0, 1.5
    s = Session()
    s.add_cylinder(radius, height)
    params = s.surface_parameters(
        [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    )
    side = EntityId(int(params.ids[list(params.types).index("Cylinder")]))
    faces_before = {int(i) for i in s.entities(EntityKind.FACE)}

    s.extend_face(side, v_max=extra)

    (new,) = new_ids(s, EntityKind.FACE, faces_before)
    m = s.mass_properties([new], precision=1e-12)
    assert m.measure[0] == pytest.approx(
        2.0 * math.pi * radius * (height + extra), rel=1e-9
    )
    assert m.centroid[0][2] == pytest.approx(0.5 * (height + extra), rel=1e-9)


def test_extend_face_refuses_the_full_period_of_a_cylinder() -> None:
    s = Session()
    s.add_cylinder(2.0, 5.0)
    params = s.surface_parameters(
        [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    )
    side = EntityId(int(params.ids[list(params.types).index("Cylinder")]))

    with pytest.raises(
        ps.PysmeshError,
        match=rf"extend_face: face {int(side)} covers the whole period of its Cylinder "
        "surface in u",
    ):
        s.extend_face(side, u_max=1.0)


def test_extend_face_refuses_an_infinite_plane_naming_its_surface_type() -> None:
    s = Session()
    s.add_brep(INFINITE_PLANE_FACE_BREP)
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    with pytest.raises(
        ps.PysmeshError,
        match=rf"extend_face: face {int(face)} lies on an unbounded Plane surface and has no "
        "boundary",
    ):
        s.extend_face(face, u_max=1.0)


def test_extend_face_refuses_all_zero_lengths() -> None:
    s, face = rectangle_session()

    with pytest.raises(
        ps.PysmeshError, match=rf"extend_face: every extension of face {int(face)} is 0"
    ):
        s.extend_face(face)


def test_extend_face_refuses_a_negative_length_naming_it() -> None:
    s, face = rectangle_session()

    with pytest.raises(
        ps.PysmeshError,
        match=rf"extend_face: the extension v_min of face {int(face)} is -1; a length is 0",
    ):
        s.extend_face(face, v_min=-1.0)


def test_extend_face_refuses_a_length_that_is_not_finite() -> None:
    s, face = rectangle_session()

    with pytest.raises(ps.PysmeshError, match=r"extend_face"):
        s.extend_face(face, u_max=math.inf)


def test_extend_face_refuses_an_edge() -> None:
    s, _ = rectangle_session()
    edge = EntityId(int(s.entities(EntityKind.EDGE)[0]))

    with pytest.raises(ps.PysmeshError, match=rf"extend_face: entity {int(edge)} is a"):
        s.extend_face(edge, u_max=1.0)


def _extended_face() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
    s, face = rectangle_session()
    s.extend_face(face, u_min=0.5, v_max=2.0)
    return s.brep(), id_space(s)


def test_extend_face_gives_identical_bytes_and_ids_in_two_sessions() -> None:
    first = _extended_face()

    second = _extended_face()

    assert first == second


# ---- extend_edge --------------------------------------------------------------------- #


def line_session() -> tuple[Session, EntityId]:
    """A free straight edge from (0, 0, 0) to (3, 4, 0): length 5."""
    s = Session()
    s.add_line((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
    return s, EntityId(int(s.entities(EntityKind.EDGE)[0]))


def edge_ends(s: Session, edge: EntityId) -> tuple[np.ndarray, np.ndarray]:
    """The first and last point of an edge in its own orientation."""
    length = float(s.mass_properties([edge], precision=1e-12).measure[0])
    start = s.point_at_length(edge, 0.0).point
    end = s.point_at_length(edge, length).point
    return np.asarray(start), np.asarray(end)


@pytest.mark.parametrize(
    ("at_end", "start", "end"),
    [
        (True, (0.0, 0.0, 0.0), (4.2, 5.6, 0.0)),
        (False, (-1.2, -1.6, 0.0), (3.0, 4.0, 0.0)),
    ],
)
def test_extend_edge_of_a_line_is_the_longer_segment(
    at_end: bool, start: tuple[float, ...], end: tuple[float, ...]
) -> None:
    s, edge = line_session()
    before = {int(i) for i in s.entities(EntityKind.EDGE)}

    s.extend_edge(edge, 2.0, at_end=at_end)

    (new,) = new_ids(s, EntityKind.EDGE, before)
    length = s.mass_properties([new], precision=1e-12).measure[0]
    assert length == pytest.approx(7.0, rel=MEASURE_RTOL)
    got_start, got_end = edge_ends(s, new)
    np.testing.assert_allclose(got_start, start, atol=1e-9)
    np.testing.assert_allclose(got_end, end, atol=1e-9)
    table = s.bounding_boxes(EntityKind.VERTEX)
    corners = {tuple(np.round(row[:3], 12)) for row in table.bbox}
    assert tuple(np.round(end if at_end else start, 12)) in corners


def test_extend_edge_of_an_arc_ends_on_the_tangent_and_keeps_the_arc() -> None:
    s = Session()
    r = math.sqrt(2.0)
    s.add_arc((2.0, 0.0, 0.0), (r, r, 0.0), (0.0, 2.0, 0.0))
    arc = EntityId(int(s.entities(EntityKind.EDGE)[0]))
    before = {int(i) for i in s.entities(EntityKind.EDGE)}

    s.extend_edge(arc, 1.0)

    (new,) = new_ids(s, EntityKind.EDGE, before)
    start, end = edge_ends(s, new)
    np.testing.assert_allclose(start, (2.0, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(end, (-1.0, 2.0, 0.0), atol=1e-9)
    at_arc_end = s.point_at_length(new, math.pi)
    np.testing.assert_allclose(at_arc_end.point, (0.0, 2.0, 0.0), atol=1e-7)
    tangent = s.curve_at(new, [at_arc_end.parameter]).tangents[0]
    np.testing.assert_allclose(tangent, (-1.0, 0.0, 0.0), atol=1e-6)


def test_extend_edge_keeps_the_source_and_names_the_new_edge_after_it() -> None:
    s, edge = line_session()
    before = {k: {int(i) for i in s.entities(k)} for k in _KINDS}

    delta = s.extend_edge(edge, 2.0)

    assert s.is_alive(edge)
    created = set(delta.created.tolist())
    for kind in (EntityKind.EDGE, EntityKind.VERTEX):
        fresh = new_ids(s, kind, before[kind])
        assert fresh
        for i in fresh:
            assert int(i) in created
            assert s.name_of(i).role is NameRole.GENERATED
            assert s.origin(i).sources.tolist() == [int(edge)]


def test_extend_edge_refuses_a_closed_circle() -> None:
    s = Session()
    s.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0)
    circle = EntityId(int(s.entities(EntityKind.EDGE)[0]))

    with pytest.raises(
        ps.PysmeshError,
        match=rf"extend_edge: edge {int(circle)} is a closed Circle; it has no free end",
    ):
        s.extend_edge(circle, 1.0)


@pytest.mark.parametrize("length", [0.0, -2.0])
def test_extend_edge_refuses_a_length_that_is_not_positive(length: float) -> None:
    s, edge = line_session()

    with pytest.raises(
        ps.PysmeshError, match=rf"extend_edge: the extension of edge {int(edge)} is"
    ):
        s.extend_edge(edge, length)


def test_extend_edge_refuses_a_face() -> None:
    s, _ = rectangle_session()
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    with pytest.raises(ps.PysmeshError, match=rf"extend_edge: entity {int(face)} is a"):
        s.extend_edge(face, 1.0)


def _extended_edge() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
    s, edge = line_session()
    s.extend_edge(edge, 2.0, at_end=False)
    return s.brep(), id_space(s)


def test_extend_edge_gives_identical_bytes_and_ids_in_two_sessions() -> None:
    first = _extended_edge()

    second = _extended_edge()

    assert first == second
