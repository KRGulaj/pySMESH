# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for the block builders: make_quad, make_quad_from_vertices, make_hexa, multi_transform.

The answers are closed forms:

* A planar quadrangle's area is the shoelace formula ``|sum(x_i y_{i+1} - x_{i+1} y_i)| / 2``
  and its centroid the polygon centroid ``sum((p_i + p_{i+1}) c_i) / (6 A)``,
  ``c_i = x_i y_{i+1} - x_{i+1} y_i``.
* A quadrangle on two opposite parallel segments ``(0,0,0)-(2,0,0)`` and ``(0,3,1)-(2,3,1)``
  is a planar rectangle of sides 2 and ``sqrt(10)``.
* A box of ``2 x 3 x 4`` has volume 24, faces of area 6, 8 and 12 at the face centres.
* The hexahedron between the square ``[0,2]^2`` at ``z = 0`` and the parallelogram lifted to
  ``z = 1 + x / 2`` has volume ``int_0^2 int_0^2 (1 + x/2) dy dx = 6``, centroid
  ``(10/9, 1, 7/9)``, a top face of area ``|(2,0,1) x (0,2,0)| = sqrt(20)``, side faces of
  area 2 (``x = 0``), 4 (``x = 2``) and 3 (``y = 0`` and ``y = 2``, trapezoids).
* A ``3 x 2`` array of ``1 x 2 x 3`` blocks has volume 36 and ``(3 + 1) * 2 + (2 + 1) * 3 +
  2 * 6 = 29`` distinct faces once glued.
* The non-planar quadrangle on ``(0,0,0), (1,0,0), (1,1,1), (0,1,0)`` lies on GEOM's filling
  surface, a plate approximation: its boundary is the four edges, it is symmetric about the
  plane ``x = y`` (so is its boundary), and it projects onto the unit square (area > 1). The
  bilinear patch ``z = xy`` over that square has area
  ``int_0^1 int_0^1 sqrt(1 + x^2 + y^2) dx dy``; GEOM's filling differs from it by 7.3e-4
  relative (measured, p4/explore/blocks_explore2.log), so the two are compared at 1e-3.

Planar faces and boxes integrate exactly: 1e-12 relative.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, Session

RTOL: float = 1e-12
ATOL: float = 1e-12

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)

Point = tuple[float, float, float]


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, by kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def state(s: Session) -> tuple[bytes, tuple[tuple[int, ...], ...], int, int]:
    """The bytes, the live ids and the two counters."""
    return s.brep(), id_space(s), s.op_count, s.issued_id_count


def new_ids(s: Session, kind: EntityKind, before: set[int]) -> list[EntityId]:
    """The live ids of a kind that are not in `before`, ascending."""
    return [EntityId(int(i)) for i in s.entities(kind) if int(i) not in before]


def lines(s: Session, points: list[Point], *, closed: bool = True) -> list[EntityId]:
    """One loose line per side of the polygon, each its own body; their edge ids."""
    out: list[EntityId] = []
    n = len(points)
    for i in range(n if closed else n - 1):
        before = {int(e) for e in s.entities(EntityKind.EDGE)}
        s.add_line(points[i], points[(i + 1) % n])
        out += new_ids(s, EntityKind.EDGE, before)
    return out


def quad_on_lines(s: Session, points: list[Point]) -> EntityId:
    """A quadrangle face on four loose lines; its id."""
    before = {int(f) for f in s.entities(EntityKind.FACE)}
    s.make_quad(lines(s, points))
    (face,) = new_ids(s, EntityKind.FACE, before)
    return face


def rectangle(s: Session, origin: Point, dx: float, dy: float) -> EntityId:
    """A free rectangle face in a plane z = const; its id."""
    before = {int(f) for f in s.entities(EntityKind.FACE)}
    s.add_rectangle(origin, (0.0, 0.0, 1.0), dx, dy)
    (face,) = new_ids(s, EntityKind.FACE, before)
    return face


def shoelace(points: list[tuple[float, float]]) -> tuple[float, NDArray[np.float64]]:
    """The area and centroid of a simple polygon in the plane."""
    p = np.asarray(points, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    area = cross.sum() / 2.0
    centroid = ((p + q) * cross[:, None]).sum(axis=0) / (6.0 * area)
    return abs(area), centroid


def geometry(
    s: Session, ids: list[EntityId]
) -> dict[int, tuple[float, NDArray[np.float64]]]:
    """Each id's measure and centroid."""
    m = s.mass_properties(ids)
    return {
        int(i): (float(a), np.array(c))
        for i, a, c in zip(ids, m.measure, m.centroid, strict=True)
    }


# ---- make_quad -------------------------------------------------------------------------- #


def test_a_planar_quad_has_the_shoelace_area_and_centroid() -> None:
    corners = [(0.0, 0.0), (4.0, 0.0), (3.0, 2.0), (1.0, 3.0)]
    s = Session()
    edges = lines(s, [(x, y, 0.0) for x, y in corners])

    delta = s.make_quad([edges[0], edges[2], edges[1], edges[3]])

    (face,) = s.entities(EntityKind.FACE)
    area, centroid = shoelace(corners)
    m = s.mass_properties([EntityId(int(face))])
    np.testing.assert_allclose(m.measure, [area], rtol=RTOL)
    np.testing.assert_allclose(m.centroid[0], [*centroid, 0.0], rtol=RTOL, atol=ATOL)
    assert s.entity_types(EntityKind.FACE).types == ("Plane",)
    assert sorted(int(e) for e in s.entities(EntityKind.EDGE)) == sorted(
        map(int, edges)
    )
    assert list(delta.created) == [int(face)]
    assert delta.valid is True


def test_every_id_after_a_quad_denotes_its_analytic_geometry() -> None:
    """The history check: falsified by tearing the quad's history (final/falsify_*)."""
    corners: list[Point] = [
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (3.0, 2.0, 0.0),
        (0.0, 2.0, 0.0),
    ]
    s = Session()
    edges = lines(s, corners)
    vertices = [EntityId(int(v)) for v in s.entities(EntityKind.VERTEX)]
    before = geometry(s, edges + vertices)

    delta = s.make_quad(edges)

    assert len(delta.deleted) == 0
    assert sorted(int(i) for i in delta.modified) == sorted(map(int, edges + vertices))
    after = geometry(s, edges + vertices)
    for i, (measure, centroid) in before.items():
        assert math.isclose(after[i][0], measure, rel_tol=RTOL, abs_tol=ATOL), i
        np.testing.assert_allclose(after[i][1], centroid, rtol=RTOL, atol=ATOL)
    # Each corner is the end of two lines: both ids now denote the one welded vertex.
    groups = {tuple(int(i) for i in g) for g in s.alias_groups(EntityKind.VERTEX)}
    assert len(groups) == 4 and all(len(g) == 2 for g in groups)
    assert sorted(i for g in groups for i in g) == sorted(map(int, vertices))
    (face,) = delta.created
    origin = s.origin(EntityId(int(face)))
    assert origin.role == NameRole.GENERATED
    assert sorted(int(i) for i in origin.sources) == sorted(map(int, edges))
    m = s.mass_properties([EntityId(int(face))])
    np.testing.assert_allclose(m.measure, [6.0], rtol=RTOL)
    np.testing.assert_allclose(m.centroid[0], (1.5, 1.0, 0.0), rtol=RTOL, atol=ATOL)


def test_a_quad_between_two_opposite_edges_is_their_rectangle() -> None:
    s = Session()
    a = lines(s, [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], closed=False)
    b = lines(s, [(0.0, 3.0, 1.0), (2.0, 3.0, 1.0)], closed=False)
    before = {int(e) for e in s.entities(EntityKind.EDGE)}

    delta = s.make_quad(a + b)

    (face,) = s.entities(EntityKind.FACE)
    m = s.mass_properties([EntityId(int(face))])
    np.testing.assert_allclose(m.measure, [2.0 * math.sqrt(10.0)], rtol=RTOL)
    np.testing.assert_allclose(m.centroid[0], (1.0, 1.5, 0.5), rtol=RTOL, atol=ATOL)
    links = new_ids(s, EntityKind.EDGE, before)
    assert len(links) == 2
    np.testing.assert_allclose(
        s.mass_properties(links).measure, [math.sqrt(10.0)] * 2, rtol=RTOL
    )
    for link in links:
        origin = s.origin(link)
        assert origin.role == NameRole.GENERATED
        assert sorted(int(i) for i in origin.sources) == sorted(map(int, a + b))
    assert int(face) in delta.created


def test_a_quad_on_four_vertices_orders_them_by_the_shortest_perimeter() -> None:
    s = Session()
    for p in [(0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
        s.add_vertex(p)
    vertices = [EntityId(int(v)) for v in s.entities(EntityKind.VERTEX)]

    delta = s.make_quad_from_vertices(vertices)

    (face,) = s.entities(EntityKind.FACE)
    m = s.mass_properties([EntityId(int(face))])
    np.testing.assert_allclose(m.measure, [2.0], rtol=RTOL)
    np.testing.assert_allclose(m.centroid[0], (1.0, 0.5, 0.0), rtol=RTOL, atol=ATOL)
    edges = [EntityId(int(e)) for e in s.entities(EntityKind.EDGE)]
    np.testing.assert_allclose(
        np.sort(s.mass_properties(edges).measure), [1.0, 1.0, 2.0, 2.0], rtol=RTOL
    )
    assert sorted(int(v) for v in s.entities(EntityKind.VERTEX)) == sorted(
        map(int, vertices)
    )
    assert sorted(int(i) for i in s.origin(EntityId(int(face))).sources) == sorted(
        map(int, vertices)
    )
    assert len(delta.deleted) == 0


def test_a_non_planar_quad_lies_on_geoms_filling_bounded_by_the_edges() -> None:
    corners: list[Point] = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 0.0),
    ]
    s = Session()
    edges = lines(s, corners)

    delta = s.make_quad(edges)

    (face,) = (EntityId(int(f)) for f in s.entities(EntityKind.FACE))
    assert len(delta.deleted) == 0
    assert sorted(int(e) for e in s.entities(EntityKind.EDGE)) == sorted(
        map(int, edges)
    )
    assert s.entity_types(EntityKind.FACE).types == ("BSpline",)
    samples = np.array(
        [
            (1 - t) * np.asarray(corners[k]) + t * np.asarray(corners[(k + 1) % 4])
            for k in range(4)
            for t in np.linspace(0.0, 1.0, 9)
        ]
    )
    # BRepOffsetAPI_MakeFilling's default 3-D tolerance.
    assert float(np.max(s.project_on_face(face, samples).distance)) < 1e-4
    m = s.mass_properties([face], precision=1e-10)
    np.testing.assert_allclose(m.centroid[0][0], m.centroid[0][1], rtol=1e-9)
    nodes, weights = np.polynomial.legendre.leggauss(40)
    x = (nodes + 1.0) / 2.0
    w = weights / 2.0
    bilinear = float(
        np.sum(w[:, None] * w[None, :] * np.sqrt(1 + x[:, None] ** 2 + x**2))
    )
    assert 1.0 < float(m.measure[0])
    assert abs(float(m.measure[0]) - bilinear) / bilinear < 1e-3


# ---- make_hexa ------------------------------------------------------------------------- #


def six_faces_of_a_box(s: Session) -> list[EntityId]:
    """The six faces of the 2 x 3 x 4 box as separate bodies: two rectangles, four quads."""
    faces = [
        rectangle(s, (0.0, 0.0, 0.0), 2.0, 3.0),
        rectangle(s, (0.0, 0.0, 4.0), 2.0, 3.0),
    ]
    for quad in (
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 4.0), (0.0, 0.0, 4.0)],
        [(0.0, 3.0, 0.0), (2.0, 3.0, 0.0), (2.0, 3.0, 4.0), (0.0, 3.0, 4.0)],
        [(0.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 3.0, 4.0), (0.0, 0.0, 4.0)],
        [(2.0, 0.0, 0.0), (2.0, 3.0, 0.0), (2.0, 3.0, 4.0), (2.0, 0.0, 4.0)],
    ):
        faces.append(quad_on_lines(s, quad))  # type: ignore[arg-type]
    return faces


BOX_FACES: dict[tuple[float, float, float], float] = {
    (1.0, 1.5, 0.0): 6.0,
    (1.0, 1.5, 4.0): 6.0,
    (1.0, 0.0, 2.0): 8.0,
    (1.0, 3.0, 2.0): 8.0,
    (0.0, 1.5, 2.0): 12.0,
    (2.0, 1.5, 2.0): 12.0,
}


def assert_box(s: Session, solid: EntityId) -> None:
    """The solid is the 2 x 3 x 4 box: volume, centroid, every face by area and centre."""
    m = s.mass_properties([solid])
    np.testing.assert_allclose(m.measure, [24.0], rtol=RTOL)
    np.testing.assert_allclose(m.centroid[0], (1.0, 1.5, 2.0), rtol=RTOL, atol=ATOL)
    table = s.entity_table(EntityKind.FACE)
    found = {
        tuple(round(float(c), 9) for c in centroid): round(float(a), 9)
        for a, centroid in zip(table.measure, table.centroid, strict=True)
    }
    assert found == {k: v for k, v in BOX_FACES.items()}


def test_a_box_from_six_faces_has_the_box_volume_and_faces() -> None:
    s = Session()
    faces = six_faces_of_a_box(s)

    delta = s.make_hexa(faces)

    (solid,) = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))
    assert_box(s, solid)
    assert list(delta.created) == [int(solid)]
    assert len(delta.deleted) == 0
    assert s.origin(solid).role == NameRole.GENERATED
    assert sorted(int(i) for i in s.origin(solid).sources) == sorted(map(int, faces))
    assert delta.valid is True


def test_every_id_after_a_hexa_denotes_its_analytic_geometry() -> None:
    """The history check: falsified by tearing the hexa's history (final/falsify_*)."""
    s = Session()
    faces = six_faces_of_a_box(s)
    edges = [EntityId(int(e)) for e in s.entities(EntityKind.EDGE)]
    vertices = [EntityId(int(v)) for v in s.entities(EntityKind.VERTEX)]
    before = geometry(s, faces + edges + vertices)

    delta = s.make_hexa(faces)

    assert len(delta.deleted) == 0
    after = geometry(s, faces + edges + vertices)
    for i, (measure, centroid) in before.items():
        assert math.isclose(after[i][0], measure, rel_tol=RTOL, abs_tol=ATOL), i
        np.testing.assert_allclose(after[i][1], centroid, rtol=RTOL, atol=ATOL)
    # 12 edges and 8 corners, each now one shape: every edge id and vertex id survives on
    # the one sewn edge or vertex at its place.
    assert len(s.entities(EntityKind.EDGE, distinct=True)) == 12
    assert len(s.entities(EntityKind.VERTEX, distinct=True)) == 8
    assert sorted(int(i) for i in s.entities(EntityKind.EDGE)) == sorted(
        map(int, edges)
    )


def test_a_box_from_two_opposite_faces_builds_its_four_sides() -> None:
    s = Session()
    bottom = rectangle(s, (0.0, 0.0, 0.0), 2.0, 3.0)
    top = rectangle(s, (0.0, 0.0, 4.0), 2.0, 3.0)
    before = {int(f) for f in s.entities(EntityKind.FACE)}

    delta = s.make_hexa([bottom, top])

    (solid,) = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))
    assert_box(s, solid)
    sides = new_ids(s, EntityKind.FACE, before)
    assert len(sides) == 4
    for side in sides:
        assert s.origin(side).role == NameRole.GENERATED
    assert int(bottom) in delta.modified and int(top) in delta.modified


def test_a_skewed_hexa_between_non_parallel_quads_has_the_closed_form() -> None:
    s = Session()
    bottom = quad_on_lines(
        s, [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
    )
    top = quad_on_lines(
        s, [(0.0, 0.0, 1.0), (2.0, 0.0, 2.0), (2.0, 2.0, 2.0), (0.0, 2.0, 1.0)]
    )

    s.make_hexa([bottom, top])

    (solid,) = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))
    m = s.mass_properties([solid])
    np.testing.assert_allclose(m.measure, [6.0], rtol=RTOL)
    np.testing.assert_allclose(
        m.centroid[0], (10 / 9, 1.0, 7 / 9), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        s.mass_properties([top]).measure, [math.sqrt(20.0)], rtol=RTOL
    )
    table = s.entity_table(EntityKind.FACE)
    np.testing.assert_allclose(
        np.sort(table.measure),
        sorted([4.0, math.sqrt(20.0), 2.0, 4.0, 3.0, 3.0]),
        rtol=RTOL,
    )


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def built() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
        s = Session()
        s.make_hexa(six_faces_of_a_box(s))
        q = Session()
        q.make_quad(
            lines(
                q, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0)]
            )
        )
        t = Session()
        bottom = rectangle(t, (0.0, 0.0, 0.0), 2.0, 3.0)
        top = rectangle(t, (0.0, 0.0, 4.0), 2.0, 3.0)
        t.make_hexa([bottom, top])
        m = Session()
        m.add_box(1.0, 2.0, 3.0)
        m.multi_transform(
            EntityId(int(m.entities(EntityKind.SOLID)[0])),
            [(1, 0, 0), (0, 2, 0)],
            [3, 2],
        )
        return (
            s.brep() + q.brep() + t.brep() + m.brep(),
            id_space(s) + id_space(q) + id_space(t) + id_space(m),
        )

    first = built()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(built())

    assert all(other == first for other in others)


# ---- multi_transform ------------------------------------------------------------------- #


def test_a_three_by_two_array_of_blocks_is_glued_into_a_compound_of_blocks() -> None:
    s = Session()
    s.add_box(1.0, 2.0, 3.0)
    (block,) = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))

    delta = s.multi_transform(block, [(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)], [3, 2])

    solids = [EntityId(int(i)) for i in s.entities(EntityKind.SOLID)]
    assert len(solids) == 6
    np.testing.assert_allclose(s.mass_properties(solids).measure.sum(), 36.0, rtol=RTOL)
    assert len(s.entities(EntityKind.FACE, distinct=True)) == 29
    assert s.check_blocks().is_compound_of_blocks
    assert len(delta.deleted) == 0


def test_every_id_after_a_multi_transform_denotes_its_analytic_geometry() -> None:
    """The history check: falsified by tearing the array's history (final/falsify_*)."""
    s = Session()
    s.add_box(1.0, 2.0, 3.0)
    original = [EntityId(int(i)) for k in _KINDS for i in s.entities(k)]
    before = geometry(s, original)
    (block,) = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))

    delta = s.multi_transform(block, [(1.0, 0.0, 0.0)], [2])

    after = geometry(s, original)
    for i, (measure, centroid) in before.items():
        assert math.isclose(after[i][0], measure, rel_tol=RTOL, abs_tol=ATOL), i
        np.testing.assert_allclose(after[i][1], centroid, rtol=RTOL, atol=ATOL)
    created = [EntityId(int(i)) for i in delta.created]
    copies = geometry(s, created)
    for i in created:
        origin = s.origin(i)
        assert origin.role == NameRole.GENERATED
        (source,) = origin.sources
        measure, centroid = copies[int(i)]
        assert math.isclose(measure, before[int(source)][0], rel_tol=RTOL, abs_tol=ATOL)
        np.testing.assert_allclose(
            centroid, before[int(source)][1] + (1.0, 0.0, 0.0), rtol=RTOL, atol=ATOL
        )
    # The glued face at x = 1 is the original's face: the copy issues no id for it.
    (glued,) = (
        int(i)
        for i in original
        if s.entity_kind(i) == EntityKind.FACE
        and np.allclose(before[int(i)][1], (1.0, 1.0, 1.5), atol=1e-9)
    )
    assert glued in delta.modified
    assert len(s.entities(EntityKind.FACE, distinct=True)) == 11


@pytest.mark.parametrize(
    ("vectors", "counts", "match"),
    [
        ([], [], "0 translation vectors are given"),
        ([(1, 0, 0)] * 3, [2, 2, 2], "3 translation vectors are given"),
        ([(1, 0, 0)], [2, 2], "2 counts are given for 1 translation vectors"),
        ([(1, 0, 0)], [0], "the count 0 along vector 0 is below 1"),
        ([(1, 0, 0)], [1], "the counts give 1 block"),
        ([(0, 0, 0)], [2], "translation vector 0 is \\(0, 0, 0\\), shorter than 1e-07"),
        ([(math.nan, 0, 0)], [2], "vectors must be finite"),
    ],
)
def test_multi_transform_refuses_degenerate_arrays(
    vectors: list[Point], counts: list[int], match: str
) -> None:
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    block = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=match) as err:
        s.multi_transform(block, vectors, counts)

    assert "Session.multi_transform" in str(err.value)
    assert state(s) == before


# ---- refusals ------------------------------------------------------------------------- #


def test_make_quad_refuses_bad_edges() -> None:
    s = Session()
    edges = lines(
        s, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    )
    open_ = lines(
        s,
        [
            (5.0, 0.0, 0.0),
            (6.0, 0.0, 0.0),
            (6.0, 1.0, 0.0),
            (5.0, 2.0, 0.0),
            (5.0, 5.0, 0.0),
        ],
        closed=False,
    )
    box = Session()
    box.add_box(1.0, 1.0, 1.0)
    box_edges = [EntityId(int(e)) for e in box.entities(EntityKind.EDGE)][:4]
    before = state(s)

    with pytest.raises(ps.PysmeshError, match="3 edges are named"):
        s.make_quad(edges[:3])
    with pytest.raises(ps.PysmeshError, match="name one edge twice"):
        s.make_quad([edges[0], edges[0], edges[1], edges[2]])
    with pytest.raises(ps.PysmeshError, match="The edges must form a closed wire"):
        s.make_quad(open_)
    with pytest.raises(ps.PysmeshError, match="belong to a SOLID body"):
        box.make_quad(box_edges)
    assert state(s) == before


def test_make_quad_from_vertices_and_make_hexa_refuse_bad_input() -> None:
    s = Session()
    for p in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)]:
        s.add_vertex(p)
    vertices = [EntityId(int(v)) for v in s.entities(EntityKind.VERTEX)]
    f = Session()
    stack = [rectangle(f, (0.0, 0.0, float(z)), 1.0, 1.0) for z in range(6)]
    a, b, c = stack[:3]
    box = Session()
    box.add_box(1.0, 1.0, 1.0)
    box_faces = [EntityId(int(x)) for x in box.entities(EntityKind.FACE)]
    before = state(s), state(f)

    with pytest.raises(
        ps.PysmeshError, match="Four not coincident points must be given"
    ):
        s.make_quad_from_vertices(vertices)
    with pytest.raises(ps.PysmeshError, match="3 vertices are named"):
        s.make_quad_from_vertices(vertices[:3])
    with pytest.raises(ps.PysmeshError, match="3 faces are named"):
        f.make_hexa([a, b, c])
    with pytest.raises(ps.PysmeshError, match="belongs to a SOLID body"):
        box.make_hexa(box_faces)
    with pytest.raises(ps.PysmeshError, match="name one face twice"):
        f.make_hexa([a, b, c, a, b, c])
    with pytest.raises(
        ps.PysmeshError, match="Impossible to build a (connected|closed) shell"
    ):
        f.make_hexa(stack)
    assert (state(s), state(f)) == before
