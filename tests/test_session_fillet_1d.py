# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for Session.fillet_1d: GEOM's MakeFillet1D on a planar wire.

The answers are the closed form of a fillet between two lines that meet at the interior
angle ``theta``: the arc's centre lies on the bisector at ``r / sin(theta / 2)`` from the
corner, its tangent points lie at ``r / tan(theta / 2)`` from the corner along each line,
and the arc spans ``pi - theta``, so its length is ``r (pi - theta)``. The centroid of an
arc of half-angle ``a`` lies at ``r sin(a) / a`` from its centre, on its bisector.

Tolerances. The copied algorithm finds a tangent point by a Newton search that stops when
``|sqrt(|d^2 - r^2| + r^2) - r| < 1e-8`` (GEOMImpl_Fillet1dPoint::HasSolution), so points
and lengths that depend on a tangent point are compared at 1e-8 absolute. The arc is built
on an exact circle through the two tangent points, so its radius and its length (radius
times the exact angle between two perpendiculars) are compared at 1e-12 relative.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, Session

SEARCH_ATOL: float = 1e-8
RTOL: float = 1e-12

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)

L_POINTS: list[tuple[float, float, float]] = [(0, 0, 0), (10, 0, 0), (10, 10, 0)]
RECT_POINTS: list[tuple[float, float, float]] = [
    (0, 0, 0),
    (6, 0, 0),
    (6, 4, 0),
    (0, 4, 0),
]
SPLIT_POINTS: list[tuple[float, float, float]] = [
    (0, 0, 0),
    (4, 0, 0),
    (5, 0, 0),
    (5, 5, 0),
]


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, by kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def state(s: Session) -> tuple[bytes, tuple[tuple[int, ...], ...], int, int]:
    """The bytes, the live ids and the two counters."""
    return s.brep(), id_space(s), s.op_count, s.issued_id_count


def vertex_at(s: Session, point: tuple[float, float, float]) -> EntityId:
    """The one live vertex at the given point."""
    table = s.entity_table(EntityKind.VERTEX)
    near = np.linalg.norm(table.centroid - np.asarray(point), axis=1) < 1e-9
    assert int(np.count_nonzero(near)) == 1
    return EntityId(int(table.ids[near][0]))


def polyline(
    points: list[tuple[float, float, float]], *, closed: bool = False
) -> tuple[Session, list[EntityId]]:
    """A session holding one polyline wire, and its edge ids."""
    s = Session()
    s.add_polyline(points, closed=closed)
    return s, [EntityId(int(i)) for i in s.entities(EntityKind.EDGE)]


def arcs_of(s: Session) -> list[EntityId]:
    """The live circle edges, ascending."""
    types = s.entity_types(EntityKind.EDGE)
    return [EntityId(int(i)) for i, t in zip(types.ids, types.types) if t == "Circle"]


def arc_centroid(
    centre: NDArray[np.float64], towards: NDArray[np.float64], r: float, half: float
) -> NDArray[np.float64]:
    """The centroid of an arc of half-angle `half` whose bisector points to `towards`."""
    direction = (towards - centre) / np.linalg.norm(towards - centre)
    return np.asarray(
        centre + direction * (r * math.sin(half) / half), dtype=np.float64
    )


@pytest.mark.parametrize("theta_deg", [60.0, 90.0, 135.0])
def test_a_corner_between_two_lines_takes_the_closed_form_arc(theta_deg: float) -> None:
    theta = math.radians(theta_deg)
    r = 1.5
    far = (10 * math.cos(theta), 10 * math.sin(theta), 0.0)
    s, edges = polyline([far, (0, 0, 0), (10, 0, 0)])
    corner = vertex_at(s, (0.0, 0.0, 0.0))

    delta = s.fillet_1d(edges, [corner], r)

    (arc,) = arcs_of(s)
    geometry = s.curve_geometry([arc])
    bisector = np.array([math.cos(theta / 2), math.sin(theta / 2), 0.0])
    np.testing.assert_allclose(
        geometry.origin[0], bisector * r / math.sin(theta / 2), atol=SEARCH_ATOL
    )
    np.testing.assert_allclose(geometry.radius[0], [r, r], rtol=RTOL)
    length = s.mass_properties([arc], precision=1e-14).measure[0]
    np.testing.assert_allclose(length, r * (math.pi - theta), rtol=RTOL)
    tangent = r / math.tan(theta / 2)
    np.testing.assert_allclose(
        s.mass_properties(edges, precision=1e-14).measure,
        [10 - tangent, 10 - tangent],
        atol=SEARCH_ATOL,
    )
    assert delta.valid is True


def test_every_id_after_a_fillet_denotes_its_analytic_geometry() -> None:
    """The history check: falsified by tearing the fillet's history (final/falsify_*)."""
    s, edges = polyline(L_POINTS)
    corner = vertex_at(s, (10.0, 0.0, 0.0))
    start = vertex_at(s, (0.0, 0.0, 0.0))
    end = vertex_at(s, (10.0, 10.0, 0.0))

    delta = s.fillet_1d(edges, [corner], 2.0)

    assert list(delta.deleted) == [corner]
    assert sorted(int(i) for i in delta.modified) == sorted(
        [int(edges[0]), int(edges[1]), int(start), int(end)]
    )
    m = s.mass_properties(edges, precision=1e-14)
    np.testing.assert_allclose(m.measure, [8.0, 8.0], atol=SEARCH_ATOL)
    np.testing.assert_allclose(m.centroid, [(4, 0, 0), (10, 6, 0)], atol=SEARCH_ATOL)
    ends = s.mass_properties([start, end]).centroid
    np.testing.assert_allclose(ends, [(0, 0, 0), (10, 10, 0)], atol=1e-12)
    (arc,) = arcs_of(s)
    created_vertices = [
        EntityId(int(i))
        for i in delta.created
        if s.entity_kind(EntityId(int(i))) == "VERTEX"
    ]
    assert sorted(int(i) for i in delta.created) == sorted(
        [int(arc)] + [int(v) for v in created_vertices]
    )
    for i in delta.created:
        origin = s.origin(EntityId(int(i)))
        assert origin.role == NameRole.GENERATED
        assert list(origin.sources) == [corner]
    arc_m = s.mass_properties([arc], precision=1e-14)
    np.testing.assert_allclose(arc_m.measure, [math.pi], rtol=RTOL)
    np.testing.assert_allclose(
        arc_m.centroid[0],
        arc_centroid(
            np.array([8.0, 2.0, 0.0]), np.array([10.0, 0.0, 0.0]), 2.0, math.pi / 4
        ),
        atol=SEARCH_ATOL,
    )
    points = s.mass_properties(created_vertices).centroid
    np.testing.assert_allclose(
        points[np.argsort(points[:, 0])], [(8, 0, 0), (10, 2, 0)], atol=SEARCH_ATOL
    )
    assert sorted(int(i) for i in s.entities(EntityKind.EDGE)) == sorted(
        [int(edges[0]), int(edges[1]), int(arc)]
    )


def test_every_corner_of_a_closed_rectangle_rounds_and_its_edges_keep_their_ids() -> (
    None
):
    s, edges = polyline(RECT_POINTS, closed=True)
    corners = [vertex_at(s, p) for p in RECT_POINTS]

    delta = s.fillet_1d(edges, [], 1.0)

    assert sorted(int(i) for i in delta.deleted) == sorted(int(c) for c in corners)
    assert sorted(int(i) for i in delta.modified) == sorted(int(e) for e in edges)
    lengths = s.mass_properties(edges, precision=1e-14).measure
    np.testing.assert_allclose(lengths, [4.0, 2.0, 4.0, 2.0], atol=2 * SEARCH_ATOL)
    arcs = arcs_of(s)
    assert len(arcs) == 4
    centres = {}
    for arc in arcs:
        origin = s.origin(arc)
        assert origin.role == NameRole.GENERATED
        (source,) = origin.sources
        centres[corners.index(EntityId(int(source)))] = s.curve_geometry([arc]).origin[
            0
        ]
        np.testing.assert_allclose(
            s.mass_properties([arc], precision=1e-14).measure, [math.pi / 2], rtol=RTOL
        )
    expected = [(1, 1, 0), (5, 1, 0), (5, 3, 0), (1, 3, 0)]
    for k in range(4):
        np.testing.assert_allclose(centres[k], expected[k], atol=SEARCH_ATOL)
    assert s.entity_count == 4 + 4 + 8
    assert delta.valid is True


def test_ignored_secant_vertices_join_collinear_edges_for_a_larger_radius() -> None:
    s, edges = polyline(SPLIT_POINTS)
    secant = vertex_at(s, (4.0, 0.0, 0.0))
    corner = vertex_at(s, (5.0, 0.0, 0.0))

    delta = s.fillet_1d(edges, [corner], 2.0)

    assert sorted(int(i) for i in delta.deleted) == sorted([int(secant), int(corner)])
    assert sorted(int(i) for i in delta.merged) == sorted(
        [int(edges[0]), int(edges[1])]
    )
    joined = s.mass_properties(edges[:2], precision=1e-14)
    np.testing.assert_allclose(joined.measure, [3.0, 3.0], atol=SEARCH_ATOL)
    np.testing.assert_allclose(joined.centroid, [(1.5, 0, 0)] * 2, atol=SEARCH_ATOL)
    (arc,) = arcs_of(s)
    np.testing.assert_allclose(
        s.curve_geometry([arc]).origin[0], (3, 2, 0), atol=SEARCH_ATOL
    )
    assert list(s.origin(arc).sources) == [corner]
    leg = s.mass_properties([edges[2]], precision=1e-14)
    np.testing.assert_allclose(leg.measure, [3.0], atol=SEARCH_ATOL)
    np.testing.assert_allclose(leg.centroid, [(5, 3.5, 0)], atol=SEARCH_ATOL)


def test_without_ignoring_secant_vertices_the_short_edge_limits_the_radius() -> None:
    s, edges = polyline(SPLIT_POINTS)
    corner = vertex_at(s, (5.0, 0.0, 0.0))
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=f"at vertex {corner} of the wire") as err:
        s.fillet_1d(edges, [corner], 2.0, ignore_secant_vertices=False)

    assert "no arc of that radius fits" in str(err.value)
    assert state(s) == before


def test_a_retained_snapshot_keeps_its_bytes_across_a_fillet() -> None:
    s, edges = polyline(RECT_POINTS, closed=True)
    before = s.brep()
    mark = s.snapshot()

    s.fillet_1d(edges, [], 1.0)
    s.restore(mark)

    assert s.brep() == before


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def filleted() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
        s, edges = polyline(RECT_POINTS, closed=True)
        s.fillet_1d(edges, [], 1.0)
        split, split_edges = polyline(SPLIT_POINTS)
        split.fillet_1d(split_edges, [], 2.0)
        return s.brep() + split.brep(), id_space(s) + id_space(split)

    first = filleted()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(filleted())

    assert all(other == first for other in others)


@pytest.mark.parametrize(
    ("radius", "match"),
    [
        (0.0, "the radius 0 of the fillets on the wire of edges 1, 2 is below"),
        (-1.0, "the radius -1 of the fillets"),
        (math.nan, r"radius must be a finite number \(got nan\)"),
        (6.0, "at vertex 4 of the wire of edges 1, 2 no fillet of radius 6"),
    ],
)
def test_fillet_1d_refuses_a_degenerate_radius(radius: float, match: str) -> None:
    s, edges = polyline([(0, 0, 0), (5, 0, 0), (5, 5, 0)])
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=match) as err:
        s.fillet_1d(edges, [], radius)

    assert "Session.fillet_1d" in str(err.value)
    assert state(s) == before


def test_fillet_1d_refuses_vertices_that_are_not_corners() -> None:
    s, edges = polyline([(0, 0, 0), (5, 0, 0), (9, 0, 0), (9, 5, 0)])
    end = vertex_at(s, (0.0, 0.0, 0.0))
    secant = vertex_at(s, (5.0, 0.0, 0.0))
    corner = vertex_at(s, (9.0, 0.0, 0.0))
    other = Session()
    other.add_polyline([(0, 0, 0), (1, 0, 0)])
    box = Session()
    box.add_box(1.0, 2.0, 3.0)
    box_edges = [EntityId(int(i)) for i in box.entities(EntityKind.EDGE)]
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=f"vertex {end} .* it ends the open wire"):
        s.fillet_1d(edges, [end], 1.0)
    with pytest.raises(ps.PysmeshError, match=f"vertex {secant} .* lie on one line"):
        s.fillet_1d(edges, [secant], 1.0)
    with pytest.raises(ps.PysmeshError, match=f"vertex {corner} .* is named twice"):
        s.fillet_1d(edges, [corner, corner], 1.0)
    with pytest.raises(
        ps.PysmeshError, match=f"entity {edges[0]} is a EDGE, not a VERTEX"
    ):
        s.fillet_1d(edges, [edges[0]], 1.0)
    with pytest.raises(ps.PysmeshError, match="are 2 of the 3 edges of their wire"):
        s.fillet_1d(edges[:2], [corner], 1.0)
    with pytest.raises(ps.PysmeshError, match="belong to a SOLID body"):
        box.fillet_1d(box_edges, [], 1.0)
    assert state(s) == before


def test_fillet_1d_refuses_a_straight_wire_and_a_non_planar_wire() -> None:
    straight, straight_edges = polyline([(0, 0, 0), (1, 0, 0), (3, 0, 0)])
    bent, bent_edges = polyline([(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)])

    with pytest.raises(ps.PysmeshError, match="has no corner"):
        straight.fillet_1d(straight_edges, [], 0.1)
    with pytest.raises(ps.PysmeshError, match="is not planar"):
        bent.fillet_1d(bent_edges, [], 0.1)
