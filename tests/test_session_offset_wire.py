# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for offset_wire.

A positive distance moves the wire away from the region it bounds; a negative one moves
it into that region. An open wire bounds the region it encloses with the chord from its
last vertex to its first.

* Unit square, +r with arc joins: four sides at distance r and four quarter arcs of radius
  r about the corners. Length ``4 + 2 pi r``. A quarter arc's centroid lies
  ``2 sqrt(2) r / pi`` from its centre along the bisector: ``2 r / pi`` along each axis.
* Unit square, +r with intersection joins: the square of side ``1 + 2 r``; -r: the square
  of side ``1 - 2 r``.
* Circle of radius R, +-d: the circle of radius ``R +- d``, length ``2 pi (R +- d)``.
* The open L from (0, 0) to (2, 0) to (2, 1): with its chord it bounds a triangle, so +r
  goes below y = 0 and right of x = 2: the side ``y = -r`` (length 2), the arc of radius r
  about (2, 0) (length ``pi r / 2``) and the side ``x = 2 + r`` (length 1). -r goes into
  the triangle: ``y = r`` from x = 0 to ``2 - r``, then ``x = 2 - r`` up to y = 1.

All edges are lines and circle arcs, measured exactly: 1e-12 relative.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, Session

R: float = 0.25
RTOL: float = 1e-12
ATOL: float = 1e-12

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)

SQUARE: list[list[float]] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
]
ELL: list[list[float]] = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, per kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def new_ids(s: Session, kind: EntityKind, before: set[int]) -> list[EntityId]:
    """The ids of ``kind`` that are alive now and were not in ``before``."""
    return [EntityId(int(i)) for i in s.entities(kind) if int(i) not in before]


def polyline_session(
    points: list[list[float]], *, closed: bool = False
) -> tuple[Session, list[EntityId]]:
    """A free polyline; its edge ids, in the order of the points."""
    s = Session()
    s.add_polyline(np.array(points, dtype=np.float64), closed=closed)
    table = s.entity_table(EntityKind.EDGE)
    starts = np.asarray(points, dtype=np.float64)
    ends = np.roll(starts, -1, axis=0)
    mids = 0.5 * (starts + ends)[: len(table.ids)]
    order = [
        int(table.ids[np.argmin(np.linalg.norm(table.centroid - m, axis=1))])
        for m in mids
    ]
    return s, [EntityId(i) for i in order]


def offset(
    s: Session, edges: list[EntityId], distance: float, join: str = "arc"
) -> tuple[ps.HistoryDelta, list[tuple[float, NDArray[np.float64], int]]]:
    """Run offset_wire; the delta and (length, centroid, id) of each new edge."""
    before = {int(i) for i in s.entities(EntityKind.EDGE)}
    delta = s.offset_wire(edges, distance, join=join)  # type: ignore[arg-type]
    fresh = new_ids(s, EntityKind.EDGE, before)
    props = s.mass_properties(fresh)
    rows = [
        (float(props.measure[k]), props.centroid[k], int(fresh[k]))
        for k in range(len(fresh))
    ]
    return delta, rows


def matches(
    rows: list[tuple[float, NDArray[np.float64], int]],
    expected: list[tuple[float, tuple[float, float, float]]],
) -> None:
    """Each expected (length, centroid) is one new edge, and there are no others."""
    assert len(rows) == len(expected)
    remaining = list(rows)
    for length, centroid in expected:
        hit = [
            r for r in remaining if np.allclose(r[1], centroid, rtol=0.0, atol=1e-12)
        ]
        assert len(hit) == 1, (length, centroid)
        assert hit[0][0] == pytest.approx(length, rel=RTOL)
        remaining.remove(hit[0])


def square_arc_rows(r: float) -> list[tuple[float, tuple[float, float, float]]]:
    """The four sides and four quarter arcs of the unit square offset by +r."""
    k = 2.0 * r / math.pi
    sides = [
        (1.0, (0.5, -r, 0.0)),
        (1.0, (1.0 + r, 0.5, 0.0)),
        (1.0, (0.5, 1.0 + r, 0.0)),
        (1.0, (-r, 0.5, 0.0)),
    ]
    arcs = [
        (0.5 * math.pi * r, (1.0 + k, -k, 0.0)),
        (0.5 * math.pi * r, (1.0 + k, 1.0 + k, 0.0)),
        (0.5 * math.pi * r, (-k, 1.0 + k, 0.0)),
        (0.5 * math.pi * r, (-k, -k, 0.0)),
    ]
    return sides + arcs


# ---- analytic ---------------------------------------------------------------------- #


def test_a_square_grows_by_sides_and_quarter_arcs_with_arc_joins() -> None:
    s, edges = polyline_session(SQUARE, closed=True)

    _, rows = offset(s, edges, R)

    matches(rows, square_arc_rows(R))
    assert sum(r[0] for r in rows) == pytest.approx(4.0 + 2.0 * math.pi * R, rel=RTOL)


def test_a_square_grows_to_a_larger_square_with_intersection_joins() -> None:
    s, edges = polyline_session(SQUARE, closed=True)

    _, rows = offset(s, edges, R, "intersection")

    side = 1.0 + 2.0 * R
    matches(
        rows,
        [
            (side, (0.5, -R, 0.0)),
            (side, (1.0 + R, 0.5, 0.0)),
            (side, (0.5, 1.0 + R, 0.0)),
            (side, (-R, 0.5, 0.0)),
        ],
    )


@pytest.mark.parametrize("join", ["arc", "intersection"])
def test_a_negative_distance_shrinks_a_square(join: str) -> None:
    s, edges = polyline_session(SQUARE, closed=True)

    _, rows = offset(s, edges, -R, join)

    side = 1.0 - 2.0 * R
    matches(
        rows,
        [
            (side, (0.5, R, 0.0)),
            (side, (1.0 - R, 0.5, 0.0)),
            (side, (0.5, 1.0 - R, 0.0)),
            (side, (R, 0.5, 0.0)),
        ],
    )


@pytest.mark.parametrize("distance", [1.0, -1.0])
def test_a_circle_offsets_to_the_concentric_circle(distance: float) -> None:
    s = Session()
    s.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0)
    circle = EntityId(int(s.entities(EntityKind.EDGE)[0]))

    _, rows = offset(s, [circle], distance)

    matches(rows, [(2.0 * math.pi * (2.0 + distance), (0.0, 0.0, 0.0))])


@pytest.mark.parametrize("points", [ELL, ELL[::-1]], ids=["forward", "reversed"])
def test_an_open_wire_moves_away_from_the_region_its_chord_closes(
    points: list[list[float]],
) -> None:
    """The side does not depend on the direction the wire runs in."""
    s, edges = polyline_session(points)

    _, rows = offset(s, edges, R)

    k = 2.0 * R / math.pi
    matches(
        rows,
        [
            (2.0, (1.0, -R, 0.0)),
            (0.5 * math.pi * R, (2.0 + k, -k, 0.0)),
            (1.0, (2.0 + R, 0.5, 0.0)),
        ],
    )


def test_an_open_wire_moves_into_the_region_its_chord_closes_when_negative() -> None:
    s, edges = polyline_session(ELL)

    _, rows = offset(s, edges, -R)

    matches(
        rows,
        [
            (2.0 - R, (0.5 * (2.0 - R), R, 0.0)),
            (1.0 - R, (2.0 - R, 0.5 * (1.0 + R), 0.0)),
        ],
    )


def test_an_open_wire_with_intersection_joins_meets_at_the_corner() -> None:
    s, edges = polyline_session(ELL)

    _, rows = offset(s, edges, R, "intersection")

    matches(
        rows,
        [
            (2.0 + R, (0.5 * (2.0 + R), -R, 0.0)),
            (1.0 + R, (2.0 + R, 0.5 * (1.0 - R), 0.0)),
        ],
    )


# ---- history against geometry -------------------------------------------------------- #


def test_the_input_stays_and_each_new_edge_is_generated_from_its_source() -> None:
    """A side from the input edge it runs along; an arc from the corner vertex it rounds."""
    s, edges = polyline_session(SQUARE, closed=True)
    before = id_space(s)
    tables = {k: s.entity_table(k) for k in _KINDS}
    corners = s.entity_table(EntityKind.VERTEX)
    sides = s.entity_table(EntityKind.EDGE)

    delta, rows = offset(s, edges, R)

    assert delta.deleted.size == 0
    assert delta.modified.size == 0
    for kind, table in tables.items():
        after = s.entity_table(kind)
        keep = np.isin(after.ids, table.ids)
        np.testing.assert_array_equal(after.ids[keep], table.ids)
        np.testing.assert_array_equal(after.measure[keep], table.measure)
        np.testing.assert_array_equal(after.centroid[keep], table.centroid)
    old = {i for ids in before for i in ids}
    new = {i for ids in id_space(s) for i in ids} - old
    assert sorted(delta.created.tolist()) == sorted(new)
    k = 2.0 * R / math.pi
    for length, centroid, edge in rows:
        assert s.name_of(EntityId(edge)).role is NameRole.GENERATED
        (source,) = s.origin(EntityId(edge)).sources.tolist()
        if length == pytest.approx(1.0, rel=RTOL):
            # The side at distance R along the outward normal of its source edge.
            row = int(np.flatnonzero(sides.ids == source)[0])
            moved = sides.centroid[row]
            assert np.linalg.norm(centroid - moved) == pytest.approx(R, rel=RTOL)
        else:
            row = int(np.flatnonzero(corners.ids == source)[0])
            corner = corners.centroid[row]
            assert np.linalg.norm(centroid - corner) == pytest.approx(
                k * math.sqrt(2.0), rel=RTOL
            )


# ---- determinism --------------------------------------------------------------------- #


def _offset_square() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
    s, edges = polyline_session(SQUARE, closed=True)
    s.offset_wire(edges, R)
    return s.brep(), id_space(s)


def test_two_fresh_sessions_give_identical_bytes_and_ids() -> None:
    first = _offset_square()

    second = _offset_square()

    assert first == second


# ---- refusals ------------------------------------------------------------------------ #


def assert_refused(s: Session, pattern: str, *args: object, **kwargs: object) -> None:
    """The call raises a PysmeshError matching ``pattern`` and leaves the session as is."""
    before = (s.brep(), id_space(s), s.op_count, s.issued_id_count)

    with pytest.raises(ps.PysmeshError, match=pattern):
        s.offset_wire(*args, **kwargs)  # type: ignore[arg-type]

    assert (s.brep(), id_space(s), s.op_count, s.issued_id_count) == before


def named(edges: list[EntityId]) -> str:
    """The edge ids as the messages list them."""
    return ", ".join(str(int(e)) for e in edges)


def test_a_straight_open_wire_is_refused_naming_its_edges() -> None:
    s, edges = polyline_session([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    assert_refused(
        s,
        rf"offset_wire: the edges \({named(edges)}\) lie on one straight line",
        edges,
        R,
    )


def test_an_open_wire_that_encloses_no_area_with_its_chord_is_refused() -> None:
    s, edges = polyline_session(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
    )

    assert_refused(
        s,
        rf"offset_wire: the open wire of edges \({named(edges)}\) and the chord from its "
        "last vertex to its first enclose no area",
        edges,
        R,
    )


def test_a_wire_that_is_not_planar_is_refused_naming_its_edges() -> None:
    s, edges = polyline_session(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    )

    assert_refused(
        s,
        rf"offset_wire: the edges \({named(edges)}\) do not lie in one plane",
        edges,
        R,
    )


def test_edges_that_do_not_connect_are_refused() -> None:
    s = Session()
    s.add_line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    s.add_line((0.0, 5.0, 0.0), (1.0, 5.0, 0.0))
    edges = [EntityId(int(i)) for i in s.entities(EntityKind.EDGE)]

    assert_refused(
        s,
        rf"offset_wire: the edges \({named(edges)}\) do not form one connected wire",
        edges,
        R,
    )


def test_an_inward_offset_past_the_middle_is_refused() -> None:
    s, edges = polyline_session(SQUARE, closed=True)

    assert_refused(
        s,
        rf"offset_wire: OCCT built no offset of the wire of edges \({named(edges)}\) at "
        r"distance -0.6",
        edges,
        -0.6,
    )


@pytest.mark.parametrize("distance", [0.0, math.nan, math.inf])
def test_a_zero_or_non_finite_distance_is_refused(distance: float) -> None:
    s, edges = polyline_session(SQUARE, closed=True)

    assert_refused(s, r"offset_wire.*distance", edges, distance)


def test_an_unknown_join_is_refused_with_its_value() -> None:
    s, edges = polyline_session(SQUARE, closed=True)

    assert_refused(
        s,
        r"offset_wire: the join is 'round'; it is 'arc' or 'intersection'",
        edges,
        R,
        join="round",
    )


def test_an_empty_edge_list_is_refused() -> None:
    s, _ = polyline_session(SQUARE, closed=True)

    assert_refused(s, r"offset_wire: edge_ids must name at least one edge", [], R)


def test_a_face_is_refused_naming_its_kind() -> None:
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0)
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    assert_refused(
        s, rf"offset_wire: entity {int(face)} is a FACE, not an EDGE", [face], R
    )
