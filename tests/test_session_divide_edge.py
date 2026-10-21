# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for Session.divide_edge: GEOM's DivideEdge, with the edge piece rule.

The answers are closed forms:

* The edge of the 3 x 7 x 11 box from ``(0, 0, 0)`` to ``(0, 7, 0)``, divided at 0.25 and
  0.5 of its length, gives pieces of length 1.75, 1.75 and 3.5 with centroids at
  ``y = 0.875, 2.625, 5.25``, and new vertices at ``y = 1.75`` and ``y = 3.5``. The two
  faces that hold it, ``x = 0`` (area 77) and ``z = 0`` (area 21), keep their ids, areas
  and centroids.
* A circle of radius 2 divided at 0.25 and 0.75 gives arcs of length ``pi``, ``2 pi`` and
  ``pi``. The arc from angle ``a`` to ``b`` has its centroid at distance
  ``r sin(h) / h`` from the centre along the bisector, ``h = (b - a) / 2``.
* On a B-spline edge, a division at 0.5 of the length gives two pieces of equal length.

Lines, circles and planar faces integrate exactly: 1e-12 relative. The B-spline lengths
are integrated adaptively to 1e-10 by both the divider and the check, so they are
compared at 1e-8 relative.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, Session

BOX_A: float = 3.0
BOX_B: float = 7.0
BOX_C: float = 11.0
RTOL: float = 1e-12
ATOL: float = 1e-12

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, by kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def state(s: Session) -> tuple[bytes, tuple[tuple[int, ...], ...], int, int]:
    """The bytes, the live ids and the two counters."""
    return s.brep(), id_space(s), s.op_count, s.issued_id_count


def geometry(s: Session) -> dict[int, tuple[float, np.ndarray]]:
    """Every live id's measure and centroid, by the rule of entity_table."""
    out: dict[int, tuple[float, np.ndarray]] = {}
    for kind in _KINDS:
        table = s.entity_table(kind)
        for i, m, c in zip(table.ids, table.measure, table.centroid, strict=True):
            out[int(i)] = (float(m), np.array(c))
    return out


def at(s: Session, kind: EntityKind, point: tuple[float, float, float]) -> EntityId:
    """The one live entity of a kind whose centroid is the given point."""
    table = s.entity_table(kind)
    near = np.linalg.norm(table.centroid - np.asarray(point), axis=1) < 1e-9
    assert int(np.count_nonzero(near)) == 1
    return EntityId(int(table.ids[near][0]))


def box_edge() -> tuple[Session, EntityId]:
    """The 3 x 7 x 11 box and its edge from (0, 0, 0) to (0, 7, 0)."""
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    edge = at(s, EntityKind.EDGE, (0.0, 3.5, 0.0))
    np.testing.assert_allclose(
        s.curve_at(edge, [0.0]).points[0], (0.0, 0.0, 0.0), atol=ATOL
    )
    return s, edge


def created_of(s: Session, delta: ps.HistoryDelta, kind: EntityKind) -> list[EntityId]:
    """The created ids of one kind, ascending."""
    return [
        EntityId(int(i))
        for i in delta.created
        if s.entity_kind(EntityId(int(i))) == kind
    ]


@pytest.mark.parametrize("by", ["length", "parameter"])
def test_a_box_edge_of_length_7_divides_into_1_75_1_75_and_3_5(by: str) -> None:
    s, edge = box_edge()
    side = at(s, EntityKind.FACE, (0.0, 3.5, 5.5))
    bottom = at(s, EntityKind.FACE, (1.5, 3.5, 0.0))

    delta = s.divide_edge(edge, [0.25, 0.5], by=by)  # type: ignore[arg-type]

    pieces = created_of(s, delta, EntityKind.EDGE)
    vertices = created_of(s, delta, EntityKind.VERTEX)
    m = s.mass_properties(pieces)
    order = np.argsort(m.centroid[:, 1])
    np.testing.assert_allclose(m.measure[order], [1.75, 1.75, 3.5], rtol=RTOL)
    np.testing.assert_allclose(
        m.centroid[order],
        [(0, 0.875, 0), (0, 2.625, 0), (0, 5.25, 0)],
        rtol=RTOL,
        atol=ATOL,
    )
    points = s.mass_properties(vertices).centroid
    np.testing.assert_allclose(
        points[np.argsort(points[:, 1])],
        [(0, 1.75, 0), (0, 3.5, 0)],
        rtol=RTOL,
        atol=ATOL,
    )
    faces = s.mass_properties([side, bottom])
    np.testing.assert_allclose(faces.measure, [BOX_B * BOX_C, BOX_A * BOX_B], rtol=RTOL)
    np.testing.assert_allclose(
        faces.centroid, [(0.0, 3.5, 5.5), (1.5, 3.5, 0.0)], rtol=RTOL, atol=ATOL
    )
    assert int(side) in delta.modified and int(bottom) in delta.modified
    assert delta.valid is True


def test_every_id_after_a_division_denotes_its_analytic_geometry() -> None:
    """The history check: falsified by tearing the division's history (final/falsify_*)."""
    s, edge = box_edge()
    before = geometry(s)

    delta = s.divide_edge(edge, [0.25, 0.5])
    after = geometry(s)

    assert list(delta.deleted) == [edge]
    assert not s.is_alive(edge)
    survivors = set(before) - {int(edge)}
    assert survivors <= set(after)
    for i in survivors:
        assert np.isclose(after[i][0], before[i][0], rtol=RTOL, atol=ATOL), i
        np.testing.assert_allclose(after[i][1], before[i][1], rtol=RTOL, atol=ATOL)
    expected = {
        (1.75, 0.875): EntityKind.EDGE,
        (1.75, 2.625): EntityKind.EDGE,
        (3.5, 5.25): EntityKind.EDGE,
        (0.0, 1.75): EntityKind.VERTEX,
        (0.0, 3.5): EntityKind.VERTEX,
    }
    assert sorted(set(after) - survivors) == sorted(int(i) for i in delta.created)
    found = {}
    for i in delta.created:
        measure, centroid = after[int(i)]
        np.testing.assert_allclose(centroid[[0, 2]], (0.0, 0.0), atol=ATOL)
        found[(round(measure, 9), round(float(centroid[1]), 9))] = s.entity_kind(
            EntityId(int(i))
        )
        origin = s.origin(EntityId(int(i)))
        assert origin.role == NameRole.GENERATED
        assert list(origin.sources) == [edge]
    assert found == expected


def test_a_circle_divides_into_arcs_of_pi_2pi_and_pi() -> None:
    s = Session()
    s.add_cylinder(2.0, 5.0)
    rim = at(s, EntityKind.EDGE, (0.0, 0.0, 0.0))
    r = 2.0

    delta = s.divide_edge(rim, [0.25, 0.75])

    arcs = created_of(s, delta, EntityKind.EDGE)
    m = s.mass_properties(arcs)
    order = np.argsort(m.measure)
    np.testing.assert_allclose(
        np.sort(m.measure), [math.pi, math.pi, 2 * math.pi], rtol=RTOL
    )
    quarter = r * math.sin(math.pi / 4) / (math.pi / 4) / math.sqrt(2.0)
    half = r * math.sin(math.pi / 2) / (math.pi / 2)
    centroids = {tuple(np.round(c, 9)) for c in m.centroid[order]}
    assert centroids == {
        (round(quarter, 9), round(quarter, 9), 0.0),
        (round(quarter, 9), round(-quarter, 9), 0.0),
        (round(-half, 9), 0.0, 0.0),
    }
    assert delta.valid is True


def test_by_length_and_by_parameter_differ_on_a_spline_and_length_halves_it() -> None:
    def spline() -> tuple[Session, EntityId]:
        s = Session()
        s.add_spline(
            [(0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (4.0, 2.5, 0.0), (9.0, 0.0, 0.0)]
        )
        return s, EntityId(int(s.entities(EntityKind.EDGE)[0]))

    s, edge = spline()
    total = float(s.mass_properties([edge], precision=1e-12).measure[0])
    p, pedge = spline()

    by_length = s.divide_edge(edge, [0.5], by="length")
    by_parameter = p.divide_edge(pedge, [0.5], by="parameter")

    halves = s.mass_properties(
        created_of(s, by_length, EntityKind.EDGE), precision=1e-12
    )
    np.testing.assert_allclose(halves.measure, [total / 2, total / 2], rtol=1e-8)
    other = p.mass_properties(
        created_of(p, by_parameter, EntityKind.EDGE), precision=1e-12
    )
    assert abs(other.measure[0] - other.measure[1]) > 1e-3 * total


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def divided() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
        s, edge = box_edge()
        s.divide_edge(edge, [0.25, 0.5])
        return s.brep(), id_space(s)

    first = divided()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(divided())

    assert all(other == first for other in others)


@pytest.mark.parametrize(
    ("positions", "by", "match"),
    [
        ([], "length", "no position is given for edge"),
        ([0.0], "length", "position 0 for edge .* is not strictly between 0 and 1"),
        ([1.0], "length", "position 1 for edge .* is not strictly between 0 and 1"),
        ([-0.5], "length", "position -0.5 for edge"),
        ([0.5, 0.25], "length", "not strictly increasing \\(0.5 then 0.25\\)"),
        ([0.5, 0.5], "length", "not strictly increasing \\(0.5 then 0.5\\)"),
        ([math.nan], "length", "positions must be finite"),
        ([0.5], "arc", "by='arc' is not one of length, parameter"),
    ],
)
def test_divide_edge_refuses_degenerate_positions(
    positions: list[float], by: str, match: str
) -> None:
    s, edge = box_edge()
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=match) as err:
        s.divide_edge(edge, positions, by=by)  # type: ignore[arg-type]

    assert "Session.divide_edge" in str(err.value)
    assert state(s) == before


def test_divide_edge_refuses_a_face_a_dead_edge_and_a_degenerate_edge() -> None:
    s, edge = box_edge()
    face = at(s, EntityKind.FACE, (0.0, 3.5, 5.5))
    s.divide_edge(edge, [0.5])
    sphere = Session()
    sphere.add_sphere(2.0)
    table = sphere.entity_table(EntityKind.EDGE)
    pole = EntityId(int(table.ids[np.argmin(table.measure)]))
    assert float(np.min(table.measure)) == 0.0

    with pytest.raises(ps.PysmeshError, match=f"entity {face} is a FACE, not an EDGE"):
        s.divide_edge(face, [0.5])
    with pytest.raises(ps.PysmeshError, match=f"entity {edge} is dead"):
        s.divide_edge(edge, [0.5])
    with pytest.raises(ps.PysmeshError, match=f"edge {pole} is degenerate"):
        sphere.divide_edge(pole, [0.5])
