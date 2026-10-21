# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for Session.close_contour: GEOM's CloseContour.

The fixtures have closed forms:

* A 3 x 7 rectangle at z = 0 whose wire leaves out its edge from ``(3, 7, 0)`` to
  ``(0, 7, 0)``: an open contour, which BRepCheck_Analyzer rejects. The gap is 3, above
  ShHealOper's maximum tolerance of 1, so ``common_vertex=False`` closes it with a new
  straight edge of length 3 and centroid ``(1.5, 7, 0)``, and the face becomes the valid
  rectangle of area 21 and centroid ``(1.5, 3.5, 0)``.
* An open polyline ``(0, 0) (3, 0) (3, 7) (0, 7) (0, g)``. For ``g = 2`` the gap is 2:
  ``common_vertex=False`` adds the edge from ``(0, 2)`` to ``(0, 0)``, of length 2 and
  centroid ``(0, 1, 0)``; ``common_vertex=True`` makes the two ends one vertex. For
  ``g = 0.5`` the gap is below 1: ShapeFix_Wire pulls the two end edges together, and
  their ends meet at the middle of the gap, ``(0, 0.25, 0)``.

Lines and planar faces integrate exactly: 1e-12 relative.
"""

from __future__ import annotations

import numpy as np
import pytest

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

# The 3 x 7 rectangle at z = 0 with the edge from (3, 7, 0) to (0, 7, 0) left out of its
# wire, which is marked open. Written from the session's own rectangle.
OPEN_RECTANGLE_BREP: bytes = b"""
CASCADE Topology V3, (c) Open Cascade
Locations 0
Curve2ds 4
1 0 0 0 1
1 0 0 1 0
1 3 0 0 1
1 0 7 1 0
Curves 4
1 0 0 0 -0 1 0
1 0 0 0 1 0 -0
1 3 0 0 -0 1 0
1 0 7 0 1 0 -0
Polygon3D 0
PolygonOnTriangulations 0
Surfaces 1
1 0 0 0 0 0 1 1 0 -0 -0 1 0
Triangulations 0

TShapes 11
Ve
1e-07
0 0 0
0 0

0101101
*
Ve
1e-07
0 7 0
0 0

0101101
*
Ed
 1e-07 1 1 0
1  1 0 0 7
2  1 1 0 0 7
0

0101000
+11 0 -10 0 *
Ve
1e-07
3 0 0
0 0

0101101
*
Ed
 1e-07 1 1 0
1  2 0 0 3
2  2 1 0 0 3
0

0101000
+11 0 -8 0 *
Ve
1e-07
3 7 0
0 0

0101101
*
Ed
 1e-07 1 1 0
1  3 0 0 7
2  3 1 0 0 7
0

0101000
+8 0 -6 0 *
Ed
 1e-07 1 1 0
1  4 0 0 3
2  4 1 0 0 3
0

0101000
+10 0 -6 0 *
Wi

0101000
-9 0 +7 0 +5 0 *
Fa
0  1e-07 1 0

0101000
+3 0 *
Co

1100000
+2 0 *

+1 0
"""


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


def open_face() -> tuple[Session, list[EntityId], EntityId]:
    """An unvalidated session holding the open rectangle: its three edges and the face."""
    s = Session(validate=False)
    s.add_brep(OPEN_RECTANGLE_BREP)
    edges = [EntityId(int(i)) for i in s.entities(EntityKind.EDGE)]
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))
    assert len(edges) == 3
    assert s.check_shape() != ()
    return s, edges, face


def open_polyline(gap: float) -> tuple[Session, list[EntityId]]:
    """The open polyline (0, 0) (3, 0) (3, 7) (0, 7) (0, gap) and its four edges."""
    s = Session()
    s.add_polyline([(0, 0, 0), (3, 0, 0), (3, 7, 0), (0, 7, 0), (0, gap, 0)])
    return s, [EntityId(int(i)) for i in s.entities(EntityKind.EDGE)]


def test_a_new_edge_closes_the_open_rectangle_and_the_face_becomes_valid() -> None:
    s, edges, face = open_face()
    right = at(s, EntityKind.EDGE, (3.0, 3.5, 0.0))
    left = at(s, EntityKind.EDGE, (0.0, 3.5, 0.0))

    delta = s.close_contour(edges)

    assert len(delta.created) == 1
    closing = EntityId(int(delta.created[0]))
    m = s.mass_properties([closing])
    assert np.isclose(m.measure[0], 3.0, rtol=RTOL)
    np.testing.assert_allclose(m.centroid[0], (1.5, 7.0, 0.0), rtol=RTOL, atol=ATOL)
    origin = s.origin(closing)
    assert origin.role == NameRole.GENERATED
    assert sorted(int(i) for i in origin.sources) == sorted([right, left])
    f = s.mass_properties([face])
    assert np.isclose(f.measure[0], 21.0, rtol=RTOL)
    np.testing.assert_allclose(f.centroid[0], (1.5, 3.5, 0.0), rtol=RTOL, atol=ATOL)
    assert delta.valid is True
    assert s.check_shape() == ()


def test_every_id_after_closing_the_rectangle_denotes_its_geometry() -> None:
    """The history check: falsified by tearing the closing's history (final/falsify_*)."""
    s, edges, face = open_face()
    before = geometry(s)

    delta = s.close_contour(edges)
    after = geometry(s)

    assert list(delta.deleted) == []
    assert sorted(before) == sorted(set(after) - {int(i) for i in delta.created})
    for i in before:
        if i == face:
            continue
        assert np.isclose(after[i][0], before[i][0], rtol=RTOL, atol=ATOL), i
        np.testing.assert_allclose(after[i][1], before[i][1], rtol=RTOL, atol=ATOL)
    assert np.isclose(after[face][0], 21.0, rtol=RTOL)
    np.testing.assert_allclose(after[face][1], (1.5, 3.5, 0.0), rtol=RTOL, atol=ATOL)
    (closing,) = (int(i) for i in delta.created)
    assert np.isclose(after[closing][0], 3.0, rtol=RTOL)
    np.testing.assert_allclose(after[closing][1], (1.5, 7.0, 0.0), rtol=RTOL, atol=ATOL)


def test_a_common_vertex_closes_the_rectangle_without_a_new_edge() -> None:
    s, edges, face = open_face()
    ends = sorted(
        [
            at(s, EntityKind.VERTEX, (3.0, 7.0, 0.0)),
            at(s, EntityKind.VERTEX, (0.0, 7.0, 0.0)),
        ]
    )

    delta = s.close_contour(edges, common_vertex=True)

    assert list(delta.created) == []
    assert sorted(int(i) for i in s.entities(EntityKind.EDGE)) == sorted(edges)
    assert [sorted(int(i) for i in g) for g in s.alias_groups(EntityKind.VERTEX)] == [
        ends
    ]
    assert float(s.tolerances([ends[0]])[0]) >= 1.5
    assert delta.valid is True
    assert s.is_alive(face)


def test_a_free_wire_with_a_gap_of_2_closes_into_one_wire_with_a_new_edge() -> None:
    s, edges = open_polyline(2.0)

    delta = s.close_contour(edges)

    assert len(delta.created) == 1
    closing = EntityId(int(delta.created[0]))
    m = s.mass_properties([closing])
    assert np.isclose(m.measure[0], 2.0, rtol=RTOL)
    np.testing.assert_allclose(m.centroid[0], (0.0, 1.0, 0.0), rtol=RTOL, atol=ATOL)
    assert sorted(int(i) for i in s.origin(closing).sources) == sorted(
        [edges[0], edges[-1]]
    )
    assert all(s.is_alive(e) for e in edges)
    # One closed wire: the five edges bound the 3 x 7 rectangle.
    s.make_face([*edges, closing])
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))
    assert np.isclose(s.mass_properties([face]).measure[0], 21.0, rtol=RTOL)


def test_a_free_wire_with_a_common_vertex_merges_its_two_ends() -> None:
    s, edges = open_polyline(2.0)
    ends = sorted(
        [
            at(s, EntityKind.VERTEX, (0.0, 0.0, 0.0)),
            at(s, EntityKind.VERTEX, (0.0, 2.0, 0.0)),
        ]
    )

    delta = s.close_contour(edges, common_vertex=True)

    assert list(delta.created) == []
    assert sorted(int(i) for i in delta.merged) == ends
    assert float(s.tolerances([ends[0]])[0]) >= 1.0


def test_a_gap_below_one_unit_is_closed_by_pulling_the_end_edges_together() -> None:
    s, edges = open_polyline(0.5)
    ends = sorted(
        [
            at(s, EntityKind.VERTEX, (0.0, 0.0, 0.0)),
            at(s, EntityKind.VERTEX, (0.0, 0.5, 0.0)),
        ]
    )

    delta = s.close_contour(edges)

    assert list(delta.created) == []
    assert list(delta.deleted) == []
    assert int(edges[0]) in delta.modified and int(edges[-1]) in delta.modified
    assert sorted(int(i) for i in delta.merged) == ends
    np.testing.assert_allclose(
        s.mass_properties([ends[0]]).centroid[0], (0.0, 0.25, 0.0), rtol=RTOL, atol=1e-9
    )


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def closed() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
        s, edges, _ = open_face()
        s.close_contour(edges)
        return s.brep(), id_space(s)

    first = closed()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(closed())

    assert all(other == first for other in others)


def test_close_contour_refuses_part_of_a_wire_a_closed_wire_and_two_bodies() -> None:
    s, edges = open_polyline(2.0)
    s.add_rectangle((10.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0)
    square = [
        EntityId(int(i)) for i in s.entities(EntityKind.EDGE) if int(i) not in edges
    ]
    before = state(s)

    with pytest.raises(ps.PysmeshError, match="are not the edges of one wire"):
        s.close_contour(edges[:2])
    with pytest.raises(ps.PysmeshError, match="is closed already"):
        s.close_contour(square)
    with pytest.raises(ps.PysmeshError, match="belong to 2 different bodies"):
        s.close_contour([edges[0], square[0]])

    assert state(s) == before


def test_close_contour_refuses_an_empty_list_a_face_and_a_dead_edge() -> None:
    s, _, face = open_face()
    s.add_line((20.0, 0.0, 0.0), (21.0, 0.0, 0.0))
    dead = EntityId(int(s.entities(EntityKind.EDGE)[-1]))
    s.remove([dead])

    with pytest.raises(
        ps.PysmeshError, match="Session.close_contour: edge_ids must name"
    ):
        s.close_contour([])
    with pytest.raises(ps.PysmeshError, match=f"entity {face} is a FACE, not an EDGE"):
        s.close_contour([face])
    with pytest.raises(ps.PysmeshError, match=f"entity {dead} is dead"):
        s.close_contour([dead])
