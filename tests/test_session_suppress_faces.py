# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for Session.suppress_faces: GEOM's SuppressFaces.

The answers are closed forms on the 3 x 7 x 11 box and on ``open_box_shell.brep`` (the cube
``[0, 2]^3`` without its face ``x = 0``):

* Suppressing the top face of the box leaves its five other faces, unchanged, as an open
  shell: total area ``2 (21 + 33 + 77) - 21 = 241``, four free edges of total length
  ``2 (3 + 7) = 20`` at ``z = 11``. No solid is left, so the box's solid id dies.
* Suppressing one more face of the open cube leaves a shell of four faces of area 4 each.
* The comparison with defeature (brief section 5, P3.4): a plate ``10 x 6 x 1`` with a
  through hole of radius 1. defeature closes it into the plain plate of volume 60;
  suppress_faces leaves the plate's six faces as an open shell, the hole in place.

Every remaining entity keeps its geometry exactly: planar faces and lines at 1e-12.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, Session

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

FIXTURES: Path = Path(__file__).resolve().parent / "fixtures"


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


def free_edges(s: Session) -> list[EntityId]:
    """The live edges held by one face."""
    pairs = s.adjacency(EntityKind.EDGE, EntityKind.FACE)
    ids, counts = np.unique(pairs.ids, return_counts=True)
    return [EntityId(int(i)) for i, c in zip(ids, counts, strict=True) if c == 1]


def box() -> tuple[Session, EntityId]:
    """The 3 x 7 x 11 box and its top face."""
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    return s, at(s, EntityKind.FACE, (1.5, 3.5, BOX_C))


def test_suppressing_the_top_face_leaves_the_box_as_an_open_shell() -> None:
    s, top = box()
    solid = EntityId(int(s.entities(EntityKind.SOLID)[0]))

    delta = s.suppress_faces([top])

    assert not s.is_alive(top)
    assert not s.is_alive(solid)
    assert int(solid) in delta.deleted
    assert len(s.entities(EntityKind.SOLID)) == 0
    faces = [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    assert len(faces) == 5
    area = float(np.sum(s.mass_properties(faces).measure))
    assert np.isclose(area, 2 * (21.0 + 33.0 + 77.0) - 21.0, rtol=RTOL)
    rim = free_edges(s)
    rim_lengths = s.mass_properties(rim)
    assert np.isclose(float(np.sum(rim_lengths.measure)), 20.0, rtol=RTOL)
    np.testing.assert_allclose(rim_lengths.centroid[:, 2], BOX_C, rtol=RTOL)
    assert not bool(s.is_closed_shell([faces[0]])[0])
    assert delta.valid is True


def test_every_id_after_suppressing_a_face_denotes_its_geometry() -> None:
    """The history check: falsified by tearing the removal's history (final/falsify_*)."""
    s, top = box()
    before = geometry(s)
    solid = int(s.entities(EntityKind.SOLID)[0])

    delta = s.suppress_faces([top])
    after = geometry(s)

    assert list(delta.created) == []
    assert sorted(int(i) for i in delta.deleted) == sorted([solid, int(top)])
    assert sorted(after) == sorted(set(before) - {solid, int(top)})
    for i, (measure, centroid) in after.items():
        assert np.isclose(measure, before[i][0], rtol=RTOL, atol=ATOL), i
        np.testing.assert_allclose(centroid, before[i][1], rtol=RTOL, atol=ATOL)


def test_suppressing_a_face_of_an_open_shell_leaves_a_shell_of_four_faces() -> None:
    s = Session()
    s.add_brep((FIXTURES / "open_box_shell.brep").read_bytes())
    face = at(s, EntityKind.FACE, (2.0, 1.0, 1.0))

    s.suppress_faces([face])

    faces = [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    np.testing.assert_allclose(s.mass_properties(faces).measure, [4.0] * 4, rtol=RTOL)


def test_suppressing_all_but_one_face_of_a_shell_leaves_that_face() -> None:
    s = Session()
    s.add_brep((FIXTURES / "open_box_shell.brep").read_bytes())
    keep = at(s, EntityKind.FACE, (2.0, 1.0, 1.0))
    others = [EntityId(int(i)) for i in s.entities(EntityKind.FACE) if int(i) != keep]

    s.suppress_faces(others)

    assert [int(i) for i in s.entities(EntityKind.FACE)] == [keep]
    assert np.isclose(s.mass_properties([keep]).measure[0], 4.0, rtol=RTOL)
    assert len(free_edges(s)) == 4


def test_defeature_closes_a_plate_hole_where_suppress_faces_leaves_it_open() -> None:
    def plate() -> tuple[Session, list[EntityId]]:
        s = Session()
        s.add_box(10.0, 6.0, 1.0)
        s.add_cylinder(1.0, 3.0, origin=(5.0, 3.0, -1.0))
        a, b = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))
        s.cut([a], [b])
        params = s.surface_parameters(s.entities(EntityKind.FACE))
        bore = [
            EntityId(int(i))
            for i, t in zip(params.ids, params.types, strict=True)
            if t == "Cylinder"
        ]
        return s, bore

    defeatured, bore = plate()
    suppressed, bore2 = plate()

    defeatured.defeature(bore)
    suppressed.suppress_faces(bore2)

    solid = EntityId(int(defeatured.entities(EntityKind.SOLID)[0]))
    assert np.isclose(defeatured.mass_properties([solid]).measure[0], 60.0, rtol=RTOL)
    assert len(suppressed.entities(EntityKind.SOLID)) == 0
    assert len(suppressed.entities(EntityKind.FACE)) == 6
    rim = suppressed.mass_properties(free_edges(suppressed)).measure
    np.testing.assert_allclose(np.sum(rim), 2 * (2 * np.pi * 1.0), rtol=1e-9)


def test_suppress_faces_leaves_a_snapshot_its_own_shapes() -> None:
    s, top = box()
    mark = s.snapshot()
    before = s.brep()

    s.suppress_faces([top])
    s.restore(mark)

    assert s.brep() == before


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def suppressed() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
        s, top = box()
        s.suppress_faces([top])
        return s.brep(), id_space(s)

    first = suppressed()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(suppressed())

    assert all(other == first for other in others)


def test_suppress_faces_refuses_no_face_an_edge_and_a_dead_face() -> None:
    s, _ = box()
    edge = at(s, EntityKind.EDGE, (1.5, 0.0, 0.0))
    s.add_box(1.0, 1.0, 1.0, origin=(20.0, 0.0, 0.0))
    other_top = at(s, EntityKind.FACE, (20.5, 0.5, 1.0))
    s.remove([EntityId(int(s.entities(EntityKind.SOLID)[-1]))])
    before = state(s)

    with pytest.raises(ps.PysmeshError, match="face_ids must name at least one face"):
        s.suppress_faces([])
    with pytest.raises(ps.PysmeshError, match=f"entity {edge} is a EDGE, not a FACE"):
        s.suppress_faces([edge])
    with pytest.raises(ps.PysmeshError, match=f"entity {other_top} is dead"):
        s.suppress_faces([other_top])

    assert state(s) == before
