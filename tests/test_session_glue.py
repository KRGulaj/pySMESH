# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for Session.glue: GEOM's MakeGlueFaces and MakeGlueEdges.

The fixture is two 3 x 7 x 11 boxes side by side, ``[0, 3]`` and ``[3, 6]`` along x, that
share the whole face ``x = 3``. Gluing moves no geometry: it replaces each group of
coincident shapes by one of them. So the closed form of every entity after the glue is
its closed form before it:

* each box keeps volume 231 and its centroid, ``(1.5, 3.5, 5.5)`` and ``(4.5, 3.5, 5.5)``;
* the shared face has area 77 and centroid ``(3, 3.5, 5.5)``, and both boxes' ids of it
  denote that one face;
* the shared face's four edges and four vertices each carry one id from each box.

Planar polyhedra integrate exactly, so measures and centroids are compared at 1e-12
relative, with an absolute floor of 1e-12 for coordinates near zero.
"""

from __future__ import annotations

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, PysmeshCancelled, Session

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


def two_boxes() -> tuple[Session, EntityId, EntityId]:
    """Two 3 x 7 x 11 boxes that share the face x = 3."""
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    s.add_box(BOX_A, BOX_B, BOX_C, origin=(BOX_A, 0.0, 0.0))
    a, b = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))
    return s, a, b


def geometry(s: Session) -> dict[int, tuple[float, np.ndarray]]:
    """Every live id's measure and centroid, by the rule of entity_table."""
    out: dict[int, tuple[float, np.ndarray]] = {}
    for kind in _KINDS:
        table = s.entity_table(kind)
        for i, m, c in zip(table.ids, table.measure, table.centroid, strict=True):
            out[int(i)] = (float(m), np.array(c))
    return out


def on_shared_plane(s: Session) -> list[int]:
    """The ids of the faces, edges and vertices that lie in the plane x = 3."""
    out: list[int] = []
    for kind in (EntityKind.FACE, EntityKind.EDGE, EntityKind.VERTEX):
        table = s.entity_table(kind)
        lo = table.bbox[:, 0]
        hi = table.bbox[:, 3]
        on = np.isclose(lo, BOX_A, rtol=0.0, atol=ATOL) & np.isclose(
            hi, BOX_A, rtol=0.0, atol=ATOL
        )
        out.extend(int(i) for i in table.ids[on])
    return sorted(out)


def test_glue_leaves_one_shared_face_carrying_both_ids() -> None:
    s, a, b = two_boxes()
    table = s.entity_table(EntityKind.FACE)
    shared = sorted(
        int(i)
        for i, c in zip(table.ids, table.centroid, strict=True)
        if np.allclose(c, (BOX_A, 3.5, 5.5), rtol=0.0, atol=ATOL)
    )
    assert len(shared) == 2

    s.glue([a, b], 1e-7)

    assert sorted(int(i) for i in s.entities(EntityKind.SOLID)) == [a, b]
    assert len(s.entities(EntityKind.FACE, distinct=True)) == 11
    assert [sorted(int(i) for i in g) for g in s.alias_groups(EntityKind.FACE)] == [
        shared
    ]
    m = s.mass_properties([EntityId(i) for i in shared])
    np.testing.assert_allclose(m.measure, [BOX_B * BOX_C] * 2, rtol=RTOL)
    np.testing.assert_allclose(
        m.centroid, [(BOX_A, 3.5, 5.5)] * 2, rtol=RTOL, atol=ATOL
    )
    v = s.mass_properties([a, b])
    np.testing.assert_allclose(v.measure, [BOX_A * BOX_B * BOX_C] * 2, rtol=RTOL)
    np.testing.assert_allclose(
        v.centroid, [(1.5, 3.5, 5.5), (4.5, 3.5, 5.5)], rtol=RTOL, atol=ATOL
    )


def test_every_id_after_a_glue_denotes_the_geometry_it_denoted_before() -> None:
    """The history check: falsified by tearing the glue's history (final/falsify_*)."""
    s, a, b = two_boxes()
    before = geometry(s)
    merged = on_shared_plane(s)
    assert len(merged) == 2 * (1 + 4 + 4)

    delta = s.glue([a, b], 1e-7)
    after = geometry(s)

    assert list(delta.created) == []
    assert list(delta.deleted) == []
    assert sorted(int(i) for i in delta.merged) == merged
    assert sorted(after) == sorted(before)
    for i, (measure, centroid) in after.items():
        assert np.isclose(measure, before[i][0], rtol=RTOL, atol=ATOL), i
        np.testing.assert_allclose(centroid, before[i][1], rtol=RTOL, atol=ATOL)


def test_glue_edges_merges_the_edges_and_keeps_the_faces_apart() -> None:
    s, a, b = two_boxes()
    shared_edges_and_vertices = [
        i
        for i in on_shared_plane(s)
        if s.entity_kind(EntityId(i)) in (EntityKind.EDGE, EntityKind.VERTEX)
    ]

    delta = s.glue([a, b], 1e-7, kind="edges")

    assert len(s.entities(EntityKind.FACE, distinct=True)) == 12
    assert len(s.entities(EntityKind.EDGE, distinct=True)) == 20
    assert sorted(int(i) for i in delta.merged) == shared_edges_and_vertices


def test_glue_without_non_solids_drops_a_free_face() -> None:
    s, a, b = two_boxes()
    s.add_rectangle((20.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0)
    free = EntityId(int(s.entities(EntityKind.FACE)[-1]))

    kept = s.snapshot()
    s.glue([a, b, free], 1e-7)
    assert s.is_alive(free)
    s.restore(kept)
    delta = s.glue([a, b, free], 1e-7, keep_non_solids=False)

    assert not s.is_alive(free)
    assert int(free) in delta.deleted
    assert sorted(int(i) for i in s.entities(EntityKind.SOLID)) == [a, b]


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def glued() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
        s, a, b = two_boxes()
        s.glue([a, b], 1e-7)
        return s.brep(), id_space(s)

    first = glued()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(glued())

    assert all(other == first for other in others)


def test_glue_refuses_bodies_with_nothing_coincident_and_names_them() -> None:
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    s.add_box(1.0, 1.0, 1.0, origin=(5.0, 0.0, 0.0))
    a, b = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))
    before = state(s)

    with pytest.raises(ps.PysmeshError, match="no faces of SOLID") as err:
        s.glue([a, b], 1e-7)

    assert f"SOLID {a}" in str(err.value) and f"SOLID {b}" in str(err.value)
    assert "tolerance 1e-07" in str(err.value)
    assert state(s) == before


@pytest.mark.parametrize(
    ("tolerance", "kind", "match"),
    [
        (0.0, "faces", "tolerance 0 is not above zero"),
        (-1.0, "faces", "tolerance -1 is not above zero"),
        (float("nan"), "faces", "tolerance must be a finite number"),
        (1e-7, "solids", "kind 'solids' is not one of faces, edges"),
    ],
)
def test_glue_refuses_a_degenerate_argument(
    tolerance: float, kind: str, match: str
) -> None:
    s, a, b = two_boxes()
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=match) as err:
        s.glue([a, b], tolerance, kind=kind)  # type: ignore[arg-type]

    assert "Session.glue" in str(err.value)
    assert state(s) == before


def test_glue_refuses_an_empty_selection_and_a_dead_id() -> None:
    s, a, _ = two_boxes()
    s.add_box(1.0, 1.0, 1.0, origin=(20.0, 0.0, 0.0))
    dead = EntityId(int(s.entities(EntityKind.SOLID)[-1]))
    s.remove([dead])

    with pytest.raises(ps.PysmeshError, match="Session.glue: no entity is named"):
        s.glue([], 1e-7)
    with pytest.raises(ps.PysmeshError, match=f"entity {dead} is dead"):
        s.glue([a, dead], 1e-7)


def test_a_cancelled_glue_raises_and_leaves_the_session_as_it_was() -> None:
    s, a, b = two_boxes()
    before = state(s)

    with pytest.raises(PysmeshCancelled):
        s.glue([a, b], 1e-7, cancel=lambda: True)

    assert state(s) == before


def test_glue_reports_progress_to_the_end() -> None:
    s, a, b = two_boxes()
    seen: list[float] = []

    s.glue([a, b], 1e-7, progress=seen.append)

    assert seen
    assert seen == sorted(seen)
    assert seen[-1] == 1.0
