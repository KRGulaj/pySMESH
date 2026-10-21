# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for Session.fill_holes: GEOM's FillHoles.

The fixtures are the comparison set of the brief, each an open shell with one hole:

* ``open_box_shell.brep``: the cube ``[0, 2]^3`` without its face ``x = 0``. The hole is the
  2 x 2 square at ``x = 0``: area 4, centroid ``(0, 1, 1)``.
* A cylinder of radius 2 and height 5 without its top cap, revolved from the profile
  ``(0, 0, 0) (2, 0, 0) (2, 0, 5)``. The hole is the disk of radius 2 at ``z = 5``: area
  ``4 pi``, centroid ``(0, 0, 5)``.
* ``sphere_hole_shell.brep``: a sphere shell of radius 3 with the part inside a cylinder
  cut away (see ``generate_fixtures.cpp``). The hole's boundary is not planar.

The new face lies on a GeomPlate surface approximated within 1e-4 (ShHealOper's 3-D
tolerance), not on the exact plane, so its area is compared with the closed form within
``perimeter x 1e-4``, and its centroid within 1e-4. Every other entity keeps its geometry
exactly: 1e-12.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, Session

PLATE_TOL: float = 1e-4
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


def free_edges(s: Session) -> list[EntityId]:
    """The live edges held by one face, with a length (not the degenerate pole edges)."""
    pairs = s.adjacency(EntityKind.EDGE, EntityKind.FACE)
    ids, counts = np.unique(pairs.ids, return_counts=True)
    once = [EntityId(int(i)) for i, c in zip(ids, counts, strict=True) if c == 1]
    return [e for e in once if s.mass_properties([e]).measure[0] > 0.0]


def open_box() -> Session:
    """The cube [0, 2]^3 without its face x = 0, as an open shell."""
    s = Session()
    s.add_brep((FIXTURES / "open_box_shell.brep").read_bytes())
    return s


def open_cylinder() -> Session:
    """The cylinder of radius 2 and height 5 without its top cap, as an open shell."""
    s = Session()
    s.add_polyline([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 5.0)])
    edges = [EntityId(int(e)) for e in s.entities(EntityKind.EDGE)]
    s.revolve(edges, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2 * math.pi)
    return s


def holed_sphere() -> Session:
    """The sphere shell of radius 3 with one non-planar hole."""
    s = Session()
    s.add_brep((FIXTURES / "sphere_hole_shell.brep").read_bytes())
    return s


def any_face(s: Session) -> EntityId:
    """One live face of the model."""
    return EntityId(int(s.entities(EntityKind.FACE)[0]))


def new_faces(s: Session, delta: ps.HistoryDelta) -> list[EntityId]:
    """The created faces of a delta."""
    return [
        EntityId(int(i))
        for i in delta.created
        if s.entity_kind(EntityId(int(i))) == EntityKind.FACE
    ]


@pytest.mark.parametrize(
    ("make", "area", "centroid", "perimeter"),
    [
        (open_box, 4.0, (0.0, 1.0, 1.0), 8.0),
        (open_cylinder, 4.0 * math.pi, (0.0, 0.0, 5.0), 4.0 * math.pi),
    ],
    ids=["box", "cylinder"],
)
def test_fill_holes_closes_the_shell_with_the_face_of_the_hole(
    make: object, area: float, centroid: tuple[float, float, float], perimeter: float
) -> None:
    s = make()  # type: ignore[operator]
    rim = free_edges(s)
    faces_before = len(s.entities(EntityKind.FACE))

    delta = s.fill_holes([any_face(s)])

    (face,) = new_faces(s, delta)
    m = s.mass_properties([face], precision=1e-10)
    assert abs(m.measure[0] - area) <= perimeter * PLATE_TOL
    np.testing.assert_allclose(m.centroid[0], centroid, atol=PLATE_TOL)
    assert len(s.entities(EntityKind.FACE)) == faces_before + 1
    assert bool(s.is_closed_shell([face])[0])
    assert free_edges(s) == []
    origin = s.origin(face)
    assert origin.role == NameRole.GENERATED
    assert sorted(int(i) for i in origin.sources) == sorted(rim)
    assert delta.valid is True


def test_every_id_after_filling_the_box_denotes_its_geometry() -> None:
    """The history check: falsified by tearing the fill's history (final/falsify_*)."""
    s = open_box()
    before = geometry(s)

    delta = s.fill_holes([any_face(s)])
    after = geometry(s)

    assert list(delta.deleted) == []
    assert sorted(set(after) - {int(i) for i in delta.created}) == sorted(before)
    for i, (measure, centroid) in before.items():
        assert np.isclose(after[i][0], measure, rtol=RTOL, atol=ATOL), i
        np.testing.assert_allclose(after[i][1], centroid, rtol=RTOL, atol=ATOL)
    (face,) = new_faces(s, delta)
    assert abs(after[face][0] - 4.0) <= 8.0 * PLATE_TOL
    np.testing.assert_allclose(after[face][1], (0.0, 1.0, 1.0), atol=PLATE_TOL)


def test_fill_holes_closes_a_hole_whose_boundary_is_not_planar() -> None:
    s = holed_sphere()
    rim = free_edges(s)
    assert len(rim) == 2

    delta = s.fill_holes([any_face(s)])

    (face,) = new_faces(s, delta)
    assert bool(s.is_closed_shell([face])[0])
    assert free_edges(s) == []
    assert sorted(int(i) for i in s.origin(face).sources) == sorted(rim)
    boundary = s.adjacency(EntityKind.FACE, EntityKind.EDGE)
    edges = {
        int(e) for f, e in zip(boundary.ids, boundary.related, strict=True) if f == face
    }
    assert set(rim) <= edges
    assert delta.valid is True


def test_fill_holes_with_edges_fills_only_the_hole_they_bound() -> None:
    s = open_cylinder()
    rim = free_edges(s)

    delta = s.fill_holes([any_face(s)], edges=rim)

    assert len(new_faces(s, delta)) == 1
    assert free_edges(s) == []


def test_fill_holes_leaves_a_snapshot_its_own_edges() -> None:
    """ShHealOper writes into the boundary edges; the session fills a copy."""
    s = open_box()
    mark = s.snapshot()
    before = s.brep()

    s.fill_holes([any_face(s)])
    s.restore(mark)

    assert s.brep() == before


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def filled() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
        s = open_box()
        s.fill_holes([any_face(s)])
        return s.brep(), id_space(s)

    first = filled()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(filled())

    assert all(other == first for other in others)


def test_fill_holes_refuses_a_closed_body_and_names_the_entities() -> None:
    s = Session()
    s.add_box(3.0, 7.0, 11.0)
    face = any_face(s)
    before = state(s)

    with pytest.raises(ps.PysmeshError, match="no hole bounded by free edges") as err:
        s.fill_holes([face])

    assert f"entities {face}" in str(err.value)
    assert state(s) == before


def test_fill_holes_refuses_an_edge_held_by_two_faces_or_of_another_body() -> None:
    s = open_box()
    pairs = s.adjacency(EntityKind.EDGE, EntityKind.FACE)
    ids, counts = np.unique(pairs.ids, return_counts=True)
    inner = EntityId(int(ids[counts == 2][0]))
    s.add_line((20.0, 0.0, 0.0), (21.0, 0.0, 0.0))
    stray = EntityId(int(s.entities(EntityKind.EDGE)[-1]))
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=f"edge {inner} is held by 2 faces"):
        s.fill_holes([any_face(s)], edges=[inner])
    with pytest.raises(ps.PysmeshError, match=f"edge {stray} is not an edge of a face"):
        s.fill_holes([any_face(s)], edges=[stray])

    assert state(s) == before


def test_fill_holes_refuses_an_empty_selection_and_a_dead_id() -> None:
    s = open_box()
    s.add_box(1.0, 1.0, 1.0, origin=(20.0, 0.0, 0.0))
    dead = EntityId(int(s.entities(EntityKind.SOLID)[-1]))
    s.remove([dead])

    with pytest.raises(ps.PysmeshError, match="Session.fill_holes: no entity is named"):
        s.fill_holes([])
    with pytest.raises(ps.PysmeshError, match=f"entity {dead} is dead"):
        s.fill_holes([dead])
