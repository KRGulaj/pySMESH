# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-05

"""Behaviour of the stateful modelling session: operations, state and contracts.

Covers the live-shape rule, entity identity across every operation, snapshot/restore,
provenance naming, the session-independence contract, and the fact that the stateless free
functions are untouched. The identity *gate* — ground-truth verification over long operation
sequences, plus its falsification — lives in ``test_session_identity.py``.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube, so that a
transposed axis or a swapped extent cannot pass unnoticed.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, ResolutionStatus, Session

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0
BOX_VOLUME: float = BOX_DX * BOX_DY * BOX_DZ
BOX_FACE_AREAS: tuple[float, ...] = (21.0, 21.0, 33.0, 33.0, 77.0, 77.0)

# Off the origin, and off it on every axis with a negative component among them. An
# operation that rebuilds geometry in its own local frame and forgets to put it back lands
# on the origin, and a fixture sitting there cannot tell the difference.
BOX_ORIGIN: tuple[float, float, float] = (2.0, -3.0, 5.0)

# A box carries 1 solid + 6 faces + 12 edges + 8 vertices.
BOX_ENTITY_COUNT: int = 27


@pytest.fixture
def box_session() -> Session:
    """A session holding one 3 x 7 x 11 box at the origin."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    return s


@pytest.fixture
def placed_box_session() -> Session:
    """A session holding one 3 x 7 x 11 box at :data:`BOX_ORIGIN`."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=BOX_ORIGIN)
    return s


@pytest.fixture
def two_box_session() -> Session:
    """Two 3 x 7 x 11 boxes meeting face to face at x = 3, not yet fused."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(BOX_DX, 0.0, 0.0))
    return s


def _ids(session: Session, kind: EntityKind) -> list[int]:
    """Live entity ids of one kind as a plain sorted list."""
    return session.entities(kind).tolist()


def _sole(session: Session, kind: EntityKind) -> EntityId:
    """The single live entity of a kind; fails the test if there is not exactly one."""
    ids = _ids(session, kind)
    assert len(ids) == 1
    return EntityId(ids[0])


# --------------------------------------------------------------------------------------- #
# One live shape
# --------------------------------------------------------------------------------------- #


def test_new_session_has_no_entities_and_no_operations() -> None:
    s = Session()

    assert s.entity_count == 0
    assert s.op_count == 0
    assert s.issued_id_count == 0


def test_add_box_issues_one_id_per_subshape_of_every_tracked_kind(
    box_session: Session,
) -> None:
    counts = {k: len(_ids(box_session, k)) for k in EntityKind}

    assert counts == {
        EntityKind.SOLID: 1,
        EntityKind.FACE: 6,
        EntityKind.EDGE: 12,
        EntityKind.VERTEX: 8,
    }
    assert box_session.entity_count == BOX_ENTITY_COUNT


def test_add_box_builds_the_requested_extents(box_session: Session) -> None:
    solids = box_session.entity_table(EntityKind.SOLID)
    faces = box_session.entity_table(EntityKind.FACE)

    assert solids.measure[0] == pytest.approx(BOX_VOLUME)
    assert sorted(faces.measure.tolist()) == pytest.approx(sorted(BOX_FACE_AREAS))


def test_add_cylinder_builds_the_analytic_volume() -> None:
    s = Session()
    radius, height = 2.0, 5.0

    s.add_cylinder(radius, height)

    volume = s.entity_table(EntityKind.SOLID).measure[0]
    assert volume == pytest.approx(math.pi * radius**2 * height)


def test_add_brep_imports_a_body_and_issues_ids(box_brep: bytes) -> None:
    s = Session()

    delta = s.add_brep(box_brep)

    assert delta.created.size == s.entity_count
    assert len(_ids(s, EntityKind.SOLID)) == 1


def test_a_later_operation_does_not_mutate_a_retained_snapshots_shape(
    box_session: Session,
) -> None:
    mark = box_session.snapshot()
    before = box_session.brep()

    box_session.add_box(1.0, 1.0, 1.0, origin=(20.0, 0.0, 0.0))
    box_session.translate((5.0, 5.0, 5.0))
    box_session.restore(mark)

    # Byte-identical, not merely equivalent: an operation that mutated a shared shape in
    # place would leave the retained state changed even though nothing "restored" it.
    assert box_session.brep() == before


@pytest.mark.parametrize(
    ("dx", "dy", "dz"),
    [(0.0, 7.0, 11.0), (3.0, -1.0, 11.0), (3.0, 7.0, 0.0)],
)
def test_add_box_with_a_non_positive_extent_raises(
    dx: float, dy: float, dz: float
) -> None:
    s = Session()

    with pytest.raises(ps.PysmeshError, match="must be > 0"):
        s.add_box(dx, dy, dz)


def test_add_cylinder_with_a_zero_axis_raises() -> None:
    s = Session()

    with pytest.raises(ps.PysmeshError, match="non-zero vector"):
        s.add_cylinder(2.0, 5.0, axis=(0.0, 0.0, 0.0))


def test_add_brep_with_malformed_bytes_raises() -> None:
    s = Session()

    with pytest.raises(ps.PysmeshError, match="BREP read"):
        s.add_brep(b"not a brep at all")


# --------------------------------------------------------------------------------------- #
# Identity across operations
# --------------------------------------------------------------------------------------- #


def test_translate_preserves_every_entity_id_one_by_one(box_session: Session) -> None:
    before = {k: _ids(box_session, k) for k in EntityKind}

    delta = box_session.translate((100.0, -3.0, 0.5))

    after = {k: _ids(box_session, k) for k in EntityKind}
    assert after == before
    assert delta.created.size == 0
    assert delta.deleted.size == 0


def test_translate_moves_the_geometry_by_exactly_the_offset(box_session: Session) -> None:
    offset = np.array([100.0, -3.0, 0.5])
    before = box_session.entity_table(EntityKind.FACE)

    box_session.translate((100.0, -3.0, 0.5))

    after = box_session.entity_table(EntityKind.FACE)
    assert np.array_equal(after.ids, before.ids)
    assert after.centroid == pytest.approx(before.centroid + offset)
    assert after.measure == pytest.approx(before.measure)


def test_rotate_preserves_every_id_and_applies_the_rotation(box_session: Session) -> None:
    before = box_session.entity_table(EntityKind.FACE)
    rot_z_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    box_session.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi / 2.0)

    after = box_session.entity_table(EntityKind.FACE)
    assert np.array_equal(after.ids, before.ids)
    assert after.centroid == pytest.approx(before.centroid @ rot_z_90.T)


def test_translating_one_body_leaves_the_other_body_untouched(
    two_box_session: Session,
) -> None:
    # The two boxes meet at x = 3 but are separate bodies with no shared sub-shapes until a
    # boolean joins them, so one may move on its own.
    solids = _ids(two_box_session, EntityKind.SOLID)
    before = two_box_session.entity_table(EntityKind.SOLID)

    two_box_session.translate((0.0, 0.0, 50.0), [EntityId(solids[1])])

    after = two_box_session.entity_table(EntityKind.SOLID)
    assert np.array_equal(after.ids, before.ids)
    assert after.centroid[0] == pytest.approx(before.centroid[0])
    assert after.centroid[1] == pytest.approx(before.centroid[1] + [0.0, 0.0, 50.0])


def test_naming_a_face_translates_the_whole_body_that_owns_it(
    two_box_session: Session,
) -> None:
    solids = _ids(two_box_session, EntityKind.SOLID)
    two_box_session.fuse([EntityId(solids[0])], [EntityId(solids[1])])
    fused = _sole(two_box_session, EntityKind.SOLID)
    face = EntityId(_ids(two_box_session, EntityKind.FACE)[0])
    small = two_box_session.add_box(1.0, 1.0, 1.0, origin=(50.0, 0.0, 0.0))
    small_solid = int(
        next(i for i in small.created if two_box_session.entity_kind(EntityId(int(i))) == EntityKind.SOLID)
    )
    before = two_box_session.entity_table(EntityKind.SOLID)

    delta = two_box_session.translate((0.0, 0.0, 5.0), [face])

    after = two_box_session.entity_table(EntityKind.SOLID)
    moved = dict(zip(after.ids.tolist(), after.centroid[:, 2].tolist(), strict=True))
    was = dict(zip(before.ids.tolist(), before.centroid[:, 2].tolist(), strict=True))
    assert delta.deleted.size == 0
    assert fused in delta.modified.tolist()
    assert moved[fused] == pytest.approx(was[fused] + 5.0)
    assert moved[small_solid] == pytest.approx(was[small_solid])


def test_translate_with_an_empty_entity_list_raises(box_session: Session) -> None:
    with pytest.raises(ps.PysmeshError, match="pass None"):
        box_session.translate((1.0, 0.0, 0.0), [])


def test_fuse_of_two_touching_boxes_yields_the_summed_volume(
    two_box_session: Session,
) -> None:
    solids = _ids(two_box_session, EntityKind.SOLID)

    two_box_session.fuse([EntityId(solids[0])], [EntityId(solids[1])])

    fused = two_box_session.entity_table(EntityKind.SOLID)
    assert fused.ids.size == 1
    assert fused.measure[0] == pytest.approx(2.0 * BOX_VOLUME)


def test_fuse_kills_exactly_the_two_seam_faces_and_keeps_the_rest(
    two_box_session: Session,
) -> None:
    faces_before = two_box_session.entity_table(EntityKind.FACE)
    solids = _ids(two_box_session, EntityKind.SOLID)

    # The seam is the pair of faces at x = 3, identified from geometry alone — an
    # independent labelling, not the registry that is under test.
    seam = {
        int(i)
        for i, c in zip(faces_before.ids, faces_before.centroid, strict=True)
        if c[0] == pytest.approx(BOX_DX)
    }
    assert len(seam) == 2

    delta = two_box_session.fuse([EntityId(solids[0])], [EntityId(solids[1])])

    survivors = set(_ids(two_box_session, EntityKind.FACE))
    assert seam.isdisjoint(survivors)
    assert seam <= set(delta.deleted.tolist())
    assert survivors == set(faces_before.ids.tolist()) - seam


def test_fuse_carries_surviving_face_ids_onto_the_same_geometry(
    two_box_session: Session,
) -> None:
    before = two_box_session.entity_table(EntityKind.FACE)
    truth = {
        int(i): (float(m), tuple(c))
        for i, m, c in zip(before.ids, before.measure, before.centroid, strict=True)
    }
    solids = _ids(two_box_session, EntityKind.SOLID)

    two_box_session.fuse([EntityId(solids[0])], [EntityId(solids[1])])

    after = two_box_session.entity_table(EntityKind.FACE)
    for i, m, c in zip(after.ids, after.measure, after.centroid, strict=True):
        want_area, want_centroid = truth[int(i)]
        assert m == pytest.approx(want_area), f"face {i} changed area"
        assert tuple(c) == pytest.approx(want_centroid), f"face {i} moved"


def test_fuse_merges_seam_vertex_ids_many_to_one(two_box_session: Session) -> None:
    solids = _ids(two_box_session, EntityKind.SOLID)

    delta = two_box_session.fuse([EntityId(solids[0])], [EntityId(solids[1])])

    # The four corners of the seam existed on both boxes; after the fuse each is one vertex
    # carrying both original ids.
    assert delta.merged.size > 0
    merged_vertices = [
        i for i in delta.merged.tolist() if two_box_session.entity_kind(EntityId(i)) == EntityKind.VERTEX
    ]
    assert len(merged_vertices) == 8


def test_a_face_cut_in_two_keeps_its_id_on_both_pieces() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    # A bar crossing the x = 3 face from edge to edge severs it into two disjoint pieces.
    s.add_box(2.0, 9.0, 3.0, origin=(2.0, -1.0, 4.0))
    solids = _ids(s, EntityKind.SOLID)

    delta = s.fuse([EntityId(solids[0])], [EntityId(solids[1])])

    split_faces = [
        i for i in delta.split.tolist() if s.entity_kind(EntityId(i)) == EntityKind.FACE
    ]
    assert split_faces, "expected the severed face to split"
    for i in split_faces:
        assert s.shape_count(EntityId(i)) == 2


def test_fillet_kills_the_filleted_edge_and_adds_a_face(box_session: Session) -> None:
    edge = EntityId(_ids(box_session, EntityKind.EDGE)[0])
    faces_before = len(_ids(box_session, EntityKind.FACE))

    box_session.fillet([edge], 0.5)

    assert not box_session.is_alive(edge)
    assert len(_ids(box_session, EntityKind.FACE)) == faces_before + 1


def test_fillet_with_an_impossible_radius_raises_naming_the_edges(
    box_session: Session,
) -> None:
    edge = EntityId(_ids(box_session, EntityKind.EDGE)[0])

    with pytest.raises(ps.PysmeshError) as excinfo:
        box_session.fillet([edge], 500.0)

    assert edge in excinfo.value.face_ids


def test_fillet_leaves_the_session_unchanged_when_it_fails(box_session: Session) -> None:
    edge = EntityId(_ids(box_session, EntityKind.EDGE)[0])
    before = box_session.brep()
    entities_before = {k: _ids(box_session, k) for k in EntityKind}

    with pytest.raises(ps.PysmeshError):
        box_session.fillet([edge], 500.0)

    assert box_session.brep() == before
    assert {k: _ids(box_session, k) for k in EntityKind} == entities_before


def test_fillet_on_a_face_id_raises_naming_the_wrong_kind(box_session: Session) -> None:
    face = EntityId(_ids(box_session, EntityKind.FACE)[0])

    with pytest.raises(ps.PysmeshError, match="is a FACE, not an EDGE"):
        box_session.fillet([face], 0.5)


def test_fuse_with_a_face_id_as_a_target_raises(two_box_session: Session) -> None:
    face = EntityId(_ids(two_box_session, EntityKind.FACE)[0])
    solid = EntityId(_ids(two_box_session, EntityKind.SOLID)[1])

    with pytest.raises(ps.PysmeshError, match="not a SOLID"):
        two_box_session.fuse([face], [solid])


def test_fuse_with_a_negative_fuzzy_value_raises(two_box_session: Session) -> None:
    solids = _ids(two_box_session, EntityKind.SOLID)

    with pytest.raises(ps.PysmeshError, match="fuzzy must be a finite value >= 0"):
        two_box_session.fuse(
            [EntityId(solids[0])], [EntityId(solids[1])], fuzzy=-1.0
        )


def test_an_id_is_never_reused_after_the_entity_dies(two_box_session: Session) -> None:
    solids = _ids(two_box_session, EntityKind.SOLID)

    delta = two_box_session.fuse([EntityId(solids[0])], [EntityId(solids[1])])
    dead = set(delta.deleted.tolist())
    two_box_session.add_box(1.0, 1.0, 1.0, origin=(50.0, 0.0, 0.0))
    for _ in range(5):
        two_box_session.translate((1.0, 0.0, 0.0))

    assert dead, "the fuse must have killed something for this test to mean anything"
    assert dead.isdisjoint(set(two_box_session.entities(EntityKind.FACE).tolist()))
    for i in dead:
        assert not two_box_session.is_alive(EntityId(i))


def test_an_id_the_session_never_issued_raises(box_session: Session) -> None:
    beyond = EntityId(box_session.issued_id_count + 1)

    with pytest.raises(ps.PysmeshError, match="never issued|not an EntityId"):
        box_session.is_alive(beyond)


def test_a_dead_id_cannot_be_used_as_an_operand(two_box_session: Session) -> None:
    solids = _ids(two_box_session, EntityKind.SOLID)
    delta = two_box_session.fuse([EntityId(solids[0])], [EntityId(solids[1])])
    dead = EntityId(int(delta.deleted[0]))

    with pytest.raises(ps.PysmeshError, match="is dead"):
        two_box_session.entity_kind(dead)


# --------------------------------------------------------------------------------------- #
# The offset family: make_thick_solid and offset
# --------------------------------------------------------------------------------------- #


def _face_sharing_an_edge(session: Session, face: EntityId) -> tuple[EntityId, EntityId]:
    """A neighbour of ``face``, and the edge the two of them share."""
    pairs = session.adjacency(EntityKind.FACE, EntityKind.EDGE)
    edges: dict[int, set[int]] = {}
    for f, e in zip(pairs.ids.tolist(), pairs.related.tolist(), strict=True):
        edges.setdefault(f, set()).add(e)
    for other, theirs in edges.items():
        shared = edges[int(face)] & theirs
        if other != int(face) and shared:
            return EntityId(other), EntityId(min(shared))
    raise AssertionError("a box face has neighbours")


def test_make_thick_solid_leaves_the_closed_form_wall_volume(
    placed_box_session: Session,
) -> None:
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    # A box's first face is the one at x = xmin, so the cavity loses a wall on five sides
    # and none on the sixth.
    thickness = 0.5
    expected = BOX_VOLUME - (BOX_DX - thickness) * (BOX_DY - 2 * thickness) * (
        BOX_DZ - 2 * thickness
    )

    placed_box_session.make_thick_solid([face], -thickness)

    volume = placed_box_session.entity_table(EntityKind.SOLID).measure[0]
    assert volume == pytest.approx(expected)


def test_make_thick_solid_with_a_positive_thickness_grows_the_wall_outward(
    placed_box_session: Session,
) -> None:
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    thickness = 0.5
    expected = (BOX_DX + thickness) * (BOX_DY + 2 * thickness) * (
        BOX_DZ + 2 * thickness
    ) - BOX_VOLUME

    placed_box_session.make_thick_solid([face], thickness)

    volume = placed_box_session.entity_table(EntityKind.SOLID).measure[0]
    assert volume == pytest.approx(expected)


def test_make_thick_solid_delta_names_every_id_it_moved(
    placed_box_session: Session,
) -> None:
    opened = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    solid = _sole(placed_box_session, EntityKind.SOLID)
    before = {k: _ids(placed_box_session, k) for k in EntityKind}

    delta = placed_box_session.make_thick_solid([opened], -0.5)

    assert delta.deleted.tolist() == [opened]
    assert delta.modified.tolist() == [solid]
    assert delta.split.tolist() == []
    assert delta.merged.tolist() == []
    assert delta.valid is True
    # A hollow at one face adds five inner walls plus the rim at the opening, with a full
    # set of edges and vertices for them, and adds no solid: the body is the same body.
    created = delta.created.tolist()
    kinds = [placed_box_session.entity_kind(EntityId(i)) for i in created]
    assert kinds.count(EntityKind.SOLID) == 0
    assert kinds.count(EntityKind.FACE) == 6
    assert kinds.count(EntityKind.EDGE) == 12
    assert kinds.count(EntityKind.VERTEX) == 8
    # Every surviving id, one by one: the solid, the five faces that were not opened, and
    # every edge and vertex of the original box.
    new_faces = [
        i
        for i in created
        if placed_box_session.entity_kind(EntityId(i)) == EntityKind.FACE
    ]
    assert _ids(placed_box_session, EntityKind.SOLID) == [solid]
    assert _ids(placed_box_session, EntityKind.FACE) == sorted(
        [i for i in before[EntityKind.FACE] if i != opened] + new_faces
    )
    assert set(before[EntityKind.EDGE]) <= set(_ids(placed_box_session, EntityKind.EDGE))
    assert set(before[EntityKind.VERTEX]) <= set(
        _ids(placed_box_session, EntityKind.VERTEX)
    )


def test_make_thick_solid_kills_an_edge_only_the_opened_faces_carried(
    placed_box_session: Session,
) -> None:
    first = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    second, shared = _face_sharing_an_edge(placed_box_session, first)

    delta = placed_box_session.make_thick_solid([first, second], -0.5)

    # That edge belonged to those two faces and to nothing else, so it goes with them.
    assert delta.deleted.tolist() == sorted([first, second, shared])
    assert not placed_box_session.is_alive(shared)


def test_make_thick_solid_keeps_the_unopened_walls_where_they_were(
    placed_box_session: Session,
) -> None:
    table = placed_box_session.entity_table(EntityKind.FACE)
    truth = {int(i): float(m) for i, m in zip(table.ids, table.measure, strict=True)}
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])

    placed_box_session.make_thick_solid([face], -0.5)

    after = placed_box_session.entity_table(EntityKind.FACE)
    for i, m in zip(after.ids, after.measure, strict=True):
        if int(i) in truth:
            assert m == pytest.approx(truth[int(i)]), f"wall {i} changed area"


def test_make_thick_solid_on_a_dead_id_raises(placed_box_session: Session) -> None:
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    placed_box_session.make_thick_solid([face], -0.5)

    with pytest.raises(ps.PysmeshError, match=f"entity {face} is dead"):
        placed_box_session.make_thick_solid([face], -0.2)


def test_make_thick_solid_on_an_edge_id_raises_naming_the_wrong_kind(
    placed_box_session: Session,
) -> None:
    edge = EntityId(_ids(placed_box_session, EntityKind.EDGE)[0])

    with pytest.raises(ps.PysmeshError, match=f"entity {edge} is a EDGE, not a FACE"):
        placed_box_session.make_thick_solid([edge], -0.5)


def test_make_thick_solid_across_two_bodies_raises(two_box_session: Session) -> None:
    faces = _ids(two_box_session, EntityKind.FACE)

    with pytest.raises(ps.PysmeshError, match="belong to 2 different bodies"):
        two_box_session.make_thick_solid([EntityId(faces[0]), EntityId(faces[-1])], -0.2)


def test_make_thick_solid_on_a_body_that_is_not_a_solid_raises(
    placed_box_session: Session,
) -> None:
    delta = placed_box_session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, 3.0)
    face = EntityId(
        next(
            i
            for i in delta.created.tolist()
            if placed_box_session.entity_kind(EntityId(i)) == EntityKind.FACE
        )
    )

    with pytest.raises(ps.PysmeshError, match="FACE body; hollowing needs a SOLID"):
        placed_box_session.make_thick_solid([face], -0.2)


@pytest.mark.parametrize("thickness", [0.0, float("nan"), float("inf")])
def test_make_thick_solid_with_a_zero_or_non_finite_thickness_raises(
    placed_box_session: Session, thickness: float
) -> None:
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])

    with pytest.raises(ps.PysmeshError, match="thickness must be a finite non-zero"):
        placed_box_session.make_thick_solid([face], thickness)


@pytest.mark.parametrize("tol", [0.0, -1.0e-7])
def test_make_thick_solid_with_a_non_positive_tol_raises(
    placed_box_session: Session, tol: float
) -> None:
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])

    with pytest.raises(ps.PysmeshError, match="tol must be > 0"):
        placed_box_session.make_thick_solid([face], -0.5, tol=tol)


def test_make_thick_solid_that_self_intersects_raises_naming_input_faces(
    placed_box_session: Session,
) -> None:
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    live = set(_ids(placed_box_session, EntityKind.FACE))

    # A wall as thick as the box's smallest extent folds the inner shell through itself.
    with pytest.raises(ps.PysmeshError) as excinfo:
        placed_box_session.make_thick_solid([face], -BOX_DX)

    blamed = set(excinfo.value.face_ids)
    assert blamed, "the failure must name the faces it broke on"
    # Live EntityIds of the input, and fewer than all of them: the result's broken faces
    # were traced back through the history rather than the whole selection being blamed.
    assert blamed < live
    assert blamed != {face}, "the blame must be traced, not the selection echoed back"


def test_make_thick_solid_leaves_the_session_unchanged_when_it_fails(
    placed_box_session: Session,
) -> None:
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    before = placed_box_session.brep()
    entities_before = {k: _ids(placed_box_session, k) for k in EntityKind}

    with pytest.raises(ps.PysmeshError):
        placed_box_session.make_thick_solid([face], -BOX_DX)

    assert placed_box_session.brep() == before
    assert {k: _ids(placed_box_session, k) for k in EntityKind} == entities_before


# --------------------------------------------------------------------------------------- #
# A hollowing that comes back as the body it was built from
#
# Measured on the released 4.1.1 wheel, this box opened at the face of largest z: at -1.50
# the analyzer rejected the result, from -1.51 to -1.55 OCCT declined, and from -1.56 down
# to -5.00 it reported success and handed back the input solid — volume 231, six faces, the
# opened face re-issued under a new id. 4.1.1 committed that. The regime the guard has to
# leave alone is the thin wall just above it: -1.40 hollows correctly to 222.936.
# --------------------------------------------------------------------------------------- #

# Half the box's smallest extent. The hollowing accepts every |thickness| under it and
# refuses every one at or beyond it: measured on this fixture, the last accepted value is
# -1.4999999991410775 and the first refused one is -1.4999999991410777.
HALF_SMALLEST_EXTENT: float = BOX_DX / 2.0


def _top_face(session: Session) -> EntityId:
    """The face of largest z, which is the one every hollowing below opens."""
    boxes = session.bounding_boxes(EntityKind.FACE)
    top, height = -1, -math.inf
    for i, b in zip(boxes.ids.tolist(), boxes.bbox.tolist(), strict=True):
        if b[2] > height:
            top, height = int(i), b[2]
    assert top >= 0
    return EntityId(top)


def _whole_state(session: Session) -> tuple[object, ...]:
    """The operation count, the id counter and every live id — what a refusal must keep.

    The BREP bytes are deliberately not here. MakeThickSolidByJoin raises the *input*
    shape's stored tolerances in place, from 1e-07 to 1.00000000111022e-07, for some
    selections. That happens inside OCCT before any post-condition runs, it is the same in
    4.1.1, and it changes no id and no geometry. The byte-for-byte contract is asserted by
    :func:`test_make_thick_solid_leaves_the_session_unchanged_when_it_fails` on a selection
    where it holds.
    """
    return (
        session.op_count,
        session.issued_id_count,
        {k: _ids(session, k) for k in EntityKind},
    )


def test_make_thick_solid_at_a_thin_wall_leaves_the_closed_form_volume(
    placed_box_session: Session,
) -> None:
    top = _top_face(placed_box_session)
    thickness = 1.4
    # Opened at the top, the cavity loses a wall on the four sides and on the bottom only.
    expected = BOX_VOLUME - (BOX_DX - 2 * thickness) * (BOX_DY - 2 * thickness) * (
        BOX_DZ - thickness
    )

    delta = placed_box_session.make_thick_solid([top], -thickness)

    assert expected == pytest.approx(BOX_VOLUME - 0.2 * 4.2 * 9.6)
    assert placed_box_session.entity_table(EntityKind.SOLID).measure[0] == pytest.approx(
        expected
    )
    assert delta.valid is True
    assert delta.deleted.tolist() == [top]
    assert len(_ids(placed_box_session, EntityKind.FACE)) == 11


def test_make_thick_solid_at_a_wall_that_clears_every_feature_hollows_the_box(
    placed_box_session: Session,
) -> None:
    top = _top_face(placed_box_session)
    thickness = 0.37
    expected = BOX_VOLUME - (BOX_DX - 2 * thickness) * (BOX_DY - 2 * thickness) * (
        BOX_DZ - thickness
    )

    delta = placed_box_session.make_thick_solid([top], -thickness)

    assert delta.valid is True
    assert delta.deleted.tolist() == [top]
    # Five inner walls and the rim, with a full set of edges and vertices under them.
    assert len(delta.created.tolist()) == 26
    assert len(_ids(placed_box_session, EntityKind.FACE)) == 11
    assert placed_box_session.entity_table(EntityKind.SOLID).measure[0] == pytest.approx(
        expected
    )


def test_make_thick_solid_the_analyzer_rejects_raises_and_changes_nothing(
    placed_box_session: Session,
) -> None:
    top = _top_face(placed_box_session)
    before = _whole_state(placed_box_session)

    # Exactly half the smallest extent: the inner shell meets itself, and the faces OCCT
    # builds there are broken rather than absent.
    with pytest.raises(ps.PysmeshError, match="rejected"):
        placed_box_session.make_thick_solid([top], -HALF_SMALLEST_EXTENT)

    assert _whole_state(placed_box_session) == before


@pytest.mark.parametrize("thickness", [-1.6, -2.0, -3.0, -5.0])
def test_make_thick_solid_that_returns_the_input_solid_raises_and_changes_nothing(
    placed_box_session: Session, thickness: float
) -> None:
    top = _top_face(placed_box_session)
    before = _whole_state(placed_box_session)

    # OCCT reports success here and BRepCheck_Analyzer accepts the shape. What comes back is
    # the input solid with the face that was to be opened re-issued under a new id, so
    # neither the `unopened` post-condition nor the analyzer can see it.
    with pytest.raises(ps.PysmeshError, match="is not a hollowed solid") as excinfo:
        placed_box_session.make_thick_solid([top], thickness)

    assert list(excinfo.value.face_ids) == [top]
    assert f"{thickness:.6f}" in str(excinfo.value)
    assert f"{BOX_VOLUME:.6f}" in str(excinfo.value)
    assert _whole_state(placed_box_session) == before


def test_make_thick_solid_opening_every_face_raises_and_changes_nothing(
    placed_box_session: Session,
) -> None:
    faces = [EntityId(i) for i in _ids(placed_box_session, EntityKind.FACE)]
    before = _whole_state(placed_box_session)

    # Every face opened leaves no wall to build, and 4.1.1 committed the input solid for it
    # at every thickness measured, -0.05 through -3.0. This thickness is one the same box
    # hollows correctly when a single face is opened, so no magnitude rule would catch it.
    with pytest.raises(ps.PysmeshError, match="is not a hollowed solid") as excinfo:
        placed_box_session.make_thick_solid(faces, -0.5)

    assert list(excinfo.value.face_ids) == faces
    assert "no cavity was cut" in str(excinfo.value)
    assert _whole_state(placed_box_session) == before


def test_make_thick_solid_that_returns_the_input_within_round_off_raises() -> None:
    # A second placement, and the only test that needs one: the volume a collapsed result
    # measures is exact at :data:`BOX_ORIGIN` and a hair under the input here, so this is
    # where the round-off case can be pinned at all.
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(0.37, -2.9, 1.3))
    table = session.entity_table(EntityKind.FACE)
    normal_to_y = [
        EntityId(int(i))
        for i, m in zip(table.ids.tolist(), table.measure.tolist(), strict=True)
        if m == pytest.approx(BOX_DX * BOX_DZ)
    ]
    opened = sorted(normal_to_y + [_top_face(session)])
    assert len(opened) == 3
    before = _whole_state(session)

    # The two walls normal to y and the top, opened at -3.0. 4.1.1 committed the input solid
    # for this one too, but measured its volume as 230.99999999999997 — under the input's
    # 231 by 2.8e-14. A volume rule cannot see a collapse that far inside round-off. The
    # wall statement is what catches it, which is why the post-condition makes both.
    with pytest.raises(ps.PysmeshError, match="wall the offset built") as excinfo:
        session.make_thick_solid(opened, -3.0)

    assert "no cavity was cut" not in str(excinfo.value)
    assert list(excinfo.value.face_ids) == opened
    assert _whole_state(session) == before


@pytest.mark.parametrize("thickness", [0.05, 0.5, 2.0])
def test_make_thick_solid_that_returns_an_inside_out_wall_raises(
    placed_box_session: Session, thickness: float
) -> None:
    faces = _ids(placed_box_session, EntityKind.FACE)
    unopened = placed_box_session.entity_table(EntityKind.FACE).measure[0]
    opened = [EntityId(i) for i in faces[1:]]
    before = _whole_state(placed_box_session)

    # Every face but one opened, thickened outward: the wall is the one remaining face
    # raised into a plate, and OCCT builds it with its orientation reversed. 4.1.1 committed
    # it, so the session reported a body of volume -(area x thickness) — -38.5 at +0.5 on
    # the face of area 77. A caller taking that for a mass gets a negative one.
    with pytest.raises(ps.PysmeshError, match="turned inside out") as excinfo:
        placed_box_session.make_thick_solid(opened, thickness)

    assert f"{-unopened * thickness:.6f}" in str(excinfo.value)
    assert list(excinfo.value.face_ids) == opened
    assert _whole_state(placed_box_session) == before


def test_make_thick_solid_hollows_just_under_half_the_smallest_extent(
    placed_box_session: Session,
) -> None:
    top = _top_face(placed_box_session)
    thickness = HALF_SMALLEST_EXTENT * (1.0 - 1.0e-8)

    delta = placed_box_session.make_thick_solid([top], -thickness)

    # The wall is the whole body bar a sliver, and it is still a hollowing: the cavity has a
    # positive volume and the inner walls are all there.
    assert delta.valid is True
    assert len(_ids(placed_box_session, EntityKind.FACE)) == 11
    volume = float(placed_box_session.entity_table(EntityKind.SOLID).measure[0])
    assert 0.0 < BOX_VOLUME - volume < 1.0e-6


@pytest.mark.parametrize("scale", [1.0, 1.0 + 1.0e-6, 1.04, 2.0, 10.0])
def test_make_thick_solid_refuses_from_half_the_smallest_extent_on(
    placed_box_session: Session, scale: float
) -> None:
    top = _top_face(placed_box_session)
    before = _whole_state(placed_box_session)

    # The boundary, pinned: every |thickness| from half the smallest extent on is refused,
    # whichever of the three ways OCCT fails at it. 4.1.1 refused only as far as 1.55 and
    # committed the input solid for everything past that, 1.04 x 1.5 = 1.56 included.
    with pytest.raises(ps.PysmeshError):
        placed_box_session.make_thick_solid([top], -HALF_SMALLEST_EXTENT * scale)

    assert _whole_state(placed_box_session) == before


def test_make_thick_solid_outward_past_the_smallest_extent_still_builds_the_wall(
    placed_box_session: Session,
) -> None:
    top = _top_face(placed_box_session)
    thickness = 2.0
    expected = (BOX_DX + 2 * thickness) * (BOX_DY + 2 * thickness) * (
        BOX_DZ + thickness
    ) - BOX_VOLUME

    delta = placed_box_session.make_thick_solid([top], thickness)

    # Outward the wall lies outside the original boundary, so nothing bounds the thickness
    # and the inward rule must not be read onto it.
    assert expected == pytest.approx(770.0)
    assert delta.valid is True
    assert placed_box_session.entity_table(EntityKind.SOLID).measure[0] == pytest.approx(
        expected
    )
    assert len(_ids(placed_box_session, EntityKind.FACE)) == 11


def test_a_cancelled_make_thick_solid_changes_nothing(
    placed_box_session: Session,
) -> None:
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    before = (
        placed_box_session.op_count,
        placed_box_session.issued_id_count,
        {k: _ids(placed_box_session, k) for k in EntityKind},
        placed_box_session.brep(),
    )

    with pytest.raises(ps.PysmeshCancelled):
        placed_box_session.make_thick_solid([face], -0.5, cancel=lambda: True)

    assert (
        placed_box_session.op_count,
        placed_box_session.issued_id_count,
        {k: _ids(placed_box_session, k) for k in EntityKind},
        placed_box_session.brep(),
    ) == before


def test_make_thick_solid_round_trips_through_a_snapshot(
    placed_box_session: Session,
) -> None:
    mark = placed_box_session.snapshot()
    before = placed_box_session.brep()
    entities_before = {k: _ids(placed_box_session, k) for k in EntityKind}
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])

    placed_box_session.make_thick_solid([face], -0.5)
    placed_box_session.restore(mark)

    assert placed_box_session.brep() == before
    assert {k: _ids(placed_box_session, k) for k in EntityKind} == entities_before


def test_offset_enlarges_the_box_by_the_distance_on_every_side(
    placed_box_session: Session,
) -> None:
    solid = _sole(placed_box_session, EntityKind.SOLID)
    d = 0.5

    placed_box_session.offset([solid], d)

    table = placed_box_session.entity_table(EntityKind.SOLID)
    assert table.measure[0] == pytest.approx(
        (BOX_DX + 2 * d) * (BOX_DY + 2 * d) * (BOX_DZ + 2 * d)
    )
    box = placed_box_session.bounding_boxes(EntityKind.SOLID).bbox[0]
    assert box.tolist() == pytest.approx(
        [
            BOX_ORIGIN[0] - d,
            BOX_ORIGIN[1] - d,
            BOX_ORIGIN[2] - d,
            BOX_ORIGIN[0] + BOX_DX + d,
            BOX_ORIGIN[1] + BOX_DY + d,
            BOX_ORIGIN[2] + BOX_DZ + d,
        ],
        abs=1.0e-6,
    )


def test_offset_with_a_negative_distance_shrinks_the_box(
    placed_box_session: Session,
) -> None:
    solid = _sole(placed_box_session, EntityKind.SOLID)
    d = 0.5

    placed_box_session.offset([solid], -d)

    assert placed_box_session.entity_table(EntityKind.SOLID).measure[0] == pytest.approx(
        (BOX_DX - 2 * d) * (BOX_DY - 2 * d) * (BOX_DZ - 2 * d)
    )


def test_offset_of_a_solid_gives_a_solid_not_a_shell(
    placed_box_session: Session,
) -> None:
    # The property a boolean or a mesh on the offset body depends on: neither takes a loose
    # shell where a volume was meant.
    solid = _sole(placed_box_session, EntityKind.SOLID)

    placed_box_session.offset([solid], 0.5)

    assert _ids(placed_box_session, EntityKind.SOLID) == [solid]
    assert placed_box_session.entity_table(EntityKind.SOLID).measure[0] > 0.0


def test_offset_delta_keeps_every_id_and_creates_none(
    placed_box_session: Session,
) -> None:
    before = {k: _ids(placed_box_session, k) for k in EntityKind}
    everything = sorted(i for ids in before.values() for i in ids)
    solid = _sole(placed_box_session, EntityKind.SOLID)

    delta = placed_box_session.offset([solid], 0.5)

    assert delta.created.tolist() == []
    assert delta.deleted.tolist() == []
    assert delta.split.tolist() == []
    assert delta.merged.tolist() == []
    assert delta.valid is True
    # Every entity of the box was rebuilt at the new distance and kept its id: the solid,
    # all six faces, all twelve edges, all eight vertices.
    assert delta.modified.tolist() == everything
    assert {k: _ids(placed_box_session, k) for k in EntityKind} == before


def test_offset_of_a_shell_gives_a_shell(open_box_shell_brep: bytes) -> None:
    s = Session()
    s.add_brep(open_box_shell_brep)
    before = {k: _ids(s, k) for k in EntityKind}
    face = EntityId(_ids(s, EntityKind.FACE)[0])

    delta = s.offset([face], 0.1)

    assert before[EntityKind.SOLID] == []
    assert _ids(s, EntityKind.SOLID) == []
    assert _ids(s, EntityKind.FACE) == before[EntityKind.FACE]
    assert delta.created.tolist() == []
    assert delta.deleted.tolist() == []


def test_offset_on_a_dead_id_raises(placed_box_session: Session) -> None:
    edge = EntityId(_ids(placed_box_session, EntityKind.EDGE)[0])
    placed_box_session.fillet([edge], 0.5)
    assert not placed_box_session.is_alive(edge)

    with pytest.raises(ps.PysmeshError, match=f"entity {edge} is dead"):
        placed_box_session.offset([edge], 0.2)


def test_offset_across_two_bodies_raises(two_box_session: Session) -> None:
    solids = _ids(two_box_session, EntityKind.SOLID)

    with pytest.raises(ps.PysmeshError, match="belong to 2 different bodies"):
        two_box_session.offset([EntityId(solids[0]), EntityId(solids[1])], 0.2)


def test_offset_of_a_body_that_is_not_a_solid_or_shell_raises(
    placed_box_session: Session,
) -> None:
    delta = placed_box_session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, 3.0)
    face = EntityId(
        next(
            i
            for i in delta.created.tolist()
            if placed_box_session.entity_kind(EntityId(i)) == EntityKind.FACE
        )
    )

    with pytest.raises(ps.PysmeshError, match="FACE body; a uniform offset needs"):
        placed_box_session.offset([face], 0.2)


@pytest.mark.parametrize("distance", [0.0, float("nan"), float("-inf")])
def test_offset_with_a_zero_or_non_finite_distance_raises(
    placed_box_session: Session, distance: float
) -> None:
    solid = _sole(placed_box_session, EntityKind.SOLID)

    with pytest.raises(ps.PysmeshError, match="distance must be a finite non-zero"):
        placed_box_session.offset([solid], distance)


@pytest.mark.parametrize("tol", [0.0, -1.0e-7])
def test_offset_with_a_non_positive_tol_raises(
    placed_box_session: Session, tol: float
) -> None:
    solid = _sole(placed_box_session, EntityKind.SOLID)

    with pytest.raises(ps.PysmeshError, match="tol must be > 0"):
        placed_box_session.offset([solid], 0.5, tol=tol)


def test_offset_that_self_intersects_raises_naming_input_faces(
    placed_box_session: Session,
) -> None:
    solid = _sole(placed_box_session, EntityKind.SOLID)
    live = set(_ids(placed_box_session, EntityKind.FACE))

    # Half the smallest extent: the two faces normal to x meet and cross.
    with pytest.raises(ps.PysmeshError) as excinfo:
        placed_box_session.offset([solid], -BOX_DX / 2.0)

    blamed = set(excinfo.value.face_ids)
    assert blamed, "the failure must name the faces it broke on"
    assert blamed < live


def test_offset_leaves_the_session_unchanged_when_it_fails(
    placed_box_session: Session,
) -> None:
    solid = _sole(placed_box_session, EntityKind.SOLID)
    before = placed_box_session.brep()
    entities_before = {k: _ids(placed_box_session, k) for k in EntityKind}

    with pytest.raises(ps.PysmeshError):
        placed_box_session.offset([solid], -BOX_DX / 2.0)

    assert placed_box_session.brep() == before
    assert {k: _ids(placed_box_session, k) for k in EntityKind} == entities_before


# --------------------------------------------------------------------------------------- #
# An offset that comes back inside out, or the wrong way
#
# `offset` does not have the hollowing's collapse: over 99 committed offsets of a box, a
# cylinder, a cone, a sphere and a torus, from -1000.0 to +50.0, not one came back as the
# input body. Two other things did, and 4.1.1 committed both. Shrinking a body past its
# smallest radius of curvature turns it inside out — a sphere of radius 3 at -3.0 measures
# 0.0 and at -5.0 measures -33.51 — and a torus shrunk far enough comes back *larger* than
# it went in. The box's own limit, half its smallest extent, was already refused in 4.1.1
# and stays covered by the two tests above.
# --------------------------------------------------------------------------------------- #

SPHERE_RADIUS: float = 3.0
TORUS_RADII: tuple[float, float] = (5.0, 1.5)


def _sphere_session() -> Session:
    """A session holding one sphere of radius :data:`SPHERE_RADIUS`."""
    s = Session()
    s.add_sphere(SPHERE_RADIUS)
    return s


def _torus_session() -> Session:
    """A session holding one torus of radii :data:`TORUS_RADII`."""
    s = Session()
    s.add_torus(*TORUS_RADII)
    return s


def test_offset_shrinks_a_sphere_to_the_closed_form_up_to_its_radius() -> None:
    session = _sphere_session()
    solid = _sole(session, EntityKind.SOLID)
    distance = -2.5

    session.offset([solid], distance)

    # The regime the guard must leave alone: a sphere of radius 0.5 is still a sphere.
    left = SPHERE_RADIUS + distance
    assert session.entity_table(EntityKind.SOLID).measure[0] == pytest.approx(
        4.0 / 3.0 * math.pi * left**3
    )


@pytest.mark.parametrize("distance", [-3.0, -3.5, -5.0])
def test_offset_past_a_sphere_radius_raises_and_changes_nothing(distance: float) -> None:
    session = _sphere_session()
    solid = _sole(session, EntityKind.SOLID)
    before = _whole_state(session)

    # A sphere's smallest radius of curvature is its radius, so nothing is left at -3.0.
    # OCCT reports success anyway and hands back a sphere turned inside out: volume 0.0 at
    # -3.0, -0.523599 at -3.5, -33.510322 at -5.0, all of which 4.1.1 committed.
    with pytest.raises(ps.PysmeshError, match="turned inside out") as excinfo:
        session.offset([solid], distance)

    assert f"{distance:.6f}" in str(excinfo.value)
    assert set(excinfo.value.face_ids) == set(_ids(session, EntityKind.FACE))
    assert _whole_state(session) == before


@pytest.mark.parametrize("distance", [-1.5, -2.0])
def test_offset_past_a_torus_tube_radius_raises_and_changes_nothing(
    distance: float,
) -> None:
    session = _torus_session()
    solid = _sole(session, EntityKind.SOLID)
    before = _whole_state(session)

    with pytest.raises(ps.PysmeshError, match="turned inside out"):
        session.offset([solid], distance)

    assert _whole_state(session) == before


def test_offset_that_grows_a_body_it_was_told_to_shrink_raises() -> None:
    session = _torus_session()
    solid = _sole(session, EntityKind.SOLID)
    volume = float(session.entity_table(EntityKind.SOLID).measure[0])
    before = _whole_state(session)

    # Far enough past the tube radius the sign stops meaning anything: the torus comes back
    # at 8907.317972 against the 222.066099 it went in with, for a distance that shrinks.
    # This one has a positive volume, so only the direction statement sees it.
    with pytest.raises(ps.PysmeshError, match="though the distance shrinks it") as excinfo:
        session.offset([solid], -11.0)

    assert "turned inside out" not in str(excinfo.value)
    assert f"{volume:.6f}" in str(excinfo.value)
    assert _whole_state(session) == before


def test_a_cancelled_offset_changes_nothing(placed_box_session: Session) -> None:
    solid = _sole(placed_box_session, EntityKind.SOLID)
    before = (
        placed_box_session.op_count,
        placed_box_session.issued_id_count,
        {k: _ids(placed_box_session, k) for k in EntityKind},
        placed_box_session.brep(),
    )

    with pytest.raises(ps.PysmeshCancelled):
        placed_box_session.offset([solid], 0.5, cancel=lambda: True)

    assert (
        placed_box_session.op_count,
        placed_box_session.issued_id_count,
        {k: _ids(placed_box_session, k) for k in EntityKind},
        placed_box_session.brep(),
    ) == before


def test_offset_round_trips_through_a_snapshot(placed_box_session: Session) -> None:
    mark = placed_box_session.snapshot()
    before = placed_box_session.brep()
    entities_before = {k: _ids(placed_box_session, k) for k in EntityKind}
    solid = _sole(placed_box_session, EntityKind.SOLID)

    placed_box_session.offset([solid], 0.5)
    placed_box_session.restore(mark)

    assert placed_box_session.brep() == before
    assert {k: _ids(placed_box_session, k) for k in EntityKind} == entities_before


# --------------------------------------------------------------------------------------- #
# extract_edges
# --------------------------------------------------------------------------------------- #


def _outer_loop(session: Session, face: EntityId) -> list[EntityId]:
    """The edge ids of a face's outer wire, in traversal order."""
    wires = session.face_wires([face])
    assert bool(wires.ordered.all())
    start, end = wires.edge_range[0]
    return [EntityId(int(i)) for i in wires.edge_id[start:end].tolist()]


def _stray_edge(session: Session) -> tuple[EntityId, EntityId]:
    """An edge, and one that shares no vertex with it."""
    pairs = session.adjacency(EntityKind.EDGE, EntityKind.VERTEX)
    ends: dict[int, set[int]] = {}
    for e, v in zip(pairs.ids.tolist(), pairs.related.tolist(), strict=True):
        ends.setdefault(e, set()).add(v)
    edges = _ids(session, EntityKind.EDGE)
    first = edges[0]
    return EntityId(first), EntityId(
        next(e for e in edges if not ends[e] & ends[first])
    )


def test_extract_edges_copies_a_solids_face_loop_at_its_own_length(
    placed_box_session: Session,
) -> None:
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    loop = _outer_loop(placed_box_session, face)

    delta = placed_box_session.extract_edges(loop)

    table = placed_box_session.entity_table(EntityKind.EDGE)
    lengths = {int(i): float(m) for i, m in zip(table.ids, table.measure, strict=True)}
    copied = [
        i
        for i in delta.created.tolist()
        if placed_box_session.entity_kind(EntityId(i)) == EntityKind.EDGE
    ]
    # A box's first face is the one at x = xmin, so its boundary is the dy by dz rectangle.
    assert sorted(lengths[i] for i in copied) == pytest.approx(
        sorted([BOX_DY, BOX_DY, BOX_DZ, BOX_DZ])
    )
    assert sum(lengths[i] for i in copied) == pytest.approx(2.0 * (BOX_DY + BOX_DZ))


def test_extract_edges_delta_creates_the_new_body_and_touches_nothing_else(
    placed_box_session: Session,
) -> None:
    before = {k: _ids(placed_box_session, k) for k in EntityKind}
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    loop = _outer_loop(placed_box_session, face)
    high_water = placed_box_session.issued_id_count

    delta = placed_box_session.extract_edges(loop)

    assert delta.deleted.tolist() == []
    assert delta.modified.tolist() == []
    assert delta.split.tolist() == []
    assert delta.merged.tolist() == []
    # A closed four-edge loop copies to four edges and four vertices, and to nothing else.
    assert delta.created.tolist() == list(range(high_water + 1, high_water + 9))
    kinds = [placed_box_session.entity_kind(EntityId(i)) for i in delta.created.tolist()]
    assert kinds.count(EntityKind.EDGE) == 4
    assert kinds.count(EntityKind.VERTEX) == 4
    # Every original id, one by one, is still exactly where it was.
    new_edges = [
        i
        for i in delta.created.tolist()
        if placed_box_session.entity_kind(EntityId(i)) == EntityKind.EDGE
    ]
    assert _ids(placed_box_session, EntityKind.SOLID) == before[EntityKind.SOLID]
    assert _ids(placed_box_session, EntityKind.FACE) == before[EntityKind.FACE]
    assert _ids(placed_box_session, EntityKind.EDGE) == sorted(
        before[EntityKind.EDGE] + new_edges
    )
    for i in loop:
        assert placed_box_session.is_alive(i)


def test_extract_edges_leaves_the_original_edges_untouched(
    placed_box_session: Session,
) -> None:
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    loop = _outer_loop(placed_box_session, face)
    table = placed_box_session.entity_table(EntityKind.EDGE)
    truth = {int(i): float(m) for i, m in zip(table.ids, table.measure, strict=True)}

    placed_box_session.extract_edges(loop)

    after = placed_box_session.entity_table(EntityKind.EDGE)
    # Still there, one by one: a copy that consumed its source would leave this vacuous.
    assert set(truth) <= set(after.ids.tolist())
    for i, m in zip(after.ids, after.measure, strict=True):
        if int(i) in truth:
            assert m == pytest.approx(truth[int(i)]), f"edge {i} changed length"


def test_an_extracted_edge_can_be_the_spine_of_a_pipe(
    placed_box_session: Session,
) -> None:
    # The reason the operation exists: pipe refuses a spine that names a solid, so before
    # this nothing could sweep along an edge of an imported part.
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    edge = _outer_loop(placed_box_session, face)[0]
    spine = [
        EntityId(i)
        for i in placed_box_session.extract_edges([edge]).created.tolist()
        if placed_box_session.entity_kind(EntityId(i)) == EntityKind.EDGE
    ]
    profile = [
        EntityId(i)
        for i in placed_box_session.add_circle(
            BOX_ORIGIN, (0.0, 0.0, 1.0), 0.25
        ).created.tolist()
        if placed_box_session.entity_kind(EntityId(i)) == EntityKind.EDGE
    ]

    delta = placed_box_session.pipe(spine, profile)

    assert delta.created.size > 0
    assert delta.valid is True


def test_an_extracted_loop_caps_a_solids_face(placed_box_session: Session) -> None:
    # The other reason: make_face refuses a solid's edges, so a hole loop could not be
    # capped. A box's first face is at x = xmin, so its cap has area dy * dz.
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])
    loop = _outer_loop(placed_box_session, face)
    copied = [
        EntityId(i)
        for i in placed_box_session.extract_edges(loop).created.tolist()
        if placed_box_session.entity_kind(EntityId(i)) == EntityKind.EDGE
    ]

    delta = placed_box_session.make_face(copied)

    capped = EntityId(int(delta.created[0]))
    table = placed_box_session.entity_table(EntityKind.FACE)
    area = {int(i): float(m) for i, m in zip(table.ids, table.measure, strict=True)}
    assert area[capped] == pytest.approx(BOX_DY * BOX_DZ)


def test_extract_edges_with_an_empty_selection_raises(
    placed_box_session: Session,
) -> None:
    with pytest.raises(ps.PysmeshError, match="must name at least one edge"):
        placed_box_session.extract_edges([])


def test_extract_edges_on_a_dead_id_raises(placed_box_session: Session) -> None:
    edge = EntityId(_ids(placed_box_session, EntityKind.EDGE)[0])
    placed_box_session.fillet([edge], 0.5)
    assert not placed_box_session.is_alive(edge)

    with pytest.raises(ps.PysmeshError, match=f"entity {edge} is dead"):
        placed_box_session.extract_edges([edge])


def test_extract_edges_on_a_face_id_raises_naming_the_wrong_kind(
    placed_box_session: Session,
) -> None:
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])

    with pytest.raises(ps.PysmeshError, match=f"entity {face} is a FACE, not an EDGE"):
        placed_box_session.extract_edges([face])


def test_extract_edges_of_a_disconnected_selection_raises_naming_the_strays(
    placed_box_session: Session,
) -> None:
    first, stray = _stray_edge(placed_box_session)

    with pytest.raises(ps.PysmeshError, match="do not join the others") as excinfo:
        placed_box_session.extract_edges([first, stray])

    assert excinfo.value.face_ids == [stray]


def test_extract_edges_leaves_the_session_unchanged_when_it_fails(
    placed_box_session: Session,
) -> None:
    first, stray = _stray_edge(placed_box_session)
    before = placed_box_session.brep()
    entities_before = {k: _ids(placed_box_session, k) for k in EntityKind}
    issued_before = placed_box_session.issued_id_count

    with pytest.raises(ps.PysmeshError):
        placed_box_session.extract_edges([first, stray])

    assert placed_box_session.brep() == before
    assert {k: _ids(placed_box_session, k) for k in EntityKind} == entities_before
    assert placed_box_session.issued_id_count == issued_before


def test_extract_edges_round_trips_through_a_snapshot(
    placed_box_session: Session,
) -> None:
    mark = placed_box_session.snapshot()
    before = placed_box_session.brep()
    entities_before = {k: _ids(placed_box_session, k) for k in EntityKind}
    face = EntityId(_ids(placed_box_session, EntityKind.FACE)[0])

    placed_box_session.extract_edges(_outer_loop(placed_box_session, face))
    placed_box_session.restore(mark)

    assert placed_box_session.brep() == before
    assert {k: _ids(placed_box_session, k) for k in EntityKind} == entities_before


# --------------------------------------------------------------------------------------- #
# Snapshot and restore
# --------------------------------------------------------------------------------------- #


def test_restore_rewinds_both_the_shape_and_the_registry(box_session: Session) -> None:
    mark = box_session.snapshot()
    entities_before = {k: _ids(box_session, k) for k in EntityKind}
    brep_before = box_session.brep()

    box_session.add_cylinder(1.0, 4.0, origin=(30.0, 0.0, 0.0))
    box_session.translate((7.0, 0.0, 0.0))
    box_session.restore(mark)

    assert {k: _ids(box_session, k) for k in EntityKind} == entities_before
    assert box_session.brep() == brep_before


def test_a_snapshot_mark_can_be_restored_more_than_once(box_session: Session) -> None:
    mark = box_session.snapshot()
    expected = box_session.brep()

    for _ in range(3):
        box_session.add_box(1.0, 1.0, 1.0, origin=(40.0, 0.0, 0.0))
        box_session.restore(mark)

    assert box_session.brep() == expected


def test_restore_does_not_rewind_the_id_counter(box_session: Session) -> None:
    mark = box_session.snapshot()
    box_session.add_box(1.0, 1.0, 1.0, origin=(40.0, 0.0, 0.0))
    abandoned = set(box_session.entities(EntityKind.SOLID).tolist())
    high_water = box_session.issued_id_count

    box_session.restore(mark)
    delta = box_session.add_cylinder(1.0, 2.0, origin=(60.0, 0.0, 0.0))

    # Every id the new branch issues is above everything the abandoned branch used, so a
    # reference held from that branch can never come back meaning something else.
    assert delta.created.min() > high_water
    assert abandoned.isdisjoint(set(delta.created.tolist()))


def test_an_id_from_an_abandoned_branch_reports_dead(box_session: Session) -> None:
    mark = box_session.snapshot()
    delta = box_session.add_box(1.0, 1.0, 1.0, origin=(40.0, 0.0, 0.0))
    orphan = EntityId(int(delta.created[0]))

    box_session.restore(mark)

    assert not box_session.is_alive(orphan)


def test_discard_snapshot_makes_the_mark_unusable(box_session: Session) -> None:
    mark = box_session.snapshot()

    box_session.discard_snapshot(mark)

    assert box_session.snapshot_count == 0
    with pytest.raises(ps.PysmeshError, match="discarded"):
        box_session.restore(mark)


def test_restore_of_an_unknown_mark_raises(box_session: Session) -> None:
    with pytest.raises(ps.PysmeshError, match="unknown snapshot mark"):
        box_session.restore(ps.SnapshotMark(99))


def test_state_op_index_tracks_the_restored_state_not_the_op_count(
    box_session: Session,
) -> None:
    mark = box_session.snapshot()
    at_snapshot = box_session.state_op_index

    box_session.add_box(1.0, 1.0, 1.0, origin=(40.0, 0.0, 0.0))
    box_session.translate((1.0, 0.0, 0.0))
    box_session.restore(mark)

    assert box_session.state_op_index == at_snapshot
    assert box_session.op_count > at_snapshot


# --------------------------------------------------------------------------------------- #
# Persistent naming
# --------------------------------------------------------------------------------------- #


def test_a_name_survives_an_operation_that_only_moves_the_entity(
    box_session: Session,
) -> None:
    face = EntityId(_ids(box_session, EntityKind.FACE)[0])
    name = box_session.name_of(face)

    box_session.translate((100.0, 0.0, 0.0))
    box_session.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.7)

    resolved = box_session.resolve(name)
    assert resolved.status is ResolutionStatus.RESOLVED
    assert resolved.ids == (face,)


def test_a_name_survives_a_boolean_that_keeps_the_entity(
    two_box_session: Session,
) -> None:
    faces = two_box_session.entity_table(EntityKind.FACE)
    # Pick a face far from the seam, so the boolean certainly keeps it.
    keep = EntityId(int(faces.ids[int(np.argmin(faces.centroid[:, 0]))]))
    name = two_box_session.name_of(keep)
    solids = _ids(two_box_session, EntityKind.SOLID)

    two_box_session.fuse([EntityId(solids[0])], [EntityId(solids[1])])

    resolved = two_box_session.resolve(name)
    assert resolved.status is ResolutionStatus.RESOLVED
    assert resolved.ids == (keep,)


def test_the_name_of_a_deleted_entity_resolves_lost(two_box_session: Session) -> None:
    faces = two_box_session.entity_table(EntityKind.FACE)
    seam = [
        EntityId(int(i))
        for i, c in zip(faces.ids, faces.centroid, strict=True)
        if c[0] == pytest.approx(BOX_DX)
    ]
    names = [two_box_session.name_of(i) for i in seam]
    solids = _ids(two_box_session, EntityKind.SOLID)

    two_box_session.fuse([EntityId(solids[0])], [EntityId(solids[1])])

    for name in names:
        resolved = two_box_session.resolve(name)
        assert resolved.status is ResolutionStatus.LOST
        assert resolved.ids == ()


def test_the_name_of_a_split_entity_resolves_ambiguous() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(2.0, 9.0, 3.0, origin=(2.0, -1.0, 4.0))
    solids = _ids(s, EntityKind.SOLID)

    delta = s.fuse([EntityId(solids[0])], [EntityId(solids[1])])

    split = [i for i in delta.split.tolist() if s.entity_kind(EntityId(i)) == EntityKind.FACE]
    assert split
    for i in split:
        resolved = s.resolve(s.name_of(EntityId(i)))
        assert resolved.status is ResolutionStatus.AMBIGUOUS
        assert resolved.shape_count == 2


def test_a_name_minted_on_an_abandoned_branch_resolves_lost(box_session: Session) -> None:
    mark = box_session.snapshot()
    delta = box_session.add_box(1.0, 1.0, 1.0, origin=(40.0, 0.0, 0.0))
    name = box_session.name_of(EntityId(int(delta.created[0])))

    box_session.restore(mark)

    assert box_session.resolve(name).status is ResolutionStatus.LOST


def test_resolving_a_name_that_was_never_minted_raises(box_session: Session) -> None:
    never = ps.Name(op_index=999, role=NameRole.GENERATED, ordinal=7)

    with pytest.raises(ps.PysmeshError, match="no entity was ever named"):
        box_session.resolve(never)


def test_two_geometrically_identical_boxes_get_distinct_names() -> None:
    s = Session()
    a = s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    b = s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(20.0, 0.0, 0.0))

    names_a = {s.name_of(EntityId(int(i))) for i in a.created}
    names_b = {s.name_of(EntityId(int(i))) for i in b.created}

    # Same geometry, different provenance. A fingerprint-based scheme would collide here.
    assert names_a.isdisjoint(names_b)


def test_a_name_records_provenance_not_position(box_session: Session) -> None:
    face = EntityId(_ids(box_session, EntityKind.FACE)[0])
    before = box_session.name_of(face)

    box_session.add_box(1.0, 1.0, 1.0, origin=(40.0, 0.0, 0.0))
    box_session.translate((13.0, 0.0, 0.0))

    assert box_session.name_of(face) == before


def test_a_filleted_faces_origin_names_the_edge_it_came_from(box_session: Session) -> None:
    edge = EntityId(_ids(box_session, EntityKind.EDGE)[0])

    delta = box_session.fillet([edge], 0.5)

    generated = [
        EntityId(int(i))
        for i in delta.created
        if box_session.origin(EntityId(int(i))).role is NameRole.GENERATED
    ]
    assert generated
    sources = {int(x) for i in generated for x in box_session.origin(i).sources}
    assert edge in sources


def test_origin_answers_for_a_dead_id_while_name_of_does_not(
    two_box_session: Session,
) -> None:
    solids = _ids(two_box_session, EntityKind.SOLID)
    delta = two_box_session.fuse([EntityId(solids[0])], [EntityId(solids[1])])
    dead = EntityId(int(delta.deleted[0]))

    assert two_box_session.origin(dead).op_index >= 1
    with pytest.raises(ps.PysmeshError, match="is dead"):
        two_box_session.name_of(dead)


# --------------------------------------------------------------------------------------- #
# Session independence and the stateless API
# --------------------------------------------------------------------------------------- #


def test_two_sessions_in_one_process_do_not_share_state() -> None:
    a = Session()
    b = Session()

    a.add_box(BOX_DX, BOX_DY, BOX_DZ)

    assert a.entity_count == BOX_ENTITY_COUNT
    assert b.entity_count == 0
    assert b.issued_id_count == 0


def test_two_sessions_issue_ids_from_independent_counters() -> None:
    a = Session()
    b = Session()

    delta_a = a.add_box(BOX_DX, BOX_DY, BOX_DZ)
    delta_b = b.add_box(BOX_DX, BOX_DY, BOX_DZ)

    # Identical models, identical ids: the counters are per-session, not global.
    assert delta_a.created.tolist() == list(range(1, BOX_ENTITY_COUNT + 1))
    assert delta_b.created.tolist() == delta_a.created.tolist()


def test_interleaved_operations_on_two_sessions_stay_independent() -> None:
    a = Session()
    b = Session()

    a.add_box(BOX_DX, BOX_DY, BOX_DZ)
    b.add_cylinder(2.0, 5.0)
    a.translate((100.0, 0.0, 0.0))
    b.translate((0.0, 0.0, -7.0))
    a.add_box(1.0, 1.0, 1.0, origin=(0.0, 0.0, 0.0))

    assert a.entity_table(EntityKind.SOLID).measure.tolist() == pytest.approx(
        [BOX_VOLUME, 1.0]
    )
    assert b.entity_table(EntityKind.SOLID).measure.tolist() == pytest.approx(
        [math.pi * 4.0 * 5.0]
    )
    assert b.entity_count == 1 + 3 + 3 + 2  # cylinder: solid, 3 faces, 3 edges, 2 vertices


def test_a_session_per_thread_produces_the_same_result_as_sequential_use() -> None:
    # The documented contract is one session per thread, not one session per process.
    results: dict[int, float] = {}

    def build(slot: int, height: float) -> None:
        s = Session()
        s.add_cylinder(2.0, height)
        results[slot] = float(s.entity_table(EntityKind.SOLID).measure[0])

    threads = [threading.Thread(target=build, args=(i, 3.0 + i)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == pytest.approx(
        {i: math.pi * 4.0 * (3.0 + i) for i in range(4)}
    )


def test_the_stateless_free_functions_still_work_alongside_a_session(
    box_brep: bytes,
) -> None:
    s = Session()
    s.add_brep(box_brep)

    shape = ps.load_brep(box_brep)
    faces = shape.faces()

    # Positional ordinals from the stateless API remain 1-based and per-kind, unchanged.
    assert [f.id for f in faces] == list(range(1, len(faces) + 1))
    assert len(_ids(s, EntityKind.FACE)) == len(faces)


def test_session_brep_round_trips_through_the_stateless_loader(
    box_session: Session,
) -> None:
    shape = ps.load_brep(box_session.brep())

    assert len(shape.faces()) == len(_ids(box_session, EntityKind.FACE))
    assert shape.solids()[0].volume == pytest.approx(BOX_VOLUME)


def test_a_non_integer_entity_id_is_rejected(box_session: Session) -> None:
    with pytest.raises(ps.PysmeshError, match="must be integers"):
        box_session.fuse([EntityId(1)], ["not an id"])  # type: ignore[list-item]
