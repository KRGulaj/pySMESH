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

# A box carries 1 solid + 6 faces + 12 edges + 8 vertices.
BOX_ENTITY_COUNT: int = 27


@pytest.fixture
def box_session() -> Session:
    """A session holding one 3 x 7 x 11 box at the origin."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
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
