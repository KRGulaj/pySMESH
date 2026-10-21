# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for limit_tolerance and change_orientation.

``limit_tolerance`` lowers every tolerance above a limit to the limit and leaves the
others, so after it each tolerance is ``min(before, limit)``; the geometry does not
change, so the box ``[0, 3] x [0, 7] x [0, 11]`` keeps its volume 231. The fixture is that
box read from a BREP whose every tolerance was rewritten from ``1e-07`` to ``0.001``.

``change_orientation`` reverses faces. A face's outward normal is its surface normal,
reversed on a reversed face, so reversing a face negates the normal that
``surface_at`` reports at every parameter, exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, Session

BOX_A: float = 3.0
BOX_B: float = 7.0
BOX_C: float = 11.0
BOX_VOLUME: float = BOX_A * BOX_B * BOX_C
LOOSE: float = 1e-3
LIMIT: float = 1e-5
EXACT_RTOL: float = 1e-12

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, per kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def sub_ids(s: Session) -> list[EntityId]:
    """Every live face, edge and vertex id: the sub-shapes that store a tolerance."""
    return [
        EntityId(int(i))
        for kind in (EntityKind.FACE, EntityKind.EDGE, EntityKind.VERTEX)
        for i in s.entities(kind)
    ]


def loose_box_session() -> tuple[Session, EntityId]:
    """The 3 x 7 x 11 box with every tolerance 1e-3, read from rewritten BREP bytes."""
    template = Session()
    template.add_box(BOX_A, BOX_B, BOX_C)
    text = template.brep().decode("ascii")
    assert "1e-07" in text
    s = Session()
    s.add_brep(text.replace("1e-07", "0.001").encode("ascii"))
    return s, EntityId(int(s.entities(EntityKind.SOLID)[0]))


# ---- limit_tolerance ----------------------------------------------------------------- #


def test_the_fixture_box_carries_the_loose_tolerance() -> None:
    s, _ = loose_box_session()

    np.testing.assert_array_equal(s.tolerances(sub_ids(s)), LOOSE)


def test_limit_tolerance_lowers_every_tolerance_above_the_limit_to_it() -> None:
    s, box = loose_box_session()

    delta = s.limit_tolerance([box], LIMIT)

    np.testing.assert_array_equal(s.tolerances(sub_ids(s)), LIMIT)
    assert delta.valid is True
    volume = s.mass_properties([box]).measure[0]
    assert volume == pytest.approx(BOX_VOLUME, rel=EXACT_RTOL)


def test_limit_tolerance_keeps_every_id_on_the_same_sub_shape() -> None:
    """History against geometry: each id keeps its measure and centroid."""
    s, box = loose_box_session()
    before = id_space(s)
    tables = {k: s.entity_table(k) for k in _KINDS}

    delta = s.limit_tolerance([box], LIMIT)

    assert id_space(s) == before
    assert delta.created.size == 0
    assert delta.deleted.size == 0
    assert sorted(delta.modified.tolist()) == sorted(i for ids in before for i in ids)
    for kind, table in tables.items():
        after = s.entity_table(kind)
        np.testing.assert_array_equal(after.ids, table.ids)
        np.testing.assert_allclose(after.measure, table.measure, rtol=EXACT_RTOL)
        np.testing.assert_allclose(
            after.centroid, table.centroid, rtol=EXACT_RTOL, atol=1e-12
        )


def test_limit_tolerance_leaves_a_snapshot_its_own_tolerances() -> None:
    s, box = loose_box_session()
    mark = s.snapshot()

    s.limit_tolerance([box], LIMIT)
    s.restore(mark)

    np.testing.assert_array_equal(s.tolerances(sub_ids(s)), LOOSE)


def test_limit_tolerance_leaves_tolerances_below_the_limit() -> None:
    """Loose vertices (1e-3) and tight edges and faces (1e-7): only the vertices change."""
    template = Session()
    template.add_box(BOX_A, BOX_B, BOX_C)
    text = template.brep().decode("ascii")
    tight_vertex = "Ve\n1e-07"
    assert tight_vertex in text
    s = Session()
    s.add_brep(text.replace(tight_vertex, "Ve\n0.001").encode("ascii"))
    box = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    vertices = [EntityId(int(i)) for i in s.entities(EntityKind.VERTEX)]
    others = [
        EntityId(int(i))
        for kind in (EntityKind.FACE, EntityKind.EDGE)
        for i in s.entities(kind)
    ]
    np.testing.assert_array_equal(s.tolerances(vertices), LOOSE)

    s.limit_tolerance([box], LIMIT)

    np.testing.assert_array_equal(s.tolerances(vertices), LIMIT)
    np.testing.assert_array_equal(s.tolerances(others), 1e-7)


def test_limit_tolerance_refuses_when_nothing_is_above_the_limit() -> None:
    s, box = loose_box_session()
    before = (s.brep(), id_space(s), s.op_count)

    with pytest.raises(
        ps.PysmeshError,
        match=r"limit_tolerance: no tolerance in the bodies of the named entities is above "
        r"0\.002 \(the largest is 0\.001\); nothing to limit",
    ):
        s.limit_tolerance([box], 2.0 * LOOSE)

    assert (s.brep(), id_space(s), s.op_count) == before


@pytest.mark.parametrize("tolerance", [1e-8, 0.0, -1.0])
def test_limit_tolerance_refuses_a_limit_below_occts_smallest_length(
    tolerance: float,
) -> None:
    s, box = loose_box_session()

    with pytest.raises(
        ps.PysmeshError, match=r"limit_tolerance: the tolerance .* is below"
    ):
        s.limit_tolerance([box], tolerance)


def test_limit_tolerance_refuses_an_empty_entity_list() -> None:
    s, _ = loose_box_session()

    with pytest.raises(ps.PysmeshError, match=r"limit_tolerance: no entity is named"):
        s.limit_tolerance([], LIMIT)


def test_limit_tolerance_refuses_a_dead_id() -> None:
    s, box = loose_box_session()
    s.remove([box])

    with pytest.raises(
        ps.PysmeshError, match=rf"limit_tolerance: entity {int(box)} is dead"
    ):
        s.limit_tolerance([box], LIMIT)


def _limited() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
    s, box = loose_box_session()
    s.limit_tolerance([box], LIMIT)
    return s.brep(), id_space(s)


def test_limit_tolerance_gives_identical_bytes_and_ids_in_two_sessions() -> None:
    first = _limited()

    second = _limited()

    assert first == second


# ---- change_orientation -------------------------------------------------------------- #


def rectangle_session() -> tuple[Session, EntityId]:
    """A free planar rectangle on z = 0, normal +z."""
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), BOX_A, BOX_B)
    return s, EntityId(int(s.entities(EntityKind.FACE)[0]))


def centre_normal(s: Session, face: EntityId) -> np.ndarray:
    """The outward normal at the middle of the face's parameter range."""
    u0, u1, v0, v1 = s.face_parameter_bounds([face])[0]
    return np.asarray(
        s.surface_at(face, [(0.5 * (u0 + u1), 0.5 * (v0 + v1))]).normals[0]
    )


def test_change_orientation_of_a_free_face_negates_its_normal() -> None:
    s, face = rectangle_session()
    before = centre_normal(s, face)
    np.testing.assert_allclose(before, (0.0, 0.0, 1.0), atol=1e-15)

    s.change_orientation([face])

    np.testing.assert_array_equal(centre_normal(s, face), -before)


def test_change_orientation_keeps_every_id_and_every_measure() -> None:
    s, face = rectangle_session()
    before = id_space(s)
    table = s.entity_table(EntityKind.FACE)

    delta = s.change_orientation([face])

    assert id_space(s) == before
    assert delta.created.size == delta.deleted.size == delta.modified.size == 0
    after = s.entity_table(EntityKind.FACE)
    np.testing.assert_allclose(after.measure, table.measure, rtol=EXACT_RTOL)
    np.testing.assert_allclose(after.centroid, table.centroid, rtol=EXACT_RTOL)


def test_change_orientation_twice_restores_the_normal() -> None:
    s, face = rectangle_session()
    before = centre_normal(s, face)

    s.change_orientation([face])
    s.change_orientation([face])

    np.testing.assert_array_equal(centre_normal(s, face), before)


def test_change_orientation_of_every_face_of_a_shell_turns_the_shell_round(
    open_box_shell_brep: bytes,
) -> None:
    s = Session()
    s.add_brep(open_box_shell_brep)
    faces = [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    before = {int(f): centre_normal(s, f) for f in faces}

    delta = s.change_orientation(faces)

    assert delta.valid is True
    for f in faces:
        np.testing.assert_array_equal(centre_normal(s, f), -before[int(f)])


def test_change_orientation_refuses_a_face_of_a_solid_naming_face_and_solid() -> None:
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    box = int(s.entities(EntityKind.SOLID)[0])
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    with pytest.raises(
        ps.PysmeshError,
        match=rf"change_orientation: face {int(face)} bounds solid {box}; reversed, it "
        "would turn the solid inside out",
    ):
        s.change_orientation([face])


def test_change_orientation_refuses_an_empty_face_list() -> None:
    s, _ = rectangle_session()

    with pytest.raises(ps.PysmeshError, match=r"change_orientation: no face is named"):
        s.change_orientation([])


def test_change_orientation_refuses_an_edge() -> None:
    s, _ = rectangle_session()
    edge = EntityId(int(s.entities(EntityKind.EDGE)[0]))

    with pytest.raises(
        ps.PysmeshError, match=rf"change_orientation: entity {int(edge)} is"
    ):
        s.change_orientation([edge])


def _reversed() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
    s, face = rectangle_session()
    s.change_orientation([face])
    return s.brep(), id_space(s)


def test_change_orientation_gives_identical_bytes_and_ids_in_two_sessions() -> None:
    first = _reversed()

    second = _reversed()

    assert first == second


def test_change_orientation_of_one_face_of_a_shell_reverses_that_face_only(
    open_box_shell_brep: bytes,
) -> None:
    s = Session()
    s.add_brep(open_box_shell_brep)
    faces = [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    before = {int(f): centre_normal(s, f) for f in faces}

    s.change_orientation([faces[0]])

    np.testing.assert_array_equal(centre_normal(s, faces[0]), -before[int(faces[0])])
    for f in faces[1:]:
        np.testing.assert_array_equal(centre_normal(s, f), before[int(f)])
