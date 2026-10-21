# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for the checks: check_boolean, check_shape, small_edges, is_closed_shell and
tolerances.

Each check is read-only: it issues no id, does not advance the operation counter and
leaves the session's bytes unchanged, which every test of this file that names a check
asserts at least once. The answers are checked against shapes whose verdict is known:

* two boxes that overlap are valid operands of every boolean, so the analyzer finds
  nothing; a face fused with a solid mixes dimensions, which ``BOPAlgo_ArgumentAnalyzer``
  reports as ``BOPAlgo_BadType``; two overlapping boxes taken together as the arguments
  interfere with each other, which it reports as ``BOPAlgo_SelfIntersect``;
* a box is valid; a torus of radius ``1e-12`` is degenerate and ``BRepCheck_Analyzer``
  rejects it (the fixture of ``test_session_heal.py``);
* the box ``[0, 3] x [0, 7] x [0, 11]`` has four edges of each length 3, 7 and 11, and a
  primitive's sub-shapes carry OCCT's default tolerance ``Precision::Confusion() = 1e-7``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, PysmeshCancelled, Session

BOX_A: float = 3.0
BOX_B: float = 7.0
BOX_C: float = 11.0
DEFAULT_TOLERANCE: float = 1e-7

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


def state(s: Session) -> tuple[bytes, tuple[tuple[int, ...], ...], int, int]:
    """The bytes, the live ids and the two counters: everything a query must keep."""
    ids = tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)
    return s.brep(), ids, s.op_count, s.issued_id_count


def two_overlapping_boxes() -> tuple[Session, EntityId, EntityId]:
    """The 3 x 7 x 11 box and a copy shifted by 1.5 along x."""
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    s.add_box(BOX_A, BOX_B, BOX_C, origin=(1.5, 0.0, 0.0))
    a, b = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))
    return s, a, b


def degenerate_torus_session() -> Session:
    """An unvalidated session holding a torus that BRepCheck_Analyzer rejects."""
    s = Session(validate=False)
    s.add_torus(1e-12, 1e-13)
    return s


# ---- check_boolean ------------------------------------------------------------------- #


@pytest.mark.parametrize("operation", ["fuse", "cut", "common", "section", "split"])
def test_check_boolean_finds_nothing_on_two_overlapping_boxes(operation: str) -> None:
    s, a, b = two_overlapping_boxes()
    before = state(s)

    findings = s.check_boolean(operation, [a], [b])  # type: ignore[arg-type]

    assert findings == ()
    assert state(s) == before


def test_check_boolean_reports_a_face_fused_with_a_solid_as_a_bad_type() -> None:
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    s.add_rectangle((-1.0, -1.0, 5.0), (0.0, 0.0, 1.0), 6.0, 10.0)
    solid = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    face_ids = {int(i) for i in s.entities(EntityKind.FACE)}
    box_faces = {int(i) for i in s.adjacency(EntityKind.SOLID, EntityKind.FACE).related}
    (free,) = face_ids - box_faces

    findings = s.check_boolean("fuse", [EntityId(free)], [solid])

    assert [f.check for f in findings] == ["BOPAlgo_BadType"]
    assert findings[0].argument_ids.tolist() == [free]
    assert findings[0].tool_ids.tolist() == [int(solid)]


def test_check_boolean_reports_two_interfering_arguments_as_a_self_intersection() -> (
    None
):
    s, a, b = two_overlapping_boxes()
    s.add_box(1.0, 1.0, 1.0, origin=(20.0, 0.0, 0.0))
    tool = EntityId(int(s.entities(EntityKind.SOLID)[-1]))

    findings = s.check_boolean("fuse", [a, b], [tool])

    checks = {f.check for f in findings}
    assert "BOPAlgo_SelfIntersect" in checks
    for f in findings:
        if f.check == "BOPAlgo_SelfIntersect":
            assert f.argument_ids.size > 0
            assert f.tool_ids.size == 0
            assert all(s.is_alive(EntityId(int(i))) for i in f.argument_ids)


def test_check_boolean_refuses_an_unknown_operation() -> None:
    s, a, b = two_overlapping_boxes()

    with pytest.raises(
        ps.PysmeshError, match=r"check_boolean: the operation is 'merge'"
    ):
        s.check_boolean("merge", [a], [b])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("which", "message"),
    [("arguments", "no argument is named"), ("tools", "no tool is named")],
)
def test_check_boolean_refuses_an_empty_operand_list(which: str, message: str) -> None:
    s, a, b = two_overlapping_boxes()
    arguments = [] if which == "arguments" else [a]
    tools = [] if which == "tools" else [b]

    with pytest.raises(ps.PysmeshError, match=rf"check_boolean: {message}"):
        s.check_boolean("cut", arguments, tools)


def test_check_boolean_refuses_a_dead_id_naming_it() -> None:
    s, a, b = two_overlapping_boxes()
    s.remove([b])

    with pytest.raises(
        ps.PysmeshError, match=rf"check_boolean: entity {int(b)} is dead"
    ):
        s.check_boolean("cut", [a], [b])


def test_a_cancelled_check_boolean_raises_and_leaves_the_session_as_it_was() -> None:
    s, a, b = two_overlapping_boxes()
    before = state(s)

    with pytest.raises(PysmeshCancelled):
        s.check_boolean("cut", [a], [b], cancel=lambda: True)

    assert state(s) == before


# ---- check_shape --------------------------------------------------------------------- #


def test_check_shape_finds_nothing_on_a_box() -> None:
    s, _, _ = two_overlapping_boxes()
    before = state(s)

    findings = s.check_shape()

    assert findings == ()
    assert state(s) == before


def test_check_shape_reports_the_degenerate_torus_with_its_statuses() -> None:
    s = degenerate_torus_session()
    before = state(s)

    findings = s.check_shape()

    assert len(findings) > 0
    for f in findings:
        assert s.is_alive(f.entity_id)
        assert f.kind in ("SOLID", "SHELL", "FACE", "WIRE", "EDGE", "VERTEX")
        assert len(f.statuses) > 0
        assert all(name.startswith("BRepCheck_") for name in f.statuses)
        assert "BRepCheck_NoError" not in f.statuses
    assert state(s) == before


def test_check_shape_checks_only_the_bodies_of_the_named_entities() -> None:
    s = degenerate_torus_session()
    s.add_box(BOX_A, BOX_B, BOX_C, origin=(10.0, 0.0, 0.0))
    box = EntityId(int(s.entities(EntityKind.SOLID)[-1]))

    assert s.check_shape([box]) == ()
    assert len(s.check_shape()) > 0


def test_check_shape_refuses_an_empty_entity_list() -> None:
    s, _, _ = two_overlapping_boxes()

    with pytest.raises(ps.PysmeshError, match=r"check_shape: at least one entity"):
        s.check_shape([])


def test_a_cancelled_check_shape_raises_and_leaves_the_session_as_it_was() -> None:
    s = degenerate_torus_session()
    before = state(s)

    with pytest.raises(PysmeshCancelled):
        s.check_shape(cancel=lambda: True)

    assert state(s) == before


# ---- small_edges --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("threshold", "length"),
    [(5.0, BOX_A), (9.0, BOX_B), (12.0, BOX_C)],
)
def test_small_edges_lists_every_box_edge_shorter_than_the_threshold(
    threshold: float, length: float
) -> None:
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    table = s.entity_table(EntityKind.EDGE)
    before = state(s)

    short = s.small_edges(threshold)

    want = sorted(
        int(i) for i, m in zip(table.ids, table.measure, strict=True) if m < threshold
    )
    assert short.tolist() == want
    assert len(want) == 4 * (1 + [BOX_A, BOX_B, BOX_C].index(length))
    assert state(s) == before


def test_small_edges_is_strict_at_an_edge_length() -> None:
    """An edge whose length equals the threshold is not listed; one ulp more lists it.

    The length is the rule's own: the adaptive rule of mass_properties reads a straight
    edge of 3 within round-off of 3, not necessarily as 3.0 exactly.
    """
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    table = s.entity_table(EntityKind.EDGE)
    shortest = float(table.measure.min())
    assert shortest == pytest.approx(BOX_A, rel=1e-15)

    assert s.small_edges(shortest).size == 0
    assert s.small_edges(math.nextafter(shortest, math.inf)).size == int(
        np.count_nonzero(table.measure == shortest)
    )


@pytest.mark.parametrize("threshold", [0.0, -1.0])
def test_small_edges_refuses_a_threshold_that_is_no_length(threshold: float) -> None:
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)

    with pytest.raises(ps.PysmeshError, match=r"small_edges: the threshold is"):
        s.small_edges(threshold)


def test_small_edges_refuses_a_threshold_that_is_not_finite() -> None:
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)

    with pytest.raises(ps.PysmeshError, match=r"small_edges"):
        s.small_edges(math.nan)


# ---- is_closed_shell ----------------------------------------------------------------- #


def test_is_closed_shell_answers_per_entity_for_its_body(
    open_box_shell_brep: bytes,
) -> None:
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    box_face = EntityId(int(s.entities(EntityKind.FACE)[0]))
    box = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    s.add_rectangle((20.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, 3.0)
    rectangle = EntityId(int(s.entities(EntityKind.FACE)[-1]))
    before_faces = {int(i) for i in s.entities(EntityKind.FACE)}
    s.add_brep(open_box_shell_brep)
    open_face = EntityId(
        min({int(i) for i in s.entities(EntityKind.FACE)} - before_faces)
    )
    before = state(s)

    closed = s.is_closed_shell([box, box_face, rectangle, open_face])

    assert closed.tolist() == [True, True, False, False]
    assert state(s) == before


def test_is_closed_shell_refuses_an_empty_entity_list() -> None:
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)

    with pytest.raises(ps.PysmeshError, match=r"is_closed_shell: no entity is named"):
        s.is_closed_shell([])


# ---- tolerances ---------------------------------------------------------------------- #


def test_tolerances_of_a_primitive_are_occts_default() -> None:
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    ids = [
        EntityId(int(i))
        for kind in (EntityKind.FACE, EntityKind.EDGE, EntityKind.VERTEX)
        for i in s.entities(kind)
    ]
    before = state(s)

    tolerances = s.tolerances(ids)

    np.testing.assert_array_equal(tolerances, np.full(len(ids), DEFAULT_TOLERANCE))
    assert state(s) == before


def test_tolerances_refuses_a_solid_naming_it() -> None:
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    box = EntityId(int(s.entities(EntityKind.SOLID)[0]))

    with pytest.raises(
        ps.PysmeshError,
        match=rf"tolerances: entity {int(box)} is a SOLID, which stores",
    ):
        s.tolerances([box])
