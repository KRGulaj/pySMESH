# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for the block checks and searches: check_blocks, propagation_chain, opposite_face,
face_by_edges, find_block_near.

The answers are counts and positions on a ``2 x 3`` grid of glued unit boxes
(``x in [0, 2]``, ``y in [0, 3]``, ``z in [0, 1]``):

* the chain of an edge parallel to x in the column ``x in [0, 1]`` holds the x-edges of that
  column, one for each of the 4 values of y and 2 of z: 8 edges;
* the chain of a y-edge in the row ``y in [0, 1]``: 3 values of x and 2 of z, 6 edges;
* the chain of a z-edge: every z-edge, 3 values of x and 4 of y, 12 edges;
* every member of a chain is parallel to its edge (the line's direction is the axis).

Every query reads ``op_count`` and ``issued_id_count`` before and after: a query changes
neither.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import BlocksErrorKind, EntityId, EntityKind, Session

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


def at(s: Session, kind: EntityKind, point: tuple[float, float, float]) -> EntityId:
    """The label of the one shape of a kind whose centroid is the point."""
    table = s.entity_table(kind)
    near = np.linalg.norm(table.centroid - np.asarray(point), axis=1) < 1e-9
    labels = {int(i) for i in table.ids[near]}
    distinct = {int(i) for i in s.entities(kind, distinct=True)}
    (label,) = labels & distinct
    return EntityId(label)


def grid() -> Session:
    """A 2 x 3 grid of glued unit boxes."""
    s = Session()
    for i in range(2):
        for j in range(3):
            s.add_box(1.0, 1.0, 1.0, origin=(float(i), float(j), 0.0))
    s.glue([EntityId(int(x)) for x in s.entities(EntityKind.SOLID)], 1e-7)
    return s


def two_boxes(*, glued: bool) -> Session:
    """Two unit boxes that touch along x = 1, glued or not."""
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    s.add_box(1.0, 1.0, 1.0, origin=(1.0, 0.0, 0.0))
    if glued:
        s.glue([EntityId(int(x)) for x in s.entities(EntityKind.SOLID)], 1e-7)
    return s


# ---- check_blocks ---------------------------------------------------------------------- #


def test_two_glued_boxes_are_a_compound_of_blocks() -> None:
    s = two_boxes(glued=True)
    before = (s.op_count, s.issued_id_count)

    check = s.check_blocks()

    assert check.is_compound_of_blocks
    assert check.findings == ()
    assert (s.op_count, s.issued_id_count) == before


def test_two_boxes_that_only_touch_are_not_glued() -> None:
    s = two_boxes(glued=False)
    solids = sorted(int(x) for x in s.entities(EntityKind.SOLID))
    before = (s.op_count, s.issued_id_count)

    check = s.check_blocks()

    assert not check.is_compound_of_blocks
    (finding,) = check.findings
    assert finding.kind == BlocksErrorKind.NOT_GLUED
    assert list(finding.ids) == solids
    assert (s.op_count, s.issued_id_count) == before


def test_a_box_with_a_split_face_has_an_extra_edge() -> None:
    s = Session()
    s.add_box(2.0, 2.0, 2.0)
    (box,) = (EntityId(int(x)) for x in s.entities(EntityKind.SOLID))
    s.add_line((1.0, 0.0, 2.0), (1.0, 2.0, 2.0))
    line = [EntityId(int(e)) for e in s.entities(EntityKind.EDGE)][-1]
    s.imprint([box], [line])
    # The top face is split in two pieces on one surface; its id denotes both.
    assert (
        sum(s.shape_count(EntityId(int(f))) for f in s.entities(EntityKind.FACE)) == 7
    )

    check = s.check_blocks([box])

    assert not check.is_compound_of_blocks
    (finding,) = check.findings
    assert finding.kind == BlocksErrorKind.EXTRA_EDGE
    assert list(finding.ids) == [int(box)]


def test_a_cylinder_is_not_a_block() -> None:
    s = Session()
    s.add_cylinder(1.0, 2.0)
    (cylinder,) = (int(x) for x in s.entities(EntityKind.SOLID))

    check = s.check_blocks()

    assert not check.is_compound_of_blocks
    (finding,) = check.findings
    assert finding.kind == BlocksErrorKind.NOT_BLOCK
    assert list(finding.ids) == [cylinder]


def test_a_grid_of_glued_boxes_is_a_compound_of_blocks() -> None:
    assert grid().check_blocks().is_compound_of_blocks


def test_check_blocks_reports_progress_to_one_and_a_cancel_stops_it() -> None:
    s = two_boxes(glued=False)
    seen: list[float] = []
    before = state(s)

    s.check_blocks(progress=seen.append)
    with pytest.raises(ps.PysmeshCancelled):
        s.check_blocks(cancel=lambda: True)

    assert seen and seen[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in itertools.pairwise(seen))
    assert state(s) == before


def test_check_blocks_refuses_an_empty_list() -> None:
    s = two_boxes(glued=True)

    with pytest.raises(ps.PysmeshError, match="Session.check_blocks: an empty list"):
        s.check_blocks([])


# ---- propagation_chain ----------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("midpoint", "axis", "count"),
    [
        ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), 8),
        ((0.0, 0.5, 0.0), (0.0, 1.0, 0.0), 6),
        ((0.0, 0.0, 0.5), (0.0, 0.0, 1.0), 12),
    ],
)
def test_a_chain_on_a_grid_has_the_analytic_count_of_parallel_edges(
    midpoint: tuple[float, float, float], axis: tuple[float, float, float], count: int
) -> None:
    s = grid()
    edge = at(s, EntityKind.EDGE, midpoint)
    before = (s.op_count, s.issued_id_count)

    chain = s.propagation_chain(edge)

    assert (s.op_count, s.issued_id_count) == before
    assert len(chain) == count
    assert int(edge) in chain
    assert list(chain) == sorted(chain)
    lines = s.curve_geometry([EntityId(int(i)) for i in chain])
    assert lines.type == ("Line",) * count
    np.testing.assert_allclose(np.abs(lines.axis @ np.asarray(axis)), 1.0, rtol=1e-12)


# ---- opposite_face, face_by_edges, find_block_near ------------------------------------- #


def test_the_opposite_of_a_blocks_bottom_is_its_top() -> None:
    s = grid()
    block = at(s, EntityKind.SOLID, (1.5, 2.5, 0.5))
    bottom = at(s, EntityKind.FACE, (1.5, 2.5, 0.0))
    side = at(s, EntityKind.FACE, (1.0, 2.5, 0.5))
    before = (s.op_count, s.issued_id_count)

    top = s.opposite_face(block, bottom)
    across = s.opposite_face(block, side)

    assert (s.op_count, s.issued_id_count) == before
    assert top == at(s, EntityKind.FACE, (1.5, 2.5, 1.0))
    assert across == at(s, EntityKind.FACE, (2.0, 2.5, 0.5))


def test_two_edges_of_one_face_find_it() -> None:
    s = grid()
    ea = at(s, EntityKind.EDGE, (1.5, 3.0, 1.0))
    eb = at(s, EntityKind.EDGE, (2.0, 2.5, 1.0))
    before = (s.op_count, s.issued_id_count)

    face = s.face_by_edges(ea, eb)

    assert (s.op_count, s.issued_id_count) == before
    assert face == at(s, EntityKind.FACE, (1.5, 2.5, 1.0))


@pytest.mark.parametrize(
    ("point", "centre"),
    [
        ((1.5, 2.5, 0.5), (1.5, 2.5, 0.5)),
        ((0.2, 0.3, 0.4), (0.5, 0.5, 0.5)),
        ((5.0, 5.0, 5.0), (1.5, 2.5, 0.5)),
        ((-1.0, 1.5, 0.5), (0.5, 1.5, 0.5)),
        ((0.5, 0.5, 1.0), (0.5, 0.5, 0.5)),
    ],
)
def test_the_block_near_a_point_holds_it_or_is_nearest(
    point: tuple[float, float, float], centre: tuple[float, float, float]
) -> None:
    s = grid()
    before = (s.op_count, s.issued_id_count)

    block = s.find_block_near(point)

    assert (s.op_count, s.issued_id_count) == before
    assert block == at(s, EntityKind.SOLID, centre)


def test_the_block_searches_refuse_naming_their_inputs() -> None:
    s = grid()
    block = at(s, EntityKind.SOLID, (0.5, 0.5, 0.5))
    far_face = at(s, EntityKind.FACE, (1.5, 2.5, 1.0))
    ea = at(s, EntityKind.EDGE, (0.5, 0.0, 0.0))
    eb = at(s, EntityKind.EDGE, (1.5, 3.0, 1.0))
    empty = Session()
    empty.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0)

    with pytest.raises(
        ps.PysmeshError, match=f"face {far_face} is not a face of solid {block}"
    ):
        s.opposite_face(block, far_face)
    with pytest.raises(
        ps.PysmeshError, match=f"entity {far_face} is a FACE, not a SOLID"
    ):
        s.opposite_face(far_face, far_face)
    with pytest.raises(
        ps.PysmeshError, match=f"edges {ea} and {eb} do not belong to one face"
    ):
        s.face_by_edges(ea, eb)
    with pytest.raises(
        ps.PysmeshError,
        match=r"the point \(1, 1, 0.5\) lies on the boundary of 4 blocks .* and inside none",
    ):
        s.find_block_near((1.0, 1.0, 0.5))
    with pytest.raises(ps.PysmeshError, match="the model has no solid"):
        empty.find_block_near((0.0, 0.0, 0.0))


def test_opposite_face_refuses_a_solid_that_is_not_a_block() -> None:
    s = Session()
    s.add_cylinder(1.0, 2.0)
    (cylinder,) = (EntityId(int(x)) for x in s.entities(EntityKind.SOLID))
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    with pytest.raises(
        ps.PysmeshError,
        match=f"solid {cylinder} is not a block GEOM's block explorer can order from face",
    ):
        s.opposite_face(cylinder, face)
