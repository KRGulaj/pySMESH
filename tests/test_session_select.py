# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for rule-based selection: Session.select, the filters and the stored form.

Every filter has an analytic test and a reverse test. The answers follow from the
geometry of the fixtures, read back by centroid, never from the order OCCT built them in:

* The box ``[0, 3] x [0, 7] x [0, 11]`` has two faces of each area 21, 33 and 77, four
  edges of each length 3, 7 and 11, and volume 231. Its top and bottom faces are
  horizontal, its four side faces vertical. Every pair of its faces meets at 90 degrees,
  so every edge is a feature edge below 90 degrees and none above it.
* A fillet of radius 1 on the top edge at ``y = 7`` joins the top face and the side face
  ``y = 7`` tangentially (G1), so the three are one continuous patch and the fillet's two
  boundary edges are not feature edges.
* The four edges of the box along ``x`` are opposite to each other across the four
  rectangles that hold them.

The filters compare measures with Shaper's margin of 1e-7; the fixtures keep every
measure at least 1 away from every threshold, so the tests do not depend on round-off.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import EntityId, EntityKind, Session

BOX_A: float = 3.0
BOX_B: float = 7.0
BOX_C: float = 11.0

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


def box_session() -> Session:
    """The 3 x 7 x 11 box at the origin."""
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    return s


def at(s: Session, kind: EntityKind, point: tuple[float, float, float]) -> EntityId:
    """The one live entity of a kind whose centroid is the given point."""
    table = s.entity_table(kind)
    near = np.linalg.norm(table.centroid - np.asarray(point), axis=1) < 1e-9
    assert int(np.count_nonzero(near)) == 1
    return EntityId(int(table.ids[near][0]))


def where(
    s: Session,
    kind: EntityKind,
    test: Callable[[NDArray[np.float64]], NDArray[np.bool_]],
) -> list[int]:
    """The live ids of a kind whose (N, 3) centroids pass a vectorised test, ascending."""
    table = s.entity_table(kind)
    keep = test(table.centroid)
    return sorted(int(i) for i in table.ids[keep])


def everything(s: Session, kind: EntityKind) -> list[int]:
    """Every live id of a kind, ascending."""
    return sorted(int(i) for i in s.entities(kind))


def select(s: Session, kind: EntityKind, *filters: object) -> list[int]:
    """Session.select of a rule over the given filters, as a list."""
    found = s.select(kind, ps.Rule(tuple(filters)))  # type: ignore[arg-type]
    assert found.dtype == np.int64
    return [int(i) for i in found]


def complement(s: Session, kind: EntityKind, ids: list[int]) -> list[int]:
    """The live ids of a kind not in ``ids``, ascending."""
    return sorted(set(everything(s, kind)) - set(ids))


# ---- the filters --------------------------------------------------------------------- #


def test_belongs_to_finds_the_faces_of_one_body_and_reverse_the_other() -> None:
    s = box_session()
    a = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    s.add_box(1.0, 1.0, 1.0, origin=(20.0, 0.0, 0.0))
    mine = where(s, EntityKind.FACE, lambda c: c[:, 0] < 10.0)
    before = state(s)

    found = select(s, EntityKind.FACE, ps.BelongsTo((a,)))
    others = select(s, EntityKind.FACE, ps.BelongsTo((a,), reverse=True))

    assert found == mine
    assert others == complement(s, EntityKind.FACE, mine)
    assert state(s) == before


def test_belongs_to_a_face_finds_its_four_edges_and_the_face_itself() -> None:
    s = box_session()
    top = at(s, EntityKind.FACE, (1.5, 3.5, BOX_C))

    edges = select(s, EntityKind.EDGE, ps.BelongsTo((top,)))
    faces = select(s, EntityKind.FACE, ps.BelongsTo((top,)))

    assert edges == where(s, EntityKind.EDGE, lambda c: np.isclose(c[:, 2], BOX_C))
    assert faces == [top]


def _filleted_box() -> tuple[Session, EntityId, EntityId, EntityId]:
    """The box with its top edge at y = 7 filleted: top face, fillet face, side y = 7."""
    s = box_session()
    s.fillet([at(s, EntityKind.EDGE, (1.5, BOX_B, BOX_C))], 1.0)
    top = at(s, EntityKind.FACE, (1.5, 3.0, BOX_C))
    side = at(s, EntityKind.FACE, (1.5, BOX_B, 5.0))
    params = s.surface_parameters(s.entities(EntityKind.FACE))
    cylinders = [
        EntityId(int(i))
        for i, t, r in zip(params.ids, params.types, params.radius1, strict=True)
        if t == "Cylinder" and math.isclose(r, 1.0, rel_tol=1e-12)
    ]
    assert len(cylinders) == 1
    return s, top, cylinders[0], side


def test_continuous_faces_follow_the_tangent_fillet_and_stop_at_sharp_edges() -> None:
    s, top, fillet, side = _filleted_box()

    patch = select(s, EntityKind.FACE, ps.ContinuousFaces((top,), 5.0))
    rest = select(s, EntityKind.FACE, ps.ContinuousFaces((top,), 5.0, reverse=True))

    assert patch == sorted([top, fillet, side])
    assert rest == complement(s, EntityKind.FACE, patch)


def test_continuous_faces_above_ninety_degrees_reach_every_face_of_a_box() -> None:
    s = box_session()
    top = at(s, EntityKind.FACE, (1.5, 3.5, BOX_C))

    found = select(s, EntityKind.FACE, ps.ContinuousFaces((top,), 95.0))

    assert found == everything(s, EntityKind.FACE)


@pytest.mark.parametrize(
    ("comparator", "value", "value_max", "lengths"),
    [
        ("lt", 5.0, 0.0, {BOX_A}),
        ("le", BOX_B, 0.0, {BOX_A, BOX_B}),
        ("gt", BOX_B, 0.0, {BOX_C}),
        ("ge", BOX_B, 0.0, {BOX_B, BOX_C}),
        ("between", BOX_A, BOX_B, {BOX_A, BOX_B}),
        ("strictly_between", BOX_A, BOX_C, {BOX_B}),
    ],
)
def test_edge_size_compares_with_shapers_margin(
    comparator: str, value: float, value_max: float, lengths: set[float]
) -> None:
    s = box_session()
    table = s.entity_table(EntityKind.EDGE)
    expected = sorted(
        int(i)
        for i, m in zip(table.ids, table.measure, strict=True)
        if any(math.isclose(m, length, rel_tol=1e-12) for length in lengths)
    )

    found = select(s, EntityKind.EDGE, ps.EdgeSize(comparator, value, value_max))  # type: ignore[arg-type]
    rest = select(
        s, EntityKind.EDGE, ps.EdgeSize(comparator, value, value_max, reverse=True)  # type: ignore[arg-type]
    )

    assert len(found) == 4 * len(lengths)
    assert found == expected
    assert rest == complement(s, EntityKind.EDGE, found)


def test_face_size_between_finds_the_faces_of_area_21_and_33() -> None:
    s = box_session()
    table = s.entity_table(EntityKind.FACE)
    expected = sorted(
        int(i) for i, m in zip(table.ids, table.measure, strict=True) if m < 40
    )

    found = select(s, EntityKind.FACE, ps.FaceSize("between", 20.0, 40.0))
    rest = select(s, EntityKind.FACE, ps.FaceSize("between", 20.0, 40.0, reverse=True))

    assert len(found) == 4
    assert found == expected
    assert rest == complement(s, EntityKind.FACE, found)


def test_volume_size_tells_the_box_from_the_unit_cube() -> None:
    s = box_session()
    big = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    s.add_box(1.0, 1.0, 1.0, origin=(20.0, 0.0, 0.0))
    small = EntityId(int(s.entities(EntityKind.SOLID)[1]))

    found = select(s, EntityKind.SOLID, ps.VolumeSize("ge", BOX_A * BOX_B * BOX_C))
    rest = select(s, EntityKind.SOLID, ps.VolumeSize("lt", 2.0, reverse=True))

    assert found == [big]
    assert rest == [big]
    assert select(s, EntityKind.SOLID, ps.VolumeSize("lt", 2.0)) == [small]


def test_external_faces_leave_out_the_face_two_solids_share() -> None:
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    s.add_box(1.0, 1.0, 1.0, origin=(1.0, 0.0, 0.0))
    s.fragment([EntityId(int(i)) for i in s.entities(EntityKind.SOLID)])
    # Both boxes' ids denote the shared face: fragment merges the coincident faces.
    shared = where(
        s, EntityKind.FACE, lambda c: np.all(np.isclose(c, (1.0, 0.5, 0.5)), axis=1)
    )
    assert len(shared) == 2
    assert s.shape_count(EntityId(shared[0])) == 1

    external = select(s, EntityKind.FACE, ps.ExternalFaces())
    internal = select(s, EntityKind.FACE, ps.ExternalFaces(reverse=True))

    assert internal == shared
    assert external == complement(s, EntityKind.FACE, shared)
    assert len(external) == 10


def test_feature_edges_are_the_box_edges_below_ninety_degrees_and_none_above() -> None:
    s = box_session()

    sharp = select(s, EntityKind.EDGE, ps.FeatureEdges(5.0))
    none = select(s, EntityKind.EDGE, ps.FeatureEdges(95.0))
    smooth = select(s, EntityKind.EDGE, ps.FeatureEdges(95.0, reverse=True))

    assert sharp == everything(s, EntityKind.EDGE)
    assert none == []
    assert smooth == everything(s, EntityKind.EDGE)


def test_feature_edges_skip_the_tangent_edges_of_a_fillet() -> None:
    s, top, fillet, side = _filleted_box()
    tangent = sorted(
        set(select(s, EntityKind.EDGE, ps.BelongsTo((fillet,))))
        & (
            set(select(s, EntityKind.EDGE, ps.BelongsTo((top,))))
            | set(select(s, EntityKind.EDGE, ps.BelongsTo((side,))))
        )
    )
    assert len(tangent) == 2

    sharp = select(s, EntityKind.EDGE, ps.FeatureEdges(5.0))
    smooth = select(s, EntityKind.EDGE, ps.FeatureEdges(5.0, reverse=True))

    assert smooth == tangent
    assert sharp == complement(s, EntityKind.EDGE, tangent)


def test_feature_edges_of_a_cylinder_are_its_two_rims_and_not_its_seam() -> None:
    s = Session()
    s.add_cylinder(2.0, 5.0)
    rims = sorted(
        [
            at(s, EntityKind.EDGE, (0.0, 0.0, 0.0)),
            at(s, EntityKind.EDGE, (0.0, 0.0, 5.0)),
        ]
    )

    found = select(s, EntityKind.EDGE, ps.FeatureEdges(5.0))

    assert found == rims


def test_horizontal_and_vertical_faces_of_a_box() -> None:
    s = box_session()
    flat = where(
        s,
        EntityKind.FACE,
        lambda c: np.isclose(c[:, 2], 0.0) | np.isclose(c[:, 2], BOX_C),
    )

    horizontal = select(s, EntityKind.FACE, ps.HorizontalFace())
    not_horizontal = select(s, EntityKind.FACE, ps.HorizontalFace(reverse=True))
    vertical = select(s, EntityKind.FACE, ps.VerticalFace())
    not_vertical = select(s, EntityKind.FACE, ps.VerticalFace(reverse=True))

    assert horizontal == flat
    assert not_horizontal == complement(s, EntityKind.FACE, flat)
    assert vertical == not_horizontal
    assert not_vertical == horizontal


def test_a_tilted_box_has_no_horizontal_face_and_a_cylinder_one_vertical_face() -> None:
    s = box_session()
    s.rotate((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), math.radians(30.0))
    c = Session()
    c.add_cylinder(2.0, 5.0)

    tilted = select(s, EntityKind.FACE, ps.HorizontalFace())
    lateral = select(c, EntityKind.FACE, ps.VerticalFace())

    assert tilted == []
    assert lateral == [at(c, EntityKind.FACE, (0.0, 0.0, 2.5))]


def test_on_geometry_finds_the_coplanar_face_and_the_collinear_edge_of_a_second_box() -> (
    None
):
    s = box_session()
    s.add_box(1.0, 1.0, 1.0, origin=(0.0, 20.0, 0.0))
    near = at(s, EntityKind.FACE, (0.0, 3.5, 5.5))
    far = at(s, EntityKind.FACE, (0.0, 20.5, 0.5))
    edge = at(s, EntityKind.EDGE, (0.0, 3.5, 0.0))
    far_edge = at(s, EntityKind.EDGE, (0.0, 20.5, 0.0))

    faces = select(s, EntityKind.FACE, ps.OnGeometry((near,)))
    edges = select(s, EntityKind.EDGE, ps.OnGeometry((edge,)))
    others = select(s, EntityKind.FACE, ps.OnGeometry((near,), reverse=True))

    assert faces == sorted([near, far])
    assert edges == sorted([edge, far_edge])
    assert others == complement(s, EntityKind.FACE, faces)


def test_on_line_finds_the_collinear_edges_and_vertices() -> None:
    s = box_session()
    s.add_box(1.0, 1.0, 1.0, origin=(10.0, 0.0, 0.0))
    axis_edge = at(s, EntityKind.EDGE, (1.5, 0.0, 0.0))

    def on_axis(c: NDArray[np.float64]) -> NDArray[np.bool_]:
        return np.isclose(c[:, 1], 0.0) & np.isclose(c[:, 2], 0.0)

    edges = select(s, EntityKind.EDGE, ps.OnLine((axis_edge,)))
    vertices = select(s, EntityKind.VERTEX, ps.OnLine((axis_edge,)))
    off = select(s, EntityKind.VERTEX, ps.OnLine((axis_edge,), reverse=True))

    assert edges == where(s, EntityKind.EDGE, on_axis)
    assert len(edges) == 2
    assert vertices == where(s, EntityKind.VERTEX, on_axis)
    assert len(vertices) == 4
    assert off == complement(s, EntityKind.VERTEX, vertices)


@pytest.mark.parametrize(
    ("kind", "count"),
    [(EntityKind.FACE, 1), (EntityKind.EDGE, 4), (EntityKind.VERTEX, 4)],
)
def test_on_plane_finds_the_boundary_of_the_top_face(
    kind: EntityKind, count: int
) -> None:
    s = box_session()
    top = at(s, EntityKind.FACE, (1.5, 3.5, BOX_C))
    on_top = where(s, kind, lambda c: np.isclose(c[:, 2], BOX_C))

    found = select(s, kind, ps.OnPlane((top,)))
    rest = select(s, kind, ps.OnPlane((top,), reverse=True))

    assert len(found) == count
    assert found == on_top
    assert rest == complement(s, kind, found)


@pytest.mark.parametrize("kind", [EntityKind.FACE, EntityKind.EDGE, EntityKind.VERTEX])
def test_on_plane_side_keeps_what_lies_above_a_rectangle_facing_up(
    kind: EntityKind,
) -> None:
    """The box's side faces and vertical edges have their middle on z = 5.5: not above."""
    s = box_session()
    s.add_rectangle((-10.0, -10.0, 5.5), (0.0, 0.0, 1.0), 1.0, 1.0)
    plane = EntityId(int(s.entities(EntityKind.FACE)[-1]))
    expected = where(s, kind, lambda c: c[:, 2] > 5.5 + 1e-6)

    found = select(s, kind, ps.OnPlaneSide(plane))
    rest = select(s, kind, ps.OnPlaneSide(plane, reverse=True))

    assert found == expected
    assert rest == complement(s, kind, found)


def test_opposite_to_edge_walks_the_four_edges_along_x() -> None:
    s = box_session()
    x_edge = at(s, EntityKind.EDGE, (1.5, 0.0, 0.0))
    along_x = where(s, EntityKind.EDGE, lambda c: np.isclose(c[:, 0], 1.5))

    found = select(s, EntityKind.EDGE, ps.OppositeToEdge(x_edge))
    rest = select(s, EntityKind.EDGE, ps.OppositeToEdge(x_edge, reverse=True))

    assert found == along_x
    assert len(found) == 4
    assert rest == complement(s, EntityKind.EDGE, found)


@pytest.mark.parametrize(
    ("location", "parts"),
    [
        ("in", {"inner"}),
        ("out", {"far"}),
        ("on", {"own"}),
        ("not_on", {"inner", "far"}),
        ("not_out", {"own", "inner"}),
        ("not_in", {"own", "far"}),
    ],
)
def test_relative_to_solid_classifies_its_own_an_inner_and_a_far_box(
    location: str, parts: set[str]
) -> None:
    s = box_session()
    solid = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    s.add_box(1.0, 1.0, 1.0, origin=(1.0, 1.0, 1.0))
    s.add_box(1.0, 1.0, 1.0, origin=(20.0, 1.0, 1.0))
    inner = where(s, EntityKind.FACE, lambda c: np.all((c >= 1.0) & (c <= 2.0), axis=1))
    far = where(s, EntityKind.FACE, lambda c: c[:, 0] > 10.0)
    groups = {
        "own": complement(s, EntityKind.FACE, inner + far),
        "inner": inner,
        "far": far,
    }
    expected = sorted(i for p in parts for i in groups[p])

    found = select(s, EntityKind.FACE, ps.RelativeToSolid(solid, location))  # type: ignore[arg-type]
    rest = select(s, EntityKind.FACE, ps.RelativeToSolid(solid, location, reverse=True))  # type: ignore[arg-type]

    assert all(len(g) == 6 for g in groups.values())
    assert found == expected
    assert rest == complement(s, EntityKind.FACE, found)


def test_topo_connected_faces_from_a_face_an_edge_and_a_vertex() -> None:
    s = box_session()
    top = at(s, EntityKind.FACE, (1.5, 3.5, BOX_C))
    bottom = at(s, EntityKind.FACE, (1.5, 3.5, 0.0))
    edge = at(s, EntityKind.EDGE, (1.5, 0.0, 0.0))
    vertex = at(s, EntityKind.VERTEX, (0.0, 0.0, 0.0))
    every = everything(s, EntityKind.FACE)

    by_face = select(s, EntityKind.FACE, ps.TopoConnectedFaces(top, False))
    by_edge = select(s, EntityKind.FACE, ps.TopoConnectedFaces(edge, False))
    by_vertex = select(s, EntityKind.FACE, ps.TopoConnectedFaces(vertex, False))
    spread = select(s, EntityKind.FACE, ps.TopoConnectedFaces(edge, True))
    not_by_face = select(
        s, EntityKind.FACE, ps.TopoConnectedFaces(top, False, reverse=True)
    )

    assert by_face == complement(s, EntityKind.FACE, [bottom])
    assert by_edge == where(
        s,
        EntityKind.FACE,
        lambda c: np.isclose(c[:, 1], 0.0) | np.isclose(c[:, 2], 0.0),
    )
    assert by_vertex == where(
        s,
        EntityKind.FACE,
        lambda c: np.isclose(c[:, 0], 0.0)
        | np.isclose(c[:, 1], 0.0)
        | np.isclose(c[:, 2], 0.0),
    )
    assert spread == every
    assert not_by_face == [bottom]


# ---- combination, split ids, no side effects ----------------------------------------- #


def test_filters_combine_with_and() -> None:
    s = box_session()
    s.add_box(1.0, 1.0, 1.0, origin=(20.0, 0.0, 0.0))
    box = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    top = at(s, EntityKind.FACE, (1.5, 3.5, BOX_C))

    found = select(
        s,
        EntityKind.FACE,
        ps.HorizontalFace(),
        ps.BelongsTo((box,)),
        ps.FaceSize("lt", 22.0),
        ps.TopoConnectedFaces(top, False),
    )

    assert found == [top]


def test_a_split_solid_is_judged_by_its_total_volume_and_by_every_piece() -> None:
    """split keeps the box's id on both halves; its total volume is still 231."""
    s = box_session()
    box = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    s.add_box(5.0, 9.0, 10.0, origin=(-1.0, -1.0, 5.5))
    tool = EntityId(int(s.entities(EntityKind.SOLID)[1]))
    s.split([box], [tool])
    assert s.shape_count(box) == 2
    s.add_rectangle((-10.0, -10.0, 3.0), (0.0, 0.0, 1.0), 1.0, 1.0)
    plane = EntityId(int(s.entities(EntityKind.FACE)[-1]))

    big = select(s, EntityKind.SOLID, ps.VolumeSize("gt", 200.0))
    above = select(s, EntityKind.SOLID, ps.OnPlaneSide(plane))

    assert box in big
    assert box not in above
    assert tool in above


def test_select_issues_nothing_and_changes_nothing() -> None:
    s, top, fillet, _ = _filleted_box()
    before = state(s)

    select(s, EntityKind.FACE, ps.ContinuousFaces((top,), 5.0), ps.ExternalFaces())
    select(s, EntityKind.EDGE, ps.FeatureEdges(5.0), ps.OnGeometry((fillet,)))
    select(
        s,
        EntityKind.FACE,
        ps.RelativeToSolid(EntityId(int(s.entities(EntityKind.SOLID)[0])), "on"),
    )

    assert state(s) == before


# ---- progress and cancel ------------------------------------------------------------- #


def _box_with_inner_and_far_boxes() -> tuple[Session, EntityId, list[int]]:
    """The box, a unit box inside it and one far from it; the box's id; the unit faces."""
    s = box_session()
    solid = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    s.add_box(1.0, 1.0, 1.0, origin=(1.0, 1.0, 1.0))
    s.add_box(1.0, 1.0, 1.0, origin=(20.0, 1.0, 1.0))
    inner = where(s, EntityKind.FACE, lambda c: np.all((c >= 1.0) & (c <= 2.0), axis=1))
    far = where(s, EntityKind.FACE, lambda c: c[:, 0] > 10.0)
    return s, solid, sorted(inner + far)


def test_select_reports_progress_over_every_filter_to_the_end() -> None:
    """A size filter, RelativeToSolid ("not_on" makes two finder calls), a native filter.

    The unit faces (area 1 < 2) are not on the box, and every face is external: the
    answer is the twelve unit faces.
    """
    s, solid, unit_faces = _box_with_inner_and_far_boxes()
    rule = ps.Rule(
        (
            ps.FaceSize("lt", 2.0),
            ps.RelativeToSolid(solid, "not_on"),
            ps.ExternalFaces(),
        )
    )
    seen: list[float] = []
    before = state(s)

    found = s.select(EntityKind.FACE, rule, progress=seen.append)

    assert [int(i) for i in found] == unit_faces
    assert len(seen) >= 3
    assert all(b > a for a, b in itertools.pairwise(seen))
    assert 0.0 <= seen[0]
    assert seen[-1] == 1.0
    assert state(s) == before


@pytest.mark.parametrize(
    "make",
    [
        lambda solid: ps.FaceSize("gt", 1.0),
        lambda solid: ps.RelativeToSolid(solid, "in"),
        lambda solid: ps.HorizontalFace(),
    ],
    ids=["size", "relative_to_solid", "native"],
)
def test_a_cancelled_select_raises_and_leaves_the_session_as_it_was(
    make: Callable[[EntityId], object],
) -> None:
    s, solid, _ = _box_with_inner_and_far_boxes()
    rule = ps.Rule((make(solid),))  # type: ignore[arg-type]
    before = state(s)

    with pytest.raises(ps.PysmeshCancelled):
        s.select(EntityKind.FACE, rule, cancel=lambda: True)

    assert state(s) == before


def test_a_cancel_after_the_first_filter_stops_select() -> None:
    s = box_session()
    asked: list[int] = []

    def cancel() -> bool:
        asked.append(1)
        return len(asked) > 1

    before = state(s)

    with pytest.raises(ps.PysmeshCancelled):
        s.select(
            EntityKind.FACE,
            ps.Rule((ps.HorizontalFace(), ps.FaceSize("gt", 1.0))),
            cancel=cancel,
        )

    assert len(asked) >= 2
    assert state(s) == before


# ---- refusals ------------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("kind", "make", "match"),
    [
        (
            EntityKind.FACE,
            lambda s: ps.VolumeSize("lt", 1.0),
            "VolumeSize does not apply to FACE",
        ),
        (
            EntityKind.SOLID,
            lambda s: ps.HorizontalFace(),
            "HorizontalFace does not apply to SOLID",
        ),
        (
            EntityKind.VERTEX,
            lambda s: ps.FeatureEdges(5.0),
            "FeatureEdges does not apply to VERTEX",
        ),
        (
            EntityKind.FACE,
            lambda s: ps.BelongsTo((EntityId(9999),)),
            "entity 9999, which this session never issued",
        ),
        (EntityKind.FACE, lambda s: ps.BelongsTo(()), "names no entity"),
        (
            EntityKind.EDGE,
            lambda s: ps.FeatureEdges(-1.0),
            "angle_deg=-1.0, below zero",
        ),
        (EntityKind.EDGE, lambda s: ps.EdgeSize("lt", -1.0), "value=-1.0, below zero"),
        (
            EntityKind.EDGE,
            lambda s: ps.EdgeSize("between", 5.0, 2.0),
            "value=5.0 above value_max=2.0",
        ),
        (EntityKind.EDGE, lambda s: ps.EdgeSize("below", 5.0), "comparator='below'"),
        (EntityKind.EDGE, lambda s: ps.EdgeSize("lt", math.nan), "not a finite number"),
        (
            EntityKind.FACE,
            lambda s: ps.RelativeToSolid(EntityId(1), "inside"),
            "location='inside'",
        ),
        (
            EntityKind.FACE,
            lambda s: ps.RelativeToSolid(EntityId(2), "in"),
            "names entity 2, a FACE; it takes a solid",
        ),
        (
            EntityKind.FACE,
            lambda s: ps.TopoConnectedFaces(EntityId(1), 1),
            "propagation=1, not a bool",
        ),
        (EntityKind.FACE, lambda s: "HorizontalFace", "str is not a selection filter"),
    ],
)
def test_select_refuses_a_filter_that_cannot_apply(
    kind: EntityKind, make: object, match: str
) -> None:
    s = box_session()
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=match) as err:
        s.select(kind, ps.Rule((make(s),)))  # type: ignore[operator]

    assert "Session.select" in str(err.value)
    assert state(s) == before


def test_select_refuses_a_dead_id_a_curved_plane_a_circle_line_and_an_empty_rule() -> (
    None
):
    s = Session()
    s.add_cylinder(2.0, 5.0)
    lateral = at(s, EntityKind.FACE, (0.0, 0.0, 2.5))
    rim = at(s, EntityKind.EDGE, (0.0, 0.0, 0.0))
    s.add_box(1.0, 1.0, 1.0, origin=(10.0, 0.0, 0.0))
    dead = EntityId(int(s.entities(EntityKind.SOLID)[1]))
    s.remove([dead])

    with pytest.raises(ps.PysmeshError, match=f"names entity {dead}, which is dead"):
        s.select(EntityKind.FACE, ps.Rule((ps.BelongsTo((dead,)),)))
    with pytest.raises(
        ps.PysmeshError, match=f"names face {lateral}, which is not planar"
    ):
        s.select(EntityKind.FACE, ps.Rule((ps.OnPlane((lateral,)),)))
    with pytest.raises(
        ps.PysmeshError, match=f"names edge {rim}, which is not a straight"
    ):
        s.select(EntityKind.EDGE, ps.Rule((ps.OnLine((rim,)),)))
    with pytest.raises(ps.PysmeshError, match="the rule holds no filter"):
        s.select(EntityKind.FACE, ps.Rule(()))
    with pytest.raises(ps.PysmeshError, match="not a Rule"):
        s.select(EntityKind.FACE, (ps.HorizontalFace(),))  # type: ignore[arg-type]


# ---- the stored form ----------------------------------------------------------------- #

_ONE_OF_EACH: tuple[object, ...] = (
    ps.BelongsTo((EntityId(1), EntityId(2))),
    ps.ContinuousFaces((EntityId(3),), 5.0, reverse=True),
    ps.EdgeSize("between", 1.0, 2.5),
    ps.ExternalFaces(),
    ps.FaceSize("lt", 3.25),
    ps.FeatureEdges(12.5, reverse=True),
    ps.HorizontalFace(),
    ps.OnGeometry((EntityId(4),)),
    ps.OnLine((EntityId(5), EntityId(6))),
    ps.OnPlane((EntityId(7),), reverse=True),
    ps.OnPlaneSide(EntityId(8)),
    ps.OppositeToEdge(EntityId(9)),
    ps.RelativeToSolid(EntityId(10), "not_out"),
    ps.TopoConnectedFaces(EntityId(11), True),
    ps.VerticalFace(reverse=True),
    ps.VolumeSize("strictly_between", 0.5, 4.0),
)


@pytest.mark.parametrize("f", _ONE_OF_EACH, ids=lambda f: type(f).__name__)
def test_every_filter_survives_a_json_round_trip(f: object) -> None:
    rule = ps.Rule((f,))  # type: ignore[arg-type]

    data = json.loads(json.dumps(ps.rule_to_dict(rule)))
    back = ps.rule_from_dict(data)

    assert back == rule
    assert data["format"] == 1
    assert data["filters"][0]["type"] == type(f).__name__


def test_a_rule_of_every_filter_survives_a_json_round_trip_in_order() -> None:
    rule = ps.Rule(_ONE_OF_EACH)  # type: ignore[arg-type]

    back = ps.rule_from_dict(json.loads(json.dumps(ps.rule_to_dict(rule))))

    assert back == rule
    assert len(back.filters) == 16


def test_the_stored_form_is_flat_with_one_key_per_field() -> None:
    rule = ps.Rule((ps.EdgeSize("lt", 2.0), ps.BelongsTo((EntityId(3),))))

    data = ps.rule_to_dict(rule)

    assert data == {
        "format": 1,
        "filters": [
            {
                "type": "EdgeSize",
                "comparator": "lt",
                "value": 2.0,
                "value_max": 0.0,
                "reverse": False,
            },
            {"type": "BelongsTo", "entities": [3], "reverse": False},
        ],
    }


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"format": 2, "filters": []}, "format 2 is not 1"),
        ({"filters": []}, "misses \\['format'\\]"),
        ({"format": 1, "filters": [], "name": "x"}, "extra keys \\['name'\\]"),
        (
            {"format": 1, "filters": [{"type": "OnShapeName", "reverse": False}]},
            "type 'OnShapeName'",
        ),
        (
            {"format": 1, "filters": [{"type": "ExternalFaces"}]},
            "misses \\['reverse'\\]",
        ),
        (
            {
                "format": 1,
                "filters": [{"type": "ExternalFaces", "reverse": False, "angle": 1.0}],
            },
            "extra keys \\['angle'\\]",
        ),
        (
            {
                "format": 1,
                "filters": [
                    {"type": "FeatureEdges", "angle_deg": "5", "reverse": False}
                ],
            },
            "angle_deg='5', not a finite number",
        ),
        (
            {
                "format": 1,
                "filters": [{"type": "OnPlaneSide", "face": True, "reverse": False}],
            },
            "face=True, not an entity id",
        ),
        ({"format": 1, "filters": "EdgeSize"}, "filters is a str, not a list"),
        ([1], "not a dict"),
    ],
)
def test_rule_from_dict_refuses_a_bad_stored_form(data: object, match: str) -> None:
    with pytest.raises(ps.PysmeshError, match=match) as err:
        ps.rule_from_dict(data)  # type: ignore[arg-type]

    assert "rule_from_dict" in str(err.value)
