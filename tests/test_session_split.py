# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for the splitting operations that give each piece its own id.

``split_by_faces``, ``split_by_plane`` and ``keep_half_space`` cut solids with
``BRepAlgoAPI_Splitter`` and apply the sixth rule of entity ids: a target solid cut into
two or more solids loses its id, and each piece gets a new id with the role ``GENERATED``
whose origin is the target. Faces, edges and vertices keep the five rules of ``split``.

Every geometric value is checked against a closed form:

* A box ``[0, a] x [0, b] x [0, c]`` cut by the plane ``x = t`` leaves the boxes
  ``[0, t]`` and ``[t, a]`` (volume ``t b c`` and ``(a - t) b c``, centroids at the middle
  of each box). Every face, edge and vertex of the result is an axis-aligned rectangle,
  segment or point, so its measure and centroid follow from its bounds.
* An L-shaped prism of unit thickness, the union of ``[0, 3] x [0, 1]`` and
  ``[0, 1] x [0, 3]``, has area 5. The plane ``x + y = 2.5`` cuts it into three pieces:
  the corner (area 3) and two tips, each the trapezoid
  ``{0 <= y <= 1, 2.5 - y <= x <= 3}`` of area ``int_0^1 (0.5 + y) dy = 1``, whose
  centroid is ``x = int_0^1 (9 - (2.5 - y)^2) / 2 dy = 59 / 24`` and
  ``y = int_0^1 y (0.5 + y) dy = 7 / 12``.

The pieces are planar polyhedra, so OCCT integrates them exactly: measures and centroids
are compared at a relative tolerance of 1e-12, with an absolute floor of 1e-12 times the
model size for coordinates near zero.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import (
    EntityId,
    EntityKind,
    NameRole,
    PysmeshCancelled,
    ResolutionStatus,
    Session,
)

BOX_A: float = 3.0
BOX_B: float = 7.0
BOX_C: float = 11.0
CUT_X: float = 1.0

# Planar pieces integrate exactly; the tolerance covers round-off only.
EXACT_RTOL: float = 1e-12
COORD_ATOL: float = 1e-12 * BOX_C

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)

Piece = tuple[tuple[float, float, float], float]


# ---- analytic model of a box cut by x = t ---------------------------------------- #


def _rect_pieces(
    lo: tuple[float, float, float], hi: tuple[float, float, float]
) -> tuple[tuple[float, float, float], float]:
    """Centroid and measure of an axis-aligned box, rectangle, segment or point."""
    centre = tuple(0.5 * (a + b) for a, b in zip(lo, hi, strict=True))
    extents = [b - a for a, b in zip(lo, hi, strict=True) if b - a > 0.0]
    measure = float(np.prod(extents)) if extents else 0.0
    return (centre[0], centre[1], centre[2]), measure


def box_cut_model(
    a: float, b: float, c: float, t: float
) -> dict[EntityKind, list[Piece]]:
    """Every sub-shape of ``[0, a] x [0, b] x [0, c]`` cut by ``x = t``, by kind.

    Each entry is the centroid and the measure (volume, area, length, 0 for a vertex) of
    one sub-shape of the result. The x coordinates of the result are ``0, t, a``.
    """
    xs = (0.0, t, a)
    model: dict[EntityKind, list[Piece]] = {k: [] for k in _KINDS}
    for i in range(2):
        model[EntityKind.SOLID].append(
            _rect_pieces((xs[i], 0.0, 0.0), (xs[i + 1], b, c))
        )
    # Faces: the three planes x = const, and each side plane cut in two along x.
    for x in xs:
        model[EntityKind.FACE].append(_rect_pieces((x, 0.0, 0.0), (x, b, c)))
    for i in range(2):
        for y in (0.0, b):
            model[EntityKind.FACE].append(
                _rect_pieces((xs[i], y, 0.0), (xs[i + 1], y, c))
            )
        for z in (0.0, c):
            model[EntityKind.FACE].append(
                _rect_pieces((xs[i], 0.0, z), (xs[i + 1], b, z))
            )
    # Edges: along x (cut in two) and the rectangles' edges on each x plane.
    for i in range(2):
        for y in (0.0, b):
            for z in (0.0, c):
                model[EntityKind.EDGE].append(
                    _rect_pieces((xs[i], y, z), (xs[i + 1], y, z))
                )
    for x in xs:
        for y in (0.0, b):
            model[EntityKind.EDGE].append(_rect_pieces((x, y, 0.0), (x, y, c)))
        for z in (0.0, c):
            model[EntityKind.EDGE].append(_rect_pieces((x, 0.0, z), (x, b, z)))
    for x in xs:
        for y in (0.0, b):
            for z in (0.0, c):
                model[EntityKind.VERTEX].append(((x, y, z), 0.0))
    return model


def _matches(got_centre: np.ndarray, got_measure: float, want: Piece) -> bool:
    """Whether one measured sub-shape is the analytic one."""
    centre, measure = want
    return bool(
        np.allclose(got_centre, centre, rtol=EXACT_RTOL, atol=COORD_ATOL)
        and np.isclose(got_measure, measure, rtol=EXACT_RTOL, atol=COORD_ATOL)
    )


def ordinal_geometry(
    session: Session,
) -> dict[EntityKind, list[tuple[np.ndarray, float]]]:
    """Centroid and measure of every sub-shape, per ordinal, from the session's BREP.

    Read back through the stateless API, so the oracle does not depend on the registry it
    checks.
    """
    shape = ps.load_brep(session.export_handoff(allow_aliases=True).brep)
    out: dict[EntityKind, list[tuple[np.ndarray, float]]] = {
        EntityKind.SOLID: [(np.asarray(s.centroid), s.volume) for s in shape.solids()],
        EntityKind.FACE: [(np.asarray(f.centroid), f.area) for f in shape.faces()],
        EntityKind.EDGE: [
            (0.5 * (np.asarray(e.bbox)[:3] + np.asarray(e.bbox)[3:]), e.length)
            for e in shape.edges()
        ],
        EntityKind.VERTEX: [(np.asarray(v.xyz), 0.0) for v in shape.vertices()],
    }
    return out


def ids_by_ordinal(
    session: Session,
) -> dict[EntityKind, tuple[tuple[EntityId, ...], ...]]:
    """Every live id of each sub-shape, per ordinal, in the order of ordinal_geometry."""
    handoff = session.export_handoff(allow_aliases=True)
    return {
        EntityKind.SOLID: handoff.solid_ids_of,
        EntityKind.FACE: handoff.face_ids_of,
        EntityKind.EDGE: handoff.edge_ids_of,
        EntityKind.VERTEX: handoff.vertex_ids_of,
    }


def id_space(session: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, per kind: the state a refused or cancelled operation must keep."""
    return tuple(tuple(int(i) for i in session.entities(k)) for k in _KINDS)


def box_session() -> tuple[Session, EntityId]:
    """A session holding the 3 x 7 x 11 box at the origin, and its solid id."""
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    return s, EntityId(int(s.entities(EntityKind.SOLID)[0]))


def l_prism_session() -> tuple[Session, EntityId]:
    """A session holding the L-shaped prism of unit thickness (area 5), one solid."""
    s = Session()
    s.add_box(3.0, 1.0, 1.0)
    s.add_box(1.0, 3.0, 1.0)
    a, b = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))
    s.fuse([a], [b], parallel=False)
    solids = s.entities(EntityKind.SOLID)
    assert solids.size == 1
    return s, EntityId(int(solids[0]))


def free_face_at_x(s: Session, x: float) -> EntityId:
    """A free planar face on the plane ``x``, covering the box's y-z section with margin."""
    s.add_polyline(
        [
            (x, -1.0, -1.0),
            (x, BOX_B + 1.0, -1.0),
            (x, BOX_B + 1.0, BOX_C + 1.0),
            (x, -1.0, BOX_C + 1.0),
        ],
        closed=True,
    )
    edges = [EntityId(int(i)) for i in s.entities(EntityKind.EDGE)]
    before = {int(i) for i in s.entities(EntityKind.FACE)}
    s.make_face(edges[-4:])
    (face,) = {int(i) for i in s.entities(EntityKind.FACE)} - before
    return EntityId(face)


# ---- the box cut: analytic result and history against geometry ----------------------- #


SplitOp = Callable[[Session, EntityId], object]


def _split_by_plane(s: Session, target: EntityId) -> object:
    return s.split_by_plane([target], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0))


def _split_by_faces(s: Session, target: EntityId) -> object:
    tool = free_face_at_x(s, CUT_X)
    return s.split_by_faces([target], [tool])


@pytest.mark.parametrize("split", [_split_by_plane, _split_by_faces])
def test_a_box_cut_at_x_1_leaves_two_boxes_of_the_analytic_volume_and_centroid(
    split: SplitOp,
) -> None:
    s, box = box_session()

    split(s, box)

    want = box_cut_model(BOX_A, BOX_B, BOX_C, CUT_X)[EntityKind.SOLID]
    got = ordinal_geometry(s)[EntityKind.SOLID]
    assert len(got) == 2
    for piece in want:
        assert sum(_matches(c, m, piece) for c, m in got) == 1


@pytest.mark.parametrize("split", [_split_by_plane, _split_by_faces])
def test_a_box_cut_gives_each_piece_a_new_generated_id_and_kills_the_target(
    split: SplitOp,
) -> None:
    s, box = box_session()

    delta = split(s, box)

    assert isinstance(delta, ps.HistoryDelta)
    pieces = [EntityId(int(i)) for i in s.entities(EntityKind.SOLID)]
    assert len(pieces) == 2
    assert not s.is_alive(box)
    assert int(box) in delta.deleted.tolist()
    for piece in pieces:
        assert int(piece) in delta.created.tolist()
        name = s.name_of(piece)
        assert name.role is NameRole.GENERATED
        assert s.origin(piece).sources.tolist() == [int(box)]
        assert s.resolve(name).status is ResolutionStatus.RESOLVED


@pytest.mark.parametrize("split", [_split_by_plane, _split_by_faces])
def test_every_id_of_a_box_cut_denotes_the_analytic_sub_shapes(split: SplitOp) -> None:
    """History against geometry: every live id, per ordinal, against the closed form.

    Each sub-shape of the result is matched to its analytic box, rectangle, segment or
    point; its ids must be the ids the history rule gives it. Before the cut each original
    sub-shape is recorded by its own bounds, so an id carried onto the wrong piece fails.
    """
    s, box = box_session()
    before: dict[int, np.ndarray] = {}
    for kind in (EntityKind.FACE, EntityKind.EDGE, EntityKind.VERTEX):
        table = s.bounding_boxes(kind)
        for i, row in zip(table.ids.tolist(), table.bbox, strict=True):
            before[int(i)] = np.asarray(row)

    delta = split(s, box)

    assert isinstance(delta, ps.HistoryDelta)
    created = set(delta.created.tolist())
    # The tool body of split_by_faces stays in the model with its own ids: neither the box's
    # nor this operation's. Its sub-shapes are not part of the box model.
    tool = set().union(*(set(id_space(s)[i]) for i in range(4))) - created - set(before)
    tool.discard(int(box))
    model = box_cut_model(BOX_A, BOX_B, BOX_C, CUT_X)
    geometry = ordinal_geometry(s)
    ids = ids_by_ordinal(s)
    for kind in (EntityKind.FACE, EntityKind.EDGE, EntityKind.VERTEX):
        rows = [
            (geo, row_ids)
            for geo, row_ids in zip(geometry[kind], ids[kind], strict=True)
            if not {int(i) for i in row_ids} <= tool
        ]
        assert len(rows) == len(model[kind])
        for (centre, measure), row_ids in rows:
            assert sum(_matches(centre, measure, w) for w in model[kind]) == 1
            assert len(row_ids) == 1
            owner = int(row_ids[0])
            if abs(centre[0] - CUT_X) <= COORD_ATOL:
                # In the cut plane: the section face, its edges and its vertices are new.
                assert owner in created
                continue
            # Elsewhere a piece lies inside the bounds its id had before the cut.
            assert owner not in created
            lo, hi = before[owner][:3], before[owner][3:]
            assert np.all(centre >= lo - COORD_ATOL)
            assert np.all(centre <= hi + COORD_ATOL)
    # Every face, edge and vertex of the box survives: on two pieces exactly when the plane
    # crosses its bounds, on one otherwise.
    for i, bounds in before.items():
        crossed = bounds[0] < CUT_X - COORD_ATOL and bounds[3] > CUT_X + COORD_ATOL
        assert s.shape_count(EntityId(i)) == (2 if crossed else 1)


@pytest.mark.parametrize("split", [_split_by_plane, _split_by_faces])
def test_a_box_side_crossed_by_the_cut_keeps_its_id_on_both_halves(
    split: SplitOp,
) -> None:
    s, box = box_session()
    faces = s.bounding_boxes(EntityKind.FACE)
    # The side y = 0 spans x in [0, 3]; the cut at x = 1 crosses it.
    side = next(
        EntityId(int(i))
        for i, row in zip(faces.ids.tolist(), faces.bbox, strict=True)
        if row[1] == 0.0 and row[4] == 0.0
    )
    name = s.name_of(side)

    split(s, box)

    assert s.is_alive(side)
    assert s.shape_count(side) == 2
    assert s.resolve(name).status is ResolutionStatus.AMBIGUOUS
    measured = s.mass_properties([side])
    assert measured.measure[0] == pytest.approx(BOX_A * BOX_C, rel=EXACT_RTOL)
    np.testing.assert_allclose(
        measured.centroid[0],
        (0.5 * BOX_A, 0.0, 0.5 * BOX_C),
        rtol=EXACT_RTOL,
        atol=COORD_ATOL,
    )


@pytest.mark.parametrize("split", [_split_by_plane, _split_by_faces])
def test_the_section_face_gets_a_new_id_with_the_analytic_area(split: SplitOp) -> None:
    s, box = box_session()

    delta = split(s, box)

    assert isinstance(delta, ps.HistoryDelta)
    created = set(delta.created.tolist())
    table = s.entity_table(EntityKind.FACE)
    on_cut = [
        (int(i), float(m))
        for i, m, c in zip(table.ids, table.measure, table.centroid, strict=True)
        if abs(c[0] - CUT_X) <= COORD_ATOL and int(i) in created
    ]
    assert len(on_cut) == 1
    section, area = on_cut[0]
    assert area == pytest.approx(BOX_B * BOX_C, rel=EXACT_RTOL)
    assert s.shape_count(EntityId(section)) == 1


def test_split_by_faces_leaves_the_tool_face_in_the_model_untouched() -> None:
    s, box = box_session()
    tool = free_face_at_x(s, CUT_X)
    area = s.mass_properties([tool]).measure[0]

    s.split_by_faces([box], [tool])

    assert s.is_alive(tool)
    assert s.shape_count(tool) == 1
    assert s.mass_properties([tool]).measure[0] == pytest.approx(area, rel=EXACT_RTOL)
    assert area == pytest.approx((BOX_B + 2.0) * (BOX_C + 2.0), rel=EXACT_RTOL)


# ---- the L-shaped prism -------------------------------------------------------------- #

# The plane x + y = 2.5 through the point (1.25, 1.25, 0), normal (1, 1, 0).
L_ORIGIN: tuple[float, float, float] = (1.25, 1.25, 0.0)
L_NORMAL: tuple[float, float, float] = (1.0, 1.0, 0.0)
TIP_X: float = 59.0 / 24.0
TIP_Y: float = 7.0 / 12.0
L_TIPS: tuple[Piece, ...] = (((TIP_X, TIP_Y, 0.5), 1.0), ((TIP_Y, TIP_X, 0.5), 1.0))
# The corner: the L's first moment (area 5, centroid (1.1, 1.1)) less the two tips'.
CORNER_XY: float = (5.0 * 1.1 - TIP_X - TIP_Y) / 3.0
L_CORNER: Piece = ((CORNER_XY, CORNER_XY, 0.5), 3.0)


def _solid_pieces(s: Session) -> list[tuple[np.ndarray, float]]:
    return ordinal_geometry(s)[EntityKind.SOLID]


def test_an_l_prism_cut_through_both_legs_leaves_the_corner_and_two_tips() -> None:
    s, l_prism = l_prism_session()

    s.split_by_plane([l_prism], L_ORIGIN, L_NORMAL)

    got = _solid_pieces(s)
    assert len(got) == 3
    for piece in (L_CORNER, *L_TIPS):
        assert sum(_matches(c, m, piece) for c, m in got) == 1
    assert not s.is_alive(l_prism)
    for piece_id in s.entities(EntityKind.SOLID):
        assert s.origin(EntityId(int(piece_id))).sources.tolist() == [int(l_prism)]


def test_keeping_the_positive_half_of_an_l_prism_keeps_both_tips_with_new_ids() -> None:
    s, l_prism = l_prism_session()

    delta = s.keep_half_space([l_prism], L_ORIGIN, L_NORMAL, "positive")

    got = _solid_pieces(s)
    assert len(got) == 2
    for piece in L_TIPS:
        assert sum(_matches(c, m, piece) for c, m in got) == 1
    assert not s.is_alive(l_prism)
    assert int(l_prism) in delta.deleted.tolist()
    for piece_id in s.entities(EntityKind.SOLID):
        piece = EntityId(int(piece_id))
        assert s.name_of(piece).role is NameRole.GENERATED
        assert s.origin(piece).sources.tolist() == [int(l_prism)]


def test_keeping_the_negative_half_of_an_l_prism_keeps_the_corner_and_its_id() -> None:
    s, l_prism = l_prism_session()

    delta = s.keep_half_space([l_prism], L_ORIGIN, L_NORMAL, "negative")

    got = _solid_pieces(s)
    assert len(got) == 1
    assert _matches(*got[0], L_CORNER)
    assert s.is_alive(l_prism)
    assert s.shape_count(l_prism) == 1
    assert int(l_prism) in delta.modified.tolist()
    assert delta.deleted.tolist() == [
        i for i in delta.deleted.tolist() if not s.is_alive(EntityId(i))
    ]


# ---- keep_half_space on the box ------------------------------------------------------ #


@pytest.mark.parametrize(
    ("keep", "lo", "hi"),
    [("positive", CUT_X, BOX_A), ("negative", 0.0, CUT_X)],
)
def test_keeping_half_a_box_keeps_the_box_on_that_side_and_its_id(
    keep: str, lo: float, hi: float
) -> None:
    s, box = box_session()

    s.keep_half_space([box], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0), keep)  # type: ignore[arg-type]

    assert s.is_alive(box)
    measured = s.mass_properties([box])
    assert measured.measure[0] == pytest.approx(
        (hi - lo) * BOX_B * BOX_C, rel=EXACT_RTOL
    )
    np.testing.assert_allclose(
        measured.centroid[0],
        (0.5 * (lo + hi), 0.5 * BOX_B, 0.5 * BOX_C),
        rtol=EXACT_RTOL,
        atol=COORD_ATOL,
    )


def test_keeping_half_a_box_kills_the_face_on_the_removed_side() -> None:
    s, box = box_session()
    faces = s.bounding_boxes(EntityKind.FACE)
    far = next(
        EntityId(int(i))
        for i, row in zip(faces.ids.tolist(), faces.bbox, strict=True)
        if row[0] == 0.0 and row[3] == 0.0
    )

    delta = s.keep_half_space([box], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0), "positive")

    assert not s.is_alive(far)
    assert int(far) in delta.deleted.tolist()
    assert len(s.entities(EntityKind.FACE)) == 6


# ---- determinism ------------------------------------------------------------------- #


def _run_box(op: str) -> tuple[bytes, tuple[tuple[int, ...], ...]]:
    s, box = box_session()
    if op == "split_by_plane":
        s.split_by_plane([box], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0))
    elif op == "split_by_faces":
        s.split_by_faces([box], [free_face_at_x(s, CUT_X)])
    else:
        s.keep_half_space([box], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0), "positive")
    handoff = s.export_handoff(allow_aliases=True)
    arrays = tuple(
        tuple(int(i) for i in a)
        for a in (handoff.solid_id, handoff.face_id, handoff.edge_id, handoff.vertex_id)
    )
    return s.brep(), arrays


@pytest.mark.parametrize("op", ["split_by_faces", "split_by_plane", "keep_half_space"])
def test_two_fresh_sessions_give_identical_bytes_and_ids(op: str) -> None:
    first = _run_box(op)

    second = _run_box(op)

    assert first[0] == second[0]
    assert first[1] == second[1]


# ---- refusals ------------------------------------------------------------------------ #


def test_split_by_plane_refuses_a_box_on_one_side_with_its_signed_distances() -> None:
    s, box = box_session()

    with pytest.raises(
        ps.PysmeshError,
        match=rf"split_by_plane: solid {int(box)} lies entirely on one side of the plane "
        r"\(signed distance 4 to 7\); nothing to split",
    ):
        s.split_by_plane([box], (-4.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_split_by_plane_refuses_a_plane_that_only_touches_a_face() -> None:
    s, box = box_session()

    with pytest.raises(ps.PysmeshError, match=r"signed distance 0 to 3\)"):
        s.split_by_plane([box], (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("keep", "message"),
    [
        ("positive", r"side of the plane that is removed .*nothing would be kept"),
        ("negative", r"side of the plane that is kept .*nothing to remove"),
    ],
)
def test_keep_half_space_refuses_a_box_on_one_side(keep: str, message: str) -> None:
    s, box = box_session()

    with pytest.raises(ps.PysmeshError, match=rf"keep_half_space: solid {int(box)} .*"):
        s.keep_half_space([box], (5.0, 0.0, 0.0), (1.0, 0.0, 0.0), keep)  # type: ignore[arg-type]
    with pytest.raises(ps.PysmeshError, match=message):
        s.keep_half_space([box], (5.0, 0.0, 0.0), (1.0, 0.0, 0.0), keep)  # type: ignore[arg-type]


def test_keep_half_space_refuses_an_unknown_side() -> None:
    s, box = box_session()

    with pytest.raises(
        ps.PysmeshError, match=r"keep_half_space: the side to keep is 'up'"
    ):
        s.keep_half_space([box], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0), "up")  # type: ignore[arg-type]


@pytest.mark.parametrize("op", ["split_by_plane", "keep_half_space"])
def test_a_zero_normal_is_refused_with_its_value(op: str) -> None:
    s, box = box_session()
    args: tuple[object, ...] = ([box], (CUT_X, 0.0, 0.0), (0.0, 0.0, 0.0))
    if op == "keep_half_space":
        args = (*args, "positive")

    with pytest.raises(
        ps.PysmeshError, match=rf"{op}: the plane normal \(0, 0, 0\) has zero length"
    ):
        getattr(s, op)(*args)


@pytest.mark.parametrize("op", ["split_by_faces", "split_by_plane", "keep_half_space"])
def test_an_empty_target_list_is_refused(op: str) -> None:
    s, _ = box_session()
    tool = free_face_at_x(s, CUT_X)
    args: dict[str, tuple[object, ...]] = {
        "split_by_faces": ([], [tool]),
        "split_by_plane": ([], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "keep_half_space": ([], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0), "positive"),
    }

    with pytest.raises(ps.PysmeshError, match=rf"{op}: no target solid is named"):
        getattr(s, op)(*args[op])


@pytest.mark.parametrize("op", ["split_by_faces", "split_by_plane", "keep_half_space"])
def test_a_dead_target_is_refused_naming_it(op: str) -> None:
    s, box = box_session()
    tool = free_face_at_x(s, CUT_X)
    s.remove([box])
    args: dict[str, tuple[object, ...]] = {
        "split_by_faces": ([box], [tool]),
        "split_by_plane": ([box], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "keep_half_space": ([box], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0), "positive"),
    }

    with pytest.raises(ps.PysmeshError, match=rf"{op}: entity {int(box)} is dead"):
        getattr(s, op)(*args[op])


@pytest.mark.parametrize("op", ["split_by_faces", "split_by_plane", "keep_half_space"])
def test_a_face_named_as_a_target_is_refused_naming_its_kind(op: str) -> None:
    s, _ = box_session()
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))
    tool = free_face_at_x(s, CUT_X)
    args: dict[str, tuple[object, ...]] = {
        "split_by_faces": ([face], [tool]),
        "split_by_plane": ([face], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "keep_half_space": ([face], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0), "positive"),
    }

    with pytest.raises(
        ps.PysmeshError, match=rf"{op}: targets entity {int(face)} is a FACE"
    ):
        getattr(s, op)(*args[op])


def test_split_by_faces_refuses_an_empty_tool_list() -> None:
    s, box = box_session()

    with pytest.raises(ps.PysmeshError, match=r"split_by_faces: no tool face is named"):
        s.split_by_faces([box], [])


def test_split_by_faces_refuses_a_solid_named_as_a_tool() -> None:
    s, box = box_session()
    s.add_box(1.0, 1.0, 1.0, origin=(1.0, 1.0, 1.0))
    other = EntityId(int(s.entities(EntityKind.SOLID)[-1]))

    with pytest.raises(
        ps.PysmeshError,
        match=rf"split_by_faces: tool {int(other)} is a SOLID, not a FACE",
    ):
        s.split_by_faces([box], [other])


def test_split_by_faces_refuses_a_face_of_the_target_itself() -> None:
    s, box = box_session()
    own = EntityId(int(s.entities(EntityKind.FACE)[0]))

    with pytest.raises(
        ps.PysmeshError,
        match=rf"split_by_faces: face {int(own)} is a face of a target solid's own body",
    ):
        s.split_by_faces([box], [own])


def test_split_by_faces_refuses_a_face_that_misses_the_target() -> None:
    s, box = box_session()
    tool = free_face_at_x(s, BOX_A + 5.0)

    with pytest.raises(
        ps.PysmeshError,
        match=rf"split_by_faces: face {int(tool)} divide none of the solid {int(box)}; "
        "nothing to split",
    ):
        s.split_by_faces([box], [tool])


@pytest.mark.parametrize("op", ["split_by_faces", "split_by_plane", "keep_half_space"])
def test_a_refused_split_leaves_the_session_exactly_as_it_was(op: str) -> None:
    s, box = box_session()
    tool = free_face_at_x(s, BOX_A + 5.0)
    before = (s.brep(), id_space(s), s.op_count, s.issued_id_count)
    args: dict[str, tuple[object, ...]] = {
        "split_by_faces": ([box], [tool]),
        "split_by_plane": ([box], (BOX_A + 5.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "keep_half_space": (
            [box],
            (BOX_A + 5.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            "positive",
        ),
    }

    with pytest.raises(ps.PysmeshError):
        getattr(s, op)(*args[op])

    assert (s.brep(), id_space(s), s.op_count, s.issued_id_count) == before


# ---- cancel -------------------------------------------------------------------------- #


@pytest.mark.parametrize("op", ["split_by_faces", "split_by_plane", "keep_half_space"])
def test_a_cancel_at_the_first_callback_leaves_the_session_exactly_as_it_was(
    op: str,
) -> None:
    s, box = box_session()
    tool = free_face_at_x(s, CUT_X)
    before = (s.brep(), id_space(s), s.op_count, s.issued_id_count)
    args: dict[str, tuple[object, ...]] = {
        "split_by_faces": ([box], [tool]),
        "split_by_plane": ([box], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "keep_half_space": ([box], (CUT_X, 0.0, 0.0), (1.0, 0.0, 0.0), "positive"),
    }

    with pytest.raises(PysmeshCancelled):
        getattr(s, op)(*args[op], cancel=lambda: True)

    assert (s.brep(), id_space(s), s.op_count, s.issued_id_count) == before


# ---- split is unchanged -------------------------------------------------------------- #


def test_split_by_a_solid_keeps_the_target_id_on_both_pieces_as_before() -> None:
    """``split`` keeps the five rules: the cut target's id survives on both pieces."""
    s, box = box_session()
    s.add_box(1.0, BOX_B + 2.0, BOX_C + 2.0, origin=(CUT_X, -1.0, -1.0))
    tool = EntityId(int(s.entities(EntityKind.SOLID)[-1]))
    name = s.name_of(box)

    delta = s.split([box], [tool])

    assert s.is_alive(box)
    assert s.shape_count(box) == 3
    assert s.resolve(name).status is ResolutionStatus.AMBIGUOUS
    assert int(box) in delta.split.tolist()
    assert s.shape_count(tool) == 1
    assert s.mass_properties([box]).measure[0] == pytest.approx(
        BOX_A * BOX_B * BOX_C, rel=EXACT_RTOL
    )
