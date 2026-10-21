# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for position_along_path.

The move maps the path's frame at its start onto its frame at the fraction ``d`` of its
length. The frame is OCCT's corrected Frenet frame: origin the curve point, z the unit
tangent, x the principal normal (towards the centre of curvature).

* Straight path from ``A`` to ``B``: the frame does not turn, so the move is the
  translation by ``d (B - A)``.
* Quarter circle of radius ``R`` about the z axis, counter-clockwise from ``(R, 0, 0)``:
  the frame at angle ``t`` is the start frame turned by ``t`` about z, so the move is the
  rotation by ``d pi / 2`` about the z axis.
* Ellipse ``(a cos t, b sin t, 0)``: the arc length ``s(t)`` is the integral of
  ``sqrt(a^2 sin^2 t + b^2 cos^2 t)``. The start point lands on ``P(t*)`` with
  ``s(t*) = d L``; a point 1 outside the start lands 1 outside ``P(t*)``, along the outward
  normal ``(cos t / a, sin t / b)`` normalised; a point 1 above lands 1 above.

``reverse`` measures the fraction from the far end, as GEOM's ``PositionAlongPath``: the
move goes to the fraction ``1 - d``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import EntityId, EntityKind, Session

R: float = 5.0
ELLIPSE_A: float = 5.0
ELLIPSE_B: float = 2.0
RTOL: float = 1e-12
ATOL: float = 1e-12 * R

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, per kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def new_ids(s: Session, kind: EntityKind, before: set[int]) -> list[EntityId]:
    """The ids of ``kind`` that are alive now and were not in ``before``."""
    return [EntityId(int(i)) for i in s.entities(kind) if int(i) not in before]


def add_edge_path(s: Session, add: str, *args: object) -> EntityId:
    """Call ``s.<add>(*args)`` and return the one edge it adds."""
    before = {int(i) for i in s.entities(EntityKind.EDGE)}
    getattr(s, add)(*args)
    (edge,) = new_ids(s, EntityKind.EDGE, before)
    return edge


def add_box_body(s: Session, origin: tuple[float, float, float]) -> EntityId:
    """A 1 x 2 x 3 box at ``origin``; its solid id."""
    before = {int(i) for i in s.entities(EntityKind.SOLID)}
    s.add_box(1.0, 2.0, 3.0, origin=origin)
    (solid,) = new_ids(s, EntityKind.SOLID, before)
    return solid


def add_point(s: Session, point: tuple[float, float, float]) -> EntityId:
    """A free vertex; its id."""
    before = {int(i) for i in s.entities(EntityKind.VERTEX)}
    s.add_vertex(point)
    (vertex,) = new_ids(s, EntityKind.VERTEX, before)
    return vertex


def related(
    s: Session, kind: EntityKind, other: EntityKind, entity: EntityId
) -> list[EntityId]:
    """The ids of ``other`` that ``entity`` (of ``kind``) is related to by adjacency."""
    pairs = s.adjacency(kind, other)
    return [EntityId(int(r)) for r in pairs.related[pairs.ids == int(entity)]]


def rotation_z(angle: float) -> NDArray[np.float64]:
    """The 3 x 3 rotation by ``angle`` about the z axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def quarter_arc_session() -> tuple[Session, EntityId, EntityId]:
    """A quarter circle of radius R about the z axis from (R, 0, 0), and a box."""
    s = Session()
    h = R / math.sqrt(2.0)
    path = add_edge_path(s, "add_arc", (R, 0.0, 0.0), (h, h, 0.0), (0.0, R, 0.0))
    box = add_box_body(s, (R + 1.0, 0.5, -1.0))
    return s, path, box


def body_tables(s: Session) -> dict[EntityKind, ps.EntityTable]:
    """The entity table of every kind."""
    return {k: s.entity_table(k) for k in _KINDS}


def ellipse_arc_length(t: float) -> float:
    """Arc length of the ellipse from parameter 0 to ``t``, by 64-point Gauss-Legendre.

    The integrand is analytic, so 64 nodes on [0, t] (t <= 2 pi) reach round-off.
    """
    nodes, weights = np.polynomial.legendre.leggauss(64)
    u = 0.5 * t * (nodes + 1.0)
    speed = np.hypot(ELLIPSE_A * np.sin(u), ELLIPSE_B * np.cos(u))
    return float(0.5 * t * np.dot(weights, speed))


def ellipse_parameter_at(length: float) -> float:
    """The ellipse parameter ``t`` with arc length ``length`` from t = 0, by Newton."""
    t = length / ELLIPSE_A
    for _ in range(50):
        step = (ellipse_arc_length(t) - length) / math.hypot(
            ELLIPSE_A * math.sin(t), ELLIPSE_B * math.cos(t)
        )
        t -= step
        if abs(step) < 1e-15:
            break
    return t


# ---- analytic ---------------------------------------------------------------------- #


def test_a_straight_path_translates_the_body_by_the_fraction_of_its_length() -> None:
    s = Session()
    path = add_edge_path(s, "add_line", (0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    box = add_box_body(s, (0.0, 3.0, 0.0))
    before = s.mass_properties([box])

    s.position_along_path(path, 0.3, [box])

    after = s.mass_properties([box])
    np.testing.assert_allclose(
        after.centroid[0], before.centroid[0] + (3.0, 0.0, 0.0), rtol=RTOL, atol=ATOL
    )
    assert after.measure[0] == pytest.approx(before.measure[0], rel=RTOL)


@pytest.mark.parametrize("distance", [0.25, 0.5, 1.0])
def test_a_quarter_arc_turns_the_body_about_the_arc_axis(distance: float) -> None:
    s, path, box = quarter_arc_session()
    before = s.mass_properties([box]).centroid[0]

    s.position_along_path(path, distance, [box])

    expected = rotation_z(distance * 0.5 * math.pi) @ before
    np.testing.assert_allclose(
        s.mass_properties([box]).centroid[0], expected, rtol=RTOL, atol=ATOL
    )


def test_reverse_moves_to_the_fraction_measured_from_the_far_end() -> None:
    s, path, box = quarter_arc_session()
    before = s.mass_properties([box]).centroid[0]

    s.position_along_path(path, 0.25, [box], reverse=True)

    expected = rotation_z(0.75 * 0.5 * math.pi) @ before
    np.testing.assert_allclose(
        s.mass_properties([box]).centroid[0], expected, rtol=RTOL, atol=ATOL
    )


def test_an_ellipse_path_places_points_at_the_fraction_of_its_arc_length() -> None:
    """The start point, a point 1 outside it and a point 1 above it, at d = 0.3.

    OCCT integrates the length to 1e-10 of it (about 2.3e-9 on this ellipse of length
    23.0), so the positions are compared at 1e-9 of the length.
    """
    s = Session()
    path = add_edge_path(
        s, "add_ellipse", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), ELLIPSE_A, ELLIPSE_B
    )
    on = add_point(s, (ELLIPSE_A, 0.0, 0.0))
    outside = add_point(s, (ELLIPSE_A + 1.0, 0.0, 0.0))
    above = add_point(s, (ELLIPSE_A, 0.0, 1.0))
    total = ellipse_arc_length(2.0 * math.pi)
    t = ellipse_parameter_at(0.3 * total)
    curve_point = np.array(
        [ELLIPSE_A * math.cos(t), ELLIPSE_B * math.sin(t), 0.0], dtype=np.float64
    )
    normal = np.array(
        [math.cos(t) / ELLIPSE_A, math.sin(t) / ELLIPSE_B, 0.0], dtype=np.float64
    )
    normal /= np.linalg.norm(normal)

    s.position_along_path(path, 0.3, [on, outside, above])

    points = s.entity_table(EntityKind.VERTEX)
    where = {int(i): points.centroid[k] for k, i in enumerate(points.ids)}
    atol = 1e-9 * total
    np.testing.assert_allclose(where[int(on)], curve_point, rtol=0.0, atol=atol)
    np.testing.assert_allclose(
        where[int(outside)], curve_point + normal, rtol=0.0, atol=atol
    )
    np.testing.assert_allclose(
        where[int(above)], curve_point + (0.0, 0.0, 1.0), rtol=0.0, atol=atol
    )


def test_the_path_starts_at_the_edges_first_vertex_in_its_own_orientation() -> None:
    """Along each of a box's 12 straight edges a point moves by ``d (end - start)``.

    ``start`` and ``end`` are the points that point_at_length reports at 0 and at the
    edge's length. Some edges are held REVERSED, so their start is the curve's last
    point: the test checks that at least one is.
    """
    s = Session()
    box = add_box_body(s, (0.0, 0.0, 0.0))
    point = add_point(s, (7.0, 8.0, 9.0))
    edges = related(s, EntityKind.SOLID, EntityKind.EDGE, box)
    reversed_edges = 0
    for edge in edges:
        length = float(s.mass_properties([edge]).measure[0])
        start = s.point_at_length(edge, 0.0).point
        end = s.point_at_length(edge, length).point
        first = s.curve_at(edge, [s.edge_parameter_bounds([edge])[0][0]]).points[0]
        reversed_edges += int(not np.allclose(start, first, rtol=0.0, atol=ATOL))
        before = s.mass_properties([point]).centroid[0]

        s.position_along_path(edge, 0.25, [point])

        np.testing.assert_allclose(
            s.mass_properties([point]).centroid[0],
            before + 0.25 * (end - start),
            rtol=RTOL,
            atol=ATOL,
        )
    assert len(edges) == 12
    assert reversed_edges > 0


# ---- history against geometry -------------------------------------------------------- #


def test_every_id_survives_on_its_own_sub_shape_turned_with_the_body() -> None:
    """Each id of the box keeps its measure; its centroid turns by 45 degrees about z."""
    s, path, box = quarter_arc_session()
    ids_before = id_space(s)
    tables = body_tables(s)
    path_ids = {int(path)} | {
        int(v) for v in related(s, EntityKind.EDGE, EntityKind.VERTEX, path)
    }
    turn = rotation_z(0.25 * math.pi)

    delta = s.position_along_path(path, 0.5, [box])

    assert id_space(s) == ids_before
    assert delta.created.size == 0
    assert delta.deleted.size == 0
    moved = sorted(i for ids in ids_before for i in ids if i not in path_ids)
    assert sorted(delta.modified.tolist()) == moved
    for kind, table in tables.items():
        after = s.entity_table(kind)
        np.testing.assert_array_equal(after.ids, table.ids)
        np.testing.assert_allclose(after.measure, table.measure, rtol=RTOL)
        for k, i in enumerate(table.ids):
            expected = (
                table.centroid[k] if int(i) in path_ids else turn @ table.centroid[k]
            )
            np.testing.assert_allclose(
                after.centroid[k], expected, rtol=RTOL, atol=ATOL
            )


# ---- determinism --------------------------------------------------------------------- #


def _placed() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
    s, path, box = quarter_arc_session()
    s.position_along_path(path, 0.5, [box])
    return s.brep(), id_space(s)


def test_two_fresh_sessions_give_identical_bytes_and_ids() -> None:
    first = _placed()

    second = _placed()

    assert first == second


# ---- refusals ------------------------------------------------------------------------ #


def assert_refused(s: Session, pattern: str, *args: object, **kwargs: object) -> None:
    """The call raises a PysmeshError matching ``pattern`` and leaves the session as is."""
    before = (s.brep(), id_space(s), s.op_count, s.issued_id_count)

    with pytest.raises(ps.PysmeshError, match=pattern):
        s.position_along_path(*args, **kwargs)  # type: ignore[arg-type]

    assert (s.brep(), id_space(s), s.op_count, s.issued_id_count) == before


def test_a_path_edge_of_a_moved_body_is_refused_naming_edge_and_body() -> None:
    s, _, box = quarter_arc_session()
    edge = related(s, EntityKind.SOLID, EntityKind.EDGE, box)[0]

    assert_refused(
        s,
        rf"position_along_path: path edge {int(edge)} belongs to solid {int(box)}, which "
        rf"entity {int(box)} moves; a path cannot move with the bodies it places",
        edge,
        0.5,
        [box],
    )


def test_a_free_path_edge_named_as_moved_is_refused() -> None:
    s, path, box = quarter_arc_session()

    assert_refused(
        s,
        rf"position_along_path: path edge {int(path)} is itself a free edge body, which "
        rf"entity {int(path)} moves",
        path,
        0.5,
        [box, path],
    )


@pytest.mark.parametrize("distance", [-0.1, 1.5])
def test_a_distance_outside_zero_to_one_is_refused_with_its_value(
    distance: float,
) -> None:
    s, path, box = quarter_arc_session()

    assert_refused(
        s,
        rf"position_along_path: distance {distance:g} along path edge {int(path)} is "
        r"outside \[0, 1\]",
        path,
        distance,
        [box],
    )


def test_a_non_finite_distance_is_refused() -> None:
    s, path, box = quarter_arc_session()

    assert_refused(s, r"position_along_path.*distance", path, math.nan, [box])


def test_the_start_of_the_path_is_refused_as_a_move_that_changes_nothing() -> None:
    s, path, box = quarter_arc_session()

    assert_refused(
        s,
        rf"position_along_path: at fraction 0 of path edge {int(path)} the frame is the "
        r"start frame; the move would change nothing",
        path,
        0.0,
        [box],
    )


def test_the_end_of_a_closed_path_is_refused_as_a_move_that_changes_nothing() -> None:
    s = Session()
    path = add_edge_path(s, "add_circle", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), R)
    box = add_box_body(s, (R + 1.0, 0.0, 0.0))

    assert_refused(
        s,
        rf"position_along_path: at fraction 1 of path edge {int(path)} the frame is the "
        r"start frame",
        path,
        1.0,
        [box],
    )


def test_an_empty_entity_list_is_refused() -> None:
    s, path, _ = quarter_arc_session()

    assert_refused(s, r"position_along_path: no entity is named", path, 0.5, [])


def test_a_dead_entity_is_refused() -> None:
    s, path, box = quarter_arc_session()
    s.remove([box])

    assert_refused(
        s, rf"position_along_path: entity {int(box)} is dead", path, 0.5, [box]
    )


def test_a_dead_path_edge_is_refused() -> None:
    s, path, box = quarter_arc_session()
    s.remove([path])

    assert_refused(
        s, rf"position_along_path: entity {int(path)} is dead", path, 0.5, [box]
    )


def test_a_path_that_is_not_an_edge_is_refused_naming_its_kind() -> None:
    s, _, box = quarter_arc_session()
    face = related(s, EntityKind.SOLID, EntityKind.FACE, box)[0]
    other = add_box_body(s, (-10.0, 0.0, 0.0))

    assert_refused(
        s,
        rf"position_along_path: entity {int(face)} is a FACE, not an EDGE",
        face,
        0.5,
        [other],
    )
