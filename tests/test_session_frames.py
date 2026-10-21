# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for the construction queries: face_frame, tangent_frame, face_face_axis and
point_at_length.

Each answer is checked against a closed form:

* The faces of the box ``[0, 3] x [0, 7] x [0, 11]`` are axis-aligned rectangles: the
  centroid of each is the middle of its bounds and its outward normal points away from
  the box centre ``(1.5, 3.5, 5.5)``.
* A cylinder of radius ``r`` on the frame ``(O, X, Y, A)`` is
  ``S(u, v) = O + r (cos u X + sin u Y) + v A``, so its outward normal is
  ``cos u X + sin u Y`` and ``dS/du / |dS/du| = -sin u X + cos u Y``.
* Two planes with normals ``n_a``, ``n_b`` meet in the line of direction
  ``n_a x n_b / |n_a x n_b|``; for the box sides ``x = 3`` and ``y = 7`` that is the edge
  ``(3, 7, z)``, along ``+z``.
* A straight edge from ``p`` to ``q`` has its point at length ``s`` at
  ``p + s (q - p) / |q - p|``; a circle of radius ``r`` turns ``s / r`` radians over a
  length ``s``.

Analytic geometry evaluates to round-off, so values are compared at 1e-12 relative, with
an absolute floor of 1e-12 times the model size for components near zero.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, Session

BOX_A: float = 3.0
BOX_B: float = 7.0
BOX_C: float = 11.0
BOX_CENTRE: np.ndarray = np.array([BOX_A, BOX_B, BOX_C]) / 2.0

RTOL: float = 1e-12
ATOL: float = 1e-12 * BOX_C


def box_session() -> Session:
    """A session holding the 3 x 7 x 11 box at the origin."""
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    return s


def face_on_plane(s: Session, axis: int, value: float) -> EntityId:
    """The live face whose bounds are the single value ``value`` on coordinate ``axis``."""
    table = s.bounding_boxes(EntityKind.FACE)
    for i, row in zip(table.ids.tolist(), table.bbox, strict=True):
        if row[axis] == value and row[axis + 3] == value:
            return EntityId(int(i))
    raise AssertionError(f"no face on plane {'xyz'[axis]} = {value}")


def counters(s: Session) -> tuple[int, int]:
    """The operation and id counters a query must not move."""
    return s.op_count, s.issued_id_count


# ---- face_frame -------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("axis", "value"),
    [(0, 0.0), (0, BOX_A), (1, 0.0), (1, BOX_B), (2, 0.0), (2, BOX_C)],
)
def test_face_frame_of_each_box_side_is_its_centre_and_outward_normal(
    axis: int, value: float
) -> None:
    s = box_session()
    face = face_on_plane(s, axis, value)

    frame = s.face_frame(face)

    want_origin = BOX_CENTRE.copy()
    want_origin[axis] = value
    want_normal = np.zeros(3)
    want_normal[axis] = 1.0 if value > 0.0 else -1.0
    np.testing.assert_allclose(frame.origin, want_origin, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(frame.normal, want_normal, rtol=RTOL, atol=ATOL)


def test_face_frame_covers_reversed_faces_and_keeps_the_planes_x_direction() -> None:
    s = box_session()
    faces = [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    params = s.surface_parameters(faces)
    assert params.reversed.any() and not params.reversed.all()

    for row, face in enumerate(faces):
        frame = s.face_frame(face)

        np.testing.assert_allclose(
            frame.x_direction, params.ref_dir[row], rtol=RTOL, atol=ATOL
        )
        assert float(np.dot(frame.x_direction, frame.normal)) == pytest.approx(
            0.0, abs=ATOL
        )
        assert float(np.linalg.norm(frame.x_direction)) == pytest.approx(1.0, rel=RTOL)
        sign = -1.0 if params.reversed[row] else 1.0
        np.testing.assert_allclose(
            frame.normal, sign * params.axis[row], rtol=RTOL, atol=ATOL
        )


def test_face_frame_origin_of_an_l_shaped_face_is_its_area_centroid() -> None:
    """The L ``[0, 3] x [0, 1]`` plus ``[0, 1] x [1, 3]``: area 5, centroid (1.1, 1.1)."""
    s = Session()
    s.add_polyline(
        [
            (0.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 3.0, 0.0),
            (0.0, 3.0, 0.0),
        ],
        closed=True,
    )
    s.make_face([EntityId(int(i)) for i in s.entities(EntityKind.EDGE)])
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    frame = s.face_frame(face)

    np.testing.assert_allclose(frame.origin, (1.1, 1.1, 0.0), rtol=RTOL, atol=1e-12)
    assert abs(float(frame.normal[2])) == pytest.approx(1.0, rel=RTOL)


def test_face_frame_refuses_a_cylindrical_face_naming_its_type() -> None:
    s = Session()
    s.add_cylinder(2.0, 5.0)
    params = s.surface_parameters(
        [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    )
    side = EntityId(int(params.ids[list(params.types).index("Cylinder")]))

    with pytest.raises(
        ps.PysmeshError,
        match=rf"face_frame: face {int(side)} is a Cylinder face, not a planar one",
    ):
        s.face_frame(side)


def test_face_frame_issues_no_id_and_moves_no_counter() -> None:
    s = box_session()
    face = face_on_plane(s, 0, BOX_A)
    before = counters(s)

    s.face_frame(face)

    assert counters(s) == before


# ---- tangent_frame ----------------------------------------------------------------- #


def cylinder_side(s: Session) -> EntityId:
    """The lateral face of the session's only cylinder."""
    params = s.surface_parameters(
        [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    )
    return EntityId(int(params.ids[list(params.types).index("Cylinder")]))


def test_tangent_frame_on_a_cylinder_is_the_closed_form() -> None:
    radius, height = 2.0, 5.0
    s = Session()
    s.add_cylinder(radius, height)
    side = cylinder_side(s)
    p = s.surface_parameters([side])
    o, a, x = p.origin[0], p.axis[0], p.ref_dir[0]
    y = np.cross(a, x)
    u = np.array([0.0, 0.3, 1.7, math.pi, 5.9])
    v = np.array([0.0, 1.0, 2.5, 4.0, height])

    sample = s.tangent_frame(side, u, v)

    cu, su = np.cos(u)[:, None], np.sin(u)[:, None]
    want_points = o + radius * (cu * x + su * y) + v[:, None] * a
    np.testing.assert_allclose(sample.points, want_points, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(sample.normals, cu * x + su * y, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(sample.tangents, -su * x + cu * y, rtol=RTOL, atol=ATOL)


def test_tangent_frame_on_a_box_side_is_its_plane_frame() -> None:
    s = box_session()
    face = face_on_plane(s, 1, BOX_B)
    params = s.surface_parameters([face])
    bounds = s.face_parameter_bounds([face])[0]
    u = np.array([bounds[0], 0.5 * (bounds[0] + bounds[1]), bounds[1]])
    v = np.array([bounds[2], 0.5 * (bounds[2] + bounds[3]), bounds[3]])

    sample = s.tangent_frame(face, u, v)

    np.testing.assert_allclose(
        sample.tangents, np.tile(params.ref_dir[0], (3, 1)), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        sample.normals, np.tile((0.0, 1.0, 0.0), (3, 1)), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        sample.points, s.surface_at(face, np.column_stack([u, v])).points, rtol=RTOL
    )


def test_tangent_frame_refuses_the_pole_of_a_sphere_naming_the_face_and_uv() -> None:
    s = Session()
    s.add_sphere(2.0)
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    with pytest.raises(
        ps.PysmeshError,
        match=rf"tangent_frame: face {int(face)}: \(u, v\) = \(0\.5, 1\.5708\) is a "
        "singular point",
    ):
        s.tangent_frame(face, [0.5], [0.5 * math.pi])


def test_tangent_frame_refuses_a_parameter_outside_the_bounds() -> None:
    s = Session()
    s.add_cylinder(2.0, 5.0)
    side = cylinder_side(s)

    with pytest.raises(
        ps.PysmeshError,
        match=rf"tangent_frame: face {int(side)}: \(u, v\) = \(1, 6\) lies outside the "
        "face's parameter bounds",
    ):
        s.tangent_frame(side, [0.0, 1.0], [1.0, 6.0])


def test_tangent_frame_refuses_u_and_v_of_different_lengths() -> None:
    s = box_session()
    face = face_on_plane(s, 0, 0.0)

    with pytest.raises(
        ps.PysmeshError, match=r"tangent_frame: 2 u values and 1 v values"
    ):
        s.tangent_frame(face, [0.0, 1.0], [0.0])


def test_tangent_frame_issues_no_id_and_moves_no_counter() -> None:
    s = Session()
    s.add_cylinder(2.0, 5.0)
    side = cylinder_side(s)
    before = counters(s)

    s.tangent_frame(side, [0.5], [1.0])

    assert counters(s) == before


# ---- face_face_axis ---------------------------------------------------------------- #


def test_the_axis_of_two_adjacent_box_sides_is_their_common_edge() -> None:
    s = box_session()
    a = face_on_plane(s, 0, BOX_A)
    b = face_on_plane(s, 1, BOX_B)

    axis = s.face_face_axis(a, b)

    # n_a x n_b = (+x) x (+y) = +z; the centroid midpoint (2.25, 5.25, 5.5) projects to
    # (3, 7, 5.5).
    np.testing.assert_allclose(axis.direction, (0.0, 0.0, 1.0), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(axis.point, (BOX_A, BOX_B, 5.5), rtol=RTOL, atol=ATOL)


def test_the_axis_of_two_box_sides_follows_the_order_of_the_faces() -> None:
    s = box_session()
    a = face_on_plane(s, 0, 0.0)
    b = face_on_plane(s, 2, BOX_C)

    forward = s.face_face_axis(a, b)
    backward = s.face_face_axis(b, a)

    # n_a x n_b = (-x) x (+z) = +y. The centroids (0, 3.5, 5.5) and (1.5, 3.5, 11) have
    # their midpoint (0.75, 3.5, 8.25), which projects to (0, 3.5, 11).
    np.testing.assert_allclose(forward.direction, (0.0, 1.0, 0.0), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        backward.direction, (0.0, -1.0, 0.0), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(forward.point, (0.0, 3.5, BOX_C), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(backward.point, forward.point, rtol=RTOL, atol=ATOL)


def test_the_axis_of_two_oblique_rectangles_is_the_line_of_their_planes() -> None:
    """Planes z = 0 and y + z = 1 meet in the line (t, 1, 0)."""
    s = Session()
    s.add_rectangle((-2.0, -2.0, 0.0), (0.0, 0.0, 1.0), 4.0, 4.0)
    s.add_rectangle((0.5, 0.0, 1.0), (0.0, 1.0, 1.0), 3.0, 2.0)
    a, b = (EntityId(int(i)) for i in s.entities(EntityKind.FACE))
    na, nb = s.face_frame(a).normal, s.face_frame(b).normal
    ca, cb = s.mass_properties([a, b]).centroid

    axis = s.face_face_axis(a, b)

    want = np.cross(na, nb)
    want /= np.linalg.norm(want)
    np.testing.assert_allclose(axis.direction, want, rtol=RTOL, atol=ATOL)
    assert abs(float(want[0])) == pytest.approx(1.0, rel=RTOL)
    mid_x = 0.5 * (ca[0] + cb[0])
    np.testing.assert_allclose(axis.point, (mid_x, 1.0, 0.0), rtol=RTOL, atol=ATOL)


def test_face_face_axis_refuses_parallel_faces_naming_both_and_the_angle() -> None:
    s = box_session()
    a = face_on_plane(s, 0, 0.0)
    b = face_on_plane(s, 0, BOX_A)

    with pytest.raises(
        ps.PysmeshError,
        match=rf"face_face_axis: faces {int(a)} and {int(b)} lie in parallel planes "
        r"\(angle 0 rad between them\)",
    ):
        s.face_face_axis(a, b)


def test_face_face_axis_refuses_a_curved_face_naming_its_type() -> None:
    s = Session()
    s.add_cylinder(2.0, 5.0)
    side = cylinder_side(s)
    cap = next(
        EntityId(int(i)) for i in s.entities(EntityKind.FACE) if int(i) != int(side)
    )

    with pytest.raises(
        ps.PysmeshError,
        match=rf"face_face_axis: face {int(side)} is a Cylinder face, not a planar one",
    ):
        s.face_face_axis(cap, side)


def test_face_face_axis_issues_no_id_and_moves_no_counter() -> None:
    s = box_session()
    a = face_on_plane(s, 0, BOX_A)
    b = face_on_plane(s, 1, BOX_B)
    before = counters(s)

    s.face_face_axis(a, b)

    assert counters(s) == before


# ---- point_at_length --------------------------------------------------------------- #


def line_session() -> tuple[Session, EntityId]:
    """A free straight edge from (0, 0, 0) to (3, 4, 0): length 5."""
    s = Session()
    s.add_line((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
    return s, EntityId(int(s.entities(EntityKind.EDGE)[0]))


@pytest.mark.parametrize(
    ("length", "from_start", "want"),
    [
        (0.0, True, (0.0, 0.0, 0.0)),
        (2.0, True, (1.2, 1.6, 0.0)),
        (5.0, True, (3.0, 4.0, 0.0)),
        (2.0, False, (1.8, 2.4, 0.0)),
        (5.0, False, (0.0, 0.0, 0.0)),
    ],
)
def test_point_at_length_on_a_line_is_the_closed_form(
    length: float, from_start: bool, want: tuple[float, float, float]
) -> None:
    s, edge = line_session()
    first, last = s.edge_parameter_bounds([edge])[0]

    at = s.point_at_length(edge, length, from_start=from_start)

    np.testing.assert_allclose(at.point, want, rtol=RTOL, atol=1e-12)
    np.testing.assert_allclose(
        s.curve_at(edge, [at.parameter]).points[0], at.point, rtol=RTOL, atol=1e-12
    )
    assert first <= at.parameter <= last


def test_point_at_length_on_a_circle_turns_length_over_radius() -> None:
    radius = 2.0
    s = Session()
    s.add_circle((1.0, -1.0, 0.5), (0.0, 0.0, 1.0), radius)
    edge = EntityId(int(s.entities(EntityKind.EDGE)[0]))
    first, _ = s.edge_parameter_bounds([edge])[0]

    at = s.point_at_length(edge, math.pi, from_start=True)

    assert at.parameter == pytest.approx(first + math.pi / radius, rel=RTOL)
    np.testing.assert_allclose(
        at.point, s.curve_at(edge, [first + math.pi / radius]).points[0], rtol=RTOL
    )


def test_point_at_length_from_either_end_meets_on_a_spline() -> None:
    """On any edge the point at s from the start is the point at L - s from the end.

    The two lengths are integrated from opposite ends of the B-spline by
    GCPnts_AbscissaPoint, each to 1e-10 of the edge's length (about 5), so the two points
    agree to 1e-8, not to round-off.
    """
    s = Session()
    s.add_spline(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.5], [3.0, 1.5, 1.0], [4.0, 3.0, 0.0]])
    )
    edge = EntityId(int(s.entities(EntityKind.EDGE)[0]))
    length = float(s.mass_properties([edge], precision=1e-12).measure[0])

    near = s.point_at_length(edge, 0.3 * length)
    far = s.point_at_length(edge, 0.7 * length, from_start=False)

    np.testing.assert_allclose(near.point, far.point, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize("length", [-0.5, 5.25])
def test_point_at_length_refuses_a_length_off_the_edge_with_both_numbers(
    length: float,
) -> None:
    s, edge = line_session()

    with pytest.raises(
        ps.PysmeshError,
        match=rf"point_at_length: length {length:g} is outside edge {int(edge)}, whose "
        "length is 5",
    ):
        s.point_at_length(edge, length)


def test_point_at_length_refuses_a_face() -> None:
    s = box_session()
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    with pytest.raises(
        ps.PysmeshError, match=rf"point_at_length: entity {int(face)} is a"
    ):
        s.point_at_length(face, 1.0)


def test_point_at_length_issues_no_id_and_moves_no_counter() -> None:
    s, edge = line_session()
    before = counters(s)

    s.point_at_length(edge, 2.0)

    assert counters(s) == before
