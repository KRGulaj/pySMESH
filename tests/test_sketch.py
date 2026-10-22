# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-22

"""Gates for pysmesh.sketch.solve: analytic sketches, the diagnosis, CAD, determinism, time.

Each analytic sketch has a closed form, and its solved geometry is compared with it at
1e-10 absolute (``CLOSED_FORM_TOL``), on sizes of order 1. The solver stops when DogLeg's
largest residual is at most 1e-10 (PlaneGCS's default), so the geometry it returns is
within that order of the exact one; the slot, the least accurate here, measured 6.6e-11.

The diagnosis rows are derived from the linearised constraints. In a rectangle held by two
horizontals, two verticals and two lengths, a length on the top line is a combination of
the bottom length and the two verticals (top length - bottom length = (x2 - x3) - (x1 - x0),
and the verticals make x1 = x2 and x0 = x3), so these four rows are one dependent group.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
import pysmesh.sketch as sk
from pysmesh import EntityId, EntityKind, Session

CLOSED_FORM_TOL: float = 1e-10
CHILD_TIMEOUT_S: float = 120.0
_RNG_SEED: int = 42


def xy(a: sk.SketchArrays, s: sk.SketchSolution, row: int) -> NDArray[np.float64]:
    """A solved point."""
    return np.array(a.point(row, s.params), dtype=np.float64)


def scalar(a: sk.SketchArrays, s: sk.SketchSolution, row: int, col: int) -> float:
    """The solved parameter in column `col` of entity row `row`."""
    return float(s.params[a.entities[row, col]])


def jitter(rng: np.random.Generator, x: float, y: float) -> sk.Point:
    """A point near (x, y): an initial guess that no constraint holds exactly."""
    dx, dy = rng.normal(0.0, 0.05, 2)
    return sk.Point(x + float(dx), y + float(dy))


def rectangle(
    width: float,
    height: float,
    lengths: tuple[tuple[int, float], ...] | None = None,
    *,
    fix_origin: bool = True,
) -> tuple[sk.SketchBuilder, list[int], list[int]]:
    """A rectangle: rows 0-1 horizontal (bottom, top), 2-3 vertical (right, left), then
    the lengths (default: bottom = width, right = height). The first corner is held at
    the origin."""
    rng = np.random.default_rng(_RNG_SEED)
    b = sk.SketchBuilder()
    corners = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    p = [b.add(sk.Point(0.0, 0.0))] + [b.add(jitter(rng, x, y)) for x, y in corners[1:]]
    lines = [b.add(sk.Line(p[k], p[(k + 1) % 4])) for k in range(4)]
    if fix_origin:
        b.fix(p[0])
    b.horizontal(lines[0])
    b.horizontal(lines[2])
    b.vertical(lines[1])
    b.vertical(lines[3])
    for k, value in ((0, width), (1, height)) if lengths is None else lengths:
        b.length(lines[k], value)
    return b, p, lines


def slot(r: float, length: float) -> tuple[sk.SketchBuilder, dict[str, int]]:
    """A slot: two half-circle arcs of radius r joined by two lines of length `length`.

    The left center is held at the origin. Rows: horizontal top, the left radius, equal
    radii, the top length, and a tangent at each of the four shared end points.
    """
    rng = np.random.default_rng(_RNG_SEED)
    b = sk.SketchBuilder()
    e: dict[str, int] = {}
    e["c1"] = b.add(sk.Point(0.0, 0.0))
    e["c2"] = b.add(jitter(rng, length, 0.0))
    e["t1"] = b.add(jitter(rng, 0.0, r))
    e["t2"] = b.add(jitter(rng, length, r))
    e["b2"] = b.add(jitter(rng, length, -r))
    e["b1"] = b.add(jitter(rng, 0.0, -r))
    e["top"] = b.add(sk.Line(e["t1"], e["t2"]))
    e["bottom"] = b.add(sk.Line(e["b2"], e["b1"]))
    e["right"] = b.add(sk.Arc(e["c2"], e["b2"], e["t2"]))
    e["left"] = b.add(sk.Arc(e["c1"], e["t1"], e["b1"]))
    b.fix(e["c1"])
    b.horizontal(e["top"])
    b.radius(e["left"], r)
    b.equal(e["left"], e["right"])
    b.length(e["top"], length)
    b.tangent(e["top"], e["right"], e["t2"])
    b.tangent(e["right"], e["bottom"], e["b2"])
    b.tangent(e["bottom"], e["left"], e["b1"])
    b.tangent(e["left"], e["top"], e["t1"])
    return b, e


def wrapped(angle: float, reference: float) -> float:
    """`angle` moved by full turns to the turn of `reference`."""
    return angle - 2.0 * math.pi * round((angle - reference) / (2.0 * math.pi))


# ---- analytic sketches ----------------------------------------------------------------- #


@pytest.mark.parametrize(("width", "height"), [(3.0, 2.0), (0.5, 7.0)])
def test_a_rectangle_with_fixed_width_and_height_has_its_closed_form_corners(
    width: float, height: float
) -> None:
    b, p, _ = rectangle(width, height)
    a = b.build()

    s = a.solve()

    assert s.status == sk.SketchStatus.SOLVED
    assert s.dof == 0
    expected = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    for row, corner in zip(p, expected, strict=True):
        np.testing.assert_allclose(xy(a, s, row), corner, rtol=0, atol=CLOSED_FORM_TOL)


def test_an_isosceles_triangle_by_symmetry_has_its_closed_form_apex() -> None:
    """Base 4 on the x axis, legs 5, symmetric about the y axis: apex at (0, sqrt(21)).

    The x and y axes are fixed lines. A and B are symmetric about the y axis, the apex C
    lies on it, A lies on the x axis, |AB| = 4 and |CA| = 5; so A = (-2, 0), B = (2, 0),
    C = (0, sqrt(5^2 - 2^2)), and |CB| = |CA|.
    """
    b = sk.SketchBuilder()
    ends = [
        b.add(sk.Point(x, y))
        for x, y in ((0.0, -1.0), (0.0, 6.0), (-9.0, 0.0), (9.0, 0.0))
    ]
    axis = b.add(sk.Line(ends[0], ends[1]))
    ground = b.add(sk.Line(ends[2], ends[3]))
    for row in ends:
        b.fix(row)
    pa = b.add(sk.Point(-1.7, 0.2))
    pb = b.add(sk.Point(2.2, -0.1))
    pc = b.add(sk.Point(0.3, 4.4))
    base = b.add(sk.Line(pa, pb))
    b.add(sk.Line(pc, pa))
    b.add(sk.Line(pc, pb))
    b.symmetric(pa, pb, axis)
    b.coincident(pc, axis)
    b.coincident(pa, ground)
    b.length(base, 4.0)
    b.distance(pc, pa, 5.0)
    a = b.build()

    s = a.solve()

    assert s.status == sk.SketchStatus.SOLVED
    np.testing.assert_allclose(xy(a, s, pa), (-2.0, 0.0), rtol=0, atol=CLOSED_FORM_TOL)
    np.testing.assert_allclose(xy(a, s, pb), (2.0, 0.0), rtol=0, atol=CLOSED_FORM_TOL)
    np.testing.assert_allclose(
        xy(a, s, pc), (0.0, math.sqrt(21.0)), rtol=0, atol=CLOSED_FORM_TOL
    )
    legs = np.linalg.norm(xy(a, s, pc) - xy(a, s, pa)), np.linalg.norm(
        xy(a, s, pc) - xy(a, s, pb)
    )
    np.testing.assert_allclose(legs, (5.0, 5.0), rtol=0, atol=CLOSED_FORM_TOL)


@pytest.mark.parametrize(("r", "length"), [(1.0, 4.0), (0.5, 3.0)])
def test_a_slot_of_two_arcs_tangent_to_two_lines_has_its_closed_form(
    r: float, length: float
) -> None:
    """Centers (0, 0) and (L, 0); the lines at y = +-r; the right arc from -pi/2 to
    pi/2, the left from pi/2 to 3 pi/2, both of radius r."""
    b, e = slot(r, length)
    a = b.build()

    s = a.solve()

    assert s.status == sk.SketchStatus.SOLVED
    assert s.dof == 0
    expected = {
        "c1": (0.0, 0.0),
        "c2": (length, 0.0),
        "t1": (0.0, r),
        "t2": (length, r),
        "b2": (length, -r),
        "b1": (0.0, -r),
    }
    for name, point in expected.items():
        np.testing.assert_allclose(
            xy(a, s, e[name]), point, rtol=0, atol=CLOSED_FORM_TOL
        )
    for arc, (a0, a1) in (
        ("right", (-math.pi / 2, math.pi / 2)),
        ("left", (math.pi / 2, 3 * math.pi / 2)),
    ):
        assert abs(scalar(a, s, e[arc], 4) - r) <= CLOSED_FORM_TOL
        start = scalar(a, s, e[arc], 5)
        end = scalar(a, s, e[arc], 6)
        assert abs(wrapped(start, a0) - a0) <= CLOSED_FORM_TOL
        assert abs((end - start) - (a1 - a0)) <= CLOSED_FORM_TOL


@pytest.mark.parametrize("degrees", [30.0, 50.0, 90.0, 135.0])
def test_a_circle_tangent_to_two_lines_sits_on_the_bisector(degrees: float) -> None:
    """Lines from the origin along x and at angle t; a circle of radius r tangent to both
    lies on the bisector, its center at (r / tan(t / 2), r)."""
    theta = math.radians(degrees)
    r = 1.0
    b = sk.SketchBuilder()
    v = b.add(sk.Point(0.0, 0.0))
    e1 = b.add(sk.Point(5.2, 0.1))
    e2 = b.add(sk.Point(5.0 * math.cos(theta) + 0.1, 5.0 * math.sin(theta) - 0.1))
    l1 = b.add(sk.Line(v, e1))
    l2 = b.add(sk.Line(v, e2))
    center_guess = (1.1 / math.tan(theta / 2), 0.9)
    c = b.add(sk.Point(*center_guess))
    circle = b.add(sk.Circle(c, 0.8))
    b.fix(v)
    b.horizontal(l1)
    b.length(l1, 5.0)
    b.length(l2, 5.0)
    b.angle(l1, l2, theta)
    b.radius(circle, r)
    b.tangent(l1, circle)
    b.tangent(l2, circle)
    a = b.build()

    s = a.solve()

    assert s.status == sk.SketchStatus.SOLVED
    np.testing.assert_allclose(
        xy(a, s, c), (r / math.tan(theta / 2), r), rtol=0, atol=CLOSED_FORM_TOL
    )
    np.testing.assert_allclose(
        xy(a, s, e2),
        (5.0 * math.cos(theta), 5.0 * math.sin(theta)),
        rtol=0,
        atol=CLOSED_FORM_TOL,
    )


def test_an_ellipse_with_its_internal_alignment_has_its_closed_form_points() -> None:
    """Center (0, 0), major axis horizontal of length 2a = 4, minor radius b = 1.

    The foci are at (+-sqrt(a^2 - b^2), 0) = (+-sqrt(3), 0), the major ends at (+-2, 0),
    the minor ends at (0, +-1). The positive minor end is a quarter turn counter-clockwise
    from the positive major end.
    """
    b = sk.SketchBuilder()
    c = b.add(sk.Point(0.0, 0.0))
    f = b.add(sk.Point(1.6, 0.1))
    ellipse = b.add(sk.Ellipse(c, f, 0.9))
    pts = {
        sk.SketchInternalRole.MAJOR_POSITIVE: b.add(sk.Point(2.1, 0.2)),
        sk.SketchInternalRole.MAJOR_NEGATIVE: b.add(sk.Point(-1.9, 0.0)),
        sk.SketchInternalRole.MINOR_POSITIVE: b.add(sk.Point(0.1, 1.1)),
        sk.SketchInternalRole.MINOR_NEGATIVE: b.add(sk.Point(0.0, -0.9)),
        sk.SketchInternalRole.FOCUS2: b.add(sk.Point(-1.6, 0.1)),
    }
    major = b.add(
        sk.Line(
            pts[sk.SketchInternalRole.MAJOR_NEGATIVE],
            pts[sk.SketchInternalRole.MAJOR_POSITIVE],
        )
    )
    b.fix(c)
    for role, row in pts.items():
        b.internal_coincident(ellipse, row, role)
    b.horizontal(major)
    b.length(major, 4.0)
    b.distance(c, pts[sk.SketchInternalRole.MINOR_POSITIVE], 1.0)
    a = b.build()

    s = a.solve()

    assert s.status == sk.SketchStatus.SOLVED
    assert s.dof == 0
    root3 = math.sqrt(3.0)
    np.testing.assert_allclose(xy(a, s, f), (root3, 0.0), rtol=0, atol=CLOSED_FORM_TOL)
    assert abs(scalar(a, s, ellipse, 3) - 1.0) <= CLOSED_FORM_TOL
    expected = {
        sk.SketchInternalRole.MAJOR_POSITIVE: (2.0, 0.0),
        sk.SketchInternalRole.MAJOR_NEGATIVE: (-2.0, 0.0),
        sk.SketchInternalRole.MINOR_POSITIVE: (0.0, 1.0),
        sk.SketchInternalRole.MINOR_NEGATIVE: (0.0, -1.0),
        sk.SketchInternalRole.FOCUS2: (-root3, 0.0),
    }
    for role, point in expected.items():
        np.testing.assert_allclose(
            xy(a, s, pts[role]), point, rtol=0, atol=CLOSED_FORM_TOL
        )


# ---- diagnosis ------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("lengths", "conflicting"),
    [
        # A top length that differs from the bottom one: the dependent group of the
        # module docstring.
        (((0, 3.0), (1, 2.0), (2, 3.5)), [2, 3, 4, 6]),
        # A second length on the bottom line: the two rows alone.
        (((0, 3.0), (1, 2.0), (0, 3.5)), [4, 6]),
    ],
)
def test_an_over_constrained_rectangle_names_its_conflicting_rows(
    lengths: tuple[tuple[int, float], ...], conflicting: list[int]
) -> None:
    """dof is PlaneGCS's count when the Jacobian has full rank: 6 unknowns less 7
    non-redundant equations = -1."""
    b, _, _ = rectangle(3.0, 2.0, lengths)
    a = b.build()

    s = a.solve()

    assert s.status == sk.SketchStatus.OVER_CONSTRAINED
    assert s.conflicting.tolist() == conflicting
    assert s.redundant.tolist() == []
    assert s.dof == -1
    np.testing.assert_array_equal(s.params, a.params)


@pytest.mark.parametrize("line", [0, 2])
def test_a_consistent_extra_length_is_redundant_and_the_rectangle_solves(
    line: int,
) -> None:
    """PlaneGCS removes the latest row of a dependent group first (GCS.cpp, the
    popularity contest's third rule), so the extra length, row 6, is the redundant one.
    """
    b, p, _ = rectangle(3.0, 2.0, ((0, 3.0), (1, 2.0), (line, 3.0)))
    a = b.build()

    s = a.solve()

    assert s.status == sk.SketchStatus.SOLVED
    assert s.redundant.tolist() == [6]
    assert s.conflicting.tolist() == []
    assert s.dof == 0
    np.testing.assert_allclose(xy(a, s, p[2]), (3.0, 2.0), rtol=0, atol=CLOSED_FORM_TOL)


@pytest.mark.parametrize(("fix_origin", "dof"), [(True, 2), (False, 4)])
def test_an_under_constrained_rectangle_has_the_known_dof(
    fix_origin: bool, dof: int
) -> None:
    """Four free corners hold 8 unknowns (6 with the origin fixed); two horizontals and
    two verticals remove 4: the width and the height are left, and the position too
    when the origin is free."""
    b, p, _ = rectangle(3.0, 2.0, (), fix_origin=fix_origin)
    a = b.build()

    s = a.solve()

    assert s.status == sk.SketchStatus.UNDER_CONSTRAINED
    assert s.dof == dof
    corners = [xy(a, s, row) for row in p]
    assert corners[0][1] == pytest.approx(corners[1][1], abs=CLOSED_FORM_TOL)
    assert corners[1][0] == pytest.approx(corners[2][0], abs=CLOSED_FORM_TOL)


def test_an_impossible_triangle_fails_and_returns_the_initial_guess() -> None:
    """Sides 1, 1 and 3 break the triangle inequality. The linearised constraints are
    independent, so the diagnosis finds no conflict, and no solver reaches the
    tolerance."""
    b = sk.SketchBuilder()
    p = [b.add(sk.Point(x, y)) for x, y in ((0.0, 0.0), (1.0, 0.5), (2.0, 0.0))]
    b.fix(p[0])
    b.distance(p[0], p[1], 1.0)
    b.distance(p[1], p[2], 1.0)
    b.distance(p[0], p[2], 3.0)
    a = b.build()

    s = a.solve()

    assert s.status == sk.SketchStatus.FAILED
    assert s.conflicting.tolist() == []
    np.testing.assert_array_equal(s.params, a.params)


def test_a_sketch_without_constraints_keeps_its_guess_with_every_unknown_free() -> None:
    b = sk.SketchBuilder()
    p = [b.add(sk.Point(x, 0.5 * x)) for x in (0.0, 1.0, 2.0)]
    b.add(sk.Line(p[0], p[1]))
    b.add(sk.Circle(p[2], 0.3))
    a = b.build()

    s = a.solve()

    assert s.status == sk.SketchStatus.UNDER_CONSTRAINED
    assert s.dof == 7
    np.testing.assert_array_equal(s.params, a.params)


@pytest.mark.parametrize(
    ("value", "status"), [(3.0, "SOLVED"), (3.5, "OVER_CONSTRAINED")]
)
def test_a_sketch_with_nothing_to_move_checks_each_row_as_it_stands(
    value: float, status: str
) -> None:
    """Every parameter fixed: a row holds or conflicts at the initial guess."""
    b = sk.SketchBuilder()
    p = [b.add(sk.Point(x, 0.0)) for x in (0.0, 3.0)]
    line = b.add(sk.Line(p[0], p[1]))
    b.fix(line)
    b.horizontal(line)
    b.length(line, value)
    a = b.build()

    s = a.solve()

    assert s.status == status
    assert s.dof == 0
    assert s.conflicting.tolist() == ([] if status == "SOLVED" else [1])
    np.testing.assert_array_equal(s.params, a.params)


# ---- the arrays ------------------------------------------------------------------------ #


def test_the_inputs_stay_as_they_were_and_untouched_parameters_keep_their_values() -> (
    None
):
    """The result is a new array. A fixed parameter and a parameter no entity uses keep
    their values exactly."""
    b, _, _ = rectangle(3.0, 2.0)
    a = b.build()
    params = np.append(a.params, 42.0)
    fixed = np.append(a.fixed, False)
    copies = [x.copy() for x in (params, a.entities, a.constraints, a.values, fixed)]

    s = sk.solve(params, a.entities, a.constraints, a.values, fixed=fixed)

    for original, copy in zip(
        (params, a.entities, a.constraints, a.values, fixed), copies, strict=True
    ):
        np.testing.assert_array_equal(original, copy)
    assert s.params is not params
    assert s.params[-1] == 42.0
    assert tuple(s.params[:2]) == (0.0, 0.0)
    np.testing.assert_allclose(s.params[2:4], (3.0, 0.0), rtol=0, atol=CLOSED_FORM_TOL)


def test_knots_never_move_even_when_the_mask_leaves_them_free() -> None:
    """A point pulled onto a B-spline moves the poles, never the knots."""
    b = sk.SketchBuilder()
    poles = [
        b.add(sk.Point(x, y))
        for x, y in ((0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0))
    ]
    spline = b.add(sk.BSpline(tuple(poles), 3, (0.0, 1.0), (4, 4)))
    q = b.add(sk.Point(2.0, 3.0))
    b.fix(q)
    b.coincident(q, spline)
    a = b.build()
    assert a.spline_indices is not None
    knots = a.spline_indices[8:10]
    fixed = a.fixed.copy()
    fixed[knots] = False

    s = sk.solve(
        a.params,
        a.entities,
        a.constraints,
        a.values,
        fixed=fixed,
        spline_offsets=a.spline_offsets,
        spline_indices=a.spline_indices,
    )

    assert s.status == sk.SketchStatus.UNDER_CONSTRAINED
    assert s.params[knots].tolist() == [0.0, 1.0]
    assert not np.array_equal(
        s.params[a.entities[poles[1], 1:3]], a.params[a.entities[poles[1], 1:3]]
    )


# ---- into CAD -------------------------------------------------------------------------- #


def test_a_solved_slot_becomes_a_face_with_the_closed_form_area() -> None:
    """The slot's face: a 2r x L rectangle and two half disks, area 2 r L + pi r^2.

    The edges come from the session's add_line and add_arc (each arc through its
    midpoint at the mean angle); make_face joins them. GProp's area is compared at 1e-9
    relative: the solved geometry is within 1e-10 of the closed form.
    """
    r, length = 1.0, 4.0
    b, e = slot(r, length)
    a = b.build()
    s = a.solve()
    session = Session()

    def at(name: str) -> tuple[float, float, float]:
        """A solved point, in 3-D."""
        x, y = xy(a, s, e[name])
        return (float(x), float(y), 0.0)

    session.add_line(at("t1"), at("t2"))
    session.add_line(at("b2"), at("b1"))
    for arc, start, end in (("right", "b2", "t2"), ("left", "t1", "b1")):
        center = xy(a, s, e["c2" if arc == "right" else "c1"])
        mid = 0.5 * (scalar(a, s, e[arc], 5) + scalar(a, s, e[arc], 6))
        radius = scalar(a, s, e[arc], 4)
        through = (
            float(center[0] + radius * math.cos(mid)),
            float(center[1] + radius * math.sin(mid)),
            0.0,
        )
        session.add_arc(at(start), through, at(end))
    edges = [EntityId(int(i)) for i in session.entities(EntityKind.EDGE)]
    session.make_face(edges)

    (face,) = session.entities(EntityKind.FACE)
    area = float(session.mass_properties([EntityId(int(face))]).measure[0])
    assert area == pytest.approx(2.0 * r * length + math.pi * r * r, rel=1e-9)


# ---- determinism, time, the GIL -------------------------------------------------------- #


def mixed_sketch() -> sk.SketchArrays:
    """A slot with an ellipse tangent to its top line and a B-spline through a point.

    The ellipse can turn and the B-spline's poles can move, so the solution is the one
    the solver's path reaches: the strictest case for bit-identical results.
    """
    b, e = slot(1.0, 4.0)
    c = b.add(sk.Point(2.0, 2.5))
    f = b.add(sk.Point(2.8, 2.6))
    ellipse = b.add(sk.Ellipse(c, f, 0.7))
    b.tangent(e["top"], ellipse)
    b.distance(c, e["c1"], 3.2)
    poles = [
        b.add(sk.Point(x, y))
        for x, y in ((6.0, 0.0), (7.0, 2.0), (8.0, -1.0), (9.0, 1.0))
    ]
    spline = b.add(sk.BSpline(tuple(poles), 3, (0.0, 1.0), (4, 4)))
    q = b.add(sk.Point(7.6, 0.9))
    b.coincident(q, spline)
    b.coincident(poles[0], e["c2"])
    return b.build()


_CHILD: str = """
import os, sys
for d in (os.environ.get("PYSMESH_OCCT_BIN"),
          os.path.join(sys.prefix, "Library", "bin")):
    if d and os.path.isdir(d):
        os.add_dll_directory(d)
sys.path.insert(0, sys.argv[1])
import numpy as np
import pysmesh.sketch as sk
data = np.load(sys.argv[2])
s = sk.solve(data["params"], data["entities"], data["constraints"], data["values"],
             fixed=data["fixed"], spline_offsets=data["spline_offsets"],
             spline_indices=data["spline_indices"])
sys.stdout.write("RESULT " + s.status + " " + s.params.tobytes().hex() + "\\n")
"""


def test_identical_arrays_give_bit_identical_params_in_fresh_processes(
    tmp_path: Path,
) -> None:
    a = mixed_sketch()
    assert a.spline_offsets is not None and a.spline_indices is not None
    data = tmp_path / "sketch.npz"
    np.savez(
        data,
        params=a.params,
        entities=a.entities,
        constraints=a.constraints,
        values=a.values,
        fixed=a.fixed,
        spline_offsets=a.spline_offsets,
        spline_indices=a.spline_indices,
    )
    package_root = str(Path(ps.__file__).resolve().parent.parent)
    here = a.solve()

    runs = [
        subprocess.run(
            [sys.executable, "-c", _CHILD, package_root, str(data)],
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
            env=dict(os.environ),
            check=False,
        )
        for _ in range(2)
    ]

    assert all(run.returncode == 0 for run in runs), [r.stderr[-2000:] for r in runs]
    lines = [
        [ln for ln in run.stdout.splitlines() if ln.startswith("RESULT ")]
        for run in runs
    ]
    expected = f"RESULT {here.status} {here.params.tobytes().hex()}"
    assert here.status == sk.SketchStatus.UNDER_CONSTRAINED
    assert lines == [[expected], [expected]]


def test_fifty_unrelated_solves_leave_the_result_bit_identical() -> None:
    a = mixed_sketch()
    first = a.solve()
    rng = np.random.default_rng(_RNG_SEED)

    for _ in range(50):
        w, h = rng.uniform(0.5, 5.0, 2)
        b, _, _ = rectangle(float(w), float(h))
        b.build().solve()
    again = a.solve()

    assert again.status == first.status
    assert again.params.tobytes() == first.params.tobytes()
    assert again.dof == first.dof


def slot_chain(count: int) -> sk.SketchArrays:
    """`count` slots in a row, 10 entities and 20 constraint rows each.

    Each slot is the slot of :func:`slot` (the first held at the origin, each next one
    placed by its distances from the previous one), plus consistent extra rows that the
    others imply: a sketch with redundant rows, as drawn sketches often are.
    """
    rng = np.random.default_rng(_RNG_SEED)
    r, length, pitch = 1.0, 4.0, 8.0
    b = sk.SketchBuilder()
    previous: tuple[int, int] | None = None
    for k in range(count):
        x0 = k * pitch
        c1 = b.add(sk.Point(0.0, 0.0) if k == 0 else jitter(rng, x0, 0.0))
        c2 = b.add(jitter(rng, x0 + length, 0.0))
        t1 = b.add(jitter(rng, x0, r))
        t2 = b.add(jitter(rng, x0 + length, r))
        b2 = b.add(jitter(rng, x0 + length, -r))
        b1 = b.add(jitter(rng, x0, -r))
        top = b.add(sk.Line(t1, t2))
        bottom = b.add(sk.Line(b2, b1))
        right = b.add(sk.Arc(c2, b2, t2))
        left = b.add(sk.Arc(c1, t1, b1))
        if previous is None:
            b.fix(c1)
            b.distance(t1, t2, length)
            b.distance(b1, b2, length)
        else:
            b.distance(c1, previous[0], pitch - length)
            b.distance(c1, previous[1], r)
        b.horizontal(top)
        b.radius(left, r)
        b.equal(left, right)
        b.length(top, length)
        b.tangent(top, right, t2)
        b.tangent(right, bottom, b2)
        b.tangent(bottom, left, b1)
        b.tangent(left, top, t1)
        b.equal(top, bottom)
        b.length(bottom, length)
        b.radius(right, r)
        b.distance(c1, c2, length)
        b.distance(t1, b1, 2 * r)
        b.distance(t2, b2, 2 * r)
        b.distance(c1, top, r)
        b.distance(c2, bottom, r)
        b.horizontal(bottom)
        b.parallel(top, bottom)
        previous = (c2, bottom)
    return b.build()


def test_a_sketch_of_200_entities_and_400_constraints_solves_in_under_a_second() -> (
    None
):
    """Brief rule: under 1 s on the build machine (measured 0.18 s)."""
    a = slot_chain(20)
    assert a.entities.shape[0] == 200
    assert a.constraints.shape[0] == 400

    start = time.perf_counter()
    s = a.solve()
    elapsed = time.perf_counter() - start

    assert s.status == sk.SketchStatus.SOLVED
    assert s.dof == 0
    assert elapsed < 1.0
    last_c2 = 10 * 19 + 1
    np.testing.assert_allclose(
        xy(a, s, last_c2), (19 * 8.0 + 4.0, 0.0), rtol=0, atol=CLOSED_FORM_TOL
    )


def test_the_solve_releases_the_gil() -> None:
    """While a thread solves, this thread keeps running inside the solve's window.

    The worker stamps the start and the end of its solve; this thread stamps each turn of
    its loop. With the GIL held for the whole native call, no stamp of this thread could
    fall inside the middle 80 % of that window.
    """
    a = slot_chain(20)
    stamps: list[float] = []
    window: list[float] = []

    def work() -> None:
        """Solve once between two stamps."""
        window.append(time.perf_counter())
        a.solve()
        window.append(time.perf_counter())

    worker = threading.Thread(target=work)
    worker.start()
    while worker.is_alive():
        stamps.append(time.perf_counter())
    worker.join()

    t0, t1 = window
    margin = 0.1 * (t1 - t0)
    inside = [t for t in stamps if t0 + margin < t < t1 - margin]
    assert len(inside) > 0
