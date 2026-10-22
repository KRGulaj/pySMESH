# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-22

"""Gates for pysmesh.sketch: every constraint kind holds after the solve.

Each case starts from an initial guess that breaks the constraint, solves, and checks the
constraint on the solved geometry with its own formula, independent of PlaneGCS:

* a point on an ellipse: its distances to the two foci sum to 2a (the focal property);
* a line tangent to an ellipse: the line's distance from the center equals the ellipse's
  support function in the line's normal, sqrt(a^2 n_u^2 + b^2 n_v^2);
* the normal of an ellipse at a point bisects the two focal radii;
* a B-spline point: the Cox-de Boor recursion on the clamped knot vector.

The solver stops when the largest constraint residual is at most 1e-10, so a constraint
holds within that order; the checks use 1e-9 (``HOLD_TOL``), on sizes of order 1.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh.sketch as sk

HOLD_TOL: float = 1e-9

Check = Callable[[sk.SketchArrays, sk.SketchSolution], None]
Case = Callable[[], tuple[sk.SketchBuilder, Check]]


def solved(b: sk.SketchBuilder) -> tuple[sk.SketchArrays, sk.SketchSolution]:
    """Build and solve; the solve holds every constraint."""
    a = b.build()
    s = a.solve()
    assert s.status in (sk.SketchStatus.SOLVED, sk.SketchStatus.UNDER_CONSTRAINED), (
        s.status,
        s.conflicting.tolist(),
    )
    return a, s


def pt(a: sk.SketchArrays, s: sk.SketchSolution, row: int) -> NDArray[np.float64]:
    """A solved point."""
    return np.array(a.point(row, s.params), dtype=np.float64)


def ends(
    a: sk.SketchArrays, s: sk.SketchSolution, line: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A solved line's start and end."""
    return pt(a, s, int(a.entities[line, 1])), pt(a, s, int(a.entities[line, 2]))


def direction(
    a: sk.SketchArrays, s: sk.SketchSolution, line: int
) -> NDArray[np.float64]:
    """A solved line's end less its start."""
    p, q = ends(a, s, line)
    return q - p


def cross(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    """The z component of u x v."""
    return float(u[0] * v[1] - u[1] * v[0])


def unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """v / |v|."""
    return v / float(np.linalg.norm(v))


def round_of(
    a: sk.SketchArrays, s: sk.SketchSolution, row: int
) -> tuple[NDArray[np.float64], float]:
    """A solved circle's or arc's center and radius."""
    kind = int(a.entities[row, 0])
    radius_col = 2 if kind == sk.SketchEntityKind.CIRCLE else 4
    return pt(a, s, int(a.entities[row, 1])), float(
        s.params[a.entities[row, radius_col]]
    )


def ellipse_of(
    a: sk.SketchArrays, s: sk.SketchSolution, row: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """A solved ellipse's center, focus, minor radius and major radius."""
    kind = int(a.entities[row, 0])
    minor_col = 3 if kind == sk.SketchEntityKind.ELLIPSE else 5
    c = pt(a, s, int(a.entities[row, 1]))
    f = pt(a, s, int(a.entities[row, 2]))
    b = float(s.params[a.entities[row, minor_col]])
    return c, f, b, math.hypot(float(np.linalg.norm(f - c)), b)


def ellipse_point(
    a: sk.SketchArrays, s: sk.SketchSolution, row: int, u: float
) -> NDArray[np.float64]:
    """The point of a solved ellipse at its parameter u: c + a cos u e1 + b sin u e2."""
    c, f, b, major = ellipse_of(a, s, row)
    e1 = unit(f - c)
    e2 = np.array([-e1[1], e1[0]])
    return c + major * math.cos(u) * e1 + b * math.sin(u) * e2


def ellipse_normal(
    a: sk.SketchArrays, s: sk.SketchSolution, row: int, p: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The normal of a solved ellipse at its point p: the sum of the focal unit radii."""
    c, f, _, _ = ellipse_of(a, s, row)
    return unit(p - f) + unit(p - (2.0 * c - f))


def bspline_value(
    poles: NDArray[np.float64],
    weights: NDArray[np.float64],
    knots: list[float],
    mults: list[int],
    degree: int,
    u: float,
) -> NDArray[np.float64]:
    """A rational B-spline's point at u, by the Cox-de Boor recursion."""
    t = [k for k, m in zip(knots, mults, strict=True) for _ in range(m)]
    n = len(poles)
    basis = [1.0 if t[i] <= u < t[i + 1] else 0.0 for i in range(len(t) - 1)]
    if u >= t[-1]:
        last = max(i for i in range(len(t) - 1) if t[i] < t[i + 1])
        basis[last] = 1.0
    for k in range(1, degree + 1):
        for i in range(len(t) - 1 - k):
            left = (u - t[i]) / (t[i + k] - t[i]) * basis[i] if t[i + k] > t[i] else 0.0
            right = (
                (t[i + k + 1] - u) / (t[i + k + 1] - t[i + 1]) * basis[i + 1]
                if t[i + k + 1] > t[i + 1]
                else 0.0
            )
            basis[i] = left + right
    nw = np.array(basis[:n]) * weights
    return np.asarray((nw[:, None] * poles).sum(axis=0) / nw.sum(), dtype=np.float64)


SPLINE_POLES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 2.0),
    (3.0, 2.0),
    (4.0, 0.0),
)


def fixed_spline(b: sk.SketchBuilder) -> tuple[int, list[int]]:
    """A cubic Bezier B-spline with fixed poles; its row and its pole rows."""
    poles = [b.add(sk.Point(x, y)) for x, y in SPLINE_POLES]
    spline = b.add(sk.BSpline(tuple(poles), 3, (0.0, 1.0), (4, 4)))
    b.fix(spline)
    return spline, poles


def spline_distance(p: NDArray[np.float64]) -> float:
    """The distance from p to the fixed B-spline: dense samples, then golden sections."""
    poles = np.array(SPLINE_POLES, dtype=np.float64)
    w = np.ones(4)

    def d(u: float) -> float:
        """The distance from p to the curve's point at u."""
        return float(
            np.linalg.norm(bspline_value(poles, w, [0.0, 1.0], [4, 4], 3, u) - p)
        )

    us = np.linspace(0.0, 1.0, 2001)
    k = int(np.argmin([d(float(u)) for u in us]))
    lo, hi = float(us[max(k - 1, 0)]), float(us[min(k + 1, len(us) - 1)])
    g = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(100):
        m1, m2 = hi - g * (hi - lo), lo + g * (hi - lo)
        if d(m1) < d(m2):
            hi = m2
        else:
            lo = m1
    return d(0.5 * (lo + hi))


def fixed_line(
    b: sk.SketchBuilder, p: tuple[float, float], q: tuple[float, float]
) -> int:
    """A line whose points do not move."""
    line = b.add(sk.Line(b.add(sk.Point(*p)), b.add(sk.Point(*q))))
    b.fix(line)
    return line


def fixed_ellipse(b: sk.SketchBuilder, arc: bool = False) -> int:
    """An ellipse centered at (0.5, -0.2), focus at distance 1.5 along 30 degrees, minor
    radius 1, that does not move; an elliptic arc from parameter -1 to 2 when `arc`."""
    c = (0.5, -0.2)
    t = math.radians(30.0)
    f = (c[0] + 1.5 * math.cos(t), c[1] + 1.5 * math.sin(t))
    rows = [b.add(sk.Point(*c)), b.add(sk.Point(*f))]
    if not arc:
        e = b.add(sk.Ellipse(rows[0], rows[1], 1.0))
    else:
        major = math.hypot(1.5, 1.0)
        e1 = (math.cos(t), math.sin(t))
        e2 = (-e1[1], e1[0])

        def at(u: float) -> tuple[float, float]:
            """The ellipse's point at parameter u."""
            return (
                c[0] + major * math.cos(u) * e1[0] + math.sin(u) * e2[0],
                c[1] + major * math.cos(u) * e1[1] + math.sin(u) * e2[1],
            )

        start, end = b.add(sk.Point(*at(-1.0))), b.add(sk.Point(*at(2.0)))
        e = b.add(sk.EllipticArc(rows[0], rows[1], start, end, 1.0))
    b.fix(e)
    return e


# ---- COINCIDENT ------------------------------------------------------------------------- #


def coincident_point() -> tuple[sk.SketchBuilder, Check]:
    """A free point coincident with a fixed point."""
    b = sk.SketchBuilder()
    p = b.add(sk.Point(1.0, 2.0))
    q = b.add(sk.Point(1.7, 1.1))
    b.fix(p)
    b.coincident(q, p)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        np.testing.assert_allclose(pt(a, s, q), (1.0, 2.0), rtol=0, atol=HOLD_TOL)

    return b, check


def coincident_line() -> tuple[sk.SketchBuilder, Check]:
    """A free point on a fixed line."""
    b = sk.SketchBuilder()
    line = fixed_line(b, (0.0, 0.0), (4.0, 1.0))
    q = b.add(sk.Point(1.0, 2.0))
    b.coincident(line, q)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        p0, _ = ends(a, s, line)
        assert abs(cross(pt(a, s, q) - p0, unit(direction(a, s, line)))) <= HOLD_TOL

    return b, check


def coincident_round(
    kind: str,
) -> tuple[sk.SketchBuilder, Check]:
    """A free point on a fixed circle or arc."""
    b = sk.SketchBuilder()
    c = b.add(sk.Point(0.0, 0.0))
    if kind == "circle":
        curve = b.add(sk.Circle(c, 2.0))
    else:
        curve = b.add(sk.Arc(c, b.add(sk.Point(2.0, 0.0)), b.add(sk.Point(0.0, 2.0))))
    b.fix(curve)
    q = b.add(sk.Point(1.5, 1.6))
    b.coincident(q, curve)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        center, r = round_of(a, s, curve)
        assert abs(float(np.linalg.norm(pt(a, s, q) - center)) - r) <= HOLD_TOL

    return b, check


def coincident_ellipse(
    arc: bool,
) -> tuple[sk.SketchBuilder, Check]:
    """A free point on a fixed ellipse or elliptic arc."""
    b = sk.SketchBuilder()
    e = fixed_ellipse(b, arc)
    q = b.add(sk.Point(1.5, 1.4))
    b.coincident(q, e)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        c, f, _, major = ellipse_of(a, s, e)
        p = pt(a, s, q)
        focal = float(np.linalg.norm(p - f) + np.linalg.norm(p - (2.0 * c - f)))
        assert abs(focal - 2.0 * major) <= HOLD_TOL

    return b, check


def coincident_bspline() -> tuple[sk.SketchBuilder, Check]:
    """A free point on a fixed B-spline."""
    b = sk.SketchBuilder()
    spline, _ = fixed_spline(b)
    q = b.add(sk.Point(2.2, 2.0))
    b.coincident(q, spline)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        assert spline_distance(pt(a, s, q)) <= HOLD_TOL

    return b, check


@pytest.mark.parametrize(
    "case",
    [
        coincident_point,
        coincident_line,
        lambda: coincident_round("circle"),
        lambda: coincident_round("arc"),
        lambda: coincident_ellipse(False),
        lambda: coincident_ellipse(True),
        coincident_bspline,
    ],
    ids=["point", "line", "circle", "arc", "ellipse", "elliptic_arc", "bspline"],
)
def test_coincident_holds_for_every_point_and_curve_pair(case: Case) -> None:
    b, check = case()

    a, s = solved(b)

    check(a, s)


# ---- TANGENT ---------------------------------------------------------------------------- #


def tangent_line_round(
    kind: str,
) -> tuple[sk.SketchBuilder, Check]:
    """A free circle or arc tangent to a fixed line."""
    b = sk.SketchBuilder()
    line = fixed_line(b, (-3.0, 0.0), (3.0, 0.0))
    c = b.add(sk.Point(0.4, 1.5))
    if kind == "circle":
        curve = b.add(sk.Circle(c, 1.0))
    else:
        curve = b.add(sk.Arc(c, b.add(sk.Point(1.4, 1.5)), b.add(sk.Point(0.4, 2.5))))
    b.tangent(line, curve)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        center, r = round_of(a, s, curve)
        p0, _ = ends(a, s, line)
        assert abs(abs(cross(center - p0, unit(direction(a, s, line)))) - r) <= HOLD_TOL

    return b, check


def tangent_circles(
    inner: bool,
) -> tuple[sk.SketchBuilder, Check]:
    """A free circle tangent to a fixed one, outside or inside it."""
    b = sk.SketchBuilder()
    c1 = b.add(sk.Point(0.0, 0.0))
    first = b.add(sk.Circle(c1, 3.0 if inner else 1.0))
    b.fix(first)
    c2 = b.add(sk.Point(0.8, 0.2) if inner else sk.Point(2.5, 0.3))
    second = b.add(sk.Circle(c2, 1.0))
    b.radius(second, 1.0)
    b.tangent(first, second)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        (p, r1), (q, r2) = round_of(a, s, first), round_of(a, s, second)
        expected = abs(r1 - r2) if inner else r1 + r2
        assert abs(float(np.linalg.norm(p - q)) - expected) <= HOLD_TOL

    return b, check


def tangent_line_ellipse() -> tuple[sk.SketchBuilder, Check]:
    """A free line tangent to a fixed ellipse."""
    b = sk.SketchBuilder()
    e = fixed_ellipse(b)
    line = b.add(sk.Line(b.add(sk.Point(-2.0, 2.4)), b.add(sk.Point(3.0, 1.8))))
    b.tangent(line, e)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        c, f, minor, major = ellipse_of(a, s, e)
        p0, _ = ends(a, s, line)
        d = unit(direction(a, s, line))
        normal = np.array([-d[1], d[0]])
        e1 = unit(f - c)
        e2 = np.array([-e1[1], e1[0]])
        support = math.sqrt((major * normal @ e1) ** 2 + (minor * normal @ e2) ** 2)
        assert abs(abs(float(normal @ (c - p0))) - support) <= HOLD_TOL

    return b, check


def tangent_arc_line_at_point() -> tuple[sk.SketchBuilder, Check]:
    """A line tangent to an arc at the end point they share."""
    b = sk.SketchBuilder()
    c = b.add(sk.Point(0.0, 0.0))
    start = b.add(sk.Point(1.0, 0.0))
    end = b.add(sk.Point(0.0, 1.0))
    arc = b.add(sk.Arc(c, start, end))
    b.fix(c)
    far = b.add(sk.Point(-2.0, 1.6))
    line = b.add(sk.Line(end, far))
    b.tangent(arc, line, end)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        center, r = round_of(a, s, arc)
        p = pt(a, s, end)
        assert abs(float(np.linalg.norm(p - center)) - r) <= HOLD_TOL
        assert abs(float(unit(p - center) @ unit(direction(a, s, line)))) <= HOLD_TOL

    return b, check


def tangent_ellipse_circle_at_point() -> tuple[sk.SketchBuilder, Check]:
    """A circle tangent to a fixed ellipse at a point on both."""
    b = sk.SketchBuilder()
    e = fixed_ellipse(b)
    c = b.add(sk.Point(2.6, 1.6))
    circle = b.add(sk.Circle(c, 0.5))
    b.radius(circle, 0.5)
    q = b.add(sk.Point(2.2, 1.2))
    b.coincident(q, e)
    b.coincident(q, circle)
    b.tangent(e, circle, q)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        p = pt(a, s, q)
        center, _ = round_of(a, s, circle)
        assert (
            abs(cross(unit(ellipse_normal(a, s, e, p)), unit(p - center))) <= HOLD_TOL
        )

    return b, check


def tangent_bspline_line_at_end_pole() -> tuple[sk.SketchBuilder, Check]:
    """A line tangent to a fixed B-spline at its first pole."""
    b = sk.SketchBuilder()
    spline, poles = fixed_spline(b)
    far = b.add(sk.Point(-2.0, -0.5))
    line = b.add(sk.Line(poles[0], far))
    b.tangent(spline, line, poles[0])

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        tangent = pt(a, s, poles[1]) - pt(a, s, poles[0])
        assert abs(cross(unit(tangent), unit(direction(a, s, line)))) <= HOLD_TOL

    return b, check


@pytest.mark.parametrize(
    "case",
    [
        lambda: tangent_line_round("circle"),
        lambda: tangent_line_round("arc"),
        lambda: tangent_circles(False),
        lambda: tangent_circles(True),
        tangent_line_ellipse,
        tangent_arc_line_at_point,
        tangent_ellipse_circle_at_point,
        tangent_bspline_line_at_end_pole,
    ],
    ids=[
        "line_circle",
        "line_arc",
        "circles_outside",
        "circles_inside",
        "line_ellipse",
        "arc_line_at_point",
        "ellipse_circle_at_point",
        "bspline_line_at_end_pole",
    ],
)
def test_tangent_holds_for_every_supported_pair(case: Case) -> None:
    b, check = case()

    a, s = solved(b)

    check(a, s)


# ---- PARALLEL, PERPENDICULAR, HORIZONTAL, VERTICAL, COLLINEAR --------------------------- #


def test_parallel_holds() -> None:
    b = sk.SketchBuilder()
    l1 = fixed_line(b, (0.0, 0.0), (3.0, 1.0))
    l2 = b.add(sk.Line(b.add(sk.Point(0.0, 2.0)), b.add(sk.Point(2.5, 3.6))))
    b.parallel(l1, l2)

    a, s = solved(b)

    assert abs(cross(unit(direction(a, s, l1)), unit(direction(a, s, l2)))) <= HOLD_TOL


def perpendicular_lines() -> tuple[sk.SketchBuilder, Check]:
    """A free line perpendicular to a fixed one."""
    b = sk.SketchBuilder()
    l1 = fixed_line(b, (0.0, 0.0), (3.0, 1.0))
    l2 = b.add(sk.Line(b.add(sk.Point(1.0, 0.5)), b.add(sk.Point(1.5, 3.0))))
    b.perpendicular(l1, l2)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        assert (
            abs(float(unit(direction(a, s, l1)) @ unit(direction(a, s, l2))))
            <= HOLD_TOL
        )

    return b, check


def perpendicular_line_circle() -> tuple[sk.SketchBuilder, Check]:
    """A free circle perpendicular to a fixed line: its center on it."""
    b = sk.SketchBuilder()
    line = fixed_line(b, (0.0, 0.0), (3.0, 1.0))
    c = b.add(sk.Point(1.0, 1.5))
    circle = b.add(sk.Circle(c, 0.5))
    b.perpendicular(line, circle)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        p0, _ = ends(a, s, line)
        assert abs(cross(pt(a, s, c) - p0, unit(direction(a, s, line)))) <= HOLD_TOL

    return b, check


def perpendicular_ellipse_line_at_point() -> tuple[sk.SketchBuilder, Check]:
    """A line perpendicular to a fixed ellipse at a point on it."""
    b = sk.SketchBuilder()
    e = fixed_ellipse(b)
    q = b.add(sk.Point(2.2, 1.2))
    far = b.add(sk.Point(3.4, 1.4))
    line = b.add(sk.Line(q, far))
    b.coincident(q, e)
    b.perpendicular(e, line, q)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        normal = unit(ellipse_normal(a, s, e, pt(a, s, q)))
        assert abs(cross(normal, unit(direction(a, s, line)))) <= HOLD_TOL

    return b, check


@pytest.mark.parametrize(
    "case",
    [
        perpendicular_lines,
        perpendicular_line_circle,
        perpendicular_ellipse_line_at_point,
    ],
    ids=["lines", "line_circle", "ellipse_line_at_point"],
)
def test_perpendicular_holds(case: Case) -> None:
    b, check = case()

    a, s = solved(b)

    check(a, s)


def test_horizontal_and_vertical_hold() -> None:
    b = sk.SketchBuilder()
    h = b.add(sk.Line(b.add(sk.Point(0.0, 0.0)), b.add(sk.Point(3.0, 0.4))))
    v = b.add(sk.Line(b.add(sk.Point(5.0, 0.0)), b.add(sk.Point(5.3, 2.0))))
    b.horizontal(h)
    b.vertical(v)

    a, s = solved(b)

    assert abs(float(direction(a, s, h)[1])) <= HOLD_TOL
    assert abs(float(direction(a, s, v)[0])) <= HOLD_TOL


@pytest.mark.parametrize("shared", [False, True], ids=["apart", "sharing_a_point"])
def test_collinear_holds(shared: bool) -> None:
    """With a shared point, the row puts only the other point on the first line."""
    b = sk.SketchBuilder()
    l1 = fixed_line(b, (0.0, 0.0), (3.0, 1.0))
    first = int(b.build().entities[l1, 2]) if shared else b.add(sk.Point(4.0, 1.8))
    l2 = b.add(sk.Line(first, b.add(sk.Point(6.0, 1.5))))
    b.collinear(l1, l2)

    a, s = solved(b)

    p0, _ = ends(a, s, l1)
    d = unit(direction(a, s, l1))
    for p in ends(a, s, l2):
        assert abs(cross(p - p0, d)) <= HOLD_TOL


# ---- EQUAL, MIDPOINT, RIGID ------------------------------------------------------------ #


def equal_lines() -> tuple[sk.SketchBuilder, Check]:
    """A free line as long as a fixed one."""
    b = sk.SketchBuilder()
    l1 = fixed_line(b, (0.0, 0.0), (3.0, 1.0))
    l2 = b.add(sk.Line(b.add(sk.Point(0.0, 2.0)), b.add(sk.Point(1.5, 3.0))))
    b.equal(l1, l2)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        lengths = [float(np.linalg.norm(direction(a, s, line))) for line in (l1, l2)]
        assert abs(lengths[0] - lengths[1]) <= HOLD_TOL

    return b, check


def equal_circle_arc() -> tuple[sk.SketchBuilder, Check]:
    """A free arc of the radius of a fixed circle."""
    b = sk.SketchBuilder()
    circle = b.add(sk.Circle(b.add(sk.Point(0.0, 0.0)), 2.0))
    b.fix(circle)
    c = b.add(sk.Point(5.0, 0.0))
    arc = b.add(sk.Arc(c, b.add(sk.Point(6.0, 0.0)), b.add(sk.Point(5.0, 1.0))))
    b.equal(circle, arc)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        center, r = round_of(a, s, arc)
        assert abs(r - 2.0) <= HOLD_TOL
        start = pt(a, s, int(a.entities[arc, 2]))
        assert abs(float(np.linalg.norm(start - center)) - 2.0) <= HOLD_TOL

    return b, check


def equal_ellipses() -> tuple[sk.SketchBuilder, Check]:
    """A free ellipse with the radii of a fixed one."""
    b = sk.SketchBuilder()
    e1 = fixed_ellipse(b)
    e2 = b.add(sk.Ellipse(b.add(sk.Point(5.0, 0.0)), b.add(sk.Point(5.8, 0.4)), 0.6))
    b.equal(e1, e2)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        _, _, b1, a1 = ellipse_of(a, s, e1)
        _, _, b2, a2 = ellipse_of(a, s, e2)
        assert abs(b1 - b2) <= HOLD_TOL
        assert abs(a1 - a2) <= HOLD_TOL

    return b, check


@pytest.mark.parametrize(
    "case",
    [equal_lines, equal_circle_arc, equal_ellipses],
    ids=["lines", "circle_arc", "ellipses"],
)
def test_equal_holds(case: Case) -> None:
    b, check = case()

    a, s = solved(b)

    check(a, s)


def midpoint_line() -> tuple[sk.SketchBuilder, Check]:
    """A free point at the middle of a fixed line."""
    b = sk.SketchBuilder()
    line = fixed_line(b, (0.0, 0.0), (3.0, 1.0))
    m = b.add(sk.Point(1.0, 1.2))
    b.midpoint(m, line)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        """The constraint holds on the solved geometry."""
        p, q = ends(a, s, line)
        np.testing.assert_allclose(pt(a, s, m), 0.5 * (p + q), rtol=0, atol=HOLD_TOL)

    return b, check


def midpoint_arc() -> tuple[sk.SketchBuilder, Check]:
    """A free point at the middle of a fixed half circle."""
    b = sk.SketchBuilder()
    arc = b.add(
        sk.Arc(
            b.add(sk.Point(0.0, 0.0)),
            b.add(sk.Point(2.0, 0.0)),
            b.add(sk.Point(-2.0, 0.0)),
        )
    )
    b.fix(arc)
    m = b.add(sk.Point(0.5, 1.5))
    b.midpoint(m, arc)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        # The upper half circle from angle 0 to pi: its midpoint is (0, 2).
        """The constraint holds on the solved geometry."""
        np.testing.assert_allclose(pt(a, s, m), (0.0, 2.0), rtol=0, atol=HOLD_TOL)

    return b, check


def midpoint_elliptic_arc() -> tuple[sk.SketchBuilder, Check]:
    """A free point at the middle of a fixed elliptic arc."""
    b = sk.SketchBuilder()
    arc = fixed_ellipse(b, arc=True)
    m = b.add(sk.Point(0.0, 1.5))
    b.midpoint(m, arc)

    def check(a: sk.SketchArrays, s: sk.SketchSolution) -> None:
        # The arc runs from parameter -1 to 2: its midpoint is at parameter 0.5.
        """The constraint holds on the solved geometry."""
        np.testing.assert_allclose(
            pt(a, s, m), ellipse_point(a, s, arc, 0.5), rtol=0, atol=HOLD_TOL
        )

    return b, check


@pytest.mark.parametrize(
    "case",
    [midpoint_line, midpoint_arc, midpoint_elliptic_arc],
    ids=["line", "arc", "elliptic_arc"],
)
def test_midpoint_holds(case: Case) -> None:
    b, check = case()

    a, s = solved(b)

    check(a, s)


def test_rigid_keeps_an_entity_at_its_initial_guess() -> None:
    """The second line pulls on the shared point; the rigid first line does not move."""
    b = sk.SketchBuilder()
    p = [b.add(sk.Point(x, y)) for x, y in ((0.0, 0.0), (2.0, 0.5), (4.0, 3.0))]
    first = b.add(sk.Line(p[0], p[1]))
    second = b.add(sk.Line(p[1], p[2]))
    b.rigid(first)
    b.length(second, 5.0)
    b.horizontal(second)

    a, s = solved(b)

    np.testing.assert_allclose(pt(a, s, p[0]), (0.0, 0.0), rtol=0, atol=HOLD_TOL)
    np.testing.assert_allclose(pt(a, s, p[1]), (2.0, 0.5), rtol=0, atol=HOLD_TOL)
    np.testing.assert_allclose(pt(a, s, p[2]), (7.0, 0.5), rtol=0, atol=HOLD_TOL)


# ---- LENGTH, ANGLE, RADIUS ------------------------------------------------------------- #


def test_length_holds() -> None:
    b = sk.SketchBuilder()
    line = b.add(sk.Line(b.add(sk.Point(0.0, 0.0)), b.add(sk.Point(2.0, 1.0))))
    b.length(line, 3.5)

    a, s = solved(b)

    assert abs(float(np.linalg.norm(direction(a, s, line))) - 3.5) <= HOLD_TOL


@pytest.mark.parametrize("theta", [0.7, -1.2, 2.5])
def test_angle_holds_counter_clockwise_from_the_first_line(theta: float) -> None:
    b = sk.SketchBuilder()
    l1 = fixed_line(b, (0.0, 0.0), (3.0, 0.0))
    guess = theta + 0.2
    l2 = b.add(
        sk.Line(
            b.add(sk.Point(0.0, 0.0)),
            b.add(sk.Point(2.0 * math.cos(guess), 2.0 * math.sin(guess))),
        )
    )
    b.angle(l1, l2, theta)

    a, s = solved(b)

    d1, d2 = direction(a, s, l1), direction(a, s, l2)
    assert abs(math.atan2(cross(d1, d2), float(d1 @ d2)) - theta) <= HOLD_TOL


@pytest.mark.parametrize("kind", ["circle", "arc"])
def test_radius_holds(kind: str) -> None:
    b = sk.SketchBuilder()
    c = b.add(sk.Point(0.0, 0.0))
    if kind == "circle":
        curve = b.add(sk.Circle(c, 1.0))
    else:
        curve = b.add(sk.Arc(c, b.add(sk.Point(1.0, 0.0)), b.add(sk.Point(0.0, 1.0))))
    b.radius(curve, 2.5)

    a, s = solved(b)

    center, r = round_of(a, s, curve)
    assert abs(r - 2.5) <= HOLD_TOL
    if kind == "arc":
        for col in (2, 3):
            point = pt(a, s, int(a.entities[curve, col]))
            assert abs(float(np.linalg.norm(point - center)) - 2.5) <= HOLD_TOL


# ---- INTERNAL_COINCIDENT, DISTANCE, SYMMETRIC ------------------------------------------ #


@pytest.mark.parametrize("arc", [False, True], ids=["ellipse", "elliptic_arc"])
@pytest.mark.parametrize("role", list(sk.SketchInternalRole)[:5], ids=lambda r: r.name)
def test_internal_coincident_puts_a_point_on_the_ellipses_characteristic_point(
    role: sk.SketchInternalRole, arc: bool
) -> None:
    """The axis ends are c +- a e1 and c +- b e2 (e2 = e1 turned a quarter turn
    counter-clockwise), the second focus 2c - f."""
    b = sk.SketchBuilder()
    e = fixed_ellipse(b, arc)
    q = b.add(sk.Point(0.3, 0.4))
    b.internal_coincident(e, q, role)

    a, s = solved(b)

    c, f, minor, major = ellipse_of(a, s, e)
    e1 = unit(f - c)
    e2 = np.array([-e1[1], e1[0]])
    expected = {
        sk.SketchInternalRole.MAJOR_POSITIVE: c + major * e1,
        sk.SketchInternalRole.MAJOR_NEGATIVE: c - major * e1,
        sk.SketchInternalRole.MINOR_POSITIVE: c + minor * e2,
        sk.SketchInternalRole.MINOR_NEGATIVE: c - minor * e2,
        sk.SketchInternalRole.FOCUS2: 2.0 * c - f,
    }[role]
    np.testing.assert_allclose(pt(a, s, q), expected, rtol=0, atol=HOLD_TOL)


def test_internal_coincident_puts_a_point_on_a_pole_and_on_a_knot() -> None:
    """A cubic B-spline of 5 poles with an inner knot at 0.5: the knot point is the
    curve's point at 0.5."""
    b = sk.SketchBuilder()
    xy = ((0.0, 0.0), (1.0, 2.0), (2.0, -1.0), (3.0, 2.0), (4.0, 0.0))
    poles = [b.add(sk.Point(x, y)) for x, y in xy]
    spline = b.add(sk.BSpline(tuple(poles), 3, (0.0, 0.5, 1.0), (4, 1, 4)))
    b.fix(spline)
    on_pole = b.add(sk.Point(1.5, 0.5))
    on_knot = b.add(sk.Point(2.5, 1.5))
    b.internal_coincident(spline, on_pole, sk.SketchInternalRole.POLE, 2)
    b.internal_coincident(spline, on_knot, sk.SketchInternalRole.KNOT, 1)

    a, s = solved(b)

    np.testing.assert_allclose(pt(a, s, on_pole), xy[2], rtol=0, atol=HOLD_TOL)
    knot_point = bspline_value(
        np.array(xy, dtype=np.float64), np.ones(5), [0.0, 0.5, 1.0], [4, 1, 4], 3, 0.5
    )
    np.testing.assert_allclose(pt(a, s, on_knot), knot_point, rtol=0, atol=HOLD_TOL)


@pytest.mark.parametrize("value", [2.0, 0.0])
def test_distance_between_two_points_holds_and_zero_is_a_coincidence(
    value: float,
) -> None:
    b = sk.SketchBuilder()
    p = b.add(sk.Point(0.0, 0.0))
    q = b.add(sk.Point(1.0, 0.7))
    b.fix(p)
    b.distance(p, q, value)

    a, s = solved(b)

    assert abs(float(np.linalg.norm(pt(a, s, q) - pt(a, s, p))) - value) <= HOLD_TOL


@pytest.mark.parametrize("value", [1.5, 0.0])
def test_distance_from_a_point_to_a_line_holds_and_zero_puts_it_on_the_line(
    value: float,
) -> None:
    b = sk.SketchBuilder()
    line = fixed_line(b, (0.0, 0.0), (3.0, 1.0))
    q = b.add(sk.Point(1.0, 2.5))
    b.distance(q, line, value)

    a, s = solved(b)

    p0, _ = ends(a, s, line)
    assert (
        abs(abs(cross(pt(a, s, q) - p0, unit(direction(a, s, line)))) - value)
        <= HOLD_TOL
    )


def test_symmetric_puts_the_second_point_at_the_mirror_image_of_the_first() -> None:
    """The mirror of (3, 1) in the line y = x is (1, 3)."""
    b = sk.SketchBuilder()
    line = fixed_line(b, (0.0, 0.0), (2.0, 2.0))
    p = b.add(sk.Point(3.0, 1.0))
    q = b.add(sk.Point(1.4, 2.5))
    b.fix(p)
    b.symmetric(p, q, line)

    a, s = solved(b)

    np.testing.assert_allclose(pt(a, s, q), (1.0, 3.0), rtol=0, atol=HOLD_TOL)
