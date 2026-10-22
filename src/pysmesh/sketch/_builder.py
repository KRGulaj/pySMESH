# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-22

"""Thin builders over the sketch arrays: frozen entity descriptions and SketchBuilder.

Part of :mod:`pysmesh.sketch`. :class:`SketchBuilder` appends rows and returns the arrays
as :class:`SketchArrays`; :func:`~pysmesh.sketch.solve` never needs it. Every method that
appends a row returns that row, which is the entity's or the constraint's id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError
from ._kinds import SketchConstraintKind, SketchEntityKind, SketchInternalRole
from ._solve import (
    _DEFAULT_MAX_ITERATIONS,
    _DEFAULT_TOLERANCE,
    CONSTRAINT_COLUMNS,
    ENTITY_COLUMNS,
    SketchSolution,
    solve,
)


@dataclass(frozen=True)
class Point:
    """A point.

    Attributes:
        x: The initial x coordinate.
        y: The initial y coordinate.
    """

    x: float
    y: float


@dataclass(frozen=True)
class Line:
    """A line segment between two points.

    Attributes:
        start: The start point's row.
        end: The end point's row.
    """

    start: int
    end: int


@dataclass(frozen=True)
class Circle:
    """A full circle.

    Attributes:
        center: The center point's row.
        radius: The initial radius, above 0.
    """

    center: int
    radius: float


@dataclass(frozen=True)
class Arc:
    """A circular arc, counter-clockwise from its start point to its end point.

    The builder computes the radius (the distance from the center to the start) and the
    two angles (each point's direction from the center, the end angle above the start
    angle by at most a full turn) from the points' initial coordinates.

    Attributes:
        center: The center point's row.
        start: The start point's row.
        end: The end point's row.
    """

    center: int
    start: int
    end: int


@dataclass(frozen=True)
class Ellipse:
    """A full ellipse.

    Attributes:
        center: The center point's row.
        focus: The row of the focus on the positive major axis.
        minor_radius: The initial minor radius, above 0.
    """

    center: int
    focus: int
    minor_radius: float


@dataclass(frozen=True)
class EllipticArc:
    """An elliptic arc, counter-clockwise from its start point to its end point.

    The builder computes the two angles (the ellipse's own parameter of each point,
    measured from the major axis, the end above the start by at most a full turn) from
    the points' initial coordinates.

    Attributes:
        center: The center point's row.
        focus: The row of the focus on the positive major axis.
        start: The start point's row.
        end: The end point's row.
        minor_radius: The initial minor radius, above 0.
    """

    center: int
    focus: int
    start: int
    end: int
    minor_radius: float


@dataclass(frozen=True)
class BSpline:
    """A B-spline curve.

    Attributes:
        poles: The pole points' rows.
        degree: The degree, 1 to 25.
        knots: The knot values, increasing. They never move.
        multiplicities: One per knot.
        weights: One per pole, above 0; None for 1.0 each.
        periodic: Whether the curve is periodic.
        rational: Whether the solver may move the weights. When False, the builder fixes
            them, and the curve keeps its weights.
    """

    poles: tuple[int, ...]
    degree: int
    knots: tuple[float, ...]
    multiplicities: tuple[int, ...]
    weights: tuple[float, ...] | None = None
    periodic: bool = False
    rational: bool = False


SketchEntity: TypeAlias = Point | Line | Circle | Arc | Ellipse | EllipticArc | BSpline


@dataclass(frozen=True)
class SketchArrays:
    """A sketch as the arrays :func:`~pysmesh.sketch.solve` takes.

    Attributes:
        params: (N,) float64, every scalar of the sketch.
        entities: (E, 8) int64, one row per entity.
        constraints: (C, 5) int64, one row per constraint.
        values: (C,) float64, each row's dimension, NaN for a non-dimensional row.
        fixed: (N,) bool, the parameters the solver does not move.
        spline_offsets: (S+1,) int64, or None without a B-spline.
        spline_indices: (L,) int64, or None without a B-spline.
    """

    params: NDArray[np.float64]
    entities: NDArray[np.int64]
    constraints: NDArray[np.int64]
    values: NDArray[np.float64]
    fixed: NDArray[np.bool_]
    spline_offsets: NDArray[np.int64] | None
    spline_indices: NDArray[np.int64] | None

    def solve(
        self,
        *,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> SketchSolution:
        """Solve these arrays with :func:`~pysmesh.sketch.solve`.

        Args:
            max_iterations: The iteration limit of each PlaneGCS solver run.
            tolerance: The residual at which a run stops as solved.

        Returns:
            The solution.
        """
        return solve(
            self.params,
            self.entities,
            self.constraints,
            self.values,
            fixed=self.fixed,
            spline_offsets=self.spline_offsets,
            spline_indices=self.spline_indices,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    def point(
        self, row: int, params: NDArray[np.float64] | None = None
    ) -> tuple[float, float]:
        """The coordinates of a point row.

        Args:
            row: A ``POINT`` row of ``entities``.
            params: The parameters to read, such as a solution's; None for these.

        Returns:
            (x, y).

        Raises:
            PysmeshError: If ``row`` is not a ``POINT`` row.
        """
        if (
            not 0 <= row < len(self.entities)
            or self.entities[row, 0] != SketchEntityKind.POINT
        ):
            raise PysmeshError(
                f"SketchArrays.point: entity row {row} is not a POINT row."
            )
        source = self.params if params is None else params
        return float(source[self.entities[row, 1]]), float(
            source[self.entities[row, 2]]
        )


def _sweep(a0: float, a1: float) -> float:
    """The end angle a1 moved by full turns to lie in (a0, a0 + 2 pi]."""
    while a1 <= a0:
        a1 += 2.0 * math.pi
    while a1 > a0 + 2.0 * math.pi:
        a1 -= 2.0 * math.pi
    return a1


class SketchBuilder:
    """Appends sketch rows and returns the arrays.

    A thin helper over the arrays of :func:`~pysmesh.sketch.solve`. :meth:`add` appends
    an entity and its parameters; each constraint method appends one constraint row;
    :meth:`build` returns the arrays. Every appended row's index is its id, which the
    solution's diagnosis uses.

    Example:
        A 3 x 2 rectangle with its first corner held at the origin::

            b = SketchBuilder()
            p = [b.add(Point(x, y)) for x, y in ((0, 0), (2.9, 0.1), (3.1, 2), (0, 1.9))]
            lines = [b.add(Line(p[k], p[(k + 1) % 4])) for k in range(4)]
            b.fix(p[0])
            b.horizontal(lines[0])
            b.horizontal(lines[2])
            b.vertical(lines[1])
            b.vertical(lines[3])
            b.length(lines[0], 3.0)
            b.length(lines[1], 2.0)
            solution = b.build().solve()
    """

    def __init__(self) -> None:
        """An empty sketch."""
        self._params: list[float] = []
        self._fixed: list[bool] = []
        self._entities: list[list[int]] = []
        self._constraints: list[list[int]] = []
        self._values: list[float] = []
        self._offsets: list[int] = [0]
        self._indices: list[int] = []

    # ---- entities ----------------------------------------------------------------------- #

    def _param(self, value: float, *, fixed: bool = False) -> int:
        """Append one parameter; its index."""
        self._params.append(float(value))
        self._fixed.append(fixed)
        return len(self._params) - 1

    def _entity_row(self, kind: SketchEntityKind, *columns: int) -> int:
        """Append one entity row, padded with -1; its index."""
        row = [int(kind), *(int(c) for c in columns)]
        self._entities.append(row + [-1] * (ENTITY_COLUMNS - len(row)))
        return len(self._entities) - 1

    def _point_xy(self, row: int, what: str) -> tuple[float, float]:
        """The coordinates of a point row, refused unless it is one."""
        if not 0 <= row < len(self._entities) or (
            self._entities[row][0] != SketchEntityKind.POINT
        ):
            raise PysmeshError(
                f"SketchBuilder.add: {what} is entity row {row}, which is not a POINT row "
                f"of this builder."
            )
        entry = self._entities[row]
        return self._params[entry[1]], self._params[entry[2]]

    def add(self, entity: SketchEntity) -> int:
        """Append an entity and its parameters.

        Args:
            entity: The entity. Its point rows are rows this builder appended.

        Returns:
            The entity's row.

        Raises:
            PysmeshError: If a point row is not a ``POINT`` row of this builder, an
                elliptic arc's focus is at its center or its minor radius is not above 0,
                or a B-spline's weights or multiplicities do not match its poles or knots.
        """
        if isinstance(entity, Point):
            return self._entity_row(
                SketchEntityKind.POINT, self._param(entity.x), self._param(entity.y)
            )
        if isinstance(entity, Line):
            self._point_xy(entity.start, "the line's start")
            self._point_xy(entity.end, "the line's end")
            return self._entity_row(SketchEntityKind.LINE, entity.start, entity.end)
        if isinstance(entity, Circle):
            self._point_xy(entity.center, "the circle's center")
            return self._entity_row(
                SketchEntityKind.CIRCLE, entity.center, self._param(entity.radius)
            )
        if isinstance(entity, Arc):
            return self._add_arc(entity)
        if isinstance(entity, Ellipse):
            self._point_xy(entity.center, "the ellipse's center")
            self._point_xy(entity.focus, "the ellipse's focus")
            return self._entity_row(
                SketchEntityKind.ELLIPSE,
                entity.center,
                entity.focus,
                self._param(entity.minor_radius),
            )
        if isinstance(entity, EllipticArc):
            return self._add_elliptic_arc(entity)
        return self._add_bspline(entity)

    def _add_arc(self, arc: Arc) -> int:
        """Append an arc with its radius and angles from its points."""
        cx, cy = self._point_xy(arc.center, "the arc's center")
        sx, sy = self._point_xy(arc.start, "the arc's start")
        ex, ey = self._point_xy(arc.end, "the arc's end")
        a0 = math.atan2(sy - cy, sx - cx)
        a1 = _sweep(a0, math.atan2(ey - cy, ex - cx))
        return self._entity_row(
            SketchEntityKind.ARC,
            arc.center,
            arc.start,
            arc.end,
            self._param(math.hypot(sx - cx, sy - cy)),
            self._param(a0),
            self._param(a1),
        )

    def _add_elliptic_arc(self, arc: EllipticArc) -> int:
        """Append an elliptic arc with its angles from its points."""
        cx, cy = self._point_xy(arc.center, "the elliptic arc's center")
        fx, fy = self._point_xy(arc.focus, "the elliptic arc's focus")
        sx, sy = self._point_xy(arc.start, "the elliptic arc's start")
        ex, ey = self._point_xy(arc.end, "the elliptic arc's end")
        c = math.hypot(fx - cx, fy - cy)
        b = float(arc.minor_radius)
        if not c > 0.0 or not b > 0.0:
            raise PysmeshError(
                f"SketchBuilder.add: the elliptic arc's focus is {c} from its center and "
                f"its minor radius is {b}; both are above 0."
            )
        ux, uy = (fx - cx) / c, (fy - cy) / c
        a = math.hypot(c, b)

        def parameter(px: float, py: float) -> float:
            """The ellipse's parameter of a point."""
            dx, dy = px - cx, py - cy
            return math.atan2((-uy * dx + ux * dy) / b, (ux * dx + uy * dy) / a)

        a0 = parameter(sx, sy)
        a1 = _sweep(a0, parameter(ex, ey))
        return self._entity_row(
            SketchEntityKind.ELLIPTIC_ARC,
            arc.center,
            arc.focus,
            arc.start,
            arc.end,
            self._param(b),
            self._param(a0),
            self._param(a1),
        )

    def _add_bspline(self, spline: BSpline) -> int:
        """Append a B-spline, its weight and knot parameters and its slot."""
        n = len(spline.poles)
        for k, pole in enumerate(spline.poles):
            self._point_xy(pole, f"pole {k} of the B-spline")
        weights = spline.weights if spline.weights is not None else (1.0,) * n
        if len(weights) != n or len(spline.knots) != len(spline.multiplicities):
            raise PysmeshError(
                f"SketchBuilder.add: the B-spline has {n} poles, {len(weights)} weights, "
                f"{len(spline.knots)} knots and {len(spline.multiplicities)} "
                f"multiplicities; there is one weight per pole and one multiplicity per "
                f"knot."
            )
        weight_params = [self._param(w, fixed=not spline.rational) for w in weights]
        knot_params = [self._param(k, fixed=True) for k in spline.knots]
        slot = len(self._offsets) - 1
        self._indices.extend(
            [*spline.poles, *weight_params, *knot_params, *spline.multiplicities]
        )
        self._offsets.append(len(self._indices))
        return self._entity_row(
            SketchEntityKind.BSPLINE, slot, spline.degree, int(spline.periodic), n
        )

    def _moving_params(self, row: int) -> list[int]:
        """The parameters an entity row moves: its points' coordinates, then its scalars."""
        entry = self._entities[row]
        kind = SketchEntityKind(entry[0])
        if kind == SketchEntityKind.POINT:
            return [entry[1], entry[2]]
        if kind == SketchEntityKind.BSPLINE:
            n = entry[4]
            slot = self._indices[self._offsets[entry[1]] : self._offsets[entry[1] + 1]]
            points, scalars = slot[:n], slot[n : 2 * n]
        elif kind == SketchEntityKind.LINE:
            points, scalars = entry[1:3], []
        elif kind == SketchEntityKind.CIRCLE:
            points, scalars = entry[1:2], entry[2:3]
        elif kind == SketchEntityKind.ARC:
            points, scalars = entry[1:4], entry[4:7]
        elif kind == SketchEntityKind.ELLIPSE:
            points, scalars = entry[1:3], entry[3:4]
        else:
            points, scalars = entry[1:5], entry[5:8]
        out: list[int] = []
        for point in points:
            out += self._moving_params(point)
        return out + scalars

    def fix(self, entity: int) -> None:
        """Mark every parameter an entity moves as fixed: the solver keeps them.

        Args:
            entity: An entity row of this builder.

        Raises:
            PysmeshError: If ``entity`` is not a row of this builder.
        """
        if not 0 <= entity < len(self._entities):
            raise PysmeshError(
                f"SketchBuilder.fix: entity row {entity} is not a row of this builder."
            )
        for i in self._moving_params(entity):
            self._fixed[i] = True

    # ---- constraints -------------------------------------------------------------------- #

    def _constraint_row(
        self, kind: SketchConstraintKind, *columns: int, value: float = math.nan
    ) -> int:
        """Append one constraint row, padded with -1, and its value; its index."""
        row = [int(kind), *(int(c) for c in columns)]
        self._constraints.append(row + [-1] * (CONSTRAINT_COLUMNS - len(row)))
        self._values.append(float(value))
        return len(self._constraints) - 1

    def coincident(self, a: int, b: int) -> int:
        """Two points coincide, or a point lies on a curve.

        Args:
            a: A point or a curve.
            b: A point or a curve; one of the two is a point.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.COINCIDENT, a, b)

    def tangent(self, a: int, b: int, point: int | None = None) -> int:
        """Two curves are tangent.

        Args:
            a: A curve.
            b: A curve.
            point: Where they touch: a point on both curves (an end point they share,
                or a point that ``COINCIDENT`` rows put on both). Needed for a pair with
                an ellipse that is not a line, and for a B-spline (its end pole).

        Returns:
            The constraint's row.
        """
        return self._constraint_row(
            SketchConstraintKind.TANGENT, a, b, -1 if point is None else point
        )

    def parallel(self, a: int, b: int) -> int:
        """Two lines are parallel.

        Args:
            a: A line.
            b: A line.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.PARALLEL, a, b)

    def perpendicular(self, a: int, b: int, point: int | None = None) -> int:
        """Two curves are perpendicular.

        Args:
            a: A curve.
            b: A curve.
            point: Where they cross, for a pair with an ellipse, an elliptic arc or a
                B-spline (its end pole).

        Returns:
            The constraint's row.
        """
        return self._constraint_row(
            SketchConstraintKind.PERPENDICULAR, a, b, -1 if point is None else point
        )

    def horizontal(self, line: int) -> int:
        """A line is horizontal.

        Args:
            line: A line.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.HORIZONTAL, line)

    def vertical(self, line: int) -> int:
        """A line is vertical.

        Args:
            line: A line.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.VERTICAL, line)

    def equal(self, a: int, b: int) -> int:
        """Two lines have equal lengths, or two round or elliptic curves equal radii.

        Args:
            a: A line, circle, arc, ellipse or elliptic arc.
            b: One of the same family.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.EQUAL, a, b)

    def collinear(self, a: int, b: int) -> int:
        """Two lines lie on one line.

        Args:
            a: A line.
            b: A line.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.COLLINEAR, a, b)

    def midpoint(self, point: int, curve: int) -> int:
        """A point is the midpoint of a line or an arc.

        Args:
            point: A point.
            curve: A line, an arc or an elliptic arc.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.MIDPOINT, point, curve)

    def rigid(self, entity: int) -> int:
        """An entity keeps its initial guess.

        Args:
            entity: Any entity.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.RIGID, entity)

    def length(self, line: int, value: float) -> int:
        """A line has a length.

        Args:
            line: A line.
            value: The length, above 0, in model units.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.LENGTH, line, value=value)

    def angle(self, a: int, b: int, value: float) -> int:
        """Two lines make an angle.

        Args:
            a: A line.
            b: A line.
            value: The angle in radians from ``a``'s direction to ``b``'s,
                counter-clockwise positive.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.ANGLE, a, b, value=value)

    def radius(self, curve: int, value: float) -> int:
        """A circle or an arc has a radius.

        Args:
            curve: A circle or an arc.
            value: The radius, above 0, in model units.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.RADIUS, curve, value=value)

    def internal_coincident(
        self, curve: int, point: int, role: SketchInternalRole, index: int | None = None
    ) -> int:
        """A point is one of a curve's characteristic points.

        Args:
            curve: An ellipse, an elliptic arc or a B-spline.
            point: A point.
            role: Which point: an axis end or the second focus of an ellipse, a pole or a
                knot of a B-spline.
            index: The pole's or the knot's index; None for an ellipse.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(
            SketchConstraintKind.INTERNAL_COINCIDENT,
            curve,
            point,
            int(role),
            -1 if index is None else index,
        )

    def distance(self, a: int, b: int, value: float) -> int:
        """Two points, or a point and a line, are at a distance.

        Args:
            a: A point or a line.
            b: A point or a line; one of the two is a point.
            value: The distance, 0 or above, in model units.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.DISTANCE, a, b, value=value)

    def symmetric(self, a: int, b: int, line: int) -> int:
        """Two points are mirror images about a line.

        Args:
            a: A point.
            b: A point.
            line: The mirror line.

        Returns:
            The constraint's row.
        """
        return self._constraint_row(SketchConstraintKind.SYMMETRIC, a, b, line)

    def build(self) -> SketchArrays:
        """The arrays of the rows appended so far.

        Returns:
            New arrays; later appends do not change them.
        """
        splines = len(self._offsets) > 1
        return SketchArrays(
            params=np.array(self._params, dtype=np.float64),
            entities=np.array(self._entities, dtype=np.int64).reshape(
                -1, ENTITY_COLUMNS
            ),
            constraints=np.array(self._constraints, dtype=np.int64).reshape(
                -1, CONSTRAINT_COLUMNS
            ),
            values=np.array(self._values, dtype=np.float64),
            fixed=np.array(self._fixed, dtype=np.bool_),
            spline_offsets=np.array(self._offsets, dtype=np.int64) if splines else None,
            spline_indices=np.array(self._indices, dtype=np.int64) if splines else None,
        )
