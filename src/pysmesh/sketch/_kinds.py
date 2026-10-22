# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-22

"""The kind codes of the sketch tables, and the statuses of a solve.

Part of :mod:`pysmesh.sketch`. Each kind code is a stable public value: a value never
changes once released, so a sketch stored as arrays reads the same in every later version.
``docs/documentation/concepts/sketch.md`` holds the full table layout.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class SketchEntityKind(IntEnum):
    """Column 0 of an ``entities`` row: the kind of the entity.

    The other columns of each kind, in order (unused columns hold -1). A "parameter" is an
    index into ``params``; a "point" is a ``POINT`` row of ``entities``.

    Attributes:
        POINT: x, y (parameters).
        LINE: start, end (points).
        CIRCLE: center (point), radius (parameter).
        ARC: center, start, end (points), radius, start angle, end angle (parameters).
            The arc runs counter-clockwise from its start angle to its end angle.
        ELLIPSE: center, focus (points), minor radius (parameter). The focus gives the
            major axis; the major radius is ``sqrt(|focus - center|^2 + minor^2)``.
        ELLIPTIC_ARC: center, focus, start, end (points), minor radius, start angle, end
            angle (parameters). The angles are the ellipse's own parameter, measured from
            the major axis.
        BSPLINE: slot, degree, periodic (0 or 1), pole count. The slot is the B-spline's
            entry in ``spline_offsets`` and ``spline_indices``.
    """

    POINT = 1
    LINE = 2
    CIRCLE = 3
    ARC = 4
    ELLIPSE = 5
    ELLIPTIC_ARC = 6
    BSPLINE = 7


class SketchConstraintKind(IntEnum):
    """Column 0 of a ``constraints`` row: the kind of the constraint.

    The other columns hold entity rows, in order (unused columns hold -1). The four
    dimensional kinds take their dimension from ``values``; the others hold NaN there.

    Attributes:
        COINCIDENT: two points, or a point and a line, circle, arc (its full circle),
            ellipse, elliptic arc (its full ellipse) or B-spline, in either order.
        TANGENT: two curves, and an optional point (column 3) where they touch. A pair
            with an ellipse that is not a line, and every pair with a B-spline, needs it.
        PARALLEL: two lines.
        PERPENDICULAR: two curves, and an optional point (column 3). A pair with an
            ellipse, an elliptic arc or a B-spline needs it.
        HORIZONTAL: one line.
        VERTICAL: one line.
        EQUAL: two lines (lengths), two circles or arcs (radii), or two ellipses or
            elliptic arcs (both radii).
        COLLINEAR: two lines.
        MIDPOINT: a point, then a line, an arc or an elliptic arc.
        RIGID: one entity, held at its initial guess.
        LENGTH: one line. Dimensional: the length, above 0.
        ANGLE: two lines. Dimensional: the angle in radians from the first line's
            direction (start to end) to the second's, counter-clockwise positive.
        RADIUS: one circle or arc. Dimensional: the radius, above 0.
        INTERNAL_COINCIDENT: a curve, a point, a role (column 3, a
            :class:`SketchInternalRole`) and an index (column 4: -1 for an ellipse, the
            pole or knot index for a B-spline).
        DISTANCE: two points, or a point and a line, in either order. Dimensional: the
            distance, 0 or above; 0 makes it a coincidence, as Shaper's.
        SYMMETRIC: two points, then a line: the points are mirror images about the line.
    """

    COINCIDENT = 1
    TANGENT = 2
    PARALLEL = 3
    PERPENDICULAR = 4
    HORIZONTAL = 5
    VERTICAL = 6
    EQUAL = 7
    COLLINEAR = 8
    MIDPOINT = 9
    RIGID = 10
    LENGTH = 11
    ANGLE = 12
    RADIUS = 13
    INTERNAL_COINCIDENT = 14
    DISTANCE = 15
    SYMMETRIC = 16


class SketchInternalRole(IntEnum):
    """Column 3 of an ``INTERNAL_COINCIDENT`` row: which characteristic point it binds.

    Attributes:
        MAJOR_POSITIVE: the end of the major axis on the focus side.
        MAJOR_NEGATIVE: the other end of the major axis.
        MINOR_POSITIVE: the end of the minor axis a quarter turn counter-clockwise from
            the major positive end.
        MINOR_NEGATIVE: the other end of the minor axis.
        FOCUS2: the second focus.
        POLE: a B-spline pole; column 4 is its index.
        KNOT: the point of a B-spline at a knot; column 4 is the knot's index.
    """

    MAJOR_POSITIVE = 1
    MAJOR_NEGATIVE = 2
    MINOR_POSITIVE = 3
    MINOR_NEGATIVE = 4
    FOCUS2 = 5
    POLE = 6
    KNOT = 7


class SketchStatus(StrEnum):
    """The outcome of :func:`pysmesh.sketch.solve`.

    Attributes:
        SOLVED: every constraint holds and no degree of freedom is left.
        UNDER_CONSTRAINED: every constraint holds, and ``dof`` degrees of freedom are
            left; the solution is the one nearest the initial guess that the solver
            reached.
        OVER_CONSTRAINED: PlaneGCS's diagnosis found conflicting constraints
            (``conflicting``); nothing is solved, and ``params`` is the initial guess.
        FAILED: no conflict was found, and the solver did not bring every constraint
            within the tolerance; ``params`` is the initial guess.
    """

    SOLVED = "SOLVED"
    UNDER_CONSTRAINED = "UNDER_CONSTRAINED"
    OVER_CONSTRAINED = "OVER_CONSTRAINED"
    FAILED = "FAILED"
