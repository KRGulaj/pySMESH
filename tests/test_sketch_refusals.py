# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-22

"""Gates for pysmesh.sketch.solve's refusals: each names the operation, the row and why.

Every case breaks one thing in otherwise valid arrays and asserts the message: the
``sketch.solve:`` prefix, the entity or constraint row (or the array) at fault, and the
offending value. None of them tells the caller what to pass.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

import numpy as np
import pytest

import pysmesh as ps
import pysmesh.sketch as sk


def base() -> sk.SketchArrays:
    """Three points, a line, a circle; one horizontal and one length row.

    Entity rows: 0, 1, 2 points (params 0-5), 3 line (0 -> 1), 4 circle (center 2,
    radius param 6). Constraint rows: 0 horizontal (line 3), 1 length 2.0 (line 3).
    """
    b = sk.SketchBuilder()
    p = [b.add(sk.Point(x, y)) for x, y in ((0.0, 0.0), (1.0, 0.2), (3.0, 1.0))]
    line = b.add(sk.Line(p[0], p[1]))
    b.add(sk.Circle(p[2], 0.5))
    b.horizontal(line)
    b.length(line, 2.0)
    return b.build()


def refused(a: sk.SketchArrays, text: str) -> None:
    """solve(a) raises PysmeshError with `text` in its message, after the prefix."""
    with pytest.raises(ps.PysmeshError, match="^sketch.solve: ") as err:
        a.solve()
    assert text in str(err.value), str(err.value)


def with_entity(a: sk.SketchArrays, row: int, values: list[int]) -> sk.SketchArrays:
    """`a` with entity row `row` replaced (padded with -1)."""
    entities = a.entities.copy()
    entities[row] = values + [-1] * (sk.ENTITY_COLUMNS - len(values))
    return replace(a, entities=entities)


def with_constraint(
    a: sk.SketchArrays, values: list[int], value: float = math.nan
) -> sk.SketchArrays:
    """`a` with one more constraint row (padded with -1) and its value."""
    row = np.array(
        [values + [-1] * (sk.CONSTRAINT_COLUMNS - len(values))], dtype=np.int64
    )
    return replace(
        a,
        constraints=np.vstack([a.constraints, row]),
        values=np.append(a.values, value),
    )


K = sk.SketchConstraintKind
E = sk.SketchEntityKind


@pytest.mark.parametrize(
    ("change", "text"),
    [
        (
            lambda a: with_entity(a, 0, [E.POINT, 0, 7]),
            (
                "entity row 0 (POINT): column 2 (y) holds parameter index 7, outside "
                "params of length 7."
            ),
        ),
        (
            lambda a: with_entity(a, 3, [E.LINE, 0, 9]),
            (
                "entity row 3 (LINE): column 2 (end point) holds entity row 9, outside "
                "the 5 entity rows."
            ),
        ),
        (
            lambda a: with_constraint(a, [K.VERTICAL, 12]),
            (
                "constraint row 2 (VERTICAL): column 1 (line) holds entity row 12, "
                "outside the 5 entity rows."
            ),
        ),
        (
            lambda a: with_constraint(a, [K.TANGENT, 3, 4, -5]),
            (
                "constraint row 2 (TANGENT): column 3 (point) holds entity row -5, "
                "outside the 5 entity rows."
            ),
        ),
    ],
    ids=[
        "param_index",
        "entity_row_in_entity",
        "entity_row_in_constraint",
        "point_column",
    ],
)
def test_an_index_out_of_range_is_refused_naming_the_row(
    change: object, text: str
) -> None:
    refused(change(base()), text)  # type: ignore[operator]  # a lambda of the table


@pytest.mark.parametrize(
    ("change", "text"),
    [
        (
            lambda a: with_constraint(a, [K.HORIZONTAL, 4]),
            (
                "constraint row 2 (HORIZONTAL): column 1 names entity row 4 (CIRCLE); "
                "it takes a LINE."
            ),
        ),
        (
            lambda a: with_constraint(a, [K.RADIUS, 3], 1.0),
            (
                "constraint row 2 (RADIUS): column 1 names entity row 3 (LINE); it "
                "takes a CIRCLE or an ARC."
            ),
        ),
        (
            lambda a: with_constraint(a, [K.COINCIDENT, 3, 4]),
            (
                "constraint row 2 (COINCIDENT) names entity row 3 (LINE) and entity row "
                "4 (CIRCLE); one of them is a point."
            ),
        ),
        (
            lambda a: with_constraint(a, [K.EQUAL, 3, 4]),
            (
                "constraint row 2 (EQUAL) names entity row 3 (LINE) and entity row 4 "
                "(CIRCLE); equal takes two lines"
            ),
        ),
        (
            lambda a: with_constraint(a, [K.MIDPOINT, 0, 4]),
            (
                "constraint row 2 (MIDPOINT): column 2 names entity row 4 (CIRCLE); it "
                "takes a LINE, an ARC or an ELLIPTIC_ARC."
            ),
        ),
        (
            lambda a: with_entity(a, 3, [E.LINE, 0, 4]),
            (
                "entity row 3 (LINE): column 2 (end point) names entity row 4 (CIRCLE), "
                "which is not a POINT."
            ),
        ),
        (
            lambda a: with_constraint(a, [K.DISTANCE, 3, 4], 1.0),
            (
                "constraint row 2 (DISTANCE) names entity row 3 (LINE) and entity row 4 "
                "(CIRCLE); a distance takes two points or a point and a line."
            ),
        ),
        (
            lambda a: with_constraint(a, [K.SYMMETRIC, 0, 1, 4]),
            (
                "constraint row 2 (SYMMETRIC): column 3 names entity row 4 (CIRCLE); it "
                "takes a LINE."
            ),
        ),
    ],
    ids=[
        "horizontal_circle",
        "radius_line",
        "coincident_no_point",
        "equal_line_circle",
        "midpoint_circle",
        "line_of_a_circle",
        "distance_circle",
        "symmetric_about_a_circle",
    ],
)
def test_a_wrong_entity_kind_is_refused_naming_the_row_and_the_kinds(
    change: object, text: str
) -> None:
    refused(change(base()), text)  # type: ignore[operator]  # a lambda of the table


def test_a_tangent_of_two_lines_is_refused() -> None:
    b = sk.SketchBuilder()
    p = [b.add(sk.Point(x, y)) for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))]
    l1 = b.add(sk.Line(p[0], p[1]))
    l2 = b.add(sk.Line(p[0], p[2]))
    b.tangent(l1, l2)

    refused(
        b.build(),
        "constraint row 0 (TANGENT) names entity row 3 (LINE) and entity row 4 (LINE); a tangent needs a circle",
    )


def test_a_tangent_of_an_ellipse_and_a_circle_without_a_point_is_refused() -> None:
    b = sk.SketchBuilder()
    c = b.add(sk.Point(0.0, 0.0))
    e = b.add(sk.Ellipse(c, b.add(sk.Point(1.0, 0.0)), 0.5))
    circle = b.add(sk.Circle(b.add(sk.Point(3.0, 0.0)), 1.0))
    b.tangent(e, circle)

    refused(
        b.build(),
        (
            "constraint row 0 (TANGENT) names entity row 2 (ELLIPSE) and entity row 4 "
            "(CIRCLE) and no point; this pair needs the point where the curves meet in "
            "column 3"
        ),
    )


def test_a_tangent_to_a_bspline_off_its_end_poles_is_refused() -> None:
    b = sk.SketchBuilder()
    poles = [
        b.add(sk.Point(x, y))
        for x, y in ((0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0))
    ]
    spline = b.add(sk.BSpline(tuple(poles), 3, (0.0, 1.0), (4, 4)))
    line = b.add(sk.Line(poles[1], b.add(sk.Point(2.0, 3.0))))
    b.tangent(spline, line, poles[1])

    refused(
        b.build(),
        (
            "constraint row 0 (TANGENT) names entity row 4 (BSPLINE) at entity row 1 "
            "(POINT); PlaneGCS gives a B-spline's direction only at the first or the "
            "last pole"
        ),
    )


@pytest.mark.parametrize(
    ("row", "value", "text"),
    [
        (
            [K.LENGTH, 3],
            math.nan,
            (
                "constraint row 2 (LENGTH) is dimensional, and values[2] is nan; a "
                "dimension is a finite number."
            ),
        ),
        (
            [K.ANGLE, 3, 3],
            math.inf,
            "constraint row 2 (ANGLE) is dimensional, and values[2] is inf",
        ),
        (
            [K.VERTICAL, 3],
            1.0,
            (
                "constraint row 2 (VERTICAL) is not dimensional, and values[2] is 1; it "
                "holds NaN."
            ),
        ),
        (
            [K.LENGTH, 3],
            -1.0,
            "constraint row 2 (LENGTH): values[2] is -1; a length is above 0.",
        ),
        (
            [K.RADIUS, 4],
            0.0,
            "constraint row 2 (RADIUS): values[2] is 0; a radius is above 0.",
        ),
        (
            [K.DISTANCE, 0, 2],
            -0.5,
            "constraint row 2 (DISTANCE): values[2] is -0.5; a distance is 0 or above.",
        ),
    ],
    ids=[
        "nan_length",
        "inf_angle",
        "value_on_vertical",
        "negative_length",
        "zero_radius",
        "negative_distance",
    ],
)
def test_a_bad_value_is_refused_naming_the_row(
    row: list[int], value: float, text: str
) -> None:
    refused(with_constraint(base(), row, value), text)


@pytest.mark.parametrize(
    ("change", "text"),
    [
        (
            lambda a: replace(a, values=a.values[:1]),
            "values must have shape (2,) (got (1,)).",
        ),
        (
            lambda a: replace(a, fixed=a.fixed[:3]),
            "fixed must have shape (7,), one flag per parameter (got (3,)).",
        ),
        (
            lambda a: replace(a, entities=a.entities[:, :7]),
            "entities must have shape (rows, 8) (got (5, 7)).",
        ),
        (
            lambda a: replace(a, constraints=a.constraints[:, :4]),
            "constraints must have shape (rows, 5) (got (2, 4)).",
        ),
        (
            lambda a: replace(a, params=a.params.astype(np.int64)),
            "params must be a NumPy array of float64 (got dtype int64).",
        ),
        (
            lambda a: replace(a, entities=a.entities.astype(np.float64)),
            "entities must be a NumPy array of int64 (got dtype float64).",
        ),
        (
            lambda a: replace(a, fixed=a.fixed.astype(np.int64)),
            "fixed must be a NumPy array of bool (got dtype int64).",
        ),
    ],
    ids=[
        "values",
        "fixed",
        "entity_width",
        "constraint_width",
        "params_dtype",
        "entities_dtype",
        "fixed_dtype",
    ],
)
def test_mismatched_arrays_are_refused_naming_the_array(
    change: object, text: str
) -> None:
    refused(change(base()), text)  # type: ignore[operator]  # a lambda of the table


@pytest.mark.parametrize(
    ("change", "text"),
    [
        (
            lambda a: with_entity(a, 3, [99, 0, 1]),
            "entity row 3 has kind code 99, which is no entity kind",
        ),
        (
            lambda a: with_constraint(a, [0, 3]),
            "constraint row 2 has kind code 0, which is no constraint kind",
        ),
        (
            lambda a: with_entity(a, 3, [E.LINE, 0, 1, 2]),
            (
                "entity row 3 (LINE): column 3 holds 2; a LINE row uses columns 1 to 2 "
                "and holds -1 after them."
            ),
        ),
        (
            lambda a: with_constraint(a, [K.HORIZONTAL, 3, 3]),
            (
                "constraint row 2 (HORIZONTAL): column 2 holds 3; this kind uses "
                "columns 1 to 1 and holds -1 after them."
            ),
        ),
        (
            lambda a: with_constraint(a, [K.PARALLEL, 3, 3]),
            "constraint row 2 (PARALLEL) names entity row 3 (LINE) twice.",
        ),
    ],
    ids=[
        "entity_kind",
        "constraint_kind",
        "entity_column",
        "constraint_column",
        "twice",
    ],
)
def test_a_malformed_row_is_refused_naming_it(change: object, text: str) -> None:
    refused(change(base()), text)  # type: ignore[operator]  # a lambda of the table


def test_degenerate_initial_guesses_are_refused_naming_the_entity() -> None:
    a = base()
    params = a.params.copy()
    params[2:4] = params[0:2]

    refused(
        replace(a, params=params),
        "entity row 3 (LINE): its points (entity rows 0 and 1) are both at (0, 0)",
    )

    params = a.params.copy()
    params[6] = 0.0
    refused(
        replace(a, params=params),
        (
            "entity row 4 (CIRCLE): its radius (parameter 6) is 0 in the initial guess; "
            "a radius is above 0."
        ),
    )

    params = a.params.copy()
    params[5] = math.nan
    refused(
        replace(a, params=params),
        "params[5] is nan; every parameter of the initial guess is finite.",
    )


def test_an_ellipse_whose_focus_is_at_its_center_is_refused() -> None:
    b = sk.SketchBuilder()
    c = b.add(sk.Point(1.0, 1.0))
    b.add(sk.Ellipse(c, b.add(sk.Point(1.0, 1.0)), 0.5))

    refused(
        b.build(),
        "entity row 2 (ELLIPSE): its focus (entity row 1) is at its center (entity row 0)",
    )


def spline_arrays() -> sk.SketchArrays:
    """Four poles (rows 0-3) and a cubic Bezier B-spline (row 4, slot 0)."""
    b = sk.SketchBuilder()
    poles = [
        b.add(sk.Point(x, y))
        for x, y in ((0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0))
    ]
    b.add(sk.BSpline(tuple(poles), 3, (0.0, 1.0), (4, 4)))
    return b.build()


def test_bspline_definitions_are_refused_naming_the_row() -> None:
    a = spline_arrays()
    assert a.spline_offsets is not None and a.spline_indices is not None

    refused(
        replace(a, spline_offsets=None, spline_indices=None),
        "entity row 4 (BSPLINE) is a B-spline, and spline_offsets is not given",
    )
    refused(
        replace(a, spline_offsets=np.array([0, 5], dtype=np.int64)),
        "spline_offsets must start at 0 and end at the length of spline_indices (12)",
    )
    indices = a.spline_indices
    params = a.params.copy()
    params[indices[8]], params[indices[9]] = 1.0, 0.0  # knots 1.0, 0.0: decreasing
    refused(
        replace(a, params=params),
        (
            "entity row 4 (BSPLINE): OCCT refuses its definition (4 poles, degree 3, 2 "
            "knots, not periodic)"
        ),
    )
    params = a.params.copy()
    params[indices[5]] = -1.0
    refused(
        replace(a, params=params),
        (
            "entity row 4 (BSPLINE): weight 1 (parameter 9) is -1 in the initial guess; "
            "a weight is above 0."
        ),
    )
    entities = with_entity(a, 4, [E.BSPLINE, 3, 3, 0, 4]).entities
    refused(
        replace(a, entities=entities),
        (
            "entity row 4 (BSPLINE): column 1 (slot) holds 3, outside the 1 slots of "
            "spline_offsets."
        ),
    )
    indices = a.spline_indices.copy()
    indices[8] = int(a.entities[0, 1])  # a knot on a pole's x
    refused(
        replace(a, spline_indices=indices),
        (
            "entity row 4 (BSPLINE) uses parameter 0 as knot 0, and another slot uses "
            "it as a value that moves"
        ),
    )


def test_bad_solver_limits_are_refused() -> None:
    a = base()

    with pytest.raises(
        ps.PysmeshError,
        match=re.escape("sketch.solve: max_iterations is 0; it is 1 or more."),
    ):
        a.solve(max_iterations=0)
    with pytest.raises(
        ps.PysmeshError,
        match=re.escape("sketch.solve: max_iterations is 2.5; it is an int"),
    ):
        a.solve(max_iterations=2.5)  # type: ignore[arg-type]  # the refusal under test
    with pytest.raises(
        ps.PysmeshError,
        match=re.escape("sketch.solve: tolerance is 0; it is a finite number above 0."),
    ):
        a.solve(tolerance=0.0)
    with pytest.raises(
        ps.PysmeshError, match=re.escape("sketch.solve: tolerance is nan")
    ):
        a.solve(tolerance=math.nan)


def test_builder_refusals_name_the_row() -> None:
    b = sk.SketchBuilder()
    p = b.add(sk.Point(0.0, 0.0))
    line = b.add(sk.Line(p, b.add(sk.Point(1.0, 0.0))))

    with pytest.raises(
        ps.PysmeshError,
        match=re.escape(
            "SketchBuilder.add: the arc's start is entity row 2, which is not a POINT row"
        ),
    ):
        b.add(sk.Arc(p, line, p))
    with pytest.raises(
        ps.PysmeshError,
        match=re.escape("SketchBuilder.add: the B-spline has 2 poles, 1 weights"),
    ):
        b.add(sk.BSpline((p, p), 1, (0.0, 1.0), (2, 2), weights=(1.0,)))
    with pytest.raises(
        ps.PysmeshError,
        match=re.escape(
            "SketchBuilder.fix: entity row 9 is not a row of this builder."
        ),
    ):
        b.fix(9)
