# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-22

"""The sketch solver: :func:`solve` and its result, :class:`SketchSolution`.

Part of :mod:`pysmesh.sketch`. The solver is a stateless free function over NumPy arrays.
It touches no :class:`~pysmesh.Session`, and it keeps no state between calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError
from .._core import sketch_solve as _sketch_solve
from ._kinds import SketchStatus

# The widths of the two tables: column 0 is the kind, the other columns its operands.
ENTITY_COLUMNS: Final[int] = 8
CONSTRAINT_COLUMNS: Final[int] = 5

# PlaneGCS's own defaults (GCS::System): 100 iterations for each solver run, and 1e-10 for
# DogLeg's largest residual, Levenberg-Marquardt's residual norm and BFGS's step. Shaper
# solves with them unchanged.
_DEFAULT_MAX_ITERATIONS: Final[int] = 100
_DEFAULT_TOLERANCE: Final[float] = 1e-10


@dataclass(frozen=True)
class SketchSolution:
    """The answer of :func:`solve`.

    Every id is a row index: a constraint's id is its row of ``constraints``. The arrays
    of rows are sorted ascending.

    Attributes:
        params: (N,) float64, a new array: the solved parameters when the status is
            ``SOLVED`` or ``UNDER_CONSTRAINED``, else a copy of the initial guess. A
            parameter that no entity uses, a knot, and a fixed parameter keep their
            values.
        status: The outcome.
        dof: The degrees of freedom left, from PlaneGCS's diagnosis: the unknowns less
            the rank of the constraint Jacobian.
        conflicting: (K,) int64, the constraint rows PlaneGCS names as conflicting.
        redundant: (K,) int64, the rows that are fully redundant: satisfied by the
            others, so they remove no degree of freedom.
        partially_redundant: (K,) int64, the rows of which some PlaneGCS equations are
            redundant and some are not.
    """

    params: NDArray[np.float64]
    status: SketchStatus
    dof: int
    conflicting: NDArray[np.int64]
    redundant: NDArray[np.int64]
    partially_redundant: NDArray[np.int64]


def solve(
    params: NDArray[np.float64],
    entities: NDArray[np.int64],
    constraints: NDArray[np.int64],
    values: NDArray[np.float64],
    *,
    fixed: NDArray[np.bool_] | None = None,
    spline_offsets: NDArray[np.int64] | None = None,
    spline_indices: NDArray[np.int64] | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> SketchSolution:
    """Solve a 2-D sketch, given as flat arrays, with FreeCAD's PlaneGCS.

    The tables, the kind codes, the units and the sign conventions are the contract of
    ``docs/documentation/concepts/sketch.md``. :class:`~pysmesh.sketch.SketchBuilder`
    writes the arrays from frozen entity descriptions; ``solve`` does not need it.

    **The result depends on the initial guess.** ``params`` is the starting point of the
    solver. An under-constrained sketch keeps the solution nearest the guess that the
    solver reaches, and a choice between mirror solutions (the side of a tangency, the
    sign of a perpendicular angle) follows the guess. A caller that replays a sketch
    passes the same initial guess, and gets bit-identical ``params`` back.

    **How it solves.** Each constraint row becomes PlaneGCS equations as SALOME Shaper's
    PlaneGCS adapter maps it (the contract page lists the ``GCS::System`` calls). The
    unknowns are the parameters that an entity moves (point coordinates, radii, arc
    angles, minor radii, B-spline weights) and that ``fixed`` does not hold, plus the
    auxiliary unknowns some constraints need. Knots never move. The solve is Shaper's:
    diagnose and solve with DogLeg; when DogLeg fails, diagnose and solve with
    Levenberg-Marquardt; when that fails, with BFGS. One fixed algorithm with fixed
    limits, on the calling thread, with the GIL released: identical arrays give
    bit-identical results.

    Args:
        params: (N,) float64. Every scalar of the sketch, finite: the unknowns' initial
            guess and the known values. Lengths are in model units, angles in radians.
        entities: (E, 8) int64. One row per entity; its id is its row.
        constraints: (C, 5) int64. One row per constraint; its id is its row.
        values: (C,) float64. The dimension of each ``LENGTH``, ``ANGLE``, ``RADIUS``
            and ``DISTANCE`` row; NaN in the other rows.
        fixed: (N,) bool, or None for none. The parameters the solver does not move.
        spline_offsets: (S+1,) int64, or None when there is no B-spline. Slot ``k`` of
            the B-spline table is ``spline_indices[spline_offsets[k]:spline_offsets[k+1]]``.
        spline_indices: (L,) int64, or None when there is no B-spline. Each slot holds
            the pole rows (one per pole), the weight parameters (one per pole), the knot
            parameters and the knot multiplicities (one per knot), in that order.
        max_iterations: The iteration limit of each PlaneGCS solver run, 1 or more.
        tolerance: Above 0: DogLeg's largest constraint residual, Levenberg-Marquardt's
            residual norm and BFGS's step at which a run stops as solved.

    Returns:
        The solved parameters, the status, the degrees of freedom left, and PlaneGCS's
        diagnosis as constraint rows.

    Raises:
        PysmeshError: ``sketch.solve:`` and the reason, naming the entity or constraint
            row, if an array has the wrong dtype or shape, an index is out of range, a
            row's kind does not fit an operand, a dimension is NaN or out of range, a
            non-dimensional row holds a value, the initial guess is degenerate (a line of
            zero length, a radius not above 0), or OCCT refuses a B-spline definition.
    """
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise PysmeshError(
            f"sketch.solve: max_iterations is {max_iterations!r}; it is an int, 1 or more."
        )
    raw = _sketch_solve(
        np.asarray(params),
        np.asarray(entities),
        np.asarray(constraints),
        np.asarray(values),
        None if fixed is None else np.asarray(fixed),
        None if spline_offsets is None else np.asarray(spline_offsets),
        None if spline_indices is None else np.asarray(spline_indices),
        max_iterations,
        float(tolerance),
    )
    return SketchSolution(
        params=np.asarray(raw["params"], dtype=np.float64),
        status=SketchStatus(str(raw["status"])),
        dof=cast(int, raw["dof"]),
        conflicting=np.asarray(raw["conflicting"], dtype=np.int64),
        redundant=np.asarray(raw["redundant"], dtype=np.int64),
        partially_redundant=np.asarray(raw["partially_redundant"], dtype=np.int64),
    )
