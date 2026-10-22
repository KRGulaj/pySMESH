# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-22

"""pySMESH sketch: the 2-D sketch constraint solver over flat arrays.

Public surface: :func:`solve` and its result :class:`SketchSolution`; the kind codes
:class:`SketchEntityKind`, :class:`SketchConstraintKind` and :class:`SketchInternalRole`;
the statuses :class:`SketchStatus`; and the thin builders :class:`SketchBuilder`,
:class:`SketchArrays` and the frozen entity descriptions (:class:`Point`, :class:`Line`,
:class:`Circle`, :class:`Arc`, :class:`Ellipse`, :class:`EllipticArc`, :class:`BSpline`).

A sketch is a set of arrays: every scalar in ``params``, one row per entity in
``entities``, one row per constraint in ``constraints``. A row's index is its id, so the
solver's diagnosis maps straight back to the caller's own lists.
``docs/documentation/concepts/sketch.md`` is the contract of the layout.

The solver is FreeCAD's PlaneGCS, vendored and linked into the native core; it is a
stateless free function and touches no :class:`~pysmesh.Session`. Building CAD from a
solution stays with the session's ``add_line``, ``add_arc``, ``add_bspline``,
``make_wire``, ``make_face``, ``fillet_1d`` and ``offset_wire``.
"""

from __future__ import annotations

from ._builder import (
    Arc,
    BSpline,
    Circle,
    Ellipse,
    EllipticArc,
    Line,
    Point,
    SketchArrays,
    SketchBuilder,
    SketchEntity,
)
from ._kinds import (
    SketchConstraintKind,
    SketchEntityKind,
    SketchInternalRole,
    SketchStatus,
)
from ._solve import (
    CONSTRAINT_COLUMNS,
    ENTITY_COLUMNS,
    SketchSolution,
    solve,
)

__all__ = [
    "CONSTRAINT_COLUMNS",
    "ENTITY_COLUMNS",
    "Arc",
    "BSpline",
    "Circle",
    "Ellipse",
    "EllipticArc",
    "Line",
    "Point",
    "SketchArrays",
    "SketchBuilder",
    "SketchConstraintKind",
    "SketchEntity",
    "SketchEntityKind",
    "SketchInternalRole",
    "SketchSolution",
    "SketchStatus",
    "solve",
]
