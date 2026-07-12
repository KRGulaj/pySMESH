"""Proximity & topology diagnostics: shape_distance and free_boundary_edges (Tier-2).

Public surface: :class:`ShapeDistanceResult`, :func:`shape_distance`,
:func:`free_boundary_edges`. These wrap the low-level ``_core`` queries (which return a raw
dict / NumPy array) in the same frozen-dataclass + 1-based-id convention as the rest of
pysmesh.

``shape_distance`` returns the exact minimum distance between two BREP shapes together with
the witness points (one on each shape) — the entity-to-entity gap check performed before
meshing to confirm the clearance between two bodies is large enough to resolve. Uses OCCT's
``BRepExtrema_DistShapeShape`` (TKBRep), the shape/shape generalisation of
:meth:`Shape.face_distance`.

``free_boundary_edges`` returns the 1-based ids (matching :meth:`Shape.edges`) of every edge
bordered by exactly one face — the naked edges of an open shell. A watertight solid has none;
a non-empty result localises a leak ("show me the hole"). Uses OCCT's edge->face ancestor map
(the free-boundary criterion of ``ShapeAnalysis_FreeBounds``), returning original edge ids
directly rather than reconstructed wires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._core import free_boundary_edges as _free_boundary_edges
from ._core import shape_distance as _shape_distance


@dataclass(frozen=True)
class ShapeDistanceResult:
    """Result of :func:`shape_distance`.

    Attributes:
        distance: Exact minimum distance between the two shapes [model units]. ``0.0`` when
            the shapes touch or overlap.
        point_a: ``(3,)`` float64 — the witness point on ``brep_a`` realising the minimum.
        point_b: ``(3,)`` float64 — the witness point on ``brep_b`` realising the minimum.
    """

    distance: float
    point_a: NDArray[np.float64]
    point_b: NDArray[np.float64]


def shape_distance(brep_a: bytes, brep_b: bytes) -> ShapeDistanceResult:
    """Exact minimum distance between two BREP shapes, with the witness points.

    Drives OCCT's ``BRepExtrema_DistShapeShape`` (default deflection =
    ``Precision::Confusion``). The GIL is released for the computation. Both shapes may be of
    any topological kind (solid, shell, face, wire, edge, vertex, compound).

    Args:
        brep_a: First shape as BREP bytes (e.g. the ``brep`` field of a result, or a shape
            serialised via ``BRepTools::Write``).
        brep_b: Second shape as BREP bytes.

    Returns:
        A :class:`ShapeDistanceResult` with the minimum distance and the two witness points.

    Raises:
        PysmeshError: On a malformed BREP, a null shape, or an OCCT computation failure.
    """
    raw = _shape_distance(brep_a, brep_b)
    return ShapeDistanceResult(
        distance=float(cast("float", raw["distance"])),
        point_a=cast("NDArray[np.float64]", raw["point_a"]),
        point_b=cast("NDArray[np.float64]", raw["point_b"]),
    )


def free_boundary_edges(brep: bytes) -> NDArray[np.int32]:
    """1-based ids of the naked boundary edges of a BREP shape.

    Returns every edge bordered by exactly one face — the open boundary of a shell. A
    watertight solid returns an empty array; a non-empty result localises a leak. Degenerate
    edges (sphere poles, cone apices) and edges with no face parent (bare wireframe input)
    are excluded. Uses OCCT's edge->face ancestor map (no GUI, no SMESH).

    Args:
        brep: Input shape as BREP bytes.

    Returns:
        ``(k,)`` int32 array of 1-based edge ids, ascending, matching :meth:`Shape.edges`.
        Empty when the shape has no free boundary.

    Raises:
        PysmeshError: On a malformed BREP or a null shape.
    """
    return _free_boundary_edges(brep)
