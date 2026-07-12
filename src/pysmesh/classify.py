"""Point/solid classification: point_in_solid (Tier-2).

Public surface: :func:`point_in_solid`. Wraps the low-level ``_core.point_in_solid``
(BRepClass3d_SolidClassifier) with tol validation, following pysmesh's bare-array return
convention for a single homogeneous result (like :meth:`Shape.face_distance`).

``point_in_solid`` is the exact inside-test used by internal flow-volume extraction: after
Gmsh caps/sews the surfaces and ``makeSolids`` yields candidate solids, a seed point selects
the enclosing solid. The test runs against the analytic B-rep (not a tessellation), so it is
exact up to ``tol``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._core import PysmeshError
from ._core import point_in_solid as _point_in_solid

_DEFAULT_TOL: float = 1.0e-7


def point_in_solid(
    brep: bytes, points: NDArray[np.float64], tol: float = _DEFAULT_TOL
) -> NDArray[np.bool_]:
    """Exact point-in-solid test against a BREP solid.

    Drives OCCT's ``BRepClass3d_SolidClassifier`` (loaded once, classified per point; the GIL
    is released for the loop). A point counts as inside only when it is *strictly* interior
    (``TopAbs_IN``); a point within ``tol`` of the boundary (``TopAbs_ON``) or outside is
    ``False`` — the correct contract for picking the solid a seed point lies inside.

    Args:
        brep: Input shape as BREP bytes. Must contain at least one ``TopAbs_SOLID`` (build one
            via sew + makeSolids first); an open shell/face has no defined interior and raises.
        points: ``(N, 3)`` float64 world-space query points. Any array-like coercible to that
            shape is accepted.
        tol: Boundary tolerance [model units] passed to ``Perform`` — the half-width of the
            ``ON`` band. Defaults to 1e-7 (OCCT's ``Precision::Confusion``). Must be > 0.

    Returns:
        ``(N,)`` bool array; ``mask[i]`` is True iff ``points[i]`` is strictly inside the solid.

    Raises:
        PysmeshError: On a malformed BREP, a shape with no solid, ``tol <= 0``, or a ``points``
            array whose shape is not ``(N, 3)``.
    """
    if not tol > 0.0:
        raise PysmeshError(f"point_in_solid: tol must be > 0 (got {tol}).")
    return _point_in_solid(brep, points, tol)
