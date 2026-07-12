"""B-rep offset operations: make_thick_solid and offset_shape (Tier-2).

Public surface: :class:`ThickSolidParams`, :class:`ThickSolidResult`,
:func:`make_thick_solid`, :class:`OffsetParams`, :class:`OffsetResult`,
:func:`offset_shape`. These wrap the low-level ``_core`` functions (which return raw
BREP bytes + NumPy arrays) in frozen dataclasses and validate parameters up front.

``make_thick_solid`` hollows a solid by removing a set of faces and building inner
offset walls at a given thickness — the primary mechanism for CHT wall-solid creation
and for building structural walls around an extracted fluid volume. Uses OCCT's
``BRepOffsetAPI_MakeThickSolid::MakeThickSolidByJoin`` (TKOffset).

``offset_shape`` uniformly offsets all faces of a shell or solid by a signed distance
(positive = outward enlargement, negative = inward shrinkage). Uses OCCT's
``BRepOffsetAPI_MakeOffsetShape::PerformByJoin`` (TKOffset).

Both return the result as BREP bytes plus a ``face_map`` (int32 array of length
``n_old_faces``): ``face_map[i - 1]`` is the new 1-based face id that original face
``i`` maps to, or ``-1`` if the face was removed. This convention matches
:func:`unify_same_domain` and composes directly onto pySMESH 1-based TopExp ordinals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._core import PysmeshError
from ._core import make_thick_solid as _make_thick_solid
from ._core import offset_shape as _offset_shape

_DEFAULT_TOL: float = 1.0e-7


@dataclass(frozen=True)
class ThickSolidParams:
    """Parameters for :func:`make_thick_solid`.

    Attributes:
        remove_face_ids: Tuple of 1-based face ids (matching :meth:`Shape.faces`) that
            become the openings of the hollowed solid. Must not be empty.
        thickness: Signed offset distance. Positive offsets face normals outward
            (enlarges / shell-thickens). Negative offsets inward (hollows the solid).
            Must be non-zero. Rule of thumb: use a negative value whose absolute value
            is less than the minimum wall thickness of the input solid.
        tol: Geometric tolerance passed to OCCT [model units]. Defaults to 1e-7
            (OCCT's ``Precision::Confusion``). Must be > 0.

    Raises:
        PysmeshError: On any out-of-range parameter.
    """

    remove_face_ids: tuple[int, ...]
    thickness: float
    tol: float = _DEFAULT_TOL

    def __post_init__(self) -> None:
        if not self.remove_face_ids:
            raise PysmeshError("ThickSolidParams.remove_face_ids must not be empty.")
        if self.thickness == 0.0:
            raise PysmeshError(
                "ThickSolidParams.thickness must be non-zero "
                "(positive = outward, negative = inward/hollow)."
            )
        if not self.tol > 0.0:
            raise PysmeshError(
                f"ThickSolidParams.tol must be > 0 (got {self.tol})."
            )


@dataclass(frozen=True)
class ThickSolidResult:
    """Result of :func:`make_thick_solid`.

    Attributes:
        brep: The hollowed solid as BREP bytes (re-loadable via :func:`load_brep`).
        face_map: ``(n_faces_in,)`` int32 — new 1-based face id per original face id.
            ``face_map[i - 1]`` is the new id that original face ``i`` maps to, or
            ``-1`` if the face was in ``remove_face_ids`` (removed / opened).
    """

    brep: bytes
    face_map: NDArray[np.int32]


def make_thick_solid(brep: bytes, params: ThickSolidParams) -> ThickSolidResult:
    """Hollow a SOLID BREP by removing selected faces and building offset inner walls.

    Drives OCCT's ``BRepOffsetAPI_MakeThickSolid::MakeThickSolidByJoin`` with
    ``BRepOffset_Skin`` mode and ``GeomAbs_Intersection`` join type. The GIL is
    released for the main OCCT call.

    Common failure mode: when the absolute value of ``params.thickness`` exceeds the
    smallest feature dimension, offset surfaces self-intersect and OCCT raises.
    ``PysmeshError.face_ids`` will contain the new face ids that failed
    ``BRepCheck_Analyzer`` — use them to identify the problem region.

    Args:
        brep: Input solid as BREP bytes. Must be a ``TopAbs_SOLID``; pass the
            ``brep`` field of a :class:`UnifyResult` or output of :func:`load_brep`
            serialised via ``BRepTools::Write``.
        params: Thickness and removal parameters.

    Returns:
        Hollowed solid BREP bytes plus face_map.

    Raises:
        PysmeshError: On a malformed BREP, non-solid input, invalid face ids,
            OCCT failure (self-intersecting offset), or BRepCheck_Analyzer error.
    """
    raw = _make_thick_solid(
        brep,
        list(params.remove_face_ids),
        params.thickness,
        params.tol,
    )
    return ThickSolidResult(
        brep=cast("bytes", raw["brep"]),
        face_map=cast("NDArray[np.int32]", raw["face_map"]),
    )


@dataclass(frozen=True)
class OffsetParams:
    """Parameters for :func:`offset_shape`.

    Attributes:
        offset: Signed offset distance [model units]. Positive = outward enlargement;
            negative = inward shrinkage. Must be non-zero. Self-intersection occurs if
            the absolute value exceeds the minimum radius of curvature or feature size.
        tol: Geometric tolerance [model units]. Defaults to 1e-7. Must be > 0.

    Raises:
        PysmeshError: On any out-of-range parameter.
    """

    offset: float
    tol: float = _DEFAULT_TOL

    def __post_init__(self) -> None:
        if self.offset == 0.0:
            raise PysmeshError(
                "OffsetParams.offset must be non-zero "
                "(positive = outward, negative = inward)."
            )
        if not self.tol > 0.0:
            raise PysmeshError(
                f"OffsetParams.tol must be > 0 (got {self.tol})."
            )


@dataclass(frozen=True)
class OffsetResult:
    """Result of :func:`offset_shape`.

    Attributes:
        brep: The offset shape as BREP bytes (re-loadable via :func:`load_brep`).
        face_map: ``(n_faces_in,)`` int32 — new 1-based face id per original face id.
            ``face_map[i - 1]`` is the new id that original face ``i`` maps to, or
            ``-1`` if the face was deleted by the offset (rare for Skin mode).
    """

    brep: bytes
    face_map: NDArray[np.int32]


def offset_shape(brep: bytes, params: OffsetParams) -> OffsetResult:
    """Uniformly offset all faces of a BREP by a signed distance.

    Drives OCCT's ``BRepOffsetAPI_MakeOffsetShape::PerformByJoin`` with
    ``BRepOffset_Skin`` mode and ``GeomAbs_Intersection`` join type (adjacent offset
    surfaces are extended to their intersection, preserving sharp corners). The GIL is
    released for the main OCCT call.

    Args:
        brep: Input shape as BREP bytes (shell or solid).
        params: Offset distance and tolerance.

    Returns:
        Offset shape BREP bytes plus face_map.

    Raises:
        PysmeshError: On a malformed BREP, invalid parameters, OCCT failure, or
            BRepCheck_Analyzer error. ``PysmeshError.face_ids`` carries the new face
            ids that failed the validity check.
    """
    raw = _offset_shape(brep, params.offset, params.tol)
    return OffsetResult(
        brep=cast("bytes", raw["brep"]),
        face_map=cast("NDArray[np.int32]", raw["face_map"]),
    )
