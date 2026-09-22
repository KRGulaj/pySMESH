# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-07-12

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
            Must be non-zero. A negative value has to stay under half the body's
            smallest extent, and has to leave every walled face's own radius standing;
            past either limit :func:`make_thick_solid` refuses rather than returning a
            body that is not the hollowed one.
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

    Two things are checked besides, because OCCT can report success and still hand
    back something that is not a hollowed solid. Both name faces by the same 1-based
    ordinal ``remove_face_ids`` uses.

    Before the kernel runs, every face the hollowing walls — every face not in
    ``remove_face_ids`` — must keep its own radius. Past that radius OCCT rebuilds the
    face at the absolute value of the negative one, which is the same surface mirrored
    through its own axis: a valid solid of positive volume with the right faces, and not
    the hollowed body. Only a cylinder, a cone, a sphere and a torus have a radius to
    check; see :meth:`pysmesh.Session.offset` for what is said about each.

    After it runs, the result must be a hollowed solid: a solid of positive volume, of
    less volume than the input when the thickness is negative, and carrying at least one
    wall the offset built rather than only the input's own faces. Past what the body can
    carry, ``MakeThickSolidByJoin`` collapses the inner shell and returns the input
    solid with the opened faces re-issued: a 2 x 2 x 2 box opened at one face and
    walled by -5.0 comes back measuring 8.0, and ``BRepCheck_Analyzer`` accepts it.

    Args:
        brep: Input solid as BREP bytes. Must be a ``TopAbs_SOLID``; pass the
            ``brep`` field of a :class:`UnifyResult` or output of :func:`load_brep`
            serialised via ``BRepTools::Write``.
        params: Thickness and removal parameters.

    Returns:
        Hollowed solid BREP bytes plus face_map.

    Raises:
        PysmeshError: On a malformed BREP, non-solid input, invalid face ids, a walled
            face whose radius the thickness does not leave standing, OCCT failure
            (self-intersecting offset), a ``BRepCheck_Analyzer`` error, or a result that
            is not a hollowed solid.
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
            Which sign spends a given face's radius depends on the face and not on the
            distance: an inward offset closes a boss and opens a bore, so a *positive*
            distance is what closes a bore.
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

    Two things are checked besides the kernel's own verdict. Before it runs, every
    face of the body must keep its own radius, because past that radius OCCT rebuilds
    the face at the absolute value of the negative one — the same surface mirrored
    through its own axis, which is a valid solid and not the offset of anything. A
    cylinder of radius 1 shrunk by 1.01 used to come back as the r = 0.01 cylinder.
    Only a cylinder, a cone, a sphere and a torus have a radius to check; see
    :meth:`pysmesh.Session.offset` for what is said about each, and note that a plane,
    a B-spline and a surface of revolution have none and are covered only by what
    follows.

    After it runs, a body with a volume must have been offset: the result is a solid of
    positive volume, smaller than the input for a negative distance and larger for a
    positive one. A shell carries no volume and is not checked, because a shell's area
    is not monotone in the distance once the shell is not convex.

    Args:
        brep: Input shape as BREP bytes (shell or solid).
        params: Offset distance and tolerance.

    Returns:
        Offset shape BREP bytes plus face_map.

    Raises:
        PysmeshError: On a malformed BREP, invalid parameters, a face whose radius the
            distance does not leave standing, OCCT failure, a ``BRepCheck_Analyzer``
            error, or a result that is not that body offset. ``PysmeshError.face_ids``
            carries the new face ids that failed the validity check, or the 1-based
            ordinals of the input faces whose radius does not survive.
    """
    raw = _offset_shape(brep, params.offset, params.tol)
    return OffsetResult(
        brep=cast("bytes", raw["brep"]),
        face_map=cast("NDArray[np.int32]", raw["face_map"]),
    )
