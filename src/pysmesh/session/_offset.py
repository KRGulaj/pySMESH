# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-20

"""pySMESH session — hollowing a solid and offsetting a body.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._base import _SessionBase
from ._types import (
    CancelPredicate,
    EntityId,
    HistoryDelta,
    ProgressCallback,
    _DEFAULT_OFFSET_TOL,
    _delta,
    _ids,
)


class _OffsetOps(_SessionBase):
    """Hollowing a solid, and offsetting a whole body."""

    __slots__ = ()

    def make_thick_solid(
        self,
        face_ids: Sequence[EntityId],
        thickness: float,
        *,
        tol: float = _DEFAULT_OFFSET_TOL,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Hollow the solid owning the named faces, opening it at those faces.

        The named faces become the openings. Every other face of the body gets an offset
        inner wall at ``thickness``, and the two shells are closed into one solid — which is
        the wall of a hollowed part. This is how a CHT wall solid is built around an
        extracted fluid volume, and how a shell is thickened for structural work.

        The owning solid is derived from the faces, so the caller never co-selects it. All
        the faces must lie on one body: one hollowing builds one shape, and a selection
        straddling two is refused rather than guessed at.

        Args:
            face_ids: Faces that become the openings. At least one, all on one solid.
            thickness: Signed wall thickness, non-zero. Negative hollows inward, so the wall
                lies inside the original boundary and the outer faces stay where they were.
                Positive thickens outward. Same convention as ``ThickSolidParams.thickness``.
                A negative value must be smaller in magnitude than half the body's smallest
                extent, and smaller than its smallest radius of curvature. Past that the
                inner walls fold through each other: OCCT then either declines, or returns
                the input body unhollowed, and both are refused rather than committed.
            tol: Coincidence tolerance for the offset, in model units (> 0).
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta. Every id in ``face_ids`` is in ``deleted``, together with every edge
            and vertex that existed only on those faces. The solid's own id is in
            ``modified`` — it still denotes the body. The new inner walls, the rim left at
            each opening, and their edges and vertices are in ``created``; the rim's origin
            names the face it was opened out of. A surviving outer face keeps its id, and
            appears in ``modified`` only if the offset actually rebuilt it.

        Raises:
            PysmeshError: If an id is dead or is not a face, if the faces straddle two
                bodies, if their body is not a solid, on a zero or non-finite ``thickness``,
                on a non-positive ``tol``, if OCCT leaves a named face in the result instead
                of opening it, if the offset self-intersects, or if the result is not a
                hollowed solid at all.

                The last two are different failures and carry different ids. A
                self-intersection puts the *input* faces that the broken result faces came
                from on ``.face_ids`` — ids the caller already holds, rather than ordinals
                into a shape that was never committed. A result that is not a hollowed solid
                puts ``face_ids`` itself there, because nothing in the result is broken to
                trace a blame back through: OCCT reported success and the shape passed
                ``BRepCheck_Analyzer``, yet it is not a wall. Three things make it one, and
                each is checked. It is a solid of positive volume — a negative one is the
                same wall turned inside out, which is what every face but one opened and a
                positive ``thickness`` returns. Hollowed inward it has less volume than the
                body it was built from — an inward ``thickness`` beyond the body's reach
                returns that body itself, and so does opening every face at any thickness.
                And it carries at least one wall the offset built, rather than only the
                input's own faces and a rim at each opening. The message names the
                thickness, both volumes and both face counts. No partial result is ever
                returned.
        """
        return _delta(
            self._s.make_thick_solid(_ids(face_ids), thickness, tol, progress, cancel)
        )

    def offset(
        self,
        entities: Sequence[EntityId],
        distance: float,
        *,
        tol: float = _DEFAULT_OFFSET_TOL,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Offset every face of the body owning the named entities by a signed distance.

        The shape-preserving grow and shrink: adjacent offset faces are extended to their
        intersection rather than joined by a fillet, so a box stays a box with sharp corners
        and a cylinder stays a cylinder.

        **A solid in gives a solid out**, and a shell gives a shell. That matters because the
        offset body is normally the operand of the next boolean or the next mesh, and neither
        takes a loose shell where a volume was meant.

        Args:
            entities: Any live entities of the body to offset. They must all belong to one
                body, which must be a solid or a shell.
            distance: Signed offset, non-zero. Positive enlarges, negative shrinks. A
                negative value larger than half the body's smallest extent, or larger than
                its smallest radius of curvature, makes the offset faces cross.
            tol: Coincidence tolerance for the offset, in model units (> 0).
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta. Every original face the offset rebuilt is in ``modified`` with its id
            kept, and so is the body's own id when the body is a solid — a shell body carries
            no id of its own, the same as a wire, and is named through its faces. Anything
            the offset genuinely deletes is in ``deleted``, which is rare in ``Skin`` mode,
            and anything new is in ``created``.

        Raises:
            PysmeshError: If an id is dead, if the entities straddle two bodies, if their
                body is neither a solid nor a shell, on a zero or non-finite ``distance``, on
                a non-positive ``tol``, or if the offset self-intersects. In that last case
                ``.face_ids`` carries the input faces the broken result faces came from, or
                every face of the body when OCCT declined before producing one. No partial
                result is ever returned.
        """
        return _delta(
            self._s.offset(_ids(entities), distance, tol, progress, cancel)
        )
