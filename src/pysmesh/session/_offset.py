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
                extent, and it must leave every walled face's own radius standing. Past the
                first the inner walls fold through each other, and OCCT either declines or
                returns the input body unhollowed. Past the second an analytic face's radius
                goes through zero, and OCCT rebuilds that face at the absolute value of the
                negative radius — the same surface mirrored through its own axis. All three
                are refused rather than committed; the radius is checked before OCCT is
                driven, so nothing is built.
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
                on a non-positive ``tol``, if a walled face's own radius does not survive the
                thickness, if OCCT leaves a named face in the result instead of opening it,
                if the offset self-intersects, or if the result is not a hollowed solid at
                all.

                The radius check runs first, before OCCT is driven, and covers every face the
                hollowing walls — that is, every face the caller did not open. ``.face_ids``
                then carries the faces whose radius does not survive. Opening a cylinder at a
                planar cap and asking for a wall thicker than its radius is the case this
                exists for: before 4.1.3 it committed a body indistinguishable from an
                unhollowed one. See :meth:`offset` for what the check says about each of the
                four analytic surfaces, and for what it does not speak for.

                A face the caller opens is removed rather than offset, so its own radius is
                never spent and is not checked. It still decides what happens to the faces
                that *are* offset: a cone's end is bounded by a cap, an offset cap slides
                along the axis and carries that end with it, and a cap that was opened stays
                where it is. The same cone therefore carries a thicker wall when the far cap
                is opened than when the near one is.

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
                value larger than half the body's smallest extent makes the offset faces
                cross, and one that spends more than a face's own radius takes that radius
                through zero. Which sign does that depends on the face, not on the distance:
                an inward offset closes a boss and opens a bore, so a *positive* distance is
                what closes a bore. Past either limit OCCT declines, or returns a body turned
                inside out, or one that grew where the distance said shrink, or one rebuilt
                at the absolute value of a negative radius. All of them are refused rather
                than committed.
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
                a non-positive ``tol``, if a face's own radius does not survive the distance,
                if the offset self-intersects, or if the result is not that body offset.

                The radius check runs first, before OCCT is driven, and covers every face of
                the body, since a uniform offset moves all of them. It speaks only for the
                four surfaces that have a closed-form radius, and says something different
                about each:

                * A **cylinder** and a **sphere** carry one radius over the whole face, and
                  the offset spends it directly: the face survives while ``radius +
                  distance`` stays above ``tol``.
                * A **torus** spends its minor radius the same way, and has a second limit
                  besides. A tube grown as wide as the ring it is swept about passes through
                  itself, so ``minor + distance`` must also stay below the major radius.
                * A **cone** has a different radius at each end, and each end is bounded by
                  a cap that the offset moves as well. The end lands at ``r_end + slide *
                  tan(half_angle) + distance / cos(half_angle)``, where ``slide`` is how far
                  that cap travels along the cone's axis — the distance itself when the cap
                  is offset too, and zero when it is an opening. Both ends are checked and
                  the smaller result decides. An end whose radius is already zero is an apex
                  with no cap to move: there is nothing there for the offset to spend, so
                  that end is passed over.

                A plane has no radius, and a B-spline or a surface of revolution has no one
                radius to test, so those faces stay with the post-conditions below.
                ``.face_ids`` carries the faces whose radius does not survive.

                A self-intersection puts the input faces the broken result faces came from on
                ``.face_ids``, or every face of the body when OCCT declined before producing
                one. A result that is not the body offset puts every face of the body there,
                since OCCT reported success and no result face is broken to trace a blame
                back through. Two things make it the body offset when the body is a solid,
                and both are checked: it is a solid of positive volume — shrinking a sphere
                by its own radius returns one turned inside out — and a negative
                ``distance`` leaves it strictly smaller while a positive one leaves it
                strictly larger. A shell body carries no solid and is not checked. No partial
                result is ever returned.

                However the operation is refused, the body is left byte for byte as it was.
                This is a guarantee, not a side effect: OCCT is handed a copy of the body and
                never the session's own shape, because ``BRepOffset_MakeOffset`` edits the
                shape it is given — raising stored tolerances, adding sub-shapes — before any
                of these checks can run. Until 4.2.0 a refused offset could leave the body
                looking identical, with every id, count and measure intact, and still change
                what the next :meth:`fillet` or :meth:`heal` produced from it.
        """
        return _delta(
            self._s.offset(_ids(entities), distance, tol, progress, cancel)
        )
