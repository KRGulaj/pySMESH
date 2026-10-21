# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""pySMESH session — the boolean family, fillet and chamfer.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import (
    CancelPredicate,
    EntityId,
    HistoryDelta,
    ProgressCallback,
    ResolutionStatus,
    Vec3,
    _DEFAULT_FUZZY,
    _delta,
    _ids,
)


class _BooleanOps(_SessionBase):
    """The boolean family, fillet and chamfer.

    Every boolean but :meth:`section` checks OCCT's answer before it commits it. A
    result that holds no solid where one must exist is refused (see each method), and so
    is a result solid with a free boundary edge, an edge that borders one face only:
    such a solid is not watertight, and the error names each free edge by its end points
    and length. OCCT's warnings on a committed result are on
    :attr:`HistoryDelta.warnings`.
    """

    __slots__ = ()

    def fuse(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Boolean-union solids, carrying every input entity's id through the history.

        Args:
            targets: Solid entity ids used as boolean arguments. At least one.
            tools: Solid entity ids used as boolean tools. At least one.
            fuzzy: Additional tolerance for the boolean, in model units. ``0.0`` uses each
                shape's own tolerance, which is right for clean geometry; a dirty import
                that fails at the default may succeed at an explicit value.
            parallel: Run the boolean's internal steps in parallel. Faster, at several
                times the peak memory: see :meth:`fragment` for the measured cost.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative,
                if OCCT reports the boolean as failed, or if OCCT returns no solid,
                which a fuse of solids never is. No partial result is ever returned.
        """
        return _delta(
            self._s.fuse(_ids(targets), _ids(tools), fuzzy, parallel, progress, cancel)
        )

    def cut(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Subtract the tool solids from the target solids, consuming both groups.

        Args:
            targets: Solid entity ids to cut from. At least one.
            tools: Solid entity ids to cut with. At least one. They are consumed.
            fuzzy: Additional tolerance for the boolean, in model units.
            parallel: Run the boolean's internal steps in parallel. Faster, at several
                times the peak memory: see :meth:`fragment` for the measured cost.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative,
                if OCCT reports the boolean as failed, or if OCCT returns no solid
                although a point inside a target lies outside every tool, farther than
                the operands' tolerance plus ``fuzzy`` from both. No partial result is
                ever returned.
        """
        return _delta(
            self._s.cut(_ids(targets), _ids(tools), fuzzy, parallel, progress, cancel)
        )

    def common(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Intersect the target solids with the tool solids, consuming both groups.

        Both operands' ids survive on the intersection, so the result is denoted by every
        id that contributed to it.

        Args:
            targets: Solid entity ids. At least one.
            tools: Solid entity ids. At least one.
            fuzzy: Additional tolerance for the boolean, in model units.
            parallel: Run the boolean's internal steps in parallel. Faster, at several
                times the peak memory: see :meth:`fragment` for the measured cost.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative,
                if OCCT reports the boolean as failed, or if OCCT returns no solid
                although a point lies inside a target and inside a tool, deeper than the
                operands' tolerance plus ``fuzzy`` in both. An empty common of operands
                that only touch, or do not meet, is accepted.
        """
        return _delta(
            self._s.common(_ids(targets), _ids(tools), fuzzy, parallel, progress, cancel)
        )

    def section(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Add the intersection curves of two solid groups, consuming neither.

        The result of a section is the intersection geometry alone, so both operand groups
        stay in the model with every id intact, and only the section's edges and vertices
        are added. Each is named against the faces it came from.

        Args:
            targets: Solid entity ids. At least one.
            tools: Solid entity ids. At least one.
            fuzzy: Additional tolerance for the boolean, in model units.
            parallel: Run the boolean's internal steps in parallel. Faster, at several
                times the peak memory: see :meth:`fragment` for the measured cost.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta; ``deleted`` and ``modified`` are empty, and ``created`` holds the
            section curves. An empty ``created`` means the operands do not intersect.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative, or
                if OCCT reports the boolean as failed.
        """
        return _delta(
            self._s.section(_ids(targets), _ids(tools), fuzzy, parallel, progress, cancel)
        )

    def split(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Split the target solids by the tool solids, consuming only the targets.

        A split target's id survives on **all** of its pieces, so its name resolves as
        :attr:`ResolutionStatus.AMBIGUOUS`. The tools are left in the model untouched.

        Args:
            targets: Solid entity ids to split. At least one.
            tools: Solid entity ids to split with. At least one. They are not consumed.
            fuzzy: Additional tolerance for the boolean, in model units.
            parallel: Run the boolean's internal steps in parallel. Faster, at several
                times the peak memory: see :meth:`fragment` for the measured cost.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative,
                if OCCT reports the boolean as failed, or if OCCT returns no solid for
                solid targets.
        """
        return _delta(
            self._s.split(_ids(targets), _ids(tools), fuzzy, parallel, progress, cancel)
        )

    def fragment(
        self,
        entities: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """General fuse: split every solid by every other and keep all the pieces.

        This is how a conformal multi-body domain is built — the shared interface exists
        once and both neighbours reference it.

        Args:
            entities: Solid entity ids. At least two.
            fuzzy: Additional tolerance for the boolean, in model units.
            parallel: Run the boolean's internal steps in parallel. The result does not
                depend on it; the peak memory does. Each OCCT worker thread holds its
                own working data: measured on 16 threads, about 5 MB per input B-spline
                face in parallel against about 0.8 MB with ``parallel=False``, 6 to 7
                times more. 512 lofted blocks (3 072 faces) peaked at 15.6 GB in
                parallel and 2.2 GB without. Pass ``parallel=False`` when 5 MB times the
                face count nears the memory you can spare.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: On fewer than two solids, a dead or non-solid id, a negative
                ``fuzzy``, a boolean OCCT reports as failed, or a result with no solid.
        """
        return _delta(
            self._s.fragment(_ids(entities), fuzzy, parallel, progress, cancel)
        )

    def split_by_faces(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Split the target solids by faces, giving each piece of a cut solid a new id.

        The piece rule (the sixth rule of entity ids): a target solid that the faces cut
        into two or more solids loses its id, and each piece gets a new id with the role
        :attr:`NameRole.GENERATED`, whose :meth:`origin` names the target. A target solid
        that stays one solid keeps its id. Faces, edges and vertices keep the rules of
        :meth:`split`: a face cut in two keeps its id on both pieces and resolves as
        :attr:`ResolutionStatus.AMBIGUOUS`, and a new cut face gets a new id.

        The tool faces are not consumed: their bodies stay in the model, unchanged. The
        boolean runs on one thread, so two runs on equal input give equal results. The
        result is checked as :meth:`split` checks its own.

        Args:
            targets: Solid entity ids to split. At least one.
            tools: Face entity ids to split with: one face, or several, which may form a
                shell. At least one. None of them may belong to a target's own body.
            fuzzy: Additional tolerance for the boolean, in model units.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta. Each cut target is in ``deleted``, and its pieces are in
            ``created``.

        Raises:
            PysmeshError: If no target or no tool is named, if an id is dead, if a target
                is not a solid or a tool is not a face, if a tool face belongs to a
                target's body, if ``fuzzy`` is negative, if OCCT reports the boolean as
                failed or returns no solid, or if the faces divide none of the targets.
                The message names the operation and the offending ids. No partial result
                is ever returned.
        """
        return _delta(
            self._s.split_by_faces(_ids(targets), _ids(tools), fuzzy, progress, cancel)
        )

    def split_by_plane(
        self,
        targets: Sequence[EntityId],
        origin: Vec3,
        normal: Vec3,
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Split the target solids by a plane, giving each piece of a cut solid a new id.

        The plane passes through ``origin`` with normal ``normal``. The cutting tool is a
        square face on that plane, centred on the projection of the targets' bounding-box
        centre, with a half-side equal to the box diagonal: it covers the targets'
        cross-section twice over. Piece ids follow the piece rule of
        :meth:`split_by_faces`. The boolean runs on one thread.

        Args:
            targets: Solid entity ids to split. At least one, and the plane must cross
                every one of them.
            origin: A point of the plane.
            normal: The plane normal. Any non-zero length.
            fuzzy: Additional tolerance for the boolean, in model units.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta. Each target is in ``deleted``, and its pieces are in ``created``.

        Raises:
            PysmeshError: If no target is named, if an id is dead or not a solid, if the
                normal has zero length or a coordinate is not finite, if ``fuzzy`` is
                negative, if a target lies entirely on one side of the plane (the message
                gives its range of signed distances from the plane; a solid that only
                touches the plane, within its tolerance plus ``fuzzy``, counts as one
                side), or if OCCT fails or leaves a crossed target whole.
        """
        ox, oy, oz = origin
        nx, ny, nz = normal
        return _delta(
            self._s.split_by_plane(
                _ids(targets), ox, oy, oz, nx, ny, nz, fuzzy, progress, cancel
            )
        )

    def keep_half_space(
        self,
        targets: Sequence[EntityId],
        origin: Vec3,
        normal: Vec3,
        keep: Literal["positive", "negative"],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Keep the material of the target solids on one side of a plane.

        ``"positive"`` keeps the points ``p`` with ``(p - origin) . normal >= 0``,
        ``"negative"`` the points with ``(p - origin) . normal <= 0``; the target material
        on the other side is removed. The operation splits the targets as
        :meth:`split_by_plane` does, then removes every piece on the other side. A target
        whose kept material is one solid keeps its id; one whose kept material is two or
        more solids (a non-convex target) loses its id, and each kept piece gets a new
        :attr:`NameRole.GENERATED` id whose origin names it. Faces wholly on the removed
        side die with their piece.

        Args:
            targets: Solid entity ids to trim. At least one, and the plane must cross
                every one of them.
            origin: A point of the plane.
            normal: The plane normal. Any non-zero length.
            keep: Which side to keep: ``"positive"`` (the side the normal points to) or
                ``"negative"``.
            fuzzy: Additional tolerance for the boolean, in model units.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta. A trimmed target that stays one solid is in ``modified``; one that
            leaves several pieces is in ``deleted``, with its kept pieces in
            ``created``.

        Raises:
            PysmeshError: If ``keep`` is neither ``"positive"`` nor ``"negative"``, if no
                target is named, if an id is dead or not a solid, if the normal has zero
                length or a coordinate is not finite, if ``fuzzy`` is negative, if a
                target lies entirely on the removed side (nothing would be kept) or
                entirely on the kept side (nothing to remove) — the message gives its
                range of signed distances — or if OCCT fails or leaves a crossed target
                whole.
        """
        if keep not in ("positive", "negative"):
            raise PysmeshError(
                "Session.keep_half_space: the side to keep is "
                f"{keep!r}; it is 'positive' or 'negative'."
            )
        ox, oy, oz = origin
        nx, ny, nz = normal
        return _delta(
            self._s.keep_half_space(
                _ids(targets),
                ox,
                oy,
                oz,
                nx,
                ny,
                nz,
                keep == "positive",
                fuzzy,
                progress,
                cancel,
            )
        )

    def fillet(
        self,
        edge_ids: Sequence[EntityId],
        radius: float,
        *,
        radius_end: float | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Round edges with a constant or linearly varying radius.

        OCCT derives the owning solid from the edges themselves, so no per-edge face or
        volume co-selection is needed. Every named edge must belong to one body.

        Args:
            edge_ids: Edge entity ids to round. At least one.
            radius: Fillet radius (> 0); the radius at the start of each edge when
                ``radius_end`` is given.
            radius_end: Radius at the end of each edge (> 0). Given, the radius evolves
                linearly along the edge.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not an edge, if the edges straddle two
                bodies, or if OCCT cannot build the fillet — in which case the edges OCCT
                blamed are carried on ``.face_ids``, falling back to every named edge when
                OCCT blames none. No partial result is ever returned.
        """
        return _delta(
            self._s.fillet(_ids(edge_ids), radius, radius_end, progress, cancel)
        )

    def chamfer(
        self,
        edge_ids: Sequence[EntityId],
        distance: float,
        *,
        distance_end: float | None = None,
        face_id: EntityId | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Bevel edges, symmetrically or with two distances against a reference face.

        Args:
            edge_ids: Edge entity ids to bevel. At least one.
            distance: Bevel distance (> 0); the distance measured on ``face_id`` when the
                two-distance form is used.
            distance_end: Distance on the other adjacent face (> 0). Must be given together
                with ``face_id``: OCCT 8.0's only face-aware chamfer is the two-distance
                form.
            face_id: The face ``distance`` is measured on. Must be given together with
                ``distance_end``.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or has the wrong kind, if only one of
                ``distance_end`` and ``face_id`` is given, if the selection straddles two
                bodies, or if OCCT cannot build the chamfer.
        """
        native_face = None if face_id is None else int(face_id)
        return _delta(
            self._s.chamfer(
                _ids(edge_ids), distance, distance_end, native_face, progress, cancel
            )
        )

    def fillet_2d(
        self, face: EntityId, vertices: Sequence[EntityId], radius: float
    ) -> HistoryDelta:
        """Round corners of a free planar face with circle arcs.

        At each named vertex, OCCT's ``BRepFilletAPI_MakeFillet2d`` replaces the corner
        by an arc of ``radius`` tangent to the two edges that meet there, and trims both
        edges to the arc. The two edges must be lines or circle arcs. The face is a body
        on its own, bounded by one wire.

        The face keeps its id and its outward normal, and each trimmed edge keeps its id.
        Each arc and its two end vertices are new, generated from the corner vertex they
        replace; the corner vertex dies.

        Args:
            face: A free planar face without holes.
            vertices: Vertices of ``face`` to round; at least one, each named once.
            radius: The radius of every arc (> 0).

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If ``vertices`` is empty or names a vertex twice, if ``radius``
                is not finite or not positive, if an id is dead or of the wrong kind, if a
                vertex is not a vertex of ``face``, if ``face`` bounds a solid or belongs
                to a shell (the message names the solid or the body's type), if it is not
                planar or has holes, or if OCCT cannot build a fillet at a vertex (the
                message names the vertex and the reason: the arc does not fit, the edges
                are tangent, or an edge is neither a line nor a circle arc).
        """
        return _delta(self._s.fillet_2d(int(face), _ids(vertices), radius))
