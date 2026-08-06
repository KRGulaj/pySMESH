"""pySMESH session — the boolean family, fillet and chamfer.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import (
    CancelPredicate,
    EntityId,
    HistoryDelta,
    ProgressCallback,
    ResolutionStatus,
    _DEFAULT_FUZZY,
    _delta,
    _ids,
)


class _BooleanOps(_SessionBase):
    """The boolean family, fillet and chamfer."""

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
            parallel: Run the boolean's internal steps in parallel.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative, or
                if OCCT reports the boolean as failed. No partial result is ever returned.
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
            parallel: Run the boolean's internal steps in parallel.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative, or
                if OCCT reports the boolean as failed. No partial result is ever returned.
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
            parallel: Run the boolean's internal steps in parallel.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative, or
                if OCCT reports the boolean as failed.
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
            parallel: Run the boolean's internal steps in parallel.
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
            parallel: Run the boolean's internal steps in parallel.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative, or
                if OCCT reports the boolean as failed.
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
            parallel: Run the boolean's internal steps in parallel.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: On fewer than two solids, a dead or non-solid id, a negative
                ``fuzzy``, or a boolean OCCT reports as failed.
        """
        return _delta(
            self._s.fragment(_ids(entities), fuzzy, parallel, progress, cancel)
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
