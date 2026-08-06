"""pySMESH session — healing, sewing, defeaturing, imprinting and removal.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import (
    CancelPredicate,
    EntityId,
    GlueMode,
    HistoryDelta,
    ProgressCallback,
    ResolutionStatus,
    _DEFAULT_ANGULAR_TOL_DEG,
    _DEFAULT_FUZZY,
    _DEFAULT_HEAL_MAX_TOLERANCE,
    _DEFAULT_HEAL_MIN_TOLERANCE,
    _DEFAULT_HEAL_PRECISION,
    _DEFAULT_LINEAR_TOL,
    _DEFAULT_SEW_TOLERANCE,
    _delta,
    _ids,
)


class _HealOps(_SessionBase):
    """Healing, sewing, defeaturing, imprinting and removal."""

    __slots__ = ()

    def heal(
        self,
        entities: Sequence[EntityId] | None = None,
        *,
        precision: float = _DEFAULT_HEAL_PRECISION,
        min_tolerance: float = _DEFAULT_HEAL_MIN_TOLERANCE,
        max_tolerance: float = _DEFAULT_HEAL_MAX_TOLERANCE,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Repair the whole model, or the bodies owning the named entities.

        Runs OCCT's ``ShapeFix_Shape``: it re-orients faces and shells, closes small gaps,
        fixes wire order and self-intersection, and tightens or loosens sub-shape tolerances
        to make the result consistent.

        **Scope is a real guarantee, not a filter.** Bodies outside the scope are never
        handed to the repair at all, so every entity in them stays byte-identical rather than
        merely unchanged-looking. That is the property a global healing pass cannot offer.

        Unlike every other operation, this one does **not** raise when the result fails
        OCCT's validity check — its input is invalid by assumption, and refusing a shape that
        is less invalid than before would make it useless on exactly the shapes it exists
        for. Read :attr:`HistoryDelta.valid` for the verdict.

        Most of what ``ShapeFix`` does preserves shape identity — a re-orientation changes a
        flag, not the geometry — so a heal typically leaves every entity id in place.

        Args:
            entities: Entities whose owning bodies are repaired. ``None`` repairs the whole
                model.
            precision: The accuracy the repair works to (> 0).
            min_tolerance: Smallest tolerance the repair may assign to a sub-shape (> 0).
            max_tolerance: Largest tolerance it may assign (>= ``min_tolerance``). This is
                the parameter that decides how dirty an import can be taken: raising it lets
                the repair close bigger gaps, at the cost of a looser model.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta, carrying the validity verdict on ``valid``.

        Raises:
            PysmeshError: On a non-positive tolerance, ``max_tolerance`` below
                ``min_tolerance``, an empty ``entities``, a dead id, or a selection that
                shares sub-shapes with bodies left out of the scope.
        """
        ids = None if entities is None else _ids(entities)
        return _delta(
            self._s.heal(
                ids, precision, min_tolerance, max_tolerance, progress, cancel
            )
        )

    def sew(
        self,
        entities: Sequence[EntityId],
        *,
        tolerance: float = _DEFAULT_SEW_TOLERANCE,
        make_solid: bool = False,
        non_manifold: bool = False,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Join the named bodies along boundaries that coincide within ``tolerance``.

        This is the repair for a model whose faces meet geometrically but share no topology —
        the usual state of a surface import — and the path from a set of faces to a solid.

        Args:
            entities: Entities whose owning bodies are sewed. At least one.
            tolerance: Largest gap between two boundaries that still counts as shared (> 0).
                Deliberately tight by default: sewing across a real gap invents topology
                rather than repairing it.
            make_solid: Close the result into a solid. Only a watertight shell bounds a
                volume, so an open one is left as a shell and ``valid`` reports on that.
            non_manifold: Allow more than two faces to meet at one edge. Off by default,
                because a non-manifold result is rarely what a CAD repair wants and is
                accepted by very little downstream.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta, carrying the validity verdict on ``valid``.

        Raises:
            PysmeshError: On a non-positive tolerance, an empty selection, a dead id, or a
                selection that shares sub-shapes with bodies left out of the scope.
        """
        return _delta(
            self._s.sew(
                _ids(entities), tolerance, make_solid, non_manifold, progress, cancel
            )
        )

    def remove_internal_wires(
        self,
        entities: Sequence[EntityId] | None = None,
        *,
        min_area: float,
        remove_faces: bool = True,
    ) -> HistoryDelta:
        """Drop holes smaller than ``min_area`` from the faces that carry them.

        Face-level defeaturing, where :meth:`defeature` is the solid-level operation: this
        removes the *wire* bounding a small hole and heals the face over it, which is what
        clears the fastener holes and vent slots a CFD model does not resolve.

        Args:
            entities: Entities whose owning bodies are processed. ``None`` processes the
                whole model.
            min_area: Holes bounding less than this area are removed (> 0). Required, not
                defaulted: the right threshold is a property of the model's units and of what
                the caller intends to resolve, and no default can be right for both.
            remove_faces: Also drop the faces a removed hole leaves stranded.

        Returns:
            The delta, carrying the validity verdict on ``valid``.

        Raises:
            PysmeshError: On a non-positive ``min_area``, an empty ``entities``, a dead id,
                or a selection that shares sub-shapes with bodies left out of the scope.
        """
        ids = None if entities is None else _ids(entities)
        return _delta(self._s.remove_internal_wires(ids, min_area, remove_faces))

    def unify_same_domain(
        self,
        entities: Sequence[EntityId] | None = None,
        *,
        unify_faces: bool = True,
        unify_edges: bool = True,
        concat_bsplines: bool = False,
        linear_tol: float = _DEFAULT_LINEAR_TOL,
        angular_tol_deg: float = _DEFAULT_ANGULAR_TOL_DEG,
    ) -> HistoryDelta:
        """Merge faces and edges that lie on one underlying surface or curve.

        The clean-up pass after a boolean, which leaves a model split along seams that carry
        no geometric meaning. This is the stateless :func:`unify_same_domain` with the
        session's identity carried across it: merged entities keep **all** their ids, so
        several ids come to denote one shape and every stale reference still resolves.

        Args:
            entities: Entities whose owning bodies are unified. ``None`` unifies the whole
                model.
            unify_faces: Merge coincident-surface faces.
            unify_edges: Merge coincident-curve edges.
            concat_bsplines: Also join B-spline pieces into one curve. Off by default: it
                changes the parametrisation, which invalidates any parameter a caller holds.
            linear_tol: Chord tolerance for deciding coplanarity (> 0).
            angular_tol_deg: Largest connection angle that still merges, in degrees. 0.0 asks
                for the tightest angle OCCT admits.

        Returns:
            The delta, carrying the validity verdict on ``valid``. Merged ids appear in
            ``merged``, not in ``deleted``.

        Raises:
            PysmeshError: If both ``unify_faces`` and ``unify_edges`` are off, on a
                non-positive ``linear_tol``, a negative ``angular_tol_deg``, an empty
                ``entities``, or a dead id.
        """
        ids = None if entities is None else _ids(entities)
        return _delta(
            self._s.unify_same_domain(
                ids,
                unify_faces,
                unify_edges,
                concat_bsplines,
                linear_tol,
                math.radians(angular_tol_deg),
            )
        )

    def defeature(
        self,
        face_ids: Sequence[EntityId],
        *,
        parallel: bool = True,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Remove the features the named faces belong to, closing the surrounding geometry.

        Hand it the faces of a boss, a pocket or a hole and OCCT deletes them and extends
        their neighbours to meet, as though the feature had never been modelled.

        A feature must be named **completely**. OCCT declines an incomplete one — a blind
        hole's cylindrical wall without the flat that caps it — and reports the refusal as a
        *warning*, leaving its success flags set and returning the input unchanged. This
        verifies every named face actually went away and fails loud naming the ones that did
        not, rather than committing a no-op as a success.

        Args:
            face_ids: Faces of the features to remove. At least one, all on one body.
            parallel: Run OCCT's internal steps in parallel.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a face, if the faces straddle two
                bodies, or if OCCT removed nothing for any named face — in which case those
                faces' ids are carried on ``.face_ids``. No partial result is ever returned.
        """
        return _delta(self._s.defeature(_ids(face_ids), parallel, progress, cancel))

    def imprint(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
        glue: GlueMode = GlueMode.OFF,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Split the targets where the tools meet them, leaving the tools in place.

        Imprinting is how an interface becomes real topology on both sides — the way a
        boundary patch, a contact region or a refinement zone is marked on a body without
        changing its shape. Unlike :meth:`split` the tools may be of any dimension, so a
        plane, a face or a wire is a legitimate tool.

        The tools are **not** consumed. Use :meth:`remove` to get rid of one afterwards.

        Args:
            targets: Entities whose owning bodies are imprinted. At least one.
            tools: Entities whose owning bodies do the imprinting. At least one, and disjoint
                from ``targets``.
            fuzzy: Additional tolerance for the operation, in model units.
            parallel: Run OCCT's internal steps in parallel.
            glue: What OCCT may assume about how the operands meet. See :class:`GlueMode`.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta. A target the tools cut clean through is **split**: its id survives on
            every piece and its name resolves as :attr:`ResolutionStatus.AMBIGUOUS`.

        Raises:
            PysmeshError: On an empty operand list, a dead id, a body named on both sides, a
                negative ``fuzzy``, or an operation OCCT reports as failed.
        """
        return _delta(
            self._s.imprint(
                _ids(targets), _ids(tools), fuzzy, parallel, int(glue), progress, cancel
            )
        )

    def remove(self, entities: Sequence[EntityId]) -> HistoryDelta:
        """Drop the bodies owning the named entities from the model.

        Every id inside a removed body dies and is never reused, so a stale reference
        resolves to :attr:`ResolutionStatus.LOST` rather than to whatever now occupies that
        position. An id on a sub-shape that a surviving body also owns stays **alive**,
        because that shape is still in the model.

        Args:
            entities: Entities whose owning bodies are removed. At least one.

        Returns:
            The delta; ``deleted`` holds every id that died and ``valid`` is ``None``,
            because a removal builds no shape to check.

        Raises:
            PysmeshError: On an empty selection, a dead id, or entities that belong to no
                body of the model.
        """
        return _delta(self._s.remove(_ids(entities)))
