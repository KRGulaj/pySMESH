# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""pySMESH session — the checks over the live shape.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import (
    BooleanCheckFinding,
    CancelPredicate,
    EntityId,
    ProgressCallback,
    ShapeCheckFinding,
    _boolean_check_findings,
    _ids,
    _shape_check_findings,
)

BooleanOperation = Literal["fuse", "cut", "common", "section", "split"]

_BOOLEAN_OPERATIONS: tuple[str, ...] = ("fuse", "cut", "common", "section", "split")


class _CheckOps(_SessionBase):
    """The checks: boolean arguments, validity, short edges, closed shells, tolerances.

    Every one is a query: it issues no id, does not touch the history and leaves the
    session exactly as it was.
    """

    __slots__ = ()

    def check_boolean(
        self,
        operation: BooleanOperation,
        arguments: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> tuple[BooleanCheckFinding, ...]:
        """Check the operands of a boolean before running it.

        OCCT's ``BOPAlgo_ArgumentAnalyzer`` runs on the arguments and on the tools, each
        taken as one compound, with every check it offers switched on: operand types,
        self-intersection, small edges, faces that cannot be rebuilt, tangency, vertex and
        edge merging, continuity and curves on surfaces. It runs on one thread.

        The arguments are one shape to the analyzer, so two arguments that overlap each
        other are reported as a self-intersection of the arguments, as OCCT reports it. A
        ``"split"`` is checked as the general builder's operands are (OCCT's
        ``BOPAlgo_UNKNOWN``): its type rule and its face-rebuild test do not apply.

        Args:
            operation: The boolean the operands are for: ``"fuse"``, ``"cut"``,
                ``"common"``, ``"section"`` or ``"split"``.
            arguments: Entity ids of the arguments (the targets), of any kind. At least
                one.
            tools: Entity ids of the tools, of any kind. At least one.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the check runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the check. It then
                raises :class:`PysmeshCancelled`; the session is unchanged either way.

        Returns:
            One :class:`BooleanCheckFinding` per finding, in OCCT's order. Empty means the
            analyzer found nothing.

        Raises:
            PysmeshError: If ``operation`` is not one of the five, if no argument or no
                tool is named, if an id is dead, or if the analyzer throws.
            PysmeshCancelled: If ``cancel`` returns ``True``.
        """
        if operation not in _BOOLEAN_OPERATIONS:
            raise PysmeshError(
                f"Session.check_boolean: the operation is {operation!r}; it is one of "
                "'fuse', 'cut', 'common', 'section' and 'split'."
            )
        rows = self._s.check_boolean(
            operation, _ids(arguments), _ids(tools), progress, cancel
        )
        return _boolean_check_findings(rows)

    def check_shape(
        self,
        entities: Sequence[EntityId] | None = None,
        *,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> tuple[ShapeCheckFinding, ...]:
        """Check the validity of the bodies owning the named entities, or of every body.

        OCCT's ``BRepCheck_Analyzer`` runs on each body, with its geometric controls, on
        one thread. Every status it gives to a sub-shape other than ``BRepCheck_NoError``
        is reported, merged over the contexts it holds in (an edge's status on each of its
        faces). OCCT gives the analyzer no progress range, so the bodies are checked one
        after the other: the progress advances, and a cancel takes effect, between two
        bodies.

        Args:
            entities: Entities whose owning bodies are checked. ``None`` checks every body
                of the model.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the check runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the check. It then
                raises :class:`PysmeshCancelled`; the session is unchanged either way.

        Returns:
            One :class:`ShapeCheckFinding` per invalid sub-shape, body by body, each body's
            solids first and vertices last. Empty means every checked body is valid.

        Raises:
            PysmeshError: If ``entities`` is given but empty, if an id is dead, or if the
                analyzer throws.
            PysmeshCancelled: If ``cancel`` returns ``True``.
        """
        ids = None if entities is None else _ids(entities)
        return _shape_check_findings(self._s.check_shape(ids, progress, cancel))

    def small_edges(self, threshold: float) -> NDArray[np.int64]:
        """The live edges shorter than a length.

        Each edge's length is taken by the rule of :meth:`mass_properties` at its default
        precision, as :meth:`entity_table` takes it; an id that denotes several edges (a
        split one) has the length of all of them together.

        Args:
            threshold: The length, in model units (> 0). An edge strictly shorter is
                listed.

        Returns:
            (K,) int64, ascending — the ids of the short edges.

        Raises:
            PysmeshError: If ``threshold`` is not finite or not positive, or if an edge
                cannot be measured.
        """
        return self._s.small_edges(threshold)

    def is_closed_shell(self, entities: Sequence[EntityId]) -> NDArray[np.bool_]:
        """Whether the body owning each named entity is bounded by closed shells.

        A body passes when it has at least one shell, when every shell's faces are
        oriented consistently (``BRepCheck_Shell::Orientation``, which checks closure as
        well), and when ``ShapeAnalysis_FreeBounds`` finds no free edge, an edge that
        borders one face only, on the shared topology (no sewing tolerance). A free face,
        a wire, an edge or a vertex has no shell and does not pass.

        Args:
            entities: Entity ids of any kind. At least one.

        Returns:
            (N,) bool — one answer per entity, for its owning body, in the order named.

        Raises:
            PysmeshError: If no entity is named or if an id is dead.
        """
        return self._s.is_closed_shell(_ids(entities))

    def tolerances(self, entities: Sequence[EntityId]) -> NDArray[np.float64]:
        """The stored tolerance of each named face, edge or vertex.

        The value is OCCT's ``BRep_Tool::Tolerance`` of the sub-shape: the radius within
        which OCCT treats its geometry as exact. For an id that denotes several shapes (a
        split one) it is the largest of their tolerances.

        Args:
            entities: Face, edge or vertex entity ids.

        Returns:
            (N,) float64 — one tolerance per id, in model units, in the order named.

        Raises:
            PysmeshError: If an id is dead or is a solid, which stores no tolerance.
        """
        return self._s.tolerances(_ids(entities))
