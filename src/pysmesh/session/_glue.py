# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""pySMESH session — gluing coincident sub-shapes with GEOM's gluer.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ._base import _SessionBase
from ._types import (
    CancelPredicate,
    EntityId,
    HistoryDelta,
    ProgressCallback,
    _delta,
    _ids,
)


class _GlueOps(_SessionBase):
    """Gluing coincident sub-shapes."""

    __slots__ = ()

    def glue(
        self,
        entities: Sequence[EntityId],
        tolerance: float,
        *,
        kind: Literal["faces", "edges"] = "faces",
        keep_non_solids: bool = True,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Merge the sub-shapes of the named bodies that coincide within a tolerance.

        GEOM's ``MakeGlueFaces`` and ``MakeGlueEdges``: ``GEOMAlgo_GlueDetector`` finds
        the faces, edges and vertices of the bodies that coincide within ``tolerance``,
        and ``GEOMAlgo_Gluer2`` builds the bodies again with one shape for each group of
        coincident shapes. With ``kind="faces"`` every group is glued, faces and the
        edges and vertices of other contacts alike; with ``kind="edges"`` only the groups
        of edges are glued, and the faces stay apart.

        **Identity.** A glued group is a merge: every id of the group survives on its
        one shape, so two ids denote the shared face of two boxes after a glue, as after a
        :meth:`fuse`. The history is the gluer's own (``Modified`` and ``IsDeleted``).

        **Glue and fragment.** Glue intersects nothing. Two faces are glued only when they
        coincide as a whole: two boxes that share a whole face come out sharing it, and
        two boxes whose faces overlap in part stay as they were. :meth:`fragment`
        intersects every operand with every other: it splits faces that overlap in part,
        so the pieces of the overlap are shared, and it also splits solids that cross.
        Glue is the cheap operation for parts built to touch; fragment is the general one.

        Args:
            entities: Entities whose owning bodies are glued. At least one.
            tolerance: The largest distance at which two shapes coincide, in model units
                (> 0). GEOM takes it from the caller, with no default.
            kind: ``"faces"`` (GEOM's ``MakeGlueFaces``) or ``"edges"``
                (``MakeGlueEdges``).
            keep_non_solids: Keep the faces, shells and other non-solid parts of the
                bodies in the result (GEOM's ``doKeepNonSolids``). False keeps the solids
                alone.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing. The
                gluer reports two steps: the detection and the glue.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta. The ids of each glued group are in ``merged``.

        Raises:
            PysmeshError: If no entity is named, an id is dead, or the bodies share
                sub-shapes with bodies left out; if ``tolerance`` is not finite or not
                above zero; if ``kind`` is not ``"faces"`` or ``"edges"``; if nothing of
                that kind coincides within the tolerance (the message names the bodies
                and the tolerance); if the tolerance makes sub-shapes of one shape
                coincide (GEOMAlgo_Gluer2's "sticked shapes"); if the gluer fails (its
                code is in the message); or if the result is invalid. Nothing is committed
                then.
            PysmeshCancelled: If ``cancel`` returned True.
        """
        return _delta(
            self._s.glue(
                _ids(entities), tolerance, kind, keep_non_solids, progress, cancel
            )
        )
