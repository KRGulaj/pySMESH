# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-20

"""pySMESH session — offsetting a whole body.

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
    """The uniform offset of a whole body."""

    __slots__ = ()

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
