# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""pySMESH session — GEOM's healing operators (ShHealOper).

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import EntityId, HistoryDelta, _delta


def _fractions(where: str, values: Sequence[float]) -> list[float]:
    """A caller's fractions as floats; refused when one is not a real number."""
    out: list[float] = []
    for v in values:
        if not isinstance(v, (int, float, np.integer, np.floating)) or isinstance(
            v, bool
        ):
            raise PysmeshError(f"{where}: the position {v!r} is not a number.")
        out.append(float(v))
    return out


class _GeomHealOps(_SessionBase):
    """GEOM's healing operators."""

    __slots__ = ()

    def divide_edge(
        self,
        edge: EntityId,
        positions: Sequence[float],
        *,
        by: Literal["length", "parameter"] = "length",
    ) -> HistoryDelta:
        """Divide an edge at fractions of its length or of its parameter range.

        GEOM's ``DivideEdge``: ``ShHealOper_EdgeDivide`` splits the edge's curve, and the
        curve on each face that holds it, at each position, so every face that uses the
        edge holds the pieces in its place. Each position is a fraction measured from the
        end at the start of the edge's curve, the first parameter that
        :meth:`edge_parameter_bounds` reports.

        * ``by="length"``: the fraction of the arc length. The parameter of each point is
          found by OCCT's ``GCPnts_AbscissaPoint`` at 1e-10 of the length, as
          :meth:`point_at_length` finds it. GEOM's own length mode integrates with a
          fixed Gauss rule, which is exact on lines and circles only.
        * ``by="parameter"``: the fraction of the parameter range, as GEOM's
          ``isByParameter``. On a line and a circle the two modes agree.

        **Identity.** The divided edge's id dies. Each piece gets a new id whose role is
        ``GENERATED`` and whose origin is the divided edge, and each new vertex gets a new
        id generated from it too. The faces that hold the edge keep their ids (they are in
        ``modified``), as do every other entity of the body.

        Args:
            edge: An edge entity id that denotes one edge, not degenerate.
            positions: Strictly increasing fractions, each strictly between 0 and 1. At
                least one.
            by: ``"length"`` or ``"parameter"``.

        Returns:
            The delta: the edge in ``deleted``, the pieces and the new vertices in
            ``created``.

        Raises:
            PysmeshError: If the id is dead, is not an edge or was split; if the edge is
                degenerate; if ``positions`` is empty, holds a value that is not in
                ``(0, 1)``, or is not strictly increasing (the message gives the values);
                if ``by`` is not ``"length"`` or ``"parameter"``; if the divider fails or
                gives a number of pieces other than ``len(positions) + 1``; or if the
                result is invalid. Nothing is committed then.
        """
        values = _fractions("Session.divide_edge", positions)
        return _delta(self._s.divide_edge(int(edge), values, by))
