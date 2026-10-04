# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""pySMESH session — the export to a mesher.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import EntityId, Handoff


class _HandoffOps(_SessionBase):
    """The export to a mesher."""

    __slots__ = ()

    def export_handoff(self, *, allow_aliases: bool = False) -> Handoff:
        """Export the live shape for a mesher, with the id of every sub-shape of it.

        Cross this boundary **once**, at the meshing handoff, on a shape nobody is editing —
        not per operation. The risk surface is then a fraction of what a per-operation map
        would carry, because there is one export to verify rather than one per edit.

        The returned arrays pair ids to sub-shapes by **position** in the per-kind traversal
        a reader of the bytes reproduces. Never by centroid: a pipe's inner and outer walls
        have the same centroid, so a centroid-keyed map mis-pairs them without saying so.

        By default the map is verified to be a bijection before it is returned. A
        same-domain merge leaves several live ids on one face, a boolean leaves both
        operands' ids on every sub-shape they share, and a split leaves one live id on
        several. All are legitimate states and all make the map ambiguous, so this
        raises naming the ids rather than handing back a map that silently drops some of
        them.

        Args:
            allow_aliases: Return a many-to-one map instead of refusing a shared
                sub-shape. Each ordinal carries its sub-shape's label, the lowest live
                id that denotes it, and :attr:`Handoff.aliases` maps every other live id
                to that label. See :class:`Handoff` for how an id resolves to its
                ordinals.

        Returns:
            The BREP bytes and one id array per entity kind, each in traversal order.

        Raises:
            PysmeshError: If the map is not a bijection and ``allow_aliases`` is False —
                the blamed faces on ``.face_ids``, every blamed id by kind in
                ``.details`` —, if an id shares a sub-shape with a lower id without
                denoting the same sub-shapes (no label map can resolve it), or if the
                BREP write fails.
        """
        raw = self._s.export_handoff(allow_aliases)
        aliases = cast("dict[int, int]", raw["aliases"])
        return Handoff(
            brep=cast("bytes", raw["brep"]),
            solid_id=cast("NDArray[np.int64]", raw["SOLID_id"]),
            face_id=cast("NDArray[np.int64]", raw["FACE_id"]),
            edge_id=cast("NDArray[np.int64]", raw["EDGE_id"]),
            vertex_id=cast("NDArray[np.int64]", raw["VERTEX_id"]),
            aliases=MappingProxyType(
                {EntityId(k): EntityId(v) for k, v in aliases.items()}
            ),
        )
