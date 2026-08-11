# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""pySMESH session — the export to a mesher.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import Handoff


class _HandoffOps(_SessionBase):
    """The export to a mesher."""

    __slots__ = ()

    def export_handoff(self) -> Handoff:
        """Export the live shape for a mesher, with the id of every sub-shape of it.

        Cross this boundary **once**, at the meshing handoff, on a shape nobody is editing —
        not per operation. The risk surface is then a fraction of what a per-operation map
        would carry, because there is one export to verify rather than one per edit.

        The returned arrays pair ids to sub-shapes by **position** in the per-kind traversal
        a reader of the bytes reproduces. Never by centroid: a pipe's inner and outer walls
        have the same centroid, so a centroid-keyed map mis-pairs them without saying so.

        The map is verified to be a bijection before it is returned. A same-domain merge
        leaves several live ids on one face and a split leaves one live id on several; both
        are legitimate states and both make the map ambiguous, so this raises naming the ids
        rather than handing back a map that silently drops some of them.

        Returns:
            The BREP bytes and one id array per entity kind, each in traversal order.

        Raises:
            PysmeshError: If the id-to-sub-shape map is not a bijection — the offending ids
                are carried on ``.face_ids`` — or if the BREP write fails.
        """
        raw = self._s.export_handoff()
        return Handoff(
            brep=cast("bytes", raw["brep"]),
            solid_id=cast("NDArray[np.int64]", raw["SOLID_id"]),
            face_id=cast("NDArray[np.int64]", raw["FACE_id"]),
            edge_id=cast("NDArray[np.int64]", raw["EDGE_id"]),
            vertex_id=cast("NDArray[np.int64]", raw["VERTEX_id"]),
        )
