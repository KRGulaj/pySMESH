# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""pySMESH session — the export to a mesher.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError
from ..step import write_step_xde
from ._base import _SessionBase
from ._types import EntityId, EntityKind, Handoff


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
                id that denotes it, and :attr:`Handoff.face_ids_of` (and the same for
                the other kinds) lists every live id of each sub-shape. See
                :class:`Handoff` for how an id resolves to its ordinals.

        Returns:
            The BREP bytes and one id array per entity kind, each in traversal order.

        Raises:
            PysmeshError: If the map is not a bijection and ``allow_aliases`` is False —
                the blamed faces on ``.face_ids``, every blamed id by kind in
                ``.details`` —, or if the BREP write fails.
        """
        raw = self._s.export_handoff(allow_aliases)

        def ids_of(kind: str) -> tuple[tuple[EntityId, ...], ...]:
            rows = cast("list[tuple[int, ...]]", raw[f"{kind}_ids_of"])
            return tuple(tuple(EntityId(i) for i in row) for row in rows)

        return Handoff(
            brep=cast("bytes", raw["brep"]),
            solid_id=cast("NDArray[np.int64]", raw["SOLID_id"]),
            face_id=cast("NDArray[np.int64]", raw["FACE_id"]),
            edge_id=cast("NDArray[np.int64]", raw["EDGE_id"]),
            vertex_id=cast("NDArray[np.int64]", raw["VERTEX_id"]),
            solid_ids_of=ids_of("SOLID"),
            face_ids_of=ids_of("FACE"),
            edge_ids_of=ids_of("EDGE"),
            vertex_ids_of=ids_of("VERTEX"),
        )

    def write_step(
        self,
        *,
        unit: str,
        face_names: Mapping[EntityId, str],
        name: str = "",
    ) -> bytes:
        """Export the live shape to STEP, naming its faces by session id.

        The names are keyed by entity id, so they survive the operations that change the
        faces' ordinals. A split id names every piece it denotes. A face that several
        ids denote, as a boolean or a same-domain merge leaves it, takes the name when
        every named id among them gives the same one, and is refused when they differ:
        choosing one silently would mislabel the face.

        Args:
            unit: The unit the coordinates are in, as :func:`write_step_xde` takes it.
            face_names: Live face id to name. Ids not in it leave their faces unnamed,
                unless another id of the same face names it.
            name: Product name for the whole shape (omitted when empty).

        Returns:
            The STEP file content as bytes. :func:`read_step_xde` reads the names
            back on its ``face_labels``.

        Raises:
            PysmeshError: If a key of ``face_names`` is not a live face, if the named
                ids of one face give different names (the ids and names are listed),
                or if :func:`write_step_xde` refuses.
        """
        faces = {int(i) for i in self._s.entities(str(EntityKind.FACE), False)}
        unknown = sorted(int(i) for i in face_names if int(i) not in faces)
        if unknown:
            raise PysmeshError(
                f"Session.write_step: face_names names {unknown}, which are not live "
                "faces."
            )
        by_ordinal: dict[int, str] = {}
        clashes: list[str] = []
        for ordinal, ids in enumerate(self._s.ordinal_ids(str(EntityKind.FACE)), 1):
            named = {
                int(i): face_names[EntityId(int(i))]
                for i in ids
                if EntityId(int(i)) in face_names
            }
            if len(set(named.values())) > 1:
                pairs = ", ".join(f"{i}: {n!r}" for i, n in sorted(named.items()))
                clashes.append(f"face #{ordinal} ({pairs})")
            elif named:
                by_ordinal[ordinal] = next(iter(named.values()))
        if clashes:
            raise PysmeshError(
                "Session.write_step: "
                + str(len(clashes))
                + " face(s) are each one face denoted by ids whose names differ: "
                + "; ".join(clashes)
                + ". Give those ids one name, or name only one of them."
            )
        return write_step_xde(
            self._s.brep(), unit=unit, name=name, face_names=by_ordinal
        )
