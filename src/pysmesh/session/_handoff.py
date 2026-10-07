# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""pySMESH session — the export to a mesher.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError
from ..step import write_step_xde
from ._base import _SessionBase
from ._types import EntityId, EntityKind, Handoff

_Value = TypeVar("_Value")


def _refuse_unknown(argument: str, keys: Iterable[EntityId], faces: set[int]) -> None:
    """Refuse the keys of one write_step argument that are not live faces."""
    unknown = sorted(int(i) for i in keys if int(i) not in faces)
    if unknown:
        raise PysmeshError(
            f"Session.write_step: {argument} names {unknown}, which are not live faces."
        )


def _checked_colors(
    face_colors: Mapping[EntityId, tuple[float, float, float]],
) -> dict[EntityId, tuple[float, float, float]]:
    """The colours as three floats each, refused unless each is finite and in [0, 1].

    write_step_xde refuses a component outside [0, 1] without naming the face, and
    writes a NaN, which reads back as another colour.
    """
    out: dict[EntityId, tuple[float, float, float]] = {}
    bad: list[str] = []
    for face, value in face_colors.items():
        try:
            rgb = tuple(float(c) for c in value)
        except (TypeError, ValueError):
            rgb = ()
        if len(rgb) == 3 and all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in rgb):
            out[face] = (rgb[0], rgb[1], rgb[2])
        else:
            bad.append(f"face {int(face)}: {value!r}")
    if bad:
        raise PysmeshError(
            "Session.write_step: face_colors gives "
            + str(len(bad))
            + " face(s) a colour that is not three finite numbers in [0, 1]: "
            + "; ".join(bad)
            + "."
        )
    return out


def _by_ordinal(
    rows: Sequence[NDArray[np.int64]],
    values: Mapping[EntityId, _Value],
    noun: str,
    verb: str,
) -> dict[int, _Value]:
    """Values keyed by live face id, carried to the 1-based face ordinals of the export.

    ``rows[k]`` lists the live ids of ordinal ``k + 1``. An ordinal takes the value its
    ids give when they agree, and is refused when they differ, listing the ids and the
    values. ``noun`` and ``verb`` word the refusal ("name", "name"; "colour", "colour").
    """
    by_ordinal: dict[int, _Value] = {}
    clashes: list[str] = []
    for ordinal, ids in enumerate(rows, 1):
        given = {
            int(i): values[EntityId(int(i))] for i in ids if EntityId(int(i)) in values
        }
        if len(set(given.values())) > 1:
            pairs = ", ".join(f"{i}: {v!r}" for i, v in sorted(given.items()))
            clashes.append(f"face #{ordinal} ({pairs})")
        elif given:
            by_ordinal[ordinal] = next(iter(given.values()))
    if clashes:
        raise PysmeshError(
            "Session.write_step: "
            + str(len(clashes))
            + f" face(s) are each one face denoted by ids whose {noun}s differ: "
            + "; ".join(clashes)
            + f". Give those ids one {noun}, or {verb} only one of them."
        )
    return by_ordinal


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
        face_colors: Mapping[EntityId, tuple[float, float, float]] | None = None,
    ) -> bytes:
        """Export the live shape to STEP, naming and colouring its faces by session id.

        The names and the colours are keyed by entity id, so they survive the operations
        that change the faces' ordinals. Both follow one rule. A split id names or
        colours every piece it denotes. A face that several ids denote, as a boolean or
        a same-domain merge leaves it, takes the name (the colour) when every id among
        them that gives one gives the same one, and is refused when they differ:
        choosing one silently would mislabel the face.

        Args:
            unit: The unit the coordinates are in, as :func:`write_step_xde` takes it.
            face_names: Live face id to name. Ids not in it leave their faces unnamed,
                unless another id of the same face names it.
            name: Product name for the whole shape (omitted when empty).
            face_colors: Live face id to ``(r, g, b)``, each component a finite number
                in ``[0, 1]``. Ids not in it leave their faces uncoloured, unless
                another id of the same face colours it. ``None`` colours no face. OCCT
                keeps a colour as three 32-bit floats, so 0.9 reads back as
                0.8999999761581421.

        Returns:
            The STEP file content as bytes. :func:`read_step_xde` reads the names
            and the colours back on its ``face_labels``.

        Raises:
            PysmeshError: If a key of ``face_names`` or ``face_colors`` is not a live
                face; if a colour is not three finite components in ``[0, 1]`` (the
                face ids and the colours are listed); if the ids of one face give
                different names or different colours (the ids and the values are
                listed); or if :func:`write_step_xde` refuses.
        """
        colors = face_colors if face_colors is not None else {}
        faces = {int(i) for i in self._s.entities(str(EntityKind.FACE), False)}
        _refuse_unknown("face_names", face_names, faces)
        _refuse_unknown("face_colors", colors, faces)
        checked = _checked_colors(colors)
        rows = self._s.ordinal_ids(str(EntityKind.FACE))
        names_by_ordinal = _by_ordinal(rows, face_names, "name", "name")
        colors_by_ordinal = _by_ordinal(rows, checked, "colour", "colour")
        return write_step_xde(
            self._s.brep(),
            unit=unit,
            name=name,
            face_names=names_by_ordinal,
            face_colors=colors_by_ordinal,
        )
