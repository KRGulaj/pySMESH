# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""pySMESH session — snapshots, the id registry's public face, names and introspection.

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
from ._types import (
    EntityId,
    EntityKind,
    EntityTable,
    Name,
    NameRole,
    Origin,
    Resolution,
    ResolutionStatus,
    SnapshotMark,
)


class _StateOps(_SessionBase):
    """Snapshots, the id registry's public face, names and introspection."""

    __slots__ = ()

    def snapshot(self) -> SnapshotMark:
        """Retain the current state and return a mark for it.

        O(1) in the model size: a shape is a handle triple and the id registry is shared
        immutably, so nothing is deep-copied. Memory grows with the number of distinct
        retained states, which is the honest cost — release one with
        :meth:`discard_snapshot` when it is no longer reachable by the undo stack.

        Returns:
            A mark usable with :meth:`restore`, any number of times.
        """
        return SnapshotMark(self._s.snapshot())

    def restore(self, mark: SnapshotMark) -> None:
        """Rewind the root shape and the id registry to a retained state.

        O(1), and re-usable: a mark stays valid until it is discarded.

        The operation counter and the id counter are **not** rewound. If they were, a later
        operation would re-issue an id the abandoned branch already used, and a reference
        held from that branch would resolve to a different entity — the one failure the id
        scheme exists to prevent. Ids issued on an abandoned branch simply report dead.

        Args:
            mark: A mark from :meth:`snapshot`.

        Raises:
            PysmeshError: If the mark is unknown or was discarded.
        """
        self._s.restore(int(mark))

    def discard_snapshot(self, mark: SnapshotMark) -> None:
        """Release a retained state so its shape and registry can be freed.

        Args:
            mark: A mark from :meth:`snapshot`.

        Raises:
            PysmeshError: If the mark is unknown.
        """
        self._s.discard_snapshot(int(mark))

    @property
    def snapshot_count(self) -> int:
        """How many retained states this session currently holds."""
        return self._s.snapshot_count()

    # ---- queries --------------------------------------------------------------------- #

    def entities(self, kind: EntityKind) -> NDArray[np.int64]:
        """Live entity ids of one kind.

        Args:
            kind: The entity kind to list.

        Returns:
            (N,) int64, ascending.
        """
        return self._s.entities(str(kind))

    def entity_table(self, kind: EntityKind) -> EntityTable:
        """Bulk geometry of every live entity of one kind.

        Args:
            kind: The entity kind to tabulate.

        Returns:
            Parallel arrays covering every live entity of that kind.
        """
        raw = self._s.entity_table(str(kind))
        return EntityTable(
            kind=kind,
            ids=cast("NDArray[np.int64]", raw["ids"]),
            measure=cast("NDArray[np.float64]", raw["measure"]),
            centroid=cast("NDArray[np.float64]", raw["centroid"]),
            bbox=cast("NDArray[np.float64]", raw["bbox"]),
            shape_count=cast("NDArray[np.int64]", raw["shape_count"]),
        )

    def entity_kind(self, entity_id: EntityId) -> EntityKind:
        """The kind of a live entity.

        Args:
            entity_id: A live entity id.

        Returns:
            The entity's kind.

        Raises:
            PysmeshError: If the id was never issued, or is dead.
        """
        return EntityKind(self._s.entity_kind(int(entity_id)))

    def is_alive(self, entity_id: EntityId) -> bool:
        """Whether an issued id still denotes something.

        Args:
            entity_id: An id this session issued.

        Returns:
            ``True`` if the entity is live, ``False`` if it is dead.

        Raises:
            PysmeshError: If this session never issued the id — which is how a positional
                ordinal from the stateless API fails loudly instead of silently denoting
                somebody else's entity.
        """
        return self._s.entity_state(int(entity_id)) == "alive"

    def shape_count(self, entity_id: EntityId) -> int:
        """How many shapes a live entity currently denotes.

        Greater than one exactly when the entity has been split.

        Args:
            entity_id: A live entity id.

        Returns:
            The number of shapes.

        Raises:
            PysmeshError: If the id was never issued, or is dead.
        """
        return self._s.shape_count(int(entity_id))

    def brep(self) -> bytes:
        """Serialise the current root shape to BREP bytes.

        The bytes alone, with no entity ids attached. Use :meth:`export_handoff` to cross the
        boundary to a mesher — it returns these bytes *and* the map from them back to this
        session's names, and it refuses to hand over a map that is not a bijection.

        Returns:
            The root compound as BREP bytes.
        """
        return self._s.brep()

    def name_of(self, entity_id: EntityId) -> Name:
        """Mint the persistent name of a live entity.

        Args:
            entity_id: A live entity id.

        Returns:
            The entity's provenance-based name.

        Raises:
            PysmeshError: If the id was never issued, or is dead.
        """
        raw = self._s.name_of(int(entity_id))
        return Name(
            op_index=cast("int", raw["op_index"]),
            role=NameRole(cast("int", raw["role"])),
            ordinal=cast("int", raw["ordinal"]),
        )

    def origin(self, entity_id: EntityId) -> Origin:
        """Full provenance of an issued id, including its input correspondence.

        Unlike :meth:`name_of` this also answers for a dead id, so a caller can explain what
        an entity *was*.

        Args:
            entity_id: An id this session issued, live or dead.

        Returns:
            The id's origin.

        Raises:
            PysmeshError: If this session never issued the id.
        """
        raw = self._s.origin(int(entity_id))
        return Origin(
            op_index=cast("int", raw["op_index"]),
            role=NameRole(cast("int", raw["role"])),
            ordinal=cast("int", raw["ordinal"]),
            sources=cast("NDArray[np.int64]", raw["sources"]),
        )

    def resolve(self, name: Name) -> Resolution:
        """Resolve a persistent name against the current state.

        Args:
            name: A name from :meth:`name_of`.

        Returns:
            The resolution. :attr:`ResolutionStatus.LOST` is a legitimate answer and must be
            handled; the session never guesses a replacement entity.

        Raises:
            PysmeshError: If no entity was ever named by that triple.
        """
        raw = self._s.resolve(name.op_index, int(name.role), name.ordinal)
        ids = cast("NDArray[np.int64]", raw["ids"])
        return Resolution(
            status=ResolutionStatus(cast("str", raw["status"])),
            ids=tuple(EntityId(int(i)) for i in ids),
            shape_count=cast("int", raw["shape_count"]),
        )

    # ---- introspection ---------------------------------------------------------------- #

    @property
    def op_count(self) -> int:
        """Operations this session has run, including those on abandoned branches."""
        return self._s.op_count()

    @property
    def state_op_index(self) -> int:
        """The index of the operation that produced the current state."""
        return self._s.state_op_index()

    @property
    def issued_id_count(self) -> int:
        """How many entity ids this session has issued in total, live and dead."""
        return self._s.issued_id_count()

    @property
    def entity_count(self) -> int:
        """How many entities are live in the current state."""
        return self._s.entity_count()

    def _debug_tear_next_history(self) -> None:
        """Drop the next operation's history, so its input ids die instead of carrying.

        A test hook, and a deliberate one. An identity test that has never been shown to
        fail is a claim, not a check, so the suite tears exactly one operation's history and
        asserts that the ground-truth comparison then fails. Never call this in production
        code: the resulting session is correct but has forgotten where its entities came
        from.
        """
        self._s._debug_tear_next_history()
