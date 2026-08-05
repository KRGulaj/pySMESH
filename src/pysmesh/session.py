"""Stateful modelling session with persistent entity identity.

The rest of pySMESH is a stateless BREP-in/BREP-out service: each entry point reads bytes,
runs one OCCT algorithm and writes bytes back. That is the right shape for one-shot work,
and the wrong shape for interactive modelling, because a serialise/deserialise boundary
destroys the shape identity that OCCT's history maps are expressed in. Once identity is
gone, operation histories cannot be composed, an undo costs a full re-parse, and every
tessellation starts from scratch.

A :class:`Session` owns one live shape and an :data:`EntityId` registry that is carried
across every operation by that operation's OCCT history. Two properties follow, and they are
the reason the class exists:

* **Ids are never reused.** A stale reference always resolves to *dead*, never to a
  different entity. A positional ordinal — which is what the stateless API returns — always
  resolves to *something*, and that is the failure mode persistent naming exists to remove.
* **Snapshot and restore are O(1).** A shape is a handle triple and the registry is shared
  immutably, so a snapshot copies handles rather than geometry, at any depth.

Two id spaces now coexist and must not be confused:

===================  =======================================  ==========================
Space                Produced by                              Meaning
===================  =======================================  ==========================
Positional ordinal   :func:`load_brep`, the free functions     1-based rank in a per-kind
                                                               ``TopExp`` traversal;
                                                               changes whenever the
                                                               topology changes
:data:`EntityId`     :class:`Session`                          Session-issued identity;
                                                               monotonic, never reused
===================  =======================================  ==========================

They are both integers, so :data:`EntityId` is a distinct ``NewType``: passing an ordinal
into a session call is caught by ``mypy --strict``, and at run time an id the session never
issued raises :class:`PysmeshError` rather than denoting somebody else's entity.

Thread contract: a :class:`Session` is **not** thread-safe. Use one session per thread.
Sessions are independent and several may coexist in one process. Long operations release the
GIL, so driving one session from two threads at once would be a genuine data race; entering
an operation while another is in flight raises instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final, NewType, cast

import numpy as np
from numpy.typing import NDArray

from ._core import PysmeshError
from ._core import Session as _Session

# A session-issued entity identity. Monotonic, never reused, and deliberately NOT the same
# type as the 1-based positional ordinals the stateless free functions return.
EntityId = NewType("EntityId", int)

# An opaque handle to a retained session state, returned by :meth:`Session.snapshot`.
SnapshotMark = NewType("SnapshotMark", int)

# OCCT's boolean tolerance default. 0.0 means "let OCCT use each shape's own tolerance",
# which is correct for clean geometry; a dirty import needs an explicit fuzzy value.
_DEFAULT_FUZZY: Final[float] = 0.0


class EntityKind(StrEnum):
    """The shape kinds a session tracks.

    This set is not a preference. OCCT's ``BRepTools_History`` records relations for exactly
    these four kinds, so an id on any other kind could not be carried across an operation
    and would silently die at the first boolean.
    """

    SOLID = "SOLID"
    FACE = "FACE"
    EDGE = "EDGE"
    VERTEX = "VERTEX"


class NameRole(IntEnum):
    """How the operation that issued an id produced the entity.

    Attributes:
        CONSTRUCTED: No input correspondence at all — a primitive, or an import.
        GENERATED: The operation's history relates it to one or more input entities.
    """

    CONSTRUCTED = 0
    GENERATED = 1


class ResolutionStatus(StrEnum):
    """Outcome of resolving a :class:`Name` against the current state.

    Attributes:
        RESOLVED: The name denotes exactly one entity occupying one shape.
        AMBIGUOUS: The entity survives, but now denotes several shapes (it was split).
        LOST: The entity is dead. This is a legitimate answer and callers must handle it;
            the session never guesses a replacement.
    """

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    LOST = "lost"


@dataclass(frozen=True)
class Name:
    """A persistent, provenance-based name for an entity.

    A name records *where an entity came from*, never what it looks like. Geometric
    fingerprint matching is deliberately not used as a resolution strategy: it mis-matches
    entities under exactly the edits that make persistent naming necessary, and a wrong
    match is far worse than an honest :attr:`ResolutionStatus.LOST`.

    Names stay valid across :meth:`Session.restore`, because the operation counter is
    session-global and is never rewound. A name minted on a branch that was later abandoned
    resolves to :attr:`ResolutionStatus.LOST`.

    Attributes:
        op_index: The 1-based index of the operation that issued the entity's id.
        role: How that operation produced the entity.
        ordinal: The 0-based rank of the entity among the entities that operation issued
            with this role, in deterministic sub-shape order.
    """

    op_index: int
    role: NameRole
    ordinal: int


@dataclass(frozen=True)
class Resolution:
    """Result of :meth:`Session.resolve`.

    Attributes:
        status: Whether the name resolved, is ambiguous, or is lost.
        ids: The surviving entity ids. Empty when ``status`` is
            :attr:`ResolutionStatus.LOST`.
        shape_count: How many shapes the entity currently denotes. Greater than one exactly
            when ``status`` is :attr:`ResolutionStatus.AMBIGUOUS`.
    """

    status: ResolutionStatus
    ids: tuple[EntityId, ...]
    shape_count: int


@dataclass(frozen=True)
class Origin:
    """Full provenance of one issued id, including its input correspondence.

    Attributes:
        op_index: The operation that issued the id.
        role: How that operation produced the entity.
        ordinal: Rank among that operation's issued entities of this role.
        sources: The input entity ids this one was generated from, ascending. Empty when
            ``role`` is :attr:`NameRole.CONSTRUCTED`.
    """

    op_index: int
    role: NameRole
    ordinal: int
    sources: NDArray[np.int64]

    @property
    def name(self) -> Name:
        """The persistent :class:`Name` this origin describes."""
        return Name(op_index=self.op_index, role=self.role, ordinal=self.ordinal)


@dataclass(frozen=True)
class HistoryDelta:
    """What one operation did to the session's id space.

    The five id arrays are not disjoint: an entity that a boolean both re-built and merged
    another onto appears in ``modified`` and in ``merged``.

    Attributes:
        op_index: The 1-based index of this operation within the session.
        op: The operation's name, e.g. ``"fuse"``.
        created: Ids issued by this operation (int64, ascending).
        deleted: Ids that died in this operation and will never be reused (int64,
            ascending).
        modified: Ids that survived but now denote different shapes than before.
        split: Ids that survived and now denote more than one shape.
        merged: Ids that survived onto a shape that other ids also denote.
    """

    op_index: int
    op: str
    created: NDArray[np.int64]
    deleted: NDArray[np.int64]
    modified: NDArray[np.int64]
    split: NDArray[np.int64]
    merged: NDArray[np.int64]


@dataclass(frozen=True)
class EntityTable:
    """Bulk geometry of every live entity of one kind, as parallel arrays.

    Parallel arrays rather than per-entity objects: the calling convention has to stay
    vectorised for a model with tens of thousands of faces. Row ``i`` of every array
    describes ``ids[i]``.

    Attributes:
        kind: The entity kind this table covers.
        ids: (N,) int64, ascending.
        measure: (N,) float64 — volume for solids, area for faces, length for edges, 0.0 for
            vertices. Summed over every shape a split entity denotes.
        centroid: (N, 3) float64, measure-weighted over a split entity's shapes.
        bbox: (N, 6) float64 — xmin, ymin, zmin, xmax, ymax, zmax, covering every shape.
        shape_count: (N,) int64 — how many shapes each entity denotes; greater than one
            after a split.
    """

    kind: EntityKind
    ids: NDArray[np.int64]
    measure: NDArray[np.float64]
    centroid: NDArray[np.float64]
    bbox: NDArray[np.float64]
    shape_count: NDArray[np.int64]


def _delta(raw: dict[str, object]) -> HistoryDelta:
    """Wrap a raw ``_core`` delta dict in its frozen dataclass."""
    return HistoryDelta(
        op_index=cast("int", raw["op_index"]),
        op=cast("str", raw["op"]),
        created=cast("NDArray[np.int64]", raw["created"]),
        deleted=cast("NDArray[np.int64]", raw["deleted"]),
        modified=cast("NDArray[np.int64]", raw["modified"]),
        split=cast("NDArray[np.int64]", raw["split"]),
        merged=cast("NDArray[np.int64]", raw["merged"]),
    )


def _ids(values: Sequence[EntityId]) -> list[int]:
    """Validate and widen a caller's entity id sequence for the native call."""
    out: list[int] = []
    for v in values:
        if not isinstance(v, (int, np.integer)) or isinstance(v, bool):
            raise PysmeshError(f"Entity ids must be integers (got {v!r}).")
        out.append(int(v))
    return out


class Session:
    """One live shape plus the entity identity carried across every operation on it.

    A session owns a single root shape — a flat compound of bodies — and a monotonically
    increasing operation counter. Operations replace the root; they never mutate a shape an
    earlier snapshot still points at, which is what makes :meth:`snapshot` sound.

    Identity is carried by each operation's OCCT history:

    * an entity modified to exactly one output **keeps its id**;
    * modified to several — the id survives on **all** of them, and the name resolves as
      :attr:`ResolutionStatus.AMBIGUOUS` rather than picking one;
    * several entities merged onto one output — **all** their ids survive on it;
    * an output with no input correspondence gets a **new** id;
    * a removed entity's id is marked **dead** and is never reused.

    Thread contract: **not thread-safe**. One session per thread. Sessions are independent,
    and two may coexist in one process with no cross-talk. Operations release the GIL, so
    entering one while another is in flight on the same session raises
    :class:`PysmeshError` rather than racing on the registry.

    Example:
        >>> s = Session()
        >>> s.add_box(3.0, 7.0, 11.0)                       # doctest: +SKIP
        >>> mark = s.snapshot()                             # O(1)
        >>> s.restore(mark)                                 # O(1)
    """

    __slots__ = ("_s",)

    def __init__(self, *, validate: bool = True) -> None:
        """Create an empty session.

        Args:
            validate: Run ``BRepCheck_Analyzer`` on each operation's result and raise if it
                is invalid, rather than letting a corrupt shape propagate. Only the shape an
                operation actually built is checked, so the cost is per-operation and not
                per-model. Turn it off only for a measured hot path.
        """
        self._s = _Session(validate)

    # ---- construction ---------------------------------------------------------------- #

    def add_brep(self, data: bytes) -> HistoryDelta:
        """Import BREP bytes as one or more new bodies.

        Args:
            data: A shape as BREP bytes (any OCCT ``BRepTools::Write`` output).

        Returns:
            The delta; every entity of the imported shape is newly issued.

        Raises:
            PysmeshError: On a malformed BREP or a null shape.
        """
        return _delta(self._s.add_brep(data))

    def add_box(
        self,
        dx: float,
        dy: float,
        dz: float,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> HistoryDelta:
        """Add an axis-aligned box.

        Args:
            dx: Extent along +x (> 0).
            dy: Extent along +y (> 0).
            dz: Extent along +z (> 0).
            origin: The box's minimum corner.

        Returns:
            The delta; the box's solid, faces, edges and vertices are all newly issued.

        Raises:
            PysmeshError: If any extent is not strictly positive.
        """
        ox, oy, oz = origin
        return _delta(self._s.add_box(dx, dy, dz, ox, oy, oz))

    def add_cylinder(
        self,
        radius: float,
        height: float,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> HistoryDelta:
        """Add a circular cylinder.

        Args:
            radius: Cylinder radius (> 0).
            height: Extent along ``axis`` (> 0).
            origin: Centre of the base circle.
            axis: Direction of the cylinder's axis; must be non-zero.

        Returns:
            The delta; every entity of the cylinder is newly issued.

        Raises:
            PysmeshError: On a non-positive radius or height, or a zero axis.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        return _delta(self._s.add_cylinder(radius, height, ox, oy, oz, ax, ay, az))

    # ---- modelling ------------------------------------------------------------------- #

    def fuse(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
    ) -> HistoryDelta:
        """Boolean-union solids, carrying every input entity's id through the history.

        Args:
            targets: Solid entity ids used as boolean arguments. At least one.
            tools: Solid entity ids used as boolean tools. At least one.
            fuzzy: Additional tolerance for the boolean, in model units. ``0.0`` uses each
                shape's own tolerance, which is right for clean geometry; a dirty import
                that fails at the default may succeed at an explicit value.
            parallel: Run the boolean's internal steps in parallel.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative, or
                if OCCT reports the boolean as failed. No partial result is ever returned.
        """
        return _delta(self._s.fuse(_ids(targets), _ids(tools), fuzzy, parallel))

    def fillet(self, edge_ids: Sequence[EntityId], radius: float) -> HistoryDelta:
        """Fillet edges with a constant radius.

        OCCT derives the owning solid from the edges themselves, so no per-edge face or
        volume co-selection is needed. Every named edge must belong to one body.

        Args:
            edge_ids: Edge entity ids to round. At least one.
            radius: Fillet radius (> 0).

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not an edge, if the edges straddle two
                bodies, or if OCCT cannot build the fillet — in which case the offending
                edge ids are carried on ``.face_ids``.
        """
        return _delta(self._s.fillet(_ids(edge_ids), radius))

    # ---- transforms ------------------------------------------------------------------ #

    def translate(
        self,
        offset: tuple[float, float, float],
        entities: Sequence[EntityId] | None = None,
    ) -> HistoryDelta:
        """Translate the whole model, or the bodies owning the named entities.

        A rigid transform changes only a shape's location, so **every entity id survives**
        and no history is involved. The session asserts that identity rather than assuming
        it: if the transform ever copied the geometry, the operation raises instead of
        silently orphaning every id in the model.

        Args:
            offset: The translation vector.
            entities: Entities whose owning bodies move. ``None`` moves the whole model.

        Returns:
            The delta; ``created`` and ``deleted`` are always empty.

        Raises:
            PysmeshError: If ``entities`` is given but empty, if an id is dead, or if the
                selected bodies share sub-shapes with bodies that stay put.
        """
        dx, dy, dz = offset
        ids = None if entities is None else _ids(entities)
        return _delta(self._s.translate(dx, dy, dz, ids))

    def rotate(
        self,
        origin: tuple[float, float, float],
        axis: tuple[float, float, float],
        angle_rad: float,
        entities: Sequence[EntityId] | None = None,
    ) -> HistoryDelta:
        """Rotate the whole model, or the bodies owning the named entities.

        Carries the same location-only identity property as :meth:`translate`.

        Args:
            origin: A point on the rotation axis.
            axis: Direction of the rotation axis; must be non-zero.
            angle_rad: Rotation angle in radians, right-handed about ``axis``.
            entities: Entities whose owning bodies rotate. ``None`` rotates the whole model.

        Returns:
            The delta; ``created`` and ``deleted`` are always empty.

        Raises:
            PysmeshError: On a zero axis, an empty ``entities``, a dead id, or a selection
                that shares sub-shapes with bodies that stay put.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        ids = None if entities is None else _ids(entities)
        return _delta(self._s.rotate(ox, oy, oz, ax, ay, az, angle_rad, ids))

    # ---- snapshot / restore ---------------------------------------------------------- #

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

        This is the handoff boundary: export once, on a shape that is not being edited,
        rather than per operation.

        Returns:
            The root compound as BREP bytes.
        """
        return self._s.brep()

    # ---- names ------------------------------------------------------------------------ #

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
