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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final, NewType, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._core import PysmeshError
from ._core import Session as _Session

# A session-issued entity identity. Monotonic, never reused, and deliberately NOT the same
# type as the 1-based positional ordinals the stateless free functions return.
EntityId = NewType("EntityId", int)

# An opaque handle to a retained session state, returned by :meth:`Session.snapshot`.
SnapshotMark = NewType("SnapshotMark", int)

# A point or a vector in model space.
Vec3: TypeAlias = tuple[float, float, float]

# A caller-supplied point list, as an (N, 3) array or any sequence of triples.
Points: TypeAlias = "NDArray[np.float64] | Sequence[Vec3]"

# OCCT's boolean tolerance default. 0.0 means "let OCCT use each shape's own tolerance",
# which is correct for clean geometry; a dirty import needs an explicit fuzzy value.
_DEFAULT_FUZZY: Final[float] = 0.0

# A full revolution. Every partial primitive sweeps through an angle in (0, 2*pi].
_FULL_TURN: Final[float] = 2.0 * math.pi

# Approximation tolerance for the constructions that fit rather than interpolate — the
# spline through points and the helix. Tight enough that the result is visually exact at
# CAD scale, loose enough that the fit converges at a usable degree.
_DEFAULT_FIT_TOL: Final[float] = 1.0e-4

# B-spline degree band for a fit through points. OCCT raises the degree until the tolerance
# is met; 3 is the lowest degree with curvature continuity, 8 is OCCT's own ceiling.
_SPLINE_DEGREE_MIN: Final[int] = 3
_SPLINE_DEGREE_MAX: Final[int] = 8

# Default degree of a B-spline built from control points, clamped to len(poles) - 1.
_DEFAULT_BSPLINE_DEGREE: Final[int] = 3

_ORIGIN: Final[Vec3] = (0.0, 0.0, 0.0)
_Z_AXIS: Final[Vec3] = (0.0, 0.0, 1.0)


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


def _points(name: str, values: Points) -> NDArray[np.float64]:
    """Normalise a caller's point list to a C-contiguous (N, 3) float64 array."""
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise PysmeshError(f"{name} must be an (N, 3) array of points (got {arr.shape}).")
    return arr


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

    def add_cone(
        self,
        radius1: float,
        radius2: float,
        height: float,
        origin: Vec3 = _ORIGIN,
        axis: Vec3 = _Z_AXIS,
        *,
        angle_rad: float = _FULL_TURN,
    ) -> HistoryDelta:
        """Add a circular cone or truncated cone.

        Args:
            radius1: Radius of the base circle, at ``origin`` (>= 0).
            radius2: Radius of the top circle, at ``height`` along ``axis`` (>= 0). Zero
                gives a sharp apex. At least one radius must be positive.
            height: Extent along ``axis`` (> 0).
            origin: Centre of the base circle.
            axis: Direction of the cone's axis; must be non-zero.
            angle_rad: Sweep about the axis, in ``(0, 2*pi]``. Less than a full turn gives
                a wedge of the cone.

        Returns:
            The delta; every entity of the cone is newly issued.

        Raises:
            PysmeshError: On a negative radius, two zero radii, a non-positive height, a
                zero axis, or a sweep outside ``(0, 2*pi]``.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        return _delta(
            self._s.add_cone(radius1, radius2, height, ox, oy, oz, ax, ay, az, angle_rad)
        )

    def add_sphere(
        self,
        radius: float,
        centre: Vec3 = _ORIGIN,
        axis: Vec3 = _Z_AXIS,
        *,
        angle_rad: float = _FULL_TURN,
    ) -> HistoryDelta:
        """Add a sphere.

        Args:
            radius: Sphere radius (> 0).
            centre: Centre of the sphere.
            axis: Direction of the pole axis; must be non-zero.
            angle_rad: Sweep about the pole axis, in ``(0, 2*pi]``. Less than a full turn
                gives a lune.

        Returns:
            The delta; every entity of the sphere is newly issued.

        Raises:
            PysmeshError: On a non-positive radius, a zero axis, or a sweep outside
                ``(0, 2*pi]``.
        """
        cx, cy, cz = centre
        ax, ay, az = axis
        return _delta(self._s.add_sphere(radius, cx, cy, cz, ax, ay, az, angle_rad))

    def add_torus(
        self,
        radius1: float,
        radius2: float,
        origin: Vec3 = _ORIGIN,
        axis: Vec3 = _Z_AXIS,
        *,
        angle_rad: float = _FULL_TURN,
    ) -> HistoryDelta:
        """Add a torus.

        Args:
            radius1: Ring radius, from the axis to the tube centre (> 0).
            radius2: Tube radius (> 0, and smaller than ``radius1``).
            origin: Centre of the ring.
            axis: Direction of the ring's axis; must be non-zero.
            angle_rad: Sweep about the axis, in ``(0, 2*pi]``.

        Returns:
            The delta; every entity of the torus is newly issued.

        Raises:
            PysmeshError: On a non-positive radius, a tube radius at or above the ring
                radius (which self-intersects), a zero axis, or a sweep outside
                ``(0, 2*pi]``.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        return _delta(
            self._s.add_torus(radius1, radius2, ox, oy, oz, ax, ay, az, angle_rad)
        )

    def add_wedge(
        self,
        dx: float,
        dy: float,
        dz: float,
        ltx: float,
        origin: Vec3 = _ORIGIN,
        axis: Vec3 = _Z_AXIS,
    ) -> HistoryDelta:
        """Add a right angular wedge: a box narrowed along x at its far y face.

        Args:
            dx: Extent along local +x at ``y = 0`` (> 0).
            dy: Extent along local +y (> 0).
            dz: Extent along local +z (> 0).
            ltx: Extent along local +x at ``y = dy`` (>= 0). Zero gives a knife edge;
                ``ltx == dx`` gives a plain box.
            origin: The wedge's local origin.
            axis: Direction of the local +z; must be non-zero.

        Returns:
            The delta; every entity of the wedge is newly issued.

        Raises:
            PysmeshError: On a non-positive extent, a negative ``ltx``, or a zero axis.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        return _delta(self._s.add_wedge(dx, dy, dz, ltx, ox, oy, oz, ax, ay, az))

    # ---- construction geometry -------------------------------------------------------- #
    #
    # The registry tracks solids, faces, edges and vertices only, so a wire carries no id of
    # its own: it is named through the ids of its edges. Every operation that consumes a
    # profile resolves the entities it is given to the single body that owns them.

    def add_line(self, start: Vec3, end: Vec3) -> HistoryDelta:
        """Add a straight edge between two points.

        Args:
            start: First point.
            end: Second point; must not coincide with ``start``.

        Returns:
            The delta; the edge and its two vertices are newly issued.

        Raises:
            PysmeshError: If the two points coincide.
        """
        x1, y1, z1 = start
        x2, y2, z2 = end
        return _delta(self._s.add_line(x1, y1, z1, x2, y2, z2))

    def add_arc(self, start: Vec3, through: Vec3, end: Vec3) -> HistoryDelta:
        """Add a circular arc through three points.

        Args:
            start: Start of the arc.
            through: A point the arc passes through.
            end: End of the arc.

        Returns:
            The delta; the edge and its vertices are newly issued.

        Raises:
            PysmeshError: If no circle passes through the three points — they are collinear
                or two of them coincide.
        """
        x1, y1, z1 = start
        x2, y2, z2 = through
        x3, y3, z3 = end
        return _delta(self._s.add_arc(x1, y1, z1, x2, y2, z2, x3, y3, z3))

    def add_circle(self, centre: Vec3, normal: Vec3, radius: float) -> HistoryDelta:
        """Add a full circular edge.

        Args:
            centre: Centre of the circle.
            normal: Normal of the circle's plane; must be non-zero.
            radius: Circle radius (> 0).

        Returns:
            The delta; the closed edge and its single seam vertex are newly issued.

        Raises:
            PysmeshError: On a non-positive radius or a zero normal.
        """
        cx, cy, cz = centre
        nx, ny, nz = normal
        return _delta(self._s.add_circle(cx, cy, cz, nx, ny, nz, radius))

    def add_polyline(self, points: Points, *, closed: bool = False) -> HistoryDelta:
        """Add a polyline wire through the given points.

        Consecutive segments share one vertex by construction, so a later
        :meth:`make_wire` or :meth:`make_face` over these edges keeps every edge id.

        Args:
            points: (N, 3) points, N >= 2.
            closed: Close the polyline back to the first point.

        Returns:
            The delta; every edge and vertex of the wire is newly issued.

        Raises:
            PysmeshError: On fewer than two points, a malformed array, or two coincident
                consecutive points.
        """
        return _delta(self._s.add_polyline(_points("points", points), closed))

    def add_spline(
        self,
        points: Points,
        *,
        degree_min: int = _SPLINE_DEGREE_MIN,
        degree_max: int = _SPLINE_DEGREE_MAX,
        tol: float = _DEFAULT_FIT_TOL,
    ) -> HistoryDelta:
        """Add a B-spline edge approximating a sequence of points.

        Args:
            points: (N, 3) points the curve passes through to within ``tol``, N >= 2.
            degree_min: Lowest degree OCCT may use (>= 1).
            degree_max: Highest degree OCCT may use (>= ``degree_min``).
            tol: Maximum distance from any input point to the fitted curve (> 0).

        Returns:
            The delta; the edge and its vertices are newly issued.

        Raises:
            PysmeshError: On fewer than two points, an invalid degree band, a non-positive
                tolerance, or a fit that does not converge.
        """
        return _delta(
            self._s.add_spline(_points("points", points), degree_min, degree_max, tol)
        )

    def add_bspline(
        self, poles: Points, *, degree: int = _DEFAULT_BSPLINE_DEGREE
    ) -> HistoryDelta:
        """Add a clamped B-spline edge over the given control points.

        The curve passes through the first and last pole and is pulled towards the others;
        use :meth:`add_spline` to pass through every point instead.

        Args:
            poles: (N, 3) control points, N >= 2.
            degree: Curve degree (>= 1), clamped to ``N - 1``.

        Returns:
            The delta; the edge and its vertices are newly issued.

        Raises:
            PysmeshError: On fewer than two poles or a degree below 1.
        """
        return _delta(self._s.add_bspline(_points("poles", poles), degree))

    def add_helix(
        self,
        centre: Vec3,
        axis: Vec3,
        diameter: float,
        pitch: float,
        turns: float,
        *,
        tol: float = _DEFAULT_FIT_TOL,
    ) -> HistoryDelta:
        """Add a helical wire.

        The helix is approximated by B-spline edges, one per turn, to within ``tol``; OCCT
        has no exact helical curve type in a B-rep edge.

        Args:
            centre: Point on the axis where the helix starts.
            axis: Direction of the helix axis; must be non-zero.
            diameter: Diameter of the helix (> 0).
            pitch: Axial rise per turn (> 0).
            turns: Number of turns (> 0); need not be whole.
            tol: Approximation tolerance (> 0).

        Returns:
            The delta; every edge and vertex of the wire is newly issued.

        Raises:
            PysmeshError: On a non-positive parameter, a zero axis, or an approximation
                that does not converge.
        """
        cx, cy, cz = centre
        ax, ay, az = axis
        return _delta(
            self._s.add_helix(cx, cy, cz, ax, ay, az, diameter, pitch, turns, tol)
        )

    def add_rectangle(
        self, origin: Vec3, normal: Vec3, dx: float, dy: float
    ) -> HistoryDelta:
        """Add a planar rectangular face.

        Args:
            origin: Corner of the rectangle, and the plane's origin.
            normal: Normal of the plane; must be non-zero. The rectangle extends along the
                plane's own two in-plane directions, which OCCT derives from the normal.
            dx: Extent along the plane's first in-plane direction (> 0).
            dy: Extent along the plane's second in-plane direction (> 0).

        Returns:
            The delta; the face, its four edges and its four vertices are newly issued.

        Raises:
            PysmeshError: On a non-positive extent or a zero normal.
        """
        ox, oy, oz = origin
        nx, ny, nz = normal
        return _delta(self._s.add_rectangle(ox, oy, oz, nx, ny, nz, dx, dy))

    def make_wire(self, edge_ids: Sequence[EntityId]) -> HistoryDelta:
        """Join loose edges and wires into one wire, consuming them.

        OCCT rebuilds any edge whose end vertex is only *coincident* with the wire's rather
        than *shared* with it. A rebuilt edge is a new entity, so its old id dies — the
        delta says which. Edges that already share vertices, such as those from
        :meth:`add_polyline` or from :meth:`section`, keep their ids.

        Args:
            edge_ids: Edges to join. Each must belong to a loose-edge or wire body, never to
                a solid, because the operation consumes the bodies it is given.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not an edge, if an edge belongs to a solid
                or a face, or if the edges do not form a connected wire.
        """
        return _delta(self._s.make_wire(_ids(edge_ids)))

    def make_face(self, edge_ids: Sequence[EntityId]) -> HistoryDelta:
        """Build a planar face bounded by the named edges, consuming them.

        A non-planar boundary raises rather than being approximated; :meth:`make_filling`
        is the operation for that case.

        Args:
            edge_ids: Edges bounding the face, forming one closed planar loop.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not an edge, if an edge belongs to a solid
                or a face, or if the edges do not bound a closed planar loop.
        """
        return _delta(self._s.make_face(_ids(edge_ids)))

    def make_filling(self, edge_ids: Sequence[EntityId]) -> HistoryDelta:
        """Fill a possibly non-planar boundary with a surface, consuming the boundary.

        The surface is an approximation, so the resulting face's edges are new geometry and
        the boundary edges' ids die.

        Args:
            edge_ids: Edges bounding the surface.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not an edge, if an edge belongs to a solid
                or a face, or if OCCT cannot fill the boundary.
        """
        return _delta(self._s.make_filling(_ids(edge_ids)))

    # ---- sweeps ----------------------------------------------------------------------- #

    def extrude(self, entities: Sequence[EntityId], vector: Vec3) -> HistoryDelta:
        """Sweep a profile linearly, consuming it.

        A sweep raises the profile's dimension: an edge becomes a face, a wire a shell, a
        face a solid. The profile survives inside the result, so its entity ids carry
        through, and each wall the sweep generates is named against the profile edge it
        came from.

        Args:
            entities: Entities identifying the profile. They must all belong to one body,
                which is the body swept.
            vector: The extrusion vector; must be non-zero.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: On a zero vector, a dead id, entities spanning two bodies, a
                profile body that is a solid or a compound, or a sweep OCCT cannot build.
        """
        vx, vy, vz = vector
        return _delta(self._s.extrude(_ids(entities), vx, vy, vz))

    def revolve(
        self,
        entities: Sequence[EntityId],
        origin: Vec3,
        axis: Vec3,
        angle_rad: float = _FULL_TURN,
    ) -> HistoryDelta:
        """Sweep a profile about an axis, consuming it.

        Args:
            entities: Entities identifying the profile. They must all belong to one body.
            origin: A point on the rotation axis.
            axis: Direction of the rotation axis; must be non-zero.
            angle_rad: Sweep angle in ``(0, 2*pi]``.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: On a zero axis, a sweep outside ``(0, 2*pi]``, a dead id, entities
                spanning two bodies, or a profile that crosses the axis.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        return _delta(self._s.revolve(_ids(entities), ox, oy, oz, ax, ay, az, angle_rad))

    def pipe(
        self, spine: Sequence[EntityId], profile: Sequence[EntityId]
    ) -> HistoryDelta:
        """Sweep a profile along a spine, consuming both.

        Args:
            spine: Entities identifying the spine body, which must be a wire or one edge.
            profile: Entities identifying the profile body.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If either selection spans two bodies, if they name the same body,
                if the spine is not a wire or edge, or if OCCT cannot sweep.
        """
        return _delta(self._s.pipe(_ids(spine), _ids(profile)))

    def pipe_shell(
        self,
        spine: Sequence[EntityId],
        profile: Sequence[EntityId],
        *,
        frenet: bool = False,
        solid: bool = True,
    ) -> HistoryDelta:
        """Sweep a profile along a spine with an explicit frame law, consuming both.

        This is the general sweep. :meth:`pipe` is the simpler one; use this when the frame
        law matters or a closed solid is wanted.

        Args:
            spine: Entities identifying the spine body, which must be a wire or one edge.
            profile: Entities identifying the profile body, which must be a wire or one
                edge.
            frenet: Use the Frenet frame rather than OCCT's corrected Frenet. The corrected
                frame avoids the twist a Frenet frame develops at an inflection, so it is
                the default.
            solid: Cap the swept shell into a solid.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If either selection spans two bodies, if they name the same body,
                if either body is not a wire or edge, if ``solid`` is asked for on a shell
                that does not close, or if OCCT cannot sweep.
        """
        return _delta(self._s.pipe_shell(_ids(spine), _ids(profile), frenet, solid))

    def thru_sections(
        self,
        sections: Sequence[Sequence[EntityId]],
        *,
        solid: bool = True,
        ruled: bool = True,
    ) -> HistoryDelta:
        """Loft through an ordered list of section wires, consuming all of them.

        Args:
            sections: One entity-id group per section, in sweep order. Each group must
                resolve to one wire or single-edge body, and no body may appear twice.
            solid: Cap the ends and return a solid rather than a shell.
            ruled: Join consecutive sections with ruled surfaces. ``False`` fits one smooth
                surface through all of them.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: On fewer than two sections, a repeated body, a section that is
                not a wire or edge, or a loft OCCT cannot build.
        """
        return _delta(
            self._s.thru_sections([_ids(s) for s in sections], solid, ruled)
        )

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

    def cut(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
    ) -> HistoryDelta:
        """Subtract the tool solids from the target solids, consuming both groups.

        Args:
            targets: Solid entity ids to cut from. At least one.
            tools: Solid entity ids to cut with. At least one. They are consumed.
            fuzzy: Additional tolerance for the boolean, in model units.
            parallel: Run the boolean's internal steps in parallel.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative, or
                if OCCT reports the boolean as failed. No partial result is ever returned.
        """
        return _delta(self._s.cut(_ids(targets), _ids(tools), fuzzy, parallel))

    def common(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
    ) -> HistoryDelta:
        """Intersect the target solids with the tool solids, consuming both groups.

        Both operands' ids survive on the intersection, so the result is denoted by every
        id that contributed to it.

        Args:
            targets: Solid entity ids. At least one.
            tools: Solid entity ids. At least one.
            fuzzy: Additional tolerance for the boolean, in model units.
            parallel: Run the boolean's internal steps in parallel.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative, or
                if OCCT reports the boolean as failed.
        """
        return _delta(self._s.common(_ids(targets), _ids(tools), fuzzy, parallel))

    def section(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
    ) -> HistoryDelta:
        """Add the intersection curves of two solid groups, consuming neither.

        The result of a section is the intersection geometry alone, so both operand groups
        stay in the model with every id intact, and only the section's edges and vertices
        are added. Each is named against the faces it came from.

        Args:
            targets: Solid entity ids. At least one.
            tools: Solid entity ids. At least one.
            fuzzy: Additional tolerance for the boolean, in model units.
            parallel: Run the boolean's internal steps in parallel.

        Returns:
            The delta; ``deleted`` and ``modified`` are empty, and ``created`` holds the
            section curves. An empty ``created`` means the operands do not intersect.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative, or
                if OCCT reports the boolean as failed.
        """
        return _delta(self._s.section(_ids(targets), _ids(tools), fuzzy, parallel))

    def split(
        self,
        targets: Sequence[EntityId],
        tools: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
    ) -> HistoryDelta:
        """Split the target solids by the tool solids, consuming only the targets.

        A split target's id survives on **all** of its pieces, so its name resolves as
        :attr:`ResolutionStatus.AMBIGUOUS`. The tools are left in the model untouched.

        Args:
            targets: Solid entity ids to split. At least one.
            tools: Solid entity ids to split with. At least one. They are not consumed.
            fuzzy: Additional tolerance for the boolean, in model units.
            parallel: Run the boolean's internal steps in parallel.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not a solid, if ``fuzzy`` is negative, or
                if OCCT reports the boolean as failed.
        """
        return _delta(self._s.split(_ids(targets), _ids(tools), fuzzy, parallel))

    def fragment(
        self,
        entities: Sequence[EntityId],
        *,
        fuzzy: float = _DEFAULT_FUZZY,
        parallel: bool = True,
    ) -> HistoryDelta:
        """General fuse: split every solid by every other and keep all the pieces.

        This is how a conformal multi-body domain is built — the shared interface exists
        once and both neighbours reference it.

        Args:
            entities: Solid entity ids. At least two.
            fuzzy: Additional tolerance for the boolean, in model units.
            parallel: Run the boolean's internal steps in parallel.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: On fewer than two solids, a dead or non-solid id, a negative
                ``fuzzy``, or a boolean OCCT reports as failed.
        """
        return _delta(self._s.fragment(_ids(entities), fuzzy, parallel))

    def fillet(
        self,
        edge_ids: Sequence[EntityId],
        radius: float,
        *,
        radius_end: float | None = None,
    ) -> HistoryDelta:
        """Round edges with a constant or linearly varying radius.

        OCCT derives the owning solid from the edges themselves, so no per-edge face or
        volume co-selection is needed. Every named edge must belong to one body.

        Args:
            edge_ids: Edge entity ids to round. At least one.
            radius: Fillet radius (> 0); the radius at the start of each edge when
                ``radius_end`` is given.
            radius_end: Radius at the end of each edge (> 0). Given, the radius evolves
                linearly along the edge.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not an edge, if the edges straddle two
                bodies, or if OCCT cannot build the fillet — in which case the edges OCCT
                blamed are carried on ``.face_ids``, falling back to every named edge when
                OCCT blames none. No partial result is ever returned.
        """
        return _delta(self._s.fillet(_ids(edge_ids), radius, radius_end))

    def chamfer(
        self,
        edge_ids: Sequence[EntityId],
        distance: float,
        *,
        distance_end: float | None = None,
        face_id: EntityId | None = None,
    ) -> HistoryDelta:
        """Bevel edges, symmetrically or with two distances against a reference face.

        Args:
            edge_ids: Edge entity ids to bevel. At least one.
            distance: Bevel distance (> 0); the distance measured on ``face_id`` when the
                two-distance form is used.
            distance_end: Distance on the other adjacent face (> 0). Must be given together
                with ``face_id``: OCCT 8.0's only face-aware chamfer is the two-distance
                form.
            face_id: The face ``distance`` is measured on. Must be given together with
                ``distance_end``.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or has the wrong kind, if only one of
                ``distance_end`` and ``face_id`` is given, if the selection straddles two
                bodies, or if OCCT cannot build the chamfer.
        """
        native_face = None if face_id is None else int(face_id)
        return _delta(
            self._s.chamfer(_ids(edge_ids), distance, distance_end, native_face)
        )

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

    def mirror(
        self,
        point: Vec3,
        normal: Vec3,
        entities: Sequence[EntityId] | None = None,
    ) -> HistoryDelta:
        """Reflect the whole model, or the bodies owning the named entities, in a plane.

        A plane reflection has determinant -1, so unlike :meth:`translate` and
        :meth:`rotate` it is **not** a location-only change: OCCT rebuilds the geometry.
        Every entity id still survives, one-to-one, carried by the transform's own history
        rather than by shape identity.

        Args:
            point: A point on the mirror plane.
            normal: Normal of the mirror plane; must be non-zero.
            entities: Entities whose owning bodies are mirrored. ``None`` mirrors the whole
                model.

        Returns:
            The delta; every surviving id appears in ``modified``, and ``created`` and
            ``deleted`` are empty.

        Raises:
            PysmeshError: On a zero normal, an empty ``entities``, a dead id, or a
                selection that shares sub-shapes with bodies that stay put.
        """
        px, py, pz = point
        nx, ny, nz = normal
        ids = None if entities is None else _ids(entities)
        return _delta(self._s.mirror(px, py, pz, nx, ny, nz, ids))

    def scale(
        self,
        factors: float | Vec3,
        centre: Vec3 = _ORIGIN,
        entities: Sequence[EntityId] | None = None,
    ) -> HistoryDelta:
        """Scale the whole model, or the bodies owning the named entities.

        A uniform factor keeps analytic surfaces analytic — a scaled cylinder is still a
        cylinder. An anisotropic one cannot, so OCCT re-approximates every non-planar
        surface as a B-spline. Either way the geometry is rebuilt and every entity id
        survives one-to-one through the transform's history.

        Args:
            factors: One factor for all three axes, or one per axis. All must be > 0.
            centre: The fixed point of the scaling.
            entities: Entities whose owning bodies are scaled. ``None`` scales the whole
                model.

        Returns:
            The delta; every surviving id appears in ``modified``, and ``created`` and
            ``deleted`` are empty.

        Raises:
            PysmeshError: On a non-positive factor, a uniform factor of exactly 1 (a no-op
                that would still consume an operation index), an empty ``entities``, a dead
                id, or a selection that shares sub-shapes with bodies that stay put.
        """
        if isinstance(factors, (int, float)):
            sx = sy = sz = float(factors)
        else:
            sx, sy, sz = factors
        cx, cy, cz = centre
        ids = None if entities is None else _ids(entities)
        return _delta(self._s.scale(sx, sy, sz, cx, cy, cz, ids))

    def copy(self, entities: Sequence[EntityId]) -> HistoryDelta:
        """Duplicate the bodies owning the named entities.

        The originals keep every id; every entity of every duplicate is a **new** identity.
        A copy is deliberately not related to its original by history: relating them would
        move the original's id onto the duplicate.

        Args:
            entities: Entities whose owning bodies are duplicated. At least one.

        Returns:
            The delta; ``created`` holds the duplicates' ids and everything else is empty.

        Raises:
            PysmeshError: On an empty selection, a dead id, or an entity that belongs to no
                body.
        """
        return _delta(self._s.copy(_ids(entities)))

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
