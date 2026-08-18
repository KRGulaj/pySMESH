# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""pySMESH session — the value types, enums and defaults the whole surface shares.

Part of the :mod:`pysmesh.session` package. Everything here is data: the id types, the
enums, the frozen dataclasses each operation returns, and the tolerance and quality
defaults with the reasoning for each. No operation lives in this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final, NewType, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError

# A session-issued entity identity. Monotonic, never reused, and deliberately NOT the same
# type as the 1-based positional ordinals the stateless free functions return.
EntityId = NewType("EntityId", int)

# An opaque handle to a retained session state, returned by :meth:`Session.snapshot`.
SnapshotMark = NewType("SnapshotMark", int)

# A point or a vector in model space.
Vec3: TypeAlias = tuple[float, float, float]

# A caller-supplied point list, as an (N, 3) array or any sequence of triples.
Points: TypeAlias = "NDArray[np.float64] | Sequence[Vec3]"

# Called with the fraction of the operation that is done: a float in [0, 1], reported
# strictly increasing, from a helper thread while the operation runs.
ProgressCallback: TypeAlias = "Callable[[float], None]"

# Asked whether to stop. Returning True cancels the operation, which then raises
# PysmeshCancelled and leaves the session exactly as it was.
CancelPredicate: TypeAlias = "Callable[[], bool]"

# ---- Tolerance and parallelism: the defaults, and why they are these ------------------- #
#
# Every operation that takes one of these takes it as a typed keyword with a default that is
# right for CAD preparation, rather than forwarding OCCT's own default unexamined.
#
# ``fuzzy`` — the extra tolerance a boolean uses to decide that two shapes touch. The default
# is 0.0, which is not "no tolerance": it means each shape's own stored tolerance decides,
# which is the correct answer for geometry built here or imported cleanly. A value is worth
# supplying only for an import whose faces do not quite meet, and it must be chosen against
# the gap rather than turned up for luck — a fuzzy value larger than the model's smallest
# real feature will merge things that are genuinely separate. Measured on two boxes with a
# gap between their facing walls: a 1e-4 gap needs fuzzy 1e-4 to fuse into one solid, and a
# 1e-3 gap is still two solids at fuzzy 1e-4.
#
# ``parallel`` — whether OCCT runs the operation's internal steps on several threads. On by
# default because the result does not depend on it: a boolean's topology counts and total
# volume are identical either way, and the mesher's node coordinates are bitwise identical.
# It is a speed setting and nothing else. Turn it off to make a profile readable, not to
# change an answer.
#
# Non-destructive mode is deliberately NOT a parameter. OCCT's booleans default to updating
# their argument shapes in place, which would reach backwards through every retained
# snapshot; every boolean here therefore runs non-destructively, and exposing a switch would
# let a caller turn off the property snapshot and restore depend on.
#
# OCCT's process-global parallel mode (``BOPAlgo_Options::SetParallelMode``) is likewise not
# exposed. It is a static that would change the behaviour of every session in the process,
# including ones on other threads, and the per-operation flag already covers the need.

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

# Healing tolerances, matching OCCT's own ``ShapeFix_Shape`` defaults. ``precision`` is the
# accuracy the repair works to; ``max_tolerance`` is how far it may loosen a sub-shape's
# tolerance to close a gap, and is the parameter that decides how dirty an import it can take.
_DEFAULT_HEAL_PRECISION: Final[float] = 1.0e-7
_DEFAULT_HEAL_MIN_TOLERANCE: Final[float] = 1.0e-7
_DEFAULT_HEAL_MAX_TOLERANCE: Final[float] = 1.0e-3

# Sewing tolerance: the largest gap between two face boundaries that still counts as a shared
# edge. Deliberately tighter than the healing maximum, because sewing across a real gap
# invents topology rather than repairing it.
_DEFAULT_SEW_TOLERANCE: Final[float] = 1.0e-6

# Same-domain merge tolerances, matching the stateless ``unify_same_domain``. An angular
# tolerance of zero asks for the tightest angle OCCT admits.
_DEFAULT_LINEAR_TOL: Final[float] = 1.0e-7
_DEFAULT_ANGULAR_TOL_DEG: Final[float] = 0.0

# Boundary tolerance for point classification: OCCT's ``Precision::Confusion``.
_DEFAULT_CLASSIFY_TOL: Final[float] = 1.0e-7

# Curvature sampling density per parametric direction. 8 x 8 is enough to find the peak of a
# smoothly varying face to a few percent; a face whose curvature varies sharply wants more,
# and the cost is quadratic in this number.
_DEFAULT_CURVATURE_SAMPLES: Final[int] = 8

# Render-mesh quality. Chord deflection in model units, and the largest turn a single mesh
# edge may span. These are display defaults, not analysis ones: they are chosen so a typical
# CAD body looks smooth at screen resolution, and a caller measuring geometry from the
# triangles should ask for far less.
_DEFAULT_DEFLECTION: Final[float] = 0.1
_DEFAULT_MESH_ANGLE_DEG: Final[float] = 20.0


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


class GlueMode(IntEnum):
    """How much OCCT may assume about how two operands meet, in :meth:`Session.imprint`.

    Gluing skips the intersection step for operands the caller declares only *touch*. It is a
    large speed-up on an assembly of coincident-faced parts, and silently wrong on operands
    that genuinely interpenetrate — so it is off unless asked for.

    Attributes:
        OFF: Full intersection. Always correct; the default.
        PARTIAL: The operands coincide over part of their boundaries and nowhere else.
        FULL: The operands coincide over whole faces.
    """

    OFF = 0
    PARTIAL = 1
    FULL = 2


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
        valid: OCCT's ``BRepCheck_Analyzer`` verdict on the shape this operation built.

            ``None`` means no verdict was taken — the operation built nothing to check
            (:meth:`Session.remove`, the transforms), or the session was constructed with
            ``validate=False``.

            Every operation but the healing family *raises* rather than committing an
            invalid result, so for those this is ``True`` whenever it is not ``None``. The
            healing family reports instead, because its input is invalid by assumption and
            refusing to commit a shape that is less invalid than before would make those
            operations useless on exactly the shapes they exist for. There, ``False`` is the
            answer to act on, not an error.
    """

    op_index: int
    op: str
    created: NDArray[np.int64]
    deleted: NDArray[np.int64]
    modified: NDArray[np.int64]
    split: NDArray[np.int64]
    merged: NDArray[np.int64]
    valid: bool | None


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


@dataclass(frozen=True)
class TypeTable:
    """Underlying geometry type of every live entity of one kind.

    Attributes:
        kind: The entity kind this table covers.
        ids: (N,) int64, ascending.
        types: One name per row — a surface type for faces (``"Plane"``, ``"Cylinder"``,
            ``"Cone"``, ``"Sphere"``, ``"Torus"``, ``"Bezier"``, ``"BSpline"``,
            ``"Revolution"``, ``"Extrusion"``, ``"Offset"``, ``"Other"``), a curve type for
            edges (``"Line"``, ``"Circle"``, ``"Ellipse"``, ``"Hyperbola"``, ``"Parabola"``,
            ``"Bezier"``, ``"BSpline"``, ``"Offset"``, ``"Other"``), and the kind's own name
            for solids and vertices. The spelling matches :attr:`FaceInfo.surface_type` from
            the stateless API exactly.
    """

    kind: EntityKind
    ids: NDArray[np.int64]
    types: tuple[str, ...]


@dataclass(frozen=True)
class SurfaceParameterTable:
    """Analytic parameters of named faces' underlying surfaces, in the order named.

    :class:`TypeTable` says a face is a ``"Cylinder"``. This says how wide it is, which is
    what a feature filter reads — "every fillet under 1 mm", "every hole under 3 mm".

    **A parameter the surface type does not define is NaN, never 0.0.** A free-form face has
    no radius at all, and a zero there would read as a very small one to the comparison a
    caller writes. Filter on the type first, or on ``np.isfinite``.

    **The frame is the surface's own, not the face's.** It is taken unflipped, so it agrees
    with the ``(u, v)`` that :meth:`Session.face_parameter_bounds` and
    :meth:`Session.surface_at` speak. On a reversed face ``axis`` is therefore *not* the
    outward normal: :attr:`reversed` says which faces those are, and
    :meth:`Session.surface_at` is the operation that already returns outward normals.

    What each type fills in:

    ==============  ==================================================================
    ``Plane``       ``origin``, ``axis`` (the normal), ``ref_dir``
    ``Cylinder``    the frame, and ``radius1`` = the radius
    ``Cone``        the frame, ``radius1`` = the radius at ``origin``, ``half_angle``
    ``Sphere``      the frame, and ``radius1`` = the radius
    ``Torus``       the frame, ``radius1`` = major radius, ``radius2`` = minor radius
    ``Revolution``  ``origin`` and ``axis`` — the profile decides the rest, and varies
    ``Extrusion``   ``axis`` — the sweep direction **up to sign**; the basis curve has no
                    one origin
    everything      nothing; the whole row is NaN
    else
    ==============  ==================================================================

    Attributes:
        ids: (N,) int64, as given.
        types: One surface type name per row, spelled exactly as :class:`TypeTable` spells
            it.
        origin: (N, 3) float64 — the frame's origin.
        axis: (N, 3) float64 — the frame's main direction: a plane's normal, a cylinder's,
            cone's or torus's axis, a sphere's pole axis, an extrusion's sweep direction.
            An extrusion's is the underlying surface's own direction, which OCCT may store
            negated against the vector the extrusion was built from: it names the line, not
            the sense.
        ref_dir: (N, 3) float64 — the frame's first in-plane direction, the one ``u`` is
            measured from. Needed to reconstruct the parametrisation, not just the shape.
        radius1: (N,) float64 — see the table above.
        radius2: (N,) float64 — a torus's minor radius, and nothing else.
        half_angle: (N,) float64 — a cone's semi-angle in radians. **Signed**: the sign says
            which way along ``axis`` the cone widens, so do not take its magnitude.
        reversed: (N,) bool — whether the face is REVERSED against its surface. Multiply
            ``axis`` by ``-1`` on these rows to get a plane's outward normal.
    """

    ids: NDArray[np.int64]
    types: tuple[str, ...]
    origin: NDArray[np.float64]
    axis: NDArray[np.float64]
    ref_dir: NDArray[np.float64]
    radius1: NDArray[np.float64]
    radius2: NDArray[np.float64]
    half_angle: NDArray[np.float64]
    reversed: NDArray[np.bool_]


@dataclass(frozen=True)
class BoundsTable:
    """Bounding box of every live entity of one kind.

    Deliberately separate from :class:`EntityTable`: a bounding box costs a fraction of a
    mass property, and a caller culling or spatially indexing a model needs only the box.

    Attributes:
        kind: The entity kind this table covers.
        ids: (N,) int64, ascending.
        bbox: (N, 6) float64 — xmin, ymin, zmin, xmax, ymax, zmax, covering every shape a
            split entity denotes.
    """

    kind: EntityKind
    ids: NDArray[np.int64]
    bbox: NDArray[np.float64]


@dataclass(frozen=True)
class MassTable:
    """Measure and centre of mass of named entities, in the order they were named.

    Attributes:
        ids: (N,) int64, as given.
        measure: (N,) float64 — volume for a solid, area for a face, length for an edge,
            0.0 for a vertex. Summed over every shape a split entity denotes.
        centroid: (N, 3) float64, measure-weighted over a split entity's shapes.
    """

    ids: NDArray[np.int64]
    measure: NDArray[np.float64]
    centroid: NDArray[np.float64]


@dataclass(frozen=True)
class AdjacencyPairs:
    """Which entities of one kind touch which entities of another.

    Row ``i`` states that ``ids[i]`` is adjacent to ``related[i]``. An entity appears once
    per neighbour, so the arrays are longer than either entity list.

    Attributes:
        kind: The kind the rows are keyed by.
        other_kind: The kind the rows point at.
        ids: (M,) int64.
        related: (M,) int64.
    """

    kind: EntityKind
    other_kind: EntityKind
    ids: NDArray[np.int64]
    related: NDArray[np.int64]


@dataclass(frozen=True)
class WireTable:
    """The wire loops of named faces, as runs of edge ids.

    :meth:`Session.adjacency` gives a face's edges as one flat set, which cannot answer the
    question a hole test asks: which edges bound the outer boundary, and which bound each
    inner loop. A wire *is* the loop, so this is the query that separates them.

    **A wire has no id.** ``WIRE`` is not an :class:`EntityKind` — ``BRepTools_History``
    fixes the tracked set at solids, faces, edges and vertices — so a loop is named by its
    owning face plus its row here, and its edges by the ids they already have.

    Row ``i`` covers one wire: it belongs to ``face_id[i]``, and its edges are
    ``edge_id[edge_range[i, 0]:edge_range[i, 1]]``. A face contributes one row per loop, so a
    solid box gives 6 rows and a bored box gives 8 — the bore's two end faces have two loops
    each.

    **Each edge is listed once per wire.** A seam edge belongs to its wire twice, once per
    orientation, and an id cannot carry an orientation: listing it twice would read as a
    duplicate rather than as a seam.

    Attributes:
        face_id: (W,) int64 — the face each wire bounds, repeated once per loop, in the order
            the faces were named.
        is_outer: (W,) bool — whether this is the face's outer boundary. Exactly one row per
            face is True; the rest are holes. From ``BRepTools::OuterWire``.
        ordered: (W,) bool — whether ``edge_id`` for this row is a real traversal of the
            loop, each edge joined to the previous one. False means OCCT's wire explorer
            stopped early on a defect in the wire, and the row fell back to the wire's edge
            map: every edge of the loop is still there, in an arbitrary order. A consumer
            that only needs the *set* of edges may ignore this; one walking the loop must not.
        edge_range: (W, 2) int32 — ``[start, end)`` into ``edge_id``.
        edge_id: (E,) int64 — every wire's edges, concatenated.
    """

    face_id: NDArray[np.int64]
    is_outer: NDArray[np.bool_]
    ordered: NDArray[np.bool_]
    edge_range: NDArray[np.int32]
    edge_id: NDArray[np.int64]


@dataclass(frozen=True)
class SurfaceSample:
    """Positions and outward normals of one face at requested parameters.

    Attributes:
        points: (N, 3) float64 — the surface point at each ``(u, v)``.
        normals: (N, 3) float64 — the unit normal, pointing **out of** the body. Zero where
            ``defined`` is False.
        defined: (N,) bool — whether a normal exists at that parameter. It does not at a
            degeneracy: a cone's apex, a sphere's pole.
    """

    points: NDArray[np.float64]
    normals: NDArray[np.float64]
    defined: NDArray[np.bool_]


@dataclass(frozen=True)
class CurvatureTable:
    """Peak absolute curvature of named faces, and where on each face it occurs.

    Attributes:
        ids: (N,) int64, as given.
        k_max: (N,) float64 — the largest ``max(|k1|, |k2|)`` found over the sample grid.
            0.0 where ``samples_used`` is 0.
        uv: (N, 2) float64 — the parameters of the sample that produced ``k_max``.
        xyz: (N, 3) float64 — that sample's position in model space.
        samples_used: (N,) int64 — how many grid points were inside the face's trimming *and*
            had a defined curvature. Below ``samples**2`` for a trimmed face; 0 means the
            face answered nowhere, and ``k_max`` is then not a measurement.
    """

    ids: NDArray[np.int64]
    k_max: NDArray[np.float64]
    uv: NDArray[np.float64]
    xyz: NDArray[np.float64]
    samples_used: NDArray[np.int64]


@dataclass(frozen=True)
class Projection:
    """Closest point on one face's surface to each query point.

    Attributes:
        points: (N, 3) float64 — the closest point.
        uv: (N, 2) float64 — its surface parameters.
        distance: (N,) float64 — the distance from the query point.
    """

    points: NDArray[np.float64]
    uv: NDArray[np.float64]
    distance: NDArray[np.float64]


@dataclass(frozen=True)
class RenderMesh:
    """Triangles, edge polylines and vertex points of the live shape, from one call.

    Three properties are contract rather than implementation, because nothing in the arrays
    states them and a consumer that builds a derived structure — a GPU buffer, a spatial
    index, an out-of-core store — has to know:

    * **The mesh is unwelded across faces.** Each face contributes its own node range. That
      is correct for B-rep shading: a hard edge at a face seam, smooth inside a curved patch.
    * **The two coincident nodes a shared edge produces are bitwise equal.** ``BRepMesh``
      discretises each edge once and both adjacent faces read that one polygon, so the
      positions agree *exactly*, not approximately. A consumer welding by exact position
      therefore gets the seam and nothing else. Measured, not assumed.
    * **Edge and vertex ids share the face ids' namespace.** All three are session
      :data:`EntityId` values, so a picked edge, the faces it bounds and the solid they
      belong to are addressable without a ``(dimension, id)`` pair.

    ``nodes`` holds every face's nodes first, in ``face_id`` order, followed by the points of
    any edge that bounds no face. Those trailing nodes carry a zero normal, because there is
    no surface at them to take one from.

    Attributes:
        nodes: (N, 3) float64 — model-space position of every mesh node.
        normals: (N, 3) float64 — unit normal per node, evaluated on the underlying surface
            at the node's own parameters and flipped to point out of the body. The zero
            vector at a degeneracy (a cone's apex, a sphere's pole) and on a free edge's
            nodes; a caller that needs a direction there must supply its own.
        tris: (M, 3) int32 — triangle connectivity, 0-based into ``nodes``. Wound so the
            right-hand normal agrees with ``normals``, including on reversed faces.
        tri_face_id: (M,) int64 — the face each triangle belongs to.
        edge_lines: (L, 2) int32 — polyline segments as node index pairs, into the same
            ``nodes``. Harvested from the discretisation ``BRepMesh`` has already computed,
            so an edge on a face costs index pairs and nothing else.
        edge_id: (L,) int64 — the edge each segment belongs to.
        vertex_xyz: (P, 3) float64 — the position of every vertex.
        vertex_id: (P,) int64 — the vertex each point belongs to.
        face_id: (F,) int64 — the faces that contributed, in traversal order. A face several
            ids denote after a merge is listed under the lowest of them; a face an id was
            split into appears once per piece.
        face_node_range: (F, 2) int32 — ``[start, end)`` into ``nodes`` per face. Empty for a
            face the mesher could not triangulate, which is listed all the same so that an
            absent face and an unmeshed one stay distinguishable.
        face_tri_range: (F, 2) int32 — ``[start, end)`` into ``tris`` per face.
        retriangulated: (K,) int64 — faces whose triangulation this call rebuilt.
        changed: (J,) int64 — faces whose emitted nodes differ from the previous call's.
            A superset of ``retriangulated``: it also contains faces that were only *moved*,
            which keep their triangulation and still land somewhere else. This is the set a
            consumer holding model-space data must refresh; ``retriangulated`` is the subset
            it cannot refresh with a rigid transform.
    """

    nodes: NDArray[np.float64]
    normals: NDArray[np.float64]
    tris: NDArray[np.int32]
    tri_face_id: NDArray[np.int64]
    edge_lines: NDArray[np.int32]
    edge_id: NDArray[np.int64]
    vertex_xyz: NDArray[np.float64]
    vertex_id: NDArray[np.int64]
    face_id: NDArray[np.int64]
    face_node_range: NDArray[np.int32]
    face_tri_range: NDArray[np.int32]
    retriangulated: NDArray[np.int64]
    changed: NDArray[np.int64]


@dataclass(frozen=True)
class Handoff:
    """The live shape as BREP, plus which entity id each of its sub-shapes is.

    This is the CAD-to-mesher boundary, crossed once on a shape nobody is editing rather than
    per operation. The bytes carry the geometry; the id arrays carry the one thing the bytes
    cannot, which is what each sub-shape is called in this session.

    **The pairing is positional, and that is the point.** Each array holds one id per ordinal
    of the traversal a reader of :attr:`brep` reproduces — ``solid_id[0]`` names the first
    solid a consumer will enumerate, and so on. Matching by centroid instead is the obvious
    shortcut and it is wrong by construction rather than merely imprecise: a pipe's inner and
    outer walls share a centroid exactly, so a centroid-keyed map collides on one of the most
    ordinary features in CAD, and collides *silently*.

    **The map is verified to be a bijection before it is handed over.** Two ordinary session
    states break it — a same-domain merge leaves several live ids on one face, and a split
    leaves one live id on several — and either makes "this id is that tag" ambiguous.
    :meth:`Session.export_handoff` raises rather than returning a map that quietly loses
    some of the caller's names.

    What remains the consumer's half: enumerate the imported shape in the same per-kind
    order, check the counts agree, and pair by position. This library cannot verify the other
    kernel's numbering for it, so that check belongs on the far side of the boundary.

    Attributes:
        brep: The root compound as BREP bytes.
        solid_id: (S,) int64 — the entity id of each solid, in traversal order.
        face_id: (F,) int64 — the entity id of each face, in traversal order.
        edge_id: (E,) int64 — the entity id of each edge, in traversal order.
        vertex_id: (V,) int64 — the entity id of each vertex, in traversal order.
    """

    brep: bytes
    solid_id: NDArray[np.int64]
    face_id: NDArray[np.int64]
    edge_id: NDArray[np.int64]
    vertex_id: NDArray[np.int64]


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
        valid=cast("bool | None", raw["valid"]),
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


def _pairs(
    name: str, values: NDArray[np.float64] | Sequence[tuple[float, float]]
) -> NDArray[np.float64]:
    """Normalise a caller's parameter list to a C-contiguous (N, 2) float64 array."""
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise PysmeshError(
            f"{name} must be an (N, 2) array of parameter pairs (got {arr.shape})."
        )
    return arr


def _surface_sample(raw: dict[str, object]) -> SurfaceSample:
    """Wrap a raw ``_core`` surface-sample dict in its frozen dataclass."""
    return SurfaceSample(
        points=cast("NDArray[np.float64]", raw["points"]),
        normals=cast("NDArray[np.float64]", raw["normals"]),
        defined=cast("NDArray[np.bool_]", raw["defined"]),
    )


def _projection(raw: dict[str, object]) -> Projection:
    """Wrap a raw ``_core`` projection dict in its frozen dataclass."""
    return Projection(
        points=cast("NDArray[np.float64]", raw["points"]),
        uv=cast("NDArray[np.float64]", raw["uv"]),
        distance=cast("NDArray[np.float64]", raw["distance"]),
    )
