# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

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

**Layout.** :class:`Session` is one class with one instance attribute, and it is assembled
here from per-area bases — the same shape the native session has, where one class declared in
one header is implemented across per-area translation units. The areas match one for one, so
a change on either side has an obvious counterpart on the other:

===================  =====================================================================
``_types``           the id types, enums, frozen dataclasses and documented defaults
``_base``            the native handle, and the only slot an instance carries
``_construct``       primitives, curve and surface construction, sweeps
``_boolean``         the boolean family, fillet and chamfer
``_transform``       the transforms, and copy
``_heal``            healing, sewing, defeaturing, imprinting and removal
``_glue``            gluing coincident sub-shapes with GEOM's gluer
``_geomalgo``        GEOM algorithms copied into pySMESH: the 1-D fillet and the
                     smoothing surface
``_blocks``          GEOM's blocks: quadrangles, hexahedra, block checks, searches
                     and arrays
``_geomheal``        GEOM's healing operators: edge division, contour closing, hole
                     filling, face suppression and the shape-process pipeline
``_locate``          locating entities with GEOM's classifiers: on a region, in
                     place of old sub-shapes, the same as another entity
``_offset``          hollowing a solid, and offsetting a body
``_query``           the geometric query surface over the live shape
``_check``           the checks: boolean arguments, validity, short edges, closed
                     shells and tolerances
``_render``          the render mesh, and the incremental delta over it
``_select``          rule-based selection: the filters, the rule and its stored form
``_handoff``         the export to a mesher, and its id-to-ordinal map
``_state``           snapshots, the id registry's public face, names and introspection
===================  =====================================================================

Nothing in the split is public: ``pysmesh.session.Session`` and every name below are exactly
what they were, and callers import them from here.
"""

from __future__ import annotations

from ._blocks import _BlockOps
from ._boolean import _BooleanOps
from ._check import _CheckOps
from ._construct import _ConstructOps
from ._geomalgo import _GeomAlgoOps
from ._geomheal import ShapeProcessValue, _GeomHealOps
from ._glue import _GlueOps
from ._handoff import _HandoffOps
from ._heal import _HealOps
from ._locate import _LocateOps
from ._offset import _OffsetOps
from ._query import _QueryOps
from ._render import _RenderOps
from ._select import (
    RULE_FORMAT,
    BelongsTo,
    ContinuousFaces,
    EdgeSize,
    ExternalFaces,
    FaceSize,
    FeatureEdges,
    Filter,
    HorizontalFace,
    OnGeometry,
    OnLine,
    OnPlane,
    OnPlaneSide,
    OppositeToEdge,
    RelativeToSolid,
    Rule,
    SizeComparator,
    SolidLocation,
    TopoConnectedFaces,
    VerticalFace,
    VolumeSize,
    _SelectOps,
    rule_from_dict,
    rule_to_dict,
)
from ._state import _StateOps
from ._transform import _TransformOps
from ._types import (
    AdjacencyPairs,
    AxisLine,
    BlocksCheck,
    BlocksErrorKind,
    BlocksFinding,
    BooleanCheckFinding,
    BoundsTable,
    BoxRegion,
    CancelPredicate,
    CurvatureTable,
    CurveParameterTable,
    CurveSample,
    CylinderRegion,
    EdgePoint,
    EntityDistance,
    EntityId,
    EntityKind,
    EntityTable,
    FaceFrame,
    GlueMode,
    Handoff,
    HistoryDelta,
    MassTable,
    Name,
    NameRole,
    Origin,
    PlaneRegion,
    Points,
    ProgressCallback,
    Projection,
    Region,
    RegionState,
    RenderMesh,
    Resolution,
    ResolutionStatus,
    ShapeCheckFinding,
    SnapshotMark,
    SphereRegion,
    SurfaceParameterTable,
    SurfaceSample,
    TangentFrameSample,
    TypeTable,
    Vec3,
    WireTable,
)


class Session(
    _BlockOps,
    _BooleanOps,
    _CheckOps,
    _ConstructOps,
    _GeomAlgoOps,
    _GeomHealOps,
    _GlueOps,
    _HandoffOps,
    _HealOps,
    _LocateOps,
    _OffsetOps,
    _QueryOps,
    _RenderOps,
    _SelectOps,
    _StateOps,
    _TransformOps,
):
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

    Arguments: every float argument must be finite. A NaN or an infinity raises
    :class:`PysmeshError` naming the operation, the argument and the value, before any
    OCCT call, and the session is left exactly as it was.

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

    __slots__ = ()


__all__ = [
    "AdjacencyPairs",
    "AxisLine",
    "BelongsTo",
    "BlocksCheck",
    "BlocksErrorKind",
    "BlocksFinding",
    "BooleanCheckFinding",
    "BoundsTable",
    "BoxRegion",
    "CancelPredicate",
    "ContinuousFaces",
    "CurvatureTable",
    "CurveParameterTable",
    "CurveSample",
    "CylinderRegion",
    "EdgePoint",
    "EdgeSize",
    "EntityDistance",
    "EntityId",
    "EntityKind",
    "EntityTable",
    "ExternalFaces",
    "FaceFrame",
    "FaceSize",
    "FeatureEdges",
    "Filter",
    "GlueMode",
    "Handoff",
    "HistoryDelta",
    "HorizontalFace",
    "MassTable",
    "Name",
    "NameRole",
    "OnGeometry",
    "OnLine",
    "OnPlane",
    "OnPlaneSide",
    "OppositeToEdge",
    "Origin",
    "PlaneRegion",
    "Points",
    "ProgressCallback",
    "Projection",
    "RULE_FORMAT",
    "Region",
    "RegionState",
    "RelativeToSolid",
    "RenderMesh",
    "Resolution",
    "ResolutionStatus",
    "Rule",
    "Session",
    "ShapeCheckFinding",
    "ShapeProcessValue",
    "SizeComparator",
    "SnapshotMark",
    "SolidLocation",
    "SphereRegion",
    "SurfaceParameterTable",
    "SurfaceSample",
    "TangentFrameSample",
    "TopoConnectedFaces",
    "TypeTable",
    "Vec3",
    "VerticalFace",
    "VolumeSize",
    "WireTable",
    "rule_from_dict",
    "rule_to_dict",
]
