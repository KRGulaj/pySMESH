# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""pySMESH session — rule-based selection.

Part of the :mod:`pysmesh.session` package. A :class:`Rule` holds filters, and
:meth:`Session.select` returns the live ids of one kind that pass every filter of it. The
filters follow the filters of SALOME Shaper's ``FiltersPlugin``: each docstring names the
Shaper behaviour it follows and states every difference. :func:`rule_to_dict` and
:func:`rule_from_dict` give a rule a stored form, a JSON-compatible dict whose layout is a
versioned contract (``docs/documentation/concepts/selection-rules.md``).

**Split ids.** An id that denotes several shapes passes a filter when the filter holds for
the whole it denotes: a size filter compares the total measure, as
:meth:`Session.mass_properties` reports it, and every other filter must hold for each of
its shapes.

**Where each filter is evaluated.** The three size filters compare, here, the measures
the native session integrates as :meth:`Session.entity_table` does, and
:class:`RelativeToSolid` is :meth:`Session.entities_on`. The twelve others need OCCT tests
those queries do not carry (planarity by ``GeomLib_IsPlanarSurface``, same-geometry tests,
wire order, a face's orientation, the G1 test at an edge's middle) and run in the native
session.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import CancelPredicate, EntityId, EntityKind, ProgressCallback

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# The comparison of a size filter: below, at most, above, at least, between (inclusive) or
# strictly between. Shaper names them inf, infEq, sup, supEq, isBetween, isStrictlyBetween.
SizeComparator: TypeAlias = Literal[
    "lt", "le", "gt", "ge", "between", "strictly_between"
]

# Where a shape lies against a solid, for :class:`RelativeToSolid`, as Shaper names it.
SolidLocation: TypeAlias = Literal["in", "out", "on", "not_on", "not_out", "not_in"]

# The version of the stored form that rule_to_dict writes and rule_from_dict reads.
RULE_FORMAT: Final[int] = 1

# Shaper's margin on every size comparison: Precision::Confusion(), in the unit of the
# measure (a length, an area or a volume alike).
_SIZE_MARGIN: Final[float] = 1.0e-7

_ALL: Final[frozenset[EntityKind]] = frozenset(EntityKind)


@dataclass(frozen=True)
class BelongsTo:
    """Entities that are a sub-shape of one of the named entities, or one of them.

    Shaper's ``BelongsTo``: a shape passes when exploring one of the named shapes finds it
    (``GeomAPI_Shape::isSubShape``). An entity is a sub-shape of itself, so a named face
    passes for the kind ``FACE``. Shaper names bodies; here any live entity can be named.

    Attributes:
        entities: The entities to look in. At least one, each live.
        reverse: Keep the entities that fail instead.
    """

    entities: tuple[EntityId, ...]
    reverse: bool = False


@dataclass(frozen=True)
class ContinuousFaces:
    """Faces joined to the named faces through edges where the two faces are G1.

    Shaper's ``ContinuousFaces``: from each named face, across each of its edges, a
    neighbour face of the same body joins when OCCT's ``LocalAnalysis_SurfaceContinuity``
    finds the two G1 at the edge's middle within ``angle_deg``, and the search goes on from
    it. The named faces pass themselves. The angle between the two normals is folded into
    ``[0, 90]`` degrees, so the test does not depend on the faces' orientations: two faces
    that fold back onto each other at 175 degrees read as 5 degrees apart.

    Attributes:
        faces: The faces to start from. At least one, each live.
        angle_deg: The largest angle between the normals, in degrees (>= 0). Shaper's
            default is 5.
        reverse: Keep the faces that fail instead.
    """

    faces: tuple[EntityId, ...]
    angle_deg: float
    reverse: bool = False


@dataclass(frozen=True)
class EdgeSize:
    """Edges whose length compares with a value.

    Shaper's ``EdgeSize``, with Shaper's margin of 1e-7 on each comparison: ``"lt"`` is
    ``length < value - 1e-7``, ``"le"`` is ``length < value + 1e-7``, ``"gt"`` is
    ``length > value + 1e-7``, ``"ge"`` is ``length > value - 1e-7``, ``"between"`` is
    ``value - 1e-7 < length < value_max + 1e-7`` and ``"strictly_between"`` is
    ``value + 1e-7 < length < value_max - 1e-7``. The length is the one
    :meth:`Session.entity_table` reports, integrated adaptively; Shaper measures with
    ``GeomAPI_Edge::length``. A negative value, or a ``value`` above ``value_max`` for the
    two between comparisons, is refused; Shaper lets no edge pass then.

    Attributes:
        comparator: One of ``"lt"``, ``"le"``, ``"gt"``, ``"ge"``, ``"between"``,
            ``"strictly_between"``.
        value: The value, in model units (>= 0).
        value_max: The upper value of the two between comparisons; the others do not read
            it, as in Shaper.
        reverse: Keep the edges that fail instead.
    """

    comparator: SizeComparator
    value: float
    value_max: float = 0.0
    reverse: bool = False


@dataclass(frozen=True)
class ExternalFaces:
    """Faces that bound at most one solid of their body and are not INTERNAL.

    Shaper's ``ExternalFaces``: a face whose orientation is ``TopAbs_INTERNAL`` fails, and
    so does a face that two solids of its body share. A free face, or a face of a shell,
    bounds no solid and passes.

    Attributes:
        reverse: Keep the faces that fail instead (the shared and internal faces).
    """

    reverse: bool = False


@dataclass(frozen=True)
class FaceSize:
    """Faces whose area compares with a value.

    Shaper's ``FaceSize``, with the comparisons and the 1e-7 margin of :class:`EdgeSize`.
    The area is the one :meth:`Session.entity_table` reports, integrated adaptively;
    Shaper integrates with ``BRepGProp::SurfaceProperties`` at 1e-6.

    Attributes:
        comparator: One of ``"lt"``, ``"le"``, ``"gt"``, ``"ge"``, ``"between"``,
            ``"strictly_between"``.
        value: The value, in model units squared (>= 0).
        value_max: The upper value of the two between comparisons.
        reverse: Keep the faces that fail instead.
    """

    comparator: SizeComparator
    value: float
    value_max: float = 0.0
    reverse: bool = False


@dataclass(frozen=True)
class FeatureEdges:
    """Edges where two faces of the edge meet at more than an angle.

    Shaper's ``FeatureEdges``: an edge passes when, for some pair of the faces of its body
    that hold it, OCCT's ``LocalAnalysis_SurfaceContinuity`` finds them not G1 at the
    edge's middle within ``angle_deg``. An angle below 1e-7 passes every edge held by two
    faces. An edge held by one face (a free boundary or a seam) passes never. A pair that
    OCCT cannot analyse does not make the edge pass. The angle is folded into ``[0, 90]``
    degrees, as in :class:`ContinuousFaces`.

    Attributes:
        angle_deg: The angle, in degrees (>= 0). Shaper's default is 5.
        reverse: Keep the edges that fail instead (the smooth ones).
    """

    angle_deg: float
    reverse: bool = False


@dataclass(frozen=True)
class HorizontalFace:
    """Planar faces whose normal is parallel to the z axis.

    Shaper's ``HorizontalFaces``: the face is planar by ``GeomLib_IsPlanarSurface`` (so a
    flat B-spline face counts), and its plane's axis is within 1e-7 rad of ``+z`` or
    ``-z``.

    Attributes:
        reverse: Keep the faces that fail instead.
    """

    reverse: bool = False


@dataclass(frozen=True)
class OnGeometry:
    """Edges or faces that lie on the curve or the surface of a named edge or face.

    Shaper's ``OnGeometry``. Two edges lie on one curve when they are the same edge, or
    when OCCT finds their untrimmed curves parallel at a distance below 1e-7. Two faces lie
    on one surface when they are the same face; when both are planar and their planes
    coincide within 1e-7; when they share one surface object; when both surfaces are
    elementary and OCCT's ``IntPatch_ImpImpIntersection`` finds them tangent; or when both
    are cylinders with the same axis line and radius within 1e-7. An edge is compared
    with the named edges, a face with the named faces.

    Attributes:
        entities: The edges and faces to compare with. At least one, each live.
        reverse: Keep the entities that fail instead.
    """

    entities: tuple[EntityId, ...]
    reverse: bool = False


@dataclass(frozen=True)
class OnLine:
    """Vertices and straight edges on the line of a named straight edge.

    Shaper's ``OnLine``: a vertex passes when it lies within 1e-7 of one of the lines; an
    edge passes when it is straight and both its ends lie within 1e-7 of one line.

    Attributes:
        edges: Straight edges whose infinite lines are tested. At least one, each live.
        reverse: Keep the entities that fail instead.
    """

    edges: tuple[EntityId, ...]
    reverse: bool = False


@dataclass(frozen=True)
class OnPlane:
    """Vertices, edges and faces in the plane of a named planar face.

    Shaper's ``OnPlane``: a vertex lies within 1e-7 of the plane; a straight edge has both
    ends within 1e-7; another planar edge lies in it (a circle by its centre and axis, any
    other curve by its two ends and its middle); a planar face's plane coincides with it
    within 1e-7.

    Attributes:
        faces: Planar faces whose infinite planes are tested. At least one, each live.
        reverse: Keep the entities that fail instead.
    """

    faces: tuple[EntityId, ...]
    reverse: bool = False


@dataclass(frozen=True)
class OnPlaneSide:
    """Entities whose middle point lies on the side of a plane its normal points to.

    Shaper's ``OnPlaneSide``: the middle point lies more than 1e-7 along the outward normal
    of the named planar face. The middle point is a vertex's point, the middle of an edge
    by arc length, the surface point at the middle of a face's parameter box, or a solid's
    centre of mass. Shaper takes a face's middle from an optimal box of its pcurves;
    ``BRepTools::UVBounds`` is taken here, which can be larger on curved pcurves.

    Attributes:
        face: A planar face. Its outward normal gives the side.
        reverse: Keep the entities that fail instead.
    """

    face: EntityId
    reverse: bool = False


@dataclass(frozen=True)
class OppositeToEdge:
    """Edges opposite to a named straight edge across quadrangles, step after step.

    Shaper's ``OppositeToEdge``: in each face of its body with one wire of four edges, the
    edge two places on in the wire's order is opposite; the search goes on from each
    opposite edge. The named edge passes itself.

    Attributes:
        edge: A straight edge.
        reverse: Keep the edges that fail instead.
    """

    edge: EntityId
    reverse: bool = False


@dataclass(frozen=True)
class RelativeToSolid:
    """Faces, edges and vertices that lie in, on or out of a named solid.

    Shaper's ``RelativeToSolid``, with GEOM's classifier: :meth:`Session.entities_on` with
    the solid as the region. ``"in"``, ``"out"`` and ``"on"`` are its states of the same
    name; ``"not_on"`` is in or out; ``"not_out"`` is its ``"on_in"``; ``"not_in"`` is its
    ``"on_out"``. Shaper classifies a shape's middle point, its first vertex and its
    distance to the solid's shells (``GeomAlgoAPI_SolidClassifier``); GEOM's
    ``GEOMAlgo_FinderShapeOn2`` classifies every vertex and the inner nodes of a
    triangulation. A shape that crosses the solid's boundary is in none of the six in
    either.

    Attributes:
        solid: A live solid that denotes one solid.
        location: One of ``"in"``, ``"out"``, ``"on"``, ``"not_on"``, ``"not_out"``,
            ``"not_in"``.
        reverse: Keep the entities that fail instead.
    """

    solid: EntityId
    location: SolidLocation
    reverse: bool = False


@dataclass(frozen=True)
class TopoConnectedFaces:
    """Faces connected to a named vertex, edge or face, in the body that holds it.

    Shaper's ``TopoConnectedFaces``: for a vertex, the faces that hold it; for an edge, the
    faces that hold it; for a face, the face and every face that shares a vertex with it.
    With ``propagation``, every face joined to those through shared edges passes too.

    Attributes:
        entity: A vertex, an edge or a face that denotes one shape.
        propagation: Extend the set across shared edges.
        reverse: Keep the faces that fail instead.
    """

    entity: EntityId
    propagation: bool
    reverse: bool = False


@dataclass(frozen=True)
class VerticalFace:
    """Planar faces with a horizontal normal, and cylinders with a vertical axis.

    Shaper's ``VerticalFaces``: a planar face whose normal has ``|z| <= 1e-7``, or a face
    on a ``Geom_CylindricalSurface`` whose axis has ``||z| - 1| <= 1e-7``.

    Attributes:
        reverse: Keep the faces that fail instead.
    """

    reverse: bool = False


@dataclass(frozen=True)
class VolumeSize:
    """Solids whose volume compares with a value.

    Shaper's ``VolumeSize``, with the comparisons and the 1e-7 margin of :class:`EdgeSize`.
    The volume is the one :meth:`Session.entity_table` reports, integrated adaptively;
    Shaper integrates with ``BRepGProp::VolumeProperties`` at 1e-6.

    Attributes:
        comparator: One of ``"lt"``, ``"le"``, ``"gt"``, ``"ge"``, ``"between"``,
            ``"strictly_between"``.
        value: The value, in model units cubed (>= 0).
        value_max: The upper value of the two between comparisons.
        reverse: Keep the solids that fail instead.
    """

    comparator: SizeComparator
    value: float
    value_max: float = 0.0
    reverse: bool = False


# One selection filter.
Filter: TypeAlias = (
    "BelongsTo | ContinuousFaces | EdgeSize | ExternalFaces | FaceSize | FeatureEdges"
    " | HorizontalFace | OnGeometry | OnLine | OnPlane | OnPlaneSide | OppositeToEdge"
    " | RelativeToSolid | TopoConnectedFaces | VerticalFace | VolumeSize"
)


@dataclass(frozen=True)
class Rule:
    """A selection rule: the entities that pass every one of its filters.

    The filters combine with AND, in order, as Shaper's ``FiltersSelection`` combines them;
    each filter's ``reverse`` negates it alone.

    Attributes:
        filters: The filters. At least one.
    """

    filters: tuple[Filter, ...]


# The filter types by the name the stored form gives them, and the kinds each applies to
# (the kinds Shaper's isSupported admits, among the four a session tracks).
_FILTERS: Final[dict[str, tuple[type[DataclassInstance], frozenset[EntityKind]]]] = {
    "BelongsTo": (BelongsTo, _ALL),
    "ContinuousFaces": (ContinuousFaces, frozenset({EntityKind.FACE})),
    "EdgeSize": (EdgeSize, frozenset({EntityKind.EDGE})),
    "ExternalFaces": (ExternalFaces, frozenset({EntityKind.FACE})),
    "FaceSize": (FaceSize, frozenset({EntityKind.FACE})),
    "FeatureEdges": (FeatureEdges, frozenset({EntityKind.EDGE})),
    "HorizontalFace": (HorizontalFace, frozenset({EntityKind.FACE})),
    "OnGeometry": (OnGeometry, frozenset({EntityKind.EDGE, EntityKind.FACE})),
    "OnLine": (OnLine, frozenset({EntityKind.EDGE, EntityKind.VERTEX})),
    "OnPlane": (
        OnPlane,
        frozenset({EntityKind.FACE, EntityKind.EDGE, EntityKind.VERTEX}),
    ),
    "OnPlaneSide": (OnPlaneSide, _ALL),
    "OppositeToEdge": (OppositeToEdge, frozenset({EntityKind.EDGE})),
    "RelativeToSolid": (
        RelativeToSolid,
        frozenset({EntityKind.FACE, EntityKind.EDGE, EntityKind.VERTEX}),
    ),
    "TopoConnectedFaces": (TopoConnectedFaces, frozenset({EntityKind.FACE})),
    "VerticalFace": (VerticalFace, frozenset({EntityKind.FACE})),
    "VolumeSize": (VolumeSize, frozenset({EntityKind.SOLID})),
}

_NAME_OF: Final[dict[type, str]] = {cls: name for name, (cls, _) in _FILTERS.items()}

_COMPARATORS: Final[frozenset[str]] = frozenset(
    {"lt", "le", "gt", "ge", "between", "strictly_between"}
)
_LOCATIONS: Final[frozenset[str]] = frozenset(
    {"in", "out", "on", "not_on", "not_out", "not_in"}
)

# The fields that hold entity ids, by kind of field.
_ID_TUPLE_FIELDS: Final[frozenset[str]] = frozenset({"entities", "faces", "edges"})
_ID_FIELDS: Final[frozenset[str]] = frozenset({"face", "edge", "solid", "entity"})
_FLOAT_FIELDS: Final[frozenset[str]] = frozenset({"angle_deg", "value", "value_max"})
_BOOL_FIELDS: Final[frozenset[str]] = frozenset({"reverse", "propagation"})
_TEXT_FIELDS: Final[dict[str, frozenset[str]]] = {
    "comparator": _COMPARATORS,
    "location": _LOCATIONS,
}


def _is_int(value: object) -> bool:
    """Whether a value is an integer and not a bool."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    """Whether a value is a real number and not a bool."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _name(f: object) -> str:
    """The stored-form name of a filter, refusing an object that is not one."""
    name = _NAME_OF.get(type(f))
    if name is None:
        raise PysmeshError(
            f"Session.select: {type(f).__name__} is not a selection filter."
        )
    return name


def _ids_field(where: str, filter_name: str, field: str, value: object) -> list[int]:
    """A filter's tuple of entity ids, as ints; refused when empty or not integers."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PysmeshError(
            f"{where}: the filter {filter_name} has {field}={value!r}, not a sequence of "
            "entity ids."
        )
    bad = [v for v in value if not _is_int(v)]
    if bad:
        raise PysmeshError(
            f"{where}: the filter {filter_name} has {field} holding {bad!r}, which are "
            "not entity ids."
        )
    if not value:
        raise PysmeshError(f"{where}: the filter {filter_name} names no entity.")
    return [int(v) for v in value]


def _check(where: str, f: object) -> tuple[str, dict[str, object]]:
    """The filter's name and its fields, each checked against the filter's contract."""
    name = _name(f)
    fields: dict[str, object] = {
        field.name: getattr(f, field.name)
        for field in dataclasses.fields(_FILTERS[name][0])
    }
    for field, value in fields.items():
        if field in _ID_TUPLE_FIELDS:
            _ids_field(where, name, field, value)
        elif field in _ID_FIELDS:
            if not _is_int(value):
                raise PysmeshError(
                    f"{where}: the filter {name} has {field}={value!r}, not an entity id."
                )
        elif field in _BOOL_FIELDS:
            if not isinstance(value, bool):
                raise PysmeshError(
                    f"{where}: the filter {name} has {field}={value!r}, not a bool."
                )
        elif field in _FLOAT_FIELDS:
            if not _is_number(value) or not math.isfinite(float(cast("float", value))):
                raise PysmeshError(
                    f"{where}: the filter {name} has {field}={value!r}, not a finite "
                    "number."
                )
        elif field in _TEXT_FIELDS and value not in _TEXT_FIELDS[field]:
            raise PysmeshError(
                f"{where}: the filter {name} has {field}={value!r}, not one of "
                f"{sorted(_TEXT_FIELDS[field])}."
            )
    if "angle_deg" in fields and float(cast("float", fields["angle_deg"])) < 0.0:
        raise PysmeshError(
            f"{where}: the filter {name} has angle_deg={fields['angle_deg']!r}, below "
            "zero."
        )
    if "comparator" in fields:
        value = float(cast("float", fields["value"]))
        value_max = float(cast("float", fields["value_max"]))
        if value < 0.0:
            raise PysmeshError(
                f"{where}: the filter {name} has value={value!r}, below zero; no size "
                "is below zero."
            )
        between = fields["comparator"] in ("between", "strictly_between")
        if between and value > value_max:
            raise PysmeshError(
                f"{where}: the filter {name} has value={value!r} above "
                f"value_max={value_max!r}; no size lies between them."
            )
    return name, fields


def _size_mask(
    measure: NDArray[np.float64], comparator: str, value: float, value_max: float
) -> NDArray[np.bool_]:
    """Shaper's size comparison of every measure, with its 1e-7 margin."""
    eps = _SIZE_MARGIN
    if comparator == "lt":
        return measure < value - eps
    if comparator == "le":
        return measure < value + eps
    if comparator == "gt":
        return measure > value + eps
    if comparator == "ge":
        return measure > value - eps
    if comparator == "between":
        return (measure > value - eps) & (measure < value_max + eps)
    return (measure > value + eps) & (measure < value_max - eps)


class _Shares:
    """Maps the progress of each native call onto its share of one select call.

    Each filter of the rule is an equal share of ``[0, 1]``, and a filter that makes
    several native calls splits its share equally among them. The caller sees one strictly
    increasing sequence that ends at 1.0.
    """

    __slots__ = ("_count", "_last", "_report")

    def __init__(self, report: ProgressCallback | None, count: int) -> None:
        """Shares of ``report`` for ``count`` filters."""
        self._report = report
        self._count = count
        self._last = -1.0

    def __call__(
        self, index: int, part: int = 0, parts: int = 1
    ) -> ProgressCallback | None:
        """The callback of native call ``part`` of ``parts`` of filter ``index``."""
        report = self._report
        if report is None:
            return None

        def forward(fraction: float) -> None:
            value = (index + (part + fraction) / parts) / self._count
            if value > self._last:
                self._last = value
                report(value)

        return forward


class _SelectOps(_SessionBase):
    """Rule-based selection."""

    __slots__ = ()

    def select(
        self,
        kind: EntityKind,
        rule: Rule,
        *,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> NDArray[np.int64]:
        """The live ids of one kind that pass every filter of a rule.

        The filters follow SALOME Shaper's ``FiltersPlugin`` and combine with AND, each
        negated alone by its ``reverse``; each filter's docstring gives its test and how
        it differs from Shaper's. An id that denotes several shapes passes a filter when
        the filter holds for the whole it denotes: a size filter compares the total
        measure, and every other filter must hold for each shape. The query issues no id
        and changes nothing.

        **Progress.** Each filter of the rule is an equal share of the range. On a large
        model the size filters, :class:`RelativeToSolid` and :class:`FeatureEdges` take
        the most time. A cancel stops a size filter between two measured shapes,
        :class:`FeatureEdges` between two edges, :class:`RelativeToSolid` in its mesh, and
        every filter between two tested ids.

        Args:
            kind: The entity kind to select.
            rule: The rule. Each of its filters must apply to ``kind``.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the query runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the query. It then
                raises :class:`PysmeshCancelled`. The query changes nothing either way.

        Returns:
            (N,) int64 — the live ids of ``kind`` that pass, ascending. Empty when none
            does.

        Raises:
            PysmeshError: If ``rule`` is not a :class:`Rule` or holds no filter; if a
                filter does not apply to ``kind`` (the message names both); if a filter
                names a dead id (the message names it), an entity of the wrong kind, a
                face that is not planar or an edge that is not straight where the filter
                needs one; if a field is of the wrong type, an angle or a size is below
                zero, or a between comparison has ``value`` above ``value_max``.
            PysmeshCancelled: If ``cancel`` returned True.
        """
        if not isinstance(rule, Rule):
            raise PysmeshError(
                f"Session.select: the rule is a {type(rule).__name__}, not a Rule."
            )
        filters = tuple(rule.filters)
        if not filters:
            raise PysmeshError("Session.select: the rule holds no filter.")
        checked = [_check("Session.select", f) for f in filters]
        for name, fields in checked:
            applies = _FILTERS[name][1]
            if kind not in applies:
                raise PysmeshError(
                    f"Session.select: the filter {name} does not apply to {kind}; it "
                    f"applies to {', '.join(sorted(str(k) for k in applies))}."
                )
            for field, value in fields.items():
                if field in _ID_TUPLE_FIELDS:
                    named = [int(i) for i in cast("Sequence[int]", value)]
                elif field in _ID_FIELDS:
                    named = [int(cast("int", value))]
                else:
                    named = []
                for i in named:
                    self._require_alive(name, i)
            if name == "RelativeToSolid":
                solid = int(cast("int", fields["solid"]))
                if self._s.entity_kind(solid) != "SOLID":
                    raise PysmeshError(
                        f"Session.select: the filter RelativeToSolid names entity "
                        f"{solid}, a {self._s.entity_kind(solid)}; it takes a solid."
                    )
                if self._s.shape_count(solid) != 1:
                    raise PysmeshError(
                        f"Session.select: the filter RelativeToSolid names entity "
                        f"{solid}, which denotes {self._s.shape_count(solid)} solids "
                        "(it was split); it takes one solid."
                    )
        ids = self._s.entities(str(kind), False)
        keep = np.ones(len(ids), dtype=np.bool_)
        shares = _Shares(progress, len(checked))
        for index, (name, fields) in enumerate(checked):
            mask = self._filter_mask(kind, name, fields, ids, shares, index, cancel)
            keep &= ~mask if fields["reverse"] else mask
        closing = shares(len(checked) - 1)
        if closing is not None:
            # OCCT's indicator sums the steps of a native call, so its last position can
            # fall short of 1.0 by rounding (1 - 3e-16 measured on a size filter). The rule
            # is complete: report its end, once.
            closing(1.0)
        return ids[keep]

    def _require_alive(self, name: str, entity: int) -> None:
        """Refuse a filter that names an id that is dead or was never issued."""
        try:
            alive = self._s.entity_state(entity) == "alive"
        except PysmeshError as err:
            raise PysmeshError(
                f"Session.select: the filter {name} names entity {entity}, which this "
                "session never issued."
            ) from err
        if not alive:
            raise PysmeshError(
                f"Session.select: the filter {name} names entity {entity}, which is dead."
            )

    def _filter_mask(
        self,
        kind: EntityKind,
        name: str,
        fields: Mapping[str, object],
        ids: NDArray[np.int64],
        shares: _Shares,
        index: int,
        cancel: CancelPredicate | None,
    ) -> NDArray[np.bool_]:
        """One checked filter over every live id of ``kind``, aligned with ``ids``."""
        if name in ("EdgeSize", "FaceSize", "VolumeSize"):
            measure = self._s.size_measures(str(kind), shares(index), cancel)
            return _size_mask(
                measure,
                cast("str", fields["comparator"]),
                float(cast("float", fields["value"])),
                float(cast("float", fields["value_max"])),
            )
        if name == "RelativeToSolid":
            solid = int(cast("int", fields["solid"]))
            states = {
                "in": ("in",),
                "out": ("out",),
                "on": ("on",),
                "not_on": ("in", "out"),
                "not_out": ("on_in",),
                "not_in": ("on_out",),
            }[cast("str", fields["location"])]
            mask = np.zeros(len(ids), dtype=np.bool_)
            for part, state in enumerate(states):
                report = shares(index, part, len(states))
                found = self._s.entities_on(
                    str(kind), "solid", [], solid, state, None, report, cancel
                )
                mask |= np.isin(ids, found)
            return mask
        entity_ids: list[int] = []
        for field in ("entities", "faces", "edges"):
            if field in fields:
                entity_ids = [int(i) for i in cast("Sequence[int]", fields[field])]
        for field in ("face", "edge", "entity"):
            if field in fields:
                entity_ids = [int(cast("int", fields[field]))]
        numbers = (
            [float(cast("float", fields["angle_deg"]))] if "angle_deg" in fields else []
        )
        flag = bool(fields.get("propagation", False))
        return self._s.filter_mask(
            str(kind), name, entity_ids, numbers, flag, shares(index), cancel
        )


def rule_to_dict(rule: Rule) -> dict[str, object]:
    """The stored form of a rule: a JSON-compatible dict, format 1.

    The layout is ``{"format": 1, "filters": [...]}``, with one dict per filter in order:
    ``"type"`` holds the filter's class name, and one key per field holds its value. Ids
    are ints, tuples of ids are lists of ints, and the other values are floats, bools and
    strings. ``docs/documentation/concepts/selection-rules.md`` states the format as a
    contract.

    Args:
        rule: The rule. Each filter's fields are checked as :meth:`Session.select` checks
            them.

    Returns:
        The dict. ``json.dumps`` writes it as it is.

    Raises:
        PysmeshError: If ``rule`` is not a :class:`Rule`, or a filter is not a filter or
            has a field of the wrong type or value.
    """
    if not isinstance(rule, Rule):
        raise PysmeshError(
            f"rule_to_dict: the rule is a {type(rule).__name__}, not a Rule."
        )
    entries: list[dict[str, object]] = []
    for f in rule.filters:
        name, fields = _check("rule_to_dict", f)
        entry: dict[str, object] = {"type": name}
        for field, value in fields.items():
            if field in _ID_TUPLE_FIELDS:
                entry[field] = [int(v) for v in cast("Sequence[int]", value)]
            elif field in _ID_FIELDS:
                entry[field] = int(cast("int", value))
            elif field in _FLOAT_FIELDS:
                entry[field] = float(cast("float", value))
            else:
                entry[field] = value
        entries.append(entry)
    return {"format": RULE_FORMAT, "filters": entries}


def rule_from_dict(data: Mapping[str, object]) -> Rule:
    """The rule that a stored form describes, as :func:`rule_to_dict` writes it.

    Args:
        data: ``{"format": 1, "filters": [...]}``, as ``json.loads`` reads it back. Every
            field of every filter must be present, including ``reverse`` and
            ``value_max``.

    Returns:
        The rule.

    Raises:
        PysmeshError: If the format is not 1; if a key is missing or extra, at the top or
            in a filter (the message lists them); if a filter type is unknown; or if a
            field has the wrong type or value.
    """
    if not isinstance(data, Mapping):
        raise PysmeshError(
            f"rule_from_dict: the data is a {type(data).__name__}, not a dict."
        )
    _exact_keys("rule_from_dict: the rule", data, {"format", "filters"})
    if not _is_int(data["format"]) or data["format"] != RULE_FORMAT:
        raise PysmeshError(
            f"rule_from_dict: the format {data['format']!r} is not {RULE_FORMAT}, the "
            "format this version reads."
        )
    entries = data["filters"]
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise PysmeshError(
            f"rule_from_dict: filters is a {type(entries).__name__}, not a list."
        )
    filters: list[Filter] = []
    for k, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise PysmeshError(
                f"rule_from_dict: filter {k} is a {type(entry).__name__}, not a dict."
            )
        name = entry.get("type")
        if not isinstance(name, str) or name not in _FILTERS:
            raise PysmeshError(
                f"rule_from_dict: filter {k} has the type {name!r}, not one of "
                f"{sorted(_FILTERS)}."
            )
        cls = _FILTERS[name][0]
        names = {field.name for field in dataclasses.fields(cls)}
        _exact_keys(f"rule_from_dict: filter {k} ({name})", entry, names | {"type"})
        values: dict[str, object] = {}
        for field in names:
            value = entry[field]
            if field in _ID_TUPLE_FIELDS:
                values[field] = tuple(
                    EntityId(i)
                    for i in _ids_field("rule_from_dict", name, field, value)
                )
            elif field in _ID_FIELDS and _is_int(value):
                values[field] = EntityId(int(cast("int", value)))
            elif field in _FLOAT_FIELDS and _is_number(value):
                values[field] = float(cast("float", value))
            else:
                values[field] = value
        built = cls(**values)
        _check("rule_from_dict", built)
        filters.append(cast("Filter", built))
    return Rule(filters=tuple(filters))


def _exact_keys(where: str, data: Mapping[str, object], wanted: set[str]) -> None:
    """Refuse a mapping whose keys are not exactly ``wanted``, listing the differences."""
    keys = set(data)
    missing = sorted(wanted - keys)
    extra = sorted(str(k) for k in keys - wanted)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"misses {missing}")
        if extra:
            parts.append(f"has the extra keys {extra}")
        raise PysmeshError(f"{where} " + " and ".join(parts) + ".")
