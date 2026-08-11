# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""pySMESH mesher — the algorithm and hypothesis catalogue.

Part of the :mod:`pysmesh.mesher` package. Each entry is a frozen dataclass carrying that
algorithm's or hypothesis's parameters, and :meth:`pysmesh.mesher.Mesher.assign` attaches one
to a sub-shape.

The split between the two is SMESH's, and it is worth stating because it is what the whole
assignment model rests on. An **algorithm** decides *how* a sub-shape is meshed and takes no
parameters of its own. A **hypothesis** supplies a number the algorithm reads — a segment
count, a maximum area, a layer thickness — and several may sit beside one algorithm. Which
of them applies where is resolved per sub-shape, so a hypothesis assigned to the whole shape
is the default and one assigned to a face overrides it there.

Dimensions matter too, and not only as bookkeeping: a 3-D algorithm normally needs a 2-D one
below it to mesh the boundary first, and a 1-D one below that. Three of the algorithms here
break that rule by meshing every dimension themselves — :class:`Cartesian3D`,
:class:`PolyhedronPerSolid3D` and :class:`Prism3D`.

**A lower-dimension algorithm assigned beside one of those is accepted and then ignored.**
SMESH calls this hiding, and it treats it as a normal state rather than a conflict, so
:meth:`~pysmesh.Mesher.assign` does not refuse it — refusing would break the ordinary pattern
of setting a model-wide default and overriding it on one solid. The consequence is that a 2-D
assignment can silently have no effect where an all-dimensional algorithm governs. Read
:attr:`~pysmesh.ComputeReport.meshed` to see which sub-shapes actually received elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

from ..viscous import ExtrusionMethod
from ._types import SubShape, _Spec


@dataclass(frozen=True)
class Algorithm(_Spec):
    """An algorithm: what meshes a sub-shape. Carries no parameters of its own."""


@dataclass(frozen=True)
class Hypothesis(_Spec):
    """A hypothesis: a parameter the algorithm beside it reads."""


# ---- Enumerated hypothesis parameters -------------------------------------------------- #


class Distribution(IntEnum):
    """How :class:`NumberOfSegments` spaces its segments along an edge.

    The integer values are persisted by SMESH; do not reorder.

    Attributes:
        REGULAR: Equal segments.
        SCALE: Lengths in geometric progression, set by ``scale_factor``.
        TABLE: Density given as a table of ``(t, density)`` pairs.
        EXPRESSION: Density given as an expression in ``t``.
    """

    REGULAR = 0
    SCALE = 1
    TABLE = 2
    EXPRESSION = 3


class QuadType(IntEnum):
    """Which way :class:`Quadrangle2D` resolves a face it cannot fill with quadrangles.

    The integer values are persisted by SMESH; do not reorder.
    """

    STANDARD = 0
    TRIANGLE_PREFERENCE = 1
    QUADRANGLE_PREFERENCE = 2
    QUADRANGLE_PREFERENCE_REVERSED = 3
    REDUCED = 4


# ---- 1-D algorithms -------------------------------------------------------------------- #


@dataclass(frozen=True)
class Regular1D(Algorithm):
    """Discretise every edge, spaced by whichever 1-D hypothesis applies there.

    The usual base of any assignment: without a 1-D algorithm and a 1-D hypothesis, nothing
    of higher dimension has a boundary to work from.
    """

    native_name: ClassVar[str] = "Regular_1D"


@dataclass(frozen=True)
class CompositeSegment1D(Algorithm):
    """Discretise a chain of C1-continuous edges as if it were one edge.

    Useful where an import has split what is geometrically a single curve into several edges:
    a segment count then applies to the whole chain instead of to each piece.
    """

    native_name: ClassVar[str] = "CompositeSegment_1D"


@dataclass(frozen=True)
class Projection1D(Algorithm):
    """Copy an edge's discretisation from another edge. Needs :class:`ProjectionSource1D`."""

    native_name: ClassVar[str] = "Projection_1D"


# ---- 2-D algorithms -------------------------------------------------------------------- #


@dataclass(frozen=True)
class Quadrangle2D(Algorithm):
    """Mapped quadrangle meshing of a face bounded by four logical sides.

    Refuses a face it cannot read as four sides — a full disk is one side, not four — and
    says so naming the face. :class:`QuadrangleParams` names a base vertex for a three-sided
    face; :class:`QuadranglePreference` changes what happens where the sides do not match.
    """

    native_name: ClassVar[str] = "Quadrangle_2D"


@dataclass(frozen=True)
class Mefisto2D(Algorithm):
    """Free triangle meshing of a face. Sized by :class:`MaxElementArea`."""

    native_name: ClassVar[str] = "MEFISTO_2D"


@dataclass(frozen=True)
class PolygonPerFace2D(Algorithm):
    """One polygonal element per face, using the edge discretisation as its boundary."""

    native_name: ClassVar[str] = "PolygonPerFace_2D"


@dataclass(frozen=True)
class Projection2D(Algorithm):
    """Copy a face's mesh from another face. Needs :class:`ProjectionSource2D`.

    This is how a periodic pair is made to match node for node, which no free mesher can
    guarantee.
    """

    native_name: ClassVar[str] = "Projection_2D"


@dataclass(frozen=True)
class Projection1D2D(Algorithm):
    """Project a face's mesh *and* its boundary discretisation from another face."""

    native_name: ClassVar[str] = "Projection_1D2D"


@dataclass(frozen=True)
class QuadFromMedialAxis1D2D(Algorithm):
    """Quad-dominant meshing of a thin face, built on its medial axis.

    The one algorithm in this catalogue that reports true progress of its own; the rest
    report at sub-mesh granularity.
    """

    native_name: ClassVar[str] = "QuadFromMedialAxis_1D2D"


@dataclass(frozen=True)
class RadialQuadrangle1D2D(Algorithm):
    """Radial quadrangle meshing of a disk or an annulus.

    Layer count comes from :class:`NumberOfLayers2D`, or from whichever 1-D hypothesis
    applies to the radial direction.
    """

    native_name: ClassVar[str] = "RadialQuadrangle_1D2D"


# ---- 3-D algorithms -------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cartesian3D(Algorithm):
    """Body-fitted Cartesian volume meshing.

    A regular grid fills the interior and is cut against the geometry at the boundary, so the
    result is hexahedra inside and **polyhedra** at every cut cell. Two consequences follow
    and neither is optional to know:

    * It meshes every dimension itself and ignores any boundary mesh. A 1-D or 2-D
      algorithm assigned beside it is accepted and then hidden, so it is effectively
      assigned alone whether or not that was intended. This is also why a Cartesian region
      does **not** conform to a neighbouring region meshed another way: it lays its own grid
      rather than growing from a shared boundary mesh.
    * Its polyhedral cells have no representation in the Inria ``.mesh`` format, so such a
      mesh cannot be written with :func:`pysmesh.write_gmf`.

    Sized by :class:`CartesianParameters3D`.
    """

    native_name: ClassVar[str] = "Cartesian_3D"


@dataclass(frozen=True)
class Hexa3D(Algorithm):
    """Structured hexahedral meshing of a block — a solid bounded by six logical faces.

    Consumes the 2-D mesh below it, so it conforms to a neighbour that does the same.
    """

    native_name: ClassVar[str] = "Hexa_3D"


@dataclass(frozen=True)
class CompositeHexa3D(Algorithm):
    """Structured hexahedral meshing of a solid whose six logical sides are each split.

    The counterpart of :class:`Hexa3D` for a block an import has cut into more than six
    faces.
    """

    native_name: ClassVar[str] = "CompositeHexa_3D"


@dataclass(frozen=True)
class HexaFromSkin3D(Algorithm):
    """Fill a solid with hexahedra derived from an existing all-quadrangle surface mesh."""

    native_name: ClassVar[str] = "HexaFromSkin_3D"


@dataclass(frozen=True)
class Prism3D(Algorithm):
    """Extrude a source face's mesh through a prismatic solid.

    Meshes the lateral faces and edges itself, so only the source face needs a 2-D algorithm.
    """

    native_name: ClassVar[str] = "Prism_3D"


@dataclass(frozen=True)
class RadialPrism3D(Algorithm):
    """O-grid between an inner and an outer shell — a pipe wall, an annulus.

    Radial layer count comes from :class:`NumberOfLayers` or :class:`LayerDistribution`.
    """

    native_name: ClassVar[str] = "RadialPrism_3D"


@dataclass(frozen=True)
class Projection3D(Algorithm):
    """Copy a solid's mesh from another solid. Needs :class:`ProjectionSource3D`."""

    native_name: ClassVar[str] = "Projection_3D"


@dataclass(frozen=True)
class PolyhedronPerSolid3D(Algorithm):
    """One polyhedral element per solid, from the face mesh bounding it.

    Meshes every dimension itself — it owns its own 1-D and 2-D sub-meshers — so a
    lower-dimension algorithm beside it is accepted and then hidden. Unlike
    :class:`Cartesian3D` it does consume an existing boundary mesh where one is present, so
    it conforms to a neighbour at a shared face.
    """

    native_name: ClassVar[str] = "PolyhedronPerSolid_3D"


# ---- 1-D hypotheses -------------------------------------------------------------------- #


@dataclass(frozen=True)
class NumberOfSegments(Hypothesis):
    """Split each edge into a fixed number of segments.

    Attributes:
        count: Segments per edge.
        distribution: How the segment lengths vary along the edge.
        scale_factor: Ratio of last to first segment, read only when ``distribution`` is
            :attr:`Distribution.SCALE`.
        table: ``(t0, d0, t1, d1, ...)`` density table, read only for
            :attr:`Distribution.TABLE`.
        expression: Density as an expression in ``t``, read only for
            :attr:`Distribution.EXPRESSION`.
        conversion_mode: 0 to treat the density as exponential, 1 to cut it at zero. Read
            only by the table and expression forms.
    """

    native_name: ClassVar[str] = "NumberOfSegments"

    count: int
    distribution: Distribution = Distribution.REGULAR
    scale_factor: float = 1.0
    table: tuple[float, ...] = ()
    expression: str = ""
    conversion_mode: int = 1


@dataclass(frozen=True)
class Arithmetic1D(Hypothesis):
    """Segment lengths in arithmetic progression from one end of the edge to the other.

    Attributes:
        start_length: Length of the first segment.
        end_length: Length of the last segment.
    """

    native_name: ClassVar[str] = "Arithmetic1D"

    start_length: float
    end_length: float


@dataclass(frozen=True)
class StartEndLength(Hypothesis):
    """Segment lengths in geometric progression between two stated end lengths.

    Attributes:
        start_length: Length of the first segment.
        end_length: Length of the last segment.
    """

    native_name: ClassVar[str] = "StartEndLength"

    start_length: float
    end_length: float


@dataclass(frozen=True)
class Geometric1D(Hypothesis):
    """Segment lengths in geometric progression from a stated first length.

    Attributes:
        start_length: Length of the first segment.
        common_ratio: Ratio between one segment and the next.
    """

    native_name: ClassVar[str] = "Geometric1D"

    start_length: float
    common_ratio: float


@dataclass(frozen=True)
class FixedPoints1D(Hypothesis):
    """Split each edge at named normalised positions, with a count per interval.

    Attributes:
        points: Normalised positions in ``(0, 1)``, ascending. The edge ends are implicit.
        segment_counts: Segments per interval — one more entry than ``points``.
    """

    native_name: ClassVar[str] = "FixedPoints1D"

    points: tuple[float, ...]
    segment_counts: tuple[int, ...]


@dataclass(frozen=True)
class Adaptive1D(Hypothesis):
    """Segment length chosen per edge so the chord stays within a deflection.

    Attributes:
        min_size: Shortest segment allowed.
        max_size: Longest segment allowed.
        deflection: Largest distance allowed between a segment and the edge it approximates.
    """

    native_name: ClassVar[str] = "Adaptive1D"

    min_size: float
    max_size: float
    deflection: float


@dataclass(frozen=True)
class AutomaticLength(Hypothesis):
    """Segment length derived from the model's own size.

    Attributes:
        fineness: 0 for coarse, 1 for fine.
    """

    native_name: ClassVar[str] = "AutomaticLength"

    fineness: float = 0.0


@dataclass(frozen=True)
class Deflection1D(Hypothesis):
    """Segment length chosen so the chord stays within a deflection.

    Attributes:
        deflection: Largest distance allowed between a segment and the edge.
    """

    native_name: ClassVar[str] = "Deflection1D"

    deflection: float


@dataclass(frozen=True)
class LocalLength(Hypothesis):
    """A target segment length, applied to every edge it governs.

    Attributes:
        length: Target segment length.
        precision: Rounding tolerance on the resulting segment count.
    """

    native_name: ClassVar[str] = "LocalLength"

    length: float
    precision: float = 1e-7


@dataclass(frozen=True)
class MaxLength(Hypothesis):
    """An upper bound on segment length.

    Attributes:
        length: Longest segment allowed.
        use_preestimated: Take the length from the model's size instead of ``length``.
    """

    native_name: ClassVar[str] = "MaxLength"

    length: float
    use_preestimated: bool = False


@dataclass(frozen=True)
class SegmentLengthAroundVertex(Hypothesis):
    """A segment length applied to the segments touching one vertex.

    Attributes:
        length: Target length next to the vertex.
    """

    native_name: ClassVar[str] = "SegmentLengthAroundVertex"

    length: float


@dataclass(frozen=True)
class Propagation(Hypothesis):
    """Carry the 1-D hypothesis of one edge to every edge opposite it on a quadrangle face.

    This is what keeps a structured mesh's opposite sides matched without stating each one.
    """

    native_name: ClassVar[str] = "Propagation"


@dataclass(frozen=True)
class LayerDistribution(Hypothesis):
    """Space the layers of :class:`RadialPrism3D` by a 1-D hypothesis.

    Attributes:
        distribution: The 1-D hypothesis that spaces the radial direction.
    """

    native_name: ClassVar[str] = "LayerDistribution"

    distribution: Hypothesis


@dataclass(frozen=True)
class QuadraticMesh(Hypothesis):
    """Generate second-order elements rather than linear ones.

    A whole-mesh switch: it changes what the algorithms build, so it is not the same thing as
    converting an existing linear mesh in place.
    """

    native_name: ClassVar[str] = "QuadraticMesh"


# ---- 2-D and 3-D hypotheses ------------------------------------------------------------ #


@dataclass(frozen=True)
class MaxElementArea(Hypothesis):
    """An upper bound on a 2-D element's area.

    It is a *bound*, not a target: a free mesher sizes its interior from the boundary
    discretisation, so this only binds where that boundary would otherwise produce elements
    larger than ``max_area``. Refining a face means refining what bounds it — a 1-D
    hypothesis, or :class:`LocalLength` scoped to the face.

    Attributes:
        max_area: Largest element area allowed.
    """

    native_name: ClassVar[str] = "MaxElementArea"

    max_area: float


@dataclass(frozen=True)
class MaxElementVolume(Hypothesis):
    """An upper bound on a 3-D element's volume.

    Attributes:
        max_volume: Largest element volume allowed.
    """

    native_name: ClassVar[str] = "MaxElementVolume"

    max_volume: float


@dataclass(frozen=True)
class QuadranglePreference(Hypothesis):
    """Prefer quadrangles over triangles where a face's sides do not match."""

    native_name: ClassVar[str] = "QuadranglePreference"


@dataclass(frozen=True)
class QuadrangleParams(Hypothesis):
    """How :class:`Quadrangle2D` reads a face that is not a plain four-sided patch.

    Attributes:
        quad_type: Which way to resolve mismatched sides.
        base_vertex: The corner to treat as the base of a three-sided face, or None.
        corner_vertices: Vertex ordinals to force as the face's corners, or empty.
    """

    native_name: ClassVar[str] = "QuadrangleParams"

    quad_type: QuadType = QuadType.STANDARD
    base_vertex: SubShape | None = None
    corner_vertices: tuple[int, ...] = ()


@dataclass(frozen=True)
class NumberOfLayers(Hypothesis):
    """Radial layer count for :class:`RadialPrism3D`.

    Attributes:
        count: Layers between the inner and the outer shell.
    """

    native_name: ClassVar[str] = "NumberOfLayers"

    count: int


@dataclass(frozen=True)
class NumberOfLayers2D(Hypothesis):
    """Radial layer count for :class:`RadialQuadrangle1D2D`.

    Attributes:
        count: Layers between the inner and the outer boundary.
    """

    native_name: ClassVar[str] = "NumberOfLayers2D"

    count: int


@dataclass(frozen=True)
class CartesianParameters3D(Hypothesis):
    """The grid :class:`Cartesian3D` cuts against the geometry.

    Spacing is stated per axis as an expression in the normalised coordinate ``t`` — a plain
    number is a constant spacing, and something like ``"5+10*t"`` grades it across the model.

    Attributes:
        spacing_x: Spacing expression along x.
        spacing_y: Spacing expression along y.
        spacing_z: Spacing expression along z.
        size_threshold: A cut cell smaller than ``1 / size_threshold`` of a full one is
            merged into its neighbour rather than kept as a sliver.
        add_edges: Also create the 1-D elements on the model's edges.
        create_faces: Also create the 2-D elements on the model's faces.
        consider_internal_faces: Treat faces interior to a solid as boundaries to cut on.
    """

    native_name: ClassVar[str] = "CartesianParameters3D"

    spacing_x: str
    spacing_y: str
    spacing_z: str
    size_threshold: float = 4.0
    spacing_from: tuple[float, ...] = (0.0, 1.0)
    add_edges: bool = False
    create_faces: bool = False
    consider_internal_faces: bool = False


# ---- Hypotheses that name another part of the model ------------------------------------ #


@dataclass(frozen=True)
class ProjectionSource1D(Hypothesis):
    """Where :class:`Projection1D` copies an edge's discretisation from.

    Attributes:
        source_edge: The edge to copy from.
        source_vertex: Which end of the source edge maps to ``target_vertex``, or None to
            let the algorithm choose.
        target_vertex: The end of the target edge it maps to, or None.
    """

    native_name: ClassVar[str] = "ProjectionSource1D"

    source_edge: SubShape
    source_vertex: SubShape | None = None
    target_vertex: SubShape | None = None


@dataclass(frozen=True)
class ProjectionSource2D(Hypothesis):
    """Where :class:`Projection2D` copies a face's mesh from.

    The two vertex pairs pin the orientation. Without them the algorithm picks a
    correspondence itself, which is fine for a face with one obvious mapping and not for a
    periodic pair where the wrong choice is a rotated mesh.

    Attributes:
        source_face: The face to copy from.
        source_vertex1: First source corner, or None.
        source_vertex2: Second source corner, or None.
        target_vertex1: The target corner ``source_vertex1`` maps to, or None.
        target_vertex2: The target corner ``source_vertex2`` maps to, or None.
    """

    native_name: ClassVar[str] = "ProjectionSource2D"

    source_face: SubShape
    source_vertex1: SubShape | None = None
    source_vertex2: SubShape | None = None
    target_vertex1: SubShape | None = None
    target_vertex2: SubShape | None = None


@dataclass(frozen=True)
class ProjectionSource3D(Hypothesis):
    """Where :class:`Projection3D` copies a solid's mesh from.

    Attributes:
        source_solid: The solid to copy from.
        source_vertex1: First source corner, or None.
        source_vertex2: Second source corner, or None.
        target_vertex1: The target corner ``source_vertex1`` maps to, or None.
        target_vertex2: The target corner ``source_vertex2`` maps to, or None.
    """

    native_name: ClassVar[str] = "ProjectionSource3D"

    source_solid: SubShape
    source_vertex1: SubShape | None = None
    source_vertex2: SubShape | None = None
    target_vertex1: SubShape | None = None
    target_vertex2: SubShape | None = None


@dataclass(frozen=True)
class ViscousLayers(Hypothesis):
    """Prism layers grown inward from named faces of a solid.

    Attributes:
        total_thickness: Total height of the layer stack.
        layer_count: Number of layers.
        stretch_factor: Ratio between one layer's thickness and the next.
        boundary: Face ordinals the layers grow on, or — when ``ignore`` is True — the faces
            they do **not** grow on.
        ignore: Read ``boundary`` as the exclusion list rather than the wall list.
        method: How a node is translated away from the wall.
        group_name: Name of the element group the layers are collected into. Required,
            because the group is the only way to find the layer cells afterwards.
    """

    native_name: ClassVar[str] = "ViscousLayers"

    total_thickness: float
    layer_count: int
    stretch_factor: float
    boundary: tuple[int, ...]
    group_name: str
    ignore: bool = False
    method: ExtrusionMethod = ExtrusionMethod.SURF_OFFSET_SMOOTH


@dataclass(frozen=True)
class ViscousLayers2D(Hypothesis):
    """Quadrangle layers grown inward from named edges of a face.

    The 2-D counterpart of :class:`ViscousLayers`, and the only 2-D form in the stack.

    Attributes:
        total_thickness: Total height of the layer stack.
        layer_count: Number of layers.
        stretch_factor: Ratio between one layer's thickness and the next.
        boundary: Edge ordinals the layers grow on, or their complement when ``ignore``.
        ignore: Read ``boundary`` as the exclusion list rather than the wall list.
        method: How a node is translated away from the wall.
        group_name: Name of the element group the layers are collected into.
    """

    native_name: ClassVar[str] = "ViscousLayers2D"

    total_thickness: float
    layer_count: int
    stretch_factor: float
    boundary: tuple[int, ...]
    group_name: str
    ignore: bool = False
    method: ExtrusionMethod = ExtrusionMethod.SURF_OFFSET_SMOOTH
