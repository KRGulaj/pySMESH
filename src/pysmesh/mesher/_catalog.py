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

from .._core import PysmeshError
from ..viscous import ExtrusionMethod, _check_layer_stack
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

    TABLE and EXPRESSION place node ``k`` of ``N`` where the integral of the density,
    taken from the start of the edge in the normalised parameter ``t``, reaches
    ``k / N`` of its total. A table is interpolated linearly between its points. An
    expression is integrated adaptively to a relative tolerance of 1e-12; a density
    that cannot be integrated, such as one with a pole inside ``(0, 1)``, fails the
    edge with a compute error.

    BETA_LAW places node ``i`` of ``n`` at the fraction
    ``1 + b (1 - r^(1 - i/n)) / (1 + r^(1 - i/n))`` of the edge, with ``b = |beta|``
    and ``r = (b + 1) / (b - 1)`` (``StdMeshers_Regular_1D::computeBetaLaw``, after
    gmsh). A ``beta`` just above 1 crowds the nodes at the start of the edge; a
    negative ``beta`` crowds them at the end.

    Attributes:
        REGULAR: Equal segments.
        SCALE: Lengths in geometric progression, set by ``scale_factor``.
        TABLE: Density given as a table of ``(t, density)`` pairs.
        EXPRESSION: Density given as an expression in ``t``.
        BETA_LAW: The boundary-layer law above, set by ``beta``.
    """

    REGULAR = 0
    SCALE = 1
    TABLE = 2
    EXPRESSION = 3
    BETA_LAW = 4


class QuadType(IntEnum):
    """Which way :class:`Quadrangle2D` resolves a face it cannot fill with quadrangles.

    The integer values are persisted by SMESH; do not reorder.
    """

    STANDARD = 0
    TRIANGLE_PREFERENCE = 1
    QUADRANGLE_PREFERENCE = 2
    QUADRANGLE_PREFERENCE_REVERSED = 3
    REDUCED = 4


# ---- 0-D algorithms ---------------------------------------------------------------- #


@dataclass(frozen=True)
class SegmentAroundVertex0D(Algorithm):
    """Make :class:`SegmentLengthAroundVertex` take effect at one vertex.

    Assign it on the vertex, beside the hypothesis. It meshes nothing itself. The 1-D
    algorithm of each edge that the vertex bounds then moves the nodes near the vertex,
    so that the segment touching it has the hypothesis's length. Without it the
    hypothesis is accepted and never read.
    """

    native_name: ClassVar[str] = "SegmentAroundVertex_0D"


# ---- 1-D algorithms -------------------------------------------------------------------- #


@dataclass(frozen=True)
class Regular1D(Algorithm):
    """Discretise every edge, spaced by whichever 1-D hypothesis applies there.

    The usual base of any assignment: without a 1-D algorithm and a 1-D hypothesis, nothing
    of higher dimension has a boundary to work from.
    """

    native_name: ClassVar[str] = "Regular_1D"


@dataclass(frozen=True)
class UseExisting1D(Algorithm):
    """Take the segments a script made on an edge as the edge's mesh.

    It creates nothing and counts as computed, so the faces around the edge are meshed
    from the script's segments. Assign it before the script runs: a compute of the edge
    by another algorithm would replace them. Make the nodes and the segments with
    :meth:`~pysmesh.Mesher.add_nodes` and :meth:`~pysmesh.Mesher.add_segments`, both
    with ``on`` naming the edge. The end nodes are the vertex nodes, which a compute of
    the vertices makes.
    """

    native_name: ClassVar[str] = "UseExisting_1D"


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
class UseExisting2D(Algorithm):
    """Take the faces a script made on a face as the face's mesh.

    The 2-D counterpart of :class:`UseExisting1D`: it creates nothing and counts as
    computed, so a 3-D algorithm meshes the solid from the script's faces. Bind the
    script's nodes and faces to the face with ``on``.
    """

    native_name: ClassVar[str] = "UseExisting_2D"


@dataclass(frozen=True)
class Quadrangle2D(Algorithm):
    """Mapped quadrangle meshing of a face bounded by four logical sides.

    Refuses a face it cannot read as four sides — a full disk is one side, not four —
    and the compute error names the face. A face with no algorithm, or an edge of it
    with no 1-D hypothesis, is named by its algorithm state instead (``NO_ALGO``,
    ``MISSING_HYP``). :class:`QuadrangleParams` names a base vertex for a three-sided
    face; :class:`QuadranglePreference` changes what happens where the sides do not
    match.

    The map from four corners shears the cells where the corners are not where the
    face turns. A planar NACA 0012 cap, 1.25 m chord, sharp trailing edge (wedge 16.54
    degrees), split into 4 edges at the trailing edge, mid-upper, leading edge and
    mid-lower, measured on SMESH 9.16: minimum angle 2.207 degrees with 30 segments on
    every edge, 0.347 degrees with them clustered toward the ends, and 0.006 degrees
    with 40 aft and 30 fore segments under ``QUADRANGLE_PREFERENCE``. With 1 or 2 edges
    it refuses: "Face must have 4 sides but not 1"
    (``StdMeshers_Quadrangle_2D.cxx:1538``). Give such a face a better block topology,
    for example a C-shaped strip along the camber line plus a nose block, or mesh it
    with triangles.
    """

    native_name: ClassVar[str] = "Quadrangle_2D"


@dataclass(frozen=True)
class Mefisto2D(Algorithm):
    """Free triangle meshing of a face, sized by its boundary or by MaxElementArea.

    The target edge length is the mean boundary segment (with no 2-D hypothesis, or with
    :class:`LengthFromEdges`). With :class:`MaxElementArea` it is 1.06 times the side of
    an equilateral triangle of area ``max_area``, capped at the longest boundary
    segment: the bound refines the face below the boundary segments, and never coarsens
    it past them. Along boundary segments too long for the bound the triangles stay
    larger than ``max_area``; the compute then succeeds with a
    :class:`~pysmesh.mesher.ComputeWarning` on the face that names the largest area.

    Its own quality step (``teamqt``, ``mefisto2/trte.c:4903``, called at
    ``aptrte.cxx:594-612``) can leave slivers beside a boundary edge. On the 4 x 4
    square with 16 segments per side, sized by its boundary, 20 of 494 triangles are
    below 5 degrees and the smallest is 0.99 degrees (SMESH 9.16). Run
    :meth:`~pysmesh.Mesher.smooth` after it: one pass lifts the smallest angle there to
    15.45 degrees (Laplacian) or 24.93 degrees (centroidal), with no triangle below 5.
    """

    native_name: ClassVar[str] = "MEFISTO_2D"


@dataclass(frozen=True)
class PolygonPerFace2D(Algorithm):
    """One polygonal element per face, using the edge discretisation as its boundary.

    With :class:`ViscousLayers2D` it grows the quadrangle layers on the chosen edges
    first, and the polygon fills the rest of the face, its sides the inner sides of the
    layer cells.
    """

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

    It meshes a ring (two wires), or one wire of at least 4 edges whose medial axis has
    two branch points, each joined by two branches to vertices: a thin strip with two
    short ends, like a river between its banks (``getSinuousEdges``,
    ``StdMeshers_QuadFromMedialAxis_1D2D.cxx:501``). Any other face fails with "Not
    implemented so far" (``:2205``). A closed thin face with a cusp has no such ends: a
    NACA 0012 cap with a sharp trailing edge fails with 1, 2 or 4 edges and 4 or 8
    layers (SMESH 9.16). Mesh such a face as :class:`Quadrangle2D` describes, or with
    triangles.
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

    With :class:`ViscousLayers` beside it, it grows prism layers on the chosen faces. It
    shrinks the shape by the total thickness, lays the grid in the shrunk shape, and
    fills the gap with layer cells. Each layer edge runs from a node of the shrunk
    boundary to its nearest point on the wall, along the normal of a plane wall, and the
    layer nodes divide it at the closed-form fractions of the stack. The shrunk mesh
    keeps every cut cell that has volume, whatever ``size_threshold`` says, so the
    layers have no gap. A stack too thick for the shape, where one shrunk surface meets
    or crosses another, fails the compute on the SOLID and leaves no cell; the error
    names the largest total thickness for which the shrink works, found by bisection.
    Keep a margin below it: where shrunk surfaces come closer together than the grid
    spacing can follow, the layer cells fold over, and the compute fails there too.

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

    With :class:`ViscousLayers` on its solid it builds the layers, one hypothesis per
    solid, as :class:`Hexa3D` does; :class:`Hexa3D` hands a block of more than six faces
    to it. Where the split of a side does not meet the layers, the mesh is the one
    :class:`Hexa3D` makes on the same block with six faces. Where it does, it differs
    there: a vertex of the split keeps its node when the layers shrink a side face (on a
    2 x 1 x 1 block of two fused cubes, up to 0.1 away from the six-face mesh), and with
    layers on the faces around a split wall SMESH's smoothing thins that wall's stack
    near the split by about 1 %.
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
    It sweeps from a face that is already meshed: to choose the source, assign the 2-D
    algorithm on that face alone (``on=SubShape(SubShapeKind.FACE, face_id)``). With a
    global 2-D algorithm only, a face that is not a quadrangle marks the source. If
    every face reads as a quadrangle, Prism3D tries the faces in turn.

    A side face whose bottom or top side has more than one edge has a composite side,
    for example where a cap edge is split under a whole one. Prism3D projects the
    opposite side onto it as a whole. It sweeps through that face only if each split
    point gets a node, and each edge of the composite side gets the number of segments
    that its own 1-D hypothesis gives. Inside an edge, the projection places the nodes,
    as on every edge of the target cap. Otherwise Prism3D tries another face as the
    source. If none fits, the compute fails and names, for each face, why it is not the
    source. The way out: split the opposite cap edge at the same points, or give the
    split edges the segments that the opposite side puts on them. A quadratic mesh is
    refused on a composite side.

    It builds no viscous layers of its own, but it sweeps the 2-D layers of its source
    face: put :class:`ViscousLayers2D` on that face, beside a 2-D algorithm assigned on
    that face alone. :meth:`~pysmesh.Mesher.compute` refuses 2-D layers on any other
    face of the solid.
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
        reversed_edges: Ordinals of the edges on which the distribution runs from
            the edge's last vertex to its first. Use it to make a graded chain of
            edges, some defined the other way round, grow the same way along every
            edge.
        beta: The parameter of :attr:`Distribution.BETA_LAW`, read only for it. It
            must satisfy ``|beta| > 1``; values in ``[-1, 1]`` are refused, where
            the law's logarithm is undefined. The default is upstream's, 1.01.
    """

    native_name: ClassVar[str] = "NumberOfSegments"

    count: int
    distribution: Distribution = Distribution.REGULAR
    scale_factor: float = 1.0
    table: tuple[float, ...] = ()
    expression: str = ""
    conversion_mode: int = 1
    reversed_edges: tuple[int, ...] = ()
    beta: float = 1.01


@dataclass(frozen=True)
class Arithmetic1D(Hypothesis):
    """Segment lengths in arithmetic progression from one end of the edge to the other.

    Attributes:
        start_length: Length of the first segment.
        end_length: Length of the last segment.
        reversed_edges: Ordinals of the edges on which the distribution runs from
            the edge's last vertex to its first. Use it to make a graded chain of
            edges, some defined the other way round, grow the same way along every
            edge.
    """

    native_name: ClassVar[str] = "Arithmetic1D"

    start_length: float
    end_length: float
    reversed_edges: tuple[int, ...] = ()


@dataclass(frozen=True)
class StartEndLength(Hypothesis):
    """Segment lengths in geometric progression between two stated end lengths.

    Attributes:
        start_length: Length of the first segment.
        end_length: Length of the last segment.
        reversed_edges: Ordinals of the edges on which the distribution runs from
            the edge's last vertex to its first. Use it to make a graded chain of
            edges, some defined the other way round, grow the same way along every
            edge.
    """

    native_name: ClassVar[str] = "StartEndLength"

    start_length: float
    end_length: float
    reversed_edges: tuple[int, ...] = ()


@dataclass(frozen=True)
class Geometric1D(Hypothesis):
    """Segment lengths in geometric progression from a stated first length.

    Attributes:
        start_length: Length of the first segment.
        common_ratio: Ratio between one segment and the next.
        reversed_edges: Ordinals of the edges on which the distribution runs from
            the edge's last vertex to its first. Use it to make a graded chain of
            edges, some defined the other way round, grow the same way along every
            edge.
    """

    native_name: ClassVar[str] = "Geometric1D"

    start_length: float
    common_ratio: float
    reversed_edges: tuple[int, ...] = ()


@dataclass(frozen=True)
class FixedPoints1D(Hypothesis):
    """Split each edge at named normalised positions, with a count per interval.

    Attributes:
        points: Normalised positions in ``(0, 1)``, ascending. The edge ends are implicit.
        segment_counts: Segments per interval — one more entry than ``points``.
        reversed_edges: Ordinals of the edges on which the distribution runs from
            the edge's last vertex to its first. Use it to make a graded chain of
            edges, some defined the other way round, grow the same way along every
            edge.
    """

    native_name: ClassVar[str] = "FixedPoints1D"

    points: tuple[float, ...]
    segment_counts: tuple[int, ...]
    reversed_edges: tuple[int, ...] = ()


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

    It takes effect only with :class:`SegmentAroundVertex0D` assigned on the same
    vertex. Without that algorithm the hypothesis is accepted and never read.

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
class PropagOfDistribution(Hypothesis):
    """Carry the relative node spacing of one edge to every edge opposite it.

    Assign it on the edge whose 1-D hypothesis is to be repeated, as for
    :class:`Propagation`. :class:`Propagation` carries the hypothesis itself, so an
    opposite edge of another length gets another node count. This one carries the
    result: each opposite edge gets the same number of nodes, at the same fractions of
    its own length.
    """

    native_name: ClassVar[str] = "PropagOfDistribution"


@dataclass(frozen=True)
class LayerDistribution(Hypothesis):
    """Space the layers of :class:`RadialPrism3D` by a 1-D hypothesis.

    Attributes:
        distribution: The 1-D hypothesis that spaces the radial direction.
    """

    native_name: ClassVar[str] = "LayerDistribution"

    distribution: Hypothesis


@dataclass(frozen=True)
class LayerDistribution2D(Hypothesis):
    """Space the rings of :class:`RadialQuadrangle1D2D` by a 1-D hypothesis.

    The 2-D counterpart of :class:`LayerDistribution`. The 1-D law runs along the
    radius from the outer curve inward, so its first segment is the ring next to the
    curve.

    Attributes:
        distribution: The 1-D hypothesis that spaces the radial direction. Its
            ``reversed_edges`` must stay empty: it spaces rings, not the nodes of an
            edge of the model.
    """

    native_name: ClassVar[str] = "LayerDistribution2D"

    distribution: Hypothesis


@dataclass(frozen=True)
class QuadraticMesh(Hypothesis):
    """Generate second-order elements rather than linear ones.

    A whole-mesh switch: it changes what the algorithms build, so it is not the same thing as
    converting an existing linear mesh in place.
    """

    native_name: ClassVar[str] = "QuadraticMesh"


@dataclass(frozen=True)
class BlockRenumber(Hypothesis):
    """Number the hexahedra and nodes of :class:`Hexa3D` like a structured grid.

    Each block gets local axes: vertex (0, 0, 0) at the origin, and the k axis from
    there to vertex (0, 0, 1). The i and j axes follow by the right-hand rule. The
    hexahedra then come in i, j, k order with i fastest, and so do the nodes. A block
    not named in ``blocks`` takes axes parallel to the global ones, with its origin at
    the corner of least x + y + z; a block with no edge parallel to a global axis is
    then left as it is.

    Attributes:
        blocks: Per block with explicit axes, ``(solid, vertex_000, vertex_001)``: the
            ordinals of the solid, of its vertex at the local origin, and of its vertex
            at the end of the k axis. The two vertices must share an edge of the block.
    """

    native_name: ClassVar[str] = "BlockRenumber"

    blocks: tuple[tuple[int, int, int], ...] = ()


@dataclass(frozen=True)
class NotConformAllowed(Hypothesis):
    """Allow a non-conformal mesh between local algorithms that mesh their own boundary.

    Global only: SMESH refuses it on a sub-shape (``SMESH_Mesh.cxx:658-670``), and
    :meth:`~pysmesh.Mesher.assign` raises. It lets two such algorithms sit on adjacent
    sub-shapes, so that their meshes need not share nodes on the common boundary. With
    the algorithms of this catalogue, no combination is known in which it changes the
    mesh: a conformal mesh stays exactly as it is.
    """

    native_name: ClassVar[str] = "NotConformAllowed"


# ---- 2-D and 3-D hypotheses ------------------------------------------------------------ #


@dataclass(frozen=True)
class MaxElementArea(Hypothesis):
    """An upper bound on a 2-D element's area.

    It is a *bound*, not a target. With :class:`Mefisto2D` it refines the face below the
    boundary segments where it asks for smaller triangles, and it never coarsens the
    face past the longest boundary segment, so a looser bound never gives a finer mesh.
    A triangle on a boundary segment keeps that segment as an edge, so where the
    segments are too long for the bound those triangles stay larger than ``max_area``:
    the compute reports a :class:`~pysmesh.mesher.ComputeWarning` naming the largest
    area. Refine what bounds the face there — a 1-D hypothesis, or :class:`LocalLength`
    scoped to the face.
    Attributes:
        max_area: Largest element area allowed.
    """

    native_name: ClassVar[str] = "MaxElementArea"

    max_area: float


@dataclass(frozen=True)
class LengthFromEdges(Hypothesis):
    """Size the triangles of :class:`Mefisto2D` from the face's own boundary.

    The target edge length of the triangles is the mean length of the segments on the
    face's boundary. Mefisto2D uses the same rule when no 2-D hypothesis applies, so
    this states the default explicitly. MEFISTO takes the length as an ideal, not as a
    bound: the mean triangle edge measures 0.72 to 1.11 times the boundary segment on a
    square with 4 to 32 segments per side (see also :class:`Mefisto2D`).
    """

    native_name: ClassVar[str] = "LengthFromEdges"


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

    Enforced nodes put a node of the mesh at each given point: the node nearest the
    point moves there, and its row and column bend to meet it. A point is used on each
    face it projects into, within 1 % of the face's size; on another face it is ignored.
    Where the hypothesis is assigned on the face itself, a point too far from the face
    fails the compute instead. :attr:`QuadType.REDUCED` makes no enforced node.

    Attributes:
        quad_type: Which way to resolve mismatched sides.
        base_vertex: The corner to treat as the base of a three-sided face, or None.
        corner_vertices: Vertex ordinals to force as the face's corners, or empty.
        enforced_vertices: Vertex ordinals whose points get an enforced node. The
            vertex's own node is the one the quadrangles meet at.
        enforced_points: ``(x, y, z)`` points that get an enforced node.
    """

    native_name: ClassVar[str] = "QuadrangleParams"

    quad_type: QuadType = QuadType.STANDARD
    base_vertex: SubShape | None = None
    corner_vertices: tuple[int, ...] = ()
    enforced_vertices: tuple[int, ...] = ()
    enforced_points: tuple[tuple[float, float, float], ...] = ()


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
    An axis can take explicit node coordinates instead; each axis needs exactly one of
    the two.

    Attributes:
        spacing_x: Spacing expression along x, or empty when ``coordinates_x`` is given.
        spacing_y: Spacing expression along y, or empty when ``coordinates_y`` is given.
        spacing_z: Spacing expression along z, or empty when ``coordinates_z`` is given.
        size_threshold: A cut cell smaller than ``1 / size_threshold`` of a full one is
            dropped, so the mesh boundary has a dent there instead of a sliver. With
            :class:`ViscousLayers` it does not apply: the mesh inside the layers keeps
            every cut cell that has volume.
        spacing_from: The range each spacing expression covers, as fractions of the
            shape's bounding box along its axis. With one expression per axis the range
            is the whole box, ``(0.0, 1.0)``; SMESH refuses any other value
            (``StdMeshers_CartesianParameters3D.cxx:158``).
        add_edges: Also create the 1-D elements on the model's edges.
        create_faces: Also create the 2-D elements on the model's faces.
        consider_internal_faces: Treat faces interior to a solid as boundaries to cut on.
        use_quanta: Replace a cut cell by a hexahedron when its volume is more than
            ``quanta`` times the volume of the hexahedron that replaces it. The
            hexahedron is built on the cell's corners, a corner outside the body taken
            where its grid line meets the boundary. So the mesh has fewer polyhedra and
            follows the boundary less closely.
        quanta: The volume fraction above which ``use_quanta`` replaces a cut cell, in
            ``[1e-6, 1]``. Read only with ``use_quanta``. The default is upstream's.
        coordinates_x: The grid's node coordinates along x, at least 2, measured along
            the grid's x axis from the global origin. Empty when ``spacing_x`` is given.
        coordinates_y: As ``coordinates_x``, along y.
        coordinates_z: As ``coordinates_x``, along z.
        fixed_point: ``(x, y, z)`` of a point the grid passes through, or empty. With
            every axis by spacing there is a grid node at it; with two, a grid line
            through it; with one, a grid plane (SMESH ``cartesian_algo.rst``).
        axis_directions: The grid's x, y and z directions as 9 numbers. They need not
            be unit vectors or orthogonal; SMESH refuses a zero direction, two parallel
            ones, or three in one plane. The default is the global axes.
        threshold_for_internal_faces: Apply ``size_threshold`` to the cells that
            internal or shared faces cut, too. Read with ``consider_internal_faces``. A
            small piece of such a cell is then left out: a hole at the face.
    """

    native_name: ClassVar[str] = "CartesianParameters3D"

    spacing_x: str = ""
    spacing_y: str = ""
    spacing_z: str = ""
    size_threshold: float = 4.0
    spacing_from: tuple[float, ...] = (0.0, 1.0)
    add_edges: bool = False
    create_faces: bool = False
    consider_internal_faces: bool = False
    use_quanta: bool = False
    quanta: float = 0.01
    coordinates_x: tuple[float, ...] = ()
    coordinates_y: tuple[float, ...] = ()
    coordinates_z: tuple[float, ...] = ()
    fixed_point: tuple[float, ...] = ()
    axis_directions: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    threshold_for_internal_faces: bool = False


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

    :class:`Hexa3D`, :class:`CompositeHexa3D`, :class:`PolyhedronPerSolid3D` and
    :class:`Cartesian3D` build them. On a solid that another algorithm meshes,
    :meth:`~pysmesh.Mesher.compute` raises before it meshes anything.

    Several hypotheses can reach one solid, each with its own face set and stack (a
    thickness per face set). :class:`PolyhedronPerSolid3D` grows each stack on its own
    faces. :class:`Hexa3D`, :class:`CompositeHexa3D` and :class:`Cartesian3D` read one
    hypothesis per solid, so
    :meth:`~pysmesh.Mesher.compute` refuses a second one there. SMESH also refuses two
    face sets that share a face, and two that hold faces sharing an edge with a
    different ``layer_count``; :meth:`~pysmesh.Mesher.compute` raises with SMESH's
    reason before it meshes anything. To detach one of several, pass
    :meth:`~pysmesh.Mesher.unassign` an instance equal to it.

    Attributes:
        total_thickness: Total height of the layer stack.
        layer_count: Number of layers.
        stretch_factor: Ratio between one layer's thickness and the next, >= 1; 1 gives
            layers of equal thickness.
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

    def __post_init__(self) -> None:
        """Refuse a stack SMESH would not grow as stated: T > 0, N >= 1, f >= 1."""
        _check_layer_stack(
            "ViscousLayers", self.total_thickness, self.layer_count, self.stretch_factor
        )


@dataclass(frozen=True)
class ViscousLayers2D(Hypothesis):
    """Quadrangle layers grown inward from named edges of a face.

    The 2-D counterpart of :class:`ViscousLayers`, and the only 2-D form in the stack.
    :class:`Quadrangle2D`, :class:`QuadFromMedialAxis1D2D`, :class:`Mefisto2D` and
    :class:`PolygonPerFace2D` build them. On a face that another algorithm meshes,
    :meth:`~pysmesh.Mesher.compute` raises before it meshes anything. That includes a
    face of a solid that :class:`Cartesian3D` or :class:`PolyhedronPerSolid3D` meshes
    (they mesh every dimension themselves), and a face of a :class:`Prism3D` solid
    without a 2-D algorithm assigned on that face alone. With :class:`Prism3D`, assign
    the 2-D algorithm on the face the sweep starts from, with the layers there: the
    sweep carries them through every level.

    Attributes:
        total_thickness: Total height of the layer stack.
        layer_count: Number of layers.
        stretch_factor: Ratio between one layer's thickness and the next, >= 1; 1 gives
            layers of equal thickness.
        boundary: Edge ordinals the layers grow on, or their complement when ``ignore``.
        ignore: Read ``boundary`` as the exclusion list rather than the wall list.
        method: Read by the 3-D :class:`ViscousLayers` only; 2-D layers have no
            extrusion method, so any value but the default is refused.
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

    def __post_init__(self) -> None:
        """Refuse a stack SMESH would not grow as stated, and a 2-D extrusion method."""
        _check_layer_stack(
            "ViscousLayers2D",
            self.total_thickness,
            self.layer_count,
            self.stretch_factor,
        )
        if self.method != ExtrusionMethod.SURF_OFFSET_SMOOTH:
            raise PysmeshError(
                "ViscousLayers2D: 2-D layers have no extrusion method "
                f"(got {self.method.name}); method is read by the 3-D "
                "ViscousLayers only (SMESH additional_hypo.rst, 'Viscous Layers "
                "2D'). Leave it at the default."
            )
