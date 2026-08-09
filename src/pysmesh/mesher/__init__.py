"""Volume and surface meshing: SMESH's algorithms, hypotheses and assignment model.

This is the meshing side of the CAD/mesher boundary. A :class:`~pysmesh.Session` owns a live
shape and hands one BREP over at the handoff; a :class:`Mesher` takes that shape back as a
plain :class:`~pysmesh.Shape` — positional ordinals, which is what the stateless API already
speaks — and turns it into a mesh.

The point of the package is not any single algorithm. It is SMESH's **assignment model**: an
algorithm plus its hypotheses are attached to a *sub-shape*, and each part of the model is
meshed by whichever assignment governs it. That is what lets one mesh be structured through
an extruded region, body-fitted in the interior, projected onto a periodic face and layered
at the walls — which a single global algorithm cannot express at all::

    mesher = Mesher(load_brep(data))
    mesher.assign(Regular1D())                                  # the whole shape
    mesher.assign(NumberOfSegments(count=4))
    mesher.assign(Quadrangle2D())
    mesher.assign(Hexa3D(), on=SubShape(SubShapeKind.SOLID, 1))  # this solid only
    report = mesher.compute()
    mesh = mesher.mesh()

Once a mesh exists, two things say whether it is any good and what parts of it mean what.
**Quality controls** measure and classify: ``mesher.quality(AspectRatio3D())`` gives one
number per cell, and ``mesher.select(BadOrientedVolume())`` gives the ids of the cells that
are inverted. **Groups** name: a named set of elements that the mesher itself maintains
across editing, so a wall named on a coarse mesh is still the wall after the mesh has been
converted to second order or split.

**Three id spaces meet here and must not be confused.** A mesh entity carries its own mesh
id; the sub-shape it sits on is named by a *positional ordinal* of the meshed shape; and the
session that produced that shape names the same sub-shape by an :data:`~pysmesh.EntityId`.
The middle one is the pivot: pairing :attr:`MeshData.element_ordinal` with the handoff's
per-kind id array carries a mesh cell all the way back to the model it came from.

Thread contract: a :class:`Mesher` is **not** thread-safe. Use one per thread. Meshing
releases the GIL, and the progress and cancel hooks are called from a helper thread.

**Layout.**

===================  =====================================================================
``_base``            the native handle every operation group is written against
``_types``          the element and sub-shape enums, and what a compute and harvest return
``_catalog``         the algorithm and hypothesis dataclasses
``_mesh``            assignment, compute and the harvest
``_controls``        the quality controls, the predicates and the filter algebra
``_group``           named groups of elements and nodes
``_edit``            the mesh editor
``_search``          location, ray casting, classification and slot cutting
``_medial``          the medial axis of a face
``_block``           block decomposition and pattern mapping
``_gmf``             Inria ``.mesh`` / ``.meshb`` interchange
===================  =====================================================================
"""

from __future__ import annotations

from ._base import _MesherBase
from ._catalog import (
    Adaptive1D,
    Algorithm,
    Arithmetic1D,
    AutomaticLength,
    Cartesian3D,
    CartesianParameters3D,
    CompositeHexa3D,
    CompositeSegment1D,
    Deflection1D,
    Distribution,
    FixedPoints1D,
    Geometric1D,
    Hexa3D,
    HexaFromSkin3D,
    Hypothesis,
    LayerDistribution,
    LocalLength,
    MaxElementArea,
    MaxElementVolume,
    MaxLength,
    Mefisto2D,
    NumberOfLayers,
    NumberOfLayers2D,
    NumberOfSegments,
    PolygonPerFace2D,
    PolyhedronPerSolid3D,
    Prism3D,
    Projection1D,
    Projection1D2D,
    Projection2D,
    Projection3D,
    ProjectionSource1D,
    ProjectionSource2D,
    ProjectionSource3D,
    Propagation,
    QuadFromMedialAxis1D2D,
    Quadrangle2D,
    QuadrangleParams,
    QuadranglePreference,
    QuadraticMesh,
    QuadType,
    RadialPrism3D,
    RadialQuadrangle1D2D,
    Regular1D,
    SegmentLengthAroundVertex,
    StartEndLength,
    ViscousLayers,
    ViscousLayers2D,
)
from ._controls import (
    And,
    Area,
    AspectRatio,
    AspectRatio3D,
    BadOrientedVolume,
    BareBorderFace,
    BareBorderVolume,
    BelongToGroup,
    CoincidentElements,
    CoincidentNodes,
    Control,
    Deflection2D,
    ElementsOnShape,
    EqualTo,
    FreeBorders,
    FreeEdges,
    FreeFaces,
    FreeNodes,
    Length,
    Length2D,
    Length3D,
    LessThan,
    ManifoldPart,
    MaxElementLength2D,
    MaxElementLength3D,
    MinimumAngle,
    MoreThan,
    MultiConnection,
    MultiConnection2D,
    NodeConnectivityNumber,
    Not,
    Or,
    OverConstrainedFace,
    OverConstrainedVolume,
    Predicate,
    QualityResult,
    RangeOfIds,
    Selection,
    Skew,
    Taper,
    Volume,
    Warping,
    _QualityOps,
    quality,
    select,
)
from ._block import (
    BLOCK_EDGE_NAMES,
    BLOCK_FACE_NAMES,
    BLOCK_VERTEX_NAMES,
    Block,
    BlockParameters,
    PatternReport,
    _PatternOps,
    block_parameters,
    block_points,
    block_shapes,
)
from ._edit import EditReport, SmoothMethod, SplitMethod, _EditOps
from ._medial import BranchEnd, MedialAxis, MedialBranch, medial_axis
from ._search import (
    ClosestElements,
    ElementsAtPoints,
    FacePatches,
    MergeObstruction,
    PointState,
    ProjectedPoints,
    RayHits,
    SharpEdges,
    SlotBoundary,
    _SearchOps,
)
from ._gmf import GmfMesh, gmf_unwritable_types, gmf_writable_group_name, read_gmf, write_gmf
from ._group import _GroupOps
from ._mesh import _MeshOps

# The hook aliases belong to the session, which defines them; the mesher shares them
# rather than declaring a second pair that means the same thing.
from ..session import CancelPredicate, ProgressCallback
from ._types import (
    GMF_REQUIRED_MARKER,
    GMF_WRITABLE_TYPES,
    ComputeReport,
    ElementDimension,
    ElementType,
    GroupSource,
    MeshData,
    MeshGroup,
    SubMeshCount,
    SubShape,
    SubShapeKind,
)


class Mesher(_MeshOps, _QualityOps, _GroupOps, _EditOps, _SearchOps, _PatternOps):
    """A shape, the algorithms and hypotheses assigned across it, and the mesh they produce.

    A mesher is built on one :class:`~pysmesh.Shape` and holds a single mesh. Assignments
    accumulate; :meth:`compute` runs them all and :meth:`mesh` reads the result back as
    arrays.

    Assignment is scoped. An assignment with no ``on`` governs the whole shape and is the
    default everywhere; one naming a sub-shape overrides it there. That is the whole of the
    model, and it is what makes a mixed mesh expressible.

    Once computed, :meth:`quality` and :meth:`select` measure and classify the result, and
    the group methods name parts of it in a way that survives editing.

    Thread contract: **not thread-safe**. One mesher per thread. :meth:`compute` releases the
    GIL and calls its hooks from a helper thread.

    Example:
        >>> m = Mesher(load_brep(data))                     # doctest: +SKIP
        >>> m.assign(Regular1D())                           # doctest: +SKIP
        >>> m.assign(NumberOfSegments(count=4))             # doctest: +SKIP
        >>> m.assign(Quadrangle2D())                        # doctest: +SKIP
        >>> m.assign(Hexa3D())                              # doctest: +SKIP
        >>> report = m.compute()                            # doctest: +SKIP
        >>> worst = m.quality(AspectRatio3D()).values.max()  # doctest: +SKIP
    """

    __slots__ = ()

    def __enter__(self) -> Mesher:
        """Enter a context that releases the mesh on exit.

        Returns:
            This mesher.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the mesh."""
        self.release()


__all__ = [
    "Adaptive1D",
    "Algorithm",
    "And",
    "Area",
    "Arithmetic1D",
    "AspectRatio",
    "AspectRatio3D",
    "AutomaticLength",
    "BLOCK_EDGE_NAMES",
    "BLOCK_FACE_NAMES",
    "BLOCK_VERTEX_NAMES",
    "BadOrientedVolume",
    "BareBorderFace",
    "BareBorderVolume",
    "BelongToGroup",
    "Block",
    "BlockParameters",
    "BranchEnd",
    "CancelPredicate",
    "Cartesian3D",
    "CartesianParameters3D",
    "ClosestElements",
    "CoincidentElements",
    "CoincidentNodes",
    "CompositeHexa3D",
    "CompositeSegment1D",
    "ComputeReport",
    "Control",
    "Deflection1D",
    "Deflection2D",
    "Distribution",
    "EditReport",
    "ElementDimension",
    "ElementType",
    "ElementsAtPoints",
    "ElementsOnShape",
    "EqualTo",
    "FacePatches",
    "FixedPoints1D",
    "FreeBorders",
    "FreeEdges",
    "FreeFaces",
    "FreeNodes",
    "GMF_REQUIRED_MARKER",
    "GMF_WRITABLE_TYPES",
    "Geometric1D",
    "GmfMesh",
    "GroupSource",
    "Hexa3D",
    "HexaFromSkin3D",
    "Hypothesis",
    "LayerDistribution",
    "Length",
    "Length2D",
    "Length3D",
    "LessThan",
    "LocalLength",
    "ManifoldPart",
    "MaxElementArea",
    "MaxElementLength2D",
    "MaxElementLength3D",
    "MaxElementVolume",
    "MaxLength",
    "MedialAxis",
    "MedialBranch",
    "Mefisto2D",
    "MergeObstruction",
    "MeshData",
    "MeshGroup",
    "Mesher",
    "MinimumAngle",
    "MoreThan",
    "MultiConnection",
    "MultiConnection2D",
    "NodeConnectivityNumber",
    "Not",
    "NumberOfLayers",
    "NumberOfLayers2D",
    "NumberOfSegments",
    "Or",
    "OverConstrainedFace",
    "OverConstrainedVolume",
    "PatternReport",
    "PointState",
    "PolygonPerFace2D",
    "PolyhedronPerSolid3D",
    "Predicate",
    "Prism3D",
    "ProgressCallback",
    "ProjectedPoints",
    "Projection1D",
    "Projection1D2D",
    "Projection2D",
    "Projection3D",
    "ProjectionSource1D",
    "ProjectionSource2D",
    "ProjectionSource3D",
    "Propagation",
    "QuadFromMedialAxis1D2D",
    "QuadType",
    "Quadrangle2D",
    "QuadrangleParams",
    "QuadranglePreference",
    "QuadraticMesh",
    "QualityResult",
    "RadialPrism3D",
    "RadialQuadrangle1D2D",
    "RangeOfIds",
    "RayHits",
    "Regular1D",
    "SegmentLengthAroundVertex",
    "Selection",
    "SharpEdges",
    "Skew",
    "SlotBoundary",
    "SmoothMethod",
    "SplitMethod",
    "StartEndLength",
    "SubMeshCount",
    "SubShape",
    "SubShapeKind",
    "Taper",
    "ViscousLayers",
    "ViscousLayers2D",
    "Volume",
    "Warping",
    "block_parameters",
    "block_points",
    "block_shapes",
    "gmf_unwritable_types",
    "gmf_writable_group_name",
    "medial_axis",
    "quality",
    "read_gmf",
    "select",
    "write_gmf",
]
