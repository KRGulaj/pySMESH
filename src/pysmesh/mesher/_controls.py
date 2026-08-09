"""pySMESH mesher — mesh quality controls: measures, predicates and the filter algebra.

Part of the :mod:`pysmesh.mesher` package. Two kinds of thing live here and they answer two
different questions.

A **control** measures one element and returns a number — a cell volume, an aspect ratio, a
skew angle. :func:`quality` evaluates one over every element it applies to and returns two
parallel arrays: the mesh ids, and the values.

A **predicate** answers yes or no about one element or node — is this volume inverted, does
this face sit on a bare border, is this node used by nothing. :func:`select` resolves one to
the ids that satisfy it. Predicates compose: :class:`And`, :class:`Or` and :class:`Not` build
a tree, and :class:`LessThan`, :class:`MoreThan` and :class:`EqualTo` turn any control into a
predicate by comparing it against a margin. That is the filter algebra, and it is what lets a
single call ask for, say, every volume of poor aspect ratio that is not already in a group.

**Why this exists even where an array-side mesh toolkit does.** Every 3-D measure here has no
array-side counterpart, because a streamed surface pipeline has neither volume cells nor the
reverse connectivity these need: a cell volume, a 3-D aspect ratio, an inverted or
over-constrained volume, a bare border. Those are the numbers that decide whether a volume
mesh will run in a solver at all.

**Where a control can run.** Both functions take a :class:`~pysmesh.MeshData`, so a mesh read
from a file can be measured with no mesher behind it. Two of the entries read the geometry
instead of the mesh — :class:`Deflection2D` and :class:`ElementsOnShape` — and one reads a
group; those work only through :meth:`~pysmesh.Mesher.quality` and
:meth:`~pysmesh.Mesher.select` on a live mesher, and say so rather than returning a number
computed from nothing.

Each control's definition is stated on its class, because a quality metric without its
definition is a number no one can act on. The formulas are SMESH's own, which follow Frey and
George, *Maillages, applications aux elements finis* (Hermes Science, 1999).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from .._core import mesh_quality as _mesh_quality
from .._core import mesh_select as _mesh_select
from ._base import _MesherBase
from ._types import ElementDimension, MeshData, SubShape, _Spec

# ---- The two families ------------------------------------------------------------------ #


@dataclass(frozen=True)
class Control(_Spec):
    """A quality measure: one number per element.

    Every control names the element family it applies to. An element of another family is
    not measured at all, and neither is one the measure is undefined on — a warping needs
    four nodes, an aspect ratio needs a cell whose shape its formula knows. Those are
    reported as :attr:`QualityResult.skipped` rather than given a value that would read as a
    perfect element.
    """


@dataclass(frozen=True)
class Predicate(_Spec):
    """A yes/no test over one mesh entity, and the building block of the filter algebra."""


# ---- Measures -------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Volume(Control):
    """Signed volume of a 3-D cell.

    Negative means the cell is inverted, which is the single most common reason a volume mesh
    is rejected by a solver. Applies to volumes.
    """

    native_name: ClassVar[str] = "Volume"


@dataclass(frozen=True)
class Area(Control):
    """Area of a 2-D element, as the magnitude of its summed triangle-fan cross products.

    Applies to faces, including polygons.
    """

    native_name: ClassVar[str] = "Area"


@dataclass(frozen=True)
class Length(Control):
    """Length of a 1-D element: the straight distance for a linear edge, and the sum of the
    two half spans for a quadratic one. Applies to edges.
    """

    native_name: ClassVar[str] = "Length"


@dataclass(frozen=True)
class AspectRatio(Control):
    """Aspect ratio of a 2-D element, normalised so that a regular element is exactly 1.

    For a triangle it is ``sqrt(3)/6 * h * p / S`` with *h* the longest side, *p* the half
    perimeter and *S* the area. For a quadrangle it is ``sqrt(1/32) * L * C1 / C2`` with *L*
    the longest of the four sides and two diagonals, *C1* the root sum of the squared sides,
    and *C2* the smallest of the four triangles three of its nodes can make. Larger is worse;
    a degenerate element gives infinity. Applies to faces, but not to polygons.
    """

    native_name: ClassVar[str] = "AspectRatio"


@dataclass(frozen=True)
class AspectRatio3D(Control):
    """Aspect ratio of a 3-D cell, normalised so that a regular cell is 1.

    A tetrahedron takes VTK's own ``TetAspectRatio``, so the value matches what a
    ParaView-side check would report. A hexahedron takes the HOMARD measure: the cell is cut
    into 24 corner tetrahedra between an edge, its face centre and the cell centre, the worst
    of those is taken, and the result is scaled so a cube reads 1. A pyramid and a prism take
    the worst of the tetrahedra their corners span. Larger is worse. Applies to volumes, but
    not to polyhedra.
    """

    native_name: ClassVar[str] = "AspectRatio3D"


@dataclass(frozen=True)
class Warping(Control):
    """How far a quadrangle departs from being planar, in degrees.

    Computed from the four corner triangles against the element's centroid; the largest of
    the four is reported. **Values below 0.1 degrees are reported as exactly 0** — upstream
    snaps them, so a nearly planar face reads as planar. Applies to four-node faces only.
    """

    native_name: ClassVar[str] = "Warping"


@dataclass(frozen=True)
class Taper(Control):
    """How unequal a quadrangle's four corner triangles are, in the range ``[0, 1]``.

    Each of the four triangles three of its nodes make is compared against their mean area,
    and the largest relative departure is reported. 0 is a parallelogram. **Values below 0.01
    are reported as exactly 0.** Applies to four-node faces only.
    """

    native_name: ClassVar[str] = "Taper"


@dataclass(frozen=True)
class Skew(Control):
    """How far a face departs from having right angles, in degrees.

    For a quadrangle it is the departure from 90 degrees of the angle between the two lines
    joining opposite edge midpoints. For a triangle it is the largest such departure over the
    three ways of taking one median against the midline of the two other sides — **not** the
    departure of an interior angle, which reads differently. **On a quadrangle, values below
    0.1 degrees are reported as exactly 0.** Applies to faces of three or four nodes.
    """

    native_name: ClassVar[str] = "Skew"


@dataclass(frozen=True)
class MinimumAngle(Control):
    """The smallest interior angle of a face, in degrees. Smaller is worse."""

    native_name: ClassVar[str] = "MinimumAngle"


@dataclass(frozen=True)
class Length2D(Control):
    """The shortest edge of a 2-D element. Applies to faces."""

    native_name: ClassVar[str] = "Length2D"


@dataclass(frozen=True)
class Length3D(Control):
    """The shortest edge of a 3-D cell. Applies to volumes."""

    native_name: ClassVar[str] = "Length3D"


@dataclass(frozen=True)
class Deflection2D(Control):
    """How far a face's centre sits from the CAD surface it lies on.

    This is the one measure that says whether a surface mesh actually follows the geometry,
    rather than whether its elements are well shaped. It reads the sub-shape each face is
    bound to, so it needs a live :class:`~pysmesh.Mesher`: on a mesh handed in as arrays it
    is refused rather than answered. A face bound to nothing reads 0. Applies to faces.
    """

    native_name: ClassVar[str] = "Deflection2D"


@dataclass(frozen=True)
class MaxElementLength2D(Control):
    """The longest span of a 2-D element — its longest side or diagonal. Applies to faces."""

    native_name: ClassVar[str] = "MaxElementLength2D"


@dataclass(frozen=True)
class MaxElementLength3D(Control):
    """The longest span of a 3-D cell, across its edges and diagonals. Applies to volumes."""

    native_name: ClassVar[str] = "MaxElementLength3D"


@dataclass(frozen=True)
class MultiConnection(Control):
    """How many elements of higher dimension share a 1-D element.

    Faces **and** volumes are counted, so a mesh edge on the skin of a structured hexahedral
    mesh reads 3: the two surface quadrangles plus the cell behind them. Applies to edges.
    """

    native_name: ClassVar[str] = "MultiConnection"


@dataclass(frozen=True)
class MultiConnection2D(Control):
    """The largest number of elements sharing any one border of a face. Applies to faces."""

    native_name: ClassVar[str] = "MultiConnection2D"


@dataclass(frozen=True)
class NodeConnectivityNumber(Control):
    """How many elements of the mesh's highest dimension use a node. Applies to nodes."""

    native_name: ClassVar[str] = "NodeConnectivityNumber"


# ---- Predicates ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FreeEdges(Predicate):
    """A face with at least one border no other face shares. Applies to faces."""

    native_name: ClassVar[str] = "FreeEdges"


@dataclass(frozen=True)
class FreeBorders(Predicate):
    """A 1-D element bordering one face or none. Applies to edges."""

    native_name: ClassVar[str] = "FreeBorders"


@dataclass(frozen=True)
class FreeNodes(Predicate):
    """A node no element uses. Applies to nodes."""

    native_name: ClassVar[str] = "FreeNodes"


@dataclass(frozen=True)
class FreeFaces(Predicate):
    """A face bounding one volume or none — the skin of a volume mesh. Applies to faces."""

    native_name: ClassVar[str] = "FreeFaces"


@dataclass(frozen=True)
class BadOrientedVolume(Predicate):
    """A cell whose faces wind inward: an inverted element. Applies to volumes."""

    native_name: ClassVar[str] = "BadOrientedVolume"


@dataclass(frozen=True)
class BareBorderFace(Predicate):
    """A face with a border carried by no 1-D element, where one is expected.

    Applies to faces.
    """

    native_name: ClassVar[str] = "BareBorderFace"


@dataclass(frozen=True)
class BareBorderVolume(Predicate):
    """A cell with a facet on the mesh boundary that no face element covers.

    A volume mesh whose skin is incomplete cannot carry a boundary condition there, which
    makes this one of the checks worth running before any solver handoff. Applies to volumes.
    """

    native_name: ClassVar[str] = "BareBorderVolume"


@dataclass(frozen=True)
class OverConstrainedFace(Predicate):
    """A face whose only free border is a whole edge — every node of it fixed. Faces."""

    native_name: ClassVar[str] = "OverConstrainedFace"


@dataclass(frozen=True)
class OverConstrainedVolume(Predicate):
    """A cell with exactly one facet on the boundary and all its nodes on it.

    Such a cell has no interior degree of freedom left, which is a solver problem rather than
    a geometry one. Applies to volumes.
    """

    native_name: ClassVar[str] = "OverConstrainedVolume"


@dataclass(frozen=True)
class CoincidentNodes(Predicate):
    """A node with another within ``tolerance`` of it. Applies to nodes.

    Attributes:
        tolerance: Distance below which two nodes count as one.
    """

    native_name: ClassVar[str] = "CoincidentNodes"

    tolerance: float = 1e-7


@dataclass(frozen=True)
class CoincidentElements(Predicate):
    """An element built on exactly the same nodes as another.

    Attributes:
        element_family: Which family to test — ``EDGE``, ``FACE`` or ``VOLUME``.
    """

    native_name: ClassVar[str] = "CoincidentElements"

    element_family: ElementDimension = ElementDimension.FACE


@dataclass(frozen=True)
class ManifoldPart(Predicate):
    """A face reachable from ``start_element`` across manifold borders only.

    Applies to faces. Selecting with it is how a single manifold shell is separated from a
    mesh that carries several joined at non-manifold borders.

    Attributes:
        angle_tolerance: Largest angle, in radians, between two faces still treated as one
            smooth region. 0 accepts any angle.
        only_manifold: Stop at a non-manifold border rather than crossing it.
        start_element: Mesh id of the face to grow from. 0 lets SMESH pick one.
    """

    native_name: ClassVar[str] = "ManifoldPart"

    angle_tolerance: float = 0.0
    only_manifold: bool = True
    start_element: int = 0


@dataclass(frozen=True)
class RangeOfIds(Predicate):
    """Membership of an explicit set of mesh ids.

    Ids here are the mesh's own, and those are **one global sequence shared by edges, faces
    and volumes** — the first volume of a mesh whose faces were numbered first does not have
    id 1. So build this from ids a harvest or a selection actually returned, never from a
    per-type position.

    Attributes:
        ids: The mesh ids to accept.
        element_family: Which family the ids belong to, which is also what a selection over
            this predicate walks.
    """

    native_name: ClassVar[str] = "RangeOfIds"

    ids: tuple[int, ...]
    element_family: ElementDimension


@dataclass(frozen=True)
class ElementsOnShape(Predicate):
    """An element lying on one sub-shape of the meshed geometry.

    Reads the geometry, so it needs a live :class:`~pysmesh.Mesher`. Unlike a group defined
    on a sub-shape, this classifies by position rather than by what the mesher bound where —
    which is what makes it the right tool on a mesh that was edited after it was computed.

    Attributes:
        on: The sub-shape to test against. ``None`` means the whole shape.
        element_family: Which family to test.
        tolerance: How far off the shape a node may still be.
        all_nodes: Require every node on the shape rather than just the first.
    """

    native_name: ClassVar[str] = "ElementsOnShape"

    element_family: ElementDimension
    on: SubShape | None = None
    tolerance: float = 1e-7
    all_nodes: bool = True


@dataclass(frozen=True)
class BelongToGroup(Predicate):
    """Membership of a named group on the same mesher.

    Attributes:
        group_name: The group to test membership of.
    """

    native_name: ClassVar[str] = "BelongToMeshGroup"

    group_name: str


# ---- The filter algebra ----------------------------------------------------------------- #


@dataclass(frozen=True)
class Not(Predicate):
    """Everything the wrapped predicate rejects.

    Attributes:
        predicate: The predicate to invert.
    """

    native_name: ClassVar[str] = "LogicalNOT"

    predicate: Predicate


@dataclass(frozen=True)
class And(Predicate):
    """Everything both predicates accept.

    Attributes:
        predicate1: The first predicate.
        predicate2: The second predicate.
    """

    native_name: ClassVar[str] = "LogicalAND"

    predicate1: Predicate
    predicate2: Predicate


@dataclass(frozen=True)
class Or(Predicate):
    """Everything either predicate accepts.

    Attributes:
        predicate1: The first predicate.
        predicate2: The second predicate.
    """

    native_name: ClassVar[str] = "LogicalOR"

    predicate1: Predicate
    predicate2: Predicate


@dataclass(frozen=True)
class LessThan(Predicate):
    """An element whose control value is below ``margin``.

    Attributes:
        control: The measure to compare. It also decides which family is walked.
        margin: The threshold.
    """

    native_name: ClassVar[str] = "LessThan"

    control: Control
    margin: float


@dataclass(frozen=True)
class MoreThan(Predicate):
    """An element whose control value is above ``margin``.

    Attributes:
        control: The measure to compare. It also decides which family is walked.
        margin: The threshold.
    """

    native_name: ClassVar[str] = "MoreThan"

    control: Control
    margin: float


@dataclass(frozen=True)
class EqualTo(Predicate):
    """An element whose control value is within ``tolerance`` of ``margin``.

    Attributes:
        control: The measure to compare. It also decides which family is walked.
        margin: The value to match.
        tolerance: How far from it still counts as equal.
    """

    native_name: ClassVar[str] = "EqualTo"

    control: Control
    margin: float
    tolerance: float = 1e-7


# ---- Results ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QualityResult:
    """One control evaluated over a mesh.

    The two arrays are parallel and keyed by mesh id, so ``element_ids[i]`` names the entity
    whose value is ``values[i]``. Pairing them with :attr:`MeshData.element_id` is what
    carries a value back to a cell; pairing that in turn with the element's sub-shape ordinal
    carries it back to the geometry.

    Attributes:
        control: The control's native name.
        family: The element family it was evaluated over.
        element_ids: (K,) int64 — mesh ids, in the mesh's own iteration order.
        values: (K,) float64 — the measure, one per id.
        skipped: How many entities of that family the control does not apply to and were
            therefore not measured at all.
    """

    control: str
    family: ElementDimension
    element_ids: NDArray[np.int64]
    values: NDArray[np.float64]
    skipped: int

    @property
    def count(self) -> int:
        """How many entities were measured."""
        return int(self.values.shape[0])


@dataclass(frozen=True)
class Selection:
    """The entities one predicate accepts.

    Attributes:
        predicate: The predicate's native name.
        family: The element family it was resolved over. ``NODE`` for the node predicates,
            so the ids are node ids rather than element ids.
        ids: (K,) int64 — the mesh ids that satisfy it, ascending in the mesh's own order.
    """

    predicate: str
    family: ElementDimension
    ids: NDArray[np.int64]

    @property
    def count(self) -> int:
        """How many entities satisfy the predicate."""
        return int(self.ids.shape[0])


def _quality_result(raw: dict[str, object]) -> QualityResult:
    """Build a :class:`QualityResult` from the native evaluation."""
    return QualityResult(
        control=cast("str", raw["control"]),
        family=ElementDimension(cast("int", raw["family"])),
        element_ids=cast("NDArray[np.int64]", raw["element_ids"]),
        values=cast("NDArray[np.float64]", raw["values"]),
        skipped=cast("int", raw["skipped"]),
    )


def _selection(raw: dict[str, object]) -> Selection:
    """Build a :class:`Selection` from the native evaluation."""
    return Selection(
        predicate=cast("str", raw["predicate"]),
        family=ElementDimension(cast("int", raw["family"])),
        ids=cast("NDArray[np.int64]", raw["ids"]),
    )


def _payload(mesh: MeshData) -> dict[str, object]:
    """The arrays the native side rebuilds a mesh from."""
    return {
        "node_coords": mesh.node_coords,
        "node_id": mesh.node_id,
        "element_offsets": mesh.element_offsets,
        "element_nodes": mesh.element_nodes,
        "element_type": mesh.element_type,
        "element_id": mesh.element_id,
    }


# ---- Free functions, over a mesh with no mesher behind it -------------------------------- #


def quality(mesh: MeshData, control: Control) -> QualityResult:
    """Evaluate one quality control over a mesh given as arrays.

    Use this on a mesh that has no mesher — one read from an interchange file, or one a
    caller built. The mesh is rebuilt internally with its ids intact, so the result is keyed
    by the same ids the input carries.

    Args:
        mesh: The mesh to measure.
        control: The measure to evaluate.

    Returns:
        The mesh ids and their values.

    Raises:
        PysmeshError: If the control reads the geometry, which arrays do not carry; if the
            mesh holds a polygon or a polyhedron, whose node count does not determine its
            shape and which therefore cannot be rebuilt from these arrays; or if the arrays
            disagree with one another.
    """
    return _quality_result(
        _mesh_quality(_payload(mesh), control.native_name, control.params())
    )


def select(mesh: MeshData, predicate: Predicate) -> Selection:
    """Resolve one predicate over a mesh given as arrays.

    Args:
        mesh: The mesh to test.
        predicate: The test, which may be a composed one.

    Returns:
        The mesh ids that satisfy it.

    Raises:
        PysmeshError: If the predicate reads the geometry or a group, neither of which arrays
            carry; if the mesh holds a polygon or a polyhedron; or if the arrays disagree
            with one another.
    """
    return _selection(
        _mesh_select(_payload(mesh), predicate.native_name, predicate.params())
    )


# ---- The mesher's own ------------------------------------------------------------------- #


class _QualityOps(_MesherBase):
    """Quality controls and selection over a mesher's live mesh."""

    __slots__ = ()

    def quality(self, control: Control) -> QualityResult:
        """Evaluate one quality control over this mesher's mesh.

        Every control works here, including the ones that read the geometry.

        Args:
            control: The measure to evaluate.

        Returns:
            The mesh ids and their values.

        Raises:
            PysmeshError: If the mesher has been released.
        """
        return _quality_result(self._m.quality(control.native_name, control.params()))

    def select(self, predicate: Predicate) -> Selection:
        """Resolve one predicate over this mesher's mesh.

        Args:
            predicate: The test, which may be a composed one.

        Returns:
            The mesh ids that satisfy it.

        Raises:
            PysmeshError: If a named group does not exist, or the mesher has been released.
        """
        return _selection(self._m.select(predicate.native_name, predicate.params()))
