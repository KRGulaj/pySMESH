"""pySMESH mesher — the value types the whole meshing surface shares.

Part of the :mod:`pysmesh.mesher` package. Everything here is data: the element and
sub-shape enums, the frozen dataclasses a compute and a harvest return, and the small
helpers that turn a native dict into them. No operation lives in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray


class SubShapeKind(IntEnum):
    """What a mesh entity is bound to, and what an assignment can be scoped to.

    ``NONE`` is a real state rather than an error: a mesh read from a file carries no shape
    at all, and an element on a sub-shape kind the geometry API does not index — a wire, a
    shell — has no ordinal to report.
    """

    NONE = 0
    SOLID = 1
    FACE = 2
    EDGE = 3
    VERTEX = 4


class ElementType(IntEnum):
    """The cell types a mesh can contain.

    These are SMESH's own entity codes, kept unchanged: they are the complete space,
    including the quadratic and polyhedral cases, and a second enum over them would only be
    a translation that can drift out of step.

    Two are worth knowing about before they are met. ``POLYHEDRON`` appears wherever the
    body-fitted Cartesian mesher cuts a cell against the geometry, and it is the one type
    whose node list is a *face stream* rather than a flat corner list — see
    :attr:`MeshData.face_sizes`. ``POLYGON`` is its 2-D counterpart.
    """

    NODE = 0
    ELEM_0D = 1
    EDGE = 2
    QUAD_EDGE = 3
    TRIANGLE = 4
    QUAD_TRIANGLE = 5
    BIQUAD_TRIANGLE = 6
    QUADRANGLE = 7
    QUAD_QUADRANGLE = 8
    BIQUAD_QUADRANGLE = 9
    POLYGON = 10
    QUAD_POLYGON = 11
    TETRAHEDRON = 12
    QUAD_TETRAHEDRON = 13
    PYRAMID = 14
    QUAD_PYRAMID = 15
    HEXAHEDRON = 16
    QUAD_HEXAHEDRON = 17
    TRIQUAD_HEXAHEDRON = 18
    PENTAHEDRON = 19
    QUAD_PENTAHEDRON = 20
    BIQUAD_PENTAHEDRON = 21
    HEXAGONAL_PRISM = 22
    POLYHEDRON = 23
    QUAD_POLYHEDRON = 24
    BALL = 25


class ElementDimension(IntEnum):
    """The dimension family a group is defined over, as SMESH counts them."""

    ALL = 0
    NODE = 1
    EDGE = 2
    FACE = 3
    VOLUME = 4
    ELEM_0D = 5
    BALL = 6


#: Element types the Inria ``.mesh`` / ``.meshb`` format can represent. Everything else —
#: every polygon and polyhedron, the quadratic pyramid and prism, the hexagonal prism, balls
#: and 0-D elements — has no keyword in the format at all, so writing one is refused rather
#: than producing a file that is quietly missing cells.
GMF_WRITABLE_TYPES: Final[frozenset[ElementType]] = frozenset(
    {
        ElementType.EDGE,
        ElementType.QUAD_EDGE,
        ElementType.TRIANGLE,
        ElementType.QUAD_TRIANGLE,
        ElementType.BIQUAD_TRIANGLE,
        ElementType.QUADRANGLE,
        ElementType.QUAD_QUADRANGLE,
        ElementType.BIQUAD_QUADRANGLE,
        ElementType.TETRAHEDRON,
        ElementType.QUAD_TETRAHEDRON,
        ElementType.PYRAMID,
        ElementType.HEXAHEDRON,
        ElementType.QUAD_HEXAHEDRON,
        ElementType.TRIQUAD_HEXAHEDRON,
        ElementType.PENTAHEDRON,
    }
)

#: The marker SMESH's GMF driver keys a writable group on. A group whose name does not
#: carry it is not written by the format at all.
GMF_REQUIRED_MARKER: Final[str] = "_required_"


@dataclass(frozen=True)
class SubShape:
    """One sub-shape of the meshed shape, named the way the geometry API names one.

    An ordinal is the 1-based rank of the sub-shape in a per-kind traversal — the same
    number :func:`pysmesh.load_brep` returns — not a mesh id and not the session's entity
    identity. Pairing it with a handoff's per-kind id array is what carries a mesh entity
    back to the model it came from.

    Attributes:
        kind: Which per-kind traversal the ordinal indexes.
        ordinal: 1-based rank within that traversal.
    """

    kind: SubShapeKind
    ordinal: int

    def __post_init__(self) -> None:
        if self.kind is SubShapeKind.NONE:
            raise ValueError("SubShape.kind must name a kind, not NONE.")
        if self.ordinal < 1:
            raise ValueError(f"SubShape.ordinal is 1-based; got {self.ordinal}.")


@dataclass(frozen=True)
class MeshGroup:
    """A named set of elements carried on the mesh.

    Attributes:
        name: The group's stored name.
        dimension: The element family the group is defined over.
        element_ids: (K,) int64 — the mesh ids of its members.
    """

    name: str
    dimension: ElementDimension
    element_ids: NDArray[np.int64]


@dataclass(frozen=True)
class MeshData:
    """A mesh as arrays: nodes, a compressed element list, and the CAD binding of both.

    The element list is compressed rather than one array per cell type, because a volume mesh
    mixes types by construction — the body-fitted Cartesian mesher emits hexahedra inside and
    polyhedra at every cut cell — and because a compressed list carries the quadratic and
    polyhedral cases with no special form: a cell's node count is the span between two
    offsets. Elements appear grouped by ascending dimension, so a consumer wanting only the
    volume cells reads one contiguous span.

    Polyhedra need one thing more. Their node list *is* a face stream: the nodes of face 1,
    then of face 2, and so on, with a node repeated once per face that uses it.
    ``face_sizes`` carries those per-face counts and ``face_offsets`` says where each
    element's share of them begins. Every other type contributes an empty span, because its
    faces follow from its type.

    Attributes:
        node_coords: (N, 3) float64 — model-space position of every node.
        node_id: (N,) int64 — the mesh's own node ids, in the same row order.
        node_kind: (N,) int8 — :class:`SubShapeKind` of the sub-shape each node sits on.
        node_ordinal: (N,) int32 — its 1-based ordinal, or 0 when bound to nothing.
        element_offsets: (M+1,) int64 — element *m* owns
            ``element_nodes[element_offsets[m]:element_offsets[m + 1]]``.
        element_nodes: (K,) int32 — connectivity as **row indices** into ``node_coords``,
            not as node ids, so a consumer indexes straight in with no lookup.
        element_type: (M,) int8 — :class:`ElementType` per element.
        element_id: (M,) int64 — the mesh's own element ids.
        element_kind: (M,) int8 — :class:`SubShapeKind` of the sub-shape each element sits
            on. This is the CAD binding: it is what makes a volume cell say which solid it
            fills and a triangle say which face it lies on.
        element_ordinal: (M,) int32 — its 1-based ordinal, or 0 when bound to nothing.
        face_offsets: (M+1,) int64 — element *m*'s share of ``face_sizes``. An empty span
            for every non-polyhedral element.
        face_sizes: (Q,) int32 — nodes per face, for the polyhedra only.
    """

    node_coords: NDArray[np.float64]
    node_id: NDArray[np.int64]
    node_kind: NDArray[np.int8]
    node_ordinal: NDArray[np.int32]
    element_offsets: NDArray[np.int64]
    element_nodes: NDArray[np.int32]
    element_type: NDArray[np.int8]
    element_id: NDArray[np.int64]
    element_kind: NDArray[np.int8]
    element_ordinal: NDArray[np.int32]
    face_offsets: NDArray[np.int64]
    face_sizes: NDArray[np.int32]

    @property
    def node_count(self) -> int:
        """Number of nodes."""
        return int(self.node_coords.shape[0])

    @property
    def element_count(self) -> int:
        """Number of elements of every dimension together."""
        return int(self.element_type.shape[0])

    def nodes_of(self, element: int) -> NDArray[np.int32]:
        """Row indices of one element's nodes.

        Args:
            element: 0-based position in the element arrays.

        Returns:
            The element's connectivity as row indices into ``node_coords``. For a polyhedron
            this is its face stream, which repeats a node once per face that uses it.

        Raises:
            IndexError: If ``element`` is out of range.
        """
        if not 0 <= element < self.element_count:
            raise IndexError(
                f"element {element} is out of range (the mesh has {self.element_count})."
            )
        start = int(self.element_offsets[element])
        end = int(self.element_offsets[element + 1])
        return self.element_nodes[start:end]

    def face_sizes_of(self, element: int) -> NDArray[np.int32]:
        """Per-face node counts of one polyhedron, empty for any other element type.

        Args:
            element: 0-based position in the element arrays.

        Returns:
            The element's per-face node counts. Empty unless it is a polyhedron.

        Raises:
            IndexError: If ``element`` is out of range.
        """
        if not 0 <= element < self.element_count:
            raise IndexError(
                f"element {element} is out of range (the mesh has {self.element_count})."
            )
        start = int(self.face_offsets[element])
        end = int(self.face_offsets[element + 1])
        return self.face_sizes[start:end]

    def count_of(self, element_type: ElementType) -> int:
        """How many elements of one type the mesh holds.

        Args:
            element_type: The type to count.

        Returns:
            The number of elements of that type.
        """
        return int(np.count_nonzero(self.element_type == int(element_type)))


@dataclass(frozen=True)
class SubMeshCount:
    """How many elements one sub-shape received.

    Attributes:
        kind: Which per-kind traversal the ordinal indexes.
        ordinal: 1-based rank within that traversal.
        elements: Number of elements bound to that sub-shape.
    """

    kind: SubShapeKind
    ordinal: int
    elements: int


@dataclass(frozen=True)
class ComputeReport:
    """What one successful compute produced.

    Attributes:
        nodes: Node count of the whole mesh.
        edges: 1-D element count.
        faces: 2-D element count.
        volumes: 3-D element count.
        meshed: One entry per sub-shape that received elements. A caller driving a mixed
            assignment reads this to tell "meshed by the algorithm I put there" from "meshed
            by an enclosing one".
    """

    nodes: int
    edges: int
    faces: int
    volumes: int
    meshed: tuple[SubMeshCount, ...]


def _mesh_data(raw: dict[str, object]) -> MeshData:
    """Build a :class:`MeshData` from the native harvest."""
    return MeshData(
        node_coords=cast("NDArray[np.float64]", raw["node_coords"]),
        node_id=cast("NDArray[np.int64]", raw["node_id"]),
        node_kind=cast("NDArray[np.int8]", raw["node_kind"]),
        node_ordinal=cast("NDArray[np.int32]", raw["node_ordinal"]),
        element_offsets=cast("NDArray[np.int64]", raw["element_offsets"]),
        element_nodes=cast("NDArray[np.int32]", raw["element_nodes"]),
        element_type=cast("NDArray[np.int8]", raw["element_type"]),
        element_id=cast("NDArray[np.int64]", raw["element_id"]),
        element_kind=cast("NDArray[np.int8]", raw["element_kind"]),
        element_ordinal=cast("NDArray[np.int32]", raw["element_ordinal"]),
        face_offsets=cast("NDArray[np.int64]", raw["face_offsets"]),
        face_sizes=cast("NDArray[np.int32]", raw["face_sizes"]),
    )


def _groups(raw: Sequence[object]) -> tuple[MeshGroup, ...]:
    """Build the group tuple from the native (name, dimension, ids) triples."""
    out: list[MeshGroup] = []
    for entry in raw:
        name, dimension, ids = cast("tuple[str, int, NDArray[np.int64]]", entry)
        out.append(
            MeshGroup(
                name=name, dimension=ElementDimension(dimension), element_ids=ids
            )
        )
    return tuple(out)


def _report(raw: dict[str, object]) -> ComputeReport:
    """Build a :class:`ComputeReport` from the native compute result."""
    meshed: list[SubMeshCount] = []
    for entry in cast("Sequence[object]", raw["meshed"]):
        kind, ordinal, count = cast("tuple[str, int, int]", entry)
        meshed.append(
            SubMeshCount(kind=SubShapeKind[kind], ordinal=ordinal, elements=count)
        )
    return ComputeReport(
        nodes=cast("int", raw["nodes"]),
        edges=cast("int", raw["edges"]),
        faces=cast("int", raw["faces"]),
        volumes=cast("int", raw["volumes"]),
        meshed=tuple(meshed),
    )
