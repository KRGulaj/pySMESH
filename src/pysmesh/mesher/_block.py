"""pySMESH mesher — block decomposition, and pattern mapping onto a face or a block.

Part of the :mod:`pysmesh.mesher` package.

A **block** is a six-faced solid read as a deformed unit cube: every point inside it has a
normalised (x, y, z) coordinate in [0, 1]^3, and the mapping runs both ways. That is the
machinery the structured hexahedral algorithms work through, and on its own it is what lets a
caller place a seed set, a sampling grid or an O-grid inside a solid without meshing it first
— and read back where an arbitrary point sits within one.

Which corner is (0, 0, 0) and which is (0, 0, 1) is the caller's choice, named by two
vertices joined by an edge. Everything else follows from it: :func:`block_shapes` reports the
eight vertices, twelve edges and six faces in the block's own fixed order, so "the face at
z = 1" becomes an ordinal of the caller's own shape.

A **pattern** is a small parametric mesh — points in the unit square or cube plus their
connectivity — mapped onto a face or a block by matching its key points to the corners. It is
how a repeating motif is laid onto geometry an algorithm would otherwise mesh generically,
and a pattern can be read back off a face that is already meshed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from .._core import Shape
from .._core import block_parameters as _block_parameters
from .._core import block_points as _block_points
from .._core import block_shapes as _block_shapes
from ._base import _MesherBase

#: The block's twelve edges, in the order :attr:`Block.edges` reports them. The name says
#: which coordinate varies along the edge and where the other two are pinned: ``Ex00`` runs
#: along x at y = 0, z = 0.
BLOCK_EDGE_NAMES: Final[tuple[str, ...]] = (
    "Ex00",
    "Ex10",
    "Ex01",
    "Ex11",
    "E0y0",
    "E1y0",
    "E0y1",
    "E1y1",
    "E00z",
    "E10z",
    "E01z",
    "E11z",
)

#: The block's eight vertices, in the order :attr:`Block.vertices` reports them. The digits
#: are the corner's own (x, y, z) parameters.
BLOCK_VERTEX_NAMES: Final[tuple[str, ...]] = (
    "V000",
    "V100",
    "V010",
    "V110",
    "V001",
    "V101",
    "V011",
    "V111",
)

#: The block's six faces, in the order :attr:`Block.faces` reports them. The name says which
#: two coordinates vary over the face and where the third is pinned: ``Fxy0`` is the face at
#: z = 0.
BLOCK_FACE_NAMES: Final[tuple[str, ...]] = ("Fxy0", "Fxy1", "Fx0z", "Fx1z", "F0yz", "F1yz")


@dataclass(frozen=True)
class Block:
    """One solid's sub-shapes, numbered in the block's own order.

    Attributes:
        solid: The 1-based SOLID ordinal the block was built on.
        vertices: (8,) int64 — the shape's VERTEX ordinals, in :data:`BLOCK_VERTEX_NAMES`
            order.
        edges: (12,) int64 — the shape's EDGE ordinals, in :data:`BLOCK_EDGE_NAMES` order.
        faces: (6,) int64 — the shape's FACE ordinals, in :data:`BLOCK_FACE_NAMES` order.
    """

    solid: int
    vertices: NDArray[np.int64]
    edges: NDArray[np.int64]
    faces: NDArray[np.int64]

    def face(self, name: str) -> int:
        """The shape's FACE ordinal for one named block face.

        Args:
            name: One of :data:`BLOCK_FACE_NAMES`, for example ``"Fxy1"``.

        Returns:
            The 1-based FACE ordinal.

        Raises:
            KeyError: If the name is not one of the six.
        """
        if name not in BLOCK_FACE_NAMES:
            raise KeyError(
                f"{name!r} is not a block face; expected one of {BLOCK_FACE_NAMES}."
            )
        return int(self.faces[BLOCK_FACE_NAMES.index(name)])

    def edge(self, name: str) -> int:
        """The shape's EDGE ordinal for one named block edge.

        Args:
            name: One of :data:`BLOCK_EDGE_NAMES`, for example ``"E00z"``.

        Returns:
            The 1-based EDGE ordinal.

        Raises:
            KeyError: If the name is not one of the twelve.
        """
        if name not in BLOCK_EDGE_NAMES:
            raise KeyError(
                f"{name!r} is not a block edge; expected one of {BLOCK_EDGE_NAMES}."
            )
        return int(self.edges[BLOCK_EDGE_NAMES.index(name)])

    def vertex(self, name: str) -> int:
        """The shape's VERTEX ordinal for one named block corner.

        Args:
            name: One of :data:`BLOCK_VERTEX_NAMES`, for example ``"V111"``.

        Returns:
            The 1-based VERTEX ordinal.

        Raises:
            KeyError: If the name is not one of the eight.
        """
        if name not in BLOCK_VERTEX_NAMES:
            raise KeyError(
                f"{name!r} is not a block vertex; expected one of {BLOCK_VERTEX_NAMES}."
            )
        return int(self.vertices[BLOCK_VERTEX_NAMES.index(name)])


@dataclass(frozen=True)
class BlockParameters:
    """Where each of a batch of model-space points sits inside a block.

    Inverting the mapping is a numerical search, so it reports how close it got as well as
    where. A point outside the block has no parameters at all, and ``converged`` is what
    says so.

    Attributes:
        parameters: (N, 3) float64 — the normalised coordinates found.
        distances: (N,) float64 — how far the found parameters land from the query point.
        converged: (N,) bool — True where the search reached the requested tolerance.
    """

    parameters: NDArray[np.float64]
    distances: NDArray[np.float64]
    converged: NDArray[np.bool_]


@dataclass(frozen=True)
class PatternReport:
    """What applying one pattern produced.

    Attributes:
        nodes_before: Node count before the pattern was applied.
        nodes_after: Node count after it.
        elements_before: Count of the elements of the pattern's own dimension, before.
        elements_after: The same count after.
    """

    nodes_before: int
    nodes_after: int
    elements_before: int
    elements_after: int


def _points(values: object, name: str) -> NDArray[np.float64]:
    """One (N, 3) float64 batch, checked here so a bad shape names its argument."""
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {array.shape}.")
    return array


def block_shapes(shape: Shape, solid: int, vertex000: int, vertex001: int) -> Block:
    """Read one solid as a block, and report its sub-shapes in the block's own order.

    Args:
        shape: The shape the solid belongs to.
        solid: The 1-based SOLID ordinal. It must be one closed shell of six four-sided
            faces.
        vertex000: The VERTEX ordinal of the corner to treat as (0, 0, 0).
        vertex001: The VERTEX ordinal of the corner to treat as (0, 0, 1). It must be joined
            to ``vertex000`` by one edge of the block.

    Returns:
        The block's eight vertices, twelve edges and six faces, as ordinals of ``shape``.

    Raises:
        PysmeshError: If the ordinals name nothing, if the solid is not a block, or if the
            two vertices are not corners joined by an edge.
    """
    raw = _block_shapes(shape, solid, vertex000, vertex001)
    return Block(
        solid=cast("int", raw["solid"]),
        vertices=cast("NDArray[np.int64]", raw["vertices"]),
        edges=cast("NDArray[np.int64]", raw["edges"]),
        faces=cast("NDArray[np.int64]", raw["faces"]),
    )


def block_points(
    shape: Shape, solid: int, vertex000: int, vertex001: int, parameters: object
) -> NDArray[np.float64]:
    """Place normalised block parameters in model space.

    Args:
        shape: The shape the solid belongs to.
        solid: The 1-based SOLID ordinal.
        vertex000: The VERTEX ordinal of the (0, 0, 0) corner.
        vertex001: The VERTEX ordinal of the (0, 0, 1) corner.
        parameters: (N, 3) float64 — normalised coordinates, each within [0, 1].

    Returns:
        (N, 3) float64 — the model-space positions.

    Raises:
        ValueError: If ``parameters`` is not (N, 3).
        PysmeshError: If a parameter is outside [0, 1], if the solid is not a block, or if
            the block cannot place a row.
    """
    raw = _block_points(
        shape, solid, vertex000, vertex001, _points(parameters, "parameters")
    )
    return cast("NDArray[np.float64]", raw["points"])


def block_parameters(
    shape: Shape,
    solid: int,
    vertex000: int,
    vertex001: int,
    points: object,
    tolerance: float = 1e-6,
) -> BlockParameters:
    """Find where model-space points sit inside a block.

    Args:
        shape: The shape the solid belongs to.
        solid: The 1-based SOLID ordinal.
        vertex000: The VERTEX ordinal of the (0, 0, 0) corner.
        vertex001: The VERTEX ordinal of the (0, 0, 1) corner.
        points: (N, 3) float64 — the query points.
        tolerance: How close the search must land to count as converged.

    Returns:
        The parameters found, how far they land from the query point, and whether the search
        converged.

    Raises:
        ValueError: If ``points`` is not (N, 3).
        PysmeshError: If the tolerance is not positive, or if the solid is not a block.
    """
    raw = _block_parameters(
        shape, solid, vertex000, vertex001, _points(points, "points"), tolerance
    )
    return BlockParameters(
        parameters=cast("NDArray[np.float64]", raw["parameters"]),
        distances=cast("NDArray[np.float64]", raw["distances"]),
        converged=cast("NDArray[np.int64]", raw["converged"]).astype(np.bool_),
    )


class _PatternOps(_MesherBase):
    """Reading a pattern off a meshed face, and mapping one onto a face or a block."""

    __slots__ = ()

    def pattern_from_face(self, face: int, project: bool = False) -> str:
        """Read the mesh on one face back as a pattern.

        Args:
            face: The 1-based FACE ordinal. It must already carry 2-D elements.
            project: Recompute each node's position in the face's parameter space by
                projecting it, rather than reading the parameters the mesher stored.

        Returns:
            The pattern as text, in SMESH's own pattern format. Hand it to
            :meth:`apply_pattern_to_face` on any face with a matching corner count.

        Raises:
            PysmeshError: If the ordinal names no face, if the face carries no elements, if
                the face is closed or too narrow to carry a pattern, or if the mesher has
                been released.
        """
        return str(self._m.pattern_from_face(int(face), project))

    def apply_pattern_to_face(
        self,
        pattern: str,
        face: int,
        vertex: int,
        reverse: bool = False,
        create_polygons: bool = False,
    ) -> PatternReport:
        """Map a 2-D pattern onto one face, creating nodes and elements.

        The pattern's first key point lands on ``vertex`` and the rest follow round the
        face's outer boundary. The pattern's key-point count and the face's vertex count
        have to match.

        Args:
            pattern: The pattern text, from :meth:`pattern_from_face` or written by hand.
            face: The 1-based FACE ordinal to map onto.
            vertex: The 1-based VERTEX ordinal the first key point lands on. It must be on
                the face's outer boundary.
            reverse: Walk the face's boundary the other way round, mirroring the pattern.
            create_polygons: Emit polygons where the pattern's cells do not match a standard
                type.

        Returns:
            The node and face counts either side of the mapping.

        Raises:
            PysmeshError: If the pattern text is malformed, if its key points and the face's
                vertices do not correspond, if the vertex is not on the outer boundary, if
                the mapping does not converge, or if the mesher has been released.
        """
        raw = self._m.apply_pattern_to_face(
            pattern, int(face), int(vertex), reverse, create_polygons
        )
        return PatternReport(
            nodes_before=cast("int", raw["nodes_before"]),
            nodes_after=cast("int", raw["nodes_after"]),
            elements_before=cast("int", raw["faces_before"]),
            elements_after=cast("int", raw["faces_after"]),
        )

    def apply_pattern_to_block(
        self,
        pattern: str,
        solid: int,
        vertex000: int,
        vertex001: int,
        create_polyhedra: bool = False,
    ) -> PatternReport:
        """Map a 3-D pattern into one block, creating nodes and cells.

        Args:
            pattern: The pattern text.
            solid: The 1-based SOLID ordinal. It must be one shell of six four-sided faces.
            vertex000: The VERTEX ordinal of the corner the pattern's (0, 0, 0) key point
                lands on.
            vertex001: The VERTEX ordinal of the corner its (0, 0, 1) key point lands on.
            create_polyhedra: Emit polyhedra where the pattern's cells do not match a
                standard type.

        Returns:
            The node and volume counts either side of the mapping.

        Raises:
            PysmeshError: If the pattern text is malformed, if the solid is not a block, if
                the mapping does not converge, or if the mesher has been released.
        """
        raw = self._m.apply_pattern_to_block(
            pattern, int(solid), int(vertex000), int(vertex001), create_polyhedra
        )
        return PatternReport(
            nodes_before=cast("int", raw["nodes_before"]),
            nodes_after=cast("int", raw["nodes_after"]),
            elements_before=cast("int", raw["volumes_before"]),
            elements_after=cast("int", raw["volumes_after"]),
        )
