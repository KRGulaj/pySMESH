# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""pySMESH mesher — searching a mesh: location, ray casting, classification, slot cutting.

Part of the :mod:`pysmesh.mesher` package. These answer questions *about* a mesh rather than
building or changing one, and three of the answers have no counterpart in a streamed surface
pipeline:

* **Ray casting.** Where does this line meet the mesh, and in what order. A leak test, a
  visibility test and a wrap's ray plan all reduce to it.
* **Point classification.** Is this point inside, outside or on a closed surface of
  triangles. It is the mesh-side counterpart of :func:`~pysmesh.point_in_solid`, which
  answers the same question against a B-rep.
* **Distance to a volume cell.** A surface-only proximity query cannot answer it at all.

**Every query takes a batch of points.** One call builds an octree over the whole mesh, so
asking one point at a time would pay for the tree per question. The arrays go in and come
back parallel to one another.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._base import _MesherBase
from ._types import ElementDimension


class PointState(IntEnum):
    """Where a point lies relative to a closed surface mesh.

    These are OCCT's own classification values, forwarded unchanged, and they are the same
    ones :func:`~pysmesh.point_in_solid` classifies a B-rep against.

    Attributes:
        IN: Strictly inside.
        OUT: Strictly outside.
        ON: On the surface, within the search tolerance.
        UNKNOWN: The mesh is not closed, so there is no inside to be in.
    """

    IN = 0
    OUT = 1
    ON = 2
    UNKNOWN = 3


@dataclass(frozen=True)
class ElementsAtPoints:
    """The elements each of a batch of points falls in or on.

    A point can be in more than one element — on a shared face, in a non-manifold region —
    and in none, so the answer is a compressed list rather than one id per point.

    Attributes:
        offsets: (N+1,) int64 — point *i* owns ``ids[offsets[i]:offsets[i + 1]]``.
        ids: (K,) int64 — the mesh ids, grouped by point.
    """

    offsets: NDArray[np.int64]
    ids: NDArray[np.int64]

    def at(self, point: int) -> NDArray[np.int64]:
        """The element ids one point fell in.

        Args:
            point: 0-based row of the query batch.

        Returns:
            The mesh ids, possibly empty.

        Raises:
            IndexError: If ``point`` is out of range.
        """
        if not 0 <= point < self.offsets.shape[0] - 1:
            raise IndexError(f"point {point} is out of range.")
        return self.ids[int(self.offsets[point]) : int(self.offsets[point + 1])]


@dataclass(frozen=True)
class ClosestElements:
    """The nearest element to each of a batch of points, and how far it is.

    Attributes:
        ids: (N,) int64 — the mesh id of the nearest element, or 0 where the mesh holds
            nothing of the family asked for.
        distances: (N,) float64 — the distance to it, or -1 where there is no element.
        closest_points: (N, 3) float64 — the point on that element nearest the query point.
    """

    ids: NDArray[np.int64]
    distances: NDArray[np.float64]
    closest_points: NDArray[np.float64]


@dataclass(frozen=True)
class ProjectedPoints:
    """Each of a batch of points projected onto the mesh.

    Attributes:
        points: (N, 3) float64 — the projected positions.
        ids: (N,) int64 — the element each point landed on, or 0 where there was none.
    """

    points: NDArray[np.float64]
    ids: NDArray[np.int64]


@dataclass(frozen=True)
class RayHits:
    """Where one ray meets the mesh, in the order it meets it.

    The ray is a half line: it starts at its origin and travels along its direction, so
    every parameter reported is non-negative and a surface behind the origin is not hit.

    Attributes:
        ids: (H,) int64 — the faces hit, nearest first.
        parameters: (H,) float64 — the distance along the direction at each hit. The
            direction is normalised, so this is a length.
        points: (H, 3) float64 — the hit positions.
        candidates: How many faces the broad phase had to test. Larger than ``H``, and the
            gap is the cost of the bounding-box test rather than an error.
        crossings: How many distinct positions along the ray the surface was met at. It is
            below ``count`` wherever the ray struck an edge two faces share, which is one
            crossing and two hits. This is the number a parity or leak test counts.
    """

    ids: NDArray[np.int64]
    parameters: NDArray[np.float64]
    points: NDArray[np.float64]
    candidates: int
    crossings: int

    @property
    def count(self) -> int:
        """How many faces the ray meets."""
        return int(self.ids.shape[0])


@dataclass(frozen=True)
class SharpEdges:
    """The edges of a surface mesh where two faces meet at a steep angle.

    An edge is given as its two end nodes rather than as a 1-D element, because most of them
    are not elements of the mesh at all — they are the creases between faces.

    Attributes:
        node1: (L,) int64 — the first node of each edge.
        node2: (L,) int64 — the second.
        medium: (L,) int64 — the mid-side node for an edge of a quadratic face, 0 otherwise.
    """

    node1: NDArray[np.int64]
    node2: NDArray[np.int64]
    medium: NDArray[np.int64]

    @property
    def count(self) -> int:
        """How many sharp edges were found."""
        return int(self.node1.shape[0])


@dataclass(frozen=True)
class FacePatches:
    """The mesh's faces partitioned into regions bounded by a given set of edges.

    Attributes:
        offsets: (P+1,) int64 — patch *i* owns ``ids[offsets[i]:offsets[i + 1]]``.
        ids: (K,) int64 — the face ids, grouped by patch.
        names: The group each patch was stored as, in patch order. Empty unless
            :meth:`~pysmesh.Mesher.separate_faces_by_edges` was given a ``name_prefix``. A
            stored patch keeps its identity across an edit; a bare index does not.
    """

    offsets: NDArray[np.int64]
    ids: NDArray[np.int64]
    names: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        """How many patches the faces fell into."""
        return int(self.offsets.shape[0]) - 1

    def at(self, patch: int) -> NDArray[np.int64]:
        """The face ids of one patch.

        Args:
            patch: 0-based patch index.

        Returns:
            The mesh ids of its faces.

        Raises:
            IndexError: If ``patch`` is out of range.
        """
        if not 0 <= patch < self.count:
            raise IndexError(f"patch {patch} is out of range (there are {self.count}).")
        return self.ids[int(self.offsets[patch]) : int(self.offsets[patch + 1])]


@dataclass(frozen=True)
class MergeObstruction:
    """Which of one element's nodes a merge must keep apart to leave it valid.

    Attributes:
        nodes: (K,) int64 — the element's connectivity as it would be after the merge, with
            0 where a node would be dropped.
        keep_apart: (M,) int64 — the nodes whose merging is what would invalidate it. Empty
            when the element survives the merge intact.
    """

    nodes: NDArray[np.int64]
    keep_apart: NDArray[np.int64]


@dataclass(frozen=True)
class SlotBoundary:
    """The edges bounding a slot cut into a triangle mesh.

    Attributes:
        node1: (L,) int64 — the first node of each boundary edge.
        node2: (L,) int64 — the second.
    """

    node1: NDArray[np.int64]
    node2: NDArray[np.int64]


def _points(values: object, name: str) -> NDArray[np.float64]:
    """One (N, 3) float64 batch, checked here so a bad shape names its argument."""
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {array.shape}.")
    return array


class _SearchOps(_MesherBase):
    """Location, ray casting, classification and the array-free geometry operations."""

    __slots__ = ()

    # ---- Location ------------------------------------------------------------------------ #

    def find_elements_by_point(
        self,
        points: object,
        family: ElementDimension = ElementDimension.ALL,
    ) -> ElementsAtPoints:
        """Find the elements each point falls in or on.

        Args:
            points: (N, 3) float64 — the query points.
            family: Which family to look in. ``ALL`` means every element of any dimension,
                excluding nodes and 0-D elements.

        Returns:
            The element ids, grouped by query point.

        Raises:
            ValueError: If ``points`` is not (N, 3).
            PysmeshError: If the family is unknown, or the mesher has been released.
        """
        raw = self._m.find_elements_by_point(
            _points(points, "points"), int(family)
        )
        return ElementsAtPoints(
            offsets=cast("NDArray[np.int64]", raw["offsets"]),
            ids=cast("NDArray[np.int64]", raw["ids"]),
        )

    def find_closest(
        self,
        points: object,
        family: ElementDimension = ElementDimension.ALL,
    ) -> NDArray[np.int64]:
        """Find the element nearest each point.

        Args:
            points: (N, 3) float64 — the query points.
            family: Which family to look in.

        Returns:
            (N,) int64 — the mesh id of the nearest element, or 0 where the mesh holds
            nothing of that family.

        Raises:
            ValueError: If ``points`` is not (N, 3).
            PysmeshError: If the family is unknown, or the mesher has been released.
        """
        raw = self._m.find_closest(_points(points, "points"), int(family))
        return cast("NDArray[np.int64]", raw["ids"])

    def closest_distance(
        self,
        points: object,
        family: ElementDimension = ElementDimension.ALL,
    ) -> ClosestElements:
        """Measure from each point to the nearest element, and say where it lands.

        With ``family=VOLUME`` this answers the distance from a point to a **volume cell**,
        which a surface-only proximity query cannot.

        Args:
            points: (N, 3) float64 — the query points.
            family: Which family to measure against.

        Returns:
            The nearest element, the distance to it, and the point on it.

        Raises:
            ValueError: If ``points`` is not (N, 3).
            PysmeshError: If the family is unknown, or the mesher has been released.
        """
        raw = self._m.closest_distance(_points(points, "points"), int(family))
        return ClosestElements(
            ids=cast("NDArray[np.int64]", raw["ids"]),
            distances=cast("NDArray[np.float64]", raw["distances"]),
            closest_points=cast("NDArray[np.float64]", raw["closest_points"]),
        )

    def project_points(
        self,
        points: object,
        family: ElementDimension = ElementDimension.FACE,
    ) -> ProjectedPoints:
        """Project each point onto the mesh.

        Args:
            points: (N, 3) float64 — the query points.
            family: Which family to project onto.

        Returns:
            The projected positions and the element each landed on.

        Raises:
            ValueError: If ``points`` is not (N, 3).
            PysmeshError: If the family is unknown, or the mesher has been released.
        """
        raw = self._m.project_points(_points(points, "points"), int(family))
        return ProjectedPoints(
            points=cast("NDArray[np.float64]", raw["points"]),
            ids=cast("NDArray[np.int64]", raw["ids"]),
        )

    def point_state(self, points: object) -> NDArray[np.int64]:
        """Classify each point against the mesh's closed surface.

        This is the mesh-side counterpart of :func:`~pysmesh.point_in_solid`. It needs the
        mesh's faces to form a closed surface; where they do not, every point reads
        ``UNKNOWN`` rather than a guess.

        Args:
            points: (N, 3) float64 — the query points.

        Returns:
            (N,) int64 — one :class:`PointState` per point.

        Raises:
            ValueError: If ``points`` is not (N, 3).
            PysmeshError: If the mesher has been released.
        """
        raw = self._m.point_state(_points(points, "points"))
        return cast("NDArray[np.int64]", raw["states"])

    # ---- Region queries ------------------------------------------------------------------- #

    def elements_in_sphere(
        self,
        centre: tuple[float, float, float],
        radius: float,
        family: ElementDimension = ElementDimension.ALL,
    ) -> NDArray[np.int64]:
        """Find the elements whose bounding box meets a sphere.

        This is a broad-phase query: an element is reported when its *bounding box* reaches
        the sphere, so the answer is a candidate set rather than an exact one.

        Args:
            centre: The sphere's centre.
            radius: Its radius.
            family: Which family to look in.

        Returns:
            (K,) int64 — the element ids.

        Raises:
            PysmeshError: If the radius is not positive, if the family is unknown, or if the
                mesher has been released.
        """
        raw = self._m.elements_in_sphere(list(centre), radius, int(family))
        return cast("NDArray[np.int64]", raw["ids"])

    def elements_in_box(
        self,
        minimum: tuple[float, float, float],
        maximum: tuple[float, float, float],
        family: ElementDimension = ElementDimension.ALL,
    ) -> NDArray[np.int64]:
        """Find the elements whose bounding box meets an axis-aligned box.

        Broad phase, as :meth:`elements_in_sphere` is.

        Args:
            minimum: The box's lower corner.
            maximum: Its upper corner.
            family: Which family to look in.

        Returns:
            (K,) int64 — the element ids.

        Raises:
            PysmeshError: If the box is inverted on any axis, if the family is unknown, or
                if the mesher has been released.
        """
        raw = self._m.elements_in_box(list(minimum), list(maximum), int(family))
        return cast("NDArray[np.int64]", raw["ids"])

    # ---- Ray casting ---------------------------------------------------------------------- #

    def elements_near_line(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        family: ElementDimension = ElementDimension.FACE,
    ) -> NDArray[np.int64]:
        """Find the elements whose bounding box an infinite line crosses.

        This is the broad phase of a ray cast, exposed under its own name so it is not
        mistaken for a hit list: it answers about *bounding boxes*, and about a line that
        extends both ways from the origin. :meth:`ray_hits` is the query that answers where
        a ray actually meets the mesh.

        Args:
            origin: A point on the line.
            direction: Its direction.
            family: Which family to look in.

        Returns:
            (K,) int64 — the element ids, in no particular order.

        Raises:
            PysmeshError: If the direction is the zero vector, if the family is unknown, or
                if the mesher has been released.
        """
        raw = self._m.elements_near_line(list(origin), list(direction), int(family))
        return cast("NDArray[np.int64]", raw["ids"])

    def ray_hits(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        tolerance: float = 1e-9,
    ) -> RayHits:
        """Cast a ray at the mesh's faces and return every face it meets, nearest first.

        The ray is a **half line**: it starts at ``origin`` and travels along ``direction``,
        so a surface behind the origin is not hit. Each face is reported at most once,
        however many triangles it was decomposed into, so a ray passing through a
        quadrangle's own diagonal counts once.

        Args:
            origin: Where the ray starts.
            direction: Which way it goes. It is normalised, so the parameters reported are
                distances.
            tolerance: How far outside a triangle a hit still counts, in barycentric units.
                It exists so that a ray striking exactly along a shared edge is not lost by
                both triangles that own it; the cost of raising it is a double count there.

        Returns:
            The faces hit, their distances along the ray, and the hit points.

        Raises:
            PysmeshError: If the direction is the zero vector, if the tolerance is negative
                or not a number, or if the mesher has been released.
        """
        raw = self._m.ray_hits(list(origin), list(direction), tolerance)
        return RayHits(
            ids=cast("NDArray[np.int64]", raw["ids"]),
            parameters=cast("NDArray[np.float64]", raw["parameters"]),
            points=cast("NDArray[np.float64]", raw["points"]),
            candidates=cast("int", raw["candidates"]),
            crossings=cast("int", raw["crossings"]),
        )

    # ---- Feature edges and patches -------------------------------------------------------- #

    def sharp_edges(self, angle: float = 45.0, add_existing: bool = False) -> SharpEdges:
        """Find the creases of a surface mesh, and its non-manifold edges.

        Args:
            angle: The smallest angle between two faces' normals, in **degrees**, that
                counts as a crease.
            add_existing: Also report the mesh's own 1-D elements as edges, whatever angle
                they sit at.

        Returns:
            The edges, as node pairs.

        Raises:
            PysmeshError: If the angle is outside 0 to 180 degrees, or the mesher has been
                released.
        """
        raw = self._m.sharp_edges(angle, add_existing)
        return SharpEdges(
            node1=cast("NDArray[np.int64]", raw["node1"]),
            node2=cast("NDArray[np.int64]", raw["node2"]),
            medium=cast("NDArray[np.int64]", raw["medium"]),
        )

    def separate_faces_by_edges(
        self, edges: SharpEdges, name_prefix: str | None = None
    ) -> FacePatches:
        """Partition the mesh's faces into regions bounded by the given edges.

        Paired with :meth:`sharp_edges` this is how a surface mesh with no CAD behind it is
        broken into the faces it came from — the substrate for naming, selecting and
        assigning boundary conditions on an imported mesh. On a mesher built without a shape
        the pair replaces the sub-shape ordinals entirely: the patches are the only division
        of the surface there is, and the face ids they carry are what a viewport picks and
        hides on.

        **A patch index is not stable on its own.** Each call re-derives the partition from
        scratch, so once faces have been deleted the same region can come back under a
        different index. Passing ``name_prefix`` stores each patch as an explicit group named
        ``f"{name_prefix}{i}"``, and *that* is stable across any edit: SMESH drops a deleted
        element from the group it was in and never renumbers a survivor. Read the groups back
        with :meth:`~pysmesh.Mesher.groups`.

        Args:
            edges: The edges to cut at, from :meth:`sharp_edges` or built by hand.
            name_prefix: Store the patches as groups under this prefix. ``None`` stores
                nothing, because a group is state and a caller only reading the partition
                should not be made to own any.

        Returns:
            The faces grouped by patch, and the group names when they were stored.

        Raises:
            PysmeshError: If an edge names a node the mesh does not have, if the three
                arrays differ in length, if a group of that name already exists — names
                address a group, so they have to be unique — or if the mesher has been
                released.
        """
        raw = self._m.separate_faces_by_edges(
            edges.node1.tolist(),
            edges.node2.tolist(),
            edges.medium.tolist(),
            "" if name_prefix is None else name_prefix,
        )
        return FacePatches(
            offsets=cast("NDArray[np.int64]", raw["offsets"]),
            ids=cast("NDArray[np.int64]", raw["ids"]),
            names=tuple(cast("Sequence[str]", raw["names"])),
        )

    # ---- Merge diagnosis and slot cutting ------------------------------------------------- #

    def merge_obstruction(
        self, element: int, groups: Iterable[Iterable[int]]
    ) -> MergeObstruction:
        """Find which nodes a proposed merge must keep apart to leave one element valid.

        Merging coincident nodes can leave a cell self-intersecting or folded rather than
        simply degenerate, and a degenerate cell is deleted while a folded one survives and
        breaks the solver. This asks the question *before* the merge: given the sets that
        would be collapsed, which of this element's nodes cause the damage.

        Args:
            element: The element to examine.
            groups: The node sets the merge would collapse, as
                :meth:`~pysmesh.Mesher.find_coincident_nodes` returns them. The first id of
                each set survives.

        Returns:
            The element's connectivity as it would be after the merge, and the nodes that
            must not be merged. An empty ``keep_apart`` means the element survives it.

        Raises:
            PysmeshError: If the id names nothing in the mesh, if a set names fewer than two
                nodes, or if the mesher has been released.
        """
        raw = self._m.de_merge(int(element), [[int(i) for i in group] for group in groups])
        return MergeObstruction(
            nodes=cast("NDArray[np.int64]", raw["nodes"]),
            keep_apart=cast("NDArray[np.int64]", raw["keep_apart"]),
        )

    def make_slot(self, width: float, segments: Iterable[int] = ()) -> SlotBoundary:
        """Cut a slot of the given width around 1-D elements lying on a triangle mesh.

        The slot is made by cutting the faces with a cylinder around each segment, so the
        mesh gains a band of the given width along the segment chain and the faces inside it
        are removed. This edits the mesh.

        Args:
            width: How wide the slot is, in model units.
            segments: The 1-D elements to cut along. Empty means every 1-D element of the
                mesh.

        Returns:
            The edges bounding the slot.

        Raises:
            PysmeshError: If the width is not positive, if an id does not name a 1-D element
                of this mesh, or if the mesher has been released.
        """
        raw = self._m.make_slot(width, [int(i) for i in segments])
        return SlotBoundary(
            node1=cast("NDArray[np.int64]", raw["node1"]),
            node2=cast("NDArray[np.int64]", raw["node2"]),
        )
