# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""pySMESH mesher — the mesh editor.

Part of the :mod:`pysmesh.mesher` package. These operations change a mesh after it has been
computed. They exist because several things a solver needs cannot be asked of a mesher at
all: a second-order mesh, an internal wall with zero thickness, a shell oriented by the cells
it bounds, and two separately meshed regions joined into one.

Groups follow every operation here. SMESH rewrites group membership as it edits — an element
that is replaced is replaced in the group, one that is deleted is dropped — so a wall named
on a coarse mesh is still the wall afterwards. That is what makes editing safe to do after
naming rather than before.

**An empty element list means the whole mesh** wherever one is accepted, which is upstream's
own convention. The three operations that would be meaningless over everything —
:meth:`~_EditOps.double_elements`, the two sweeps and :meth:`~_EditOps.reorient` — refuse an
empty list by name instead.

Every operation reports the four element counts either side of itself, because what an edit
did is only readable against what was there before it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._base import _MesherBase
from ._controls import AspectRatio, Control


class SplitMethod(IntEnum):
    """How :meth:`~pysmesh.Mesher.split_volumes` cuts each cell.

    The integer values are SMESH's own; do not reorder.

    Attributes:
        HEXA_TO_5: Each hexahedron into 5 tetrahedra.
        HEXA_TO_6: Each hexahedron into 6 tetrahedra.
        HEXA_TO_24: Each hexahedron into 24 tetrahedra, through its centre.
        HEXA_TO_2_PRISMS: Each hexahedron into 2 prisms, cutting the facet the normal picks.
        HEXA_TO_4_PRISMS: Each hexahedron into 4 prisms.
    """

    HEXA_TO_5 = 1
    HEXA_TO_6 = 2
    HEXA_TO_24 = 3
    HEXA_TO_2_PRISMS = 4
    HEXA_TO_4_PRISMS = 5


class SmoothMethod(IntEnum):
    """How :meth:`~pysmesh.Mesher.smooth` moves each free node.

    The integer values are SMESH's own; do not reorder.

    Attributes:
        LAPLACIAN: Towards the average of the nodes linked to it. Cheap, and it shrinks a
            convex region slightly.
        CENTROIDAL: Towards the area-weighted average of the centroids of the elements
            around it. Costlier, and it keeps element size more even.
    """

    LAPLACIAN = 0
    CENTROIDAL = 1


@dataclass(frozen=True)
class EditReport:
    """The mesh's element counts either side of one editing operation.

    Attributes:
        nodes_before: Node count before the operation.
        nodes_after: Node count after it.
        edges_before: 1-D element count before.
        edges_after: 1-D element count after.
        faces_before: 2-D element count before.
        faces_after: 2-D element count after.
        volumes_before: 3-D element count before.
        volumes_after: 3-D element count after.
        groups_merged: How many sets of coincident nodes or equal elements were collapsed.
            0 for every operation that merges nothing.
    """

    nodes_before: int
    nodes_after: int
    edges_before: int
    edges_after: int
    faces_before: int
    faces_after: int
    volumes_before: int
    volumes_after: int
    groups_merged: int

    @property
    def elements_before(self) -> int:
        """Edges, faces and volumes together, before the operation."""
        return self.edges_before + self.faces_before + self.volumes_before

    @property
    def elements_after(self) -> int:
        """Edges, faces and volumes together, after the operation."""
        return self.edges_after + self.faces_after + self.volumes_after


def _report(raw: dict[str, object]) -> EditReport:
    """Build an :class:`EditReport` from the native counts."""
    return EditReport(
        nodes_before=cast("int", raw["nodes_before"]),
        nodes_after=cast("int", raw["nodes_after"]),
        edges_before=cast("int", raw["edges_before"]),
        edges_after=cast("int", raw["edges_after"]),
        faces_before=cast("int", raw["faces_before"]),
        faces_after=cast("int", raw["faces_after"]),
        volumes_before=cast("int", raw["volumes_before"]),
        volumes_after=cast("int", raw["volumes_after"]),
        groups_merged=cast("int", raw["groups_merged"]),
    )


@dataclass(frozen=True)
class RemovalReport(EditReport):
    """What one removal took away, entity by entity.

    The two id arrays are the point of this type. A count says how much went; a caller
    holding named selections, a viewport highlight or a per-element field has to remap it
    against the survivors, and only the ids it actually lost let it do that.

    They are read back from the mesh rather than echoed from the request, because the two
    never quite match: an id listed twice is removed once, and a removal takes entities
    nobody named — every element built on a removed node, and the nodes the free-node sweep
    finds carrying nothing.

    Attributes:
        elements: (K,) int64 — the element ids that are gone, ascending.
        nodes: (J,) int64 — the node ids that are gone, ascending.
    """

    elements: NDArray[np.int64]
    nodes: NDArray[np.int64]


def _removal(raw: dict[str, object]) -> RemovalReport:
    """Build a :class:`RemovalReport` from the native counts and id arrays."""
    return RemovalReport(
        nodes_before=cast("int", raw["nodes_before"]),
        nodes_after=cast("int", raw["nodes_after"]),
        edges_before=cast("int", raw["edges_before"]),
        edges_after=cast("int", raw["edges_after"]),
        faces_before=cast("int", raw["faces_before"]),
        faces_after=cast("int", raw["faces_after"]),
        volumes_before=cast("int", raw["volumes_before"]),
        volumes_after=cast("int", raw["volumes_after"]),
        groups_merged=cast("int", raw["groups_merged"]),
        elements=cast("NDArray[np.int64]", raw["elements"]),
        nodes=cast("NDArray[np.int64]", raw["nodes"]),
    )


def _ids(values: Iterable[int]) -> list[int]:
    """One id list, as the native layer reads it."""
    return [int(value) for value in values]


def _groups(raw: Sequence[object]) -> tuple[NDArray[np.int64], ...]:
    """The native list of id arrays, as a tuple."""
    return tuple(cast("NDArray[np.int64]", entry) for entry in raw)


class _EditOps(_MesherBase):
    """Everything that changes a mesh after it has been computed."""

    __slots__ = ()

    # ---- Element order ----------------------------------------------------------------- #

    def convert_to_quadratic(
        self, force_3d: bool = True, bi_quadratic: bool = False
    ) -> None:
        """Convert the whole mesh to second order, in place.

        This is the only path in this stack to a P2 mesh, which several solvers want.

        Element ids are preserved — each linear cell becomes the quadratic cell of the same
        id — so a group of elements is unchanged by this. Nodes are added, so a group of
        nodes still names the same nodes and does not gain the new ones.

        Args:
            force_3d: Put each medium node at the midpoint of the straight segment joining
                its two corners. False places it on the CAD edge or surface the element was
                built from instead, which is what makes a curved second-order mesh actually
                follow the geometry — at the cost of a mesh whose medium nodes move when the
                geometry does.
            bi_quadratic: Also add the face-centre and cell-centre nodes, giving the
                bi-quadratic and tri-quadratic forms rather than the plain quadratic ones.

        Raises:
            PysmeshError: If the mesher has been released.
        """
        self._m.convert_to_quadratic(force_3d, bi_quadratic)

    def convert_from_quadratic(self) -> bool:
        """Convert the whole mesh back to first order, in place.

        Returns:
            What SMESH reports, which is **always True** — the operation walks the mesh and
            drops medium nodes wherever it finds them, and reports the same thing whether it
            found any or not. Forwarded rather than reinterpreted, because inventing a
            "nothing to do" answer here would mean this binding deciding what counts as
            quadratic. To tell whether anything changed, compare the node count.

        Raises:
            PysmeshError: If the mesher has been released.
        """
        return bool(self._m.convert_from_quadratic())

    def split_quadratic_into_linear(self, elements: Iterable[int] = ()) -> EditReport:
        """Split bi-quadratic elements into linear ones, adding no nodes.

        A bi-quadratic triangle becomes 3 linear quadrangles, a bi-quadratic quadrangle 4,
        and a tri-quadratic hexahedron 8 linear hexahedra. The existing face-centre and
        cell-centre nodes become corners, which is why no node is created. Quadratic elements
        of lower dimension beside a split one are split too, so the mesh stays conforming.

        Args:
            elements: The elements to split. Empty means the whole mesh.

        Returns:
            The counts either side of the split.

        Raises:
            PysmeshError: If an id names nothing in the mesh, or the mesher has been
                released.
        """
        return _report(self._m.split_quadratic_into_linear(_ids(elements)))

    # ---- Volume splitting --------------------------------------------------------------- #

    def split_volumes(
        self,
        method: SplitMethod = SplitMethod.HEXA_TO_2_PRISMS,
        facet_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> EditReport:
        """Split every volume cell of the mesh.

        This is how a structured hexahedral block is handed to a solver that takes simplices
        only. Groups of volumes follow the split: a cell in a group is replaced by the cells
        it became, and all of them are in the group afterwards.

        Args:
            method: How to cut each cell.
            facet_normal: Which facet of each hexahedron is cut into two triangles, chosen as
                the one this direction points along. Read only by the two prism methods; the
                tetrahedral ones ignore it.

        Returns:
            The counts either side of the split.

        Raises:
            PysmeshError: If the mesh has no volume cells, if ``facet_normal`` is the zero
                vector, or if the mesher has been released.
        """
        raw = self._m.split_volumes(
            int(method), facet_normal[0], facet_normal[1], facet_normal[2]
        )
        return _report(raw)

    # ---- Coincidence and merging --------------------------------------------------------- #

    def find_coincident_nodes(
        self,
        tolerance: float = 1e-7,
        nodes: Iterable[int] = (),
        separate_corners_and_medium: bool = False,
    ) -> tuple[NDArray[np.int64], ...]:
        """Find sets of nodes that lie within ``tolerance`` of one another.

        This is the half of a merge that answers rather than changes, so a caller can look
        at what would collapse before collapsing it.

        Args:
            tolerance: Distance below which two nodes count as one.
            nodes: The nodes to search among. Empty means the whole mesh.
            separate_corners_and_medium: Keep a corner node and a medium node of a quadratic
                element out of the same set, even when they coincide. Merging the two would
                make the element degenerate.

        Returns:
            One int64 array of node ids per set found. The first id of each set is the one
            that would survive a merge.

        Raises:
            PysmeshError: If the tolerance is negative or not a number, if an id names no
                node, or if the mesher has been released.
        """
        return _groups(
            cast(
                "Sequence[object]",
                self._m.find_coincident_nodes(
                    tolerance, _ids(nodes), separate_corners_and_medium
                ),
            )
        )

    def merge_node_groups(
        self, groups: Iterable[Iterable[int]], avoid_making_holes: bool = False
    ) -> EditReport:
        """Collapse each given set of nodes to its first member.

        Args:
            groups: The sets to merge, as returned by :meth:`find_coincident_nodes` or built
                by hand. The first id of each set survives; the rest are replaced by it
                everywhere they are used.
            avoid_making_holes: Where merging would leave an element invalid but not
                degenerate, keep the nodes that caused it apart rather than making the hole.

        Returns:
            The counts either side of the merge.

        Raises:
            PysmeshError: If a set names fewer than two nodes, if an id names no node, or if
                the mesher has been released.
        """
        return _report(
            self._m.merge_node_groups([_ids(group) for group in groups], avoid_making_holes)
        )

    def merge_nodes(self, tolerance: float = 1e-7) -> EditReport:
        """Find nodes within ``tolerance`` of one another and collapse each set to one.

        This is how two separately meshed regions are made to share their interface. An
        element that collapses onto itself is deleted, and every group drops it.

        Args:
            tolerance: Distance below which two nodes count as one. 0 asks SMESH to pick a
                tolerance from the mesh itself.

        Returns:
            The counts either side of the merge, with ``groups_merged`` saying how many sets
            were found.

        Raises:
            PysmeshError: If ``tolerance`` is negative or not a number, or if the mesher has
                been released.
        """
        return _report(self._m.merge_nodes(tolerance))

    def find_equal_elements(
        self, elements: Iterable[int] = ()
    ) -> tuple[NDArray[np.int64], ...]:
        """Find sets of elements built on exactly the same nodes.

        Two elements on the same nodes are a duplicate, which most solvers reject. They
        arise from an imported mesh, from a sweep run twice, and from a merge.

        Args:
            elements: The elements to search among. Empty means the whole mesh.

        Returns:
            One int64 array of element ids per set found.

        Raises:
            PysmeshError: If an id names nothing in the mesh, or the mesher has been
                released.
        """
        return _groups(
            cast("Sequence[object]", self._m.find_equal_elements(_ids(elements)))
        )

    def merge_equal_elements(self) -> EditReport:
        """Remove all but one of every set of elements built on the same nodes.

        Returns:
            The counts either side of the merge, with ``groups_merged`` saying how many
            elements were removed.

        Raises:
            PysmeshError: If the mesher has been released.
        """
        return _report(self._m.merge_equal_elements())

    # ---- Smoothing ------------------------------------------------------------------------ #

    def smooth(
        self,
        method: SmoothMethod = SmoothMethod.LAPLACIAN,
        iterations: int = 1,
        target_aspect_ratio: float = 1.0,
        on_shape: bool = True,
        elements: Iterable[int] = (),
        fixed_nodes: Iterable[int] = (),
    ) -> EditReport:
        """Move the free nodes of a surface mesh to improve its element shapes.

        Nodes on a CAD edge and nodes on a free border are always fixed, so smoothing never
        moves the boundary. With ``on_shape`` it moves the rest **in the parameter space of
        the face each node sits on**, which keeps every node on the CAD surface — the
        property a mesh smoothed as raw coordinates loses immediately on any curved face.

        Args:
            method: Laplacian or centroidal.
            iterations: How many passes to run. Each pass moves every free node once.
            target_aspect_ratio: Stop early once the worst element's aspect ratio is at or
                below this. 1 is a regular element, so the default means "run every pass".
            on_shape: Move nodes in the parameter space of their face rather than in model
                space. Requires the mesh to be bound to geometry; on a mesh with no CAD it
                has no effect.
            elements: The elements whose nodes may move. Empty means the whole mesh.
            fixed_nodes: Extra nodes to hold still, beyond the boundary ones.

        Returns:
            The counts either side. Smoothing creates and deletes nothing, so they match; it
            is the node *positions* that change.

        Raises:
            PysmeshError: If ``iterations`` is below 1, if ``target_aspect_ratio`` is below
                1, if an id names nothing, or if the mesher has been released.
        """
        return _report(
            self._m.smooth(
                int(method),
                iterations,
                target_aspect_ratio,
                on_shape,
                _ids(elements),
                _ids(fixed_nodes),
            )
        )

    # ---- Orientation ---------------------------------------------------------------------- #

    def reorient(self, elements: Iterable[int]) -> int:
        """Reverse the orientation of the named elements.

        Args:
            elements: The elements to reverse.

        Returns:
            How many were reversed.

        Raises:
            PysmeshError: If the list is empty, if an id names nothing, or if the mesher has
                been released.
        """
        raw = self._m.reorient(_ids(elements))
        return cast("int", raw["reoriented"])

    def reorient_2d(
        self,
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        faces: Iterable[int] = (),
        reference_faces: Iterable[int] = (),
        allow_non_manifold: bool = False,
    ) -> int:
        """Make a set of faces consistently oriented with one another.

        Orientation is propagated across shared edges, so the set has to be connected for
        one call to reach all of it. Which way the whole set ends up pointing is decided by
        ``reference_faces`` if any are given, and otherwise by ``direction``.

        This is the winding-only answer. Where the faces bound volume cells,
        :meth:`reorient_2d_by_3d` is the stronger one — it can tell inward from outward,
        which winding alone cannot.

        Args:
            direction: The normal the set is made to agree with, used when no reference
                faces are given.
            faces: The faces to orient. Empty means every face of the mesh.
            reference_faces: Faces whose current orientation is taken as correct.
            allow_non_manifold: Continue across an edge shared by more than two faces
                instead of stopping there.

        Returns:
            How many faces were reversed.

        Raises:
            PysmeshError: If an id names something that is not a face, or the mesher has
                been released.
        """
        raw = self._m.reorient_2d(
            list(direction), _ids(faces), _ids(reference_faces), allow_non_manifold
        )
        return cast("int", raw["reoriented"])

    def reorient_2d_by_3d(
        self,
        faces: Iterable[int] = (),
        volumes: Iterable[int] = (),
        outside_normal: bool = True,
    ) -> int:
        """Orient faces from the volume cells they bound.

        This is the operation an imported surface mesh usually needs and no array-side
        winding rule can supply: a face between two cells, or on the skin of a solid, has an
        unambiguous outward direction only because there is a cell behind it. A shell whose
        faces disagree with one another — the ordinary state of a mesh assembled from
        several sources — is made consistent in one call.

        Args:
            faces: The faces to orient. Empty means every face of the mesh.
            volumes: The cells to take the orientation from. Empty means every cell.
            outside_normal: Point each face away from the cell behind it. False points it
                inward.

        Returns:
            How many faces were reversed.

        Raises:
            PysmeshError: If the mesh has no volume cells, if an id is of the wrong family,
                or if the mesher has been released.
        """
        raw = self._m.reorient_2d_by_3d(_ids(faces), _ids(volumes), outside_normal)
        return cast("int", raw["reoriented"])

    # ---- Face splitting and fusing ------------------------------------------------------- #

    def quad_to_tri(
        self,
        elements: Iterable[int] = (),
        criterion: Control | None = None,
        diagonal_13: bool = True,
    ) -> EditReport:
        """Split quadrangles into triangles.

        Args:
            elements: The faces to split. Empty means every face of the mesh.
            criterion: The measure used to choose which diagonal to cut along — the split
                that gives the better value wins. ``None`` cuts every quadrangle along the
                same diagonal instead, which is what a caller wants when the halves have to
                line up with something else.
            diagonal_13: With no criterion, cut from the first node to the third rather than
                from the second to the fourth.

        Returns:
            The counts either side of the split.

        Raises:
            PysmeshError: If SMESH cannot split the faces given, if an id names something
                that is not a face, or if the mesher has been released.
        """
        name = "" if criterion is None else criterion.native_name
        params = {} if criterion is None else criterion.params()
        return _report(self._m.quad_to_tri(_ids(elements), name, params, diagonal_13))

    def tri_to_quad(
        self,
        elements: Iterable[int] = (),
        criterion: Control | None = None,
        max_angle: float = 0.0,
    ) -> EditReport:
        """Fuse neighbouring triangles into quadrangles.

        Args:
            elements: The triangles to fuse. Empty means every face of the mesh.
            criterion: The measure used to choose which neighbour to fuse with — the pair
                that gives the better value wins. ``None`` uses the 2-D aspect ratio, which
                is the measure this is normally judged by.
            max_angle: The largest angle between two triangles' normals, in radians, at
                which they may still be fused. 0 fuses only coplanar pairs.

        Returns:
            The counts either side of the fusion.

        Raises:
            PysmeshError: If ``max_angle`` is negative, if SMESH cannot fuse the faces
                given, if an id names something that is not a face, or if the mesher has
                been released.
        """
        measure = AspectRatio() if criterion is None else criterion
        return _report(
            self._m.tri_to_quad(
                _ids(elements), measure.native_name, measure.params(), max_angle
            )
        )

    # ---- Duplication ---------------------------------------------------------------------- #

    def double_elements(self, elements: Iterable[int]) -> EditReport:
        """Create a second element on the nodes of each named one.

        This is how an internal wall — a baffle, a zero-thickness membrane — is expressed in
        a CFD mesh, and there is no other way to express one. The duplicate shares every
        node with its original, so the two are geometrically coincident and the solver sees
        two sides of one surface.

        Args:
            elements: The elements to duplicate. Naming none is refused rather than taken to
                mean the whole mesh, because doubling a whole mesh is never what a wall
                means.

        Returns:
            The counts either side of the duplication.

        Raises:
            PysmeshError: If the list is empty, if an id names nothing, or if the mesher has
                been released.
        """
        return _report(self._m.double_elements(_ids(elements)))

    # ---- Sweeps --------------------------------------------------------------------------- #

    def extrusion_sweep(
        self,
        elements: Iterable[int],
        step: tuple[float, float, float],
        steps: int = 1,
        make_boundary: bool = True,
        tolerance: float = 1e-6,
    ) -> EditReport:
        """Sweep elements along a straight vector, filling the swept region with cells.

        A face swept once gives a cell of one dimension higher; an edge gives a face. The
        source elements stay where they are.

        Args:
            elements: The elements to sweep.
            step: One step of the sweep, as a vector. The total displacement is this times
                ``steps``.
            steps: How many cells to lay along the sweep.
            make_boundary: Also create the elements that close the sides of the swept
                region.
            tolerance: Distance below which a swept node is taken to have landed on an
                existing one.

        Returns:
            The counts either side of the sweep.

        Raises:
            PysmeshError: If the list is empty, if ``steps`` is below 1, if ``step`` is the
                zero vector, or if the mesher has been released.
        """
        return _report(
            self._m.extrusion_sweep(
                _ids(elements), list(step), steps, make_boundary, tolerance
            )
        )

    def rotation_sweep(
        self,
        elements: Iterable[int],
        axis_origin: tuple[float, float, float],
        axis_direction: tuple[float, float, float],
        angle: float,
        steps: int = 1,
        tolerance: float = 1e-6,
        make_walls: bool = True,
    ) -> EditReport:
        """Sweep elements around an axis, filling the swept region with cells.

        Args:
            elements: The elements to sweep.
            axis_origin: A point on the axis.
            axis_direction: The axis direction. The rotation is right-handed about it.
            angle: One step of the sweep, in radians. The total rotation is this times
                ``steps``.
            steps: How many cells to lay along the sweep.
            tolerance: Distance below which a swept node is taken to have landed on an
                existing one. This is what closes a full turn onto itself.
            make_walls: Also create the elements that close the sides of the swept region.

        Returns:
            The counts either side of the sweep.

        Raises:
            PysmeshError: If the list is empty, if ``steps`` is below 1, if the axis
                direction is the zero vector, or if the mesher has been released.
        """
        return _report(
            self._m.rotation_sweep(
                _ids(elements),
                list(axis_origin),
                list(axis_direction),
                angle,
                steps,
                tolerance,
                make_walls,
            )
        )

    # ---- Surface offset ------------------------------------------------------------------- #

    def offset(
        self,
        value: float,
        elements: Iterable[int] = (),
        copy_elements: bool = True,
        fix_self_intersection: bool = False,
    ) -> EditReport:
        """Build an offset surface from a triangle mesh, into this mesh.

        Each node moves along the average normal of the faces around it, and the result is
        added to this mesh. A positive value offsets along the normals, a negative one
        against them.

        The source must be **linear triangles**, and SMESH checks the whole mesh rather than
        the faces given — so a mesh that also holds quadrangles is refused even when the
        faces named are all triangles. Split them with :meth:`quad_to_tri` first.

        Args:
            value: The offset distance.
            elements: The faces to offset. Empty means every face of the mesh.
            copy_elements: Keep the source faces. False removes them, leaving only the
                offset surface.
            fix_self_intersection: Repair the places where the offset surface runs into
                itself, which happens wherever the offset exceeds a local radius of
                curvature. Costly, and worth it on anything but a smooth convex surface.

        Returns:
            The counts either side of the offset.

        Raises:
            PysmeshError: If the mesh is not all linear triangles, if the offset produced no
                elements, if an id names something that is not a face, or if the mesher has
                been released.
        """
        return _report(
            self._m.offset(value, _ids(elements), copy_elements, fix_self_intersection)
        )

    # ---- Sewing --------------------------------------------------------------------------- #

    def sew_free_border(
        self,
        border: tuple[int, int, int],
        side: tuple[int, int] | tuple[int, int, int],
        side_is_free_border: bool = True,
        create_polygons: bool = False,
        create_polyhedra: bool = False,
    ) -> EditReport:
        """Join a free border of the mesh to another border or to a chain of elements.

        A free border is a chain of element edges used by one element only — the open rim of
        a surface patch. Sewing replaces the nodes of one rim by the nodes of the other, so
        the two patches become one mesh. Where the two rims have different numbers of
        segments, nodes are inserted on the longer one.

        Args:
            border: Three node ids naming the first free border: its first node, the node
                next along it, and its last node. The direction the two middle nodes imply
                is what pairs the two rims up.
            side: Two or three node ids naming the other side, read the same way. Two is
                enough when the two rims have the same number of segments.
            side_is_free_border: The second side is a free border too. False means it is a
                chain of element edges inside the mesh.
            create_polygons: Where a face has to take extra nodes, make it a polygon instead
                of splitting it.
            create_polyhedra: The same for a volume cell.

        Returns:
            The counts either side of the sew.

        Raises:
            PysmeshError: If either border cannot be found from the nodes given, if the
                nodes are not on the border they were offered for, if a volume cell shares a
                link the sew would have to split, or if the mesher has been released.
        """
        return _report(
            self._m.sew_free_border(
                _ids(border),
                _ids(side),
                side_is_free_border,
                create_polygons,
                create_polyhedra,
            )
        )

    def sew_side_elements(
        self,
        side1: Iterable[int],
        side2: Iterable[int],
        first_nodes: tuple[int, int],
        second_nodes: tuple[int, int],
    ) -> EditReport:
        """Join two matching sets of elements by merging their nodes.

        This is the sew for two whole patches rather than two rims: every node of ``side1``
        is merged with the node of ``side2`` that corresponds to it. The two sets must hold
        the same number of elements and have the same connectivity, and the correspondence
        is fixed by naming one pair of linked nodes on each side.

        Args:
            side1: The elements of the first side.
            side2: The elements of the second side.
            first_nodes: A node on side 1's border and the node on side 2 that matches it.
            second_nodes: A second such pair, linked to the first pair on their own sides.
                The two pairs together fix the orientation of the correspondence.

        Returns:
            The counts either side of the sew.

        Raises:
            PysmeshError: If either side is empty, if the two sides differ in size or
                connectivity, if the nodes named are not on the borders, or if the mesher
                has been released.
        """
        return _report(
            self._m.sew_side_elements(
                _ids(side1), _ids(side2), _ids(first_nodes), _ids(second_nodes)
            )
        )

    # ---- Deletion ----------------------------------------------------------------------- #

    def remove_elements(
        self, elements: Iterable[int], free_nodes: bool = False
    ) -> RemovalReport:
        """Delete elements from the mesh.

        This is deletion, not merging: the cells named here are taken away and nothing
        replaces them. :meth:`merge_nodes` and :meth:`merge_equal_elements` collapse
        entities onto one another and rewrite what used them; this leaves a hole.

        Groups follow, without being told to. SMESH drops a deleted element from every
        explicit group it belonged to, and never renumbers a survivor — so a wall, or a patch
        stored as a group, is still itself afterwards, minus whatever went.

        Nodes are **not** removed with the elements by default. A node is an entity in its own
        right and the ones on the boundary of a deleted patch usually still carry cells;
        ``free_nodes`` sweeps the ones left carrying nothing, which is what dropping a whole
        patch of a surface wants.

        Args:
            elements: The element ids to delete. Listing one twice deletes it once.
            free_nodes: Also delete every node the removal leaves with no element on it.

        Returns:
            The counts either side of the deletion, and the ids that actually went.

        Raises:
            PysmeshError: If an id names nothing in the mesh — the whole call is refused
                before anything is deleted — or if the mesher has been released.
        """
        return _removal(self._m.remove_elements(_ids(elements), free_nodes))

    def remove_nodes(self, nodes: Iterable[int]) -> RemovalReport:
        """Delete nodes from the mesh, and every element built on them.

        The cascade is not optional and is not a convenience: an element whose corner has
        gone is not a cell at all, so SMESH removes it with the node. The elements it took
        are named in the report, because they are the part a caller did not ask for and has
        to remap against.

        To delete a node without disturbing the elements around it, merge it onto a
        neighbour with :meth:`merge_node_groups` instead — that rewrites their connectivity
        rather than dropping them.

        Args:
            nodes: The node ids to delete. Listing one twice deletes it once.

        Returns:
            The counts either side of the deletion, and the ids that actually went — the
            nodes named, and every element that went with them.

        Raises:
            PysmeshError: If an id names nothing in the mesh — the whole call is refused
                before anything is deleted — or if the mesher has been released.
        """
        return _removal(self._m.remove_nodes(_ids(nodes)))
