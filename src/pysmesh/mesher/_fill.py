# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-13

"""pySMESH mesher — filling a mesh from arrays instead of computing it from geometry.

Part of the :mod:`pysmesh.mesher` package. This is the injection path: a
:class:`~pysmesh.Mesher` built with ``shape=None`` starts empty, and these operations put
nodes and elements into it directly.

It exists because a discrete body has no B-rep to compute from. An imported STL, OBJ or PLY,
a shrink-wrap result, the boundary another mesher produced, a mesh read back from a file —
none of them has geometry behind it, and before this there was no way to hand one to a mesher
at all. With it, the whole editing and search surface applies to such a body: the quality
controls measure it, :meth:`~pysmesh.Mesher.sharp_edges` and
:meth:`~pysmesh.Mesher.separate_faces_by_edges` partition it into patches, the editor repairs
it, and :meth:`~pysmesh.Mesher.remove_elements` cuts pieces out of it.

**Two id conventions meet here, and they are not interchangeable.**
:meth:`~_FillOps.add_elements` and its typed wrappers take **node ids** — what
:meth:`~_FillOps.add_nodes` returned. :meth:`~pysmesh.Mesher.from_arrays` takes **row
indices** into the coordinate table it is given, because at that point no ids exist yet. The
one that matches a harvest is the row index: :attr:`MeshData.element_nodes` is row-indexed
too.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._base import _MesherBase
from ._types import ElementType, MeshData


class _FillOps(_MesherBase):
    """Putting nodes and elements into a mesh directly."""

    __slots__ = ()

    def add_nodes(self, coords: NDArray[np.float64]) -> NDArray[np.int64]:
        """Insert nodes at the given positions.

        The nodes are bound to no sub-shape, on a mesher with a shape as much as on one
        without: a caller-supplied point has no geometry to sit on, and inventing one would
        be this binding deciding where it belongs.

        Args:
            coords: (N, 3) float64 — model-space position of each new node.

        Returns:
            (N,) int64 — the new mesh ids, in the same row order. These are what every other
            operation on this mesher addresses a node by.

        Raises:
            PysmeshError: If ``coords`` is not (N, 3), or the mesher has been released.
        """
        return self._m.add_nodes(np.ascontiguousarray(coords, dtype=np.float64))

    def add_elements(
        self, element_type: ElementType, connectivity: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """Insert elements of one type from a table of node ids.

        Args:
            element_type: The cell type to build. Its node count fixes the column count and
                is checked against it, so a triangle handed four columns is refused rather
                than quietly built as a quadrangle.
            connectivity: (M, k) integer — one row per element, holding the **node ids** its
                corners are, in SMESH's own node order for that type.

        Returns:
            (M,) int64 — the new mesh ids, in the same row order.

        Raises:
            PysmeshError: If ``element_type`` is a polygon or a polyhedron, whose node count
                does not determine its shape; if the column count does not match the type; if
                a row names a node the mesh does not have; if the type is
                :attr:`ElementType.BALL`, which carries a diameter this path cannot give it;
                or if the mesher has been released.
        """
        table = np.ascontiguousarray(connectivity, dtype=np.int64)
        return self._m.add_elements(int(element_type), table)

    def add_segments(self, connectivity: NDArray[np.int64]) -> NDArray[np.int64]:
        """Insert linear 1-D elements.

        Args:
            connectivity: (M, 2) integer — the two node ids of each segment.

        Returns:
            (M,) int64 — the new mesh ids.

        Raises:
            PysmeshError: If a row names a node the mesh does not have, or the mesher has been
                released.
        """
        return self.add_elements(ElementType.EDGE, connectivity)

    def add_triangles(self, connectivity: NDArray[np.int64]) -> NDArray[np.int64]:
        """Insert linear triangles.

        Args:
            connectivity: (M, 3) integer — the three node ids of each triangle.

        Returns:
            (M,) int64 — the new mesh ids.

        Raises:
            PysmeshError: If a row names a node the mesh does not have, or the mesher has been
                released.
        """
        return self.add_elements(ElementType.TRIANGLE, connectivity)

    def add_quadrangles(self, connectivity: NDArray[np.int64]) -> NDArray[np.int64]:
        """Insert linear quadrangles.

        Args:
            connectivity: (M, 4) integer — the four node ids of each quadrangle.

        Returns:
            (M,) int64 — the new mesh ids.

        Raises:
            PysmeshError: If a row names a node the mesh does not have, or the mesher has been
                released.
        """
        return self.add_elements(ElementType.QUADRANGLE, connectivity)

    def add_tetrahedra(self, connectivity: NDArray[np.int64]) -> NDArray[np.int64]:
        """Insert linear tetrahedra.

        Args:
            connectivity: (M, 4) integer — the four node ids of each tetrahedron.

        Returns:
            (M,) int64 — the new mesh ids.

        Raises:
            PysmeshError: If a row names a node the mesh does not have, or the mesher has been
                released.
        """
        return self.add_elements(ElementType.TETRAHEDRON, connectivity)

    def fill_from_mesh(self, mesh: MeshData) -> None:
        """Fill an empty mesher from a harvest, keeping every id.

        This is what turns a mesh that is only arrays back into a live one —
        :func:`~pysmesh.read_gmf` reads a file into arrays and drops the binding, and this is
        the way back. Ids are kept rather than reassigned, so a group, a control result or a
        selection keyed on the old ids still means the same entities.

        The CAD binding is **not** restored: ``node_kind`` / ``element_ordinal`` and the rest
        describe a shape this mesher does not have, so everything comes back bound to nothing.

        Args:
            mesh: The arrays to rebuild from, as :meth:`~pysmesh.Mesher.mesh` returns them.

        Raises:
            PysmeshError: If the mesher already holds anything — the arrays carry absolute
                ids and a partial fill would collide; if an id is duplicated or not positive;
                if a row's connectivity does not name a row of ``node_coords``; if the mesh
                holds a polygon or a polyhedron, which cannot be rebuilt from these arrays
                alone; or if the mesher has been released.
        """
        self._m.fill_from_mesh(
            {
                "node_coords": mesh.node_coords,
                "node_id": mesh.node_id,
                "element_offsets": mesh.element_offsets,
                "element_nodes": mesh.element_nodes,
                "element_type": mesh.element_type,
                "element_id": mesh.element_id,
            }
        )
