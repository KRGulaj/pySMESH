# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""pySMESH session — GEOM's blocks: hexahedral decomposition.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import (
    BlocksCheck,
    CancelPredicate,
    EntityId,
    HistoryDelta,
    PipeTShape,
    ProgressCallback,
    Vec3,
    _blocks_check,
    _delta,
    _ids,
    _pipe_t_shape,
)


def _vectors(where: str, vectors: Sequence[Vec3]) -> list[tuple[float, float, float]]:
    """The translation vectors as float triples; refused when one is not three numbers."""
    out: list[tuple[float, float, float]] = []
    for v in vectors:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (3,):
            raise PysmeshError(
                f"{where}: the translation vector {v!r} is not three numbers."
            )
        out.append((float(arr[0]), float(arr[1]), float(arr[2])))
    return out


def _counts(where: str, counts: Sequence[int]) -> list[int]:
    """The counts as ints; refused when one is not an integer."""
    out: list[int] = []
    for c in counts:
        if not isinstance(c, (int, np.integer)) or isinstance(c, bool):
            raise PysmeshError(f"{where}: the count {c!r} is not an integer.")
        out.append(int(c))
    return out


class _BlockOps(_SessionBase):
    """GEOM's blocks: quadrangles, hexahedra, block checks and searches."""

    __slots__ = ()

    def make_quad(self, edges: Sequence[EntityId]) -> HistoryDelta:
        """Build a quadrangle face on 4 edges, or between 2 opposite edges.

        GEOM's ``MakeQuad`` and ``MakeQuad2Edges``: the quad branches of
        ``GEOMImpl_BlockDriver``, copied into pySMESH.

        * **4 edges**, in any order. Their ends must meet at 4 distinct corners, within
          the vertex tolerances (``BRepTools::Compare``); ends that only coincide are
          welded onto one vertex. The wire is closed.
        * **2 edges**: two straight edges link their ends, pairing the ends with the
          shorter sum of lengths.

        The face lies on any surface that holds the edges: a plane when they are planar,
        otherwise GEOM's filling surface (``BRepOffsetAPI_MakeFilling`` with the edges as
        C0 constraints, then the face on that surface with the edges as its boundary,
        healed by GEOM's default shape processing). A filling is a plate approximation,
        not the bilinear patch through the four edges.

        **Identity.** The bodies that own the edges (loose edges and wires) are consumed,
        as :meth:`make_face` consumes them. Each edge keeps its id on the face; two ends
        welded onto one vertex keep both their ids on it (``merged``). The face and the
        linking edges of the 2-edge form get new ids whose role is ``GENERATED`` and whose
        origin is the given edges.

        Args:
            edges: 4 edges, or 2 opposite edges.

        Returns:
            The delta: the face (and any linking edges) in ``created``.

        Raises:
            PysmeshError: If not 4 or 2 edges are named, an id is dead, is not an edge,
                was split or is named twice; if an edge belongs to a face or solid body;
                or if GEOM's driver builds no face (the message gives GEOM's reason, such
                as "The edges must form a closed wire").
        """
        return _delta(self._s.make_quad(_ids(edges)))

    def make_quad_from_vertices(self, vertices: Sequence[EntityId]) -> HistoryDelta:
        """Build a quadrangle face with 4 given corners.

        GEOM's ``MakeQuad4Vertices``, copied: the corners are ordered by the shortest
        perimeter of the three quadrangles they can form, and the face lies on any surface
        under their closed polygon (a plane when they are coplanar, GEOM's filling
        otherwise, as :meth:`make_quad`).

        **Identity.** The vertex bodies are consumed; each vertex keeps its id as a corner
        of the face. The four edges are new, each generated from its two corners; the face
        is new, generated from the four corners.

        Args:
            vertices: 4 free vertices (:meth:`add_vertex`), in any order.

        Returns:
            The delta: the face and its 4 edges in ``created``.

        Raises:
            PysmeshError: If not 4 vertices are named, an id is dead, is not a vertex or is
                named twice; if a vertex belongs to another kind of body; if two corners
                coincide within 1e-7; or if GEOM's driver builds no face.
        """
        return _delta(self._s.make_quad_from_vertices(_ids(vertices)))

    def make_hexa(self, faces: Sequence[EntityId]) -> HistoryDelta:
        """Build a hexahedral solid on 6 faces, or between 2 opposite faces.

        GEOM's ``MakeHexa`` and ``MakeHexa2Faces``: the hexa branches of
        ``GEOMImpl_BlockDriver`` and GEOM's block explorer ``GEOMImpl_Block6Explorer``,
        copied into pySMESH.

        * **6 faces**, in any order. They are joined into one shell (a
          ``BRepTools_Quilt`` over shared edges, then GEOM's sewing
          ``ShHealOper_Sewing`` at the largest face tolerance when they share none), which
          must be closed.
        * **2 faces**, each bounded by one wire of 4 edges. The explorer pairs their
          corners by the shortest sum of distances and builds the 4 linking edges and the
          4 side faces: planar where the side's edges are planar, otherwise one ruled
          surface through both wires (``BRepOffsetAPI_ThruSections``).

        The shell becomes a solid, turned so that its outside is outward.

        **Identity.** The bodies of the faces (free faces and shells) are consumed. Each
        face keeps its id on the solid, and edges or vertices that the sewing joins keep
        all their ids on the joined one (``merged``). The solid is new, generated from the
        faces; the linking edges and side faces of the 2-face form are new, generated from
        the corners and the edges they join.

        Args:
            faces: 6 faces, or 2 opposite faces.

        Returns:
            The delta: the solid (and any new faces and edges) in ``created``.

        Raises:
            PysmeshError: If not 6 or 2 faces are named, an id is dead, is not a face, was
                split or is named twice; if a face belongs to a solid body; or if GEOM's
                driver builds no solid (the message gives GEOM's reason, such as
                "Impossible to build a closed shell on the given faces").
        """
        return _delta(self._s.make_hexa(_ids(faces)))

    def check_blocks(
        self,
        entities: Sequence[EntityId] | None = None,
        *,
        tolerance: float = 1e-12,
        use_c1: bool = False,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> BlocksCheck:
        """Check whether the bodies form a compound of blocks.

        GEOM's ``CheckCompoundOfBlocks`` (``GEOMImpl_IBlocksOperations``), copied: every
        solid must be a block (6 faces, each one wire of 4 edges), the blocks must be glued
        (a face two blocks touch along is one face), and connected. The defaults are
        ``geomBuilder.CheckCompoundOfBlocks``'s.

        * A solid with 6 faces of more than 4 edges, or with seam or degenerated edges,
          that ``BlockFix_CheckTool`` finds could be a block once merged is
          :attr:`BlocksErrorKind.EXTRA_EDGE`; another non-block is
          :attr:`BlocksErrorKind.NOT_BLOCK`. Free faces and shells are not checked
          (GEOM records their non-quadrangles but does not report them); an edge, wire or
          vertex body is a non-block.
        * Two blocks that touch along coincident faces that are not one face are
          :attr:`BlocksErrorKind.NOT_GLUED` (``GEOMAlgo_GlueAnalyser`` at 1e-7).
        * Blocks that no chain of blocks connects to the largest connected set are
          :attr:`BlocksErrorKind.NOT_CONNECTED`.

        With no block at all, the answer is ``False`` with no finding, as GEOM gives it.

        A query: it changes nothing and issues no id.

        Args:
            entities: Entities whose owning bodies are checked. ``None`` checks every body.
            tolerance: The angle in radians under which two edges of a face continue each
                other (GEOM's ``theAngTolerance``). Read only when ``use_c1`` is true.
            use_c1: Count a face whose wire has more than 4 edges as a quadrangle when they
                form 4 bounds of C1 continuity (GEOM's ``theIsUseC1``).
            progress: Called with the fraction done, from 0 to 1.
            cancel: Asked whether to stop; a cancel raises
                :class:`~pysmesh.PysmeshCancelled` between two bodies or two stages.

        Returns:
            The verdict and the findings.

        Raises:
            PysmeshError: If ``entities`` is empty or an id is dead; if ``tolerance`` is
                not finite; or if the glue analyser fails.
            PysmeshCancelled: If ``cancel`` asked to stop.
        """
        ids = None if entities is None else _ids(entities)
        angular = float(tolerance) if use_c1 else -1.0
        return _blocks_check(self._s.check_blocks(ids, angular, progress, cancel))

    def propagation_chain(self, edge: EntityId) -> NDArray[np.int64]:
        """The chain of opposite edges through quadrangles that holds an edge.

        GEOM's ``Propagate``, copied for one chain: from the edge, every wire of exactly 4
        edges that holds an edge of the chain adds the edge opposite to it, pass by pass,
        across the faces of every body of the model. On a structured block mesh these are
        the edges that take one number of segments.

        Each edge of the chain is given once, by its label: the lowest live id that
        denotes it (:meth:`entities` with ``distinct=True`` lists the same ids).

        A query: it changes nothing and issues no id.

        Args:
            edge: An edge id that denotes one edge.

        Returns:
            (K,) int64, ascending: the chain, the edge's own label among them.

        Raises:
            PysmeshError: If the id is dead, is not an edge or was split.
        """
        return self._s.propagation_chain(int(edge))

    def opposite_face(self, block: EntityId, face: EntityId) -> EntityId:
        """The face of a block opposite to one of its faces.

        GEOM's ``GetOppositeFace``, copied: GEOM's block explorer orders the block from the
        face and returns the face across from it.

        A query: it changes nothing and issues no id.

        Args:
            block: A solid id of a block (6 faces of 4 edges).
            face: A face of that solid.

        Returns:
            The opposite face's label (the lowest live id that denotes it).

        Raises:
            PysmeshError: If an id is dead or of the wrong kind; if the face is not a face
                of the solid; or if the solid is not a block the explorer can order (the
                message gives GEOM's reason).
        """
        return EntityId(int(self._s.opposite_face(int(block), int(face))))

    def face_by_edges(self, edge_a: EntityId, edge_b: EntityId) -> EntityId:
        """The one face of the model that holds two edges.

        GEOM's ``GetFaceByEdges``, copied.

        A query: it changes nothing and issues no id.

        Args:
            edge_a: An edge id.
            edge_b: Another edge id.

        Returns:
            The face's label (the lowest live id that denotes it).

        Raises:
            PysmeshError: If an id is dead or not an edge; or if no face, or more than
                one, holds both edges (the message names them).
        """
        return EntityId(int(self._s.face_by_edges(int(edge_a), int(edge_b))))

    def find_block_near(self, point: Vec3) -> EntityId:
        """The solid that holds a point, or the one nearest to it.

        GEOM's ``GetBlockNearPoint``, copied: the first solid with the point inside; else
        the one solid with the point on its boundary; else, of the solids that leave the
        point out, the nearest (``BRepExtrema_DistShapeShape``). Solids are taken in the
        order of the model, so a tie gives the same solid in every run.

        A query: it changes nothing and issues no id.

        Args:
            point: The point.

        Returns:
            The solid's label (the lowest live id that denotes it).

        Raises:
            PysmeshError: If the point is not finite; if the model has no solid; or if the
                point lies on the boundary of several solids and inside none (the message
                names them).
        """
        x, y, z = point
        return EntityId(int(self._s.find_block_near(float(x), float(y), float(z))))

    def multi_transform(
        self,
        block: EntityId,
        vectors: Sequence[Vec3],
        counts: Sequence[int],
    ) -> HistoryDelta:
        """Copy a block along 1 or 2 translation vectors and glue the copies.

        GEOM's ``MakeMultiTransformation1D`` and ``2D``, with the steps given as vectors
        rather than as pairs of the block's faces. Along ``vectors[k]`` there are
        ``counts[k]`` blocks, the original counted, at ``0, 1, ..., counts[k] - 1`` times
        the vector; with two vectors, every combination. Each copy is the session's copy of
        the body, moved by a translation, and the original and its copies are then glued
        as :meth:`glue` glues faces (``GEOMAlgo_Gluer2``), at the block's largest vertex
        tolerance, as GEOM glues them. Copies that touch nothing stand apart.

        **Identity.** The original keeps every id. Each entity of a copy gets a new id
        whose role is ``GENERATED`` and whose origin is the original's entity at the same
        place. A glued face keeps every id it had (the third rule); between two copies it
        gets one new id.

        Args:
            block: An entity of the body to copy.
            vectors: 1 or 2 translation vectors.
            counts: For each vector, the number of blocks along it, at least 1.

        Returns:
            The delta: the copies in ``created``, the original's glued entities in
            ``modified``.

        Raises:
            PysmeshError: If not 1 or 2 vectors are given or the counts do not match them;
                if a count is below 1, or all are 1; if a vector with a count above 1 is
                shorter than 1e-7; if a value is not finite; or if the glue fails (nothing
                is committed then).
        """
        where = "Session.multi_transform"
        return _delta(
            self._s.multi_transform(
                int(block), _vectors(where, vectors), _counts(where, counts)
            )
        )

    def remove_extra_edges(
        self,
        entities: Sequence[EntityId],
        *,
        union_faces: bool = False,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Remove seam, degenerated and extra edges, as GEOM's ``RemoveExtraEdges`` does.

        GEOM's ``BLOCK_REMOVE_EXTRA`` branch of ``GEOMImpl_BlockDriver``, with
        ``BlockFix_BlockFixAPI::Perform`` and its BlockFix steps copied into pySMESH, on
        the bodies that own the entities, in this order:

        1. A copy of the bodies (``TNaming_CopyShape``), then a second one
           (``BRepBuilderAPI_Copy``) for the rotation.
        2. ``RotateSphereSpace``: a spherical face that reaches a pole of its sphere is
           put on the same sphere with its axis turned, so that no pole lies on the face
           (``BlockFix_SphereSpaceModifier``). The turned copy is kept when it is valid.
        3. ``RefillProblemFaces``: a free-form face with a degenerated edge at a singular
           point of its surface is replaced by a filling face.
        4. With ``union_faces`` only: the faces on one surface are united
           (``ShapeUpgrade_UnifySameDomain``, faces only, B-splines joined, both
           tolerances 1e-7).
        5. The locations are removed (``ShapeUpgrade_RemoveLocations``).
        6. The edges on one curve are united (``ShapeUpgrade_UnifySameDomain``, edges
           only, B-splines joined, linear tolerance 1e-7, angular tolerance 1e-5).
        7. ``FixRanges``: a cylindrical or spherical face whose U range leaves
           ``[0, 2 pi]`` is put on its surface turned about the axis
           (``BlockFix_PeriodicSurfaceModifier``).
        8. A result that ``BRepCheck_Analyzer`` rejects is repaired by GEOM's default
           ``ShapeProcess``: FixShape, FixFaceSize, DropSmallEdges and SameParameter with
           the values of GEOM's resource file. No resource file is read.

        ``union_faces`` is ``doUnionFaces`` of ``geomBuilder.RemoveExtraEdges``, with its
        default: GEOM passes ``OptimumNbFaces`` 0 when it is true and -1 when it is false.
        Without it, the faces of a box fused from two boxes stay split, and an edge
        between them joins no other edge.

        **Identity.** Each step writes its history, and every id is followed through the
        steps in order. A united face or edge keeps the ids of every face or edge united
        into it (the third rule). A removed edge or vertex dies. A rebuilt face, edge or
        vertex keeps its id.

        Args:
            entities: Entities whose owning bodies are improved.
            union_faces: Unite the faces on one surface too.
            progress: Called with the fraction done, from 0 to 1.
            cancel: Asked whether to stop; a cancel raises
                :class:`~pysmesh.PysmeshCancelled` between two steps.

        Returns:
            The delta: united faces and edges in ``merged``, removed edges and vertices
            in ``deleted``.

        Raises:
            PysmeshError: If ``entities`` is empty or an id is dead; if OCCT fails or the
                repair reports a failure; or if the result has as many solids, faces,
                edges and vertices as the bodies: nothing was removed or united, and
                nothing is committed.
            PysmeshCancelled: If ``cancel`` asked to stop. The session is unchanged.
        """
        return _delta(
            self._s.remove_extra_edges(
                _ids(entities), bool(union_faces), progress, cancel
            )
        )

    def divided_disk(
        self,
        radius: float,
        ratio: float,
        *,
        orientation: Literal["xy", "yz", "zx"] = "xy",
        pattern: Literal["square", "hexagon"] = "square",
    ) -> HistoryDelta:
        """Add a disk divided into quadrangles, for a block mesh of a pipe's cross-section.

        GEOM's ``MakeDividedDisk``: the geometry code of
        ``AdvancedEngine_DividedDiskDriver``, copied into pySMESH. The disk is centred at
        the origin in the plane ``orientation`` (GEOM's 1 = OXY, 2 = OYZ, 3 = OZX; the
        plane's normal is z, x and y). Its faces are planar quadrangles:

        * ``"square"``: an inner square whose corners lie on the plane's axes at
          ``a = ratio / 100 * radius`` from the centre (a square of side ``a sqrt(2)``
          turned by 45 degrees), and 4 faces between it and the circle; 5 faces.
        * ``"hexagon"``: an inner regular hexagon whose corners lie at ``a`` from the
          centre, cut into 6 quadrangles at the centre, and 12 faces between it and the
          circle; 18 faces.

        ``ratio`` is GEOM's ``theRatio``, the distance of the inner block's corners from
        the centre in percent of the radius (GEOM's builder passes 67). The result is one
        shell body; every id is ``CONSTRUCTED``. Sweep it (:meth:`extrude`) to get a
        pipe of hexahedral blocks.

        Args:
            radius: The disk's radius, above 1e-7.
            ratio: Strictly between 0 and 100.
            orientation: The plane of the disk.
            pattern: The division pattern.

        Returns:
            The delta: the shell's faces, edges and vertices in ``created``.

        Raises:
            PysmeshError: If ``radius`` is not above 1e-7, ``ratio`` is not strictly
                between 0 and 100, a value is not finite, or ``orientation`` or
                ``pattern`` is not one of the names above.
        """
        return _delta(
            self._s.divided_disk(float(radius), float(ratio), orientation, pattern)
        )

    def pipe_t_shape(
        self,
        main_radius: float,
        main_thickness: float,
        main_half_length: float,
        branch_radius: float,
        branch_thickness: float,
        branch_half_length: float,
        *,
        partition: bool = True,
    ) -> PipeTShape:
        """Add the wall of a T junction of two pipes, divided into blocks for a hex mesh.

        GEOM's ``MakePipeTShape`` for the wall only: no thickness reduction, chamfer or
        fillet (GEOM's ``theRL`` ... ``theLthinI`` all 0). The method of GEOM's advanced
        operations is ported to OCCT calls, step by step.

        **Placement, as GEOM's.** The main pipe runs along x from ``-main_half_length``
        to ``+main_half_length``; the branch runs along +z from the main pipe's axis
        (``z = 0``) to ``z = branch_half_length``: that is the branch's whole length from
        the axis, which GEOM calls its half length. The radii are the inner radii; each
        wall is ``thickness`` thick.

        **partition** (GEOM's ``theHexMesh``, default true). The quarter ``x <= 0, y <= 0``
        is cut by the plane ``z = 0``, by the plane through the y axis towards
        ``(R1+W1, 0, (L1 + R2+W2) / 2)`` (when the outer radii differ), and by the face
        across the wall where the pipes meet; GEOM checks the quarter is a compound of
        blocks; it is mirrored in ``x = 0`` and ``y = 0`` and the four quarters' faces are
        glued. Without it the wall is one solid.

        **Groups.** GEOM's named groups, in GEOM's order: with ``partition``, the three
        pipe ends (``JUNCTION_FACE_1`` .. ``3``, GEOM's sorting of the end faces) and the
        edge chains GEOM names (``THICKNESS``, ``CIRCULAR_QUARTER_PIPE`` (twice),
        ``HALF_LENGTH_MAIN_PIPE``, ``HALF_LENGTH_INCIDENT_PIPE``, ``FLANGE``); then
        ``INTERNAL_FACES``, the faces on the two inner cylinders. pySMESH adds
        ``EXTERNAL_FACES``, the faces on the two outer cylinders by the same rule; GEOM
        names no such group.

        Every id is ``CONSTRUCTED``; the result is one body per block (one solid without
        ``partition``).

        Args:
            main_radius: Inner radius ``R1`` of the main pipe.
            main_thickness: Wall thickness ``W1`` of the main pipe.
            main_half_length: Half length ``L1`` of the main pipe.
            branch_radius: Inner radius ``R2`` of the branch, at most ``R1``.
            branch_thickness: Wall thickness ``W2`` of the branch, with
                ``R2 + W2 <= R1 + W1``; equal outer radii need equal inner radii.
            branch_half_length: Length ``L2`` of the branch from the main axis, above
                ``R1 + W1``; ``R2 + W2`` must stay below ``L1``.
            partition: Divide the wall into blocks.

        Returns:
            The delta and the named groups.

        Raises:
            PysmeshError: If a value is not above 1e-7 or not finite; if the dimensions
                break one of GEOM's rules above (the message gives all six values); or if
                a step of GEOM's construction fails (the message gives GEOM's reason).
        """
        return _pipe_t_shape(
            self._s.pipe_t_shape(
                float(main_radius),
                float(main_thickness),
                float(main_half_length),
                float(branch_radius),
                float(branch_thickness),
                float(branch_half_length),
                bool(partition),
            )
        )
