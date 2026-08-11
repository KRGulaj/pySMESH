# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""pySMESH session — the geometric query surface over the live shape.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import (
    AdjacencyPairs,
    BoundsTable,
    CurvatureTable,
    EntityId,
    EntityKind,
    MassTable,
    Points,
    Projection,
    SurfaceSample,
    TypeTable,
    Vec3,
    _DEFAULT_CLASSIFY_TOL,
    _DEFAULT_CURVATURE_SAMPLES,
    _ids,
    _pairs,
    _points,
    _projection,
    _surface_sample,
)


class _QueryOps(_SessionBase):
    """The geometric query surface over the live shape."""

    __slots__ = ()

    def entity_types(self, kind: EntityKind) -> TypeTable:
        """Underlying geometry type of every live entity of one kind.

        Args:
            kind: The entity kind to classify.

        Returns:
            Ids and one type name per id. Faces get a surface type, edges a curve type; a
            solid or vertex gets its own kind's name, because neither has a distinct
            underlying geometry.
        """
        raw = self._s.entity_types(str(kind))
        return TypeTable(
            kind=kind,
            ids=cast("NDArray[np.int64]", raw["ids"]),
            types=tuple(cast("list[str]", raw["types"])),
        )

    def bounding_boxes(self, kind: EntityKind) -> BoundsTable:
        """Bounding box of every live entity of one kind.

        The cheap bulk query. :meth:`entity_table` also returns boxes, but pays for mass
        properties to do it — on a large assembly that is seconds rather than milliseconds.
        Use this one for culling, spatial indexing and picking.

        Args:
            kind: The entity kind to bound.

        Returns:
            Ids and their boxes, ascending by id.
        """
        raw = self._s.bounding_boxes(str(kind))
        return BoundsTable(
            kind=kind,
            ids=cast("NDArray[np.int64]", raw["ids"]),
            bbox=cast("NDArray[np.float64]", raw["bbox"]),
        )

    def mass_properties(self, entities: Sequence[EntityId]) -> MassTable:
        """Measure and centre of mass of the named entities.

        Each entity is measured by its own kind — volume for a solid, area for a face, length
        for an edge — never by walking a parent. That distinction matters: OCCT's linear
        properties of a *solid* visit every edge once per owning face, so a total edge length
        taken that way comes out doubled.

        Args:
            entities: Entity ids, of any kinds.

        Returns:
            The measures and centroids, in the order the entities were named.

        Raises:
            PysmeshError: If an id was never issued, or is dead.
        """
        raw = self._s.mass_properties(_ids(entities))
        return MassTable(
            ids=cast("NDArray[np.int64]", raw["ids"]),
            measure=cast("NDArray[np.float64]", raw["measure"]),
            centroid=cast("NDArray[np.float64]", raw["centroid"]),
        )

    def face_parameter_bounds(
        self, face_ids: Sequence[EntityId]
    ) -> NDArray[np.float64]:
        """Parameter domain of the named faces.

        Args:
            face_ids: Face entity ids. Each must denote exactly one face.

        Returns:
            (N, 4) float64 — umin, umax, vmin, vmax, in the order named.

        Raises:
            PysmeshError: If an id is dead, is not a face, or was split.
        """
        return self._s.face_parameter_bounds(_ids(face_ids))

    def edge_parameter_bounds(
        self, edge_ids: Sequence[EntityId]
    ) -> NDArray[np.float64]:
        """Parameter range of the named edges.

        Args:
            edge_ids: Edge entity ids. Each must denote exactly one edge.

        Returns:
            (N, 2) float64 — first, last parameter, in the order named.

        Raises:
            PysmeshError: If an id is dead, is not an edge, or was split.
        """
        return self._s.edge_parameter_bounds(_ids(edge_ids))

    def adjacency(self, kind: EntityKind, other_kind: EntityKind) -> AdjacencyPairs:
        """Which entities of one kind touch which entities of another.

        Which way the relation runs follows from the two kinds, so one method answers both
        questions a caller has. Towards a lower dimension it is the **boundary**:
        ``adjacency(FACE, EDGE)`` gives each face's edges. Towards a higher one it is the
        **ancestors**: ``adjacency(EDGE, FACE)`` gives each edge's faces. The two are the
        same relation read from either end, and both are needed — the first to walk a body
        down, the second to find what a picked edge belongs to.

        Args:
            kind: The kind to key the rows by.
            other_kind: The kind to relate them to. Must differ from ``kind``.

        Returns:
            One row per (entity, neighbour) pair.

        Raises:
            PysmeshError: If the two kinds are the same.
        """
        raw = self._s.adjacency(str(kind), str(other_kind))
        return AdjacencyPairs(
            kind=kind,
            other_kind=other_kind,
            ids=cast("NDArray[np.int64]", raw["ids"]),
            related=cast("NDArray[np.int64]", raw["related"]),
        )

    def surface_at(
        self, face_id: EntityId, uv: NDArray[np.float64] | Sequence[tuple[float, float]]
    ) -> SurfaceSample:
        """Positions and outward normals of a face at the given parameters.

        The normal points **out of the body**, not along the surface's own parametrisation:
        a reversed face's surface normal points inward, and every consumer of a normal — a
        boundary condition, a shading pass, an offset direction — means the outward one.

        Args:
            face_id: A face entity id denoting exactly one face.
            uv: (N, 2) parameters. Get the valid range from
                :meth:`face_parameter_bounds`.

        Returns:
            The points, the normals, and which of them are defined.

        Raises:
            PysmeshError: If the id is dead, is not a face, was split, or if ``uv`` is not
                (N, 2).
        """
        return _surface_sample(self._s.surface_at(int(face_id), _pairs("uv", uv)))

    def curvature(
        self,
        face_ids: Sequence[EntityId],
        *,
        samples: int = _DEFAULT_CURVATURE_SAMPLES,
    ) -> CurvatureTable:
        """Peak absolute curvature of each named face, over a grid of its parameter domain.

        **The grid is the point of the operation.** Sampling one point at a face's parametric
        centre is exact only for a face of constant curvature and arbitrarily wrong
        otherwise: on a cone tapering from radius 4 to radius 1, the centre sample reads
        0.358 against a true peak of 0.894. A curvature-driven sizing field built from centre
        samples therefore under-refines every tapering, blending or varying face in the
        model — which is exactly where refinement is needed.

        Two details the result depends on. The reported value is ``max(|k1|, |k2|)``, not the
        larger *signed* principal curvature: those are ordered by value, so on a cylinder they
        are ``(0, -1/R)`` and the larger one is 0. And samples land at cell centres, so none
        sits on a seam or a pole, and a sample outside the face's own trimming — inside a
        hole, or on the cut-away part of a trimmed patch — is discarded rather than reporting
        a curvature the face does not have.

        Args:
            face_ids: Face entity ids, each denoting exactly one face.
            samples: Grid resolution per parametric direction (>= 1). Cost is quadratic in
                this; ``samples=1`` is exactly the single-centre-sample behaviour, kept
                addressable so the difference can be measured rather than argued.

        Returns:
            The peak curvature of each face and where it occurs, in the order named.

        Raises:
            PysmeshError: On an empty ``face_ids``, ``samples`` below 1, or an id that is
                dead, is not a face, or was split.
        """
        raw = self._s.curvature(_ids(face_ids), samples)
        return CurvatureTable(
            ids=cast("NDArray[np.int64]", raw["ids"]),
            k_max=cast("NDArray[np.float64]", raw["k_max"]),
            uv=cast("NDArray[np.float64]", raw["uv"]),
            xyz=cast("NDArray[np.float64]", raw["xyz"]),
            samples_used=cast("NDArray[np.int64]", raw["samples_used"]),
        )

    def project_on_face(self, face_id: EntityId, points: Points) -> Projection:
        """Closest point on a face's surface to each of the given points.

        Projection is onto the face's **underlying surface**, not onto its trimmed boundary,
        so the result may lie outside the face itself. That is OCCT's contract for this
        operation and it is the right one for parameter recovery; for a distance to the
        trimmed face, use the stateless :func:`shape_distance`.

        Args:
            face_id: A face entity id denoting exactly one face.
            points: (N, 3) query points.

        Returns:
            The closest points, their surface parameters, and the distances.

        Raises:
            PysmeshError: If the id is dead, is not a face, was split, if ``points`` is not
                (N, 3), or if OCCT finds no projection — which happens for a point on a
                surface of revolution's own axis, where no nearest point is unique.
        """
        return _projection(
            self._s.project_on_face(int(face_id), _points("points", points))
        )

    def entities_in_box(
        self,
        kind: EntityKind,
        minimum: Vec3,
        maximum: Vec3,
        *,
        strict: bool = False,
    ) -> NDArray[np.int64]:
        """Live entities of one kind whose bounding box meets the given box.

        A bounding-box test, so it over-selects: an entity whose box overlaps but whose
        geometry does not is returned. That is the useful contract for a broad phase — narrow
        it with an exact test on the far smaller result.

        Args:
            kind: The entity kind to search.
            minimum: The query box's minimum corner.
            maximum: The query box's maximum corner. Each component must be >= ``minimum``'s.
            strict: Require the entity's box to lie **inside** the query box, rather than
                merely overlap it.

        Returns:
            (N,) int64 entity ids, ascending.

        Raises:
            PysmeshError: If any component of ``maximum`` is below ``minimum``'s.
        """
        xmin, ymin, zmin = minimum
        xmax, ymax, zmax = maximum
        return self._s.entities_in_box(
            str(kind), xmin, ymin, zmin, xmax, ymax, zmax, strict
        )

    def contains(
        self,
        solid_ids: Sequence[EntityId],
        points: Points,
        *,
        tol: float = _DEFAULT_CLASSIFY_TOL,
    ) -> NDArray[np.bool_]:
        """Whether each point lies strictly inside each named solid.

        Strictly inside only: a point within ``tol`` of the boundary counts as *on* it and
        reads ``False``. That is the right contract for choosing a seed point for a volume,
        where a point on the wall is not in the volume.

        Args:
            solid_ids: Solid entity ids, each denoting exactly one solid.
            points: (N, 3) query points.
            tol: Half-width of the boundary band, in model units (> 0).

        Returns:
            (S, N) bool — row ``i`` answers for ``solid_ids[i]``.

        Raises:
            PysmeshError: On an empty ``solid_ids``, a non-positive ``tol``, a ``points``
                array that is not (N, 3), or an id that is dead, is not a solid, or was
                split.
        """
        return self._s.contains(_ids(solid_ids), _points("points", points), tol)
