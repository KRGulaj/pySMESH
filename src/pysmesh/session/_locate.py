# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""pySMESH session — locating entities with GEOM's classifiers.

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
    BoxRegion,
    CancelPredicate,
    CylinderRegion,
    EntityId,
    EntityKind,
    PlaneRegion,
    ProgressCallback,
    Region,
    RegionState,
    SnapshotMark,
    SphereRegion,
    _ids,
)


def _region_values(region: Region) -> tuple[str, list[float], int | None]:
    """The native region type, its numbers and its solid id, for one region argument."""
    if isinstance(region, PlaneRegion):
        return "plane", [*map(float, region.origin), *map(float, region.normal)], None
    if isinstance(region, CylinderRegion):
        values = [*map(float, region.origin), *map(float, region.direction)]
        return "cylinder", [*values, float(region.radius)], None
    if isinstance(region, SphereRegion):
        return "sphere", [*map(float, region.centre), float(region.radius)], None
    if isinstance(region, BoxRegion):
        return "box", [*map(float, region.minimum), *map(float, region.maximum)], None
    if isinstance(region, (int, np.integer)) and not isinstance(region, bool):
        return "solid", [], int(region)
    raise PysmeshError(
        "Session.entities_on: the region is a "
        f"{type(region).__name__}, not a PlaneRegion, CylinderRegion, SphereRegion, "
        "BoxRegion or the EntityId of a solid."
    )


def _ordinals(values: Sequence[int]) -> list[int]:
    """Validate and widen a caller's ordinal sequence for the native call."""
    out: list[int] = []
    for v in values:
        if not isinstance(v, (int, np.integer)) or isinstance(v, bool):
            raise PysmeshError(
                f"Session.in_place: ordinals must be integers (got {v!r})."
            )
        out.append(int(v))
    return out


class _LocateOps(_SessionBase):
    """Locating entities: on a region, in place of old sub-shapes, the same as another."""

    __slots__ = ()

    def entities_on(
        self,
        kind: EntityKind,
        region: Region,
        state: RegionState,
        *,
        tolerance: float | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> NDArray[np.int64]:
        """Live entities of one kind that lie on, inside or outside a region.

        GEOM's ``GetShapesOnPlane``, ``GetShapesOnCylinder``, ``GetShapesOnSphere``,
        ``GetShapesOnBox`` and ``GetShapesOnShape``, with the classifier settings GEOM
        gives them: ``GEOMAlgo_FinderShapeOn2`` with ``GEOMAlgo_ClsfSurf`` (plane,
        cylinder, sphere), ``GEOMAlgo_ClsfBox`` (box) or ``GEOMAlgo_ClsfSolid`` (a solid
        of the model). It classifies each vertex, and the inner nodes of each edge's and
        face's triangulation; a face or a solid is in a state when its boundary and its
        inner points all are.

        **The plane's sides.** GEOMAlgo takes the signed distance ``n . (p - origin)``
        along the plane normal: above the tolerance is "out", below minus the tolerance is
        "in". So "out" is the side the normal points to.

        **Sample points.** The finder samples the nodes of a triangulation. It runs on a
        private copy of the model, meshed as GEOM meshes it (``GEOMUtils::
        CheckTriangulation``: deflection 0.001 relative to the model's box, angle 0.349066
        rad), so the query writes nothing into the model's faces and its answer does not
        depend on an earlier :meth:`tessellate`. A face is decided by its sample points:
        a curved face that crosses the region between two of them can be reported on one
        side.

        **Split ids.** An id is in the state when every shape it denotes is: a split
        solid with pieces on both sides of a plane is neither "in" nor "out".

        **Progress.** The query copies the model, meshes the copy and runs the finder.
        ``progress`` gives the three 10 %, 60 % and 30 % of the range. A cancel stops the
        mesh, and stops the finder between two of its 32 runs: the finder runs on 32
        consecutive parts of the entities of ``kind``, which gives the answer of one run
        because the finder decides each entity from its own boundary and inner points.
        The copy runs to its end once started.

        Args:
            kind: The entity kind to search.
            region: A :class:`PlaneRegion`, :class:`CylinderRegion`, :class:`SphereRegion`
                or :class:`BoxRegion`, or the :data:`EntityId` of a live solid of the
                model that denotes one solid.
            state: ``"on"``, ``"in"``, ``"out"``, ``"on_in"`` (on or in) or ``"on_out"``
                (on or out), GEOM's ``ST_ON``, ``ST_IN``, ``ST_OUT``, ``ST_ONIN`` and
                ``ST_ONOUT``.
            tolerance: The distance within which a point is "on" the region, in model
                units (> 0). ``None`` takes GEOM's value for the region: the largest
                vertex tolerance of the model for a plane, a cylinder or a sphere; 1e-4
                for a box; the largest vertex tolerance bounded to ``[1e-7, 1e-4]`` for a
                solid.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the query runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the query. It then
                raises :class:`PysmeshCancelled`. The query changes nothing either way.

        Returns:
            (N,) int64 — the live ids of ``kind`` in the state, ascending. Empty when none
            is.

        Raises:
            PysmeshError: If the region is of another type, has a zero normal or
                direction, a radius that is not above zero, or a box extent that is not
                above zero; if the solid id is dead, is not a solid or was split; if
                ``state`` is not one of the five; if ``tolerance`` is not finite or not
                above zero; if the model has no face and no edge to mesh, or a void box
                (GEOM's "Cannot build triangulation on the shape"); or if GEOMAlgo
                reports a failure (its code and meaning are in the message).
            PysmeshCancelled: If ``cancel`` returned True.
        """
        name, values, solid = _region_values(region)
        return self._s.entities_on(
            str(kind), name, values, solid, state, tolerance, progress, cancel
        )

    def in_place(
        self,
        kind: EntityKind,
        source: bytes | SnapshotMark,
        old: Sequence[int] | Sequence[EntityId],
        *,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> tuple[NDArray[np.int64], ...]:
        """The live entities that now stand in place of old sub-shapes of one kind.

        GEOM's ``GetInPlace``: ``GEOMAlgo_GetInPlace`` with the current model as the shape
        searched and each old sub-shape in turn as the argument, built with GEOM's
        two-argument constructor, so its tolerances are GEOM's. The tolerance is the
        largest vertex tolerance of the model and of the old sub-shape, at least 1e-7; the
        mass tolerance is that times the largest side of their common box; the centre of
        mass tolerance is the tolerance. It finds the sub-shapes of the model that lie in
        the old sub-shape by geometry, so it answers across an operation that rebuilt or
        merged the shape: a face that a fillet trimmed, or a face that
        :meth:`unify_same_domain` merged with its neighbour.

        Two forms of ``source`` say which old sub-shapes are meant:

        * BREP bytes, such as :attr:`Handoff.brep`: ``old`` holds 0-based ordinals of
          ``kind`` in the per-kind traversal that :meth:`export_handoff` defines. Ordinal
          ``i`` is the sub-shape whose id the handoff lists at position ``i`` of its
          ``<kind>_id`` array.
        * A :data:`SnapshotMark`: ``old`` holds ids of ``kind`` that are live in that
          snapshot. A split id stands for all its shapes.

        Args:
            kind: The kind of the old sub-shapes, and of the entities found.
            source: The BREP bytes, or a snapshot mark that is not discarded.
            old: The old sub-shapes, as ordinals or as ids (see above). At least one.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the query runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the query. It then
                raises :class:`PysmeshCancelled`. The query changes nothing either way.

        Returns:
            One (N,) int64 array per entry of ``old``, in the same order: every live id of
            the current model that denotes a sub-shape GetInPlace finds for that entry,
            ascending. An empty array means nothing was found, which is an answer, not a
            failure.

        Raises:
            PysmeshError: If ``old`` is empty; with bytes, if they are not a BREP OCCT can
                read or an ordinal is out of range (the message gives the count); with a
                mark, if it is unknown or discarded, or an id is not live in that
                snapshot or is of another kind; or if GEOMAlgo_GetInPlace fails on an
                entry (the message names the entry and the code).
            PysmeshCancelled: If ``cancel`` returned True.
        """
        if isinstance(source, (bytes, bytearray, memoryview)):
            raw = self._s.in_place_brep(
                str(kind), bytes(source), _ordinals(old), progress, cancel
            )
        elif isinstance(source, (int, np.integer)) and not isinstance(source, bool):
            raw = self._s.in_place_snapshot(
                str(kind),
                int(source),
                _ids(cast("Sequence[EntityId]", old)),
                progress,
                cancel,
            )
        else:
            raise PysmeshError(
                "Session.in_place: the source is a "
                f"{type(source).__name__}, not BREP bytes or a SnapshotMark."
            )
        return tuple(raw)

    def find_same(
        self, entity: EntityId, within: Sequence[EntityId]
    ) -> NDArray[np.int64]:
        """Live entities that GEOM's ``GetSame`` rule calls the same as one entity.

        The rule is GEOM's ``GetSameIDs``, copied into pySMESH (``isSameEdge``,
        ``isSameFace``, ``isSameSolid``), with GEOM's tolerance 1e-7:

        * vertices: the points are within 1e-7;
        * edges: the end points coincide in either order, and the points at 1/3 and 2/3
          of the parameter range lie on the other edge's curve within its range;
        * faces: the same number of edges; the corners of the boxes of their vertices
          coincide; the surface points at 1/3 and 2/3 of the parameter box lie on the
          other surface within its parameter box; each edge has a same edge in the other
          face;
        * solids: the same number of faces; the corners of the vertex boxes coincide;
          each face has a same face in the other solid.

        The entity is often one of a separate body. If ``within`` names its own body, the
        entity itself is in the answer.

        Args:
            entity: A live entity of any kind that denotes one shape.
            within: Entities whose owning bodies are searched. At least one.

        Returns:
            (N,) int64 — every live id of the entity's kind, in those bodies, whose shape
            the rule calls the same, ascending. Empty when none is.

        Raises:
            PysmeshError: If an id is dead, if ``entity`` was split, or if ``within`` is
                empty or names an entity of no body.
        """
        return self._s.find_same(int(entity), _ids(within))
