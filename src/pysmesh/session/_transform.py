# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""pySMESH session — the transforms, and copy.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import EntityId, HistoryDelta, Vec3, _ORIGIN, _delta, _ids


class _TransformOps(_SessionBase):
    """The transforms, and copy."""

    __slots__ = ()

    def translate(
        self,
        offset: tuple[float, float, float],
        entities: Sequence[EntityId] | None = None,
    ) -> HistoryDelta:
        """Translate the whole model, or the bodies owning the named entities.

        A rigid transform changes only a shape's location, so **every entity id survives**
        and no history is involved. The session asserts that identity rather than assuming
        it: if the transform ever copied the geometry, the operation raises instead of
        silently orphaning every id in the model.

        Args:
            offset: The translation vector.
            entities: Entities whose owning bodies move. ``None`` moves the whole model.

        Returns:
            The delta; ``created`` and ``deleted`` are always empty.

        Raises:
            PysmeshError: If ``entities`` is given but empty, if an id is dead, or if the
                selected bodies share sub-shapes with bodies that stay put.
        """
        dx, dy, dz = offset
        ids = None if entities is None else _ids(entities)
        return _delta(self._s.translate(dx, dy, dz, ids))

    def rotate(
        self,
        origin: tuple[float, float, float],
        axis: tuple[float, float, float],
        angle_rad: float,
        entities: Sequence[EntityId] | None = None,
    ) -> HistoryDelta:
        """Rotate the whole model, or the bodies owning the named entities.

        Carries the same location-only identity property as :meth:`translate`.

        Args:
            origin: A point on the rotation axis.
            axis: Direction of the rotation axis; must be non-zero.
            angle_rad: Rotation angle in radians, right-handed about ``axis``.
            entities: Entities whose owning bodies rotate. ``None`` rotates the whole model.

        Returns:
            The delta; ``created`` and ``deleted`` are always empty.

        Raises:
            PysmeshError: On a zero axis, an empty ``entities``, a dead id, or a selection
                that shares sub-shapes with bodies that stay put.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        ids = None if entities is None else _ids(entities)
        return _delta(self._s.rotate(ox, oy, oz, ax, ay, az, angle_rad, ids))

    def mirror(
        self,
        point: Vec3,
        normal: Vec3,
        entities: Sequence[EntityId] | None = None,
    ) -> HistoryDelta:
        """Reflect the whole model, or the bodies owning the named entities, in a plane.

        A plane reflection has determinant -1, so unlike :meth:`translate` and
        :meth:`rotate` it is **not** a location-only change: OCCT rebuilds the geometry.
        Every entity id still survives, one-to-one, carried by the transform's own history
        rather than by shape identity.

        Args:
            point: A point on the mirror plane.
            normal: Normal of the mirror plane; must be non-zero.
            entities: Entities whose owning bodies are mirrored. ``None`` mirrors the whole
                model.

        Returns:
            The delta; every surviving id appears in ``modified``, and ``created`` and
            ``deleted`` are empty.

        Raises:
            PysmeshError: On a zero normal, an empty ``entities``, a dead id, or a
                selection that shares sub-shapes with bodies that stay put.
        """
        px, py, pz = point
        nx, ny, nz = normal
        ids = None if entities is None else _ids(entities)
        return _delta(self._s.mirror(px, py, pz, nx, ny, nz, ids))

    def scale(
        self,
        factors: float | Vec3,
        centre: Vec3 = _ORIGIN,
        entities: Sequence[EntityId] | None = None,
    ) -> HistoryDelta:
        """Scale the whole model, or the bodies owning the named entities.

        A uniform factor keeps analytic surfaces analytic — a scaled cylinder is still a
        cylinder. An anisotropic one cannot, so OCCT re-approximates every non-planar
        surface as a B-spline. Either way the geometry is rebuilt and every entity id
        survives one-to-one through the transform's history.

        Args:
            factors: One factor for all three axes, or one per axis. All must be > 0.
            centre: The fixed point of the scaling.
            entities: Entities whose owning bodies are scaled. ``None`` scales the whole
                model.

        Returns:
            The delta; every surviving id appears in ``modified``, and ``created`` and
            ``deleted`` are empty.

        Raises:
            PysmeshError: On a non-positive factor, a uniform factor of exactly 1 (a no-op
                that would still consume an operation index), an empty ``entities``, a dead
                id, or a selection that shares sub-shapes with bodies that stay put.
        """
        if isinstance(factors, (int, float)):
            sx = sy = sz = float(factors)
        else:
            sx, sy, sz = factors
        cx, cy, cz = centre
        ids = None if entities is None else _ids(entities)
        return _delta(self._s.scale(sx, sy, sz, cx, cy, cz, ids))

    def position_along_path(
        self,
        path_edge: EntityId,
        distance: float,
        entities: Sequence[EntityId],
        *,
        reverse: bool = False,
    ) -> HistoryDelta:
        """Move the bodies owning the named entities along a path edge.

        The move is rigid. It takes the path's frame at its start onto its frame at
        ``distance``, a fraction of the path's length. The frame is OCCT's corrected
        Frenet frame (``GeomFill_CorrectedFrenet``), as GEOM's ``PositionAlongPath`` uses
        it: its origin is the curve point, its z axis the unit tangent and its x axis the
        principal normal. On a twisted curve OCCT turns the normal about the tangent to
        keep the frame's twist small. The start is the path edge's first vertex in its own
        orientation, as for :meth:`point_at_length`. So a body at the start of the path
        arrives at the same place relative to the frame at ``distance``: on a straight
        path the move is a translation, on a circular arc a rotation about the arc's
        axis.

        Like :meth:`translate`, the move changes only locations, so **every entity id
        survives**. The path edge stays where it is.

        Args:
            path_edge: The edge to move along.
            distance: The fraction of the path's length, in ``[0, 1]``, measured by arc
                length.
            entities: Entities whose owning bodies move; at least one. No other body
                moves.
            reverse: Measure ``distance`` from the path's far end, as GEOM does: the
                bodies move to the fraction ``1 - distance``.

        Returns:
            The delta; ``created`` and ``deleted`` are always empty.

        Raises:
            PysmeshError: If ``entities`` is empty, if an id is dead, if ``path_edge`` is
                not one edge or has no length, if ``distance`` is not finite or is outside
                ``[0, 1]``, if a moved body owns the path edge (the message names the
                body), if the frame at the fraction is the start frame (the move would
                change nothing), or if the moved bodies share sub-shapes with bodies that
                stay put.
        """
        return _delta(
            self._s.position_along_path(
                int(path_edge), distance, _ids(entities), reverse
            )
        )

    def copy(self, entities: Sequence[EntityId]) -> HistoryDelta:
        """Duplicate the bodies owning the named entities.

        The originals keep every id; every entity of every duplicate is a **new** identity.
        A copy is deliberately not related to its original by history: relating them would
        move the original's id onto the duplicate.

        Args:
            entities: Entities whose owning bodies are duplicated. At least one.

        Returns:
            The delta; ``created`` holds the duplicates' ids and everything else is empty.

        Raises:
            PysmeshError: On an empty selection, a dead id, or an entity that belongs to no
                body.
        """
        return _delta(self._s.copy(_ids(entities)))
