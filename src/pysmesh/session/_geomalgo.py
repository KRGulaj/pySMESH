# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""pySMESH session — GEOM algorithms copied into pySMESH: the 1-D fillet, the smoothing surface.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._base import _SessionBase
from ._types import EntityId, HistoryDelta, Points, _delta, _ids, _points


class _GeomAlgoOps(_SessionBase):
    """GEOM algorithms copied into pySMESH: curves and surfaces."""

    __slots__ = ()

    def fillet_1d(
        self,
        edges: Sequence[EntityId],
        vertices: Sequence[EntityId],
        radius: float,
        *,
        ignore_secant_vertices: bool = True,
    ) -> HistoryDelta:
        """Round the corners of a planar wire with circular arcs of one radius.

        GEOM's ``MakeFillet1D``: the algorithm class ``GEOMImpl_Fillet1d`` and the logic
        of ``GEOMImpl_Fillet1dDriver``, copied into pySMESH. At each corner the two edges
        that meet there are projected onto the plane of the corner, and the algorithm
        searches the first edge for the point whose circle centre lies at ``radius`` from
        the second edge (a Newton search). The arc between the two tangent points replaces
        the corner, and both edges are trimmed to the arc's ends. Lines, circles and other
        planar curves are all accepted.

        The geometry is the closed form of a fillet: between two lines at the angle
        ``theta``, the arc's centre lies on the bisector at ``radius / sin(theta / 2)``
        from the corner, its tangent points at ``radius / tan(theta / 2)`` along each
        edge, and its length is ``radius * (pi - theta)``. The search stops when the
        squared-distance residual satisfies ``|sqrt(|d^2 - r^2| + r^2) - r| < 1e-8``
        (``GEOMImpl_Fillet1dPoint::HasSolution``), so the tangent points are found to
        about 1e-8 of a model unit; the arc's centre and radius are exact for those points.

        **Corners.** A corner is a vertex that joins two edges of the wire, where the two
        edges do not lie on one line through it and do not continue each other with a
        common tangent (GEOM's ``AreEdgesC1``). An empty ``vertices`` rounds every corner,
        as GEOM does; the ends of an open wire and the joins of collinear or tangent edges
        are then left as they are.

        **ignore_secant_vertices.** GEOM's ``doIgnoreSecantVertices``, with GEOM's default.
        When a corner's arc does not fit on the edges next to it, the edges of the wire
        that continue each other with a common tangent (collinear lines, arcs of one
        circle) are joined into one edge each, which allows a larger radius, and the
        fillet is tried again (GEOM's ``FuseCollinearEdges``). With ``False`` the arc must
        fit on the edges as they are.

        **Identity.** The wire body is replaced by the filleted wire. Each trimmed edge
        keeps its id (it is in ``modified``); edges joined by the fallback all keep their
        ids on the one edge (it is in ``merged``); an edge an arc uses up entirely dies.
        Each arc and its two end vertices get new ids whose role is ``GENERATED`` and whose
        origin is the corner vertex, which dies. The other vertices keep their ids; a
        vertex between two joined edges dies.

        **Differences from GEOM.** GEOM leaves a corner where no arc fits as it is, and
        fails only when no corner at all is rounded. Here such a corner is refused, naming
        the vertex, and nothing is committed. A named vertex that is not a corner is
        refused too, where GEOM skips it. The wire must be planar: GEOM fits one plane to
        each corner.

        Args:
            edges: Every edge of one free wire body, in any order. The wire may be open
                or closed.
            vertices: Corner vertices of that wire to round. Empty rounds every corner.
            radius: The fillet radius, at least 1e-7 (``Precision::Confusion``).
            ignore_secant_vertices: Join the edges that continue each other with a common
                tangent when an arc does not fit otherwise.

        Returns:
            The delta: the arcs and their vertices in ``created``, the corner vertices in
            ``deleted``, the trimmed edges in ``modified``.

        Raises:
            PysmeshError: If an id is dead, is not an edge or vertex of the right kind, or
                was split; if the edges are not every edge of one free wire body; if the
                radius is below 1e-7; if a named vertex is not a corner of the wire or is
                named twice; if the wire has no corner or is not planar; or if no arc of
                the radius fits at a corner (the message names the vertex). Nothing is
                committed then.
        """
        return _delta(
            self._s.fillet_1d(
                _ids(edges), _ids(vertices), float(radius), ignore_secant_vertices
            )
        )

    def smoothing_surface(
        self,
        points: Points,
        *,
        max_points: int = 2,
        max_degree: int = 8,
        max_distance: float = 0.0,
    ) -> HistoryDelta:
        """Add a free face whose surface approximates a cloud of points.

        GEOM's ``MakeSmoothingSurface``: the geometry code of
        ``AdvancedEngine_SmoothingSurfaceDriver``, copied into pySMESH. The points'
        average plane (``GeomPlate_BuildAveragePlane``) is the initial surface; a plate
        surface through every point (``GeomPlate_BuildPlateSurface``, degree 3, each
        point a G0 constraint) deforms it; and ``GeomPlate_MakeApprox`` approximates the
        plate by a B-spline surface (3-D tolerance 1e-4, continuity C1). The face is
        bounded by the plate's parameter range, the box of the points' projections on the
        average plane. The defaults are ``AdvancedGEOMBuilder.MakeSmoothingSurface``'s.

        Points on a plane give a plate with no deformation: the face lies on that plane.

        The face is one new body; every id of it is ``CONSTRUCTED``. The order of the
        points is kept, so identical arrays give identical bytes (GEOM collects them
        from a map of vertices, whose order follows shape addresses).

        **Accuracy.** The plate passes through the points within its own error ``e``
        (``GeomPlate_BuildPlateSurface::G0Error``). The B-spline approximates the plate
        within 1e-4 where OCCT can reach it, and its deviation from the plate at the
        points is held below ``max(1e-4, 10 * max_distance)`` (``GeomPlate_MakeApprox``
        with its order-0 criterion).

        Args:
            points: (N, 3) points, N >= 3, not all on one line.
            max_points: The maximum number of Bezier pieces of the B-spline surface
                (GEOM's ``theNbMax``), at least 1.
            max_degree: The maximum degree of the B-spline (GEOM's ``theDegMax``), at
                least 1.
            max_distance: GEOM's ``theDMax``, the tolerance the criterion at the points
                is built from. ``0`` takes GEOM's computed value, ``max(1e-4, 10 * e)``.

        Returns:
            The delta: the face, its edges and vertices in ``created``.

        Raises:
            PysmeshError: If ``points`` is not an (N, 3) array, has fewer than 3 points,
                holds a NaN or an infinity, or its points lie on one line; if
                ``max_points`` or ``max_degree`` is below 1 or ``max_distance`` is
                negative; or if OCCT builds no face.
        """
        return _delta(
            self._s.smoothing_surface(
                _points("points", points),
                int(max_points),
                int(max_degree),
                float(max_distance),
            )
        )
