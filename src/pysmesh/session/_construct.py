"""pySMESH session — primitives, curve and surface construction, sweeps.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import (
    CancelPredicate,
    EntityId,
    HistoryDelta,
    Points,
    ProgressCallback,
    Vec3,
    _DEFAULT_BSPLINE_DEGREE,
    _DEFAULT_FIT_TOL,
    _FULL_TURN,
    _ORIGIN,
    _SPLINE_DEGREE_MAX,
    _SPLINE_DEGREE_MIN,
    _Z_AXIS,
    _delta,
    _ids,
    _points,
)


class _ConstructOps(_SessionBase):
    """Primitives, curve and surface construction, sweeps."""

    __slots__ = ()

    def add_brep(
        self,
        data: bytes,
        *,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Import BREP bytes as one or more new bodies.

        Args:
            data: A shape as BREP bytes (any OCCT ``BRepTools::Write`` output).
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the read runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the read. It then
                raises :class:`PysmeshCancelled`, and the session is left exactly as it was.

        Returns:
            The delta; every entity of the imported shape is newly issued.

        Raises:
            PysmeshError: On a malformed BREP or a null shape.
        """
        return _delta(self._s.add_brep(data, progress, cancel))

    def add_box(
        self,
        dx: float,
        dy: float,
        dz: float,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> HistoryDelta:
        """Add an axis-aligned box.

        Args:
            dx: Extent along +x (> 0).
            dy: Extent along +y (> 0).
            dz: Extent along +z (> 0).
            origin: The box's minimum corner.

        Returns:
            The delta; the box's solid, faces, edges and vertices are all newly issued.

        Raises:
            PysmeshError: If any extent is not strictly positive.
        """
        ox, oy, oz = origin
        return _delta(self._s.add_box(dx, dy, dz, ox, oy, oz))

    def add_cylinder(
        self,
        radius: float,
        height: float,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> HistoryDelta:
        """Add a circular cylinder.

        Args:
            radius: Cylinder radius (> 0).
            height: Extent along ``axis`` (> 0).
            origin: Centre of the base circle.
            axis: Direction of the cylinder's axis; must be non-zero.

        Returns:
            The delta; every entity of the cylinder is newly issued.

        Raises:
            PysmeshError: On a non-positive radius or height, or a zero axis.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        return _delta(self._s.add_cylinder(radius, height, ox, oy, oz, ax, ay, az))

    def add_cone(
        self,
        radius1: float,
        radius2: float,
        height: float,
        origin: Vec3 = _ORIGIN,
        axis: Vec3 = _Z_AXIS,
        *,
        angle_rad: float = _FULL_TURN,
    ) -> HistoryDelta:
        """Add a circular cone or truncated cone.

        Args:
            radius1: Radius of the base circle, at ``origin`` (>= 0).
            radius2: Radius of the top circle, at ``height`` along ``axis`` (>= 0). Zero
                gives a sharp apex. At least one radius must be positive.
            height: Extent along ``axis`` (> 0).
            origin: Centre of the base circle.
            axis: Direction of the cone's axis; must be non-zero.
            angle_rad: Sweep about the axis, in ``(0, 2*pi]``. Less than a full turn gives
                a wedge of the cone.

        Returns:
            The delta; every entity of the cone is newly issued.

        Raises:
            PysmeshError: On a negative radius, two zero radii, a non-positive height, a
                zero axis, or a sweep outside ``(0, 2*pi]``.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        return _delta(
            self._s.add_cone(radius1, radius2, height, ox, oy, oz, ax, ay, az, angle_rad)
        )

    def add_sphere(
        self,
        radius: float,
        centre: Vec3 = _ORIGIN,
        axis: Vec3 = _Z_AXIS,
        *,
        angle_rad: float = _FULL_TURN,
    ) -> HistoryDelta:
        """Add a sphere.

        Args:
            radius: Sphere radius (> 0).
            centre: Centre of the sphere.
            axis: Direction of the pole axis; must be non-zero.
            angle_rad: Sweep about the pole axis, in ``(0, 2*pi]``. Less than a full turn
                gives a lune.

        Returns:
            The delta; every entity of the sphere is newly issued.

        Raises:
            PysmeshError: On a non-positive radius, a zero axis, or a sweep outside
                ``(0, 2*pi]``.
        """
        cx, cy, cz = centre
        ax, ay, az = axis
        return _delta(self._s.add_sphere(radius, cx, cy, cz, ax, ay, az, angle_rad))

    def add_torus(
        self,
        radius1: float,
        radius2: float,
        origin: Vec3 = _ORIGIN,
        axis: Vec3 = _Z_AXIS,
        *,
        angle_rad: float = _FULL_TURN,
    ) -> HistoryDelta:
        """Add a torus.

        Args:
            radius1: Ring radius, from the axis to the tube centre (> 0).
            radius2: Tube radius (> 0, and smaller than ``radius1``).
            origin: Centre of the ring.
            axis: Direction of the ring's axis; must be non-zero.
            angle_rad: Sweep about the axis, in ``(0, 2*pi]``.

        Returns:
            The delta; every entity of the torus is newly issued.

        Raises:
            PysmeshError: On a non-positive radius, a tube radius at or above the ring
                radius (which self-intersects), a zero axis, or a sweep outside
                ``(0, 2*pi]``.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        return _delta(
            self._s.add_torus(radius1, radius2, ox, oy, oz, ax, ay, az, angle_rad)
        )

    def add_wedge(
        self,
        dx: float,
        dy: float,
        dz: float,
        ltx: float,
        origin: Vec3 = _ORIGIN,
        axis: Vec3 = _Z_AXIS,
    ) -> HistoryDelta:
        """Add a right angular wedge: a box narrowed along x at its far y face.

        Args:
            dx: Extent along local +x at ``y = 0`` (> 0).
            dy: Extent along local +y (> 0).
            dz: Extent along local +z (> 0).
            ltx: Extent along local +x at ``y = dy`` (>= 0). Zero gives a knife edge;
                ``ltx == dx`` gives a plain box.
            origin: The wedge's local origin.
            axis: Direction of the local +z; must be non-zero.

        Returns:
            The delta; every entity of the wedge is newly issued.

        Raises:
            PysmeshError: On a non-positive extent, a negative ``ltx``, or a zero axis.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        return _delta(self._s.add_wedge(dx, dy, dz, ltx, ox, oy, oz, ax, ay, az))

    # ---- construction geometry -------------------------------------------------------- #
    #
    # The registry tracks solids, faces, edges and vertices only, so a wire carries no id of
    # its own: it is named through the ids of its edges. Every operation that consumes a
    # profile resolves the entities it is given to the single body that owns them.

    def add_vertex(self, point: Vec3) -> HistoryDelta:
        """Add a standalone vertex body.

        This is the only construction that adds a point to the model as an entity in its own
        right rather than as the boundary of something else. The vertex is a body like any
        other: it carries an :data:`EntityId`, it moves under a transform, it is swept to an
        edge by :meth:`extrude` or :meth:`revolve`, and :meth:`remove` drops it.

        Args:
            point: Position of the vertex.

        Returns:
            The delta; the vertex is newly issued.
        """
        x, y, z = point
        return _delta(self._s.add_vertex(x, y, z))

    def add_line(self, start: Vec3, end: Vec3) -> HistoryDelta:
        """Add a straight edge between two points.

        Args:
            start: First point.
            end: Second point; must not coincide with ``start``.

        Returns:
            The delta; the edge and its two vertices are newly issued.

        Raises:
            PysmeshError: If the two points coincide.
        """
        x1, y1, z1 = start
        x2, y2, z2 = end
        return _delta(self._s.add_line(x1, y1, z1, x2, y2, z2))

    def add_arc(self, start: Vec3, through: Vec3, end: Vec3) -> HistoryDelta:
        """Add a circular arc through three points.

        Args:
            start: Start of the arc.
            through: A point the arc passes through.
            end: End of the arc.

        Returns:
            The delta; the edge and its vertices are newly issued.

        Raises:
            PysmeshError: If no circle passes through the three points — they are collinear
                or two of them coincide.
        """
        x1, y1, z1 = start
        x2, y2, z2 = through
        x3, y3, z3 = end
        return _delta(self._s.add_arc(x1, y1, z1, x2, y2, z2, x3, y3, z3))

    def add_circle(self, centre: Vec3, normal: Vec3, radius: float) -> HistoryDelta:
        """Add a full circular edge.

        Args:
            centre: Centre of the circle.
            normal: Normal of the circle's plane; must be non-zero.
            radius: Circle radius (> 0).

        Returns:
            The delta; the closed edge and its single seam vertex are newly issued.

        Raises:
            PysmeshError: On a non-positive radius or a zero normal.
        """
        cx, cy, cz = centre
        nx, ny, nz = normal
        return _delta(self._s.add_circle(cx, cy, cz, nx, ny, nz, radius))

    def add_ellipse(
        self,
        centre: Vec3,
        normal: Vec3,
        rx: float,
        ry: float,
        *,
        x_dir: Vec3 | None = None,
    ) -> HistoryDelta:
        """Add a full elliptical edge.

        The ellipse is exact geometry, not an approximation. Close it with :meth:`make_face`
        for an elliptical disk. Equal radii give a circle-shaped ellipse; :meth:`add_circle`
        is the exact circle constructor, and its edge reports the curve type ``Circle``.

        Unlike a circle, an ellipse has an in-plane orientation: the normal fixes the plane
        but not where the major axis points in it. ``x_dir`` names that direction when it
        matters.

        Args:
            centre: Centre of the ellipse.
            normal: Normal of the ellipse's plane; must be non-zero.
            rx: Radius along the plane's first in-plane direction (> 0).
            ry: Radius along the plane's second in-plane direction (> 0). ``ry > rx`` is not
                an error — it is the same ellipse with its major axis along the second
                direction.
            x_dir: The plane's first in-plane direction. Must be non-zero and not parallel
                to ``normal``; its component along ``normal`` is removed. ``None`` leaves
                OCCT to derive it from the normal, which for ``normal = (0, 0, 1)`` gives
                ``x_dir = (1, 0, 0)``.

        Returns:
            The delta; the closed edge and its single seam vertex are newly issued.

        Raises:
            PysmeshError: On a non-positive radius, a zero normal, or an ``x_dir`` that is
                zero or parallel to ``normal``.
        """
        cx, cy, cz = centre
        nx, ny, nz = normal
        return _delta(self._s.add_ellipse(cx, cy, cz, nx, ny, nz, rx, ry, x_dir))

    def add_polyline(self, points: Points, *, closed: bool = False) -> HistoryDelta:
        """Add a polyline wire through the given points.

        Consecutive segments share one vertex by construction, so a later
        :meth:`make_wire` or :meth:`make_face` over these edges keeps every edge id.

        Args:
            points: (N, 3) points, N >= 2.
            closed: Close the polyline back to the first point.

        Returns:
            The delta; every edge and vertex of the wire is newly issued.

        Raises:
            PysmeshError: On fewer than two points, a malformed array, or two coincident
                consecutive points.
        """
        return _delta(self._s.add_polyline(_points("points", points), closed))

    def add_spline(
        self,
        points: Points,
        *,
        degree_min: int = _SPLINE_DEGREE_MIN,
        degree_max: int = _SPLINE_DEGREE_MAX,
        tol: float = _DEFAULT_FIT_TOL,
    ) -> HistoryDelta:
        """Add a B-spline edge approximating a sequence of points.

        Args:
            points: (N, 3) points the curve passes through to within ``tol``, N >= 2.
            degree_min: Lowest degree OCCT may use (>= 1).
            degree_max: Highest degree OCCT may use (>= ``degree_min``).
            tol: Maximum distance from any input point to the fitted curve (> 0).

        Returns:
            The delta; the edge and its vertices are newly issued.

        Raises:
            PysmeshError: On fewer than two points, an invalid degree band, a non-positive
                tolerance, or a fit that does not converge.
        """
        return _delta(
            self._s.add_spline(_points("points", points), degree_min, degree_max, tol)
        )

    def add_bspline(
        self, poles: Points, *, degree: int = _DEFAULT_BSPLINE_DEGREE
    ) -> HistoryDelta:
        """Add a clamped B-spline edge over the given control points.

        The curve passes through the first and last pole and is pulled towards the others;
        use :meth:`add_spline` to pass through every point instead.

        Args:
            poles: (N, 3) control points, N >= 2.
            degree: Curve degree (>= 1), clamped to ``N - 1``.

        Returns:
            The delta; the edge and its vertices are newly issued.

        Raises:
            PysmeshError: On fewer than two poles or a degree below 1.
        """
        return _delta(self._s.add_bspline(_points("poles", poles), degree))

    def add_helix(
        self,
        centre: Vec3,
        axis: Vec3,
        diameter: float,
        pitch: float,
        turns: float,
        *,
        tol: float = _DEFAULT_FIT_TOL,
    ) -> HistoryDelta:
        """Add a helical wire.

        The helix is approximated by B-spline edges, one per turn, to within ``tol``; OCCT
        has no exact helical curve type in a B-rep edge.

        Args:
            centre: Point on the axis where the helix starts.
            axis: Direction of the helix axis; must be non-zero.
            diameter: Diameter of the helix (> 0).
            pitch: Axial rise per turn (> 0).
            turns: Number of turns (> 0); need not be whole.
            tol: Approximation tolerance (> 0).

        Returns:
            The delta; every edge and vertex of the wire is newly issued.

        Raises:
            PysmeshError: On a non-positive parameter, a zero axis, or an approximation
                that does not converge.
        """
        cx, cy, cz = centre
        ax, ay, az = axis
        return _delta(
            self._s.add_helix(cx, cy, cz, ax, ay, az, diameter, pitch, turns, tol)
        )

    def add_rectangle(
        self, origin: Vec3, normal: Vec3, dx: float, dy: float
    ) -> HistoryDelta:
        """Add a planar rectangular face.

        Args:
            origin: Corner of the rectangle, and the plane's origin.
            normal: Normal of the plane; must be non-zero. The rectangle extends along the
                plane's own two in-plane directions, which OCCT derives from the normal.
            dx: Extent along the plane's first in-plane direction (> 0).
            dy: Extent along the plane's second in-plane direction (> 0).

        Returns:
            The delta; the face, its four edges and its four vertices are newly issued.

        Raises:
            PysmeshError: On a non-positive extent or a zero normal.
        """
        ox, oy, oz = origin
        nx, ny, nz = normal
        return _delta(self._s.add_rectangle(ox, oy, oz, nx, ny, nz, dx, dy))

    def make_wire(self, edge_ids: Sequence[EntityId]) -> HistoryDelta:
        """Join loose edges and wires into one wire, consuming them.

        OCCT rebuilds any edge whose end vertex is only *coincident* with the wire's rather
        than *shared* with it. A rebuilt edge is a new entity, so its old id dies — the
        delta says which. Edges that already share vertices, such as those from
        :meth:`add_polyline` or from :meth:`section`, keep their ids.

        Args:
            edge_ids: Edges to join. Each must belong to a loose-edge or wire body, never to
                a solid, because the operation consumes the bodies it is given.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not an edge, if an edge belongs to a solid
                or a face, or if the edges do not form a connected wire.
        """
        return _delta(self._s.make_wire(_ids(edge_ids)))

    def make_face(self, edge_ids: Sequence[EntityId]) -> HistoryDelta:
        """Build a planar face bounded by the named edges, consuming them.

        A non-planar boundary raises rather than being approximated; :meth:`make_filling`
        is the operation for that case.

        Args:
            edge_ids: Edges bounding the face, forming one closed planar loop.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not an edge, if an edge belongs to a solid
                or a face, or if the edges do not bound a closed planar loop.
        """
        return _delta(self._s.make_face(_ids(edge_ids)))

    def make_filling(
        self,
        edge_ids: Sequence[EntityId],
        *,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Fill a possibly non-planar boundary with a surface, consuming the boundary.

        The surface is an approximation, so the resulting face's edges are new geometry and
        the boundary edges' ids die.

        Args:
            edge_ids: Edges bounding the surface.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If an id is dead or is not an edge, if an edge belongs to a solid
                or a face, or if OCCT cannot fill the boundary.
        """
        return _delta(self._s.make_filling(_ids(edge_ids), progress, cancel))

    # ---- sweeps ----------------------------------------------------------------------- #

    def extrude(self, entities: Sequence[EntityId], vector: Vec3) -> HistoryDelta:
        """Sweep a profile linearly, consuming it.

        A sweep raises the profile's dimension: an edge becomes a face, a wire a shell, a
        face a solid. The profile survives inside the result, so its entity ids carry
        through, and each wall the sweep generates is named against the profile edge it
        came from.

        Args:
            entities: Entities identifying the profile. They must all belong to one body,
                which is the body swept.
            vector: The extrusion vector; must be non-zero.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: On a zero vector, a dead id, entities spanning two bodies, a
                profile body that is a solid or a compound, or a sweep OCCT cannot build.
        """
        vx, vy, vz = vector
        return _delta(self._s.extrude(_ids(entities), vx, vy, vz))

    def revolve(
        self,
        entities: Sequence[EntityId],
        origin: Vec3,
        axis: Vec3,
        angle_rad: float = _FULL_TURN,
    ) -> HistoryDelta:
        """Sweep a profile about an axis, consuming it.

        Args:
            entities: Entities identifying the profile. They must all belong to one body.
            origin: A point on the rotation axis.
            axis: Direction of the rotation axis; must be non-zero.
            angle_rad: Sweep angle in ``(0, 2*pi]``.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: On a zero axis, a sweep outside ``(0, 2*pi]``, a dead id, entities
                spanning two bodies, or a profile that crosses the axis.
        """
        ox, oy, oz = origin
        ax, ay, az = axis
        return _delta(self._s.revolve(_ids(entities), ox, oy, oz, ax, ay, az, angle_rad))

    def pipe(
        self,
        spine: Sequence[EntityId],
        profile: Sequence[EntityId],
        *,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Sweep a profile along a spine, consuming both.

        Args:
            spine: Entities identifying the spine body, which must be a wire or one edge.
            profile: Entities identifying the profile body.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If either selection spans two bodies, if they name the same body,
                if the spine is not a wire or edge, or if OCCT cannot sweep.
        """
        return _delta(self._s.pipe(_ids(spine), _ids(profile), progress, cancel))

    def pipe_shell(
        self,
        spine: Sequence[EntityId],
        profile: Sequence[EntityId],
        *,
        frenet: bool = False,
        solid: bool = True,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Sweep a profile along a spine with an explicit frame law, consuming both.

        This is the general sweep. :meth:`pipe` is the simpler one; use this when the frame
        law matters or a closed solid is wanted.

        Args:
            spine: Entities identifying the spine body, which must be a wire or one edge.
            profile: Entities identifying the profile body, which must be a wire or one
                edge.
            frenet: Use the Frenet frame rather than OCCT's corrected Frenet. The corrected
                frame avoids the twist a Frenet frame develops at an inflection, so it is
                the default.
            solid: Cap the swept shell into a solid.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: If either selection spans two bodies, if they name the same body,
                if either body is not a wire or edge, if ``solid`` is asked for on a shell
                that does not close, or if OCCT cannot sweep.
        """
        return _delta(
            self._s.pipe_shell(_ids(spine), _ids(profile), frenet, solid, progress, cancel)
        )

    def thru_sections(
        self,
        sections: Sequence[Sequence[EntityId]],
        *,
        solid: bool = True,
        ruled: bool = True,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> HistoryDelta:
        """Loft through an ordered list of section wires, consuming all of them.

        Args:
            sections: One entity-id group per section, in sweep order. Each group must
                resolve to one wire or single-edge body, and no body may appear twice.
            solid: Cap the ends and return a solid rather than a shell.
            ruled: Join consecutive sections with ruled surfaces. ``False`` fits one smooth
                surface through all of them.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The delta for this operation.

        Raises:
            PysmeshError: On fewer than two sections, a repeated body, a section that is
                not a wire or edge, or a loft OCCT cannot build.
        """
        return _delta(
            self._s.thru_sections(
                [_ids(s) for s in sections], solid, ruled, progress, cancel
            )
        )
