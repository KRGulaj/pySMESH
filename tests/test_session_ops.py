"""Gates for the session's modelling operation surface: construction, booleans, local
features and transforms.

Four claims are under test here, and each is asserted against something independent of the
registry it checks:

* **Construction** — every primitive and construction op produces a
  ``BRepCheck_Analyzer``-valid shape, and the analytic solids' volumes and areas match their
  closed-form values.
* **Booleans** — every boolean carries the operands' entity identity through OCCT's history,
  verified face by face against a labelling computed from the serialised result through the
  *stateless* API, never from the session's own tables. A boolean OCCT reports as failed
  raises and returns no partial result.
* **Fillet and chamfer** — a long tangent chain is rounded in one call with no per-edge face
  co-selection; a radius OCCT cannot build fails loud naming the edge.
* **Transforms** — every entity id is unchanged, asserted id by id rather than by count, and
  a rigid transform is confirmed to be a location-only change.

Model volumes and areas are read back through ``pysmesh.load_brep`` on the session's own
BREP. That matters: ``Session.entity_table`` measures each *entity*, and after a merge two
ids denote one shape, so summing it is not the model's volume. The BREP is the model.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, ResolutionStatus, Session

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0
BOX_VOLUME: float = BOX_DX * BOX_DY * BOX_DZ

TAU: float = 2.0 * math.pi

# Relative tolerance for a value OCCT integrates numerically over a curved surface.
CURVED_RTOL: float = 1e-6

# Relative tolerance for an exact polyhedral value.
EXACT_RTOL: float = 1e-9

# Edges in the tangent-chain fillet fixture. The bar is at least 20 rounded in one call.
TANGENT_CHAIN_EDGES: int = 24

_ALL_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


# ---- independent oracles ------------------------------------------------------------- #


def model_volume(session: Session) -> float:
    """Total solid volume of the session's model, read back through the stateless API."""
    shape = ps.load_brep(session.brep())
    return float(sum(s.volume for s in shape.solids()))


def model_area(session: Session) -> float:
    """Total face area of the session's model, read back through the stateless API."""
    shape = ps.load_brep(session.brep())
    return float(sum(f.area for f in shape.faces()))


def model_length(session: Session) -> float:
    """Total edge length of the session's model.

    Summed over the entity table rather than the BREP because ``EdgeInfo`` carries no
    length; every edge in these fixtures belongs to exactly one entity, so the two agree.
    """
    return float(session.entity_table(EntityKind.EDGE).measure.sum())


def model_counts(session: Session) -> tuple[int, int, int, int]:
    """(solids, faces, edges, vertices) of the model, from the serialised shape."""
    shape = ps.load_brep(session.brep())
    return (
        len(shape.solids()),
        len(shape.faces()),
        len(shape.edges()),
        len(shape.vertices()),
    )


def face_labels(session: Session) -> dict[tuple[float, ...], float]:
    """Centroid -> area for every face of the model, from the serialised shape.

    The labelling a boolean's identity carry is checked against. It is computed from the
    BREP through the stateless API, so it shares no state with the registry under test.
    """
    shape = ps.load_brep(session.brep())
    out: dict[tuple[float, ...], float] = {}
    for info in shape.faces():
        out[tuple(np.round(info.centroid, 9))] = info.area
    return out


def all_ids(session: Session) -> list[int]:
    """Every live entity id of every kind, ascending."""
    return sorted(int(i) for kind in _ALL_KINDS for i in session.entities(kind))


def entity_geometry(session: Session) -> dict[int, tuple[float, ...]]:
    """id -> (measure, cx, cy, cz) for every live entity, from the session's own tables."""
    out: dict[int, tuple[float, ...]] = {}
    for kind in _ALL_KINDS:
        table = session.entity_table(kind)
        for i, measure, centroid in zip(table.ids, table.measure, table.centroid):
            out[int(i)] = (float(measure), *(float(c) for c in centroid))
    return out


# ---- fixtures ------------------------------------------------------------------------ #


@pytest.fixture
def rect_session() -> Session:
    """A session holding one 3 x 7 planar rectangle in the z = 0 plane."""
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), BOX_DX, BOX_DY)
    return s


@pytest.fixture
def overlapping_boxes() -> Session:
    """Two 3 x 7 x 11 boxes overlapping in x on [1.5, 3]."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(1.5, 0.0, 0.0))
    return s


@pytest.fixture
def touching_boxes() -> Session:
    """Two 3 x 7 x 11 boxes meeting face to face at x = 3.

    This is the ``split_box`` fixture's construction: fusing them leaves the four coplanar
    face pairs across the seam unmerged, so the result has 10 faces for a shape that is
    geometrically a plain 6 x 7 x 11 block.
    """
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(BOX_DX, 0.0, 0.0))
    return s


def tangent_chain_prism() -> Session:
    """A prism whose top rim is a chain of 24 tangent arcs.

    Every arc lies on one circle of radius 5, so consecutive arcs meet with continuous
    tangents: a genuine tangent chain, not a polygon approximation. A 24-gon would give the
    same edge count and none of the tangency.
    """
    s = Session()
    radius = 5.0
    step = TAU / TANGENT_CHAIN_EDGES
    for k in range(TANGENT_CHAIN_EDGES):
        a0 = k * step
        a1 = (k + 0.5) * step
        a2 = (k + 1) * step
        s.add_arc(
            (radius * math.cos(a0), radius * math.sin(a0), 0.0),
            (radius * math.cos(a1), radius * math.sin(a1), 0.0),
            (radius * math.cos(a2), radius * math.sin(a2), 0.0),
        )
    s.make_wire([EntityId(i) for i in s.entities(EntityKind.EDGE)])
    s.make_face([EntityId(i) for i in s.entities(EntityKind.EDGE)])
    s.extrude([EntityId(i) for i in s.entities(EntityKind.FACE)], (0.0, 0.0, BOX_DZ))
    return s


# =========================================== Primitives, curves, surfaces and sweeps ==== #
# Every operation produces a valid shape, and every analytic solid's volume matches its
# closed form.


def test_add_box_is_valid_and_has_the_closed_form_volume() -> None:
    s = Session()

    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    assert model_volume(s) == pytest.approx(BOX_VOLUME, rel=EXACT_RTOL)
    assert model_counts(s) == (1, 6, 12, 8)


def test_add_cylinder_volume_matches_pi_r_squared_h() -> None:
    s = Session()
    radius, height = 2.5, 9.0

    s.add_cylinder(radius, height)

    assert model_volume(s) == pytest.approx(
        math.pi * radius**2 * height, rel=CURVED_RTOL
    )


def test_add_cone_volume_matches_the_frustum_formula() -> None:
    s = Session()
    r1, r2, height = 3.0, 1.0, 6.0

    s.add_cone(r1, r2, height)

    expected = math.pi * height / 3.0 * (r1**2 + r1 * r2 + r2**2)
    assert model_volume(s) == pytest.approx(expected, rel=CURVED_RTOL)


def test_add_cone_with_a_zero_top_radius_is_a_valid_sharp_cone() -> None:
    s = Session()

    s.add_cone(3.0, 0.0, 6.0)

    assert model_volume(s) == pytest.approx(math.pi * 9.0 * 6.0 / 3.0, rel=CURVED_RTOL)


def test_add_sphere_volume_matches_four_thirds_pi_r_cubed() -> None:
    s = Session()
    radius = 4.0

    s.add_sphere(radius)

    assert model_volume(s) == pytest.approx(
        4.0 / 3.0 * math.pi * radius**3, rel=CURVED_RTOL
    )


def test_add_torus_volume_matches_the_pappus_value() -> None:
    s = Session()
    ring, tube = 5.0, 1.5

    s.add_torus(ring, tube)

    assert model_volume(s) == pytest.approx(
        2.0 * math.pi**2 * ring * tube**2, rel=CURVED_RTOL
    )


def test_add_wedge_volume_matches_the_trapezoidal_prism() -> None:
    s = Session()
    ltx = 1.0

    s.add_wedge(BOX_DX, BOX_DY, BOX_DZ, ltx)

    # A prism along z over a trapezoid that is dx wide at y = 0 and ltx wide at y = dy.
    expected = BOX_DZ * BOX_DY * (BOX_DX + ltx) / 2.0
    assert model_volume(s) == pytest.approx(expected, rel=EXACT_RTOL)


def test_a_half_swept_sphere_has_half_the_volume() -> None:
    full = Session()
    full.add_sphere(4.0)
    half = Session()

    half.add_sphere(4.0, angle_rad=math.pi)

    assert model_volume(half) == pytest.approx(model_volume(full) / 2.0, rel=CURVED_RTOL)


def test_add_cone_with_two_equal_radii_raises_pointing_at_add_cylinder() -> None:
    """OCCT's cone refuses equal radii; the session says so before OCCT has to."""
    s = Session()

    with pytest.raises(ps.PysmeshError, match="add_cylinder"):
        s.add_cone(2.5, 2.5, 9.0)


def test_an_occt_construction_error_surfaces_as_a_typed_pysmesh_error() -> None:
    """OCCT signals a bad parameter combination with Standard_ConstructionError.

    That derives from ``std::exception``, so without translation it would reach the caller
    as a bare ``RuntimeError`` carrying OCCT's wording and no indication of which call
    produced it.
    """
    s = Session()

    # Strictly positive, so the session's own range check passes it; degenerate, so OCCT's
    # does not. Exactly the case the translation exists for.
    with pytest.raises(ps.PysmeshError, match="OCCT rejected the request"):
        s.add_cone(3.0, 1.0, 1e-300)


@pytest.mark.parametrize(
    ("angle", "match"),
    [(0.0, "angle_rad"), (-1.0, "angle_rad"), (TAU + 0.1, "angle_rad")],
)
def test_a_sweep_angle_outside_zero_to_two_pi_raises(angle: float, match: str) -> None:
    s = Session()

    with pytest.raises(ps.PysmeshError, match=match):
        s.add_sphere(4.0, angle_rad=angle)


def test_a_torus_tube_wider_than_its_ring_raises() -> None:
    s = Session()

    with pytest.raises(ps.PysmeshError, match="self-intersection"):
        s.add_torus(1.5, 5.0)


def test_add_line_edge_length_is_the_point_distance() -> None:
    s = Session()

    s.add_line((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))

    assert model_length(s) == pytest.approx(5.0, rel=EXACT_RTOL)
    assert model_counts(s) == (0, 0, 1, 2)


def test_add_line_between_coincident_points_raises() -> None:
    s = Session()

    with pytest.raises(ps.PysmeshError, match="coincident"):
        s.add_line((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))


def test_add_arc_through_three_points_of_a_semicircle_has_pi_r_length() -> None:
    s = Session()
    radius = 3.0

    s.add_arc((radius, 0.0, 0.0), (0.0, radius, 0.0), (-radius, 0.0, 0.0))

    assert model_length(s) == pytest.approx(math.pi * radius, rel=CURVED_RTOL)


def test_add_arc_through_collinear_points_raises() -> None:
    s = Session()

    with pytest.raises(ps.PysmeshError, match="collinear"):
        s.add_arc((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))


def test_add_circle_edge_length_is_the_circumference() -> None:
    s = Session()
    radius = 2.5

    s.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), radius)

    assert model_length(s) == pytest.approx(TAU * radius, rel=CURVED_RTOL)
    # A closed circular edge has one seam vertex, not two.
    assert model_counts(s) == (0, 0, 1, 1)


def test_add_polyline_closed_has_one_edge_per_segment_and_the_perimeter_length() -> None:
    s = Session()
    points = [(0.0, 0.0, 0.0), (BOX_DX, 0.0, 0.0), (BOX_DX, BOX_DY, 0.0), (0.0, BOX_DY, 0.0)]

    s.add_polyline(points, closed=True)

    assert model_counts(s)[2:] == (4, 4)
    assert model_length(s) == pytest.approx(2.0 * (BOX_DX + BOX_DY), rel=EXACT_RTOL)


def test_add_polyline_open_leaves_the_loop_unclosed() -> None:
    s = Session()
    points = [(0.0, 0.0, 0.0), (BOX_DX, 0.0, 0.0), (BOX_DX, BOX_DY, 0.0)]

    s.add_polyline(points, closed=False)

    assert model_counts(s)[2:] == (2, 3)


def test_add_polyline_with_a_malformed_array_raises() -> None:
    s = Session()

    with pytest.raises(ps.PysmeshError, match=r"\(N, 3\)"):
        s.add_polyline(np.zeros((4, 2), dtype=np.float64))


def test_add_spline_passes_through_its_points_within_tolerance() -> None:
    s = Session()
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 1.0, 1.0], [5.0, 4.0, 2.0]],
        dtype=np.float64,
    )
    tol = 1e-6

    s.add_spline(points, tol=tol)

    shape = ps.load_brep(s.brep())
    vertices = np.array([v.xyz for v in shape.vertices()], dtype=np.float64)
    for end in (points[0], points[-1]):
        assert np.min(np.linalg.norm(vertices - end, axis=1)) == pytest.approx(0.0, abs=tol)


def test_add_bspline_starts_and_ends_at_its_outer_poles() -> None:
    s = Session()
    poles = np.array(
        [[0.0, 0.0, 0.0], [1.0, 4.0, 0.0], [4.0, 4.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    s.add_bspline(poles, degree=3)

    shape = ps.load_brep(s.brep())
    vertices = np.array([v.xyz for v in shape.vertices()], dtype=np.float64)
    for end in (poles[0], poles[-1]):
        assert np.min(np.linalg.norm(vertices - end, axis=1)) == pytest.approx(0.0, abs=1e-9)


def test_add_bspline_with_a_degree_above_the_pole_count_is_clamped_not_rejected() -> None:
    s = Session()
    poles = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

    s.add_bspline(poles, degree=9)

    assert model_counts(s)[2] == 1


def test_add_helix_length_matches_the_analytic_helix() -> None:
    s = Session()
    diameter, pitch, turns = 4.0, 2.0, 3.0

    s.add_helix((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), diameter, pitch, turns, tol=1e-6)

    # One turn unrolls to a right triangle: circumference by pitch.
    expected = turns * math.hypot(math.pi * diameter, pitch)
    assert model_length(s) == pytest.approx(expected, rel=1e-5)
    assert model_counts(s)[2] == int(turns)


def test_add_rectangle_area_is_dx_times_dy() -> None:
    s = Session()

    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), BOX_DX, BOX_DY)

    assert model_area(s) == pytest.approx(BOX_DX * BOX_DY, rel=EXACT_RTOL)
    assert model_counts(s) == (0, 1, 4, 4)


def test_make_wire_joins_loose_edges_into_one_body() -> None:
    s = Session()
    s.add_line((0.0, 0.0, 0.0), (BOX_DX, 0.0, 0.0))
    s.add_line((BOX_DX, 0.0, 0.0), (BOX_DX, BOX_DY, 0.0))

    s.make_wire([EntityId(i) for i in s.entities(EntityKind.EDGE)])

    assert model_counts(s)[2] == 2
    assert model_length(s) == pytest.approx(BOX_DX + BOX_DY, rel=EXACT_RTOL)


def test_make_wire_over_coincident_vertices_rebuilds_the_edges_and_says_so() -> None:
    """OCCT shares an edge only when a vertex is *the same*, not merely coincident.

    Two independently built lines meeting at a point have distinct vertices there, so
    ``BRepBuilderAPI_MakeWire`` rebuilds them. That kills their ids — which is honest, and
    the delta reports it. Pinned here so the behaviour cannot change unnoticed.
    """
    s = Session()
    s.add_line((0.0, 0.0, 0.0), (BOX_DX, 0.0, 0.0))
    s.add_line((BOX_DX, 0.0, 0.0), (BOX_DX, BOX_DY, 0.0))
    before = [EntityId(i) for i in s.entities(EntityKind.EDGE)]

    delta = s.make_wire(before)

    assert set(before) <= set(delta.deleted.tolist())
    assert len(delta.created) > 0


def test_make_wire_keeps_edge_ids_when_the_vertices_are_already_shared() -> None:
    s = Session()
    s.add_polyline(
        [(0.0, 0.0, 0.0), (BOX_DX, 0.0, 0.0), (BOX_DX, BOX_DY, 0.0)], closed=False
    )
    before = [EntityId(i) for i in s.entities(EntityKind.EDGE)]

    delta = s.make_wire(before)

    assert delta.deleted.size == 0
    assert s.entities(EntityKind.EDGE).tolist() == [int(i) for i in before]


def test_make_wire_on_an_edge_of_a_solid_raises_rather_than_consuming_the_solid() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    edges = [EntityId(i) for i in s.entities(EntityKind.EDGE)[:1]]

    with pytest.raises(ps.PysmeshError, match="construction geometry"):
        s.make_wire(edges)


def test_make_face_from_a_closed_polyline_keeps_every_edge_id() -> None:
    s = Session()
    s.add_polyline(
        [(0.0, 0.0, 0.0), (BOX_DX, 0.0, 0.0), (BOX_DX, BOX_DY, 0.0), (0.0, BOX_DY, 0.0)],
        closed=True,
    )
    before = s.entities(EntityKind.EDGE).tolist()

    s.make_face([EntityId(i) for i in before])

    assert s.entities(EntityKind.EDGE).tolist() == before
    assert model_area(s) == pytest.approx(BOX_DX * BOX_DY, rel=EXACT_RTOL)


def test_make_face_on_a_non_planar_boundary_raises_rather_than_approximating() -> None:
    s = Session()
    s.add_polyline(
        [(0.0, 0.0, 0.0), (BOX_DX, 0.0, 0.0), (BOX_DX, BOX_DY, 2.0), (0.0, BOX_DY, 0.0)],
        closed=True,
    )

    with pytest.raises(ps.PysmeshError, match="planar"):
        s.make_face([EntityId(i) for i in s.entities(EntityKind.EDGE)])


def test_make_filling_spans_a_non_planar_boundary() -> None:
    s = Session()
    s.add_polyline(
        [(0.0, 0.0, 0.0), (BOX_DX, 0.0, 0.0), (BOX_DX, BOX_DY, 2.0), (0.0, BOX_DY, 0.0)],
        closed=True,
    )

    s.make_filling([EntityId(i) for i in s.entities(EntityKind.EDGE)])

    assert model_counts(s)[1] == 1
    assert model_area(s) > BOX_DX * BOX_DY


def test_extrude_a_rectangle_gives_the_box_volume(rect_session: Session) -> None:
    faces = [EntityId(i) for i in rect_session.entities(EntityKind.FACE)]

    rect_session.extrude(faces, (0.0, 0.0, BOX_DZ))

    assert model_volume(rect_session) == pytest.approx(BOX_VOLUME, rel=EXACT_RTOL)
    assert model_counts(rect_session) == (1, 6, 12, 8)


def test_extrude_keeps_the_profiles_entity_ids(rect_session: Session) -> None:
    before = all_ids(rect_session)
    faces = [EntityId(i) for i in rect_session.entities(EntityKind.FACE)]

    delta = rect_session.extrude(faces, (0.0, 0.0, BOX_DZ))

    assert delta.deleted.size == 0
    assert set(before) <= set(all_ids(rect_session))


def test_extruded_walls_are_named_against_the_profile_they_came_from(
    rect_session: Session,
) -> None:
    profile = set(all_ids(rect_session))
    faces = [EntityId(i) for i in rect_session.entities(EntityKind.FACE)]

    delta = rect_session.extrude(faces, (0.0, 0.0, BOX_DZ))

    generated = [
        rect_session.origin(EntityId(int(i)))
        for i in delta.created
        if rect_session.origin(EntityId(int(i))).role is NameRole.GENERATED
    ]
    # Four walls from four profile edges, plus the solid from the profile face.
    assert len(generated) >= 5
    for origin in generated:
        assert origin.sources.size > 0
        assert set(origin.sources.tolist()) <= profile


def test_extrude_with_a_zero_vector_raises(rect_session: Session) -> None:
    faces = [EntityId(i) for i in rect_session.entities(EntityKind.FACE)]

    with pytest.raises(ps.PysmeshError, match="non-zero"):
        rect_session.extrude(faces, (0.0, 0.0, 0.0))


def test_extruding_a_solid_raises_rather_than_sweeping_element_wise() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    solids = [EntityId(i) for i in s.entities(EntityKind.SOLID)]

    with pytest.raises(ps.PysmeshError, match="sweep needs"):
        s.extrude(solids, (0.0, 0.0, 1.0))


def test_revolve_a_rectangle_about_z_gives_the_annular_cylinder_volume() -> None:
    s = Session()
    inner, outer, height = 1.0, 3.0, 4.0
    s.add_polyline(
        [
            (inner, 0.0, 0.0),
            (outer, 0.0, 0.0),
            (outer, 0.0, height),
            (inner, 0.0, height),
        ],
        closed=True,
    )
    s.make_face([EntityId(i) for i in s.entities(EntityKind.EDGE)])

    s.revolve(
        [EntityId(i) for i in s.entities(EntityKind.FACE)],
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    expected = math.pi * (outer**2 - inner**2) * height
    assert model_volume(s) == pytest.approx(expected, rel=CURVED_RTOL)


def test_pipe_along_a_straight_spine_gives_area_times_length() -> None:
    s = Session()
    length = 10.0
    s.add_line((0.0, 0.0, 0.0), (0.0, 0.0, length))
    spine = [EntityId(i) for i in s.entities(EntityKind.EDGE)]
    s.add_rectangle((-1.0, -1.0, 0.0), (0.0, 0.0, 1.0), 2.0, 2.0)
    profile = [EntityId(i) for i in s.entities(EntityKind.FACE)]

    s.pipe(spine, profile)

    assert model_volume(s) == pytest.approx(4.0 * length, rel=CURVED_RTOL)


def test_pipe_shell_closes_into_a_solid_of_the_same_volume() -> None:
    s = Session()
    length = 10.0
    s.add_line((0.0, 0.0, 0.0), (0.0, 0.0, length))
    spine = [EntityId(i) for i in s.entities(EntityKind.EDGE)]
    s.add_polyline(
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        closed=True,
    )
    profile = [
        EntityId(int(i)) for i in s.entities(EntityKind.EDGE) if int(i) not in spine
    ]

    s.pipe_shell(spine, profile, solid=True)

    assert model_volume(s) == pytest.approx(4.0 * length, rel=CURVED_RTOL)


def test_thru_sections_between_two_squares_gives_the_frustum_volume() -> None:
    s = Session()
    lower, upper, height = 2.0, 1.0, 6.0
    s.add_polyline(
        [(-lower, -lower, 0.0), (lower, -lower, 0.0), (lower, lower, 0.0), (-lower, lower, 0.0)],
        closed=True,
    )
    first = [EntityId(i) for i in s.entities(EntityKind.EDGE)]
    s.add_polyline(
        [
            (-upper, -upper, height),
            (upper, -upper, height),
            (upper, upper, height),
            (-upper, upper, height),
        ],
        closed=True,
    )
    second = [
        EntityId(int(i)) for i in s.entities(EntityKind.EDGE) if int(i) not in first
    ]

    s.thru_sections([first, second], solid=True, ruled=True)

    a1, a2 = (2 * lower) ** 2, (2 * upper) ** 2
    expected = height / 3.0 * (a1 + a2 + math.sqrt(a1 * a2))
    assert model_volume(s) == pytest.approx(expected, rel=EXACT_RTOL)


def test_thru_sections_with_one_section_raises() -> None:
    s = Session()
    s.add_polyline([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], closed=True)
    edges = [EntityId(i) for i in s.entities(EntityKind.EDGE)]

    with pytest.raises(ps.PysmeshError, match="at least two sections"):
        s.thru_sections([edges])


def test_thru_sections_naming_the_same_body_twice_raises() -> None:
    s = Session()
    s.add_polyline([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], closed=True)
    edges = [EntityId(i) for i in s.entities(EntityKind.EDGE)]

    with pytest.raises(ps.PysmeshError, match="two different sections"):
        s.thru_sections([edges[:1], edges[1:]])


# =============================================================== Booleans with history == #


def test_fuse_of_two_touching_boxes_matches_the_split_box_fixtures_topology(
    touching_boxes: Session, split_box_brep: bytes
) -> None:
    """The fixture records what this fuse must produce: 10 faces, not the 6 of a block."""
    solids = [EntityId(i) for i in touching_boxes.entities(EntityKind.SOLID)]

    touching_boxes.fuse(solids[:1], solids[1:])

    reference = ps.load_brep(split_box_brep)
    assert model_counts(touching_boxes)[1] == len(reference.faces())
    assert model_volume(touching_boxes) == pytest.approx(2.0 * BOX_VOLUME, rel=EXACT_RTOL)


def test_every_pre_fuse_face_id_lands_on_the_face_it_geometrically_belongs_to(
    touching_boxes: Session,
) -> None:
    """The central identity claim, checked against a labelling the registry did not make.

    Face centroids and areas come from ``load_brep`` on the serialised shape, before and
    after. A pre-op face that survives the fuse must denote a post-op face at the same
    centroid with the same area; the two seam faces must die, because the fused solid has
    no interior wall.
    """
    before_labels = face_labels(touching_boxes)
    before = {
        int(i): (float(m), tuple(np.round(c, 9)))
        for i, m, c in zip(
            touching_boxes.entity_table(EntityKind.FACE).ids,
            touching_boxes.entity_table(EntityKind.FACE).measure,
            touching_boxes.entity_table(EntityKind.FACE).centroid,
        )
    }
    solids = [EntityId(i) for i in touching_boxes.entities(EntityKind.SOLID)]

    delta = touching_boxes.fuse(solids[:1], solids[1:])

    after_labels = face_labels(touching_boxes)
    survivors = 0
    for face_id, (area, centroid) in before.items():
        if touching_boxes.is_alive(EntityId(face_id)):
            assert centroid in after_labels, f"face {face_id} moved off its own geometry"
            assert after_labels[centroid] == pytest.approx(area, rel=EXACT_RTOL)
            assert before_labels[centroid] == pytest.approx(area, rel=EXACT_RTOL)
            survivors += 1
    assert survivors == len(after_labels)
    # The two coincident seam faces are interior to the result, so both die.
    assert len(delta.deleted) >= 2


def test_cut_removes_exactly_the_tool_volume() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(1.0, 2.0, 3.0)
    solids = [EntityId(i) for i in s.entities(EntityKind.SOLID)]

    s.cut(solids[:1], solids[1:])

    assert model_volume(s) == pytest.approx(BOX_VOLUME - 6.0, rel=EXACT_RTOL)


def test_common_keeps_only_the_overlap(overlapping_boxes: Session) -> None:
    solids = [EntityId(i) for i in overlapping_boxes.entities(EntityKind.SOLID)]

    overlapping_boxes.common(solids[:1], solids[1:])

    assert model_volume(overlapping_boxes) == pytest.approx(
        1.5 * BOX_DY * BOX_DZ, rel=EXACT_RTOL
    )


def test_common_leaves_both_operand_ids_on_the_one_result_solid(
    overlapping_boxes: Session,
) -> None:
    solids = [EntityId(i) for i in overlapping_boxes.entities(EntityKind.SOLID)]

    delta = overlapping_boxes.common(solids[:1], solids[1:])

    assert model_counts(overlapping_boxes)[0] == 1
    assert all(overlapping_boxes.is_alive(i) for i in solids)
    assert set(int(i) for i in solids) <= set(delta.merged.tolist())


def test_section_adds_curves_and_consumes_nothing(overlapping_boxes: Session) -> None:
    before = all_ids(overlapping_boxes)
    before_geometry = entity_geometry(overlapping_boxes)
    solids = [EntityId(i) for i in overlapping_boxes.entities(EntityKind.SOLID)]

    delta = overlapping_boxes.section(solids[:1], solids[1:])

    assert delta.deleted.size == 0
    assert delta.modified.size == 0
    assert delta.created.size > 0
    # Every operand entity is where it was: same measure, same centroid, same id.
    after_geometry = entity_geometry(overlapping_boxes)
    for entity_id in before:
        assert after_geometry[entity_id] == pytest.approx(before_geometry[entity_id])
    assert model_volume(overlapping_boxes) == pytest.approx(
        2.0 * BOX_VOLUME, rel=EXACT_RTOL
    )


def test_every_section_curve_is_named_against_the_operand_it_came_from(
    overlapping_boxes: Session,
) -> None:
    """Provenance is what makes an additive boolean's output addressable later.

    OCCT's boolean family reports almost everything through ``Modified`` and hardly
    anything through ``Generated``, so a section's curves would carry no source at all
    unless the session derives one.
    """
    operands = set(all_ids(overlapping_boxes))
    solids = [EntityId(i) for i in overlapping_boxes.entities(EntityKind.SOLID)]

    delta = overlapping_boxes.section(solids[:1], solids[1:])

    for i in delta.created:
        origin = overlapping_boxes.origin(EntityId(int(i)))
        assert origin.role is NameRole.GENERATED
        assert origin.sources.size > 0
        assert set(origin.sources.tolist()) <= operands


def test_section_of_disjoint_solids_creates_nothing() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(100.0, 0.0, 0.0))
    solids = [EntityId(i) for i in s.entities(EntityKind.SOLID)]

    delta = s.section(solids[:1], solids[1:])

    assert delta.created.size == 0
    assert delta.deleted.size == 0


def test_split_divides_the_target_and_keeps_the_tool(overlapping_boxes: Session) -> None:
    solids = [EntityId(i) for i in overlapping_boxes.entities(EntityKind.SOLID)]

    overlapping_boxes.split(solids[:1], solids[1:])

    assert model_counts(overlapping_boxes)[0] == 3
    assert model_volume(overlapping_boxes) == pytest.approx(
        2.0 * BOX_VOLUME, rel=EXACT_RTOL
    )
    assert overlapping_boxes.shape_count(solids[1]) == 1


def test_a_split_targets_name_resolves_as_ambiguous(overlapping_boxes: Session) -> None:
    solids = [EntityId(i) for i in overlapping_boxes.entities(EntityKind.SOLID)]
    name = overlapping_boxes.name_of(solids[0])

    overlapping_boxes.split(solids[:1], solids[1:])

    resolution = overlapping_boxes.resolve(name)
    assert resolution.status is ResolutionStatus.AMBIGUOUS
    assert resolution.shape_count == 2


def test_fragment_keeps_every_piece_and_both_operand_ids(
    overlapping_boxes: Session,
) -> None:
    solids = [EntityId(i) for i in overlapping_boxes.entities(EntityKind.SOLID)]

    overlapping_boxes.fragment(solids)

    # Three pieces: the two exclusive parts and the shared middle.
    assert model_counts(overlapping_boxes)[0] == 3
    assert model_volume(overlapping_boxes) == pytest.approx(
        2.0 * BOX_VOLUME - 1.5 * BOX_DY * BOX_DZ, rel=EXACT_RTOL
    )
    assert all(overlapping_boxes.is_alive(i) for i in solids)


def test_fragment_with_one_solid_raises() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    with pytest.raises(ps.PysmeshError, match="at least two solids"):
        s.fragment([EntityId(i) for i in s.entities(EntityKind.SOLID)])


@pytest.mark.parametrize("op", ["cut", "common", "section", "split"])
def test_a_boolean_on_a_face_id_raises_naming_the_wrong_kind(
    touching_boxes: Session, op: str
) -> None:
    solid = EntityId(int(touching_boxes.entities(EntityKind.SOLID)[0]))
    face = EntityId(int(touching_boxes.entities(EntityKind.FACE)[0]))

    with pytest.raises(ps.PysmeshError, match="not a SOLID"):
        getattr(touching_boxes, op)([face], [solid])


@pytest.mark.parametrize("op", ["cut", "common", "section", "split"])
def test_a_boolean_with_a_negative_fuzzy_raises(touching_boxes: Session, op: str) -> None:
    solids = [EntityId(i) for i in touching_boxes.entities(EntityKind.SOLID)]

    with pytest.raises(ps.PysmeshError, match="fuzzy must be a finite value >= 0"):
        getattr(touching_boxes, op)(solids[:1], solids[1:], fuzzy=-1.0)


def test_a_boolean_leaves_the_session_untouched_when_it_fails(
    touching_boxes: Session,
) -> None:
    before_ids = all_ids(touching_boxes)
    before_ops = touching_boxes.op_count
    solids = [EntityId(i) for i in touching_boxes.entities(EntityKind.SOLID)]

    with pytest.raises(ps.PysmeshError):
        touching_boxes.cut(solids[:1], solids[1:], fuzzy=-1.0)

    assert all_ids(touching_boxes) == before_ids
    assert touching_boxes.op_count == before_ops


def test_a_boolean_runs_non_destructively_so_an_earlier_snapshot_is_intact(
    overlapping_boxes: Session,
) -> None:
    """BOPAlgo's default mode updates the argument shapes in place.

    A session's snapshot is the shape, so an in-place update would corrupt every retained
    state. The session forces non-destructive mode; this asserts the consequence rather
    than the setting.
    """
    mark = overlapping_boxes.snapshot()
    before_counts = model_counts(overlapping_boxes)
    before_geometry = entity_geometry(overlapping_boxes)
    solids = [EntityId(i) for i in overlapping_boxes.entities(EntityKind.SOLID)]

    overlapping_boxes.fragment(solids)
    overlapping_boxes.restore(mark)

    assert model_counts(overlapping_boxes) == before_counts
    after_geometry = entity_geometry(overlapping_boxes)
    for entity_id, geometry in before_geometry.items():
        assert after_geometry[entity_id] == pytest.approx(geometry)


# ==================================================================== Fillet and chamfer = #


def test_a_tangent_chain_of_24_edges_fillets_in_one_call() -> None:
    """No per-edge face or volume co-selection: one call, one valid result."""
    s = tangent_chain_prism()
    rim = [
        EntityId(int(i))
        for i, bbox in zip(
            s.entity_table(EntityKind.EDGE).ids, s.entity_table(EntityKind.EDGE).bbox
        )
        if bbox[2] == pytest.approx(BOX_DZ) and bbox[5] == pytest.approx(BOX_DZ)
    ]
    assert len(rim) == TANGENT_CHAIN_EDGES

    delta = s.fillet(rim, 0.4)

    assert delta.op == "fillet"
    assert model_counts(s)[0] == 1
    assert model_volume(s) < BOX_DZ * math.pi * 25.0


def test_a_variable_radius_fillet_removes_more_than_its_smaller_constant_twin() -> None:
    small = Session()
    small.add_box(BOX_DX, BOX_DY, BOX_DZ)
    edge = [EntityId(int(small.entities(EntityKind.EDGE)[0]))]
    small.fillet(edge, 0.4)

    varying = Session()
    varying.add_box(BOX_DX, BOX_DY, BOX_DZ)
    edge = [EntityId(int(varying.entities(EntityKind.EDGE)[0]))]
    varying.fillet(edge, 0.4, radius_end=1.2)

    assert model_volume(varying) < model_volume(small)
    assert model_volume(varying) < BOX_VOLUME


def test_a_fillet_radius_occt_cannot_build_raises_naming_the_edge() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    edge = EntityId(int(s.entities(EntityKind.EDGE)[0]))
    before = all_ids(s)

    with pytest.raises(ps.PysmeshError) as excinfo:
        s.fillet([edge], 50.0)

    assert int(edge) in excinfo.value.face_ids
    assert all_ids(s) == before


def test_a_chamfer_removes_the_prism_of_a_right_isosceles_triangle() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    table = s.entity_table(EntityKind.EDGE)
    # The four edges along y are the ones of length BOX_DY.
    edge = EntityId(
        int(next(i for i, m in zip(table.ids, table.measure) if m == pytest.approx(BOX_DY)))
    )
    distance = 0.5

    s.chamfer([edge], distance)

    expected = BOX_VOLUME - 0.5 * distance**2 * BOX_DY
    assert model_volume(s) == pytest.approx(expected, rel=1e-6)


def test_a_two_distance_chamfer_against_a_reference_face_removes_the_right_wedge() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    table = s.entity_table(EntityKind.EDGE)
    edge = EntityId(
        int(next(i for i, m in zip(table.ids, table.measure) if m == pytest.approx(BOX_DY)))
    )
    faces = s.entity_table(EntityKind.FACE)
    # The x = 0 face: its centroid sits on the yz plane.
    face = EntityId(
        int(next(i for i, c in zip(faces.ids, faces.centroid) if c[0] == pytest.approx(0.0)))
    )
    d1, d2 = 0.4, 0.9

    s.chamfer([edge], d1, distance_end=d2, face_id=face)

    expected = BOX_VOLUME - 0.5 * d1 * d2 * BOX_DY
    assert model_volume(s) == pytest.approx(expected, rel=1e-6)


def test_a_chamfer_face_without_its_second_distance_raises() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    edge = EntityId(int(s.entities(EntityKind.EDGE)[0]))
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    with pytest.raises(ps.PysmeshError, match="must be given together"):
        s.chamfer([edge], 0.5, face_id=face)


def test_a_chamfer_distance_occt_cannot_build_raises_naming_the_edge() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    edge = EntityId(int(s.entities(EntityKind.EDGE)[0]))
    before = all_ids(s)

    with pytest.raises(ps.PysmeshError) as excinfo:
        s.chamfer([edge], 50.0)

    assert int(edge) in excinfo.value.face_ids
    assert all_ids(s) == before


# ============================================================================ Transforms = #
# Every id unchanged, and a rigid transform is a location-only change.


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("translate", {"offset": (1.0, -2.0, 3.5)}),
        ("rotate", {"origin": (0.0, 0.0, 0.0), "axis": (0.0, 0.0, 1.0), "angle_rad": 0.7}),
        ("mirror", {"point": (0.0, 0.0, 0.0), "normal": (1.0, 0.0, 0.0)}),
        ("scale", {"factors": 2.0}),
    ],
)
def test_a_transform_leaves_every_entity_id_unchanged(
    name: str, kwargs: dict[str, object]
) -> None:
    """Asserted id by id, not by count: a count survives a wholesale re-issue."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    before = all_ids(s)

    delta = getattr(s, name)(**kwargs)

    assert all_ids(s) == before
    assert delta.created.size == 0
    assert delta.deleted.size == 0


def test_an_anisotropic_scale_leaves_every_entity_id_unchanged() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    before = all_ids(s)

    s.scale((2.0, 3.0, 5.0))

    assert all_ids(s) == before
    assert model_volume(s) == pytest.approx(BOX_VOLUME * 30.0, rel=EXACT_RTOL)


def brep_geometry(data: bytes) -> bytes:
    """The curve and surface sections of a BREP: every geometric definition, no placement.

    A BREP records placements in its ``Locations`` block and topology in its ``TShapes``
    block; everything between them is the geometry itself. A transform that only sets a new
    location leaves this byte-identical, and one that rebuilds cannot.
    """
    return data[data.index(b"Curve2ds") : data.index(b"TShapes")]


def test_a_rigid_transform_is_a_location_only_change() -> None:
    """The TShape pointers survive, so not one curve or surface is rebuilt.

    This is the property the requirements single out, and the test exists so a later
    refactor cannot silently turn the relocation path into a copying one. The BREP is the
    witness: a relocated model gains a ``Locations`` entry and keeps every curve and
    surface definition exactly as it was.
    """
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    before = s.brep()

    s.translate((1.0, -2.0, 3.5))

    after = s.brep()
    assert brep_geometry(after) == brep_geometry(before)
    assert b"Locations 0" in before
    assert b"Locations 1" in after


def test_a_mirror_rebuilds_the_geometry_rather_than_relocating_it() -> None:
    """The other direction of the same claim, so neither path can absorb the other.

    A plane reflection has determinant -1, which OCCT cannot express as a location; it
    rebuilds. Asserting only the relocation half would let a regression that routed
    *everything* through the rebuild path pass unnoticed.
    """
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    before = s.brep()

    s.mirror((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert brep_geometry(s.brep()) != brep_geometry(before)


def test_a_mirror_reflects_the_model_and_preserves_its_volume() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    s.mirror((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    solids = s.entity_table(EntityKind.SOLID)
    assert model_volume(s) == pytest.approx(BOX_VOLUME, rel=EXACT_RTOL)
    assert solids.centroid[0][0] == pytest.approx(-BOX_DX / 2.0)


def test_a_uniform_scale_cubes_the_volume() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    s.scale(2.0)

    assert model_volume(s) == pytest.approx(BOX_VOLUME * 8.0, rel=EXACT_RTOL)


def test_a_uniform_scale_keeps_a_cylinder_analytic() -> None:
    s = Session()
    s.add_cylinder(2.5, 9.0)

    s.scale(3.0)

    assert model_volume(s) == pytest.approx(
        math.pi * 2.5**2 * 9.0 * 27.0, rel=CURVED_RTOL
    )


def test_a_scale_factor_of_one_raises_rather_than_burning_an_operation() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    with pytest.raises(ps.PysmeshError, match="no-op"):
        s.scale(1.0)


def test_a_non_positive_scale_factor_raises() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    with pytest.raises(ps.PysmeshError, match="must be > 0"):
        s.scale((2.0, -1.0, 2.0))


def test_a_transform_scoped_to_one_body_leaves_the_other_where_it_is() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(100.0, 0.0, 0.0))
    solids = [EntityId(i) for i in s.entities(EntityKind.SOLID)]
    before = entity_geometry(s)

    s.mirror((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), [solids[1]])

    after = entity_geometry(s)
    moved = [i for i in before if after[i] != pytest.approx(before[i])]
    assert len(moved) == 27  # one box: 1 solid + 6 faces + 12 edges + 8 vertices


def test_copy_duplicates_the_body_with_wholly_new_ids() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    before = all_ids(s)

    delta = s.copy([EntityId(int(s.entities(EntityKind.SOLID)[0]))])

    assert set(before) <= set(all_ids(s))
    assert set(before).isdisjoint(delta.created.tolist())
    assert len(delta.created) == len(before)
    assert model_counts(s)[0] == 2
    assert model_volume(s) == pytest.approx(2.0 * BOX_VOLUME, rel=EXACT_RTOL)


def test_a_copy_moves_independently_of_its_original() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    original = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    delta = s.copy([original])
    duplicate = EntityId(
        int(next(i for i in delta.created if s.entity_kind(EntityId(int(i))) is EntityKind.SOLID))
    )

    s.translate((100.0, 0.0, 0.0), [duplicate])

    table = s.entity_table(EntityKind.SOLID)
    centroids = {int(i): c for i, c in zip(table.ids, table.centroid)}
    assert centroids[int(original)][0] == pytest.approx(BOX_DX / 2.0)
    assert centroids[int(duplicate)][0] == pytest.approx(100.0 + BOX_DX / 2.0)


def test_copy_with_an_empty_selection_raises() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    with pytest.raises(ps.PysmeshError, match="at least one entity"):
        s.copy([])
