# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""Gates for the session's geometric query surface.

Two claims are under test:

* **Every query answers**, against the session's own entity ids, and its answer matches a
  closed-form value or an independent read of the same model through the stateless API.
* **Curvature is measured, not sampled once.** The headline item, because the query this
  replaces takes a single sample at each face's parametric centre and is therefore exact only
  for a face of constant curvature. It is validated against the analytic curvature of a
  sphere, a cylinder and a torus at three sampling densities, and against a **cone**, where
  the centre sample is provably wrong: on a 4-to-1 taper it reads 0.358 against a true peak
  of 0.894.

The cone is the load-bearing fixture. Its sampled peak has a closed form at *every* density —
``cos(alpha) / (r1 + (r2 - r1)(n - 0.5)/n)`` — so the gate asserts the exact value the grid
must produce rather than a tolerance band, and the single-sample case falls out of it as
``n = 1``.

Two OCCT behaviours are pinned here because a plausible implementation gets both wrong: the
principal curvatures are *signed and ordered by value*, so a cylinder's larger one is 0 and a
curvature map keyed on it reports every cylinder as flat; and a reversed face's surface
normal points into the body.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, Session

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0
BOX_VOLUME: float = BOX_DX * BOX_DY * BOX_DZ
BOX_CENTRE: tuple[float, float, float] = (BOX_DX / 2.0, BOX_DY / 2.0, BOX_DZ / 2.0)
BOX_EDGE_LENGTH: float = 4.0 * (BOX_DX + BOX_DY + BOX_DZ)

TAU: float = 2.0 * math.pi

EXACT_RTOL: float = 1e-9
CURVED_RTOL: float = 1e-6

# Sampling densities the curvature gate is asserted at.
SAMPLE_DENSITIES: tuple[int, ...] = (1, 4, 16)

# The cone the centre sample misreads.
CONE_R1: float = 4.0
CONE_R2: float = 1.0
CONE_H: float = 6.0
CONE_HALF_ANGLE: float = math.atan((CONE_R1 - CONE_R2) / CONE_H)

HOLE_RADIUS: float = 0.5

# Analytic surfaces the parameter query is asserted against.
SPHERE_R: float = 4.0
TORUS_MAJOR: float = 5.0
TORUS_MINOR: float = 1.5

# Free-form geometry: no analytic parameters exist for any of it.
FREEFORM_RIM: np.ndarray = np.array(
    [[0.0, 0.0, 0.0], [4.0, 0.0, 1.0], [4.0, 5.0, 0.0], [0.0, 5.0, 2.0]]
)
EXTRUDE_PROFILE: np.ndarray = np.array(
    [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 1.0, 0.0], [5.0, 3.0, 0.0]]
)
EXTRUDE_VECTOR: tuple[float, float, float] = (0.0, 0.0, 4.0)
REVOLVE_PROFILE: np.ndarray = np.array(
    [[2.0, 0.0, 0.0], [2.5, 0.0, 1.0], [2.2, 0.0, 2.0], [3.0, 0.0, 3.0]]
)


# ---- oracles -------------------------------------------------------------------------- #


def cone_peak_curvature(samples: int) -> float:
    """Peak |kappa| an n x n cell-centred grid must find on the truncated cone.

    A cone's principal curvatures are 0 along the ruling and ``cos(alpha) / r`` around it,
    where ``r`` is the distance from the axis. The grid's extreme sample sits at the cell
    centre nearest the small end, so its radius — and therefore the answer — is exact.
    """
    radius = CONE_R1 + (CONE_R2 - CONE_R1) * (samples - 0.5) / samples
    return math.cos(CONE_HALF_ANGLE) / radius


def ids_of(session: Session, kind: EntityKind) -> list[EntityId]:
    """Live entity ids of one kind, as the typed id."""
    return [EntityId(int(i)) for i in session.entities(kind)]


def face_of_type(session: Session, name: str) -> EntityId:
    """The single face of the model whose surface is of the named type."""
    table = session.entity_types(EntityKind.FACE)
    matches = [EntityId(int(i)) for i, t in zip(table.ids, table.types) if t == name]
    assert len(matches) == 1, f"expected exactly one {name} face, got {len(matches)}"
    return matches[0]


def face_centre_uv(session: Session, face: EntityId) -> np.ndarray:
    """The (1, 2) parametric centre of one face."""
    umin, umax, vmin, vmax = session.face_parameter_bounds([face])[0]
    return np.array([[0.5 * (umin + umax), 0.5 * (vmin + vmax)]])


# ---- fixtures ------------------------------------------------------------------------- #


@pytest.fixture
def box() -> Session:
    """A session holding one 3 x 7 x 11 box at the origin."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    return s


@pytest.fixture
def cone() -> Session:
    """A truncated cone tapering from radius 4 to radius 1 over a height of 6."""
    s = Session()
    s.add_cone(CONE_R1, CONE_R2, CONE_H)
    return s


@pytest.fixture
def bored_box() -> Session:
    """A 3 x 7 x 11 box with a small through hole on its axis, along z."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_cylinder(HOLE_RADIUS, BOX_DZ + 2.0, origin=(*BOX_CENTRE[:2], -1.0))
    solid, tool = ids_of(s, EntityKind.SOLID)
    s.cut([solid], [tool])
    return s


# ================================================================== Types and bounds ==


def test_entity_types_names_a_boxs_planes_and_lines(box: Session) -> None:
    faces = box.entity_types(EntityKind.FACE)
    edges = box.entity_types(EntityKind.EDGE)

    assert faces.kind is EntityKind.FACE
    assert len(faces.types) == len(faces.ids) == 6
    assert set(faces.types) == {"Plane"}
    assert set(edges.types) == {"Line"}


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        ("cylinder", "Cylinder"),
        ("cone", "Cone"),
        ("sphere", "Sphere"),
        ("torus", "Torus"),
    ],
)
def test_entity_types_recognises_each_analytic_surface(build: str, expected: str) -> None:
    s = Session()
    if build == "cylinder":
        s.add_cylinder(2.5, 9.0)
    elif build == "cone":
        s.add_cone(CONE_R1, CONE_R2, CONE_H)
    elif build == "sphere":
        s.add_sphere(4.0)
    else:
        s.add_torus(5.0, 1.5)

    assert expected in s.entity_types(EntityKind.FACE).types


def test_entity_types_recognises_a_circular_edge() -> None:
    s = Session()
    s.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 5.0)

    assert s.entity_types(EntityKind.EDGE).types == ("Circle",)


def test_bounding_boxes_cover_the_model(box: Session) -> None:
    table = box.bounding_boxes(EntityKind.SOLID)

    assert table.bbox.shape == (1, 6)
    lo, hi = table.bbox[0, :3], table.bbox[0, 3:]
    assert np.all(lo <= [0.0, 0.0, 0.0])
    assert np.all(hi >= [BOX_DX, BOX_DY, BOX_DZ])


def test_bounding_boxes_agree_with_the_full_entity_table(box: Session) -> None:
    """The cheap query and the expensive one must not disagree about the same geometry."""
    cheap = box.bounding_boxes(EntityKind.FACE)
    full = box.entity_table(EntityKind.FACE)

    assert np.array_equal(cheap.ids, full.ids)
    assert np.allclose(cheap.bbox, full.bbox, rtol=0.0, atol=0.0)


# ============================================================== Mass properties ==


def test_mass_properties_of_a_solid_match_the_closed_form(box: Session) -> None:
    solid = ids_of(box, EntityKind.SOLID)[0]

    table = box.mass_properties([solid])

    assert table.measure[0] == pytest.approx(BOX_VOLUME, rel=EXACT_RTOL)
    assert table.centroid[0] == pytest.approx(BOX_CENTRE, rel=EXACT_RTOL)


def test_mass_properties_measure_each_kind_by_its_own_dimension(box: Session) -> None:
    faces = box.mass_properties(ids_of(box, EntityKind.FACE))
    edges = box.mass_properties(ids_of(box, EntityKind.EDGE))
    vertices = box.mass_properties(ids_of(box, EntityKind.VERTEX))

    surface = 2.0 * (BOX_DX * BOX_DY + BOX_DY * BOX_DZ + BOX_DX * BOX_DZ)
    assert faces.measure.sum() == pytest.approx(surface, rel=EXACT_RTOL)
    assert edges.measure.sum() == pytest.approx(BOX_EDGE_LENGTH, rel=EXACT_RTOL)
    assert np.all(vertices.measure == 0.0)


def test_summing_edge_measures_does_not_double_count_shared_edges(box: Session) -> None:
    """OCCT's linear properties of a *solid* visit every edge once per owning face.

    Taken that way a box's total edge length comes out at 168 instead of 84 — a silent factor
    of two. The query measures each entity by its own kind, so the sum is the real total.
    """
    total = box.mass_properties(ids_of(box, EntityKind.EDGE)).measure.sum()

    assert total == pytest.approx(BOX_EDGE_LENGTH, rel=EXACT_RTOL)
    assert total != pytest.approx(2.0 * BOX_EDGE_LENGTH, rel=1e-3)


def test_mass_properties_rejects_a_dead_id(box: Session) -> None:
    solid = ids_of(box, EntityKind.SOLID)[0]
    box.remove([solid])

    with pytest.raises(ps.PysmeshError, match="dead"):
        box.mass_properties([solid])


# ========================================================== Parameter bounds ==


def test_face_parameter_bounds_span_the_faces_extent(box: Session) -> None:
    bounds = box.face_parameter_bounds(ids_of(box, EntityKind.FACE))

    assert bounds.shape == (6, 4)
    spans = sorted(
        tuple(sorted((round(b[1] - b[0], 9), round(b[3] - b[2], 9)))) for b in bounds
    )
    assert spans == sorted(
        [
            (BOX_DX, BOX_DY),
            (BOX_DX, BOX_DY),
            (BOX_DX, BOX_DZ),
            (BOX_DX, BOX_DZ),
            (BOX_DY, BOX_DZ),
            (BOX_DY, BOX_DZ),
        ]
    )


def test_edge_parameter_bounds_span_each_edges_length(box: Session) -> None:
    edges = ids_of(box, EntityKind.EDGE)

    bounds = box.edge_parameter_bounds(edges)
    lengths = box.mass_properties(edges).measure

    assert bounds.shape == (12, 2)
    assert np.allclose(bounds[:, 1] - bounds[:, 0], lengths, rtol=EXACT_RTOL)


def test_face_parameter_bounds_rejects_an_edge_id(box: Session) -> None:
    edge = ids_of(box, EntityKind.EDGE)[0]

    with pytest.raises(ps.PysmeshError, match="not a FACE"):
        box.face_parameter_bounds([edge])


# =============================================================================== Adjacency ==


def test_adjacency_gives_a_boxs_face_to_edge_incidences(box: Session) -> None:
    pairs = box.adjacency(EntityKind.FACE, EntityKind.EDGE)

    assert pairs.ids.size == 24  # 6 faces x 4 edges
    assert len(set(pairs.related.tolist())) == 12


def test_adjacency_gives_a_boxs_edge_to_face_incidences(box: Session) -> None:
    pairs = box.adjacency(EntityKind.EDGE, EntityKind.FACE)

    assert pairs.ids.size == 24  # 12 edges x 2 faces
    assert len(set(pairs.ids.tolist())) == 12


def test_the_two_adjacency_directions_are_the_same_relation(box: Session) -> None:
    """Boundary and ancestors are one relation read from either end, so they must transpose."""
    down = box.adjacency(EntityKind.FACE, EntityKind.EDGE)
    up = box.adjacency(EntityKind.EDGE, EntityKind.FACE)

    assert set(zip(down.ids.tolist(), down.related.tolist())) == set(
        zip(up.related.tolist(), up.ids.tolist())
    )


def test_adjacency_relates_faces_to_the_solid_that_owns_them(box: Session) -> None:
    solid = ids_of(box, EntityKind.SOLID)[0]

    pairs = box.adjacency(EntityKind.FACE, EntityKind.SOLID)

    assert pairs.ids.size == 6
    assert set(pairs.related.tolist()) == {int(solid)}


def test_adjacency_rejects_a_kind_related_to_itself(box: Session) -> None:
    with pytest.raises(ps.PysmeshError, match="must differ"):
        box.adjacency(EntityKind.FACE, EntityKind.FACE)


# ================================================================ Positions and normals ==


def test_surface_at_returns_points_on_the_face(box: Session) -> None:
    face = ids_of(box, EntityKind.FACE)[0]
    uv = face_centre_uv(box, face)

    sample = box.surface_at(face, uv)

    centroid = box.mass_properties([face]).centroid[0]
    assert sample.points.shape == (1, 3)
    assert sample.points[0] == pytest.approx(centroid, abs=1e-9)
    assert bool(sample.defined[0])


def test_every_box_face_normal_points_out_of_the_body(box: Session) -> None:
    """A reversed face's *surface* normal points inward, so this is a real trap.

    Checked on all six faces at once: the dot product of the normal with the vector from the
    body centre to the face must be positive for every one of them.
    """
    centre = np.asarray(BOX_CENTRE)

    for face in ids_of(box, EntityKind.FACE):
        sample = box.surface_at(face, face_centre_uv(box, face))

        outward = float(np.dot(sample.normals[0], sample.points[0] - centre))
        assert outward > 0.0, f"face {face} normal points inward"
        assert np.linalg.norm(sample.normals[0]) == pytest.approx(1.0, rel=EXACT_RTOL)


def test_surface_at_rejects_parameters_that_are_not_pairs(box: Session) -> None:
    face = ids_of(box, EntityKind.FACE)[0]

    with pytest.raises(ps.PysmeshError, match=r"\(N, 2\)"):
        box.surface_at(face, np.zeros((3, 3)))


# ================================================================== Curvature ==


@pytest.mark.parametrize("samples", SAMPLE_DENSITIES)
def test_curvature_of_a_sphere_is_one_over_its_radius(samples: int) -> None:
    radius = 4.0
    s = Session()
    s.add_sphere(radius)
    face = face_of_type(s, "Sphere")

    table = s.curvature([face], samples=samples)

    assert table.k_max[0] == pytest.approx(1.0 / radius, rel=CURVED_RTOL)
    assert table.samples_used[0] > 0


@pytest.mark.parametrize("samples", SAMPLE_DENSITIES)
def test_curvature_of_a_cylinder_is_one_over_its_radius(samples: int) -> None:
    """The signed-curvature trap, stated as a test.

    A cylinder's principal curvatures are ``(0, -1/R)`` and OCCT orders them by value, so the
    *larger* one is 0. An implementation that reads it reports every cylinder as flat, which
    is silent and catastrophic for a curvature-driven sizing field.
    """
    radius = 2.5
    s = Session()
    s.add_cylinder(radius, 9.0)
    face = face_of_type(s, "Cylinder")

    table = s.curvature([face], samples=samples)

    assert table.k_max[0] == pytest.approx(1.0 / radius, rel=CURVED_RTOL)
    assert table.k_max[0] > 0.0


@pytest.mark.parametrize("samples", SAMPLE_DENSITIES)
def test_curvature_of_a_torus_is_one_over_its_tube_radius(samples: int) -> None:
    ring, tube = 5.0, 1.5
    s = Session()
    s.add_torus(ring, tube)
    face = face_of_type(s, "Torus")

    table = s.curvature([face], samples=samples)

    assert table.k_max[0] == pytest.approx(1.0 / tube, rel=CURVED_RTOL)


@pytest.mark.parametrize("samples", SAMPLE_DENSITIES)
def test_curvature_of_a_cone_matches_the_grids_closed_form(
    cone: Session, samples: int
) -> None:
    """Gate: the sampled peak is exact at every density, so there is nothing to hand-wave.

    ``samples=1`` is the single-centre-sample behaviour the port exists to fix, and it is
    included deliberately: it must reproduce the wrong answer exactly, which is what makes
    the improvement at higher densities a measurement rather than a claim.
    """
    face = face_of_type(cone, "Cone")

    table = cone.curvature([face], samples=samples)

    assert table.k_max[0] == pytest.approx(cone_peak_curvature(samples), rel=CURVED_RTOL)
    assert table.samples_used[0] == samples * samples


def test_the_centre_sample_of_a_cone_is_provably_wrong(cone: Session) -> None:
    """The face the requirement asks for: one where a single centre sample misreads badly."""
    face = face_of_type(cone, "Cone")
    analytic_peak = math.cos(CONE_HALF_ANGLE) / CONE_R2

    centre_only = cone.curvature([face], samples=1).k_max[0]
    gridded = cone.curvature([face], samples=64).k_max[0]

    assert centre_only == pytest.approx(
        math.cos(CONE_HALF_ANGLE) / (0.5 * (CONE_R1 + CONE_R2)), rel=CURVED_RTOL
    )
    assert centre_only < 0.45 * analytic_peak
    assert gridded > 0.97 * analytic_peak


def test_curvature_converges_upwards_as_the_grid_refines(cone: Session) -> None:
    face = face_of_type(cone, "Cone")

    found = [cone.curvature([face], samples=n).k_max[0] for n in (1, 4, 16, 64)]

    assert found == sorted(found)
    assert found[-1] < math.cos(CONE_HALF_ANGLE) / CONE_R2


def test_curvature_reports_where_the_peak_is_not_only_how_large(cone: Session) -> None:
    """The location is half the answer: a sizing field needs to know *where* to refine."""
    face = face_of_type(cone, "Cone")

    table = cone.curvature([face], samples=32)

    radius = math.hypot(table.xyz[0, 0], table.xyz[0, 1])
    assert table.uv.shape == (1, 2)
    assert radius == pytest.approx(math.cos(CONE_HALF_ANGLE) / table.k_max[0], rel=1e-6)
    assert radius < 0.5 * (CONE_R1 + CONE_R2), "the peak must be at the narrow end"


def test_curvature_skips_samples_outside_the_faces_trimming(bored_box: Session) -> None:
    """A sample inside a hole is on the surface but not on the face.

    Taking it would report a curvature the face does not have. The hole covers a known
    fraction of the face, so the number of discarded samples is predictable.
    """
    table = bored_box.mass_properties(ids_of(bored_box, EntityKind.FACE))
    holed = [
        EntityId(int(i))
        for i, m in zip(table.ids, table.measure)
        if BOX_DX * BOX_DY - 1.0 < m < BOX_DX * BOX_DY
    ]
    assert len(holed) == 2, "the fixture must have two faces carrying the hole"
    samples = 21

    result = bored_box.curvature(holed, samples=samples)

    hole_fraction = math.pi * HOLE_RADIUS**2 / (BOX_DX * BOX_DY)
    expected_used = samples**2 * (1.0 - hole_fraction)
    assert np.all(result.samples_used < samples**2)
    assert np.all(np.abs(result.samples_used - expected_used) < 0.02 * samples**2)


def test_curvature_rejects_a_sample_count_below_one(cone: Session) -> None:
    face = face_of_type(cone, "Cone")

    with pytest.raises(ps.PysmeshError, match="samples must be"):
        cone.curvature([face], samples=0)


def test_curvature_rejects_an_empty_face_list(cone: Session) -> None:
    with pytest.raises(ps.PysmeshError, match="at least one"):
        cone.curvature([])


# ================================================================== Projection ==


def test_projecting_onto_a_sphere_finds_the_nearest_surface_point() -> None:
    radius = 4.0
    s = Session()
    s.add_sphere(radius)
    face = face_of_type(s, "Sphere")

    result = s.project_on_face(face, [(10.0, 0.0, 0.0)])

    assert result.distance[0] == pytest.approx(10.0 - radius, rel=CURVED_RTOL)
    assert result.points[0] == pytest.approx((radius, 0.0, 0.0), abs=1e-7)


def test_a_projections_parameters_reproduce_its_point(box: Session) -> None:
    """Ties the two surface queries together: the uv it returns must evaluate back."""
    face = ids_of(box, EntityKind.FACE)[0]
    query = [(1.0, 2.0, 3.0)]

    projected = box.project_on_face(face, query)
    evaluated = box.surface_at(face, projected.uv)

    assert evaluated.points[0] == pytest.approx(projected.points[0], abs=1e-9)


def test_projecting_rejects_points_that_are_not_triples(box: Session) -> None:
    face = ids_of(box, EntityKind.FACE)[0]

    with pytest.raises(ps.PysmeshError, match=r"\(N, 3\)"):
        box.project_on_face(face, np.zeros((2, 2)))


# ============================================================ Spatial search ==


def test_entities_in_box_finds_everything_inside_a_generous_box(box: Session) -> None:
    hits = box.entities_in_box(
        EntityKind.FACE, (-1.0, -1.0, -1.0), (BOX_DX + 1, BOX_DY + 1, BOX_DZ + 1)
    )

    assert sorted(hits.tolist()) == sorted(int(i) for i in box.entities(EntityKind.FACE))


def test_a_strict_search_excludes_a_partly_overlapping_entity(box: Session) -> None:
    """Containment and overlap are different questions, and both get asked."""
    half = (BOX_DX + 1.0, BOX_DY + 1.0, BOX_DZ / 2.0)

    overlapping = box.entities_in_box(EntityKind.FACE, (-1.0, -1.0, -1.0), half)
    contained = box.entities_in_box(
        EntityKind.FACE, (-1.0, -1.0, -1.0), half, strict=True
    )

    assert set(contained.tolist()) < set(overlapping.tolist())
    assert contained.size == 1  # only the z = 0 face lies wholly below the cut


def test_entities_in_box_rejects_an_inverted_box(box: Session) -> None:
    with pytest.raises(ps.PysmeshError, match="max"):
        box.entities_in_box(EntityKind.FACE, (1.0, 0.0, 0.0), (0.0, 1.0, 1.0))


# ============================================================== Containment ==


def test_containment_separates_inside_from_outside_and_on(box: Session) -> None:
    solid = ids_of(box, EntityKind.SOLID)[0]
    points = [BOX_CENTRE, (100.0, 0.0, 0.0), (0.0, BOX_DY / 2.0, BOX_DZ / 2.0)]

    mask = box.contains([solid], points)

    assert mask.shape == (1, 3)
    assert mask[0, 0], "the body centre is inside"
    assert not mask[0, 1], "a far point is outside"
    assert not mask[0, 2], "a point on the wall is not strictly inside"


def test_containment_answers_for_several_solids_at_once() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(20.0, 0.0, 0.0))
    first, second = ids_of(s, EntityKind.SOLID)
    points = [BOX_CENTRE, (20.0 + BOX_DX / 2.0, BOX_DY / 2.0, BOX_DZ / 2.0)]

    mask = s.contains([first, second], points)

    assert mask.shape == (2, 2)
    assert np.array_equal(mask, np.eye(2, dtype=bool))


def test_containment_rejects_an_entity_that_is_not_a_solid(box: Session) -> None:
    face = ids_of(box, EntityKind.FACE)[0]

    with pytest.raises(ps.PysmeshError, match="not a SOLID"):
        box.contains([face], [BOX_CENTRE])


def test_containment_rejects_a_non_positive_tolerance(box: Session) -> None:
    solid = ids_of(box, EntityKind.SOLID)[0]

    with pytest.raises(ps.PysmeshError, match="tol"):
        box.contains([solid], [BOX_CENTRE], tol=0.0)


# ========================================================= Analytic surface parameters ==


def test_surface_parameters_types_agree_with_entity_types(bored_box: Session) -> None:
    faces = ids_of(bored_box, EntityKind.FACE)

    params = bored_box.surface_parameters(faces)
    table = bored_box.entity_types(EntityKind.FACE)

    by_id = dict(zip(table.ids.tolist(), table.types))
    assert params.ids.tolist() == [int(f) for f in faces]
    assert params.types == tuple(by_id[int(f)] for f in faces)


def test_surface_parameters_read_a_bores_radius_off_the_surface(
    bored_box: Session,
) -> None:
    """The headline use: a hole's size, exact rather than estimated from a box or a sample."""
    bore = face_of_type(bored_box, "Cylinder")

    params = bored_box.surface_parameters([bore])

    assert params.radius1[0] == pytest.approx(HOLE_RADIUS, rel=EXACT_RTOL)
    assert np.isnan(params.radius2[0])
    assert np.isnan(params.half_angle[0])
    assert params.axis[0] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)


def test_surface_parameters_of_a_cone_reproduce_its_taper(cone: Session) -> None:
    """radius1 is the radius at ``origin``; half_angle carries the taper, sign included.

    The gate is the cone's own definition -- ``r(t) = radius1 + t * tan(half_angle)`` -- so it
    fails on a magnitude-only half_angle, which is the plausible wrong implementation.
    """
    face = face_of_type(cone, "Cone")

    params = cone.surface_parameters([face])

    r_at_far_end = params.radius1[0] + CONE_H * math.tan(params.half_angle[0])
    assert params.radius1[0] == pytest.approx(CONE_R1, rel=EXACT_RTOL)
    assert r_at_far_end == pytest.approx(CONE_R2, rel=CURVED_RTOL)
    assert abs(params.half_angle[0]) == pytest.approx(CONE_HALF_ANGLE, rel=CURVED_RTOL)


def test_surface_parameters_of_a_sphere_give_the_radius_its_area_implies() -> None:
    """Checked against the closed-form area 4*pi*R^2, measured independently."""
    s = Session()
    s.add_sphere(SPHERE_R)
    face = face_of_type(s, "Sphere")

    params = s.surface_parameters([face])
    area = s.mass_properties([face]).measure[0]

    assert params.radius1[0] == pytest.approx(SPHERE_R, rel=EXACT_RTOL)
    assert area == pytest.approx(4.0 * math.pi * params.radius1[0] ** 2, rel=CURVED_RTOL)
    assert np.isnan(params.radius2[0])


def test_surface_parameters_of_a_torus_give_both_radii_the_right_way_round() -> None:
    """Major and minor are not interchangeable.

    ``4*pi^2*R*r`` is symmetric in the two, so the area alone cannot say which is which; the
    ordering is asserted alongside it.
    """
    s = Session()
    s.add_torus(TORUS_MAJOR, TORUS_MINOR)
    face = face_of_type(s, "Torus")

    params = s.surface_parameters([face])
    area = s.mass_properties([face]).measure[0]

    assert params.radius1[0] == pytest.approx(TORUS_MAJOR, rel=EXACT_RTOL)
    assert params.radius2[0] == pytest.approx(TORUS_MINOR, rel=EXACT_RTOL)
    assert params.radius1[0] > params.radius2[0]
    assert area == pytest.approx(
        4.0 * math.pi**2 * TORUS_MAJOR * TORUS_MINOR, rel=CURVED_RTOL
    )


def test_a_planes_axis_and_reversed_flag_give_the_outward_normal(box: Session) -> None:
    """The frame is the surface's, unflipped, so ``axis`` alone is not the outward normal.

    ``reversed`` is what closes the gap, and it must agree with the flip
    :meth:`Session.surface_at` already applies -- on every face, not on average.
    """
    faces = ids_of(box, EntityKind.FACE)

    params = box.surface_parameters(faces)

    for i, face in enumerate(faces):
        sign = -1.0 if params.reversed[i] else 1.0
        expected = box.surface_at(face, face_centre_uv(box, face)).normals[0]
        assert params.axis[i] * sign == pytest.approx(expected, abs=1e-12)


def test_a_planes_origin_lies_on_the_plane(box: Session) -> None:
    faces = ids_of(box, EntityKind.FACE)

    params = box.surface_parameters(faces)
    centroids = box.mass_properties(faces).centroid

    for i in range(len(faces)):
        in_plane = centroids[i] - params.origin[i]
        assert float(np.dot(in_plane, params.axis[i])) == pytest.approx(0.0, abs=1e-9)
        assert np.linalg.norm(params.axis[i]) == pytest.approx(1.0, rel=EXACT_RTOL)
        assert float(np.dot(params.axis[i], params.ref_dir[i])) == pytest.approx(
            0.0, abs=1e-12
        )


def test_a_free_form_face_reports_no_parameters_at_all() -> None:
    """NaN, not 0.0 -- the distinction a size filter depends on.

    A zero radius passes ``radius1 < 1.0``, so a filter looking for small fillets would
    collect every free-form face in the model. NaN fails every comparison instead.
    """
    s = Session()
    s.add_polyline(FREEFORM_RIM, closed=True)
    s.make_filling(ids_of(s, EntityKind.EDGE))
    face = face_of_type(s, "BSpline")

    params = s.surface_parameters([face])

    assert np.all(np.isnan(params.origin[0]))
    assert np.all(np.isnan(params.axis[0]))
    assert np.all(np.isnan(params.ref_dir[0]))
    assert np.isnan(params.radius1[0])
    assert not bool(params.radius1[0] < 1.0)  # the filter must not pick it up


def test_surface_parameters_of_an_extrusion_name_its_sweep_line() -> None:
    """OCCT may store the basis direction negated.

    The sweep line is therefore contract; its sense is not.
    """
    s = Session()
    profile = s.add_spline(EXTRUDE_PROFILE)
    s.extrude([EntityId(int(i)) for i in profile.created], EXTRUDE_VECTOR)
    face = face_of_type(s, "Extrusion")

    params = s.surface_parameters([face])

    unit = np.asarray(EXTRUDE_VECTOR) / np.linalg.norm(EXTRUDE_VECTOR)
    assert abs(float(np.dot(params.axis[0], unit))) == pytest.approx(1.0, rel=CURVED_RTOL)
    assert np.all(np.isnan(params.origin[0]))
    assert np.isnan(params.radius1[0])


def test_surface_parameters_of_a_revolution_give_only_its_axis() -> None:
    """A revolved free-form profile has an axis and no radius: the radius varies along it."""
    s = Session()
    profile = s.add_spline(REVOLVE_PROFILE)
    s.revolve(
        [EntityId(int(i)) for i in profile.created], (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    )
    face = face_of_type(s, "Revolution")

    params = s.surface_parameters([face])

    assert params.origin[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert abs(float(params.axis[0][2])) == pytest.approx(1.0, rel=EXACT_RTOL)
    assert np.isnan(params.radius1[0])
    assert np.all(np.isnan(params.ref_dir[0]))


def test_surface_parameters_follow_the_order_named(bored_box: Session) -> None:
    faces = ids_of(bored_box, EntityKind.FACE)
    shuffled = list(reversed(faces))

    params = bored_box.surface_parameters(shuffled)

    assert params.ids.tolist() == [int(f) for f in shuffled]


def test_surface_parameters_rejects_an_edge_id(box: Session) -> None:
    edge = ids_of(box, EntityKind.EDGE)[0]

    with pytest.raises(ps.PysmeshError, match="not a FACE"):
        box.surface_parameters([edge])


def test_a_bores_outward_normal_points_into_the_hole(bored_box: Session) -> None:
    """The registry may hold a face in the orientation an operation's history produced.

    Inside a cut result the bore is REVERSED, and reading the history's copy makes the
    outward normal point into the material -- the opposite of what every consumer of a normal
    means. Gated by stepping off the surface both ways and asking the solid which side it
    owns, so it cannot be satisfied by agreeing with a stored flag.
    """
    solid = ids_of(bored_box, EntityKind.SOLID)[0]
    bore = face_of_type(bored_box, "Cylinder")
    step = 1e-3

    sample = bored_box.surface_at(bore, face_centre_uv(bored_box, bore))
    point, normal = sample.points[0], sample.normals[0]

    outward = bored_box.contains([solid], [point + step * normal])
    inward = bored_box.contains([solid], [point - step * normal])
    assert not bool(outward[0, 0]), "the normal points into the material"
    assert bool(inward[0, 0])
    assert bool(bored_box.surface_parameters([bore]).reversed[0])


# ================================================================== Wire loops ==


def test_a_box_face_has_one_wire_of_four_edges(box: Session) -> None:
    faces = ids_of(box, EntityKind.FACE)

    wires = box.face_wires(faces)

    assert wires.face_id.tolist() == [int(f) for f in faces]
    assert wires.is_outer.all()
    assert wires.ordered.all()
    assert (wires.edge_range[:, 1] - wires.edge_range[:, 0]).tolist() == [4] * 6
    assert wires.edge_id.size == 24


def test_a_bored_faces_hole_is_a_separate_inner_wire(bored_box: Session) -> None:
    """The whole point of the query: adjacency cannot tell the hole from the boundary."""
    faces = ids_of(bored_box, EntityKind.FACE)

    wires = bored_box.face_wires(faces)

    inner = np.flatnonzero(~wires.is_outer)
    assert inner.size == 2, "the two end faces each carry one hole"
    for row in inner:
        lo, hi = wires.edge_range[row]
        hole = [EntityId(int(e)) for e in wires.edge_id[lo:hi]]
        assert len(hole) == 1
        length = bored_box.mass_properties(hole).measure[0]
        assert length == pytest.approx(TAU * HOLE_RADIUS, rel=CURVED_RTOL)


def test_the_hole_edges_are_circles_of_the_bores_radius(bored_box: Session) -> None:
    """Cross-checks the two new queries against each other.

    The loop the hole is bounded by must be a circle of exactly the radius the bore's
    surface reports.
    """
    faces = ids_of(bored_box, EntityKind.FACE)
    bore = face_of_type(bored_box, "Cylinder")

    wires = bored_box.face_wires(faces)
    radius = bored_box.surface_parameters([bore]).radius1[0]
    edge_types = bored_box.entity_types(EntityKind.EDGE)

    named = dict(zip(edge_types.ids.tolist(), edge_types.types))
    for row in np.flatnonzero(~wires.is_outer):
        lo, hi = wires.edge_range[row]
        edge = EntityId(int(wires.edge_id[lo]))
        assert named[int(edge)] == "Circle"
        length = bored_box.mass_properties([edge]).measure[0]
        assert length == pytest.approx(TAU * radius, rel=CURVED_RTOL)


def test_every_face_has_exactly_one_outer_wire(bored_box: Session) -> None:
    faces = ids_of(bored_box, EntityKind.FACE)

    wires = bored_box.face_wires(faces)

    for face in faces:
        rows = wires.face_id == int(face)
        assert int(wires.is_outer[rows].sum()) == 1, f"face {face}"


def test_a_faces_wires_partition_its_adjacency_edges(bored_box: Session) -> None:
    """Loops and the flat edge set are two readings of one boundary, so they must agree."""
    faces = ids_of(bored_box, EntityKind.FACE)

    wires = bored_box.face_wires(faces)
    pairs = bored_box.adjacency(EntityKind.FACE, EntityKind.EDGE)

    for face in faces:
        rows = np.flatnonzero(wires.face_id == int(face))
        from_wires: list[int] = []
        for row in rows:
            lo, hi = wires.edge_range[row]
            from_wires.extend(int(e) for e in wires.edge_id[lo:hi])
        from_adjacency = set(pairs.related[pairs.ids == int(face)].tolist())
        assert len(from_wires) == len(set(from_wires)), f"face {face} lists an edge twice"
        assert set(from_wires) == from_adjacency, f"face {face}"


def test_a_seam_edge_is_listed_once_in_its_wire(bored_box: Session) -> None:
    """The bore is closed, so its single wire crosses the seam twice.

    An id cannot carry an orientation, so the seam is listed once: two circles plus one
    seam edge, three distinct ids.
    """
    bore = face_of_type(bored_box, "Cylinder")

    wires = bored_box.face_wires([bore])

    assert wires.face_id.size == 1
    lo, hi = wires.edge_range[0]
    edges = wires.edge_id[lo:hi].tolist()
    assert len(edges) == 3
    assert len(set(edges)) == 3


def test_an_ordered_wire_is_a_closed_chain_of_edges(box: Session) -> None:
    """``ordered`` claims a traversal.

    Consecutive edges must therefore share a vertex, and the last must close back onto the
    first.
    """
    faces = ids_of(box, EntityKind.FACE)

    wires = box.face_wires(faces)
    pairs = box.adjacency(EntityKind.EDGE, EntityKind.VERTEX)

    ends = {
        int(e): set(pairs.related[pairs.ids == int(e)].tolist())
        for e in set(pairs.ids.tolist())
    }
    for row in range(len(wires.face_id)):
        assert bool(wires.ordered[row])
        lo, hi = wires.edge_range[row]
        loop = [int(e) for e in wires.edge_id[lo:hi]]
        for a, b in zip(loop, loop[1:] + loop[:1]):
            assert ends[a] & ends[b], f"edges {a} and {b} are not joined"


def test_face_wires_follow_the_order_named(bored_box: Session) -> None:
    faces = ids_of(bored_box, EntityKind.FACE)
    shuffled = list(reversed(faces))

    wires = bored_box.face_wires(shuffled)

    seen: list[int] = []
    for face in wires.face_id.tolist():
        if face not in seen:
            seen.append(face)
    assert seen == [int(f) for f in shuffled]


def test_the_edge_ranges_tile_the_edge_array(bored_box: Session) -> None:
    """CSR contract: the ranges are contiguous, in order, and cover every id exactly once."""
    wires = bored_box.face_wires(ids_of(bored_box, EntityKind.FACE))

    assert wires.edge_range.dtype == np.int32
    assert wires.edge_range[0, 0] == 0
    assert int(wires.edge_range[-1, 1]) == wires.edge_id.size
    assert np.array_equal(wires.edge_range[1:, 0], wires.edge_range[:-1, 1])
    assert np.all(wires.edge_range[:, 1] > wires.edge_range[:, 0])


def test_face_wires_of_no_faces_is_empty(box: Session) -> None:
    wires = box.face_wires([])

    assert wires.face_id.size == 0
    assert wires.edge_id.size == 0
    assert wires.edge_range.shape == (0, 2)


def test_face_wires_rejects_an_edge_id(box: Session) -> None:
    edge = ids_of(box, EntityKind.EDGE)[0]

    with pytest.raises(ps.PysmeshError, match="not a FACE"):
        box.face_wires([edge])


# ============================================== The root the queries resolve against ==


def test_a_query_after_an_edit_sees_the_new_topology(box: Session) -> None:
    """The face queries resolve ids against the live root, and that resolution is cached.

    A cache that outlived its root would answer from the shape before the edit, so the gate
    queries once to fill it, edits, and requires the second answer to describe the result.
    """
    before = box.surface_parameters(ids_of(box, EntityKind.FACE))
    assert len(before.ids) == 6

    box.add_cylinder(HOLE_RADIUS, BOX_DZ + 2.0, origin=(*BOX_CENTRE[:2], -1.0))
    solid, tool = ids_of(box, EntityKind.SOLID)
    box.cut([solid], [tool])

    after = box.surface_parameters(ids_of(box, EntityKind.FACE))
    wires = box.face_wires(ids_of(box, EntityKind.FACE))
    assert "Cylinder" in after.types
    assert int((~wires.is_outer).sum()) == 2


def test_a_query_after_a_restore_sees_the_restored_topology(box: Session) -> None:
    """Restore puts an earlier root back, and the queries must follow it back."""
    mark = box.snapshot()
    box.add_cylinder(HOLE_RADIUS, BOX_DZ + 2.0, origin=(*BOX_CENTRE[:2], -1.0))
    solid, tool = ids_of(box, EntityKind.SOLID)
    box.cut([solid], [tool])
    assert "Cylinder" in box.surface_parameters(ids_of(box, EntityKind.FACE)).types

    box.restore(mark)

    after = box.surface_parameters(ids_of(box, EntityKind.FACE))
    wires = box.face_wires(ids_of(box, EntityKind.FACE))
    assert after.types == ("Plane",) * 6
    assert bool(wires.is_outer.all())


# ====================================================== Queries are queries ==


def test_no_query_advances_the_session(box: Session) -> None:
    """A query must not consume an operation index or touch the id space.

    Worth asserting rather than assuming: every query goes through the same class as the
    operations, and an accidental commit would be invisible until a name stopped resolving.
    """
    solid = ids_of(box, EntityKind.SOLID)[0]
    face = ids_of(box, EntityKind.FACE)[0]
    edge = ids_of(box, EntityKind.EDGE)[0]
    ops_before = box.op_count
    ids_before = box.issued_id_count

    box.entity_types(EntityKind.FACE)
    box.bounding_boxes(EntityKind.EDGE)
    box.mass_properties([solid, face, edge])
    box.face_parameter_bounds([face])
    box.edge_parameter_bounds([edge])
    box.adjacency(EntityKind.FACE, EntityKind.EDGE)
    box.surface_parameters([face])
    box.face_wires([face])
    box.surface_at(face, face_centre_uv(box, face))
    box.curvature([face], samples=4)
    box.project_on_face(face, [BOX_CENTRE])
    box.entities_in_box(EntityKind.FACE, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    box.contains([solid], [BOX_CENTRE])

    assert box.op_count == ops_before
    assert box.issued_id_count == ids_before
    assert box.state_op_index == 1
