# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""Gates for mesh search: location, ray casting, classification, offset and slot cutting.

Three of these answers have no counterpart in a surface-array pipeline, and each is gated
against a value computed outside the library:

* **Ray casting** against a torus, whose intersection count is known in closed form — a line
  through the hole meets it 0 times, one through the tube 2, one across the whole ring 4. The
  falsification is in the same table: a line that misses entirely must report 0, so an
  implementation that reported "some faces" for everything would fail.
* **Point classification** against a closed shell, with a point deliberately *on* a face and
  one far outside, plus the case the requirement does not name but that a caller will meet:
  a mesh that is not closed, where the honest answer is that there is no inside.
* **The offset**, against the analytic offset of a sphere: every node of the offset surface
  must land at the source radius plus the distance.

Fixture sizing follows the project rule: 3 x 7 x 11, never a unit cube.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    ElementDimension,
    ElementType,
    Hexa3D,
    Mefisto2D,
    Mesher,
    NumberOfSegments,
    PointState,
    PysmeshError,
    Quadrangle2D,
    Regular1D,
    Session,
)

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0

TORUS_MAJOR: float = 5.0
TORUS_MINOR: float = 2.0

SPHERE_RADIUS: float = 4.0


# ---- Fixtures ---------------------------------------------------------------------------- #


def _mesher(shape: ps.Shape, segments: int, volumes: bool = True) -> Mesher:
    """A computed mesher on one shape."""
    mesher = Mesher(shape)
    mesher.assign(Regular1D())
    mesher.assign(NumberOfSegments(count=segments))
    mesher.assign(Quadrangle2D())
    if volumes:
        mesher.assign(Hexa3D())
    mesher.compute()
    return mesher


@pytest.fixture()
def box_mesher() -> Iterator[Mesher]:
    """A 3 x 3 x 3 hexahedral mesh of the box, with its skin."""
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    mesher = _mesher(ps.load_brep(session.brep()), 3)
    yield mesher
    mesher.release()


@pytest.fixture()
def open_mesher() -> Iterator[Mesher]:
    """A single face: a surface that is not closed, so nothing is inside it."""
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), BOX_DX, BOX_DY)
    mesher = _mesher(ps.load_brep(session.brep()), 2, volumes=False)
    yield mesher
    mesher.release()


@pytest.fixture(scope="module")
def torus_mesh() -> Iterator[Mesher]:
    """A triangulated torus, whose ray intersections are known in closed form."""
    session = Session()
    session.add_torus(TORUS_MAJOR, TORUS_MINOR)
    mesher = _mesher(ps.load_brep(session.brep()), 24, volumes=False)
    mesher.quad_to_tri()
    yield mesher
    mesher.release()


# ---- Location ----------------------------------------------------------------------------- #


def test_find_elements_by_point_locates_the_cell_containing_an_interior_point(
    box_mesher: Mesher,
) -> None:
    # The centre of the cell at the box's own corner, computed from the fixture rather than
    # read from the mesh.
    point = np.array([[BOX_DX / 6.0, BOX_DY / 6.0, BOX_DZ / 6.0]])

    found = box_mesher.find_elements_by_point(point, ElementDimension.VOLUME)

    assert found.at(0).shape[0] == 1
    cell = int(found.at(0)[0])
    mesh = box_mesher.mesh()
    row = int(np.flatnonzero(mesh.element_id == cell)[0])
    corners = mesh.node_coords[mesh.nodes_of(row)]
    assert corners[:, 0].min() == pytest.approx(0.0)
    assert corners[:, 0].max() == pytest.approx(BOX_DX / 3.0)


def test_find_elements_by_point_reports_nothing_for_a_point_outside(
    box_mesher: Mesher,
) -> None:
    found = box_mesher.find_elements_by_point(
        np.array([[100.0, 100.0, 100.0]]), ElementDimension.VOLUME
    )

    assert found.at(0).shape[0] == 0


def test_find_elements_by_point_answers_a_batch_in_order(box_mesher: Mesher) -> None:
    points = np.array(
        [
            [BOX_DX / 6.0, BOX_DY / 6.0, BOX_DZ / 6.0],
            [100.0, 100.0, 100.0],
            [BOX_DX / 2.0, BOX_DY / 2.0, BOX_DZ / 2.0],
        ]
    )

    found = box_mesher.find_elements_by_point(points, ElementDimension.VOLUME)

    assert found.offsets.shape[0] == 4
    assert found.at(0).shape[0] == 1
    assert found.at(1).shape[0] == 0
    assert found.at(2).shape[0] >= 1


def test_find_closest_returns_a_cell_for_a_point_far_outside(box_mesher: Mesher) -> None:
    ids = box_mesher.find_closest(
        np.array([[-50.0, BOX_DY / 2.0, BOX_DZ / 2.0]]), ElementDimension.VOLUME
    )

    assert int(ids[0]) != 0


def test_closest_distance_to_a_volume_cell_matches_the_geometry(
    box_mesher: Mesher,
) -> None:
    # A point 10 units along -x from the middle of the x = 0 wall. The nearest cell face is
    # that wall, so the distance is exactly 10.
    point = np.array([[-10.0, BOX_DY / 2.0, BOX_DZ / 2.0]])

    answer = box_mesher.closest_distance(point, ElementDimension.VOLUME)

    assert float(answer.distances[0]) == pytest.approx(10.0, abs=1e-9)
    assert float(answer.closest_points[0][0]) == pytest.approx(0.0, abs=1e-9)


def test_closest_distance_is_zero_inside_a_cell(box_mesher: Mesher) -> None:
    point = np.array([[BOX_DX / 2.0, BOX_DY / 2.0, BOX_DZ / 2.0]])

    answer = box_mesher.closest_distance(point, ElementDimension.VOLUME)

    assert float(answer.distances[0]) == pytest.approx(0.0, abs=1e-9)


def test_project_points_lands_on_the_wall_it_faces(box_mesher: Mesher) -> None:
    point = np.array([[-5.0, BOX_DY / 2.0, BOX_DZ / 2.0]])

    projected = box_mesher.project_points(point, ElementDimension.FACE)

    assert float(projected.points[0][0]) == pytest.approx(0.0, abs=1e-9)
    assert float(projected.points[0][1]) == pytest.approx(BOX_DY / 2.0, abs=1e-9)
    assert int(projected.ids[0]) != 0


def test_find_closest_refuses_a_batch_that_is_not_three_columns(
    box_mesher: Mesher,
) -> None:
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        box_mesher.find_closest(np.zeros((4, 2)))


def test_find_closest_refuses_an_unknown_family(box_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="Unknown element family"):
        box_mesher.find_closest(np.zeros((1, 3)), 99)  # type: ignore[arg-type]


# ---- Point classification ----------------------------------------------------------------- #


def test_point_state_classifies_inside_outside_and_on(box_mesher: Mesher) -> None:
    points = np.array(
        [
            [BOX_DX / 2.0, BOX_DY / 2.0, BOX_DZ / 2.0],  # deep inside
            [-100.0, -100.0, -100.0],  # far outside
            [0.0, BOX_DY / 2.0, BOX_DZ / 2.0],  # exactly on the x = 0 wall
        ]
    )

    states = box_mesher.point_state(points)

    assert int(states[0]) == PointState.IN
    assert int(states[1]) == PointState.OUT
    assert int(states[2]) == PointState.ON


def test_point_state_on_an_open_surface_reports_everything_outside(
    open_mesher: Mesher,
) -> None:
    # An open surface has no inside, and the searcher says so by classifying every point as
    # OUT rather than as UNKNOWN. Pinned because the difference matters: a caller cannot tell
    # "outside a closed body" from "the body was never closed" by reading one answer, and has
    # to check for a free border itself.
    states = open_mesher.point_state(
        np.array([[BOX_DX / 2.0, BOX_DY / 2.0, 5.0], [BOX_DX / 2.0, BOX_DY / 2.0, 0.0]])
    )

    assert int(states[0]) == PointState.OUT
    assert int(states[1]) == PointState.OUT
    # What a caller has to check instead: a surface with a free edge is not closed, so the
    # classification above is answering about a body that is not there.
    assert open_mesher.select(ps.FreeEdges()).count > 0


def test_point_state_agrees_with_the_brep_classifier(box_mesher: Mesher) -> None:
    # A second oracle, and one this library did not compute from the mesh: the B-rep
    # classifier shipped for the same question.
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    points = np.array(
        [
            [BOX_DX / 2.0, BOX_DY / 2.0, BOX_DZ / 2.0],
            [BOX_DX * 2.0, BOX_DY / 2.0, BOX_DZ / 2.0],
            [BOX_DX / 4.0, BOX_DY / 4.0, BOX_DZ / 4.0],
        ]
    )

    states = box_mesher.point_state(points)
    brep = ps.point_in_solid(session.brep(), points)

    for state, inside in zip(states, brep, strict=True):
        assert (int(state) == PointState.IN) == bool(inside)


# ---- Region queries ----------------------------------------------------------------------- #


def test_elements_in_box_over_the_whole_model_returns_every_cell(
    box_mesher: Mesher,
) -> None:
    ids = box_mesher.elements_in_box(
        (-1.0, -1.0, -1.0), (BOX_DX + 1.0, BOX_DY + 1.0, BOX_DZ + 1.0),
        ElementDimension.VOLUME,
    )

    assert ids.shape[0] == 27


def test_elements_in_box_over_a_corner_returns_fewer(box_mesher: Mesher) -> None:
    ids = box_mesher.elements_in_box(
        (-1.0, -1.0, -1.0), (BOX_DX / 3.0 - 0.01, BOX_DY / 3.0 - 0.01, BOX_DZ / 3.0 - 0.01),
        ElementDimension.VOLUME,
    )

    assert ids.shape[0] == 1


def test_elements_in_sphere_around_the_centre_finds_the_middle_cell(
    box_mesher: Mesher,
) -> None:
    ids = box_mesher.elements_in_sphere(
        (BOX_DX / 2.0, BOX_DY / 2.0, BOX_DZ / 2.0), 0.1, ElementDimension.VOLUME
    )

    assert ids.shape[0] >= 1


def test_elements_in_sphere_far_from_the_mesh_finds_nothing(box_mesher: Mesher) -> None:
    ids = box_mesher.elements_in_sphere(
        (1000.0, 1000.0, 1000.0), 1.0, ElementDimension.VOLUME
    )

    assert ids.shape[0] == 0


def test_elements_in_box_refuses_an_inverted_box(box_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="inverted"):
        box_mesher.elements_in_box((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))


def test_elements_in_sphere_refuses_a_radius_of_zero(box_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="radius"):
        box_mesher.elements_in_sphere((0.0, 0.0, 0.0), 0.0)


# ---- Ray casting -------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "origin", "direction", "expected"),
    [
        # Up the torus axis, through the hole: the surface is never met.
        ("axis through the hole", (0.0, 0.0, -20.0), (0.0, 0.0, 1.0), 0),
        # Up through the tube itself: in one side, out the other.
        ("axial through the tube", (TORUS_MAJOR, 0.0, -20.0), (0.0, 0.0, 1.0), 2),
        # Across the whole ring in its own plane: through both tubes, four surfaces.
        ("across the ring", (-20.0, 0.0, 0.0), (1.0, 0.0, 0.0), 4),
        # In the plane but above it, missing the body entirely.
        ("clear of the body", (-20.0, 0.0, 5.0), (1.0, 0.0, 0.0), 0),
    ],
)
def test_ray_hits_matches_the_analytic_intersection_count_of_a_torus(
    torus_mesh: Mesher,
    name: str,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    expected: int,
) -> None:
    del name
    hits = torus_mesh.ray_hits(origin, direction)

    # `crossings` is the count of distinct positions along the ray. It is what a parity test
    # counts, and what the closed form predicts; `count` is above it wherever the ray strikes
    # an edge that several triangles share, which these axis-aligned rays do.
    assert hits.crossings == expected


def test_ray_hits_reports_its_hits_nearest_first(torus_mesh: Mesher) -> None:
    hits = torus_mesh.ray_hits((-20.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert bool(np.all(np.diff(hits.parameters) >= 0.0))


def test_ray_hits_lands_where_the_torus_analytically_is(torus_mesh: Mesher) -> None:
    # Across the ring: the four crossings are at |x| = R +- r, so the first is at 20 - (R + r).
    hits = torus_mesh.ray_hits((-20.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    distinct = np.unique(np.round(hits.parameters, 6))
    expected = np.array(
        [
            20.0 - (TORUS_MAJOR + TORUS_MINOR),
            20.0 - (TORUS_MAJOR - TORUS_MINOR),
            20.0 + (TORUS_MAJOR - TORUS_MINOR),
            20.0 + (TORUS_MAJOR + TORUS_MINOR),
        ]
    )
    assert distinct.shape[0] == 4
    assert np.allclose(distinct, expected, atol=0.05)


def test_ray_hits_is_a_half_line_and_ignores_what_is_behind_it(
    torus_mesh: Mesher,
) -> None:
    forward = torus_mesh.ray_hits((-20.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    backward = torus_mesh.ray_hits((-20.0, 0.0, 0.0), (-1.0, 0.0, 0.0))

    assert forward.crossings == 4
    assert backward.crossings == 0


def test_ray_hits_from_inside_the_tube_meets_the_wall_once(torus_mesh: Mesher) -> None:
    hits = torus_mesh.ray_hits((TORUS_MAJOR, 0.0, 0.0), (0.0, 0.0, 1.0))

    assert hits.crossings == 1
    assert float(hits.parameters.min()) == pytest.approx(TORUS_MINOR, abs=0.05)


def test_elements_near_line_is_a_broad_phase_and_says_so(torus_mesh: Mesher) -> None:
    # The broad phase answers about bounding boxes and about a line, not a half line, so it
    # reports more than the ray meets. Both properties are asserted, because a caller reading
    # it as a hit list would be wrong in both directions.
    candidates = torus_mesh.elements_near_line(
        (-20.0, 0.0, 0.0), (1.0, 0.0, 0.0), ElementDimension.FACE
    )
    hits = torus_mesh.ray_hits((-20.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert candidates.shape[0] >= hits.count
    assert set(hits.ids.tolist()) <= set(candidates.tolist())


def test_elements_near_line_far_from_the_mesh_finds_nothing(torus_mesh: Mesher) -> None:
    candidates = torus_mesh.elements_near_line(
        (1000.0, 1000.0, -1.0), (0.0, 0.0, 1.0), ElementDimension.FACE
    )

    assert candidates.shape[0] == 0


def test_ray_hits_refuses_a_zero_direction(box_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="zero vector"):
        box_mesher.ray_hits((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_ray_hits_refuses_a_tolerance_that_is_not_a_number(box_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="tolerance"):
        box_mesher.ray_hits((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), float("nan"))


# ---- Feature edges and patches ------------------------------------------------------------ #


def test_sharp_edges_finds_every_crease_of_a_box(box_mesher: Mesher) -> None:
    edges = box_mesher.sharp_edges(45.0)

    # A box has 12 model edges; at 3 segments each is 3 element edges long.
    assert edges.count == 12 * 3
    assert bool((edges.medium == 0).all())


def test_sharp_edges_at_a_shallow_angle_finds_the_same_creases(
    box_mesher: Mesher,
) -> None:
    # The box's creases are all 90 degrees, so lowering the threshold cannot add any and
    # raising it past 90 must lose them all. Both directions, so the angle is shown to matter.
    assert box_mesher.sharp_edges(10.0).count == 36
    assert box_mesher.sharp_edges(120.0).count == 0


def test_separate_faces_by_edges_recovers_the_boxs_six_faces(box_mesher: Mesher) -> None:
    edges = box_mesher.sharp_edges(45.0)

    patches = box_mesher.separate_faces_by_edges(edges)

    assert patches.count == 6
    assert sorted(patches.at(i).shape[0] for i in range(6)) == [9] * 6


def test_separate_faces_by_edges_with_no_edges_gives_one_patch(
    box_mesher: Mesher,
) -> None:
    empty = ps.SharpEdges(
        node1=np.zeros(0, dtype=np.int64),
        node2=np.zeros(0, dtype=np.int64),
        medium=np.zeros(0, dtype=np.int64),
    )

    patches = box_mesher.separate_faces_by_edges(empty)

    assert patches.count == 1


def test_separate_faces_by_edges_refuses_an_unknown_node(box_mesher: Mesher) -> None:
    bad = ps.SharpEdges(
        node1=np.array([10**9], dtype=np.int64),
        node2=np.array([10**9 + 1], dtype=np.int64),
        medium=np.array([0], dtype=np.int64),
    )

    with pytest.raises(PysmeshError, match="does not have"):
        box_mesher.separate_faces_by_edges(bad)


def test_sharp_edges_refuses_an_angle_outside_the_half_turn(box_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="between 0 and 180"):
        box_mesher.sharp_edges(200.0)


# ---- Merge diagnosis and slot cutting ----------------------------------------------------- #


def test_merge_obstruction_leaves_a_cell_alone_when_the_merge_misses_it(
    box_mesher: Mesher,
) -> None:
    mesh = box_mesher.mesh()
    row = int(np.flatnonzero(mesh.element_type == int(ElementType.HEXAHEDRON))[0])
    cell = int(mesh.element_id[row])
    own = {int(mesh.node_id[i]) for i in mesh.nodes_of(row)}
    elsewhere = [
        int(mesh.node_id[i])
        for i in range(mesh.node_count)
        if int(mesh.node_id[i]) not in own
    ][:2]

    answer = box_mesher.merge_obstruction(cell, [elsewhere])

    assert answer.nodes.shape[0] == 8
    assert answer.keep_apart.shape[0] == 0


def test_merge_obstruction_names_the_medium_node_a_merge_would_strand(
    box_mesher: Mesher,
) -> None:
    # The case the diagnosis exists for: on a second-order element, merging a medium node
    # with one of its own corners leaves the element invalid rather than degenerate, so the
    # merge has to be undone for those nodes.
    box_mesher.convert_to_quadratic()
    mesh = box_mesher.mesh()
    row = int(np.flatnonzero(mesh.element_type == int(ElementType.QUAD_EDGE))[0])
    segment = int(mesh.element_id[row])
    nodes = [int(mesh.node_id[i]) for i in mesh.nodes_of(row)]

    answer = box_mesher.merge_obstruction(segment, [[nodes[0], nodes[2]]])

    # Both entries name the surviving node: after the substitution the corner and the medium
    # are the same node, which is precisely what makes the element invalid.
    assert answer.keep_apart.shape[0] == 2
    assert set(answer.keep_apart.tolist()) == {nodes[0]}
    assert nodes[2] not in set(answer.nodes.tolist())


def test_merge_obstruction_reports_nothing_for_a_volume_cell(
    box_mesher: Mesher,
) -> None:
    # Pinned because it looks like a pass and is not one: the volume branch upstream
    # re-derives its comparison from the element's *current* nodes rather than from the
    # proposed ones, so it compares a thing with itself and can never report. A caller must
    # not read an empty answer here as "this merge is safe for the cell".
    mesh = box_mesher.mesh()
    row = int(np.flatnonzero(mesh.element_type == int(ElementType.HEXAHEDRON))[0])
    cell = int(mesh.element_id[row])
    corners = [int(mesh.node_id[i]) for i in mesh.nodes_of(row)]

    answer = box_mesher.merge_obstruction(cell, [[corners[0], corners[2]]])

    assert answer.nodes.shape[0] == 8
    assert answer.keep_apart.shape[0] == 0


def test_merge_obstruction_refuses_an_unknown_id(box_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="no element with id"):
        box_mesher.merge_obstruction(10**9, [])


def test_make_slot_cuts_a_band_into_a_triangle_mesh() -> None:
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    mesher = _mesher(ps.load_brep(session.brep()), 3, volumes=False)
    mesher.quad_to_tri()
    faces_before = mesher.mesh().count_of(ElementType.TRIANGLE)

    boundary = mesher.make_slot(0.3)

    assert boundary.node1.shape[0] > 0
    assert boundary.node1.shape[0] == boundary.node2.shape[0]
    assert mesher.mesh().count_of(ElementType.TRIANGLE) != faces_before
    mesher.release()


def test_make_slot_refuses_a_width_of_zero(box_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="width"):
        box_mesher.make_slot(0.0)


def test_make_slot_refuses_an_id_that_is_not_a_segment(box_mesher: Mesher) -> None:
    mesh = box_mesher.mesh()
    row = int(np.flatnonzero(mesh.element_type == int(ElementType.HEXAHEDRON))[0])

    with pytest.raises(PysmeshError, match="1-D element"):
        box_mesher.make_slot(0.3, [int(mesh.element_id[row])])


# ---- The offset, against the analytic offset of a sphere ---------------------------------- #


def _sphere_mesher(segments: int) -> Mesher:
    """A triangulated sphere, whose offset is known in closed form."""
    session = Session()
    session.add_sphere(SPHERE_RADIUS)
    mesher = Mesher(ps.load_brep(session.brep()))
    mesher.assign(Regular1D())
    mesher.assign(NumberOfSegments(count=segments))
    mesher.assign(Mefisto2D())
    mesher.compute()
    return mesher


def _live_radii(mesher: Mesher) -> NDArray[np.float64]:
    """Radii of the nodes the mesh's elements actually use.

    Removing the source faces leaves their nodes behind unused, so a radius taken over the
    whole node array would mix the two surfaces. Only the nodes an element references are
    part of the answer.
    """
    mesh = mesher.mesh()
    used: set[int] = set()
    for element in range(mesh.element_count):
        if mesh.element_type[element] != int(ElementType.TRIANGLE):
            continue
        used.update(int(i) for i in mesh.nodes_of(element))
    rows = np.array(sorted(used), dtype=np.int64)
    return np.asarray(np.linalg.norm(mesh.node_coords[rows], axis=1), dtype=np.float64)


def test_offset_of_a_sphere_lands_on_the_analytic_offset_sphere() -> None:
    mesher = _sphere_mesher(24)
    distance = 1.0

    mesher.offset(distance, copy_elements=False)

    radii = _live_radii(mesher)
    # Every node of the offset surface sits on the sphere of radius R + d. The source nodes
    # are on the sphere exactly, and the offset runs along the averaged normal, so the only
    # error is the facet approximation of that normal.
    # The tolerance is the facet approximation of the normal, not a fudge: the convergence
    # test below shows it shrinking as the source mesh is refined.
    assert float(radii.min()) == pytest.approx(SPHERE_RADIUS + distance, rel=0.05)
    assert float(radii.max()) == pytest.approx(SPHERE_RADIUS + distance, rel=0.05)
    mesher.release()


def test_a_negative_offset_of_a_sphere_shrinks_it_by_the_same_law() -> None:
    mesher = _sphere_mesher(24)
    distance = -1.0

    mesher.offset(distance, copy_elements=False)

    radii = _live_radii(mesher)
    assert float(radii.mean()) == pytest.approx(SPHERE_RADIUS + distance, rel=0.05)
    mesher.release()


def test_the_offset_converges_on_the_analytic_sphere_as_the_mesh_is_refined() -> None:
    # Without this the tolerance above would be a fixture property rather than a measurement
    # of the method.
    errors = []
    for segments in (8, 24):
        mesher = _sphere_mesher(segments)
        mesher.offset(1.0, copy_elements=False)
        errors.append(float(np.abs(_live_radii(mesher) - (SPHERE_RADIUS + 1.0)).max()))
        mesher.release()

    assert errors[1] < errors[0]


def test_every_search_refuses_a_released_mesher() -> None:
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    mesher = _mesher(ps.load_brep(session.brep()), 2)
    points = np.zeros((1, 3))
    mesher.release()

    for call in (
        lambda: mesher.find_elements_by_point(points),
        lambda: mesher.find_closest(points),
        lambda: mesher.closest_distance(points),
        lambda: mesher.project_points(points),
        lambda: mesher.point_state(points),
        lambda: mesher.elements_in_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        lambda: mesher.elements_in_sphere((0.0, 0.0, 0.0), 1.0),
        lambda: mesher.elements_near_line((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        lambda: mesher.ray_hits((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        lambda: mesher.sharp_edges(),
        lambda: mesher.merge_obstruction(1, []),
        lambda: mesher.make_slot(0.1),
    ):
        with pytest.raises(PysmeshError, match="released"):
            call()
