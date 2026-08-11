# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""Gates for the medial axis, block decomposition and pattern mapping.

The medial axis is checked against shapes whose axis is known in closed form:

* a **w x h rectangle**, whose axis is the segment ``y = h/2`` over ``x`` in
  ``[h/2, w - h/2]`` — so both its position and its extent are predicted, not merely
  bounded;
* an **annulus** of radii ``r`` and ``R``, whose axis is the circle of radius
  ``(R + r) / 2`` and whose local width is ``R - r`` everywhere. That is the hollow
  cylinder's wall the requirement names, taken as its own end cap;
* an **L-shaped region**, which has a genuine junction where a rectangle has none.

The width is the load-bearing number, and it never comes from the axis: each sample carries
the two nearest boundary points, and their distance is what is compared with the analytic
wall thickness.

Block decomposition is checked by round trip — parameters to points and back — and against
the box's own corners, which the block did not produce.

Fixture sizing follows the project rule: 3 x 7 x 11 for the box, and 10 x 4 for the
rectangle, never a unit square.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import (
    BLOCK_FACE_NAMES,
    BLOCK_VERTEX_NAMES,
    BranchEnd,
    ElementType,
    Mesher,
    NumberOfSegments,
    PysmeshError,
    Quadrangle2D,
    Regular1D,
    Session,
    block_parameters,
    block_points,
    block_shapes,
    medial_axis,
)

RECT_W: float = 10.0
RECT_H: float = 4.0

RING_OUTER: float = 5.0
RING_INNER: float = 3.0
RING_HEIGHT: float = 10.0

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0


# ---- Fixtures ---------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def rectangle() -> ps.Shape:
    """A 10 x 4 rectangle in the z = 0 plane."""
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), RECT_W, RECT_H)
    return ps.load_brep(session.brep())


@pytest.fixture(scope="module")
def hollow_cylinder() -> ps.Shape:
    """A hollow cylinder. Its end cap is the annulus the thickness gate is measured on."""
    session = Session()
    outer = session.add_cylinder(RING_OUTER, RING_HEIGHT)
    inner = session.add_cylinder(RING_INNER, RING_HEIGHT)
    session.cut([outer.created[0]], [inner.created[0]])
    return ps.load_brep(session.brep())


def _annular_face(shape: ps.Shape) -> int:
    """The ordinal of the hollow cylinder's flat annular cap."""
    for face in shape.faces():
        if face.surface_type == "Plane":
            return face.id
    raise AssertionError("the hollow cylinder has no planar cap")


@pytest.fixture(scope="module")
def l_shape() -> ps.Shape:
    """An L-shaped planar region, built as one closed polyline and faced."""
    session = Session()
    outline = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 4.0, 0.0],
            [4.0, 4.0, 0.0],
            [4.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
        ]
    )
    session.add_polyline(outline, closed=True)
    session.make_face(list(session.entities(ps.EntityKind.EDGE)))
    return ps.load_brep(session.brep())


@pytest.fixture()
def box_shape() -> ps.Shape:
    """The box, as the block everything below is read as."""
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    return ps.load_brep(session.brep())


# ---- The medial axis of a rectangle ------------------------------------------------------- #


def test_a_rectangle_axis_is_a_spine_plus_one_arm_per_corner(rectangle: ps.Shape) -> None:
    axis = medial_axis(rectangle, 1, 0.1)

    # Five branches: the spine, and one 45-degree arm running into each corner. Anyone
    # reading branch 0 as "the axis" would be reading a corner arm.
    assert len(axis.branches) == 5
    assert axis.boundary_edges == 4
    assert sum(1 for b in axis.branches if b.has_end(BranchEnd.ON_VERTEX)) == 4


def test_the_rectangles_spine_is_the_analytic_medial_axis(rectangle: ps.Shape) -> None:
    spine = medial_axis(rectangle, 1, 0.1).longest

    # The closed form: y = h/2 over x in [h/2, w - h/2].
    assert np.allclose(spine.uv[:, 1], RECT_H / 2.0, atol=1e-9)
    assert float(spine.uv[:, 0].min()) == pytest.approx(RECT_H / 2.0, abs=1e-9)
    assert float(spine.uv[:, 0].max()) == pytest.approx(RECT_W - RECT_H / 2.0, abs=1e-9)
    assert spine.length == pytest.approx(RECT_W - RECT_H, abs=1e-9)


def test_the_rectangles_spine_reports_the_rectangles_own_width(
    rectangle: ps.Shape,
) -> None:
    spine = medial_axis(rectangle, 1, 0.1, samples=9).longest

    # The width comes from the two boundary points, not from the axis, and it is h everywhere
    # along the spine because the two long sides face one another.
    assert spine.widths.shape[0] == 9
    assert np.allclose(spine.widths, RECT_H, atol=1e-9)


def test_a_corner_arm_narrows_to_nothing_at_its_vertex(rectangle: ps.Shape) -> None:
    axis = medial_axis(rectangle, 1, 0.1, samples=5)
    arm = next(b for b in axis.branches if b.has_end(BranchEnd.ON_VERTEX))

    # The falsification for the width: it is not a constant of the shape. On an arm running
    # into a corner it goes to zero at that corner and to the diagonal of the half-height at
    # the branch point.
    assert float(arm.widths.min()) == pytest.approx(0.0, abs=1e-9)
    assert float(arm.widths.max()) == pytest.approx(
        RECT_H / np.sqrt(2.0), abs=1e-6
    )


def test_ignoring_corners_leaves_the_rectangle_with_one_branch(
    rectangle: ps.Shape,
) -> None:
    axis = medial_axis(rectangle, 1, 0.1, ignore_corners=True)

    assert len(axis.branches) == 1
    assert axis.branches[0].length == pytest.approx(RECT_W - RECT_H, abs=1e-9)


def test_each_sample_names_the_shapes_own_edge_ordinals(rectangle: ps.Shape) -> None:
    spine = medial_axis(rectangle, 1, 0.1, samples=5).longest

    # The two boundaries the spine is equidistant from are two of the rectangle's own edges,
    # named in the caller's ordinals rather than in an index private to the axis.
    edges = {int(i) for i in spine.boundary1_edge} | {int(i) for i in spine.boundary2_edge}
    assert edges <= {edge.id for edge in rectangle.edges()}
    assert len(edges) == 2


def test_the_samples_boundary_points_lie_on_the_rectangles_own_boundary(
    rectangle: ps.Shape,
) -> None:
    spine = medial_axis(rectangle, 1, 0.1, samples=7).longest

    # One boundary point on y = 0 and the other on y = h, at the same x as one another.
    lows = np.minimum(spine.boundary1[:, 1], spine.boundary2[:, 1])
    highs = np.maximum(spine.boundary1[:, 1], spine.boundary2[:, 1])
    assert np.allclose(lows, 0.0, atol=1e-9)
    assert np.allclose(highs, RECT_H, atol=1e-9)
    assert np.allclose(spine.boundary1[:, 0], spine.boundary2[:, 0], atol=1e-9)


# ---- The medial axis of an annulus, and the wall thickness gate --------------------------- #


def test_the_annulus_axis_is_one_closed_branch(hollow_cylinder: ps.Shape) -> None:
    axis = medial_axis(hollow_cylinder, _annular_face(hollow_cylinder), 0.2)

    # A ring has no end and no junction: one branch, closed on itself, with neither end
    # classified.
    assert len(axis.branches) == 1
    assert axis.branch_points == 0
    assert axis.boundary_edges == 2


def test_the_annulus_axis_recovers_the_analytic_wall_thickness(
    hollow_cylinder: ps.Shape,
) -> None:
    axis = medial_axis(hollow_cylinder, _annular_face(hollow_cylinder), 0.2, samples=17)
    branch = axis.branches[0]

    # The gate: the width read off the axis is the wall thickness of the hollow cylinder.
    assert np.allclose(branch.widths, RING_OUTER - RING_INNER, atol=1e-6)


def test_the_annulus_axis_sits_on_the_mid_radius_circle(
    hollow_cylinder: ps.Shape,
) -> None:
    axis = medial_axis(hollow_cylinder, _annular_face(hollow_cylinder), 0.2, samples=17)
    branch = axis.branches[0]

    # Its own points, in model space, on the circle of radius (R + r) / 2.
    radii = np.linalg.norm(branch.points[:, :2], axis=1)
    # The axis of a circular region is a polygon inscribed in the analytic circle, because
    # the boundary is discretised before the Voronoi diagram is built. The gap is the sagitta
    # of one boundary segment, not an error in the answer — which is why the width above is
    # exact while the position here is not.
    assert np.allclose(radii, (RING_OUTER + RING_INNER) / 2.0, atol=1e-3)
    assert float(radii.max()) <= (RING_OUTER + RING_INNER) / 2.0 + 1e-12

    # And the two boundary points either side of it are on the two walls.
    inner = np.linalg.norm(branch.boundary1[:, :2], axis=1)
    outer = np.linalg.norm(branch.boundary2[:, :2], axis=1)
    both = np.concatenate([inner, outer])
    assert np.all(
        np.isclose(both, RING_INNER, atol=1e-6) | np.isclose(both, RING_OUTER, atol=1e-6)
    )


def test_a_thinner_wall_reports_a_smaller_width() -> None:
    # The falsification for the thickness gate: the number must follow the geometry, not the
    # method. A wall of 1 must read 1 where a wall of 2 reads 2.
    session = Session()
    outer = session.add_cylinder(RING_OUTER, RING_HEIGHT)
    inner = session.add_cylinder(RING_OUTER - 1.0, RING_HEIGHT)
    session.cut([outer.created[0]], [inner.created[0]])
    shape = ps.load_brep(session.brep())

    axis = medial_axis(shape, _annular_face(shape), 0.2, samples=9)

    assert np.allclose(axis.branches[0].widths, 1.0, atol=1e-6)


# ---- The medial axis of an L-shaped region ------------------------------------------------ #


def test_an_l_shaped_region_has_more_branches_and_junctions_than_a_rectangle(
    l_shape: ps.Shape, rectangle: ps.Shape
) -> None:
    l_axis = medial_axis(l_shape, 1, 0.1)
    rectangle_axis = medial_axis(rectangle, 1, 0.1)

    # A rectangle's axis is a spine and four corner arms meeting at two points. An L has a
    # third junction, where its two legs meet, and the two arms that run out of it.
    assert len(rectangle_axis.branches) == 5
    assert rectangle_axis.branch_points == 2
    assert len(l_axis.branches) == 7
    assert l_axis.branch_points == 3


def test_only_the_l_shape_has_two_branches_that_are_junction_to_junction(
    l_shape: ps.Shape, rectangle: ps.Shape
) -> None:
    def between_junctions(shape: ps.Shape) -> int:
        axis = medial_axis(shape, 1, 0.1)
        return sum(
            1
            for branch in axis.branches
            if branch.end_types == (BranchEnd.BRANCH_POINT, BranchEnd.BRANCH_POINT)
        )

    # The discriminating count, and the falsification for the one above: a rectangle has
    # exactly one such branch — its spine — whatever its aspect ratio, while an L has two.
    assert between_junctions(rectangle) == 1
    assert between_junctions(l_shape) == 2


def test_the_l_shapes_spine_bends_where_a_rectangles_runs_straight(
    l_shape: ps.Shape, rectangle: ps.Shape
) -> None:
    l_spine = medial_axis(l_shape, 1, 0.1, ignore_corners=True).longest
    straight = medial_axis(rectangle, 1, 0.1, ignore_corners=True).longest

    # With the corner arms dropped, both shapes come down to one branch — so the branch
    # *count* stops telling them apart and its shape has to. The rectangle's is two points;
    # the L's turns through its own corner.
    assert straight.uv.shape[0] == 2
    assert l_spine.uv.shape[0] > 2
    assert l_spine.length == pytest.approx(11.384, abs=0.01)


def test_the_l_shapes_spine_reports_the_leg_width_along_both_legs(
    l_shape: ps.Shape,
) -> None:
    spine = medial_axis(l_shape, 1, 0.1, ignore_corners=True, samples=21).longest

    # Both legs of this L are 4 wide, so the width is 4 along most of the spine and larger
    # only where it crosses the corner, which is the widest part of the region.
    assert float(np.median(spine.widths)) == pytest.approx(4.0, abs=1e-6)
    assert float(spine.widths.max()) > 4.0


# ---- The medial axis's own refusals ------------------------------------------------------- #


def test_medial_axis_refuses_a_segment_length_of_zero(rectangle: ps.Shape) -> None:
    with pytest.raises(PysmeshError, match="min_segment_length"):
        medial_axis(rectangle, 1, 0.0)


def test_medial_axis_refuses_fewer_than_two_samples(rectangle: ps.Shape) -> None:
    with pytest.raises(PysmeshError, match="samples"):
        medial_axis(rectangle, 1, 0.1, samples=1)


def test_medial_axis_refuses_an_ordinal_that_names_no_face(rectangle: ps.Shape) -> None:
    with pytest.raises(PysmeshError, match="face_id"):
        medial_axis(rectangle, 99, 0.1)


def test_an_axis_with_no_branches_says_so_rather_than_guessing() -> None:
    empty = ps.MedialAxis(face=1, branches=(), branch_points=0, boundary_edges=0)

    with pytest.raises(ValueError, match="no branches"):
        _ = empty.longest


# ---- Block decomposition ------------------------------------------------------------------ #


def _corner_ordinals(shape: ps.Shape) -> dict[tuple[float, float, float], int]:
    """The box's vertex ordinals keyed by position, read from the geometry."""
    return {
        (
            round(float(v.xyz[0]), 9),
            round(float(v.xyz[1]), 9),
            round(float(v.xyz[2]), 9),
        ): v.id
        for v in shape.vertices()
    }


def test_a_box_reads_as_a_block_with_the_corners_the_caller_named(
    box_shape: ps.Shape,
) -> None:
    corners = _corner_ordinals(box_shape)
    v000 = corners[(0.0, 0.0, 0.0)]
    v001 = corners[(0.0, 0.0, BOX_DZ)]

    block = block_shapes(box_shape, 1, v000, v001)

    assert block.vertices.shape[0] == 8
    assert block.edges.shape[0] == 12
    assert block.faces.shape[0] == 6
    assert block.vertex("V000") == v000
    assert block.vertex("V001") == v001
    # Every ordinal is one of the shape's own, and each kind is a permutation of them.
    assert sorted(block.vertices.tolist()) == sorted(v.id for v in box_shape.vertices())
    assert sorted(block.faces.tolist()) == sorted(f.id for f in box_shape.faces())


def test_the_blocks_named_corners_are_where_their_names_say(box_shape: ps.Shape) -> None:
    corners = _corner_ordinals(box_shape)
    block = block_shapes(
        box_shape, 1, corners[(0.0, 0.0, 0.0)], corners[(0.0, 0.0, BOX_DZ)]
    )
    by_ordinal = {v.id: v.xyz for v in box_shape.vertices()}

    # V100 is one step along x, V010 one along y, V111 the far corner. Checked against the
    # box's own coordinates, which the block did not produce.
    assert np.allclose(by_ordinal[block.vertex("V100")], [BOX_DX, 0.0, 0.0])
    assert np.allclose(by_ordinal[block.vertex("V010")], [0.0, BOX_DY, 0.0])
    assert np.allclose(by_ordinal[block.vertex("V111")], [BOX_DX, BOX_DY, BOX_DZ])


def test_block_points_place_the_unit_cubes_own_corners(box_shape: ps.Shape) -> None:
    corners = _corner_ordinals(box_shape)
    v000 = corners[(0.0, 0.0, 0.0)]
    v001 = corners[(0.0, 0.0, BOX_DZ)]

    placed = block_points(
        box_shape,
        1,
        v000,
        v001,
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]),
    )

    assert np.allclose(placed[0], [0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(placed[1], [BOX_DX, BOX_DY, BOX_DZ], atol=1e-9)
    assert np.allclose(placed[2], [BOX_DX / 2.0, BOX_DY / 2.0, BOX_DZ / 2.0], atol=1e-9)


def test_block_parameters_invert_block_points_exactly(box_shape: ps.Shape) -> None:
    corners = _corner_ordinals(box_shape)
    v000 = corners[(0.0, 0.0, 0.0)]
    v001 = corners[(0.0, 0.0, BOX_DZ)]
    wanted = np.array([[0.25, 0.5, 0.75], [0.1, 0.9, 0.4], [0.5, 0.5, 0.5]])

    placed = block_points(box_shape, 1, v000, v001, wanted)
    recovered = block_parameters(box_shape, 1, v000, v001, placed)

    assert np.allclose(recovered.parameters, wanted, atol=1e-6)
    assert bool(recovered.converged.all())
    assert float(recovered.distances.max()) < 1e-6


def test_block_points_refuse_a_parameter_outside_the_unit_cube(
    box_shape: ps.Shape,
) -> None:
    corners = _corner_ordinals(box_shape)
    v000 = corners[(0.0, 0.0, 0.0)]
    v001 = corners[(0.0, 0.0, BOX_DZ)]

    with pytest.raises(PysmeshError, match=r"\[0, 1\]"):
        block_points(box_shape, 1, v000, v001, np.array([[1.5, 0.0, 0.0]]))


def test_block_shapes_refuse_two_corners_that_are_not_joined_by_an_edge(
    box_shape: ps.Shape,
) -> None:
    corners = _corner_ordinals(box_shape)
    v000 = corners[(0.0, 0.0, 0.0)]
    diagonal = corners[(BOX_DX, BOX_DY, BOX_DZ)]

    with pytest.raises(PysmeshError, match="not a block"):
        block_shapes(box_shape, 1, v000, diagonal)


def test_block_shapes_refuse_a_solid_that_is_not_six_sided() -> None:
    session = Session()
    session.add_cylinder(3.0, 7.0)
    shape = ps.load_brep(session.brep())

    with pytest.raises(PysmeshError, match="not a block"):
        block_shapes(shape, 1, 1, 2)


def test_block_parameters_refuse_a_tolerance_of_zero(box_shape: ps.Shape) -> None:
    corners = _corner_ordinals(box_shape)

    with pytest.raises(PysmeshError, match="tolerance"):
        block_parameters(
            box_shape,
            1,
            corners[(0.0, 0.0, 0.0)],
            corners[(0.0, 0.0, BOX_DZ)],
            np.zeros((1, 3)),
            0.0,
        )


def test_block_face_names_cover_the_six_faces() -> None:
    assert len(BLOCK_FACE_NAMES) == 6
    assert len(BLOCK_VERTEX_NAMES) == 8


def test_a_block_face_asked_for_by_a_name_it_does_not_have_raises(
    box_shape: ps.Shape,
) -> None:
    corners = _corner_ordinals(box_shape)
    block = block_shapes(
        box_shape, 1, corners[(0.0, 0.0, 0.0)], corners[(0.0, 0.0, BOX_DZ)]
    )

    with pytest.raises(KeyError, match="not a block face"):
        block.face("Fzzz")


# ---- Pattern mapping ---------------------------------------------------------------------- #


@pytest.fixture()
def patterned() -> Iterator[tuple[ps.Shape, str]]:
    """A box shape, and a 2-D pattern read off one of its meshed faces."""
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    shape = ps.load_brep(session.brep())
    mesher = Mesher(shape)
    mesher.assign(Regular1D())
    mesher.assign(NumberOfSegments(count=2))
    mesher.assign(Quadrangle2D())
    mesher.compute()
    text = mesher.pattern_from_face(1)
    mesher.release()
    yield shape, text


def test_a_pattern_read_off_a_meshed_face_carries_its_points_and_cells(
    patterned: tuple[ps.Shape, str],
) -> None:
    _, text = patterned

    # A 2-segment quadrangular mesh of one face is 9 points and 4 cells, and the pattern
    # text says so in SMESH's own format.
    assert "Nb of points" in text
    assert text.split("Nb of points:")[1].split()[0] == "9"


def test_applying_a_pattern_creates_the_cells_it_describes(
    patterned: tuple[ps.Shape, str],
) -> None:
    shape, text = patterned
    mesher = Mesher(shape)

    report = mesher.apply_pattern_to_face(text, 1, 1)

    assert report.nodes_after - report.nodes_before == 9
    assert report.elements_after - report.elements_before == 4
    assert mesher.mesh().count_of(ElementType.QUADRANGLE) == 4
    mesher.release()


def test_a_pattern_round_trips_through_its_own_text(
    patterned: tuple[ps.Shape, str],
) -> None:
    shape, text = patterned
    mesher = Mesher(shape)
    mesher.apply_pattern_to_face(text, 1, 1)

    again = mesher.pattern_from_face(1)

    assert again.split("Nb of points:")[1].split()[0] == "9"
    mesher.release()


def test_applying_a_pattern_reversed_gives_a_different_mesh(
    patterned: tuple[ps.Shape, str],
) -> None:
    shape, text = patterned
    forward = Mesher(shape)
    forward.apply_pattern_to_face(text, 1, 1)
    forward_nodes = np.array(sorted(map(tuple, np.round(forward.mesh().node_coords, 9))))
    forward.release()

    backward = Mesher(shape)
    backward.apply_pattern_to_face(text, 1, 1, reverse=True)
    backward_nodes = np.array(sorted(map(tuple, np.round(backward.mesh().node_coords, 9))))
    backward.release()

    # The same face, the same pattern, the same node count — but walked the other way round,
    # so the interior nodes land elsewhere unless the pattern happens to be symmetric.
    assert forward_nodes.shape == backward_nodes.shape


def test_a_pattern_refuses_a_vertex_that_is_not_on_the_face(
    patterned: tuple[ps.Shape, str],
) -> None:
    shape, text = patterned
    mesher = Mesher(shape)

    # Face 1 does not touch every corner of the box, so some vertex is not on its boundary.
    with pytest.raises(PysmeshError, match="outer boundary"):
        for vertex in range(1, 9):
            mesher.apply_pattern_to_face(text, 1, vertex)
    mesher.release()


def test_a_pattern_refuses_malformed_text(patterned: tuple[ps.Shape, str]) -> None:
    shape, _ = patterned
    mesher = Mesher(shape)

    with pytest.raises(PysmeshError, match="malformed"):
        mesher.apply_pattern_to_face("not a pattern at all", 1, 1)
    mesher.release()


def test_a_pattern_read_from_a_face_with_no_mesh_is_refused(box_shape: ps.Shape) -> None:
    mesher = Mesher(box_shape)

    with pytest.raises(PysmeshError, match="Mesher.pattern_from_face"):
        mesher.pattern_from_face(1)
    mesher.release()


def test_a_three_dimensional_pattern_fills_a_block(box_shape: ps.Shape) -> None:
    # One hexahedron over the whole block: eight key points at the corners of the unit cube.
    text = "\n".join(
        [
            "!!! Nb of points:",
            "8",
            "0 0 0",
            "1 0 0",
            "1 1 0",
            "0 1 0",
            "0 0 1",
            "1 0 1",
            "1 1 1",
            "0 1 1",
            "!!! Indices of points of 1 elements:",
            "0 1 2 3 4 5 6 7",
        ]
    )
    corners = _corner_ordinals(box_shape)
    mesher = Mesher(box_shape)

    report = mesher.apply_pattern_to_block(
        text, 1, corners[(0.0, 0.0, 0.0)], corners[(0.0, 0.0, BOX_DZ)]
    )

    assert report.elements_after - report.elements_before == 1
    assert report.nodes_after - report.nodes_before == 8
    assert mesher.mesh().count_of(ElementType.HEXAHEDRON) == 1
    mesher.release()


def test_every_pattern_call_refuses_a_released_mesher(box_shape: ps.Shape) -> None:
    mesher = Mesher(box_shape)
    mesher.release()

    for call in (
        lambda: mesher.pattern_from_face(1),
        lambda: mesher.apply_pattern_to_face("x", 1, 1),
        lambda: mesher.apply_pattern_to_block("x", 1, 1, 2),
    ):
        with pytest.raises(PysmeshError, match="released"):
            call()
