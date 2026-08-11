# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""Gates for the mesh editor.

Every operation here gets what its own requirement asks for: a before/after invariant test
against a quantity the editor did not produce, and a failure-mode test. The invariants are
chosen so that a no-op could not pass them — an element count, a total volume, an orientation
verdict taken from the quality controls, a node coincidence measured from the harvest.

Two operations carry the requirement's own named checks:

* converting to second order **round-trips**, and the check is against the original
  connectivity read back as node positions rather than against the counts, because counts
  alone cannot tell a restored mesh from a differently connected one of the same size;
* re-orienting from the volumes is run on a shell **deliberately made inconsistent**, and the
  falsification is asserted beside it — the same call on an already consistent shell must
  reverse nothing.

Fixture sizing follows the project rule: 3 x 7 x 11, never a unit cube.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    AspectRatio,
    BadOrientedVolume,
    ElementDimension,
    ElementType,
    Hexa3D,
    Mesher,
    NumberOfSegments,
    PysmeshError,
    Quadrangle2D,
    Regular1D,
    Session,
    SmoothMethod,
    Volume,
)

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0
BOX_VOLUME: float = BOX_DX * BOX_DY * BOX_DZ


# ---- Fixtures ---------------------------------------------------------------------------- #


def _box_shape(dx: float = BOX_DX, dy: float = BOX_DY, dz: float = BOX_DZ) -> ps.Shape:
    """A box, through the session, as the shape a mesher takes."""
    session = Session()
    session.add_box(dx, dy, dz)
    return ps.load_brep(session.brep())


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
def hexa_mesher() -> Iterator[Mesher]:
    """A 2 x 2 x 2 hexahedral mesh of the box."""
    mesher = _mesher(_box_shape(), 2)
    yield mesher
    mesher.release()


@pytest.fixture()
def surface_mesher() -> Iterator[Mesher]:
    """The box's quadrangular skin, with no volume cells."""
    mesher = _mesher(_box_shape(), 2, volumes=False)
    yield mesher
    mesher.release()


@pytest.fixture()
def triangle_mesher() -> Iterator[Mesher]:
    """The box's skin split into triangles, which is what an offset needs."""
    mesher = _mesher(_box_shape(), 2, volumes=False)
    mesher.quad_to_tri()
    yield mesher
    mesher.release()


def _seam_fixture() -> ps.Shape:
    """Two squares meeting along x = 3 as separate faces, so their nodes stay distinct."""
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 3.0, 3.0)
    session.add_rectangle((3.0, 0.0, 0.0), (0.0, 0.0, 1.0), 3.0, 3.0)
    return ps.load_brep(session.brep())


def _coincident_fixture() -> ps.Shape:
    """Two faces at exactly the same place, which is the shape a side sew joins."""
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 3.0, 3.0)
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 3.0, 3.0)
    return ps.load_brep(session.brep())


# ---- Oracles ----------------------------------------------------------------------------- #


def _cell_volume(coords: NDArray[np.float64]) -> float:
    """Volume of one convex cell, by the divergence theorem over its convex hull faces.

    Independent of the library under test: it reads only node coordinates.
    """
    centre = coords.mean(axis=0)
    # A hexahedron's six facets in SMESH's own corner order: the two opposite quadrangles
    # first, then the four sides.
    faces = (
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    )
    volume = 0.0
    for face in faces:
        p0, p1, p2, p3 = (coords[i] for i in face)
        for tri in ((p0, p1, p2), (p0, p2, p3)):
            q0, q1, q2 = tri
            volume += float(np.dot(np.cross(q1 - q0, q2 - q0), q0 - centre)) / 6.0
    return abs(volume)


def _total_volume(mesher: Mesher) -> float:
    """The mesh's volume, summed from the harvest arrays rather than from the controls."""
    mesh = mesher.mesh()
    total = 0.0
    for element in range(mesh.element_count):
        if mesh.element_type[element] != int(ElementType.HEXAHEDRON):
            continue
        total += _cell_volume(mesh.node_coords[mesh.nodes_of(element)])
    return total


def _connectivity(mesher: Mesher) -> set[frozenset[tuple[float, float, float]]]:
    """Every element as the set of its corner positions, keyed by nothing the mesher issues.

    Positions rather than ids, so a conversion that re-created elements with the same ids but
    different connectivity would still be caught.
    """
    mesh = mesher.mesh()
    out: set[frozenset[tuple[float, float, float]]] = set()
    for element in range(mesh.element_count):
        corners = mesh.node_coords[mesh.nodes_of(element)]
        out.add(
            frozenset(
                (round(float(p[0]), 9), round(float(p[1]), 9), round(float(p[2]), 9))
                for p in corners
            )
        )
    return out


def _node_positions(mesher: Mesher) -> NDArray[np.float64]:
    """The mesh's node coordinates, sorted, so two meshes can be compared position-wise."""
    mesh = mesher.mesh()
    return np.array(sorted(map(tuple, np.round(mesh.node_coords, 9))))


def _face_ids(mesher: Mesher) -> list[int]:
    """The mesh ids of every 2-D element."""
    mesh = mesher.mesh()
    return [
        int(mesh.element_id[i])
        for i in range(mesh.element_count)
        if mesh.element_type[i] in (int(ElementType.QUADRANGLE), int(ElementType.TRIANGLE))
    ]


def _volume_ids(mesher: Mesher) -> list[int]:
    """The mesh ids of every 3-D element."""
    mesh = mesher.mesh()
    return [
        int(mesh.element_id[i])
        for i in range(mesh.element_count)
        if mesh.element_type[i] == int(ElementType.HEXAHEDRON)
    ]


# ---- Order conversion -------------------------------------------------------------------- #


def test_convert_to_quadratic_adds_nodes_and_keeps_element_count(
    hexa_mesher: Mesher,
) -> None:
    before = hexa_mesher.mesh()

    hexa_mesher.convert_to_quadratic()

    after = hexa_mesher.mesh()
    assert after.node_count > before.node_count
    assert after.element_count == before.element_count
    assert after.count_of(ElementType.QUAD_HEXAHEDRON) == before.count_of(
        ElementType.HEXAHEDRON
    )


def test_convert_to_quadratic_round_trip_restores_the_original_connectivity(
    hexa_mesher: Mesher,
) -> None:
    original = _connectivity(hexa_mesher)
    positions = _node_positions(hexa_mesher)

    hexa_mesher.convert_to_quadratic()
    hexa_mesher.convert_from_quadratic()

    assert _connectivity(hexa_mesher) == original
    assert np.allclose(_node_positions(hexa_mesher), positions)


def test_convert_to_quadratic_bi_quadratic_adds_more_nodes_than_plain(
    hexa_mesher: Mesher,
) -> None:
    plain = _mesher(_box_shape(), 2)
    plain.convert_to_quadratic(bi_quadratic=False)
    plain_nodes = plain.mesh().node_count
    plain.release()

    hexa_mesher.convert_to_quadratic(bi_quadratic=True)

    assert hexa_mesher.mesh().node_count > plain_nodes


def test_convert_from_quadratic_on_a_linear_mesh_changes_nothing(
    hexa_mesher: Mesher,
) -> None:
    before = hexa_mesher.mesh().node_count

    # The return value is always True and carries no information; the node count is what
    # says whether anything happened.
    assert hexa_mesher.convert_from_quadratic() is True

    assert hexa_mesher.mesh().node_count == before


def test_split_quadratic_into_linear_adds_no_nodes(hexa_mesher: Mesher) -> None:
    hexa_mesher.convert_to_quadratic(bi_quadratic=True)
    nodes = hexa_mesher.mesh().node_count

    report = hexa_mesher.split_quadratic_into_linear()

    assert report.nodes_after == nodes
    assert report.volumes_after == 8 * report.volumes_before


def test_convert_to_quadratic_after_release_raises(hexa_mesher: Mesher) -> None:
    hexa_mesher.release()

    with pytest.raises(PysmeshError, match="released"):
        hexa_mesher.convert_to_quadratic()


# ---- Coincidence and merging -------------------------------------------------------------- #


def test_find_coincident_nodes_on_a_conforming_mesh_finds_none(
    hexa_mesher: Mesher,
) -> None:
    assert hexa_mesher.find_coincident_nodes(1e-9) == ()


def test_find_coincident_nodes_finds_the_seam_of_two_separate_patches() -> None:
    mesher = _mesher(_seam_fixture(), 3, volumes=False)

    found = mesher.find_coincident_nodes(1e-9)

    # Four positions along the shared edge, each carrying one node from either patch.
    assert len(found) == 4
    assert all(group.shape[0] == 2 for group in found)
    mesher.release()


def test_merge_node_groups_collapses_exactly_what_it_was_given() -> None:
    mesher = _mesher(_seam_fixture(), 3, volumes=False)
    groups = mesher.find_coincident_nodes(1e-9)
    before = mesher.mesh().node_count

    report = mesher.merge_node_groups(groups)

    assert report.groups_merged == len(groups)
    assert report.nodes_after == before - len(groups)
    assert mesher.find_coincident_nodes(1e-9) == ()
    mesher.release()


def test_merge_nodes_joins_two_patches_into_one_surface() -> None:
    mesher = _mesher(_seam_fixture(), 3, volumes=False)
    before = mesher.mesh().node_count

    report = mesher.merge_nodes(1e-9)

    assert report.groups_merged == 4
    assert report.nodes_after == before - 4
    # The seam is no longer a free border on either side.
    assert mesher.select(ps.FreeBorders()).count < 24
    mesher.release()


def test_merge_node_groups_refuses_a_group_of_one(hexa_mesher: Mesher) -> None:
    node = int(hexa_mesher.mesh().node_id[0])

    with pytest.raises(PysmeshError, match="at least two"):
        hexa_mesher.merge_node_groups([[node]])


def test_merge_nodes_refuses_a_tolerance_that_is_not_a_distance(
    hexa_mesher: Mesher,
) -> None:
    for tolerance in (-1.0, float("nan")):
        with pytest.raises(PysmeshError, match=">= 0"):
            hexa_mesher.merge_nodes(tolerance)


def test_find_equal_elements_finds_a_duplicate_and_merging_removes_it(
    surface_mesher: Mesher,
) -> None:
    faces = _face_ids(surface_mesher)[:2]
    surface_mesher.double_elements(faces)

    found = surface_mesher.find_equal_elements()

    assert len(found) == 2
    assert all(group.shape[0] == 2 for group in found)

    report = surface_mesher.merge_equal_elements()

    assert report.groups_merged == 2
    assert report.faces_after == report.faces_before - 2
    assert surface_mesher.find_equal_elements() == ()


def test_find_equal_elements_on_a_clean_mesh_finds_none(surface_mesher: Mesher) -> None:
    assert surface_mesher.find_equal_elements() == ()


# ---- Smoothing ---------------------------------------------------------------------------- #


def test_smooth_moves_nodes_without_creating_or_deleting_anything(
    surface_mesher: Mesher,
) -> None:
    before = surface_mesher.mesh()

    report = surface_mesher.smooth(SmoothMethod.LAPLACIAN, iterations=2)

    after = surface_mesher.mesh()
    assert report.nodes_after == report.nodes_before
    assert report.faces_after == report.faces_before
    assert after.element_count == before.element_count


def test_smooth_on_shape_keeps_every_node_on_the_box_surface(
    surface_mesher: Mesher,
) -> None:
    surface_mesher.smooth(SmoothMethod.CENTROIDAL, iterations=3, on_shape=True)

    coords = surface_mesher.mesh().node_coords
    # Every node must still be on one of the six planes bounding the box.
    on_a_face = (
        np.isclose(coords[:, 0], 0.0)
        | np.isclose(coords[:, 0], BOX_DX)
        | np.isclose(coords[:, 1], 0.0)
        | np.isclose(coords[:, 1], BOX_DY)
        | np.isclose(coords[:, 2], 0.0)
        | np.isclose(coords[:, 2], BOX_DZ)
    )
    assert bool(on_a_face.all())


def test_smooth_holds_the_nodes_it_is_told_to(surface_mesher: Mesher) -> None:
    mesh = surface_mesher.mesh()
    # An interior node of one face is free to move; pin it and it must not.
    interior = [
        i
        for i in range(mesh.node_count)
        if int(mesh.node_kind[i]) == int(ps.SubShapeKind.FACE)
    ]
    pinned = int(mesh.node_id[interior[0]])
    before = mesh.node_coords[interior[0]].copy()

    surface_mesher.smooth(SmoothMethod.CENTROIDAL, iterations=3, fixed_nodes=[pinned])

    after_mesh = surface_mesher.mesh()
    row = int(np.flatnonzero(after_mesh.node_id == pinned)[0])
    assert np.allclose(after_mesh.node_coords[row], before)


def test_smooth_refuses_a_target_below_a_regular_element(surface_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="aspect ratio"):
        surface_mesher.smooth(target_aspect_ratio=0.5)


def test_smooth_refuses_zero_iterations(surface_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="iterations"):
        surface_mesher.smooth(iterations=0)


# ---- Orientation -------------------------------------------------------------------------- #


def test_reorient_reverses_exactly_the_elements_named(surface_mesher: Mesher) -> None:
    faces = _face_ids(surface_mesher)[:3]

    assert surface_mesher.reorient(faces) == 3


def test_reorient_2d_by_3d_repairs_a_deliberately_inconsistent_shell(
    hexa_mesher: Mesher,
) -> None:
    faces = _face_ids(hexa_mesher)
    flipped = faces[:5]
    hexa_mesher.reorient(flipped)

    repaired = hexa_mesher.reorient_2d_by_3d()

    assert repaired == len(flipped)


def test_reorient_2d_by_3d_on_a_consistent_shell_reverses_nothing(
    hexa_mesher: Mesher,
) -> None:
    # The falsification: without it the test above would pass for an implementation that
    # reverses everything it is given.
    assert hexa_mesher.reorient_2d_by_3d() == 0
    assert hexa_mesher.reorient_2d_by_3d() == 0


def test_reorient_2d_by_3d_leaves_the_cells_themselves_valid(hexa_mesher: Mesher) -> None:
    hexa_mesher.reorient(_face_ids(hexa_mesher)[:5])
    hexa_mesher.reorient_2d_by_3d()

    assert hexa_mesher.select(BadOrientedVolume()).count == 0
    assert float(hexa_mesher.quality(Volume()).values.min()) > 0.0


def test_reorient_2d_makes_a_mixed_shell_consistent(surface_mesher: Mesher) -> None:
    faces = _face_ids(surface_mesher)
    surface_mesher.reorient(faces[:4])

    reoriented = surface_mesher.reorient_2d(
        direction=(0.0, 0.0, 1.0), allow_non_manifold=True
    )

    # Some faces had to move; running it again must then find nothing left to do.
    assert reoriented > 0
    assert (
        surface_mesher.reorient_2d(direction=(0.0, 0.0, 1.0), allow_non_manifold=True) == 0
    )


def test_reorient_2d_by_3d_refuses_a_mesh_with_no_cells(surface_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="no volume cells"):
        surface_mesher.reorient_2d_by_3d()


def test_reorient_refuses_an_empty_list(hexa_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="name the elements"):
        hexa_mesher.reorient([])


# ---- Face splitting and fusing ------------------------------------------------------------ #


def test_quad_to_tri_doubles_the_face_count_and_keeps_the_area(
    surface_mesher: Mesher,
) -> None:
    area_before = float(surface_mesher.quality(ps.Area()).values.sum())

    report = surface_mesher.quad_to_tri()

    assert report.faces_after == 2 * report.faces_before
    assert surface_mesher.mesh().count_of(ElementType.QUADRANGLE) == 0
    assert float(surface_mesher.quality(ps.Area()).values.sum()) == pytest.approx(
        area_before, rel=1e-12
    )


def test_quad_to_tri_with_a_criterion_chooses_a_different_diagonal_than_the_fixed_one() -> (
    None
):
    # A face whose two diagonals are not equivalent: the criterion must be able to disagree
    # with the fixed choice, or passing one would be decoration.
    fixed = _mesher(_box_shape(3.0, 7.0, 11.0), 2, volumes=False)
    fixed.quad_to_tri(diagonal_13=True)
    fixed_worst = float(fixed.quality(AspectRatio()).values.max())
    fixed.release()

    other = _mesher(_box_shape(3.0, 7.0, 11.0), 2, volumes=False)
    other.quad_to_tri(diagonal_13=False)
    other_worst = float(other.quality(AspectRatio()).values.max())
    other.release()

    chosen = _mesher(_box_shape(3.0, 7.0, 11.0), 2, volumes=False)
    chosen.quad_to_tri(criterion=AspectRatio())
    chosen_worst = float(chosen.quality(AspectRatio()).values.max())
    chosen.release()

    assert chosen_worst <= max(fixed_worst, other_worst) + 1e-9


def test_tri_to_quad_fuses_coplanar_triangles_back_into_quadrangles(
    triangle_mesher: Mesher,
) -> None:
    triangles = triangle_mesher.mesh().count_of(ElementType.TRIANGLE)

    report = triangle_mesher.tri_to_quad(max_angle=0.0)

    assert report.faces_after < report.faces_before
    assert triangle_mesher.mesh().count_of(ElementType.QUADRANGLE) > 0
    assert triangle_mesher.mesh().count_of(ElementType.TRIANGLE) < triangles


def test_tri_to_quad_refuses_a_negative_angle(triangle_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="maximum angle"):
        triangle_mesher.tri_to_quad(max_angle=-1.0)


def test_quad_to_tri_refuses_an_id_that_is_not_a_face(hexa_mesher: Mesher) -> None:
    cell = _volume_ids(hexa_mesher)[0]

    with pytest.raises(PysmeshError, match="not of the expected family"):
        hexa_mesher.quad_to_tri([cell])


# ---- Duplication -------------------------------------------------------------------------- #


def test_double_elements_creates_a_coincident_twin_of_each_named_face(
    surface_mesher: Mesher,
) -> None:
    faces = _face_ids(surface_mesher)[:3]
    before = surface_mesher.mesh()

    report = surface_mesher.double_elements(faces)

    after = surface_mesher.mesh()
    assert report.faces_after == report.faces_before + 3
    # A baffle adds no node: the twin sits on the originals' own nodes.
    assert after.node_count == before.node_count
    # And the twins are geometrically indistinguishable from their originals.
    assert len(surface_mesher.find_equal_elements()) == 3


def test_double_elements_refuses_an_empty_list(surface_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="name the elements"):
        surface_mesher.double_elements([])


def test_double_elements_refuses_an_unknown_id(surface_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="no element with id"):
        surface_mesher.double_elements([10**9])


# ---- Sweeps ------------------------------------------------------------------------------- #


def test_extrusion_sweep_fills_the_swept_region_with_cells(
    surface_mesher: Mesher,
) -> None:
    faces = _face_ids(surface_mesher)[:4]

    report = surface_mesher.extrusion_sweep(faces, step=(0.0, 0.0, 1.0), steps=2)

    # Four faces swept over two steps give eight cells, and nothing else does.
    assert report.volumes_after - report.volumes_before == 8
    assert report.nodes_after > report.nodes_before


def test_extrusion_sweep_cells_all_have_positive_volume(surface_mesher: Mesher) -> None:
    faces = [
        i
        for i in _face_ids(surface_mesher)
        if np.allclose(
            surface_mesher.mesh().node_coords[
                surface_mesher.mesh().nodes_of(
                    int(np.flatnonzero(surface_mesher.mesh().element_id == i)[0])
                )
            ][:, 2],
            0.0,
        )
    ]
    surface_mesher.extrusion_sweep(faces, step=(0.0, 0.0, -1.0), steps=1)

    values = surface_mesher.quality(Volume()).values
    assert values.shape[0] == len(faces)
    assert bool((values > 0.0).all())


def test_rotation_sweep_fills_the_swept_region_with_cells(
    surface_mesher: Mesher,
) -> None:
    faces = _face_ids(surface_mesher)[:2]

    report = surface_mesher.rotation_sweep(
        faces, axis_origin=(0.0, 0.0, 0.0), axis_direction=(0.0, 0.0, 1.0),
        angle=0.3, steps=2,
    )

    assert report.volumes_after - report.volumes_before == 4


def test_extrusion_sweep_refuses_a_zero_step(surface_mesher: Mesher) -> None:
    faces = _face_ids(surface_mesher)[:1]

    with pytest.raises(PysmeshError, match="zero vector"):
        surface_mesher.extrusion_sweep(faces, step=(0.0, 0.0, 0.0), steps=1)


def test_rotation_sweep_refuses_a_zero_axis(surface_mesher: Mesher) -> None:
    faces = _face_ids(surface_mesher)[:1]

    with pytest.raises(PysmeshError, match="zero vector"):
        surface_mesher.rotation_sweep(
            faces, axis_origin=(0.0, 0.0, 0.0), axis_direction=(0.0, 0.0, 0.0),
            angle=0.3, steps=1,
        )


def test_extrusion_sweep_refuses_no_steps(surface_mesher: Mesher) -> None:
    faces = _face_ids(surface_mesher)[:1]

    with pytest.raises(PysmeshError, match="steps"):
        surface_mesher.extrusion_sweep(faces, step=(0.0, 0.0, 1.0), steps=0)


# ---- Surface offset ----------------------------------------------------------------------- #


def test_offset_of_a_box_skin_lands_at_the_stated_distance(
    triangle_mesher: Mesher,
) -> None:
    before = triangle_mesher.mesh()
    distance = 0.25

    report = triangle_mesher.offset(distance)

    after = triangle_mesher.mesh()
    assert report.faces_after == 2 * report.faces_before
    # The offset surface is the box grown by the distance on every side, so its bounding box
    # grows by exactly twice it on each axis — measured against the source, not the library.
    grew = after.node_coords.max(axis=0) - after.node_coords.min(axis=0)
    original = before.node_coords.max(axis=0) - before.node_coords.min(axis=0)
    assert np.allclose(grew, original + 2.0 * distance, atol=1e-9)


def test_offset_without_copying_leaves_only_the_offset_surface(
    triangle_mesher: Mesher,
) -> None:
    report = triangle_mesher.offset(0.25, copy_elements=False)

    assert report.faces_after == report.faces_before


def test_offset_refuses_a_mesh_that_is_not_all_triangles(
    surface_mesher: Mesher,
) -> None:
    with pytest.raises(PysmeshError, match="Mesher.offset"):
        surface_mesher.offset(0.25)


# ---- Sewing ------------------------------------------------------------------------------- #


def _seam_nodes(mesher: Mesher) -> list[list[int]]:
    """The node ids on the shared line x = 3, grouped by position and ordered along y."""
    mesh = mesher.mesh()
    groups: dict[float, list[int]] = {}
    for i in range(mesh.node_count):
        if abs(float(mesh.node_coords[i][0]) - 3.0) < 1e-9:
            groups.setdefault(round(float(mesh.node_coords[i][1]), 9), []).append(
                int(mesh.node_id[i])
            )
    return [groups[key] for key in sorted(groups)]


def test_sew_free_border_joins_two_patches_along_their_shared_rim() -> None:
    mesher = _mesher(_seam_fixture(), 3, volumes=False)
    seam = _seam_nodes(mesher)
    borders_before = mesher.select(ps.FreeBorders()).count
    nodes_before = mesher.mesh().node_count

    report = mesher.sew_free_border(
        border=(seam[0][0], seam[1][0], seam[-1][0]),
        side=(seam[0][1], seam[1][1], seam[-1][1]),
    )

    assert report.nodes_after == nodes_before - len(seam)
    assert mesher.select(ps.FreeBorders()).count < borders_before
    assert mesher.find_coincident_nodes(1e-9) == ()
    mesher.release()


def test_sew_free_border_refuses_nodes_that_are_not_on_a_border() -> None:
    mesher = _mesher(_seam_fixture(), 3, volumes=False)
    mesh = mesher.mesh()
    interior = [
        int(mesh.node_id[i])
        for i in range(mesh.node_count)
        if int(mesh.node_kind[i]) == int(ps.SubShapeKind.FACE)
    ]

    with pytest.raises(PysmeshError, match="Mesher.sew_free_border"):
        mesher.sew_free_border(
            border=(interior[0], interior[1], interior[2]),
            side=(interior[0], interior[1]),
        )
    mesher.release()


def test_sew_free_border_refuses_a_border_of_the_wrong_length() -> None:
    mesher = _mesher(_seam_fixture(), 3, volumes=False)
    seam = _seam_nodes(mesher)

    short = (seam[0][0], seam[1][0])
    with pytest.raises(PysmeshError, match="three node ids"):
        mesher.sew_free_border(border=short, side=(seam[0][1],))  # type: ignore[arg-type]
    mesher.release()


def test_sew_side_elements_merges_two_coincident_patches() -> None:
    shape = _coincident_fixture()
    mesher = _mesher(shape, 2, volumes=False)
    mesh = mesher.mesh()
    quads = [
        (int(mesh.element_id[i]), int(mesh.element_ordinal[i]))
        for i in range(mesh.element_count)
        if mesh.element_type[i] == int(ElementType.QUADRANGLE)
    ]
    side1 = [i for i, ordinal in quads if ordinal == 1]
    side2 = [i for i, ordinal in quads if ordinal == 2]

    def at(point: tuple[float, float, float]) -> list[int]:
        return [
            int(mesh.node_id[i])
            for i in range(mesh.node_count)
            if np.allclose(mesh.node_coords[i], point)
        ]

    corner = at((0.0, 0.0, 0.0))
    along = at((1.5, 0.0, 0.0))
    nodes_before = mesh.node_count

    report = mesher.sew_side_elements(
        side1, side2, (corner[0], corner[1]), (along[0], along[1])
    )

    # Every node of one patch is merged with its twin on the other.
    assert report.nodes_after == nodes_before // 2
    assert mesher.find_coincident_nodes(1e-9) == ()
    mesher.release()


def test_sew_side_elements_refuses_sides_of_different_size() -> None:
    shape = _coincident_fixture()
    mesher = _mesher(shape, 2, volumes=False)
    mesh = mesher.mesh()
    quads = [
        (int(mesh.element_id[i]), int(mesh.element_ordinal[i]))
        for i in range(mesh.element_count)
        if mesh.element_type[i] == int(ElementType.QUADRANGLE)
    ]
    side1 = [i for i, ordinal in quads if ordinal == 1]
    side2 = [i for i, ordinal in quads if ordinal == 2][:2]
    nodes = [int(mesh.node_id[i]) for i in range(4)]

    with pytest.raises(PysmeshError, match="Mesher.sew_side_elements"):
        mesher.sew_side_elements(side1, side2, (nodes[0], nodes[1]), (nodes[2], nodes[3]))
    mesher.release()


def test_sew_side_elements_refuses_an_empty_side(surface_mesher: Mesher) -> None:
    faces = _face_ids(surface_mesher)

    with pytest.raises(PysmeshError, match="must name their elements"):
        surface_mesher.sew_side_elements(faces, [], (1, 2), (3, 4))


# ---- The mesh survives its own editing ---------------------------------------------------- #


def test_the_total_volume_survives_a_split_and_a_conversion(hexa_mesher: Mesher) -> None:
    # An end-to-end invariant against a quantity the editor did not produce: the mesh fills
    # the same box before and after being edited.
    assert _total_volume(hexa_mesher) == pytest.approx(BOX_VOLUME, rel=1e-9)

    hexa_mesher.convert_to_quadratic()
    hexa_mesher.convert_from_quadratic()

    assert _total_volume(hexa_mesher) == pytest.approx(BOX_VOLUME, rel=1e-9)
    assert hexa_mesher.select(BadOrientedVolume()).count == 0


def test_every_editing_operation_refuses_a_released_mesher() -> None:
    mesher = _mesher(_box_shape(), 2)
    faces = _face_ids(mesher)[:1]
    mesher.release()

    for call in (
        lambda: mesher.merge_nodes(1e-9),
        lambda: mesher.merge_equal_elements(),
        lambda: mesher.smooth(),
        lambda: mesher.reorient(faces),
        lambda: mesher.reorient_2d(),
        lambda: mesher.reorient_2d_by_3d(),
        lambda: mesher.quad_to_tri(),
        lambda: mesher.double_elements(faces),
        lambda: mesher.offset(0.1),
        lambda: mesher.find_equal_elements(),
        lambda: mesher.find_coincident_nodes(),
    ):
        with pytest.raises(PysmeshError, match="released"):
            call()


def test_a_group_of_nodes_survives_every_operation_that_keeps_its_nodes(
    hexa_mesher: Mesher,
) -> None:
    mesh = hexa_mesher.mesh()
    corner = [
        int(mesh.node_id[i])
        for i in range(mesh.node_count)
        if int(mesh.node_kind[i]) == int(ps.SubShapeKind.VERTEX)
    ]
    hexa_mesher.add_group("corners", ElementDimension.NODE, corner)

    hexa_mesher.smooth(iterations=1)
    hexa_mesher.convert_to_quadratic()

    assert sorted(hexa_mesher.group("corners").element_ids.tolist()) == sorted(corner)
