# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-13

"""Gates for the shape-free mesher: filling from arrays, and deleting from a mesh.

Two capabilities are under test, and both are about a mesh that has no CAD behind it.

**The fill.** A mesher built with no shape must be a real mesher, not a container: the
assertions here are that the editor, the search surface, the quality controls and the groups
all work on one, and that the operations which resolve a sub-shape ordinal refuse it by name
rather than failing somewhere further in.

**The deletion.** A removal is checked against what it took, never against a count alone —
the ids reported must be exactly the entities that stopped existing, including the ones
nobody named: the elements a removed node carried, and the nodes the free-node sweep found.
The patch-group test is the requirement in its own words: a patch stored as a group must
still be that patch after part of it is deleted.

Geometry is a folded strip rather than a plane, because a flat sheet has no sharp edge and
the partition it produces could not tell a working `sharp_edges` from one that returns
nothing. Every fixture is asymmetric (3 x 7 x 11 scaled), so a transposition or an axis mix-up
cannot pass.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    Area,
    ElementDimension,
    ElementType,
    Mesher,
    NumberOfSegments,
    PysmeshError,
    Quadrangle2D,
    Regular1D,
    Session,
    SubShape,
    SubShapeKind,
)

STRIP_DX: float = 3.0
STRIP_DY: float = 7.0
STRIP_DZ: float = 11.0


# ---- Fixtures ---------------------------------------------------------------------------- #


def _folded_strip() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Two rectangles meeting at a right angle along y: 6 nodes, 4 triangles, 1 sharp edge.

    The fold is a 90-degree dihedral, so a sharp-edge search at any angle below 90 finds
    exactly the one edge along it and the partition falls into exactly two patches.
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [STRIP_DX, 0.0, 0.0],
            [0.0, STRIP_DY, 0.0],
            [STRIP_DX, STRIP_DY, 0.0],
            [STRIP_DX, 0.0, STRIP_DZ],
            [STRIP_DX, STRIP_DY, STRIP_DZ],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=np.int64)
    return points, triangles


@pytest.fixture
def strip() -> Mesher:
    """A shape-free mesher holding the folded strip."""
    points, triangles = _folded_strip()
    return Mesher.from_arrays(points, triangles)


def _box_mesher() -> Mesher:
    """A shape-backed mesher with a computed quadrangle mesh, for the shared-surface tests."""
    session = Session()
    session.add_box(STRIP_DX, STRIP_DY, STRIP_DZ)
    mesher = Mesher(ps.load_brep(session.brep()))
    mesher.assign(Regular1D())
    mesher.assign(NumberOfSegments(count=2))
    mesher.assign(Quadrangle2D())
    mesher.compute()
    return mesher


# ---- Building one without a shape --------------------------------------------------------- #


def test_mesher_without_a_shape_reports_has_shape_false() -> None:
    mesher = Mesher()

    assert mesher.has_shape is False
    assert mesher.is_open() is True


def test_mesher_on_a_shape_reports_has_shape_true(box_brep: bytes) -> None:
    mesher = Mesher(ps.load_brep(box_brep))

    assert mesher.has_shape is True


def test_add_nodes_returns_one_id_per_row_and_places_them_where_asked() -> None:
    mesher = Mesher()
    points = np.array(
        [[0.0, 0.0, 0.0], [STRIP_DX, STRIP_DY, STRIP_DZ]], dtype=np.float64
    )

    ids = mesher.add_nodes(points)

    assert ids.shape == (2,)
    assert len(set(ids.tolist())) == 2
    harvest = mesher.mesh()
    order = np.argsort(harvest.node_id)
    np.testing.assert_allclose(harvest.node_coords[order], points)


def test_add_triangles_binds_them_to_the_nodes_it_was_given() -> None:
    mesher = Mesher()
    ids = mesher.add_nodes(
        np.array([[0.0, 0.0, 0.0], [STRIP_DX, 0.0, 0.0], [0.0, STRIP_DY, 0.0]])
    )

    elements = mesher.add_triangles(ids[np.newaxis, :])

    assert elements.shape == (1,)
    harvest = mesher.mesh()
    assert harvest.element_count == 1
    assert int(harvest.element_type[0]) == ElementType.TRIANGLE
    # The area of the right triangle the three nodes span, from the mesh rather than the input.
    values = mesher.quality(Area()).values
    np.testing.assert_allclose(values, [0.5 * STRIP_DX * STRIP_DY])


def test_from_arrays_reads_connectivity_as_row_indices(strip: Mesher) -> None:
    harvest = strip.mesh()

    assert harvest.node_count == 6
    assert harvest.element_count == 4
    # Total area: two rectangles, each STRIP_DX by its own extent.
    total = float(strip.quality(Area()).values.sum())
    assert total == pytest.approx(STRIP_DX * STRIP_DY + STRIP_DY * STRIP_DZ)


def test_from_arrays_refuses_a_row_index_past_the_node_table() -> None:
    points, triangles = _folded_strip()
    triangles[0, 0] = len(points)

    with pytest.raises(IndexError, match="row index"):
        Mesher.from_arrays(points, triangles)


def test_add_elements_refuses_a_column_count_the_type_does_not_have() -> None:
    mesher = Mesher()
    ids = mesher.add_nodes(np.zeros((4, 3)) + np.arange(4)[:, np.newaxis])

    # Four columns given as a triangle would otherwise dispatch on the count alone and build
    # a quadrangle under the name the caller asked for.
    with pytest.raises(PysmeshError, match="has 3 nodes"):
        mesher.add_elements(ElementType.TRIANGLE, ids[np.newaxis, :])


def test_add_elements_refuses_a_polyhedron() -> None:
    mesher = Mesher()

    with pytest.raises(PysmeshError, match="polygon or a polyhedron"):
        mesher.add_elements(ElementType.POLYHEDRON, np.zeros((0, 4), dtype=np.int64))


def test_add_elements_refuses_a_node_the_mesh_does_not_have() -> None:
    mesher = Mesher()
    ids = mesher.add_nodes(np.zeros((3, 3)) + np.arange(3)[:, np.newaxis])
    conn = np.array([[ids[0], ids[1], int(ids.max()) + 100]], dtype=np.int64)

    with pytest.raises(PysmeshError, match="which the mesh does not have"):
        mesher.add_triangles(conn)


def test_from_mesh_round_trips_every_id(strip: Mesher) -> None:
    original = strip.mesh()

    rebuilt = Mesher.from_mesh(original).mesh()

    np.testing.assert_array_equal(np.sort(rebuilt.node_id), np.sort(original.node_id))
    np.testing.assert_array_equal(
        np.sort(rebuilt.element_id), np.sort(original.element_id)
    )
    np.testing.assert_allclose(
        rebuilt.node_coords[np.argsort(rebuilt.node_id)],
        original.node_coords[np.argsort(original.node_id)],
    )


def test_fill_from_mesh_refuses_a_mesher_that_already_holds_something(
    strip: Mesher,
) -> None:
    harvest = strip.mesh()

    with pytest.raises(PysmeshError, match="not empty"):
        strip.fill_from_mesh(harvest)


# ---- What a shape-free mesher cannot do ---------------------------------------------------- #


@pytest.mark.parametrize(
    ("operation", "call"),
    [
        ("Mesher.compute", lambda m: m.compute()),
        ("Mesher.assign", lambda m: m.assign(Regular1D())),
        ("Mesher.unassign", lambda m: m.unassign(Regular1D())),
        (
            "Mesher.add_group_on_shape",
            lambda m: m.add_group_on_shape(
                "wall", ElementDimension.FACE, SubShape(SubShapeKind.FACE, 1)
            ),
        ),
        ("Mesher.pattern_from_face", lambda m: m.pattern_from_face(1)),
    ],
)
def test_shape_dependent_operations_name_themselves_when_there_is_no_shape(
    strip: Mesher, operation: str, call: object
) -> None:
    with pytest.raises(PysmeshError, match="has no shape") as raised:
        call(strip)  # type: ignore[operator]

    assert operation in str(raised.value)


# ---- The surface that does work on one ------------------------------------------------------ #


def test_sharp_edges_finds_the_fold_of_a_shape_free_mesh(strip: Mesher) -> None:
    edges = strip.sharp_edges(angle=45.0)

    assert edges.node1.shape == (1,)
    # The fold runs along y at x = STRIP_DX: both its nodes sit there.
    harvest = strip.mesh()
    row = {int(i): r for r, i in enumerate(harvest.node_id)}
    for node in (int(edges.node1[0]), int(edges.node2[0])):
        assert harvest.node_coords[row[node], 0] == pytest.approx(STRIP_DX)


def test_separate_faces_by_edges_partitions_a_shape_free_mesh(strip: Mesher) -> None:
    patches = strip.separate_faces_by_edges(strip.sharp_edges(angle=45.0))

    assert patches.count == 2
    assert sorted(len(patches.at(i)) for i in range(2)) == [2, 2]
    assert patches.names == ()


def test_a_shape_free_mesher_carries_groups_and_filters(strip: Mesher) -> None:
    faces = strip.mesh().element_id[:2].tolist()

    strip.add_group("front", ElementDimension.FACE, faces)

    stored = {group.name: group.element_ids.tolist() for group in strip.groups()}
    assert stored["front"] == sorted(faces)


# ---- Deletion ------------------------------------------------------------------------------- #


def test_remove_elements_reports_exactly_the_elements_that_went(strip: Mesher) -> None:
    before = strip.mesh().element_id.tolist()
    target = before[1]

    report = strip.remove_elements([target])

    assert report.elements.tolist() == [target]
    assert report.nodes.tolist() == []
    assert report.faces_before - report.faces_after == 1
    assert strip.mesh().element_id.tolist() == [i for i in before if i != target]


def test_remove_elements_listing_one_id_twice_removes_it_once(strip: Mesher) -> None:
    target = int(strip.mesh().element_id[0])

    report = strip.remove_elements([target, target])

    assert report.elements.tolist() == [target]
    assert report.faces_before - report.faces_after == 1


def test_remove_elements_keeps_the_nodes_by_default(strip: Mesher) -> None:
    report = strip.remove_elements([int(strip.mesh().element_id[0])])

    assert report.nodes.tolist() == []
    assert report.nodes_before == report.nodes_after


def test_remove_elements_with_free_nodes_takes_the_orphans_and_names_them(
    strip: Mesher,
) -> None:
    harvest = strip.mesh()
    # Element 0 is the triangle (0, 1, 2); node row 0 is used by it alone.
    only_node = int(harvest.node_id[np.argsort(harvest.node_id)][0])

    report = strip.remove_elements([int(harvest.element_id[0])], free_nodes=True)

    assert report.nodes.tolist() == [only_node]
    assert report.nodes_before - report.nodes_after == 1
    assert only_node not in strip.mesh().node_id.tolist()


def test_remove_elements_refuses_an_unknown_id_before_deleting_anything(
    strip: Mesher,
) -> None:
    before = strip.mesh().element_id.tolist()

    with pytest.raises(PysmeshError, match="no element with id"):
        strip.remove_elements([before[0], max(before) + 100])

    assert strip.mesh().element_id.tolist() == before


def test_remove_nodes_takes_every_element_built_on_them_and_names_those(
    strip: Mesher,
) -> None:
    harvest = strip.mesh()
    order = np.argsort(harvest.node_id)
    # The node at the far corner of the folded half, used by two of the four triangles.
    node = int(harvest.node_id[order][5])
    carried = sorted(
        int(harvest.element_id[e])
        for e in range(harvest.element_count)
        if order[5] in harvest.nodes_of(e).tolist()
    )

    report = strip.remove_nodes([node])

    assert report.nodes.tolist() == [node]
    assert report.elements.tolist() == carried
    assert report.faces_before - report.faces_after == len(carried)


def test_remove_nodes_refuses_an_unknown_id_before_deleting_anything(
    strip: Mesher,
) -> None:
    before = strip.mesh().node_id.tolist()

    with pytest.raises(PysmeshError, match="no node with id"):
        strip.remove_nodes([before[0], max(before) + 100])

    assert strip.mesh().node_id.tolist() == before


def test_removing_everything_leaves_an_empty_reusable_mesher(strip: Mesher) -> None:
    harvest = strip.mesh()

    strip.remove_elements(harvest.element_id.tolist(), free_nodes=True)

    emptied = strip.mesh()
    assert emptied.node_count == 0
    assert emptied.element_count == 0
    # The point of emptying rather than releasing: the same mesher takes a new soup.
    points, triangles = _folded_strip()
    ids = strip.add_nodes(points)
    strip.add_triangles(ids[triangles])
    assert strip.mesh().element_count == 4


# ---- Patch identity across a deletion --------------------------------------------------------- #


def test_named_patches_survive_a_removal_with_their_index(strip: Mesher) -> None:
    patches = strip.separate_faces_by_edges(
        strip.sharp_edges(angle=45.0), name_prefix="patch_"
    )
    assert patches.names == ("patch_0", "patch_1")
    kept = {name: patches.at(i).tolist() for i, name in enumerate(patches.names)}

    strip.remove_elements(patches.at(0)[:1])

    after = {group.name: group.element_ids.tolist() for group in strip.groups()}
    # The deleted face left its group; every other membership is untouched.
    assert after["patch_0"] == kept["patch_0"][1:]
    assert after["patch_1"] == kept["patch_1"]


def test_named_patches_refuse_a_prefix_already_in_use(strip: Mesher) -> None:
    edges = strip.sharp_edges(angle=45.0)
    strip.separate_faces_by_edges(edges, name_prefix="patch_")

    with pytest.raises(PysmeshError, match="already exists"):
        strip.separate_faces_by_edges(edges, name_prefix="patch_")


def test_a_group_of_faces_drops_a_deleted_member_on_a_shape_backed_mesher() -> None:
    mesher = _box_mesher()
    harvest = mesher.mesh()
    # The harvest carries the edge segments too; a FACE group holds only the 2-D cells.
    faces = harvest.element_id[harvest.element_type == ElementType.QUADRANGLE].tolist()
    mesher.add_group("wall", ElementDimension.FACE, faces)

    report = mesher.remove_elements(faces[:2])

    stored = {group.name: group.element_ids.tolist() for group in mesher.groups()}
    assert report.elements.tolist() == sorted(faces[:2])
    assert stored["wall"] == sorted(faces[2:])
