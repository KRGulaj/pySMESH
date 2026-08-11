# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""Gates for the meshing assignment model, the body-fitted Cartesian mesher, and the harvest.

Four claims are under test, and each is checked against something the binding did not
produce.

* **The assignment model works, and it is what makes a mixed mesh possible.** An algorithm
  plus its hypotheses attach to a sub-shape; a global assignment is the default and one on a
  sub-shape overrides it there. The gate meshes one solid three different ways, then meshes
  two solids of one model by two different algorithms and asserts the result is
  **conforming node by node** at the internal boundary — a silently non-conforming mixed
  mesh being the failure mode the whole package risks.
* **The body-fitted Cartesian mesher works on a solid that is not a box**, and its result
  passes a quality verdict. The verdict is computed here from the returned arrays rather
  than taken from the mesh library, so it is an independent oracle: a cell's volume is
  summed by the divergence theorem over its own faces, which needs nothing but the
  coordinates and the connectivity.
* **A failure is loud and names the sub-shape.** An impossible assignment must raise with
  SMESH's own reason attached to the sub-shape that failed, not produce a silently empty
  mesh. The counter-case is asserted too, so the check cannot pass vacuously.
* **The harvest is complete and self-consistent.** Every element's connectivity indexes the
  node array, every polyhedron's face stream sums to its node count, and every element and
  node names the sub-shape it sits on.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    Cartesian3D,
    CartesianParameters3D,
    ElementType,
    EntityKind,
    Hexa3D,
    LocalLength,
    MaxElementArea,
    Mefisto2D,
    Mesher,
    MeshData,
    NumberOfSegments,
    PolyhedronPerSolid3D,
    PysmeshCancelled,
    PysmeshError,
    Quadrangle2D,
    Regular1D,
    Session,
    SubShape,
    SubShapeKind,
)

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0

# The block the Cartesian gate meshes: a box with a through hole, so the grid has to cut
# curved geometry rather than land on axis-aligned planes.
BLOCK_SIZE: float = 8.0
BLOCK_HEIGHT: float = 6.0
BORE_RADIUS: float = 1.5


# ---- Fixtures and helpers ------------------------------------------------------------- #


def _box_shape(dx: float = BOX_DX, dy: float = BOX_DY, dz: float = BOX_DZ) -> ps.Shape:
    """A single-solid box, through the session so the bytes are the real handoff bytes."""
    session = Session()
    session.add_box(dx, dy, dz)
    return ps.load_brep(session.brep())


def _bored_block_shape() -> ps.Shape:
    """A block with a through hole — a solid the Cartesian grid must cut curved faces on."""
    session = Session()
    session.add_box(
        BLOCK_SIZE, BLOCK_SIZE, BLOCK_HEIGHT, origin=(-BLOCK_SIZE / 2, -BLOCK_SIZE / 2, 0.0)
    )
    block = list(session.entities(EntityKind.SOLID))
    session.add_cylinder(BORE_RADIUS, BLOCK_HEIGHT)
    bore = [e for e in session.entities(EntityKind.SOLID) if e not in block]
    session.cut(block, bore)
    return ps.load_brep(session.brep())


def _stacked_solids_shape(side: float = 4.0) -> ps.Shape:
    """Two boxes sharing one face, as two solids of one model.

    A plain fuse of two face-touching boxes returns **one** solid — the seam face is internal
    to the result and is dropped — so the general fuse is used instead. It is the only
    operation that leaves an internal boundary for a mixed assignment to be conforming
    across, which is the whole point of the fixture.
    """
    session = Session()
    session.add_box(side, side, side)
    lower = list(session.entities(EntityKind.SOLID))
    session.add_box(side, side, side, origin=(0.0, 0.0, side))
    upper = [e for e in session.entities(EntityKind.SOLID) if e not in lower]
    session.fragment(lower + upper)
    return ps.load_brep(session.brep())


@pytest.fixture()
def box_mesher() -> Iterator[Mesher]:
    """A mesher on the 3 x 7 x 11 box, released afterwards."""
    with Mesher(_box_shape()) as mesher:
        yield mesher


def _structured(mesher: Mesher, segments: int) -> None:
    """Assign the cheapest fully structured hexahedral recipe to the whole shape."""
    mesher.assign(Regular1D())
    mesher.assign(NumberOfSegments(count=segments))
    mesher.assign(Quadrangle2D())
    mesher.assign(Hexa3D())


# The faces of each linear cell type, as node positions in SMESH's own ordering, wound so the
# outward normal points away from the cell. Taken from the element definitions rather than
# guessed: this table is what makes the volume oracle independent of the library under test.
_CELL_FACES: dict[ElementType, tuple[tuple[int, ...], ...]] = {
    ElementType.TETRAHEDRON: ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
    ElementType.PYRAMID: ((0, 1, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)),
    ElementType.PENTAHEDRON: (
        (0, 1, 2),
        (3, 5, 4),
        (0, 3, 4, 1),
        (1, 4, 5, 2),
        (2, 5, 3, 0),
    ),
    ElementType.HEXAHEDRON: (
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ),
}


def _polygon_volume_term(points: NDArray[np.float64]) -> float:
    """One face's contribution to a closed cell's volume, by the divergence theorem.

    The face is fanned from its first point, and each triangle contributes
    ``dot(a, cross(b, c)) / 6``. Summed over a closed, consistently wound surface this is the
    enclosed volume, with no assumption that any face is planar.
    """
    total = 0.0
    for i in range(1, len(points) - 1):
        a, b, c = points[0], points[i], points[i + 1]
        total += float(np.dot(a, np.cross(b, c))) / 6.0
    return total


def _cell_volume(mesh: MeshData, element: int) -> float:
    """Signed volume of one 3-D cell, computed from the arrays alone.

    Independent of the meshing library by construction: it reads only node coordinates and
    connectivity. A polyhedron's faces come from its own face stream; every other type's come
    from the table above.

    Args:
        mesh: The mesh holding the cell.
        element: 0-based position in the element arrays.

    Returns:
        The cell's signed volume. Positive for a correctly oriented cell.
    """
    nodes = mesh.nodes_of(element)
    kind = ElementType(int(mesh.element_type[element]))
    if kind is ElementType.POLYHEDRON:
        total = 0.0
        cursor = 0
        for size in mesh.face_sizes_of(element):
            face = nodes[cursor : cursor + int(size)]
            total += _polygon_volume_term(mesh.node_coords[face])
            cursor += int(size)
        return total
    faces = _CELL_FACES[kind]
    return sum(_polygon_volume_term(mesh.node_coords[nodes[list(f)]]) for f in faces)


def _volume_cells(mesh: MeshData) -> list[int]:
    """Positions of every 3-D element in the arrays."""
    volumetric = {
        int(ElementType.TETRAHEDRON),
        int(ElementType.PYRAMID),
        int(ElementType.PENTAHEDRON),
        int(ElementType.HEXAHEDRON),
        int(ElementType.POLYHEDRON),
    }
    return [i for i, t in enumerate(mesh.element_type.tolist()) if int(t) in volumetric]


# ---- The assignment model -------------------------------------------------------------- #


def test_assign_global_algorithm_and_hypothesis_meshes_the_whole_shape(
    box_mesher: Mesher,
) -> None:
    _structured(box_mesher, segments=3)

    report = box_mesher.compute()

    assert report.volumes == 27
    assert report.nodes == 4 * 4 * 4
    assert box_mesher.mesh().count_of(ElementType.HEXAHEDRON) == 27


def test_assignments_report_what_was_attached_and_where(box_mesher: Mesher) -> None:
    box_mesher.assign(Regular1D())
    box_mesher.assign(NumberOfSegments(count=2), on=SubShape(SubShapeKind.EDGE, 1))

    attached = box_mesher.assignments()

    assert attached == (
        ("Regular_1D", None),
        ("NumberOfSegments", SubShape(SubShapeKind.EDGE, 1)),
    )


def _free_surface(mesher: Mesher, max_area: float) -> None:
    """Assign a free triangle surface mesh to the whole shape."""
    mesher.assign(Regular1D())
    mesher.assign(LocalLength(length=1.5))
    mesher.assign(Mefisto2D())
    mesher.assign(MaxElementArea(max_area=max_area))


def test_a_hypothesis_on_a_sub_shape_overrides_the_global_one() -> None:
    """Scoping is the whole assignment model; without it there is only a global recipe."""
    with Mesher(_box_shape()) as plain:
        _free_surface(plain, max_area=4.0)
        uniform = plain.compute().faces

    with Mesher(_box_shape()) as scoped:
        _free_surface(scoped, max_area=4.0)
        scoped.assign(LocalLength(length=0.4), on=SubShape(SubShapeKind.FACE, 1))
        refined = scoped.compute()

    assert refined.faces > uniform
    # The refinement must land on the face it was scoped to, not across the model.
    on_face_one = [
        m for m in refined.meshed if m.kind is SubShapeKind.FACE and m.ordinal == 1
    ]
    others = [m for m in refined.meshed if m.kind is SubShapeKind.FACE and m.ordinal != 1]
    assert on_face_one[0].elements > max(m.elements for m in others)


def test_unassign_removes_an_assignment_and_leaves_the_rest(box_mesher: Mesher) -> None:
    _structured(box_mesher, segments=2)

    box_mesher.unassign(Hexa3D())

    assert [name for name, _ in box_mesher.assignments()] == [
        "Regular_1D",
        "NumberOfSegments",
        "Quadrangle_2D",
    ]
    assert box_mesher.compute().volumes == 0


def test_unassign_something_never_assigned_raises(box_mesher: Mesher) -> None:
    box_mesher.assign(Regular1D())

    with pytest.raises(PysmeshError, match="not assigned"):
        box_mesher.unassign(Hexa3D())


def test_a_second_algorithm_of_the_same_dimension_is_refused(box_mesher: Mesher) -> None:
    box_mesher.assign(Quadrangle2D())

    with pytest.raises(PysmeshError, match="already assigned"):
        box_mesher.assign(Mefisto2D())


def test_computing_with_nothing_assigned_raises(box_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="nothing is assigned"):
        box_mesher.compute()


def test_an_out_of_range_ordinal_raises_naming_the_kind(box_mesher: Mesher) -> None:
    with pytest.raises(PysmeshError, match="Invalid face_id 999"):
        box_mesher.assign(Quadrangle2D(), on=SubShape(SubShapeKind.FACE, 999))


def test_a_sub_shape_ordinal_below_one_is_refused_before_the_native_call() -> None:
    with pytest.raises(ValueError, match="1-based"):
        SubShape(SubShapeKind.FACE, 0)


def test_a_sub_shape_kind_of_none_is_refused() -> None:
    with pytest.raises(ValueError, match="must name a kind"):
        SubShape(SubShapeKind.NONE, 1)


@pytest.mark.parametrize(
    ("segments", "expected_cells"), [(1, 1), (2, 8), (3, 27), (4, 64)]
)
def test_segment_count_sets_the_structured_cell_count(
    segments: int, expected_cells: int
) -> None:
    with Mesher(_box_shape()) as mesher:
        _structured(mesher, segments=segments)

        assert mesher.compute().volumes == expected_cells


# ---- Three algorithms on one solid ------------------------------------------------------ #


def test_one_solid_meshes_by_three_different_algorithms_each_with_its_own_hypotheses() -> None:
    """The assignment model's headline claim, with a quality verdict on every result.

    Three distinct 3-D recipes, each with a hypothesis set of its own, each producing cells
    whose volumes are all positive and sum to the solid's own volume. The verdict is computed
    from the returned arrays, not read from the mesh library.
    """
    expected_volume = BOX_DX * BOX_DY * BOX_DZ
    produced: dict[str, int] = {}

    recipes: dict[str, list[object]] = {
        "structured hexahedra": [
            Regular1D(),
            NumberOfSegments(count=3),
            Quadrangle2D(),
            Hexa3D(),
        ],
        "one polyhedron per solid": [PolyhedronPerSolid3D()],
        "body-fitted Cartesian": [
            Cartesian3D(),
            CartesianParameters3D(spacing_x="1.0", spacing_y="1.0", spacing_z="1.0"),
        ],
    }

    for name, recipe in recipes.items():
        with Mesher(_box_shape()) as mesher:
            for item in recipe:
                mesher.assign(item)  # type: ignore[arg-type]  # Algorithm | Hypothesis
            report = mesher.compute()
            mesh = mesher.mesh()

            cells = _volume_cells(mesh)
            volumes = [_cell_volume(mesh, c) for c in cells]

            assert report.volumes > 0, name
            assert len(cells) == report.volumes, name
            assert min(volumes) > 0.0, f"{name}: {sum(v <= 0 for v in volumes)} bad cells"
            assert math.isclose(sum(volumes), expected_volume, rel_tol=1e-9), name
            produced[name] = report.volumes

    # Distinct recipes must actually produce distinct meshes, or the loop proved nothing.
    assert len(set(produced.values())) == len(produced), produced


# ---- A mixed assignment, and its conformity --------------------------------------------- #


def test_two_solids_meshed_by_different_algorithms_conform_node_by_node() -> None:
    """The failure mode this whole package risks is a silently non-conforming mixed mesh.

    Two solids share one face. One is meshed structured, the other by the polyhedral mesher.
    Conformity is asserted per node of the shared face: every node on it must be a single
    node that cells of **both** solids use, not two coincident ones. The counter-check is in
    the next test, so a passing result here is not a property of any two algorithms.
    """
    shape = _stacked_solids_shape()
    solids = list(range(1, len(shape.solids()) + 1))
    assert len(solids) == 2, "the fixture must have two solids to have an interface at all"

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D(), on=SubShape(SubShapeKind.SOLID, solids[0]))
        mesher.assign(PolyhedronPerSolid3D(), on=SubShape(SubShapeKind.SOLID, solids[1]))
        mesher.compute()
        mesh = mesher.mesh()

    # The interface is the one face both solids own. Read it from the mesh: a node of the
    # interface is one used by cells of both solids.
    owners: dict[int, set[int]] = {}
    for element in _volume_cells(mesh):
        if int(mesh.element_kind[element]) != int(SubShapeKind.SOLID):
            continue
        solid = int(mesh.element_ordinal[element])
        for node in mesh.nodes_of(element):
            owners.setdefault(int(node), set()).add(solid)

    shared = {node for node, s in owners.items() if len(s) > 1}
    assert shared, "no node is used by both solids — the mesh is not conforming at all"

    # Every node that lies on the interface plane must be one of them. The plane is the only
    # place the two solids meet, so this is the node-by-node statement.
    side = 4.0
    on_plane = {
        int(i)
        for i in range(mesh.node_count)
        if math.isclose(float(mesh.node_coords[i, 2]), side, abs_tol=1e-9)
        and len(owners.get(int(i), set())) > 0
    }
    assert on_plane, "the fixture put no node on the interface plane"
    assert on_plane <= shared, f"{len(on_plane - shared)} interface nodes are duplicated"


def test_a_cartesian_region_does_not_conform_and_the_check_can_see_it() -> None:
    """The falsification: the conformity check must fail on a mesh that is not conforming.

    The body-fitted Cartesian mesher lays its own grid and ignores any boundary mesh, so two
    solids meshed with it independently do not meet. Without this, the previous test would be
    asserting a property of the fixture rather than of the assignment.
    """
    shape = _stacked_solids_shape()
    solids = list(range(1, len(shape.solids()) + 1))

    with Mesher(shape) as mesher:
        mesher.assign(Cartesian3D())
        mesher.assign(
            CartesianParameters3D(spacing_x="1.3", spacing_y="1.3", spacing_z="1.3")
        )
        mesher.compute()
        mesh = mesher.mesh()

    owners: dict[int, set[int]] = {}
    for element in _volume_cells(mesh):
        if int(mesh.element_kind[element]) != int(SubShapeKind.SOLID):
            continue
        for node in mesh.nodes_of(element):
            owners.setdefault(int(node), set()).add(int(mesh.element_ordinal[element]))

    side = 4.0
    on_plane = [
        node
        for node, s in owners.items()
        if math.isclose(float(mesh.node_coords[node, 2]), side, abs_tol=1e-9)
    ]
    shared = [node for node in on_plane if len(owners[node]) > 1]

    assert len(solids) == 2
    assert on_plane, "the fixture put no node on the interface plane"
    assert not shared, "a Cartesian region unexpectedly conformed; the oracle proves nothing"


# ---- The body-fitted Cartesian mesher --------------------------------------------------- #


def test_cartesian_meshes_a_bored_block_and_every_cell_has_positive_volume() -> None:
    """The gate for the un-excluded translation units, met through the shipped binding.

    The solid is deliberately not a box: a block with a through hole, so the grid has to cut
    curved faces. The verdict is per cell and independent of the mesh library.
    """
    with Mesher(_bored_block_shape()) as mesher:
        mesher.assign(Cartesian3D())
        mesher.assign(
            CartesianParameters3D(spacing_x="1.0", spacing_y="1.0", spacing_z="1.0")
        )
        report = mesher.compute()
        mesh = mesher.mesh()

    cells = _volume_cells(mesh)
    volumes = [_cell_volume(mesh, c) for c in cells]
    expected = BLOCK_SIZE * BLOCK_SIZE * BLOCK_HEIGHT - math.pi * BORE_RADIUS**2 * (
        BLOCK_HEIGHT
    )

    assert report.volumes > 0
    assert min(volumes) > 0.0, f"{sum(v <= 0 for v in volumes)} cells have non-positive volume"
    # The Cartesian mesh approximates the bore with flat cut faces, so it is close to but not
    # equal to the analytic volume. 2 % on a bore of this size is the discretisation, not an
    # error in the cells.
    assert math.isclose(sum(volumes), expected, rel_tol=2e-2)


def test_cartesian_emits_hexahedra_inside_and_polyhedra_at_the_cut_cells() -> None:
    """The property that decides the array layout, asserted rather than assumed."""
    with Mesher(_bored_block_shape()) as mesher:
        mesher.assign(Cartesian3D())
        mesher.assign(
            CartesianParameters3D(spacing_x="1.0", spacing_y="1.0", spacing_z="1.0")
        )
        mesher.compute()
        mesh = mesher.mesh()

    assert mesh.count_of(ElementType.HEXAHEDRON) > 0
    assert mesh.count_of(ElementType.POLYHEDRON) > 0


def test_every_polyhedron_face_stream_sums_to_its_node_count() -> None:
    """A polyhedron's node list *is* a face stream; the two must agree or it is
    unreadable.
    """
    with Mesher(_bored_block_shape()) as mesher:
        mesher.assign(Cartesian3D())
        mesher.assign(
            CartesianParameters3D(spacing_x="1.0", spacing_y="1.0", spacing_z="1.0")
        )
        mesher.compute()
        mesh = mesher.mesh()

    checked = 0
    for element in range(mesh.element_count):
        if int(mesh.element_type[element]) != int(ElementType.POLYHEDRON):
            assert mesh.face_sizes_of(element).size == 0
            continue
        checked += 1
        assert int(mesh.face_sizes_of(element).sum()) == len(mesh.nodes_of(element))
        assert mesh.face_sizes_of(element).min() >= 3

    assert checked > 0, "the fixture produced no polyhedron to check"


def test_an_all_dimension_algorithm_hides_a_lower_dimension_one_rather_than_refusing_it() -> (
    None
):
    """SMESH treats hiding as a normal state, so the 2-D assignment is accepted and ignored.

    Pinned because the natural expectation is the opposite, and because a caller whose 2-D
    assignment silently has no effect has no other way to find out. The observable
    consequence is in the report: the faces receive no elements of their own.
    """
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Cartesian3D())
        mesher.assign(
            CartesianParameters3D(spacing_x="1.0", spacing_y="1.0", spacing_z="1.0")
        )

        mesher.assign(Quadrangle2D())  # accepted, and then hidden
        report = mesher.compute()

    assert report.volumes > 0
    assert not [m for m in report.meshed if m.kind is SubShapeKind.FACE]


# ---- Failure reporting ------------------------------------------------------------------ #


def test_an_impossible_assignment_raises_naming_the_failing_faces() -> None:
    """A mapped quadrangle mesher cannot read a disk as four sides, and must say so."""
    session = Session()
    session.add_cylinder(2.0, 5.0)
    shape = ps.load_brep(session.brep())

    with Mesher(shape) as mesher:
        _structured(mesher, segments=3)

        with pytest.raises(PysmeshError) as raised:
            mesher.compute()

    assert "meshing failed" in str(raised.value)
    assert raised.value.face_ids, "the error names no face"
    assert "4 sides" in raised.value.details
    assert "Quadrangle_2D" in raised.value.details


def test_the_same_shape_meshes_when_the_algorithm_can_read_it() -> None:
    """The counter-case: without it the failure test would pass on any broken compute."""
    session = Session()
    session.add_cylinder(2.0, 5.0)
    shape = ps.load_brep(session.brep())

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(LocalLength(length=1.0))
        mesher.assign(Mefisto2D())
        mesher.assign(MaxElementArea(max_area=2.0))

        report = mesher.compute()

    assert report.faces > 0


def test_a_failed_compute_keeps_the_partial_mesh_for_diagnosis() -> None:
    """A failure is information: how far the assignment got is the diagnostic."""
    session = Session()
    session.add_cylinder(2.0, 5.0)
    shape = ps.load_brep(session.brep())

    with Mesher(shape) as mesher:
        _structured(mesher, segments=3)
        with pytest.raises(PysmeshError):
            mesher.compute()

        assert mesher.mesh().element_count > 0


# ---- The harvest ------------------------------------------------------------------------ #


def test_every_element_connectivity_indexes_the_node_array(box_mesher: Mesher) -> None:
    _structured(box_mesher, segments=3)
    box_mesher.compute()

    mesh = box_mesher.mesh()

    assert mesh.element_nodes.min() >= 0
    assert mesh.element_nodes.max() < mesh.node_count
    assert int(mesh.element_offsets[-1]) == mesh.element_nodes.size
    assert mesh.element_offsets.size == mesh.element_count + 1


def test_every_element_and_node_names_the_sub_shape_it_sits_on(box_mesher: Mesher) -> None:
    _structured(box_mesher, segments=2)
    box_mesher.compute()

    mesh = box_mesher.mesh()

    assert int(mesh.element_kind.min()) > int(SubShapeKind.NONE)
    assert int(mesh.node_kind.min()) > int(SubShapeKind.NONE)
    assert int(mesh.element_ordinal.min()) >= 1
    # A hexahedron of a one-solid box can only be on solid 1; a quadrangle on one of 6 faces.
    hexes = mesh.element_type == int(ElementType.HEXAHEDRON)
    assert set(mesh.element_kind[hexes].tolist()) == {int(SubShapeKind.SOLID)}
    assert set(mesh.element_ordinal[hexes].tolist()) == {1}
    quads = mesh.element_type == int(ElementType.QUADRANGLE)
    assert set(mesh.element_kind[quads].tolist()) == {int(SubShapeKind.FACE)}
    assert set(mesh.element_ordinal[quads].tolist()) == set(range(1, 7))


def test_elements_are_grouped_by_ascending_dimension(box_mesher: Mesher) -> None:
    """A consumer wanting only the volume cells must be able to read one contiguous span."""
    _structured(box_mesher, segments=2)
    box_mesher.compute()

    mesh = box_mesher.mesh()
    dimension = {
        int(ElementType.EDGE): 1,
        int(ElementType.QUADRANGLE): 2,
        int(ElementType.HEXAHEDRON): 3,
    }
    order = [dimension[int(t)] for t in mesh.element_type.tolist()]

    assert order == sorted(order)


def test_the_report_names_which_sub_shapes_received_elements(box_mesher: Mesher) -> None:
    _structured(box_mesher, segments=2)

    report = box_mesher.compute()

    solids = [m for m in report.meshed if m.kind is SubShapeKind.SOLID]
    faces = [m for m in report.meshed if m.kind is SubShapeKind.FACE]
    assert len(solids) == 1
    assert solids[0].elements == 8
    assert len(faces) == 6


def test_node_ids_are_unique_and_element_ids_are_unique(box_mesher: Mesher) -> None:
    _structured(box_mesher, segments=3)
    box_mesher.compute()

    mesh = box_mesher.mesh()

    assert len(set(mesh.node_id.tolist())) == mesh.node_count
    assert len(set(mesh.element_id.tolist())) == mesh.element_count


def test_nodes_of_an_out_of_range_element_raises(box_mesher: Mesher) -> None:
    _structured(box_mesher, segments=2)
    box_mesher.compute()
    mesh = box_mesher.mesh()

    with pytest.raises(IndexError, match="out of range"):
        mesh.nodes_of(mesh.element_count)


def test_reading_a_mesh_never_advances_anything(box_mesher: Mesher) -> None:
    _structured(box_mesher, segments=2)
    box_mesher.compute()

    first = box_mesher.mesh()
    second = box_mesher.mesh()

    assert np.array_equal(first.node_coords, second.node_coords)
    assert np.array_equal(first.element_nodes, second.element_nodes)


# ---- Lifetime ---------------------------------------------------------------------------- #


def test_a_released_mesher_refuses_every_operation() -> None:
    mesher = Mesher(_box_shape())
    mesher.release()

    assert not mesher.is_open()
    with pytest.raises(PysmeshError, match="released"):
        mesher.assign(Regular1D())


def test_release_is_idempotent() -> None:
    mesher = Mesher(_box_shape())

    mesher.release()
    mesher.release()

    assert not mesher.is_open()


def test_two_meshers_coexist_without_cross_talk() -> None:
    with Mesher(_box_shape()) as first, Mesher(_box_shape()) as second:
        _structured(first, segments=2)
        _structured(second, segments=3)

        assert first.compute().volumes == 8
        assert second.compute().volumes == 27


# ---- Progress and cancellation ------------------------------------------------------------ #


def _slow_cartesian(mesher: Mesher, spacing: str) -> None:
    """Assign a body-fitted Cartesian run fine enough for a poller to observe it."""
    mesher.assign(Cartesian3D())
    mesher.assign(
        CartesianParameters3D(spacing_x=spacing, spacing_y=spacing, spacing_z=spacing)
    )


def test_progress_is_monotone_and_ends_at_one() -> None:
    """The reported values are real and ordered; what they are *worth* is the next test."""
    seen: list[float] = []

    with Mesher(_bored_block_shape()) as mesher:
        _slow_cartesian(mesher, spacing="0.12")
        mesher.compute(progress=seen.append)

    assert len(seen) > 5, f"only {len(seen)} updates from a run that takes ~350 ms"
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in seen)


def test_progress_inside_one_algorithm_is_a_tick_count_not_a_fraction_of_the_work() -> None:
    """Pinned because the number looks like a fraction and is not one.

    SMESH computes the fraction of *sub-meshes* already done exactly, and interpolates inside
    a running algorithm with a counter that advances once per enquiry. An algorithm that
    meshes the whole model in one call therefore reports values that creep up from near zero
    and jump to 1.0 at the end. A caller sizing a progress bar on the value would see it sit
    still for the whole run, so the contract says this rather than implying otherwise.
    """
    seen: list[float] = []

    with Mesher(_bored_block_shape()) as mesher:
        _slow_cartesian(mesher, spacing="0.12")
        mesher.compute(progress=seen.append)

    assert max(seen[:-1]) < 0.1, seen[:5]
    assert seen[-1] == pytest.approx(1.0)


def test_a_pre_set_cancel_flag_stops_even_a_mesh_that_finishes_quickly() -> None:
    """The floor a poll-driven contract has to close at the start rather than document."""
    with Mesher(_box_shape()) as mesher:
        _structured(mesher, segments=2)

        with pytest.raises(PysmeshCancelled):
            mesher.compute(cancel=lambda: True)


def test_a_cancelled_compute_leaves_no_partial_mesh() -> None:
    with Mesher(_box_shape()) as mesher:
        _structured(mesher, segments=6)

        with pytest.raises(PysmeshCancelled):
            mesher.compute(cancel=lambda: True)

        assert mesher.mesh().element_count == 0


def test_a_raising_progress_callback_cancels_and_the_caller_gets_its_own_exception() -> None:
    class Stop(RuntimeError):
        """Raised by the hook under test."""

    def explode(_: float) -> None:
        raise Stop("stop")

    with Mesher(_bored_block_shape()) as mesher:
        _slow_cartesian(mesher, spacing="0.12")

        with pytest.raises(Stop):
            mesher.compute(progress=explode)

        assert mesher.mesh().element_count == 0


def test_the_same_mesh_completes_when_the_predicate_says_no() -> None:
    """The falsification: without it an always-raising implementation would pass the above."""
    with Mesher(_box_shape()) as mesher:
        _structured(mesher, segments=6)

        report = mesher.compute(cancel=lambda: False)

    assert report.volumes == 6**3


def test_a_non_callable_progress_hook_is_refused(box_mesher: Mesher) -> None:
    _structured(box_mesher, segments=2)

    with pytest.raises(PysmeshError, match="progress must be callable"):
        box_mesher.compute(progress=42)  # type: ignore[arg-type]  # the point of the test


def test_a_non_callable_cancel_hook_is_refused(box_mesher: Mesher) -> None:
    _structured(box_mesher, segments=2)

    with pytest.raises(PysmeshError, match="cancel must be callable"):
        box_mesher.compute(cancel=42)  # type: ignore[arg-type]  # the point of the test


def test_hooks_cost_nothing_measurable_on_a_mesh_that_uses_them() -> None:
    """Reporting must not change what is produced, whatever it costs."""
    with Mesher(_box_shape()) as plain:
        _structured(plain, segments=8)
        bare = plain.compute()
        bare_nodes = plain.mesh().node_coords

    with Mesher(_box_shape()) as hooked:
        _structured(hooked, segments=8)
        seen: list[float] = []
        hooked_report = hooked.compute(progress=seen.append, cancel=lambda: False)
        hooked_nodes = hooked.mesh().node_coords

    assert hooked_report.volumes == bare.volumes
    assert np.array_equal(hooked_nodes, bare_nodes)


@pytest.mark.slow
def test_a_cancel_mid_algorithm_stops_a_long_cartesian_run_in_budget() -> None:
    """Cancellation latency, on the one 3-D algorithm that polls the flag inside its loop.

    Stated as it is rather than as a flat bound, because the bound is a property of SMESH:
    a cancel is honoured between sub-meshes for every algorithm, and *within* a run only by
    the three that poll it. This is one of those three.
    """
    started = time.perf_counter()
    requested: list[float] = []

    def cancel_after_a_moment() -> bool:
        if time.perf_counter() - started < 0.15:
            return False
        if not requested:
            requested.append(time.perf_counter())
        return True

    with Mesher(_bored_block_shape()) as mesher:
        _slow_cartesian(mesher, spacing="0.06")

        with pytest.raises(PysmeshCancelled):
            mesher.compute(cancel=cancel_after_a_moment)
        latency = time.perf_counter() - requested[0]

        assert mesher.mesh().element_count == 0

    assert latency < 2.0, f"the cancel took {latency * 1000:.0f} ms to land"
