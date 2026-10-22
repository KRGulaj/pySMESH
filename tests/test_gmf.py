# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""Gates for the Inria ``.mesh`` / ``.meshb`` interchange.

The format is what MMG and fTetWild read and write, so the value of this binding is that a
mesh survives the trip. Three claims are under test, and the last two matter more than the
first because they are about what the format *cannot* do.

* **A mesh round-trips with its counts and its groups.** Written and read back, node and
  per-type element counts are preserved exactly, and a required-entity group comes back with
  the same membership.
* **What cannot be represented is refused, not dropped.** The format has no polygon and no
  polyhedron, so a body-fitted Cartesian mesh — which is hexahedra plus polyhedra at every
  cut cell — has no faithful representation at all. Writing one raises naming the element,
  rather than producing a file that is quietly missing its cut cells. The same holds for a
  group whose name the format's one group channel cannot carry.
* **What the round trip loses is stated and asserted.** The writer emits each element's
  sub-shape index as its reference and the reader discards it, so a mesh read from a file is
  bound to no geometry. Element ids are renumbered per type. Both are asserted here so that a
  consumer reading this suite learns the limits rather than discovering them.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    Cartesian3D,
    CartesianParameters3D,
    ElementDimension,
    ElementType,
    GroupSource,
    EntityKind,
    GmfMesh,
    Hexa3D,
    Mefisto2D,
    MaxElementArea,
    Mesher,
    MeshData,
    MeshGroup,
    NumberOfSegments,
    PysmeshError,
    Quadrangle2D,
    Regular1D,
    Session,
    SubShapeKind,
    gmf_unwritable_types,
    gmf_writable_group_name,
    read_gmf,
    write_gmf,
)

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0


# ---- Fixtures --------------------------------------------------------------------------- #


def _box_shape() -> ps.Shape:
    """The 3 x 7 x 11 box, through the session."""
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    return ps.load_brep(session.brep())


@pytest.fixture(scope="module")
def hexa_mesh() -> MeshData:
    """A structured hexahedral mesh: edges, quadrangles and hexahedra, all writable."""
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.compute()
        return mesher.mesh()


@pytest.fixture(scope="module")
def triangle_mesh() -> MeshData:
    """A free surface mesh: edges and triangles, for the required-triangle group channel."""
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Mefisto2D())
        mesher.assign(MaxElementArea(max_area=4.0))
        mesher.compute()
        return mesher.mesh()


@pytest.fixture(scope="module")
def cartesian_mesh() -> MeshData:
    """A body-fitted Cartesian mesh, whose cut cells are polyhedra the format cannot hold."""
    session = Session()
    session.add_box(8.0, 8.0, 6.0, origin=(-4.0, -4.0, 0.0))
    block = list(session.entities(EntityKind.SOLID))
    session.add_cylinder(1.5, 6.0)
    bore = [e for e in session.entities(EntityKind.SOLID) if e not in block]
    session.cut(block, bore)

    with Mesher(ps.load_brep(session.brep())) as mesher:
        mesher.assign(Cartesian3D())
        mesher.assign(
            CartesianParameters3D(spacing_x="1.0", spacing_y="1.0", spacing_z="1.0")
        )
        mesher.compute()
        return mesher.mesh()


def _type_counts(mesh: MeshData) -> dict[int, int]:
    """How many elements of each type the mesh holds."""
    values, counts = np.unique(mesh.element_type, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts, strict=True)}


# ---- The round trip ---------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix", [".mesh", ".meshb"])
def test_a_mesh_round_trips_with_its_counts_in_both_formats(
    hexa_mesh: MeshData, tmp_path: Path, suffix: str
) -> None:
    """The extension picks text or binary; both must carry the same mesh."""
    path = tmp_path / f"box{suffix}"

    write_gmf(path, hexa_mesh)
    back = read_gmf(path)

    assert path.is_file()
    assert back.mesh.node_count == hexa_mesh.node_count
    assert back.mesh.element_count == hexa_mesh.element_count
    assert _type_counts(back.mesh) == _type_counts(hexa_mesh)


def test_the_binary_and_text_forms_hold_the_same_mesh(
    hexa_mesh: MeshData, tmp_path: Path
) -> None:
    """Different files, same content — otherwise one of the two formats is lossy."""
    write_gmf(tmp_path / "a.mesh", hexa_mesh)
    write_gmf(tmp_path / "a.meshb", hexa_mesh)

    text = read_gmf(tmp_path / "a.mesh")
    binary = read_gmf(tmp_path / "a.meshb")

    assert (tmp_path / "a.mesh").read_bytes() != (tmp_path / "a.meshb").read_bytes()
    assert _type_counts(text.mesh) == _type_counts(binary.mesh)
    assert np.allclose(
        np.sort(text.mesh.node_coords, axis=0),
        np.sort(binary.mesh.node_coords, axis=0),
    )


def test_node_coordinates_survive_the_round_trip(
    hexa_mesh: MeshData, tmp_path: Path
) -> None:
    """Counts alone would pass on a mesh whose geometry was mangled."""
    path = tmp_path / "coords.meshb"

    write_gmf(path, hexa_mesh)
    back = read_gmf(path)

    before = np.sort(hexa_mesh.node_coords, axis=0)
    after = np.sort(back.mesh.node_coords, axis=0)
    assert np.allclose(before, after, atol=1e-12)


def test_cell_connectivity_survives_the_round_trip(
    hexa_mesh: MeshData, tmp_path: Path
) -> None:
    """Every cell must come back over the same points, as a set of vertex positions.

    Compared as sorted coordinate tuples rather than as node indices, because the format
    renumbers: the connectivity is the same mesh even though the ids are not the same ids.
    """
    path = tmp_path / "conn.meshb"
    write_gmf(path, hexa_mesh)
    back = read_gmf(path)

    def cell_signatures(mesh: MeshData, element_type: ElementType) -> set[tuple[float, ...]]:
        out: set[tuple[float, ...]] = set()
        for element in range(mesh.element_count):
            if int(mesh.element_type[element]) != int(element_type):
                continue
            points = mesh.node_coords[mesh.nodes_of(element)]
            out.add(tuple(np.round(np.sort(points, axis=0), 9).ravel().tolist()))
        return out

    before = cell_signatures(hexa_mesh, ElementType.HEXAHEDRON)
    after = cell_signatures(back.mesh, ElementType.HEXAHEDRON)

    assert before
    assert before == after


def test_a_triangle_mesh_round_trips_too(
    triangle_mesh: MeshData, tmp_path: Path
) -> None:
    path = tmp_path / "tri.meshb"

    write_gmf(path, triangle_mesh)
    back = read_gmf(path)

    assert back.mesh.count_of(ElementType.TRIANGLE) == triangle_mesh.count_of(
        ElementType.TRIANGLE
    )
    assert back.mesh.count_of(ElementType.EDGE) == triangle_mesh.count_of(ElementType.EDGE)


# ---- Groups ------------------------------------------------------------------------------ #


def test_a_required_group_round_trips_with_its_membership(
    triangle_mesh: MeshData, tmp_path: Path
) -> None:
    """The format's one group channel, exercised end to end."""
    triangles = [
        int(triangle_mesh.element_id[i])
        for i in range(triangle_mesh.element_count)
        if int(triangle_mesh.element_type[i]) == int(ElementType.TRIANGLE)
    ]
    marked = triangles[:5]
    group = MeshGroup(
        name=gmf_writable_group_name("Triangles"),
        dimension=ElementDimension.FACE,
        source=GroupSource.EXPLICIT,
        element_ids=np.array(marked, dtype=np.int64),
    )
    path = tmp_path / "grouped.meshb"

    write_gmf(path, triangle_mesh, [group])
    back = read_gmf(path)

    required = [g for g in back.groups if "Triangles" in g.name]
    assert len(required) == 1, [g.name for g in back.groups]
    assert required[0].element_ids.size == len(marked)


def test_a_group_the_format_cannot_carry_is_refused_naming_it(
    triangle_mesh: MeshData, tmp_path: Path
) -> None:
    """Refused rather than dropped: a group that vanishes silently is the worse outcome."""
    group = MeshGroup(
        name="inlet",
        dimension=ElementDimension.FACE,
        source=GroupSource.EXPLICIT,
        element_ids=triangle_mesh.element_id[:3],
    )

    with pytest.raises(PysmeshError, match="required entity"):
        write_gmf(tmp_path / "bad.meshb", triangle_mesh, [group])


def test_a_mesh_with_no_groups_writes_and_reads_back_with_none(
    hexa_mesh: MeshData, tmp_path: Path
) -> None:
    """The control for the group tests: nothing in, nothing out."""
    path = tmp_path / "plain.meshb"

    write_gmf(path, hexa_mesh)
    back = read_gmf(path)

    assert back.groups == ()


@pytest.mark.parametrize(
    "suffix", ["Vertices", "Edges", "Triangles", "Quadrilaterals"]
)
def test_every_group_suffix_the_format_defines_is_accepted(suffix: str) -> None:
    name = gmf_writable_group_name(suffix)

    assert name.endswith(suffix)
    assert "_required_" in name


def test_a_suffix_the_format_does_not_define_is_refused() -> None:
    with pytest.raises(ValueError, match="required-entity sets"):
        gmf_writable_group_name("Tetrahedra")


# ---- What the format cannot hold ---------------------------------------------------------- #


def test_a_cartesian_mesh_cannot_be_written_and_says_which_element_stops_it(
    cartesian_mesh: MeshData, tmp_path: Path
) -> None:
    """The headline limit: the body-fitted mesher's cut cells have no keyword in the format."""
    assert cartesian_mesh.count_of(ElementType.POLYHEDRON) > 0

    with pytest.raises(PysmeshError, match="polyhedron"):
        write_gmf(tmp_path / "cartesian.meshb", cartesian_mesh)


def test_the_unwritable_types_can_be_asked_for_before_trying(
    cartesian_mesh: MeshData, hexa_mesh: MeshData
) -> None:
    """Asking beforehand is the difference between choosing another route and catching."""
    assert gmf_unwritable_types(cartesian_mesh) == (ElementType.POLYHEDRON,)
    assert gmf_unwritable_types(hexa_mesh) == ()


def test_the_per_element_cad_binding_does_not_survive_the_round_trip(
    hexa_mesh: MeshData, tmp_path: Path
) -> None:
    """Written as each element's reference, and discarded by the reader. Asserted, not assumed.

    Stated as a gate because a consumer that assumed otherwise would silently lose the link
    between a mesh cell and the face it came from — the thing the whole handoff exists for.
    """
    path = tmp_path / "binding.meshb"
    write_gmf(path, hexa_mesh)
    back = read_gmf(path)

    assert int(hexa_mesh.element_kind.min()) > int(SubShapeKind.NONE)
    assert set(back.mesh.element_kind.tolist()) == {int(SubShapeKind.NONE)}
    assert set(back.mesh.element_ordinal.tolist()) == {0}
    assert set(back.mesh.node_kind.tolist()) == {int(SubShapeKind.NONE)}


def test_ids_are_reassigned_from_position_rather_than_carried(tmp_path: Path) -> None:
    """The format numbers per type from 1, so an id is not a durable name across a file.

    Shown on a hand-built mesh with deliberately sparse ids, because a freshly computed mesh
    numbers its own entities 1..n in the same order and would come back looking preserved —
    see the next test. A consumer keying anything on a mesh id across a file must not rely on
    that coincidence.
    """
    mesh = MeshData(
        node_coords=np.array(
            [[0.0, 0.0, 0.0], [BOX_DX, 0.0, 0.0], [0.0, BOX_DY, 0.0], [BOX_DX, BOX_DY, 0.0]]
        ),
        node_id=np.array([10, 11, 12, 13], dtype=np.int64),
        node_kind=np.zeros(4, dtype=np.int8),
        node_ordinal=np.zeros(4, dtype=np.int32),
        element_offsets=np.array([0, 3, 6], dtype=np.int64),
        element_nodes=np.array([0, 1, 2, 1, 3, 2], dtype=np.int32),
        element_type=np.full(2, int(ElementType.TRIANGLE), dtype=np.int8),
        element_id=np.array([100, 200], dtype=np.int64),
        element_kind=np.zeros(2, dtype=np.int8),
        element_ordinal=np.zeros(2, dtype=np.int32),
        face_offsets=np.zeros(3, dtype=np.int64),
        face_sizes=np.array([], dtype=np.int32),
    )
    path = tmp_path / "sparse.meshb"

    write_gmf(path, mesh)
    back = read_gmf(path)

    assert back.mesh.node_id.tolist() == [1, 2, 3, 4]
    assert back.mesh.element_id.tolist() == [1, 2]
    # The mesh itself is intact; only the names changed.
    assert np.allclose(
        np.sort(back.mesh.node_coords, axis=0), np.sort(mesh.node_coords, axis=0)
    )


def test_a_mesh_built_from_arrays_writes_as_readily_as_one_from_the_mesher(
    tmp_path: Path,
) -> None:
    """The interchange is array-in / array-out, so a caller's own mesh can be written."""
    mesh = MeshData(
        node_coords=np.array(
            [[0.0, 0.0, 0.0], [BOX_DX, 0.0, 0.0], [0.0, BOX_DY, 0.0], [0.0, 0.0, BOX_DZ]]
        ),
        node_id=np.arange(1, 5, dtype=np.int64),
        node_kind=np.zeros(4, dtype=np.int8),
        node_ordinal=np.zeros(4, dtype=np.int32),
        element_offsets=np.array([0, 4], dtype=np.int64),
        element_nodes=np.array([0, 1, 2, 3], dtype=np.int32),
        element_type=np.array([int(ElementType.TETRAHEDRON)], dtype=np.int8),
        element_id=np.array([1], dtype=np.int64),
        element_kind=np.zeros(1, dtype=np.int8),
        element_ordinal=np.zeros(1, dtype=np.int32),
        face_offsets=np.zeros(2, dtype=np.int64),
        face_sizes=np.array([], dtype=np.int32),
    )
    path = tmp_path / "one_tetra.meshb"

    write_gmf(path, mesh)
    back = read_gmf(path)

    assert back.mesh.count_of(ElementType.TETRAHEDRON) == 1
    assert back.mesh.node_count == 4


def test_ids_of_a_freshly_computed_mesh_happen_to_come_back_unchanged(
    hexa_mesh: MeshData, tmp_path: Path
) -> None:
    """Recorded as a coincidence, not a guarantee — the previous test is the mechanism."""
    path = tmp_path / "ids.meshb"

    write_gmf(path, hexa_mesh)
    back = read_gmf(path)

    assert set(back.mesh.element_id.tolist()) == set(hexa_mesh.element_id.tolist())
    assert back.mesh.element_count == hexa_mesh.element_count


# ---- Failure modes ------------------------------------------------------------------------ #


def test_reading_a_file_that_is_not_there_raises_naming_it(tmp_path: Path) -> None:
    missing = tmp_path / "absent.meshb"

    with pytest.raises(PysmeshError, match="absent.meshb"):
        read_gmf(missing)


def test_reading_a_file_that_is_not_a_mesh_raises(tmp_path: Path) -> None:
    junk = tmp_path / "junk.meshb"
    junk.write_bytes(b"this is not an Inria mesh file")

    with pytest.raises(PysmeshError):
        read_gmf(junk)


def test_writing_to_an_extension_the_format_does_not_know_raises(
    hexa_mesh: MeshData, tmp_path: Path
) -> None:
    with pytest.raises(PysmeshError, match="extension|writing"):
        write_gmf(tmp_path / "box.vtu", hexa_mesh)


def test_a_mesh_whose_arrays_disagree_is_refused(hexa_mesh: MeshData, tmp_path: Path) -> None:
    """The arrays are a contract; a truncated one must not reach the driver."""
    broken = MeshData(
        node_coords=hexa_mesh.node_coords,
        node_id=hexa_mesh.node_id,
        node_kind=hexa_mesh.node_kind,
        node_ordinal=hexa_mesh.node_ordinal,
        element_offsets=hexa_mesh.element_offsets[:-1],
        element_nodes=hexa_mesh.element_nodes,
        element_type=hexa_mesh.element_type,
        element_id=hexa_mesh.element_id,
        element_kind=hexa_mesh.element_kind,
        element_ordinal=hexa_mesh.element_ordinal,
        face_offsets=hexa_mesh.face_offsets,
        face_sizes=hexa_mesh.face_sizes,
    )

    with pytest.raises(PysmeshError, match="disagree"):
        write_gmf(tmp_path / "broken.meshb", broken)


def test_writing_from_the_mesher_is_the_same_as_writing_its_mesh(tmp_path: Path) -> None:
    """The convenience path must not be a second implementation."""
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.compute()

        mesher.write_gmf(tmp_path / "from_mesher.meshb")
        write_gmf(tmp_path / "from_mesh.meshb", mesher.mesh())

    assert (tmp_path / "from_mesher.meshb").read_bytes() == (
        tmp_path / "from_mesh.meshb"
    ).read_bytes()


# ---- Files written by another engine ------------------------------------------------ #


# MMG3D 5.8.0 wrote these files with its own writer: the 3 x 7 x 11 box, remeshed to
# hmax = 1.5 (fixtures/engine_gmf/generate_mmg.py, dev-only, needs mmgpy). Beside
# vertices, tetrahedra, triangles and edges, MMG writes Corners, RequiredVertices,
# Normals, Tangents and Ridges, so the reader meets the keywords of a real engine file.
ENGINE_DIR: Path = Path(__file__).resolve().parent / "fixtures" / "engine_gmf"
MMG_FILES: tuple[str, ...] = ("mmg3d_box.mesh", "mmg3d_box.meshb")


def _cells(mesh: MeshData, element_type: ElementType, size: int) -> NDArray[np.float64]:
    """(k, size, 3) node coordinates of every element of one fixed-size type."""
    chosen = np.flatnonzero(mesh.element_type == int(element_type))
    starts = mesh.element_offsets[chosen]
    nodes = mesh.element_nodes[starts[:, None] + np.arange(size)]
    return mesh.node_coords[nodes]


@pytest.mark.parametrize("name", MMG_FILES)
def test_an_mmg_file_reads_back_as_the_box_it_meshes(name: str) -> None:
    """The tetrahedra fill the box and the triangles cover its boundary.

    The box faces are planar, so remeshing keeps them: the volume is
    3 * 7 * 11 = 231 and the boundary area is 2 * (3 * 7 + 3 * 11 + 7 * 11) = 262. The
    eight required vertices are the box corners.

    Every tetrahedron must come out in SMESH's orientation. SMESH's order is Inria's
    with the second and third nodes swapped (``DriverGMF_Read`` adds
    ``iN[0], iN[2], iN[1], iN[3]``), so ``det(n1 - n0, n2 - n0, n3 - n0)`` is negative
    for every valid tetrahedron of the read mesh, and its volume is minus a sixth of it.
    """
    box = np.array([BOX_DX, BOX_DY, BOX_DZ])

    result = read_gmf(ENGINE_DIR / name)

    mesh = result.mesh
    assert isinstance(result, GmfMesh)
    tets = _cells(mesh, ElementType.TETRAHEDRON, 4)
    volumes = -np.linalg.det(tets[:, 1:] - tets[:, :1]) / 6.0
    assert tets.shape[0] > 0
    assert float(volumes.min()) > 0.0
    assert float(volumes.sum()) == pytest.approx(float(np.prod(box)), rel=1e-12)
    tris = _cells(mesh, ElementType.TRIANGLE, 3)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = 0.5 * float(np.linalg.norm(normals, axis=1).sum())
    expected_area = 2.0 * (BOX_DX * BOX_DY + BOX_DX * BOX_DZ + BOX_DY * BOX_DZ)
    assert area == pytest.approx(expected_area, rel=1e-12)
    # MMG computes the new boundary points itself: they lie on the faces to round-off.
    slack = 1e-12 * BOX_DZ
    assert float(mesh.node_coords.min()) >= -slack
    assert np.all(mesh.node_coords.max(axis=0) <= box + slack)
    (required,) = (g for g in result.groups if g.dimension == ElementDimension.NODE)
    corners = mesh.node_coords[np.isin(mesh.node_id, required.element_ids)]
    expected_corners = {
        (BOX_DX * (i & 1), BOX_DY * ((i >> 1) & 1), BOX_DZ * ((i >> 2) & 1))
        for i in range(8)
    }
    assert {tuple(map(float, c)) for c in corners} == expected_corners


def test_the_ascii_and_binary_mmg_files_hold_one_mesh() -> None:
    """MMG wrote one mesh twice; the two files read back to the same mesh.

    The ASCII file prints coordinates in decimal, so they agree to round-off, not bit
    for bit.
    """
    ascii_path, binary_path = (ENGINE_DIR / name for name in MMG_FILES)

    ascii_mesh = read_gmf(ascii_path).mesh
    binary_mesh = read_gmf(binary_path).mesh

    assert ascii_mesh.node_count == binary_mesh.node_count
    np.testing.assert_allclose(
        ascii_mesh.node_coords, binary_mesh.node_coords, rtol=0.0, atol=1e-12
    )
    np.testing.assert_array_equal(ascii_mesh.element_type, binary_mesh.element_type)
    np.testing.assert_array_equal(
        ascii_mesh.element_offsets, binary_mesh.element_offsets
    )
    np.testing.assert_array_equal(ascii_mesh.element_nodes, binary_mesh.element_nodes)
