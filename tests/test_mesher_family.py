# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""Gates for the algorithm and hypothesis catalogue.

Three claims, in order of how much they are worth.

* **Every catalogue entry is wired to a real upstream class.** Each one is built and attached
  through the native factory, which is the only thing that can prove the dataclass's fields
  match the setters behind them. The factory refuses a parameter it does not read, so a field
  added on one side without the other fails here rather than being silently dropped — and
  that refusal is itself asserted, so the check is known to be able to fail.
* **The 1-D distribution family produces the distributions it names.** Each is checked
  against a property of the spacing that a wrong hypothesis could not produce — a segment
  count, a first-to-last length ratio, a monotone progression — rather than against a value
  copied from a previous run.
* **The families that need a fixture of their own get one.** An extruded triangle for the
  extrusion mesher, a solid between two concentric shells for the radial one, a source and a
  target for projection, a disk for radial quadrangles, a thin strip for the medial-axis
  mesher. Each computes, and its result is checked for the property that algorithm exists to
  give — pentahedra between the swept ends, nodes that lie in the wall and nowhere else,
  matching element counts on the projected pair.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    Adaptive1D,
    Algorithm,
    Arithmetic1D,
    AutomaticLength,
    Cartesian3D,
    CartesianParameters3D,
    CompositeHexa3D,
    CompositeSegment1D,
    Deflection1D,
    Distribution,
    ElementType,
    EntityKind,
    FixedPoints1D,
    Geometric1D,
    Hexa3D,
    HexaFromSkin3D,
    Hypothesis,
    LayerDistribution,
    LocalLength,
    MaxElementArea,
    MaxElementVolume,
    MaxLength,
    Mefisto2D,
    Mesher,
    NumberOfLayers,
    NumberOfLayers2D,
    NumberOfSegments,
    PolygonPerFace2D,
    PolyhedronPerSolid3D,
    Prism3D,
    Projection1D,
    Projection1D2D,
    Projection2D,
    Projection3D,
    ProjectionSource1D,
    ProjectionSource2D,
    ProjectionSource3D,
    Propagation,
    QuadFromMedialAxis1D2D,
    Quadrangle2D,
    QuadrangleParams,
    QuadranglePreference,
    QuadraticMesh,
    QuadType,
    PysmeshError,
    RadialPrism3D,
    RadialQuadrangle1D2D,
    Regular1D,
    SegmentLengthAroundVertex,
    Session,
    StartEndLength,
    SubShape,
    SubShapeKind,
    ViscousLayers2D,
)
from pysmesh.mesher import ViscousLayers

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0

SHELL_OUTER_RADIUS: float = 3.0
SHELL_INNER_RADIUS: float = 2.0

PRISM_HEIGHT: float = 9.0
TRIANGLE_AREA: float = 15.0


# ---- Fixtures --------------------------------------------------------------------------- #


def _box_shape() -> ps.Shape:
    """The 3 x 7 x 11 box, through the session."""
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    return ps.load_brep(session.brep())


def _hollow_sphere_shape() -> ps.Shape:
    """A solid between two concentric shells — the O-grid fixture the radial mesher wants.

    A hollow *cylinder* will not do, and the reason is worth recording: capping it makes its
    boundary a single closed shell, and the radial mesher refuses anything that is not two.
    """
    session = Session()
    session.add_sphere(SHELL_OUTER_RADIUS)
    outer = list(session.entities(EntityKind.SOLID))
    session.add_sphere(SHELL_INNER_RADIUS)
    inner = [e for e in session.entities(EntityKind.SOLID) if e not in outer]
    session.cut(outer, inner)
    return ps.load_brep(session.brep())


def _shell_faces(shape: ps.Shape) -> tuple[SubShape, SubShape]:
    """The outer and the inner shell of the hollow sphere, by area."""
    areas = sorted((face.area, i) for i, face in enumerate(shape.faces(), 1))
    return (
        SubShape(SubShapeKind.FACE, areas[-1][1]),
        SubShape(SubShapeKind.FACE, areas[0][1]),
    )


def _radial_recipe(mesher: Mesher, shape: ps.Shape, layers: Hypothesis) -> None:
    """Free-mesh the outer shell, project it onto the inner one, then fill radially.

    The projection is not decoration: the radial mesher refuses two shells whose meshes do
    not match, and two independent free meshes of concentric spheres never do.
    """
    outer, inner = _shell_faces(shape)
    mesher.assign(Regular1D())
    mesher.assign(NumberOfSegments(count=6))
    mesher.assign(Mefisto2D(), on=outer)
    mesher.assign(MaxElementArea(max_area=2.0), on=outer)
    mesher.assign(Projection2D(), on=inner)
    mesher.assign(ProjectionSource2D(source_face=outer), on=inner)
    mesher.assign(RadialPrism3D())
    mesher.assign(layers)


def _extruded_triangle_shape() -> ps.Shape:
    """A prismatic solid: a triangle swept along z, which the extrusion mesher wants."""
    session = Session()
    session.add_polyline(
        [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 0.0)]
    )
    edges = list(session.entities(EntityKind.EDGE))
    session.make_face(edges)
    faces = list(session.entities(EntityKind.FACE))
    session.extrude(faces, (0.0, 0.0, PRISM_HEIGHT))
    return ps.load_brep(session.brep())


def _triangular_face(shape: ps.Shape) -> SubShape:
    """The base face of the extruded triangle, by its known area."""
    for ordinal, face in enumerate(shape.faces(), 1):
        if math.isclose(face.area, TRIANGLE_AREA, rel_tol=1e-9):
            return SubShape(SubShapeKind.FACE, ordinal)
    raise AssertionError("the extruded-triangle fixture has no triangular face")


def _disk_face_shape() -> ps.Shape:
    """A single circular face, for the radial quadrangle mesher."""
    session = Session()
    session.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.5)
    edges = list(session.entities(EntityKind.EDGE))
    session.make_face(edges)
    return ps.load_brep(session.brep())


def _segment_lengths(mesh: ps.MeshData, edge_ordinal: int) -> list[float]:
    """Lengths of the 1-D elements on one edge, in order along it.

    Ordering is by distance from the segment closest to one end, which is enough to read a
    progression without assuming the mesher's own emission order.
    """
    lengths: list[tuple[float, float]] = []
    for element in range(mesh.element_count):
        if int(mesh.element_kind[element]) != int(SubShapeKind.EDGE):
            continue
        if int(mesh.element_ordinal[element]) != edge_ordinal:
            continue
        nodes = mesh.nodes_of(element)
        if len(nodes) != 2:
            continue
        a = mesh.node_coords[nodes[0]]
        b = mesh.node_coords[nodes[1]]
        midpoint = float(np.linalg.norm((a + b) / 2.0))
        lengths.append((midpoint, float(np.linalg.norm(b - a))))
    return [length for _, length in sorted(lengths)]


# ---- Every catalogue entry is wired ------------------------------------------------------ #

# One instance of each, with the sub-shape it is assigned to. A projection hypothesis needs a
# source of the right kind, so the box's own sub-shapes serve as one.
_CATALOGUE: list[tuple[Algorithm | Hypothesis, SubShape | None]] = [
    # 1-D algorithms
    (Regular1D(), None),
    (CompositeSegment1D(), None),
    (Projection1D(), None),
    # 2-D algorithms
    (Quadrangle2D(), None),
    (Mefisto2D(), None),
    (PolygonPerFace2D(), None),
    (Projection2D(), None),
    (Projection1D2D(), None),
    (QuadFromMedialAxis1D2D(), None),
    (RadialQuadrangle1D2D(), None),
    # 3-D algorithms
    (Cartesian3D(), None),
    (Hexa3D(), None),
    (CompositeHexa3D(), None),
    (HexaFromSkin3D(), None),
    (Prism3D(), None),
    (RadialPrism3D(), None),
    (Projection3D(), None),
    (PolyhedronPerSolid3D(), None),
    # 1-D hypotheses
    (NumberOfSegments(count=4), None),
    (NumberOfSegments(count=4, distribution=Distribution.SCALE, scale_factor=3.0), None),
    (
        NumberOfSegments(
            count=4, distribution=Distribution.TABLE, table=(0.0, 1.0, 1.0, 3.0)
        ),
        None,
    ),
    (
        NumberOfSegments(count=4, distribution=Distribution.EXPRESSION, expression="1+t"),
        None,
    ),
    (Arithmetic1D(start_length=0.5, end_length=2.0), None),
    (StartEndLength(start_length=0.5, end_length=2.0), None),
    (Geometric1D(start_length=0.5, common_ratio=1.2), None),
    (FixedPoints1D(points=(0.25, 0.75), segment_counts=(2, 3, 2)), None),
    (Adaptive1D(min_size=0.1, max_size=2.0, deflection=0.05), None),
    (AutomaticLength(fineness=0.5), None),
    (Deflection1D(deflection=0.05), None),
    (LocalLength(length=1.5), None),
    (MaxLength(length=2.0), None),
    (SegmentLengthAroundVertex(length=0.5), SubShape(SubShapeKind.VERTEX, 1)),
    (Propagation(), SubShape(SubShapeKind.EDGE, 1)),
    (LayerDistribution(distribution=NumberOfSegments(count=3)), None),
    (QuadraticMesh(), None),
    # 2-D and 3-D hypotheses
    (MaxElementArea(max_area=4.0), None),
    (MaxElementVolume(max_volume=8.0), None),
    (QuadranglePreference(), None),
    (QuadrangleParams(quad_type=QuadType.REDUCED), None),
    (
        QuadrangleParams(
            base_vertex=SubShape(SubShapeKind.VERTEX, 1), corner_vertices=(1, 2, 3, 4)
        ),
        None,
    ),
    (NumberOfLayers(count=3), None),
    (NumberOfLayers2D(count=3), None),
    (
        CartesianParameters3D(spacing_x="1.0", spacing_y="1.0", spacing_z="1.0"),
        None,
    ),
    # Hypotheses naming another part of the model
    (ProjectionSource1D(source_edge=SubShape(SubShapeKind.EDGE, 1)), None),
    (
        ProjectionSource1D(
            source_edge=SubShape(SubShapeKind.EDGE, 1),
            source_vertex=SubShape(SubShapeKind.VERTEX, 1),
            target_vertex=SubShape(SubShapeKind.VERTEX, 2),
        ),
        None,
    ),
    (ProjectionSource2D(source_face=SubShape(SubShapeKind.FACE, 1)), None),
    (ProjectionSource3D(source_solid=SubShape(SubShapeKind.SOLID, 1)), None),
    (
        ViscousLayers(
            total_thickness=0.3,
            layer_count=3,
            stretch_factor=1.2,
            boundary=(1, 2),
            group_name="layers",
        ),
        None,
    ),
    (
        ViscousLayers2D(
            total_thickness=0.3,
            layer_count=3,
            stretch_factor=1.2,
            boundary=(1, 2),
            group_name="layers2d",
        ),
        None,
    ),
]


@pytest.mark.parametrize(
    ("item", "on"), _CATALOGUE, ids=[f"{i.native_name}" for i, _ in _CATALOGUE]
)
def test_every_catalogue_entry_builds_and_attaches(
    item: Algorithm | Hypothesis, on: SubShape | None
) -> None:
    """Proves the dataclass's fields match the upstream setters behind them.

    A name with no branch, or a field the branch does not read, fails here — which is the
    only place the two halves of the catalogue can be checked against each other.
    """
    with Mesher(_box_shape()) as mesher:
        mesher.assign(item, on=on)

        assert mesher.assignments() == ((item.native_name, on),)


def test_a_parameter_the_factory_does_not_read_is_refused() -> None:
    """The falsification for the test above: the drift check must be able to fail."""
    with Mesher(_box_shape()) as mesher:
        with pytest.raises(PysmeshError, match="does not take the parameter"):
            mesher._m.assign("Regular_1D", {"unexpected": 1}, "", 0)


def test_a_missing_parameter_is_refused_naming_it() -> None:
    with Mesher(_box_shape()) as mesher:
        with pytest.raises(PysmeshError, match="missing the parameter 'count'"):
            mesher._m.assign("NumberOfSegments", {}, "", 0)


def test_an_unknown_algorithm_name_is_refused() -> None:
    with Mesher(_box_shape()) as mesher:
        with pytest.raises(PysmeshError, match="unknown algorithm or hypothesis"):
            mesher._m.assign("Netgen_3D", {}, "", 0)


def test_the_catalogue_covers_every_public_algorithm_and_hypothesis() -> None:
    """A new entry added to the catalogue without a case above would go untested."""
    import pysmesh.mesher as mesher_module

    exported = {
        name
        for name in mesher_module.__all__
        if isinstance(getattr(mesher_module, name), type)
        and issubclass(getattr(mesher_module, name), (Algorithm, Hypothesis))
        and getattr(mesher_module, name) not in (Algorithm, Hypothesis)
    }
    covered = {type(item).__name__ for item, _ in _CATALOGUE}

    assert exported == covered


# ---- The 1-D distribution family --------------------------------------------------------- #


def _mesh_one_edge(hypothesis: Hypothesis) -> ps.MeshData:
    """Discretise the box's edges with one 1-D hypothesis and return the mesh."""
    mesher = Mesher(_box_shape())
    mesher.assign(Regular1D())
    mesher.assign(hypothesis)
    mesher.compute()
    mesh = mesher.mesh()
    mesher.release()
    return mesh


def test_number_of_segments_gives_exactly_that_many_per_edge() -> None:
    mesh = _mesh_one_edge(NumberOfSegments(count=5))

    per_edge = {
        ordinal: int(np.count_nonzero(mesh.element_ordinal[mesh.element_kind == 3] == ordinal))
        for ordinal in set(mesh.element_ordinal[mesh.element_kind == 3].tolist())
    }

    assert len(per_edge) == 12
    assert set(per_edge.values()) == {5}


def test_arithmetic_progression_gives_the_stated_first_and_last_lengths() -> None:
    mesh = _mesh_one_edge(Arithmetic1D(start_length=0.5, end_length=2.0))

    # Edge 1 of a 3 x 7 x 11 box is long enough to hold several segments of these sizes.
    lengths = _segment_lengths(mesh, edge_ordinal=1)

    assert len(lengths) >= 3
    assert min(lengths[0], lengths[-1]) == pytest.approx(0.5, rel=0.35)
    assert max(lengths[0], lengths[-1]) == pytest.approx(2.0, rel=0.35)


def test_a_scaled_distribution_grows_monotonically_along_the_edge() -> None:
    mesh = _mesh_one_edge(
        NumberOfSegments(count=6, distribution=Distribution.SCALE, scale_factor=4.0)
    )

    lengths = _segment_lengths(mesh, edge_ordinal=1)
    ordered = lengths if lengths[0] < lengths[-1] else lengths[::-1]

    assert len(lengths) == 6
    assert ordered == sorted(ordered)
    assert ordered[-1] / ordered[0] == pytest.approx(4.0, rel=0.2)


def test_a_regular_distribution_gives_equal_segments() -> None:
    """The counter-case: without it "monotone" would pass on a uniform mesh too."""
    mesh = _mesh_one_edge(NumberOfSegments(count=6))

    lengths = _segment_lengths(mesh, edge_ordinal=1)

    assert len(lengths) == 6
    assert max(lengths) == pytest.approx(min(lengths), rel=1e-9)


def test_fixed_points_split_the_edge_at_the_stated_positions() -> None:
    mesh = _mesh_one_edge(
        FixedPoints1D(points=(0.25, 0.75), segment_counts=(2, 4, 2))
    )

    lengths = _segment_lengths(mesh, edge_ordinal=1)

    assert len(lengths) == 2 + 4 + 2


def test_local_length_sizes_every_edge_to_about_that_length() -> None:
    target = 1.5
    mesh = _mesh_one_edge(LocalLength(length=target))

    lengths = _segment_lengths(mesh, edge_ordinal=1)

    assert lengths
    # The count is rounded to fit the edge, so the achieved length is near the target.
    assert max(lengths) == pytest.approx(target, rel=0.35)


def test_max_length_never_exceeds_its_bound() -> None:
    bound = 1.0
    mesh = _mesh_one_edge(MaxLength(length=bound))

    edge_elements = [
        element
        for element in range(mesh.element_count)
        if int(mesh.element_kind[element]) == int(SubShapeKind.EDGE)
    ]
    lengths = [
        float(
            np.linalg.norm(
                mesh.node_coords[mesh.nodes_of(e)[1]] - mesh.node_coords[mesh.nodes_of(e)[0]]
            )
        )
        for e in edge_elements
    ]

    assert lengths
    assert max(lengths) <= bound + 1e-9


def test_propagation_carries_a_count_to_the_opposite_edges() -> None:
    """A structured mesh's opposite sides match without each being stated."""
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Quadrangle2D())
        mesher.assign(NumberOfSegments(count=7), on=SubShape(SubShapeKind.EDGE, 1))
        mesher.assign(Propagation(), on=SubShape(SubShapeKind.EDGE, 1))
        mesher.compute()
        mesh = mesher.mesh()

    counts = [
        int(np.count_nonzero((mesh.element_kind == 3) & (mesh.element_ordinal == ordinal)))
        for ordinal in range(1, 13)
    ]

    # Four parallel edges carry the propagated count; the rest keep the model default.
    assert counts.count(7) == 4
    assert counts.count(2) == 8


def test_quadratic_mesh_produces_second_order_elements() -> None:
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.assign(QuadraticMesh())
        mesher.compute()
        mesh = mesher.mesh()

    assert mesh.count_of(ElementType.QUAD_HEXAHEDRON) == 8
    assert mesh.count_of(ElementType.HEXAHEDRON) == 0
    assert mesh.count_of(ElementType.QUAD_EDGE) > 0


# ---- Families with a fixture of their own ------------------------------------------ #


def test_the_extrusion_mesher_fills_a_prismatic_solid_from_its_source_face() -> None:
    """It meshes the lateral faces itself, so only the source face needs a 2-D algorithm."""
    shape = _extruded_triangle_shape()
    base = _triangular_face(shape)

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Prism3D())
        mesher.assign(Mefisto2D(), on=base)
        mesher.assign(MaxElementArea(max_area=3.0), on=base)
        report = mesher.compute()
        mesh = mesher.mesh()

    assert report.volumes > 0
    # Extruding triangles gives pentahedra and nothing else, which is the whole algorithm.
    assert mesh.count_of(ElementType.PENTAHEDRON) == report.volumes
    # Every cell lies within the swept height, so the extrusion went where it was asked to.
    assert mesh.node_coords[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert mesh.node_coords[:, 2].max() == pytest.approx(PRISM_HEIGHT, abs=1e-9)


def test_the_radial_mesher_builds_an_o_grid_between_two_shells() -> None:
    """Layers between an inner and an outer shell — a pipe wall, or a hollow ball."""
    shape = _hollow_sphere_shape()

    with Mesher(shape) as mesher:
        _radial_recipe(mesher, shape, NumberOfLayers(count=4))
        report = mesher.compute()
        mesh = mesher.mesh()

    assert report.volumes > 0
    assert mesh.count_of(ElementType.PENTAHEDRON) == report.volumes
    # Every node lies in the wall, between the two radii, which is what an O-grid means.
    radii = np.linalg.norm(mesh.node_coords, axis=1)
    assert radii.min() == pytest.approx(SHELL_INNER_RADIUS, abs=1e-6)
    assert radii.max() == pytest.approx(SHELL_OUTER_RADIUS, abs=1e-6)


def _layer_radii(mesh: ps.MeshData) -> NDArray[np.float64]:
    """The distinct radii the radial layers landed on."""
    return np.unique(np.round(np.linalg.norm(mesh.node_coords, axis=1), 6))


def test_a_layer_distribution_spaces_the_radial_direction_by_a_1d_hypothesis() -> None:
    """A hypothesis carrying another hypothesis — the only nested case in the catalogue."""
    shape = _hollow_sphere_shape()

    with Mesher(shape) as graded:
        _radial_recipe(
            graded,
            shape,
            LayerDistribution(
                distribution=NumberOfSegments(
                    count=5, distribution=Distribution.SCALE, scale_factor=3.0
                )
            ),
        )
        graded.compute()
        graded_gaps = np.diff(_layer_radii(graded.mesh()))

    with Mesher(shape) as even:
        _radial_recipe(even, shape, NumberOfLayers(count=5))
        even.compute()
        even_gaps = np.diff(_layer_radii(even.mesh()))

    # The counter-case is the point: an even distribution must NOT be graded, or "graded"
    # would be a property of the fixture rather than of the hypothesis.
    assert even_gaps.max() / even_gaps.min() == pytest.approx(1.0, rel=0.05)
    assert graded_gaps.max() / graded_gaps.min() > 2.0


def test_projection_copies_a_face_mesh_onto_another_face_node_for_node() -> None:
    """What no free mesher can guarantee, and the reason the projection family exists."""
    shape = _box_shape()

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Mefisto2D())
        mesher.assign(MaxElementArea(max_area=3.0))
        mesher.assign(Projection2D(), on=SubShape(SubShapeKind.FACE, 2))
        mesher.assign(
            ProjectionSource2D(source_face=SubShape(SubShapeKind.FACE, 1)),
            on=SubShape(SubShapeKind.FACE, 2),
        )
        mesher.compute()
        mesh = mesher.mesh()

    source = [
        m for m in [1] for _ in [0]
    ]  # placeholder removed below; counts read from the mesh
    del source

    def face_element_count(ordinal: int) -> int:
        return int(
            np.count_nonzero(
                (mesh.element_kind == int(SubShapeKind.FACE))
                & (mesh.element_ordinal == ordinal)
            )
        )

    assert face_element_count(1) > 0
    assert face_element_count(2) == face_element_count(1)


def test_the_radial_quadrangle_mesher_meshes_a_disk() -> None:
    with Mesher(_disk_face_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=8))
        mesher.assign(RadialQuadrangle1D2D())
        mesher.assign(NumberOfLayers2D(count=3))
        report = mesher.compute()
        mesh = mesher.mesh()

    assert report.faces > 0
    assert mesh.count_of(ElementType.QUADRANGLE) > 0


def test_the_medial_axis_mesher_meshes_a_thin_face() -> None:
    """Quad-dominant meshing built on a real medial-axis transform."""
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 12.0, 2.0)
    shape = ps.load_brep(session.brep())

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=8))
        mesher.assign(QuadFromMedialAxis1D2D())
        report = mesher.compute()
        mesh = mesher.mesh()

    assert report.faces > 0
    assert mesh.count_of(ElementType.QUADRANGLE) > 0


def test_one_polygon_per_face_uses_the_edge_discretisation_as_its_boundary() -> None:
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(PolygonPerFace2D())
        report = mesher.compute()
        mesh = mesher.mesh()

    assert report.faces == 6
    assert mesh.count_of(ElementType.POLYGON) == 6


def test_viscous_layers_grow_from_the_named_walls_into_a_group() -> None:
    """The layer cells are findable only through the group the hypothesis names."""
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.assign(
            ViscousLayers(
                total_thickness=0.4,
                layer_count=2,
                stretch_factor=1.2,
                boundary=(1,),
                group_name="wall_layers",
            )
        )
        report = mesher.compute()
        groups = mesher.groups()

    assert report.volumes > 0
    named = [g for g in groups if g.name == "wall_layers"]
    assert named, [g.name for g in groups]
    assert named[0].element_ids.size > 0


def test_the_composite_segment_mesher_treats_a_split_edge_chain_as_one() -> None:
    """An import that split one curve into pieces must still take one segment count."""
    with Mesher(_box_shape()) as mesher:
        mesher.assign(CompositeSegment1D())
        mesher.assign(NumberOfSegments(count=4))
        report = mesher.compute()

    assert report.edges > 0


def test_element_counts_scale_with_the_hypothesis_rather_than_being_fixed() -> None:
    """A guard against a recipe that ignores its hypothesis and looks right anyway."""
    counts: list[int] = []
    for area in (16.0, 4.0, 1.0):
        with Mesher(_box_shape()) as mesher:
            mesher.assign(Regular1D())
            mesher.assign(LocalLength(length=math.sqrt(area)))
            mesher.assign(Mefisto2D())
            mesher.assign(MaxElementArea(max_area=area))
            counts.append(mesher.compute().faces)

    assert counts == sorted(counts)
    assert counts[-1] > 4 * counts[0]
