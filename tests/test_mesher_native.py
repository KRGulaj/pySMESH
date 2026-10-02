# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-03

"""Gates for the native catalogue entries added with SMESH 9.16.

These entries are reachable only through the native factory,
``Mesher._m.assign(name, params, kind, ordinal)``. A later phase writes their typed
Python wrappers. Each test drives one entry through that path and checks a property
that the upstream specification states, on a shape small enough to reason about by
hand:

* ``SegmentAroundVertex_0D`` gives the segment at its vertex the length that
  ``SegmentLengthAroundVertex`` names (SMESH ``segments_around_vertex_algo.rst``).
* ``PropagOfDistribution`` carries the *relative* node distribution to the opposite
  edge of a quadrangle face (SMESH ``additional_hypo.rst``).
* ``NumberOfSegments`` with the beta law (new in 9.16) puts the nodes at the closed
  form of the law, and refuses a ``beta`` that the law is undefined for (SMESH
  ``1d_meshing_hypo.rst``).
* ``LengthFromEdges`` sizes the MEFISTO triangles from the mean boundary segment
  (SMESH ``2d_meshing_hypo.rst``), and is MEFISTO's default.
* ``LayerDistribution2D`` spaces the rings of ``RadialQuadrangle_1D2D`` by a 1-D
  hypothesis, starting at the circle (SMESH ``radial_quadrangle_1D2D_algo.rst``).
* ``UseExisting_1D`` and ``UseExisting_2D`` make nothing and count as computed, so a
  script's elements stand as that sub-shape's mesh (SMESH
  ``define_mesh_by_script.rst``).
* ``BlockRenumber`` numbers a Hexa_3D block like a structured i, j, k grid (SMESH
  ``3d_meshing_hypo.rst``).
* ``NotConformAllowed`` is global only (SMESH ``SMESH_Mesh.cxx``).
* ``CartesianParameters3D`` quanta (new in 9.16) replaces boundary polyhedra by whole
  grid cells (SMESH ``cartesian_algo.rst``).
* ``Cartesian_3D`` with ``ViscousLayers`` (new in 9.16, reachable through the existing
  entries) grows the layers at their geometric closed form.

The upstream sources cited are those of SMESH ``V9_16_0``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    Arithmetic1D,
    Cartesian3D,
    CartesianParameters3D,
    Distribution,
    ElementType,
    Hexa3D,
    LayerDistribution,
    Mefisto2D,
    Mesher,
    NumberOfSegments,
    Propagation,
    PysmeshError,
    Quadrangle2D,
    RadialQuadrangle1D2D,
    Regular1D,
    SegmentLengthAroundVertex,
    Session,
    SubShape,
    SubShapeKind,
    ViscousLayers,
)

LINE_LENGTH: float = 10.0
SQUARE_SIDE: float = 4.0
DISK_RADIUS: float = 2.5
SPHERE_RADIUS: float = 2.0
BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0

# The trapezoid of the propagation tests: a 4-long bottom, a 2-long top, height 2.
TRAPEZOID: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0),
    (3.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
)

# NumberOfSegments::DistrType::DT_BetaLaw (StdMeshers_NumberOfSegments.hxx). The
# public Distribution enum has no member for it yet.
DT_BETA_LAW: int = 4


# ---- Fixtures --------------------------------------------------------------------- #


def _line_shape(length: float = LINE_LENGTH) -> ps.Shape:
    """One straight edge from the origin along +x."""
    session = Session()
    session.add_line((0.0, 0.0, 0.0), (length, 0.0, 0.0))
    return ps.load_brep(session.brep())


def _trapezoid_shape() -> ps.Shape:
    """A planar four-sided face whose opposite sides have different lengths."""
    session = Session()
    session.add_polyline(np.array(TRAPEZOID, dtype=np.float64), closed=True)
    session.make_face(list(session.entities(ps.EntityKind.EDGE)))
    return ps.load_brep(session.brep())


def _vertex_at(shape: ps.Shape, point: tuple[float, float, float]) -> int:
    """The ordinal of the vertex of ``shape`` at ``point``."""
    for vertex in shape.vertices():
        if np.allclose(vertex.xyz, point, atol=1e-9):
            return vertex.id
    raise AssertionError(f"no vertex at {point}")


def _edge_at_height(shape: ps.Shape, y: float) -> int:
    """The ordinal of the horizontal edge of ``shape`` at height ``y``.

    The bounding box is grown by the edge tolerance (1e-7), hence the 1e-6 test.
    """
    for edge in shape.edges():
        box = edge.bbox
        if abs(box[1] - y) < 1e-6 and abs(box[4] - y) < 1e-6:
            return edge.id
    raise AssertionError(f"no horizontal edge at y = {y}")


def _sorted_x(mesh: ps.MeshData) -> NDArray[np.float64]:
    """The x coordinate of every node, ascending."""
    return np.sort(np.ascontiguousarray(mesh.node_coords[:, 0], dtype=np.float64))


def _beta_law(beta: float, count: int, length: float) -> NDArray[np.float64]:
    """Node positions of the beta law on a straight edge, ends included.

    The law of SMESH 9.16 (``StdMeshers_Regular_1D::computeBetaLaw``, after gmsh):
    for ``i = 1 .. n-1``, ``x_i / L = 1 + b (1 - r**(1 - i/n)) / (1 + r**(1 - i/n))``
    with ``b = |beta|`` and ``r = (b + 1) / (b - 1)``. A negative ``beta`` mirrors
    the positions.
    """
    b = abs(beta)
    r = (b + 1.0) / (b - 1.0)
    eta = np.arange(1, count, dtype=np.float64) / count
    power = r ** (1.0 - eta)
    t = 1.0 + b * (1.0 - power) / (1.0 + power)
    if beta < 0:
        t = np.sort(1.0 - t)
    return np.concatenate(([0.0], t * length, [length]))


def _beta_params(beta: float, count: int) -> dict[str, object]:
    """NumberOfSegments parameters for the beta law, as the native factory reads."""
    params = NumberOfSegments(count=count).params()
    params["distribution"] = DT_BETA_LAW
    params["beta"] = beta
    return params


# ---- SegmentAroundVertex_0D ------------------------------------------------------- #


def test_segment_around_vertex_sets_the_length_at_its_vertex() -> None:
    """The segment at the vertex takes the length; the far end keeps its spacing.

    Spec (SMESH ``segments_around_vertex_algo.rst``): the edge is discretised by its
    1-D hypothesis, then the nodes near the vertex are moved to give the length that
    "Length Near Vertex" names. Five equal segments of a 10-long edge are 2.0 each.
    """
    shape = _line_shape()
    origin = _vertex_at(shape, (0.0, 0.0, 0.0))

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=5))
        mesher._m.assign("SegmentAroundVertex_0D", {}, "VERTEX", origin)
        mesher.assign(
            SegmentLengthAroundVertex(length=0.5),
            on=SubShape(SubShapeKind.VERTEX, origin),
        )
        mesher.compute()
        x = _sorted_x(mesher.mesh())

    assert x[1] - x[0] == pytest.approx(0.5, abs=1e-9)
    assert x[-1] - x[-2] == pytest.approx(2.0, abs=1e-9)
    assert x[0] == pytest.approx(0.0, abs=1e-12)
    assert x[-1] == pytest.approx(LINE_LENGTH, abs=1e-12)


def test_length_around_vertex_is_unused_without_its_0d_algorithm() -> None:
    """The falsification: without the 0-D algorithm the hypothesis is never read."""
    shape = _line_shape()
    origin = _vertex_at(shape, (0.0, 0.0, 0.0))

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=5))
        mesher.assign(
            SegmentLengthAroundVertex(length=0.5),
            on=SubShape(SubShapeKind.VERTEX, origin),
        )
        mesher.compute()
        x = _sorted_x(mesher.mesh())

    np.testing.assert_allclose(np.diff(x), np.full(5, 2.0), atol=1e-9)


# ---- PropagOfDistribution --------------------------------------------------------- #


def _top_and_bottom_fractions(
    propagation: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Node fractions along the 4-long bottom and the 2-long top of the trapezoid.

    The bottom carries Arithmetic1D(0.5 -> 1.5) and the propagation hypothesis named
    ``propagation``; every other edge has two equal segments.
    """
    shape = _trapezoid_shape()
    bottom = SubShape(SubShapeKind.EDGE, _edge_at_height(shape, 0.0))

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Arithmetic1D(start_length=0.5, end_length=1.5), on=bottom)
        if propagation == "Propagation":
            mesher.assign(Propagation(), on=bottom)
        else:
            mesher._m.assign(propagation, {}, "EDGE", bottom.ordinal)
        mesher.compute()
        xyz = mesher.mesh().node_coords

    on_bottom = np.sort(xyz[np.abs(xyz[:, 1]) < 1e-9, 0]) / 4.0
    on_top = np.sort(xyz[np.abs(xyz[:, 1] - 2.0) < 1e-9, 0] - 1.0) / 2.0
    return on_bottom, on_top


def test_propag_of_distribution_repeats_the_relative_spacing() -> None:
    """Same node count, same fractions of length, on an opposite edge half as long.

    Spec (SMESH ``additional_hypo.rst``, "Propagation of Node Distribution on
    Opposite Edges"): the opposite edge gets "the same number of nodes and the same
    relations between segment lengths". The propagation may run either way along the
    top edge, so the fractions are compared both ways.
    """
    bottom, top = _top_and_bottom_fractions("PropagOfDistribution")

    assert bottom.size == top.size
    assert bottom.size > 3
    deviation = min(
        float(np.abs(bottom - top).max()),
        float(np.abs(bottom - np.sort(1.0 - top)).max()),
    )
    assert deviation < 1e-9


def test_plain_propagation_carries_the_absolute_lengths_instead() -> None:
    """The falsification: Propagation copies Arithmetic1D's lengths, not fractions.

    On the 2-long top edge the absolute 0.5 -> 1.5 progression needs fewer segments
    than on the 4-long bottom, so the node counts differ.
    """
    bottom, top = _top_and_bottom_fractions("Propagation")

    assert top.size < bottom.size


# ---- NumberOfSegments: the beta law (new in 9.16) --------------------------------- #


@pytest.mark.parametrize("beta", [1.01, 1.1, 1.5, -1.05])
def test_the_beta_law_places_every_node_at_its_closed_form(beta: float) -> None:
    """Ten segments on a 10-long edge, against the law of SMESH 9.16.

    Reference: ``StdMeshers_Regular_1D::computeBetaLaw`` (SMESH ``V9_16_0``), the law
    of gmsh's ``meshGEdge.cpp``; documented in SMESH ``1d_meshing_hypo.rst``, "Beta
    Law Distribution". A straight edge has an exact arc-length parametrisation, so
    the positions hold to round-off.
    """
    with Mesher(_line_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher._m.assign("NumberOfSegments", _beta_params(beta, 10), "", 0)
        mesher.compute()
        x = _sorted_x(mesher.mesh())

    expected = _beta_law(beta, 10, LINE_LENGTH)
    np.testing.assert_allclose(x, expected, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("beta", [1.01, -1.01])
def test_the_sign_of_beta_chooses_where_the_nodes_crowd(beta: float) -> None:
    """Spec: a negative beta distributes the positions "in the opposite direction".

    With beta just above 1 the law crowds the nodes at the start of the edge; its
    negative crowds them at the end.
    """
    with Mesher(_line_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher._m.assign("NumberOfSegments", _beta_params(beta, 10), "", 0)
        mesher.compute()
        lengths = np.diff(_sorted_x(mesher.mesh()))

    first, last = float(lengths[0]), float(lengths[-1])
    assert (first < last) == (beta > 0)
    assert math.isclose(float(lengths.sum()), LINE_LENGTH, rel_tol=1e-12)


@pytest.mark.parametrize("beta", [1.0, 0.5, 0.0, -1.0])
def test_the_beta_law_refuses_a_beta_inside_the_unit_interval(beta: float) -> None:
    """Degenerate input: upstream documents ``[-1, 1]`` as forbidden for ``log(r)``.

    SMESH ``1d_meshing_hypo.rst``: "Values between [-1, 1] are forbidden to ensure
    validity of the log". ``SetBeta`` does not check it, so the factory must.
    """
    with (
        Mesher(_line_shape()) as mesher,
        pytest.raises(PysmeshError, match=r"needs \|beta\| > 1"),
    ):
        mesher._m.assign("NumberOfSegments", _beta_params(beta, 10), "", 0)


# ---- LengthFromEdges (MEFISTO_2D) ------------------------------------------------- #


def _square_shape() -> ps.Shape:
    """A 4 x 4 planar square face."""
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), SQUARE_SIDE, SQUARE_SIDE)
    return ps.load_brep(session.brep())


def _mefisto_square(segments: int, length_from_edges: bool) -> ps.MeshData:
    """MEFISTO_2D on the square, ``segments`` boundary segments per side."""
    with Mesher(_square_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=segments))
        mesher.assign(Mefisto2D())
        if length_from_edges:
            mesher._m.assign("LengthFromEdges", {}, "", 0)
        mesher.compute()
        return mesher.mesh()


def _triangles(mesh: ps.MeshData) -> NDArray[np.int32]:
    """The node rows of every triangle, shape (T, 3)."""
    rows = [
        mesh.nodes_of(i)
        for i in range(mesh.element_count)
        if int(mesh.element_type[i]) == int(ElementType.TRIANGLE)
    ]
    return np.asarray(rows, dtype=np.int32)


def test_length_from_edges_is_the_default_size_of_mefisto() -> None:
    """Assigning LengthFromEdges gives the mesh MEFISTO makes with no hypothesis.

    Upstream: ``StdMeshers_MEFISTO_2D::CheckHypothesis`` (V9_9_0, carried forward)
    states "can work with no hypothesis, LengthFromEdges is default one".
    """
    with_hypothesis = _mefisto_square(8, length_from_edges=True)
    without = _mefisto_square(8, length_from_edges=False)

    np.testing.assert_array_equal(
        np.sort(with_hypothesis.node_coords, axis=0),
        np.sort(without.node_coords, axis=0),
    )


@pytest.mark.parametrize("segments", [8, 16])
def test_length_from_edges_triangles_follow_the_boundary_spacing(segments: int) -> None:
    """The triangles fill the square, at the size of the mean boundary segment.

    Spec (SMESH ``2d_meshing_hypo.rst``): LengthFromEdges "defines the maximum linear
    size of mesh faces as an average length of mesh edges approximating the meshed
    face boundary". MEFISTO takes that length as its ideal edge length
    (``areteideale``), not as a hard bound, so the measured quantity is the mean
    triangle edge, held within a factor 1.5 of the boundary segment ``h``.
    """
    mesh = _mefisto_square(segments, length_from_edges=True)
    xyz = mesh.node_coords
    tri = _triangles(mesh)
    h = 4.0 * SQUARE_SIDE / (4 * segments)

    p0, p1, p2 = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum())
    edges = np.concatenate(
        [np.linalg.norm(a - b, axis=1) for a, b in ((p0, p1), (p1, p2), (p2, p0))]
    )
    assert area == pytest.approx(SQUARE_SIDE**2, abs=1e-9)
    assert np.all(np.abs(xyz[:, 2]) < 1e-12)
    assert h / 1.5 < float(edges.mean()) < 1.5 * h


# ---- LayerDistribution2D (RadialQuadrangle_1D2D) ---------------------------------- #


def _disk_shape() -> ps.Shape:
    """A disk of radius 2.5 in the z = 0 plane."""
    session = Session()
    session.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), DISK_RADIUS)
    session.make_face(list(session.entities(ps.EntityKind.EDGE)))
    return ps.load_brep(session.brep())


def _scale_positions(count: int, scale: float) -> NDArray[np.float64]:
    """Fractions of length of the SCALE law, start and end included.

    ``StdMeshers_Regular_1D`` (DT_Scale): ``s_i / L = (1 - a**i) / (1 - a**n)`` with
    ``a = scale ** (1 / (n - 1))``, so the last segment is ``scale`` times the first.
    """
    a = scale ** (1.0 / (count - 1))
    i = np.arange(count + 1, dtype=np.float64)
    return (1.0 - a**i) / (1.0 - a**count)


def test_layer_distribution_2d_spaces_the_rings_from_the_circle_inwards() -> None:
    """Ring radii follow a geometric 1-D law laid from the circle to the centre.

    Spec (SMESH ``radial_quadrangle_1D2D_algo.rst``): the distribution of layers "can
    be set with any 1D Hypothesis" and "is applied to the longest radial edge starting
    from its end lying on the elliptic curve". Four layers whose thickness grows three
    times from the circle to the centre put the rings at ``R (1 - s_i / L)``.
    """
    inner = NumberOfSegments(count=4, distribution=Distribution.SCALE, scale_factor=3.0)
    params = LayerDistribution(distribution=inner).params()

    with Mesher(_disk_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=8))
        mesher.assign(RadialQuadrangle1D2D())
        mesher._m.assign("LayerDistribution2D", params, "", 0)
        mesher.compute()
        xyz = mesher.mesh().node_coords

    radii = np.unique(np.round(np.hypot(xyz[:, 0], xyz[:, 1]), 9))
    expected = np.sort(DISK_RADIUS * (1.0 - _scale_positions(4, 3.0)))
    np.testing.assert_allclose(radii, expected, rtol=0.0, atol=1e-9)


# ---- UseExisting_1D / UseExisting_2D ---------------------------------------------- #


def _box_shape() -> ps.Shape:
    """The 3 x 7 x 11 box."""
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    return ps.load_brep(session.brep())


def test_use_existing_2d_makes_nothing_on_its_face_and_lets_the_rest_mesh() -> None:
    """The face under UseExisting_2D stays empty; the five others get 3 x 3 quads.

    Spec (SMESH ``define_mesh_by_script.rst``): "Use Faces to be Created Manually"
    lets a script create the 2-D mesh; the algorithm itself creates nothing, and the
    sub-mesh counts as computed (``StdMeshers_UseExisting_2D::Compute``).
    """
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Quadrangle2D())
        mesher._m.assign("UseExisting_2D", {}, "FACE", 1)
        mesher.compute()
        mesh = mesher.mesh()

    on_faces = mesh.element_kind == int(SubShapeKind.FACE)
    per_face = {
        ordinal: int(np.count_nonzero(on_faces & (mesh.element_ordinal == ordinal)))
        for ordinal in range(1, 7)
    }
    assert per_face == {1: 0, 2: 9, 3: 9, 4: 9, 5: 9, 6: 9}


def test_use_existing_1d_makes_no_segment() -> None:
    """Spec: "Use Edges to be Created Manually" creates no segment of its own."""
    with Mesher(_line_shape()) as mesher:
        mesher._m.assign("UseExisting_1D", {}, "", 0)
        mesher.compute()
        mesh = mesher.mesh()

    assert int(np.count_nonzero(mesh.element_type == int(ElementType.EDGE))) == 0


# ---- BlockRenumber (Hexa_3D) ------------------------------------------------------ #


def _hexa_box(renumber: bool) -> ps.MeshData:
    """Hexa_3D on the box, 3 segments per edge, with or without BlockRenumber."""
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        if renumber:
            mesher._m.assign("BlockRenumber", {}, "", 0)
        mesher.compute()
        return mesher.mesh()


def _grid_index(points: NDArray[np.float64], cells: int) -> NDArray[np.int64]:
    """Lexicographic ``i + n (j + n k)`` index of grid points of the box, i fastest."""
    step = np.array([BOX_DX, BOX_DY, BOX_DZ], dtype=np.float64) / cells
    ijk = np.rint(points / step).astype(np.int64)
    n = cells + 1
    return np.asarray(ijk[:, 0] + n * (ijk[:, 1] + n * ijk[:, 2]), dtype=np.int64)


def _node_order(mesh: ps.MeshData) -> NDArray[np.int64]:
    """Grid index of every node, in increasing node id."""
    order = np.argsort(mesh.node_id)
    return _grid_index(mesh.node_coords[order], 3)


def test_block_renumber_numbers_a_box_like_a_structured_grid() -> None:
    """Hexahedra and nodes come in i, j, k order from the corner at the origin.

    Spec (SMESH ``3d_meshing_hypo.rst``, "Renumber hypothesis"): it gives "hexahedra and
    nodes ordered like in a structured grid"; for a block with edges parallel to the
    global axes the block axes are the global ones, and the k axis runs from (0,0,0)
    along +z by default (SMESH ``doc/examples/filters_ex39.py``). i and j follow by the
    right-hand rule: x and y.
    """
    mesh = _hexa_box(renumber=True)

    hexa = np.flatnonzero(mesh.element_type == int(ElementType.HEXAHEDRON))
    hexa = hexa[np.argsort(mesh.element_id[hexa])]
    centroids = np.array(
        [mesh.node_coords[mesh.nodes_of(i)].mean(axis=0) for i in hexa]
    )
    step = np.array([BOX_DX, BOX_DY, BOX_DZ], dtype=np.float64) / 3
    ijk = np.floor(centroids / step).astype(np.int64)
    cell_index = ijk[:, 0] + 3 * (ijk[:, 1] + 3 * ijk[:, 2])
    np.testing.assert_array_equal(cell_index, np.arange(27))
    np.testing.assert_array_equal(_node_order(mesh), np.arange(64))


def test_without_block_renumber_the_nodes_are_not_in_grid_order() -> None:
    """The falsification: the plain Hexa_3D numbering is not the structured one."""
    mesh = _hexa_box(renumber=False)

    assert not np.array_equal(_node_order(mesh), np.arange(64))


# ---- NotConformAllowed ------------------------------------------------------------ #


def test_not_conform_allowed_is_refused_on_a_sub_shape() -> None:
    """Spec: the hypothesis "can be only global" (SMESH ``SMESH_Mesh.cxx:658-668``)."""
    with (
        Mesher(_box_shape()) as mesher,
        pytest.raises(PysmeshError, match="SMESH refused 'NotConformAllowed'"),
    ):
        mesher._m.assign("NotConformAllowed", {}, "FACE", 1)


def test_not_conform_allowed_leaves_a_conformal_mesh_unchanged() -> None:
    """Assigned globally it is accepted, and a mesh that is conformal anyway is kept."""
    with Mesher(_box_shape()) as mesher:
        mesher._m.assign("NotConformAllowed", {}, "", 0)
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.compute()
        allowed = mesher.mesh()

    plain = _hexa_box(renumber=False)
    assert allowed.count_of(ElementType.HEXAHEDRON) == 27
    np.testing.assert_array_equal(
        np.sort(allowed.node_coords, axis=0), np.sort(plain.node_coords, axis=0)
    )


# ---- CartesianParameters3D quanta (new in 9.16) ----------------------------------- #


def _sphere_shape() -> ps.Shape:
    """A sphere of radius 2."""
    session = Session()
    session.add_sphere(SPHERE_RADIUS)
    return ps.load_brep(session.brep())


def _cartesian_sphere(quanta: float | None) -> ps.MeshData:
    """Cartesian_3D on the sphere at spacing 0.5, with quanta when given."""
    params = CartesianParameters3D(
        spacing_x="0.5", spacing_y="0.5", spacing_z="0.5"
    ).params()
    if quanta is not None:
        params["use_quanta"] = True
        params["quanta"] = quanta
    with Mesher(_sphere_shape()) as mesher:
        mesher.assign(Cartesian3D())
        mesher._m.assign("CartesianParameters3D", params, "", 0)
        mesher.compute()
        return mesher.mesh()


def test_the_smallest_quanta_turns_every_cut_cell_into_one_hexahedron() -> None:
    """No polyhedron is left, and each one became exactly one hexahedron of its cell.

    Spec (SMESH ``cartesian_algo.rst``, "Set Quanta"): a boundary polyhedron is
    replaced by a hexahedron "if the volume of the polyhedron divided by the
    equivalent hexahedron is bigger than Quanta"; at quanta 1e-6 every cut cell
    qualifies. The replacement is built on the corners of the cell, a corner outside
    the body taken where its grid line meets the boundary
    (``StdMeshers_Cartesian_3D_Hexahedron.cxx:2974-2982``), so every hexahedron fits
    in one 0.5 cell of the grid: the sphere's 4-wide box holds 8 cells per axis.
    """
    plain = _cartesian_sphere(None)
    quantized = _cartesian_sphere(1e-6)

    polyhedra = plain.count_of(ElementType.POLYHEDRON)
    assert polyhedra > 0
    assert quantized.count_of(ElementType.POLYHEDRON) == 0
    assert quantized.count_of(ElementType.HEXAHEDRON) == (
        plain.count_of(ElementType.HEXAHEDRON) + polyhedra
    )
    hexa = np.flatnonzero(quantized.element_type == int(ElementType.HEXAHEDRON))
    extents = np.array(
        [np.ptp(quantized.node_coords[quantized.nodes_of(i)], axis=0) for i in hexa]
    )
    assert np.all(extents > 0.0)
    assert np.all(extents <= 0.5 + 1e-9)


def test_a_larger_quanta_keeps_fewer_boundary_cells() -> None:
    """The quanta is a lower bound on the kept volume fraction, so raising it can only
    drop cells: the hexahedron count does not grow from quanta 1e-6 to 0.5 to 0.9."""
    counts = [
        _cartesian_sphere(q).count_of(ElementType.HEXAHEDRON) for q in (1e-6, 0.5, 0.9)
    ]

    assert counts[0] >= counts[1] >= counts[2]
    assert counts[0] > counts[2]


@pytest.mark.parametrize("quanta", [0.0, 1.5])
def test_a_quanta_outside_its_range_is_refused(quanta: float) -> None:
    """Degenerate input: SetQuanta accepts ``[1e-6, 1]`` only
    (``StdMeshers_CartesianParameters3D.cxx:804``)."""
    params = CartesianParameters3D(
        spacing_x="0.5", spacing_y="0.5", spacing_z="0.5"
    ).params()
    params["use_quanta"] = True
    params["quanta"] = quanta
    with (
        Mesher(_sphere_shape()) as mesher,
        pytest.raises(PysmeshError, match="quanta must lie in"),
    ):
        mesher._m.assign("CartesianParameters3D", params, "", 0)


# ---- Cartesian_3D with viscous layers (new in 9.16) ------------------------------- #


def test_cartesian_viscous_layers_sit_at_the_geometric_closed_form() -> None:
    """Three layers, total 0.3, stretch 1.2, grown off the x = 0 wall of the box.

    Spec (SMESH ``cartesian_algo.rst`` with ``additional_hypo.rst``, "Viscous
    Layers"): layer ``i`` is ``stretch`` times layer ``i - 1`` and the layers add to
    the total thickness, so ``t1 = T (q - 1) / (q**n - 1)``. Behind the layers the
    grid fills the remaining 2.7 with whole cells of the actual spacing
    ``L / round(L / spacing)`` (SMESH ``cartesian_algo.rst``), here 0.9.
    """
    total, layers, stretch = 0.3, 3, 1.2
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Cartesian3D())
        mesher.assign(
            CartesianParameters3D(spacing_x="1.0", spacing_y="1.0", spacing_z="1.0")
        )
        mesher.assign(
            ViscousLayers(
                total_thickness=total,
                layer_count=layers,
                stretch_factor=stretch,
                boundary=(1,),
                group_name="layers",
            )
        )
        mesher.compute()
        mesh = mesher.mesh()

    x = np.unique(np.round(mesh.node_coords[:, 0], 9))
    t1 = total * (stretch - 1.0) / (stretch**layers - 1.0)
    expected = np.concatenate(
        ([0.0], np.cumsum(t1 * stretch ** np.arange(layers)), [total + 0.9])
    )
    np.testing.assert_allclose(x[: layers + 2], expected, rtol=0.0, atol=1e-9)
    assert mesh.count_of(ElementType.POLYHEDRON) == 0
