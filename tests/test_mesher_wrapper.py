# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-04

"""The typed public wrappers of the SMESH 9.16 catalogue entries and operations.

Each test drives one public class, field or method and checks the effect that the
upstream specification states, against a closed form on a shape small enough to reason
about by hand. The native paths behind them keep their own tests in
``tests/test_mesher_native.py``. The upstream sources cited are those of SMESH
``V9_16_0``.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    Arithmetic1D,
    BlockRenumber,
    BoundaryDimension,
    Cartesian3D,
    CartesianParameters3D,
    Distribution,
    ElementType,
    Hexa3D,
    Hypothesis,
    LayerDistribution2D,
    LengthFromEdges,
    LocalLength,
    MaxElementArea,
    Mefisto2D,
    Mesher,
    NotConformAllowed,
    NumberOfSegments,
    PropagOfDistribution,
    PysmeshError,
    Quadrangle2D,
    QuadrangleParams,
    RadialQuadrangle1D2D,
    Regular1D,
    ScaledJacobian,
    SegmentAroundVertex0D,
    SegmentLengthAroundVertex,
    Session,
    SmoothMethod,
    SplitMethod,
    SubShape,
    SubShapeKind,
    UseExisting1D,
    UseExisting2D,
    Volume,
    Warping3D,
)

LINE_LENGTH: float = 10.0
SQUARE_SIDE: float = 4.0
DISK_RADIUS: float = 2.5
BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0
TOL: float = 1e-9

# The trapezoid of the propagation test: a 4-long bottom, a 2-long top, height 2.
TRAPEZOID: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0),
    (3.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
)


# ---- Helpers ---------------------------------------------------------------------- #


def _line_shape(length: float = LINE_LENGTH) -> ps.Shape:
    """One straight edge from the origin along +x."""
    session = Session()
    session.add_line((0.0, 0.0, 0.0), (length, 0.0, 0.0))
    return ps.load_brep(session.brep())


def _vertex_at(shape: ps.Shape, point: tuple[float, float, float]) -> int:
    """The ordinal of the vertex of ``shape`` at ``point``."""
    for vertex in shape.vertices():
        if np.allclose(vertex.xyz, point, atol=TOL):
            return vertex.id
    raise AssertionError(f"no vertex at {point}")


def _trapezoid_shape() -> ps.Shape:
    """A planar four-sided face whose opposite sides have different lengths."""
    session = Session()
    session.add_polyline(np.array(TRAPEZOID, dtype=np.float64), closed=True)
    session.make_face(list(session.entities(ps.EntityKind.EDGE)))
    return ps.load_brep(session.brep())


def _edge_at_height(shape: ps.Shape, y: float) -> int:
    """The ordinal of the horizontal edge of ``shape`` at height ``y``."""
    for edge in shape.edges():
        box = edge.bbox
        if abs(box[1] - y) < TOL and abs(box[4] - y) < TOL:
            return edge.id
    raise AssertionError(f"no horizontal edge at y = {y}")


def _square_shape() -> ps.Shape:
    """A 4 x 4 planar square face."""
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), SQUARE_SIDE, SQUARE_SIDE)
    return ps.load_brep(session.brep())


def _triangle_area(mesh: ps.MeshData) -> float:
    """The summed area of the triangles of ``mesh``."""
    rows = [
        mesh.nodes_of(i)
        for i in range(mesh.element_count)
        if int(mesh.element_type[i]) == int(ElementType.TRIANGLE)
    ]
    tri = np.asarray(rows, dtype=np.int64)
    p0, p1, p2 = (mesh.node_coords[tri[:, k]] for k in range(3))
    return 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum())


def _mean_triangle_edge(mesh: ps.MeshData) -> float:
    """The mean edge length over the triangles of ``mesh``, shared edges twice."""
    rows = [
        mesh.nodes_of(i)
        for i in range(mesh.element_count)
        if int(mesh.element_type[i]) == int(ElementType.TRIANGLE)
    ]
    tri = np.asarray(rows, dtype=np.int64)
    p = mesh.node_coords
    pairs = ((0, 1), (1, 2), (2, 0))
    lengths = [np.linalg.norm(p[tri[:, a]] - p[tri[:, b]], axis=1) for a, b in pairs]
    return float(np.concatenate(lengths).mean())


def _sorted_x(mesh: ps.MeshData) -> NDArray[np.float64]:
    """The x coordinate of every node, ascending."""
    return np.sort(np.ascontiguousarray(mesh.node_coords[:, 0], dtype=np.float64))


# ---- W1.1 SegmentAroundVertex0D --------------------------------------------------- #


def test_segment_around_vertex_0d_gives_the_vertex_segment_its_length() -> None:
    """The segment at the vertex takes the length; the far end keeps its spacing.

    Spec (SMESH ``segments_around_vertex_algo.rst``): the edge is split by its 1-D
    hypothesis, then the nodes near the vertex move so that the segment there has the
    length of ``SegmentLengthAroundVertex``. Five equal segments of a 10-long edge are
    2.0 each, so the far segment stays 2.0.
    """
    shape = _line_shape()
    origin = SubShape(SubShapeKind.VERTEX, _vertex_at(shape, (0.0, 0.0, 0.0)))

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=5))
        mesher.assign(SegmentAroundVertex0D(), on=origin)
        mesher.assign(SegmentLengthAroundVertex(length=0.5), on=origin)
        mesher.compute()
        x = _sorted_x(mesher.mesh())

    assert x[1] - x[0] == pytest.approx(0.5, abs=TOL)
    assert x[-1] - x[-2] == pytest.approx(2.0, abs=TOL)
    assert (x[0], x[-1]) == pytest.approx((0.0, LINE_LENGTH), abs=TOL)


# ---- W1.3 PropagOfDistribution ---------------------------------------------------- #


def test_propag_of_distribution_repeats_the_fractions_on_a_shorter_edge() -> None:
    """The 2-long top edge gets the node fractions of the 4-long bottom edge.

    Spec (SMESH ``additional_hypo.rst``, "Propagation of Node Distribution on Opposite
    Edges"): the opposite edge gets "the same number of nodes and the same relations
    between segment lengths". Arithmetic1D(0.5, 1.5) on the 4-long bottom gives
    ``n = 2 L / (a + b) = 4`` segments of 0.5, 5/6, 7/6 and 1.5, so the nodes sit at the
    fractions 0, 1/8, 1/3, 5/8 and 1. The top edge carries them in either direction.
    """
    shape = _trapezoid_shape()
    bottom = SubShape(SubShapeKind.EDGE, _edge_at_height(shape, 0.0))
    fractions = np.array([0.0, 1.0 / 8.0, 1.0 / 3.0, 5.0 / 8.0, 1.0])

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Arithmetic1D(start_length=0.5, end_length=1.5), on=bottom)
        mesher.assign(PropagOfDistribution(), on=bottom)
        mesher.compute()
        xyz = mesher.mesh().node_coords

    on_bottom = np.sort(xyz[np.abs(xyz[:, 1]) < TOL, 0]) / 4.0
    on_top = np.sort(xyz[np.abs(xyz[:, 1] - 2.0) < TOL, 0] - 1.0) / 2.0
    np.testing.assert_allclose(on_bottom, fractions, atol=TOL)
    forward = np.allclose(on_top, fractions, atol=TOL)
    backward = np.allclose(on_top, np.sort(1.0 - fractions), atol=TOL)
    assert forward or backward, on_top


# ---- W1.4 LengthFromEdges --------------------------------------------------------- #


def _mefisto_square(segments: int, length_from_edges: bool) -> ps.MeshData:
    """MEFISTO on the square, ``segments`` boundary segments per side."""
    with Mesher(_square_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=segments))
        mesher.assign(Mefisto2D())
        if length_from_edges:
            mesher.assign(LengthFromEdges())
        mesher.compute()
        return mesher.mesh()


@pytest.mark.parametrize("segments", [8, 16, 32])
def test_length_from_edges_sizes_the_triangles_by_the_boundary_segment(
    segments: int,
) -> None:
    """The triangles fill the square at the size of the boundary segment ``h``.

    Spec (SMESH ``2d_meshing_hypo.rst``): LengthFromEdges "defines the maximum linear
    size of mesh faces as an average length of mesh edges approximating the meshed
    face boundary"; here ``h = 4 / segments``. MEFISTO takes it as an ideal edge
    length (``aptrte.cxx``: an edge "should" lie between 0.65 and 1.3 of it), so the
    mean triangle edge is held within a factor 1.5 of ``h``. It is also MEFISTO's
    default (``StdMeshers_MEFISTO_2D::CheckHypothesis``), so the mesh equals the one
    made with no 2-D hypothesis. The triangles fill the 4 x 4 square: area 16.
    """
    h = SQUARE_SIDE / segments

    mesh = _mefisto_square(segments, length_from_edges=True)

    default = _mefisto_square(segments, length_from_edges=False)
    np.testing.assert_array_equal(
        np.sort(mesh.node_coords, axis=0), np.sort(default.node_coords, axis=0)
    )
    assert _triangle_area(mesh) == pytest.approx(SQUARE_SIDE**2, abs=TOL)
    assert h / 1.5 < _mean_triangle_edge(mesh) < 1.5 * h


# ---- W1.5 LayerDistribution2D ----------------------------------------------------- #


def _disk_shape() -> ps.Shape:
    """A disk of radius 2.5 in the z = 0 plane."""
    session = Session()
    session.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), DISK_RADIUS)
    session.make_face(list(session.entities(ps.EntityKind.EDGE)))
    return ps.load_brep(session.brep())


def test_layer_distribution_2d_spaces_the_rings_by_the_scale_law_inward() -> None:
    """Ring radii follow the SCALE law laid from the circle to the centre.

    Spec (SMESH ``radial_quadrangle_1D2D_algo.rst``): the layer distribution "can be set
    with any 1D Hypothesis" and "is applied to the longest radial edge starting from its
    end lying on the elliptic curve". The SCALE law (``StdMeshers_Regular_1D``,
    DT_Scale) puts node ``i`` of ``n`` at ``s_i / L = (1 - a^i) / (1 - a^n)`` with
    ``a = scale^(1 / (n - 1))``. Four layers growing three times from the circle inward
    put the rings at ``R (1 - s_i / L)``.
    """
    inner = NumberOfSegments(count=4, distribution=Distribution.SCALE, scale_factor=3.0)
    a = 3.0 ** (1.0 / 3.0)
    i = np.arange(5, dtype=np.float64)
    expected = np.sort(DISK_RADIUS * (1.0 - (1.0 - a**i) / (1.0 - a**4)))

    with Mesher(_disk_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=8))
        mesher.assign(RadialQuadrangle1D2D())
        mesher.assign(LayerDistribution2D(distribution=inner))
        mesher.compute()
        xyz = mesher.mesh().node_coords

    radii = np.unique(np.round(np.hypot(xyz[:, 0], xyz[:, 1]), 9))
    np.testing.assert_allclose(radii, expected, rtol=0.0, atol=TOL)


# ---- W1.7 NotConformAllowed ------------------------------------------------------- #


def _box_shape() -> ps.Shape:
    """The 3 x 7 x 11 box."""
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    return ps.load_brep(session.brep())


def _hexa_box(*extra: Hypothesis) -> ps.MeshData:
    """Hexa3D on the box, 3 segments per edge, ``extra`` assigned globally first."""
    with Mesher(_box_shape()) as mesher:
        for hypothesis in extra:
            mesher.assign(hypothesis)
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.compute()
        return mesher.mesh()


def test_not_conform_allowed_on_a_sub_shape_is_refused() -> None:
    """Spec: the hypothesis "can be only global" (``SMESH_Mesh.cxx:658-670``)."""
    with (
        Mesher(_box_shape()) as mesher,
        pytest.raises(PysmeshError, match="NotConformAllowed"),
    ):
        mesher.assign(NotConformAllowed(), on=SubShape(SubShapeKind.FACE, 1))


def test_not_conform_allowed_globally_keeps_a_conformal_mesh() -> None:
    """Accepted globally; the 3 x 3 x 3 block mesh is node for node the plain one."""
    allowed = _hexa_box(NotConformAllowed())

    plain = _hexa_box()
    assert allowed.count_of(ElementType.HEXAHEDRON) == 27
    np.testing.assert_array_equal(
        np.sort(allowed.node_coords, axis=0), np.sort(plain.node_coords, axis=0)
    )


# ---- W1.6 BlockRenumber ----------------------------------------------------------- #


def _structured_order(
    mesh: ps.MeshData, origin: NDArray[np.float64], axes: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Grid index ``i + n (j + n k)`` of each hexahedron and node, in mesh id order.

    ``axes`` holds the unit i, j, k directions as rows, ``origin`` the local origin.
    The box has 3 cells along each edge, so a cell spans a third of each edge.
    """
    dims = np.abs(axes @ np.array([BOX_DX, BOX_DY, BOX_DZ], dtype=np.float64))
    step = dims / 3.0
    hexa = np.flatnonzero(mesh.element_type == int(ElementType.HEXAHEDRON))
    hexa = hexa[np.argsort(mesh.element_id[hexa])]
    centres = np.array([mesh.node_coords[mesh.nodes_of(i)].mean(axis=0) for i in hexa])
    cell = np.floor(((centres - origin) @ axes.T) / step).astype(np.int64)
    order = np.argsort(mesh.node_id)
    point = np.rint(((mesh.node_coords[order] - origin) @ axes.T) / step)
    point = point.astype(np.int64)
    cells = cell[:, 0] + 3 * (cell[:, 1] + 3 * cell[:, 2])
    nodes = point[:, 0] + 4 * (point[:, 1] + 4 * point[:, 2])
    return np.asarray(cells, dtype=np.int64), np.asarray(nodes, dtype=np.int64)


def test_block_renumber_numbers_a_box_along_the_global_axes() -> None:
    """Hexahedra and nodes in i, j, k = x, y, z order from the origin, i fastest.

    Spec (SMESH ``3d_meshing_hypo.rst``, "Renumber hypothesis"): it gives "hexahedra
    and nodes ordered like in a structured grid"; a block with edges parallel to the
    global axes takes them as its axes. The origin is the corner of least x + y + z
    (``StdMeshers_RenumberHelper::GetVertex000``): (0, 0, 0).
    """
    mesh = _hexa_box(BlockRenumber())

    cells, nodes = _structured_order(mesh, np.zeros(3), np.eye(3))
    np.testing.assert_array_equal(cells, np.arange(27))
    np.testing.assert_array_equal(nodes, np.arange(64))


def test_block_renumber_takes_the_axes_of_the_named_vertices() -> None:
    """k from (3, 7, 11) to (3, 7, 0): i = -y and j = -x by the right-hand rule.

    Spec (``3d_meshing_hypo.rst``): vertex (0,0,0) is the origin, vertex (0,0,1) the
    end of the k axis, and "axes i and j are found automatically using the right-hand
    rule". The two other edges at (3, 7, 11) run along -x and -y; i x j = k = -z holds
    for i = -y, j = -x (``StdMeshers_Hexa_3D.cxx``, ``arrangeForRenumber``).
    """
    shape = _box_shape()
    origin = _vertex_at(shape, (BOX_DX, BOX_DY, BOX_DZ))
    k_end = _vertex_at(shape, (BOX_DX, BOX_DY, 0.0))

    mesh = _hexa_box(BlockRenumber(blocks=((1, origin, k_end),)))

    axes = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    corner = np.array([BOX_DX, BOX_DY, BOX_DZ])
    cells, nodes = _structured_order(mesh, corner, axes)
    np.testing.assert_array_equal(cells, np.arange(27))
    np.testing.assert_array_equal(nodes, np.arange(64))


def test_block_renumber_refuses_vertices_that_share_no_edge() -> None:
    """Opposite corners cannot set a k axis: the compute fails and says why.

    Spec (``StdMeshers_BlockRenumber::CheckHypothesis``): for a block of 8 vertices the
    two vertices must be "connected by an edge".
    """
    shape = _box_shape()
    origin = _vertex_at(shape, (0.0, 0.0, 0.0))
    far = _vertex_at(shape, (BOX_DX, BOX_DY, BOX_DZ))

    with pytest.raises(PysmeshError, match="meshing failed") as caught:
        _hexa_box(BlockRenumber(blocks=((1, origin, far),)))

    assert "not connected by an edge" in caught.value.details


# ---- W1.2 UseExisting1D, UseExisting2D and the bound fill ------------------------- #


def _xyz_of(mesh: ps.MeshData, element: int) -> NDArray[np.float64]:
    """The node coordinates of one element, (k, 3)."""
    return np.asarray(mesh.node_coords[mesh.nodes_of(element)], dtype=np.float64)


def _edge_between(
    shape: ps.Shape, a: tuple[float, float, float], b: tuple[float, float, float]
) -> int:
    """The ordinal of the straight edge of ``shape`` from ``a`` to ``b``."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    for edge in shape.edges():
        box = np.asarray(edge.bbox)
        if np.allclose(box[:3], lo, atol=TOL) and np.allclose(box[3:], hi, atol=TOL):
            return edge.id
    raise AssertionError(f"no edge from {a} to {b}")


def _vertex_node(mesh: ps.MeshData, ordinal: int) -> int:
    """The id of the node that a compute put on vertex ``ordinal``."""
    on_vertex = (mesh.node_kind == int(SubShapeKind.VERTEX)) & (
        mesh.node_ordinal == ordinal
    )
    (row,) = np.flatnonzero(on_vertex)
    return int(mesh.node_id[row])


def _coons_nodes(
    bottom: NDArray[np.float64], width: float, height: float
) -> NDArray[np.float64]:
    """The nodes of a 3 x 3 mapped mesh of a width x height rectangle, sorted.

    The bottom side has its nodes at the fractions ``bottom``, the three other sides 3
    equal segments. Quadrangle_2D puts node (i, j) at the fraction
    ``x = (1 - y) b_i + y t_i`` with ``y = j / 3`` and ``t_i = i / 3``
    (``StdMeshers_Quadrangle_2D``, ``computeQuadDominant``); the Coons patch of a
    rectangle maps it to ``(x W, y H)``.
    """
    y = np.arange(4, dtype=np.float64)[:, None] / 3.0
    t = np.arange(4, dtype=np.float64)[None, :] / 3.0
    x = (1.0 - y) * bottom[None, :] + y * t
    grid = np.stack([(x * width).ravel(), np.repeat(y.ravel() * height, 4)], axis=1)
    return grid[np.lexsort((grid[:, 1], grid[:, 0]))]


def test_use_existing_1d_takes_script_segments_and_the_faces_mesh_from_them() -> None:
    """Script nodes at x = 0.5 and 2 on the edge shape the two faces beside it.

    Spec (SMESH ``define_mesh_by_script.rst``): "Use Edges to be Created Manually"
    lets a script make the 1-D mesh, whose nodes and elements "must be assigned to
    geometry entities ... in order to be used by an algorithm of upper dimension". The
    box edge from (0, 0, 0) to (3, 0, 0) gets nodes at the fractions 0, 1/6, 2/3, 1;
    every other edge 3 equal segments. Quadrangle2D then maps both faces at the edge
    from these nodes: the closed form of :func:`_coons_nodes`.
    """
    shape = _box_shape()
    edge = SubShape(
        SubShapeKind.EDGE, _edge_between(shape, (0.0, 0.0, 0.0), (BOX_DX, 0.0, 0.0))
    )
    start = _vertex_at(shape, (0.0, 0.0, 0.0))
    end = _vertex_at(shape, (BOX_DX, 0.0, 0.0))
    inner_xyz = np.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(UseExisting1D(), on=edge)
        mesher.compute()
        first = mesher.mesh()
        inner = mesher.add_nodes(inner_xyz, on=edge)
        chain = [_vertex_node(first, start), *inner.tolist(), _vertex_node(first, end)]
        segments = mesher.add_segments(np.array(list(pairwise(chain))), on=edge)
        mesher.assign(Quadrangle2D())
        mesher.compute()
        mesh = mesher.mesh()

    bottom = np.array([0.0, 0.5, 2.0, BOX_DX]) / BOX_DX
    on_edge = (mesh.element_kind == int(SubShapeKind.EDGE)) & (
        mesh.element_ordinal == edge.ordinal
    )
    assert sorted(mesh.element_id[on_edge].tolist()) == sorted(segments.tolist())
    xyz = mesh.node_coords
    for plane, (axis, height) in {1: (2, BOX_DZ), 2: (1, BOX_DY)}.items():
        on_face = np.abs(xyz[:, plane]) < TOL
        got = xyz[on_face][:, [0, axis]]
        got = got[np.lexsort((got[:, 1], got[:, 0]))]
        np.testing.assert_allclose(got, _coons_nodes(bottom, BOX_DX, height), atol=TOL)


def test_use_existing_2d_takes_script_faces_and_the_solid_meshes_from_them() -> None:
    """Script quadrangles on one face make that face's mesh; Hexa3D fills the box.

    Spec (``define_mesh_by_script.rst``): "Use Faces to be Created Manually", then a 3-D
    algorithm. The script copies the 3 x 3 quadrangles Quadrangle2D would make on the
    face at z = 0, bound to that face; Hexa3D then fills the box with 27 hexahedra of
    total volume 3 x 7 x 11 = 231.
    """
    shape = _box_shape()
    bottom = next(
        f.id for f in shape.faces() if abs(f.bbox[2]) < TOL and abs(f.bbox[5]) < TOL
    )
    face = SubShape(SubShapeKind.FACE, bottom)

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(UseExisting2D(), on=face)
        mesher.compute()
        first = mesher.mesh()
        xyz = first.node_coords
        step = np.array([BOX_DX, BOX_DY]) / 3.0
        ij = np.rint(xyz[:, :2] / step).astype(np.int64)
        at_bottom = np.abs(xyz[:, 2]) < TOL
        ids = {
            (int(i), int(j)): int(n)
            for (i, j), n, keep in zip(ij, first.node_id, at_bottom, strict=True)
            if keep
        }
        pairs = [(i, j) for i in (1, 2) for j in (1, 2)]
        interior = np.array([[i * step[0], j * step[1], 0.0] for i, j in pairs])
        made = mesher.add_nodes(interior, on=face)
        for k, (i, j) in enumerate(pairs):
            ids[(i, j)] = int(made[k])
        quads = np.array(
            [
                [ids[(i, j)], ids[(i, j + 1)], ids[(i + 1, j + 1)], ids[(i + 1, j)]]
                for i in range(3)
                for j in range(3)
            ]
        )
        mesher.add_quadrangles(quads, on=face)
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.compute()
        mesh = mesher.mesh()

    hexa = np.flatnonzero(mesh.element_type == int(ElementType.HEXAHEDRON))
    assert hexa.size == 27
    volume = sum(float(np.prod(np.ptp(_xyz_of(mesh, i), axis=0))) for i in hexa)
    assert volume == pytest.approx(BOX_DX * BOX_DY * BOX_DZ, rel=1e-12)


def test_a_node_given_its_curve_parameter_is_bound_and_a_wrong_one_refused() -> None:
    """A node at u is accepted with u; with another u, or off the edge, it is refused.

    The edge's own parameter range and the curve point at u come from the session
    queries (``Session.edge_parameter_bounds``, ``Session.curve_at``). A refusal adds no
    node.
    """
    session = Session()
    session.add_line((0.0, 0.0, 0.0), (LINE_LENGTH, 0.0, 0.0))
    (edge_id,) = session.entities(ps.EntityKind.EDGE).tolist()
    lo, hi = session.edge_parameter_bounds([edge_id])[0]
    u = lo + 0.25 * (hi - lo)
    point = session.curve_at(edge_id, [u]).points
    edge = SubShape(SubShapeKind.EDGE, 1)

    with Mesher(ps.load_brep(session.brep())) as mesher:
        made = mesher.add_nodes(point, on=edge, parameters=np.array([u]))
        with pytest.raises(PysmeshError, match="does not lie on EDGE 1"):
            mesher.add_nodes(point, on=edge, parameters=np.array([u + 0.1 * (hi - lo)]))
        with pytest.raises(PysmeshError, match="does not lie on EDGE 1"):
            mesher.add_nodes(np.array([[5.0, 1.0, 0.0]]), on=edge)
        count = mesher.mesh().node_coords.shape[0]

    assert made.shape == (1,)
    assert count == 1


def test_an_element_bound_to_a_sub_shape_of_another_dimension_is_refused() -> None:
    """A segment cannot be bound to a face: the dimensions differ."""
    with Mesher(_box_shape()) as mesher:
        nodes = mesher.add_nodes(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))

        with pytest.raises(PysmeshError, match="dimensions differ"):
            mesher.add_segments(nodes[None, :], on=SubShape(SubShapeKind.FACE, 1))


# ---- W2.1 Distribution.BETA_LAW and NumberOfSegments.beta ------------------------- #


def _beta_law(beta: float, count: int, length: float) -> NDArray[np.float64]:
    """Node positions of the beta law on a straight edge, ends included.

    ``x_i / L = 1 + b (1 - r^(1 - i/n)) / (1 + r^(1 - i/n))`` for ``i = 1 .. n-1``, with
    ``b = |beta|`` and ``r = (b + 1) / (b - 1)``; a negative ``beta`` mirrors them.
    """
    b = abs(beta)
    r = (b + 1.0) / (b - 1.0)
    power = r ** (1.0 - np.arange(1, count, dtype=np.float64) / count)
    t = 1.0 + b * (1.0 - power) / (1.0 + power)
    if beta < 0:
        t = np.sort(1.0 - t)
    return np.concatenate(([0.0], t * length, [length]))


@pytest.mark.parametrize("beta", [1.01, 1.5, -1.05])
def test_number_of_segments_beta_law_places_the_nodes_at_the_closed_form(
    beta: float,
) -> None:
    """Ten segments on a 10-long edge sit at the beta law (SMESH ``computeBetaLaw``).

    Spec (SMESH ``1d_meshing_hypo.rst``, "Beta Law Distribution"); a straight edge has
    an exact arc-length parametrisation, so the positions hold to round-off.
    """
    hypothesis = NumberOfSegments(
        count=10, distribution=Distribution.BETA_LAW, beta=beta
    )

    with Mesher(_line_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(hypothesis)
        mesher.compute()
        x = _sorted_x(mesher.mesh())

    np.testing.assert_allclose(x, _beta_law(beta, 10, LINE_LENGTH), rtol=0.0, atol=TOL)


@pytest.mark.parametrize("beta", [1.0, 0.5, -1.0])
def test_number_of_segments_beta_law_refuses_a_beta_in_the_unit_interval(
    beta: float,
) -> None:
    """Degenerate input: ``[-1, 1]`` is forbidden for the law's logarithm."""
    hypothesis = NumberOfSegments(
        count=10, distribution=Distribution.BETA_LAW, beta=beta
    )

    with (
        Mesher(_line_shape()) as mesher,
        pytest.raises(PysmeshError, match=r"needs \|beta\| > 1"),
    ):
        mesher.assign(hypothesis)


def test_number_of_segments_ignores_beta_under_another_law() -> None:
    """A beta under the REGULAR law is not read: five equal segments of 2."""
    with Mesher(_line_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=5, beta=3.0))
        mesher.compute()
        x = _sorted_x(mesher.mesh())

    np.testing.assert_allclose(np.diff(x), np.full(5, 2.0), atol=TOL)


# ---- W2.2 CartesianParameters3D.use_quanta and quanta ----------------------------- #


def _sphere_shape() -> ps.Shape:
    """A sphere of radius 2."""
    session = Session()
    session.add_sphere(2.0)
    return ps.load_brep(session.brep())


def _cartesian(shape: ps.Shape, parameters: CartesianParameters3D) -> ps.MeshData:
    """Cartesian3D on ``shape`` with ``parameters``."""
    with Mesher(shape) as mesher:
        mesher.assign(Cartesian3D())
        mesher.assign(parameters)
        mesher.compute()
        return mesher.mesh()


def test_use_quanta_turns_every_cut_cell_of_a_sphere_into_one_hexahedron() -> None:
    """At quanta 1e-6 no polyhedron is left: each became one hexahedron of its cell.

    Spec (SMESH ``cartesian_algo.rst``, "Set Quanta"): a boundary polyhedron is
    replaced by a hexahedron "if the volume of the polyhedron divided by the
    equivalent hexahedron is bigger than Quanta". At 1e-6 every cut cell qualifies, so
    the hexahedron count becomes the old hexahedra plus the old polyhedra, and each
    hexahedron fits in one 0.5 cell of the grid.
    """
    spacing = {"spacing_x": "0.5", "spacing_y": "0.5", "spacing_z": "0.5"}
    plain = _cartesian(_sphere_shape(), CartesianParameters3D(**spacing))

    quantized = _cartesian(
        _sphere_shape(), CartesianParameters3D(**spacing, use_quanta=True, quanta=1e-6)
    )

    polyhedra = plain.count_of(ElementType.POLYHEDRON)
    assert polyhedra > 0
    assert quantized.count_of(ElementType.POLYHEDRON) == 0
    assert quantized.count_of(ElementType.HEXAHEDRON) == (
        plain.count_of(ElementType.HEXAHEDRON) + polyhedra
    )
    hexa = np.flatnonzero(quantized.element_type == int(ElementType.HEXAHEDRON))
    extents = np.array([np.ptp(_xyz_of(quantized, i), axis=0) for i in hexa])
    assert np.all(extents > 0.0)
    assert np.all(extents <= 0.5 + TOL)


@pytest.mark.parametrize("quanta", [0.0, 1.5])
def test_a_quanta_outside_the_unit_range_is_refused(quanta: float) -> None:
    """Degenerate input: SetQuanta accepts ``[1e-6, 1]`` only."""
    parameters = CartesianParameters3D(
        spacing_x="0.5",
        spacing_y="0.5",
        spacing_z="0.5",
        use_quanta=True,
        quanta=quanta,
    )

    with (
        Mesher(_sphere_shape()) as mesher,
        pytest.raises(PysmeshError, match="quanta must lie in"),
    ):
        mesher.assign(parameters)


# ---- W2.3 CartesianParameters3D grid setters -------------------------------------- #


def _cartesian_volume(
    shape: ps.Shape, parameters: CartesianParameters3D
) -> tuple[ps.MeshData, float]:
    """Cartesian3D on ``shape``: the mesh and the summed volume of its cells."""
    with Mesher(shape) as mesher:
        mesher.assign(Cartesian3D())
        mesher.assign(parameters)
        mesher.compute()
        volume = float(np.sum(mesher.quality(Volume()).values))
        return mesher.mesh(), volume


def test_explicit_coordinates_put_the_grid_planes_where_they_are_given() -> None:
    """x nodes at exactly 0, 0.5, 2, 3: 3 x 2 x 2 hexahedra filling the 3 x 7 x 11 box.

    Spec (SMESH ``cartesian_algo.rst``): "You can specify the Coordinates of grid
    nodes" (``StdMeshers_CartesianParameters3D::SetGrid``).
    """
    parameters = CartesianParameters3D(
        coordinates_x=(0.0, 0.5, 2.0, 3.0), spacing_y="3.5", spacing_z="5.5"
    )

    mesh, volume = _cartesian_volume(_box_shape(), parameters)

    np.testing.assert_array_equal(
        np.unique(np.round(mesh.node_coords[:, 0], 12)), [0.0, 0.5, 2.0, 3.0]
    )
    assert mesh.count_of(ElementType.HEXAHEDRON) == 12
    assert volume == pytest.approx(BOX_DX * BOX_DY * BOX_DZ, rel=1e-12)


@pytest.mark.parametrize(
    ("spacing", "coordinates", "word"),
    [("1.0", (0.0, 3.0), "both"), ("", (), "neither")],
)
def test_an_axis_with_both_or_neither_grid_definitions_is_refused(
    spacing: str, coordinates: tuple[float, ...], word: str
) -> None:
    """Each axis takes a spacing or coordinates; both, or neither, is refused."""
    parameters = CartesianParameters3D(
        spacing_x=spacing, coordinates_x=coordinates, spacing_y="1", spacing_z="1"
    )

    with (
        Mesher(_box_shape()) as mesher,
        pytest.raises(PysmeshError, match=f"spacing_x or coordinates_x, not {word}"),
    ):
        mesher.assign(parameters)


@pytest.mark.parametrize("fixed", [True, False])
def test_a_fixed_point_puts_a_grid_node_at_it(fixed: bool) -> None:
    """Spec (``cartesian_algo.rst``, "Fixed Point"): with every direction by spacing,
    "there will be a mesh node at the Fixed Point". The unit spacing alone puts none
    at (0.25, 0.5, 0.75), which is the falsification."""
    point = (0.25, 0.5, 0.75)
    parameters = CartesianParameters3D(
        spacing_x="1",
        spacing_y="1",
        spacing_z="1",
        fixed_point=point if fixed else (),
    )

    mesh, volume = _cartesian_volume(_box_shape(), parameters)

    at_point = np.all(np.abs(mesh.node_coords - np.array(point)) < TOL, axis=1)
    assert bool(np.any(at_point)) == fixed
    assert volume == pytest.approx(BOX_DX * BOX_DY * BOX_DZ, rel=1e-12)


def test_axis_directions_align_the_grid_with_a_rotated_box() -> None:
    """A 4-cube turned 30 degrees about z, grid axes turned with it: 512 whole cubes.

    Spec (``cartesian_algo.rst``, "Directions of Axes"): the grid's axes follow the
    given directions (``SetAxisDirs``). Aligned with the cube's faces, the 0.5 grid
    cuts nothing: (4 / 0.5)^3 = 512 hexahedra, no polyhedron, volume 64. With the global
    axes the same cube is cut at every side face, which leaves polyhedra.
    """
    angle = math.radians(30.0)
    c, s = math.cos(angle), math.sin(angle)
    session = Session()
    session.add_box(4.0, 4.0, 4.0)
    session.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), angle)
    shape = ps.load_brep(session.brep())
    spacing = {"spacing_x": "0.5", "spacing_y": "0.5", "spacing_z": "0.5"}

    turned, volume = _cartesian_volume(
        shape,
        CartesianParameters3D(
            **spacing, axis_directions=(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0)
        ),
    )

    plain, _ = _cartesian_volume(shape, CartesianParameters3D(**spacing))
    assert turned.count_of(ElementType.HEXAHEDRON) == 512
    assert turned.count_of(ElementType.POLYHEDRON) == 0
    assert volume == pytest.approx(64.0, rel=1e-12)
    assert plain.count_of(ElementType.POLYHEDRON) > 0


def test_parallel_axis_directions_are_refused() -> None:
    """Degenerate input: SetAxisDirs refuses two parallel directions."""
    parameters = CartesianParameters3D(
        spacing_x="1",
        spacing_y="1",
        spacing_z="1",
        axis_directions=(1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )

    with (
        Mesher(_box_shape()) as mesher,
        pytest.raises(PysmeshError, match="Parallel axis directions"),
    ):
        mesher.assign(parameters)


@pytest.mark.parametrize(("threshold", "expected"), [(False, 2.0), (True, 1.95)])
def test_the_threshold_on_a_shared_face_leaves_out_the_thin_slab(
    threshold: bool, expected: float
) -> None:
    """Two unit-section boxes share the face x = 1.05; the 0.5 grid cuts a 0.05 slab.

    Spec (``cartesian_algo.rst``): "Apply Threshold to Shared / Internal Faces"
    applies the size threshold to the cells the shared face cuts, "that can cause
    appearance of holes inside the mesh". A 0.05-wide piece of a 0.5 cell is below
    1 / size_threshold = 1/4, so with the option the slab 1.0 <= x <= 1.05 is left out:
    volume 2 - 0.05 = 1.95. Without it the mesh fills both boxes: volume 2.
    """
    session = Session()
    session.add_box(1.05, 1.0, 1.0)
    session.add_box(0.95, 1.0, 1.0, origin=(1.05, 0.0, 0.0))
    session.fragment(session.entities(ps.EntityKind.SOLID).tolist())
    parameters = CartesianParameters3D(
        spacing_x="0.5",
        spacing_y="0.5",
        spacing_z="0.5",
        consider_internal_faces=True,
        threshold_for_internal_faces=threshold,
    )

    _, volume = _cartesian_volume(ps.load_brep(session.brep()), parameters)

    assert volume == pytest.approx(expected, rel=1e-12)


# ---- W2.4 QuadrangleParams enforced nodes ----------------------------------------- #

ENFORCED: tuple[float, float, float] = (1.5, 2.5, 0.0)


def _quad_square(params: QuadrangleParams | None, *, with_vertex: bool) -> ps.MeshData:
    """Quadrangle2D on the 4 x 4 square, 4 segments per side, with ``params``.

    With ``with_vertex`` the shape also holds a free vertex at :data:`ENFORCED`.
    """
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), SQUARE_SIDE, SQUARE_SIDE)
    if with_vertex:
        session.add_vertex(ENFORCED)
    with Mesher(ps.load_brep(session.brep())) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=4))
        mesher.assign(Quadrangle2D())
        if params is not None:
            mesher.assign(params)
        mesher.compute()
        return mesher.mesh()


def _face_node_at(mesh: ps.MeshData, point: tuple[float, float, float]) -> bool:
    """True if a node bound to a face lies at ``point``."""
    on_face = mesh.node_kind == int(SubShapeKind.FACE)
    near = np.all(np.abs(mesh.node_coords - np.array(point)) < TOL, axis=1)
    return bool(np.any(on_face & near))


def _quad_uses_node_at(mesh: ps.MeshData, point: tuple[float, float, float]) -> bool:
    """True if a quadrangle of ``mesh`` has a node at ``point``."""
    quads = np.flatnonzero(mesh.element_type == int(ElementType.QUADRANGLE))
    target = np.array(point)
    return any(
        bool(np.any(np.all(np.abs(_xyz_of(mesh, i) - target) < TOL, axis=1)))
        for i in quads
    )


def _quad_area(mesh: ps.MeshData) -> tuple[float, float]:
    """The summed area of the quadrangles and the smallest signed area among them."""
    quads = np.flatnonzero(mesh.element_type == int(ElementType.QUADRANGLE))
    areas = []
    for i in quads:
        p = _xyz_of(mesh, i)
        cross = np.cross(p[2] - p[0], p[3] - p[1])
        areas.append(0.5 * float(cross[2]))
    signed = np.array(areas)
    return float(np.abs(signed).sum()), float(np.min(np.sign(signed[0]) * signed))


def test_an_enforced_point_puts_a_node_on_it_and_the_face_stays_filled() -> None:
    """A node of the face at (1.5, 2.5, 0), off the regular 1.0 grid; area still 16.

    Spec (SMESH ``2d_meshing_hypo.rst``, "Enforced nodes"): the points "where the
    algorithm should create nodes"; the node closest to the point moves to it. Every
    quadrangle keeps the orientation of the first, so none is folded.
    """
    plain = _quad_square(None, with_vertex=False)

    mesh = _quad_square(
        QuadrangleParams(enforced_points=(ENFORCED,)), with_vertex=False
    )

    assert _face_node_at(mesh, ENFORCED)
    assert not _face_node_at(plain, ENFORCED)
    area, smallest = _quad_area(mesh)
    assert area == pytest.approx(SQUARE_SIDE**2, rel=1e-12)
    assert smallest > 0.0


def test_an_enforced_vertex_brings_its_node_into_the_face_mesh() -> None:
    """A free vertex of the shape, named by ordinal: the quadrangles meet at its node.

    The vertex has its own node from the start; upstream makes that node the enforced
    one (``StdMeshers_Quadrangle_2D::getEnforcedUV``). Without the hypothesis no
    quadrangle uses it.
    """
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), SQUARE_SIDE, SQUARE_SIDE)
    session.add_vertex(ENFORCED)
    vertex = _vertex_at(ps.load_brep(session.brep()), ENFORCED)
    plain = _quad_square(None, with_vertex=True)

    mesh = _quad_square(QuadrangleParams(enforced_vertices=(vertex,)), with_vertex=True)

    assert _quad_uses_node_at(mesh, ENFORCED)
    assert not _quad_uses_node_at(plain, ENFORCED)
    assert _quad_area(mesh)[0] == pytest.approx(SQUARE_SIDE**2, rel=1e-12)


# ---- W3.1 split_volumes(avoid_over_constrained) ----------------------------------- #


def _over_constrained(mesh: ps.MeshData) -> int:
    """How many tetrahedra have all 4 nodes on a node of a 2-D element."""
    faces = np.flatnonzero(
        (mesh.element_type == int(ElementType.QUADRANGLE))
        | (mesh.element_type == int(ElementType.TRIANGLE))
    )
    on_surface = np.zeros(mesh.node_coords.shape[0], dtype=bool)
    for i in faces:
        on_surface[mesh.nodes_of(i)] = True
    tetra = np.flatnonzero(mesh.element_type == int(ElementType.TETRAHEDRON))
    return sum(int(np.all(on_surface[mesh.nodes_of(i)])) for i in tetra)


def _tetra_volume(mesh: ps.MeshData) -> float:
    """The summed volume of the tetrahedra of ``mesh``."""
    tetra = np.flatnonzero(mesh.element_type == int(ElementType.TETRAHEDRON))
    total = 0.0
    for i in tetra:
        p = _xyz_of(mesh, i)
        total += abs(float(np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0])))) / 6
    return total


def _split_block(avoid: bool) -> ps.MeshData:
    """The 2 x 2 x 2 Hexa3D block of the box, split by HEXA_TO_6."""
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.compute()
        mesher.split_volumes(SplitMethod.HEXA_TO_6, avoid_over_constrained=avoid)
        return mesher.mesh()


def test_avoid_over_constrained_splits_every_block_cell_through_the_centre_node() -> (
    None
):
    """2 x 2 x 2 hexahedra into 48 tetrahedra, none with all 4 nodes on the boundary.

    Spec (SMESH ``SMESH_MeshEditor::SplitVolumes``, 9.16): with
    ``avoidOverConstrainedVolumes`` the split "will choose the variant with no
    over-constrained volumes". Each cell of the block has one interior node, the
    block centre, and one of the 4 diagonals of HEXA_TO_6 runs through it: then each
    of the 6 tetrahedra holds it. Without the option some tetrahedra are
    over-constrained. Both fill the box: volume 231.
    """
    plain = _split_block(avoid=False)

    avoided = _split_block(avoid=True)

    assert avoided.count_of(ElementType.TETRAHEDRON) == 48
    assert _over_constrained(avoided) == 0
    assert _over_constrained(plain) > 0
    for mesh in (plain, avoided):
        assert _tetra_volume(mesh) == pytest.approx(BOX_DX * BOX_DY * BOX_DZ, rel=1e-12)


# ---- W3.2 make_boundary_mesh ------------------------------------------------------ #

GRID: tuple[int, int, int] = (2, 3, 4)


def _hexa_grid(mesher: Mesher, a: int, b: int, c: int) -> NDArray[np.int64]:
    """Fill ``mesher`` with an a x b x c grid of unit hexahedra and no other element."""
    i, j, k = np.meshgrid(
        np.arange(a + 1), np.arange(b + 1), np.arange(c + 1), indexing="ij"
    )
    points = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1).astype(np.float64)
    ids = mesher.add_nodes(points).reshape(a + 1, b + 1, c + 1)
    cells = [
        [
            ids[x, y, z],
            ids[x + 1, y, z],
            ids[x + 1, y + 1, z],
            ids[x, y + 1, z],
            ids[x, y, z + 1],
            ids[x + 1, y, z + 1],
            ids[x + 1, y + 1, z + 1],
            ids[x, y + 1, z + 1],
        ]
        for x in range(a)
        for y in range(b)
        for z in range(c)
    ]
    return mesher.add_elements(ElementType.HEXAHEDRON, np.array(cells))


def test_boundary_faces_of_a_hexahedral_grid_number_2_ab_bc_ca() -> None:
    """A 2 x 3 x 4 grid has 2 (ab + bc + ca) = 52 free facets: 52 quadrangles made.

    Spec (``SMESH_MeshEditor::MakeBoundaryMesh``, BND_2DFROM3D): a 2-D element on each
    free facet of the volumes. A second call makes nothing: the faces exist.
    """
    a, b, c = GRID
    with Mesher() as mesher:
        _hexa_grid(mesher, a, b, c)

        made = mesher.make_boundary_mesh(BoundaryDimension.FACES_OF_VOLUMES)

        again = mesher.make_boundary_mesh(BoundaryDimension.FACES_OF_VOLUMES)
        mesh = mesher.mesh()
    assert made.size == 2 * (a * b + b * c + c * a)
    assert again.size == 0
    assert mesh.count_of(ElementType.QUADRANGLE) == made.size


def test_boundary_edges_of_a_hexahedral_grid_are_the_edges_of_its_surface() -> None:
    """The edges on the grid's surface: (p+1) q + p (q+1) per box face p x q, less the
    4 (a + b + c) edges on the box's 12 sides, which two faces share.

    Spec (BND_1DFROM3D): a 1-D element on each edge of each free facet.
    """
    a, b, c = GRID

    def on_face(p: int, q: int) -> int:
        return (p + 1) * q + p * (q + 1)

    expected = 2 * (on_face(a, b) + on_face(b, c) + on_face(a, c)) - 4 * (a + b + c)
    with Mesher() as mesher:
        _hexa_grid(mesher, a, b, c)

        made = mesher.make_boundary_mesh(BoundaryDimension.EDGES_OF_VOLUMES)

    assert made.size == expected


def test_all_elements_puts_a_face_on_every_facet_shared_or_free() -> None:
    """Every distinct facet of the grid: a b (c+1) + a (b+1) c + (a+1) b c = 98."""
    a, b, c = GRID
    with Mesher() as mesher:
        _hexa_grid(mesher, a, b, c)

        made = mesher.make_boundary_mesh(
            BoundaryDimension.FACES_OF_VOLUMES, all_elements=True
        )

    assert made.size == a * b * (c + 1) + a * (b + 1) * c + (a + 1) * b * c


def test_the_boundary_of_one_cell_as_a_set_is_its_six_facets() -> None:
    """One interior-touching cell, ``around_elements``: its 6 facets, shared or not."""
    a, b, c = GRID
    with Mesher() as mesher:
        cells = _hexa_grid(mesher, a, b, c)

        made = mesher.make_boundary_mesh(
            BoundaryDimension.FACES_OF_VOLUMES,
            elements=[int(cells[0])],
            around_elements=True,
        )

    assert made.size == 6


def test_boundary_edges_of_a_quadrangle_grid_number_2_p_plus_q() -> None:
    """A 3 x 5 grid of quadrangles has 2 (3 + 5) = 16 free edges (BND_1DFROM2D)."""
    p, q = 3, 5
    with Mesher() as mesher:
        i, j = np.meshgrid(np.arange(p + 1), np.arange(q + 1), indexing="ij")
        points = np.stack([i.ravel(), j.ravel(), 0 * i.ravel()], axis=1)
        ids = mesher.add_nodes(points.astype(np.float64)).reshape(p + 1, q + 1)
        quads = [
            [ids[x, y], ids[x + 1, y], ids[x + 1, y + 1], ids[x, y + 1]]
            for x in range(p)
            for y in range(q)
        ]
        mesher.add_quadrangles(np.array(quads))

        made = mesher.make_boundary_mesh(BoundaryDimension.EDGES_OF_FACES)

    assert made.size == 2 * (p + q)


def test_a_face_given_to_a_volume_boundary_is_refused() -> None:
    """Degenerate input: faces of volumes read volumes only, every one is checked."""
    with Mesher() as mesher:
        nodes = mesher.add_nodes(np.eye(3, dtype=np.float64))
        (face,) = mesher.add_triangles(nodes[None, :])

        with pytest.raises(PysmeshError, match="not one of the volumes"):
            mesher.make_boundary_mesh(
                BoundaryDimension.FACES_OF_VOLUMES, elements=[int(face)]
            )


# ---- W3.3 ray_volumes ------------------------------------------------------------- #


def _block_mesh() -> tuple[Mesher, ps.MeshData]:
    """The 3 x 3 x 3 Hexa3D block of the 3 x 7 x 11 box, still open."""
    mesher = Mesher(_box_shape())
    mesher.assign(Regular1D())
    mesher.assign(NumberOfSegments(count=3))
    mesher.assign(Quadrangle2D())
    mesher.assign(Hexa3D())
    mesher.compute()
    return mesher, mesher.mesh()


def _cell_at(mesh: ps.MeshData, point: tuple[float, float, float]) -> int:
    """The id of the hexahedron of the block whose box holds ``point``."""
    for i in np.flatnonzero(mesh.element_type == int(ElementType.HEXAHEDRON)):
        p = _xyz_of(mesh, int(i))
        if np.all(p.min(axis=0) < point) and np.all(point < p.max(axis=0)):
            return int(mesh.element_id[i])
    raise AssertionError(f"no cell at {point}")


def test_a_ray_along_x_enters_and_leaves_each_cell_at_its_planes() -> None:
    """From x = -1 along +x at y = z = 1: entered at t = 1, 2, 3, left at 2, 3, 4.

    Spec (``SMESH_MeshAlgos::IntersectRayVolume``): tMin and tMax are where the ray
    meets the cell's near and far facets. The block's cells are 1 wide in x, so the
    planes x = 0, 1, 2, 3 are at t = x + 1.
    """
    mesher, mesh = _block_mesh()
    with mesher:
        hits = mesher.ray_volumes((-1.0, 1.0, 1.0), (1.0, 0.0, 0.0))

    expected = [_cell_at(mesh, (x + 0.5, 1.0, 1.0)) for x in range(3)]
    assert hits.ids.tolist() == expected
    np.testing.assert_allclose(hits.entry, [1.0, 2.0, 3.0], atol=TOL)
    np.testing.assert_allclose(hits.exit, [2.0, 3.0, 4.0], atol=TOL)
    assert np.all(hits.facet_entry >= 0)
    assert np.all(hits.facet_exit >= 0)


def test_a_ray_from_inside_a_cell_and_cut_short_reports_what_it_reaches() -> None:
    """From x = 1.5 with length 1: its own cell (entry -0.5, exit 0.5) and the next
    one, left at the ray's end (exit 1.0, no exit facet); the cell behind is not met."""
    mesher, mesh = _block_mesh()
    with mesher:
        hits = mesher.ray_volumes((1.5, 1.0, 1.0), (1.0, 0.0, 0.0), length=1.0)

    assert hits.ids.tolist() == [
        _cell_at(mesh, (1.5, 1.0, 1.0)),
        _cell_at(mesh, (2.5, 1.0, 1.0)),
    ]
    np.testing.assert_allclose(hits.entry, [-0.5, 0.5], atol=TOL)
    np.testing.assert_allclose(hits.exit, [0.5, 1.0], atol=TOL)
    assert hits.facet_exit.tolist()[1] == -1


def test_a_ray_volume_query_refuses_a_zero_length() -> None:
    """Degenerate input: a ray of length 0 meets nothing and is refused."""
    mesher, _ = _block_mesh()
    with mesher, pytest.raises(PysmeshError, match="length must be > 0"):
        mesher.ray_volumes((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), length=0.0)


# ---- W3.4 Warping3D and W3.5 ScaledJacobian --------------------------------------- #


def _one_cell(points: list[tuple[float, float, float]], kind: ElementType) -> Mesher:
    """A mesher holding one cell of ``kind`` on ``points``, in that node order."""
    mesher = Mesher()
    nodes = mesher.add_nodes(np.array(points, dtype=np.float64))
    mesher.add_elements(kind, nodes[None, :])
    return mesher


@pytest.mark.parametrize("z", [0.25, 1.0])
def test_warping_3d_of_a_hexahedron_with_a_saddle_top_is_atan_z(z: float) -> None:
    """Top corners at heights z, -z, z, -z over a 2 x 2 square: warping atan(z) degrees.

    Spec (SMESH ``warping.rst``): the warp angle of a quadrangle is the arcsine of the
    corner height ``h`` over the plane through its edge midpoints, divided by the half
    edge ``l``; ``Warping3D`` (``warping_3d.rst``) takes the largest over the facets. On
    this saddle every corner has ``h = z`` and ``l = sqrt(1 + z^2)``, so the angle is
    ``asin(z / sqrt(1 + z^2)) = atan(z)``. The four side facets lie in the planes
    x = +-1 and y = +-1 and the bottom is flat, so they read 0.
    """
    bottom = [
        (-1.0, -1.0, -2.0),
        (1.0, -1.0, -2.0),
        (1.0, 1.0, -2.0),
        (-1.0, 1.0, -2.0),
    ]
    top = [(-1.0, -1.0, z), (1.0, -1.0, -z), (1.0, 1.0, z), (-1.0, 1.0, -z)]

    with _one_cell(bottom + top, ElementType.HEXAHEDRON) as mesher:
        result = mesher.quality(Warping3D())

    assert result.values.tolist() == pytest.approx([math.degrees(math.atan(z))])
    assert result.skipped == 0


def test_warping_3d_skips_a_tetrahedron_which_has_no_quadrangle_facet() -> None:
    """A tetrahedron has no 4-node facet, so no warping: skipped, not read as 0."""
    corners = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

    with _one_cell(corners, ElementType.TETRAHEDRON) as mesher:
        result = mesher.quality(Warping3D())

    assert result.values.size == 0
    assert result.skipped == 1


@pytest.mark.parametrize("shear", [0.0, 0.5, 2.0])
def test_scaled_jacobian_of_a_sheared_block_is_the_cosine_of_its_shear(
    shear: float,
) -> None:
    """A unit square extruded along (s, 0, 1): every cell reads 1 / sqrt(1 + s^2).

    Spec (SMESH ``scaled_jacobian.rst``, ``SMDS_VolumeTool::GetScaledJacobian``): at
    each corner the determinant of the unit edge vectors, the smallest reported. The
    cells of the extruded block are parallelepipeds with edges (1, 0, 0), (0, 1, 0) and
    (s, 0, 1), whose unit vectors span the volume ``1 / sqrt(1 + s^2)``; s = 0 is the
    right-angled cell, which reads 1.
    """
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0)
    session.extrude(list(session.entities(ps.EntityKind.FACE)), (shear, 0.0, 1.0))

    with Mesher(ps.load_brep(session.brep())) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.compute()
        result = mesher.quality(ScaledJacobian())

    assert result.values.size == 8
    expected = 1.0 / math.sqrt(1.0 + shear**2)
    np.testing.assert_allclose(result.values, expected, rtol=1e-12)


def test_scaled_jacobian_of_a_regular_tetrahedron_is_one() -> None:
    """A regular tetrahedron reads 1 in magnitude (``scaled_jacobian.rst``)."""
    corners = [(1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)]

    with _one_cell(corners, ElementType.TETRAHEDRON) as mesher:
        result = mesher.quality(ScaledJacobian())

    assert abs(float(result.values[0])) == pytest.approx(1.0, rel=1e-12)


# ---- W3.7 smoothing in the parameter space of a periodic face --------------------- #


@pytest.mark.parametrize("method", [SmoothMethod.LAPLACIAN, SmoothMethod.CENTROIDAL])
def test_smoothing_a_uniform_cylinder_mesh_across_its_seam_moves_nothing(
    method: SmoothMethod,
) -> None:
    """A uniform 12 x 12 mapped mesh of a cylinder's side is a fixed point: it stays.

    Every interior node of a uniform grid is the average of its neighbours in (u, v),
    and the centre of its cells, so either smoother leaves it in place, provided the
    neighbours across the seam u = 0 = 2 pi are taken on the near side (SMESH 9.16,
    ``42e25f073``, "Laplacian Smoothing 2D"). Every node stays on the radius-1 surface.
    """
    session = Session()
    session.add_cylinder(1.0, 2.0)
    shape = ps.load_brep(session.brep())
    side = next(
        f.id
        for f in shape.faces()
        if abs(f.bbox[2]) < TOL and abs(f.bbox[5] - 2.0) < TOL
    )

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=12))
        mesher.assign(Quadrangle2D(), on=SubShape(SubShapeKind.FACE, side))
        mesher.compute()
        before = mesher.mesh()
        mesher.smooth(method=method, iterations=3, on_shape=True)
        after = mesher.mesh()

    xyz_before = before.node_coords[np.argsort(before.node_id)]
    xyz_after = after.node_coords[np.argsort(after.node_id)]
    assert np.abs(xyz_after - xyz_before).max() < 1e-12
    np.testing.assert_allclose(
        np.hypot(xyz_after[:, 0], xyz_after[:, 1]), 1.0, atol=TOL
    )


# ---- E6 MaxElementArea on Mefisto2D ----------------------------------------------- #

MEFISTO_BOUNDS: tuple[float, ...] = (4.0, 1.0, 0.25, 0.0625)


def _triangle_areas(mesh: ps.MeshData) -> NDArray[np.float64]:
    """The area of each triangle of ``mesh``."""
    rows = [
        mesh.nodes_of(i)
        for i in range(mesh.element_count)
        if int(mesh.element_type[i]) == int(ElementType.TRIANGLE)
    ]
    tri = np.asarray(rows, dtype=np.int64)
    p0, p1, p2 = (mesh.node_coords[tri[:, k]] for k in range(3))
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def _mefisto(
    shape: ps.Shape, edge_sizing: Hypothesis, *face_sizing: Hypothesis
) -> tuple[ps.MeshData, ps.ComputeReport]:
    """MEFISTO on ``shape``: Regular1D with ``edge_sizing``, then ``face_sizing``."""
    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(edge_sizing)
        mesher.assign(Mefisto2D())
        for hypothesis in face_sizing:
            mesher.assign(hypothesis)
        report = mesher.compute()
        return mesher.mesh(), report


def test_max_element_area_bounds_every_mefisto_triangle_as_it_tightens() -> None:
    """The 4 x 4 square, 8 segments per side, max_area 4, 1, 0.25 and 0.0625.

    Spec (SMESH ``2d_meshing_hypo.rst``, "Max Element Area"): the hypothesis sets "the
    maximum area of mesh faces". So every triangle has an area of at most A, triangles
    that fill the area 16 number at least 16 / A, and a smaller A never gives fewer
    triangles. The bounds 0.25 and 0.0625 ask for triangles at and below the boundary
    segment of 0.5, where MEFISTO used to clamp the bound back to the boundary size.
    """
    counts = []
    reports = []
    for bound in MEFISTO_BOUNDS:
        mesh, report = _mefisto(
            _square_shape(), NumberOfSegments(count=8), MaxElementArea(max_area=bound)
        )

        areas = _triangle_areas(mesh)
        assert areas.max() <= bound * (1.0 + TOL), bound
        assert areas.size >= SQUARE_SIDE**2 / bound, bound
        assert areas.sum() == pytest.approx(SQUARE_SIDE**2, rel=TOL)
        counts.append(int(areas.size))
        reports.append(report)
    assert counts == sorted(counts)
    assert all(report.warnings == () for report in reports)


@pytest.mark.parametrize("bound", [1.0, 4.0])
def test_a_max_element_area_the_boundary_already_meets_leaves_the_mesh_alone(
    bound: float,
) -> None:
    """The 3 x 7 x 11 box with LocalLength(1.0): the boundary-sized mesh has every
    triangle within 1.0, so a bound of 1.0 or 4.0 changes nothing.

    A bound limits the triangles; it is not a size to grow them to. MEFISTO used to keep
    an edge bound up to 2.05 times the longest boundary segment, so MaxElementArea(1.0)
    grew the interior past the 1 m boundary: 440 triangles, the largest of area 1.77,
    slivers of 0.187 degrees, against 1136 triangles with MaxElementArea(4.0).
    """
    plain, _ = _mefisto(_box_shape(), LocalLength(length=1.0))

    mesh, report = _mefisto(
        _box_shape(), LocalLength(length=1.0), MaxElementArea(max_area=bound)
    )

    np.testing.assert_array_equal(
        np.sort(mesh.node_coords, axis=0), np.sort(plain.node_coords, axis=0)
    )
    assert _triangle_areas(mesh).max() <= bound
    assert report.warnings == ()


def test_a_max_element_area_the_boundary_cannot_meet_is_a_warning() -> None:
    """The 4 x 4 square with 2 segments per side and max_area 0.25.

    A triangle on a boundary segment of length 2 has an area of 0.25 only if its third
    node lies within 0.25 of that segment; MEFISTO's triangles there are larger. The
    mesh is kept, and the compute reports a warning on the face, naming the bound and
    the largest area, which does exceed the bound.
    """
    bound = 0.25

    mesh, report = _mefisto(
        _square_shape(), NumberOfSegments(count=2), MaxElementArea(max_area=bound)
    )

    largest = float(_triangle_areas(mesh).max())
    assert largest > bound
    assert _triangle_areas(mesh).sum() == pytest.approx(SQUARE_SIDE**2, rel=TOL)
    assert len(report.warnings) == 1
    warning = report.warnings[0]
    assert (warning.kind, warning.ordinal, warning.algorithm) == (
        SubShapeKind.FACE,
        1,
        "MEFISTO_2D",
    )
    assert f"MaxElementArea {bound:g} is not met" in warning.text
    assert f"{largest:.6g}" in warning.text
