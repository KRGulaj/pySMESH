# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-05

"""NETGEN through pySMESH: the mesh oracles of every NETGEN recipe, on the test shapes.

Each test asserts a fact that any correct mesh of the shape has, never a stored NETGEN
output, so the same tests hold for a later netgen version:

1. Conformity: every face of a volume element lies on one or two volume elements; the
   faces that lie on one are 2-D elements of the mesh; every node is used; no two nodes
   coincide.
2. Validity: SMESH finds no badly oriented volume (``BadOrientedVolume``), every
   tetrahedron has a positive volume in netgen's own orientation (``Element::Volume`` of
   ``meshtype.cpp``: ``-((p2-p1) x (p3-p1)) . (p4-p1) / 6``, which NETGENPlugin keeps
   for SMDS), and no tetrahedron is a sliver: the smallest dihedral angle is above 2
   degrees.
3. Geometry: every node on a face lies on that face, and every 2-D element on the
   boundary faces away from the volume element it bounds.
4. Measure: the volume elements fill the CAD volume (``Session.mass_properties``). A
   solid with planar faces is filled exactly, to 1e-9 relative. Where a face is curved,
   its nodes lie on it and its triangles are chords: a triangle whose edges are at most
   ``L`` long lies within the sagitta ``L^2 / (8 R)`` of a surface whose smallest radius
   of curvature is ``R``, so the volume differs by at most the curved area times that
   sagitta. On a convex solid the mesh is inside: it never exceeds the CAD volume.
5. Size: a segment spans at most 1.5 local sizes, because netgen divides an edge into
   ``max(1, floor(I + 0.5))`` segments where ``I`` is the integral of ``1/h`` along it
   (``occgenmesh.cpp:272``). Away from a restricted point the local size grows at most
   linearly, ``h(d) <= h0 + g d`` with ``g`` the growth rate (``localh.cpp``, ``SetH``),
   so a segment that starts where ``h = h0`` is at most ``h0 (exp(1.5 g) - 1) / g``
   long: 1.89 ``h0`` at the default ``g = 0.3``. netgen gives no such bound for the
   edges of triangles and tetrahedra, whose fronts aim at the local size and whose
   improvement passes trade size for shape; there the mean edge is at most 1.5
   ``max_size``, and no edge exceeds 4 ``max_size`` (a mesh that ignored ``max_size``
   has edges of the body's size).
6. Sub-mesh semantics: where another algorithm meshed an edge or a face first, NETGEN
   meshes on those nodes and elements, with the same ids.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    BadOrientedVolume,
    ElementType,
    EntityKind,
    Fineness,
    LocalLength,
    MaxElementVolume,
    MeshData,
    Mesher,
    Netgen1D2D,
    Netgen1D2D3D,
    Netgen2D,
    Netgen3D,
    NetgenParameters,
    NetgenParameters2D,
    NetgenSimpleParameters2D,
    NetgenSimpleParameters3D,
    NumberOfSegments,
    PysmeshError,
    Quadrangle2D,
    Regular1D,
    Session,
    SubShape,
    SubShapeKind,
)

# Corner node counts and faces (as corner positions) of the volume elements NETGEN
# makes, in SMDS order; a quadratic element lists its corners first.
_VOLUME_FACES: dict[int, tuple[tuple[int, ...], ...]] = {
    int(ElementType.TETRAHEDRON): ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    int(ElementType.QUAD_TETRAHEDRON): ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    int(ElementType.PYRAMID): (
        (0, 1, 2, 3),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ),
}
_CORNERS: dict[int, int] = {
    int(ElementType.EDGE): 2,
    int(ElementType.QUAD_EDGE): 2,
    int(ElementType.TRIANGLE): 3,
    int(ElementType.QUAD_TRIANGLE): 3,
    int(ElementType.QUADRANGLE): 4,
    int(ElementType.TETRAHEDRON): 4,
    int(ElementType.QUAD_TETRAHEDRON): 4,
    int(ElementType.PYRAMID): 5,
}
_DIMENSION: dict[int, int] = {
    int(ElementType.EDGE): 1,
    int(ElementType.QUAD_EDGE): 1,
    int(ElementType.TRIANGLE): 2,
    int(ElementType.QUAD_TRIANGLE): 2,
    int(ElementType.QUADRANGLE): 2,
    int(ElementType.TETRAHEDRON): 3,
    int(ElementType.QUAD_TETRAHEDRON): 3,
    int(ElementType.PYRAMID): 3,
}
_SLIVER_DEGREES = 2.0
# The longest segment that starts where the local size is h0, over h0, at the default
# growth rate 0.3: (exp(1.5 g) - 1) / g (oracle 5 of the module docstring).
_GRADED_SEGMENT = (math.exp(1.5 * 0.3) - 1.0) / 0.3


# ---- The test shapes --------------------------------------------------------------- #


@dataclass(frozen=True)
class Case:
    """A test solid, its CAD volume and the curvature data of its curved faces.

    Attributes:
        name: The case id.
        shape: The shape to mesh.
        volume: The CAD volume, from ``Session.mass_properties``.
        max_size: The ``max_size`` the recipes mesh it with.
        curved_area: The area of the curved faces; 0 for a solid with planar faces.
        min_radius: The smallest radius of curvature of the curved faces.
        convex: Whether the solid is convex, so an inscribed mesh lies inside it.
    """

    name: str
    shape: ps.Shape
    volume: float
    max_size: float
    curved_area: float = 0.0
    min_radius: float = math.inf
    convex: bool = True


def _distinct_volume(s: Session) -> float:
    """The CAD volume of the session's solids, each shape once."""
    solids = s.entities(EntityKind.SOLID, distinct=True).tolist()
    return float(s.mass_properties(solids).measure.sum())


def _make_case(name: str) -> Case:
    """Build one test solid."""
    s = Session()
    if name == "box":
        s.add_box(1.0, 2.0, 3.0)
        return Case(name, ps.load_brep(s.brep()), _distinct_volume(s), 0.6)
    if name == "cylinder":
        s.add_cylinder(1.0, 3.0)
        return Case(
            name, ps.load_brep(s.brep()), _distinct_volume(s), 0.4, 6.0 * math.pi, 1.0
        )
    if name == "sphere":
        s.add_sphere(1.0)
        return Case(
            name, ps.load_brep(s.brep()), _distinct_volume(s), 0.4, 4.0 * math.pi, 1.0
        )
    if name == "torus":
        s.add_torus(3.0, 1.0)
        area = 4.0 * math.pi**2 * 3.0 * 1.0
        return Case(
            name, ps.load_brep(s.brep()), _distinct_volume(s), 0.6, area, 1.0, False
        )
    if name == "partitioned":
        # A 2 x 1 x 1 box fragmented with the unit box inside it: two solids that share
        # the internal face x = 1.
        s.add_box(2.0, 1.0, 1.0)
        s.add_box(1.0, 1.0, 1.0)
        s.fragment(s.entities(EntityKind.SOLID).tolist())
        return Case(name, ps.load_brep(s.brep()), _distinct_volume(s), 0.4)
    if name == "stacked":
        # A unit box on top of a 2 x 2 x 1 box: the shared face is part of the larger
        # top.
        s.add_box(2.0, 2.0, 1.0)
        s.add_box(1.0, 1.0, 1.0, origin=(0.5, 0.5, 1.0))
        s.fragment(s.entities(EntityKind.SOLID).tolist())
        return Case(
            name, ps.load_brep(s.brep()), _distinct_volume(s), 0.4, convex=False
        )
    if name == "fillet":
        # A 2 x 2 x 2 box with one edge rounded at radius 0.5: a quarter cylinder face.
        s.add_box(2.0, 2.0, 2.0)
        s.fillet(s.entities(EntityKind.EDGE).tolist()[:1], 0.5)
        area = 0.5 * math.pi * 0.5 * 2.0
        return Case(name, ps.load_brep(s.brep()), _distinct_volume(s), 0.4, area, 0.5)
    raise ValueError(name)


_SOLIDS = ("box", "cylinder", "sphere", "torus", "partitioned", "stacked", "fillet")
_RECIPES = ("1d2d3d", "1d2d+3d", "regular+2d+3d", "quadrangle+3d")
# Quadrangle2D needs four-sided faces: the box and the partitioned box. Regular1D puts
# nodes on the zero-length edges of the sphere's poles, by design (Quadrangle2D needs
# them on a degenerated side), so a sphere meshed from Regular1D has coincident nodes at
# the poles whatever meshes the face; that recipe is left out for the sphere.
_RUNS = [
    (solid, recipe)
    for solid in _SOLIDS
    for recipe in _RECIPES
    if (recipe != "quadrangle+3d" or solid in ("box", "partitioned"))
    and (recipe != "regular+2d+3d" or solid != "sphere")
]


def _assign(mesher: Mesher, recipe: str, max_size: float) -> None:
    """Assign one NETGEN recipe to the whole shape."""
    if recipe == "1d2d3d":
        mesher.assign(Netgen1D2D3D())
        mesher.assign(NetgenParameters(max_size=max_size))
    elif recipe == "1d2d+3d":
        mesher.assign(Netgen1D2D())
        mesher.assign(NetgenParameters2D(max_size=max_size))
        mesher.assign(Netgen3D())
        mesher.assign(NetgenParameters(max_size=max_size))
    elif recipe == "regular+2d+3d":
        mesher.assign(Regular1D())
        mesher.assign(LocalLength(length=max_size))
        mesher.assign(Netgen2D())
        mesher.assign(Netgen3D())
        mesher.assign(NetgenParameters(max_size=max_size))
    else:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Quadrangle2D())
        mesher.assign(Netgen3D())
        mesher.assign(NetgenParameters(max_size=max_size))


@dataclass(frozen=True)
class Run:
    """One computed recipe on one case.

    Attributes:
        case: The test solid.
        recipe: The recipe name.
        mesh: The mesh.
        bad_oriented: How many volumes SMESH's BadOrientedVolume selects.
    """

    case: Case
    recipe: str
    mesh: MeshData
    bad_oriented: int


@pytest.fixture(scope="module")
def runs() -> Iterator[dict[tuple[str, str], Run]]:
    """Every recipe computed once on every case it applies to."""
    cases = {name: _make_case(name) for name in _SOLIDS}
    out: dict[tuple[str, str], Run] = {}
    for solid, recipe in _RUNS:
        case = cases[solid]
        with Mesher(case.shape) as mesher:
            _assign(mesher, recipe, case.max_size)
            mesher.compute()
            bad = int(mesher.select(BadOrientedVolume()).ids.size)
            out[(solid, recipe)] = Run(case, recipe, mesher.mesh(), bad)
    yield out


# ---- Mesh helpers ------------------------------------------------------------------ #


def _elements(mesh: MeshData, dimension: int) -> list[int]:
    """Positions of the elements of one dimension."""
    return [
        e
        for e in range(mesh.element_count)
        if _DIMENSION.get(int(mesh.element_type[e])) == dimension
    ]


def _corners(mesh: MeshData, e: int) -> NDArray[np.int32]:
    """The corner rows of one element."""
    return mesh.nodes_of(e)[: _CORNERS[int(mesh.element_type[e])]]


def _volume_faces(mesh: MeshData) -> Counter[frozenset[int]]:
    """How many volume elements each volume face (a set of node rows) lies on."""
    uses: Counter[frozenset[int]] = Counter()
    for e in _elements(mesh, 3):
        c = _corners(mesh, e)
        for face in _VOLUME_FACES[int(mesh.element_type[e])]:
            uses[frozenset(int(c[i]) for i in face)] += 1
    return uses


def _edge_lengths(mesh: MeshData, dimension: int) -> NDArray[np.float64]:
    """Every corner-to-corner edge length of the elements of one dimension."""
    xyz = mesh.node_coords
    out: list[float] = []
    for e in _elements(mesh, dimension):
        c = _corners(mesh, e)
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                out.append(float(np.linalg.norm(xyz[c[i]] - xyz[c[j]])))
    return np.asarray(out, dtype=np.float64)


def _tetra_volumes(mesh: MeshData) -> NDArray[np.float64]:
    """The netgen-oriented volume of every tetrahedron, from its corners."""
    xyz = mesh.node_coords
    out: list[float] = []
    for e in _elements(mesh, 3):
        if int(mesh.element_type[e]) not in (
            int(ElementType.TETRAHEDRON),
            int(ElementType.QUAD_TETRAHEDRON),
        ):
            continue
        a, b, c, d = xyz[_corners(mesh, e)]
        out.append(-float(np.dot(np.cross(b - a, c - a), d - a)) / 6.0)
    return np.asarray(out, dtype=np.float64)


def _total_volume(mesh: MeshData) -> float:
    """The volume of every volume element, a pyramid as two tetrahedra."""
    xyz = mesh.node_coords
    total = 0.0
    for e in _elements(mesh, 3):
        c = _corners(mesh, e)
        if int(mesh.element_type[e]) == int(ElementType.PYRAMID):
            pieces = ((c[0], c[1], c[2], c[4]), (c[0], c[2], c[3], c[4]))
        else:
            pieces = ((c[0], c[1], c[2], c[3]),)
        for p in pieces:
            a, b, cc, d = xyz[list(p)]
            total += abs(float(np.dot(np.cross(b - a, cc - a), d - a))) / 6.0
    return total


def _min_dihedral(mesh: MeshData) -> float:
    """The smallest interior dihedral angle of the tetrahedra, in degrees."""
    xyz = mesh.node_coords
    smallest = 180.0
    pairs = (
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
    )
    for e in _elements(mesh, 3):
        if int(mesh.element_type[e]) not in (
            int(ElementType.TETRAHEDRON),
            int(ElementType.QUAD_TETRAHEDRON),
        ):
            continue
        c = _corners(mesh, e)
        for i, j, k, m in pairs:
            p, q, r, w = xyz[c[i]], xyz[c[j]], xyz[c[k]], xyz[c[m]]
            edge = q - p
            u = r - p - np.dot(r - p, edge) / np.dot(edge, edge) * edge
            v = w - p - np.dot(w - p, edge) / np.dot(edge, edge) * edge
            cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            smallest = min(
                smallest, math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
            )
    return smallest


# ---- Oracle 1: conformity ---------------------------------------------------------- #


@pytest.mark.parametrize(("solid", "recipe"), _RUNS)
def test_every_netgen_recipe_meshes_every_test_solid_conformingly(
    runs: dict[tuple[str, str], Run], solid: str, recipe: str
) -> None:
    """Each volume face lies on 1 or 2 volume elements, the faces on one are 2-D
    elements,
    every node is used, and no two nodes coincide."""
    mesh = runs[(solid, recipe)].mesh
    uses = _volume_faces(mesh)
    surface = {frozenset(int(n) for n in _corners(mesh, e)) for e in _elements(mesh, 2)}
    used = np.zeros(mesh.node_count, dtype=bool)
    for e in range(mesh.element_count):
        used[mesh.nodes_of(e)] = True
    xyz = mesh.node_coords
    scale = float(np.ptp(xyz, axis=0).max())
    order = np.lexsort(xyz.T)
    gaps = np.linalg.norm(np.diff(xyz[order], axis=0), axis=1)

    assert uses
    assert max(uses.values()) <= 2
    assert {face for face, n in uses.items() if n == 1} <= surface
    assert bool(used.all())
    assert float(gaps.min()) > 1e-9 * scale


def test_a_shared_face_is_meshed_once_and_bounds_both_solids(
    runs: dict[tuple[str, str], Run],
) -> None:
    """The face both solids of the partitioned box share lies on two volumes."""
    mesh = runs[("partitioned", "1d2d3d")].mesh
    uses = _volume_faces(mesh)
    internal = [
        frozenset(int(n) for n in _corners(mesh, e))
        for e in _elements(mesh, 2)
        if np.allclose(mesh.node_coords[_corners(mesh, e)][:, 0], 1.0)
    ]

    assert internal
    assert all(uses[face] == 2 for face in internal)


# ---- Oracle 2: validity ------------------------------------------------------------ #


@pytest.mark.parametrize(("solid", "recipe"), _RUNS)
def test_every_netgen_volume_element_is_valid_and_no_tetrahedron_is_a_sliver(
    runs: dict[tuple[str, str], Run], solid: str, recipe: str
) -> None:
    """No badly oriented volume, positive tetrahedra, no dihedral at or below 2 deg."""
    run = runs[(solid, recipe)]

    volumes = _tetra_volumes(run.mesh)
    smallest = _min_dihedral(run.mesh)

    assert run.bad_oriented == 0
    assert volumes.size > 0
    assert float(volumes.min()) > 0.0
    assert smallest > _SLIVER_DEGREES, smallest


# ---- Oracle 3: geometry ------------------------------------------------------------ #


@pytest.mark.parametrize(("solid", "recipe"), _RUNS)
def test_every_face_node_lies_on_its_face_and_the_boundary_faces_outward(
    runs: dict[tuple[str, str], Run], solid: str, recipe: str
) -> None:
    """Face nodes lie on their face; boundary 2-D elements face out of the volume."""
    run = runs[(solid, recipe)]
    mesh = run.mesh
    xyz = mesh.node_coords
    scale = float(np.ptp(xyz, axis=0).max())
    distances = [0.0]
    for f in range(1, len(run.case.shape.faces()) + 1):
        rows = np.nonzero(
            (mesh.node_kind == int(SubShapeKind.FACE)) & (mesh.node_ordinal == f)
        )[0]
        if rows.size:
            distances.append(float(run.case.shape.face_distance(f, xyz[rows]).max()))
    apex: dict[frozenset[int], int] = {}
    for e in _elements(mesh, 3):
        c = _corners(mesh, e)
        for face in _VOLUME_FACES[int(mesh.element_type[e])]:
            key = frozenset(int(c[i]) for i in face)
            others = [int(c[i]) for i in range(len(c)) if i not in face]
            apex[key] = others[0]
    uses = _volume_faces(mesh)
    inward = 0
    for e in _elements(mesh, 2):
        c = _corners(mesh, e)
        key = frozenset(int(n) for n in c)
        if uses.get(key) != 1:
            continue
        normal = np.cross(xyz[c[1]] - xyz[c[0]], xyz[c[2]] - xyz[c[0]])
        if float(np.dot(normal, xyz[apex[key]] - xyz[c].mean(axis=0))) >= 0.0:
            inward += 1

    assert max(distances) <= 1e-9 * scale
    assert inward == 0


# ---- Oracle 4: measure ------------------------------------------------------------- #


@pytest.mark.parametrize(("solid", "recipe"), _RUNS)
def test_the_volume_elements_fill_the_cad_volume_within_the_chord_bound(
    runs: dict[tuple[str, str], Run], solid: str, recipe: str
) -> None:
    """Exact on planar faces; within curved area x L^2 / (8 R) on curved ones."""
    run = runs[(solid, recipe)]
    case = run.case

    meshed = _total_volume(run.mesh)
    deficit = case.volume - meshed

    if case.curved_area == 0.0:
        assert abs(deficit) <= 1e-9 * case.volume
        return
    longest = float(_edge_lengths(run.mesh, 2).max())
    bound = case.curved_area * longest**2 / (8.0 * case.min_radius)
    assert abs(deficit) <= bound, (deficit, bound)
    if case.convex:
        assert deficit >= -1e-12 * case.volume


# ---- Oracle 5: size ---------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("solid", "recipe"), [r for r in _RUNS if r[1] in ("1d2d3d", "1d2d+3d")]
)
def test_max_size_bounds_the_segments_and_the_elements(
    runs: dict[tuple[str, str], Run], solid: str, recipe: str
) -> None:
    """Segments at most 1.5 max_size; 2-D and 3-D edges at most 4, on mean 1.5."""
    run = runs[(solid, recipe)]
    h = run.case.max_size

    segments = _edge_lengths(run.mesh, 1)
    surface = _edge_lengths(run.mesh, 2)
    volume = _edge_lengths(run.mesh, 3)

    assert float(segments.max()) <= 1.5 * h * (1 + 1e-9)
    for lengths in (surface, volume):
        assert float(lengths.max()) <= 4.0 * h
        assert float(lengths.mean()) <= 1.5 * h


def _segments_on(mesh: MeshData, edge: int) -> NDArray[np.float64]:
    """The lengths of the segments bound to one edge."""
    xyz = mesh.node_coords
    out = [
        float(np.linalg.norm(xyz[_corners(mesh, e)[0]] - xyz[_corners(mesh, e)[1]]))
        for e in _elements(mesh, 1)
        if mesh.element_kind[e] == int(SubShapeKind.EDGE)
        and mesh.element_ordinal[e] == edge
    ]
    return np.asarray(out, dtype=np.float64)


def test_a_local_size_on_an_edge_bounds_its_segments() -> None:
    """A local size of 0.1 on edge 1 of a box meshed at 0.8: its segments are at most
    0.15.

    The plugin restricts the size to 0.1 at points spaced 0.1 / 1.5 along the edge
    (``setLocalSize``), so between them the size is at most 0.1 + 0.3 x 0.033 = 0.11;
    a segment spans at most 1.5 of it. The other edges keep the global size: their
    longest segment is above 0.4.
    """
    case = _make_case("box")
    edge = SubShape(SubShapeKind.EDGE, 1)
    with Mesher(case.shape) as mesher:
        mesher.assign(Netgen1D2D3D())
        mesher.assign(NetgenParameters(max_size=0.8, local_sizes=((edge, 0.1),)))
        mesher.compute()
        mesh = mesher.mesh()

    local = _segments_on(mesh, 1)
    elsewhere = max(
        float(_segments_on(mesh, e).max())
        for e in range(2, len(case.shape.edges()) + 1)
    )

    assert local.size > 0
    assert float(local.max()) <= 1.5 * 0.1 * (1.0 + 0.3 / 3.0)
    assert elsewhere > 0.4


def test_a_local_size_on_a_vertex_bounds_the_segments_that_touch_it() -> None:
    """A local size of 0.05 at vertex 1 of a box meshed at 0.8: each of the three
    segments
    at it is at most 1.89 x 0.05 long (the graded bound of oracle 5)."""
    case = _make_case("box")
    vertex = SubShape(SubShapeKind.VERTEX, 1)
    with Mesher(case.shape) as mesher:
        mesher.assign(Netgen1D2D3D())
        mesher.assign(NetgenParameters(max_size=0.8, local_sizes=((vertex, 0.05),)))
        mesher.compute()
        mesh = mesher.mesh()
    xyz = mesh.node_coords
    at_vertex = np.nonzero(
        (mesh.node_kind == int(SubShapeKind.VERTEX)) & (mesh.node_ordinal == 1)
    )[0]
    touching = [
        float(np.linalg.norm(xyz[_corners(mesh, e)[0]] - xyz[_corners(mesh, e)[1]]))
        for e in _elements(mesh, 1)
        if int(at_vertex[0]) in _corners(mesh, e).tolist()
    ]

    assert at_vertex.size == 1
    assert len(touching) == 3
    assert max(touching) <= _GRADED_SEGMENT * 0.05


# ---- Oracle 6: sub-mesh semantics -------------------------------------------------- #


def _nodes_and_elements(
    mesh: MeshData, dimension: int
) -> tuple[dict[int, tuple[float, ...]], set[tuple[int, ...]]]:
    """The nodes (id to coordinates) and the elements of one dimension."""
    nodes: dict[int, tuple[float, ...]] = {}
    elements: set[tuple[int, ...]] = set()
    for e in _elements(mesh, dimension):
        rows = mesh.nodes_of(e)
        ids = tuple(int(mesh.node_id[r]) for r in rows)
        elements.add((int(mesh.element_id[e]), *ids))
        for r, i in zip(rows, ids, strict=True):
            nodes[i] = tuple(float(x) for x in mesh.node_coords[r])
    return nodes, elements


@pytest.mark.parametrize("solid", ["box", "cylinder", "partitioned"])
def test_netgen_2d_and_3d_mesh_on_the_segments_of_regular_1d(solid: str) -> None:
    """Edges that Regular1D meshed first keep their nodes and segments: ids, places."""
    case = _make_case(solid)
    with Mesher(case.shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=5))
        mesher.compute()
        before = _nodes_and_elements(mesher.mesh(), 1)
        mesher.assign(Netgen2D())
        mesher.assign(Netgen3D())
        mesher.assign(NetgenParameters(max_size=case.max_size))
        mesher.compute()
        mesh = mesher.mesh()

    after = _nodes_and_elements(mesh, 1)
    uses = _volume_faces(mesh)
    surface = {frozenset(int(n) for n in _corners(mesh, e)) for e in _elements(mesh, 2)}

    assert after == before
    assert max(uses.values()) <= 2
    assert {face for face, n in uses.items() if n == 1} <= surface


@pytest.mark.parametrize("solid", ["box", "partitioned"])
def test_netgen_3d_meshes_on_the_quadrangles_of_quadrangle_2d(solid: str) -> None:
    """Faces that Quadrangle2D meshed first keep their quads; pyramids close them."""
    case = _make_case(solid)
    with Mesher(case.shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Quadrangle2D())
        mesher.compute()
        before = _nodes_and_elements(mesher.mesh(), 2)
        mesher.assign(Netgen3D())
        mesher.assign(NetgenParameters(max_size=case.max_size))
        mesher.compute()
        mesh = mesher.mesh()

    after = _nodes_and_elements(mesh, 2)
    uses = _volume_faces(mesh)
    quads = [frozenset(int(n) for n in _corners(mesh, e)) for e in _elements(mesh, 2)]
    pyramids = Counter(int(t) for t in mesh.element_type)[int(ElementType.PYRAMID)]

    assert after == before
    assert all(uses[q] in (1, 2) for q in quads)
    assert pyramids == sum(uses[q] for q in quads)
    assert abs(_total_volume(mesh) - case.volume) <= 1e-9 * case.volume


def test_netgen_1d2d3d_meshes_on_a_local_regular_1d_edge() -> None:
    """Regular1D with 7 segments on edge 1 beside Netgen1D2D3D: 7 segments there."""
    case = _make_case("box")
    edge = SubShape(SubShapeKind.EDGE, 1)
    with Mesher(case.shape) as mesher:
        mesher.assign(Netgen1D2D3D())
        mesher.assign(NetgenParameters(max_size=case.max_size))
        mesher.assign(Regular1D(), on=edge)
        mesher.assign(NumberOfSegments(count=7), on=edge)
        mesher.compute()
        mesh = mesher.mesh()

    uses = _volume_faces(mesh)
    surface = {frozenset(int(n) for n in _corners(mesh, e)) for e in _elements(mesh, 2)}

    assert _segments_on(mesh, 1).size == 7
    assert max(uses.values()) <= 2
    assert {face for face, n in uses.items() if n == 1} <= surface
    assert abs(_total_volume(mesh) - case.volume) <= 1e-9 * case.volume


# ---- A face with an internal edge and an internal vertex --------------------------- #


def _face_with_internal_edge_and_vertex() -> tuple[ps.Shape, float]:
    """A 4 x 2 rectangle with an internal edge and an internal vertex imprinted."""
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 4.0, 2.0)
    face = s.entities(EntityKind.FACE).tolist()
    s.add_line((1.0, 0.5, 0.0), (3.0, 1.5, 0.0))
    line = s.entities(EntityKind.EDGE).tolist()[-1:]
    s.add_vertex((1.0, 1.5, 0.0))
    point = s.entities(EntityKind.VERTEX).tolist()[-1:]
    s.imprint(face, line + point)
    s.remove(line + point)
    return ps.load_brep(s.brep()), 8.0


@pytest.mark.parametrize("recipe", ["1d2d", "regular+2d"])
def test_a_face_with_an_internal_edge_and_vertex_is_meshed_along_them(
    recipe: str,
) -> None:
    """The internal edge's segments lie on two triangles each, a node sits on the
    internal
    vertex, and the triangles fill the face's area exactly."""
    shape, area = _face_with_internal_edge_and_vertex()
    with Mesher(shape) as mesher:
        if recipe == "1d2d":
            mesher.assign(Netgen1D2D())
            mesher.assign(NetgenParameters2D(max_size=0.5))
        else:
            mesher.assign(Regular1D())
            mesher.assign(LocalLength(length=0.5))
            mesher.assign(Netgen2D())
        mesher.compute()
        mesh = mesher.mesh()
    xyz = mesh.node_coords
    triangle_edges: Counter[frozenset[int]] = Counter()
    total = 0.0
    for e in _elements(mesh, 2):
        c = _corners(mesh, e)
        total += 0.5 * float(
            np.linalg.norm(np.cross(xyz[c[1]] - xyz[c[0]], xyz[c[2]] - xyz[c[0]]))
        )
        for i in range(3):
            triangle_edges[frozenset((int(c[i]), int(c[(i + 1) % 3])))] += 1
    on_line = [
        frozenset(int(n) for n in _corners(mesh, e))
        for e in _elements(mesh, 1)
        if abs(
            float(
                np.cross(
                    xyz[_corners(mesh, e)[1]] - xyz[_corners(mesh, e)[0]],
                    (2.0, 1.0, 0.0),
                )[2]
            )
        )
        < 1e-9
        and 1.0 - 1e-9 <= float(xyz[_corners(mesh, e)][:, 0].min())
        and float(xyz[_corners(mesh, e)][:, 0].max()) <= 3.0 + 1e-9
        and float(xyz[_corners(mesh, e)][:, 1].min()) > 1e-9
    ]
    vertex = np.nonzero(np.linalg.norm(xyz - (1.0, 1.5, 0.0), axis=1) < 1e-9)[0]

    assert on_line
    assert all(triangle_edges[s] == 2 for s in on_line)
    assert vertex.size == 1
    assert abs(total - area) <= 1e-9 * area


# ---- Fineness, second order, simple parameters ------------------------------------- #


def test_the_element_count_grows_from_very_coarse_to_very_fine_on_a_sphere() -> None:
    """With a max_size the sphere never reaches, the curvature preset sets the size.

    A sphere of radius 2: netgen 6.2.2101 fails the surface of a unit sphere at COARSE
    ("Problem in Surface mesh generation"), a defect of its own (see the G4 tests).
    """
    s = Session()
    s.add_sphere(2.0)
    shape = ps.load_brep(s.brep())
    counts: list[int] = []
    for fineness in (
        Fineness.VERY_COARSE,
        Fineness.COARSE,
        Fineness.MODERATE,
        Fineness.FINE,
        Fineness.VERY_FINE,
    ):
        with Mesher(shape) as mesher:
            mesher.assign(Netgen1D2D3D())
            mesher.assign(NetgenParameters(max_size=100.0, fineness=fineness))
            counts.append(mesher.compute().volumes)

    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)


def test_second_order_puts_every_mid_edge_node_of_the_sphere_on_the_sphere() -> None:
    """Quadratic tetrahedra; every boundary node, mid-edge ones too, is at radius 1."""
    case = _make_case("sphere")
    with Mesher(case.shape) as mesher:
        mesher.assign(Netgen1D2D3D())
        mesher.assign(NetgenParameters(max_size=0.5, second_order=True))
        mesher.compute()
        mesh = mesher.mesh()
    types = {int(t) for t in mesh.element_type}
    boundary = mesh.node_kind != int(SubShapeKind.SOLID)
    radii = np.linalg.norm(mesh.node_coords[boundary], axis=1)
    quadratic = [
        e
        for e in range(mesh.element_count)
        if int(mesh.element_type[e]) == int(ElementType.QUAD_TETRAHEDRON)
    ]
    mid_on_boundary = {int(r) for e in _elements(mesh, 2) for r in mesh.nodes_of(e)[3:]}

    assert types <= {
        int(ElementType.QUAD_EDGE),
        int(ElementType.QUAD_TRIANGLE),
        int(ElementType.QUAD_TETRAHEDRON),
    }
    assert quadratic
    assert mid_on_boundary
    assert float(np.abs(radii - 1.0).max()) <= 1e-9


@pytest.mark.parametrize("dimension", [2, 3])
def test_simple_parameters_give_every_edge_of_a_cube_that_many_segments(
    dimension: int,
) -> None:
    """number_of_segments=4 on a cube: four segments on each edge.

    The plugin sizes each edge as its length / 3.6 and restricts the local size to that
    along the edge (``setLocalSize``); a cube's edges have one length, so no edge's size
    reaches into another, and each edge spans 3.6 sizes: ``floor(3.6 + 0.5) = 4``
    segments (oracle 5). NetgenSimpleParameters2D documents the count on edges of
    several lengths.
    """
    s = Session()
    s.add_box(2.0, 2.0, 2.0)
    shape = ps.load_brep(s.brep())
    with Mesher(shape) as mesher:
        if dimension == 2:
            mesher.assign(Netgen1D2D())
            mesher.assign(NetgenSimpleParameters2D(number_of_segments=4))
        else:
            mesher.assign(Netgen1D2D3D())
            mesher.assign(NetgenSimpleParameters3D(number_of_segments=4))
        mesher.compute()
        mesh = mesher.mesh()

    counts = [_segments_on(mesh, e).size for e in range(1, len(shape.edges()) + 1)]

    assert counts == [4] * len(shape.edges())


def test_max_element_volume_bounds_the_netgen_3d_tetrahedra() -> None:
    """MaxElementVolume 0.01 beside Netgen3D: the tetrahedra fill the box finer than the
    default, and their mean volume is at most the bound."""
    case = _make_case("box")
    with Mesher(case.shape) as mesher:
        mesher.assign(Netgen1D2D())
        mesher.assign(NetgenParameters2D(max_size=0.3))
        mesher.assign(Netgen3D())
        mesher.assign(MaxElementVolume(max_volume=0.01))
        mesher.compute()
        mesh = mesher.mesh()

    volumes = _tetra_volumes(mesh)

    assert float(volumes.mean()) <= 0.01
    assert abs(float(volumes.sum()) - case.volume) <= 1e-9 * case.volume


def test_quad_allowed_gives_a_quad_dominant_surface() -> None:
    """Netgen1D2D with quad_allowed on a box: quadrangles appear, the area is exact."""
    case = _make_case("box")
    with Mesher(case.shape) as mesher:
        mesher.assign(Netgen1D2D())
        mesher.assign(NetgenParameters2D(max_size=0.3, quad_allowed=True))
        mesher.compute()
        mesh = mesher.mesh()
    xyz = mesh.node_coords
    area = 0.0
    for e in _elements(mesh, 2):
        c = _corners(mesh, e)
        for k in range(1, len(c) - 1):
            area += 0.5 * float(
                np.linalg.norm(
                    np.cross(xyz[c[k]] - xyz[c[0]], xyz[c[k + 1]] - xyz[c[0]])
                )
            )
    quads = Counter(int(t) for t in mesh.element_type)[int(ElementType.QUADRANGLE)]

    assert quads > 0
    assert abs(area - 22.0) <= 1e-9 * 22.0


# ---- The typed classes refuse what the plugin would misread ------------------------ #


@pytest.mark.parametrize(
    "field",
    [{"growth_rate": 0.2}, {"segments_per_edge": 2.0}, {"segments_per_radius": 3.0}],
)
def test_a_preset_field_is_refused_unless_the_fineness_is_user_defined(
    field: dict[str, float],
) -> None:
    """A field the fineness preset controls needs Fineness.USER_DEFINED."""
    with pytest.raises(PysmeshError, match="USER_DEFINED"):
        NetgenParameters(fineness=Fineness.FINE, **field)


def test_a_user_defined_fineness_takes_the_three_preset_fields() -> None:
    """With USER_DEFINED the three fields are accepted and sent."""
    item = NetgenParameters2D(
        fineness=Fineness.USER_DEFINED,
        growth_rate=0.25,
        segments_per_edge=2.0,
        segments_per_radius=4.0,
    )

    params = item.params()

    assert params["growth_rate"] == 0.25
    assert params["segments_per_edge"] == 2.0
    assert params["segments_per_radius"] == 4.0
    assert "chordal_error" not in params


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_size": 0.0},
        {"max_size": 1.0, "min_size": 2.0},
        {"min_size": -1.0},
        {"fineness": Fineness.USER_DEFINED, "growth_rate": 1.5},
        {"chordal_error": 0.0},
        {"surface_optimization_steps": -1},
        {"threads": 0},
        {"local_sizes": ((SubShape(SubShapeKind.FACE, 1), 0.0),)},
    ],
)
def test_netgen_parameters_refuse_a_value_the_plugin_would_misread(
    kwargs: dict[str, object],
) -> None:
    """Sizes must be positive, min_size at most max_size, growth_rate in (0, 1]."""
    with pytest.raises(PysmeshError):
        NetgenParameters(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"number_of_segments": 3, "local_length": 0.5},
        {"number_of_segments": 0},
        {"local_length": -1.0},
        {"number_of_segments": 3, "max_element_area": 0.0},
    ],
)
def test_simple_parameters_refuse_no_segment_rule_two_or_a_bad_size(
    kwargs: dict[str, object],
) -> None:
    """Exactly one of number_of_segments and local_length, every size positive."""
    with pytest.raises(PysmeshError):
        NetgenSimpleParameters2D(**kwargs)  # type: ignore[arg-type]


def test_a_local_size_on_an_ordinal_the_shape_lacks_is_refused() -> None:
    """The factory resolves every local size's sub-shape before anything is assigned."""
    item = NetgenParameters(local_sizes=((SubShape(SubShapeKind.FACE, 99), 0.1),))
    with (
        Mesher(_make_case("box").shape) as mesher,
        pytest.raises(PysmeshError, match="99"),
    ):
        mesher.assign(item)


def test_replace_keeps_the_frozen_validation() -> None:
    """dataclasses.replace runs the same checks as the constructor."""
    item = NetgenParameters(max_size=1.0)

    with pytest.raises(PysmeshError, match="USER_DEFINED"):
        replace(item, growth_rate=0.2)
