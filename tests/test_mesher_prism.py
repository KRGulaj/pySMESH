# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-04

"""Gates for Prism3D: a side face over a split edge (B1), and the source face (S4).

Report ``defect_sweep_4.2.2.md`` §3 B1: a straight prism on a regular n-gon
(circumradius 1, height 1), one bottom edge split at its midpoint, the top edge above it
whole, so the side face between them has 5 edges. Regular1D with 4 segments, 2 on each
half-edge, so that every face can be structured.

* **n = 5 meshes as a prism.** Prism3D tried that 5-edge face as the bottom, rejected
  it, then meshed the solid from another face, and still reported the rejected face's
  error.
  The oracle is the geometry: a mesh of a polyhedron whose nodes lie on its planar faces
  fills it exactly, so the cell volumes sum to the n-gon area times the height; every
  cell has a positive volume; the solid is a straight prism, so every node of the top
  cap has a node of the bottom cap 1 below it.
* **n = 6 is refused, naming the real cause.** Prism3D finds no face it can sweep from.
  With the top cap as the bottom, the 5-edge face has 1 edge on its bottom side and 2 on
  its top side, and Prism3D projects only onto the first edge of a top side
  (``StdMeshers_Prism_3D.cxx`` computeWalls), so it cannot mesh that face. The message
  must say so for that face, and name the way out, not report the error of a rejected
  candidate.
* **The way out meshes.** With the top edge split too, at the same point, the side face
  is two quadrangles: the prism fills its volume, the caps are 1 apart node for node,
  and the cells are the bottom faces times the layers.
* **Unmatched segments still fill the solid.** With 4 segments on each half-edge under
  4 on the whole top edge (repro ``r3h.py``), Prism3D sweeps between two side faces. The
  cells are warped, so the oracle is their exact volume with bilinear faces, the
  boundary nodes on the prism's faces, and positive cells.

Report §8 S4: two unit boxes stacked and fused, so every face is a quadrangle and each
side is two faces. Nothing marks a source face, so Prism3D tries the faces in turn.

* **The search meshes it.** It failed with "Wrong source face", the error of a rejected
  candidate, though a later face was meshed from (the B1 cause).
* **A 2-D algorithm on one face makes it the source.** SMESH sweeps from a face that is
  already meshed (``StdMeshers_Prism_3D.cxx`` Compute; ``prism_3d_algo.rst``: "It is
  enough to define a sub-mesh on either the top or the base face").
The oracle is the closed form: with n segments on every edge, n x n x 2n hexahedra fill
the volume 2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    ElementType,
    EntityKind,
    MaxElementArea,
    Mefisto2D,
    Mesher,
    NumberOfSegments,
    Prism3D,
    PysmeshError,
    Quadrangle2D,
    Regular1D,
    Session,
    SubShape,
    SubShapeKind,
)

# Segments on every edge, and so the layers of the prism.
LAYERS: int = 4
# Segments on each half of the split edge: together they match the whole top edge.
HALF_SEGMENTS: int = 2
HEIGHT: float = 1.0
# The brief's bound on the mesh volume; the exact fill leaves only rounding.
VOLUME_RTOL: float = 1e-9
# How close a cap node must sit to the plane of its cap, and to its partner below.
NODE_TOL: float = 1e-9
# The 2-D base mesher for the n-gon caps and the side faces.
BASES: tuple[str, ...] = ("quadrangle", "mefisto")
MEFISTO_MAX_AREA: float = 0.02


def _split_edge_prism(n: int, split_top: bool = False) -> ps.Shape:
    """The B1 solid: an n-gon prism, one bottom edge split, the top edge above whole.

    With ``split_top`` the top edge is split too, at the same point: the way out, which
    turns the 5-edge side face into two quadrangles.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n + 1)[:-1]
    bottom = np.c_[np.cos(t), np.sin(t), np.zeros(n)]
    top = bottom + (0.0, 0.0, HEIGHT)
    mid = 0.5 * (bottom[0] + bottom[1])
    top_mid = mid + (0.0, 0.0, HEIGHT)
    if split_top:
        loops = [
            np.vstack([bottom[:1], mid, bottom[1:]]),
            np.vstack([top[:1], top_mid, top[1:]])[::-1],
            np.array([bottom[0], mid, top_mid, top[0]]),
            np.array([mid, bottom[1], top[1], top_mid]),
        ]
    else:
        loops = [
            np.vstack([bottom[:1], mid, bottom[1:]]),
            top[::-1],
            np.array([bottom[0], mid, bottom[1], top[1], top[0]]),
        ]
    for i in range(1, n):
        j = (i + 1) % n
        loops.append(np.array([bottom[i], bottom[j], top[j], top[i]]))
    s = Session()
    for loop in loops:
        before = set(s.entities(EntityKind.EDGE).tolist())
        s.add_polyline(loop, closed=True)
        s.make_face(
            [e for e in s.entities(EntityKind.EDGE).tolist() if e not in before]
        )
    s.sew(s.entities(EntityKind.FACE).tolist(), make_solid=True)
    return ps.load_brep(s.brep())


def _assign(
    m: Mesher, shape: ps.Shape, n: int, base: str, half_segments: int = HALF_SEGMENTS
) -> None:
    """Regular1D, 4 segments and ``half_segments`` per half-edge, the base, Prism3D."""
    half = math.sin(math.pi / n)
    halves = [e.id for e in shape.edges() if abs(e.length - half) < NODE_TOL]
    assert len(halves) in (2, 4)
    m.assign(Regular1D())
    m.assign(NumberOfSegments(count=LAYERS))
    for e in halves:
        m.assign(
            NumberOfSegments(count=half_segments), on=SubShape(SubShapeKind.EDGE, e)
        )
    if base == "quadrangle":
        m.assign(Quadrangle2D())
    else:
        m.assign(Mefisto2D())
        m.assign(MaxElementArea(max_area=MEFISTO_MAX_AREA))
    m.assign(Prism3D())


def _unlifted_top_nodes(xyz: NDArray[np.float64]) -> int:
    """How many top cap nodes have no bottom cap node exactly below them."""
    top = xyz[np.abs(xyz[:, 2] - HEIGHT) < NODE_TOL, :2]
    bottom = xyz[np.abs(xyz[:, 2]) < NODE_TOL, :2]
    gap = np.linalg.norm(top[:, None, :] - bottom[None, :, :], axis=2).min(axis=1)
    return int(np.count_nonzero(gap > NODE_TOL))


def _bottom_face_count(mesh: ps.MeshData) -> int:
    """How many 2-D elements lie on the bottom cap, z = 0."""
    count = 0
    for k in range(len(mesh.element_type)):
        rows = mesh.element_nodes[mesh.element_offsets[k] : mesh.element_offsets[k + 1]]
        if mesh.element_kind[k] == int(SubShapeKind.FACE) and bool(
            np.all(np.abs(mesh.node_coords[rows, 2]) < NODE_TOL)
        ):
            count += 1
    return count


@pytest.mark.parametrize("base", BASES)
def test_a_split_edge_pentagon_prism_fills_its_volume_with_lifted_caps(
    base: str,
) -> None:
    """n = 5: volume = area x height within 1e-9, positive cells, caps 1 apart (B1)."""
    shape = _split_edge_prism(5)
    area = 0.5 * 5 * math.sin(2.0 * math.pi / 5)

    with Mesher(shape) as m:
        _assign(m, shape, 5, base)
        m.compute()
        volumes = m.quality(ps.Volume()).values
        xyz = m.mesh().node_coords

    assert float(volumes.sum()) == pytest.approx(area * HEIGHT, rel=VOLUME_RTOL)
    assert float(volumes.min()) > 0.0
    assert _unlifted_top_nodes(xyz) == 0


@pytest.mark.parametrize("base", BASES)
def test_a_split_edge_hexagon_prism_is_refused_naming_the_composite_side(
    base: str,
) -> None:
    """n = 6: refused for the 5-edge face's composite top side, no stale error (B1)."""
    shape = _split_edge_prism(6)

    with Mesher(shape) as m:
        _assign(m, shape, 6, base)
        with pytest.raises(PysmeshError) as caught:
            m.compute()

    details = caught.value.details
    assert "No FACE can be the bottom of the prism." in details
    assert (
        "has 1 EDGE(s) on its bottom side and 2 on its top side; "
        "a composite horizontal side is not supported. Split the EDGE opposite the "
        "composite side at the same points"
    ) in details


# Segments on every edge of the stacked boxes.
STACK_SEGMENTS: int = 3
STACK_VOLUME: float = 2.0
# Finds the base face by its box; above the 1e-7 tolerance pad of older boxes.
FACE_TOL: float = 1e-6


def _stacked_boxes() -> ps.Shape:
    """Two unit boxes, one on the other, fused into one solid of 10 quadrangles."""
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    lower = s.entities(EntityKind.SOLID).tolist()
    s.add_box(1.0, 1.0, 1.0, origin=(0.0, 0.0, 1.0))
    upper = [i for i in s.entities(EntityKind.SOLID).tolist() if i not in lower]
    s.fuse(lower, upper)
    return ps.load_brep(s.brep())


def _stacked_mesh(source: str) -> tuple[NDArray[np.float64], int]:
    """Cell volumes and hexahedra; Quadrangle2D on every face or on the base only."""
    shape = _stacked_boxes()
    base = [f.id for f in shape.faces() if f.bbox[5] < FACE_TOL]
    with Mesher(shape) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=STACK_SEGMENTS))
        if source == "base":
            m.assign(Quadrangle2D(), on=SubShape(SubShapeKind.FACE, base[0]))
        else:
            m.assign(Quadrangle2D())
        m.assign(Prism3D())
        m.compute()
        volumes = m.quality(ps.Volume()).values
        types = m.mesh().element_type
    return volumes, int(np.count_nonzero(types == int(ElementType.HEXAHEDRON)))


@pytest.mark.parametrize("source", ["base", "search"])
def test_stacked_boxes_fill_volume_2_with_the_closed_form_hexahedra(
    source: str,
) -> None:
    """n x n x 2n hexahedra, positive, of volume 2; base assigned or searched (S4)."""
    n = STACK_SEGMENTS

    volumes, hexahedra = _stacked_mesh(source)

    assert hexahedra == n * n * 2 * n == len(volumes)
    assert float(volumes.min()) > 0.0
    assert float(volumes.sum()) == pytest.approx(STACK_VOLUME, rel=VOLUME_RTOL)


# The way out and the r3h case (brief amendment 7). A conforming mesh whose boundary
# nodes lie on the planar faces fills the solid exactly when every cell is measured with
# its faces as bilinear surfaces, the surfaces two neighbours share: by the divergence
# theorem, V = 1/3 of the flux of X through the cell's faces (3 x 3 Gauss points, exact
# for a bilinear face). SMESH's Volume control splits each cell into tetrahedra by its
# own diagonals, which neighbours need not share, so on warped cells its sum is not the
# solid's volume.
GAUSS_POINTS: NDArray[np.float64] = np.array(
    [0.5 - math.sqrt(0.15), 0.5, 0.5 + math.sqrt(0.15)], dtype=np.float64
)
GAUSS_WEIGHTS: NDArray[np.float64] = np.array([5.0, 8.0, 5.0], dtype=np.float64) / 18.0
CELL_FACES: dict[int, tuple[tuple[int, ...], ...]] = {
    int(ElementType.PENTAHEDRON): (
        (0, 1, 2),
        (3, 4, 5),
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (2, 0, 3, 5),
    ),
    int(ElementType.HEXAHEDRON): (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
}


def _face_flux(p: NDArray[np.float64], centre: NDArray[np.float64]) -> float:
    """The flux of X through one cell face, its normal pointing away from the centre."""
    if len(p) == 3:
        n = 0.5 * np.cross(p[1] - p[0], p[2] - p[0])
        c = p.mean(axis=0)
        flux = float(np.dot(c, n))
        return flux if float(np.dot(c - centre, n)) >= 0.0 else -flux
    total = 0.0
    for i, u in enumerate(GAUSS_POINTS):
        for j, v in enumerate(GAUSS_POINTS):
            x = (
                (1 - u) * (1 - v) * p[0]
                + u * (1 - v) * p[1]
                + u * v * p[2]
                + (1 - u) * v * p[3]
            )
            xu = (1 - v) * (p[1] - p[0]) + v * (p[2] - p[3])
            xv = (1 - u) * (p[3] - p[0]) + u * (p[2] - p[1])
            total += float(GAUSS_WEIGHTS[i] * GAUSS_WEIGHTS[j]) * float(
                np.dot(x, np.cross(xu, xv))
            )
    mid = p.mean(axis=0)
    normal = np.cross(p[1] - p[0] + p[2] - p[3], p[3] - p[0] + p[2] - p[1])
    return total if float(np.dot(mid - centre, normal)) >= 0.0 else -total


def _exact_volume(mesh: ps.MeshData) -> float:
    """The sum of the cells' volumes with bilinear faces."""
    xyz, off, con = mesh.node_coords, mesh.element_offsets, mesh.element_nodes
    total = 0.0
    for k, kind in enumerate(mesh.element_type):
        faces = CELL_FACES.get(int(kind))
        if faces is None:
            continue
        nodes = con[off[k] : off[k + 1]]
        centre = xyz[nodes].mean(axis=0)
        total += (
            sum(_face_flux(xyz[[nodes[i] for i in f]], centre) for f in faces) / 3.0
        )
    return total


def _off_surface_nodes(mesh: ps.MeshData, n: int) -> int:
    """How many nodes on a face, edge or vertex lie off the prism's surface."""
    t = np.linspace(0.0, 2.0 * np.pi, n + 1)[:-1]
    corners = np.c_[np.cos(t), np.sin(t)]
    edges = np.roll(corners, -1, axis=0) - corners
    normals = np.c_[edges[:, 1], -edges[:, 0]] / np.linalg.norm(edges, axis=1)[:, None]
    bound = np.isin(
        mesh.node_kind,
        [int(SubShapeKind.FACE), int(SubShapeKind.EDGE), int(SubShapeKind.VERTEX)],
    )
    xyz = mesh.node_coords[bound]
    side = np.einsum("pnk,nk->pn", xyz[:, None, :2] - corners[None, :, :], normals)
    gap = np.minimum(
        np.minimum(np.abs(xyz[:, 2]), np.abs(xyz[:, 2] - HEIGHT)),
        np.abs(side).min(axis=1),
    )
    return int(np.count_nonzero(gap > NODE_TOL))


@pytest.mark.parametrize("base", BASES)
def test_the_way_out_a_prism_with_both_cap_edges_split_meshes_as_a_prism(
    base: str,
) -> None:
    """Top edge split too: exact volume, positive cells, caps 1 apart, cells (B1)."""
    shape = _split_edge_prism(5, split_top=True)
    area = 0.5 * 5 * math.sin(2.0 * math.pi / 5)

    with Mesher(shape) as m:
        _assign(m, shape, 5, base)
        m.compute()
        volumes = m.quality(ps.Volume()).values
        mesh = m.mesh()

    bottom_faces = _bottom_face_count(mesh)
    assert float(volumes.sum()) == pytest.approx(area * HEIGHT, rel=VOLUME_RTOL)
    assert float(volumes.min()) > 0.0
    assert _unlifted_top_nodes(mesh.node_coords) == 0
    assert len(volumes) == bottom_faces * LAYERS


@pytest.mark.parametrize("base", BASES)
def test_a_split_edge_prism_with_unmatched_segments_fills_its_volume_exactly(
    base: str,
) -> None:
    """r3h: 4 segments per half-edge under 4 above: exact fill, on-face nodes (B1)."""
    shape = _split_edge_prism(5)
    area = 0.5 * 5 * math.sin(2.0 * math.pi / 5)

    with Mesher(shape) as m:
        _assign(m, shape, 5, base, half_segments=LAYERS)
        m.compute()
        volumes = m.quality(ps.Volume()).values
        mesh = m.mesh()

    assert _exact_volume(mesh) == pytest.approx(area * HEIGHT, rel=VOLUME_RTOL)
    assert float(volumes.min()) > 0.0
    assert _off_surface_nodes(mesh, 5) == 0
