# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-05

"""NETGEN_Remesher_2D: a triangle mesh with no shape, meshed again by NETGEN.

The remesher reads the triangles as an STL surface, cuts it into charts at its feature
edges (adjacent normals more than ``ridge_angle`` apart) and meshes each chart anew
(``NETGENPlugin_Remesher_2D.cxx``). The oracles, on an icosphere inscribed in the unit
sphere and on a triangulated cube:

* **Closed 2-manifold:** every edge of the result is shared by exactly two triangles.
* **On the input:** netgen places every new node on an input triangle, so the distance
  of each node to the input surface is round-off (bound 1e-12).
* **Area:** an output triangle spans input triangles whose normals differ from its own
  by at most the largest angle ``theta`` between adjacent input normals, so it covers
  its patch of the input with the factor cos(theta) or better: |A_out - A_in| <= (1 -
  cos theta) A_in. A planar-faced input keeps its area to round-off.
* **Feature edges:** the cube's 12 edges (dihedral 90 deg, above the ridge angle of 30
  deg) are chart boundaries, so the result keeps each of them as triangle edges end to
  end.
* **Size:** netgen's target edge is ``max_size``; the mean edge is at most 1.25 times it
  and the longest at most twice it. Halving ``max_size`` multiplies the triangle count
  by about 4 (area over the square of the size).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    ElementDimension,
    ElementType,
    Mesher,
    NetgenRemesher2D,
    NetgenRemesherParameters2D,
    PysmeshCancelled,
    PysmeshError,
    Regular1D,
    Session,
    SubShape,
    SubShapeKind,
    load_brep,
)

_ON_SURFACE: float = 1e-12

Points = NDArray[np.float64]
Triangles = NDArray[np.int64]


def _icosphere(levels: int) -> tuple[Points, Triangles]:
    """A unit icosahedron split ``levels`` times, every node on the unit sphere."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    corners = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]  # fmt: skip
    tris = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    nodes = [np.asarray(p, dtype=np.float64) / np.linalg.norm(p) for p in corners]
    for _ in range(levels):
        middle: dict[tuple[int, int], int] = {}
        finer = []
        for a, b, c in tris:
            ab, bc, ca = (
                _midpoint(nodes, middle, *e) for e in ((a, b), (b, c), (c, a))
            )
            finer += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        tris = finer
    return np.asarray(nodes, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def _midpoint(
    nodes: list[NDArray[np.float64]], middle: dict[tuple[int, int], int], a: int, b: int
) -> int:
    """The node between nodes a and b, pushed onto the unit sphere, made once."""
    key = (min(a, b), max(a, b))
    if key not in middle:
        p = nodes[a] + nodes[b]
        nodes.append(p / np.linalg.norm(p))
        middle[key] = len(nodes) - 1
    return middle[key]


def _cube(side: float, n: int) -> tuple[Points, Triangles]:
    """The cube [0, side]^3, each face an n x n grid of squares cut in two, outward."""
    index: dict[tuple[int, int, int], int] = {}
    xyz: list[tuple[float, float, float]] = []

    def node(ijk: tuple[int, int, int]) -> int:
        if ijk not in index:
            index[ijk] = len(xyz)
            xyz.append((side * ijk[0] / n, side * ijk[1] / n, side * ijk[2] / n))
        return index[ijk]

    tris: list[tuple[int, int, int]] = []
    for axis in range(3):
        for level, outward_low in ((0, True), (n, False)):
            for a in range(n):
                for b in range(n):
                    quad = []
                    for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        ijk = [0, 0, 0]
                        ijk[axis] = level
                        ijk[(axis + 1) % 3] = a + da
                        ijk[(axis + 2) % 3] = b + db
                        quad.append(node((ijk[0], ijk[1], ijk[2])))
                    if outward_low:
                        quad.reverse()
                    tris += [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]
    return np.asarray(xyz, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def _distance_to_triangles(points: Points, corners: Points) -> Points:
    """The distance from each point to the nearest triangle of ``corners`` (M, 3, 3).

    The closest point on a triangle by its Voronoi regions (Ericson, Real-Time Collision
    Detection, 2005, section 5.1.5), over every point and triangle pair.
    """
    a, b, c = (corners[None, :, i, :] for i in range(3))
    p = points[:, None, :]
    ab, ac = b - a, c - a
    d1, d2 = np.sum(ab * (p - a), -1), np.sum(ac * (p - a), -1)
    d3, d4 = np.sum(ab * (p - b), -1), np.sum(ac * (p - b), -1)
    d5, d6 = np.sum(ab * (p - c), -1), np.sum(ac * (p - c), -1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = 1.0 / (va + vb + vc)
        closest = a + (vb * inner)[..., None] * ab + (vc * inner)[..., None] * ac
        t_bc = ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None]
        t_ac = (d2 / (d2 - d6))[..., None]
        t_ab = (d1 / (d1 - d3))[..., None]
    # Edge regions first, vertex regions last: a vertex region wins where they meet.
    regions = [
        ((va <= 0) & (d4 >= d3) & (d5 >= d6), b + t_bc * (c - b)),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + t_ac * ac),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + t_ab * ab),
        ((d6 >= 0) & (d5 <= d6), c),
        ((d3 >= 0) & (d4 <= d3), b),
        ((d1 <= 0) & (d2 <= 0), a),
    ]
    for mask, q in regions:
        closest = np.where(mask[..., None], np.broadcast_to(q, closest.shape), closest)
    return np.sqrt(np.min(np.sum((p - closest) ** 2, -1), axis=1))


def _area(xyz: Points, tris: Triangles) -> float:
    """The total area of the triangles."""
    a, b, c = xyz[tris[:, 0]], xyz[tris[:, 1]], xyz[tris[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _edge_use(tris: Triangles) -> dict[tuple[int, int], int]:
    """How many triangles use each undirected edge."""
    use: dict[tuple[int, int], int] = {}
    for t in tris:
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            use[key] = use.get(key, 0) + 1
    return use


def _largest_adjacent_normal_angle(xyz: Points, tris: Triangles) -> float:
    """The largest angle, in radians, between the normals of two adjacent triangles."""
    n = np.cross(xyz[tris[:, 1]] - xyz[tris[:, 0]], xyz[tris[:, 2]] - xyz[tris[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    owners: dict[tuple[int, int], list[int]] = {}
    for k, t in enumerate(tris):
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            owners.setdefault((min(a, b), max(a, b)), []).append(k)
    return max(
        math.acos(min(1.0, float(np.dot(n[p], n[q])))) for p, q in owners.values()
    )


def _remesh(
    xyz: Points, tris: Triangles, params: NetgenRemesherParameters2D
) -> tuple[Points, Triangles, int]:
    """Remesh the triangles; the new nodes, triangles and segment count."""
    with Mesher.from_arrays(xyz, tris) as mesher:
        mesher.assign(NetgenRemesher2D())
        mesher.assign(params)
        report = mesher.compute()
        md = mesher.mesh()
    is_tri = md.element_type == int(ElementType.TRIANGLE)
    assert report.faces == int(is_tri.sum()) and report.volumes == 0
    start = md.element_offsets[:-1][is_tri]
    out = md.element_nodes[start[:, None] + np.arange(3)].astype(np.int64)
    return md.node_coords, out, report.edges


def _edge_lengths(xyz: Points, tris: Triangles) -> Points:
    """The length of every undirected triangle edge."""
    keys = np.asarray(list(_edge_use(tris)), dtype=np.int64)
    return np.linalg.norm(xyz[keys[:, 0]] - xyz[keys[:, 1]], axis=1)


@pytest.mark.parametrize("size", [0.2, 0.1])
def test_a_remeshed_sphere_is_closed_lies_on_the_input_and_keeps_its_area(
    size: float,
) -> None:
    """An icosphere of 320 triangles, its 162 nodes on the unit sphere, remeshed."""
    xyz0, tris0 = _icosphere(2)
    theta = _largest_adjacent_normal_angle(xyz0, tris0)

    xyz, tris, _ = _remesh(xyz0, tris0, NetgenRemesherParameters2D(max_size=size))

    assert set(_edge_use(tris).values()) == {2}
    assert float(_distance_to_triangles(xyz, xyz0[tris0]).max()) <= _ON_SURFACE
    a0, a1 = _area(xyz0, tris0), _area(xyz, tris)
    assert abs(a1 - a0) <= (1.0 - math.cos(theta)) * a0
    lengths = _edge_lengths(xyz, tris)
    assert float(lengths.mean()) <= 1.25 * size
    assert float(lengths.max()) <= 2.0 * size


def test_halving_max_size_makes_about_four_times_the_triangles() -> None:
    """Triangle count ~ area / size^2: the ratio lies in [3, 5]."""
    xyz0, tris0 = _icosphere(2)

    coarse = _remesh(xyz0, tris0, NetgenRemesherParameters2D(max_size=0.2))[1]
    fine = _remesh(xyz0, tris0, NetgenRemesherParameters2D(max_size=0.1))[1]

    assert 3.0 <= len(fine) / len(coarse) <= 5.0


def _cube_edge_coverage(xyz: Points, tris: Triangles, side: float) -> list[float]:
    """For each of the cube's 12 edges, the length of the triangle edges lying on it."""
    keys = np.asarray(list(_edge_use(tris)), dtype=np.int64)
    p, q = xyz[keys[:, 0]], xyz[keys[:, 1]]
    covered = []
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3
        for u in (0.0, side):
            for v in (0.0, side):
                on = (
                    (np.abs(p[:, b] - u) <= _ON_SURFACE)
                    & (np.abs(p[:, c] - v) <= _ON_SURFACE)
                    & (np.abs(q[:, b] - u) <= _ON_SURFACE)
                    & (np.abs(q[:, c] - v) <= _ON_SURFACE)
                )
                covered.append(float(np.linalg.norm(p[on] - q[on], axis=1).sum()))
    return covered


@pytest.mark.parametrize("size", [0.3, 0.15])
def test_a_remeshed_cube_keeps_its_twelve_feature_edges_and_its_area(
    size: float,
) -> None:
    """A 2 x 2 x 2 cube of 192 triangles, remeshed at ``size``.

    Each cube edge is a feature edge (90 deg > ridge angle 30 deg): the triangle edges
    on it add up to its length, 2. The faces are planar, so the area stays 24 to
    round-off.
    """
    xyz0, tris0 = _cube(2.0, 4)

    xyz, tris, segments = _remesh(
        xyz0, tris0, NetgenRemesherParameters2D(max_size=size)
    )

    assert set(_edge_use(tris).values()) == {2}
    assert float(_distance_to_triangles(xyz, xyz0[tris0]).max()) <= _ON_SURFACE
    assert _area(xyz, tris) == pytest.approx(24.0, rel=1e-12)
    assert _cube_edge_coverage(xyz, tris, 2.0) == pytest.approx([2.0] * 12, rel=1e-12)
    assert segments > 0
    lengths = _edge_lengths(xyz, tris)
    assert float(lengths.mean()) <= 1.25 * size
    assert float(lengths.max()) <= 2.0 * size


def test_make_groups_of_surfaces_gives_one_planar_group_per_cube_face() -> None:
    """The six charts of the cube are its six faces: six groups, each in one plane."""
    xyz0, tris0 = _cube(2.0, 4)

    with Mesher.from_arrays(xyz0, tris0) as mesher:
        mesher.assign(NetgenRemesher2D())
        mesher.assign(
            NetgenRemesherParameters2D(max_size=0.3, make_groups_of_surfaces=True)
        )
        report = mesher.compute()
        md = mesher.mesh()
        groups = mesher.groups()

    names = sorted(g.name for g in groups)
    assert names == [f"Surface_{i}" for i in range(1, 7)]
    assert sum(len(g.element_ids) for g in groups) == report.faces
    row = {int(i): k for k, i in enumerate(md.element_id)}
    for g in groups:
        nodes = np.concatenate(
            [
                md.element_nodes[
                    md.element_offsets[row[i]] : md.element_offsets[row[i] + 1]
                ]
                for i in g.element_ids.tolist()
            ]
        )
        assert int((np.ptp(md.node_coords[nodes], axis=0) <= _ON_SURFACE).sum()) == 1


@pytest.mark.parametrize("fixed", [False, True])
def test_fixed_edges_keep_their_nodes_on_a_feature_edge(fixed: bool) -> None:
    """The cube edge y = z = 0 has input nodes at x = 0, 0.5, 1, 1.5, 2.

    Remeshed at 0.3 the edge gets its own spacing, and only its ends stay. With its
    input segments in a fixed_edges group, every one of the five nodes stays: each ends
    a feature line. The remesh removes the old segments, so the group ends empty.
    """
    xyz0, tris0 = _cube(2.0, 4)
    line = sorted(
        (i for i, p in enumerate(xyz0) if p[1] == 0.0 and p[2] == 0.0),
        key=lambda i: xyz0[i][0],
    )

    with Mesher.from_arrays(xyz0, tris0) as mesher:
        ids = mesher.mesh().node_id
        name = None
        if fixed:
            pairs = np.asarray([[ids[a], ids[b]] for a, b in pairwise(line)])
            mesher.add_group(
                "fixed", ElementDimension.EDGE, mesher.add_segments(pairs).tolist()
            )
            name = "fixed"
        mesher.assign(NetgenRemesher2D())
        mesher.assign(NetgenRemesherParameters2D(max_size=0.3, fixed_edges=name))
        mesher.compute()
        xyz = mesher.mesh().node_coords
        left = [len(g.element_ids) for g in mesher.groups()]
    kept = [
        bool((np.linalg.norm(xyz - xyz0[i], axis=1) <= _ON_SURFACE).any()) for i in line
    ]

    assert len(line) == 5
    assert kept == ([True] * 5 if fixed else [True, False, False, False, True])
    assert left == ([0] if fixed else [])


def test_a_non_manifold_input_raises_with_the_remeshers_reason() -> None:
    """Three triangles on one edge: the remesher refuses a non-manifold mesh."""
    xyz = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64
    )
    tris = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)

    with Mesher.from_arrays(xyz, tris) as mesher:
        mesher.assign(NetgenRemesher2D())
        mesher.assign(NetgenRemesherParameters2D(max_size=0.3))
        with pytest.raises(PysmeshError, match="remeshing failed") as raised:
            mesher.compute()

    assert (
        "Non-manifold mesh. Only manifold mesh can be re-meshed" in raised.value.details
    )
    assert "NETGEN_Remesher_2D" in raised.value.details


def test_faces_without_area_are_refused_before_netgen_runs() -> None:
    """Two triangles on one line have no surface: refused, naming the area."""
    xyz = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
    tris = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

    with Mesher.from_arrays(xyz, tris) as mesher:
        mesher.assign(NetgenRemesher2D())
        mesher.assign(NetgenRemesherParameters2D(max_size=0.3))
        with pytest.raises(PysmeshError, match="total area is 0") as raised:
            mesher.compute()
        left = mesher.mesh().element_count

    assert "degenerate" in raised.value.details
    assert left == 2


def test_a_removed_fixed_edges_group_is_refused_at_compute() -> None:
    """The parameters hold the group by id: once it is removed, compute says so."""
    xyz0, tris0 = _cube(2.0, 2)

    with Mesher.from_arrays(xyz0, tris0) as mesher:
        ids = mesher.mesh().node_id
        segment = mesher.add_segments(np.asarray([[ids[0], ids[1]]]))
        mesher.add_group("fixed", ElementDimension.EDGE, segment.tolist())
        mesher.assign(NetgenRemesher2D())
        mesher.assign(NetgenRemesherParameters2D(max_size=0.5, fixed_edges="fixed"))
        mesher.remove_group("fixed")
        with pytest.raises(PysmeshError, match="no longer exists"):
            mesher.compute()


def test_fixed_edges_must_name_an_edge_group_of_the_mesher() -> None:
    xyz0, tris0 = _cube(2.0, 2)

    with Mesher.from_arrays(xyz0, tris0) as mesher:
        mesher.add_group("faces", ElementDimension.FACE, [1])
        mesher.assign(NetgenRemesher2D())
        with pytest.raises(PysmeshError, match="names no group"):
            mesher.assign(NetgenRemesherParameters2D(fixed_edges="missing"))
        with pytest.raises(PysmeshError, match="EDGE elements"):
            mesher.assign(NetgenRemesherParameters2D(fixed_edges="faces"))


def test_the_remesher_is_refused_on_a_mesher_with_a_shape() -> None:
    s = Session()
    s.add_box(3.0, 7.0, 11.0)

    with Mesher(load_brep(s.brep())) as mesher:
        with pytest.raises(PysmeshError, match="has a shape"):
            mesher.assign(NetgenRemesher2D())
        with pytest.raises(PysmeshError, match="has a shape"):
            mesher.assign(NetgenRemesherParameters2D())


def test_a_mesher_with_no_shape_takes_the_remesher_only_on_the_whole_mesh() -> None:
    xyz0, tris0 = _cube(2.0, 2)

    with Mesher.from_arrays(xyz0, tris0) as mesher:
        with pytest.raises(PysmeshError, match="has no shape"):
            mesher.assign(Regular1D())
        with pytest.raises(PysmeshError, match="on the whole mesh"):
            mesher.assign(NetgenRemesher2D(), on=SubShape(SubShapeKind.FACE, 1))
        mesher.assign(NetgenRemesher2D())

        assert mesher.assignments() == (("NETGEN_Remesher_2D", None),)


def test_a_remesh_of_a_mesher_with_no_face_is_refused() -> None:
    with Mesher() as mesher:
        mesher.add_nodes(np.zeros((1, 3), dtype=np.float64))
        mesher.assign(NetgenRemesher2D())
        with pytest.raises(PysmeshError, match="holds no face"):
            mesher.compute()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_size", 0.0),
        ("min_size", -1.0),
        ("ridge_angle", 0.0),
        ("chart_angle", 181.0),
        ("line_length_factor", 0.0),
        ("fixed_edges", ""),
    ],
)
def test_remesher_parameters_refuse_a_value_netgen_would_misread(
    field: str, value: object
) -> None:
    with pytest.raises(PysmeshError, match=field):
        NetgenRemesherParameters2D(**{field: value})  # type: ignore[arg-type]


_CHILD_DIGESTS: str = """
import hashlib, json, os, sys
occt = os.environ.get("PYSMESH_OCCT_BIN")
if occt:
    os.add_dll_directory(occt)
lib = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(lib):
    os.add_dll_directory(lib)
sys.path.insert(0, sys.argv[1])
import numpy as np
import pysmesh as ps

data = np.load(sys.argv[2])
digests = []
for size in (0.15, 0.1, 0.2, 0.15):
    with ps.Mesher.from_arrays(data["xyz"], data["tris"]) as m:
        m.assign(ps.NetgenRemesher2D())
        m.assign(ps.NetgenRemesherParameters2D(max_size=size))
        m.compute()
        md = m.mesh()
    h = hashlib.sha256()
    for a in (md.node_coords, md.element_offsets, md.element_nodes):
        h.update(a.tobytes())
    digests.append(h.hexdigest())
with open(sys.argv[3], "w", encoding="utf-8") as out:
    json.dump(digests, out)
"""


def test_a_remesh_is_the_same_in_every_process_and_on_repeat(tmp_path: Path) -> None:
    """Four remeshes in each of two child processes: one mesh per size, bit for bit.

    Before NETGENPlugin_remesher_stl_topology.patch the remesher built netgen's size
    tree from bytes that were not the bounding box: the meshes varied, and about one
    process in three ended in a stack overflow.
    """
    xyz0, tris0 = _icosphere(2)
    np.savez(tmp_path / "input.npz", xyz=xyz0, tris=tris0)
    package_root = str(Path(ps.__file__).resolve().parent.parent)

    runs = []
    for k in range(2):
        out = tmp_path / f"digests{k}.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _CHILD_DIGESTS,
                package_root,
                str(tmp_path / "input.npz"),
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=600.0,
            env=dict(os.environ),
            check=False,
        )
        assert proc.returncode == 0, f"exit {proc.returncode:#x}: {proc.stderr[-2000:]}"
        runs.append(json.loads(out.read_text(encoding="utf-8")))

    assert runs[0] == runs[1]
    assert runs[0][0] == runs[0][3]
    assert len(set(runs[0][:3])) == 3


_CHILD_NO_PARAMETERS: str = """
import os, sys
occt = os.environ.get("PYSMESH_OCCT_BIN")
if occt:
    os.add_dll_directory(occt)
lib = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(lib):
    os.add_dll_directory(lib)
sys.path.insert(0, sys.argv[1])
import numpy as np
import pysmesh as ps

data = np.load(sys.argv[2])
with ps.Mesher.from_arrays(data["xyz"], data["tris"]) as m:
    m.assign(ps.NetgenRemesher2D())
    m.compute()
    md = m.mesh()
np.savez(sys.argv[3], xyz=md.node_coords, offsets=md.element_offsets,
         nodes=md.element_nodes, types=md.element_type)
"""


def test_the_remesher_runs_without_parameters(tmp_path: Path) -> None:
    """With no NetgenRemesherParameters2D the size is the box diagonal over 10.

    The icosphere's box diagonal is 2 sqrt(3) = 3.46, so the target edge is 0.346. In a
    child process: before NETGENPlugin_remesher_no_parameters.patch this compute read
    address 0.
    """
    xyz0, tris0 = _icosphere(2)
    np.savez(tmp_path / "input.npz", xyz=xyz0, tris=tris0)
    package_root = str(Path(ps.__file__).resolve().parent.parent)
    size = 2.0 * math.sqrt(3.0) / 10.0

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD_NO_PARAMETERS,
            package_root,
            str(tmp_path / "input.npz"),
            str(tmp_path / "output.npz"),
        ],
        capture_output=True,
        text=True,
        timeout=600.0,
        env=dict(os.environ),
        check=False,
    )

    assert proc.returncode == 0, f"exit {proc.returncode:#x}: {proc.stderr[-2000:]}"
    out = np.load(tmp_path / "output.npz")
    is_tri = out["types"] == int(ElementType.TRIANGLE)
    start = out["offsets"][:-1][is_tri]
    tris = out["nodes"][start[:, None] + np.arange(3)].astype(np.int64)
    assert set(_edge_use(tris).values()) == {2}
    lengths = _edge_lengths(out["xyz"], tris)
    assert float(lengths.mean()) <= 1.25 * size
    assert float(lengths.max()) <= 2.0 * size


@pytest.mark.parametrize("delay", [0.1, 0.25, 0.5])
def test_a_cancelled_remesh_leaves_the_input_or_a_whole_remesh(delay: float) -> None:
    """A cancel ``delay`` s into the remesh of a 1280-triangle icosphere at 0.02 (~5 s).

    The mesh after the cancel is the input, as the details say, or, when the cancel came
    after the remesher replaced the mesh, a closed remesh; never a part of one. Before
    NETGENPlugin_remesher_partial_result.patch netgen's stopped surface meshing came
    back as NG_OK, and a cancel 0.25 s in left about 21 800 triangles with 372 open
    edges. The same mesher then remeshes to the end.
    """
    xyz0, tris0 = _icosphere(3)
    start = time.perf_counter()

    def cancel_after_the_delay() -> bool:
        return time.perf_counter() - start >= delay

    with Mesher.from_arrays(xyz0, tris0) as mesher:
        mesher.assign(NetgenRemesher2D())
        mesher.assign(NetgenRemesherParameters2D(max_size=0.02))
        with pytest.raises(PysmeshCancelled, match="cancelled") as raised:
            mesher.compute(cancel=cancel_after_the_delay)
        kept = mesher.mesh()
        report = mesher.compute()
    is_tri = kept.element_type == int(ElementType.TRIANGLE)
    begin = kept.element_offsets[:-1][is_tri]
    tris = kept.element_nodes[begin[:, None] + np.arange(3)].astype(np.int64)

    assert set(_edge_use(tris).values()) == {2}
    if "as it was before" in raised.value.details:
        assert np.array_equal(kept.node_coords, xyz0)
        assert len(tris) == len(tris0)
    else:
        assert "remeshed one" in raised.value.details
        assert len(tris) > 10 * len(tris0)
    assert report.faces > 10 * len(tris0)
