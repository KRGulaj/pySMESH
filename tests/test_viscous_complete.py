# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-04

"""Viscous layers on every path that builds them, against the closed form of a stack.

A stack of ``N`` layers growing by the factor ``f`` and totalling ``T`` has the first
layer ``t1 = T (f - 1) / (f^N - 1)``, or ``T / N`` when ``f = 1``
(``StdMeshers_ViscousLayers::Get1stLayerThickness``, SMESH ``V9_16_0``), and layer ``k``
ends at ``t1 (f^k - 1) / (f - 1)`` from the wall, ``k t1`` when ``f = 1``. The tests
read the layer node planes of the meshes and compare them with these positions.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    Area,
    BadOrientedVolume,
    BareBorderVolume,
    Cartesian3D,
    CartesianParameters3D,
    CompositeHexa3D,
    ElementType,
    ExtrusionMethod,
    Hexa3D,
    Hypothesis,
    Mefisto2D,
    Mesh,
    Mesher,
    NumberOfSegments,
    PolygonPerFace2D,
    PolyhedronPerSolid3D,
    Prism3D,
    PysmeshError,
    QuadFromMedialAxis1D2D,
    Quadrangle2D,
    RadialQuadrangle1D2D,
    Regular1D,
    Session,
    ViscousLayerBuilder,
    ViscousLayers,
    ViscousLayers2D,
    VLParams,
    Volume,
    compute_viscous_layers,
    first_layer_thickness,
)

TOL: float = 1e-9


def _layer_ends(total: float, factor: float, count: int) -> NDArray[np.float64]:
    """Where layers 1 .. N end, measured from the wall: the closed form above."""
    k = np.arange(1, count + 1, dtype=np.float64)
    if factor == 1.0:
        return k * total / count
    first = total * (factor - 1.0) / (factor**count - 1.0)
    return first * (factor**k - 1.0) / (factor - 1.0)


# ---- L1 first_layer_thickness ------------------------------------------------------ #


@pytest.mark.parametrize(
    ("total", "factor", "count"),
    [(0.3, 1.2, 3), (1.0, 1.0, 4), (2.0, 1.5, 1), (0.01, 1.05, 30)],
)
def test_first_layer_thickness_is_the_closed_form_and_the_stack_sums_to_t(
    total: float, factor: float, count: int
) -> None:
    """t1 = T (f - 1) / (f^N - 1), T / N at f = 1; the N layers add up to T."""
    expected = (
        total / count
        if factor == 1.0
        else total * (factor - 1.0) / (factor**count - 1.0)
    )

    first = first_layer_thickness(total, factor, count)

    assert first == pytest.approx(expected, rel=1e-14)
    layers = first * factor ** np.arange(count, dtype=np.float64)
    assert float(layers.sum()) == pytest.approx(total, rel=1e-12)


@pytest.mark.parametrize(
    ("total", "factor", "count", "word"),
    [
        (0.0, 1.2, 3, "total_thickness"),
        (-1.0, 1.2, 3, "total_thickness"),
        (0.3, 1.2, 0, "layer_count"),
        (0.3, 0.9, 3, "stretch_factor"),
    ],
)
def test_first_layer_thickness_refuses_a_stack_smesh_cannot_grow(
    total: float, factor: float, count: int, word: str
) -> None:
    """Degenerate input: upstream answers T / N for f < 1 and builds nothing at 0."""
    with pytest.raises(PysmeshError, match=word):
        first_layer_thickness(total, factor, count)


# ---- L2 validation, and f = 1 on every path ---------------------------------------- #

STACK: tuple[float, int] = (0.3, 3)


def _unit_box() -> ps.Shape:
    """A 1 x 1 x 1 box at the origin."""
    session = Session()
    session.add_box(1.0, 1.0, 1.0)
    return ps.load_brep(session.brep())


def _unit_square() -> ps.Shape:
    """A 1 x 1 square face at the origin, in z = 0."""
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0)
    return ps.load_brep(session.brep())


def _at_x0(items: list[object]) -> int:
    """The ordinal of the face or edge that lies in the plane x = 0."""
    for item in items:
        box = item.bbox  # type: ignore[attr-defined]
        if abs(box[0]) < TOL and abs(box[3]) < TOL:
            return int(item.id)  # type: ignore[attr-defined]
    raise AssertionError("nothing in the plane x = 0")


def _planes(xyz: NDArray[np.float64], total: float) -> NDArray[np.float64]:
    """The distinct x of the nodes in the layer stack at x = 0, the wall left out."""
    x = np.unique(np.round(xyz[:, 0], 12))
    return x[(x > TOL) & (x <= total + TOL)]


@pytest.mark.parametrize("cls", [ViscousLayers, ViscousLayers2D])
@pytest.mark.parametrize(
    ("total", "count", "factor", "word"),
    [
        (0.0, 3, 1.2, "total_thickness"),
        (-0.3, 3, 1.2, "total_thickness"),
        (0.3, 0, 1.2, "layer_count"),
        (0.3, 3, 0.8, "stretch_factor"),
    ],
)
def test_a_viscous_hypothesis_refuses_a_stack_smesh_cannot_grow(
    cls: type[ViscousLayers | ViscousLayers2D],
    total: float,
    count: int,
    factor: float,
    word: str,
) -> None:
    """Degenerate input, refused in Python: upstream's setters check nothing
    (``StdMeshers_ViscousLayers.cxx:1304-1339``)."""
    with pytest.raises(PysmeshError, match=word):
        cls(
            total_thickness=total,
            layer_count=count,
            stretch_factor=factor,
            boundary=(1,),
            group_name="bl",
        )


def test_viscous_layers_2d_refuses_an_extrusion_method() -> None:
    """2-D layers have no extrusion method (``additional_hypo.rst``); named, refused."""
    with pytest.raises(PysmeshError, match="3-D ViscousLayers"):
        ViscousLayers2D(
            total_thickness=0.3,
            layer_count=3,
            stretch_factor=1.2,
            boundary=(1,),
            group_name="bl",
            method=ExtrusionMethod.FACE_OFFSET,
        )


@pytest.mark.parametrize("factor", [1.0, 1.2])
def test_hexa_3d_layers_sit_at_the_closed_form_including_f_1(factor: float) -> None:
    """Hexa3D with ViscousLayers on the face x = 0: node planes at the closed form."""
    total, count = STACK
    box = _unit_box()

    with Mesher(box) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=4))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.assign(
            ViscousLayers(
                total_thickness=total,
                layer_count=count,
                stretch_factor=factor,
                boundary=(_at_x0(box.faces()),),
                group_name="bl",
            )
        )
        mesher.compute()
        xyz = mesher.mesh().node_coords

    np.testing.assert_allclose(
        _planes(xyz, total), _layer_ends(total, factor, count), atol=TOL
    )


@pytest.mark.parametrize("factor", [1.0, 1.2])
def test_quadrangle_2d_layers_sit_at_the_closed_form_including_f_1(
    factor: float,
) -> None:
    """Quadrangle2D, ViscousLayers2D on the edge x = 0: lines at the closed form."""
    total, count = STACK
    square = _unit_square()

    with Mesher(square) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=4))
        mesher.assign(Quadrangle2D())
        mesher.assign(
            ViscousLayers2D(
                total_thickness=total,
                layer_count=count,
                stretch_factor=factor,
                boundary=(_at_x0(square.edges()),),
                group_name="bl",
            )
        )
        mesher.compute()
        xyz = mesher.mesh().node_coords

    np.testing.assert_allclose(
        _planes(xyz, total), _layer_ends(total, factor, count), atol=TOL
    )


def _inject(shape: ps.Shape, m: dict[str, NDArray[np.float64]]) -> Mesh:
    """A Mesh carrying the committed classified box surface mesh (see conftest)."""
    mesh = Mesh(shape)
    sid = mesh.add_nodes(m["nodes"])
    for fid in np.unique(m["face_ids"]):
        sel = m["face_ids"] == fid
        mesh.classify_on_face(sid[m["face_node_ids"][sel]], int(fid), m["face_uv"][sel])
    for eid in np.unique(m["edge_ids"]):
        sel = m["edge_ids"] == eid
        mesh.classify_on_edge(sid[m["edge_node_ids"][sel]], int(eid), m["edge_t"][sel])
    for nl, vid in zip(m["vertex_node_ids"], m["vertex_ids"], strict=True):
        mesh.classify_on_vertex(int(sid[nl]), int(vid))
    for eid in np.unique(m["segment_edge_ids"]):
        sel = m["segment_edge_ids"] == eid
        mesh.add_segments(sid[m["segments"][sel]].astype(np.int64), int(eid))
    for fid in np.unique(m["tri_face_ids"]):
        sel = m["tri_face_ids"] == fid
        mesh.add_triangles(sid[m["tris"][sel]].astype(np.int64), int(fid))
    mesh.validate()
    return mesh


def test_compute_viscous_layers_accepts_f_1_and_grows_equal_layers(
    box_brep: bytes, box_mesh: dict[str, NDArray[np.float64]]
) -> None:
    """The standalone path at f = 1, layers on all six faces of the box.

    Each node lies at a depth from the nearest box face, along the wall normal; the
    depths of all the nodes are exactly 0, T/N, ..., T and the interior beyond, so
    the layer planes sit at k T/N. N prisms stand on each wall triangle.
    """
    total, count = 0.1, 5
    shape = ps.load_brep(box_brep)
    mesh = _inject(shape, box_mesh)

    result = compute_viscous_layers(
        mesh,
        VLParams(
            face_ids=tuple(f.id for f in shape.faces()),
            total_thickness=total,
            n_layers=count,
            stretch_factor=1.0,
            group_name="BL",
        ),
    )

    xyz = result.node_coords
    lo, hi = box_mesh["nodes"].min(axis=0), box_mesh["nodes"].max(axis=0)
    depth = np.minimum(xyz - lo, hi - xyz).min(axis=1)
    in_stack = np.unique(np.round(depth[depth <= total + TOL], 9))
    assert result.prism_connectivity.shape[0] == count * box_mesh["tris"].shape[0]
    np.testing.assert_allclose(
        in_stack, np.r_[0.0, _layer_ends(total, 1.0, count)], atol=1e-9
    )


# ---- L3 ignore=True on the catalogue path ------------------------------------------ #

# The child grows layers on five faces of the unit box with the catalogue path, once by
# ignoring the face x = 0 and once by listing the other five, and prints, per axis, the
# node planes within T of each side. It sets up the DLL search as tests/conftest.py
# does.
_IGNORE_CHILD: str = """
import json, os, sys
occt = os.environ.get("PYSMESH_OCCT_BIN")
if occt:
    os.add_dll_directory(occt)
lib = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(lib):
    os.add_dll_directory(lib)
sys.path.insert(0, sys.argv[1])
import numpy as np
import pysmesh as ps

T, N, F = 0.3, 3, 1.2
s = ps.Session()
s.add_box(1.0, 1.0, 1.0)
box = ps.load_brep(s.brep())
x0 = next(f.id for f in box.faces() if abs(f.bbox[0]) < 1e-9 and abs(f.bbox[3]) < 1e-9)
out = {}
others = tuple(f.id for f in box.faces() if f.id != x0)
for ignore, walls in ((True, (x0,)), (False, others)):
    with ps.Mesher(box) as m:
        m.assign(ps.Regular1D())
        m.assign(ps.NumberOfSegments(count=4))
        m.assign(ps.Quadrangle2D())
        m.assign(ps.Hexa3D())
        m.assign(ps.ViscousLayers(total_thickness=T, layer_count=N, stretch_factor=F,
                                  boundary=walls, ignore=ignore, group_name="bl"))
        m.compute()
        xyz = m.mesh().node_coords
    sides = []
    for axis in range(3):
        v = np.unique(np.round(xyz[:, axis], 9))
        sides.append(v[(v > 1e-12) & (v <= T + 1e-9)].tolist())
        sides.append((1.0 - v[(v < 1 - 1e-12) & (v >= 1 - T - 1e-9)]).tolist())
    nodes = np.sort(np.round(xyz, 9), axis=0).tolist()
    out[str(ignore)] = {"sides": sides, "nodes": nodes}
print("IGNORE-RESULT " + json.dumps(out))
"""


def test_ignore_on_the_catalogue_path_grows_layers_on_the_other_faces_every_time() -> (
    None
):
    """Five fresh processes: each exits cleanly and grows the stack on five faces.

    ``viscous.cpp`` used to avoid ``toIgnore=true`` because it corrupted the heap. On
    SMESH 9.16, in a child process each time: the closed-form planes stand at the five
    wall sides (x = 1, y = 0, y = 1, z = 0, z = 1) and not at the ignored side x = 0,
    and the mesh is node for node the one the explicit list of five walls gives.
    """
    total, count = 0.3, 3
    ends = _layer_ends(total, 1.2, count)
    package_root = str(Path(ps.__file__).resolve().parent.parent)

    for _ in range(5):
        proc = subprocess.run(
            [sys.executable, "-c", _IGNORE_CHILD, package_root],
            capture_output=True,
            text=True,
            timeout=300.0,
            env=dict(os.environ),
            check=False,
        )

        assert proc.returncode == 0, proc.stderr[-2000:]
        line = next(
            ln for ln in proc.stdout.splitlines() if ln.startswith("IGNORE-RESULT ")
        )
        result = json.loads(line.removeprefix("IGNORE-RESULT "))
        ignored, listed = result["True"], result["False"]
        for side, planes in enumerate(ignored["sides"]):
            hits = [any(abs(p - e) < TOL for p in planes) for e in ends]
            assert all(hits) == (side != 0), (side, planes)
        assert ignored["nodes"] == listed["nodes"]


# ---- L4 the two-step builder ------------------------------------------------------- #


def _inner_hexa(mesher: Mesher) -> None:
    """Assign a 2 x 2 x 2 structured mesh to ``mesher``."""
    mesher.assign(Regular1D())
    mesher.assign(NumberOfSegments(count=2))
    mesher.assign(Quadrangle2D())
    mesher.assign(Hexa3D())


def _inner_cartesian(mesher: Mesher) -> None:
    """A Cartesian mesh on ``mesher``, with the faces and edges add_layers reads."""
    mesher.assign(Cartesian3D())
    mesher.assign(
        CartesianParameters3D(
            spacing_x="0.35",
            spacing_y="0.5",
            spacing_z="0.5",
            create_faces=True,
            add_edges=True,
        )
    )


def _build_box(
    builder: ViscousLayerBuilder, inner_kind: str
) -> tuple[ps.Shape, ps.MeshData, float, dict[str, int]]:
    """Shrink the unit box, mesh the shrunk box, add the layers; volume and groups."""
    box = _unit_box()
    with Mesher(box) as outer:
        shrunk = outer.shrink_geometry(builder)
        with Mesher(shrunk) as inner:
            (_inner_hexa if inner_kind == "hexa" else _inner_cartesian)(inner)
            inner.compute()
            outer.add_layers(builder, inner)
        mesh = outer.mesh()
        volume = float(np.sum(outer.quality(Volume()).values))
        groups = {g.name: int(g.element_ids.size) for g in outer.groups()}
    return shrunk, mesh, volume, groups


@pytest.mark.parametrize("inner_kind", ["hexa", "cartesian"])
@pytest.mark.parametrize("factor", [1.0, 1.2])
def test_builder_layers_on_one_face_of_a_box_sit_at_the_closed_form(
    inner_kind: str, factor: float
) -> None:
    """Layers on x = 0: shrunk box [T, 1] x [0, 1]^2, planes at the closed form.

    Spec (SMESH ``additional_hypo.rst``, "Viscous Layers API";
    ``test_vlapi_growthlayer``): the builder offsets the boundary faces inward by ``T``,
    the inner mesh fills the shrunk shape, and ``AddLayers`` fills the gap with N
    layers. The wall face carries a 2 x 2 quadrangle mesh, so the group holds N x 4
    prisms, and the mesh fills the box: volume 1.
    """
    total, count = STACK
    x0 = _at_x0(_unit_box().faces())
    builder = ViscousLayerBuilder(
        total_thickness=total,
        layer_count=count,
        stretch_factor=factor,
        boundary=(x0,),
        ignore=False,
        group_name="layers",
    )

    shrunk, mesh, volume, groups = _build_box(builder, inner_kind)

    boxes = np.array([f.bbox for f in shrunk.faces()])
    corners = np.r_[boxes[:, :3].min(axis=0), boxes[:, 3:].max(axis=0)]
    np.testing.assert_allclose(corners, [total, 0, 0, 1, 1, 1], atol=TOL)
    np.testing.assert_allclose(
        _planes(mesh.node_coords, total), _layer_ends(total, factor, count), atol=TOL
    )
    assert groups == {"layers": count * 4}
    assert volume == pytest.approx(1.0, rel=1e-9)


def test_builder_ignore_form_grows_layers_on_every_other_face() -> None:
    """``ignore=True``, x = 0 listed: the stack on the five other faces only."""
    total, count = STACK
    x0 = _at_x0(_unit_box().faces())
    builder = ViscousLayerBuilder(
        total_thickness=total,
        layer_count=count,
        stretch_factor=1.2,
        boundary=(x0,),
        group_name="layers",
    )

    _, mesh, volume, groups = _build_box(builder, "hexa")

    ends = _layer_ends(total, 1.2, count)
    xyz = mesh.node_coords
    for axis in range(3):
        values = np.unique(np.round(xyz[:, axis], 12))
        far = np.sort(1.0 - values[values >= 1.0 - total - TOL])[1:]
        np.testing.assert_allclose(far, ends, atol=TOL)
        near = values[(values > TOL) & (values <= total + TOL)]
        if axis == 0:
            assert near.size == 0
        else:
            np.testing.assert_allclose(near, ends, atol=TOL)
    assert groups == {"layers": count * 4 * 5}
    assert volume == pytest.approx(1.0, rel=1e-9)


def _rings(
    shape: ps.Shape, builder: ViscousLayerBuilder, inner_2d: object, segments: int
) -> tuple[ps.Shape, ps.MeshData, int, float, dict[str, int]]:
    """Shrink a face, mesh it with ``inner_2d``, add the layers."""
    with Mesher(shape) as outer:
        shrunk = outer.shrink_geometry(builder)
        with Mesher(shrunk) as inner:
            inner.assign(Regular1D())
            inner.assign(NumberOfSegments(count=segments))
            inner.assign(inner_2d)  # type: ignore[arg-type]
            inner.compute()
            inner_faces = inner.compute().faces
            outer.add_layers(builder, inner)
        mesh = outer.mesh()
        area = float(np.sum(outer.quality(Area()).values))
        groups = {g.name: int(g.element_ids.size) for g in outer.groups()}
    return shrunk, mesh, inner_faces, area, groups


def test_builder_in_a_square_grows_rings_at_the_closed_form() -> None:
    """A 1 x 1 square, 2 layers 0.2 thick: the shrunk square [0.2, 0.8]^2, rings at 0.1
    and 0.2 from each edge, 4 x 3 x 2 = 24 layer quadrangles in the group, area 1.

    The second check is the upstream count (``test_vlapi_growthlayer.py``): the layers
    add 4 x (segments per edge) x (layers) faces to the inner mesh.
    """
    builder = ViscousLayerBuilder(
        total_thickness=0.2, layer_count=2, stretch_factor=1.0, group_name="rings"
    )

    shrunk, mesh, inner_faces, area, groups = _rings(
        _unit_square(), builder, Quadrangle2D(), 3
    )

    np.testing.assert_allclose(
        shrunk.faces()[0].bbox, [0.2, 0.2, 0, 0.8, 0.8, 0], atol=TOL
    )
    for axis in range(2):
        values = np.unique(np.round(mesh.node_coords[:, axis], 12))
        np.testing.assert_allclose(values[:3], [0.0, 0.1, 0.2], atol=TOL)
        np.testing.assert_allclose(values[-3:], [0.8, 0.9, 1.0], atol=TOL)
    assert groups == {"rings": 24}
    assert mesh.count_of(ElementType.QUADRANGLE) == inner_faces + 4 * 3 * 2
    assert area == pytest.approx(1.0, rel=1e-9)


def test_builder_in_a_disk_grows_circular_rings() -> None:
    """A disk of radius 5, 6 layers 0.5 thick at f = 1.2, 12 segments on its circle.

    The rings stand at the radii ``5 - (closed form)``, measured on the nodes; the
    layers add segments x layers = 72 faces (``test_vlapi_growthlayer.py``, the disk
    case).
    """
    total, count, factor = 0.5, 6, 1.2
    session = Session()
    session.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 5.0)
    session.make_face(list(session.entities(ps.EntityKind.EDGE)))
    builder = ViscousLayerBuilder(
        total_thickness=total, layer_count=count, stretch_factor=factor
    )

    _, mesh, inner_faces, _, _ = _rings(
        ps.load_brep(session.brep()), builder, Mefisto2D(), 12
    )

    radii = np.unique(np.round(np.hypot(*mesh.node_coords[:, :2].T), 9))
    np.testing.assert_allclose(
        np.sort(5.0 - radii[radii >= 5.0 - total - TOL])[1:],
        _layer_ends(total, factor, count),
        atol=1e-9,
    )
    assert mesh.element_count - mesh.count_of(ElementType.EDGE) == (
        inner_faces + 12 * count
    )


def test_a_second_shrink_replaces_the_first_and_the_layers_follow_it() -> None:
    """Two shrinks on one mesher: add_layers builds on the second (lifecycle patch)."""
    x0 = _at_x0(_unit_box().faces())
    first = ViscousLayerBuilder(0.2, 2, 1.0, boundary=(x0,), ignore=False)
    second = ViscousLayerBuilder(0.3, 3, 1.0, boundary=(x0,), ignore=False)

    with Mesher(_unit_box()) as outer:
        outer.shrink_geometry(first)
        shrunk = outer.shrink_geometry(second)
        with Mesher(shrunk) as inner:
            _inner_hexa(inner)
            inner.compute()
            outer.add_layers(second, inner)
        xyz = outer.mesh().node_coords

    np.testing.assert_allclose(_planes(xyz, 0.3), [0.1, 0.2, 0.3], atol=TOL)


def test_add_layers_before_shrink_geometry_is_refused() -> None:
    """There is no shrunk shape yet; upstream read an unset pointer here."""
    builder = ViscousLayerBuilder(0.2, 2, 1.0)

    with (
        Mesher(_unit_box()) as outer,
        Mesher(_unit_box()) as inner,
        pytest.raises(PysmeshError, match="call shrink_geometry"),
    ):
        outer.add_layers(builder, inner)


def test_add_layers_refuses_an_inner_mesher_on_another_shape() -> None:
    """The inner mesh is paired with the shrunk shape by TopExp index: refused."""
    builder = ViscousLayerBuilder(0.2, 2, 1.0)

    with Mesher(_unit_box()) as outer, Mesher(_unit_box()) as inner:
        outer.shrink_geometry(builder)
        _inner_hexa(inner)
        inner.compute()

        with pytest.raises(PysmeshError, match="not on the shape"):
            outer.add_layers(builder, inner)


def test_add_layers_refuses_a_builder_other_than_the_shrinks() -> None:
    """The layers must be built with the parameters the shape was shrunk by."""
    with Mesher(_unit_box()) as outer:
        shrunk = outer.shrink_geometry(ViscousLayerBuilder(0.2, 2, 1.0))
        with Mesher(shrunk) as inner:
            _inner_hexa(inner)
            inner.compute()

            with pytest.raises(PysmeshError, match="differs"):
                outer.add_layers(ViscousLayerBuilder(0.3, 2, 1.0), inner)


# ---- L5 Cartesian_3D with viscous layers ------------------------------------------ #

CARTESIAN_STACK: tuple[float, int, float] = (0.2, 3, 1.2)
_FACETS: dict[int, tuple[tuple[int, ...], ...]] = {
    int(ElementType.TETRAHEDRON): ((0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
    int(ElementType.PYRAMID): (
        (0, 1, 2, 3),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ),
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
_SURFACE: frozenset[int] = frozenset(
    {int(ElementType.TRIANGLE), int(ElementType.QUADRANGLE), int(ElementType.POLYGON)}
)


def _hex_prism(turn: float) -> ps.Shape:
    """A regular hexagonal prism: circumradius 1, height 1, turned ``turn`` deg on z."""
    a0 = math.radians(turn)
    corners = np.array(
        [
            [math.cos(a0 + k * math.pi / 3), math.sin(a0 + k * math.pi / 3), 0.0]
            for k in range(6)
        ],
        dtype=np.float64,
    )
    session = Session()
    session.add_polyline(corners, closed=True)
    session.make_face(list(session.entities(ps.EntityKind.EDGE)))
    session.extrude(list(session.entities(ps.EntityKind.FACE)), (0.0, 0.0, 1.0))
    return ps.load_brep(session.brep())


def _shape_of(build: str) -> ps.Shape:
    """A wedge, a cylinder, a sphere, or a 2 x 2 x 2 block with a bore of radius 0.4."""
    session = Session()
    if build == "wedge":
        session.add_wedge(2.0, 2.0, 2.0, 0.5)
    elif build == "cylinder":
        session.add_cylinder(1.0, 2.0)
    elif build == "sphere":
        session.add_sphere(1.0)
    else:
        session.add_box(2.0, 2.0, 2.0)
        block = session.entities(ps.EntityKind.SOLID).tolist()
        session.add_cylinder(0.4, 2.0, origin=(1.0, 1.0, 0.0))
        everything = session.entities(ps.EntityKind.SOLID).tolist()
        session.cut(block, [i for i in everything if i not in block])
    return ps.load_brep(session.brep())


def _assign_cartesian_layers(
    mesher: Mesher, spacing: str, ignored: tuple[int, ...], total: float
) -> None:
    """Cartesian3D at ``spacing``, the stack on every face but ``ignored``."""
    _, count, factor = CARTESIAN_STACK
    mesher.assign(Cartesian3D())
    mesher.assign(
        CartesianParameters3D(spacing_x=spacing, spacing_y=spacing, spacing_z=spacing)
    )
    mesher.assign(
        ViscousLayers(
            total_thickness=total,
            layer_count=count,
            stretch_factor=factor,
            boundary=ignored,
            ignore=True,
            group_name="bl",
        )
    )


def _cartesian_layers(
    shape: ps.Shape, spacing: str, ignored: tuple[int, ...] = ()
) -> tuple[ps.MeshData, ps.ComputeReport, int, int]:
    """Mesh ``shape`` with Cartesian layers: the mesh, the report, two bad-cell counts.

    The counts are the cells with a boundary facet that no face covers, and the inverted
    cells.
    """
    with Mesher(shape) as mesher:
        _assign_cartesian_layers(mesher, spacing, ignored, CARTESIAN_STACK[0])
        report = mesher.compute()
        bare = mesher.select(BareBorderVolume()).count
        inverted = mesher.select(BadOrientedVolume()).count
        mesh = mesher.mesh()
    return mesh, report, bare, inverted


def _shared_split_volume(mesh: ps.MeshData) -> float:
    """The cell volumes summed with each facet split the same way in both its cells.

    Each facet is fanned into triangles from its centroid, and each triangle makes a
    tetrahedron with the cell centroid (the layer and cut cells are star-shaped about
    it). Two cells that share a warped facet then split it alike, so on a conforming
    mesh the sum is the volume its skin encloses: a void or an overlap changes it. The
    Volume control splits each cell on its own, so on the warped corner cells it does
    not add up.
    """
    xyz = mesh.node_coords
    total = 0.0
    for row in range(mesh.element_count):
        kind = int(mesh.element_type[row])
        nodes = mesh.nodes_of(row)
        if kind == int(ElementType.POLYHEDRON):
            facets = np.split(nodes, np.cumsum(mesh.face_sizes_of(row))[:-1])
        elif kind in _FACETS:
            facets = [nodes[list(f)] for f in _FACETS[kind]]
        else:
            continue
        centre = xyz[np.unique(nodes)].mean(axis=0)
        for facet in facets:
            ring = xyz[facet] - centre
            cross = np.cross(ring, np.roll(ring, -1, axis=0))
            total += float(np.abs(cross @ ring.mean(axis=0)).sum()) / 6.0
    return total


def _offsets(
    distance: NDArray[np.float64], keep: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """The distinct wall distances of the kept nodes, up to the total thickness."""
    near = keep & (distance > -TOL) & (distance <= CARTESIAN_STACK[0] + TOL)
    return np.unique(np.round(distance[near], 9))


def _stack_planes() -> NDArray[np.float64]:
    """The wall, then the end of each layer: the closed form of CARTESIAN_STACK."""
    total, count, factor = CARTESIAN_STACK
    return np.concatenate(([0.0], _layer_ends(total, factor, count)))


def test_cartesian_layers_on_the_inclined_wall_of_a_wedge_sit_at_the_closed_form() -> (
    None
):
    """The wall 0.8 x + 0.6 y = 1.6 of add_wedge(2, 2, 2, 0.5), mid-wall nodes.

    A layer edge runs along the normal of a plane wall, and the layer nodes divide it at
    the closed-form fractions, so the planes are exact. The wedge has volume
    (2 + 0.5) / 2 * 2 * 2 = 5.
    """
    mesh, _, bare, inverted = _cartesian_layers(_shape_of("wedge"), "0.25")

    xyz = mesh.node_coords
    distance = 1.6 - xyz @ np.array([0.8, 0.6, 0.0])
    along = xyz @ np.array([-0.6, 0.8, 0.0])
    keep = (np.abs(along) < 0.5) & (xyz[:, 2] > 0.2 + TOL) & (xyz[:, 2] < 1.8 - TOL)
    np.testing.assert_allclose(_offsets(distance, keep), _stack_planes(), atol=TOL)
    assert (bare, inverted) == (0, 0)
    assert _shared_split_volume(mesh) == pytest.approx(5.0, rel=TOL)


@pytest.mark.parametrize("caps", ["layered", "ignored"])
@pytest.mark.parametrize("turn", [0.0, 15.0, 30.0])
def test_cartesian_layers_fill_a_hexagonal_prism_with_no_void(
    turn: float, caps: str
) -> None:
    """Inclined walls at grid spacing 0.1, the shape the layers failed on.

    At 0 and 30 deg a vertical edge lies on an end plane of the grid, where the offset
    mesh doubled its nodes (duplicate_nodes patch); cut cells dropped from the offset
    mesh left voids in the layers (offset_small_cells patch). The prism has volume
    3 sqrt(3) / 2. The wall between corners 0 and 1 has its outward normal at
    ``turn + 30`` deg, at the apothem cos(30 deg) from the axis.
    """
    shape = _hex_prism(turn)
    flat = tuple(int(f.id) for f in shape.faces() if abs(f.bbox[2] - f.bbox[5]) < TOL)
    ignored = flat if caps == "ignored" else ()

    mesh, _, bare, inverted = _cartesian_layers(shape, "0.1", ignored)

    xyz = mesh.node_coords
    angle = math.radians(turn + 30.0)
    normal = np.array([math.cos(angle), math.sin(angle), 0.0])
    tangent = np.array([-normal[1], normal[0], 0.0])
    rim = 0.0 if ignored else CARTESIAN_STACK[0]
    keep = (np.abs(xyz @ tangent) < 0.2) & (xyz[:, 2] > rim + TOL)
    keep &= xyz[:, 2] < 1.0 - rim - TOL
    distance = math.cos(math.pi / 6.0) - xyz @ normal
    np.testing.assert_allclose(_offsets(distance, keep), _stack_planes(), atol=TOL)
    assert (bare, inverted) == (0, 0)
    assert _shared_split_volume(mesh) == pytest.approx(1.5 * math.sqrt(3.0), rel=TOL)


def test_cartesian_layers_on_a_cylinder_sit_at_the_closed_form_and_skin_it() -> None:
    """Radius 1, height 2, spacing 0.25: the layer edges of the side run radially.

    The radial offsets of mid-height nodes are the closed form. Every face element lies
    on the skin: on a cap, or on the side within the sagitta of its chords (a chord of
    at most 0.4 has a sagitta below 0.02). A cut cell dropped from the offset mesh used
    to leave faces inside, at radius 0.78, and layer cells with uncovered facets.
    """
    mesh, _, bare, inverted = _cartesian_layers(_shape_of("cylinder"), "0.25")

    xyz = mesh.node_coords
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    keep = (xyz[:, 2] > 0.2 + TOL) & (xyz[:, 2] < 1.8 - TOL)
    np.testing.assert_allclose(_offsets(1.0 - radius, keep), _stack_planes(), atol=TOL)
    assert (bare, inverted) == (0, 0)
    rows = [
        r for r in range(mesh.element_count) if int(mesh.element_type[r]) in _SURFACE
    ]
    centres = np.array([xyz[mesh.nodes_of(r)].mean(axis=0) for r in rows])
    on_cap = (centres[:, 2] < TOL) | (centres[:, 2] > 2.0 - TOL)
    on_side = np.hypot(centres[:, 0], centres[:, 1]) > 0.98
    assert bool(np.all(on_cap | on_side))


@pytest.mark.parametrize("build", ["sphere", "bored"])
def test_cartesian_layers_on_curved_faces_leave_no_sub_mesh_failed(build: str) -> None:
    """A sphere (its seam and poles), a block with a bore (the bore's seam).

    The mesh was complete, but the EDGE and VERTEX sub-meshes with no element of their
    own were left unmarked, so compute() failed on them with no message
    (viscous_submeshes patch). Now it succeeds with no warning, and every boundary
    facet is covered.
    """
    _, report, bare, inverted = _cartesian_layers(_shape_of(build), "0.25")

    assert report.volumes > 0
    assert report.warnings == ()
    assert (bare, inverted) == (0, 0)


def test_a_failed_cartesian_layer_compute_leaves_no_cell() -> None:
    """Layers of 0.3 in the bored block: the offset bore meets the offset walls.

    The offset shape gives no cell, so the compute fails loudly on the SOLID, and the
    target holds no copy of an offset mesh.
    """
    with Mesher(_shape_of("bored")) as mesher:
        _assign_cartesian_layers(mesher, "0.25", (), 0.3)

        with pytest.raises(PysmeshError, match="meshing failed") as raised:
            mesher.compute()

        assert "SOLID 1" in raised.value.details
        assert mesher.mesh().element_count == 0


def test_layers_too_thick_for_the_shape_fail_naming_the_reason() -> None:
    """Layers of 0.3 in the bored block: the compute fails and says why.

    Offset by 0.3, the bore (radius 0.4) reaches radius 0.7 and the walls come to 0.7
    from its axis, so the offset surfaces touch. BRepOffset_MakeOffset then reports
    success with an empty solid, which used to reach the caller as "SOLID 1: no
    message".
    """
    with Mesher(_shape_of("bored")) as mesher:
        _assign_cartesian_layers(mesher, "0.25", (), 0.3)

        with pytest.raises(PysmeshError, match="meshing failed") as raised:
            mesher.compute()

    assert "no message" not in raised.value.details
    assert "too thick for the shape" in raised.value.details


# ---- L6 the algorithms that build viscous layers ----------------------------------- #

# Group sizes of a stack of STACK[1] layers on one wall: 4 x 4 quadrangles on the box
# face x = 0 (4 segments, or grid spacing 0.25), 4 segments on the square edge x = 0, 8
# on the long edge of the 4 x 1 strip.
_LAYERS_3D: tuple[str, ...] = ("Hexa3D", "PolyhedronPerSolid3D", "Cartesian3D")


def _assign_3d(mesher: Mesher, algorithm: str) -> None:
    """``algorithm`` on the unit box: 4 segments per edge, or a 0.25 Cartesian grid."""
    if algorithm == "Cartesian3D":
        mesher.assign(Cartesian3D())
        mesher.assign(
            CartesianParameters3D(spacing_x="0.25", spacing_y="0.25", spacing_z="0.25")
        )
        return
    mesher.assign(Regular1D())
    mesher.assign(NumberOfSegments(count=4))
    mesher.assign(Quadrangle2D())
    mesher.assign(
        {
            "Hexa3D": Hexa3D,
            "PolyhedronPerSolid3D": PolyhedronPerSolid3D,
            "CompositeHexa3D": CompositeHexa3D,
            "Prism3D": Prism3D,
        }[algorithm]()
    )


@pytest.mark.parametrize("algorithm", _LAYERS_3D)
def test_a_3d_layer_algorithm_grows_the_stack_at_the_closed_form(
    algorithm: str,
) -> None:
    """ViscousLayers on the face x = 0 of the unit box: 3 planes, 4 x 4 x 3 cells."""
    total, count = STACK
    box = _unit_box()

    with Mesher(box) as mesher:
        _assign_3d(mesher, algorithm)
        mesher.assign(
            ViscousLayers(
                total_thickness=total,
                layer_count=count,
                stretch_factor=1.2,
                boundary=(_at_x0(box.faces()),),
                group_name="bl",
            )
        )
        report = mesher.compute()
        xyz = mesher.mesh().node_coords
        groups = {g.name: int(g.element_ids.size) for g in mesher.groups()}

    np.testing.assert_allclose(
        _planes(xyz, total), _layer_ends(total, 1.2, count), atol=TOL
    )
    assert groups == {"bl": 4 * 4 * count}
    assert report.warnings == ()


def _strip() -> ps.Shape:
    """A 4 x 1 strip at the origin, in z = 0: two long sides and two short ends."""
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 4.0, 1.0)
    return ps.load_brep(session.brep())


def _at_y0(items: list[object]) -> int:
    """The ordinal of the edge that lies on the line y = 0."""
    for item in items:
        box = item.bbox  # type: ignore[attr-defined]
        if abs(box[1]) < TOL and abs(box[4]) < TOL:
            return int(item.id)  # type: ignore[attr-defined]
    raise AssertionError("nothing on y = 0")


@pytest.mark.parametrize(
    ("algorithm", "segments"),
    [("Quadrangle2D", 4), ("Mefisto2D", 4), ("QuadFromMedialAxis1D2D", 8)],
)
def test_a_2d_layer_algorithm_grows_the_stack_at_the_closed_form(
    algorithm: str, segments: int
) -> None:
    """ViscousLayers2D on the edge y = 0: 3 lines, one cell per segment per layer.

    Quadrangle2D and Mefisto2D on the unit square, 4 segments per edge;
    QuadFromMedialAxis1D2D on the 4 x 1 strip, the face it is made for, 8 segments per
    edge. Mefisto2D, carried forward from SMESH V9_9_0, lists ViscousLayers2D as
    compatible and builds the layers in its compute (amendment 12, E8).
    """
    total, count = STACK
    face = _strip() if algorithm == "QuadFromMedialAxis1D2D" else _unit_square()

    with Mesher(face) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=segments))
        mesher.assign(
            {
                "Quadrangle2D": Quadrangle2D,
                "Mefisto2D": Mefisto2D,
                "QuadFromMedialAxis1D2D": QuadFromMedialAxis1D2D,
            }[algorithm]()
        )
        mesher.assign(
            ViscousLayers2D(
                total_thickness=total,
                layer_count=count,
                stretch_factor=1.2,
                boundary=(_at_y0(face.edges()),),
                group_name="bl",
            )
        )
        report = mesher.compute()
        y = np.unique(np.round(mesher.mesh().node_coords[:, 1], 12))
        groups = {g.name: int(g.element_ids.size) for g in mesher.groups()}

    np.testing.assert_allclose(
        y[(y > TOL) & (y <= total + TOL)], _layer_ends(total, 1.2, count), atol=TOL
    )
    assert groups == {"bl": segments * count}
    assert report.warnings == ()


def _disk() -> ps.Shape:
    """A disk of radius 2 at the origin, in z = 0."""
    session = Session()
    session.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0)
    session.make_face(list(session.entities(ps.EntityKind.EDGE)))
    return ps.load_brep(session.brep())


@pytest.mark.parametrize(
    "algorithm", ["Prism3D", "PolygonPerFace2D", "RadialQuadrangle1D2D"]
)
def test_an_algorithm_that_builds_no_layers_refuses_them_before_computing(
    algorithm: str,
) -> None:
    """The layers were dropped with no word (Prism3D, RadialQuadrangle1D2D: no group, no
    layer planes), or PolygonPerFace2D built them and then failed ("Less that 3 nodes on
    the wire") with the layer cells left in the mesh. Now the compute refuses, names the
    algorithm, and meshes nothing.
    """
    total, count = STACK
    shape = {
        "Prism3D": _unit_box,
        "PolygonPerFace2D": _unit_square,
        "RadialQuadrangle1D2D": _disk,
    }[algorithm]()
    with Mesher(shape) as mesher:
        if algorithm == "Prism3D":
            _assign_3d(mesher, algorithm)
            layers: Hypothesis = ViscousLayers(
                total_thickness=total,
                layer_count=count,
                stretch_factor=1.2,
                boundary=(1,),
                group_name="bl",
            )
        else:
            mesher.assign(Regular1D())
            mesher.assign(NumberOfSegments(count=8))
            mesher.assign(
                PolygonPerFace2D()
                if algorithm == "PolygonPerFace2D"
                else RadialQuadrangle1D2D()
            )
            layers = ViscousLayers2D(
                total_thickness=total,
                layer_count=count,
                stretch_factor=1.2,
                boundary=(1,),
                group_name="bl",
            )
        mesher.assign(layers)

        with pytest.raises(
            PysmeshError, match="does not build viscous layers"
        ) as raised:
            mesher.compute()

        assert mesher.mesh().element_count == 0
    native = {
        "Prism3D": "Prism_3D",
        "PolygonPerFace2D": "PolygonPerFace_2D",
        "RadialQuadrangle1D2D": "RadialQuadrangle_1D2D",
    }[algorithm]
    assert native in str(raised.value)


# The child assigns ViscousLayers with CompositeHexa3D on the unit box and computes. On
# the reference the process died with an access violation: the layers fail there, and
# CompositeHexa_3D then read a null proxy mesh.
_COMPOSITE_CHILD: str = """
import os, sys
occt = os.environ.get("PYSMESH_OCCT_BIN")
if occt:
    os.add_dll_directory(occt)
lib = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(lib):
    os.add_dll_directory(lib)
sys.path.insert(0, sys.argv[1])
import pysmesh as ps

s = ps.Session()
s.add_box(1.0, 1.0, 1.0)
with ps.Mesher(ps.load_brep(s.brep())) as m:
    m.assign(ps.Regular1D())
    m.assign(ps.NumberOfSegments(count=4))
    m.assign(ps.Quadrangle2D())
    m.assign(ps.CompositeHexa3D())
    m.assign(ps.ViscousLayers(total_thickness=0.3, layer_count=3, stretch_factor=1.2,
                              boundary=(1,), group_name="bl"))
    try:
        m.compute()
        print("COMPOSITE-RESULT computed")
    except ps.PysmeshError as e:
        print("COMPOSITE-RESULT refused " + str(e))
"""


def test_composite_hexa_3d_refuses_layers_instead_of_crashing() -> None:
    """In a child process: CompositeHexa3D with ViscousLayers raises, it does not crash.

    Made to read the hypothesis, CompositeHexa_3D still cannot use the layers: the layer
    quadrangles give the side faces more rows than the opposite faces, and its box grid
    gets null nodes (StdMeshers_CompositeHexa_3D_viscous_layers.patch).
    """
    package_root = str(Path(ps.__file__).resolve().parent.parent)

    proc = subprocess.run(
        [sys.executable, "-c", _COMPOSITE_CHILD, package_root],
        capture_output=True,
        text=True,
        timeout=300.0,
        env=dict(os.environ),
        check=False,
    )

    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])
    line = next(
        ln for ln in proc.stdout.splitlines() if ln.startswith("COMPOSITE-RESULT ")
    )
    assert line.startswith("COMPOSITE-RESULT refused ")
    assert "CompositeHexa_3D does not build viscous layers" in line


# ---- VL6 several ViscousLayers hypotheses on one solid ------------------------------ #


def _at_x1(items: list[object]) -> int:
    """The ordinal of the face that lies in the plane x = 1."""
    for item in items:
        box = item.bbox  # type: ignore[attr-defined]
        if abs(box[0] - 1.0) < TOL and abs(box[3] - 1.0) < TOL:
            return int(item.id)  # type: ignore[attr-defined]
    raise AssertionError("nothing in the plane x = 1")


def _layer_set(
    total: float, count: int, factor: float, walls: tuple[int, ...], group: str
) -> ViscousLayers:
    """A ViscousLayers hypothesis on ``walls``, its cells in ``group``."""
    return ViscousLayers(
        total_thickness=total,
        layer_count=count,
        stretch_factor=factor,
        boundary=walls,
        group_name=group,
    )


@pytest.mark.parametrize("count_b", [2, 3])
def test_two_face_sets_on_one_solid_each_grow_their_own_closed_form_stack(
    count_b: int,
) -> None:
    """PolyhedronPerSolid3D with one hypothesis on x = 0 and another on x = 1.

    The faces x = 0 and x = 1 share no edge, so the layer counts may differ. Each wall gets
    the planes of its own stack; the layer cells of each go into its own group (4 x 4
    quadrangles per wall); the cells fill the unit box exactly.
    """
    box = _unit_box()
    wall_a, wall_b = _at_x0(box.faces()), _at_x1(box.faces())

    with Mesher(box) as mesher:
        _assign_3d(mesher, "PolyhedronPerSolid3D")
        mesher.assign(_layer_set(0.3, 3, 1.2, (wall_a,), "bl_a"))
        mesher.assign(_layer_set(0.2, count_b, 1.0, (wall_b,), "bl_b"))
        report = mesher.compute()
        x = np.unique(np.round(mesher.mesh().node_coords[:, 0], 12))
        groups = {g.name: int(g.element_ids.size) for g in mesher.groups()}
        volume = float(mesher.quality(Volume()).values.sum())
        inverted = mesher.select(BadOrientedVolume()).count

    np.testing.assert_allclose(
        x[(x > TOL) & (x <= 0.3 + TOL)], _layer_ends(0.3, 1.2, 3), atol=TOL
    )
    near_b = np.sort(1.0 - x[(x >= 0.8 - TOL) & (x < 1.0 - TOL)])
    np.testing.assert_allclose(near_b, _layer_ends(0.2, 1.0, count_b), atol=TOL)
    assert groups == {"bl_a": 16 * 3, "bl_b": 16 * count_b}
    assert volume == pytest.approx(1.0, rel=1e-12)
    assert (inverted, report.warnings) == (0, ())


@pytest.mark.parametrize(
    ("set_b", "count_b", "reason"),
    [
        ("y0", 3, "Several hypotheses define Viscous Layers on the face"),
        ("y0_only_n2", 2, "different number of viscous layers on adjacent faces"),
    ],
)
def test_layer_face_sets_that_do_not_fit_together_raise_smesh_reason(
    set_b: str, count_b: int, reason: str
) -> None:
    """Two face sets that SMESH refuses: they share the face y = 0, or they are on the
    adjacent faces x = 0 and y = 0 with 3 and 2 layers (``StdMeshers_ViscousLayers.cxx``,
    ``findFacesWithLayers``). The reference meshed no volume and said nothing. Now the
    compute raises with SMESH's reason, naming the face by its ordinal, and meshes nothing.
    """
    box = _unit_box()
    x0, y0 = _at_x0(box.faces()), _at_y0(box.faces())
    walls_a = (x0, y0) if set_b == "y0" else (x0,)

    with Mesher(box) as mesher:
        _assign_3d(mesher, "PolyhedronPerSolid3D")
        mesher.assign(_layer_set(0.3, 3, 1.2, walls_a, "bl_a"))
        mesher.assign(_layer_set(0.2, count_b, 1.0, (y0,), "bl_b"))

        with pytest.raises(PysmeshError, match=reason) as raised:
            mesher.compute()

        assert mesher.mesh().element_count == 0
    assert "SOLID 1" in str(raised.value)
    if set_b == "y0":
        assert f"FACE {y0}" in str(raised.value)


@pytest.mark.parametrize("algorithm", ["Hexa3D", "Cartesian3D"])
def test_a_second_layer_hypothesis_on_an_algorithm_that_reads_one_is_refused(
    algorithm: str,
) -> None:
    """Hexa_3D takes one ViscousLayers per solid (``StdMeshers_Hexa_3D.cxx:136-147``):
    with two, the reference meshed no volume and said nothing. Cartesian_3D keeps the last
    one it lists (``StdMeshers_Cartesian_3D.cxx:114-125``): the reference built only one
    stack. Now the compute refuses, names the solid and the algorithm, and meshes nothing.
    """
    box = _unit_box()
    wall_a, wall_b = _at_x0(box.faces()), _at_x1(box.faces())

    with Mesher(box) as mesher:
        _assign_3d(mesher, algorithm)
        mesher.assign(_layer_set(0.3, 3, 1.2, (wall_a,), "bl_a"))
        mesher.assign(_layer_set(0.2, 3, 1.0, (wall_b,), "bl_b"))

        with pytest.raises(PysmeshError, match="reads one ViscousLayers") as raised:
            mesher.compute()

        assert mesher.mesh().element_count == 0
    native = {"Hexa3D": "Hexa_3D", "Cartesian3D": "Cartesian_3D"}[algorithm]
    assert native in str(raised.value)
    assert "SOLID 1" in str(raised.value)


def test_unassign_removes_exactly_the_layer_hypothesis_it_is_given() -> None:
    """Two ViscousLayers on one solid, the second one detached: the first one's stack is
    built, and only its group exists. The reference detached the first one by name.
    """
    box = _unit_box()
    second = _layer_set(0.2, 2, 1.0, (_at_x1(box.faces()),), "bl_b")

    with Mesher(box) as mesher:
        _assign_3d(mesher, "PolyhedronPerSolid3D")
        mesher.assign(_layer_set(0.3, 3, 1.2, (_at_x0(box.faces()),), "bl_a"))
        mesher.assign(second)
        mesher.unassign(second)
        mesher.compute()
        x = np.unique(np.round(mesher.mesh().node_coords[:, 0], 12))
        groups = {g.name: int(g.element_ids.size) for g in mesher.groups()}

    np.testing.assert_allclose(
        x[(x > TOL) & (x <= 0.3 + TOL)], _layer_ends(0.3, 1.2, 3), atol=TOL
    )
    assert groups == {"bl_a": 16 * 3}


def test_a_solid_meshes_again_once_the_second_hexa_layer_hypothesis_is_detached() -> None:
    """Hexa3D with two ViscousLayers, then the second one detached: the solid meshes with
    the first one's stack. On the reference the solid stayed unmeshed: removing a
    hypothesis never checked the algorithm again (``SMESH_subMesh.cxx``, state
    ``MISSING_HYP``), so the compute succeeded with no volume.
    """
    total, count = STACK
    box = _unit_box()
    second = _layer_set(0.2, 3, 1.0, (_at_x1(box.faces()),), "bl_b")

    with Mesher(box) as mesher:
        _assign_3d(mesher, "Hexa3D")
        mesher.assign(_layer_set(total, count, 1.2, (_at_x0(box.faces()),), "bl_a"))
        mesher.assign(second)
        mesher.unassign(second)
        report = mesher.compute()
        xyz = mesher.mesh().node_coords
        groups = {g.name: int(g.element_ids.size) for g in mesher.groups()}

    np.testing.assert_allclose(
        _planes(xyz, total), _layer_ends(total, 1.2, count), atol=TOL
    )
    assert groups == {"bl_a": 4 * 4 * count}
    assert report.volumes == 4 * 4 * 4 + 4 * 4 * count


def test_unassign_refuses_a_layer_hypothesis_equal_to_none_of_several() -> None:
    """Two ViscousLayers on one solid, and a third, different one given to unassign: it
    raises, names the sub-shape, and detaches nothing. The reference detached the first.
    """
    box = _unit_box()

    with Mesher(box) as mesher:
        _assign_3d(mesher, "PolyhedronPerSolid3D")
        mesher.assign(_layer_set(0.3, 3, 1.2, (_at_x0(box.faces()),), "bl_a"))
        mesher.assign(_layer_set(0.2, 2, 1.0, (_at_x1(box.faces()),), "bl_b"))
        before = mesher.assignments()

        with pytest.raises(PysmeshError, match="2 'ViscousLayers'"):
            mesher.unassign(_layer_set(0.25, 2, 1.0, (1,), "bl_c"))

        assert mesher.assignments() == before


# ---- VL7 the extrusion methods on the catalogue path -------------------------------- #


def _method_stack(
    algorithm: str, method: ExtrusionMethod, two_walls: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, int], ps.ComputeReport]:
    """The unit box with the stack on x = 0 (and y = 0) by ``method``: every node, the
    nodes of the layer cells, the group sizes, the report."""
    total, count = STACK
    box = _unit_box()
    walls = (_at_x0(box.faces()),)
    if two_walls:
        walls += (_at_y0(box.faces()),)
    with Mesher(box) as mesher:
        _assign_3d(mesher, algorithm)
        mesher.assign(
            ViscousLayers(
                total_thickness=total,
                layer_count=count,
                stretch_factor=1.2,
                boundary=walls,
                group_name="bl",
                method=method,
            )
        )
        report = mesher.compute()
        mesh = mesher.mesh()
        layer_ids = {int(i) for g in mesher.groups() for i in g.element_ids}
        groups = {g.name: int(g.element_ids.size) for g in mesher.groups()}
    rows = [r for r in range(mesh.element_count) if int(mesh.element_id[r]) in layer_ids]
    layer_nodes = np.unique(np.concatenate([mesh.nodes_of(r) for r in rows]))
    return mesh.node_coords, mesh.node_coords[layer_nodes], groups, report


@pytest.mark.parametrize("algorithm", ["Hexa3D", "PolyhedronPerSolid3D"])
@pytest.mark.parametrize("method", list(ExtrusionMethod))
def test_each_extrusion_method_grows_the_closed_form_stack_on_a_flat_wall(
    method: ExtrusionMethod, algorithm: str
) -> None:
    """ViscousLayers on the wall x = 0 of the unit box, by each extrusion method, through
    Mesher. On a flat wall each method moves a node along the wall normal by the
    closed-form depth (SMESH ``additional_hypo.rst``, "Viscous Layers"), so the stack
    nodes lie on the planes x = d_k, and the wall gets 4 x 4 x N layer cells.
    """
    total, count = STACK

    xyz, _, groups, report = _method_stack(algorithm, method, two_walls=False)

    np.testing.assert_allclose(
        _planes(xyz, total), _layer_ends(total, 1.2, count), atol=TOL
    )
    assert groups == {"bl": 4 * 4 * count}
    assert report.warnings == ()


@pytest.mark.parametrize("algorithm", ["Hexa3D", "PolyhedronPerSolid3D"])
def test_smoothed_layers_on_two_adjacent_walls_keep_the_closed_form_on_each(
    algorithm: str,
) -> None:
    """SURF_OFFSET_SMOOTH on the walls x = 0 and y = 0: the stacks meet along the edge
    between them, and away from it (y > T) the x = 0 stack lies on the closed-form planes.
    """
    total, count = STACK

    xyz, _, groups, report = _method_stack(
        algorithm, ExtrusionMethod.SURF_OFFSET_SMOOTH, two_walls=True
    )

    away = xyz[xyz[:, 1] > total + TOL]
    np.testing.assert_allclose(
        _planes(away, total), _layer_ends(total, 1.2, count), atol=TOL
    )
    assert groups == {"bl": 2 * 4 * 4 * count}
    assert report.warnings == ()


@pytest.mark.parametrize("algorithm", ["Hexa3D", "PolyhedronPerSolid3D"])
@pytest.mark.parametrize(
    "method", [ExtrusionMethod.FACE_OFFSET, ExtrusionMethod.NODE_OFFSET]
)
def test_unsmoothed_layers_on_two_adjacent_walls_stop_short_with_a_warning(
    method: ExtrusionMethod, algorithm: str
) -> None:
    """FACE_OFFSET and NODE_OFFSET do not smooth the layers (``StdMeshers_ViscousLayers.cxx``
    ``AverageHyp::ToSmooth``), so on the walls x = 0 and y = 0 the two stacks collide along
    the edge between them and SMESH stops the inflation short of T. That is upstream's
    local limiting (``:5005-5015``): the compute succeeds with a warning on the solid that
    states the average thickness it reached, below T, and no layer node lies deeper than T
    from the nearer wall.
    """
    total, count = STACK

    _, layer_xyz, groups, report = _method_stack(algorithm, method, two_walls=True)

    (warning,) = report.warnings
    assert (warning.kind, warning.ordinal) == (ps.SubShapeKind.SOLID, 1)
    head = f"Thickness {total:g} of viscous layers not reached, "
    assert warning.text.startswith(head + "average reached thickness is ")
    reached = float(warning.text.rsplit(" ", 1)[1])
    assert 0.0 < reached < total
    depth = np.minimum(layer_xyz[:, 0], layer_xyz[:, 1])
    assert float(depth.max()) <= total + TOL
    assert groups == {"bl": 2 * 4 * 4 * count}
