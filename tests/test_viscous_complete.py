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
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    ExtrusionMethod,
    Hexa3D,
    Mesh,
    Mesher,
    NumberOfSegments,
    PysmeshError,
    Quadrangle2D,
    Regular1D,
    Session,
    ViscousLayers,
    ViscousLayers2D,
    VLParams,
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
