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
