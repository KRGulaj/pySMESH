# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-05

"""A NETGEN size given to a face lands on that face, and nowhere else.

NETGENPlugin limits a face's element size through ``OCCGeometry::SetFaceMaxH``, for a
local size on a face and for a chordal error on a curved face
(``NETGENPlugin_Mesher.cxx``, ``SetLocalSize`` and ``SetLocalSizeForChordalError``).
netgen 6.2.2101 has two overloads: ``SetFaceMaxH(int facenr, double, const
MeshingParameters&)`` takes a 1-based face number, ``SetFaceMaxH(size_t facenr,
double)`` a 0-based one (``occgeom.hpp:353-380``). The plugin called the 1-based one
with ``faceNgID - 1``: the size went to another face, and face 1 got none
(``NETGENPlugin_face_maxh_index.patch``). Oracle 5 of ``test_mesher_netgen.py``: a face
whose own size is ``h`` has a mean triangle edge of at most 1.5 ``h``.

A chordal error ``c`` sizes a curved face from its curvature. Under OCCT 8 the plugin
read one of the two principal curvatures before OCCT computed it, so a cylinder face got
no size at all (``NETGENPlugin_curvature_before_read.patch``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    ElementType,
    MeshData,
    Mesher,
    Netgen1D2D,
    Netgen1D2D3D,
    NetgenParameters,
    NetgenParameters2D,
    Session,
    SubShape,
    SubShapeKind,
)

_LOCAL = 0.1
_CHORDAL = 0.002
# The truncated cone of the chordal tests: base radius, top radius, height, along +z.
_CONE = (1.0, 0.3, 2.0)


def _triangle_edges(mesh: MeshData, face: int) -> NDArray[np.float64]:
    """The edge lengths of the triangles bound to one face."""
    xyz = mesh.node_coords
    out: list[float] = []
    for e in range(mesh.element_count):
        if (
            int(mesh.element_type[e]) != int(ElementType.TRIANGLE)
            or mesh.element_kind[e] != int(SubShapeKind.FACE)
            or mesh.element_ordinal[e] != face
        ):
            continue
        c = mesh.nodes_of(e)
        for i in range(3):
            out.append(float(np.linalg.norm(xyz[c[i]] - xyz[c[(i + 1) % 3]])))
    return np.asarray(out, dtype=np.float64)


def _opposite(box: ps.Shape, face: int) -> int:
    """The face of an axis-aligned box parallel to ``face`` and away from it."""
    boxes = [np.asarray(f.bbox).reshape(2, 3) for f in box.faces()]
    flat = int(np.argmin(np.ptp(boxes[face - 1], axis=0)))
    for other, b in enumerate(boxes, start=1):
        if other != face and np.ptp(b, axis=0)[flat] == 0.0:
            return other
    raise AssertionError(f"face {face} has no parallel face")


@pytest.mark.parametrize("face", [1, 2, 3, 4, 5, 6])
def test_a_local_size_on_a_face_lands_on_that_face_and_not_the_opposite_one(
    face: int,
) -> None:
    """A size of 0.1 on one face of a 4 x 4 x 4 cube meshed at 1.0.

    The face: mean triangle edge at most 0.15. The parallel face, 4 away and sharing no
    edge: mean triangle edge at least 0.3, since the size grows by at most the growth
    rate per unit distance from the refined face (0.1 + 0.3 x 4 = 1.3 there) and is
    capped by max_size 1.0.
    """
    s = Session()
    s.add_box(4.0, 4.0, 4.0)
    cube = ps.load_brep(s.brep())

    with Mesher(cube) as mesher:
        mesher.assign(Netgen1D2D3D())
        where = SubShape(SubShapeKind.FACE, face)
        mesher.assign(NetgenParameters(max_size=1.0, local_sizes=((where, _LOCAL),)))
        mesher.compute()
        mesh = mesher.mesh()
    on_face = _triangle_edges(mesh, face)
    opposite = _triangle_edges(mesh, _opposite(cube, face))

    assert float(on_face.mean()) <= 1.5 * _LOCAL
    assert float(opposite.mean()) >= 3.0 * _LOCAL


def _curved(kind: str) -> tuple[ps.Shape, int]:
    """A cylinder or sphere of radius 1, or the truncated cone, and its curved face."""
    s = Session()
    if kind == "cylinder":
        s.add_cylinder(1.0, 3.0)
    elif kind == "sphere":
        s.add_sphere(1.0)
    else:
        s.add_cone(*_CONE)
    shape = ps.load_brep(s.brep())
    curved = [
        i
        for i, f in enumerate(shape.faces(), start=1)
        if np.ptp(np.asarray(f.bbox).reshape(2, 3)[:, 2]) > 0.0
    ]
    assert len(curved) == 1
    return shape, curved[0]


def _centroid_deviation(kind: str, mesh: MeshData, face: int) -> NDArray[np.float64]:
    """The distance from each triangle centroid of the face to the exact surface."""
    xyz = mesh.node_coords
    rows = [
        e
        for e in range(mesh.element_count)
        if int(mesh.element_type[e]) == int(ElementType.TRIANGLE)
        and mesh.element_kind[e] == int(SubShapeKind.FACE)
        and mesh.element_ordinal[e] == face
    ]
    c = np.asarray([xyz[mesh.nodes_of(e)].mean(axis=0) for e in rows])
    rho = np.hypot(c[:, 0], c[:, 1])
    if kind == "cylinder":
        return np.abs(1.0 - rho)
    if kind == "sphere":
        return np.abs(1.0 - np.linalg.norm(c, axis=1))
    # The cone: distance to its generatrix in the meridian half-plane (rho, z).
    r1, r2, height = _CONE
    return np.abs(
        ((rho - r1) * height - c[:, 2] * (r2 - r1)) / math.hypot(height, r2 - r1)
    )


def _mesh_curved(shape: ps.Shape, chordal: float | None) -> MeshData:
    """The surface mesh of a shape at max_size 0.4, with or without a chordal error."""
    with Mesher(shape) as mesher:
        mesher.assign(Netgen1D2D())
        mesher.assign(NetgenParameters2D(max_size=0.4, chordal_error=chordal))
        mesher.compute()
        return mesher.mesh()


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_a_chordal_error_sizes_a_face_of_constant_curvature(kind: str) -> None:
    """chordal_error 0.002 on a cylinder and a sphere of radius 1, max_size 0.4.

    For a face of radius R the plugin targets h = 0.95 sqrt(3) sqrt(c (2R - c)), 0.104
    here (``elemSizeForChordalError``). An equilateral triangle of side h inscribed in
    the surface has a circumradius of 0.95 sqrt(c (2R - c)), so its centroid lies at
    most about 0.95^2 c = 0.9 c inside. Oracle 5: mean triangle edge at most 1.5 h.
    Deviation: the mean centroid deviation is at most c with the chordal error, and
    above c without.
    """
    shape, face = _curved(kind)
    size = 0.95 * math.sqrt(3.0) * math.sqrt(_CHORDAL * (2.0 - _CHORDAL))

    coarse = _mesh_curved(shape, None)
    fine = _mesh_curved(shape, _CHORDAL)

    assert float(_triangle_edges(fine, face).mean()) <= 1.5 * size
    assert float(_centroid_deviation(kind, fine, face).mean()) <= _CHORDAL
    assert float(_centroid_deviation(kind, coarse, face).mean()) > _CHORDAL


def test_a_chordal_error_sizes_a_face_of_varying_curvature() -> None:
    """chordal_error 0.002 on a truncated cone, radius 1 to 0.3, max_size 0.4.

    The cone is neither a cylinder, a sphere nor a torus: the plugin samples its
    curvature at the nodes of an OCCT triangulation of deflection c and along its edges.
    The mean centroid deviation is at most c with the chordal error, and above c
    without.
    """
    shape, face = _curved("cone")

    coarse = _mesh_curved(shape, None)
    fine = _mesh_curved(shape, _CHORDAL)

    assert float(_centroid_deviation("cone", fine, face).mean()) <= _CHORDAL
    assert float(_centroid_deviation("cone", coarse, face).mean()) > _CHORDAL
