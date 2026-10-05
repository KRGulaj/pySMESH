# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-05

"""A NETGEN size given to a face lands on that face, and nowhere else.

NETGENPlugin limits a face's element size through ``OCCGeometry::SetFaceMaxH``, for a
local size on a face and for a chordal error on a curved face (``NETGENPlugin_Mesher.cxx``,
``SetLocalSize`` and ``SetLocalSizeForChordalError``). netgen 6.2.2101 has two overloads:
``SetFaceMaxH(int facenr, double, const MeshingParameters&)`` takes a 1-based face number,
``SetFaceMaxH(size_t facenr, double)`` a 0-based one (``occgeom.hpp:353-380``). The plugin
called the 1-based one with ``faceNgID - 1``: the size went to another face, and face 1
got none (``NETGENPlugin_face_maxh_index.patch``). Oracle 5 of ``test_mesher_netgen.py``:
a face whose own size is ``h`` has a mean triangle edge of at most 1.5 ``h``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    ElementType,
    MeshData,
    Mesher,
    Netgen1D2D3D,
    NetgenParameters,
    Session,
    SubShape,
    SubShapeKind,
)

_LOCAL = 0.1


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
