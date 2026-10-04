# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-04

"""Gates for Prism3D on a prism whose one side face spans a split bottom edge.

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
  must say so for that face, not report the error of a rejected candidate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
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


def _split_edge_prism(n: int) -> ps.Shape:
    """The B1 solid: an n-gon prism, one bottom edge split, the top edge above whole."""
    t = np.linspace(0.0, 2.0 * np.pi, n + 1)[:-1]
    bottom = np.c_[np.cos(t), np.sin(t), np.zeros(n)]
    top = bottom + (0.0, 0.0, HEIGHT)
    mid = 0.5 * (bottom[0] + bottom[1])
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


def _assign(m: Mesher, shape: ps.Shape, n: int, base: str) -> None:
    """Regular1D, 4 segments and 2 per half-edge, the base mesher, then Prism3D."""
    half = math.sin(math.pi / n)
    halves = [e.id for e in shape.edges() if abs(e.length - half) < NODE_TOL]
    assert len(halves) == 2
    m.assign(Regular1D())
    m.assign(NumberOfSegments(count=LAYERS))
    for e in halves:
        m.assign(
            NumberOfSegments(count=HALF_SEGMENTS), on=SubShape(SubShapeKind.EDGE, e)
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
        "a composite horizontal side is not supported"
    ) in details
