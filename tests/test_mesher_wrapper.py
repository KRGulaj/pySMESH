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

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    Mesher,
    NumberOfSegments,
    Regular1D,
    SegmentAroundVertex0D,
    SegmentLengthAroundVertex,
    Session,
    SubShape,
    SubShapeKind,
)

LINE_LENGTH: float = 10.0
TOL: float = 1e-9


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
