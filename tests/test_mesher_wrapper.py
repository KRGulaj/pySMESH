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
    Arithmetic1D,
    Mesher,
    NumberOfSegments,
    PropagOfDistribution,
    Regular1D,
    SegmentAroundVertex0D,
    SegmentLengthAroundVertex,
    Session,
    SubShape,
    SubShapeKind,
)

LINE_LENGTH: float = 10.0
TOL: float = 1e-9

# The trapezoid of the propagation test: a 4-long bottom, a 2-long top, height 2.
TRAPEZOID: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0),
    (3.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
)


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


def _trapezoid_shape() -> ps.Shape:
    """A planar four-sided face whose opposite sides have different lengths."""
    session = Session()
    session.add_polyline(np.array(TRAPEZOID, dtype=np.float64), closed=True)
    session.make_face(list(session.entities(ps.EntityKind.EDGE)))
    return ps.load_brep(session.brep())


def _edge_at_height(shape: ps.Shape, y: float) -> int:
    """The ordinal of the horizontal edge of ``shape`` at height ``y``."""
    for edge in shape.edges():
        box = edge.bbox
        if abs(box[1] - y) < TOL and abs(box[4] - y) < TOL:
            return edge.id
    raise AssertionError(f"no horizontal edge at y = {y}")


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


# ---- W1.3 PropagOfDistribution ---------------------------------------------------- #


def test_propag_of_distribution_repeats_the_fractions_on_a_shorter_edge() -> None:
    """The 2-long top edge gets the node fractions of the 4-long bottom edge.

    Spec (SMESH ``additional_hypo.rst``, "Propagation of Node Distribution on Opposite
    Edges"): the opposite edge gets "the same number of nodes and the same relations
    between segment lengths". Arithmetic1D(0.5, 1.5) on the 4-long bottom gives
    ``n = 2 L / (a + b) = 4`` segments of 0.5, 5/6, 7/6 and 1.5, so the nodes sit at the
    fractions 0, 1/8, 1/3, 5/8 and 1. The top edge carries them in either direction.
    """
    shape = _trapezoid_shape()
    bottom = SubShape(SubShapeKind.EDGE, _edge_at_height(shape, 0.0))
    fractions = np.array([0.0, 1.0 / 8.0, 1.0 / 3.0, 5.0 / 8.0, 1.0])

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Arithmetic1D(start_length=0.5, end_length=1.5), on=bottom)
        mesher.assign(PropagOfDistribution(), on=bottom)
        mesher.compute()
        xyz = mesher.mesh().node_coords

    on_bottom = np.sort(xyz[np.abs(xyz[:, 1]) < TOL, 0]) / 4.0
    on_top = np.sort(xyz[np.abs(xyz[:, 1] - 2.0) < TOL, 0] - 1.0) / 2.0
    np.testing.assert_allclose(on_bottom, fractions, atol=TOL)
    forward = np.allclose(on_top, fractions, atol=TOL)
    backward = np.allclose(on_top, np.sort(1.0 - fractions), atol=TOL)
    assert forward or backward, on_top
