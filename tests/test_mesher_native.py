# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-03

"""Gates for the native catalogue entries added with SMESH 9.16.

These entries are reachable only through the native factory,
``Mesher._m.assign(name, params, kind, ordinal)``. A later phase writes their typed
Python wrappers. Each test drives one entry through that path and checks a property
that the upstream specification states, on a shape small enough to reason about by
hand:

* ``SegmentAroundVertex_0D`` gives the segment at its vertex the length that
  ``SegmentLengthAroundVertex`` names (SMESH ``segments_around_vertex_algo.rst``).
* ``PropagOfDistribution`` carries the *relative* node distribution to the opposite
  edge of a quadrangle face (SMESH ``additional_hypo.rst``).
* ``NumberOfSegments`` with the beta law (new in 9.16) puts the nodes at the closed
  form of the law, and refuses a ``beta`` that the law is undefined for (SMESH
  ``1d_meshing_hypo.rst``).

The upstream sources cited are those of SMESH ``V9_16_0``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    Arithmetic1D,
    Mesher,
    NumberOfSegments,
    Propagation,
    PysmeshError,
    Regular1D,
    SegmentLengthAroundVertex,
    Session,
    SubShape,
    SubShapeKind,
)

LINE_LENGTH: float = 10.0

# The trapezoid of the propagation tests: a 4-long bottom, a 2-long top, height 2.
TRAPEZOID: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0),
    (3.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
)

# NumberOfSegments::DistrType::DT_BetaLaw (StdMeshers_NumberOfSegments.hxx). The
# public Distribution enum has no member for it yet.
DT_BETA_LAW: int = 4


# ---- Fixtures --------------------------------------------------------------------- #


def _line_shape(length: float = LINE_LENGTH) -> ps.Shape:
    """One straight edge from the origin along +x."""
    session = Session()
    session.add_line((0.0, 0.0, 0.0), (length, 0.0, 0.0))
    return ps.load_brep(session.brep())


def _trapezoid_shape() -> ps.Shape:
    """A planar four-sided face whose opposite sides have different lengths."""
    session = Session()
    session.add_polyline(np.array(TRAPEZOID, dtype=np.float64), closed=True)
    session.make_face(list(session.entities(ps.EntityKind.EDGE)))
    return ps.load_brep(session.brep())


def _vertex_at(shape: ps.Shape, point: tuple[float, float, float]) -> int:
    """The ordinal of the vertex of ``shape`` at ``point``."""
    for vertex in shape.vertices():
        if np.allclose(vertex.xyz, point, atol=1e-9):
            return vertex.id
    raise AssertionError(f"no vertex at {point}")


def _edge_at_height(shape: ps.Shape, y: float) -> int:
    """The ordinal of the horizontal edge of ``shape`` at height ``y``.

    The bounding box is grown by the edge tolerance (1e-7), hence the 1e-6 test.
    """
    for edge in shape.edges():
        box = edge.bbox
        if abs(box[1] - y) < 1e-6 and abs(box[4] - y) < 1e-6:
            return edge.id
    raise AssertionError(f"no horizontal edge at y = {y}")


def _sorted_x(mesh: ps.MeshData) -> NDArray[np.float64]:
    """The x coordinate of every node, ascending."""
    return np.sort(np.ascontiguousarray(mesh.node_coords[:, 0], dtype=np.float64))


def _beta_law(beta: float, count: int, length: float) -> NDArray[np.float64]:
    """Node positions of the beta law on a straight edge, ends included.

    The law of SMESH 9.16 (``StdMeshers_Regular_1D::computeBetaLaw``, after gmsh):
    for ``i = 1 .. n-1``, ``x_i / L = 1 + b (1 - r**(1 - i/n)) / (1 + r**(1 - i/n))``
    with ``b = |beta|`` and ``r = (b + 1) / (b - 1)``. A negative ``beta`` mirrors
    the positions.
    """
    b = abs(beta)
    r = (b + 1.0) / (b - 1.0)
    eta = np.arange(1, count, dtype=np.float64) / count
    power = r ** (1.0 - eta)
    t = 1.0 + b * (1.0 - power) / (1.0 + power)
    if beta < 0:
        t = np.sort(1.0 - t)
    return np.concatenate(([0.0], t * length, [length]))


def _beta_params(beta: float, count: int) -> dict[str, object]:
    """NumberOfSegments parameters for the beta law, as the native factory reads."""
    params = NumberOfSegments(count=count).params()
    params["distribution"] = DT_BETA_LAW
    params["beta"] = beta
    return params


# ---- SegmentAroundVertex_0D ------------------------------------------------------- #


def test_segment_around_vertex_sets_the_length_at_its_vertex() -> None:
    """The segment at the vertex takes the length; the far end keeps its spacing.

    Spec (SMESH ``segments_around_vertex_algo.rst``): the edge is discretised by its
    1-D hypothesis, then the nodes near the vertex are moved to give the length that
    "Length Near Vertex" names. Five equal segments of a 10-long edge are 2.0 each.
    """
    shape = _line_shape()
    origin = _vertex_at(shape, (0.0, 0.0, 0.0))

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=5))
        mesher._m.assign("SegmentAroundVertex_0D", {}, "VERTEX", origin)
        mesher.assign(
            SegmentLengthAroundVertex(length=0.5),
            on=SubShape(SubShapeKind.VERTEX, origin),
        )
        mesher.compute()
        x = _sorted_x(mesher.mesh())

    assert x[1] - x[0] == pytest.approx(0.5, abs=1e-9)
    assert x[-1] - x[-2] == pytest.approx(2.0, abs=1e-9)
    assert x[0] == pytest.approx(0.0, abs=1e-12)
    assert x[-1] == pytest.approx(LINE_LENGTH, abs=1e-12)


def test_length_around_vertex_is_unused_without_its_0d_algorithm() -> None:
    """The falsification: without the 0-D algorithm the hypothesis is never read."""
    shape = _line_shape()
    origin = _vertex_at(shape, (0.0, 0.0, 0.0))

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=5))
        mesher.assign(
            SegmentLengthAroundVertex(length=0.5),
            on=SubShape(SubShapeKind.VERTEX, origin),
        )
        mesher.compute()
        x = _sorted_x(mesher.mesh())

    np.testing.assert_allclose(np.diff(x), np.full(5, 2.0), atol=1e-9)


# ---- PropagOfDistribution --------------------------------------------------------- #


def _top_and_bottom_fractions(
    propagation: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Node fractions along the 4-long bottom and the 2-long top of the trapezoid.

    The bottom carries Arithmetic1D(0.5 -> 1.5) and the propagation hypothesis named
    ``propagation``; every other edge has two equal segments.
    """
    shape = _trapezoid_shape()
    bottom = SubShape(SubShapeKind.EDGE, _edge_at_height(shape, 0.0))

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Arithmetic1D(start_length=0.5, end_length=1.5), on=bottom)
        if propagation == "Propagation":
            mesher.assign(Propagation(), on=bottom)
        else:
            mesher._m.assign(propagation, {}, "EDGE", bottom.ordinal)
        mesher.compute()
        xyz = mesher.mesh().node_coords

    on_bottom = np.sort(xyz[np.abs(xyz[:, 1]) < 1e-9, 0]) / 4.0
    on_top = np.sort(xyz[np.abs(xyz[:, 1] - 2.0) < 1e-9, 0] - 1.0) / 2.0
    return on_bottom, on_top


def test_propag_of_distribution_repeats_the_relative_spacing() -> None:
    """Same node count, same fractions of length, on an opposite edge half as long.

    Spec (SMESH ``additional_hypo.rst``, "Propagation of Node Distribution on
    Opposite Edges"): the opposite edge gets "the same number of nodes and the same
    relations between segment lengths". The propagation may run either way along the
    top edge, so the fractions are compared both ways.
    """
    bottom, top = _top_and_bottom_fractions("PropagOfDistribution")

    assert bottom.size == top.size
    assert bottom.size > 3
    deviation = min(
        float(np.abs(bottom - top).max()),
        float(np.abs(bottom - np.sort(1.0 - top)).max()),
    )
    assert deviation < 1e-9


def test_plain_propagation_carries_the_absolute_lengths_instead() -> None:
    """The falsification: Propagation copies Arithmetic1D's lengths, not fractions.

    On the 2-long top edge the absolute 0.5 -> 1.5 progression needs fewer segments
    than on the 4-long bottom, so the node counts differ.
    """
    bottom, top = _top_and_bottom_fractions("Propagation")

    assert top.size < bottom.size


# ---- NumberOfSegments: the beta law (new in 9.16) --------------------------------- #


@pytest.mark.parametrize("beta", [1.01, 1.1, 1.5, -1.05])
def test_the_beta_law_places_every_node_at_its_closed_form(beta: float) -> None:
    """Ten segments on a 10-long edge, against the law of SMESH 9.16.

    Reference: ``StdMeshers_Regular_1D::computeBetaLaw`` (SMESH ``V9_16_0``), the law
    of gmsh's ``meshGEdge.cpp``; documented in SMESH ``1d_meshing_hypo.rst``, "Beta
    Law Distribution". A straight edge has an exact arc-length parametrisation, so
    the positions hold to round-off.
    """
    with Mesher(_line_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher._m.assign("NumberOfSegments", _beta_params(beta, 10), "", 0)
        mesher.compute()
        x = _sorted_x(mesher.mesh())

    expected = _beta_law(beta, 10, LINE_LENGTH)
    np.testing.assert_allclose(x, expected, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("beta", [1.01, -1.01])
def test_the_sign_of_beta_chooses_where_the_nodes_crowd(beta: float) -> None:
    """Spec: a negative beta distributes the positions "in the opposite direction".

    With beta just above 1 the law crowds the nodes at the start of the edge; its
    negative crowds them at the end.
    """
    with Mesher(_line_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher._m.assign("NumberOfSegments", _beta_params(beta, 10), "", 0)
        mesher.compute()
        lengths = np.diff(_sorted_x(mesher.mesh()))

    first, last = float(lengths[0]), float(lengths[-1])
    assert (first < last) == (beta > 0)
    assert math.isclose(float(lengths.sum()), LINE_LENGTH, rel_tol=1e-12)


@pytest.mark.parametrize("beta", [1.0, 0.5, 0.0, -1.0])
def test_the_beta_law_refuses_a_beta_inside_the_unit_interval(beta: float) -> None:
    """Degenerate input: upstream documents ``[-1, 1]`` as forbidden for ``log(r)``.

    SMESH ``1d_meshing_hypo.rst``: "Values between [-1, 1] are forbidden to ensure
    validity of the log". ``SetBeta`` does not check it, so the factory must.
    """
    with Mesher(_line_shape()) as mesher:
        with pytest.raises(PysmeshError, match=r"needs \|beta\| > 1"):
            mesher._m.assign("NumberOfSegments", _beta_params(beta, 10), "", 0)
