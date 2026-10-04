# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-03

"""Gates for the TABLE and EXPRESSION distributions of :class:`NumberOfSegments`.

Both distributions place node ``k`` of ``N`` where the integral of the density,
taken from the start of the edge, reaches ``k / N`` of its total. Each test computes
that position in closed form and asserts the mesh against it. The oracle is SMESH's own
input, not the target the density was built from (report ``defect_sweep_4.2.2.md``
§8 S1, S2 and §13.2).

* **TABLE.** SMESH interpolates the ``(t, density)`` table linearly, so the integral is
  piecewise quadratic and each node is the root of a quadratic on its interval. The
  table samples ``1 / h`` at every node of a geometric target on a 15 m edge, with a
  wall cell ``h0`` from 3 um to 3 mm.
* **EXPRESSION.** The density ``1 / (a + t)`` integrates to ``ln((a + t) / a)``, so node
  ``k`` sits at ``t_k = a (((1 + a) / a)^(k / N) - 1)``.

Every node must lie within ``1e-10 L`` of its closed form. Before the fix the TABLE form
failed with "no message" for a 3 um or 30 um wall cell and put nodes metres away for
300 um, and the EXPRESSION form put nodes up to 11 m away.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    Arithmetic1D,
    Distribution,
    FixedPoints1D,
    Geometric1D,
    Hypothesis,
    LayerDistribution,
    Mesher,
    NumberOfSegments,
    PysmeshError,
    Regular1D,
    Session,
    StartEndLength,
)

EDGE_LENGTH: float = 15.0
NODE_TOLERANCE: float = 1e-10 * EDGE_LENGTH
WALL_CELL_RTOL: float = 1e-6

TABLE_WALL_CELLS: tuple[float, ...] = (3e-6, 3e-5, 3e-4, 3e-3)
TABLE_COUNTS: tuple[int, ...] = (100, 200)


def _line_nodes(hyp: ps.Hypothesis) -> NDArray[np.float64]:
    """The sorted x of every node of a 15 m line meshed by ``Regular1D`` and ``hyp``."""
    s = Session()
    s.add_line((0.0, 0.0, 0.0), (EDGE_LENGTH, 0.0, 0.0))
    with Mesher(ps.load_brep(s.brep())) as m:
        m.assign(Regular1D())
        m.assign(hyp)
        m.compute()
        return np.sort(np.asarray(m.mesh().node_coords, dtype=np.float64)[:, 0])


def _geometric_target(h0: float, n: int) -> NDArray[np.float64]:
    """Normalised nodes of ``n`` cells in geometric progression, first cell ``h0``."""
    lo, hi = 1.0 + 1e-12, 3.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if h0 * (mid**n - 1.0) / (mid - 1.0) < EDGE_LENGTH:
            lo = mid
        else:
            hi = mid
    ratio = 0.5 * (lo + hi)
    cells = h0 * ratio ** np.arange(n, dtype=np.float64)
    x = np.concatenate([[0.0], np.cumsum(cells)]) / EDGE_LENGTH
    x[-1] = 1.0
    return x


def _density_table(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """The density ``1 / h`` at every node of ``x``: end cells, else the mean cell."""
    h = np.diff(x)
    return 1.0 / np.concatenate([[h[0]], 0.5 * (h[1:] + h[:-1]), [h[-1]]])


def _table_closed_form(
    x: NDArray[np.float64], d: NDArray[np.float64], n: int
) -> NDArray[np.float64]:
    """Normalised node positions of the linearly interpolated density table ``(x, d)``.

    On ``[x_i, x_i+1]`` the integral is ``F_i + d_i s + a s^2 / 2`` with ``s = t - x_i``
    and the slope ``a``. Node ``k`` solves it for ``k F(1) / n`` with the root
    ``s = 2 r / (d_i + sqrt(d_i^2 + 2 a r))``, which has no cancellation.
    """
    cells = np.diff(x)
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * cells)])
    target = integral[-1] * np.arange(1, n, dtype=np.float64) / n
    i = np.clip(np.searchsorted(integral, target, side="right") - 1, 0, cells.size - 1)
    slope = (d[i + 1] - d[i]) / cells[i]
    rest = target - integral[i]
    s = 2.0 * rest / (d[i] + np.sqrt(d[i] ** 2 + 2.0 * slope * rest))
    return np.concatenate([[0.0], x[i] + s, [1.0]])


def _table_case(
    h0: float, n: int, mirrored: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The meshed nodes (m) and their closed form (m) for one density table."""
    x = _geometric_target(h0, n)
    d = _density_table(x)
    if mirrored:
        x, d = 1.0 - x[::-1], d[::-1].copy()
    hyp = NumberOfSegments(
        count=n, distribution=Distribution.TABLE, table=tuple(np.c_[x, d].ravel())
    )
    return _line_nodes(hyp), EDGE_LENGTH * _table_closed_form(x, d, n)


@pytest.mark.parametrize("mirrored", [False, True], ids=["forward", "mirrored"])
@pytest.mark.parametrize("n", TABLE_COUNTS)
@pytest.mark.parametrize("h0", TABLE_WALL_CELLS)
def test_a_table_distribution_puts_every_node_at_its_closed_form(
    h0: float, n: int, mirrored: bool
) -> None:
    """Each node lies within 1e-10 L of the root of the table's piecewise quadratic."""
    nodes, exact = _table_case(h0, n, mirrored)

    error = np.abs(nodes - exact)

    assert nodes.size == n + 1
    assert error.max() <= NODE_TOLERANCE, (h0, n, mirrored, float(error.max()))


@pytest.mark.parametrize("n", TABLE_COUNTS)
@pytest.mark.parametrize("h0", TABLE_WALL_CELLS)
def test_a_mirrored_table_reflected_back_lands_on_the_forward_nodes(
    h0: float, n: int
) -> None:
    """The mirrored table, reflected back, gives the forward nodes within 1e-10 L."""
    forward, _ = _table_case(h0, n, mirrored=False)
    mirrored, _ = _table_case(h0, n, mirrored=True)

    reflected = EDGE_LENGTH - mirrored[::-1]

    assert np.abs(forward - reflected).max() <= NODE_TOLERANCE


@pytest.mark.parametrize("n", TABLE_COUNTS)
@pytest.mark.parametrize("h0", TABLE_WALL_CELLS)
def test_the_wall_cell_of_a_table_distribution_equals_its_closed_form(
    h0: float, n: int
) -> None:
    """The wall cell, forward and mirrored, equals the closed form to 1e-6 relative."""
    forward, exact = _table_case(h0, n, mirrored=False)
    mirrored, _ = _table_case(h0, n, mirrored=True)

    wall_exact = exact[1] - exact[0]
    wall_forward = forward[1] - forward[0]
    wall_mirrored = mirrored[-1] - mirrored[-2]

    assert wall_forward == pytest.approx(wall_exact, rel=WALL_CELL_RTOL, abs=0.0)
    assert wall_mirrored == pytest.approx(wall_exact, rel=WALL_CELL_RTOL, abs=0.0)


EXPRESSION_OFFSETS: tuple[float, ...] = (0.03, 3e-3, 3e-4, 3e-5)
EXPRESSION_COUNT: int = 100


@pytest.mark.parametrize("a", EXPRESSION_OFFSETS)
def test_an_expression_distribution_puts_every_node_at_its_closed_form(
    a: float,
) -> None:
    """Density ``1 / (a + t)``: node k sits at ``a (((1 + a) / a)^(k / N) - 1)``."""
    n = EXPRESSION_COUNT
    hyp = NumberOfSegments(
        count=n, distribution=Distribution.EXPRESSION, expression=f"1/({a!r}+t)"
    )
    k = np.arange(n + 1, dtype=np.float64)
    exact = EDGE_LENGTH * a * (((1.0 + a) / a) ** (k / n) - 1.0)

    nodes = _line_nodes(hyp)
    error = float(np.abs(nodes - exact).max())

    assert nodes.size == n + 1
    assert error <= NODE_TOLERANCE, (a, error)


def test_an_expression_with_no_finite_integral_is_refused_naming_the_edge() -> None:
    """``1 / (t - c)^2`` passes SMESH's 501-point check but has no finite integral.

    The pole at ``c = 0.3001`` lies between the sample points ``0.300`` and ``0.302``
    that ``CheckExpressionFunction`` evaluates, so the hypothesis is accepted. Its
    integral over ``[0, 1]`` diverges, so the adaptive rule cannot converge, and the
    compute must fail with the edge named instead of placing nodes from a wrong total.
    """
    hyp = NumberOfSegments(
        count=10, distribution=Distribution.EXPRESSION, expression="1/(t-0.3001)^2"
    )

    with pytest.raises(ps.PysmeshError) as info:
        _line_nodes(hyp)

    assert "EDGE 1" in info.value.details
    assert "did not converge" in info.value.details


# ------------------------------------------------ Reversed edges (report §4 C1) --- #

# A straight chain A-B-C of two unit edges, the second defined from C to B. On each
# edge, a distribution measured from the edge's own start places nodes at the fractions
# below (StdMeshers_Regular_1D.cxx). The parameters are chosen so that each progression
# fits the edge exactly, which leaves compensateError() and distributeError() nothing
# to move:
# * NumberOfSegments SCALE, n = 4, factor 8: alpha = 8^(1/3) = 2, node i at
#   (1 - alpha^i) / (1 - alpha^n) = (2^i - 1) / 15 (:1026-1045);
# * Arithmetic1D 0.1 to 0.3: n = int(2 / 0.4 + 0.5) = 5 segments of 0.1, 0.15, 0.2,
#   0.25 and 0.3;
# * StartEndLength 1/7 to 4/7, and Geometric1D 1/7 with ratio 2: segments 1/7, 2/7, 4/7;
# * FixedPoints1D at 0.25 with 1 and 2 segments: 0.25, then 0.625.
# With the second edge reversed, both edges grade the same way from A towards C.
CHAIN_TOL: float = 1e-12
REVERSED_CASES: dict[str, tuple[float, ...]] = {
    "NumberOfSegments": tuple((2.0**i - 1.0) / 15.0 for i in range(5)),
    "Arithmetic1D": (0.0, 0.1, 0.25, 0.45, 0.7, 1.0),
    "StartEndLength": (0.0, 1.0 / 7.0, 3.0 / 7.0, 1.0),
    "Geometric1D": (0.0, 1.0 / 7.0, 3.0 / 7.0, 1.0),
    "FixedPoints1D": (0.0, 0.25, 0.625, 1.0),
}


def _reversible(name: str, reversed_edges: tuple[int, ...]) -> Hypothesis:
    """The C1 hypothesis of one kind, with the given edges reversed."""
    if name == "NumberOfSegments":
        return NumberOfSegments(
            count=4,
            distribution=Distribution.SCALE,
            scale_factor=8.0,
            reversed_edges=reversed_edges,
        )
    if name == "Arithmetic1D":
        return Arithmetic1D(0.1, 0.3, reversed_edges=reversed_edges)
    if name == "StartEndLength":
        return StartEndLength(1.0 / 7.0, 4.0 / 7.0, reversed_edges=reversed_edges)
    if name == "Geometric1D":
        return Geometric1D(1.0 / 7.0, 2.0, reversed_edges=reversed_edges)
    return FixedPoints1D((0.25,), (1, 2), reversed_edges=reversed_edges)


def _chain() -> tuple[ps.Shape, int]:
    """The chain A-B-C and the ordinal of its second edge, defined from C to B."""
    s = Session()
    s.add_line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    s.add_line((2.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    shape = ps.load_brep(s.brep())
    second = [e.id for e in shape.edges() if e.bbox[3] > 1.5]
    assert len(second) == 1
    return shape, second[0]


@pytest.mark.parametrize("name", sorted(REVERSED_CASES))
def test_a_reversed_edge_grades_both_edges_of_a_chain_from_the_same_end(
    name: str,
) -> None:
    """Edge 1 from A and the reversed edge 2 from B follow one closed form (C1)."""
    shape, second = _chain()

    with Mesher(shape) as m:
        m.assign(Regular1D())
        m.assign(_reversible(name, (second,)))
        m.compute()
        x = np.unique(m.mesh().node_coords[:, 0])

    expected = REVERSED_CASES[name]
    assert x[x <= 1.0 + CHAIN_TOL] == pytest.approx(expected, abs=CHAIN_TOL)
    assert x[x >= 1.0 - CHAIN_TOL] - 1.0 == pytest.approx(expected, abs=CHAIN_TOL)


def test_reversed_edges_naming_no_edge_are_refused() -> None:
    """An ordinal beyond the edges of the shape raises PysmeshError (C1)."""
    shape, _ = _chain()

    with Mesher(shape) as m, pytest.raises(PysmeshError, match="99"):
        m.assign(_reversible("Arithmetic1D", (99,)))


def test_reversed_edges_inside_a_layer_distribution_are_refused() -> None:
    """A layer distribution spaces layers, not edge nodes: refused by name (C1)."""
    shape, _ = _chain()
    inner = NumberOfSegments(count=3, reversed_edges=(1,))

    with Mesher(shape) as m, pytest.raises(PysmeshError, match="reversed_edges"):
        m.assign(LayerDistribution(distribution=inner))
