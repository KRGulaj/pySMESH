# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for Session.pipe_t_shape: GEOM's MakePipeTShape, the wall only.

The main pipe runs along x on ``[-L1, L1]`` with inner radius ``R1`` and outer radius
``A1 = R1 + W1``; the branch runs along z on ``[0, L2]`` with ``R2`` and ``A2 = R2 + W2``.
The closed forms, each computed by an independent 1-D integral (Gauss-Legendre on the
substitution ``y = r sin(t)``, smooth on ``[-pi/2, pi/2]``):

* the overlap of a main cylinder of radius ``a`` and a branch cylinder of radius
  ``b <= a`` (the branch above the axis) is ``V(a, b) = int_{-b}^{b} 2 sqrt(b^2 - y^2)
  sqrt(a^2 - y^2) dy``; the wall is the outer union minus the inner union:
  ``(2 L1 pi A1^2 + L2 pi A2^2 - V(A1, A2)) - (2 L1 pi R1^2 + L2 pi R2^2 - V(R1, R2))``;
* the faces on the main cylinder of radius ``a`` cover ``2 pi a 2 L1`` less the hole the
  branch of radius ``b`` cuts, ``H(a, b) = int_{-b}^{b} 2 sqrt(b^2 - y^2) a /
  sqrt(a^2 - y^2) dy``; the faces on the branch cylinder cover
  ``int_0^{2 pi} (L2 - sqrt(a^2 - b^2 sin^2 t)) b dt``;
* each pipe end is an annulus: ``pi (A^2 - R^2)``, centred on the pipe's axis at the end.

Tolerances. The wall's faces are bounded by OCCT's intersection curves, approximated within
the boolean's tolerance, so measures are compared at 1e-7 relative (measured: 5.6e-11 on
the unpartitioned volume). The partition cuts each quarter with GEOM's filling face
(``GEOMImpl_Block6Explorer::MakeFace`` on a non-planar wire: ``BRepOffsetAPI_MakeFilling``,
3-D tolerance 1e-4). Its edges carry tolerances up to 5.3e-4, and they are the only edges
above 1e-6. The partitioned volume is 1.18e-6 relative below the closed form, the same in
each quarter, so partitioned measures are compared at 1e-5 relative.

GEOM's own counts of the edge chains (AdvancedEngine_IOperations::MakeGroups): THICKNESS
21 edges (17 when the outer radii are equal), CIRCULAR_QUARTER_PIPE 6, the others 8.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, Session

DIMS: tuple[float, ...] = (2.0, 0.5, 6.0, 1.0, 0.4, 5.0)
EQUAL_OUTER: tuple[float, ...] = (2.0, 0.5, 6.0, 2.0, 0.5, 5.0)
WALL_RTOL: float = 1e-7
PARTITION_RTOL: float = 1e-5

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(200)

Integrand: TypeAlias = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def integral(f: Integrand, lo: float, hi: float) -> float:
    """The integral of f over [lo, hi] by 200-point Gauss-Legendre."""
    t = 0.5 * (hi - lo) * _NODES + 0.5 * (hi + lo)
    return float(0.5 * (hi - lo) * np.sum(_WEIGHTS * f(t)))


def over_disk(
    a: float, b: float, g: Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
) -> float:
    """The integral over the disk x^2 + y^2 <= b^2 of g(a, y), by y = b sin(t)."""

    def f(t: NDArray[np.float64]) -> NDArray[np.float64]:
        y = b * np.sin(t)
        chord = 2 * b * np.cos(t)
        return chord * g(a, y) * b * np.cos(t)

    return integral(f, -math.pi / 2, math.pi / 2)


def overlap(a: float, b: float) -> float:
    """V(a, b): the main cylinder of radius a meets the upward branch of radius b."""
    return over_disk(a, b, lambda a_, y: np.sqrt(a_**2 - y**2))


def wall_volume(
    r1: float, w1: float, l1: float, r2: float, w2: float, l2: float
) -> float:
    """The closed-form volume of the wall."""
    a1, a2 = r1 + w1, r2 + w2
    outer = 2 * l1 * math.pi * a1**2 + l2 * math.pi * a2**2 - overlap(a1, a2)
    inner = 2 * l1 * math.pi * r1**2 + l2 * math.pi * r2**2 - overlap(r1, r2)
    return outer - inner


def cylinder_faces_area(a: float, l1: float, b: float, l2: float) -> float:
    """The area on the main cylinder of radius a and the branch cylinder of radius b."""
    hole = over_disk(a, b, lambda a_, y: a_ / np.sqrt(a_**2 - y**2))

    def height(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return (l2 - np.sqrt(a**2 - (b * np.sin(t)) ** 2)) * b

    return 2 * math.pi * a * 2 * l1 - hole + integral(height, 0.0, 2 * math.pi)


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, by kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def state(s: Session) -> tuple[bytes, tuple[tuple[int, ...], ...], int, int]:
    """The bytes, the live ids and the two counters."""
    return s.brep(), id_space(s), s.op_count, s.issued_id_count


def total_volume(s: Session) -> float:
    """The sum of every solid's volume."""
    solids = [EntityId(int(i)) for i in s.entities(EntityKind.SOLID)]
    return float(s.mass_properties(solids, precision=1e-10).measure.sum())


def group(t: ps.PipeTShape, name: str) -> list[ps.PipeTShapeGroup]:
    """The groups of a name."""
    return [g for g in t.groups if g.name == name]


def area_and_centroid(
    s: Session, ids: tuple[EntityId, ...]
) -> tuple[float, np.ndarray]:
    """The total area of faces and their area centroid."""
    m = s.mass_properties(list(ids), precision=1e-10)
    area = float(m.measure.sum())
    return area, (m.measure[:, None] * m.centroid).sum(axis=0) / area


def test_the_unpartitioned_wall_has_the_closed_form_volume() -> None:
    s = Session()

    t = s.pipe_t_shape(*DIMS, partition=False)

    assert len(s.entities(EntityKind.SOLID)) == 1
    np.testing.assert_allclose(total_volume(s), wall_volume(*DIMS), rtol=WALL_RTOL)
    assert t.delta.valid is True
    assert all(
        s.origin(EntityId(int(i))).role == NameRole.CONSTRUCTED for i in t.delta.created
    )
    assert [g.name for g in t.groups] == ["INTERNAL_FACES", "EXTERNAL_FACES"]


@pytest.mark.parametrize("dims", [DIMS, EQUAL_OUTER])
def test_the_partitioned_wall_is_a_compound_of_blocks_of_the_same_volume(
    dims: tuple[float, ...],
) -> None:
    s = Session()

    s.pipe_t_shape(*dims)

    assert s.check_blocks().is_compound_of_blocks
    np.testing.assert_allclose(total_volume(s), wall_volume(*dims), rtol=PARTITION_RTOL)
    assert len(s.entities(EntityKind.SOLID)) == (16 if dims == DIMS else 12)


def test_the_pipe_ends_are_the_three_junction_groups() -> None:
    r1, w1, l1, r2, w2, l2 = DIMS
    s = Session()

    t = s.pipe_t_shape(*DIMS)

    ends = [group(t, f"JUNCTION_FACE_{k}") for k in (1, 2, 3)]
    assert all(len(g) == 1 and len(g[0].ids) == 4 for g in ends)
    expected = [
        (math.pi * ((r1 + w1) ** 2 - r1**2), (-l1, 0.0, 0.0)),
        (math.pi * ((r2 + w2) ** 2 - r2**2), (0.0, 0.0, l2)),
        (math.pi * ((r1 + w1) ** 2 - r1**2), (l1, 0.0, 0.0)),
    ]
    for (g,), (area, centre) in zip(ends, expected, strict=True):
        measured, centroid = area_and_centroid(s, g.ids)
        np.testing.assert_allclose(measured, area, rtol=WALL_RTOL)
        np.testing.assert_allclose(centroid, centre, atol=1e-9)


@pytest.mark.parametrize("partition", [False, True])
def test_the_internal_and_external_faces_cover_the_cylinders(partition: bool) -> None:
    r1, w1, l1, r2, w2, l2 = DIMS
    s = Session()

    t = s.pipe_t_shape(*DIMS, partition=partition)

    rtol = PARTITION_RTOL if partition else WALL_RTOL
    for name, a, b in (
        ("INTERNAL_FACES", r1, r2),
        ("EXTERNAL_FACES", r1 + w1, r2 + w2),
    ):
        (g,) = group(t, name)
        assert g.kind == EntityKind.FACE
        area, _ = area_and_centroid(s, g.ids)
        np.testing.assert_allclose(area, cylinder_faces_area(a, l1, b, l2), rtol=rtol)
        types = s.entity_types(EntityKind.FACE)
        kinds = {
            t_ for i, t_ in zip(types.ids, types.types, strict=True) if int(i) in g.ids
        }
        assert kinds == {"Cylinder"}


@pytest.mark.parametrize(
    ("dims", "counts"),
    [
        (
            DIMS,
            {
                "THICKNESS": [21],
                "CIRCULAR_QUARTER_PIPE": [6, 8],
                "HALF_LENGTH_MAIN_PIPE": [8],
                "HALF_LENGTH_INCIDENT_PIPE": [8],
                "FLANGE": [8],
            },
        ),
        (
            EQUAL_OUTER,
            {
                "THICKNESS": [17],
                "CIRCULAR_QUARTER_PIPE": [6, 6],
                "HALF_LENGTH_MAIN_PIPE": [8],
                "HALF_LENGTH_INCIDENT_PIPE": [8],
            },
        ),
    ],
)
def test_the_edge_chains_have_geoms_counts_and_are_chains(
    dims: tuple[float, ...], counts: dict[str, list[int]]
) -> None:
    s = Session()

    t = s.pipe_t_shape(*dims)

    edge_groups = [g for g in t.groups if g.kind == EntityKind.EDGE]
    found: dict[str, list[int]] = {}
    for g in edge_groups:
        found.setdefault(g.name, []).append(len(g.ids))
        assert sorted(int(i) for i in s.propagation_chain(g.ids[0])) == list(
            map(int, g.ids)
        )
    assert {k: sorted(v) for k, v in found.items()} == counts
    (thickness,) = group(t, "THICKNESS")
    lengths = s.mass_properties(list(thickness.ids)).measure
    assert np.any(np.isclose(lengths, dims[1], rtol=WALL_RTOL))
    assert np.any(np.isclose(lengths, dims[4], rtol=WALL_RTOL))


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def built() -> tuple[bytes, tuple[tuple[int, ...], ...], tuple[object, ...]]:
        s = Session()
        t = s.pipe_t_shape(*DIMS)
        return s.brep(), id_space(s), t.groups

    first = built()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(built())

    assert all(other == first for other in others)


@pytest.mark.parametrize(
    ("dims", "match"),
    [
        ((0.0, 0.5, 6.0, 1.0, 0.4, 5.0), "main_radius 0 is not above 1e-07"),
        ((2.0, 0.5, 6.0, 1.0, -0.4, 5.0), "branch_thickness -0.4 is not above 1e-07"),
        ((1.0, 0.5, 6.0, 2.0, 0.4, 5.0), "R2 <= R1"),
        ((2.0, 0.5, 6.0, 1.5, 1.5, 5.0), "R2\\+W2 <= R1\\+W1"),
        ((2.0, 0.5, 6.0, 1.5, 1.0, 5.0), "R2 = R1 when R2\\+W2 = R1\\+W1"),
        ((2.0, 0.5, 6.0, 1.0, 0.4, 2.4), "R1\\+W1 < L2"),
        ((2.0, 0.5, 1.3, 1.0, 0.4, 5.0), "R2\\+W2 < L1"),
        ((2.0, math.nan, 6.0, 1.0, 0.4, 5.0), "main_thickness must be a finite number"),
    ],
)
def test_pipe_t_shape_refuses_geoms_impossible_dimensions(
    dims: tuple[float, ...], match: str
) -> None:
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=match) as err:
        s.pipe_t_shape(*dims)

    assert "Session.pipe_t_shape" in str(err.value)
    assert state(s) == before
