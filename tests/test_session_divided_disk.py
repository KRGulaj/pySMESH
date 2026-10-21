# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for Session.divided_disk: GEOM's MakeDividedDisk.

The answers are closed forms for a disk of radius ``r`` whose inner block has its corners
at ``a = ratio / 100 * r`` from the centre:

* the faces cover the disk: their areas sum to ``pi r^2``;
* ``"square"``: 5 faces; the inner square (corners on the axes) has area ``2 a^2``;
* ``"hexagon"``: 18 faces; the 6 inner faces meet at the centre and form the regular
  hexagon of circumradius ``a``, area ``3 sqrt(3) / 2 a^2``;
* the disk lies in its plane: every face's centroid has the normal coordinate 0, and the
  disk's box spans ``[-r, r]`` in the two in-plane coordinates.

The outer faces are bounded by circle arcs; GProp integrates them exactly enough for
1e-12 relative.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, Session

RTOL: float = 1e-12

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, by kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def state(s: Session) -> tuple[bytes, tuple[tuple[int, ...], ...], int, int]:
    """The bytes, the live ids and the two counters."""
    return s.brep(), id_space(s), s.op_count, s.issued_id_count


@pytest.mark.parametrize(
    ("pattern", "faces", "inner_count", "inner_area"),
    [
        ("square", 5, 1, lambda a: 2.0 * a**2),
        ("hexagon", 18, 6, lambda a: 1.5 * math.sqrt(3.0) * a**2),
    ],
)
@pytest.mark.parametrize("ratio", [30.0, 67.0])
def test_a_divided_disk_covers_the_disk_and_its_inner_block_has_the_closed_form(
    pattern: str,
    faces: int,
    inner_count: int,
    inner_area: Callable[[float], float],
    ratio: float,
) -> None:
    r = 2.5
    a = ratio / 100.0 * r
    s = Session()

    delta = s.divided_disk(r, ratio, pattern=pattern)  # type: ignore[arg-type]

    table = s.entity_table(EntityKind.FACE)
    assert len(table.ids) == faces
    np.testing.assert_allclose(table.measure.sum(), math.pi * r**2, rtol=RTOL)
    radius_of_centroid = np.linalg.norm(table.centroid, axis=1)
    inner = radius_of_centroid < a
    assert int(np.count_nonzero(inner)) == inner_count
    np.testing.assert_allclose(
        float(table.measure[inner].sum()),
        inner_area(a),
        rtol=RTOL,
    )
    assert s.entity_types(EntityKind.FACE).types == ("Plane",) * faces
    assert all(
        s.origin(EntityId(int(i))).role == NameRole.CONSTRUCTED for i in delta.created
    )
    assert len(s.entities(EntityKind.SOLID)) == 0


@pytest.mark.parametrize(("orientation", "normal"), [("xy", 2), ("yz", 0), ("zx", 1)])
def test_the_orientation_puts_the_disk_in_its_plane(
    orientation: str, normal: int
) -> None:
    r = 3.0
    s = Session()

    s.divided_disk(r, 50.0, orientation=orientation)  # type: ignore[arg-type]

    table = s.entity_table(EntityKind.FACE)
    np.testing.assert_allclose(table.centroid[:, normal], 0.0, atol=1e-12)
    points = s.entity_table(EntityKind.VERTEX).centroid
    np.testing.assert_allclose(points[:, normal], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.max(np.linalg.norm(points, axis=1)), r, rtol=RTOL)
    in_plane = [k for k in range(3) if k != normal]
    # The square pattern's outer vertices lie on the plane's two axes, at +-r.
    on_rim = np.isclose(np.linalg.norm(points, axis=1), r, rtol=RTOL)
    np.testing.assert_allclose(
        np.sort(np.abs(points[on_rim][:, in_plane]).max(axis=1)), [r] * 4, rtol=RTOL
    )


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def disks() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
        s = Session()
        s.divided_disk(2.0, 67.0, pattern="hexagon", orientation="zx")
        s.divided_disk(1.0, 40.0)
        return s.brep(), id_space(s)

    first = disks()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(disks())

    assert all(other == first for other in others)


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((0.0, 50.0), {}, "the radius 0 is not above 1e-07"),
        ((-1.0, 50.0), {}, "the radius -1 is not above 1e-07"),
        ((1.0, 0.0), {}, "the ratio 0 is not strictly between 0 and 100"),
        ((1.0, 100.0), {}, "the ratio 100 is not strictly between 0 and 100"),
        ((1.0, math.nan), {}, "ratio must be a finite number"),
        (
            (1.0, 50.0),
            {"orientation": "xz"},
            "the orientation 'xz' is not one of xy, yz, zx",
        ),
        (
            (1.0, 50.0),
            {"pattern": "circle"},
            "the pattern 'circle' is not one of square, hexagon",
        ),
    ],
)
def test_divided_disk_refuses_degenerate_input(
    args: tuple[float, float], kwargs: dict[str, str], match: str
) -> None:
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=match) as err:
        s.divided_disk(*args, **kwargs)  # type: ignore[arg-type]

    assert "Session.divided_disk" in str(err.value)
    assert state(s) == before
