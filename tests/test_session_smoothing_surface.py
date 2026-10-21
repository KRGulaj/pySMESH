# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for Session.smoothing_surface: GEOM's MakeSmoothingSurface over a point cloud.

The answers:

* Points sampled on the plane ``z = 0.3 x + 0.2 y + 1`` (seeded) lie on the average plane,
  so the plate is not deformed and the face is that plane: every point lies on the face
  within 1e-12 (round-off of the B-spline evaluation), and the face normal at each point
  is ``(-0.3, -0.2, 1) / |(-0.3, -0.2, 1)|`` within 1e-9 rad (brief section 4.1).
* Points sampled on a sphere cap of radius 10 (polar angle up to 30 degrees, seeded):
  GEOM's approximation holds its deviation at the points below
  ``max(1e-4, 10 * max_distance)`` (GeomPlate_MakeApprox's order-0 criterion), and the
  plate passes through the points within its own G0 error, so every point lies within
  that bound of the face. Measured for this set (p4/explore/sphere_measure.log):
  9.7e-5 at ``max_distance`` 1e-5, 4.8e-4 at 1e-4 and 1e-3.
* Three points span exactly their plane: each lies on the face within 1e-12.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, Session

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


def plane_points() -> NDArray[np.float64]:
    """40 seeded points on z = 0.3 x + 0.2 y + 1."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(-5.0, 5.0, size=(40, 2))
    return np.column_stack([xy, 0.3 * xy[:, 0] + 0.2 * xy[:, 1] + 1.0])


def sphere_points() -> NDArray[np.float64]:
    """60 seeded points on a cap of the sphere of radius 10 about the origin."""
    rng = np.random.default_rng(42)
    theta = rng.uniform(0.0, math.radians(30.0), 60)
    phi = rng.uniform(0.0, 2.0 * math.pi, 60)
    return 10.0 * np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def the_face(s: Session) -> EntityId:
    """The session's one face."""
    (face,) = s.entities(EntityKind.FACE)
    return EntityId(int(face))


def test_points_on_a_plane_give_that_plane() -> None:
    points = plane_points()
    s = Session()

    delta = s.smoothing_surface(points)

    face = the_face(s)
    projection = s.project_on_face(face, points)
    assert float(np.max(projection.distance)) < 1e-12
    normals = s.surface_at(face, projection.uv).normals
    expected = np.array([-0.3, -0.2, 1.0]) / np.linalg.norm([-0.3, -0.2, 1.0])
    deviation = np.arccos(np.clip(np.abs(normals @ expected), -1.0, 1.0))
    assert float(np.max(deviation)) < 1e-9
    assert delta.valid is True


@pytest.mark.parametrize("max_distance", [1e-5, 1e-4, 1e-3])
def test_points_on_a_sphere_cap_lie_within_the_approximation_criterion(
    max_distance: float,
) -> None:
    points = sphere_points()
    s = Session()

    s.smoothing_surface(points, max_distance=max_distance)

    distance = s.project_on_face(the_face(s), points).distance
    assert float(np.max(distance)) < max(1e-4, 10.0 * max_distance)


def test_three_points_lie_on_the_face() -> None:
    points = np.array([(0.0, 0.0, 0.0), (4.0, 1.0, 0.5), (1.0, 3.0, -0.5)])
    s = Session()

    s.smoothing_surface(points)

    assert float(np.max(s.project_on_face(the_face(s), points).distance)) < 1e-12


def test_the_face_is_one_new_body_of_constructed_ids() -> None:
    s = Session()
    s.add_box(1.0, 2.0, 3.0)
    before = {int(i) for k in _KINDS for i in s.entities(k)}

    delta = s.smoothing_surface(plane_points())

    created = {int(i) for i in delta.created}
    after = {int(i) for k in _KINDS for i in s.entities(k)}
    assert after - before == created
    assert len(delta.modified) == 0 and len(delta.deleted) == 0
    kinds = sorted(s.entity_kind(EntityId(i)) for i in created)
    assert kinds.count(EntityKind.FACE) == 1
    assert all(s.origin(EntityId(i)).role == NameRole.CONSTRUCTED for i in created)


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def smoothed() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
        s = Session()
        s.smoothing_surface(sphere_points())
        return s.brep(), id_space(s)

    first = smoothed()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(smoothed())

    assert all(other == first for other in others)


@pytest.mark.parametrize(
    ("points", "kwargs", "match"),
    [
        ([(0, 0, 0), (1, 0, 0)], {}, "points needs at least 3 points \\(got 2\\)"),
        (
            [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)],
            {},
            "the 4 points lie on one line",
        ),
        (
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)],
            {},
            "points 2 and 3 \\(0-based rows\\) coincide within 1e-07",
        ),
        ([(0, 0, 0), (1, 0, 0), (0, math.nan, 0)], {}, "row 2 holds nan"),
        ([(0, 0), (1, 0), (0, 1)], {}, "must be an \\(N, 3\\) array"),
        (
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            {"max_points": 0},
            "max_points 0 is below 1",
        ),
        (
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            {"max_degree": 0},
            "max_degree 0 is below 1",
        ),
        (
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            {"max_distance": -1.0},
            "max_distance -1 is negative",
        ),
        (
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            {"max_distance": math.inf},
            "max_distance must be a finite number \\(got inf\\)",
        ),
    ],
)
def test_smoothing_surface_refuses_degenerate_input(
    points: list[tuple[float, ...]], kwargs: dict[str, float], match: str
) -> None:
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=match) as err:
        s.smoothing_surface(np.array(points, dtype=np.float64), **kwargs)  # type: ignore[arg-type]

    assert "smoothing_surface" in str(err.value) or "(N, 3)" in str(err.value)
    assert state(s) == before
