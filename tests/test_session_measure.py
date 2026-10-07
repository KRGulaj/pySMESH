# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-07

"""Gates for the measure rule: volumes, areas and centroids against oracles GProp cannot set.

OCCT's adaptive rule (``BRepGProp_Face``) has no case for a surface of extrusion or of
revolution. It integrates such a face as one span of degree 2, never refines along the
basis curve, and reports an error of about 1e-16. On a prism over a closed B-spline it read
the volume 10 % off, and the value stepped with the precision instead of converging.

Every oracle here is independent of GProp:

* the area and the centroid of a planar profile by the shoelace sums of densely sampled
  curve points (:meth:`Session.curve_at`);
* Pappus's theorems for a profile revolved a full turn;
* the length and the centroid of an edge by pySMESH's own Gauss-Kronrod edge rule.

The profile is the one the defect was found on: a B-spline fitted through 37 points of an
ellipse with semi-axes 2.9 and 1.7, closed on its first point, with the
:meth:`Session.add_spline` defaults (degree 3, 40 poles).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from pysmesh import EntityId, EntityKind, Session

# The profile: an ellipse of semi-axes A_AXIS and B_AXIS, sampled at SAMPLES points.
A_AXIS: float = 2.9
B_AXIS: float = 1.7
SAMPLES: int = 37
# The prism's height, and the revolution axis: the line x = AXIS_X, z = 0, along y.
HEIGHT: float = 4.2
AXIS_X: float = -5.0
# Curve samples for the shoelace sums. The polygon's error falls as 1 / N^2; measured on this
# profile it is 4.6e-11 of the area at this density (1.8e-8 at 20 001).
SHOELACE_SAMPLES: int = 400_001

DEFAULT_PRECISION: float = 1e-6
# The precisions the convergence gates are asserted at.
PRECISIONS: tuple[float, ...] = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)


def _new_ids(s: Session, kind: EntityKind, before: set[int]) -> list[EntityId]:
    """The live ids of one kind that are not in ``before``."""
    return [EntityId(int(i)) for i in s.entities(kind) if int(i) not in before]


def _ids(s: Session, kind: EntityKind) -> set[int]:
    """The live ids of one kind."""
    return {int(i) for i in s.entities(kind)}


def _profile(s: Session) -> tuple[EntityId, EntityId]:
    """The closed spline profile in the plane z = 0 and the planar face it bounds."""
    t = 2.0 * math.pi * np.arange(SAMPLES) / SAMPLES
    points = np.c_[A_AXIS * np.cos(t), B_AXIS * np.sin(t), np.zeros(SAMPLES)]
    edges = _ids(s, EntityKind.EDGE)
    s.add_spline(np.vstack([points, points[:1]]))
    edge = _new_ids(s, EntityKind.EDGE, edges)
    faces = _ids(s, EntityKind.FACE)
    s.make_face(edge)
    return edge[0], _new_ids(s, EntityKind.FACE, faces)[0]


def _shoelace(s: Session, edge: EntityId) -> tuple[float, NDArray[np.float64]]:
    """The area inside a closed planar edge in z = 0, and its centroid, by shoelace sums."""
    t0, t1 = s.edge_parameter_bounds([edge])[0]
    points = s.curve_at(edge, np.linspace(t0, t1, SHOELACE_SAMPLES)).points[:-1]
    x, y = points[:, 0], points[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = 0.5 * float(cross.sum())
    centroid = np.array(
        [
            float(((x + xn) * cross).sum()) / (6.0 * area),
            float(((y + yn) * cross).sum()) / (6.0 * area),
            0.0,
        ]
    )
    return abs(area), centroid


def _prism() -> tuple[Session, EntityId, float, NDArray[np.float64]]:
    """The prism of the profile over HEIGHT, its solid id, cap area and cap centroid."""
    s = Session()
    edge, cap = _profile(s)
    area, centroid = _shoelace(s, edge)
    solids = _ids(s, EntityKind.SOLID)
    s.extrude([cap], (0.0, 0.0, HEIGHT))
    return s, _new_ids(s, EntityKind.SOLID, solids)[0], area, centroid


def _revolved() -> tuple[Session, EntityId, float]:
    """The profile revolved a full turn about the axis x = AXIS_X, z = 0, along y.

    Returns the session, the solid's id, and its volume by Pappus: 2 pi rbar A, with A and
    the distance rbar from the axis to the profile's centroid from the shoelace sums.
    """
    s = Session()
    edge, cap = _profile(s)
    area, centroid = _shoelace(s, edge)
    solids = _ids(s, EntityKind.SOLID)
    s.revolve([cap], (AXIS_X, 0.0, 0.0), (0.0, 1.0, 0.0))
    volume = 2.0 * math.pi * (float(centroid[0]) - AXIS_X) * area
    return s, _new_ids(s, EntityKind.SOLID, solids)[0], volume


# ------------------------------------------- The volume of a free-form prism (M1) --- #


def test_a_spline_prism_volume_at_the_default_equals_cap_area_times_height() -> None:
    """The defect: 71.799 at the default against 65.050 (+10 %), reported error 1.9e-17."""
    s, solid, area, _ = _prism()

    table = s.mass_properties([solid])

    assert float(table.measure[0]) == pytest.approx(
        area * HEIGHT, rel=DEFAULT_PRECISION
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_spline_prism_volume_converges_within_ten_times_the_precision(
    precision: float,
) -> None:
    """The defect read 57.97, 71.80 and 61.48 as the precision went from 1e-4 to 1e-9."""
    s, solid, area, _ = _prism()

    table = s.mass_properties([solid], precision=precision)

    assert float(table.measure[0]) == pytest.approx(area * HEIGHT, rel=10.0 * precision)


def test_a_spline_prism_centroid_lies_half_the_height_above_the_cap_centroid() -> None:
    """The prism's centroid lies above the cap's, at half its height (1e-9 of its size).

    The defect put it 0.235 off the cap's centroid in x.
    """
    s, solid, _, centroid = _prism()
    expected = centroid + np.array([0.0, 0.0, 0.5 * HEIGHT])

    table = s.mass_properties([solid])

    assert np.linalg.norm(table.centroid[0] - expected) <= 1e-9 * HEIGHT


@pytest.mark.parametrize("precision", [None, 1e-6, 1e-7, 1e-8, 1e-9])
def test_a_revolved_spline_profile_volume_meets_pappus_within_ten_times_the_precision(
    precision: float | None,
) -> None:
    """Pappus: V = 2 pi rbar A, to the default precision without one, else to 10 times it.

    The defect read 1.5e-5 high at the default and 3.2e-7 low at 1e-9.
    """
    s, solid, volume = _revolved()
    bound = DEFAULT_PRECISION if precision is None else 10.0 * precision

    table = s.mass_properties([solid], precision=precision)

    assert float(table.measure[0]) == pytest.approx(volume, rel=bound)
