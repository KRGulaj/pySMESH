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


# ------------------------------------ The area of a free-form swept face (M2) --- #

# The analytic sides of (c): radii, height and the cone's top radius.
CYLINDER_R: float = 1.7
CONE_R1: float = 2.3
CONE_R2: float = 0.9
SIDE_H: float = 3.1


def _face_of_type(s: Session, name: str) -> EntityId:
    """The single face of the session whose surface is of the named type."""
    table = s.entity_types(EntityKind.FACE)
    found = [EntityId(int(i)) for i, t in zip(table.ids, table.types) if t == name]
    assert len(found) == 1, (name, list(table.types))
    return found[0]


def _edge_rule(s: Session, edge: EntityId) -> tuple[float, NDArray[np.float64]]:
    """An edge's length and centroid by pySMESH's Gauss-Kronrod edge rule at 1e-12."""
    table = s.mass_properties([edge], precision=1e-12)
    return float(table.measure[0]), np.asarray(table.centroid[0], dtype=np.float64)


def _swept_side(revolve: bool) -> tuple[Session, EntityId, EntityId]:
    """The prism or the revolved solid of the profile, its swept face and the profile edge."""
    s = Session()
    edge, cap = _profile(s)
    if revolve:
        s.revolve([cap], (AXIS_X, 0.0, 0.0), (0.0, 1.0, 0.0))
        return s, _face_of_type(s, "Revolution"), edge
    s.extrude([cap], (0.0, 0.0, HEIGHT))
    return s, _face_of_type(s, "Extrusion"), edge


def _ring_weighted_axial(s: Session, edge: EntityId) -> float:
    """The mean of the profile's y weighted by r ds, r its distance from the axis.

    The centroid of a surface of revolution lies on the axis at this height (Pappus's
    surface theorem weights each arc element by the ring it sweeps). Polygon sums over the
    curve samples.
    """
    t0, t1 = s.edge_parameter_bounds([edge])[0]
    points = s.curve_at(edge, np.linspace(t0, t1, SHOELACE_SAMPLES)).points
    middle = 0.5 * (points[1:] + points[:-1])
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    ring = (middle[:, 0] - AXIS_X) * ds
    return float((middle[:, 1] * ring).sum() / ring.sum())


def test_a_spline_prism_side_area_equals_the_curve_length_times_the_height() -> None:
    """The defect: 4.6e-4 high at the default, reported error 0, and no convergence."""
    s, side, edge = _swept_side(revolve=False)
    length, _ = _edge_rule(s, edge)

    table = s.mass_properties([side])

    assert float(table.measure[0]) == pytest.approx(length * HEIGHT, rel=1e-9)


def test_a_spline_prism_side_centroid_lies_half_the_height_above_the_curve() -> None:
    """The side's centroid is the curve's, raised by half the height (1e-9 of the height).

    The defect put it 0.056 off the curve's centroid in x.
    """
    s, side, edge = _swept_side(revolve=False)
    _, centroid = _edge_rule(s, edge)
    expected = centroid + np.array([0.0, 0.0, 0.5 * HEIGHT])

    table = s.mass_properties([side])

    assert np.linalg.norm(table.centroid[0] - expected) <= 1e-9 * HEIGHT


def test_a_revolved_spline_side_area_at_1e_9_meets_pappus_within_1e_8() -> None:
    """Pappus: A = 2 pi r L, L the curve's length and r its centroid's distance from the axis.

    The defect read 3.4e-8 high at 1e-9. At the default both rules are within 1e-6 (the
    defect 8.5e-7, the B-spline copy 1.7e-7), and at 1e-8 the defect stays within ten times
    the precision, so only 1e-9 tells them apart.
    """
    s, side, edge = _swept_side(revolve=True)
    length, centroid = _edge_rule(s, edge)
    expected = 2.0 * math.pi * (float(centroid[0]) - AXIS_X) * length

    table = s.mass_properties([side], precision=1e-9)

    assert float(table.measure[0]) == pytest.approx(expected, rel=1e-8)


def test_a_revolved_spline_side_centroid_lies_on_the_axis_at_the_weighted_height() -> (
    None
):
    """On the axis, at the profile's y weighted by r ds (1e-9 of the axis distance).

    The defect put it 1.8e-4 off the axis.
    """
    s, side, edge = _swept_side(revolve=True)
    expected = np.array([AXIS_X, _ring_weighted_axial(s, edge), 0.0])

    table = s.mass_properties([side])

    assert np.linalg.norm(table.centroid[0] - expected) <= 1e-9 * abs(AXIS_X)


def test_a_cylinder_and_a_cone_side_area_equals_its_closed_form() -> None:
    """Analytic sides keep the exact rule: 2 pi r h and pi (r1 + r2) s, within 1e-14."""
    s = Session()
    s.add_cylinder(CYLINDER_R, SIDE_H, (0.37, -2.9, 1.3), (0.3, -0.4, 0.866))
    s.add_cone(CONE_R1, CONE_R2, SIDE_H, (5.0, 1.0, -2.0), (0.0, 0.6, 0.8))
    slant = math.hypot(SIDE_H, CONE_R1 - CONE_R2)
    expected = {
        "Cylinder": 2.0 * math.pi * CYLINDER_R * SIDE_H,
        "Cone": math.pi * (CONE_R1 + CONE_R2) * slant,
    }
    sides = {name: _face_of_type(s, name) for name in expected}

    table = s.mass_properties(list(sides.values()))

    for (name, area), measured in zip(expected.items(), table.measure, strict=True):
        assert float(measured) == pytest.approx(area, rel=1e-14), name
