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
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
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


def test_match_faces_finds_each_swept_face_by_the_centroid_faces_reports() -> None:
    """Shape.match_faces and Shape.faces take their centroids from one rule.

    The defect: match_faces took GProp's fixed rule and faces() its adaptive rule, so on
    the prism's extruded side the two centroids lay 0.06 apart.
    """
    s, _, _ = _swept_side(revolve=False)
    shape = ps.load_brep(s.brep())
    faces = shape.faces()
    centroids = np.array([f.centroid for f in faces], dtype=np.float64)

    ids = shape.match_faces(centroids, tol=1e-9)

    assert ids.tolist() == [f.id for f in faces]


# --------------------------------------------- Exactness on analytic solids (M3) --- #

# The plate, PLATE_X x PLATE_Y x PLATE_Z, its four vertical corners rounded by CORNER_R.
PLATE_X: float = 60.0
PLATE_Y: float = 40.0
PLATE_Z: float = 7.0
CORNER_R: float = 0.7
# Through holes (x, y, radius). The first is the 2 mm hole of the hole gate.
THROUGH_HOLES: tuple[tuple[float, float, float], ...] = (
    (8.0, 8.0, 1.0),
    (16.0, 8.0, 1.5),
    (30.0, 10.0, 4.5),
)
# A blind hole down from the top face: (x, y, radius, depth).
BLIND_HOLE: tuple[float, float, float, float] = (50.0, 10.0, 2.0, 4.0)
# A cylindrical boss on the top face: (x, y, radius, height).
BOSS: tuple[float, float, float, float] = (12.0, 30.0, 3.0, 5.0)
# A rectangular pocket in the top face: corner (x, y), size (dx, dy), depth. Its four
# vertical inner corners are rounded by POCKET_R.
POCKET: tuple[float, float, float, float, float] = (30.0, 22.0, 20.0, 12.0, 3.0)
POCKET_R: float = 1.0
# The area a fillet of radius 1 takes out of a square corner: 1 - pi / 4.
BLEND: float = 1.0 - math.pi / 4.0


def _vertical_edges(
    s: Session, xs: tuple[float, float], ys: tuple[float, float]
) -> list[EntityId]:
    """The vertical line edges standing at x in ``xs`` and y in ``ys``."""
    table = s.bounding_boxes(EntityKind.EDGE)
    found = []
    for i, b in zip(table.ids, table.bbox, strict=True):
        upright = b[0] == b[3] and b[1] == b[4] and b[5] > b[2]
        if upright and b[0] in xs and b[1] in ys:
            found.append(EntityId(int(i)))
    return found


def _add_solid(s: Session, add: Callable[[Session], object]) -> list[EntityId]:
    """Run ``add`` and return the solids it issued."""
    before = _ids(s, EntityKind.SOLID)
    add(s)
    return _new_ids(s, EntityKind.SOLID, before)


def _plate(hole_2mm: bool, corners: int) -> tuple[Session, EntityId]:
    """The plate with every feature; ``corners`` of its vertical corners rounded."""
    s = Session()
    s.add_box(PLATE_X, PLATE_Y, PLATE_Z)
    rounded = _vertical_edges(s, (0.0, PLATE_X), (0.0, PLATE_Y))[:corners]
    s.fillet(rounded, CORNER_R)
    holes = THROUGH_HOLES if hole_2mm else THROUGH_HOLES[1:]
    tools = [(x, y, -1.0, r, PLATE_Z + 2.0) for x, y, r in holes]
    x, y, r, depth = BLIND_HOLE
    tools.append((x, y, PLATE_Z - depth, r, depth + 1.0))
    for x, y, z, r, h in tools:
        body = list(s.entities(EntityKind.SOLID))
        tool = _add_solid(
            s, lambda t, x=x, y=y, z=z, r=r, h=h: t.add_cylinder(r, h, (x, y, z))
        )
        s.cut([EntityId(int(i)) for i in body], tool)
    x, y, r, h = BOSS
    body = [EntityId(int(i)) for i in s.entities(EntityKind.SOLID)]
    boss = _add_solid(s, lambda t: t.add_cylinder(r, h, (x, y, PLATE_Z)))
    s.fuse(body, boss)
    x, y, dx, dy, depth = POCKET
    body = [EntityId(int(i)) for i in s.entities(EntityKind.SOLID)]
    pocket = _add_solid(
        s, lambda t: t.add_box(dx, dy, depth + 1.0, origin=(x, y, PLATE_Z - depth))
    )
    s.fillet(_vertical_edges(s, (x, x + dx), (y, y + dy)), POCKET_R)
    s.cut(body, pocket)
    solids = [EntityId(int(i)) for i in s.entities(EntityKind.SOLID)]
    assert len(solids) == 1
    return s, solids[0]


def _plate_volume(hole_2mm: bool, corners: int) -> float:
    """The plate's volume in closed form: the box less or plus each feature."""
    holes = THROUGH_HOLES if hole_2mm else THROUGH_HOLES[1:]
    _, _, blind_r, blind_depth = BLIND_HOLE
    _, _, boss_r, boss_h = BOSS
    _, _, dx, dy, depth = POCKET
    return math.fsum(
        [
            PLATE_X * PLATE_Y * PLATE_Z,
            -corners * BLEND * CORNER_R**2 * PLATE_Z,
            *(-math.pi * r * r * PLATE_Z for _, _, r in holes),
            -math.pi * blind_r**2 * blind_depth,
            math.pi * boss_r**2 * boss_h,
            -(dx * dy - 4.0 * BLEND * POCKET_R**2) * depth,
        ]
    )


@pytest.fixture(scope="module")
def plates() -> dict[str, tuple[Session, EntityId]]:
    """The full plate, the plate without its 2 mm hole, and one with 3 round corners."""
    return {
        "full": _plate(hole_2mm=True, corners=4),
        "no_hole": _plate(hole_2mm=False, corners=4),
        "three_corners": _plate(hole_2mm=True, corners=3),
    }


def _volume(plate: tuple[Session, EntityId]) -> float:
    """The plate's volume at the default precision."""
    s, solid = plate
    return float(s.mass_properties([solid]).measure[0])


def test_the_plate_volume_equals_its_closed_form(
    plates: dict[str, tuple[Session, EntityId]],
) -> None:
    """Holes, a blind hole, a boss, a filleted pocket and rounded corners: within 1e-13.

    Planes and cylinders only, every edge a line or a circle. The adaptive rule read it
    6.6e-13 high at the default and reported an error of 4e-16.
    """
    expected = _plate_volume(hole_2mm=True, corners=4)

    volume = _volume(plates["full"])

    assert volume == pytest.approx(expected, rel=1e-13)


def test_removing_the_2_mm_hole_changes_the_volume_by_pi_r2_h(
    plates: dict[str, tuple[Session, EntityId]],
) -> None:
    """The volume a caller removes with the hole, pi r^2 h, within 1e-12 of itself."""
    r = THROUGH_HOLES[0][2]

    removed = _volume(plates["no_hole"]) - _volume(plates["full"])

    assert removed == pytest.approx(math.pi * r * r * PLATE_Z, rel=1e-12)


def test_rounding_one_more_corner_changes_the_volume_by_its_blend_times_length(
    plates: dict[str, tuple[Session, EntityId]],
) -> None:
    """A corner fillet removes (r^2 - pi r^2 / 4) L, within 1e-11 of itself.

    The adaptive rule read the difference 4.1e-9 off.
    """
    removed = _volume(plates["three_corners"]) - _volume(plates["full"])

    assert removed == pytest.approx(BLEND * CORNER_R**2 * PLATE_Z, rel=1e-11)


# The primitives of gate (d): each builder and the closed forms of its solid's volume,
# its face areas and its edge lengths (a degenerated edge, a sphere's pole, measures 0).
_R1, _R2, _H = 2.3, 0.9, 3.1
_SLANT = math.hypot(_H, _R1 - _R2)
PRIMITIVES: dict[str, tuple[Callable[[Session], object], dict[str, list[float]]]] = {
    "box": (
        lambda s: s.add_box(3.0, 7.0, 11.0, origin=(0.37, -2.9, 4.2)),
        {
            "volume": [231.0],
            "areas": [21.0, 21.0, 33.0, 33.0, 77.0, 77.0],
            "lengths": [3.0] * 4 + [7.0] * 4 + [11.0] * 4,
        },
    ),
    "cylinder": (
        lambda s: s.add_cylinder(1.7, 4.2, (0.37, -2.9, 1.3), (0.3, -0.4, 0.866)),
        {
            "volume": [math.pi * 1.7**2 * 4.2],
            "areas": [2.0 * math.pi * 1.7 * 4.2] + [math.pi * 1.7**2] * 2,
            "lengths": [2.0 * math.pi * 1.7] * 2 + [4.2],
        },
    ),
    "cone": (
        lambda s: s.add_cone(_R1, _R2, _H, (1.0, 2.0, 3.0), (0.6, 0.0, 0.8)),
        {
            "volume": [math.pi * _H * (_R1**2 + _R1 * _R2 + _R2**2) / 3.0],
            "areas": [
                math.pi * (_R1 + _R2) * _SLANT,
                math.pi * _R1**2,
                math.pi * _R2**2,
            ],
            "lengths": [2.0 * math.pi * _R1, 2.0 * math.pi * _R2, _SLANT],
        },
    ),
    "sphere": (
        lambda s: s.add_sphere(2.3, (1.0, 2.0, 3.0)),
        {
            "volume": [4.0 / 3.0 * math.pi * 2.3**3],
            "areas": [4.0 * math.pi * 2.3**2],
            "lengths": [0.0, 0.0, math.pi * 2.3],
        },
    ),
    "torus": (
        lambda s: s.add_torus(5.0, 1.2, (-1.0, 0.5, 2.0), (0.0, 0.6, 0.8)),
        {
            "volume": [2.0 * math.pi**2 * 5.0 * 1.2**2],
            "areas": [4.0 * math.pi**2 * 5.0 * 1.2],
            "lengths": [2.0 * math.pi * 1.2, 2.0 * math.pi * (5.0 + 1.2)],
        },
    ),
}


@pytest.mark.parametrize("name", sorted(PRIMITIVES))
def test_every_measure_of_a_primitive_is_its_closed_form_within_1e_14(
    name: str,
) -> None:
    """The solid's volume, every face's area and every edge's length, within 1e-14.

    A pole's degenerated edge measures exactly 0.
    """
    build, expected = PRIMITIVES[name]
    s = Session()
    build(s)
    measured = {
        key: sorted(float(m) for m in s.mass_properties(sorted(_ids(s, kind))).measure)
        for key, kind in (
            ("volume", EntityKind.SOLID),
            ("areas", EntityKind.FACE),
            ("lengths", EntityKind.EDGE),
        )
    }

    for key, values in expected.items():
        assert measured[key] == pytest.approx(sorted(values), rel=1e-14, abs=0.0), key
