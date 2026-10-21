# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for locating entities: entities_on, in_place and find_same.

All three are queries: each test that calls one asserts that the bytes, the live ids,
``op_count`` and ``issued_id_count`` are what they were before.

The answers are checked against closed forms on axis-aligned boxes, a cylinder and a
sphere, where the region each entity lies in is known exactly:

* The box ``[0, 3] x [0, 7] x [0, 11]`` has its bottom face, four edges and four vertices
  on the plane ``z = 0``. Every other face, edge and vertex lies above it, on the side the
  normal ``+z`` points to, which GEOMAlgo calls "out": a face touching the plane along an
  edge and otherwise above it is out, because GEOMAlgo_StateCollector reads ON and OUT
  samples as OUT. The plane ``z = 5.5`` crosses the four side faces, so they are in no
  state, the bottom face is in and the top face is out.
* A cylinder of radius 2 has its lateral face on the cylinder region of the same axis and
  radius, its two caps in it, and its two circles and its seam on it.
* After a fillet of radius 1 on the top edge at ``y = 7``, the top face is the rectangle
  ``[0, 3] x [0, 6]``: area 18, centroid ``(1.5, 3, 11)``. A pin ``[1, 2] x [1, 2]``
  through the top face cuts a unit square out of it: area 20, centroid
  ``((21 * 1.5 - 1.5) / 20, (21 * 3.5 - 1.5) / 20, 11) = (1.5, 3.6, 11)``. Two unit boxes
  side by side, fused and unified, have one top face of area 2 centred at ``(1, 0.5, 1)``.

Measures and centroids of planar faces are compared at 1e-12 relative, the round-off of an
exact integration. The fillet's trimmed face is compared at 1e-9 relative, because the
fillet's boundary line comes from OCCT's fillet solver, which works to its own tolerance.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube, except where
a closed form needs unit boxes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import (
    BoxRegion,
    CylinderRegion,
    EntityId,
    EntityKind,
    PlaneRegion,
    PysmeshCancelled,
    Session,
    SphereRegion,
)

BOX_A: float = 3.0
BOX_B: float = 7.0
BOX_C: float = 11.0
PLANAR_RTOL: float = 1e-12
FILLET_RTOL: float = 1e-9

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)

# A 3 x 7 rectangle at z = 0 whose wire lists its four edges out of order. It is valid by
# BRepCheck_Analyzer; ShapeFix_Wire reorders the wire, so heal rebuilds the face.
REORDERED_RECTANGLE_BREP: bytes = b"""
CASCADE Topology V3, (c) Open Cascade
Locations 0
Curve2ds 4
1 0 0 0 1
1 0 0 1 0
1 3 0 0 1
1 0 7 1 0
Curves 4
1 0 0 0 -0 1 0
1 0 0 0 1 0 -0
1 3 0 0 -0 1 0
1 0 7 0 1 0 -0
Polygon3D 0
PolygonOnTriangulations 0
Surfaces 1
1 0 0 0 0 0 1 1 0 -0 -0 1 0
Triangulations 0

TShapes 11
Ve
1e-07
0 0 0
0 0

0101101
*
Ve
1e-07
0 7 0
0 0

0101101
*
Ed
 1e-07 1 1 0
1  1 0 0 7
2  1 1 0 0 7
0

0101000
+11 0 -10 0 *
Ve
1e-07
3 0 0
0 0

0101101
*
Ed
 1e-07 1 1 0
1  2 0 0 3
2  2 1 0 0 3
0

0101000
+11 0 -8 0 *
Ve
1e-07
3 7 0
0 0

0101101
*
Ed
 1e-07 1 1 0
1  3 0 0 7
2  3 1 0 0 7
0

0101000
+8 0 -6 0 *
Ed
 1e-07 1 1 0
1  4 0 0 3
2  4 1 0 0 3
0

0101000
+10 0 -6 0 *
Wi

0101100
-9 0 +5 0 +7 0 -4 0 *
Fa
0  1e-07 1 0

0101000
+3 0 *
Co

1100000
+2 0 *

+1 0
"""


def state(s: Session) -> tuple[bytes, tuple[tuple[int, ...], ...], int, int]:
    """The bytes, the live ids and the two counters: everything a query must keep."""
    ids = tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)
    return s.brep(), ids, s.op_count, s.issued_id_count


def box_session() -> Session:
    """The 3 x 7 x 11 box at the origin."""
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C)
    return s


def centroids(s: Session, ids: np.ndarray) -> np.ndarray:
    """(N, 3) centroids of the given entities, by the rule of mass_properties."""
    if len(ids) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return s.mass_properties([EntityId(int(i)) for i in ids]).centroid


def face_at(s: Session, centroid: tuple[float, float, float]) -> EntityId:
    """The one live face whose centroid is the given point."""
    table = s.entity_table(EntityKind.FACE)
    hits = [
        EntityId(int(i))
        for i, c in zip(table.ids, table.centroid, strict=True)
        if np.allclose(c, centroid, rtol=0.0, atol=1e-9)
    ]
    assert len(hits) == 1
    return hits[0]


def edge_at(s: Session, centroid: tuple[float, float, float]) -> EntityId:
    """The one live edge whose centroid is the given point."""
    table = s.entity_table(EntityKind.EDGE)
    hits = [
        EntityId(int(i))
        for i, c in zip(table.ids, table.centroid, strict=True)
        if np.allclose(c, centroid, rtol=0.0, atol=1e-9)
    ]
    assert len(hits) == 1
    return hits[0]


# ---- entities_on: planes ------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kind", "count", "total_measure"),
    [
        (EntityKind.FACE, 1, BOX_A * BOX_B),
        (EntityKind.EDGE, 4, 2.0 * (BOX_A + BOX_B)),
        (EntityKind.VERTEX, 4, 0.0),
    ],
)
def test_entities_on_the_plane_of_the_bottom_face_finds_its_boundary(
    kind: EntityKind, count: int, total_measure: float
) -> None:
    s = box_session()
    before = state(s)

    found = s.entities_on(kind, PlaneRegion((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), "on")

    assert len(found) == count
    assert list(found) == sorted(found)
    np.testing.assert_allclose(centroids(s, found)[:, 2], 0.0, rtol=0.0, atol=1e-12)
    measure = s.mass_properties([EntityId(int(i)) for i in found]).measure
    assert math.isclose(float(np.sum(measure)), total_measure, rel_tol=PLANAR_RTOL)
    assert state(s) == before


def test_entities_on_out_is_the_side_the_plane_normal_points_to() -> None:
    s = box_session()
    up = PlaneRegion((0.0, 0.0, 5.5), (0.0, 0.0, 1.0))
    down = PlaneRegion((0.0, 0.0, 5.5), (0.0, 0.0, -1.0))
    bottom = face_at(s, (1.5, 3.5, 0.0))
    top = face_at(s, (1.5, 3.5, 11.0))

    up_in = s.entities_on(EntityKind.FACE, up, "in")
    up_out = s.entities_on(EntityKind.FACE, up, "out")
    down_in = s.entities_on(EntityKind.FACE, down, "in")
    down_out = s.entities_on(EntityKind.FACE, down, "out")

    assert list(up_in) == [bottom]
    assert list(up_out) == [top]
    assert list(down_in) == [top]
    assert list(down_out) == [bottom]
    assert len(s.entities_on(EntityKind.FACE, up, "on")) == 0


def test_entities_on_a_face_touching_the_plane_reads_as_on_its_side() -> None:
    """A side face has samples ON the plane z = 0 and OUT of it: GEOMAlgo reads it OUT."""
    s = box_session()
    plane = PlaneRegion((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    bottom = face_at(s, (1.5, 3.5, 0.0))
    every = sorted(int(i) for i in s.entities(EntityKind.FACE))

    out = s.entities_on(EntityKind.FACE, plane, "out")
    on_out = s.entities_on(EntityKind.FACE, plane, "on_out")
    on_in = s.entities_on(EntityKind.FACE, plane, "on_in")
    inside = s.entities_on(EntityKind.FACE, plane, "in")
    solid_out = s.entities_on(EntityKind.SOLID, plane, "out")

    assert sorted(out) == sorted(set(every) - {bottom})
    assert list(on_out) == every
    assert list(on_in) == [bottom]
    assert len(inside) == 0
    assert list(solid_out) == [int(s.entities(EntityKind.SOLID)[0])]


def test_entities_on_a_split_id_across_the_plane_is_neither_in_nor_out() -> None:
    """split keeps the cut solid's id on both pieces, which lie on the two sides."""
    s = box_session()
    solid = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    s.add_box(5.0, 9.0, 10.0, origin=(-1.0, -1.0, 5.5))
    tool = EntityId(int(s.entities(EntityKind.SOLID)[-1]))
    s.split([solid], [tool])
    assert s.shape_count(solid) == 2
    plane = PlaneRegion((0.0, 0.0, 5.5), (0.0, 0.0, 1.0))

    inside = s.entities_on(EntityKind.SOLID, plane, "in")
    out = s.entities_on(EntityKind.SOLID, plane, "out")
    on_in = s.entities_on(EntityKind.SOLID, plane, "on_in")

    assert solid not in inside
    assert solid not in out
    assert solid not in on_in


def test_entities_on_the_default_tolerance_is_the_largest_vertex_tolerance() -> None:
    """A face 1e-5 above the plane is not on it at 1e-7, and is at 1e-4."""
    s = Session()
    s.add_box(BOX_A, BOX_B, BOX_C, origin=(0.0, 0.0, 1e-5))
    bottom = face_at(s, (1.5, 3.5, 1e-5))
    plane = PlaneRegion((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    default = s.entities_on(EntityKind.FACE, plane, "on")
    loose = s.entities_on(EntityKind.FACE, plane, "on", tolerance=1e-4)

    assert len(default) == 0
    assert list(loose) == [bottom]


def test_entities_on_writes_nothing_and_ignores_an_earlier_tessellation() -> None:
    """The finder samples a private copy, meshed as GEOM meshes it."""
    s = box_session()
    plane = PlaneRegion((0.0, 0.0, 5.5), (0.0, 0.0, 1.0))
    first = s.entities_on(EntityKind.FACE, plane, "in")
    s.tessellate(deflection=5.0, angle_deg=60.0)
    before = state(s)

    again = s.entities_on(EntityKind.FACE, plane, "in")

    assert list(again) == list(first)
    assert state(s) == before


def test_entities_on_an_empty_model_finds_nothing() -> None:
    s = Session()

    found = s.entities_on(EntityKind.FACE, PlaneRegion((0, 0, 0), (0, 0, 1)), "on")

    assert found.dtype == np.int64
    assert len(found) == 0


# ---- entities_on: cylinder, sphere, box, solid --------------------------------------- #


def test_entities_on_a_cylinder_region_finds_the_lateral_face_and_the_caps() -> None:
    s = Session()
    s.add_cylinder(2.0, 5.0)
    region = CylinderRegion((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0)
    lateral = face_at(s, (0.0, 0.0, 2.5))
    caps = sorted([face_at(s, (0.0, 0.0, 0.0)), face_at(s, (0.0, 0.0, 5.0))])

    on_faces = s.entities_on(EntityKind.FACE, region, "on")
    in_faces = s.entities_on(EntityKind.FACE, region, "in")
    on_edges = s.entities_on(EntityKind.EDGE, region, "on")
    out_faces = s.entities_on(EntityKind.FACE, region, "out")

    assert list(on_faces) == [lateral]
    assert list(in_faces) == caps
    assert len(on_edges) == 3
    lengths = s.mass_properties([EntityId(int(i)) for i in on_edges]).measure
    assert math.isclose(
        float(np.sum(lengths)), 2.0 * 2.0 * math.pi * 2.0 + 5.0, rel_tol=1e-9
    )
    assert len(out_faces) == 0


def test_entities_on_a_sphere_region_finds_its_face_on_and_a_box_inside() -> None:
    s = Session()
    s.add_sphere(2.0)
    sphere_face = int(s.entities(EntityKind.FACE)[0])
    s.add_box(1.0, 1.0, 1.0, origin=(-0.5, -0.5, -0.5))
    box_faces = sorted(int(i) for i in s.entities(EntityKind.FACE)[1:])
    region = SphereRegion((0.0, 0.0, 0.0), 2.0)

    on = s.entities_on(EntityKind.FACE, region, "on")
    inside = s.entities_on(EntityKind.FACE, region, "in")

    assert list(on) == [sphere_face]
    assert list(inside) == box_faces


@pytest.mark.parametrize(
    ("minimum", "maximum", "state_name"),
    [
        ((0.0, 0.0, 0.0), (BOX_A, BOX_B, BOX_C), "on"),
        ((-1.0, -1.0, -1.0), (4.0, 8.0, 12.0), "in"),
        ((10.0, 10.0, 10.0), (12.0, 12.0, 12.0), "out"),
    ],
)
def test_entities_on_a_box_region_classifies_every_face(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    state_name: str,
) -> None:
    s = box_session()
    every = sorted(int(i) for i in s.entities(EntityKind.FACE))

    found = s.entities_on(EntityKind.FACE, BoxRegion(minimum, maximum), state_name)  # type: ignore[arg-type]

    assert list(found) == every


def test_entities_on_a_solid_region_finds_the_faces_inside_and_on_it() -> None:
    s = box_session()
    outer = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    outer_faces = sorted(int(i) for i in s.entities(EntityKind.FACE))
    s.add_box(1.0, 1.0, 1.0, origin=(1.0, 1.0, 1.0))
    inner_faces = sorted(
        {int(i) for i in s.entities(EntityKind.FACE)} - set(outer_faces)
    )
    s.add_box(1.0, 1.0, 1.0, origin=(20.0, 0.0, 0.0))
    far_faces = sorted(
        {int(i) for i in s.entities(EntityKind.FACE)}
        - set(outer_faces)
        - set(inner_faces)
    )
    before = state(s)

    on = s.entities_on(EntityKind.FACE, outer, "on")
    inside = s.entities_on(EntityKind.FACE, outer, "in")
    out = s.entities_on(EntityKind.FACE, outer, "out")

    assert list(on) == outer_faces
    assert list(inside) == inner_faces
    assert list(out) == far_faces
    assert state(s) == before


# ---- entities_on: refusals ----------------------------------------------------------- #


@pytest.mark.parametrize(
    ("region", "state_name", "tolerance", "match"),
    [
        (PlaneRegion((0, 0, 0), (0, 0, 0)), "on", None, "plane normal"),
        (CylinderRegion((0, 0, 0), (0, 0, 1), 0.0), "on", None, "radius 0"),
        (SphereRegion((0, 0, 0), -1.0), "on", None, "radius -1"),
        (BoxRegion((0, 0, 0), (1, 0, 1)), "on", None, "extends 0 along y"),
        (PlaneRegion((0, 0, 0), (0, 0, 1)), "inside", None, "'inside'"),
        (PlaneRegion((0, 0, 0), (0, 0, 1)), "on", 0.0, "tolerance 0"),
        (PlaneRegion((0, 0, 0), (0, 0, 1)), "on", math.nan, "tolerance"),
        (PlaneRegion((math.inf, 0, 0), (0, 0, 1)), "on", None, "finite"),
        ("plane", "on", None, "is a str"),
    ],
)
def test_entities_on_refuses_a_degenerate_region(
    region: object, state_name: str, tolerance: float | None, match: str
) -> None:
    s = box_session()
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=match) as err:
        s.entities_on(EntityKind.FACE, region, state_name, tolerance=tolerance)  # type: ignore[arg-type]

    assert "Session.entities_on" in str(err.value)
    assert state(s) == before


def test_entities_on_refuses_a_region_id_that_is_not_one_live_solid() -> None:
    s = box_session()
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))
    other = Session()
    other.add_box(1.0, 1.0, 1.0)
    s.add_box(1.0, 1.0, 1.0, origin=(10.0, 0.0, 0.0))
    dead = EntityId(int(s.entities(EntityKind.SOLID)[1]))
    s.remove([dead])

    with pytest.raises(ps.PysmeshError, match=f"region entity {face} is a FACE"):
        s.entities_on(EntityKind.FACE, face, "on")
    with pytest.raises(ps.PysmeshError, match=f"entity {dead} is dead"):
        s.entities_on(EntityKind.FACE, dead, "on")


# ---- in_place ------------------------------------------------------------------------ #


def _found(
    s: Session, source: str, mark: ps.SnapshotMark, handoff: ps.Handoff, face: EntityId
) -> list[int]:
    """in_place of one old face, by the snapshot or by the handoff bytes."""
    if source == "snapshot":
        answer = s.in_place(EntityKind.FACE, mark, [face])
    else:
        ordinal = list(handoff.face_id).index(face)
        answer = s.in_place(EntityKind.FACE, handoff.brep, [ordinal])
    assert len(answer) == 1
    assert answer[0].dtype == np.int64
    return [int(i) for i in answer[0]]


@pytest.mark.parametrize("source", ["snapshot", "bytes"])
def test_in_place_after_a_fillet_finds_the_trimmed_face(source: str) -> None:
    s = box_session()
    top = face_at(s, (1.5, 3.5, BOX_C))
    edge = edge_at(s, (1.5, BOX_B, BOX_C))
    handoff = s.export_handoff()
    mark = s.snapshot()
    s.fillet([edge], 1.0)
    before = state(s)

    found = _found(s, source, mark, handoff, top)

    assert found == [top]
    m = s.mass_properties([top])
    assert math.isclose(float(m.measure[0]), 18.0, rel_tol=FILLET_RTOL)
    np.testing.assert_allclose(m.centroid[0], (1.5, 3.0, BOX_C), rtol=FILLET_RTOL)
    assert state(s) == before


@pytest.mark.parametrize("source", ["snapshot", "bytes"])
def test_in_place_after_a_fuse_finds_the_face_the_pin_cut(source: str) -> None:
    s = box_session()
    solid = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    top = face_at(s, (1.5, 3.5, BOX_C))
    s.add_box(1.0, 1.0, 2.0, origin=(1.0, 1.0, BOX_C - 1.0))
    pin = EntityId(int(s.entities(EntityKind.SOLID)[1]))
    handoff = s.export_handoff()
    mark = s.snapshot()
    s.fuse([solid], [pin])
    assert top in s.entities(EntityKind.FACE)

    found = _found(s, source, mark, handoff, top)

    assert found == [top]
    m = s.mass_properties([top])
    assert math.isclose(float(m.measure[0]), 20.0, rel_tol=PLANAR_RTOL)
    np.testing.assert_allclose(m.centroid[0], (1.5, 3.6, BOX_C), rtol=PLANAR_RTOL)


@pytest.mark.parametrize("source", ["snapshot", "bytes"])
def test_in_place_after_unify_finds_the_face_the_old_one_was_merged_into(
    source: str,
) -> None:
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    s.add_box(1.0, 1.0, 1.0, origin=(1.0, 0.0, 0.0))
    a, b = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))
    s.fuse([a], [b])
    left_top = face_at(s, (0.5, 0.5, 1.0))
    right_top = face_at(s, (1.5, 0.5, 1.0))
    handoff = s.export_handoff(allow_aliases=True)
    mark = s.snapshot()
    s.unify_same_domain()

    found = _found(s, source, mark, handoff, left_top)

    assert found == sorted([left_top, right_top])
    m = s.mass_properties([EntityId(i) for i in found])
    np.testing.assert_allclose(m.measure, [2.0, 2.0], rtol=PLANAR_RTOL)
    np.testing.assert_allclose(m.centroid, [(1.0, 0.5, 1.0)] * 2, rtol=PLANAR_RTOL)
    assert all(s.shape_count(EntityId(i)) == 1 for i in found)


@pytest.mark.parametrize("source", ["snapshot", "bytes"])
def test_in_place_after_a_heal_finds_the_rebuilt_face(source: str) -> None:
    s = Session(validate=False)
    s.add_brep(REORDERED_RECTANGLE_BREP)
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))
    handoff = s.export_handoff()
    mark = s.snapshot()
    delta = s.heal()
    assert list(delta.modified) == [face]

    found = _found(s, source, mark, handoff, face)

    assert found == [face]
    m = s.mass_properties([face])
    assert math.isclose(float(m.measure[0]), 21.0, rel_tol=PLANAR_RTOL)
    np.testing.assert_allclose(m.centroid[0], (1.5, 3.5, 0.0), rtol=PLANAR_RTOL)


def test_in_place_of_a_removed_face_is_an_empty_answer() -> None:
    s = box_session()
    s.add_box(1.0, 1.0, 1.0, origin=(10.0, 0.0, 0.0))
    far = EntityId(int(s.entities(EntityKind.SOLID)[1]))
    far_face = face_at(s, (10.5, 0.5, 1.0))
    top = face_at(s, (1.5, 3.5, BOX_C))
    mark = s.snapshot()
    s.remove([far])

    answer = s.in_place(EntityKind.FACE, mark, [far_face, top])

    assert len(answer) == 2
    assert len(answer[0]) == 0
    assert list(answer[1]) == [top]


@pytest.mark.parametrize(
    ("make_old", "match"),
    [
        (
            lambda h, m: (h.brep, [6]),
            "ordinal 6 is out of range: the source holds 6 FACE",
        ),
        (lambda h, m: (h.brep, [-1]), "ordinal -1 is out of range"),
        (lambda h, m: (b"not a brep", [0]), "not a BREP"),
        (lambda h, m: (h.brep, []), "nothing to find"),
        (lambda h, m: (m, []), "nothing to find"),
        (
            lambda h, m: (ps.SnapshotMark(7), [EntityId(2)]),
            "snapshot mark 7 is unknown",
        ),
        (lambda h, m: (m, [EntityId(1)]), "entity 1 is a SOLID in the snapshot"),
        (lambda h, m: ("bytes", [0]), "is a str"),
    ],
)
def test_in_place_refuses_a_bad_source_or_entry(make_old: object, match: str) -> None:
    s = box_session()
    handoff = s.export_handoff()
    mark = s.snapshot()
    source, old = make_old(handoff, mark)  # type: ignore[operator]
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=match) as err:
        s.in_place(EntityKind.FACE, source, old)

    assert "Session.in_place" in str(err.value)
    assert state(s) == before


def test_in_place_refuses_an_id_not_live_in_the_snapshot_and_a_discarded_mark() -> None:
    s = box_session()
    mark = s.snapshot()
    s.add_box(1.0, 1.0, 1.0, origin=(10.0, 0.0, 0.0))
    later = face_at(s, (10.5, 0.5, 1.0))
    s.discard_snapshot(s.snapshot())
    discarded = ps.SnapshotMark(1)

    with pytest.raises(
        ps.PysmeshError, match=f"entity {later} is not live in the snapshot"
    ):
        s.in_place(EntityKind.FACE, mark, [later])
    with pytest.raises(ps.PysmeshError, match="snapshot mark 1 was discarded"):
        s.in_place(EntityKind.FACE, discarded, [later])


def test_in_place_reports_progress_to_the_end_and_issues_nothing() -> None:
    s = box_session()
    faces = [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    mark = s.snapshot()
    seen: list[float] = []
    before = state(s)

    answer = s.in_place(EntityKind.FACE, mark, faces, progress=seen.append)

    assert [list(a) for a in answer] == [[f] for f in faces]
    assert seen
    assert seen == sorted(seen)
    assert seen[-1] == 1.0
    assert state(s) == before


def test_a_cancelled_in_place_raises_and_leaves_the_session_as_it_was() -> None:
    s = box_session()
    faces = [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    mark = s.snapshot()
    before = state(s)

    with pytest.raises(PysmeshCancelled):
        s.in_place(EntityKind.FACE, mark, faces, cancel=lambda: True)

    assert state(s) == before


# ---- find_same ----------------------------------------------------------------------- #


def two_boxes(shift: float) -> tuple[Session, EntityId, EntityId]:
    """The 3 x 7 x 11 box and a copy of it moved by ``shift`` along x."""
    s = box_session()
    s.add_box(BOX_A, BOX_B, BOX_C, origin=(shift, 0.0, 0.0))
    a, b = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))
    return s, a, b


def body_ids(s: Session, solid: EntityId, kind: EntityKind) -> set[int]:
    """The live ids of one kind in the body of one solid."""
    pairs = s.adjacency(EntityKind.SOLID, kind)
    return {
        int(o) for e, o in zip(pairs.ids, pairs.related, strict=True) if int(e) == solid
    }


@pytest.mark.parametrize(
    ("kind", "centroid"),
    [
        (EntityKind.FACE, (1.5, 3.5, BOX_C)),
        (EntityKind.EDGE, (1.5, BOX_B, BOX_C)),
        (EntityKind.VERTEX, (BOX_A, BOX_B, BOX_C)),
    ],
)
def test_find_same_finds_the_coincident_entity_of_a_copy(
    kind: EntityKind, centroid: tuple[float, float, float]
) -> None:
    s, a, b = two_boxes(0.0)
    table = s.entity_table(kind)
    same_place = [
        EntityId(int(i))
        for i, c in zip(table.ids, table.centroid, strict=True)
        if np.allclose(c, centroid, rtol=0.0, atol=1e-9)
    ]
    assert len(same_place) == 2
    mine = min(same_place, key=lambda i: i not in body_ids(s, a, kind))
    theirs = [i for i in same_place if i != mine]
    before = state(s)

    found = s.find_same(mine, [b])

    assert list(found) == theirs
    np.testing.assert_allclose(centroids(s, found)[0], centroid, rtol=PLANAR_RTOL)
    assert state(s) == before


def test_find_same_of_a_solid_in_its_own_body_and_a_copy_finds_both() -> None:
    s, a, b = two_boxes(0.0)

    found = s.find_same(a, [a, b])

    assert list(found) == [a, b]


@pytest.mark.parametrize(("shift", "expected"), [(5e-8, True), (1e-6, False)])
def test_find_same_uses_geoms_tolerance_of_1e_7(shift: float, expected: bool) -> None:
    s, a, b = two_boxes(shift)

    found = s.find_same(a, [b])

    assert (list(found) == [b]) is expected


def test_find_same_refuses_a_split_entity_a_dead_id_and_an_empty_search() -> None:
    s = box_session()
    solid = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    s.add_box(5.0, 9.0, 10.0, origin=(-1.0, -1.0, 5.5))
    tool = EntityId(int(s.entities(EntityKind.SOLID)[-1]))
    s.add_box(1.0, 1.0, 1.0, origin=(10.0, 0.0, 0.0))
    dead = EntityId(int(s.entities(EntityKind.SOLID)[-1]))
    s.remove([dead])
    s.split([solid], [tool])

    with pytest.raises(ps.PysmeshError, match=f"entity {solid} denotes 2 SOLIDs"):
        s.find_same(solid, [solid])
    with pytest.raises(ps.PysmeshError, match=f"entity {dead} is dead"):
        s.find_same(dead, [tool])
    with pytest.raises(ps.PysmeshError, match="at least one entity"):
        s.find_same(tool, [])
