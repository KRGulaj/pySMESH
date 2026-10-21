# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for extrude's keyword arguments symmetric and draft_angle.

The profile is a rectangle ``a x b`` on the plane z = 0, swept along z by ``h``.

* Symmetric: the solid is ``[0, a] x [0, b] x [-h, h]``: volume ``2 a b h``, centroid at
  z = 0, and the profile face is its face at z = -h.
* A positive draft ``alpha`` narrows the solid: the far face is the rectangle offset
  inward by ``d = h tan(alpha)``, so the solid is a prismatoid of volume
  ``h / 6 (A0 + 4 Am + A1)`` with ``A0 = a b``, ``Am = (a - d)(b - d)`` and
  ``A1 = (a - 2d)(b - 2d)``. The wall over a side of length ``l`` is a trapezoid of
  parallel sides ``l`` and ``l - 2d`` and height ``h / cos(alpha)``: area
  ``(l - d) h / cos(alpha)``.
* A negative draft widens it, and OCCT's arc join rounds the four corners: the section at
  height z is the rectangle grown by ``k z`` (``k = tan|alpha|``) with quarter-circle
  corners, of area ``a b + 2 (a + b) k z + pi k^2 z^2``, so the volume is
  ``a b h + (a + b) k h^2 + pi k^2 h^3 / 3``.

Every face of these solids is a plane, a cylinder or a cone, which OCCT integrates
exactly: measures are compared at 1e-12 relative.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, NameRole, Session

A: float = 3.0
B: float = 7.0
H: float = 5.0
ALPHA: float = math.radians(10.0)
RTOL: float = 1e-12
ATOL: float = 1e-12 * H

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, per kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def rectangle_session() -> tuple[Session, EntityId, list[EntityId]]:
    """A free rectangle a x b on z = 0, corner at the origin; its face and edge ids."""
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), A, B)
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))
    edges = [EntityId(int(i)) for i in s.entities(EntityKind.EDGE)]
    return s, face, edges


def solid_volume(s: Session) -> tuple[float, np.ndarray]:
    """Volume and centroid of the session's only solid."""
    (solid,) = (EntityId(int(i)) for i in s.entities(EntityKind.SOLID))
    m = s.mass_properties([solid])
    return float(m.measure[0]), np.asarray(m.centroid[0])


# ---- symmetric ----------------------------------------------------------------------- #


def test_a_symmetric_extrusion_runs_from_minus_to_plus_the_vector() -> None:
    s, _, _ = rectangle_session()
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    s.extrude([face], (0.0, 0.0, H), symmetric=True)

    volume, centroid = solid_volume(s)
    assert volume == pytest.approx(2.0 * A * B * H, rel=RTOL)
    np.testing.assert_allclose(centroid, (0.5 * A, 0.5 * B, 0.0), rtol=RTOL, atol=ATOL)
    table = s.bounding_boxes(EntityKind.SOLID)
    np.testing.assert_allclose(table.bbox[0], (0.0, 0.0, -H, A, B, H), atol=ATOL)


def test_a_symmetric_extrusion_carries_the_profile_ids_onto_its_start() -> None:
    s, face, edges = rectangle_session()

    delta = s.extrude([face], (0.0, 0.0, H), symmetric=True)

    assert s.is_alive(face)
    m = s.mass_properties([face])
    assert m.measure[0] == pytest.approx(A * B, rel=RTOL)
    np.testing.assert_allclose(
        m.centroid[0], (0.5 * A, 0.5 * B, -H), rtol=RTOL, atol=ATOL
    )
    for e in edges:
        assert s.is_alive(e)
        assert s.mass_properties([e]).centroid[0][2] == pytest.approx(-H, rel=RTOL)
    # Each wall is generated from its profile edge and stands on it: area l x 2h.
    for f in delta.created.tolist():
        fid = EntityId(f)
        if s.entity_kind(fid) is not EntityKind.FACE:
            continue
        origin = s.origin(fid)
        if origin.role is NameRole.GENERATED and origin.sources.size == 1:
            (edge,) = origin.sources.tolist()
            if edge in [int(e) for e in edges]:
                length = s.mass_properties([EntityId(edge)]).measure[0]
                area = s.mass_properties([fid]).measure[0]
                assert area == pytest.approx(length * 2.0 * H, rel=RTOL)


def test_a_symmetric_extrusion_of_an_edge_is_a_face_twice_as_tall() -> None:
    s = Session()
    s.add_line((0.0, 0.0, 0.0), (A, 0.0, 0.0))
    edge = EntityId(int(s.entities(EntityKind.EDGE)[0]))

    s.extrude([edge], (0.0, 0.0, H), symmetric=True)

    (face,) = (EntityId(int(i)) for i in s.entities(EntityKind.FACE))
    m = s.mass_properties([face])
    assert m.measure[0] == pytest.approx(A * 2.0 * H, rel=RTOL)
    np.testing.assert_allclose(m.centroid[0], (0.5 * A, 0.0, 0.0), rtol=RTOL, atol=ATOL)


# ---- draft --------------------------------------------------------------------------- #


def prismatoid_volume(a: float, b: float, h: float, alpha: float) -> float:
    """Volume of the rectangle a x b swept by h with the inward draft alpha."""
    d = h * math.tan(alpha)
    return h / 6.0 * (a * b + 4.0 * (a - d) * (b - d) + (a - 2.0 * d) * (b - 2.0 * d))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_positive_draft_narrows_the_solid_to_the_prismatoid(sign: float) -> None:
    """The sweep runs along the vector, with the normal or against it."""
    s, face, _ = rectangle_session()

    s.extrude([face], (0.0, 0.0, sign * H), draft_angle=ALPHA)

    volume, centroid = solid_volume(s)
    assert volume == pytest.approx(prismatoid_volume(A, B, H, ALPHA), rel=RTOL)
    assert centroid[0] == pytest.approx(0.5 * A, rel=RTOL)
    assert centroid[1] == pytest.approx(0.5 * B, rel=RTOL)
    assert sign * centroid[2] > 0.0
    types = s.entity_types(EntityKind.FACE)
    assert list(types.types) == ["Plane"] * 6


def test_a_drafted_wall_is_the_trapezoid_over_its_profile_edge() -> None:
    s, face, edges = rectangle_session()
    lengths = {int(e): s.mass_properties([e]).measure[0] for e in edges}
    d = H * math.tan(ALPHA)

    delta = s.extrude([face], (0.0, 0.0, H), draft_angle=ALPHA)

    walls = 0
    for f in delta.created.tolist():
        fid = EntityId(f)
        if s.entity_kind(fid) is not EntityKind.FACE:
            continue
        sources = s.origin(fid).sources.tolist()
        if len(sources) == 1 and sources[0] in lengths:
            walls += 1
            length = lengths[sources[0]]
            area = s.mass_properties([fid]).measure[0]
            assert area == pytest.approx((length - d) * H / math.cos(ALPHA), rel=RTOL)
    assert walls == 4


def test_a_drafted_extrusion_carries_the_profile_ids_onto_its_start() -> None:
    s, face, edges = rectangle_session()
    before = {int(e): s.mass_properties([e]).centroid[0] for e in edges}

    s.extrude([face], (0.0, 0.0, H), draft_angle=ALPHA)

    m = s.mass_properties([face])
    assert m.measure[0] == pytest.approx(A * B, rel=RTOL)
    np.testing.assert_allclose(
        m.centroid[0], (0.5 * A, 0.5 * B, 0.0), rtol=RTOL, atol=ATOL
    )
    for e in edges:
        np.testing.assert_allclose(
            s.mass_properties([e]).centroid[0], before[int(e)], rtol=RTOL, atol=ATOL
        )
    d = H * math.tan(ALPHA)
    far = [
        EntityId(int(i))
        for i, c in zip(*_face_centroids(s), strict=True)
        if abs(c[2] - H) <= ATOL
    ]
    assert len(far) == 1
    assert s.mass_properties(far).measure[0] == pytest.approx(
        (A - 2.0 * d) * (B - 2.0 * d), rel=RTOL
    )
    assert s.origin(far[0]).sources.tolist() == [int(face)]


def _face_centroids(s: Session) -> tuple[list[int], np.ndarray]:
    table = s.entity_table(EntityKind.FACE)
    return [int(i) for i in table.ids], table.centroid


def test_a_negative_draft_widens_the_solid_with_rounded_corners() -> None:
    s, face, _ = rectangle_session()
    k = math.tan(ALPHA)

    s.extrude([face], (0.0, 0.0, H), draft_angle=-ALPHA)

    volume, _ = solid_volume(s)
    want = A * B * H + (A + B) * k * H**2 + math.pi * k**2 * H**3 / 3.0
    assert volume == pytest.approx(want, rel=RTOL)
    types = sorted(s.entity_types(EntityKind.FACE).types)
    assert types.count("Plane") == 6
    assert len(types) == 10


def test_a_symmetric_draft_starts_at_minus_the_vector() -> None:
    """At 5 degrees over 2 h = 10 the profile narrows by 0.875 a side, less than a / 2."""
    s, face, _ = rectangle_session()
    alpha = math.radians(5.0)

    s.extrude([face], (0.0, 0.0, H), draft_angle=alpha, symmetric=True)

    volume, _ = solid_volume(s)
    assert volume == pytest.approx(prismatoid_volume(A, B, 2.0 * H, alpha), rel=RTOL)
    m = s.mass_properties([face])
    np.testing.assert_allclose(
        m.centroid[0], (0.5 * A, 0.5 * B, -H), rtol=RTOL, atol=ATOL
    )


def test_a_draft_that_closes_the_profile_is_refused() -> None:
    """At 10 degrees over 10 the sides move in by 1.76, past half the width 1.5.

    The long walls meet where the inward offset ``z tan(alpha)`` is ``a / 2``: at
    ``z = a / (2 tan(alpha)) = 8.50692``, short of the sweep length 10. OCCT trims the
    walls at their meeting line within the drafted solid's tolerance, 1e-5 (measured on
    OCCT 8.0.1: 8.506931), so the reported height is compared at that tolerance.
    """
    s, face, _ = rectangle_session()
    before = (s.brep(), id_space(s), s.op_count)
    ridge = A / (2.0 * math.tan(ALPHA))

    with pytest.raises(
        ps.PysmeshError,
        match=rf"extrude: at the draft angle {ALPHA:.6g} rad the walls meet "
        rf"(\S+) from the profile, short of the sweep length {2.0 * H:.6g}",
    ) as caught:
        s.extrude([face], (0.0, 0.0, 2.0 * H), draft_angle=ALPHA)

    reported = re.search(r"walls meet (\S+) from", str(caught.value))
    assert reported is not None
    assert float(reported.group(1)) == pytest.approx(ridge, abs=1e-5)
    assert (s.brep(), id_space(s), s.op_count) == before


def test_a_draft_on_a_wire_profile_is_refused_naming_its_kind() -> None:
    s = Session()
    s.add_polyline([(0.0, 0.0, 0.0), (A, 0.0, 0.0), (A, B, 0.0)])
    edges = [EntityId(int(i)) for i in s.entities(EntityKind.EDGE)]

    with pytest.raises(
        ps.PysmeshError,
        match=r"extrude: a draft angle applies to a planar face profile; the named "
        "entities belong to a WIRE body",
    ):
        s.extrude(edges, (0.0, 0.0, H), draft_angle=ALPHA)


def test_a_draft_on_a_curved_face_is_refused_naming_its_surface_type() -> None:
    s = Session()
    s.add_cylinder(2.0, H)
    params = s.surface_parameters(
        [EntityId(int(i)) for i in s.entities(EntityKind.FACE)]
    )
    side = EntityId(int(params.ids[list(params.types).index("Cylinder")]))
    before = {int(i) for i in s.entities(EntityKind.FACE)}
    s.extend_face(side, v_max=1.0)
    (curved,) = (
        EntityId(int(i)) for i in s.entities(EntityKind.FACE) if int(i) not in before
    )

    with pytest.raises(
        ps.PysmeshError,
        match=r"extrude: a draft angle applies to a planar face profile",
    ):
        s.extrude([curved], (0.0, 0.0, H), draft_angle=ALPHA)


def test_a_draft_off_the_normal_is_refused_with_the_angle() -> None:
    s, face, _ = rectangle_session()

    with pytest.raises(
        ps.PysmeshError,
        match=r"extrude: a draft angle applies to a sweep along the profile face's "
        r"normal; the vector is 0\.785398 rad off it",
    ):
        s.extrude([face], (H, 0.0, H), draft_angle=ALPHA)


@pytest.mark.parametrize("angle", [0.5 * math.pi, -2.0])
def test_a_draft_angle_outside_a_quarter_turn_is_refused(angle: float) -> None:
    s, face, _ = rectangle_session()

    with pytest.raises(
        ps.PysmeshError, match=r"extrude: the draft angle .* is not between"
    ):
        s.extrude([face], (0.0, 0.0, H), draft_angle=angle)


def test_a_refused_draft_leaves_the_session_as_it_was() -> None:
    s, face, _ = rectangle_session()
    before = (s.brep(), id_space(s), s.op_count, s.issued_id_count)

    with pytest.raises(ps.PysmeshError):
        s.extrude([face], (H, 0.0, H), draft_angle=ALPHA)

    assert (s.brep(), id_space(s), s.op_count, s.issued_id_count) == before


# ---- determinism --------------------------------------------------------------------- #


def _swept(draft: float, symmetric: bool) -> tuple[bytes, tuple[tuple[int, ...], ...]]:
    s, face, _ = rectangle_session()
    s.extrude([face], (0.0, 0.0, H), draft_angle=draft, symmetric=symmetric)
    return s.brep(), id_space(s)


@pytest.mark.parametrize(
    ("draft", "symmetric"), [(0.0, True), (ALPHA, False), (-ALPHA, True)]
)
def test_two_fresh_sessions_give_identical_bytes_and_ids(
    draft: float, symmetric: bool
) -> None:
    first = _swept(draft, symmetric)

    second = _swept(draft, symmetric)

    assert first == second
