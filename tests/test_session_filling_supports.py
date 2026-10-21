# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for make_filling's keyword argument support_faces.

The positive cases fill the top rim of a truncated cone: base radius 3 on z = 0, top
radius 1 on z = 1. The rim is one closed circle edge, shared by the top disc and the cone
face. The cone face's outward normal at the rim point ``(cos t, sin t, 1)`` is
``(cos t, sin t, 2) / sqrt(5)``: ``atan(1 / 2)`` = 0.4636 rad off the z axis.

* The rim lies on the plane z = 1, so a filling with a C0 constraint only is the flat
  disc, whose normal is the z axis.
* With the cone face as the support of the rim (``GeomAbs_G1``), the filling must be
  tangent to the cone along the rim: its normal there is the cone's normal.

``BRepOffsetAPI_MakeFilling``'s defaults set the tolerances: 1e-4 for the distance to a
boundary edge (Tol3d) and 1e-2 rad for the tangency (TolAng). The tests use those.

The refusal cases use the open side shell of a square frustum, sewn from four planar
faces, whose four top edges bound the unit square on z = 0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import EntityId, EntityKind, Session

SLOPE: float = 0.5
TOL_3D: float = 1e-4
TOL_ANGLE: float = 1e-2
RIM_SAMPLES: int = 12

# The frustum's top edge midpoints.
TOP_MIDPOINTS: tuple[tuple[float, float, float], ...] = (
    (0.5, 0.0, 0.0),
    (1.0, 0.5, 0.0),
    (0.5, 1.0, 0.0),
    (0.0, 0.5, 0.0),
)

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, per kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def new_ids(s: Session, kind: EntityKind, before: set[int]) -> list[EntityId]:
    """The ids of ``kind`` that are alive now and were not in ``before``."""
    return [EntityId(int(i)) for i in s.entities(kind) if int(i) not in before]


def edge_at(s: Session, midpoint: tuple[float, float, float]) -> EntityId:
    """The one live edge whose centroid is ``midpoint``."""
    table = s.entity_table(EntityKind.EDGE)
    hits = np.flatnonzero(
        np.all(np.isclose(table.centroid, midpoint, rtol=0.0, atol=1e-12), axis=1)
    )
    assert hits.size == 1
    return EntityId(int(table.ids[hits[0]]))


def faces_of_edge(s: Session, edge: EntityId) -> list[EntityId]:
    """The faces that contain ``edge``."""
    pairs = s.adjacency(EntityKind.EDGE, EntityKind.FACE)
    return [EntityId(int(f)) for f in pairs.related[pairs.ids == int(edge)]]


def frustum_session() -> tuple[Session, dict[EntityId, EntityId]]:
    """The sewn side shell; each top edge mapped to the side face that holds it."""
    s = Session()
    top = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    bottom = [(-1.0, -1.0), (2.0, -1.0), (2.0, 2.0), (-1.0, 2.0)]
    for k in range(4):
        (a, b), (c, d) = top[k], top[(k + 1) % 4]
        (e, f), (g, h) = bottom[(k + 1) % 4], bottom[k]
        points = [[a, b, 0.0], [c, d, 0.0], [e, f, -SLOPE], [g, h, -SLOPE]]
        before = {int(i) for i in s.entities(EntityKind.EDGE)}
        s.add_polyline(np.array(points, dtype=np.float64), closed=True)
        s.make_face(new_ids(s, EntityKind.EDGE, before))
    s.sew([EntityId(int(i)) for i in s.entities(EntityKind.FACE)])
    supports = {}
    for midpoint in TOP_MIDPOINTS:
        edge = edge_at(s, midpoint)
        (face,) = faces_of_edge(s, edge)
        supports[edge] = face
    return s, supports


def cone_session() -> tuple[Session, EntityId, EntityId]:
    """The truncated cone; its top rim and the cone face."""
    s = Session()
    s.add_cone(3.0, 1.0, 1.0)
    rim = edge_at(s, (0.0, 0.0, 1.0))
    faces = faces_of_edge(s, rim)
    (cone,) = [f for f in faces if s.mass_properties([f]).centroid[0][2] < 1.0 - 1e-9]
    return s, rim, cone


def rim_points() -> NDArray[np.float64]:
    """Twelve points on the rim, evenly spaced in angle."""
    t = np.linspace(0.0, 2.0 * math.pi, RIM_SAMPLES, endpoint=False)
    return np.stack([np.cos(t), np.sin(t), np.ones_like(t)], axis=1)


def cone_normals() -> NDArray[np.float64]:
    """The cone's outward unit normals at the rim points."""
    t = np.linspace(0.0, 2.0 * math.pi, RIM_SAMPLES, endpoint=False)
    return np.stack([np.cos(t), np.sin(t), np.full_like(t, 2.0)], axis=1) / math.sqrt(
        5.0
    )


def angles(normals: NDArray[np.float64], expected: NDArray[np.float64]) -> float:
    """The largest angle between matching rows, either way round."""
    cosines = np.clip(np.abs(np.sum(normals * expected, axis=1)), 0.0, 1.0)
    return float(np.max(np.arccos(cosines)))


def filled_rim(s: Session, rim: EntityId, cone: EntityId) -> EntityId:
    """Fill the rim with the cone face as its support; the new face's id."""
    faces = {int(i) for i in s.entities(EntityKind.FACE)}
    s.make_filling([rim], support_faces={rim: cone})
    (face,) = new_ids(s, EntityKind.FACE, faces)
    return face


# ---- analytic ---------------------------------------------------------------------- #


def test_the_cone_normal_at_the_rim_is_tilted_away_from_the_flat_filling() -> None:
    """The tangency check below is not met by the flat C0 filling."""
    s, _, cone = cone_session()

    uv = s.project_on_face(cone, rim_points()).uv
    normals = np.asarray(s.surface_at(cone, uv).normals)

    assert angles(normals, cone_normals()) < 1e-12
    flat = np.tile([0.0, 0.0, 1.0], (RIM_SAMPLES, 1))
    assert angles(normals, flat) == pytest.approx(math.atan(0.5), rel=1e-12)


def test_the_filling_is_tangent_to_the_support_along_the_rim() -> None:
    s, rim, cone = cone_session()

    face = filled_rim(s, rim, cone)

    uv = s.project_on_face(face, rim_points()).uv
    normals = np.asarray(s.surface_at(face, uv).normals)
    assert angles(normals, cone_normals()) <= TOL_ANGLE


def test_the_filling_passes_through_the_rim() -> None:
    s, rim, cone = cone_session()

    face = filled_rim(s, rim, cone)

    assert float(np.max(s.project_on_face(face, rim_points()).distance)) <= TOL_3D


# ---- history against geometry -------------------------------------------------------- #


def test_the_support_keeps_every_id_and_the_filling_is_new() -> None:
    """Nothing is consumed: every id of the cone keeps its measure and centroid."""
    s, rim, cone = cone_session()
    before = id_space(s)
    tables = {k: s.entity_table(k) for k in _KINDS}

    delta = s.make_filling([rim], support_faces={rim: cone})

    assert delta.deleted.size == 0
    for kind, table in tables.items():
        after = s.entity_table(kind)
        keep = np.isin(after.ids, table.ids)
        np.testing.assert_array_equal(after.ids[keep], table.ids)
        np.testing.assert_array_equal(after.measure[keep], table.measure)
        np.testing.assert_array_equal(after.centroid[keep], table.centroid)
    old = {i for ids in before for i in ids}
    new = {i for ids in id_space(s) for i in ids} - old
    assert sorted(delta.created.tolist()) == sorted(new)
    assert len(new_ids(s, EntityKind.FACE, old)) == 1
    assert len(new_ids(s, EntityKind.SOLID, old)) == 0


# ---- the default path ---------------------------------------------------------------- #


def free_square() -> tuple[Session, list[EntityId]]:
    """The unit square on z = 0 as four free edges."""
    s = Session()
    s.add_polyline(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        closed=True,
    )
    return s, [EntityId(int(i)) for i in s.entities(EntityKind.EDGE)]


def test_an_empty_support_mapping_is_the_default_filling() -> None:
    first, edges = free_square()
    first.make_filling(edges)
    second, edges = free_square()

    second.make_filling(edges, support_faces={})

    assert (second.brep(), id_space(second)) == (first.brep(), id_space(first))


# ---- determinism --------------------------------------------------------------------- #


def _filled() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
    s, rim, cone = cone_session()
    filled_rim(s, rim, cone)
    return s.brep(), id_space(s)


def test_two_fresh_sessions_give_identical_bytes_and_ids() -> None:
    first = _filled()

    second = _filled()

    assert first == second


# ---- cancel -------------------------------------------------------------------------- #


def test_a_cancel_at_the_first_callback_leaves_the_session_as_it_was() -> None:
    s, rim, cone = cone_session()
    before = (s.brep(), id_space(s), s.op_count, s.issued_id_count)

    with pytest.raises(ps.PysmeshCancelled):
        s.make_filling([rim], support_faces={rim: cone}, cancel=lambda: True)

    assert (s.brep(), id_space(s), s.op_count, s.issued_id_count) == before


# ---- refusals ------------------------------------------------------------------------ #


def assert_refused(s: Session, pattern: str, *args: object, **kwargs: object) -> None:
    """The call raises a PysmeshError matching ``pattern`` and leaves the session as is."""
    before = (s.brep(), id_space(s), s.op_count, s.issued_id_count)

    with pytest.raises(ps.PysmeshError, match=pattern):
        s.make_filling(*args, **kwargs)  # type: ignore[arg-type]

    assert (s.brep(), id_space(s), s.op_count, s.issued_id_count) == before


def test_a_support_key_that_is_not_a_boundary_edge_is_refused() -> None:
    s, supports = frustum_session()
    edges = list(supports)
    face = supports[edges[0]]
    named = ", ".join(str(int(e)) for e in edges[1:])

    assert_refused(
        s,
        rf"make_filling: support edge {int(edges[0])} is not one of the boundary edges "
        rf"\({named}\)",
        edges[1:],
        support_faces={edges[0]: face},
    )


def test_a_support_face_that_does_not_hold_its_edge_is_refused() -> None:
    s, supports = frustum_session()
    edges = list(supports)
    wrong = dict(supports)
    wrong[edges[0]] = supports[edges[2]]

    assert_refused(
        s,
        rf"make_filling: face {int(supports[edges[2]])} does not contain edge "
        rf"{int(edges[0])}",
        edges,
        support_faces=wrong,
    )


def test_a_support_that_is_not_a_face_is_refused_naming_its_kind() -> None:
    s, supports = frustum_session()
    edges = list(supports)
    wrong = dict(supports)
    wrong[edges[0]] = edges[1]

    assert_refused(
        s,
        rf"make_filling: entity {int(edges[1])} is a EDGE, not a FACE",
        edges,
        support_faces=wrong,
    )


def test_a_dead_support_face_is_refused() -> None:
    s, supports = frustum_session()
    s.add_rectangle((5.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0)
    other = EntityId(int(s.entities(EntityKind.FACE)[-1]))
    s.remove([other])
    edges = list(supports)
    wrong = dict(supports)
    wrong[edges[0]] = other

    assert_refused(
        s,
        rf"make_filling: entity {int(other)} is dead",
        edges,
        support_faces=wrong,
    )


def test_an_unsupported_edge_of_a_face_is_refused_as_before() -> None:
    """Edges without a support are consumed, so they must be free edges or wires."""
    s, supports = frustum_session()
    edges = list(supports)
    partial = {e: supports[e] for e in edges[1:]}

    assert_refused(
        s,
        r"make_filling: the named edges belong to a SHELL body",
        edges,
        support_faces=partial,
    )


def test_edges_of_two_bodies_that_only_touch_are_refused_naming_the_vertex() -> None:
    """A side's top edge and a free polyline meet at coincident, unshared vertices.

    Measured on OCCT 8.0.1: BRepOffsetAPI_MakeFilling leaves that wire open and the face
    invalid (BRepCheck_NotClosed), with or without the support.
    """
    s, supports = frustum_session()
    e1 = edge_at(s, (0.5, 0.0, 0.0))
    before = {int(i) for i in s.entities(EntityKind.EDGE)}
    s.add_polyline(
        np.array(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    )
    free = new_ids(s, EntityKind.EDGE, before)
    named = ", ".join(str(int(e)) for e in [e1, *free])

    assert_refused(
        s,
        rf"make_filling: vertex \d+ of edge {int(e1)} is an end of 1 of the boundary "
        rf"edges, not 2; the boundary edges \({named}\) do not meet at shared vertices",
        [e1, *free],
        support_faces={e1: supports[e1]},
    )
