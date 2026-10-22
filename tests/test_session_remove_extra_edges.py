# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-22

"""Gates for Session.remove_extra_edges: GEOM's RemoveExtraEdges with every id carried.

The comparison set is the one GEOM's BlockFix was compared on before (a box fused from two
boxes, a cylinder fused from two cylinders, a box cut and fused back, the quasi-planar
B-spline pair, a revolved annulus), built in the session, with a box edge divided in two
and a half-ball wedge. The vendored ``BlockFix_BlockFixAPI`` (common_geometry_lib V9_16_0)
ran on the session's own BREP of each, with and without GEOM's driver copy, and gave these
(solids, faces, edges, vertices):

=========================  ================  =================  ================
shape                      input             OptimumNbFaces -1  OptimumNbFaces 0
=========================  ================  =================  ================
fused boxes                1, 10, 20, 12     unchanged          1, 6, 12, 8
fused cylinders            1, 4, 5, 3        unchanged          1, 3, 3, 2
box cut and fused back     1, 10, 20, 12     unchanged          1, 6, 12, 8
quasi-planar pair          0, 2, 7, 6        unchanged          0, 1, 4, 4
revolved annulus           1, 4, 6, 4        unchanged          unchanged
divided box edge           1, 6, 13, 9       1, 6, 12, 8        1, 6, 12, 8
half-ball wedge            1, 3, 5, 2        unchanged          1, 2, 4, 2
=========================  ================  =================  ================

``union_faces`` is GEOM's ``doUnionFaces``: OptimumNbFaces 0 when true, -1 when false. An
unchanged result is a refusal here: an operation that changes nothing is one. The measures
of each result are closed forms (a box, a cylinder, a half ball), compared at 1e-9 relative:
the stateless measures run at GProp precision 1e-6, and the quasi-planar face is a B-spline
within 4.2e-8 of its plane. Ids are checked against the analytic box at 1e-12 relative.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import EntityId, EntityKind, Session

EXACT_RTOL: float = 1e-12
MEASURE_RTOL: float = 1e-9
FIXTURE: Path = Path(__file__).resolve().parent / "fixtures" / "quasi_planar_pair.brep"

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)

Counts = tuple[int, int, int, int]
Measures = tuple[float, float, float]


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, by kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def state(s: Session) -> tuple[bytes, tuple[tuple[int, ...], ...], int, int]:
    """The bytes, the live ids and the two counters."""
    return s.brep(), id_space(s), s.op_count, s.issued_id_count


def counts(s: Session) -> Counts:
    """(solids, faces, edges, vertices) of the model, from the serialised shape."""
    shape = ps.load_brep(s.brep())
    return (
        len(shape.solids()),
        len(shape.faces()),
        len(shape.edges()),
        len(shape.vertices()),
    )


def measures(s: Session) -> Measures:
    """The sums of the volumes, the areas and the lengths, each shape once."""
    shape = ps.load_brep(s.brep())
    return (
        sum(x.volume for x in shape.solids()),
        sum(f.area for f in shape.faces()),
        sum(e.length for e in shape.edges()),
    )


def ids_of(s: Session, kind: EntityKind) -> list[EntityId]:
    """The live ids of one kind."""
    return [EntityId(int(i)) for i in s.entities(kind)]


def bodies(s: Session) -> list[EntityId]:
    """The solids, or the faces of a model without one."""
    return [EntityId(int(i)) for i in s.entities(EntityKind.SOLID)] or [
        EntityId(int(i)) for i in s.entities(EntityKind.FACE)
    ]


def fused_boxes() -> Session:
    """A 6 x 7 x 11 box fused from two 3 x 7 x 11 boxes: its faces split at x = 3."""
    s = Session()
    s.add_box(3.0, 7.0, 11.0)
    s.add_box(3.0, 7.0, 11.0, origin=(3.0, 0.0, 0.0))
    a, b = s.entities(EntityKind.SOLID)
    s.fuse([a], [b])
    return s


def fused_cylinders() -> Session:
    """A cylinder of radius 2 and height 10 fused from two: split at z = 5."""
    s = Session()
    s.add_cylinder(2.0, 5.0)
    s.add_cylinder(2.0, 5.0, origin=(0.0, 0.0, 5.0))
    a, b = s.entities(EntityKind.SOLID)
    s.fuse([a], [b])
    return s


def box_cut_fused_back() -> Session:
    """A 3 x 7 x 11 box cut by the plane x = 1.5 and fused back."""
    s = Session()
    s.add_box(3.0, 7.0, 11.0)
    s.split_by_plane(ids_of(s, EntityKind.SOLID), (1.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    a, b = s.entities(EntityKind.SOLID)
    s.fuse([a], [b])
    return s


def quasi_planar_pair() -> Session:
    """Two B-spline faces within 4.2e-8 of z = 0, over [0, 6] x [0, 7], sewn at x = 3."""
    s = Session()
    s.add_brep(FIXTURE.read_bytes())
    return s


def revolved_annulus() -> Session:
    """A rectangle turned a full turn about z: a thick tube."""
    s = Session()
    s.add_rectangle((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), 1.0, 3.0)
    s.revolve(ids_of(s, EntityKind.FACE), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    return s


def divided_box() -> Session:
    """A 2 x 3 x 4 box whose edge along z at the origin is divided at a quarter."""
    s = Session()
    s.add_box(2.0, 3.0, 4.0)
    edges = ids_of(s, EntityKind.EDGE)
    m = s.mass_properties(edges)
    (along_z,) = (
        e
        for e, length, c in zip(edges, m.measure, m.centroid, strict=True)
        if math.isclose(float(length), 4.0) and np.allclose(c[:2], 0.0)
    )
    s.divide_edge(along_z, [0.25])
    return s


def half_ball() -> Session:
    """Half a ball of radius 2: the sphere turned half a turn about z."""
    s = Session()
    s.add_sphere(2.0, angle_rad=math.pi)
    return s


def quarter_ball() -> Session:
    """A quarter of a ball of radius 2: the sphere turned a quarter turn about z."""
    s = Session()
    s.add_sphere(2.0, angle_rad=math.pi / 2)
    return s


@pytest.mark.parametrize(
    ("build", "union", "expected_counts", "expected"),
    [
        (fused_boxes, True, (1, 6, 12, 8), (462.0, 370.0, 96.0)),
        (
            fused_cylinders,
            True,
            (1, 3, 3, 2),
            (40.0 * math.pi, 48.0 * math.pi, 8.0 * math.pi + 10.0),
        ),
        (box_cut_fused_back, True, (1, 6, 12, 8), (231.0, 262.0, 84.0)),
        (quasi_planar_pair, True, (0, 1, 4, 4), (0.0, 42.0, 26.0)),
        (divided_box, False, (1, 6, 12, 8), (24.0, 52.0, 36.0)),
        (divided_box, True, (1, 6, 12, 8), (24.0, 52.0, 36.0)),
        (
            half_ball,
            True,
            (1, 2, 4, 2),
            (16.0 * math.pi / 3.0, 12.0 * math.pi, 4.0 * math.pi),
        ),
    ],
)
def test_the_comparison_set_gives_geoms_counts_and_the_closed_form_measures(
    build: Callable[[], Session],
    union: bool,
    expected_counts: Counts,
    expected: Measures,
) -> None:
    s = build()

    delta = s.remove_extra_edges(bodies(s), union_faces=union)

    assert counts(s) == expected_counts
    np.testing.assert_allclose(measures(s), expected, rtol=MEASURE_RTOL, atol=1e-12)
    assert delta.valid is True
    assert len(delta.created) == 0


@pytest.mark.parametrize(
    ("build", "union"),
    [
        (fused_boxes, False),
        (fused_cylinders, False),
        (box_cut_fused_back, False),
        (quasi_planar_pair, False),
        (revolved_annulus, False),
        (revolved_annulus, True),
        (half_ball, False),
    ],
)
def test_a_result_with_the_inputs_counts_is_refused(
    build: Callable[[], Session], union: bool
) -> None:
    s = build()
    named = bodies(s)
    before = state(s)
    n = counts(s)

    with pytest.raises(ps.PysmeshError) as err:
        s.remove_extra_edges(named, union_faces=union)

    text = str(err.value)
    assert text.startswith("Session.remove_extra_edges: the bodies of the entities ")
    assert str(int(named[0])) in text
    assert f"{n[1]} faces, {n[2]} edges, {n[3]} vertices" in text
    assert "nothing is committed" in text
    assert state(s) == before


# ---- identity -------------------------------------------------------------------------- #


def planes_and_lines(s: Session, kind: EntityKind) -> dict[int, NDArray[np.float64]]:
    """Each id's bounding box: a face of the box is flat along one axis, an edge along two."""
    table = s.bounding_boxes(kind)
    return {
        int(i): np.asarray(b, dtype=np.float64)
        for i, b in zip(table.ids, table.bbox, strict=True)
    }


def on_plane_x(bbox: NDArray[np.float64], x: float) -> bool:
    """True when the box lies in the plane at `x`."""
    return math.isclose(bbox[0], x, abs_tol=1e-9) and math.isclose(
        bbox[3], x, abs_tol=1e-9
    )


def box_face_on(
    bbox: NDArray[np.float64], lo: NDArray[np.float64], hi: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """The area and centroid of the face of the box [lo, hi] on the plane of `bbox`."""
    flat = int(np.argmin(bbox[3:] - bbox[:3]))
    size = hi - lo
    centre = 0.5 * (lo + hi)
    centre[flat] = bbox[flat]
    others = [k for k in range(3) if k != flat]
    return float(size[others[0]] * size[others[1]]), centre


def box_edge_on(
    bbox: NDArray[np.float64], lo: NDArray[np.float64], hi: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """The length and centroid of the edge of the box [lo, hi] on the line of `bbox`."""
    along = int(np.argmax(bbox[3:] - bbox[:3]))
    centre = bbox[:3].copy()
    centre[along] = 0.5 * (lo[along] + hi[along])
    return float(hi[along] - lo[along]), centre


@pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (10.0, -4.0, 2.5)])
def test_every_id_after_uniting_the_fused_boxes_denotes_its_analytic_geometry(
    offset: tuple[float, float, float],
) -> None:
    """Faces united keep all their ids; the split edges join; the seam at x = 3 dies.

    With an offset the bodies are placed (a location, not new geometry), which GEOM's
    ShapeUpgrade_RemoveLocations step takes off.
    """
    s = fused_boxes()
    if offset != (0.0, 0.0, 0.0):
        s.translate(offset)
    lo = np.array(offset, dtype=np.float64)
    hi = lo + np.array([6.0, 7.0, 11.0], dtype=np.float64)
    seam_x = lo[0] + 3.0
    faces = planes_and_lines(s, EntityKind.FACE)
    edges = planes_and_lines(s, EntityKind.EDGE)
    vertices = planes_and_lines(s, EntityKind.VERTEX)

    delta = s.remove_extra_edges(bodies(s), union_faces=True)

    live = {k: {int(i) for i in s.entities(k)} for k in _KINDS}
    assert live[EntityKind.FACE] == set(faces)
    for i, bbox in faces.items():
        area, centre = box_face_on(bbox, lo, hi)
        m = s.mass_properties([EntityId(i)])
        np.testing.assert_allclose(m.measure[0], area, rtol=EXACT_RTOL)
        np.testing.assert_allclose(m.centroid[0], centre, rtol=EXACT_RTOL, atol=1e-12)
    on_seam = {i for i, b in edges.items() if on_plane_x(b, seam_x)}
    assert live[EntityKind.EDGE] == set(edges) - on_seam
    for i in live[EntityKind.EDGE]:
        length, centre = box_edge_on(edges[i], lo, hi)
        m = s.mass_properties([EntityId(i)])
        np.testing.assert_allclose(m.measure[0], length, rtol=EXACT_RTOL)
        np.testing.assert_allclose(m.centroid[0], centre, rtol=EXACT_RTOL, atol=1e-12)
    seam_vertices = {i for i, b in vertices.items() if on_plane_x(b, seam_x)}
    assert live[EntityKind.VERTEX] == set(vertices) - seam_vertices
    table = s.entity_table(EntityKind.VERTEX)
    for i, c in zip(table.ids, table.centroid, strict=True):
        np.testing.assert_allclose(c, vertices[int(i)][:3], rtol=EXACT_RTOL, atol=1e-12)
    assert len(delta.created) == 0
    assert {int(i) for i in delta.deleted} == on_seam | seam_vertices


def test_the_pieces_of_a_divided_edge_join_and_both_ids_denote_the_whole_edge() -> None:
    s = divided_box()
    edges = planes_and_lines(s, EntityKind.EDGE)
    vertices = planes_and_lines(s, EntityKind.VERTEX)
    pieces = [
        i for i, b in edges.items() if np.allclose(b[[0, 1, 3, 4]], 0.0, atol=1e-9)
    ]
    (middle,) = (i for i, b in vertices.items() if np.allclose(b[:3], (0.0, 0.0, 1.0)))

    s.remove_extra_edges(bodies(s))

    assert len(pieces) == 2
    m = s.mass_properties([EntityId(i) for i in pieces])
    np.testing.assert_allclose(m.measure, [4.0, 4.0], rtol=EXACT_RTOL)
    np.testing.assert_allclose(m.centroid, [[0.0, 0.0, 2.0]] * 2, atol=1e-12)
    assert not s.is_alive(EntityId(middle))
    after = planes_and_lines(s, EntityKind.EDGE)
    assert set(after) == set(edges)
    for i in set(edges) - set(pieces):
        np.testing.assert_allclose(after[i], edges[i], atol=1e-12)


def test_the_half_disks_of_a_half_ball_unite_into_one_disk() -> None:
    """The two planar faces of the wedge lie in the plane y = 0: one disk of area 4 pi."""
    s = half_ball()
    types = s.entity_types(EntityKind.FACE)
    planes = [
        int(i) for i, t in zip(types.ids, types.types, strict=True) if t == "Plane"
    ]
    (sphere,) = (
        int(i) for i, t in zip(types.ids, types.types, strict=True) if t == "Sphere"
    )

    s.remove_extra_edges(bodies(s), union_faces=True)

    m = s.mass_properties([EntityId(i) for i in planes])
    np.testing.assert_allclose(m.measure, [4.0 * math.pi] * 2, rtol=EXACT_RTOL)
    np.testing.assert_allclose(m.centroid, [[0.0, 0.0, 0.0]] * 2, atol=1e-12)
    m = s.mass_properties([EntityId(sphere)])
    np.testing.assert_allclose(m.measure[0], 8.0 * math.pi, rtol=EXACT_RTOL)
    np.testing.assert_allclose(m.centroid[0], (0.0, 1.0, 0.0), atol=1e-12)


def test_a_turned_sphere_face_that_changes_its_area_is_refused() -> None:
    """pySMESH's check, not GEOM's: GEOM's rotation of a quarter-ball face changes it.

    The quarter ball's spherical face reaches both poles. GEOM's RotateSphereSpace puts it
    on a turned sphere as a face 3.8e-4 smaller (by GProp) whose mesh reaches 0.033 outside
    the wedge, though BRepCheck accepts the solid; the vendored class gives the same shape.
    """
    s = quarter_ball()
    before = state(s)

    with pytest.raises(
        ps.PysmeshError, match="RotateSphereSpace put the sphere face"
    ) as err:
        s.remove_extra_edges(bodies(s))

    assert "the bodies of the entities 1," in str(err.value)
    assert "nothing is committed" in str(err.value)
    assert state(s) == before


# ---- determinism, progress, cancel, refusals ------------------------------------------ #


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def built() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
        s = fused_boxes()
        s.remove_extra_edges(bodies(s), union_faces=True)
        return s.brep(), id_space(s)

    first = built()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(built())

    assert all(other == first for other in others)


def test_progress_reaches_one_and_a_cancel_leaves_the_session_as_it_was() -> None:
    s = fused_boxes()
    seen: list[float] = []
    t = fused_boxes()
    before = state(t)

    s.remove_extra_edges(bodies(s), union_faces=True, progress=seen.append)
    with pytest.raises(ps.PysmeshCancelled):
        t.remove_extra_edges(bodies(t), union_faces=True, cancel=lambda: True)

    assert seen and seen[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in itertools.pairwise(seen))
    assert state(t) == before


def test_remove_extra_edges_refuses_no_entity_and_a_dead_id() -> None:
    s = fused_boxes()
    before = state(s)

    with pytest.raises(ps.PysmeshError, match="Session.remove_extra_edges: no entity"):
        s.remove_extra_edges([])
    with pytest.raises(ps.PysmeshError, match="remove_extra_edges"):
        s.remove_extra_edges([EntityId(10_000)])

    assert state(s) == before
