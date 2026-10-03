# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""Gates for the session's repair surface: healing, sewing, defeaturing, imprinting, removal.

Three claims are under test, and each is asserted against something the registry did not
produce:

* **Every repair reports a verdict.** Each operation returns a full history delta carrying
  OCCT's ``BRepCheck_Analyzer`` result, and the healing family commits — and reports — a
  shape it could not fully repair rather than refusing it. Both directions are covered: a
  clean model heals to ``valid=True``, a degenerate one to ``valid=False``, and neither
  raises.
* **Scope is a guarantee.** Healing a subset leaves every entity outside the scope with the
  identical shape it had before, asserted two ways: the delta names none of them, which is a
  statement about shape identity, and their geometry read back through the stateless API is
  *exactly* — not approximately — unchanged.
* **Defeaturing removes exactly what was named, and reports the survivors.** Removing a hole
  restores the closed-form volume of the unbored solid, kills the hole's face ids and leaves
  every other id alive. An incomplete feature — which OCCT declines while still reporting
  success — fails loud naming the faces that did not go away.

Model volumes and counts are read back through ``pysmesh.load_brep`` on the session's own
BREP, never summed from the session's tables: after a merge two ids denote one shape, so the
tables are not the model.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, GlueMode, ResolutionStatus, Session

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0
BOX_VOLUME: float = BOX_DX * BOX_DY * BOX_DZ

# A through hole small enough to be a defeaturing target on this box.
HOLE_RADIUS: float = 0.5
HOLE_VOLUME: float = math.pi * HOLE_RADIUS**2 * BOX_DZ

EXACT_RTOL: float = 1e-9
CURVED_RTOL: float = 1e-6

_ALL_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


# ---- independent oracles ------------------------------------------------------------- #


def model_volume(session: Session) -> float:
    """Total solid volume of the model, read back through the stateless API."""
    return float(sum(s.volume for s in ps.load_brep(session.brep()).solids()))


def model_counts(session: Session) -> tuple[int, int, int, int]:
    """(solids, faces, edges, vertices) of the model, from the serialised shape."""
    shape = ps.load_brep(session.brep())
    return (
        len(shape.solids()),
        len(shape.faces()),
        len(shape.edges()),
        len(shape.vertices()),
    )


def face_geometry(session: Session) -> dict[tuple[float, ...], float]:
    """Exact centroid -> area for every face of the model, from the serialised shape.

    Deliberately unrounded: the scoping gate wants bit-identical geometry, so any rounding
    here would hide exactly the difference it is looking for.
    """
    return {
        tuple(float(c) for c in info.centroid): info.area
        for info in ps.load_brep(session.brep()).faces()
    }


def live_ids(session: Session) -> set[int]:
    """Every live entity id of every kind."""
    return {int(i) for kind in _ALL_KINDS for i in session.entities(kind)}


def ids_of(session: Session, kind: EntityKind) -> list[EntityId]:
    """Live entity ids of one kind, as the typed id."""
    return [EntityId(int(i)) for i in session.entities(kind)]


def body_entities(session: Session, solid: EntityId) -> set[int]:
    """Every entity of one solid, including the solid itself.

    Walked through the adjacency query rather than the registry's internals, so the set is
    built the same way a consumer would build it.
    """
    out = {int(solid)}
    for kind in (EntityKind.FACE, EntityKind.EDGE, EntityKind.VERTEX):
        pairs = session.adjacency(EntityKind.SOLID, kind)
        out |= {int(i) for i in pairs.related[pairs.ids == int(solid)]}
    return out


# ---- fixtures ------------------------------------------------------------------------ #


@pytest.fixture
def two_boxes() -> Session:
    """Two well-separated 3 x 7 x 11 boxes, sharing nothing."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(20.0, 0.0, 0.0))
    return s


@pytest.fixture
def bored_box() -> Session:
    """A 3 x 7 x 11 box with a small through hole on its axis, along z."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_cylinder(
        HOLE_RADIUS, BOX_DZ + 2.0, origin=(BOX_DX / 2.0, BOX_DY / 2.0, -1.0)
    )
    solid, tool = ids_of(s, EntityKind.SOLID)
    s.cut([solid], [tool])
    return s


@pytest.fixture
def fused_boxes() -> Session:
    """Two 3 x 7 x 11 boxes fused across a shared face.

    The fuse leaves the four coplanar face pairs across the seam unmerged, so the result has
    10 faces for a shape that is geometrically a plain 6 x 7 x 11 block. That is the model
    same-domain merging exists to clean up.
    """
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(BOX_DX, 0.0, 0.0))
    a, b = ids_of(s, EntityKind.SOLID)
    s.fuse([a], [b])
    return s


def hole_face(session: Session) -> EntityId:
    """The cylindrical face of the bored box's hole."""
    table = session.entity_types(EntityKind.FACE)
    cylindrical = [
        EntityId(int(i)) for i, t in zip(table.ids, table.types) if t == "Cylinder"
    ]
    assert len(cylindrical) == 1
    return cylindrical[0]


# =============================================================== Healing and its verdict ==


def test_healing_a_clean_model_changes_no_entity_and_reports_valid() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    before = live_ids(s)

    delta = s.heal()

    assert delta.valid is True
    assert live_ids(s) == before
    assert delta.created.size == 0
    assert delta.deleted.size == 0
    assert delta.modified.size == 0


def test_healing_commits_a_shape_it_cannot_repair_and_says_so() -> None:
    """The whole reason the healing family reports instead of raising.

    A torus of radius 1e-12 is degenerate: OCCT builds it and ``BRepCheck_Analyzer`` rejects
    it. A strict session refuses that shape outright, so the fixture is built unvalidated.
    Healing it must then *commit* — an improvement the caller can keep — while stating that
    the result is still invalid.
    """
    with pytest.raises(ps.PysmeshError, match="invalid shape"):
        Session().add_torus(1e-12, 1e-13)
    s = Session(validate=False)
    s.add_torus(1e-12, 1e-13)

    delta = s.heal()

    assert delta.valid is False
    assert s.op_count == 2


def test_healing_one_body_leaves_every_entity_of_the_other_identical(
    two_boxes: Session,
) -> None:
    """Gate: heal on a scoped subset provably leaves entities outside the scope untouched.

    Two independent statements. The delta names no out-of-scope id, which is a claim about
    *shape identity* — the registry marks an entity modified exactly when its shapes are no
    longer the same shapes. And the geometry read back through the stateless API is bit-for-
    bit what it was, which shares no state with the registry.
    """
    first, second = ids_of(two_boxes, EntityKind.SOLID)
    kept = body_entities(two_boxes, second)
    geometry_before = face_geometry(two_boxes)

    delta = two_boxes.heal([first])

    touched = set(delta.modified) | set(delta.deleted) | set(delta.created)
    assert kept, "the fixture must have out-of-scope faces to protect"
    assert touched.isdisjoint(kept)
    assert face_geometry(two_boxes) == geometry_before


def test_healing_rejects_a_max_tolerance_below_the_minimum() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    with pytest.raises(ps.PysmeshError, match="max_tolerance"):
        s.heal(max_tolerance=1e-9, min_tolerance=1e-3)


def test_healing_rejects_a_non_positive_precision() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    with pytest.raises(ps.PysmeshError, match="precision"):
        s.heal(precision=0.0)


def test_healing_an_empty_selection_raises_rather_than_healing_everything() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    with pytest.raises(ps.PysmeshError, match="pass None"):
        s.heal([])


# ============================================================================== Sewing ==


def test_sewing_two_abutting_faces_shares_their_common_edge() -> None:
    """The repair a surface import needs: faces that meet but share no topology."""
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), BOX_DX, BOX_DY)
    s.add_rectangle((BOX_DX, 0.0, 0.0), (0.0, 0.0, 1.0), BOX_DX, BOX_DY)
    assert model_counts(s)[1:] == (2, 8, 8)

    delta = s.sew(ids_of(s, EntityKind.FACE))

    assert delta.valid is True
    assert model_counts(s)[1:] == (2, 7, 6)
    assert delta.merged.size > 0
    assert delta.deleted.size == 0


def test_sewing_a_closed_shell_can_close_it_into_a_solid() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    delta = s.sew(ids_of(s, EntityKind.SOLID), make_solid=True)

    assert delta.valid is True
    assert model_volume(s) == pytest.approx(BOX_VOLUME, rel=EXACT_RTOL)
    assert model_counts(s)[0] == 1


def test_sewing_refuses_to_fake_a_solid_from_an_open_shell() -> None:
    """An open shell bounds no volume, so ``make_solid`` raises rather than invent one.

    Two abutting rectangles sew into one open shell with 6 free edges. Before report A8
    the shell was committed with no signal; the caller asked for a solid and got none.
    """
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), BOX_DX, BOX_DY)
    s.add_rectangle((BOX_DX, 0.0, 0.0), (0.0, 0.0, 1.0), BOX_DX, BOX_DY)
    brep = s.brep()

    with pytest.raises(ps.PysmeshError, match="6 free boundary edge"):
        s.sew(ids_of(s, EntityKind.FACE), make_solid=True)

    assert s.brep() == brep


def test_sewing_rejects_a_non_positive_tolerance() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    with pytest.raises(ps.PysmeshError, match="tolerance"):
        s.sew(ids_of(s, EntityKind.SOLID), tolerance=0.0)


def test_sewing_an_empty_selection_raises() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    with pytest.raises(ps.PysmeshError, match="at least one"):
        s.sew([])


# ------------------------------------ Sewing into solids: the interior is inside --- #

# The 4.2.1 defect's fixture: an open tube about +z, and two planar caps copied from
# its rims.
TUBE_BASE: tuple[float, float, float] = (0.37, -2.9, 1.3)
TUBE_RADIUS: float = 1.3
TUBE_HEIGHT: float = 7.0
TUBE_INSIDE: tuple[float, float, float] = (0.5, -2.7, 4.0)
TUBE_OUTSIDE: tuple[float, float, float] = (3.0, 3.0, 3.0)
# A second tube, apart from the first.
APART_BASE: tuple[float, float, float] = (5.3, 1.1, -0.7)
APART_RADIUS: float = 0.7
APART_HEIGHT: float = 4.1
SEW_TOL: float = 1e-6


def cylinder_volume(radius: float, height: float) -> float:
    return math.pi * radius**2 * height


def loose_tube(
    session: Session, base: tuple[float, float, float], radius: float, height: float
) -> list[EntityId]:
    """An open cylindrical face about +z and two planar caps, sharing no topology.

    The wall is a circle extruded by ``height``. Each cap is a face made on a copy of one of
    the wall's rims, so the three faces meet only geometrically, as after a surface import.
    Returns [wall, cap at the base, cap at the top].
    """
    circle = created(
        session, session.add_circle(base, (0.0, 0.0, 1.0), radius), EntityKind.EDGE
    )
    wall = created(
        session, session.extrude(circle, (0.0, 0.0, height)), EntityKind.FACE
    )
    pairs = session.adjacency(EntityKind.FACE, EntityKind.EDGE)
    types = session.entity_types(EntityKind.EDGE)
    circles = {int(i) for i, t in zip(types.ids, types.types) if t == "Circle"}
    edges = pairs.related[pairs.ids == int(wall[0])]
    rims = sorted(
        (EntityId(int(e)) for e in edges if int(e) in circles),
        key=lambda e: float(session.mass_properties([e]).centroid[0][2]),
    )
    caps = []
    for rim in rims:
        copy = created(session, session.extract_edges([rim]), EntityKind.EDGE)
        caps.extend(created(session, session.make_face(copy), EntityKind.FACE))
    return [wall[0], *caps]


def loose_box_faces(
    session: Session,
    size: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> list[EntityId]:
    """The six faces of a box as six unsewn planar faces, the box itself removed."""
    box = created(session, session.add_box(*size, origin), EntityKind.SOLID)[0]
    own = session.adjacency(EntityKind.SOLID, EntityKind.FACE)
    pairs = session.adjacency(EntityKind.FACE, EntityKind.EDGE)
    faces = []
    for face in own.related[own.ids == int(box)]:
        edges = [EntityId(int(e)) for e in pairs.related[pairs.ids == int(face)]]
        copy = created(session, session.extract_edges(edges), EntityKind.EDGE)
        faces.extend(created(session, session.make_face(copy), EntityKind.FACE))
    session.remove([box])
    return faces


def solids_created(session: Session, delta: ps.HistoryDelta) -> list[EntityId]:
    return created(session, delta, EntityKind.SOLID)


def adaptive_volumes(session: Session, solids: list[EntityId]) -> list[float]:
    return [float(v) for v in session.mass_properties(solids, precision=1e-12).measure]


@pytest.mark.parametrize("caps_first", [False, True])
def test_sewing_a_tube_and_its_caps_commits_the_solid_with_its_interior_inside(
    caps_first: bool,
) -> None:
    """The 4.2.1 defect: this sew committed -37.165041091967, with every point test inverted.

    BRepCheck_Analyzer accepts both orientations, so ``valid`` was True and the caller had no
    signal. Naming the caps first changed nothing.
    """
    s = Session()
    wall, bottom, top = loose_tube(s, TUBE_BASE, TUBE_RADIUS, TUBE_HEIGHT)
    named = [bottom, top, wall] if caps_first else [wall, bottom, top]

    delta = s.sew(named, tolerance=SEW_TOL, make_solid=True)

    solid = solids_created(s, delta)
    assert len(solid) == 1
    assert delta.valid is True
    assert adaptive_volumes(s, solid) == pytest.approx(
        [cylinder_volume(TUBE_RADIUS, TUBE_HEIGHT)], rel=1e-12
    )
    assert model_volume(s) == pytest.approx(
        cylinder_volume(TUBE_RADIUS, TUBE_HEIGHT), rel=1e-12
    )
    assert s.contains(solid, [TUBE_INSIDE, TUBE_OUTSIDE]).tolist() == [[True, False]]
    assert all(s.is_alive(face) for face in (wall, bottom, top))
    assert delta.deleted.size == 0


def test_sewing_six_loose_box_faces_still_commits_the_box() -> None:
    """The case 4.2.1 already got right keeps its answer."""
    s = Session()
    faces = loose_box_faces(s, (BOX_DX, BOX_DY, BOX_DZ), (0.0, 0.0, 0.0))

    delta = s.sew(faces, tolerance=SEW_TOL, make_solid=True)

    solid = solids_created(s, delta)
    assert len(solid) == 1
    assert adaptive_volumes(s, solid) == pytest.approx([BOX_VOLUME], rel=1e-12)
    assert all(s.is_alive(face) for face in faces)


def test_sewing_a_thin_sheet_still_commits_it() -> None:
    """A sheet 0.001 thick encloses less than the first, coarse integral can settle.

    Its volume, 0.021, is below that stage's error bound, 1e-3 x its diagonal x its area / 3
    = 0.107, so the check integrates again at the precision derived from the tolerance. The
    sheet is a real solid, and it is committed.
    """
    s = Session()
    faces = loose_box_faces(s, (BOX_DX, BOX_DY, 0.001), (0.0, 0.0, 0.0))

    delta = s.sew(faces, tolerance=SEW_TOL, make_solid=True)

    assert adaptive_volumes(s, solids_created(s, delta)) == pytest.approx(
        [BOX_DX * BOX_DY * 0.001], rel=1e-12
    )


def test_sewing_two_tubes_apart_makes_two_solids() -> None:
    """4.2.1 put both shells into one solid, two outer shells, at -43.4765 = -(V1 + V2)."""
    s = Session()
    first = loose_tube(s, TUBE_BASE, TUBE_RADIUS, TUBE_HEIGHT)
    second = loose_tube(s, APART_BASE, APART_RADIUS, APART_HEIGHT)

    delta = s.sew([*first, *second], tolerance=SEW_TOL, make_solid=True)

    solids = solids_created(s, delta)
    assert adaptive_volumes(s, solids) == pytest.approx(
        [
            cylinder_volume(TUBE_RADIUS, TUBE_HEIGHT),
            cylinder_volume(APART_RADIUS, APART_HEIGHT),
        ],
        rel=1e-12,
    )
    inside_second = (APART_BASE[0], APART_BASE[1], APART_BASE[2] + APART_HEIGHT / 2.0)
    assert s.contains(solids, [TUBE_INSIDE, inside_second, TUBE_OUTSIDE]).tolist() == [
        [True, False, False],
        [False, True, False],
    ]


def test_sewing_a_tube_inside_a_tube_makes_a_hollow_solid() -> None:
    """A closed shell inside another, their surfaces apart, is that solid's cavity."""
    s = Session()
    outer = loose_tube(s, (0.37, -2.9, 0.3), 2.5, 9.0)
    inner = loose_tube(s, TUBE_BASE, TUBE_RADIUS, TUBE_HEIGHT)

    delta = s.sew([*outer, *inner], tolerance=SEW_TOL, make_solid=True)

    solid = solids_created(s, delta)
    assert len(solid) == 1
    hollow = cylinder_volume(2.5, 9.0) - cylinder_volume(TUBE_RADIUS, TUBE_HEIGHT)
    assert adaptive_volumes(s, solid) == pytest.approx([hollow], rel=1e-12)
    in_the_wall = (0.37 + 1.9, -2.9, 4.0)
    assert s.contains(solid, [TUBE_INSIDE, in_the_wall]).tolist() == [[False, True]]


def test_sewing_a_tube_inside_a_cavity_makes_it_a_solid_of_its_own() -> None:
    """Depth two: a shell inside a cavity is an island, the outer shell of a second solid."""
    s = Session()
    outer = loose_tube(s, (0.37, -2.9, 0.3), 2.5, 9.0)
    cavity = loose_tube(s, (0.37, -2.9, 1.3), 1.6, 7.0)
    island = loose_tube(s, (0.37, -2.9, 2.3), 0.7, 5.0)

    delta = s.sew([*outer, *cavity, *island], tolerance=SEW_TOL, make_solid=True)

    solids = solids_created(s, delta)
    hollow = cylinder_volume(2.5, 9.0) - cylinder_volume(1.6, 7.0)
    assert adaptive_volumes(s, solids) == pytest.approx(
        [hollow, cylinder_volume(0.7, 5.0)], rel=1e-12
    )
    points = [(0.37, -2.9, 4.0), (0.37 + 1.2, -2.9, 4.0), (0.37 + 2.1, -2.9, 4.0)]
    assert s.contains(solids, points).tolist() == [
        [False, False, True],
        [True, False, False],
    ]


def test_sewing_two_crossing_tubes_makes_two_solids() -> None:
    """Shells that cross are nested in neither direction: each bounds its own solid."""
    s = Session()
    first = loose_tube(s, TUBE_BASE, TUBE_RADIUS, TUBE_HEIGHT)
    second = loose_tube(s, (1.37, -2.9, 3.3), 1.0, 7.0)

    delta = s.sew([*first, *second], tolerance=SEW_TOL, make_solid=True)

    assert adaptive_volumes(s, solids_created(s, delta)) == pytest.approx(
        [cylinder_volume(TUBE_RADIUS, TUBE_HEIGHT), cylinder_volume(1.0, 7.0)],
        rel=1e-12,
    )


def test_sewing_a_box_resting_on_a_box_makes_two_solids() -> None:
    """Shells that touch from outside are not nested: every face of the top box is outside."""
    s = Session()
    lower = loose_box_faces(s, (BOX_DX, BOX_DY, BOX_DZ), (0.0, 0.0, 0.0))
    upper = loose_box_faces(s, (1.0, 2.0, 1.5), (1.0, 2.0, BOX_DZ))

    delta = s.sew([*lower, *upper], tolerance=SEW_TOL, make_solid=True)

    assert adaptive_volumes(s, solids_created(s, delta)) == pytest.approx(
        [BOX_VOLUME, 3.0], rel=1e-12
    )


def test_sewing_a_tube_standing_inside_a_tube_is_refused_and_changes_nothing() -> None:
    """The inner tube's base lies on the outer tube's base: cavity or crossing is undecided."""
    s = Session()
    outer = loose_tube(s, (0.37, -2.9, 0.3), 2.5, 9.0)
    inner = loose_tube(s, (0.37, -2.9, 0.3), TUBE_RADIUS, TUBE_HEIGHT)
    before = unchanged_state(s)

    with pytest.raises(ps.PysmeshError, match="undecided") as excinfo:
        s.sew([*outer, *inner], tolerance=SEW_TOL, make_solid=True)

    assert excinfo.value.face_ids == sorted(int(f) for f in inner)
    named = ", ".join(str(int(f)) for f in sorted(outer))
    assert f"faces [{named}]" in str(excinfo.value)
    assert unchanged_state(s) == before


def test_sewing_a_face_onto_its_own_copy_is_refused_and_changes_nothing() -> None:
    """A pillow: two coincident faces sew into a closed shell that bounds no volume.

    4.2.1 committed it as a solid of volume 0.0, and reported it valid.
    """
    s = Session()
    face = created(
        s, s.add_rectangle(TUBE_BASE, (0.0, 0.0, 1.0), 3.0, 7.0), EntityKind.FACE
    )
    twin = created(s, s.copy(face), EntityKind.FACE)
    before = unchanged_state(s)

    with pytest.raises(ps.PysmeshError, match="encloses a volume of 0") as excinfo:
        s.sew([*face, *twin], tolerance=SEW_TOL, make_solid=True)

    assert excinfo.value.face_ids == sorted(int(f) for f in (*face, *twin))
    assert "the session is unchanged" in str(excinfo.value)
    assert unchanged_state(s) == before


def test_sewing_a_shell_that_folds_through_itself_is_refused_and_changes_nothing(
    folded_loft: Callable[[Session], list[list[EntityId]]],
) -> None:
    """The classifier alone would orient it: the check on its volume refuses it.

    A ruled loft through the folded sections, left open, and two caps copied from its end
    sections sew into a closed shell that crosses itself. The point at infinity classifies
    outside it, and the solid it bounds has volume -8.51.
    """
    s = Session()
    sections = folded_loft(s)
    caps = []
    for section in (sections[0], sections[-1]):
        copy = created(s, s.extract_edges(section), EntityKind.EDGE)
        caps.extend(created(s, s.make_face(copy), EntityKind.FACE))
    wall = created(s, s.thru_sections(sections, solid=False), EntityKind.FACE)
    before = unchanged_state(s)

    with pytest.raises(ps.PysmeshError, match="disagree") as excinfo:
        s.sew([*wall, *caps], tolerance=SEW_TOL, make_solid=True)

    assert excinfo.value.face_ids == sorted(int(f) for f in (*wall, *caps))
    assert unchanged_state(s) == before


def test_sewing_into_a_solid_keeps_a_face_that_sewed_to_nothing() -> None:
    """4.2.1 replaced the whole sewing result by the solid and deleted the loose face."""
    s = Session()
    tube = loose_tube(s, TUBE_BASE, TUBE_RADIUS, TUBE_HEIGHT)
    far = s.add_rectangle((20.0, 20.0, 20.0), (0.0, 0.0, 1.0), 3.0, 7.0)
    loose = created(s, far, EntityKind.FACE)

    delta = s.sew([*tube, *loose], tolerance=SEW_TOL, make_solid=True)

    assert delta.deleted.size == 0
    assert s.is_alive(loose[0])
    assert adaptive_volumes(s, solids_created(s, delta)) == pytest.approx(
        [cylinder_volume(TUBE_RADIUS, TUBE_HEIGHT)], rel=1e-12
    )
    assert model_counts(s)[:2] == (1, 4)


# ============================================================== Internal-wire removal ==


def test_removing_internal_wires_clears_a_small_hole(bored_box: Session) -> None:
    assert model_volume(bored_box) == pytest.approx(
        BOX_VOLUME - HOLE_VOLUME, rel=CURVED_RTOL
    )

    delta = bored_box.remove_internal_wires(min_area=1.0)

    assert delta.valid is True
    assert model_volume(bored_box) == pytest.approx(BOX_VOLUME, rel=EXACT_RTOL)
    assert model_counts(bored_box) == (1, 6, 12, 8)


def test_removing_internal_wires_keeps_a_hole_above_the_threshold(
    bored_box: Session,
) -> None:
    """The threshold is the whole point: a hole bigger than it is a feature, not noise."""
    before = model_volume(bored_box)

    bored_box.remove_internal_wires(min_area=0.1 * math.pi * HOLE_RADIUS**2)

    assert model_volume(bored_box) == pytest.approx(before, rel=CURVED_RTOL)


def test_removing_internal_wires_rejects_a_non_positive_min_area(
    bored_box: Session,
) -> None:
    with pytest.raises(ps.PysmeshError, match="min_area"):
        bored_box.remove_internal_wires(min_area=0.0)


# =========================================================== Same-domain merging ==


def test_merging_same_domain_faces_halves_a_fused_seam(fused_boxes: Session) -> None:
    assert model_counts(fused_boxes)[1] == 10

    delta = fused_boxes.unify_same_domain()

    assert delta.valid is True
    assert model_counts(fused_boxes) == (1, 6, 12, 8)
    assert model_volume(fused_boxes) == pytest.approx(2 * BOX_VOLUME, rel=EXACT_RTOL)


def test_merging_keeps_every_face_id_alive_rather_than_deleting_the_losers(
    fused_boxes: Session,
) -> None:
    """A merge is many-to-one, and the rule is that *all* the ids survive on the result.

    So after merging 10 faces into 6, all 10 ids are still alive and each still resolves —
    which is what makes a reference held before the merge safe to use after it.
    """
    before = set(int(i) for i in fused_boxes.entities(EntityKind.FACE))

    delta = fused_boxes.unify_same_domain()

    after = set(int(i) for i in fused_boxes.entities(EntityKind.FACE))
    assert before <= after
    assert set(delta.deleted).isdisjoint(before)
    assert delta.merged.size > 0
    for face in before:
        status = fused_boxes.resolve(fused_boxes.name_of(EntityId(face))).status
        assert status is ResolutionStatus.RESOLVED


def test_merging_rejects_having_both_modes_off(fused_boxes: Session) -> None:
    with pytest.raises(ps.PysmeshError, match="at least one"):
        fused_boxes.unify_same_domain(unify_faces=False, unify_edges=False)


# ========================================================================= Defeaturing ==


def test_defeaturing_a_hole_restores_the_unbored_volume(bored_box: Session) -> None:
    """Gate: defeaturing removes exactly the intended faces and reports the survivors."""
    hole = hole_face(bored_box)
    faces_before = {int(i) for i in bored_box.entities(EntityKind.FACE)}

    delta = bored_box.defeature([hole])

    assert model_volume(bored_box) == pytest.approx(BOX_VOLUME, rel=EXACT_RTOL)
    assert model_counts(bored_box) == (1, 6, 12, 8)
    assert int(hole) in set(delta.deleted)
    assert not bored_box.is_alive(hole)
    survivors = faces_before - {int(hole)} - set(delta.deleted)
    assert survivors, "the box's own faces must survive the removal"
    for face in survivors:
        assert bored_box.is_alive(EntityId(face))


def test_defeaturing_fails_loud_when_occt_removes_nothing() -> None:
    """OCCT declines an incomplete feature as a *warning*, keeping its success flags set.

    Believing them would commit a no-op and tell the caller their feature is gone while it is
    still there, so the operation checks that every named face actually went away.
    """
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    face = ids_of(s, EntityKind.FACE)[0]
    ops_before = s.op_count
    ids_before = live_ids(s)

    with pytest.raises(ps.PysmeshError, match="removed no feature") as excinfo:
        s.defeature([face])

    assert excinfo.value.face_ids == [int(face)]
    assert s.op_count == ops_before
    assert live_ids(s) == ids_before


def test_defeaturing_rejects_an_entity_that_is_not_a_face(bored_box: Session) -> None:
    solid = ids_of(bored_box, EntityKind.SOLID)[0]

    with pytest.raises(ps.PysmeshError, match="not a FACE"):
        bored_box.defeature([solid])


def test_defeaturing_rejects_faces_from_two_bodies(two_boxes: Session) -> None:
    first, second = (
        int(i)
        for i in two_boxes.adjacency(EntityKind.SOLID, EntityKind.FACE).related[[0, -1]]
    )

    with pytest.raises(ps.PysmeshError, match="different bodies"):
        two_boxes.defeature([EntityId(first), EntityId(second)])


def test_defeaturing_an_empty_selection_raises(bored_box: Session) -> None:
    with pytest.raises(ps.PysmeshError, match="at least one"):
        bored_box.defeature([])


# ---- defeaturing: a removal must change the body ------------------------------------- #
#
# OCCT fills a removed feature by extending its neighbours, and a neighbour on the same
# surface extends straight back over the named face. The face is deleted and the body is
# the one that went in. These gates pin the refusal of that no-op, per feature, and pin that
# a real feature — however small against its body — still goes.

PLATE_DX: float = 13.0
PLATE_DY: float = 11.0
PLATE_DZ: float = 3.0
PLATE_ORIGIN: tuple[float, float, float] = (0.37, -2.9, 1.3)
SPLIT_RADIUS: float = 1.2
SPLIT_AXIS: tuple[float, float] = (6.47, 2.4)
SECOND_RADIUS: float = 0.8
SECOND_AXIS: tuple[float, float] = (2.5, 0.0)
CUTTER_Z0: float = 0.3
CUTTER_HEIGHT: float = 5.0

FILLET_PLATE: tuple[float, float, float] = (61.0, 37.0, 7.0)


def created(session: Session, delta: ps.HistoryDelta, kind: EntityKind) -> list[EntityId]:
    """The ids of one kind that an operation issued."""
    return [
        EntityId(int(i))
        for i in delta.created
        if session.entity_kind(EntityId(int(i))) is kind
    ]


def half_cylinder(
    session: Session, start: float, axis: tuple[float, float], radius: float
) -> EntityId:
    """A half-cylinder about +z, built by revolving an axial rectangle through pi."""
    cx, cy = axis
    ux, uy = math.cos(start), math.sin(start)
    corners = np.array(
        [
            [cx, cy, CUTTER_Z0],
            [cx + radius * ux, cy + radius * uy, CUTTER_Z0],
            [cx + radius * ux, cy + radius * uy, CUTTER_Z0 + CUTTER_HEIGHT],
            [cx, cy, CUTTER_Z0 + CUTTER_HEIGHT],
        ],
        dtype=np.float64,
    )
    edges = created(session, session.add_polyline(corners, closed=True), EntityKind.EDGE)
    face = created(session, session.make_face(edges), EntityKind.FACE)
    swept = session.revolve(face, (cx, cy, 0.0), (0.0, 0.0, 1.0), math.pi)
    return created(session, swept, EntityKind.SOLID)[0]


def cut_split_hole(
    session: Session, body: EntityId, axis: tuple[float, float], radius: float
) -> None:
    """Cut a through hole whose wall is two half-cylinder faces on one surface."""
    first = half_cylinder(session, 0.0, axis, radius)
    second = half_cylinder(session, math.pi, axis, radius)
    session.fuse([first], [second])
    tools = [
        EntityId(int(i))
        for i in session.entities(EntityKind.SOLID)
        if int(i) != int(body)
    ]
    session.cut([body], tools)


def cylinder_faces(session: Session) -> list[EntityId]:
    """Every cylindrical face, ascending."""
    table = session.entity_types(EntityKind.FACE)
    return [
        EntityId(int(i)) for i, t in zip(table.ids, table.types) if t == "Cylinder"
    ]


def unchanged_state(session: Session) -> tuple[int, int, set[int], bytes]:
    """Everything a refused operation must leave exactly as it was."""
    return session.op_count, session.issued_id_count, live_ids(session), session.brep()


@pytest.fixture
def split_wall_plate() -> Session:
    """A 13 x 11 x 3 plate with one through hole whose wall is two half-cylinders."""
    s = Session()
    box = s.add_box(PLATE_DX, PLATE_DY, PLATE_DZ, PLATE_ORIGIN)
    body = created(s, box, EntityKind.SOLID)
    cut_split_hole(s, body[0], SPLIT_AXIS, SPLIT_RADIUS)
    return s


def test_the_split_wall_plate_is_the_plate_its_closed_form_describes(
    split_wall_plate: Session,
) -> None:
    """Fixture check: V = 13 x 11 x 3 - pi r^2 x 3, and two wall faces of pi r h each."""
    walls = cylinder_faces(split_wall_plate)

    areas = split_wall_plate.mass_properties(walls).measure

    assert model_volume(split_wall_plate) == pytest.approx(
        PLATE_DX * PLATE_DY * PLATE_DZ - math.pi * SPLIT_RADIUS**2 * PLATE_DZ,
        rel=EXACT_RTOL,
    )
    assert len(walls) == 2
    assert areas == pytest.approx([math.pi * SPLIT_RADIUS * PLATE_DZ] * 2, rel=EXACT_RTOL)


def test_defeaturing_one_half_of_a_split_wall_is_refused_and_changes_nothing(
    split_wall_plate: Session,
) -> None:
    """The measured 4.2.0 defect: deleting one half grew the other into the whole wall.

    OCCT deleted the named face, extended its sibling over the region it covered, and
    reported success with a volume change of 0 of 415.43.
    """
    first, second = cylinder_faces(split_wall_plate)
    before = unchanged_state(split_wall_plate)

    with pytest.raises(ps.PysmeshError, match="left the body's volume") as excinfo:
        split_wall_plate.defeature([first])

    assert excinfo.value.face_ids == [int(first)]
    assert f"continues into face {int(second)}" in str(excinfo.value)
    assert "unify_same_domain" in str(excinfo.value)
    assert "tolerance" in excinfo.value.details
    assert unchanged_state(split_wall_plate) == before


def test_defeaturing_both_halves_of_a_split_wall_removes_the_hole(
    split_wall_plate: Session,
) -> None:
    """Both halves are the complete feature: the hole goes, the volume grows pi r^2 h."""
    walls = cylinder_faces(split_wall_plate)
    volume_before = model_volume(split_wall_plate)

    delta = split_wall_plate.defeature(walls)

    grown = model_volume(split_wall_plate) - volume_before
    assert grown == pytest.approx(math.pi * SPLIT_RADIUS**2 * PLATE_DZ, rel=1e-12)
    assert {int(w) for w in walls} <= set(delta.deleted.tolist())
    assert cylinder_faces(split_wall_plate) == []


@pytest.mark.parametrize("radius", [0.7, 0.01])
def test_defeaturing_a_small_fillet_on_a_large_plate_still_removes_it(
    radius: float,
) -> None:
    """A real feature passes however small it is against its body.

    The closed form of a fillet of radius r along a straight edge of length L is
    (1 - pi/4) r^2 L. At r = 0.7 that is 0.736 of 15798, 4.7e-5 relative. At r = 0.01 it is
    1.5e-4, under 1e-7 x the plate's whole area (5.9e-4), against a patch tolerance of
    1.4e-8. Its area change, (2 - pi/2) r L = 0.03, is what a body tolerance would still
    see; the dimple test below takes that away as well.
    """
    dx, dy, dz = FILLET_PLATE
    s = Session()
    s.add_box(dx, dy, dz)
    edges = ids_of(s, EntityKind.EDGE)
    props = s.mass_properties(edges)
    corner = [
        e
        for e, length, c in zip(edges, props.measure, props.centroid)
        if length == pytest.approx(dz) and c[0] > dx / 2.0 and c[1] > dy / 2.0
    ]
    s.fillet(corner, radius)
    fillet = cylinder_faces(s)
    volume_before = model_volume(s)

    s.defeature(fillet)

    removed = (1.0 - math.pi / 4.0) * radius**2 * dz
    assert model_volume(s) - volume_before == pytest.approx(removed, rel=1e-6)
    assert model_volume(s) == pytest.approx(dx * dy * dz, rel=EXACT_RTOL)


def test_defeaturing_a_shallow_dimple_is_judged_on_its_own_patch_not_the_body() -> None:
    """A feature whose volume AND area changes both sit under a whole-body tolerance.

    A sphere of radius R = 2500 dipped h = 2e-4 into the 61 x 37 x 7 plate leaves a dimple
    of chord radius 1. Its cap volume pi h^2 (3R - h) / 3 = 3.14e-4 is under 1e-7 x the
    plate's area (5.9e-4), and its area change 1.3e-7 is under 2e-7 x the plate's edge
    length (8.6e-5), so a tolerance taken over the body refuses it. Over the dimple's own
    patch the volume tolerance is 3.1e-7, a thousandth of the change.
    """
    radius, depth = 2500.0, 2e-4
    dx, dy, dz = FILLET_PLATE
    s = Session()
    plate = created(s, s.add_box(dx, dy, dz), EntityKind.SOLID)
    ball = s.add_sphere(radius, centre=(dx / 2.0, dy / 2.0, dz + radius - depth))
    s.cut(plate, created(s, ball, EntityKind.SOLID))
    table = s.entity_types(EntityKind.FACE)
    dimple = [EntityId(int(i)) for i, t in zip(table.ids, table.types) if t == "Sphere"]
    volume_before = model_volume(s)

    s.defeature(dimple)

    cap = math.pi * depth**2 * (3.0 * radius - depth) / 3.0
    assert model_volume(s) - volume_before == pytest.approx(cap, rel=1e-6)
    assert model_counts(s) == (1, 6, 12, 8)


def test_defeaturing_refuses_only_the_half_wall_beside_a_complete_hole(
    split_wall_plate: Session,
) -> None:
    """Per feature, not per call: the complete hole's change must not hide the no-op."""
    solid = ids_of(split_wall_plate, EntityKind.SOLID)[0]
    first, _ = cylinder_faces(split_wall_plate)
    cutter = split_wall_plate.add_cylinder(
        SECOND_RADIUS, CUTTER_HEIGHT, origin=(*SECOND_AXIS, CUTTER_Z0)
    )
    split_wall_plate.cut([solid], created(split_wall_plate, cutter, EntityKind.SOLID))
    walls = cylinder_faces(split_wall_plate)
    radii = split_wall_plate.surface_parameters(walls).radius1
    complete = [w for w, r in zip(walls, radii) if r == pytest.approx(SECOND_RADIUS)]
    assert len(complete) == 1
    before = unchanged_state(split_wall_plate)

    with pytest.raises(ps.PysmeshError, match="left the body's volume") as excinfo:
        split_wall_plate.defeature([complete[0], first])

    assert excinfo.value.face_ids == [int(first)]
    assert unchanged_state(split_wall_plate) == before


def test_defeaturing_refuses_every_half_wall_named_together(
    split_wall_plate: Session,
) -> None:
    """Two separate no-ops in one call: both are named, and nothing else is."""
    solid = ids_of(split_wall_plate, EntityKind.SOLID)[0]
    cut_split_hole(split_wall_plate, solid, SECOND_AXIS, SECOND_RADIUS)
    a_first, a_second, b_first, b_second = cylinder_faces(split_wall_plate)
    before = unchanged_state(split_wall_plate)

    with pytest.raises(ps.PysmeshError, match="left the body's volume") as excinfo:
        split_wall_plate.defeature([b_first, a_first, b_second])

    assert excinfo.value.face_ids == [int(a_first)]
    assert unchanged_state(split_wall_plate) == before

    with pytest.raises(ps.PysmeshError, match="left the body's volume") as excinfo:
        split_wall_plate.defeature([a_first, b_first])

    assert excinfo.value.face_ids == [int(a_first), int(b_first)]
    assert unchanged_state(split_wall_plate) == before


@pytest.mark.parametrize("piece_area", [16.0, 127.0])
def test_defeaturing_one_piece_of_an_imprinted_face_is_refused(piece_area: float) -> None:
    """The zero-volume removal: a face split by an imprint only merges back when removed.

    That is a same-domain merge, not a feature removal, so it is refused and the message
    names the operation that does it. Pieces are named on a fresh session loaded from the
    imprinted BREP, as an imported model presents them: inside the imprinting session one
    id denotes both pieces and cannot name either alone.
    """
    source = Session()
    source.add_box(PLATE_DX, PLATE_DY, PLATE_DZ)
    tool = source.add_rectangle((4.0, 3.0, PLATE_DZ), (0.0, 0.0, 1.0), 4.0, 4.0)
    source.imprint(
        ids_of(source, EntityKind.SOLID)[:1], created(source, tool, EntityKind.FACE)
    )
    source.remove(created(source, tool, EntityKind.FACE))
    s = Session()
    s.add_brep(source.brep())
    faces = ids_of(s, EntityKind.FACE)
    piece = [f for f, a in zip(faces, s.mass_properties(faces).measure) if a == piece_area]
    assert len(piece) == 1
    before = unchanged_state(s)

    with pytest.raises(ps.PysmeshError, match="unify_same_domain") as excinfo:
        s.defeature(piece)

    assert excinfo.value.face_ids == [int(piece[0])]
    assert unchanged_state(s) == before
    s.unify_same_domain()
    assert model_counts(s) == (1, 6, 12, 8)
    assert model_volume(s) == pytest.approx(PLATE_DX * PLATE_DY * PLATE_DZ, rel=EXACT_RTOL)


def test_defeaturing_half_of_a_free_form_split_wall_is_refused() -> None:
    """The no-op on a B-spline surface, where GProp's fixed rule is too coarse to see it.

    A tube swept along a spline is imprinted by a plane and fused back, which leaves its
    wall two B-spline faces on one surface. GProp's fixed Gauss rule measures the no-op's
    volume change at 3.0e-6 against a tolerance of 1.9e-6, and 4.2.0 committed it.
    """
    sweep = Session()
    spine = sweep.add_spline(
        [(0.0, 0.0, 0.0), (0.0, 1.2, 2.0), (0.0, 0.4, 4.0), (0.0, 1.5, 6.0)]
    )
    circle = sweep.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)
    profile = sweep.make_face(created(sweep, circle, EntityKind.EDGE))
    tube = sweep.pipe(
        created(sweep, spine, EntityKind.EDGE), created(sweep, profile, EntityKind.FACE)
    )
    plane = sweep.add_rectangle((-5.0, -5.0, 3.1), (0.0, 0.0, 1.0), 10.0, 10.0)
    sweep.imprint(
        created(sweep, tube, EntityKind.SOLID), created(sweep, plane, EntityKind.FACE)
    )
    sweep.remove(created(sweep, plane, EntityKind.FACE))
    s = Session()
    s.add_brep(sweep.brep())
    s.fuse(ids_of(s, EntityKind.SOLID)[:1], ids_of(s, EntityKind.SOLID)[1:])
    table = s.entity_types(EntityKind.FACE)
    wall = [EntityId(int(i)) for i, t in zip(table.ids, table.types) if t == "BSpline"]
    assert len(wall) == 2
    before = unchanged_state(s)

    for face in wall:
        with pytest.raises(ps.PysmeshError, match="left the body's volume") as excinfo:
            s.defeature([face])

        assert excinfo.value.face_ids == [int(face)]
        assert unchanged_state(s) == before


# =========================================================================== Imprinting ==


def test_imprinting_a_plane_splits_the_target_and_keeps_the_tool() -> None:
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_rectangle((-1.0, -1.0, BOX_DZ / 2.0), (0.0, 0.0, 1.0), 6.0, 10.0)
    box = ids_of(s, EntityKind.SOLID)[0]
    tool = ids_of(s, EntityKind.FACE)[-1]

    delta = s.imprint([box], [tool])

    assert delta.valid is True
    assert model_counts(s)[0] == 2
    assert model_volume(s) == pytest.approx(BOX_VOLUME, rel=EXACT_RTOL)
    assert s.is_alive(tool), "an imprint must not consume its tool"


def test_an_imprinted_target_keeps_its_id_on_every_piece() -> None:
    """A split: the id survives on every piece, and the name says so rather than guessing."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_rectangle((-1.0, -1.0, BOX_DZ / 2.0), (0.0, 0.0, 1.0), 6.0, 10.0)
    box = ids_of(s, EntityKind.SOLID)[0]
    tool = ids_of(s, EntityKind.FACE)[-1]
    name = s.name_of(box)

    s.imprint([box], [tool])

    assert s.is_alive(box)
    assert s.shape_count(box) == 2
    assert s.resolve(name).status is ResolutionStatus.AMBIGUOUS


def test_imprinting_with_full_gluing_handles_face_coincident_bodies() -> None:
    """Gluing is the fast path for operands that only touch, which is this fixture exactly."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(BOX_DX, 0.0, 0.0))
    a, b = ids_of(s, EntityKind.SOLID)

    delta = s.imprint([a], [b], glue=GlueMode.FULL)

    assert delta.valid is True
    assert model_volume(s) == pytest.approx(2 * BOX_VOLUME, rel=EXACT_RTOL)
    assert s.is_alive(b)


def test_imprinting_rejects_a_body_named_on_both_sides(two_boxes: Session) -> None:
    first = ids_of(two_boxes, EntityKind.SOLID)[0]

    with pytest.raises(ps.PysmeshError, match="both targets and tools"):
        two_boxes.imprint([first], [first])


def test_imprinting_rejects_an_empty_tool_list(two_boxes: Session) -> None:
    first = ids_of(two_boxes, EntityKind.SOLID)[0]

    with pytest.raises(ps.PysmeshError, match="tools must name"):
        two_boxes.imprint([first], [])


# ============================================================================== Removal ==


def test_removing_a_body_kills_its_ids_and_leaves_the_rest_alone(
    two_boxes: Session,
) -> None:
    first, second = ids_of(two_boxes, EntityKind.SOLID)
    doomed = body_entities(two_boxes, second)
    survivors = live_ids(two_boxes) - doomed

    delta = two_boxes.remove([second])

    assert model_counts(two_boxes)[0] == 1
    assert model_volume(two_boxes) == pytest.approx(BOX_VOLUME, rel=EXACT_RTOL)
    assert doomed <= set(delta.deleted)
    assert survivors <= live_ids(two_boxes)
    assert two_boxes.is_alive(first)


def test_a_removed_id_reports_dead_rather_than_denoting_something_else(
    two_boxes: Session,
) -> None:
    _, second = ids_of(two_boxes, EntityKind.SOLID)
    name = two_boxes.name_of(second)

    two_boxes.remove([second])

    assert not two_boxes.is_alive(second)
    assert two_boxes.resolve(name).status is ResolutionStatus.LOST


def test_removal_builds_nothing_so_it_reports_no_verdict(two_boxes: Session) -> None:
    _, second = ids_of(two_boxes, EntityKind.SOLID)

    delta = two_boxes.remove([second])

    assert delta.valid is None


def test_removing_an_empty_selection_raises(two_boxes: Session) -> None:
    with pytest.raises(ps.PysmeshError, match="at least one"):
        two_boxes.remove([])


def test_removing_an_already_dead_id_raises(two_boxes: Session) -> None:
    _, second = ids_of(two_boxes, EntityKind.SOLID)
    two_boxes.remove([second])

    with pytest.raises(ps.PysmeshError, match="dead"):
        two_boxes.remove([second])


# ================================================================= The verdict contract ==


@pytest.mark.parametrize("operation", ["heal", "sew", "remove_internal_wires", "unify"])
def test_every_repair_operation_reports_a_validity_verdict(operation: str) -> None:
    """Gate: each repair returns a full delta and a ``BRepCheck_Analyzer`` verdict."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    if operation == "heal":
        delta = s.heal()
    elif operation == "sew":
        delta = s.sew(ids_of(s, EntityKind.SOLID))
    elif operation == "remove_internal_wires":
        delta = s.remove_internal_wires(min_area=1.0)
    else:
        delta = s.unify_same_domain()

    assert isinstance(delta.valid, bool)
    assert delta.op == ("unify_same_domain" if operation == "unify" else operation)
    for field in (delta.created, delta.deleted, delta.modified, delta.split, delta.merged):
        assert isinstance(field, np.ndarray)


def test_a_repair_reports_its_verdict_even_in_an_unvalidated_session() -> None:
    """With ``validate=False`` the verdict is still taken, because here it is the answer.

    A heal that cannot say whether it succeeded has not done its job, so the check is part of
    the operation rather than a safety net the caller can switch off.
    """
    s = Session(validate=False)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    assert s.heal().valid is True
    assert s.add_box(1.0, 2.0, 3.0, origin=(30.0, 0.0, 0.0)).valid is None


# ====================================== The cost of closing a sewn shell (report P2) ==


def _production_solid_25(industrial_step_brep: bytes) -> bytes:
    """Solid 25 of the production assembly (436 faces) alone, as BREP bytes."""
    s = Session()
    s.add_brep(industrial_step_brep)
    solids = sorted({int(x) for x in s.entities(EntityKind.SOLID)})
    s.remove([i for i in solids if i != 25])
    return s.brep()


def _best_sew_seconds(body: bytes, make_solid: bool) -> float:
    """The fastest of three sews of every solid of ``body``, in seconds."""
    best = math.inf
    for _ in range(3):
        s = Session()
        s.add_brep(body)
        named = sorted({int(x) for x in s.entities(EntityKind.SOLID)})
        start = time.perf_counter()
        s.sew(named, tolerance=SEW_TOL, make_solid=make_solid)
        best = min(best, time.perf_counter() - start)
    return best


def test_closing_the_production_solid_costs_less_than_three_sews(
    industrial_step_brep: bytes,
) -> None:
    """Report P2: ``make_solid=True`` cost 7.3 times the sew on this solid (1 509 ms).

    The volume integral that settles the shell's inside was the whole extra cost. A
    tessellated copy settles the sign of a volume this far above its tolerance, so the
    integral is skipped and closing the shell costs about 1.7 sews.
    """
    body = _production_solid_25(industrial_step_brep)

    shell = _best_sew_seconds(body, make_solid=False)
    solid = _best_sew_seconds(body, make_solid=True)

    assert solid < 3.0 * shell, (solid, shell)


def test_closing_a_sewn_shell_leaves_no_triangulation_on_model_or_snapshot() -> None:
    """The sign check tessellates a copy, so no face of the session gets triangles."""
    s = Session()
    faces = loose_box_faces(s, (BOX_DX, BOX_DY, BOX_DZ), (0.0, 0.0, 0.0))
    mark = s.snapshot()

    s.sew(faces, tolerance=SEW_TOL, make_solid=True)
    after = s.brep()
    s.restore(mark)
    before = s.brep()

    assert b"Triangulations 0" in after
    assert b"Triangulations 0" in before


# =================================== make_solid that cannot make a solid (report A8) ==


def _wrap_around_face_with_circle_caps(s: Session) -> list[EntityId]:
    """A lateral face lofted round a circle and two exact-circle caps; return the faces.

    The loft interpolates 13 straight generators round a unit circle, the last a copy of
    the first, so its rims are B-spline interpolants of the circle. The caps are exact
    circles. Rims and caps differ by more than 1e-6, so at that tolerance the sew joins
    the two generators of the lateral face and nothing else (report A8: 4 free edges).
    """
    sections = []
    for p in np.linspace(0.0, 2.0 * np.pi, 13)[:-1]:
        before = set(ids_of(s, EntityKind.EDGE))
        x, y = float(np.cos(p)), float(np.sin(p))
        s.add_line((x, y, 0.0), (x, y, 1.0))
        sections.append([e for e in ids_of(s, EntityKind.EDGE) if e not in before])
    before = set(ids_of(s, EntityKind.EDGE))
    s.copy(sections[0])
    sections.append([e for e in ids_of(s, EntityKind.EDGE) if e not in before])
    s.thru_sections(sections, solid=False, ruled=False)
    for z in (0.0, 1.0):
        before = set(ids_of(s, EntityKind.EDGE))
        s.add_circle((0.0, 0.0, z), (0.0, 0.0, 1.0), 1.0)
        s.make_face([e for e in ids_of(s, EntityKind.EDGE) if e not in before])
    return ids_of(s, EntityKind.FACE)


def test_make_solid_refuses_a_sew_leaving_an_open_shell_naming_free_edges() -> None:
    """The capped wrap-around face sews into an open shell: 4 free edges, no solid."""
    s = Session()
    faces = _wrap_around_face_with_circle_caps(s)
    brep = s.brep()

    with pytest.raises(ps.PysmeshError) as info:
        s.sew(faces, tolerance=SEW_TOL, make_solid=True)

    assert "4 free boundary edge" in str(info.value)
    assert info.value.details.count("Free edge") == 4
    assert s.brep() == brep


def test_sewing_without_make_solid_still_commits_the_open_shell() -> None:
    """``make_solid=False`` is unchanged: the open shell is committed, valid."""
    s = Session()
    faces = _wrap_around_face_with_circle_caps(s)

    delta = s.sew(faces, tolerance=SEW_TOL)

    assert delta.valid is True
    assert model_counts(s)[0] == 0
    assert len(ps.free_boundary_edges(s.brep())) == 4
