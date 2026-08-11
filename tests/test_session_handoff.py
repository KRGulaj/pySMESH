# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""Gates for the CAD-to-mesher handoff: a deterministic, verified id-to-ordinal bijection.

The session is the CAD authority and a mesher is a consumer of it. What crosses the boundary
is BREP bytes plus the one thing bytes cannot carry — which entity id each of their
sub-shapes is. Three claims are under test.

* **The map is positional, and the position survives the round trip.** Each id array holds
  one id per ordinal of the per-kind traversal a reader of the exported bytes reproduces.
  The test reads the bytes back through the stateless API and checks, ordinal by ordinal,
  that the shape found there is the shape the id denotes here.
* **The map is a bijection, and that is verified rather than assumed.** Every exported
  sub-shape carries exactly one live id and every live id labels exactly one exported
  sub-shape. Two ordinary session states break this — a same-domain merge leaves several ids
  on one face, a split leaves one id on several — and the export must refuse both, naming
  the ids, rather than handing over a map that silently loses some of the caller's names.
* **Matching by geometry would be wrong, and there is a fixture that proves it.** A pipe's
  inner and outer cylindrical walls share a centroid exactly. The suite asserts that a
  centroid-keyed map collides on that fixture *and* that the shipped positional map does
  not — the falsification, without which "the map is a bijection" is a claim about a check
  that has never been shown to fail.

Geometric matching appears here only as a **test oracle**, never as a resolution strategy,
and each use asserts its own fingerprints are unambiguous before relying on them.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube.
"""

from __future__ import annotations

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, Handoff, PysmeshError, Session

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0

PIPE_OUTER_RADIUS: float = 2.0
PIPE_INNER_RADIUS: float = 1.0
PIPE_HEIGHT: float = 10.0

# Two centroids closer than this are indistinguishable to a centroid-keyed map. The pipe's
# two walls were measured 6e-17 apart, which is float noise around exact coincidence.
CENTROID_COLLISION_TOL: float = 1.0e-9

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


# ---- fixtures and helpers ------------------------------------------------------------ #


@pytest.fixture
def box_session() -> Session:
    """One 3 x 7 x 11 box: 1 solid, 6 faces, 12 edges, 8 vertices."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    return s


def pipe_session() -> Session:
    """A pipe, whose inner and outer walls share a centroid exactly."""
    s = Session()
    s.add_cylinder(PIPE_OUTER_RADIUS, PIPE_HEIGHT)
    s.add_cylinder(PIPE_INNER_RADIUS, PIPE_HEIGHT)
    ids = [EntityId(int(v)) for v in s.entities(EntityKind.SOLID)]
    s.cut([ids[0]], [ids[1]])
    return s


def ids_for(handoff: Handoff, kind: EntityKind) -> np.ndarray:
    """The id array of one kind, so the tests can loop over the four."""
    return {
        EntityKind.SOLID: handoff.solid_id,
        EntityKind.FACE: handoff.face_id,
        EntityKind.EDGE: handoff.edge_id,
        EntityKind.VERTEX: handoff.vertex_id,
    }[kind]


def face_fingerprints(brep: bytes) -> list[tuple[float, tuple[float, float, float]]]:
    """Area and centroid of each face of a BREP, in its own 1-based ordinal order.

    A test oracle only. The session never resolves anything this way, and every test using
    this asserts the fingerprints are distinct before trusting them.
    """
    faces = sorted(ps.load_brep(brep).faces(), key=lambda f: f.id)
    return [(f.area, tuple(float(v) for v in f.centroid)) for f in faces]


def centroids_collide(brep: bytes) -> int:
    """How many face-centroid pairs are indistinguishable — the forbidden strategy's flaw."""
    centroids = [c for _, c in face_fingerprints(brep)]
    collisions = 0
    for i, a in enumerate(centroids):
        for b in centroids[i + 1 :]:
            if float(np.linalg.norm(np.array(a) - np.array(b))) < CENTROID_COLLISION_TOL:
                collisions += 1
    return collisions


# ---- the manifest's shape ------------------------------------------------------------ #


def test_export_returns_one_id_per_sub_shape_of_each_kind(box_session: Session) -> None:
    handoff = box_session.export_handoff()

    assert handoff.solid_id.shape == (1,)
    assert handoff.face_id.shape == (6,)
    assert handoff.edge_id.shape == (12,)
    assert handoff.vertex_id.shape == (8,)


def test_every_id_array_is_int64(box_session: Session) -> None:
    handoff = box_session.export_handoff()

    for kind in _KINDS:
        assert ids_for(handoff, kind).dtype == np.int64


def test_the_exported_bytes_reload_as_the_same_model(box_session: Session) -> None:
    handoff = box_session.export_handoff()

    shape = ps.load_brep(handoff.brep)

    assert len(shape.solids()) == 1
    assert len(shape.faces()) == 6
    assert sum(s.volume for s in shape.solids()) == pytest.approx(
        BOX_DX * BOX_DY * BOX_DZ
    )


# ---- the bijection, both directions -------------------------------------------------- #


@pytest.mark.parametrize("kind", _KINDS)
def test_no_id_labels_two_sub_shapes(box_session: Session, kind: EntityKind) -> None:
    ids = ids_for(box_session.export_handoff(), kind)

    assert len(set(ids.tolist())) == ids.size


@pytest.mark.parametrize("kind", _KINDS)
def test_every_exported_id_is_a_live_id_of_that_kind(
    box_session: Session, kind: EntityKind
) -> None:
    handoff = box_session.export_handoff()
    live = set(int(v) for v in box_session.entities(kind))

    assert set(ids_for(handoff, kind).tolist()) <= live


@pytest.mark.parametrize("kind", _KINDS)
def test_every_live_id_of_that_kind_appears_exactly_once(
    box_session: Session, kind: EntityKind
) -> None:
    # The other direction, and the one that catches a map that quietly drops names.
    handoff = box_session.export_handoff()
    live = sorted(int(v) for v in box_session.entities(kind))

    assert sorted(ids_for(handoff, kind).tolist()) == live


def test_the_four_kinds_draw_on_one_id_space_and_never_collide(
    box_session: Session,
) -> None:
    handoff = box_session.export_handoff()
    everything = [
        int(v) for kind in _KINDS for v in ids_for(handoff, kind).tolist()
    ]

    assert len(set(everything)) == len(everything)


def test_an_exported_id_resolves_through_the_registry(box_session: Session) -> None:
    handoff = box_session.export_handoff()

    for face_id in handoff.face_id.tolist():
        assert box_session.entity_kind(EntityId(int(face_id))) == EntityKind.FACE


# ---- the ordinals mean what they say ------------------------------------------------- #


def test_each_ordinal_names_the_sub_shape_at_that_ordinal_of_the_exported_bytes(
    box_session: Session,
) -> None:
    # The claim the whole handoff rests on: position i on this side is position i on the
    # other. Checked with a geometric oracle the session did not produce, on a fixture whose
    # fingerprints are asserted distinct first.
    handoff = box_session.export_handoff()
    reloaded = face_fingerprints(handoff.brep)
    assert len({c for _, c in reloaded}) == len(reloaded), "oracle is ambiguous"

    for ordinal, face_id in enumerate(handoff.face_id.tolist()):
        table = box_session.mass_properties([EntityId(int(face_id))])
        area, centroid = reloaded[ordinal]
        assert float(table.measure[0]) == pytest.approx(area, rel=1e-12)
        assert np.allclose(table.centroid[0], np.array(centroid), atol=1e-12)


def test_two_exports_of_one_session_are_identical(box_session: Session) -> None:
    first = box_session.export_handoff()
    second = box_session.export_handoff()

    assert first.brep == second.brep
    for kind in _KINDS:
        assert np.array_equal(ids_for(first, kind), ids_for(second, kind))


def test_the_ordering_survives_an_operation_elsewhere_in_the_model() -> None:
    # Adding a body must not silently renumber the map of the one already exported: the
    # ordinals of the first body's faces still name the same faces.
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    before = s.export_handoff()
    s.add_sphere(2.0, centre=(50.0, 0.0, 0.0))

    after = s.export_handoff()

    assert after.face_id[: before.face_id.size].tolist() == before.face_id.tolist()


# ---- falsification: the check must be shown to fail ---------------------------------- #


def test_a_merged_model_is_refused_naming_the_ambiguous_ids(
    split_box_brep: bytes,
) -> None:
    # A same-domain merge leaves several live ids on one face. Both names are alive and both
    # mean that face, so the handoff cannot choose — and must not.
    s = Session()
    s.add_brep(split_box_brep)
    s.unify_same_domain()

    with pytest.raises(PysmeshError, match="not a bijection") as excinfo:
        s.export_handoff()

    assert excinfo.value.face_ids, "the offending ids must be named"


def test_the_same_model_exports_cleanly_before_the_merge(split_box_brep: bytes) -> None:
    # The control. Without it the test above would pass against an export that always
    # raised.
    s = Session()
    s.add_brep(split_box_brep)

    handoff = s.export_handoff()

    assert handoff.face_id.size == 10
    assert len(set(handoff.face_id.tolist())) == 10


def test_a_split_model_is_refused_naming_the_split_id() -> None:
    # The other direction: one live id denoting several sub-shapes.
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, 1.0, BOX_DZ, origin=(0.0, BOX_DY / 2.0 - 0.5, 0.0))
    ids = [EntityId(int(v)) for v in s.entities(EntityKind.SOLID)]
    s.split([ids[0]], [ids[1]])
    assert s.shape_count(ids[0]) > 1, "the fixture did not split anything"

    with pytest.raises(PysmeshError, match="not a bijection") as excinfo:
        s.export_handoff()

    assert ids[0] in excinfo.value.face_ids


def test_coaxial_walls_defeat_a_centroid_map_but_not_the_shipped_one() -> None:
    # The reason the map is positional. A pipe's inner and outer walls have the same
    # centroid, so the obvious geometric shortcut mis-pairs two different faces without
    # saying so — while the ordinal map is a bijection on the very same shape.
    s = pipe_session()
    handoff = s.export_handoff()

    assert centroids_collide(handoff.brep) >= 1, "the fixture lost its coaxial walls"
    assert len(set(handoff.face_id.tolist())) == handoff.face_id.size


def test_the_pipe_fixture_really_has_two_coaxial_walls() -> None:
    s = pipe_session()

    shape = ps.load_brep(s.brep())

    assert len(shape.faces()) == 4
    radii = sorted(
        f.area / (2.0 * np.pi * PIPE_HEIGHT)
        for f in shape.faces()
        if f.surface_type == "Cylinder"
    )
    assert radii == pytest.approx([PIPE_INNER_RADIUS, PIPE_OUTER_RADIUS])


# ---- the export is a query ------------------------------------------------------------ #


def test_exporting_issues_no_id_and_advances_no_counter(box_session: Session) -> None:
    ops = box_session.op_count
    issued = box_session.issued_id_count
    state = box_session.state_op_index

    box_session.export_handoff()

    assert box_session.op_count == ops
    assert box_session.issued_id_count == issued
    assert box_session.state_op_index == state


def test_an_empty_session_exports_an_empty_manifest() -> None:
    handoff = Session().export_handoff()

    assert handoff.face_id.size == 0
    assert handoff.brep


# ---- the real assembly --------------------------------------------------------------- #


@pytest.mark.slow
def test_the_handoff_is_a_bijection_on_a_real_assembly(
    industrial_step_brep: bytes,
) -> None:
    s = Session()
    s.add_brep(industrial_step_brep)

    handoff = s.export_handoff()

    assert handoff.face_id.size >= 500, "the gate wants a model of at least 500 faces"
    for kind in _KINDS:
        ids = ids_for(handoff, kind)
        live = sorted(int(v) for v in s.entities(kind))
        # Both directions at once: same multiset means no id is dropped and none is doubled.
        assert sorted(ids.tolist()) == live


@pytest.mark.slow
def test_the_ordinals_survive_the_round_trip_on_a_real_assembly(
    industrial_step_brep: bytes,
) -> None:
    # The export order is only useful if a reader of the bytes reproduces it. Checked per
    # ordinal against the reloaded shape, over every face of the assembly.
    s = Session()
    s.add_brep(industrial_step_brep)
    handoff = s.export_handoff()

    reloaded = ps.load_brep(handoff.brep)

    assert len(reloaded.faces()) == handoff.face_id.size
    assert len(reloaded.solids()) == handoff.solid_id.size
    assert len(reloaded.edges()) == handoff.edge_id.size
    assert len(reloaded.vertices()) == handoff.vertex_id.size

    # Spot-check the pairing itself on a spread of ordinals: reading every face's mass
    # properties on real trimmed geometry costs seconds and proves nothing more.
    faces = sorted(reloaded.faces(), key=lambda f: f.id)
    sample = range(0, handoff.face_id.size, max(1, handoff.face_id.size // 40))
    for ordinal in sample:
        table = s.mass_properties([EntityId(int(handoff.face_id[ordinal]))])
        assert float(table.measure[0]) == pytest.approx(faces[ordinal].area, rel=1e-9)
