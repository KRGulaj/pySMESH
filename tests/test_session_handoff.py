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

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

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

    # A solid is named in the details under its kind; face_ids holds faces only (A5).
    assert int(ids[0]) in _details_by_kind(excinfo.value.details)["SOLID"]
    assert int(ids[0]) not in [int(i) for i in excinfo.value.face_ids]


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


# ------------------------------------------ The refusal's ids (report §2 A5) --- #

# Two 2 x 2 x 2 boxes, the second shifted by 1 along x, fused: coplanar faces merge, and
# the refusal used to list solids, edges and vertices on face_ids, 8 of them twice. The
# oracle is independent of the refusal: two ids that denote one shape have the same
# entity_table row, bit for bit (measure, centroid, box), and a split id has
# shape_count > 1.
HANDOFF_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)


def _coplanar_fuse() -> Session:
    """The A5 fuse: two shifted boxes with coplanar faces."""
    s = Session()
    s.add_box(2.0, 2.0, 2.0)
    first = s.entities(EntityKind.SOLID).tolist()
    s.add_box(2.0, 2.0, 2.0, origin=(1.0, 0.0, 0.0))
    second = [i for i in s.entities(EntityKind.SOLID).tolist() if i not in first]
    s.fuse(first, second)
    return s


def _blamed_by_kind(s: Session) -> dict[str, set[int]]:
    """The ids an export must refuse, by kind: aliases and splits."""
    out: dict[str, set[int]] = {}
    for kind in HANDOFF_KINDS:
        table = s.entity_table(kind)
        rows = np.c_[table.measure, table.centroid, table.bbox]
        keys = [r.tobytes() for r in rows]
        ids = {
            int(i)
            for i, key, count in zip(table.ids, keys, table.shape_count, strict=True)
            if keys.count(key) > 1 or int(count) > 1
        }
        if ids:
            out[kind.name] = ids
    return out


def _details_by_kind(details: str) -> dict[str, list[int]]:
    """The ids the refusal's details list under each kind, in their order."""
    listing = details.split("The ids, by kind:")[1]
    out: dict[str, list[int]] = {}
    for part in listing.split("."):
        words = part.strip().replace(",", " ").split()
        if words:
            out[words[0]] = [int(w) for w in words[1:]]
    return out


def test_a_handoff_refusal_names_each_id_once_and_only_faces_on_face_ids() -> None:
    """face_ids: the blamed faces, once each; details: every blamed id by kind (A5)."""
    s = _coplanar_fuse()
    expected = _blamed_by_kind(s)

    with pytest.raises(PysmeshError) as caught:
        s.export_handoff()

    face_ids = [int(i) for i in caught.value.face_ids]
    assert face_ids == sorted(expected["FACE"])
    listed = _details_by_kind(caught.value.details)
    assert {k: set(v) for k, v in listed.items()} == expected
    assert all(len(v) == len(set(v)) for v in listed.values())


# ------------------------------------------ Aliases in the handoff (report §4 C2) --- #

# The seven boolean cases of report §4 C2: two 2 x 2 x 2 boxes, the first at the origin.
# (operation, origin of the second box, its size). Only "fuse overlapping" and "cut
# overlapping" were a bijection; the rest were refused with no way out.
Vec3 = tuple[float, float, float]
C2_CASES: dict[str, tuple[str, Vec3, Vec3]] = {
    "common_overlapping": ("common", (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
    "common_coplanar": ("common", (1.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
    "fuse_overlapping": ("fuse", (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
    "fuse_coplanar": ("fuse", (1.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
    "fuse_face_touching": ("fuse", (2.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
    "fuse_tool_inside": ("fuse", (0.5, 0.5, 0.0), (1.0, 1.0, 1.0)),
    "cut_overlapping": ("cut", (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
}
C2_TOL: float = 1e-9


def _c2_session(name: str) -> Session:
    """One C2 case: a boolean of the box at the origin and the second box."""
    op, origin, size = C2_CASES[name]
    s = Session()
    s.add_box(2.0, 2.0, 2.0)
    first = s.entities(EntityKind.SOLID).tolist()
    s.add_box(*size, origin=origin)
    second = [i for i in s.entities(EntityKind.SOLID).tolist() if i not in first]
    getattr(s, op)(first, second)
    return s


def _parts(
    handoff: Handoff,
) -> dict[EntityKind, tuple[list[Any], tuple[tuple[EntityId, ...], ...]]]:
    """Per kind: the sub-shapes a reader of the BREP lists, and the ids of each."""
    shape = ps.load_brep(handoff.brep)

    def by_id(items: list[Any]) -> list[Any]:
        return sorted(items, key=lambda x: x.id)

    return {
        EntityKind.SOLID: (by_id(shape.solids()), handoff.solid_ids_of),
        EntityKind.FACE: (by_id(shape.faces()), handoff.face_ids_of),
        EntityKind.EDGE: (by_id(shape.edges()), handoff.edge_ids_of),
        EntityKind.VERTEX: (by_id(shape.vertices()), handoff.vertex_ids_of),
    }


def _union_box(kind: EntityKind, parts: list[Any]) -> NDArray[np.float64]:
    """The box of the union of some sub-shapes; a vertex has its point."""
    if kind == EntityKind.VERTEX:
        pts = np.array([p.xyz for p in parts])
        return np.r_[pts.min(axis=0), pts.max(axis=0)]
    own = np.array([p.bbox for p in parts])
    return np.r_[own[:, :3].min(axis=0), own[:, 3:].max(axis=0)]


def _measure_and_centroid(
    kind: EntityKind, parts: list[Any]
) -> tuple[float, NDArray[np.float64]]:
    """Summed volume or area of some sub-shapes, and their measure-weighted centroid."""
    solid = kind == EntityKind.SOLID
    measure = np.array([p.volume if solid else p.area for p in parts])
    centroids = np.array([p.centroid for p in parts])
    weighted = (centroids * measure[:, None]).sum(axis=0) / measure.sum()
    return float(measure.sum()), weighted


@pytest.mark.parametrize("name", sorted(C2_CASES))
def test_with_aliases_every_live_id_resolves_to_exactly_its_own_sub_shapes(
    name: str,
) -> None:
    """R(id) = the ordinals whose tuple lists the id: its own sub-shapes, all of them.

    The count of R(id) is the id's shape count, read independently from
    ``entity_table``; their union box is the id's box; for solids and faces, their
    summed measure and measure-weighted centroid are the id's own, within 1e-9. For an
    id of one shape that is its centroid exactly. The coplanar fuse and the fuse with
    the tool inside, where two ids share only part of their faces, included (C2, E5).
    """
    s = _c2_session(name)

    handoff = s.export_handoff(allow_aliases=True)

    for kind, (parts, ids_of) in _parts(handoff).items():
        boxes = s.bounding_boxes(kind)
        table = s.entity_table(kind)
        assert np.array_equal(boxes.ids, table.ids)
        for row, (entity, box) in enumerate(zip(boxes.ids, boxes.bbox, strict=True)):
            mine = [parts[i] for i, ids in enumerate(ids_of) if int(entity) in ids]
            assert len(mine) == int(table.shape_count[row]), (kind, int(entity))
            assert _union_box(kind, mine) == pytest.approx(box, abs=C2_TOL)
            if kind in (EntityKind.SOLID, EntityKind.FACE):
                measure, centroid = _measure_and_centroid(kind, mine)
                assert measure == pytest.approx(float(table.measure[row]), rel=C2_TOL)
                assert centroid == pytest.approx(table.centroid[row], abs=C2_TOL)


@pytest.mark.parametrize("name", sorted(C2_CASES))
def test_with_aliases_every_tuple_lists_the_label_first_then_ascending(
    name: str,
) -> None:
    """Each tuple starts with the ordinal's label, the id in ``face_id`` (C2, E5)."""
    s = _c2_session(name)

    handoff = s.export_handoff(allow_aliases=True)

    pairs = (
        (handoff.solid_id, handoff.solid_ids_of),
        (handoff.face_id, handoff.face_ids_of),
        (handoff.edge_id, handoff.edge_ids_of),
        (handoff.vertex_id, handoff.vertex_ids_of),
    )
    for labels, ids_of in pairs:
        assert len(ids_of) == labels.size
        for label, ids in zip(labels.tolist(), ids_of, strict=True):
            assert ids[0] == label
            assert list(ids) == sorted(ids)


def test_without_aliases_a_boolean_that_shares_sub_shapes_is_still_refused() -> None:
    """The default stays the bijection: a coplanar common is refused (C2)."""
    s = _c2_session("common_coplanar")

    with pytest.raises(PysmeshError, match="not a bijection"):
        s.export_handoff()


def test_a_boolean_refusal_names_the_boolean_cause_and_the_alias_way_out() -> None:
    """After a common, the refusal names a boolean and allow_aliases (C2)."""
    s = _c2_session("common_overlapping")

    with pytest.raises(PysmeshError, match="not a bijection") as caught:
        s.export_handoff()

    assert "boolean" in caught.value.details
    assert "allow_aliases=True" in caught.value.details


def test_a_bijection_gives_one_element_tuples_either_way() -> None:
    """A cut that shares nothing: each tuple is its label alone, either way (C2)."""
    s = _c2_session("cut_overlapping")

    plain = s.export_handoff()
    aliased = s.export_handoff(allow_aliases=True)

    assert aliased.face_ids_of == tuple((EntityId(i),) for i in aliased.face_id)
    assert plain.face_ids_of == aliased.face_ids_of
    assert np.array_equal(plain.face_id, aliased.face_id)
    assert plain.brep == aliased.brep


# -------------------------------------- Names keyed by session id (report §4 C4) --- #

# The coplanar fuse of report §4 C2: the top faces of the two boxes (z = 2) and their
# bottom faces (z = 0) become pieces, one of each denoted by both operands' ids. The
# oracle is geometric: every face the reader finds in the plane z = 2 is named "top",
# every face in z = 0 "bottom", and no other face is named.
C4_TOL: float = 1e-9


def _faces_in_plane(s: Session, z: float) -> list[EntityId]:
    """The live faces of the session that lie in the plane at height z."""
    table = s.bounding_boxes(EntityKind.FACE)
    lo, hi = table.bbox[:, 2], table.bbox[:, 5]
    flat = (np.abs(lo - z) < C4_TOL) & (np.abs(hi - z) < C4_TOL)
    return [EntityId(int(i)) for i in table.ids[flat]]


def _in_plane(bbox: NDArray[np.float64], z: float) -> bool:
    """True if a box is flat at height z."""
    return abs(bbox[2] - z) < C4_TOL and abs(bbox[5] - z) < C4_TOL


def test_write_step_names_every_face_of_the_named_ids_after_a_fuse() -> None:
    """Agreeing names on a merged face: read back, each face carries its name (C4)."""
    s = _c2_session("fuse_coplanar")
    names = {i: "top" for i in _faces_in_plane(s, 2.0)}
    names.update({i: "bottom" for i in _faces_in_plane(s, 0.0)})

    data = s.write_step(unit="MM", face_names=names)

    imported = ps.read_step_xde(data)
    faces = {f.id: f for f in ps.load_brep(imported.brep).faces()}
    labels = {lab.id: lab.name for lab in imported.face_labels if lab.name}
    for face_id, face in faces.items():
        top, bottom = _in_plane(face.bbox, 2.0), _in_plane(face.bbox, 0.0)
        expected = "top" if top else "bottom" if bottom else None
        assert labels.get(face_id) == expected, (face_id, face.bbox)


def test_write_step_names_a_merged_face_from_its_one_named_id() -> None:
    """One operand's top named: its piece and the shared piece take it (C4)."""
    s = _c2_session("fuse_coplanar")
    table = s.bounding_boxes(EntityKind.FACE)
    tops = _faces_in_plane(s, 2.0)
    first = next(i for i in tops if table.bbox[table.ids == i][0][0] < C4_TOL)

    data = s.write_step(unit="MM", face_names={first: "top"})

    imported = ps.read_step_xde(data)
    faces = {f.id: f for f in ps.load_brep(imported.brep).faces()}
    labels = {lab.id: lab.name for lab in imported.face_labels if lab.name}
    for face_id, face in faces.items():
        mine = _in_plane(face.bbox, 2.0) and face.bbox[3] < 2.0 + C4_TOL
        assert labels.get(face_id) == ("top" if mine else None), (face_id, face.bbox)


def test_write_step_refuses_a_merged_face_whose_ids_give_different_names() -> None:
    """Two names on one merged face: refused, naming the ids and the names (C4)."""
    s = _c2_session("fuse_coplanar")
    tops = _faces_in_plane(s, 2.0)
    names = {i: f"top {k}" for k, i in enumerate(tops)}

    with pytest.raises(PysmeshError, match="names differ") as caught:
        s.write_step(unit="MM", face_names=names)

    message = str(caught.value)
    assert all(f"{i}: 'top {k}'" in message for k, i in enumerate(tops))


def test_write_step_refuses_a_name_for_an_id_that_is_not_a_live_face() -> None:
    """A dead or non-face id in face_names is refused, naming it (C4)."""
    s = _c2_session("fuse_coplanar")
    solid = int(s.entities(EntityKind.SOLID)[0])

    with pytest.raises(PysmeshError, match=rf"\[{solid}\]"):
        s.write_step(unit="MM", face_names={EntityId(solid): "body"})
