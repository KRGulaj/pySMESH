"""Gates for the session's render mesh and its incremental delta.

Five claims are under test, and they are the render-mesh contract rather than a tessellator's
usual "does it produce triangles":

* **The delta is exact, both directions.** After an operation, the set of faces the call
  reports as changed must equal the set whose emitted nodes actually differ — checked against
  a per-face node-block diff that the session did not produce. Over-reporting makes a
  consumer rebuild everything and silently costs the whole feature; under-reporting leaves a
  stale render. Both are asserted, and both are shown to be detectable.
* **The work is O(faces touched).** Counted by faces re-triangulated, never by wall clock.
* **Shared-edge nodes are bitwise equal, and interior nodes are not merged.** A consumer that
  welds by exact position needs to know which of the two it is handed, so both directions are
  asserted rather than one.
* **Every edge has a polyline and every vertex a point**, from the same call, indexed into
  the same nodes and labelled with the same id space as the triangles.
* **Tessellating is not an operation.** No id is issued and no counter advances.

The two-body fixture is what makes the O(k) claim measurable: an operation on one body leaves
the other's faces untouched by construction, so a delta naming any of them is a demonstrable
over-report rather than a judgement call.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from pysmesh import EntityId, EntityKind, PysmeshError, RenderMesh, Session

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0

# Far enough apart that a boolean on one cannot reach the other.
FAR_X: float = 20.0

# A bar that crosses one box completely, so the faces it meets are genuinely rebuilt.
BAR_ORIGIN: tuple[float, float, float] = (1.0, -1.0, 4.0)
BAR_SIZE: tuple[float, float, float] = (1.0, 9.0, 1.0)

COARSE: float = 0.5
FINE: float = 0.05

# Deflections that straddle the point where the chord criterion starts to bind on the
# cylinder fixture. Above CHORD_COARSE the default angular criterion dominates and asking for
# less deflection changes nothing, which would make a refinement test pass vacuously.
CHORD_COARSE: float = 0.1
CHORD_FINE: float = 0.005

CYL_RADIUS: float = 4.0
CYL_HEIGHT: float = 9.0
SPHERE_RADIUS: float = 5.0

# The real assembly is modelled in metres with parts of about 0.003 cubic metres, so a
# display-scale deflection there is three orders of magnitude below the synthetic fixtures'.
ASSEMBLY_DEFLECTION: float = 0.001
ASSEMBLY_ANGLE_DEG: float = 20.0


# ---- helpers -------------------------------------------------------------------------- #


def ids_of(session: Session, kind: EntityKind) -> set[int]:
    """Live entity ids of one kind."""
    return {int(i) for i in session.entities(kind)}


def node_blocks(mesh: RenderMesh) -> dict[int, tuple[bytes, ...]]:
    """Face id -> its node block(s), as raw bytes so the comparison is exact.

    An oracle the session did not produce: it reads the emitted coordinates back rather than
    asking which faces the session believes it rebuilt. A split id contributes one block per
    piece, sorted so the comparison does not depend on traversal order.
    """
    out: dict[int, list[bytes]] = {}
    for row, face_id in enumerate(mesh.face_id):
        lo, hi = mesh.face_node_range[row]
        out.setdefault(int(face_id), []).append(mesh.nodes[lo:hi].tobytes())
    return {k: tuple(sorted(v)) for k, v in out.items()}


def changed_by_geometry(before: RenderMesh, after: RenderMesh) -> set[int]:
    """Faces whose emitted nodes differ between two meshes, measured from the arrays."""
    a, b = node_blocks(before), node_blocks(after)
    return {fid for fid, blk in b.items() if fid not in a or a[fid] != blk}


def face_of_node(mesh: RenderMesh, node: int) -> int:
    """The ``face_id`` row whose node range contains ``node``, or -1 for a free-edge node."""
    for row in range(len(mesh.face_id)):
        lo, hi = mesh.face_node_range[row]
        if lo <= node < hi:
            return row
    return -1


# ---- fixtures ------------------------------------------------------------------------- #


@pytest.fixture
def box() -> Session:
    """One 3 x 7 x 11 box."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    return s


@pytest.fixture
def two_boxes() -> Session:
    """Two disjoint boxes: an operation on one provably cannot touch the other."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(FAR_X, 0.0, 0.0))
    return s


def bar_through_first_box(session: Session) -> EntityId:
    """Fuse a bar clean through the first box, and return the target solid's id."""
    solids = sorted(ids_of(session, EntityKind.SOLID))
    target = EntityId(solids[0])
    before = ids_of(session, EntityKind.SOLID)
    session.add_box(*BAR_SIZE, origin=BAR_ORIGIN)
    bar = EntityId(next(iter(ids_of(session, EntityKind.SOLID) - before)))
    session.fuse([target], [bar])
    return target


# ---- the arrays exist and agree with each other --------------------------------------- #


def test_a_box_tessellates_to_twelve_triangles_over_twenty_four_nodes(box: Session) -> None:
    mesh = box.tessellate(deflection=COARSE)

    assert mesh.tris.shape == (12, 3)
    assert mesh.nodes.shape == (24, 3)
    assert mesh.normals.shape == mesh.nodes.shape
    assert len(mesh.face_id) == 6


def test_every_array_has_the_dtype_the_contract_states(box: Session) -> None:
    mesh = box.tessellate(deflection=COARSE)

    assert mesh.nodes.dtype == np.float64
    assert mesh.normals.dtype == np.float64
    assert mesh.vertex_xyz.dtype == np.float64
    assert mesh.tris.dtype == np.int32
    assert mesh.edge_lines.dtype == np.int32
    assert mesh.tri_face_id.dtype == np.int64
    assert mesh.edge_id.dtype == np.int64
    assert mesh.vertex_id.dtype == np.int64


def test_every_triangle_index_lands_inside_its_own_face_node_range(box: Session) -> None:
    mesh = box.tessellate(deflection=COARSE)

    for row in range(len(mesh.face_id)):
        node_lo, node_hi = mesh.face_node_range[row]
        tri_lo, tri_hi = mesh.face_tri_range[row]
        block = mesh.tris[tri_lo:tri_hi]

        assert block.min() >= node_lo
        assert block.max() < node_hi


def test_triangle_face_ids_agree_with_the_per_face_ranges(box: Session) -> None:
    mesh = box.tessellate(deflection=COARSE)

    for row, face_id in enumerate(mesh.face_id):
        lo, hi = mesh.face_tri_range[row]

        assert set(mesh.tri_face_id[lo:hi].tolist()) == {int(face_id)}


def test_box_normals_point_outward_on_all_six_faces(box: Session) -> None:
    mesh = box.tessellate(deflection=COARSE)
    centre = np.array([BOX_DX, BOX_DY, BOX_DZ]) / 2.0

    outward = np.einsum("ij,ij->i", mesh.normals, mesh.nodes - centre)

    assert np.all(outward > 0.0)
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)


def test_curved_faces_get_more_nodes_at_a_finer_deflection() -> None:
    coarse, fine = Session(), Session()
    coarse.add_cylinder(CYL_RADIUS, CYL_HEIGHT)
    fine.add_cylinder(CYL_RADIUS, CYL_HEIGHT)

    a = coarse.tessellate(deflection=CHORD_COARSE)
    b = fine.tessellate(deflection=CHORD_FINE)

    assert b.nodes.shape[0] > 2 * a.nodes.shape[0]


def test_a_sphere_tessellates_close_to_its_analytic_area() -> None:
    s = Session()
    s.add_sphere(SPHERE_RADIUS)

    mesh = s.tessellate(deflection=0.01)
    corners = mesh.nodes[mesh.tris]
    area = float(
        np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
        ).sum()
        / 2.0
    )

    assert area == pytest.approx(4.0 * math.pi * SPHERE_RADIUS**2, rel=2e-3)


# ---- (d) shared-edge nodes are bitwise equal, interior nodes are not merged ------------ #


def test_three_faces_meeting_at_a_corner_place_bitwise_identical_nodes(box: Session) -> None:
    """A box corner is shared by exactly three faces, and their nodes must agree exactly."""
    mesh = box.tessellate(deflection=COARSE)
    corner = np.array([0.0, 0.0, 0.0])

    at_corner = np.flatnonzero(np.all(mesh.nodes == corner, axis=1))
    faces = {face_of_node(mesh, int(n)) for n in at_corner}

    assert len(at_corner) == 3
    assert len(faces) == 3


def test_every_shared_edge_node_pair_is_bitwise_equal_not_merely_close(box: Session) -> None:
    """Both adjacent faces read one edge discretisation, so the positions agree exactly."""
    mesh = box.tessellate(deflection=COARSE)

    unique, counts = np.unique(mesh.nodes, axis=0, return_counts=True)
    coincident = int(counts[counts > 1].sum())

    # A box's 24 nodes are its 8 corners, each contributed by 3 faces: every one of them is
    # part of a coincident group, and the groups are exact because np.unique compares bitwise.
    assert coincident == 24
    assert len(unique) == 8


def test_the_mesh_is_not_welded_so_each_face_keeps_its_own_nodes(box: Session) -> None:
    """The other direction: coincident nodes are kept, not collapsed to one."""
    mesh = box.tessellate(deflection=COARSE)

    assert mesh.nodes.shape[0] == 24
    assert sum(int(hi - lo) for lo, hi in mesh.face_node_range) == 24


def test_no_two_nodes_are_close_without_being_bitwise_equal() -> None:
    """The one outcome a consumer cannot use: an approximately-coincident seam node.

    Welding by exact position matches a seam pair only when the two positions are bitwise
    equal. A pair that is merely within a tolerance would be silently left unwelded and would
    tear the surface, so the claim under test is that no such pair exists at all — not that
    the pairs are close.
    """
    s = Session()
    s.add_cylinder(CYL_RADIUS, CYL_HEIGHT)
    mesh = s.tessellate(deflection=FINE)

    separation = np.linalg.norm(mesh.nodes[:, None, :] - mesh.nodes[None, :, :], axis=2)
    identical = np.all(mesh.nodes[:, None, :] == mesh.nodes[None, :, :], axis=2)
    close_but_not_equal = (separation < 1e-9) & ~identical

    assert not close_but_not_equal.any()
    assert identical.sum() > mesh.nodes.shape[0], "some pair must coincide at all"


def test_interior_nodes_of_adjacent_faces_are_never_merged() -> None:
    """A cylinder's wall and cap share an edge; nothing off that edge may coincide."""
    s = Session()
    s.add_cylinder(CYL_RADIUS, CYL_HEIGHT)
    mesh = s.tessellate(deflection=FINE)

    _, counts = np.unique(mesh.nodes, axis=0, return_counts=True)
    coincident = mesh.nodes[
        np.isin(
            np.unique(mesh.nodes, axis=0, return_inverse=True)[1],
            np.flatnonzero(counts > 1),
        )
    ]

    # Every coincident position lies on a cap plane, which is where the shared edges are.
    on_a_cap = np.isclose(coincident[:, 2], 0.0) | np.isclose(coincident[:, 2], CYL_HEIGHT)
    assert np.all(on_a_cap)


# ---- (e) every edge has a polyline and every vertex a point ---------------------------- #


def test_every_edge_of_a_solid_carries_a_polyline(box: Session) -> None:
    mesh = box.tessellate(deflection=COARSE)

    assert {int(i) for i in mesh.edge_id} == ids_of(box, EntityKind.EDGE)


def test_every_vertex_of_a_solid_carries_a_point(box: Session) -> None:
    mesh = box.tessellate(deflection=COARSE)

    assert {int(i) for i in mesh.vertex_id} == ids_of(box, EntityKind.VERTEX)
    assert mesh.vertex_xyz.shape == (8, 3)


def test_edge_and_vertex_coverage_holds_on_a_curved_model() -> None:
    s = Session()
    s.add_cylinder(CYL_RADIUS, CYL_HEIGHT)
    s.add_sphere(SPHERE_RADIUS, centre=(FAR_X, 0.0, 0.0))
    mesh = s.tessellate(deflection=FINE)

    assert {int(i) for i in mesh.edge_id} == ids_of(s, EntityKind.EDGE)
    assert {int(i) for i in mesh.vertex_id} == ids_of(s, EntityKind.VERTEX)


def test_polyline_segments_index_real_nodes_and_lie_on_the_geometry(box: Session) -> None:
    mesh = box.tessellate(deflection=COARSE)
    a = mesh.nodes[mesh.edge_lines[:, 0]]
    b = mesh.nodes[mesh.edge_lines[:, 1]]

    lengths = np.linalg.norm(b - a, axis=1)

    assert mesh.edge_lines.min() >= 0
    assert mesh.edge_lines.max() < mesh.nodes.shape[0]
    # A box edge is one straight segment, so the polylines are the twelve edge lengths.
    assert sorted(np.round(lengths, 9)) == sorted([BOX_DX] * 4 + [BOX_DY] * 4 + [BOX_DZ] * 4)


def test_polyline_nodes_are_the_face_nodes_not_a_second_discretisation(box: Session) -> None:
    """An edge on a face costs index pairs and nothing else."""
    mesh = box.tessellate(deflection=COARSE)

    assert all(face_of_node(mesh, int(n)) >= 0 for n in mesh.edge_lines.ravel())


def test_a_free_edge_gets_a_polyline_of_its_own_nodes() -> None:
    """An edge bounding no face has no triangulation to index, so its points are appended."""
    s = Session()
    s.add_line((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    mesh = s.tessellate(deflection=COARSE)
    free = [int(n) for n in mesh.edge_lines.ravel() if face_of_node(mesh, int(n)) < 0]

    assert {int(i) for i in mesh.edge_id} == ids_of(s, EntityKind.EDGE)
    assert mesh.nodes.shape[0] == 26
    assert len(free) == 2


def test_a_wire_body_with_no_faces_still_yields_polylines_and_points() -> None:
    s = Session()
    s.add_polyline([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 5.0, 0.0)])

    mesh = s.tessellate(deflection=COARSE)

    assert mesh.tris.shape == (0, 3)
    assert {int(i) for i in mesh.edge_id} == ids_of(s, EntityKind.EDGE)
    assert {int(i) for i in mesh.vertex_id} == ids_of(s, EntityKind.VERTEX)


# ---- (c) every id comes from the one global entity space ------------------------------- #


def test_triangle_edge_and_vertex_ids_are_all_session_entity_ids(box: Session) -> None:
    mesh = box.tessellate(deflection=COARSE)

    assert {int(i) for i in mesh.tri_face_id} <= ids_of(box, EntityKind.FACE)
    assert {int(i) for i in mesh.edge_id} <= ids_of(box, EntityKind.EDGE)
    assert {int(i) for i in mesh.vertex_id} <= ids_of(box, EntityKind.VERTEX)


def test_the_three_id_arrays_never_collide(box: Session) -> None:
    """One namespace across dimensions: a face id is never also an edge or vertex id."""
    mesh = box.tessellate(deflection=COARSE)
    faces = {int(i) for i in mesh.tri_face_id}
    edges = {int(i) for i in mesh.edge_id}
    verts = {int(i) for i in mesh.vertex_id}

    assert faces & edges == set()
    assert faces & verts == set()
    assert edges & verts == set()


def test_a_face_id_from_the_mesh_resolves_through_the_registry(box: Session) -> None:
    mesh = box.tessellate(deflection=COARSE)
    face_id = EntityId(int(mesh.tri_face_id[0]))

    assert box.entity_kind(face_id) == EntityKind.FACE
    assert box.is_alive(face_id)


def test_an_edge_id_from_the_mesh_bounds_the_face_beside_it(box: Session) -> None:
    """The point of one namespace: a picked edge and its faces need no (dim, id) pair."""
    mesh = box.tessellate(deflection=COARSE)
    pairs = box.adjacency(EntityKind.FACE, EntityKind.EDGE)
    edge_id = int(mesh.edge_id[0])

    owners = {int(f) for f, e in zip(pairs.ids, pairs.related) if int(e) == edge_id}

    assert len(owners) == 2
    assert owners <= {int(i) for i in mesh.face_id}


# ---- (b) the delta is exact, and the work is O(faces touched) -------------------------- #


def test_a_first_tessellation_reports_every_face(box: Session) -> None:
    mesh = box.tessellate(deflection=COARSE)

    assert {int(i) for i in mesh.retriangulated} == ids_of(box, EntityKind.FACE)
    assert {int(i) for i in mesh.changed} == ids_of(box, EntityKind.FACE)


def test_tessellating_twice_with_no_operation_between_reports_nothing(box: Session) -> None:
    """The strongest over-report check available, and it needs no oracle."""
    box.tessellate(deflection=COARSE)
    again = box.tessellate(deflection=COARSE)

    assert again.retriangulated.size == 0
    assert again.changed.size == 0


def test_the_second_tessellation_returns_the_same_arrays_as_the_first(box: Session) -> None:
    """Nothing was rebuilt, so nothing may have moved either."""
    first = box.tessellate(deflection=COARSE)
    second = box.tessellate(deflection=COARSE)

    assert np.array_equal(first.nodes, second.nodes)
    assert np.array_equal(first.tris, second.tris)
    assert np.array_equal(first.tri_face_id, second.tri_face_id)


def test_a_non_incremental_call_rebuilds_everything(box: Session) -> None:
    """The opposite direction, so the incremental path cannot absorb the full one."""
    box.tessellate(deflection=COARSE)

    full = box.tessellate(deflection=COARSE, incremental=False)

    assert {int(i) for i in full.retriangulated} == ids_of(box, EntityKind.FACE)


def test_an_operation_on_one_body_leaves_the_other_body_untouched(
    two_boxes: Session,
) -> None:
    """Gate (b): O(faces touched), counted by re-triangulated faces, never by wall clock."""
    untouched = sorted(ids_of(two_boxes, EntityKind.SOLID))[1]
    before = two_boxes.tessellate(deflection=COARSE)
    far_faces = {
        int(f)
        for f, s in zip(
            two_boxes.adjacency(EntityKind.FACE, EntityKind.SOLID).ids,
            two_boxes.adjacency(EntityKind.FACE, EntityKind.SOLID).related,
        )
        if int(s) == untouched
    }

    bar_through_first_box(two_boxes)
    after = two_boxes.tessellate(deflection=COARSE)

    assert before.retriangulated.size == 12
    assert after.retriangulated.size < 12
    assert {int(i) for i in after.retriangulated} & far_faces == set()


def test_the_reported_delta_equals_the_faces_whose_nodes_actually_changed(
    two_boxes: Session,
) -> None:
    """Both failure directions at once, against an oracle read back from the arrays."""
    before = two_boxes.tessellate(deflection=COARSE)
    bar_through_first_box(two_boxes)
    after = two_boxes.tessellate(deflection=COARSE)

    reported = {int(i) for i in after.changed}
    measured = changed_by_geometry(before, after)

    assert reported - measured == set(), "over-reported: consumer rebuilds what did not move"
    assert measured - reported == set(), "under-reported: consumer keeps a stale render"


def test_both_delta_assertions_are_shown_to_bite(two_boxes: Session) -> None:
    """Falsification. A check that has never been made to fail is a claim, not a check.

    The two assertions above are re-run against a deliberately broken delta in each
    direction, so a future refactor cannot leave one of them vacuously true.
    """
    before = two_boxes.tessellate(deflection=COARSE)
    bar_through_first_box(two_boxes)
    after = two_boxes.tessellate(deflection=COARSE)
    measured = changed_by_geometry(before, after)
    every_face = ids_of(two_boxes, EntityKind.FACE)

    over_reporting = every_face
    under_reporting: set[int] = set()

    assert measured != set(), "the operation must actually change something"
    assert measured != every_face, "some face must be left alone, or over-reporting is free"
    assert over_reporting - measured != set(), "the over-report assertion would fire"
    assert measured - under_reporting != set(), "the under-report assertion would fire"


def test_a_face_created_by_the_operation_is_always_in_the_delta(
    two_boxes: Session,
) -> None:
    """Under-reporting's clearest case: a new face has no triangulation to keep."""
    two_boxes.tessellate(deflection=COARSE)
    faces_before = ids_of(two_boxes, EntityKind.FACE)

    bar_through_first_box(two_boxes)
    after = two_boxes.tessellate(deflection=COARSE)
    fresh = ids_of(two_boxes, EntityKind.FACE) - faces_before

    assert fresh != set()
    assert fresh <= {int(i) for i in after.retriangulated}


def test_a_relocated_face_is_reported_changed_but_not_retriangulated(box: Session) -> None:
    """The case that separates the two answers, and the one a single array would get wrong.

    A rigid transform changes only the shape's location, so ``BRepMesh`` keeps every
    triangulation — and every node the mesh emits still moves. Reporting nothing would leave
    a consumer with a stale render at the old position.
    """
    first = box.tessellate(deflection=COARSE)
    box.translate((100.0, 0.0, 0.0))

    after = box.tessellate(deflection=COARSE)

    assert after.retriangulated.size == 0
    assert {int(i) for i in after.changed} == ids_of(box, EntityKind.FACE)
    assert after.nodes[:, 0].min() == pytest.approx(first.nodes[:, 0].min() + 100.0)


def test_a_relocation_moves_the_nodes_rigidly(box: Session) -> None:
    first = box.tessellate(deflection=COARSE)
    box.translate((100.0, 0.0, 0.0))

    after = box.tessellate(deflection=COARSE)

    assert np.allclose(after.nodes - first.nodes, [100.0, 0.0, 0.0])
    assert np.array_equal(after.tris, first.tris)


def test_a_finer_deflection_re_triangulates_a_curved_face() -> None:
    s = Session()
    s.add_cylinder(CYL_RADIUS, CYL_HEIGHT)
    coarse = s.tessellate(deflection=CHORD_COARSE)

    finer = s.tessellate(deflection=CHORD_FINE)

    assert finer.retriangulated.size > 0
    assert finer.nodes.shape[0] > coarse.nodes.shape[0]


def test_a_coarser_request_is_ignored_unless_the_call_is_non_incremental() -> None:
    """OCCT will not lower the quality of a triangulation it already has.

    Pinned because it is invisible from the arrays alone: asking for a coarser mesh and
    getting the finer one back looks like the request was honoured.
    """
    s = Session()
    s.add_cylinder(CYL_RADIUS, CYL_HEIGHT)
    fine = s.tessellate(deflection=CHORD_FINE)

    kept = s.tessellate(deflection=CHORD_COARSE)
    forced = s.tessellate(deflection=CHORD_COARSE, incremental=False)

    assert kept.nodes.shape[0] == fine.nodes.shape[0]
    assert kept.retriangulated.size == 0
    assert forced.nodes.shape[0] < fine.nodes.shape[0]


def test_removing_a_body_drops_its_faces_from_the_mesh(two_boxes: Session) -> None:
    two_boxes.tessellate(deflection=COARSE)
    doomed = EntityId(sorted(ids_of(two_boxes, EntityKind.SOLID))[1])

    two_boxes.remove([doomed])
    after = two_boxes.tessellate(deflection=COARSE)

    assert len(after.face_id) == 6
    assert after.changed.size == 0


def test_a_restored_snapshot_keeps_its_cached_mesh(box: Session) -> None:
    """A snapshot shares the shape, so it shares the triangulation cached on it."""
    mark = box.snapshot()
    box.tessellate(deflection=COARSE)
    box.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(FAR_X, 0.0, 0.0))
    box.restore(mark)

    after = box.tessellate(deflection=COARSE)

    assert after.retriangulated.size == 0
    assert len(after.face_id) == 6


# ---- tessellating is not an operation --------------------------------------------------- #


def test_tessellating_issues_no_id_and_advances_no_counter(box: Session) -> None:
    ops, issued, state = box.op_count, box.issued_id_count, box.state_op_index

    box.tessellate(deflection=COARSE)
    box.tessellate(deflection=COARSE, incremental=False)

    assert box.op_count == ops
    assert box.issued_id_count == issued
    assert box.state_op_index == state


def test_tessellating_does_not_disturb_the_entity_table(box: Session) -> None:
    before = box.entity_table(EntityKind.FACE)

    box.tessellate(deflection=COARSE)
    after = box.entity_table(EntityKind.FACE)

    assert np.array_equal(before.ids, after.ids)
    assert np.allclose(before.measure, after.measure)


def test_two_sessions_tessellate_without_cross_talk() -> None:
    a, b = Session(), Session()
    a.add_box(BOX_DX, BOX_DY, BOX_DZ)
    b.add_cylinder(CYL_RADIUS, CYL_HEIGHT)

    a.tessellate(deflection=COARSE)
    mesh_b = b.tessellate(deflection=COARSE)
    again_a = a.tessellate(deflection=COARSE)

    assert again_a.changed.size == 0
    assert {int(i) for i in mesh_b.face_id} == ids_of(b, EntityKind.FACE)


# ---- parameters ------------------------------------------------------------------------ #


def test_an_empty_session_tessellates_to_empty_arrays() -> None:
    mesh = Session().tessellate(deflection=COARSE)

    assert mesh.nodes.shape == (0, 3)
    assert mesh.tris.shape == (0, 3)
    assert mesh.edge_lines.shape == (0, 2)
    assert mesh.face_id.size == 0
    assert mesh.changed.size == 0


@pytest.mark.parametrize("deflection", [0.0, -1.0, 1e-12])
def test_a_deflection_below_occts_floor_is_refused(box: Session, deflection: float) -> None:
    with pytest.raises(PysmeshError, match="deflection"):
        box.tessellate(deflection=deflection)


@pytest.mark.parametrize("angle_deg", [0.0, -5.0, 180.0, 400.0])
def test_an_angle_outside_the_open_half_turn_is_refused(
    box: Session, angle_deg: float
) -> None:
    with pytest.raises(PysmeshError, match="angle_deg"):
        box.tessellate(angle_deg=angle_deg)


def test_parallel_and_serial_meshes_are_identical(box: Session) -> None:
    """A parallel mesher that quietly differs is worse than a slow one."""
    serial = box.tessellate(deflection=FINE, parallel=False, incremental=False)
    parallel = box.tessellate(deflection=FINE, parallel=True, incremental=False)

    assert np.array_equal(serial.nodes, parallel.nodes)
    assert np.array_equal(serial.tris, parallel.tris)


def test_relative_mode_scales_the_deflection_to_each_edge() -> None:
    """Relative means per edge, not per model.

    The cylinder's rim is about 25 units long, so 0.01 relative asks for a deflection of
    about 0.25 there — far coarser than 0.01 absolute. The direction is the claim: a caller
    who reads "relative" as a fraction of the *model* would expect the opposite.
    """
    absolute, relative = Session(), Session()
    absolute.add_cylinder(CYL_RADIUS, CYL_HEIGHT)
    relative.add_cylinder(CYL_RADIUS, CYL_HEIGHT)

    a = absolute.tessellate(deflection=0.01)
    r = relative.tessellate(deflection=0.01, relative=True)

    assert r.tris.shape[0] > 0
    assert r.nodes.shape[0] < a.nodes.shape[0]


def test_a_finer_angle_refines_a_curved_face() -> None:
    coarse, fine = Session(), Session()
    coarse.add_cylinder(CYL_RADIUS, CYL_HEIGHT)
    fine.add_cylinder(CYL_RADIUS, CYL_HEIGHT)

    a = coarse.tessellate(deflection=1.0, angle_deg=30.0)
    b = fine.tessellate(deflection=1.0, angle_deg=5.0)

    assert b.nodes.shape[0] > a.nodes.shape[0]


# ---- the same claims at production scale ------------------------------------------------ #
#
# The synthetic fixtures make each property provable; a real assembly makes it useful. These
# need the git-ignored STEP and are marked slow, so they skip with a named reason rather than
# failing on a fresh checkout.


@pytest.fixture(scope="module")
def assembly(industrial_step_brep: bytes) -> Session:
    """The real industrial assembly, loaded into one session."""
    s = Session()
    s.add_brep(industrial_step_brep)
    return s


def bar_through_one_solid(session: Session) -> EntityId:
    """Fuse a bar through the first solid, sized from that solid's own bounding box.

    Every dimension is a fraction of the target's diagonal. An absolute size works on a
    3 x 7 x 11 box and silently engulfs a part modelled in metres, which leaves the boolean
    nothing to carry and the test asserting nothing.
    """
    table = session.entity_table(EntityKind.SOLID)
    target = EntityId(int(table.ids[0]))
    lo = table.bbox[0][:3]
    hi = table.bbox[0][3:]
    diagonal = float(np.linalg.norm(hi - lo))
    centre = (lo + hi) / 2.0
    before = ids_of(session, EntityKind.SOLID)
    session.add_box(
        diagonal * 0.1,
        diagonal * 0.1,
        diagonal * 2.0,
        origin=(
            centre[0] - diagonal * 0.05,
            centre[1] - diagonal * 0.05,
            lo[2] - diagonal * 0.5,
        ),
    )
    tool = EntityId(next(iter(ids_of(session, EntityKind.SOLID) - before)))
    session.fuse([target], [tool])
    return target


@pytest.mark.slow
def test_the_render_mesh_covers_every_edge_and_vertex_of_a_real_assembly(
    assembly: Session,
) -> None:
    """Gate (e) at scale: a tessellation returning triangles alone would not pass."""
    mesh = assembly.tessellate(deflection=ASSEMBLY_DEFLECTION)

    assert mesh.tris.shape[0] > 0
    assert {int(i) for i in mesh.edge_id} == ids_of(assembly, EntityKind.EDGE)
    assert {int(i) for i in mesh.vertex_id} == ids_of(assembly, EntityKind.VERTEX)
    assert {int(i) for i in mesh.tri_face_id} <= ids_of(assembly, EntityKind.FACE)


@pytest.mark.slow
def test_re_tessellation_after_one_operation_touches_a_small_fraction_of_a_real_assembly(
    industrial_step_brep: bytes,
) -> None:
    """Gate (b) at scale, counted in faces and cross-checked against the node blocks."""
    session = Session()
    session.add_brep(industrial_step_brep)
    before = session.tessellate(deflection=ASSEMBLY_DEFLECTION)
    total = len(before.face_id)

    bar_through_one_solid(session)
    after = session.tessellate(deflection=ASSEMBLY_DEFLECTION)

    reported = {int(i) for i in after.changed}
    measured = changed_by_geometry(before, after)

    assert reported == measured
    assert 0 < after.retriangulated.size < total // 20, (
        f"re-triangulated {after.retriangulated.size} of {total} faces"
    )


@pytest.mark.slow
def test_a_full_tessellation_beats_the_reference_mesher_on_a_real_assembly(
    industrial_step_path: str, industrial_step_brep: bytes
) -> None:
    """Gate (a): the ratio is the claim, so it is measured on one machine, both directions.

    The reference runs at its own defaults, which is the comparison the requirement names.
    That makes the test **conservative rather than matched**: at those defaults the reference
    produces a far coarser mesh than this call does, so it is being timed on less work. Both
    triangle counts are asserted into the failure message for exactly that reason — the ratio
    means nothing without them.

    Matching the two on visual quality instead — driving the reference at 360/20 elements per
    full turn, the same angular criterion used here — was tried and abandoned: it does not
    complete on this assembly within 420 s and passes 24 GB of resident memory. That is the
    honest reason the conservative comparison is the one in the suite.

    Import time is excluded on both sides: the comparison is between two surface meshers, not
    between two file readers.

    The reference package is not a dependency of this project and is not installed by CI, so
    the test skips by name when it is absent, exactly as the assembly fixture does.
    """
    gmsh = pytest.importorskip("gmsh", reason="reference mesher not installed")

    session = Session()
    session.add_brep(industrial_step_brep)
    start = time.perf_counter()
    mesh = session.tessellate(deflection=ASSEMBLY_DEFLECTION, angle_deg=ASSEMBLY_ANGLE_DEG)
    ours = time.perf_counter() - start

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(industrial_step_path)
        gmsh.model.occ.synchronize()
        start = time.perf_counter()
        gmsh.model.mesh.generate(2)
        theirs = time.perf_counter() - start
        kinds, tags, _ = gmsh.model.mesh.getElements(2)
        reference_tris = sum(len(t) for k, t in zip(kinds, tags) if k == 2)
    finally:
        gmsh.finalize()

    detail = (
        f"reference {theirs:.2f} s for {reference_tris} triangles; "
        f"here {ours:.2f} s for {mesh.tris.shape[0]} triangles"
    )
    assert theirs / ours >= 5.0, detail
    assert mesh.tris.shape[0] > reference_tris, f"the comparison is not conservative: {detail}"
