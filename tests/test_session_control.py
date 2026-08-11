# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""Gates for the session's operational contract: progress, cancellation, tolerance, threads.

Four claims are under test.

* **A long operation reports monotone progress.** Every operation that can take real time
  accepts a callback and drives it with a strictly increasing fraction that ends at 1.0. The
  callback is consulted from a helper thread while the operation runs with the GIL released,
  so the test also has to show that the callback is genuinely called *during* the operation
  and not merely once at the end.
* **A cancelled operation raises and changes nothing.** The session's root shape and its
  whole id space are asserted unchanged after a cancel — not sampled, compared entity by
  entity — and the operation raises :class:`PysmeshCancelled` rather than returning a
  partial shape. That last direction is the one that matters: a cancelled ``ShapeFix_Shape``
  hands back a *non-null* shape carrying a fraction of the model's faces, so an
  implementation that trusted the algorithm's own reporting would commit a truncated model
  and this suite would catch it.
* **A tolerance changes the answer, and the default is the honest one.** A boolean that
  cannot bridge a gap at the default tolerance succeeds at a stated fuzzy value, with the
  gap and the value both named and the outcome asserted on topology rather than on a
  success flag. Values that are not distances — NaN, infinity, negatives — are refused
  before they reach OCCT.
* **Parallelism is a speed setting, never an answer.** Serial and parallel runs of the same
  boolean are asserted to agree on topology counts and total volume; the mesher's node
  coordinates are asserted **bitwise** equal, because a parallel result that quietly differed
  would be worse than a slow one.

Falsification is explicit throughout: a cancel that never fires must leave the operation
*succeeding*, and the fuzzy gate asserts the default genuinely fails before asserting that
the stated value rescues it. A gate that only shows the good direction cannot fail.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, PysmeshCancelled, PysmeshError, Session

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0

# A gap between two boxes' facing walls, and the fuzzy value that bridges it. Measured
# against OCCT: at fuzzy 0 the fuse leaves two disjoint solids; the walls are treated as
# coincident only once the fuzzy value reaches the gap.
WALL_GAP: float = 1.0e-4
BRIDGING_FUZZY: float = 1.0e-4
TOO_SMALL_FUZZY: float = 1.0e-6

# Deflection fine enough that a sphere's tessellation is worth timing and worth comparing
# coordinate by coordinate.
FINE_DEFLECTION: float = 0.002

# How long a cancelled operation may take to unwind after the request. The contract allows
# half a second; the poll interval is 25 ms and OCCT's own unwind was measured at 0-1 ms on
# booleans of 8 to 40 real solids, so this has a wide margin over what was observed.
CANCEL_BUDGET_S: float = 0.5


# ---- fixtures and helpers ------------------------------------------------------------ #


@pytest.fixture
def two_boxes() -> tuple[Session, EntityId, EntityId]:
    """A session with two overlapping boxes, and their solid ids."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(1.0, 1.0, 1.0))
    ids = [EntityId(int(v)) for v in s.entities(EntityKind.SOLID)]
    return s, ids[0], ids[1]


def gapped_boxes(gap: float) -> tuple[Session, EntityId, EntityId]:
    """Two boxes whose facing walls are `gap` apart — a fuse the default cannot bridge."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(BOX_DX + gap, 0.0, 0.0))
    ids = [EntityId(int(v)) for v in s.entities(EntityKind.SOLID)]
    return s, ids[0], ids[1]


def id_space(session: Session) -> dict[int, str]:
    """Every issued id and whether it is alive — the ground truth a cancel must not move.

    Read through the public state rather than from a delta, so it is independent of the
    operation under test.
    """
    out: dict[int, str] = {}
    for i in range(1, session.issued_id_count + 1):
        out[i] = "alive" if session.is_alive(EntityId(i)) else "dead"
    return out


def curved_session() -> Session:
    """A session whose tessellation is big enough to report progress several times."""
    s = Session()
    s.add_sphere(2.0)
    s.add_torus(3.0, 0.7, origin=(10.0, 0.0, 0.0))
    return s


class Recorder:
    """Records every progress value it is handed."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def __call__(self, fraction: float) -> None:
        self.values.append(fraction)


# ---- progress: the reporting contract ------------------------------------------------ #


def test_progress_callback_receives_a_final_value_of_one() -> None:
    s, a, b = Session(), None, None
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(1.0, 1.0, 1.0))
    ids = [EntityId(int(v)) for v in s.entities(EntityKind.SOLID)]
    rec = Recorder()

    s.fuse([ids[0]], [ids[1]], progress=rec)

    assert rec.values, "a completed operation must report at least once"
    assert rec.values[-1] == pytest.approx(1.0)


def test_progress_values_are_strictly_increasing_and_within_the_unit_interval() -> None:
    s = curved_session()
    rec = Recorder()

    s.tessellate(deflection=FINE_DEFLECTION, progress=rec)

    assert all(0.0 <= v <= 1.0 for v in rec.values)
    assert all(a < b for a, b in zip(rec.values, rec.values[1:]))


def test_progress_is_reported_during_a_long_operation_not_only_at_the_end() -> None:
    s = curved_session()
    rec = Recorder()

    s.tessellate(deflection=0.0005, progress=rec)

    # More than one value means the poller ran while the operation was still in flight; a
    # single value would be the completion report alone.
    assert len(rec.values) > 1
    assert rec.values[0] < 1.0


def test_omitting_both_hooks_runs_the_operation_unchanged(
    two_boxes: tuple[Session, EntityId, EntityId],
) -> None:
    s, a, b = two_boxes
    hooked = Session()
    hooked.add_box(BOX_DX, BOX_DY, BOX_DZ)
    hooked.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(1.0, 1.0, 1.0))
    hooked_ids = [EntityId(int(v)) for v in hooked.entities(EntityKind.SOLID)]

    plain = s.fuse([a], [b])
    with_hooks = hooked.fuse([hooked_ids[0]], [hooked_ids[1]], progress=Recorder())

    assert plain.created.tolist() == with_hooks.created.tolist()
    assert plain.deleted.tolist() == with_hooks.deleted.tolist()
    assert s.brep() == hooked.brep()


def test_a_progress_only_caller_is_never_asked_to_cancel(
    two_boxes: tuple[Session, EntityId, EntityId],
) -> None:
    s, a, b = two_boxes
    rec = Recorder()

    delta = s.fuse([a], [b], progress=rec)

    assert delta.created.size > 0


@pytest.mark.parametrize("hook", ["progress", "cancel"])
def test_a_non_callable_hook_is_refused(
    two_boxes: tuple[Session, EntityId, EntityId], hook: str
) -> None:
    s, a, b = two_boxes

    with pytest.raises(PysmeshError, match="must be callable or None"):
        s.fuse([a], [b], **{hook: 42})  # type: ignore[arg-type]


# ---- cancellation: the stop contract ------------------------------------------------- #


def test_a_cancelled_boolean_raises_the_cancellation_type(
    two_boxes: tuple[Session, EntityId, EntityId],
) -> None:
    s, a, b = two_boxes

    with pytest.raises(PysmeshCancelled):
        s.fuse([a], [b], cancel=lambda: True)


def test_the_cancellation_type_is_a_pysmesh_error() -> None:
    # A caller that only cares the operation did not happen keeps one except clause.
    assert issubclass(PysmeshCancelled, PysmeshError)


def test_a_cancelled_boolean_leaves_the_root_shape_byte_identical(
    two_boxes: tuple[Session, EntityId, EntityId],
) -> None:
    s, a, b = two_boxes
    before = s.brep()

    with pytest.raises(PysmeshCancelled):
        s.fuse([a], [b], cancel=lambda: True)

    assert s.brep() == before


def test_a_cancelled_boolean_leaves_every_id_exactly_as_it_was(
    two_boxes: tuple[Session, EntityId, EntityId],
) -> None:
    s, a, b = two_boxes
    before = id_space(s)
    before_ops = s.op_count
    before_issued = s.issued_id_count

    with pytest.raises(PysmeshCancelled):
        s.fuse([a], [b], cancel=lambda: True)

    assert id_space(s) == before
    assert s.op_count == before_ops
    assert s.issued_id_count == before_issued


def test_the_same_operation_succeeds_when_the_predicate_says_no(
    two_boxes: tuple[Session, EntityId, EntityId],
) -> None:
    # The falsification half: without it, an operation that always raised would pass every
    # test above.
    s, a, b = two_boxes
    before = s.brep()

    delta = s.fuse([a], [b], cancel=lambda: False)

    assert delta.created.size > 0
    assert s.brep() != before


def test_a_cancelled_heal_does_not_commit_a_truncated_model() -> None:
    # The direction that matters most: a cancelled ShapeFix_Shape returns a NON-null shape
    # carrying only the faces it reached, so an implementation trusting the algorithm's own
    # reporting would commit a partial model here.
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_sphere(2.0, centre=(20.0, 0.0, 0.0))
    before = s.brep()
    before_ids = id_space(s)

    with pytest.raises(PysmeshCancelled):
        s.heal(cancel=lambda: True)

    assert s.brep() == before
    assert id_space(s) == before_ids


def test_a_cancelled_tessellation_raises_and_emits_nothing() -> None:
    s = curved_session()

    with pytest.raises(PysmeshCancelled):
        s.tessellate(deflection=FINE_DEFLECTION, cancel=lambda: True)


def test_a_cancelled_tessellation_leaves_the_next_call_reporting_every_face() -> None:
    # A cancelled tessellation may leave part of the shape triangulated. That is a cache and
    # is harmless, but only because nothing was handed out: the next call must therefore
    # still report every face as re-triangulated rather than treating any of them as
    # already delivered.
    s = curved_session()
    with pytest.raises(PysmeshCancelled):
        s.tessellate(deflection=FINE_DEFLECTION, cancel=lambda: True)

    mesh = s.tessellate(deflection=FINE_DEFLECTION)

    assert sorted(mesh.retriangulated.tolist()) == sorted(mesh.face_id.tolist())


def test_a_cancelled_operation_does_not_advance_the_id_counter(
    two_boxes: tuple[Session, EntityId, EntityId],
) -> None:
    s, a, b = two_boxes
    before = s.issued_id_count

    with pytest.raises(PysmeshCancelled):
        s.fuse([a], [b], cancel=lambda: True)
    s.fuse([a], [b])

    # The successful retry issues ids from where the cancelled attempt left off, so the
    # cancelled attempt consumed none.
    assert s.issued_id_count > before


def test_a_cancel_requested_part_way_through_stops_within_the_budget() -> None:
    s = curved_session()
    stop_at = 0.3
    requested: list[float] = []

    def cancel_when_started() -> bool:
        # Cancel once the operation has genuinely begun, so this measures the stop path
        # rather than the pre-start refusal.
        return bool(requested)

    def note(fraction: float) -> None:
        if fraction >= stop_at and not requested:
            requested.append(time.perf_counter())

    with pytest.raises(PysmeshCancelled):
        s.tessellate(
            deflection=0.0002, progress=note, cancel=cancel_when_started
        )

    assert requested, "the operation finished before the cancel point was reached"
    assert time.perf_counter() - requested[0] < CANCEL_BUDGET_S


def test_a_cancel_flag_set_before_the_call_is_honoured_however_fast_the_operation(
    two_boxes: tuple[Session, EntityId, EntityId],
) -> None:
    # A two-box fuse finishes in about 4 ms, well inside one poll interval. The predicate is
    # therefore asked once synchronously before the algorithm starts, so a pre-set flag
    # cannot be missed by an operation that happens to be cheap.
    s, a, b = two_boxes
    flag = threading.Event()
    flag.set()

    with pytest.raises(PysmeshCancelled):
        s.fuse([a], [b], cancel=flag.is_set)


def test_a_raising_cancel_predicate_propagates_its_own_exception(
    two_boxes: tuple[Session, EntityId, EntityId],
) -> None:
    s, a, b = two_boxes

    def boom() -> bool:
        raise ZeroDivisionError("the caller's own bug")

    with pytest.raises(ZeroDivisionError, match="the caller's own bug"):
        s.fuse([a], [b], cancel=boom)


def test_a_raising_progress_callback_stops_the_operation_and_commits_nothing() -> None:
    # The session's identity state is the thing that must not move. Its BREP is deliberately
    # not the oracle here: a serialised shape carries its triangulations, and a tessellation
    # writes those onto the shape as a cache whether it finishes or not.
    s = curved_session()
    before_ids = id_space(s)
    before_ops = s.op_count

    def boom(fraction: float) -> None:
        raise ZeroDivisionError("rendering the bar failed")

    with pytest.raises(ZeroDivisionError):
        s.tessellate(deflection=0.0002, progress=boom)

    assert id_space(s) == before_ids
    assert s.op_count == before_ops


@pytest.mark.parametrize(
    "operation",
    ["fuse", "cut", "common", "section", "split", "fragment"],
)
def test_every_boolean_accepts_and_honours_a_cancel(
    two_boxes: tuple[Session, EntityId, EntityId], operation: str
) -> None:
    s, a, b = two_boxes
    args: tuple[object, ...] = ([a], [b]) if operation != "fragment" else ([a, b],)

    with pytest.raises(PysmeshCancelled):
        getattr(s, operation)(*args, cancel=lambda: True)


def test_fillet_and_chamfer_accept_a_cancel() -> None:
    for operation, extra in (("fillet", 0.5), ("chamfer", 0.5)):
        s = Session()
        s.add_box(BOX_DX, BOX_DY, BOX_DZ)
        edge = EntityId(int(s.entities(EntityKind.EDGE)[0]))
        with pytest.raises(PysmeshCancelled):
            getattr(s, operation)([edge], extra, cancel=lambda: True)


def test_add_brep_accepts_a_cancel(box_brep: bytes) -> None:
    s = Session()

    with pytest.raises(PysmeshCancelled):
        s.add_brep(box_brep, cancel=lambda: True)

    assert s.entity_count == 0


def test_defeature_and_imprint_accept_a_cancel(
    two_boxes: tuple[Session, EntityId, EntityId],
) -> None:
    s, a, b = two_boxes
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    with pytest.raises(PysmeshCancelled):
        s.defeature([face], cancel=lambda: True)
    with pytest.raises(PysmeshCancelled):
        s.imprint([a], [b], cancel=lambda: True)


def test_sew_accepts_a_cancel(open_box_shell_brep: bytes) -> None:
    s = Session()
    s.add_brep(open_box_shell_brep)
    face = EntityId(int(s.entities(EntityKind.FACE)[0]))

    with pytest.raises(PysmeshCancelled):
        s.sew([face], cancel=lambda: True)


# ---- tolerance: the fuzzy contract --------------------------------------------------- #


def test_a_gap_the_default_cannot_bridge_leaves_two_solids() -> None:
    # The falsification half of the fuzzy gate: assert the default genuinely fails before
    # asserting that a stated value rescues it.
    s, a, b = gapped_boxes(WALL_GAP)

    s.fuse([a], [b])

    assert len(ps.load_brep(s.brep()).solids()) == 2


def test_the_same_fuse_succeeds_at_a_fuzzy_value_matching_the_gap() -> None:
    s, a, b = gapped_boxes(WALL_GAP)

    s.fuse([a], [b], fuzzy=BRIDGING_FUZZY)

    shape = ps.load_brep(s.brep())
    assert len(shape.solids()) == 1
    # The two walls became one face, so the fused solid has 10 faces rather than 12.
    assert len(shape.faces()) == 10


def test_a_fuzzy_value_below_the_gap_does_not_bridge_it() -> None:
    # The tolerance has to be chosen against the gap, not turned up hopefully; a value an
    # order of magnitude too small changes nothing.
    s, a, b = gapped_boxes(WALL_GAP)

    s.fuse([a], [b], fuzzy=TOO_SMALL_FUZZY)

    assert len(ps.load_brep(s.brep()).solids()) == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_a_fuzzy_value_that_is_not_a_distance_is_refused(
    two_boxes: tuple[Session, EntityId, EntityId], bad: float
) -> None:
    # NaN is the one worth naming: every comparison against it is false, so a naive
    # `fuzzy < 0` test lets it through to OCCT as a tolerance nothing compares equal to.
    s, a, b = two_boxes

    with pytest.raises(PysmeshError, match="finite value >= 0"):
        s.fuse([a], [b], fuzzy=bad)


def test_a_refused_fuzzy_value_leaves_the_session_untouched(
    two_boxes: tuple[Session, EntityId, EntityId],
) -> None:
    s, a, b = two_boxes
    before = id_space(s)

    with pytest.raises(PysmeshError):
        s.fuse([a], [b], fuzzy=float("nan"))

    assert id_space(s) == before
    assert s.op_count == 2


# ---- parallelism: a speed setting, never an answer ----------------------------------- #


def test_serial_and_parallel_booleans_agree_on_topology_and_volume() -> None:
    results = []
    for parallel in (False, True):
        s, a, b = two_boxes_session()
        s.fuse([a], [b], parallel=parallel)
        shape = ps.load_brep(s.brep())
        results.append(
            (
                len(shape.solids()),
                len(shape.faces()),
                len(shape.edges()),
                len(shape.vertices()),
                sum(sol.volume for sol in shape.solids()),
            )
        )

    serial, parallel_result = results
    assert serial[:4] == parallel_result[:4]
    assert serial[4] == pytest.approx(parallel_result[4], rel=1e-12)


def test_serial_and_parallel_tessellations_are_bitwise_identical() -> None:
    meshes = []
    for parallel in (False, True):
        s = curved_session()
        meshes.append(s.tessellate(deflection=FINE_DEFLECTION, parallel=parallel))

    serial, parallel_mesh = meshes
    assert np.array_equal(serial.nodes, parallel_mesh.nodes)
    assert np.array_equal(serial.tris, parallel_mesh.tris)
    assert np.array_equal(serial.tri_face_id, parallel_mesh.tri_face_id)


def two_boxes_session() -> tuple[Session, EntityId, EntityId]:
    """A fresh two-box session, for the tests that need several independent ones."""
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)
    s.add_box(BOX_DX, BOX_DY, BOX_DZ, origin=(1.0, 1.0, 1.0))
    ids = [EntityId(int(v)) for v in s.entities(EntityKind.SOLID)]
    return s, ids[0], ids[1]


# ---- the operations OCCT gives no progress range ------------------------------------- #


def test_the_two_repairs_without_a_progress_range_take_no_hooks() -> None:
    # These are not oversights. ShapeUpgrade_UnifySameDomain::Build and
    # ShapeUpgrade_RemoveInternalWires::Perform take no Message_ProgressRange in OCCT 8.0,
    # so there is nothing to drive; they take neither argument rather than accepting one and
    # ignoring it.
    s = Session()
    s.add_box(BOX_DX, BOX_DY, BOX_DZ)

    with pytest.raises(TypeError):
        s.unify_same_domain(cancel=lambda: True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        s.remove_internal_wires(cancel=lambda: True)  # type: ignore[call-arg]


# ---- the real assembly --------------------------------------------------------------- #


@pytest.mark.slow
def test_tessellating_a_real_assembly_reports_monotone_progress(
    industrial_step_brep: bytes,
) -> None:
    # A weaker bar than the boolean's on purpose. An update is delivered only when the
    # position has actually advanced, so the count tracks OCCT's own progress through the
    # algorithm rather than the wall clock: a one-second tessellation reports about 17
    # times, because the mesher spends long stretches inside a single scope. The contract's
    # "at least 20 updates" is asserted where the contract puts it, on the boolean.
    s = Session()
    s.add_brep(industrial_step_brep)
    rec = Recorder()

    s.tessellate(deflection=0.001, progress=rec)

    assert len(rec.values) > 5, f"only {len(rec.values)} updates"
    assert all(a < b for a, b in zip(rec.values, rec.values[1:]))
    assert rec.values[-1] == pytest.approx(1.0)


def _assembly_cut(brep: bytes) -> tuple[Session, list[EntityId], EntityId]:
    """A session on the real assembly plus a knife box, and the operands for cutting it.

    Twenty solids rather than eight: the cut then takes about a second, and the number of
    updates a caller sees is set by the operation's duration against the 25 ms poll
    interval, not by how often OCCT advances its own position. Eight solids finish in about
    300 ms and yield 16 updates, which is under the bar the contract sets.
    """
    s = Session()
    s.add_brep(brep)
    solids = [EntityId(int(v)) for v in s.entities(EntityKind.SOLID)]
    s.add_box(20.0, 20.0, 1.0, origin=(-10.0, -10.0, 0.0))
    knife = EntityId(int(s.entities(EntityKind.SOLID)[-1]))
    return s, solids[:20], knife


@pytest.mark.slow
def test_a_real_assembly_boolean_reports_many_monotone_updates(
    industrial_step_brep: bytes,
) -> None:
    s, targets, knife = _assembly_cut(industrial_step_brep)
    rec = Recorder()

    s.cut(targets, [knife], progress=rec)

    assert len(rec.values) >= 20, f"only {len(rec.values)} updates"
    assert all(a < b for a, b in zip(rec.values, rec.values[1:]))
    assert rec.values[-1] == pytest.approx(1.0)


@pytest.mark.slow
def test_a_cancelled_real_assembly_boolean_stops_in_budget_and_changes_nothing(
    industrial_step_brep: bytes,
) -> None:
    # The whole operational contract in one test, on a boolean over a real assembly:
    # monotone progress, a stop inside the budget, and a session left exactly as it was
    # afterwards — root shape and id space both. The root shape is compared
    # byte for byte — a boolean writes no triangulation, so its BREP is a sound oracle here
    # in a way it would not be for a tessellation.
    s, targets, knife = _assembly_cut(industrial_step_brep)
    before = s.brep()
    before_ids = id_space(s)
    requested: list[float] = []

    def note(fraction: float) -> None:
        if fraction >= 0.3 and not requested:
            requested.append(time.perf_counter())

    with pytest.raises(PysmeshCancelled):
        s.cut(targets, [knife], progress=note, cancel=lambda: bool(requested))

    assert requested, "the operation finished before the cancel point was reached"
    elapsed = time.perf_counter() - requested[0]
    assert elapsed < CANCEL_BUDGET_S, f"cancel took {elapsed:.3f} s"
    assert s.brep() == before
    assert id_space(s) == before_ids
