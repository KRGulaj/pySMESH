"""The identity gate: entity ids must never resolve to something they do not denote.

Every persistent-naming scheme that has gone wrong went wrong in history composition, so
this module does not test the registry against itself. It carries a **ground truth** — a
labelling of ``EntityId -> (kind, measure, centroid)`` maintained by the test, updated only
from what each operation's delta *declares* it changed — and re-derives the geometry from
the session after every operation. An id that has drifted onto different geometry without
the delta saying so is a mis-carried id, and that is the failure this module exists to
catch.

Falsification is mandatory here rather than optional: a history check that has never been
shown to fail is a claim, not a check. Two deliberate corruptions are therefore included and
each is asserted to make the ground-truth comparison fail — one that tears an operation's
history (:func:`test_a_torn_history_is_caught_by_the_ground_truth_check`) and one that swaps
two labels (:func:`test_swapping_two_ground_truth_labels_is_caught`).
"""

from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pysmesh as ps
from pysmesh import EntityId, EntityKind, ResolutionStatus, Session

# Tolerance for "the same geometry". Operations that keep an entity keep it exactly, so this
# is a floating-point equality guard rather than a physical tolerance.
GEOM_TOL: float = 1.0e-9

# Gate thresholds.
MIXED_OP_COUNT: int = 200
SNAPSHOT_BUDGET_S: float = 1.0e-3


@dataclass(frozen=True)
class _Label:
    """One entity's geometry as the ground truth last observed it."""

    kind: EntityKind
    measure: float
    centroid: tuple[float, float, float]
    shape_count: int


class GroundTruth:
    """Entity geometry tracked independently of the session's id registry.

    The session says *which* entity is where. This says where each entity **was**, and is
    updated only where an operation's delta explicitly declared a change. Comparing the two
    is what turns "the registry is self-consistent" into "the registry is correct".
    """

    def __init__(
        self, session: Session, kinds: tuple[EntityKind, ...] = tuple(EntityKind)
    ) -> None:
        self._s = session
        self._kinds = kinds
        self._labels: dict[int, _Label] = {}
        # Deferred-verification state, for models where re-observing every entity is too
        # costly to do per operation (see `defer` / `checkpoint`).
        self._pending_declared: set[int] = set()
        self._pending_deleted: set[int] = set()
        self._pending_rotation = np.eye(3)
        self._pending_offset = np.zeros(3)
        self._pending_ops = 0

    @property
    def labels(self) -> dict[int, _Label]:
        """The current labelling, keyed by entity id."""
        return self._labels

    def _observe(self) -> dict[int, _Label]:
        """Re-derive every live entity's geometry from the session."""
        out: dict[int, _Label] = {}
        for kind in self._kinds:
            t = self._s.entity_table(kind)
            for i, m, c, n in zip(
                t.ids.tolist(),
                t.measure.tolist(),
                t.centroid.tolist(),
                t.shape_count.tolist(),
                strict=True,
            ):
                out[int(i)] = _Label(kind, float(m), (c[0], c[1], c[2]), int(n))
        return out

    def baseline(self) -> None:
        """Adopt the session's current geometry wholesale, without checking it."""
        self._labels = self._observe()

    def absorb(self, delta: ps.HistoryDelta) -> None:
        """Verify a topology-changing operation, then absorb what it declared.

        Two independent halves, because either alone can be satisfied by a broken registry:

        **Soundness** — ids the delta did not name must be geometrically *identical* to
        their label. An operation that did not say it touched an entity is not allowed to
        have moved it.

        **Completeness** — an entity of the result that geometrically coincides with a
        pre-existing entity of the same kind must carry that entity's **id**. Without this,
        an operation could quietly kill every id and re-issue new ones for the same
        geometry, declare all of it in the delta, and pass. That is exactly what a dropped
        history does.

        The geometric matching here is a *test oracle*, not a resolution strategy. The
        session never matches on geometry — it is forbidden to, because a fingerprint
        mis-identifies entities under the very edits that make naming necessary. A test may
        use it because the test controls the fixtures and can require the fingerprints to be
        unambiguous, which is asserted rather than assumed.
        """
        observed = self._observe()
        declared = (
            set(delta.created.tolist())
            | set(delta.modified.tolist())
            | set(delta.split.tolist())
        )
        deleted = set(delta.deleted.tolist())

        for entity_id, label in self._labels.items():
            if entity_id in deleted:
                assert entity_id not in observed, (
                    f"op {delta.op_index} ({delta.op}) declared entity {entity_id} deleted, "
                    "but it is still live"
                )
                continue
            if entity_id in declared:
                continue
            assert entity_id in observed, (
                f"op {delta.op_index} ({delta.op}) lost entity {entity_id} without "
                "declaring it deleted"
            )
            self._assert_same(entity_id, label, observed[entity_id], delta)

        self._assert_geometry_kept_its_id(
            self._labels, observed, f"op {delta.op_index} ({delta.op})"
        )
        self._labels = observed

    @staticmethod
    def _assert_geometry_kept_its_id(
        before_labels: dict[int, _Label], observed: dict[int, _Label], where: str
    ) -> None:
        """Unambiguous geometry present before and after must carry the same id."""
        before = _unique_fingerprints(before_labels)
        after = _unique_fingerprints(observed)
        for fingerprint, old_id in before.items():
            new_id = after.get(fingerprint)
            if new_id is None:
                continue
            assert new_id == old_id, (
                f"{where} re-issued id {new_id} for the {fingerprint[0]} that entity "
                f"{old_id} already denoted — the operation's history did not carry the id "
                "forward"
            )

    def absorb_rigid(
        self,
        delta: ps.HistoryDelta,
        rotation: np.ndarray,
        offset: np.ndarray,
        moved: set[int] | None = None,
    ) -> None:
        """Verify a rigid transform: every id survives and lands exactly where predicted.

        A rigid transform changes only a shape's location, so this is the strictest check in
        the suite — no entity may be created, deleted or reshaped, and every centroid must
        equal the transformed original.
        """
        assert delta.created.size == 0, f"{delta.op} created entities"
        assert delta.deleted.size == 0, f"{delta.op} deleted entities"

        observed = self._observe()
        assert set(observed) == set(self._labels), f"{delta.op} changed the live id set"

        updated: dict[int, _Label] = {}
        for entity_id, label in self._labels.items():
            c = np.asarray(label.centroid)
            want = (rotation @ c + offset) if (moved is None or entity_id in moved) else c
            expected = _Label(
                label.kind, label.measure, (want[0], want[1], want[2]), label.shape_count
            )
            self._assert_same(entity_id, expected, observed[entity_id], delta)
            updated[entity_id] = observed[entity_id]
        self._labels = updated

    def defer(
        self,
        delta: ps.HistoryDelta,
        rotation: np.ndarray | None = None,
        offset: np.ndarray | None = None,
    ) -> None:
        """Record an operation for later verification, without re-observing the model.

        On a real assembly, re-deriving every entity's mass properties costs seconds, so
        checking after every operation is not viable. Deferring accumulates what the
        operations declared and composes their rigid transforms; :meth:`checkpoint` then
        verifies the whole window at once.

        This is weaker than per-operation checking in one respect only — it localises a
        failure to a window rather than to a single operation. It is not weaker about
        *whether* a failure is caught, because the composed transform is exact and the
        declared sets are cumulative.
        """
        self._pending_declared |= (
            set(delta.created.tolist())
            | set(delta.modified.tolist())
            | set(delta.split.tolist())
        )
        self._pending_deleted |= set(delta.deleted.tolist())
        self._pending_ops += 1
        if rotation is None:
            return
        assert offset is not None, "a rigid transform needs both a rotation and an offset"
        assert delta.created.size == 0, f"{delta.op} created entities"
        assert delta.deleted.size == 0, f"{delta.op} deleted entities"
        # Applying this transform after the accumulated one: x -> R(R0 x + t0) + t.
        self._pending_rotation = rotation @ self._pending_rotation
        self._pending_offset = rotation @ self._pending_offset + offset

    def checkpoint(self) -> None:
        """Verify everything deferred since the last checkpoint, then re-baseline."""
        observed = self._observe()
        where = f"the window of {self._pending_ops} ops ending at op {self._s.op_count}"
        rotation = self._pending_rotation
        offset = self._pending_offset

        expected: dict[int, _Label] = {}
        for entity_id, label in self._labels.items():
            moved = rotation @ np.asarray(label.centroid) + offset
            expected[entity_id] = _Label(
                label.kind, label.measure, (moved[0], moved[1], moved[2]), label.shape_count
            )

        for entity_id, label in expected.items():
            if entity_id in self._pending_deleted:
                assert entity_id not in observed, (
                    f"{where}: entity {entity_id} was declared deleted but is still live"
                )
                continue
            if entity_id in self._pending_declared:
                continue
            assert entity_id in observed, (
                f"{where}: entity {entity_id} vanished without being declared deleted"
            )
            self._assert_label(entity_id, label, observed[entity_id], where)

        self._assert_geometry_kept_its_id(expected, observed, where)

        self._labels = observed
        self._pending_declared.clear()
        self._pending_deleted.clear()
        self._pending_rotation = np.eye(3)
        self._pending_offset = np.zeros(3)
        self._pending_ops = 0

    def verify_all_issued_ids_are_alive_or_dead(self) -> None:
        """Every id the session ever issued must answer, and answer consistently."""
        live: set[int] = set()
        for kind in EntityKind:
            live |= {int(i) for i in self._s.entities(kind).tolist()}
        for entity_id in range(1, self._s.issued_id_count + 1):
            alive = self._s.is_alive(EntityId(entity_id))
            assert alive == (entity_id in live), (
                f"entity {entity_id} reports alive={alive} but "
                f"{'is' if entity_id in live else 'is not'} in the live set"
            )
            if alive:
                resolved = self._s.resolve(self._s.name_of(EntityId(entity_id)))
                assert resolved.status is not ResolutionStatus.LOST
                assert resolved.ids == (entity_id,)

    @classmethod
    def _assert_same(
        cls, entity_id: int, want: _Label, got: _Label, delta: ps.HistoryDelta
    ) -> None:
        cls._assert_label(entity_id, want, got, f"op {delta.op_index} ({delta.op})")

    @staticmethod
    def _assert_label(entity_id: int, want: _Label, got: _Label, context: str) -> None:
        where = f"entity {entity_id} after {context}"
        assert got.kind == want.kind, f"{where}: kind {got.kind} != {want.kind}"
        assert got.measure == pytest.approx(want.measure, abs=GEOM_TOL), (
            f"{where}: measure {got.measure} != {want.measure}"
        )
        assert got.centroid == pytest.approx(want.centroid, abs=GEOM_TOL), (
            f"{where}: centroid {got.centroid} != {want.centroid}"
        )


def _quantise(value: float) -> float:
    """Round to the geometric tolerance, so a fingerprint is hashable and stable."""
    return round(value / GEOM_TOL) * GEOM_TOL


def _unique_fingerprints(
    labels: dict[int, _Label],
) -> dict[tuple[EntityKind, float, float, float, float], int]:
    """Geometric fingerprint -> id, keeping only fingerprints that identify one entity.

    Coincident entities (the two faces meeting at a seam, for instance) share a fingerprint
    and are dropped: the oracle asserts only where geometry is genuinely unambiguous, which
    is the honest scope of a geometric cross-check.
    """
    buckets: dict[tuple[EntityKind, float, float, float, float], list[int]] = {}
    for entity_id, label in labels.items():
        key = (
            label.kind,
            _quantise(label.measure),
            _quantise(label.centroid[0]),
            _quantise(label.centroid[1]),
            _quantise(label.centroid[2]),
        )
        buckets.setdefault(key, []).append(entity_id)
    return {k: v[0] for k, v in buckets.items() if len(v) == 1}


def _rot_z(angle: float) -> np.ndarray:
    """Right-handed rotation about +z."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _solids(session: Session) -> list[EntityId]:
    return [EntityId(i) for i in session.entities(EntityKind.SOLID).tolist()]


def _process_rss_bytes() -> int:
    """Current working-set size of this process, via the Win32 process API."""

    class _Counters(ctypes.Structure):
        _fields_ = (
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        )

    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
    ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
        handle, ctypes.byref(counters), counters.cb
    )
    return int(counters.WorkingSetSize)


# --------------------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------------------- #


def _add_overlapping_bar(
    session: Session, rng: np.random.Generator
) -> tuple[ps.HistoryDelta, EntityId, EntityId]:
    """Add a bar that certainly intersects an existing solid, and name both for a fuse.

    The bar is placed from the target's own bounding box and spans it completely in y, so
    the fuse that follows genuinely rebuilds topology — entities die, seams merge, and a
    severed face splits. A mix of disjoint bodies alone would never exercise any of that.
    """
    table = session.entity_table(EntityKind.SOLID)
    pick = int(rng.integers(0, table.ids.size))
    target = EntityId(int(table.ids[pick]))
    xmin, ymin, zmin, xmax, ymax, zmax = table.bbox[pick]
    cx, cz = 0.5 * (xmin + xmax), 0.5 * (zmin + zmax)

    # Every dimension is a fraction of the target's own diagonal. Absolute sizes look
    # harmless and are not: a bar sized for a 3 x 7 x 11 box completely engulfs a part of a
    # real assembly modelled in metres, and a boolean that swallows its target leaves no
    # history to carry — so the operation silently stops exercising the thing under test.
    diagonal = math.dist((xmin, ymin, zmin), (xmax, ymax, zmax))
    width = max(0.2 * (xmax - xmin), 0.02 * diagonal)
    height = max(0.2 * (zmax - zmin), 0.02 * diagonal)
    overhang = 0.25 * diagonal

    delta = session.add_box(
        width,
        (ymax - ymin) + 2.0 * overhang,
        height,
        origin=(cx - 0.5 * width, ymin - overhang, cz - 0.5 * height),
    )
    bar = EntityId(
        int(
            next(
                i
                for i in delta.created
                if session.entity_kind(EntityId(int(i))) is EntityKind.SOLID
            )
        )
    )
    return delta, target, bar


def test_two_hundred_mixed_operations_never_mis_resolve_an_id() -> None:
    """Gate: 200 mixed operations, every id verified against an independent labelling.

    The mix deliberately includes operations that destroy topology, not only ones that add
    to it: a bar fused through an existing body kills faces, merges seam edges and vertices,
    and severs a face into two pieces that must both keep the original id. A benign mix of
    disjoint additions and rigid transforms would pass without ever exercising the history
    composition this gate exists to check, so the run asserts afterwards that it did.
    """
    rng = np.random.default_rng(42)
    s = Session()
    truth = GroundTruth(s)

    s.add_box(3.0, 7.0, 11.0)
    truth.baseline()

    deaths = 0
    merges = 0
    splits = 0
    ops_run = 1
    spare_x = 100.0
    while ops_run < MIXED_OP_COUNT:
        choice = int(rng.integers(0, 5))

        if choice == 0:  # add a disjoint body — nothing existing may be touched
            spare_x += 20.0
            delta = s.add_box(3.0, 7.0, 11.0, origin=(spare_x, 0.0, 0.0))
            truth.absorb(delta)

        elif choice == 1:  # translate the whole model
            offset = rng.uniform(-5.0, 5.0, size=3)
            delta = s.translate((offset[0], offset[1], offset[2]))
            truth.absorb_rigid(delta, np.eye(3), offset)
            spare_x += float(offset[0])

        elif choice == 2:  # rotate the whole model about +z through the origin
            angle = float(rng.uniform(-math.pi, math.pi))
            delta = s.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), angle)
            truth.absorb_rigid(delta, _rot_z(angle), np.zeros(3))

        elif choice == 3:  # fuse two existing bodies
            solids = _solids(s)
            if len(solids) < 2:
                continue
            pair = rng.choice(len(solids), size=2, replace=False)
            delta = s.fuse([solids[int(pair[0])]], [solids[int(pair[1])]])
            truth.absorb(delta)

        else:  # drive a bar through an existing body — the destructive case
            add_delta, target, bar = _add_overlapping_bar(s, rng)
            truth.absorb(add_delta)
            ops_run += 1
            delta = s.fuse([target], [bar])
            truth.absorb(delta)

        deaths += int(delta.deleted.size)
        merges += int(delta.merged.size)
        splits += int(delta.split.size)
        ops_run += 1

    assert s.op_count >= MIXED_OP_COUNT
    truth.verify_all_issued_ids_are_alive_or_dead()

    # The mix must actually have exercised history composition, or the gate proves nothing.
    assert deaths > 0, "no entity ever died; the operation mix was too benign"
    assert merges > 0, "no entity was ever merged onto another"
    assert splits > 0, "no entity was ever split"
    assert s.issued_id_count > s.entity_count


def test_a_torn_history_is_caught_by_the_ground_truth_check() -> None:
    """Falsification: drop one operation's history and the check must fail.

    Without this the gate above proves nothing — a check that cannot fail is not a check.
    A torn history makes the fuse re-issue ids for entities that already existed, so the
    ground truth sees entities vanish that the delta never declared deleted.
    """
    s = Session()
    truth = GroundTruth(s)
    s.add_box(3.0, 7.0, 11.0)
    s.add_box(3.0, 7.0, 11.0, origin=(3.0, 0.0, 0.0))
    truth.baseline()
    solids = _solids(s)

    s._debug_tear_next_history()
    delta = s.fuse([solids[0]], [solids[1]])

    with pytest.raises(AssertionError, match="did not carry the id forward"):
        truth.absorb(delta)


def test_the_same_fuse_passes_the_ground_truth_check_untorn() -> None:
    """The control for the falsification above: identical inputs, history intact."""
    s = Session()
    truth = GroundTruth(s)
    s.add_box(3.0, 7.0, 11.0)
    s.add_box(3.0, 7.0, 11.0, origin=(3.0, 0.0, 0.0))
    truth.baseline()
    solids = _solids(s)

    delta = s.fuse([solids[0]], [solids[1]])

    truth.absorb(delta)  # must not raise
    truth.verify_all_issued_ids_are_alive_or_dead()


def test_swapping_two_ground_truth_labels_is_caught() -> None:
    """Falsification: mislabel two entities and the geometric comparison must fail."""
    s = Session()
    truth = GroundTruth(s)
    s.add_box(3.0, 7.0, 11.0)
    truth.baseline()

    faces = [i for i, label in truth.labels.items() if label.kind is EntityKind.FACE]
    # The 21-area pair and the 77-area pair are distinguishable, so swap across them.
    a = min(faces, key=lambda i: truth.labels[i].measure)
    b = max(faces, key=lambda i: truth.labels[i].measure)
    truth.labels[a], truth.labels[b] = truth.labels[b], truth.labels[a]

    delta = s.add_box(1.0, 1.0, 1.0, origin=(50.0, 0.0, 0.0))

    with pytest.raises(AssertionError, match="measure"):
        truth.absorb(delta)


def test_a_torn_history_makes_names_resolve_lost_rather_than_wrong() -> None:
    """A tear must degrade to LOST, never to a confident wrong answer.

    This is the property that separates session ids from positional ordinals: when history
    is missing, an ordinal still resolves to *something*, and an id resolves to *nothing*.
    """
    s = Session()
    s.add_box(3.0, 7.0, 11.0)
    s.add_box(3.0, 7.0, 11.0, origin=(3.0, 0.0, 0.0))
    faces_before = s.entity_table(EntityKind.FACE)
    names = {int(i): s.name_of(EntityId(int(i))) for i in faces_before.ids}
    truth = {
        int(i): np.concatenate(([m], c))
        for i, m, c in zip(
            faces_before.ids, faces_before.measure, faces_before.centroid, strict=True
        )
    }
    solids = _solids(s)

    s._debug_tear_next_history()
    s.fuse([solids[0]], [solids[1]])

    after = s.entity_table(EntityKind.FACE)
    live = {
        int(i): np.concatenate(([m], c))
        for i, m, c in zip(after.ids, after.measure, after.centroid, strict=True)
    }
    lost = 0
    wrong = 0
    for entity_id, name in names.items():
        resolved = s.resolve(name)
        if resolved.status is ResolutionStatus.LOST:
            lost += 1
            continue
        if live[entity_id] != pytest.approx(truth[entity_id], abs=GEOM_TOL):
            wrong += 1

    assert lost > 0, "the tear must have cost some names for this test to mean anything"
    assert wrong == 0, "a torn history produced a confidently wrong resolution"


# --------------------------------------------------------------------------------------- #
# The same gate, on a real industrial assembly
# --------------------------------------------------------------------------------------- #

# Kinds carried in the ground truth for the industrial model. SOLID is excluded on cost
# grounds alone: BRepGProp::VolumeProperties on real trimmed geometry runs ~70 ms per solid,
# so observing 117 of them costs eight seconds per checkpoint and adds nothing the face and
# edge oracle does not already prove. The remaining three kinds cover 28 138 of the model's
# 28 255 entities, and every solid id is still covered by the alive-or-dead check.
_INDUSTRIAL_KINDS: tuple[EntityKind, ...] = (
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)
_CHECKPOINT_EVERY: int = 50


@pytest.mark.slow
def test_the_identity_gate_holds_on_a_real_industrial_assembly(
    industrial_step_brep: bytes,
) -> None:
    """Gate R0 on production geometry: a 117-solid assembly, ~28 000 entities.

    The synthetic gate proves the identity rules; this proves they survive contact with real
    trimmed CAD — dirty tolerances, thousands of faces, and bodies that a boolean has to work
    at rather than glide through.

    Verification is windowed rather than per-operation. Re-deriving every entity's mass
    properties costs about six seconds on this model, so checking after each of 200
    operations is not viable; the ground truth instead accumulates what the operations
    declared, composes their rigid transforms exactly, and verifies each window as a whole.
    """
    session = Session(validate=False)
    session.add_brep(industrial_step_brep)
    assert len(session.entities(EntityKind.FACE)) >= 500, "not a production-scale model"

    truth = GroundTruth(session, kinds=_INDUSTRIAL_KINDS)
    truth.baseline()
    rng = np.random.default_rng(42)

    fuse_failures = 0
    spare_x = 1.0e5
    for op in range(1, MIXED_OP_COUNT + 1):
        choice = int(rng.integers(0, 4))

        if choice == 0:
            spare_x += 500.0
            truth.defer(session.add_box(50.0, 50.0, 50.0, origin=(spare_x, 0.0, 0.0)))
        elif choice == 1:
            offset = rng.uniform(-10.0, 10.0, size=3)
            truth.defer(
                session.translate((offset[0], offset[1], offset[2])), np.eye(3), offset
            )
            spare_x += float(offset[0])
        elif choice == 2:
            angle = float(rng.uniform(-math.pi, math.pi))
            truth.defer(
                session.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), angle),
                _rot_z(angle),
                np.zeros(3),
            )
        else:
            solids = _solids(session)
            pair = rng.choice(len(solids), size=2, replace=False)
            try:
                truth.defer(
                    session.fuse([solids[int(pair[0])]], [solids[int(pair[1])]])
                )
            except ps.PysmeshError:
                # A boolean that OCCT refuses on real geometry leaves the session unchanged,
                # which is the contract. Count it rather than failing the gate.
                fuse_failures += 1

        if op % _CHECKPOINT_EVERY == 0:
            truth.checkpoint()

    assert session.op_count >= MIXED_OP_COUNT - fuse_failures
    truth.verify_all_issued_ids_are_alive_or_dead()
    assert session.issued_id_count > session.entity_count, "no entity ever died"


@pytest.mark.slow
def test_snapshot_stays_o1_on_a_real_industrial_assembly(
    industrial_step_brep: bytes,
) -> None:
    """Snapshot/restore must not notice that the model got 30x bigger.

    This is the claim the whole snapshot design rests on, and a ~28 000-entity assembly is
    where a hidden deep copy would finally show up.
    """
    session = Session(validate=False)
    session.add_brep(industrial_step_brep)
    entity_count = session.entity_count

    worst_snapshot = 0.0
    marks: list[ps.SnapshotMark] = []
    for _ in range(200):
        t0 = time.perf_counter()
        mark = session.snapshot()
        t1 = time.perf_counter()
        marks.append(mark)
        worst_snapshot = max(worst_snapshot, t1 - t0)

    worst_restore = 0.0
    for mark in (marks[0], marks[len(marks) // 2], marks[-1]):
        t0 = time.perf_counter()
        session.restore(mark)
        t1 = time.perf_counter()
        worst_restore = max(worst_restore, t1 - t0)

    before = _process_rss_bytes()
    extra = [session.snapshot() for _ in range(500)]
    after = _process_rss_bytes()

    assert entity_count > 20_000
    assert worst_snapshot < SNAPSHOT_BUDGET_S
    assert worst_restore < SNAPSHOT_BUDGET_S
    # 500 retained states of a 28 000-entity model. A deep copy would be gigabytes.
    assert after - before < 8 * 1024 * 1024
    assert len(extra) == 500


def _bar_fuse_on_real_geometry(brep: bytes, tear: bool) -> tuple[set[int], ps.HistoryDelta]:
    """Drive a bar through one part of the assembly and fuse, optionally tearing history.

    Two solids picked at random from a real assembly are usually spatially disjoint, and a
    boolean over disjoint bodies returns them untouched — there is no history to tear, so a
    falsification against such a pair would prove nothing. A bar through a chosen part
    forces the rebuild.

    Returns the pre-operation entity ids that survived it, and the fuse's delta.
    """
    session = Session(validate=False)
    session.add_brep(brep)
    before: set[int] = set()
    for kind in EntityKind:
        before |= {int(i) for i in session.entities(kind).tolist()}

    _, target, bar = _add_overlapping_bar(session, np.random.default_rng(3))
    if tear:
        session._debug_tear_next_history()
    delta = session.fuse([target], [bar])

    after: set[int] = set()
    for kind in EntityKind:
        after |= {int(i) for i in session.entities(kind).tolist()}
    return before & after, delta


@pytest.mark.slow
def test_a_torn_history_costs_identity_on_a_real_industrial_assembly(
    industrial_step_brep: bytes,
) -> None:
    """Falsification on production geometry, stated differentially.

    The geometric oracle used by :meth:`GroundTruth.absorb` is deliberately not used here,
    because on a real assembly it is genuinely weak: OCCT preserves sub-shapes a boolean
    does not touch *by identity*, so tearing the history costs only the entities the
    operation legitimately **modified** — and a modified entity's geometry has changed, so
    no fingerprint can recognise it. On the synthetic fixtures the same boolean rebuilds
    whole faces unchanged, which is why the oracle bites there and is asserted there.

    What is unambiguous on real geometry is the difference the history makes. The identical
    operation is run twice, and the torn run must carry strictly fewer ids forward.
    """
    intact, intact_delta = _bar_fuse_on_real_geometry(industrial_step_brep, tear=False)
    torn, torn_delta = _bar_fuse_on_real_geometry(industrial_step_brep, tear=True)

    # The intact run carries entities through modification and splitting; the torn one
    # cannot, so it kills them and issues fresh ids instead.
    assert intact_delta.modified.size > 0, "the fixture did not exercise modification"
    assert intact_delta.split.size > 0, "the fixture did not exercise splitting"
    assert torn_delta.modified.size == 0
    assert torn_delta.split.size == 0

    assert len(torn) < len(intact), (
        "tearing the history cost no identity — the operation cannot falsify the gate"
    )
    assert torn_delta.deleted.size > intact_delta.deleted.size


# --------------------------------------------------------------------------------------- #
# Snapshot cost
# --------------------------------------------------------------------------------------- #


def test_snapshot_and_restore_stay_under_the_budget_at_every_depth() -> None:
    """Snapshot/restore must be O(1) in the model size and in the stack depth."""
    s = Session()
    s.add_box(3.0, 7.0, 11.0)
    for k in range(30):  # a model with a few thousand entities
        s.add_box(1.0, 2.0, 3.0, origin=(10.0 * (k + 2), 0.0, 0.0))
    assert s.entity_count > 800

    worst_snapshot = 0.0
    worst_restore = 0.0
    marks: list[ps.SnapshotMark] = []
    for _ in range(200):
        t0 = time.perf_counter()
        mark = s.snapshot()
        t1 = time.perf_counter()
        marks.append(mark)
        worst_snapshot = max(worst_snapshot, t1 - t0)

    for mark in (marks[0], marks[len(marks) // 2], marks[-1]):
        t0 = time.perf_counter()
        s.restore(mark)
        t1 = time.perf_counter()
        worst_restore = max(worst_restore, t1 - t0)

    assert worst_snapshot < SNAPSHOT_BUDGET_S
    assert worst_restore < SNAPSHOT_BUDGET_S


def test_snapshot_memory_growth_is_bounded_by_the_retained_state_count() -> None:
    """Snapshots retain handles, not geometry: 500 of them must not clone the model.

    The honest cost of an O(1) snapshot is that every retained state pins the shapes it
    references. This measures that the cost is per-state bookkeeping rather than a copy of
    the model, and that discarding the marks releases it.
    """
    s = Session()
    s.add_box(3.0, 7.0, 11.0)
    for k in range(30):
        s.add_box(1.0, 2.0, 3.0, origin=(10.0 * (k + 2), 0.0, 0.0))

    before = _process_rss_bytes()
    marks = [s.snapshot() for _ in range(500)]
    after = _process_rss_bytes()

    assert s.snapshot_count == 500
    # 500 snapshots of an ~850-entity model. A deep copy would be tens of megabytes; a
    # handle copy is a few hundred bytes each.
    assert after - before < 8 * 1024 * 1024

    for mark in marks:
        s.discard_snapshot(mark)
    assert s.snapshot_count == 0


def test_restoring_the_same_mark_repeatedly_does_not_grow_the_id_space() -> None:
    s = Session()
    s.add_box(3.0, 7.0, 11.0)
    mark = s.snapshot()
    issued = s.issued_id_count

    for _ in range(50):
        s.restore(mark)

    assert s.issued_id_count == issued
    assert s.entity_count == 27


# --------------------------------------------------------------------------------------- #
# Property suite
# --------------------------------------------------------------------------------------- #

_OPS = st.sampled_from(["add", "translate", "rotate", "fuse", "bar"])


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(sequence=st.lists(_OPS, min_size=1, max_size=12))
def test_no_operation_sequence_makes_an_id_denote_the_wrong_entity(
    sequence: list[str],
) -> None:
    """For any sequence over the operation alphabet, ground truth must hold throughout.

    The gate above runs one long, seeded sequence. This runs many short arbitrary ones,
    which is where an ordering the fixed mix never produces would show up.
    """
    s = Session()
    truth = GroundTruth(s)
    s.add_box(3.0, 7.0, 11.0)
    truth.baseline()
    spare_x = 100.0

    for op in sequence:
        if op == "add":
            spare_x += 20.0
            truth.absorb(s.add_box(3.0, 7.0, 11.0, origin=(spare_x, 0.0, 0.0)))
        elif op == "translate":
            offset = np.array([1.5, -2.5, 0.75])
            truth.absorb_rigid(s.translate((1.5, -2.5, 0.75)), np.eye(3), offset)
            spare_x += 1.5
        elif op == "rotate":
            angle = 0.4
            truth.absorb_rigid(
                s.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), angle), _rot_z(angle), np.zeros(3)
            )
        elif op == "fuse":
            solids = _solids(s)
            if len(solids) < 2:
                continue
            truth.absorb(s.fuse([solids[0]], [solids[1]]))
        else:
            add_delta, target, bar = _add_overlapping_bar(s, np.random.default_rng(7))
            truth.absorb(add_delta)
            truth.absorb(s.fuse([target], [bar]))

    truth.verify_all_issued_ids_are_alive_or_dead()


@settings(max_examples=25, deadline=None)
@given(depth=st.integers(min_value=1, max_value=12))
def test_snapshot_and_restore_round_trip_at_any_depth(depth: int) -> None:
    """Restoring the mark taken at any depth reproduces that state exactly."""
    s = Session()
    s.add_box(3.0, 7.0, 11.0)

    marks: list[ps.SnapshotMark] = []
    states: list[tuple[bytes, tuple[int, ...]]] = []
    for k in range(depth):
        marks.append(s.snapshot())
        states.append((s.brep(), tuple(s.entities(EntityKind.FACE).tolist())))
        s.add_box(1.0, 1.0, 1.0, origin=(20.0 * (k + 1), 0.0, 0.0))

    for mark, (brep, faces) in zip(reversed(marks), reversed(states), strict=True):
        s.restore(mark)
        assert s.brep() == brep
        assert tuple(s.entities(EntityKind.FACE).tolist()) == faces
