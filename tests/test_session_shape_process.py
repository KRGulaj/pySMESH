# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""Gates for Session.shape_process: GEOM's ShapeProcess.

The answers are closed forms on a cylinder of radius ``R = 2`` and height ``H = 5``: volume
``pi R^2 H = 20 pi``, lateral area ``2 pi R H = 20 pi``, two caps of area ``4 pi``, a seam
edge of length 5 at ``x = 2``, two circles of length ``4 pi``.

* SplitClosedFaces with one split point cuts the lateral face along the line ``x = -2``,
  opposite its seam: two halves of area ``10 pi``, with centroids ``(0, +-2R/pi, H/2)``
  (the centroid of a half cylinder lies ``2R/pi`` from the axis). It adds that line, an
  edge of length 5, and splits each circle at its end, a new vertex.
* SplitAngle with the angle ``pi/2`` cuts it into four quarters of area ``5 pi``, with
  centroids ``(+-a, +-a, H/2)``, ``a = R sin(pi/2) / (pi/2) = 4/pi``.
* On the cylinder translated by ``(10, 0, 0)`` (a placed body) the answers move with it.

Measures and centroids come from entity_table and load_brep, integrated adaptively at the
relative precision 1e-6, which reads these surfaces within 1e-10: areas and lengths are
compared at 1e-9 relative, centroids at 1e-9 absolute.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import EntityId, EntityKind, PysmeshCancelled, Session
from pysmesh.session import ShapeProcessValue

R: float = 2.0
H: float = 5.0
RTOL: float = 1e-9
ATOL: float = 1e-9
CHILD_TIMEOUT_S: float = 120.0

_KINDS: tuple[EntityKind, ...] = (
    EntityKind.SOLID,
    EntityKind.FACE,
    EntityKind.EDGE,
    EntityKind.VERTEX,
)

SPLIT_CLOSED: dict[str, ShapeProcessValue] = {
    "SplitClosedFaces.CloseTolerance": 1e-7,
    "SplitClosedFaces.MaxTolerance": 1e-7,
    "SplitClosedFaces.NbSplitPoints": 1,
    "SplitClosedFaces.SegmentSurfaceMode": True,
}
SPLIT_ANGLE: dict[str, ShapeProcessValue] = {
    "SplitAngle.Angle": math.pi / 2,
    "SplitAngle.MaxTolerance": 1e-7,
}

REFERENCE_PAGE: Path = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "documentation"
    / "concepts"
    / "shape-process.md"
)

# OCCT's own value for each parameter that GEOM's ShHealing file does not set, read at the
# OCCT 8.0.1 source (ShapeProcess_OperLibrary.cxx). A parameter the operator reads only
# when it is present, with no value in OCCT either, takes a small valid value here.
OCCT_VALUES: dict[str, ShapeProcessValue] = {
    "SetTolerance.Mode": 0,
    "SetTolerance.Value": 1e-7,
    "SetTolerance.Ratio": 1.0,
    "SetTolerance.Regularity": 1e-10,
    "BSplineRestriction.ConicalSurfMode": False,
    "BSplineRestriction.CylindricalSurfMode": False,
    "BSplineRestriction.ToroidalSurfMode": False,
    "BSplineRestriction.SphericalSurfMode": False,
    "SurfaceToBSpline.LinearExtrusionMode": True,
    "SurfaceToBSpline.RevolutionMode": True,
    "SurfaceToBSpline.OffsetMode": True,
    "SplitContinuity.Tolerance2d": 1e-9,
    "SplitContinuity.Curve2dContinuity": "C1",
    "SplitContinuity.MaxTolerance": 1e-7,
    "FixWireGaps.Tolerance3d": 1e-7,
    "DropSmallSolids.FixMode": 0,
    "FixShape.FixVertexPositionMode": 0,
    "FixShape.FixVertexToleranceMode": -1,
    "FixShape.FixShellOrientationMode": -1,
    "FixShape.FixFaceOrientationMode": -1,
    "FixShape.FixAddNaturalBoundMode": -1,
    "FixShape.RemoveSmallAreaFaceMode": -1,
    "FixShape.FixSplitFaceMode": -1,
    "FixShape.RemoveLoopMode": -1,
    "FixShape.FixNotchedEdgesMode": -1,
    "FixShape.FixTailMode": 0,
    "FixShape.MaxTailAngle": 0.0,
    "FixShape.MaxTailWidth": -1.0,
    "SplitClosedEdges.NbSplitPoints": 1,
}

# The DLL directories and the package path a child process needs before its import.
_CHILD: str = """
import hashlib, os, sys
for d in (os.environ.get("PYSMESH_OCCT_BIN"),
          os.path.join(sys.prefix, "Library", "bin")):
    if d and os.path.isdir(d):
        os.add_dll_directory(d)
sys.path.insert(0, sys.argv[1])
import pysmesh as ps
from pysmesh import EntityId, EntityKind as K
s = ps.Session()
s.add_cylinder(2.0, 5.0)
s.shape_process(
    [EntityId(int(s.entities(K.SOLID)[0]))],
    ["SplitClosedFaces"],
    {"SplitClosedFaces.CloseTolerance": 1e-7, "SplitClosedFaces.MaxTolerance": 1e-7,
     "SplitClosedFaces.NbSplitPoints": 1, "SplitClosedFaces.SegmentSurfaceMode": True},
)
print("RESULT", hashlib.sha256(s.brep()).hexdigest(),
      [[int(i) for i in s.entities(k)] for k in (K.SOLID, K.FACE, K.EDGE, K.VERTEX)])
"""

# A ShHealing resource file whose every value differs from the call's.
_SHHEALING: str = """ShapeProcess.exec.op : SplitAngle
ShapeProcess.SplitAngle.Angle : 0.5
ShapeProcess.SplitAngle.MaxTolerance : 0.5
ShapeProcess.SplitClosedFaces.CloseTolerance : 0.5
ShapeProcess.SplitClosedFaces.MaxTolerance : 0.5
ShapeProcess.SplitClosedFaces.NbSplitPoints : 3
ShapeProcess.SplitClosedFaces.SegmentSurfaceMode : 0
"""


def id_space(s: Session) -> tuple[tuple[int, ...], ...]:
    """Every live id, by kind."""
    return tuple(tuple(int(i) for i in s.entities(k)) for k in _KINDS)


def state(s: Session) -> tuple[bytes, tuple[tuple[int, ...], ...], int, int]:
    """The bytes, the live ids and the two counters."""
    return s.brep(), id_space(s), s.op_count, s.issued_id_count


def geometry(s: Session) -> dict[int, tuple[float, np.ndarray]]:
    """Every live id's measure and centroid, by the rule of entity_table."""
    out: dict[int, tuple[float, np.ndarray]] = {}
    for kind in _KINDS:
        table = s.entity_table(kind)
        for i, m, c in zip(table.ids, table.measure, table.centroid, strict=True):
            out[int(i)] = (float(m), np.array(c))
    return out


def cylinder(offset: float = 0.0) -> tuple[Session, EntityId]:
    """The cylinder of radius 2 and height 5, translated by ``(offset, 0, 0)``."""
    s = Session()
    s.add_cylinder(R, H)
    solid = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    if offset != 0.0:
        s.translate((offset, 0.0, 0.0), [solid])
    return s, solid


def lateral_faces(s: Session) -> list[ps.FaceInfo]:
    """The faces of the model that lie on a cylinder, ordered by centroid."""
    faces = [f for f in ps.load_brep(s.brep()).faces() if f.surface_type == "Cylinder"]
    return sorted(
        faces, key=lambda f: (round(f.centroid[0], 6), round(f.centroid[1], 6))
    )


def id_at(s: Session, kind: EntityKind, point: tuple[float, float, float]) -> int:
    """The one live id of a kind whose centroid is the point."""
    table = s.entity_table(kind)
    near = np.linalg.norm(table.centroid - np.asarray(point), axis=1) < 1e-6
    assert int(np.count_nonzero(near)) == 1, point
    return int(table.ids[near][0])


def test_split_closed_faces_cuts_the_lateral_face_into_two_halves() -> None:
    s, solid = cylinder()
    lateral = id_at(s, EntityKind.FACE, (0.0, 0.0, H / 2))

    delta = s.shape_process([solid], ["SplitClosedFaces"], SPLIT_CLOSED)

    halves = lateral_faces(s)
    assert len(halves) == 2
    np.testing.assert_allclose([f.area for f in halves], [10 * math.pi] * 2, rtol=RTOL)
    np.testing.assert_allclose(
        [f.centroid for f in halves],
        [(0.0, -2 * R / math.pi, H / 2), (0.0, 2 * R / math.pi, H / 2)],
        atol=ATOL,
    )
    assert s.is_alive(EntityId(lateral))
    assert s.shape_count(EntityId(lateral)) == 2
    assert list(delta.deleted) == []
    assert delta.valid is True


@pytest.mark.parametrize("offset", [0.0, 10.0], ids=["unplaced", "placed"])
def test_split_angle_cuts_the_lateral_face_into_four_quarters(offset: float) -> None:
    """A translated cylinder is a placed body: the quarters move with it."""
    s, solid = cylinder(offset)
    lateral = id_at(s, EntityKind.FACE, (offset, 0.0, H / 2))
    a = R * math.sin(math.pi / 2) / (math.pi / 2)

    delta = s.shape_process([solid], ["SplitAngle"], SPLIT_ANGLE)

    quarters = lateral_faces(s)
    np.testing.assert_allclose([f.area for f in quarters], [5 * math.pi] * 4, rtol=RTOL)
    np.testing.assert_allclose(
        [f.centroid for f in quarters],
        [
            (offset - a, -a, H / 2),
            (offset - a, a, H / 2),
            (offset + a, -a, H / 2),
            (offset + a, a, H / 2),
        ],
        atol=ATOL,
    )
    assert s.shape_count(EntityId(lateral)) == 4
    assert list(delta.deleted) == []


def test_every_id_after_splitting_denotes_its_geometry() -> None:
    """The history check: falsified by tearing the operation's history (final/falsify_*).

    Every id keeps its measure and centroid: the lateral face and the two circles are
    split, and a split id is measured over all its pieces. The new entities are the line
    ``x = -2`` (length 5) and its two ends.
    """
    s, solid = cylinder()
    before = geometry(s)

    delta = s.shape_process([solid], ["SplitClosedFaces"], SPLIT_CLOSED)
    after = geometry(s)

    assert list(delta.deleted) == []
    for i, (measure, centroid) in before.items():
        assert np.isclose(after[i][0], measure, rtol=RTOL), i
        np.testing.assert_allclose(after[i][1], centroid, atol=ATOL)
    created = {int(i) for i in delta.created}
    assert set(after) == set(before) | created
    new = sorted(
        (
            str(s.entity_kind(EntityId(i))),
            round(after[i][0], 9),
            *np.round(after[i][1], 9),
        )
        for i in created
    )
    assert new == sorted(
        [
            ("EDGE", 5.0, -R, 0.0, H / 2),
            ("VERTEX", 0.0, -R, 0.0, 0.0),
            ("VERTEX", 0.0, -R, 0.0, H),
        ]
    )


@pytest.mark.parametrize("offset", [0.0, 10.0], ids=["unplaced", "placed"])
def test_a_modification_operator_keeps_every_id_of_the_cylinder(offset: float) -> None:
    """ElementaryToRevolution (a BRepTools_Modification) puts the lateral face on a
    surface of revolution and changes no measure: every id stays on its entity, also
    when the body is placed (ShapeProcess records no history below a placed body).

    The adaptive integration reads the volume bounded by a surface of revolution within
    2e-6 relative (measured), not the 1e-10 of a cylinder: measures and centroids are
    compared at 1e-5 here.
    """
    s, solid = cylinder(offset)
    before = geometry(s)

    delta = s.shape_process([solid], ["ElementaryToRevolution"], {})
    after = geometry(s)

    assert list(delta.deleted) == []
    assert list(delta.created) == []
    assert sorted(after) == sorted(before)
    for i, (measure, centroid) in before.items():
        assert np.isclose(after[i][0], measure, rtol=1e-5), i
        np.testing.assert_allclose(after[i][1], centroid, atol=1e-5)
    types = {f.surface_type for f in ps.load_brep(s.brep()).faces()}
    assert types == {"Plane", "Revolution"}


def test_shape_process_leaves_a_snapshot_its_own_shapes() -> None:
    """The operators work on a TNaming_CopyShape copy of the bodies."""
    s, solid = cylinder()
    mark = s.snapshot()
    before = s.brep()

    s.shape_process([solid], ["SplitClosedFaces"], SPLIT_CLOSED)
    s.restore(mark)

    assert s.brep() == before


def test_eight_sessions_between_unrelated_ones_give_identical_bytes_and_ids() -> None:
    def processed() -> tuple[bytes, tuple[tuple[int, ...], ...]]:
        s, solid = cylinder()
        s.shape_process([solid], ["SplitAngle"], SPLIT_ANGLE)
        return s.brep(), id_space(s)

    first = processed()
    others = []

    for k in range(7):
        churn = [Session() for _ in range(k + 1)]
        for i, c in enumerate(churn):
            c.add_box(1.0 + i, 2.0, 3.0)
            c.add_cylinder(0.5, 2.0)
        others.append(processed())

    assert all(other == first for other in others)


def test_the_result_does_not_depend_on_the_shhealing_resource_files(
    tmp_path: Path,
) -> None:
    """A child with CSF_ShHealingDefaults and CSF_ShHealingUserDefaults set to directories
    whose ShHealing file holds other values gives the bytes and ids of a child without.
    """
    defaults = tmp_path / "defaults"
    user = tmp_path / "user"
    for d in (defaults, user):
        d.mkdir()
        (d / "ShHealing").write_text(_SHHEALING, encoding="ascii")
    package_root = str(Path(ps.__file__).resolve().parent.parent)
    plain = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CSF_ShHealingDefaults", "CSF_ShHealingUserDefaults")
    }
    with_files = {
        **plain,
        "CSF_ShHealingDefaults": str(defaults),
        "CSF_ShHealingUserDefaults": str(user),
    }

    runs = [
        subprocess.run(
            [sys.executable, "-c", _CHILD, package_root],
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
            env=env,
            check=False,
        )
        for env in (plain, with_files)
    ]

    results = [
        [ln for ln in run.stdout.splitlines() if ln.startswith("RESULT ")]
        for run in runs
    ]
    assert all(run.returncode == 0 for run in runs), [r.stderr[-2000:] for r in runs]
    assert len(results[0]) == 1
    assert results[0] == results[1]


def test_a_cancelled_shape_process_raises_and_leaves_the_session_as_it_was() -> None:
    s, solid = cylinder()
    before = state(s)

    with pytest.raises(PysmeshCancelled):
        s.shape_process([solid], ["SplitAngle"], SPLIT_ANGLE, cancel=lambda: True)

    assert state(s) == before


def test_shape_process_reports_progress_to_the_end() -> None:
    s, solid = cylinder()
    seen: list[float] = []

    s.shape_process(
        [solid],
        ["SplitClosedFaces", "SplitAngle"],
        {**SPLIT_CLOSED, **SPLIT_ANGLE},
        progress=seen.append,
    )

    assert seen
    assert seen == sorted(seen)
    assert seen[-1] == 1.0


@pytest.mark.parametrize(
    ("operators", "parameters", "match"),
    [
        ([], {}, "no operator is named"),
        (["Foo", "Bar"], {}, "'Foo', 'Bar' are not a ShapeProcess operator"),
        (
            ["SplitAngle"],
            {"SplitAngle.Angle": 1.0},
            "read SplitAngle.MaxTolerance, which is not given",
        ),
        (
            ["SplitAngle"],
            {**SPLIT_ANGLE, "FixShape.Tolerance3d": 1e-7},
            "no named operator reads FixShape.Tolerance3d",
        ),
        (
            ["SplitAngle"],
            {**SPLIT_ANGLE, "SplitAngle.Angle": "1.0"},
            r"SplitAngle.Angle takes a real number; got '1.0' \(str\)",
        ),
        (
            ["SplitAngle"],
            {**SPLIT_ANGLE, "SplitAngle.Angle": True},
            r"SplitAngle.Angle takes a real number; got True \(bool\)",
        ),
        (
            ["SplitAngle"],
            {**SPLIT_ANGLE, "SplitAngle.Angle": math.inf},
            "SplitAngle.Angle must be a finite number",
        ),
        (
            ["SplitClosedFaces"],
            {**SPLIT_CLOSED, "SplitClosedFaces.SegmentSurfaceMode": 1},
            r"SegmentSurfaceMode takes a bool; got 1 \(int\)",
        ),
        (
            ["SplitClosedFaces"],
            {**SPLIT_CLOSED, "SplitClosedFaces.NbSplitPoints": 1.0},
            r"NbSplitPoints takes an integer; got 1.0 \(float\)",
        ),
        (
            ["SplitClosedFaces"],
            {**SPLIT_CLOSED, "SplitClosedFaces.NbSplitPoints": 2**40},
            "outside the range of a 32-bit integer",
        ),
        (
            ["SplitContinuity"],
            {
                "SplitContinuity.Tolerance3d": 1e-7,
                "SplitContinuity.Tolerance2d": 1e-9,
                "SplitContinuity.CurveContinuity": "C7",
                "SplitContinuity.SurfaceContinuity": "C1",
                "SplitContinuity.Curve2dContinuity": "C1",
                "SplitContinuity.MaxTolerance": 1e-7,
            },
            "CurveContinuity takes a continuity, one of C0, G1, C1, G2, C2, C3, CN",
        ),
    ],
    ids=[
        "no_operator",
        "unknown_operators",
        "missing",
        "unread",
        "str_for_real",
        "bool_for_real",
        "infinite",
        "int_for_bool",
        "float_for_integer",
        "integer_range",
        "bad_continuity",
    ],
)
def test_shape_process_refuses_a_bad_sequence_or_parameter(
    operators: list[str], parameters: dict[str, ShapeProcessValue], match: str
) -> None:
    s, solid = cylinder()
    before = state(s)

    with pytest.raises(ps.PysmeshError, match=match) as err:
        s.shape_process([solid], operators, parameters)

    assert "Session.shape_process" in str(err.value)
    assert state(s) == before


def test_shape_process_refuses_no_entity_a_dead_id_and_a_change_of_nothing() -> None:
    s = Session()
    s.add_box(3.0, 7.0, 11.0)
    box = EntityId(int(s.entities(EntityKind.SOLID)[0]))
    s.add_box(1.0, 1.0, 1.0, origin=(20.0, 0.0, 0.0))
    dead = EntityId(int(s.entities(EntityKind.SOLID)[-1]))
    s.remove([dead])
    before = state(s)

    with pytest.raises(ps.PysmeshError, match="no entity is named"):
        s.shape_process([], ["DirectFaces"], {})
    with pytest.raises(ps.PysmeshError, match=f"entity {dead} is dead"):
        s.shape_process([dead], ["DirectFaces"], {})
    with pytest.raises(
        ps.PysmeshError,
        match=f"the sequence 'DirectFaces' changed nothing in the bodies of the entities {box}",
    ):
        s.shape_process([box], ["DirectFaces"], {})

    assert state(s) == before


def test_shape_process_refuses_a_solid_placed_twice_in_its_bodies() -> None:
    """One solid TShape placed at x = 0 and x = 10 (an instanced part), both named."""
    s, _ = cylinder()
    text = s.brep().decode()
    placed_twice = text.replace(
        "Locations 0\n", "Locations 1\n1\n 1 0 0 10 \n 0 1 0 0 \n 0 0 1 0 \n"
    ).replace("Co\n\n1100000\n+2 0 *", "Co\n\n1100000\n+2 0 +2 1 *")
    t = Session()
    t.add_brep(placed_twice.encode())
    solids = [EntityId(int(i)) for i in t.entities(EntityKind.SOLID)]
    before = state(t)

    with pytest.raises(ps.PysmeshError, match="a SOLID of the bodies is placed twice"):
        t.shape_process(solids, ["SplitAngle"], SPLIT_ANGLE)

    assert len(solids) == 2
    assert state(t) == before


def _reference_table() -> dict[str, dict[str, tuple[str, str]]]:
    """The reference page's tables: operator -> parameter -> (type, GEOM's value)."""
    text = REFERENCE_PAGE.read_text(encoding="utf-8")
    table: dict[str, dict[str, tuple[str, str]]] = {}
    operator = ""
    for line in text.splitlines():
        heading = re.match(r"^### (\w+)$", line)
        if heading:
            operator = heading.group(1)
            table[operator] = {}
            continue
        row = re.match(r"^\| `(\w+)\.(\w+)` \| (\w+) \| (\S+) \|$", line)
        if row:
            assert row.group(1) == operator, line
            table[operator][row.group(2)] = (row.group(3), row.group(4))
    return table


def _value(kind: str, text: str) -> ShapeProcessValue:
    """A reference-table value of a type, as shape_process takes it."""
    if kind == "real":
        return float(text)
    if kind == "integer":
        return int(text)
    if kind == "bool":
        return text == "1"
    return text


def test_the_reference_page_lists_every_parameter_each_operator_reads() -> None:
    """For each of the 18 operators the page's parameters are accepted, none is missing or
    unread, and leaving out any one is refused as missing."""
    table = _reference_table()
    s, solid = cylinder()

    accepted: list[str] = []
    for operator, rows in table.items():
        params: dict[str, ShapeProcessValue] = {}
        for name, (kind, geom) in rows.items():
            full = f"{operator}.{name}"
            params[full] = OCCT_VALUES[full] if geom == "-" else _value(kind, geom)
        mark = s.snapshot()
        try:
            s.shape_process([solid], [operator], params)
        except ps.PysmeshError as err:
            assert "changed nothing" in str(err), (operator, str(err))
        s.restore(mark)
        for full in params:
            short = {k: v for k, v in params.items() if k != full}
            with pytest.raises(ps.PysmeshError, match=f"read {re.escape(full)}, which"):
                s.shape_process([solid], [operator], short)
        accepted.append(operator)

    assert len(accepted) == 18
    assert sum(len(rows) for rows in table.values()) == 121
