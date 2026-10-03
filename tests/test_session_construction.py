# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-03

"""Gates for the construction operations: sweeps, lofts, fillets and faces.

Each claim is asserted against an oracle the operation under test does not produce
(report ``defect_sweep_4.2.2.md`` §2, §4, §6, §9 and §18):

* **A2, every failure is a PysmeshError.** A sweep built its OCCT builder outside the
  ``try`` that converts OCCT's exceptions, so a bad parameter reached Python as a raw
  ``RuntimeError``. The oracle is the public contract: ``PysmeshError`` naming the
  operation, and a session that is byte for byte as it was.
* **A3, a loft writes onto its inputs.** ``thru_sections`` gave OCCT the session's own
  section edges, and OCCT wrote pcurves, surfaces and continuity onto them, also when
  the loft was then refused. The oracle is the BREP of the session, and of a snapshot
  taken before the loft, byte for byte.
* **O4, a bowed end section.** OCCT closes a solid loft with a planar face on each end
  section, and a section bowed out of its plane gave an invalid cap and a message that
  named nothing. The oracle is the bow itself: ``w sin(pi x)`` spreads the section's
  points exactly ``w`` across its best-fit plane.
* **C3, a closed loft.** A ring of circles could not be lofted round onto itself. The
  oracles are a construction without the closed path (two half lofts, sewn) and the
  torus ``2 pi^2 R r^2`` with a stated distance bound.
* **V4, a slit.** A ring closed by a copy of its first section committed two coincident
  caps inside the solid. It must be refused, naming the closed loft as the cure.
* **M2, a face that depends on the process.** ``make_face`` joined two ends 1.8e-16
  apart at either end, run to run. The oracle is five fresh processes and the stated
  rule.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    EntityId,
    EntityKind,
    PysmeshError,
    Session,
    free_boundary_edges,
    load_brep,
)


def _state(s: Session) -> tuple[int, int, bytes]:
    """What a refused operation must leave alone: op count, issued ids, BREP bytes."""
    return (s.op_count, s.issued_id_count, s.brep())


def _square() -> Session:
    """A session holding one planar unit square face."""
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0)
    return s


def _faces(s: Session) -> list[int]:
    """The live face ids of a session."""
    return sorted(int(i) for i in s.entities(EntityKind.FACE))


# ---- A2: an OCCT exception outside the try ----------------------------------------- #


@pytest.mark.parametrize(
    "vector",
    [(math.inf, 0.0, 0.0), (0.0, 0.0, math.inf), (1e308, 1e308, 0.0)],
    ids=["inf_x", "inf_z", "overflowing_norm"],
)
def test_an_extrude_vector_occt_refuses_raises_pysmesh_error_and_changes_nothing(
    vector: tuple[float, float, float],
) -> None:
    """OCCT throws in BRepPrimAPI_MakePrism's constructor; the caller gets PysmeshError.

    An infinite component makes OCCT throw ``BRep_Builder::Infinite parameter``. Two
    components of 1e308 give a norm that overflows to infinity and then a zero direction
    (``gp_Dir() - input vector has zero norm``). The reference let both out raw.
    """
    s = _square()
    before = _state(s)

    with pytest.raises(PysmeshError, match="Session.extrude"):
        s.extrude(_faces(s), vector)

    assert _state(s) == before


@pytest.mark.parametrize(
    "origin", [(math.nan, 0.0, 0.0), (math.inf, 0.0, 0.0)], ids=["nan", "inf"]
)
def test_a_revolve_origin_occt_refuses_raises_pysmesh_error_and_changes_nothing(
    origin: tuple[float, float, float],
) -> None:
    """A non-finite axis origin makes OCCT throw ``NCollection_Sequence::Value``."""
    s = _square()
    before = _state(s)

    with pytest.raises(PysmeshError, match="Session.revolve"):
        s.revolve(_faces(s), origin, (0.0, 1.0, 0.0), math.pi)

    assert _state(s) == before


# ---- A3: a loft writes onto the session's section edges ---------------------------- #

# Two unit triangles, the second shifted across and tilted back, so that the loft folds
# through itself: the solid encloses a volume of -0.2876 and is refused.
FOLDING_TRIANGLES: tuple[tuple[float, float, float, float, float], ...] = (
    # z, shift x, shift y, tilt about x, turn
    (0.0, 1.299, 1.139, 1.213, 1.612),
    (0.606, -0.756, 1.237, -0.223, -0.073),
)


def _triangle(
    z: float, sx: float, sy: float, tilt: float, turn: float
) -> list[tuple[float, float, float]]:
    """A unit triangle turned by ``turn``, tilted about x by ``tilt``, then shifted."""
    points = []
    for q in range(3):
        a = turn + 2.0 * math.pi * q / 3.0
        x, y = math.cos(a), math.sin(a)
        points.append((x + sx, y * math.cos(tilt) + sy, z + y * math.sin(tilt)))
    return points


def _circles(s: Session, count: int) -> list[list[int]]:
    """``count`` coaxial circles up the z axis, radii 1, 1.2, 1.4, ..."""
    return [
        [
            int(i)
            for i in s.add_circle(
                (0.0, 0.0, float(k)), (0.0, 0.0, 1.0), 1.0 + 0.2 * k
            ).created
        ]
        for k in range(count)
    ]


def _squares(s: Session, count: int) -> list[list[int]]:
    """``count`` closed unit squares up the z axis, each turned 0.2 rad more."""
    sections = []
    for k in range(count):
        points = [
            (
                math.cos(0.2 * k + q * math.pi / 2.0),
                math.sin(0.2 * k + q * math.pi / 2.0),
                float(k),
            )
            for q in range(4)
        ]
        sections.append([int(i) for i in s.add_polyline(points, closed=True).created])
    return sections


@pytest.mark.parametrize("ruled", [False, True], ids=["smooth", "ruled"])
def test_a_refused_loft_leaves_the_session_byte_identical(ruled: bool) -> None:
    """The loft is refused for its negative volume; the session's BREP is unchanged."""
    s = Session()
    sections = [
        [int(i) for i in s.add_polyline(_triangle(*t), closed=True).created]
        for t in FOLDING_TRIANGLES
    ]
    before = _state(s)

    with pytest.raises(PysmeshError, match="encloses a volume of -0.28"):
        s.thru_sections(sections, solid=True, ruled=ruled)

    assert _state(s) == before


@pytest.mark.parametrize(
    ("build", "solid"),
    [(_circles, True), (_squares, False)],
    ids=["circles_solid", "squares_shell"],
)
def test_a_ruled_loft_leaves_a_snapshot_taken_before_it_byte_identical(
    build: Callable[[Session, int], list[list[int]]], solid: bool
) -> None:
    """A snapshot shares the section edges; restored after the loft, it is unchanged."""
    s = Session()
    sections = build(s, 3)
    mark = s.snapshot()
    before = s.brep()

    s.thru_sections(sections, solid=solid, ruled=True)
    s.restore(mark)

    assert s.brep() == before


def test_a_ruled_loft_carries_the_ids_of_its_section_edges_and_vertices() -> None:
    """Every section edge and vertex id of a ruled loft stays alive on the loft's edges.

    The ruled loft keeps each section as a row of its edges, so each id must denote a
    shape of the result: 12 edges and 12 vertices for three squares.
    """
    s = Session()
    sections = _squares(s, 3)
    edges = {int(i) for i in s.entities(EntityKind.EDGE)}
    vertices = {int(i) for i in s.entities(EntityKind.VERTEX)}

    s.thru_sections(sections, solid=True, ruled=True)

    assert edges <= {int(i) for i in s.entities(EntityKind.EDGE)}
    assert vertices <= {int(i) for i in s.entities(EntityKind.VERTEX)}
    assert len(edges) == 12
    assert len(vertices) == 12


# ---- O4: a solid loft whose end section is not planar ------------------------------ #

NACA_POINTS: int = 60
NACA_THICKNESS: float = 0.12
# A bow of w sin(pi x) out of the section plane, over the chord x in [0, 1]. It is 0 at
# both ends of the chord and w at mid-chord, so the section's points spread exactly w
# across the plane z = const, the best-fit plane of the bow.
BOWS: tuple[float, ...] = (1e-7, 1e-5, 1e-2)
NUMBER: str = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _naca() -> NDArray[np.float64]:
    """NACA 0012, chord 1, sharp trailing edge: closed loop TE, upper, LE, lower, TE."""
    b = np.linspace(0.0, np.pi, NACA_POINTS)
    x = 0.5 * (1.0 - np.cos(b))
    yt = (
        5.0
        * NACA_THICKNESS
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1036 * x**4
        )
    )
    upper = np.c_[x[::-1], yt[::-1]]
    lower = np.c_[x[1:], -yt[1:]]
    return np.asarray(np.vstack([upper, lower]), dtype=np.float64)


def _bowed_wing(s: Session, bows: tuple[float, float, float]) -> list[list[int]]:
    """NACA sections at z = 0, 1, 2, each two splines; section k is bowed by bows[k]."""
    sections = []
    for z, bow in zip((0.0, 1.0, 2.0), bows, strict=True):
        p2 = _naca()
        p = np.c_[p2, z + bow * np.sin(np.pi * p2[:, 0])]
        before = {int(i) for i in s.entities(EntityKind.EDGE)}
        half = len(p) // 2
        s.add_spline(p[: half + 1])
        s.add_spline(p[half:])
        new = [int(i) for i in s.entities(EntityKind.EDGE) if int(i) not in before]
        s.make_wire(new)
        sections.append(
            [int(i) for i in s.entities(EntityKind.EDGE) if int(i) not in before]
        )
    return sections


@pytest.mark.parametrize("bow", BOWS)
@pytest.mark.parametrize(("which", "index"), [("first", 0), ("last", 2)])
def test_a_solid_loft_with_a_bowed_end_section_names_it_and_its_deviation(
    which: str, index: int, bow: float
) -> None:
    """The refusal names the end section and gives a deviation within 10 % of w."""
    s = Session()
    bows = [0.0, 0.0, 0.0]
    bows[index] = bow
    sections = _bowed_wing(s, (bows[0], bows[1], bows[2]))
    before = _state(s)

    with pytest.raises(PysmeshError) as info:
        s.thru_sections(sections, solid=True, ruled=False)

    message = str(info.value)
    assert f"section {index + 1} of 3 ({which})" in message
    found = re.search(rf"spread ({NUMBER}) across", message)
    assert found is not None, message
    assert float(found.group(1)) == pytest.approx(bow, rel=0.1)
    assert _state(s) == before


@pytest.mark.parametrize(
    ("bows", "solid"),
    [((0.0, 1e-2, 0.0), True), ((0.0, 0.0, 0.0), True), ((1e-2, 0.0, 1e-2), False)],
    ids=["middle_bowed", "planar", "shell"],
)
def test_a_loft_whose_end_sections_need_no_planar_cap_is_committed(
    bows: tuple[float, float, float], solid: bool
) -> None:
    """A bowed middle section, a planar loft and a shell need no cap on a bow."""
    s = Session()
    sections = _bowed_wing(s, bows)

    delta = s.thru_sections(sections, solid=solid, ruled=False)

    assert delta.valid is True
    assert len(s.entities(EntityKind.SOLID)) == (1 if solid else 0)


# ---- C3: a closed loft ------------------------------------------------------------- #

RING_R: float = 1.0
RING_TUBE: float = 0.2
RING_STATIONS: int = 8
# 2 pi^2 R r^2, the torus the ring of circles samples.
TORUS_VOLUME: float = 2.0 * math.pi**2 * RING_R * RING_TUBE**2
# Parameter grid per face for the distance of the smooth loft from the torus. The
# largest distance found changes by 0.1 % from 65 to 257 points a side.
TORUS_GRID: int = 129


def _ring_station(s: Session, phi: float) -> list[int]:
    """A tube-radius circle round the ring at angle ``phi``, normal along the ring."""
    centre = (RING_R * math.cos(phi), RING_R * math.sin(phi), 0.0)
    normal = (-math.sin(phi), math.cos(phi), 0.0)
    return [int(i) for i in s.add_circle(centre, normal, RING_TUBE).created]


def _ring(s: Session) -> list[list[int]]:
    """The stations of the ring, with the first named again as the last."""
    stations = [
        _ring_station(s, 2.0 * math.pi * k / RING_STATIONS)
        for k in range(RING_STATIONS)
    ]
    return [*stations, stations[0]]


def _volume(s: Session) -> float:
    """The volume of the session's solids, integrated adaptively."""
    solids = sorted(int(i) for i in s.entities(EntityKind.SOLID))
    return float(s.mass_properties(solids, precision=1e-12).measure.sum())


def _copy_of(s: Session, section: list[int]) -> list[int]:
    """The edges of a copy of one section."""
    before = {int(i) for i in s.entities(EntityKind.EDGE)}
    s.copy(section)
    return [int(i) for i in s.entities(EntityKind.EDGE) if int(i) not in before]


@pytest.mark.parametrize("ruled", [True, False], ids=["ruled", "smooth"])
def test_a_closed_loft_has_no_free_edge_no_planar_face_and_is_solid_at_the_seam(
    ruled: bool,
) -> None:
    """The ring closes onto itself: no cap, no slit, solid where the caps would be.

    Points of the old cap plane (y = 0, x > 0) up to 0.95 r from the tube axis must
    classify inside: a slit there would put them on the boundary.
    """
    s = Session()

    s.thru_sections(_ring(s), solid=True, ruled=ruled)

    shape = load_brep(s.brep())
    assert [f.surface_type for f in shape.faces() if f.surface_type == "Plane"] == []
    assert len(free_boundary_edges(s.brep())) == 0
    rho = np.linspace(0.0, 0.95 * RING_TUBE, 6)
    theta = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    points = np.array(
        [(RING_R + q * math.cos(t), 0.0, q * math.sin(t)) for q in rho for t in theta]
    )
    solids = sorted(int(i) for i in s.entities(EntityKind.SOLID))
    assert bool(np.all(s.contains(solids, points)))


def test_a_closed_ruled_loft_has_the_volume_of_two_sewn_half_lofts() -> None:
    """The oracle is built without the closed path: two half lofts, sewn into one solid.

    Each half runs through copies of stations 0 and 4, as the report's workaround does,
    so that its strips are the closed loft's strips.
    """
    s = Session()
    s.thru_sections(_ring(s), solid=True, ruled=True)
    closed = _volume(s)
    halves = Session()
    stations = _ring(halves)[:-1]
    last = _copy_of(halves, stations[0])
    middle = _copy_of(halves, stations[4])
    halves.thru_sections(stations[:5], solid=False, ruled=True)
    halves.thru_sections([middle, *stations[5:], last], solid=False, ruled=True)
    faces = sorted(int(i) for i in halves.entities(EntityKind.FACE))

    halves.sew(faces, tolerance=1e-6, make_solid=True)

    assert closed == pytest.approx(_volume(halves), rel=1e-9)
    assert closed == pytest.approx(0.710861, rel=1e-6)


def test_a_closed_smooth_loft_is_within_its_distance_bound_of_the_torus() -> None:
    """|V - 2 pi^2 R r^2| <= eps A + 2 pi^2 R eps^2, eps the distance from the torus.

    If the loft surface lies within eps of the torus surface, the solid lies between the
    tori of tube radius r - eps and r + eps, whose volumes are 2 pi^2 R (r -+ eps)^2.
    That gives the bound, with A = 4 pi^2 R r the torus area. eps is the largest
    distance of a 129 x 129 parameter grid on each face from the torus, taken 5 %
    larger for the points between the grid points.
    """
    s = Session()
    s.thru_sections(_ring(s), solid=True, ruled=False)
    gap = 0.0
    for face in sorted(int(i) for i in s.entities(EntityKind.FACE)):
        u0, u1, v0, v1 = s.face_parameter_bounds([EntityId(face)])[0]
        u, v = np.meshgrid(
            np.linspace(u0, u1, TORUS_GRID), np.linspace(v0, v1, TORUS_GRID)
        )
        p = s.surface_at(EntityId(face), np.c_[u.ravel(), v.ravel()]).points
        d = np.abs(np.hypot(np.hypot(p[:, 0], p[:, 1]) - RING_R, p[:, 2]) - RING_TUBE)
        gap = max(gap, float(d.max()))
    eps = 1.05 * gap
    bound = (
        eps * 4.0 * math.pi**2 * RING_R * RING_TUBE + 2.0 * math.pi**2 * RING_R * eps**2
    )

    volume = _volume(s)

    assert abs(volume - TORUS_VOLUME) <= bound


@pytest.mark.parametrize("repeat", [1, 2], ids=["second", "third"])
def test_a_section_named_again_other_than_first_as_last_is_refused(repeat: int) -> None:
    """Only the first section may come back, as the last; the message names both."""
    s = Session()
    stations = _ring(s)[:-1]
    sections = [*stations[:4], stations[repeat]]

    with pytest.raises(PysmeshError, match=rf"sections \({repeat + 1} and 5\)"):
        s.thru_sections(sections, solid=True, ruled=True)


# ---- V4: a loft closed by a copy of its first section ------------------------------ #


@pytest.mark.parametrize("ruled", [True, False], ids=["ruled", "smooth"])
def test_a_ring_closed_by_a_copied_section_is_refused_pointing_to_the_closed_loft(
    ruled: bool,
) -> None:
    """Two coincident caps make a slit of zero thickness; the refusal names the cure."""
    s = Session()
    stations = _ring(s)[:-1]
    copy = _copy_of(s, stations[0])
    before = _state(s)

    with pytest.raises(PysmeshError, match="lie in one plane and meet") as info:
        s.thru_sections([*stations, copy], solid=True, ruled=ruled)

    assert "name the first section again as the last" in info.value.details
    assert _state(s) == before


# ---- M2: make_face on two near-coincident ends ------------------------------------- #

# The B-spline arc of a radius-1.5 half disk ends at (-1.5, 1.5 sin(pi)), which is
# (-1.5, 1.8e-16); the closing line starts at (-1.5, 0). make_face joins the two ends.
M2_RUNS: int = 5
M2_CHILD_TIMEOUT_S: float = 120.0
_M2_CHILD: str = """
import hashlib, math, os, sys
occt = os.environ.get("PYSMESH_OCCT_BIN")
if occt:
    os.add_dll_directory(occt)
lib = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(lib):
    os.add_dll_directory(lib)
sys.path.insert(0, sys.argv[1])
import numpy as np
import pysmesh as ps

r = 1.5
s = ps.Session()
t = np.linspace(0.0, math.pi, 13)
s.add_spline(np.c_[r * np.cos(t), r * np.sin(t), 0.0 * t])
s.add_line((-r, 0.0, 0.0), (r, 0.0, 0.0))
arc, line = sorted(int(i) for i in s.entities(ps.EntityKind.EDGE))
s.make_face([arc, line] if sys.argv[2] == "arc_first" else [line, arc])
brep = s.brep()
corner = [v.xyz for v in ps.load_brep(brep).vertices() if v.xyz[0] < 0.0]
sha = hashlib.sha256(brep).hexdigest()
print("M2-RESULT", sha, repr(float(corner[0][1])), len(corner))
"""


def _m2_child(order: str) -> tuple[str, float, int]:
    """Build the half disk in a fresh process; its BREP hash and the kept corner's y.

    Raises:
        AssertionError: The child crashed or reported no result.
    """
    package_root = str(Path(ps.__file__).resolve().parent.parent)
    proc = subprocess.run(
        [sys.executable, "-c", _M2_CHILD, package_root, order],
        capture_output=True,
        text=True,
        timeout=M2_CHILD_TIMEOUT_S,
        env=dict(os.environ),
        check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("M2-RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-2000:]
    _, sha, y, count = lines[0].split()
    return sha, float(y), int(count)


@pytest.mark.parametrize(
    ("order", "kept_y"),
    [("arc_first", 1.5 * math.sin(math.pi)), ("line_first", 0.0)],
)
def test_make_face_on_near_coincident_ends_is_the_same_in_every_process(
    order: str, kept_y: float
) -> None:
    """Five processes give one BREP; the joined corner sits on the end named first.

    The rule: ends that coincide within their tolerances but not exactly are joined at
    the end of the edge named first (each edge's first vertex before its last). The two
    candidates are 1.8e-16 apart; the arc's end is its B-spline evaluated at its last
    parameter, which may differ from 1.5 sin(pi) in the last bit, so 1e-24 separates
    them.
    """
    runs = [_m2_child(order) for _ in range(M2_RUNS)]

    assert len({sha for sha, _, _ in runs}) == 1
    assert {count for _, _, count in runs} == {1}
    assert runs[0][1] == pytest.approx(kept_y, abs=1e-24)
