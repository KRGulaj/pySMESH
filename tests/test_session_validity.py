# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-03

"""Gates for the results a session commits: a committed shape is the shape asked for.

``BRepCheck_Analyzer`` tests the local consistency of the topology only. These tests
assert the properties a mesher needs, each against an oracle that does not depend on the
operation under test (report ``defect_sweep_4.2.2.md`` §6):

* **V1, an empty boolean result.** ``common`` and ``cut`` near a coincident face could
  return nothing for operands that overlap. Each cell of the near-coincident grids must
  now raise or give the closed-form volume.
* **V2, a solid that is not watertight.** A ``cut`` grazing the wing's nose committed
  a solid with an edge on one face only. ``free_boundary_edges`` on the BREP is the
  oracle.
* **V3, an inside-out solid on import.** A box whose solid names its shell reversed
  loaded at volume -1. The oracle is the box's own volume and the point classifier.
* **V5, a sweep that crosses itself.** A Frenet sweep of a profile tilted from the
  spine's start tangent committed a surface that passes through itself. With the profile
  perpendicular to the spine the tube is exact, and its volume is ``pi r^2 L`` (Pappus).
* **A7, an invalid result names nothing.** A near-coincident ``common`` raised "invalid
  shape" with empty details. It must name a ``BRepCheck`` status and the faces.

Volumes are integrated adaptively on a fresh session read from the BREP, so that one
solid carries one id (report §4 C5) and the rule is the adaptive one (report §5 D3).
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import EntityId, EntityKind, PysmeshError, Session

FUZZY_VALUES: tuple[float, ...] = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
OFFSETS: tuple[float, ...] = (
    -1e-4,
    -3e-5,
    -1e-5,
    -3e-6,
    -1e-6,
    -3e-7,
    -1e-7,
    0.0,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
)

# The half-space box: y in [d, d + 3], x and z in [-1.5, 1.5].
HALF_SPACE_SIZE: float = 3.0
HALF_SPACE_LOW: float = -1.5

# A fuzzy boolean may treat two features closer than fuzzy plus the shape tolerance as
# one, so a correct result is the closed form at an offset within that distance of d.
# The loft's own tolerance stays below this bound; the primitives carry 1e-7.
SHAPE_TOLERANCE_BOUND: float = 2e-6
VOLUME_PRECISION: float = 1e-8

# The 1-edge NACA 0012 wing of the report: 3 sections, chord 1 - 0.1 z, span 2.
WING_SPAN: float = 2.0
WING_SECTIONS: int = 3
WING_POINTS: int = 60
WING_THICKNESS: float = 0.12
# Inside the half-space box (z in [0, 1.5]) the wing's chord plane covers
# int_0^1.5 (1 - 0.1 z) dz = 1.3875. At height y the trailing edge recedes by
# y / 0.14535 (the NACA 00xx closed-TE slope at x = c), so the section area at height
# y is 1.3875 - 1.5 |y| / 0.14535 to first order.
WING_PLANFORM_IN_BOX: float = 1.3875
WING_TE_RECESSION: float = 1.5 / 0.14535


def _naca(chord: float) -> NDArray[np.float64]:
    """NACA 00xx with a sharp trailing edge: closed loop TE, upper, LE, lower, TE."""
    b = np.linspace(0.0, np.pi, WING_POINTS)
    x = 0.5 * (1.0 - np.cos(b))
    yt = (
        5.0
        * WING_THICKNESS
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
    return np.asarray(np.vstack([upper, lower]) * chord, dtype=np.float64)


def _new_ids(s: Session, kind: EntityKind, before: set[EntityId]) -> list[EntityId]:
    """The ids of ``kind`` that are alive now and were not in ``before``."""
    return [e for e in s.entities(kind) if e not in before]


@functools.cache
def _wing_brep(two_edge: bool) -> bytes:
    """The lofted wing as BREP bytes, lofted once per process.

    1-edge: each section one closed spline. 2-edge: each section an upper and a lower
    spline joined into a wire.
    """
    s = Session()
    sections = []
    for z in np.linspace(0.0, WING_SPAN, WING_SECTIONS):
        p = np.c_[_naca(1.0 - 0.1 * float(z)), np.full(2 * WING_POINTS - 1, z)]
        before = set(s.entities(EntityKind.EDGE))
        if two_edge:
            half = len(p) // 2
            s.add_spline(p[: half + 1])
            s.add_spline(p[half:])
            s.make_wire(_new_ids(s, EntityKind.EDGE, before))
        else:
            s.add_spline(p)
        sections.append(_new_ids(s, EntityKind.EDGE, before))
    s.thru_sections(sections, solid=True, ruled=False)
    return s.brep()


def _import(s: Session, brep: bytes) -> list[EntityId]:
    """Add ``brep`` to ``s``; return the new solid ids."""
    before = set(s.entities(EntityKind.SOLID))
    s.add_brep(brep)
    return _new_ids(s, EntityKind.SOLID, before)


def _wing(s: Session) -> list[EntityId]:
    """Add the 1-edge wing to ``s``; return its solid id."""
    return _import(s, _wing_brep(two_edge=False))


def _wing_two_edge(s: Session) -> list[EntityId]:
    """Add the 2-edge wing to ``s``; return its solid id."""
    return _import(s, _wing_brep(two_edge=True))


def _sphere(s: Session) -> list[EntityId]:
    """Add the unit sphere at the origin to ``s``; return its solid id."""
    s.add_sphere(1.0)
    return list(s.entities(EntityKind.SOLID))


def _volume(s: Session) -> float:
    """The adaptive volume of every distinct solid of the session's BREP."""
    fresh = Session()
    fresh.add_brep(s.brep())
    ids = list(fresh.entities(EntityKind.SOLID))
    if not ids:
        return 0.0
    return float(fresh.mass_properties(ids, precision=VOLUME_PRECISION).measure.sum())


def _half_space_boolean(
    body: Callable[[Session], list[EntityId]], op: str, d: float, fuzzy: float
) -> float | None:
    """``op`` of the body with the box y > d: the result's volume, None on a raise."""
    s = Session()
    a = body(s)
    before = set(s.entities(EntityKind.SOLID))
    s.add_box(
        HALF_SPACE_SIZE,
        HALF_SPACE_SIZE,
        HALF_SPACE_SIZE,
        origin=(HALF_SPACE_LOW, d, HALF_SPACE_LOW),
    )
    b = _new_ids(s, EntityKind.SOLID, before)
    try:
        getattr(s, op)(a, b, fuzzy=fuzzy)
    except PysmeshError:
        return None
    return _volume(s)


def _sphere_cap(d: float) -> float:
    """Volume of the unit sphere above y = d: pi h^2 (3 - h) / 3, h = 1 - d."""
    h = 1.0 - d
    return math.pi * h * h * (3.0 - h) / 3.0


def _wing_above(d: float, v0: float) -> float:
    """The wing's volume above y = d inside the box, from its volume ``v0`` above y = 0.

    The slab between y = 0 and y = d has the section area 1.3875 - 10.32 |y|, so its
    volume is 1.3875 d - 5.16 d |d|. The leading edge adds a term of order d^3.
    """
    return v0 - WING_PLANFORM_IN_BOX * d + 0.5 * WING_TE_RECESSION * d * abs(d)


def _bad_cells(
    volumes: dict[float, float | None], exact: Callable[[float], float], d: float
) -> list[str]:
    """The cells of one row whose volume is no closed form within fuzzy of d.

    The closed form decreases as d grows, so a result within fuzzy plus the shape
    tolerance of d lies between the closed forms at the two ends of that interval.
    """
    bad = []
    for fuzzy, volume in volumes.items():
        if volume is None:
            continue
        reach = fuzzy + SHAPE_TOLERANCE_BOUND
        low, high = exact(d + reach), exact(d - reach)
        margin = VOLUME_PRECISION * high
        if not low - margin <= volume <= high + margin:
            bad.append(
                f"fuzzy={fuzzy:g}: volume {volume:.9g} not in [{low:.9g}, {high:.9g}]"
            )
    return bad


@pytest.mark.parametrize("d", OFFSETS)
def test_common_of_the_sphere_and_a_near_coincident_half_space_is_correct_or_refused(
    d: float,
) -> None:
    """Every fuzzy value: a raise, or the spherical cap above y = d (report §6 V1)."""
    volumes = {fz: _half_space_boolean(_sphere, "common", d, fz) for fz in FUZZY_VALUES}

    bad = _bad_cells(volumes, _sphere_cap, d)

    assert not bad, bad


@pytest.fixture(scope="module")
def wing_above_chord_plane() -> float:
    """The wing's volume above y = 0 inside the box: the exact-coincidence common."""
    volume = _half_space_boolean(_wing, "common", 0.0, 0.0)
    assert volume is not None
    return volume


@pytest.mark.parametrize("d", OFFSETS)
def test_common_of_the_wing_and_a_near_coincident_half_space_is_correct_or_refused(
    d: float, wing_above_chord_plane: float
) -> None:
    """Every fuzzy value: a raise, or the wing above y = d from the slab closed form."""
    volumes = {fz: _half_space_boolean(_wing, "common", d, fz) for fz in FUZZY_VALUES}

    bad = _bad_cells(volumes, lambda y: _wing_above(y, wing_above_chord_plane), d)

    assert not bad, bad


def test_cut_of_the_wing_by_a_near_coincident_half_space_is_correct_or_refused(
    wing_above_chord_plane: float,
) -> None:
    """The cut keeps the wing below y = d (d = -1e-6): its volume less the rest."""
    s = Session()
    _wing(s)
    wing_volume = _volume(s)
    d = -1e-6
    volumes = {fz: _half_space_boolean(_wing, "cut", d, fz) for fz in FUZZY_VALUES}

    bad = _bad_cells(
        volumes,
        lambda y: wing_volume - _wing_above(-y, wing_above_chord_plane),
        -d,
    )

    assert not bad, bad


def test_a_boolean_reports_occts_warnings_on_the_delta() -> None:
    """OCCT warns while it builds the sphere's cap at y = -1e-7; the delta has it."""
    s = Session()
    a = _sphere(s)
    before = set(s.entities(EntityKind.SOLID))
    s.add_box(3.0, 3.0, 3.0, origin=(-1.5, -1e-7, -1.5))
    b = _new_ids(s, EntityKind.SOLID, before)

    delta = s.common(a, b)

    assert "BOPAlgo_AlertUnableToOrientTheShape" in delta.warnings
    assert all(isinstance(w, str) for w in delta.warnings)


def test_a_clean_boolean_reports_no_warning() -> None:
    """Two overlapping boxes: the common is their overlap, and OCCT warns of nothing."""
    s = Session()
    s.add_box(2.0, 2.0, 2.0)
    a = list(s.entities(EntityKind.SOLID))
    s.add_box(2.0, 2.0, 2.0, origin=(1.0, 1.0, 1.0))
    b = _new_ids(s, EntityKind.SOLID, set(a))

    delta = s.common(a, b)

    assert delta.warnings == ()
    assert _volume(s) == pytest.approx(1.0, rel=1e-12)


def test_the_empty_common_of_two_disjoint_spheres_is_accepted() -> None:
    """Disjoint spheres whose bounding boxes overlap: an empty common is the answer."""
    s = Session()
    s.add_sphere(1.0)
    a = list(s.entities(EntityKind.SOLID))
    s.add_sphere(1.0, centre=(1.5, 1.5, 1.5))
    b = _new_ids(s, EntityKind.SOLID, set(a))

    s.common(a, b)

    assert list(s.entities(EntityKind.SOLID)) == []


def test_the_common_of_two_boxes_touching_on_a_face_is_accepted() -> None:
    """Face-touching boxes share no interior, so no result of theirs is refused."""
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    a = list(s.entities(EntityKind.SOLID))
    s.add_box(1.0, 1.0, 1.0, origin=(1.0, 0.0, 0.0))
    b = _new_ids(s, EntityKind.SOLID, set(a))

    s.common(a, b)

    assert _volume(s) == 0.0


def test_the_empty_cut_of_a_solid_inside_its_tool_is_accepted() -> None:
    """A box inside a larger box: cutting it by the larger one leaves nothing."""
    s = Session()
    s.add_box(1.0, 1.0, 1.0, origin=(1.0, 1.0, 1.0))
    a = list(s.entities(EntityKind.SOLID))
    s.add_box(3.0, 3.0, 3.0)
    b = _new_ids(s, EntityKind.SOLID, set(a))

    s.cut(a, b)

    assert list(s.entities(EntityKind.SOLID)) == []


@pytest.mark.parametrize(
    ("body", "free_edges"), [(_wing, 2), (_wing_two_edge, 1)], ids=["1-edge", "2-edge"]
)
def test_a_cut_that_leaves_a_free_boundary_edge_is_refused_naming_the_edges(
    body: Callable[[Session], list[EntityId]], free_edges: int
) -> None:
    """The wing cut by the box y > 1e-4 at fuzzy 1e-4 left 2 (1-edge) or 1 free edge."""
    s = Session()
    a = body(s)
    before = set(s.entities(EntityKind.SOLID))
    s.add_box(3.0, 3.0, 3.0, origin=(-1.5, 1e-4, -1.5))
    b = _new_ids(s, EntityKind.SOLID, before)
    brep = s.brep()

    with pytest.raises(PysmeshError) as info:
        s.cut(a, b, fuzzy=1e-4)

    number = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    lengths = [float(x) for x in re.findall(rf"length ({number})", info.value.details)]
    assert f"{free_edges} free boundary edge" in str(info.value)
    assert len(lengths) == free_edges
    assert all(length > 0.0 for length in lengths)
    assert s.brep() == brep


# ---- V3: an inside-out solid on import --------------------------------------------- #


def _inside_out_box() -> bytes:
    """A unit box whose solid names its shell reversed (``+`` to ``-``, report V3)."""
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    text = s.brep().decode()
    k = text.rindex("So\n\n0100000\n+")
    return (text[:k] + text[k:].replace("\n+", "\n-", 1)).encode()


INSIDE_AND_FAR: list[list[float]] = [[0.5, 0.5, 0.5], [1000.0, 1000.0, 1000.0]]


def test_add_brep_refuses_an_inside_out_solid_naming_it() -> None:
    """The reversed box is refused by default; the session stays empty."""
    s = Session()

    with pytest.raises(PysmeshError) as info:
        s.add_brep(_inside_out_box())

    assert "inside out" in str(info.value)
    assert "solid 1" in str(info.value)
    assert list(s.entities(EntityKind.SOLID)) == []


def test_load_brep_refuses_an_inside_out_solid_naming_it() -> None:
    """The stateless reader refuses the reversed box the same way."""
    with pytest.raises(PysmeshError) as info:
        ps.load_brep(_inside_out_box())

    assert "inside out" in str(info.value)
    assert "solid 1" in str(info.value)


def test_add_brep_reverses_an_inside_out_solid_on_request() -> None:
    """``inside_out="reverse"``: volume +1, inside is inside, and the delta says so."""
    s = Session()

    delta = s.add_brep(_inside_out_box(), inside_out="reverse")

    solids = list(s.entities(EntityKind.SOLID))
    assert _volume(s) == pytest.approx(1.0, rel=1e-12)
    assert s.contains(solids, INSIDE_AND_FAR).tolist() == [[True, False]]
    assert any("solid 1" in w and "reversed" in w for w in delta.warnings)


def test_load_brep_reverses_an_inside_out_solid_on_request() -> None:
    """The stateless reader reverses the box on request: its volume is +1."""
    shape = ps.load_brep(_inside_out_box(), inside_out="reverse")

    assert shape.solids()[0].volume == pytest.approx(1.0, rel=1e-12)


# ---- V5: a sweep that crosses itself ----------------------------------------------- #

SWEEP_SPINE: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 3.0),
    (0.0, 2.0, 6.0),
    (2.0, 2.0, 9.0),
]
SWEEP_RADIUS: float = 0.5
# The profile normal of report V5: the chord to the second spine point, 31.7 degrees
# from the spline's own start tangent.
SWEEP_TILTED_NORMAL: tuple[float, float, float] = (
    1.0 / math.sqrt(10.0),
    0.0,
    3.0 / math.sqrt(10.0),
)
# The swept surface approximates the exact tube: its volume is within 7.5e-8 relative
# of pi r^2 L when the profile is perpendicular to the spine; 1e-6 leaves a margin of
# 13.
SWEEP_VOLUME_RTOL: float = 1e-6


def _sweep(perpendicular: bool, frenet: bool) -> tuple[Session, float]:
    """Sweep the circle along the spline; return the session and the spine's length.

    The profile is perpendicular to the spine's start tangent, or tilted as in report
    V5. The sweep runs here, so a refusal raises from this call.
    """
    s = Session()
    s.add_spline(SWEEP_SPINE)
    spine = list(s.entities(EntityKind.EDGE))
    length = float(s.mass_properties(spine, precision=1e-9).measure[0])
    if perpendicular:
        start = s.edge_parameter_bounds(spine)[0, 0]
        normal = tuple(float(x) for x in s.curve_at(spine[0], [start]).tangents[0])
    else:
        normal = SWEEP_TILTED_NORMAL
    before = set(s.entities(EntityKind.EDGE))
    s.add_circle((0.0, 0.0, 0.0), normal, SWEEP_RADIUS)
    profile = _new_ids(s, EntityKind.EDGE, before)
    s.pipe_shell(spine, profile, frenet=frenet, solid=True)
    return s, length


def test_a_frenet_sweep_that_crosses_itself_is_refused_naming_the_faces() -> None:
    """Report V5: the tilted profile's Frenet sweep passed BRepCheck_Analyzer."""
    with pytest.raises(PysmeshError) as info:
        _sweep(perpendicular=False, frenet=True)

    assert "interfere" in str(info.value)
    assert "face" in info.value.details


def test_a_frenet_sweep_of_a_perpendicular_profile_is_the_exact_tube() -> None:
    """Pappus: a circle swept perpendicular to a spine it never folds over: pi r^2 L."""
    s, length = _sweep(perpendicular=True, frenet=True)

    volume = _volume(s)

    assert volume == pytest.approx(
        math.pi * SWEEP_RADIUS**2 * length, rel=SWEEP_VOLUME_RTOL
    )


def test_a_corrected_frenet_sweep_of_the_tilted_profile_is_accepted() -> None:
    """``frenet=False`` builds a consistent solid from the same tilted profile."""
    s, _ = _sweep(perpendicular=False, frenet=False)

    assert len(list(s.entities(EntityKind.SOLID))) == 1
    assert _volume(s) > 0.0


# ---- A7: "invalid shape" names the reason and the faces --------------------------- #


@pytest.mark.parametrize(
    ("body", "origin", "fuzzy"),
    [
        (lambda s: s.add_torus(1.0, 0.3), (-1.5, -1e-6, -1.5), 3e-7),
        (lambda s: s.add_sphere(1.0), (-3e-7, -1.5, -1.5), 0.0),
    ],
    ids=["torus-y-1e-6-fuzzy-3e-7", "sphere-x-3e-7"],
)
def test_an_invalid_boolean_result_names_the_brepcheck_status_and_the_faces(
    body: Callable[[Session], object], origin: tuple[float, float, float], fuzzy: float
) -> None:
    """Report A7: the near-coincident commons raised "invalid shape", naming nothing."""
    s = Session()
    body(s)
    a = list(s.entities(EntityKind.SOLID))
    s.add_box(3.0, 3.0, 3.0, origin=origin)
    b = _new_ids(s, EntityKind.SOLID, set(a))
    faces = {int(f) for f in s.entities(EntityKind.FACE)}

    with pytest.raises(PysmeshError, match="invalid shape") as info:
        s.common(a, b, fuzzy=fuzzy)

    assert re.search(r"BRepCheck_\w+", info.value.details)
    assert info.value.face_ids
    assert set(info.value.face_ids) <= faces


# ---- V5 on a loft whose sections meet at one point (amendment 2, report O3) ------- #

# The wedge of report O3: 5 circles of radius 0.1 in planes rotated about the y axis by
# -30 to +30 degrees, each passing rho from the origin.
POLE_RADIUS: float = 0.1
POLE_SECTIONS: int = 5
POLE_HALF_ANGLE: float = math.radians(30.0)
# The adaptive volume (1e-9) of the rho = 1e-4 wedge, measured on the reference package
# (main at 7d22a1db9), which committed it before the self-interference check existed.
# The test builds its circles with math, the measurement with numpy, whose sine and
# cosine may differ in the last bit, so the two volumes agree to rounding only.
POLE_WEDGE_VOLUME_1E_4: float = 0.003293179662682907
POLE_WEDGE_RTOL: float = 1e-12


def _pole_wedge(s: Session, rho: float) -> list[EntityId]:
    """Loft the O3 wedge with its circles ``rho`` from the origin; return its solid."""
    sections = []
    for th in np.linspace(-POLE_HALF_ANGLE, POLE_HALF_ANGLE, POLE_SECTIONS):
        radial = np.array([math.cos(th), 0.0, -math.sin(th)])
        normal = (math.sin(th), 0.0, math.cos(th))
        centre = tuple(float(x) for x in (POLE_RADIUS + rho) * radial)
        before = set(s.entities(EntityKind.EDGE))
        s.add_circle(centre, normal, POLE_RADIUS)
        sections.append(_new_ids(s, EntityKind.EDGE, before))
    before = set(s.entities(EntityKind.SOLID))
    s.thru_sections(sections, solid=True, ruled=False)
    return _new_ids(s, EntityKind.SOLID, before)


def test_a_loft_through_circles_that_meet_at_one_point_is_refused() -> None:
    """rho = 0: the first and last circles meet at the origin, pinching the boundary."""
    s = Session()

    with pytest.raises(PysmeshError, match="interferes with itself") as info:
        _pole_wedge(s, 0.0)

    assert info.value.details.count("edge ") >= 2
    assert "apart" in info.value.details
    assert list(s.entities(EntityKind.SOLID)) == []


def test_a_loft_through_circles_kept_apart_is_accepted_with_its_volume() -> None:
    """rho = 1e-4: accepted, and its volume is the reference package's (to rounding)."""
    s = Session()
    _pole_wedge(s, 1e-4)

    volume = _volume(s)

    assert volume == pytest.approx(POLE_WEDGE_VOLUME_1E_4, rel=POLE_WEDGE_RTOL, abs=0.0)
