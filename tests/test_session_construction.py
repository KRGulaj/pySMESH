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
  section edges, and OCCT wrote pcurves, surfaces and continuity onto them, also when the
  loft was then refused. The oracle is the BREP of the session, and of a snapshot taken
  before the loft, byte for byte.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from pysmesh import EntityKind, PysmeshError, Session


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


# ---- A2: an OCCT exception outside the try ------------------------------------------- #


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


# ---- A3: a loft writes onto the session's section edges ------------------------------ #

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
    """The loft is refused for its negative volume, and the session's BREP is unchanged."""
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
    """Every section edge and vertex id of a ruled loft stays alive, on the loft's edges.

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
