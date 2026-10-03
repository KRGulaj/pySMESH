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
"""

from __future__ import annotations

import math

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
