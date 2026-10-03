# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-03

"""Gates for what a compute reports: warnings, missing assignments, and failed hooks.

Each claim is asserted against an oracle the report itself does not produce (report
``defect_sweep_4.2.2.md`` §2):

* **A1, a warning is not a failure.** SMESH marks a sub-mesh computed with a warning
  (``COMPERR_WARNING``, ``SMESH_ComputeError.hxx``). Quadrangle_2D asked for a REDUCED
  transition on a face whose four sides all differ falls back to the STANDARD one and
  says so. The oracle is the STANDARD mesh of the same face, computed separately.
"""

from __future__ import annotations

import numpy as np

import pysmesh as ps
from pysmesh import (
    Mesher,
    NumberOfSegments,
    Quadrangle2D,
    QuadrangleParams,
    QuadType,
    Regular1D,
    Session,
    SubShape,
    SubShapeKind,
)

RECT_DX: float = 4.0
RECT_DY: float = 3.0
# Segment counts on the four sides of the rectangle: no two opposite sides match.
SIDE_COUNTS: tuple[int, ...] = (4, 6, 8, 10)


def _rectangle() -> ps.Shape:
    """A 4 x 3 planar rectangle with its 4 edges."""
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), RECT_DX, RECT_DY)
    return ps.load_brep(s.brep())


def _quad_mesh(quad_type: QuadType) -> tuple[ps.ComputeReport, ps.MeshData]:
    """Quadrangle_2D on the rectangle with the unequal side counts, one quad type."""
    with Mesher(_rectangle()) as m:
        m.assign(Regular1D())
        for ordinal, count in enumerate(SIDE_COUNTS, start=1):
            m.assign(
                NumberOfSegments(count=count), on=SubShape(SubShapeKind.EDGE, ordinal)
            )
        m.assign(Quadrangle2D())
        m.assign(QuadrangleParams(quad_type=quad_type))
        report = m.compute()
        return report, m.mesh()


def test_a_reduced_transition_that_falls_back_is_a_warning_not_a_failure() -> None:
    """REDUCED falls back to STANDARD: the same mesh, and one warning naming the face."""
    _, standard = _quad_mesh(QuadType.STANDARD)

    report, reduced = _quad_mesh(QuadType.REDUCED)

    np.testing.assert_array_equal(reduced.node_coords, standard.node_coords)
    np.testing.assert_array_equal(reduced.element_nodes, standard.element_nodes)
    assert len(report.warnings) == 1
    warning = report.warnings[0]
    assert (warning.kind, warning.ordinal) == (SubShapeKind.FACE, 1)
    assert warning.algorithm == "Quadrangle_2D"
    assert "'Standard' transion has been used" in warning.text


def test_a_clean_compute_reports_no_warning() -> None:
    """The STANDARD transition on the same face warns of nothing."""
    report, _ = _quad_mesh(QuadType.STANDARD)

    assert report.warnings == ()
