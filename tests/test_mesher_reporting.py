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
* **A6, a missing assignment is named.** A sub-shape with no algorithm, or with an
  algorithm that lacks its hypothesis, holds an algorithm state, not a compute error
  (``SMESH_subMesh::GetAlgoState``). The failure must name each such sub-shape with its
  state. The oracle is the assignment the test made.
"""

from __future__ import annotations

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import (
    Hexa3D,
    MaxElementArea,
    Mefisto2D,
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


# ---- A6: a missing algorithm or hypothesis is named --------------------------------- #


def _box() -> ps.Shape:
    """The unit box: 6 faces, 12 edges."""
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    return ps.load_brep(s.brep())


def _missing_lines(error: ps.PysmeshError) -> list[str]:
    """The lines of ``details`` that report an algorithm state."""
    return [line for line in error.details.splitlines() if "algorithm state" in line]


def test_an_edge_without_a_1d_hypothesis_is_named_missing_its_hypothesis() -> None:
    """Regular1D everywhere, NumberOfSegments on 11 of the 12 edges: edge 12 lacks one."""
    box = _box()
    with Mesher(box) as m:
        m.assign(Regular1D())
        for ordinal in range(1, 12):
            m.assign(NumberOfSegments(count=4), on=SubShape(SubShapeKind.EDGE, ordinal))
        m.assign(Quadrangle2D())
        m.assign(Hexa3D())

        with pytest.raises(ps.PysmeshError) as info:
            m.compute()

    assert _missing_lines(info.value) == [
        "EDGE 12: Regular_1D is missing a hypothesis it needs (algorithm state MISSING_HYP)"
    ]
    assert "meshing failed on 1 sub-shape(s)" in str(info.value)


def test_regular_1d_with_no_hypothesis_names_every_edge() -> None:
    """No 1-D hypothesis anywhere: all 12 edges are missing one."""
    with Mesher(_box()) as m:
        m.assign(Regular1D())
        m.assign(Quadrangle2D())
        m.assign(Hexa3D())

        with pytest.raises(ps.PysmeshError) as info:
            m.compute()

    lines = _missing_lines(info.value)
    assert [line.split(":")[0] for line in lines] == [f"EDGE {k}" for k in range(1, 13)]
    assert all("MISSING_HYP" in line for line in lines)


def test_a_face_without_a_2d_algorithm_is_named_with_no_algorithm() -> None:
    """Quadrangle2D on 5 of the 6 faces: face 6 has no algorithm, and is on face_ids."""
    with Mesher(_box()) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=4))
        m.assign(Hexa3D())
        for ordinal in range(1, 6):
            m.assign(Quadrangle2D(), on=SubShape(SubShapeKind.FACE, ordinal))

        with pytest.raises(ps.PysmeshError) as info:
            m.compute()

    assert _missing_lines(info.value) == [
        "FACE 6: no algorithm is assigned to it (algorithm state NO_ALGO)"
    ]
    assert list(info.value.face_ids) == [6]


def test_the_degenerate_pole_edges_of_a_sphere_are_named_missing_a_hypothesis() -> None:
    """Hypotheses on the meridian only: the two pole edges (length 0) get none."""
    s = Session()
    s.add_sphere(1.0)
    sphere = ps.load_brep(s.brep())
    poles = [e.id for e in sphere.edges() if e.length < 1e-9]
    with Mesher(sphere) as m:
        m.assign(Regular1D())
        for edge in sphere.edges():
            if edge.length >= 1e-9:
                m.assign(
                    NumberOfSegments(count=8), on=SubShape(SubShapeKind.EDGE, edge.id)
                )
        m.assign(Mefisto2D())
        m.assign(MaxElementArea(max_area=0.05))

        with pytest.raises(ps.PysmeshError) as info:
            m.compute()

    named = [line.split(":")[0] for line in _missing_lines(info.value)]
    assert poles == [1, 3]
    assert named == ["EDGE 1", "EDGE 3"]
