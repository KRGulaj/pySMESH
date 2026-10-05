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
* **M3, an algorithm without its hypothesis fails the compute.** A sub-mesh whose
  algorithm misses its hypothesis is never computed, and ``SMESH_Gen::Compute`` returns
  false only for a compute that failed (``sequentialComputeSubMeshes``). So Cartesian3D,
  Projection3D or RadialPrism3D alone gave a success with no volume, and Projection2D
  on one face a success with that face empty. The compute must fail and name the
  sub-shape, its algorithm and its state. A VERTEX keeps its node, but
  SegmentAroundVertex0D without its length does nothing, so it is named too. A sub-shape
  with no algorithm of its own, meshed by an enclosing algorithm, and a solid with no
  3-D algorithm under a surface
  mesh, are no error. The oracle is the assignment the test made, and the closed-form
  element counts of the meshes that must stay.
* **M1, a raising hook leaves no partial mesh.** The hook's exception must reach the
  caller with its own type, and the mesh must be empty afterwards, as for a cancel.
* **N1, a refused value is a PysmeshError.** An upstream setter refuses a bad value with
  ``SALOME_Exception``, which reached Python as a raw ``RuntimeError``. The oracle is
  the upstream throw site, cited beside each case.
* **R1, an edge too short for Adaptive1D is reported.** An edge shorter than the min
  size keeps one segment shorter than the min size, against the hypothesis's bounds.
  The compute must report it as a warning on that edge. The oracle is the edge length.
"""

from __future__ import annotations

import numpy as np
import pytest

import pysmesh as ps
from pysmesh import (
    Adaptive1D,
    Arithmetic1D,
    AutomaticLength,
    Cartesian3D,
    CartesianParameters3D,
    Deflection1D,
    Distribution,
    Geometric1D,
    Hexa3D,
    LayerDistribution,
    LocalLength,
    MaxElementArea,
    MaxElementVolume,
    MaxLength,
    Mefisto2D,
    Mesher,
    NumberOfLayers,
    NumberOfLayers2D,
    NumberOfSegments,
    Projection2D,
    Projection3D,
    Quadrangle2D,
    QuadrangleParams,
    QuadType,
    RadialPrism3D,
    Regular1D,
    SegmentAroundVertex0D,
    SegmentLengthAroundVertex,
    Session,
    StartEndLength,
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
    """REDUCED falls back to STANDARD: the same mesh, one warning naming the face."""
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


# ---- A6: a missing algorithm or hypothesis is named -------------------------------- #


def _box() -> ps.Shape:
    """The unit box: 6 faces, 12 edges."""
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    return ps.load_brep(s.brep())


def _missing_lines(error: ps.PysmeshError) -> list[str]:
    """The lines of ``details`` that report an algorithm state."""
    return [line for line in error.details.splitlines() if "algorithm state" in line]


def test_an_edge_without_a_1d_hypothesis_is_named_missing_its_hypothesis() -> None:
    """Regular1D everywhere, NumberOfSegments on 11 of 12 edges: edge 12 lacks one."""
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
        (
            "EDGE 12: Regular_1D is missing a hypothesis it needs "
            "(algorithm state MISSING_HYP)"
        )
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


# ---- M3: an algorithm without its hypothesis fails the compute --------------------- #

# Segments on every edge of the box in the M3 cases: 6 faces x 4 x 4 quadrangles, 4**3
# hexahedra.
M3_SEGMENTS: int = 4


def _hollow_box() -> ps.Shape:
    """A 3 x 3 x 3 box less the unit box at its centre: one solid between two shells."""
    s = Session()
    s.add_box(3.0, 3.0, 3.0)
    outer = s.entities(ps.EntityKind.SOLID).tolist()
    s.add_box(1.0, 1.0, 1.0, origin=(1.0, 1.0, 1.0))
    inner = [i for i in s.entities(ps.EntityKind.SOLID).tolist() if i not in outer]
    s.cut(outer, inner)
    return ps.load_brep(s.brep())


def _surface_algorithms(m: Mesher) -> None:
    """Regular1D with 4 segments and Quadrangle2D on the whole shape."""
    m.assign(Regular1D())
    m.assign(NumberOfSegments(count=M3_SEGMENTS))
    m.assign(Quadrangle2D())


@pytest.mark.parametrize(
    ("algorithm", "native", "surface", "hollow"),
    [
        (Cartesian3D(), "Cartesian_3D", False, False),
        (Projection3D(), "Projection_3D", True, False),
        (RadialPrism3D(), "RadialPrism_3D", True, True),
    ],
    ids=["Cartesian3D", "Projection3D", "RadialPrism3D"],
)
def test_a_solid_whose_algorithm_misses_its_hypothesis_is_named(
    algorithm: ps.Algorithm, native: str, surface: bool, hollow: bool
) -> None:
    """The 3-D algorithm without its hypothesis: SOLID 1 named, not 0 volumes (M3)."""
    with Mesher(_hollow_box() if hollow else _box()) as m:
        if surface:
            _surface_algorithms(m)
        m.assign(algorithm)

        with pytest.raises(ps.PysmeshError) as info:
            m.compute()

    assert _missing_lines(info.value) == [
        (
            f"SOLID 1: {native} is missing a hypothesis it needs "
            "(algorithm state MISSING_HYP)"
        )
    ]


def test_a_face_whose_projection_has_no_source_is_named() -> None:
    """Projection2D on the bottom face without ProjectionSource2D: that face (M3)."""
    box = _box()
    bottom = min(box.faces(), key=lambda f: float(f.bbox[5])).id
    with Mesher(box) as m:
        _surface_algorithms(m)
        m.assign(Projection2D(), on=SubShape(SubShapeKind.FACE, bottom))

        with pytest.raises(ps.PysmeshError) as info:
            m.compute()

    assert _missing_lines(info.value) == [
        (
            f"FACE {bottom}: Projection_2D is missing a hypothesis it needs "
            "(algorithm state MISSING_HYP)"
        )
    ]
    assert list(info.value.face_ids) == [bottom]


def test_a_vertex_algorithm_without_its_length_is_named() -> None:
    """SegmentAroundVertex0D without SegmentLengthAroundVertex does nothing (M3)."""
    with Mesher(_box()) as m:
        _surface_algorithms(m)
        m.assign(SegmentAroundVertex0D(), on=SubShape(SubShapeKind.VERTEX, 1))

        with pytest.raises(ps.PysmeshError) as info:
            m.compute()

    assert _missing_lines(info.value) == [
        (
            "VERTEX 1: SegmentAroundVertex_0D is missing a hypothesis it needs "
            "(algorithm state MISSING_HYP)"
        )
    ]


def test_a_surface_mesh_without_a_3d_algorithm_is_no_error() -> None:
    """No 3-D algorithm: 6 x 4 x 4 quadrangles, no volume, and no error (M3)."""
    with Mesher(_box()) as m:
        _surface_algorithms(m)

        report = m.compute()

    assert (report.faces, report.volumes) == (6 * M3_SEGMENTS**2, 0)


def test_sub_shapes_meshed_by_an_all_dimensional_algorithm_are_no_error() -> None:
    """Cartesian3D meshes the faces and edges, which have no algorithm: 4**3 cells."""
    spacing = str(1.0 / M3_SEGMENTS)
    with Mesher(_box()) as m:
        m.assign(Cartesian3D())
        m.assign(
            CartesianParameters3D(
                spacing_x=spacing, spacing_y=spacing, spacing_z=spacing
            )
        )

        report = m.compute()

    assert report.volumes == M3_SEGMENTS**3


# ---- M1: a hook that raises leaves no mesh ----------------------------------------- #

# A structured box of 50 segments a side takes long enough (about 0.2 s) for the
# progress hook to run several times. It is not a Cartesian mesh, whose own cancel path
# clears it.
HOOKED_SEGMENTS: int = 50


class _HookStop(RuntimeError):
    """Raised by the progress hook under test."""


def test_a_progress_hook_that_raises_half_way_leaves_no_mesh() -> None:
    """The hook's exception reaches the caller, and the mesh is cleared (report M1)."""
    calls: list[float] = []

    def hook(fraction: float) -> None:
        calls.append(fraction)
        if len(calls) == 2:
            raise _HookStop("half-way")

    s = Session()
    s.add_box(3.0, 7.0, 11.0)
    with Mesher(ps.load_brep(s.brep())) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=HOOKED_SEGMENTS))
        m.assign(Quadrangle2D())
        m.assign(Hexa3D())

        with pytest.raises(_HookStop):
            m.compute(progress=hook)

        assert len(calls) == 2
        assert m.mesh().element_count == 0


# ---- N1: an upstream setter that refuses a value raises PysmeshError --------------- #

# One invalid value per catalogue hypothesis whose upstream setter refuses it, each with
# the StdMeshers source line that throws (SMESH V9_16_0), and LocalLength's negative
# precision, which upstream accepts (its SetPrecision tests the old value) and the
# binding refuses.
REFUSED_VALUES: dict[str, ps.Hypothesis] = {
    # StdMeshers_NumberOfSegments.cxx:116 "number of segments must be positive"
    "NumberOfSegments-count-0": NumberOfSegments(count=0),
    # :184 "scale factor must be positive"
    "NumberOfSegments-scale-0": NumberOfSegments(
        count=5, distribution=Distribution.SCALE, scale_factor=0.0
    ),
    # :262 "odd size of vector of table function"
    "NumberOfSegments-table-odd": NumberOfSegments(
        count=5, distribution=Distribution.TABLE, table=(0.0, 1.0, 1.0)
    ),
    # :459 "invalid expression syntax"
    "NumberOfSegments-expression": NumberOfSegments(
        count=5, distribution=Distribution.EXPRESSION, expression="t*"
    ),
    # StdMeshers_Arithmetic1D.cxx:80 "length must be positive"
    "Arithmetic1D-start-negative": Arithmetic1D(start_length=-1.0, end_length=1.0),
    # StdMeshers_StartEndLength.cxx:79
    "StartEndLength-start-0": StartEndLength(start_length=0.0, end_length=1.0),
    # StdMeshers_Geometric1D.cxx:64 and :81 "Zero factor is not allowed"
    "Geometric1D-start-negative": Geometric1D(start_length=-1.0, common_ratio=1.1),
    "Geometric1D-ratio-0": Geometric1D(start_length=1.0, common_ratio=0.0),
    # StdMeshers_Adaptive1D.cxx:958 and :945
    "Adaptive1D-min-0": Adaptive1D(min_size=0.0, max_size=1.0, deflection=0.1),
    "Adaptive1D-deflection-0": Adaptive1D(min_size=0.1, max_size=1.0, deflection=0.0),
    # StdMeshers_AutomaticLength.cxx:89 "theFineness is out of range [0.0-1.0]"
    "AutomaticLength-fineness-2": AutomaticLength(fineness=2.0),
    # StdMeshers_Deflection1D.cxx:81 "Value must be positive"
    "Deflection1D-0": Deflection1D(deflection=0.0),
    # StdMeshers_LocalLength.cxx:84 "length must be positive"
    "LocalLength-length-0": LocalLength(length=0.0),
    "LocalLength-length-negative": LocalLength(length=-1.0),
    # Accepted upstream (StdMeshers_LocalLength.cxx:110 tests _precision, the old value)
    "LocalLength-precision-negative": LocalLength(length=1.0, precision=-1.0),
    # StdMeshers_MaxLength.cxx:79
    "MaxLength-0": MaxLength(length=0.0),
    # StdMeshers_SegmentLengthAroundVertex.cxx:81
    "SegmentLengthAroundVertex-0": SegmentLengthAroundVertex(length=0.0),
    # StdMeshers_MaxElementArea.cxx:79 and StdMeshers_MaxElementVolume.cxx:79
    "MaxElementArea-0": MaxElementArea(max_area=0.0),
    "MaxElementVolume-negative": MaxElementVolume(max_volume=-1.0),
    # StdMeshers_NumberOfLayers.cxx:78, shared by NumberOfLayers2D
    "NumberOfLayers-0": NumberOfLayers(count=0),
    "NumberOfLayers2D-0": NumberOfLayers2D(count=0),
    # StdMeshers_CartesianParameters3D.cxx:256 "threshold must be > 1.0"
    "CartesianParameters3D-threshold-1": CartesianParameters3D(
        spacing_x="1.0", spacing_y="1.0", spacing_z="1.0", size_threshold=1.0
    ),
    # The nested 1-D hypothesis of a layer distribution refuses its count (:116)
    "LayerDistribution-count-0": LayerDistribution(
        distribution=NumberOfSegments(count=0)
    ),
}


@pytest.mark.parametrize(
    "hypothesis", REFUSED_VALUES.values(), ids=REFUSED_VALUES.keys()
)
def test_an_invalid_hypothesis_value_raises_pysmesh_error(
    hypothesis: ps.Hypothesis,
) -> None:
    """Every refused value reaches Python as PysmeshError naming the hypothesis (N1)."""
    with Mesher(_box()) as m:
        with pytest.raises(ps.PysmeshError) as info:
            m.assign(hypothesis)

        assert hypothesis.native_name in str(info.value)
        assert m.assignments() == ()


# ---- R1: an edge too short for Adaptive1D's min size is reported ------------------- #

ADAPTIVE_MIN: float = 0.05
# A 3 x 7 rectangle with one corner cut off by an edge 0.02 sqrt(2) = 0.028284 long.
CUT: float = 0.02
SHORT_EDGE: float = CUT * 2.0**0.5


def _cut_rectangle() -> ps.Shape:
    """A 3 x 7 rectangle with one corner cut off: one edge shorter than min size."""
    s = Session()
    s.add_polyline(
        [
            (0.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.0, 7.0, 0.0),
            (CUT, 7.0, 0.0),
            (0.0, 7.0 - CUT, 0.0),
        ],
        closed=True,
    )
    s.make_face(list(s.entities(ps.EntityKind.EDGE)))
    return ps.load_brep(s.brep())


def test_an_edge_shorter_than_adaptive_min_size_is_reported_as_a_warning() -> None:
    """The short edge keeps one segment; the report names it with its length (R1)."""
    shape = _cut_rectangle()
    short = [e.id for e in shape.edges() if e.length < ADAPTIVE_MIN]
    with Mesher(shape) as m:
        m.assign(Regular1D())
        m.assign(ps.Adaptive1D(min_size=ADAPTIVE_MIN, max_size=1.0, deflection=0.01))

        report = m.compute()

    assert len(short) == 1
    assert [(w.kind, w.ordinal) for w in report.warnings] == [
        (SubShapeKind.EDGE, short[0])
    ]
    assert f"{SHORT_EDGE:.6g}" in report.warnings[0].text
    assert "min size" in report.warnings[0].text
