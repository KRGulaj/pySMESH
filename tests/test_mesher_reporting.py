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
* **#39, a layer shrink failure names its face.** SMESH writes its own shape index into
  "Can't shrink 2D mesh on face <index>". The line must give the face ordinal, and
  ``.face_ids`` must hold it. The oracle is the face found by its area.
* **#25, a prism under a global 2-D algorithm names its cause.** A global Netgen1D2D
  puts triangles on the side faces of a Prism3D solid. In one solid order SMESH says
  "More than 2 faces with not quadrangle elements"; in the other it says nothing. Both
  must name the solid, its triangle faces and Quadrangle2D. The oracle is the face set
  of the solid, found by geometry.
* **#24, a failed NETGEN 2-D face gives its sizes.** NETGEN says only "Problem in
  Surface mesh generation". The line must give the longest segment on the face's edges,
  the smallest extent of its box and the size asked, and name the coarse edges when the
  segment is longer than the face is wide. The oracles are the mesh nodes, the Session
  box and the assigned size; a face that fails for another reason gets no such cause.
* **#30, a NETGEN 3-D stop gives the open front.** "Stop meshing since too many attempts"
  names no place. The line must give the open front of the kept mesh: its face count and
  the box of each patch, with every centroid in ``.bad_element_centroids``. The oracles
  are the injected triangle count and the unit box.
"""

from __future__ import annotations

import re

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


# ---- A failure that names bad elements says where they are -------------------------- #

# The surface of a unit box from NETGEN 2D at size 0.25, injected with one node of face 1
# moved 0.15 into the box and 0.05 across the neighbouring face: its triangles then cut
# that face's triangles, and NETGEN 3D fails with "Intersecting triangles".
INJECTED_SIZE: float = 0.25


def _box_surface_with_a_crossing_node() -> tuple[Mesher, tuple[tuple[int, float], ...]]:
    """A mesher holding the crossed surface, NETGEN 3D assigned; the two face planes.

    The planes are (axis, value) of face 1 and of the face its moved node crosses.
    """
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    shape = ps.load_brep(s.brep())
    with Mesher(shape) as surface:
        surface.assign(ps.Netgen1D2D())
        surface.assign(ps.NetgenParameters2D(max_size=INJECTED_SIZE, threads=4))
        surface.compute()
        md = surface.mesh()
    xyz = md.node_coords.copy()
    on_face_1 = np.flatnonzero(
        (md.node_kind == int(SubShapeKind.FACE)) & (md.node_ordinal == 1)
    )
    c = xyz[on_face_1]
    normal_axis = int(np.argmin(np.ptp(c, axis=0)))
    side = float(np.round(c[0, normal_axis]))
    other = (normal_axis + 1) % 3
    k = on_face_1[int(np.argmin(c[:, other]))]
    xyz[k, normal_axis] = 0.15 if side == 0.0 else 0.85
    xyz[k, other] = -0.05

    m = Mesher(shape)
    m.assign(ps.UseExisting1D())
    m.assign(ps.UseExisting2D())
    m.assign(ps.Netgen3D())
    m.assign(ps.NetgenParameters(max_size=INJECTED_SIZE, threads=4))
    ids = m.add_nodes(xyz)
    for kind, code, add in (
        (SubShapeKind.EDGE, ps.ElementType.EDGE, m.add_segments),
        (SubShapeKind.FACE, ps.ElementType.TRIANGLE, m.add_triangles),
    ):
        rows = np.flatnonzero(md.element_type == int(code))
        for ordinal in np.unique(md.element_ordinal[rows]):
            sel = rows[md.element_ordinal[rows] == ordinal]
            nodes = np.stack([md.nodes_of(int(i)) for i in sel])
            add(ids[nodes], on=SubShape(kind, int(ordinal)))
    return m, ((normal_axis, side), (other, 0.0))


def test_intersecting_triangles_are_located_by_their_centroids() -> None:
    """Each centroid is in the box, within one element size of a face that crosses."""
    m, planes = _box_surface_with_a_crossing_node()

    with m, pytest.raises(ps.PysmeshError) as info:
        m.compute()

    centroids = info.value.bad_element_centroids
    assert "Intersecting triangles" in info.value.details
    assert "bad element(s), centroid(s) of the first" in info.value.details
    assert isinstance(centroids, np.ndarray)
    assert centroids.dtype == np.float64
    assert centroids.ndim == 2 and centroids.shape[0] > 0 and centroids.shape[1] == 3
    assert bool(np.all((centroids >= 0.0) & (centroids <= 1.0)))
    distance = np.min(
        np.stack([np.abs(centroids[:, axis] - value) for axis, value in planes]), axis=0
    )
    assert bool(np.all(distance <= INJECTED_SIZE))


def test_a_failure_with_no_bad_element_has_an_empty_centroid_array() -> None:
    """A missing hypothesis names no element: the array is (0, 3) float64."""
    box = _box()
    with Mesher(box) as m:
        m.assign(Regular1D())
        m.assign(Quadrangle2D())
        m.assign(Hexa3D())

        with pytest.raises(ps.PysmeshError) as info:
            m.compute()

    centroids = info.value.bad_element_centroids
    assert isinstance(centroids, np.ndarray)
    assert centroids.dtype == np.float64
    assert centroids.shape == (0, 3)


# The strip of the top face that carries no layer: 0.05 wide across the 2 x 2 box.
STRIP_LOW: float = 0.975
STRIP_HIGH: float = 1.025
STRIP_AREA: float = 2.0 * (STRIP_HIGH - STRIP_LOW)


def _box_with_a_narrow_top_strip() -> ps.Shape:
    """A 2 x 2 x 1 box whose top face is split into three strips at y 0.975 and 1.025."""
    s = Session()
    s.add_box(2.0, 2.0, 1.0)
    solid = int(s.entities(ps.EntityKind.SOLID)[0])
    for y in (STRIP_LOW, STRIP_HIGH):
        delta = s.add_line((0.0, y, 1.0), (2.0, y, 1.0))
        edge = [
            int(i) for i in delta.created if s.entity_kind(int(i)) == ps.EntityKind.EDGE
        ]
        s.imprint([solid], edge)
        s.remove(edge)
    return ps.load_brep(s.brep())


def test_a_viscous_layer_shrink_failure_names_the_face_by_its_ordinal() -> None:
    """The face the layers cannot shrink into is named as FACE <ordinal>, in face_ids.

    A 0.3 stack grows on every face but the 0.05-wide strip of the top face. The strip
    must shrink by 0.3 from both sides, which it cannot. SMESH writes its own shape index
    into the text (StdMeshers_ViscousLayers.cxx, "shrinking 2D mesh on face <index>").
    The oracle is the strip itself, found by its area: 2 x 0.05.
    """
    shape = _box_with_a_narrow_top_strip()
    strip = [f.id for f in shape.faces() if abs(f.area - STRIP_AREA) < 1e-9]
    walls = tuple(f.id for f in shape.faces() if f.id not in strip)
    with Mesher(shape) as m:
        m.assign(ps.Netgen1D2D3D())
        m.assign(ps.NetgenParameters(max_size=0.2, threads=4))
        m.assign(
            ps.ViscousLayers(
                total_thickness=0.3,
                layer_count=3,
                stretch_factor=1.2,
                boundary=walls,
                group_name="BL",
            )
        )

        with pytest.raises(ps.PysmeshError) as info:
            m.compute()

    named = re.findall(r"2D mesh on FACE (\d+)", info.value.details)
    ordinals = {int(n) for n in named}
    assert len(strip) == 1
    assert ordinals == set(strip)
    assert strip[0] in list(info.value.face_ids)


def _two_boxes_sharing_a_face() -> ps.Shape:
    """Two unit boxes side by side, fragmented: they share the face x = 1."""
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    s.add_box(1.0, 1.0, 1.0, origin=(1.0, 0.0, 0.0))
    s.fragment([int(i) for i in s.entities(ps.EntityKind.SOLID)])
    return ps.load_brep(s.brep())


def _faces_of_solid(shape: ps.Shape, solid: int) -> set[int]:
    """The faces whose centroid lies in the bounding box of ``solid``."""
    box = np.asarray(shape.solids()[solid - 1].bbox)
    return {
        f.id
        for f in shape.faces()
        if bool(np.all(np.asarray(f.centroid) >= box[:3] - 1e-9))
        and bool(np.all(np.asarray(f.centroid) <= box[3:] + 1e-9))
    }


def _prism_in_a_netgen_region(shape: ps.Shape, prism: int, quad_sides: bool) -> Mesher:
    """Global Netgen1D2D + Netgen3D, Prism3D on one solid; optionally quadrangle sides."""
    m = Mesher(shape)
    m.assign(ps.Netgen1D2D())
    m.assign(ps.NetgenParameters2D(max_size=0.25, threads=4))
    m.assign(ps.Netgen3D())
    m.assign(ps.NetgenParameters(max_size=0.25, threads=4))
    on = SubShape(SubShapeKind.SOLID, prism)
    m.assign(ps.Prism3D(), on=on)
    if quad_sides:
        for face in _faces_of_solid(shape, prism):
            box = shape.faces()[face - 1].bbox
            if abs(box[5] - box[2]) > 1e-9:  # a side face: not z = const
                m.assign(Quadrangle2D(), on=SubShape(SubShapeKind.FACE, face))
        m.assign(Regular1D(), on=on)
        m.assign(NumberOfSegments(count=4), on=on)
    return m


@pytest.mark.parametrize("prism", [1, 2])
def test_a_prism_whose_side_faces_take_triangles_is_named_with_the_cause(
    prism: int,
) -> None:
    """Both solid orders name the Prism_3D solid, its triangle faces and Quadrangle2D.

    The global Netgen1D2D reaches every face of the prism solid. As the first solid,
    Prism_3D refuses it ("More than 2 faces with not quadrangle elements: 6"). As the
    second, SMESH left no error text at all ("failed on 0 sub-shape(s)"). The oracle is
    the face set of the prism solid, found by geometry: the global algorithm reaches all
    six of its faces.
    """
    shape = _two_boxes_sharing_a_face()
    expected = _faces_of_solid(shape, prism)
    m = _prism_in_a_netgen_region(shape, prism, quad_sides=False)

    with m, pytest.raises(ps.PysmeshError) as info:
        m.compute()

    lines = [
        line
        for line in info.value.details.splitlines()
        if line.startswith(f"SOLID {prism}:")
    ]
    assert len(lines) == 1
    assert "(algorithm Prism_3D)" in lines[0]
    assert "Assign Quadrangle2D on the side faces" in lines[0]
    named = re.search(r"\(FACE ([0-9, ]+)\) take triangles", lines[0])
    assert named is not None
    assert {int(f) for f in named.group(1).split(", ")} == expected
    assert len(expected) == 6


@pytest.mark.parametrize("prism", [1, 2])
def test_a_prism_with_quadrangle_side_faces_meshes_in_either_order(prism: int) -> None:
    """Quadrangle2D on the side faces, Regular1D + 4 segments on the prism solid: no error.

    Prism3D sweeps a source face through the solid, so the prism solid holds swept cells
    only: every element there is a pentahedron or a hexahedron, and none a tetrahedron.
    """
    shape = _two_boxes_sharing_a_face()
    with _prism_in_a_netgen_region(shape, prism, quad_sides=True) as m:
        report = m.compute()
        md = m.mesh()

    in_prism = (md.element_kind == int(SubShapeKind.SOLID)) & (
        md.element_ordinal == prism
    )
    swept = (int(ps.ElementType.PENTAHEDRON), int(ps.ElementType.HEXAHEDRON))
    assert report.volumes > 0
    assert int(in_prism.sum()) > 0
    assert bool(np.all(np.isin(md.element_type[in_prism], swept)))


NUMBER: str = r"([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)"
SIZES_PATTERN: str = (
    rf"FACE (\d+): .*longest segment on its edges {NUMBER}, smallest extent of its "
    rf"box {NUMBER}, size its hypothesis asks for {NUMBER}"
)
COARSE_EDGES_SENTENCE: str = "The edges of this face were meshed coarser than the face"


def _smallest_extent(box: np.ndarray) -> float:
    """The smallest extent of an (xmin..zmax) box above 1e-7 (a plane face has one 0)."""
    extents = np.asarray(box[3:], dtype=np.float64) - np.asarray(
        box[:3], dtype=np.float64
    )
    return float(extents[extents > 1e-7].min())


def _failure_and_mesh(m: Mesher) -> tuple[ps.PysmeshError, ps.MeshData]:
    """Run a compute that must fail; return its error and the partial mesh it keeps."""
    with pytest.raises(ps.PysmeshError) as info:
        m.compute()
    return info.value, m.mesh()


def _longest_segment_on_face_edges(
    shape: ps.Shape, md: ps.MeshData, face: int
) -> float:
    """The longest mesh segment on an edge of ``face`` (edges inside the face's box)."""
    box = np.asarray(shape.faces()[face - 1].bbox)
    edges = [
        e.id
        for e in shape.edges()
        if bool(np.all(np.asarray(e.bbox)[:3] >= box[:3] - 1e-9))
        and bool(np.all(np.asarray(e.bbox)[3:] <= box[3:] + 1e-9))
    ]
    on_edges = (
        (md.element_type == int(ps.ElementType.EDGE))
        & (md.element_kind == int(SubShapeKind.EDGE))
        & np.isin(md.element_ordinal, edges)
    )
    ends = md.node_coords[
        np.stack([md.nodes_of(int(i)) for i in np.flatnonzero(on_edges)])
    ]
    return float(np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1).max())


def test_a_narrow_face_after_coarse_neighbours_gives_its_sizes_and_the_cause() -> None:
    """A 74 mm face whose long edges a 0.5 neighbour meshed first: three numbers, a cause.

    Each face of a 2 x 0.074 x 1 slab gets its own Netgen1D2D: max_size 0.5 on the big
    faces, 0.01 on the narrow ones. A narrow face meshed after its big neighbours finds
    its long edges in segments longer than it is wide, and NETGEN says only "Problem in
    Surface mesh generation". The oracles: the longest segment from the mesh nodes on the
    face's edges, the smallest extent of the face's box from Session.bounding_boxes, and
    the 0.01 the test assigned.
    """
    s = Session()
    s.add_box(2.0, 0.074, 1.0)
    handoff = s.export_handoff()
    boxes = s.bounding_boxes(ps.EntityKind.FACE)
    shape = ps.load_brep(handoff.brep)
    narrow = {
        f.id for f in shape.faces() if _smallest_extent(np.asarray(f.bbox)) < 0.08
    }
    m = Mesher(shape)
    for face in shape.faces():
        on = SubShape(SubShapeKind.FACE, face.id)
        m.assign(ps.Netgen1D2D(), on=on)
        size = 0.01 if face.id in narrow else 0.5
        m.assign(ps.NetgenParameters2D(max_size=size, threads=4), on=on)

    with m:
        error, md = _failure_and_mesh(m)

    found = [
        (line, re.search(SIZES_PATTERN, line))
        for line in error.details.splitlines()
        if "Problem in Surface mesh generation" in line
    ]
    assert len(found) >= 1
    for line, match in found:
        assert match is not None
        face, longest, extent, asked = (
            int(match.group(1)),
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        )
        row = int(np.flatnonzero(boxes.ids == handoff.face_id[face - 1])[0])
        assert face in narrow
        assert longest == pytest.approx(
            _longest_segment_on_face_edges(shape, md, face), rel=1e-5
        )
        assert extent == pytest.approx(_smallest_extent(boxes.bbox[row]), rel=1e-5)
        assert asked == pytest.approx(0.01, rel=1e-12)
        assert longest > extent
        assert COARSE_EDGES_SENTENCE in line


def test_a_netgen_face_that_fails_for_another_reason_gets_no_coarse_edge_sentence() -> (
    None
):
    """A disk whose circle has one segment: NETGEN 2D refuses it, and no cause is guessed.

    Netgen2D reads the circle's one Regular1D segment ("Too few segments: 1"). That one
    segment starts and ends at the circle's vertex, so its length is 0: shorter than the
    disk's diameter 2, the closed-form smallest extent of its box.
    """
    s = Session()
    circle = s.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)
    s.make_face(
        [int(i) for i in circle.created if s.entity_kind(int(i)) == ps.EntityKind.EDGE]
    )
    m = Mesher(ps.load_brep(s.brep()))
    m.assign(Regular1D())
    m.assign(NumberOfSegments(count=1))
    m.assign(ps.Netgen2D())
    m.assign(ps.NetgenParameters2D(max_size=0.1, threads=4))

    with m, pytest.raises(ps.PysmeshError) as info:
        m.compute()

    match = re.search(SIZES_PATTERN, info.value.details)
    assert match is not None
    assert float(match.group(2)) == pytest.approx(0.0, abs=1e-12)
    assert float(match.group(3)) == pytest.approx(2.0, rel=1e-6)
    assert float(match.group(4)) == pytest.approx(0.1, rel=1e-12)
    assert COARSE_EDGES_SENTENCE not in info.value.details


FRONT_PATTERN: str = (
    rf"open front of the kept mesh: (\d+) face\(s\) in (\d+) patch\(es\).*"
    rf"largest: (\d+) face\(s\) in \({NUMBER}, {NUMBER}, {NUMBER}\) to "
    rf"\({NUMBER}, {NUMBER}, {NUMBER}\)"
)


def _box_surface_reversed_into_netgen_3d() -> tuple[Mesher, int]:
    """The unit box surface of NETGEN 2D, every triangle reversed, under Netgen3D.

    NETGEN 3D's advancing front (Delaunay off) cannot fill a surface whose triangles all
    point inward, and stops with "too many attempts". Returns the mesher and the number
    of injected triangles.
    """
    s = Session()
    s.add_box(1.0, 1.0, 1.0)
    shape = ps.load_brep(s.brep())
    with Mesher(shape) as surface:
        surface.assign(ps.Netgen1D2D())
        surface.assign(ps.NetgenParameters2D(max_size=0.1, threads=4))
        surface.compute()
        md = surface.mesh()
    m = Mesher(shape)
    m.assign(ps.UseExisting1D())
    m.assign(ps.UseExisting2D())
    m.assign(ps.Netgen3D())
    m.assign(ps.NetgenParameters(max_size=0.1, threads=4, use_delaunay=False))
    ids = m.add_nodes(md.node_coords)
    triangles = 0
    for etype, add, kind in (
        (ps.ElementType.EDGE, m.add_segments, SubShapeKind.EDGE),
        (ps.ElementType.TRIANGLE, m.add_triangles, SubShapeKind.FACE),
    ):
        rows = np.flatnonzero(md.element_type == int(etype))
        for ordinal in np.unique(md.element_ordinal[rows]):
            sel = rows[md.element_ordinal[rows] == ordinal]
            nodes = np.stack([md.nodes_of(int(i)) for i in sel])
            if etype == ps.ElementType.TRIANGLE:
                nodes = nodes[:, ::-1]
                triangles += len(sel)
            add(ids[nodes], on=SubShape(kind, int(ordinal)))
    return m, triangles


def test_too_many_attempts_names_the_open_front_and_its_patch_boxes() -> None:
    """The line gives the open face count and patch boxes; centroids lie in the solid.

    NETGEN kept no tetrahedron here, so its front at the stop is the whole injected
    surface: every one of the injected triangles, in one patch whose box is the unit
    box. The oracles: the injected triangle count and the box [0, 1]^3.
    """
    m, triangles = _box_surface_reversed_into_netgen_3d()

    with m, pytest.raises(ps.PysmeshError) as info:
        m.compute()

    details = info.value.details
    match = re.search(FRONT_PATTERN, details)
    centroids = info.value.bad_element_centroids
    assert "too many attempts" in details
    assert match is not None
    faces, patches, largest = (int(match.group(i)) for i in (1, 2, 3))
    low = np.array([float(match.group(i)) for i in (4, 5, 6)])
    high = np.array([float(match.group(i)) for i in (7, 8, 9)])
    assert faces == triangles
    assert patches == 1
    assert largest == triangles
    np.testing.assert_allclose(low, 0.0, atol=1e-6)
    np.testing.assert_allclose(high, 1.0, atol=1e-6)
    assert centroids.shape == (triangles, 3)
    assert bool(np.all((centroids >= 0.0) & (centroids <= 1.0)))


def test_a_clean_netgen_3d_compute_reports_no_front() -> None:
    """Netgen1D2D3D on the unit box meshes it: no error and no front to report."""
    with Mesher(_box()) as m:
        m.assign(ps.Netgen1D2D3D())
        m.assign(ps.NetgenParameters(max_size=0.25, threads=4))

        report = m.compute()

    assert report.volumes > 0
    assert all("open front" not in w.text for w in report.warnings)
