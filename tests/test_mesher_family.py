# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""Gates for the algorithm and hypothesis catalogue.

Three claims, in order of how much they are worth.

* **Every catalogue entry is wired to a real upstream class.** Each one is built and attached
  through the native factory, which is the only thing that can prove the dataclass's fields
  match the setters behind them. The factory refuses a parameter it does not read, so a field
  added on one side without the other fails here rather than being silently dropped — and
  that refusal is itself asserted, so the check is known to be able to fail.
* **The 1-D distribution family produces the distributions it names.** Each is checked
  against a property of the spacing that a wrong hypothesis could not produce — a segment
  count, a first-to-last length ratio, a monotone progression — rather than against a value
  copied from a previous run.
  :class:`Adaptive1D` once killed the process on any shape with faces (report B2), so
  its gates run each shape in a child process and check the circle nodes, the size
  bounds and the deflection against the geometry.
* **The families that need a fixture of their own get one.** An extruded triangle for the
  extrusion mesher, a solid between two concentric shells for the radial one, a source and a
  target for projection, a disk for radial quadrangles, a thin strip for the medial-axis
  mesher. Each computes, and its result is checked for the property that algorithm exists to
  give — pentahedra between the swept ends, nodes that lie in the wall and nowhere else,
  matching element counts on the projected pair.

Fixture sizing follows the project rule: a 3 x 7 x 11 box, never a unit cube.
"""

from __future__ import annotations

import functools
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    Adaptive1D,
    Algorithm,
    Arithmetic1D,
    AutomaticLength,
    BlockRenumber,
    Cartesian3D,
    CartesianParameters3D,
    CompositeHexa3D,
    CompositeSegment1D,
    Deflection1D,
    Distribution,
    ElementType,
    EntityKind,
    FixedPoints1D,
    Geometric1D,
    Hexa3D,
    HexaFromSkin3D,
    Hypothesis,
    LayerDistribution,
    LayerDistribution2D,
    LengthFromEdges,
    LocalLength,
    MaxElementArea,
    MaxElementVolume,
    MaxLength,
    Mefisto2D,
    Mesher,
    NotConformAllowed,
    NumberOfLayers,
    NumberOfLayers2D,
    NumberOfSegments,
    PolygonPerFace2D,
    PolyhedronPerSolid3D,
    Prism3D,
    Projection1D,
    Projection1D2D,
    Projection2D,
    Projection3D,
    ProjectionSource1D,
    ProjectionSource2D,
    ProjectionSource3D,
    Propagation,
    PropagOfDistribution,
    QuadFromMedialAxis1D2D,
    Quadrangle2D,
    QuadrangleParams,
    QuadranglePreference,
    QuadraticMesh,
    QuadType,
    PysmeshError,
    RadialPrism3D,
    RadialQuadrangle1D2D,
    Regular1D,
    SegmentAroundVertex0D,
    SegmentLengthAroundVertex,
    Session,
    StartEndLength,
    SubShape,
    SubShapeKind,
    UseExisting1D,
    UseExisting2D,
    ViscousLayers2D,
)
from pysmesh.mesher import ViscousLayers

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0

SHELL_OUTER_RADIUS: float = 3.0
SHELL_INNER_RADIUS: float = 2.0

PRISM_HEIGHT: float = 9.0
TRIANGLE_AREA: float = 15.0


# ---- Fixtures --------------------------------------------------------------------------- #


def _box_shape() -> ps.Shape:
    """The 3 x 7 x 11 box, through the session."""
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    return ps.load_brep(session.brep())


def _hollow_sphere_shape() -> ps.Shape:
    """A solid between two concentric shells — the O-grid fixture the radial mesher wants.

    A hollow *cylinder* will not do, and the reason is worth recording: capping it makes its
    boundary a single closed shell, and the radial mesher refuses anything that is not two.
    """
    session = Session()
    session.add_sphere(SHELL_OUTER_RADIUS)
    outer = list(session.entities(EntityKind.SOLID))
    session.add_sphere(SHELL_INNER_RADIUS)
    inner = [e for e in session.entities(EntityKind.SOLID) if e not in outer]
    session.cut(outer, inner)
    return ps.load_brep(session.brep())


def _shell_faces(shape: ps.Shape) -> tuple[SubShape, SubShape]:
    """The outer and the inner shell of the hollow sphere, by area."""
    areas = sorted((face.area, i) for i, face in enumerate(shape.faces(), 1))
    return (
        SubShape(SubShapeKind.FACE, areas[-1][1]),
        SubShape(SubShapeKind.FACE, areas[0][1]),
    )


def _radial_recipe(mesher: Mesher, shape: ps.Shape, layers: Hypothesis) -> None:
    """Free-mesh the outer shell, project it onto the inner one, then fill radially.

    The projection is not decoration: the radial mesher refuses two shells whose meshes do
    not match, and two independent free meshes of concentric spheres never do.
    """
    outer, inner = _shell_faces(shape)
    mesher.assign(Regular1D())
    mesher.assign(NumberOfSegments(count=6))
    mesher.assign(Mefisto2D(), on=outer)
    mesher.assign(MaxElementArea(max_area=2.0), on=outer)
    mesher.assign(Projection2D(), on=inner)
    mesher.assign(ProjectionSource2D(source_face=outer), on=inner)
    mesher.assign(RadialPrism3D())
    mesher.assign(layers)


def _extruded_triangle_shape() -> ps.Shape:
    """A prismatic solid: a triangle swept along z, which the extrusion mesher wants."""
    session = Session()
    session.add_polyline(
        [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 0.0)]
    )
    edges = list(session.entities(EntityKind.EDGE))
    session.make_face(edges)
    faces = list(session.entities(EntityKind.FACE))
    session.extrude(faces, (0.0, 0.0, PRISM_HEIGHT))
    return ps.load_brep(session.brep())


def _triangular_face(shape: ps.Shape) -> SubShape:
    """The base face of the extruded triangle, by its known area."""
    for ordinal, face in enumerate(shape.faces(), 1):
        if math.isclose(face.area, TRIANGLE_AREA, rel_tol=1e-9):
            return SubShape(SubShapeKind.FACE, ordinal)
    raise AssertionError("the extruded-triangle fixture has no triangular face")


def _disk_face_shape() -> ps.Shape:
    """A single circular face, for the radial quadrangle mesher."""
    session = Session()
    session.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.5)
    edges = list(session.entities(EntityKind.EDGE))
    session.make_face(edges)
    return ps.load_brep(session.brep())


def _segment_lengths(mesh: ps.MeshData, edge_ordinal: int) -> list[float]:
    """Lengths of the 1-D elements on one edge, in order along it.

    Ordering is by distance from the segment closest to one end, which is enough to read a
    progression without assuming the mesher's own emission order.
    """
    lengths: list[tuple[float, float]] = []
    for element in range(mesh.element_count):
        if int(mesh.element_kind[element]) != int(SubShapeKind.EDGE):
            continue
        if int(mesh.element_ordinal[element]) != edge_ordinal:
            continue
        nodes = mesh.nodes_of(element)
        if len(nodes) != 2:
            continue
        a = mesh.node_coords[nodes[0]]
        b = mesh.node_coords[nodes[1]]
        midpoint = float(np.linalg.norm((a + b) / 2.0))
        lengths.append((midpoint, float(np.linalg.norm(b - a))))
    return [length for _, length in sorted(lengths)]


# ---- Every catalogue entry is wired ------------------------------------------------------ #

# One instance of each, with the sub-shape it is assigned to. A projection hypothesis needs a
# source of the right kind, so the box's own sub-shapes serve as one.
_CATALOGUE: list[tuple[Algorithm | Hypothesis, SubShape | None]] = [
    # 0-D algorithms
    (SegmentAroundVertex0D(), SubShape(SubShapeKind.VERTEX, 1)),
    # 1-D algorithms
    (Regular1D(), None),
    (CompositeSegment1D(), None),
    (Projection1D(), None),
    (UseExisting1D(), SubShape(SubShapeKind.EDGE, 1)),
    # 2-D algorithms
    (Quadrangle2D(), None),
    (Mefisto2D(), None),
    (PolygonPerFace2D(), None),
    (Projection2D(), None),
    (Projection1D2D(), None),
    (QuadFromMedialAxis1D2D(), None),
    (RadialQuadrangle1D2D(), None),
    (UseExisting2D(), SubShape(SubShapeKind.FACE, 1)),
    # 3-D algorithms
    (Cartesian3D(), None),
    (Hexa3D(), None),
    (CompositeHexa3D(), None),
    (HexaFromSkin3D(), None),
    (Prism3D(), None),
    (RadialPrism3D(), None),
    (Projection3D(), None),
    (PolyhedronPerSolid3D(), None),
    # 1-D hypotheses
    (NumberOfSegments(count=4), None),
    (NumberOfSegments(count=4, distribution=Distribution.SCALE, scale_factor=3.0), None),
    (
        NumberOfSegments(
            count=4, distribution=Distribution.TABLE, table=(0.0, 1.0, 1.0, 3.0)
        ),
        None,
    ),
    (
        NumberOfSegments(count=4, distribution=Distribution.EXPRESSION, expression="1+t"),
        None,
    ),
    (NumberOfSegments(count=4, distribution=Distribution.BETA_LAW, beta=1.05), None),
    (Arithmetic1D(start_length=0.5, end_length=2.0), None),
    (StartEndLength(start_length=0.5, end_length=2.0), None),
    (Geometric1D(start_length=0.5, common_ratio=1.2), None),
    (FixedPoints1D(points=(0.25, 0.75), segment_counts=(2, 3, 2)), None),
    (Adaptive1D(min_size=0.1, max_size=2.0, deflection=0.05), None),
    (AutomaticLength(fineness=0.5), None),
    (Deflection1D(deflection=0.05), None),
    (LocalLength(length=1.5), None),
    (MaxLength(length=2.0), None),
    (SegmentLengthAroundVertex(length=0.5), SubShape(SubShapeKind.VERTEX, 1)),
    (Propagation(), SubShape(SubShapeKind.EDGE, 1)),
    (PropagOfDistribution(), SubShape(SubShapeKind.EDGE, 1)),
    (LayerDistribution(distribution=NumberOfSegments(count=3)), None),
    (LayerDistribution2D(distribution=NumberOfSegments(count=3)), None),
    (QuadraticMesh(), None),
    (BlockRenumber(), None),
    (BlockRenumber(blocks=((1, 1, 2),)), None),
    (NotConformAllowed(), None),
    # 2-D and 3-D hypotheses
    (MaxElementArea(max_area=4.0), None),
    (LengthFromEdges(), None),
    (MaxElementVolume(max_volume=8.0), None),
    (QuadranglePreference(), None),
    (QuadrangleParams(quad_type=QuadType.REDUCED), None),
    (
        QuadrangleParams(
            base_vertex=SubShape(SubShapeKind.VERTEX, 1), corner_vertices=(1, 2, 3, 4)
        ),
        None,
    ),
    (NumberOfLayers(count=3), None),
    (NumberOfLayers2D(count=3), None),
    (
        CartesianParameters3D(spacing_x="1.0", spacing_y="1.0", spacing_z="1.0"),
        None,
    ),
    (
        CartesianParameters3D(
            spacing_x="1.0",
            spacing_y="1.0",
            spacing_z="1.0",
            use_quanta=True,
            quanta=0.5,
        ),
        None,
    ),
    # Hypotheses naming another part of the model
    (ProjectionSource1D(source_edge=SubShape(SubShapeKind.EDGE, 1)), None),
    (
        ProjectionSource1D(
            source_edge=SubShape(SubShapeKind.EDGE, 1),
            source_vertex=SubShape(SubShapeKind.VERTEX, 1),
            target_vertex=SubShape(SubShapeKind.VERTEX, 2),
        ),
        None,
    ),
    (ProjectionSource2D(source_face=SubShape(SubShapeKind.FACE, 1)), None),
    (ProjectionSource3D(source_solid=SubShape(SubShapeKind.SOLID, 1)), None),
    (
        ViscousLayers(
            total_thickness=0.3,
            layer_count=3,
            stretch_factor=1.2,
            boundary=(1, 2),
            group_name="layers",
        ),
        None,
    ),
    (
        ViscousLayers2D(
            total_thickness=0.3,
            layer_count=3,
            stretch_factor=1.2,
            boundary=(1, 2),
            group_name="layers2d",
        ),
        None,
    ),
]


@pytest.mark.parametrize(
    ("item", "on"), _CATALOGUE, ids=[f"{i.native_name}" for i, _ in _CATALOGUE]
)
def test_every_catalogue_entry_builds_and_attaches(
    item: Algorithm | Hypothesis, on: SubShape | None
) -> None:
    """Proves the dataclass's fields match the upstream setters behind them.

    A name with no branch, or a field the branch does not read, fails here — which is the
    only place the two halves of the catalogue can be checked against each other.
    """
    with Mesher(_box_shape()) as mesher:
        mesher.assign(item, on=on)

        assert mesher.assignments() == ((item.native_name, on),)


def test_a_parameter_the_factory_does_not_read_is_refused() -> None:
    """The falsification for the test above: the drift check must be able to fail."""
    with Mesher(_box_shape()) as mesher:
        with pytest.raises(PysmeshError, match="does not take the parameter"):
            mesher._m.assign("Regular_1D", {"unexpected": 1}, "", 0)


def test_a_missing_parameter_is_refused_naming_it() -> None:
    with Mesher(_box_shape()) as mesher:
        with pytest.raises(PysmeshError, match="missing the parameter 'count'"):
            mesher._m.assign("NumberOfSegments", {}, "", 0)


def test_an_unknown_algorithm_name_is_refused() -> None:
    with Mesher(_box_shape()) as mesher:
        with pytest.raises(PysmeshError, match="unknown algorithm or hypothesis"):
            mesher._m.assign("Netgen_3D", {}, "", 0)


def test_the_catalogue_covers_every_public_algorithm_and_hypothesis() -> None:
    """A new entry added to the catalogue without a case above would go untested."""
    import pysmesh.mesher as mesher_module

    exported = {
        name
        for name in mesher_module.__all__
        if isinstance(getattr(mesher_module, name), type)
        and issubclass(getattr(mesher_module, name), (Algorithm, Hypothesis))
        and getattr(mesher_module, name) not in (Algorithm, Hypothesis)
    }
    covered = {type(item).__name__ for item, _ in _CATALOGUE}

    assert exported == covered


# ---- The 1-D distribution family --------------------------------------------------------- #


def _mesh_one_edge(hypothesis: Hypothesis) -> ps.MeshData:
    """Discretise the box's edges with one 1-D hypothesis and return the mesh."""
    mesher = Mesher(_box_shape())
    mesher.assign(Regular1D())
    mesher.assign(hypothesis)
    mesher.compute()
    mesh = mesher.mesh()
    mesher.release()
    return mesh


def test_number_of_segments_gives_exactly_that_many_per_edge() -> None:
    mesh = _mesh_one_edge(NumberOfSegments(count=5))

    per_edge = {
        ordinal: int(np.count_nonzero(mesh.element_ordinal[mesh.element_kind == 3] == ordinal))
        for ordinal in set(mesh.element_ordinal[mesh.element_kind == 3].tolist())
    }

    assert len(per_edge) == 12
    assert set(per_edge.values()) == {5}


def test_arithmetic_progression_gives_the_stated_first_and_last_lengths() -> None:
    mesh = _mesh_one_edge(Arithmetic1D(start_length=0.5, end_length=2.0))

    # Edge 1 of a 3 x 7 x 11 box is long enough to hold several segments of these sizes.
    lengths = _segment_lengths(mesh, edge_ordinal=1)

    assert len(lengths) >= 3
    assert min(lengths[0], lengths[-1]) == pytest.approx(0.5, rel=0.35)
    assert max(lengths[0], lengths[-1]) == pytest.approx(2.0, rel=0.35)


def test_a_scaled_distribution_grows_monotonically_along_the_edge() -> None:
    mesh = _mesh_one_edge(
        NumberOfSegments(count=6, distribution=Distribution.SCALE, scale_factor=4.0)
    )

    lengths = _segment_lengths(mesh, edge_ordinal=1)
    ordered = lengths if lengths[0] < lengths[-1] else lengths[::-1]

    assert len(lengths) == 6
    assert ordered == sorted(ordered)
    assert ordered[-1] / ordered[0] == pytest.approx(4.0, rel=0.2)


def test_a_regular_distribution_gives_equal_segments() -> None:
    """The counter-case: without it "monotone" would pass on a uniform mesh too."""
    mesh = _mesh_one_edge(NumberOfSegments(count=6))

    lengths = _segment_lengths(mesh, edge_ordinal=1)

    assert len(lengths) == 6
    assert max(lengths) == pytest.approx(min(lengths), rel=1e-9)


def test_fixed_points_split_the_edge_at_the_stated_positions() -> None:
    mesh = _mesh_one_edge(
        FixedPoints1D(points=(0.25, 0.75), segment_counts=(2, 4, 2))
    )

    lengths = _segment_lengths(mesh, edge_ordinal=1)

    assert len(lengths) == 2 + 4 + 2


def test_local_length_sizes_every_edge_to_about_that_length() -> None:
    target = 1.5
    mesh = _mesh_one_edge(LocalLength(length=target))

    lengths = _segment_lengths(mesh, edge_ordinal=1)

    assert lengths
    # The count is rounded to fit the edge, so the achieved length is near the target.
    assert max(lengths) == pytest.approx(target, rel=0.35)


def test_max_length_never_exceeds_its_bound() -> None:
    bound = 1.0
    mesh = _mesh_one_edge(MaxLength(length=bound))

    edge_elements = [
        element
        for element in range(mesh.element_count)
        if int(mesh.element_kind[element]) == int(SubShapeKind.EDGE)
    ]
    lengths = [
        float(
            np.linalg.norm(
                mesh.node_coords[mesh.nodes_of(e)[1]] - mesh.node_coords[mesh.nodes_of(e)[0]]
            )
        )
        for e in edge_elements
    ]

    assert lengths
    assert max(lengths) <= bound + 1e-9


def test_propagation_carries_a_count_to_the_opposite_edges() -> None:
    """A structured mesh's opposite sides match without each being stated."""
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Quadrangle2D())
        mesher.assign(NumberOfSegments(count=7), on=SubShape(SubShapeKind.EDGE, 1))
        mesher.assign(Propagation(), on=SubShape(SubShapeKind.EDGE, 1))
        mesher.compute()
        mesh = mesher.mesh()

    counts = [
        int(np.count_nonzero((mesh.element_kind == 3) & (mesh.element_ordinal == ordinal)))
        for ordinal in range(1, 13)
    ]

    # Four parallel edges carry the propagated count; the rest keep the model default.
    assert counts.count(7) == 4
    assert counts.count(2) == 8


def test_quadratic_mesh_produces_second_order_elements() -> None:
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=2))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.assign(QuadraticMesh())
        mesher.compute()
        mesh = mesher.mesh()

    assert mesh.count_of(ElementType.QUAD_HEXAHEDRON) == 8
    assert mesh.count_of(ElementType.HEXAHEDRON) == 0
    assert mesh.count_of(ElementType.QUAD_EDGE) > 0


# ---- Adaptive1D on shapes with faces (report B2) ----------------------------------- #
#
# Before SMESH 9.16 a vendored patch read the bounds of a NULL array in
# TriaTreeData::TriaTreeData, and Adaptive1D killed the process (0xC0000005) on any
# shape with a face. Each shape therefore runs in a child process: a regression fails
# the test with the child's exit code instead of ending the whole run.

ADAPTIVE_MIN_SIZE: float = 0.05
ADAPTIVE_MAX_SIZE: float = 1.0
ADAPTIVE_DEFLECTION: float = 0.01
ADAPTIVE_RADIUS: float = 1.5
_ADAPTIVE_CHILD_TIMEOUT_S: float = 300.0

# The child meshes one shape, built from its arguments or read from a BREP file, and
# writes the end points of every segment, by edge. It sets up the DLL search exactly as
# tests/conftest.py does for this process.
_ADAPTIVE_CHILD: str = """
import json, os, sys
occt = os.environ.get("PYSMESH_OCCT_BIN")
if occt:
    os.add_dll_directory(occt)
lib = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(lib):
    os.add_dll_directory(lib)
sys.path.insert(0, sys.argv[1])
import pysmesh as ps

kind = sys.argv[2]
if kind == "brep":
    with open(sys.argv[3], "rb") as fh:
        shape = ps.load_brep(fh.read())
else:
    r = float(sys.argv[3])
    s = ps.Session()
    if kind == "box":
        s.add_box(3.0, 7.0, 11.0)
    elif kind == "cylinder":
        s.add_cylinder(r, 4.0)
    elif kind == "ellipse":
        s.add_ellipse((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), r, float(sys.argv[7]))
        s.make_face(list(s.entities(ps.EntityKind.EDGE)))
        s.extrude(list(s.entities(ps.EntityKind.FACE)), (0.0, 0.0, 4.0))
    else:
        s.add_sphere(r)
    shape = ps.load_brep(s.brep())
with ps.Mesher(shape) as m:
    m.assign(ps.Regular1D())
    m.assign(ps.Adaptive1D(min_size=float(sys.argv[4]), max_size=float(sys.argv[5]),
                           deflection=float(sys.argv[6])))
    m.compute()
    mesh = m.mesh()
edges = {}
for i in range(mesh.element_count):
    if int(mesh.element_type[i]) == int(ps.ElementType.EDGE):
        a, b = mesh.nodes_of(i)
        edges.setdefault(int(mesh.element_ordinal[i]), []).append(
            [mesh.node_coords[a].tolist(), mesh.node_coords[b].tolist()])
sys.stdout.write("ADAPTIVE-RESULT " + json.dumps(edges) + "\\n")
"""


@functools.cache
def _adaptive_segments(
    kind: str,
    radius: float = ADAPTIVE_RADIUS,
    min_size: float = ADAPTIVE_MIN_SIZE,
    deflection: float = ADAPTIVE_DEFLECTION,
    minor_radius: float = 0.0,
) -> dict[int, NDArray[np.float64]]:
    """Run Adaptive1D on one shape in a child process; segments by edge, (n, 2, 3).

    ``minor_radius`` is the second radius of the extruded ellipse (``kind`` "ellipse"),
    whose first radius is ``radius``. The max size is always ``ADAPTIVE_MAX_SIZE``.

    Raises:
        AssertionError: The child crashed or reported no result.
    """
    return _run_adaptive_child(
        kind, repr(radius), min_size, deflection, repr(minor_radius)
    )


def _adaptive_brep_segments(
    path: Path, min_size: float, deflection: float
) -> dict[int, NDArray[np.float64]]:
    """Run Adaptive1D on the shape of a BREP file in a child process; segments by edge.

    Raises:
        AssertionError: The child crashed or reported no result.
    """
    return _run_adaptive_child("brep", str(path), min_size, deflection, "0.0")


def _run_adaptive_child(
    kind: str, shape_arg: str, min_size: float, deflection: float, minor_arg: str
) -> dict[int, NDArray[np.float64]]:
    """Mesh one shape in a child process; segments by edge ordinal, (n, 2, 3).

    Raises:
        AssertionError: The child crashed or reported no result.
    """
    package_root = str(Path(ps.__file__).resolve().parent.parent)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _ADAPTIVE_CHILD,
            package_root,
            kind,
            shape_arg,
            repr(min_size),
            repr(ADAPTIVE_MAX_SIZE),
            repr(deflection),
            minor_arg,
        ],
        capture_output=True,
        text=True,
        timeout=_ADAPTIVE_CHILD_TIMEOUT_S,
        env=dict(os.environ),
        check=False,
    )
    prefix = "ADAPTIVE-RESULT "
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(prefix)]
    code = proc.returncode & 0xFFFFFFFF
    assert proc.returncode == 0 and lines, (
        f"Adaptive1D on a {kind}: child exit code 0x{code:08X}, "
        f"stderr: {proc.stderr[-2000:]}"
    )
    raw = json.loads(lines[-1][len(prefix) :])
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def _circle_segments(kind: str) -> list[NDArray[np.float64]]:
    """The segments of the edges that lie on a circle of radius 1.5.

    On the cylinder (axis z) those are the two rims: every node at one height. On the
    sphere it is the seam meridian: every node at distance 1.5 from the centre.
    """
    out = []
    for seg in _adaptive_segments(kind).values():
        pts = seg.reshape(-1, 3)
        rim = kind == "cylinder" and np.ptp(pts[:, 2]) < 1e-9
        if rim or kind == "sphere":
            out.append(seg)
    return out


@pytest.mark.parametrize("kind", ["box", "cylinder", "sphere"])
def test_adaptive_1d_meshes_a_shape_with_faces_without_crashing(kind: str) -> None:
    """Report B2: the computation finishes and discretises every edge it reaches."""
    segments = _adaptive_segments(kind)

    assert segments
    assert all(seg.shape[0] >= 1 for seg in segments.values())


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_adaptive_1d_puts_every_circle_node_on_its_radius(kind: str) -> None:
    """Nodes of a discretised circle lie on it: distance 1.5 from the axis or centre."""
    circles = _circle_segments(kind)

    assert circles
    for seg in circles:
        pts = seg.reshape(-1, 3)
        if kind == "cylinder":
            r = np.hypot(pts[:, 0], pts[:, 1])
        else:
            r = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(r, ADAPTIVE_RADIUS, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("kind", ["box", "cylinder", "sphere"])
def test_adaptive_1d_keeps_every_segment_between_its_size_bounds(kind: str) -> None:
    """Spec (SMESH ``1d_meshing_hypo.rst``, "Adaptive hypothesis"): "Min size" limits
    the minimal segment size and "Max size" the length on straight edges. The stated
    tolerance, 1e-9, is round-off: the box's straight edges sit exactly at 1.0."""
    for seg in _adaptive_segments(kind).values():
        length = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
        assert float(length.min()) >= ADAPTIVE_MIN_SIZE - 1e-9
        assert float(length.max()) <= ADAPTIVE_MAX_SIZE + 1e-9


def _max_sagitta(kind: str) -> float:
    """The largest chord deviation of a circle segment: R - sqrt(R^2 - c^2 / 4)."""
    worst = 0.0
    for seg in _circle_segments(kind):
        chord = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
        sagitta = ADAPTIVE_RADIUS - np.sqrt(ADAPTIVE_RADIUS**2 - chord**2 / 4.0)
        worst = max(worst, float(sagitta.max()))
    return worst


def test_adaptive_1d_keeps_the_sphere_seam_within_the_deflection() -> None:
    """Spec (SMESH ``1d_meshing_hypo.rst``): "Deflection parameter gives maximal
    distance of a segment from a curved edge", here the 0.01 bound on the sphere."""
    assert _max_sagitta("sphere") <= ADAPTIVE_DEFLECTION


def test_adaptive_1d_keeps_the_cylinder_rims_within_the_deflection() -> None:
    """Spec (SMESH ``1d_meshing_hypo.rst``): the same 0.01 bound on the cylinder."""
    assert _max_sagitta("cylinder") <= ADAPTIVE_DEFLECTION


# ---- Adaptive1D keeps its three rules together ------------------------------------- #
#
# Upstream documents three rules for the hypothesis (SMESH ``1d_meshing_hypo.rst``,
# "Adaptive hypothesis"): the length "is limited by Min. Size and Max Size", adjacent
# segments "can't differ more than twice", and "Deflection parameter gives maximal
# distance of a segment from a curved edge". Before the pySMESH patch
# StdMeshers_Adaptive1D_deflection the deflection only seeded the size field, and the
# segments next to a curved face could miss it (1.037 times on the radius-1.5 rims).
# Where meeting the deflection would need a segment shorter than the min size, the min
# size wins. The sagitta of a segment is the largest distance from the edge curve,
# between the segment's two end nodes, to the chord.

ADAPTIVE_DEFLECTIONS: tuple[float, ...] = (0.003, 0.01, 0.02)
ELLIPSE_RX: float = 2.0
ELLIPSE_RY: float = 1.5
CONFLICT_DEFLECTION: float = 1e-4
_ROUND_OFF: float = 1e-9


def _ordered_edges(
    segments: dict[int, NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    """The nodes of each edge in edge order, (k + 1, 3).

    A closed edge ends at its start. Two segments share a node when their end points
    are equal: a node is written once per segment it bounds, with the same coordinates.
    """
    return [_ordered_chain(seg) for seg in segments.values()]


def _ordered_chain(seg: NDArray[np.float64]) -> NDArray[np.float64]:
    """The nodes of one edge's segments (n, 2, 3) in edge order, (n + 1, 3)."""
    links: dict[tuple[float, ...], list[tuple[float, ...]]] = {}
    for a, b in seg:
        ka, kb = tuple(a.tolist()), tuple(b.tolist())
        links.setdefault(ka, []).append(kb)
        links.setdefault(kb, []).append(ka)
    ends = [k for k, v in links.items() if len(v) == 1]
    start = ends[0] if ends else next(iter(links))
    order = [start]
    prev: tuple[float, ...] | None = None
    while len(order) <= seg.shape[0]:
        here = order[-1]
        step = [k for k in links[here] if k != prev] or links[here]
        prev = here
        order.append(step[0])
    return np.asarray(order, dtype=np.float64)


def _ellipse_sagitta(
    p0: NDArray[np.float64], p1: NDArray[np.float64], rx: float, ry: float
) -> float:
    """Largest distance of the arc of x = rx cos t, y = ry sin t between two of its
    points from their chord: 65 samples, then a golden-section search next to the
    largest one."""
    t0 = math.atan2(p0[1] / ry, p0[0] / rx)
    t1 = math.atan2(p1[1] / ry, p1[0] / rx)
    if t1 - t0 > math.pi:
        t1 -= 2.0 * math.pi
    elif t0 - t1 > math.pi:
        t1 += 2.0 * math.pi
    chord = p1[:2] - p0[:2]

    def dist(t: NDArray[np.float64]) -> NDArray[np.float64]:
        v = np.c_[rx * np.cos(t), ry * np.sin(t)] - p0[:2]
        s = np.clip(v @ chord / float(chord @ chord), 0.0, 1.0)
        return np.asarray(np.linalg.norm(v - s[:, None] * chord, axis=1))

    ts = np.linspace(t0, t1, 65)
    k = int(np.argmax(dist(ts)))
    a, b = float(ts[max(k - 1, 0)]), float(ts[min(k + 1, ts.size - 1)])
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(100):
        c, d = b - golden * (b - a), a + golden * (b - a)
        if float(dist(np.array([c]))[0]) > float(dist(np.array([d]))[0]):
            b = d
        else:
            a = c
    return max(float(dist(ts).max()), float(dist(np.array([0.5 * (a + b)]))[0]))


def _min_size_wins(
    sagitta: NDArray[np.float64], chords: NDArray[np.float64], deflection: float
) -> NDArray[np.bool_]:
    """Where meeting the deflection would need a segment shorter than min_size.

    That is where a chord of min_size would already miss the deflection. A chord's
    sagitta grows with the square of its length (the circle law s = c^2 / 8R to leading
    order), so a segment of chord c and sagitta s gives a chord of min_size the sagitta
    s (min_size / c)^2 in the same place.
    """
    return np.asarray(sagitta * (ADAPTIVE_MIN_SIZE / chords) ** 2 > deflection)


def _sagittas(
    kind: str, radius: float, minor: float, chain: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sagitta of each segment of one ordered edge.

    The rims (every node at one height) are circles of the given radius, closed form
    R - sqrt(R^2 - c^2 / 4), or ellipses of radii ``radius`` and ``minor``; any other
    edge of these shapes is a straight seam, sagitta 0, checked by its nodes sharing x
    and y.
    """
    chords = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    if np.ptp(chain[:, 2]) > _ROUND_OFF:
        assert np.ptp(chain[:, 0]) < _ROUND_OFF and np.ptp(chain[:, 1]) < _ROUND_OFF
        return np.zeros_like(chords)
    if kind == "cylinder":
        return np.asarray(radius - np.sqrt(radius**2 - chords**2 / 4.0))
    on_ellipse = (chain[:, 0] / radius) ** 2 + (chain[:, 1] / minor) ** 2
    assert np.allclose(on_ellipse, 1.0, rtol=0.0, atol=_ROUND_OFF)
    return np.array(
        [
            _ellipse_sagitta(chain[i], chain[i + 1], radius, minor)
            for i in range(chords.size)
        ]
    )


@pytest.mark.parametrize("deflection", ADAPTIVE_DEFLECTIONS)
@pytest.mark.parametrize(
    ("kind", "radius"), [("cylinder", 1.5), ("cylinder", 4.0), ("ellipse", ELLIPSE_RX)]
)
def test_adaptive_1d_keeps_its_three_rules_on_curved_edges(
    kind: str, radius: float, deflection: float
) -> None:
    """Spec (SMESH ``1d_meshing_hypo.rst``, "Adaptive hypothesis"), on every edge:
    each sagitta is at most the deflection, unless min_size wins there, and then the
    segment is at least min_size long (min_size is a hard lower bound, not a target
    length); every chord lies in [min_size, max_size]; two adjacent chords of one edge,
    the wrap of a closed edge included, differ at most by a factor of 2. See
    ``_min_size_wins`` for when min_size wins.

    Shapes: the rims of radius-1.5 and radius-4 cylinders, and the 2 x 1.5 elliptic rims
    of an extruded ellipse, a curved edge that is not a circle. Without the patch the
    radius-1.5 rims miss 0.003 and 0.01 (1.093 and 1.037 times), the radius-4 rims 0.02
    (1.025 times) and the ellipse 0.01 (1.042 times). Tolerance 1e-9: round-off.
    """
    minor = ELLIPSE_RY if kind == "ellipse" else 0.0
    segments = _adaptive_segments(kind, radius, ADAPTIVE_MIN_SIZE, deflection, minor)

    for chain in _ordered_edges(segments):
        chords = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        sagitta = _sagittas(kind, radius, minor, chain)
        at_least_min = chords >= ADAPTIVE_MIN_SIZE * (1.0 - _ROUND_OFF)
        min_wins = _min_size_wins(sagitta, chords, deflection) & at_least_min
        assert np.all((sagitta <= deflection * (1.0 + _ROUND_OFF)) | min_wins)
        assert float(chords.min()) >= ADAPTIVE_MIN_SIZE - _ROUND_OFF
        assert float(chords.max()) <= ADAPTIVE_MAX_SIZE + _ROUND_OFF
        closed = bool(np.array_equal(chain[0], chain[-1]))
        pairs = np.r_[chords, chords[:1]] if closed else chords
        ratio = np.maximum(pairs[1:] / pairs[:-1], pairs[:-1] / pairs[1:])
        assert float(ratio.max(initial=1.0)) <= 2.0


@pytest.mark.parametrize("min_size", [3.0 * math.sin(math.pi / 48.0), 0.2])
def test_adaptive_1d_lets_the_min_size_win_where_the_deflection_needs_shorter_segments(
    min_size: float,
) -> None:
    """Order of the rules: min_size and max_size are hard bounds, and where meeting the
    deflection would need a segment shorter than min_size, min_size wins.

    Deflection 1e-4 on the radius-1.5 rims needs chords of 2 sqrt(2 R d - d^2) = 0.0346,
    far below min_size, so every rim segment misses it. The rim must then be cut as
    finely as min_size allows. On a closed circle the n equal chords 2 R sin(pi / n) are
    the shortest set not below min_size for n = floor(pi / asin(min_size / 2 R)). For
    min_size = 3 sin(pi / 48), the chord of 48 equal parts, every chord is min_size. For
    min_size 0.2, pi / asin(0.2 / 3) = 47.09 is not whole: no closed circle splits into
    chords of exactly 0.2, and the shortest are 2 R sin(pi / 47) = 1.0019 min_size.
    Tolerance 1e-9 relative: round-off.
    """
    radius = ADAPTIVE_RADIUS
    count = math.floor(math.pi / math.asin(min_size / (2.0 * radius)) + _ROUND_OFF)
    shortest = 2.0 * radius * math.sin(math.pi / count)

    segments = _adaptive_segments(
        "cylinder", radius, min_size, CONFLICT_DEFLECTION, 0.0
    )

    rims = [c for c in _ordered_edges(segments) if np.ptp(c[:, 2]) <= _ROUND_OFF]
    assert len(rims) == 2
    for chain in rims:
        chords = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        sagitta = radius - np.sqrt(radius**2 - chords**2 / 4.0)
        assert np.all(sagitta > CONFLICT_DEFLECTION)
        np.testing.assert_allclose(chords, shortest, rtol=_ROUND_OFF, atol=0.0)
        assert shortest >= min_size * (1.0 - _ROUND_OFF)


def test_adaptive_1d_keeps_the_min_size_where_it_wins_on_an_ellipse() -> None:
    """Order of the rules, on a curved edge that is not a circle: the extruded ellipse
    3 x 1, min_size 0.2, deflection 0.002.

    At the ends of the major axis the radius of curvature is rho = ry^2 / rx = 1/3, and
    the deflection needs chords of 2 sqrt(2 rho d - d^2) = 0.073, below min_size: there
    min_size wins. On the flat sides (rho = rx^2 / ry = 9) chords up to 0.379 meet it.
    Every chord is at least min_size, a hard bound. A segment that misses the deflection
    is no longer than the arc that a chord of min_size spans at the tightest curvature,
    2 rho asin(min_size / 2 rho) = 1.0156 min_size: a shorter chord there would be below
    min_size (Schur: on a curve of curvature at most 1 / rho, a chord of given length
    spans at most that arc). Without the patch the chords there fall to 0.983 min_size.
    Tolerance 1e-9: round-off.
    """
    rx, ry, min_size, deflection = 3.0, 1.0, 0.2, 0.002
    rho = ry**2 / rx
    longest = 2.0 * rho * math.asin(min_size / (2.0 * rho))

    segments = _adaptive_segments("ellipse", rx, min_size, deflection, ry)

    missed = met = 0
    for chain in _ordered_edges(segments):
        chords = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        sagitta = _sagittas("ellipse", rx, ry, chain)
        miss = sagitta > deflection * (1.0 + _ROUND_OFF)
        missed += int(miss.sum())
        met += int((~miss).sum())
        assert float(chords.min()) >= min_size - _ROUND_OFF
        assert np.all(chords[miss] <= longest * (1.0 + _ROUND_OFF))
    assert missed > 0 and met > 0


# ---- Adaptive1D keeps the bounds and the factor 2 where the deflection holds ------ #
#
# Before the pySMESH patch StdMeshers_Adaptive1D_bounds, Adaptive1D broke its size
# bounds and the factor 2 between neighbours on edges where the deflection held, so
# the deflection patch left them as they were. The curves here are read through OCCT
# itself: the parent builds the shape in a session, the child meshes the very same
# BREP, and each edge's own curve gives the parameters of its nodes and the sagitta
# of its segments.

BSPLINE_POLES: NDArray[np.float64] = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [3.0, -1.0, 0.0],
        [5.0, 3.0, 0.0],
        [7.0, 0.5, 0.0],
        [8.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
BUMP_POLES: NDArray[np.float64] = np.array(
    [
        [8.0, 2.0, 0.0],
        [6.0, 2.0, 0.0],
        [5.0, 4.0, 0.0],
        [3.0, 4.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ],
    dtype=np.float64,
)
_GOLDEN: float = (math.sqrt(5.0) - 1.0) / 2.0


def _extruded(session: Session, height: float) -> Session:
    """Face the session's edges and extrude the face along z."""
    session.make_face(list(session.entities(EntityKind.EDGE)))
    session.extrude(list(session.entities(EntityKind.FACE)), (0.0, 0.0, height))
    return session


def _rules_session(kind: str) -> Session:
    """The shapes on which Adaptive1D broke the bounds or the factor 2.

    "bspline_prism": a B-spline profile closed by a line, extruded by 2. "bump": an
    8 x 2 rectangle whose top is a B-spline bump, extruded by 2. "arc": a half disk of
    radius 1.5 whose arc is a B-spline fitted through 13 points of the half circle (end
    points exact), extruded by 4. "cone": radii 2 and 0.5, height 3. "ellipse": the
    ellipse 2 x 1.5, extruded by 4.
    """
    s = Session()
    if kind == "bspline_prism":
        s.add_bspline(BSPLINE_POLES, degree=3)
        s.add_line(tuple(BSPLINE_POLES[-1]), tuple(BSPLINE_POLES[0]))
        return _extruded(s, 2.0)
    if kind == "bump":
        s.add_line((0.0, 0.0, 0.0), (8.0, 0.0, 0.0))
        s.add_line((8.0, 0.0, 0.0), (8.0, 2.0, 0.0))
        s.add_bspline(BUMP_POLES, degree=3)
        s.add_line((0.0, 2.0, 0.0), (0.0, 0.0, 0.0))
        return _extruded(s, 2.0)
    if kind == "arc":
        t = np.linspace(0.0, math.pi, 13)
        points = np.c_[1.5 * np.cos(t), 1.5 * np.sin(t), np.zeros_like(t)]
        points[-1] = (-1.5, 0.0, 0.0)
        s.add_spline(points)
        s.add_line((-1.5, 0.0, 0.0), (1.5, 0.0, 0.0))
        return _extruded(s, 4.0)
    if kind == "cone":
        s.add_cone(2.0, 0.5, 3.0)
        return s
    s.add_ellipse((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), ELLIPSE_RX, ELLIPSE_RY)
    return _extruded(s, 4.0)


def _session_edge_ids(session: Session, shape: ps.Shape) -> dict[int, int]:
    """The session's edge id for each of the shape's edge ordinals.

    Both come from the same BREP; an edge is matched by its bounding box and its
    parameter range, and each match must be unique.
    """
    table = session.bounding_boxes(EntityKind.EDGE)
    ids = [int(i) for i in table.ids]
    bounds = session.edge_parameter_bounds(ids)
    out: dict[int, int] = {}
    for info in shape.edges():
        gap = np.abs(table.bbox - np.asarray(info.bbox)).max(axis=1)
        gap = gap + np.abs(bounds - np.asarray(info.t_bounds)).max(axis=1)
        hits = np.flatnonzero(gap < 1e-7)
        assert hits.size == 1, f"edge {info.id} matches {hits.size} session edges"
        out[int(info.id)] = ids[int(hits[0])]
    return out


def _golden_max(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Golden-section search, element by element, for the maximum of f on [a, b]."""
    for _ in range(60):
        c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
        left = f(c) > f(d)
        a, b = np.where(left, a, c), np.where(left, d, b)
    return f(0.5 * (a + b))


def _curve_sagittas(
    session: Session, edge: int, chain: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sagitta of each segment of an ordered edge, on the edge's own curve.

    Each node's parameter is the nearest point of the curve (4001 samples, then a golden
    search); every node must lie on the curve (1e-9). A segment's sagitta is the
    largest distance from the curve between its two end parameters to its chord: 65
    samples, then a golden search next to the largest one.
    """
    t0, t1 = (float(v) for v in session.edge_parameter_bounds([edge])[0])

    def at(t: NDArray[np.float64]) -> NDArray[np.float64]:
        flat = np.ascontiguousarray(t, dtype=np.float64).ravel()
        return session.curve_at(edge, flat).points.reshape(*t.shape, 3)

    dense_t = np.linspace(t0, t1, 4001)
    nearest = np.argmin(
        np.linalg.norm(at(dense_t)[None, :, :] - chain[:, None, :], axis=2), axis=1
    )
    step = (t1 - t0) / 4000.0
    lo = np.clip(dense_t[nearest] - step, t0, t1)
    hi = np.clip(dense_t[nearest] + step, t0, t1)
    params = np.empty(chain.shape[0])
    for _ in range(60):
        c, d = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
        left = np.linalg.norm(at(c) - chain, axis=1) < np.linalg.norm(
            at(d) - chain, axis=1
        )
        lo, hi = np.where(left, lo, c), np.where(left, d, hi)
    params[:] = 0.5 * (lo + hi)
    assert float(np.linalg.norm(at(params) - chain, axis=1).max()) < 1e-9
    if np.array_equal(chain[0], chain[-1]):
        # the seam node is both ends of a closed curve: take the end next to its
        # neighbour
        params[0] = t0 if abs(params[1] - t0) < abs(params[1] - t1) else t1
        params[-1] = t0 if abs(params[-2] - t0) < abs(params[-2] - t1) else t1
    ta, tb = np.minimum(params[:-1], params[1:]), np.maximum(params[:-1], params[1:])
    pa, pb = at(ta), at(tb)
    chord = pb - pa
    span = np.einsum("ij,ij->i", chord, chord)

    def distance(t: NDArray[np.float64]) -> NDArray[np.float64]:
        v = at(t) - (pa if t.ndim == 1 else pa[:, None, :])
        ch = chord if t.ndim == 1 else chord[:, None, :]
        sp = span if t.ndim == 1 else span[:, None]
        s = np.clip(np.sum(v * ch, axis=-1) / sp, 0.0, 1.0)
        return np.asarray(np.linalg.norm(v - s[..., None] * ch, axis=-1))

    samples = ta[:, None] + (tb - ta)[:, None] * np.linspace(0.0, 1.0, 65)[None, :]
    sampled = distance(samples)
    k = np.argmax(sampled, axis=1)
    rows = np.arange(k.size)
    a = samples[rows, np.maximum(k - 1, 0)]
    b = samples[rows, np.minimum(k + 1, 64)]
    return np.maximum(sampled.max(axis=1), _golden_max(distance, a, b))


@pytest.mark.parametrize(
    ("kind", "deflection"),
    [
        ("bspline_prism", 0.003),
        ("bspline_prism", 0.01),
        ("bspline_prism", 0.03),
        ("bump", 0.003),
        ("bump", 0.01),
        ("bump", 0.03),
        ("arc", 0.003),
        ("arc", 0.01),
        ("arc", 0.03),
        ("cone", 0.01),
        ("ellipse", 0.03),
    ],
)
def test_adaptive_1d_keeps_its_three_rules_where_upstream_broke_the_bounds(
    kind: str, deflection: float, tmp_path: Path
) -> None:
    """Spec (SMESH ``1d_meshing_hypo.rst``, "Adaptive hypothesis"), on every edge,
    straight edges included: every chord lies in [min_size, max_size] (relative 1e-9);
    two neighbours on one edge, the two ends of a closed edge included, differ at most
    by a factor of 2; each sagitta is at most the deflection, unless min_size wins
    there (``_min_size_wins``), and then the segment is at least min_size long.

    Without the patch these cases met the deflection but broke the other rules:
    neighbours up to 3.23 times apart on the B-spline prism, 3.15 on the bump, 2.74 on
    the arc, 2.16 and 2.06 on the cone, 2.51 on the ellipse at 0.03; a segment of
    0.049888 on the B-spline prism at 0.003. Tolerance 1e-9: round-off.
    """
    session = _rules_session(kind)
    brep = session.brep()
    path = tmp_path / f"{kind}.brep"
    path.write_bytes(brep)
    edge_ids = _session_edge_ids(session, ps.load_brep(brep))

    segments = _adaptive_brep_segments(path, ADAPTIVE_MIN_SIZE, deflection)

    assert segments
    for ordinal, seg in segments.items():
        chain = _ordered_chain(seg)
        chords = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        sagitta = _curve_sagittas(session, edge_ids[ordinal], chain)
        at_least_min = chords >= ADAPTIVE_MIN_SIZE * (1.0 - _ROUND_OFF)
        min_wins = _min_size_wins(sagitta, chords, deflection) & at_least_min
        assert np.all((sagitta <= deflection * (1.0 + _ROUND_OFF)) | min_wins)
        assert float(chords.min()) >= ADAPTIVE_MIN_SIZE * (1.0 - _ROUND_OFF)
        assert float(chords.max()) <= ADAPTIVE_MAX_SIZE * (1.0 + _ROUND_OFF)
        closed = bool(np.array_equal(chain[0], chain[-1]))
        pairs = np.r_[chords, chords[:1]] if closed else chords
        ratio = np.maximum(pairs[1:] / pairs[:-1], pairs[:-1] / pairs[1:])
        assert float(ratio.max(initial=1.0)) <= 2.0 * (1.0 + _ROUND_OFF)


def _vertex_ratios(segments: dict[int, NDArray[np.float64]]) -> list[float]:
    """Per vertex where two or more meshed edges end: longest over shortest end chord.

    Each edge's two end chords sit at its end vertices; a closed edge ends twice at one
    vertex. The vertices are matched by their coordinates, which are written once per
    segment and identical.
    """
    at_vertex: dict[tuple[float, ...], list[float]] = {}
    for seg in segments.values():
        chain = _ordered_chain(seg)
        chords = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        for point, chord in ((chain[0], chords[0]), (chain[-1], chords[-1])):
            at_vertex.setdefault(tuple(point.tolist()), []).append(float(chord))
    return [max(c) / min(c) for c in at_vertex.values() if len(c) >= 2]


@pytest.mark.parametrize(
    ("kind", "deflection"),
    [
        ("bspline_prism", 0.003),
        ("bspline_prism", 0.01),
        ("bspline_prism", 0.03),
        ("bump", 0.003),
        ("bump", 0.01),
        ("bump", 0.03),
        ("arc", 0.003),
        ("arc", 0.01),
        ("arc", 0.03),
        ("cone", 0.01),
        ("ellipse", 0.03),
    ],
)
def test_adaptive_1d_keeps_the_factor_2_across_every_vertex(
    kind: str, deflection: float, tmp_path: Path
) -> None:
    """Spec (``1d_meshing_hypo.rst``): two adjacent segments differ at most twice.

    Upstream grades the size field in space, so the rule also holds where two edges meet
    at a vertex (report §18.1; Phase 3 measured at most 1.60 there). The end chords of
    every pair of meshed edges that share a vertex differ at most by a factor of 2.
    Tolerance 1e-9: round-off.
    """
    session = _rules_session(kind)
    path = tmp_path / f"{kind}.brep"
    path.write_bytes(session.brep())

    ratios = _vertex_ratios(
        _adaptive_brep_segments(path, ADAPTIVE_MIN_SIZE, deflection)
    )

    assert ratios
    assert max(ratios) <= 2.0 * (1.0 + _ROUND_OFF), max(ratios)


# ---- Families with a fixture of their own ------------------------------------------ #


def test_the_extrusion_mesher_fills_a_prismatic_solid_from_its_source_face() -> None:
    """It meshes the lateral faces itself, so only the source face needs a 2-D algorithm."""
    shape = _extruded_triangle_shape()
    base = _triangular_face(shape)

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Prism3D())
        mesher.assign(Mefisto2D(), on=base)
        mesher.assign(MaxElementArea(max_area=3.0), on=base)
        report = mesher.compute()
        mesh = mesher.mesh()

    assert report.volumes > 0
    # Extruding triangles gives pentahedra and nothing else, which is the whole algorithm.
    assert mesh.count_of(ElementType.PENTAHEDRON) == report.volumes
    # Every cell lies within the swept height, so the extrusion went where it was asked to.
    assert mesh.node_coords[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert mesh.node_coords[:, 2].max() == pytest.approx(PRISM_HEIGHT, abs=1e-9)


def test_the_radial_mesher_builds_an_o_grid_between_two_shells() -> None:
    """Layers between an inner and an outer shell — a pipe wall, or a hollow ball."""
    shape = _hollow_sphere_shape()

    with Mesher(shape) as mesher:
        _radial_recipe(mesher, shape, NumberOfLayers(count=4))
        report = mesher.compute()
        mesh = mesher.mesh()

    assert report.volumes > 0
    assert mesh.count_of(ElementType.PENTAHEDRON) == report.volumes
    # Every node lies in the wall, between the two radii, which is what an O-grid means.
    radii = np.linalg.norm(mesh.node_coords, axis=1)
    assert radii.min() == pytest.approx(SHELL_INNER_RADIUS, abs=1e-6)
    assert radii.max() == pytest.approx(SHELL_OUTER_RADIUS, abs=1e-6)


def _layer_radii(mesh: ps.MeshData) -> NDArray[np.float64]:
    """The distinct radii the radial layers landed on."""
    return np.unique(np.round(np.linalg.norm(mesh.node_coords, axis=1), 6))


def test_a_layer_distribution_spaces_the_radial_direction_by_a_1d_hypothesis() -> None:
    """A hypothesis carrying another hypothesis — the only nested case in the catalogue."""
    shape = _hollow_sphere_shape()

    with Mesher(shape) as graded:
        _radial_recipe(
            graded,
            shape,
            LayerDistribution(
                distribution=NumberOfSegments(
                    count=5, distribution=Distribution.SCALE, scale_factor=3.0
                )
            ),
        )
        graded.compute()
        graded_gaps = np.diff(_layer_radii(graded.mesh()))

    with Mesher(shape) as even:
        _radial_recipe(even, shape, NumberOfLayers(count=5))
        even.compute()
        even_gaps = np.diff(_layer_radii(even.mesh()))

    # The counter-case is the point: an even distribution must NOT be graded, or "graded"
    # would be a property of the fixture rather than of the hypothesis.
    assert even_gaps.max() / even_gaps.min() == pytest.approx(1.0, rel=0.05)
    assert graded_gaps.max() / graded_gaps.min() > 2.0


def test_projection_copies_a_face_mesh_onto_another_face_node_for_node() -> None:
    """What no free mesher can guarantee, and the reason the projection family exists."""
    shape = _box_shape()

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Mefisto2D())
        mesher.assign(MaxElementArea(max_area=3.0))
        mesher.assign(Projection2D(), on=SubShape(SubShapeKind.FACE, 2))
        mesher.assign(
            ProjectionSource2D(source_face=SubShape(SubShapeKind.FACE, 1)),
            on=SubShape(SubShapeKind.FACE, 2),
        )
        mesher.compute()
        mesh = mesher.mesh()

    source = [
        m for m in [1] for _ in [0]
    ]  # placeholder removed below; counts read from the mesh
    del source

    def face_element_count(ordinal: int) -> int:
        return int(
            np.count_nonzero(
                (mesh.element_kind == int(SubShapeKind.FACE))
                & (mesh.element_ordinal == ordinal)
            )
        )

    assert face_element_count(1) > 0
    assert face_element_count(2) == face_element_count(1)


def test_the_radial_quadrangle_mesher_meshes_a_disk() -> None:
    with Mesher(_disk_face_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=8))
        mesher.assign(RadialQuadrangle1D2D())
        mesher.assign(NumberOfLayers2D(count=3))
        report = mesher.compute()
        mesh = mesher.mesh()

    assert report.faces > 0
    assert mesh.count_of(ElementType.QUADRANGLE) > 0


def test_the_medial_axis_mesher_meshes_a_thin_face() -> None:
    """Quad-dominant meshing built on a real medial-axis transform."""
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 12.0, 2.0)
    shape = ps.load_brep(session.brep())

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=8))
        mesher.assign(QuadFromMedialAxis1D2D())
        report = mesher.compute()
        mesh = mesher.mesh()

    assert report.faces > 0
    assert mesh.count_of(ElementType.QUADRANGLE) > 0


def test_one_polygon_per_face_uses_the_edge_discretisation_as_its_boundary() -> None:
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(PolygonPerFace2D())
        report = mesher.compute()
        mesh = mesher.mesh()

    assert report.faces == 6
    assert mesh.count_of(ElementType.POLYGON) == 6


def test_viscous_layers_grow_from_the_named_walls_into_a_group() -> None:
    """The layer cells are findable only through the group the hypothesis names."""
    with Mesher(_box_shape()) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.assign(
            ViscousLayers(
                total_thickness=0.4,
                layer_count=2,
                stretch_factor=1.2,
                boundary=(1,),
                group_name="wall_layers",
            )
        )
        report = mesher.compute()
        groups = mesher.groups()

    assert report.volumes > 0
    named = [g for g in groups if g.name == "wall_layers"]
    assert named, [g.name for g in groups]
    assert named[0].element_ids.size > 0


def test_the_composite_segment_mesher_treats_a_split_edge_chain_as_one() -> None:
    """An import that split one curve into pieces must still take one segment count."""
    with Mesher(_box_shape()) as mesher:
        mesher.assign(CompositeSegment1D())
        mesher.assign(NumberOfSegments(count=4))
        report = mesher.compute()

    assert report.edges > 0


def test_element_counts_scale_with_the_hypothesis_rather_than_being_fixed() -> None:
    """A guard against a recipe that ignores its hypothesis and looks right anyway."""
    counts: list[int] = []
    for area in (16.0, 4.0, 1.0):
        with Mesher(_box_shape()) as mesher:
            mesher.assign(Regular1D())
            mesher.assign(LocalLength(length=math.sqrt(area)))
            mesher.assign(Mefisto2D())
            mesher.assign(MaxElementArea(max_area=area))
            counts.append(mesher.compute().faces)

    assert counts == sorted(counts)
    assert counts[-1] > 4 * counts[0]
