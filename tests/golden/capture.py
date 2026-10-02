# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-02

"""Capture the golden outputs that the OCCT 8.0.1 and SMESH V9_16_0 upgrade must reproduce.

The test suite asserts properties. This records *values*: topology counts, adaptive volumes
and areas, element counts by type, quality statistics, 1-D spacings, and the exact outcome of
every known defect from ``docs/reports/defect_sweep_4.2.2.md``. Each upgrade phase captures
again and :mod:`compare` diffs the two files, so a behaviour change in a dependency shows up
as a named number that moved rather than as a test that happens to still pass.

Every probe is deterministic and self-contained. A probe that raises is recorded, not
skipped: ``{"error": "<type>: <message>"}`` is a golden value too, because a changed failure
is a changed behaviour. Probes are grouped:

* ``geometry`` — OCCT through :class:`pysmesh.Session` and the stateless free functions;
* ``mesh`` — SMESH through :class:`pysmesh.Mesher` and :func:`pysmesh.compute_viscous_layers`;
* ``defect`` — the report's reproductions. These are *expected* to change in Phase 4, and
  :mod:`compare` lists them apart from the rest.

Usage (from the repository root, inside the build env):
    conda run -n flux-pysmesh-build python tests/golden/capture.py <out.json>
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final, cast

import numpy as np

_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import pysmesh as ps  # noqa: E402
from pysmesh import (  # noqa: E402
    Arithmetic1D,
    AspectRatio,
    AspectRatio3D,
    AutomaticLength,
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
    LayerDistribution,
    LocalLength,
    MaxElementArea,
    MaxLength,
    Mefisto2D,
    Mesher,
    MinimumAngle,
    NumberOfLayers,
    NumberOfLayers2D,
    NumberOfSegments,
    PolygonPerFace2D,
    PolyhedronPerSolid3D,
    Prism3D,
    Projection1D2D,
    Projection2D,
    ProjectionSource2D,
    Propagation,
    PysmeshError,
    QuadFromMedialAxis1D2D,
    Quadrangle2D,
    QuadrangleParams,
    QuadraticMesh,
    QuadType,
    RadialPrism3D,
    RadialQuadrangle1D2D,
    Regular1D,
    Session,
    StartEndLength,
    SubShape,
    SubShapeKind,
    Volume,
)
from pysmesh.mesher import ViscousLayers  # noqa: E402

Value = int | float | str | bool | None
Probe = Callable[[], dict[str, Value]]

# The project's box fixture: never a unit cube (tests/test_mesher_family.py).
_BOX: Final[tuple[float, float, float]] = (3.0, 7.0, 11.0)
_VOLUME_PRECISION: Final[float] = 1e-9
_STEP_FILE: Final[Path] = _ROOT / "test_files" / "perrinn_f1.step"

_PROBES: dict[str, tuple[str, Probe]] = {}


def probe(group: str, name: str) -> Callable[[Probe], Probe]:
    """Register a probe under ``group`` with a stable name."""

    def wrap(fn: Probe) -> Probe:
        if name in _PROBES:
            raise ValueError(f"duplicate probe name {name!r}")
        _PROBES[name] = (group, fn)
        return fn

    return wrap


# ---- Shared builders -------------------------------------------------------------------- #


def _naca(chord: float, t: float = 0.12, n: int = 60) -> np.ndarray:
    """NACA 00xx with a sharp trailing edge; closed loop TE -> upper -> LE -> lower -> TE."""
    b = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(b))
    yt = (
        5
        * t
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
    return np.vstack([upper, lower]) * chord


def _new(s: Session, kind: EntityKind, before: set[int]) -> list[int]:
    """Live ids of ``kind`` not in ``before``."""
    return [int(e) for e in s.entities(kind) if int(e) not in before]


def _ids(s: Session, kind: EntityKind) -> set[int]:
    """The live ids of one kind."""
    return {int(e) for e in s.entities(kind)}


def _wing(s: Session, two_edge: bool, ruled: bool = False, nsec: int = 3) -> list[int]:
    """NACA 0012 wing, chord 1.0 -> 0.8 over span 2.0; returns the solid ids."""
    sections = []
    for z in np.linspace(0.0, 2.0, nsec):
        p2 = _naca(1.0 - 0.1 * z)
        p = np.c_[p2, np.full(len(p2), z)]
        before = _ids(s, EntityKind.EDGE)
        if two_edge:
            k = len(p) // 2
            s.add_spline(p[: k + 1])
            s.add_spline(p[k:])
            s.make_wire(_new(s, EntityKind.EDGE, before))
        else:
            s.add_spline(p)
        sections.append(_new(s, EntityKind.EDGE, before))
    before = _ids(s, EntityKind.SOLID)
    s.thru_sections(sections, solid=True, ruled=ruled)
    return _new(s, EntityKind.SOLID, before)


def _two_boxes(
    s: Session, offset: tuple[float, float, float]
) -> tuple[list[int], list[int]]:
    """Two 2 x 2 x 2 boxes, the second at ``offset``."""
    s.add_box(2.0, 2.0, 2.0)
    a = list(_ids(s, EntityKind.SOLID))
    s.add_box(2.0, 2.0, 2.0, origin=offset)
    return a, _new(s, EntityKind.SOLID, set(a))


def _geo(s: Session) -> dict[str, Value]:
    """Topology from the BREP (not ids, which alias), adaptive measures, validity."""
    brep = s.brep()
    shape = ps.load_brep(brep)
    out: dict[str, Value] = {
        "solids": len(shape.solids()),
        "faces": len(shape.faces()),
        "edges": len(shape.edges()),
        "vertices": len(shape.vertices()),
        "free_edges": int(len(ps.free_boundary_edges(brep))),
        "solid_ids": len(_ids(s, EntityKind.SOLID)),
        "face_ids": len(_ids(s, EntityKind.FACE)),
    }
    # Measured on a fresh session read from the BREP: there every shape carries exactly one
    # id. Summing over the live session's ids would count an aliased shape once per id
    # (report C5) - a common() of two boxes would read twice its volume.
    fresh = Session()
    fresh.add_brep(brep)
    solids = sorted(_ids(fresh, EntityKind.SOLID))
    faces = sorted(_ids(fresh, EntityKind.FACE))
    if solids:
        out["volume"] = float(
            fresh.mass_properties(solids, precision=_VOLUME_PRECISION).measure.sum()
        )
    if faces:
        out["area"] = float(
            fresh.mass_properties(faces, precision=_VOLUME_PRECISION).measure.sum()
        )
    return out


def _guard(fn: Callable[[], dict[str, Value]]) -> dict[str, Value]:
    """Run one probe body; a raise is recorded as the probe's value."""
    try:
        return fn()
    except PysmeshError as e:
        first = str(e).splitlines()[0][:240]
        details = str(getattr(e, "details", "") or "").replace("\n", " | ")[:400]
        return {"error": f"PysmeshError: {first}", "details": details}
    # A non-PysmeshError escape (report A2) is itself a golden value, so it is caught too.
    except Exception as e:  # noqa: BLE001
        first = str(e).splitlines()[0][:240] if str(e) else ""
        return {"error": f"{type(e).__name__}: {first}"}


def _box_shape() -> ps.Shape:
    """The 3 x 7 x 11 box as a stateless shape."""
    s = Session()
    s.add_box(*_BOX)
    return ps.load_brep(s.brep())


def _mesh_stats(m: Mesher) -> dict[str, Value]:
    """Counts by element type and quality statistics of a computed mesher."""
    report = m.compute()
    md = m.mesh()
    out: dict[str, Value] = {
        "nodes": report.nodes,
        "edges": report.edges,
        "faces": report.faces,
        "volumes": report.volumes,
    }
    for t in ElementType:
        c = md.count_of(t)
        if c:
            out[f"n_{t.name}"] = int(c)
    for label, control in (
        ("min_angle", MinimumAngle()),
        ("aspect2d", AspectRatio()),
        ("aspect3d", AspectRatio3D()),
        ("cell_volume", Volume()),
    ):
        q = m.quality(control)
        if q.count:
            out[f"{label}_min"] = float(q.values.min())
            out[f"{label}_max"] = float(q.values.max())
            out[f"{label}_sum"] = float(q.values.sum())
    return out


def _segments(md: ps.MeshData, edge_ordinal: int) -> np.ndarray:
    """Segment lengths on one edge, ordered by the midpoint's distance from the origin."""
    rows = []
    for e in range(md.element_count):
        if int(md.element_kind[e]) != int(SubShapeKind.EDGE):
            continue
        if int(md.element_ordinal[e]) != edge_ordinal:
            continue
        n = md.nodes_of(e)
        if len(n) != 2:
            continue
        a, b = md.node_coords[n[0]], md.node_coords[n[1]]
        rows.append(
            (float(np.linalg.norm((a + b) / 2.0)), float(np.linalg.norm(b - a)))
        )
    return np.array([length for _, length in sorted(rows)], dtype=np.float64)


def _spacing(
    hyp: ps.Hypothesis, shape: ps.Shape | None = None, edge: int = 1
) -> dict[str, Value]:
    """1-D discretisation of one edge under one hypothesis."""
    with Mesher(shape or _box_shape()) as m:
        m.assign(Regular1D())
        m.assign(hyp)
        m.compute()
        seg = _segments(m.mesh(), edge)
    return {
        "segments": int(seg.size),
        "first": float(seg[0]),
        "last": float(seg[-1]),
        "min": float(seg.min()),
        "max": float(seg.max()),
        "sum": float(seg.sum()),
    }


# ---- Geometry: OCCT --------------------------------------------------------------------- #


def _primitive(build: Callable[[Session], object]) -> Probe:
    def run() -> dict[str, Value]:
        s = Session()
        build(s)
        return _geo(s)

    return run


for _name, _build in {
    "box": lambda s: s.add_box(*_BOX),
    "cylinder": lambda s: s.add_cylinder(1.5, 4.0),
    "cone": lambda s: s.add_cone(2.0, 0.5, 3.0),
    "sphere": lambda s: s.add_sphere(1.7),
    "torus": lambda s: s.add_torus(3.0, 0.8),
    "wedge": lambda s: s.add_wedge(3.0, 2.0, 4.0, 1.0),
}.items():
    probe("geometry", f"primitive/{_name}")(_primitive(_build))


def _boolean(op: str, offset: tuple[float, float, float]) -> Probe:
    def run() -> dict[str, Value]:
        s = Session()
        a, b = _two_boxes(s, offset)
        if op == "fragment":
            s.fragment(a + b)
        else:
            getattr(s, op)(a, b)
        return _geo(s)

    return run


for _op in ("fuse", "cut", "common", "fragment", "split"):
    probe("geometry", f"boolean/{_op}/overlap")(_boolean(_op, (1.0, 1.0, 1.0)))
    probe("geometry", f"boolean/{_op}/coplanar")(_boolean(_op, (1.0, 0.0, 0.0)))


@probe("geometry", "boolean/box_cut_cylinder")
def _box_cut_cylinder() -> dict[str, Value]:
    s = Session()
    s.add_box(4.0, 4.0, 2.0)
    box = list(_ids(s, EntityKind.SOLID))
    s.add_cylinder(1.0, 4.0, origin=(2.0, 2.0, -1.0))
    s.cut(box, _new(s, EntityKind.SOLID, set(box)))
    return _geo(s)


@probe("geometry", "boolean/sphere_common_box")
def _sphere_common_box() -> dict[str, Value]:
    s = Session()
    s.add_sphere(1.0)
    a = list(_ids(s, EntityKind.SOLID))
    s.add_box(2.0, 2.0, 2.0, origin=(0.2, 0.3, -1.0))
    s.common(a, _new(s, EntityKind.SOLID, set(a)))
    return _geo(s)


@probe("geometry", "boolean/section_edges")
def _section() -> dict[str, Value]:
    s = Session()
    a, b = _two_boxes(s, (1.0, 1.0, 1.0))
    s.section(a, b)
    return _geo(s)


@probe("geometry", "fillet/box_edge")
def _fillet() -> dict[str, Value]:
    s = Session()
    s.add_box(*_BOX)
    s.fillet([min(_ids(s, EntityKind.EDGE))], 0.5)
    return _geo(s)


@probe("geometry", "chamfer/box_edge")
def _chamfer() -> dict[str, Value]:
    s = Session()
    s.add_box(*_BOX)
    s.chamfer([min(_ids(s, EntityKind.EDGE))], 0.4)
    return _geo(s)


for _two in (False, True):
    for _ruled in (False, True):
        _sections = "2edge" if _two else "1edge"
        _tag = f"loft/wing_{_sections}_{'ruled' if _ruled else 'smooth'}"

        def _loft(two: bool = _two, ruled: bool = _ruled) -> dict[str, Value]:
            s = Session()
            _wing(s, two, ruled)
            return _geo(s)

        probe("geometry", _tag)(_loft)


@probe("geometry", "loft/circles_ruled_3")
def _loft_circles() -> dict[str, Value]:
    s = Session()
    secs = []
    for i, z in enumerate((0.0, 1.0, 2.0)):
        before = _ids(s, EntityKind.EDGE)
        s.add_circle((0.0, 0.0, z), (0.0, 0.0, 1.0), 1.0 + 0.2 * i)
        secs.append(_new(s, EntityKind.EDGE, before))
    s.thru_sections(secs, solid=True, ruled=True)
    return _geo(s)


@probe("geometry", "sweep/pipe_shell_spline")
def _pipe_shell() -> dict[str, Value]:
    s = Session()
    spine = np.array(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 3.0), (0.0, 2.0, 6.0), (2.0, 2.0, 9.0)]
    )
    before = _ids(s, EntityKind.EDGE)
    s.add_spline(spine)
    sp = _new(s, EntityKind.EDGE, before)
    before = _ids(s, EntityKind.EDGE)
    d = np.array([1.0, 0.0, 3.0]) / math.sqrt(10.0)
    s.add_circle((0.0, 0.0, 0.0), tuple(d), 0.5)
    s.pipe_shell(sp, _new(s, EntityKind.EDGE, before), frenet=False, solid=True)
    return _geo(s)


@probe("geometry", "sweep/revolve_rectangle")
def _revolve() -> dict[str, Value]:
    s = Session()
    s.add_polyline(
        np.array([(1.0, 0, 0), (2.0, 0, 0), (2.0, 0, 1.0), (1.0, 0, 1.0)]), closed=True
    )
    s.make_face(sorted(_ids(s, EntityKind.EDGE)))
    s.revolve(
        sorted(_ids(s, EntityKind.FACE)), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi
    )
    return _geo(s)


@probe("geometry", "offset/thick_solid_box")
def _thick() -> dict[str, Value]:
    s = Session()
    s.add_box(*_BOX)
    s.make_thick_solid([min(_ids(s, EntityKind.FACE))], -0.3)
    return _geo(s)


@probe("geometry", "offset/sphere")
def _offset_sphere() -> dict[str, Value]:
    s = Session()
    s.add_sphere(1.7)
    s.offset(sorted(_ids(s, EntityKind.SOLID)), 0.25)
    return _geo(s)


@probe("geometry", "sew/six_box_faces")
def _sew() -> dict[str, Value]:
    loose = Session()
    for loop in (
        [(0, 0, 0), (3, 0, 0), (3, 7, 0), (0, 7, 0)],
        [(0, 0, 11), (0, 7, 11), (3, 7, 11), (3, 0, 11)],
        [(0, 0, 0), (0, 7, 0), (0, 7, 11), (0, 0, 11)],
        [(3, 0, 0), (3, 0, 11), (3, 7, 11), (3, 7, 0)],
        [(0, 0, 0), (0, 0, 11), (3, 0, 11), (3, 0, 0)],
        [(0, 7, 0), (3, 7, 0), (3, 7, 11), (0, 7, 11)],
    ):
        before = _ids(loose, EntityKind.EDGE)
        loose.add_polyline(np.array(loop, dtype=np.float64), closed=True)
        loose.make_face(_new(loose, EntityKind.EDGE, before))
    loose.sew(sorted(_ids(loose, EntityKind.FACE)), tolerance=1e-6, make_solid=True)
    return _geo(loose)


@probe("geometry", "unify/split_box_fixture")
def _unify() -> dict[str, Value]:
    s = Session()
    s.add_brep((_ROOT / "tests" / "fixtures" / "split_box.brep").read_bytes())
    s.unify_same_domain()
    return _geo(s)


@probe("geometry", "heal/split_box_fixture")
def _heal() -> dict[str, Value]:
    s = Session()
    s.add_brep((_ROOT / "tests" / "fixtures" / "split_box.brep").read_bytes())
    s.heal(sorted(_ids(s, EntityKind.SOLID)))
    return _geo(s)


@probe("geometry", "defeature/box_hole")
def _defeature() -> dict[str, Value]:
    s = Session()
    s.add_box(4.0, 4.0, 2.0)
    box = list(_ids(s, EntityKind.SOLID))
    s.add_cylinder(1.0, 4.0, origin=(2.0, 2.0, -1.0))
    s.cut(box, _new(s, EntityKind.SOLID, set(box)))
    table = s.entity_table(EntityKind.FACE)
    types = s.entity_types(EntityKind.FACE)
    cyl = [
        int(i)
        for i, t in zip(types.ids, types.types, strict=True)
        if str(t).lower().startswith("cyl")
    ]
    del table
    s.defeature(cyl)
    return _geo(s)


@probe("geometry", "tessellate/session")
def _tess() -> dict[str, Value]:
    out: dict[str, Value] = {}
    for label, build in (
        ("sphere", lambda s: s.add_sphere(1.7)),
        ("torus", lambda s: s.add_torus(3.0, 0.8)),
        ("wing", lambda s: _wing(s, True)),
    ):
        s = Session()
        build(s)
        mesh = s.tessellate(deflection=1e-3, incremental=False)
        out[f"{label}_tris"] = int(len(mesh.tris))
        out[f"{label}_nodes"] = int(len(mesh.nodes))
    return out


@probe("geometry", "tessellate/stateless")
def _tess_free() -> dict[str, Value]:
    s = Session()
    s.add_sphere(1.7)
    r = ps.tessellate(s.brep(), ps.TessellateParams(lin_defl=1e-3))
    return {"tris": int(len(r.tris)), "nodes": int(len(r.nodes))}


@probe("geometry", "query/bbox_and_distance")
def _query() -> dict[str, Value]:
    s = Session()
    s.add_box(*_BOX)
    s.add_sphere(1.0, centre=(6.0, 3.5, 5.5))
    solids = sorted(_ids(s, EntityKind.SOLID))
    bb = s.bounding_boxes(EntityKind.SOLID).bbox
    d = s.distance(solids[0], solids[1])
    inside = s.contains(solids[:1], np.array([[1.5, 3.5, 5.5], [10.0, 0.0, 0.0]]))
    return {
        "bbox0_xmin": float(bb[0, 0]),
        "bbox0_xmax": float(bb[0, 3]),
        "bbox1_xmin": float(bb[1, 0]),
        "distance": float(d.distance),
        "contains_in": bool(inside[0][0]),
        "contains_out": bool(inside[0][1]),
    }


@probe("geometry", "exchange/step_round_trip")
def _step() -> dict[str, Value]:
    s = Session()
    s.add_box(*_BOX)
    back = ps.read_step_xde(
        ps.write_step_xde(s.brep(), unit="M", face_names={1: "inlet"})
    )
    s2 = Session()
    s2.add_brep(back.brep)
    out = _geo(s2)
    out["face_labels"] = len(back.face_labels)
    out["unit"] = back.unit_name
    return out


@probe("geometry", "exchange/iges_round_trip")
def _iges() -> dict[str, Value]:
    s = Session()
    s.add_box(*_BOX)
    # read_iges takes a path, not bytes (write_iges returns bytes), so round-trip via a file.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "box.igs"
        path.write_bytes(ps.write_iges(s.brep(), unit="M"))
        back = ps.read_iges(path)
    s2 = Session()
    s2.add_brep(back.brep)
    return _geo(s2)


@probe("geometry", "exchange/production_step")
def _production() -> dict[str, Value]:
    if not _STEP_FILE.is_file():
        return {"skipped": "test_files/perrinn_f1.step not present"}
    imp = ps.read_step_xde(str(_STEP_FILE))
    shape = ps.load_brep(imp.brep)
    return {
        "solids": len(shape.solids()),
        "faces": len(shape.faces()),
        "edges": len(shape.edges()),
        "volume_fixed_rule": float(sum(x.volume for x in shape.solids())),
        "face_labels": len(imp.face_labels),
        "solid_labels": len(imp.solid_labels),
        "tris_defl_1mm": int(
            len(ps.tessellate(imp.brep, ps.TessellateParams(lin_defl=1e-3)).tris)
        ),
    }


# ---- Mesh: SMESH ------------------------------------------------------------------------ #


for _label, _hyp in {
    "regular_5": NumberOfSegments(count=5),
    "scale_5_x3": NumberOfSegments(
        count=5, distribution=Distribution.SCALE, scale_factor=3.0
    ),
    "table_5": NumberOfSegments(
        count=5, distribution=Distribution.TABLE, table=(0.0, 1.0, 1.0, 3.0)
    ),
    "expression_5": NumberOfSegments(
        count=5, distribution=Distribution.EXPRESSION, expression="1+t*t"
    ),
    "arithmetic": Arithmetic1D(start_length=0.5, end_length=2.0),
    "start_end": StartEndLength(start_length=0.5, end_length=2.0),
    "geometric": Geometric1D(start_length=0.5, common_ratio=1.2),
    "fixed_points": FixedPoints1D(points=(0.25, 0.75), segment_counts=(2, 3, 2)),
    "local_length": LocalLength(length=1.5),
    "max_length": MaxLength(length=2.0),
    "automatic": AutomaticLength(fineness=0.5),
}.items():
    probe("mesh", f"1d/{_label}")(lambda hyp=_hyp: _spacing(hyp))


def _cylinder_shape() -> ps.Shape:
    """A radius-1.5, height-4 cylinder; edge 1 is a circle."""
    s = Session()
    s.add_cylinder(1.5, 4.0)
    return ps.load_brep(s.brep())


@probe("mesh", "1d/deflection_circle")
def _deflection() -> dict[str, Value]:
    shape = _cylinder_shape()
    circle = max(range(len(shape.edges())), key=lambda i: shape.edges()[i].length) + 1
    return _spacing(Deflection1D(deflection=0.01), shape, circle)


@probe("mesh", "1d/adaptive_circle")
def _adaptive() -> dict[str, Value]:
    shape = _cylinder_shape()
    circle = max(range(len(shape.edges())), key=lambda i: shape.edges()[i].length) + 1
    return _spacing(
        ps.Adaptive1D(min_size=0.05, max_size=1.0, deflection=0.01), shape, circle
    )


@probe("mesh", "1d/composite_segment")
def _composite() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(CompositeSegment1D())
        m.assign(NumberOfSegments(count=4))
        return _mesh_stats(m)


@probe("mesh", "3d/hexa_structured_box")
def _hexa() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=4))
        m.assign(Quadrangle2D())
        m.assign(Hexa3D())
        return _mesh_stats(m)


@probe("mesh", "3d/hexa_quadratic_box")
def _hexa_quadratic() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=2))
        m.assign(QuadraticMesh())
        m.assign(Quadrangle2D())
        m.assign(Hexa3D())
        return _mesh_stats(m)


@probe("mesh", "3d/hexa_propagation")
def _propagation() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=3))
        m.assign(NumberOfSegments(count=7), on=SubShape(SubShapeKind.EDGE, 1))
        m.assign(Propagation(), on=SubShape(SubShapeKind.EDGE, 1))
        m.assign(Quadrangle2D())
        m.assign(Hexa3D())
        return _mesh_stats(m)


@probe("mesh", "3d/composite_hexa_box")
def _composite_hexa() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=3))
        m.assign(Quadrangle2D())
        m.assign(CompositeHexa3D())
        return _mesh_stats(m)


@probe("mesh", "3d/hexa_from_skin_refuses_geometry")
def _skin() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=3))
        m.assign(Quadrangle2D())
        m.assign(HexaFromSkin3D())
        return _mesh_stats(m)


@probe("mesh", "2d/mefisto_box")
def _mefisto_box() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(LocalLength(length=1.0))
        m.assign(Mefisto2D())
        m.assign(MaxElementArea(max_area=1.0))
        return _mesh_stats(m)


@probe("mesh", "2d/mefisto_sphere")
def _mefisto_sphere() -> dict[str, Value]:
    s = Session()
    s.add_sphere(1.7)
    with Mesher(ps.load_brep(s.brep())) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=12))
        m.assign(Mefisto2D())
        m.assign(MaxElementArea(max_area=0.1))
        return _mesh_stats(m)


@probe("mesh", "2d/polygon_per_face_and_polyhedron")
def _poly() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=3))
        m.assign(PolygonPerFace2D())
        m.assign(PolyhedronPerSolid3D())
        return _mesh_stats(m)


@probe("mesh", "2d/radial_quadrangle_disk")
def _radial_quad() -> dict[str, Value]:
    s = Session()
    s.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.5)
    s.make_face(sorted(_ids(s, EntityKind.EDGE)))
    with Mesher(ps.load_brep(s.brep())) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=8))
        m.assign(RadialQuadrangle1D2D())
        m.assign(NumberOfLayers2D(count=3))
        return _mesh_stats(m)


@probe("mesh", "2d/quad_from_medial_axis_strip")
def _medial_mesh() -> dict[str, Value]:
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 12.0, 2.0)
    with Mesher(ps.load_brep(s.brep())) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=8))
        m.assign(QuadFromMedialAxis1D2D())
        return _mesh_stats(m)


@probe("mesh", "2d/projection_opposite_faces")
def _projection() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=3))
        m.assign(Mefisto2D())
        m.assign(MaxElementArea(max_area=3.0))
        m.assign(Projection2D(), on=SubShape(SubShapeKind.FACE, 2))
        m.assign(
            ProjectionSource2D(source_face=SubShape(SubShapeKind.FACE, 1)),
            on=SubShape(SubShapeKind.FACE, 2),
        )
        return _mesh_stats(m)


@probe("mesh", "2d/projection_1d2d_opposite_faces")
def _projection_1d2d() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=3))
        m.assign(Mefisto2D())
        m.assign(MaxElementArea(max_area=3.0))
        m.assign(Projection1D2D(), on=SubShape(SubShapeKind.FACE, 2))
        m.assign(
            ProjectionSource2D(source_face=SubShape(SubShapeKind.FACE, 1)),
            on=SubShape(SubShapeKind.FACE, 2),
        )
        return _mesh_stats(m)


@probe("mesh", "3d/prism_extruded_triangle")
def _prism() -> dict[str, Value]:
    s = Session()
    s.add_polyline(np.array([(0.0, 0, 0), (6.0, 0, 0), (0.0, 5.0, 0), (0.0, 0, 0)]))
    s.make_face(sorted(_ids(s, EntityKind.EDGE)))
    s.extrude(sorted(_ids(s, EntityKind.FACE)), (0.0, 0.0, 9.0))
    shape = ps.load_brep(s.brep())
    base = next(
        i
        for i, f in enumerate(shape.faces(), 1)
        if math.isclose(f.area, 15.0, rel_tol=1e-9)
    )
    with Mesher(shape) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=3))
        m.assign(Prism3D())
        m.assign(Mefisto2D(), on=SubShape(SubShapeKind.FACE, base))
        m.assign(MaxElementArea(max_area=3.0), on=SubShape(SubShapeKind.FACE, base))
        return _mesh_stats(m)


def _hollow_sphere() -> tuple[ps.Shape, SubShape, SubShape]:
    """Solid between concentric shells r 3 and 2; returns (shape, outer, inner)."""
    s = Session()
    s.add_sphere(3.0)
    outer = list(_ids(s, EntityKind.SOLID))
    s.add_sphere(2.0)
    s.cut(outer, _new(s, EntityKind.SOLID, set(outer)))
    shape = ps.load_brep(s.brep())
    areas = sorted((f.area, i) for i, f in enumerate(shape.faces(), 1))
    return (
        shape,
        SubShape(SubShapeKind.FACE, areas[-1][1]),
        SubShape(SubShapeKind.FACE, areas[0][1]),
    )


def _radial(layers: ps.Hypothesis) -> dict[str, Value]:
    shape, outer, inner = _hollow_sphere()
    with Mesher(shape) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=6))
        m.assign(Mefisto2D(), on=outer)
        m.assign(MaxElementArea(max_area=2.0), on=outer)
        m.assign(Projection2D(), on=inner)
        m.assign(ProjectionSource2D(source_face=outer), on=inner)
        m.assign(RadialPrism3D())
        m.assign(layers)
        return _mesh_stats(m)


probe("mesh", "3d/radial_prism_layers")(lambda: _radial(NumberOfLayers(count=4)))
probe("mesh", "3d/radial_prism_distribution")(
    lambda: _radial(
        LayerDistribution(
            distribution=NumberOfSegments(
                count=5, distribution=Distribution.SCALE, scale_factor=3.0
            )
        )
    )
)


@probe("mesh", "3d/cartesian_sphere")
def _cartesian() -> dict[str, Value]:
    s = Session()
    s.add_sphere(1.7)
    with Mesher(ps.load_brep(s.brep())) as m:
        m.assign(Cartesian3D())
        m.assign(
            CartesianParameters3D(spacing_x="0.4", spacing_y="0.4", spacing_z="0.4")
        )
        return _mesh_stats(m)


@probe("mesh", "3d/viscous_layers_hexa_box")
def _viscous_hyp() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=3))
        m.assign(Quadrangle2D())
        m.assign(Hexa3D())
        m.assign(
            ViscousLayers(
                total_thickness=0.4,
                layer_count=2,
                stretch_factor=1.2,
                boundary=(1,),
                group_name="wall",
            )
        )
        out = _mesh_stats(m)
        out["group_elements"] = int(
            next(g for g in m.groups() if g.name == "wall").element_ids.size
        )
        return out


@probe("mesh", "viscous/compute_viscous_layers_box_fixture")
def _viscous_free() -> dict[str, Value]:
    fixtures = _ROOT / "tests" / "fixtures"
    shape = ps.load_brep((fixtures / "box.brep").read_bytes())
    m = {p.stem: np.load(p) for p in (fixtures / "box_mesh").glob("*.npy")}
    mesh = ps.Mesh(shape)
    sid = mesh.add_nodes(m["nodes"])
    for fid in np.unique(m["face_ids"]):
        sel = m["face_ids"] == fid
        mesh.classify_on_face(sid[m["face_node_ids"][sel]], int(fid), m["face_uv"][sel])
    for eid in np.unique(m["edge_ids"]):
        sel = m["edge_ids"] == eid
        mesh.classify_on_edge(sid[m["edge_node_ids"][sel]], int(eid), m["edge_t"][sel])
    for nl, vid in zip(m["vertex_node_ids"], m["vertex_ids"], strict=True):
        mesh.classify_on_vertex(int(sid[nl]), int(vid))
    for eid in np.unique(m["segment_edge_ids"]):
        sel = m["segment_edge_ids"] == eid
        mesh.add_segments(sid[m["segments"][sel]].astype(np.int64), int(eid))
    for fid in np.unique(m["tri_face_ids"]):
        sel = m["tri_face_ids"] == fid
        mesh.add_triangles(sid[m["tris"][sel]].astype(np.int64), int(fid))
    mesh.validate()
    res = ps.compute_viscous_layers(
        mesh,
        ps.VLParams(
            face_ids=tuple(f.id for f in shape.faces()),
            total_thickness=0.05,
            n_layers=3,
            stretch_factor=1.2,
        ),
    )
    p = res.node_coords[res.prism_connectivity]
    lateral = np.linalg.norm(p[:, 0, :] - p[:, 3, :], axis=1)
    return {
        "prisms": int(res.prism_connectivity.shape[0]),
        "nodes": int(res.node_coords.shape[0]),
        "inner_tris": int(res.inner_surface_tris.shape[0]),
        "failed_faces": len(res.failed_face_ids),
        "warnings": len(res.warnings),
        "layer_height_min": float(lateral.min()),
        "layer_height_max": float(lateral.max()),
    }


@probe("mesh", "edit/smooth_merge_quad_to_tri")
def _edit() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(LocalLength(length=1.0))
        m.assign(Mefisto2D())
        m.assign(MaxElementArea(max_area=1.0))
        m.compute()
        before = m.quality(MinimumAngle())
        m.smooth(iterations=5)
        after = m.quality(MinimumAngle())
        merged = m.merge_nodes(tolerance=1e-7)
        return {
            "min_angle_before": float(before.values.min()),
            "min_angle_after_smooth": float(after.values.min()),
            "merge_nodes_report": str(merged),
        }


@probe("mesh", "medial/rectangle_axis")
def _medial_axis() -> dict[str, Value]:
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 12.0, 2.0)
    ma = ps.medial_axis(ps.load_brep(s.brep()), 1, 0.1)
    return {
        "branches": len(ma.branches),
        "branch_points": ma.branch_points,
        "boundary_edges": ma.boundary_edges,
    }


# ---- Known defects (docs/reports/defect_sweep_4.2.2.md): expected to change in Phase 4 ---- #


def _geometric_table(h0: float, n: int, length: float) -> tuple[float, ...]:
    """Density table 1/h at every node of a geometric target (report S1)."""
    lo, hi = 1.0 + 1e-12, 3.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if h0 * (mid**n - 1) / (mid - 1) < length else (lo, mid)
    x = (
        np.concatenate([[0.0], np.cumsum(h0 * (0.5 * (lo + hi)) ** np.arange(n))])
        / length
    )
    x[-1] = 1.0
    h = np.diff(x)
    d = 1.0 / np.concatenate([[h[0]], 0.5 * (h[1:] + h[:-1]), [h[-1]]])
    return tuple(np.c_[x, d].ravel())


def _line_spacing(hyp: ps.Hypothesis, length: float = 15.0) -> dict[str, Value]:
    s = Session()
    s.add_line((0.0, 0.0, 0.0), (length, 0.0, 0.0))
    with Mesher(ps.load_brep(s.brep())) as m:
        m.assign(Regular1D())
        m.assign(hyp)
        m.compute()
        x = np.sort(m.mesh().node_coords[:, 0])
    return {
        "nodes": int(x.size),
        "first_cell": float(x[1] - x[0]),
        "last_cell": float(x[-1] - x[-2]),
    }


for _h0 in (3e-6, 3e-4):
    probe("defect", f"S1/table_15m_first_cell_{_h0:g}")(
        lambda h0=_h0: _line_spacing(
            NumberOfSegments(
                count=100,
                distribution=Distribution.TABLE,
                table=_geometric_table(h0, 100, 15.0),
            )
        )
    )
probe("defect", "S2/expression_15m_1_over_0.0003_plus_t")(
    lambda: _line_spacing(
        NumberOfSegments(
            count=100, distribution=Distribution.EXPRESSION, expression="1/(0.0003+t)"
        )
    )
)


@probe("defect", "O1/ruled_loft_4_circles")
def _o1() -> dict[str, Value]:
    s = Session()
    secs = []
    for i, z in enumerate(np.linspace(0.0, 3.0, 4)):
        before = _ids(s, EntityKind.EDGE)
        s.add_circle((0.0, 0.0, float(z)), (0.0, 0.0, 1.0), 1.0 + 0.1 * i)
        secs.append(_new(s, EntityKind.EDGE, before))
    brep = s.brep()
    out = _guard(lambda: (s.thru_sections(secs, solid=False, ruled=True), _geo(s))[1])
    out["A3_session_brep_changed"] = s.brep() != brep
    return out


@probe("defect", "B1/prism_unequal_edge_counts")
def _b1() -> dict[str, Value]:
    t = np.linspace(0.0, 2 * np.pi, 6)[:-1]
    bottom = np.c_[np.cos(t), np.sin(t), np.zeros(5)]
    top = bottom + (0.0, 0.0, 1.0)
    mid = 0.5 * (bottom[0] + bottom[1])
    loops = [
        np.vstack([bottom[:1], mid, bottom[1:]]),
        top[::-1],
        np.array([bottom[0], mid, bottom[1], top[1], top[0]]),
    ]
    for i in range(1, 5):
        j = (i + 1) % 5
        loops.append(np.array([bottom[i], bottom[j], top[j], top[i]]))
    s = Session()
    for loop in loops:
        before = _ids(s, EntityKind.EDGE)
        s.add_polyline(loop, closed=True)
        s.make_face(_new(s, EntityKind.EDGE, before))
    s.sew(sorted(_ids(s, EntityKind.FACE)), make_solid=True)
    shape = ps.load_brep(s.brep())
    half = 0.5 * 2 * math.sin(math.pi / 5)
    halves = [e.id for e in shape.edges() if abs(e.length - half) < 1e-9]
    with Mesher(shape) as m:
        m.assign(Regular1D())
        m.assign(NumberOfSegments(count=4))
        for e in halves:
            m.assign(NumberOfSegments(count=2), on=SubShape(SubShapeKind.EDGE, e))
        m.assign(Quadrangle2D())
        m.assign(Prism3D())
        return _guard(lambda: _mesh_stats(m))


@probe("defect", "A1/reduced_quad_warning")
def _a1() -> dict[str, Value]:
    p2 = _naca(1.25, n=80)
    p = np.c_[p2, np.zeros(len(p2))]
    k = len(p) // 2
    q = k // 2
    s = Session()
    for a, b in ((0, q), (q, k), (k, k + q), (k + q, len(p) - 1)):
        s.add_spline(p[a : b + 1])
    s.make_face(sorted(_ids(s, EntityKind.EDGE)))
    shape = ps.load_brep(s.brep())
    with Mesher(shape) as m:
        m.assign(Regular1D())
        for e in shape.edges():
            n = 30 if e.bbox[3] > 1.2 else 20
            m.assign(NumberOfSegments(count=n), on=SubShape(SubShapeKind.EDGE, e.id))
        m.assign(Quadrangle2D())
        m.assign(QuadrangleParams(quad_type=QuadType.REDUCED))
        out = _guard(lambda: _mesh_stats(m))
        out["kept_faces"] = int(
            m.mesh().count_of(ElementType.QUADRANGLE)
            + m.mesh().count_of(ElementType.TRIANGLE)
        )
        return out


@probe("defect", "A6/missing_hypothesis_message")
def _a6() -> dict[str, Value]:
    with Mesher(_box_shape()) as m:
        m.assign(Regular1D())
        m.assign(Quadrangle2D())
        m.assign(Hexa3D())
        return _guard(lambda: _mesh_stats(m))


@probe("defect", "V1/sphere_near_coincident_grid")
def _v1() -> dict[str, Value]:
    fuzz = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
    offsets = (
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
    empty = invalid = 0
    for d in offsets:
        for fz in fuzz:
            s = Session()
            s.add_sphere(1.0)
            a = list(_ids(s, EntityKind.SOLID))
            s.add_box(3.0, 3.0, 3.0, origin=(-1.5, d, -1.5))
            try:
                s.common(a, _new(s, EntityKind.SOLID, set(a)), fuzzy=fz)
                empty += not _ids(s, EntityKind.SOLID)
            except PysmeshError:
                invalid += 1
    return {"empty": empty, "invalid": invalid, "cells": len(fuzz) * len(offsets)}


@probe("defect", "V2/wing_cut_free_edges")
def _v2() -> dict[str, Value]:
    s = Session()
    w = _wing(s, False)
    s.add_box(3.0, 3.0, 3.0, origin=(-1.5, 1e-4, -1.5))
    s.cut(w, _new(s, EntityKind.SOLID, set(w)), fuzzy=1e-4)
    return _geo(s)


@probe("defect", "V3/inside_out_box_import")
def _v3() -> dict[str, Value]:
    src = Session()
    src.add_box(1.0, 1.0, 1.0)
    text = src.brep().decode()
    k = text.rindex("So\n\n0100000\n+")
    bad = (text[:k] + text[k:].replace("\n+", "\n-", 1)).encode()
    s = Session()
    d = s.add_brep(bad)
    solids = sorted(_ids(s, EntityKind.SOLID))
    return {"valid": d.valid, "volume": float(s.mass_properties(solids).measure[0])}


@probe("defect", "D1_D2/bbox_padding_and_spline")
def _d12() -> dict[str, Value]:
    s = Session()
    s.add_line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    p = _naca(2.0)
    s.add_spline(np.c_[p, np.zeros(len(p))] + (5.0, 0.0, 0.0))
    bb = s.bounding_boxes(EntityKind.EDGE).bbox
    return {"line_xmin": float(bb[0, 0]), "spline_xmin": float(bb[1, 0])}


@probe("defect", "D3/fixed_rule_wing_volume")
def _d3() -> dict[str, Value]:
    s = Session()
    w = _wing(s, False)
    return {
        "fixed": float(s.mass_properties(w).measure[0]),
        "adaptive": float(s.mass_properties(w, precision=1e-7).measure[0]),
    }


@probe("defect", "O4/bowed_end_section")
def _o4() -> dict[str, Value]:
    s = Session()
    secs = []
    for k, z in enumerate((0.0, 1.0, 2.0)):
        p2 = _naca(1.0)
        bow = 1e-5 if k == 0 else 0.0
        p = np.c_[p2, z + bow * np.sin(np.pi * p2[:, 0])]
        before = _ids(s, EntityKind.EDGE)
        h = len(p) // 2
        s.add_spline(p[: h + 1])
        s.add_spline(p[h:])
        s.make_wire(_new(s, EntityKind.EDGE, before))
        secs.append(_new(s, EntityKind.EDGE, before))
    return _guard(lambda: (s.thru_sections(secs, solid=True), _geo(s))[1])


@probe("defect", "C5/common_alias_ids")
def _c5() -> dict[str, Value]:
    s = Session()
    s.add_cylinder(1.0, 1.0)
    a = list(_ids(s, EntityKind.SOLID))
    s.add_box(2.0, 2.0, 1.0)
    s.common(a, _new(s, EntityKind.SOLID, set(a)))
    return _geo(s)


@probe("defect", "C2/export_handoff_after_common")
def _c2() -> dict[str, Value]:
    s = Session()
    a, b = _two_boxes(s, (1.0, 1.0, 1.0))
    s.common(a, b)
    return _guard(lambda: {"solid_ids": int(s.export_handoff().solid_id.size)})


# ---- Driver ----------------------------------------------------------------------------- #


def _meta() -> dict[str, Value]:
    """Provenance of the capture."""
    bi = ps._build_info
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=9", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"
    return {
        "git_head": head,
        "core_git_sha": bi.GIT_SHA,
        "occt": bi.OCCT_VERSION,
        "vtk": bi.VTK_VERSION,
        "boost": bi.BOOST_VERSION,
        "with_netgen": bi.WITH_NETGEN,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# Native libraries write to stdout (OCCT's XSTEP banner, SMESH traces), so a child reports
# its result on one line carrying this prefix and everything else on stdout is ignored.
_RESULT_PREFIX: Final[str] = "GOLDEN-RESULT "
_CHILD_TIMEOUT_S: Final[float] = 900.0


def _run_child(name: str) -> dict[str, Value]:
    """Run one probe in a fresh interpreter; a native crash becomes the probe's value.

    One process per probe is deliberate. A crash inside SMESH or OCCT (an access violation,
    not a Python exception) would otherwise end the whole capture, and it is exactly the
    kind of behaviour change a dependency upgrade can introduce or remove.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe", name],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout: no result within {_CHILD_TIMEOUT_S:.0f} s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            return json.loads(line[len(_RESULT_PREFIX) :])
    code = proc.returncode & 0xFFFFFFFF
    return {"error": f"crash: exit code 0x{code:08X}, no result reported"}


def capture() -> dict[str, object]:
    """Run every registered probe, in registration order, each in its own process."""
    probes: dict[str, object] = {}
    for name, (group, _fn) in _PROBES.items():
        print(f"[capture] {name}", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        values = _run_child(name)
        probes[name] = {
            "group": group,
            "seconds": round(time.perf_counter() - t0, 3),
            "values": values,
        }
    return {"meta": _meta(), "probes": probes}


def main(argv: list[str]) -> int:
    """Command-line entry point; see the module docstring."""
    if len(argv) == 2 and argv[0] == "--probe":
        values = _guard(_PROBES[argv[1]][1])
        # Leading newline: a native writer may leave stdout mid-line (the IGES dump does).
        print("\n" + _RESULT_PREFIX + json.dumps(values), flush=True)
        return 0
    if len(argv) != 1:
        raise SystemExit(__doc__)
    data = capture()
    Path(argv[0]).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    probes = cast("dict[str, dict[str, dict[str, Value]]]", data["probes"])
    errors = {
        n: p["values"]["error"] for n, p in probes.items() if "error" in p["values"]
    }
    print(f"{argv[0]}: {len(probes)} probes, {len(errors)} recorded an error")
    for name, error in errors.items():
        print(f"  {name}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
