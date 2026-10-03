# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-03

"""Gates for non-finite arguments: a NaN or an infinity is refused before OCCT or SMESH.

A NaN passes every ``<=`` and ``<`` test, so the argument checks let it through, and an
infinity passed most of them. OCCT then crashed the process (``extrude`` with a NaN
vector, ``mirror`` with a NaN normal) or built garbage that the session committed
(``translate``, ``add_circle`` with a NaN centre) (finding F1, brief amendment 3).

The oracle is the contract: every public float argument is finite, or the call raises
``PysmeshError`` naming the operation, the argument and the value, and a session is left
byte for byte as it was. Each case gives one argument NaN, +inf or -inf and every other
argument a valid value. A case that crashed the reference process runs in a child
process, so that the crash fails one case and not the run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps

# The session every session case starts from, as source: the crash cases run it again in
# a child process. A unit box (solid, faces, edges), a loose unit square face at x = 3,
# a loose line along z at x = 5 (a spine), and a loose circle at its foot (a profile).
WORLD: str = """
import pysmesh as ps
from pysmesh import EntityKind as K

s = ps.Session()
s.add_box(1.0, 1.0, 1.0)
box = int(s.entities(K.SOLID)[0])
box_faces = [int(i) for i in s.entities(K.FACE)]
box_edges = [int(i) for i in s.entities(K.EDGE)]
s.add_rectangle((3.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0)
rect = [int(i) for i in s.entities(K.FACE) if int(i) not in box_faces]
before = {int(i) for i in s.entities(K.EDGE)}
s.add_line((5.0, 0.0, 0.0), (5.0, 0.0, 2.0))
spine = [int(i) for i in s.entities(K.EDGE) if int(i) not in before]
before = {int(i) for i in s.entities(K.EDGE)}
s.add_circle((5.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2)
profile = [int(i) for i in s.entities(K.EDGE) if int(i) not in before]
pts = [(0.0, 0.0, 0.0), (1.0, 0.5, 0.0), (2.0, 0.0, 0.0)]
"""

# The source of each bad value; repr() of a non-finite float is not valid Python.
BAD: dict[str, str] = {
    "nan": "float('nan')",
    "+inf": "float('inf')",
    "-inf": "float('-inf')",
}

# (operation, argument, call). In the call, `v` is the bad value; WORLD's names are
# bound.
SESSION_CASES: list[tuple[str, str, str]] = [
    ("add_box", "dx", "s.add_box(v, 1.0, 1.0)"),
    ("add_box", "dy", "s.add_box(1.0, v, 1.0)"),
    ("add_box", "dz", "s.add_box(1.0, 1.0, v)"),
    ("add_box", "origin", "s.add_box(1.0, 1.0, 1.0, origin=(v, 0.0, 0.0))"),
    ("add_cylinder", "radius", "s.add_cylinder(v, 1.0)"),
    ("add_cylinder", "height", "s.add_cylinder(1.0, v)"),
    ("add_cylinder", "origin", "s.add_cylinder(1.0, 1.0, origin=(0.0, v, 0.0))"),
    ("add_cylinder", "axis", "s.add_cylinder(1.0, 1.0, axis=(0.0, v, 1.0))"),
    ("add_cone", "radius1", "s.add_cone(v, 1.0, 1.0)"),
    ("add_cone", "radius2", "s.add_cone(0.5, v, 1.0)"),
    ("add_cone", "height", "s.add_cone(0.5, 1.0, v)"),
    ("add_cone", "origin", "s.add_cone(0.5, 1.0, 1.0, origin=(v, 0.0, 0.0))"),
    ("add_cone", "axis", "s.add_cone(0.5, 1.0, 1.0, axis=(v, 0.0, 1.0))"),
    ("add_cone", "angle_rad", "s.add_cone(0.5, 1.0, 1.0, angle_rad=v)"),
    ("add_sphere", "radius", "s.add_sphere(v)"),
    ("add_sphere", "centre", "s.add_sphere(1.0, centre=(0.0, 0.0, v))"),
    ("add_sphere", "axis", "s.add_sphere(1.0, axis=(v, 0.0, 1.0))"),
    ("add_sphere", "angle_rad", "s.add_sphere(1.0, angle_rad=v)"),
    ("add_torus", "radius1", "s.add_torus(v, 0.2)"),
    ("add_torus", "radius2", "s.add_torus(1.0, v)"),
    ("add_torus", "origin", "s.add_torus(1.0, 0.2, origin=(v, 0.0, 0.0))"),
    ("add_torus", "axis", "s.add_torus(1.0, 0.2, axis=(0.0, v, 1.0))"),
    ("add_torus", "angle_rad", "s.add_torus(1.0, 0.2, angle_rad=v)"),
    ("add_wedge", "dx", "s.add_wedge(v, 1.0, 1.0, 0.5)"),
    ("add_wedge", "dy", "s.add_wedge(1.0, v, 1.0, 0.5)"),
    ("add_wedge", "dz", "s.add_wedge(1.0, 1.0, v, 0.5)"),
    ("add_wedge", "ltx", "s.add_wedge(1.0, 1.0, 1.0, v)"),
    ("add_wedge", "origin", "s.add_wedge(1.0, 1.0, 1.0, 0.5, origin=(v, 0.0, 0.0))"),
    ("add_wedge", "axis", "s.add_wedge(1.0, 1.0, 1.0, 0.5, axis=(v, 0.0, 1.0))"),
    ("add_vertex", "point", "s.add_vertex((v, 0.0, 0.0))"),
    ("add_line", "start", "s.add_line((v, 0.0, 0.0), (1.0, 0.0, 0.0))"),
    ("add_line", "end", "s.add_line((0.0, 0.0, 0.0), (1.0, 0.0, v))"),
    ("add_arc", "start", "s.add_arc((v, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0))"),
    (
        "add_arc",
        "through",
        "s.add_arc((0.0, 0.0, 0.0), (1.0, v, 0.0), (2.0, 0.0, 0.0))",
    ),
    ("add_arc", "end", "s.add_arc((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, v))"),
    ("add_circle", "centre", "s.add_circle((0.0, 0.0, v), (0.0, 0.0, 1.0), 1.0)"),
    ("add_circle", "normal", "s.add_circle((0.0, 0.0, 0.0), (0.0, v, 1.0), 1.0)"),
    ("add_circle", "radius", "s.add_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), v)"),
    (
        "add_ellipse",
        "centre",
        "s.add_ellipse((v, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, 1.0)",
    ),
    (
        "add_ellipse",
        "normal",
        "s.add_ellipse((0.0, 0.0, 0.0), (v, 0.0, 1.0), 2.0, 1.0)",
    ),
    ("add_ellipse", "rx", "s.add_ellipse((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), v, 1.0)"),
    ("add_ellipse", "ry", "s.add_ellipse((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, v)"),
    (
        "add_ellipse",
        "x_dir",
        (
            "s.add_ellipse((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, 1.0, "
            "x_dir=(1.0, v, 0.0))"
        ),
    ),
    ("add_polyline", "points", "s.add_polyline([(0.0, 0.0, 0.0), (v, 1.0, 0.0)])"),
    ("add_spline", "points", "s.add_spline([*pts[:2], (3.0, v, 0.0)])"),
    ("add_spline", "tol", "s.add_spline(pts, tol=v)"),
    ("add_bspline", "poles", "s.add_bspline([*pts[:2], (3.0, 0.0, v)])"),
    (
        "add_helix",
        "centre",
        "s.add_helix((v, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 0.5, 2.0)",
    ),
    ("add_helix", "axis", "s.add_helix((0.0, 0.0, 0.0), (0.0, v, 1.0), 1.0, 0.5, 2.0)"),
    (
        "add_helix",
        "diameter",
        "s.add_helix((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), v, 0.5, 2.0)",
    ),
    (
        "add_helix",
        "pitch",
        "s.add_helix((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, v, 2.0)",
    ),
    (
        "add_helix",
        "turns",
        "s.add_helix((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 0.5, v)",
    ),
    (
        "add_helix",
        "tol",
        "s.add_helix((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 0.5, 2.0, tol=v)",
    ),
    (
        "add_rectangle",
        "origin",
        "s.add_rectangle((v, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0)",
    ),
    (
        "add_rectangle",
        "normal",
        "s.add_rectangle((0.0, 0.0, 0.0), (0.0, v, 1.0), 1.0, 1.0)",
    ),
    (
        "add_rectangle",
        "dx",
        "s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), v, 1.0)",
    ),
    (
        "add_rectangle",
        "dy",
        "s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, v)",
    ),
    ("extrude", "vector", "s.extrude(rect, (v, 0.0, 1.0))"),
    ("revolve", "origin", "s.revolve(rect, (v, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0)"),
    ("revolve", "axis", "s.revolve(rect, (0.0, 0.0, 0.0), (0.0, v, 1.0), 1.0)"),
    ("revolve", "angle_rad", "s.revolve(rect, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), v)"),
    ("fuse", "fuzzy", "s.fuse([box], rect, fuzzy=v)"),
    ("cut", "fuzzy", "s.cut([box], rect, fuzzy=v)"),
    ("common", "fuzzy", "s.common([box], rect, fuzzy=v)"),
    ("section", "fuzzy", "s.section([box], rect, fuzzy=v)"),
    ("split", "fuzzy", "s.split([box], rect, fuzzy=v)"),
    ("fragment", "fuzzy", "s.fragment([box, *rect], fuzzy=v)"),
    ("imprint", "fuzzy", "s.imprint([box], rect, fuzzy=v)"),
    ("fillet", "radius", "s.fillet(box_edges[:1], v)"),
    ("fillet", "radius_end", "s.fillet(box_edges[:1], 0.1, radius_end=v)"),
    ("chamfer", "distance", "s.chamfer(box_edges[:1], v)"),
    (
        "chamfer",
        "distance_end",
        "s.chamfer(box_edges[:1], 0.1, distance_end=v, face_id=box_faces[0])",
    ),
    ("make_thick_solid", "thickness", "s.make_thick_solid(box_faces[:1], v)"),
    ("make_thick_solid", "tol", "s.make_thick_solid(box_faces[:1], 0.1, tol=v)"),
    ("offset", "distance", "s.offset([box], v)"),
    ("offset", "tol", "s.offset([box], 0.1, tol=v)"),
    ("translate", "offset", "s.translate((v, 0.0, 0.0))"),
    ("rotate", "origin", "s.rotate((v, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)"),
    ("rotate", "axis", "s.rotate((0.0, 0.0, 0.0), (0.0, v, 1.0), 1.0)"),
    ("rotate", "angle_rad", "s.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), v)"),
    ("mirror", "point", "s.mirror((v, 0.0, 0.0), (0.0, 0.0, 1.0))"),
    ("mirror", "normal", "s.mirror((0.0, 0.0, 0.0), (v, 0.0, 1.0))"),
    ("scale", "factors", "s.scale((2.0, v, 2.0))"),
    ("scale", "centre", "s.scale(2.0, (v, 0.0, 0.0))"),
    ("heal", "precision", "s.heal(precision=v)"),
    ("heal", "min_tolerance", "s.heal(min_tolerance=v)"),
    ("heal", "max_tolerance", "s.heal(max_tolerance=v)"),
    ("sew", "tolerance", "s.sew(box_faces, tolerance=v)"),
    ("remove_internal_wires", "min_area", "s.remove_internal_wires(min_area=v)"),
    ("unify_same_domain", "linear_tol", "s.unify_same_domain(linear_tol=v)"),
    ("unify_same_domain", "angular_tol_deg", "s.unify_same_domain(angular_tol_deg=v)"),
    ("mass_properties", "precision", "s.mass_properties([box], precision=v)"),
    ("surface_at", "uv", "s.surface_at(box_faces[0], [(v, 0.0)])"),
    ("curve_at", "t", "s.curve_at(box_edges[0], [v])"),
    ("project_on_face", "points", "s.project_on_face(box_faces[0], [(v, 0.0, 0.0)])"),
    (
        "entities_in_box",
        "minimum",
        "s.entities_in_box(K.FACE, (v, 0.0, 0.0), (1.0, 1.0, 1.0))",
    ),
    (
        "entities_in_box",
        "maximum",
        "s.entities_in_box(K.FACE, (0.0, 0.0, 0.0), (1.0, v, 1.0))",
    ),
    ("contains", "points", "s.contains([box], [(v, 0.5, 0.5)])"),
    ("contains", "tol", "s.contains([box], [(0.5, 0.5, 0.5)], tol=v)"),
    ("tessellate", "deflection", "s.tessellate(deflection=v)"),
    ("tessellate", "angle_deg", "s.tessellate(angle_deg=v)"),
]

# The module functions' cases start from a unit box as BREP bytes and as a Shape.
MODULE_WORLD: str = """
import numpy as np
import pysmesh as ps

s = ps.Session()
s.add_box(1.0, 1.0, 1.0)
brep = s.brep()
shape = ps.load_brep(brep)
"""

MODULE_CASES: list[tuple[str, str, str]] = [
    ("offset_shape", "offset", "ps.offset_shape(brep, ps.OffsetParams(offset=v))"),
    (
        "offset_shape",
        "tol",
        "ps.offset_shape(brep, ps.OffsetParams(offset=0.1, tol=v))",
    ),
    (
        "make_thick_solid",
        "thickness",
        (
            "ps.make_thick_solid(brep, "
            "ps.ThickSolidParams(remove_face_ids=(1,), thickness=v))"
        ),
    ),
    (
        "make_thick_solid",
        "tol",
        "ps.make_thick_solid(brep, ps.ThickSolidParams((1,), 0.1, tol=v))",
    ),
    ("point_in_solid", "points", "ps.point_in_solid(brep, np.array([[v, 0.5, 0.5]]))"),
    (
        "point_in_solid",
        "tol",
        "ps.point_in_solid(brep, np.array([[0.5, 0.5, 0.5]]), v)",
    ),
    ("tessellate", "lin_defl", "ps.tessellate(brep, ps.TessellateParams(lin_defl=v))"),
    (
        "tessellate",
        "ang_defl_deg",
        "ps.tessellate(brep, ps.TessellateParams(ang_defl_deg=v))",
    ),
    (
        "unify_same_domain",
        "linear_tol",
        "ps.unify_same_domain(brep, ps.UnifyParams(linear_tol=v))",
    ),
    (
        "unify_same_domain",
        "angular_tol_deg",
        "ps.unify_same_domain(brep, ps.UnifyParams(angular_tol_deg=v))",
    ),
    (
        "Shape.face_distance",
        "points",
        "shape.face_distance(1, np.array([[v, 0.0, 0.0]]))",
    ),
    (
        "Shape.match_faces",
        "centroids",
        "shape.match_faces(np.array([[v, 0.5, 0.0]]), 1e-6)",
    ),
    ("Shape.match_faces", "tol", "shape.match_faces(np.array([[0.5, 0.5, 0.0]]), v)"),
    ("Mesh.add_nodes", "coords", "ps.Mesh(shape).add_nodes(np.array([[v, 0.0, 0.0]]))"),
]

# (hypothesis, field, constructor). `m` is pysmesh.mesher and `v` the bad value. A tuple
# field gets the bad value in one place.
HYPOTHESIS_CASES: list[tuple[str, str, str]] = [
    (
        "Adaptive1D",
        "min_size",
        "m.Adaptive1D(min_size=v, max_size=1.0, deflection=0.01)",
    ),
    (
        "Adaptive1D",
        "max_size",
        "m.Adaptive1D(min_size=0.1, max_size=v, deflection=0.01)",
    ),
    (
        "Adaptive1D",
        "deflection",
        "m.Adaptive1D(min_size=0.1, max_size=1.0, deflection=v)",
    ),
    ("Arithmetic1D", "start_length", "m.Arithmetic1D(start_length=v, end_length=2.0)"),
    ("Arithmetic1D", "end_length", "m.Arithmetic1D(start_length=0.5, end_length=v)"),
    ("AutomaticLength", "fineness", "m.AutomaticLength(fineness=v)"),
    (
        "CartesianParameters3D",
        "size_threshold",
        (
            "m.CartesianParameters3D(spacing_x='1.0', spacing_y='1.0', "
            "spacing_z='1.0', "
            "size_threshold=v)"
        ),
    ),
    (
        "CartesianParameters3D",
        "spacing_from",
        (
            "m.CartesianParameters3D(spacing_x='1.0', spacing_y='1.0', "
            "spacing_z='1.0', "
            "spacing_from=(0.0, v))"
        ),
    ),
    ("Deflection1D", "deflection", "m.Deflection1D(deflection=v)"),
    (
        "FixedPoints1D",
        "points",
        "m.FixedPoints1D(points=(0.25, v), segment_counts=(2, 4, 2))",
    ),
    ("Geometric1D", "start_length", "m.Geometric1D(start_length=v, common_ratio=1.2)"),
    ("Geometric1D", "common_ratio", "m.Geometric1D(start_length=0.5, common_ratio=v)"),
    ("LocalLength", "length", "m.LocalLength(length=v)"),
    ("LocalLength", "precision", "m.LocalLength(length=1.0, precision=v)"),
    ("MaxElementArea", "max_area", "m.MaxElementArea(max_area=v)"),
    ("MaxElementVolume", "max_volume", "m.MaxElementVolume(max_volume=v)"),
    ("MaxLength", "length", "m.MaxLength(length=v)"),
    (
        "NumberOfSegments",
        "scale_factor",
        (
            "m.NumberOfSegments(count=4, distribution=m.Distribution.SCALE, "
            "scale_factor=v)"
        ),
    ),
    (
        "NumberOfSegments",
        "table",
        (
            "m.NumberOfSegments(count=4, distribution=m.Distribution.TABLE, "
            "table=(0.0, 1.0, 1.0, v))"
        ),
    ),
    ("SegmentLengthAroundVertex", "length", "m.SegmentLengthAroundVertex(length=v)"),
    (
        "StartEndLength",
        "start_length",
        "m.StartEndLength(start_length=v, end_length=2.0)",
    ),
    (
        "StartEndLength",
        "end_length",
        "m.StartEndLength(start_length=0.5, end_length=v)",
    ),
    (
        "ViscousLayers",
        "total_thickness",
        (
            "m.ViscousLayers(total_thickness=v, layer_count=3, stretch_factor=1.2, "
            "boundary=(1, 2), group_name='layers')"
        ),
    ),
    (
        "ViscousLayers",
        "stretch_factor",
        (
            "m.ViscousLayers(total_thickness=0.3, layer_count=3, stretch_factor=v, "
            "boundary=(1, 2), group_name='layers')"
        ),
    ),
    (
        "ViscousLayers2D",
        "total_thickness",
        (
            "m.ViscousLayers2D(total_thickness=v, layer_count=3, stretch_factor=1.2, "
            "boundary=(1, 2), group_name='layers2d')"
        ),
    ),
    (
        "ViscousLayers2D",
        "stretch_factor",
        (
            "m.ViscousLayers2D(total_thickness=0.3, layer_count=3, stretch_factor=v, "
            "boundary=(1, 2), group_name='layers2d')"
        ),
    ),
]

# The cases that crashed the reference process (stack overflow, 0xC00000FD), measured.
CRASHED: set[tuple[str, str, str]] = {
    ("extrude", "vector", "nan"),
    ("mirror", "normal", "nan"),
    ("mirror", "normal", "+inf"),
    ("mirror", "normal", "-inf"),
}
CHILD_TIMEOUT_S: float = 120.0

# The DLL directories and the package path a child process needs before its import.
_CHILD_PRELUDE: str = """
import os, sys
for d in (os.environ.get("PYSMESH_OCCT_BIN"),
          os.path.join(sys.prefix, "Library", "bin")):
    if d and os.path.isdir(d):
        os.add_dll_directory(d)
sys.path.insert(0, sys.argv[1])
import pysmesh as ps
"""

# One session case in a child: "refused" is a PysmeshError naming the argument and the
# value, with the session unchanged.
_CASE_CHILD: str = _CHILD_PRELUDE + """
world, call, arg, value, text = sys.argv[2:7]
g = {}
exec(world, g)
s = g["s"]
before = (s.op_count, s.issued_id_count, s.brep())
g["v"] = eval(value)
try:
    eval(call, g)
    print("RESULT committed")
except ps.PysmeshError as e:
    same = (s.op_count, s.issued_id_count, s.brep()) == before
    ok = same and arg in str(e) and text in str(e)
    print("RESULT", "refused" if ok else "wrong", "|", str(e)[:300])
except Exception as e:
    print("RESULT raw", type(e).__name__, "|", str(e)[:300])
"""

# The hypothesis sweep in a child: every case assigned to a Mesher on a unit box.
_HYPOTHESIS_CHILD: str = _CHILD_PRELUDE + """
import pysmesh.mesher as m
rows = eval(sys.argv[2])
s = ps.Session()
s.add_box(1.0, 1.0, 1.0)
shape = ps.load_brep(s.brep())
for name, field, text, source, call in rows:
    try:
        hypothesis = eval(call, {"m": m, "v": eval(source)})
        with ps.Mesher(shape) as mesher:
            mesher.assign(hypothesis)
        print("CASE assigned", name, field, text, flush=True)
    except ps.PysmeshError as e:
        ok = field in str(e) and text.removeprefix("+") in str(e)
        print("CASE", "refused" if ok else "wrong", name, field, text, "|",
              str(e)[:200], flush=True)
    except Exception as e:
        print("CASE raw", name, field, text, type(e).__name__, str(e)[:200], flush=True)
"""


def _text(value: str) -> str:
    """How a refusal writes the bad value: nan, inf or -inf."""
    return value.removeprefix("+")


def _state(s: ps.Session) -> tuple[int, int, bytes]:
    """What a refused call must leave alone: op count, issued ids, BREP bytes."""
    return (s.op_count, s.issued_id_count, s.brep())


def _child(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a script in a child process with this package on its path."""
    package_root = str(Path(ps.__file__).resolve().parent.parent)
    return subprocess.run(
        [sys.executable, "-c", script, package_root, *args],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
        env=dict(os.environ),
        check=False,
    )


def _refuses(world: str, call: str, arg: str, value: str) -> None:
    """Run one case in this process; it must raise naming the argument and the value."""
    g: dict[str, object] = {}
    exec(world, g)  # noqa: S102 - the case table's own setup source
    s = g.get("s")
    before = _state(s) if isinstance(s, ps.Session) else None
    g["v"] = float(_text(value))

    with pytest.raises(ps.PysmeshError) as info:
        eval(call, g)

    assert arg in str(info.value)
    assert _text(value) in str(info.value)
    if isinstance(s, ps.Session):
        assert _state(s) == before


def _inject(shape: ps.Shape, m: dict[str, NDArray[np.generic]]) -> ps.Mesh:
    """A Mesh holding the box surface mesh of the fixture, classified on its shape."""
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
    return mesh


@pytest.mark.parametrize("value", list(BAD))
@pytest.mark.parametrize(
    ("op", "arg", "call"), SESSION_CASES, ids=[f"{o}-{a}" for o, a, _ in SESSION_CASES]
)
def test_a_session_operation_refuses_a_non_finite_argument(
    op: str, arg: str, call: str, value: str
) -> None:
    """Every float argument of the session API: NaN, +inf and -inf are refused."""
    if (op, arg, value) in CRASHED:
        proc = _child(_CASE_CHILD, WORLD, call, arg, BAD[value], _text(value))
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
        code = proc.returncode & 0xFFFFFFFF
        assert lines, f"child exit code 0x{code:08X}"
        assert lines[0].startswith("RESULT refused"), lines[0]
        return

    _refuses(WORLD, call, arg, value)


@pytest.mark.parametrize("value", list(BAD))
@pytest.mark.parametrize(
    ("op", "arg", "call"), MODULE_CASES, ids=[f"{o}-{a}" for o, a, _ in MODULE_CASES]
)
def test_a_module_function_refuses_a_non_finite_argument(
    op: str, arg: str, call: str, value: str
) -> None:
    """The standalone functions that reach OCCT or SMESH: NaN, +inf, -inf refused."""
    _refuses(MODULE_WORLD, call, arg, value)


@pytest.mark.parametrize("value", list(BAD))
@pytest.mark.parametrize("arg", ["total_thickness", "stretch_factor"])
def test_viscous_layers_refuse_a_non_finite_parameter(
    arg: str, value: str, box_brep: bytes, box_mesh: dict[str, NDArray[np.generic]]
) -> None:
    """compute_viscous_layers on the box's injected surface mesh refuses the value."""
    mesh = _inject(ps.load_brep(box_brep), box_mesh)
    params = {"total_thickness": 0.05, "stretch_factor": 1.2}
    params[arg] = float(_text(value))

    with pytest.raises(ps.PysmeshError) as info:
        ps.compute_viscous_layers(
            mesh,
            ps.VLParams(
                face_ids=(1,),
                total_thickness=params["total_thickness"],
                n_layers=2,
                stretch_factor=params["stretch_factor"],
            ),
        )

    assert arg in str(info.value)


def test_a_hypothesis_refuses_a_non_finite_field() -> None:
    """One bad value per numeric field of the catalogue, NaN, +inf and -inf, in a child.

    Each hypothesis is assigned to a Mesher on a box; the assignment must raise naming
    the field and the value. A child process, so that a value SMESH crashes on fails the
    test and not the run.
    """
    rows = [
        (name, field, text, BAD[text], call)
        for name, field, call in HYPOTHESIS_CASES
        for text in BAD
    ]

    proc = _child(_HYPOTHESIS_CHILD, repr(rows))

    results = [ln for ln in proc.stdout.splitlines() if ln.startswith("CASE ")]
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert len(results) == len(rows)
    assert [r for r in results if not r.startswith("CASE refused")] == []
