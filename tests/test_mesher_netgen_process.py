# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-05

"""A process that loads pySMESH ends cleanly, with netgen linked statically into _core.

netgen 6.2.2101 registers its archivable classes in a map in ngcore (``archive.cpp``,
``type_register``), and each ``RegisterClassForArchive`` object took its class out of
the map again in its destructor. In netgen's own build ngcore is a separate DLL, loaded
first and unloaded last. Linked statically into ``_core``, netgen's registering objects
(one of them ``regob`` in ``csg/brick.cpp``) are built before ``archive.cpp`` registers
the destructor of the map's owner, so at exit the map is freed first and the later
destructors read freed memory. Measured on the branch before the fix: 6 of 20 processes
that only imported pySMESH ended with an access violation (0xC0000005) in
``Archive::IsRegistered``, called from ``~RegisterClassForArchive`` during
``DLL_PROCESS_DETACH``. netgen fixed it upstream by not unregistering at all (e1d71a78,
"no need to remove archive type infos", in 6.2.2105).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pysmesh as ps

_RUNS = 20

_CHILD: str = """
import os, sys
occt = os.environ.get("PYSMESH_OCCT_BIN")
if occt:
    os.add_dll_directory(occt)
lib = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(lib):
    os.add_dll_directory(lib)
sys.path.insert(0, sys.argv[1])
import pysmesh
print("EXIT-CHILD imported " + pysmesh.__version__, flush=True)
"""


def test_a_process_that_imports_pysmesh_exits_cleanly_every_time() -> None:
    """Twenty fresh processes import pySMESH and end: each one exits with code 0.

    At the crash rate measured before the fix, 6 in 20, all twenty pass with chance
    0.7^20 = 8e-4.
    """
    package_root = str(Path(ps.__file__).resolve().parent.parent)

    results = [
        subprocess.run(
            [sys.executable, "-c", _CHILD, package_root],
            capture_output=True,
            text=True,
            timeout=120.0,
            env=dict(os.environ),
            check=False,
        )
        for _ in range(_RUNS)
    ]

    codes = [proc.returncode for proc in results]
    assert codes == [0] * _RUNS, codes
    assert all("EXIT-CHILD imported" in proc.stdout for proc in results)


# ---- A NETGEN compute on the 2 MB main thread ----------------------------------------- #

# The main thread of python.exe reserves 2 MB of stack. This NETGEN 1D-2D compute of a
# wing in a box (10 faces, a Quadrangle_2D cap of 4 segments on the narrowest wall face)
# needs more than 2 MB: run on the caller's thread it ended the process with 0xC00000FD.
# The compute runs on a thread of its own with a 256 MB stack reservation.
_WING_BOX_CHILD: str = """
import os, sys
occt = os.environ.get("PYSMESH_OCCT_BIN")
if occt:
    os.add_dll_directory(occt)
lib = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(lib):
    os.add_dll_directory(lib)
sys.path.insert(0, sys.argv[1])
import numpy as np
import pysmesh as ps
from pysmesh.session import EntityKind
from pysmesh.mesher import (Mesher, Netgen1D2D, NetgenParameters2D, NumberOfSegments,
                            Quadrangle2D, Regular1D, SubShape, SubShapeKind)


def edges(s, d):
    return [int(i) for i in d.created if s.entity_kind(int(i)) == EntityKind.EDGE]


def section(s, z, chord, x0):
    beta = np.linspace(0.0, np.pi, 41)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3
                - 0.1015 * x**4)
    up = np.column_stack([x0 + chord * x[::-1], chord * yt[::-1], np.full(41, z)])
    lo = np.column_stack([x0 + chord * x, -chord * yt, np.full(41, z)])
    e = (edges(s, s.add_spline(up)) + edges(s, s.add_spline(lo))
         + edges(s, s.add_line(tuple(lo[-1]), tuple(up[0]))))
    d = s.make_wire(e)
    return ([i for i in e if s.is_alive(i)] + edges(s, d))[0]


s = ps.Session()
sections = [[section(s, 0.0, 1.0, 0.0)], [section(s, 2.0, 0.6, 0.3)]]
s.thru_sections(sections, solid=True, ruled=True)
wing = int(max(s.entities(EntityKind.SOLID)))
s.add_box(8.0, 6.0, 4.0, origin=(-3.0, -3.0, 0.0))
box = int(max(s.entities(EntityKind.SOLID)))
s.cut([box], [wing])
shape = ps.load_brep(s.brep())
walls = [f.id for f in shape.faces()
         if f.bbox[0] > -2.9 and f.bbox[3] < 4.9 and f.bbox[1] > -2.9 and f.bbox[4] < 2.9
         and f.bbox[5] > 1e-9]
areas = {f.id: f.area for f in shape.faces()}
cap = min(walls, key=lambda i: areas[i])
m = Mesher(shape)
m.assign(Netgen1D2D())
m.assign(NetgenParameters2D(max_size=0.4, threads=8, local_sizes=tuple(
    (SubShape(SubShapeKind.FACE, f), 0.05) for f in walls)))
on = SubShape(SubShapeKind.FACE, cap)
m.assign(Regular1D(), on=on)
m.assign(NumberOfSegments(count=4), on=on)
m.assign(Quadrangle2D(), on=on)
try:
    report = m.compute()
    print("WING-BOX-CHILD ok " + str(report.faces), flush=True)
except ps.PysmeshError as exc:
    print("WING-BOX-CHILD error " + str(exc) + " | " + exc.details, flush=True)
"""


def test_a_netgen_compute_that_needs_more_than_2_mb_of_stack_ends_normally() -> None:
    """The compute ends with its own result on the child's 2 MB main thread.

    The child calls compute() on its main thread. On a 4 MB thread this input ended with
    "Problem in Surface mesh generation" on face 7: that refusal is NETGEN's answer for
    it, and the process stays up.
    """
    package_root = str(Path(ps.__file__).resolve().parent.parent)

    proc = subprocess.run(
        [sys.executable, "-c", _WING_BOX_CHILD, package_root],
        capture_output=True,
        text=True,
        timeout=600.0,
        env=dict(os.environ),
        check=False,
    )

    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])
    result = [line for line in proc.stdout.splitlines() if "WING-BOX-CHILD" in line]
    assert result == [
        "WING-BOX-CHILD error Mesher.compute: meshing failed on 1 sub-shape(s). | "
        "FACE 7: NgException at Surface meshing: Problem in Surface mesh generation "
        "(algorithm NETGEN_2D)"
    ]
