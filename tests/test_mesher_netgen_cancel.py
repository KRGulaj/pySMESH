# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-05

"""A cancel stops a long NETGEN compute in a stated time and leaves a reusable Mesher.

The cancel predicate reaches netgen through ``SMESH_Gen::CancelCompute``, which sets
netgen's global flag ``multithread.terminate`` (``NETGENPlugin_NETGEN_2D3D::CancelCompute``).
netgen checks that flag between its steps and per inserted point in its Delaunay loop,
not inside every step. Measured on the reference machine (16 threads), 23 cancels spread
over four runs of 5 to 33 s (``g2/np9_latency.txt`` in the record directory): the worst
latency was 2.2 s, a cancel early in volume meshing. A stage timing of that case: netgen's
volume step ran on for 0.7 to 2.1 s after the flag was set, and the plugin then copied the
partial mesh into SMESH (0.16 to 0.47 s) before pySMESH cleared it. The stated bound is
5 s, about twice the worst measured latency.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pysmesh as ps

_LATENCY_BOUND_S = 5.0
# Seconds into the run: in surface meshing, and early in volume meshing, where netgen
# answers a cancel slowest.
_DELAYS_S = (0.5, 2.4)

_CHILD: str = """
import hashlib, json, os, sys, time
occt = os.environ.get("PYSMESH_OCCT_BIN")
if occt:
    os.add_dll_directory(occt)
lib = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(lib):
    os.add_dll_directory(lib)
sys.path.insert(0, sys.argv[1])
import pysmesh as ps

s = ps.Session()
s.add_torus(3.0, 1.0)
torus = ps.load_brep(s.brep())

def assign(m):
    m.assign(ps.Netgen1D2D3D())
    m.assign(ps.NetgenParameters(max_size=0.12))

def digest(mesh):
    h = hashlib.sha256()
    for a in (mesh.node_coords, mesh.element_offsets, mesh.element_nodes,
              mesh.element_type):
        h.update(a.tobytes())
    return h.hexdigest()

with ps.Mesher(torus) as m:
    assign(m)
    m.compute()
    fresh = digest(m.mesh())

runs = []
for delay in json.loads(sys.argv[3]):
    start = time.perf_counter()
    asked = []
    def cancel():
        if time.perf_counter() - start < delay:
            return False
        if not asked:
            asked.append(time.perf_counter())
        return True
    with ps.Mesher(torus) as m:
        assign(m)
        try:
            m.compute(cancel=cancel)
            outcome = "computed"
        except ps.PysmeshCancelled:
            outcome = "cancelled"
        latency = time.perf_counter() - asked[0] if asked else None
        left = m.mesh().element_count
        m.compute()
        again = digest(m.mesh())
    runs.append({"outcome": outcome, "latency": latency, "left": left,
                 "again_equals_fresh": again == fresh})
with open(sys.argv[2], "w", encoding="utf-8") as out:
    json.dump(runs, out)
"""


def test_a_netgen_cancel_lands_in_bound_leaves_no_file_and_the_mesher_meshes_again(
    tmp_path: Path,
) -> None:
    """A torus (R 3, r 1) at max_size 0.12, about 5 s and 207 000 tetrahedra.

    For each delay, in a child process with its own temporary directory and working
    directory: the compute raises PysmeshCancelled within the bound, no element is left,
    the same Mesher then computes the mesh that a fresh Mesher computes (bit for bit),
    and no file appears in either directory.
    """
    package_root = str(Path(ps.__file__).resolve().parent.parent)
    temp, cwd = tmp_path / "temp", tmp_path / "cwd"
    temp.mkdir()
    cwd.mkdir()
    result_file = tmp_path / "runs.json"
    env = dict(os.environ, TMP=str(temp), TEMP=str(temp), TMPDIR=str(temp))

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            package_root,
            str(result_file),
            json.dumps(_DELAYS_S),
        ],
        capture_output=True,
        text=True,
        timeout=600.0,
        cwd=cwd,
        env=env,
        check=False,
    )

    assert proc.returncode == 0, f"exit {proc.returncode:#x}: {proc.stderr[-2000:]}"
    runs = json.loads(result_file.read_text(encoding="utf-8"))
    assert [r["outcome"] for r in runs] == ["cancelled"] * len(_DELAYS_S)
    assert all(r["latency"] < _LATENCY_BOUND_S for r in runs), runs
    assert all(r["left"] == 0 and r["again_equals_fresh"] for r in runs), runs
    assert sorted(os.listdir(temp)) == [] and sorted(os.listdir(cwd)) == []
