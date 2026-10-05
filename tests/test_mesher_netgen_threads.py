# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-05

"""NETGEN under threads: two meshers at once, and the same mesh at any thread count.

netgen keeps its state in globals (its meshing parameters, its cancel flag, its streams,
NETGENPlugin's local-size maps), and pySMESH releases the GIL during a compute, so two
threads could run netgen at once. Every compute that runs a NETGEN algorithm holds one
process-wide lock for its whole run (``Mesher::compute``, ``mesher_netgen.cpp``):

* Thread safety: two threads, each with its own :class:`~pysmesh.Mesher`, compute NETGEN
  meshes at the same time, 20 rounds; each result equals the sequential result bit for
  bit, and the process does not crash (it runs as a child process).
* Determinism: the same input gives the same mesh, node coordinates and connectivity bit
  for bit, at 1, 4 and 16 threads, at the default (one thread per hardware thread), and
  on repeat. NETGENPlugin's default thread count is kept on this condition.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pysmesh as ps
from pysmesh import EntityKind, Mesher, Netgen1D2D3D, NetgenParameters, Session

_CHILD: str = """
import hashlib, json, os, sys, threading
occt = os.environ.get("PYSMESH_OCCT_BIN")
if occt:
    os.add_dll_directory(occt)
lib = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(lib):
    os.add_dll_directory(lib)
sys.path.insert(0, sys.argv[1])
import pysmesh as ps

def shape(kind):
    s = ps.Session()
    if kind == "sphere":
        s.add_sphere(1.0)
    else:
        s.add_cylinder(1.0, 3.0)
    return ps.load_brep(s.brep())

def digest(kind, mesh_shape):
    with ps.Mesher(mesh_shape) as m:
        m.assign(ps.Netgen1D2D3D())
        m.assign(ps.NetgenParameters(max_size=0.4))
        m.compute()
        mesh = m.mesh()
    h = hashlib.sha256()
    for a in (mesh.node_coords, mesh.element_offsets, mesh.element_nodes,
              mesh.element_type):
        h.update(a.tobytes())
    return h.hexdigest()

shapes = {k: shape(k) for k in ("sphere", "cylinder")}
reference = {k: digest(k, s) for k, s in shapes.items()}
rounds = []
for _ in range(20):
    barrier = threading.Barrier(2)
    got = {}
    def work(kind):
        barrier.wait()
        got[kind] = digest(kind, shapes[kind])
    threads = [threading.Thread(target=work, args=(k,)) for k in shapes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rounds.append(got == reference)
with open(sys.argv[2], "w", encoding="utf-8") as out:
    json.dump({"rounds": rounds}, out)
"""


def test_two_threads_mesh_with_netgen_at_once_and_match_the_sequential_meshes(
    tmp_path: Path,
) -> None:
    """20 rounds of a sphere and a cylinder meshed together, in a child process."""
    package_root = str(Path(ps.__file__).resolve().parent.parent)
    result_file = tmp_path / "rounds.json"

    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, package_root, str(result_file)],
        capture_output=True,
        text=True,
        timeout=600.0,
        env=dict(os.environ),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr[-2000:]
    rounds = json.loads(result_file.read_text(encoding="utf-8"))["rounds"]
    assert rounds == [True] * 20


def _shape(kind: str) -> tuple[ps.Shape, float]:
    """A unit sphere, a torus or a partitioned box, and the max_size to mesh it with."""
    s = Session()
    if kind == "sphere":
        s.add_sphere(1.0)
        size = 0.3
    elif kind == "torus":
        s.add_torus(3.0, 1.0)
        size = 0.6
    else:
        s.add_box(2.0, 1.0, 1.0)
        s.add_box(1.0, 1.0, 1.0)
        s.fragment(s.entities(EntityKind.SOLID).tolist())
        size = 0.3
    return ps.load_brep(s.brep()), size


def _mesh_digest(shape: ps.Shape, size: float, threads: int | None) -> str:
    """The SHA-256 of the node coordinates and the connectivity of one compute."""
    with Mesher(shape) as mesher:
        mesher.assign(Netgen1D2D3D())
        mesher.assign(NetgenParameters(max_size=size, threads=threads))
        mesher.compute()
        mesh = mesher.mesh()
    h = hashlib.sha256()
    for a in (
        mesh.node_coords,
        mesh.element_offsets,
        mesh.element_nodes,
        mesh.element_type,
    ):
        h.update(a.tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("kind", ["sphere", "torus", "partitioned"])
def test_netgen_gives_the_same_mesh_at_any_thread_count_and_on_repeat(
    kind: str,
) -> None:
    """1, 4, 16 threads, the default, and 1 thread again: one mesh, bit for bit."""
    shape, size = _shape(kind)

    digests = [_mesh_digest(shape, size, n) for n in (1, 4, 16, None, 1)]

    assert len(set(digests)) == 1
