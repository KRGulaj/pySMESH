# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-05

"""A NETGEN compute leaves its host process as it found it, and says why it failed.

pySMESH runs inside a host application. For every NETGEN compute, a success, a failure
and a cancel alike:

* nothing is written to fd 1 or fd 2, for the whole life of the process;
* the working directory does not change, and no file appears in it;
* no file appears in the temporary directory: NETGEN keeps its debug stream in memory;
* the environment variables KEEP_NETGEN_OUTPUT, SALOME_NETGEN_DISABLE_MULTITHREADING,
  NGS_NUM_THREADS and NGPROFILE change none of that.

NETGENPlugin V9_16_0 moved the process into a new temporary directory for each compute
and left it there, pointed std::cout at a log file in a second one, and read netgen's
errors from "test.out" in the working directory; NGPROFILE made netgen write
"netgen.prof" into the working directory at exit
(patches/netgenplugin/NETGENPlugin_runtime_containment.patch,
patches/netgen/netgen_no_ngprofile.patch).

Each case runs in a child process whose working directory and temporary directory (TMP,
TEMP) are fresh empty directories, so a file from any other process on the machine
cannot reach the check. The child writes its findings to a third directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pysmesh as ps
from pysmesh import (
    Mesher,
    Netgen2D,
    Netgen3D,
    NumberOfSegments,
    PysmeshError,
    Regular1D,
    Session,
)

_CHILD: str = """
import json, os, sys
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
        s.add_torus(3.0, 1.0)
    return ps.load_brep(s.brep())

found = {}
start = os.getcwd()
sphere, torus = shape("sphere"), shape("torus")

with ps.Mesher(sphere) as m:
    m.assign(ps.Netgen1D2D3D())
    m.assign(ps.NetgenParameters(max_size=0.4))
    found["success"] = m.compute().volumes > 0
found["cwd_after_success"] = os.getcwd() == start

with ps.Mesher(torus) as m:
    m.assign(ps.Regular1D())
    m.assign(ps.NumberOfSegments(count=3))
    m.assign(ps.Netgen2D())
    m.assign(ps.Netgen3D())
    try:
        m.compute()
        found["failure"] = "computed"
    except ps.PysmeshError as exc:
        found["failure"] = type(exc).__name__ + ": " + exc.details
found["cwd_after_failure"] = os.getcwd() == start

asked = [0]
def cancel():
    asked[0] += 1
    return asked[0] > 1

with ps.Mesher(torus) as m:
    m.assign(ps.Netgen1D2D3D())
    m.assign(ps.NetgenParameters(max_size=0.08))
    try:
        m.compute(cancel=cancel)
        found["cancel"] = "computed"
    except ps.PysmeshCancelled:
        found["cancel"] = "cancelled"
found["cwd_after_cancel"] = os.getcwd() == start

with open(sys.argv[2], "w", encoding="utf-8") as out:
    json.dump(found, out)
"""


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"KEEP_NETGEN_OUTPUT": "1"},
        {"SALOME_NETGEN_DISABLE_MULTITHREADING": "1"},
        {"NGS_NUM_THREADS": "2"},
        {"NGPROFILE": "1"},
    ],
    ids=[
        "no-variable",
        "KEEP_NETGEN_OUTPUT",
        "DISABLE_MULTITHREADING",
        "NGS_NUM_THREADS",
        "NGPROFILE",
    ],
)
def test_a_netgen_compute_writes_no_byte_no_file_and_keeps_the_working_directory(
    tmp_path: Path, env: dict[str, str]
) -> None:
    """A success, a failure and a cancel: the process writes nothing and leaves no file.

    The checks hold for the whole child process, its exit included (NGPROFILE wrote at
    exit), and the working directory is the same after each compute.
    """
    work, temp, results = tmp_path / "work", tmp_path / "temp", tmp_path / "results"
    for folder in (work, temp, results):
        folder.mkdir()
    package_root = str(Path(ps.__file__).resolve().parent.parent)
    child_env = {**os.environ, "TMP": str(temp), "TEMP": str(temp), **env}
    result_file = results / "found.json"

    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, package_root, str(result_file)],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300.0,
        env=child_env,
        check=False,
    )

    found = json.loads(result_file.read_text(encoding="utf-8"))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout == ""
    assert proc.stderr == ""
    assert sorted(p.name for p in work.iterdir()) == []
    assert sorted(p.name for p in temp.iterdir()) == []
    assert found["success"] is True
    assert found["failure"].startswith("PysmeshError: ")
    assert found["cancel"] == "cancelled"
    assert found["cwd_after_success"] and found["cwd_after_failure"]
    assert found["cwd_after_cancel"]


def test_a_netgen_failure_raises_with_netgens_own_reason() -> None:
    """The torus face fails with 3 segments per edge, and NETGEN_3D names what it read.

    NETGEN_3D finds edges that lie on more than two surface triangles and reports them
    in its debug stream ("Edge a - b multiple times in surface mesh",
    ``Mesh::CheckConsistentBoundary``); the plugin reads that text back (``ReadErrors``)
    from the wrapper's memory.
    """
    s = Session()
    s.add_torus(3.0, 1.0)
    shape = ps.load_brep(s.brep())

    with Mesher(shape) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Netgen2D())
        mesher.assign(Netgen3D())
        with pytest.raises(PysmeshError) as raised:
            mesher.compute()

    assert "Some edges multiple times in surface mesh" in raised.value.details
    assert "NETGEN_3D" in raised.value.details
