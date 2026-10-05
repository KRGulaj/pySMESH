# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-05

"""A process that loads pySMESH ends cleanly, with netgen linked statically into _core.

netgen 6.2.2101 registers its archivable classes in a map in ngcore (``archive.cpp``,
``type_register``), and each ``RegisterClassForArchive`` object took its class out of the
map again in its destructor. In netgen's own build ngcore is a separate DLL, loaded first
and unloaded last. Linked statically into ``_core``, netgen's registering objects (one of
them ``regob`` in ``csg/brick.cpp``) are built before ``archive.cpp`` registers the
destructor of the map's owner, so at exit the map is freed first and the later
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
