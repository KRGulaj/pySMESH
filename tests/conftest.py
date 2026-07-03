"""Shared pytest setup for pySMESH2.

Makes the native ``_core`` extension importable and its dynamic dependencies (OCCT + VTK
DLLs, provided by the host conda env) discoverable on Windows:

* ``os.add_dll_directory(sys.prefix/Library/bin)`` — the conda layout where OCCT/VTK DLLs
  live. VTK is deliberately resolved from the host env here, never bundled.
* ``PYSMESH2_BUILD_DIR`` env var — for in-place (pre-install, B1) builds, points at the
  CMake build dir that contains ``_core.pyd``; prepended to ``sys.path`` so tests run
  against the freshly built extension without installing a wheel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_dll_dir = Path(sys.prefix) / "Library" / "bin"
if _dll_dir.is_dir() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_dll_dir))

_build_dir = os.environ.get("PYSMESH2_BUILD_DIR")
if _build_dir and _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)
