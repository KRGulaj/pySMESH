"""Shared pytest setup for pySMESH.

Makes the ``pysmesh`` package (with its native ``_core`` extension and generated
``_build_info.py``) importable, and its dynamic dependencies (VTK/OCCT/Boost DLLs from the
host conda env) discoverable on Windows.

The CMake build copies ``_core.pyd`` and ``_build_info.py`` into ``src/pysmesh`` for
in-place import (see the root ``CMakeLists.txt``); here we only need to put the repo's
``src`` directory on ``sys.path``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_repo_src = Path(__file__).resolve().parent.parent / "src"
if str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

_dll_dir = Path(sys.prefix) / "Library" / "bin"
if _dll_dir.is_dir() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_dll_dir))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Directory holding the committed BREP fixtures."""
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def box_brep(fixtures_dir: Path) -> bytes:
    """Raw BREP bytes for the unit box fixture (see ``tests/fixtures/generate.py``)."""
    return (fixtures_dir / "box.brep").read_bytes()
