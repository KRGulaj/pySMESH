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

# Prefer an installed ``pysmesh`` (the repaired wheel, exercised in CI) over the source tree.
# Only fall back to ``src`` for a local dev build (CMake copies ``_core``/``_build_info`` into
# ``src/pysmesh``). This lets the same test suite validate both layouts.
_repo_src = Path(__file__).resolve().parent.parent / "src"
try:
    import pysmesh  # noqa: F401
except ImportError:
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
    """Raw BREP bytes for the unit box fixture (see ``generate_fixtures.cpp``)."""
    return (fixtures_dir / "box.brep").read_bytes()


@pytest.fixture(scope="session")
def box_mesh(fixtures_dir: Path) -> dict[str, "np.ndarray"]:
    """Classified structured surface mesh of ``box.brep`` (see ``generate_fixtures.cpp``).

    Keys are the ``box_mesh/*.npy`` basenames. Node ids in the arrays are 0-based
    fixture-local indices into ``nodes``; ``face_ids``/``edge_ids``/``vertex_ids`` are
    pySMESH's 1-based TopExp ordinals.
    """
    import numpy as np

    d = fixtures_dir / "box_mesh"
    return {p.stem: np.load(p) for p in d.glob("*.npy")}


@pytest.fixture(scope="session")
def sphere_brep(fixtures_dir: Path) -> bytes:
    """Raw BREP bytes for the unit sphere fixture (curved, see ``generate_fixtures.cpp``)."""
    return (fixtures_dir / "sphere.brep").read_bytes()


@pytest.fixture(scope="session")
def sphere_mesh(fixtures_dir: Path) -> dict[str, "np.ndarray"]:
    """Classified BRepMesh surface mesh of ``sphere.brep`` (doubly-curved, one wall face)."""
    import numpy as np

    d = fixtures_dir / "sphere_mesh"
    return {p.stem: np.load(p) for p in d.glob("*.npy")}
