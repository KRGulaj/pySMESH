"""pySMESH — standalone SMESH ViscousLayers bindings.

Public surface (Tier-1): :func:`load_brep`, :class:`Shape`, :class:`Mesh`, the per-entity
info types, and :class:`PysmeshError`. Viscous-layer bindings arrive in a later release.

Import-time contract: ``_core`` links VTK dynamically against whatever VTK the host
process provides. The build was compiled against a specific VTK version; importing into an
environment with a different VTK is an ABI hazard, so the version is hard-checked here and
raises :class:`ImportError` on mismatch rather than risking a silent crash.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import _build_info

# --- Locate the dynamic dependencies (VTK/OCCT/Boost DLLs live in the conda env) ------- #
# conda's Python already has its Library/bin on the DLL search path; this is belt-and-
# suspenders for embedded / Nuitka hosts. It never adds a second VTK — only makes the
# host's own DLLs findable.
_dll_dir = Path(sys.prefix) / "Library" / "bin"
if _dll_dir.is_dir() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_dll_dir))


def _check_vtk_version() -> None:
    """Fail loudly if the host VTK differs from the one ``_core`` was built against."""
    try:
        # VTK ships no py.typed marker; the untyped-import ignore is expected and honest.
        import vtk  # type: ignore[import-untyped]  # noqa: PLC0415 - lazy host dependency
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "pysmesh requires VTK "
            f"{_build_info.VTK_VERSION} in the host environment, but VTK is not "
            "importable. Install the matching conda-forge vtk build."
        ) from exc

    host = vtk.VTK_VERSION
    if host != _build_info.VTK_VERSION:
        raise ImportError(
            "pysmesh was built against VTK "
            f"{_build_info.VTK_VERSION} but the host environment provides VTK {host}. "
            "These share an ABI-bound datastructure (vtkUnstructuredGrid); rebuild "
            "pysmesh against the host VTK or align the versions."
        )


_check_vtk_version()

from ._core import (  # noqa: E402 - must follow the VTK check
    EdgeInfo,
    FaceInfo,
    Mesh,
    PysmeshError,
    Shape,
    VertexInfo,
    load_brep,
)
from .viscous import (  # noqa: E402 - must follow the VTK check (imports _core)
    ExtrusionMethod,
    VLParams,
    VLResult,
    compute_viscous_layers,
)

__all__ = [
    "EdgeInfo",
    "ExtrusionMethod",
    "FaceInfo",
    "Mesh",
    "PysmeshError",
    "Shape",
    "VLParams",
    "VLResult",
    "VertexInfo",
    "compute_viscous_layers",
    "load_brep",
]
