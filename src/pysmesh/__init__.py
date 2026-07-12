"""pySMESH — standalone SMESH ViscousLayers + OCCT same-domain healing bindings.

Public surface: :func:`load_brep`, :class:`Shape`, :class:`Mesh`, and the per-entity info
types for geometry query and surface-mesh injection; :func:`compute_viscous_layers` (with
:class:`VLParams` / :class:`VLResult` / :class:`ExtrusionMethod`) for boundary-layer prism
generation; :func:`unify_same_domain` (with :class:`UnifyParams` / :class:`UnifyResult`) for
B-rep same-domain face/edge merging; and :class:`PysmeshError` for every library failure.

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
    SolidInfo,
    VertexInfo,
    load_brep,
)
from .classify import (  # noqa: E402 - must follow the VTK check (imports _core)
    point_in_solid,
)
from .distance import (  # noqa: E402 - must follow the VTK check (imports _core)
    ShapeDistanceResult,
    free_boundary_edges,
    shape_distance,
)
from .offset import (  # noqa: E402 - must follow the VTK check (imports _core)
    OffsetParams,
    OffsetResult,
    ThickSolidParams,
    ThickSolidResult,
    make_thick_solid,
    offset_shape,
)
from .step import (  # noqa: E402 - must follow the VTK check (imports _core)
    EntityLabel,
    StepImport,
    read_step_xde,
    write_step_xde,
)
from .tessellate import (  # noqa: E402 - must follow the VTK check (imports _core)
    TessellateParams,
    TessellateResult,
    tessellate,
)
from .unify import (  # noqa: E402 - must follow the VTK check (imports _core)
    UnifyParams,
    UnifyResult,
    unify_same_domain,
)
from .viscous import (  # noqa: E402 - must follow the VTK check (imports _core)
    ExtrusionMethod,
    VLParams,
    VLResult,
    compute_viscous_layers,
)

__all__ = [
    "EdgeInfo",
    "EntityLabel",
    "ExtrusionMethod",
    "FaceInfo",
    "Mesh",
    "OffsetParams",
    "OffsetResult",
    "PysmeshError",
    "Shape",
    "ShapeDistanceResult",
    "SolidInfo",
    "StepImport",
    "TessellateParams",
    "TessellateResult",
    "ThickSolidParams",
    "ThickSolidResult",
    "UnifyParams",
    "UnifyResult",
    "VLParams",
    "VLResult",
    "VertexInfo",
    "compute_viscous_layers",
    "free_boundary_edges",
    "load_brep",
    "make_thick_solid",
    "offset_shape",
    "point_in_solid",
    "read_step_xde",
    "shape_distance",
    "tessellate",
    "unify_same_domain",
    "write_step_xde",
]
