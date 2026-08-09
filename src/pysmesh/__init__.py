"""pySMESH — standalone SMESH ViscousLayers + OCCT same-domain healing bindings.

Public surface: :func:`load_brep`, :class:`Shape`, :class:`Mesh`, and the per-entity info
types for geometry query and surface-mesh injection; :func:`compute_viscous_layers` (with
:class:`VLParams` / :class:`VLResult` / :class:`ExtrusionMethod`) for boundary-layer prism
generation; :func:`unify_same_domain` (with :class:`UnifyParams` / :class:`UnifyResult`) for
B-rep same-domain face/edge merging; and :class:`PysmeshError` for every library failure, with
:class:`PysmeshCancelled` for the subset a caller stopped.

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
    PysmeshCancelled,
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
from .mesher import (  # noqa: E402 - must follow the VTK check (imports _core)
    Adaptive1D,
    Algorithm,
    Arithmetic1D,
    AutomaticLength,
    Cartesian3D,
    CartesianParameters3D,
    CompositeHexa3D,
    CompositeSegment1D,
    ComputeReport,
    Deflection1D,
    Distribution,
    ElementDimension,
    ElementType,
    FixedPoints1D,
    Geometric1D,
    GmfMesh,
    Hexa3D,
    HexaFromSkin3D,
    Hypothesis,
    LayerDistribution,
    LocalLength,
    MaxElementArea,
    MaxElementVolume,
    MaxLength,
    Mefisto2D,
    MeshData,
    MeshGroup,
    Mesher,
    NumberOfLayers,
    NumberOfLayers2D,
    NumberOfSegments,
    PolygonPerFace2D,
    PolyhedronPerSolid3D,
    Prism3D,
    Projection1D,
    Projection1D2D,
    Projection2D,
    Projection3D,
    ProjectionSource1D,
    ProjectionSource2D,
    ProjectionSource3D,
    Propagation,
    QuadFromMedialAxis1D2D,
    QuadType,
    Quadrangle2D,
    QuadrangleParams,
    QuadranglePreference,
    QuadraticMesh,
    RadialPrism3D,
    RadialQuadrangle1D2D,
    Regular1D,
    SegmentLengthAroundVertex,
    StartEndLength,
    SubMeshCount,
    SubShape,
    SubShapeKind,
    ViscousLayers2D,
    gmf_unwritable_types,
    gmf_writable_group_name,
    read_gmf,
    write_gmf,
)
from .offset import (  # noqa: E402 - must follow the VTK check (imports _core)
    OffsetParams,
    OffsetResult,
    ThickSolidParams,
    ThickSolidResult,
    make_thick_solid,
    offset_shape,
)
from .session import (  # noqa: E402 - must follow the VTK check (imports _core)
    AdjacencyPairs,
    BoundsTable,
    CurvatureTable,
    EntityId,
    EntityKind,
    EntityTable,
    GlueMode,
    Handoff,
    HistoryDelta,
    MassTable,
    Name,
    NameRole,
    Origin,
    Projection,
    RenderMesh,
    Resolution,
    CancelPredicate,
    ProgressCallback,
    ResolutionStatus,
    Session,
    SnapshotMark,
    SurfaceSample,
    TypeTable,
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
    "Adaptive1D",
    "AdjacencyPairs",
    "Algorithm",
    "Arithmetic1D",
    "AutomaticLength",
    "BoundsTable",
    "CancelPredicate",
    "Cartesian3D",
    "CartesianParameters3D",
    "CompositeHexa3D",
    "CompositeSegment1D",
    "ComputeReport",
    "CurvatureTable",
    "Deflection1D",
    "Distribution",
    "EdgeInfo",
    "ElementDimension",
    "ElementType",
    "EntityId",
    "EntityKind",
    "EntityLabel",
    "EntityTable",
    "ExtrusionMethod",
    "FaceInfo",
    "FixedPoints1D",
    "Geometric1D",
    "GlueMode",
    "GmfMesh",
    "Handoff",
    "Hexa3D",
    "HexaFromSkin3D",
    "HistoryDelta",
    "Hypothesis",
    "LayerDistribution",
    "LocalLength",
    "MassTable",
    "MaxElementArea",
    "MaxElementVolume",
    "MaxLength",
    "Mefisto2D",
    "Mesh",
    "MeshData",
    "MeshGroup",
    "Mesher",
    "Name",
    "NameRole",
    "NumberOfLayers",
    "NumberOfLayers2D",
    "NumberOfSegments",
    "OffsetParams",
    "OffsetResult",
    "Origin",
    "PolygonPerFace2D",
    "PolyhedronPerSolid3D",
    "Prism3D",
    "ProgressCallback",
    "Projection",
    "Projection1D",
    "Projection1D2D",
    "Projection2D",
    "Projection3D",
    "ProjectionSource1D",
    "ProjectionSource2D",
    "ProjectionSource3D",
    "Propagation",
    "PysmeshCancelled",
    "PysmeshError",
    "QuadFromMedialAxis1D2D",
    "QuadType",
    "Quadrangle2D",
    "QuadrangleParams",
    "QuadranglePreference",
    "QuadraticMesh",
    "RadialPrism3D",
    "RadialQuadrangle1D2D",
    "Regular1D",
    "RenderMesh",
    "Resolution",
    "ResolutionStatus",
    "SegmentLengthAroundVertex",
    "Session",
    "Shape",
    "ShapeDistanceResult",
    "SnapshotMark",
    "SolidInfo",
    "StartEndLength",
    "StepImport",
    "SubMeshCount",
    "SubShape",
    "SubShapeKind",
    "SurfaceSample",
    "TessellateParams",
    "TessellateResult",
    "ThickSolidParams",
    "ThickSolidResult",
    "TypeTable",
    "UnifyParams",
    "UnifyResult",
    "VLParams",
    "VLResult",
    "VertexInfo",
    "ViscousLayers2D",
    "compute_viscous_layers",
    "free_boundary_edges",
    "gmf_unwritable_types",
    "gmf_writable_group_name",
    "load_brep",
    "make_thick_solid",
    "offset_shape",
    "point_in_solid",
    "read_gmf",
    "read_step_xde",
    "shape_distance",
    "tessellate",
    "unify_same_domain",
    "write_gmf",
    "write_step_xde",
]
