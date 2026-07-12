"""B-rep tessellation for viewport rendering (Tier-2).

Public surface: :class:`TessellateParams`, :class:`TessellateResult`, and :func:`tessellate`.
These wrap the low-level ``_core.tessellate`` (which returns a raw dict of NumPy arrays) in
frozen dataclasses and validate parameters up front.

``tessellate`` drives OCCT's ``BRepMesh_IncrementalMesh`` to produce a lightweight
triangulated render mesh. The GIL is released for the mesh computation. Each face contributes
its own node range — nodes at face boundaries are **not** welded — which gives hard edges at
face seams (correct for B-rep topology) and smooth shading within curved patches (from the
surface normal at each node's UV). Output face ids in ``tri_face_id`` match
:meth:`Shape.faces` 1-based TopExp ordinals, so the result composes directly onto Gmsh tags
or any system that uses pySMESH's id convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._core import PysmeshError
from ._core import tessellate as _tessellate

# Defaults matching OCCT's recommended "display quality" range for interactive viewports.
# Relative mode (relative=True) with lin_defl=0.005 gives a mesh that tracks the geometry
# to 0.5 % of the bounding-box diagonal — a good balance for large B-rep assemblies.
_DEFAULT_LIN_DEFL: float = 0.1
_DEFAULT_ANG_DEFL_DEG: float = 20.0


@dataclass(frozen=True)
class TessellateParams:
    """Tessellation quality parameters for :func:`tessellate`.

    Attributes:
        lin_defl: Chord deflection in model units (absolute mode) or as a fraction of the
            bounding-box diagonal (relative mode). Controls how far a mesh edge may deviate
            from the true curve. Must be > 0.
        ang_defl_deg: Maximum angular deflection per mesh edge in degrees. Governs curvature
            sampling; smaller values yield finer meshes on tightly curved geometry. Must be
            in (0, 180).
        relative: If ``True``, ``lin_defl`` is interpreted as a fraction of the shape's
            bounding-box diagonal rather than an absolute model-unit distance.

    Raises:
        PysmeshError: On any out-of-range parameter.
    """

    lin_defl: float = _DEFAULT_LIN_DEFL
    ang_defl_deg: float = _DEFAULT_ANG_DEFL_DEG
    relative: bool = False

    def __post_init__(self) -> None:
        if not self.lin_defl > 0.0:
            raise PysmeshError(
                f"TessellateParams.lin_defl must be > 0 (got {self.lin_defl})."
            )
        if not (0.0 < self.ang_defl_deg < 180.0):
            raise PysmeshError(
                f"TessellateParams.ang_defl_deg must be in (0, 180) "
                f"(got {self.ang_defl_deg})."
            )


@dataclass(frozen=True)
class TessellateResult:
    """Result of :func:`tessellate`.

    Attributes:
        nodes: ``(N, 3)`` float64 — world-space XYZ of each mesh node.
        tris: ``(M, 3)`` int32 — triangle connectivity; 0-based indices into ``nodes``.
        tri_face_id: ``(M,)`` int32 — 1-based face id (matching :meth:`Shape.faces`) per
            triangle. Composes directly onto Gmsh tags.
        normals: ``(N, 3)`` float64 — world-space outward unit normals per node, evaluated
            from the underlying Geom_Surface at each node's UV coordinates via
            ``GeomLProp_SLProps``. Zero vector at degenerate surface points (poles, singular
            UV); callers that need a normal there should fall back to face-normal averaging.
    """

    nodes: NDArray[np.float64]
    tris: NDArray[np.int32]
    tri_face_id: NDArray[np.int32]
    normals: NDArray[np.float64]


def tessellate(brep: bytes, params: TessellateParams | None = None) -> TessellateResult:
    """Tessellate a BREP shape into a triangulated render mesh.

    Drives OCCT's ``BRepMesh_IncrementalMesh`` then harvests the per-face
    ``Poly_Triangulation`` into flat NumPy arrays. Nodes at face boundaries are not welded —
    each face contributes its own node range — so hard edges at face seams and smooth shading
    within curved patches are both correct without any post-processing.

    The GIL is released for ``BRepMesh_IncrementalMesh::Perform()``.

    Args:
        brep: Input shape as BREP bytes (e.g. from :func:`load_brep`'s source, or any OCCT
            ``BRepTools::Write`` output, including :attr:`UnifyResult.brep`).
        params: Tessellation parameters. Defaults to :class:`TessellateParams` (absolute
            chord deflection 0.1, angular deflection 20°).

    Returns:
        nodes ``(N,3)`` float64, tris ``(M,3)`` int32 (0-based indices), tri_face_id
        ``(M,)`` int32 (1-based), normals ``(N,3)`` float64.

    Raises:
        PysmeshError: On a malformed BREP or invalid parameters.
    """
    p = params if params is not None else TessellateParams()
    raw = _tessellate(
        brep,
        p.lin_defl,
        math.radians(p.ang_defl_deg),
        p.relative,
    )
    return TessellateResult(
        nodes=cast("NDArray[np.float64]", raw["nodes"]),
        tris=cast("NDArray[np.int32]", raw["tris"]),
        tri_face_id=cast("NDArray[np.int32]", raw["tri_face_id"]),
        normals=cast("NDArray[np.float64]", raw["normals"]),
    )
