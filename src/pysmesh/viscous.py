"""Viscous boundary-layer prism generation (Tier-1).

Public surface: :class:`ExtrusionMethod`, :class:`VLParams`, :class:`VLResult`, and
:func:`compute_viscous_layers`. These wrap the low-level ``_core.compute_viscous_layers``
(which returns raw NumPy arrays) in frozen dataclasses and validate parameters up front.

The connectivity arrays (``prism_connectivity``, ``inner_surface_tris``) hold **0-based row
indices into** ``node_coords`` / ``node_ids`` — VTK-ready, so a consumer can build a
``vtkUnstructuredGrid`` directly. ``node_ids`` carries the originating SMESH id per row for
cross-step reconciliation (e.g. deduplicating the VL/interior-fill interface by identity).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._core import Mesh, PysmeshError
from ._core import compute_viscous_layers as _compute_viscous_layers


class ExtrusionMethod(IntEnum):
    """Layer-extrusion strategy — mirrors ``StdMeshers_ViscousLayers::ExtrusionMethod``.

    The integer values are persisted by SMESH (``SaveTo``/``LoadFrom``); do not reorder.
    """

    SURF_OFFSET_SMOOTH = 0
    FACE_OFFSET = 1
    NODE_OFFSET = 2


@dataclass(frozen=True)
class VLParams:
    """Viscous-layer parameters.

    Attributes:
        face_ids: Wall face ids (1-based, from :meth:`Shape.faces`). If ``is_ignore`` is
            True these are instead the faces to *exclude* (layers grow on all others).
        total_thickness: Total layer stack thickness T [m] (T > 0). The caller converts
            from first-cell height via ``T = dy1 * (g**N - 1) / (g - 1)``.
        n_layers: Number of layers N (N >= 1).
        stretch_factor: Geometric growth ratio g between consecutive layers (g > 1).
        is_ignore: If True, ``face_ids`` is the excluded set rather than the wall set.
        method: Extrusion strategy.
        group_name: Non-empty name of the SMESH group collecting the layer prisms; prism
            harvest depends on it.

    Raises:
        PysmeshError: On any out-of-range or empty parameter.
    """

    face_ids: tuple[int, ...]
    total_thickness: float
    n_layers: int
    stretch_factor: float
    is_ignore: bool = False
    method: ExtrusionMethod = ExtrusionMethod.SURF_OFFSET_SMOOTH
    group_name: str = "BL"

    def __post_init__(self) -> None:
        if len(self.face_ids) == 0:
            raise PysmeshError("VLParams.face_ids must not be empty.")
        if any(fid < 1 for fid in self.face_ids):
            raise PysmeshError("VLParams.face_ids must be 1-based positive ids.")
        if not self.total_thickness > 0.0:
            raise PysmeshError(
                f"VLParams.total_thickness must be > 0 (got {self.total_thickness})."
            )
        if self.n_layers < 1:
            raise PysmeshError(f"VLParams.n_layers must be >= 1 (got {self.n_layers}).")
        if not self.stretch_factor > 1.0:
            raise PysmeshError(
                f"VLParams.stretch_factor must be > 1.0 (got {self.stretch_factor})."
            )
        if not self.group_name:
            raise PysmeshError("VLParams.group_name must be non-empty.")


@dataclass(frozen=True)
class VLResult:
    """Result of :func:`compute_viscous_layers`.

    Attributes:
        prism_connectivity: (K, 6) int32 — row indices into ``node_coords``, VTK wedge order.
        node_coords: (P, 3) float64 — every node after the compute.
        node_ids: (P,) int64 — SMESH id per row of ``node_coords``.
        inner_surface_tris: (S, 3) int32 — row indices, the shrunk proxy surface.
        inner_surface_face_map: (S,) int32 — source wall face_id per proxy triangle.
        failed_face_ids: Wall faces that received no layers (no proxy sub-mesh).
        warnings: Non-fatal per-solid messages surfaced by SMESH.
    """

    prism_connectivity: NDArray[np.int32]
    node_coords: NDArray[np.float64]
    node_ids: NDArray[np.int64]
    inner_surface_tris: NDArray[np.int32]
    inner_surface_face_map: NDArray[np.int32]
    failed_face_ids: tuple[int, ...]
    warnings: tuple[str, ...]


def compute_viscous_layers(mesh: Mesh, params: VLParams) -> VLResult:
    """Grow viscous prism layers on an injected surface mesh.

    Args:
        mesh: A :class:`Mesh` carrying an injected, classified surface mesh on a solid.
        params: Validated viscous-layer parameters.

    Returns:
        The prisms, full node table, shrunk inner surface, and per-face failure list.

    Raises:
        PysmeshError: If the shape has no solid, or SMESH reports a hard failure (its
            per-solid ``SMESH_ComputeError`` text is attached as ``.details``).
    """
    raw = _compute_viscous_layers(
        mesh,
        list(params.face_ids),
        params.is_ignore,
        params.total_thickness,
        params.n_layers,
        params.stretch_factor,
        int(params.method),
        params.group_name,
    )
    # _core returns validated NumPy arrays / lists (see viscous.cpp); the dict is typed as
    # dict[str, object], so narrow each field here.
    return VLResult(
        prism_connectivity=cast("NDArray[np.int32]", raw["prism_connectivity"]),
        node_coords=cast("NDArray[np.float64]", raw["node_coords"]),
        node_ids=cast("NDArray[np.int64]", raw["node_ids"]),
        inner_surface_tris=cast("NDArray[np.int32]", raw["inner_surface_tris"]),
        inner_surface_face_map=cast("NDArray[np.int32]", raw["inner_surface_face_map"]),
        failed_face_ids=tuple(cast("list[int]", raw["failed_face_ids"])),
        warnings=tuple(cast("list[str]", raw["warnings"])),
    )
