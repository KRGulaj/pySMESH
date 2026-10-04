# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-07-03

"""Viscous boundary-layer prism generation (Tier-1).

Public surface: :class:`ExtrusionMethod`, :class:`VLParams`, :class:`VLResult`,
:func:`compute_viscous_layers` and :func:`first_layer_thickness`. These wrap the
low-level ``_core`` functions (which return raw NumPy arrays) in frozen dataclasses and
validate parameters up front.

A stack of ``N`` layers growing by the factor ``f`` and totalling ``T`` has the first
layer ``t1 = T (f - 1) / (f^N - 1)``, or ``T / N`` when ``f = 1``, and layer ``k`` ends
at ``t1 (f^k - 1) / (f - 1)`` from the wall (``k t1`` when ``f = 1``).

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
from ._core import first_layer_thickness as _first_layer_thickness


def _check_layer_stack(
    owner: str, total_thickness: float, layer_count: int, stretch_factor: float
) -> None:
    """Refuse a layer stack that SMESH would not grow as stated (report L2).

    Upstream checks none of the three (``StdMeshers_ViscousLayers.cxx:1304-1339``): a
    factor below 1 reads as uniform layers (``Get1stLayerThickness`` answers ``T / N``
    for any ``f^N - 1 <= 0``), and a thickness or count of 0 builds nothing.
    """
    if not total_thickness > 0.0:
        raise PysmeshError(
            f"{owner}: total_thickness must be > 0 (got {total_thickness})."
        )
    if layer_count < 1:
        raise PysmeshError(f"{owner}: layer_count must be >= 1 (got {layer_count}).")
    if not stretch_factor >= 1.0:
        raise PysmeshError(
            f"{owner}: stretch_factor must be >= 1, 1 for uniform layers "
            f"(got {stretch_factor})."
        )


def first_layer_thickness(
    total_thickness: float, stretch_factor: float, layer_count: int
) -> float:
    """The thickness of the first layer of a stack, at the wall.

    ``t1 = T (f - 1) / (f^N - 1)``, or ``T / N`` when ``f = 1``
    (``StdMeshers_ViscousLayers::Get1stLayerThickness``). This is the inverse a caller
    with a target first-cell height needs before choosing ``T``.

    Args:
        total_thickness: Total thickness ``T`` of the stack, > 0.
        stretch_factor: Ratio ``f`` of one layer's thickness to the one before, >= 1.
        layer_count: Number of layers ``N``, >= 1.

    Returns:
        The first layer's thickness.

    Raises:
        PysmeshError: If ``T <= 0``, ``N < 1``, ``f < 1``, or a value is not finite.
    """
    _check_layer_stack(
        "first_layer_thickness", total_thickness, layer_count, stretch_factor
    )
    return float(_first_layer_thickness(total_thickness, stretch_factor, layer_count))


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
        stretch_factor: Geometric growth ratio g between consecutive layers (g >= 1;
            1 gives layers of equal thickness).
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
        _check_layer_stack(
            "VLParams", self.total_thickness, self.n_layers, self.stretch_factor
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
