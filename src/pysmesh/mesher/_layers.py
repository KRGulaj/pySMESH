# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-04

"""pySMESH mesher — viscous layers by the two-step builder.

Part of the :mod:`pysmesh.mesher` package. SMESH's ``StdMeshers_ViscousLayerBuilder`` is
not an algorithm to assign and compute. It works in two steps around a mesh made by any
mesher:

1. :meth:`~_LayerOps.shrink_geometry` offsets the shape inward by the layer thickness on
   the boundary faces and returns the shrunk shape.
2. The caller meshes that shape with any :class:`~pysmesh.Mesher`.
3. :meth:`~_LayerOps.add_layers` clears the first mesher's mesh, copies the inner mesh
   into it, and fills the gap with prism layers (quadrangles in a face).

A stack of ``N`` layers growing by ``f`` and totalling ``T`` has the first layer
``t1 = T (f - 1) / (f^N - 1)``, ``T / N`` at ``f = 1``; layer ``k`` ends at
``t1 (f^k - 1) / (f - 1)`` from the wall.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from ..viscous import _check_layer_stack
from ._base import _MesherBase
from ._types import ComputeReport, _report

if TYPE_CHECKING:
    from .._core import Shape
    from . import Mesher


@dataclass(frozen=True)
class ViscousLayerBuilder:
    """The parameters of the two-step viscous-layer builder.

    The builder offsets faces with OCCT's face offset, the only method upstream
    implements (``GetMethod`` is never read), so it has no method field. In a face (a
    mesher built on a shape with no solid and one face) the whole face shrinks, so the
    layers grow on every edge: there ``boundary`` must stay empty and ``ignore`` True.

    Attributes:
        total_thickness: Total thickness ``T`` of the stack, > 0.
        layer_count: Number of layers ``N``, >= 1.
        stretch_factor: Ratio ``f`` of one layer's thickness to the one before, >= 1.
        boundary: Face ordinals of the mesher's shape: the faces the layers grow on, or
            with ``ignore`` the faces they do not grow on. Empty with ``ignore``: every
            face.
        ignore: Read ``boundary`` as the faces without layers.
        group_name: Name of the volume group the layer cells go into, or empty for no
            group.

    Raises:
        PysmeshError: If ``T <= 0``, ``N < 1`` or ``f < 1``.
    """

    total_thickness: float
    layer_count: int
    stretch_factor: float
    boundary: tuple[int, ...] = ()
    ignore: bool = True
    group_name: str = ""

    def __post_init__(self) -> None:
        """Refuse a stack SMESH would not grow as stated: T > 0, N >= 1, f >= 1."""
        _check_layer_stack(
            "ViscousLayerBuilder",
            self.total_thickness,
            self.layer_count,
            self.stretch_factor,
        )

    def params(self) -> dict[str, object]:
        """The parameter dict the native builder reads.

        Returns:
            One entry per field.
        """
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["boundary"] = list(self.boundary)
        return values


class _LayerOps(_MesherBase):
    """Viscous layers by the two-step builder."""

    __slots__ = ()

    def shrink_geometry(self, builder: ViscousLayerBuilder) -> Shape:
        """Offset this mesher's shape inward by the layer thickness; return the result.

        Call it on a mesher built on the original shape: the builder resolves its
        boundary faces through this mesher's own sub-shape index. Mesh the returned
        shape with any mesher, then call :meth:`add_layers` here with that mesher.

        Args:
            builder: The layer parameters.

        Returns:
            The shrunk shape. Build the inner mesher on this object, or on a copy that
            lists the same sub-shapes in the same order (a BREP round trip does).

        Raises:
            PysmeshError: If the mesher has no shape, if the shape has neither a solid
                nor exactly one face, if ``boundary`` is not empty or ``ignore`` False
                in a face, if an ordinal is out of range, or if SMESH cannot offset the
                shape (its text is in ``.details``).
        """
        return self._m.shrink_geometry(builder.params())

    def add_layers(self, builder: ViscousLayerBuilder, inner: Mesher) -> ComputeReport:
        """Fill this mesher with the inner mesh and the viscous layers around it.

        SMESH clears this mesher's mesh first, so a mesh it held is replaced. The layer
        cells go into the group ``builder.group_name`` when it is set.

        Args:
            builder: The builder given to :meth:`shrink_geometry`, field for field.
            inner: A computed mesher on the shape :meth:`shrink_geometry` returned.

        Returns:
            The counts and the meshed sub-shapes, as :meth:`compute` reports them.

        Raises:
            PysmeshError: If :meth:`shrink_geometry` was not called here first, if
                ``builder`` differs from the one it was given, if ``inner`` is this
                mesher, is on another shape or holds no mesh, or if SMESH cannot build
                the layers (its text is in ``.details``).
        """
        return _report(self._m.add_layers(builder.params(), inner._m))
