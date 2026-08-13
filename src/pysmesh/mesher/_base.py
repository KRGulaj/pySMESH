# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""pySMESH mesher — the object every operation group is implemented against.

Part of the :mod:`pysmesh.mesher` package. This holds the one piece of state a mesher has —
the native mesher object — so that each per-area mixin can be written against it without any
of them owning it.
"""

from __future__ import annotations

from .._core import Mesher as _Mesher
from .._core import Shape


class _MesherBase:
    """The native mesher handle, and the only instance attribute a mesher has.

    Declared once here rather than on each mixin: several classes contributing to one
    ``__slots__`` layout would be an instance lay-out conflict, so this owns the slot and
    every operation group derives from it with empty slots of its own.
    """

    __slots__ = ("_m",)

    _m: _Mesher

    def __init__(self, shape: Shape | None = None) -> None:
        """Build a mesher on a shape, or an empty one with no geometry behind it.

        Args:
            shape: The shape to mesh, from :func:`~pysmesh.load_brep` or another loader.
                ``None`` builds a mesher with no shape: an empty mesh to be filled from
                arrays, which is how a discrete body with no B-rep reaches this surface. See
                :attr:`has_shape` for what such a mesher cannot do.
        """
        self._m = _Mesher(shape)

    @property
    def has_shape(self) -> bool:
        """Whether this mesher was built on a shape.

        A mesher without one carries no sub-shape ordinals, so everything that names one
        refuses it: :meth:`~pysmesh.Mesher.compute`, :meth:`~pysmesh.Mesher.assign` and
        :meth:`~pysmesh.Mesher.unassign`, :meth:`~pysmesh.Mesher.add_group_on_shape`, the
        three pattern-mapping calls, ``smooth(in_uv_space=True)``, and the two controls that
        read the geometry — :class:`~pysmesh.ElementsOnShape` and
        :class:`~pysmesh.Deflection2D`. Every other operation is written against the mesh
        alone and works the same either way.

        Returns:
            False for a mesher built with ``shape=None``.
        """
        return bool(self._m.has_shape())

    def release(self) -> None:
        """Free the underlying mesh. Idempotent; the mesher is unusable afterwards."""
        self._m.release()

    def is_open(self) -> bool:
        """Whether the mesher still holds a mesh.

        Returns:
            False once :meth:`release` has run.
        """
        return bool(self._m.is_open())
