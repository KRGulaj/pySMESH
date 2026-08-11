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

    def __init__(self, shape: Shape) -> None:
        """Build a mesher on a shape.

        Args:
            shape: The shape to mesh, from :func:`~pysmesh.load_brep` or another loader.
        """
        self._m = _Mesher(shape)

    def release(self) -> None:
        """Free the underlying mesh. Idempotent; the mesher is unusable afterwards."""
        self._m.release()

    def is_open(self) -> bool:
        """Whether the mesher still holds a mesh.

        Returns:
            False once :meth:`release` has run.
        """
        return bool(self._m.is_open())
