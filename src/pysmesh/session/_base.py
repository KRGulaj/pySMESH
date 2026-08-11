# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-06

"""pySMESH session — the object every operation group is implemented against.

Part of the :mod:`pysmesh.session` package. This holds the one piece of state a session
has — the native session object — so that each per-area mixin can be written against it
without any of them owning it.
"""

from __future__ import annotations

from .._core import Session as _Session


class _SessionBase:
    """The native session handle, and the only instance attribute a session has.

    Declared once here rather than on each mixin: several classes contributing to one
    ``__slots__`` layout would be an instance lay-out conflict, so this owns the slot and
    every operation group derives from it with empty slots of its own.
    """

    __slots__ = ("_s",)

    _s: _Session

    def __init__(self, *, validate: bool = True) -> None:
        """Create an empty session.

        Args:
            validate: Run ``BRepCheck_Analyzer`` on each operation's result and raise if it
                is invalid, rather than letting a corrupt shape propagate. Only the shape an
                operation actually built is checked, so the cost is per-operation and not
                per-model. Turn it off only for a measured hot path.
        """
        self._s = _Session(validate)
