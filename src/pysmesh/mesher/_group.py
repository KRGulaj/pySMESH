# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""pySMESH mesher — named groups of elements and nodes.

Part of the :mod:`pysmesh.mesher` package. A group is a named set of mesh entities that the
**mesher** maintains, rather than one a consumer re-derives after the fact — and that
difference is the whole of its value. SMESH's own editing operations call into the groups as
they work, so a group defined on a coarse mesh still names the right cells after the mesh has
been converted to second order, split, or merged. Re-deriving membership from geometry after
each of those steps is what this replaces, and it is the step that goes wrong.

Three kinds exist and they differ in what maintains the membership:

======================================  ==================================================
:attr:`~pysmesh.GroupSource.EXPLICIT`   an id list, carried through editing by SMESH itself
:attr:`~pysmesh.GroupSource.SHAPE`      everything the mesher bound to one sub-shape
:attr:`~pysmesh.GroupSource.FILTER`     everything a predicate accepts, re-evaluated
======================================  ==================================================

Only the first can be edited by hand. The other two are defined by their source, and an
attempt to add to one is refused naming which source governs it rather than silently ignored.

Names address a group here. SMESH itself allows two groups of one name and addresses them by
an integer id it also uses for persistence; a duplicate name is refused at creation so that
every later call means exactly one group.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

from .._core import PysmeshError
from ._base import _MesherBase
from ._controls import Predicate
from ._types import ElementDimension, MeshGroup, SubShape, _groups


class _GroupOps(_MesherBase):
    """Creating, reading, editing and removing groups."""

    __slots__ = ()

    # ---- Reading ---------------------------------------------------------------------- #

    def groups(self) -> tuple[MeshGroup, ...]:
        """Every group on the mesh, with its membership as of now.

        Algorithms create groups of their own where they have something to name — a viscous
        layer stack collects its cells into the group its hypothesis names.

        Returns:
            One entry per group.
        """
        return _groups(cast("Sequence[object]", self._m.groups()))

    def group(self, name: str) -> MeshGroup:
        """One group by name.

        Args:
            name: The group's name.

        Returns:
            The group and its membership as of now.

        Raises:
            PysmeshError: If the mesh has no group of that name.
        """
        for entry in self.groups():
            if entry.name == name:
                return entry
        raise PysmeshError(f"Mesher.group: the mesh has no group named {name!r}.")

    def group_names(self) -> tuple[str, ...]:
        """The names of every group on the mesh.

        Returns:
            One name per group, in the mesh's own order.
        """
        return tuple(entry.name for entry in self.groups())

    # ---- Creating --------------------------------------------------------------------- #

    def add_group(
        self, name: str, family: ElementDimension, ids: Iterable[int] = ()
    ) -> None:
        """Create a group from an explicit list of mesh ids.

        This is the kind SMESH carries through editing: converting to second order, splitting
        a cell or merging nodes all update the membership rather than leaving it stale.

        Args:
            name: A name unique within this mesher.
            family: The element family the group holds — ``NODE``, ``EDGE``, ``FACE``,
                ``VOLUME``, ``ELEM_0D`` or ``BALL``.
            ids: The mesh ids to start with. More can be added later.

        Raises:
            PysmeshError: If the name is empty or already taken, or an id names nothing of
                that family in the mesh.
        """
        self._m.add_group(name, int(family), [int(i) for i in ids])

    def add_group_on_shape(
        self, name: str, family: ElementDimension, on: SubShape
    ) -> None:
        """Create a group holding everything the mesher bound to one sub-shape.

        The membership follows the mesh: re-computing after a finer hypothesis gives the
        group the new elements without any call.

        Args:
            name: A name unique within this mesher.
            family: The element family the group holds.
            on: The sub-shape whose elements it names.

        Raises:
            PysmeshError: If the name is taken, or the sub-shape is not part of the shape.
        """
        self._m.add_group_on_shape(name, int(family), on.kind.name, on.ordinal)

    def add_group_on_filter(
        self, name: str, family: ElementDimension, predicate: Predicate
    ) -> None:
        """Create a group holding everything a predicate accepts.

        The membership is re-evaluated whenever the mesh has changed since it was last read,
        so a group of, say, every inverted cell stays right across an edit rather than
        recording one moment.

        Args:
            name: A name unique within this mesher.
            family: The element family the group holds. It should match the predicate's own
                family, or the group holds nothing.
            predicate: The test, which may be a composed one.

        Raises:
            PysmeshError: If the name is taken, or the predicate is unknown or names a group
                that does not exist.
        """
        self._m.add_group_on_filter(
            name, int(family), predicate.native_name, predicate.params()
        )

    # ---- Editing and removing ---------------------------------------------------------- #

    def add_to_group(self, name: str, ids: Iterable[int]) -> None:
        """Add mesh ids to an explicit group.

        Args:
            name: The group's name.
            ids: The mesh ids to add. An id already in the group is accepted.

        Raises:
            PysmeshError: If there is no such group, if it is defined by a sub-shape or a
                filter rather than by an id list, or if an id names nothing of its family.
        """
        self._m.edit_group(name, [int(i) for i in ids], True)

    def remove_from_group(self, name: str, ids: Iterable[int]) -> None:
        """Remove mesh ids from an explicit group.

        Args:
            name: The group's name.
            ids: The mesh ids to remove.

        Raises:
            PysmeshError: If there is no such group, if it is defined by a sub-shape or a
                filter, or if an id is not in it.
        """
        self._m.edit_group(name, [int(i) for i in ids], False)

    def remove_group(self, name: str) -> None:
        """Delete a group. The elements themselves are untouched.

        Args:
            name: The group's name.

        Raises:
            PysmeshError: If the mesh has no group of that name.
        """
        self._m.remove_group(name)
