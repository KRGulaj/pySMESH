# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""pySMESH mesher — Inria ``.mesh`` / ``.meshb`` interchange.

Part of the :mod:`pysmesh.mesher` package. The format is what MMG and fTetWild read and
write, and SMESH's driver for it is already compiled into this wheel, so this is a binding
rather than a port.

What it is not is a lossless container, and the three limits are measured rather than
assumed. They are stated on the two functions below because each one changes what a caller
can do with the file.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from .._core import read_gmf as _read_gmf
from .._core import write_gmf as _write_gmf
from ._types import (
    GMF_REQUIRED_MARKER,
    GMF_WRITABLE_TYPES,
    ElementType,
    MeshData,
    MeshGroup,
    _groups,
    _mesh_data,
)


@dataclass(frozen=True)
class GmfMesh:
    """A mesh read from an Inria file.

    Attributes:
        mesh: The nodes and elements. Every entity reports itself bound to nothing: the
            format does carry a per-element reference, and the writer emits the sub-shape
            index there, but the reader discards it — so a file cannot restore a CAD binding
            even when the file it came from had one.
        groups: The groups the file carried. Only two kinds exist in the format: the
            "required entity" sets, read back under the names ``_required_Vertices``,
            ``_required_Edges``, ``_required_Triangles`` and ``_required_Quadrilaterals``,
            and the ``Fault_*`` sets an upstream mesher may have written.
    """

    mesh: MeshData
    groups: tuple[MeshGroup, ...]


def read_gmf(path: str | os.PathLike[str]) -> GmfMesh:
    """Read an Inria ``.mesh`` or ``.meshb`` file.

    The extension selects the format: ``.mesh`` is text and ``.meshb`` is binary. Both are
    read by the same code.

    Args:
        path: File to read.

    Returns:
        The mesh and its groups.

    Raises:
        PysmeshError: If the file cannot be read, is empty, or holds data the driver had to
            skip. The message names which of those it was.
    """
    raw = _read_gmf(os.fspath(path))
    return GmfMesh(
        mesh=_mesh_data(cast("dict[str, object]", raw["mesh"])),
        groups=_groups(cast("Sequence[object]", raw["groups"])),
    )


def write_gmf(
    path: str | os.PathLike[str],
    mesh: MeshData,
    groups: Sequence[MeshGroup] = (),
) -> None:
    """Write a mesh to an Inria ``.mesh`` or ``.meshb`` file.

    The extension selects the format: ``.mesh`` is text and ``.meshb`` is binary.

    Two things are refused rather than silently dropped, because a file that is quietly
    missing part of a mesh is worse than no file:

    * **An element the format cannot represent.** The format has keywords for edges,
      triangles, quadrangles, tetrahedra, pyramids, hexahedra and prisms only. Every polygon
      and polyhedron is outside it — which means a body-fitted Cartesian mesh cannot be
      written at all, since its cut cells are polyhedra — and so are the quadratic pyramid
      and prism, the hexagonal prism, balls and 0-D elements.
    * **A group the format cannot carry.** Only "required entity" groups exist in it, keyed
      on a name containing ``_required_``. A group named anything else would not appear in
      the file.

    Note that mesh ids are not preserved across a round trip: the format numbers elements
    per type, so reading back gives a mesh with the same cells under different ids.

    Args:
        path: File to write.
        mesh: The mesh to write.
        groups: Groups to write. Each name must contain ``_required_``; leave the sequence
            empty to write none.

    Raises:
        PysmeshError: If the mesh holds an element or a group the format cannot represent,
            if the arrays disagree with one another, or if the file cannot be written. The
            message names the offending element or group.
    """
    payload = {
        "node_coords": mesh.node_coords,
        "node_id": mesh.node_id,
        "element_offsets": mesh.element_offsets,
        "element_nodes": mesh.element_nodes,
        "element_type": mesh.element_type,
        "element_id": mesh.element_id,
    }
    encoded: list[object] = [
        (g.name, int(g.dimension), g.element_ids) for g in groups
    ]
    _write_gmf(os.fspath(path), payload, encoded)


def gmf_unwritable_types(mesh: MeshData) -> tuple[ElementType, ...]:
    """Which of a mesh's element types the Inria format cannot represent.

    Answering this before calling :func:`write_gmf` is the difference between choosing
    another route and catching an exception.

    Args:
        mesh: The mesh to inspect.

    Returns:
        The distinct offending types, ascending. Empty when the mesh is writable.
    """
    present = {ElementType(int(code)) for code in set(mesh.element_type.tolist())}
    return tuple(sorted(present - GMF_WRITABLE_TYPES))


def gmf_writable_group_name(suffix: str) -> str:
    """Build a group name the Inria format will carry.

    Args:
        suffix: One of ``Vertices``, ``Edges``, ``Triangles`` or ``Quadrilaterals``.

    Returns:
        The name to give the group.

    Raises:
        ValueError: If ``suffix`` is not one the format defines.
    """
    allowed = ("Vertices", "Edges", "Triangles", "Quadrilaterals")
    if suffix not in allowed:
        raise ValueError(
            f"the Inria format defines required-entity sets for {allowed}, not {suffix!r}."
        )
    return f"{GMF_REQUIRED_MARKER}{suffix}"
