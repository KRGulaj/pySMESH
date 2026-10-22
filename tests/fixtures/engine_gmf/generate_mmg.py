# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-22

"""Write the engine-made Inria fixtures of ``tests/test_gmf.py`` with MMG3D.

Dev-only: it needs ``mmgpy`` (MMG's Python binding), which is not a pySMESH dependency, and
it is not run by the test suite. The files it writes are committed next to it.

The input is the box ``[0, 3] x [0, 7] x [0, 11]`` cut into six tetrahedra along its main
diagonal (the Kuhn split), with one surface reference per box face. MMG3D remeshes it to
``hmax = 1.5`` and writes the result with its own writer, as ASCII ``.mesh`` and binary
``.meshb``. The box faces are planar, so the remeshed volume is the box's, 231, and the
boundary area is 262.

Usage:
    python generate_mmg.py <output directory>
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import mmgpy
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

SIZE: Final[tuple[float, float, float]] = (3.0, 7.0, 11.0)
HMAX: Final[float] = 1.5
STEM: Final[str] = "mmg3d_box"


def box_vertices() -> NDArray[np.float64]:
    """The eight corners; corner ``i`` has bit 0 for x, bit 1 for y and bit 2 for z."""
    a, b, c = SIZE
    return np.array(
        [[a * (i & 1), b * ((i >> 1) & 1), c * ((i >> 2) & 1)] for i in range(8)],
        dtype=np.float64,
    )


def kuhn_tetrahedra() -> NDArray[np.int32]:
    """Six positively oriented tetrahedra along the diagonal from corner 0 to corner 7."""
    tets = [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ]
    return np.array(tets, dtype=np.int32)


def boundary_triangles() -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """The twelve boundary triangles of the Kuhn split, and one reference per box face."""
    faces = [
        ([0, 2, 3], 1), ([0, 3, 1], 1),  # z = 0
        ([4, 5, 7], 2), ([4, 7, 6], 2),  # z = c
        ([0, 1, 5], 3), ([0, 5, 4], 3),  # y = 0
        ([2, 6, 7], 4), ([2, 7, 3], 4),  # y = b
        ([0, 4, 6], 5), ([0, 6, 2], 5),  # x = 0
        ([1, 3, 7], 6), ([1, 7, 5], 6),  # x = a
    ]  # fmt: skip
    tris = np.array([f for f, _ in faces], dtype=np.int32)
    refs = np.array([r for _, r in faces], dtype=np.int32)
    return tris, refs


def main(argv: list[str]) -> int:
    """Write ``mmg3d_box.mesh`` and ``mmg3d_box.meshb`` into the directory ``argv[1]``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(argv) != 2:
        logger.error("usage: python generate_mmg.py <output directory>")
        return 2
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)

    vertices = box_vertices()
    tets = kuhn_tetrahedra()
    tris, refs = boundary_triangles()
    # MMG takes the triangles only after their count is declared with the other sizes.
    mesh = mmgpy.MmgMesh3D()
    mesh.set_mesh_size(
        vertices=len(vertices), tetrahedra=len(tets), triangles=len(tris)
    )
    mesh.set_vertices(vertices)
    mesh.set_tetrahedra(tets)
    mesh.set_triangles(tris, refs)
    mesh.remesh(hmax=HMAX, verbose=-1)

    for suffix in (".mesh", ".meshb"):
        path = out / f"{STEM}{suffix}"
        mesh.save(str(path))
        logger.info(
            "MMG %s wrote %s (%d bytes)", mmgpy.MMG_VERSION, path, path.stat().st_size
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
