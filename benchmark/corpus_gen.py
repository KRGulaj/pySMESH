# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-11

"""Generate the synthetic face-count ladder.

N identical unit spheres on a cubic lattice, exported as one STEP file. Face count is
exactly N and the per-face tessellation work is identical across the ladder, so any change
in the gmsh/pySMESH time ratio with N isolates per-face dispatch cost from per-face work.

Spheres are spaced 3 R apart so no two touch; no boolean is performed.
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
CORPUS.mkdir(exist_ok=True)

RADIUS: float = 10.0
PITCH: float = 30.0
LADDER: tuple[int, ...] = (1, 10, 100, 1000, 5000)


def sphere_grid(n: int, path: Path) -> None:
    """Write a STEP file holding exactly n disjoint spheres (n faces)."""
    side = math.ceil(n ** (1.0 / 3.0))
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"spheres_{n}")
    made = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if made >= n:
                    break
                gmsh.model.occ.addSphere(i * PITCH, j * PITCH, k * PITCH, RADIUS)
                made += 1
    gmsh.model.occ.synchronize()
    nf = len(gmsh.model.getEntities(2))
    gmsh.write(str(path))
    gmsh.finalize()
    print(f"  {path.name:<28s} faces={nf:<6d} {path.stat().st_size / 1e6:>8.2f} MB")


CONE_R1: float = 20.0
CONE_R2: float = 5.0
CONE_H: float = 40.0


def truncated_cone(path: Path) -> None:
    """A truncated cone: one tapering lateral face plus two planar caps.

    Chosen because a tapering surface is where single-point curvature sampling
    under-reports, and because the exact distance from a point to the lateral surface has a
    closed form, so achieved chordal deviation needs no projection library.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cone")
    gmsh.model.occ.addCone(0.0, 0.0, 0.0, 0.0, 0.0, CONE_H, CONE_R1, CONE_R2)
    gmsh.model.occ.synchronize()
    nf = len(gmsh.model.getEntities(2))
    gmsh.write(str(path))
    gmsh.finalize()
    print(f"  {path.name:<28s} faces={nf:<6d} {path.stat().st_size / 1e6:>8.2f} MB")


def main() -> None:
    print("synthetic sphere ladder:")
    for n in LADDER:
        out = CORPUS / f"spheres_{n:05d}.step"
        if out.exists():
            print(f"  {out.name:<28s} (exists)")
            continue
        sphere_grid(n, out)
    print("tapering surface:")
    cone = CORPUS / "cone.step"
    if cone.exists():
        print(f"  {cone.name:<28s} (exists)")
    else:
        truncated_cone(cone)


if __name__ == "__main__":
    main()
