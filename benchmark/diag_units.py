# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-11

"""Diagnose the triangle-count mismatch on three real STEP bodies.

Hypothesis: the two importers disagree on the model's length unit, so one absolute chordal
deflection is not the same physical tolerance on both sides. Compares the bounding box of
the shape as each library actually loaded it.
"""

from __future__ import annotations

import os
from pathlib import Path

import gmsh
import numpy as np

import pysmesh

def gmsh_bench_dir() -> Path:
    """The gmsh source distribution's `benchmarks/` directory.

    Those CAD models belong to gmsh and are referenced here, never redistributed. Set the
    GMSH_BENCH environment variable to a gmsh source checkout. Cases whose files are absent
    are skipped rather than failing the run.
    """
    return Path(os.environ.get("GMSH_BENCH", "gmsh-benchmarks")).expanduser()


GB = gmsh_bench_dir()
FILES = [
    GB / "step" / "part.step",
    GB / "step" / "U_Joint_2.stp",
    GB / "statreport" / "917_fusee.stp",
    GB / "statreport" / "Block.stp",
    GB / "statreport" / "Kurbelwelle.stp",
    GB / "step" / "wrenchnut.stp",
]


def gmsh_bbox(p: Path) -> tuple[float, float]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("m")
    gmsh.model.occ.importShapes(str(p))
    gmsh.model.occ.synchronize()
    bb = gmsh.model.getBoundingBox(-1, -1)
    nf = len(gmsh.model.getEntities(2))
    gmsh.finalize()
    diag = sum((bb[i + 3] - bb[i]) ** 2 for i in range(3)) ** 0.5
    return diag, nf


def pysmesh_bbox(p: Path) -> tuple[float, float, str]:
    imp = pysmesh.read_step_xde(str(p))
    s = pysmesh.Session()
    s.add_brep(imp.brep)
    rm = s.tessellate(deflection=1e9, angle_deg=60.0, relative=False, incremental=False)
    n = rm.nodes
    diag = float(np.linalg.norm(n.max(axis=0) - n.min(axis=0)))
    nf = len(s.entities("FACE"))
    unit = getattr(imp, "unit", "?")
    return diag, nf, str(unit)


def main() -> None:
    print(f"{'file':<28s} {'gmsh diag':>12s} {'pysmesh diag':>13s} {'ratio':>8s} "
          f"{'gf':>5s} {'pf':>5s}  unit")
    for p in FILES:
        if not p.exists():
            print(f"{p.name:<28s} MISSING")
            continue
        gd, gnf = gmsh_bbox(p)
        pd, pnf, unit = pysmesh_bbox(p)
        print(f"{p.name:<28s} {gd:>12.4f} {pd:>13.4f} {pd / gd:>8.4f} "
              f"{gnf:>5d} {pnf:>5d}  {unit}")


if __name__ == "__main__":
    main()
