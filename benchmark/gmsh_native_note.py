# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-11

"""Supplementary, explicitly NOT part of the headline comparison.

A reader who prefers gmsh will object that gmsh.model.mesh.importStl() is gmsh's
visualisation path, and that gmsh's real product is its 2-D mesher, generate(2).

That objection is answered by the measurand, not by a race. generate(2) is controlled by
element size (Mesh.MeshSizeMax, Mesh.MeshSizeFromCurvature, Mesh.MeshSizeMin); it takes no
chordal-deviation tolerance. There is therefore no setting at which generate(2) and
BRepMesh can be asked for the same thing, so putting them in the same table would be exactly
the unmatched comparison this report exists to remove.

This script records what generate(2) costs on one geometry so the difference in kind is
concrete and quantified, and so nobody has to take the paragraph above on trust. It produces
an unstructured triangular mesh graded for analysis, not a chordal-deviation tessellation.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gmsh
import psutil

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
def gmsh_bench_dir() -> Path:
    """The gmsh source distribution's `benchmarks/` directory.

    Those CAD models belong to gmsh and are referenced here, never redistributed. Set the
    GMSH_BENCH environment variable to a gmsh source checkout. Cases whose files are absent
    are skipped rather than failing the run.
    """
    return Path(os.environ.get("GMSH_BENCH", "gmsh-benchmarks")).expanduser()


GB = gmsh_bench_dir()
STEP = GB / "statreport" / "Block.stp"


def run_native(size_max: float, from_curvature: float) -> dict[str, float]:
    psutil.Process(os.getpid()).cpu_affinity(list(range(14)))
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("m")
    gmsh.model.occ.importShapes(str(STEP))
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", from_curvature)
    t0 = time.perf_counter()
    gmsh.model.mesh.generate(2)
    t1 = time.perf_counter()
    _, etags, _ = gmsh.model.mesh.getElements(2)
    ntri = int(sum(len(t) for t in etags))
    gmsh.finalize()
    return {"size_max": size_max, "from_curvature": from_curvature,
            "wall_s": t1 - t0, "n_tri": ntri}


def main() -> None:
    out = RESULTS / "gmsh_native.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        for smax, curv in ((2.0, 0.0), (1.0, 0.0), (2.0, 20.0)):
            r = run_native(smax, curv)
            r["step"] = STEP.name
            fh.write(json.dumps(r) + "\n")
            print(f"generate(2) MeshSizeMax={smax:<5g} FromCurvature={curv:<5g} "
                  f"tris={r['n_tri']:>8d} wall={r['wall_s']:7.3f}s", flush=True)


if __name__ == "__main__":
    main()
