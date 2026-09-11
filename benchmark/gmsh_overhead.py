# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-11

"""Isolate gmsh's BRepMesh time from its mesh-construction overhead.

gmsh::model::mesh::importStl() is, per face (gmsh.cpp:5863):
    buildSTLTriangulation()   -> OCCT BRepMesh_IncrementalMesh, the tessellation proper
    storeSTLAsMesh()          -> allocation of MVertex / MTriangle objects

A second importStl() on the same model calls deleteMesh(), then hits the STL cache in
buildSTLTriangulation (GFace.cpp:1558 returns early when stl_triangles is non-empty) and
re-runs storeSTLAsMesh only. The second call therefore measures gmsh's conversion cost
alone, which lets the report state how much of gmsh's time is tessellation and how much is
gmsh's own data structures.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gmsh
import psutil

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
RESULTS = HERE / "results"
def gmsh_bench_dir() -> Path:
    """The gmsh source distribution's `benchmarks/` directory.

    Those CAD models belong to gmsh and are referenced here, never redistributed. Set the
    GMSH_BENCH environment variable to a gmsh source checkout. Cases whose files are absent
    are skipped rather than failing the run.
    """
    return Path(os.environ.get("GMSH_BENCH", "gmsh-benchmarks")).expanduser()


GB = gmsh_bench_dir()
ANG = 0.5

CASES = [
    ("spheres_100", CORPUS / "spheres_00100.step", 0.1),
    ("spheres_1000", CORPUS / "spheres_01000.step", 0.1),
    ("block_533f", GB / "statreport" / "Block.stp", 0.05),
    ("fusee_249f", GB / "statreport" / "917_fusee.stp", 0.1),
]


def measure(step: Path, lin: float) -> dict[str, float]:
    psutil.Process(os.getpid()).cpu_affinity(list(range(14)))
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("m")
    gmsh.model.occ.importShapes(str(step))
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.StlLinearDeflectionRelative", 0)
    gmsh.option.setNumber("Mesh.StlLinearDeflection", lin)
    gmsh.option.setNumber("Mesh.StlAngularDeflection", ANG)

    t0 = time.perf_counter()
    gmsh.model.mesh.importStl()  # build + convert
    t1 = time.perf_counter()
    gmsh.model.mesh.importStl()  # convert only (STL cached)
    t2 = time.perf_counter()
    gmsh.model.mesh.importStl()  # convert only, repeat
    t3 = time.perf_counter()

    _, etags, _ = gmsh.model.mesh.getElements(2)
    ntri = int(sum(len(t) for t in etags))
    gmsh.finalize()

    convert = min(t2 - t1, t3 - t2)
    total = t1 - t0
    return {
        "n_tri": ntri,
        "total_s": total,
        "convert_s": convert,
        "brepmesh_s": total - convert,
        "convert_pct": 100.0 * convert / total,
    }


def main() -> None:
    out = RESULTS / "gmsh_overhead.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        for name, step, lin in CASES:
            if not step.exists():
                print(f"SKIP {name}")
                continue
            r = measure(step, lin)
            r |= {"case": name, "lin": lin}
            fh.write(json.dumps(r) + "\n")
            print(
                f"{name:<14s} lin={lin:<6g} tris={r['n_tri']:>8d} "
                f"importStl={r['total_s']:7.3f}s = BRepMesh {r['brepmesh_s']:7.3f}s "
                f"+ convert {r['convert_s']:6.3f}s ({r['convert_pct']:4.1f}%)",
                flush=True,
            )


if __name__ == "__main__":
    main()
