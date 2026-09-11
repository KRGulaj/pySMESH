# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-11

"""Achieved chordal deviation on a real imported STEP body, where no closed form exists.

Each side's triangles are projected onto the exact B-rep surface of the face that produced
them, by that side's own OCC kernel:

  gmsh     gmsh.model.mesh.getElements(2, tag) gives that surface's triangles, and
           gmsh.model.getClosestPoint(2, tag, xyz) projects them (OCC 7.8.1).
  pySMESH  RenderMesh.tri_face_id gives the owning face, and Session.project_on_face
           projects onto it (OCCT 8.0.0).

Using each kernel on its own output avoids having to match face identifiers between two
importers, which is the step that silently went wrong with units. The two projectors are
cross-checked against the closed form on the sphere and the cone, where they agree exactly,
so neither is flattering its own side.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import gmsh
import numpy as np
import psutil

import pysmesh
import deviation as dv

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
ANG = 0.5
ORDER = 4  # 15 barycentric samples per triangle

CASES = [
    ("block_533f", GB / "statreport" / "Block.stp", 0.05),
    ("fusee_249f", GB / "statreport" / "917_fusee.stp", 0.1),
    ("kurbelwelle_58f", GB / "statreport" / "Kurbelwelle.stp", 0.02),
]


def gmsh_dev(step: Path, lin: float) -> dict[str, float]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("m")
    gmsh.model.occ.importShapes(str(step))
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.StlLinearDeflectionRelative", 0)
    gmsh.option.setNumber("Mesh.StlLinearDeflection", lin)
    gmsh.option.setNumber("Mesh.StlAngularDeflection", ANG)
    gmsh.model.mesh.importStl()

    devs: list[np.ndarray] = []
    skipped = [0]
    for _, tag in gmsh.model.getEntities(2):
        try:
            _, etags, enodes = gmsh.model.mesh.getElements(2, tag)
        except Exception:  # noqa: BLE001
            continue
        if not etags or len(etags[0]) == 0:
            continue
        conn = np.concatenate([np.asarray(e, dtype=np.int64) for e in enodes])
        ntags, ncoords, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
        allt, allc, _ = gmsh.model.mesh.getNodes()
        lut = np.zeros(int(np.max(allt)) + 1, dtype=np.int64)
        lut[np.asarray(allt, dtype=np.int64)] = np.arange(len(allt))
        coords = np.asarray(allc, dtype=np.float64).reshape(-1, 3)
        tris = coords[lut[conn]].reshape(-1, 3, 3)
        pts = dv.sample_triangles(tris, ORDER, interior=True)
        try:
            cc, _ = gmsh.model.getClosestPoint(2, tag, pts.reshape(-1).tolist())
        except Exception:  # noqa: BLE001
            skipped[0] += 1
            continue
        proj = np.asarray(cc, dtype=np.float64).reshape(-1, 3)
        devs.append(np.linalg.norm(pts - proj, axis=1))
    gmsh.finalize()
    r = dv.summarise(np.concatenate(devs))
    r['faces_skipped'] = skipped[0]
    return r


def pysmesh_dev(step: Path, lin_mm: float) -> dict[str, float]:
    imp = pysmesh.read_step_xde(str(step))
    lin = lin_mm / (float(imp.length_unit) * 1000.0)
    s = pysmesh.Session()
    s.add_brep(imp.brep)
    rm = s.tessellate(deflection=lin, angle_deg=math.degrees(ANG),
                      relative=False, incremental=False)
    nodes, tris, tfi = rm.nodes, rm.tris, rm.tri_face_id
    # tri_face_id from Session.tessellate carries EntityIds (session_tessellate.cpp assigns
    # label_of(face)), not TopExp ordinals. Group by the value itself.
    devs: list[np.ndarray] = []
    skipped = [0]
    for fid in np.unique(tfi):
        sel = tfi == fid
        if not sel.any():
            continue
        t = nodes[tris[sel]]
        pts = dv.sample_triangles(t, ORDER, interior=True)
        try:
            out = s.project_on_face(int(fid), pts)
        except Exception:  # noqa: BLE001
            skipped[0] += 1
            continue
        devs.append(np.asarray(out.distance, dtype=np.float64))
    d = np.concatenate(devs)
    scale = float(imp.length_unit) * 1000.0  # back to millimetres
    r = dv.summarise(d * scale)
    r['faces_skipped'] = skipped[0]
    return r


def main() -> None:
    out = RESULTS / "real_quality.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        for name, step, lin in CASES:
            if not step.exists():
                print(f"SKIP {name}")
                continue
            psutil.Process(os.getpid()).cpu_affinity(list(range(14)))
            g = gmsh_dev(step, lin)
            p = pysmesh_dev(step, lin)
            rel = abs(g["dev_max"] - p["dev_max"]) / max(g["dev_max"], 1e-300)
            rec = {"case": name, "lin_mm": lin, "gmsh": g, "pysmesh": p,
                   "dev_max_rel_diff_pct": 100.0 * rel, "void": bool(rel > 0.05)}
            fh.write(json.dumps(rec) + "\n")
            print(
                f"{name:<16s} lin={lin:<6g}mm  devmax g={g['dev_max']:.5e} "
                f"p={p['dev_max']:.5e} rel={100 * rel:6.2f}%  "
                f"rms g={g['dev_rms']:.5e} p={p['dev_rms']:.5e} "
                f"{'VOID' if rel > 0.05 else 'ok'}",
                flush=True,
            )


if __name__ == "__main__":
    main()
