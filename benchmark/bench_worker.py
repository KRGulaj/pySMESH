# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-11

"""One benchmark cell: one tool, one geometry, one tolerance, N repetitions.

Runs in its own process so that peak resident set and CPU affinity are attributable to a
single tessellator. Emits one JSON object on stdout.

The timed region is the same operation on both sides: from a loaded, live CAD model held in
memory to a triangle soup available to the caller. Geometry import (STEP -> OCC shape) is
performed once, outside the timed region, on both sides.

Tools
    gmsh      gmsh.model.mesh.importStl(), which runs OCCT BRepMesh_IncrementalMesh once per
              face (GModelIO_OCC.cpp:6339, called from the TopExp_Explorer loop at :6487),
              then harvests via getElements/getNodes.
    pysmesh   Session.tessellate(incremental=False), which runs BRepMesh_IncrementalMesh once
              over the whole shape (session_tessellate.cpp:105) and harvests to NumPy.
    pysmesh-serial   as above with parallel=False: the single-threaded control.
    pysmesh-free     the stateless pysmesh.tessellate(brep_bytes) entry point, which also
              parses the BREP inside the timed region. Reported separately, never in the
              headline.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

_PROC = psutil.Process(os.getpid())


def cpu_seconds() -> float:
    """User+system CPU seconds for this process and every child."""
    t = _PROC.cpu_times()
    total = t.user + t.system
    for c in _PROC.children(recursive=True):
        try:
            ct = c.cpu_times()
            total += ct.user + ct.system
        except psutil.Error:
            pass
    return total


def peak_rss_bytes() -> int:
    mi = _PROC.memory_info()
    return int(getattr(mi, "peak_wset", mi.rss))


# --------------------------------------------------------------------------------------- #
# gmsh
# --------------------------------------------------------------------------------------- #
def _gmsh_session(step: Path, lin: float, ang: float):  # noqa: ANN202
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("m")
    gmsh.model.occ.importShapes(str(step))
    gmsh.model.occ.synchronize()
    # Absolute chordal deviation on both sides: OCCT's Relative mode is defined per edge,
    # so an absolute tolerance is the only unambiguous shared control.
    gmsh.option.setNumber("Mesh.StlLinearDeflectionRelative", 0)
    gmsh.option.setNumber("Mesh.StlLinearDeflection", lin)
    gmsh.option.setNumber("Mesh.StlAngularDeflection", ang)
    return gmsh


def run_gmsh(step: Path, lin: float, ang: float, reps: int, keep: bool) -> dict[str, Any]:
    import gmsh

    walls: list[float] = []
    cpus: list[float] = []
    ntri = 0
    tris_out = None
    for i in range(reps + 1):  # rep 0 is the discarded warm-up
        g = _gmsh_session(step, lin, ang)
        gc.collect()
        c0, w0 = cpu_seconds(), time.perf_counter()
        g.model.mesh.importStl()
        _, etags, enodes = g.model.mesh.getElements(2)
        ntags, ncoords, _ = g.model.mesh.getNodes()
        w1, c1 = time.perf_counter(), cpu_seconds()
        ntri = int(sum(len(t) for t in etags))
        if i > 0:
            walls.append(w1 - w0)
            cpus.append(c1 - c0)
        if keep and i == reps:
            coords = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)
            tagmap = np.zeros(int(np.max(ntags)) + 1, dtype=np.int64)
            tagmap[np.asarray(ntags, dtype=np.int64)] = np.arange(len(ntags))
            conn = np.concatenate([np.asarray(e, dtype=np.int64) for e in enodes])
            tris_out = coords[tagmap[conn]].reshape(-1, 3, 3)
        gmsh.finalize()
    return {"walls": walls, "cpus": cpus, "n_tri": ntri, "tris": tris_out}


# --------------------------------------------------------------------------------------- #
# pySMESH
# --------------------------------------------------------------------------------------- #
def native_lin(imp: Any, lin_mm: float) -> float:
    """Convert a deflection expressed in millimetres into the model's own length unit.

    gmsh's STEP reader rescales every model to millimetres (OCCT's xstep.cascade.unit
    default), while pysmesh.read_step_xde preserves the file's declared unit. Asking both
    for the same numeric deflection therefore asks for different physical tolerances on any
    file not already in millimetres: on a metre file it is a 1000x coarser request, which
    silently produces a coarser mesh in less time and a meaningless speed-up.

    length_unit is metres per model unit, so millimetres per model unit is length_unit*1000.
    """
    mm_per_unit = float(imp.length_unit) * 1000.0
    if not mm_per_unit > 0.0:
        raise RuntimeError(f"bad length_unit {imp.length_unit!r}")
    return lin_mm / mm_per_unit


def run_pysmesh(
    step: Path, lin: float, ang: float, reps: int, parallel: bool, keep: bool
) -> dict[str, Any]:
    import pysmesh

    imp = pysmesh.read_step_xde(str(step))
    lin = native_lin(imp, lin)
    s = pysmesh.Session()
    s.add_brep(imp.brep)

    walls: list[float] = []
    cpus: list[float] = []
    ntri = 0
    tris_out = None
    for i in range(reps + 1):
        gc.collect()
        c0, w0 = cpu_seconds(), time.perf_counter()
        rm = s.tessellate(
            deflection=lin,
            angle_deg=math.degrees(ang),
            relative=False,
            parallel=parallel,
            incremental=False,
        )
        nodes, tris = rm.nodes, rm.tris
        w1, c1 = time.perf_counter(), cpu_seconds()
        ntri = int(tris.shape[0])
        if i > 0:
            walls.append(w1 - w0)
            cpus.append(c1 - c0)
        if keep and i == reps:
            tris_out = nodes[tris]
    return {"walls": walls, "cpus": cpus, "n_tri": ntri, "tris": tris_out}


def run_pysmesh_free(
    step: Path, lin: float, ang: float, reps: int, keep: bool
) -> dict[str, Any]:
    import pysmesh

    imp = pysmesh.read_step_xde(str(step))
    lin = native_lin(imp, lin)
    brep = imp.brep
    walls: list[float] = []
    cpus: list[float] = []
    ntri = 0
    tris_out = None
    for i in range(reps + 1):
        gc.collect()
        c0, w0 = cpu_seconds(), time.perf_counter()
        res = pysmesh.tessellate(
            brep, pysmesh.TessellateParams(lin, math.degrees(ang), False)
        )
        w1, c1 = time.perf_counter(), cpu_seconds()
        ntri = int(res.tris.shape[0])
        if i > 0:
            walls.append(w1 - w0)
            cpus.append(c1 - c0)
        if keep and i == reps:
            tris_out = res.nodes[res.tris]
    return {"walls": walls, "cpus": cpus, "n_tri": ntri, "tris": tris_out}


# --------------------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True,
                    choices=["gmsh", "pysmesh", "pysmesh-serial", "pysmesh-free"])
    ap.add_argument("--step", required=True)
    ap.add_argument("--lin", type=float, required=True)
    ap.add_argument("--ang", type=float, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--affinity", default="")
    ap.add_argument("--dump-tris", default="")
    args = ap.parse_args()

    if args.affinity:
        cpus = [int(x) for x in args.affinity.split(",")]
        _PROC.cpu_affinity(cpus)

    step = Path(args.step)
    keep = bool(args.dump_tris)
    lin_native = args.lin
    if args.tool != "gmsh":
        import pysmesh

        lin_native = native_lin(pysmesh.read_step_xde(str(step)), args.lin)

    t_load0 = time.perf_counter()
    if args.tool == "gmsh":
        r = run_gmsh(step, args.lin, args.ang, args.reps, keep)
    elif args.tool == "pysmesh":
        r = run_pysmesh(step, args.lin, args.ang, args.reps, True, keep)
    elif args.tool == "pysmesh-serial":
        r = run_pysmesh(step, args.lin, args.ang, args.reps, False, keep)
    else:
        r = run_pysmesh_free(step, args.lin, args.ang, args.reps, keep)
    total_s = time.perf_counter() - t_load0

    tris = r.pop("tris", None)
    if keep and tris is not None:
        np.save(args.dump_tris, np.ascontiguousarray(tris, dtype=np.float64))

    out = {
        "tool": args.tool,
        "step": step.name,
        "lin": args.lin,
        "lin_mm": args.lin,
        "lin_native": lin_native,
        "ang": args.ang,
        "reps": args.reps,
        "n_tri": r["n_tri"],
        "wall_s": r["walls"],
        "cpu_s": r["cpus"],
        "peak_rss_mb": peak_rss_bytes() / (1024 * 1024),
        "affinity": args.affinity,
        "total_s": total_s,
    }
    sys.stdout.write("@@JSON@@" + json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
