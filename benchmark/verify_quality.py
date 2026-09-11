# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-11

"""Matched-quality verification: were both tessellators asked for, and did they achieve,
the same thing?

For each (geometry, tolerance) it dumps both triangle soups, tests whether they are the same
set of triangles, and measures achieved chordal deviation against an exact analytic surface.
Results go to results/quality.jsonl.

If achieved deviations differ by more than a few percent at the same requested tolerance,
the comparison at that tolerance is void and is reported as such rather than used.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import os
import sys
from pathlib import Path

import numpy as np

import deviation as dv

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
PY = Path(sys.executable)
AFFINITY = ",".join(str(i) for i in range(14))
ANG = 0.5

RADIUS = 10.0
PITCH = 30.0
CONE_R1, CONE_R2, CONE_H = 20.0, 5.0, 40.0

SAMPLE_ORDER = 6  # 28 barycentric points per triangle
MAX_TRIS_FOR_DEV = 400_000  # subsample above this, with a fixed seed


def dump(tool: str, step: Path, lin: float, out: Path) -> int:
    cmd = [
        str(PY), str(HERE / "bench_worker.py"), "--tool", tool, "--step", str(step),
        "--lin", str(lin), "--ang", str(ANG), "--reps", "1",
        "--affinity", AFFINITY, "--dump-tris", str(out),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    line = next((x for x in p.stdout.splitlines() if x.startswith("@@JSON@@")), None)
    if line is None:
        raise RuntimeError(f"{tool} failed on {step.name}: {p.stderr[-400:]}")
    return int(json.loads(line[len("@@JSON@@"):])["n_tri"])


def deviations(tris: np.ndarray, kind: str, side: int) -> dict[str, float]:
    rng = np.random.default_rng(42)
    if len(tris) > MAX_TRIS_FOR_DEV:
        sel = rng.choice(len(tris), MAX_TRIS_FOR_DEV, replace=False)
        tris = tris[sel]
    pts = dv.sample_triangles(tris, SAMPLE_ORDER)
    if kind == "sphere_grid":
        d = dv.dev_sphere_grid(pts, PITCH, RADIUS, side)
    elif kind == "cone":
        d = dv.dev_truncated_cone(pts, CONE_R1, CONE_R2, CONE_H)
    else:
        raise ValueError(kind)
    return dv.summarise(d)


CASES = [
    ("sphere1", CORPUS / "spheres_00001.step", "sphere_grid", 1,
     (1.0, 0.3, 0.1, 0.03, 0.01, 0.003)),
    ("cone", CORPUS / "cone.step", "cone", 1,
     (1.0, 0.3, 0.1, 0.03, 0.01, 0.003)),
    ("spheres100", CORPUS / "spheres_00100.step", "sphere_grid", 5, (0.1,)),
    ("spheres1000", CORPUS / "spheres_01000.step", "sphere_grid", 10, (0.1,)),
]


def main() -> None:
    out_path = RESULTS / "quality.jsonl"
    with out_path.open("a", encoding="utf-8") as fh, tempfile.TemporaryDirectory() as td:
        for name, step, kind, side, lins in CASES:
            if not step.exists():
                print(f"SKIP {name}: missing {step}")
                continue
            for lin in lins:
                ga = Path(td) / "g.npy"
                pa = Path(td) / "p.npy"
                ng = dump("gmsh", step, lin, ga)
                npy = dump("pysmesh", step, lin, pa)
                gt = np.load(ga)
                pt = np.load(pa)
                same, maxd = dv.identical(gt, pt)
                gq = deviations(gt, kind, side)
                pq = deviations(pt, kind, side)
                rel = abs(gq["dev_max"] - pq["dev_max"]) / max(gq["dev_max"], 1e-300)
                rec = {
                    "case": name, "lin": lin, "ang": ANG,
                    "n_tri_gmsh": ng, "n_tri_pysmesh": npy,
                    "tri_delta_pct": 100.0 * (npy - ng) / ng,
                    "identical": bool(same), "identity_max_coord_diff": maxd,
                    "gmsh": gq, "pysmesh": pq,
                    "dev_max_rel_diff_pct": 100.0 * rel,
                    "void": bool(rel > 0.05),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(
                    f"{name:<12s} lin={lin:<7g} tris {ng:>8d}/{npy:<8d} "
                    f"same={str(same):<5s} devmax g={gq['dev_max']:.4e} "
                    f"p={pq['dev_max']:.4e} rel={100 * rel:6.3f}% "
                    f"{'VOID' if rel > 0.05 else 'ok'}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
