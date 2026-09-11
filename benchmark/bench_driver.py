# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-11

"""Benchmark driver: runs the full (tool x geometry x tolerance) matrix.

Each cell runs in a fresh subprocess pinned to the same logical processors. Raw per-run data
is appended to results/raw_runs.jsonl; nothing is aggregated here beyond echoing progress.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
PY = Path(sys.executable)

# LP 0-13: the 6 P-cores and 8 E-cores. LP 14-15 are the LP-E cores and are excluded, which
# matches this machine's documented worker pool.
AFFINITY = ",".join(str(i) for i in range(14))

TOOLS = ("gmsh", "pysmesh", "pysmesh-serial", "pysmesh-free")
ANG = 0.5  # radians, both sides


@dataclass(frozen=True)
class Case:
    name: str
    path: Path
    lins: tuple[float, ...]
    reps: int


def gmsh_bench_dir() -> Path:
    """Where the gmsh source distribution's `benchmarks/` directory lives.

    Those CAD models are gmsh's, not this project's, so they are referenced rather than
    redistributed. Point GMSH_BENCH at a gmsh source checkout, or pass --gmsh-bench. Cases
    whose files are absent are skipped with a message, so the synthetic suites still run
    without it.
    """
    return Path(os.environ.get("GMSH_BENCH", "gmsh-benchmarks")).expanduser()


def big_assembly_path() -> Path | None:
    """An optional large proprietary assembly, supplied by the operator.

    Set BIG_ASSEMBLY to a STEP file to run the `big` suite. There is no default, because no
    such file ships with this repository.
    """
    p = os.environ.get("BIG_ASSEMBLY", "")
    return Path(p).expanduser() if p else None



def ladder_cases(reps: int) -> list[Case]:
    """Synthetic face-count ladder at one fixed tolerance."""
    return [
        Case(f"spheres_{n}", CORPUS / f"spheres_{n:05d}.step", (0.1,), reps)
        for n in (1, 10, 100, 1000, 5000)
    ]


def sweep_cases(reps: int) -> list[Case]:
    """Tolerance sweeps: >= 5 values spanning ~2 orders of magnitude."""
    return [
        Case("sweep_sphere1", CORPUS / "spheres_00001.step",
             (1.0, 0.3, 0.1, 0.03, 0.01, 0.003), reps),
        Case("sweep_cone", CORPUS / "cone.step",
             (1.0, 0.3, 0.1, 0.03, 0.01, 0.003), reps),
        Case("sweep_spheres1000", CORPUS / "spheres_01000.step",
             (1.0, 0.3, 0.1, 0.03, 0.01), reps),
        Case("sweep_block", gmsh_bench_dir() / "statreport" / "Block.stp",
             (0.5, 0.2, 0.05, 0.02, 0.005), reps),
    ]


def real_cases(reps: int) -> list[Case]:
    """Real imported STEP bodies, from gmsh's own published benchmark corpus."""
    s = gmsh_bench_dir() / "step"
    r = gmsh_bench_dir() / "statreport"
    return [
        Case("part_26f", s / "part.step", (0.05,), reps),
        Case("wrenchnut_36f", s / "wrenchnut.stp", (0.05,), reps),
        Case("kurbelwelle_58f", r / "Kurbelwelle.stp", (0.02,), reps),
        Case("laufrad_123f", s / "1385_Laufrad_1000.stp", (0.2,), reps),
        Case("ujoint_133f", s / "U_Joint_2.stp", (0.05,), reps),
        Case("zylkopf_137f", r / "Zylkopf.stp", (0.02,), reps),
        Case("fusee_249f", r / "917_fusee.stp", (0.1,), reps),
        Case("block_533f", r / "Block.stp", (0.05,), reps),
    ]


def big_cases(reps: int) -> list[Case]:
    p = big_assembly_path()
    if p is None:
        return []
    return [Case("big_assembly", p, (1.0, 0.25), reps)]


def run_cell(tool: str, case: Case, lin: float, reps: int) -> dict | None:
    cmd = [
        str(PY), str(HERE / "bench_worker.py"),
        "--tool", tool, "--step", str(case.path),
        "--lin", str(lin), "--ang", str(ANG),
        "--reps", str(reps), "--affinity", AFFINITY,
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    el = time.perf_counter() - t0
    line = next((x for x in p.stdout.splitlines() if x.startswith("@@JSON@@")), None)
    if line is None:
        print(f"    FAILED {tool:<15s} rc={p.returncode} {p.stderr.strip()[:300]}")
        return None
    rec = json.loads(line[len("@@JSON@@"):])
    rec["case"] = case.name
    rec["driver_elapsed_s"] = el
    w = sorted(rec["wall_s"])
    med = w[len(w) // 2]
    print(f"    {tool:<15s} tris={rec['n_tri']:>9d} med={med:9.4f}s "
          f"rss={rec['peak_rss_mb']:7.1f}MB")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="ladder",
                    choices=["ladder", "sweep", "real", "big", "all"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tools", default=",".join(TOOLS))
    ap.add_argument("--out", default="raw_runs.jsonl")
    ap.add_argument("--cases", default="", help="comma-separated case names to include")
    ap.add_argument("--gmsh-bench", default="",
                    help="path to the gmsh source distribution's benchmarks/ directory")
    ap.add_argument("--big-assembly", default="",
                    help="STEP file for the 'big' suite; none ships with this repo")
    args = ap.parse_args()
    if args.gmsh_bench:
        os.environ["GMSH_BENCH"] = args.gmsh_bench
    if args.big_assembly:
        os.environ["BIG_ASSEMBLY"] = args.big_assembly

    suites = {
        "ladder": ladder_cases, "sweep": sweep_cases,
        "real": real_cases, "big": big_cases,
    }
    if args.suite == "all":
        cases = [c for f in suites.values() for c in f(args.reps)]
    else:
        cases = suites[args.suite](args.reps)

    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.name in wanted]

    tools = args.tools.split(",")
    out = RESULTS / args.out
    print(f"host={platform.node()} affinity=LP[{AFFINITY}] reps={args.reps} -> {out}")

    with out.open("a", encoding="utf-8") as fh:
        for case in cases:
            if not case.path.exists():
                print(f"  SKIP {case.name}: missing {case.path}")
                continue
            for lin in case.lins:
                print(f"  {case.name}  lin={lin}")
                for tool in tools:
                    rec = run_cell(tool, case, lin, args.reps)
                    if rec is not None:
                        rec["suite"] = args.suite
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
    print("done")


if __name__ == "__main__":
    sys.exit(main())
