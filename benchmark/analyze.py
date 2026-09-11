# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-11

"""Aggregate raw runs into the tables used by the report.

Reports median with min and max, never mean alone. Emits Markdown tables to stdout and a
tidy CSV to results/summary.csv.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

TOOL_ORDER = ["gmsh", "pysmesh-serial", "pysmesh", "pysmesh-free"]
TOOL_LABEL = {
    "gmsh": "gmsh",
    "pysmesh-serial": "pySMESH (1 thread)",
    "pysmesh": "pySMESH (parallel)",
    "pysmesh-free": "pySMESH (stateless)",
}


def med(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def key(r: dict[str, Any]) -> tuple[str, float]:
    return (r["case"], r["lin"])


def main() -> None:
    runs = load(RESULTS / "raw_runs.jsonl")
    if not runs:
        print("no runs yet")
        return

    grouped: dict[tuple[str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in runs:
        grouped[key(r)][r["tool"]] = r

    rows: list[dict[str, Any]] = []
    for (case, lin), tools in grouped.items():
        base = tools.get("gmsh")
        for tool, r in tools.items():
            w = r["wall_s"]
            c = r["cpu_s"]
            row = {
                "case": case,
                "lin": lin,
                "tool": tool,
                "n_tri": r["n_tri"],
                "wall_med": med(w),
                "wall_min": min(w),
                "wall_max": max(w),
                "cpu_med": med(c),
                "cpu_wall": med(c) / max(med(w), 1e-12),
                "peak_rss_mb": r["peak_rss_mb"],
                "reps": len(w),
                "speedup_vs_gmsh": (med(base["wall_s"]) / med(w)) if base else None,
                "tri_match_gmsh": (r["n_tri"] == base["n_tri"]) if base else None,
            }
            rows.append(row)

    rows.sort(key=lambda r: (r["case"], r["lin"], TOOL_ORDER.index(r["tool"])))

    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    cur = None
    for r in rows:
        if (r["case"], r["lin"]) != cur:
            cur = (r["case"], r["lin"])
            print(f"\n### {r['case']}  lin={r['lin']:g}  ({r['n_tri']:,} triangles)")
            print("| tool | wall median | min | max | CPU s | CPU/wall | peak RSS | vs gmsh |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|")
        sp = r["speedup_vs_gmsh"]
        spx = "1.00x (ref)" if r["tool"] == "gmsh" else f"{sp:.2f}x"
        flag = "" if r["tri_match_gmsh"] else " **TRI MISMATCH**"
        print(
            f"| {TOOL_LABEL[r['tool']]}{flag} | {r['wall_med']:.4f} s | {r['wall_min']:.4f} | "
            f"{r['wall_max']:.4f} | {r['cpu_med']:.3f} | {r['cpu_wall']:.2f} | "
            f"{r['peak_rss_mb']:.0f} MB | {spx} |"
        )
    print(f"\nwrote {RESULTS / 'summary.csv'}")


if __name__ == "__main__":
    main()
