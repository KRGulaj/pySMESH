# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-02

"""Compare a golden capture against the baseline, probe by probe and value by value.

Integers, booleans and strings must match exactly. Floats must match within a relative
tolerance (``--rtol``, default 1e-9) with an absolute floor (``--atol``, default 1e-12), so
round-off that a dependency upgrade may legitimately move is separated from a real change.
A probe that gained or lost an ``error`` is always a difference.

The report is grouped. ``geometry`` and ``mesh`` differences are regressions to explain.
``defect`` differences are the known defects of ``docs/reports/defect_sweep_4.2.2.md``, which
Phase 4 is meant to change; they are listed apart so an intended fix is not mistaken for
drift, and so an *unintended* change to a defect is still seen.

Usage:
    python tests/golden/compare.py <baseline.json> <candidate.json> [--rtol R] [--atol A]

Exits 0 when nothing outside the ``defect`` group differs, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _differs(a: object, b: object, rtol: float, atol: float) -> bool:
    """Whether two golden values differ under the tolerance rules above."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is not b
    if isinstance(a, int) and isinstance(b, int):
        return a != b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) or math.isnan(b):
            return not (math.isnan(a) and math.isnan(b))
        return abs(a - b) > max(atol, rtol * max(abs(a), abs(b)))
    return a != b


def compare(
    base: dict[str, dict[str, object]],
    cand: dict[str, dict[str, object]],
    rtol: float,
    atol: float,
) -> dict[str, list[str]]:
    """Differences per group: ``{"geometry": [...], "mesh": [...], "defect": [...]}``."""
    out: dict[str, list[str]] = {}
    for name in sorted(set(base) | set(cand)):
        group = str((base.get(name) or cand.get(name) or {}).get("group", "?"))
        lines = out.setdefault(group, [])
        if name not in cand:
            lines.append(f"{name}: probe REMOVED")
            continue
        if name not in base:
            lines.append(f"{name}: probe ADDED {cand[name]['values']}")
            continue
        bv = base[name]["values"]
        cv = cand[name]["values"]
        assert isinstance(bv, dict) and isinstance(cv, dict)
        for key in sorted(set(bv) | set(cv)):
            if key not in cv:
                lines.append(f"{name}.{key}: removed (was {bv[key]!r})")
            elif key not in bv:
                lines.append(f"{name}.{key}: added {cv[key]!r}")
            elif _differs(bv[key], cv[key], rtol, atol):
                lines.append(f"{name}.{key}: {bv[key]!r} -> {cv[key]!r}")
    return {g: v for g, v in out.items() if v}


def main(argv: list[str]) -> int:
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args(argv)
    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    cand = json.loads(args.candidate.read_text(encoding="utf-8"))
    print(f"baseline  {base['meta']}")
    print(f"candidate {cand['meta']}")
    diffs = compare(base["probes"], cand["probes"], args.rtol, args.atol)
    for group in ("geometry", "mesh", "defect", "?"):
        lines = diffs.get(group, [])
        label = (
            "expected to change in Phase 4"
            if group == "defect"
            else "regressions to explain"
        )
        print(f"\n== {group}: {len(lines)} difference(s) ({label})")
        for line in lines:
            print(f"  {line}")
    blocking = sum(len(v) for g, v in diffs.items() if g != "defect")
    print("\nMATCH" if blocking == 0 else f"\n{blocking} blocking difference(s)")
    return 0 if blocking == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
