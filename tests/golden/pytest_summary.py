# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-02

"""Reduce a pytest JUnit XML file to a stable per-test outcome table, or compare two.

The upgrade plan (OCCT 8.0.1, SMESH V9_16_0) gates every phase on the test suite matching
the 4.2.2 baseline. A JUnit file carries absolute paths, host names and timestamps, so it is
reduced here to what the gate reads: one row per test node id with its outcome and its
duration. Durations are recorded, not compared: they vary between runs on one machine.

Usage:
    python tests/golden/pytest_summary.py summarize <junit.xml> <out.json>
    python tests/golden/pytest_summary.py compare <baseline.json> <candidate.json>

``compare`` exits non-zero when any test changed outcome or appeared or disappeared, and
lists each one.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final

_OUTCOMES: Final[tuple[str, ...]] = ("failure", "error", "skipped")


def _node_id(classname: str, name: str) -> str:
    """Rebuild the pytest node id from JUnit's dotted class name and the test name."""
    module, _, cls = classname.rpartition(".")
    if module.startswith("tests.") and cls and cls[0].isupper():
        return f"{module.replace('.', '/')}.py::{cls}::{name}"
    return f"{classname.replace('.', '/')}.py::{name}"


def summarize(junit: Path) -> dict[str, object]:
    """Read one JUnit file into ``{"totals": ..., "tests": {node_id: {...}}}``."""
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests: dict[str, dict[str, object]] = {}
    for suite in suites:
        for case in suite.iter("testcase"):
            outcome = "passed"
            message = ""
            for child in case:
                if child.tag in _OUTCOMES:
                    outcome = child.tag
                    message = (
                        (child.get("message") or "").splitlines()[0][:200]
                        if child.get("message")
                        else ""
                    )
            node = _node_id(case.get("classname", ""), case.get("name", ""))
            row: dict[str, object] = {
                "outcome": outcome,
                "seconds": round(float(case.get("time", "0")), 3),
            }
            if message:
                row["message"] = message
            tests[node] = row
    totals: dict[str, int] = {}
    for row in tests.values():
        key = str(row["outcome"])
        totals[key] = totals.get(key, 0) + 1
    return {
        "totals": dict(sorted(totals.items())),
        "tests": dict(sorted(tests.items())),
    }


def compare(baseline: dict[str, object], candidate: dict[str, object]) -> list[str]:
    """Every test whose outcome differs, or that exists on one side only."""
    base = baseline["tests"]
    cand = candidate["tests"]
    assert isinstance(base, dict) and isinstance(cand, dict)
    lines: list[str] = []
    for node in sorted(set(base) | set(cand)):
        if node not in cand:
            lines.append(f"REMOVED  {node} (was {base[node]['outcome']})")
        elif node not in base:
            lines.append(f"ADDED    {node} ({cand[node]['outcome']})")
        elif base[node]["outcome"] != cand[node]["outcome"]:
            lines.append(
                f"CHANGED  {node}: {base[node]['outcome']} -> {cand[node]['outcome']}"
                + (
                    f" | {cand[node].get('message', '')}"
                    if cand[node].get("message")
                    else ""
                )
            )
    return lines


def main(argv: list[str]) -> int:
    """Command-line entry point; see the module docstring."""
    if len(argv) == 3 and argv[0] == "summarize":
        data = summarize(Path(argv[1]))
        Path(argv[2]).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        print(f"{argv[2]}: {data['totals']}")
        return 0
    if len(argv) == 3 and argv[0] == "compare":
        base = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        cand = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        diff = compare(base, cand)
        print(f"baseline {base['totals']}  candidate {cand['totals']}")
        for line in diff:
            print(line)
        print("MATCH" if not diff else f"{len(diff)} difference(s)")
        return 0 if not diff else 1
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
