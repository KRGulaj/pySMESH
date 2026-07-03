"""Assert a repaired wheel bundles OCCT + Boost but never VTK.

The whole point of pySMESH's packaging is that OCCT/Boost are private (bundled) while VTK is
resolved from the host env (never bundled). delvewheel name-mangles vendored DLLs into a
``pysmesh.libs`` directory inside the wheel; this checks that directory's contents.

Usage:
    python ci/check_wheel.py <wheel-or-glob> [...]
"""

from __future__ import annotations

import sys
import zipfile
from glob import glob


def _check(wheel: str) -> None:
    with zipfile.ZipFile(wheel) as zf:
        dll_names = [
            name.rsplit("/", 1)[-1].lower()
            for name in zf.namelist()
            if name.lower().endswith(".dll")
        ]

    has_occt = any(n.startswith("tk") for n in dll_names)  # OCCT toolkits: TKernel, TKMath...
    has_boost = any("boost" in n for n in dll_names)
    vtk_bundled = [n for n in dll_names if n.startswith("vtk")]

    problems: list[str] = []
    if not has_occt:
        problems.append("no OCCT (TK*.dll) DLLs bundled")
    if not has_boost:
        problems.append("no Boost DLLs bundled")
    if vtk_bundled:
        problems.append(f"VTK DLLs must NOT be bundled, found: {sorted(vtk_bundled)}")

    if problems:
        raise SystemExit(f"{wheel}: " + "; ".join(problems))
    print(f"OK {wheel}: OCCT+Boost bundled, no VTK ({len(dll_names)} DLLs total)")


def main(argv: list[str]) -> None:
    wheels = [w for pattern in argv for w in glob(pattern)]
    if not wheels:
        raise SystemExit("no wheels matched")
    for wheel in wheels:
        _check(wheel)


if __name__ == "__main__":
    main(sys.argv[1:])
