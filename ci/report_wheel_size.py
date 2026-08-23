"""Report a repaired wheel's size, broken down by bundled dependency.

Private VTK is a size trade-off, so it is measured on every build rather than estimated.
Two numbers matter:

1. The wheel file size. PyPI rejects any single file over 100 MB.
2. The VTK share of it. That is the actual cost of the 4.0.0 packaging change, and it is
   the number the README and NOTICE.md quote.

Exits non-zero if the wheel would be rejected by PyPI, so the limit is a build failure
rather than a surprise at upload time.

Usage:
    python ci/report_wheel_size.py <wheel>
"""

from __future__ import annotations

import os
import sys
import zipfile
from collections import defaultdict

# PyPI's default per-file limit. An increase must be requested at github.com/pypi/support.
_PYPI_FILE_LIMIT_BYTES = 100 * 1024 * 1024


def _bucket(name: str) -> str:
    """Classify one archive member into a reporting bucket."""
    stem = name.rsplit("/", 1)[-1].lower()
    if not stem.endswith(".dll"):
        return "python + extension"
    if stem.startswith("vtk"):
        return "VTK (private since 4.0.0)"
    if stem.startswith("tk"):
        return "OCCT"
    if "boost" in stem:
        return "Boost"
    return "other DLLs"


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:6.1f} MB"


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("usage: python ci/report_wheel_size.py <wheel>")
    wheel = argv[0]

    compressed: defaultdict[str, int] = defaultdict(int)
    uncompressed: defaultdict[str, int] = defaultdict(int)
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            b = _bucket(info.filename)
            compressed[b] += info.compress_size
            uncompressed[b] += info.file_size

    on_disk = os.path.getsize(wheel)
    print(f"\n{os.path.basename(wheel)}")
    print(f"  wheel on disk: {_mb(on_disk)}  (PyPI limit {_mb(_PYPI_FILE_LIMIT_BYTES)})")
    print(f"  {'bucket':<28} {'in wheel':>10}   {'unpacked':>10}")
    for b in sorted(compressed, key=lambda k: -compressed[k]):
        print(f"  {b:<28} {_mb(compressed[b])}   {_mb(uncompressed[b])}")

    if on_disk > _PYPI_FILE_LIMIT_BYTES:
        print(
            f"\nFAIL: {_mb(on_disk)} exceeds PyPI's {_mb(_PYPI_FILE_LIMIT_BYTES)} per-file "
            "limit. Either trim the bundle or request a limit increase at "
            "https://github.com/pypi/support before publishing.",
            file=sys.stderr,
        )
        return 1

    headroom = _PYPI_FILE_LIMIT_BYTES - on_disk
    print(f"\nOK: {_mb(headroom)} of headroom under the PyPI per-file limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
