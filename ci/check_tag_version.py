"""Assert a release tag matches the version it will publish.

``pyproject.toml`` carries a static ``version``. Tags do not feed it. Nothing else in the
pipeline notices when the two disagree, and the consequences land on PyPI, where a version
number can be yanked but never reused:

- Tagging ``v4.0.0rc1`` while ``version = "4.0.0"`` uploads the real release number during
  what was meant to be a rehearsal, spending it on a build nobody reviewed.
- Tagging ``v4.1.0`` while ``version = "4.0.0"`` silently republishes the old version, or
  fails the upload as a duplicate after a full build.

So the tag is checked against the version before anything is built.

Pre-release tags follow ``vN.N.NrcN``, matching the ``rc`` filter that keeps
``publish-pypi`` off a rehearsal tag in ``.github/workflows/ci.yml``.

Usage:
    python ci/check_tag_version.py                  # tag from GITHUB_REF_NAME
    python ci/check_tag_version.py v4.0.0           # tag given explicitly
    python ci/check_tag_version.py v4.0.0 dist/*.whl  # also check the built wheel
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from glob import glob
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# PEP 440 subset this project uses: X.Y.Z, optionally followed by rcN.
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:rc\d+)?$")


def _project_version() -> str:
    """Return the ``version`` declared in pyproject.toml."""
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)
    try:
        return str(data["project"]["version"])
    except KeyError as exc:
        raise SystemExit("pyproject.toml has no [project] version") from exc


def _wheel_version(path: str) -> str:
    """Return the version encoded in a wheel filename."""
    parts = os.path.basename(path).split("-")
    if len(parts) < 5:
        raise SystemExit(f"unparseable wheel filename: {path}")
    return parts[1]


def main(argv: list[str]) -> int:
    tag = argv[0] if argv else os.environ.get("GITHUB_REF_NAME", "")
    if not tag:
        print("no tag given and GITHUB_REF_NAME is unset; nothing to check.")
        return 0

    version = _project_version()

    if not tag.startswith("v"):
        print(f"FAIL: tag '{tag}' does not start with 'v'.", file=sys.stderr)
        return 1

    tag_version = tag[1:]
    if not _VERSION_RE.match(tag_version):
        print(
            f"FAIL: tag '{tag}' is not a supported version. Use vX.Y.Z or vX.Y.ZrcN.",
            file=sys.stderr,
        )
        return 1

    if tag_version != version:
        print(
            f"FAIL: tag '{tag}' does not match pyproject version '{version}'.\n"
            f"      The build would publish {version} under a {tag} tag.\n"
            f"      Set version = \"{tag_version}\" in pyproject.toml, or retag as v{version}.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: tag {tag} matches pyproject version {version}")

    wheels = [w for pattern in argv[1:] for w in glob(pattern)]
    for wheel in wheels:
        built = _wheel_version(wheel)
        if built != version:
            print(
                f"FAIL: {os.path.basename(wheel)} carries version '{built}', "
                f"expected '{version}'.",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {os.path.basename(wheel)} carries version {built}")

    if "rc" in tag_version:
        print("Pre-release tag: TestPyPI only, publish-pypi is skipped.")
    else:
        print("Final tag: publishes to TestPyPI and then to PyPI.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
