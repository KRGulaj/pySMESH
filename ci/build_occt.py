# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-02

"""Build OCCT from a pinned tag, unpatched, with only the toolkits that ``_core`` needs.

pySMESH bundles OCCT as shared libraries (DLLs). Up to 4.2.2 they came from the
conda-forge package ``occt=8.0.0``. They now come from this script. The local build and
CI both run it. It does four things:

1. It fetches the pinned tag into ``<root>/src`` and verifies the commit hash. It checks
   out with ``core.autocrlf=false``, so every file equals the upstream blob. It applies
   no patch. It refuses a source tree with modified tracked files.
2. It configures with CMake and Ninja into ``<root>/build``.
3. It builds and installs into ``<root>/install``.
4. It writes a stamp file into the install prefix. A later run with the same inputs
   finds the stamp and reuses the install.

One input set defines the build: the tag, the commit, the CMake options and the MSVC
toolset. The toolset is ``VCToolsVersion``, which ``vcvars64.bat`` and
``ilammy/msvc-dev-cmd`` set. The ``cache-key`` command prints a key derived from exactly
that set. CI uses it as the ``actions/cache`` key.

The CMake options mirror the conda-forge feedstock (``conda-forge/occt-feedstock``,
``recipe/bld.bat``) that built ``occt 8.0.0 all_h8ecc14b_202``. They are equal wherever
an option can change behaviour: Release, IPO on, OCCT exceptions kept in Release, no
TBB, the native memory manager, the default optimisation profile. They differ where the
feedstock builds something that ``_core`` does not use: no Draw, VTK, FreeImage,
RapidJSON, OpenGL, Tcl/Tk, FreeType, Draco, OpenVR or FFmpeg. No documentation, samples
or tests are built.

Usage:
    python ci/build_occt.py build --root <dir> [--version 8.0.1] [--jobs N]
    python ci/build_occt.py verify --root <dir> [--version 8.0.1]
    python ci/build_occt.py cache-key [--version 8.0.1]

Run every command from a shell that has the MSVC x64 environment. ``build`` also needs
``cmake``, ``ninja`` and ``git`` on ``PATH``. ``verify`` exits non-zero unless
``<root>/install`` holds a complete install that matches the input set. CI runs it after
a cache restore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

OCCT_REPO_URL: Final[str] = "https://github.com/Open-Cascade-SAS/OCCT.git"

# The stamp file that marks a complete install. It lives inside the install prefix, so a
# cached prefix carries its own proof of what it holds.
STAMP_NAME: Final[str] = "pysmesh_occt_build.json"


@dataclass(frozen=True)
class OcctPin:
    """One pinned OCCT release: the upstream tag and the commit it must resolve to."""

    version: str
    tag: str
    commit: str


# 8.0.1 is the version pySMESH ships. 8.0.0 is the control build. It is the exact source
# that conda-forge built: the feedstock downloads the GitHub archive of tag V8_0_0. If a
# from-source 8.0.0 reproduces the 4.2.2 baseline, these options change no result. Then
# any difference on 8.0.1 comes from the OCCT version.
OCCT_PINS: Final[dict[str, OcctPin]] = {
    "8.0.1": OcctPin("8.0.1", "V8_0_1", "b8f597c677811d1f9f4d8a97f5ae2825c0353a42"),
    "8.0.0": OcctPin("8.0.0", "V8_0_0", "d3056ef80c9668f395da40f5fd7be186cae4501f"),
}
DEFAULT_VERSION: Final[str] = "8.0.1"

# Every OCCT toolkit that a pySMESH target links (CMakeLists.txt and
# cmake/*/CMakeLists.txt). OCCT adds the transitive closure from each toolkit's
# EXTERNLIB.cmake. TKBool, TKDE, TKHLR and TKService join it that way: TKV3d and TKVCAF
# pull TKService in.
OCCT_TOOLKITS: Final[tuple[str, ...]] = (
    "TKernel",
    "TKMath",
    "TKG2d",
    "TKG3d",
    "TKGeomBase",
    "TKBRep",
    "TKGeomAlgo",
    "TKTopAlgo",
    "TKPrim",
    "TKBO",
    "TKShHealing",
    "TKMesh",
    "TKOffset",
    "TKFillet",
    "TKFeat",
    "TKHelix",
    "TKExpress",
    "TKV3d",
    "TKXSBase",
    "TKDESTEP",
    "TKDEIGES",
    "TKDESTL",
    "TKCDF",
    "TKLCAF",
    "TKCAF",
    "TKVCAF",
    "TKXCAF",
)

# The toolkits that the closure adds. ``verify`` requires these as well.
OCCT_CLOSURE_TOOLKITS: Final[tuple[str, ...]] = ("TKBool", "TKDE", "TKHLR", "TKService")

# Every OCCT module is off. BUILD_ADDITIONAL_TOOLKITS selects the toolkits above.
OCCT_MODULES: Final[tuple[str, ...]] = (
    "FoundationClasses",
    "ModelingData",
    "ModelingAlgorithms",
    "Visualization",
    "ApplicationFramework",
    "DataExchange",
    "Draw",
)

# The CMake options. They hold no path, so they can enter the cache key. Each option
# that can change behaviour equals the feedstock's value. Where the feedstock sets none,
# it equals OCCT's default. PROVENANCE.md, "How OCCT is built", has the comparison.
CMAKE_OPTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("CMAKE_BUILD_TYPE", "Release"),
    ("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "ON"),
    ("BUILD_LIBRARY_TYPE", "Shared"),
    ("BUILD_CPP_STANDARD", "C++17"),
    ("BUILD_RELEASE_DISABLE_EXCEPTIONS", "OFF"),
    ("BUILD_ENABLE_FPE_SIGNAL_HANDLER", "OFF"),
    ("BUILD_OPT_PROFILE", "Default"),
    ("USE_MMGR_TYPE", "NATIVE"),
    ("BUILD_WITH_DEBUG", "OFF"),
    ("BUILD_USE_PCH", "OFF"),
    ("INSTALL_DIR_LAYOUT", "Unix"),
    *((f"BUILD_MODULE_{module}", "OFF") for module in OCCT_MODULES),
    ("BUILD_ADDITIONAL_TOOLKITS", ";".join(OCCT_TOOLKITS)),
    ("USE_TBB", "OFF"),
    ("USE_FREETYPE", "OFF"),
    ("USE_FREEIMAGE", "OFF"),
    ("USE_RAPIDJSON", "OFF"),
    ("USE_DRACO", "OFF"),
    ("USE_OPENVR", "OFF"),
    ("USE_FFMPEG", "OFF"),
    ("USE_VTK", "OFF"),
    ("USE_TK", "OFF"),
    ("USE_OPENGL", "OFF"),
    ("USE_GLES2", "OFF"),
    ("USE_EIGEN", "OFF"),
    ("BUILD_GTEST", "OFF"),
    ("BUILD_DOC_Overview", "OFF"),
    ("BUILD_DOC_RefMan", "OFF"),
    ("INSTALL_TEST_CASES", "OFF"),
)


@dataclass(frozen=True)
class BuildInputs:
    """The input set that defines one OCCT build, and so its cache key."""

    pin: OcctPin
    cmake_options: tuple[tuple[str, str], ...]
    msvc_toolset: str

    def as_record(self) -> dict[str, object]:
        """The input set as a JSON-ready record, in a stable key order."""
        return {
            "version": self.pin.version,
            "tag": self.pin.tag,
            "commit": self.pin.commit,
            "msvc_toolset": self.msvc_toolset,
            "cmake_options": [f"{name}={value}" for name, value in self.cmake_options],
        }

    def cache_key(self) -> str:
        """A readable key whose last field is a digest of the whole input set."""
        canonical = json.dumps(self.as_record(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return (
            f"occt-{self.pin.version}-{self.pin.commit[:12]}"
            f"-msvc{self.msvc_toolset}-{digest}"
        )


@dataclass(frozen=True)
class BuildTree:
    """The three directories of one build: source, build and install."""

    src: Path
    build: Path
    install: Path

    @classmethod
    def under(cls, root: Path) -> BuildTree:
        """The standard layout below ``root``."""
        root = root.resolve()
        return cls(src=root / "src", build=root / "build", install=root / "install")


def _msvc_toolset() -> str:
    """The MSVC toolset version of the active developer environment."""
    toolset = os.environ.get("VCToolsVersion", "").strip().rstrip("\\")
    if not toolset:
        raise SystemExit(
            "VCToolsVersion is not set. Run from an MSVC x64 developer environment "
            "(vcvars64.bat, or ilammy/msvc-dev-cmd in CI)."
        )
    return toolset


def _inputs(version: str) -> BuildInputs:
    """The input set for one pinned version, with the active MSVC toolset."""
    if version not in OCCT_PINS:
        raise SystemExit(
            f"no OCCT pin for version {version!r}; known: {sorted(OCCT_PINS)}"
        )
    return BuildInputs(OCCT_PINS[version], CMAKE_OPTIONS, _msvc_toolset())


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run one command. Return its stdout. Raise on a non-zero exit."""
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=cwd, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    if proc.returncode != 0:
        logger.error("command failed with exit code %d", proc.returncode)
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def _run_streamed(cmd: list[str]) -> None:
    """Run one long command with its output passed through. Raise on a non-zero exit."""
    logger.info("$ %s", " ".join(cmd))
    sys.stdout.flush()
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _require_tools(names: tuple[str, ...]) -> None:
    """Fail loudly if a required build tool is not on PATH."""
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"required tools not on PATH: {', '.join(missing)}")


def _git(src: Path, *args: str) -> str:
    """Run git in ``src`` with line-ending conversion off."""
    return _run(["git", "-c", "core.autocrlf=false", "-C", str(src), *args]).strip()


def fetch_source(pin: OcctPin, src: Path) -> None:
    """Fetch the pinned tag into ``src``, verify the commit and the clean tree."""
    if not (src / ".git").is_dir():
        if src.exists() and any(src.iterdir()):
            raise SystemExit(f"{src} exists, is not empty and is not a git checkout")
        src.mkdir(parents=True, exist_ok=True)
        _git(src, "init", "--quiet")
        _git(src, "remote", "add", "origin", OCCT_REPO_URL)
    head = _git(src, "rev-parse", "HEAD") if _has_commit(src) else ""
    if head != pin.commit:
        refspec = f"refs/tags/{pin.tag}:refs/tags/{pin.tag}"
        _git(src, "fetch", "--depth", "1", "origin", refspec)
        tagged = _git(src, "rev-parse", f"refs/tags/{pin.tag}^{{commit}}")
        if tagged != pin.commit:
            raise SystemExit(
                f"tag {pin.tag} resolves to {tagged}, expected {pin.commit}. "
                "The upstream tag moved; do not build it."
            )
        _git(src, "checkout", "--quiet", "--detach", pin.commit)
    head = _git(src, "rev-parse", "HEAD")
    if head != pin.commit:
        raise SystemExit(f"{src} is at {head}, expected {pin.commit}")
    dirty = _git(src, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit(
            f"{src} has modified tracked files; the OCCT build is unpatched:\n{dirty}"
        )
    logger.info("source %s at %s (%s), clean", src, head, pin.tag)


def _has_commit(src: Path) -> bool:
    """Whether the repository in ``src`` has any checked-out commit yet."""
    proc = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "--verify", "--quiet", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def configure(inputs: BuildInputs, tree: BuildTree) -> None:
    """Configure the OCCT build tree with Ninja."""
    defines = [f"-D{name}={value}" for name, value in inputs.cmake_options]
    _run_streamed(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            str(tree.src),
            "-B",
            str(tree.build),
            f"-DCMAKE_INSTALL_PREFIX={tree.install.as_posix()}",
            *defines,
        ]
    )


def build_and_install(tree: BuildTree, jobs: int) -> None:
    """Build every configured toolkit, then install into the prefix."""
    _run_streamed(["cmake", "--build", str(tree.build), "--parallel", str(jobs)])
    _run_streamed(["cmake", "--install", str(tree.build)])


def _expected_files(install: Path) -> list[Path]:
    """The files a complete install must hold."""
    files = [
        install / "lib" / "cmake" / "opencascade" / "OpenCASCADEConfig.cmake",
        install / "include" / "opencascade" / "Standard_Version.hxx",
    ]
    for toolkit in (*OCCT_TOOLKITS, *OCCT_CLOSURE_TOOLKITS):
        files.append(install / "bin" / f"{toolkit}.dll")
        files.append(install / "lib" / f"{toolkit}.lib")
    return files


def _installed_version(install: Path) -> str:
    """The OCCT version that the installed Standard_Version.hxx declares."""
    header = install / "include" / "opencascade" / "Standard_Version.hxx"
    for line in header.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["#define", "OCC_VERSION_COMPLETE"]:
            return parts[2].strip('"')
    raise SystemExit(f"{header} defines no OCC_VERSION_COMPLETE")


def check_install(inputs: BuildInputs, install: Path) -> list[str]:
    """Every reason the install does not match ``inputs``; empty when it does."""
    stamp = install / STAMP_NAME
    if not stamp.is_file():
        return [f"no stamp file {stamp}"]
    problems: list[str] = []
    recorded = json.loads(stamp.read_text(encoding="utf-8"))
    if recorded.get("inputs") != inputs.as_record():
        problems.append(f"stamp inputs differ from the requested build: {stamp}")
    if recorded.get("cache_key") != inputs.cache_key():
        problems.append(
            f"stamp cache key {recorded.get('cache_key')} != {inputs.cache_key()}"
        )
    missing = [str(path) for path in _expected_files(install) if not path.is_file()]
    if missing:
        problems.append("missing files: " + ", ".join(missing))
    else:
        found = _installed_version(install)
        if found != inputs.pin.version:
            problems.append(f"installed OCCT is {found}, expected {inputs.pin.version}")
    return problems


def write_stamp(inputs: BuildInputs, install: Path) -> None:
    """Record the input set in the install prefix, after a complete install."""
    record = {"cache_key": inputs.cache_key(), "inputs": inputs.as_record()}
    (install / STAMP_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def cmd_build(version: str, root: Path, jobs: int) -> int:
    """Fetch, configure, build and install, unless a matching install exists."""
    inputs = _inputs(version)
    tree = BuildTree.under(root)
    if not check_install(inputs, tree.install):
        logger.info("reusing %s (cache key %s)", tree.install, inputs.cache_key())
        return 0
    _require_tools(("git", "cmake", "ninja", "cl"))
    fetch_source(inputs.pin, tree.src)
    if tree.install.exists():
        logger.info("removing incomplete or stale install %s", tree.install)
        shutil.rmtree(tree.install)
    configure(inputs, tree)
    build_and_install(tree, jobs)
    write_stamp(inputs, tree.install)
    problems = check_install(inputs, tree.install)
    if problems:
        raise SystemExit(
            "install is incomplete after the build: " + "; ".join(problems)
        )
    logger.info("installed OCCT %s into %s", version, tree.install)
    logger.info("cache key %s", inputs.cache_key())
    return 0


def cmd_verify(version: str, root: Path) -> int:
    """Exit non-zero unless ``<root>/install`` is a complete, matching install."""
    inputs = _inputs(version)
    install = BuildTree.under(root).install
    problems = check_install(inputs, install)
    if problems:
        for problem in problems:
            logger.error("%s", problem)
        return 1
    logger.info("OK %s: OCCT %s, cache key %s", install, version, inputs.cache_key())
    return 0


def main(argv: list[str]) -> int:
    """Parse the command line and run one command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify", "cache-key"):
        cmd = sub.add_parser(name)
        cmd.add_argument(
            "--version", default=DEFAULT_VERSION, choices=sorted(OCCT_PINS)
        )
        if name != "cache-key":
            cmd.add_argument("--root", type=Path, required=True)
        if name == "build":
            cmd.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[build_occt] %(message)s")

    if args.command == "cache-key":
        sys.stdout.write(_inputs(args.version).cache_key() + "\n")
        return 0
    if args.command == "verify":
        return cmd_verify(args.version, args.root)
    return cmd_build(args.version, args.root, args.jobs)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
