# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-07-03

"""Build provenance, and the absence of any import-time host dependency.

Before 4.0.0 this file drove an import-time VTK version contract: ``_core`` resolved VTK
from the host environment, so ``pysmesh/__init__.py`` compared ``vtk.VTK_VERSION`` against
the compiled-in value and raised ``ImportError`` on a mismatch.

That contract is gone. VTK is bundled privately into the wheel, so there is no host VTK to
agree with. The tests here assert the replacement contract: ``_build_info`` still records
what the binary was built against, and importing ``pysmesh`` imposes no requirement on the
environment at all.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
# The native classes a consumer reaches (report C6). The native Session and Mesher are
# private: the documented Python classes of the same names wrap them.
NATIVE_PUBLIC_CLASSES: tuple[str, ...] = (
    "Shape",
    "FaceInfo",
    "EdgeInfo",
    "SolidInfo",
    "VertexInfo",
    "Mesh",
    "MeshStats",
    "PysmeshError",
    "PysmeshCancelled",
)


def test_build_info_has_expected_fields() -> None:
    from pysmesh import _build_info

    assert isinstance(_build_info.VTK_VERSION, str)
    assert isinstance(_build_info.OCCT_VERSION, str)
    assert isinstance(_build_info.BOOST_VERSION, str)
    assert _build_info.WITH_NETGEN is True


def test_build_info_records_a_real_vtk_version() -> None:
    """The value is provenance now, but it must still name an actual build."""
    from pysmesh import _build_info

    parts = _build_info.VTK_VERSION.split(".")

    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])


def test_import_does_not_require_the_vtk_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import pysmesh`` must succeed with no importable ``vtk``.

    The build environment has VTK installed (it is the build dependency), so absence is
    simulated by blocking the module and forcing a fresh import. Before 4.0.0 this raised
    ``ImportError`` by design; passing now is the 4.0.0 contract.
    """
    monkeypatch.setitem(sys.modules, "vtk", None)
    for name in [m for m in sys.modules if m == "pysmesh" or m.startswith("pysmesh.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("pysmesh")

    assert module.Session is not None


def test_import_ignores_a_mismatched_host_vtk(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host VTK at any version must not affect the import.

    This is the inverse of the pre-4.0.0 test. A wrong host version used to be a hard
    ``ImportError``; it is now irrelevant, because ``_core`` never resolves against it.
    """
    vtk = pytest.importorskip("vtk", reason="build env ships VTK; nothing to mismatch without it")
    monkeypatch.setattr(vtk, "VTK_VERSION", "0.0.0-wrong", raising=True)
    for name in [m for m in sys.modules if m == "pysmesh" or m.startswith("pysmesh.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("pysmesh")

    assert module.Session is not None


def test_version_equals_the_version_in_pyproject() -> None:
    """pysmesh.__version__ is pyproject.toml's [project] version (C6)."""
    import pysmesh

    expected = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pysmesh.__version__ == expected["project"]["version"]


@pytest.mark.parametrize("name", NATIVE_PUBLIC_CLASSES)
def test_a_public_native_class_has_a_docstring(name: str) -> None:
    """Every native class a consumer reaches carries a non-empty docstring (C6)."""
    from pysmesh import _core

    doc = getattr(_core, name).__doc__

    assert isinstance(doc, str)
    assert doc.strip() != ""


def test_every_public_entity_is_documented_and_the_reference_page_counts_them() -> None:
    """ci/count_documented.py passes, and reference/index.md states its count (C6)."""
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "ci" / "count_documented.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    page = (_ROOT / "docs" / "documentation" / "reference" / "index.md").read_text(
        encoding="utf-8"
    )

    total = re.search(r"public entities : (\d+)", proc.stdout)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert total is not None
    assert f"All {total.group(1)} of pySMESH's public entities" in page
