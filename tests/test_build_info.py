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
import sys

import pytest


def test_build_info_has_expected_fields() -> None:
    from pysmesh import _build_info

    assert isinstance(_build_info.VTK_VERSION, str)
    assert isinstance(_build_info.OCCT_VERSION, str)
    assert isinstance(_build_info.BOOST_VERSION, str)
    assert _build_info.WITH_NETGEN is False


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
