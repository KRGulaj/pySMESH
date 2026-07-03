"""Import-time VTK compatibility contract.

pySMESH's ``_core`` links VTK dynamically against the host environment. The package
hard-checks the host VTK version against the one it was built against and raises
ImportError on mismatch — this test drives both the happy path and the mismatch path by
monkeypatching ``vtk.VTK_VERSION`` and reimporting the package.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def test_build_info_has_expected_fields() -> None:
    from pysmesh import _build_info

    assert isinstance(_build_info.VTK_VERSION, str)
    assert isinstance(_build_info.OCCT_VERSION, str)
    assert _build_info.WITH_NETGEN is False


def test_import_accepts_matching_host_vtk() -> None:
    import vtk

    import pysmesh  # imports cleanly in the build/test env

    assert pysmesh._build_info.VTK_VERSION == vtk.VTK_VERSION


def test_import_rejects_mismatched_vtk(monkeypatch: pytest.MonkeyPatch) -> None:
    import vtk

    monkeypatch.setattr(vtk, "VTK_VERSION", "0.0.0-wrong", raising=True)
    # Force a fresh import of the package so __init__ re-runs the check; monkeypatch
    # restores these sys.modules entries at teardown.
    monkeypatch.delitem(sys.modules, "pysmesh", raising=False)
    monkeypatch.delitem(sys.modules, "pysmesh._build_info", raising=False)

    with pytest.raises(ImportError, match="VTK"):
        importlib.import_module("pysmesh")
