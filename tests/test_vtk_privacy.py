# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-23

"""The 4.0.0 architecture guard: no VTK object may cross the Python boundary.

Bundling VTK privately into the wheel is safe for exactly one reason. Nothing shares VTK
objects between ``_core`` and the host process. Every result leaves as a NumPy array or as
BREP bytes.

Break that and the failure is silent and severe. A ``vtkUnstructuredGrid`` built inside
``_core`` is an instance of a class from the bundled, name-mangled VTK copy. The host's
``vtkUnstructuredGrid`` is a different C++ type even when both report the same version
string, because they come from different DLLs. Passing one where the other is expected is
memory corruption, not a type error.

These tests are the tripwire. They fail on the commit that introduces such a binding, not
in a consumer's process months later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# NOTE: ``pysmesh`` and ``numpy`` are imported inside the runtime tests below, not here. The
# four static guards are pure source checks and must stay runnable in a bare checkout with
# no built extension, so a lint-only job can enforce the architecture without a full build.

_REPO = Path(__file__).resolve().parent.parent
_BINDINGS = _REPO / "src" / "bindings"
_STUBS = _REPO / "src" / "pysmesh" / "_core.pyi"

# A VTK C++ type name: 'vtk' followed by an upper-case letter (vtkUnstructuredGrid,
# vtkDataArray, vtkSmartPointer). Lower-case 'vtk...' in prose is not a type.
_VTK_TYPE = re.compile(r"\bvtk[A-Z]\w*")

# pybind11 lines that publish something to Python. A VTK type on one of these is the defect.
_EXPORTING_CALL = re.compile(r"\.(def|def_readonly|def_readwrite|def_property\w*|attr)\b")


def _strip_comments(source: str) -> str:
    """Remove // and /* */ comments so prose about VTK ordering is not a false positive."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def test_no_vtk_type_in_any_pybind_export() -> None:
    """No ``.def(...)`` in the bindings may mention a VTK type."""
    offenders: list[str] = []

    for path in sorted(_BINDINGS.rglob("*.cpp")):
        for lineno, line in enumerate(_strip_comments(path.read_text("utf-8")).splitlines(), 1):
            if _EXPORTING_CALL.search(line) and _VTK_TYPE.search(line):
                offenders.append(f"{path.relative_to(_REPO)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "a VTK type is exported to Python. The bundled VTK is a private, name-mangled copy, "
        "so such an object is not interchangeable with the host's VTK even at an identical "
        "version. Return NumPy arrays instead:\n  " + "\n  ".join(offenders)
    )


def test_no_vtk_type_registered_with_pybind() -> None:
    """No VTK type may be given a pybind11 caster or class binding."""
    offenders: list[str] = []
    registrations = re.compile(r"py::class_\s*<|PYBIND11_(?:DECLARE_HOLDER_TYPE|MAKE_OPAQUE)")

    for path in sorted(_BINDINGS.rglob("*.cpp")) + sorted(_BINDINGS.rglob("*.h")):
        for lineno, line in enumerate(_strip_comments(path.read_text("utf-8")).splitlines(), 1):
            if registrations.search(line) and _VTK_TYPE.search(line):
                offenders.append(f"{path.relative_to(_REPO)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "a VTK type is registered with pybind11, which would make it constructible or "
        "returnable from Python:\n  " + "\n  ".join(offenders)
    )


def test_typed_stubs_name_no_vtk_type() -> None:
    """The public stubs are the API contract; a VTK type must not appear in them."""
    found = _VTK_TYPE.findall(_strip_comments(_STUBS.read_text("utf-8")))

    assert not found, f"_core.pyi exposes VTK types: {sorted(set(found))}"


def test_package_does_not_import_vtk() -> None:
    """No module in the shipped package may import ``vtk`` at runtime."""
    offenders: list[str] = []
    import_vtk = re.compile(r"^\s*(?:import\s+vtk|from\s+vtk[\s.]+import)\b", re.MULTILINE)

    for path in sorted((_REPO / "src" / "pysmesh").rglob("*.py")):
        if import_vtk.search(path.read_text("utf-8")):
            offenders.append(str(path.relative_to(_REPO)))

    assert not offenders, (
        "the package imports vtk. Since 4.0.0 pysmesh carries its own VTK and must not "
        "resolve the host's:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "attribute",
    ["nodes", "tris", "face_id"],
)
def test_tessellate_returns_plain_numpy(attribute: str) -> None:
    """The render path returns arrays, never a VTK object."""
    import numpy as np

    import pysmesh

    session = pysmesh.Session()
    session.add_box(1.0, 2.0, 3.0)

    value = getattr(session.tessellate(deflection=0.1, angle_deg=20.0), attribute)

    assert isinstance(value, np.ndarray)
    assert type(value).__module__.startswith("numpy")


def test_mesh_harvest_returns_plain_numpy() -> None:
    """``Mesher.mesh()`` is the SMDS read-back, the closest thing to a VTK leak."""
    import numpy as np

    import pysmesh

    session = pysmesh.Session()
    session.add_box(1.0, 2.0, 3.0)
    render = session.tessellate(deflection=0.1, angle_deg=20.0)
    mesher = pysmesh.Mesher.from_arrays(render.nodes, render.tris)

    harvest = mesher.mesh()

    try:
        assert isinstance(harvest.node_coords, np.ndarray)
        assert isinstance(harvest.element_nodes, np.ndarray)
        assert harvest.node_coords.dtype == np.float64
    finally:
        mesher.release()
