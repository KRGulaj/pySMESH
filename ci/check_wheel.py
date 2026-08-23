"""Assert a repaired wheel is self-contained, and carries an honest ABI tag.

Since 4.0.0 pySMESH has no shared native dependency: OCCT, Boost **and VTK** are all private
to ``_core.pyd``. delvewheel name-mangles the whole DLL closure into a ``pysmesh.libs``
directory inside the wheel; this checks that directory's contents.

The VTK assertion is inverted from earlier versions. Before 4.0.0 a bundled ``vtk*.dll`` was
a failure, because VTK was resolved from the host environment. It is now a requirement: a
wheel without it would fall back to whatever VTK the host happens to expose, which is the
silent-ABI hazard the private copy exists to remove.

It also checks the wheel's ABI tag. ``_core`` is a pybind11 module, and pybind11 does not
support ``Py_LIMITED_API``, so the built extension is version-locked
(``_core.cp313-win_amd64.pyd``). An ``abi3`` tag on such a wheel is a false promise: pip
would install it on a later CPython and the import would fail. The tag must therefore name
one interpreter version, not the stable ABI.

Usage:
    python ci/check_wheel.py <wheel-or-glob> [...]
"""

from __future__ import annotations

import os
import sys
import zipfile
from glob import glob

# OCCT DataExchange + OCAF/XDE toolkits the B1 STEP-import feature links; delvewheel may
# name-mangle the DLLs (e.g. ``tkdestep-<hash>.dll``), so match on the toolkit stem as a prefix.
_B1_XDE_TOOLKITS = ("tkdestep", "tkxcaf", "tklcaf", "tkcaf", "tkcdf", "tkxsbase")

# Modelling toolkits the v2 (Tier C) operation surface is built on. All are in ``_core.pyd``'s
# transitive DLL closure today and must stay bundled.
#
# Scope note, so this list is read correctly: MSVC records an import entry only for a DLL
# whose import library actually contributed a symbol, so a toolkit named in CMake but not yet
# *called* by ``src/bindings`` produces no dependency and delvewheel does not vendor it.
# TKFeat (BRepFeat_SplitShape) is therefore linked and proven usable by ``tests/probe`` (the
# ``v2_probe`` target) but is deliberately absent from this list until a binding calls it —
# add each stem here in the same commit that adds its binding. TKDEIGES joined the list in
# 3.3.0, the release that added ``read_iges`` / ``write_iges``. Reachability of the whole v2
# surface is gated by ``v2_probe``; this file gates only what actually ships.
_V2_MODELLING_TOOLKITS = (
    "tkprim",  # BRepPrimAPI_Make* (primitives)
    "tkbo",  # BRepAlgoAPI_* / BOPAlgo_* (booleans, defeature/split)
    "tkfillet",  # BRepFilletAPI_Make{Fillet,Chamfer}
    "tkbool",  # BRepFill_PipeShell, reached via BRepOffsetAPI_MakePipeShell
    "tkoffset",  # BRepOffsetAPI_* (sweeps/filling, v1 offsets)
    "tkshhealing",  # ShapeFix_* / ShapeUpgrade_*
    "tktopalgo",  # BRepBuilderAPI_*, BRepGProp, BRepCheck_Analyzer
    "tkmesh",  # BRepMesh_IncrementalMesh
    "tkhelix",  # HelixBRep_BuilderHelix, called by Session.add_helix
    "tkexpress",  # ExprIntrp, reached by the mesher's expression-based 1-D distributions
    "tkdeiges",  # IGESControl_{Reader,Writer}, called by read_iges / write_iges
)


def _check_abi_tag(wheel: str) -> list[str]:
    """Return tag problems for ``wheel``; empty if the ABI tag is honest.

    A wheel filename is ``{name}-{version}-{python}-{abi}-{platform}.whl``. For a pybind11
    extension the ABI tag must equal the interpreter tag (``cp313-cp313``). ``abi3`` claims
    the stable ABI, which pybind11 never produces.
    """
    stem = os.path.basename(wheel)
    if stem.lower().endswith(".whl"):
        stem = stem[: -len(".whl")]
    parts = stem.split("-")
    if len(parts) < 5:
        return [f"unparseable wheel filename (expected 5 dash-separated fields): {stem}"]
    python_tag, abi_tag = parts[-3], parts[-2]

    problems: list[str] = []
    if abi_tag == "abi3":
        problems.append(
            "ABI tag is 'abi3', but pybind11 does not build a stable-ABI module. pip would "
            "install this wheel on a newer CPython and the import would fail. Remove "
            "'py-api' from [tool.scikit-build.wheel] in pyproject.toml."
        )
    elif abi_tag != python_tag:
        problems.append(
            f"ABI tag '{abi_tag}' does not match the interpreter tag '{python_tag}'; a "
            "pybind11 extension is locked to the version it was built against."
        )
    return problems


def _check(wheel: str) -> None:
    tag_problems = _check_abi_tag(wheel)
    with zipfile.ZipFile(wheel) as zf:
        dll_names = [
            name.rsplit("/", 1)[-1].lower()
            for name in zf.namelist()
            if name.lower().endswith(".dll")
        ]

    has_occt = any(n.startswith("tk") for n in dll_names)  # OCCT toolkits: TKernel, TKMath...
    has_boost = any("boost" in n for n in dll_names)
    vtk_bundled = [n for n in dll_names if n.startswith("vtk")]
    # The three components _core links (CMakeLists.txt: CommonCore, CommonDataModel,
    # FiltersVerdict). Match on the stem because delvewheel mangles the DLL names.
    missing_vtk = [
        mod
        for mod in ("vtkcommoncore", "vtkcommondatamodel", "vtkfiltersverdict")
        if not any(n.startswith(mod) for n in dll_names)
    ]
    missing_xde = [
        tk for tk in _B1_XDE_TOOLKITS if not any(n.startswith(tk) for n in dll_names)
    ]
    missing_modelling = [
        tk for tk in _V2_MODELLING_TOOLKITS if not any(n.startswith(tk) for n in dll_names)
    ]

    problems: list[str] = list(tag_problems)
    if not has_occt:
        problems.append("no OCCT (TK*.dll) DLLs bundled")
    if not has_boost:
        problems.append("no Boost DLLs bundled")
    if not vtk_bundled:
        problems.append(
            "no VTK DLLs bundled. Since 4.0.0 VTK is private to _core and must ship inside "
            "the wheel; an unbundled build would resolve VTK from the host env, which is the "
            "ABI hazard this design removes. Check that delvewheel is NOT passed "
            "--exclude 'vtk*.dll'."
        )
    elif missing_vtk:
        problems.append(
            "VTK components _core links are missing from the bundle: " + ", ".join(missing_vtk)
        )
    if missing_xde:
        problems.append(
            "B1 STEP-import (XDE) toolkits missing from the bundle: " + ", ".join(missing_xde)
        )
    if missing_modelling:
        problems.append(
            "v2 modelling toolkits missing from the bundle: " + ", ".join(missing_modelling)
        )

    if problems:
        raise SystemExit(f"{wheel}: " + "; ".join(problems))
    print(
        f"OK {wheel}: honest ABI tag, self-contained "
        f"(OCCT + Boost + {len(vtk_bundled)} VTK DLLs, {len(dll_names)} total)"
    )


def main(argv: list[str]) -> None:
    wheels = [w for pattern in argv for w in glob(pattern)]
    if not wheels:
        raise SystemExit("no wheels matched")
    for wheel in wheels:
        _check(wheel)


if __name__ == "__main__":
    main(sys.argv[1:])
