"""Assert pySMESH imports and works with NO ``vtk`` package available.

This is the 4.0.0 contract reduced to one executable check. Run it in a bare virtual
environment that has numpy and the pysmesh wheel, and nothing else. If ``import vtk``
succeeds there, the environment is wrong and the check refuses to report a false pass.

Before 4.0.0 this script could not have passed: ``pysmesh/__init__.py`` imported ``vtk``
and raised ``ImportError`` when it was absent.

Scope, stated precisely. This proves no ``vtk`` **Python package** is needed. It does not by
itself prove no host VTK **DLL** was loaded, because a CI shell may still carry the build
env's ``Library/bin`` on PATH. That second guarantee comes from delvewheel instead: it
name-mangles every bundled DLL and rewrites ``_core.pyd``'s import table to match, so
``_core`` resolves names that exist only inside ``pysmesh.libs``. A host VTK cannot satisfy
them whatever is on PATH. ``ci/check_wheel.py`` asserts those mangled DLLs are present.

Usage:
    python ci/assert_no_host_vtk.py
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import vtk  # noqa: F401
    except ImportError:
        pass
    else:
        print(
            "ENVIRONMENT ERROR: a 'vtk' package is importable here, so this check cannot "
            "prove the wheel is self-contained. Run it in a venv without vtk installed.",
            file=sys.stderr,
        )
        return 2

    import numpy as np

    import pysmesh

    # Exercise the paths that actually touch VTK inside _core: SMDS (the mesh data
    # structure is a vtkUnstructuredGrid subclass) and the quality controls (FiltersVerdict).
    session = pysmesh.Session()
    session.add_box(3.0, 7.0, 11.0)
    render = session.tessellate(deflection=0.05, angle_deg=17.0)

    mesher = pysmesh.Mesher.from_arrays(render.nodes, render.tris)
    harvest = mesher.mesh()
    if harvest.element_count != render.tris.shape[0]:
        print(
            f"FAIL: harvested {harvest.element_count} elements, expected "
            f"{render.tris.shape[0]}",
            file=sys.stderr,
        )
        return 1

    # A quality control routes through VTK's verdict library, so this fails loudly if
    # FiltersVerdict did not get bundled. Mesher inherits _QualityOps, hence the method form.
    aspect = mesher.quality(pysmesh.AspectRatio())
    if not np.all(np.isfinite(aspect.values)):
        print("FAIL: non-finite aspect ratios from the verdict path", file=sys.stderr)
        return 1

    patches = mesher.separate_faces_by_edges(mesher.sharp_edges(angle=31.0))
    if patches.count != 6:
        print(f"FAIL: expected 6 box patches, got {patches.count}", file=sys.stderr)
        return 1

    mesher.release()
    print(
        f"OK: pysmesh {pysmesh._build_info.GIT_SHA} imported and meshed with no host VTK "
        f"({harvest.element_count} elements, {patches.count} patches, "
        f"bundled VTK {pysmesh._build_info.VTK_VERSION})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
