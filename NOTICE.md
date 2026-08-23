# Third-Party Notices

pySMESH (`pysmesh`, LGPL-2.1-only) is a thin pybind11 binding around the SALOME Platform
meshing stack. Its single shared library, `pysmesh/_core.pyd`, statically links a minimal
slice of SMESH + KERNEL and links the rest dynamically. This file lists every third-party
component that ends up in, or is required at runtime by, the shipped wheel.

Full upstream URLs, commits, and the patch index are in [PROVENANCE.md](PROVENANCE.md).

| Component | License | How it ships | Obligation & how it is met |
|---|---|---|---|
| SALOME **SMESH** (vendored + patched) | LGPL-2.1 | **static** in `_core.pyd` | Complete corresponding source is this public repo (`extern/smesh/` + `patches/`); the relinking right is preserved because the whole binary is rebuildable from source. |
| SALOME **KERNEL** (minimal slice) | LGPL-2.1 | **static** in `_core.pyd` | As SMESH — source in `extern/kernel/`, CORBA compiled out (`SALOME_LIGHT`). |
| SALOME **GEOM** (`GEOMUtils` only) | LGPL-2.1 | **static** in `_core.pyd` | Source slice in `extern/geom/src/GEOMUtils/`. |
| **MEFISTO2** `trte.c` (f2c) + **pthread** shim | LGPL-2.1 | **static** in `_core.pyd` | Source in `extern/mefisto2/`, `extern/pthread/` (via `looooo/SMESH`). |
| **Open CASCADE Technology (OCCT) 8.0.0** | LGPL-2.1 **with the exception** | **dynamic**, DLLs **bundled into the wheel** | LGPL static-linking exception is not even relied on (OCCT is dynamic); the relinking right holds because pySMESH is fully open and rebuildable. Build recipe in PROVENANCE.md. |
| **Boost 1.90** | BSL-1.0 | **dynamic**, DLLs **bundled into the wheel** | BSL-1.0 is permissive (notice only); this entry is the notice. |
| **VTK 9.6.2** | BSD-3-Clause | **dynamic**, DLLs **bundled into the wheel** | BSD-3-Clause is permissive (notice only); this entry is the notice. Private to `_core.pyd` since 4.0.0, name-mangled like OCCT/Boost. Only three components are linked (`CommonCore`, `CommonDataModel`, `FiltersVerdict`), so the bundle carries no rendering, IO, or Python-wrapper modules. |
| **pybind11 3.0.3** | BSD-3-Clause | header-only (compile time) | Notice only; this entry is the notice. |
| **NumPy** | BSD-3-Clause | runtime (pip dependency) | Notice only. |

## OCCT toolkits bundled

OCCT ships as many per-domain toolkit DLLs; pySMESH bundles those its `_core.pyd` links
(directly or transitively) at wheel-repair time. All are the same component and licence as the
OCCT row above (LGPL-2.1 with the exception) — this list is enumeration, not a new obligation.
Beyond the modelling/meshing toolkits (TKernel, TKMath, TKBRep, TKG2d/TKG3d, TKGeomBase,
TKGeomAlgo, TKTopAlgo, TKPrim, TKBO, TKMesh, TKShHealing, TKOffset, …), the **B1 STEP-import
feature** (`read_step_xde` / `write_step_xde`) adds the OCCT **DataExchange + OCAF/XDE** stack:

- **TKDESTEP** — STEP reader/writer (OCCT-8.0 rename of the former TKSTEP).
- **TKXCAF**, **TKVCAF** — eXtended Data Exchange shape/colour/name tools.
- **TKLCAF**, **TKCAF**, **TKCDF** — OCAF document core (document, attributes, storage driver).
- **TKXSBase** — data-exchange base (interface model, static parameters).

These DLLs are added to the wheel (a few MB) with no new licence text; the relinking right holds
as for all OCCT toolkits because pySMESH is fully open and rebuildable from this repo.

The **v2 (Tier C) modelling surface** links three further toolkits. All are the same OCCT
component and licence, and none is a new obligation:

- **TKFeat** — `BRepFeat_SplitShape` (imprint), the feature-operation toolkit.
- **TKHelix** — `HelixBRep_BuilderHelix` / `HelixGeom_*` (helical wire construction).
- **TKDEIGES** — `IGESControl_Reader` / `IGESCAFControl_Reader` (IGES import).

These three appear in the wheel only once a binding actually calls them: MSVC records an
import entry per *used* symbol, so a toolkit named in the link line but not referenced by
`src/bindings` contributes no DLL dependency and delvewheel does not vendor it.
`ci/check_wheel.py` therefore asserts the toolkits that ship today and names these three as
the ones to add alongside their bindings.

## Why VTK is now treated the same as OCCT/Boost

Up to 3.4.0, VTK was the one shared runtime dependency. The wheel linked the host's VTK and
`import pysmesh` enforced an exact version match, because `vtkUnstructuredGrid` was assumed
to be shared across the SMESH/host boundary.

That assumption did not hold. No VTK object ever crossed the boundary. `_core` links only
three VTK components (`CommonCore`, `CommonDataModel`, `FiltersVerdict`) and every result
leaves as a NumPy array or BREP bytes, so the host and `_core` never touch each other's VTK
objects. The sharing bought nothing and cost every consumer an exact-version constraint.

Since 4.0.0 all three of OCCT, Boost and VTK are private implementation details of
`_core.pyd`. Their DLLs are bundled into the wheel at repair time and name-mangled, so their
versions are invisible to the host and free to resolve at build time. A consumer may run any
VTK, or none.

This is safe for exactly one reason, and it is the reason to protect: **no VTK object may
cross the Python boundary.** The bundled copy is name-mangled, so a `vtkUnstructuredGrid`
built inside `_core` is a different C++ type from the host's, even at an identical version
string. `tests/test_vtk_privacy.py` fails the build if a binding ever exports one.

## Full license texts

- LGPL-2.1: [LICENSE](LICENSE) (this project and all vendored SALOME sources).
- OCCT LGPL-2.1 exception, Boost BSL-1.0, VTK BSD-3, pybind11 BSD-3, NumPy BSD-3: carried by
  their respective upstream distributions (conda-forge packages / source repos linked in
  PROVENANCE.md).
