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
| **VTK 9.6.2** | BSD-3-Clause | **dynamic**, resolved from the **host** env — **never bundled** | The one runtime dependency shared with the host (flux): SMESH's SMDS is built on `vtkUnstructuredGrid`, so two VTK copies in one process is the hazard that must be avoided. The version is pinned and hard-checked at import. |
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

## Why VTK is treated differently from OCCT/Boost

OCCT and Boost are private implementation details of `_core.pyd`: their DLLs are bundled
into the wheel (at repair time) and nothing else in the host process links them, so their
version is invisible to the host and free to resolve at build time. VTK is **not** private —
`vtkUnstructuredGrid` objects are shared across the SMESH/host boundary, so the wheel must
link the *same* VTK the host already loaded. That is enforced by an exact version match
checked at `import pysmesh` (see `src/pysmesh/__init__.py`); a mismatch raises `ImportError`
rather than risking a silent ABI crash.

## Full license texts

- LGPL-2.1: [LICENSE](LICENSE) (this project and all vendored SALOME sources).
- OCCT LGPL-2.1 exception, Boost BSL-1.0, VTK BSD-3, pybind11 BSD-3, NumPy BSD-3: carried by
  their respective upstream distributions (conda-forge packages / source repos linked in
  PROVENANCE.md).
