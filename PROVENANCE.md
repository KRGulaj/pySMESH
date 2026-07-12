# Provenance

pySMESH is built on top of the SALOME Platform's meshing stack. This file lists everything
we pulled in from outside the project — where it came from, at what commit, and why — so
anyone auditing the license (LGPL-2.1) or the build can trace every file back to its source.
Nothing here is invented; where we had to write something ourselves to bridge a gap, that's
called out explicitly.

Thanks to the SALOME Platform team and to the maintainers of `looooo/SMESH` and
`conda-forge/smesh-feedstock` — this project would not have gotten off the ground without
their prior work making a standalone, Windows-buildable SMESH possible.

## Vendored upstream sources

All vendored under `extern/`, kept pristine — nothing here is hand-edited. `prepare.py`
copies the parts we compile into a git-ignored `staged/` tree and applies the patches below
there.

| Component | Upstream | Version | Local path |
|---|---|---|---|
| SALOME SMESH | [SalomePlatform/smesh](https://github.com/SalomePlatform/smesh) | tag `V9_9_0` | `extern/smesh/` |
| SALOME KERNEL | [SalomePlatform/kernel](https://github.com/SalomePlatform/kernel) | tag `V9_9_0` | `extern/kernel/` |
| SALOME GEOM (`GEOMUtils` only) | [SalomePlatform/geom](https://github.com/SalomePlatform/geom) | tag `V9_9_0` | `extern/geom/src/GEOMUtils/` |

SMESH and KERNEL are imported as squashed git subtrees, matched to the same `V9_9_0` release
since SALOME versions its modules together. GEOM is a sparse copy of one directory, not a
full subtree — GEOM itself is a large CORBA/GUI module we don't need, and SMESH only reaches
into it for `GEOMUtils.cxx` (a handful of standalone OCCT geometry helpers used by
`SMESH_Mesh`, `SMESH_Controls`, and the Cartesian mesher).

KERNEL is needed because SMESH's data structures (`SMESHDS_Mesh`, `SMESH_ProxyMesh`) depend
on `smIdType`, a KERNEL-defined typedef that decides whether node/element IDs are 32- or
64-bit. Only KERNEL's `Basics/`, `SALOMELocalTrace/`, and `Utils/` are actually compiled —
the CORBA/communication layer is never built and is compiled out via `SALOME_LIGHT`.

We build for 64-bit Windows, so `SALOME_USE_64BIT_IDS` is on and `smIdType` resolves to
`int64_t` everywhere.

## Small pieces borrowed from looooo/SMESH

[looooo/SMESH](https://github.com/looooo/SMESH) is an actively maintained fork that keeps
SALOME's meshing libraries buildable standalone on Windows/MSVC, and conda-forge's own SMESH
package is built from it. Two small non-SALOME files come from there because upstream SMESH
doesn't ship a Windows-buildable equivalent:

| File | Local path | Why we need it |
|---|---|---|
| `extra/MEFISTO2/trte.c` | `extern/mefisto2/trte.c` | Upstream SMESH ships the MEFISTO2 triangulator's core routine only as Fortran (`trte.f`). There's no Fortran compiler in an MSVC toolchain, so we use looooo's f2c-translated C version instead. |
| `extra/pthread/{pthread.h,semaphore.h}` | `extern/pthread/` | A tiny header-only POSIX-pthread shim over Win32 `SRWLOCK`, needed because the conda toolchain has no `winpthreads`. |

Both are licensed LGPL-2.1, same as SMESH itself.

## Patches

`patches/{kernel,geom,smesh,occt8}/*.patch`, applied by `prepare.py` in that order. Most come
from looooo/SMESH's own patch set (Windows/MSVC fixes, VTK 9.4–9.6 API breaks); the `occt8/`
pair comes from conda-forge's `smesh-feedstock` recipe, which is the only place we found a
working OCCT 8.0 compatibility pass for this codebase. NETGEN-related patches are left out —
we don't build NETGEN.

One detail worth recording honestly: we vendor SALOME's official `V9_9_0` **tags**, while
looooo's patches were written against slightly newer commits on the `V9_9_0` **branch**. Most
patches apply cleanly regardless; a few of looooo's source patches turn out to already be
satisfied by our tag and are skipped automatically (`prepare.py` uses `patch -N`, so this is
detected, not silently ignored). A handful of small deltas that looooo's branch already had
weren't available as clean patches against the tag, so we reproduced the same fix as a plain
source edit in `prepare.py` instead of forcing a patch to apply — each one is commented at the
call site with what it does and why (CORBA stripped from KERNEL, a couple of OCCT 8.0
NCollection/hasher requirements, one renamed OCCT toolkit). Nothing there is a design
decision of ours; it's the same fix looooo already made, just written by hand because the
diff didn't line up byte-for-byte.

### Patch index

Source key: **L** = `looooo/SMESH` patch series (the Windows/MSVC standalone-SMESH fork
conda-forge builds from); **C** = `conda-forge/smesh-feedstock` recipe. `prepare.py` applies
these with `patch -N --fuzz=2`, so files already satisfied by our `V9_9_0` tag are skipped
and logged (not silently ignored). MinGW/gcc-only patches are no-ops under our MSVC build.

| Patch | Src | What it fixes |
|---|---|---|
| `geom/GEOMUtils.patch` | L | Build `GEOMUtils.cxx` standalone (the one GEOM file SMESH needs). |
| `kernel/Kernel.patch` | L | KERNEL standalone base (Basics/trace/Utils; CORBA severed). |
| `kernel/Kernel_mingw_gcc15.patch` | L | MinGW/gcc-15 fix (skipped under MSVC). |
| `kernel/Kernel_msvc_pthread.patch` | L | `pthread_self()` comparison against the Win32 pthread shim. |
| `kernel/Kernel_msvc_set_unexpected.patch` | L | `std::set_unexpected`/`set_terminate` removed in C++17 MSVC. |
| `kernel/Kernel_occt781.patch` | L | OCCT 7.8.1 API deltas in KERNEL. |
| `occt8/0003-boost-regex-str-enum.patch` | C | Boost regex `str(ENUM)` → `str(int(ENUM))`. |
| `occt8/0004-occt-8.0-compat.patch` | C | The OCCT-8.0 pass (streams, `::Raise()`→`throw`, NCollection). |
| `smesh/SMDS_UnstructuredGrid_vtk94.patch` | L | VTK ≥9.4 `vtkCellArray`/cell-type accessor changes. |
| `smesh/SMDS_MeshVolume_vtk96.patch` | L | VTK ≥9.6 `GetFaceStream()` signature (`vtkIdList*`). |
| `smesh/SMDS_VtkCellIterator_vtk96.patch` | L | VTK ≥9.6 `GetFaceStream()` on the cell-iterator path. |
| `smesh/SMESH_MeshEditor_vtk96.patch` | L | VTK ≥9.6 `GetCellLinks()`→`GetLinks()` rename. |
| `smesh/SMESH_Mesh.patch` | L | MED export made conditional (build without libMED/HDF5). |
| `smesh/SMESH_SMDS.patch` | L | SMDS standalone build fixups. |
| `smesh/SMESH_MeshAlgos.patch` | L | `SMESH_MeshAlgos` build fixups. |
| `smesh/SMESH_Controls.patch` | L | `SMESH_Controls` fixups (partly superseded by the tag). |
| `smesh/SMESH_ControlPnt.patch` | L | `SMESH_ControlPnt` build fixups. |
| `smesh/SMESH_Slot.patch` | L | `SMESH_Slot` build fixups. |
| `smesh/SMESH_File_mingw.patch` | L | MinGW file I/O fix (skipped under MSVC). |
| `smesh/SMESH_MesherHelper_msvc.patch` | L | MSVC `_DEBUG_`-only variable-order warning. |
| `smesh/SMESH_occt781.patch` | L | OCCT 7.8.1 API deltas in SMESH core. |
| `smesh/StdMeshers_Quadrangle_2D_msvc.patch` | L | `<windows.h>` `#define near` collision in the 2D mesher. |
| `smesh/StdMeshers_Adaptive1D.patch` | L | `StdMeshers_Adaptive1D` build fixups. |
| `smesh/StdMeshers_Projection_2D.patch` | L | `StdMeshers_Projection_2D` build fixups. |
| `smesh/StdMeshers_ViscousLayers.patch` | L | ViscousLayers build fixups (the payload algorithm). |
| `smesh/mefisto.patch` | L | Wire the f2c `trte.c` into the MEFISTO2 target. |

## OCCT toolkits linked & bundled

`_core.pyd` links OCCT dynamically; the wheel bundles (at delvewheel-repair time) every OCCT
toolkit it needs directly or transitively. All are conda-forge `occt=8.0.0`, LGPL-2.1 with the
exception (see [NOTICE.md](NOTICE.md)); this records *which* toolkits and *why*, not a new source.

- Modelling / meshing (present since B2–B3): TKernel, TKMath, TKG2d, TKG3d, TKGeomBase,
  TKGeomAlgo, TKBRep, TKTopAlgo, TKPrim, TKBO, TKMesh, TKShHealing, TKOffset, plus the
  DataExchange STL toolkit **TKDESTL** (the OCCT-8.0 rename of TKSTL — see the patch note above).
- **DataExchange + OCAF/XDE (added for B1 `read_step_xde`/`write_step_xde`)**: **TKDESTEP**
  (STEP reader/writer; OCCT-8.0 rename of TKSTEP, mirroring the TKSTL→TKDESTL precedent),
  **TKXCAF**/**TKVCAF** (XDE shape/colour/name tools), **TKLCAF**/**TKCAF**/**TKCDF** (OCAF
  document core), **TKXSBase** (data-exchange base). Explicitly listed in the root
  `CMakeLists.txt` `_core` link block. `ci/check_wheel.py` asserts these are bundled.

## Reference-only repositories

Several other SMESH-adjacent projects were used for guidance but never copied from directly:
[trelau/SMESH](https://github.com/trelau/SMESH), [trelau/pySMESH](https://github.com/trelau/pySMESH)
(the closest thing to a prior pybind11 binding for this library — useful for cross-checking API names),
[trelau/pyOCCT](https://github.com/trelau/pyOCCT), SalomePlatform's [shaper](https://github.com/SalomePlatform/shaper)
and [geom](https://github.com/SalomePlatform/geom) (Phase 2, unrelated to the current milestones), and
[FreeCAD/FreeCAD](https://github.com/FreeCAD/FreeCAD) (kept only for one older NETGENPlugin file used to
confirm an API usage pattern). None of it ships in the wheel.
