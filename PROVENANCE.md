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

## Reference-only repositories

Several other SMESH-adjacent projects were used for guidance but never copied from directly:
[trelau/SMESH](https://github.com/trelau/SMESH), [trelau/pySMESH](https://github.com/trelau/pySMESH)
(the closest thing to a prior pybind11 binding for this library — useful for cross-checking API names),
[trelau/pyOCCT](https://github.com/trelau/pyOCCT), SalomePlatform's [shaper](https://github.com/SalomePlatform/shaper)
and [geom](https://github.com/SalomePlatform/geom) (Phase 2, unrelated to the current milestones), and
[FreeCAD/FreeCAD](https://github.com/FreeCAD/FreeCAD) (kept only for one older NETGENPlugin file used to
confirm an API usage pattern). None of it ships in the wheel.

## License

Everything vendored here is LGPL-2.1 (or LGPL-2.1-or-later), same as this project. Static
linking a LGPL library into a closed-source application is fine under LGPL-2.1 as long as the
LGPL portions remain replaceable and their source stays available — which is the entire
reason this project is a separate open-source wheel rather than code embedded in flux.
