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
there. Every SALOME tree is at the same release, `V9_16_0` (2026-05-20), since SALOME versions
its modules together.

| Component | Upstream | Version | Local path | Import |
|---|---|---|---|---|
| SALOME SMESH | [SalomePlatform/smesh](https://github.com/SalomePlatform/smesh) | tag `V9_16_0`, `b4e78157e55cf3be47713911f1e6cfa2ba8e9f1c` | `extern/smesh/` | squashed subtree: `d5c62b136` + merge `66014401a` |
| SALOME KERNEL | [SalomePlatform/kernel](https://github.com/SalomePlatform/kernel) | tag `V9_16_0`, `3de71b3a6d0b73f9dc011c4e0de36012253030c5` | `extern/kernel/` | squashed subtree: `a10077419` + merge `79e3f3671` |
| SALOME salome_bootstrap | [SalomePlatform/salome_bootstrap](https://github.com/SalomePlatform/salome_bootstrap) | tag `V9_16_0`, `679a7b147192b3e3ac798c6dca59034edec2f78b` | `extern/salome_bootstrap/` | squashed subtree: `cdcd692c3` + merge `4b3c6a672` |
| SALOME GEOM (`GEOMUtils` only) | [SalomePlatform/geom](https://github.com/SalomePlatform/geom) | tag `V9_16_0`, `7d0a18e1554c25bca2c2d9882e0621ea3f7f9c81` | `extern/geom/src/GEOMUtils/` | sparse copy of one directory: `06bb22494` |
| MEFISTO (carried forward) | SALOME SMESH | tag `V9_9_0`, `4df9beadff0df879bcabb7c5472188f18cd30b42` | `extern/mefisto2/` | verbatim copy: `0825b092b` (see below) |

Up to 4.2.2 the SALOME trees were at tag `V9_9_0`. Each subtree tree hash equals the upstream
tree of its tag, and each copied file equals its upstream blob.

GEOM is a sparse copy of one directory, not a full subtree — GEOM itself is a large CORBA/GUI
module we don't need, and SMESH only reaches into it for `GEOMUtils.cxx` (a handful of
standalone OCCT geometry helpers used by `SMESH_Mesh`, the `SMESH_Controls` classifier and the
Cartesian mesher). Since `V9_16_0`, `GEOMUtils.cxx` calls three functions of the GEOMAlgo
package, which SALOME moved out of GEOM into
[SalomePlatform/common_geometry_lib](https://github.com/SalomePlatform/common_geometry_lib)
(geom commit `f5db6d7cf`). pySMESH does not vendor that repository: the pySMESH patch
`geom/GEOMUtils_GEOMAlgo.patch` defines the three functions inside `GEOMUtils.cxx`, with bodies
verbatim from `common_geometry_lib` tag `V9_16_0` (`e7b5227096fef835a448004ee06cf5938850b1c4`,
`src/GEOMAlgo/GEOMAlgo_AlgoTools.cxx`, LGPL-2.1).

KERNEL is needed because SMESH's data structures (`SMESHDS_Mesh`, `SMESH_ProxyMesh`) depend
on `smIdType`, a KERNEL-defined typedef that decides whether node/element IDs are 32- or
64-bit. Since `V9_16_0` SALOME keeps KERNEL's basics in salome_bootstrap. pySMESH compiles four
directories: `Basics/`, `SALOMELocalTrace/` and `Exception/` (the `SALOME_Exception` class)
from `extern/salome_bootstrap/__RUN_SALOME__/`, and `Utils/` from `extern/kernel/src/`. The
CORBA/communication layer is never built and is compiled out via `SALOME_LIGHT`.
`smIdType.hxx` is generated from `Basics/smIdType.hxx.in`.

We build for 64-bit Windows, so `SALOME_USE_64BIT_IDS` is on and `smIdType` resolves to
`int64_t` everywhere.

The SALOME trace macros of `V9_16_0` (`SALOMELocalTrace/utilities.h`) write only when the
environment variable `SALOME_VERBOSE` is set to a value above 0 (KERNEL commits `5b7bbcc7` and
`e7d8bf15`). With it unset, SMESH writes no trace to stdout.

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

## MEFISTO carried forward from SMESH V9_9_0

SALOME removed the MEFISTO 2-D triangulator from SMESH in commit `9d7121c88` ("bos #29628
Remove MEFISTO 2D algorithm", committed 2022-06-08). pySMESH keeps `Mefisto2D` as public
API. The five files below are verbatim copies from SMESH tag `V9_9_0` (commit
`4df9beadff0df879bcabb7c5472188f18cd30b42`). Each blob hash equals the upstream blob.

| File | Local path | Upstream path at `V9_9_0` |
|---|---|---|
| `aptrte.cxx` | `extern/mefisto2/aptrte.cxx` | `src/MEFISTO2/aptrte.cxx` |
| `aptrte.h` | `extern/mefisto2/aptrte.h` | `src/MEFISTO2/aptrte.h` |
| `Rn.h` | `extern/mefisto2/Rn.h` | `src/MEFISTO2/Rn.h` |
| `StdMeshers_MEFISTO_2D.cxx` | `extern/mefisto2/StdMeshers_MEFISTO_2D.cxx` | `src/StdMeshers/StdMeshers_MEFISTO_2D.cxx` |
| `StdMeshers_MEFISTO_2D.hxx` | `extern/mefisto2/StdMeshers_MEFISTO_2D.hxx` | `src/StdMeshers/StdMeshers_MEFISTO_2D.hxx` |

The Fortran sources `trte.f` and `areteideale.f` are not carried. `trte.c` (above) replaces
`trte.f`. `aptrte.cxx` already defines the `areteideale_` that `trte.c` calls, and
`patches/smesh/mefisto.patch` gives it the f2c signature. These files are licensed LGPL-2.1,
same as SMESH itself.

`prepare.py` stages them where `V9_9_0` kept them (`MEFISTO2/` and `StdMeshers/`). Three
patches change them:

- `mefisto.patch` (looooo) wires the f2c `trte.c` in.
- The `StdMeshers_MEFISTO_2D.cxx` hunk of the OCCT 8.0 pass (`occt8/0004`) ports it to OCCT 8.
- `MEFISTO_2D_max_element_area.patch` (pySMESH) makes `MaxElementArea` bound the triangles.
  Before it, `aptrte` clamped the bound to the boundary edge lengths, so the hypothesis had no
  effect. A face with no 2-D hypothesis, or with `LengthFromEdges`, meshes as on `V9_9_0`.

## Patches

`patches/{kernel,geom,smesh,occt8}/*.patch`: 37 patches, applied by `prepare.py` in the order
of its `PATCH_MANIFEST`, which the index below follows.

- 12 come from looooo/SMESH's own patch set: the Windows/MSVC fixes and the MED strip.
- 2, the `occt8/` pair, come from conda-forge's `smesh-feedstock` recipe. It is the only working
  OCCT 8.0 compatibility pass for this code base that we found.
- 23 are pySMESH's own. Two make `V9_16_0` build (`geom/GEOMUtils_GEOMAlgo`,
  `smesh/SMESH_Gen_no_qt`). The other 21 fix SMESH defects. Each header states the defect, the
  root cause with the upstream lines, and what does not change.

NETGEN-related patches are left out, because pySMESH does not build NETGEN.

`patches/occt801/` is a different kind: it patches OCCT itself, not SMESH, and
`ci/build_occt.py` applies it, not `prepare.py`. See
[How OCCT is built](#how-occt-is-built).

Every patch is re-ported to the `V9_16_0` trees, and `prepare.py` applies each one exactly:
`patch -p1 -N --fuzz=0`, and any hunk that fails, is already applied or targets a file that is
not staged stops the run. Up to 4.2.2 the patches targeted looooo's `V9_9_0` branch pin, so
`prepare.py` applied them with `--fuzz=2` and skipped patches that the `V9_9_0` tag already
carried. On `V9_16_0` fuzz 2 placed two hunks in the wrong place, so the port removed both the
fuzz and the skip path. A re-ported patch keeps its original header and adds a note on what the
port changed.

### Patch index

Source key: **L** = `looooo/SMESH` patch series (the Windows/MSVC standalone-SMESH fork
conda-forge builds from); **C** = `conda-forge/smesh-feedstock` recipe; **P** = pySMESH.
MinGW/gcc-only patches are no-ops under our MSVC build.

| Patch | Src | State at `V9_16_0` | What it fixes |
|---|---|---|---|
| `kernel/Kernel.patch` | L | unchanged | KERNEL standalone base (Basics/trace/Utils; CORBA severed). |
| `kernel/Kernel_mingw_gcc15.patch` | L | unchanged | MinGW/gcc-15 fix (no-op under MSVC). |
| `kernel/Kernel_msvc_pthread.patch` | L | unchanged | `pthread_self()` comparison against the Win32 pthread shim. |
| `kernel/Kernel_msvc_set_unexpected.patch` | L | unchanged | `std::set_unexpected`/`set_terminate` removed in C++17 MSVC. |
| `geom/GEOMUtils_GEOMAlgo.patch` | P | new | `GEOMUtils.cxx` without `common_geometry_lib` (see above). |
| `smesh/mefisto.patch` | L | unchanged | Wire the f2c `trte.c` into the MEFISTO2 target. |
| `smesh/SMESH_Mesh.patch` | L | re-ported | MED export made conditional (build without libMED/HDF5); the `WITH_MED` guards now also cover `SMESH_DriverMesh.cxx`, new in `V9_16_0`, whose entry points throw without MED. |
| `smesh/SMESH_Gen_no_qt.patch` | P | new | `SMESH_Gen.cxx` includes `<QString>`/`<QProcess>` for a function compiled only `#ifndef WIN32` (SMESH `be238b4bb`); the includes take the same guard. |
| `smesh/SMESH_MeshAlgos.patch` | L | unchanged | `SMESH_MeshAlgos` build fixups. |
| `smesh/SMESH_Slot.patch` | L | unchanged | `SMESH_Slot` build fixups. |
| `smesh/SMESH_SMDS.patch` | L | unchanged | SMDS standalone build fixups. |
| `smesh/SMESH_occt781.patch` | L | re-ported | OCCT 7.8.1 API deltas; at `V9_16_0` only `DriverGMF.cxx` (`boost::filesystem::extension`) and `SMESHDS_DataMapOfShape.hxx` (`::HashCode`) remain. |
| `smesh/SMESH_File_mingw.patch` | L | unchanged | MinGW file I/O fix (no-op under MSVC). |
| `smesh/StdMeshers_Quadrangle_2D_msvc.patch` | L | unchanged | `<windows.h>` `#define near` collision in the 2D mesher. |
| `smesh/StdMeshers_Cartesian_3D_cancel.patch` | P | new | A cancel stops `Cartesian_3D` again, as in `V9_9_0`. SMESH `9f7d4a55e` passes `_computeCanceled` by value into `Grid::GridInitAndInterserctWithShape`, so its three cancel checks test a copy, and `Compute` ignores the result. The flag now goes by `volatile bool&`, and `Compute` returns false on a cancel. A run without a cancel builds the same mesh. |
| `smesh/StdMeshers_Cartesian_VL_cancel.patch` | P | new | A cancel stops `Cartesian_3D` with viscous layers (SMESH `c9294ee68`, `d3c3260cd`). `Compute` builds the offset shape, then calls itself; the inner call reset `_computeCanceled`, so a cancel during the offset step was lost, and `ViscousBuilder::MakeViscousLayers` had no check. The inner call keeps the flag, `Compute` returns false after the offset step on a cancel, and the layer step reads the flag at its phase boundaries. A run without a cancel builds the same mesh. |
| `smesh/StdMeshers_Adaptive1D_deflection.patch` | P | new | `Adaptive1D` holds its documented deflection. The deflection only seeded the size field, and next to a curved face the final segments missed it (1.037 times on a radius-1.5 rim; code unchanged since SMESH `7b33bc39f`). Each edge's segments are checked; an edge with a miss is placed again on a lowered, graded size field. Where the deflection would need segments shorter than min size, min size wins. An edge whose segments meet the deflection keeps them bit for bit. |
| `smesh/StdMeshers_Adaptive1D_bounds.patch` | P | new | Applies after the deflection patch. `Adaptive1D` keeps min size, max size and the factor 2 between neighbours on every edge, where it broke them (neighbours up to 3.23 times apart, segments at 0.998 times min size). The factor 2 holds along each edge and across the ends of a closed edge; across vertices nothing changes. An edge that meets all three rules keeps its segments bit for bit. |
| `smesh/StdMeshers_Adaptive1D_short_edges.patch` | P | new | Applies after the bounds patch. An edge shorter than `Adaptive1D`'s min size (a closed edge: three times min size) keeps the one or three segments the bounds allow, which are shorter than min size; the compute said nothing. The edge's sub-mesh now carries a `COMPERR_WARNING` with the edge length, the min size and the segment count, which reaches `ComputeReport.warnings`. The warning is set on the edge's own sub-mesh, because SMESH computes the edges of one hypothesis together. No node moves. |
| `smesh/StdMeshers_Distribution_table.patch` | P | new | The TABLE distribution places node k of N where the integral of its own table reaches k/N, to 1e-13 m on a 15 m edge. It failed with an empty error for a 3 um or 30 um wall cell and put nodes up to 2.8 m away for 300 um. `buildDistribution` solved each node from the previous one (SMESH `6b471bcc5`), so the errors added up; `computeParamByFunc` bisected to an absolute 1e-4 of the edge (SMESH `1f0895c06`). Each node now solves against the running total from the start, to 1e-14 of the edge; `FunctionTable::integral(i, d)` no longer reads two uninitialised values. REGULAR, SCALE and BETA_LAW do not change. |
| `smesh/StdMeshers_Distribution_expression.patch` | P | new | Applies after the table patch. The EXPRESSION distribution integrates its density with the adaptive Gauss-Kronrod rule to 1e-12 relative, so `1/(a+t)` nodes are exact to 1e-13 m on a 15 m edge (they were up to 11.1 m off). `FunctionExpr::integral` used one 20-point Gauss rule over the whole interval (SMESH `6b471bcc5`) and returned 0 on a failure. A density the rule cannot integrate, such as one with a pole, now fails the edge with a compute error. Cartesian spacing functions call only `FunctionExpr::value`, so no Cartesian grid changes. |
| `smesh/StdMeshers_Prism_3D_composite_side.patch` | P | new | Prism_3D no longer fails a solid it meshed with the error of a face it tried as the bottom and rejected: the search of `initPrism` (SMESH `9a54694a0`) never cleared those errors. A side face that is not meshed yet and has a composite horizontal side is refused, naming the way out (split the opposite edge at the same points), because `computeWalls` projects only onto the first edge of a top side and built a mesh that does not follow the edges (volume off by up to 1.3e-3 on report B1). A search that finds no bottom names the reason of every face; its text was empty. Other solids mesh as before. |
| `smesh/SMESH_Mesh_hypothesis_status.patch` | P | new | `SMESH_Mesh::AddHypothesis` keeps the worst status it found. Its last check (`CheckHypothesesOnSubMeshes`, SMESH `9dd9f7684`) assigned its result outright, so a `HYP_CONCURRENT` found on the ancestors never reached the caller, and a fatal status found on the sub-shapes was turned into `HYP_OK` after the hypothesis had been removed for it. |
| `smesh/StdMeshers_ViscousLayerBuilder_lifecycle.patch` | P | new | `StdMeshers_ViscousLayerBuilder` frees the inner `StdMeshers_ViscousLayers2D` and the offset builder with its `TmpMesh` (the destructor was empty), frees the first offset builder on a second `GetShrinkGeometry`, refuses `AddLayers` before `GetShrinkGeometry` (it read an unset pointer), and gives the inner hypothesis its own id, which left the builder out of the generator's map. |
| `smesh/StdMeshers_Cartesian_VL_group_2d.patch` | P | new | The two-step builder's layers in a face go into the named group: `makeFaces` did not mark the layer quadrangles and the group was always a volume group, so the group of a 2-D build stayed empty. The 3-D path is unchanged. |
| `smesh/StdMeshers_Cartesian_VL_duplicate_nodes.patch` | P | new | Cartesian_3D with viscous layers binds every node of an offset VERTEX or EDGE that an end plane of the grid passes through. The offset mesh doubles the nodes there: at a VERTEX the boundary faces used one copy and the projection bound the other, and the EDGE got no segment, so the projection skipped it. The layers failed with "bad mesh on offset geometry" (a hexagonal prism at spacing 0.1). The duplicates are merged, an EDGE is projected when it has nodes, and a failure clears the target mesh and names every SOLID. |
| `smesh/StdMeshers_Cartesian_3D_offset_mesh_leak.patch` | P | new | Cartesian_3D with viscous layers frees the offset `TmpMesh` it makes for each compute; it was freed on no path. |
| `smesh/StdMeshers_Cartesian_3D_viscous_submeshes.patch` | P | new | After a successful compute with viscous layers, Cartesian_3D marks the target's sub-meshes computed, as its plain path does: a seam EDGE or a pole VERTEX with no element of its own no longer reads as failed on a complete mesh. |
| `smesh/StdMeshers_Cartesian_3D_offset_small_cells.patch` | P | new | Cartesian_3D with viscous layers keeps every cut cell of its offset mesh that has volume. A cell dropped under `size_threshold` left a dent whose faces went to the SOLID, so no layer grew there: a cylinder had 192 layer cells with an uncovered facet and 5 % less layer volume. The plain path still applies `size_threshold`. |
| `smesh/StdMeshers_Cartesian_VL_offset_makers_leak.patch` | P | new | The viscous layer builder of Cartesian_3D frees the `BRepOffset_MakeOffset` it makes for each solid; each compute with layers, and each shrink of the two-step builder, leaked them (1 MB per compute of a 2 x 2 x 2 box at spacing 0.1). |
| `smesh/SMDS_UnstructuredGrid_links_leak.patch` | P | new | An SMDS mesh frees its inverse-element links with its grid. Since VTK 9 the grid holds them in a `vtkSmartPointer`, and the raw-pointer `Register`/`Delete` calls of `BuildLinks` left a second reference, so every mesh that built links leaked them on destruction or clear (0.8 MB per compute of a Cartesian mesh with viscous layers at spacing 0.1). |
| `smesh/MEFISTO_2D_max_element_area.patch` | P | new | Applies after `mefisto.patch`. `MaxElementArea` bounds the `MEFISTO_2D` triangles: `aptrte` clamped the edge bound up to the shortest boundary edge (so a tighter bound had no effect) and, over 2.05 x the longest, down to it (so a looser bound could give a finer mesh). The bound now refines below the boundary edges and never coarsens past the longest one; a face whose largest triangle still exceeds the bound gets a compute warning. A face with no 2-D hypothesis or with `LengthFromEdges` meshes as before. |
| `smesh/StdMeshers_CompositeHexa_3D_viscous_layers.patch` | P | new | `CompositeHexa_3D` refuses a `ViscousLayers` hypothesis with a compute error instead of crashing: it took the hypothesis by name, but the layer builder found none (`CompositeHexa_3D` lists no compatible hypothesis), and `Compute` read the null proxy mesh that came back. Made compatible, the layer cells on the side faces break its box grid, so the layers are refused. |
| `smesh/StdMeshers_Cartesian_VL_offset_error.patch` | P | new | Cartesian_3D with viscous layers says why it cannot offset the shape. `BRepOffset_MakeOffset` can report success with an empty solid (layers too thick, offset surfaces meeting), and the null shape went on with no error text: "SOLID 1: no message". `MakeOffsetSolid` now names the empty result and a failed glue, and `Compute` reports the error through `SMESH_Algo::error`: thrown as a `std::string` `SALOME_Exception`, it lost its first 7 characters in `SMESH_subMesh`. |
| `smesh/SMESH_subMesh_salome_exception_text.patch` | P | new | A `SALOME_Exception` that an algorithm's `Compute` throws keeps its whole text in the sub-mesh's compute error. `ComputeStateEngine` skipped 7 characters of every text, for the "Salome " of the "Salome Exception" prefix (SMESH `9a54694a0`), but only the `const char*` constructor adds that prefix; a text built from a `std::string` or an `SMESH_Comment` (KERNEL `b741d902`) lost its first 7 characters, and a shorter one was read past its end. The skip now applies only to a text that starts with the prefix, which keeps exactly its former text. |
| `occt8/0003-boost-regex-str-enum.patch` | C | unchanged | Boost regex `str(ENUM)` → `str(int(ENUM))`. |
| `occt8/0004-occt-8.0-compat.patch` | C | re-ported | The OCCT-8.0 pass (streams, `::Raise()`→`throw`, NCollection), extended to the code that is new in `V9_16_0`; the NETGEN and MeshVSLink sections, never staged, are dropped. |

### Patches that `V9_16_0` made obsolete

Removed by the port, each with the `V9_16_0` evidence:

| Patch | Src | Why it is gone |
|---|---|---|
| `geom/GEOMUtils.patch` | L | `V3d_Coordinate` is gone upstream (geom `922cc08ee`, OCCT 7.7 port); `GEOMUtils.cxx:920` uses `Standard_Real`. |
| `kernel/Kernel_occt781.patch` | L | salome_bootstrap `Basics/smIdType.hxx.in` has `<cstddef>` and the functor `smIdHasher` itself. |
| `smesh/SMESH_ControlPnt.patch` | L | `SMESH_ControlPnt.cxx:166-174` reads `Poly_Triangulation::Node()`. |
| `smesh/SMESH_Controls.patch` | L | `SMESH_ControlsClassifier.hxx:70` holds the projector by pointer. |
| `smesh/SMESH_MesherHelper_msvc.patch` | L | `SMESH_MesherHelper.cxx:5042` declares `nbfaces` unconditionally. |
| `smesh/StdMeshers_Adaptive1D.patch` | L | Its `TriaTreeData` port read the bounds of a NULL array and crashed `Adaptive1D` on any shape with a face; `StdMeshers_Adaptive1D.cxx:321-323` has a correct port. |
| `smesh/StdMeshers_Projection_2D.patch` | L | Guarded upstream by `OCC_VERSION_LARGE < 0x07070000` (`:70`, `:1900`). |
| `smesh/StdMeshers_ViscousLayers.patch` | L | Guarded upstream (`:57`, `:1832`). |
| `smesh/SMDS_UnstructuredGrid_vtk94.patch` | L | VTK 9.4 to 9.6 port upstream (SMESH `3978cf104`, `aa4ed9bd7`, `8940544ce`). |
| `smesh/SMDS_MeshVolume_vtk96.patch` | L | `GetFaceStream(id, vtkIdList*)` upstream (`aa4ed9bd7`). |
| `smesh/SMDS_VtkCellIterator_vtk96.patch` | L | As above (`aa4ed9bd7`). |
| `smesh/SMESH_MeshEditor_vtk96.patch` | L | `GetLinks()` upstream (`SMESH_MeshEditor.cxx:11763`, `:12121`). |

### Source edits made by pySMESH itself

Six deltas are applied by `prepare.py` as exact string replacements rather than as patch
files, because no upstream patch exists for them (`_apply_source_edits`, and
`_apply_smds_mesh_vtk_alloc` from looooo's `prepare.py`). They are modifications of
already-vendored SALOME source, not new vendoring. Each one was re-checked on `V9_16_0`: the
upstream code it fixes is unchanged there.

| Edit | File | What / why |
|---|---|---|
| `gethostname` include | `Kernel/Basics/Basics_Utils.cxx` | Needs `<winsock2.h>` on Windows. |
| `vtkPoints` pre-allocation | `SMESH/SMDS/SMDS_Mesh.cxx` | `SetNumberOfPoints(chunkSize)` instead of `0` in the constructor and in `Clear()`, so a first `InsertPoint` does not crash on Windows (looooo). |
| **`CompositeHexa_3D` include guard** | `SMESH/StdMeshers/StdMeshers_CompositeHexa_3D.hxx` | The header carries `StdMeshers_CompositeSegment_1D`'s guard (`_SMESH_CompositeSegment_1D_HXX_`) verbatim, so whichever of the two headers is included second is silenced and its class is never declared. The collision is symmetric, so no include order fixes it, and any translation unit needing both algorithms cannot compile. Renamed to `_SMESH_CompositeHexa_3D_HXX_`. |
| **`Prism_3D` adaptors override `EvalD0`** | `SMESH/StdMeshers/StdMeshers_Prism_3D.hxx` | OCCT 8.0 made `Adaptor3d_Curve::Value` and `Adaptor3d_Surface::Value` **non-virtual** inlines forwarding to a new virtual `EvalD0`, whose base implementation raises `Standard_NotImplemented`. Prism_3D's three 3-D adaptors still define `Value`, which now only *hides* the base one. For `TVerticalEdgeAdaptor` and `THorizontalEdgeAdaptor` that failed **`Prism_3D` on every solid**; for `TSideFace` it failed every compute that reaches the block approach (report B1: "Adaptor3d_Surface::EvalD0"). Each now overrides `EvalD0` to forward to its own `Value`. `Adaptor2d_Curve2d::Value` is still virtual in 8.0.1, so the three 2-D adaptors need nothing. |
| **`Prism_3D` per-generator helper singletons** | `SMESH/StdMeshers/StdMeshers_Prism_3D.cxx` | Three helper algorithms (`TQuadrangleAlgo`, `TProjction1dAlgo`, `TProjction2dAlgo`) are cached in function-local statics built against the **first** `SMESH_Gen` they ever see. SALOME has one process-global generator, so that holds there. pySMESH gives each `Mesher` its own, and `~SMESH_Gen` nullifies the `_gen` of every hypothesis registered with it — these singletons included. A **second** `Prism_3D` compute in one process then runs through a singleton whose generator is gone and **segfaults** (reproduced deterministically). Each site now rebuilds the singleton when the generator differs from the one it was built against; deleting the stale one is safe because `~SMESH_Hypothesis` is guarded on `_gen`. |
| **`ManifoldPart::process()` out-of-bounds face walk** | `SMESH/Controls/SMESH_Controls.cxx` | The walk starts at the requested face and wraps at the end of its own vector, but it advances the index itself and the wrap statement sits **after** a `continue` that skips an already-treated face. Since `findConnected()` treats a whole connected region at once, the last face is normally already treated when the walk reaches it, the wrap is skipped, and the index runs off the end — an **access violation**, measured on a three-face fixture. With the start element at index 0 the loop also cannot terminate by its own condition. Rewritten as a bounded modulo walk (`fi = (aStartIndx + fj) % aNbFaces`, `fj` from 0 to the face count), which is the documented intent: visit every face exactly once, starting at the requested one. Behaviour is otherwise unchanged; pinned by the `CTLBIND` section of `tests/probe`. |

Three `V9_9_0` edits are gone, because `V9_16_0` carries the fix:

| Edit | Why it is gone |
|---|---|
| `SMESH_TLink` default ctor + hasher functor (`SMESH_TypeDefs.hxx`) | Upstream moved the hasher into `SMESH_TLinkHasher` (`SMESH_TypeDefs.hxx:172-191`), and OCCT 8.0.1 maps copy-construct their keys, so no default ctor is needed. |
| `ElementsOnShape` out-of-line copy ctor / `operator=` (`SMESH_ControlsDef.hxx`, `SMESH_Controls.cxx`) | `Classifier` is a complete type in `SMESH_ControlsClassifier.hxx` (SMESH `df79d42e3`), so MSVC no longer meets C2036. |
| `V3d_Coordinate` removal (`GEOMUtils.cxx`) | Upstream (geom `922cc08ee`). |

**The include-guard edit is what un-blocks the algorithm catalogue.** It surfaced only when a
single translation unit first needed both composite algorithms, which is why the earlier
stages did not meet it: each SALOME translation unit includes one of the two headers, never
both. Nothing in the compiled behaviour changes — the guard is a name.

**Three of the six are runtime defects rather than build fixups**, and all three surfaced the
same way: by *driving* a class from a binding rather than linking and constructing it. The two
`Prism_3D` edits and the `ManifoldPart` one make a class that compiles and links perfectly
either fail on every input or corrupt memory. Construct-and-link checking cannot see any of
them, which is why the acceptance gates for each package run the code.

`StdMeshers` is built from a plain `file(GLOB)` with no exclusions, and the whole `StdMeshers`
family is exercised by the `v2_probe` target (`tests/probe`).

## OCCT toolkits linked & bundled

`_core.pyd` links OCCT dynamically. At repair time `delvewheel` bundles every OCCT toolkit in
its DLL closure. All of them come from our own build of OCCT 8.0.1 (see
[How OCCT is built](#how-occt-is-built)), under LGPL-2.1 with the exception (see
[NOTICE.md](NOTICE.md)). This section records which toolkits ship and why; it adds no source.
Up to 4.2.2 they came from the conda-forge package `occt=8.0.0`.

The root `CMakeLists.txt` names the toolkits that the bindings call. The wheel ships 29:

| Toolkits | Why |
|---|---|
| TKernel, TKMath, TKG2d, TKG3d, TKGeomBase, TKGeomAlgo, TKBRep | foundation classes and geometry |
| TKTopAlgo, TKPrim, TKBO, TKFillet, TKOffset, TKShHealing, TKHelix | the session's modelling operations: `BRepBuilderAPI_*`, primitives, booleans (`BRepAlgoAPI_*`), fillet and chamfer, sweeps and offsets, healing, helices |
| TKBool | `BRepFill_PipeShell`, which `BRepOffsetAPI_MakePipeShell` reaches |
| TKMesh | `BRepMesh_IncrementalMesh`, the tessellation |
| TKExpress | `ExprIntrp`, which the mesher's expression-based 1-D distributions reach |
| TKDE, TKXSBase, TKDESTEP, TKDEIGES | STEP and IGES read and write |
| TKXCAF, TKVCAF, TKLCAF, TKCAF, TKCDF | the OCAF and XDE document behind the STEP names and labels |
| TKV3d, TKService, TKHLR | pulled in by TKVCAF (TKV3d, TKService) and by TKV3d (TKHLR); pySMESH calls none of their API |

MSVC records an import only for a DLL whose import library supplies a symbol. A toolkit that the
link line names but that no binding calls is therefore no dependency, and `delvewheel` does not
bundle it. Today that is **TKFeat** (`BRepFeat_SplitShape`, proven usable by the `v2_probe`
target) and **TKDESTL** (`cmake/SMESH` links it for `DriverSTL`, which no binding calls).
`ci/check_wheel.py` asserts the toolkits that ship. Add a toolkit there in the same commit as the
binding that first calls it.

SMESH `V9_16_0` adds no toolkit. Its `SMESH_DriverShape.cxx` reads and writes STEP through
`STEPControl`, so `cmake/SMESH` links TKDESTEP and TKXSBase, which `_core` links already.

## How OCCT is built

Up to 4.2.2 OCCT came from the conda-forge package `occt=8.0.0`. Since then
`ci/build_occt.py` builds it from source. The local build and CI run the same script, and
CI caches the result on the script's input set (tag, commit, patches, CMake options, MSVC
toolset).

| Item | Value |
|---|---|
| Upstream | [Open-Cascade-SAS/OCCT](https://github.com/Open-Cascade-SAS/OCCT) |
| Tag | `V8_0_1` |
| Commit | `b8f597c677811d1f9f4d8a97f5ae2825c0353a42` (the script refuses any other) |
| Patches | `patches/occt801/*.patch`, applied in file-name order (table below) |
| Library type | shared (DLLs), bundled into the wheel and name-mangled by delvewheel |
| Toolchain | MSVC v143, CMake, Ninja |
| Found by | `CMakeLists.txt`, only under `PYSMESH_OCCT_ROOT`, only at exactly 8.0.1 |

The patches are our own. None is upstream, and none was reported upstream. Each file
starts with a header: the defect, the symptom, the root cause and the changed functions.
The script resets the source tree to the pinned commit, then applies each patch with
`git apply`. A patch that does not apply cleanly stops the build. Each patch's SHA-256
is part of the input set, so it is part of the CI cache key.

| Patch | Defect | OCCT file and function | What it changes |
|---|---|---|---|
| `0001-thrusections-generated-seam-edge.patch` | O1 | `BRepOffsetAPI_ThruSections.cxx`, `BRepOffsetAPI_ThruSections::Generated()` | A ruled loft through sections of one closed edge each: the walk along the longitudinal edges picked the next edge by list position, and a seam edge is listed twice. `Generated()` of a section vertex threw `FindFromKey` with 4+ sections, and returned a section edge with 3. The next edge is now the face's edge at the vertex that is not degenerated and not a section edge. |
| `0002-chamfer-three-corner-plane-line.patch` | F2 | `ChFi3d_ChBuilder_C3.cxx`, `ChFi3d_ChBuilder::PerformThreeCorner()` | A chamfer of three edges that meet at a corner: the curve where the corner plane cuts the end face is set only when the intersection gives one line. On a sphere wedge of 1.5 pi it gives two arcs, and the code read through the null curve: the process died. A null curve now throws `StdFail_NotDone`, which `ChFi3d_Builder::Compute()` turns into a faulty vertex, so the chamfer reports not done. A clean failure, not a computed corner. |
| `0003-pipe-unbuilt-sweep-face.patch` | F2 | `BRepFill_Pipe.cxx`, `BRepFill_Pipe::MakeShape()` | A face profile swept along a spine: `MakeShape()` never asks the sweep whether it is done, and read the shape type of a face the failed sweep left null: the process died. A null face now throws `StdFail_NotDone` from the `BRepOffsetAPI_MakePipe` constructor. A clean failure, not a computed pipe. |

The CMake options are the script's `CMAKE_OPTIONS`:

- `CMAKE_BUILD_TYPE=Release`, `CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`,
  `BUILD_LIBRARY_TYPE=Shared`, `BUILD_CPP_STANDARD=C++17`, `INSTALL_DIR_LAYOUT=Unix`.
- `BUILD_RELEASE_DISABLE_EXCEPTIONS=OFF`. OCCT's default is ON, which defines
  `No_Exception` and compiles out the `Standard_*_Raise_if` range checks.
- `BUILD_ENABLE_FPE_SIGNAL_HANDLER=OFF`, `BUILD_OPT_PROFILE=Default`,
  `USE_MMGR_TYPE=NATIVE`, `BUILD_WITH_DEBUG=OFF`, `BUILD_USE_PCH=OFF`.
- Every `BUILD_MODULE_*` is OFF. `BUILD_ADDITIONAL_TOOLKITS` names the 27 toolkits that a
  pySMESH target links. OCCT adds their closure: TKBool, TKDE, TKHLR and TKService.
- OFF: `USE_TBB`, `USE_FREETYPE`, `USE_FREEIMAGE`, `USE_RAPIDJSON`, `USE_DRACO`,
  `USE_OPENVR`, `USE_FFMPEG`, `USE_VTK`, `USE_TK`, `USE_OPENGL`, `USE_GLES2`, `USE_EIGEN`,
  `BUILD_GTEST`, `BUILD_DOC_Overview`, `BUILD_DOC_RefMan`, `INSTALL_TEST_CASES`.

No compiler flag is added. The flags are CMake's MSVC defaults plus OCCT's own
`adm/cmake/occt_defs_flags.cmake`: `/W4 /GR /EHa /fp:precise /MD /O2 /Ob2 /DNDEBUG /GL`,
and `/LTCG` at link time.

The options mirror the conda-forge feedstock that built `occt 8.0.0 all_h8ecc14b_202`
([conda-forge/occt-feedstock](https://github.com/conda-forge/occt-feedstock),
`recipe/bld.bat`) wherever an option can change behaviour. They differ only in what is not
built: Draw, VTK, FreeImage, RapidJSON, FreeType, OpenGL and Tcl/Tk. A control build of
8.0.0 from source, with these options, reproduced the 4.2.2 golden baseline bit for bit.

FreeType is OFF. TKService, which TKV3d and TKVCAF pull in, builds without it. FreeType
serves only OCCT's text rendering (`Font_*`, `StdPrs_BRepFont`), and pySMESH calls none of
it. So the wheel bundles no FreeType DLL.

## Reference-only repositories

These projects were read for guidance. Nothing in the wheel is copied from them, except the two
looooo/SMESH files and the patches that the sections above name.

| Project | Used for |
|---|---|
| [looooo/SMESH](https://github.com/looooo/SMESH) | the standalone Windows build, and the source of the **L** patches |
| [conda-forge/smesh-feedstock](https://github.com/conda-forge/smesh-feedstock) | the source of the **C** patches (the OCCT 8.0 pass) |
| [montylab3d/smesh](https://github.com/montylab3d/smesh) | a second standalone SMESH build, to cross-check build fixes |
| [trelau/SMESH](https://github.com/trelau/SMESH), [trelau/pySMESH](https://github.com/trelau/pySMESH) | a prior binding of this library, to cross-check API names |
| [trelau/pyOCCT](https://github.com/trelau/pyOCCT) | OCCT binding patterns |
| [SalomePlatform/geom](https://github.com/SalomePlatform/geom), [SalomePlatform/shaper](https://github.com/SalomePlatform/shaper) | SALOME geometry modules, to check how upstream calls GEOM and OCCT |
| [FreeCAD/FreeCAD](https://github.com/FreeCAD/FreeCAD) (`src/3rdParty/salomesmesh`) | FreeCAD's bundled SMESH copy, to compare its fixes with ours |
