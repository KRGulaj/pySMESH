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
| SALOME GEOM (`GEOMUtils`, `ShHealOper`) | [SalomePlatform/geom](https://github.com/SalomePlatform/geom) | tag `V9_16_0`, `7d0a18e1554c25bca2c2d9882e0621ea3f7f9c81` | `extern/geom/src/GEOMUtils/`, `extern/geom/src/ShHealOper/` | sparse copy of two directories: `06bb22494` (`GEOMUtils`), `6fb5fa5eb` (`ShHealOper`) |
| SALOME common_geometry_lib (`GEOMAlgo`, `BlockFix`, `LICENSE`) | [SalomePlatform/common_geometry_lib](https://github.com/SalomePlatform/common_geometry_lib) | tag `V9_16_0`, `e7b5227096fef835a448004ee06cf5938850b1c4` | `extern/common_geometry_lib/` | sparse copy of two directories and the licence: `74c292f4f` |
| PlaneGCS (FreeCAD `src/Mod/Sketcher/App/planegcs`, 11 files, and the repository `LICENSE`) | [FreeCAD/FreeCAD](https://github.com/FreeCAD/FreeCAD) | tag `1.1.4`, `4fd3bf320d9566a27e60069fc8387448aaa3a094` | `extern/planegcs/` | verbatim copy of the files: `53b32f76a` |
| Eigen (`Eigen/`, `COPYING.MPL2`, `COPYING.README`) | [libeigen/eigen](https://gitlab.com/libeigen/eigen) | tag `3.4.1`, `d71c30c47858effcbd39967097a2d99ee48db464` | `extern/eigen/` | verbatim copy of the tag tree: `5ead91d3b` |
| MEFISTO (carried forward) | SALOME SMESH | tag `V9_9_0`, `4df9beadff0df879bcabb7c5472188f18cd30b42` | `extern/mefisto2/` | verbatim copy: `0825b092b` (see below) |
| netgen (`libsrc`, `nglib`, `LICENSE`, `AUTHORS`) | [NGSolve/netgen](https://github.com/NGSolve/netgen) | tag `v6.2.2101`, `5e489319c60926daa836cecff39f0e92779032ba` | `extern/netgen/` | verbatim copy of the tag tree: `6732723fa` |
| SALOME NETGENPlugin (`src/NETGENPlugin`, `LICENSE`) | [SalomePlatform/netgenplugin](https://github.com/SalomePlatform/netgenplugin) | tag `V9_16_0`, `553d8c3d7977acf19bf87ff8a9b2c36a0691332b` | `extern/netgenplugin/` | verbatim copy of the tag tree: `b2176930a` |
| zlib | [zlib.net](https://zlib.net) | release 1.3.2, `zlib-1.3.2.tar.gz` (SHA-256 `bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16`) | `extern/zlib/` | the 48 top-level files of the archive, unmodified: `a9e977847` |

Up to 4.2.2 the SALOME trees were at tag `V9_9_0`. Each subtree tree hash equals the upstream
tree of its tag, and each copied file equals its upstream blob.

GEOM is a sparse copy of two directories, not a full subtree — GEOM itself is a large
CORBA/GUI module we don't need. SMESH reaches into it for `GEOMUtils.cxx` (a handful of
standalone OCCT geometry helpers used by `SMESH_Mesh`, the `SMESH_Controls` classifier and the
Cartesian mesher). `ShHealOper` is GEOM's shape-healing package; it is copied from the tag
`V9_16_0` (tree `ddf44997`), not from a later commit. Both compile into the static library
`Geom` (`cmake/Geom/CMakeLists.txt`).

Since `V9_16_0`, `GEOMUtils.cxx` calls three functions of the GEOMAlgo package, which SALOME
moved out of GEOM into
[SalomePlatform/common_geometry_lib](https://github.com/SalomePlatform/common_geometry_lib)
(geom commit `f5db6d7cf`). The pySMESH patch `geom/GEOMUtils_GEOMAlgo.patch` defines the three
functions inside `GEOMUtils.cxx`, with bodies verbatim from `common_geometry_lib` tag
`V9_16_0` (`e7b5227096fef835a448004ee06cf5938850b1c4`, `src/GEOMAlgo/GEOMAlgo_AlgoTools.cxx`,
LGPL-2.1), in a file-local namespace.

common_geometry_lib is now vendored as well: its `GEOMAlgo` and `BlockFix` packages, at the
same tag (trees `61d54511` and `cdf3a80c`), compile into the static library `CommonGeomLib`
(`cmake/CommonGeomLib/CMakeLists.txt`). `GeomAnaTool` is not vendored. `ShHealOper` needs
`GEOMAlgo_AlgoTools::FixCurveOnSurfaceTolerances`, so `Geom` links `CommonGeomLib`. The patch
`geom/GEOMUtils_GEOMAlgo.patch` stays. Its three functions have internal linkage
(`` `anonymous namespace'::GEOMAlgo_AlgoTools ``), and the library's are external functions of
the namespace `GEOMAlgo_AlgoTools`, so the two never meet at link time: `v2_probe` links both
objects with no duplicate or ambiguous symbol. With `CommonGeomLib` in every link that has
`Geom`, the patch is redundant in substance, and a later change can drop it in favour of the
library's own `GEOMAlgo_AlgoTools.hxx`.

`ShHealOper_ShapeProcess::Perform` copies its input with `TNaming_CopyShape::CopyTool`, a
static function of OCCT's TKCAF. It reads no OCAF document, and TKCAF is already in the link
set (STEP names), so it adds no DLL.

`ShHealOper_ShapeProcess` reads its operators and their parameters from a `Resource_Manager`
named `ShHealing`. That manager loads a resource file only when the environment variable
`CSF_ShHealingDefaults` or `CSF_ShHealingUserDefaults` names a directory
(`Resource_Manager::GetResourcePath`), and pySMESH ships no such file and sets neither
variable. `SetOperators` writes the key `ShapeProcess.exec.op` and `SetParameter` writes
`ShapeProcess.<name>` into the manager's per-object copy
(`ShapeProcess_Context::LoadResourceManager`), so the two calls are the whole input. An
operator that finds no value for a parameter uses the default in its code
(`ShapeProcess_OperLibrary.cxx`). The probe's `SH1` case runs the class this way.

netgen and NETGENPlugin are copied, not subtrees. Each copied directory's git tree hash
equals the tag's (netgen `libsrc` `be4cb70f`, `nglib` `d9d08d25`; NETGENPlugin
`src/NETGENPlugin` `ed7f0d87`). netgen is at 6.2.2101 because SALOME's
`netgen62ForSalome.patch` targets it; NETGENPlugin `V9_16_0` is the release of the other
SALOME trees. Left out of netgen: the GUI (`ng`), Python, the tests and the CMake build;
`cmake/Netgen/CMakeLists.txt` compiles `libsrc/{core,csg,general,geom2d,gprim,interface,
linalg,meshing,occ,stlgeom}` and `nglib`, with `visualization/visual_dummy.cpp` for the
viewer hooks. Left out of NETGENPlugin: `adm_local`, `bin`, `doc`, `idl`, `resources`,
`src/GUI` and `src/NETGEN`; `cmake/NETGENPlugin/CMakeLists.txt` compiles the ten
sources of the algorithms and hypotheses, not the `_i` (CORBA), `Remote`, `SA`, `Runner`
and `DriverParam` files.

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

## PlaneGCS and Eigen

PlaneGCS is FreeCAD's 2-D geometric constraint solver. The 11 files of
`src/Mod/Sketcher/App/planegcs` at tag `1.1.4` are copied verbatim into `extern/planegcs/`,
each blob equal to the tag's, with FreeCAD's root `LICENSE` (the LGPL-2.1 text). Every file
carries the notice "GNU Library General Public License ... version 2 of the License, or (at
your option) any later version" (LGPL-2.0-or-later), with the copyright of Konstantinos
Poulios (2011) or, for `Geo.cpp`, Victor Titov (2014). `cmake/PlaneGCS/CMakeLists.txt`
compiles them into the static library `PlaneGCS`.

The sources include four FreeCAD headers that pySMESH does not vendor. pySMESH provides each
as a small shim of its own in `cmake/PlaneGCS/`:

| FreeCAD header | Shim | What it does |
|---|---|---|
| `src/Mod/Sketcher/SketcherGlobal.h` (as `../../SketcherGlobal.h`) | `SketcherGlobal.h` | an empty `SketcherExport`: the library is static. `prepare.py` stages it two levels above the sources, where the relative include leads. |
| `src/FCConfig.h` | `FCConfig.h` | nothing: `GCS.cpp` includes it and uses none of its macros. |
| `src/Base/Console.h` | `Base/Console.h` | `Base::Console().log()` and `.warning()` discard their message: pySMESH writes nothing to the standard streams. |
| `src/boost_graph_adjacency_list.hpp` | `boost_graph_adjacency_list.hpp` | includes `<boost/graph/adjacency_list.hpp>` with `BOOST_ALLOW_DEPRECATED_HEADERS`, as FreeCAD's does. |

`PlaneGCS` compiles as C++20, because `Constraints.cpp` and `GCS.cpp` use `<numbers>`; the
rest of pySMESH stays at C++17. Boost.Graph and Boost.Math are header-only and come from the
build environment's Boost.

`System::diagnose` started two QR decompositions with `std::async`, so one could run on a
worker thread. `patches/planegcs/GCS_diagnose_serial.patch` gives both calls the policy
`std::launch::deferred`, which runs them on the calling thread. These were the only thread
starts on the solve and diagnose paths.

Eigen 3.4.1 is header-only. `extern/eigen/` holds `Eigen/` (its tree equals the tag's) and the
licence files `COPYING.MPL2` and `COPYING.README`; `unsupported/` is not vendored. `PlaneGCS`
puts `extern/eigen` first on its include path and defines `EIGEN_MPL2_ONLY` for every
translation unit that includes a PlaneGCS header, so an include of Eigen's LGPL code is a
compile error. Compiled with `/showIncludes`, `GCS.cpp` reads every Eigen header from
`extern/eigen`; the build environment's own Eigen (under `Library/include/eigen3`) is on no
include path.

## Algorithm code copied from GEOM

A few GEOM functions that pySMESH needs live in `GEOMImpl`, GEOM's driver layer, which is
not vendored (it depends on OCAF documents and `GEOM_Function`). They are copied into
pySMESH's own sources, unchanged but for the namespace they live in, with their origin and
licence in a comment above them. All are from SALOME GEOM tag `V9_16_0`
(`7d0a18e1554c25bca2c2d9882e0621ea3f7f9c81`), LGPL-2.1-or-later, used under LGPL-2.1.

| Function | Upstream file | Local file | Used by |
|---|---|---|---|
| `MAX_TOLERANCE`, `isSameEdge`, `isSameFace`, `isSameSolid` | `src/GEOMImpl/GEOMImpl_IShapesOperations.cxx` | `src/bindings/session_locate.cpp` | `Session.find_same` (GEOM's `GetSameIDs` rule) |
| `GEOMImpl_Fillet1d`, `GEOMImpl_Fillet1dPoint` (the whole class) | `src/GEOMImpl/GEOMImpl_Fillet1d.hxx`, `.cxx` | `src/bindings/geom_fillet1d.hpp`, `.cpp` (four upstream to-do comments removed, their facts kept as plain comments) | `Session.fillet_1d` |
| `MakeEdgeFromWire` (`GEOMImpl_ShapeDriver`), `AreEdgesC1` (`GEOMImpl_HealingDriver`) | `src/GEOMImpl/GEOMImpl_ShapeDriver.cxx`, `src/GEOMImpl/GEOMImpl_HealingDriver.cxx` | `src/bindings/geom_edge_from_wire.hpp`, `.cpp` (two upstream to-do comments removed, their facts kept as plain comments) | `Session.fillet_1d` (the joining of collinear edges) |
| `anotherVertex`, `takePlane`, `addEdgeRelation` (`GEOMImpl_Fillet1dDriver`) | `src/GEOMImpl/GEOMImpl_Fillet1dDriver.cxx` | `src/bindings/session_geomalgo.cpp` | `Session.fillet_1d` |
| `MakeSmoothingSurfaceUnClosed` (`AdvancedEngine_SmoothingSurfaceDriver`) | `src/AdvancedEngine/AdvancedEngine_SmoothingSurfaceDriver.cxx` | `src/bindings/session_geomalgo.cpp` | `Session.smoothing_surface` |
| `GEOMImpl_Block6Explorer` (the whole class) | `src/GEOMImpl/GEOMImpl_Block6Explorer.hxx`, `.cxx` | `src/bindings/geom_block6.hpp`, `.cpp` | `Session.make_quad`, `make_quad_from_vertices`, `make_hexa`, `opposite_face`, the block searches |
| The quad and hexa branches of `GEOMImpl_BlockDriver::Execute` (`BLOCK_FACE_FOUR_EDGES`, `BLOCK_FACE_TWO_EDGES`, `BLOCK_FACE_FOUR_PNT`, `BLOCK_SIX_FACES`, `BLOCK_TWO_FACES`) | `src/GEOMImpl/GEOMImpl_BlockDriver.cxx` | `src/bindings/session_blocks.cpp` | `Session.make_quad`, `make_quad_from_vertices`, `make_hexa` |
| `IsQuadrangle`, `AddBlocksFrom`, `FindConnected`, `HasAnyConnection`, the steps of `CheckCompoundOfBlocks`, `Propagate`, `GetOppositeFace`, `GetFaceByEdges`, `GetBlockNearPoint` (`GEOMImpl_IBlocksOperations`) | `src/GEOMImpl/GEOMImpl_IBlocksOperations.cxx` | `src/bindings/session_blocks.cpp` | `Session.check_blocks`, `propagation_chain`, `opposite_face`, `face_by_edges`, `find_block_near` |
| `MakeDiskSquare`, `MakeDiskHexagon`, `TransformShape`, `WPlaneTransform` (`AdvancedEngine_DividedDiskDriver`) | `src/AdvancedEngine/AdvancedEngine_DividedDiskDriver.cxx` | `src/bindings/session_advanced.cpp` | `Session.divided_disk` |
| `MakePipeTShape`, `MakeQuarterPipeTShape` (`AdvancedEngine_PipeTShapeDriver`) | `src/AdvancedEngine/AdvancedEngine_PipeTShapeDriver.cxx` | `src/bindings/session_advanced.cpp` | `Session.pipe_t_shape` |
| `MakeWireFromEdges` (`GEOMImpl_ShapeDriver`, its `GEOMImpl_WBM_FixTolerance` mode) | `src/GEOMImpl/GEOMImpl_ShapeDriver.cxx` | `src/bindings/session_advanced.cpp` | `Session.pipe_t_shape` |
| The `BLOCK_REMOVE_EXTRA` branch of `GEOMImpl_BlockDriver::Execute` (its steps) | `src/GEOMImpl/GEOMImpl_BlockDriver.cxx` | `src/bindings/session_blockfix.cpp` | `Session.remove_extra_edges` |

Four of these carry OCCT 8.0.1 port changes and nothing else: an iterator header that
OCCT 8.0.1 removed is replaced by the container header that keeps its typedef, and a
`Standard_Failure::Raise`, which OCCT 8.0.1 removed, is a `throw` of the same exception
with the same text. `MakeEdgeFromWire` also hands back the history of the `ShapeFix_Shape`
it runs, read from the fixer's own context; `MakeSmoothingSurfaceUnClosed` keeps only its
OCCT 7.7 and later adaptor branch.

`Session.fillet_1d` (`src/bindings/session_geomalgo.cpp`) also takes the driver logic of
`GEOMImpl_Fillet1dDriver::Execute` and `MakeFillet`, and of
`GEOMImpl_HealingDriver::FuseCollinearEdges` as the driver calls it, in their order. It
changes four things. Each step writes its history. The wire is a private copy, so
`FuseCollinearEdges` makes no copy of its own. A corner where no arc fits ends the call
with a refusal, where GEOM's last pass leaves the corner sharp. `GEOMUtils::FixShapeTolerance`
is spelled out in its steps, so that the history of its `ShapeFix_Shape` is read.

The block code changes only how the history is kept and how GEOM's driver reports a
failure. The copied explorer records the edges and faces it makes, and its `MakeFace` and
`MakeAnyFace` return the history of the face they build. `MakeAnyFace` heals an invalid
face with GEOM's default `ShHealOper_ShapeProcess`, whose sequence and parameters GEOM
reads from its `ShHealing` resource file; pySMESH runs the same steps
(`compose_shape_process` in `src/bindings/session_geomheal.cpp`) with the values of that
file at `V9_16_0` and the two overrides `MakeAnyFace` sets, and reads no file. GEOM's
sewing of six faces (`ShHealOper_Sewing`) is asked to record edges and vertices too (its
`SetHistoryLevel`), which changes no shape it builds. `GetBlockNearPoint` keeps its
blocks in the order of the model rather than in a map keyed by shape address, so that a
tie gives the same block in every run. `Propagate` starts at the named edge. The
builders run on copies of their inputs, because GEOM's code edits tolerances in place.

`Session.pipe_t_shape` (`src/bindings/session_advanced.cpp`) ports the method of
`AdvancedEngine_IOperations::MakePipeTShape` for the wall (no thickness reduction,
chamfer or fillet). GEOM builds it from GEOM operation calls on GEOM objects:
`MakePipeTShapePartition`, `MakePipeTShapeMirrorAndGlue`, `MakeGroups` and
`MakeInternalGroup`. pySMESH makes the OCCT and GEOM-algorithm calls that those
operations make, in GEOM's order, with GEOM's tolerances and argument checks; no GEOM
operation code is copied for them. Two things differ. The partition's
`GEOMAlgo_Splitter` runs serially, so that every run gives the same result. GEOM finds
the pipe-end faces in place of the common of the shape and its bounding box; pySMESH
takes the shape's faces that lie on that box (`GEOMAlgo_FinderShapeOn2`), which are the
same faces, sorted the same way. pySMESH also adds the faces on the outer cylinders by
the rule of `MakeInternalGroup`, a group GEOM does not make.

`Session.remove_extra_edges` also copies code of SALOME common_geometry_lib tag `V9_16_0`
(`e7b5227096fef835a448004ee06cf5938850b1c4`, LGPL-2.1-or-later, used under LGPL-2.1),
the vendored `extern/common_geometry_lib`:

| Function | Upstream file | Local file |
|---|---|---|
| `ComputeMaxTolOfFace`, `FixResult`, `BlockFix::RotateSphereSpace`, `BlockFix::RefillProblemFaces`, `BlockFix::FixRanges` | `src/BlockFix/BlockFix.cxx` | `src/bindings/geom_blockfix.cpp` |
| `BlockFix_BlockFixAPI::Perform` | `src/BlockFix/BlockFix_BlockFixAPI.cxx` | `src/bindings/geom_blockfix.cpp` |

These functions keep each step's context, modifier and substitution local, so their
history is out of reach; the copies write it. A script writes `geom_blockfix.cpp` from the
vendored sources, and every change is marked `pySMESH:` in it. `Perform` becomes a function
of its members, keeps only its OCCT 7.5.3 and later branches and the two face unification
values `RemoveExtraEdges` passes (0 and -1), and stops after a cancel. Each step's history
is kept, and `compose_histories` follows every input sub-shape through the steps in order.
pySMESH adds one check: a sphere or cylinder face that `RotateSphereSpace` or `FixRanges`
puts on a turned surface must keep its area within 1e-6 relative, where the turned shape is
used. GEOM has no such check. On a quarter-ball wedge, its rotation gives a face 3.8e-4
smaller whose mesh reaches 0.033 outside the wedge. The vendored classes
`BlockFix_SphereSpaceModifier` and `BlockFix_PeriodicSurfaceModifier` are called as they
are.

One function is adapted, not copied unchanged. `mesh_as_geom` in
`src/bindings/session_locate.cpp` takes the steps of `GEOMAlgo_AlgoTools::MeshShape` and
`GEOMAlgo_AlgoTools_CanBeMeshed`, from SALOME common_geometry_lib tag `V9_16_0`
(`e7b5227096fef835a448004ee06cf5938850b1c4`, `src/GEOMAlgo/GEOMAlgo_AlgoTools.cxx`,
LGPL-2.1-or-later, used under LGPL-2.1), for the arguments `GEOMUtils::CheckTriangulation`
gives it. Its one change is a progress range for the mesher, so that a cancel stops
`Session.entities_on` in its mesh. The mesh parameters are the ones of GEOM's call.

`Session.shape_process` (`src/bindings/session_geomheal.cpp`) takes the steps of
`ShHealOper_ShapeProcess::Perform`, from the vendored GEOM tag `V9_16_0`
(`src/ShHealOper/ShHealOper_ShapeProcess.cxx`), in its order, and keeps the
`TNaming_CopyShape` copy map that `Perform` keeps local. It changes three things and none
of the shape OCCT builds: the sequence reads no resource file (an empty resource name,
every resource set by the caller), its history goes down to the vertices, and each body
is processed without its placement and placed back.

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

`patches/{kernel,smesh,occt8}/*.patch` and `patches/geom/GEOMUtils_GEOMAlgo.patch`: 40
patches, applied by `prepare.py` in the order of its `PATCH_MANIFEST`, which the index below
follows.

- 12 come from looooo/SMESH's own patch set: the Windows/MSVC fixes and the MED strip.
- 2, the `occt8/` pair, come from conda-forge's `smesh-feedstock` recipe. It is the only working
  OCCT 8.0 compatibility pass for this code base that we found.
- 26 are pySMESH's own. Two make `V9_16_0` build (`geom/GEOMUtils_GEOMAlgo`,
  `smesh/SMESH_Gen_no_qt`). The other 24 fix SMESH defects. Each header states the defect, the
  root cause with the upstream lines, and what does not change.

`patches/netgen/` (9 patches) and `patches/netgenplugin/` (10 patches), applied by
`prepare.py` after the SMESH ones, patch the netgen and NETGENPlugin slices staged at
`staged/src/Netgen` and `staged/src/NETGENPlugin` (looooo's layout). They are in the
NETGEN part of the index below.

`patches/common_geometry_lib/` (4 patches), `patches/geom/ShHealOper_*.patch` (1) and
`patches/planegcs/` (1), applied by `prepare.py` after the NETGENPlugin ones, patch the
geometry libraries staged at `staged/src/CommonGeomLib`, `staged/src/Geom` and
`staged/src/PlaneGCS`. They are in the geometry part of the index below.

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
| `smesh/StdMeshers_Prism_3D_composite_side.patch` | P | new | Prism_3D meshes a prism whose side face has a composite horizontal side (report B1): `computeWalls` projected only onto the first edge of a top side, built a mesh that does not follow the edges (volume off by up to 1.3e-3), and 5.0.0 refused it. It now projects the opposite side as a whole, each node at its share of the side's length, as for a composite vertical side. It accepts the face only if the edges of the composite side join smoothly (a split edge), each split point gets a node, and each edge of the composite side gets the segments of its own 1-D hypothesis (`SMESH_Algo::Evaluate`, so no hypothesis is dropped); otherwise the search for the bottom goes on from where 5.0.0 stopped it, and a search that finds none names the reason of every face and the way out. A quadratic mesh on a composite side is refused. A prism that is a translation places its top nodes by it. Two not quadrilateral faces that share an edge no longer stop the search, which then skips a face that touches that edge (as the bottom, it made the edge vertical, and the sweep gave 8 segments to edges of 4). The search of `initPrism` (SMESH `9a54694a0`) no longer fails a solid it meshed with the error of a face it rejected. Other solids mesh as before, bit for bit. |
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
| `smesh/StdMeshers_CompositeHexa_3D_viscous_layers.patch` | P | new | `CompositeHexa_3D` builds viscous layers, one `ViscousLayers` per solid, and so does `Hexa_3D` on a block of more than 6 faces, which it hands over. Upstream it built the layers before it loaded the side grids, so the layer nodes on the shrunk side edges gave the grids extra rows and the box grid null nodes, and without the hypothesis in its compatible list it read a null proxy mesh: the process crashed (SMESH `0635c9fc8`, `0743fa8f5`); the Phase 4 version of this patch refused the layers. It now loads the side grids first, builds the layers, and maps each grid node to its node in the layer proxy mesh, as `Hexa_3D` does. Without layers the mesh is unchanged. |
| `smesh/StdMeshers_Cartesian_VL_offset_error.patch` | P | new | Cartesian_3D with viscous layers says why it cannot offset the shape, and names the largest total thickness for which it can. `BRepOffset_MakeOffset` can report success with an empty solid (layers too thick, offset surfaces meeting) or, thicker, with an invalid solid partly outside the shape, and either went on with no error text: "SOLID 1: no message", or cells outside the shape. `MakeOffsetSolid` now names the empty result, the invalid one (`BRepCheck_Analyzer`) and a failed glue, and ends the text with the largest workable thickness, found by bisection of the offset step only (at most 12 offset steps, on the failure path only; a successful compute makes no extra offset). `Compute` reports the error through `SMESH_Algo::error`: thrown as a `std::string` `SALOME_Exception`, it lost its first 7 characters in `SMESH_subMesh`. |
| `smesh/StdMeshers_Cartesian_VL_inverted_layers.patch` | P | new | Applies after the offset error patch. Cartesian_3D with viscous layers refuses layer cells that fold over. Where the offset surfaces are closer together than the grid can follow, the layer cells there were inverted and the compute succeeded (57 inverted hexahedra on a bored block at 0.2999). `MakeViscousLayers` now counts the inverted layer cells; if there is one, it clears the mesh and fails every SOLID with the reason. A mesh with no inverted layer cell is unchanged. |
| `smesh/StdMeshers_PolygonPerFace_2D_viscous_layers.patch` | P | new | `PolygonPerFace_2D` builds the `ViscousLayers2D` it is given. It built the layer quadrangles and then failed with "Less that 3 nodes on the wire": it read the wire with `StdMeshers_FaceSide::GetOrderedNodes()`, which neither scales nor orients the points of a proxy sub-mesh (SMESH `fe7d1d576`), and the hypothesis was not in its compatible list. A wire with layers is now read with `GetUVPtStruct()`, as `Quadrangle_2D` does, and the hypothesis is listed and checked. A face with no `ViscousLayers2D` meshes as before. |
| `smesh/SMESH_subMesh_salome_exception_text.patch` | P | new | A `SALOME_Exception` that an algorithm's `Compute` throws keeps its whole text in the sub-mesh's compute error. `ComputeStateEngine` skipped 7 characters of every text, for the "Salome " of the "Salome Exception" prefix (SMESH `9a54694a0`), but only the `const char*` constructor adds that prefix; a text built from a `std::string` or an `SMESH_Comment` (KERNEL `b741d902`) lost its first 7 characters, and a shorter one was read past its end. The skip now applies only to a text that starts with the prefix, which keeps exactly its former text. |
| `smesh/SMESH_subMesh_remove_hypothesis_state.patch` | P | new | A sub-mesh whose algorithm refused its hypotheses (`MISSING_HYP`) is checked again when a hypothesis is removed from it or from a shape above it. Upstream does nothing on those two events in that state (SMESH `bc37f0b49`), so `Hexa_3D`, which takes one `ViscousLayers`, stayed unmeshable after the second one was removed, and the compute left the solid empty with no error. Any other state, and a sub-mesh whose algorithm still refuses, are unchanged. |
| `occt8/0003-boost-regex-str-enum.patch` | C | unchanged | Boost regex `str(ENUM)` → `str(int(ENUM))`. |
| `occt8/0004-occt-8.0-compat.patch` | C | re-ported | The OCCT-8.0 pass (streams, `::Raise()`→`throw`, NCollection), extended to the code that is new in `V9_16_0`; the NETGEN and MeshVSLink sections, never staged, are dropped. |

### NETGEN patch index

Source key as above, plus **S** = SALOME (`netgenplugin` `src/NETGEN`) and **U** = netgen
upstream (a backport). All apply at fuzz 0 in this order.

| Patch | Src | What it does |
|---|---|---|
| `netgen/netgen62ForSalome.patch` | S | SALOME's netgen 6.2 patch, verbatim (`netgenplugin` `V9_16_0`, `src/NETGEN/netgen62ForSalome.patch`): the hooks NETGENPlugin calls. |
| `netgen/occgenmesh_OCCT76.patch` | L | Reads `Poly_Triangulation` nodes through `Node()` and `UVNode()` (OCCT 7.6+). |
| `netgen/Partition_Loop3d_occt781.patch` | L | The netgen section of looooo's `netgen_occt781.patch`: an include for OCCT 7.8. |
| `netgen/0004-occt-8.0-netgen-partition.patch` | C | The netgen `Partition_*` sections of conda-forge's `0004-occt-8.0-compat.patch`. |
| `netgen/0005-occt-8.0-netgen-occ.patch` | C | conda-forge's OCCT 8.0 pass on netgen's OCC files. |
| `netgen/occgeom_save_without_stl.patch` | P | `OCCGeometry::Save` refuses STL output, so `_core` does not import OCCT's TKDESTL. |
| `netgen/netgen_console_writes.patch` | P | Every direct `cout`/`cerr`/`clog` write of the compiled sources goes to netgen's own streams, which NETGENPlugin silences. |
| `netgen/netgen_no_ngprofile.patch` | P | NGPROFILE no longer writes `netgen.prof` into the working directory. |
| `netgen/e1d71a78_no_need_to_remove_archive_type_infos.patch` | U | Backport of netgen `e1d71a78` (v6.2.2105): a process that imported `_core` crashed at exit in about 6 of 20 runs (static destruction order of ngcore's archive register). |
| `netgenplugin/NETGENPlugin_occt8.patch` | C | The NETGENPlugin sections of conda-forge's `0004-occt-8.0-compat.patch`, re-ported to `V9_16_0`. |
| `netgenplugin/NETGENPlugin_local_size_by_subshape.patch` | P | A local size names a sub-shape by kind and ordinal, resolved through the mesh, not through the SALOME study (CORBA). |
| `netgenplugin/NETGENPlugin_runtime_containment.patch` | P | No working-directory change, no file, no redirect of `std::cout`; netgen's messages, logger and debug stream silenced or kept in memory; `SALOME_NETGEN_DISABLE_MULTITHREADING` not read. |
| `netgenplugin/NETGENPlugin_edge_local_size_ends.patch` | P | A local size on an edge holds up to its last vertex, and on an edge shorter than size / 1.5 (simple parameters gave a segment too few). |
| `netgenplugin/NETGENPlugin_face_maxh_index.patch` | P | `SetFaceMaxH` takes the 1-based face number: a size given to a face went to the face before it. |
| `netgenplugin/NETGENPlugin_curvature_before_read.patch` | P | The chordal error reads a face's curvature after OCCT computes it (OCCT 8's `RequireCurvature` read the cached value first): cylinder faces got no size. |
| `netgenplugin/NETGENPlugin_debug_text_threads.patch` | P | netgen's in-memory debug stream takes one write at a time: worker threads reallocated it at once and corrupted the heap. |
| `netgenplugin/NETGENPlugin_remesher_stl_topology.patch` | P | The remesher reads its STL topology through `STLGeometry*`: a cast to the second base read a garbage bounding box (random stack overflow). |
| `netgenplugin/NETGENPlugin_remesher_no_parameters.patch` | P | The remesher with no parameters hypothesis no longer reads address 0 (`LoadLocalMeshSize` of a null name). |
| `netgenplugin/NETGENPlugin_remesher_partial_result.patch` | P | A stopped or failed STL surface meshing (nglib returns NG_OK for both) is not taken for the remeshed mesh. |

### Geometry library patch index

All pySMESH's own (**P**). Each header states the reason in its first line. The OCCT 8.0.1
port changes includes and one type name; no compiled behaviour changes. All apply at fuzz 0
in this order.

| Patch | Root | What it does |
|---|---|---|
| `common_geometry_lib/GEOMAlgo_BlockFix_occt8_iterator_headers.patch` | `src/CommonGeomLib` | OCCT 8.0 removed the iterator headers of the TopTools, TColStd and BRep containers (for example `TopTools_ListIteratorOfListOfShape.hxx`); each include names the container header, which still declares the iterator typedef. |
| `common_geometry_lib/GEOMAlgo_occt8_standard_primitive_types.patch` | `src/CommonGeomLib` | OCCT 8.0 removed `Standard_PrimitiveTypes.hxx`; seven enum headers include `Standard_TypeDef.hxx` instead. |
| `common_geometry_lib/GEOMAlgo_occt8_bopds_list_of_pave_block.patch` | `src/CommonGeomLib` | OCCT 8.0 removed `BOPDS_ListOfPaveBlock`; `GEOMAlgo_WireSolid.cxx` names the list type `BOPDS_DS::PaveBlocks` returns. |
| `common_geometry_lib/GEOMAlgo_BlockFix_occt8_typedef_includes.patch` | `src/CommonGeomLib` | OCCT 8.0 headers no longer include the TopTools, TColStd and TColgp typedef headers; six files include the ones they use. |
| `geom/ShHealOper_occt8_iterator_headers.patch` | `src/Geom` | As the first one, for five iterator headers of `ShHealOper`. |
| `planegcs/GCS_diagnose_serial.patch` | `src/PlaneGCS` | `System::diagnose` runs its two QR decompositions on the calling thread (`std::launch::deferred` instead of the default launch policy of `std::async`). |

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

The geometry libraries add no toolkit either. `CommonGeomLib` links the toolkits of the OCCT
packages its sources include (each mapped by OCCT's own `PACKAGES.cmake`): TKernel, TKMath,
TKG2d, TKG3d, TKGeomBase, TKGeomAlgo, TKBRep, TKTopAlgo, TKBO, TKShHealing, TKOffset, TKMesh and
TKV3d. `ShHealOper` adds TKCAF (`TNaming_CopyShape`). PlaneGCS calls no OCCT. Every one of
these toolkits is in the closure above.

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

## How netgen is built

netgen, its core library (ngcore), nglib, zlib and NETGENPlugin build inside the `_core`
CMake project, as static libraries, from the staged trees: `cmake/Netgen/CMakeLists.txt`
(targets `pysmesh_zlib`, `ngcore`, `netgen`, `nglib`) and
`cmake/NETGENPlugin/CMakeLists.txt` (target `NETGENPlugin`). Every translation unit that
includes a netgen header (netgen, the plugin, `src/bindings/mesher_netgen.cpp` and the
probe's `tests/probe/probe_netgen.cpp`) compiles with the one define set
`PYSMESH_NETGEN_DEFINITIONS` of `cmake/Netgen/CMakeLists.txt`. Another set would give
`netgen::Mesh` another layout in one of them (`NO_PARALLEL_THREADS`, for example, is
not in the set). netgen links OCCT from the same install as the rest of `_core` and
adds no DLL to the wheel.
`netgen_version.hpp` is generated from `cmake/Netgen/netgen_version.hpp.in`.

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
