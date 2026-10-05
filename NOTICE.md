# Third-Party Notices

pySMESH (`pysmesh`, LGPL-2.1-only) is a pybind11 binding around the SALOME Platform meshing
stack and Open CASCADE Technology. Its one native library, `pysmesh/_core.pyd`, statically
links a minimal slice of SALOME SMESH and KERNEL, and netgen with SALOME's NETGENPlugin. It
links OCCT, Boost and VTK dynamically, and the wheel bundles those DLLs. This file lists
every third-party component that the shipped wheel contains or needs at runtime.

Upstream URLs, commits and the patch index are in [PROVENANCE.md](PROVENANCE.md).

| Component | License | How it ships | Obligation and how it is met |
|---|---|---|---|
| SALOME **SMESH** `V9_16_0` (vendored, patched) | LGPL-2.1 | **static** in `_core.pyd` | The complete corresponding source is this repository (`extern/smesh/` and `patches/`). The relinking right holds, because the whole binary rebuilds from that source. |
| SALOME **KERNEL** and **salome_bootstrap** `V9_16_0` (minimal slice) | LGPL-2.1 | **static** in `_core.pyd` | As SMESH. Source in `extern/kernel/` (`Utils`) and `extern/salome_bootstrap/` (`Basics`, `SALOMELocalTrace`, `Exception`). CORBA is compiled out (`SALOME_LIGHT`). |
| SALOME **GEOM** `V9_16_0` (`GEOMUtils` only) | LGPL-2.1 | **static** in `_core.pyd` | Source slice in `extern/geom/src/GEOMUtils/`. `patches/geom/GEOMUtils_GEOMAlgo.patch` compiles three functions of SALOME **common_geometry_lib** `V9_16_0` (LGPL-2.1) into it. |
| **MEFISTO2** (SMESH `V9_9_0`, carried forward), `trte.c` (f2c), **pthread** shim | LGPL-2.1 | **static** in `_core.pyd` | Source in `extern/mefisto2/` (`aptrte`, `Rn.h` and `StdMeshers_MEFISTO_2D` verbatim from SMESH `V9_9_0`; `trte.c` from `looooo/SMESH`) and `extern/pthread/` (from `looooo/SMESH`). |
| **netgen** `v6.2.2101` (vendored slice: `libsrc`, `nglib`; patched) | LGPL-2.1 | **static** in `_core.pyd` | The complete corresponding source is this repository (`extern/netgen/` and `patches/netgen/`, including SALOME's `netgen62ForSalome.patch`). The relinking right holds, because the whole binary rebuilds from that source. |
| **NETGENPlugin** `V9_16_0` (vendored: `src/NETGENPlugin`; patched) | LGPL-2.1 | **static** in `_core.pyd` | Source in `extern/netgenplugin/` and `patches/netgenplugin/`. As SMESH. |
| **GEOM PARTITION** (`Partition_*` in netgen's `libsrc/occ`) | LGPL-2.1 | **static** in `_core.pyd` | Part of the netgen slice; copyright OPEN CASCADE, EADS/CCR, LIP6, CEA/DEN, CEDRAT, EDF R&D, LEG, PRINCIPIA R&D, BUREAU VERITAS. As netgen. |
| **gzstream** (netgen's `libsrc/general/gzstream.*`) | LGPL-2.1 | **static** in `_core.pyd` | Part of the netgen slice; copyright Deepak Bandyopadhyay, Lutz Kettner. As netgen. |
| **moodycamel ConcurrentQueue** (netgen's `libsrc/core/concurrentqueue.h`) | BSL-1.0 | **static** in `_core.pyd`, header compiled in | Permissive. The header carries the licence text; this entry is the notice. |
| **zlib 1.3.2** (vendored: `extern/zlib/`) | zlib | **static** in `_core.pyd` | Permissive, notice only. Built from the release archive, unmodified; it serves netgen's gzstream. This entry is the notice (copyright Jean-loup Gailly and Mark Adler). |
| **Open CASCADE Technology (OCCT) 8.0.1** | LGPL-2.1 **with the OCCT exception** | **dynamic**, DLLs bundled into the wheel | **Modified.** `ci/build_occt.py` builds it from the upstream tag `V8_0_1` and applies the patches in `patches/occt801/`. PROVENANCE.md ("How OCCT is built") lists each patch and the build recipe. OCCT is linked dynamically, so the exception is not needed. The relinking right holds, because pySMESH is open and rebuilds from this repository. |
| **Boost 1.90** (`filesystem`, `serialization` DLLs) | BSL-1.0 | **dynamic**, DLLs bundled into the wheel | Permissive, notice only. This entry is the notice. |
| **VTK 9.6.2** | BSD-3-Clause | **dynamic**, DLLs bundled into the wheel | Permissive, notice only. This entry is the notice. `_core.pyd` links three VTK modules (`CommonCore`, `CommonDataModel`, `FiltersVerdict`). The wheel carries their closure: 17 VTK DLLs, which include VTK's own third-party modules (`verdict`, `kissfft`, `loguru`, `scn`, `vtksys`, `token`) under the licences that VTK carries for them. No rendering, IO or Python-wrapper module is bundled. |
| **oneTBB 2023.1** (`tbb12.dll`) | Apache-2.0 | **dynamic**, DLL bundled into the wheel | Pulled in by the conda-forge VTK build (`vtkCommonCore`, `vtkCommonDataModel`). OCCT is built without TBB. Apache-2.0 asks for the licence and this notice; the licence text is in the oneTBB distribution. |
| **{fmt} 12.1** (`fmt.dll`) | MIT | **dynamic**, DLL bundled into the wheel | Pulled in by the conda-forge VTK build. Permissive, notice only. This entry is the notice. |
| **pugixml 1.15** (`pugixml.dll`) | MIT | **dynamic**, DLL bundled into the wheel | Pulled in by `vtkCommonDataModel`. Permissive, notice only. This entry is the notice. |
| **Microsoft Visual C++ runtime** (`msvcp140.dll`) | Microsoft Visual C++ Redistributable terms | **dynamic**, DLL bundled into the wheel | Distributable code of the MSVC toolset. `delvewheel` bundles it so that the wheel runs on a machine without the redistributable installed. |
| **pybind11 3.x** | BSD-3-Clause | header-only, compile time | Notice only. This entry is the notice. |
| **NumPy** | BSD-3-Clause | runtime, a pip dependency | Notice only. |

The Boost, VTK, oneTBB, {fmt} and pugixml versions are those of the build environment
(`ci/environment.yml`). Only VTK is pinned. `delvewheel` name-mangles every bundled DLL, so none
of them can collide with a copy that the host process loads.

## OCCT toolkits bundled

OCCT ships as per-domain toolkit DLLs. The wheel bundles the 29 toolkits in the DLL closure of
`_core.pyd`. All have the component and licence of the OCCT row above. This list enumerates them;
it adds no obligation.

- **Foundation and modelling:** TKernel, TKMath, TKG2d, TKG3d, TKGeomBase, TKGeomAlgo, TKBRep,
  TKTopAlgo, TKPrim, TKBO, TKBool, TKFillet, TKOffset, TKShHealing, TKMesh, TKHelix, TKExpress.
- **Data exchange:** TKDE, TKDESTEP (STEP), TKDEIGES (IGES), TKXSBase.
- **OCAF and XDE** (STEP names and labels): TKXCAF, TKVCAF, TKLCAF, TKCAF, TKCDF.
- **Pulled in by TKVCAF and TKV3d:** TKV3d, TKService, TKHLR. pySMESH calls none of their own
  API.

MSVC records an import only for a DLL whose import library supplies a symbol. So a toolkit that
the link line names but that no binding calls is not a dependency, and `delvewheel` does not
bundle it. Today that is TKFeat and TKDESTL. `ci/check_wheel.py` asserts the toolkits that ship.

## Full license texts

- LGPL-2.1: [LICENSE](LICENSE). It covers this project, every vendored SALOME source, netgen
  (`extern/netgen/LICENSE`), NETGENPlugin (`extern/netgenplugin/LICENSE`), GEOM PARTITION and
  gzstream.
- zlib: `extern/zlib/LICENSE`. ConcurrentQueue BSL-1.0: in the header,
  `extern/netgen/libsrc/core/concurrentqueue.h`.
- The OCCT LGPL-2.1 exception, Boost BSL-1.0, VTK BSD-3-Clause, oneTBB Apache-2.0, {fmt} MIT,
  pugixml MIT, pybind11 BSD-3-Clause and NumPy BSD-3-Clause: carried by their upstream
  distributions (the OCCT repository at tag `V8_0_1`, and the conda-forge packages or the source
  repositories that PROVENANCE.md links for the rest).
