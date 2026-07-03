# pySMESH — Implementation Plan v0.1.0

**open-source · LGPL-2.1 · cp313-win_amd64**

- **Target:** `pysmesh-0.1.0-cp313-win_amd64.whl`
- **VTK pin:** `9.6.2`
- **OCCT:** `8.0.0` (static)
- **Boost:** `libboost-devel` (static)
- **Total effort:** ~2.5–4 weeks

---

## Current Repo State

| Component | Status | Details |
|-----------|--------|---------|
| `extern/smesh/` | ⚠️ needs patching | SALOME SMESH 9.9.0 — git subtree, raw upstream (full-platform CMakeLists) |
| `ci/environment.yml` | ✓ complete | python=3.13, vtk=9.6.2 (pinned), occt=8.0.0, full toolchain |
| `pyproject.toml / CMakeLists.txt / PROVENANCE.md` | ⚠️ empty stubs | created but not filled |
| Reference sources | ✓ ready | [trelau/SMESH](https://github.com/trelau/SMESH), [trelau/pySMESH](https://github.com/trelau/pySMESH), [looooo/SMESH](https://github.com/looooo/SMESH), [conda-forge/smesh-feedstock](https://github.com/conda-forge/smesh-feedstock), [SalomePlatform/shaper](https://github.com/SalomePlatform/shaper), [SalomePlatform/geom](https://github.com/SalomePlatform/geom), [trelau/pyOCCT](https://github.com/trelau/pyOCCT), [FreeCAD/FreeCAD](https://github.com/FreeCAD/FreeCAD), [montylab3d/smesh](https://github.com/montylab3d/smesh) |

---

## Milestone B0: Audit & Patch Inventory

**Effort:** 1–2 days

**Exit Criterion:** PROVENANCE.md filled; VL/ProxyMesh/SMESHDS headers read and vendored into `docs/upstream_notes/`; patch catalog assembled and confirmed against target (VTK 9.6.2, OCCT 8.0.0, MSVC).

### Header Audit — Read Before Writing Any Binding Code

1. **Read** `extern/smesh/src/StdMeshers/StdMeshers_ViscousLayers.hxx` and `.cxx`. Record the exact **`Compute()` signature**, `ExtrusionMethod` enum values, and group-collection behavior. Copy the relevant declarations into `docs/upstream_notes/ViscousLayers_notes.md`. This is the surface that the binding must mirror exactly.

2. **Read** `extern/smesh/src/SMESHDS/SMESHDS_Mesh.hxx`. Record exact spellings: `SetNodeOnFace`, `SetNodeOnEdge`, `SetNodeOnVertex`, `AddFaceWithID`, `SetMeshElementOnShape`. These are used in `bindings/mesh.cpp` — any spelling error is a link-time failure.

3. **Read** `extern/smesh/src/SMESH/SMESH_ProxyMesh.hxx`. Find the **proxy face access pattern** — how a plugin iterates the proxy submeshes of a wall face after `Compute()`. Cross-reference with `extern/smesh/src/NETGENPlugin/NETGENPlugin_Mesher.cpp` to confirm the consumption idiom. Document in `docs/upstream_notes/ProxyMesh_notes.md`.

### Patch Catalog

**Primary source:** `looooo_SMESH/patch/` (most current, tracks OCCT 8.0 + VTK 9.6).  
**Supplement with:** `conda_smesh/recipe/patches/` for OCCT 8.0 compat.  
**Reference:** `ref-trelau-smesh/patch/` only where looooo's version is absent.

| Family | Patch file | Source | What it fixes |
|--------|------------|--------|---------------|
| `0xx-strip` | `000-strip-med.patch` | `trelau: SMESH_Mesh.patch` | Makes MED export conditional on `WITH_MED` — lets SMESH build without libMED/HDF5 |
| `0xx-strip` | `001-strip-kernel-corba.patch` | `trelau: Kernel.patch` | Severs SALOME KERNEL's exception/trace utilities from CORBA (`SALOMEconfig.h`, `CORBA_SERVER_HEADER`). This is the "kernel shims" work. |
| `1xx-msvc` | `100-msvc-pthread-shim` (copy, not patch) | `looooo_SMESH/extra/pthread/` | Header-only POSIX-pthread on top of Win32 SRWLOCK/CreateThread — eliminates the PTHREAD_INCLUDE_DIR wart |
| `1xx-msvc` | `101-msvc-pthread.patch` | `looooo: Kernel_msvc_pthread.patch` | `myThread.p == pthread_self().p` comparison fix once the shim is in use |
| `1xx-msvc` | `102-msvc-set-unexpected.patch` | `looooo: Kernel_msvc_set_unexpected.patch` | `std::set_unexpected`/`set_terminate` removed in C++17 MSVC — no-op shim |
| `1xx-msvc` | `103-msvc-quadrangle.patch` | `looooo: StdMeshers_Quadrangle_2D_msvc.patch` | `<windows.h>` `#define near` macro collision in 2D mesher |
| `1xx-msvc` | `104-msvc-mesher-helper.patch` | `looooo: SMESH_MesherHelper_msvc.patch` | MSVC `_DEBUG_`-only variable-order warning |
| `1xx-msvc` | `105-msvc-export-macros.patch` | *write ourselves* | Collapse `SMESH_EXPORT`, `SMDS_EXPORT`, `SMESHDS_EXPORT`, `StdMeshers_EXPORT` to empty for static builds (`SMESH_STATIC` define) |
| `2xx-vtk` | `200-vtk94-cell-array.patch` | `looooo: SMDS_UnstructuredGrid_vtk94.patch` | VTK ≥9.4 `vtkCellArray` / cell-type accessor changes affecting SMDS grid compaction |
| `2xx-vtk` | `201-vtk96-getfacestream.patch` | `looooo: SMDS_MeshVolume_vtk96.patch` | VTK ≥9.6 `GetFaceStream()`: raw `vtkIdType*` out-param → `vtkIdList*` — the core SMDS/VTK 9.6 ABI break |
| `2xx-vtk` | `202-vtk96-celliterator.patch` | `looooo: SMDS_VtkCellIterator_vtk96.patch` | Same VTK ≥9.6 GetFaceStream signature change, cell-iterator path |
| `2xx-vtk` | `203-vtk96-getcelllinks.patch` | `looooo: SMESH_MeshEditor_vtk96.patch` | VTK ≥9.6 `vtkUnstructuredGrid::GetCellLinks()` renamed to `GetLinks()` |
| `2xx-vtk` | *string-replace in `prepare.py`* | `looooo: prepare.py → _apply_smds_mesh_vtk_alloc()` | Windows/VTK 9 `vtkPoints::InsertPoint` crash — `SetNumberOfPoints` pre-allocation in `SMDS_Mesh.cxx` |
| `3xx-occt` | `300-occt80-compat.patch` | `conda_smesh: 0004-occt-8.0-compat.patch` | 1198-line OCCT 8.0 pass: `Standard_Stream.hxx`→`Standard_OStream.hxx`, `TopTools_ListIteratorOfListOfShape` removal, `::Raise()`→`throw` across Geom/SMESH/NETGENPlugin |
| `3xx-occt` | `301-boost-regex.patch` | `conda_smesh: 0003-boost-regex-str-enum.patch` | Boost regex `matchResult.str(ENUM)` → `str(int(ENUM))` for stricter Boost enum-to-int overload |
| `4xx-vl` | `400-vl-backport-*.patch` | *diff trelau vs looooo VL files* | Cherry-pick any VL bugfixes in `looooo_SMESH` not yet in the trelau base (the actively-maintained path) |

### Important Note

**Check `SALOME_USE_64BIT_IDS`** — must be defined when `CMAKE_SIZEOF_VOID_P == 8` (64-bit Windows). Source: `ref-freecad-smesh — SetupSalomeSMESH.cmake`. Linker errors against SMESH on 64-bit are the symptom if this is missing.

### Tasks

1. Fill `PROVENANCE.md`: upstream SMESH URL + commit (`salome-platform/smesh` tag `V9_9_0`), looooo/SMESH commit `21ac164`, conda-forge smesh-feedstock commit `87d32bc`, trelau/SMESH as historical reference. Index every patch with its source repo + commit. List any stubs invented to isolate MED/Driver libs.

---

## Milestone B1: Skeleton Builds

**Effort:** 3–5 days

**Exit Criterion:** SMESH static libs (`smesh_kernel_shims`, `SMDS`, `SMESHDS`, `SMESH_core`, `StdMeshers`) link cleanly under MSVC. "Hello SMESHDS" test passes: construct `SMESH_Mesh` on a box shape, add 1 node, verify `NbNodes() == 1`.

### Directory Structure

Create the full directory skeleton:

```
cmake/
patches/
src/pysmesh/
src/bindings/
tests/fixtures/
examples/
.github/workflows/
docs/upstream_notes/
```

### pyproject.toml

Write `pyproject.toml`: scikit-build-core backend, pybind11 build dependency, package metadata (`name=pysmesh`, `version=0.1.0`, `license=LGPL-2.1-only`, Python ≥3.13, `platforms=win_amd64`). Include `[tool.scikit-build] cmake.build-type = "Release"` and the `py.typed` marker in `package-data`.

### Root CMakeLists.txt

1. **Find dependencies:**
   - VTK (dynamic): `find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel)` via `CMAKE_PREFIX_PATH=$CONDA_PREFIX/Library`
   - OCCT (static .lib): locate via `OpenCASCADE_DIR` or `CMAKE_PREFIX_PATH`
   - Boost (static/header-only): `find_package(Boost REQUIRED COMPONENTS filesystem thread serialization regex)` — the compiled components list comes from trelau's root `CMakeLists.txt` audit during B0
   - pybind11: via `find_package(pybind11 REQUIRED)`

2. **SMESH static targets:** Add subdirs in dependency order:
   - `cmake/Kernel/` → `cmake/SMDS/` → `cmake/SMESHDS/` → `cmake/SMESH/` → `cmake/StdMeshers/`
   - Each subdir's `CMakeLists.txt` is ported from `ref-trelau-smesh/cmake/` (already tested, working build scripts)
   - All targets compile `STATIC`

3. **Compiler flags (MSVC):** Add to all targets: `/EHsc /bigobj /MP /utf-8`  
   Defines: `NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS SALOME_USE_64BIT_IDS SMESH_STATIC`

4. **pybind11 extension:** `pybind11_add_module(_core src/bindings/module.cpp ...)`
   - Link: all 5 SMESH static targets + `VTK::CommonCore VTK::CommonDataModel` (dynamic)
   - Add options: `PYSMESH_WITH_NETGEN OFF`, `PYSMESH_DEV_ASSERTS ON` (in CI builds)

### cmake/ — Per-Module Build Scripts

1. Port `ref-trelau-smesh/cmake/Kernel/CMakeLists.txt` → `cmake/Kernel/CMakeLists.txt`. This provides the minimal SALOME KERNEL shims (Basics/trace/utils) that sever SMESH from the real CORBA-coupled KERNEL. Wire in the pthread shim (`extra/pthread/`, from looooo) as a header-only interface target here.

2. Port `ref-trelau-smesh/cmake/SMESH/CMakeLists.txt` → `cmake/SMESH/CMakeLists.txt` for SMDS + SMESHDS + SMESH core + StdMeshers. Reference the glob patterns and per-target `target_link_libraries` from the trelau version — these are the load-bearing lines that enumerate which OCCT toolkits each target needs.

### prepare.py

1. Write `prepare.py` (idempotent, Python 3.13):
   - (a) copy `cmake/` template CMakeLists files over corresponding `extern/smesh/src/` subdirs
   - (b) apply all patches in `patches/` in numeric order via `python-patch`
   - (c) apply the `SMDS_Mesh.cxx` string-replace for the VTK InsertPoint crash (porting `looooo_SMESH/prepare.py`'s `_apply_smds_mesh_vtk_alloc()`)
   - (d) guard: check a sentinel file to skip if already prepared
   - Source pattern: `looooo_SMESH/prepare.py`

2. Run `python prepare.py`. First CMake configure attempt. Expect link errors — iterate until all 5 static targets compile and link. Document every unexpected stub needed to isolate a dragged-in MED/Driver lib in `PROVENANCE.md`.

### Fixture Generation

1. Write `tests/fixtures/generate.py`: use OCCT directly (`BRepPrimAPI_MakeBox`, `BRepPrimAPI_MakeCylinder`) to produce `box.brep` and `cylinder.brep` via `BRepTools::Write`. These are deterministic — commit the output BREPs. Record generation command in the file's module docstring.

2. "Hello SMESHDS" smoke: a minimal `tests/test_hello_smeshds.py` — call `_core.create_test_mesh()` (a thin C++ test helper) that constructs `SMESH_Mesh` on a box, adds 1 node, returns `NbNodes()`. Assert it equals 1. This is B1's exit gate. Remove the test helper once Tier-1 bindings (B2) exist.

---

## Milestone B2: Tier-1 Bindings

**Effort:** 4–6 days

**Exit Criterion:** Full §6 API implemented. `test_shape.py` and `test_mesh_injection.py` green. `import pysmesh` passes VTK check and type stubs are mypy-clean.

### src/bindings/module.cpp

`PYBIND11_MODULE(_core, m)` entry. Register `PysmeshError` exception with `.details` and `.face_ids` attributes. Install OCCT Handle<> holder type declaration: `PYBIND11_DECLARE_HOLDER_TYPE(T, opencascade::handle<T>, true)` — copy the exact pattern from `ref-trelau-pysmesh/inc/pySMESH_Common.hxx`. Aggregate submodule bindings from `shape.cpp`, `mesh.cpp`, `viscous.cpp`.

### src/bindings/shape.cpp

1. **`load_brep(bytes) → Shape`:** `BRepTools::Read` from a `std::istringstream`. Raise `PysmeshError` on parse failure or null shape. Reference: `ref-trelau-pysmesh/src/SMESH.cxx` OCC wrapping patterns.

2. **`Shape.faces() → list[FaceInfo]`:** `TopExp_Explorer` over the shape for `TopAbs_FACE`. Per face: 1-based id, area+centroid via `GProp_GProps`/`BRepGProp::SurfaceProperties`, bbox via `Bnd_Box`/`BRepBndLib::Add`, uv_bounds via `BRep_Tool::Surface` + `GeomLib::GetKnotSequence` or face parameter range.

3. **`Shape.edges()`, `Shape.vertices()`:** Same Explorer pattern for edges (`TopAbs_EDGE`) and vertices (`TopAbs_VERTEX`). Edge: id, length via `GProp_GProps`/`BRepGProp::LinearProperties`, bbox, parameter bounds from `BRep_Tool::Curve`.

4. **`Shape.face_distance(face_id, points(N,3)) → NDArray[f64]`:** For each point, `BRepExtrema_DistShapeShape(face_topo, BRepBuilderAPI_MakeVertex(gp_Pnt(...))).Value()`. GIL released for the loop. This is the tie-break helper for Phase 2's face-map. No approximation — exact BRepExtrema.

### src/bindings/mesh.cpp

1. **`Mesh.__init__(shape)`:** Construct `SMESH_Mesh` via the SMESH_Gen pattern. Call `ShapeToMesh(shape.topo)`. Store in a `std::unique_ptr<SMESH_Mesh>` member. Reference: `ref-trelau-pysmesh/src/SMESH.cxx`, `ref-trelau-pysmesh/src/SMESHDS.cxx`.

2. **`Mesh.add_nodes(coords(N,3) f64) → node_ids(N,) i64`:** Iterate rows, call `SMESHDS_Mesh::AddNode(x,y,z)`, collect returned `SMDS_MeshNode*` IDs. Return as NumPy int64 array (copy — node IDs are not stable pointers).

3. **`Mesh.classify_on_face(node_ids, face_id, uv(N,2))`:** Call `SMESHDS_Mesh::SetNodeOnFace(node, face_id, u, v)` per node. `classify_on_edge(node_ids, edge_id, t(N,))`: `SetNodeOnEdge(node, edge_id, t)`. `classify_on_vertex(node_id, vertex_id)`: `SetNodeOnVertex(node, vertex_id)`.

4. **`Mesh.add_triangles(conn(M,3) i64, face_id)`:** For each row, resolve node pointers by ID (`SMESHDS_Mesh::FindNode`), call `AddFaceWithID(n0,n1,n2,face_id)` + `SetMeshElementOnShape(elem, face_id)`. Raise `PysmeshError` if any node ID is not found (name the offender).

5. **`Mesh.validate()`:** Walk all nodes and check every node has been classified (`GetPosition()->GetTypeOfPosition() != SMDS_TOP_UNSPEC`). Walk all wall face submeshes and check each has elements. Raise listing every gap — never silently pass an invalid mesh.

6. **`Mesh.stats() → MeshStats`:** `SMESHDS_Mesh::NbNodes()`, `NbFaces()`, per-face element counts. `Mesh.release()`: explicit destructor call. Wire `__exit__` and `__del__` to the same path.

### src/pysmesh/__init__.py

1. **DLL directory setup:** `os.add_dll_directory(Path(sys.prefix) / "Library" / "bin")` when detectable (belt-and-suspenders for VTK DLLs in conda + Nuitka). Then VTK version check: `import vtk; assert vtk.VTK_VERSION == _build_info.VTK_VERSION` — raise `ImportError` with both versions in the message on mismatch. Then `from ._core import ...` re-exports.

2. **Configure `_build_info.py.in` in CMake:** `VTK_VERSION` (the ONLY version checked at runtime), `OCCT_VERSION`, `BOOST_VERSION` (informational), `GIT_SHA`, `BUILD_DATE`, `WITH_NETGEN`.

3. **Write `_core.pyi` type stubs** covering all Tier-1 classes and functions with full signatures. Verify `mypy --strict` passes against them.

### Tests

1. **Write `tests/test_shape.py`:** box faces = 6, total area = 6·a² (analytic, cite in docstring). `face_distance` = 0.0 for a point on the face surface; = d (computed independently) for a point off-surface. Degenerate: empty bytes → `PysmeshError`.

2. **Write `tests/test_mesh_injection.py`:** node+element count round-trip. `validate()` raises naming an unclassified node ID. `validate()` raises naming a face_id that has no elements. Bad `face_id` in `add_triangles` raises with the id.

3. **Write `tests/test_build_info.py`:** monkeypatch `vtk.VTK_VERSION` to a wrong value → `ImportError` raised on import (reload the module in the test to trigger the check).

---

## Milestone B3: VL End-to-End

**Effort:** 3–5 days

**Exit Criterion:** `test_viscous_layers.py` green on box and wing fixtures. Wing fixture: zero inverted prisms. Phase 2 integration can begin against this output.

### src/bindings/viscous.cpp

1. **VLParams → SMESH setters:** Construct `StdMeshers_ViscousLayers`. Call setters: `SetTotalThickness`, `SetNumberLayers`, `SetStretchFactor`, `SetFaces(ids, isToIgnore)`, `SetMethod(ExtrusionMethod)`, `SetGroupName`. Exact method names and enum values from the B0 header audit. Reference: `ref-trelau-pysmesh/src/StdMeshers.cxx` — the VL binding block starting around line 280 is a near-complete working answer to this task.

2. **Compute + ProxyMesh:** Call `Compute(SMESH_Mesh&, TopoDS_Shape&)` → `SMESH_ProxyMesh::Ptr`. **Release the GIL for the entire call.** On failure, collect `SMESH_ComputeError` messages and surface them as `PysmeshError.details`. Distinguish complete failure (raise) from partial failure (populate `failed_face_ids` in the result).

3. **Prism harvest:** Find the named group (`params.group_name`) in `SMESH_Mesh::GetGroups()`. Iterate its elements (type `SMDSAbs_Volume`). For each 6-node prism element, read node IDs in VTK wedge order (nodes 0–2 = bottom triangle, 3–5 = top, same winding). Build `prism_connectivity(K,6) int32`.

4. **Inner surface harvest:** For each wall face id in `VLParams.face_ids` (or all wall faces if `is_ignore=True`), call `SMESH_ProxyMesh::GetProxySubMesh(face_topo)` → iterate its elements for the proxy triangles. Record source `face_id` per triangle → `inner_surface_face_map(S,) int32`. Mirror the NETGENPlugin consumption idiom documented in B0 step 3.

5. **Node extraction:** Walk all nodes in the post-compute mesh: `SMESHDS_Mesh::GetMeshIterator` over nodes → build `node_coords(P,3) f64` and `node_ids(P,) i64` in the same row order.

6. **Dev assertions** (compiled in when `PYSMESH_DEV_ASSERTS=ON`): verify prism node ordering against VTK wedge convention. Catches ordering bugs at CI time without shipping the check.

### Wing Fixture

1. Copy `lhs_wing.brep` from [trelau/pySMESH](https://github.com/trelau/pySMESH) `examples/models/` into `tests/fixtures/`. Check the license terms before redistributing — if restricted, generate a synthetic swept NACA profile with OCCT (`BRepOffsetAPI_ThruSections`) instead, and note the substitution in `PROVENANCE.md`.

2. Generate the matching surface mesh `wing_surface.npz` with classification tables (`face_id`, `uv`, `edge_id`, `t`). Document the generation script and its invocation in the file's module docstring. The generation script is *not* run by CI — the committed `.npz` is the fixture.

### Tests

1. **Write `tests/test_viscous_layers.py`** — box fixture, 5 layers, all 6 faces as walls:
   - K = 5 × n_wall_tris prisms (exact)
   - First-layer height = T(g−1)/(g^N−1) within 1% (cite geometric series formula in docstring)
   - Layer-height ratio = g within 1%
   - Inner surface tri count = wall tri count
   - `is_ignore` complement equivalence: both directions produce the same result

2. **Wing sub-test:** zero inverted prisms — compute scalar triple products via NumPy vectorized cross product. Assert `np.all(volumes > 0)`. Coverage: ≥98% of wall faces have layer elements, or listed in `failed_face_ids`.

3. **Degenerate inputs:** `total_thickness ≤ 0` raises `PysmeshError`. `n_layers ≤ 0` raises. `stretch_factor ≤ 1.0` raises. Invalid `face_ids` raises naming the bad ids.

4. **Write `examples/box_bl.py`** end-to-end smoke test: load box BREP → inject a surface mesh → `compute_viscous_layers` → print stats. This doubles as the CI smoke test and the README quickstart example.

---

## Milestone B4: Release v0.1.0

**Effort:** 1–2 days

**Exit Criterion:** DoD checklist fully verified. GitHub Release published with `pysmesh-0.1.0-cp313-win_amd64.whl`. Repo public under LGPL-2.1.

### .github/workflows/ci.yml

1. Single job, `windows-latest` runner. Steps:
   - (1) micromamba env from `ci/environment.yml` → `flux-pysmesh-build`
   - (2) `python prepare.py`
   - (3) `python -m build --wheel` via scikit-build-core
   - (4) `pytest tests/ -v`
   - (5) upload wheel as artifact
   - Add `PYSMESH_DEV_ASSERTS=ON` as a CMake flag in CI only.

2. **VTK-pin-drift check:** a small CI step reads flux's `environment.yml` for the `vtk>=` resolved version (or reads `flux-vtk-version.txt` if published) and fails with a diff if `ci/environment.yml`'s `vtk=` pin has drifted. This is the only cross-repo coupling point.

### Documentation

1. **Write `README.md`** (English): what pySMESH is and why it exists (the flux VTK/OCCT env-conflict incident), quickstart install snippet, `examples/box_bl.py` walkthrough, build-from-source instructions (`conda create` + `prepare.py` + `pip install`), the flux relationship, and a note on binary size (static OCCT baked in → tens of MB; this is expected, not a build error).

2. **Write `NOTICE.md`:** the five-row table from phase_1.md §2 — SMESH (LGPL-2.1, static), OCCT (LGPL-2.1 + exception, static), Boost (BSL-1.0, static), VTK (BSD-3, dynamic), pybind11 (BSD-3, header-only). Add any discovered stubs from the B1 link audit.

3. **Finalize `PROVENANCE.md`:** verify all fields are filled (upstream URLs + commits, patch index with one row per `patches/*.patch` file, stub inventory). Every patch must name its source repo + commit + what it fixes in one line.

### Definition of Done

Before tagging v0.1.0, verify each item:

- [ ] Install wheel into the unmodified flux env → `import pysmesh` passes VTK check → `examples/box_bl.py` runs to completion
- [ ] `conda list --explicit` diff before/after wheel install: exactly one new line (`pysmesh` pip entry) — no `occt`, no `boost`, no downgraded `vtk`/`pyside6`/`mkl`
- [ ] All tests green in CI on `windows-latest`; wing fixture produces zero inverted prisms
- [ ] `LICENSE` (LGPL-2.1), `NOTICE.md`, `PROVENANCE.md` complete — every patch has a source reference
- [ ] `_core.pyi` type stubs ship in the wheel; `mypy --strict` passes against them on an import
- [ ] Phase 2 `install_pysmesh.py` can consume the GitHub Release URL (VTK-pin check passes; smoke test passes)

---

## Key Risks

| Risk | Milestone | Mitigation |
|------|-----------|-----------|
| SMESH export macros fight static build (`dllimport`/`dllexport` on static targets) | B1 | Patch 105 collapses `*_EXPORT` to empty under `SMESH_STATIC` define — trelau already crossed most of this terrain |
| Hidden link dependency drags in MED/Driver libs | B1 | B1 exit criterion forces the discovery early; stub + record in PROVENANCE |
| VL `Compute()` API differs from the assumed signature | B0→B3 | B0 reads and vendors the header before any binding code is written |
| OCCT static build has CMake export quirks | B1 | `BUILD_LIBRARY_TYPE=Static` is an officially supported OCCT mode; budget one extra day in B1 |
| flux bumps VTK after a pySMESH release | B4+ | Import-time hard check fails loud; rebuild is one CI run against the new pin — no OCCT/Boost impact (static) |
| `_core.pyd` binary size surprises (static OCCT) | B4 | Documented in README; tens of MB is the correct outcome — still smaller than shipping OCCT DLLs + transitive deps |

---

**pySMESH v0.1.0 plan · 2026-07-02**  
target: cp313-win_amd64 · vtk=9.6.2 pin · LGPL-2.1
