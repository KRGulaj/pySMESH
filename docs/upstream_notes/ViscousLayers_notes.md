# `StdMeshers_ViscousLayers` — Upstream Header/Source Notes

Source: `extern/smesh/src/StdMeshers/StdMeshers_ViscousLayers.hxx` (declarations) and
`StdMeshers_ViscousLayers.cxx` (implementation, ~13000 lines total — only the
public-API-relevant sections were read in full; the `VISCOUS_3D` internal namespace
implementing the actual prism inflation is not touched by the binding).

## Exact public surface

```cpp
class STDMESHERS_EXPORT StdMeshers_ViscousLayers : public SMESH_Hypothesis
{
public:
  StdMeshers_ViscousLayers(int hypId, SMESH_Gen* gen);

  void   SetBndShapes(const std::vector<int>& shapeIds, bool toIgnore);
  std::vector<int> GetBndShapes() const;
  bool   IsToIgnoreShapes() const;

  void   SetTotalThickness(double thickness);
  double GetTotalThickness() const;

  void   SetNumberLayers(int nb);
  int    GetNumberLayers() const;

  void   SetStretchFactor(double factor);
  double GetStretchFactor() const;

  enum ExtrusionMethod { SURF_OFFSET_SMOOTH, FACE_OFFSET, NODE_OFFSET };
  void   SetMethod( ExtrusionMethod how );
  ExtrusionMethod GetMethod() const;

  void SetGroupName(const std::string& name);
  const std::string& GetGroupName() const;
  static SMDS_MeshGroup* CreateGroup( const std::string&  theName,
                                      SMESH_Mesh&         theMesh,
                                      SMDSAbs_ElementType theType);

  SMESH_ProxyMesh::Ptr Compute(SMESH_Mesh&         theMesh,
                               const TopoDS_Shape& theShape,
                               const bool          toMakeN2NMap=false) const;

  bool IsShapeWithLayers(int shapeIndex) const;
};
```

## Correction to `plan.md`

`plan.md`'s B3 task 1 names the setter `SetFaces(ids, isToIgnore)`. **The real method is
`SetBndShapes(const std::vector<int>& shapeIds, bool toIgnore)`.** `shapeIds` are SMESHDS
shape indices (see `ProxyMesh_notes.md` — **not** the binding's own 1-based `face_id`
values without translation). Use this exact spelling in `bindings/viscous.cpp`.

`ExtrusionMethod` has exactly 3 values: `SURF_OFFSET_SMOOTH`, `FACE_OFFSET`,
`NODE_OFFSET`. `VLParams.method` should be a Python `enum.IntEnum` mirroring these three,
in this order (their integer values are persisted via `SaveTo`/`LoadFrom`, so do not
reorder).

## `Compute()` behavior — read carefully before writing `bindings/viscous.cpp`

1. **Requires SOLIDs.** `Compute()` explores `theShape` for `TopAbs_SOLID` — see
   `_ViscousBuilder::Compute` (`StdMeshers_ViscousLayers.cxx:2018`), which returns an
   error immediately if `theShape` has no SOLID (`"No SOLID's in theShape"`). The shape
   passed to `compute_viscous_layers` must be (or contain) a 3D solid, not a bare shell —
   confirm this at the Python API boundary and raise `PysmeshError` early with a clear
   message if the loaded `Shape` has no solids, rather than letting the opaque upstream
   error surface.

2. **Multi-solid shapes return an aggregate `SMESH_ProxyMesh`.** If `theShape` contains N
   solids, `Compute()` builds a per-solid `SMESH_ProxyMesh` internally and wraps them in
   one aggregate `SMESH_ProxyMesh(components)` (`StdMeshers_ViscousLayers.cxx:1339-1382`).
   The binding does not need special-case logic for N>1 — `GetProxySubMesh`/`GetFaces`
   dispatch transparently through the aggregate.

3. **CRITICAL — hard-failure error text is NOT returned from `Compute()`.**
   `StdMeshers_ViscousLayers::Compute()` (line 1340) calls
   `builder.Compute(theMesh, theShape)`, which returns a `SMESH_ComputeErrorPtr`. On
   failure (`err && !err->IsOK()`), the wrapper **discards `err` and returns a null/empty
   `SMESH_ProxyMesh::Ptr()`** (line 1347-1348). The only place the actual error text is
   stored is inside `_ViscousBuilder::error()` (line 1922), which writes into
   **`SMESH_subMesh::GetComputeError()` on the per-solid submesh** of the mesh
   (`sm->GetComputeError()`, keyed by `SMESH_subMesh* sm = mesh.GetSubMeshContaining(solidId)`),
   not into any value reachable from the public `Compute()` return.

   **Binding implication:** after `Compute()` returns a null `Ptr`, `viscous.cpp` must
   walk `TopExp_Explorer(shape, TopAbs_SOLID)`, call
   `theMesh.GetSubMesh(exp.Current())->GetComputeError()` for each solid, and collect
   every non-OK `myComment` string into `PysmeshError.details`. `plan.md`'s B3 task 2
   ("collect `SMESH_ComputeError` messages") is correct in intent but must be implemented
   via per-solid submesh lookup, not via any field on the returned `ProxyMesh::Ptr`.

4. **Partial success surfaces via `_warning`, not `_error`.** On the success path (line
   1363-1369), if a per-solid `_MeshOfSolid::_warning` is set and not OK, it's copied to
   `sm->GetComputeError()` too — so the same post-`Compute()` submesh walk described above
   also picks up partial-failure warnings. Distinguish "hard fail" (`Compute()` returned
   null) from "partial fail" (`Compute()` returned non-null but some submesh has a
   non-OK `GetComputeError()`) — the latter should populate `VLResult.failed_face_ids`
   rather than raising.

5. **`toMakeN2NMap` (3rd param, default `false`)** — when `true`, computes a node-to-node
   correspondence map (`_MeshOfSolid::_n2nMapComputed`) between original and shrunk
   surface nodes, used by callers that need to relate pre/post-VL surface meshes 1:1. Not
   required for Tier-1 (`prism_connectivity`/`node_coords` extraction does not need it) —
   leave at the default `false` unless a later milestone needs the correspondence.

## Group creation — read before B3 task 3 ("prism harvest")

`StdMeshers_ViscousLayers::CreateGroup(theName, theMesh, theType)`
(`StdMeshers_ViscousLayers.cxx:1450-1476`) is a **static** helper, separate from
`Compute()`. Internal callers (`StdMeshers_ViscousLayers.cxx:10661`) invoke it themselves
as `CreateGroup(eos->_hyp.GetGroupName(), *mesh, SMDSAbs_Volume)` during the volume-prism
creation pass — i.e. **`Compute()` creates the group itself internally** whenever
`GetGroupName()` is non-empty (checked via `ToCreateGroup()`, line 630), using
`SMDSAbs_Volume` as the element type. The binding does not need to call `CreateGroup`
itself — set `SetGroupName(params.group_name)` before calling `Compute()`, then look the
group up afterward via `theMesh.GetGroups()` (`plan.md`'s B3 task 3 approach is correct
as stated). If `group_name` is left empty, no group is created and the only way to
recover prisms is via the returned `ProxyMesh`'s wall-face proxy submeshes plus a full
volume-element scan — prefer requiring a non-empty `group_name` in `VLParams` to avoid
this fallback path entirely.

## Sources
- `extern/smesh/src/StdMeshers/StdMeshers_ViscousLayers.hxx` (full read)
- `extern/smesh/src/StdMeshers/StdMeshers_ViscousLayers.cxx` lines 1290-1480, 1900-1960,
  2018-2091, 10650-10670 (targeted read via grep + Read)
- Cross-check for the VL binding pattern: [trelau/pySMESH](https://github.com/trelau/pySMESH) `src/StdMeshers.cxx`
  (VL binding block from line ~280) — note this reference binds an **older** SMESH
  (8.3.0.4) where `SetBndShapes` may have been named differently; verify against this
  file's actual call before copying signatures, don't assume it matches 9.9.0 verbatim.
