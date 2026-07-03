# `SMESH_ProxyMesh` — Upstream Header Notes + Consumption Idiom

Source: `extern/smesh/src/SMESH/SMESH_ProxyMesh.hxx` (full read).

## Exact public surface used by the binding

```cpp
class SMESH_EXPORT SMESH_ProxyMesh
{
public:
  typedef boost::shared_ptr<SMESH_ProxyMesh> Ptr;

  class SMESH_EXPORT SubMesh : public SMESHDS_SubMesh
  {
  public:
    virtual smIdType             NbElements() const;
    virtual smIdType             NbNodes() const;
    virtual SMDS_ElemIteratorPtr GetElements() const;
    virtual SMDS_NodeIteratorPtr GetNodes() const;
  };

  const SMESHDS_SubMesh* GetSubMesh(const TopoDS_Shape& shape) const;
  const SubMesh*         GetProxySubMesh(const TopoDS_Shape& shape) const;  // nullable
  SMDS_ElemIteratorPtr   GetFaces() const;
  SMDS_ElemIteratorPtr   GetFaces(const TopoDS_Shape& face) const;
  smIdType               NbFaces() const;
  SMESHDS_Mesh*          GetMeshDS() const;
};
```

`GetProxySubMesh` takes a `const TopoDS_Shape&` directly — always the actual
`TopoDS_Face` object, never an integer index. This is consistent with the
`SetNodeOnFace`/`SetMeshElementOnShape` shape-vs-index duality documented in
`docs/upstream_notes/SMESHDS_Mesh_notes.md` (see "Shape-index hazard" there) — the
binding should standardize on **always passing `TopoDS_Face&`/`TopoDS_Shape&` overloads**
everywhere in `bindings/*.cpp` and never touch SMESHDS's internal integer index directly.

`GetProxySubMesh(face)` returns `nullptr` for faces that received no substitution (e.g. a
wall face where VL failed, or a face untouched by VL because it's not in `SetBndShapes`'
active set). B3 task 4 ("inner surface harvest") must null-check this per wall face and
route unsubstituted faces into `failed_face_ids` rather than dereferencing a null pointer.

## Consumption idiom — confirmed pattern from a real caller

`extern/smesh/src/NETGENPlugin/` **does not exist in this repo's `extern/smesh/`
subtree** (NETGENPlugin is a separate SALOME module, not vendored here). `plan.md`'s B0
task 3 named `NETGENPlugin_Mesher.cpp` as the cross-reference source; that exact file is
only available locally in `third_party_ref/ref-freecad-smesh/src/3rdParty/salomesmesh/src/NETGENPlugin/NETGENPlugin_Mesher.cpp`
(FreeCAD's vendored copy, SMESH ~7.7.1-era). This is a *different* viscous-layers variant
(`StdMeshers_ViscousLayers2D`, 2D/BRepMesh-side, not the 3D solid-prism algorithm this
project binds) but it confirms the same access idiom used throughout SMESH/NETGENPlugin
for proxy submesh consumption:

```cpp
// NETGENPlugin_Mesher.cpp:2838-2856 (2D VL variant, own OCCGeometry face map)
for ( int faceID = 1; faceID <= occgeo.fmap.Extent(); ++faceID )
{
  const TopoDS_Face& F = TopoDS::Face( occgeo.fmap( faceID ) );
  viscousMesh = StdMeshers_ViscousLayers2D::Compute( *_mesh, F );
  if ( !viscousMesh )
    return false;
}
...
// NETGENPlugin_Mesher.cpp:2940-2943 — GetProxySubMesh queried per-face via TopExp_Explorer,
// not via any custom integer index:
for (TopExp_Explorer face(occgeo.somap(iS), TopAbs_FACE); face.More(); face.Next())
  if ( Adaptor->GetProxySubMesh( face.Current() ) )
    { ... }
```

Confirms: real callers always re-derive the `TopoDS_Face&` from a shape explorer (their
own `occgeo.fmap`, analogous to our `Shape`'s face list) and pass that reference straight
into `GetProxySubMesh`/`Compute`. No caller anywhere in this codebase passes a raw
integer id into these 3D-mesh-facing APIs — the binding should not invent that pattern
either.

## Binding design consequence for B3

`bindings/viscous.cpp` step 4 ("inner surface harvest") should:
1. Resolve each wall `face_id` (our own 1-based ordinal from `Shape.faces()`) to its
   `TopoDS_Face` via the same face list the `Mesh`/`Shape` already built at construction
   (see `SMESHDS_Mesh_notes.md`'s "Shape-index hazard" — `Mesh` must retain this list).
2. Call `proxyMesh->GetProxySubMesh(face_topo)`; skip (record in `failed_face_ids`) if
   null.
3. Iterate `SubMesh::GetElements()` for the proxy triangles; record `face_id` per
   triangle into `inner_surface_face_map`.

## Sources
- `extern/smesh/src/SMESH/SMESH_ProxyMesh.hxx` (full read)
- `third_party_ref/ref-freecad-smesh/src/3rdParty/salomesmesh/src/NETGENPlugin/NETGENPlugin_Mesher.cpp`
  lines 2838-2943 (grep + targeted read) — confirms the idiom but is a 2D-VL / older-SMESH
  reference only; do not copy its API calls verbatim, only the access pattern.
