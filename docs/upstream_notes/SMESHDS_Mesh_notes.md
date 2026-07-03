# `SMESHDS_Mesh` — Upstream Header/Source Notes

Source: `extern/smesh/src/SMESHDS/SMESHDS_Mesh.hxx` (full read, 678 lines) and targeted
reads of `SMESHDS_Mesh.cxx` (`ShapeToMesh`, `ShapeToIndex`, `SetNodeOnFace`,
`SetMeshElementOnShape`).

## Exact spellings confirmed (for `bindings/mesh.cpp`)

```cpp
virtual SMDS_MeshNode* AddNode(double x, double y, double z);
virtual SMDS_MeshNode* AddNodeWithID(double x, double y, double z, smIdType ID);

virtual SMDS_MeshFace* AddFaceWithID(smIdType n1, smIdType n2, smIdType n3, smIdType ID);
virtual SMDS_MeshFace* AddFaceWithID(const SMDS_MeshNode* n1, const SMDS_MeshNode* n2,
                                     const SMDS_MeshNode* n3, smIdType ID);
virtual SMDS_MeshFace* AddFace(const SMDS_MeshNode* n1, const SMDS_MeshNode* n2,
                               const SMDS_MeshNode* n3);

// Shape-reference overloads (preferred — see hazard below):
void SetNodeOnFace  (const SMDS_MeshNode* aNode, const TopoDS_Face&   S, double u=0., double v=0.);
void SetNodeOnEdge  (const SMDS_MeshNode* aNode, const TopoDS_Edge&   S, double u=0.);
void SetNodeOnVertex(const SMDS_MeshNode* aNode, const TopoDS_Vertex& S);
void SetMeshElementOnShape(const SMDS_MeshElement* anElt, const TopoDS_Shape& S);

// Integer-index overloads (DO NOT USE from the binding — see hazard below):
void SetNodeOnFace  (const SMDS_MeshNode* aNode, int Index, double u=0., double v=0.);
void SetNodeOnEdge  (const SMDS_MeshNode* aNode, int Index, double u=0.);
void SetNodeOnVertex(const SMDS_MeshNode* aNode, int Index);
void SetMeshElementOnShape(const SMDS_MeshElement* anElt, int Index);

int ShapeToIndex(const TopoDS_Shape& S) const;   // internal index lookup, if ever needed
const TopoDS_Shape& IndexToShape(int ShapeIndex) const;
```

All node/element factory and classification methods match `plan.md`'s B2 assumptions in
name. `FindNode` is declared on the base `SMDS_Mesh`, not `SMESHDS_Mesh` — confirm the
exact signature there before writing `add_triangles`'s node-resolution loop (not yet
audited in this pass; low risk, base-class lookup-by-ID is standard SMDS).

## CRITICAL — shape-index hazard (corrects an implicit assumption in `plan.md`)

`plan.md`'s B2 task 3/4 describe `classify_on_face(node_ids, face_id, uv)` and
`add_triangles(conn, face_id)` calling `SetNodeOnFace(node, face_id, u, v)` /
`SetMeshElementOnShape(elem, face_id)` — phrased as if `face_id` (the plan's own 1-based
ordinal, presumably from a `TopAbs_FACE`-only `TopExp_Explorer` as used in
`Shape.faces()`) can be passed straight into the **integer-`Index` overloads** shown
above.

**This is unsafe and must not be implemented that way.** `SMESHDS_Mesh::ShapeToMesh()`
(`SMESHDS_Mesh.cxx:104-140`) populates its internal shape-index map via:

```cpp
myShape = S;
TopExp::MapShapes(myShape, myIndexToShape);   // whole-shape, ALL sub-shape kinds
```

`TopExp::MapShapes(shape, map)` with no type filter walks the shape across **every**
`TopAbs_ShapeEnum` kind (compound, compsolid, solid, shell, face, wire, edge, vertex, in
that enum order) and assigns each unique sub-shape the next index **in that traversal's
first-occurrence order** — solids and shells are indexed before any face, so a shape's
first face is very unlikely to land on SMESHDS index 1. This numbering has **no fixed
relationship** to a `TopExp_Explorer(shape, TopAbs_FACE)`-only traversal (which is what
`Shape.faces()` naturally produces for the Python-facing `face_id`).

If `Mesh.classify_on_face(node_ids, face_id, uv)` were to call
`SetNodeOnFace(node, face_id, u, v)` (the `int Index` overload) using the plan's
`face_id`, nodes would silently be classified onto the wrong face (or onto a solid/shell
"sub-mesh" slot, since `NewSubMesh(Index)` doesn't distinguish shape kind) whenever the
shape has any solid/shell/wire/edge/vertex sub-shapes indexed before that face — i.e.
essentially always, for any shape with volume. This is a silent-corruption bug class, not
a crash — it would pass `Mesh.validate()` (which only checks
`GetTypeOfPosition() != SMDS_TOP_UNSPEC`, not *which* shape) and only surface downstream
as wrong viscous-layer geometry or wrong `inner_surface_face_map` attribution.

### Required fix (binding design decision — apply in B2/B3, not just noted)

`Mesh` must **always use the `TopoDS_Shape&` overloads**, never the `int Index`
overloads. Concretely:

1. `Mesh.__init__(shape)` must retain the same per-kind, explorer-ordered face/edge/vertex
   `TopoDS_*` lists that `Shape` built (i.e. `Mesh` holds a reference to the originating
   `Shape` object, or a copy of its ordered `vector<TopoDS_Face>` / `vector<TopoDS_Edge>` /
   `vector<TopoDS_Vertex>`), so a Python-facing `face_id`/`edge_id`/`vertex_id` can be
   resolved back to the exact `TopoDS_Face`/`TopoDS_Edge`/`TopoDS_Vertex` object used when
   `Shape.faces()`/`.edges()`/`.vertices()` first assigned that id.
2. `classify_on_face(node_ids, face_id, uv)` resolves `face_id → TopoDS_Face` via that
   list, then calls the **shape-reference** `SetNodeOnFace(node, face_topo, u, v)`.
   Same pattern for edge/vertex classification.
3. `add_triangles(conn, face_id)` resolves `face_id → TopoDS_Face` the same way, then
   calls `SetMeshElementOnShape(elem, face_topo)` (shape-reference overload).
4. Never call `ShapeToIndex()`/`SetNodeOnFace(..., int Index, ...)` from the binding.
   `ShapeToIndex()` remains useful only if some future feature needs to *report* SMESHDS's
   internal index (e.g. for debugging), never as an input derived from our own `face_id`.

This changes `Mesh`'s internal state shape versus what `plan.md` implied (a bare
`std::unique_ptr<SMESH_Mesh>` member) — it needs the id→`TopoDS_*` lookup tables too.
Cheap to build (store during `__init__`, same explorer pass `Shape` already did).

## Gap discovered — KERNEL vendoring is not yet in the repo at all

`SMESHDS_Mesh.hxx` and `SMESH_ProxyMesh.hxx` both `#include <smIdType.hxx>`. This header
is **not part of SMESH** — it is generated from SALOME KERNEL's
`Basics/smIdType.hxx.in` via `configure_file()`. Confirmed in both reference builds'
Kernel CMake scripts, e.g. [looooo/SMESH](https://github.com/looooo/SMESH) `cmake/Kernel/CMakeLists.txt:42`:

```cmake
configure_file(${Kernel_SRC_DIR}/Basics/smIdType.hxx.in ${Kernel_SRC_DIR}/Basics/smIdType.hxx)
```

Both [looooo/SMESH](https://github.com/looooo/SMESH) and [trelau/SMESH](https://github.com/trelau/SMESH)
resolve `Kernel_SRC_DIR` to a **git submodule** checkout of SALOME KERNEL (`external/Kernel` / `src/Kernel`
pointing at `git.salome-platform.org`'s or a fork's KERNEL repo) that was **not fetched** in upstream clones
(submodules present as empty directories only).
The Kernel `CMakeLists.txt` in both reference repos compiles a *minimal* slice of KERNEL —
not the whole module (which pulls CORBA/ORB) — specifically:

```
Basics/Basics_Utils.cxx
Basics/Basics_DirUtils.cxx
Basics/BasicsGenericDestructor.cxx
Basics/smIdType.hxx.in            (configured, not compiled)
SALOMELocalTrace/BaseTraceCollector.cxx
SALOMELocalTrace/FileTraceCollector.cxx
SALOMELocalTrace/LocalTraceBufferPool.cxx
SALOMELocalTrace/LocalTraceCollector.cxx
Utils/duplicate.cxx
Utils/OpUtil.cxx
Utils/Utils_SALOME_Exception.cxx
Utils/Utils_ExceptHandlers.cxx
```
plus all headers (`*.h*`) in those same three directories.

**`plan.md`'s "Current Repo State" table and B1 plan do not mention vendoring KERNEL at
all** — they only track `extern/smesh/`. This is a real gap: pySMESH needs a new
`extern/kernel/` (or `third_party/kernel/`) vendor tree containing exactly this file list,
sourced from SALOME KERNEL at a tag compatible with SMESH `V9_9_0` (needs its own version
check — KERNEL and SMESH are versioned together upstream, so pull KERNEL tag `V9_9_0` to
match, then verify `Basics/smIdType.hxx.in`'s `SALOME_USE_64BIT_IDS` conditional compiles
against `extern/smesh`'s expectations). This should be added as an explicit B1 task
("vendor minimal KERNEL slice") before the `cmake/Kernel/CMakeLists.txt` port task, not
assumed to already exist. Track this file list as `PROVENANCE.md`'s KERNEL stub-inventory
entry — it is vendored upstream source, not an invented stub, so it needs its own
upstream URL + commit, same as `extern/smesh`.

## Citation correction

`plan.md`'s "Important Note" on `SALOME_USE_64BIT_IDS` cites
`ref-freecad-smesh — SetupSalomeSMESH.cmake` as the source. **That file does not exist in
this project's local `ref-freecad-smesh` clone** — the clone is a sparse checkout
containing only `src/3rdParty/salomesmesh/**`; no `cMake/FreeCAD_Helpers/` directory was
fetched, and no `SALOME_USE_64BIT_IDS` string appears anywhere in the local clone. The
`SALOME_USE_64BIT_IDS` requirement itself is real and independently re-confirmed here
directly from `extern/smesh`'s own `smIdType` usage plus the KERNEL
`smIdType.hxx.in`/`SMESH_smIdType.idl.in` mechanism — but the specific FreeCAD citation
should be dropped from `PROVENANCE.md` since it cannot be verified from the material this
project actually has on disk.

## Sources
- `extern/smesh/src/SMESHDS/SMESHDS_Mesh.hxx` (full read)
- `extern/smesh/src/SMESHDS/SMESHDS_Mesh.cxx` lines 104-140, 188-234, 1375-1430 (targeted)
- `extern/smesh/src/SMDS/SMDS_Mesh.hxx` (smIdType include, line 46)
- `extern/smesh/idl/SMESH_smIdType.idl.in` (confirms `smIdType` is a KERNEL-provided
  configured typedef, `@SMESH_ID_TYPE@`)
- [looooo/SMESH](https://github.com/looooo/SMESH) `cmake/Kernel/CMakeLists.txt` lines 3-47 (KERNEL minimal
  file list + `smIdType.hxx.in` configure_file mechanism)
- [trelau/SMESH](https://github.com/trelau/SMESH) `cmake/Kernel/CMakeLists.txt` (cross-check, same
  pattern)
