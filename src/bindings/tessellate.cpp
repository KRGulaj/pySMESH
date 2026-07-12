// pySMESH binding — tessellation: tessellate.
//
// Wraps OCCT's BRepMesh_IncrementalMesh (TKMesh) to produce a lightweight triangulation of
// a B-rep shape for viewport rendering. Returns four flat NumPy arrays — nodes (N,3) float64,
// tris (M,3) int32 (0-based), tri_face_id (M,) int32 (1-based), normals (N,3) float64 —
// suitable for direct GPU upload via a zero-copy shared-memory path.
//
// Face ids in tri_face_id are 1-based TopExp::MapShapes ordinals with a faces-only type
// filter, identical to the ids Shape.faces() and Mesh.add_triangles() use. The per-kind map
// is built exactly as ShapeData does, so the ordinals are byte-for-byte identical for the
// same BREP bytes.
//
// Normals are evaluated at each BRepMesh UV node against the underlying Geom_Surface via
// GeomLProp_SLProps (TKGeomAlgo). Nodes at face boundaries are NOT welded: each face
// contributes its own node range, giving sharp edges at face seams and correct smooth shading
// within curved patches. Triangle winding is corrected for REVERSED face orientation so all
// outward normals are consistent.
//
// The GIL is released for BRepMesh_IncrementalMesh::Perform().
// Toolkits used: TKMesh (BRepMesh_IncrementalMesh, Poly_Triangulation),
//                TKBRep (BRep_Tool, BRepTools, BRep_Builder),
//                TKGeomAlgo (GeomLProp_SLProps),
//                TKG3d (Geom_Surface) — all already linked transitively via Geom + SMESH_*.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <GeomLProp_SLProps.hxx>
#include <Geom_Surface.hxx>
#include <Poly_Triangulation.hxx>
#include <Standard_Handle.hxx>
#include <Standard_Type.hxx>
#include <TopAbs_Orientation.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Trsf.hxx>

#include "common.hpp"

namespace pysmesh {
namespace {

// Deserialize BREP bytes into a TopoDS_Shape. Mirrors unify.cpp / shape.cpp.
TopoDS_Shape read_brep(const py::bytes& data) {
  const std::string buffer = data;
  std::istringstream stream(buffer);
  TopoDS_Shape shape;
  BRep_Builder builder;
  try {
    BRepTools::Read(shape, stream, builder);
  } catch (const std::exception& e) {
    throw PysmeshError(std::string("BREP read failed: ") + e.what());
  }
  if (shape.IsNull()) {
    throw PysmeshError("BREP read produced a null shape (empty or malformed data)");
  }
  return shape;
}

// Tessellate a BREP shape into render-ready flat arrays.
//
// Each face contributes its own node range (no coordinate welding). Triangle winding is
// corrected for REVERSED faces; per-node normals are evaluated from the surface UV via
// GeomLProp_SLProps. Nodes at degenerate surface points (poles, singular UV) carry a zero
// normal — the caller should handle or filter these.
py::dict tessellate(const py::bytes& brep, double lin_defl, double ang_defl, bool relative) {
  static constexpr double kPi = 3.14159265358979323846;

  if (!(lin_defl > 0.0)) {
    throw PysmeshError("lin_defl must be > 0 (got " + std::to_string(lin_defl) + ")");
  }
  if (!(ang_defl > 0.0) || !(ang_defl < kPi)) {
    throw PysmeshError("ang_defl must be in (0, pi) rad (got " + std::to_string(ang_defl) +
                       ")");
  }

  const TopoDS_Shape shape = read_brep(brep);

  // Build a per-kind face map matching ShapeData so our face ids are identical to
  // Shape.faces(), Mesh.add_triangles(), etc.
  TopTools_IndexedMapOfShape fmap;
  TopExp::MapShapes(shape, TopAbs_FACE, fmap);
  const int nfaces = fmap.Extent();

  // Run BRepMesh_IncrementalMesh with the GIL released — the mesher is pure C++ and
  // parallel=true is safe since we own this local shape copy.
  {
    py::gil_scoped_release release;
    BRepMesh_IncrementalMesh mesher(shape, lin_defl,
                                    static_cast<Standard_Boolean>(relative), ang_defl,
                                    Standard_True /*parallel*/);
    mesher.Perform();
  }

  // First pass: count total nodes and triangles so we can allocate once.
  std::vector<int> node_offset(nfaces + 2, 0);  // node_offset[fi] = global node start for fi
  int total_nodes = 0;
  int total_tris = 0;
  for (int fi = 1; fi <= nfaces; ++fi) {
    node_offset[fi] = total_nodes;
    TopLoc_Location loc;
    Handle(Poly_Triangulation) T =
        BRep_Tool::Triangulation(TopoDS::Face(fmap.FindKey(fi)), loc);
    if (!T.IsNull()) {
      total_nodes += T->NbNodes();
      total_tris += T->NbTriangles();
    }
  }

  // Allocate output arrays (C-contiguous, owned copies).
  py::array_t<double> nodes_arr({static_cast<py::ssize_t>(total_nodes), py::ssize_t{3}});
  py::array_t<std::int32_t> tris_arr({static_cast<py::ssize_t>(total_tris), py::ssize_t{3}});
  py::array_t<std::int32_t> tfi_arr(static_cast<py::ssize_t>(total_tris));
  py::array_t<double> normals_arr({static_cast<py::ssize_t>(total_nodes), py::ssize_t{3}});

  double* nodes = nodes_arr.mutable_data();
  std::int32_t* tris = tris_arr.mutable_data();
  std::int32_t* tfi = tfi_arr.mutable_data();
  double* normals = normals_arr.mutable_data();

  // Zero-init normals; degenerate surface points keep the zero vector.
  std::fill(normals, normals + 3 * total_nodes, 0.0);

  int tri_out = 0;

  // Second pass: harvest per-face triangulation and compute normals.
  for (int fi = 1; fi <= nfaces; ++fi) {
    const TopoDS_Face& face = TopoDS::Face(fmap.FindKey(fi));
    TopLoc_Location loc;
    Handle(Poly_Triangulation) T = BRep_Tool::Triangulation(face, loc);
    if (T.IsNull()) {
      continue;
    }

    const gp_Trsf trsf = loc.Transformation();
    const bool rev = (face.Orientation() == TopAbs_REVERSED);
    const int noff = node_offset[fi];
    const int nn = T->NbNodes();

    // Nodes: apply face location (rigid motion) to move from face-local to world space.
    for (int i = 1; i <= nn; ++i) {
      gp_Pnt p = T->Node(i);
      p.Transform(trsf);
      const int gi = noff + (i - 1);
      nodes[3 * gi + 0] = p.X();
      nodes[3 * gi + 1] = p.Y();
      nodes[3 * gi + 2] = p.Z();
    }

    // Normals: evaluate the underlying Geom_Surface at each node's UV coordinates.
    // BRepMesh always produces UV nodes for surface meshes (HasUVNodes() is always true
    // after IncrementalMesh::Perform on a non-null triangulation), but guard for safety.
    if (T->HasUVNodes()) {
      TopLoc_Location surf_loc;
      Handle(Geom_Surface) surf = BRep_Tool::Surface(face, surf_loc);
      const gp_Trsf surf_trsf = surf_loc.Transformation();

      // Reuse one SLProps instance across all nodes on this face (surface is fixed).
      GeomLProp_SLProps props(surf, 1 /*order*/, 1.0e-9 /*resolution*/);
      for (int i = 1; i <= nn; ++i) {
        const gp_Pnt2d uv = T->UVNode(i);
        props.SetParameters(uv.X(), uv.Y());
        if (props.IsNormalDefined()) {
          gp_Dir n = props.Normal();
          if (rev) {
            n.Reverse();
          }
          // surf_trsf is a rigid motion; Transform on a gp_Dir applies only the rotation.
          n.Transform(surf_trsf);
          const int gi = noff + (i - 1);
          normals[3 * gi + 0] = n.X();
          normals[3 * gi + 1] = n.Y();
          normals[3 * gi + 2] = n.Z();
        }
      }
    }

    // Triangles: reindex from face-local 1-based to global 0-based; fix winding for REVERSED.
    const int nt = T->NbTriangles();
    for (int k = 1; k <= nt; ++k) {
      int a = 0, b = 0, c = 0;
      T->Triangle(k).Get(a, b, c);
      const int ga = noff + (a - 1);
      const int gb = noff + (b - 1);
      const int gc = noff + (c - 1);
      tris[3 * tri_out + 0] = static_cast<std::int32_t>(ga);
      // Swap b↔c to flip winding and match the corrected outward normal for REVERSED faces.
      tris[3 * tri_out + 1] = static_cast<std::int32_t>(rev ? gc : gb);
      tris[3 * tri_out + 2] = static_cast<std::int32_t>(rev ? gb : gc);
      tfi[tri_out] = static_cast<std::int32_t>(fi);
      ++tri_out;
    }
  }

  py::dict out;
  out["nodes"] = nodes_arr;
  out["tris"] = tris_arr;
  out["tri_face_id"] = tfi_arr;
  out["normals"] = normals_arr;
  return out;
}

}  // namespace

void bind_tessellate(py::module_& m) {
  m.def("tessellate", &tessellate, py::arg("brep"), py::arg("lin_defl"), py::arg("ang_defl"),
        py::arg("relative") = false,
        "Tessellate a BREP shape into a triangulated render mesh "
        "(BRepMesh_IncrementalMesh + Poly_Triangulation harvest). "
        "Returns a dict: nodes (N,3) float64 world XYZ; tris (M,3) int32 0-based node "
        "indices; tri_face_id (M,) int32 1-based face ids matching Shape.faces(); "
        "normals (N,3) float64 outward unit normals from GeomLProp_SLProps at UV nodes "
        "(zero vector at degenerate surface points). "
        "lin_defl: chord deflection; ang_defl: angular deflection [rad]; "
        "relative: interpret lin_defl as fraction of bounding-box diagonal. "
        "GIL released for Perform(). Low-level: use pysmesh.tessellate for the dataclass "
        "wrapper.");
}

}  // namespace pysmesh
