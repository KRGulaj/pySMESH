// pySMESH binding — topology healing: unify_same_domain.
//
// Wraps OCCT's ShapeUpgrade_UnifySameDomain (TKShHealing) — a pure B-rep operation that
// merges adjacent faces lying on the same underlying surface (and collinear edges) into a
// single face/edge, removing the artificial seams that over-segmented STEP imports carry.
// Unlike a mesher's "compound" hint, this is real geometric healing: the seam face/edge is
// deleted from the topology, so a downstream mesher never places nodes along it.
//
// No SMESH here — this is OCCT-only and shares just the BREP bytes bridge, ShapeData index
// convention, and PysmeshError with the rest of pysmesh. Input and output both cross the
// boundary as raw BREP bytes (BRepTools::Read/Write), exactly like load_brep.
//
// Index convention (matches Shape.faces()/edges() and flux's tag composition): face/edge
// ids are 1-based TopExp::MapShapes ordinals with a per-kind type filter. face_map[i-1] is
// the new 1-based face id that original face id i survives as, or -1 if it was removed;
// edge_map is the same for edges. Merged originals share one survivor (many-to-one).

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <BRepCheck_Analyzer.hxx>
#include <BRepTools.hxx>
#include <BRepTools_History.hxx>
#include <BRep_Builder.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <Standard_Handle.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS_Shape.hxx>

#include "common.hpp"

namespace pysmesh {
namespace {

// Read a BREP shape from in-memory bytes (mirrors shape.cpp::load_brep, which is file-local
// there). Raises on parse failure or a null result.
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

// Serialize a shape back to BREP bytes.
py::bytes write_brep(const TopoDS_Shape& shape) {
  std::ostringstream stream;
  try {
    BRepTools::Write(shape, stream);
  } catch (const std::exception& e) {
    throw PysmeshError(std::string("BREP write failed: ") + e.what());
  }
  return py::bytes(stream.str());
}

// Map every original sub-shape (1-based in old_map) to its surviving 1-based id in new_map:
//   removed          -> -1
//   modified/merged  -> the (first) survivor recorded in history, looked up in new_map
//   unchanged        -> itself (IsSame) in new_map
// A survivor that cannot be located in the unified shape is a history/topology inconsistency
// and raises rather than silently emitting a wrong id.
py::array_t<std::int32_t> build_map(const TopTools_IndexedMapOfShape& old_map,
                                    const TopTools_IndexedMapOfShape& new_map,
                                    const Handle(BRepTools_History) & hist, const char* kind) {
  const int n = old_map.Extent();
  py::array_t<std::int32_t> out(static_cast<py::ssize_t>(n));
  std::int32_t* d = out.mutable_data();
  for (int i = 1; i <= n; ++i) {
    const TopoDS_Shape& s = old_map.FindKey(i);
    if (!hist.IsNull() && hist->IsRemoved(s)) {
      d[i - 1] = -1;
      continue;
    }
    TopoDS_Shape survivor = s;
    if (!hist.IsNull()) {
      const TopTools_ListOfShape& mod = hist->Modified(s);
      if (!mod.IsEmpty()) {
        survivor = mod.First();
      }
    }
    const int idx = new_map.FindIndex(survivor);
    if (idx == 0) {
      throw PysmeshError(std::string("unify_same_domain: original ") + kind + " id " +
                          std::to_string(i) +
                          " could not be mapped to the unified shape "
                          "(history/topology inconsistency).");
    }
    d[i - 1] = static_cast<std::int32_t>(idx);
  }
  return out;
}

py::dict unify_same_domain(const py::bytes& brep, bool unify_faces, bool unify_edges,
                            bool concat_bsplines, double linear_tol, double angular_tol_rad) {
  const TopoDS_Shape shape = read_brep(brep);

  // Original sub-shape maps (source of the 1-based ids), captured before Build().
  TopTools_IndexedMapOfShape old_faces;
  TopTools_IndexedMapOfShape old_edges;
  TopExp::MapShapes(shape, TopAbs_FACE, old_faces);
  TopExp::MapShapes(shape, TopAbs_EDGE, old_edges);

  // ctor arg order is (shape, UnifyEdges, UnifyFaces, ConcatBSplines).
  ShapeUpgrade_UnifySameDomain unifier(shape, unify_edges, unify_faces, concat_bsplines);
  unifier.SetLinearTolerance(linear_tol);
  unifier.SetAngularTolerance(angular_tol_rad);
  // Safe mode: operate on a copy so the input sub-shape maps above stay identity-stable and
  // the history bridges old -> new cleanly.
  unifier.SetSafeInputMode(true);

  TopoDS_Shape result;
  {
    py::gil_scoped_release release;
    try {
      unifier.Build();
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("ShapeUpgrade_UnifySameDomain::Build failed: ") +
                          e.what());
    }
    result = unifier.Shape();
  }

  if (result.IsNull()) {
    throw PysmeshError("unify_same_domain produced a null shape.");
  }
  // Fail loud on an invalid healed shape rather than shipping a corrupt BREP downstream.
  if (!BRepCheck_Analyzer(result).IsValid()) {
    throw PysmeshError("unify_same_domain produced an invalid shape "
                        "(BRepCheck_Analyzer reported errors).");
  }

  TopTools_IndexedMapOfShape new_faces;
  TopTools_IndexedMapOfShape new_edges;
  TopExp::MapShapes(result, TopAbs_FACE, new_faces);
  TopExp::MapShapes(result, TopAbs_EDGE, new_edges);

  const Handle(BRepTools_History) hist = unifier.History();

  py::dict out;
  out["brep"] = write_brep(result);
  out["n_faces_before"] = old_faces.Extent();
  out["n_faces_after"] = new_faces.Extent();
  out["n_edges_before"] = old_edges.Extent();
  out["n_edges_after"] = new_edges.Extent();
  out["face_map"] = build_map(old_faces, new_faces, hist, "face");
  out["edge_map"] = build_map(old_edges, new_edges, hist, "edge");
  return out;
}

}  // namespace

void bind_unify(py::module_& m) {
  m.def("unify_same_domain", &unify_same_domain, py::arg("brep"), py::arg("unify_faces"),
        py::arg("unify_edges"), py::arg("concat_bsplines"), py::arg("linear_tol"),
        py::arg("angular_tol_rad"),
        "Merge same-domain faces/edges of a BREP shape (OCCT ShapeUpgrade_UnifySameDomain). "
        "Returns a dict: unified BREP bytes, before/after face+edge counts, and int32 "
        "face_map/edge_map (old 1-based id -> new 1-based id, -1 if removed). Low-level: the "
        "pysmesh.unify_same_domain wrapper builds UnifyParams/UnifyResult around it.");
}

}  // namespace pysmesh
