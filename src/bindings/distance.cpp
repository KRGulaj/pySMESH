// pySMESH binding — proximity & topology diagnostics: shape_distance, free_boundary_edges.
//
// Two OCCT-only queries that share just the BREP bytes bridge, the 1-based TopExp index
// convention, and PysmeshError with the rest of pysmesh (no SMESH here).
//
//   shape_distance(brep_a, brep_b) -> {distance, point_a, point_b}
//     Exact minimum distance between two shapes plus the witness points, one on each shape.
//     Wraps BRepExtrema_DistShapeShape (TKBRep) — the shape/shape generalisation of the
//     point/face query already exposed as Shape.face_distance. Unblocks the report's §2.9
//     entity-to-entity gap check (is the gap between two bodies large enough to mesh?).
//
//   free_boundary_edges(brep) -> edge_ids
//     The 1-based ids (matching Shape.edges()) of every edge bordered by exactly ONE face —
//     the naked edges of an open shell. On a watertight solid every edge is shared by two
//     faces, so the result is empty; a non-empty result localises the hole. Unblocks §7.3
//     leak detection ("show me the hole"). Uses the edge->face ancestor map (TopExp), the
//     same free-boundary criterion as ShapeAnalysis_FreeBounds, but returns original edge
//     ids directly rather than reconstructed wires.
//
// Index convention (matches Shape.edges() and flux's tag composition): edge ids are 1-based
// TopExp::MapShapes ordinals with a per-kind (edges-only) type filter.

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Pnt.hxx>

#include "common.hpp"

namespace pysmesh {
namespace {

// Read a BREP shape from in-memory bytes (mirrors shape.cpp::load_brep / unify.cpp::read_brep,
// which are file-local there). Raises on parse failure or a null result.
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

// Build a (3,) float64 NumPy array from a gp_Pnt (owns a copy).
py::array_t<double> point3(const gp_Pnt& p) {
  py::array_t<double> out(3);
  double* d = out.mutable_data();
  d[0] = p.X();
  d[1] = p.Y();
  d[2] = p.Z();
  return out;
}

py::dict shape_distance(const py::bytes& brep_a, const py::bytes& brep_b) {
  const TopoDS_Shape a = read_brep(brep_a);
  const TopoDS_Shape b = read_brep(brep_b);

  double distance = 0.0;
  gp_Pnt pa;
  gp_Pnt pb;
  {
    py::gil_scoped_release release;
    try {
      // The 2-shape constructor performs the computation (default deflection =
      // Precision::Confusion()); F/A are obsolete per the OCCT 8.0 header.
      BRepExtrema_DistShapeShape ext(a, b);
      if (!ext.IsDone() || ext.NbSolution() < 1) {
        py::gil_scoped_acquire acquire;
        throw PysmeshError(
            "shape_distance: BRepExtrema_DistShapeShape failed to find a solution "
            "(check that both shapes are non-empty and valid).");
      }
      distance = ext.Value();
      pa = ext.PointOnShape1(1);
      pb = ext.PointOnShape2(1);
    } catch (const PysmeshError&) {
      throw;
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("shape_distance: BRepExtrema_DistShapeShape raised: ") +
                          e.what());
    }
  }

  py::dict out;
  out["distance"] = distance;
  out["point_a"] = point3(pa);
  out["point_b"] = point3(pb);
  return out;
}

py::array_t<std::int32_t> free_boundary_edges(const py::bytes& brep) {
  const TopoDS_Shape shape = read_brep(brep);

  // Edge id source of truth: edges-only 1-based ordinals, identical to Shape.edges().
  TopTools_IndexedMapOfShape edges;
  TopExp::MapShapes(shape, TopAbs_EDGE, edges);

  // Ancestor multiplicity: MapShapesAndAncestors (NON-unique) appends the parent face once
  // per occurrence, so a periodic seam edge (same face on both sides) counts 2 and is NOT
  // flagged, while a manifold shared edge counts 2 (two distinct faces) and a naked boundary
  // edge counts 1. This is exactly the free-boundary criterion.
  TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
  TopExp::MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);

  std::vector<std::int32_t> free_ids;
  const int n = edges.Extent();
  for (int i = 1; i <= n; ++i) {
    const TopoDS_Edge& e = TopoDS::Edge(edges.FindKey(i));
    // Degenerate edges (sphere poles, cone apices) collapse to a point and carry no real
    // boundary — never a leak.
    if (BRep_Tool::Degenerated(e)) {
      continue;
    }
    // Edges with no face parent (bare wires / wireframe input) are not surface boundaries.
    if (!edge_faces.Contains(e)) {
      continue;
    }
    if (edge_faces.FindFromKey(e).Extent() == 1) {
      free_ids.push_back(static_cast<std::int32_t>(i));
    }
  }

  py::array_t<std::int32_t> out(static_cast<py::ssize_t>(free_ids.size()));
  std::copy(free_ids.begin(), free_ids.end(), out.mutable_data());
  return out;
}

}  // namespace

void bind_distance(py::module_& m) {
  m.def("shape_distance", &shape_distance, py::arg("brep_a"), py::arg("brep_b"),
        "Exact minimum distance between two BREP shapes (OCCT BRepExtrema_DistShapeShape). "
        "Returns a dict: 'distance' (float) and the witness points 'point_a'/'point_b' "
        "((3,) float64), one on each shape. Low-level: the pysmesh.shape_distance wrapper "
        "builds a ShapeDistanceResult around it.");
  m.def("free_boundary_edges", &free_boundary_edges, py::arg("brep"),
        "1-based ids (matching Shape.edges()) of every edge bordered by exactly one face — "
        "the naked boundary edges of an open shell. Empty for a watertight solid; a "
        "non-empty result localises a leak. Degenerate edges and edges with no face parent "
        "are excluded.");
}

}  // namespace pysmesh
