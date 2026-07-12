// pySMESH binding — geometry: load_brep, Shape, FaceInfo/EdgeInfo/VertexInfo,
// face_distance. See src/bindings/common.hpp for ShapeData (the id -> TopoDS_* source of
// truth) and PysmeshError.

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepGProp.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GProp_GProps.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <gp_Pnt.hxx>

#include "common.hpp"

namespace pysmesh {
namespace {

// Build a 1-D float64 NumPy array (owns a copy of the data).
py::array_t<double> vec1d(const double* data, std::size_t n) {
  py::array_t<double> out(static_cast<py::ssize_t>(n));
  std::copy(data, data + n, out.mutable_data());
  return out;
}

// Canonical name of a face's underlying geometry (BRepAdaptor_Surface::GetType). Feeds
// flux's feature-recognition defeature ("remove all cylindrical holes < 3 mm" needs the
// type, not just sqrt(area)). The set mirrors GeomAbs_SurfaceType exactly.
const char* surface_type_name(GeomAbs_SurfaceType t) {
  switch (t) {
    case GeomAbs_Plane:
      return "Plane";
    case GeomAbs_Cylinder:
      return "Cylinder";
    case GeomAbs_Cone:
      return "Cone";
    case GeomAbs_Sphere:
      return "Sphere";
    case GeomAbs_Torus:
      return "Torus";
    case GeomAbs_BezierSurface:
      return "Bezier";
    case GeomAbs_BSplineSurface:
      return "BSpline";
    case GeomAbs_SurfaceOfRevolution:
      return "Revolution";
    case GeomAbs_SurfaceOfExtrusion:
      return "Extrusion";
    case GeomAbs_OffsetSurface:
      return "Offset";
    case GeomAbs_OtherSurface:
      return "Other";
  }
  return "Other";
}

// ---- Per-entity info structs (returned by Shape.faces()/.edges()/.vertices()) ------ //
struct FaceInfo {
  int id;
  double area;
  std::array<double, 3> centroid;
  std::array<double, 6> bbox;       // xmin, ymin, zmin, xmax, ymax, zmax
  std::array<double, 4> uv_bounds;  // umin, umax, vmin, vmax
  std::string surface_type;         // Plane/Cylinder/Cone/Sphere/Torus/BSpline/...
};
struct EdgeInfo {
  int id;
  double length;
  std::array<double, 6> bbox;
  std::array<double, 2> t_bounds;  // first, last curve parameter
};
struct VertexInfo {
  int id;
  std::array<double, 3> xyz;
};

std::array<double, 6> bbox_of(const TopoDS_Shape& s) {
  Bnd_Box box;
  BRepBndLib::Add(s, box);
  std::array<double, 6> b{};
  box.Get(b[0], b[1], b[2], b[3], b[4], b[5]);
  return b;
}

// ---- Shape -------------------------------------------------------------------------//
class Shape {
 public:
  explicit Shape(std::shared_ptr<ShapeData> data) : data_(std::move(data)) {}

  std::shared_ptr<ShapeData> data() const { return data_; }

  std::vector<FaceInfo> faces() const {
    std::vector<FaceInfo> out;
    const int n = data_->faces.Extent();
    out.reserve(n);
    for (int i = 1; i <= n; ++i) {
      const TopoDS_Face& f = TopoDS::Face(data_->faces.FindKey(i));
      GProp_GProps props;
      BRepGProp::SurfaceProperties(f, props);
      const gp_Pnt c = props.CentreOfMass();
      std::array<double, 4> uv{};
      BRepTools::UVBounds(f, uv[0], uv[1], uv[2], uv[3]);
      const BRepAdaptor_Surface surf(f);
      out.push_back(FaceInfo{i, props.Mass(), {c.X(), c.Y(), c.Z()}, bbox_of(f), uv,
                             surface_type_name(surf.GetType())});
    }
    return out;
  }

  std::vector<EdgeInfo> edges() const {
    std::vector<EdgeInfo> out;
    const int n = data_->edges.Extent();
    out.reserve(n);
    for (int i = 1; i <= n; ++i) {
      const TopoDS_Edge& e = TopoDS::Edge(data_->edges.FindKey(i));
      GProp_GProps props;
      BRepGProp::LinearProperties(e, props);
      double first = 0.0, last = 0.0;
      BRep_Tool::Range(e, first, last);
      out.push_back(EdgeInfo{i, props.Mass(), bbox_of(e), {first, last}});
    }
    return out;
  }

  std::vector<VertexInfo> vertices() const {
    std::vector<VertexInfo> out;
    const int n = data_->vertices.Extent();
    out.reserve(n);
    for (int i = 1; i <= n; ++i) {
      const TopoDS_Vertex& v = TopoDS::Vertex(data_->vertices.FindKey(i));
      const gp_Pnt p = BRep_Tool::Pnt(v);
      out.push_back(VertexInfo{i, {p.X(), p.Y(), p.Z()}});
    }
    return out;
  }

  // Exact minimum distance from each of N points to the given face (BRepExtrema).
  // Exists for flux's gmsh-tag <-> OCCT-face tie-break (Phase 2 §5). GIL released.
  py::array_t<double> face_distance(int face_id, const py::object& points_obj) const {
    const TopoDS_Face face = data_->face(face_id);  // validates face_id
    Array2d points = as_2d_f64(points_obj, "points", 3);
    const py::ssize_t n = points.shape(0);
    py::array_t<double> out(n);

    const double* pts = points.data();
    double* dist = out.mutable_data();
    {
      py::gil_scoped_release release;
      for (py::ssize_t i = 0; i < n; ++i) {
        const gp_Pnt p(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
        const TopoDS_Shape vtx = BRepBuilderAPI_MakeVertex(p).Vertex();
        BRepExtrema_DistShapeShape ext(face, vtx);
        if (!ext.IsDone()) {
          py::gil_scoped_acquire acquire;
          throw PysmeshError("BRepExtrema distance computation failed for point " +
                              std::to_string(i) + " against face_id " +
                              std::to_string(face_id));
        }
        dist[i] = ext.Value();
      }
    }
    return out;
  }

  // Face pairs sharing an edge: (face_i, face_j, edge_id) with face_i < face_j, one row per
  // shared edge (all ids 1-based, matching faces()/edges()). Built from the edge->face
  // ancestor map so flux can walk fillet/tangent chains and remap markers in one native call
  // instead of N Gmsh round-trips (report §7.1/§8.2). Degenerate edges (poles/apices) and seam
  // edges (same face both sides -> no distinct pair) contribute nothing; a non-manifold edge
  // (>2 faces) emits every unique face pair.
  std::vector<std::tuple<int, int, int>> face_adjacency() const {
    TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
    TopExp::MapShapesAndAncestors(data_->shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);

    std::vector<std::tuple<int, int, int>> out;
    const int ne = data_->edges.Extent();
    for (int ei = 1; ei <= ne; ++ei) {
      const TopoDS_Edge& e = TopoDS::Edge(data_->edges.FindKey(ei));
      if (BRep_Tool::Degenerated(e) || !edge_faces.Contains(e)) {
        continue;
      }
      // Distinct 1-based face ids around this edge (a seam lists the same face twice).
      std::vector<int> fids;
      for (const TopoDS_Shape& fs : edge_faces.FindFromKey(e)) {
        const int fid = data_->faces.FindIndex(fs);
        if (fid >= 1 && std::find(fids.begin(), fids.end(), fid) == fids.end()) {
          fids.push_back(fid);
        }
      }
      std::sort(fids.begin(), fids.end());
      for (std::size_t a = 0; a < fids.size(); ++a) {
        for (std::size_t b = a + 1; b < fids.size(); ++b) {
          out.emplace_back(fids[a], fids[b], ei);
        }
      }
    }
    return out;
  }

  // Nearest face (by centroid) for each of Q query points (Q,3): 1-based face id, or -1 where
  // the nearest face centroid is farther than tol. Collapses flux's O(F*Q) ordinal<->tag
  // matching loop into one native call and is the honest home for the centroid data faces()
  // already exposes (report §4.5 persistent-naming fallback, §8.2 marker remap). GIL released
  // for the numeric sweep.
  py::array_t<std::int32_t> match_faces(const py::object& centroids_obj, double tol) const {
    if (!(tol > 0.0)) {
      throw PysmeshError("match_faces: tol must be > 0 (got " + std::to_string(tol) + ").");
    }
    Array2d centroids = as_2d_f64(centroids_obj, "centroids", 3);
    const py::ssize_t q = centroids.shape(0);

    // Precompute face centroids (OCCT calls stay under the GIL, ahead of the numeric loop).
    const int nf = data_->faces.Extent();
    std::vector<double> fc(static_cast<std::size_t>(nf) * 3);
    for (int i = 1; i <= nf; ++i) {
      const TopoDS_Face& f = TopoDS::Face(data_->faces.FindKey(i));
      GProp_GProps props;
      BRepGProp::SurfaceProperties(f, props);
      const gp_Pnt c = props.CentreOfMass();
      fc[3 * (i - 1) + 0] = c.X();
      fc[3 * (i - 1) + 1] = c.Y();
      fc[3 * (i - 1) + 2] = c.Z();
    }

    py::array_t<std::int32_t> out(q);
    const double* qp = centroids.data();
    std::int32_t* ids = out.mutable_data();
    const double tol2 = tol * tol;
    {
      py::gil_scoped_release release;
      for (py::ssize_t k = 0; k < q; ++k) {
        const double x = qp[3 * k];
        const double y = qp[3 * k + 1];
        const double z = qp[3 * k + 2];
        double best = std::numeric_limits<double>::infinity();
        int best_id = -1;
        for (int i = 0; i < nf; ++i) {
          const double dx = x - fc[3 * i];
          const double dy = y - fc[3 * i + 1];
          const double dz = z - fc[3 * i + 2];
          const double d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < best) {
            best = d2;
            best_id = i + 1;
          }
        }
        ids[k] = (best <= tol2) ? static_cast<std::int32_t>(best_id) : -1;
      }
    }
    return out;
  }

 private:
  std::shared_ptr<ShapeData> data_;
};

// ---- load_brep ---------------------------------------------------------------------//
Shape load_brep(const py::bytes& data) {
  const std::string buffer = data;  // copy the bytes into a std::string
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
  return Shape(std::make_shared<ShapeData>(shape));
}

}  // namespace

// Exposed to mesh.cpp so Mesh can share the same ShapeData as its Shape argument.
std::shared_ptr<ShapeData> shape_data_of(const py::object& shape_obj) {
  return shape_obj.cast<Shape&>().data();
}

void bind_shape(py::module_& m) {
  py::class_<FaceInfo>(m, "FaceInfo")
      .def_readonly("id", &FaceInfo::id)
      .def_readonly("area", &FaceInfo::area)
      .def_property_readonly(
          "centroid", [](const FaceInfo& f) { return vec1d(f.centroid.data(), 3); })
      .def_property_readonly("bbox",
                             [](const FaceInfo& f) { return vec1d(f.bbox.data(), 6); })
      .def_property_readonly(
          "uv_bounds", [](const FaceInfo& f) { return vec1d(f.uv_bounds.data(), 4); })
      .def_readonly("surface_type", &FaceInfo::surface_type)
      .def("__repr__", [](const FaceInfo& f) {
        return "<FaceInfo id=" + std::to_string(f.id) + " " + f.surface_type +
               " area=" + std::to_string(f.area) + ">";
      });

  py::class_<EdgeInfo>(m, "EdgeInfo")
      .def_readonly("id", &EdgeInfo::id)
      .def_readonly("length", &EdgeInfo::length)
      .def_property_readonly("bbox",
                             [](const EdgeInfo& e) { return vec1d(e.bbox.data(), 6); })
      .def_property_readonly(
          "t_bounds", [](const EdgeInfo& e) { return vec1d(e.t_bounds.data(), 2); })
      .def("__repr__", [](const EdgeInfo& e) {
        return "<EdgeInfo id=" + std::to_string(e.id) +
               " length=" + std::to_string(e.length) + ">";
      });

  py::class_<VertexInfo>(m, "VertexInfo")
      .def_readonly("id", &VertexInfo::id)
      .def_property_readonly("xyz",
                             [](const VertexInfo& v) { return vec1d(v.xyz.data(), 3); })
      .def("__repr__", [](const VertexInfo& v) {
        return "<VertexInfo id=" + std::to_string(v.id) + ">";
      });

  py::class_<Shape>(m, "Shape")
      .def("faces", &Shape::faces,
           "List every unique face with id (1-based), area, centroid, bbox, uv_bounds.")
      .def("edges", &Shape::edges,
           "List every unique edge with id, length, bbox, curve-parameter bounds.")
      .def("vertices", &Shape::vertices, "List every unique vertex with id and xyz.")
      .def("face_distance", &Shape::face_distance, py::arg("face_id"), py::arg("points"),
           "Exact minimum distance (N,) from each of N points (N,3) to the face.")
      .def("face_adjacency", &Shape::face_adjacency,
           "Face pairs sharing an edge: list of (face_i, face_j, edge_id) with face_i<face_j, "
           "one per shared edge (1-based ids). Degenerate and seam edges contribute none; a "
           "non-manifold edge (>2 faces) emits every unique pair.")
      .def("match_faces", &Shape::match_faces, py::arg("centroids"), py::arg("tol"),
           "Nearest face by centroid for each of Q query points (Q,3): (Q,) int32 1-based face "
           "ids, -1 where the nearest face centroid is farther than tol. tol must be > 0.");

  m.def("load_brep", &load_brep, py::arg("data"),
        "Read a BREP shape from in-memory bytes. Raises on parse failure or null shape.");
}

}  // namespace pysmesh
