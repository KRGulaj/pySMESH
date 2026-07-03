// pySMESH binding — geometry: load_brep, Shape, FaceInfo/EdgeInfo/VertexInfo,
// face_distance. See src/bindings/common.hpp for ShapeData (the id -> TopoDS_* source of
// truth) and PysmeshError.

#include <array>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepGProp.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <Bnd_Box.hxx>
#include <GProp_GProps.hxx>
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

// ---- Per-entity info structs (returned by Shape.faces()/.edges()/.vertices()) ------ //
struct FaceInfo {
  int id;
  double area;
  std::array<double, 3> centroid;
  std::array<double, 6> bbox;       // xmin, ymin, zmin, xmax, ymax, zmax
  std::array<double, 4> uv_bounds;  // umin, umax, vmin, vmax
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
      out.push_back(FaceInfo{i, props.Mass(), {c.X(), c.Y(), c.Z()}, bbox_of(f), uv});
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
      .def("__repr__", [](const FaceInfo& f) {
        return "<FaceInfo id=" + std::to_string(f.id) +
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
           "Exact minimum distance (N,) from each of N points (N,3) to the face.");

  m.def("load_brep", &load_brep, py::arg("data"),
        "Read a BREP shape from in-memory bytes. Raises on parse failure or null shape.");
}

}  // namespace pysmesh
