// pySMESH binding — shared infrastructure.
//
// Defines the typed exception carried across the C++/Python boundary, the refcounted
// shape container that solves the SMESHDS shape-index hazard (docs/upstream_notes/
// SMESHDS_Mesh_notes.md §CRITICAL), and small NumPy <-> OCCT helpers shared by shape.cpp
// and mesh.cpp.

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>

namespace pysmesh {

namespace py = pybind11;

// ---- Typed failure ---------------------------------------------------------------- //
// Every library failure surfaces as pysmesh._core.PysmeshError (a RuntimeError
// subclass) carrying .details (SMESH/OCCT message text) and .face_ids (offending faces,
// where applicable). The Python exception type is created and the translator registered
// in module.cpp via register_error_type().
class PysmeshError : public std::runtime_error {
 public:
  std::string details;
  std::vector<int> face_ids;

  explicit PysmeshError(const std::string& message, std::string details_ = std::string(),
                         std::vector<int> face_ids_ = std::vector<int>())
      : std::runtime_error(message),
        details(std::move(details_)),
        face_ids(std::move(face_ids_)) {}
};

void register_error_type(py::module_& m);

// ---- Shared shape container ------------------------------------------------------- //
// Holds the loaded TopoDS_Shape plus per-kind, 1-based indexed maps of its unique
// sub-shapes. The maps are the authoritative source of every Python-facing
// face_id/edge_id/vertex_id: they are built by TopExp::MapShapes with an explicit type
// filter (faces-only / edges-only / vertices-only), so the ids are stable and have a
// fixed meaning.
//
// This is the fix for the shape-index hazard: SMESHDS_Mesh's own internal index map
// (built by an unfiltered TopExp::MapShapes over ALL sub-shape kinds) does NOT match a
// faces-only ordinal, so a raw face_id must never be passed to SMESHDS's int-Index
// overloads. Both Shape and Mesh share one ShapeData (via shared_ptr); Mesh resolves
// face_id -> TopoDS_Face& through it and calls only the shape-reference overloads.
struct ShapeData {
  TopoDS_Shape shape;
  TopTools_IndexedMapOfShape faces;
  TopTools_IndexedMapOfShape edges;
  TopTools_IndexedMapOfShape vertices;

  explicit ShapeData(const TopoDS_Shape& s) : shape(s) {
    TopExp::MapShapes(shape, TopAbs_FACE, faces);
    TopExp::MapShapes(shape, TopAbs_EDGE, edges);
    TopExp::MapShapes(shape, TopAbs_VERTEX, vertices);
  }

  // 1-based id -> TopoDS_* resolution. Raise PysmeshError naming the bad id on any
  // out-of-range access — never let an invalid id reach OCCT/SMESHDS.
  const TopoDS_Face& face(int face_id) const {
    if (face_id < 1 || face_id > faces.Extent()) {
      throw PysmeshError("Invalid face_id " + std::to_string(face_id) + " (shape has " +
                          std::to_string(faces.Extent()) + " faces)");
    }
    return TopoDS::Face(faces.FindKey(face_id));
  }
  const TopoDS_Edge& edge(int edge_id) const {
    if (edge_id < 1 || edge_id > edges.Extent()) {
      throw PysmeshError("Invalid edge_id " + std::to_string(edge_id) + " (shape has " +
                          std::to_string(edges.Extent()) + " edges)");
    }
    return TopoDS::Edge(edges.FindKey(edge_id));
  }
  const TopoDS_Vertex& vertex(int vertex_id) const {
    if (vertex_id < 1 || vertex_id > vertices.Extent()) {
      throw PysmeshError("Invalid vertex_id " + std::to_string(vertex_id) +
                          " (shape has " + std::to_string(vertices.Extent()) +
                          " vertices)");
    }
    return TopoDS::Vertex(vertices.FindKey(vertex_id));
  }
};

// ---- NumPy helpers ---------------------------------------------------------------- //
// A validated, C-contiguous float64 (N, ncols) view. Raises PysmeshError on wrong ndim
// or column count (naming both), so callers get a clear message instead of an OCCT crash.
using Array2d = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Array1i = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using Array2i = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

inline Array2d as_2d_f64(const py::object& obj, const char* name, int ncols) {
  Array2d arr = obj.cast<Array2d>();
  if (arr.ndim() != 2 || arr.shape(1) != ncols) {
    throw PysmeshError(std::string(name) + " must have shape (N, " +
                        std::to_string(ncols) + ")");
  }
  return arr;
}

}  // namespace pysmesh

// ---- Cross-file Mesh internals seam ----------------------------------------------- //
// viscous.cpp needs the SMESH_Mesh / SMESH_Gen / ShapeData held by a Python Mesh object.
// These accessors are defined in mesh.cpp (where the Mesh class is visible) and mirror
// shape.cpp's shape_data_of. Forward-declare the SMESH types at global scope to avoid
// pulling their heavy headers into every translation unit that includes common.hpp.
class SMESH_Mesh;
class SMESH_Gen;
class SMESH_Hypothesis;

namespace pysmesh {

SMESH_Mesh& mesh_smesh(const py::object& mesh_obj);
SMESH_Gen& mesh_gen(const py::object& mesh_obj);
std::shared_ptr<ShapeData> mesh_shape_data(const py::object& mesh_obj);

// Hypothesis ownership: viscous.cpp creates the throwaway VL algo/hyp on the heap and hands
// them to the Mesh, which frees them at release() AFTER its SMESH_Gen is gone (SMESH's own
// contract — ~SMESH_Gen only NullifyGen()s hyps, never deletes them). next id is 1-based and
// unique within the Mesh's gen.
int mesh_next_hyp_id(const py::object& mesh_obj);
void mesh_adopt_hypothesis(const py::object& mesh_obj, SMESH_Hypothesis* hyp);

}  // namespace pysmesh
