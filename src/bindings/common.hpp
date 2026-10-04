// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-07-03

// pySMESH binding — shared infrastructure.
//
// Defines the typed exception carried across the C++/Python boundary, the refcounted
// shape container that solves the SMESHDS shape-index hazard (docs/upstream_notes/
// SMESHDS_Mesh_notes.md §CRITICAL), and small NumPy <-> OCCT helpers shared by shape.cpp
// and mesh.cpp.

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
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
#include <TopoDS_Solid.hxx>
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

// An operation the caller stopped, surfaced as pysmesh._core.PysmeshCancelled — a SUBCLASS
// of PysmeshError, so every existing `except PysmeshError` still catches it and nothing that
// already handles failures has to change.
//
// It is a separate type because the two outcomes need different handling and telling them
// apart by message text would be fragile: a cancellation is the normal end of a long
// operation the user interrupted, and a failure is a defect to report. Both leave the
// session exactly as it was; neither returns a partial shape.
class CancelledError : public PysmeshError {
 public:
  explicit CancelledError(const std::string& message,
                          std::string details_ = std::string())
      : PysmeshError(message, std::move(details_)) {}
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
  TopTools_IndexedMapOfShape solids;
  TopTools_IndexedMapOfShape faces;
  TopTools_IndexedMapOfShape edges;
  TopTools_IndexedMapOfShape vertices;

  explicit ShapeData(const TopoDS_Shape& s) : shape(s) {
    TopExp::MapShapes(shape, TopAbs_SOLID, solids);
    TopExp::MapShapes(shape, TopAbs_FACE, faces);
    TopExp::MapShapes(shape, TopAbs_EDGE, edges);
    TopExp::MapShapes(shape, TopAbs_VERTEX, vertices);
  }

  // 1-based id -> TopoDS_* resolution. Raise PysmeshError naming the bad id on any
  // out-of-range access — never let an invalid id reach OCCT/SMESHDS.
  const TopoDS_Solid& solid(int solid_id) const {
    if (solid_id < 1 || solid_id > solids.Extent()) {
      throw PysmeshError("Invalid solid_id " + std::to_string(solid_id) + " (shape has " +
                          std::to_string(solids.Extent()) + " solids)");
    }
    return TopoDS::Solid(solids.FindKey(solid_id));
  }
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

// ---- Finite arguments ------------------------------------------------------------- //
// A NaN passes every `<=` and `<` test, and an infinity passed most of the range checks,
// so neither was caught before it reached OCCT or SMESH, which then crashed the process
// or built garbage that was committed (report F1). Every public float argument goes
// through one of these before any OCCT or SMESH call. `where` is the operation as the
// caller names it ("Session.extrude", "tessellate"), `name` the argument.

// A number for a message: "nan", "inf" or "-inf" for the non-finite values.
// TopTools_ShapeSet::Read writes "File was not written with this version of the topology"
// to std::cout and returns a null shape when BREP data holds no version line
// (TopTools_ShapeSet.cxx:698). Every BREP reader checks for that line first and raises the
// message it gives a null shape, so no line reaches stdout before the error (report A4).
inline void require_brep_header(const std::string& data, const std::string& null_message) {
  if (data.find("CASCADE Topology V") == std::string::npos) {
    throw PysmeshError(null_message);
  }
}

inline std::string number_text(double v) {
  if (std::isnan(v)) {
    return "nan";
  }
  if (std::isinf(v)) {
    return v > 0.0 ? "inf" : "-inf";
  }
  std::ostringstream s;
  s << v;
  return s.str();
}

inline void require_finite(const std::string& where, const char* name, double v) {
  if (!std::isfinite(v)) {
    throw PysmeshError(where + ": " + name + " must be a finite number (got " +
                       number_text(v) + ").");
  }
}

inline void require_finite(const std::string& where, const char* name, double x, double y,
                           double z) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
    throw PysmeshError(where + ": " + name + " must be finite (got (" + number_text(x) +
                       ", " + number_text(y) + ", " + number_text(z) + ")).");
  }
}

// Every value of an array argument of `rows` rows of `width` values; the first value that
// is not finite is named with its row.
inline void require_finite(const std::string& where, const char* name, const double* data,
                           std::size_t rows, std::size_t width) {
  for (std::size_t i = 0; i < rows * width; ++i) {
    if (!std::isfinite(data[i])) {
      throw PysmeshError(where + ": " + name + " must be finite (row " +
                         std::to_string(i / width) + " holds " + number_text(data[i]) +
                         ").");
    }
  }
}

// A float64 (N, ncols) array whose every value is finite, named `name` of `where` in a
// refusal.
inline Array2d as_2d_f64(const py::object& obj, const char* where, const char* name,
                         int ncols) {
  Array2d arr = obj.cast<Array2d>();
  if (arr.ndim() != 2 || arr.shape(1) != ncols) {
    throw PysmeshError(std::string(where) + ": " + name + " must have shape (N, " +
                       std::to_string(ncols) + ")");
  }
  require_finite(where, name, arr.data(), static_cast<std::size_t>(arr.shape(0)),
                 static_cast<std::size_t>(ncols));
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
