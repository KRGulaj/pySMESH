// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — block decomposition, and pattern mapping onto a face or a block.
//
// A **block** is a six-faced solid seen as a deformed unit cube: every point in it has a
// normalised (x, y, z) coordinate in [0, 1]^3, and the mapping between those and model space
// runs both ways. That is the machinery the structured hexahedral algorithms work through,
// and on its own it is what a caller needs to lay a field, a seed set or an O-grid inside a
// solid without meshing it first.
//
// The block's own sub-shape numbering is fixed by upstream and is what makes it addressable:
// eight vertices, twelve edges and six faces in a stated order, decided by which two vertices
// the caller nominates as the (0,0,0) and (0,0,1) corners. `block_shapes` reports that
// numbering in the caller's own ordinals, so "the face at z = 1" is answerable.
//
// A **pattern** is a small parametric mesh — points in the unit square or cube plus their
// connectivity — mapped onto a face or a block by matching its key points to the corners.
// It is how a repeating motif is laid onto geometry that an algorithm would otherwise mesh
// generically.
//
// One property of the upstream API decides the shape of both halves: the block is a *live*
// object that has to be loaded from geometry before it can answer anything, and loading it is
// most of the cost. Every call here therefore takes a batch of parameters or points, so the
// load is paid once for the whole question rather than once per point.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <SMESHDS_Mesh.hxx>
#include <SMESH_Block.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_Pattern.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfOrientedShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Vertex.hxx>
#include <gp_Pnt.hxx>
#include <gp_XYZ.hxx>

namespace pysmesh {

std::shared_ptr<ShapeData> shape_data_of(const py::object& shape_obj);

namespace mesher {
namespace {

// The outer shell of a solid. A block is defined on a shell, and a solid with more than one
// is not a block at all — an inner void makes the six-face correspondence meaningless.
TopoDS_Shell block_shell(const ShapeData& data, int solid_ordinal, const char* owner) {
  const TopoDS_Shape& solid = data.solid(solid_ordinal);
  TopoDS_Shell shell;
  int found = 0;
  for (TopExp_Explorer ex(solid, TopAbs_SHELL); ex.More(); ex.Next()) {
    if (found == 0) {
      shell = TopoDS::Shell(ex.Current());
    }
    ++found;
  }
  if (found == 0) {
    throw PysmeshError(std::string(owner) + ": solid " + std::to_string(solid_ordinal) +
                       " has no shell.");
  }
  if (found > 1) {
    throw PysmeshError(std::string(owner) + ": solid " + std::to_string(solid_ordinal) +
                       " has " + std::to_string(found) +
                       " shells. A block is one closed shell of six faces.");
  }
  return shell;
}

// Load the block, or say which part of the requirement the solid failed. Upstream reports a
// refusal as a plain false, so the message has to name what a block is.
void load_block(SMESH_Block& block, const ShapeData& data, int solid_ordinal, int vertex000,
                int vertex001, TopTools_IndexedMapOfOrientedShape& map, const char* owner) {
  const TopoDS_Shell shell = block_shell(data, solid_ordinal, owner);
  const TopoDS_Vertex& v000 = data.vertex(vertex000);
  const TopoDS_Vertex& v001 = data.vertex(vertex001);
  if (!block.LoadBlockShapes(shell, v000, v001, map)) {
    throw PysmeshError(
        std::string(owner) + ": solid " + std::to_string(solid_ordinal) +
        " is not a block. It must be one shell of six four-sided faces, and vertex " +
        std::to_string(vertex000) + " and vertex " + std::to_string(vertex001) +
        " must be two corners of it joined by one edge.");
  }
}

}  // namespace

// ---- The block's own sub-shape numbering ------------------------------------------------ //

py::dict block_shapes(const py::object& shape_obj, int solid_ordinal, int vertex000,
                      int vertex001) {
  const char* owner = "block_shapes";
  const std::shared_ptr<ShapeData> data = shape_data_of(shape_obj);

  SMESH_Block block;
  TopTools_IndexedMapOfOrientedShape map;
  load_block(block, *data, solid_ordinal, vertex000, vertex001, map, owner);

  // Ids 1-8 are the vertices, 9-20 the edges and 21-26 the faces, in upstream's own order.
  // Each is reported as the caller's per-kind ordinal, never as an index private to the
  // block, so the answer joins straight back to the geometry the caller queried.
  std::vector<std::int64_t> vertices;
  std::vector<std::int64_t> edges;
  std::vector<std::int64_t> faces;
  for (int id = SMESH_Block::ID_V000; id <= SMESH_Block::ID_F1yz; ++id) {
    if (id > map.Extent()) {
      throw PysmeshError(std::string(owner) + ": the block is missing sub-shape " +
                         std::to_string(id) + ".");
    }
    const TopoDS_Shape& sub = map.FindKey(id);
    if (SMESH_Block::IsVertexID(id)) {
      vertices.push_back(data->vertices.FindIndex(sub));
    } else if (SMESH_Block::IsEdgeID(id)) {
      edges.push_back(data->edges.FindIndex(sub));
    } else {
      faces.push_back(data->faces.FindIndex(sub));
    }
  }

  py::dict out;
  out["solid"] = solid_ordinal;
  out["vertices"] = vector_to_array(vertices);
  out["edges"] = vector_to_array(edges);
  out["faces"] = vector_to_array(faces);
  return out;
}

// ---- Parameters to points, and back ----------------------------------------------------- //

py::dict block_points(const py::object& shape_obj, int solid_ordinal, int vertex000,
                      int vertex001, const py::object& parameters) {
  const char* owner = "block_points";
  const std::shared_ptr<ShapeData> data = shape_data_of(shape_obj);
  const auto table = point_table(parameters, "block_points: parameters", 3);
  const py::ssize_t rows = table.shape(0);
  const double* values = table.data();
  for (py::ssize_t i = 0; i < 3 * rows; ++i) {
    if (!(values[i] >= 0.0 && values[i] <= 1.0)) {
      throw PysmeshError(std::string(owner) +
                         ": every block parameter must be within [0, 1]; row " +
                         std::to_string(i / 3) + " is outside it.");
    }
  }

  SMESH_Block block;
  TopTools_IndexedMapOfOrientedShape map;
  load_block(block, *data, solid_ordinal, vertex000, vertex001, map, owner);

  std::vector<double> points;
  points.reserve(static_cast<std::size_t>(rows) * 3);
  for (py::ssize_t i = 0; i < rows; ++i) {
    gp_XYZ where;
    if (!block.ShellPoint(gp_XYZ(values[3 * i], values[3 * i + 1], values[3 * i + 2]),
                          where)) {
      throw PysmeshError(std::string(owner) + ": the block could not place row " +
                         std::to_string(i) + ".");
    }
    points.insert(points.end(), {where.X(), where.Y(), where.Z()});
  }

  py::dict out;
  out["points"] = rows_to_array(points, 3);
  return out;
}

py::dict block_parameters(const py::object& shape_obj, int solid_ordinal, int vertex000,
                          int vertex001, const py::object& points, double tolerance) {
  const char* owner = "block_parameters";
  const std::shared_ptr<ShapeData> data = shape_data_of(shape_obj);
  const auto table = point_table(points, "block_parameters: points", 3);
  const py::ssize_t rows = table.shape(0);
  const double* values = table.data();
  if (!(tolerance > 0.0)) {
    throw PysmeshError(std::string(owner) + ": the tolerance must be > 0 (got " +
                       std::to_string(tolerance) + ").");
  }

  SMESH_Block block;
  TopTools_IndexedMapOfOrientedShape map;
  load_block(block, *data, solid_ordinal, vertex000, vertex001, map, owner);
  block.SetTolerance(tolerance);

  std::vector<double> parameters;
  std::vector<double> distances;
  std::vector<std::int64_t> converged;
  parameters.reserve(static_cast<std::size_t>(rows) * 3);
  for (py::ssize_t i = 0; i < rows; ++i) {
    gp_XYZ found;
    // The inversion is a numerical search, so it reports how close it got as well as where.
    // A point outside the block has no parameters and the distance is what says so.
    const bool ran = block.ComputeParameters(
        gp_Pnt(values[3 * i], values[3 * i + 1], values[3 * i + 2]), found);
    parameters.insert(parameters.end(), {found.X(), found.Y(), found.Z()});
    distances.push_back(ran ? block.DistanceReached() : -1.0);
    converged.push_back(ran && block.IsToleranceReached() ? 1 : 0);
  }

  py::dict out;
  out["parameters"] = rows_to_array(parameters, 3);
  out["distances"] = vector_to_array(distances);
  out["converged"] = vector_to_array(converged);
  return out;
}

// ---- Pattern mapping -------------------------------------------------------------------- //

namespace {

// Upstream's own words for a refused pattern. The enum is the only channel there is.
std::string pattern_error_text(int code) {
  switch (code) {
    case SMESH_Pattern::ERR_OK:
      return "the pattern was applied";
    case SMESH_Pattern::ERR_READ_NB_POINTS:
    case SMESH_Pattern::ERR_READ_POINT_COORDS:
    case SMESH_Pattern::ERR_READ_TOO_FEW_POINTS:
    case SMESH_Pattern::ERR_READ_3D_COORD:
    case SMESH_Pattern::ERR_READ_NO_KEYPOINT:
    case SMESH_Pattern::ERR_READ_BAD_INDEX:
    case SMESH_Pattern::ERR_READ_ELEM_POINTS:
    case SMESH_Pattern::ERR_READ_NO_ELEMS:
    case SMESH_Pattern::ERR_READ_BAD_KEY_POINT:
      return "the pattern text is malformed";
    case SMESH_Pattern::ERR_LOAD_EMPTY_SUBMESH:
      return "the sub-shape carries no elements to make a pattern from";
    case SMESH_Pattern::ERR_LOADF_NARROW_FACE:
      return "the face is too narrow";
    case SMESH_Pattern::ERR_LOADF_CLOSED_FACE:
      return "the face is closed, and a pattern needs an open parameter domain";
    case SMESH_Pattern::ERR_LOADF_CANT_PROJECT:
      return "the nodes could not be projected onto the face";
    case SMESH_Pattern::ERR_LOADV_BAD_SHAPE:
    case SMESH_Pattern::ERR_APPLV_BAD_SHAPE:
      return "the solid is not a block of six faces";
    case SMESH_Pattern::ERR_LOADV_COMPUTE_PARAMS:
      return "the point parameters inside the block could not be computed";
    case SMESH_Pattern::ERR_APPL_NOT_COMPUTED:
    case SMESH_Pattern::ERR_MAKEM_NOT_COMPUTED:
      return "the mapping did not converge";
    case SMESH_Pattern::ERR_APPL_NOT_LOADED:
      return "no pattern was loaded";
    case SMESH_Pattern::ERR_APPL_BAD_DIMENTION:
      return "the pattern's dimension does not match the shape it was applied to";
    case SMESH_Pattern::ERR_APPL_BAD_NB_VERTICES:
      return "the pattern's key points and the shape's vertices do not correspond";
    case SMESH_Pattern::ERR_APPLF_BAD_TOPOLOGY:
      return "the pattern's own topology is not usable on a face";
    case SMESH_Pattern::ERR_APPLF_BAD_VERTEX:
      return "the vertex given is not on the face's outer boundary";
    case SMESH_Pattern::ERR_APPLF_BAD_FACE_GEOM:
      return "the face's geometry is not usable";
    default:
      return "the pattern operation failed inside SMESH";
  }
}

void require_pattern(const SMESH_Pattern& pattern, bool ok, const char* owner) {
  if (!ok) {
    throw PysmeshError(std::string(owner) + ": " +
                       pattern_error_text(static_cast<int>(pattern.GetErrorCode())) + ".");
  }
}

}  // namespace

std::string Mesher::pattern_from_face(int face_ordinal, bool project) {
  ensure_open();
  const char* owner = "Mesher.pattern_from_face";
  ensure_shape(owner);
  const TopoDS_Face& face = TopoDS::Face(sub_shape("FACE", face_ordinal));

  SMESH_Pattern pattern;
  require_pattern(pattern, pattern.Load(mesh_, face, project), owner);
  std::ostringstream text;
  require_pattern(pattern, pattern.Save(text), owner);
  return text.str();
}

py::dict Mesher::apply_pattern_to_face(const std::string& text, int face_ordinal,
                                       int vertex_ordinal, bool reverse,
                                       bool create_polygons) {
  ensure_open();
  const char* owner = "Mesher.apply_pattern_to_face";
  ensure_shape(owner);
  const TopoDS_Face& face = TopoDS::Face(sub_shape("FACE", face_ordinal));
  const TopoDS_Vertex& vertex = TopoDS::Vertex(sub_shape("VERTEX", vertex_ordinal));

  const std::int64_t faces_before = static_cast<std::int64_t>(meshDS_->NbFaces());
  const std::int64_t nodes_before = static_cast<std::int64_t>(meshDS_->NbNodes());

  SMESH_Pattern pattern;
  require_pattern(pattern, pattern.Load(text.c_str()), owner);
  require_pattern(pattern, pattern.Apply(face, vertex, reverse), owner);
  require_pattern(pattern, pattern.MakeMesh(mesh_, create_polygons, false), owner);
  meshDS_->Modified();

  py::dict out;
  out["nodes_before"] = nodes_before;
  out["nodes_after"] = static_cast<std::int64_t>(meshDS_->NbNodes());
  out["faces_before"] = faces_before;
  out["faces_after"] = static_cast<std::int64_t>(meshDS_->NbFaces());
  return out;
}

py::dict Mesher::apply_pattern_to_block(const std::string& text, int solid_ordinal,
                                        int vertex000, int vertex001,
                                        bool create_polyhedra) {
  ensure_open();
  const char* owner = "Mesher.apply_pattern_to_block";
  ensure_shape(owner);
  const TopoDS_Shell shell = block_shell(shape_data(), solid_ordinal, owner);
  const TopoDS_Vertex& v000 = TopoDS::Vertex(sub_shape("VERTEX", vertex000));
  const TopoDS_Vertex& v001 = TopoDS::Vertex(sub_shape("VERTEX", vertex001));

  const std::int64_t volumes_before = static_cast<std::int64_t>(meshDS_->NbVolumes());
  const std::int64_t nodes_before = static_cast<std::int64_t>(meshDS_->NbNodes());

  SMESH_Pattern pattern;
  require_pattern(pattern, pattern.Load(text.c_str()), owner);
  require_pattern(pattern, pattern.Apply(shell, v000, v001), owner);
  require_pattern(pattern, pattern.MakeMesh(mesh_, false, create_polyhedra), owner);
  meshDS_->Modified();

  py::dict out;
  out["nodes_before"] = nodes_before;
  out["nodes_after"] = static_cast<std::int64_t>(meshDS_->NbNodes());
  out["volumes_before"] = volumes_before;
  out["volumes_after"] = static_cast<std::int64_t>(meshDS_->NbVolumes());
  return out;
}

}  // namespace mesher
}  // namespace pysmesh
