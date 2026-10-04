// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-04

// pySMESH binding — viscous layers by the two-step builder (report L4).
//
// StdMeshers_ViscousLayerBuilder (SMESH d3c3260cd, "Decompose Viscous Layer API") is not an
// algorithm one assigns and computes. GetShrinkGeometry offsets the shape inward by the layer
// thickness on the boundary faces; the caller meshes that shrunk shape with any mesher; then
// AddLayers clears this mesher's mesh, copies the shrunk mesh into it and fills the gap with
// prismatic layers. So one native builder lives in the mesher on the original shape from the
// shrink to the layers, and four facts of the upstream code decide the checks here:
//
//   * The builder resolves its boundary faces through this mesh's ShapeToIndex
//     (StdMeshers_Cartesian_VL.cxx:825-874): the ordinals are this mesher's, translated
//     before they reach SMESH.
//   * AddLayers correlates the inner mesh's shape with the offset shape by TopExp::MapShapes
//     index (:1130-1134, 1162-1163). So the inner mesher must be on the very shape the shrink
//     returned, or on a copy with the same sub-shapes in the same order; anything else is
//     refused before SMESH reads it.
//   * Only the face-offset method exists (GetMethod is never read), so there is no method.
//   * Nothing in MakeViscousLayers reads a generator: the inner mesher keeps its own.
//
// A SALOME_Exception from either step reaches the caller as PysmeshError.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

#include <string>
#include <vector>

#include <BRep_Tool.hxx>
#include <Precision.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_Mesh.hxx>
#include <StdMeshers_ViscousLayerBuilder.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>

namespace pysmesh {

// A Python Shape on an existing OCCT shape, without a BREP round trip. Defined in shape.cpp.
py::object shape_object_of(const TopoDS_Shape& shape);

namespace mesher {
namespace {

// Whether `a` and `b` list the same sub-shapes in the same TopExp order: the same shape, or
// a copy of it such as a BREP round trip gives. Counts per kind, then every vertex point.
bool same_map_order(const TopoDS_Shape& a, const TopoDS_Shape& b) {
  if (a.IsSame(b)) {
    return true;
  }
  const TopAbs_ShapeEnum kinds[] = {TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE,
                                    TopAbs_EDGE, TopAbs_VERTEX};
  for (const TopAbs_ShapeEnum kind : kinds) {
    TopTools_IndexedMapOfShape ma, mb;
    TopExp::MapShapes(a, kind, ma);
    TopExp::MapShapes(b, kind, mb);
    if (ma.Extent() != mb.Extent()) {
      return false;
    }
    if (kind == TopAbs_VERTEX) {
      for (int i = 1; i <= ma.Extent(); ++i) {
        const gp_Pnt pa = BRep_Tool::Pnt(TopoDS::Vertex(ma.FindKey(i)));
        const gp_Pnt pb = BRep_Tool::Pnt(TopoDS::Vertex(mb.FindKey(i)));
        if (pa.Distance(pb) > Precision::Confusion()) {
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace

bool Mesher::LayerRequest::operator==(const LayerRequest& o) const {
  return total_thickness == o.total_thickness && layer_count == o.layer_count &&
         stretch_factor == o.stretch_factor && boundary == o.boundary &&
         ignore == o.ignore && group_name == o.group_name;
}

Mesher::LayerRequest Mesher::layer_request(const py::dict& values, const char* op) const {
  Params p(op, values);
  LayerRequest r;
  r.total_thickness = p.number("total_thickness");
  r.layer_count = p.integer("layer_count");
  r.stretch_factor = p.number("stretch_factor");
  r.boundary = p.integers("boundary");
  r.ignore = p.flag("ignore");
  r.group_name = p.text("group_name");
  p.done();
  return r;
}

// The shape the builder works on. The root of a shape read from BREP is a compound, and
// MakeOffsetShape treats a compound by gluing the offset solids with BOPAlgo_Builder
// (StdMeshers_Cartesian_VL.cxx:965-975), which returns a null shape for one solid (measured
// on a box). So one solid goes in as itself, the direct path (:977), and several as the
// root. A shape with no solid and one face goes in as that face: MakeOffsetShape takes the
// 2-D path only for a TopAbs_FACE (:991-994).
const TopoDS_Shape& Mesher::layer_shape(const char* op) const {
  if (data_->solids.Extent() == 1) {
    return data_->solid(1);
  }
  if (data_->solids.Extent() > 1) {
    return data_->shape;
  }
  if (data_->faces.Extent() == 1) {
    return data_->face(1);
  }
  throw PysmeshError(std::string(op) + ": the shape has no solid and " +
                     std::to_string(data_->faces.Extent()) +
                     " faces; the builder grows layers in solids, or in one face.");
}

py::object Mesher::shrink_geometry(const py::dict& values) {
  ensure_open();
  ensure_shape("Mesher.shrink_geometry");
  const LayerRequest request = layer_request(values, "Mesher.shrink_geometry");
  const TopoDS_Shape& target = layer_shape("Mesher.shrink_geometry");
  const bool in_face = target.ShapeType() == TopAbs_FACE;

  // The 2-D path offsets the whole wire of the face (BRepOffsetAPI_MakeOffset, :1000-1003),
  // so layers have to grow on every edge of it: a partial edge list would leave the shrunk
  // face short of the boundary where no layer fills the gap.
  if (in_face && !(request.ignore && request.boundary.empty())) {
    throw PysmeshError("Mesher.shrink_geometry: in a face the builder shrinks the whole "
                       "face, so the layers grow on every edge; leave boundary empty with "
                       "ignore=True.");
  }
  std::vector<int> indices;
  for (const int ordinal : request.boundary) {
    indices.push_back(meshDS_->ShapeToIndex(sub_shape(in_face ? "EDGE" : "FACE", ordinal)));
  }

  if (vl_builder_ == nullptr) {
    auto* builder = new StdMeshers_ViscousLayerBuilder(gen_->GetANewId(), gen_.get());
    owned_.emplace_back(builder);
    vl_builder_ = builder;
  }
  vl_builder_->SetTotalThickness(request.total_thickness);
  vl_builder_->SetNumberLayers(request.layer_count);
  vl_builder_->SetStretchFactor(request.stretch_factor);
  vl_builder_->SetGroupName(request.group_name);
  vl_builder_->SetBndShapes(indices, request.ignore);

  TopoDS_Shape shrunk;
  std::string error;
  {
    py::gil_scoped_release release;
    try {
      shrunk = vl_builder_->GetShrinkGeometry(*mesh_, target);
    } catch (const std::exception& e) {
      error = e.what() != nullptr ? e.what() : "unknown error";
    }
  }
  if (!error.empty() || shrunk.IsNull()) {
    vl_shrunk_.Nullify();
    throw PysmeshError("Mesher.shrink_geometry: SMESH could not offset the shape.",
                       error.empty() ? std::string("The offset shape is null.") : error);
  }
  vl_request_ = request;
  vl_shrunk_ = shrunk;
  return shape_object_of(shrunk);
}

py::dict Mesher::add_layers(const py::dict& values, Mesher& inner) {
  ensure_open();
  ensure_shape("Mesher.add_layers");
  if (vl_builder_ == nullptr || vl_shrunk_.IsNull()) {
    throw PysmeshError("Mesher.add_layers: call shrink_geometry on this mesher first; there "
                       "is no shrunk shape to add layers to.");
  }
  const LayerRequest request = layer_request(values, "Mesher.add_layers");
  if (!(request == vl_request_)) {
    throw PysmeshError("Mesher.add_layers: the builder differs from the one shrink_geometry "
                       "was given; pass the same ViscousLayerBuilder to both.");
  }
  if (&inner == this) {
    throw PysmeshError("Mesher.add_layers: the inner mesh must come from another Mesher, "
                       "built on the shape shrink_geometry returned.");
  }
  inner.ensure_open();
  inner.ensure_shape("Mesher.add_layers (inner mesher)");
  if (!same_map_order(inner.data_->shape, vl_shrunk_)) {
    throw PysmeshError("Mesher.add_layers: the inner mesher is not on the shape "
                       "shrink_geometry returned.",
                       "SMESH pairs the inner mesh with the shrunk shape by TopExp index, so "
                       "the inner mesher must be built on that Shape, or on a copy with the "
                       "same sub-shapes in the same order.");
  }
  if (inner.meshDS_->NbNodes() == 0) {
    throw PysmeshError("Mesher.add_layers: the inner mesher holds no mesh; compute it first.");
  }

  const TopoDS_Shape& target = layer_shape("Mesher.add_layers");
  std::string error;
  {
    py::gil_scoped_release release;
    try {
      vl_builder_->AddLayers(*inner.mesh_, *mesh_, target);
    } catch (const std::exception& e) {
      error = e.what() != nullptr ? e.what() : "unknown error";
    }
  }
  meshDS_->Modified();
  if (!error.empty()) {
    throw PysmeshError("Mesher.add_layers: SMESH could not build the layers.", error);
  }
  return success_report(py::list());
}

}  // namespace mesher
}  // namespace pysmesh
