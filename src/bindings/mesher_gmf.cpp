// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — Inria .mesh / .meshb, through SMESH's own GMF driver.
//
// The format is the native interchange of MMG and fTetWild, and the driver is already
// compiled into the wheel, so this is a binding rather than a port. What it is not is a
// lossless container, and three limits are measured rather than assumed:
//
//   * **The format has no polygon and no polyhedron.** The upstream writer iterates exactly
//     the edge, triangle, quadrangle, tetrahedron, pyramid, hexahedron and prism families,
//     so a body-fitted Cartesian mesh — which is hexahedra plus polyhedra at the cut cells —
//     would lose its cut cells with no error at all. Writing refuses such a mesh by name
//     rather than emitting a quietly incomplete file.
//   * **Quadratic pyramids and prisms are not written either.** The linear forms are; the
//     quadratic ones have no branch upstream. Same treatment.
//   * **The per-element sub-shape reference does not survive.** The writer emits it as each
//     element's GMF reference, and the reader parses it into a local and drops it. So a round
//     trip keeps the mesh and loses its CAD binding, and a mesh read from a file reports
//     every element as bound to nothing.
//
// Groups are carried on one channel only: GMF's "required entities", which upstream keys on
// a group name containing `_required_` followed by Vertices, Edges, Triangles or
// Quadrilaterals. A group named anything else is not written, so one supplied here is
// refused rather than dropped.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <DriverGMF_Read.hxx>
#include <DriverGMF_Write.hxx>
#include <Driver_Mesh.h>
#include <SMDSAbs_ElementType.hxx>
#include <SMESHDS_Group.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_Group.hxx>
#include <SMESH_Mesh.hxx>

namespace pysmesh {
namespace mesher {
namespace {

const char* status_text(Driver_Mesh::Status status) {
  switch (status) {
    case Driver_Mesh::DRS_OK:
      return "ok";
    case Driver_Mesh::DRS_EMPTY:
      return "the file contains no mesh";
    case Driver_Mesh::DRS_WARN_RENUMBER:
      return "the file has overlapping element number ranges, so its numbers were ignored";
    case Driver_Mesh::DRS_WARN_SKIP_ELEM:
      return "some elements were skipped because their data is incorrect";
    case Driver_Mesh::DRS_WARN_DESCENDING:
      return "some elements were skipped for descending connectivity";
    case Driver_Mesh::DRS_TOO_LARGE_MESH:
      return "the mesh is too large for this format";
    default:
      return "the driver failed";
  }
}

// The element families the GMF writer actually emits, checked against SMDS's entity type.
// Anything outside this set has no representation in the format at all.
bool is_writable(int type) {
  switch (static_cast<SMDSAbs_EntityType>(type)) {
    case SMDSEntity_Edge:
    case SMDSEntity_Quad_Edge:
    case SMDSEntity_Triangle:
    case SMDSEntity_Quad_Triangle:
    case SMDSEntity_BiQuad_Triangle:
    case SMDSEntity_Quadrangle:
    case SMDSEntity_Quad_Quadrangle:
    case SMDSEntity_BiQuad_Quadrangle:
    case SMDSEntity_Tetra:
    case SMDSEntity_Quad_Tetra:
    case SMDSEntity_Pyramid:
    case SMDSEntity_Hexa:
    case SMDSEntity_Quad_Hexa:
    case SMDSEntity_TriQuad_Hexa:
    case SMDSEntity_Penta:
      return true;
    default:
      return false;
  }
}

const char* type_name(int type) {
  switch (static_cast<SMDSAbs_EntityType>(type)) {
    case SMDSEntity_Polygon:
    case SMDSEntity_Quad_Polygon:
      return "a polygon";
    case SMDSEntity_Polyhedra:
    case SMDSEntity_Quad_Polyhedra:
      return "a polyhedron (the body-fitted Cartesian mesher emits these at its cut cells)";
    case SMDSEntity_Quad_Pyramid:
      return "a quadratic pyramid";
    case SMDSEntity_Quad_Penta:
    case SMDSEntity_BiQuad_Penta:
      return "a quadratic prism";
    case SMDSEntity_Hexagonal_Prism:
      return "a hexagonal prism";
    case SMDSEntity_Ball:
      return "a ball";
    case SMDSEntity_0D:
      return "a 0-D element";
    default:
      return "an element of a type";
  }
}

}  // namespace

void write_gmf(const std::string& path, const py::dict& mesh, const py::list& groups) {
  if (!mesh.contains("element_type") || !mesh.contains("element_id")) {
    throw PysmeshError("write_gmf: the mesh is missing 'element_type' or 'element_id'.");
  }
  const auto types =
      mesh["element_type"].cast<py::array_t<std::int8_t, py::array::c_style |
                                                             py::array::forcecast>>();
  const auto element_ids =
      mesh["element_id"].cast<py::array_t<std::int64_t, py::array::c_style |
                                                            py::array::forcecast>>();
  if (element_ids.shape(0) != types.shape(0)) {
    throw PysmeshError("write_gmf: element_type and element_id disagree on the element "
                       "count.");
  }

  // Refuse a mesh the format cannot hold, naming the first offending element and its type.
  // The alternative is a file that is silently missing its cut cells.
  for (py::ssize_t i = 0; i < types.shape(0); ++i) {
    if (!is_writable(types.data()[i])) {
      throw PysmeshError(
          "write_gmf: element " + std::to_string(element_ids.data()[i]) + " is " +
          type_name(types.data()[i]) +
          " that the Inria .mesh format cannot represent, so the file would be silently "
          "incomplete.");
    }
  }

  ScratchMesh scratch;
  SMESHDS_Mesh& ds = scratch.ds();
  try {
    rebuild_mesh(ds, mesh);
  } catch (const PysmeshError& failure) {
    throw PysmeshError(std::string("write_gmf: ") + failure.what(), failure.details);
  }

  for (const py::handle& item : groups) {
    const py::tuple entry = item.cast<py::tuple>();
    if (entry.size() != 3) {
      throw PysmeshError("write_gmf: each group must be a (name, element_type, ids) triple.");
    }
    const std::string name = entry[0].cast<std::string>();
    if (name.find("_required_") == std::string::npos) {
      throw PysmeshError(
          "write_gmf: the Inria .mesh format carries only 'required entity' groups, so the "
          "group '" +
          name +
          "' would not be written at all. Name it '_required_Vertices', '_required_Edges', "
          "'_required_Triangles' or '_required_Quadrilaterals', or leave it out.");
    }
    const auto element_type = static_cast<SMDSAbs_ElementType>(entry[1].cast<int>());
    SMESH_Group* group = scratch.mesh().AddGroup(element_type, name.c_str());
    SMESHDS_Group* group_ds = dynamic_cast<SMESHDS_Group*>(group->GetGroupDS());
    if (group_ds == nullptr) {
      throw PysmeshError("write_gmf: could not create the group '" + name + "'.");
    }
    group_ds->SetStoreName(name.c_str());
    for (const std::int64_t id : entry[2].cast<std::vector<std::int64_t>>()) {
      group_ds->Add(static_cast<smIdType>(id));
    }
  }

  DriverGMF_Write writer;
  writer.SetFile(path);
  writer.SetMesh(&ds);
  writer.SetExportRequiredGroups(true);
  Driver_Mesh::Status status = Driver_Mesh::DRS_FAIL;
  {
    py::gil_scoped_release release;
    status = writer.Perform();
  }
  if (status != Driver_Mesh::DRS_OK) {
    throw PysmeshError("write_gmf: writing '" + path + "' failed — " + status_text(status) +
                       ".");
  }
}

py::dict read_gmf(const std::string& path) {
  ScratchMesh scratch;
  DriverGMF_Read reader;
  reader.SetFile(path);
  reader.SetMesh(&scratch.ds());
  reader.SetMakeRequiredGroups(true);
  reader.SetMakeFaultGroups(true);

  Driver_Mesh::Status status = Driver_Mesh::DRS_FAIL;
  {
    py::gil_scoped_release release;
    status = reader.Perform();
  }
  if (status != Driver_Mesh::DRS_OK) {
    throw PysmeshError("read_gmf: reading '" + path + "' failed — " + status_text(status) +
                       ".");
  }

  py::dict out;
  // No shape: the reader drops the per-element reference the writer emitted, so every
  // element and node truthfully reports itself bound to nothing.
  out["mesh"] = harvest_arrays(scratch.ds(), nullptr);
  out["groups"] = harvest_groups(scratch.ds());
  return out;
}

}  // namespace mesher
}  // namespace pysmesh
