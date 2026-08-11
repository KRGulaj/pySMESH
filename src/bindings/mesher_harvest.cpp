// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — the mesh out: nodes, elements, their CAD binding, and groups.
//
// The layout is a compressed row list rather than one array per element type, for two
// reasons that both come from what the consumer does with it. A volume mesh mixes cell types
// by construction — the body-fitted Cartesian mesher emits hexahedra in the interior and
// polyhedra where the grid meets the geometry, measured at 336 and 24 on a bored block — so a
// per-type array set would be a dozen arrays whose union the caller has to rebuild anyway.
// And a compressed list handles the quadratic and polyhedral cases with no special case at
// all, because a cell's node count is just the span between two offsets.
//
// Polyhedra need one thing more: their node list *is* a face stream, so the split between
// faces has to travel with it. `face_sizes` carries the per-face node counts of every
// polyhedron, and `face_offsets` says where each element's share of them starts — an empty
// span for every other element type, whose faces follow from its type.
//
// Both nodes and elements carry the sub-shape they sit on, as the caller's own (kind,
// ordinal) pair. That is what closes the loop back to the geometry: paired with the handoff's
// per-kind id arrays, a cell knows the face it lies on and the face knows its identity in the
// modelling session.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <SMDSAbs_ElementType.hxx>
#include <SMDS_ElemIterator.hxx>
#include <SMDS_Mesh.hxx>
#include <SMDS_MeshElement.hxx>
#include <SMDS_MeshNode.hxx>
#include <SMDS_MeshVolume.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_Mesh.hxx>

namespace pysmesh {
namespace mesher {
namespace {

// The kind codes crossing the boundary. 0 is "bound to nothing", which is a real state: a
// mesh read from a file has no shape at all, and an element of a shape kind the caller's
// Shape does not index (a WIRE, a SHELL) has no ordinal to give.
constexpr std::int8_t kKindNone = 0;

std::int8_t kind_code(const char* kind) {
  if (kind == nullptr || kind[0] == '\0') return kKindNone;
  if (kind[0] == 'S') return 1;  // SOLID
  if (kind[0] == 'F') return 2;  // FACE
  if (kind[0] == 'E') return 3;  // EDGE
  return 4;                      // VERTEX
}

}  // namespace

py::dict Mesher::mesh_arrays() const { return harvest_arrays(meshDS(), this); }

py::dict harvest_arrays(const SMESHDS_Mesh& ds, const Mesher* owner) {
  // A mesh read from a file carries no shape, so there is nothing to translate a SMESHDS
  // index into. It reports "bound to nothing" rather than a made-up ordinal.
  auto binding_of = [owner](int shape_index) -> std::pair<const char*, int> {
    if (owner == nullptr) {
      return {"", 0};
    }
    return owner->ordinal_of_shape_index(shape_index);
  };

  // ---- Nodes ------------------------------------------------------------------------ //
  // Row order is the iterator's order, and every element's connectivity is expressed as a
  // row index rather than an SMDS id, so a consumer indexes straight into the coordinate
  // array with no lookup of its own. The ids travel alongside for the cases that need them.
  const py::ssize_t n_nodes = static_cast<py::ssize_t>(ds.NbNodes());
  py::array_t<double> node_coords({n_nodes, py::ssize_t(3)});
  py::array_t<std::int64_t> node_id(n_nodes);
  py::array_t<std::int8_t> node_kind(n_nodes);
  py::array_t<std::int32_t> node_ordinal(n_nodes);

  double* coord = node_coords.mutable_data();
  std::int64_t* nid = node_id.mutable_data();
  std::int8_t* nkind = node_kind.mutable_data();
  std::int32_t* nord = node_ordinal.mutable_data();

  // SMDS ids are sparse after editing, so the id -> row map is a dense table indexed by id
  // rather than a hash: it is one allocation and one store per node instead of a rehash.
  std::vector<std::int32_t> row_of_id(static_cast<std::size_t>(ds.MaxNodeID()) + 1, -1);
  {
    std::int32_t row = 0;
    for (SMDS_NodeIteratorPtr it = ds.nodesIterator(); it->more(); ++row) {
      const SMDS_MeshNode* node = it->next();
      coord[3 * row] = node->X();
      coord[3 * row + 1] = node->Y();
      coord[3 * row + 2] = node->Z();
      const std::int64_t id = static_cast<std::int64_t>(node->GetID());
      nid[row] = id;
      const std::pair<const char*, int> at = binding_of(node->getshapeId());
      nkind[row] = kind_code(at.first);
      nord[row] = static_cast<std::int32_t>(at.second);
      if (id >= 0 && static_cast<std::size_t>(id) < row_of_id.size()) {
        row_of_id[static_cast<std::size_t>(id)] = row;
      }
    }
  }

  auto row_of = [&row_of_id](const SMDS_MeshNode* node) -> std::int32_t {
    const std::int64_t id = static_cast<std::int64_t>(node->GetID());
    const std::int32_t row =
        (id >= 0 && static_cast<std::size_t>(id) < row_of_id.size())
            ? row_of_id[static_cast<std::size_t>(id)]
            : -1;
    if (row < 0) {
      throw PysmeshError("Mesher.mesh: an element references a node absent from the mesh.");
    }
    return row;
  };

  // ---- Elements --------------------------------------------------------------------- //
  std::vector<std::int64_t> offsets;
  std::vector<std::int32_t> connectivity;
  std::vector<std::int8_t> types;
  std::vector<std::int64_t> ids;
  std::vector<std::int8_t> kinds;
  std::vector<std::int32_t> ordinals;
  std::vector<std::int64_t> face_offsets;
  std::vector<std::int32_t> face_sizes;

  const std::size_t n_elements = static_cast<std::size_t>(ds.NbEdges() + ds.NbFaces() +
                                                          ds.NbVolumes() + ds.Nb0DElements() +
                                                          ds.NbBalls());
  offsets.reserve(n_elements + 1);
  types.reserve(n_elements);
  ids.reserve(n_elements);
  kinds.reserve(n_elements);
  ordinals.reserve(n_elements);
  face_offsets.reserve(n_elements + 1);
  offsets.push_back(0);
  face_offsets.push_back(0);

  // Grouped by dimension, ascending, so a consumer that wants only the volume cells reads
  // one contiguous span rather than filtering. SMDSAbs_All would interleave them.
  constexpr SMDSAbs_ElementType kOrder[] = {SMDSAbs_0DElement, SMDSAbs_Ball, SMDSAbs_Edge,
                                            SMDSAbs_Face, SMDSAbs_Volume};
  for (const SMDSAbs_ElementType type : kOrder) {
    for (SMDS_ElemIteratorPtr it = ds.elementsIterator(type); it->more();) {
      const SMDS_MeshElement* elem = it->next();
      const int nb = elem->NbNodes();
      for (int i = 0; i < nb; ++i) {
        connectivity.push_back(row_of(elem->GetNode(i)));
      }
      offsets.push_back(static_cast<std::int64_t>(connectivity.size()));
      types.push_back(static_cast<std::int8_t>(element_type_code(elem->GetEntityType())));
      ids.push_back(static_cast<std::int64_t>(elem->GetID()));
      const std::pair<const char*, int> at = binding_of(elem->getshapeId());
      kinds.push_back(kind_code(at.first));
      ordinals.push_back(static_cast<std::int32_t>(at.second));

      // A polyhedron's node list is its face stream; every other type's faces follow from
      // its type, so it contributes an empty span here.
      if (elem->GetEntityType() == SMDSEntity_Polyhedra ||
          elem->GetEntityType() == SMDSEntity_Quad_Polyhedra) {
        const SMDS_MeshVolume* volume = SMDS_Mesh::DownCast<SMDS_MeshVolume>(elem);
        if (volume == nullptr) {
          throw PysmeshError("Mesher.mesh: a polyhedral element is not a volume element.");
        }
        for (const int q : volume->GetQuantities()) {
          face_sizes.push_back(static_cast<std::int32_t>(q));
        }
      }
      face_offsets.push_back(static_cast<std::int64_t>(face_sizes.size()));
    }
  }

  py::dict out;
  out["node_coords"] = node_coords;
  out["node_id"] = node_id;
  out["node_kind"] = node_kind;
  out["node_ordinal"] = node_ordinal;
  out["element_offsets"] = vector_to_array(offsets);
  out["element_nodes"] = vector_to_array(connectivity);
  out["element_type"] = vector_to_array(types);
  out["element_id"] = vector_to_array(ids);
  out["element_kind"] = vector_to_array(kinds);
  out["element_ordinal"] = vector_to_array(ordinals);
  out["face_offsets"] = vector_to_array(face_offsets);
  out["face_sizes"] = vector_to_array(face_sizes);
  return out;
}

}  // namespace mesher
}  // namespace pysmesh
