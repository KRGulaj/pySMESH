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

// A shape-free SMESH mesh, owned for the duration of one read or write. The teardown order
// is the same one the Mesher documents: the wrapper before the generator.
class ScratchMesh {
 public:
  ScratchMesh() {
    gen_ = std::make_unique<SMESH_Gen>();
    mesh_ = gen_->CreateMesh(false);
  }
  ~ScratchMesh() {
    delete mesh_;
    mesh_ = nullptr;
    gen_.reset();
  }

  ScratchMesh(const ScratchMesh&) = delete;
  ScratchMesh& operator=(const ScratchMesh&) = delete;

  SMESH_Mesh& mesh() { return *mesh_; }
  SMESHDS_Mesh& ds() { return *mesh_->GetMeshDS(); }

 private:
  std::unique_ptr<SMESH_Gen> gen_;
  SMESH_Mesh* mesh_ = nullptr;
};

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

// Rebuild one element with its own id. SMESHDS overloads AddFaceWithID / AddVolumeWithID by
// arity, so the node count is the whole dispatch — which is exactly what the compressed row
// list carries.
bool add_element(SMESHDS_Mesh& ds, int type, const std::vector<smIdType>& n, smIdType id) {
  const std::size_t k = n.size();
  switch (static_cast<SMDSAbs_ElementType>(
      static_cast<SMDSAbs_EntityType>(type) <= SMDSEntity_Quad_Edge ? SMDSAbs_Edge
      : static_cast<SMDSAbs_EntityType>(type) <= SMDSEntity_Quad_Polygon ? SMDSAbs_Face
                                                                         : SMDSAbs_Volume)) {
    case SMDSAbs_Edge:
      if (k == 2) return ds.AddEdgeWithID(n[0], n[1], id) != nullptr;
      if (k == 3) return ds.AddEdgeWithID(n[0], n[1], n[2], id) != nullptr;
      return false;
    case SMDSAbs_Face:
      if (k == 3) return ds.AddFaceWithID(n[0], n[1], n[2], id) != nullptr;
      if (k == 4) return ds.AddFaceWithID(n[0], n[1], n[2], n[3], id) != nullptr;
      if (k == 6)
        return ds.AddFaceWithID(n[0], n[1], n[2], n[3], n[4], n[5], id) != nullptr;
      if (k == 7)
        return ds.AddFaceWithID(n[0], n[1], n[2], n[3], n[4], n[5], n[6], id) != nullptr;
      if (k == 8)
        return ds.AddFaceWithID(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], id) != nullptr;
      if (k == 9)
        return ds.AddFaceWithID(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], id) !=
               nullptr;
      return false;
    default:
      if (k == 4) return ds.AddVolumeWithID(n[0], n[1], n[2], n[3], id) != nullptr;
      if (k == 5) return ds.AddVolumeWithID(n[0], n[1], n[2], n[3], n[4], id) != nullptr;
      if (k == 6)
        return ds.AddVolumeWithID(n[0], n[1], n[2], n[3], n[4], n[5], id) != nullptr;
      if (k == 8)
        return ds.AddVolumeWithID(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], id) !=
               nullptr;
      if (k == 10)
        return ds.AddVolumeWithID(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9],
                                  id) != nullptr;
      if (k == 20)
        return ds.AddVolumeWithID(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9],
                                  n[10], n[11], n[12], n[13], n[14], n[15], n[16], n[17],
                                  n[18], n[19], id) != nullptr;
      if (k == 27)
        return ds.AddVolumeWithID(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9],
                                  n[10], n[11], n[12], n[13], n[14], n[15], n[16], n[17],
                                  n[18], n[19], n[20], n[21], n[22], n[23], n[24], n[25],
                                  n[26], id) != nullptr;
      return false;
  }
}

template <class T>
py::array_t<T, py::array::c_style | py::array::forcecast> field(const py::dict& mesh,
                                                                const char* key) {
  if (!mesh.contains(key)) {
    throw PysmeshError(std::string("write_gmf: the mesh is missing '") + key + "'.");
  }
  return mesh[key].cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
}

}  // namespace

void write_gmf(const std::string& path, const py::dict& mesh, const py::list& groups) {
  const auto coords = field<double>(mesh, "node_coords");
  const auto node_ids = field<std::int64_t>(mesh, "node_id");
  const auto offsets = field<std::int64_t>(mesh, "element_offsets");
  const auto connectivity = field<std::int32_t>(mesh, "element_nodes");
  const auto types = field<std::int8_t>(mesh, "element_type");
  const auto element_ids = field<std::int64_t>(mesh, "element_id");

  const py::ssize_t n_nodes = node_ids.shape(0);
  if (coords.ndim() != 2 || coords.shape(0) != n_nodes || coords.shape(1) != 3) {
    throw PysmeshError("write_gmf: node_coords must have shape (N, 3) matching node_id.");
  }
  const py::ssize_t n_elements = types.shape(0);
  if (element_ids.shape(0) != n_elements || offsets.shape(0) != n_elements + 1) {
    throw PysmeshError("write_gmf: element_type, element_id and element_offsets disagree on "
                       "the element count.");
  }

  // Refuse a mesh the format cannot hold, naming the first offending element and its type.
  // The alternative is a file that is silently missing its cut cells.
  for (py::ssize_t i = 0; i < n_elements; ++i) {
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

  const double* xyz = coords.data();
  for (py::ssize_t i = 0; i < n_nodes; ++i) {
    const smIdType id = static_cast<smIdType>(node_ids.data()[i]);
    if (ds.AddNodeWithID(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], id) == nullptr) {
      throw PysmeshError("write_gmf: could not add node with id " + std::to_string(id) +
                         " (a duplicate or non-positive id).");
    }
  }

  std::vector<smIdType> nodes;
  for (py::ssize_t i = 0; i < n_elements; ++i) {
    const std::int64_t from = offsets.data()[i];
    const std::int64_t to = offsets.data()[i + 1];
    nodes.clear();
    for (std::int64_t j = from; j < to; ++j) {
      const std::int32_t row = connectivity.data()[j];
      if (row < 0 || row >= n_nodes) {
        throw PysmeshError("write_gmf: element_nodes[" + std::to_string(j) +
                           "] is not a row of node_coords.");
      }
      nodes.push_back(static_cast<smIdType>(node_ids.data()[row]));
    }
    const smIdType id = static_cast<smIdType>(element_ids.data()[i]);
    if (!add_element(ds, types.data()[i], nodes, id)) {
      throw PysmeshError("write_gmf: could not rebuild element " +
                         std::to_string(element_ids.data()[i]) + " (" +
                         std::to_string(nodes.size()) + " nodes, type " +
                         std::to_string(static_cast<int>(types.data()[i])) + ").");
    }
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
