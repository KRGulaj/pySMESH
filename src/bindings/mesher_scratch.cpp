// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — a shape-free mesh, the rebuild of one from plain arrays, and the fill.
//
// Two consumers need the same thing: the Inria writer has to hand SMESH a real SMDS mesh
// before its driver can serialise one, and the quality controls have to run on a mesh a
// caller supplied as arrays — one read from a file, or one built by hand. Both are the same
// operation, so it lives in one place rather than being copied into each.
//
// The rebuild keeps every id. That matters more than it looks: a control reports its values
// keyed by element id, and a group's membership is a set of ids, so a rebuild that renumbered
// would quietly break the correspondence between what a caller passed in and what comes back.
//
// The third thing here is the *fill*: `Mesher::add_nodes` and `Mesher::add_elements`, which
// are how a discrete body with no B-rep behind it — an imported STL, a shrink-wrap result, a
// boundary another mesher produced — becomes a live SMESH mesh that the whole editing and
// search surface applies to. It sits beside the rebuild because it is the same act at a
// different granularity, and because both have to know which entity types a rectangular
// table can express at all.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRep_Tool.hxx>
#include <Precision.hxx>
#include <SMDSAbs_ElementType.hxx>
#include <SMDS_MeshCell.hxx>
#include <SMDS_MeshNode.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_Mesh.hxx>
#include <ShapeAnalysis_Curve.hxx>
#include <ShapeAnalysis_Surface.hxx>
#include <TopoDS.hxx>

namespace pysmesh {
namespace mesher {
namespace {

// Which family an entity type belongs to. The rebuild dispatches on it first and on the node
// count second, because SMESHDS overloads AddFaceWithID / AddVolumeWithID by arity.
SMDSAbs_ElementType family_of(int type) {
  const SMDSAbs_EntityType entity = static_cast<SMDSAbs_EntityType>(type);
  if (entity <= SMDSEntity_0D) return SMDSAbs_0DElement;
  if (entity <= SMDSEntity_Quad_Edge) return SMDSAbs_Edge;
  if (entity <= SMDSEntity_Quad_Polygon) return SMDSAbs_Face;
  if (entity == SMDSEntity_Ball) return SMDSAbs_Ball;
  return SMDSAbs_Volume;
}

// A polygon's and a polyhedron's node counts carry no shape information at all — the same
// count means a different cell — so they have no arity-keyed constructor and cannot be
// rebuilt through this path. Named rather than silently dropped.
bool is_free_form(int type) {
  switch (static_cast<SMDSAbs_EntityType>(type)) {
    case SMDSEntity_Polygon:
    case SMDSEntity_Quad_Polygon:
    case SMDSEntity_Polyhedra:
    case SMDSEntity_Quad_Polyhedra:
      return true;
    default:
      return false;
  }
}

bool add_element(SMESHDS_Mesh& ds, int type, const std::vector<smIdType>& n, smIdType id) {
  const std::size_t k = n.size();
  switch (family_of(type)) {
    case SMDSAbs_0DElement:
      if (k == 1) return ds.Add0DElementWithID(n[0], id) != nullptr;
      return false;
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
    case SMDSAbs_Volume:
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
      if (k == 12)
        return ds.AddVolumeWithID(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9],
                                  n[10], n[11], id) != nullptr;
      if (k == 13)
        return ds.AddVolumeWithID(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9],
                                  n[10], n[11], n[12], id) != nullptr;
      if (k == 15)
        return ds.AddVolumeWithID(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9],
                                  n[10], n[11], n[12], n[13], n[14], id) != nullptr;
      if (k == 18)
        return ds.AddVolumeWithID(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9],
                                  n[10], n[11], n[12], n[13], n[14], n[15], n[16], n[17],
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
    default:
      return false;
  }
}

template <class T>
py::array_t<T, py::array::c_style | py::array::forcecast> field(const py::dict& mesh,
                                                                const char* key) {
  if (!mesh.contains(key)) {
    throw PysmeshError(std::string("the mesh is missing '") + key + "'.");
  }
  return mesh[key].cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
}

}  // namespace

ScratchMesh::ScratchMesh() {
  gen_ = std::make_unique<SMESH_Gen>();
  mesh_ = gen_->CreateMesh(false);
}

ScratchMesh::~ScratchMesh() {
  delete mesh_;
  mesh_ = nullptr;
  gen_.reset();
}

SMESHDS_Mesh& ScratchMesh::ds() const { return *mesh_->GetMeshDS(); }

void rebuild_mesh(SMESHDS_Mesh& ds, const py::dict& mesh) {
  const auto coords = field<double>(mesh, "node_coords");
  const auto node_ids = field<std::int64_t>(mesh, "node_id");
  const auto offsets = field<std::int64_t>(mesh, "element_offsets");
  const auto connectivity = field<std::int32_t>(mesh, "element_nodes");
  const auto types = field<std::int8_t>(mesh, "element_type");
  const auto element_ids = field<std::int64_t>(mesh, "element_id");

  const py::ssize_t n_nodes = node_ids.shape(0);
  if (coords.ndim() != 2 || coords.shape(0) != n_nodes || coords.shape(1) != 3) {
    throw PysmeshError("node_coords must have shape (N, 3) matching node_id.");
  }
  const py::ssize_t n_elements = types.shape(0);
  if (element_ids.shape(0) != n_elements || offsets.shape(0) != n_elements + 1) {
    throw PysmeshError(
        "element_type, element_id and element_offsets disagree on the element count.");
  }

  const double* xyz = coords.data();
  for (py::ssize_t i = 0; i < n_nodes; ++i) {
    const smIdType id = static_cast<smIdType>(node_ids.data()[i]);
    if (ds.AddNodeWithID(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], id) == nullptr) {
      throw PysmeshError("could not add node with id " + std::to_string(id) +
                         " (a duplicate or non-positive id).");
    }
  }

  std::vector<smIdType> nodes;
  for (py::ssize_t i = 0; i < n_elements; ++i) {
    if (is_free_form(types.data()[i])) {
      throw PysmeshError(
          "element " + std::to_string(element_ids.data()[i]) +
          " is a polygon or a polyhedron, whose node count does not determine its shape, so "
          "it cannot be rebuilt from the element arrays alone.");
    }
    const std::int64_t from = offsets.data()[i];
    const std::int64_t to = offsets.data()[i + 1];
    nodes.clear();
    for (std::int64_t j = from; j < to; ++j) {
      const std::int32_t row = connectivity.data()[j];
      if (row < 0 || row >= n_nodes) {
        throw PysmeshError("element_nodes[" + std::to_string(j) +
                           "] is not a row of node_coords.");
      }
      nodes.push_back(static_cast<smIdType>(node_ids.data()[row]));
    }
    const smIdType id = static_cast<smIdType>(element_ids.data()[i]);
    if (!add_element(ds, types.data()[i], nodes, id)) {
      throw PysmeshError("could not rebuild element " +
                         std::to_string(element_ids.data()[i]) + " (" +
                         std::to_string(nodes.size()) + " nodes, type " +
                         std::to_string(static_cast<int>(types.data()[i])) + ").");
    }
  }

  // Publish the build. SMDS records that it changed but only advances its modification time
  // when asked, and a mesh assembled here has never been asked — so its time is still 0.
  // Several controls cache against that time and treat "0 == 0" as "nothing has changed
  // since I last looked", which on a fresh mesh means they never look at all. Measured: the
  // coincident-node test found nothing on a mesh with two nodes at the same point.
  ds.Modified();
}

// ---- The fill -------------------------------------------------------------------------- //

namespace {

// The dimension of a sub-shape kind, as the element families count it.
int dimension_of(const std::string& kind) {
  if (kind == "VERTEX") return 0;
  if (kind == "EDGE") return 1;
  if (kind == "FACE") return 2;
  return 3;
}

// The dimension of an element type, or -1 for a ball, which no sub-shape kind matches.
int element_dimension(int type) {
  switch (family_of(type)) {
    case SMDSAbs_0DElement:
      return 0;
    case SMDSAbs_Edge:
      return 1;
    case SMDSAbs_Face:
      return 2;
    case SMDSAbs_Volume:
      return 3;
    default:
      return -1;
  }
}

// How far a node may lie from the sub-shape it is bound to: the sub-shape's own tolerance,
// and never less than Precision::Confusion().
double binding_tolerance(const TopoDS_Shape& s) {
  double tol = Precision::Confusion();
  if (s.ShapeType() == TopAbs_VERTEX) tol = BRep_Tool::Tolerance(TopoDS::Vertex(s));
  if (s.ShapeType() == TopAbs_EDGE) tol = BRep_Tool::Tolerance(TopoDS::Edge(s));
  if (s.ShapeType() == TopAbs_FACE) tol = BRep_Tool::Tolerance(TopoDS::Face(s));
  return std::max(tol, Precision::Confusion());
}

// Where on its sub-shape each node sits: u on an edge, (u, v) on a face. Computed, and
// checked, for every row before any node is added, so a refusal leaves the mesh unchanged.
std::vector<std::array<double, 2>> node_sites(const TopoDS_Shape& s, const std::string& kind,
                                              int ordinal, const double* xyz, py::ssize_t n,
                                              const py::object& parameters) {
  const double tol = binding_tolerance(s);
  const std::string where = kind + " " + std::to_string(ordinal);
  auto refuse = [&](py::ssize_t row, double gap, const std::string& why) {
    throw PysmeshError("Mesher.add_nodes: the node at row " + std::to_string(row) +
                       " does not lie on " + where + ": " + why + " (gap " +
                       std::to_string(gap) + ", tolerance " + std::to_string(tol) + ").");
  };
  const int columns = kind == "EDGE" ? 1 : kind == "FACE" ? 2 : 0;
  std::vector<double> given;
  if (!parameters.is_none()) {
    if (columns == 0) {
      throw PysmeshError("Mesher.add_nodes: parameters are read on an EDGE or a FACE only, "
                         "not on a " + kind + ".");
    }
    const auto table = point_table(parameters.attr("reshape")(n, columns), "parameters",
                                   columns);
    given.assign(table.data(), table.data() + table.size());
  }
  std::vector<std::array<double, 2>> sites(static_cast<std::size_t>(n), {0.0, 0.0});
  for (py::ssize_t i = 0; i < n; ++i) {
    const gp_Pnt p(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    std::array<double, 2>& site = sites[static_cast<std::size_t>(i)];
    if (kind == "VERTEX") {
      const double gap = p.Distance(BRep_Tool::Pnt(TopoDS::Vertex(s)));
      if (gap > tol) refuse(i, gap, "it is not at the vertex");
    } else if (kind == "EDGE") {
      const BRepAdaptor_Curve curve(TopoDS::Edge(s));
      double gap = 0.0;
      if (!given.empty()) {
        site[0] = given[static_cast<std::size_t>(i)];
        if (site[0] < curve.FirstParameter() - Precision::PConfusion() ||
            site[0] > curve.LastParameter() + Precision::PConfusion()) {
          refuse(i, 0.0, "u = " + std::to_string(site[0]) + " is outside [" +
                             std::to_string(curve.FirstParameter()) + ", " +
                             std::to_string(curve.LastParameter()) + "]");
        }
        gap = p.Distance(curve.Value(site[0]));
      } else {
        gp_Pnt proj;
        gap = ShapeAnalysis_Curve().Project(curve, p, tol, proj, site[0]);
      }
      if (gap > tol) refuse(i, gap, "the edge at u is that far from it");
    } else if (kind == "FACE") {
      const TopoDS_Face& face = TopoDS::Face(s);
      double gap = 0.0;
      if (!given.empty()) {
        site = {given[2 * static_cast<std::size_t>(i)],
                given[2 * static_cast<std::size_t>(i) + 1]};
        gap = p.Distance(BRepAdaptor_Surface(face).Value(site[0], site[1]));
      } else {
        ShapeAnalysis_Surface surface(BRep_Tool::Surface(face));
        const gp_Pnt2d uv = surface.ValueOfUV(p, tol);
        site = {uv.X(), uv.Y()};
        gap = surface.Gap();
      }
      if (gap > tol) refuse(i, gap, "the surface at (u, v) is that far from it");
      const BRepClass_FaceClassifier inside(face, gp_Pnt2d(site[0], site[1]), tol);
      if (inside.State() == TopAbs_OUT) refuse(i, gap, "it is outside the face boundary");
    } else {
      BRepClass3d_SolidClassifier inside(s, p, tol);
      if (inside.State() == TopAbs_OUT) refuse(i, 0.0, "it is outside the solid");
    }
  }
  return sites;
}

}  // namespace

py::array_t<std::int64_t> Mesher::add_nodes(const py::object& coords, const std::string& kind,
                                            int ordinal, const py::object& parameters) {
  ensure_open();
  const auto table = point_table(coords, "coords", 3);
  const py::ssize_t n = table.shape(0);
  const double* xyz = table.data();

  TopoDS_Shape target;
  std::vector<std::array<double, 2>> sites;
  if (!kind.empty()) {
    target = sub_shape(kind, ordinal);  // validates the kind and the ordinal
    sites = node_sites(target, kind, ordinal, xyz, n, parameters);
  } else if (!parameters.is_none()) {
    throw PysmeshError("Mesher.add_nodes: parameters need a sub-shape to be read on.");
  }

  py::array_t<std::int64_t> ids(n);
  std::int64_t* out = ids.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    const SMDS_MeshNode* node = meshDS_->AddNode(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    if (node == nullptr) {
      throw PysmeshError("Mesher.add_nodes: SMESH refused the node at row " +
                         std::to_string(i) + ".");
    }
    if (!kind.empty()) {
      const std::array<double, 2>& site = sites[static_cast<std::size_t>(i)];
      if (kind == "VERTEX") meshDS_->SetNodeOnVertex(node, TopoDS::Vertex(target));
      if (kind == "EDGE") meshDS_->SetNodeOnEdge(node, TopoDS::Edge(target), site[0]);
      if (kind == "FACE") {
        meshDS_->SetNodeOnFace(node, TopoDS::Face(target), site[0], site[1]);
      }
      if (kind == "SOLID") meshDS_->SetNodeInVolume(node, TopoDS::Solid(target));
    }
    out[i] = static_cast<std::int64_t>(node->GetID());
  }
  if (n > 0) {
    meshDS_->Modified();
  }
  return ids;
}

py::array_t<std::int64_t> Mesher::add_elements(int type, const py::object& connectivity,
                                               const std::string& kind, int ordinal) {
  ensure_open();
  if (type < 0 || type >= static_cast<int>(SMDSEntity_Last)) {
    throw PysmeshError("Mesher.add_elements: " + std::to_string(type) +
                       " is not an element type.");
  }
  TopoDS_Shape target;
  if (!kind.empty()) {
    target = sub_shape(kind, ordinal);  // validates the kind and the ordinal
    if (element_dimension(type) != dimension_of(kind)) {
      throw PysmeshError("Mesher.add_elements: an element of type " + std::to_string(type) +
                         " cannot be bound to " + kind + " " + std::to_string(ordinal) +
                         ": the dimensions differ.");
    }
  }
  const SMDSAbs_EntityType entity = static_cast<SMDSAbs_EntityType>(type);
  if (entity == SMDSEntity_Node) {
    throw PysmeshError("Mesher.add_elements: NODE is not an element type. Use add_nodes.");
  }
  if (is_free_form(type)) {
    throw PysmeshError(
        "Mesher.add_elements: a polygon or a polyhedron cannot be given as a table, because "
        "its node count does not determine its shape — the same count means a different "
        "cell. Only the fixed-arity types can be added this way.");
  }

  const auto conn =
      connectivity.cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>();
  if (conn.ndim() != 2) {
    throw PysmeshError("Mesher.add_elements: connectivity must have shape (M, k).");
  }
  // The arity is asked of SMDS rather than tabulated here. A triangle handed four columns
  // would otherwise dispatch on the count alone and build a *quadrangle* under the name the
  // caller asked for, which is the one failure mode of this call that is silent.
  const py::ssize_t arity = static_cast<py::ssize_t>(SMDS_MeshCell::NbNodes(entity));
  if (conn.shape(1) != arity) {
    throw PysmeshError("Mesher.add_elements: element type " + std::to_string(type) +
                       " has " + std::to_string(arity) + " nodes, but connectivity has " +
                       std::to_string(conn.shape(1)) + " columns.");
  }

  const py::ssize_t m = conn.shape(0);
  py::array_t<std::int64_t> ids(m);
  std::int64_t* out = ids.mutable_data();
  const std::int64_t* rows = conn.data();

  // A fresh id per element, taken above everything the mesh already holds. SMDS numbers every
  // element of every dimension in one sequence, so one running counter is the whole of it.
  smIdType next = meshDS_->MaxElementID() + 1;
  std::vector<smIdType> nodes(static_cast<std::size_t>(arity));
  for (py::ssize_t i = 0; i < m; ++i) {
    for (py::ssize_t j = 0; j < arity; ++j) {
      const std::int64_t node_id = rows[i * arity + j];
      if (meshDS_->FindNode(static_cast<smIdType>(node_id)) == nullptr) {
        throw PysmeshError("Mesher.add_elements: row " + std::to_string(i) + " names node " +
                           std::to_string(node_id) + ", which the mesh does not have.");
      }
      nodes[static_cast<std::size_t>(j)] = static_cast<smIdType>(node_id);
    }
    if (!add_element(*meshDS_, type, nodes, next)) {
      throw PysmeshError("Mesher.add_elements: SMESH refused row " + std::to_string(i) +
                         " as element type " + std::to_string(type) + ".");
    }
    if (!kind.empty()) {
      meshDS_->SetMeshElementOnShape(meshDS_->FindElement(next), target);
    }
    out[i] = static_cast<std::int64_t>(next);
    ++next;
  }
  if (m > 0) {
    meshDS_->Modified();
  }
  return ids;
}

void Mesher::fill_from_mesh(const py::dict& mesh) {
  ensure_open();
  // The arrays carry absolute ids, so filling on top of anything would collide on the first
  // id already in use and leave the mesh half-built. Refused up front instead.
  if (meshDS_->NbNodes() != 0 || meshDS_->NbElements() != 0) {
    throw PysmeshError("Mesher.fill_from_mesh: the mesh is not empty.",
                       "The arrays name absolute node and element ids and this call keeps "
                       "them, so it can only fill a mesh that holds nothing yet.");
  }
  rebuild_mesh(*meshDS_, mesh);
}

}  // namespace mesher
}  // namespace pysmesh
