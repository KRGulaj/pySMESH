// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-07-03

// pySMESH binding — surface-mesh injection: Mesh, node/element insertion, CAD
// classification, removal, validate/stats. See common.hpp for ShapeData and PysmeshError.
//
// The class always resolves a Python-facing face_id/edge_id/vertex_id to the exact
// TopoDS_* object (via the shared ShapeData) and calls the SMESHDS shape-reference
// overloads — never the int-Index overloads (SMESHDS_Mesh_notes.md §CRITICAL).
//
// Removal goes through SMESH_MeshEditor::Remove rather than SMESHDS directly, because the
// editor is what notifies the VERTEX sub-meshes that an entity they own has gone. What went
// is then read back from the mesh, never echoed from the request: an id listed twice is
// removed once, and removing a node takes every element built on it with it.

#include <list>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <SMDS_ElemIterator.hxx>
#include <SMDS_MeshEdge.hxx>
#include <SMDS_MeshElement.hxx>
#include <SMDS_MeshNode.hxx>
#include <SMDS_Position.hxx>
#include <SMDS_TypeOfPosition.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESHDS_SubMesh.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_Hypothesis.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_MeshEditor.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Vertex.hxx>

#include "common.hpp"

namespace pysmesh {

// Defined in shape.cpp — hand the Mesh the same ShapeData its Shape argument holds.
std::shared_ptr<ShapeData> shape_data_of(const py::object& shape_obj);

namespace {

struct MeshStats {
  std::int64_t n_nodes;
  std::int64_t n_faces;
  std::vector<std::pair<int, std::int64_t>> per_face_element_counts;  // (face_id, count)
};

// What one removal took away. The two id lists are the point of it: a count says how much
// went, and a caller remapping named selections, a viewport highlight or a per-element field
// against the survivors needs to know exactly which entities it lost.
struct MeshRemoval {
  std::vector<std::int64_t> elements;
  std::vector<std::int64_t> nodes;
  std::int64_t n_nodes_before;
  std::int64_t n_nodes_after;
  std::int64_t n_elements_before;
  std::int64_t n_elements_after;
};

Array1i to_array(const std::vector<std::int64_t>& values) {
  Array1i out(static_cast<py::ssize_t>(values.size()));
  if (!values.empty()) {
    std::copy(values.begin(), values.end(), out.mutable_data());
  }
  return out;
}

// ---- Mesh --------------------------------------------------------------------------//
class Mesh {
 public:
  explicit Mesh(const py::object& shape_obj) : data_(shape_data_of(shape_obj)) {
    gen_ = std::make_unique<SMESH_Gen>();
    mesh_ = gen_->CreateMesh(false);  // owned by gen_; freed when gen_ is reset
    mesh_->ShapeToMesh(data_->shape);
    meshDS_ = mesh_->GetMeshDS();
  }

  ~Mesh() { release(); }

  // Insert N nodes; return their SMESH ids as an int64 (N,) array.
  Array1i add_nodes(const py::object& coords_obj) {
    ensure_open();
    Array2d coords = as_2d_f64(coords_obj, "coords", 3);
    const py::ssize_t n = coords.shape(0);
    Array1i ids(n);
    const double* c = coords.data();
    std::int64_t* out = ids.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
      const SMDS_MeshNode* node = meshDS_->AddNode(c[3 * i], c[3 * i + 1], c[3 * i + 2]);
      out[i] = static_cast<std::int64_t>(node->GetID());
    }
    return ids;
  }

  void classify_on_face(const Array1i& node_ids, int face_id, const py::object& uv_obj) {
    ensure_open();
    const TopoDS_Face& face = data_->face(face_id);  // validates face_id
    Array2d uv = as_2d_f64(uv_obj, "uv", 2);
    const py::ssize_t n = node_ids.shape(0);
    if (uv.shape(0) != n) {
      throw PysmeshError("node_ids and uv must have matching length");
    }
    const std::int64_t* ids = node_ids.data();
    const double* p = uv.data();
    for (py::ssize_t i = 0; i < n; ++i) {
      meshDS_->SetNodeOnFace(find_node(ids[i]), face, p[2 * i], p[2 * i + 1]);
    }
  }

  void classify_on_edge(const Array1i& node_ids, int edge_id, const py::object& t_obj) {
    ensure_open();
    const TopoDS_Edge& edge = data_->edge(edge_id);
    py::array_t<double, py::array::c_style | py::array::forcecast> t =
        t_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    if (t.ndim() != 1 || t.shape(0) != node_ids.shape(0)) {
      throw PysmeshError("t must be 1-D with the same length as node_ids");
    }
    const std::int64_t* ids = node_ids.data();
    const double* tv = t.data();
    for (py::ssize_t i = 0; i < node_ids.shape(0); ++i) {
      meshDS_->SetNodeOnEdge(find_node(ids[i]), edge, tv[i]);
    }
  }

  void classify_on_vertex(std::int64_t node_id, int vertex_id) {
    ensure_open();
    meshDS_->SetNodeOnVertex(find_node(node_id), data_->vertex(vertex_id));
  }

  // Add M segments (node-id pairs) bound to the edge's submesh. Viscous layers require
  // 1-D elements on the edges bounding the wall faces (StdMeshers_ViscousLayers.cxx:4392
  // errors "Not meshed EDGE" on an edge submesh with zero elements); classification of the
  // edge nodes alone is not sufficient. A real 2-D surface mesh (e.g. from Gmsh) always
  // carries these edge segments, so injection must too.
  // Returns the new elements' SMESH ids, which are what remove_elements addresses them by.
  Array1i add_segments(const py::object& conn_obj, int edge_id) {
    ensure_open();
    const TopoDS_Edge& edge = data_->edge(edge_id);
    Array2i conn = conn_obj.cast<Array2i>();
    if (conn.ndim() != 2 || conn.shape(1) != 2) {
      throw PysmeshError("conn must have shape (M, 2)");
    }
    const py::ssize_t m = conn.shape(0);
    const std::int64_t* c = conn.data();
    Array1i ids(m);
    std::int64_t* out = ids.mutable_data();
    for (py::ssize_t i = 0; i < m; ++i) {
      const SMDS_MeshNode* n0 = find_node(c[2 * i]);
      const SMDS_MeshNode* n1 = find_node(c[2 * i + 1]);
      SMDS_MeshEdge* elem = meshDS_->AddEdge(n0, n1);
      if (elem == nullptr) {
        throw PysmeshError("Failed to add segment " + std::to_string(i) + " on edge_id " +
                            std::to_string(edge_id));
      }
      meshDS_->SetMeshElementOnShape(elem, edge);
      out[i] = static_cast<std::int64_t>(elem->GetID());
    }
    return ids;
  }

  // Add M triangles (node-id triples) bound to the face's submesh. Returns their ids.
  Array1i add_triangles(const py::object& conn_obj, int face_id) {
    ensure_open();
    const TopoDS_Face& face = data_->face(face_id);
    Array2i conn = conn_obj.cast<Array2i>();
    if (conn.ndim() != 2 || conn.shape(1) != 3) {
      throw PysmeshError("conn must have shape (M, 3)");
    }
    const py::ssize_t m = conn.shape(0);
    const std::int64_t* c = conn.data();
    Array1i ids(m);
    std::int64_t* out = ids.mutable_data();
    for (py::ssize_t i = 0; i < m; ++i) {
      const SMDS_MeshNode* n0 = find_node(c[3 * i]);
      const SMDS_MeshNode* n1 = find_node(c[3 * i + 1]);
      const SMDS_MeshNode* n2 = find_node(c[3 * i + 2]);
      SMDS_MeshFace* elem = meshDS_->AddFace(n0, n1, n2);
      if (elem == nullptr) {
        throw PysmeshError("Failed to add triangle " + std::to_string(i) +
                            " on face_id " + std::to_string(face_id));
      }
      meshDS_->SetMeshElementOnShape(elem, face);
      out[i] = static_cast<std::int64_t>(elem->GetID());
    }
    return ids;
  }

  // Delete elements. Nothing is put in their place: this leaves a hole in the mesh, which
  // is what dropping a patch of an injected surface means.
  //
  // Nodes are not removed with them unless `free_nodes` is set. A node is an entity of its
  // own and the ones on the boundary of a deleted patch usually still carry other elements;
  // the sweep takes only the ones left carrying nothing.
  MeshRemoval remove_elements(const Array1i& element_ids, bool free_nodes) {
    ensure_open();
    // Resolve every id first, so a bad one refuses the whole call rather than leaving the
    // mesh half-edited.
    std::set<std::int64_t> targets;
    std::set<const SMDS_MeshNode*> touched;
    const std::int64_t* ids = element_ids.data();
    for (py::ssize_t i = 0; i < element_ids.shape(0); ++i) {
      const SMDS_MeshElement* element = find_element(ids[i]);
      targets.insert(static_cast<std::int64_t>(element->GetID()));
      if (!free_nodes) {
        continue;
      }
      for (SMDS_NodeIteratorPtr it = element->nodeIterator(); it->more();) {
        touched.insert(it->next());
      }
    }

    MeshRemoval out;
    out.n_nodes_before = static_cast<std::int64_t>(meshDS_->NbNodes());
    out.n_elements_before = static_cast<std::int64_t>(meshDS_->NbElements());

    std::list<smIdType> to_remove;
    for (const std::int64_t id : targets) {
      to_remove.push_back(static_cast<smIdType>(id));
    }
    SMESH_MeshEditor editor(mesh_);
    editor.Remove(to_remove, /*isNodes=*/false);

    for (const std::int64_t id : targets) {
      if (meshDS_->FindElement(static_cast<smIdType>(id)) == nullptr) {
        out.elements.push_back(id);
      }
    }

    if (free_nodes) {
      std::list<smIdType> orphans;
      for (const SMDS_MeshNode* node : touched) {
        if (node->NbInverseElements() == 0) {
          orphans.push_back(node->GetID());
        }
      }
      if (!orphans.empty()) {
        editor.Remove(orphans, /*isNodes=*/true);
        for (const smIdType id : orphans) {
          if (meshDS_->FindNode(id) == nullptr) {
            out.nodes.push_back(static_cast<std::int64_t>(id));
          }
        }
      }
    }

    meshDS_->Modified();
    out.n_nodes_after = static_cast<std::int64_t>(meshDS_->NbNodes());
    out.n_elements_after = static_cast<std::int64_t>(meshDS_->NbElements());
    return out;
  }

  // Delete nodes, and every element built on them. The cascade is not a convenience: an
  // element whose corner has gone is not a cell at all, so SMESH removes it with the node.
  // The elements it took are named in the result, because they are the part the caller did
  // not ask for.
  MeshRemoval remove_nodes(const Array1i& node_ids) {
    ensure_open();
    std::set<std::int64_t> targets;
    std::set<std::int64_t> cascade;
    const std::int64_t* ids = node_ids.data();
    for (py::ssize_t i = 0; i < node_ids.shape(0); ++i) {
      const SMDS_MeshNode* node = find_node(ids[i]);
      targets.insert(static_cast<std::int64_t>(node->GetID()));
      // Recorded before the removal, because afterwards there is nothing left to ask.
      for (SMDS_ElemIteratorPtr it = node->GetInverseElementIterator(); it->more();) {
        cascade.insert(static_cast<std::int64_t>(it->next()->GetID()));
      }
    }

    MeshRemoval out;
    out.n_nodes_before = static_cast<std::int64_t>(meshDS_->NbNodes());
    out.n_elements_before = static_cast<std::int64_t>(meshDS_->NbElements());

    std::list<smIdType> to_remove;
    for (const std::int64_t id : targets) {
      to_remove.push_back(static_cast<smIdType>(id));
    }
    SMESH_MeshEditor editor(mesh_);
    editor.Remove(to_remove, /*isNodes=*/true);

    for (const std::int64_t id : targets) {
      if (meshDS_->FindNode(static_cast<smIdType>(id)) == nullptr) {
        out.nodes.push_back(id);
      }
    }
    for (const std::int64_t id : cascade) {
      if (meshDS_->FindElement(static_cast<smIdType>(id)) == nullptr) {
        out.elements.push_back(id);
      }
    }

    meshDS_->Modified();
    out.n_nodes_after = static_cast<std::int64_t>(meshDS_->NbNodes());
    out.n_elements_after = static_cast<std::int64_t>(meshDS_->NbElements());
    return out;
  }

  // Every node classified onto a face/edge/vertex, every face carrying >=1 element.
  // Raise PysmeshError listing all gaps; never silently pass an invalid mesh.
  void validate() {
    ensure_open();
    std::vector<std::int64_t> unclassified;
    for (SMDS_NodeIteratorPtr it = meshDS_->nodesIterator(); it->more();) {
      const SMDS_MeshNode* node = it->next();
      const SMDS_TypeOfPosition t = node->GetPosition()->GetTypeOfPosition();
      if (t == SMDS_TOP_3DSPACE || t == SMDS_TOP_UNSPEC) {
        unclassified.push_back(static_cast<std::int64_t>(node->GetID()));
      }
    }

    std::vector<int> empty_faces;
    for (int fid = 1; fid <= data_->faces.Extent(); ++fid) {
      const TopoDS_Face& face = TopoDS::Face(data_->faces.FindKey(fid));
      const SMESHDS_SubMesh* sub = meshDS_->MeshElements(face);
      if (sub == nullptr || sub->NbElements() == 0) {
        empty_faces.push_back(fid);
      }
    }

    if (unclassified.empty() && empty_faces.empty()) {
      return;
    }
    std::string msg = "Mesh validation failed:";
    if (!unclassified.empty()) {
      msg += "\n  unclassified node ids: " + join(unclassified);
    }
    if (!empty_faces.empty()) {
      std::string faces;
      for (std::size_t i = 0; i < empty_faces.size(); ++i) {
        faces += (i ? ", " : "") + std::to_string(empty_faces[i]);
      }
      msg += "\n  face_ids with no elements: " + faces;
    }
    throw PysmeshError(msg, /*details=*/std::string(), empty_faces);
  }

  MeshStats stats() {
    ensure_open();
    MeshStats s;
    s.n_nodes = static_cast<std::int64_t>(meshDS_->NbNodes());
    s.n_faces = static_cast<std::int64_t>(meshDS_->NbFaces());
    for (int fid = 1; fid <= data_->faces.Extent(); ++fid) {
      const TopoDS_Face& face = TopoDS::Face(data_->faces.FindKey(fid));
      const SMESHDS_SubMesh* sub = meshDS_->MeshElements(face);
      s.per_face_element_counts.emplace_back(
          fid, sub ? static_cast<std::int64_t>(sub->NbElements()) : 0);
    }
    return s;
  }

  void release() {
    // Idempotent teardown. SMESH_Gen::CreateMesh does `new SMESH_Mesh` and stores it in
    // its studyContext, but ~SMESH_Gen deletes only the document (the SMESHDS grid) and
    // hypotheses — never the SMESH_Mesh wrapper. So delete the wrapper explicitly, and do
    // it *before* gen_.reset(): ~SMESH_Mesh dereferences _document and _gen to remove
    // itself from both maps, which must still be alive. ~SMESH_Gen then runs clean (the
    // mesh already unregistered itself), with no double-free.
    if (mesh_ != nullptr) {
      delete mesh_;
      mesh_ = nullptr;
    }
    meshDS_ = nullptr;
    gen_.reset();
    // Free adopted VL hypotheses LAST: ~SMESH_Gen has now NullifyGen()'d them, so
    // ~SMESH_Hypothesis won't touch the freed gen. Deleting them earlier (while the gen and
    // mesh still hold the VL shrink state that references them) corrupts the heap.
    owned_hyps_.clear();
  }

  bool is_open() const { return mesh_ != nullptr; }

  // Internals seam for viscous.cpp (see common.hpp). All three assert the mesh is open.
  SMESH_Mesh& smesh() {
    ensure_open();
    return *mesh_;
  }
  SMESH_Gen& gen() {
    ensure_open();
    return *gen_;
  }
  std::shared_ptr<ShapeData> shape_data() const {
    if (mesh_ == nullptr) {
      throw PysmeshError("Mesh has been released");
    }
    return data_;
  }
  int next_hyp_id() const { return static_cast<int>(owned_hyps_.size()) + 1; }
  void adopt_hypothesis(SMESH_Hypothesis* hyp) { owned_hyps_.emplace_back(hyp); }

 private:
  void ensure_open() const {
    if (mesh_ == nullptr) {
      throw PysmeshError("Mesh has been released");
    }
  }

  const SMDS_MeshNode* find_node(std::int64_t node_id) const {
    const SMDS_MeshNode* node = meshDS_->FindNode(node_id);
    if (node == nullptr) {
      throw PysmeshError("Unknown node id " + std::to_string(node_id));
    }
    return node;
  }

  const SMDS_MeshElement* find_element(std::int64_t element_id) const {
    const SMDS_MeshElement* element = meshDS_->FindElement(element_id);
    if (element == nullptr) {
      throw PysmeshError("Unknown element id " + std::to_string(element_id));
    }
    return element;
  }

  static std::string join(const std::vector<std::int64_t>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) {
      s += (i ? ", " : "") + std::to_string(v[i]);
    }
    return s;
  }

  std::shared_ptr<ShapeData> data_;
  std::unique_ptr<SMESH_Gen> gen_;
  SMESH_Mesh* mesh_ = nullptr;      // owned by gen_
  SMESHDS_Mesh* meshDS_ = nullptr;  // owned by mesh_
  // VL algo/hyp created by compute_viscous_layers; freed in release() after gen_ (see there).
  std::vector<std::unique_ptr<SMESH_Hypothesis>> owned_hyps_;
};

}  // namespace

// Internals seam (declared in common.hpp) — hand viscous.cpp the SMESH objects a Mesh owns.
SMESH_Mesh& mesh_smesh(const py::object& mesh_obj) {
  return mesh_obj.cast<Mesh&>().smesh();
}
SMESH_Gen& mesh_gen(const py::object& mesh_obj) { return mesh_obj.cast<Mesh&>().gen(); }
std::shared_ptr<ShapeData> mesh_shape_data(const py::object& mesh_obj) {
  return mesh_obj.cast<Mesh&>().shape_data();
}
int mesh_next_hyp_id(const py::object& mesh_obj) {
  return mesh_obj.cast<Mesh&>().next_hyp_id();
}
void mesh_adopt_hypothesis(const py::object& mesh_obj, SMESH_Hypothesis* hyp) {
  mesh_obj.cast<Mesh&>().adopt_hypothesis(hyp);
}

void bind_mesh(py::module_& m) {
  py::class_<MeshStats>(m, "MeshStats")
      .def_readonly("n_nodes", &MeshStats::n_nodes)
      .def_readonly("n_faces", &MeshStats::n_faces)
      .def_property_readonly(
          "per_face_element_counts",
          [](const MeshStats& s) {
            py::dict d;
            for (const auto& kv : s.per_face_element_counts) {
              d[py::int_(kv.first)] = py::int_(kv.second);
            }
            return d;
          })
      .def("__repr__", [](const MeshStats& s) {
        return "<MeshStats n_nodes=" + std::to_string(s.n_nodes) +
               " n_faces=" + std::to_string(s.n_faces) + ">";
      });

  py::class_<MeshRemoval>(m, "MeshRemoval")
      .def_property_readonly(
          "elements", [](const MeshRemoval& r) { return to_array(r.elements); },
          "int64 (K,) — the element ids that are gone, ascending.")
      .def_property_readonly(
          "nodes", [](const MeshRemoval& r) { return to_array(r.nodes); },
          "int64 (J,) — the node ids that are gone, ascending.")
      .def_readonly("n_nodes_before", &MeshRemoval::n_nodes_before)
      .def_readonly("n_nodes_after", &MeshRemoval::n_nodes_after)
      .def_readonly("n_elements_before", &MeshRemoval::n_elements_before)
      .def_readonly("n_elements_after", &MeshRemoval::n_elements_after)
      .def("__repr__", [](const MeshRemoval& r) {
        return "<MeshRemoval elements=" + std::to_string(r.elements.size()) +
               " nodes=" + std::to_string(r.nodes.size()) + ">";
      });

  py::class_<Mesh>(m, "Mesh")
      .def(py::init<const py::object&>(), py::arg("shape"),
           "Create an SMESH mesh bound to the given Shape.")
      .def("add_nodes", &Mesh::add_nodes, py::arg("coords"),
           "Insert (N,3) node coords; return their SMESH ids as int64 (N,).")
      .def("classify_on_face", &Mesh::classify_on_face, py::arg("node_ids"),
           py::arg("face_id"), py::arg("uv"),
           "Classify nodes onto a face with per-node (u, v) parameters.")
      .def("classify_on_edge", &Mesh::classify_on_edge, py::arg("node_ids"),
           py::arg("edge_id"), py::arg("t"),
           "Classify nodes onto an edge with per-node curve parameter t.")
      .def("classify_on_vertex", &Mesh::classify_on_vertex, py::arg("node_id"),
           py::arg("vertex_id"), "Classify a single node onto a vertex.")
      .def("add_segments", &Mesh::add_segments, py::arg("conn"), py::arg("edge_id"),
           "Add (M,2) segments (node-id pairs) bound to the edge submesh; return their "
           "SMESH ids as int64 (M,). Required on edges bounding wall faces for "
           "viscous layers.")
      .def("add_triangles", &Mesh::add_triangles, py::arg("conn"), py::arg("face_id"),
           "Add (M,3) triangles (node-id triples) bound to the face submesh; return "
           "their SMESH ids as int64 (M,).")
      .def("remove_elements", &Mesh::remove_elements, py::arg("element_ids"),
           py::arg("free_nodes") = false,
           "Delete elements by id. Returns a MeshRemoval naming the ids that actually "
           "went; an id listed twice is removed once. Nodes stay unless free_nodes is "
           "set, which also deletes every node the removal leaves with no element on it. "
           "An unknown id refuses the whole call before anything is deleted.")
      .def("remove_nodes", &Mesh::remove_nodes, py::arg("node_ids"),
           "Delete nodes by id, and every element built on them. Returns a MeshRemoval "
           "naming both — the nodes asked for and the elements that went with them. An "
           "unknown id refuses the whole call before anything is deleted.")
      .def("validate", &Mesh::validate,
           "Raise PysmeshError unless every node is classified and every face has "
           "elements.")
      .def("stats", &Mesh::stats, "Return node/face counts and per-face element counts.")
      .def("release", &Mesh::release, "Explicitly free the underlying SMESH mesh.")
      .def("__enter__", [](Mesh& self) -> Mesh& { return self; })
      .def("__exit__",
           [](Mesh& self, const py::object&, const py::object&, const py::object&) {
             self.release();
           });
}

}  // namespace pysmesh
