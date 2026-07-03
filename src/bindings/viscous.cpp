// pySMESH binding — viscous prism layers: compute_viscous_layers.
//
// Wraps StdMeshers_ViscousLayers::Compute on an *injected* surface mesh. The upstream
// algorithm (StdMeshers_ViscousLayers.cxx, _ViscousBuilder::findSolidsWithLayers) does NOT
// take the hypothesis from a bare object: it discovers, per solid, a 3D algo whose
// GetUsedHypothesis() returns a ViscousLayers hyp, and requires every FACE sub-mesh to be
// meshed. So this binding, before calling Compute:
//   1. assigns a 3D algo (StdMeshers_Hexa_3D — the lightest compiled algo that lists
//      "ViscousLayers" as a compatible auxiliary hypothesis) and the VL hyp to each solid,
//   2. marks every face/edge/vertex sub-mesh SetIsAlwaysComputed(true) so the injected 2D
//      mesh counts as "computed",
//   3. runs Compute (GIL released), then
//   4. removes the two stack-allocated hyps from the mesh before they destruct (they hold
//      no ownership of the mesh, but the mesh's sub-meshes would otherwise keep dangling
//      pointers to them).
//
// Error text is NOT on the returned ProxyMesh::Ptr (B0 finding): after Compute, the per-
// solid sub-mesh GetComputeError() is walked — null Ptr => hard failure (raise), non-null
// Ptr + a non-OK sub-mesh error => partial failure (warnings + failed_face_ids).
//
// Shape-index hazard (B0): SetBndShapes wants SMESHDS shape *indices*, so each Python-facing
// face_id is resolved face_id -> TopoDS_Face (via the shared ShapeData) -> ShapeToIndex.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <SMDS_ElemIterator.hxx>
#include <SMDS_MeshElement.hxx>
#include <SMDS_MeshNode.hxx>
#include <SMDSAbs_ElementType.hxx>
#include <SMESHDS_GroupBase.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_Algo.hxx>
#include <SMESH_ComputeError.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_Group.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_ProxyMesh.hxx>
#include <SMESH_subMesh.hxx>
#include <StdMeshers_ViscousLayers.hxx>
#include <TopAbs.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS_Shape.hxx>

#include "common.hpp"

namespace pysmesh {
namespace {

// A minimal 3D algo whose sole purpose is to satisfy StdMeshers_ViscousLayers::Compute's
// discovery: _ViscousBuilder::findSolidsWithLayers requires the solid sub-mesh to carry a
// 3D algo whose GetUsedHypothesis() returns the VL hyp. It is never run (the binding calls
// VL.Compute directly, not SMESH_Gen::Compute), so its meshing methods are stubs. Its name
// must NOT be "Hexa_3D": VL special-cases that name as a "structured" algo (line ~2335,
// notSupportAlgos) and takes a shrink path that corrupts the heap whenever a face has no
// layers — exactly flux's normal case (farfield/symmetry faces). Any other name selects
// VL's correct unstructured shrink path. Listing "ViscousLayers" as compatible makes
// GetUsedHypothesis return the VL hyp.
class VLHostAlgo : public SMESH_3D_Algo {
 public:
  VLHostAlgo(int hyp_id, SMESH_Gen* gen) : SMESH_3D_Algo(hyp_id, gen) {
    _name = "PySMESHViscousHost";
    _compatibleHypothesis.push_back("ViscousLayers");
  }
  bool CheckHypothesis(SMESH_Mesh&, const TopoDS_Shape&,
                       SMESH_Hypothesis::Hypothesis_Status& status) override {
    status = SMESH_Hypothesis::HYP_OK;
    return true;
  }
  bool Compute(SMESH_Mesh&, const TopoDS_Shape&) override { return false; }
  bool Evaluate(SMESH_Mesh&, const TopoDS_Shape&, MapShapeNbElems&) override {
    return false;
  }
};

// Row-index map: SMESH node id -> 0-based row in node_coords/node_ids.
using Id2Row = std::unordered_map<std::int64_t, std::int32_t>;

void assert_status(SMESH_Hypothesis::Hypothesis_Status status, const char* what) {
  if (SMESH_Hypothesis::IsStatusFatal(status)) {
    throw PysmeshError(std::string(what) + " failed (hypothesis status " +
                        std::to_string(static_cast<int>(status)) + ")");
  }
}

// Wall faces = the faces layers grow on. SetBndShapes(is_ignore=false): face_ids ARE the
// walls; is_ignore=true: walls are every face NOT listed. Returns 1-based face_ids.
std::vector<int> wall_face_ids(const ShapeData& data, const std::vector<int>& face_ids,
                               bool is_ignore) {
  if (!is_ignore) {
    return face_ids;
  }
  std::vector<bool> ignored(static_cast<std::size_t>(data.faces.Extent()) + 1, false);
  for (int fid : face_ids) {
    if (fid >= 1 && fid <= data.faces.Extent()) {
      ignored[static_cast<std::size_t>(fid)] = true;
    }
  }
  std::vector<int> walls;
  for (int fid = 1; fid <= data.faces.Extent(); ++fid) {
    if (!ignored[static_cast<std::size_t>(fid)]) {
      walls.push_back(fid);
    }
  }
  return walls;
}

py::dict compute_viscous_layers(const py::object& mesh_obj, const std::vector<int>& face_ids,
                                bool is_ignore, double total_thickness, int n_layers,
                                double stretch_factor, int method,
                                const std::string& group_name) {
  SMESH_Mesh& mesh = mesh_smesh(mesh_obj);
  SMESH_Gen& gen = mesh_gen(mesh_obj);
  const std::shared_ptr<ShapeData> data = mesh_shape_data(mesh_obj);
  SMESHDS_Mesh* meshDS = mesh.GetMeshDS();
  const TopoDS_Shape& shape = data->shape;

  if (group_name.empty()) {
    throw PysmeshError("group_name must be non-empty; prism harvest relies on the "
                        "ViscousLayers element group.");
  }
  // One compute per Mesh: a second run would stack a second algo/VL hyp on the solid
  // (multiple algos on one sub-mesh is an error state). Create a fresh Mesh for another run.
  if (mesh_next_hyp_id(mesh_obj) != 1) {
    throw PysmeshError("compute_viscous_layers may be called only once per Mesh; build a "
                        "new Mesh for another run.");
  }
  {
    TopExp_Explorer solids(shape, TopAbs_SOLID);
    if (!solids.More()) {
      throw PysmeshError("compute_viscous_layers requires a shape with at least one solid; "
                          "the loaded shape has none.");
    }
  }

  // Resolve the wall set up front and always pass it explicitly with toIgnore=false.
  // StdMeshers_ViscousLayers's toIgnore=true path corrupts the heap on this SMESH version
  // (a coarse box with one non-wall face crashes, while the identical wall set passed
  // explicitly with toIgnore=false is fine); normalizing here sidesteps that path entirely
  // and keeps the Python is_ignore semantics intact.
  const std::vector<int> walls = wall_face_ids(*data, face_ids, is_ignore);

  // Shape-index-hazard translation: face_id -> TopoDS_Face -> SMESHDS index.
  std::vector<int> shape_ids;
  shape_ids.reserve(walls.size());
  for (int fid : walls) {
    shape_ids.push_back(meshDS->ShapeToIndex(data->face(fid)));  // data->face validates fid
  }

  // Hypotheses: a 3D algo (never run — only satisfies GetAlgo()/GetUsedHypothesis) + VL.
  // Heap-allocated and handed to the Mesh, which frees them at release() after its gen is
  // gone. They MUST outlive this call: the shrink path registers mesh-side state that
  // references the hyps, so freeing them here would corrupt the heap.
  const int algo_id = mesh_next_hyp_id(mesh_obj);
  VLHostAlgo* algo = new VLHostAlgo(algo_id, &gen);
  mesh_adopt_hypothesis(mesh_obj, algo);
  const int vl_id = mesh_next_hyp_id(mesh_obj);
  StdMeshers_ViscousLayers* vl = new StdMeshers_ViscousLayers(vl_id, &gen);
  mesh_adopt_hypothesis(mesh_obj, vl);

  vl->SetTotalThickness(total_thickness);
  vl->SetNumberLayers(n_layers);
  vl->SetStretchFactor(stretch_factor);
  vl->SetMethod(static_cast<StdMeshers_ViscousLayers::ExtrusionMethod>(method));
  vl->SetGroupName(group_name);
  vl->SetBndShapes(shape_ids, /*toIgnore=*/false);  // walls listed explicitly (see above)

  for (TopExp_Explorer solids(shape, TopAbs_SOLID); solids.More(); solids.Next()) {
    assert_status(mesh.AddHypothesis(solids.Current(), algo_id), "assigning 3D algorithm");
    assert_status(mesh.AddHypothesis(solids.Current(), vl_id),
                  "assigning ViscousLayers hypothesis");
  }

  // The injected 2D mesh is authoritative: mark every sub-mesh computed AFTER AddHypothesis
  // (whose notifications would otherwise clear the flag) so findSolidsWithLayers accepts it.
  for (int i = 1; i <= data->faces.Extent(); ++i) {
    mesh.GetSubMesh(data->faces.FindKey(i))->SetIsAlwaysComputed(true);
  }
  for (int i = 1; i <= data->edges.Extent(); ++i) {
    mesh.GetSubMesh(data->edges.FindKey(i))->SetIsAlwaysComputed(true);
  }
  for (int i = 1; i <= data->vertices.Extent(); ++i) {
    mesh.GetSubMesh(data->vertices.FindKey(i))->SetIsAlwaysComputed(true);
  }

  SMESH_ProxyMesh::Ptr proxy;
  {
    py::gil_scoped_release release;
    proxy = vl->Compute(mesh, shape, /*toMakeN2NMap=*/false);
  }

  // Per-solid error walk (text is on the sub-mesh, not on the returned Ptr).
  std::vector<std::string> solid_errors;
  for (TopExp_Explorer solids(shape, TopAbs_SOLID); solids.More(); solids.Next()) {
    const SMESH_ComputeErrorPtr err = mesh.GetSubMesh(solids.Current())->GetComputeError();
    if (err && !err->IsOK() && !err->myComment.empty()) {
      solid_errors.push_back(err->myComment);
    }
  }

  auto join = [](const std::vector<std::string>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) {
      s += (i ? "; " : "") + v[i];
    }
    return s;
  };

  if (!proxy) {
    std::string details = join(solid_errors);
    if (details.empty()) {
      details = "ViscousLayers Compute() returned no proxy mesh and no per-solid error text.";
    }
    throw PysmeshError("Viscous-layer computation failed.", details);
  }

  // ---- Harvest -------------------------------------------------------------------- //
  // Node table (all post-compute nodes): row order = nodesIterator order.
  const std::int64_t n_nodes = static_cast<std::int64_t>(meshDS->NbNodes());
  py::array_t<double> node_coords({static_cast<py::ssize_t>(n_nodes), py::ssize_t(3)});
  py::array_t<std::int64_t> node_ids(static_cast<py::ssize_t>(n_nodes));
  double* coord = node_coords.mutable_data();
  std::int64_t* nid = node_ids.mutable_data();
  Id2Row id2row;
  id2row.reserve(static_cast<std::size_t>(n_nodes) * 2);
  {
    std::int32_t row = 0;
    for (SMDS_NodeIteratorPtr it = meshDS->nodesIterator(); it->more(); ++row) {
      const SMDS_MeshNode* node = it->next();
      coord[3 * row] = node->X();
      coord[3 * row + 1] = node->Y();
      coord[3 * row + 2] = node->Z();
      const std::int64_t id = static_cast<std::int64_t>(node->GetID());
      nid[row] = id;
      id2row.emplace(id, row);
    }
  }

  auto row_of = [&id2row](const SMDS_MeshNode* node) -> std::int32_t {
    const auto it = id2row.find(static_cast<std::int64_t>(node->GetID()));
    if (it == id2row.end()) {
      throw PysmeshError("Harvested element references a node absent from the mesh.");
    }
    return it->second;
  };

  // Prisms via the named volume group. Node order == SMESH penta order == VTK wedge order.
  std::vector<std::int32_t> prism_rows;
  {
    SMESH_Mesh::GroupIteratorPtr git = mesh.GetGroups();
    const SMESHDS_GroupBase* grp = nullptr;
    while (git->more()) {
      SMESH_Group* g = git->next();
      if (g && g->GetGroupDS() && g->GetGroupDS()->GetType() == SMDSAbs_Volume &&
          group_name == g->GetName()) {
        grp = g->GetGroupDS();
        break;
      }
    }
    if (grp != nullptr) {
      for (SMDS_ElemIteratorPtr it = grp->GetElements(); it->more();) {
        const SMDS_MeshElement* elem = it->next();
        if (elem->NbNodes() != 6) {
          continue;  // only 6-node prisms are wedge-mappable; skip degenerate layer cells
        }
        // SMESH penta nodes are the two triangular caps (0,1,2) / (3,4,5) with lateral
        // edges 0-3, 1-4, 2-5, but wound so the caps' CCW normal points out of the layer
        // (inward-grown prisms) — i.e. VTK_WEDGE with reversed caps, giving a negative VTK
        // Jacobian. Reorder to {0,2,1,3,5,4} so the emitted cell is a positive VTK_WEDGE.
        static constexpr int WEDGE[6] = {0, 2, 1, 3, 5, 4};
        for (const int i : WEDGE) {
          prism_rows.push_back(row_of(elem->GetNode(i)));
        }
      }
    }
  }
  const py::ssize_t n_prisms = static_cast<py::ssize_t>(prism_rows.size() / 6);
  py::array_t<std::int32_t> prism_connectivity({n_prisms, py::ssize_t(6)});
  std::copy(prism_rows.begin(), prism_rows.end(), prism_connectivity.mutable_data());

  // Inner (shrunk) surface: proxy sub-mesh triangles per wall face; null => failed face.
  std::vector<std::int32_t> inner_rows;
  std::vector<std::int32_t> inner_face_map;
  std::vector<int> failed_faces;
  for (int fid : walls) {
    const SMESH_ProxyMesh::SubMesh* psm = proxy->GetProxySubMesh(data->face(fid));
    if (psm == nullptr) {
      failed_faces.push_back(fid);
      continue;
    }
    for (SMDS_ElemIteratorPtr it = psm->GetElements(); it->more();) {
      const SMDS_MeshElement* tri = it->next();
      if (tri->NbNodes() != 3) {
        continue;
      }
      inner_rows.push_back(row_of(tri->GetNode(0)));
      inner_rows.push_back(row_of(tri->GetNode(1)));
      inner_rows.push_back(row_of(tri->GetNode(2)));
      inner_face_map.push_back(fid);
    }
  }
  const py::ssize_t n_tris = static_cast<py::ssize_t>(inner_face_map.size());
  py::array_t<std::int32_t> inner_surface_tris({n_tris, py::ssize_t(3)});
  std::copy(inner_rows.begin(), inner_rows.end(), inner_surface_tris.mutable_data());
  py::array_t<std::int32_t> inner_surface_face_map(n_tris);
  std::copy(inner_face_map.begin(), inner_face_map.end(),
            inner_surface_face_map.mutable_data());

  // The hyps stay assigned and registered; the Mesh owns them and frees them at release()
  // (do not RemoveHypothesis / free here — see the allocation note above).

  py::dict out;
  out["prism_connectivity"] = prism_connectivity;
  out["node_coords"] = node_coords;
  out["node_ids"] = node_ids;
  out["inner_surface_tris"] = inner_surface_tris;
  out["inner_surface_face_map"] = inner_surface_face_map;
  out["failed_face_ids"] = py::cast(failed_faces);
  out["warnings"] = py::cast(solid_errors);
  return out;
}

}  // namespace

void bind_viscous(py::module_& m) {
  m.def("compute_viscous_layers", &compute_viscous_layers, py::arg("mesh"),
        py::arg("face_ids"), py::arg("is_ignore"), py::arg("total_thickness"),
        py::arg("n_layers"), py::arg("stretch_factor"), py::arg("method"),
        py::arg("group_name"),
        "Compute viscous prism layers on an injected surface mesh. Returns a dict of "
        "NumPy arrays (row-indexed connectivity) + failed_face_ids + warnings. Low-level: "
        "the pysmesh.compute_viscous_layers wrapper builds VLParams/VLResult around it.");
}

}  // namespace pysmesh
