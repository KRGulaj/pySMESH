// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — the mesher's ownership, assignment model, compute and error reporting.
//
// See mesher/mesher.hpp for the file split and for the three hazards this code exists to
// keep away from the caller.

#include "mesher/mesher.hpp"

#include "compute_thread.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <list>
#include <map>
#include <set>
#include <utility>

#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <GeomLProp_SLProps.hxx>
#include <Geom_Surface.hxx>
#include <Precision.hxx>
#include <SMDS_ElemIterator.hxx>
#include <SMDS_MeshNode.hxx>
#include <SMDS_MeshElement.hxx>
#include <SMDS_VolumeTool.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESHDS_SubMesh.hxx>
#include <SMESH_Algo.hxx>
#include <SMESH_ComputeError.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_HypoFilter.hxx>
#include <SMESH_Hypothesis.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_subMesh.hxx>
#include <StdMeshers_ViscousLayers.hxx>
#include <StdMeshers_ViscousLayers2D.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <gp_Dir.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt.hxx>
#include <gp_XYZ.hxx>

namespace pysmesh {

// Defined in shape.cpp — the Mesher shares the ShapeData its Shape argument holds, so a
// Python-facing ordinal always resolves to the same TopoDS_* object the caller queried.
std::shared_ptr<ShapeData> shape_data_of(const py::object& shape_obj);

namespace mesher {
namespace {

// The four kinds a caller can name. A sub-shape of any other kind — a WIRE, a SHELL, a
// COMPOUND — carries no algorithm of its own in SMESH's model and has no ordinal in the
// caller's Shape, so naming one is a caller error rather than a silent no-op.
constexpr const char* kKindNames[] = {"SOLID", "FACE", "EDGE", "VERTEX"};
constexpr TopAbs_ShapeEnum kKindTypes[] = {TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE,
                                          TopAbs_VERTEX};

const char* kind_name_of(TopAbs_ShapeEnum type) {
  for (std::size_t i = 0; i < 4; ++i) {
    if (kKindTypes[i] == type) {
      return kKindNames[i];
    }
  }
  return "";
}

// SMESH's own words for a refused assignment. AddHypothesis takes a std::string* error
// out-parameter, but it comes back EMPTY even on a fatal status (measured: a second 1-D
// algorithm on one shape gives HYP_ALREADY_EXIST with no text at all), so the status enum is
// the only channel there is and it has to be spelled out here.
const char* status_text(SMESH_Hypothesis::Hypothesis_Status status) {
  switch (status) {
    case SMESH_Hypothesis::HYP_OK:
      return "accepted";
    case SMESH_Hypothesis::HYP_MISSING:
      return "the algorithm is missing a hypothesis it needs";
    case SMESH_Hypothesis::HYP_CONCURRENT:
      return "several applicable hypotheses are assigned to enclosing sub-shapes";
    case SMESH_Hypothesis::HYP_BAD_PARAMETER:
      return "the hypothesis carries a bad parameter value";
    case SMESH_Hypothesis::HYP_HIDDEN_ALGO:
      return "this algorithm is hidden by a higher-dimension one that meshes all dimensions";
    case SMESH_Hypothesis::HYP_HIDING_ALGO:
      return "this algorithm meshes all dimensions and hides the lower-dimension ones";
    case SMESH_Hypothesis::HYP_INCOMPATIBLE:
      return "the hypothesis does not fit the algorithm it was assigned beside";
    case SMESH_Hypothesis::HYP_NOTCONFORM:
      return "the hypothesis would produce a non-conforming mesh";
    case SMESH_Hypothesis::HYP_ALREADY_EXIST:
      return "another algorithm or hypothesis of the same priority is already assigned there";
    case SMESH_Hypothesis::HYP_BAD_DIM:
      return "the dimension does not match the sub-shape";
    case SMESH_Hypothesis::HYP_BAD_SUBSHAPE:
      return "the sub-shape is not part of the meshed shape";
    case SMESH_Hypothesis::HYP_BAD_GEOMETRY:
      return "the sub-shape's geometry is not what the algorithm expects";
    case SMESH_Hypothesis::HYP_NEED_SHAPE:
      return "the algorithm works on a shape only";
    case SMESH_Hypothesis::HYP_INCOMPAT_HYPS:
      return "the additional hypotheses assigned there are incompatible with one another";
    default:
      return "refused for an unknown reason";
  }
}

// Whether `shape` needs an algorithm of its own: an algorithm of an enclosing sub-shape
// needs the mesh of `shape` as its boundary (NeedDiscreteBoundary()), and no enclosing
// algorithm meshes all dimensions itself, as Cartesian_3D does. Otherwise a NO_ALGO state is
// not a fault: a solid with no 3-D algorithm under a surface mesh, or a face under an
// all-dimensional one.
bool needs_own_algorithm(SMESH_Mesh& mesh, const TopoDS_Shape& shape) {
  bool needed = false;
  for (const TopoDS_Shape& above : mesh.GetAncestors(shape)) {
    SMESH_subMesh* sub = mesh.GetSubMeshContaining(above);
    const SMESH_Algo* algo = sub != nullptr ? sub->GetAlgo() : nullptr;
    if (algo == nullptr) {
      continue;
    }
    if (!algo->NeedDiscreteBoundary()) {
      return false;
    }
    needed = true;
  }
  return needed;
}

std::string where(const std::string& kind, int ordinal) {
  if (kind.empty()) {
    return "the whole shape";
  }
  return kind + " " + std::to_string(ordinal);
}

// Resolves the entry strings that a hypothesis stores instead of shapes. Upstream they are
// study entries resolved by the CORBA layer; here an entry is "KIND:ordinal", built by the
// catalogue from the caller's ordinals (BlockRenumber's explicit form, report W1.6). The
// other callbacks are hooks of the SALOME study, which a Mesher does not have: they do
// nothing, and IsLoaded() is true, so SMESH_Mesh::NotifySubMeshesHypothesisModification
// skips its reload branch (SMESH_Mesh.cxx:1276). SMESH_Mesh owns and deletes the object.
class EntryCallUp : public SMESH_Mesh::TCallUp {
 public:
  explicit EntryCallUp(std::shared_ptr<ShapeData> data) : data_(std::move(data)) {}

  void RemoveGroup(const int) override {}
  void HypothesisModified(int, bool) override {}
  void Load() override {}
  bool IsLoaded() override { return true; }

  // A null shape for an entry that names nothing, which upstream reports as a bad
  // parameter (StdMeshers_BlockRenumber::CheckHypothesis). Never throws into SMESH.
  TopoDS_Shape GetShapeByEntry(const std::string& entry) override {
    const std::size_t colon = entry.find(':');
    if (colon == std::string::npos) {
      return TopoDS_Shape();
    }
    const std::string kind = entry.substr(0, colon);
    try {
      const int ordinal = std::stoi(entry.substr(colon + 1));
      if (kind == "SOLID") return data_->solid(ordinal);
      if (kind == "FACE") return data_->face(ordinal);
      if (kind == "EDGE") return data_->edge(ordinal);
      if (kind == "VERTEX") return data_->vertex(ordinal);
    } catch (const std::exception&) {
      return TopoDS_Shape();
    }
    return TopoDS_Shape();
  }

 private:
  std::shared_ptr<ShapeData> data_;
};

// The elements a compute error names as bad, as text for its line, with each element's
// centroid appended to `centroids` (x y z per element, model coordinates).
//
// NETGENPlugin_Mesher::ReadErrors stores the triangles NETGEN reports as intersecting, and
// the edges it finds several times in a surface mesh, as SMESH_BadInputElements of the
// failed sub-mesh. Each is a free element over mesh nodes, so the mean of its nodes is a
// point in the model where the failure is. NETGEN reports one pair of triangles once per
// front element that meets it, so an element over the same nodes as an earlier one is
// skipped. The text gives the count and the centroids of the first kListed; an error
// with no bad element gives an empty text.
std::string bad_elements_text(const SMESH_ComputeError& err, std::vector<double>& centroids) {
  constexpr std::size_t kListed = 5;
  const auto* bad = dynamic_cast<const SMESH_BadInputElements*>(&err);
  if (bad == nullptr || !bad->HasBadElems()) {
    return std::string();
  }
  std::size_t count = 0;
  std::string listed;
  std::set<std::vector<smIdType>> seen;
  for (const SMDS_MeshElement* elem : bad->myBadElements) {
    const int n = elem != nullptr ? elem->NbNodes() : 0;
    if (n == 0) {
      continue;
    }
    std::vector<smIdType> key;
    for (int i = 0; i < n; ++i) {
      key.push_back(elem->GetNode(i)->GetID());
    }
    std::sort(key.begin(), key.end());
    if (!seen.insert(key).second) {
      continue;
    }
    double c[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < n; ++i) {
      const SMDS_MeshNode* node = elem->GetNode(i);
      c[0] += node->X();
      c[1] += node->Y();
      c[2] += node->Z();
    }
    for (double& v : c) {
      v /= n;
    }
    centroids.insert(centroids.end(), c, c + 3);
    if (count < kListed) {
      char buf[96];
      std::snprintf(buf, sizeof(buf), "(%.6g, %.6g, %.6g)", c[0], c[1], c[2]);
      listed += std::string(count == 0 ? "" : ", ") + buf;
    }
    ++count;
  }
  if (count == 0) {
    return std::string();
  }
  return "; " + std::to_string(count) + " bad element(s), centroid(s) of the first " +
         std::to_string(std::min(count, kListed)) + ": " + listed;
}

}  // namespace

// ---- Shared value helpers -------------------------------------------------------------- //

SMDSAbs_ElementType family_of(int code) {
  if (code < 0 || code > static_cast<int>(SMDSAbs_Ball)) {
    throw PysmeshError("Unknown element family " + std::to_string(code) +
                       " (expected one of ALL, NODE, EDGE, FACE, VOLUME, ELEM_0D, BALL).");
  }
  return static_cast<SMDSAbs_ElementType>(code);
}

py::array_t<double, py::array::c_style | py::array::forcecast> point_table(
    const py::object& values, const char* name, int columns) {
  py::array_t<double, py::array::c_style | py::array::forcecast> table =
      values.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  if (table.ndim() != 2 || table.shape(1) != columns) {
    throw PysmeshError(std::string(name) + " must have shape (N, " + std::to_string(columns) +
                       ").");
  }
  return table;
}

void require_triple(const std::vector<double>& values, const char* name) {
  if (values.size() != 3) {
    throw PysmeshError(std::string(name) + " must be three numbers (got " +
                       std::to_string(values.size()) + ").");
  }
}

// ---- Params -------------------------------------------------------------------------- //

py::object Params::take(const char* key) {
  if (!values_.contains(key)) {
    throw PysmeshError(std::string(owner_) + " is missing the parameter '" + key + "'.");
  }
  consumed_.emplace_back(key);
  return values_[key];
}

bool Params::has(const char* key) const { return values_.contains(key); }

// A NaN or an infinity in a numeric field is refused here, for every hypothesis and every
// quality or selection parameter, before any setter passes it to SMESH (report F1).
double Params::number(const char* key) {
  const double v = take(key).cast<double>();
  require_finite(std::string("Mesher: ") + owner_, key, v);
  return v;
}

int Params::integer(const char* key) { return take(key).cast<int>(); }
bool Params::flag(const char* key) { return take(key).cast<bool>(); }
std::string Params::text(const char* key) { return take(key).cast<std::string>(); }

std::vector<double> Params::numbers(const char* key) {
  std::vector<double> v = take(key).cast<std::vector<double>>();
  require_finite(std::string("Mesher: ") + owner_, key, v.data(), v.size(), 1);
  return v;
}

std::vector<int> Params::integers(const char* key) {
  return take(key).cast<std::vector<int>>();
}

std::vector<std::vector<int>> Params::integer_rows(const char* key) {
  return take(key).cast<std::vector<std::vector<int>>>();
}

std::vector<std::vector<double>> Params::number_rows(const char* key) {
  const auto rows = take(key).cast<std::vector<std::vector<double>>>();
  for (const std::vector<double>& row : rows) {
    for (const double v : row) {
      require_finite(std::string("Mesher: ") + owner_, key, v);
    }
  }
  return rows;
}

std::vector<std::int64_t> Params::ids(const char* key) {
  return take(key).cast<std::vector<std::int64_t>>();
}

std::pair<std::string, int> Params::subshape(const char* key) {
  const py::tuple pair = take(key).cast<py::tuple>();
  if (pair.size() != 2) {
    throw PysmeshError(std::string(owner_) + ": '" + key +
                       "' must be a (kind, ordinal) pair naming one sub-shape.");
  }
  return {pair[0].cast<std::string>(), pair[1].cast<int>()};
}

py::dict Params::nested(const char* key) { return take(key).cast<py::dict>(); }

py::list Params::list(const char* key) { return take(key).cast<py::list>(); }

void Params::done() const {
  std::vector<std::string> extra;
  for (const auto& item : values_) {
    const std::string key = item.first.cast<std::string>();
    bool used = false;
    for (const std::string& seen : consumed_) {
      used = used || seen == key;
    }
    if (!used) {
      extra.push_back(key);
    }
  }
  if (extra.empty()) {
    return;
  }
  std::string names;
  for (std::size_t i = 0; i < extra.size(); ++i) {
    names += (i ? ", " : "") + extra[i];
  }
  throw PysmeshError(std::string(owner_) + " does not take the parameter(s): " + names + ".");
}

// ---- ComputeDriver --------------------------------------------------------------------//

ComputeDriver::ComputeDriver(SMESH_Mesh& mesh, SMESH_Gen& gen, const TopoDS_Shape& shape,
                             const ProgressHooks& hooks)
    : mesh_(mesh), gen_(gen), shape_(shape), hooks_(hooks) {
  if (!hooks_.active()) {
    return;
  }
  if (!(hooks_.interval_s > 0.0)) {
    throw PysmeshError("Mesher.compute: the progress poll interval must be > 0 s (got " +
                       std::to_string(hooks_.interval_s) + ").");
  }
  // Ask once, synchronously, before anything starts — the same floor the OCCT-side driver
  // closes. A mesh that finishes inside one poll interval would otherwise run to completion
  // however emphatically a caller's pre-set flag said no.
  if (!hooks_.should_cancel.is_none() && hooks_.should_cancel().cast<bool>()) {
    cancelled_.store(true, std::memory_order_relaxed);
    return;
  }
  worker_ = std::thread([this] { poll(); });
}

ComputeDriver::~ComputeDriver() {
  stop_thread();
  if (hook_error_) {
    py::gil_scoped_acquire acquire;
    hook_error_ = nullptr;
  }
}

void ComputeDriver::request_cancel() {
  cancelled_.store(true, std::memory_order_relaxed);
  gen_.CancelCompute(mesh_, shape_);
}

void ComputeDriver::finish() {
  if (finished_) {
    return;
  }
  finished_ = true;
  stop_thread();

  if (hook_error_) {
    const std::exception_ptr raised = hook_error_;
    hook_error_ = nullptr;
    std::rethrow_exception(raised);
  }

  // A bar has to reach the end, and the poller cannot deliver the last value because the
  // mesher finishes between two ticks. Only for a run that completed: reporting 1.0 for a
  // cancelled one would be a lie.
  if (cancelled() || hooks_.on_progress.is_none()) {
    return;
  }
  if (last_reported_ < 1.0) {
    last_reported_ = 1.0;
    hooks_.on_progress(1.0);
  }
}

void ComputeDriver::stop_thread() {
  if (!worker_.joinable()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  // The poller may be blocked acquiring the GIL for one last tick, so joining while holding
  // it would deadlock the two against each other.
  if (PyGILState_Check()) {
    py::gil_scoped_release release;
    worker_.join();
  } else {
    worker_.join();
  }
}

void ComputeDriver::poll() {
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait_for(lock, std::chrono::duration<double>(hooks_.interval_s),
                     [this] { return stop_; });
      if (stop_) {
        return;
      }
    }

    // Read the position outside the GIL: GetComputeProgress walks SMESH's own structures and
    // has nothing to do with Python. It is measured safe to call while Compute() runs.
    const double position = mesh_.GetComputeProgress();

    py::gil_scoped_acquire acquire;
    try {
      // Report first, then ask, so a caller cancelling in response to a value has seen it.
      if (!hooks_.on_progress.is_none() && position > last_reported_ && position < 1.0) {
        last_reported_ = position;
        hooks_.on_progress(position);
      }
      if (!hooks_.should_cancel.is_none() && hooks_.should_cancel().cast<bool>()) {
        request_cancel();
        return;
      }
    } catch (...) {
      // A hook that raises is a cancel: the run stops, nothing is returned, and the
      // exception reaches the caller from finish() with its own type and traceback.
      hook_error_ = std::current_exception();
      request_cancel();
      return;
    }
  }
}

// ---- Mesher ---------------------------------------------------------------------------//

Mesher::Mesher(const py::object& shape_obj) {
  gen_ = std::make_unique<SMESH_Gen>();
  mesh_ = gen_->CreateMesh(false);  // owned by gen_; freed in release() before it
  // None builds a mesh with no geometry behind it. ShapeToMesh() is what sets SMESH's own
  // _isShapeToMesh flag, so *not* calling it is the whole of the difference — the mesh is
  // otherwise the same object, and everything written against SMESHDS works on it unchanged.
  if (!shape_obj.is_none()) {
    data_ = shape_data_of(shape_obj);
    mesh_->ShapeToMesh(data_->shape);
    mesh_->SetCallUp(new EntryCallUp(data_));
  }
  meshDS_ = mesh_->GetMeshDS();
  if (data_ != nullptr) {
    build_index_map();
  }
}

Mesher::~Mesher() { release(); }

void Mesher::release() {
  // The one teardown order that does not corrupt the heap, established by mesh.cpp:
  // ~SMESH_Gen deletes the document and NullifyGen()s the hypotheses but never deletes the
  // SMESH_Mesh wrapper, and ~SMESH_Mesh dereferences both _document and _gen to unregister
  // itself. So the wrapper goes first, the generator second, and the hypotheses last — by
  // then they no longer point at a live generator.
  if (mesh_ != nullptr) {
    delete mesh_;
    mesh_ = nullptr;
  }
  meshDS_ = nullptr;
  gen_.reset();
  owned_.clear();
  assigned_.clear();
}

void Mesher::ensure_open() const {
  if (mesh_ == nullptr) {
    throw PysmeshError("Mesher has been released.");
  }
}

void Mesher::refuse_without_shape(const char* op, const std::string& name,
                                  const std::string& kind) const {
  if (!works_without_shape(name)) {
    throw PysmeshError(std::string(op) + ": this mesher has no shape, so '" + name +
                           "' has nothing to run on.",
                       "A mesher built from arrays takes only the NETGEN remesher "
                       "(NetgenRemesher2D with NetgenRemesherParameters2D), which meshes "
                       "the mesh's own triangles again.");
  }
  throw PysmeshError(std::string(op) + ": this mesher has no shape, so '" + name +
                     "' goes on the whole mesh (on=None), not on " + kind + ".");
}

void Mesher::ensure_shape(const char* op) const {
  if (data_ != nullptr) {
    return;
  }
  throw PysmeshError(std::string(op) + ": this mesher has no shape.",
                     "It was built from arrays rather than from geometry, so it has no "
                     "sub-shape ordinals to name and nothing for an algorithm to run on. "
                     "Everything that works on the mesh alone — the editor, the search "
                     "surface, the groups by id or by filter — works here; only the "
                     "operations that resolve a sub-shape do not.");
}

SMESH_Mesh& Mesher::smesh() const {
  ensure_open();
  return *mesh_;
}

SMESH_Gen& Mesher::gen() const {
  ensure_open();
  return *gen_;
}

SMESHDS_Mesh& Mesher::meshDS() const {
  ensure_open();
  return *meshDS_;
}

const ShapeData& Mesher::shape_data() const {
  ensure_open();
  ensure_shape("Mesher.shape_data");
  return *data_;
}

const TopoDS_Shape& Mesher::sub_shape(const std::string& kind, int ordinal) const {
  ensure_open();
  ensure_shape("Mesher.sub_shape");
  if (kind.empty()) {
    return data_->shape;
  }
  if (kind == "SOLID") {
    return data_->solid(ordinal);
  }
  if (kind == "FACE") {
    return data_->face(ordinal);
  }
  if (kind == "EDGE") {
    return data_->edge(ordinal);
  }
  if (kind == "VERTEX") {
    return data_->vertex(ordinal);
  }
  throw PysmeshError("Unknown sub-shape kind '" + kind +
                     "' (expected SOLID, FACE, EDGE or VERTEX).");
}

void Mesher::build_index_map() {
  // SMESHDS indexes every sub-shape of every kind in one unfiltered sequence, so its index
  // is not a per-kind ordinal and must never leave this file. Inverting it once here means a
  // harvest of a million elements costs one array lookup each instead of a shape-map probe.
  const int extent = meshDS_->MaxShapeIndex();
  index_to_ordinal_.assign(static_cast<std::size_t>(extent) + 1, {"", 0});
  for (std::size_t k = 0; k < 4; ++k) {
    const TopTools_IndexedMapOfShape* map = nullptr;
    switch (kKindTypes[k]) {
      case TopAbs_SOLID:
        map = &data_->solids;
        break;
      case TopAbs_FACE:
        map = &data_->faces;
        break;
      case TopAbs_EDGE:
        map = &data_->edges;
        break;
      default:
        map = &data_->vertices;
        break;
    }
    for (int i = 1; i <= map->Extent(); ++i) {
      const int index = meshDS_->ShapeToIndex(map->FindKey(i));
      if (index > 0 && index <= extent) {
        index_to_ordinal_[static_cast<std::size_t>(index)] = {kKindNames[k], i};
      }
    }
  }
}

std::string Mesher::describe_concurrency(const TopoDS_Shape& target,
                                         SMESH_Hypothesis* hyp) const {
  // The same search as SMESH_subMesh::CheckConcurrentHypothesis: a sub-shape with no
  // similar hypothesis of its own, whose nearest ancestors of one type carry different
  // similar hypotheses. Similar: the same type and dimension, not `hyp` itself, and for
  // an auxiliary hypothesis the same name (getSimilarAttached, SMESH_subMesh.cxx:2227).
  SMESH_HypoFilter similar(SMESH_HypoFilter::HasType(hyp->GetType()));
  similar.And(SMESH_HypoFilter::HasDim(hyp->GetDim()));
  similar.AndNot(SMESH_HypoFilter::Is(hyp));
  if (hyp->IsAuxiliary()) {
    similar.And(SMESH_HypoFilter::HasName(hyp->GetName()));
  } else {
    similar.AndNot(SMESH_HypoFilter::IsAuxiliary());
  }
  auto name_of = [this](const TopoDS_Shape& s) {
    const std::pair<const char*, int> at = ordinal_of_shape_index(meshDS_->ShapeToIndex(s));
    return std::string(at.first[0] != 0 ? at.first : "sub-shape") + " " +
           std::to_string(at.second);
  };
  SMESH_subMesh* sub = mesh_->GetSubMesh(target);
  for (SMESH_subMeshIteratorPtr it = sub->getDependsOnIterator(false, false); it->more();) {
    SMESH_subMesh* sm = it->next();
    if (!sm->IsApplicableHypothesis(hyp) ||
        sm->CheckConcurrentHypothesis(hyp) != SMESH_Hypothesis::HYP_CONCURRENT) {
      continue;
    }
    const TopoDS_Shape& shared = sm->GetSubShape();
    std::string owners;
    std::string kinds;
    TopAbs_ShapeEnum level = TopAbs_SHAPE;
    for (const TopoDS_Shape& ancestor : mesh_->GetAncestors(shared)) {
      const SMESH_Hypothesis* found = mesh_->GetHypothesis(ancestor, similar, false);
      if (found == nullptr) {
        continue;
      }
      if (level == TopAbs_SHAPE) {
        level = ancestor.ShapeType();
      } else if (ancestor.ShapeType() != level) {
        break;
      }
      owners += (owners.empty() ? "" : " and ") + name_of(ancestor) + " (" +
                found->GetName() + ")";
      kinds = found->GetName();
    }
    return name_of(shared) + " lies on " + owners +
           ", which carry different hypotheses, so which of them meshes it is "
           "undefined. Assigning '" + hyp->GetName() + "' made SMESH check the sub-shapes "
           "it governs and find it. Assign one " + kinds + " on " + name_of(shared) +
           " itself first: a hypothesis on the sub-shape takes priority over those on the "
           "shapes around it.";
  }
  return "SMESH reported HYP_CONCURRENT, but no sub-shape with two different similar "
         "hypotheses on its ancestors was found under the assigned shape.";
}

void Mesher::refuse_unread_layers() const {
  // Only some algorithms build layers in their Compute: Hexa_3D, CompositeHexa_3D
  // (StdMeshers_CompositeHexa_3D_viscous_layers.patch), PolyhedronPerSolid_3D,
  // Cartesian_3D, NETGEN_3D and NETGEN_2D3D read ViscousLayers; Quadrangle_2D,
  // QuadFromMedialAxis_1D2D, MEFISTO_2D, PolygonPerFace_2D
  // (StdMeshers_PolygonPerFace_2D_viscous_layers.patch), NETGEN_2D and NETGEN_2D_ONLY
  // read ViscousLayers2D. The NETGEN algorithms take several ViscousLayers per solid
  // (NETGENPlugin_NETGEN_3D.cxx, CheckHypothesis). The compatible lists do not tell: RadialQuadrangle_1D2D inherits
  // ViscousLayers2D from Quadrangle_2D and builds no layer. Any other algorithm meshes the
  // sub-shape with no layer and no word (Prism_3D, RadialQuadrangle_1D2D).
  //
  // A building algorithm can still refuse the layer hypotheses that reach it: Hexa_3D
  // (StdMeshers_Hexa_3D.cxx:136-147) and CompositeHexa_3D take one ViscousLayers per solid,
  // Cartesian_3D keeps the last one it lists (StdMeshers_Cartesian_3D.cxx:114-125), and the
  // layer check of StdMeshers_ViscousLayers::CheckHypothesis refuses face sets that do not
  // fit together.
  // SMESH then leaves the sub-mesh MISSING_HYP, or drops the other hypotheses, and its
  // Compute returns true with no element there (SMESH_Gen.cxx:249-253). So the layer check
  // runs here first, with SMESH's reason; the algorithm runs it again in its own check
  // before its Compute (SMESH_subMesh.cxx, COMPUTE), so this call changes no result.
  struct LayerKind {
    TopAbs_ShapeEnum type;
    const char* kind_name;
    const char* hypothesis;
    std::set<std::string> builders;
    const char* listed;
    std::set<std::string> read_one;  // builders that read one hypothesis per sub-shape
  };
  const LayerKind kinds[] = {
      {TopAbs_SOLID, "SOLID", "ViscousLayers",
       {"Hexa_3D", "CompositeHexa_3D", "PolyhedronPerSolid_3D", "Cartesian_3D", "NETGEN_3D",
        "NETGEN_2D3D"},
       "Hexa_3D, CompositeHexa_3D, PolyhedronPerSolid_3D, Cartesian_3D, NETGEN_3D and "
       "NETGEN_2D3D",
       {"Hexa_3D", "CompositeHexa_3D", "Cartesian_3D"}},
      {TopAbs_FACE, "FACE", "ViscousLayers2D",
       {"Quadrangle_2D", "QuadFromMedialAxis_1D2D", "MEFISTO_2D", "PolygonPerFace_2D",
        "NETGEN_2D", "NETGEN_2D_ONLY"},
       "Quadrangle_2D, QuadFromMedialAxis_1D2D, MEFISTO_2D, PolygonPerFace_2D, NETGEN_2D "
       "and NETGEN_2D_ONLY",
       {}},
  };
  for (const LayerKind& kind : kinds) {
    SMESH_HypoFilter filter(SMESH_HypoFilter::HasName(kind.hypothesis));
    for (TopExp_Explorer ex(data_->shape, kind.type); ex.More(); ex.Next()) {
      std::list<const SMESHDS_Hypothesis*> found;
      const int count = mesh_->GetHypotheses(ex.Current(), filter, found,
                                             /*andAncestors=*/true);
      if (count == 0) {
        continue;
      }
      SMESH_subMesh* sub = mesh_->GetSubMeshContaining(ex.Current());
      SMESH_Algo* algo = sub != nullptr ? sub->GetAlgo() : nullptr;
      const std::pair<const char*, int> at =
          ordinal_of_shape_index(meshDS_->ShapeToIndex(ex.Current()));
      const std::string place =
          std::string(at.first[0] != 0 ? at.first : kind.kind_name) + " " +
          std::to_string(at.second);
      if (kind.type == TopAbs_FACE) {
        refuse_face_layers_meshed_from_above(ex.Current(), place, algo);
      }
      if (algo == nullptr || algo->GetName() == nullptr) {
        continue;  // no algorithm at all: the compute reports what is missing
      }
      const std::string name = algo->GetName();
      if (kind.builders.count(name) == 0) {
        throw PysmeshError(
            std::string("Mesher.compute: ") + kind.hypothesis + " reaches " + place +
                ", whose algorithm " + name + " does not build viscous layers.",
            std::string("Only ") + kind.listed + " build " + kind.hypothesis +
                "; with " + name + " the layers would be missing, or the compute would "
                "fail after building some. Assign one of those algorithms there, or assign "
                "the layers only to the sub-shapes such an algorithm meshes.");
      }
      if (count > 1 && kind.read_one.count(name) != 0) {
        throw PysmeshError(
            "Mesher.compute: " + std::to_string(count) + " " + kind.hypothesis +
                " hypotheses reach " + place + ", but its algorithm " + name +
                " reads one " + kind.hypothesis + " per " + kind.kind_name + ".",
            "With several, " + name + " would mesh nothing there, or build one stack and "
            "drop the others. Assign one hypothesis per " + std::string(kind.kind_name) +
            ", or use PolyhedronPerSolid_3D, which grows each hypothesis's stack on its "
            "own faces.");
      }
      SMESH_Hypothesis::Hypothesis_Status status = SMESH_Hypothesis::HYP_OK;
      const SMESH_ComputeErrorPtr why =
          kind.type == TopAbs_SOLID
              ? StdMeshers_ViscousLayers::CheckHypothesis(*mesh_, ex.Current(), status)
              : StdMeshers_ViscousLayers2D::CheckHypothesis(*mesh_, ex.Current(), status);
      if (why && !why->IsOK()) {
        const std::string reason =
            why->myComment.empty() ? std::string(status_text(status))
                                   : with_ordinals(why->myComment);
        throw PysmeshError(
            "Mesher.compute: the " + std::string(kind.hypothesis) + " hypotheses on " +
                place + " do not fit together: " + reason + " (algorithm " + name + ").",
            "SMESH checks them before it meshes " + place + ": a face set may share no "
            "face with another one, and faces that share an edge need the same number of "
            "layers. Change the face sets so that they meet these rules.");
      }
    }
  }
}

bool Mesher::uses_netgen() const {
  for (const Assignment& a : assigned_) {
    if (is_netgen_algorithm(a.name)) {
      return true;
    }
  }
  return false;
}

void Mesher::refuse_face_layers_meshed_from_above(const TopoDS_Shape& face,
                                                  const std::string& place,
                                                  const SMESH_Algo* own) const {
  // ViscousLayers2D is read by the FACE's own 2-D algorithm. An algorithm of an enclosing
  // SOLID that meshes faces itself (NeedDiscreteBoundary() false) leaves that 2-D algorithm
  // out: Cartesian_3D and PolyhedronPerSolid_3D mesh every face of their solid, and
  // Prism_3D every face but the source of its sweep, which carries a 2-D algorithm. Their
  // layers were dropped with no word, or the compute failed after meshing ("Less that 3
  // nodes on the wire", "no message").
  for (const TopoDS_Shape& above : mesh_->GetAncestors(face)) {
    if (above.ShapeType() != TopAbs_SOLID) {
      continue;
    }
    SMESH_subMesh* solid_sub = mesh_->GetSubMeshContaining(above);
    const SMESH_Algo* outer = solid_sub != nullptr ? solid_sub->GetAlgo() : nullptr;
    if (outer == nullptr || outer->GetName() == nullptr || outer->NeedDiscreteBoundary()) {
      continue;
    }
    const std::string name = outer->GetName();
    const std::pair<const char*, int> at =
        ordinal_of_shape_index(meshDS_->ShapeToIndex(above));
    const std::string solid = std::string(at.first[0] != 0 ? at.first : "SOLID") + " " +
                              std::to_string(at.second);
    if (name == "Cartesian_3D" || name == "PolyhedronPerSolid_3D") {
      throw PysmeshError(
          "Mesher.compute: ViscousLayers2D reaches " + place + ", a face of " + solid +
              ", whose algorithm " + name +
              " meshes every dimension itself and builds no 2-D layers.",
          name + " meshes the faces of " + solid + " without their 2-D algorithms, so "
          "no ViscousLayers2D is read there: the layers would be missing, or the compute "
          "would fail after building some. Grow the layers with ViscousLayers on " + solid +
          " instead; " + name + " builds them.");
    }
    if (own == nullptr) {
      throw PysmeshError(
          "Mesher.compute: ViscousLayers2D reaches " + place +
              ", which has no 2-D algorithm of its own: " + name + " of " + solid +
              " meshes it and builds no 2-D layers.",
          "Only Quadrangle_2D, QuadFromMedialAxis_1D2D, MEFISTO_2D, PolygonPerFace_2D, "
          "NETGEN_2D and NETGEN_2D_ONLY build ViscousLayers2D. With " + name +
          ", assign one of them on that face alone, "
          "with the layers there: the sweep starts from it and carries its layers "
          "through the solid.");
    }
    // A 2-D algorithm inherited from the shape above leaves the choice of the face the
    // sweep starts from to Prism_3D's own search, which may take another face: then the
    // layers stayed on this face, and the cells of the sweep did not fit it (76 cells on a
    // block of 4 x 4 x 4 with 12 layer quadrangles, no error). Only a 2-D algorithm on the
    // face itself makes it the start of the sweep, before the search runs.
    TopoDS_Shape assigned_to;
    gen_->GetAlgo(mesh_->GetSubMesh(face), &assigned_to);
    if (!assigned_to.IsSame(face)) {
      throw PysmeshError(
          "Mesher.compute: ViscousLayers2D reaches " + place + ", whose 2-D algorithm " +
              std::string(own->GetName() != nullptr ? own->GetName() : "") +
              " is assigned to a shape around it: " + name + " of " + solid +
              " chooses the face its sweep starts from, may mesh " + place +
              " itself, and then builds no 2-D layers there.",
          "Assign the 2-D algorithm on that face alone, beside the layers: the sweep then "
          "starts from it and carries its layers through the solid.");
    }
  }
}

std::string Mesher::with_ordinals(const std::string& text) const {
  std::string out;
  std::size_t i = 0;
  while (i < text.size()) {
    out += text[i];
    if (text[i] != '#' || i + 1 >= text.size() || !std::isdigit(static_cast<unsigned char>(
                                                       text[i + 1]))) {
      ++i;
      continue;
    }
    std::size_t end = i + 1;
    while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) {
      ++end;
    }
    const std::string digits = text.substr(i + 1, end - i - 1);
    out += digits;
    const std::pair<const char*, int> at =
        digits.size() > 9 ? std::pair<const char*, int>{"", 0}
                          : ordinal_of_shape_index(std::stoi(digits));
    if (at.first[0] != 0) {
      out += std::string(" (") + at.first + " " + std::to_string(at.second) + ")";
    }
    i = end;
  }
  return out;
}

void Mesher::refuse_netgen_faces_with_unmeshed_degenerate_edges() const {
  // NETGENPlugin builds NETGEN's input from the edges SMESH meshed before it
  // (NETGENPlugin_Mesher.cxx, FillNgMesh), and skips degenerate edges there. A face of
  // revolution whose seam a 1-D algorithm meshed and whose degenerate pole edges were left
  // to NETGEN ended the process with an access violation inside NETGENPlugin, in 2 to 3
  // runs of 3 to 5; the outcome moved with the stack layout, so it is a read of invalid
  // memory. With the poles meshed by a 1-D algorithm too, or with no 1-D algorithm on any
  // edge, the face meshed every time. So that one mix is refused before SMESH runs.
  const auto edge_algorithm = [this](const TopoDS_Shape& edge) -> const SMESH_Algo* {
    SMESH_subMesh* sub = mesh_->GetSubMeshContaining(edge);
    const SMESH_Algo* algo = sub != nullptr ? sub->GetAlgo() : nullptr;
    return algo != nullptr && algo->GetDim() == 1 ? algo : nullptr;
  };
  const auto list_of = [](const std::vector<int>& ordinals) {
    std::string text;
    for (const int o : ordinals) {
      text += (text.empty() ? "" : ", ") + std::to_string(o);
    }
    return text;
  };
  std::set<int> checked;
  for (TopExp_Explorer ex(data_->shape, TopAbs_FACE); ex.More(); ex.Next()) {
    const int face = ordinal_of_shape_index(meshDS_->ShapeToIndex(ex.Current())).second;
    SMESH_subMesh* sub = mesh_->GetSubMeshContaining(ex.Current());
    const SMESH_Algo* algo = sub != nullptr ? sub->GetAlgo() : nullptr;
    const std::string name = algo != nullptr && algo->GetName() != nullptr ? algo->GetName()
                                                                            : "";
    if ((name != "NETGEN_2D" && name != "NETGEN_2D_ONLY") || face <= 0 ||
        !checked.insert(face).second) {
      continue;
    }
    std::vector<int> bare_poles;
    std::vector<int> meshed_by_1d;
    std::string one_d;
    TopTools_IndexedMapOfShape edges;
    TopExp::MapShapes(ex.Current(), TopAbs_EDGE, edges);
    for (int i = 1; i <= edges.Extent(); ++i) {
      const TopoDS_Edge& edge = TopoDS::Edge(edges.FindKey(i));
      const int ordinal = ordinal_of_shape_index(meshDS_->ShapeToIndex(edge)).second;
      const SMESH_Algo* own = edge_algorithm(edge);
      if (BRep_Tool::Degenerated(edge)) {
        const SMESHDS_SubMesh* held = meshDS_->MeshElements(edge);
        const bool has_mesh = held != nullptr && (held->NbElements() > 0 || held->NbNodes() > 0);
        if (own == nullptr && !has_mesh) {
          bare_poles.push_back(ordinal);
        }
      } else if (own != nullptr) {
        meshed_by_1d.push_back(ordinal);
        one_d = own->GetName() != nullptr ? own->GetName() : "";
      }
    }
    if (bare_poles.empty() || meshed_by_1d.empty()) {
      continue;
    }
    throw PysmeshError(
        "Mesher.compute: FACE " + std::to_string(face) + ", meshed by " + name +
            ", has degenerate EDGE " + list_of(bare_poles) +
            " with no mesh and no 1-D algorithm, while its EDGE " + list_of(meshed_by_1d) +
            " has a 1-D algorithm of its own (" + one_d +
            "). NETGENPlugin reads invalid memory on such a face and can end the process, "
            "so nothing was computed.",
        "Give the degenerate EDGE " + list_of(bare_poles) +
            " a 1-D algorithm as well (Regular1D with a 1-D hypothesis on them), or remove "
            "the 1-D algorithm from EDGE " + list_of(meshed_by_1d) +
            " so that NETGEN meshes every edge of FACE " + std::to_string(face) +
            ". Either way the face meshes; only this mix fails.",
        {face});
  }
}

std::string Mesher::injected_face_orientation(const TopoDS_Face& face) const {
  const SMESHDS_SubMesh* held = meshDS_->MeshElements(face);
  if (held == nullptr || held->NbElements() == 0) {
    return std::string();
  }
  std::vector<const SMDS_MeshElement*> elements;
  for (SMDS_ElemIteratorPtr it = held->GetElements(); it->more();) {
    const SMDS_MeshElement* e = it->next();
    if (e->GetType() == SMDSAbs_Face && e->NbCornerNodes() >= 3) {
      elements.push_back(e);
    }
  }
  // Orientation is consistent when the two elements at an inner edge run it in opposite
  // directions. Edge key: (smaller id, larger id); value: the first direction and the
  // first element's index, so that elements joined by an edge form connected parts.
  std::map<std::pair<smIdType, smIdType>, std::pair<bool, std::size_t>> runs;
  std::vector<std::size_t> part(elements.size());
  for (std::size_t i = 0; i < part.size(); ++i) {
    part[i] = i;
  }
  const auto find = [&part](std::size_t a) {
    while (part[a] != a) {
      part[a] = part[part[a]];
      a = part[a];
    }
    return a;
  };
  std::size_t same_way = 0;
  for (std::size_t i = 0; i < elements.size(); ++i) {
    const SMDS_MeshElement* e = elements[i];
    const int n = e->NbCornerNodes();
    for (int k = 0; k < n; ++k) {
      const smIdType a = e->GetNode(k)->GetID();
      const smIdType b = e->GetNode((k + 1) % n)->GetID();
      const auto key = std::make_pair(std::min(a, b), std::max(a, b));
      const auto found = runs.emplace(key, std::make_pair(a < b, i));
      if (found.second) {
        continue;
      }
      same_way += found.first->second.first == (a < b) ? 1 : 0;
      const std::size_t p = find(i);
      const std::size_t q = find(found.first->second.second);
      if (p != q) {
        part[p] = q;
      }
    }
  }
  if (same_way > 0) {
    return "mixed: " + std::to_string(same_way) +
           " of its inner edges are run the same way by both of their elements";
  }

  // Each connected part has one orientation; its sign is the area-weighted vote of its
  // elements' normals against the face's outward normal at their centroids. A vote, not
  // one element, so that a few folded elements do not decide it. On a curved face the
  // normal needs a projection, so up to kSampled elements per part vote there.
  constexpr std::size_t kSampled = 64;
  const BRepAdaptor_Surface surface(face, /*Restriction=*/false);
  const bool planar = surface.GetType() == GeomAbs_Plane;
  const bool reversed_face = face.Orientation() == TopAbs_REVERSED;
  const Handle(Geom_Surface) geometry = BRep_Tool::Surface(face);
  std::map<std::size_t, std::vector<std::size_t>> parts;
  for (std::size_t i = 0; i < elements.size(); ++i) {
    parts[find(i)].push_back(i);
  }
  std::size_t inward = 0;
  for (const auto& p : parts) {
    const std::vector<std::size_t>& members = p.second;
    const std::size_t step = std::max<std::size_t>(1, members.size() / kSampled);
    double vote = 0.0;
    for (std::size_t j = 0; j < members.size(); j += planar ? 1 : step) {
      const SMDS_MeshElement* e = elements[members[j]];
      const int n = e->NbCornerNodes();
      gp_XYZ area(0.0, 0.0, 0.0);
      gp_XYZ centre(0.0, 0.0, 0.0);
      for (int k = 0; k < n; ++k) {
        const SMDS_MeshNode* a = e->GetNode(k);
        const SMDS_MeshNode* b = e->GetNode((k + 1) % n);
        area += gp_XYZ(a->X(), a->Y(), a->Z()).Crossed(gp_XYZ(b->X(), b->Y(), b->Z()));
        centre += gp_XYZ(a->X(), a->Y(), a->Z());
      }
      gp_Dir normal;
      if (planar) {
        normal = surface.Plane().Axis().Direction();
      } else {
        GeomAPI_ProjectPointOnSurf projection(gp_Pnt(centre / n), geometry);
        if (projection.NbPoints() == 0) {
          continue;
        }
        double u = 0.0, v = 0.0;
        projection.LowerDistanceParameters(u, v);
        GeomLProp_SLProps props(geometry, u, v, 1, Precision::Confusion());
        if (!props.IsNormalDefined()) {
          continue;
        }
        normal = props.Normal();
      }
      vote += 0.5 * area.Dot(normal.XYZ()) * (reversed_face ? -1.0 : 1.0);
    }
    inward += vote < 0.0 ? 1 : 0;
  }
  if (inward == 0) {
    return std::string();
  }
  if (inward < parts.size()) {
    return "mixed: " + std::to_string(inward) + " of its " + std::to_string(parts.size()) +
           " connected parts point into the solid";
  }
  return "reversed: its elements point into the solid";
}

void Mesher::refuse_misoriented_injected_faces() const {
  // NETGENPlugin_NETGEN_3D passes the elements on the faces of a solid to NETGEN as they
  // are (NETGENPlugin_NETGEN_3D.cxx:380), and NETGEN needs each to point out of the solid.
  // It reverses a face itself only when the meshed shape is a compound or a compsolid and
  // the face is not internal and bounds more than one solid (:227-239); such a face must
  // only be consistent. An internal face goes to NETGEN in both orientations. A face
  // reversed whole gave "Some edges multiple times in surface mesh", every face reversed
  // "Stop meshing since too many attempts". A face that holds elements before the compute
  // keeps them, as UseExisting_2D does, so every such face of a NETGEN_3D solid is checked.
  const TopAbs_ShapeEnum main_type = mesh_->GetShapeToMesh().ShapeType();
  const bool shared_may_flip = main_type == TopAbs_COMPOUND || main_type == TopAbs_COMPSOLID;
  std::vector<std::string> named;
  std::vector<int> faces;
  std::set<std::pair<int, int>> checked;
  for (TopExp_Explorer solid(data_->shape, TopAbs_SOLID); solid.More(); solid.Next()) {
    SMESH_subMesh* sub = mesh_->GetSubMeshContaining(solid.Current());
    const SMESH_Algo* algo = sub != nullptr ? sub->GetAlgo() : nullptr;
    if (algo == nullptr || algo->GetName() == nullptr ||
        std::string(algo->GetName()) != "NETGEN_3D") {
      continue;
    }
    const int solid_ordinal =
        ordinal_of_shape_index(meshDS_->ShapeToIndex(solid.Current())).second;
    for (TopExp_Explorer ex(solid.Current(), TopAbs_FACE); ex.More(); ex.Next()) {
      const TopoDS_Face& face = TopoDS::Face(ex.Current());
      const int ordinal = ordinal_of_shape_index(meshDS_->ShapeToIndex(face)).second;
      if (face.Orientation() == TopAbs_INTERNAL || ordinal <= 0 ||
          !checked.insert({solid_ordinal, ordinal}).second) {
        continue;
      }
      std::string why = injected_face_orientation(face);
      int solids_around = 0;
      for (const TopoDS_Shape& above : mesh_->GetAncestors(face)) {
        solids_around += above.ShapeType() == TopAbs_SOLID ? 1 : 0;
      }
      if (shared_may_flip && solids_around > 1 && why.rfind("reversed", 0) == 0) {
        continue;  // NETGENPlugin reverses this face for this solid itself
      }
      if (why.empty()) {
        continue;
      }
      named.push_back("FACE " + std::to_string(ordinal) + " of SOLID " +
                      std::to_string(solid_ordinal) + " (" + why + ")");
      if (std::find(faces.begin(), faces.end(), ordinal) == faces.end()) {
        faces.push_back(ordinal);
      }
    }
  }
  if (named.empty()) {
    return;
  }
  std::string list;
  for (const std::string& n : named) {
    list += (list.empty() ? "" : ", ") + n;
  }
  throw PysmeshError(
      "Mesher.compute: NETGEN_3D would read " + std::to_string(named.size()) +
          " face(s) whose existing elements do not all point out of their solid: " + list +
          ". Nothing was computed.",
      "NETGEN_3D passes the elements on a face to NETGEN as they are, and NETGEN needs each "
      "boundary element to point out of the solid: its node order turns about the outward "
      "normal by the right-hand rule. A reversed face makes NETGEN fail with \"Some edges "
      "multiple times in surface mesh\", every face reversed with \"Stop meshing since too "
      "many attempts\". Reverse the node order of the elements of the named faces, or of "
      "their reversed part. A face that two solids of a compound share needs one "
      "orientation only: NETGENPlugin reverses it for one of them.",
      faces);
}

Mesher::OpenFront Mesher::open_front(const TopoDS_Shape& solid) const {
  // A face is keyed by its corner node ids, sorted, so that the face of one volume and the
  // same face of its neighbour, or of a boundary element, compare equal whatever their node
  // order. Up to 4 corners fit an array, padded with -1; a face with more corners (of a
  // polyhedron or a polygon) goes to a map of its own.
  using Key = std::array<smIdType, 4>;
  const auto key_of = [](const std::vector<smIdType>& ids) {
    Key key{-1, -1, -1, -1};
    std::copy(ids.begin(), ids.end(), key.begin());
    std::sort(key.begin(), key.begin() + static_cast<std::ptrdiff_t>(ids.size()));
    return key;
  };
  std::vector<smIdType> ids;

  // The boundary: every 2-D element on a face of the solid.
  std::vector<Key> boundary;
  std::set<std::vector<smIdType>> boundary_wide;
  std::vector<const SMDS_MeshElement*> boundary_elements;
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(solid, TopAbs_FACE, faces);
  for (int i = 1; i <= faces.Extent(); ++i) {
    const SMESHDS_SubMesh* held = meshDS_->MeshElements(faces.FindKey(i));
    if (held == nullptr) {
      continue;
    }
    for (SMDS_ElemIteratorPtr it = held->GetElements(); it->more();) {
      const SMDS_MeshElement* e = it->next();
      if (e->GetType() != SMDSAbs_Face) {
        continue;
      }
      ids.clear();
      for (int k = 0; k < e->NbCornerNodes(); ++k) {
        ids.push_back(e->GetNode(k)->GetID());
      }
      boundary_elements.push_back(e);
      if (ids.size() <= 4) {
        boundary.push_back(key_of(ids));
      } else {
        std::sort(ids.begin(), ids.end());
        boundary_wide.insert(ids);
      }
    }
  }
  std::sort(boundary.begin(), boundary.end());

  // Every face of every volume element of the solid, with the element and the face index
  // that give its nodes back.
  struct VolumeFace {
    Key key;
    const SMDS_MeshElement* volume;
    int face;
  };
  std::vector<VolumeFace> volume_faces;
  std::map<std::vector<smIdType>, std::pair<int, VolumeFace>> wide_faces;
  std::size_t volumes = 0;
  // The corner nodes of face `f` of volume `v`, whose corners load_corners(v) read. SMDS
  // stores the corner nodes of a quadratic cell first, so the face table of its linear
  // type indexes them. Each node read asks the VTK grid for the whole cell, so the corners
  // are read once per volume; only a polyhedron needs the volume tool. Measured on 207 632
  // tetrahedra, the whole search: 0.52 s with the volume tool on every cell, 0.30 s with
  // the tables.
  SMDS_VolumeTool tool;
  std::vector<const SMDS_MeshNode*> corner;
  std::vector<const SMDS_MeshNode*> face_nodes;
  const auto load_corners = [&corner](const SMDS_MeshElement* v) {
    corner.clear();
    const std::size_t count = static_cast<std::size_t>(v->NbCornerNodes());
    for (SMDS_NodeIteratorPtr it = v->nodeIterator(); it->more() && corner.size() < count;) {
      corner.push_back(it->next());
    }
  };
  const auto face_of = [&](const SMDS_MeshElement* v, int f) {
    face_nodes.clear();
    if (v->IsPoly()) {
      tool.Set(v);
      const SMDS_MeshNode* const* nodes = tool.GetFaceNodes(f);
      for (int i = 0; i < tool.NbFaceNodes(f); ++i) {
        if (!v->IsMediumNode(nodes[i])) {
          face_nodes.push_back(nodes[i]);
        }
      }
      return;
    }
    const SMDS_VolumeTool::VolumeType type = SMDS_VolumeTool::GetType(v->NbCornerNodes());
    const int* index = SMDS_VolumeTool::GetFaceNodesIndices(type, f, /*external=*/false);
    for (int i = 0; i < SMDS_VolumeTool::NbFaceNodes(type, f); ++i) {
      face_nodes.push_back(corner[static_cast<std::size_t>(index[i])]);
    }
  };
  const auto face_count = [&tool](const SMDS_MeshElement* v) {
    if (v->IsPoly()) {
      return tool.Set(v) ? tool.NbFaces() : 0;
    }
    const SMDS_VolumeTool::VolumeType type = SMDS_VolumeTool::GetType(v->NbCornerNodes());
    return type == SMDS_VolumeTool::UNKNOWN ? 0 : SMDS_VolumeTool::NbFaces(type);
  };
  if (const SMESHDS_SubMesh* held = meshDS_->MeshElements(solid)) {
    volume_faces.reserve(4 * static_cast<std::size_t>(held->NbElements()));
    for (SMDS_ElemIteratorPtr it = held->GetElements(); it->more();) {
      const SMDS_MeshElement* v = it->next();
      const int nb_faces = v->GetType() == SMDSAbs_Volume ? face_count(v) : 0;
      if (nb_faces == 0) {
        continue;
      }
      ++volumes;
      load_corners(v);
      for (int f = 0; f < nb_faces; ++f) {
        face_of(v, f);
        ids.clear();
        for (const SMDS_MeshNode* n : face_nodes) {
          ids.push_back(n->GetID());
        }
        if (ids.size() <= 4) {
          volume_faces.push_back({key_of(ids), v, f});
        } else {
          std::sort(ids.begin(), ids.end());
          auto& seen = wide_faces[ids];
          seen.first += 1;
          seen.second = {Key{}, v, f};
        }
      }
    }
  }
  std::sort(volume_faces.begin(), volume_faces.end(),
            [](const VolumeFace& a, const VolumeFace& b) { return a.key < b.key; });

  // The open faces, each as its corner nodes.
  std::vector<std::vector<const SMDS_MeshNode*>> open;
  const auto add_volume_face = [&](const VolumeFace& vf) {
    load_corners(vf.volume);
    face_of(vf.volume, vf.face);
    open.push_back(face_nodes);
  };
  OpenFront front;
  if (volumes == 0) {
    // NETGEN kept no volume element, so its front at the stop is the whole boundary.
    front.no_volume = true;
    std::set<std::vector<smIdType>> seen;
    for (const SMDS_MeshElement* e : boundary_elements) {
      std::vector<const SMDS_MeshNode*> face;
      std::vector<smIdType> key;
      for (int i = 0; i < e->NbCornerNodes(); ++i) {
        face.push_back(e->GetNode(i));
        key.push_back(e->GetNode(i)->GetID());
      }
      std::sort(key.begin(), key.end());
      if (seen.insert(key).second) {
        open.push_back(face);
      }
    }
  } else {
    for (std::size_t i = 0; i < volume_faces.size();) {
      std::size_t j = i + 1;
      while (j < volume_faces.size() && volume_faces[j].key == volume_faces[i].key) {
        ++j;
      }
      if (j == i + 1 &&
          !std::binary_search(boundary.begin(), boundary.end(), volume_faces[i].key)) {
        add_volume_face(volume_faces[i]);
      }
      i = j;
    }
    for (const auto& wide : wide_faces) {
      if (wide.second.first == 1 && boundary_wide.count(wide.first) == 0) {
        add_volume_face(wide.second.second);
      }
    }
  }
  front.faces = open.size();
  if (open.empty()) {
    return front;
  }

  // Patches: open faces joined by a shared node (union-find over the faces).
  std::vector<std::size_t> parent(open.size());
  for (std::size_t i = 0; i < parent.size(); ++i) {
    parent[i] = i;
  }
  const auto find = [&parent](std::size_t a) {
    while (parent[a] != a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  };
  std::map<const SMDS_MeshNode*, std::size_t> first_face;
  for (std::size_t i = 0; i < open.size(); ++i) {
    double c[3] = {0.0, 0.0, 0.0};
    for (const SMDS_MeshNode* n : open[i]) {
      c[0] += n->X();
      c[1] += n->Y();
      c[2] += n->Z();
      const auto at = first_face.emplace(n, i).first;
      const std::size_t a = find(i);
      const std::size_t b = find(at->second);
      if (a != b) {
        parent[a] = b;
      }
    }
    for (double& v : c) {
      v /= static_cast<double>(open[i].size());
    }
    front.centroids.insert(front.centroids.end(), c, c + 3);
  }
  std::map<std::size_t, Patch> patches;
  for (std::size_t i = 0; i < open.size(); ++i) {
    Patch& p = patches[find(i)];
    p.faces += 1;
    for (const SMDS_MeshNode* n : open[i]) {
      const double xyz[3] = {n->X(), n->Y(), n->Z()};
      for (int d = 0; d < 3; ++d) {
        p.low[d] = std::min(p.low[d], xyz[d]);
        p.high[d] = std::max(p.high[d], xyz[d]);
      }
    }
  }
  for (const auto& p : patches) {
    front.patches.push_back(p.second);
  }
  std::stable_sort(front.patches.begin(), front.patches.end(),
                   [](const Patch& a, const Patch& b) { return a.faces > b.faces; });
  return front;
}

std::string Mesher::open_front_text(const OpenFront& front) {
  constexpr std::size_t kListed = 5;
  const auto point = [](const double* xyz) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "(%.6g, %.6g, %.6g)", xyz[0], xyz[1], xyz[2]);
    return std::string(buf);
  };
  std::string text = "; open front of the kept mesh: " + std::to_string(front.faces) +
                     " face(s) in " + std::to_string(front.patches.size()) + " patch(es)";
  if (front.no_volume) {
    text += " (NETGEN kept no volume element, so the whole boundary is open)";
  }
  text += ", the box of the " + std::to_string(std::min(front.patches.size(), kListed)) +
          " largest: ";
  for (std::size_t i = 0; i < front.patches.size() && i < kListed; ++i) {
    const Patch& p = front.patches[i];
    text += std::string(i == 0 ? "" : ", ") + std::to_string(p.faces) + " face(s) in " +
            point(p.low) + " to " + point(p.high);
  }
  return text;
}

std::string Mesher::netgen_face_sizes(const TopoDS_Shape& face, int ordinal,
                                      const SMESH_Algo& algo) const {
  // NETGEN's surface mesher throws "Problem in Surface mesh generation" and drops the
  // reason (occgenmesh.cpp:935-957). The usual one: the face's edges were meshed before it,
  // by a neighbour with a coarser size, and their segments are longer than the face is
  // wide, so the 2-D front cannot grade down from them inside the face. Three numbers show
  // it: the longest segment already on the face's edges, the smallest extent of the face's
  // box, and the size the face's hypothesis asks for.
  double longest = 0.0;
  std::size_t segments = 0;
  TopTools_IndexedMapOfShape edges;
  TopExp::MapShapes(face, TopAbs_EDGE, edges);
  for (int i = 1; i <= edges.Extent(); ++i) {
    const SMESHDS_SubMesh* held = meshDS_->MeshElements(edges.FindKey(i));
    if (held == nullptr) {
      continue;
    }
    for (SMDS_ElemIteratorPtr it = held->GetElements(); it->more();) {
      const SMDS_MeshElement* segment = it->next();
      if (segment->GetType() != SMDSAbs_Edge || segment->NbCornerNodes() != 2) {
        continue;
      }
      const SMDS_MeshNode* a = segment->GetNode(0);
      const SMDS_MeshNode* b = segment->GetNode(1);
      longest = std::max(longest, gp_Pnt(a->X(), a->Y(), a->Z())
                                      .Distance(gp_Pnt(b->X(), b->Y(), b->Z())));
      ++segments;
    }
  }
  // The box of the geometry, as Session.bounding_boxes computes it. A plane face along an
  // axis has an extent of zero across itself, so only extents above the confusion count.
  Bnd_Box box;
  BRepBndLib::AddOptimal(face, box, /*useTriangulation=*/false, /*useShapeTolerance=*/false);
  double extent = RealLast();
  if (!box.IsVoid()) {
    double lo[3], hi[3];
    box.Get(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
    for (int d = 0; d < 3; ++d) {
      if (hi[d] - lo[d] > Precision::Confusion()) {
        extent = std::min(extent, hi[d] - lo[d]);
      }
    }
  }
  const double asked =
      netgen_size_asked(*mesh_, algo, face, "FACE:" + std::to_string(ordinal));
  const auto number = [](double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return std::string(buf);
  };
  std::string text = "; longest segment on its edges ";
  text += segments == 0 ? std::string("none (no edge was meshed)") : number(longest);
  text += ", smallest extent of its box " +
          (extent < RealLast() ? number(extent) : std::string("none"));
  text += ", size its hypothesis asks for " +
          (asked > 0.0 ? number(asked) : std::string("none (NETGEN's default)"));
  if (segments > 0 && extent < RealLast() && longest > extent) {
    text += ". The edges of this face were meshed coarser than the face can take, before "
            "it: explicit segment counts on its edges (Regular1D with NumberOfSegments) "
            "avoid it";
  }
  return text;
}

std::string Mesher::prism_side_face_cause(const TopoDS_Shape& solid) const {
  // Prism_3D sweeps quadrangles through the side faces of its solid, and refuses a solid
  // with more than two faces that hold anything else: "More than 2 faces with not
  // quadrangle elements" (StdMeshers_Prism_3D.cxx). A 2-D algorithm that reaches a side
  // face, a global one included, meshes that face with triangles before Prism_3D runs.
  // When the Prism_3D solid is computed after a solid that shares a face with it, SMESH
  // cleans those triangles and the error text again, and the compute fails with no text.
  // So a face counts when it holds an element that is not a quadrangle, or when its 2-D
  // algorithm makes triangles.
  SMESH_subMesh* sub = mesh_->GetSubMeshContaining(solid);
  const SMESH_Algo* algo = sub != nullptr ? sub->GetAlgo() : nullptr;
  if (algo == nullptr || algo->GetName() == nullptr ||
      std::string(algo->GetName()) != "Prism_3D") {
    return std::string();
  }
  static const std::set<std::string> kTriangleMakers = {"NETGEN_2D", "NETGEN_2D_ONLY",
                                                        "MEFISTO_2D"};
  std::set<int> faces;
  std::set<std::string> makers;
  TopTools_IndexedMapOfShape solid_faces;
  TopExp::MapShapes(solid, TopAbs_FACE, solid_faces);
  for (int i = 1; i <= solid_faces.Extent(); ++i) {
    const TopoDS_Shape& face = solid_faces.FindKey(i);
    SMESH_subMesh* face_sub = mesh_->GetSubMeshContaining(face);
    const SMESH_Algo* face_algo = face_sub != nullptr ? face_sub->GetAlgo() : nullptr;
    const std::string maker =
        face_algo != nullptr && face_algo->GetName() != nullptr ? face_algo->GetName() : "";
    bool not_quadrangles = false;
    if (const SMESHDS_SubMesh* held = meshDS_->MeshElements(face)) {
      for (SMDS_ElemIteratorPtr it = held->GetElements(); it->more() && !not_quadrangles;) {
        not_quadrangles = it->next()->NbCornerNodes() != 4;
      }
    }
    if (!not_quadrangles && kTriangleMakers.count(maker) == 0) {
      continue;
    }
    const int ordinal = ordinal_of_shape_index(meshDS_->ShapeToIndex(face)).second;
    if (ordinal > 0) {
      faces.insert(ordinal);
    }
    if (!maker.empty()) {
      makers.insert(maker);
    }
  }
  if (faces.size() <= 2) {
    return std::string();
  }
  std::string face_list;
  for (const int f : faces) {
    face_list += (face_list.empty() ? "" : ", ") + std::to_string(f);
  }
  std::string maker_list;
  for (const std::string& m : makers) {
    maker_list += (maker_list.empty() ? "" : " and ") + m;
  }
  return ". Prism_3D needs quadrangles on every face but its source and its target, and " +
         std::to_string(faces.size()) + " faces of this solid (FACE " + face_list +
         ") take triangles" + (maker_list.empty() ? "" : " from " + maker_list) +
         ": a 2-D algorithm that reaches a side face, a global one included, meshes it "
         "before Prism_3D runs. Assign Quadrangle2D on the side faces, with Regular1D and a "
         "1-D hypothesis on their edges";
}

std::string Mesher::with_shrink_face_ordinals(const std::string& text,
                                              std::vector<int>& faces) const {
  // StdMeshers_ViscousLayers names the face it could not shrink by its SMESHDS shape index:
  // "Can't shrink 2D mesh on face <index>" and "Infinite loop at shrinking 2D mesh on face
  // <index>" (StdMeshers_ViscousLayers.cxx:11644-11647). That index counts every sub-shape
  // of every kind, so it reads as a face ordinal that the model does not have.
  static const std::string kPhrase = "2D mesh on face ";
  std::string out;
  std::size_t from = 0;
  for (std::size_t at = text.find(kPhrase); at != std::string::npos;
       at = text.find(kPhrase, from)) {
    const std::size_t digits = at + kPhrase.size();
    std::size_t end = digits;
    while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) {
      ++end;
    }
    const std::pair<const char*, int> face =
        end > digits && end - digits <= 9
            ? ordinal_of_shape_index(std::stoi(text.substr(digits, end - digits)))
            : std::pair<const char*, int>{"", 0};
    if (std::string(face.first) == "FACE") {
      out += text.substr(from, at - from) + "2D mesh on FACE " + std::to_string(face.second);
      faces.push_back(face.second);
    } else {
      out += text.substr(from, end - from);
    }
    from = end;
  }
  return out + text.substr(from);
}

std::pair<const char*, int> Mesher::ordinal_of_shape_index(int shape_index) const {
  if (shape_index <= 0 ||
      static_cast<std::size_t>(shape_index) >= index_to_ordinal_.size()) {
    return {"", 0};
  }
  return index_to_ordinal_[static_cast<std::size_t>(shape_index)];
}

void Mesher::assign(const std::string& name, const py::dict& params, const std::string& kind,
                    int ordinal) {
  ensure_open();
  // A mesher with no shape takes the NETGEN remesher and its parameters, on the whole mesh:
  // SMESH holds them on its pseudo-shape (SMESH_Mesh::PseudoShape) and computes them on the
  // mesh alone. Everything else needs a shape.
  if (data_ == nullptr && !(kind.empty() && works_without_shape(name))) {
    refuse_without_shape("Mesher.assign", name, kind);
  }
  // The remesher and its parameters only work on a mesh alone: on a shape, NETGEN_Remesher_2D
  // computes nothing and returns false (NETGENPlugin_Remesher_2D.cxx).
  if (data_ != nullptr && works_without_shape(name)) {
    throw PysmeshError("Mesher.assign: '" + name + "' remeshes a mesher with no shape; this "
                       "one has a shape.",
                       "Build the mesher with shape=None (or Mesher.from_arrays) from the "
                       "triangles to remesh. On a shape, mesh the faces with Netgen1D2D or "
                       "Netgen2D.");
  }
  const TopoDS_Shape& target =
      data_ == nullptr ? mesh_->GetShapeToMesh() : sub_shape(kind, ordinal);  // validates

  SMESH_Hypothesis* hyp = build(name, params);  // ownership taken inside build()
  const int hyp_id = hyp->GetID();

  std::string detail;
  const SMESH_Hypothesis::Hypothesis_Status status =
      mesh_->AddHypothesis(target, hyp_id, &detail);
  if (SMESH_Hypothesis::IsStatusFatal(status)) {
    // Leave the hypothesis owned but unassigned: it is registered in the generator's maps
    // and freeing it here would leave a dangling entry behind.
    throw PysmeshError("Mesher.assign: SMESH refused '" + name + "' on " +
                           where(kind, ordinal) + " — " + status_text(status) + ".",
                       detail);
  }
  if (status == SMESH_Hypothesis::HYP_CONCURRENT) {
    // An ambiguous model: a sub-shape under `target` is governed by two different
    // hypotheses of one kind on shapes around it, and which one meshes it is undefined
    // (SMESH_subMesh::CheckConcurrentHypothesis). Undo the assignment and say where.
    const std::string why = describe_concurrency(target, hyp);
    mesh_->RemoveHypothesis(target, hyp_id);
    throw PysmeshError("Mesher.assign: '" + name + "' on " + where(kind, ordinal) +
                           " makes the model ambiguous (SMESH status HYP_CONCURRENT); it "
                           "was not assigned.",
                       why);
  }
  // A NETGEN algorithm judges the hypotheses it reads together in its own CheckHypothesis:
  // NETGEN_2D_ONLY reads at most one of MaxElementArea, LengthFromEdges and
  // NETGEN_Parameters_2D on a face, and QuadranglePreference not beside the parameters
  // (HYP_CONCURRENT, HYP_INCOMPAT_HYPS). For a hypothesis assigned on the face SMESH
  // returns that status. For one assigned on an ancestor of the face it does not: the face
  // only turns MISSING_HYP and AddHypothesis returns HYP_OK (SMESH_subMesh.cxx,
  // AlgoStateEngine, ADD_FATHER_HYP), and the conflict surfaces at compute. So the NETGEN
  // algorithms that now read `hyp` are asked here, and a conflict undoes the assignment.
  int conflict_status = SMESH_Hypothesis::HYP_OK;
  const std::string conflict = netgen_conflict(target, hyp, conflict_status);
  if (!conflict.empty()) {
    mesh_->RemoveHypothesis(target, hyp_id);
    const auto st = static_cast<SMESH_Hypothesis::Hypothesis_Status>(conflict_status);
    throw PysmeshError("Mesher.assign: '" + name + "' on " + where(kind, ordinal) +
                           " gives a NETGEN algorithm hypotheses it cannot combine (SMESH "
                           "status " + std::string(st == SMESH_Hypothesis::HYP_CONCURRENT
                                                       ? "HYP_CONCURRENT"
                                                       : "HYP_INCOMPAT_HYPS") +
                           ": " + status_text(st) + "); it was not assigned.",
                       conflict);
  }
  assigned_.push_back({name, kind, ordinal, hyp_id, params});
}

std::string Mesher::netgen_conflict(const TopoDS_Shape& target, SMESH_Hypothesis* hyp,
                                    int& status) const {
  SMESH_subMesh* top = mesh_->GetSubMesh(target);
  for (SMESH_subMeshIteratorPtr it = top->getDependsOnIterator(/*includeSelf=*/true,
                                                               /*complexFirst=*/false);
       it->more();) {
    SMESH_subMesh* sm = it->next();
    SMESH_Algo* algo = sm->GetAlgo();
    if (algo == nullptr || algo->GetName() == nullptr ||
        !is_netgen_algorithm(algo->GetName())) {
      continue;
    }
    const TopoDS_Shape& shape = sm->GetSubShape();
    const std::list<const SMESHDS_Hypothesis*>& used =
        algo->GetUsedHypothesis(*mesh_, shape, /*ignoreAuxiliary=*/false);
    if (std::find(used.begin(), used.end(), hyp) == used.end()) {
      continue;
    }
    SMESH_Hypothesis::Hypothesis_Status st = SMESH_Hypothesis::HYP_OK;
    algo->CheckHypothesis(*mesh_, shape, st);
    if (st != SMESH_Hypothesis::HYP_CONCURRENT && st != SMESH_Hypothesis::HYP_INCOMPAT_HYPS) {
      continue;
    }
    status = st;
    std::string names;
    for (const SMESHDS_Hypothesis* h : algo->GetUsedHypothesis(*mesh_, shape, false)) {
      names += (names.empty() ? "" : ", ") + std::string(h->GetName());
    }
    const std::pair<const char*, int> at = ordinal_of_shape_index(meshDS_->ShapeToIndex(shape));
    std::string text = std::string(algo->GetName()) + " on " +
                       (at.first[0] != 0 ? at.first : "sub-shape") + " " +
                       std::to_string(at.second) + " would read " + names + ".";
    const SMESH_ComputeErrorPtr reason = algo->GetComputeError();
    if (reason && !reason->myComment.empty()) {
      text += " " + reason->myComment + ".";
    }
    if (std::string(algo->GetName()) == "NETGEN_2D_ONLY") {
      text += " NETGEN_2D_ONLY reads at most one of MaxElementArea, LengthFromEdges and "
              "NETGEN_Parameters_2D on a face, and QuadranglePreference not beside "
              "NETGEN_Parameters_2D (NETGENPlugin_NETGEN_2D_ONLY.cxx, CheckHypothesis).";
    }
    return text;
  }
  return std::string();
}

void Mesher::unassign(const std::string& name, const py::dict& params, const std::string& kind,
                      int ordinal) {
  ensure_open();
  if (data_ == nullptr && !(kind.empty() && works_without_shape(name))) {
    refuse_without_shape("Mesher.unassign", name, kind);
  }
  const TopoDS_Shape& target =
      data_ == nullptr ? mesh_->GetShapeToMesh() : sub_shape(kind, ordinal);
  std::vector<std::vector<Assignment>::iterator> named;
  for (auto it = assigned_.begin(); it != assigned_.end(); ++it) {
    if (it->name == name && it->kind == kind && it->ordinal == ordinal) {
      named.push_back(it);
    }
  }
  if (named.empty()) {
    throw PysmeshError("Mesher.unassign: '" + name + "' is not assigned to " +
                       where(kind, ordinal) + ".");
  }
  // SMESH attaches several auxiliary hypotheses of one type to one sub-shape (several
  // ViscousLayers, each with its own face set). The name alone then does not say which one
  // the caller means, so the parameters decide.
  auto chosen = named.front();
  if (named.size() > 1) {
    chosen = assigned_.end();
    for (const auto& it : named) {
      if (it->params.equal(params)) {
        chosen = it;
        break;
      }
    }
    if (chosen == assigned_.end()) {
      throw PysmeshError(
          "Mesher.unassign: " + std::to_string(named.size()) + " '" + name +
              "' hypotheses are assigned to " + where(kind, ordinal) +
              ", and none has the parameters given; nothing was detached.",
          "With several of one name on one sub-shape, unassign detaches the one equal to "
          "the instance it is given, field for field. Pass an instance equal to the one "
          "to detach.");
    }
  }
  const SMESH_Hypothesis::Hypothesis_Status status =
      mesh_->RemoveHypothesis(target, chosen->hyp_id);
  if (SMESH_Hypothesis::IsStatusFatal(status)) {
    throw PysmeshError("Mesher.unassign: SMESH refused to detach '" + name + "' from " +
                       where(kind, ordinal) + " — " + status_text(status) + ".");
  }
  assigned_.erase(chosen);
}

py::list Mesher::assignments() const {
  ensure_open();
  py::list out;
  for (const Assignment& a : assigned_) {
    out.append(py::make_tuple(a.name, a.kind, a.ordinal));
  }
  return out;
}

py::dict Mesher::compute(const py::object& progress, const py::object& cancel) {
  ensure_open();
  // With no shape only the NETGEN remesher computes: on the mesh's own triangles, on the
  // pseudo-shape that holds it (Mesher::assign).
  const bool shape_free = data_ == nullptr;
  if (shape_free) {
    bool remesher = false;
    for (const Assignment& a : assigned_) {
      remesher = remesher || a.name == "NETGEN_Remesher_2D";
    }
    if (!remesher) {
      ensure_shape("Mesher.compute");
    }
    if (meshDS_->NbFaces() == 0) {
      throw PysmeshError("Mesher.compute: the NETGEN remesher has nothing to remesh: this "
                         "mesher holds no face.");
    }
    check_remesher_input(*mesh_);
  }
  if (assigned_.empty()) {
    throw PysmeshError("Mesher.compute: nothing is assigned. Assign at least an algorithm "
                       "before computing.");
  }

  if (!shape_free) {
    refuse_unread_layers();
    refuse_misoriented_injected_faces();
    refuse_netgen_faces_with_unmeshed_degenerate_edges();
  }

  ProgressHooks hooks;
  if (!progress.is_none()) {
    if (!py::hasattr(progress, "__call__")) {
      throw PysmeshError("Mesher.compute: progress must be callable or None.");
    }
    hooks.on_progress = progress;
  }
  if (!cancel.is_none()) {
    if (!py::hasattr(cancel, "__call__")) {
      throw PysmeshError("Mesher.compute: cancel must be callable or None.");
    }
    hooks.should_cancel = cancel;
  }

  // netgen keeps its state in globals, so a compute that runs a NETGEN algorithm holds the
  // process-wide NETGEN lock for its whole run. The lock is taken before the progress
  // driver starts: a cancel then reaches netgen's global cancel flag only while this
  // compute owns netgen. Declared first, it is released last.
  std::unique_lock<std::mutex> netgen_lock;
  if (uses_netgen()) {
    py::gil_scoped_release release;
    netgen_lock = std::unique_lock<std::mutex>(netgen_mutex());
  }
  // SMESH_Gen::CancelCompute sets a flag that only PrepareCompute clears, and Compute stops
  // at the first sub-mesh while it is set. SALOME calls PrepareCompute before every Compute
  // (SMESH_Gen_i::Compute); without it, one cancel made every later compute of this mesher
  // fail with no error text. Called before the driver starts, so it never clears a cancel
  // of this run.
  const TopoDS_Shape& target = shape_free ? mesh_->GetShapeToMesh() : data_->shape;
  // The remesher marks the pseudo-shape always computed once it has replaced the mesh
  // (NETGENPlugin_Remesher_2D.cxx); a set mark before the run means SMESH runs nothing.
  SMESH_subMesh* const pseudo = shape_free ? mesh_->GetSubMesh(target) : nullptr;
  const bool remeshed_before = pseudo != nullptr && pseudo->IsAlwaysComputed();
  // A failed or cancelled remesh leaves the pseudo-shape FAILED_TO_COMPUTE, and SMESH
  // computes nothing there until its algorithm state changes. The mesh is the input either
  // way, so the state is reset and the remesher runs on it again.
  if (shape_free && !remeshed_before) {
    pseudo->ComputeStateEngine(SMESH_subMesh::MODIF_ALGO_STATE);
  }
  gen_->PrepareCompute(*mesh_, target);
  ComputeDriver driver(*mesh_, *gen_, target, hooks);
  bool ok = false;
  if (!driver.cancelled()) {
    py::gil_scoped_release release;
    run_on_compute_thread([&] { ok = gen_->Compute(*mesh_, target); });
  }
  // finish() re-raises an exception a hook threw, with its own type. A raising hook is a
  // cancel, so the mesh is cleared first: a cancel leaves no partial mesh, and before this
  // the re-raise skipped the clear below (report M1).
  try {
    driver.finish();
  } catch (...) {
    clear_mesh();
    throw;
  }

  // The driver's own flag decides a cancellation, never Compute()'s return value: a cancel
  // landing late gives a complete mesh and the same `false`, and an ordinary failure gives
  // `false` with no cancel at all. Checked before the failure path so a cancelled run is not
  // reported as an impossible assignment.
  if (driver.cancelled() && shape_free) {
    // The remesher replaces the mesh only after netgen finished, and binds nothing to a
    // sub-shape, so there is nothing to clear: the mesh is the input or the whole result.
    const bool replaced = !remeshed_before && pseudo->IsAlwaysComputed();
    throw CancelledError("Mesher.compute: cancelled by the caller.",
                         replaced ? "The cancel came after the NETGEN remesher replaced the "
                                    "mesh: the mesh is the remeshed one."
                                  : "The NETGEN remesher did not replace the mesh: the mesh "
                                    "is as it was before this compute.");
  }
  if (driver.cancelled()) {
    clear_mesh();
    throw CancelledError("Mesher.compute: cancelled by the caller.",
                         "The mesh was cleared: nothing partial is returned. Cancellation is "
                         "not preemptive — only a few algorithms poll it inside their own "
                         "loop, so a long single algorithm runs to its end before stopping.");
  }

  if (shape_free) {
    return shape_free_outcome(ok);
  }

  // SMESH_ComputeError is attached to the sub-mesh that actually failed, not to the
  // top-level one. A Quadrangle_2D failure on a cylinder is reported on the two circular
  // FACEs while the enclosing SOLID reports nothing, so every dimension has to be walked.
  //
  // A COMPERR_WARNING is not a failure: SMESH defines it as "algo reports error but sub-mesh
  // is computed anyway" (SMESH_ComputeError.hxx:55) and marks the sub-mesh COMPUTE_OK for it
  // (SMESH_subMesh.cxx, ComputeStateEngine). IsOK() is false for a warning and IsKO() is not,
  // so IsKO() decides, and a warning goes on the report instead (report A1).
  std::vector<std::string> failures;
  std::vector<int> failed_faces;
  // A shared face is explored once per solid, so its error is read once, by its address.
  std::set<const SMESH_ComputeError*> read_bad;
  std::vector<double> bad_centroids;
  py::list warnings;
  for (std::size_t k = 0; k < 4; ++k) {
    for (TopExp_Explorer ex(data_->shape, kKindTypes[k]); ex.More(); ex.Next()) {
      SMESH_subMesh* sub = mesh_->GetSubMeshContaining(ex.Current());
      if (sub == nullptr) {
        continue;
      }
      const SMESH_ComputeErrorPtr err = sub->GetComputeError();
      if (!err || err->IsOK()) {
        continue;
      }
      const int index = meshDS_->ShapeToIndex(ex.Current());
      const std::pair<const char*, int> at = ordinal_of_shape_index(index);
      const char* algorithm = err->myAlgo != nullptr && err->myAlgo->GetName() != nullptr
                                  ? err->myAlgo->GetName()
                                  : "";
      const std::string kind = at.first[0] ? at.first : kKindNames[k];
      if (!err->IsKO()) {
        warnings.append(py::make_tuple(kind, at.second, std::string(algorithm),
                                       err->myComment));
        continue;
      }
      std::string line = kind + " " + std::to_string(at.second) + ": ";
      std::vector<int> shrink_faces;
      line += err->myComment.empty() ? std::string("no message")
                                     : with_shrink_face_ordinals(err->myComment, shrink_faces);
      for (const int face : shrink_faces) {
        if (std::find(failed_faces.begin(), failed_faces.end(), face) == failed_faces.end()) {
          failed_faces.push_back(face);
        }
      }
      // myAlgo is the algorithm object, not its name — naming it is what makes the message
      // actionable, because the failure is nearly always the algorithm rather than the shape.
      if (algorithm[0] != '\0') {
        line += std::string(" (algorithm ") + algorithm + ")";
      }
      if (read_bad.insert(err.get()).second) {
        line += bad_elements_text(*err, bad_centroids);
      }
      if (kKindTypes[k] == TopAbs_SOLID) {
        line += prism_side_face_cause(ex.Current());
      }
      const std::string algorithm_name = algorithm;
      if (kKindTypes[k] == TopAbs_FACE && err->myAlgo != nullptr &&
          (algorithm_name == "NETGEN_2D" || algorithm_name == "NETGEN_2D_ONLY")) {
        line += netgen_face_sizes(ex.Current(), at.second, *err->myAlgo);
      }
      // NETGEN 3-D stops a whole subdomain with no location ("Stop meshing since too many
      // attempts", meshfunc.cpp:218-234). Where it names no bad element, the open front of
      // the mesh it kept shows where the volume mesh ends.
      const auto* bad = dynamic_cast<const SMESH_BadInputElements*>(err.get());
      if (kKindTypes[k] == TopAbs_SOLID &&
          (algorithm_name == "NETGEN_3D" || algorithm_name == "NETGEN_2D3D") &&
          (bad == nullptr || !bad->HasBadElems())) {
        const OpenFront front = open_front(ex.Current());
        if (front.faces > 0) {
          line += open_front_text(front);
          bad_centroids.insert(bad_centroids.end(), front.centroids.begin(),
                               front.centroids.end());
        }
      }
      bool seen = false;
      for (const std::string& s : failures) {
        seen = seen || s == line;
      }
      if (!seen) {
        failures.push_back(line);
      }
      if (kKindTypes[k] == TopAbs_FACE && at.second > 0) {
        failed_faces.push_back(at.second);
      }
    }
  }

  // A missing algorithm, or an algorithm without the hypothesis it needs, is an algorithm
  // state of the sub-mesh, not a compute error: no sub-mesh carries an error text (report
  // A6: "failed on 0 sub-shape(s)"). So each sub-mesh that was not computed is asked for its
  // state. NO_ALGO counts only after a failed compute, where an enclosing algorithm needs
  // this mesh. A sub-mesh whose algorithm misses its hypothesis is never computed, but
  // SMESH_Gen::Compute returns false only for a compute that failed
  // (SMESH_Gen::sequentialComputeSubMeshes), so it returned true with that sub-shape left
  // without a mesh (M3). So the states are asked after every compute, and MISSING_HYP fails
  // a compute that SMESH reports as done. A VERTEX keeps its node whatever its algorithm
  // does: only its 0-D algorithm (SegmentAroundVertex_0D) counts, when it misses its
  // hypothesis and so does nothing.
  const bool failed = !ok || !failures.empty();
  std::vector<std::pair<std::pair<std::size_t, int>, std::string>> states;
  for (std::size_t k = 0; k < 4; ++k) {
    const bool is_vertex = kKindTypes[k] == TopAbs_VERTEX;
    for (TopExp_Explorer ex(data_->shape, kKindTypes[k]); ex.More(); ex.Next()) {
      SMESH_subMesh* sub = mesh_->GetSubMeshContaining(ex.Current());
      if (sub == nullptr || (sub->IsMeshComputed() && !is_vertex) ||
          sub->GetAlgoState() == SMESH_subMesh::HYP_OK ||
          (is_vertex && sub->GetAlgoState() != SMESH_subMesh::MISSING_HYP)) {
        continue;
      }
      const std::pair<const char*, int> at =
          ordinal_of_shape_index(meshDS_->ShapeToIndex(ex.Current()));
      bool seen = false;
      for (const auto& s : states) {
        seen = seen || (s.first.first == k && s.first.second == at.second);
      }
      if (seen || at.second <= 0) {
        continue;
      }
      std::string line = std::string(kKindNames[k]) + " " + std::to_string(at.second) + ": ";
      if (sub->GetAlgoState() == SMESH_subMesh::NO_ALGO) {
        if (!failed || !needs_own_algorithm(*mesh_, ex.Current())) {
          continue;
        }
        line += "no algorithm is assigned to it (algorithm state NO_ALGO)";
      } else {
        const SMESH_Algo* algo = sub->GetAlgo();
        line += std::string(algo != nullptr && algo->GetName() != nullptr ? algo->GetName()
                                                                          : "its algorithm") +
                " is missing a hypothesis it needs (algorithm state MISSING_HYP)";
      }
      states.push_back({{k, at.second}, line});
    }
  }

  if (failed || !states.empty()) {
    std::sort(states.begin(), states.end());
    for (const auto& s : states) {
      failures.push_back(s.second);
      if (kKindTypes[s.first.first] == TopAbs_FACE) {
        failed_faces.push_back(s.first.second);
      }
    }
    // SMESH can fail with no error text anywhere: a solid that failed first is reset by a
    // later sub-mesh event, which clears its error (a Prism_3D solid computed after a
    // NETGEN solid that shares a face with it). Each solid with an algorithm and no mesh
    // is then named.
    if (failures.empty()) {
      std::set<int> named;
      for (TopExp_Explorer ex(data_->shape, TopAbs_SOLID); ex.More(); ex.Next()) {
        SMESH_subMesh* sub = mesh_->GetSubMeshContaining(ex.Current());
        const SMESH_Algo* algo = sub != nullptr ? sub->GetAlgo() : nullptr;
        const int ordinal = ordinal_of_shape_index(meshDS_->ShapeToIndex(ex.Current())).second;
        if (algo == nullptr || algo->GetName() == nullptr || sub->IsMeshComputed() ||
            ordinal <= 0 || !named.insert(ordinal).second) {
          continue;
        }
        failures.push_back("SOLID " + std::to_string(ordinal) +
                           ": not meshed, and its algorithm left no error text (algorithm " +
                           algo->GetName() + ")" + prism_side_face_cause(ex.Current()));
      }
    }
    std::string details;
    for (std::size_t i = 0; i < failures.size(); ++i) {
      details += (i ? "\n" : "") + failures[i];
    }
    if (details.empty()) {
      details = "SMESH reported no per-sub-shape error text. The most common cause is an "
                "algorithm assigned to a sub-shape it cannot mesh at all.";
    }
    throw PysmeshError("Mesher.compute: meshing failed on " +
                           std::to_string(failures.size()) + " sub-shape(s).",
                       details, failed_faces, bad_centroids);
  }

  refuse_unfilled_netgen_solids(warnings);
  return success_report(warnings);
}

void Mesher::refuse_unfilled_netgen_solids(const py::list& warnings) const {
  // NETGEN_2D3D counts its result good when it holds any volume element
  // (NETGENPlugin_Mesher.cxx:3770). After viscous layers that stopped early it returned the
  // prisms alone, with 0.904 of 11.215 of the solid filled, and SMESH reported success.
  // So every solid that a NETGEN 3-D algorithm meshed must have a closed volume mesh.
  std::vector<std::string> lines;
  std::vector<double> centroids;
  std::set<int> checked;
  for (TopExp_Explorer ex(data_->shape, TopAbs_SOLID); ex.More(); ex.Next()) {
    SMESH_subMesh* sub = mesh_->GetSubMeshContaining(ex.Current());
    const SMESH_Algo* algo = sub != nullptr ? sub->GetAlgo() : nullptr;
    const std::string name = algo != nullptr && algo->GetName() != nullptr ? algo->GetName()
                                                                            : "";
    const int ordinal = ordinal_of_shape_index(meshDS_->ShapeToIndex(ex.Current())).second;
    if ((name != "NETGEN_3D" && name != "NETGEN_2D3D") || ordinal <= 0 ||
        !checked.insert(ordinal).second) {
      continue;
    }
    const OpenFront front = open_front(ex.Current());
    if (front.faces == 0) {
      continue;
    }
    lines.push_back("SOLID " + std::to_string(ordinal) +
                    ": its volume mesh does not fill it (algorithm " + name + ")" +
                    open_front_text(front));
    centroids.insert(centroids.end(), front.centroids.begin(), front.centroids.end());
  }
  if (lines.empty()) {
    return;
  }
  std::string details;
  for (const std::string& line : lines) {
    details += (details.empty() ? "" : "\n") + line;
  }
  std::string reported;
  for (const py::handle w : warnings) {
    const py::tuple t = w.cast<py::tuple>();
    reported += "\n" + t[0].cast<std::string>() + " " + std::to_string(t[1].cast<int>()) +
                " warned (algorithm " + t[2].cast<std::string>() +
                "): " + t[3].cast<std::string>();
  }
  details += reported;
  details += "\nNETGEN reported success, but the volume mesh of each solid named above has "
             "an open front, so part of the solid is empty. NETGEN_2D3D after viscous "
             "layers that stopped early (\"Thickness ... of viscous layers not reached\") "
             "returns the layers alone; Netgen1D2D with Netgen3D fills the same layers.";
  throw PysmeshError("Mesher.compute: " + std::to_string(lines.size()) +
                         " solid(s) meshed by NETGEN are not filled.",
                     details, {}, centroids);
}

py::dict Mesher::success_report(const py::list& warnings) const {
  py::dict out;
  out["nodes"] = static_cast<std::int64_t>(meshDS_->NbNodes());
  out["edges"] = static_cast<std::int64_t>(meshDS_->NbEdges());
  out["faces"] = static_cast<std::int64_t>(meshDS_->NbFaces());
  out["volumes"] = static_cast<std::int64_t>(meshDS_->NbVolumes());

  // Which sub-shapes actually received elements. A caller driving a mixed assignment needs
  // this to tell "meshed by the algorithm I put there" from "meshed by an enclosing one".
  py::list meshed;
  for (std::size_t k = 0; k < 4; ++k) {
    for (TopExp_Explorer ex(data_->shape, kKindTypes[k]); ex.More(); ex.Next()) {
      const SMESHDS_SubMesh* sub = meshDS_->MeshElements(ex.Current());
      if (sub == nullptr || sub->NbElements() == 0) {
        continue;
      }
      const std::pair<const char*, int> at =
          ordinal_of_shape_index(meshDS_->ShapeToIndex(ex.Current()));
      if (at.second > 0) {
        meshed.append(py::make_tuple(at.first, at.second,
                                     static_cast<std::int64_t>(sub->NbElements())));
      }
    }
  }
  out["meshed"] = meshed;
  out["warnings"] = warnings;
  return out;
}

py::dict Mesher::shape_free_outcome(bool ok) const {
  // The remesher reports on the pseudo-shape's sub-mesh. Its only warning ("No faces in
  // input mesh") cannot arise, since compute() refuses a mesh with no face first, so any
  // error text there is a failure.
  SMESH_subMesh* sub = mesh_->GetSubMesh(mesh_->GetShapeToMesh());
  const SMESH_ComputeErrorPtr err = sub != nullptr ? sub->GetComputeError() : nullptr;
  if (!ok || (err && !err->IsOK())) {
    std::string details =
        err && !err->myComment.empty() ? err->myComment : std::string("no message");
    if (err && err->myAlgo != nullptr && err->myAlgo->GetName() != nullptr) {
      details += std::string(" (algorithm ") + err->myAlgo->GetName() + ")";
    }
    throw PysmeshError("Mesher.compute: remeshing failed.", details);
  }
  py::dict out;
  out["nodes"] = static_cast<std::int64_t>(meshDS_->NbNodes());
  out["edges"] = static_cast<std::int64_t>(meshDS_->NbEdges());
  out["faces"] = static_cast<std::int64_t>(meshDS_->NbFaces());
  out["volumes"] = static_cast<std::int64_t>(meshDS_->NbVolumes());
  out["meshed"] = py::list();
  out["warnings"] = py::list();
  return out;
}

void Mesher::clear_mesh() {
  if (mesh_ != nullptr) {
    mesh_->Clear();
  }
}

int element_type_code(SMDSAbs_EntityType type) { return static_cast<int>(type); }

}  // namespace mesher
}  // namespace pysmesh
