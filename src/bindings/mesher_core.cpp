// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — the mesher's ownership, assignment model, compute and error reporting.
//
// See mesher/mesher.hpp for the file split and for the three hazards this code exists to
// keep away from the caller.

#include "mesher/mesher.hpp"

#include <algorithm>
#include <chrono>
#include <list>
#include <map>
#include <set>
#include <utility>

#include <SMDS_ElemIterator.hxx>
#include <SMDS_MeshElement.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_Algo.hxx>
#include <SMESH_ComputeError.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_HypoFilter.hxx>
#include <SMESH_Hypothesis.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_subMesh.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>

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
  // Only some algorithms build layers in their Compute: Hexa_3D, PolyhedronPerSolid_3D and
  // Cartesian_3D read ViscousLayers; Quadrangle_2D, QuadFromMedialAxis_1D2D and MEFISTO_2D
  // read ViscousLayers2D. The compatible lists do not tell: RadialQuadrangle_1D2D inherits
  // ViscousLayers2D from Quadrangle_2D and builds no layer. Any other algorithm meshes the
  // sub-shape with no layer and no word (Prism_3D, RadialQuadrangle_1D2D), fails after
  // building half of them (PolygonPerFace_2D), or crashed (CompositeHexa_3D).
  struct LayerKind {
    TopAbs_ShapeEnum type;
    const char* kind_name;
    const char* hypothesis;
    std::set<std::string> builders;
    const char* listed;
  };
  const LayerKind kinds[] = {
      {TopAbs_SOLID, "SOLID", "ViscousLayers",
       {"Hexa_3D", "PolyhedronPerSolid_3D", "Cartesian_3D"},
       "Hexa_3D, PolyhedronPerSolid_3D and Cartesian_3D"},
      {TopAbs_FACE, "FACE", "ViscousLayers2D",
       {"Quadrangle_2D", "QuadFromMedialAxis_1D2D", "MEFISTO_2D"},
       "Quadrangle_2D, QuadFromMedialAxis_1D2D and MEFISTO_2D"},
  };
  for (const LayerKind& kind : kinds) {
    SMESH_HypoFilter filter(SMESH_HypoFilter::HasName(kind.hypothesis));
    for (TopExp_Explorer ex(data_->shape, kind.type); ex.More(); ex.Next()) {
      if (mesh_->GetHypothesis(ex.Current(), filter, /*andAncestors=*/true) == nullptr) {
        continue;
      }
      SMESH_subMesh* sub = mesh_->GetSubMeshContaining(ex.Current());
      SMESH_Algo* algo = sub != nullptr ? sub->GetAlgo() : nullptr;
      if (algo == nullptr || algo->GetName() == nullptr) {
        continue;  // no algorithm of its own: the compute reports what is missing
      }
      const std::string name = algo->GetName();
      if (kind.builders.count(name) != 0) {
        continue;
      }
      const std::pair<const char*, int> at =
          ordinal_of_shape_index(meshDS_->ShapeToIndex(ex.Current()));
      throw PysmeshError(
          std::string("Mesher.compute: ") + kind.hypothesis + " reaches " +
              (at.first[0] != 0 ? at.first : kind.kind_name) + " " +
              std::to_string(at.second) + ", whose algorithm " + name +
              " does not build viscous layers.",
          std::string("Only ") + kind.listed + " build " + kind.hypothesis +
              "; with " + name + " the layers would be missing, or the compute would fail "
              "after building some. Assign one of those algorithms there, or assign the "
              "layers only to the sub-shapes such an algorithm meshes.");
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
  ensure_shape("Mesher.assign");
  const TopoDS_Shape& target = sub_shape(kind, ordinal);  // validates kind and ordinal

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
  assigned_.push_back({name, kind, ordinal, hyp_id});
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

void Mesher::unassign(const std::string& name, const std::string& kind, int ordinal) {
  ensure_open();
  ensure_shape("Mesher.unassign");
  const TopoDS_Shape& target = sub_shape(kind, ordinal);
  for (auto it = assigned_.begin(); it != assigned_.end(); ++it) {
    if (it->name != name || it->kind != kind || it->ordinal != ordinal) {
      continue;
    }
    const SMESH_Hypothesis::Hypothesis_Status status =
        mesh_->RemoveHypothesis(target, it->hyp_id);
    if (SMESH_Hypothesis::IsStatusFatal(status)) {
      throw PysmeshError("Mesher.unassign: SMESH refused to detach '" + name + "' from " +
                         where(kind, ordinal) + " — " + status_text(status) + ".");
    }
    assigned_.erase(it);
    return;
  }
  throw PysmeshError("Mesher.unassign: '" + name + "' is not assigned to " +
                     where(kind, ordinal) + ".");
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
  ensure_shape("Mesher.compute");
  if (assigned_.empty()) {
    throw PysmeshError("Mesher.compute: nothing is assigned. Assign at least an algorithm "
                       "before computing.");
  }

  refuse_unread_layers();

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
  ComputeDriver driver(*mesh_, *gen_, data_->shape, hooks);
  bool ok = false;
  if (!driver.cancelled()) {
    py::gil_scoped_release release;
    ok = gen_->Compute(*mesh_, data_->shape);
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
  if (driver.cancelled()) {
    clear_mesh();
    throw CancelledError("Mesher.compute: cancelled by the caller.",
                         "The mesh was cleared: nothing partial is returned. Cancellation is "
                         "not preemptive — only a few algorithms poll it inside their own "
                         "loop, so a long single algorithm runs to its end before stopping.");
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
      line += err->myComment.empty() ? std::string("no message") : err->myComment;
      // myAlgo is the algorithm object, not its name — naming it is what makes the message
      // actionable, because the failure is nearly always the algorithm rather than the shape.
      if (algorithm[0] != '\0') {
        line += std::string(" (algorithm ") + algorithm + ")";
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

  if (!ok || !failures.empty()) {
    // A missing algorithm, or an algorithm without the hypothesis it needs, is an algorithm
    // state of the sub-mesh, not a compute error: SMESH_Gen::Compute returns false and no
    // sub-mesh carries an error text (report A6: "failed on 0 sub-shape(s)"). So each
    // sub-mesh that was not computed is asked for its state. A VERTEX takes no algorithm of
    // its own, and NO_ALGO counts only where an enclosing algorithm needs this mesh.
    std::vector<std::pair<std::pair<std::size_t, int>, std::string>> states;
    for (std::size_t k = 0; k < 3; ++k) {
      for (TopExp_Explorer ex(data_->shape, kKindTypes[k]); ex.More(); ex.Next()) {
        SMESH_subMesh* sub = mesh_->GetSubMeshContaining(ex.Current());
        if (sub == nullptr || sub->IsMeshComputed() ||
            sub->GetAlgoState() == SMESH_subMesh::HYP_OK) {
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
          if (!needs_own_algorithm(*mesh_, ex.Current())) {
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
    std::sort(states.begin(), states.end());
    for (const auto& s : states) {
      failures.push_back(s.second);
      if (kKindTypes[s.first.first] == TopAbs_FACE) {
        failed_faces.push_back(s.first.second);
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
                       details, failed_faces);
  }

  return success_report(warnings);
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

void Mesher::clear_mesh() {
  if (mesh_ != nullptr) {
    mesh_->Clear();
  }
}

int element_type_code(SMDSAbs_EntityType type) { return static_cast<int>(type); }

}  // namespace mesher
}  // namespace pysmesh
