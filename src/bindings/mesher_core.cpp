// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — the mesher's ownership, assignment model, compute and error reporting.
//
// See mesher/mesher.hpp for the file split and for the three hazards this code exists to
// keep away from the caller.

#include "mesher/mesher.hpp"

#include <chrono>
#include <map>
#include <set>
#include <utility>

#include <SMDS_ElemIterator.hxx>
#include <SMDS_MeshElement.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_Algo.hxx>
#include <SMESH_ComputeError.hxx>
#include <SMESH_Gen.hxx>
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

std::string where(const std::string& kind, int ordinal) {
  if (kind.empty()) {
    return "the whole shape";
  }
  return kind + " " + std::to_string(ordinal);
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

double Params::number(const char* key) { return take(key).cast<double>(); }
int Params::integer(const char* key) { return take(key).cast<int>(); }
bool Params::flag(const char* key) { return take(key).cast<bool>(); }
std::string Params::text(const char* key) { return take(key).cast<std::string>(); }

std::vector<double> Params::numbers(const char* key) {
  return take(key).cast<std::vector<double>>();
}

std::vector<int> Params::integers(const char* key) {
  return take(key).cast<std::vector<int>>();
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
  assigned_.push_back({name, kind, ordinal, hyp_id});
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

  ComputeDriver driver(*mesh_, *gen_, data_->shape, hooks);
  bool ok = false;
  if (!driver.cancelled()) {
    py::gil_scoped_release release;
    ok = gen_->Compute(*mesh_, data_->shape);
  }
  driver.finish();

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
  std::vector<std::string> failures;
  std::vector<int> failed_faces;
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
      std::string line = std::string(at.first[0] ? at.first : kKindNames[k]) + " " +
                         std::to_string(at.second) + ": ";
      line += err->myComment.empty() ? std::string("no message") : err->myComment;
      // myAlgo is the algorithm object, not its name — naming it is what makes the message
      // actionable, because the failure is nearly always the algorithm rather than the shape.
      if (err->myAlgo != nullptr && err->myAlgo->GetName() != nullptr) {
        line += std::string(" (algorithm ") + err->myAlgo->GetName() + ")";
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
