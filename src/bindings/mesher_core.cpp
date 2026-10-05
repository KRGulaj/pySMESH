// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — the mesher's ownership, assignment model, compute and error reporting.
//
// See mesher/mesher.hpp for the file split and for the three hazards this code exists to
// keep away from the caller.

#include "mesher/mesher.hpp"

#include <algorithm>
#include <cctype>
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
#include <StdMeshers_ViscousLayers.hxx>
#include <StdMeshers_ViscousLayers2D.hxx>
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
  //
  // A building algorithm can still refuse the layer hypotheses that reach it: Hexa_3D
  // takes one ViscousLayers per solid (StdMeshers_Hexa_3D.cxx:136-147) and Cartesian_3D
  // keeps the last one it lists (StdMeshers_Cartesian_3D.cxx:114-125), and the layer check
  // of StdMeshers_ViscousLayers::CheckHypothesis refuses face sets that do not fit together.
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
       {"Hexa_3D", "PolyhedronPerSolid_3D", "Cartesian_3D"},
       "Hexa_3D, PolyhedronPerSolid_3D and Cartesian_3D",
       {"Hexa_3D", "Cartesian_3D"}},
      {TopAbs_FACE, "FACE", "ViscousLayers2D",
       {"Quadrangle_2D", "QuadFromMedialAxis_1D2D", "MEFISTO_2D"},
       "Quadrangle_2D, QuadFromMedialAxis_1D2D and MEFISTO_2D",
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
          "Only Quadrangle_2D, QuadFromMedialAxis_1D2D and MEFISTO_2D build "
          "ViscousLayers2D. With " + name + ", assign one of them on that face alone, "
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
  assigned_.push_back({name, kind, ordinal, hyp_id, params});
}

void Mesher::unassign(const std::string& name, const py::dict& params, const std::string& kind,
                      int ordinal) {
  ensure_open();
  ensure_shape("Mesher.unassign");
  const TopoDS_Shape& target = sub_shape(kind, ordinal);
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
