// pySMESH binding — the SMESH meshing surface: algorithms, hypotheses and the assignment
// model that decides which of them applies where.
//
// This is the counterpart of `Session` on the meshing side of the CAD/mesher boundary. A
// session owns a live shape and hands one BREP over at the handoff; a `Mesher` takes that
// shape back as a plain `Shape` — positional TopExp ordinals, exactly what the free-function
// API already speaks — and turns it into a volume or surface mesh.
//
// The one thing worth stating up front is that this is *not* "bind an algorithm". SMESH's
// value is its assignment model: an algorithm plus its hypotheses are attached to a
// sub-shape, and a sub-mesh resolves which of them applies to each part of the model. That is
// what lets one mesh be structured through an extruded region, body-fitted in the interior
// and layered at the walls, which a single global algorithm cannot express.
//
// Three hazards shape the code, all measured rather than assumed:
//
//   * **Hypothesis ids must come from SMESH_Gen::GetANewId().** Composite algorithms allocate
//     ids from the generator inside their own constructors, so a caller-side counter aliases
//     entries in the generator's maps. The symptom is remote from the cause — a later
//     assignment comes back HYP_ALREADY_EXIST.
//   * **SMESHDS shape indices are a private id space.** They are built by an *unfiltered*
//     TopExp walk over every kind at once, so they do not match a faces-only ordinal. Every
//     id crossing this boundary is a per-kind ordinal of the caller's own `Shape`, translated
//     through `ShapeData` in both directions; a SMESHDS index never reaches a signature.
//   * **Teardown order is load-bearing.** ~SMESH_Gen deletes the document and the hypotheses
//     but never the SMESH_Mesh wrapper, and ~SMESH_Mesh dereferences both. So: delete the
//     mesh, then the generator, then the hypotheses — the order mesh.cpp already establishes.
//
// File split, mirroring the session's:
//
//   mesher_core.cpp     ownership, sub-shape resolution, assignment, compute, error reporting
//   mesher_catalog.cpp  the algorithm and hypothesis factory
//   mesher_harvest.cpp  the mesh out: nodes, elements and their CAD binding, groups
//   mesher_gmf.cpp      Inria .mesh / .meshb read and write
//   mesher_bind.cpp     the pybind11 surface

#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <SMDSAbs_ElementType.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopoDS_Shape.hxx>

#include "../common.hpp"
#include "../session/progress.hpp"

class SMESHDS_Mesh;
class SMESH_Gen;
class SMESH_Hypothesis;
class SMESH_Mesh;

namespace pysmesh {
namespace mesher {

using session::ProgressHooks;

// ---- Parameter reader --------------------------------------------------------------- //
// The pybind surface takes a hypothesis name and a dict of its parameters, because the
// typed public API is the frozen dataclass on the Python side and duplicating 40 typed
// signatures here would give two places for one contract to drift.
//
// What keeps that honest is this reader: every key a factory branch consumes is recorded,
// and `done()` refuses anything left over. So a field added to a dataclass without a
// matching branch fails loud on the first call instead of being silently dropped.
class Params {
 public:
  Params(const char* owner, const py::dict& values) : owner_(owner), values_(values) {}

  double number(const char* key);
  int integer(const char* key);
  bool flag(const char* key);
  std::string text(const char* key);
  std::vector<double> numbers(const char* key);
  std::vector<int> integers(const char* key);

  // A ("KIND", ordinal) pair naming a sub-shape of the meshed shape, for the hypotheses that
  // point at one (a projection source, a base vertex). Ordinals, never SMESHDS indices.
  std::pair<std::string, int> subshape(const char* key);

  // A nested {"name": ..., "params": {...}} spec, for the hypotheses that carry another
  // hypothesis (a layer distribution is a 1-D hypothesis inside a 3-D one).
  py::dict nested(const char* key);

  // Raise unless every key has been consumed.
  void done() const;

 private:
  py::object take(const char* key);

  const char* owner_;
  py::dict values_;
  std::vector<std::string> consumed_;
};

// ---- Progress and cancellation ------------------------------------------------------ //
// SMESH has no Message_ProgressIndicator, so the OCCT-side driver does not fit: progress is
// *pulled* from SMESH_Mesh::GetComputeProgress() and a break is *pushed* into
// SMESH_Gen::CancelCompute(). Both are safe to call from another thread while Compute() runs
// — measured — which is what makes the same poll-from-a-helper-thread shape work here.
//
// Two limits are real and are stated rather than discovered:
//
//   * **Cancellation is not preemptive.** Only three StdMeshers algorithms poll the flag
//     inside their own loop; every other one can be broken only *between* sub-meshes. So the
//     latency is bounded by the longest single algorithm run, not by the poll interval.
//   * **Progress is exact only at sub-mesh granularity.** Within one algorithm SMESH falls
//     back to a tick counter that advances once per call, so the fraction reported inside a
//     running algorithm is a function of how often it is asked, not of the work done.
class ComputeDriver {
 public:
  ComputeDriver(SMESH_Mesh& mesh, SMESH_Gen& gen, const TopoDS_Shape& shape,
                const ProgressHooks& hooks);
  ~ComputeDriver();

  ComputeDriver(const ComputeDriver&) = delete;
  ComputeDriver& operator=(const ComputeDriver&) = delete;

  // Stop the poller, deliver the final position, and re-raise whatever a hook threw. Called
  // with the GIL held, after Compute() has returned.
  void finish();

  // True when a cancel was requested — by the predicate, or by a hook raising.
  //
  // This, and not Compute()'s own return value, is what decides a cancellation. A cancel
  // landing late gives a complete mesh and the same `false`, and an ordinary algorithm
  // failure gives `false` with no cancel at all.
  bool cancelled() const { return cancelled_.load(std::memory_order_relaxed); }

 private:
  void poll();
  void stop_thread();
  void request_cancel();

  SMESH_Mesh& mesh_;
  SMESH_Gen& gen_;
  TopoDS_Shape shape_;
  ProgressHooks hooks_;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::atomic<bool> cancelled_{false};
  bool stop_ = false;
  bool finished_ = false;
  double last_reported_ = -1.0;
  std::exception_ptr hook_error_;
};

// ---- Mesher ------------------------------------------------------------------------- //
class Mesher {
 public:
  explicit Mesher(const py::object& shape_obj);
  ~Mesher();

  Mesher(const Mesher&) = delete;
  Mesher& operator=(const Mesher&) = delete;

  // Attach an algorithm or a hypothesis to a sub-shape. An empty `kind` means the whole
  // shape, which is how a global assignment is expressed. Raises naming the offending
  // sub-shape when SMESH refuses the assignment.
  void assign(const std::string& name, const py::dict& params, const std::string& kind,
              int ordinal);

  // Detach a previously assigned algorithm or hypothesis from the same sub-shape.
  void unassign(const std::string& name, const std::string& kind, int ordinal);

  // Names of everything assigned, in assignment order, as (name, kind, ordinal) triples.
  py::list assignments() const;

  // Run the mesher. Raises on failure naming every sub-shape that failed, and on a cancel.
  // Returns the per-dimension element counts and the per-sub-shape state on success.
  py::dict compute(const py::object& progress, const py::object& cancel);

  // The mesh as arrays: node coordinates, an element CSR, the CAD binding of both, and the
  // per-face node split a polyhedron needs. Never advances anything.
  py::dict mesh_arrays() const;

  // The mesh's groups, as (name, element_type, ids) triples.
  py::list groups() const;

  void release();
  bool is_open() const { return mesh_ != nullptr; }

  // Internals seam for mesher_catalog.cpp / mesher_harvest.cpp / mesher_gmf.cpp.
  SMESH_Mesh& smesh() const;
  SMESH_Gen& gen() const;
  SMESHDS_Mesh& meshDS() const;
  const ShapeData& shape_data() const;

  // 1-based per-kind ordinal -> the sub-shape, and back. `kind` empty means the whole shape.
  const TopoDS_Shape& sub_shape(const std::string& kind, int ordinal) const;

  // A SMESHDS shape index -> the caller's own (kind, ordinal). Returns ("", 0) for an
  // element or node bound to nothing, and for a sub-shape kind the caller's Shape does not
  // index (a WIRE or a SHELL, which carry no elements of their own).
  std::pair<const char*, int> ordinal_of_shape_index(int shape_index) const;

  // Build a hypothesis or algorithm by name, with its id drawn from the generator. Defined
  // in mesher_catalog.cpp; ownership passes to the Mesher.
  SMESH_Hypothesis* build(const std::string& name, const py::dict& params);

 private:
  void ensure_open() const;
  void clear_mesh();

  struct Assignment {
    std::string name;
    std::string kind;
    int ordinal;
    int hyp_id;
  };

  std::shared_ptr<ShapeData> data_;
  std::unique_ptr<SMESH_Gen> gen_;
  SMESH_Mesh* mesh_ = nullptr;      // owned by gen_, deleted before it
  SMESHDS_Mesh* meshDS_ = nullptr;  // owned by mesh_
  std::vector<std::unique_ptr<SMESH_Hypothesis>> owned_;
  std::vector<Assignment> assigned_;

  // SMESHDS shape index -> (kind, ordinal), built once from ShapeData so a harvest of a
  // million elements does not do a map lookup per element through OCCT.
  std::vector<std::pair<const char*, int>> index_to_ordinal_;
  void build_index_map();
};

// The Python element-type code, which is SMDS's own entity type. Exposed unchanged because
// it is the complete cell-type space, including the quadratic and polyhedral cases, and
// inventing a second enum over it would only add a translation that can drift.
int element_type_code(SMDSAbs_EntityType type);

// The harvest, defined in mesher_harvest.cpp. `owner` may be null — a mesh read from a file
// has no shape, and then every element and node reports "bound to nothing" rather than
// inventing a binding.
py::dict harvest_arrays(const SMESHDS_Mesh& ds, const Mesher* owner);
py::list harvest_groups(const SMESHDS_Mesh& ds);

// Free functions defined in mesher_gmf.cpp.
py::dict read_gmf(const std::string& path);
void write_gmf(const std::string& path, const py::dict& mesh, const py::list& groups);

}  // namespace mesher
}  // namespace pysmesh
