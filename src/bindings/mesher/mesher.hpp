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
//   mesher_scratch.cpp  a shape-free mesh, and the rebuild of one from plain arrays
//   mesher_controls.cpp the quality controls: numerical functors, predicates, filter algebra
//   mesher_groups.cpp   named element and node groups
//   mesher_edit.cpp     the mesh editor
//   mesher_search.cpp   element search, ray casting, point classification, offset, slot
//   mesher_medial.cpp   the medial axis of a face
//   mesher_block.cpp    block decomposition and pattern mapping
//   mesher_gmf.cpp      Inria .mesh / .meshb read and write
//   mesher_bind.cpp     the pybind11 surface

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <SMDSAbs_ElementType.hxx>
#include <SMESH_Controls.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopoDS_Shape.hxx>

#include "../common.hpp"
#include "../session/progress.hpp"

class SMDS_Mesh;
class SMESHDS_GroupBase;
class SMESHDS_Mesh;
class SMESH_Gen;
class SMESH_Hypothesis;
class SMESH_Mesh;

namespace pysmesh {
namespace mesher {

using session::ProgressHooks;

// Every array crossing this boundary is built as a std::vector first, because none of the
// sizes is known before the walk that fills it.
template <class T>
py::array_t<T> vector_to_array(const std::vector<T>& values) {
  py::array_t<T> out(static_cast<py::ssize_t>(values.size()));
  if (!values.empty()) {
    std::copy(values.begin(), values.end(), out.mutable_data());
  }
  return out;
}

// The same, for a table: `values` is row-major and its length must be a whole number of rows.
template <class T>
py::array_t<T> rows_to_array(const std::vector<T>& values, py::ssize_t columns) {
  const py::ssize_t rows = columns > 0 ? static_cast<py::ssize_t>(values.size()) / columns : 0;
  py::array_t<T> out({rows, columns});
  if (!values.empty()) {
    std::copy(values.begin(), values.end(), out.mutable_data());
  }
  return out;
}

// The element family an id space belongs to, as SMESH counts them. Passed through unchanged
// because it is SMDS's own enumeration and a second one over it could only drift. Defined in
// mesher_core.cpp.
SMDSAbs_ElementType family_of(int code);

// A validated, C-contiguous float64 (N, columns) view of a Python array. Raises naming the
// argument rather than letting a wrong shape reach OCCT.
py::array_t<double, py::array::c_style | py::array::forcecast> point_table(
    const py::object& values, const char* name, int columns);

// A point or a direction given as three numbers, checked here so a two-element tuple fails
// with the argument's name rather than as an out-of-range read.
void require_triple(const std::vector<double>& values, const char* name);

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

  // Mesh ids, which are wider than an int by construction: SMDS numbers every node and
  // element in one 64-bit sequence.
  std::vector<std::int64_t> ids(const char* key);

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

  // The mesh's groups, as (name, element_family, source, ids) tuples.
  py::list groups() const;

  // ---- Quality controls --------------------------------------------------------------- //
  // One numerical control evaluated over every element it applies to, and one predicate
  // resolved to the ids that satisfy it. Both run on the live mesh, so the controls that
  // need the geometry — a deflection, a shape membership — work here and only here.
  py::dict quality(const std::string& name, const py::dict& params) const;
  py::dict select(const std::string& name, const py::dict& params) const;

  // ---- Groups ------------------------------------------------------------------------- //
  // Three kinds, which differ in what maintains their membership rather than in what they
  // are: an explicit id list that SMESH carries through editing, a set defined by a
  // sub-shape, and a set defined by a predicate and re-evaluated on demand.
  void add_group(const std::string& name, int family, const std::vector<std::int64_t>& ids);
  void add_group_on_shape(const std::string& name, int family, const std::string& kind,
                          int ordinal);
  void add_group_on_filter(const std::string& name, int family, const std::string& predicate,
                           const py::dict& params);
  void remove_group(const std::string& name);
  void edit_group(const std::string& name, const std::vector<std::int64_t>& ids, bool add);

  // ---- Editing (mesher_edit.cpp) -------------------------------------------------------- //
  // Every operation reports the four element counts either side of itself, because a mesh
  // edit is only meaningful against what it changed. An empty id list means "the whole
  // mesh", which is upstream's own convention for these calls.
  void convert_to_quadratic(bool force_3d, bool bi_quadratic);
  bool convert_from_quadratic();
  py::dict split_volumes(int method, double nx, double ny, double nz);
  py::dict split_quadratic_into_linear(const std::vector<std::int64_t>& elements);
  py::dict merge_nodes(double tolerance);
  py::list find_coincident_nodes(double tolerance, const std::vector<std::int64_t>& nodes,
                                 bool separate_corners_and_medium) const;
  py::dict merge_node_groups(const py::list& groups, bool avoid_making_holes);
  py::list find_equal_elements(const std::vector<std::int64_t>& elements) const;
  py::dict merge_equal_elements();
  py::dict smooth(int method, int iterations, double target_aspect_ratio, bool in_uv_space,
                  const std::vector<std::int64_t>& elements,
                  const std::vector<std::int64_t>& fixed_nodes);
  py::dict reorient(const std::vector<std::int64_t>& elements);
  py::dict reorient_2d(const std::vector<double>& direction,
                       const std::vector<std::int64_t>& faces,
                       const std::vector<std::int64_t>& reference_faces,
                       bool allow_non_manifold);
  py::dict reorient_2d_by_3d(const std::vector<std::int64_t>& faces,
                             const std::vector<std::int64_t>& volumes, bool outside_normal);
  py::dict quad_to_tri(const std::vector<std::int64_t>& elements, const std::string& criterion,
                       const py::dict& criterion_params, bool diagonal_13);
  py::dict tri_to_quad(const std::vector<std::int64_t>& elements, const std::string& criterion,
                       const py::dict& criterion_params, double max_angle);
  py::dict double_elements(const std::vector<std::int64_t>& elements);
  py::dict extrusion_sweep(const std::vector<std::int64_t>& elements,
                           const std::vector<double>& step, int steps, bool make_boundary,
                           double tolerance);
  py::dict rotation_sweep(const std::vector<std::int64_t>& elements,
                          const std::vector<double>& origin,
                          const std::vector<double>& direction, double angle, int steps,
                          double tolerance, bool make_walls);
  py::dict offset(double value, const std::vector<std::int64_t>& elements, bool copy_elements,
                  bool fix_self_intersection);
  py::dict sew_free_border(const std::vector<std::int64_t>& border,
                           const std::vector<std::int64_t>& side, bool side_is_free_border,
                           bool create_polygons, bool create_polyhedra);
  py::dict sew_side_elements(const std::vector<std::int64_t>& side1,
                             const std::vector<std::int64_t>& side2,
                             const std::vector<std::int64_t>& first_nodes,
                             const std::vector<std::int64_t>& second_nodes);

  // ---- Search and ray casting (mesher_search.cpp) --------------------------------------- //
  // Every query takes a batch of points, because one searcher builds an octree over the whole
  // mesh and answering one point with it would pay for the tree per question.
  py::dict find_elements_by_point(const py::object& points, int family) const;
  py::dict find_closest(const py::object& points, int family) const;
  py::dict elements_near_line(const std::vector<double>& origin,
                              const std::vector<double>& direction, int family) const;
  py::dict ray_hits(const std::vector<double>& origin, const std::vector<double>& direction,
                    double tolerance) const;
  py::dict elements_in_sphere(const std::vector<double>& centre, double radius,
                              int family) const;
  py::dict elements_in_box(const std::vector<double>& minimum,
                           const std::vector<double>& maximum, int family) const;
  py::dict point_state(const py::object& points) const;
  py::dict project_points(const py::object& points, int family) const;
  py::dict closest_distance(const py::object& points, int family) const;
  py::dict sharp_edges(double angle, bool add_existing) const;
  py::dict separate_faces_by_edges(const py::object& node1, const py::object& node2,
                                   const py::object& medium) const;
  py::dict de_merge(std::int64_t element, const py::list& groups) const;
  py::dict make_slot(double width, const std::vector<std::int64_t>& segments);

  // ---- Pattern mapping (mesher_block.cpp) ----------------------------------------------- //
  std::string pattern_from_face(int face_ordinal, bool project);
  py::dict apply_pattern_to_face(const std::string& text, int face_ordinal, int vertex_ordinal,
                                 bool reverse, bool create_polygons);
  py::dict apply_pattern_to_block(const std::string& text, int solid_ordinal, int vertex000,
                                  int vertex001, bool create_polyhedra);

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

  // The group of that name, or null. Defined in mesher_groups.cpp.
  SMESHDS_GroupBase* group_ds(const std::string& name) const;

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

// The groups, as (name, element_family, source, ids) tuples. Defined in mesher_groups.cpp.
py::list harvest_groups(const SMESHDS_Mesh& ds);

// ---- Scratch mesh (mesher_scratch.cpp) ------------------------------------------------ //
// A shape-free SMESH mesh, owned for the duration of one operation. The teardown order is
// the Mesher's own: the wrapper before the generator.
class ScratchMesh {
 public:
  ScratchMesh();
  ~ScratchMesh();

  ScratchMesh(const ScratchMesh&) = delete;
  ScratchMesh& operator=(const ScratchMesh&) = delete;

  SMESH_Mesh& mesh() const { return *mesh_; }
  SMESHDS_Mesh& ds() const;

 private:
  std::unique_ptr<SMESH_Gen> gen_;
  SMESH_Mesh* mesh_ = nullptr;
};

// Rebuild a mesh from the arrays a harvest produced, keeping every id. Raises naming the
// offending row on any disagreement between the arrays. Polyhedra and polygons have no
// arity-keyed constructor, so a mesh carrying one is refused by name rather than half-built.
void rebuild_mesh(SMESHDS_Mesh& ds, const py::dict& mesh);

// ---- Quality controls (mesher_controls.cpp) ------------------------------------------- //
// `owner` may be null. The controls that read the geometry — Deflection2D, ElementsOnShape —
// and the one that reads a group refuse a null owner by name rather than returning a value
// computed from nothing.
SMESH::Controls::NumericalFunctorPtr build_functor(const std::string& name,
                                                   const py::dict& params,
                                                   const Mesher* owner);
SMESH::Controls::PredicatePtr build_predicate(const std::string& name, const py::dict& params,
                                              const Mesher* owner);
py::dict evaluate_quality(const SMDS_Mesh& mesh, const Mesher* owner, const std::string& name,
                          const py::dict& params);
py::dict evaluate_selection(const SMDS_Mesh& mesh, const Mesher* owner,
                            const std::string& name, const py::dict& params);

// The same two over a mesh handed in as arrays, for a mesh that has no mesher behind it —
// one read from a file, or one a caller built.
py::dict mesh_quality(const py::dict& mesh, const std::string& name, const py::dict& params);
py::dict mesh_select(const py::dict& mesh, const std::string& name, const py::dict& params);

// ---- Medial axis and block decomposition ----------------------------------------------- //
// Both read the geometry rather than a mesh, so both take a `Shape` — the same positional
// ordinals the stateless API already speaks — and neither needs a mesher at all.

// The medial axis of one face: its branches, their end types, and per sample the two nearest
// boundary points, whose distance is the local width. Defined in mesher_medial.cpp.
py::dict medial_axis(const py::object& shape_obj, int face_ordinal, double min_segment_length,
                     bool ignore_corners, int samples);

// The 27 sub-shapes of a six-faced solid in block order, and the mapping between normalised
// block parameters and model space. Defined in mesher_block.cpp.
py::dict block_shapes(const py::object& shape_obj, int solid_ordinal, int vertex000,
                      int vertex001);
py::dict block_points(const py::object& shape_obj, int solid_ordinal, int vertex000,
                      int vertex001, const py::object& parameters);
py::dict block_parameters(const py::object& shape_obj, int solid_ordinal, int vertex000,
                          int vertex001, const py::object& points, double tolerance);

// Free functions defined in mesher_gmf.cpp.
py::dict read_gmf(const std::string& path);
void write_gmf(const std::string& path, const py::dict& mesh, const py::list& groups);

}  // namespace mesher
}  // namespace pysmesh
