// pySMESH binding — Session: the boolean family, fillet and chamfer.
//
// Every BRepAlgoAPI_BuilderAlgo descendant shares its arguments, options, history and error
// channels, so one driver (run_bop) serves all six booleans; only the operand assignment and
// whether the operation replaces or extends the model differ.
//
// Fillet and chamfer sit here rather than with construction because they are the same kind
// of operation: a local rework of an existing body, driven by named edges, whose history is
// the only thing keeping the surrounding entity ids alive.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

namespace pysmesh {
namespace session {

// ---- modelling operations --------------------------------------------------------- //

py::dict Session::fuse(const std::vector<EntityId>& targets, const std::vector<EntityId>& tools,
                double fuzzy, bool parallel, const py::object& progress,
                const py::object& cancel) {
  OpGuard guard(in_op_);
  require_operands("fuse", targets, tools, fuzzy);
  BRepAlgoAPI_Fuse op;
  op.SetArguments(solids_of("fuse", "targets", targets));
  op.SetTools(solids_of("fuse", "tools", tools));
  return run_bop("fuse", op, bodies_of(targets, tools), /*additive=*/false, fuzzy, parallel,
                 hooks_of("fuse", progress, cancel));
}

py::dict Session::cut(const std::vector<EntityId>& targets, const std::vector<EntityId>& tools,
               double fuzzy, bool parallel, const py::object& progress,
               const py::object& cancel) {
  OpGuard guard(in_op_);
  require_operands("cut", targets, tools, fuzzy);
  BRepAlgoAPI_Cut op;
  op.SetArguments(solids_of("cut", "targets", targets));
  op.SetTools(solids_of("cut", "tools", tools));
  return run_bop("cut", op, bodies_of(targets, tools), /*additive=*/false, fuzzy, parallel,
                 hooks_of("cut", progress, cancel));
}

py::dict Session::common(const std::vector<EntityId>& targets,
                         const std::vector<EntityId>& tools, double fuzzy, bool parallel,
                         const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  require_operands("common", targets, tools, fuzzy);
  BRepAlgoAPI_Common op;
  op.SetArguments(solids_of("common", "targets", targets));
  op.SetTools(solids_of("common", "tools", tools));
  return run_bop("common", op, bodies_of(targets, tools), /*additive=*/false, fuzzy, parallel,
                 hooks_of("common", progress, cancel));
}

// The section curves of targets against tools. Additive: the result of a section is the
// intersection geometry alone, so both operand groups stay in the model and only the
// section's vertices and edges are added.
py::dict Session::section(const std::vector<EntityId>& targets,
                          const std::vector<EntityId>& tools, double fuzzy, bool parallel,
                          const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  require_operands("section", targets, tools, fuzzy);
  BRepAlgoAPI_Section op;
  op.SetArguments(solids_of("section", "targets", targets));
  op.SetTools(solids_of("section", "tools", tools));
  return run_bop("section", op, {}, /*additive=*/true, fuzzy, parallel,
                 hooks_of("section", progress, cancel));
}

// Split the targets by the tools. The tools are not consumed: OCCT's Splitter excludes
// their split parts from the result, and a tool the caller still holds ids for must not
// disappear from the model as a side effect.
py::dict Session::split(const std::vector<EntityId>& targets,
                        const std::vector<EntityId>& tools, double fuzzy, bool parallel,
                        const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  require_operands("split", targets, tools, fuzzy);
  BRepAlgoAPI_Splitter op;
  op.SetArguments(solids_of("split", "targets", targets));
  op.SetTools(solids_of("split", "tools", tools));
  return run_bop("split", op, bodies_of(targets, {}), /*additive=*/false, fuzzy, parallel,
                 hooks_of("split", progress, cancel));
}

// The general fuse: every operand is split by every other and the result keeps all the
// pieces. This is the operation a conformal multi-body CFD domain is built with.
py::dict Session::fragment(const std::vector<EntityId>& entity_ids, double fuzzy,
                           bool parallel, const py::object& progress,
                           const py::object& cancel) {
  OpGuard guard(in_op_);
  if (entity_ids.size() < 2) {
    throw PysmeshError("Session.fragment: at least two solids are required (got " +
                       std::to_string(entity_ids.size()) + ").");
  }
  require_fuzzy("fragment", fuzzy);
  BRepAlgoAPI_BuilderAlgo op;
  op.SetArguments(solids_of("fragment", "entity_ids", entity_ids));
  return run_bop("fragment", op, bodies_of(entity_ids, {}), /*additive=*/false, fuzzy,
                 parallel, hooks_of("fragment", progress, cancel));
}

// ---- fillet and chamfer ----------------------------------------------------------- //

// radius_end, when given, makes the radius evolve linearly along each named edge from
// radius to radius_end (OCCT's two-radius Add).
py::dict Session::fillet(const std::vector<EntityId>& edge_ids, double radius,
                  const std::optional<double>& radius_end, const py::object& progress,
                  const py::object& cancel) {
  OpGuard guard(in_op_);
  require_positive("radius", radius);
  if (radius_end.has_value()) {
    require_positive("radius_end", *radius_end);
  }
  std::vector<EntityId> kept;
  const std::vector<TopoDS_Shape> edges = edges_of("fillet", edge_ids, &kept);

  // OCCT's fillet takes edges and derives the owning solid itself, so the caller never has
  // to co-select the solid or a reference face per edge. Every named edge must belong to
  // the same body, because one fillet operation builds one shape.
  const TopoDS_Shape owner = sole_owner_body(body_of_subshape(), edges);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({owner});

  ProgressDriver driver("fillet", hooks_of("fillet", progress, cancel));
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::vector<TopoDS_Shape> faulty;
  {
    py::gil_scoped_release release;
    BRepFilletAPI_MakeFillet mk(owner);
    for (const TopoDS_Shape& e : edges) {
      if (radius_end.has_value()) {
        mk.Add(radius, *radius_end, TopoDS::Edge(e));
      } else {
        mk.Add(radius, TopoDS::Edge(e));
      }
    }
    try {
      mk.Build(driver.range());
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(
          std::string("Session.fillet: BRepFilletAPI_MakeFillet::Build failed: ") +
          e.what());
    }
    if (mk.IsDone()) {
      result = mk.Shape();
      hist = history_of(owner, mk);
    } else {
      faulty = faulty_edges(mk);
    }
  }
  driver.finish();
  // Before the failure path, not after it: an operation the caller stopped has no faulty
  // edges to blame, and reporting the radius as unbuildable would be a false diagnostic.
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("fillet");
  }
  if (result.IsNull()) {
    throw PysmeshError(
        "Session.fillet: OCCT could not build a fillet of radius " +
            std::to_string(radius) + " on the named edges.",
        "BRepFilletAPI_MakeFillet::IsDone() is false. The radius is most likely larger "
        "than the local geometry admits. No partial result is returned.",
        blamed_ids(kept, edges, faulty));
  }
  return commit(concat(survivors, result), hist, "fillet", result);
}

// distance_end + face_id give OCCT's two-distance chamfer, where the first distance is
// measured on the named reference face. That is the only form in OCCT 8.0 that takes a
// face at all — there is no (distance, edge, face) overload.
py::dict Session::chamfer(const std::vector<EntityId>& edge_ids, double distance,
                   const std::optional<double>& distance_end,
                   const std::optional<EntityId>& face_id, const py::object& progress,
                   const py::object& cancel) {
  OpGuard guard(in_op_);
  require_positive("distance", distance);
  if (distance_end.has_value()) {
    require_positive("distance_end", *distance_end);
  }
  if (face_id.has_value() != distance_end.has_value()) {
    throw PysmeshError(
        "Session.chamfer: face_id and distance_end must be given together. OCCT 8.0's "
        "only face-aware chamfer is the two-distance form.");
  }
  const std::vector<TopoDS_Shape> edges = edges_of("chamfer", edge_ids);

  TopoDS_Face reference;
  if (face_id.has_value()) {
    const EntityRecord& rec = require_alive("chamfer", *face_id);
    if (rec.kind != TopAbs_FACE) {
      throw PysmeshError("Session.chamfer: entity " + std::to_string(*face_id) +
                         " is a " + kind_name(rec.kind) + ", not a FACE.");
    }
    if (rec.shapes.size() != 1) {
      throw PysmeshError("Session.chamfer: face " + std::to_string(*face_id) +
                         " was split and denotes several faces; name one of them.");
    }
    reference = TopoDS::Face(rec.shapes.front());
  }

  std::vector<TopoDS_Shape> selection = edges;
  if (!reference.IsNull()) {
    selection.push_back(reference);
  }
  const TopoDS_Shape owner = sole_owner_body(body_of_subshape(), selection);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({owner});

  ProgressDriver driver("chamfer", hooks_of("chamfer", progress, cancel));
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::vector<TopoDS_Shape> faulty;
  {
    py::gil_scoped_release release;
    BRepFilletAPI_MakeChamfer mk(owner);
    for (const TopoDS_Shape& e : edges) {
      if (reference.IsNull()) {
        mk.Add(distance, TopoDS::Edge(e));
      } else {
        mk.Add(distance, *distance_end, TopoDS::Edge(e), reference);
      }
    }
    try {
      mk.Build(driver.range());
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(
          std::string("Session.chamfer: BRepFilletAPI_MakeChamfer::Build failed: ") +
          e.what());
    }
    if (mk.IsDone()) {
      result = mk.Shape();
      hist = history_of(owner, mk);
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("chamfer");
  }
  if (result.IsNull()) {
    throw PysmeshError(
        "Session.chamfer: OCCT could not build a chamfer of distance " +
            std::to_string(distance) + " on the named edges.",
        "BRepFilletAPI_MakeChamfer::IsDone() is false. The distance is most likely "
        "larger than the adjacent faces admit. No partial result is returned.",
        ids_as_int(edge_ids));
  }
  return commit(concat(survivors, result), hist, "chamfer", result);
}

// The fuzzy value is the one tolerance a caller supplies to a boolean, and it is checked
// rather than forwarded.
//
// `!(fuzzy >= 0.0)` rather than `fuzzy < 0.0` is the point of the rewrite: every comparison
// against NaN is false, so the negated form let NaN straight through to SetFuzzyValue, where
// it becomes a tolerance nothing compares equal to and the boolean's behaviour is undefined.
// Infinity is refused for the same reason — it is not a distance.
void Session::require_fuzzy(const char* op, double fuzzy) {
  if (!(fuzzy >= 0.0) || !std::isfinite(fuzzy)) {
    throw PysmeshError(std::string("Session.") + op +
                       ": fuzzy must be a finite value >= 0 (got " + std::to_string(fuzzy) +
                       ").");
  }
}

void Session::require_operands(const char* op, const std::vector<EntityId>& targets,
                               const std::vector<EntityId>& tools, double fuzzy) {
  if (targets.empty()) {
    throw PysmeshError(std::string("Session.") + op +
                       ": targets must name at least one solid.");
  }
  if (tools.empty()) {
    throw PysmeshError(std::string("Session.") + op +
                       ": tools must name at least one solid.");
  }
  require_fuzzy(op, fuzzy);
}

// The root bodies owning the solids two operand lists name. Resolved through the
// sub-shape -> body map rather than by comparing the solids themselves, so a solid nested
// inside a compound body still removes the right body from the model.
std::vector<TopoDS_Shape> Session::bodies_of(const std::vector<EntityId>& a,
                                      const std::vector<EntityId>& b) const {
  const ShapeKeyed<TopoDS_Shape> owners = body_of_subshape();
  std::vector<TopoDS_Shape> out;
  for (const std::vector<EntityId>* list : {&a, &b}) {
    for (EntityId id : *list) {
      const EntityRecord& rec = require_alive("boolean", id);
      for (const TopoDS_Shape& s : rec.shapes) {
        const auto it = owners.find(s);
        if (it != owners.end()) {
          append_unique(out, it->second);
        }
      }
    }
  }
  return out;
}

// One driver for the whole BOP family. Every BRepAlgoAPI_BuilderAlgo descendant shares
// the arguments, options, history and error channels, so only the operand assignment (on
// the concrete type, because SetTools is not virtual) and whether the operation replaces
// or extends the model differ. `op` must already carry its arguments and tools.
py::dict Session::run_bop(const char* op_name, BRepAlgoAPI_BuilderAlgo& op,
                   const std::vector<TopoDS_Shape>& consumed, bool additive, double fuzzy,
                   bool parallel, const ProgressHooks& hooks) {
  // Bodies the boolean does not consume pass straight through and keep their identity.
  const std::vector<TopoDS_Shape> survivors =
      additive ? root_bodies(state_.root) : bodies_excluding(consumed);

  // Constructed under the GIL, before it is released, and destroyed after it is back: the
  // driver holds Python references and starts the thread that touches them.
  ProgressDriver driver(op_name, hooks);
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::string errors;
  {
    py::gil_scoped_release release;
    // The history IS the naming substrate, not a diagnostic: without it every id in the
    // consumed bodies would die at this operation.
    op.SetToFillHistory(true);
    // Not an option, and not a performance knob. BOPAlgo's default is destructive: it
    // updates the argument TShapes in place. A session's whole snapshot contract rests on
    // an operation never mutating a shape an earlier state still points at, so every
    // boolean runs non-destructively and OCCT copies what it needs to change. It is
    // deliberately not a parameter: exposing it would let a caller switch off the property
    // every retained snapshot depends on.
    op.SetNonDestructive(true);
    op.SetRunParallel(parallel);
    if (fuzzy > 0.0) {
      op.SetFuzzyValue(fuzzy);
    }
    try {
      op.Build(driver.range());
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.") + op_name + ": OCCT's boolean threw: " +
                         e.what());
    }
    if (!op.IsDone() || op.HasErrors()) {
      std::ostringstream s;
      op.DumpErrors(s);
      errors = s.str();
    } else {
      result = op.Shape();
      hist = op.History();
    }
  }
  driver.finish();
  // A cancelled boolean records its own BOPAlgo_AlertUserBreak, so it lands in the failure
  // path above with an error string. It is caught here first, because "the caller stopped
  // it" and "the geometry defeated it" need different handling and the message text is not
  // a safe way to tell them apart.
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled(op_name);
  }
  if (!errors.empty() || result.IsNull()) {
    throw PysmeshError(std::string("Session.") + op_name +
                           ": the boolean failed; no partial result is returned.",
                       errors, {});
  }
  return commit(concat(survivors, result), hist, op_name, result);
}

// The edges of every contour OCCT could not build a fillet on. This is the diagnostic
// that turns "the fillet failed" into "the fillet failed on these edges"; when the
// builder cannot report it, the caller is blamed for every edge it named instead.
std::vector<TopoDS_Shape> Session::faulty_edges(BRepFilletAPI_MakeFillet& mk) {
  std::vector<TopoDS_Shape> out;
  try {
    for (int i = 1; i <= mk.NbFaultyContours(); ++i) {
      const int contour = mk.FaultyContour(i);
      for (int j = 1; j <= mk.NbEdges(contour); ++j) {
        out.push_back(mk.Edge(contour, j));
      }
    }
  } catch (const std::exception&) {
    out.clear();
  }
  return out;
}

// Entity ids for the named edges that OCCT blamed, or all of them when it blamed none.
std::vector<int> Session::blamed_ids(const std::vector<EntityId>& edge_ids,
                                     const std::vector<TopoDS_Shape>& edges,
                                     const std::vector<TopoDS_Shape>& faulty) {
  if (faulty.empty()) {
    return ids_as_int(edge_ids);
  }
  std::vector<int> out;
  for (std::size_t i = 0; i < edges.size() && i < edge_ids.size(); ++i) {
    for (const TopoDS_Shape& f : faulty) {
      if (edges[i].IsSame(f)) {
        out.push_back(static_cast<int>(edge_ids[i]));
        break;
      }
    }
  }
  return out.empty() ? ids_as_int(edge_ids) : out;
}

NCollection_List<TopoDS_Shape> Session::solids_of(const char* op, const char* argname,
                                           const std::vector<EntityId>& ids) const {
  NCollection_List<TopoDS_Shape> out;
  for (EntityId id : ids) {
    const EntityRecord& rec = require_alive(op, id);
    if (rec.kind != TopAbs_SOLID) {
      throw PysmeshError(std::string("Session.") + op + ": " + argname + " entity " +
                         std::to_string(id) + " is a " + kind_name(rec.kind) +
                         ", not a SOLID.");
    }
    for (const TopoDS_Shape& s : rec.shapes) {
      out.Append(s);
    }
  }
  return out;
}

}  // namespace session
}  // namespace pysmesh
