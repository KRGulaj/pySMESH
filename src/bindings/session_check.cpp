// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH binding — Session: the checks over the live shape.
//
// Five read-only answers to "is this shape fit for the next operation": the argument check
// of a boolean (BOPAlgo_ArgumentAnalyzer), the validity of the bodies (BRepCheck_Analyzer),
// the short edges, whether a body's shells are closed, and the stored tolerances. Like every
// query they issue no id, take no OpGuard and do not touch the history; the two that can run
// long on a whole model take progress and cancel, and a cancel changes nothing because
// nothing was going to change.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include <BOPAlgo_ArgumentAnalyzer.hxx>
#include <BOPAlgo_CheckResult.hxx>
#include <BOPAlgo_CheckStatus.hxx>
#include <BOPAlgo_Operation.hxx>
#include <BRepCheck_Shell.hxx>
#include <Message_ProgressScope.hxx>
#include <ShapeAnalysis_FreeBounds.hxx>

#include "shape_checks.hpp"

namespace pysmesh {
namespace session {
namespace {

// The enumerator name of a BOPAlgo_CheckStatus, as BOPAlgo_CheckStatus.hxx spells it.
const char* check_status_name(BOPAlgo_CheckStatus status) {
  switch (status) {
    case BOPAlgo_CheckUnknown:
      return "BOPAlgo_CheckUnknown";
    case BOPAlgo_BadType:
      return "BOPAlgo_BadType";
    case BOPAlgo_SelfIntersect:
      return "BOPAlgo_SelfIntersect";
    case BOPAlgo_TooSmallEdge:
      return "BOPAlgo_TooSmallEdge";
    case BOPAlgo_NonRecoverableFace:
      return "BOPAlgo_NonRecoverableFace";
    case BOPAlgo_IncompatibilityOfVertex:
      return "BOPAlgo_IncompatibilityOfVertex";
    case BOPAlgo_IncompatibilityOfEdge:
      return "BOPAlgo_IncompatibilityOfEdge";
    case BOPAlgo_IncompatibilityOfFace:
      return "BOPAlgo_IncompatibilityOfFace";
    case BOPAlgo_OperationAborted:
      return "BOPAlgo_OperationAborted";
    case BOPAlgo_GeomAbs_C0:
      return "BOPAlgo_GeomAbs_C0";
    case BOPAlgo_InvalidCurveOnSurface:
      return "BOPAlgo_InvalidCurveOnSurface";
    case BOPAlgo_NotValid:
      return "BOPAlgo_NotValid";
  }
  return "BOPAlgo_CheckUnknown";
}

// The boolean whose argument rules a check applies. BOPAlgo_Operation has no splitter: a
// split is checked as the general builder's arguments are, BOPAlgo_UNKNOWN, which skips the
// operation-specific type rule and the face-rebuild test exactly as OCCT does for it.
BOPAlgo_Operation operation_of(const std::string& operation) {
  if (operation == "fuse") return BOPAlgo_FUSE;
  if (operation == "cut") return BOPAlgo_CUT;
  if (operation == "common") return BOPAlgo_COMMON;
  if (operation == "section") return BOPAlgo_SECTION;
  if (operation == "split") return BOPAlgo_UNKNOWN;
  throw PysmeshError("Session.check_boolean: the operation is '" + operation +
                     "'; it is one of 'fuse', 'cut', 'common', 'section' and 'split'.");
}

TopoDS_Shape compound_of(const std::vector<TopoDS_Shape>& shapes) {
  TopoDS_Compound c;
  BRep_Builder b;
  b.MakeCompound(c);
  for (const TopoDS_Shape& s : shapes) {
    b.Add(c, s);
  }
  return c;
}

}  // namespace

// Every live id of a shape; for a shape of a kind that carries no id (a shell, a wire, a
// compound), the ids of its sub-shapes of the highest dimension that carries ids.
std::vector<EntityId> Session::ids_within(const TopoDS_Shape& shape) const {
  std::vector<EntityId> out;
  const auto direct = state_.registry->by_shape.find(shape);
  if (direct != state_.registry->by_shape.end()) {
    out = direct->second;
    return out;
  }
  for (TopAbs_ShapeEnum kind : kEntityKinds) {
    ShapeSet subs;
    TopExp::MapShapes(shape, kind, subs);
    for (int i = 1; i <= subs.Extent(); ++i) {
      const auto it = state_.registry->by_shape.find(subs.FindKey(i));
      if (it != state_.registry->by_shape.end()) {
        out.insert(out.end(), it->second.begin(), it->second.end());
      }
    }
    if (!out.empty()) {
      break;
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

py::list Session::check_boolean(const std::string& operation,
                                const std::vector<EntityId>& arguments,
                                const std::vector<EntityId>& tools,
                                const py::object& progress, const py::object& cancel) const {
  const BOPAlgo_Operation op = operation_of(operation);
  if (arguments.empty()) {
    throw PysmeshError("Session.check_boolean: no argument is named; there is nothing to "
                       "check.");
  }
  if (tools.empty()) {
    throw PysmeshError("Session.check_boolean: no tool is named; a " + operation +
                       " has arguments and tools.");
  }
  std::vector<TopoDS_Shape> argument_shapes;
  for (EntityId id : arguments) {
    const EntityRecord& rec = require_alive("check_boolean", id);
    argument_shapes.insert(argument_shapes.end(), rec.shapes.begin(), rec.shapes.end());
  }
  std::vector<TopoDS_Shape> tool_shapes;
  for (EntityId id : tools) {
    const EntityRecord& rec = require_alive("check_boolean", id);
    tool_shapes.insert(tool_shapes.end(), rec.shapes.begin(), rec.shapes.end());
  }
  const TopoDS_Shape shape1 = compound_of(argument_shapes);
  const TopoDS_Shape shape2 = compound_of(tool_shapes);

  ProgressDriver driver("check_boolean", hooks_of("check_boolean", progress, cancel));
  NCollection_List<BOPAlgo_CheckResult> results;
  {
    py::gil_scoped_release release;
    try {
      BOPAlgo_ArgumentAnalyzer analyzer;
      analyzer.SetShape1(shape1);
      analyzer.SetShape2(shape2);
      analyzer.OperationType() = op;
      analyzer.StopOnFirstFaulty() = false;
      analyzer.ArgumentTypeMode() = true;
      analyzer.SelfInterMode() = true;
      analyzer.SmallEdgeMode() = true;
      analyzer.RebuildFaceMode() = true;
      analyzer.TangentMode() = true;
      analyzer.MergeVertexMode() = true;
      analyzer.MergeEdgeMode() = true;
      analyzer.ContinuityMode() = true;
      analyzer.CurveOnSurfaceMode() = true;
      analyzer.SetRunParallel(false);
      analyzer.Perform(driver.range());
      results = analyzer.GetCheckResult();
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.check_boolean: BOPAlgo_ArgumentAnalyzer "
                                     "threw: ") +
                         e.what());
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("check_boolean");
  }

  // The ids of a finding's shapes on one side: its faulty sub-shapes, or the whole operand
  // when OCCT names none (a type mismatch names the operands only).
  const auto side_ids = [&](const NCollection_List<TopoDS_Shape>& faulty,
                            const TopoDS_Shape& whole) {
    std::vector<EntityId> ids;
    if (faulty.IsEmpty()) {
      if (!whole.IsNull()) {
        ids = ids_within(whole);
      }
      return ids;
    }
    for (const TopoDS_Shape& s : faulty) {
      const std::vector<EntityId> found = ids_within(s);
      ids.insert(ids.end(), found.begin(), found.end());
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
  };
  py::list out;
  for (const BOPAlgo_CheckResult& r : results) {
    py::dict row;
    row["check"] = py::str(check_status_name(r.GetCheckStatus()));
    row["argument_ids"] = ids_array(side_ids(r.GetFaultyShapes1(), r.GetShape1()));
    row["tool_ids"] = ids_array(side_ids(r.GetFaultyShapes2(), r.GetShape2()));
    out.append(row);
  }
  return out;
}

py::list Session::check_shape(const std::optional<std::vector<EntityId>>& entity_ids,
                              const py::object& progress, const py::object& cancel) const {
  const std::vector<TopoDS_Shape> bodies =
      entity_ids.has_value() ? owner_bodies("check_shape", *entity_ids)
                             : root_bodies(state_.root);

  // BRepCheck_Analyzer takes no progress range, so the bodies are checked one at a time:
  // the progress advances per body and a cancel stops between two bodies.
  ProgressDriver driver("check_shape", hooks_of("check_shape", progress, cancel));
  std::vector<std::vector<shape_checks::CheckFinding>> per_body(bodies.size());
  {
    py::gil_scoped_release release;
    try {
      Message_ProgressScope scope(driver.range(), "check_shape",
                                  static_cast<double>(std::max<std::size_t>(bodies.size(), 1)));
      for (std::size_t i = 0; i < bodies.size() && scope.More(); ++i) {
        per_body[i] = shape_checks::check_findings(bodies[i]);
        scope.Next();
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.check_shape: BRepCheck_Analyzer threw: ") +
                         e.what());
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("check_shape");
  }

  // One record per sub-shape, its statuses merged over every context, in the order the
  // findings come: solids first, vertices last, each kind in TopExp::MapShapes order.
  py::list out;
  for (std::size_t b = 0; b < bodies.size(); ++b) {
    std::vector<TopoDS_Shape> subs;
    std::vector<std::vector<std::string>> statuses;
    for (const shape_checks::CheckFinding& f : per_body[b]) {
      std::size_t k = 0;
      while (k < subs.size() && !subs[k].IsSame(f.shape)) {
        ++k;
      }
      if (k == subs.size()) {
        subs.push_back(f.shape);
        statuses.emplace_back();
      }
      if (std::find(statuses[k].begin(), statuses[k].end(), f.status) == statuses[k].end()) {
        statuses[k].push_back(f.status);
      }
    }
    for (std::size_t k = 0; k < subs.size(); ++k) {
      // A shell or a wire carries no id: it is named by the solid or the face that holds it.
      TopoDS_Shape named = subs[k];
      const TopAbs_ShapeEnum kind = subs[k].ShapeType();
      if (kind == TopAbs_SHELL || kind == TopAbs_WIRE) {
        const TopAbs_ShapeEnum holder = kind == TopAbs_SHELL ? TopAbs_SOLID : TopAbs_FACE;
        for (TopExp_Explorer ex(bodies[b], holder); ex.More(); ex.Next()) {
          bool holds = false;
          for (TopExp_Explorer in(ex.Current(), kind); in.More() && !holds; in.Next()) {
            holds = in.Current().IsSame(subs[k]);
          }
          if (holds) {
            named = ex.Current();
            break;
          }
        }
      }
      const std::vector<EntityId> ids = ids_within(named);
      if (ids.empty()) {
        throw PysmeshError(std::string("Session.check_shape: an invalid ") +
                           shape_checks::kind_text(subs[k]) +
                           " of the session root carries no entity id. The registry and "
                           "the root have diverged; this is a bug, not a caller error.");
      }
      py::dict row;
      row["entity_id"] = ids.front();
      row["kind"] = py::str(shape_checks::kind_text(subs[k]));
      py::list names;
      for (const std::string& s : statuses[k]) {
        names.append(py::str(s));
      }
      row["statuses"] = names;
      out.append(row);
    }
  }
  return out;
}

py::array_t<std::int64_t> Session::small_edges(double threshold) const {
  finite_arg("small_edges", "threshold", threshold);
  if (!(threshold > 0.0)) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.6g", threshold);
    throw PysmeshError(std::string("Session.small_edges: the threshold is ") + text +
                       "; no edge is shorter than a length of zero or less.");
  }
  const std::vector<EntityId> ids = ids_of_kind(TopAbs_EDGE);
  std::vector<TopoDS_Shape> all;
  std::vector<std::size_t> first(ids.size() + 1, 0);
  for (std::size_t k = 0; k < ids.size(); ++k) {
    const EntityRecord& rec = state_.registry->alive.at(ids[k]);
    first[k] = all.size();
    all.insert(all.end(), rec.shapes.begin(), rec.shapes.end());
  }
  first[ids.size()] = all.size();
  std::vector<shape_checks::Measure> lengths;
  {
    py::gil_scoped_release release;
    lengths = shape_checks::measures(all, shape_checks::kDefaultMassPrecision);
  }
  std::vector<EntityId> out;
  for (std::size_t k = 0; k < ids.size(); ++k) {
    double length = 0.0;
    for (std::size_t j = first[k]; j < first[k + 1]; ++j) {
      shape_checks::require_measured(lengths[j], "Session.small_edges",
                                     "EDGE " + std::to_string(ids[k]));
      length += lengths[j].mass;
    }
    if (length < threshold) {
      out.push_back(ids[k]);
    }
  }
  return ids_array(out);
}

py::array_t<bool> Session::is_closed_shell(const std::vector<EntityId>& entity_ids) const {
  if (entity_ids.empty()) {
    throw PysmeshError("Session.is_closed_shell: no entity is named; there is no body to "
                       "test.");
  }
  const ShapeKeyed<TopoDS_Shape> owners = body_of_subshape();
  std::vector<TopoDS_Shape> bodies;
  for (EntityId id : entity_ids) {
    const EntityRecord& rec = require_alive("is_closed_shell", id);
    const auto it = owners.find(rec.shapes.front());
    if (it == owners.end()) {
      throw PysmeshError("Session.is_closed_shell: entity " + std::to_string(id) +
                         " belongs to no body of the session root.");
    }
    bodies.push_back(it->second);
  }
  const auto n = static_cast<py::ssize_t>(bodies.size());
  py::array_t<bool> out(n);
  bool* p = out.mutable_data();
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
      const TopoDS_Shape& body = bodies[static_cast<std::size_t>(i)];
      // Closed: no edge of the body's shells borders one face only (ShapeAnalysis_FreeBounds
      // on the shared topology, no sewing). Oriented: every shell's faces are oriented
      // consistently (BRepCheck_Shell::Orientation, which checks closure too). A body with
      // no shell, a free face, a wire, an edge or a vertex, is no closed shell.
      bool closed = false;
      for (TopExp_Explorer ex(body, TopAbs_SHELL); ex.More(); ex.Next()) {
        closed = true;
        BRepCheck_Shell check(TopoDS::Shell(ex.Current()));
        if (check.Orientation(true) != BRepCheck_NoError) {
          closed = false;
          break;
        }
      }
      if (closed) {
        const ShapeAnalysis_FreeBounds free(body, /*splitclosed=*/false, /*splitopen=*/false);
        closed = !TopoDS_Iterator(free.GetClosedWires()).More() &&
                 !TopoDS_Iterator(free.GetOpenWires()).More();
      }
      p[i] = closed;
    }
  }
  return out;
}

py::array_t<double> Session::tolerances(const std::vector<EntityId>& entity_ids) const {
  const auto n = static_cast<py::ssize_t>(entity_ids.size());
  py::array_t<double> out(n);
  double* p = out.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    const EntityId id = entity_ids[static_cast<std::size_t>(i)];
    const EntityRecord& rec = require_alive("tolerances", id);
    double tol = 0.0;
    for (const TopoDS_Shape& s : rec.shapes) {
      switch (s.ShapeType()) {
        case TopAbs_VERTEX:
          tol = std::max(tol, BRep_Tool::Tolerance(TopoDS::Vertex(s)));
          break;
        case TopAbs_EDGE:
          tol = std::max(tol, BRep_Tool::Tolerance(TopoDS::Edge(s)));
          break;
        case TopAbs_FACE:
          tol = std::max(tol, BRep_Tool::Tolerance(TopoDS::Face(s)));
          break;
        default:
          throw PysmeshError("Session.tolerances: entity " + std::to_string(id) + " is a " +
                             kind_name(rec.kind) +
                             ", which stores no tolerance; a face, an edge or a vertex does.");
      }
    }
    p[i] = tol;
  }
  return out;
}

}  // namespace session
}  // namespace pysmesh
