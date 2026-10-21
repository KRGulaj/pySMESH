// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH binding — Session: gluing coincident sub-shapes with GEOM's gluer.
//
// glue is GEOM's MakeGlueFaces and MakeGlueEdges (GEOMImpl_GlueDriver::GlueWithWarnings):
// GEOMAlgo_Gluer2 detects the faces, edges and vertices of the bodies that coincide within
// a tolerance (GEOMAlgo_GlueDetector) and builds the bodies again with one shape for each
// group. It intersects nothing: two faces are glued only when they coincide as a whole.
// fragment, by contrast, intersects every operand with every other, and splits faces that
// overlap in part.
//
// The history is the gluer's own: Modified gives the shape a glued group became, IsDeleted
// the shapes that left the result. Every id of a glued group therefore survives on the one
// shape of the group, as the third rule of entity ids has it for a merge.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include <GEOMAlgo_Gluer2.hxx>
#include <Message_ProgressScope.hxx>
#include <TopTools_DataMapOfShapeListOfShape.hxx>

namespace pysmesh {
namespace session {

namespace {

// The meaning of GEOMAlgo_Gluer2's error codes, from the list at the end of
// GEOMAlgo_Gluer2.cxx and its CheckData.
std::string gluer_error_text(int code) {
  switch (code) {
    case 11: return "GEOMAlgo_GlueDetector failed";
    case 13:
    case 14: return "PerformImagesToWork failed";
    case 20: return "the shapes are not of one kind";
    case 21: return "a shape is not a boundary representation";
    case 30: return "the shapes to glue are not consistent";
    default: return "an undocumented error";
  }
}

}  // namespace

py::dict Session::glue(const std::vector<EntityId>& entity_ids, double tolerance,
                       const std::string& kind, bool keep_non_solids,
                       const py::object& progress, const py::object& cancel) {
  const char* op = "glue";
  finite_arg(op, "tolerance", tolerance);
  OpGuard guard(in_op_);
  if (!(tolerance > 0.0)) {
    throw PysmeshError("Session.glue: the tolerance " + number_text(tolerance) +
                       " is not above zero; no two shapes coincide at that tolerance.");
  }
  if (kind != "faces" && kind != "edges") {
    throw PysmeshError("Session.glue: the kind '" + kind + "' is not one of faces, edges.");
  }
  const ProgressHooks hooks = hooks_of(op, progress, cancel);
  if (entity_ids.empty()) {
    throw PysmeshError("Session.glue: no entity is named; there is no body to glue.");
  }
  const std::vector<TopoDS_Shape> scope = scoped_bodies(entity_ids, op);
  const std::vector<TopoDS_Shape> untouched = bodies_excluding(scope);
  const TopoDS_Shape input = make_root(scope);

  // The bodies, by the ids they carry, for the refusals.
  std::string bodies;
  for (const TopoDS_Shape& body : scope) {
    const auto it = state_.registry->by_shape.find(body);
    const std::string name =
        it != state_.registry->by_shape.end() && !it->second.empty()
            ? std::string(kind_name(body.ShapeType())) + " " + std::to_string(it->second.front())
            : std::string("a ") + TopAbs::ShapeTypeToString(body.ShapeType()) + " body";
    bodies += (bodies.empty() ? "" : ", ") + name;
  }

  ProgressDriver driver(op, hooks);
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  int error = 0;
  int warning = 0;
  bool nothing = false;
  {
    py::gil_scoped_release release;
    try {
      Message_ProgressScope steps(driver.range(), "glue", 2.0);
      GEOMAlgo_Gluer2 gluer;
      gluer.SetArgument(input);
      gluer.SetTolerance(tolerance);
      gluer.SetKeepNonSolids(keep_non_solids);
      if (steps.More()) {
        gluer.Detect();
        error = gluer.ErrorStatus();
        warning = gluer.WarningStatus();
        steps.Next();
      }
      if (steps.More() && error == 0 && warning != 2) {
        if (kind == "edges") {
          // GEOM's MakeGlueEdges: glue only the detected groups of edges.
          TopTools_DataMapOfShapeListOfShape edges;
          for (TopTools_DataMapOfShapeListOfShape::Iterator it(gluer.ShapesDetected()); it.More();
               it.Next()) {
            if (it.Key().ShapeType() == TopAbs_EDGE) {
              edges.Bind(it.Key(), it.Value());
            }
          }
          nothing = edges.IsEmpty();
          gluer.SetShapesToGlue(edges);
        }
        if (!nothing) {
          gluer.Perform();
          error = gluer.ErrorStatus();
          warning = gluer.WarningStatus();
          nothing = error == 0 && warning == 1;
          if (error == 0 && !nothing) {
            result = gluer.Shape();
            NCollection_List<TopoDS_Shape> arguments;
            arguments.Append(input);
            hist = new BRepTools_History(arguments, gluer);
          }
        }
        steps.Next();
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.glue: OCCT threw: ") + e.what());
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled(op);
  }
  if (warning == 2) {
    throw PysmeshError("Session.glue: at the tolerance " + number_text(tolerance) +
                       ", sub-shapes of one shape of " + bodies +
                       " coincide with each other (GEOMAlgo_Gluer2 warning 2, sticked "
                       "shapes: the two vertices of an edge, for example). Nothing is "
                       "committed; the session is unchanged.");
  }
  if (error != 0) {
    throw PysmeshError("Session.glue: GEOMAlgo_Gluer2 failed with code " +
                       std::to_string(error) + " (" + gluer_error_text(error) + ") on " +
                       bodies + ".");
  }
  if (nothing || result.IsNull()) {
    throw PysmeshError("Session.glue: no " + kind + " of " + bodies +
                       " coincide within the tolerance " + number_text(tolerance) +
                       "; nothing is glued.");
  }
  return commit(concat(untouched, result), hist, op, result);
}

}  // namespace session
}  // namespace pysmesh
