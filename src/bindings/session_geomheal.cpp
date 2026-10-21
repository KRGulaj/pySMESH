// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH binding — Session: GEOM's healing operators (ShHealOper).
//
// Each operation here is the session form of one GEOM healing operation
// (GEOMImpl_HealingDriver): divide_edge (DivideEdge), close_contour (CloseContour),
// fill_holes (FillHoles), suppress_faces (SuppressFaces) and shape_process (ShapeProcess).
// Every ShHealOper tool records its substitutions in a ShapeBuild_ReShape context
// (ShHealOper_Tool::Context), and BRepTools_ReShape::History turns that context into the
// history the registry carries ids across: an explicit Replace or Remove of the operator,
// and every container that ReShape::Apply rebuilt around it (BRepTools_ReShape::applyImpl
// records each one as Modified).
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include <GCPnts_AbscissaPoint.hxx>
#include <ShHealOper_EdgeDivide.hxx>
#include <TColStd_SequenceOfReal.hxx>

namespace pysmesh {
namespace session {

namespace {

// The text of a ShHealOper error status.
std::string shheal_error_text(int status) {
  switch (status) {
    case ShHealOper_NotError: return "no error was reported";
    case ShHealOper_InvalidParameters: return "invalid parameters";
    case ShHealOper_ErrorExecution: return "the execution failed";
    default: return "an undocumented status";
  }
}

// The bodies of the root that hold `sub`, and the others.
std::pair<std::vector<TopoDS_Shape>, std::vector<TopoDS_Shape>> split_bodies(
    const TopoDS_Shape& root, const TopoDS_Shape& sub) {
  std::vector<TopoDS_Shape> holding;
  std::vector<TopoDS_Shape> other;
  for (const TopoDS_Shape& body : root_bodies(root)) {
    ShapeSet parts;
    TopExp::MapShapes(body, sub.ShapeType(), parts);
    (parts.Contains(sub) ? holding : other).push_back(body);
  }
  return {holding, other};
}

}  // namespace

// ---- divide_edge -------------------------------------------------------------------- //

py::dict Session::divide_edge(EntityId edge_id, const std::vector<double>& positions,
                              const std::string& by) {
  const char* op = "divide_edge";
  require_finite(std::string("Session.") + op, "positions", positions.data(), positions.size(),
                 1);
  OpGuard guard(in_op_);
  if (by != "length" && by != "parameter") {
    throw PysmeshError("Session.divide_edge: by='" + by + "' is not one of length, parameter.");
  }
  const TopoDS_Edge edge = sole_edge(op, edge_id);
  if (positions.empty()) {
    throw PysmeshError("Session.divide_edge: no position is given for edge " +
                       std::to_string(edge_id) + "; there is nothing to divide.");
  }
  for (std::size_t i = 0; i < positions.size(); ++i) {
    if (!(positions[i] > 0.0 && positions[i] < 1.0)) {
      throw PysmeshError("Session.divide_edge: the position " + number_text(positions[i]) +
                         " for edge " + std::to_string(edge_id) +
                         " is not strictly between 0 and 1.");
    }
    if (i > 0 && !(positions[i] > positions[i - 1])) {
      throw PysmeshError("Session.divide_edge: the positions for edge " +
                         std::to_string(edge_id) + " are not strictly increasing (" +
                         number_text(positions[i - 1]) + " then " + number_text(positions[i]) +
                         ").");
    }
  }
  if (BRep_Tool::Degenerated(edge)) {
    throw PysmeshError("Session.divide_edge: edge " + std::to_string(edge_id) +
                       " is degenerate; it has no length to divide.");
  }

  const auto [holding, untouched] = split_bodies(state_.root, edge);
  const TopoDS_Shape input = make_root(holding);

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  bool done = false;
  int status = 0;
  std::string failed_at;
  {
    py::gil_scoped_release release;
    try {
      // The fractions of the parameter range. By length they are found here, by the
      // adaptive arc length at 1e-10 of the length (the tolerance overloads of
      // GCPnts_AbscissaPoint, as point_at_length uses them): ShHealOper's own length mode
      // integrates with a fixed Gauss rule.
      TColStd_SequenceOfReal values;
      BRepAdaptor_Curve curve(edge);
      const double first = curve.FirstParameter();
      const double last = curve.LastParameter();
      if (by == "parameter") {
        for (double p : positions) {
          values.Append(p);
        }
      } else {
        const double length = GCPnts_AbscissaPoint::Length(curve, first, last, 1e-10);
        for (double p : positions) {
          GCPnts_AbscissaPoint at(1e-10 * length, curve, p * length, first);
          if (!at.IsDone()) {
            failed_at = number_text(p);
            break;
          }
          values.Append((at.Parameter() - first) / (last - first));
        }
      }
      if (failed_at.empty()) {
        ShHealOper_EdgeDivide divider(input);
        done = divider.Perform(edge, values, /*theDivideParamMode=*/true);
        status = divider.GetErrorStatus();
        if (done) {
          result = divider.GetResultShape();
          hist = divider.Context()->History();
          // The vertices that divide the edge are generated from it.
          ShapeSet before;
          TopExp::MapShapes(input, TopAbs_VERTEX, before);
          for (const TopoDS_Shape& piece : hist->Modified(edge)) {
            for (TopExp_Explorer v(piece, TopAbs_VERTEX); v.More(); v.Next()) {
              if (!before.Contains(v.Current())) {
                hist->AddGenerated(edge, v.Current());
              }
            }
          }
        }
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.divide_edge: OCCT threw: ") + e.what());
    }
  }
  if (!failed_at.empty()) {
    throw PysmeshError("Session.divide_edge: OCCT found no point at the fraction " + failed_at +
                       " of the length of edge " + std::to_string(edge_id) + ".");
  }
  if (!done || result.IsNull()) {
    throw PysmeshError("Session.divide_edge: ShHealOper_EdgeDivide did not divide edge " +
                       std::to_string(edge_id) + " (" + shheal_error_text(status) + ").");
  }
  // Each position must give one piece more: a position the divider merged away (two
  // values closer than its tolerance) would leave the caller's pieces miscounted.
  ShapeSet pieces;
  for (const TopoDS_Shape& piece : hist->Modified(edge)) {
    pieces.Add(piece);
  }
  if (pieces.Extent() != static_cast<int>(positions.size()) + 1) {
    throw PysmeshError("Session.divide_edge: ShHealOper_EdgeDivide divided edge " +
                       std::to_string(edge_id) + " into " + std::to_string(pieces.Extent()) +
                       " pieces, not " + std::to_string(positions.size() + 1) +
                       "; nothing is committed.");
  }
  return commit(concat(untouched, result), hist, op, result, Validation::Strict, {},
                PieceRule::new_piece_ids(TopAbs_EDGE));
}

}  // namespace session
}  // namespace pysmesh
