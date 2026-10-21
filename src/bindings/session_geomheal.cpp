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
#include <ShHealOper_CloseContour.hxx>
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

// The ids, as "1, 2, 3".
std::string id_list(const std::vector<EntityId>& ids) {
  std::string out;
  for (EntityId id : ids) {
    out += (out.empty() ? "" : ", ") + std::to_string(id);
  }
  return out;
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

// ---- close_contour ------------------------------------------------------------------ //

namespace {

// The 3-D curve of an edge, without its location, for telling a copy of an edge (same
// curve) from a new edge.
Handle(Geom_Curve) curve_of(const TopoDS_Edge& edge) {
  TopLoc_Location loc;
  double first = 0.0;
  double last = 0.0;
  return BRep_Tool::Curve(edge, loc, first, last);
}

// The history of a contour closing, as the session names it: a new edge that closes a gap
// is generated from the edges whose ends it joins, never a modification of one of them.
//
// ShHealOper_CloseContour, closing a gap in a face, replaces the edge before the gap with a
// wire of two edges: a copy of it (on its own curve, with the new end vertex) and the new
// edge (ShHealOper_CloseContour::buildEdge). The context's history therefore reports the
// new edge as a modification of the edge before the gap. Here, for an input edge with
// several images, the images that are not on the input edge's own curve are taken out of
// its Modified list. (An edge with one image keeps it whatever its curve: ShapeFix_Wire's
// FixGaps3d gives the edges it pulls together new curves.) Every new edge of the result,
// that one and a free closing edge alike, is then recorded as Generated from each input
// edge whose image shares a vertex with it.
void adapt_closing_history(const Handle(BRepTools_History)& hist,
                           const std::vector<TopoDS_Edge>& inputs, const TopoDS_Shape& input,
                           const TopoDS_Shape& result) {
  ShapeSet images;  // the images of the input edges after the adaptation
  for (const TopoDS_Edge& e : inputs) {
    const NCollection_List<TopoDS_Shape>& mods = hist->Modified(e);
    if (mods.Extent() > 1) {
      std::vector<TopoDS_Shape> kept;
      for (const TopoDS_Shape& m : mods) {
        if (m.ShapeType() == TopAbs_EDGE && curve_of(TopoDS::Edge(m)) == curve_of(e)) {
          kept.push_back(m);
        }
      }
      if (kept.size() != 1) {
        throw PysmeshError("Session.close_contour: an edge has " +
                           std::to_string(mods.Extent()) + " images, of which " +
                           std::to_string(kept.size()) +
                           " are on its own curve; the copy of the edge cannot be told from "
                           "the closing edge. This is a bug, not a caller error.");
      }
      hist->ReplaceModified(e, kept.front());
    }
    if (hist->Modified(e).IsEmpty() && !hist->IsRemoved(e)) {
      images.Add(e);
    }
    for (const TopoDS_Shape& m : hist->Modified(e)) {
      images.Add(m);
    }
  }
  ShapeSet before;
  TopExp::MapShapes(input, TopAbs_EDGE, before);
  ShapeSet after;
  TopExp::MapShapes(result, TopAbs_EDGE, after);
  for (int i = 1; i <= after.Extent(); ++i) {
    const TopoDS_Shape& c = after.FindKey(i);
    if (before.Contains(c) || images.Contains(c)) {
      continue;
    }
    ShapeSet ends;
    TopExp::MapShapes(c, TopAbs_VERTEX, ends);
    for (std::size_t k = 0; k < inputs.size(); ++k) {
      const auto touches = [&ends](const TopoDS_Shape& edge) {
        for (TopExp_Explorer v(edge, TopAbs_VERTEX); v.More(); v.Next()) {
          if (ends.Contains(v.Current())) {
            return true;
          }
        }
        return false;
      };
      bool joined = false;
      if (hist->Modified(inputs[k]).IsEmpty()) {
        joined = touches(inputs[k]);
      }
      for (const TopoDS_Shape& m : hist->Modified(inputs[k])) {
        joined = joined || touches(m);
      }
      if (joined) {
        hist->AddGenerated(inputs[k], c);
      }
    }
  }
}

}  // namespace

py::dict Session::close_contour(const std::vector<EntityId>& edge_ids, bool common_vertex) {
  const char* op = "close_contour";
  OpGuard guard(in_op_);
  std::vector<EntityId> named;
  const std::vector<TopoDS_Shape> edges = edges_of(op, edge_ids, &named);
  const TopoDS_Shape body = sole_body(op, edge_ids);

  // The one wire, of the body's faces or of the free wire body, whose edges are exactly the
  // named ones.
  ShapeSet wanted;
  for (const TopoDS_Shape& e : edges) {
    wanted.Add(e);
  }
  TopoDS_Wire wire;
  int matches = 0;
  for (TopExp_Explorer w(body, TopAbs_WIRE); w.More(); w.Next()) {
    ShapeSet in_wire;
    TopExp::MapShapes(w.Current(), TopAbs_EDGE, in_wire);
    bool same = in_wire.Extent() == wanted.Extent();
    for (int i = 1; i <= in_wire.Extent() && same; ++i) {
      same = wanted.Contains(in_wire.FindKey(i));
    }
    if (same && (wire.IsNull() || !wire.IsSame(w.Current()))) {
      wire = TopoDS::Wire(w.Current());
      ++matches;
    }
  }
  if (body.ShapeType() == TopAbs_EDGE && wanted.Extent() == 1) {
    wire = BRepBuilderAPI_MakeWire(TopoDS::Edge(body)).Wire();
    matches = 1;
  }
  if (matches != 1) {
    throw PysmeshError("Session.close_contour: the edges " + id_list(edge_ids) +
                       " are not the edges of one wire of their body; a contour to close is "
                       "one open wire of a face, or a free wire.");
  }
  TopoDS_Vertex first, last;
  TopExp::Vertices(wire, first, last);
  if (!first.IsNull() && first.IsSame(last)) {
    throw PysmeshError("Session.close_contour: the wire of the edges " + id_list(edge_ids) +
                       " is closed already; there is no gap to close.");
  }

  const std::vector<TopoDS_Shape> untouched = bodies_excluding({body});
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  bool done = false;
  int status = 0;
  {
    py::gil_scoped_release release;
    try {
      // GEOM's CloseContour: Perform(wire, isCommonVertex, !isCommonVertex).
      ShHealOper_CloseContour closer(body);
      done = closer.Perform(wire, common_vertex, !common_vertex);
      status = closer.GetErrorStatus();
      if (done) {
        result = closer.GetResultShape();
        hist = closer.Context()->History();
        std::vector<TopoDS_Edge> inputs;
        for (const TopoDS_Shape& e : edges) {
          inputs.push_back(TopoDS::Edge(e));
        }
        adapt_closing_history(hist, inputs, body, result);
        // For a free wire ShHealOper returns the wire and the closing edges side by side in
        // a compound (ShHealOper_CloseContour::build adds myFreeEdges). They share their
        // end vertices, so one closed wire is built over the same edges.
        if (result.ShapeType() == TopAbs_COMPOUND &&
            (body.ShapeType() == TopAbs_WIRE || body.ShapeType() == TopAbs_EDGE)) {
          NCollection_List<TopoDS_Shape> parts;
          ShapeSet all;
          TopExp::MapShapes(result, TopAbs_EDGE, all);
          for (int i = 1; i <= all.Extent(); ++i) {
            parts.Append(all.FindKey(i));
          }
          BRepBuilderAPI_MakeWire joined;
          joined.Add(parts);
          ShapeSet kept;
          if (joined.IsDone()) {
            TopExp::MapShapes(joined.Wire(), TopAbs_EDGE, kept);
          }
          bool same = joined.IsDone() && kept.Extent() == all.Extent();
          for (int i = 1; i <= all.Extent() && same; ++i) {
            same = kept.Contains(all.FindKey(i));
          }
          if (!same) {
            throw PysmeshError("Session.close_contour: the closed contour's edges do not "
                               "join into one wire over the same edges; nothing is "
                               "committed. This is a bug, not a caller error.");
          }
          result = joined.Wire();
        }
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.close_contour: OCCT threw: ") + e.what());
    }
  }
  if (!done || result.IsNull()) {
    throw PysmeshError("Session.close_contour: ShHealOper_CloseContour closed nothing for the "
                       "edges " + id_list(edge_ids) + " (" + shheal_error_text(status) + ").");
  }
  return commit(concat(untouched, result), hist, op, result, Validation::Report);
}

}  // namespace session
}  // namespace pysmesh
