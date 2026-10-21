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
#include <GEOMAlgo_AlgoTools.hxx>
#include <Message_Messenger.hxx>
#include <Message_Printer.hxx>
#include <NCollection_DataMap.hxx>
#include <Resource_Manager.hxx>
#include <ShHealOper_CloseContour.hxx>
#include <ShHealOper_EdgeDivide.hxx>
#include <ShHealOper_FillHoles.hxx>
#include <ShHealOper_RemoveFace.hxx>
#include <ShapeAnalysis_ShapeTolerance.hxx>
#include <ShapeProcessAPI_ApplySequence.hxx>
#include <ShapeProcess_ShapeContext.hxx>
#include <TColStd_SequenceOfReal.hxx>
#include <TNaming_CopyShape.hxx>
#include <TopTools_SequenceOfShape.hxx>

#include <cstdio>
#include <set>

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

}  // namespace

// The history `hist` of an operation on `input`, keeping only the images of the kind of
// the shape they are images of.
//
// BRepTools_ReShape::History replaces a shape that a wire or a shell replaced by the edges
// or faces of it (its Add helper explodes the two containers). That is right for an edge
// divided into a wire of pieces, and wrong for a solid that an operation left as an open
// shell: the solid would be "modified" into faces, and its id would carry over onto them.
// An id never changes kind. So the Modified images of another kind are dropped, and a shape
// that keeps none dies with the operation, as one that is no longer in the model does.
// Generated and Removed relations are kept as they are.
Handle(BRepTools_History) kind_preserving(const Handle(BRepTools_History)& hist,
                                          const TopoDS_Shape& input) {
  Handle(BRepTools_History) out = new BRepTools_History;
  for (TopAbs_ShapeEnum k : kEntityKinds) {
    ShapeSet subs;
    TopExp::MapShapes(input, k, subs);
    for (int i = 1; i <= subs.Extent(); ++i) {
      const TopoDS_Shape& s = subs.FindKey(i);
      if (hist->IsRemoved(s)) {
        out->Remove(s);
        continue;
      }
      for (const TopoDS_Shape& m : hist->Modified(s)) {
        if (m.ShapeType() == k) {
          out->AddModified(s, m);
        }
      }
      for (const TopoDS_Shape& g : hist->Generated(s)) {
        out->AddGenerated(s, g);
      }
    }
  }
  return out;
}

namespace {

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
          hist = kind_preserving(divider.Context()->History(), input);
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
        hist = kind_preserving(closer.Context()->History(), body);
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

// ---- fill_holes --------------------------------------------------------------------- //

py::dict Session::fill_holes(const std::vector<EntityId>& entity_ids,
                             const std::optional<std::vector<EntityId>>& edge_ids) {
  const char* op = "fill_holes";
  OpGuard guard(in_op_);
  if (entity_ids.empty()) {
    throw PysmeshError("Session.fill_holes: no entity is named; there is no body to fill.");
  }
  const std::vector<TopoDS_Shape> scope = scoped_bodies(entity_ids, op);
  const std::vector<TopoDS_Shape> untouched = bodies_excluding(scope);
  const TopoDS_Shape input = make_root(scope);

  // The edges that bound the holes to fill: free boundary edges of the bodies, each held by
  // one face of them.
  std::vector<TopoDS_Shape> bounds;
  if (edge_ids.has_value()) {
    std::vector<EntityId> kept;
    bounds = edges_of(op, *edge_ids, &kept);
    NCollection_IndexedDataMap<TopoDS_Shape, NCollection_List<TopoDS_Shape>,
                               TopTools_ShapeMapHasher>
        edge_faces;
    TopExp::MapShapesAndUniqueAncestors(input, TopAbs_EDGE, TopAbs_FACE, edge_faces);
    for (std::size_t i = 0; i < bounds.size(); ++i) {
      const int k = edge_faces.FindIndex(bounds[i]);
      if (k == 0) {
        throw PysmeshError("Session.fill_holes: edge " + std::to_string(kept[i]) +
                           " is not an edge of a face of the bodies of the named entities.");
      }
      if (edge_faces.FindFromIndex(k).Extent() != 1) {
        throw PysmeshError("Session.fill_holes: edge " + std::to_string(kept[i]) + " is held by " +
                           std::to_string(edge_faces.FindFromIndex(k).Extent()) +
                           " faces; a hole is bounded by free edges, each held by one face.");
      }
    }
  }

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  bool done = false;
  int status = 0;
  {
    py::gil_scoped_release release;
    try {
      // ShHealOper_FillHoles writes the new face's pcurves into the boundary edges it is
      // given (ShHealOper_FillHoles::addFace, BRep_Builder::UpdateEdge), and a retained
      // state shares those edges. It works on a copy with its own topology instead; the
      // copy's history carries every id onto the copy first.
      BRepBuilderAPI_Copy copier(input, /*copyGeom=*/false, /*copyMesh=*/false);
      const TopoDS_Shape copy = copier.Shape();
      ShHealOper_FillHoles filler(copy);
      // GEOM's FillHoles: the default GeomPlate parameters (InitParameters()).
      if (edge_ids.has_value()) {
        TopTools_SequenceOfShape seq;
        for (const TopoDS_Shape& e : bounds) {
          seq.Append(copier.ModifiedShape(e));
        }
        done = filler.Fill(seq);
      } else {
        done = filler.Fill();
      }
      status = filler.GetErrorStatus();
      if (done) {
        result = filler.GetResultShape();
        NCollection_List<TopoDS_Shape> arguments;
        arguments.Append(input);
        hist = new BRepTools_History(arguments, copier);
        hist->Merge(kind_preserving(filler.Context()->History(), copy));
        // Each new face is generated from the edges of the hole it fills.
        ShapeKeyed<TopoDS_Shape> input_of;
        ShapeSet in_edges;
        TopExp::MapShapes(input, TopAbs_EDGE, in_edges);
        for (int i = 1; i <= in_edges.Extent(); ++i) {
          const TopoDS_Shape& e = in_edges.FindKey(i);
          const NCollection_List<TopoDS_Shape>& images = hist->Modified(e);
          if (images.IsEmpty()) {
            input_of.emplace(e, e);
          }
          for (const TopoDS_Shape& m : images) {
            input_of.emplace(m, e);
          }
        }
        ShapeSet old_faces;
        TopExp::MapShapes(input, TopAbs_FACE, old_faces);
        ShapeSet images_of_faces;
        for (int i = 1; i <= old_faces.Extent(); ++i) {
          for (const TopoDS_Shape& m : hist->Modified(old_faces.FindKey(i))) {
            images_of_faces.Add(m);
          }
        }
        for (TopExp_Explorer f(result, TopAbs_FACE); f.More(); f.Next()) {
          if (images_of_faces.Contains(f.Current()) || old_faces.Contains(f.Current())) {
            continue;
          }
          for (TopExp_Explorer e(f.Current(), TopAbs_EDGE); e.More(); e.Next()) {
            const auto it = input_of.find(e.Current());
            if (it != input_of.end()) {
              hist->AddGenerated(it->second, f.Current());
            }
          }
        }
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.fill_holes: OCCT threw: ") + e.what());
    }
  }
  if (!done || result.IsNull()) {
    throw PysmeshError(
        std::string("Session.fill_holes: ShHealOper_FillHoles filled nothing in the bodies of "
                    "the entities ") +
        id_list(entity_ids) +
        (edge_ids.has_value() ? " for the edges " + id_list(*edge_ids) : std::string()) +
        // Fill returns false with no error when it finds no loop at all, and with
        // InvalidParameters when the loops it finds are not holes (prepareWires).
        (status == ShHealOper_ErrorExecution
             ? ": GeomPlate built no surface for a hole (" + shheal_error_text(status) + ")."
             : ": they have no hole bounded by free edges."));
  }
  return commit(concat(untouched, result), hist, op, result);
}

// ---- suppress_faces ----------------------------------------------------------------- //

py::dict Session::suppress_faces(const std::vector<EntityId>& face_ids) {
  const char* op = "suppress_faces";
  OpGuard guard(in_op_);
  const std::vector<TopoDS_Shape> faces = faces_of(op, face_ids);
  const std::vector<TopoDS_Shape> scope = scoped_bodies(face_ids, op);
  const std::vector<TopoDS_Shape> untouched = bodies_excluding(scope);
  const TopoDS_Shape input = make_root(scope);

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  bool done = false;
  int status = 0;
  {
    py::gil_scoped_release release;
    try {
      // GEOM's SuppressFaces: ShHealOper_RemoveFace::Perform(faces). The session's bodies
      // go in as one compound, which ShHealOper treats part by part; GEOM glues the parts
      // of a compound afterwards, which here would join bodies the session keeps apart.
      ShHealOper_RemoveFace remover(input);
      TopTools_SequenceOfShape seq;
      for (const TopoDS_Shape& f : faces) {
        seq.Append(f);
      }
      done = remover.Perform(seq);
      status = remover.GetErrorStatus();
      if (done) {
        result = remover.GetResultShape();
        hist = kind_preserving(remover.Context()->History(), input);
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.suppress_faces: OCCT threw: ") + e.what());
    }
  }
  if (!done) {
    throw PysmeshError("Session.suppress_faces: ShHealOper_RemoveFace removed none of the "
                       "faces " + id_list(face_ids) + " (" + shheal_error_text(status) + ").");
  }
  // Every named face must be gone: a face ShHealOper did not reach would leave the call a
  // partial success.
  ShapeSet left;
  if (!result.IsNull()) {
    TopExp::MapShapes(result, TopAbs_FACE, left);
  }
  std::vector<EntityId> stayed;
  for (std::size_t i = 0; i < faces.size(); ++i) {
    if (left.Contains(faces[i])) {
      stayed.push_back(face_ids[i]);
    }
  }
  if (!stayed.empty()) {
    throw PysmeshError("Session.suppress_faces: the faces " + id_list(stayed) +
                       " are still in the result; nothing is committed.",
                       "", ids_as_int(stayed));
  }
  return commit(concat(untouched, result), hist, op, result, Validation::Report);
}

// ---- shape_process ------------------------------------------------------------------ //

namespace {

// The type of one ShapeProcess parameter, as the operator reads it: ShapeProcess_Context's
// GetReal, GetInteger and GetBoolean, and ShapeProcess_ShapeContext's GetContinuity.
enum class ParamKind { Real, Integer, Boolean, Continuity };

struct ParamSpec {
  const char* name;
  ParamKind kind;
};

struct OperatorSpec {
  const char* name;
  std::vector<ParamSpec> params;
};

// Every operator that ShapeProcess_OperLibrary::Init registers in OCCT 8.0.1, with every
// parameter its function reads, read at the source (ShapeProcess_OperLibrary.cxx). A
// parameter that the function reads only under a condition (SetTolerance's Value and Ratio
// when Mode > 0) or only to set a tool's value when it is present (ToBezier's MaxTolerance)
// is listed too: every one has a default inside OCCT, and shape_process takes none.
const std::vector<OperatorSpec>& operator_specs() {
  using K = ParamKind;
  static const std::vector<OperatorSpec> specs = {
      {"DirectFaces", {}},
      {"SameParameter", {{"Force", K::Boolean}, {"Tolerance3d", K::Real}}},
      {"SetTolerance",
       {{"Mode", K::Integer}, {"Value", K::Real}, {"Ratio", K::Real}, {"Regularity", K::Real}}},
      {"SplitAngle", {{"Angle", K::Real}, {"MaxTolerance", K::Real}}},
      {"BSplineRestriction",
       {{"SurfaceMode", K::Boolean},         {"Curve3dMode", K::Boolean},
        {"Curve2dMode", K::Boolean},         {"Tolerance3d", K::Real},
        {"Tolerance2d", K::Real},            {"Continuity3d", K::Continuity},
        {"Continuity2d", K::Continuity},     {"RequiredDegree", K::Integer},
        {"RequiredNbSegments", K::Integer},  {"PreferDegree", K::Boolean},
        {"RationalToPolynomial", K::Boolean}, {"MaxDegree", K::Integer},
        {"MaxNbSegments", K::Integer},       {"OffsetSurfaceMode", K::Boolean},
        {"OffsetCurve3dMode", K::Boolean},   {"OffsetCurve2dMode", K::Boolean},
        {"LinearExtrusionMode", K::Boolean}, {"RevolutionMode", K::Boolean},
        {"SegmentSurfaceMode", K::Boolean},  {"ConvCurve3dMode", K::Boolean},
        {"ConvCurve2dMode", K::Boolean},     {"BezierMode", K::Boolean},
        {"PlaneMode", K::Boolean},           {"ConicalSurfMode", K::Boolean},
        {"CylindricalSurfMode", K::Boolean}, {"ToroidalSurfMode", K::Boolean},
        {"SphericalSurfMode", K::Boolean}}},
      {"ElementaryToRevolution", {}},
      {"SweptToElementary", {}},
      {"SurfaceToBSpline",
       {{"LinearExtrusionMode", K::Boolean}, {"RevolutionMode", K::Boolean},
        {"OffsetMode", K::Boolean}}},
      {"ToBezier",
       {{"Curve3dMode", K::Boolean},    {"Curve2dMode", K::Boolean},
        {"SurfaceMode", K::Boolean},    {"Line3dMode", K::Boolean},
        {"Circle3dMode", K::Boolean},   {"Conic3dMode", K::Boolean},
        {"SegmentSurfaceMode", K::Boolean}, {"PlaneMode", K::Boolean},
        {"RevolutionMode", K::Boolean}, {"ExtrusionMode", K::Boolean},
        {"BSplineMode", K::Boolean},    {"MaxTolerance", K::Real},
        {"MinCurveLength", K::Real},    {"EdgeMode", K::Boolean}}},
      {"SplitContinuity",
       {{"Tolerance3d", K::Real},
        {"Tolerance2d", K::Real},
        {"CurveContinuity", K::Continuity},
        {"SurfaceContinuity", K::Continuity},
        {"Curve2dContinuity", K::Continuity},
        {"MaxTolerance", K::Real}}},
      {"SplitClosedFaces",
       {{"CloseTolerance", K::Real},
        {"MaxTolerance", K::Real},
        {"NbSplitPoints", K::Integer},
        {"SegmentSurfaceMode", K::Boolean}}},
      {"FixWireGaps", {{"Tolerance3d", K::Real}}},
      {"FixFaceSize", {{"Tolerance", K::Real}}},
      {"DropSmallSolids",
       {{"FixMode", K::Integer},
        {"VolumeThreshold", K::Real},
        {"WidthFactorThreshold", K::Real},
        {"MergeSolids", K::Boolean}}},
      {"DropSmallEdges", {{"Tolerance3d", K::Real}}},
      {"FixShape",
       {{"Tolerance3d", K::Real},
        {"MinTolerance3d", K::Real},
        {"MaxTolerance3d", K::Real},
        {"FixFreeShellMode", K::Integer},
        {"FixFreeFaceMode", K::Integer},
        {"FixFreeWireMode", K::Integer},
        {"FixSameParameterMode", K::Integer},
        {"FixSolidMode", K::Integer},
        {"FixVertexPositionMode", K::Integer},
        {"FixVertexToleranceMode", K::Integer},
        {"FixShellMode", K::Integer},
        {"FixShellOrientationMode", K::Integer},
        {"CreateOpenSolidMode", K::Boolean},
        {"FixFaceMode", K::Integer},
        {"FixFaceOrientationMode", K::Integer},
        {"FixWireMode", K::Integer},
        {"FixOrientationMode", K::Integer},
        {"FixAddNaturalBoundMode", K::Integer},
        {"FixMissingSeamMode", K::Integer},
        {"FixSmallAreaWireMode", K::Integer},
        {"RemoveSmallAreaFaceMode", K::Integer},
        {"FixIntersectingWiresMode", K::Integer},
        {"FixLoopWiresMode", K::Integer},
        {"FixSplitFaceMode", K::Integer},
        {"ModifyTopologyMode", K::Boolean},
        {"ModifyGeometryMode", K::Boolean},
        {"ClosedWireMode", K::Boolean},
        {"PreferencePCurveMode", K::Boolean},
        {"FixReorderMode", K::Integer},
        {"FixSmallMode", K::Integer},
        {"FixConnectedMode", K::Integer},
        {"FixEdgeCurvesMode", K::Integer},
        {"FixDegeneratedMode", K::Integer},
        {"FixLackingMode", K::Integer},
        {"FixSelfIntersectionMode", K::Integer},
        {"RemoveLoopMode", K::Integer},
        {"FixReversed2dMode", K::Integer},
        {"FixRemovePCurveMode", K::Integer},
        {"FixRemoveCurve3dMode", K::Integer},
        {"FixAddPCurveMode", K::Integer},
        {"FixAddCurve3dMode", K::Integer},
        {"FixShiftedMode", K::Integer},
        {"FixSeamMode", K::Integer},
        {"FixEdgeSameParameterMode", K::Integer},
        {"FixNotchedEdgesMode", K::Integer},
        {"FixTailMode", K::Integer},
        {"MaxTailAngle", K::Real},
        {"MaxTailWidth", K::Real},
        {"FixSelfIntersectingEdgeMode", K::Integer},
        {"FixIntersectingEdgesMode", K::Integer},
        {"FixNonAdjacentIntersectingEdgesMode", K::Integer}}},
      {"SplitClosedEdges", {{"NbSplitPoints", K::Integer}}},
      {"SplitCommonVertex", {}},
  };
  return specs;
}

const char* kind_text(ParamKind kind) {
  switch (kind) {
    case ParamKind::Real: return "a real number";
    case ParamKind::Integer: return "an integer";
    case ParamKind::Boolean: return "a bool";
    case ParamKind::Continuity: return "a continuity, one of C0, G1, C1, G2, C2, C3, CN";
  }
  return "";
}

std::string name_list(const std::vector<std::string>& names) {
  std::string out;
  for (const std::string& n : names) {
    out += (out.empty() ? "" : ", ") + n;
  }
  return out;
}

// The resource text of one parameter value, or a refusal that names the parameter. A real
// is written with 17 significant digits, which strtod reads back as the same double;
// Resource_Manager::SetResource(double), which GEOM's SetParameter calls, keeps 6 ("%g").
std::string resource_value(const std::string& full, ParamKind kind, const py::handle& value) {
  const std::string got = py::str(py::repr(value)).cast<std::string>() + " (" +
                          py::str(py::type::of(value).attr("__name__")).cast<std::string>() +
                          ")";
  const auto wrong = [&]() {
    return PysmeshError("Session.shape_process: " + full + " takes " + kind_text(kind) +
                        "; got " + got + ".");
  };
  const bool is_bool = py::isinstance<py::bool_>(value);
  switch (kind) {
    case ParamKind::Real: {
      if (is_bool || !(py::isinstance<py::int_>(value) || py::isinstance<py::float_>(value))) {
        throw wrong();
      }
      const double v = value.cast<double>();
      finite_arg("shape_process", full.c_str(), v);
      char text[32];
      std::snprintf(text, sizeof text, "%.17g", v);
      return text;
    }
    case ParamKind::Integer: {
      if (is_bool || !py::isinstance<py::int_>(value)) {
        throw wrong();
      }
      const py::int_ as_int = py::reinterpret_borrow<py::int_>(value);
      if (py::int_(std::numeric_limits<int>::min()) > as_int ||
          as_int > py::int_(std::numeric_limits<int>::max())) {
        throw PysmeshError("Session.shape_process: " + full + " is " + got +
                           ", outside the range of a 32-bit integer.");
      }
      return std::to_string(value.cast<int>());
    }
    case ParamKind::Boolean: {
      if (!is_bool) {
        throw wrong();
      }
      return value.cast<bool>() ? "1" : "0";
    }
    case ParamKind::Continuity: {
      if (!py::isinstance<py::str>(value)) {
        throw wrong();
      }
      const std::string v = value.cast<std::string>();
      for (const char* c : {"C0", "G1", "C1", "G2", "C2", "C3", "CN"}) {
        if (v == c) {
          return v;
        }
      }
      throw wrong();
    }
  }
  throw wrong();
}

// The resources of one shape_process call, "Operator.Parameter" -> text: every parameter
// of every named operator, each once. Refused, naming the names, for an unknown operator,
// a missing parameter, a parameter no named operator reads, or a value of the wrong type.
std::vector<std::pair<std::string, std::string>> shape_process_resources(
    const std::vector<std::string>& operators, const py::dict& parameters) {
  std::vector<const OperatorSpec*> named;
  std::vector<std::string> unknown;
  for (const std::string& name : operators) {
    const OperatorSpec* found = nullptr;
    for (const OperatorSpec& spec : operator_specs()) {
      if (name == spec.name) {
        found = &spec;
      }
    }
    if (found == nullptr) {
      unknown.push_back("'" + name + "'");
    } else if (std::find(named.begin(), named.end(), found) == named.end()) {
      named.push_back(found);
    }
  }
  if (!unknown.empty()) {
    std::vector<std::string> all;
    for (const OperatorSpec& spec : operator_specs()) {
      all.emplace_back(spec.name);
    }
    throw PysmeshError("Session.shape_process: " + name_list(unknown) +
                       (unknown.size() == 1 ? " is not" : " are not") +
                       " a ShapeProcess operator. The operators are: " + name_list(all) + ".");
  }

  std::map<std::string, py::handle> given;
  for (const auto& item : parameters) {
    if (!py::isinstance<py::str>(item.first)) {
      throw PysmeshError("Session.shape_process: a parameter name is " +
                         py::str(py::repr(item.first)).cast<std::string>() +
                         ", not a str such as 'FixShape.Tolerance3d'.");
    }
    given.emplace(item.first.cast<std::string>(), item.second);
  }
  std::vector<std::pair<std::string, std::string>> out;
  std::vector<std::string> missing;
  std::set<std::string> read;
  for (const OperatorSpec* spec : named) {
    for (const ParamSpec& p : spec->params) {
      const std::string full = std::string(spec->name) + "." + p.name;
      read.insert(full);
      const auto it = given.find(full);
      if (it == given.end()) {
        missing.push_back(full);
        continue;
      }
      out.emplace_back(full, resource_value(full, p.kind, it->second));
    }
  }
  if (!missing.empty()) {
    throw PysmeshError("Session.shape_process: the named operators read " + name_list(missing) +
                       ", which " + (missing.size() == 1 ? "is" : "are") +
                       " not given. Every parameter of a named operator must be given; there "
                       "are no defaults.");
  }
  std::vector<std::string> extra;
  for (const auto& [name, value] : given) {
    if (read.count(name) == 0) {
      extra.push_back(name);
    }
  }
  if (!extra.empty()) {
    throw PysmeshError("Session.shape_process: no named operator reads " + name_list(extra) +
                       "; a parameter is 'Operator.Name' for an operator of the sequence.");
  }
  return out;
}

// A printer that keeps the alarms and failures a ShapeProcess context sends.
// ShapeProcess::Perform catches an operator's exception and only sends
// "SP.Sequence.Error.Except" at Message_Alarm; the call refuses on it instead.
class AlarmLog : public Message_Printer {
 public:
  const std::vector<std::string>& alarms() const { return alarms_; }

 protected:
  void send(const TCollection_AsciiString& text, const Message_Gravity gravity) const override {
    if (gravity >= Message_Alarm) {
      alarms_.emplace_back(text.ToCString());
    }
  }

 private:
  mutable std::vector<std::string> alarms_;
};

// The sub-shapes of kind `k` of `shape`, by TShape. Each TShape has one placement in a
// result here; a TShape placed twice maps to a null shape, which matches nothing.
std::map<const Standard_Transient*, TopoDS_Shape> by_tshape(const TopoDS_Shape& shape,
                                                            TopAbs_ShapeEnum k) {
  std::map<const Standard_Transient*, TopoDS_Shape> out;
  ShapeSet subs;
  TopExp::MapShapes(shape, k, subs);
  for (int i = 1; i <= subs.Extent(); ++i) {
    const TopoDS_Shape& s = subs.FindKey(i);
    const auto [it, fresh] = out.emplace(s.TShape().get(), s);
    if (!fresh && !it->second.IsSame(s)) {
      it->second.Nullify();
    }
  }
  return out;
}

// The history of shape_process: each input sub-shape goes to its TNaming_CopyShape copy,
// and the copy to its image in the ShapeProcess context map (keyed by sub-shapes at the
// null location, ShapeProcess_ShapeContext RecModif), or stays itself where the result
// still holds it. Only images of the kind of the shape count, the rule of kind_preserving;
// a shape left with none is removed.
//
// The copy is explored in the same order as the input (TNaming_CopyShape::Translate adds
// the sub-shapes in the order of TopoDS_Iterator), and each pair is checked against the
// copy map. An image is found in the result by its TShape: ShapeBuild_ReShape records a
// replacement without its placement, and each body was processed in its own frame.
Handle(BRepTools_History) shape_process_history(
    const TopoDS_Shape& input, const TopoDS_Shape& copy,
    const NCollection_IndexedDataMap<Handle(Standard_Transient), Handle(Standard_Transient)>&
        copy_map,
    const NCollection_DataMap<TopoDS_Shape, TopoDS_Shape, TopTools_ShapeMapHasher>& images,
    const TopoDS_Shape& result) {
  Handle(BRepTools_History) hist = new BRepTools_History;
  for (TopAbs_ShapeEnum k : kEntityKinds) {
    ShapeSet before;
    ShapeSet copies;
    TopExp::MapShapes(input, k, before);
    TopExp::MapShapes(copy, k, copies);
    const std::map<const Standard_Transient*, TopoDS_Shape> after = by_tshape(result, k);
    if (before.Extent() != copies.Extent()) {
      throw PysmeshError("Session.shape_process: the copy holds " +
                         std::to_string(copies.Extent()) + " " + kind_name(k) +
                         "(s) for the input's " + std::to_string(before.Extent()) +
                         ". This is a bug, not a caller error.");
    }
    const auto in_result = [&after](const TopoDS_Shape& s) {
      const auto it = after.find(s.TShape().get());
      return it == after.end() ? TopoDS_Shape() : it->second;
    };
    for (int i = 1; i <= before.Extent(); ++i) {
      const TopoDS_Shape& s = before.FindKey(i);
      const TopoDS_Shape& c = copies.FindKey(i);
      const Handle(Standard_Transient)* copied = copy_map.Seek(s.TShape());
      if (copied == nullptr || copied->get() != c.TShape().get()) {
        throw PysmeshError(std::string("Session.shape_process: the copy of a ") +
                           kind_name(k) +
                           " is not where TNaming_CopyShape put it. This is a bug, not a "
                           "caller error.");
      }
      std::vector<TopoDS_Shape> kept;
      if (const TopoDS_Shape* image = images.Seek(c.Located(TopLoc_Location()))) {
        if (!image->IsNull()) {
          ShapeSet pieces;
          TopExp::MapShapes(*image, k, pieces);
          for (int j = 1; j <= pieces.Extent(); ++j) {
            const TopoDS_Shape found = in_result(pieces.FindKey(j));
            if (!found.IsNull()) {
              kept.push_back(found);
            }
          }
        }
      } else {
        const TopoDS_Shape found = in_result(c);
        if (!found.IsNull()) {
          kept.push_back(found);
        }
      }
      if (kept.empty()) {
        hist->Remove(s);
      }
      for (const TopoDS_Shape& m : kept) {
        hist->AddModified(s, m);
      }
    }
  }
  return hist;
}

// The context map is keyed by sub-shapes at the null location, so one TShape placed twice
// in the bodies (an instanced part) would have one entry for two entities. Refused.
void require_single_placements(const TopoDS_Shape& input) {
  for (TopAbs_ShapeEnum k : kEntityKinds) {
    ShapeSet subs;
    TopExp::MapShapes(input, k, subs);
    std::map<const Standard_Transient*, TopLoc_Location> placed;
    for (int i = 1; i <= subs.Extent(); ++i) {
      const TopoDS_Shape& s = subs.FindKey(i);
      const auto [it, fresh] = placed.emplace(s.TShape().get(), s.Location());
      if (!fresh && !it->second.IsEqual(s.Location())) {
        throw PysmeshError(std::string("Session.shape_process: a ") + kind_name(k) +
                           " of the bodies is placed twice (an instanced part). ShapeProcess "
                           "records its history by shape without placement, so the two "
                           "entities cannot be told apart; nothing is committed.");
      }
    }
  }
}

// The bodies of `copy`, each in its own frame, as one compound, and the placement of each.
//
// ShapeProcess_OperLibrary::ApplyModifier modifies each child of a compound without its
// placement and places the result back, but ShapeProcess_ShapeContext records the history
// from the placed root, where the unplaced keys do not match: under a compound of placed
// bodies, a BRepTools_Modification operator records no sub-shape at all. With the bodies
// unplaced the two frames are one. The result is placed back by place_back().
TopoDS_Compound unplaced_bodies(const TopoDS_Shape& copy,
                                std::vector<std::pair<TopoDS_Shape, TopLoc_Location>>& bodies) {
  TopoDS_Compound work;
  BRep_Builder builder;
  builder.MakeCompound(work);
  for (TopoDS_Iterator it(copy); it.More(); it.Next()) {
    const TopoDS_Shape body = it.Value().Located(TopLoc_Location());
    builder.Add(work, body);
    bodies.emplace_back(body, it.Value().Location());
  }
  return work;
}

// The result of the unplaced bodies, each body placed where its input body was: the body's
// image in the context map, or the body itself, carries the placement. A result body that
// is the image of no input body, or of two with different placements, has no placement to
// take; refused when any body was placed.
TopoDS_Shape place_back(
    const TopoDS_Shape& local, const std::vector<std::pair<TopoDS_Shape, TopLoc_Location>>& bodies,
    const NCollection_DataMap<TopoDS_Shape, TopoDS_Shape, TopTools_ShapeMapHasher>& images) {
  bool placed = false;
  std::map<const Standard_Transient*, TopLoc_Location> placement_of;
  bool conflict = false;
  for (const auto& [body, placement] : bodies) {
    placed = placed || !placement.IsIdentity();
    const TopoDS_Shape* image = images.Seek(body);
    const TopoDS_Shape img = image != nullptr ? *image : body;
    if (img.IsNull()) {
      continue;
    }
    std::vector<TopoDS_Shape> tops{img};
    if (img.ShapeType() == TopAbs_COMPOUND && body.ShapeType() != TopAbs_COMPOUND) {
      for (TopoDS_Iterator piece(img); piece.More(); piece.Next()) {
        tops.push_back(piece.Value());
      }
    }
    for (const TopoDS_Shape& t : tops) {
      const auto [it, fresh] = placement_of.emplace(t.TShape().get(), placement);
      conflict = conflict || (!fresh && !it->second.IsEqual(placement));
    }
  }
  if (!placed) {
    return local;
  }
  TopoDS_Compound out;
  BRep_Builder builder;
  builder.MakeCompound(out);
  for (TopoDS_Iterator it(local); it.More(); it.Next()) {
    const auto found = placement_of.find(it.Value().TShape().get());
    if (conflict || found == placement_of.end()) {
      throw PysmeshError("Session.shape_process: a body of the result is the image of no "
                         "placed body of the input, or of two with different placements, so "
                         "its placement is unknown; nothing is committed.");
    }
    builder.Add(out, it.Value().Moved(found->second));
  }
  return out;
}

}  // namespace

py::dict Session::shape_process(const std::vector<EntityId>& entity_ids,
                                const std::vector<std::string>& operators,
                                const py::dict& parameters, const py::object& progress,
                                const py::object& cancel) {
  const char* op = "shape_process";
  OpGuard guard(in_op_);
  if (entity_ids.empty()) {
    throw PysmeshError("Session.shape_process: no entity is named; there is no body to "
                       "process.");
  }
  if (operators.empty()) {
    throw PysmeshError("Session.shape_process: no operator is named; the sequence is empty.");
  }
  const std::vector<std::pair<std::string, std::string>> resources =
      shape_process_resources(operators, parameters);
  std::string sequence_text;
  for (const std::string& name : operators) {
    sequence_text += (sequence_text.empty() ? "" : " ") + name;
  }
  const std::string ids = id_list(entity_ids);
  return rework(
      entity_ids, op, hooks_of(op, progress, cancel),
      [&resources, &sequence_text, &ids](const TopoDS_Shape& input,
                                         const Message_ProgressRange& range, TopoDS_Shape& out,
                                         Handle(BRepTools_History) & hist) {
        require_single_placements(input);
        // The steps of ShHealOper_ShapeProcess::Perform (GEOM V9_16_0,
        // src/ShHealOper/ShHealOper_ShapeProcess.cxx; Copyright (C) 2007-2026 CEA, EDF,
        // OPEN CASCADE; Copyright (C) 2003-2007 OPEN CASCADE, EADS/CCR, LIP6, CEA/DEN,
        // CEDRAT, EDF R&D, LEG, PRINCIPIA R&D, BUREAU VERITAS; LGPL-2.1 or later, used here
        // under LGPL-2.1; recorded in PROVENANCE.md), in its order, with the copy map
        // kept: the tolerance before; the copy that leaves the input untouched
        // (TNaming_CopyShape::CopyTool); the sequence (ShapeProcessAPI_ApplySequence);
        // PrepareShape; GEOMAlgo_AlgoTools::FixCurveOnSurfaceTolerances; the isDone test.
        // Three changes, none of which changes the shape OCCT builds:
        //  * the sequence's resource manager is the empty one that an empty resource name
        //    gives (ShapeProcess_Context::Init), not one read from the CSF_ShHealingDefaults
        //    and CSF_ShHealingUserDefaults files, and every resource is set here;
        //  * the history goes down to the vertices (TopAbs_VERTEX, where ShHealOper's
        //    constructor sets TopAbs_EDGE): the detalisation is read only where
        //    ShapeProcess_ShapeContext records the history;
        //  * the bodies go in without their placement and are placed back (unplaced_bodies).
        ShapeAnalysis_ShapeTolerance tolerance;
        const double initial_tol = tolerance.Tolerance(input, 0);
        NCollection_IndexedDataMap<Handle(Standard_Transient), Handle(Standard_Transient)>
            copy_map;
        TopoDS_Shape copy;
        TNaming_CopyShape::CopyTool(input, copy_map, copy);
        std::vector<std::pair<TopoDS_Shape, TopLoc_Location>> bodies;
        const TopoDS_Compound work = unplaced_bodies(copy, bodies);
        ShapeProcessAPI_ApplySequence sequence("", "ShapeProcess");
        sequence.Context()->SetDetalisation(TopAbs_VERTEX);
        Handle(AlarmLog) log = new AlarmLog;
        sequence.Context()->SetMessenger(new Message_Messenger(log));
        const Handle(Resource_Manager)& rm = sequence.Context()->ResourceManager();
        rm->SetResource("ShapeProcess.exec.op", sequence_text.c_str());
        for (const auto& [name, value] : resources) {
          rm->SetResource(("ShapeProcess." + name).c_str(), value.c_str());
        }
        const TopoDS_Shape local =
            sequence.PrepareShape(work, /*fillmap=*/false, TopAbs_EDGE, range);
        if (range.UserBreak()) {
          return;  // rework raises the cancel and commits nothing
        }
        if (!log->alarms().empty()) {
          std::string text;
          for (const std::string& a : log->alarms()) {
            text += (text.empty() ? "" : "; ") + a;
          }
          throw PysmeshError("Session.shape_process: an operator of the sequence '" +
                             sequence_text + "' failed in OCCT on the bodies of the entities " +
                             ids + " (ShapeProcess reported: " + text +
                             "); nothing is committed.");
        }
        out = place_back(local, bodies, sequence.Context()->Map());
        const bool tol_changed = GEOMAlgo_AlgoTools::FixCurveOnSurfaceTolerances(out);
        // ShHealOper_ShapeProcess's isDone: the result is another shape, or a tolerance
        // changed by more than Precision::Confusion().
        bool done = !work.IsSame(local);
        if (!done) {
          done = std::abs(initial_tol - tolerance.Tolerance(out, 0)) > Precision::Confusion() ||
                 tol_changed;
        }
        if (!done) {
          throw PysmeshError("Session.shape_process: the sequence '" + sequence_text +
                             "' changed nothing in the bodies of the entities " + ids +
                             "; nothing is committed.");
        }
        hist = shape_process_history(input, copy, copy_map, sequence.Context()->Map(), out);
      });
}

}  // namespace session
}  // namespace pysmesh
