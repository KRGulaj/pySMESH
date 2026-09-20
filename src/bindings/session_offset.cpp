// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-09-20

// pySMESH binding — Session: the offset family, hollowing and uniform offsetting.
//
// Both operations drive TKOffset's BRepOffsetAPI with BRepOffset_Skin and
// GeomAbs_Intersection, the same two modes the stateless make_thick_solid and offset_shape
// use. What the session adds is identity: a wall the offset rebuilds keeps the id it had,
// so a boundary condition or a mesh group named on the original body still names the same
// wall afterwards.
//
// Two things about this family are specific to it, and both are handled here rather than by
// the shared plumbing:
//
//   * BRepOffsetAPI_MakeOffsetShape::PerformByJoin reports a face's offset through
//     Generated(), not Modified() — measured on a box, where Modified() is empty for every
//     one of the six faces. The session migrates an id across a Modified relation only, so
//     history_of() would kill every face id the offset touched and issue six new ones. The
//     relation OCCT describes there is its own definition of *modified*: BRepTools_History
//     documents `modified` as a replacement of the same dimension and reserves `generated`
//     for a change of dimension. offset_history() below therefore reads both lists and
//     files each target by the only thing that decides the relation — whether its shape
//     type matches its source's.
//   * A failed offset has to blame something, and the only ids that exist at that moment
//     are the ones the operation was given. The stateless module answers with the 1-based
//     ordinals of the result's invalid faces; an ordinal put on PysmeshError.face_ids here
//     would be read as an EntityId and would denote somebody else's entity, which is the
//     exact confusion the session exists to remove. So a failing new face is traced back
//     through the history to the original face it came from, and that face's EntityId is
//     what the caller is handed.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

namespace pysmesh {
namespace session {

namespace {

// Every sub-shape of a shape, as a set that answers "is this in the result".
ShapeSet shape_set(const TopoDS_Shape& s) {
  ShapeSet out;
  if (!s.IsNull()) {
    TopExp::MapShapes(s, out);
  }
  return out;
}

// Everything the algorithm relates to one input shape, copied out of it.
//
// Copied, not referenced. BRepBuilderAPI_MakeShape's two history accessors each hand back a
// reference to a list the algorithm owns, and nothing in the contract says the two are
// distinct objects or that one survives a call to the other. Holding both references at once
// would be a bet on an implementation detail.
//
// The call order is load-bearing: BRepOffset_MakeOffset primes its shared history map inside
// Modified(), and Generated() answers with an empty list when it is called first.
NCollection_List<TopoDS_Shape> related_to(BRepOffsetAPI_MakeOffsetShape& mk,
                                          const TopoDS_Shape& s) {
  NCollection_List<TopoDS_Shape> out;
  for (const TopoDS_Shape& t : mk.Modified(s)) {
    out.Append(t);
  }
  for (const TopoDS_Shape& t : mk.Generated(s)) {
    out.Append(t);
  }
  return out;
}

// The history of one BRepOffsetAPI run, in the form the registry carries ids across.
//
// Not history_of(): that template asks BRepTools_History's own constructor to copy the
// algorithm's three relations verbatim, and verbatim is wrong here for the reason the file
// header gives. This reads the same two lists and re-files each target by dimension, which
// is the rule BRepTools_History's own documentation states: a target of the same shape type
// is a modification, one of a different type is a generation.
//
// `opened` names the faces a hollowing was asked to turn into openings, and is empty for a
// uniform offset. Those faces are gone by the caller's own request, so none of them is
// allowed to take a Modified relation however OCCT files it: MakeThickSolidByJoin reports
// the rim it leaves behind — the opened face with the cavity cut out of it — as modified
// from that face, and honouring it would slide the opening's id onto a wall of the result.
// The rim is recorded as *generated* from the opened face instead, which keeps the
// provenance without keeping the id.
Handle(BRepTools_History) offset_history(const TopoDS_Shape& argument,
                                         BRepOffsetAPI_MakeOffsetShape& mk,
                                         const ShapeSet& produced, const ShapeSet& opened) {
  Handle(BRepTools_History) hist = new BRepTools_History;
  const ShapeSet args = shape_set(argument);
  bool body_related = false;
  for (int i = 1; i <= args.Extent(); ++i) {
    const TopoDS_Shape& s = args(i);
    if (!BRepTools_History::IsSupportedType(s)) {
      continue;
    }
    const bool is_body = s.IsSame(argument);
    const bool is_opening = opened.Contains(s);
    bool related = false;
    if (!mk.IsDeleted(s)) {
      for (const TopoDS_Shape& t : related_to(mk, s)) {
        if (t.IsSame(s) || !produced.Contains(t) || !BRepTools_History::IsSupportedType(t)) {
          continue;
        }
        if (t.ShapeType() == s.ShapeType() && !is_opening) {
          hist->AddModified(s, t);
          related = true;
        } else {
          hist->AddGenerated(s, t);
        }
      }
    }
    if (is_body) {
      body_related = related;
      continue;
    }
    // An input that the result neither contains nor descends from is gone. Saying so is
    // what puts a hollowed solid's opened faces in `deleted`, instead of leaving their ids
    // to be carried onto whatever ends up in the same place.
    if (!related && !produced.Contains(s)) {
      hist->Remove(s);
    }
  }

  // The body itself, when the algorithm said nothing about it.
  //
  // PerformByJoin does relate it — measured on a box, where the offset solid comes back
  // through the solid's own Generated() — but MakeThickSolidByJoin relates nothing to the
  // solid it was handed, so the body's id would die at every hollow and a caller's handle
  // to "the part" would break. The result's solids are that body rebuilt, which is a
  // modification by BRepTools_History's own definition: the dimension is unchanged. Two or
  // more solids in the result make it a split, and the id survives on each, which is the
  // session's existing answer to that shape of question rather than a new one.
  if (!body_related && !produced.Contains(argument) &&
      BRepTools_History::IsSupportedType(argument)) {
    for (int i = 1; i <= produced.Extent(); ++i) {
      if (produced(i).ShapeType() == argument.ShapeType()) {
        hist->AddModified(argument, produced(i));
      }
    }
  }
  return hist;
}

// Faces of the result that BRepCheck_Analyzer rejects on their own.
//
// Reached only when the whole-shape check has already failed, so the per-face pass costs
// nothing on a healthy result. Geometric controls stay on — the same setting commit() uses —
// so the two verdicts cannot disagree about what "invalid" means.
std::vector<TopoDS_Shape> invalid_faces(const TopoDS_Shape& result) {
  std::vector<TopoDS_Shape> out;
  for (TopExp_Explorer ex(result, TopAbs_FACE); ex.More(); ex.Next()) {
    if (!BRepCheck_Analyzer(ex.Current()).IsValid()) {
      out.push_back(ex.Current());
    }
  }
  return out;
}

}  // namespace

// ---- the shared failure diagnostic ----------------------------------------------------- //

// The ids of the input faces that the named broken result faces came from.
//
// `blamed` holds faces of the result, and those carry no EntityId — the operation has not
// committed and never will. Each is walked back through the history to the input faces it
// was modified or generated from, and those faces' ids are what the caller is handed: ids
// that are alive, that denote entities the caller already holds, and that name the region of
// the input the offset could not handle. When nothing traces back, the operands are blamed.
std::vector<int> Session::offset_blame(const TopoDS_Shape& argument,
                                       const Handle(BRepTools_History) & hist,
                                       const std::vector<TopoDS_Shape>& blamed,
                                       const std::vector<EntityId>& fallback) const {
  ShapeSet culprits;
  for (const TopoDS_Shape& b : blamed) {
    culprits.Add(b);
  }
  std::vector<EntityId> out;
  if (!hist.IsNull() && culprits.Extent() > 0) {
    for (TopExp_Explorer ex(argument, TopAbs_FACE); ex.More(); ex.Next()) {
      const TopoDS_Shape& s = ex.Current();
      bool hit = false;
      for (const TopoDS_Shape& t : hist->Modified(s)) {
        hit = hit || culprits.Contains(t);
      }
      for (const TopoDS_Shape& t : hist->Generated(s)) {
        hit = hit || culprits.Contains(t);
      }
      if (hit) {
        ids_on(s, out);
      }
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out.empty() ? ids_as_int(fallback) : ids_as_int(out);
}

// Every live id on one sub-shape, appended. A merge leaves several, and all of them name
// the shape, so all of them are reported.
void Session::ids_on(const TopoDS_Shape& s, std::vector<EntityId>& out) const {
  const auto it = state_.registry->by_shape.find(s);
  if (it != state_.registry->by_shape.end()) {
    out.insert(out.end(), it->second.begin(), it->second.end());
  }
}

// The live ids of a body's faces, ascending.
//
// What an offset blames when OCCT declines outright: PerformByJoin that never reaches
// IsDone() leaves no history, so there is nothing to trace a failure back through, and the
// operand list a caller passed may name edges or vertices rather than faces. Reporting the
// body's faces keeps PysmeshError.face_ids a face channel and says which body failed, which
// is every diagnostic OCCT left to give.
std::vector<EntityId> Session::face_ids_of(const TopoDS_Shape& body) const {
  std::vector<EntityId> out;
  for (TopExp_Explorer ex(body, TopAbs_FACE); ex.More(); ex.Next()) {
    ids_on(ex.Current(), out);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// Neither an offset distance nor a wall thickness has a meaning at zero: the algorithm would
// build the input again and the operation would report a rebuild of the whole body as its
// answer. Refused here rather than forwarded.
//
// `!std::isfinite(v)` first, and deliberately: NaN compares unequal to zero, so a plain
// `v == 0.0` test lets it through to OCCT as a distance nothing can be measured against.
void Session::require_non_zero(const char* op, const char* name, double v) {
  if (!std::isfinite(v) || v == 0.0) {
    throw PysmeshError(std::string("Session.") + op + ": " + name +
                       " must be a finite non-zero value (got " + std::to_string(v) +
                       "). Positive offsets outward, negative inward.");
  }
}

// ---- make_thick_solid ------------------------------------------------------------------ //

py::dict Session::make_thick_solid(const std::vector<EntityId>& face_ids, double thickness,
                                   double tol, const py::object& progress,
                                   const py::object& cancel) {
  OpGuard guard(in_op_);
  require_non_zero("make_thick_solid", "thickness", thickness);
  require_positive("tol", tol);
  const std::vector<TopoDS_Shape> faces = faces_of("make_thick_solid", face_ids);
  const TopoDS_Shape owner = sole_body("make_thick_solid", face_ids);
  if (owner.ShapeType() != TopAbs_SOLID) {
    throw PysmeshError(
        std::string("Session.make_thick_solid: the named faces belong to a ") +
        std::string(TopAbs::ShapeTypeToString(owner.ShapeType())) +
        " body; hollowing needs a SOLID. MakeThickSolidByJoin builds the wall between a "
        "solid's boundary and its offset, and an open body has no such wall.");
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({owner});

  ShapeSet opened;
  for (const TopoDS_Shape& f : faces) {
    opened.Add(f);
  }

  ProgressDriver driver("make_thick_solid", hooks_of("make_thick_solid", progress, cancel));
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::vector<TopoDS_Shape> bad;
  std::vector<EntityId> unopened;
  {
    py::gil_scoped_release release;
    BRepOffsetAPI_MakeThickSolid mk;
    NCollection_List<TopoDS_Shape> closing;
    for (const TopoDS_Shape& f : faces) {
      closing.Append(f);
    }
    try {
      mk.MakeThickSolidByJoin(owner, closing, thickness, tol, BRepOffset_Skin,
                              /*Intersection=*/Standard_False,
                              /*SelfInter=*/Standard_False, GeomAbs_Intersection,
                              /*RemoveIntEdges=*/Standard_False, driver.range());
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(
          std::string("Session.make_thick_solid: "
                      "BRepOffsetAPI_MakeThickSolid::MakeThickSolidByJoin failed: ") +
              e.what(),
          "", ids_as_int(face_ids));
    }
    if (mk.IsDone()) {
      result = mk.Shape();
      if (!result.IsNull()) {
        const ShapeSet produced = shape_set(result);
        // The post-condition that makes the delta trustworthy: a face the caller named is
        // an opening, so it must not still be a face of the result. If one is, OCCT did
        // not open it, and committing would report an id as dead while the wall it names
        // is still there.
        for (std::size_t i = 0; i < faces.size(); ++i) {
          if (produced.Contains(faces[i])) {
            unopened.push_back(face_ids[i]);
          }
        }
        hist = offset_history(owner, mk, produced, opened);
        // Checked here rather than left to commit(), because only here does the history
        // that traces a broken result face back to the input face it came from still
        // exist. commit() re-checks and is what actually refuses the shape; this pass is
        // what puts ids on the refusal.
        if (validate_ && !BRepCheck_Analyzer(result).IsValid()) {
          bad = invalid_faces(result);
        }
      }
    }
  }
  driver.finish();
  // Before either failure path: a hollowing the caller stopped produced nothing, and
  // blaming the named faces for a self-intersection that was never computed would be a
  // false diagnostic.
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("make_thick_solid");
  }
  if (result.IsNull()) {
    throw PysmeshError(
        "Session.make_thick_solid: OCCT could not hollow the body at thickness " +
            std::to_string(thickness) + ".",
        "BRepOffsetAPI_MakeThickSolid::IsDone() is false. |thickness| is most likely "
        "larger than the body's smallest feature, so the offset walls self-intersect. "
        "No partial result is returned.",
        ids_as_int(face_ids));
  }
  if (!unopened.empty()) {
    throw PysmeshError(
        "Session.make_thick_solid: OCCT left " + std::to_string(unopened.size()) +
            " of the " + std::to_string(faces.size()) +
            " named faces in the result instead of opening them.",
        "MakeThickSolidByJoin reported success, but those faces are still faces of the "
        "hollowed body. No partial result is returned.",
        ids_as_int(unopened));
  }
  if (!bad.empty()) {
    throw PysmeshError(
        "Session.make_thick_solid: the hollowed solid is invalid (BRepCheck_Analyzer "
        "rejected " +
            std::to_string(bad.size()) + " of its faces); the session is unchanged.",
        "The offset walls most likely self-intersect. Reduce |thickness|, or open a face "
        "set whose walls do not fold onto each other. The ids reported are the input "
        "faces the broken result faces came from.",
        offset_blame(owner, hist, bad, face_ids));
  }
  return commit(concat(survivors, result), hist, "make_thick_solid", result);
}

// ---- offset ---------------------------------------------------------------------------- //

py::dict Session::offset(const std::vector<EntityId>& entity_ids, double distance,
                         double tol, const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  require_non_zero("offset", "distance", distance);
  require_positive("tol", tol);
  const TopoDS_Shape owner = sole_body("offset", entity_ids);
  if (owner.ShapeType() != TopAbs_SOLID && owner.ShapeType() != TopAbs_SHELL) {
    throw PysmeshError(std::string("Session.offset: the named entities belong to a ") +
                       std::string(TopAbs::ShapeTypeToString(owner.ShapeType())) +
                       " body; a uniform offset needs a SOLID or a SHELL.");
  }
  // Taken before the run, while the registry still describes the body OCCT is about to
  // replace: it is what a failure blames when OCCT leaves no history to trace one through.
  const std::vector<EntityId> body_faces = face_ids_of(owner);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({owner});

  ProgressDriver driver("offset", hooks_of("offset", progress, cancel));
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::vector<TopoDS_Shape> bad;
  {
    py::gil_scoped_release release;
    BRepOffsetAPI_MakeOffsetShape mk;
    try {
      mk.PerformByJoin(owner, distance, tol, BRepOffset_Skin,
                       /*Intersection=*/Standard_False, /*SelfInter=*/Standard_False,
                       GeomAbs_Intersection, /*RemoveIntEdges=*/Standard_False,
                       driver.range());
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(
          std::string(
              "Session.offset: BRepOffsetAPI_MakeOffsetShape::PerformByJoin failed: ") +
              e.what(),
          "", ids_as_int(body_faces));
    }
    if (mk.IsDone()) {
      result = mk.Shape();
      if (!result.IsNull()) {
        // No opened faces: a uniform offset removes nothing, so every face of the body is
        // free to keep its id.
        hist = offset_history(owner, mk, shape_set(result), ShapeSet());
        if (validate_ && !BRepCheck_Analyzer(result).IsValid()) {
          bad = invalid_faces(result);
        }
      }
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("offset");
  }
  if (result.IsNull()) {
    throw PysmeshError(
        "Session.offset: OCCT could not offset the body by " + std::to_string(distance) +
            ".",
        "BRepOffsetAPI_MakeOffsetShape::IsDone() is false. |distance| is most likely "
        "larger than the body's smallest radius of curvature or feature, so the offset "
        "faces self-intersect. No partial result is returned.",
        ids_as_int(body_faces));
  }
  if (!bad.empty()) {
    throw PysmeshError(
        "Session.offset: the offset body is invalid (BRepCheck_Analyzer rejected " +
            std::to_string(bad.size()) + " of its faces); the session is unchanged.",
        "The offset faces most likely self-intersect. Reduce |distance|. The ids reported "
        "are the input faces the broken result faces came from.",
        offset_blame(owner, hist, bad, body_faces));
  }
  return commit(concat(survivors, result), hist, "offset", result);
}

}  // namespace session
}  // namespace pysmesh
