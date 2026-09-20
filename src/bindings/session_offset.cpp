// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-09-20

// pySMESH binding — Session: the uniform offset of a whole body.
//
// offset drives TKOffset's BRepOffsetAPI_MakeOffsetShape with BRepOffset_Skin and
// GeomAbs_Intersection, the same two modes the stateless offset_shape uses. What the session
// adds is identity: a face the offset rebuilds keeps the id it had, so a boundary condition
// or a mesh group named on the original body still names the same face afterwards.
//
// Two things about this operation are specific to it, and both are handled here rather than
// by the shared plumbing:
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
Handle(BRepTools_History) offset_history(const TopoDS_Shape& argument,
                                         BRepOffsetAPI_MakeOffsetShape& mk,
                                         const ShapeSet& produced) {
  Handle(BRepTools_History) hist = new BRepTools_History;
  const ShapeSet args = shape_set(argument);
  for (int i = 1; i <= args.Extent(); ++i) {
    const TopoDS_Shape& s = args(i);
    if (!BRepTools_History::IsSupportedType(s)) {
      continue;
    }
    bool related = false;
    if (!mk.IsDeleted(s)) {
      for (const TopoDS_Shape& t : related_to(mk, s)) {
        if (t.IsSame(s) || !produced.Contains(t) || !BRepTools_History::IsSupportedType(t)) {
          continue;
        }
        if (t.ShapeType() == s.ShapeType()) {
          hist->AddModified(s, t);
          related = true;
        } else {
          hist->AddGenerated(s, t);
        }
      }
    }
    // An input that the result neither contains nor descends from is gone, and saying so
    // is what keeps its id from being carried onto whatever ends up in the same place.
    if (!related && !produced.Contains(s)) {
      hist->Remove(s);
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
        hist = offset_history(owner, mk, shape_set(result));
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
