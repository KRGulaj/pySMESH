// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-07-12

// pySMESH binding — B-rep offset operations: make_thick_solid and offset_shape.
//
// make_thick_solid wraps BRepOffsetAPI_MakeThickSolid (TKOffset): hollows a solid by
// removing selected faces and building offset inner walls. Use case: CHT wall solid
// creation around an extracted fluid volume, or shell thickening for structural FEA.
//
// offset_shape wraps BRepOffsetAPI_MakeOffsetShape (TKOffset): uniformly offsets all
// faces of a shell or solid by a signed distance (positive = outward, negative = inward).
//
// Both return raw BREP bytes plus a face_map (old 1-based face id -> new 1-based face id,
// -1 if removed), following the same convention as unify_same_domain. make_thick_solid
// reads history from Modified()/IsDeleted() (and pre-marks remove_face_ids as -1 so the map
// is independent of history completeness for the removed set); offset_shape reads history
// from Generated()/IsDeleted() (PerformByJoin records the offset relationship through
// Generated(), leaving Modified() empty).
//
// BRepCheck_Analyzer validates the result before returning; invalid new-face ids are
// surfaced in PysmeshError.face_ids.
//
// GIL released for the main OCCT compute call (MakeThickSolidByJoin / PerformByJoin).
// Toolkits: TKOffset (BRepOffsetAPI_*), TKBRep (BRepCheck_Analyzer, BRepTools, BRep_Builder).

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <BRepBuilderAPI_MakeShape.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepOffset_Mode.hxx>
#include <BRepOffsetAPI_MakeOffsetShape.hxx>
#include <BRepOffsetAPI_MakeThickSolid.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <GeomAbs_JoinType.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS_Shape.hxx>

#include "common.hpp"
#include "offset_guard.hpp"

namespace pysmesh {
namespace {

// Deserialize BREP bytes. Mirrors unify.cpp / tessellate.cpp.
TopoDS_Shape read_brep(const py::bytes& data) {
  const std::string buffer = data;
  std::istringstream stream(buffer);
  TopoDS_Shape shape;
  BRep_Builder builder;
  try {
    BRepTools::Read(shape, stream, builder);
  } catch (const std::exception& e) {
    throw PysmeshError(std::string("BREP read failed: ") + e.what());
  }
  if (shape.IsNull()) {
    throw PysmeshError("BREP read produced a null shape (empty or malformed data)");
  }
  return shape;
}

// Serialize a shape to BREP bytes.
py::bytes write_brep(const TopoDS_Shape& shape) {
  std::ostringstream stream;
  try {
    BRepTools::Write(shape, stream);
  } catch (const std::exception& e) {
    throw PysmeshError(std::string("BREP write failed: ") + e.what());
  }
  return py::bytes(stream.str());
}

// Build face_map for offset_shape from BRepOffsetAPI_MakeOffsetShape history.
//
// For the PerformByJoin path, each original face is *generated* into exactly one offset
// face (verified empirically on box/cylinder/sphere: Generated(s) yields one face and the
// mapping is bijective). Note that Modified(s) returns an empty list here — PerformByJoin
// records the offset relationship through Generated(), not Modified() — so the map is built
// from Generated() with IsDeleted() honoured for suppressed faces.
static py::array_t<std::int32_t> build_offset_face_map(
    BRepOffsetAPI_MakeOffsetShape& mk,
    const TopTools_IndexedMapOfShape& old_faces,
    const TopTools_IndexedMapOfShape& new_faces) {
  const int n_old = old_faces.Extent();
  py::array_t<std::int32_t> out(static_cast<py::ssize_t>(n_old));
  std::int32_t* d = out.mutable_data();

  for (int i = 1; i <= n_old; ++i) {
    const TopoDS_Shape& s = old_faces.FindKey(i);
    if (mk.IsDeleted(s)) {
      d[i - 1] = -1;
      continue;
    }
    // PerformByJoin exposes history via Generated(); Modified() is empty for this path.
    // Modified() must be queried first: it lazily primes the shared history map that
    // Generated() reads from (Generated() returns empty if called before Modified()).
    const TopTools_ListOfShape& mod = mk.Modified(s);
    const TopTools_ListOfShape& gen = mk.Generated(s);
    int idx = 0;
    if (!gen.IsEmpty()) {
      idx = new_faces.FindIndex(gen.First());
    }
    if (idx == 0 && !mod.IsEmpty()) {
      idx = new_faces.FindIndex(mod.First());
    }
    if (idx == 0) {
      idx = new_faces.FindIndex(s);  // fallback: unchanged face reused verbatim
    }
    if (idx == 0) {
      throw PysmeshError("offset_shape: original face id " + std::to_string(i) +
                          " could not be mapped to the result shape "
                          "(no Generated/Modified history and face not preserved).");
    }
    d[i - 1] = static_cast<std::int32_t>(idx);
  }
  return out;
}

// Collect 1-based new-face ids whose individual BRepCheck fails.
// Returns empty if the whole-shape check passes.
static std::vector<int> invalid_new_face_ids(const TopoDS_Shape& shape,
                                              const TopTools_IndexedMapOfShape& new_faces) {
  BRepCheck_Analyzer ana(shape, Standard_False);
  if (ana.IsValid()) return {};
  std::vector<int> bad;
  for (int i = 1; i <= new_faces.Extent(); ++i) {
    if (!BRepCheck_Analyzer(new_faces.FindKey(i), Standard_False).IsValid()) {
      bad.push_back(i);
    }
  }
  return bad;
}

// The radius pre-condition, in this module's own vocabulary.
//
// Identical to Session's, and deliberately so: it is the same arithmetic reached from a
// second entry point, and offset_guard holds the arithmetic once. What differs is the name
// a failing face is reported by. The session answers with an EntityId, which denotes the
// face across every later operation; here there are no ids, so a face is named by its
// 1-based ordinal in the input shape's face map — the same ordinal `remove_face_ids` is
// given in and `face_map` is indexed by.
static void require_surviving_radii(const char* op, const TopoDS_Shape& shape,
                                    const TopTools_IndexedMapOfShape& faces, double distance,
                                    double tol) {
  offset_guard::EdgeOwners edge_owners;
  TopExp::MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_owners);

  offset_guard::ShapeSet moving;
  std::vector<TopoDS_Shape> list;
  for (int i = 1; i <= faces.Extent(); ++i) {
    moving.Add(faces.FindKey(i));
    list.push_back(faces.FindKey(i));
  }

  const std::vector<std::pair<TopoDS_Shape, offset_guard::FaceRadius>> failing =
      offset_guard::radii_that_vanish(list, distance, edge_owners, moving, tol);
  if (failing.empty()) {
    return;
  }

  // The ordinal is taken from the shape's own face map, not from `faces`, because a
  // hollowing is handed only the faces it walls and the caller counts every face.
  TopTools_IndexedMapOfShape all_faces;
  TopExp::MapShapes(shape, TopAbs_FACE, all_faces);
  std::vector<int> blamed;
  for (const std::pair<TopoDS_Shape, offset_guard::FaceRadius>& f : failing) {
    const int idx = all_faces.FindIndex(f.first);
    if (idx > 0) {
      blamed.push_back(idx);
    }
  }
  const int worst = all_faces.FindIndex(failing.front().first);
  const std::string named =
      worst > 0 ? " (face " + std::to_string(worst) + ")" : std::string();

  std::sort(blamed.begin(), blamed.end());
  blamed.erase(std::unique(blamed.begin(), blamed.end()), blamed.end());
  throw PysmeshError(
      std::string(op) + ": " + std::to_string(blamed.size()) +
          " of the faces being offset by " + std::to_string(distance) +
          " do not survive it. The " +
          offset_guard::radius_phrase(failing.front().second, named) + ".",
      offset_guard::radius_detail(), blamed);
}

// The faces the algorithm left in place of the openings.
//
// Modified() is queried first and Generated() second: BRepOffset_MakeOffset primes its
// shared history map inside Modified(), and Generated() answers with an empty list when it
// is called first. Both lists are read, because the post-condition's question is only "did
// this face come from an opening".
static offset_guard::ShapeSet rims_of(BRepOffsetAPI_MakeThickSolid& mk,
                                      const TopTools_ListOfShape& opened) {
  offset_guard::ShapeSet rims;
  for (const TopoDS_Shape& f : opened) {
    for (const TopoDS_Shape& t : mk.Modified(f)) {
      rims.Add(t);
    }
    for (const TopoDS_Shape& t : mk.Generated(f)) {
      rims.Add(t);
    }
  }
  return rims;
}

// ---- make_thick_solid ----------------------------------------------------------------

py::dict make_thick_solid(const py::bytes& brep, const std::vector<int>& remove_face_ids,
                           double thickness, double tol) {
  if (remove_face_ids.empty()) {
    throw PysmeshError("make_thick_solid: remove_face_ids must not be empty");
  }
  if (thickness == 0.0) {
    throw PysmeshError(
        "make_thick_solid: thickness must be non-zero "
        "(positive = outward thickening, negative = inward hollowing)");
  }
  if (!(tol > 0.0)) {
    throw PysmeshError("make_thick_solid: tol must be > 0 (got " + std::to_string(tol) + ")");
  }

  const TopoDS_Shape shape = read_brep(brep);
  if (shape.ShapeType() != TopAbs_SOLID) {
    throw PysmeshError(
        "make_thick_solid: input shape must be a SOLID "
        "(got ShapeType " + std::to_string(static_cast<int>(shape.ShapeType())) + ")");
  }

  // Build old-face map and validate / resolve the requested removal ids.
  TopTools_IndexedMapOfShape old_faces;
  TopExp::MapShapes(shape, TopAbs_FACE, old_faces);
  const int nf = old_faces.Extent();

  TopTools_ListOfShape faces_to_remove;
  for (int fid : remove_face_ids) {
    if (fid < 1 || fid > nf) {
      throw PysmeshError(
          "make_thick_solid: invalid face_id " + std::to_string(fid) +
          " (shape has " + std::to_string(nf) + " faces)",
          "", {fid});
    }
    faces_to_remove.Append(old_faces.FindKey(fid));
  }

  // Every face the caller did not open is the one that gets an inner wall, so those are the
  // faces whose radii have to survive. The openings are removed rather than offset, and a
  // cap that is removed does not slide, which is what decided the outcome before 4.2.0: a
  // cylinder of radius 2 opened at its curved wall was declined by OCCT, and the same
  // cylinder opened at a planar cap committed a wall thicker than the body — volume 87.81
  // against the solid's own 87.96 at a thickness of -2.10.
  TopTools_IndexedMapOfShape walled;
  for (int i = 1; i <= nf; ++i) {
    bool opened = false;
    for (int fid : remove_face_ids) {
      if (fid == i) {
        opened = true;
        break;
      }
    }
    if (!opened) {
      walled.Add(old_faces.FindKey(i));
    }
  }
  require_surviving_radii("make_thick_solid", shape, walled, thickness, tol);

  // Run OCCT with GIL released — this is the expensive, pure-C++ step.
  BRepOffsetAPI_MakeThickSolid mk;
  {
    py::gil_scoped_release release;
    try {
      mk.MakeThickSolidByJoin(shape, faces_to_remove, thickness, tol, BRepOffset_Skin,
                              Standard_False /*intersection*/, Standard_False /*selfInter*/,
                              GeomAbs_Intersection);
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(
          std::string("BRepOffsetAPI_MakeThickSolid::MakeThickSolidByJoin failed: ") + e.what(),
          "", std::vector<int>(remove_face_ids));
    }
  }

  if (!mk.IsDone()) {
    throw PysmeshError("BRepOffsetAPI_MakeThickSolid::IsDone() is false "
                        "(self-intersecting offset or degenerate geometry)",
                        "", std::vector<int>(remove_face_ids));
  }
  const TopoDS_Shape result = mk.Shape();
  if (result.IsNull()) {
    throw PysmeshError("BRepOffsetAPI_MakeThickSolid produced a null shape");
  }

  TopTools_IndexedMapOfShape new_faces;
  TopExp::MapShapes(result, TopAbs_FACE, new_faces);

  const std::vector<int> bad = invalid_new_face_ids(result, new_faces);
  if (!bad.empty()) {
    throw PysmeshError(
        "make_thick_solid produced an invalid shape (BRepCheck_Analyzer reported errors "
        "— likely self-intersecting offset; reduce |thickness| or simplify removed faces)",
        "", bad);
  }

  // What the result must be, beyond being valid. Past what the body can carry,
  // MakeThickSolidByJoin collapses the inner shell, reports IsDone(), and hands back the
  // input solid with the opened faces re-issued: a 3 x 7 x 11 box opened at one face and
  // walled by -5.0 comes back measuring 231.0, the input exactly. BRepCheck_Analyzer
  // accepts it, so only a statement about what a hollowed solid IS can see it.
  offset_guard::ShapeSet opened_set;
  for (const TopoDS_Shape& f : faces_to_remove) {
    opened_set.Add(f);
  }
  const std::string collapsed = offset_guard::not_a_thick_solid(
      shape, result, thickness, opened_set, rims_of(mk, faces_to_remove));
  if (!collapsed.empty()) {
    throw PysmeshError(
        "make_thick_solid: the result at thickness " + std::to_string(thickness) +
            " is not a hollowed solid. " + collapsed,
        "MakeThickSolidByJoin reported success and BRepCheck_Analyzer accepted the shape, "
        "but what came back is not the body hollowed. Reduce |thickness|, or open a "
        "different face set. Nothing was returned.",
        std::vector<int>(remove_face_ids));
  }

  // Build face_map: explicitly-removed face ids → -1 (deterministic, regardless of history);
  // remaining faces → from Modified() / unchanged fallback.
  const int n = old_faces.Extent();
  py::array_t<std::int32_t> face_map_arr(static_cast<py::ssize_t>(n));
  std::int32_t* fm = face_map_arr.mutable_data();

  std::vector<bool> is_removed(static_cast<std::size_t>(n + 1), false);
  for (int fid : remove_face_ids) {
    is_removed[static_cast<std::size_t>(fid)] = true;
  }

  for (int i = 1; i <= n; ++i) {
    if (is_removed[static_cast<std::size_t>(i)]) {
      fm[i - 1] = -1;
      continue;
    }
    const TopoDS_Shape& s = old_faces.FindKey(i);
    if (mk.IsDeleted(s)) {
      fm[i - 1] = -1;
      continue;
    }
    const TopTools_ListOfShape& mod = mk.Modified(s);
    const TopoDS_Shape& candidate = mod.IsEmpty() ? s : mod.First();
    int idx = new_faces.FindIndex(candidate);
    if (idx == 0 && !mod.IsEmpty()) {
      idx = new_faces.FindIndex(s);
    }
    if (idx == 0) {
      throw PysmeshError("make_thick_solid: original face id " + std::to_string(i) +
                          " could not be mapped to the result shape "
                          "(topology history inconsistency).");
    }
    fm[i - 1] = static_cast<std::int32_t>(idx);
  }

  py::dict out;
  out["brep"] = write_brep(result);
  out["face_map"] = face_map_arr;
  return out;
}

// ---- offset_shape --------------------------------------------------------------------

py::dict offset_shape(const py::bytes& brep, double offset, double tol) {
  if (offset == 0.0) {
    throw PysmeshError(
        "offset_shape: offset must be non-zero "
        "(positive = outward enlargement, negative = inward shrinkage)");
  }
  if (!(tol > 0.0)) {
    throw PysmeshError("offset_shape: tol must be > 0 (got " + std::to_string(tol) + ")");
  }

  const TopoDS_Shape shape = read_brep(brep);

  TopTools_IndexedMapOfShape old_faces;
  TopExp::MapShapes(shape, TopAbs_FACE, old_faces);

  // A uniform offset moves every face of the body, so every one of them has to survive it.
  // Measured on a cylinder of radius 2 and height 7: at -2.01 this module committed a body
  // of volume 0.0009361946107697187, which is exactly the r = 0.01 cylinder — OCCT builds
  // the surface at the absolute value of the negative radius, the cylinder mirrored through
  // its own axis.
  require_surviving_radii("offset_shape", shape, old_faces, offset, tol);

  BRepOffsetAPI_MakeOffsetShape mk;
  {
    py::gil_scoped_release release;
    try {
      mk.PerformByJoin(shape, offset, tol, BRepOffset_Skin,
                       Standard_False /*intersection*/, Standard_False /*selfInter*/,
                       GeomAbs_Intersection);
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(
          std::string("BRepOffsetAPI_MakeOffsetShape::PerformByJoin failed: ") + e.what());
    }
  }

  if (!mk.IsDone()) {
    throw PysmeshError("BRepOffsetAPI_MakeOffsetShape::IsDone() is false "
                        "(self-intersecting offset or degenerate geometry)");
  }
  const TopoDS_Shape result = mk.Shape();
  if (result.IsNull()) {
    throw PysmeshError("BRepOffsetAPI_MakeOffsetShape produced a null shape");
  }

  TopTools_IndexedMapOfShape new_faces;
  TopExp::MapShapes(result, TopAbs_FACE, new_faces);

  const std::vector<int> bad = invalid_new_face_ids(result, new_faces);
  if (!bad.empty()) {
    throw PysmeshError(
        "offset_shape produced an invalid shape (BRepCheck_Analyzer reported errors "
        "— likely self-intersecting offset; reduce |offset| or use a simpler shape)",
        "", bad);
  }

  // A solid offset inward is the set of its own points at least |offset| from its boundary,
  // which is a proper subset, and offset outward it is a proper superset. The cover for
  // every face the radius rule cannot speak for — a plane, a B-spline, a surface of
  // revolution.
  const std::string wrong = offset_guard::not_an_offset_body(shape, result, offset);
  if (!wrong.empty()) {
    throw PysmeshError(
        "offset_shape: the result at distance " + std::to_string(offset) +
            " is not that body offset. " + wrong,
        "PerformByJoin reported success and BRepCheck_Analyzer accepted the shape, but what "
        "came back is not the body offset. Reduce |offset|. Nothing was returned.");
  }

  py::dict out;
  out["brep"] = write_brep(result);
  out["face_map"] = build_offset_face_map(mk, old_faces, new_faces);
  return out;
}

}  // namespace

void bind_offset(py::module_& m) {
  m.def(
      "make_thick_solid", &make_thick_solid, py::arg("brep"), py::arg("remove_face_ids"),
      py::arg("thickness"), py::arg("tol"),
      "Hollow a SOLID BREP by removing selected faces and building offset inner walls "
      "(BRepOffsetAPI_MakeThickSolid::MakeThickSolidByJoin, TKOffset). "
      "thickness > 0: offset outward from face normals (enlarges); "
      "thickness < 0: offset inward (hollows). "
      "remove_face_ids: 1-based face ids matching Shape.faces() to open. "
      "Raises PysmeshError(.face_ids = offending new-face ids) on OCCT failure or "
      "BRepCheck_Analyzer invalidation. "
      "Returns dict: brep (bytes), face_map int32 (n_old_faces,) — new 1-based face id, "
      "-1 for removed/deleted faces. GIL released for MakeThickSolidByJoin(). "
      "Low-level: use pysmesh.make_thick_solid with ThickSolidParams.");
  m.def(
      "offset_shape", &offset_shape, py::arg("brep"), py::arg("offset"), py::arg("tol"),
      "Uniformly offset all faces of a BREP by a signed distance "
      "(BRepOffsetAPI_MakeOffsetShape::PerformByJoin, BRepOffset_Skin, "
      "GeomAbs_Intersection join, TKOffset). "
      "offset > 0 enlarges; offset < 0 shrinks. "
      "Returns dict: brep (bytes), face_map int32 (n_old_faces,) — new 1-based face id "
      "per original face, -1 if deleted. GIL released for PerformByJoin(). "
      "Low-level: use pysmesh.offset_shape with OffsetParams.");
}

}  // namespace pysmesh
