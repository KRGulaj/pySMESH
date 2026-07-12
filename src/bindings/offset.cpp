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
