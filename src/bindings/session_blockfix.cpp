// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-22

// pySMESH binding — Session.remove_extra_edges: GEOM's RemoveExtraEdges, every id carried.
//
// GEOM's RemoveExtraEdges is the BLOCK_REMOVE_EXTRA branch of GEOMImpl_BlockDriver::Execute
// (SALOME GEOM tag V9_16_0, 7d0a18e1554c25bca2c2d9882e0621ea3f7f9c81): a TNaming copy of the
// shape, BlockFix_BlockFixAPI::Perform, and GEOM's default ShHealOper_ShapeProcess on a
// result that BRepCheck rejects. The branch is followed here step by step; Perform and the
// BlockFix steps it calls are the copies of geom_blockfix.cpp, which write each step's
// history. compose_histories follows every input sub-shape through the steps in order.
// pySMESH differs from GEOM in two places:
//  * a face that GEOM's RotateSphereSpace or FixRanges puts on a turned sphere or cylinder
//    must keep its area (require_turned_faces_kept);
//  * a result that BRepCheck rejects is refused, not repaired: GEOM's repair takes the
//    tolerances of its resource file, absolute lengths (FixFaceSize and DropSmallEdges
//    0.05), which would remove faces and edges below that size without a word.
// Copyright (C) 2007-2026 CEA, EDF, OPEN CASCADE, LGPL-2.1 or later, used here under
// LGPL-2.1, and recorded in PROVENANCE.md.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include "geom_blockfix.hpp"
#include "shape_checks.hpp"

#include <BRepAdaptor_Surface.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepGProp.hxx>
#include <GProp_GProps.hxx>
#include <Message_ProgressScope.hxx>
#include <TColStd_IndexedDataMapOfTransientTransient.hxx>
#include <TNaming_CopyShape.hxx>
#include <TopExp.hxx>
#include <TopoDS_Iterator.hxx>

#include <array>
#include <string>
#include <vector>

namespace pysmesh {
namespace session {

namespace {

// Pairs the children of `before` and `after`, place by place, into `hist`.
void pair_children(const TopoDS_Shape& before, const TopoDS_Shape& after,
                   const Handle(BRepTools_History)& hist, ShapeSet& seen) {
  TopoDS_Iterator a(before);
  TopoDS_Iterator b(after);
  for (; a.More() && b.More(); a.Next(), b.Next()) {
    const TopoDS_Shape& x = a.Value();
    const TopoDS_Shape& y = b.Value();
    if (x.ShapeType() != y.ShapeType()) {
      throw PysmeshError(std::string("Session.remove_extra_edges: a ") +
                         TopAbs::ShapeTypeToString(x.ShapeType()) + " was rebuilt as a " +
                         TopAbs::ShapeTypeToString(y.ShapeType()) +
                         "; the rebuild did not keep the structure. This is a bug, not a "
                         "caller error.");
    }
    // A shape reached twice (a shared face, a seam edge) was paired, with its children, at
    // its first place: the rebuild gives one shape one image.
    if (seen.Contains(x)) {
      continue;
    }
    seen.Add(x);
    if (BRepTools_History::IsSupportedType(x) && !x.IsSame(y)) {
      hist->AddModified(x, y);
    }
    pair_children(x, y, hist, seen);
  }
  if (a.More() || b.More()) {
    throw PysmeshError(std::string("Session.remove_extra_edges: a rebuilt ") +
                       TopAbs::ShapeTypeToString(before.ShapeType()) +
                       " holds another number of children; the rebuild did not keep the "
                       "structure. This is a bug, not a caller error.");
  }
}

// The images of `s` in a substitution: its copies, each followed to its own copies.
void substitution_images(const BRepTools_Substitution& subst, const TopoDS_Shape& s,
                         int depth, std::vector<TopoDS_Shape>& out) {
  if (depth > 64) {
    throw PysmeshError(
        "Session.remove_extra_edges: a substitution of RefillProblemFaces copies a shape "
        "through more than 64 levels. This is a bug, not a caller error.");
  }
  for (const TopoDS_Shape& x : subst.Copy(s)) {
    if (subst.IsCopied(x)) {
      substitution_images(subst, x, depth + 1, out);
    } else {
      out.push_back(x);
    }
  }
}

// The faces, edges, vertices and solids of `shape`, counted.
std::array<int, 4> counts_of(const TopoDS_Shape& shape) {
  std::array<int, 4> out{};
  int i = 0;
  for (TopAbs_ShapeEnum k : kEntityKinds) {
    ShapeSet subs;
    TopExp::MapShapes(shape, k, subs);
    out[i++] = subs.Extent();
  }
  return out;
}

// "S solids, F faces, E edges, V vertices".
std::string counts_text(const std::array<int, 4>& c) {
  return std::to_string(c[0]) + " solids, " + std::to_string(c[1]) + " faces, " +
         std::to_string(c[2]) + " edges, " + std::to_string(c[3]) + " vertices";
}

// A face's area, by GProp at a precision of 1e-10.
double area_of(const TopoDS_Shape& face) {
  GProp_GProps props;
  BRepGProp::SurfaceProperties(face, props, 1e-10);
  return props.Mass();
}

// True for a face on a sphere or a cylinder: the faces that GEOM's RotateSphereSpace and
// FixRanges put on a turned copy of their surface.
bool turned_kind(const TopoDS_Shape& face) {
  const GeomAbs_SurfaceType type = BRepAdaptor_Surface(TopoDS::Face(face)).GetType();
  return type == GeomAbs_Sphere || type == GeomAbs_Cylinder;
}

// The ids, as "1, 2, 3".
std::string ids_text(const std::vector<EntityId>& ids) {
  std::string out;
  for (EntityId id : ids) {
    out += (out.empty() ? "" : ", ") + std::to_string(id);
  }
  return out;
}

// The BRepCheck statuses of `shape`, as "EDGE 3: BRepCheck_InvalidCurveOnSurface on FACE 2;
// ...", the first 20 of them; each ordinal counts its kind in TopExp::MapShapes order.
std::string findings_text(const TopoDS_Shape& shape,
                          const std::vector<shape_checks::CheckFinding>& findings) {
  if (findings.empty()) {
    return "BRepCheck_Analyzer reported no status on any sub-shape";
  }
  const auto named = [&](const TopoDS_Shape& sub) {
    ShapeSet all;
    TopExp::MapShapes(shape, sub.ShapeType(), all);
    return std::string(shape_checks::kind_text(sub)) + " " + std::to_string(all.FindIndex(sub));
  };
  constexpr std::size_t kListed = 20;
  std::string out;
  for (std::size_t i = 0; i < findings.size() && i < kListed; ++i) {
    const shape_checks::CheckFinding& f = findings[i];
    out += (i == 0 ? "" : "; ") + named(f.shape) + ": " + f.status;
    if (!f.context.IsNull()) {
      out += " on " + named(f.context);
    }
  }
  if (findings.size() > kListed) {
    out += "; and " + std::to_string(findings.size() - kListed) + " more";
  }
  return out;
}

}  // namespace

Handle(BRepTools_History) structural_history(const TopoDS_Shape& before,
                                             const TopoDS_Shape& after) {
  Handle(BRepTools_History) hist = new BRepTools_History;
  if (before.ShapeType() != after.ShapeType()) {
    throw PysmeshError(std::string("Session.remove_extra_edges: a ") +
                       TopAbs::ShapeTypeToString(before.ShapeType()) + " was rebuilt as a " +
                       TopAbs::ShapeTypeToString(after.ShapeType()) +
                       "; the rebuild did not keep the structure. This is a bug, not a "
                       "caller error.");
  }
  ShapeSet seen;
  if (BRepTools_History::IsSupportedType(before) && !before.IsSame(after)) {
    hist->AddModified(before, after);
  }
  seen.Add(before);
  pair_children(before, after, hist, seen);
  return hist;
}

Handle(BRepTools_History) substitution_history(const TopoDS_Shape& shape,
                                               const BRepTools_Substitution& subst) {
  Handle(BRepTools_History) hist = new BRepTools_History;
  for (TopAbs_ShapeEnum k : kEntityKinds) {
    ShapeSet subs;
    TopExp::MapShapes(shape, k, subs);
    for (int i = 1; i <= subs.Extent(); ++i) {
      const TopoDS_Shape& s = subs.FindKey(i);
      if (!subst.IsCopied(s)) {
        continue;
      }
      std::vector<TopoDS_Shape> images;
      substitution_images(subst, s, 0, images);
      if (images.empty()) {
        hist->Remove(s);
      }
      for (const TopoDS_Shape& m : images) {
        if (m.ShapeType() == k) {
          hist->AddModified(s, m);
        }
      }
    }
  }
  return hist;
}

void require_turned_faces_kept(const char* step, const TopoDS_Shape& before,
                               const TopoDS_Shape& after,
                               const std::vector<Handle(BRepTools_History)>& steps) {
  // On a quarter-sphere wedge GEOM's RotateSphereSpace gives a spherical face 3.8e-4 smaller
  // by GProp whose mesh reaches 0.033 outside the wedge, at tolerance 1e-7, and BRepCheck
  // accepts the solid (p4/explore/ree_sphere_mesh.log). The vendored class gives the same
  // shape. A turned face of another area is not the input face.
  const Handle(BRepTools_History) hist = compose_histories(before, steps);
  ShapeSet in_faces;
  TopExp::MapShapes(before, TopAbs_FACE, in_faces);
  ShapeSet out_faces;
  TopExp::MapShapes(after, TopAbs_FACE, out_faces);
  ShapeKeyed<int> sources;
  std::vector<std::pair<TopoDS_Shape, TopoDS_Shape>> pairs;
  for (int i = 1; i <= in_faces.Extent(); ++i) {
    const TopoDS_Shape& f = in_faces.FindKey(i);
    if (hist->IsRemoved(f)) {
      continue;
    }
    std::vector<TopoDS_Shape> images;
    for (const TopoDS_Shape& m : hist->Modified(f)) {
      if (out_faces.Contains(m)) {
        images.push_back(m);
      }
    }
    if (hist->Modified(f).IsEmpty() && out_faces.Contains(f)) {
      images.push_back(f);
    }
    for (const TopoDS_Shape& m : images) {
      ++sources[m];
    }
    if (images.size() == 1) {
      pairs.emplace_back(f, images.front());
    }
  }
  for (const auto& [f, g] : pairs) {
    if (f.IsSame(g) || sources[g] != 1 || !turned_kind(f) || !turned_kind(g)) {
      continue;
    }
    const double a = area_of(f);
    const double b = area_of(g);
    if (std::abs(b - a) > 1e-6 * std::abs(a)) {
      GProp_GProps props;
      BRepGProp::SurfaceProperties(f, props, 1e-10);
      const gp_Pnt c = props.CentreOfMass();
      const bool sphere = BRepAdaptor_Surface(TopoDS::Face(f)).GetType() == GeomAbs_Sphere;
      throw TurnedFaceError(std::string("GEOM's ") + step + " put the " +
                            (sphere ? "sphere" : "cylinder") + " face of area " +
                            number_text(a) + " centred at (" + number_text(c.X()) + ", " +
                            number_text(c.Y()) + ", " + number_text(c.Z()) +
                            ") on a turned surface as a face of area " + number_text(b) + " (" +
                            number_text((b - a) / a) +
                            " relative): the result is not the input shape");
    }
  }
}

py::dict Session::remove_extra_edges(const std::vector<EntityId>& entity_ids, bool union_faces,
                                     const py::object& progress, const py::object& cancel) {
  const char* op = "remove_extra_edges";
  OpGuard guard(in_op_);
  const ProgressHooks hooks = hooks_of(op, progress, cancel);
  if (entity_ids.empty()) {
    throw PysmeshError(
        "Session.remove_extra_edges: no entity is named; there is no body to improve.");
  }
  const std::vector<TopoDS_Shape> scope = scoped_bodies(entity_ids, op);
  const std::vector<TopoDS_Shape> untouched = bodies_excluding(scope);
  const TopoDS_Shape input = make_root(scope);

  ProgressDriver driver(op, hooks);
  TopoDS_Shape result;
  std::vector<Handle(BRepTools_History)> steps;
  bool rejected = false;
  std::vector<shape_checks::CheckFinding> findings;
  {
    py::gil_scoped_release release;
    try {
      // GEOMImpl_BlockDriver::Execute, BLOCK_REMOVE_EXTRA, in its order.
      Message_ProgressScope aPS(driver.range(), "remove_extra_edges", 7.0);

      // Copy shape to avoid problems (Mantis issue 0021683)
      TopoDS_Shape aShapeCopy;
      TColStd_IndexedDataMapOfTransientTransient aMapTShapes;
      TNaming_CopyShape::CopyTool(input, aMapTShapes, aShapeCopy);
      steps.push_back(structural_history(input, aShapeCopy));

      // 1. Improve solids with seam and/or degenerated edges. GEOM's tool keeps its default
      // tolerance (Precision::Confusion()); geomBuilder.RemoveExtraEdges passes
      // OptimumNbFaces 0 to unite faces and -1 not to.
      TopoDS_Shape aFixedExtra = geomcopy::BlockFixAPI_Perform(
          aShapeCopy, Precision::Confusion(), union_faces ? 0 : -1, aPS, steps);

      // Repair result: GEOM runs its default ShHealOper_ShapeProcess on a result that
      // BRepCheck rejects, with the absolute tolerances of its resource file. pySMESH runs
      // GEOM's check and refuses such a result.
      if (!aFixedExtra.IsNull() && aPS.More()) {
        BRepCheck_Analyzer ana(aFixedExtra, false);
        if (!ana.IsValid()) {
          rejected = true;
          findings = shape_checks::check_findings(aFixedExtra, false);
        }
        aPS.Next();
      }
      result = aFixedExtra;
    } catch (const PysmeshError&) {
      py::gil_scoped_acquire acquire;
      throw;
    } catch (const TurnedFaceError& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.remove_extra_edges: on the bodies of the entities ") +
                         ids_text(entity_ids) + ", " + e.what() + "; nothing is committed.");
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.remove_extra_edges: OCCT threw on the bodies "
                                     "of the entities ") +
                         ids_text(entity_ids) + ": " + e.what() + "; nothing is committed.");
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled(op);
  }
  if (rejected) {
    throw PysmeshError("Session.remove_extra_edges: GEOM's BlockFix gave a shape that "
                       "BRepCheck_Analyzer rejects for the bodies of the entities " +
                       ids_text(entity_ids) + " (" + std::to_string(findings.size()) +
                       " status(es): " + findings_text(result, findings) +
                       "). GEOM repairs such a shape with the healing of its resource file, "
                       "whose tolerances are absolute lengths; pySMESH does not repair it. "
                       "Nothing is committed.");
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.remove_extra_edges: GEOM's BlockFix produced no shape for the "
                       "bodies of the entities " +
                       ids_text(entity_ids) + "; nothing is committed.");
  }
  // A result with the input's counts removed no edge and united nothing: a refusal (an
  // operation that changes nothing is one).
  const std::array<int, 4> before = counts_of(input);
  const std::array<int, 4> after = counts_of(result);
  if (before == after) {
    throw PysmeshError("Session.remove_extra_edges: the bodies of the entities " +
                       ids_text(entity_ids) + " keep " + counts_text(before) +
                       (union_faces ? "" : " with faces not united") +
                       ": GEOM's BlockFix removed no seam, degenerated or extra edge and "
                       "united nothing; nothing is committed.");
  }
  const Handle(BRepTools_History) hist = compose_histories(input, steps);

  return commit(concat(untouched, result), hist, op, result, Validation::Report);
}

}  // namespace session
}  // namespace pysmesh
