// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-03

// pySMESH binding — what a committed shape must be, beyond BRepCheck_Analyzer.
//
// See shape_checks.hpp for what each statement asserts and why it is sound.

#include "shape_checks.hpp"

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <utility>

#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepGProp.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepTools.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>
#include <BRepTools_ReShape.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <GProp_GProps.hxx>
#include <Geom2d_Curve.hxx>
#include <NCollection_IndexedDataMap.hxx>
#include <NCollection_IndexedMap.hxx>
#include <NCollection_List.hxx>
#include <OSD_Parallel.hxx>
#include <Poly_Triangle.hxx>
#include <Poly_Triangulation.hxx>
#include <Precision.hxx>
#include <Standard_Failure.hxx>
#include <TopAbs_State.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Solid.hxx>
#include <TopoDS_Vertex.hxx>
#include <gp.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Vec.hxx>

namespace pysmesh {
namespace shape_checks {
namespace {

using ShapeMap = NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher>;

// Candidate points per solid, across all its faces, before the edge midpoints and vertices.
// The witness search runs only on a result that is already suspect, so its cost is paid
// rarely; the budget keeps it bounded on a solid of hundreds of faces.
constexpr int kGridBudget = 1024;
constexpr int kMaxGrid = 7;

// The parameters of the candidates on one face: a grid over its parameter box, kept where
// the face's own 2-D classifier places them inside, then the midpoint of every edge and every
// vertex, read on the face's own pcurves.
std::vector<gp_Pnt2d> face_parameters(const TopoDS_Face& face, int grid) {
  std::vector<gp_Pnt2d> out;
  double u0 = 0.0, u1 = 0.0, v0 = 0.0, v1 = 0.0;
  BRepTools::UVBounds(face, u0, u1, v0, v1);
  BRepTopAdaptor_FClass2d inside(face, Precision::PConfusion());
  for (int i = 0; i < grid; ++i) {
    for (int j = 0; j < grid; ++j) {
      const gp_Pnt2d uv(u0 + (i + 0.5) / grid * (u1 - u0), v0 + (j + 0.5) / grid * (v1 - v0));
      if (inside.Perform(uv) == TopAbs_IN) {
        out.push_back(uv);
      }
    }
  }
  for (TopExp_Explorer e(face, TopAbs_EDGE); e.More(); e.Next()) {
    const TopoDS_Edge& edge = TopoDS::Edge(e.Current());
    if (BRep_Tool::Degenerated(edge)) {
      continue;
    }
    double first = 0.0, last = 0.0;
    const Handle(Geom2d_Curve) pcurve = BRep_Tool::CurveOnSurface(edge, face, first, last);
    if (!pcurve.IsNull()) {
      out.push_back(pcurve->Value(0.5 * (first + last)));
    }
  }
  for (TopExp_Explorer v(face, TopAbs_VERTEX); v.More(); v.Next()) {
    try {
      out.push_back(BRep_Tool::Parameters(TopoDS::Vertex(v.Current()), face));
    } catch (const Standard_Failure&) {
      // A vertex with no parameters on this face (no pcurve through it) offers no
      // candidate here; it is still reached from the faces that do carry it.
    }
  }
  return out;
}

// Points near the boundary of `solid`, moved into it by `depth` along the face normal, and
// kept where the solid itself classifies them IN at `tol`. Both normal directions are tried,
// so the answer does not depend on the face orientation being right.
std::vector<gp_Pnt> interior_candidates(const TopoDS_Shape& solid, double depth, double tol) {
  ShapeMap faces;
  TopExp::MapShapes(solid, TopAbs_FACE, faces);
  std::vector<gp_Pnt> out;
  if (faces.Extent() == 0) {
    return out;
  }
  const int grid = std::clamp(
      static_cast<int>(std::sqrt(static_cast<double>(kGridBudget) / faces.Extent())), 1,
      kMaxGrid);
  BRepClass3d_SolidClassifier own(solid);
  for (int f = 1; f <= faces.Extent(); ++f) {
    const TopoDS_Face& face = TopoDS::Face(faces.FindKey(f));
    const BRepAdaptor_Surface surface(face);
    for (const gp_Pnt2d& uv : face_parameters(face, grid)) {
      gp_Pnt p;
      gp_Vec du, dv;
      surface.D1(uv.X(), uv.Y(), p, du, dv);
      gp_Vec normal = du.Crossed(dv);
      if (normal.Magnitude() <= gp::Resolution()) {
        continue;
      }
      normal.Normalize();
      for (const double side : {1.0, -1.0}) {
        const gp_Pnt c = p.Translated(side * depth * normal);
        own.Perform(c, tol);
        if (own.State() == TopAbs_IN) {
          out.push_back(c);
          break;
        }
      }
    }
  }
  return out;
}

// One classifier per solid, each loaded once.
std::vector<std::unique_ptr<BRepClass3d_SolidClassifier>> classifiers(
    const std::vector<TopoDS_Shape>& solids) {
  std::vector<std::unique_ptr<BRepClass3d_SolidClassifier>> out;
  out.reserve(solids.size());
  for (const TopoDS_Shape& s : solids) {
    out.push_back(std::make_unique<BRepClass3d_SolidClassifier>(s));
  }
  return out;
}

// A candidate of `from` that a solid of `against` classifies IN.
std::optional<gp_Pnt> inside_any(const std::vector<TopoDS_Shape>& from,
                                 const std::vector<TopoDS_Shape>& against, double depth,
                                 double tol) {
  const auto others = classifiers(against);
  for (const TopoDS_Shape& s : from) {
    for (const gp_Pnt& p : interior_candidates(s, depth, tol)) {
      for (const auto& c : others) {
        c->Perform(p, tol);
        if (c->State() == TopAbs_IN) {
          return p;
        }
      }
    }
  }
  return std::nullopt;
}

}  // namespace

double max_tolerance(const std::vector<TopoDS_Shape>& shapes) {
  double tol = Precision::Confusion();
  for (const TopoDS_Shape& s : shapes) {
    for (TopExp_Explorer v(s, TopAbs_VERTEX); v.More(); v.Next()) {
      tol = std::max(tol, BRep_Tool::Tolerance(TopoDS::Vertex(v.Current())));
    }
    for (TopExp_Explorer e(s, TopAbs_EDGE); e.More(); e.Next()) {
      tol = std::max(tol, BRep_Tool::Tolerance(TopoDS::Edge(e.Current())));
    }
    for (TopExp_Explorer f(s, TopAbs_FACE); f.More(); f.Next()) {
      tol = std::max(tol, BRep_Tool::Tolerance(TopoDS::Face(f.Current())));
    }
  }
  return tol;
}

std::vector<TopoDS_Shape> solids_of(const std::vector<TopoDS_Shape>& shapes) {
  std::vector<TopoDS_Shape> out;
  for (const TopoDS_Shape& s : shapes) {
    ShapeMap solids;
    TopExp::MapShapes(s, TopAbs_SOLID, solids);
    for (int i = 1; i <= solids.Extent(); ++i) {
      out.push_back(solids.FindKey(i));
    }
  }
  return out;
}

std::optional<gp_Pnt> point_inside_both(const std::vector<TopoDS_Shape>& a,
                                        const std::vector<TopoDS_Shape>& b, double depth,
                                        double tol) {
  const std::vector<TopoDS_Shape> sa = solids_of(a);
  const std::vector<TopoDS_Shape> sb = solids_of(b);
  if (std::optional<gp_Pnt> p = inside_any(sa, sb, depth, tol)) {
    return p;
  }
  return inside_any(sb, sa, depth, tol);
}

std::optional<gp_Pnt> point_inside_first_outside_second(const std::vector<TopoDS_Shape>& a,
                                                        const std::vector<TopoDS_Shape>& b,
                                                        double depth, double tol) {
  const auto others = classifiers(solids_of(b));
  for (const TopoDS_Shape& s : solids_of(a)) {
    for (const gp_Pnt& p : interior_candidates(s, depth, tol)) {
      bool outside = true;
      for (const auto& c : others) {
        c->Perform(p, tol);
        if (c->State() != TopAbs_OUT) {
          outside = false;
          break;
        }
      }
      if (outside) {
        return p;
      }
    }
  }
  return std::nullopt;
}

std::string point_text(const gp_Pnt& p) {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "(%.9g, %.9g, %.9g)", p.X(), p.Y(), p.Z());
  return buf;
}

std::vector<TopoDS_Shape> free_boundary_edges(const TopoDS_Shape& shape) {
  ShapeMap edges;
  TopExp::MapShapes(shape, TopAbs_EDGE, edges);
  NCollection_IndexedDataMap<TopoDS_Shape, NCollection_List<TopoDS_Shape>,
                             TopTools_ShapeMapHasher>
      edge_faces;
  TopExp::MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
  std::vector<TopoDS_Shape> out;
  for (int i = 1; i <= edges.Extent(); ++i) {
    const TopoDS_Edge& e = TopoDS::Edge(edges.FindKey(i));
    if (BRep_Tool::Degenerated(e) || !edge_faces.Contains(e)) {
      continue;
    }
    if (edge_faces.FindFromKey(e).Extent() == 1) {
      out.push_back(e);
    }
  }
  return out;
}

TessellatedVolume tessellated_volume(const TopoDS_Shape& solid, double deflection) {
  TessellatedVolume out;
  const TopoDS_Shape copy =
      BRepBuilderAPI_Copy(solid, /*copyGeom=*/false, /*copyMesh=*/false).Shape();
  // Only the linear deflection bounds the volume, so the angular one is left coarse: it
  // would refine curved faces for no gain in the bound.
  BRepMesh_IncrementalMesh mesher(copy, deflection, /*isRelative=*/false, /*angDeflection=*/1.5,
                                  /*isInParallel=*/true);
  for (TopExp_Explorer ex(copy, TopAbs_FACE); ex.More(); ex.Next()) {
    const TopoDS_Face& face = TopoDS::Face(ex.Current());
    TopLoc_Location location;
    const Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(face, location);
    if (tri.IsNull() || tri->NbTriangles() == 0) {
      return TessellatedVolume();
    }
    const gp_Trsf trsf = location.Transformation();
    const bool reversed = face.Orientation() == TopAbs_REVERSED;
    for (int i = 1; i <= tri->NbTriangles(); ++i) {
      int n1 = 0, n2 = 0, n3 = 0;
      tri->Triangle(i).Get(n1, n2, n3);
      if (reversed) {
        std::swap(n2, n3);
      }
      const gp_XYZ p1 = tri->Node(n1).Transformed(trsf).XYZ();
      const gp_XYZ p2 = tri->Node(n2).Transformed(trsf).XYZ();
      const gp_XYZ p3 = tri->Node(n3).Transformed(trsf).XYZ();
      out.volume += p1.Dot(p2.Crossed(p3)) / 6.0;
      out.area += 0.5 * (p2 - p1).Crossed(p3 - p1).Modulus();
    }
  }
  out.complete = true;
  return out;
}

EnclosedVolume enclosed_volume(const TopoDS_Shape& solid, bool precise) {
  const double eps = Precision::Confusion();
  EnclosedVolume out;
  Bnd_Box box;
  BRepBndLib::Add(solid, box);
  const double diagonal = std::max(std::sqrt(box.SquareExtent()), eps);
  if (!precise) {
    const double deflection = kSignDeflection * diagonal;
    const TessellatedVolume tess = tessellated_volume(solid, deflection);
    if (tess.complete && tess.volume > (deflection + eps) * tess.area) {
      out.volume = tess.volume;
      out.area = tess.area;
      out.tolerance = eps * tess.area;
      out.exact = false;
      return out;
    }
  }
  GProp_GProps surface;
  BRepGProp::SurfaceProperties(solid, surface);
  out.area = surface.Mass();
  out.tolerance = eps * out.area;
  // The most the faces' contributions can add up to, whatever cancels between them.
  const double lever = diagonal * out.area / 3.0;
  const double tight = std::min(0.3 * eps / diagonal, kAdaptiveEpsCap);
  for (const double precision : {kAdaptiveEpsCap, tight}) {
    GProp_GProps volume;
    const double reached = BRepGProp::VolumeProperties(solid, volume, precision);
    out.volume = volume.Mass();
    if (std::abs(out.volume) > out.tolerance + std::max(precision, reached) * lever ||
        precision <= tight) {
      break;
    }
  }
  return out;
}


std::vector<InsideOutSolid> inside_out_solids(const TopoDS_Shape& shape) {
  std::vector<InsideOutSolid> out;
  const std::vector<TopoDS_Shape> solids = solids_of({shape});
  // One classification per solid, the solids side by side: each classifier reads its own
  // solid only, and the states land in their own slots, so the answer does not depend on
  // the order the threads finish in. On the production assembly (117 solids) this check
  // cost 0.40 s in one thread.
  std::vector<TopAbs_State> states(solids.size(), TopAbs_UNKNOWN);
  OSD_Parallel::For(0, static_cast<int>(solids.size()), [&](const int i) {
    BRepClass3d_SolidClassifier where(solids[static_cast<std::size_t>(i)]);
    where.PerformInfinitePoint(Precision::Confusion());
    states[static_cast<std::size_t>(i)] = where.State();
  });
  for (std::size_t i = 0; i < solids.size(); ++i) {
    if (states[i] != TopAbs_IN) {
      continue;
    }
    const EnclosedVolume enclosed = enclosed_volume(solids[i]);
    if (enclosed.volume < -enclosed.tolerance) {
      out.push_back({static_cast<int>(i) + 1, enclosed.volume});
    }
  }
  return out;
}

TopoDS_Shape reverse_solids(const TopoDS_Shape& shape, const std::vector<InsideOutSolid>& solids) {
  const std::vector<TopoDS_Shape> all = solids_of({shape});
  const Handle(BRepTools_ReShape) reshape = new BRepTools_ReShape;
  BRep_Builder builder;
  for (const InsideOutSolid& s : solids) {
    const TopoDS_Shape& old = all[static_cast<std::size_t>(s.ordinal - 1)];
    TopoDS_Solid fixed;
    builder.MakeSolid(fixed);
    for (TopoDS_Iterator it(old); it.More(); it.Next()) {
      builder.Add(fixed, it.Value().Reversed());
    }
    if (shape.IsSame(old)) {
      return fixed;
    }
    reshape->Replace(old, fixed);
  }
  return reshape->Apply(shape);
}

std::string inside_out_refusal(const std::string& op, const std::vector<InsideOutSolid>& solids) {
  std::string names;
  for (std::size_t i = 0; i < solids.size(); ++i) {
    char volume[32];
    std::snprintf(volume, sizeof(volume), "%.9g", solids[i].volume);
    names += (i == 0 ? "" : ", ") + std::string("solid ") +
             std::to_string(solids[i].ordinal) + " (volume " + volume + ")";
  }
  return op + ": the BREP holds " + std::to_string(solids.size()) +
         " solid(s) that are inside out: " + names +
         ". The point at infinity classifies inside each, and its volume is negative. Pass "
         "inside_out=\"reverse\" to reverse them; nothing is imported.";
}

std::string inside_out_reversed(const InsideOutSolid& solid) {
  char volume[32];
  std::snprintf(volume, sizeof(volume), "%.9g", solid.volume);
  return "solid " + std::to_string(solid.ordinal) + " of the BREP was inside out (volume " +
         volume + "); it is reversed";
}

bool reverse_inside_out(const std::string& op, const std::string& policy) {
  if (policy == "reverse") {
    return true;
  }
  if (policy == "raise") {
    return false;
  }
  throw PysmeshError(op + ": inside_out must be \"raise\" or \"reverse\" (got \"" + policy +
                     "\").");
}

std::string edge_text(const TopoDS_Shape& edge) {
  const TopoDS_Edge& e = TopoDS::Edge(edge);
  const gp_Pnt a = BRep_Tool::Pnt(TopExp::FirstVertex(e));
  const gp_Pnt b = BRep_Tool::Pnt(TopExp::LastVertex(e));
  const BRepAdaptor_Curve curve(e);
  char length[32];
  std::snprintf(length, sizeof(length), "%.9g", GCPnts_AbscissaPoint::Length(curve));
  return "from " + point_text(a) + " to " + point_text(b) + ", length " + length;
}

}  // namespace shape_checks
}  // namespace pysmesh
