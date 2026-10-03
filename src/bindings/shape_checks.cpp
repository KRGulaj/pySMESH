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
#include <sstream>
#include <queue>
#include <utility>

#include <BOPAlgo_CheckerSI.hxx>
#include <BOPDS_DS.hxx>
#include <BOPDS_Pair.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepCheck.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepCheck_Result.hxx>
#include <BRepCheck_Status.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepGProp.hxx>
#include <BRepLib_FindSurface.hxx>
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
#include <GeomAbs_Shape.hxx>
#include <IntTools_Curve.hxx>
#include <IntTools_FaceFace.hxx>
#include <IntTools_PntOn2Faces.hxx>
#include <IntTools_PntOnFace.hxx>
#include <Geom2d_Curve.hxx>
#include <Geom_Plane.hxx>
#include <NCollection_Array1.hxx>
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
#include <gp_Dir.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Vec.hxx>
#include <math.hxx>
#include <math_Vector.hxx>

namespace pysmesh {
namespace shape_checks {
namespace {

using ShapeMap = NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher>;

// Grid points per solid, across all its faces, and the cap on all its candidates (grid
// points, edge midpoints, vertices). The witness search runs only on a result that is already
// suspect, but a correct empty result tests every candidate: a cut of 20 parts of the
// production assembly by a box that holds them all made 40 s of it before the caps and the
// order of the tests below.
constexpr int kGridBudget = 256;
constexpr int kMaxGrid = 7;
constexpr std::size_t kCandidateCap = 512;

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

// Points near the boundary of `solid`, moved by `depth` against the outward normal of their
// face (the surface normal, reversed on a REVERSED face), at most kCandidateCap of them. On a
// solid whose faces point out these lie inside it. They are not classified here: a candidate
// becomes a witness only once its own solid classifies it IN (Solid::inside), which is tested
// last because it is the dear test on a solid of many faces.
std::vector<gp_Pnt> interior_candidates(const TopoDS_Shape& solid, double depth) {
  ShapeMap faces;
  TopExp::MapShapes(solid, TopAbs_FACE, faces);
  std::vector<gp_Pnt> out;
  if (faces.Extent() == 0) {
    return out;
  }
  const int grid = std::clamp(
      static_cast<int>(std::sqrt(static_cast<double>(kGridBudget) / faces.Extent())), 1,
      kMaxGrid);
  for (int f = 1; f <= faces.Extent() && out.size() < kCandidateCap; ++f) {
    const TopoDS_Face& face = TopoDS::Face(faces.FindKey(f));
    const BRepAdaptor_Surface surface(face);
    const double outward = face.Orientation() == TopAbs_REVERSED ? -1.0 : 1.0;
    for (const gp_Pnt2d& uv : face_parameters(face, grid)) {
      gp_Pnt p;
      gp_Vec du, dv;
      surface.D1(uv.X(), uv.Y(), p, du, dv);
      gp_Vec normal = du.Crossed(dv);
      if (normal.Magnitude() <= gp::Resolution()) {
        continue;
      }
      normal.Normalize();
      out.push_back(p.Translated(-outward * depth * normal));
      if (out.size() >= kCandidateCap) {
        break;
      }
    }
  }
  return out;
}

// One solid of an operand group: its box, grown by the tolerance, and its classifier, built on
// first use.
class Solid {
 public:
  Solid(const TopoDS_Shape& shape, double tol) : shape_(shape), tol_(tol) {
    BRepBndLib::Add(shape_, box_);
    box_.Enlarge(tol_);
  }

  const TopoDS_Shape& shape() const { return shape_; }

  // The state of `p` at the tolerance; OUT without classifying when `p` is outside the box.
  TopAbs_State state(const gp_Pnt& p) {
    if (box_.IsOut(p)) {
      return TopAbs_OUT;
    }
    if (!classifier_) {
      classifier_ = std::make_unique<BRepClass3d_SolidClassifier>(shape_);
    }
    classifier_->Perform(p, tol_);
    return classifier_->State();
  }

  bool inside(const gp_Pnt& p) { return state(p) == TopAbs_IN; }

 private:
  TopoDS_Shape shape_;
  double tol_;
  Bnd_Box box_;
  std::unique_ptr<BRepClass3d_SolidClassifier> classifier_;
};

std::vector<Solid> solids_with_boxes(const std::vector<TopoDS_Shape>& shapes, double tol) {
  std::vector<Solid> out;
  for (const TopoDS_Shape& s : solids_of(shapes)) {
    out.emplace_back(s, tol);
  }
  return out;
}

// A candidate of `from` that a solid of `against` classifies IN, and its own solid too.
std::optional<gp_Pnt> inside_any(std::vector<Solid>& from, std::vector<Solid>& against,
                                 double depth) {
  for (Solid& s : from) {
    for (const gp_Pnt& p : interior_candidates(s.shape(), depth)) {
      for (Solid& other : against) {
        if (other.inside(p)) {
          if (s.inside(p)) {
            return p;
          }
          break;
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
  std::vector<Solid> sa = solids_with_boxes(a, tol);
  std::vector<Solid> sb = solids_with_boxes(b, tol);
  if (std::optional<gp_Pnt> p = inside_any(sa, sb, depth)) {
    return p;
  }
  return inside_any(sb, sa, depth);
}

std::optional<gp_Pnt> point_inside_first_outside_second(const std::vector<TopoDS_Shape>& a,
                                                        const std::vector<TopoDS_Shape>& b,
                                                        double depth, double tol) {
  std::vector<Solid> sa = solids_with_boxes(a, tol);
  std::vector<Solid> sb = solids_with_boxes(b, tol);
  for (Solid& s : sa) {
    for (const gp_Pnt& p : interior_candidates(s.shape(), depth)) {
      bool outside = true;
      for (Solid& other : sb) {
        if (other.state(p) != TopAbs_OUT) {
          outside = false;
          break;
        }
      }
      if (outside && s.inside(p)) {
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

namespace {

// A sub-shape of `shape` in words: kind, 1-based ordinal among its kind, and its geometry.
std::string sub_shape_text(const TopoDS_Shape& shape, const TopoDS_Shape& sub) {
  ShapeMap all;
  TopExp::MapShapes(shape, sub.ShapeType(), all);
  const std::string ordinal = std::to_string(all.FindIndex(sub));
  if (sub.ShapeType() == TopAbs_FACE) {
    static const char* const kSurfaces[] = {"Plane",          "Cylinder",        "Cone",
                                            "Sphere",         "Torus",           "BezierSurface",
                                            "BSplineSurface", "SurfaceOfRevolution",
                                            "SurfaceOfExtrusion", "OffsetSurface",
                                            "OtherSurface"};
    const int type = BRepAdaptor_Surface(TopoDS::Face(sub)).GetType();
    return "face " + ordinal + " (" + kSurfaces[std::clamp(type, 0, 10)] + ")";
  }
  if (sub.ShapeType() == TopAbs_EDGE) {
    return "edge " + ordinal + " (" + edge_text(sub) + ")";
  }
  if (sub.ShapeType() == TopAbs_VERTEX) {
    return "vertex " + ordinal + " " + point_text(BRep_Tool::Pnt(TopoDS::Vertex(sub)));
  }
  return "sub-shape " + ordinal;
}

// The distance from `p` to the boundary of `face`: its edges, its seams included.
double distance_to_boundary(const gp_Pnt& p, const TopoDS_Face& face) {
  const TopoDS_Vertex v = BRepBuilderAPI_MakeVertex(p).Vertex();
  double best = RealLast();
  for (TopExp_Explorer e(face, TopAbs_EDGE); e.More(); e.Next()) {
    if (BRep_Tool::Degenerated(TopoDS::Edge(e.Current()))) {
      continue;
    }
    BRepExtrema_DistShapeShape d(v, e.Current());
    if (d.IsDone()) {
      best = std::min(best, d.Value());
    }
  }
  return best;
}

// Where `face` passes through itself away from its own boundary, if anywhere.
//
// BOPAlgo_CheckerSI intersects each free-form face with itself (IntTools_FaceFace(F, F)) and
// reports the face on any intersection. A face whose surface closes on itself through a
// sharp seam — a loft through a one-edge section with a sharp trailing edge — meets itself
// along that seam, which is its own boundary edge, and is reported although nothing
// crosses. So the same intersection is repeated here, and a point of it counts only when it
// lies farther from every edge of the face than ten times the intersection's tolerance and
// the face's.
std::optional<std::pair<gp_Pnt, double>> face_self_crossing(const TopoDS_Face& face) {
  IntTools_FaceFace ff;
  ff.Perform(face, face, false);
  if (!ff.IsDone()) {
    return std::nullopt;
  }
  const double tol_face = BRep_Tool::Tolerance(face);
  std::optional<std::pair<gp_Pnt, double>> worst;
  const auto consider = [&](const gp_Pnt& p, double tol) {
    const double d = distance_to_boundary(p, face);
    if (d > 10.0 * std::max(tol, tol_face) && (!worst || d > worst->second)) {
      worst = std::make_pair(p, d);
    }
  };
  for (const IntTools_Curve& c : ff.Lines()) {
    double first = 0.0, last = 0.0;
    gp_Pnt a, b;
    if (!c.Bounds(first, last, a, b)) {
      continue;
    }
    for (int k = 1; k <= 9; ++k) {
      gp_Pnt p;
      if (c.D0(first + (last - first) * k / 10.0, p)) {
        consider(p, c.Tolerance());
      }
    }
  }
  for (const IntTools_PntOn2Faces& q : ff.Points()) {
    consider(q.P1().Pnt(), tol_face);
  }
  return worst;
}

}  // namespace

std::string self_interference_refusal(const std::string& op, const TopoDS_Shape& shape) {
  std::string pairs;
  int count = 0;
  const auto add = [&](const std::string& names) {
    ++count;
    pairs += "Interference " + std::to_string(count) + ": " + names + ". ";
  };
  for (const TopoDS_Shape& solid : solids_of({shape})) {
    // The pairs: BOPAlgo_CheckerSI up to level 4 (vertex/vertex, vertex/edge, edge/edge,
    // vertex/face, edge/face). Two faces that cross each other cross along a curve that
    // reaches their boundaries, so an edge of one meets the other. Level 5 adds face/face
    // pairs and the face's intersection with itself, which reports every face that closes
    // through a sharp seam, so that part is done below instead, once, with the seam left
    // out (face_self_crossing).
    BOPAlgo_CheckerSI checker;
    NCollection_List<TopoDS_Shape> args;
    args.Append(solid);
    checker.SetArguments(args);
    checker.SetNonDestructive(true);
    checker.SetLevelOfCheck(4);
    checker.Perform();
    if (!checker.HasErrors()) {
      const BOPDS_DS& ds = *checker.PDS();
      for (NCollection_Map<BOPDS_Pair>::Iterator it(ds.Interferences()); it.More();
           it.Next()) {
        int n1 = 0, n2 = 0;
        it.Value().Indices(n1, n2);
        if (ds.IsNewShape(n1) || ds.IsNewShape(n2)) {
          continue;
        }
        add(sub_shape_text(shape, ds.Shape(n1)) + " with " + sub_shape_text(shape, ds.Shape(n2)));
      }
    }
    // The faces against themselves, for the free-form ones: OCCT's CheckerSI skips the
    // plane, the cylinder, the cone, the sphere and a torus whose tube clears its axis,
    // which cannot cross themselves.
    for (TopExp_Explorer ex(solid, TopAbs_FACE); ex.More(); ex.Next()) {
      const TopoDS_Face& face = TopoDS::Face(ex.Current());
      const BRepAdaptor_Surface surface(face, false);
      const GeomAbs_SurfaceType type = surface.GetType();
      if (type == GeomAbs_Plane || type == GeomAbs_Cylinder || type == GeomAbs_Cone ||
          type == GeomAbs_Sphere ||
          (type == GeomAbs_Torus && surface.Torus().MajorRadius() >
                                        surface.Torus().MinorRadius() + Precision::Confusion())) {
        continue;
      }
      const std::optional<std::pair<gp_Pnt, double>> crossing = face_self_crossing(face);
      if (!crossing) {
        continue;
      }
      char away[32];
      std::snprintf(away, sizeof(away), "%.3g", crossing->second);
      add(sub_shape_text(shape, face) + " crosses itself at " + point_text(crossing->first) +
          ", " + away + " from its edges");
    }
  }
  if (count == 0) {
    return std::string();
  }
  return op + ": the solid's boundary interferes with itself in " + std::to_string(count) +
         " place(s), so the solid has no consistent inside; the session is unchanged.\n" +
         pairs;
}

std::vector<CheckFinding> check_findings(const TopoDS_Shape& shape) {
  std::vector<CheckFinding> out;
  const BRepCheck_Analyzer analyzer(shape);
  const auto name_of = [](BRepCheck_Status status) {
    std::ostringstream s;
    BRepCheck::Print(status, s);
    std::string text = s.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
      text.pop_back();
    }
    return text;
  };
  const auto add = [&](const TopoDS_Shape& sub, BRepCheck_Status status,
                       const TopoDS_Shape& context) {
    if (status == BRepCheck_NoError) {
      return;
    }
    const std::string name = name_of(status);
    for (const CheckFinding& f : out) {
      if (f.shape.IsSame(sub) && f.status == name && f.context.IsSame(context)) {
        return;
      }
    }
    out.push_back({sub, name, context});
  };
  for (const TopAbs_ShapeEnum kind : {TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE,
                                      TopAbs_EDGE, TopAbs_VERTEX}) {
    ShapeMap subs;
    TopExp::MapShapes(shape, kind, subs);
    for (int i = 1; i <= subs.Extent(); ++i) {
      const TopoDS_Shape& sub = subs.FindKey(i);
      Handle(BRepCheck_Result) result;
      try {
        result = analyzer.Result(sub);
      } catch (const Standard_Failure&) {
        continue;  // a sub-shape the analyzer did not record has no status to report
      }
      if (result.IsNull()) {
        continue;
      }
      for (const BRepCheck_Status status : result->Status()) {
        add(sub, status, TopoDS_Shape());
      }
      for (result->InitContextIterator(); result->MoreShapeInContext();
           result->NextShapeInContext()) {
        for (const BRepCheck_Status status : result->StatusOnShape()) {
          add(sub, status, result->ContextualShape());
        }
      }
    }
  }
  return out;
}

std::vector<TopoDS_Shape> invalid_faces(const TopoDS_Shape& shape) {
  std::vector<TopoDS_Shape> out;
  for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next()) {
    if (!BRepCheck_Analyzer(ex.Current()).IsValid()) {
      out.push_back(ex.Current());
    }
  }
  return out;
}

const char* kind_text(const TopoDS_Shape& shape) {
  switch (shape.ShapeType()) {
    case TopAbs_COMPOUND:
      return "COMPOUND";
    case TopAbs_COMPSOLID:
      return "COMPSOLID";
    case TopAbs_SOLID:
      return "SOLID";
    case TopAbs_SHELL:
      return "SHELL";
    case TopAbs_FACE:
      return "FACE";
    case TopAbs_WIRE:
      return "WIRE";
    case TopAbs_EDGE:
      return "EDGE";
    case TopAbs_VERTEX:
      return "VERTEX";
    default:
      return "SHAPE";
  }
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

namespace {

// ---- mass properties at a precision ---------------------------------------------------- //

// One piece of an edge's parameter range under the 15-point Gauss-Kronrod rule.
struct Piece {
  double a = 0.0;
  double b = 0.0;
  double length = 0.0;  // the Kronrod estimate of the arc length over [a, b]
  double error = 0.0;   // |Kronrod - Gauss|: the rule's estimate of the error in `length`
  gp_XYZ moment;        // the Kronrod estimate of the integral of C(t) |C'(t)| over [a, b]
};

// The 15-point Kronrod rule and the 7-point Gauss rule nested in it, from OCCT's tables.
class KronrodRule {
 public:
  KronrodRule() : kronrod_p_(1, 15), kronrod_w_(1, 15), gauss_w_(1, 7) {
    math_Vector gauss_p(1, 7);
    if (!math::KronrodPointsAndWeights(15, kronrod_p_, kronrod_w_) ||
        !math::OrderedGaussPointsAndWeights(7, gauss_p, gauss_w_)) {
      throw PysmeshError(
          "mass properties: OCCT's math tables gave no 15-point Gauss-Kronrod rule.");
    }
  }

  // Every even Kronrod node is a node of the nested Gauss rule (math.hxx), so one set of
  // curve evaluations gives both estimates.
  Piece integrate(const BRepAdaptor_Curve& c, double a, double b) const {
    const double half = 0.5 * (b - a);
    const double mid = 0.5 * (a + b);
    double kronrod = 0.0;
    double gauss = 0.0;
    gp_XYZ moment;
    for (int i = 1; i <= 15; ++i) {
      gp_Pnt p;
      gp_Vec d1;
      c.D1(mid + half * kronrod_p_(i), p, d1);
      const double speed = d1.Magnitude();
      kronrod += kronrod_w_(i) * speed;
      moment += p.XYZ() * (kronrod_w_(i) * speed);
      if (i % 2 == 0) {
        gauss += gauss_w_(i / 2) * speed;
      }
    }
    return {a, b, half * kronrod, std::abs(half * (kronrod - gauss)), moment * half};
  }

 private:
  math_Vector kronrod_p_;
  math_Vector kronrod_w_;
  math_Vector gauss_w_;
};

// A point's measure is 0 and its centre is itself. The bounding box is exact for a point,
// and it is also the centre of a degenerate edge, which has no curve to integrate.
Measure point_props(const TopoDS_Shape& s) {
  Bnd_Box box;
  BRepBndLib::Add(s, box);
  double a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
  box.Get(a, b, c, d, e, f);
  Measure out;
  out.centroid = gp_XYZ(0.5 * (a + d), 0.5 * (b + e), 0.5 * (c + f));
  return out;
}

// Bisections one edge may take before the rule stops. A piece's error falls as a high power
// of its width on a smooth curve, so a converging edge needs a few dozen; the cap only ends
// an edge the rule cannot resolve, and the error it then reports is above the precision.
constexpr int kMaxEdgeBisections = 20000;

// Arc length and centroid of an edge, integrated adaptively.
//
// BRepGProp has no adaptive rule for a curve: LinearProperties takes no precision. Its fixed
// rule is exact on a line or a circle and not on a free-form edge. On a spline through six
// points it measured 18.5768806517 against a true length of 18.5656473408, 6.05e-4 high.
//
// The range is first split at every parameter where the curve's continuity drops below CN,
// the knots of a B-spline. Each span is then polynomial, where Gauss-Kronrod converges
// fastest. The rule then bisects the piece with the largest error estimate until the
// summed estimate is within precision x the length. That is the criterion GProp applies to a
// face's area. The centroid's first moments are integrated on the same nodes, so the
// centroid follows the rule the length follows.
Measure adaptive_edge(const TopoDS_Edge& e, double precision, const KronrodRule& rule) {
  BRepAdaptor_Curve c(e);
  const int spans = c.NbIntervals(GeomAbs_CN);
  NCollection_Array1<double> bounds(1, spans + 1);
  c.Intervals(bounds, GeomAbs_CN);

  const auto worse = [](const Piece& x, const Piece& y) { return x.error < y.error; };
  std::priority_queue<Piece, std::vector<Piece>, decltype(worse)> pieces(worse);
  double length = 0.0;
  double error = 0.0;
  for (int i = 1; i <= spans; ++i) {
    const Piece p = rule.integrate(c, bounds(i), bounds(i + 1));
    length += p.length;
    error += p.error;
    pieces.push(p);
  }
  for (int n = 0; n < kMaxEdgeBisections && error > precision * length; ++n) {
    const Piece worst = pieces.top();
    pieces.pop();
    const double mid = 0.5 * (worst.a + worst.b);
    const Piece left = rule.integrate(c, worst.a, mid);
    const Piece right = rule.integrate(c, mid, worst.b);
    length += left.length + right.length - worst.length;
    error += left.error + right.error - worst.error;
    pieces.push(left);
    pieces.push(right);
  }

  // Summed again from the pieces: the running totals carry the round-off of every update.
  Measure out;
  gp_XYZ moment;
  double sum_error = 0.0;
  while (!pieces.empty()) {
    const Piece& p = pieces.top();
    out.mass += p.length;
    sum_error += p.error;
    moment += p.moment;
    pieces.pop();
  }
  // An edge whose curve does not move has no length to weight a centroid by: it is a point.
  if (!(out.mass > 0.0)) {
    return point_props(e);
  }
  out.centroid = moment / out.mass;
  out.error = sum_error / out.mass;
  return out;
}

// One shape's properties integrated adaptively to `precision`, a relative error.
//
// A solid and a face go to GProp's adaptive rule, which refines each face until two steps
// agree to `precision` relative and returns its estimate of the relative error reached over
// the whole shape (BRepGProp.hxx). An edge goes to adaptive_edge above.
Measure measure_with(const TopoDS_Shape& s, double precision, const KronrodRule& rule) {
  GProp_GProps props;
  Measure out;
  switch (s.ShapeType()) {
    case TopAbs_SOLID:
      out.error = BRepGProp::VolumeProperties(s, props, precision);
      break;
    case TopAbs_FACE:
      out.error = BRepGProp::SurfaceProperties(s, props, precision);
      break;
    case TopAbs_EDGE: {
      const TopoDS_Edge& e = TopoDS::Edge(s);
      if (BRep_Tool::Degenerated(e) || !BRep_Tool::IsGeometric(e)) {
        return point_props(s);
      }
      return adaptive_edge(e, precision, rule);
    }
    default:
      return point_props(s);
  }
  out.mass = props.Mass();
  out.centroid = props.CentreOfMass().XYZ();
  return out;
}

}  // namespace

Measure measure(const TopoDS_Shape& s, double precision) {
  static const KronrodRule rule;
  return measure_with(s, precision, rule);
}

std::vector<Measure> measures(const std::vector<TopoDS_Shape>& shapes, double precision) {
  static const KronrodRule rule;
  std::vector<Measure> out(shapes.size());
  // Each shape is integrated on its own, so the order of the threads changes nothing.
  OSD_Parallel::For(0, static_cast<int>(shapes.size()), [&](const int i) {
    out[static_cast<std::size_t>(i)] =
        measure_with(shapes[static_cast<std::size_t>(i)], precision, rule);
  });
  return out;
}

Bnd_Box exact_box(const TopoDS_Shape& s) {
  Bnd_Box box;
  BRepBndLib::AddOptimal(s, box, /*useTriangulation=*/false, /*useShapeTolerance=*/false);
  return box;
}

std::vector<Bnd_Box> exact_boxes(const std::vector<TopoDS_Shape>& shapes) {
  std::vector<Bnd_Box> out(shapes.size());
  OSD_Parallel::For(0, static_cast<int>(shapes.size()), [&](const int i) {
    out[static_cast<std::size_t>(i)] = exact_box(shapes[static_cast<std::size_t>(i)]);
  });
  return out;
}

std::optional<double> out_of_plane_spread(const TopoDS_Shape& wire) {
  // Any tolerance: the plane is wanted, not a verdict. The largest box side bounds every
  // distance from it.
  Bnd_Box box;
  BRepBndLib::Add(wire, box);
  if (box.IsVoid()) {
    return std::nullopt;
  }
  double x0, y0, z0, x1, y1, z1;
  box.Get(x0, y0, z0, x1, y1, z1);
  const double reach = std::max({x1 - x0, y1 - y0, z1 - z0, Precision::Confusion()});
  BRepLib_FindSurface finder(wire, reach, /*OnlyPlane=*/true);
  if (!finder.Found()) {
    return std::nullopt;
  }
  const Handle(Geom_Plane) plane = Handle(Geom_Plane)::DownCast(finder.Surface());
  if (plane.IsNull()) {
    return std::nullopt;
  }
  const gp_Pln pln = plane->Pln().Transformed(finder.Location().Transformation());
  const gp_Dir normal = pln.Axis().Direction();
  const gp_Pnt origin = pln.Location();
  constexpr int kSamples = 65;
  double lo = 0.0;
  double hi = 0.0;
  bool first = true;
  for (TopExp_Explorer ex(wire, TopAbs_EDGE); ex.More(); ex.Next()) {
    const TopoDS_Edge& edge = TopoDS::Edge(ex.Current());
    if (BRep_Tool::Degenerated(edge)) {
      continue;
    }
    const BRepAdaptor_Curve curve(edge);
    for (int i = 0; i < kSamples; ++i) {
      const double t = curve.FirstParameter() + (curve.LastParameter() - curve.FirstParameter()) *
                                                    i / (kSamples - 1);
      const double d = gp_Vec(origin, curve.Value(t)).Dot(gp_Vec(normal));
      lo = first ? d : std::min(lo, d);
      hi = first ? d : std::max(hi, d);
      first = false;
    }
  }
  if (first) {
    return std::nullopt;
  }
  return hi - lo;
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
