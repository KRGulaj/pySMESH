// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-06

// pySMESH binding — Session: primitives, construction geometry and sweeps.
//
// Everything that adds new geometry to the model rather than reworking what is there. The
// add_* operations consume nothing; make_wire/make_face/make_filling and the sweeps consume
// the bodies they are given, because they replace a profile with what was built from it.
//
// Wires carry no EntityId of their own — BRepTools_History supports four shape kinds and
// WIRE is not among them — so a wire is named through the ids of its edges, and each of
// these operations resolves the entities it is given to the single body that owns them.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include <BRepBuilderAPI_FindPlane.hxx>
#include <BRepOffsetAPI_MakeEvolved.hxx>
#include <GProp_PEquation.hxx>
#include <GeomLib.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_BoundedCurve.hxx>
#include <Geom_BoundedSurface.hxx>
#include <Geom_RectangularTrimmedSurface.hxx>

#include "shape_checks.hpp"

namespace pysmesh {
namespace session {
namespace {

// How far a B-spline fitted through coplanar points leaves their plane.
struct SplineDrift {
  double curve;  // the largest distance of the curve from the plane
  double poles;  // the largest distance of a pole from the plane
};

// The drift of `curve` from the plane of `points`, or nothing when the points are not
// coplanar within `tol` (GProp_PEquation: their spread across the plane), or lie on one
// line or at one point, so that no plane is theirs.
//
// The curve, not its poles, decides. BRepBuilderAPI_MakeFace finds the plane of a wire with
// BRepLib_FindSurface, which samples each B-spline edge at its knots and degree - 1 points
// between them and compares those points with the plane (BRepLib_FindSurface.cxx:388-391,
// :558-560). It never reads a pole. A pole can stand off the plane by more than the curve
// does (the curve is a convex combination of its poles), so a test of the poles would
// refuse curves that make_face accepts. The curve is sampled more densely than
// FindSurface does: 8 x degree points per knot span.
std::optional<SplineDrift> spline_drift(const NCollection_Array1<gp_Pnt>& points,
                                        const Geom_BSplineCurve& curve, double tol) {
  const GProp_PEquation fit(points, tol);
  if (!fit.IsPlanar()) {
    return std::nullopt;
  }
  const gp_Pln plane = fit.Plane();
  SplineDrift drift{0.0, 0.0};
  for (int i = 1; i <= curve.NbPoles(); ++i) {
    drift.poles = std::max(drift.poles, plane.Distance(curve.Pole(i)));
  }
  const int steps = 8 * curve.Degree();
  for (int k = 1; k < curve.NbKnots(); ++k) {
    const double a = curve.Knot(k);
    const double b = curve.Knot(k + 1);
    for (int s = 0; s <= steps; ++s) {
      drift.curve = std::max(drift.curve, plane.Distance(curve.Value(a + (b - a) * s / steps)));
    }
  }
  return drift;
}

// Throw the self-interference refusal of a swept or lofted result, if there is one. The
// message is the first line of the refusal; the details are the pairs it names and `hint`.
void refuse_self_interference(const std::string& refusal, const char* hint) {
  if (refusal.empty()) {
    return;
  }
  const std::size_t cut = refusal.find('\n');
  throw PysmeshError(refusal.substr(0, cut), refusal.substr(cut + 1) + hint, {});
}

constexpr const char* kSweepHint =
    "A Frenet frame turns with the spine's curvature, so a profile that is not perpendicular "
    "to the spine's start tangent can sweep through itself; use frenet=False, or a "
    "perpendicular profile. A profile larger than the spine's radius of curvature does the "
    "same.";
constexpr const char* kLoftHint =
    "Sections that meet each other, such as circles through one common point, pinch the "
    "boundary there: keep them apart where they would meet (a wedge of 0.1-radius circles "
    "lofts and cuts cleanly with the circles 1e-4 from the common point). Sections whose loft "
    "has to twist or bend sharply between them make the surface pass through itself: space "
    "or align them, or add sections between them.";

// BRepOffsetAPI_ThruSections' default pres3d, the tolerance its end caps are planed at.
constexpr double kLoftPres3d = 1.0e-6;

constexpr const char* kCapHint =
    "Make each end section planar, every point within its edge tolerance of one plane. A bow "
    "in a middle section needs no cap and is accepted. Or loft with solid=False and close the "
    "ends yourself.";

// Whether OCCT's end cap on `wire` fails: the face PerformPlan (BRepOffsetAPI_ThruSections.cxx)
// builds on it, a plane found at kLoftPres3d or else any face MakeFace finds, is missing or
// rejected by BRepCheck_Analyzer. Built on a copy, so that the section keeps no pcurve.
bool planar_cap_fails(const TopoDS_Wire& wire) {
  const TopoDS_Wire copy = TopoDS::Wire(BRepBuilderAPI_Copy(wire).Shape());
  TopoDS_Face face;
  BRepBuilderAPI_FindPlane finder(copy, kLoftPres3d);
  if (finder.Found()) {
    face = BRepBuilderAPI_MakeFace(finder.Plane(), copy);
  } else {
    BRepBuilderAPI_MakeFace any(copy);
    if (any.IsDone()) {
      face = any.Face();
    }
  }
  return face.IsNull() || !BRepCheck_Analyzer(face).IsValid();
}

// One end section of a solid loft whose cap fails, in words: its index, how far it is from
// planar, and what became of its cap.
std::string cap_fault_text(std::size_t index, std::size_t count, const TopoDS_Wire& wire,
                           const std::string& what) {
  std::ostringstream s;
  s << "section " << index + 1 << " of " << count << " (" << (index == 0 ? "first" : "last")
    << ") is not planar: ";
  const std::optional<double> spread = shape_checks::out_of_plane_spread(wire);
  if (spread.has_value()) {
    s << "its points spread " << *spread << " across the plane that fits them best";
  } else {
    s << "its points define no plane";
  }
  s << ", and " << what << ".";
  return s.str();
}

// The refusal of a solid loft whose end cap is missing or invalid (report O4), or empty.
//
// OCCT closes a solid loft with a planar face on the first and on the last section. If a
// section is not planar within its edge tolerance (1e-7), the face it gets is invalid; if it
// is not planar within kLoftPres3d, it gets none, and the solid keeps an open shell. Each
// cap OCCT built is checked as it is. A missing cap counts only while the shell is open (a
// closed loft needs none). OCCT stops at the first section it cannot cap, so when the first
// cap is missing, the last section is judged by building its cap the same way on a copy.
std::string loft_cap_refusal(const TopoDS_Shape& result, const TopoDS_Shape& first_cap,
                             const TopoDS_Shape& last_cap, const std::vector<TopoDS_Wire>& wires) {
  bool open = true;
  for (TopExp_Explorer ex(result, TopAbs_SHELL); ex.More(); ex.Next()) {
    open = !ex.Current().Closed();
  }
  const std::size_t count = wires.size();
  std::vector<std::string> faults;
  const auto judge = [&](std::size_t index, const TopoDS_Shape& cap, bool missing_counts) {
    const TopoDS_Wire& wire = wires[index];
    if (!cap.IsNull()) {
      const std::vector<shape_checks::CheckFinding> findings = shape_checks::check_findings(cap);
      if (!findings.empty()) {
        faults.push_back(cap_fault_text(index, count, wire,
                                        "the planar face OCCT closed the solid with on it is "
                                        "invalid (" +
                                            findings.front().status + ")"));
      }
    } else if (open && missing_counts) {
      faults.push_back(cap_fault_text(index, count, wire,
                                      "OCCT could not close the solid with a planar face on it"));
    }
  };
  judge(0, first_cap, true);
  const bool first_missing = first_cap.IsNull() && open;
  judge(count - 1, last_cap, !first_missing || planar_cap_fails(wires[count - 1]));
  if (faults.empty()) {
    return std::string();
  }
  std::string text = "Session.thru_sections: the solid loft has no valid cap: ";
  for (std::size_t i = 0; i < faults.size(); ++i) {
    text += (i == 0 ? "" : " Also ") + faults[i];
  }
  return text + " Nothing is committed; the session is unchanged.";
}

// The refusal of a solid loft whose two end caps lie in one plane and meet (report V4), or
// empty. A loft closed by a copy of its first section gets two coincident caps: a solid
// with a slit of zero thickness through it, which BRepCheck_Analyzer accepts. Caps in one
// plane that stay apart, as at the two ends of a U, are fine.
std::string coincident_caps_refusal(const TopoDS_Shape& first_cap, const TopoDS_Shape& last_cap) {
  if (first_cap.IsNull() || last_cap.IsNull()) {
    return std::string();
  }
  const BRepAdaptor_Surface a(TopoDS::Face(first_cap), false);
  const BRepAdaptor_Surface b(TopoDS::Face(last_cap), false);
  if (a.GetType() != GeomAbs_Plane || b.GetType() != GeomAbs_Plane) {
    return std::string();
  }
  const double tol = shape_checks::max_tolerance({first_cap, last_cap});
  const gp_Pln pa = a.Plane();
  const gp_Pln pb = b.Plane();
  if (!pa.Axis().IsParallel(pb.Axis(), Precision::Angular()) ||
      pa.Distance(pb.Location()) > tol) {
    return std::string();
  }
  BRepExtrema_DistShapeShape gap(first_cap, last_cap);
  if (!gap.IsDone() || gap.Value() > tol) {
    return std::string();
  }
  std::ostringstream s;
  s << "Session.thru_sections: the caps of the first and the last section lie in one plane "
       "and meet (gap "
    << gap.Value() << ", tolerance " << tol
    << "), so the solid has a slit of zero thickness between them. Nothing is committed; the "
       "session is unchanged.";
  return s.str();
}

constexpr const char* kSlitHint =
    "To close a loft round onto itself, name the first section again as the last: that is a "
    "closed loft, with no caps. A copy of the first section makes two caps.";

// One lateral face of a loft, with its loft parameter: 0 at the first section, k at
// section k + 1, linear in the face's v, which runs across the sections. A ruled face spans
// one section interval, a smooth face all of them.
struct LoftFace {
  TopoDS_Face face;
  BRepAdaptor_Surface surface;
  double u0 = 0.0, u1 = 0.0, v0 = 0.0, v1 = 0.0;
  double g0 = 0.0, g1 = 0.0;  // the loft parameter at v0 and at v1

  double loft_parameter(double v) const { return g0 + (g1 - g0) * (v - v0) / (v1 - v0); }
};

// A point that the loft covers twice, at loft parameters half a section interval or more
// apart.
struct LoftFold {
  int face_a = 0;  // 1-based ordinals in the loft, TopExp::MapShapes order
  int face_b = 0;
  double g_a = 0.0;
  double g_b = 0.0;
  gp_Pnt point;
};

// How far apart two loft parameters are, in section intervals. A closed loft (the first
// section named again as the last) is cyclic over its `spans` intervals.
double loft_gap(double a, double b, int spans, bool closed) {
  const double d = std::abs(a - b);
  return closed ? std::min(d, spans - d) : d;
}

// Whether every point of wire `b` lies on wire `a` within `tol`: 9 points per edge of `b`,
// after a test of the two bounding boxes.
bool wire_lies_on(const TopoDS_Wire& b, const TopoDS_Wire& a, double tol) {
  Bnd_Box ba, bb;
  BRepBndLib::Add(a, ba);
  BRepBndLib::Add(b, bb);
  if (ba.IsVoid() || bb.IsVoid() || bb.CornerMin().Distance(ba.CornerMin()) > 2.0 * tol ||
      bb.CornerMax().Distance(ba.CornerMax()) > 2.0 * tol) {
    return false;
  }
  for (TopExp_Explorer ex(b, TopAbs_EDGE); ex.More(); ex.Next()) {
    const TopoDS_Edge& edge = TopoDS::Edge(ex.Current());
    if (BRep_Tool::Degenerated(edge)) {
      continue;
    }
    const BRepAdaptor_Curve curve(edge);
    for (int k = 0; k <= 8; ++k) {
      const double t =
          curve.FirstParameter() + (curve.LastParameter() - curve.FirstParameter()) * k / 8;
      BRepExtrema_DistShapeShape d(BRepBuilderAPI_MakeVertex(curve.Value(t)).Vertex(), a);
      if (!d.IsDone() || d.Value() > tol) {
        return false;
      }
    }
  }
  return true;
}

// Gauss-Newton on S_a(ua, va) - S_b(ub, vb) = 0 from x = (ua, va, ub, vb): the minimum-norm
// step over the free parameters, each kept inside its face's parameter box. A parameter at
// its bound whose step leaves the box is held there for that step. Returns the distance
// |S_a - S_b| at the final x.
double refine_loft_pair(const LoftFace& a, const LoftFace& b, std::array<double, 4>& x,
                        double tol) {
  const std::array<double, 4> lo = {a.u0, a.v0, b.u0, b.v0};
  const std::array<double, 4> hi = {a.u1, a.v1, b.u1, b.v1};
  double dist = std::numeric_limits<double>::infinity();
  for (int it = 0; it < 30; ++it) {
    gp_Pnt pa, pb;
    gp_Vec dua, dva, dub, dvb;
    a.surface.D1(x[0], x[1], pa, dua, dva);
    b.surface.D1(x[2], x[3], pb, dub, dvb);
    const gp_Vec f(pb, pa);
    dist = f.Magnitude();
    if (dist <= tol) {
      return dist;
    }
    const std::array<gp_Vec, 4> col = {dua, dva, -dub, -dvb};
    std::array<bool, 4> free = {true, true, true, true};
    std::array<double, 4> step = {0.0, 0.0, 0.0, 0.0};
    for (int pass = 0; pass < 4; ++pass) {
      // (J J^T) y = -f over the free columns; the step is J^T y.
      double m[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
      for (int k = 0; k < 4; ++k) {
        if (!free[k]) {
          continue;
        }
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            m[i][j] += col[k].Coord(i + 1) * col[k].Coord(j + 1);
          }
        }
      }
      const double mu = 1.0e-12 * (m[0][0] + m[1][1] + m[2][2]) + 1.0e-300;
      for (int i = 0; i < 3; ++i) {
        m[i][i] += mu;
      }
      const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                         m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
      if (!(std::abs(det) > 0.0)) {
        return dist;
      }
      const double r[3] = {-f.X(), -f.Y(), -f.Z()};
      double y[3];
      for (int c = 0; c < 3; ++c) {  // Cramer's rule, column c replaced by r
        double n[3][3];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            n[i][j] = (j == c) ? r[i] : m[i][j];
          }
        }
        y[c] = (n[0][0] * (n[1][1] * n[2][2] - n[1][2] * n[2][1]) -
                n[0][1] * (n[1][0] * n[2][2] - n[1][2] * n[2][0]) +
                n[0][2] * (n[1][0] * n[2][1] - n[1][1] * n[2][0])) /
               det;
      }
      bool held = false;
      for (int k = 0; k < 4; ++k) {
        step[k] = free[k] ? col[k].X() * y[0] + col[k].Y() * y[1] + col[k].Z() * y[2] : 0.0;
        if (free[k] && ((x[k] <= lo[k] && step[k] < 0.0) || (x[k] >= hi[k] && step[k] > 0.0))) {
          free[k] = false;
          held = true;
        }
      }
      if (!held) {
        break;
      }
    }
    for (int k = 0; k < 4; ++k) {
      x[k] = std::clamp(x[k] + step[k], lo[k], hi[k]);
    }
  }
  return dist;
}

// A point of the loft `result` that two of its lateral faces, or one face twice, cover at
// loft parameters half a section interval or more apart, or nothing. Such a point is a fold:
// the surface comes back through a place it passed before. The two sides of a seam, of a
// closed section or of a closed loft, have the same loft parameter and do not count.
//
// A grid of samples on each lateral face finds candidate pairs: two samples closer than the
// sum of their grid spacings, with loft parameters half an interval less one sample step
// or more apart. The candidates are refined by refine_loft_pair, closest first, at most 2
// for each pair of faces and half-interval cells of their loft parameters, 64 in all. A
// fold is reported only when the refined points are within kLoftPres3d of each other and
// their loft parameters are still half an interval or more apart.
//
// `wires` are the sections in loft order, the first repeated as the last for a closed loft.
// The loft parameter is cyclic for a closed loft, and also when the last section lies where
// the first does (a copy of it): the loft then closes too, through two coincident edges.
// A ruled loft has one face per section edge per interval, interval by interval in
// TopExp_Explorer order (BRepOffsetAPI_ThruSections::Generated relies on the same order);
// the end of a ruled face's v-range that lies on the earlier section is found by distance.
std::optional<LoftFold> loft_fold(const TopoDS_Shape& result, const TopoDS_Shape& first,
                                  const TopoDS_Shape& last, const std::vector<TopoDS_Wire>& wires,
                                  bool ruled, bool closed) {
  if (!closed && wires.size() > 2) {
    const double tol = std::max(
        shape_checks::max_tolerance({wires.front(), wires.back()}), kLoftPres3d);
    closed = wire_lies_on(wires.back(), wires.front(), tol) &&
             wire_lies_on(wires.front(), wires.back(), tol);
  }
  ShapeSet caps;
  TopExp::MapShapes(first, TopAbs_FACE, caps);
  TopExp::MapShapes(last, TopAbs_FACE, caps);
  std::vector<TopoDS_Face> lateral;
  for (TopExp_Explorer ex(result, TopAbs_FACE); ex.More(); ex.Next()) {
    if (!caps.Contains(ex.Current())) {
      lateral.push_back(TopoDS::Face(ex.Current()));
    }
  }
  const int spans = static_cast<int>(wires.size()) - 1;
  if (lateral.empty() || spans < 1) {
    return std::nullopt;
  }
  if (ruled && lateral.size() % static_cast<std::size_t>(spans) != 0) {
    throw std::runtime_error("the ruled loft has " + std::to_string(lateral.size()) +
                             " lateral faces, not a multiple of its " +
                             std::to_string(spans) + " section intervals");
  }
  const std::size_t per_span = lateral.size() / static_cast<std::size_t>(spans);

  std::vector<LoftFace> faces;
  faces.reserve(lateral.size());
  for (std::size_t i = 0; i < lateral.size(); ++i) {
    LoftFace lf{lateral[i], BRepAdaptor_Surface(lateral[i])};
    BRepTools::UVBounds(lf.face, lf.u0, lf.u1, lf.v0, lf.v1);
    if (!(lf.v1 > lf.v0) || !(lf.u1 > lf.u0)) {
      continue;
    }
    if (ruled) {
      const int s = static_cast<int>(i / per_span);
      const TopoDS_Vertex at_v0 = BRepBuilderAPI_MakeVertex(
          lf.surface.Value(0.5 * (lf.u0 + lf.u1), lf.v0));
      BRepExtrema_DistShapeShape to_earlier(at_v0, wires[s]);
      BRepExtrema_DistShapeShape to_later(at_v0, wires[s + 1]);
      const bool forward = !to_earlier.IsDone() || !to_later.IsDone() ||
                           to_earlier.Value() <= to_later.Value();
      lf.g0 = forward ? s : s + 1;
      lf.g1 = forward ? s + 1 : s;
    } else {
      lf.g0 = 0.0;
      lf.g1 = spans;
    }
    faces.push_back(lf);
  }

  // The samples, each with its grid spacing: the distance to its farthest grid neighbour.
  constexpr int kNu = 33;
  const int nv = ruled ? 9 : 8 * spans + 1;
  const double g_step = 1.0 / 8.0;  // the loft parameter between two samples along v
  struct Sample {
    gp_Pnt point;
    double u, v, g, reach;
    int face;
  };
  std::vector<Sample> samples;
  samples.reserve(faces.size() * kNu * nv);
  double h = 0.0;
  for (std::size_t fi = 0; fi < faces.size(); ++fi) {
    const LoftFace& lf = faces[fi];
    const std::size_t base = samples.size();
    for (int i = 0; i < kNu; ++i) {
      const double u = lf.u0 + (lf.u1 - lf.u0) * i / (kNu - 1);
      for (int j = 0; j < nv; ++j) {
        const double v = lf.v0 + (lf.v1 - lf.v0) * j / (nv - 1);
        samples.push_back(
            {lf.surface.Value(u, v), u, v, lf.loft_parameter(v), 0.0, static_cast<int>(fi)});
      }
    }
    for (int i = 0; i < kNu; ++i) {
      for (int j = 0; j < nv; ++j) {
        Sample& s = samples[base + i * nv + j];
        if (i + 1 < kNu) {
          const double d = s.point.Distance(samples[base + (i + 1) * nv + j].point);
          s.reach = std::max(s.reach, d);
          samples[base + (i + 1) * nv + j].reach =
              std::max(samples[base + (i + 1) * nv + j].reach, d);
        }
        if (j + 1 < nv) {
          const double d = s.point.Distance(samples[base + i * nv + j + 1].point);
          s.reach = std::max(s.reach, d);
          samples[base + i * nv + j + 1].reach =
              std::max(samples[base + i * nv + j + 1].reach, d);
        }
        h = std::max(h, s.reach);
      }
    }
  }
  if (!(h > 0.0)) {
    return std::nullopt;
  }

  // Candidate pairs through a hash of cells of side h, the largest grid spacing.
  const auto cell = [h](double c) { return static_cast<std::int64_t>(std::floor(c / h)); };
  const auto key = [](std::int64_t i, std::int64_t j, std::int64_t k) {
    return ((i & 0x1FFFFF) << 42) | ((j & 0x1FFFFF) << 21) | (k & 0x1FFFFF);
  };
  std::unordered_map<std::int64_t, std::vector<int>> cells;
  for (int n = 0; n < static_cast<int>(samples.size()); ++n) {
    const gp_Pnt& p = samples[n].point;
    cells[key(cell(p.X()), cell(p.Y()), cell(p.Z()))].push_back(n);
  }
  struct Candidate {
    double distance;
    int a, b;
  };
  std::vector<Candidate> candidates;
  for (int n = 0; n < static_cast<int>(samples.size()); ++n) {
    const Sample& s = samples[n];
    const std::int64_t ci = cell(s.point.X()), cj = cell(s.point.Y()), ck = cell(s.point.Z());
    for (std::int64_t di = -1; di <= 1; ++di) {
      for (std::int64_t dj = -1; dj <= 1; ++dj) {
        for (std::int64_t dk = -1; dk <= 1; ++dk) {
          const auto found = cells.find(key(ci + di, cj + dj, ck + dk));
          if (found == cells.end()) {
            continue;
          }
          for (const int m : found->second) {
            if (m <= n) {
              continue;
            }
            const Sample& t = samples[m];
            const double d = s.point.Distance(t.point);
            if (d <= s.reach + t.reach && loft_gap(s.g, t.g, spans, closed) >= 0.5 - g_step) {
              candidates.push_back({d, n, m});
            }
          }
        }
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& p, const Candidate& q) { return p.distance < q.distance; });

  ShapeSet ordinals;
  TopExp::MapShapes(result, TopAbs_FACE, ordinals);
  std::map<std::array<int, 4>, int> tried;
  int refinements = 0;
  for (const Candidate& c : candidates) {
    if (refinements >= 64) {
      break;
    }
    const Sample& s = samples[c.a];
    const Sample& t = samples[c.b];
    const Sample& lower = s.face <= t.face ? s : t;
    const Sample& upper = s.face <= t.face ? t : s;
    int& per_cell = tried[{lower.face, upper.face, static_cast<int>(std::floor(2.0 * lower.g)),
                           static_cast<int>(std::floor(2.0 * upper.g))}];
    if (per_cell >= 2) {
      continue;
    }
    ++per_cell;
    ++refinements;
    const LoftFace& fa = faces[s.face];
    const LoftFace& fb = faces[t.face];
    std::array<double, 4> x = {s.u, s.v, t.u, t.v};
    if (refine_loft_pair(fa, fb, x, kLoftPres3d) > kLoftPres3d) {
      continue;
    }
    const double ga = fa.loft_parameter(x[1]);
    const double gb = fb.loft_parameter(x[3]);
    if (loft_gap(ga, gb, spans, closed) < 0.5) {
      continue;
    }
    LoftFold fold;
    fold.face_a = ordinals.FindIndex(fa.face);
    fold.face_b = ordinals.FindIndex(fb.face);
    fold.g_a = ga;
    fold.g_b = gb;
    const gp_Pnt pa = fa.surface.Value(x[0], x[1]);
    const gp_Pnt pb = fb.surface.Value(x[2], x[3]);
    fold.point = gp_Pnt(0.5 * (pa.XYZ() + pb.XYZ()));
    return fold;
  }
  return std::nullopt;
}

// The section interval of loft parameter `g`, in words: "between sections k and k + 1",
// 1-based, as the caller named them.
std::string loft_interval_text(double g, int spans) {
  const int k = std::clamp(static_cast<int>(std::floor(g)), 0, spans - 1);
  return "between sections " + std::to_string(k + 1) + " and " + std::to_string(k + 2);
}

constexpr const char* kFoldHint =
    "The loft covers this point twice, from places half a section interval or more apart "
    "along it. Stations that go back along the loft do this, such as sections at z = 0, 1, "
    "0.6, 1.6. Keep the sections in order along the loft, or space them so that the surface "
    "between them does not turn back.";

}  // namespace

// ---- construction operations ------------------------------------------------------ //

// An imported solid is checked for its inside (report V3): BRepCheck_Analyzer accepts a
// solid whose shell bounds its complement, so the import used to commit a unit box at
// volume -1 with every point test inverted. `inside_out` decides what happens to one:
// "raise" refuses the import, "reverse" reverses it and says so on the delta.
py::dict Session::add_brep(const py::bytes& data, const std::string& inside_out,
                           const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  const bool reverse = shape_checks::reverse_inside_out("Session.add_brep", inside_out);
  const std::string buffer = data;
  require_brep_header(
      buffer, "Session.add_brep: BREP read produced a null shape (empty or malformed data).");
  ProgressDriver driver("add_brep", hooks_of("add_brep", progress, cancel));
  TopoDS_Shape imported;
  std::vector<shape_checks::InsideOutSolid> wrong;
  {
    py::gil_scoped_release release;
    std::istringstream stream(buffer);
    BRep_Builder builder;
    try {
      BRepTools::Read(imported, stream, builder, driver.range());
      if (!imported.IsNull()) {
        wrong = shape_checks::inside_out_solids(imported, "Session.add_brep");
      }
    } catch (const PysmeshError&) {
      // A solid the measure rule cannot measure is refused by name, not as a failed read.
      py::gil_scoped_acquire acquire;
      throw;
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.add_brep: BREP read failed: ") + e.what());
    }
  }
  driver.finish();
  // A cancelled read leaves a null shape, which the check below would report as malformed
  // data. Naming the real cause first keeps the two apart.
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("add_brep");
  }
  if (imported.IsNull()) {
    throw PysmeshError(
        "Session.add_brep: BREP read produced a null shape (empty or malformed data).");
  }
  if (wrong.empty()) {
    return add_bodies(imported, "add_brep");
  }
  if (!reverse) {
    throw PysmeshError(shape_checks::inside_out_refusal("Session.add_brep", wrong));
  }
  std::vector<std::string> warnings;
  for (const shape_checks::InsideOutSolid& s : wrong) {
    warnings.push_back(shape_checks::inside_out_reversed(s));
  }
  const TopoDS_Shape fixed = shape_checks::reverse_solids(imported, wrong);
  return commit(concat(root_bodies(state_.root), fixed), Handle(BRepTools_History)(),
                "add_brep", fixed, Validation::Strict, warnings);
}

py::dict Session::add_box(double dx, double dy, double dz, double ox, double oy, double oz) {
  finite_arg("add_box", "dx", dx);
  finite_arg("add_box", "dy", dy);
  finite_arg("add_box", "dz", dz);
  finite_arg("add_box", "origin", ox, oy, oz);
  OpGuard guard(in_op_);
  require_positive("dx", dx);
  require_positive("dy", dy);
  require_positive("dz", dz);
  const TopoDS_Shape solid = build_shape("add_box", [&] {
    return BRepPrimAPI_MakeBox(gp_Pnt(ox, oy, oz), dx, dy, dz).Shape();
  });
  return add_bodies(solid, "add_box");
}

py::dict Session::add_cylinder(double radius, double height, double ox, double oy, double oz,
                        double ax, double ay, double az) {
  finite_arg("add_cylinder", "radius", radius);
  finite_arg("add_cylinder", "height", height);
  finite_arg("add_cylinder", "origin", ox, oy, oz);
  finite_arg("add_cylinder", "axis", ax, ay, az);
  OpGuard guard(in_op_);
  require_positive("radius", radius);
  require_positive("height", height);
  const gp_Vec axis(ax, ay, az);
  if (axis.Magnitude() <= 0.0) {
    throw PysmeshError("Session.add_cylinder: axis must be a non-zero vector.");
  }
  const gp_Ax2 frame(gp_Pnt(ox, oy, oz), gp_Dir(axis));
  const TopoDS_Shape solid = build_shape("add_cylinder", [&] {
    return BRepPrimAPI_MakeCylinder(frame, radius, height).Shape();
  });
  return add_bodies(solid, "add_cylinder");
}

py::dict Session::add_cone(double radius1, double radius2, double height, double ox, double oy,
                    double oz, double ax, double ay, double az, double angle_rad) {
  finite_arg("add_cone", "radius1", radius1);
  finite_arg("add_cone", "radius2", radius2);
  finite_arg("add_cone", "height", height);
  finite_arg("add_cone", "origin", ox, oy, oz);
  finite_arg("add_cone", "axis", ax, ay, az);
  finite_arg("add_cone", "angle_rad", angle_rad);
  OpGuard guard(in_op_);
  require_non_negative("radius1", radius1);
  require_non_negative("radius2", radius2);
  if (radius1 <= 0.0 && radius2 <= 0.0) {
    throw PysmeshError("Session.add_cone: at least one radius must be > 0.");
  }
  if (radius1 == radius2) {
    throw PysmeshError(
        "Session.add_cone: OCCT's cone needs two different radii (both are " +
        std::to_string(radius1) + "). Use add_cylinder for a straight tube.");
  }
  require_positive("height", height);
  require_sweep_angle("add_cone", angle_rad);
  const gp_Ax2 frame = frame_of("add_cone", ox, oy, oz, ax, ay, az);
  const TopoDS_Shape solid = build_shape("add_cone", [&] {
    return BRepPrimAPI_MakeCone(frame, radius1, radius2, height, angle_rad).Shape();
  });
  return add_bodies(solid, "add_cone");
}

py::dict Session::add_sphere(double radius, double cx, double cy, double cz, double ax,
                             double ay, double az, double angle_rad) {
  finite_arg("add_sphere", "radius", radius);
  finite_arg("add_sphere", "centre", cx, cy, cz);
  finite_arg("add_sphere", "axis", ax, ay, az);
  finite_arg("add_sphere", "angle_rad", angle_rad);
  OpGuard guard(in_op_);
  require_positive("radius", radius);
  require_sweep_angle("add_sphere", angle_rad);
  const gp_Ax2 frame = frame_of("add_sphere", cx, cy, cz, ax, ay, az);
  const TopoDS_Shape solid = build_shape("add_sphere", [&] {
    return BRepPrimAPI_MakeSphere(frame, radius, angle_rad).Shape();
  });
  return add_bodies(solid, "add_sphere");
}

py::dict Session::add_torus(double radius1, double radius2, double ox, double oy, double oz,
                     double ax, double ay, double az, double angle_rad) {
  finite_arg("add_torus", "radius1", radius1);
  finite_arg("add_torus", "radius2", radius2);
  finite_arg("add_torus", "origin", ox, oy, oz);
  finite_arg("add_torus", "axis", ax, ay, az);
  finite_arg("add_torus", "angle_rad", angle_rad);
  OpGuard guard(in_op_);
  require_positive("radius1", radius1);
  require_positive("radius2", radius2);
  if (radius2 >= radius1) {
    throw PysmeshError("Session.add_torus: radius2 (the tube radius, " +
                       std::to_string(radius2) +
                       ") must be smaller than radius1 (the ring radius, " +
                       std::to_string(radius1) + ") for a self-intersection-free torus.");
  }
  require_sweep_angle("add_torus", angle_rad);
  const gp_Ax2 frame = frame_of("add_torus", ox, oy, oz, ax, ay, az);
  const TopoDS_Shape solid = build_shape("add_torus", [&] {
    return BRepPrimAPI_MakeTorus(frame, radius1, radius2, angle_rad).Shape();
  });
  return add_bodies(solid, "add_torus");
}

// A STEP right angular wedge: a box of dx * dy * dz whose face at y = dy is narrowed to
// ltx along x. ltx == dx is a plain box; ltx == 0 is a wedge with a knife edge.
py::dict Session::add_wedge(double dx, double dy, double dz, double ltx, double ox, double oy,
                     double oz, double ax, double ay, double az) {
  finite_arg("add_wedge", "dx", dx);
  finite_arg("add_wedge", "dy", dy);
  finite_arg("add_wedge", "dz", dz);
  finite_arg("add_wedge", "ltx", ltx);
  finite_arg("add_wedge", "origin", ox, oy, oz);
  finite_arg("add_wedge", "axis", ax, ay, az);
  OpGuard guard(in_op_);
  require_positive("dx", dx);
  require_positive("dy", dy);
  require_positive("dz", dz);
  require_non_negative("ltx", ltx);
  const gp_Ax2 frame = frame_of("add_wedge", ox, oy, oz, ax, ay, az);
  const TopoDS_Shape solid = build_shape("add_wedge", [&] {
    return BRepPrimAPI_MakeWedge(frame, dx, dy, dz, ltx).Shape();
  });
  return add_bodies(solid, "add_wedge");
}

// ---- construction geometry -------------------------------------------------------- //
//
// These add curve and surface bodies to the root. The registry tracks SOLID/FACE/EDGE/
// VERTEX only (that set is fixed by BRepTools_History), so a WIRE body carries no id of
// its own: it is named through the ids of its edges, and every operation that consumes a
// profile resolves the named entities to the single body that owns them.

// A standalone vertex body. The only construction that adds a point to the model as an
// entity in its own right rather than as the boundary of something else.
py::dict Session::add_vertex(double x, double y, double z) {
  finite_arg("add_vertex", "point", x, y, z);
  OpGuard guard(in_op_);
  const TopoDS_Shape vertex = build_shape(
      "add_vertex", [&] { return BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex(); });
  return add_bodies(vertex, "add_vertex");
}

py::dict Session::add_line(double x1, double y1, double z1, double x2, double y2, double z2) {
  finite_arg("add_line", "start", x1, y1, z1);
  finite_arg("add_line", "end", x2, y2, z2);
  OpGuard guard(in_op_);
  const gp_Pnt a(x1, y1, z1);
  const gp_Pnt b(x2, y2, z2);
  if (a.Distance(b) <= 0.0) {
    throw PysmeshError("Session.add_line: the two points are coincident.");
  }
  const TopoDS_Shape edge =
      build_shape("add_line", [&] { return BRepBuilderAPI_MakeEdge(a, b).Edge(); });
  return add_bodies(edge, "add_line");
}

// Three-point arc: through p1, ending at p3, passing through p2.
py::dict Session::add_arc(double x1, double y1, double z1, double x2, double y2, double z2,
                   double x3, double y3, double z3) {
  finite_arg("add_arc", "start", x1, y1, z1);
  finite_arg("add_arc", "through", x2, y2, z2);
  finite_arg("add_arc", "end", x3, y3, z3);
  OpGuard guard(in_op_);
  const TopoDS_Shape edge = try_build("add_arc", [&]() -> TopoDS_Shape {
    GC_MakeArcOfCircle mk(gp_Pnt(x1, y1, z1), gp_Pnt(x2, y2, z2), gp_Pnt(x3, y3, z3));
    if (!mk.IsDone()) {
      return TopoDS_Shape();
    }
    return BRepBuilderAPI_MakeEdge(mk.Value()).Edge();
  });
  if (edge.IsNull()) {
    throw PysmeshError(
        "Session.add_arc: no circular arc passes through the three points (they are "
        "collinear or two of them coincide).");
  }
  return add_bodies(edge, "add_arc");
}

py::dict Session::add_circle(double cx, double cy, double cz, double nx, double ny, double nz,
                      double radius) {
  finite_arg("add_circle", "centre", cx, cy, cz);
  finite_arg("add_circle", "normal", nx, ny, nz);
  finite_arg("add_circle", "radius", radius);
  OpGuard guard(in_op_);
  require_positive("radius", radius);
  const gp_Ax2 frame = frame_of("add_circle", cx, cy, cz, nx, ny, nz);
  const TopoDS_Shape edge = build_shape("add_circle", [&] {
    return BRepBuilderAPI_MakeEdge(new Geom_Circle(gp_Circ(frame, radius))).Edge();
  });
  return add_bodies(edge, "add_circle");
}

// A full elliptical edge, rx along the plane's first in-plane direction and ry along its
// second.
//
// The in-plane orientation is part of an ellipse's geometry, which is what separates this
// from add_circle: a normal alone fixes the plane but not where the major axis points in it,
// so x_dir names the first in-plane direction when the caller cares, and OCCT derives it
// from the normal when the caller does not.
py::dict Session::add_ellipse(double cx, double cy, double cz, double nx, double ny,
                              double nz, double rx, double ry,
                              const std::optional<std::array<double, 3>>& x_dir) {
  finite_arg("add_ellipse", "centre", cx, cy, cz);
  finite_arg("add_ellipse", "normal", nx, ny, nz);
  finite_arg("add_ellipse", "rx", rx);
  finite_arg("add_ellipse", "ry", ry);
  if (x_dir.has_value()) {
    finite_arg("add_ellipse", "x_dir", (*x_dir)[0], (*x_dir)[1], (*x_dir)[2]);
  }
  OpGuard guard(in_op_);
  require_positive("rx", rx);
  require_positive("ry", ry);
  const gp_Pnt centre(cx, cy, cz);
  const gp_Dir normal = direction_of("add_ellipse", "normal", nx, ny, nz);
  gp_Ax2 frame(centre, normal);
  if (x_dir.has_value()) {
    const std::array<double, 3>& d = *x_dir;
    const gp_Dir x = direction_of("add_ellipse", "x_dir", d[0], d[1], d[2]);
    if (x.IsParallel(normal, Precision::Angular())) {
      throw PysmeshError(
          "Session.add_ellipse: x_dir is parallel to normal, so it names no direction in "
          "the ellipse's plane.");
    }
    frame = gp_Ax2(centre, normal, x);
  }
  // gp_Elips demands major >= minor, but ry > rx is not a caller error: it is the same
  // ellipse with its major axis along the plane's second direction. Naming that direction
  // as the major one expresses it exactly, so no case is refused and none is approximated.
  const gp_Ax2 axes(centre, normal, ry > rx ? frame.YDirection() : frame.XDirection());
  const double major = std::max(rx, ry);
  const double minor = std::min(rx, ry);
  const TopoDS_Shape edge = build_shape("add_ellipse", [&] {
    return BRepBuilderAPI_MakeEdge(new Geom_Ellipse(gp_Elips(axes, major, minor))).Edge();
  });
  return add_bodies(edge, "add_ellipse");
}

// A polyline through the given points. Unlike make_wire this shares one vertex between
// consecutive segments by construction, so no edge is ever rebuilt to connect it.
py::dict Session::add_polyline(const PointArray& points, bool closed) {
  OpGuard guard(in_op_);
  const std::vector<gp_Pnt> pts = points_of("add_polyline", "points", points, 2);
  const TopoDS_Shape wire = try_build("add_polyline", [&]() -> TopoDS_Shape {
    BRepBuilderAPI_MakePolygon poly;
    for (const gp_Pnt& p : pts) {
      poly.Add(p);
    }
    if (closed) {
      poly.Close();
    }
    if (!poly.IsDone()) {
      return TopoDS_Shape();
    }
    return poly.Wire();
  });
  if (wire.IsNull()) {
    throw PysmeshError(
        "Session.add_polyline: OCCT could not build the polygon (consecutive points are "
        "most likely coincident).");
  }
  return add_bodies(wire, "add_polyline");
}

// A B-spline approximating the given points to within tol. This is the "spline through
// points" construction; add_bspline takes control points instead.
py::dict Session::add_spline(const PointArray& points, int degree_min, int degree_max,
                             double tol) {
  finite_arg("add_spline", "tol", tol);
  OpGuard guard(in_op_);
  const std::vector<gp_Pnt> pts = points_of("add_spline", "points", points, 2);
  require_positive("tol", tol);
  if (degree_min < 1 || degree_max < degree_min) {
    throw PysmeshError("Session.add_spline: need 1 <= degree_min <= degree_max (got " +
                       std::to_string(degree_min) + ", " + std::to_string(degree_max) +
                       ").");
  }
  std::optional<SplineDrift> drift;
  const TopoDS_Shape edge = try_build("add_spline", [&]() -> TopoDS_Shape {
    NCollection_Array1<gp_Pnt> arr(1, static_cast<int>(pts.size()));
    for (std::size_t i = 0; i < pts.size(); ++i) {
      arr.SetValue(static_cast<int>(i) + 1, pts[i]);
    }
    GeomAPI_PointsToBSpline fit(arr, degree_min, degree_max, GeomAbs_C2, tol);
    if (!fit.IsDone()) {
      return TopoDS_Shape();
    }
    drift = spline_drift(arr, *fit.Curve(), tol);
    return BRepBuilderAPI_MakeEdge(fit.Curve()).Edge();
  });
  if (edge.IsNull()) {
    throw PysmeshError(
        "Session.add_spline: GeomAPI_PointsToBSpline could not approximate the points "
        "to the requested tolerance.");
  }
  // When the fit needs more poles than there are points, OCCT gives up the approximation
  // and interpolates (Approx_BSplComputeLine.gxx:952-956). That interpolation took its end
  // tangents from separate fits and put poles of 121 coplanar points 2.5e-3 off their
  // plane at tol 1e-7, so make_face on the edge failed. A curve that leaves the plane of
  // its coplanar points by more than tol is refused rather than kept.
  if (drift.has_value() && drift->curve > tol) {
    std::ostringstream s;
    s << "Session.add_spline: the " << pts.size() << " points lie in one plane within tol "
      << tol << ", but the fitted curve leaves that plane by " << drift->curve
      << " (its poles by up to " << drift->poles
      << "), more than tol, so a face or a loft cap on it would not be planar. Nothing "
         "is committed; the session is unchanged.";
    throw PysmeshError(
        s.str(),
        "GeomAPI_PointsToBSpline interpolates the points when it needs more poles than "
        "there are points, and that interpolation can leave their plane. Use add_bspline "
        "with poles of your own in the plane, add_polyline, a different point count, or a "
        "looser tol.",
        {});
  }
  return add_bodies(edge, "add_spline");
}

// A clamped, uniformly-knotted B-spline over the given control points. The degree is
// clamped to len(poles) - 1, because a higher degree has no valid knot vector.
py::dict Session::add_bspline(const PointArray& poles, int degree) {
  OpGuard guard(in_op_);
  const std::vector<gp_Pnt> pts = points_of("add_bspline", "poles", poles, 2);
  if (degree < 1) {
    throw PysmeshError("Session.add_bspline: degree must be >= 1 (got " +
                       std::to_string(degree) + ").");
  }
  const int n = static_cast<int>(pts.size());
  const int p = std::min(degree, n - 1);
  const TopoDS_Shape edge = try_build("add_bspline", [&]() -> TopoDS_Shape {
    NCollection_Array1<gp_Pnt> arr(1, n);
    for (int i = 0; i < n; ++i) {
      arr.SetValue(i + 1, pts[static_cast<std::size_t>(i)]);
    }
    const int nk = n - p + 1;
    NCollection_Array1<double> knots(1, nk);
    NCollection_Array1<int> mults(1, nk);
    for (int i = 1; i <= nk; ++i) {
      knots.SetValue(i, static_cast<double>(i - 1));
      mults.SetValue(i, 1);
    }
    mults.SetValue(1, p + 1);
    mults.SetValue(nk, p + 1);
    return BRepBuilderAPI_MakeEdge(new Geom_BSplineCurve(arr, knots, mults, p)).Edge();
  });
  return add_bodies(edge, "add_bspline");
}

// A helical wire. TKHelix's HelixBRep_BuilderHelix is the only helix facade OCCT 8.0 has;
// there is no BRepPrimAPI-style one. It approximates, so the result is a B-spline wire
// whose deviation from the exact helix is bounded by tol.
py::dict Session::add_helix(double cx, double cy, double cz, double ax, double ay, double az,
                     double diameter, double pitch, double turns, double tol) {
  finite_arg("add_helix", "centre", cx, cy, cz);
  finite_arg("add_helix", "axis", ax, ay, az);
  finite_arg("add_helix", "diameter", diameter);
  finite_arg("add_helix", "pitch", pitch);
  finite_arg("add_helix", "turns", turns);
  finite_arg("add_helix", "tol", tol);
  OpGuard guard(in_op_);
  require_positive("diameter", diameter);
  require_positive("pitch", pitch);
  require_positive("turns", turns);
  require_positive("tol", tol);
  const gp_Dir dir = direction_of("add_helix", "axis", ax, ay, az);
  int status = 0;
  const TopoDS_Shape wire = try_build("add_helix", [&]() -> TopoDS_Shape {
    HelixBRep_BuilderHelix mk;
    NCollection_Array1<double> pitches(1, 1);
    pitches.SetValue(1, pitch);
    NCollection_Array1<double> nb_turns(1, 1);
    nb_turns.SetValue(1, turns);
    mk.SetApproxParameters(tol, 8, GeomAbs_C2);
    mk.SetParameters(gp_Ax3(gp_Pnt(cx, cy, cz), dir), diameter, pitches, nb_turns);
    mk.Perform();
    status = mk.ErrorStatus();
    return status == 0 ? mk.Shape() : TopoDS_Shape();
  });
  if (status != 0 || wire.IsNull()) {
    throw PysmeshError(
        "Session.add_helix: HelixBRep_BuilderHelix failed.",
        "ErrorStatus " + std::to_string(status) +
            " (2 = approximation failed, 10 = radius below tolerance, 11 = pitch below "
            "tolerance, 12 = height below tolerance).",
        {});
  }
  return add_bodies(wire, "add_helix");
}

// A planar rectangular face, dx by dy in the frame's own x/y directions.
py::dict Session::add_rectangle(double ox, double oy, double oz, double nx, double ny,
                                double nz, double dx, double dy) {
  finite_arg("add_rectangle", "origin", ox, oy, oz);
  finite_arg("add_rectangle", "normal", nx, ny, nz);
  finite_arg("add_rectangle", "dx", dx);
  finite_arg("add_rectangle", "dy", dy);
  OpGuard guard(in_op_);
  require_positive("dx", dx);
  require_positive("dy", dy);
  const gp_Ax2 frame = frame_of("add_rectangle", ox, oy, oz, nx, ny, nz);
  const TopoDS_Shape face = build_shape("add_rectangle", [&] {
    return BRepBuilderAPI_MakeFace(gp_Pln(gp_Ax3(frame)), 0.0, dx, 0.0, dy).Face();
  });
  return add_bodies(face, "add_rectangle");
}

// Join loose edges and wires into one wire, consuming them.
py::dict Session::make_wire(const std::vector<EntityId>& edge_ids) {
  OpGuard guard(in_op_);
  const std::vector<TopoDS_Shape> edges = edges_of("make_wire", edge_ids);
  const std::vector<TopoDS_Shape> owners = owner_bodies("make_wire", edge_ids);
  for (const TopoDS_Shape& b : owners) {
    require_curve_body("make_wire", b);
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(owners);
  TopoDS_Shape wire;
  {
    py::gil_scoped_release release;
    BRepBuilderAPI_MakeWire mk;
    NCollection_List<TopoDS_Shape> list;
    for (const TopoDS_Shape& e : weld_near_ends(edges)) {
      list.Append(e);
    }
    mk.Add(list);
    if (mk.IsDone()) {
      wire = mk.Wire();
    }
  }
  if (wire.IsNull()) {
    throw PysmeshError(
        "Session.make_wire: the named edges do not form a connected wire.",
        "BRepBuilderAPI_MakeWire reports DisconnectedWire: every edge after the first "
        "must share or geometrically touch a vertex of the wire built so far.",
        ids_as_int(edge_ids));
  }
  return commit(concat(survivors, wire), Handle(BRepTools_History)(), "make_wire", wire);
}

// Copy the named edges into one new loose wire body, leaving the originals alone.
py::dict Session::extract_edges(const std::vector<EntityId>& edge_ids) {
  OpGuard guard(in_op_);
  // No require_curve_body: the whole point is to reach the edges of a solid or a face,
  // which make_wire, make_face and make_filling all refuse because they consume what they
  // are given. This consumes nothing, so any owner is legitimate.
  const std::vector<TopoDS_Shape> edges = edges_of("extract_edges", edge_ids);

  TopoDS_Shape wire;
  std::vector<EntityId> disconnected;
  {
    py::gil_scoped_release release;
    // Copied as one compound rather than edge by edge. A compound copy carries the sharing
    // with it, so two edges that met at one vertex still meet at one vertex afterwards; copy
    // them separately and every junction becomes two coincident vertices that the wire
    // builder then has to weld.
    TopoDS_Compound bundle;
    BRep_Builder builder;
    builder.MakeCompound(bundle);
    for (const TopoDS_Shape& e : edges) {
      builder.Add(bundle, e);
    }
    BRepBuilderAPI_Copy copier;
    copier.Perform(bundle);
    std::vector<TopoDS_Shape> copies;
    for (TopoDS_Iterator it(copier.Shape()); it.More(); it.Next()) {
      copies.push_back(it.Value());
    }
    if (copies.size() == edges.size()) {
      // Added one at a time, sweeping until nothing more connects, rather than through
      // Add(list). The list form reports DisconnectedWire while still returning IsDone()
      // true over the part it managed to join, which would commit a fraction of the
      // selection as the answer. Sweeping also says WHICH edges are left over, and does not
      // depend on the order the caller happened to name them in: an edge that cannot join
      // the chain yet gets another attempt once the chain has grown.
      BRepBuilderAPI_MakeWire mk;
      std::vector<bool> placed(copies.size(), false);
      for (bool growing = true; growing;) {
        growing = false;
        for (std::size_t i = 0; i < copies.size(); ++i) {
          if (placed[i]) {
            continue;
          }
          mk.Add(TopoDS::Edge(copies[i]));
          if (mk.IsDone()) {
            placed[i] = true;
            growing = true;
          }
        }
      }
      for (std::size_t i = 0; i < placed.size(); ++i) {
        if (!placed[i]) {
          disconnected.push_back(edge_ids[i]);
        }
      }
      if (disconnected.empty()) {
        wire = mk.Wire();
      }
    }
  }
  if (!disconnected.empty()) {
    throw PysmeshError(
        "Session.extract_edges: " + std::to_string(disconnected.size()) + " of the " +
            std::to_string(edges.size()) +
            " named edges do not join the others into one connected wire.",
        "Every edge must share an end vertex with the chain, or have one within the two "
        "vertices' tolerance. The ids reported are the edges left over once every edge "
        "that could be joined had been.",
        ids_as_int(disconnected));
  }
  if (wire.IsNull()) {
    throw PysmeshError("Session.extract_edges: OCCT could not copy the named edges into a "
                       "wire.",
                       "BRepBuilderAPI_Copy or BRepBuilderAPI_MakeWire produced nothing.",
                       ids_as_int(edge_ids));
  }
  // Committed with no history, for the reason copy() gives: a relation between an original
  // and its duplicate would move the original's id onto the duplicate. The originals keep
  // their ids because they are still in the model, untouched; every entity of the new wire
  // is a new identity.
  return add_bodies(wire, "extract_edges");
}

// A planar face bounded by the named edges, consuming them.
py::dict Session::make_face(const std::vector<EntityId>& edge_ids) {
  OpGuard guard(in_op_);
  const std::vector<TopoDS_Shape> edges = edges_of("make_face", edge_ids);
  const std::vector<TopoDS_Shape> owners = owner_bodies("make_face", edge_ids);
  for (const TopoDS_Shape& b : owners) {
    require_curve_body("make_face", b);
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(owners);
  TopoDS_Shape face;
  {
    py::gil_scoped_release release;
    const TopoDS_Wire wire = wire_over("make_face", edges, owners);
    // OnlyPlane: a non-planar boundary is a fail-loud here rather than a silently
    // approximated surface. make_filling is the operation for that case.
    BRepBuilderAPI_MakeFace mk(wire, /*OnlyPlane=*/true);
    if (mk.IsDone()) {
      face = mk.Face();
    }
  }
  if (face.IsNull()) {
    throw PysmeshError(
        "Session.make_face: the named edges do not bound a closed planar face.",
        "BRepBuilderAPI_MakeFace(wire, OnlyPlane=true) failed. Use make_filling for a "
        "non-planar boundary.",
        ids_as_int(edge_ids));
  }
  return commit(concat(survivors, face), Handle(BRepTools_History)(), "make_face", face);
}

// A surface filling the named boundary edges, consuming them. Unlike make_face this
// handles a non-planar boundary; the surface is an approximation, so the result's edges
// are new geometry and the boundary edges' ids die.
py::dict Session::make_filling(const std::vector<EntityId>& edge_ids,
                               const std::optional<std::map<EntityId, EntityId>>& support_faces,
                               const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  if (support_faces.has_value() && !support_faces->empty()) {
    return filling_with_supports(edge_ids, *support_faces, progress, cancel);
  }
  const std::vector<TopoDS_Shape> edges = edges_of("make_filling", edge_ids);
  const std::vector<TopoDS_Shape> owners = owner_bodies("make_filling", edge_ids);
  for (const TopoDS_Shape& b : owners) {
    require_curve_body("make_filling", b);
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(owners);
  ProgressDriver driver("make_filling", hooks_of("make_filling", progress, cancel));
  TopoDS_Shape face;
  {
    py::gil_scoped_release release;
    BRepOffsetAPI_MakeFilling mk;
    for (const TopoDS_Shape& e : edges) {
      mk.Add(TopoDS::Edge(e), GeomAbs_C0);
    }
    try {
      mk.Build(driver.range());
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(
          std::string("Session.make_filling: BRepOffsetAPI_MakeFilling::Build failed: ") +
          e.what());
    }
    if (mk.IsDone()) {
      face = mk.Shape();
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("make_filling");
  }
  if (face.IsNull()) {
    throw PysmeshError("Session.make_filling: OCCT could not fill the named boundary.",
                       "BRepOffsetAPI_MakeFilling::IsDone() is false.",
                       ids_as_int(edge_ids));
  }
  return commit(concat(survivors, face), Handle(BRepTools_History)(), "make_filling",
                face);
}

// make_filling with support faces, as GEOM's MakeFaceWithConstraints builds it
// (GEOMImpl_FillingDriver, FILLING_ON_CONSTRAINTS): each supported edge is added with its
// face and GeomAbs_G1, every other edge with GeomAbs_C0, in the order named. A supported
// edge is an edge of its support face, so it belongs to that face's body: it is not
// consumed, and its body stays as it is. The other edges follow make_filling's own rule:
// loose edges and wires, consumed.
py::dict Session::filling_with_supports(const std::vector<EntityId>& edge_ids,
                                        const std::map<EntityId, EntityId>& support_faces,
                                        const py::object& progress,
                                        const py::object& cancel) {
  static constexpr const char* kOp = "make_filling";
  const std::vector<TopoDS_Shape> edges = edges_of(kOp, edge_ids);
  std::string named;
  for (EntityId id : edge_ids) {
    named += (named.empty() ? "" : ", ") + std::to_string(id);
  }
  std::vector<TopoDS_Face> supports(edges.size());
  for (const auto& [edge_id, face_id] : support_faces) {
    if (std::find(edge_ids.begin(), edge_ids.end(), edge_id) == edge_ids.end()) {
      throw PysmeshError("Session.make_filling: support edge " + std::to_string(edge_id) +
                             " is not one of the boundary edges (" + named + ").",
                         "", {static_cast<int>(edge_id)});
    }
    const TopoDS_Shape& edge = require_alive(kOp, edge_id).shapes.front();
    const TopoDS_Face face = sole_face(kOp, face_id);
    ShapeSet face_edges;
    TopExp::MapShapes(face, TopAbs_EDGE, face_edges);
    if (!face_edges.Contains(edge)) {
      throw PysmeshError("Session.make_filling: face " + std::to_string(face_id) +
                             " does not contain edge " + std::to_string(edge_id) +
                             "; the filling can be tangent to a face only along an edge of "
                             "that face.",
                         "", {static_cast<int>(edge_id), static_cast<int>(face_id)});
    }
    for (std::size_t i = 0; i < edges.size(); ++i) {
      if (!edges[i].IsSame(edge)) {
        continue;
      }
      if (!supports[i].IsNull() && !supports[i].IsSame(face)) {
        throw PysmeshError("Session.make_filling: edge " + std::to_string(edge_id) +
                               " is given two support faces.",
                           "", {static_cast<int>(edge_id), static_cast<int>(face_id)});
      }
      supports[i] = face;
    }
  }
  // BRepFill_Filling maps each boundary vertex to one new vertex by identity, so edges
  // that only touch at coincident vertices of different bodies leave the new wire open
  // and the face invalid (BRepCheck_NotClosed; measured on OCCT 8.0.1, with or without a
  // support). Every end vertex is shared by exactly two boundary edges, or none is built.
  {
    NCollection_IndexedDataMap<TopoDS_Shape, int, TopTools_ShapeMapHasher> ends;
    std::vector<std::pair<TopoDS_Shape, EntityId>> first_edge;
    for (std::size_t i = 0; i < edges.size(); ++i) {
      TopoDS_Vertex a, b;
      TopExp::Vertices(TopoDS::Edge(edges[i]), a, b);
      for (const TopoDS_Vertex& v : {a, b}) {
        if (v.IsNull()) {
          continue;
        }
        const int k = ends.FindIndex(v);
        if (k == 0) {
          ends.Add(v, 1);
          first_edge.emplace_back(v, edge_ids[i]);
        } else {
          ++ends.ChangeFromIndex(k);
        }
      }
    }
    for (const auto& [v, edge_id] : first_edge) {
      const int count = ends.FindFromKey(v);
      if (count == 2) {
        continue;
      }
      const auto ids = state_.registry->by_shape.find(v);
      const std::string vertex = ids == state_.registry->by_shape.end()
                                     ? std::string("an end vertex")
                                     : "vertex " + std::to_string(ids->second.front());
      throw PysmeshError("Session.make_filling: " + vertex + " of edge " +
                             std::to_string(edge_id) + " is an end of " +
                             std::to_string(count) +
                             " of the boundary edges, not 2; the boundary edges (" + named +
                             ") do not meet at shared vertices, so they bound no face.",
                         "", {static_cast<int>(edge_id)});
    }
  }
  std::vector<EntityId> free_ids;
  for (EntityId id : edge_ids) {
    if (support_faces.find(id) == support_faces.end()) {
      free_ids.push_back(id);
    }
  }
  std::vector<TopoDS_Shape> consumed;
  if (!free_ids.empty()) {
    consumed = owner_bodies(kOp, free_ids);
    for (const TopoDS_Shape& b : consumed) {
      require_curve_body(kOp, b);
    }
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(consumed);
  ProgressDriver driver(kOp, hooks_of(kOp, progress, cancel));
  TopoDS_Shape face;
  {
    py::gil_scoped_release release;
    try {
      BRepOffsetAPI_MakeFilling mk;
      for (std::size_t i = 0; i < edges.size(); ++i) {
        if (supports[i].IsNull()) {
          mk.Add(TopoDS::Edge(edges[i]), GeomAbs_C0);
        } else {
          mk.Add(TopoDS::Edge(edges[i]), supports[i], GeomAbs_G1);
        }
      }
      mk.Build(driver.range());
      if (mk.IsDone()) {
        face = mk.Shape();
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.make_filling: BRepOffsetAPI_MakeFilling with "
                                     "support faces failed: ") +
                             e.what(),
                         "", ids_as_int(edge_ids));
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled(kOp);
  }
  if (face.IsNull()) {
    throw PysmeshError("Session.make_filling: OCCT could not fill the boundary edges (" +
                           named + ") tangent to their support faces.",
                       "BRepOffsetAPI_MakeFilling::IsDone() is false.",
                       ids_as_int(edge_ids));
  }
  return commit(concat(survivors, face), Handle(BRepTools_History)(), kOp, face);
}

// ---- extending a face or an edge ----------------------------------------------------- //
//
// Both add a new free body and leave the source where it is. OCCT gives no history for a
// surface or a curve rebuilt by GeomLib, so the relation is stated here: every sub-shape of
// the new body is Generated from the source, which names it against the source.

namespace {

// Six significant digits, as a refusal text prints a value.
std::string extend_num(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return buf;
}

Handle(BRepTools_History) generated_from_source(const TopoDS_Shape& source,
                                                const TopoDS_Shape& built) {
  Handle(BRepTools_History) hist = new BRepTools_History;
  for (const TopoDS_Shape& s : registered_subshapes(built)) {
    hist->AddGenerated(source, s);
  }
  return hist;
}

}  // namespace

py::dict Session::extend_face(EntityId face_id, double u_min, double u_max, double v_min,
                              double v_max) {
  const double lengths[4] = {u_min, u_max, v_min, v_max};
  const char* names[4] = {"u_min", "u_max", "v_min", "v_max"};
  for (int i = 0; i < 4; ++i) {
    finite_arg("extend_face", names[i], lengths[i]);
  }
  OpGuard guard(in_op_);
  bool any = false;
  for (int i = 0; i < 4; ++i) {
    if (lengths[i] < 0.0) {
      throw PysmeshError(std::string("Session.extend_face: the extension ") + names[i] +
                         " of face " + std::to_string(face_id) + " is " +
                         extend_num(lengths[i]) + "; a length is 0 or more.",
                         "", {static_cast<int>(face_id)});
    }
    any = any || lengths[i] > 0.0;
  }
  if (!any) {
    throw PysmeshError("Session.extend_face: every extension of face " +
                           std::to_string(face_id) + " is 0; nothing to extend.",
                       "", {static_cast<int>(face_id)});
  }
  const TopoDS_Face face = sole_face("extend_face", face_id);
  const std::string type = surface_type_name(BRepAdaptor_Surface(face, false).GetType());
  double u0 = 0.0, u1 = 0.0, v0 = 0.0, v1 = 0.0;
  BRepTools::UVBounds(face, u0, u1, v0, v1);
  if (Precision::IsInfinite(u0) || Precision::IsInfinite(u1) || Precision::IsInfinite(v0) ||
      Precision::IsInfinite(v1)) {
    throw PysmeshError("Session.extend_face: face " + std::to_string(face_id) +
                           " lies on an unbounded " + type +
                           " surface and has no boundary; there is no side to extend from.",
                       "", {static_cast<int>(face_id)});
  }
  const Handle(Geom_Surface) surface = BRep_Tool::Surface(face);
  // A face that covers a whole period of its surface has no free side in that direction:
  // its two sides meet at the seam.
  for (const bool in_u : {true, false}) {
    const bool asked = in_u ? (u_min > 0.0 || u_max > 0.0) : (v_min > 0.0 || v_max > 0.0);
    const bool periodic = in_u ? surface->IsUPeriodic() : surface->IsVPeriodic();
    if (!asked || !periodic) {
      continue;
    }
    const double span = in_u ? u1 - u0 : v1 - v0;
    const double period = in_u ? surface->UPeriod() : surface->VPeriod();
    if (span >= period - Precision::PConfusion()) {
      const char* dir = in_u ? "u" : "v";
      throw PysmeshError("Session.extend_face: face " + std::to_string(face_id) +
                             " covers the whole period of its " + type + " surface in " +
                             dir + " (" + extend_num(span) + "); its two sides in " + dir +
                             " meet at the seam, so there is none to extend.",
                         "", {static_cast<int>(face_id)});
    }
  }

  TopoDS_Shape built;
  std::string failed;
  {
    py::gil_scoped_release release;
    try {
      Handle(Geom_BoundedSurface) bounded =
          new Geom_RectangularTrimmedSurface(surface, u0, u1, v0, v1);
      for (int i = 0; i < 4 && failed.empty(); ++i) {
        if (lengths[i] <= 0.0) {
          continue;
        }
        // GeomLib replaces the handle with the extended B-spline, and leaves it alone when
        // it does nothing.
        const Handle(Geom_BoundedSurface) before = bounded;
        GeomLib::ExtendSurfByLength(bounded, lengths[i], /*Cont=*/1, /*InU=*/i < 2,
                                    /*After=*/i % 2 == 1);
        if (bounded == before) {
          failed = std::string("GeomLib::ExtendSurfByLength left the surface unchanged for ") +
                   names[i] + " = " + extend_num(lengths[i]);
        }
      }
      if (failed.empty()) {
        BRepBuilderAPI_MakeFace mk(bounded, Precision::Confusion());
        if (mk.IsDone()) {
          built = mk.Face();
          if (face.Orientation() == TopAbs_REVERSED) {
            built.Reverse();
          }
        } else {
          failed = "BRepBuilderAPI_MakeFace could not build a face on the extended surface";
        }
      }
    } catch (const std::exception& e) {
      failed = std::string("OCCT threw: ") + e.what();
    }
  }
  if (!failed.empty()) {
    throw PysmeshError("Session.extend_face: OCCT could not extend face " +
                           std::to_string(face_id) + " (" + type + "): " + failed + ".",
                       "", {static_cast<int>(face_id)});
  }
  return commit(concat(root_bodies(state_.root), built), generated_from_source(face, built),
                "extend_face", built);
}

py::dict Session::extend_edge(EntityId edge_id, double length, bool at_end) {
  finite_arg("extend_edge", "length", length);
  OpGuard guard(in_op_);
  if (!(length > 0.0)) {
    throw PysmeshError("Session.extend_edge: the extension of edge " + std::to_string(edge_id) +
                           " is " + extend_num(length) + "; a length is more than 0.",
                       "", {static_cast<int>(edge_id)});
  }
  const TopoDS_Edge edge = sole_edge("extend_edge", edge_id);
  double first = 0.0, last = 0.0;
  const Handle(Geom_Curve) curve =
      BRep_Tool::Degenerated(edge) ? Handle(Geom_Curve)() : BRep_Tool::Curve(edge, first, last);
  if (curve.IsNull()) {
    throw PysmeshError("Session.extend_edge: edge " + std::to_string(edge_id) +
                           " has no 3-D curve (a degenerated edge), so it has no tangent to "
                           "extend along.",
                       "", {static_cast<int>(edge_id)});
  }
  const std::string type = curve_type_name(BRepAdaptor_Curve(edge).GetType());
  TopoDS_Vertex v_first, v_last;
  TopExp::Vertices(edge, v_first, v_last);
  if (!v_first.IsNull() && v_first.IsSame(v_last)) {
    throw PysmeshError("Session.extend_edge: edge " + std::to_string(edge_id) +
                           " is a closed " + type + "; it has no free end to extend.",
                       "", {static_cast<int>(edge_id)});
  }
  // The edge's end in the orientation the root holds it in is the curve's last parameter on
  // a FORWARD edge and its first one on a REVERSED edge.
  const bool forward = edge.Orientation() != TopAbs_REVERSED;
  const bool after = at_end == forward;
  gp_Pnt end_point;
  gp_Vec d1;
  curve->D1(after ? last : first, end_point, d1);
  if (d1.Magnitude() <= Precision::Confusion()) {
    throw PysmeshError("Session.extend_edge: the tangent of edge " + std::to_string(edge_id) +
                           " at that end is the null vector; there is no direction to extend "
                           "along.",
                       "", {static_cast<int>(edge_id)});
  }
  const gp_Vec outward = (after ? d1 : -d1).Normalized();
  const gp_Pnt target = end_point.Translated(outward * length);

  TopoDS_Shape built;
  std::string failed;
  {
    py::gil_scoped_release release;
    try {
      Handle(Geom_BoundedCurve) bounded = new Geom_TrimmedCurve(curve, first, last);
      const Handle(Geom_BoundedCurve) before = bounded;
      GeomLib::ExtendCurveToPoint(bounded, target, /*Cont=*/1, after);
      if (bounded == before) {
        failed = "GeomLib::ExtendCurveToPoint left the curve unchanged";
      } else {
        BRepBuilderAPI_MakeEdge mk(bounded);
        if (mk.IsDone()) {
          built = mk.Edge();
          if (!forward) {
            built.Reverse();
          }
        } else {
          failed = "BRepBuilderAPI_MakeEdge could not build an edge on the extended curve";
        }
      }
    } catch (const std::exception& e) {
      failed = std::string("OCCT threw: ") + e.what();
    }
  }
  if (!failed.empty()) {
    throw PysmeshError("Session.extend_edge: OCCT could not extend edge " +
                           std::to_string(edge_id) + " (" + type + ") by " +
                           extend_num(length) + ": " + failed + ".",
                       "", {static_cast<int>(edge_id)});
  }
  return commit(concat(root_bodies(state_.root), built), generated_from_source(edge, built),
                "extend_edge", built);
}

// ---- sweeps ----------------------------------------------------------------------- //
//
// Each sweep consumes the profile body and raises its dimension: an edge sweeps to a
// face, a wire to a shell, a face to a solid. The profile survives inside the result as
// the sweep's first shape, so its entity ids carry through structurally, and the walls
// the sweep generates are named against the profile edges they came from.

py::dict Session::extrude(const std::vector<EntityId>& entity_ids, double vx, double vy,
                   double vz, double draft_angle, bool symmetric) {
  finite_arg("extrude", "vector", vx, vy, vz);
  finite_arg("extrude", "draft_angle", draft_angle);
  OpGuard guard(in_op_);
  const gp_Vec vec(vx, vy, vz);
  if (vec.Magnitude() <= 0.0) {
    throw PysmeshError("Session.extrude: the extrusion vector must be non-zero.");
  }
  if (draft_angle != 0.0 || symmetric) {
    return extrude_variant(entity_ids, vec, draft_angle, symmetric);
  }
  const TopoDS_Shape profile = sole_body("extrude", entity_ids);
  require_sweepable("extrude", profile);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({profile});

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  {
    py::gil_scoped_release release;
    // Every OCCT call runs inside the try, the history query included: whatever throws
    // after Build() reaches the caller as PysmeshError too (report A2).
    const char* stage = "BRepPrimAPI_MakePrism failed";
    try {
      BRepPrimAPI_MakePrism mk(profile, vec, /*Copy=*/false);
      mk.Build();
      if (mk.IsDone()) {
        stage = "reading the history of BRepPrimAPI_MakePrism failed";
        result = mk.Shape();
        hist = history_of(profile, mk);
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.extrude: ") + stage + ": " + e.what());
    }
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.extrude: OCCT could not sweep the profile.", "",
                       ids_as_int(entity_ids));
  }
  return commit(concat(survivors, result), hist, "extrude", result);
}

namespace {

// How far a draft sweep's vector may lean off the profile plane's normal, in radians: the
// angular resolution a caller's normal survives a scaling by the sweep length with.
constexpr double kDraftAxisTol = 1e-9;

constexpr const char* kDraftHint =
    "A positive draft moves every side of the profile inward by |vector| tan(draft_angle) "
    "over the sweep; where that reaches half the profile's width, the opposite walls meet and "
    "pass through each other.";

// The history of a drafted sweep of `spine` by BRepOffsetAPI_MakeEvolved along the one-edge
// profile `edge` (first vertex on the spine). MakeEvolved answers GeneratedShapes for a spine
// edge against each profile sub-shape, and gives the two caps; it answers nothing for a
// spine vertex. So: the spine face becomes the cap at the spine (Modified), each spine edge
// its image there (Modified), each spine vertex the matching end of that image (Modified, by
// the end that coincides with it); each spine edge generates its side face and its edge on
// the far cap, and the spine face generates the far cap.
Handle(BRepTools_History) drafted_history(const TopoDS_Shape& spine, const TopoDS_Edge& edge,
                                          BRepOffsetAPI_MakeEvolved& mk) {
  Handle(BRepTools_History) hist = new BRepTools_History;
  TopoDS_Vertex v0, v1;
  TopExp::Vertices(edge, v0, v1);
  ShapeSet spine_edges;
  TopExp::MapShapes(spine, TopAbs_EDGE, spine_edges);
  ShapeSet near_edges;
  ShapeSet mapped_vertices;
  for (int i = 1; i <= spine_edges.Extent(); ++i) {
    const TopoDS_Shape& e = spine_edges(i);
    const NCollection_List<TopoDS_Shape>& near = mk.GeneratedShapes(e, v0);
    for (const TopoDS_Shape& img : near) {
      hist->AddModified(e, img);
      near_edges.Add(img);
    }
    for (const TopoDS_Shape& f : mk.GeneratedShapes(e, edge)) {
      hist->AddGenerated(e, f);
    }
    for (const TopoDS_Shape& img : mk.GeneratedShapes(e, v1)) {
      hist->AddGenerated(e, img);
    }
    if (near.Extent() != 1 || near.First().ShapeType() != TopAbs_EDGE) {
      continue;
    }
    TopoDS_Vertex a, b, c, d;
    TopExp::Vertices(TopoDS::Edge(e), a, b);
    TopExp::Vertices(TopoDS::Edge(near.First()), c, d);
    if (a.IsNull() || b.IsNull() || c.IsNull() || d.IsNull()) {
      continue;
    }
    const bool straight = BRep_Tool::Pnt(a).Distance(BRep_Tool::Pnt(c)) <=
                          BRep_Tool::Pnt(a).Distance(BRep_Tool::Pnt(d));
    const TopoDS_Vertex pairs[2][2] = {{a, straight ? c : d}, {b, straight ? d : c}};
    for (const auto& pair : pairs) {
      if (!mapped_vertices.Contains(pair[0])) {
        hist->AddModified(pair[0], pair[1]);
        mapped_vertices.Add(pair[0]);
      }
    }
  }
  // The cap at the spine is the one holding the spine edges' images.
  const TopoDS_Shape caps[2] = {mk.Top(), mk.Bottom()};
  int near_cap = 0;
  for (int k = 0; k < 2; ++k) {
    for (TopExp_Explorer ex(caps[k], TopAbs_EDGE); ex.More(); ex.Next()) {
      if (near_edges.Contains(ex.Current())) {
        near_cap = k;
      }
    }
  }
  ShapeSet spine_faces;
  TopExp::MapShapes(spine, TopAbs_FACE, spine_faces);
  if (spine_faces.Extent() == 1) {
    hist->AddModified(spine_faces(1), caps[near_cap]);
    hist->AddGenerated(spine_faces(1), caps[1 - near_cap]);
  }
  return hist;
}

// The evolved sweep lists the same faces in an order that varies from run to run:
// BRepFill_Evolved gathers them through maps keyed by shape address (two fresh sessions gave
// equal geometry in a different order, so different bytes and ids). It also returns the
// same solid as a REVERSED shell of REVERSED faces in some runs and a FORWARD shell of
// FORWARD faces in others (24 of 40 runs in one process, at -10 degrees, symmetric). Each
// shell is rebuilt FORWARD in a FORWARD solid, each face carrying the orientation it has in
// the solid, from the same faces, same TShapes, in an order of their own geometry: area
// centroid, area, then box. Equal input then gives equal bytes and equal ids, and the
// history, which names the faces themselves (IsSame ignores orientation), holds as it is.
TopoDS_Shape in_geometric_order(const TopoDS_Shape& shape) {
  BRep_Builder builder;
  std::vector<TopoDS_Shape> solids;
  for (TopExp_Explorer ex(shape, TopAbs_SOLID); ex.More(); ex.Next()) {
    const TopoDS_Shape& solid = ex.Current();
    TopoDS_Solid rebuilt;
    builder.MakeSolid(rebuilt);
    for (TopoDS_Iterator sh(solid, /*cumOri=*/true, /*cumLoc=*/false); sh.More(); sh.Next()) {
      std::vector<std::pair<std::array<double, 10>, TopoDS_Shape>> faces;
      for (TopoDS_Iterator f(sh.Value(), /*cumOri=*/true, /*cumLoc=*/false); f.More();
           f.Next()) {
        GProp_GProps props;
        BRepGProp::SurfaceProperties(f.Value(), props);
        const gp_Pnt c = props.CentreOfMass();
        double x0, y0, z0, x1, y1, z1;
        shape_checks::exact_box(f.Value()).Get(x0, y0, z0, x1, y1, z1);
        faces.push_back({{c.X(), c.Y(), c.Z(), props.Mass(), x0, y0, z0, x1, y1, z1},
                         f.Value()});
      }
      std::stable_sort(faces.begin(), faces.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });
      TopoDS_Shell shell;
      builder.MakeShell(shell);
      for (const auto& f : faces) {
        builder.Add(shell, f.second);
      }
      shell.Closed(sh.Value().Closed());
      shell.Location(sh.Value().Location());
      builder.Add(rebuilt, shell);
    }
    rebuilt.Location(solid.Location());
    solids.push_back(rebuilt);
  }
  if (solids.size() == 1) {
    return solids.front();
  }
  return make_root(solids);
}

}  // namespace

py::dict Session::extrude_variant(const std::vector<EntityId>& entity_ids, const gp_Vec& vec,
                                  double draft_angle, bool symmetric) {
  const TopoDS_Shape profile = sole_body("extrude", entity_ids);
  require_sweepable("extrude", profile);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({profile});
  const bool drafted = draft_angle != 0.0;
  gp_Dir axis;
  if (drafted) {
    if (!(std::abs(draft_angle) < 0.5 * M_PI)) {
      throw PysmeshError("Session.extrude: the draft angle " + extend_num(draft_angle) +
                             " rad is not between -pi/2 and pi/2; it leans the walls flat or "
                             "past.",
                         "", ids_as_int(entity_ids));
    }
    if (profile.ShapeType() != TopAbs_FACE) {
      throw PysmeshError("Session.extrude: a draft angle applies to a planar face profile; the "
                         "named entities belong to a " +
                             std::string(TopAbs::ShapeTypeToString(profile.ShapeType())) +
                             " body.",
                         "", ids_as_int(entity_ids));
    }
    const BRepAdaptor_Surface surface(TopoDS::Face(profile), false);
    if (surface.GetType() != GeomAbs_Plane) {
      throw PysmeshError("Session.extrude: a draft angle applies to a planar face profile; the "
                         "profile face is a " +
                             std::string(surface_type_name(surface.GetType())) + " face.",
                         "", ids_as_int(entity_ids));
    }
    axis = surface.Plane().Axis().Direction();
    const double off = vec.Angle(gp_Vec(axis));
    const double tilt = std::min(off, M_PI - off);
    if (tilt > kDraftAxisTol) {
      throw PysmeshError("Session.extrude: a draft angle applies to a sweep along the profile "
                         "face's normal; the vector is " +
                             extend_num(tilt) + " rad off it.",
                         "", ids_as_int(entity_ids));
    }
  }

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::string interference;
  double length = vec.Magnitude();
  double reached = 0.0;
  {
    py::gil_scoped_release release;
    const char* stage = "moving the profile by -vector failed";
    try {
      // A symmetric sweep starts at -vector: the profile is moved there, by a change of
      // location, and swept twice as far. The move's history carries the profile's ids onto
      // the moved profile, and the sweep's history is composed after it.
      TopoDS_Shape base = profile;
      gp_Vec sweep = vec;
      Handle(BRepTools_History) moved;
      if (symmetric) {
        gp_Trsf shift;
        shift.SetTranslation(-vec);
        BRepBuilderAPI_Transform mover(profile, shift, /*Copy=*/false);
        base = mover.Shape();
        moved = history_of(profile, mover);
        sweep = 2.0 * vec;
        length = sweep.Magnitude();
      }
      Handle(BRepTools_History) swept;
      if (!drafted) {
        stage = "BRepPrimAPI_MakePrism failed";
        BRepPrimAPI_MakePrism mk(base, sweep, /*Copy=*/false);
        mk.Build();
        if (mk.IsDone()) {
          stage = "reading the history of BRepPrimAPI_MakePrism failed";
          result = mk.Shape();
          swept = history_of(base, mk);
        }
      } else {
        // The evolved sweep of the face along one straight profile edge. The profile is given
        // in the global frame (theIsAxeProf): its z runs along the spine plane's own axis,
        // whatever the face's orientation, and +y moves the far cap into the face, so a
        // positive draft narrows the solid (measured on OCCT 8.0.1).
        stage = "BRepOffsetAPI_MakeEvolved failed";
        const double z = sweep.Dot(gp_Vec(axis)) > 0.0 ? length : -length;
        const TopoDS_Edge edge =
            BRepBuilderAPI_MakeEdge(gp_Pnt(0.0, 0.0, 0.0),
                                    gp_Pnt(0.0, length * std::tan(draft_angle), z))
                .Edge();
        const TopoDS_Wire wire = BRepBuilderAPI_MakeWire(edge).Wire();
        BRepOffsetAPI_MakeEvolved mk(base, wire, GeomAbs_Arc, /*theIsAxeProf=*/true,
                                     /*theIsSolid=*/true, /*theIsProfOnSpine=*/false,
                                     Precision::Confusion(), /*theIsVolume=*/false,
                                     /*theRunInParallel=*/false);
        mk.Build();
        if (mk.IsDone()) {
          stage = "reading the history of BRepOffsetAPI_MakeEvolved failed";
          result = in_geometric_order(mk.Shape());
          swept = drafted_history(base, edge, mk);
          stage = "checking the drafted solid for self-interference failed";
          interference = shape_checks::self_interference_refusal("Session.extrude", result);
          // How far the solid reaches along the profile's normal: the z range of its box in
          // the profile plane's frame. Where a narrowing draft makes the walls meet before
          // the sweep ends, MakeEvolved gives a valid solid that stops there, in a ridge.
          stage = "measuring the drafted solid's height failed";
          gp_Trsf to_plane;
          to_plane.SetTransformation(
              BRepAdaptor_Surface(TopoDS::Face(base), false).Plane().Position());
          double x0, y0, z0, x1, y1, z1;
          shape_checks::exact_box(result.Moved(TopLoc_Location(to_plane)))
              .Get(x0, y0, z0, x1, y1, z1);
          reached = std::max(std::abs(z0), std::abs(z1));
        }
      }
      if (!result.IsNull()) {
        if (moved.IsNull()) {
          hist = swept;
        } else {
          moved->Merge(swept);
          hist = moved;
        }
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.extrude: ") + stage + ": " + e.what());
    }
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.extrude: OCCT could not sweep the profile" +
                           std::string(drafted ? " with the draft angle " +
                                                     extend_num(draft_angle) + " rad"
                                               : std::string()) +
                           ".",
                       "", ids_as_int(entity_ids));
  }
  refuse_self_interference(interference, kDraftHint);
  if (drafted &&
      reached < length - std::max(shape_checks::max_tolerance({result}), Precision::Confusion())) {
    throw PysmeshError("Session.extrude: at the draft angle " + extend_num(draft_angle) +
                           " rad the walls meet " + extend_num(reached) +
                           " from the profile, short of the sweep length " +
                           extend_num(length) + "; the solid would end there in a ridge.",
                       kDraftHint, ids_as_int(entity_ids));
  }
  return commit(concat(survivors, result), hist, "extrude", result);
}

py::dict Session::revolve(const std::vector<EntityId>& entity_ids, double ox, double oy,
                          double oz, double ax, double ay, double az, double angle_rad) {
  finite_arg("revolve", "origin", ox, oy, oz);
  finite_arg("revolve", "axis", ax, ay, az);
  finite_arg("revolve", "angle_rad", angle_rad);
  OpGuard guard(in_op_);
  require_sweep_angle("revolve", angle_rad);
  const gp_Ax1 axis(gp_Pnt(ox, oy, oz), direction_of("revolve", "axis", ax, ay, az));
  const TopoDS_Shape profile = sole_body("revolve", entity_ids);
  require_sweepable("revolve", profile);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({profile});

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  {
    py::gil_scoped_release release;
    const char* stage = "BRepPrimAPI_MakeRevol failed";
    try {
      BRepPrimAPI_MakeRevol mk(profile, axis, angle_rad, /*Copy=*/false);
      mk.Build();
      if (mk.IsDone()) {
        stage = "reading the history of BRepPrimAPI_MakeRevol failed";
        result = mk.Shape();
        hist = history_of(profile, mk);
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.revolve: ") + stage + ": " + e.what());
    }
  }
  if (result.IsNull()) {
    throw PysmeshError(
        "Session.revolve: OCCT could not revolve the profile.",
        "A profile that crosses the axis cannot be revolved without self-intersection.",
        ids_as_int(entity_ids));
  }
  return commit(concat(survivors, result), hist, "revolve", result);
}

py::dict Session::pipe(const std::vector<EntityId>& spine_ids,
                const std::vector<EntityId>& profile_ids, const py::object& progress,
                const py::object& cancel) {
  OpGuard guard(in_op_);
  const TopoDS_Shape spine_body = sole_body("pipe", spine_ids);
  const TopoDS_Shape profile = sole_body("pipe", profile_ids);
  if (spine_body.IsSame(profile)) {
    throw PysmeshError("Session.pipe: the spine and the profile are the same body.");
  }
  require_sweepable("pipe", profile);
  const TopoDS_Wire spine = wire_of_body("pipe", "spine_ids", spine_body);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({spine_body, profile});

  ProgressDriver driver("pipe", hooks_of("pipe", progress, cancel));
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::string interference;
  {
    py::gil_scoped_release release;
    const char* stage = "BRepOffsetAPI_MakePipe failed";
    try {
      BRepOffsetAPI_MakePipe mk(spine, profile);
      mk.Build(driver.range());
      if (mk.IsDone()) {
        stage = "reading the history of BRepOffsetAPI_MakePipe failed";
        result = mk.Shape();
        hist = history_of(profile, mk);
        stage = "checking the swept shape for self-interference failed";
        interference = shape_checks::self_interference_refusal("Session.pipe", result);
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.pipe: ") + stage + ": " + e.what());
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("pipe");
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.pipe: OCCT could not sweep the profile along the spine.",
                       "", ids_as_int(profile_ids));
  }
  refuse_self_interference(interference, kSweepHint);
  return commit(concat(survivors, result), hist, "pipe", result);
}

// The general sweep. Unlike pipe it exposes the frame law (Frenet vs corrected Frenet)
// and can close the shell into a solid, which is what a swept CFD body normally needs.
py::dict Session::pipe_shell(const std::vector<EntityId>& spine_ids,
                      const std::vector<EntityId>& profile_ids, bool frenet, bool solid,
                      const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  const TopoDS_Shape spine_body = sole_body("pipe_shell", spine_ids);
  const TopoDS_Shape profile = sole_body("pipe_shell", profile_ids);
  if (spine_body.IsSame(profile)) {
    throw PysmeshError("Session.pipe_shell: the spine and the profile are the same body.");
  }
  const TopoDS_Wire spine = wire_of_body("pipe_shell", "spine_ids", spine_body);
  const TopoDS_Wire prof = wire_of_body("pipe_shell", "profile_ids", profile);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({spine_body, profile});

  ProgressDriver driver("pipe_shell", hooks_of("pipe_shell", progress, cancel));
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::string detail;
  std::string interference;
  {
    py::gil_scoped_release release;
    const char* stage = "setting up BRepOffsetAPI_MakePipeShell failed";
    try {
      BRepOffsetAPI_MakePipeShell mk(spine);
      mk.SetMode(frenet);
      mk.Add(prof);
      if (!mk.IsReady()) {
        detail = "BRepOffsetAPI_MakePipeShell::IsReady() is false.";
      } else {
        stage = "BRepOffsetAPI_MakePipeShell failed";
        mk.Build(driver.range());
        if (mk.IsDone()) {
          stage = "BRepOffsetAPI_MakePipeShell::MakeSolid failed";
          if (solid && !mk.MakeSolid()) {
            detail = "MakeSolid() failed: the swept shell is not closed.";
          } else {
            stage = "reading the history of BRepOffsetAPI_MakePipeShell failed";
            result = mk.Shape();
            hist = history_of(profile, mk);
            stage = "checking the swept shape for self-interference failed";
            interference =
                shape_checks::self_interference_refusal("Session.pipe_shell", result);
          }
        }
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.pipe_shell: ") + stage + ": " + e.what());
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("pipe_shell");
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.pipe_shell: OCCT could not sweep the profile.", detail,
                       ids_as_int(profile_ids));
  }
  refuse_self_interference(interference, kSweepHint);
  return commit(concat(survivors, result), hist, "pipe_shell", result);
}

// Loft through an ordered list of section wires, consuming all of them.
py::dict Session::thru_sections(const std::vector<std::vector<EntityId>>& sections, bool solid,
                         bool ruled, const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  if (sections.size() < 2) {
    throw PysmeshError("Session.thru_sections: at least two sections are required (got " +
                       std::to_string(sections.size()) + ").");
  }
  // The first section named again as the last closes the loft round onto itself (report
  // C3). OCCT takes its closed path only when the first and the last wire are the same
  // shape (BRepOffsetAPI_ThruSections.cxx: myWires(1).IsSame(myWires(nbSects))), and
  // wire_of_body builds a new wire on each call, so the last section reuses the first wire.
  // A closed loft needs two other sections at least; any other repeat is refused.
  std::vector<TopoDS_Shape> bodies;
  std::vector<TopoDS_Wire> wires;
  bool closed = false;
  for (std::size_t k = 0; k < sections.size(); ++k) {
    // A section is one wire. Edges of several loose bodies are a wire not yet joined, so
    // the refusal names the section and the operation that joins it, rather than
    // sole_body's advice to split the call, which a loft cannot follow.
    const std::vector<TopoDS_Shape> owners = owner_bodies("thru_sections", sections[k]);
    if (owners.size() != 1) {
      throw PysmeshError("Session.thru_sections: section " + std::to_string(k + 1) + " of " +
                         std::to_string(sections.size()) + " names entities of " +
                         std::to_string(owners.size()) +
                         " different bodies. A section is one wire or one single-edge "
                         "body: join the section's edges into one wire with make_wire, "
                         "then name that wire.");
    }
    const TopoDS_Shape body = owners.front();
    for (std::size_t j = 0; j < bodies.size(); ++j) {
      if (!bodies[j].IsSame(body)) {
        continue;
      }
      if (j == 0 && k + 1 == sections.size() && k >= 3) {
        closed = true;
        continue;
      }
      throw PysmeshError("Session.thru_sections: the same body was named as two different "
                         "sections (" +
                         std::to_string(j + 1) + " and " + std::to_string(k + 1) +
                         "). Only the first section may be named again, as the last, to "
                         "close the loft through two other sections at least.");
    }
    bodies.push_back(body);
    wires.push_back(closed ? wires.front() : wire_of_body("thru_sections", "sections", body));
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(bodies);

  ProgressDriver driver("thru_sections", hooks_of("thru_sections", progress, cancel));
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::optional<EnclosedVolume> hollow;
  std::string cap_refusal;
  std::string slit_refusal;
  std::string interference;
  std::optional<LoftFold> fold;
  {
    py::gil_scoped_release release;
    // The history query once threw here outside the try (report A2, through O1), and a raw
    // RuntimeError reached the caller. Every OCCT call now runs inside it.
    const char* stage = "copying the sections failed";
    try {
      // The builder writes pcurves, surfaces and continuity onto the edges it lofts through,
      // even when it then fails (report A3). Those edges are the session's own, and every
      // retained snapshot shares them, so it lofts deep copies. Not SetMutableInput(false):
      // OCCT then lofts copies of its own and reports no map from a section edge to its
      // copy (ThruSections has no Modified()), so every section edge and vertex id of a
      // ruled loft died. The copier's map carries them instead, as modified.
      TopoDS_Compound originals;
      BRep_Builder builder;
      builder.MakeCompound(originals);
      for (std::size_t k = 0; k < wires.size(); ++k) {
        if (!(closed && k + 1 == wires.size())) {
          builder.Add(originals, wires[k]);
        }
      }
      BRepBuilderAPI_Copy copier(originals, /*copyGeom=*/true, /*copyMesh=*/false);
      NCollection_List<TopoDS_Shape> copies;
      stage = "BRepOffsetAPI_ThruSections failed";
      BRepOffsetAPI_ThruSections mk(solid, ruled);
      for (const TopoDS_Wire& w : wires) {
        const TopoDS_Wire copy = TopoDS::Wire(copier.ModifiedShape(w));
        copies.Append(copy);
        mk.AddWire(copy);
      }
      mk.Build(driver.range());
      if (mk.IsDone()) {
        stage = "reading the history of BRepOffsetAPI_ThruSections failed";
        result = mk.Shape();
        NCollection_List<TopoDS_Shape> args;
        for (const TopoDS_Shape& b : bodies) {
          args.Append(b);
        }
        // Each section's own sub-shapes -> their copies -> what the loft made of them.
        hist = new BRepTools_History(args, copier);
        hist->Merge(BRepTools_History(copies, mk));
        // The caps first: a solid whose cap is missing has an open shell, and its volume
        // means nothing. A closed loft has no caps: OCCT closes it round onto itself.
        if (solid && !closed) {
          stage = "checking the end caps of the lofted solid failed";
          cap_refusal = loft_cap_refusal(result, mk.FirstShape(), mk.LastShape(), wires);
        }
        // OCCT orients the lofted solid itself, and on a loft that folds through itself its
        // answer is arbitrary. Measured over 30 seeded random ruled lofts through three
        // tilted sections, two came back with volumes -7.85 and -8.51 and
        // BRepCheck_Analyzer accepted both, so the solid's own volume is checked before it
        // is committed. It is not re-oriented: reversing a surface that crosses itself does
        // not give it an inside.
        stage = "measuring the volume of the lofted solid failed";
        for (TopExp_Explorer ex(result, TopAbs_SOLID); ex.More() && solid && cap_refusal.empty();
             ex.Next()) {
          const EnclosedVolume enclosed =
              enclosed_volume(ex.Current(), "Session.thru_sections", "the lofted solid");
          if (enclosed.volume <= enclosed.tolerance) {
            hollow = enclosed;
            break;
          }
        }
        if (solid && !closed && cap_refusal.empty() && !hollow.has_value()) {
          stage = "checking the end caps of the lofted solid for a slit failed";
          slit_refusal = coincident_caps_refusal(mk.FirstShape(), mk.LastShape());
        }
        if (solid && cap_refusal.empty() && slit_refusal.empty() && !hollow.has_value()) {
          stage = "checking the lofted solid for self-interference failed";
          interference =
              shape_checks::self_interference_refusal("Session.thru_sections", result);
        }
        // BRepCheck does not test a face for self-intersection, and the check above leaves
        // out the points near a face's own edges and seams, so sheets of a loft that meet
        // only along a seam, or a shell that is not a solid, pass both. The loft parameter
        // tells such a fold from a seam (loft_fold).
        if (cap_refusal.empty() && slit_refusal.empty() && !hollow.has_value() &&
            interference.empty()) {
          stage = "checking the loft for a fold failed";
          fold = loft_fold(result, mk.FirstShape(), mk.LastShape(), wires, ruled, closed);
        }
      }
    } catch (const PysmeshError&) {
      // A lofted solid the measure rule cannot measure is refused by name, with no stage.
      py::gil_scoped_acquire acquire;
      throw;
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.thru_sections: ") + stage + ": " + e.what());
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("thru_sections");
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.thru_sections: OCCT could not loft the sections.",
                       "Sections must all be closed or all be open, and must not "
                       "self-intersect when joined.",
                       {});
  }
  if (!cap_refusal.empty()) {
    throw PysmeshError(cap_refusal, kCapHint, {});
  }
  if (hollow.has_value()) {
    std::ostringstream s;
    s << "Session.thru_sections: the lofted solid encloses a volume of " << hollow->volume
      << ", at or below Precision::Confusion() x its area (" << hollow->tolerance
      << "), so its interior is not on its inside. Nothing is committed; the session is "
         "unchanged.";
    throw PysmeshError(s.str(),
                       "A loft whose surface folds through itself does this: a ruled loft "
                       "through sections tilted towards each other is the measured case. "
                       "Space or align the sections so that consecutive ones do not cross "
                       "each other's path.",
                       {});
  }
  if (!slit_refusal.empty()) {
    throw PysmeshError(slit_refusal, kSlitHint, {});
  }
  refuse_self_interference(interference, kLoftHint);
  if (fold.has_value()) {
    const int spans = static_cast<int>(wires.size()) - 1;
    std::ostringstream s;
    s << "Session.thru_sections: the loft folds back through itself: FACE " << fold->face_a
      << " " << loft_interval_text(fold->g_a, spans) << " and FACE " << fold->face_b << " "
      << loft_interval_text(fold->g_b, spans) << " meet at (" << fold->point.X() << ", "
      << fold->point.Y() << ", " << fold->point.Z()
      << "). Nothing is committed; the session is unchanged.";
    throw PysmeshError(s.str(), kFoldHint, {});
  }
  return commit(concat(survivors, result), hist, "thru_sections", result);
}

}  // namespace session
}  // namespace pysmesh
