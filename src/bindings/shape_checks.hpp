// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-03

// pySMESH binding — what a committed shape must be, beyond BRepCheck_Analyzer.
//
// BRepCheck_Analyzer tests the local consistency of the topology: each edge has its curves,
// each wire closes, each face's wires are oriented. It does not test the properties a mesher
// needs, and OCCT can hand back a shape that passes it and is still not the answer: an empty
// compound for two solids that overlap, a solid with an edge on one face only, a solid turned
// inside out (report defect_sweep_4.2.2.md §6). The statements here test those properties.
// Each one is sound: it refuses only on a witness that the answer is wrong — a point proven
// inside both operands of an empty common, an edge with one face — never on a heuristic.
//
// The session and the stateless module share them, so they are here and not in either one.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <Bnd_Box.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Pnt.hxx>
#include <gp_XYZ.hxx>

namespace pysmesh {
namespace shape_checks {

// The largest tolerance of any vertex, edge or face of the shapes. Precision::Confusion()
// when there is none.
double max_tolerance(const std::vector<TopoDS_Shape>& shapes);

// The solids of the shapes, in TopExp::MapShapes order.
std::vector<TopoDS_Shape> solids_of(const std::vector<TopoDS_Shape>& shapes);

// A point that lies inside a solid of `a` and inside a solid of `b`, each deeper than `tol`
// from the solid's boundary. Such a point proves that the interiors of the two groups
// overlap by more than `tol`, so their common cannot be empty.
//
// The candidates sit on the boundary of each solid of both groups — a parameter grid on
// every face, the midpoint of every edge on it and every vertex, at most 512 per solid —
// moved into the solid against the outward face normal by `depth`, which must exceed `tol`.
// A candidate counts only where BRepClass3d_SolidClassifier places it IN a solid of the other
// group at `tol`, and then IN its own solid at `tol`. The classifier is OCCT's point
// classifier, independent of the Boolean Component whose answer is under test. Finding no
// witness proves nothing.
std::optional<gp_Pnt> point_inside_both(const std::vector<TopoDS_Shape>& a,
                                        const std::vector<TopoDS_Shape>& b, double depth,
                                        double tol);

// A point that lies inside a solid of `a` and outside every solid of `b`, each farther than
// `tol` from the boundary. It proves that `a` minus `b` is not empty. The candidates are
// those of point_inside_both, taken from the solids of `a` only.
std::optional<gp_Pnt> point_inside_first_outside_second(const std::vector<TopoDS_Shape>& a,
                                                        const std::vector<TopoDS_Shape>& b,
                                                        double depth, double tol);

// "(x, y, z)" at 9 significant digits.
std::string point_text(const gp_Pnt& p);

// The edges of `shape` bordered by exactly one face, in TopExp::MapShapes order: the naked
// boundary of an open shell, and the leak in a solid that is not watertight. The face count
// is taken with repetition (TopExp::MapShapesAndAncestors), so a periodic seam, listed twice
// by its one face, counts two and is not free. A degenerated edge (a pole, an apex) bounds
// nothing and is skipped, and so is an edge with no face, which is not a surface boundary.
std::vector<TopoDS_Shape> free_boundary_edges(const TopoDS_Shape& shape);

// One edge in words: "from (x, y, z) to (x, y, z), length L".
std::string edge_text(const TopoDS_Shape& edge);

// The signed volume and the area of a tessellation of `solid` at the absolute deflection
// `deflection`.
//
// The volume is the sum of p0 . (p1 x p2) / 6 over the triangles, each oriented by its face,
// so a solid whose faces point out gets a positive volume. Every triangle lies within the
// deflection of its face, so the true volume differs from this one by at most
// `deflection` x the area. `complete` is false when a face got no triangles; the numbers
// then mean nothing.
//
// The tessellation is made on a copy of the solid's topology (BRepBuilderAPI_Copy), so no
// face of the caller's shape, of a session or of a retained snapshot, is left holding a
// triangulation. The copy shares the geometry, so it costs little.
struct TessellatedVolume {
  double volume = 0.0;
  double area = 0.0;
  bool complete = false;
};

TessellatedVolume tessellated_volume(const TopoDS_Shape& solid, double deflection);

// GProp's adaptive rule falls back to its fixed rule for any Eps above 1e-3
// (BRepGProp.hxx: "if Eps > 0.001 algorithm performs non-adaptive integration").
constexpr double kAdaptiveEpsCap = 1e-3;

// The volume a solid encloses, integrated precisely enough to trust its sign, and the least
// volume it must enclose to have an inside at all.
struct EnclosedVolume {
  double volume = 0.0;
  double area = 0.0;
  // eps x area, eps = Precision::Confusion(). A solid enclosing no more than this has, on
  // average, two sides closer than the distance at which OCCT treats two points as one.
  double tolerance = 0.0;
  // False when the volume and the area are a tessellation's (see enclosed_volume): then the
  // volume is certainly above the tolerance, but it is only within kSignDeflection x the
  // bounding-box diagonal x the area of the true one.
  bool exact = true;
};

// The deflection of the sign check in enclosed_volume, relative to the bounding-box diagonal.
constexpr double kSignDeflection = 1e-3;

// What a solid-making operation checks before it commits a solid: BRepCheck_Analyzer accepts
// a solid whose shell bounds its complement, so it cannot be the check.
//
// The volume is integrated with GProp's adaptive rule, because the fixed rule can get its
// sign wrong. It integrates each face about a point near the shape, so a face contributes up
// to D x its area / 3, D the bounding-box diagonal, and the contributions cancel down to the
// volume. A relative error e on them moves the volume by up to e x D x A / 3, while a sheet
// of thickness t encloses t x A / 2, so the sign can go when t < 2 e D / 3. The fixed rule's
// area error on one face of the production assembly was 26 %.
//
// The integral is taken in two stages, because only the verdict against the tolerance
// matters, and a tight precision costs: on the assembly's 436-face solid, 6.6 s at the
// precision below against 1.2 s at kAdaptiveEpsCap.
//
//   * First at kAdaptiveEpsCap. With e the larger of that and the error GProp reports, the
//     volume is settled when it clears the tolerance by more than e x D x A / 3. Over the
//     assembly's 117 solids, the real error at this stage was at most 1/60 of that bound, and
//     116 of them settle here.
//   * Otherwise at the precision the defeature check derives: e x D x A / 3 set to a tenth of
//     eps x A gives e = 0.3 x eps / D, so the verdict at the tolerance is the volume's own.
//
// Most solids enclose far more than the tolerance, and for them the integral is the whole
// cost: 1.2 s of a 1.4 s sew on the assembly's 436-face solid. So, unless `precise` is set,
// a sign check runs first (shape_checks::tessellated_volume): a tessellated copy at the
// absolute deflection d = kSignDeflection x D has a volume V_t within d x A_t of the true
// one, A_t its area. When V_t > (d + eps) x A_t the true volume is above eps x A_t, the
// verdict is settled, and the integral is skipped; `exact` is then false. Otherwise the
// integral runs as above, so every refusal reports the integrated volume.
EnclosedVolume enclosed_volume(const TopoDS_Shape& solid, bool precise = false);

// A solid of a shape whose matter is outside its boundary: the point at infinity classifies
// inside it, and the volume it encloses is negative beyond the tolerance.
struct InsideOutSolid {
  int ordinal = 0;  // 1-based, in TopExp::MapShapes(shape, TopAbs_SOLID) order
  double volume = 0.0;
};

// Every inside-out solid of `shape`. The classifier decides alone for a solid it finds
// outside of the point at infinity, which is every solid of a valid model, so a valid model
// pays one classification per solid. A solid it finds inside is integrated
// (enclosed_volume), and counts only when the volume agrees.
std::vector<InsideOutSolid> inside_out_solids(const TopoDS_Shape& shape);

// `shape` with each named solid replaced by a solid whose shells are reversed, so its matter
// is inside. The faces are kept; only the solid and its shells are new.
TopoDS_Shape reverse_solids(const TopoDS_Shape& shape, const std::vector<InsideOutSolid>& solids);

// The refusal of an import that holds inside-out solids, and the line that reports one
// reversed solid. `op` names the operation, e.g. "Session.add_brep".
std::string inside_out_refusal(const std::string& op, const std::vector<InsideOutSolid>& solids);
std::string inside_out_reversed(const InsideOutSolid& solid);

// Why the solids of `shape` are refused for interfering with themselves, or an empty string.
//
// Two tests run on each solid. BOPAlgo_CheckerSI, the Boolean Component's own
// intersection of the pairs of sub-shapes, up to edges against faces; and each free-form
// face intersected with itself (IntTools_FaceFace), where a crossing counts only away from
// the face's own edges, so a face that closes through a sharp seam is not reported. A pair
// that intersects where the topology does not join it, or a face that crosses itself, is a
// boundary that passes through itself: the solid has no consistent inside, although
// BRepCheck_Analyzer accepts it. Each finding names the sub-shapes by kind and 1-based
// ordinal in `shape` (TopExp::MapShapes order per kind), with their geometry. `op` names
// the operation.
std::string self_interference_refusal(const std::string& op, const TopoDS_Shape& shape);

// One status BRepCheck_Analyzer reports on a sub-shape of a shape it rejects. `context` is
// the shape the status holds in (an edge's status on one face), or null.
struct CheckFinding {
  TopoDS_Shape shape;
  std::string status;  // the BRepCheck_Status name, e.g. "BRepCheck_NotClosed"
  TopoDS_Shape context;
};

// Every status other than BRepCheck_NoError that BRepCheck_Analyzer reports on `shape` and
// its sub-shapes, solids first and vertices last, each kind in TopExp::MapShapes order, each
// (sub-shape, status, context) once.
std::vector<CheckFinding> check_findings(const TopoDS_Shape& shape);

// The faces of `shape` that BRepCheck_Analyzer rejects on their own, geometric controls on.
std::vector<TopoDS_Shape> invalid_faces(const TopoDS_Shape& shape);

// The kind of a shape in capitals: "SOLID", "SHELL", "FACE", "WIRE", "EDGE", "VERTEX".
const char* kind_text(const TopoDS_Shape& shape);

// The import policy for inside-out solids, from the caller's string: true for "reverse",
// false for "raise". Anything else is refused, naming `op`.
bool reverse_inside_out(const std::string& op, const std::string& policy);

// One shape's measure by its own kind (volume of a solid, area of a face, length of an edge,
// 0 for a vertex), its centre of mass, and the relative error the rule reports reaching.
//
// A solid and a face go to GProp's adaptive rule, which refines each face until two steps
// agree to `precision` relative and returns its estimate of the relative error reached
// (BRepGProp.hxx). An edge goes to an adaptive Gauss-Kronrod rule along its curve, because
// GProp has no adaptive rule for a curve. A vertex, and an edge with no curve, is a point.
struct Measure {
  double mass = 0.0;
  gp_XYZ centroid;
  double error = 0.0;
};

// The default relative precision of every measure the library reports (report D3). GProp's
// fixed rule read a wing lofted through one-edge sections 20 % low; at 1e-6 the adaptive
// rule reads it within 3.6e-10 of the integral of its section area along the span.
constexpr double kDefaultMassPrecision = 1e-6;

Measure measure(const TopoDS_Shape& s, double precision);

// The measures of many shapes, integrated in parallel, one shape per task.
std::vector<Measure> measures(const std::vector<TopoDS_Shape>& shapes, double precision);

// The box of a shape's geometry (reports D1, D2): BRepBndLib::AddOptimal without the
// triangulation and without the shape's tolerance, so a line's box is its two end points.
// A B-spline or Bezier curve or surface is bounded by its own points, found by
// optimisation, and OCCT pads that box by Precision::Confusion() = 1e-7
// (GeomBndLib_SplineHelpers.pxx, CurveBoxOptimal: aBox.Enlarge(anEps)). BRepBndLib::Add
// bounded the control polygon, 2.72 mm outside a NACA spline of 2 m chord, and padded
// every box by the shape tolerance.
Bnd_Box exact_box(const TopoDS_Shape& s);

// The boxes of many shapes, computed in parallel, one shape per task.
std::vector<Bnd_Box> exact_boxes(const std::vector<TopoDS_Shape>& shapes);

// How far a wire is from planar: the spread of its points across the plane that fits them
// best (BRepLib_FindSurface, OnlyPlane: a least-squares plane), as the largest minus the
// smallest signed distance of 65 points per edge. A wire bowed by w sin(pi x) over a chord
// x in [0, 1] spreads w. Empty when the points define no plane (they lie on one line).
std::optional<double> out_of_plane_spread(const TopoDS_Shape& wire);

}  // namespace shape_checks
}  // namespace pysmesh
