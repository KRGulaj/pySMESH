// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH binding — Session: the shape tests behind the selection filters.
//
// Session.select (src/pysmesh/session/_select.py) combines filters whose behaviour follows
// the filters of SALOME Shaper's FiltersPlugin. Four of them are composed in Python: the
// three size filters compare the measures of size_measures, below, and RelativeToSolid is
// entities_on with the solid as the region. The twelve here need OCCT
// predicates those queries do not carry: the planarity test of GeomLib_IsPlanarSurface, the
// same-geometry tests, the order of a wire's edges, the orientation of a face, and the G1
// test of two faces at an edge's middle. Each is written here against OCCT, after the
// behaviour of the Shaper filter it is named after; no Shaper code is compiled into
// pySMESH, and every place where the behaviour differs is stated where it happens.
//
// filter_mask answers one filter for every live id of one kind, in ascending id order. An id
// that denotes several shapes (a split id) passes only when every shape it denotes passes.
// Read-only: no id is issued and no shape of the model is written.
//
// Shaper's filters look for neighbours within the result body that holds the shape. A
// session keeps every solid as a body of its own, and two of them share sub-shapes after a
// fragment, where Shaper's result is one compound. So the neighbours here are found over the
// whole model: bodies that share no sub-shape have no neighbour in common anyway.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include <deque>

#include <BRepBuilderAPI_FindPlane.hxx>
#include <GCPnts_UniformAbscissa.hxx>
#include <GeomAPI_ExtremaCurveCurve.hxx>
#include <GeomAdaptor_Curve.hxx>
#include <GeomLib_IsPlanarSurface.hxx>
#include <Geom_Conic.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_ElementarySurface.hxx>
#include <Geom_Hyperbola.hxx>
#include <Geom_Line.hxx>
#include <Geom_OffsetSurface.hxx>
#include <Geom_Parabola.hxx>
#include <Geom_Plane.hxx>
#include <Geom_RectangularTrimmedSurface.hxx>
#include <Geom_SurfaceOfLinearExtrusion.hxx>
#include <Geom_SurfaceOfRevolution.hxx>
#include <Geom_SweptSurface.hxx>
#include <IntPatch_ImpImpIntersection.hxx>
#include <BRepTopAdaptor_TopolTool.hxx>
#include <GeomAdaptor_Surface.hxx>
#include <LocalAnalysis_SurfaceContinuity.hxx>
#include <OSD_Parallel.hxx>
#include <ShapeAnalysis_Surface.hxx>
#include <Standard_Failure.hxx>

namespace pysmesh {
namespace session {

namespace {

// Shaper's linear tolerance in these tests: Precision::Confusion().
constexpr double kLinear = 1.0e-7;

constexpr double kPi = 3.14159265358979323846;

// The surface of a face with every rectangular trim taken off.
Handle(Geom_Surface) base_surface(const TopoDS_Face& face) {
  Handle(Geom_Surface) s = BRep_Tool::Surface(face);
  while (!s.IsNull() && s->IsKind(STANDARD_TYPE(Geom_RectangularTrimmedSurface))) {
    s = Handle(Geom_RectangularTrimmedSurface)::DownCast(s)->BasisSurface();
  }
  return s;
}

// The 3-D curve of an edge with every trim taken off; null for a degenerate edge.
Handle(Geom_Curve) base_curve(const TopoDS_Edge& edge) {
  double first = 0.0;
  double last = 0.0;
  Handle(Geom_Curve) c = BRep_Tool::Curve(edge, first, last);
  while (!c.IsNull() && c->IsKind(STANDARD_TYPE(Geom_TrimmedCurve))) {
    c = Handle(Geom_TrimmedCurve)::DownCast(c)->BasisCurve();
  }
  return c;
}

// Whether a face lies in a plane: GeomLib_IsPlanarSurface on the untrimmed surface, as
// Shaper's GeomAPI_Shape::isPlanar tests it.
bool face_is_planar(const TopoDS_Face& face) {
  const Handle(Geom_Surface) s = base_surface(face);
  return !s.IsNull() && GeomLib_IsPlanarSurface(s).IsPlanar();
}

// The plane of a planar face, its normal reversed for a REVERSED face, as Shaper's
// GeomAPI_Face::getPlane builds it: GeomLib_IsPlanarSurface on the surface, or the moved
// plane of an offset surface over a plane.
std::optional<gp_Pln> face_plane(const TopoDS_Face& face) {
  const Handle(Geom_Surface) s = BRep_Tool::Surface(face);
  if (s.IsNull()) {
    return std::nullopt;
  }
  gp_Pln pln;
  GeomLib_IsPlanarSurface test(s);
  if (test.IsPlanar()) {
    pln = test.Plan();
  } else if (s->IsKind(STANDARD_TYPE(Geom_OffsetSurface))) {
    const Handle(Geom_OffsetSurface) offset = Handle(Geom_OffsetSurface)::DownCast(s);
    const Handle(Geom_Surface) basis = offset->BasisSurface();
    if (!basis->IsKind(STANDARD_TYPE(Geom_Plane))) {
      return std::nullopt;
    }
    pln = Handle(Geom_Plane)::DownCast(basis)->Pln();
    pln.Translate(gp_Vec(pln.Axis().Direction().XYZ() * offset->Offset()));
  } else {
    return std::nullopt;
  }
  double a = 0.0, b = 0.0, c = 0.0, d = 0.0;
  pln.Coefficients(a, b, c, d);
  if (face.Orientation() == TopAbs_REVERSED) {
    a = -a;
    b = -b;
    c = -c;
    d = -d;
  }
  return gp_Pln(a, b, c, d);
}

// The cylinder of a face on a Geom_CylindricalSurface.
std::optional<gp_Cylinder> face_cylinder(const TopoDS_Face& face) {
  const Handle(Geom_Surface) s = BRep_Tool::Surface(face);
  if (s.IsNull() || !s->IsKind(STANDARD_TYPE(Geom_CylindricalSurface))) {
    return std::nullopt;
  }
  return Handle(Geom_CylindricalSurface)::DownCast(s)->Cylinder();
}

bool edge_is_line(const TopoDS_Edge& edge) {
  const Handle(Geom_Curve) c = base_curve(edge);
  return !c.IsNull() && c->IsKind(STANDARD_TYPE(Geom_Line));
}

// The points of an edge's curve at its first and last parameter.
std::optional<std::pair<gp_Pnt, gp_Pnt>> edge_ends(const TopoDS_Edge& edge) {
  double first = 0.0;
  double last = 0.0;
  const Handle(Geom_Curve) c = BRep_Tool::Curve(edge, first, last);
  if (c.IsNull()) {
    return std::nullopt;
  }
  return std::make_pair(c->Value(first), c->Value(last));
}

// The line through a straight edge's two ends.
std::optional<gp_Lin> edge_line(const TopoDS_Edge& edge) {
  if (!edge_is_line(edge)) {
    return std::nullopt;
  }
  const auto ends = edge_ends(edge);
  if (!ends || ends->first.Distance(ends->second) <= gp::Resolution()) {
    return std::nullopt;
  }
  return gp_Lin(ends->first, gp_Dir(gp_Vec(ends->first, ends->second)));
}

// Whether an edge lies in some plane: BRepBuilderAPI_FindPlane finds one, or the edge is a
// line or a conic (Shaper's GeomAPI_Shape::isPlanar for an edge).
bool edge_is_planar(const TopoDS_Edge& edge) {
  if (BRepBuilderAPI_FindPlane(edge).Found()) {
    return true;
  }
  const Handle(Geom_Curve) c = base_curve(edge);
  return !c.IsNull() && (c->IsKind(STANDARD_TYPE(Geom_Line)) ||
                         c->IsKind(STANDARD_TYPE(Geom_Conic)));
}

// Whether a planar edge lies in a plane, as Shaper's GeomAPI_Edge::isInPlane tests it: a
// line by its two ends, a circle by its centre and axis, any other curve by its two ends
// and its parameter middle.
bool edge_in_plane(const TopoDS_Edge& edge, const gp_Pln& plane) {
  double first = 0.0;
  double last = 0.0;
  const Handle(Geom_Curve) c = BRep_Tool::Curve(edge, first, last);
  if (c.IsNull()) {
    return false;
  }
  const double tol2 = Precision::SquareConfusion();
  if (c->IsKind(STANDARD_TYPE(Geom_Line))) {
    return plane.SquareDistance(c->Value(first)) < tol2 &&
           plane.SquareDistance(c->Value(last)) < tol2;
  }
  const Handle(Geom_Curve) basis = base_curve(edge);
  if (!basis.IsNull() && basis->IsKind(STANDARD_TYPE(Geom_Circle))) {
    const Handle(Geom_Circle) circle = Handle(Geom_Circle)::DownCast(basis);
    const double dot = plane.Axis().Direction().Dot(circle->Axis().Direction());
    return plane.SquareDistance(circle->Location()) < tol2 &&
           std::abs(std::abs(dot) - 1.0) < Precision::Confusion();
  }
  return plane.SquareDistance(c->Value(first)) < tol2 &&
         plane.SquareDistance(c->Value(0.5 * (first + last))) < tol2 &&
         plane.SquareDistance(c->Value(last)) < tol2;
}

// The middle of an edge by arc length: the second of three uniformly spaced points
// (GCPnts_UniformAbscissa), as Shaper's GeomAPI_Edge::middlePoint takes it.
std::optional<gp_Pnt> edge_middle(const TopoDS_Edge& edge) {
  double first = 0.0;
  double last = 0.0;
  const Handle(Geom_Curve) c = BRep_Tool::Curve(edge, first, last);
  if (c.IsNull()) {
    return std::nullopt;
  }
  GeomAdaptor_Curve adaptor(c, first, last);
  GCPnts_UniformAbscissa points(adaptor, 3);
  if (!points.IsDone()) {
    return std::nullopt;
  }
  return adaptor.Value(points.Parameter(2));
}

// The middle point of a shape, after Shaper's GeomAPI_Shape::middlePoint: a vertex's point,
// an edge's arc-length middle, a face's surface point at the middle of its parameter box,
// a solid's centre of mass. For a face Shaper takes the middle of an optimal box of the
// face's pcurves; BRepTools::UVBounds is used here, which can be larger on curved pcurves.
std::optional<gp_Pnt> middle_point(const TopoDS_Shape& shape) {
  switch (shape.ShapeType()) {
    case TopAbs_VERTEX:
      return BRep_Tool::Pnt(TopoDS::Vertex(shape));
    case TopAbs_EDGE:
      return edge_middle(TopoDS::Edge(shape));
    case TopAbs_FACE: {
      const TopoDS_Face& face = TopoDS::Face(shape);
      const Handle(Geom_Surface) s = BRep_Tool::Surface(face);
      if (s.IsNull()) {
        return std::nullopt;
      }
      double u0 = 0.0, u1 = 0.0, v0 = 0.0, v1 = 0.0;
      BRepTools::UVBounds(face, u0, u1, v0, v1);
      return s->Value(0.5 * (u0 + u1), 0.5 * (v0 + v1));
    }
    case TopAbs_SOLID: {
      GProp_GProps props;
      BRepGProp::VolumeProperties(shape, props, 1.0e-4);
      return props.CentreOfMass();
    }
    default:
      return std::nullopt;
  }
}

// The G1 test of two faces at a point, as Shaper's GeomAlgoAPI_ShapeTools::isContinuousFaces
// makes it: the point projected on both surfaces, then LocalAnalysis_SurfaceContinuity of
// order G1 with 0.001 for its zero, C0, C1 and C2 thresholds and the angle for G1. OCCT
// folds the angle between the two normals into [0, pi/2], so the test does not depend on
// the faces' orientations.
enum class Continuity { kG1, kNotG1, kError };

Continuity continuity(const TopoDS_Face& f1, const TopoDS_Face& f2, const gp_Pnt& point,
                      double angle_rad) {
  try {
    const Handle(Geom_Surface) s1 = BRep_Tool::Surface(f1);
    const Handle(Geom_Surface) s2 = BRep_Tool::Surface(f2);
    if (s1.IsNull() || s2.IsNull()) {
      return Continuity::kError;
    }
    const gp_Pnt2d uv1 = ShapeAnalysis_Surface(s1).ValueOfUV(point, Precision::Confusion());
    const gp_Pnt2d uv2 = ShapeAnalysis_Surface(s2).ValueOfUV(point, Precision::Confusion());
    LocalAnalysis_SurfaceContinuity test(s1, uv1.X(), uv1.Y(), s2, uv2.X(), uv2.Y(),
                                         GeomAbs_G1, 0.001, 0.001, 0.001, 0.001, angle_rad);
    return test.IsG1() ? Continuity::kG1 : Continuity::kNotG1;
  } catch (const Standard_Failure&) {
    return Continuity::kError;
  }
}

// Whether two edges lie on one curve, as Shaper's GeomAPI_Edge::isSameGeometry tests it: the
// same edge, or untrimmed curves that GeomAPI_ExtremaCurveCurve finds parallel at a
// distance below 1e-7.
bool same_curve(const TopoDS_Edge& a, const TopoDS_Edge& b) {
  if (a.IsSame(b)) {
    return true;
  }
  const Handle(Geom_Curve) ca = base_curve(a);
  const Handle(Geom_Curve) cb = base_curve(b);
  if (ca.IsNull() || cb.IsNull()) {
    return false;
  }
  try {
    GeomAPI_ExtremaCurveCurve extrema(ca, cb);
    return extrema.Extrema().IsParallel() && extrema.TotalLowerDistance() < kLinear;
  } catch (const Standard_Failure&) {
    return false;
  }
}

// Whether two coincident cylinders, after Shaper's GeomAPI_Cylinder::isCoincident.
bool same_cylinder(const gp_Cylinder& a, const gp_Cylinder& b) {
  if (std::abs(a.Radius() - b.Radius()) >= kLinear) {
    return false;
  }
  const gp_Dir& da = a.Position().Direction();
  if (!da.IsParallel(b.Position().Direction(), Precision::Angular())) {
    return false;
  }
  const gp_Vec between(a.Location(), b.Location());
  return between.SquareMagnitude() < kLinear * kLinear ||
         between.IsParallel(gp_Vec(da), Precision::Angular());
}

// Whether two planes coincide, after Shaper's GeomAPI_Pln::isCoincident: the first holds the
// second's location within 1e-7, and the axes are parallel within 1e-7 rad.
bool same_plane(const gp_Pln& a, const gp_Pln& b) {
  return a.Contains(b.Location(), kLinear) && a.Axis().IsParallel(b.Axis(), kLinear);
}

// Whether two faces lie on one surface, after Shaper's GeomAPI_Face::isSameGeometry.
bool same_surface(const TopoDS_Face& a, const TopoDS_Face& b) {
  if (a.IsSame(b)) {
    return true;
  }
  if (face_is_planar(a) && face_is_planar(b)) {
    const std::optional<gp_Pln> pa = face_plane(a);
    const std::optional<gp_Pln> pb = face_plane(b);
    return pa && pb && same_plane(*pa, *pb);
  }
  const Handle(Geom_Surface) sa = base_surface(a);
  const Handle(Geom_Surface) sb = base_surface(b);
  if (sa.IsNull() || sb.IsNull()) {
    return false;
  }
  if (sa == sb) {
    return true;
  }
  if (sa->IsKind(STANDARD_TYPE(Geom_ElementarySurface)) &&
      sb->IsKind(STANDARD_TYPE(Geom_ElementarySurface))) {
    try {
      Handle(GeomAdaptor_Surface) ga = new GeomAdaptor_Surface(sa);
      Handle(GeomAdaptor_Surface) gb = new GeomAdaptor_Surface(sb);
      Handle(BRepTopAdaptor_TopolTool) ta = new BRepTopAdaptor_TopolTool();
      Handle(BRepTopAdaptor_TopolTool) tb = new BRepTopAdaptor_TopolTool();
      IntPatch_ImpImpIntersection inter(ga, ta, gb, tb, Precision::Confusion(),
                                        Precision::Confusion());
      if (!inter.IsDone() || inter.IsEmpty()) {
        return false;
      }
      return inter.TangentFaces();
    } catch (const Standard_Failure&) {
      return false;
    }
  }
  const auto cylinder_like = [](const Handle(Geom_Surface)& s) {
    return s->IsKind(STANDARD_TYPE(Geom_CylindricalSurface)) ||
           s->IsKind(STANDARD_TYPE(Geom_SweptSurface));
  };
  if (cylinder_like(sa) && cylinder_like(sb)) {
    const std::optional<gp_Cylinder> ca = face_cylinder(a);
    const std::optional<gp_Cylinder> cb = face_cylinder(b);
    if (ca && cb) {
      return same_cylinder(*ca, *cb);
    }
    const bool both_extrusions = sa->IsKind(STANDARD_TYPE(Geom_SurfaceOfLinearExtrusion)) &&
                                 sb->IsKind(STANDARD_TYPE(Geom_SurfaceOfLinearExtrusion));
    const bool both_revolutions = sa->IsKind(STANDARD_TYPE(Geom_SurfaceOfRevolution)) &&
                                  sb->IsKind(STANDARD_TYPE(Geom_SurfaceOfRevolution));
    if (both_extrusions || both_revolutions) {
      const Handle(Geom_SweptSurface) wa = Handle(Geom_SweptSurface)::DownCast(sa);
      const Handle(Geom_SweptSurface) wb = Handle(Geom_SweptSurface)::DownCast(sb);
      if (wa->Direction().IsParallel(wb->Direction(), Precision::Angular())) {
        try {
          GeomAPI_ExtremaCurveCurve extrema(wa->BasisCurve(), wb->BasisCurve());
          return extrema.Extrema().IsParallel() && extrema.TotalLowerDistance() < kLinear;
        } catch (const Standard_Failure&) {
          return false;
        }
      }
    }
  }
  return false;
}

using Ancestors = NCollection_IndexedDataMap<TopoDS_Shape, NCollection_List<TopoDS_Shape>,
                                             TopTools_ShapeMapHasher>;

// The faces of the model that hold each of its edges (or vertices), each face once.
Ancestors faces_by(const TopoDS_Shape& root, TopAbs_ShapeEnum kind) {
  Ancestors out;
  TopExp::MapShapesAndUniqueAncestors(root, kind, TopAbs_FACE, out);
  return out;
}

const NCollection_List<TopoDS_Shape>& faces_of_key(const Ancestors& map,
                                                   const TopoDS_Shape& key) {
  static const NCollection_List<TopoDS_Shape> kNone;
  const int i = map.FindIndex(key);
  return i == 0 ? kNone : map.FindFromIndex(i);
}

// The faces continuous with `start` across its edges, `start` included: Shaper's
// ContinuousFaces closure.
void continuous_closure(const TopoDS_Face& start, const Ancestors& edge_faces,
                        double angle_rad, ShapeSet& out) {
  std::deque<TopoDS_Face> todo;
  if (out.Add(start) != 0) {
    todo.push_back(start);
  }
  while (!todo.empty()) {
    const TopoDS_Face face = todo.front();
    todo.pop_front();
    for (TopExp_Explorer e(face, TopAbs_EDGE); e.More(); e.Next()) {
      const TopoDS_Edge& edge = TopoDS::Edge(e.Current());
      const NCollection_List<TopoDS_Shape>& next = faces_of_key(edge_faces, edge);
      if (next.IsEmpty()) {
        continue;
      }
      const std::optional<gp_Pnt> middle = edge_middle(edge);
      for (const TopoDS_Shape& g : next) {
        if (out.Contains(g) || !middle) {
          continue;
        }
        if (continuity(face, TopoDS::Face(g), *middle, angle_rad) == Continuity::kG1) {
          out.Add(g);
          todo.push_back(TopoDS::Face(g));
        }
      }
    }
  }
}

// The edge two places on from `edge` in the one wire of a face with four edges, in the
// order BRepTools_WireExplorer gives them; null when the face is not such a quadrangle.
TopoDS_Shape opposite_in_quad(const TopoDS_Shape& edge, const TopoDS_Shape& face) {
  TopExp_Explorer wires(face, TopAbs_WIRE);
  if (!wires.More()) {
    return TopoDS_Shape();
  }
  const TopoDS_Wire wire = TopoDS::Wire(wires.Current());
  wires.Next();
  if (wires.More()) {
    return TopoDS_Shape();  // a face with a hole is not a quadrangle
  }
  ShapeSet distinct;
  TopExp::MapShapes(wire, TopAbs_EDGE, distinct);
  if (distinct.Extent() != 4) {
    return TopoDS_Shape();
  }
  std::vector<TopoDS_Shape> order;
  for (BRepTools_WireExplorer w(wire); w.More(); w.Next()) {
    order.push_back(w.Current());
  }
  if (order.size() != 4) {
    return TopoDS_Shape();
  }
  for (std::size_t i = 0; i < 4; ++i) {
    if (order[i].IsSame(edge)) {
      return order[(i + 2) % 4];
    }
  }
  return TopoDS_Shape();
}

std::string filter_text(const std::string& filter) { return "the filter " + filter; }

}  // namespace

py::array_t<bool> Session::filter_mask(const std::string& kind, const std::string& filter,
                                       const std::vector<EntityId>& entity_ids,
                                       const std::vector<double>& numbers, bool flag,
                                       const py::object& progress,
                                       const py::object& cancel) const {
  const char* op = "select";
  const ProgressHooks hooks = hooks_of(op, progress, cancel);
  const TopAbs_ShapeEnum k = kind_from_name(kind);
  const TopoDS_Shape root = state_.root;
  const std::vector<EntityId> ids = ids_of_kind(k);

  // Every shape of the root, with the orientation the root gives it, by kind.
  const auto root_map = [&root](TopAbs_ShapeEnum of) {
    ShapeSet m;
    TopExp::MapShapes(root, of, m);
    return m;
  };
  const ShapeSet candidates = root_map(k);
  const auto in_root = [](const ShapeSet& m, const TopoDS_Shape& s) {
    const int i = m.FindIndex(s);
    if (i == 0) {
      throw PysmeshError("Session.select: an entity is alive in the registry but is not in "
                         "the session root. The registry and the root have diverged; this "
                         "is a bug, not a caller error.");
    }
    return m.FindKey(i);
  };

  // The shapes of the filter's own entities, checked: alive, of an allowed kind, and, where
  // the filter needs it, planar or straight.
  const auto argument = [&](EntityId id, std::initializer_list<TopAbs_ShapeEnum> allowed,
                            const char* what) {
    const EntityRecord& rec = require_alive(op, id);
    bool ok = false;
    for (TopAbs_ShapeEnum a : allowed) {
      ok = ok || rec.kind == a;
    }
    if (!ok) {
      throw PysmeshError("Session.select: " + filter_text(filter) + " names entity " +
                         std::to_string(id) + ", a " + kind_name(rec.kind) + "; it takes " +
                         what + ".");
    }
    const ShapeSet m = root_map(rec.kind);
    std::vector<TopoDS_Shape> out;
    for (const TopoDS_Shape& s : rec.shapes) {
      out.push_back(in_root(m, s));
    }
    return out;
  };
  const auto one_shape = [&](EntityId id, std::initializer_list<TopAbs_ShapeEnum> allowed,
                             const char* what) {
    const std::vector<TopoDS_Shape> shapes = argument(id, allowed, what);
    if (shapes.size() != 1) {
      throw PysmeshError("Session.select: " + filter_text(filter) + " names entity " +
                         std::to_string(id) + ", which denotes " +
                         std::to_string(shapes.size()) +
                         " shapes (it was split); it takes one shape.");
    }
    return shapes.front();
  };
  const auto plane_of = [&](EntityId id) {
    const TopoDS_Shape s = one_shape(id, {TopAbs_FACE}, "a planar face");
    const std::optional<gp_Pln> p = face_plane(TopoDS::Face(s));
    if (!p) {
      throw PysmeshError("Session.select: " + filter_text(filter) + " names face " +
                         std::to_string(id) + ", which is not planar; it takes a planar "
                         "face.");
    }
    return *p;
  };
  const auto line_of = [&](EntityId id) {
    const TopoDS_Shape s = one_shape(id, {TopAbs_EDGE}, "a straight edge");
    const std::optional<gp_Lin> l = edge_line(TopoDS::Edge(s));
    if (!l) {
      throw PysmeshError("Session.select: " + filter_text(filter) + " names edge " +
                         std::to_string(id) + ", which is not a straight segment; it takes "
                         "a straight edge.");
    }
    return std::make_pair(s, *l);
  };
  const auto angle_of = [&]() {
    if (numbers.size() != 1) {
      throw PysmeshError("Session.select: " + filter_text(filter) + " takes one angle.");
    }
    finite_arg(op, "angle_deg", numbers[0]);
    if (!(numbers[0] >= 0.0)) {
      throw PysmeshError("Session.select: " + filter_text(filter) + " has the angle " +
                         number_text(numbers[0]) + " degrees, below zero.");
    }
    return numbers[0];
  };
  if (entity_ids.empty() &&
      (filter == "BelongsTo" || filter == "ContinuousFaces" || filter == "OnGeometry" ||
       filter == "OnLine" || filter == "OnPlane")) {
    throw PysmeshError("Session.select: " + filter_text(filter) + " names no entity.");
  }

  // One progress range for the filter, in two steps: the branch below prepares the filter's
  // set or test, then every live id is tested. FeatureEdges, the one long preparation,
  // takes one step per edge of the model.
  ProgressDriver driver(op, hooks);
  Message_ProgressScope scope(driver.range(), "select", 2.0);
  Message_ProgressRange prepare = scope.Next();

  // The per-shape test of the filter. Each branch prepares what the test needs first.
  std::function<bool(const TopoDS_Shape&)> test;
  ShapeSet accepted;  // for the filters that compute their set up front

  if (filter == "BelongsTo") {
    for (EntityId id : entity_ids) {
      for (const TopoDS_Shape& s : argument(id, {TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE,
                                                 TopAbs_VERTEX},
                                            "an entity of any kind")) {
        TopExp::MapShapes(s, k, accepted);
      }
    }
  } else if (filter == "ContinuousFaces") {
    const double angle = angle_of() * kPi / 180.0;
    std::vector<TopoDS_Shape> starts;
    for (EntityId id : entity_ids) {
      for (const TopoDS_Shape& s : argument(id, {TopAbs_FACE}, "faces")) {
        starts.push_back(s);
      }
    }
    py::gil_scoped_release release;
    const Ancestors edge_faces = faces_by(root, TopAbs_EDGE);
    for (const TopoDS_Shape& f : starts) {
      continuous_closure(TopoDS::Face(f), edge_faces, angle, accepted);
    }
  } else if (filter == "ExternalFaces") {
    // The number of solids of the model that hold each face.
    ShapeKeyed<int> solid_count;
    Ancestors solids;
    TopExp::MapShapesAndUniqueAncestors(root, TopAbs_FACE, TopAbs_SOLID, solids);
    for (int i = 1; i <= solids.Extent(); ++i) {
      solid_count[solids.FindKey(i)] = solids.FindFromIndex(i).Extent();
    }
    test = [solid_count](const TopoDS_Shape& face) {
      if (face.Orientation() == TopAbs_INTERNAL) {
        return false;
      }
      const auto it = solid_count.find(face);
      return it == solid_count.end() || it->second <= 1;
    };
  } else if (filter == "FeatureEdges") {
    const double angle_deg = angle_of();
    const double angle = angle_deg * kPi / 180.0;
    py::gil_scoped_release release;
    {
      const Ancestors edge_faces = faces_by(root, TopAbs_EDGE);
      Message_ProgressScope edges(prepare, "FeatureEdges",
                                  static_cast<double>(std::max(edge_faces.Extent(), 1)));
      for (int i = 1; i <= edge_faces.Extent() && edges.More(); ++i, edges.Next()) {
        const TopoDS_Edge& edge = TopoDS::Edge(edge_faces.FindKey(i));
        std::vector<TopoDS_Face> faces;
        for (const TopoDS_Shape& f : edge_faces.FindFromIndex(i)) {
          faces.push_back(TopoDS::Face(f));
        }
        const std::optional<gp_Pnt> middle = edge_middle(edge);
        bool feature = false;
        for (std::size_t a = 0; a < faces.size() && !feature; ++a) {
          for (std::size_t b = a + 1; b < faces.size() && !feature; ++b) {
            if (angle_deg < Precision::Confusion()) {
              feature = true;
            } else if (middle &&
                       continuity(faces[a], faces[b], *middle, angle) == Continuity::kNotG1) {
              feature = true;
            }
          }
        }
        if (feature) {
          accepted.Add(edge);
        }
      }
    }
  } else if (filter == "HorizontalFace") {
    test = [](const TopoDS_Shape& face) {
      const TopoDS_Face& f = TopoDS::Face(face);
      if (!face_is_planar(f)) {
        return false;
      }
      const std::optional<gp_Pln> p = face_plane(f);
      return p && p->Axis().Direction().IsParallel(gp::DZ(), kLinear);
    };
  } else if (filter == "VerticalFace") {
    test = [](const TopoDS_Shape& face) {
      const TopoDS_Face& f = TopoDS::Face(face);
      if (const std::optional<gp_Pln> p = face_plane(f)) {
        return std::abs(p->Axis().Direction().Z()) <= kLinear;
      }
      if (const std::optional<gp_Cylinder> c = face_cylinder(f)) {
        return std::abs(std::abs(c->Position().Direction().Z()) - 1.0) <= kLinear;
      }
      return false;
    };
  } else if (filter == "OnGeometry") {
    std::vector<TopoDS_Shape> refs;
    for (EntityId id : entity_ids) {
      for (const TopoDS_Shape& s : argument(id, {TopAbs_EDGE, TopAbs_FACE}, "edges or faces")) {
        refs.push_back(s);
      }
    }
    test = [refs](const TopoDS_Shape& s) {
      for (const TopoDS_Shape& r : refs) {
        if (r.ShapeType() != s.ShapeType()) {
          continue;
        }
        if (s.ShapeType() == TopAbs_EDGE ? same_curve(TopoDS::Edge(r), TopoDS::Edge(s))
                                         : same_surface(TopoDS::Face(r), TopoDS::Face(s))) {
          return true;
        }
      }
      return false;
    };
  } else if (filter == "OnLine") {
    std::vector<gp_Lin> lines;
    for (EntityId id : entity_ids) {
      lines.push_back(line_of(id).second);
    }
    test = [lines](const TopoDS_Shape& s) {
      if (s.ShapeType() == TopAbs_VERTEX) {
        const gp_Pnt p = BRep_Tool::Pnt(TopoDS::Vertex(s));
        for (const gp_Lin& l : lines) {
          if (l.Contains(p, kLinear)) {
            return true;
          }
        }
        return false;
      }
      const TopoDS_Edge& e = TopoDS::Edge(s);
      const auto ends = edge_ends(e);
      if (!edge_is_line(e) || !ends) {
        return false;
      }
      for (const gp_Lin& l : lines) {
        if (l.Contains(ends->first, kLinear) && l.Contains(ends->second, kLinear)) {
          return true;
        }
      }
      return false;
    };
  } else if (filter == "OnPlane") {
    std::vector<gp_Pln> planes;
    for (EntityId id : entity_ids) {
      planes.push_back(plane_of(id));
    }
    test = [planes](const TopoDS_Shape& s) {
      if (s.ShapeType() == TopAbs_VERTEX) {
        const gp_Pnt p = BRep_Tool::Pnt(TopoDS::Vertex(s));
        for (const gp_Pln& pl : planes) {
          if (pl.Distance(p) < kLinear) {
            return true;
          }
        }
        return false;
      }
      if (s.ShapeType() == TopAbs_EDGE) {
        const TopoDS_Edge& e = TopoDS::Edge(s);
        if (edge_is_line(e)) {
          const auto ends = edge_ends(e);
          for (const gp_Pln& pl : planes) {
            if (ends && pl.Distance(ends->first) < kLinear &&
                pl.Distance(ends->second) < kLinear) {
              return true;
            }
          }
          return false;
        }
        if (edge_is_planar(e)) {
          for (const gp_Pln& pl : planes) {
            if (edge_in_plane(e, pl)) {
              return true;
            }
          }
        }
        return false;
      }
      const TopoDS_Face& f = TopoDS::Face(s);
      if (!face_is_planar(f)) {
        return false;
      }
      const std::optional<gp_Pln> own = face_plane(f);
      for (const gp_Pln& pl : planes) {
        if (own && same_plane(pl, *own)) {
          return true;
        }
      }
      return false;
    };
  } else if (filter == "OnPlaneSide") {
    if (entity_ids.size() != 1) {
      throw PysmeshError("Session.select: " + filter_text(filter) + " takes one face.");
    }
    const gp_Pln plane = plane_of(entity_ids.front());
    test = [plane](const TopoDS_Shape& s) {
      const std::optional<gp_Pnt> m = middle_point(s);
      if (!m) {
        return false;
      }
      const gp_Vec offset(plane.Location(), *m);
      return offset.Dot(gp_Vec(plane.Axis().Direction())) > kLinear;
    };
  } else if (filter == "OppositeToEdge") {
    if (entity_ids.size() != 1) {
      throw PysmeshError("Session.select: " + filter_text(filter) + " takes one edge.");
    }
    const TopoDS_Shape start = line_of(entity_ids.front()).first;
    const Ancestors edge_faces = faces_by(root, TopAbs_EDGE);
    std::deque<TopoDS_Shape> todo{start};
    accepted.Add(start);
    while (!todo.empty()) {
      const TopoDS_Shape edge = todo.front();
      todo.pop_front();
      for (const TopoDS_Shape& face : faces_of_key(edge_faces, edge)) {
        const TopoDS_Shape opposite = opposite_in_quad(edge, face);
        if (!opposite.IsNull() && !accepted.Contains(opposite)) {
          accepted.Add(opposite);
          todo.push_back(opposite);
        }
      }
    }
  } else if (filter == "TopoConnectedFaces") {
    if (entity_ids.size() != 1) {
      throw PysmeshError("Session.select: " + filter_text(filter) + " takes one entity.");
    }
    const TopoDS_Shape from =
        one_shape(entity_ids.front(), {TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE},
                  "a vertex, an edge or a face");
    const Ancestors vertex_faces = faces_by(root, TopAbs_VERTEX);
    const Ancestors edge_faces = faces_by(root, TopAbs_EDGE);
    if (from.ShapeType() == TopAbs_VERTEX) {
      for (const TopoDS_Shape& f : faces_of_key(vertex_faces, from)) {
        accepted.Add(f);
      }
    } else if (from.ShapeType() == TopAbs_EDGE) {
      for (const TopoDS_Shape& f : faces_of_key(edge_faces, from)) {
        accepted.Add(f);
      }
    } else {
      // The face and every face that shares a vertex with it.
      accepted.Add(from);
      for (TopExp_Explorer v(from, TopAbs_VERTEX); v.More(); v.Next()) {
        for (const TopoDS_Shape& f : faces_of_key(vertex_faces, v.Current())) {
          accepted.Add(f);
        }
      }
    }
    if (flag) {
      // Propagation: every face connected to those through shared edges.
      std::deque<TopoDS_Shape> todo;
      for (int i = 1; i <= accepted.Extent(); ++i) {
        todo.push_back(accepted.FindKey(i));
      }
      while (!todo.empty()) {
        const TopoDS_Shape face = todo.front();
        todo.pop_front();
        for (TopExp_Explorer e(face, TopAbs_EDGE); e.More(); e.Next()) {
          for (const TopoDS_Shape& g : faces_of_key(edge_faces, e.Current())) {
            if (!accepted.Contains(g)) {
              accepted.Add(g);
              todo.push_back(g);
            }
          }
        }
      }
    }
  } else {
    throw PysmeshError("Session.select: '" + filter + "' is not a filter of this module.");
  }
  prepare.Close();

  // The test of every live id, with the GIL released so that the poller can report. The
  // registry is held here: a published registry is never written.
  const RegistryPtr registry = state_.registry;
  std::vector<char> passed(ids.size(), 0);
  {
    py::gil_scoped_release release;
    Message_ProgressScope each(scope.Next(), "select",
                               static_cast<double>(std::max<std::size_t>(ids.size(), 1)));
    for (std::size_t i = 0; i < ids.size() && each.More(); ++i, each.Next()) {
      const EntityRecord& rec = registry->alive.at(ids[i]);
      bool all = true;
      for (const TopoDS_Shape& s : rec.shapes) {
        const TopoDS_Shape shape = in_root(candidates, s);
        all = all && (test ? test(shape) : accepted.Contains(shape));
      }
      passed[i] = all ? 1 : 0;
    }
  }
  scope.Close();
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled(op);
  }

  py::array_t<bool> out(static_cast<py::ssize_t>(ids.size()));
  bool* mask = out.mutable_data();
  for (std::size_t i = 0; i < ids.size(); ++i) {
    mask[i] = passed[i] != 0;
  }
  return out;
}

py::array_t<double> Session::size_measures(const std::string& kind, const py::object& progress,
                                           const py::object& cancel) const {
  const char* op = "select";
  const ProgressHooks hooks = hooks_of(op, progress, cancel);
  const std::vector<EntityId> ids = ids_of_kind(kind_from_name(kind));
  std::vector<TopoDS_Shape> all;
  std::vector<std::size_t> first(ids.size() + 1, 0);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const EntityRecord& rec = state_.registry->alive.at(ids[i]);
    first[i] = all.size();
    all.insert(all.end(), rec.shapes.begin(), rec.shapes.end());
  }
  first[ids.size()] = all.size();

  // Each shape is measured on its own, as shape_checks::measures measures it for
  // entity_table, so the values are entity_table's. The ranges are taken in order before
  // the parallel loop, OCCT's pattern for a progress range in a parallel loop: a scope is
  // not thread-safe, the indicator is.
  ProgressDriver driver(op, hooks);
  std::vector<shape_checks::Measure> measures(all.size());
  {
    py::gil_scoped_release release;
    Message_ProgressScope scope(driver.range(), "select",
                                static_cast<double>(std::max<std::size_t>(all.size(), 1)));
    std::vector<Message_ProgressRange> ranges;
    ranges.reserve(all.size());
    for (std::size_t j = 0; j < all.size(); ++j) {
      ranges.push_back(scope.Next());
    }
    OSD_Parallel::For(0, static_cast<int>(all.size()), [&](const int j) {
      const auto at = static_cast<std::size_t>(j);
      Message_ProgressScope one(ranges[at], nullptr, 1.0);
      if (one.More()) {
        measures[at] = shape_checks::measure(all[at], shape_checks::kDefaultMassPrecision);
      }
    });
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled(op);
  }

  py::array_t<double> out(static_cast<py::ssize_t>(ids.size()));
  double* total = out.mutable_data();
  for (std::size_t i = 0; i < ids.size(); ++i) {
    total[i] = 0.0;
    for (std::size_t j = first[i]; j < first[i + 1]; ++j) {
      shape_checks::require_measured(measures[j], "Session.select",
                                     kind + " " + std::to_string(ids[i]));
      total[i] += measures[j].mass;
    }
  }
  return out;
}

}  // namespace session
}  // namespace pysmesh
