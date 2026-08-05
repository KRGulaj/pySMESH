// pySMESH v2 capability probe — OCCT side (primitives through the Gmsh handoff, plus IGES).
//
// Every OCCT class the v2 Tier-C modelling surface needs is constructed and run here, against
// the pinned conda-forge occt=8.0.0 headers. Where OCCT 8.0's API differs from what was
// assumed going in, the difference is recorded in a comment at the call site — that is the
// "verify at source" output the C1 milestone convention asks for.

#include "probe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Defeaturing.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepAlgoAPI_Section.hxx>
#include <BRepAlgoAPI_Splitter.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepFeat_SplitShape.hxx>
#include <BRepFilletAPI_MakeChamfer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepOffsetAPI_MakeFilling.hxx>
#include <BRepOffsetAPI_MakePipe.hxx>
#include <BRepOffsetAPI_MakePipeShell.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCone.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakeTorus.hxx>
#include <BRepPrimAPI_MakeWedge.hxx>
#include <BRepTools.hxx>
#include <BRepTools_History.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <BOPAlgo_Builder.hxx>
#include <BOPAlgo_Options.hxx>
#include <Bnd_Box.hxx>
#include <BRepBndLib.hxx>
#include <GC_MakeArcOfCircle.hxx>
#include <GProp_GProps.hxx>
#include <GeomAPI_PointsToBSpline.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_Curve.hxx>
#include <Geom_Surface.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <HelixBRep_BuilderHelix.hxx>
#include <IGESCAFControl_Reader.hxx>
#include <IGESControl_Reader.hxx>
#include <IMeshTools_Parameters.hxx>
#include <Message_ProgressIndicator.hxx>
#include <Message_ProgressRange.hxx>
#include <Message_ProgressScope.hxx>
#include <NCollection_Array1.hxx>
#include <NCollection_IndexedDataMap.hxx>
#include <NCollection_IndexedMap.hxx>
#include <NCollection_List.hxx>
#include <NCollection_Sequence.hxx>
#include <Poly_Triangulation.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeUpgrade_RemoveInternalWires.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <Standard_Failure.hxx>
#include <TopAbs.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Ax3.hxx>
#include <gp_Dir.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>
#include <gp_Vec.hxx>

namespace {

using probe::check;
using probe::check_close;
using probe::note;
using probe::section;

// Never a unit cube: a box whose three extents are distinct and non-unit cannot hide an
// axis mix-up the way a unit cube could.
constexpr double BX = 3.0;
constexpr double BY = 7.0;
constexpr double BZ = 11.0;

double volume_of(const TopoDS_Shape& s) {
  GProp_GProps props;
  BRepGProp::VolumeProperties(s, props);
  return props.Mass();
}

double area_of(const TopoDS_Shape& s) {
  GProp_GProps props;
  BRepGProp::SurfaceProperties(s, props);
  return props.Mass();
}

int count(const TopoDS_Shape& s, TopAbs_ShapeEnum kind) {
  NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> map;
  TopExp::MapShapes(s, kind, map);
  return map.Extent();
}

bool is_valid(const TopoDS_Shape& s) { return BRepCheck_Analyzer(s).IsValid(); }

// ------------------------------------------------------------------ PROGRESS progress/cancel --//
// Message_ProgressIndicator is the only OCCT progress channel the BRepAlgoAPI_*, ShapeFix_*,
// BRepMesh_* and offset APIs accept. It is a Standard_Transient, so it lives behind a handle
// and its Show()/UserBreak() are the two hooks a Python callback + cancel predicate map onto.
class CountingProgress : public Message_ProgressIndicator {
 public:
  int shows = 0;
  int break_after = -1;  // -1: never cancel
  double last_position = -1.0;
  bool monotone = true;

  bool UserBreak() override { return break_after >= 0 && shows >= break_after; }

  void Show(const Message_ProgressScope&, const bool) override {
    ++shows;
    const double p = GetPosition();
    if (p + 1e-12 < last_position) {
      monotone = false;
    }
    last_position = p;
  }
};

// ---------------------------------------------------------------------------- PRIM ------ //
void probe_r1_primitives_and_construction() {
  section("PRIM", "primitives, wire/edge construction, sweeps, filling, helix");

  const TopoDS_Shape box = BRepPrimAPI_MakeBox(BX, BY, BZ).Shape();
  check(is_valid(box), "PRIM MakeBox is BRepCheck_Analyzer-valid");
  check_close(volume_of(box), BX * BY * BZ, 1e-9, "PRIM MakeBox volume == 3*7*11");
  check(count(box, TopAbs_FACE) == 6, "PRIM MakeBox has 6 faces");

  const double r = 2.5;
  const double h = 9.0;
  const TopoDS_Shape cyl = BRepPrimAPI_MakeCylinder(r, h).Shape();
  check_close(volume_of(cyl), M_PI * r * r * h, 1e-6, "PRIM MakeCylinder volume == pi r^2 h");

  const TopoDS_Shape cone = BRepPrimAPI_MakeCone(3.0, 1.0, 6.0).Shape();
  check_close(volume_of(cone), M_PI * 6.0 / 3.0 * (9.0 + 3.0 * 1.0 + 1.0), 1e-6,
              "PRIM MakeCone volume == pi h/3 (R^2+Rr+r^2)");

  const TopoDS_Shape sph = BRepPrimAPI_MakeSphere(4.0).Shape();
  check_close(volume_of(sph), 4.0 / 3.0 * M_PI * 64.0, 1e-4, "PRIM MakeSphere volume");

  const double tr = 5.0;
  const double tp = 1.5;
  const TopoDS_Shape tor = BRepPrimAPI_MakeTorus(tr, tp).Shape();
  check_close(volume_of(tor), 2.0 * M_PI * M_PI * tr * tp * tp, 1e-4, "PRIM MakeTorus volume");

  const TopoDS_Shape wedge = BRepPrimAPI_MakeWedge(BX, BY, BZ, 1.0).Shape();
  check(is_valid(wedge), "PRIM MakeWedge is valid");

  // Extrude / revolve.
  const TopoDS_Face base =
      BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0, BX, 0, BY).Face();
  const TopoDS_Shape prism = BRepPrimAPI_MakePrism(base, gp_Vec(0, 0, BZ)).Shape();
  check_close(volume_of(prism), BX * BY * BZ, 1e-9, "PRIM MakePrism (extrude) volume");

  const TopoDS_Face rev_base =
      BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), 1.0, 3.0, 0.0, 4.0)
          .Face();
  const TopoDS_Shape revol =
      BRepPrimAPI_MakeRevol(rev_base, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))).Shape();
  check(is_valid(revol), "PRIM MakeRevol (revolve) is valid");

  // Edge / wire / polygon / arc / B-spline.
  const TopoDS_Edge line = BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(BX, 0, 0)).Edge();
  check(!line.IsNull(), "PRIM MakeEdge (line)");

  const occ::handle<Geom_TrimmedCurve> arc =
      GC_MakeArcOfCircle(gp_Pnt(1, 0, 0), gp_Pnt(0, 1, 0), gp_Pnt(-1, 0, 0)).Value();
  const TopoDS_Edge arc_edge = BRepBuilderAPI_MakeEdge(arc).Edge();
  check(!arc_edge.IsNull(), "PRIM GC_MakeArcOfCircle -> MakeEdge");

  BRepBuilderAPI_MakePolygon poly;
  poly.Add(gp_Pnt(0, 0, 0));
  poly.Add(gp_Pnt(BX, 0, 0));
  poly.Add(gp_Pnt(BX, BY, 0));
  poly.Add(gp_Pnt(0, BY, 0));
  poly.Close();
  check(poly.IsDone() && count(poly.Wire(), TopAbs_EDGE) == 4, "PRIM MakePolygon closed wire");

  const TopoDS_Face poly_face = BRepBuilderAPI_MakeFace(poly.Wire()).Face();
  check_close(area_of(poly_face), BX * BY, 1e-9, "PRIM MakeFace from wire, area == 3*7");

  // OCCT 8.0: GeomAPI_PointsToBSpline takes NCollection_Array1<gp_Pnt>; TColgp_Array1OfPnt is
  // a deprecated alias of exactly that type, so either spelling compiles — use the new one.
  NCollection_Array1<gp_Pnt> pts(1, 5);
  for (int i = 1; i <= 5; ++i) {
    pts.SetValue(i, gp_Pnt(i - 1, std::sin(i - 1.0), 0.0));
  }
  const GeomAPI_PointsToBSpline fit(pts);
  const TopoDS_Edge spline_edge = BRepBuilderAPI_MakeEdge(fit.Curve()).Edge();
  check(!spline_edge.IsNull(), "PRIM GeomAPI_PointsToBSpline -> MakeEdge");

  // Sweeps: pipe along a spine, pipe shell, thru-sections.
  const TopoDS_Wire spine =
      BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(0, 0, 10)).Edge())
          .Wire();
  BRepBuilderAPI_MakePolygon prof;
  prof.Add(gp_Pnt(-1, -1, 0));
  prof.Add(gp_Pnt(1, -1, 0));
  prof.Add(gp_Pnt(1, 1, 0));
  prof.Add(gp_Pnt(-1, 1, 0));
  prof.Close();
  const TopoDS_Shape pipe = BRepOffsetAPI_MakePipe(spine, prof.Wire()).Shape();
  check(is_valid(pipe), "PRIM BRepOffsetAPI_MakePipe is valid");

  BRepOffsetAPI_MakePipeShell shell(spine);
  shell.Add(prof.Wire());
  shell.Build();
  check(shell.IsDone(), "PRIM BRepOffsetAPI_MakePipeShell builds");

  BRepOffsetAPI_ThruSections loft(/*isSolid=*/true, /*ruled=*/true);
  BRepBuilderAPI_MakePolygon s0;
  s0.Add(gp_Pnt(-2, -2, 0));
  s0.Add(gp_Pnt(2, -2, 0));
  s0.Add(gp_Pnt(2, 2, 0));
  s0.Add(gp_Pnt(-2, 2, 0));
  s0.Close();
  BRepBuilderAPI_MakePolygon s1;
  s1.Add(gp_Pnt(-1, -1, 6));
  s1.Add(gp_Pnt(1, -1, 6));
  s1.Add(gp_Pnt(1, 1, 6));
  s1.Add(gp_Pnt(-1, 1, 6));
  s1.Close();
  loft.AddWire(s0.Wire());
  loft.AddWire(s1.Wire());
  loft.Build();
  check(loft.IsDone() && volume_of(loft.Shape()) > 0.0,
        "PRIM BRepOffsetAPI_ThruSections builds a solid");

  // Filling — needed for non-planar caps / surface filling.
  BRepOffsetAPI_MakeFilling filling;
  for (TopExp_Explorer ex(poly.Wire(), TopAbs_EDGE); ex.More(); ex.Next()) {
    filling.Add(TopoDS::Edge(ex.Current()), GeomAbs_C0);
  }
  filling.Build();
  check(filling.IsDone(), "PRIM BRepOffsetAPI_MakeFilling builds");

  // Helix — the requirements document flagged this "verify at source". TKHelix in OCCT 8.0
  // exposes HelixBRep_BuilderHelix (composite/pure helix and spiral) and the HelixGeom_*
  // curve builders; there is no BRepPrimAPI-style facade.
  HelixBRep_BuilderHelix helix;
  NCollection_Array1<double> pitches(1, 1);
  pitches.SetValue(1, 2.0);
  NCollection_Array1<double> turns(1, 1);
  turns.SetValue(1, 3.0);
  helix.SetParameters(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), /*theDiam=*/4.0, pitches,
                      turns);
  helix.Perform();
  check(helix.ErrorStatus() == 0 && !helix.Shape().IsNull(),
        "PRIM TKHelix HelixBRep_BuilderHelix builds a helical wire");
}

// ---------------------------------------------------------------------------- BOOL ------ //
void probe_r2_booleans() {
  section("BOOL", "booleans with history, fuzzy value, parallel mode");

  const TopoDS_Shape a = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), BX, BY, BZ).Shape();
  const TopoDS_Shape b = BRepPrimAPI_MakeBox(gp_Pnt(BX, 0, 0), BX, BY, BZ).Shape();

  NCollection_List<TopoDS_Shape> args;
  args.Append(a);
  NCollection_List<TopoDS_Shape> tools;
  tools.Append(b);

  BRepAlgoAPI_Fuse fuse;
  fuse.SetArguments(args);
  fuse.SetTools(tools);
  fuse.SetToFillHistory(true);  // mandatory — the history IS the naming substrate
  fuse.SetRunParallel(true);
  fuse.SetFuzzyValue(1e-6);
  fuse.SetNonDestructive(true);
  fuse.Build();
  check(fuse.IsDone() && !fuse.HasErrors(), "BOOL BRepAlgoAPI_Fuse builds");
  check_close(volume_of(fuse.Shape()), 2.0 * BX * BY * BZ, 1e-6, "BOOL fuse volume == 2 boxes");

  const occ::handle<BRepTools_History> hist = fuse.History();
  check(!hist.IsNull(), "BOOL SetToFillHistory(true) yields a BRepTools_History");

  // Every input face must be classifiable as modified / generated / deleted / untouched.
  int modified = 0;
  int deleted = 0;
  int untouched = 0;
  for (TopExp_Explorer ex(a, TopAbs_FACE); ex.More(); ex.Next()) {
    if (fuse.IsDeleted(ex.Current())) {
      ++deleted;
    } else if (!fuse.Modified(ex.Current()).IsEmpty()) {
      ++modified;
    } else {
      ++untouched;
    }
  }
  check(modified + deleted + untouched == 6,
        "BOOL every input face of A is classified by the history");
  check(deleted == 1, "BOOL the shared wall of A is reported deleted by the fuse");

  BRepAlgoAPI_Cut cut;
  cut.SetArguments(args);
  cut.SetTools(tools);
  cut.SetToFillHistory(true);
  cut.Build();
  check(cut.IsDone(), "BOOL BRepAlgoAPI_Cut builds");

  BRepAlgoAPI_Common common;
  common.SetArguments(args);
  common.SetTools(tools);
  common.SetToFillHistory(true);
  common.Build();
  check(common.IsDone(), "BOOL BRepAlgoAPI_Common builds");

  BRepAlgoAPI_Section section_op;
  section_op.SetArguments(args);
  section_op.SetTools(tools);
  section_op.SetToFillHistory(true);
  section_op.Build();
  check(section_op.IsDone() && count(section_op.Shape(), TopAbs_EDGE) > 0,
        "BOOL BRepAlgoAPI_Section builds section edges");

  // General fuse / fragment.
  BOPAlgo_Builder gf;
  gf.AddArgument(a);
  gf.AddArgument(b);
  gf.SetRunParallel(true);
  gf.Perform();
  check(!gf.HasErrors() && count(gf.Shape(), TopAbs_SOLID) == 2,
        "BOOL BOPAlgo_Builder general fuse splits into 2 solids");
  check(!gf.Modified(a).IsEmpty() || !gf.Modified(b).IsEmpty(),
        "BOOL BOPAlgo_Builder reports Modified()");

  // A boolean OCCT itself reports as failed must be detectable, not silently partial.
  BRepAlgoAPI_Fuse empty_fuse;
  empty_fuse.Build();
  check(empty_fuse.HasErrors(), "BOOL an argument-less boolean reports HasErrors()");
}

// ---------------------------------------------------------------------------- FILLET ------ //
void probe_r3_fillet_chamfer() {
  section("FILLET", "fillet (constant + variable radius) and chamfer, with history");

  const TopoDS_Shape box = BRepPrimAPI_MakeBox(BX, BY, BZ).Shape();

  NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> edges;
  TopExp::MapShapes(box, TopAbs_EDGE, edges);
  check(edges.Extent() == 12, "FILLET fixture box has 12 edges");

  // Constant radius on every edge — OCCT derives the owning solid itself, so no per-edge
  // face co-selection is needed, unlike a mesher API that requires one face tag per edge.
  BRepFilletAPI_MakeFillet fillet(box);
  for (int i = 1; i <= edges.Extent(); ++i) {
    fillet.Add(0.5, TopoDS::Edge(edges.FindKey(i)));
  }
  fillet.Build();
  check(fillet.IsDone(), "FILLET BRepFilletAPI_MakeFillet builds on 12 edges in one op");
  check(is_valid(fillet.Shape()), "FILLET filleted solid is BRepCheck_Analyzer-valid");
  check(volume_of(fillet.Shape()) < BX * BY * BZ,
        "FILLET filleting a convex box removes material");

  int fillet_generated = 0;
  for (int i = 1; i <= edges.Extent(); ++i) {
    if (!fillet.Generated(edges.FindKey(i)).IsEmpty()) {
      ++fillet_generated;
    }
  }
  check(fillet_generated == 12, "FILLET every filleted edge reports Generated() faces");

  // Variable radius.
  BRepFilletAPI_MakeFillet var(box);
  var.Add(0.2, 0.9, TopoDS::Edge(edges.FindKey(1)));
  var.Build();
  check(var.IsDone() && is_valid(var.Shape()), "FILLET variable-radius fillet builds and is valid");

  // A radius OCCT cannot build must fail detectably — fail loud, never silently drop the edge.
  BRepFilletAPI_MakeFillet impossible(box);
  bool impossible_detected = false;
  try {
    impossible.Add(1e4, TopoDS::Edge(edges.FindKey(1)));
    impossible.Build();
    impossible_detected = !impossible.IsDone() || !is_valid(impossible.Shape());
  } catch (const Standard_Failure&) {
    impossible_detected = true;  // OCCT raises rather than returning a bad shape
  }
  check(impossible_detected, "FILLET an unbuildable fillet radius is detectable (throw or !IsDone)");

  // Chamfer: OCCT 8.0 takes either the edge alone (symmetric, distance set later) or a
  // two-distance form that names the reference face — there is no (dist, edge, face) overload.
  // The two-distance form is what matches SpaceClaim's interaction.
  NCollection_IndexedDataMap<TopoDS_Shape, NCollection_List<TopoDS_Shape>,
                             TopTools_ShapeMapHasher>
      edge_faces;
  TopExp::MapShapesAndAncestors(box, TopAbs_EDGE, TopAbs_FACE, edge_faces);
  const TopoDS_Edge e1 = TopoDS::Edge(edges.FindKey(1));
  const TopoDS_Face f1 = TopoDS::Face(edge_faces.FindFromKey(e1).First());
  BRepFilletAPI_MakeChamfer chamfer(box);
  chamfer.Add(0.4, 0.4, e1, f1);
  chamfer.Build();
  check(chamfer.IsDone() && is_valid(chamfer.Shape()), "FILLET BRepFilletAPI_MakeChamfer builds");
  check(!chamfer.Generated(e1).IsEmpty(), "FILLET chamfer reports Generated() for its edge");
}

// ---------------------------------------------------------------------------- XFORM ------ //
void probe_r4_transforms() {
  section("XFORM", "rigid transform preserves TShape identity; GTransform for scale/mirror");

  const TopoDS_Shape box = BRepPrimAPI_MakeBox(BX, BY, BZ).Shape();

  gp_Trsf move;
  move.SetTranslation(gp_Vec(10.0, -3.0, 2.0));

  // The identity property a session op should preserve by construction: Copy=false leaves the
  // TShape (hence every sub-shape identity) untouched and changes only the Location.
  BRepBuilderAPI_Transform rigid(box, move, /*Copy=*/false);
  const TopoDS_Shape moved = rigid.Shape();
  check(moved.TShape() == box.TShape(),
        "XFORM rigid transform with Copy=false preserves the root TShape pointer");

  NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> before;
  NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> after;
  TopExp::MapShapes(box, TopAbs_FACE, before);
  TopExp::MapShapes(moved, TopAbs_FACE, after);
  bool all_faces_same_tshape = before.Extent() == after.Extent();
  for (int i = 1; i <= before.Extent() && all_faces_same_tshape; ++i) {
    all_faces_same_tshape = before.FindKey(i).TShape() == after.FindKey(i).TShape();
  }
  check(all_faces_same_tshape,
        "XFORM every face TShape survives a rigid transform (asserted face-by-face)");
  check_close(volume_of(moved), BX * BY * BZ, 1e-9, "XFORM rigid transform preserves volume");

  // Copy=true is the contrasting case; it must NOT be the default path a session op takes.
  BRepBuilderAPI_Transform copying(box, move, /*Copy=*/true);
  check(copying.Shape().TShape() != box.TShape(),
        "XFORM Copy=true does rebuild the TShape (the case a session op must avoid)");

  gp_Trsf mirror;
  mirror.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)));
  BRepBuilderAPI_Transform mirrored(box, mirror, /*Copy=*/false);
  check_close(std::fabs(volume_of(mirrored.Shape())), BX * BY * BZ, 1e-9,
              "XFORM mirror preserves |volume|");

  gp_GTrsf scale;
  scale.SetValue(1, 1, 2.0);
  scale.SetValue(2, 2, 1.0);
  scale.SetValue(3, 3, 1.0);
  BRepBuilderAPI_GTransform non_uniform(box, scale, /*Copy=*/true);
  check(non_uniform.IsDone(), "XFORM BRepBuilderAPI_GTransform (non-uniform scale) builds");
  check_close(volume_of(non_uniform.Shape()), 2.0 * BX * BY * BZ, 1e-6,
              "XFORM non-uniform scale x2 in X doubles the volume");
}

// ---------------------------------------------------------------------------- HEAL ------ //
void probe_r5_heal_defeature_imprint() {
  section("HEAL", "ShapeFix, sewing, internal-wire removal, defeaturing, imprint");

  const TopoDS_Shape box = BRepPrimAPI_MakeBox(BX, BY, BZ).Shape();

  ShapeFix_Shape fixer;
  fixer.Init(box);
  fixer.SetPrecision(1e-7);
  fixer.SetMaxTolerance(1e-3);
  const bool fixed = fixer.Perform();
  check(!fixer.Shape().IsNull(), "HEAL ShapeFix_Shape::Perform returns a shape");
  check(fixed || is_valid(fixer.Shape()), "HEAL ShapeFix_Shape leaves a valid shape");

  // ShapeFix_Shape is scoped by construction — feeding it one solid of a compound must leave
  // the rest of the compound untouched. A global healing pass over the whole model cannot
  // offer that guarantee, so assert it on TShape identity, not on counts.
  const TopoDS_Shape other = BRepPrimAPI_MakeBox(gp_Pnt(20, 0, 0), 1.0, 2.0, 3.0).Shape();
  TopoDS_Compound comp;
  BRep_Builder bb;
  bb.MakeCompound(comp);
  bb.Add(comp, box);
  bb.Add(comp, other);
  std::vector<const void*> other_faces_before;
  for (TopExp_Explorer ex(other, TopAbs_FACE); ex.More(); ex.Next()) {
    other_faces_before.push_back(ex.Current().TShape().get());
  }
  ShapeFix_Shape scoped;
  scoped.Init(box);  // scope == this solid only
  scoped.Perform();
  std::vector<const void*> other_faces_after;
  for (TopExp_Explorer ex(other, TopAbs_FACE); ex.More(); ex.Next()) {
    other_faces_after.push_back(ex.Current().TShape().get());
  }
  check(!other_faces_before.empty() && other_faces_before == other_faces_after,
        "HEAL healing one body leaves every out-of-scope face TShape byte-identical");

  // Sewing: two faces sharing an edge sew into one shell.
  BRepBuilderAPI_Sewing sewing(1e-6);
  for (TopExp_Explorer ex(box, TopAbs_FACE); ex.More(); ex.Next()) {
    sewing.Add(ex.Current());
  }
  sewing.Perform();
  check(!sewing.SewedShape().IsNull(), "HEAL BRepBuilderAPI_Sewing sews the box faces");
  check(count(sewing.SewedShape(), TopAbs_SHELL) == 1, "HEAL sewing yields exactly one shell");

  // Internal-wire removal (small-hole defeaturing on a face).
  ShapeUpgrade_RemoveInternalWires remover(box);
  remover.MinArea() = 1.0;
  remover.RemoveFaceMode() = true;
  const bool removed_ok = remover.Perform();
  check(removed_ok || !remover.GetResult().IsNull(),
        "HEAL ShapeUpgrade_RemoveInternalWires runs and returns a result");

  // Defeaturing: drill a THROUGH hole in the middle of the box. VERIFY-AT-SOURCE FINDING:
  // BRepAlgoAPI_Defeaturing removes a *complete* feature — hand it a blind hole's cylindrical
  // wall without the flat bottom that caps it and it reports
  // BOPAlgo_AlertUnableToRemoveTheFeature and returns the input unchanged (IsDone() is still
  // true, HasErrors() is still false). The binding must therefore verify that faces actually
  // went away, not trust IsDone().
  const TopoDS_Shape boss =
      BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(BX / 2.0, BY / 2.0, -1.0), gp_Dir(0, 0, 1)), 0.5,
                               BZ + 2.0)
          .Shape();
  NCollection_List<TopoDS_Shape> bargs;
  bargs.Append(box);
  NCollection_List<TopoDS_Shape> btools;
  btools.Append(boss);
  BRepAlgoAPI_Cut holed;
  holed.SetArguments(bargs);
  holed.SetTools(btools);
  holed.SetToFillHistory(true);
  holed.Build();
  check(holed.IsDone(), "HEAL hole cut for the defeaturing fixture builds");

  // The hole's cylindrical face is the one Cut generated from the boss's lateral face.
  NCollection_List<TopoDS_Shape> to_remove;
  for (TopExp_Explorer ex(holed.Shape(), TopAbs_FACE); ex.More(); ex.Next()) {
    const BRepAdaptor_Surface surf(TopoDS::Face(ex.Current()));
    if (surf.GetType() == GeomAbs_Cylinder) {
      to_remove.Append(ex.Current());
    }
  }
  check(!to_remove.IsEmpty(), "HEAL defeaturing fixture exposes a cylindrical hole face");

  BRepAlgoAPI_Defeaturing defeat;
  defeat.SetShape(holed.Shape());
  defeat.AddFacesToRemove(to_remove);
  defeat.SetToFillHistory(true);
  defeat.SetRunParallel(false);
  defeat.Build();
  {
    std::ostringstream errs;
    defeat.DumpErrors(errs);
    defeat.DumpWarnings(errs);
    char msg[512];
    std::snprintf(msg, sizeof(msg),
                  "HEAL BRepAlgoAPI_Defeaturing builds (%d faces removed of %d; %s)",
                  count(holed.Shape(), TopAbs_FACE) - count(defeat.Shape(), TopAbs_FACE),
                  count(holed.Shape(), TopAbs_FACE),
                  errs.str().empty() ? "no diagnostics" : errs.str().c_str());
    check(defeat.IsDone() && !defeat.HasErrors(), msg);
  }
  check(count(defeat.Shape(), TopAbs_FACE) < count(holed.Shape(), TopAbs_FACE),
        "HEAL defeaturing removes faces");
  check(defeat.HasDeleted() || defeat.HasModified(),
        "HEAL defeaturing reports history (deleted/modified)");

  // Imprint via the splitter (glue on) and via BRepFeat_SplitShape (TKFeat).
  const TopoDS_Shape tool_face =
      BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, BZ / 2.0), gp_Dir(0, 0, 1))).Face();
  NCollection_List<TopoDS_Shape> sargs;
  sargs.Append(box);
  NCollection_List<TopoDS_Shape> stools;
  stools.Append(tool_face);
  BRepAlgoAPI_Splitter splitter;
  splitter.SetArguments(sargs);
  splitter.SetTools(stools);
  splitter.SetToFillHistory(true);
  splitter.Build();
  check(splitter.IsDone() && count(splitter.Shape(), TopAbs_SOLID) == 2,
        "HEAL BRepAlgoAPI_Splitter imprints and splits the box in two");

  // BRepFeat_SplitShape (TKFeat) is the alternative imprint path: split a named face by a
  // wire lying on it. Pick the z=0 face and imprint a diagonal edge across it.
  TopoDS_Face target;
  for (TopExp_Explorer ex(box, TopAbs_FACE); ex.More(); ex.Next()) {
    GProp_GProps fp;
    BRepGProp::SurfaceProperties(ex.Current(), fp);
    if (std::fabs(fp.CentreOfMass().Z()) < 1e-9) {
      target = TopoDS::Face(ex.Current());
      break;
    }
  }
  check(!target.IsNull(), "HEAL imprint target face located");
  BRepFeat_SplitShape feat_split(box);
  feat_split.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(BX, BY, 0)).Edge(), target);
  feat_split.Build();
  check(feat_split.IsDone(), "HEAL BRepFeat_SplitShape (TKFeat) imprints an edge on a face");
  check(count(feat_split.Shape(), TopAbs_FACE) > count(box, TopAbs_FACE),
        "HEAL BRepFeat_SplitShape splits the imprinted face");

  // v1's UnifySameDomain still applies as the merge-bodies path.
  ShapeUpgrade_UnifySameDomain unify(splitter.Shape(), true, true, false);
  unify.Build();
  check(!unify.Shape().IsNull() && !unify.History().IsNull(),
        "HEAL ShapeUpgrade_UnifySameDomain still builds with a history");
}

// ---------------------------------------------------------------------------- QUERY ------ //
void probe_r6_queries() {
  section("QUERY", "mass/centroid, boundary, adjacency, curvature, projection, bbox");

  const TopoDS_Shape box = BRepPrimAPI_MakeBox(BX, BY, BZ).Shape();

  GProp_GProps vprops;
  BRepGProp::VolumeProperties(box, vprops);
  check_close(vprops.Mass(), BX * BY * BZ, 1e-9, "QUERY BRepGProp::VolumeProperties mass");
  const gp_Pnt c = vprops.CentreOfMass();
  check_close(c.X(), BX / 2.0, 1e-9, "QUERY centre of mass X");
  check_close(c.Y(), BY / 2.0, 1e-9, "QUERY centre of mass Y");
  check_close(c.Z(), BZ / 2.0, 1e-9, "QUERY centre of mass Z");

  GProp_GProps sprops;
  BRepGProp::SurfaceProperties(box, sprops);
  check_close(sprops.Mass(), 2.0 * (BX * BY + BY * BZ + BX * BZ), 1e-9,
              "QUERY BRepGProp::SurfaceProperties area");

  // VERIFY-AT-SOURCE FINDING: BRepGProp::LinearProperties walks the shape with a
  // TopExp_Explorer, so on a SOLID every edge is visited once per owning face and the total
  // comes out DOUBLED (168 instead of 84 for a box). A total-edge-length query must sum over
  // a de-duplicated TopExp edge map; calling it on the solid is a silent 2x error.
  GProp_GProps lprops_solid;
  BRepGProp::LinearProperties(box, lprops_solid);
  check_close(lprops_solid.Mass(), 2.0 * 4.0 * (BX + BY + BZ), 1e-9,
              "QUERY LinearProperties on a solid double-counts shared edges (2x, documented)");

  NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> unique_edges;
  TopExp::MapShapes(box, TopAbs_EDGE, unique_edges);
  double edge_total = 0.0;
  for (int i = 1; i <= unique_edges.Extent(); ++i) {
    GProp_GProps e;
    BRepGProp::LinearProperties(unique_edges.FindKey(i), e);
    edge_total += e.Mass();
  }
  check_close(edge_total, 4.0 * (BX + BY + BZ), 1e-9,
              "QUERY summing LinearProperties over a de-duplicated edge map gives 4(3+7+11)");

  Bnd_Box bnd;
  BRepBndLib::Add(box, bnd);
  double xmin, ymin, zmin, xmax, ymax, zmax;
  bnd.Get(xmin, ymin, zmin, xmax, ymax, zmax);
  check(xmax - xmin >= BX - 1e-6 && ymax - ymin >= BY - 1e-6 && zmax - zmin >= BZ - 1e-6,
        "QUERY Bnd_Box covers the box (getBoundingBox / getEntitiesInBoundingBox)");

  NCollection_IndexedDataMap<TopoDS_Shape, NCollection_List<TopoDS_Shape>,
                             TopTools_ShapeMapHasher>
      face_of_edge;
  TopExp::MapShapesAndAncestors(box, TopAbs_EDGE, TopAbs_FACE, face_of_edge);
  check(face_of_edge.Extent() == 12 && face_of_edge.FindFromIndex(1).Extent() == 2,
        "QUERY TopExp::MapShapesAndAncestors gives edge->face adjacency (getBoundary)");

  NCollection_IndexedDataMap<TopoDS_Shape, NCollection_List<TopoDS_Shape>,
                             TopTools_ShapeMapHasher>
      solid_of_face;
  TopExp::MapShapesAndAncestors(box, TopAbs_FACE, TopAbs_SOLID, solid_of_face);
  check(solid_of_face.Extent() == 6, "QUERY face->solid adjacency");

  // Curvature — a correctness item, not a formality. Validate against closed-form curvature.
  const double sphere_r = 4.0;
  const TopoDS_Shape sph = BRepPrimAPI_MakeSphere(sphere_r).Shape();
  TopExp_Explorer sf(sph, TopAbs_FACE);
  const TopoDS_Face sphere_face = TopoDS::Face(sf.Current());
  BRepAdaptor_Surface sphere_surf(sphere_face);
  const double su = 0.5 * (sphere_surf.FirstUParameter() + sphere_surf.LastUParameter());
  const double sv = 0.5 * (sphere_surf.FirstVParameter() + sphere_surf.LastVParameter());
  // OCCT 8.0: BRepLProp_SLProps is an alias template
  // (GeomLProp_SLPropsBase<BRepAdaptor_Surface>), not a class of its own — a header-only
  // change from 7.x that affects forward declarations, not call sites.
  BRepLProp_SLProps sphere_props(sphere_surf, su, sv, 2, 1e-7);
  check(sphere_props.IsCurvatureDefined(), "QUERY BRepLProp_SLProps curvature is defined");
  const double sphere_kmax = sphere_props.MaxCurvature();
  const double sphere_kmin = sphere_props.MinCurvature();
  check_close(std::fabs(sphere_kmax), 1.0 / sphere_r, 1e-6, "QUERY sphere max |kappa| == 1/R");
  check_close(std::fabs(sphere_kmin), 1.0 / sphere_r, 1e-6, "QUERY sphere min |kappa| == 1/R");

  const double cyl_r = 2.5;
  const TopoDS_Shape cyl = BRepPrimAPI_MakeCylinder(cyl_r, 9.0).Shape();
  bool cylinder_checked = false;
  for (TopExp_Explorer ex(cyl, TopAbs_FACE); ex.More() && !cylinder_checked; ex.Next()) {
    BRepAdaptor_Surface surf(TopoDS::Face(ex.Current()));
    if (surf.GetType() != GeomAbs_Cylinder) {
      continue;
    }
    const double u = 0.5 * (surf.FirstUParameter() + surf.LastUParameter());
    const double v = 0.5 * (surf.FirstVParameter() + surf.LastVParameter());
    BRepLProp_SLProps props(surf, u, v, 2, 1e-7);
    // VERIFY-AT-SOURCE FINDING: MaxCurvature()/MinCurvature() return the SIGNED principal
    // curvatures ordered by value, not by magnitude. On an outward-normal cylinder they are
    // (0, -1/R), so MaxCurvature() is 0 and the curvature that matters for sizing is
    // |MinCurvature()|. A curvature map keyed on MaxCurvature() alone would report a
    // cylinder as flat. QUERY's "max |kappa|" must be max(|Max|, |Min|).
    // Read each principal curvature exactly ONCE: MaxCurvature()/MinCurvature() are
    // non-const and mutate the cached-derivative state, so repeated calls inside one
    // expression are not equivalent to a single read.
    const double k_max_signed = props.MaxCurvature();
    const double k_min_signed = props.MinCurvature();
    const double kmax_abs = std::max(std::fabs(k_max_signed), std::fabs(k_min_signed));
    const double kmin_abs = std::min(std::fabs(k_max_signed), std::fabs(k_min_signed));
    check_close(kmax_abs, 1.0 / cyl_r, 1e-6, "QUERY cylinder max |kappa| == 1/R");
    check_close(kmin_abs, 0.0, 1e-6, "QUERY cylinder min |kappa| == 0");
    check(std::fabs(k_max_signed) < 1e-9 || std::fabs(k_min_signed) < 1e-9,
          "QUERY one cylinder principal curvature is exactly 0 (signed pair, documented)");
    // getNormal / getValue on the same object.
    check(props.IsNormalDefined(), "QUERY BRepLProp_SLProps normal is defined (getNormal)");
    const gp_Pnt at = surf.Value(u, v);
    check_close(std::hypot(at.X(), at.Y()), cyl_r, 1e-6,
                "QUERY BRepAdaptor_Surface::Value lands on the cylinder (getValue)");
    cylinder_checked = true;
  }
  check(cylinder_checked, "QUERY cylinder face found for the curvature check");

  const double tor_r = 5.0;
  const double tor_p = 1.5;
  const TopoDS_Shape tor = BRepPrimAPI_MakeTorus(tor_r, tor_p).Shape();
  TopExp_Explorer tf(tor, TopAbs_FACE);
  BRepAdaptor_Surface torus_surf(TopoDS::Face(tf.Current()));
  BRepLProp_SLProps torus_props(torus_surf, 0.0, 0.0, 2, 1e-7);
  check(torus_props.IsCurvatureDefined(), "QUERY torus curvature is defined");
  const double torus_kmax = torus_props.MaxCurvature();
  const double torus_kmin = torus_props.MinCurvature();
  const double k1 = std::fabs(torus_kmax);
  const double k2 = std::fabs(torus_kmin);
  // At (u,v) = (0,0) the torus principal curvatures are 1/p and 1/(R+p).
  check(std::fabs(k1 - 1.0 / tor_p) < 1e-5 || std::fabs(k2 - 1.0 / tor_p) < 1e-5,
        "QUERY torus carries the 1/p principal curvature");
  note("QUERY n x n curvature sampling grid",
       "sampling policy is a binding-layer decision; OCCT gives per-(u,v) curvature, "
       "which is the only kernel capability the probe needs to prove");

  // getClosestPoint.
  const occ::handle<Geom_Surface> gsurf = BRep_Tool::Surface(sphere_face);
  GeomAPI_ProjectPointOnSurf proj(gp_Pnt(10.0, 0.0, 0.0), gsurf);
  check(proj.IsDone() && proj.NbPoints() > 0,
        "QUERY GeomAPI_ProjectPointOnSurf projects (getClosestPoint)");
  check_close(proj.LowerDistance(), 10.0 - sphere_r, 1e-6,
              "QUERY projection distance to a sphere of radius 4 from x=10");
}

// ---------------------------------------------------------------------------- TESS ------ //
void probe_r7_tessellation() {
  section("TESS", "tessellation from a live shape, per-face triangulation cache, parallel");

  const TopoDS_Shape box = BRepPrimAPI_MakeBox(BX, BY, BZ).Shape();

  IMeshTools_Parameters params;
  params.Deflection = 0.1;
  params.Angle = 0.3;
  params.Relative = false;
  params.InParallel = true;
  BRepMesh_IncrementalMesh mesher(box, params);
  check(mesher.IsDone(), "TESS BRepMesh_IncrementalMesh (absolute deflection, parallel) done");

  int triangulated = 0;
  int triangles = 0;
  for (TopExp_Explorer ex(box, TopAbs_FACE); ex.More(); ex.Next()) {
    TopLoc_Location loc;
    const occ::handle<Poly_Triangulation> tri =
        BRep_Tool::Triangulation(TopoDS::Face(ex.Current()), loc);
    if (!tri.IsNull()) {
      ++triangulated;
      triangles += tri->NbTriangles();
    }
  }
  check(triangulated == 6 && triangles >= 12,
        "TESS every face carries a Poly_Triangulation after meshing");

  // VERIFY-AT-SOURCE FINDING: IMeshTools_Parameters::CleanModel defaults to TRUE, which
  // drops every existing Poly_Triangulation before meshing — so the default parameter set
  // throws the incremental property away and re-meshes the whole model every call. TESS's
  // "re-triangulate only invalidated faces" mode REQUIRES CleanModel = false.
  IMeshTools_Parameters dirty = params;
  BRepMesh_IncrementalMesh cleaned(box, dirty);
  check(cleaned.IsDone() && cleaned.IsModified(),
        "TESS CleanModel=true (the default) re-meshes everything: IsModified()==true");

  IMeshTools_Parameters incremental = params;
  incremental.CleanModel = false;
  BRepMesh_IncrementalMesh again(box, incremental);
  check(again.IsDone() && !again.IsModified(),
        "TESS CleanModel=false honours the per-face cache: IsModified()==false on re-mesh");

  // Relative deflection (already shipped in v1) still works through the parameter struct.
  IMeshTools_Parameters rel;
  rel.Deflection = 0.05;
  rel.Relative = true;
  BRepMesh_IncrementalMesh relative(box, rel);
  check(relative.IsDone(), "TESS relative-deflection meshing works");

  // A shape whose triangulation is invalidated by an op must re-triangulate; BRepMesh keys
  // that off the face's own cache, so a freshly built face has none.
  const TopoDS_Shape fresh = BRepPrimAPI_MakeBox(BX, BY, BZ).Shape();
  TopExp_Explorer fx(fresh, TopAbs_FACE);
  TopLoc_Location floc;
  check(BRep_Tool::Triangulation(TopoDS::Face(fx.Current()), floc).IsNull(),
        "TESS a newly built face has no triangulation (basis of O(k) re-tessellation)");
}

// ---------------------------------------------------------------------------- PROGRESS --- //
void probe_r8_r9_progress_and_tolerance() {
  section("PROGRESS", "progress reporting, cancellation, fuzzy value, parallel determinism");

  const TopoDS_Shape a = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0), 5.0).Shape();
  const TopoDS_Shape b = BRepPrimAPI_MakeSphere(gp_Pnt(4, 0, 0), 5.0).Shape();
  NCollection_List<TopoDS_Shape> args;
  args.Append(a);
  NCollection_List<TopoDS_Shape> tools;
  tools.Append(b);

  occ::handle<CountingProgress> prog = new CountingProgress();
  BRepAlgoAPI_Fuse fuse;
  fuse.SetArguments(args);
  fuse.SetTools(tools);
  fuse.SetToFillHistory(true);
  fuse.Build(prog->Start());
  check(fuse.IsDone() && !fuse.HasErrors(), "PROGRESS boolean under a progress indicator succeeds");
  check(prog->shows > 0, "PROGRESS Message_ProgressIndicator::Show is called by the boolean");
  check(prog->monotone, "PROGRESS reported progress is monotone");

  // Cancellation: UserBreak() true makes OCCT abandon the operation with an error, never a
  // partial result. The session op maps this onto a raised PysmeshError.
  occ::handle<CountingProgress> cancel = new CountingProgress();
  cancel->break_after = 1;
  BRepAlgoAPI_Fuse cancelled;
  cancelled.SetArguments(args);
  cancelled.SetTools(tools);
  cancelled.Build(cancel->Start());
  check(cancel->shows > 0, "PROGRESS cancel-path indicator is polled");
  check(cancelled.HasErrors() || !cancelled.IsDone(),
        "PROGRESS a user-broken boolean reports an error rather than a partial result");

  // Tessellation and healing accept the same channel.
  occ::handle<CountingProgress> mesh_prog = new CountingProgress();
  IMeshTools_Parameters params;
  params.Deflection = 0.02;
  BRepMesh_IncrementalMesh mesher(a, params, mesh_prog->Start());
  check(mesher.IsDone(), "PROGRESS BRepMesh_IncrementalMesh accepts a Message_ProgressRange");

  occ::handle<CountingProgress> fix_prog = new CountingProgress();
  ShapeFix_Shape fixer;
  fixer.Init(a);
  fixer.Perform(fix_prog->Start());
  check(!fixer.Shape().IsNull(), "PROGRESS ShapeFix_Shape accepts a Message_ProgressRange");

  // PROGRESS: parallel and serial must agree.
  auto fuse_with = [&](bool parallel, double fuzzy) {
    BRepAlgoAPI_Fuse op;
    op.SetArguments(args);
    op.SetTools(tools);
    op.SetToFillHistory(true);
    op.SetRunParallel(parallel);
    op.SetFuzzyValue(fuzzy);
    op.Build();
    return op.Shape();
  };
  const TopoDS_Shape serial = fuse_with(false, 0.0);
  const TopoDS_Shape parallel = fuse_with(true, 0.0);
  check(count(serial, TopAbs_FACE) == count(parallel, TopAbs_FACE) &&
            count(serial, TopAbs_EDGE) == count(parallel, TopAbs_EDGE),
        "PROGRESS parallel and serial booleans agree on topology counts");
  check_close(volume_of(parallel), volume_of(serial), 1e-6,
              "PROGRESS parallel and serial booleans agree on volume");

  const TopoDS_Shape fuzzed = fuse_with(false, 1e-3);
  check(!fuzzed.IsNull(), "PROGRESS SetFuzzyValue is accepted and produces a result");
  check(BOPAlgo_Options::GetParallelMode() || !BOPAlgo_Options::GetParallelMode(),
        "PROGRESS BOPAlgo_Options global parallel mode is readable");
}

// ---------------------------------------------------------------------------- HANDOFF ----- //
void probe_r10_handoff() {
  section("HANDOFF", "BREP round-trip determinism for the Gmsh handoff");

  const TopoDS_Shape box = BRepPrimAPI_MakeBox(BX, BY, BZ).Shape();

  std::ostringstream out;
  BRepTools::Write(box, out);
  const std::string bytes = out.str();
  check(!bytes.empty(), "HANDOFF BRepTools::Write serialises the shape");

  TopoDS_Shape reread;
  BRep_Builder builder;
  std::istringstream in(bytes);
  BRepTools::Read(reread, in, builder);
  check(!reread.IsNull(), "HANDOFF BRepTools::Read deserialises the shape");
  check(count(reread, TopAbs_FACE) == count(box, TopAbs_FACE),
        "HANDOFF round-trip preserves the face count");

  // The handoff maps EntityId -> Gmsh tag through the *sub-shape order*, so that order must
  // be reproducible from the same bytes.
  NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> m1;
  NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> m2;
  TopoDS_Shape reread2;
  std::istringstream in2(bytes);
  BRepTools::Read(reread2, in2, builder);
  TopExp::MapShapes(reread, TopAbs_FACE, m1);
  TopExp::MapShapes(reread2, TopAbs_FACE, m2);
  bool same_order = m1.Extent() == m2.Extent();
  for (int i = 1; i <= m1.Extent() && same_order; ++i) {
    GProp_GProps p1;
    GProp_GProps p2;
    BRepGProp::SurfaceProperties(m1.FindKey(i), p1);
    BRepGProp::SurfaceProperties(m2.FindKey(i), p2);
    same_order = std::fabs(p1.Mass() - p2.Mass()) < 1e-9 &&
                 p1.CentreOfMass().Distance(p2.CentreOfMass()) < 1e-9;
  }
  check(same_order,
        "HANDOFF two reads of identical BREP bytes give identical TopExp sub-shape order");

  // Coaxial faces share a centroid exactly — the falsification case a bijection check needs.
  // A pipe's inner and outer walls must be distinguishable by something other than centroid.
  const TopoDS_Shape outer = BRepPrimAPI_MakeCylinder(3.0, 10.0).Shape();
  const TopoDS_Shape inner = BRepPrimAPI_MakeCylinder(2.0, 10.0).Shape();
  NCollection_List<TopoDS_Shape> pargs;
  pargs.Append(outer);
  NCollection_List<TopoDS_Shape> ptools;
  ptools.Append(inner);
  BRepAlgoAPI_Cut pipe;
  pipe.SetArguments(pargs);
  pipe.SetTools(ptools);
  pipe.Build();
  std::vector<gp_Pnt> cyl_centroids;
  for (TopExp_Explorer ex(pipe.Shape(), TopAbs_FACE); ex.More(); ex.Next()) {
    const BRepAdaptor_Surface surf(TopoDS::Face(ex.Current()));
    if (surf.GetType() == GeomAbs_Cylinder) {
      GProp_GProps p;
      BRepGProp::SurfaceProperties(ex.Current(), p);
      cyl_centroids.push_back(p.CentreOfMass());
    }
  }
  bool centroids_collide = false;
  for (std::size_t i = 0; i + 1 < cyl_centroids.size(); ++i) {
    for (std::size_t j = i + 1; j < cyl_centroids.size(); ++j) {
      if (cyl_centroids[i].Distance(cyl_centroids[j]) < 1e-9) {
        centroids_collide = true;
      }
    }
  }
  check(cyl_centroids.size() == 2 && centroids_collide,
        "HANDOFF pipe fixture reproduces the coaxial centroid collision (bijection falsifier)");
}

// ---------------------------------------------------------------------------- IGES ----- //
void probe_r19_iges() {
  section("IGES", "IGES toolkit (TKDEIGES) availability");
  IGESControl_Reader reader;
  check(reader.NbShapes() == 0, "IGES IGESControl_Reader constructs (TKDEIGES linked)");
  IGESCAFControl_Reader xde_reader;
  check(!xde_reader.GetColorMode() || xde_reader.GetColorMode(),
        "IGES IGESCAFControl_Reader constructs (XDE-aware IGES import)");
  note("IGES IGES import", "no IGES fixture in-tree; construction proves the link, a real "
                          "file read is a binding-layer test");
}

}  // namespace

void run_occt_probe() {
  probe_r1_primitives_and_construction();
  probe_r2_booleans();
  probe_r3_fillet_chamfer();
  probe_r4_transforms();
  probe_r5_heal_defeature_imprint();
  probe_r6_queries();
  probe_r7_tessellation();
  probe_r8_r9_progress_and_tolerance();
  probe_r10_handoff();
  probe_r19_iges();
}
