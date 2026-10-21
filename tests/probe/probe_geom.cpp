// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH v2 capability probe — the geometry libraries: SALOME common_geometry_lib
// (GEOMAlgo, BlockFix), GEOM's ShHealOper, and FreeCAD's PlaneGCS with Eigen 3.4.1. Each is
// linked statically, from the same staged sources and patches that _core links, and each
// case checks a result whose answer is known:
//
//   GA1  GEOMAlgo_Splitter cuts a 2 x 1 x 1 box by the plane x = 0.5: two solids, of
//        volume 0.5 and 1.5 (planar faces, so exact to round-off).
//   BF1  BlockFix_BlockFixAPI on two unit boxes fused side by side: the fuse leaves the four
//        long sides split in two (10 faces); BlockFix merges each pair back (6 faces) and
//        keeps the volume 2.
//   SH1  ShHealOper_ShapeProcess with its operators and parameters set by SetOperators and
//        SetParameter, so no ShHealing resource file is read: it runs on a 1 x 2 x 3 box and
//        returns one valid solid of volume 6, and GetOperators and GetParameter read back
//        what was set.
//   GC1  PlaneGCS solves a triangle whose sides are fixed at 3, 4 and 5: the angle opposite
//        the side of 5 is a right angle (the converse of Pythagoras), to 1e-12 rad. The
//        diagnosis, whose two QR decompositions patches/planegcs/ runs on this thread,
//        reports no degree of freedom left.

#include "probe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepGProp.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <GProp_GProps.hxx>
#include <NCollection_List.hxx>
#include <TColStd_SequenceOfAsciiString.hxx>
#include <TCollection_AsciiString.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <NCollection_IndexedMap.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Dir.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt.hxx>

#include <BlockFix_BlockFixAPI.hxx>
#include <GEOMAlgo_Splitter.hxx>
#include <ShHealOper_ShapeProcess.hxx>

#include <GCS.h>

namespace {

using probe::check;
using probe::check_close;
using probe::section;

int count_of(const TopoDS_Shape& s, TopAbs_ShapeEnum kind) {
  NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> m;
  TopExp::MapShapes(s, kind, m);
  return m.Extent();
}

double volume_of(const TopoDS_Shape& s) {
  GProp_GProps props;
  BRepGProp::VolumeProperties(s, props);
  return props.Mass();
}

void probe_geomalgo_splitter() {
  section("GA1", "GEOMAlgo_Splitter: a box cut by a plane face");
  const TopoDS_Shape box = BRepPrimAPI_MakeBox(2.0, 1.0, 1.0).Shape();
  // The face covers the box's cross-section x = 0.5 with a margin of 1 on each side.
  const TopoDS_Shape plane =
      BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0.5, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), -1.0,
                              2.0, -1.0, 2.0)
          .Face();
  GEOMAlgo_Splitter splitter;
  splitter.AddArgument(box);
  splitter.AddTool(plane);
  splitter.SetLimit(TopAbs_SOLID);
  splitter.SetRunParallel(false);
  splitter.SetNonDestructive(true);
  splitter.Perform();
  check(!splitter.HasErrors(), "GEOMAlgo_Splitter::Perform reports no error");
  const TopoDS_Shape result = splitter.Shape();
  check(!result.IsNull(), "GEOMAlgo_Splitter returns a shape");
  if (result.IsNull()) {
    return;
  }
  check(count_of(result, TopAbs_SOLID) == 2, "the box is cut into 2 solids");
  std::vector<double> volumes;
  for (TopExp_Explorer ex(result, TopAbs_SOLID); ex.More(); ex.Next()) {
    volumes.push_back(volume_of(ex.Current()));
  }
  if (volumes.size() == 2) {
    const double small = std::min(volumes[0], volumes[1]);
    const double large = std::max(volumes[0], volumes[1]);
    check_close(small, 0.5, 1e-12, "the solid on x < 0.5 has volume 0.5 x 1 x 1");
    check_close(large, 1.5, 1e-12, "the solid on x > 0.5 has volume 1.5 x 1 x 1");
  }
  check(BRepCheck_Analyzer(result).IsValid(), "the split result is valid");
}

void probe_blockfix() {
  section("BF1", "BlockFix_BlockFixAPI: two fused boxes back to 6 faces");
  const TopoDS_Shape a = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 1.0, 1.0, 1.0).Shape();
  const TopoDS_Shape b = BRepPrimAPI_MakeBox(gp_Pnt(1.0, 0.0, 0.0), 1.0, 1.0, 1.0).Shape();
  BRepAlgoAPI_Fuse fuse;
  NCollection_List<TopoDS_Shape> args;
  NCollection_List<TopoDS_Shape> tools;
  args.Append(a);
  tools.Append(b);
  fuse.SetArguments(args);
  fuse.SetTools(tools);
  fuse.SetRunParallel(false);
  fuse.SetNonDestructive(true);
  fuse.Build();
  check(fuse.IsDone() && !fuse.HasErrors(), "BRepAlgoAPI_Fuse of two touching boxes is done");
  const TopoDS_Shape fused = fuse.Shape();
  check(count_of(fused, TopAbs_FACE) == 10,
        "the fuse leaves 10 faces (4 long sides split in two, 2 ends)");

  Handle(BlockFix_BlockFixAPI) fixer = new BlockFix_BlockFixAPI();
  fixer->SetShape(fused);
  fixer->Tolerance() = 1e-7;
  fixer->OptimumNbFaces() = 6;
  fixer->Perform();
  const TopoDS_Shape fixed = fixer->Shape();
  check(!fixed.IsNull(), "BlockFix_BlockFixAPI returns a shape");
  if (fixed.IsNull()) {
    return;
  }
  check(count_of(fixed, TopAbs_SOLID) == 1, "BlockFix keeps one solid");
  check(count_of(fixed, TopAbs_FACE) == 6, "BlockFix merges the split sides: 6 faces");
  check_close(volume_of(fixed), 2.0, 1e-12, "BlockFix keeps the volume 2");
  check(BRepCheck_Analyzer(fixed).IsValid(), "the BlockFix result is valid");
}

void probe_shapeprocess() {
  section("SH1", "ShHealOper_ShapeProcess: explicit operators, no resource file");
  // The operators read a resource file only through these two environment variables
  // (Resource_Manager::GetResourcePath). The probe checks that neither is set, so the
  // operators and parameters below are the whole input.
  check(std::getenv("CSF_ShHealingDefaults") == nullptr &&
            std::getenv("CSF_ShHealingUserDefaults") == nullptr,
        "no ShHealing resource file is named in the environment");
  const TopoDS_Shape box = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape();

  ShHealOper_ShapeProcess healer;
  TColStd_SequenceOfAsciiString operators;
  operators.Append("FixShape");
  operators.Append("SameParameter");
  healer.SetOperators(operators);
  healer.SetParameter("FixShape.Tolerance3d", "1e-07");
  healer.SetParameter("FixShape.MaxTolerance3d", "1e-03");
  healer.SetParameter("FixShape.MinTolerance3d", "1e-07");
  healer.SetParameter("SameParameter.Force", "0");
  healer.SetParameter("SameParameter.Tolerance3d", "1e-07");

  TColStd_SequenceOfAsciiString read_back;
  check(healer.GetOperators(read_back) && read_back.Length() == 2 &&
            read_back.Value(1) == "FixShape" && read_back.Value(2) == "SameParameter",
        "GetOperators reads back the 2 operators set by SetOperators");
  TCollection_AsciiString tolerance;
  check(healer.GetParameter("FixShape.MaxTolerance3d", tolerance) &&
            tolerance.IsRealValue() && std::fabs(tolerance.RealValue() - 1e-3) < 1e-18,
        "GetParameter reads back FixShape.MaxTolerance3d = 1e-3");

  TopoDS_Shape healed;
  healer.Perform(box, healed);
  check(!healed.IsNull(), "ShHealOper_ShapeProcess::Perform returns a shape");
  if (healed.IsNull()) {
    return;
  }
  check(count_of(healed, TopAbs_SOLID) == 1, "the healed box is one solid");
  check(count_of(healed, TopAbs_FACE) == 6, "the healed box has 6 faces");
  check_close(volume_of(healed), 6.0, 1e-12, "the healed box keeps the volume 1 x 2 x 3");
  check(BRepCheck_Analyzer(healed).IsValid(), "the healed box is valid");
  check(!healed.IsSame(box), "Perform works on a copy: the input box is not the result");
}

void probe_planegcs() {
  section("GC1", "PlaneGCS: the 3-4-5 triangle has a right angle");
  // A fixed at the origin; B on the x axis (its y fixed); C free. The unknowns are B.x,
  // C.x and C.y; the three side lengths are driving constants. AB = 3, BC = 4, CA = 5, so
  // the angle at B, opposite CA, is the right angle.
  double ax = 0.0, ay = 0.0;
  double bx = 2.5, by = 0.0;
  double cx = 2.0, cy = 3.0;
  double ab = 3.0, bc = 4.0, ca = 5.0;
  GCS::Point pa;
  pa.x = &ax;
  pa.y = &ay;
  GCS::Point pb;
  pb.x = &bx;
  pb.y = &by;
  GCS::Point pc;
  pc.x = &cx;
  pc.y = &cy;

  GCS::System system;
  system.addConstraintP2PDistance(pa, pb, &ab, 1);
  system.addConstraintP2PDistance(pb, pc, &bc, 2);
  system.addConstraintP2PDistance(pc, pa, &ca, 3);
  GCS::VEC_pD unknowns = {&bx, &cx, &cy};
  system.declareUnknowns(unknowns);
  system.initSolution(GCS::DogLeg);
  const int status = system.solve(true, GCS::DogLeg);
  check(status == GCS::Success, "the DogLeg solver finds a solution that zeroes the error");
  system.applySolution();

  check_close(std::hypot(bx - ax, by - ay), 3.0, 1e-12, "|AB| = 3");
  check_close(std::hypot(cx - bx, cy - by), 4.0, 1e-12, "|BC| = 4");
  check_close(std::hypot(ax - cx, ay - cy), 5.0, 1e-12, "|CA| = 5");
  // The angle at B from atan2(|BA x BC|, BA . BC), which keeps full precision near pi/2.
  const double bax = ax - bx, bay = ay - by;
  const double bcx = cx - bx, bcy = cy - by;
  const double angle = std::atan2(std::fabs(bax * bcy - bay * bcx), bax * bcx + bay * bcy);
  check_close(angle, 1.57079632679489661923, 1e-12,
              "the angle at B, opposite the side of 5, is pi/2 rad");

  const int dofs = system.diagnose(GCS::DogLeg);
  check(dofs == 0, "diagnose: the triangle with B on the x axis has 0 degrees of freedom");
  GCS::VEC_I conflicting;
  system.getConflicting(conflicting);
  check(conflicting.empty(), "diagnose: no conflicting constraint");
}

}  // namespace

void run_geom_probe() {
  probe_geomalgo_splitter();
  probe_blockfix();
  probe_shapeprocess();
  probe_planegcs();
}
