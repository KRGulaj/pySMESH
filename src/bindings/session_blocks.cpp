// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH binding — Session: GEOM's blocks (hexahedral decomposition).
//
//   make_quad, make_quad_from_vertices  GEOM's MakeQuad, MakeQuad2Edges, MakeQuad4Vertices
//   make_hexa                           GEOM's MakeHexa, MakeHexa2Faces
//   check_blocks                        GEOM's CheckCompoundOfBlocks
//   propagation_chain                   GEOM's Propagate, for the chain of one edge
//   opposite_face, face_by_edges,       GEOM's GetOppositeFace, GetFaceByEdges,
//   find_block_near                     GetBlockNearPoint
//   multi_transform                     GEOM's MakeMultiTransformation1D and 2D, by vectors
//
// The GEOM code these need is GEOMImpl_BlockDriver's quad and hexa branches and the
// topology searches of GEOMImpl_IBlocksOperations, copied here without the driver shell,
// and GEOMImpl_Block6Explorer, copied whole (geom_block6.cpp). All are SALOME GEOM tag
// V9_16_0 (7d0a18e1554c25bca2c2d9882e0621ea3f7f9c81), Copyright (C) 2007-2026 CEA, EDF,
// OPEN CASCADE, Copyright (C) 2003-2007 OPEN CASCADE, EADS/CCR, LIP6, CEA/DEN, CEDRAT, EDF
// R&D, LEG, PRINCIPIA R&D, BUREAU VERITAS, LGPL-2.1 or later, used here under LGPL-2.1,
// and recorded in PROVENANCE.md. Every copied block names its origin, and records the
// history the id registry carries ids across.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include "geom_block6.hpp"

#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepLib.hxx>
#include <BRepTools_Quilt.hxx>
#include <BlockFix_CheckTool.hxx>
#include <GC_MakeLine.hxx>
#include <GEOMAlgo_CoupleOfShapes.hxx>
#include <GEOMAlgo_GlueAnalyser.hxx>
#include <GEOMAlgo_Gluer2.hxx>
#include <GEOMAlgo_ListOfCoupleOfShapes.hxx>
#include <Geom_Line.hxx>
#include <Message_ProgressScope.hxx>
#include <ShHealOper_Sewing.hxx>
#include <TColStd_Array2OfInteger.hxx>
#include <TColStd_MapOfInteger.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_MapOfShape.hxx>

namespace pysmesh {

namespace geomcopy {

// GEOM's default ShHealOper_ShapeProcess, as GEOMImpl_Block6Explorer::MakeAnyFace runs it:
// the sequence and parameters of GEOM's resource file resources/ShHealing (V9_16_0), with
// FixFaceSize.Tolerance and DropSmallEdges.Tolerance3d set to 1e-05 as MakeAnyFace sets
// them. Parameters the file does not name keep OCCT's own defaults there and here alike.
// compose_shape_process reads no resource file, so a host's CSF_ShHealing* variables change
// nothing.
Standard_Boolean HealAsGeom(const TopoDS_Shape& theFace, TopoDS_Shape& theHealed,
                            Handle(BRepTools_History)& theHistory) {
  static const std::vector<std::pair<std::string, std::string>> kGeomShHealing = {
      {"FixShape.Tolerance3d", "1.e-7"},
      {"FixShape.MaxTolerance3d", "1"},
      {"FixShape.MinTolerance3d", "1.e-7"},
      {"FixShape.FixFreeShellMode", "-1"},
      {"FixShape.FixFreeFaceMode", "-1"},
      {"FixShape.FixFreeWireMode", "-1"},
      {"FixShape.FixSameParameterMode", "-1"},
      {"FixShape.FixSolidMode", "-1"},
      {"FixShape.FixShellMode", "-1"},
      {"FixShape.FixFaceMode", "-1"},
      {"FixShape.FixWireMode", "-1"},
      {"FixShape.FixOrientationMode", "-1"},
      {"FixShape.FixMissingSeamMode", "-1"},
      {"FixShape.FixSmallAreaWireMode", "-1"},
      {"FixShape.ModifyTopologyMode", "0"},
      {"FixShape.ModifyGeometryMode", "1"},
      {"FixShape.ClosedWireMode", "1"},
      {"FixShape.PreferencePCurveMode", "1"},
      {"FixShape.FixReorderMode", "-1"},
      {"FixShape.FixSmallMode", "-1"},
      {"FixShape.FixConnectedMode", "-1"},
      {"FixShape.FixEdgeCurvesMode", "-1"},
      {"FixShape.FixDegeneratedMode", "-1"},
      {"FixShape.FixLackingMode", "-1"},
      {"FixShape.FixSelfIntersectionMode", "-1"},
      {"FixShape.FixReversed2dMode", "-1"},
      {"FixShape.FixRemovePCurveMode", "-1"},
      {"FixShape.FixRemoveCurve3dMode", "-1"},
      {"FixShape.FixAddPCurveMode", "-1"},
      {"FixShape.FixAddCurve3dMode", "-1"},
      {"FixShape.FixSeamMode", "-1"},
      {"FixShape.FixShiftedMode", "-1"},
      {"FixShape.FixEdgeSameParameterMode", "0"},
      {"FixShape.FixSelfIntersectingEdgeMode", "-1"},
      {"FixShape.FixIntersectingEdgesMode", "-1"},
      {"FixShape.FixNonAdjacentIntersectingEdgesMode", "-1"},
      {"FixShape.FixIntersectingWiresMode", "-1"},
      {"FixShape.FixLoopWiresMode", "-1"},
      {"FixShape.CreateOpenSolidMode", "0"},
      {"FixFaceSize.Tolerance", "1e-05"},
      {"DropSmallEdges.Tolerance3d", "1e-05"},
      {"SameParameter.Force", "0"},
      {"SameParameter.Tolerance3d", "1.e-7"},
  };
  TopoDS_Compound input;
  BRep_Builder builder;
  builder.MakeCompound(input);
  builder.Add(input, theFace);
  const session::ShapeProcessRun run = session::compose_shape_process(
      input, "FixShape FixFaceSize DropSmallEdges SameParameter", kGeomShHealing,
      Message_ProgressRange());
  if (run.cancelled || !run.alarms.empty() || !run.done || run.out.IsNull()) {
    return Standard_False;
  }
  TopoDS_Iterator it(run.out);
  if (!it.More()) {
    return Standard_False;
  }
  theHealed = it.Value();
  theHistory = run.hist;
  return Standard_True;
}

}  // namespace geomcopy

namespace session {

Handle(BRepTools_History) compose_histories(
    const TopoDS_Shape& input, const std::vector<Handle(BRepTools_History)>& steps) {
  // A shape's images through one step: its Modified images, or itself when the step does
  // not name it, or nothing when the step removes it.
  const auto through = [](const Handle(BRepTools_History)& step, const TopoDS_Shape& s,
                          std::vector<TopoDS_Shape>& out) {
    if (step->IsRemoved(s)) {
      return;
    }
    const NCollection_List<TopoDS_Shape>& mod = step->Modified(s);
    if (mod.IsEmpty()) {
      append_unique(out, s);
      return;
    }
    for (const TopoDS_Shape& m : mod) {
      append_unique(out, m);
    }
  };
  Handle(BRepTools_History) out = new BRepTools_History;
  for (TopAbs_ShapeEnum k : kEntityKinds) {
    ShapeSet subs;
    TopExp::MapShapes(input, k, subs);
    for (int i = 1; i <= subs.Extent(); ++i) {
      const TopoDS_Shape& s = subs.FindKey(i);
      std::vector<TopoDS_Shape> modified{s};
      std::vector<TopoDS_Shape> generated;
      for (const Handle(BRepTools_History)& step : steps) {
        if (step.IsNull()) {
          continue;
        }
        std::vector<TopoDS_Shape> next_generated;
        for (const TopoDS_Shape& g : generated) {
          through(step, g, next_generated);
        }
        for (const TopoDS_Shape& m : modified) {
          for (const TopoDS_Shape& g : step->Generated(m)) {
            append_unique(next_generated, g);
          }
        }
        for (const TopoDS_Shape& g : generated) {
          for (const TopoDS_Shape& gg : step->Generated(g)) {
            append_unique(next_generated, gg);
          }
        }
        std::vector<TopoDS_Shape> next_modified;
        for (const TopoDS_Shape& m : modified) {
          through(step, m, next_modified);
        }
        modified = std::move(next_modified);
        generated = std::move(next_generated);
      }
      const bool kept = modified.size() == 1 && modified.front().IsSame(s);
      if (!kept) {
        for (const TopoDS_Shape& m : modified) {
          if (m.ShapeType() == k) {
            out->AddModified(s, m);
          }
        }
      }
      for (const TopoDS_Shape& g : generated) {
        bool is_image = false;
        for (const TopoDS_Shape& m : modified) {
          is_image = is_image || m.IsSame(g);
        }
        if (!is_image && !g.IsSame(s)) {
          out->AddGenerated(s, g);
        }
      }
    }
  }
  return out;
}

namespace {

using geomcopy::GEOMImpl_Block6Explorer;

// The ids, ascending and each once, as "1, 2, 3".
std::string id_text(std::vector<EntityId> ids) {
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  std::string out;
  for (EntityId id : ids) {
    out += (out.empty() ? "" : ", ") + std::to_string(id);
  }
  return out;
}

// A compound of shapes, in the order given.
TopoDS_Compound bundle_of(const std::vector<TopoDS_Shape>& shapes) {
  TopoDS_Compound out;
  BRep_Builder builder;
  builder.MakeCompound(out);
  for (const TopoDS_Shape& s : shapes) {
    builder.Add(out, s);
  }
  return out;
}

// Each shape's copy in a BRepBuilderAPI_Copy of their compound, in order. The block
// builders edit tolerances and curves in place (BRepLib::SameParameter, LimitTolerance),
// and the session's shapes are shared by its snapshots, so they build on copies; the copy's
// history carries every id onto them.
std::vector<TopoDS_Shape> copies_of(const BRepBuilderAPI_Copy& copier,
                                    const std::vector<TopoDS_Shape>& shapes) {
  std::vector<TopoDS_Shape> out;
  for (const TopoDS_Shape& s : shapes) {
    out.push_back(copier.ModifiedShape(s));
  }
  return out;
}

// What a Standard_Failure says, for a refusal.
std::string failure_text(const std::exception& e) {
  const std::string what = e.what();
  return what.empty() ? std::string("an OCCT exception with no message") : what;
}

// A body made only of free vertices: a VERTEX body.
void require_vertex_body(const char* op, const TopoDS_Shape& body, EntityId id) {
  if (body.ShapeType() != TopAbs_VERTEX) {
    throw PysmeshError(std::string("Session.") + op + ": vertex " + std::to_string(id) +
                       " belongs to a " + TopAbs::ShapeTypeToString(body.ShapeType()) +
                       " body. This operation consumes its inputs, so it accepts only free "
                       "vertex bodies.");
  }
}

// A body of free faces: a FACE or SHELL body.
void require_face_body(const char* op, const TopoDS_Shape& body, EntityId id) {
  if (body.ShapeType() != TopAbs_FACE && body.ShapeType() != TopAbs_SHELL) {
    throw PysmeshError(std::string("Session.") + op + ": face " + std::to_string(id) +
                       " belongs to a " + TopAbs::ShapeTypeToString(body.ShapeType()) +
                       " body. This operation consumes its inputs, so it accepts only free "
                       "faces and shells.");
  }
}

// ---- GEOMImpl_BlockDriver::Execute, the quad branches -------------------------------- //

// What one branch built: the face, its history from the branch's inputs, or the reason
// GEOM's driver raises.
struct QuadBuild {
  TopoDS_Shape face;
  Handle(BRepTools_History) hist = new BRepTools_History;  // the wire's step
  Handle(BRepTools_History) face_hist;                      // the face's step
  std::string failure;
};

// BLOCK_FACE_FOUR_EDGES, copied: the four edges must have 4 distinct corners
// (BRepTools::Compare), the wire is built in the order of their connexity, it must be
// closed, and GEOMImpl_Block6Explorer::MakeFace builds a face on any surface under it.
// pySMESH: the edges are added to the wire one by one, as the four-edge constructor adds
// them, so the image of each is read, and Standard_Failure::Raise is a failure text.
QuadBuild quad_four_edges(const TopoDS_Edge (&anArgs)[4]) {
  QuadBuild out;
  // count corner vertices
  TopTools_MapOfShape aVertMap;
  for (Standard_Integer ii = 0; ii < 4; ii++) {
    TopoDS_Edge anEdge = anArgs[ii];
    TopoDS_Vertex V1, V2;
    TopExp::Vertices(anEdge, V1, V2, Standard_True);
    if (V1.IsNull() || V2.IsNull()) {
      out.failure = "Bad edge for face construction: vertex is not defined";
      return out;
    }
    if (BRepTools::Compare(V1,V2)) {
      out.failure = "Edge ends are too close";
      return out;
    }
    Standard_Boolean isCoin1 = Standard_False, isCoin2 = Standard_False;
    TopTools_MapIteratorOfMapOfShape anIter (aVertMap);
    for (; anIter.More(); anIter.Next()) {
      TopoDS_Vertex V = TopoDS::Vertex(anIter.Key());
      if (BRepTools::Compare(V,V1)) isCoin1 = Standard_True;
      if (BRepTools::Compare(V,V2)) isCoin2 = Standard_True;
    }
    if (!isCoin1) aVertMap.Add(V1);
    if (!isCoin2) aVertMap.Add(V2);
  }
  if (aVertMap.Extent() != 4) {
    out.failure = "The edges must form a closed wire";
    return out;
  }

  TopoDS_Edge anEdge1 = anArgs[0];
  TopoDS_Edge anEdge2 = anArgs[1];
  TopoDS_Edge anEdge3 = anArgs[2];
  TopoDS_Edge anEdge4 = anArgs[3];

  // check, if anEdge1 has common/coincident vertex with anEdge2,
  Standard_Boolean isConnected12 = Standard_False;
  TopoDS_Vertex V11, V12, V21, V22;
  TopExp::Vertices(anEdge1, V11, V12, Standard_True);
  TopExp::Vertices(anEdge2, V21, V22, Standard_True);
  if (BRepTools::Compare(V11,V21) || BRepTools::Compare(V11,V22) ||
      BRepTools::Compare(V12,V21) || BRepTools::Compare(V12,V22)) {
    // the edges have common vertex
    isConnected12 = Standard_True;
  }

  // build wire in right order, corresponding to edges connexity
  BRepBuilderAPI_MakeWire MW;
  const TopoDS_Edge anOrder[4] = {anEdge1, isConnected12 ? anEdge2 : anEdge3,
                                  isConnected12 ? anEdge3 : anEdge2, anEdge4};
  for (const TopoDS_Edge& anE : anOrder) {
    MW.Add(anE);
    if (MW.IsDone())
      geomcopy::RecordWireImage(anE, MW.Edge(), out.hist);
  }

  if (!MW.IsDone()) {
    out.failure = "Impossible to build a connected wire from the given edges";
    return out;
  }
  TopoDS_Wire aWire = MW;

  // check the wire closure
  TopoDS_Vertex aV1, aV2;
  TopExp::Vertices(aWire, aV1, aV2);
  if ( !aV1.IsNull() && !aV2.IsNull() && aV1.IsSame(aV2) )
    aWire.Closed( true );

  if (!aWire.Closed()) {
    out.failure = "Impossible to build a closed wire from the given edges";
    return out;
  }

  // try to build face on the wire
  GEOMImpl_Block6Explorer::MakeFace(aWire, Standard_False, out.face, out.face_hist);
  if (out.face.IsNull()) {
    out.failure = "Face construction failed";
    return out;
  }
  return out;
}

// BLOCK_FACE_TWO_EDGES, copied: two straight edges link the ends of the given edges, the
// pairing of the ends being the one with the shorter sum of lengths, and the face is built
// on the four. pySMESH: each linking edge is generated from the two given edges, the
// wire's images are read, and Standard_Failure::Raise is a failure text.
QuadBuild quad_two_edges(const TopoDS_Edge& anEdge1, const TopoDS_Edge& anEdge2) {
  QuadBuild out;
  const Standard_Real prec = Precision::Confusion();

  if (anEdge1.IsSame(anEdge2)) {
    out.failure = "The edges must be different";
    return out;
  }

  // create two edges, linking ends of the given edges
  TopoDS_Vertex V11, V12, V21, V22;
  TopExp::Vertices(anEdge1, V11, V12, Standard_False);
  TopExp::Vertices(anEdge2, V21, V22, Standard_False);
  if (V11.IsNull() || V12.IsNull() ||
      V21.IsNull() || V22.IsNull()) {
    out.failure = "Bad edge for face construction: vertex is not defined";
    return out;
  }

  BRepAdaptor_Curve C1 (anEdge1);
  BRepAdaptor_Curve C2 (anEdge2);
  gp_Pnt P11, P12, P21, P22;

  // Mantis issue 0020599: Creation of a quadrangle face from 2 edges: SIGSEGV
  P11 = C1.Value(C1.FirstParameter());
  P12 = C1.Value(C1.LastParameter());
  P21 = C2.Value(C2.FirstParameter());
  P22 = C2.Value(C2.LastParameter());

  if (P11.Distance(P21) < prec || P12.Distance(P22) < prec ||
      P11.Distance(P22) < prec || P12.Distance(P21) < prec) {
    out.failure = "Given edges have too close ends";
    return out;
  }

  Standard_Real per11 = P11.Distance(P21) + P12.Distance(P22);
  Standard_Real per12 = P11.Distance(P22) + P12.Distance(P21);

  BRep_Builder BB;

  TopoDS_Edge anEdge3;
  TopoDS_Edge anEdge4;

  // Mantis issue 0020599: Creation of a quadrangle face from 2 edges: SIGSEGV
  if (per11 < per12) {
    Handle(Geom_Line) Line1 = GC_MakeLine(P11, P21).Value();
    Handle(Geom_Line) Line2 = GC_MakeLine(P12, P22).Value();

    BB.MakeEdge(anEdge3, Line1, Precision::Confusion());
    BB.Range(anEdge3, 0., P11.Distance(P21));
    BB.Add(anEdge3, V11.Oriented(TopAbs_FORWARD));
    BB.Add(anEdge3, V21.Oriented(TopAbs_REVERSED));

    BB.MakeEdge(anEdge4, Line2, Precision::Confusion());
    BB.Range(anEdge4, 0., P12.Distance(P22));
    BB.Add(anEdge4, V12.Oriented(TopAbs_FORWARD));
    BB.Add(anEdge4, V22.Oriented(TopAbs_REVERSED));
  }
  else {
    Handle(Geom_Line) Line1 = GC_MakeLine(P11, P22).Value();
    Handle(Geom_Line) Line2 = GC_MakeLine(P12, P21).Value();

    BB.MakeEdge(anEdge3, Line1, Precision::Confusion());
    BB.Range(anEdge3, 0., P11.Distance(P22));
    BB.Add(anEdge3, V11.Oriented(TopAbs_FORWARD));
    BB.Add(anEdge3, V22.Oriented(TopAbs_REVERSED));

    BB.MakeEdge(anEdge4, Line2, Precision::Confusion());
    BB.Range(anEdge4, 0., P12.Distance(P21));
    BB.Add(anEdge4, V12.Oriented(TopAbs_FORWARD));
    BB.Add(anEdge4, V21.Oriented(TopAbs_REVERSED));
  }
  for (const TopoDS_Edge& aLink : {anEdge3, anEdge4}) {
    out.hist->AddGenerated(anEdge1, aLink);
    out.hist->AddGenerated(anEdge2, aLink);
  }

  // build a wire
  BRepBuilderAPI_MakeWire MW;
  for (const TopoDS_Edge& anE : {anEdge1, anEdge3, anEdge2, anEdge4}) {
    MW.Add(anE);
    if (MW.IsDone())
      geomcopy::RecordWireImage(anE, MW.Edge(), out.hist);
  }
  if (!MW.IsDone()) {
    out.failure = "Wire construction failed";
    return out;
  }

  TopoDS_Wire aWire = MW;

  TopoDS_Vertex aV1, aV2;
  TopExp::Vertices(aWire, aV1, aV2);
  if ( !aV1.IsNull() && !aV2.IsNull() && aV1.IsSame(aV2) )
    aWire.Closed( true );

  if (!aWire.Closed()) {
    out.failure = "Impossible to build a closed wire from the given edges";
    return out;
  }

  // try to build face on the wire
  GEOMImpl_Block6Explorer::MakeFace(aWire, Standard_False, out.face, out.face_hist);
  if (out.face.IsNull()) {
    out.failure = "Face construction failed";
    return out;
  }
  return out;
}

// BLOCK_FACE_FOUR_PNT, copied: the corners are ordered by the shortest perimeter of the
// three quadrangles they can form, and the face is built on their closed polygon.
// pySMESH: each polygon edge is generated from its two corners and the face from all four,
// and Standard_Failure::Raise is a failure text.
QuadBuild quad_four_vertices(TopoDS_Vertex V1, TopoDS_Vertex V2, TopoDS_Vertex V3,
                             TopoDS_Vertex V4) {
  QuadBuild out;
  const Standard_Real prec = Precision::Confusion();

  gp_Pnt P1 = BRep_Tool::Pnt(V1);
  gp_Pnt P2 = BRep_Tool::Pnt(V2);
  gp_Pnt P3 = BRep_Tool::Pnt(V3);
  gp_Pnt P4 = BRep_Tool::Pnt(V4);

  if (P1.Distance(P2) < prec || P1.Distance(P3) < prec ||
      P1.Distance(P4) < prec || P2.Distance(P3) < prec ||
      P2.Distance(P4) < prec || P3.Distance(P4) < prec) {
    out.failure = "Four not coincident points must be given";
    return out;
  }

  // calculate perimeters
  Standard_Real per1234 = P1.Distance(P2) + P2.Distance(P3) +
                          P3.Distance(P4) + P4.Distance(P1);
  Standard_Real per1243 = P1.Distance(P2) + P2.Distance(P4) +
                          P4.Distance(P3) + P3.Distance(P1);
  Standard_Real per1324 = P1.Distance(P3) + P3.Distance(P2) +
                          P2.Distance(P4) + P4.Distance(P1);

  // order vertices
  if (per1243 < per1234 && per1243 < per1324) {
    TopoDS_Vertex Vtmp = V3;
    V3 = V4;
    V4 = Vtmp;
  } else if (per1324 < per1234 && per1324 < per1243) {
    TopoDS_Vertex Vtmp = V3;
    V3 = V2;
    V2 = Vtmp;
  } else {
  }

  // build wire
  BRepBuilderAPI_MakePolygon aMkPoly (V1, V2, V3, V4, Standard_True);
  if (!aMkPoly.IsDone()) {
    out.failure = "Polygon construction failed";
    return out;
  }

  // try to build face on the wire
  aMkPoly.Close();
  GEOMImpl_Block6Explorer::MakeFace(aMkPoly, Standard_False, out.face, out.face_hist);
  if (out.face.IsNull()) {
    out.failure = "Face construction failed";
    return out;
  }
  for (TopExp_Explorer anExp (aMkPoly.Wire(), TopAbs_EDGE); anExp.More(); anExp.Next()) {
    for (TopoDS_Iterator aV (anExp.Current()); aV.More(); aV.Next())
      out.hist->AddGenerated(aV.Value(), anExp.Current());
  }
  for (const TopoDS_Vertex& aV : {V1, V2, V3, V4})
    out.hist->AddGenerated(aV, out.face);
  return out;
}

// ---- GEOMImpl_BlockDriver::Execute, the hexa branches -------------------------------- //

// BLOCK_SIX_FACES and BLOCK_TWO_FACES, copied: the faces (the two given ones and the four
// GEOMImpl_Block6Explorer builds between them) go into a BRepTools_Quilt; when they make
// more than one shell, ShHealOper_Sewing sews them at the largest face tolerance; the
// shell becomes a solid, reversed when the point at infinity is inside it, and
// BRepLib::SameParameter runs on it. pySMESH: the history of the quilt's copies, the
// explorer's new faces and edges and the sewing's context is recorded, the solid is
// generated from the faces, and Standard_Failure::Raise is a failure text.
struct HexaBuild {
  TopoDS_Shape solid;
  std::vector<Handle(BRepTools_History)> steps;  // in order
  std::string failure;
};

HexaBuild hexa_of(const std::vector<TopoDS_Shape>& anArgs) {
  HexaBuild out;
  BRep_Builder B;
  const Standard_Real prec = Precision::Confusion();
  TopoDS_Shape aShape;
  BRepTools_Quilt Glue;
  std::vector<TopoDS_Shape> aQuilted;
  const bool sixFaces = anArgs.size() == 6;

  if (sixFaces) {
    // Make block (hexahedral solid) from six faces
    for (const TopoDS_Shape& anArg : anArgs) {
      if (anArg.ShapeType() != TopAbs_FACE) {
        out.failure = "Shape for block construction is not a face";
        return out;
      }
      Glue.Add(anArg);
      aQuilted.push_back(anArg);
    }
  }
  else {
    // Make block (hexahedral solid) from two opposite faces
    if (anArgs[0].ShapeType() != TopAbs_FACE ||
        anArgs[1].ShapeType() != TopAbs_FACE) {
      out.failure = "Shape for block construction is not a face";
      return out;
    }

    // Get wires of the given faces
    TopExp_Explorer wires1 (anArgs[0], TopAbs_WIRE);
    TopExp_Explorer wires2 (anArgs[1], TopAbs_WIRE);
    if (!wires1.More() || !wires2.More()) {
      out.failure = "A face for the block has no wires";
      return out;
    }
    wires1.Next();
    wires2.Next();
    if (wires1.More() || wires2.More()) {
      out.failure = "A face for the block has more than one wire";
      return out;
    }

    GEOMImpl_Block6Explorer aBlockTool;
    aBlockTool.InitByTwoFaces(anArgs[0], anArgs[1]);

    // Construct the linking faces and add them in the gluing tool
    aQuilted = {anArgs[0],
                aBlockTool.GetFace(3, Standard_True),
                aBlockTool.GetFace(4, Standard_True),
                aBlockTool.GetFace(5, Standard_True),
                aBlockTool.GetFace(6, Standard_True),
                anArgs[1]};
    for (const TopoDS_Shape& aFace : aQuilted)
      Glue.Add(aFace);
    out.steps.push_back(aBlockTool.History());
  }

  // pySMESH: what the quilt copied.
  Handle(BRepTools_History) aQuiltHistory = new BRepTools_History;
  for (const TopoDS_Shape& aFace : aQuilted) {
    for (TopAbs_ShapeEnum aKind : {TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX}) {
      for (TopExp_Explorer anExp (aFace, aKind); anExp.More(); anExp.Next()) {
        if (Glue.IsCopied(anExp.Current()))
          aQuiltHistory->AddModified(anExp.Current(), Glue.Copy(anExp.Current()));
      }
    }
  }
  out.steps.push_back(aQuiltHistory);

  TopExp_Explorer exp (Glue.Shells(), TopAbs_SHELL);
  Standard_Integer ish = 0;
  TopTools_MapOfShape mapShape;
  for (; exp.More(); exp.Next()) {
    if (mapShape.Add(exp.Current())) {
      aShape = exp.Current();
      ish++;
    }
  }

  if (ish > 1) {
    aShape = Glue.Shells();
    Standard_Real aTol = prec; // Precision::Confusion()
    TopExp_Explorer expF (aShape, TopAbs_FACE);
    TopTools_MapOfShape mapF;
    TopoDS_Shell Shell;
    B.MakeShell(Shell);
    for (; expF.More(); expF.Next()) {
      if (mapF.Add(expF.Current())) {
        B.Add(Shell, expF.Current());
        Standard_Real aToler = BRep_Tool::Tolerance(TopoDS::Face(expF.Current()));
        if (aToler > aTol)
          aTol = aToler;
      }
    }
    ShHealOper_Sewing aHealer (Shell, aTol);
    // pySMESH: the sewing records the edges and vertices it replaces too (its own history
    // level, TopAbs_FACE by default). The extra records are replacements of input
    // sub-shapes that the sewn shape no longer holds, so ShHealOper's Apply over the sewn
    // shape changes nothing more.
    aHealer.SetHistoryLevel(TopAbs_VERTEX);
    if (aHealer.Perform()) {
      aShape = aHealer.GetResultShape();
      out.steps.push_back(aHealer.Context()->History());
    }
    else {
      out.failure = "Impossible to build a connected shell on the given faces";
      return out;
    }
  }

  if (sixFaces) {
    if (!aShape.Closed()) {
      out.failure = "Impossible to build a closed shell on the given faces";
      return out;
    }
  }

  if (aShape.ShapeType() != TopAbs_SHELL) {
    out.failure = std::string("the faces give a ") + TopAbs::ShapeTypeToString(aShape.ShapeType()) +
                  ", not one shell";
    return out;
  }

  TopoDS_Solid Sol;
  B.MakeSolid(Sol);
  B.Add(Sol, aShape);
  BRepClass3d_SolidClassifier SC (Sol);
  SC.PerformInfinitePoint(prec);
  if (SC.State() == TopAbs_IN) {
    B.MakeSolid(Sol);
    B.Add(Sol, aShape.Reversed());
  }
  aShape = Sol;
  BRepLib::SameParameter(aShape, 1.E-5, Standard_True);
  out.solid = aShape;
  Handle(BRepTools_History) aSolidHistory = new BRepTools_History;
  for (TopExp_Explorer aFace (out.solid, TopAbs_FACE); aFace.More(); aFace.Next())
    aSolidHistory->AddGenerated(aFace.Current(), out.solid);
  out.steps.push_back(aSolidHistory);
  return out;
}

}  // namespace

// ---- make_quad ---------------------------------------------------------------------- //

py::dict Session::make_quad(const std::vector<EntityId>& edge_ids) {
  const char* op = "make_quad";
  OpGuard guard(in_op_);
  const std::string named = id_text(edge_ids);
  if (edge_ids.size() != 4 && edge_ids.size() != 2) {
    throw PysmeshError("Session.make_quad: " + std::to_string(edge_ids.size()) +
                           " edges are named (" + named +
                           "); a quadrangle takes 4 edges, or 2 opposite edges.",
                       "", ids_as_int(edge_ids));
  }
  const std::vector<TopoDS_Shape> edges = edges_of(op, edge_ids);
  if (edges.size() != edge_ids.size()) {
    throw PysmeshError("Session.make_quad: the edges " + named + " name one edge twice.", "",
                       ids_as_int(edge_ids));
  }
  const std::vector<TopoDS_Shape> owners = owner_bodies(op, edge_ids);
  for (const TopoDS_Shape& b : owners) {
    require_curve_body(op, b);
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(owners);
  const TopoDS_Compound input = bundle_of(edges);

  QuadBuild built;
  Handle(BRepTools_History) hist;
  std::string thrown;
  {
    py::gil_scoped_release release;
    try {
      BRepBuilderAPI_Copy copier(input, /*copyGeom=*/Standard_True, /*copyMesh=*/Standard_False);
      hist = history_of(input, copier);
      const std::vector<TopoDS_Shape> c = copies_of(copier, edges);
      if (c.size() == 4) {
        const TopoDS_Edge args[4] = {TopoDS::Edge(c[0]), TopoDS::Edge(c[1]),
                                     TopoDS::Edge(c[2]), TopoDS::Edge(c[3])};
        built = quad_four_edges(args);
      } else {
        built = quad_two_edges(TopoDS::Edge(c[0]), TopoDS::Edge(c[1]));
      }
      if (built.failure.empty()) {
        hist = compose_histories(input, {hist, built.hist, built.face_hist});
      }
    } catch (const std::exception& e) {
      thrown = failure_text(e);
    }
  }
  if (!thrown.empty() || !built.failure.empty()) {
    throw PysmeshError("Session.make_quad: GEOM's block driver built no quadrangle on the "
                       "edges " + named + ": " +
                           (thrown.empty() ? built.failure : thrown) + ".",
                       "", ids_as_int(edge_ids));
  }
  return commit(concat(survivors, built.face), hist, op, built.face);
}

py::dict Session::make_quad_from_vertices(const std::vector<EntityId>& vertex_ids) {
  const char* op = "make_quad_from_vertices";
  OpGuard guard(in_op_);
  const std::string named = id_text(vertex_ids);
  if (vertex_ids.size() != 4) {
    throw PysmeshError("Session.make_quad_from_vertices: " +
                           std::to_string(vertex_ids.size()) + " vertices are named (" + named +
                           "); a quadrangle takes 4 corners.",
                       "", ids_as_int(vertex_ids));
  }
  const ShapeKeyed<TopoDS_Shape> owners_of = body_of_subshape();
  std::vector<TopoDS_Shape> vertices;
  std::vector<TopoDS_Shape> owners;
  for (EntityId id : vertex_ids) {
    const EntityRecord& rec = require_alive(op, id);
    if (rec.kind != TopAbs_VERTEX) {
      throw PysmeshError("Session.make_quad_from_vertices: entity " + std::to_string(id) +
                             " is a " + kind_name(rec.kind) + ", not a VERTEX.",
                         "", {static_cast<int>(id)});
    }
    const TopoDS_Shape& v = rec.shapes.front();
    for (const TopoDS_Shape& seen : vertices) {
      if (seen.IsSame(v)) {
        throw PysmeshError("Session.make_quad_from_vertices: the vertices " + named +
                               " name one vertex twice.",
                           "", ids_as_int(vertex_ids));
      }
    }
    const auto it = owners_of.find(v);
    if (it == owners_of.end()) {
      throw PysmeshError("Session.make_quad_from_vertices: entity " + std::to_string(id) +
                         " does not belong to any body of the session root.");
    }
    require_vertex_body(op, it->second, id);
    vertices.push_back(v);
    append_unique(owners, it->second);
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(owners);
  const TopoDS_Compound input = bundle_of(vertices);

  QuadBuild built;
  Handle(BRepTools_History) hist;
  std::string thrown;
  {
    py::gil_scoped_release release;
    try {
      BRepBuilderAPI_Copy copier(input, /*copyGeom=*/Standard_True, /*copyMesh=*/Standard_False);
      hist = history_of(input, copier);
      const std::vector<TopoDS_Shape> c = copies_of(copier, vertices);
      built = quad_four_vertices(TopoDS::Vertex(c[0]), TopoDS::Vertex(c[1]),
                                 TopoDS::Vertex(c[2]), TopoDS::Vertex(c[3]));
      if (built.failure.empty()) {
        hist = compose_histories(input, {hist, built.hist, built.face_hist});
      }
    } catch (const std::exception& e) {
      thrown = failure_text(e);
    }
  }
  if (!thrown.empty() || !built.failure.empty()) {
    throw PysmeshError("Session.make_quad_from_vertices: GEOM's block driver built no "
                       "quadrangle on the vertices " + named + ": " +
                           (thrown.empty() ? built.failure : thrown) + ".",
                       "", ids_as_int(vertex_ids));
  }
  return commit(concat(survivors, built.face), hist, op, built.face);
}

// ---- make_hexa ---------------------------------------------------------------------- //

py::dict Session::make_hexa(const std::vector<EntityId>& face_ids) {
  const char* op = "make_hexa";
  OpGuard guard(in_op_);
  const std::string named = id_text(face_ids);
  if (face_ids.size() != 6 && face_ids.size() != 2) {
    throw PysmeshError("Session.make_hexa: " + std::to_string(face_ids.size()) +
                           " faces are named (" + named +
                           "); a hexahedron takes 6 faces, or 2 opposite faces.",
                       "", ids_as_int(face_ids));
  }
  const ShapeKeyed<TopoDS_Shape> owners_of = body_of_subshape();
  std::vector<TopoDS_Shape> faces;
  std::vector<TopoDS_Shape> owners;
  for (EntityId id : face_ids) {
    const TopoDS_Face face = sole_face(op, id);
    for (const TopoDS_Shape& seen : faces) {
      if (seen.IsSame(face)) {
        throw PysmeshError("Session.make_hexa: the faces " + named + " name one face twice.",
                           "", ids_as_int(face_ids));
      }
    }
    const auto it = owners_of.find(face);
    if (it == owners_of.end()) {
      throw PysmeshError("Session.make_hexa: entity " + std::to_string(id) +
                         " does not belong to any body of the session root.");
    }
    require_face_body(op, it->second, id);
    faces.push_back(face);
    append_unique(owners, it->second);
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(owners);
  const TopoDS_Compound input = bundle_of(faces);

  HexaBuild built;
  Handle(BRepTools_History) hist;
  std::string thrown;
  {
    py::gil_scoped_release release;
    try {
      BRepBuilderAPI_Copy copier(input, /*copyGeom=*/Standard_True, /*copyMesh=*/Standard_False);
      hist = history_of(input, copier);
      built = hexa_of(copies_of(copier, faces));
      if (built.failure.empty()) {
        std::vector<Handle(BRepTools_History)> steps{hist};
        steps.insert(steps.end(), built.steps.begin(), built.steps.end());
        hist = compose_histories(input, steps);
      }
    } catch (const std::exception& e) {
      thrown = failure_text(e);
    }
  }
  if (!thrown.empty() || !built.failure.empty()) {
    throw PysmeshError("Session.make_hexa: GEOM's block driver built no hexahedron on the "
                       "faces " + named + ": " +
                           (thrown.empty() ? built.failure : thrown) + ".",
                       "", ids_as_int(face_ids));
  }
  return commit(concat(survivors, built.solid), hist, op, built.solid);
}

// ---- check_blocks ------------------------------------------------------------------- //

namespace {

// GEOM's IsQuadrangle (GEOMImpl_IBlocksOperations.cxx), copied unchanged: one wire of 4
// edges, or, with a non-negative theToleranceC1, of more edges that form exactly 4 bounds
// of C1 continuity at that angular tolerance.
Standard_Boolean IsQuadrangle(const TopoDS_Face   &theFace,
                              const Standard_Real  theToleranceC1)
{
  TopExp_Explorer aFExp (theFace, TopAbs_WIRE);

  if (!aFExp.More()) {
    // no wire in the face
    return Standard_False;
  }

  TopoDS_Shape aWire = aFExp.Current();

  aFExp.Next();

  if (aFExp.More()) {
    // multiple wires in the face
    return Standard_False;
  }

  // Check number of edges in the face
  Standard_Integer    aNbEdges = 0;
  TopTools_MapOfShape aMapEdges;
  TopExp_Explorer     aWExp(aWire, TopAbs_EDGE);

  for (; aWExp.More(); aWExp.Next()) {
    if (aMapEdges.Add(aWExp.Current())) {
      aNbEdges++;

      if (aNbEdges > 4) {
        break;
      }
    }
  }

  if (aNbEdges < 4) {
    return Standard_False;
  }

  if (aNbEdges > 4) {
    if (theToleranceC1 < 0.) {
      return Standard_False;
    }

    // Check if a wire has 4 bounds of C1 continuity.
    BRepTools_WireExplorer aWireExp(TopoDS::Wire(aWire), theFace);
    TopTools_ListOfShape   anEdges;

    for (aNbEdges = 0; aWireExp.More(); aWireExp.Next()) {
      const TopoDS_Edge &anEdge = aWireExp.Current();

      // Skip degenerated edges.
      if (!BRep_Tool::Degenerated(anEdge)) {
        anEdges.Append(anEdge);
        ++aNbEdges;
      }
    }

    if (aNbEdges < 4) {
      return Standard_False;
    }

    // Compute number of sharp corners.
    anEdges.Append(anEdges.First()); // To make a loop.

    TopTools_ListIteratorOfListOfShape anIter(anEdges);
    Standard_Real      aPar[2];
    Standard_Integer   aNbCorners = 0;
    TopoDS_Edge        anEdge1    = TopoDS::Edge(anEdges.First());
    Handle(Geom_Curve) aCurve1    = BRep_Tool::Curve(anEdge1, aPar[0], aPar[1]);
    Handle(Geom_Curve) aCurve2;
    TopoDS_Edge        anEdge2;
    TopoDS_Vertex      aCommonVtx;
    gp_Pnt             aPnt;
    gp_Vec             aVec1;
    gp_Vec             aVec2;
    Standard_Boolean   isReversed1 = (anEdge1.Orientation() == TopAbs_REVERSED);
    Standard_Boolean   isReversed2;

    for (anIter.Next(); anIter.More(); anIter.Next()) {
      TopoDS_Edge anEdge2 = TopoDS::Edge(anIter.Value());

      if (!TopExp::CommonVertex(anEdge1, anEdge2, aCommonVtx)) {
        // NEVERREACHED
        return Standard_False;
      }

      // Check the angle between tangent vectors of 2 curves at this point.
      Standard_Real aParam1 = BRep_Tool::Parameter(aCommonVtx, anEdge1);
      Standard_Real aParam2 = BRep_Tool::Parameter(aCommonVtx, anEdge2);

      aCurve2     = BRep_Tool::Curve(anEdge2, aPar[0], aPar[1]);
      isReversed2 = (anEdge2.Orientation() == TopAbs_REVERSED);
      aCurve1->D1(aParam1, aPnt, aVec1);
      aCurve2->D1(aParam2, aPnt, aVec2);

      if (isReversed1) {
        aVec1.Reverse();
      }

      if (isReversed2) {
        aVec2.Reverse();
      }
      const Standard_Real anAngle = aVec1.Angle(aVec2);

      if (anAngle > theToleranceC1) {
        ++aNbCorners;

        if (aNbCorners > 4) {
          break;
        }
      }

      // Go to the next couple of edges.
      anEdge1     = anEdge2;
      aCurve1     = aCurve2;
      isReversed1 = isReversed2;
    }

    // Check the total number of corners.
    if (aNbCorners != 4) {
      return Standard_False;
    }
  }

  return Standard_True;
}

// GEOMImpl_IBlocksOperations::AddBlocksFrom, copied unchanged but for being a free
// function: blocks (6 quadrangles), non-blocks, hexahedral solids with degenerated or seam
// edges (BlockFix_CheckTool), and non-quadrangular faces.
void AddBlocksFrom (const TopoDS_Shape&   theShape,
                    TopTools_ListOfShape& BLO,
                    TopTools_ListOfShape& NOT,
                    TopTools_ListOfShape& EXT,
                    TopTools_ListOfShape& NOQ,
                    const Standard_Real   theToleranceC1)
{
  TopAbs_ShapeEnum aType = theShape.ShapeType();
  switch (aType) {
  case TopAbs_COMPOUND:
  case TopAbs_COMPSOLID:
    {
      TopoDS_Iterator It (theShape);
      for (; It.More(); It.Next()) {
        AddBlocksFrom(It.Value(), BLO, NOT, EXT, NOQ, theToleranceC1);
      }
    }
    break;
  case TopAbs_SOLID:
    {
      // Check, if there are seam or degenerated edges
      BlockFix_CheckTool aTool;
      aTool.SetShape(theShape);
      aTool.SetAngTolerance(theToleranceC1);
      aTool.Perform();
      if (aTool.NbPossibleBlocks() > 0) {
        EXT.Append(theShape);
      } else {
        // Count faces and edges in each face to recognize blocks
        TopTools_MapOfShape mapFaces;
        Standard_Integer nbFaces = 0;
        Standard_Boolean hasNonQuadr = Standard_False;
        TopExp_Explorer expF (theShape, TopAbs_FACE);

        for (; expF.More(); expF.Next()) {
          TopoDS_Face aF = TopoDS::Face(expF.Current());

          if (mapFaces.Add(aF)) {
            nbFaces++;

            if (!IsQuadrangle(aF, theToleranceC1)) {
              hasNonQuadr = Standard_True;
              NOQ.Append(aF);//0021483
            }
          }
        }

        if (nbFaces == 6 && !hasNonQuadr) {
          BLO.Append(theShape);
        } else {
          NOT.Append(theShape);
        }
      }
    }
    break;
  case TopAbs_SHELL: //0021483
  case TopAbs_FACE: //0021483
    {
      // Count edges in each face
      TopTools_MapOfShape mapFaces;
      TopExp_Explorer expF (theShape, TopAbs_FACE);
      for (; expF.More(); expF.Next()) {
        TopoDS_Face aF = TopoDS::Face(expF.Current());

        if (mapFaces.Add(aF)) {
          if (!IsQuadrangle(aF, theToleranceC1)) {
            NOQ.Append(aF);//0021483
          }
        }
      }
    }
    break;
  default:
    NOT.Append(theShape);
  }
}

// GEOMImpl_IBlocksOperations.cxx's relation codes and the two searches over them, copied
// unchanged.
#define REL_NOT_CONNECTED 0
#define REL_OK            1
#define REL_NOT_GLUED     2

void FindConnected (const Standard_Integer         theBlockIndex,
                    const TColStd_Array2OfInteger& theRelations,
                    TColStd_MapOfInteger&          theProcessedMap,
                    TColStd_MapOfInteger&          theConnectedMap)
{
  theConnectedMap.Add(theBlockIndex);
  theProcessedMap.Add(theBlockIndex);

  Standard_Integer nbBlocks = theRelations.ColLength();
  Standard_Integer col = 1;
  for (; col <= nbBlocks; col++) {
    if (theRelations(theBlockIndex, col) == REL_OK ||
        theRelations(theBlockIndex, col) == REL_NOT_GLUED) {
      if (!theProcessedMap.Contains(col)) {
        FindConnected(col, theRelations, theProcessedMap, theConnectedMap);
      }
    }
  }
}

Standard_Boolean HasAnyConnection (const Standard_Integer         theBlockIndex,
                                   const TColStd_MapOfInteger&    theWith,
                                   const TColStd_Array2OfInteger& theRelations,
                                   TColStd_MapOfInteger&          theProcessedMap)
{
  theProcessedMap.Add(theBlockIndex);

  Standard_Integer nbBlocks = theRelations.ColLength();
  Standard_Integer col = 1;
  for (; col <= nbBlocks; col++) {
    if (theRelations(theBlockIndex, col) != REL_NOT_CONNECTED) {
      if (!theProcessedMap.Contains(col)) {
        if (theWith.Contains(col))
          return Standard_True;
        if (HasAnyConnection(col, theWith, theRelations, theProcessedMap))
          return Standard_True;
      }
    }
  }

  return Standard_False;
}

// GEOM's BCErrorType (GEOMImpl_IBlocksOperations.hxx), in GEOM's order. The value of
// NOT_GLUED (4) is what GEOM writes into the relation table for a pair the glue analyser
// finds, so the order matters (see check_blocks).
enum BCErrorType { NOT_BLOCK, EXTRA_EDGE, INVALID_CONNECTION, NOT_CONNECTED, NOT_GLUED };

const char* bc_error_name(int error) {
  switch (error) {
    case NOT_BLOCK: return "NOT_BLOCK";
    case EXTRA_EDGE: return "EXTRA_EDGE";
    case INVALID_CONNECTION: return "INVALID_CONNECTION";
    case NOT_CONNECTED: return "NOT_CONNECTED";
    case NOT_GLUED: return "NOT_GLUED";
    default: return "UNKNOWN";
  }
}

struct BCError {
  int error = NOT_BLOCK;
  std::vector<TopoDS_Shape> incriminated;
};

}  // namespace

py::dict Session::check_blocks(const std::optional<std::vector<EntityId>>& entity_ids,
                               double angular_tolerance, const py::object& progress,
                               const py::object& cancel) const {
  const char* op = "check_blocks";
  finite_arg(op, "tolerance", angular_tolerance);
  const ProgressHooks hooks = hooks_of(op, progress, cancel);
  std::vector<TopoDS_Shape> bodies;
  if (entity_ids.has_value()) {
    if (entity_ids->empty()) {
      throw PysmeshError("Session.check_blocks: an empty list of entities is named; there is "
                         "no body to check.");
    }
    bodies = owner_bodies(op, *entity_ids);
  } else {
    bodies = root_bodies(state_.root);
  }
  const TopoDS_Compound aBlockOrComp = bundle_of(bodies);
  // GEOM's call takes -1 for no C1 test (geomBuilder.CheckCompoundOfBlocks).
  const Standard_Real theToleranceC1 = angular_tolerance;

  ProgressDriver driver(op, hooks);
  Standard_Boolean isCompOfBlocks = Standard_True;
  std::vector<BCError> theErrors;
  std::string failure;
  {
    py::gil_scoped_release release;
    try {
      // GEOMImpl_IBlocksOperations::CheckCompoundOfBlocks, copied. pySMESH: the bodies are
      // checked one by one (AddBlocksFrom recurses into the compound in the same order), so
      // that progress and cancel come between them; the errors keep their shapes, which
      // the session names by id; and the failure of the glue analyser is a failure text.
      Message_ProgressScope steps(driver.range(), "check_blocks",
                                  static_cast<double>(bodies.size()) + 2.0);

      // 1. Separate blocks from non-blocks
      TopTools_ListOfShape NOT; // Not blocks
      TopTools_ListOfShape EXT; // Hexahedral solids, having degenerated and/or seam edges
      TopTools_ListOfShape BLO; // All blocks from the given compound
      TopTools_ListOfShape NOQ; // All non-quadrangular faces
      for (TopoDS_Iterator anIt (aBlockOrComp); anIt.More() && steps.More(); anIt.Next()) {
        AddBlocksFrom(anIt.Value(), BLO, NOT, EXT, NOQ, theToleranceC1);
        steps.Next();
      }

      // Report non-blocks
      if (NOT.Extent() > 0) {
        isCompOfBlocks = Standard_False;
        BCError anErr;
        anErr.error = NOT_BLOCK;
        TopTools_ListIteratorOfListOfShape it (NOT);
        for (; it.More(); it.Next()) {
          anErr.incriminated.push_back(it.Value());
        }
        theErrors.push_back(anErr);
      }

      // Report solids, having degenerated and/or seam edges
      if (EXT.Extent() > 0) {
        isCompOfBlocks = Standard_False;
        BCError anErr;
        anErr.error = EXTRA_EDGE;
        TopTools_ListIteratorOfListOfShape it (EXT);
        for (; it.More(); it.Next()) {
          anErr.incriminated.push_back(it.Value());
        }
        theErrors.push_back(anErr);
      }

      Standard_Integer nbBlocks = BLO.Extent();
      if (nbBlocks == 0) {
        isCompOfBlocks = Standard_False;
      }
      if (nbBlocks > 1 && steps.More()) {
        // Prepare data for 2. and 3.
        TColStd_Array2OfInteger aRelations (1, nbBlocks, 1, nbBlocks);
        aRelations.Init(REL_NOT_CONNECTED);

        TopTools_IndexedMapOfShape mapBlocks;

        BRep_Builder BB;
        TopoDS_Compound aComp;
        BB.MakeCompound(aComp);

        TopTools_ListIteratorOfListOfShape BLOit (BLO);
        for (; BLOit.More(); BLOit.Next()) {
          mapBlocks.Add(BLOit.Value());
          BB.Add(aComp, BLOit.Value());
        }

        // 2. Find glued blocks (having shared faces)
        TopTools_IndexedDataMapOfShapeListOfShape mapFaceBlocks;
        GEOMImpl_Block6Explorer::MapShapesAndAncestors
          (aComp, TopAbs_FACE, TopAbs_SOLID, mapFaceBlocks);

        Standard_Integer prevInd = 0, curInd = 0;
        Standard_Integer ind = 1, nbFaces = mapFaceBlocks.Extent();
        for (; ind <= nbFaces; ind++) {
          const TopTools_ListOfShape& aGluedBlocks = mapFaceBlocks.FindFromIndex(ind);
          if (aGluedBlocks.Extent() > 1) { // Shared face found
            TopTools_ListIteratorOfListOfShape aGluedBlocksIt (aGluedBlocks);
            TopoDS_Shape prevBlock, curBlock;
            for (; aGluedBlocksIt.More(); aGluedBlocksIt.Next()) {
              curBlock = aGluedBlocksIt.Value();
              if (!prevBlock.IsNull()) {
                prevInd = mapBlocks.FindIndex(prevBlock);
                curInd  = mapBlocks.FindIndex(curBlock);
                aRelations.SetValue(prevInd, curInd, REL_OK);
                aRelations.SetValue(curInd, prevInd, REL_OK);
              }
              prevBlock = curBlock;
            }
          }
        }

        // 3. Find not glued blocks
        GEOMAlgo_GlueAnalyser aGD;

        aGD.SetShape(aComp);
        aGD.SetTolerance(Precision::Confusion());
        aGD.SetCheckGeometry(Standard_True);
        aGD.Perform();
        steps.Next();

        Standard_Integer iErr;
        iErr = aGD.ErrorStatus();
        if (iErr) {
          failure = "GEOMAlgo_GlueAnalyser failed with code " + std::to_string(iErr);
        }

        // Report not glued blocks
        if (failure.empty() && aGD.HasSolidsToGlue()) {
          isCompOfBlocks = Standard_False;
          Standard_Integer aSx1Ind, aSx2Ind;

          const GEOMAlgo_ListOfCoupleOfShapes& aLCS = aGD.SolidsToGlue();
          GEOMAlgo_ListIteratorOfListOfCoupleOfShapes aItCS (aLCS);
          for (; aItCS.More(); aItCS.Next()) {
            const GEOMAlgo_CoupleOfShapes& aCS = aItCS.Value();
            const TopoDS_Shape& aSx1 = aCS.Shape1();
            const TopoDS_Shape& aSx2 = aCS.Shape2();

            aSx1Ind = mapBlocks.FindIndex(aSx1);
            aSx2Ind = mapBlocks.FindIndex(aSx2);
            aRelations.SetValue(aSx1Ind, aSx2Ind, NOT_GLUED);
            aRelations.SetValue(aSx2Ind, aSx1Ind, NOT_GLUED);

            BCError anErr;
            anErr.error = NOT_GLUED;
            anErr.incriminated.push_back(aSx1);
            anErr.incriminated.push_back(aSx2);
            theErrors.push_back(anErr);
          }
        }

        if (failure.empty()) {
          // 4. Find largest set of connected (good connection or not glued) blocks
          Standard_Integer ibl = 1;
          TColStd_MapOfInteger aProcessedMap;
          TColStd_MapOfInteger aLargestSet;
          TColStd_MapOfInteger aCurrentSet;
          for (ibl = 1; ibl <= nbBlocks; ibl++) {
            if (!aProcessedMap.Contains(ibl)) {
              aCurrentSet.Clear();
              FindConnected(ibl, aRelations, aProcessedMap, aCurrentSet);
              if (aCurrentSet.Extent() > aLargestSet.Extent()) {
                aLargestSet = aCurrentSet;
              }
            }
          }

          // 5. Report all blocks, isolated from <aLargestSet>
          BCError anErr;
          anErr.error = NOT_CONNECTED;
          Standard_Boolean hasIsolated = Standard_False;
          for (ibl = 1; ibl <= nbBlocks; ibl++) {
            if (!aLargestSet.Contains(ibl)) {
              aProcessedMap.Clear();
              if (!HasAnyConnection(ibl, aLargestSet, aRelations, aProcessedMap)) {
                // report connection absence
                hasIsolated = Standard_True;
                anErr.incriminated.push_back(mapBlocks.FindKey(ibl));
              }
            }
          }
          if (hasIsolated) {
            isCompOfBlocks = Standard_False;
            theErrors.push_back(anErr);
          }
        }
        steps.Next();
      }
    } catch (const std::exception& e) {
      failure = failure_text(e);
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled(op);
  }
  if (!failure.empty()) {
    throw PysmeshError("Session.check_blocks: the check failed in OCCT: " + failure + ".");
  }
  py::list findings;
  for (const BCError& anErr : theErrors) {
    std::vector<EntityId> ids;
    for (const TopoDS_Shape& s : anErr.incriminated) {
      for (EntityId id : ids_within(s)) {
        ids.push_back(id);
      }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    py::dict f;
    f["kind"] = bc_error_name(anErr.error);
    f["ids"] = ids_array(ids);
    findings.append(f);
  }
  py::dict out;
  out["is_compound_of_blocks"] = static_cast<bool>(isCompOfBlocks);
  out["findings"] = findings;
  return out;
}

// ---- propagation_chain -------------------------------------------------------------- //

py::array_t<std::int64_t> Session::propagation_chain(EntityId edge_id) const {
  const char* op = "propagation_chain";
  const TopoDS_Edge edge = sole_edge(op, edge_id);
  // GEOMImpl_IBlocksOperations::Propagate, copied for the one chain that holds the edge:
  // the edge's wires are walked pass by pass; a wire of exactly 4 edges holding an edge of
  // the chain adds its opposite edge. pySMESH: the chain starts at the named edge rather
  // than at each edge of the shape in turn, and each edge is given by its label (the lowest
  // live id that denotes it), sorted, instead of GEOM's sorted sub-shapes
  // (GEOMUtils::SortShapes).
  TopTools_IndexedDataMapOfShapeListOfShape MEW;
  GEOMImpl_Block6Explorer::MapShapesAndAncestors
    (state_.root, TopAbs_EDGE, TopAbs_WIRE, MEW);

  TopTools_MapOfShape mapAcceptedEdges;
  TopoDS_Shape curE = MEW.FindKey(MEW.FindIndex(edge));

  // Build the chain
  TopTools_ListOfShape currentChain;
  TopTools_ListOfShape listPrevEdges;

  currentChain.Append(curE);
  listPrevEdges.Append(curE);
  mapAcceptedEdges.Add(curE);

  // Collect all edges pass by pass
  while (listPrevEdges.Extent() > 0) {
    // List of edges, added to chain on this cycle pass
    TopTools_ListOfShape listCurEdges;

    // Find the next portion of edges
    TopTools_ListIteratorOfListOfShape itE (listPrevEdges);
    for (; itE.More(); itE.Next()) {
      TopoDS_Shape anE = itE.Value();

      // Iterate on faces, having edge <anE>
      TopTools_ListIteratorOfListOfShape itW (MEW.FindFromKey(anE));
      for (; itW.More(); itW.Next()) {
        TopoDS_Shape aW = itW.Value();
        TopoDS_Shape anOppE;

        BRepTools_WireExplorer aWE (TopoDS::Wire(aW));
        Standard_Integer nb = 1, found = 0;
        TopTools_Array1OfShape anEdges (1,4);
        for (; aWE.More(); aWE.Next(), nb++) {
          if (nb > 4) {
            found = 0;
            break;
          }
          anEdges(nb) = aWE.Current();
          if (anEdges(nb).IsSame(anE)) found = nb;
        }

        if (nb == 5 && found > 0) {
          // Quadrangle face found, get an opposite edge
          Standard_Integer opp = found + 2;
          if (opp > 4) opp -= 4;
          anOppE = anEdges(opp);

          if (!mapAcceptedEdges.Contains(anOppE)) {
            // Add found edge to the chain
            currentChain.Append(anOppE);
            listCurEdges.Append(anOppE);
            mapAcceptedEdges.Add(anOppE);
          }
        } // if (nb == 5 && found > 0)
      } // for (; itF.More(); itF.Next())
    } // for (; itE.More(); itE.Next())

    listPrevEdges = listCurEdges;
  } // while (listPrevEdges.Extent() > 0)

  std::vector<EntityId> ids;
  for (TopTools_ListIteratorOfListOfShape it (currentChain); it.More(); it.Next()) {
    ids.push_back(label_of(op, it.Value()));
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids_array(ids);
}

// ---- opposite_face, face_by_edges, find_block_near ----------------------------------- //

EntityId Session::opposite_face(EntityId block_id, EntityId face_id) const {
  const char* op = "opposite_face";
  const EntityRecord& block = require_alive(op, block_id);
  if (block.kind != TopAbs_SOLID) {
    throw PysmeshError("Session.opposite_face: entity " + std::to_string(block_id) +
                           " is a " + kind_name(block.kind) + ", not a SOLID.",
                       "", {static_cast<int>(block_id)});
  }
  if (block.shapes.size() != 1) {
    throw PysmeshError("Session.opposite_face: solid " + std::to_string(block_id) +
                           " denotes " + std::to_string(block.shapes.size()) +
                           " solids (it was split); name one of them instead.",
                       "", {static_cast<int>(block_id)});
  }
  const TopoDS_Face face = sole_face(op, face_id);
  ShapeSet block_faces;
  TopExp::MapShapes(block.shapes.front(), TopAbs_FACE, block_faces);
  if (!block_faces.Contains(face)) {
    throw PysmeshError("Session.opposite_face: face " + std::to_string(face_id) +
                           " is not a face of solid " + std::to_string(block_id) + ".",
                       "", {static_cast<int>(block_id), static_cast<int>(face_id)});
  }
  // GEOMImpl_IBlocksOperations::GetOppositeFace, copied: GEOMImpl_Block6Explorer orders the
  // block from the face, and the opposite face is its face 2.
  TopoDS_Shape aShape;
  std::string failure;
  try {
    GEOMImpl_Block6Explorer aBlockTool;
    aBlockTool.InitByBlockAndFace(block.shapes.front(), block_faces.FindKey(
                                                             block_faces.FindIndex(face)));
    aShape = aBlockTool.GetFace(2);
  } catch (const std::exception& e) {
    failure = failure_text(e);
  }
  if (!failure.empty() || aShape.IsNull()) {
    throw PysmeshError("Session.opposite_face: solid " + std::to_string(block_id) +
                           " is not a block GEOM's block explorer can order from face " +
                           std::to_string(face_id) + ": " +
                           (failure.empty() ? std::string("no opposite face") : failure) + ".",
                       "", {static_cast<int>(block_id), static_cast<int>(face_id)});
  }
  return label_of(op, aShape);
}

EntityId Session::face_by_edges(EntityId edge_a, EntityId edge_b) const {
  const char* op = "face_by_edges";
  const TopoDS_Edge anArg1 = sole_edge(op, edge_a);
  const TopoDS_Edge anArg2 = sole_edge(op, edge_b);
  // GEOMImpl_IBlocksOperations::GetFaceByEdges, copied: the faces of the model holding
  // both edges, which must be exactly one. GEOM's search for an edge of the shape similar
  // to a given edge that is not in it does not apply: a live edge id is in the model.
  TopTools_IndexedDataMapOfShapeListOfShape MEF;
  GEOMImpl_Block6Explorer::MapShapesAndAncestors(state_.root, TopAbs_EDGE, TopAbs_FACE, MEF);

  const TopTools_ListOfShape& aFacesOfE1 = MEF.FindFromKey(anArg1);
  const TopTools_ListOfShape& aFacesOfE2 = MEF.FindFromKey(anArg2);

  std::vector<TopoDS_Shape> found;
  TopTools_ListIteratorOfListOfShape anIterF1 (aFacesOfE1);
  for (; anIterF1.More(); anIterF1.Next()) {

    TopTools_ListIteratorOfListOfShape anIterF2 (aFacesOfE2);
    for (; anIterF2.More(); anIterF2.Next()) {

      if (anIterF1.Value().IsSame(anIterF2.Value())) {
        // Store the face, defined by two edges
        append_unique(found, anIterF1.Value());
      }
    }
  }
  const std::string pair = "edges " + std::to_string(edge_a) + " and " + std::to_string(edge_b);
  if (found.empty()) {
    throw PysmeshError("Session.face_by_edges: " + pair + " do not belong to one face.", "",
                       {static_cast<int>(edge_a), static_cast<int>(edge_b)});
  }
  if (found.size() > 1) {
    std::vector<EntityId> ids;
    for (const TopoDS_Shape& f : found) {
      ids.push_back(label_of(op, f));
    }
    throw PysmeshError("Session.face_by_edges: " + pair + " belong to " +
                           std::to_string(found.size()) + " faces (" + id_text(ids) + ").",
                       "", {static_cast<int>(edge_a), static_cast<int>(edge_b)});
  }
  return label_of(op, found.front());
}

EntityId Session::find_block_near(double px, double py_, double pz) const {
  const char* op = "find_block_near";
  finite_arg(op, "point", px, py_, pz);
  const gp_Pnt aPnt(px, py_, pz);
  Standard_Real PX = px, PY = py_, PZ = pz;
  // GEOMImpl_IBlocksOperations::GetBlockNearPoint, copied: the first block holding the
  // point inside; or the one block with the point on its boundary; or, of the blocks whose
  // box or volume leave the point out, the nearest (BRepExtrema_DistShapeShape).
  // pySMESH: the classified blocks are kept in the order of the model's solids, not in a
  // map keyed by shape address, so that the nearest of two at one distance is the same in
  // every run; the point is a point, not a vertex shape, for the distance.
  TopoDS_Shape aShape;
  Standard_Integer nearest = 2, nbFound = 0;
  std::vector<std::pair<TopoDS_Shape, Standard_Integer>> mapShapeDist;
  TopTools_MapOfShape aSeen;
  std::string failure;
  try {
    TopExp_Explorer exp (state_.root, TopAbs_SOLID);
    for (; exp.More(); exp.Next()) {
      TopoDS_Shape aSolid = exp.Current();

      if (aSeen.Add(aSolid)) {
        Standard_Integer aDistance = 2;

        // 1.a. Classify relatively Bounding box
        Standard_Real Xmin, Ymin, Zmin, Xmax, Ymax, Zmax;
        Bnd_Box BB;
        BRepBndLib::Add(aSolid, BB);
        BB.Get(Xmin, Ymin, Zmin, Xmax, Ymax, Zmax);
        if (PX < Xmin || Xmax < PX ||
            PY < Ymin || Ymax < PY ||
            PZ < Zmin || Zmax < PZ) {
          // OUT of bounding box
          aDistance = 1;
        } else {
          // 1.b. Classify relatively the solid itself
          BRepClass3d_SolidClassifier SC (aSolid, aPnt, Precision::Confusion());
          if (SC.State() == TopAbs_IN) {
            aDistance = -1;
          } else if (SC.State() == TopAbs_ON) {
            aDistance = 0;
          } else { // OUT
            aDistance = 1;
          }
        }

        if (aDistance < nearest) {
          nearest = aDistance;
          aShape = aSolid;
          nbFound = 1;

          // A first found block, containing the point inside, will be returned.
          // It is the solution, if there are no intersecting blocks in the compound.
          if (nearest == -1) break;

        } else if (aDistance == nearest) {
          nbFound++;
        } else {
        }

        mapShapeDist.emplace_back(aSolid, aDistance);
      }
    }

    // 2. Define block, containing the point or having minimum distance to it
    if (nbFound > 1 && nearest == 1) {
      // The point is outside some blocks and there are
      // no blocks, having the point inside or on boundary.
      // We will get a nearest block
      const TopoDS_Vertex aVert = BRepBuilderAPI_MakeVertex(aPnt);
      Standard_Real minDist = RealLast();
      for (const auto& [aSolid, aDist] : mapShapeDist) {
        if (aDist == 1) {
          BRepExtrema_DistShapeShape aDistTool (aVert, aSolid);
          if (!aDistTool.IsDone()) {
            failure = "Can not find a distance from the given point to one of blocks";
            break;
          }
          Standard_Real aValue = aDistTool.Value();
          if (aValue < minDist) {
            minDist = aValue;
            aShape = aSolid;
          }
        }
      }
    }
  } catch (const std::exception& e) {
    failure = failure_text(e);
  }
  const std::string where = "(" + number_text(px) + ", " + number_text(py_) + ", " +
                            number_text(pz) + ")";
  if (!failure.empty()) {
    throw PysmeshError("Session.find_block_near: the search near " + where +
                       " failed in OCCT: " + failure + ".");
  }
  if (nbFound == 0) {
    throw PysmeshError("Session.find_block_near: the model has no solid; there is no block "
                       "near " + where + ".");
  }
  if (nbFound > 1 && nearest == 0) {
    // The point is on boundary of some blocks and there are
    // no blocks, having the point inside their volume
    std::vector<EntityId> ids;
    for (const auto& [aSolid, aDist] : mapShapeDist) {
      if (aDist == 0) {
        ids.push_back(label_of(op, aSolid));
      }
    }
    throw PysmeshError("Session.find_block_near: the point " + where +
                       " lies on the boundary of " + std::to_string(nbFound) +
                       " blocks (solids " + id_text(ids) + ") and inside none.");
  }
  return label_of(op, aShape);
}

// ---- multi_transform ---------------------------------------------------------------- //

py::dict Session::multi_transform(EntityId block_id, const std::vector<std::array<double, 3>>& vectors,
                                  const std::vector<int>& counts) {
  const char* op = "multi_transform";
  for (const std::array<double, 3>& v : vectors) {
    finite_arg(op, "vectors", v[0], v[1], v[2]);
  }
  OpGuard guard(in_op_);
  if (vectors.empty() || vectors.size() > 2) {
    throw PysmeshError("Session.multi_transform: " + std::to_string(vectors.size()) +
                       " translation vectors are given; a 1-D or 2-D array of copies takes 1 "
                       "or 2.");
  }
  if (counts.size() != vectors.size()) {
    throw PysmeshError("Session.multi_transform: " + std::to_string(counts.size()) +
                       " counts are given for " + std::to_string(vectors.size()) +
                       " translation vectors; each vector has one count.");
  }
  int total = 1;
  for (std::size_t k = 0; k < counts.size(); ++k) {
    if (counts[k] < 1) {
      throw PysmeshError("Session.multi_transform: the count " + std::to_string(counts[k]) +
                         " along vector " + std::to_string(k) +
                         " is below 1; a count includes the original.");
    }
    total *= counts[k];
  }
  if (total < 2) {
    throw PysmeshError("Session.multi_transform: the counts give 1 block, the original "
                       "alone; there is nothing to copy.");
  }
  std::vector<gp_Vec> steps;
  for (std::size_t k = 0; k < vectors.size(); ++k) {
    const gp_Vec v(vectors[k][0], vectors[k][1], vectors[k][2]);
    if (counts[k] > 1 && v.Magnitude() <= Precision::Confusion()) {
      throw PysmeshError("Session.multi_transform: translation vector " + std::to_string(k) +
                         " is (" + number_text(v.X()) + ", " + number_text(v.Y()) + ", " +
                         number_text(v.Z()) + "), shorter than 1e-07; its copies would lie "
                         "on the original.");
    }
    steps.push_back(v);
  }
  const TopoDS_Shape body = sole_body(op, {block_id});
  const std::vector<TopoDS_Shape> untouched = bodies_excluding({body});
  const std::string block_text = std::string(kind_name(body.ShapeType())) + " body of entity " +
                                 std::to_string(block_id);

  TopoDS_Shape result;
  Handle(BRepTools_History) hist = new BRepTools_History;
  std::string failure;
  {
    py::gil_scoped_release release;
    try {
      // The block's tolerance in its vertices, as GEOMImpl_BlockDriver's
      // BLOCK_MULTI_TRANSFORM branches take it for the glue.
      Standard_Real aTol = Precision::Confusion();
      for (TopExp_Explorer expV (body, TopAbs_VERTEX); expV.More(); expV.Next()) {
        aTol = Max(BRep_Tool::Tolerance(TopoDS::Vertex(expV.Current())), aTol);
      }
      // The copies: the session's copy (BRepBuilderAPI_Copy) moved by the session's
      // translation (a Location), each sub-shape generated from the original's at the same
      // place of TopExp::MapShapes.
      std::vector<TopoDS_Shape> blocks{body};
      const std::vector<TopoDS_Shape> original = registered_subshapes(body);
      const int n1 = counts[0];
      const int n2 = counts.size() > 1 ? counts[1] : 1;
      for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
          if (i == 0 && j == 0) {
            continue;
          }
          gp_Vec shift = steps[0] * i;
          if (steps.size() > 1) {
            shift += steps[1] * j;
          }
          gp_Trsf trsf;
          trsf.SetTranslation(shift);
          BRepBuilderAPI_Copy copier(body, /*copyGeom=*/Standard_True,
                                     /*copyMesh=*/Standard_False);
          const TopoDS_Shape moved = copier.Shape().Moved(TopLoc_Location(trsf));
          const std::vector<TopoDS_Shape> placed = registered_subshapes(moved);
          if (placed.size() != original.size()) {
            failure = "a copy of the block has another number of sub-shapes";
            break;
          }
          for (std::size_t k = 0; k < original.size(); ++k) {
            if (placed[k].TShape() != copier.ModifiedShape(original[k]).TShape()) {
              failure = "a copy's sub-shape is not where BRepBuilderAPI_Copy put it";
              break;
            }
            hist->AddGenerated(original[k], placed[k]);
          }
          blocks.push_back(moved);
        }
      }
      if (failure.empty()) {
        // The glue: GEOMAlgo_Gluer2 on the block and its copies, as Session.glue runs it
        // with kind "faces", at the block's tolerance; solids only (GEOM's GlueFaces with
        // doKeepNonSolids false).
        const TopoDS_Compound input = bundle_of(blocks);
        GEOMAlgo_Gluer2 gluer;
        gluer.SetArgument(input);
        gluer.SetTolerance(aTol);
        gluer.SetKeepNonSolids(Standard_False);
        gluer.Detect();
        int error = gluer.ErrorStatus();
        if (error == 0 && gluer.WarningStatus() != 2) {
          gluer.Perform();
          error = gluer.ErrorStatus();
          if (error == 0 && gluer.WarningStatus() != 1) {
            result = gluer.Shape();
            NCollection_List<TopoDS_Shape> arguments;
            arguments.Append(input);
            hist->Merge(new BRepTools_History(arguments, gluer));
          } else if (error == 0) {
            // Nothing coincides (warning 1): the copies stand apart, as GEOM's
            // GlueFaces leaves them, and are committed unglued.
            result = input;
          }
        }
        if (failure.empty() && result.IsNull()) {
          failure = "GEOMAlgo_Gluer2 failed with error " + std::to_string(error) +
                    " and warning " + std::to_string(gluer.WarningStatus());
        }
      }
    } catch (const std::exception& e) {
      failure = failure_text(e);
    }
  }
  if (!failure.empty()) {
    throw PysmeshError("Session.multi_transform: the copies of the " + block_text +
                       " were not glued: " + failure + "; nothing is committed.");
  }
  return commit(concat(untouched, result), hist, op, result);
}

}  // namespace session
}  // namespace pysmesh
