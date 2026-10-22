// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH binding — Session: GEOM's advanced shapes for block meshing.
//
//   divided_disk   GEOM's MakeDividedDisk (AdvancedEngine_DividedDiskDriver), copied;
//   pipe_t_shape   GEOM's MakePipeTShape, the method of AdvancedEngine_IOperations ported
//                  to OCCT calls (see that section).
//
// The GEOM code is SALOME GEOM tag V9_16_0 (7d0a18e1554c25bca2c2d9882e0621ea3f7f9c81),
// src/AdvancedEngine, Copyright (C) 2007-2026 CEA, EDF, OPEN CASCADE, LGPL-2.1 or later,
// used here under LGPL-2.1, and recorded in PROVENANCE.md.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include "geom_block6.hpp"

#include <GEOMAlgo_ClsfBox.hxx>
#include <GEOMAlgo_ClsfSurf.hxx>
#include <GEOMAlgo_FinderShapeOn2.hxx>
#include <GEOMAlgo_Gluer2.hxx>
#include <GEOMAlgo_Splitter.hxx>
#include <GEOMUtils.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_Plane.hxx>
#include <ShapeAnalysis_Edge.hxx>
#include <ShapeAnalysis_Wire.hxx>
#include <ShapeExtend_WireData.hxx>
#include <ShapeFix_Edge.hxx>
#include <ShapeFix_Wire.hxx>
#include <StdFail_NotDone.hxx>
#include <TColStd_IndexedDataMapOfTransientTransient.hxx>
#include <TNaming_CopyShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_MapOfShape.hxx>
#include <gp.hxx>

namespace pysmesh {
namespace session {

namespace {

// ---- divided_disk: AdvancedEngine_DividedDiskDriver.cxx ------------------------------- //

// MakeDiskHexagon, MakeDiskSquare, TransformShape and WPlaneTransform are
// AdvancedEngine_DividedDiskDriver's const members, copied as free functions; the code is
// unchanged.

//=======================================================================
//function : MakeDiskHexagon
//purpose  :
//=======================================================================
TopoDS_Shell MakeDiskHexagon(double R, double Ratio)
{
  // Geometry
  gp_Dir ZDir(0,0,1);
  gp_Dir XDir(1,0,0);
  gp_Pnt Orig(0,0,0);

  // Circle
  gp_Ax1 Ax1(Orig,ZDir);
  gp_Ax2 Ax(Orig,ZDir,XDir);
  gp_Circ aCircle(Ax, R);

  // Points
//   gp_Pnt P4(0.01*Ratio*R,0,0);
//   gp_Pnt P3(R,0,0);
//   gp_Pnt P2 = P3.Rotated(Ax1,-M_PI/6.0);
//   gp_Pnt P1(P4.X(),
//             P4.X()/sqrt(3.0),0);
  gp_Pnt P1(0.01*Ratio*R*sqrt(3.0)/2,0,0);
  gp_Pnt P2(R,0,0);
  gp_Pnt P3 = P2.Rotated(Ax1,M_PI/6.0);
  gp_Pnt P4(P1.X(),
            P1.X()/sqrt(3.0),0);


  //surfaces
  gp_Ax2 anAx (gp::XOY());
  Handle(Geom_Plane) aPlane = new Geom_Plane (anAx);

  // Topology

  // Vertices
  TopoDS_Vertex O  = BRepBuilderAPI_MakeVertex(Orig);
  TopoDS_Vertex V1_init = BRepBuilderAPI_MakeVertex(P1);
  TopoDS_Vertex V2_init = BRepBuilderAPI_MakeVertex(P2);
  TopoDS_Vertex V3 = BRepBuilderAPI_MakeVertex(P3);
  TopoDS_Vertex V4 = BRepBuilderAPI_MakeVertex(P4);

  TopoDS_Vertex V1 = V1_init;
  TopoDS_Vertex V2 = V2_init;

  //Rotation
  gp_Trsf myTrsf;
  myTrsf.SetRotation(Ax1, M_PI/3.0);

  BRepBuilderAPI_Transform xform(myTrsf);
  xform.Perform(V1,Standard_True);
  TopoDS_Vertex V1_60 = TopoDS::Vertex(xform.Shape());
  xform.Perform(V2,Standard_True);
  TopoDS_Vertex V2_60 = TopoDS::Vertex(xform.Shape());

  // Declaration of shapes (used in the loop)
  TopoDS_Edge E1, E2, E3, E4, E5, E6, E7, E8, E9;
  TopoDS_Wire W1, W2, W3;
  TopoDS_Face F1, F2, F3;
  TopoDS_Shell S;

  BRep_Builder aBuilder;
  aBuilder.MakeShell(S);

  // Initialisation of edges
  TopoDS_Edge E1_init = BRepBuilderAPI_MakeEdge(V1,TopoDS::Vertex(V2.Reversed()));
  E1 = E1_init;
  TopoDS_Edge E8_init = BRepBuilderAPI_MakeEdge(O,TopoDS::Vertex(V1.Reversed()));
  E8 = E8_init;

  for (int i=1;i<=6;i++)
  {
    // Edges

    // for Face1
    E2 = BRepBuilderAPI_MakeEdge(aCircle, V2, TopoDS::Vertex(V3.Reversed()));
    E3 = BRepBuilderAPI_MakeEdge(V3,TopoDS::Vertex(V4.Reversed()));
    E4 = BRepBuilderAPI_MakeEdge(V4,TopoDS::Vertex(V1.Reversed()));

    // for Face2
    if (i==6)
    {
      E5 = BRepBuilderAPI_MakeEdge(aCircle, V3, TopoDS::Vertex(V2_init.Reversed()));
      E7 = BRepBuilderAPI_MakeEdge(V1_init,TopoDS::Vertex(V4.Reversed()));
    }
    else
    {
      E5 = BRepBuilderAPI_MakeEdge(aCircle, V3, TopoDS::Vertex(V2_60.Reversed()));
      E7 = BRepBuilderAPI_MakeEdge(V1_60,TopoDS::Vertex(V4.Reversed()));
    }
    E6 = BRepBuilderAPI_MakeEdge(V2_60,TopoDS::Vertex(V1_60.Reversed()));

    // for Face3
    E9 = BRepBuilderAPI_MakeEdge(V1_60,TopoDS::Vertex(O.Reversed()));


    // Wires

    //Wire1
    aBuilder.MakeWire(W1);
    if (i==1)
      aBuilder.Add(W1,E1);
    else
      aBuilder.Add(W1,TopoDS::Edge(E1.Reversed()));
    aBuilder.Add(W1,E2);
    aBuilder.Add(W1,E3);
    aBuilder.Add(W1,E4);

    // Wire 2
    aBuilder.MakeWire(W2);
    aBuilder.Add(W2,TopoDS::Edge(E3.Reversed()));
    aBuilder.Add(W2,E5);
    if (i==6)
      aBuilder.Add(W2,TopoDS::Edge(E1_init.Reversed()));
    else
      aBuilder.Add(W2,E6);
    aBuilder.Add(W2,E7);

    // Wire3
    aBuilder.MakeWire(W3);
    if (i==1)
      aBuilder.Add(W3,E8);
    else
      aBuilder.Add(W3,TopoDS::Edge(E8.Reversed()));
    aBuilder.Add(W3,TopoDS::Edge(E4.Reversed()));
    aBuilder.Add(W3,TopoDS::Edge(E7.Reversed()));
    if (i==6)
      aBuilder.Add(W3,TopoDS::Edge(E8_init.Reversed()));
    else
      aBuilder.Add(W3,E9);

    // Faces creation
    F1 = BRepBuilderAPI_MakeFace(aPlane,W1);
    F2 = BRepBuilderAPI_MakeFace(aPlane,W2);
    F3 = BRepBuilderAPI_MakeFace(aPlane,W3);

    //Shell
    aBuilder.Add(S, F1);
    aBuilder.Add(S, F2);
    aBuilder.Add(S, F3);

    // rotation
    V1=V1_60;
    V2=V2_60;

    xform.Perform(V1_60,Standard_True);
    V1_60 = TopoDS::Vertex(xform.Shape());
    xform.Perform(V2_60,Standard_True);
    V2_60 = TopoDS::Vertex(xform.Shape());
    xform.Perform(V3,Standard_True);
    V3    = TopoDS::Vertex(xform.Shape());
    xform.Perform(V4,Standard_True);
    V4    = TopoDS::Vertex(xform.Shape());

    // "Increment" of edges
    E1=E6;
    E8=E9;
  }

  return S;
}

//=======================================================================
//function : MakeDiskSquare
//purpose  :
//=======================================================================
TopoDS_Shape MakeDiskSquare(double R, double Ratio)
{
  // Geometry
  gp_Dir ZDir(0,0,1);
  gp_Dir XDir(1,0,0);
  gp_Pnt Orig(0,0,0);

  // Circle
  gp_Ax1 Ax1(Orig,ZDir);
  gp_Ax2 Ax(Orig,ZDir,XDir);
  gp_Circ aCircle(Ax, R);

  // Points
  gp_Pnt P1(0.01*Ratio*R,0,0);
  gp_Pnt P2(R,0,0);

  //surfaces
  gp_Ax2 anAx (gp::XOY());
  Handle(Geom_Plane) aPlane = new Geom_Plane (anAx);

  // Topology

  // Vertices
  TopoDS_Vertex V1_init = BRepBuilderAPI_MakeVertex(P1);
  TopoDS_Vertex V2_init = BRepBuilderAPI_MakeVertex(P2);

  TopoDS_Vertex V1 = V1_init;
  TopoDS_Vertex V2 = V2_init;

  //Rotation
  gp_Trsf myTrsf;
  myTrsf.SetRotation(Ax1, M_PI/2);

  BRepBuilderAPI_Transform xform(myTrsf);
  xform.Perform(V1,Standard_True);
  TopoDS_Vertex V1_rotated = TopoDS::Vertex(xform.Shape());
  xform.Perform(V2,Standard_True);
  TopoDS_Vertex V2_rotated = TopoDS::Vertex(xform.Shape());

  // Declaration of shapes (used in the loop)
  TopoDS_Edge E1, E2, E3, E4;
  TopoDS_Wire W1, W2;
  TopoDS_Face F1, F2;
  TopoDS_Shell S;

  BRep_Builder aBuilder;
  aBuilder.MakeWire(W2);  // Central Wire
  aBuilder.MakeShell(S);  // Shell

  // Initialisation of edges
  TopoDS_Edge E1_init = BRepBuilderAPI_MakeEdge(V1,TopoDS::Vertex(V2.Reversed()));
  E1 = E1_init;

  for (int i=1;i<=4;i++)
  {
    // Edges
    // for Face1

    E3 = BRepBuilderAPI_MakeEdge(V2_rotated,TopoDS::Vertex(V1_rotated.Reversed()));
    if (i == 4)
    {
      E2 = BRepBuilderAPI_MakeEdge(aCircle, V2, TopoDS::Vertex(V2_init.Reversed()));
      E4 = BRepBuilderAPI_MakeEdge(V1_init,TopoDS::Vertex(V1.Reversed()));
    }
    else
    {
      E2 = BRepBuilderAPI_MakeEdge(aCircle, V2, TopoDS::Vertex(V2_rotated.Reversed()));
      E4 = BRepBuilderAPI_MakeEdge(V1_rotated,TopoDS::Vertex(V1.Reversed()));
    }

    // Wires
    //Wire1
    aBuilder.MakeWire(W1);
    if (i==1)
      aBuilder.Add(W1,E1);
    else
      aBuilder.Add(W1,TopoDS::Edge(E1.Reversed()));
    aBuilder.Add(W1,E2);
    if (i==4)
      aBuilder.Add(W1,TopoDS::Edge(E1_init.Reversed()));
    else
      aBuilder.Add(W1,E3);
    aBuilder.Add(W1,E4);

    // Wire central
    aBuilder.Add(W2,TopoDS::Edge(E4.Reversed()));

    // Faces creation
    F1 = BRepBuilderAPI_MakeFace(aPlane,W1);

    //Shell
    aBuilder.Add(S, F1);

    // rotation
    V1=V1_rotated;
    V2=V2_rotated;

    xform.Perform(V1_rotated,Standard_True);
    V1_rotated = TopoDS::Vertex(xform.Shape());
    xform.Perform(V2_rotated,Standard_True);
    V2_rotated = TopoDS::Vertex(xform.Shape());

    // "Increment" of edges
    E1=E3;
  }
  // Central square Face
  F2 = BRepBuilderAPI_MakeFace(aPlane,W2);
  aBuilder.Add(S, F2);

  return S;
}

//=======================================================================
//function :  WPlaneTransform
//purpose  :  Perform shape transformation accordingly with the given
//            Working Plane
//=======================================================================
TopoDS_Shape WPlaneTransform(TopoDS_Shape theShape, gp_Ax3 theWPlane)
{
  gp_Trsf aTrans;
  aTrans.SetTransformation(theWPlane);
  aTrans.Invert();
  BRepBuilderAPI_Transform aTransformation (theShape, aTrans, Standard_False);
  return aTransformation.Shape();
}

//=======================================================================
//function :  TrasformShape(TopoDS_Shape aShape,int theOrientation)
//purpose  :  Perform shape transformation accordingly with specified
//            orientation
//=======================================================================
TopoDS_Shape TransformShape(TopoDS_Shape theShape, int theOrientation)
{
  gp_Dir N, Vx;
  gp_Pnt theOrigin = gp::Origin();

  switch(theOrientation)
  {
    case 1:
    {
      N = gp::DZ();
      Vx = gp::DX();
      break;
    }
    case 2:
    {
      N = gp::DX();
      Vx = gp::DY();
      break;
    }
    case 3:
    {
      N = gp::DY();
      Vx = gp::DZ();
      break;
    }
  }

  gp_Ax3 aWPlane = gp_Ax3(theOrigin, N, Vx);

  return WPlaneTransform(theShape, aWPlane);
}

}  // namespace

// ---- divided_disk ------------------------------------------------------------------- //

py::dict Session::divided_disk(double radius, double ratio, const std::string& orientation,
                               const std::string& pattern) {
  const char* op = "divided_disk";
  finite_arg(op, "radius", radius);
  finite_arg(op, "ratio", ratio);
  OpGuard guard(in_op_);
  // GEOM's orientation codes (AdvancedEngine_IOperations::MakeDividedDisk: 1 = OXY,
  // 2 = OYZ, 3 = OZX; TransformShape: the plane's normal Z, X, Y) and its patterns
  // (SQUARE = 0, HEXAGON = 1).
  int code = 0;
  if (orientation == "xy") {
    code = 1;
  } else if (orientation == "yz") {
    code = 2;
  } else if (orientation == "zx") {
    code = 3;
  } else {
    throw PysmeshError("Session.divided_disk: the orientation '" + orientation +
                       "' is not one of xy, yz, zx.");
  }
  if (pattern != "square" && pattern != "hexagon") {
    throw PysmeshError("Session.divided_disk: the pattern '" + pattern +
                       "' is not one of square, hexagon.");
  }
  if (!(radius > Precision::Confusion())) {
    throw PysmeshError("Session.divided_disk: the radius " + number_text(radius) +
                       " is not above 1e-07.");
  }
  if (!(ratio > 0.0 && ratio < 100.0)) {
    throw PysmeshError("Session.divided_disk: the ratio " + number_text(ratio) +
                       " is not strictly between 0 and 100; it is the distance of the inner "
                       "block's corners from the centre, in percent of the radius.");
  }
  const TopoDS_Shape disk = build_shape(op, [&] {
    const TopoDS_Shape aDisk =
        pattern == "square" ? MakeDiskSquare(radius, ratio) : MakeDiskHexagon(radius, ratio);
    return TransformShape(aDisk, code);
  });
  return add_bodies(disk, op);
}

// ---- pipe_t_shape: AdvancedEngine's MakePipeTShape, ported --------------------------- //
//
// GEOM builds the T-shape in AdvancedEngine_IOperations::MakePipeTShape from GEOM operation
// calls on GEOM objects (MakePipeTShapePartition, MakePipeTShapeMirrorAndGlue, MakeGroups,
// MakeInternalGroup). Each call is re-expressed here with the OCCT and GEOM-algorithm
// calls it makes, in GEOM's order, for the wall only (no thickness reduction, chamfer or
// fillet: GEOM's theRL ... theLthinI all 0, TSHAPE_BASIC). The driver's own functions
// (MakePipeTShape, MakeQuarterPipeTShape) are copied. Every id of the result is
// constructed, so no history is kept.

namespace {

//=======================================================================
//function : MakePipeTShape (AdvancedEngine_PipeTShapeDriver), copied; Raise -> throw
//=======================================================================
TopoDS_Shape MakePipeTShapeWall (const double r1, const double w1, const double l1,
                                 const double r2, const double w2, const double l2)
{
  double r1Ext = r1 + w1;
  double r2Ext = r2 + w2;

  gp_Pnt aP0 (0, 0, 0);
  gp_Pnt aP1 (-l1, 0, 0);
  gp_Vec aVX = gp::DX(), aVZ = gp::DZ();
  gp_Ax2 anAxes1 (aP1, aVX, aVZ);
  gp_Ax2 anAxes2 (aP0, aVZ, aVX);

  // Build the initial pipes
  BRepPrimAPI_MakeCylinder C1Int (anAxes1, r1, Abs(2 * l1));
  BRepPrimAPI_MakeCylinder C1Ext (anAxes1, r1Ext, Abs(2 * l1));
  BRepPrimAPI_MakeCylinder C2Int (anAxes2, r2, Abs(l2));
  BRepPrimAPI_MakeCylinder C2Ext (anAxes2, r2Ext, Abs(l2));
  C1Int.Build();
  C1Ext.Build();
  C2Int.Build();
  C2Ext.Build();
  if (!C1Int.IsDone() || !C1Ext.IsDone() || !C2Int.IsDone() || !C2Ext.IsDone()) {
    throw StdFail_NotDone("Cannot build cylinders");
  }

  // Fuse the 2 pipes
  BRepAlgoAPI_Fuse fuse1 (C1Ext.Shape(), C2Ext.Shape());
  if (!fuse1.IsDone()) {
    throw StdFail_NotDone("Cannot fuse cylinders");
  }

  // Remove small radius main pipe
  BRepAlgoAPI_Cut cut1 (fuse1.Shape(), C1Int.Shape());
  if (!cut1.IsDone()) {
    throw StdFail_NotDone("Couldn't cut cylinders");
  }

  // Remove small radius incident pipe => Te
  BRepAlgoAPI_Cut Te (cut1.Shape(), C2Int.Shape());
  if (!Te.IsDone()) {
    throw StdFail_NotDone("Couldn't cut cylinders");
  }

  TopoDS_Shape aShape = Te.Shape();

  // VSR: 30/12/2014: temporary workaround about intersection curves problem
  GEOMUtils::FixShapeCurves(aShape);

  return aShape;
}

//=======================================================================
//function : MakeQuarterPipeTShape (AdvancedEngine_PipeTShapeDriver), copied; Raise -> throw
//=======================================================================
TopoDS_Shape MakeQuarterPipeTShape (const double r1, const double w1, const double l1,
                                    const double r2, const double w2, const double l2)
{
  TopoDS_Shape Te = MakePipeTShapeWall(r1, w1, l1, r2, w2, l2);
  if (Te.IsNull())
    throw StdFail_NotDone("Couldn't build Pipe TShape");

  // Get a quarter of shape => Te2
  double r1Ext = r1 + w1;
  BRepPrimAPI_MakeBox box1 (gp_Pnt(0, -2*r1Ext, -2*r1Ext), gp_Pnt( Abs(2 * l1), 2*r1Ext, Abs(2*l2)));
  BRepPrimAPI_MakeBox box2 (gp_Pnt(0,  2*r1Ext, -2*r1Ext), gp_Pnt(-Abs(2 * l1), 0,       Abs(2*l2)));
  box1.Build();
  box2.Build();
  if (!box1.IsDone() || !box2.IsDone()) {
    throw StdFail_NotDone("Couldn't build boxes");
  }
  BRepAlgoAPI_Cut cut3 (Te, box1.Shape());
  if (!cut3.IsDone()) {
    throw StdFail_NotDone("Couldn't cut Pipe Tshape with box");
  }
  BRepAlgoAPI_Cut Te4 (cut3.Shape(), box2.Shape());
  if (!Te4.IsDone()) {
    throw StdFail_NotDone("Couldn't cut Pipe Tshape with box");
  }

  TopoDS_Shape aShape = Te4.Shape();

  // VSR: 30/12/2014: temporary workaround about intersection curves problem
  GEOMUtils::FixShapeCurves(aShape);

  return Te4.Shape();
}

// GEOM's MakeBoxDXDYDZ(dx, dy, dz) moved by TranslateDXDYDZ(x, y, z).
TopoDS_Shape box_at(double x, double y, double z, double dx, double dy, double dz) {
  return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape();
}

// GEOMImpl_IShapesOperations::getShapesOnBoxIDs, ported: the sub-shapes of `aShape` of a
// type in a state against a box (GEOMAlgo_FinderShapeOn2 with GEOMAlgo_ClsfBox, tolerance
// 1e-4), in the finder's order, after GEOMUtils::CheckTriangulation for edges and faces.
TopTools_ListOfShape shapes_on_box(const TopoDS_Shape& aBox, const TopoDS_Shape& aShape,
                                   TopAbs_ShapeEnum theShapeType, GEOMAlgo_State theState) {
  // Check presence of triangulation, build if need
  if (theShapeType != TopAbs_VERTEX && !GEOMUtils::CheckTriangulation(aShape)) {
    throw StdFail_NotDone("Cannot build triangulation on the shape");
  }

  // Call algo
  GEOMAlgo_FinderShapeOn2 aFinder;
  Standard_Real aTol = 0.0001; // default value

  Handle(GEOMAlgo_ClsfBox) aClsfBox = new GEOMAlgo_ClsfBox;
  aClsfBox->SetBox(aBox);

  aFinder.SetShape(aShape);
  aFinder.SetTolerance(aTol);
  aFinder.SetClsf(aClsfBox);
  aFinder.SetShapeType(theShapeType);
  aFinder.SetState(theState);
  aFinder.Perform();

  // Interpret results
  Standard_Integer iErr = aFinder.ErrorStatus();
  if (iErr) {
    throw StdFail_NotDone(("GEOMAlgo_FinderShapeOn2 error " + std::to_string(iErr)).c_str());
  }
  return aFinder.Shapes();
}

// AdvancedEngine_IOperations::GetFacesOnSurf, ported: the faces of `theShape` on a surface
// (GEOMAlgo_FinderShapeOn2 with GEOMAlgo_ClsfSurf, state ON, 3 to 100 inner points), after
// GEOMUtils::CheckTriangulation (GEOM's caller has meshed the shape before).
TopTools_ListOfShape faces_on_surf(const TopoDS_Shape& theShape,
                                   const Handle(Geom_Surface)& theSurface,
                                   const Standard_Real theTolerance) {
  if (!GEOMUtils::CheckTriangulation(theShape)) {
    throw StdFail_NotDone("Cannot build triangulation on the shape");
  }
  GEOMAlgo_FinderShapeOn2   aFinder;
  Handle(GEOMAlgo_ClsfSurf) aClsfSurf = new GEOMAlgo_ClsfSurf;

  aClsfSurf->SetSurface(theSurface);
  aFinder.SetShape(theShape);
  aFinder.SetTolerance(theTolerance);
  aFinder.SetClsf(aClsfSurf);
  aFinder.SetShapeType(TopAbs_FACE);
  aFinder.SetState(GEOMAlgo_ST_ON);
  aFinder.SetNbPntsMin(3);
  aFinder.SetNbPntsMax(100);
  aFinder.Perform();

  Standard_Integer iErr = aFinder.ErrorStatus();
  if (iErr) {
    throw StdFail_NotDone(("GEOMAlgo_FinderShapeOn2 error " + std::to_string(iErr)).c_str());
  }
  return aFinder.Shapes();
}

// GEOM's MakePlanePntVec(P, V, size): a square face of side `size` centred at P, normal V
// (GEOMImpl_PlaneDriver, PLANE_PNT_VEC).
TopoDS_Shape plane_face(const gp_Pnt& aP, const gp_Vec& aV, double theSize) {
  const double aSize = theSize / 2.0;
  gp_Pln aPln (aP, aV);
  return BRepBuilderAPI_MakeFace(aPln, -aSize, +aSize, -aSize, +aSize).Shape();
}

// GEOMImpl_ShapeDriver::MakeWireFromEdges, copied for edges given as shapes, in its mode
// GEOMImpl_WBM_FixTolerance (GEOMImpl_IShapesOperations::MakeWire's default); Raise ->
// throw.
TopoDS_Wire MakeWireFromEdges(const std::vector<TopoDS_Shape>& theEdges,
                              const Standard_Real theTolerance)
{
  BRep_Builder B;

  TopoDS_Wire aWire;
  B.MakeWire(aWire);

  // add edges
  for (const TopoDS_Shape& aShape_i : theEdges) {
    if (aShape_i.IsNull()) {
      throw Standard_NullObject("Shape for wire construction is null");
    }
    if (aShape_i.ShapeType() == TopAbs_EDGE || aShape_i.ShapeType() == TopAbs_WIRE) {
      TopExp_Explorer exp (aShape_i, TopAbs_EDGE);
      for (; exp.More(); exp.Next())
        B.Add(aWire, TopoDS::Edge(exp.Current()));
    } else {
      throw Standard_TypeMismatch
        ("Shape for wire construction is neither an edge nor a wire");
    }
  }

  // fix edges order
  Handle(ShapeFix_Wire) aFW = new ShapeFix_Wire;
  aFW->Load(aWire);
  aFW->FixReorder();

  if (aFW->StatusReorder(ShapeExtend_FAIL1)) {
    throw Standard_ConstructionError("Wire construction failed: several loops detected");
  }
  else if (aFW->StatusReorder(ShapeExtend_FAIL)) {
    throw Standard_ConstructionError("Wire construction failed");
  }
  else {
  }

  // IMP 0019766: Building a Wire from unconnected edges by introducing a tolerance
  bool isClosed = false;
  Handle(ShapeAnalysis_Wire) asaw = aFW->Analyzer();
  if (asaw->CheckGap3d(1)) { // between last and first edges
    Standard_Real dist = asaw->MinDistance3d();
    if (dist < theTolerance)
      isClosed = true;
  }
  aFW->ClosedWireMode() = isClosed;
  aFW->FixConnected(theTolerance);

  if (aFW->StatusConnected(ShapeExtend_FAIL)) {
    throw Standard_ConstructionError("Wire construction failed: cannot build connected wire");
  }
  // IMP 0019766

  if (aFW->StatusConnected(ShapeExtend_DONE3)) {
    // Confused with <prec> but not Analyzer.Precision(), set the same
    aFW->SetPrecision(theTolerance);
    aFW->FixGapsByRangesMode() = Standard_True;
    if (aFW->FixGaps3d()) {
      Handle(ShapeExtend_WireData) sbwd = aFW->WireData();
      Handle(ShapeFix_Edge) aFe = new ShapeFix_Edge;
      for (Standard_Integer iedge = 1; iedge <= sbwd->NbEdges(); iedge++) {
        TopoDS_Edge aEdge = TopoDS::Edge(sbwd->Edge(iedge));
        aFe->FixVertexTolerance(aEdge);
        aFe->FixSameParameter(aEdge);
      }
    }
    else if (aFW->StatusGaps3d(ShapeExtend_FAIL)) {
      throw Standard_ConstructionError("Wire construction failed: cannot fix 3d gaps");
    }
  }
  aWire = aFW->WireAPIMake();

  return aWire;
}

// AdvancedEngine_IOperations::MakePipeTShapePartition, ported for the wall (isNormal, no
// chamfer or fillet): the quarter T-shape is split into blocks by the plane z = 0, by the
// plane through the y axis towards (R1+W1, 0, (L1 + R2+W2) / 2) when the two outer radii
// differ, and by the face across the wall at the junction (its boundary: the inner and the
// outer intersection curves and two straight edges between their ends), with
// GEOMAlgo_Splitter (GEOM's MakePartition: limit SOLID, compounds taken apart). GEOM runs
// the splitter in parallel; here it runs serially, for identical results run to run.
// `failure` receives GEOM's error text.
TopoDS_Shape partition_quarter(const TopoDS_Shape& theShape, double theR1, double theW1,
                               double theL1, double theR2, double theW2, double theL2,
                               std::string& failure) {
  const double theH = 0, theW = 0, theRF = 0;

  // Useful values
  double aSize = 2*(theL1 + theL2);
  double aR1Ext = theR1 + theW1;
  double aR2Ext = theR2 + theW2;
  double theVertCylinderRadius = aR2Ext + theW + theRF;

  // Common edges on internal cylinder
  const TopoDS_Shape box_i = box_at(-theR2, -theR2, 0, theR2, theR2, theR1);
  const TopTools_ListOfShape edges_i = shapes_on_box(box_i, theShape, TopAbs_EDGE, GEOMAlgo_ST_IN);
  if (edges_i.IsEmpty()) {
    failure = "Internal edges not found";
    return TopoDS_Shape();
  }
  const TopoDS_Shape arete_intersect_int = edges_i.First();

  // search for vertices located on both internal pipes
  const TopTools_ListOfShape vertices_i =
    shapes_on_box(box_i, theShape, TopAbs_VERTEX, GEOMAlgo_ST_ONIN);
  if (vertices_i.IsEmpty()) {
    failure = "Internal vertices not found";
    return TopoDS_Shape();
  }

  TopoDS_Vertex vi1, vi2;
  double d1min = theR2+theW2, d2min=theR2+theW2;
  for (const TopoDS_Shape& v : vertices_i) {
    TopoDS_Vertex aVertex = TopoDS::Vertex(v);
    gp_Pnt aP = BRep_Tool::Pnt(aVertex);
    if (Abs(aP.X()) <= Precision::Confusion()) {
      if (Abs(aP.Y()) < d1min) {
        vi1 = aVertex;
        d1min = Abs(aP.Y());
      }
    } else if (Abs(aP.Y()) <= Precision::Confusion()) {
      if (Abs(aP.X()) < d2min) {
        vi2 = aVertex;
        d2min = Abs(aP.X());
      }
    }
  }
  if (vi1.IsNull() || vi2.IsNull()) {
    failure = "Cannot find internal intersection vertices";
    return TopoDS_Shape();
  }

  // isNormal
  TopoDS_Vertex ve1, ve2;
  TopoDS_Vertex vertex1, vertex2;

  const TopoDS_Shape box_e = box_at(-aR2Ext, -aR2Ext, 0, aR2Ext, aR2Ext, aR1Ext);

  // search for vertices located on both external pipes
  const TopTools_ListOfShape vertices_e =
    shapes_on_box(box_e, theShape, TopAbs_VERTEX, GEOMAlgo_ST_ONIN);
  if (vertices_e.IsEmpty()) {
    failure = "External vertices not found";
    return TopoDS_Shape();
  }

  double d1max = 0, d2max = 0;
  for (const TopoDS_Shape& v : vertices_e) {
    TopoDS_Vertex aVertex = TopoDS::Vertex(v);
    gp_Pnt aP = BRep_Tool::Pnt(aVertex);
    if (Abs(aP.X()) <= Precision::Confusion()) {
      if (Abs(aP.Y()) > d1max) {
        ve1 = aVertex;
        vertex1 = aVertex;
        d1max = Abs(aP.Y());
      }
    } else if (Abs(aP.Y()) <= Precision::Confusion()) {
      if (Abs(aP.X()) > d2max) {
        ve2 = aVertex;
        vertex2 = aVertex;
        d2max = Abs(aP.X());
      }
    }
  }
  if (ve1.IsNull() || ve2.IsNull()) {
    failure = "Cannot find external intersection vertices";
    return TopoDS_Shape();
  }

  // Common edges on external cylinder
  const TopTools_ListOfShape edges_e = shapes_on_box(box_e, theShape, TopAbs_EDGE, GEOMAlgo_ST_IN);
  if (edges_e.IsEmpty()) {
    failure = "External edges not found";
    return TopoDS_Shape();
  }
  TopoDS_Shape arete_intersect_ext;
  ShapeAnalysis_Edge sae;
  for (const TopoDS_Shape& anObj : edges_e) {
    TopoDS_Edge anEdge = TopoDS::Edge(anObj);
    if ( !anEdge.IsNull() &&
         (sae.FirstVertex(anEdge).IsSame(vertex1) || sae.LastVertex(anEdge).IsSame(vertex1)) &&
         (sae.FirstVertex(anEdge).IsSame(vertex2) || sae.LastVertex(anEdge).IsSame(vertex2))) {
      arete_intersect_ext = anObj;
    }
  }

  // MakeLineTwoPnt (GEOMImpl_LineDriver, LINE_TWO_PNT): the edge on the two vertices.
  if (BRep_Tool::Pnt(ve1).Distance(BRep_Tool::Pnt(vi1)) < Precision::Confusion() ||
      BRep_Tool::Pnt(ve2).Distance(BRep_Tool::Pnt(vi2)) < Precision::Confusion()) {
    failure = "Edge 1 could not be built";
    return TopoDS_Shape();
  }
  const TopoDS_Shape edge_e1 = BRepBuilderAPI_MakeEdge(ve1, vi1).Shape();
  const TopoDS_Shape edge_e2 = BRepBuilderAPI_MakeEdge(ve2, vi2).Shape();

  const TopoDS_Wire wire_t = MakeWireFromEdges(
      {arete_intersect_int, edge_e1, arete_intersect_ext, edge_e2}, 1e-7);
  if (wire_t.IsNull()) {
    failure = "Impossible to build wire";
    return TopoDS_Shape();
  }
  // MakeFace(wire_t, false): GEOMImpl_ShapeDriver FACE_WIRE, which needs a closed wire,
  // then GEOMImpl_Block6Explorer::MakeFace.
  TopoDS_Vertex aV1, aV2;
  TopExp::Vertices(wire_t, aV1, aV2);
  if (aV1.IsNull() || aV2.IsNull() || !aV1.IsSame(aV2)) {
    failure = "Shape for face construction is not closed";
    return TopoDS_Shape();
  }
  TopoDS_Shape face_t;
  Handle(BRepTools_History) aFaceHistory;
  geomcopy::GEOMImpl_Block6Explorer::MakeFace(wire_t, Standard_False, face_t, aFaceHistory);
  if (face_t.IsNull()) {
    failure = "Impossible to build face";
    return TopoDS_Shape();
  }

  // Planes
  const gp_Pnt aP0 (0, 0, 0);
  const TopoDS_Shape aPlnOZ = plane_face(aP0, gp_Vec(0, 0, 1), aSize);
  const TopoDS_Shape aPlnOXZ =
    plane_face(aP0, gp_Vec(aR1Ext, 0, 0.5*(theL1+theVertCylinderRadius)), aSize);

  // Partition (GEOMImpl_PartitionDriver, PARTITION_PARTITION: compounds taken apart into
  // simple shapes, limit SOLID, no non-limit shapes)
  GEOMAlgo_Splitter PS;
  TopTools_ListOfShape aSimple;
  GEOMUtils::AddSimpleShapes(theShape, aSimple);
  for (const TopoDS_Shape& aS : aSimple)
    PS.AddArgument(aS);
  PS.AddTool(aPlnOZ);
  if (Abs(aR1Ext - aR2Ext) > Precision::Confusion())
    PS.AddTool(aPlnOXZ);
  PS.AddTool(face_t);
  PS.SetLimitMode(0);
  PS.SetLimit(TopAbs_SOLID);
  PS.SetRunParallel(Standard_False);
  PS.Perform();
  const TopoDS_Shape Te3 = PS.Shape();
  if (PS.HasErrors() || Te3.IsNull()) {
    failure = "Impossible to build partition of TShape";
    return TopoDS_Shape();
  }

  // Last verification: result should be a block
  std::string check_failure;
  if (!is_compound_of_blocks(Te3, -1, check_failure)) {
    failure = "TShape is not a compound of block" +
              (check_failure.empty() ? std::string() : " (" + check_failure + ")");
    return TopoDS_Shape();
  }
  return Te3;
}

// GEOM's MirrorPlaneCopy(shape, plane through the origin with normal `dir`)
// (GEOMImpl_MirrorDriver, MIRROR_PLANE_COPY).
TopoDS_Shape mirror_copy(const TopoDS_Shape& anOriginal, const gp_Dir& dir) {
  gp_Trsf aTrsf;
  aTrsf.SetMirror(gp_Ax2(gp::Origin(), dir));
  BRepBuilderAPI_Transform aTransformation (anOriginal, aTrsf, Standard_False);
  return aTransformation.Shape();
}

// GEOM's MakeGlueFaces of one shape (GEOMImpl_GlueDriver::GlueWithWarnings, GLUE_FACES):
// GEOMAlgo_Gluer2 detects and glues every coincident group.
TopoDS_Shape glue_faces(const TopoDS_Shape& theShape, const Standard_Real theTolerance,
                        const Standard_Boolean doKeepNonSolids) {
  GEOMAlgo_Gluer2 aGA;
  aGA.SetArgument(theShape);
  aGA.SetTolerance(theTolerance);
  aGA.SetKeepNonSolids(doKeepNonSolids);
  aGA.Detect();
  if (aGA.WarningStatus() == 2) {
    throw Standard_Failure("GLUE_ERROR_STICKED_SHAPES");
  }
  if (aGA.ErrorStatus()) {
    throw Standard_Failure(("Error in GEOMAlgo_Gluer2 with code " +
                            std::to_string(aGA.ErrorStatus())).c_str());
  }
  aGA.Perform();
  if (aGA.ErrorStatus()) {
    throw Standard_Failure(("Error in GEOMAlgo_Gluer2 with code " +
                            std::to_string(aGA.ErrorStatus())).c_str());
  }
  return aGA.Shape();
}

// AdvancedEngine_IOperations::MakePipeTShapeMirrorAndGlue, ported: the quarter is mirrored
// in the planes x = 0 and y = 0 into the whole T-shape, and the four quarters' faces are
// glued at 1e-7 (MakeGlueFaces, keeping non-solids); when that result is invalid, a copy of
// the four is glued at their largest vertex tolerance instead.
TopoDS_Shape mirror_and_glue(const TopoDS_Shape& theShape) {
  const TopoDS_Shape Te4 = mirror_copy(theShape, gp::DX());
  const TopoDS_Shape Te5 = mirror_copy(theShape, gp::DY());
  const TopoDS_Shape Te6 = mirror_copy(Te4, gp::DY());

  BRep_Builder B;
  TopoDS_Compound Te7;
  B.MakeCompound(Te7);
  for (const TopoDS_Shape& aS : {theShape, Te4, Te5, Te6})
    B.Add(Te7, aS);

  // Copy source shape
  TopoDS_Shape aShapeCopy;
  TColStd_IndexedDataMapOfTransientTransient aMapTShapes;
  TNaming_CopyShape::CopyTool(Te7, aMapTShapes, aShapeCopy);

  TopoDS_Shape aShape = glue_faces(Te7, 1e-7, Standard_True);
  BRepCheck_Analyzer anAna (aShape, Standard_True);

  if (!anAna.IsValid()) {
    // Try to do gluing with the tolerance equal to maximal
    // tolerance of vertices of the source shape.
    Standard_Real aTolMax = -RealLast();

    for (TopExp_Explorer ExV (aShapeCopy, TopAbs_VERTEX); ExV.More(); ExV.Next()) {
      TopoDS_Vertex aVertex = TopoDS::Vertex(ExV.Current());
      Standard_Real aTol = BRep_Tool::Tolerance(aVertex);

      if (aTol > aTolMax) {
        aTolMax = aTol;
      }
    }

    // Perform gluing
    aShape = glue_faces(aShapeCopy, aTolMax, Standard_True);
  }
  return aShape;
}

// One named group of the T-shape.
struct TShapeGroup {
  std::string name;
  TopAbs_ShapeEnum kind = TopAbs_FACE;
  std::vector<TopoDS_Shape> shapes;
};

// AdvancedEngine_IOperations::MakeGroups for TSHAPE_BASIC with FIND_GROUPS_BY_POINTS (both
// set at V9_16_0), ported. The junction faces: GEOM takes the common of the shape with the
// shell of its bounding box, sorts its faces (GEOMUtils::SortShapes), cuts them into groups
// of 4 and finds the shape's faces in place of each; here the shape's faces on that box
// (GEOMAlgo_FinderShapeOn2, ClsfBox, state ON) are sorted the same way, which are the same
// faces. The edge groups: GEOM's Propagate chains, each named after the first of six
// reference edges (GEOMUtils::GetEdgeNearPoint of six points) it holds.
void make_groups(const TopoDS_Shape& aShape, double theR1, double theW1, double theL1,
                 double theR2, double theW2, double theL2, std::vector<TShapeGroup>& theSeq) {
  double aR1Ext = theR1 + theW1;
  double aR2Ext = theR2 + theW2;

  // Junction faces.
  const TopoDS_Shape aBox = box_at(-theL1, -aR1Ext, -aR1Ext, 2*theL1, 2*aR1Ext, aR1Ext+theL2);
  TopTools_ListOfShape aCommonFaces = shapes_on_box(aBox, aShape, TopAbs_FACE, GEOMAlgo_ST_ON);
  GEOMUtils::SortShapes(aCommonFaces, Standard_False);
  std::vector<TopoDS_Shape> sorted(aCommonFaces.begin(), aCommonFaces.end());
  std::vector<std::vector<TopoDS_Shape>> aCompoundOfFacesList;
  for (int i = 0; i + 4 <= static_cast<int>(sorted.size()); i += 4) {
    aCompoundOfFacesList.push_back({sorted.begin() + i, sorted.begin() + i + 4});
  }
  if (aCompoundOfFacesList.size() == 3) {
    for (int k = 0; k < 3; ++k) {
      theSeq.push_back({"JUNCTION_FACE_" + std::to_string(k + 1), TopAbs_FACE,
                        aCompoundOfFacesList[static_cast<std::size_t>(k)]});
    }
  }

  // Edge groups: result of propagate.
  const std::vector<std::vector<TopoDS_Shape>> aSeqPropagate =
      propagation_chains(aShape, TopoDS_Shape());
  const auto edge_near = [&aShape](const gp_Pnt& aPnt) {
    BRepBuilderAPI_MakeVertex mkVertex (aPnt);
    return GEOMUtils::GetEdgeNearPoint(aShape, TopoDS::Vertex(mkVertex.Shape()));
  };
  // "Thickness" group (a)
  const TopoDS_Shape anEdgeA = edge_near(gp_Pnt(-theL1, 0, theR1 + theW1/2.));
  // "Circular quarter of pipe" group (b)
  const TopoDS_Shape anEdgeB =
      edge_near(gp_Pnt(-theL1, -aR1Ext * sin(M_PI/4.), -aR1Ext * sin(M_PI/4.)));
  // "Circular quarter of pipe" group (c)
  const TopoDS_Shape anEdgeC =
      edge_near(gp_Pnt(-theL1, -aR1Ext * sin(M_PI/4.), aR1Ext * sin(M_PI/4.)));
  // "Main pipe half length" group (d)
  const TopoDS_Shape anEdgeD = edge_near(gp_Pnt(-theL1/2., 0, -aR1Ext));
  // "Incident pipe half length" group (e)
  double aTol10 = Precision::Confusion() * 10.;
  const TopoDS_Shape anEdgeE = edge_near(gp_Pnt(-aR2Ext, 0, theL2 - aTol10));
  // "Flange" group (f)
  double aFx = - aR2Ext - aTol10;
  const TopoDS_Shape anEdgeF = edge_near(gp_Pnt(aFx, 0, aR1Ext));

  for (const std::vector<TopoDS_Shape>& aGroup : aSeqPropagate) {
    TopTools_IndexedMapOfShape anEdgesMap;
    for (const TopoDS_Shape& anEdge : aGroup)
      anEdgesMap.Add(anEdge);
    const char* name = nullptr;
    if (anEdgesMap.Contains(anEdgeA)) name = "THICKNESS";
    else if (anEdgesMap.Contains(anEdgeB)) name = "CIRCULAR_QUARTER_PIPE";
    else if (anEdgesMap.Contains(anEdgeC)) name = "CIRCULAR_QUARTER_PIPE";
    else if (anEdgesMap.Contains(anEdgeD)) name = "HALF_LENGTH_MAIN_PIPE";
    else if (anEdgesMap.Contains(anEdgeE)) name = "HALF_LENGTH_INCIDENT_PIPE";
    else if (anEdgesMap.Contains(anEdgeF)) name = "FLANGE";
    if (name != nullptr)
      theSeq.push_back({name, TopAbs_EDGE, aGroup});
  }
}

// AdvancedEngine_IOperations::MakeInternalGroup with no thickness reduction, ported: the
// faces on the main pipe's cylinder of radius `theR1` (axis x) whose first vertex has
// |x| <= theLen1, and those on the incident pipe's cylinder of radius `theR2` (axis z) whose
// first vertex has z <= theLen2, at the shape's largest vertex tolerance. GEOM names them
// INTERNAL_FACES.
std::vector<TopoDS_Shape> faces_on_pipes(const TopoDS_Shape& aShape, double theR1,
                                         double theLen1, double theR2, double theLen2) {
  // Compute tolerance
  Standard_Real aMaxTol = -RealLast();
  TopExp_Explorer anExp(aShape, TopAbs_VERTEX);

  for (; anExp.More(); anExp.Next()) {
    TopoDS_Vertex aVertex = TopoDS::Vertex(anExp.Current());

    if (aVertex.IsNull() == Standard_False) {
      const Standard_Real aTol = BRep_Tool::Tolerance(aVertex);

      if (aTol > aMaxTol) {
        aMaxTol = aTol;
      }
    }
  }

  gp_Ax2 anAxis1 (gp::Origin(), gp::DX(), gp::DZ());
  gp_Ax2 anAxis2 (gp::Origin(), gp::DZ(), gp::DX());
  const Handle(Geom_Surface) aSurface[2] = {new Geom_CylindricalSurface(anAxis1, theR1),
                                            new Geom_CylindricalSurface(anAxis2, theR2)};

  std::vector<TopoDS_Shape> aFaces;
  for (int i = 0; i < 2; i++) {
    for (const TopoDS_Shape& aFace : faces_on_surf(aShape, aSurface[i], aMaxTol)) {
      TopExp_Explorer aVtxExp(aFace, TopAbs_VERTEX);
      Standard_Boolean isInside = Standard_False;
      if (aVtxExp.More()) {
        gp_Pnt aPnt = BRep_Tool::Pnt(TopoDS::Vertex(aVtxExp.Current()));
        if (i == 0) {
          // Check if the point is inside the main pipe.
          isInside = (Abs(aPnt.X()) <= theLen1);
        } else { // i == 1
          // Check if the point is inside the incident pipe.
          isInside = (aPnt.Z() <= theLen2);
        }
      }
      if (isInside) {
        aFaces.push_back(aFace);
      }
    }
  }
  return aFaces;
}

}  // namespace

// ---- pipe_t_shape ------------------------------------------------------------------- //

py::dict Session::pipe_t_shape(double r1, double w1, double l1, double r2, double w2,
                               double l2, bool partition) {
  const char* op = "pipe_t_shape";
  finite_arg(op, "main_radius", r1);
  finite_arg(op, "main_thickness", w1);
  finite_arg(op, "main_half_length", l1);
  finite_arg(op, "branch_radius", r2);
  finite_arg(op, "branch_thickness", w2);
  finite_arg(op, "branch_half_length", l2);
  OpGuard guard(in_op_);
  const std::pair<const char*, double> named[] = {
      {"main_radius", r1},   {"main_thickness", w1},   {"main_half_length", l1},
      {"branch_radius", r2}, {"branch_thickness", w2}, {"branch_half_length", l2}};
  for (const auto& [name, value] : named) {
    if (!(value > Precision::Confusion())) {
      throw PysmeshError(std::string("Session.pipe_t_shape: ") + name + " " +
                         number_text(value) + " is not above 1e-07.");
    }
  }
  // AdvancedEngine_PipeTShapeDriver::Execute's checks for TSHAPE_BASIC, with GEOM's
  // epsilon (Precision::Approximation()).
  const double epsilon = Precision::Approximation();
  const double aR1Ext = r1 + w1;
  const double aR2Ext = r2 + w2;
  const std::string dims = " (main pipe: radius " + number_text(r1) + ", thickness " +
                           number_text(w1) + ", half length " + number_text(l1) +
                           "; branch: radius " + number_text(r2) + ", thickness " +
                           number_text(w2) + ", half length " + number_text(l2) + ")";
  if (r2 > r1 + epsilon) {
    throw PysmeshError("Session.pipe_t_shape: the branch radius is larger than the main "
                       "radius; GEOM's T-shape needs R2 <= R1" + dims + ".");
  }
  if (aR2Ext > aR1Ext + epsilon) {
    throw PysmeshError("Session.pipe_t_shape: the branch's outer radius is larger than the "
                       "main pipe's; GEOM's T-shape needs R2+W2 <= R1+W1" + dims + ".");
  }
  if (std::abs(aR2Ext - aR1Ext) < epsilon && std::abs(r2 - r1) > epsilon) {
    throw PysmeshError("Session.pipe_t_shape: the outer radii are equal and the inner ones "
                       "differ; GEOM's T-shape needs R2 = R1 when R2+W2 = R1+W1" + dims + ".");
  }
  if (aR1Ext >= l2 + epsilon) {
    throw PysmeshError("Session.pipe_t_shape: the main pipe's outer radius reaches the "
                       "branch's end; GEOM's T-shape needs R1+W1 < L2" + dims + ".");
  }
  if (aR2Ext >= l1 + epsilon) {
    throw PysmeshError("Session.pipe_t_shape: the branch's outer radius reaches the main "
                       "pipe's ends; GEOM's T-shape needs R2+W2 < L1" + dims + ".");
  }

  TopoDS_Shape result;
  std::vector<TShapeGroup> groups;
  std::string failure;
  {
    py::gil_scoped_release release;
    try {
      // AdvancedEngine_IOperations::MakePipeTShape: the driver's shape, then, prepared for a
      // hexahedral mesh, the partition and the mirror and glue of the quarter, the groups
      // of MakeGroups, and always the internal group.
      TopoDS_Shape aShape = partition ? MakeQuarterPipeTShape(r1, w1, l1, r2, w2, l2)
                                      : MakePipeTShapeWall(r1, w1, l1, r2, w2, l2);
      if (partition && failure.empty()) {
        const TopoDS_Shape Te3 = partition_quarter(aShape, r1, w1, l1, r2, w2, l2, failure);
        if (failure.empty()) {
          aShape = mirror_and_glue(Te3);
        }
      }
      if (failure.empty()) {
        if (partition) {
          make_groups(aShape, r1, w1, l1, r2, w2, l2, groups);
        }
        groups.push_back(
            {"INTERNAL_FACES", TopAbs_FACE, faces_on_pipes(aShape, r1, l1, r2, l2)});
        // The searches meshed the shape; the triangulation is not part of the result.
        BRepTools::Clean(aShape);
        result = aShape;
      }
    } catch (const std::exception& e) {
      failure = e.what();
    }
  }
  if (!failure.empty() || result.IsNull()) {
    throw PysmeshError("Session.pipe_t_shape: GEOM's T-shape construction failed" + dims +
                       ": " + (failure.empty() ? std::string("no shape was built") : failure) +
                       ".");
  }
  py::dict delta = add_bodies(result, op);
  py::list out_groups;
  for (const TShapeGroup& g : groups) {
    std::vector<EntityId> ids;
    for (const TopoDS_Shape& s : g.shapes) {
      ids.push_back(label_of(op, s));
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    py::dict d;
    d["name"] = g.name;
    d["kind"] = kind_name(g.kind);
    d["ids"] = ids_array(ids);
    out_groups.append(d);
  }
  py::dict out;
  out["delta"] = delta;
  out["groups"] = out_groups;
  return out;
}

}  // namespace session
}  // namespace pysmesh
