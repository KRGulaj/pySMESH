// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-22

// pySMESH binding — GEOM's BlockFix steps, copied with their history (see geom_blockfix.hpp).
//
// The functions below are SALOME common_geometry_lib tag V9_16_0, src/BlockFix/BlockFix.cxx
// and src/BlockFix/BlockFix_BlockFixAPI.cxx (BlockFix_BlockFixAPI::Perform):
//
//   Copyright (C) 2007-2026  CEA, EDF, OPEN CASCADE
//   Copyright (C) 2003-2007  OPEN CASCADE, EADS/CCR, LIP6, CEA/DEN,
//   CEDRAT, EDF R&D, LEG, PRINCIPIA R&D, BUREAU VERITAS
//   This library is free software; you can redistribute it and/or modify it under the
//   terms of the GNU Lesser General Public License as published by the Free Software
//   Foundation; either version 2.1 of the License, or (at your option) any later version.
//
// Used here under LGPL-2.1 and recorded in PROVENANCE.md. The code is unchanged but for the
// lines marked "pySMESH:", which do five things:
//  * each step writes its history (the ReShape that ShapeCustom::ApplyModifier fills, the
//    ReShape of FixResult, the substitution of RefillProblemFaces, the unifiers' own
//    histories, and the rebuild of ShapeUpgrade_RemoveLocations), appended to theSteps;
//  * Perform is a free function of its members (theShape, myTolerance, myOptimumNbFaces)
//    that returns the result, and it calls the copies, not BlockFix's own functions;
//  * Perform keeps only the OCCT >= 7.5.3 branches and the face unification of
//    myOptimumNbFaces 0 and -1, the two values GEOM's RemoveExtraEdges passes; the branch
//    for a number of faces above 1 (BlockFix_UnionFaces) is not copied;
//  * a progress scope gives each step a share and stops the steps after a cancel;
//  * a face that RotateSphereSpace or FixRanges puts on a turned sphere or cylinder must
//    keep its area where the turned shape is used (require_turned_faces_kept).

#include "geom_blockfix.hpp"

#include "session/session.hpp"

#include <BlockFix_PeriodicSurfaceModifier.hxx>
#include <BlockFix_SphereSpaceModifier.hxx>

#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepOffsetAPI_MakeFilling.hxx>
#include <BRepTools.hxx>
#include <BRepTools_Modifier.hxx>
#include <BRepTools_Substitution.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <Extrema_ExtPS.hxx>
#include <GeomAdaptor_Surface.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_SphericalSurface.hxx>
#include <Geom_Surface.hxx>
#include <Precision.hxx>
#include <ShapeAnalysis_Edge.hxx>
#include <ShapeAnalysis_Surface.hxx>
#include <ShapeBuild_ReShape.hxx>
#include <ShapeCustom.hxx>
#include <ShapeExtend_WireData.hxx>
#include <ShapeFix.hxx>
#include <ShapeFix_Edge.hxx>
#include <ShapeFix_Face.hxx>
#include <ShapeFix_Wire.hxx>
#include <ShapeUpgrade_RemoveLocations.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_DataMapOfShapeShape.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_MapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Vertex.hxx>

#include <Basics_OCCTVersion.hxx>

namespace pysmesh {
namespace geomcopy {

static Standard_Real ComputeMaxTolOfFace(const TopoDS_Face& theFace)
{
  Standard_Real MaxTol = BRep_Tool::Tolerance(theFace);

  TopTools_IndexedMapOfShape aMap;
  TopExp::MapShapes(theFace, TopAbs_EDGE, aMap);
  for (Standard_Integer i = 1; i <= aMap.Extent(); i++)
  {
    const TopoDS_Edge& anEdge = TopoDS::Edge(aMap(i));
    Standard_Real aTol = BRep_Tool::Tolerance(anEdge);
    if (aTol > MaxTol)
      MaxTol = aTol;
  }

  aMap.Clear();
  TopExp::MapShapes(theFace, TopAbs_VERTEX, aMap);
  for (Standard_Integer i = 1; i <= aMap.Extent(); i++)
  {
    const TopoDS_Vertex& aVertex = TopoDS::Vertex(aMap(i));
    Standard_Real aTol = BRep_Tool::Tolerance(aVertex);
    if (aTol > MaxTol)
      MaxTol = aTol;
  }

  return MaxTol;
}

//=======================================================================
//function : FixResult
//purpose  : auxiliary
//=======================================================================
static void FixResult(const TopoDS_Shape& result,
                      Handle(ShapeBuild_ReShape)& Context,
                      const Standard_Real Tol)
{
  for (TopExp_Explorer ex_f(result,TopAbs_FACE); ex_f.More(); ex_f.Next()) {
    TopoDS_Shape aShape = Context->Apply(ex_f.Current().Oriented(TopAbs_FORWARD));
    // face could not be dropped or split on this step
    TopoDS_Face aFace = TopoDS::Face(aShape);
    TopLoc_Location L;
    Handle(Geom_Surface) Surf = BRep_Tool::Surface(aFace,L);

    if( Surf->IsKind(STANDARD_TYPE(Geom_SphericalSurface)) ||
        Surf->IsKind(STANDARD_TYPE(Geom_CylindricalSurface)) ) {

      Standard_Integer nbWires = 0;
      for (TopExp_Explorer ex_w(aFace,TopAbs_WIRE); ex_w.More(); ex_w.Next()) {
        nbWires++;
        Handle(ShapeFix_Wire) sfw = new ShapeFix_Wire(TopoDS::Wire(ex_w.Current()),
                                                      aFace,
                                                      Precision::Confusion());
        sfw->FixReorder();
        if(sfw->StatusReorder ( ShapeExtend_FAIL ))
          continue;

        sfw->SetPrecision(2.*Tol);
        sfw->FixShifted();

        Standard_Boolean isDone = sfw->LastFixStatus ( ShapeExtend_DONE );
        isDone |= sfw->FixDegenerated();

        // remove degenerated edges from not degenerated points
        ShapeAnalysis_Edge sae;
        Handle(ShapeExtend_WireData) sewd = sfw->WireData();
        Standard_Integer i;
        for( i = 1; i<=sewd->NbEdges();i++) {
          TopoDS_Edge E = sewd->Edge(i);
          if(BRep_Tool::Degenerated(E)&&!sae.HasPCurve(E,aFace)) {
            sewd->Remove(i);
            isDone = Standard_True;
            i--;
          }
        }

        //isDone |= sfw->FixLacking(); // commented by skl 22.03.2005 (PAL8395)

        // remove neighbour seam edges
        if(isDone) {
          for( i = 1; i<sewd->NbEdges();i++) {
            if(sewd->IsSeam(i) && sewd->IsSeam(i+1)) {
              isDone = Standard_True;
              sewd->Remove(i);
              sewd->Remove(i);
              i--;
            }
          }
          if(sewd->IsSeam(1) && sewd->IsSeam(sewd->NbEdges())) {
            sewd->Remove(1);
            sewd->Remove(sewd->NbEdges());
          }
        }

        if(isDone) {
          TopoDS_Wire ResWire = sfw->Wire();
          Context->Replace(ex_w.Current(), ResWire);
        };
      }
      // Implement fix orientation in case of several wires
      if(nbWires > 1) {
        TopoDS_Face aFixedFace = TopoDS::Face(Context->Apply(aFace));
        Handle(ShapeFix_Face) sff = new ShapeFix_Face(aFixedFace);
        if(sff->FixOrientation())
          Context->Replace(aFixedFace,sff->Face());
      }
    }
  }
}

//=======================================================================
//function : RotateSphereSpace
//purpose  :
//=======================================================================
TopoDS_Shape RotateSphereSpace (const TopoDS_Shape& S,
                                const Standard_Real Tol,
                                const Standard_Boolean theTrySmallRotation,
                                const Message_ProgressRange& theRange,  // pySMESH:
                                std::vector<Handle(BRepTools_History)>& theSteps)
{
  // Create a modification description
  Handle(BlockFix_SphereSpaceModifier) SR = new BlockFix_SphereSpaceModifier;
  SR->SetTolerance(Tol);
  SR->SetTrySmallRotation(theTrySmallRotation);

  TopTools_DataMapOfShapeShape context;
  BRepTools_Modifier MD;
  // pySMESH: ApplyModifier records each sub-shape the modifier replaces in a ReShape
  // of its own (OCCT's aReShape argument), and that ReShape's history is the step's.
  Handle(ShapeBuild_ReShape) aRecord = new ShapeBuild_ReShape;
  TopoDS_Shape result = ShapeCustom::ApplyModifier ( S, SR, context, MD, theRange,
                                                     aRecord );
  theSteps.push_back(aRecord->History());

  Handle(ShapeBuild_ReShape) RS = new ShapeBuild_ReShape;
  FixResult(result,RS,Tol);
  result = RS->Apply(result);
  theSteps.push_back(RS->History());  // pySMESH: FixResult's step

  ShapeFix_Edge sfe;
  for (TopExp_Explorer exp (result, TopAbs_EDGE); exp.More(); exp.Next()) {
    TopoDS_Edge E = TopoDS::Edge(exp.Current());
    sfe.FixVertexTolerance (E);
  }

  ShapeFix::SameParameter(result, Standard_False);
  return result;
}

//=======================================================================
//function : RefillProblemFaces
//purpose  :
//=======================================================================
TopoDS_Shape RefillProblemFaces (const TopoDS_Shape& aShape,
                                 Handle(BRepTools_History)& theHistory)  // pySMESH:
{
  Standard_Integer NbSamples = 10;

#if OCC_VERSION_LARGE > 0x07040000
  TopTools_IndexedDataMapOfShapeListOfShape VFmap;
  TopExp::MapShapesAndUniqueAncestors(aShape, TopAbs_VERTEX, TopAbs_FACE, VFmap);
#endif

  TopTools_ListOfShape theFaces;

  TopExp_Explorer Explo(aShape, TopAbs_FACE);
  for (; Explo.More(); Explo.Next())
  {
    TopoDS_Face aFace = TopoDS::Face(Explo.Current());
    BRepAdaptor_Surface BAsurf(aFace);
    GeomAbs_SurfaceType SurfType = BAsurf.GetType();
    if (SurfType >= GeomAbs_BezierSurface)
    {
      TopExp_Explorer fexp(aFace, TopAbs_EDGE);
      for (; fexp.More(); fexp.Next())
      {
        const TopoDS_Edge& anEdge = TopoDS::Edge(fexp.Current());
        if (BRep_Tool::Degenerated(anEdge))
        {
          TopoDS_Vertex V1, V2;
          TopExp::Vertices(anEdge, V1, V2);
          if (V1.IsSame(V2))
          {
            gp_Pnt aPnt = BRep_Tool::Pnt(V1);
            Standard_Real TolV = BRep_Tool::Tolerance(V1);
            Handle(Geom_Surface) aSurf = BRep_Tool::Surface(aFace);
            Handle(ShapeAnalysis_Surface) Analyser = new ShapeAnalysis_Surface(aSurf);
            if (Analyser->IsDegenerated(aPnt, TolV))
            {
              theFaces.Append(aFace);
              break;
            }
          }
        }
      }
    }
  }

  //Now all problem faces are collected in the list "theFaces"
  BRepTools_Substitution aSubst;
  TopTools_ListIteratorOfListOfShape itl(theFaces);
  for (; itl.More(); itl.Next())
  {
    TopoDS_Face aFace = TopoDS::Face(itl.Value());
    aFace.Orientation(TopAbs_FORWARD);

    //Compute proper tolerance
    Standard_Real MaxTolOfFace = ComputeMaxTolOfFace(aFace);
#if OCC_VERSION_LARGE > 0x07040000
    Bnd_Box aBndBox;
    BRepBndLib::Add(aFace, aBndBox);
    Standard_Real Xmin, Ymin, Zmin, Xmax, Ymax, Zmax;
    aBndBox.Get(Xmin, Ymin, Zmin, Xmax, Ymax, Zmax);
    Standard_Real LinSize = Min(Xmax - Xmin, Min(Ymax - Ymin, Zmax - Zmin));
    Standard_Real LinTol = Max(0.001*LinSize, 1.5*MaxTolOfFace);
#endif
    
    BRepAdaptor_Surface BAsurf(aFace, Standard_False);
    BRepOffsetAPI_MakeFilling Filler(3, 10);
    TopExp_Explorer Explo(aFace, TopAbs_EDGE);
    for (; Explo.More(); Explo.Next())
    {
      const TopoDS_Edge& anEdge = TopoDS::Edge(Explo.Current());
      if (BRep_Tool::Degenerated(anEdge) ||
          BRepTools::IsReallyClosed(anEdge, aFace))
        continue;
      
      Filler.Add(anEdge, GeomAbs_C0);
      //Filler.Add(anEdge, aFace, GeomAbs_G1);
    }
    Standard_Real Umin, Umax, Vmin, Vmax;
    BRepTools::UVBounds(aFace, Umin, Umax, Vmin, Vmax);
    Standard_Real DeltaU = (Umax - Umin)/NbSamples,
      DeltaV = (Vmax - Vmin)/NbSamples;
    for (Standard_Integer i = 1; i < NbSamples; i++)
      for (Standard_Integer j = 1; j < NbSamples; j++) {
        Filler.Add(Umin + i*DeltaU, Vmin + j*DeltaV, aFace, GeomAbs_G1);
      }

    Filler.Build();
    if (Filler.IsDone())
    {
      TopoDS_Face aNewFace = TopoDS::Face(Filler.Shape());
      aNewFace.Orientation(TopAbs_FORWARD);
      Handle(Geom_Surface) aNewSurf = BRep_Tool::Surface(aNewFace);
      GeomAdaptor_Surface GAnewsurf(aNewSurf);
      Extrema_ExtPS Projector;
      Projector.Initialize(GAnewsurf, GAnewsurf.FirstUParameter(), GAnewsurf.LastUParameter(),
                           GAnewsurf.FirstVParameter(), GAnewsurf.LastVParameter(),
                           Precision::Confusion(), Precision::Confusion());
      Standard_Real MaxSqDist = 0.;
      for (Standard_Integer i = 0; i < NbSamples; i++)
        for (Standard_Integer j = 0; j < NbSamples; j++)
        {
          gp_Pnt aPoint = BAsurf.Value(Umin + DeltaU/2 + i*DeltaU,
                                       Vmin + DeltaV/2 + j*DeltaV);
          Projector.Perform(aPoint);
          if (Projector.IsDone())
          {
            Standard_Real LocalMinSqDist = RealLast();
            for (Standard_Integer ind = 1; ind <= Projector.NbExt(); ind++)
            {
              Standard_Real aSqDist = Projector.SquareDistance(ind);
              if (aSqDist < LocalMinSqDist)
                LocalMinSqDist = aSqDist;
            }
            if (!Precision::IsInfinite(LocalMinSqDist) &&
                LocalMinSqDist > MaxSqDist)
              MaxSqDist = LocalMinSqDist;
          }
        }
      Standard_Real MaxDist = Sqrt(MaxSqDist);
#if OCC_VERSION_LARGE > 0x07040000
      if (MaxDist < LinTol)
#else
      if (MaxDist < Max(1.e-4, 1.5*MaxTolOfFace))
#endif
      {
        TopTools_IndexedMapOfShape Emap;
#if OCC_VERSION_LARGE > 0x07040000
        TopTools_MapOfShape Vmap;
#endif
        TopExp::MapShapes(aFace, TopAbs_EDGE, Emap);
        for (Standard_Integer i = 1; i <= Emap.Extent(); i++)
        {
          TopoDS_Edge anEdge = TopoDS::Edge(Emap(i));
          anEdge.Orientation(TopAbs_FORWARD);
#if OCC_VERSION_LARGE > 0x07040000
          TopoDS_Vertex V1, V2;
          TopExp::Vertices(anEdge, V1, V2);
          TopTools_ListOfShape ListV1, ListV2;
#endif
          TopTools_ListOfShape Ledge;
          if (!BRep_Tool::Degenerated(anEdge) &&
              !BRepTools::IsReallyClosed(anEdge, aFace))
          {
            const TopTools_ListOfShape& Ledges = Filler.Generated(anEdge);
            if (!Ledges.IsEmpty()) {
              TopoDS_Edge NewEdge = TopoDS::Edge(Ledges.First());
              Ledge.Append(NewEdge.Oriented(TopAbs_FORWARD));
            }

#if OCC_VERSION_LARGE > 0x07040000
            TopoDS_Vertex NewV1 = TopoDS::Vertex(Filler.Generated(V1).First());
            ListV1.Append(NewV1.Oriented(TopAbs_FORWARD));

            if (!V1.IsSame(V2)) {
              TopoDS_Vertex NewV2 = TopoDS::Vertex(Filler.Generated(V2).First());
              ListV2.Append(NewV2.Oriented(TopAbs_FORWARD));
            }
#endif
          }
          aSubst.Substitute(anEdge, Ledge);
#if OCC_VERSION_LARGE > 0x07040000
          if (!Vmap.Contains(V1) &&
              (!ListV1.IsEmpty() || VFmap.FindFromKey(V1).Extent() == 1))
          {
            aSubst.Substitute(V1, ListV1);
            Vmap.Add(V1);
          }
          if (!Vmap.Contains(V2) &&
              (!ListV2.IsEmpty() || VFmap.FindFromKey(V2).Extent() == 1))
          {
            aSubst.Substitute(V2.Oriented(TopAbs_FORWARD), ListV2);
            Vmap.Add(V2);
          }
#endif
        }
        TopTools_ListOfShape Lface;
        BRepAdaptor_Surface NewBAsurf(aNewFace);
        gp_Pnt MidPnt;
        gp_Vec D1U, D1V, Normal, NewNormal;
        Handle(Geom_Surface) aSurf = BRep_Tool::Surface(aFace);
        aSurf->D1((Umin+Umax)*0.5, (Vmin+Vmax)*0.5, MidPnt, D1U, D1V);
        Normal = D1U ^ D1V;
        NewBAsurf.D1((NewBAsurf.FirstUParameter() + NewBAsurf.LastUParameter())*0.5,
                     (NewBAsurf.FirstVParameter() + NewBAsurf.LastVParameter())*0.5,
                     MidPnt, D1U, D1V);
        NewNormal = D1U ^ D1V;
        if (Normal * NewNormal < 0.)
          aNewFace.Reverse();
        Lface.Append(aNewFace);
        aSubst.Substitute(aFace, Lface);
      }
    }
  }
  aSubst.Build(aShape);
  // pySMESH: the substitution's history.
  theHistory = session::substitution_history(aShape, aSubst);

  TopoDS_Shape Result = aShape;
  if (aSubst.IsCopied(aShape))
    Result = aSubst.Copy(aShape).First();

  BRepTools::RemoveUnusedPCurves(Result);

  return Result;
}

//=======================================================================
//function : FixRanges
//purpose  :
//=======================================================================
TopoDS_Shape FixRanges (const TopoDS_Shape& S,
                        const Standard_Real Tol,
                        const Message_ProgressRange& theRange,  // pySMESH:
                        std::vector<Handle(BRepTools_History)>& theSteps)
{
  // Create a modification description
  Handle(BlockFix_PeriodicSurfaceModifier) SR = new BlockFix_PeriodicSurfaceModifier;
  SR->SetTolerance(Tol);

  TopTools_DataMapOfShapeShape context;
  BRepTools_Modifier MD;
  // pySMESH: ApplyModifier records each sub-shape the modifier replaces in a ReShape
  // of its own (OCCT's aReShape argument), and that ReShape's history is the step's.
  Handle(ShapeBuild_ReShape) aRecord = new ShapeBuild_ReShape;
  TopoDS_Shape result = ShapeCustom::ApplyModifier ( S, SR, context, MD, theRange,
                                                     aRecord );
  theSteps.push_back(aRecord->History());

  Handle(ShapeBuild_ReShape) RS = new ShapeBuild_ReShape;
  FixResult(result,RS,Tol);
  result = RS->Apply(result);
  theSteps.push_back(RS->History());  // pySMESH: FixResult's step

  ShapeFix_Edge sfe;
  for(TopExp_Explorer exp(result,TopAbs_EDGE); exp.More(); exp.Next()) {
    TopoDS_Edge E = TopoDS::Edge(exp.Current());
    sfe.FixVertexTolerance (E);
  }

  ShapeFix::SameParameter(result,Standard_False);

  return result;
}

//=======================================================================
//function : Perform
//purpose  :
//=======================================================================
TopoDS_Shape BlockFixAPI_Perform(const TopoDS_Shape& theShape,  // pySMESH: the members
                                 const Standard_Real myTolerance,
                                 const Standard_Integer myOptimumNbFaces,
                                 Message_ProgressScope& thePS,
                                 std::vector<Handle(BRepTools_History)>& theSteps)
{
  TopoDS_Shape myShape;  // pySMESH: the member
  // processing spheres with degenerativities
  TopoDS_Shape aShape = theShape;
  // Copy the shape to avoid modification of initial shape
  // while trying the approach with small rotation
  BRepBuilderAPI_Copy aMC (aShape);
  if (!aMC.IsDone()) return aShape;  // pySMESH: myShape is the shape set
  TopoDS_Shape aSCopy = aMC.Shape();
  // pySMESH: the copy's history, then the rotation's steps; they count only when the
  // rotated copy is taken.
  std::vector<Handle(BRepTools_History)> aRotateSteps;
  NCollection_List<TopoDS_Shape> anArguments;
  anArguments.Append(aShape);
  aRotateSteps.push_back(new BRepTools_History(anArguments, aMC));
  TopoDS_Shape aNewShape = RotateSphereSpace(aSCopy, myTolerance, Standard_True,
                                             thePS.Next(), aRotateSteps);
  BRepCheck_Analyzer ana (aNewShape, false);
  if (ana.IsValid()) {
    if (aNewShape == aSCopy)
      myShape = aShape;
    else {
      myShape = aNewShape;
      // pySMESH: the turned faces keep their areas; the copy's own history is left out.
      session::require_turned_faces_kept(
          "RotateSphereSpace", aSCopy, aNewShape,
          std::vector<Handle(BRepTools_History)>(aRotateSteps.begin() + 1,
                                                 aRotateSteps.end()));
      theSteps.insert(theSteps.end(), aRotateSteps.begin(), aRotateSteps.end());  // pySMESH:
    }
  }
  else {
    // pySMESH: the steps of this call, checked as above.
    std::vector<Handle(BRepTools_History)> aFallbackSteps;
    myShape = RotateSphereSpace(aShape, myTolerance, Standard_False,
                                Message_ProgressRange(), aFallbackSteps);
    session::require_turned_faces_kept("RotateSphereSpace", aShape, myShape,
                                       aFallbackSteps);
    theSteps.insert(theSteps.end(), aFallbackSteps.begin(), aFallbackSteps.end());
  }

  // try to approximate non-canonic surfaces
  // with singularities on boundaries by filling
  if (!thePS.More()) return TopoDS_Shape();  // pySMESH: cancelled
  Handle(BRepTools_History) aRefillHistory;  // pySMESH:
  myShape = RefillProblemFaces(myShape, aRefillHistory);
  theSteps.push_back(aRefillHistory);  // pySMESH:
  thePS.Next();  // pySMESH:

  // faces unification
  TopoDS_Shape aResult = myShape;
  // pySMESH: the branches for OCCT < 7.5.3 and for myOptimumNbFaces > 1
  // (BlockFix_UnionFaces) are not copied.
  if (!thePS.More()) return TopoDS_Shape();  // pySMESH: cancelled
  if (myOptimumNbFaces != -1) {
    // use OCCT algo ShapeUpgrade_UnifySameDomain
    ShapeUpgrade_UnifySameDomain Unifier;
    //only faces
    Standard_Boolean isUnifyEdges = Standard_False;
    Standard_Boolean isUnifyFaces = Standard_True;
    Standard_Boolean isConcatBSplines = Standard_True;
    Unifier.Initialize(myShape, isUnifyEdges, isUnifyFaces, isConcatBSplines);
    //Unifier.SetLinearTolerance(myTolerance);
    Unifier.SetLinearTolerance(Precision::Confusion());
    Unifier.SetAngularTolerance(Precision::Confusion());
    //Unifier.SetAngularTolerance(1e-5);
    //Unifier.SetAngularTolerance(0.1);
    Unifier.Build();
    aResult = Unifier.Shape();
    theSteps.push_back(Unifier.History());  // pySMESH:
  }
  else {
    // myOptimumNbFaces == -1 means do not union faces
  }
  thePS.Next();  // pySMESH:

  // avoid problem with degenerated edges appearance
  // due to shape quality regress
  ShapeUpgrade_RemoveLocations RemLoc;
  if (!thePS.More()) return TopoDS_Shape();  // pySMESH: cancelled
  RemLoc.Remove(aResult);
  // pySMESH: the rebuild keeps the structure; its history pairs the two shapes.
  theSteps.push_back(session::structural_history(aResult, RemLoc.GetResult()));
  aResult = RemLoc.GetResult();
  thePS.Next();  // pySMESH:

  // edges unification
  if (!thePS.More()) return TopoDS_Shape();  // pySMESH: cancelled
  ShapeUpgrade_UnifySameDomain Unifier;
  Standard_Boolean isUnifyEdges = Standard_True;
  Standard_Boolean isUnifyFaces = Standard_False; //only edges
  Standard_Boolean isConcatBSplines = Standard_True;
  Unifier.Initialize(aResult, isUnifyEdges, isUnifyFaces, isConcatBSplines);
  Unifier.SetLinearTolerance(myTolerance);
  Unifier.SetAngularTolerance(1e-5);
  Unifier.Build();
  myShape = Unifier.Shape();
  theSteps.push_back(Unifier.History());  // pySMESH:
  thePS.Next();  // pySMESH:

  if (!thePS.More()) return TopoDS_Shape();  // pySMESH: cancelled
  std::vector<Handle(BRepTools_History)> aRangeSteps;  // pySMESH:
  TopoDS_Shape aRes = FixRanges(myShape,myTolerance, thePS.Next(), aRangeSteps);
  session::require_turned_faces_kept("FixRanges", myShape, aRes, aRangeSteps);  // pySMESH:
  theSteps.insert(theSteps.end(), aRangeSteps.begin(), aRangeSteps.end());  // pySMESH:
  myShape = aRes;
  return myShape;  // pySMESH:
}

}  // namespace geomcopy
}  // namespace pysmesh
