// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH binding — GEOM's edge-from-wire and C1 test, copied (see geom_edge_from_wire.hpp).
//
// MakeEdgeFromWire is SALOME GEOM tag V9_16_0, src/GEOMImpl/GEOMImpl_ShapeDriver.cxx
// (GEOMImpl_ShapeDriver::MakeEdgeFromWire), and AreEdgesC1 is
// src/GEOMImpl/GEOMImpl_HealingDriver.cxx (GEOMImpl_HealingDriver::AreEdgesC1); both
// Copyright (C) 2007-2026 CEA, EDF, OPEN CASCADE, LGPL-2.1 or later, used here under
// LGPL-2.1; recorded in PROVENANCE.md. They are free functions here instead of static
// members, and the code is unchanged but for one addition and one OCCT 8.0.1 port change:
// MakeEdgeFromWire hands back the history of the ShapeFix_Shape it runs on the wire
// (theFixHistory), read from the fixer's own context after Perform, so a caller can carry
// ids across it; and its one Standard_Failure::Raise, gone in OCCT 8.0.1, is a throw of the
// same exception with the same text. Two upstream to-do comments on CurType are removed;
// each keeps its fact as a plain comment.

#include "geom_edge_from_wire.hpp"

#include <BRepAdaptor_Curve.hxx>
#include <BRepLib.hxx>
#include <BRepLib_MakeEdge.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <BRep_Tool.hxx>
#include <ElCLib.hxx>
#include <GeomAdaptor_Curve.hxx>
#include <GeomConvert.hxx>
#include <GeomConvert_CompCurveToBSplineCurve.hxx>
#include <GeomLProp.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_Curve.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <Precision.hxx>
#include <ShapeBuild_ReShape.hxx>
#include <ShapeFix_Shape.hxx>
#include <Standard_ConstructionError.hxx>
#include <TColGeom_Array1OfBSplineCurve.hxx>
#include <TColGeom_HArray1OfBSplineCurve.hxx>
#include <TColGeom_SequenceOfCurve.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColStd_HArray1OfInteger.hxx>
#include <TColStd_SequenceOfReal.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_SequenceOfShape.hxx>
#include <TopoDS.hxx>
#include <gp_Circ.hxx>
#include <gp_Elips.hxx>
#include <gp_Hypr.hxx>
#include <gp_Lin.hxx>
#include <gp_Parab.hxx>

namespace pysmesh {
namespace geomcopy {

TopoDS_Edge MakeEdgeFromWire(const TopoDS_Shape& aWire,
                             const Standard_Real LinTol,
                             const Standard_Real AngTol,
                             Handle(BRepTools_History)* theFixHistory)
{
    TopoDS_Edge ResEdge;

    BRepLib::BuildCurves3d(aWire);
    Handle(ShapeFix_Shape) Fixer = new ShapeFix_Shape(aWire);
    Fixer->SetPrecision(LinTol);
    Fixer->SetMaxTolerance(LinTol);
    Fixer->Perform();
    TopoDS_Wire theWire = TopoDS::Wire(Fixer->Shape());
    if (theFixHistory != nullptr)
      *theFixHistory = Fixer->Context()->History();

    TColGeom_SequenceOfCurve CurveSeq;
    TopTools_SequenceOfShape LocSeq;
    TColStd_SequenceOfReal FparSeq;
    TColStd_SequenceOfReal LparSeq;
    TColStd_SequenceOfReal TolSeq;
    GeomAbs_CurveType CurType = GeomAbs_OtherCurve; // CurType is initialized explicitly to avoid a warning (see below)
    TopoDS_Vertex FirstVertex, LastVertex;
    Standard_Real aPntShiftDist = 0.;

    BRepTools_WireExplorer wexp(theWire) ;
    for (; wexp.More(); wexp.Next())
    {
      TopoDS_Edge anEdge = wexp.Current();
      Standard_Real fpar, lpar;
      TopLoc_Location aLoc;
      Handle(Geom_Curve) aCurve = BRep_Tool::Curve(anEdge, aLoc, fpar, lpar);
      if (aCurve.IsNull())
        continue;

      BRepAdaptor_Curve BAcurve(anEdge);
      GeomAbs_CurveType aType = BAcurve.GetType();

      Handle(Geom_Curve) aBasisCurve = BAcurve.Curve().Curve();

      if (aBasisCurve->IsPeriodic())
        ElCLib::AdjustPeriodic(aBasisCurve->FirstParameter(), aBasisCurve->LastParameter(),
                               Precision::PConfusion(), fpar, lpar);

      if (CurveSeq.IsEmpty())
      {
        CurveSeq.Append(aCurve);
        TopoDS_Shape aLocShape;
        aLocShape.Location(aLoc);
        aLocShape.Orientation(wexp.Orientation());
        LocSeq.Append(aLocShape);
        FparSeq.Append(fpar);
        LparSeq.Append(lpar);
        CurType = aType;                     // CurType is initialized explicitly to avoid a warning (see above)
        FirstVertex = wexp.CurrentVertex();
      }
      else
      {
        Standard_Boolean Done = Standard_False;
        Standard_Real NewFpar, NewLpar;
        Handle(Geom_Geometry) aTrsfGeom = CurveSeq.Last()->Transformed
                      (LocSeq.Last().Location().Transformation());
        GeomAdaptor_Curve GAprevcurve(Handle(Geom_Curve)::DownCast(aTrsfGeom));
        TopoDS_Vertex CurVertex = wexp.CurrentVertex();
        TopoDS_Vertex CurFirstVer = TopExp::FirstVertex(anEdge);
        TopAbs_Orientation ConnectByOrigin = (CurVertex.IsSame(CurFirstVer))? TopAbs_FORWARD : TopAbs_REVERSED;
        if (aCurve == CurveSeq.Last() && aLoc.IsEqual(LocSeq.Last().Location()))
        {
          NewFpar = fpar;
          NewLpar = lpar;
          if (aBasisCurve->IsPeriodic())
          {
            if (NewLpar < NewFpar)
              NewLpar += aBasisCurve->Period();
            if (ConnectByOrigin == TopAbs_FORWARD)
              ElCLib::AdjustPeriodic(FparSeq.Last(),
                                     FparSeq.Last() + aBasisCurve->Period(),
                                     Precision::PConfusion(), NewFpar, NewLpar);
            else
              ElCLib::AdjustPeriodic(FparSeq.Last() - aBasisCurve->Period(),
                                     FparSeq.Last(),
                                     Precision::PConfusion(), NewFpar, NewLpar);
          }
          Done = Standard_True;
        }
        else if (aType == CurType &&
                 aType != GeomAbs_BezierCurve &&
                 aType != GeomAbs_BSplineCurve &&
                 aType != GeomAbs_OtherCurve)
        {
          switch (aType)
          {
          case GeomAbs_Line:
            {
              gp_Lin aLine    = BAcurve.Line();
              gp_Lin PrevLine = GAprevcurve.Line();
              if (aLine.Contains(PrevLine.Location(), LinTol) &&
                  aLine.Direction().IsParallel(PrevLine.Direction(), AngTol))
              {
                gp_Pnt P1 = ElCLib::Value(fpar, aLine);
                gp_Pnt P2 = ElCLib::Value(lpar, aLine);
                NewFpar = ElCLib::Parameter(PrevLine, P1);
                NewLpar = ElCLib::Parameter(PrevLine, P2);

                // Compute shift
                if (ConnectByOrigin == TopAbs_FORWARD) {
                  gp_Pnt aNewP2 = ElCLib::Value(NewLpar, PrevLine);

                  aPntShiftDist += P2.Distance(aNewP2);
                } else {
                  gp_Pnt aNewP1 = ElCLib::Value(NewFpar, PrevLine);

                  aPntShiftDist += P1.Distance(aNewP1);
                }

                if (NewLpar < NewFpar)
                {
                  Standard_Real MemNewFpar = NewFpar;
                  NewFpar = NewLpar;
                  NewLpar = MemNewFpar;
                  ConnectByOrigin = TopAbs::Reverse(ConnectByOrigin);
                }
                Done = Standard_True;
              }
              break;
            }
          case GeomAbs_Circle:
            {
              gp_Circ aCircle    = BAcurve.Circle();
              gp_Circ PrevCircle = GAprevcurve.Circle();
              if (aCircle.Location().Distance(PrevCircle.Location()) <= LinTol &&
                  Abs(aCircle.Radius() - PrevCircle.Radius()) <= LinTol &&
                  aCircle.Axis().IsParallel(PrevCircle.Axis(), AngTol))
              {
                const Standard_Boolean isFwd = ConnectByOrigin == TopAbs_FORWARD;

                if (aCircle.Axis().Direction() * PrevCircle.Axis().Direction() < 0.)
                {
                  Standard_Real memfpar = fpar;
                  fpar = lpar;
                  lpar = memfpar;
                  ConnectByOrigin = TopAbs::Reverse(ConnectByOrigin);
                }
                gp_Pnt P1 = ElCLib::Value(fpar, aCircle);
                gp_Pnt P2 = ElCLib::Value(lpar, aCircle);
                NewFpar = ElCLib::Parameter(PrevCircle, P1);
                NewLpar = ElCLib::Parameter(PrevCircle, P2);

                // Compute shift
                if (isFwd) {
                  gp_Pnt aNewP2 = ElCLib::Value(NewLpar, PrevCircle);

                  aPntShiftDist += P2.Distance(aNewP2);
                } else {
                  gp_Pnt aNewP1 = ElCLib::Value(NewFpar, PrevCircle);

                  aPntShiftDist += P1.Distance(aNewP1);
                }

                if (NewLpar < NewFpar)
                  NewLpar += 2.*M_PI;
                //Standard_Real MemNewFpar = NewFpar, MemNewLpar =  NewLpar;
                if (ConnectByOrigin == TopAbs_FORWARD)
                  ElCLib::AdjustPeriodic(FparSeq.Last(),
                                         FparSeq.Last() + 2.*M_PI,
                                         Precision::PConfusion(), NewFpar, NewLpar);
                else
                  ElCLib::AdjustPeriodic(FparSeq.Last() - 2.*M_PI,
                                         FparSeq.Last(),
                                         Precision::PConfusion(), NewFpar, NewLpar);
                Done = Standard_True;
              }
              break;
            }
          case GeomAbs_Ellipse:
            {
              gp_Elips anEllipse   = BAcurve.Ellipse();
              gp_Elips PrevEllipse = GAprevcurve.Ellipse();
              if (anEllipse.Focus1().Distance(PrevEllipse.Focus1()) <= LinTol &&
                  anEllipse.Focus2().Distance(PrevEllipse.Focus2()) <= LinTol &&
                  Abs(anEllipse.MajorRadius() - PrevEllipse.MajorRadius()) <= LinTol &&
                  Abs(anEllipse.MinorRadius() - PrevEllipse.MinorRadius()) <= LinTol &&
                  anEllipse.Axis().IsParallel(PrevEllipse.Axis(), AngTol))
              {
                const Standard_Boolean isFwd = ConnectByOrigin == TopAbs_FORWARD;

                if (anEllipse.Axis().Direction() * PrevEllipse.Axis().Direction() < 0.)
                {
                  Standard_Real memfpar = fpar;
                  fpar = lpar;
                  lpar = memfpar;
                  ConnectByOrigin = TopAbs::Reverse(ConnectByOrigin);
                }
                gp_Pnt P1 = ElCLib::Value(fpar, anEllipse);
                gp_Pnt P2 = ElCLib::Value(lpar, anEllipse);
                NewFpar = ElCLib::Parameter(PrevEllipse, P1);
                NewLpar = ElCLib::Parameter(PrevEllipse, P2);

                // Compute shift
                if (isFwd) {
                  gp_Pnt aNewP2 = ElCLib::Value(NewLpar, PrevEllipse);

                  aPntShiftDist += P2.Distance(aNewP2);
                } else {
                  gp_Pnt aNewP1 = ElCLib::Value(NewFpar, PrevEllipse);

                  aPntShiftDist += P1.Distance(aNewP1);
                }

                if (NewLpar < NewFpar)
                  NewLpar += 2.*M_PI;
                if (ConnectByOrigin == TopAbs_FORWARD)
                  ElCLib::AdjustPeriodic(FparSeq.Last(),
                                         FparSeq.Last() + 2.*M_PI,
                                         Precision::PConfusion(), NewFpar, NewLpar);
                else
                  ElCLib::AdjustPeriodic(FparSeq.Last() - 2.*M_PI,
                                         FparSeq.Last(),
                                         Precision::PConfusion(), NewFpar, NewLpar);
                Done = Standard_True;
              }
              break;
            }
          case GeomAbs_Hyperbola:
            {
              gp_Hypr aHypr    = BAcurve.Hyperbola();
              gp_Hypr PrevHypr = GAprevcurve.Hyperbola();
              if (aHypr.Focus1().Distance(PrevHypr.Focus1()) <= LinTol &&
                  aHypr.Focus2().Distance(PrevHypr.Focus2()) <= LinTol &&
                  Abs(aHypr.MajorRadius() - PrevHypr.MajorRadius()) <= LinTol &&
                  Abs(aHypr.MinorRadius() - PrevHypr.MinorRadius()) <= LinTol &&
                  aHypr.Axis().IsParallel(PrevHypr.Axis(), AngTol))
              {
                gp_Pnt P1 = ElCLib::Value(fpar, aHypr);
                gp_Pnt P2 = ElCLib::Value(lpar, aHypr);
                NewFpar = ElCLib::Parameter(PrevHypr, P1);
                NewLpar = ElCLib::Parameter(PrevHypr, P2);

                // Compute shift
                if (ConnectByOrigin == TopAbs_FORWARD) {
                  gp_Pnt aNewP2 = ElCLib::Value(NewLpar, PrevHypr);

                  aPntShiftDist += P2.Distance(aNewP2);
                } else {
                  gp_Pnt aNewP1 = ElCLib::Value(NewFpar, PrevHypr);

                  aPntShiftDist += P1.Distance(aNewP1);
                }

                if (NewLpar < NewFpar)
                {
                  Standard_Real MemNewFpar = NewFpar;
                  NewFpar = NewLpar;
                  NewLpar = MemNewFpar;
                  ConnectByOrigin = TopAbs::Reverse(ConnectByOrigin);
                }
                Done = Standard_True;
              }
              break;
            }
          case GeomAbs_Parabola:
            {
              gp_Parab aParab    = BAcurve.Parabola();
              gp_Parab PrevParab = GAprevcurve.Parabola();
              if (aParab.Location().Distance(PrevParab.Location()) <= LinTol &&
                  aParab.Focus().Distance(PrevParab.Focus()) <= LinTol &&
                  Abs(aParab.Focal() - PrevParab.Focal()) <= LinTol &&
                  aParab.Axis().IsParallel(PrevParab.Axis(), AngTol))
              {
                gp_Pnt P1 = ElCLib::Value(fpar, aParab);
                gp_Pnt P2 = ElCLib::Value(lpar, aParab);
                NewFpar = ElCLib::Parameter(PrevParab, P1);
                NewLpar = ElCLib::Parameter(PrevParab, P2);

                // Compute shift
                if (ConnectByOrigin == TopAbs_FORWARD) {
                  gp_Pnt aNewP2 = ElCLib::Value(NewLpar, PrevParab);

                  aPntShiftDist += P2.Distance(aNewP2);
                } else {
                  gp_Pnt aNewP1 = ElCLib::Value(NewFpar, PrevParab);

                  aPntShiftDist += P1.Distance(aNewP1);
                }

                if (NewLpar < NewFpar)
                {
                  Standard_Real MemNewFpar = NewFpar;
                  NewFpar = NewLpar;
                  NewLpar = MemNewFpar;
                  ConnectByOrigin = TopAbs::Reverse(ConnectByOrigin);
                }
                Done = Standard_True;
              }
              break;
            }
          default:
            break;
          } //end of switch (aType)
        } // end of else if (aType == CurType && ...
        if (Done)
        {
          if (NewFpar < FparSeq.Last())
            FparSeq(FparSeq.Length()) = NewFpar;
          else
            LparSeq(LparSeq.Length()) = NewLpar;
        }
        else
        {
          CurveSeq.Append(aCurve);
          TopoDS_Shape aLocShape;
          aLocShape.Location(aLoc);
          aLocShape.Orientation(wexp.Orientation());
          LocSeq.Append(aLocShape);
          FparSeq.Append(fpar);
          LparSeq.Append(lpar);
          TolSeq.Append(aPntShiftDist + BRep_Tool::Tolerance(CurVertex));
          aPntShiftDist = 0.;
          CurType = aType;
        }
      } // end of else (CurveSeq.IsEmpty()) -> not first time
    } // end for (; wexp.More(); wexp.Next())

    LastVertex = wexp.CurrentVertex();
    TolSeq.Append(aPntShiftDist + BRep_Tool::Tolerance(LastVertex));

    FirstVertex.Orientation(TopAbs_FORWARD);
    LastVertex.Orientation(TopAbs_REVERSED);

    if (!CurveSeq.IsEmpty())
    {
      Standard_Integer nb_curve = CurveSeq.Length();   //number of curves
      TColGeom_Array1OfBSplineCurve tab(0,nb_curve-1);                    //array of the curves
      TColStd_Array1OfReal tabtolvertex(0,nb_curve-1); //(0,nb_curve-2);  //array of the tolerances

      Standard_Integer i;

      if (nb_curve > 1)
      {
        for (i = 1; i <= nb_curve; i++)
        {
          if (CurveSeq(i)->IsInstance(STANDARD_TYPE(Geom_TrimmedCurve)))
            CurveSeq(i) = (*((Handle(Geom_TrimmedCurve)*)&(CurveSeq(i))))->BasisCurve();

          Handle(Geom_TrimmedCurve) aTrCurve = new Geom_TrimmedCurve(CurveSeq(i), FparSeq(i), LparSeq(i));
          tab(i-1) = GeomConvert::CurveToBSplineCurve(aTrCurve);
          tab(i-1)->Transform(LocSeq(i).Location().Transformation());
          GeomConvert::C0BSplineToC1BSplineCurve(tab(i-1), Precision::Confusion());
          if (LocSeq(i).Orientation() == TopAbs_REVERSED)
            tab(i-1)->Reverse();

          //Temporary
          //char* name = new char[100];
          //sprintf(name, "c%d", i);
          //DrawTrSurf::Set(name, tab(i-1));

          if (i > 1)
            tabtolvertex(i-2) = TolSeq(i-1);
        } // end for (i = 1; i <= nb_curve; i++)
        tabtolvertex(nb_curve-1) = TolSeq(TolSeq.Length());

        Standard_Boolean closed_flag = Standard_False;
        Standard_Real closed_tolerance = 0.;
        if (FirstVertex.IsSame(LastVertex) &&
            GeomLProp::Continuity(tab(0), tab(nb_curve-1),
                                  tab(0)->FirstParameter(),
                                  tab(nb_curve-1)->LastParameter(),
                                  Standard_False, Standard_False, LinTol, AngTol) >= GeomAbs_G1)
        {
          closed_flag = Standard_True ;
          closed_tolerance = BRep_Tool::Tolerance(FirstVertex);
        }

        Handle(TColGeom_HArray1OfBSplineCurve)  concatcurve;     //array of the concatenated curves
        Handle(TColStd_HArray1OfInteger)        ArrayOfIndices;  //array of the remining Vertex
        GeomConvert::ConcatC1(tab,
                              tabtolvertex,
                              ArrayOfIndices,
                              concatcurve,
                              closed_flag,
                              closed_tolerance);   //C1 concatenation

        if (concatcurve->Length() > 1)
        {
          GeomConvert_CompCurveToBSplineCurve Concat(concatcurve->Value(concatcurve->Lower()));

          for (i = concatcurve->Lower()+1; i <= concatcurve->Upper(); i++)
            Concat.Add( concatcurve->Value(i), LinTol, Standard_True );

          concatcurve->SetValue(concatcurve->Lower(), Concat.BSplineCurve());
        }
        // rnc : prevents the driver from building an edge without C1 continuity
        if (concatcurve->Value(concatcurve->Lower())->Continuity()==GeomAbs_C0){
          throw Standard_ConstructionError("Construction aborted : The given Wire has sharp bends between some Edges, no valid Edge can be built");
        }

        Standard_Boolean isValidEndVtx = Standard_True;

        if (closed_flag) {
          // Check if closed curve is reordered.
          Handle(Geom_BSplineCurve) aCurve  = concatcurve->Value(concatcurve->Lower());
          Standard_Real      aFPar   = aCurve->FirstParameter();
          gp_Pnt             aPFirst;
          gp_Pnt             aPntVtx = BRep_Tool::Pnt(FirstVertex);
          Standard_Real      aTolVtx = BRep_Tool::Tolerance(FirstVertex);

          aCurve->D0(aFPar, aPFirst);

          if (!aPFirst.IsEqual(aPntVtx, aTolVtx)) {
            // The curve is reordered. Find the new first and last vertices.
            TopTools_IndexedMapOfShape aMapVtx;
            TopExp::MapShapes(theWire, TopAbs_VERTEX, aMapVtx);

            const Standard_Integer aNbVtx = aMapVtx.Extent();
            Standard_Integer       iVtx;

            for (iVtx = 1; iVtx <= aNbVtx; iVtx++) {
              const TopoDS_Vertex aVtx = TopoDS::Vertex(aMapVtx.FindKey(iVtx));
              const gp_Pnt        aPnt = BRep_Tool::Pnt(aVtx);
              const Standard_Real aTol = BRep_Tool::Tolerance(aVtx);

              if (aPFirst.IsEqual(aPnt, aTol)) {
                // The coincident vertex is found.
                FirstVertex = aVtx;
                LastVertex  = aVtx;
                FirstVertex.Orientation(TopAbs_FORWARD);
                LastVertex.Orientation(TopAbs_REVERSED);
                break;
              }
            }

            if (iVtx > aNbVtx) {
              // It is necessary to create new vertices.
              isValidEndVtx = Standard_False;
            }
          }
        }

        if (isValidEndVtx) {
          ResEdge = BRepLib_MakeEdge(concatcurve->Value(concatcurve->Lower()),
                                     FirstVertex, LastVertex,
                                     concatcurve->Value(concatcurve->Lower())->FirstParameter(),
                                     concatcurve->Value(concatcurve->Lower())->LastParameter());
        } else {
          ResEdge = BRepLib_MakeEdge(concatcurve->Value(concatcurve->Lower()),
                                     concatcurve->Value(concatcurve->Lower())->FirstParameter(),
                                     concatcurve->Value(concatcurve->Lower())->LastParameter());
        }
      }
      else
      {
        if (CurveSeq(1)->IsInstance(STANDARD_TYPE(Geom_TrimmedCurve)))
          CurveSeq(1) = (*((Handle(Geom_TrimmedCurve)*)&(CurveSeq(1))))->BasisCurve();

        Handle(Geom_Curve) aNewCurve =
          Handle(Geom_Curve)::DownCast(CurveSeq(1)->Copy());

        aNewCurve->Transform(LocSeq(1).Location().Transformation());

        if (LocSeq(1).Orientation() == TopAbs_REVERSED) {
          const TopoDS_Vertex aVtxTmp = FirstVertex;

          FirstVertex = LastVertex;
          LastVertex  = aVtxTmp;
          FirstVertex.Orientation(TopAbs_FORWARD);
          LastVertex.Orientation(TopAbs_REVERSED);
        }

        ResEdge = BRepLib_MakeEdge(aNewCurve,
                                   FirstVertex, LastVertex,
                                   FparSeq(1), LparSeq(1));

        if (LocSeq(1).Orientation() == TopAbs_REVERSED) {
          ResEdge.Reverse();
        }
      }
    }

    return ResEdge;
}

Standard_Boolean AreEdgesC1 (const TopoDS_Edge& E1, const TopoDS_Edge& E2)
{
  BRepAdaptor_Curve aCurve1 (E1);
  BRepAdaptor_Curve aCurve2 (E2);

  if (aCurve1.Continuity() == GeomAbs_C0 || aCurve2.Continuity() == GeomAbs_C0)
    return Standard_False;

  Standard_Real tol, tolMax = Precision::Confusion();
  for (TopExp_Explorer ExV1 (E1, TopAbs_VERTEX); ExV1.More(); ExV1.Next()) {
    TopoDS_Vertex Vertex = TopoDS::Vertex(ExV1.Current());
    tol = BRep_Tool::Tolerance(Vertex);
      if (tol > tolMax)
        tolMax = tol;
  }
  for (TopExp_Explorer ExV2 (E2, TopAbs_VERTEX); ExV2.More(); ExV2.Next()) {
    TopoDS_Vertex Vertex = TopoDS::Vertex(ExV2.Current());
    tol = BRep_Tool::Tolerance(Vertex);
      if (tol > tolMax)
        tolMax = tol;
  }

  Standard_Real f1, l1, f2, l2;
  f1 = aCurve1.FirstParameter();
  l1 = aCurve1.LastParameter();
  f2 = aCurve2.FirstParameter();
  l2 = aCurve2.LastParameter();

  if (f1 > l1) {
    Standard_Real tmp = f1;
    f1 = l1;
    l1 = tmp;
  }

  if (f2 > l2) {
    Standard_Real tmp = f2;
    f2 = l2;
    l2 = tmp;
  }

  gp_Pnt pf1, pl1, pf2, pl2;
  gp_Vec vf1, vl1, vf2, vl2;
  aCurve1.D1(f1, pf1, vf1);
  aCurve1.D1(l1, pl1, vl1);
  aCurve2.D1(f2, pf2, vf2);
  aCurve2.D1(l2, pl2, vl2);

  // pf1--->---pl1.pf2--->---pl2
  if (pl1.SquareDistance(pf2) < tolMax*tolMax) {
    if (vl1.Angle(vf2) < Precision::Angular())
      return Standard_True;
  }
  // pl1---<---pf1.pf2--->---pl2
  else if (pf1.SquareDistance(pf2) < tolMax*tolMax) {
    if (vf1.Angle(-vf2) < Precision::Angular())
      return Standard_True;
  }
  // pf1--->---pl1.pl2---<---pf2
  else if (pl1.SquareDistance(pl2) < tolMax*tolMax) {
    if (vl1.Angle(-vl2) < Precision::Angular())
      return Standard_True;
  }
  // pl1---<---pf1.pl2---<---pf2
  else {
    if (vf1.Angle(vl2) < Precision::Angular())
      return Standard_True;
  }

  return Standard_False;
}

}  // namespace geomcopy
}  // namespace pysmesh
