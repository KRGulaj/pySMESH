// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH binding — GEOM's 1-D fillet algorithm, copied.
//
// GEOMImpl_Fillet1d and GEOMImpl_Fillet1dPoint are copied from SALOME GEOM tag V9_16_0
// (7d0a18e1554c25bca2c2d9882e0621ea3f7f9c81), src/GEOMImpl/GEOMImpl_Fillet1d.hxx and
// GEOMImpl_Fillet1d.cxx:
//
//   Copyright (C) 2007-2026  CEA, EDF, OPEN CASCADE
//   This library is free software; you can redistribute it and/or modify it under the
//   terms of the GNU Lesser General Public License as published by the Free Software
//   Foundation; either version 2.1 of the License, or (at your option) any later version.
//
// Used here under LGPL-2.1 and recorded in PROVENANCE.md. The class has no GEOM dependency.
// The copy is unchanged but for three things: it lives in namespace pysmesh::geomcopy; the
// Standard_EXPORT markers are dropped, because the class is compiled into _core and exported
// from nothing; and two OCCT 8.0.1 port changes in the .cpp (see there).
//
// The algorithm rounds the corner between two planar edges with an arc of a given radius:
// it projects both curves onto the plane, searches the first curve for the points whose
// circle centre lies at the radius from the second curve (a Newton search on the squared
// distance minus the squared radius, per sampled interval), and keeps the solution nearest
// to the corner. Result() then builds the arc and the two edges trimmed to its ends.

#pragma once

#include <Geom2d_Curve.hxx>
#include <Geom_Plane.hxx>
#include <TColStd_ListOfReal.hxx>
#include <TColStd_SequenceOfInteger.hxx>
#include <TColStd_SequenceOfReal.hxx>
#include <TopoDS_Edge.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>

namespace pysmesh {
namespace geomcopy {

class GEOMImpl_Fillet1dPoint;

/**
* GEOMImpl_Fillet1d is 1D fillet algorithm on two planar edges with given radius
*/

class GEOMImpl_Fillet1d
{
public:
  //! Constructor
  //! The fillet 1D algorithm is initialised by two edges and plane
  GEOMImpl_Fillet1d(const TopoDS_Edge& theEdge1,
                    const TopoDS_Edge& theEdge2,
                    const gp_Pln&      thePlane);
  //! Makes fillet with given radius
  //! @returns Standard_True, if at least one result computed
  Standard_Boolean Perform(const Standard_Real theRadius);

  //! Returns result fillet edge and modified edges as out parameters
  TopoDS_Edge Result(const gp_Pnt& thePoint, TopoDS_Edge& theEdge1, TopoDS_Edge& theEdge2);

private:
  //! private methods
  void performInterval(const Standard_Real theStart,
                       const Standard_Real theEnd,
                       const Standard_Integer theNBSteps);
  void fillPoint(GEOMImpl_Fillet1dPoint*);
  void fillDiff(GEOMImpl_Fillet1dPoint*, Standard_Real, Standard_Boolean);
  void performNewton(GEOMImpl_Fillet1dPoint*, GEOMImpl_Fillet1dPoint*);
  Standard_Boolean processPoint(GEOMImpl_Fillet1dPoint*, GEOMImpl_Fillet1dPoint*, Standard_Real);

private:
  //! private fields
  TopoDS_Edge myEdge1, myEdge2;
  Handle(Geom_Plane) myPlane;
  Handle(Geom2d_Curve) myCurve1, myCurve2;
  Standard_Real myStart1, myEnd1, myStart2, myEnd2, myRadius;
  TColStd_ListOfReal myResultParams;
  TColStd_SequenceOfInteger myResultOrientation;
  Standard_Boolean myStartSide, myEdgesExchnged;
  Standard_Integer myDegreeOfRecursion;
};


/**
* GEOMImpl_Fillet1dPoint is an internal class for 1D fillet algorithm
*   to store and compare computed solutions on edges
*/

class GEOMImpl_Fillet1dPoint
{
public:
  //! Puiblic methods

  //! Constructor
  GEOMImpl_Fillet1dPoint(Standard_Real theParam)
    {myParam = theParam;}

  //! Make copy of point
  //!WARNING: Copies only field values: myParam, myV, myD, myValid
  GEOMImpl_Fillet1dPoint* Copy(); // warning: this is not the full copy!

  //! Set/Get parameter
  inline void SetParam(Standard_Real theParam)
    {myParam = theParam;}
  inline Standard_Real GetParam() const
    {return myParam;}
  inline void SetParam2(const Standard_Real theParam2)
    {myParam2 = theParam2;}
  inline Standard_Real GetParam2()
    { return myParam2 ; }

  //! Returns validity
  inline Standard_Boolean IsValid(int theIndex)
    {return (Standard_Boolean)myValid.Value(theIndex);}

  //! Get values
  inline Standard_Integer GetNBValues() {return myV.Length();}
  inline Standard_Real GetValue(Standard_Integer theIndex)
    {return myV.Value(theIndex);}
  inline Standard_Real GetDiff(Standard_Integer theIndex)
    {return myD.Value(theIndex);}
  inline Standard_Integer GetNear(Standard_Integer theIndex)
    {return myNear.Value(theIndex);}

  //! Set/Get center point
  inline void SetCenter(const gp_Pnt2d thePoint)
    {myCenter = thePoint;}
  inline const gp_Pnt2d GetCenter()
    {return myCenter;}

  void AddValue(Standard_Real theValue, Standard_Boolean theIsValid);

  //! compute difference between this and given point
  Standard_Boolean ComputeDifference(GEOMImpl_Fillet1dPoint*);
  void FilterPoints(GEOMImpl_Fillet1dPoint*);

  //! Checks if point contains solution and returns the index of it if any
  Standard_Integer HasSolution(Standard_Real theRadius);
  //! Remove solution by index
  void RemoveSolution(Standard_Integer theIndex);

private:
  //! Private fields
  gp_Pnt2d myCenter;
  Standard_Real myParam, myParam2;
  TColStd_SequenceOfReal myV, myD;
  TColStd_SequenceOfInteger myValid, myNear;
};

}  // namespace geomcopy
}  // namespace pysmesh
