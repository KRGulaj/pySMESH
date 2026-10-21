// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH binding — GEOM's edge-from-wire and C1 test, copied.
//
// Two functions of GEOM's driver layer that the 1-D fillet's fallback path needs, copied
// from SALOME GEOM tag V9_16_0 (see geom_edge_from_wire.cpp for the origin and licence):
//
//   MakeEdgeFromWire  one edge over a wire of edges that continue each other with C1
//                     continuity: one curve when they all lie on one line, circle, ellipse,
//                     hyperbola or parabola, otherwise one B-spline by C1 concatenation
//                     (GeomConvert::ConcatC1);
//   AreEdgesC1        whether two edges meet with their tangents parallel within
//                     Precision::Angular at a common end (within the larger vertex
//                     tolerance), neither curve being C0.

#pragma once

#include <BRepTools_History.hxx>
#include <Standard_Handle.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Shape.hxx>

namespace pysmesh {
namespace geomcopy {

// theFixHistory, when not null, receives the history of the ShapeFix_Shape the function
// runs on the wire before it reads the curves: the end vertices of the new edge are the
// fixed wire's.
TopoDS_Edge MakeEdgeFromWire(const TopoDS_Shape& aWire, const Standard_Real LinTol,
                             const Standard_Real AngTol,
                             Handle(BRepTools_History)* theFixHistory);

Standard_Boolean AreEdgesC1(const TopoDS_Edge& E1, const TopoDS_Edge& E2);

}  // namespace geomcopy
}  // namespace pysmesh
