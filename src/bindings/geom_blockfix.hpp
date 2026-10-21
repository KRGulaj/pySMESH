// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-22

// pySMESH binding — GEOM's BlockFix steps, copied with their history.
//
// SALOME common_geometry_lib tag V9_16_0, src/BlockFix/BlockFix.cxx (RotateSphereSpace,
// RefillProblemFaces, FixRanges and their static helpers) and
// src/BlockFix/BlockFix_BlockFixAPI.cxx (BlockFix_BlockFixAPI::Perform), LGPL-2.1 or later,
// used here under LGPL-2.1 and recorded in PROVENANCE.md. The vendored functions keep each
// step's context, modifier and substitution local, so their history is out of reach; these
// copies write it. geom_blockfix.cpp is written by a script from the vendored sources and
// lists each change.

#pragma once

#include <BRepTools_History.hxx>
#include <BRepTools_Substitution.hxx>
#include <Message_ProgressRange.hxx>
#include <Message_ProgressScope.hxx>
#include <Standard_Real.hxx>
#include <TopoDS_Shape.hxx>

#include <stdexcept>
#include <vector>

namespace pysmesh {
namespace session {

// The history of a rebuild that keeps the structure of `before`: `after` holds, at each
// place of `before`, one shape of the same type, with the same children in the same order
// (ShapeUpgrade_RemoveLocations, TNaming_CopyShape). Each face, edge, vertex and solid of
// `before`, as TopoDS_Iterator reaches it with its placement, is modified into the shape at
// the same place of `after`. Fails loudly if the two structures differ. Defined in
// session_blockfix.cpp.
Handle(BRepTools_History) structural_history(const TopoDS_Shape& before,
                                             const TopoDS_Shape& after);

// The history of a BRepTools_Substitution built on `shape`: each sub-shape it copied goes
// to its copies, each followed to its own copies where it was rebuilt too; one copied to
// nothing is removed. Defined in session_blockfix.cpp.
Handle(BRepTools_History) substitution_history(const TopoDS_Shape& shape,
                                               const BRepTools_Substitution& subst);

// What require_turned_faces_kept throws: the text says which face changed and how.
struct TurnedFaceError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// pySMESH's check of GEOM's RotateSphereSpace and FixRanges (`step`), which put a
// spherical or cylindrical face on a turned copy of its surface: such a face keeps its
// area. Each sphere or cylinder face of `before` that `steps` (the step's histories)
// carry onto one face of `after` must keep its area within 1e-6 relative; otherwise
// TurnedFaceError. Defined in session_blockfix.cpp.
void require_turned_faces_kept(const char* step, const TopoDS_Shape& before,
                               const TopoDS_Shape& after,
                               const std::vector<Handle(BRepTools_History)>& steps);

}  // namespace session

namespace geomcopy {

// BlockFix::RotateSphereSpace, copied: each step's history appended to theSteps.
TopoDS_Shape RotateSphereSpace(const TopoDS_Shape& S, const Standard_Real Tol,
                               const Standard_Boolean theTrySmallRotation,
                               const Message_ProgressRange& theRange,
                               std::vector<Handle(BRepTools_History)>& theSteps);

// BlockFix::RefillProblemFaces, copied: the substitution's history in theHistory.
TopoDS_Shape RefillProblemFaces(const TopoDS_Shape& aShape,
                                Handle(BRepTools_History)& theHistory);

// BlockFix::FixRanges, copied: each step's history appended to theSteps.
TopoDS_Shape FixRanges(const TopoDS_Shape& S, const Standard_Real Tol,
                       const Message_ProgressRange& theRange,
                       std::vector<Handle(BRepTools_History)>& theSteps);

// BlockFix_BlockFixAPI::Perform, copied as a function of its members: the improved shape,
// with each step's history appended to theSteps, in order. thePS needs 6 steps; after a
// cancel the function returns a null shape. myOptimumNbFaces is 0 (unite faces) or -1.
TopoDS_Shape BlockFixAPI_Perform(const TopoDS_Shape& theShape, const Standard_Real myTolerance,
                                 const Standard_Integer myOptimumNbFaces,
                                 Message_ProgressScope& thePS,
                                 std::vector<Handle(BRepTools_History)>& theSteps);

}  // namespace geomcopy
}  // namespace pysmesh
