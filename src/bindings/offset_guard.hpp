// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-09-22

// pySMESH binding — what an offset must be, shared by both entry points.
//
// TKOffset's BRepOffsetAPI_MakeThickSolid and BRepOffsetAPI_MakeOffsetShape are reached from
// two places: the stateless offset.cpp, which takes BREP bytes and answers with BREP bytes,
// and Session, which carries entity ids across the operation. The kernel behaves the same
// way from both, so the statements that decide whether its answer is the offset at all
// belong to neither file. They are here, and each entry point supplies only its own
// vocabulary for naming the faces it blames.
//
// There are two kinds of statement, and the difference is not a matter of taste:
//
//   * A PRE-CONDITION on the faces going in. Past a face's own radius OCCT rebuilds that
//     face at the absolute value of the negative radius — the same surface mirrored through
//     its own axis. The result is a valid solid of positive volume with the right topology
//     and the right number of faces, so no statement about the result can separate it from
//     an honest one. Measured on a cone of radii 2 and 0.7: at -1.3 the offset commits
//     0.244998596 where the honest answer is 0.130473507, and the two results have the same
//     three faces, the same two vertices and the same one solid.
//   * A POST-CONDITION on the body that comes back. These catch the failures that leave a
//     mark on the result — a wall that collapsed onto the input, a body turned inside out,
//     a body that grew where the distance said shrink — and they are the only cover a face
//     with no closed-form radius has.
//
// Both are needed. Neither subsumes the other.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <NCollection_IndexedDataMap.hxx>
#include <NCollection_IndexedMap.hxx>
#include <NCollection_List.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>

namespace pysmesh {
namespace offset_guard {

// A set of shapes that answers "is this one of them", and a map from an edge to the faces
// that own it. Both spell out the types the two entry points already hold, so neither has to
// convert anything to call in here.
using ShapeSet = NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher>;
using EdgeOwners = NCollection_IndexedDataMap<TopoDS_Shape, NCollection_List<TopoDS_Shape>,
                                              TopTools_ShapeMapHasher>;

// Why an analytic face does not survive the offset it was given.
//
//   Vanishes  its radius reaches zero, and past zero OCCT mirrors the surface.
//   Spindle   a torus's tube reaches the ring's own axis of revolution, and the surface
//             passes through itself. OCCT commits it, and reports the ring's volume for a
//             body that no longer is one.
enum class Fail : int { None = 0, Vanishes = 1, Spindle = 2 };

// What an offset would do to one analytic face's own radius.
//
// `analytic` is false for every surface with no closed-form radius — a plane, a B-spline, a
// surface of revolution — and those faces are not the subject of this check at all.
struct FaceRadius {
  bool analytic = false;
  const char* surface = "";
  double before = 0.0;  // the radius the face carries now
  double after = 0.0;   // the radius the offset would leave it with
  double limit = 0.0;   // a torus's major radius; zero for everything else
  Fail fail = Fail::None;

  // How much room the face has left. Negative means it has none, and the more negative it
  // is the worse the case, so one ordering covers both kinds of failure.
  double margin() const { return fail == Fail::Spindle ? limit - after : after; }
};

// What the offset leaves of one face's radius.
//
// `moving` names the faces this operation offsets — every face of the body for a uniform
// offset, every face the caller did not open for a hollowing. It is read for the *other*
// faces, the ones bounding the cone being measured: a cap that is offset too slides along
// the axis and takes the end of the cone with it, and one that was opened stays where it is.
// `edge_owners` maps each edge of the body to the faces that own it, and is what finds those
// neighbours. Both are built once per operation by the caller.
FaceRadius offset_radius(const TopoDS_Face& face, double distance, const EdgeOwners& edge_owners,
                         const ShapeSet& moving, double tol);

// Every face of `faces` whose radius does not survive `distance`, worst first.
std::vector<std::pair<TopoDS_Shape, FaceRadius>> radii_that_vanish(
    const std::vector<TopoDS_Shape>& faces, double distance, const EdgeOwners& edge_owners,
    const ShapeSet& moving, double tol);

// One failing face in words. `named` is what the caller knows the face by, already
// parenthesised — an EntityId in the session, a 1-based ordinal in the stateless module —
// and is dropped into the sentence beside the radius it belongs to. Empty names nothing.
std::string radius_phrase(const FaceRadius& r, const std::string& named);

// The explanation both entry points carry on PysmeshError.details for a radius refusal.
const char* radius_detail();

// The volumes of a shape's solids: how many there are, what they add up to, and the smallest
// of them. Every statement both post-conditions make is made in these three numbers and the
// input solid's own volume.
struct SolidVolumes {
  int count = 0;
  double total = 0.0;
  double least = 0.0;
};

SolidVolumes solid_volumes(const TopoDS_Shape& s);

// What was observed, when the result of a hollowing is not a hollowed solid, or of a uniform
// offset is not that body offset. Empty when it is.
//
// `opened` names the faces the caller asked to turn into openings, and `rims` the faces the
// algorithm's own history relates to them — the opening's edge left behind once the cavity
// is cut. A rim is neither a face of the input nor a wall the offset built, and telling it
// from a wall is what makes the third statement below able to fire at all.
std::string not_a_thick_solid(const TopoDS_Shape& owner, const TopoDS_Shape& result,
                              double thickness, const ShapeSet& opened, const ShapeSet& rims);

std::string not_an_offset_body(const TopoDS_Shape& owner, const TopoDS_Shape& result,
                               double distance);

}  // namespace offset_guard
}  // namespace pysmesh
