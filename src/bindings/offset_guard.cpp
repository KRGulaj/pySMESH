// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-09-22

// pySMESH binding — what an offset must be, shared by both entry points.
//
// See offset_guard.hpp for the split between the pre-condition and the two post-conditions.
// Every number quoted in this file was measured against the kernel, not derived and assumed.

#include "offset_guard.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <BRepAdaptor_Surface.hxx>
#include <BRepGProp.hxx>
#include <BRep_Tool.hxx>
#include <GProp_GProps.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <gp_Cone.hxx>
#include <gp_Torus.hxx>

namespace pysmesh {
namespace offset_guard {

// The radius an analytic face's own surface has now, and the one the offset would give it.
//
// Two things about the arithmetic, both measured rather than assumed:
//
//   * BRepOffset_Skin moves every face along its **outward** normal, and a REVERSED face's
//     outward normal is its surface's own negated. So a bore shrinks where a boss grows.
//     Measured on a tube of radii 3 and 1: at distance -0.5 the outer wall comes back at
//     2.5 and the bore at 1.5; at +0.5, 3.5 and 0.5. A rule that read the sign off the
//     distance alone would refuse every legitimate hollow part, and would miss the bore of
//     radius 1 collapsing at +1.5 — which OCCT commits, as a spurious bore of radius 0.5.
//   * A cone's radius moves by `d * cos(SemiAngle)` over the face's own parameter range,
//     not by `d`. Measured on a cone of radii 2 and 0.7: at -0.5 the surface comes back with
//     RefRadius 1.5160887466058641, and 2 - 0.5 * cos(-0.25436805855326594) is
//     1.5160887466058641 exactly. The offset cone keeps its SemiAngle and slides its origin
//     along the axis, which is the same cone described from a different station.
//
// A cone's radius varies along the face, so the smallest one over the face's own parameter
// range is what has to survive: r(v) = RefRadius + v * sin(SemiAngle), linear in v, so the
// minimum is at one of the two ends.
FaceRadius offset_radius(const TopoDS_Face& face, double distance) {
  FaceRadius out;
  const BRepAdaptor_Surface surf(face);
  const double d = (face.Orientation() == TopAbs_REVERSED) ? -distance : distance;
  switch (surf.GetType()) {
    case GeomAbs_Cylinder:
      out.analytic = true;
      out.surface = "cylinder";
      out.before = surf.Cylinder().Radius();
      out.after = out.before + d;
      break;
    case GeomAbs_Sphere:
      out.analytic = true;
      out.surface = "sphere";
      out.before = surf.Sphere().Radius();
      out.after = out.before + d;
      break;
    case GeomAbs_Torus:
      // The minor radius: the tube's own. The major radius is the ring's and an offset
      // leaves it where it is, measured on a torus of radii 5 and 1.5 at -0.5, -1.0 and
      // -1.4 — major 5.0 every time.
      out.analytic = true;
      out.surface = "torus";
      out.before = surf.Torus().MinorRadius();
      out.after = out.before + d;
      break;
    case GeomAbs_Cone: {
      const gp_Cone c = surf.Cone();
      const double s = std::sin(c.SemiAngle());
      out.analytic = true;
      out.surface = "cone";
      out.before = std::min(c.RefRadius() + surf.FirstVParameter() * s,
                            c.RefRadius() + surf.LastVParameter() * s);
      out.after = out.before + d * std::cos(c.SemiAngle());
      break;
    }
    default:
      break;
  }
  return out;
}

std::vector<std::pair<TopoDS_Shape, FaceRadius>> radii_that_vanish(
    const std::vector<TopoDS_Shape>& faces, double distance, double tol) {
  std::vector<std::pair<TopoDS_Shape, FaceRadius>> out;
  for (const TopoDS_Shape& s : faces) {
    if (s.ShapeType() != TopAbs_FACE) {
      continue;
    }
    const FaceRadius r = offset_radius(TopoDS::Face(s), distance);
    if (r.analytic && r.after <= tol) {
      out.emplace_back(s, r);
    }
  }
  std::stable_sort(out.begin(), out.end(),
                   [](const std::pair<TopoDS_Shape, FaceRadius>& a,
                      const std::pair<TopoDS_Shape, FaceRadius>& b) {
                     return a.second.after < b.second.after;
                   });
  return out;
}

std::string radius_phrase(const FaceRadius& r, const std::string& named) {
  return std::string(r.surface) + " of radius " + std::to_string(r.before) + named +
         " would be left with " + std::to_string(r.after);
}

const char* radius_detail() {
  return "An offset takes an analytic face's radius to that face's own side of zero and no "
         "further: at zero the surface degenerates, and past it OCCT rebuilds it at the "
         "absolute value of the negative radius — the same surface mirrored through its own "
         "axis, which is a valid solid that is not the offset of anything. Reduce the "
         "magnitude, or offset a face set whose radii survive. Nothing was built.";
}

SolidVolumes solid_volumes(const TopoDS_Shape& s) {
  SolidVolumes out;
  for (TopExp_Explorer ex(s, TopAbs_SOLID); ex.More(); ex.Next()) {
    GProp_GProps props;
    BRepGProp::VolumeProperties(ex.Current(), props);
    const double v = props.Mass();
    out.least = (out.count == 0) ? v : std::min(out.least, v);
    out.total += v;
    ++out.count;
  }
  return out;
}

// What was observed, when the result of a hollowing is not a hollowed solid. Empty when it
// is one.
//
// Measured on the released 4.1.1 wheel, a 3 x 7 x 11 box opened at the face of largest z:
// once |thickness| passes 1.55 — the box's smallest extent is 3 — MakeThickSolidByJoin
// collapses the inner shell, reports IsDone(), and hands back a shape BRepCheck_Analyzer
// accepts and that *is* the input solid: volume 231, six faces, and the face that was to be
// opened re-issued under a new id. An `unopened` post-condition cannot see it: the opened
// face is genuinely gone, and the face standing in its place is a different shape. Opening
// every face of a solid does the same thing at every thickness measured, -0.05 through -3.0.
//
// The three statements below are what a thick solid *is*. None is a rule about how large a
// thickness may be: a magnitude rule would have to know the body's smallest feature, and
// would have to refuse the thin wall at -1.40 — cavity 8.064 of 231 — that OCCT builds
// correctly.
//
//   1. It is a solid, and a solid's volume is positive. A negative one is the same wall
//      turned inside out. Measured on the box with every face but one opened and a positive
//      thickness: the plate comes back as -(the unopened face's area x thickness), -38.5 at
//      +0.5, and 4.1.1 committed it, so the session reported a body of negative volume.
//   2. Hollowed inward, the wall lies inside the boundary it was built from, so its volume
//      is strictly less than that body's. Outward the wall lies outside instead, and its
//      volume stands in no fixed relation to the input's — a box opened at one face and
//      thickened by 2.0 gives 770 against an input of 231, by 0.5 gives 137 — so the
//      statement is made for a negative thickness only, which is the sign it is derived for.
//   3. The result carries the walls the offset built: at least one face that is neither a
//      face of the input nor a rim generated from an opened face. Vacuous when every face
//      was opened, because then there is no wall to build and statement 2 carries the case.
//
// All three are made, because each catches a measured case the others miss. Opening every
// face of the box at -0.5 leaves no wall to look for, and only statement 2 sees it. Opening
// the two walls normal to y and the top at -3.0 returns the input solid measuring
// 230.99999999999997 against the input's 231 — under it by 2.8e-14, which is round-off, not
// a cavity — and only statement 3 sees that. The inside-out plate has a cavity and has its
// walls, and only statement 1 sees it.
std::string not_a_thick_solid(const TopoDS_Shape& owner, const TopoDS_Shape& result,
                              double thickness, const ShapeSet& opened, const ShapeSet& rims) {
  std::vector<std::string> broken;

  const SolidVolumes got = solid_volumes(result);
  const double volume = got.total;
  if (got.count == 0) {
    broken.push_back("It holds no solid at all.");
  } else if (got.least <= 0.0) {
    broken.push_back("It holds a solid of volume " + std::to_string(got.least) +
                     ", so the wall came back turned inside out.");
  }

  const double input_volume = solid_volumes(owner).total;
  if (thickness < 0.0 && volume >= input_volume) {
    broken.push_back("Its volume " + std::to_string(volume) +
                     " is not less than the input solid's " + std::to_string(input_volume) +
                     ", so no cavity was cut.");
  }

  ShapeSet input_faces;
  TopExp::MapShapes(owner, TopAbs_FACE, input_faces);
  int unopened_faces = 0;
  for (int i = 1; i <= input_faces.Extent(); ++i) {
    if (!opened.Contains(input_faces(i))) {
      ++unopened_faces;
    }
  }
  int walls = 0;
  int result_faces = 0;
  for (TopExp_Explorer ex(result, TopAbs_FACE); ex.More(); ex.Next()) {
    ++result_faces;
    if (!input_faces.Contains(ex.Current()) && !rims.Contains(ex.Current())) {
      ++walls;
    }
  }
  if (unopened_faces > 0 && walls == 0) {
    broken.push_back("No face of it is a wall the offset built. Each one is a face of the "
                     "input or a rim of an opening, and " +
                     std::to_string(unopened_faces) +
                     " faces were left unopened for walls to stand on.");
  }

  if (broken.empty()) {
    return std::string();
  }
  std::string why;
  for (const std::string& line : broken) {
    why += line + " ";
  }
  return why + "The result has " + std::to_string(result_faces) + " faces and volume " +
         std::to_string(volume) + "; the input solid had " +
         std::to_string(input_faces.Extent()) + " faces and volume " +
         std::to_string(input_volume) + ".";
}

// What was observed, when the result of a uniform offset is not that body offset. Empty
// when it is.
//
// The hollowing's collapse does not happen here: over 99 committed offsets of a box, a
// cylinder, a cone, a sphere and a torus, from -1000.0 to +50.0, not one came back as the
// input body. Two other things did, both committed by 4.1.1:
//
//   * a sphere of radius 3 shrunk by 3.0 comes back with volume 0.0, and by 5.0 with
//     -33.51; a torus of radii 5 and 1.5 comes back negative from -1.5 down, -24.67 at
//     -2.0. The body is inside out, and the session reported a negative volume for it.
//   * the same torus at -11.0 and beyond comes back *larger* than it went in: 8907 against
//     222, for a distance that shrinks.
//
// Unlike a hollowing, an offset keeps the body it was given, so the sign carries a statement
// in both directions: a solid offset inward is the set of its own points at least |distance|
// from its boundary, which is a proper subset, and offset outward it is a proper superset.
// Hence two statements, and the first is the hollowing's own:
//
//   1. It is a solid, and a solid's volume is positive.
//   2. A negative distance shrinks the body and a positive one grows it, strictly.
//
// Both cases quoted above are now refused by the radius pre-condition before OCCT is
// driven, so these two statements are the cover for the faces the pre-condition cannot
// speak for — a B-spline wall, a surface of revolution — rather than the first line of
// defence they were in 4.1.2.
//
// A shell body is not checked. It carries no solid to measure, and a shell's area is not
// monotone in the offset distance once the shell is not convex, so there is nothing to state
// here that would be a theorem rather than a guess. Measured on the five-face open box
// shell: every distance from -0.5 to +5.0 offsets correctly, and -1.0 and beyond is already
// refused by the analyzer.
std::string not_an_offset_body(const TopoDS_Shape& owner, const TopoDS_Shape& result,
                               double distance) {
  if (owner.ShapeType() != TopAbs_SOLID) {
    return std::string();
  }
  std::vector<std::string> broken;

  const SolidVolumes got = solid_volumes(result);
  if (got.count == 0) {
    broken.push_back("It holds no solid, though the body it was built from is one.");
  } else if (got.least <= 0.0) {
    broken.push_back("It holds a solid of volume " + std::to_string(got.least) +
                     ", so the body came back turned inside out.");
  }

  const double input_volume = solid_volumes(owner).total;
  if (distance < 0.0 && got.count > 0 && got.total >= input_volume) {
    broken.push_back("Its volume " + std::to_string(got.total) +
                     " is not less than the input solid's " + std::to_string(input_volume) +
                     ", though the distance shrinks it.");
  }
  if (distance > 0.0 && got.count > 0 && got.total <= input_volume) {
    broken.push_back("Its volume " + std::to_string(got.total) +
                     " is not more than the input solid's " + std::to_string(input_volume) +
                     ", though the distance grows it.");
  }

  if (broken.empty()) {
    return std::string();
  }
  std::string why;
  for (const std::string& line : broken) {
    why += line + " ";
  }
  return why + "The result holds " + std::to_string(got.count) + " solid(s) of volume " +
         std::to_string(got.total) + "; the input solid's volume was " +
         std::to_string(input_volume) + ".";
}

}  // namespace offset_guard
}  // namespace pysmesh
