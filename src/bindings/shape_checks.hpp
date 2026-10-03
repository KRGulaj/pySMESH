// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-03

// pySMESH binding — what a committed shape must be, beyond BRepCheck_Analyzer.
//
// BRepCheck_Analyzer tests the local consistency of the topology: each edge has its curves,
// each wire closes, each face's wires are oriented. It does not test the properties a mesher
// needs, and OCCT can hand back a shape that passes it and is still not the answer: an empty
// compound for two solids that overlap, a solid with an edge on one face only, a solid turned
// inside out (report defect_sweep_4.2.2.md §6). The statements here test those properties.
// Each one is sound: it refuses only on a witness that the answer is wrong — a point proven
// inside both operands of an empty common, an edge with one face — never on a heuristic.
//
// The session and the stateless module share them, so they are here and not in either one.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <TopoDS_Shape.hxx>
#include <gp_Pnt.hxx>

namespace pysmesh {
namespace shape_checks {

// The largest tolerance of any vertex, edge or face of the shapes. Precision::Confusion()
// when there is none.
double max_tolerance(const std::vector<TopoDS_Shape>& shapes);

// The solids of the shapes, in TopExp::MapShapes order.
std::vector<TopoDS_Shape> solids_of(const std::vector<TopoDS_Shape>& shapes);

// A point that lies inside a solid of `a` and inside a solid of `b`, each deeper than `tol`
// from the solid's boundary. Such a point proves that the interiors of the two groups
// overlap by more than `tol`, so their common cannot be empty.
//
// The candidates sit on the boundary of each solid of both groups — a parameter grid on
// every face, the midpoint of every edge on it and every vertex — moved into the solid along
// the face normal by `depth`, which must exceed `tol`. A candidate counts only where
// BRepClass3d_SolidClassifier places it IN its own solid at `tol`, and then IN a solid of
// the other group at `tol`. The classifier is OCCT's point classifier, independent of the
// Boolean Component whose answer is under test. Finding no witness proves nothing.
std::optional<gp_Pnt> point_inside_both(const std::vector<TopoDS_Shape>& a,
                                        const std::vector<TopoDS_Shape>& b, double depth,
                                        double tol);

// A point that lies inside a solid of `a` and outside every solid of `b`, each farther than
// `tol` from the boundary. It proves that `a` minus `b` is not empty. The candidates are
// those of point_inside_both, taken from the solids of `a` only.
std::optional<gp_Pnt> point_inside_first_outside_second(const std::vector<TopoDS_Shape>& a,
                                                        const std::vector<TopoDS_Shape>& b,
                                                        double depth, double tol);

// "(x, y, z)" at 9 significant digits.
std::string point_text(const gp_Pnt& p);

// The edges of `shape` bordered by exactly one face, in TopExp::MapShapes order: the naked
// boundary of an open shell, and the leak in a solid that is not watertight. The face count
// is taken with repetition (TopExp::MapShapesAndAncestors), so a periodic seam, listed twice
// by its one face, counts two and is not free. A degenerated edge (a pole, an apex) bounds
// nothing and is skipped, and so is an edge with no face, which is not a surface boundary.
std::vector<TopoDS_Shape> free_boundary_edges(const TopoDS_Shape& shape);

// One edge in words: "from (x, y, z) to (x, y, z), length L".
std::string edge_text(const TopoDS_Shape& edge);

// The signed volume and the area of a tessellation of `solid` at the absolute deflection
// `deflection`.
//
// The volume is the sum of p0 . (p1 x p2) / 6 over the triangles, each oriented by its face,
// so a solid whose faces point out gets a positive volume. Every triangle lies within the
// deflection of its face, so the true volume differs from this one by at most
// `deflection` x the area. `complete` is false when a face got no triangles; the numbers
// then mean nothing.
//
// The tessellation is made on a copy of the solid's topology (BRepBuilderAPI_Copy), so no
// face of the caller's shape, of a session or of a retained snapshot, is left holding a
// triangulation. The copy shares the geometry, so it costs little.
struct TessellatedVolume {
  double volume = 0.0;
  double area = 0.0;
  bool complete = false;
};

TessellatedVolume tessellated_volume(const TopoDS_Shape& solid, double deflection);

}  // namespace shape_checks
}  // namespace pysmesh
