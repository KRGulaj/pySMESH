// pySMESH binding — the medial axis of a face, and a constrained Delaunay over one.
//
// The **medial axis** of a 2-D region is the set of centres of the maximal circles that fit
// inside it. It is the natural answer to two questions a mesh-preparation workflow keeps
// asking and has no other exact way to answer: *where is the centreline of this thin region*
// and *how thick is it here*. SMESH computes it over Boost's Voronoi diagram, exactly, rather
// than by sampling.
//
// Four properties of the upstream class decide the shape of this binding, all measured:
//
//   * **A branch is not a dense polyline.** `getPoints` returns one point per medial-axis edge
//     plus one, so a straight branch is exactly two points. The axis of a rectangle is a spine
//     plus four corner arms, not one line.
//   * **The points come back in the face's own UV space**, scaled by a factor chosen when the
//     axis was built and kept private. `MedialAxis::getPoints` is the entry point that undoes
//     the scale; the `Branch` overload beside it does not, unless it is handed that same
//     private number.
//   * **A boundary point is (edge index, parameter on that edge)**, not a position. Turning it
//     into a point needs the edge it names, which is why this binding carries the ordered edge
//     list across and reports the caller's own EDGE ordinal beside each sample.
//   * **Branch 0 is not the spine.** Branches come out in construction order, so a thickness
//     query has to choose the branch it wants rather than index blindly.
//
// **`SMESH_Delaunay` is not bound, because it cannot answer anything under the pinned OCCT.**
// It builds its triangulation by handing OCCT's triangulator a bare vertex array, and the
// triangulator comes back having marked every triangle deleted and its live-element set empty
// — measured on a 12-node face: 62 elements, 62 deleted, 0 live. Its own entry point then
// finds no triangle beside any boundary node, so node traversal and point location both
// return nothing for every input. Declaring the boundary as frontier links and driving the
// triangulator directly was tried and behaves identically (58 elements, 58 deleted), so the
// cause is below SMESH rather than in how it is called. The finding reaches further than this
// file: the projection utilities build their own subclass of the same class.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <SMESH_Block.hxx>
#include <SMESH_MAT2d.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <gp_Pnt.hxx>
#include <gp_XY.hxx>

namespace pysmesh {

// Defined in shape.cpp — a medial axis reads the geometry, not a mesh, so it takes the same
// `Shape` the stateless API already speaks.
std::shared_ptr<ShapeData> shape_data_of(const py::object& shape_obj);

namespace mesher {
namespace {

// The face's edges in wire order, which is the order the medial axis indexes its boundary
// points by. Upstream's own accessor, so the two orders cannot disagree.
std::vector<TopoDS_Edge> ordered_edges(const TopoDS_Face& face) {
  std::list<TopoDS_Edge> edges;
  std::list<int> per_wire;
  SMESH_Block::GetOrderedEdges(face, edges, per_wire);
  return std::vector<TopoDS_Edge>(edges.begin(), edges.end());
}

gp_Pnt point_on_edge(const TopoDS_Edge& edge, double parameter) {
  BRepAdaptor_Curve curve(edge);
  return curve.Value(parameter);
}

}  // namespace

// ---- The medial axis -------------------------------------------------------------------- //

py::dict medial_axis(const py::object& shape_obj, int face_ordinal, double min_segment_length,
                     bool ignore_corners, int samples) {
  const std::shared_ptr<ShapeData> data = shape_data_of(shape_obj);
  const TopoDS_Face& face = data->face(face_ordinal);
  if (!(min_segment_length > 0.0)) {
    throw PysmeshError("medial_axis: min_segment_length must be > 0 (got " +
                       std::to_string(min_segment_length) +
                       "). It is the boundary discretisation step the axis is built from.");
  }
  if (samples < 2) {
    throw PysmeshError("medial_axis: samples must be >= 2 (got " + std::to_string(samples) +
                       "), so that both ends of every branch are measured.");
  }

  const std::vector<TopoDS_Edge> edges = ordered_edges(face);
  if (edges.empty()) {
    throw PysmeshError("medial_axis: face " + std::to_string(face_ordinal) +
                       " has no bounding edges.");
  }
  SMESH_MAT2d::MedialAxis axis(face, edges, min_segment_length, ignore_corners);

  BRepAdaptor_Surface surface(face);
  py::list branches;
  for (std::size_t b = 0; b < axis.nbBranches(); ++b) {
    const SMESH_MAT2d::Branch* branch = axis.getBranch(b);
    if (branch == nullptr) {
      continue;
    }

    // The axis itself, in the face's own parameter space and in model space beside it.
    std::vector<gp_XY> uv;
    axis.getPoints(branch, uv);
    std::vector<double> uv_flat;
    std::vector<double> xyz_flat;
    uv_flat.reserve(uv.size() * 2);
    xyz_flat.reserve(uv.size() * 3);
    for (const gp_XY& p : uv) {
      uv_flat.insert(uv_flat.end(), {p.X(), p.Y()});
      const gp_Pnt at = surface.Value(p.X(), p.Y());
      xyz_flat.insert(xyz_flat.end(), {at.X(), at.Y(), at.Z()});
    }

    // The two nearest boundary points at each sampled position along the branch. Their
    // distance is the local width, which is what a thin-region query is after.
    std::vector<double> parameters;
    std::vector<double> first_xyz;
    std::vector<double> second_xyz;
    std::vector<double> widths;
    std::vector<std::int64_t> first_edge;
    std::vector<std::int64_t> second_edge;
    for (int s = 0; s < samples; ++s) {
      const double t = static_cast<double>(s) / static_cast<double>(samples - 1);
      SMESH_MAT2d::BoundaryPoint bp1;
      SMESH_MAT2d::BoundaryPoint bp2;
      if (!branch->getBoundaryPoints(t, bp1, bp2)) {
        continue;
      }
      if (bp1._edgeIndex >= edges.size() || bp2._edgeIndex >= edges.size()) {
        continue;
      }
      const gp_Pnt p1 = point_on_edge(edges[bp1._edgeIndex], bp1._param);
      const gp_Pnt p2 = point_on_edge(edges[bp2._edgeIndex], bp2._param);
      parameters.push_back(t);
      first_xyz.insert(first_xyz.end(), {p1.X(), p1.Y(), p1.Z()});
      second_xyz.insert(second_xyz.end(), {p2.X(), p2.Y(), p2.Z()});
      widths.push_back(p1.Distance(p2));
      // The caller's own EDGE ordinal, never an index private to the axis.
      first_edge.push_back(data->edges.FindIndex(edges[bp1._edgeIndex]));
      second_edge.push_back(data->edges.FindIndex(edges[bp2._edgeIndex]));
    }

    py::dict entry;
    entry["uv"] = rows_to_array(uv_flat, 2);
    entry["points"] = rows_to_array(xyz_flat, 3);
    entry["end_types"] =
        py::make_tuple(static_cast<int>(branch->getEnd(false)->_type),
                       static_cast<int>(branch->getEnd(true)->_type));
    entry["parameters"] = vector_to_array(parameters);
    entry["boundary1"] = rows_to_array(first_xyz, 3);
    entry["boundary2"] = rows_to_array(second_xyz, 3);
    entry["boundary1_edge"] = vector_to_array(first_edge);
    entry["boundary2_edge"] = vector_to_array(second_edge);
    entry["widths"] = vector_to_array(widths);
    branches.append(entry);
  }

  py::dict out;
  out["face"] = face_ordinal;
  out["branches"] = branches;
  // A branch point is where three or more branches meet. An L-shaped region has one; a
  // rectangle has none, whatever the number of branches its corner arms add.
  out["branch_points"] = static_cast<std::int64_t>(axis.getBranchPoints().size());
  out["boundary_edges"] = static_cast<std::int64_t>(axis.getBoundary().nbEdges());
  return out;
}

}  // namespace mesher
}  // namespace pysmesh
