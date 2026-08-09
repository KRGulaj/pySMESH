// pySMESH binding — searching a mesh: location, ray casting, classification, offset, slot.
//
// This is the part of SMESH that answers questions *about* a mesh rather than building or
// changing one, and two of its answers have no counterpart anywhere else in this stack: a ray
// cast against a mesh, and an in/out classification against a closed surface of triangles.
// A third — the distance from a point to a **volume** cell — is likewise unavailable to a
// surface-only pipeline.
//
// Three things shape the code:
//
//   * **Every query takes a batch of points.** One searcher builds an octree over the whole
//     mesh, so answering a single point through it would pay for the tree per question. The
//     batch form builds it once and is what a caller with a point cloud actually wants.
//   * **The searcher's line query is a broad phase.** `GetElementsNearLine` returns every
//     element whose *bounding box* the line crosses, which is a candidate set and not a hit
//     list. It is exposed as such, under its own name; `ray_hits` is the narrow phase over
//     it, and that is the one that answers "where does this ray meet the mesh".
//   * **The narrow phase runs over SMESH's own triangulation of each face**, so a quadrangle
//     and a polygon are handled the same way the rest of the library handles them, rather
//     than by a second decomposition written here that could disagree with it.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Bnd_B3.hxx>
#include <SMDS_ElemIterator.hxx>
#include <SMDS_MeshElement.hxx>
#include <SMDS_MeshNode.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_MeshAlgos.hxx>
#include <SMESH_TypeDefs.hxx>
#include <TopAbs_State.hxx>
#include <gp_Ax1.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>

namespace pysmesh {
namespace mesher {
namespace {

using Searcher = std::unique_ptr<SMESH_ElementSearcher>;

Searcher searcher_for(SMDS_Mesh& mesh) {
  Searcher out(SMESH_MeshAlgos::GetElementSearcher(mesh));
  if (!out) {
    throw PysmeshError("The mesh search structure could not be built.");
  }
  return out;
}

std::vector<std::int64_t> ids_of(const std::vector<const SMDS_MeshElement*>& elements) {
  std::vector<std::int64_t> out;
  out.reserve(elements.size());
  for (const SMDS_MeshElement* element : elements) {
    out.push_back(static_cast<std::int64_t>(element->GetID()));
  }
  return out;
}

// A ray against one triangle, by Möller and Trumbore. Returns the ray parameter of the hit,
// or a negative number when there is none. `tolerance` widens the barycentric test so a ray
// striking exactly on a shared edge is not lost by both triangles that own it.
double ray_triangle(const gp_Pnt& origin, const gp_Dir& direction, const gp_XYZ& a,
                    const gp_XYZ& b, const gp_XYZ& c, double tolerance) {
  const gp_XYZ e1 = b - a;
  const gp_XYZ e2 = c - a;
  const gp_XYZ d(direction.X(), direction.Y(), direction.Z());
  const gp_XYZ p = d ^ e2;
  const double det = e1 * p;
  if (std::fabs(det) < 1e-300) {
    return -1.0;  // the ray runs in the triangle's own plane
  }
  const double inv = 1.0 / det;
  const gp_XYZ t = origin.XYZ() - a;
  const double u = (t * p) * inv;
  if (u < -tolerance || u > 1.0 + tolerance) {
    return -1.0;
  }
  const gp_XYZ q = t ^ e1;
  const double v = (d * q) * inv;
  if (v < -tolerance || u + v > 1.0 + tolerance) {
    return -1.0;
  }
  return (e2 * q) * inv;
}

}  // namespace

// ---- Location --------------------------------------------------------------------------- //

py::dict Mesher::find_elements_by_point(const py::object& points, int family) const {
  ensure_open();
  const SMDSAbs_ElementType type = family_of(family);
  const auto table = point_table(points, "Mesher.find_elements_by_point: points", 3);
  const py::ssize_t rows = table.shape(0);
  const double* xyz = table.data();

  Searcher searcher = searcher_for(*meshDS_);
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> ids;
  offsets.reserve(static_cast<std::size_t>(rows) + 1);
  offsets.push_back(0);
  std::vector<const SMDS_MeshElement*> found;
  for (py::ssize_t i = 0; i < rows; ++i) {
    found.clear();
    searcher->FindElementsByPoint(gp_Pnt(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]), type,
                                  found);
    for (const std::int64_t id : ids_of(found)) {
      ids.push_back(id);
    }
    offsets.push_back(static_cast<std::int64_t>(ids.size()));
  }

  py::dict out;
  out["offsets"] = vector_to_array(offsets);
  out["ids"] = vector_to_array(ids);
  return out;
}

py::dict Mesher::find_closest(const py::object& points, int family) const {
  ensure_open();
  const SMDSAbs_ElementType type = family_of(family);
  const auto table = point_table(points, "Mesher.find_closest: points", 3);
  const py::ssize_t rows = table.shape(0);
  const double* xyz = table.data();

  Searcher searcher = searcher_for(*meshDS_);
  std::vector<std::int64_t> ids;
  ids.reserve(static_cast<std::size_t>(rows));
  for (py::ssize_t i = 0; i < rows; ++i) {
    const SMDS_MeshElement* element = searcher->FindClosestTo(
        gp_Pnt(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]), type);
    // 0 is not a mesh id, so it is the honest "nothing of that family is in the mesh".
    ids.push_back(element == nullptr ? 0 : static_cast<std::int64_t>(element->GetID()));
  }

  py::dict out;
  out["ids"] = vector_to_array(ids);
  return out;
}

py::dict Mesher::closest_distance(const py::object& points, int family) const {
  ensure_open();
  const SMDSAbs_ElementType type = family_of(family);
  const auto table = point_table(points, "Mesher.closest_distance: points", 3);
  const py::ssize_t rows = table.shape(0);
  const double* xyz = table.data();

  Searcher searcher = searcher_for(*meshDS_);
  std::vector<std::int64_t> ids;
  std::vector<double> distances;
  std::vector<double> closest;
  ids.reserve(static_cast<std::size_t>(rows));
  distances.reserve(static_cast<std::size_t>(rows));
  closest.reserve(static_cast<std::size_t>(rows) * 3);
  for (py::ssize_t i = 0; i < rows; ++i) {
    const gp_Pnt point(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    const SMDS_MeshElement* element = searcher->FindClosestTo(point, type);
    if (element == nullptr) {
      ids.push_back(0);
      distances.push_back(-1.0);
      closest.insert(closest.end(), {0.0, 0.0, 0.0});
      continue;
    }
    gp_XYZ where;
    // The element-taking overload dispatches on the family, so a volume cell answers here
    // exactly as a face does — which is the measurement a surface-only pipeline cannot make.
    const double distance = SMESH_MeshAlgos::GetDistance(element, point, &where);
    ids.push_back(static_cast<std::int64_t>(element->GetID()));
    distances.push_back(distance);
    closest.insert(closest.end(), {where.X(), where.Y(), where.Z()});
  }

  py::dict out;
  out["ids"] = vector_to_array(ids);
  out["distances"] = vector_to_array(distances);
  out["closest_points"] = rows_to_array(closest, 3);
  return out;
}

py::dict Mesher::project_points(const py::object& points, int family) const {
  ensure_open();
  const SMDSAbs_ElementType type = family_of(family);
  const auto table = point_table(points, "Mesher.project_points: points", 3);
  const py::ssize_t rows = table.shape(0);
  const double* xyz = table.data();

  Searcher searcher = searcher_for(*meshDS_);
  std::vector<double> projected;
  std::vector<std::int64_t> ids;
  projected.reserve(static_cast<std::size_t>(rows) * 3);
  ids.reserve(static_cast<std::size_t>(rows));
  for (py::ssize_t i = 0; i < rows; ++i) {
    const SMDS_MeshElement* onto = nullptr;
    const gp_XYZ where = searcher->Project(
        gp_Pnt(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]), type, &onto);
    projected.insert(projected.end(), {where.X(), where.Y(), where.Z()});
    ids.push_back(onto == nullptr ? 0 : static_cast<std::int64_t>(onto->GetID()));
  }

  py::dict out;
  out["points"] = rows_to_array(projected, 3);
  out["ids"] = vector_to_array(ids);
  return out;
}

py::dict Mesher::point_state(const py::object& points) const {
  ensure_open();
  const auto table = point_table(points, "Mesher.point_state: points", 3);
  const py::ssize_t rows = table.shape(0);
  const double* xyz = table.data();

  Searcher searcher = searcher_for(*meshDS_);
  std::vector<std::int64_t> states;
  states.reserve(static_cast<std::size_t>(rows));
  for (py::ssize_t i = 0; i < rows; ++i) {
    // OCCT's own TopAbs_State, forwarded unchanged: IN, OUT, ON, UNKNOWN.
    states.push_back(static_cast<std::int64_t>(
        searcher->GetPointState(gp_Pnt(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]))));
  }

  py::dict out;
  out["states"] = vector_to_array(states);
  return out;
}

// ---- Region queries --------------------------------------------------------------------- //

py::dict Mesher::elements_in_sphere(const std::vector<double>& centre, double radius,
                                    int family) const {
  ensure_open();
  require_triple(centre, "Mesher.elements_in_sphere: centre");
  if (!(radius > 0.0)) {
    throw PysmeshError("Mesher.elements_in_sphere: the radius must be > 0 (got " +
                       std::to_string(radius) + ").");
  }
  std::vector<const SMDS_MeshElement*> found;
  searcher_for(*meshDS_)->GetElementsInSphere(gp_XYZ(centre[0], centre[1], centre[2]), radius,
                                              family_of(family), found);

  py::dict out;
  out["ids"] = vector_to_array(ids_of(found));
  return out;
}

py::dict Mesher::elements_in_box(const std::vector<double>& minimum,
                                 const std::vector<double>& maximum, int family) const {
  ensure_open();
  require_triple(minimum, "Mesher.elements_in_box: minimum");
  require_triple(maximum, "Mesher.elements_in_box: maximum");
  for (int axis = 0; axis < 3; ++axis) {
    if (!(maximum[axis] >= minimum[axis])) {
      throw PysmeshError("Mesher.elements_in_box: the box is inverted on axis " +
                         std::to_string(axis) + ".");
    }
  }
  Bnd_B3d box;
  box.Add(gp_XYZ(minimum[0], minimum[1], minimum[2]));
  box.Add(gp_XYZ(maximum[0], maximum[1], maximum[2]));

  std::vector<const SMDS_MeshElement*> found;
  searcher_for(*meshDS_)->GetElementsInBox(box, family_of(family), found);

  py::dict out;
  out["ids"] = vector_to_array(ids_of(found));
  return out;
}

// ---- Ray casting ------------------------------------------------------------------------ //

py::dict Mesher::elements_near_line(const std::vector<double>& origin,
                                    const std::vector<double>& direction, int family) const {
  ensure_open();
  require_triple(origin, "Mesher.elements_near_line: origin");
  require_triple(direction, "Mesher.elements_near_line: direction");
  if (direction[0] == 0.0 && direction[1] == 0.0 && direction[2] == 0.0) {
    throw PysmeshError("Mesher.elements_near_line: the direction must not be the zero "
                       "vector.");
  }
  const gp_Ax1 line(gp_Pnt(origin[0], origin[1], origin[2]),
                    gp_Dir(direction[0], direction[1], direction[2]));

  std::vector<const SMDS_MeshElement*> found;
  searcher_for(*meshDS_)->GetElementsNearLine(line, family_of(family), found);

  py::dict out;
  out["ids"] = vector_to_array(ids_of(found));
  return out;
}

py::dict Mesher::ray_hits(const std::vector<double>& origin,
                          const std::vector<double>& direction, double tolerance) const {
  ensure_open();
  require_triple(origin, "Mesher.ray_hits: origin");
  require_triple(direction, "Mesher.ray_hits: direction");
  if (direction[0] == 0.0 && direction[1] == 0.0 && direction[2] == 0.0) {
    throw PysmeshError("Mesher.ray_hits: the direction must not be the zero vector.");
  }
  if (!(tolerance >= 0.0)) {
    throw PysmeshError("Mesher.ray_hits: the tolerance must be >= 0 (got " +
                       std::to_string(tolerance) + ").");
  }
  const gp_Pnt start(origin[0], origin[1], origin[2]);
  const gp_Dir along(direction[0], direction[1], direction[2]);

  // Broad phase: every face whose bounding box the infinite line crosses.
  std::vector<const SMDS_MeshElement*> candidates;
  searcher_for(*meshDS_)->GetElementsNearLine(gp_Ax1(start, along), SMDSAbs_Face, candidates);

  // Narrow phase, over SMESH's own triangulation of each candidate, so a quadrangle and a
  // polygon are decomposed the way the rest of the library decomposes them.
  SMESH_MeshAlgos::Triangulate triangulator;
  std::vector<const SMDS_MeshNode*> nodes;
  std::vector<std::pair<double, std::int64_t>> hits;
  for (const SMDS_MeshElement* face : candidates) {
    nodes.clear();
    const int triangles = triangulator.GetTriangles(face, nodes);
    double best = -1.0;
    for (int t = 0; t < triangles; ++t) {
      const double at = ray_triangle(start, along, SMESH_NodeXYZ(nodes[3 * t]),
                                     SMESH_NodeXYZ(nodes[3 * t + 1]),
                                     SMESH_NodeXYZ(nodes[3 * t + 2]), tolerance);
      // One face contributes at most one hit however many triangles it was cut into, so a
      // ray through a quadrangle's own diagonal is not counted twice.
      if (at >= 0.0 && (best < 0.0 || at < best)) {
        best = at;
      }
    }
    if (best >= 0.0) {
      hits.emplace_back(best, static_cast<std::int64_t>(face->GetID()));
    }
  }
  std::sort(hits.begin(), hits.end());

  std::vector<std::int64_t> ids;
  std::vector<double> parameters;
  std::vector<double> points;
  ids.reserve(hits.size());
  parameters.reserve(hits.size());
  points.reserve(hits.size() * 3);
  // Two faces met at the same distance are one crossing of the surface, not two: a ray that
  // strikes a shared edge legitimately meets both faces that own it. The face list is the
  // geometric answer and keeps both; the crossing count is what a parity or leak test needs.
  std::int64_t crossings = 0;
  double previous = 0.0;
  for (const std::pair<double, std::int64_t>& hit : hits) {
    if (ids.empty() || std::fabs(hit.first - previous) > 1e-9 * (1.0 + std::fabs(hit.first))) {
      ++crossings;
      previous = hit.first;
    }
    ids.push_back(hit.second);
    parameters.push_back(hit.first);
    const gp_Pnt where = start.Translated(gp_Vec(along) * hit.first);
    points.insert(points.end(), {where.X(), where.Y(), where.Z()});
  }

  py::dict out;
  out["ids"] = vector_to_array(ids);
  out["parameters"] = vector_to_array(parameters);
  out["points"] = rows_to_array(points, 3);
  out["candidates"] = static_cast<std::int64_t>(candidates.size());
  out["crossings"] = crossings;
  return out;
}

// ---- Feature edges and patches ---------------------------------------------------------- //

py::dict Mesher::sharp_edges(double angle, bool add_existing) const {
  ensure_open();
  if (!(angle >= 0.0) || angle > 180.0) {
    throw PysmeshError("Mesher.sharp_edges: the angle must be between 0 and 180 degrees "
                       "(got " +
                       std::to_string(angle) + ").");
  }
  const std::vector<SMESH_MeshAlgos::Edge> edges =
      SMESH_MeshAlgos::FindSharpEdges(meshDS_, angle, add_existing);

  std::vector<std::int64_t> first;
  std::vector<std::int64_t> second;
  std::vector<std::int64_t> medium;
  first.reserve(edges.size());
  second.reserve(edges.size());
  medium.reserve(edges.size());
  for (const SMESH_MeshAlgos::Edge& edge : edges) {
    first.push_back(static_cast<std::int64_t>(edge._node1->GetID()));
    second.push_back(static_cast<std::int64_t>(edge._node2->GetID()));
    // 0 where the edge is linear, because a mesh id is never 0.
    medium.push_back(edge._medium == nullptr
                         ? 0
                         : static_cast<std::int64_t>(edge._medium->GetID()));
  }

  py::dict out;
  out["node1"] = vector_to_array(first);
  out["node2"] = vector_to_array(second);
  out["medium"] = vector_to_array(medium);
  return out;
}

py::dict Mesher::separate_faces_by_edges(const py::object& node1, const py::object& node2,
                                         const py::object& medium) const {
  ensure_open();
  const std::vector<std::int64_t> first = node1.cast<std::vector<std::int64_t>>();
  const std::vector<std::int64_t> second = node2.cast<std::vector<std::int64_t>>();
  const std::vector<std::int64_t> middle = medium.cast<std::vector<std::int64_t>>();
  if (first.size() != second.size() || first.size() != middle.size()) {
    throw PysmeshError("Mesher.separate_faces_by_edges: node1, node2 and medium must be the "
                       "same length.");
  }

  std::vector<SMESH_MeshAlgos::Edge> edges;
  edges.reserve(first.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    SMESH_MeshAlgos::Edge edge;
    edge._node1 = meshDS_->FindNode(static_cast<smIdType>(first[i]));
    edge._node2 = meshDS_->FindNode(static_cast<smIdType>(second[i]));
    edge._medium = middle[i] == 0 ? nullptr
                                  : meshDS_->FindNode(static_cast<smIdType>(middle[i]));
    if (edge._node1 == nullptr || edge._node2 == nullptr) {
      throw PysmeshError("Mesher.separate_faces_by_edges: edge " + std::to_string(i) +
                         " names a node the mesh does not have.");
    }
    edges.push_back(edge);
  }

  const std::vector<std::vector<const SMDS_MeshElement*>> patches =
      SMESH_MeshAlgos::SeparateFacesByEdges(meshDS_, edges);

  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> ids;
  offsets.reserve(patches.size() + 1);
  offsets.push_back(0);
  for (const std::vector<const SMDS_MeshElement*>& patch : patches) {
    for (const std::int64_t id : ids_of(patch)) {
      ids.push_back(id);
    }
    offsets.push_back(static_cast<std::int64_t>(ids.size()));
  }

  py::dict out;
  out["offsets"] = vector_to_array(offsets);
  out["ids"] = vector_to_array(ids);
  return out;
}

// ---- Merge diagnosis and slot cutting --------------------------------------------------- //

py::dict Mesher::de_merge(std::int64_t element, const py::list& groups) const {
  ensure_open();
  const SMDS_MeshElement* target = meshDS_->FindElement(static_cast<smIdType>(element));
  if (target == nullptr) {
    throw PysmeshError("Mesher.de_merge: the mesh has no element with id " +
                       std::to_string(element) + ".");
  }

  // The proposed merge, as the map each group implies: the first node of a group survives
  // and the rest are replaced by it. Upstream takes the element's connectivity *as it would
  // be afterwards* and appends the nodes that must be kept apart to it, so the map has to be
  // applied here — handing it the element's current nodes would ask about no merge at all.
  std::map<const SMDS_MeshNode*, const SMDS_MeshNode*> replaced;
  for (const py::handle& entry : groups) {
    const std::vector<std::int64_t> ids = entry.cast<std::vector<std::int64_t>>();
    if (ids.size() < 2) {
      throw PysmeshError("Mesher.de_merge: every group must name at least two nodes.");
    }
    const SMDS_MeshNode* survivor = meshDS_->FindNode(static_cast<smIdType>(ids[0]));
    if (survivor == nullptr) {
      throw PysmeshError("Mesher.de_merge: the mesh has no node with id " +
                         std::to_string(ids[0]) + ".");
    }
    for (std::size_t i = 1; i < ids.size(); ++i) {
      const SMDS_MeshNode* gone = meshDS_->FindNode(static_cast<smIdType>(ids[i]));
      if (gone == nullptr) {
        throw PysmeshError("Mesher.de_merge: the mesh has no node with id " +
                           std::to_string(ids[i]) + ".");
      }
      replaced[gone] = survivor;
    }
  }

  std::vector<const SMDS_MeshNode*> replacement;
  replacement.reserve(static_cast<std::size_t>(target->NbNodes()));
  for (SMDS_NodeIteratorPtr it = target->nodeIterator(); it->more();) {
    const SMDS_MeshNode* node = it->next();
    const auto found = replaced.find(node);
    replacement.push_back(found == replaced.end() ? node : found->second);
  }

  std::vector<const SMDS_MeshNode*> keep_apart;
  SMESH_MeshAlgos::DeMerge(target, replacement, keep_apart);

  std::vector<std::int64_t> replacement_ids;
  std::vector<std::int64_t> keep_apart_ids;
  for (const SMDS_MeshNode* node : replacement) {
    replacement_ids.push_back(node == nullptr ? 0 : static_cast<std::int64_t>(node->GetID()));
  }
  for (const SMDS_MeshNode* node : keep_apart) {
    keep_apart_ids.push_back(node == nullptr ? 0 : static_cast<std::int64_t>(node->GetID()));
  }

  py::dict out;
  out["nodes"] = vector_to_array(replacement_ids);
  out["keep_apart"] = vector_to_array(keep_apart_ids);
  return out;
}

py::dict Mesher::make_slot(double width, const std::vector<std::int64_t>& segments) {
  ensure_open();
  if (!(width > 0.0)) {
    throw PysmeshError("Mesher.make_slot: the width must be > 0 (got " +
                       std::to_string(width) + ").");
  }

  TIDSortedElemSet chosen;
  for (const std::int64_t id : segments) {
    const SMDS_MeshElement* element = meshDS_->FindElement(static_cast<smIdType>(id));
    if (element == nullptr || element->GetType() != SMDSAbs_Edge) {
      throw PysmeshError("Mesher.make_slot: " + std::to_string(id) +
                         " does not name a 1-D element of this mesh.");
    }
    chosen.insert(element);
  }
  SMDS_ElemIteratorPtr it = chosen.empty()
                                ? meshDS_->elementsIterator(SMDSAbs_Edge)
                                : SMESHUtils::elemSetIterator(chosen);

  std::vector<SMDS_MeshGroup*> nothing_to_update;
  std::vector<SMESH_MeshAlgos::Edge> boundary;
  try {
    boundary = SMESH_MeshAlgos::MakeSlot(it, width, meshDS_, nothing_to_update);
  } catch (const PysmeshError&) {
    throw;
  } catch (const std::exception& failure) {
    throw PysmeshError(std::string("Mesher.make_slot: ") + failure.what());
  }
  meshDS_->Modified();

  std::vector<std::int64_t> first;
  std::vector<std::int64_t> second;
  for (const SMESH_MeshAlgos::Edge& edge : boundary) {
    first.push_back(static_cast<std::int64_t>(edge._node1->GetID()));
    second.push_back(static_cast<std::int64_t>(edge._node2->GetID()));
  }

  py::dict out;
  out["node1"] = vector_to_array(first);
  out["node2"] = vector_to_array(second);
  return out;
}

}  // namespace mesher
}  // namespace pysmesh
