// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — the mesh editor.
//
// SMESH_MeshEditor is the part of the library that changes a mesh after it has been computed,
// and the operations bound here are the ones a solver-facing workflow cannot do any other way
// in this stack: a second-order conversion, an internal wall, a re-orientation decided by the
// volumes a shell bounds, a sweep, a surface offset, and the merge/sew family that makes two
// separately meshed regions into one.
//
// Four things shape the code, all measured rather than assumed:
//
//   * **Nothing here publishes the mesh.** SMDS records that it changed but advances its
//     modification time only when asked, and the editor asks in exactly one of its dozens of
//     operations — the SALOME layer above it does the rest. A group defined by a filter, and
//     several controls, test that time to decide whether to re-evaluate. So every operation
//     here ends with `Modified()`.
//   * **An empty id list means "the whole mesh"** in upstream's own calls, and that convention
//     is kept rather than replaced, because several of these operations are only useful over
//     everything.
//   * **`Offset` dereferences its target mesh without checking it**, so the null that would
//     naturally mean "in place" is a crash. The target is always this mesh.
//   * **The offset refuses a mesh it cannot handle by throwing**, and it tests the *whole*
//     source mesh rather than the faces it was given, so offsetting a triangular patch of a
//     mesh that also holds quadrangles is refused. Both are translated here.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

#include <cstdint>
#include <list>
#include <set>
#include <string>
#include <vector>

#include <SMDS_ElemIterator.hxx>
#include <SMDS_MeshElement.hxx>
#include <SMDS_MeshNode.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_ControlsDef.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_MeshEditor.hxx>
#include <gp_Ax1.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>

namespace pysmesh {
namespace mesher {
namespace {

// The four counts an edit is measured against. Every operation reports them either side of
// itself, because "what changed" is the only thing a caller can act on.
struct Counts {
  std::int64_t nodes;
  std::int64_t edges;
  std::int64_t faces;
  std::int64_t volumes;
};

Counts counts_of(const SMESHDS_Mesh& ds) {
  return {static_cast<std::int64_t>(ds.NbNodes()), static_cast<std::int64_t>(ds.NbEdges()),
          static_cast<std::int64_t>(ds.NbFaces()),
          static_cast<std::int64_t>(ds.NbVolumes())};
}

py::dict report(const Counts& before, const Counts& after, std::int64_t merged) {
  py::dict out;
  out["nodes_before"] = before.nodes;
  out["nodes_after"] = after.nodes;
  out["edges_before"] = before.edges;
  out["edges_after"] = after.edges;
  out["faces_before"] = before.faces;
  out["faces_after"] = after.faces;
  out["volumes_before"] = before.volumes;
  out["volumes_after"] = after.volumes;
  out["groups_merged"] = merged;
  return out;
}

// Publish the edit, so that a filtered group and the change-tracking controls see it.
void publish(SMESHDS_Mesh& ds) { ds.Modified(); }

const SMDS_MeshElement* element_of(const SMESHDS_Mesh& ds, std::int64_t id,
                                   const char* owner) {
  const SMDS_MeshElement* element = ds.FindElement(static_cast<smIdType>(id));
  if (element == nullptr) {
    throw PysmeshError(std::string(owner) + ": the mesh has no element with id " +
                       std::to_string(id) + ".");
  }
  return element;
}

const SMDS_MeshNode* node_of(const SMESHDS_Mesh& ds, std::int64_t id, const char* owner) {
  const SMDS_MeshNode* node = ds.FindNode(static_cast<smIdType>(id));
  if (node == nullptr) {
    throw PysmeshError(std::string(owner) + ": the mesh has no node with id " +
                       std::to_string(id) + ".");
  }
  return node;
}

// An id list to an element set. Empty stays empty, which is upstream's "the whole mesh".
TIDSortedElemSet element_set(const SMESHDS_Mesh& ds, const std::vector<std::int64_t>& ids,
                             const char* owner) {
  TIDSortedElemSet out;
  for (const std::int64_t id : ids) {
    out.insert(element_of(ds, id, owner));
  }
  return out;
}

// The same, refusing anything that is not of the wanted family — a face id handed to a
// volume argument is a caller error and is worth naming rather than quietly ignoring.
TIDSortedElemSet element_set_of_family(const SMESHDS_Mesh& ds,
                                       const std::vector<std::int64_t>& ids,
                                       SMDSAbs_ElementType family, const char* owner) {
  TIDSortedElemSet out;
  for (const std::int64_t id : ids) {
    const SMDS_MeshElement* element = element_of(ds, id, owner);
    if (element->GetType() != family) {
      throw PysmeshError(std::string(owner) + ": element " + std::to_string(id) +
                         " is not of the expected family.");
    }
    out.insert(element);
  }
  return out;
}

// The criterion a face-splitting operation chooses a diagonal or a neighbour by. Named
// through the same control catalogue the quality surface uses, so there is one list of
// measures in this binding rather than two.
SMESH::Controls::NumericalFunctorPtr criterion_of(const std::string& name,
                                                  const py::dict& params, const Mesher* owner,
                                                  const char* caller) {
  SMESH::Controls::NumericalFunctorPtr functor = build_functor(name, params, owner);
  if (!functor) {
    throw PysmeshError(std::string(caller) + ": unknown quality control '" + name +
                       "' as the splitting criterion.");
  }
  return functor;
}

// SMESH's own words for a refused sew. The enum is the only channel there is.
const char* sew_error_text(int code) {
  switch (code) {
    case SMESH_MeshEditor::SEW_OK:
      return "the sew succeeded";
    case SMESH_MeshEditor::SEW_BORDER1_NOT_FOUND:
      return "the first free border was not found from the nodes given";
    case SMESH_MeshEditor::SEW_BORDER2_NOT_FOUND:
      return "the second free border was not found from the nodes given";
    case SMESH_MeshEditor::SEW_BOTH_BORDERS_NOT_FOUND:
      return "neither free border was found from the nodes given";
    case SMESH_MeshEditor::SEW_BAD_SIDE_NODES:
      return "the nodes given do not lie on the side they were offered for";
    case SMESH_MeshEditor::SEW_VOLUMES_TO_SPLIT:
      return "a volume element shares a link the sew would have to split";
    case SMESH_MeshEditor::SEW_DIFF_NB_OF_ELEMENTS:
      return "the two sides hold a different number of elements";
    case SMESH_MeshEditor::SEW_TOPO_DIFF_SETS_OF_ELEMENTS:
      return "the two sides do not have matching connectivity";
    case SMESH_MeshEditor::SEW_BAD_SIDE1_NODES:
      return "the first side's nodes are not on its border, or are not linked";
    case SMESH_MeshEditor::SEW_BAD_SIDE2_NODES:
      return "the second side's nodes are not on its border, or are not linked";
    default:
      return "the sew failed inside SMESH";
  }
}

// The split methods upstream defines, kept as its own values so the two enumerations cannot
// drift. The prism forms need a facet to split into triangles, which is what the normal
// selects; the tetrahedral forms ignore it.
bool splits_into_prisms(int method) {
  return method == SMESH_MeshEditor::HEXA_TO_2_PRISMS ||
         method == SMESH_MeshEditor::HEXA_TO_4_PRISMS;
}

}  // namespace

// ---- Order conversion ------------------------------------------------------------------ //

void Mesher::convert_to_quadratic(bool force_3d, bool bi_quadratic) {
  ensure_open();
  SMESH_MeshEditor editor(mesh_);
  editor.ConvertToQuadratic(force_3d, bi_quadratic);
  publish(*meshDS_);
}

bool Mesher::convert_from_quadratic() {
  ensure_open();
  SMESH_MeshEditor editor(mesh_);
  const bool converted = editor.ConvertFromQuadratic();
  publish(*meshDS_);
  return converted;
}

py::dict Mesher::split_quadratic_into_linear(const std::vector<std::int64_t>& elements) {
  ensure_open();
  const Counts before = counts_of(*meshDS_);
  SMESH_MeshEditor editor(mesh_);
  TIDSortedElemSet chosen =
      element_set(*meshDS_, elements, "Mesher.split_quadratic_into_linear");
  if (chosen.empty()) {
    // This is the one editing call upstream reads an empty set as "nothing" rather than as
    // "everything", so the package's own convention is applied here instead.
    for (const SMDSAbs_ElementType family : {SMDSAbs_Edge, SMDSAbs_Face, SMDSAbs_Volume}) {
      for (SMDS_ElemIteratorPtr it = meshDS_->elementsIterator(family); it->more();) {
        chosen.insert(it->next());
      }
    }
  }
  editor.SplitBiQuadraticIntoLinear(chosen);
  publish(*meshDS_);
  return report(before, counts_of(*meshDS_), 0);
}

// ---- Volume splitting ------------------------------------------------------------------ //

py::dict Mesher::split_volumes(int method, double nx, double ny, double nz) {
  ensure_open();
  if (method < SMESH_MeshEditor::HEXA_TO_5 || method > SMESH_MeshEditor::HEXA_TO_4_PRISMS) {
    throw PysmeshError("Mesher.split_volumes: unknown split method " +
                       std::to_string(method) + ".");
  }
  if (nx == 0.0 && ny == 0.0 && nz == 0.0) {
    throw PysmeshError("Mesher.split_volumes: the facet normal must not be the zero vector.");
  }

  SMESH_MeshEditor editor(mesh_);
  TIDSortedElemSet volumes;
  for (SMDS_ElemIteratorPtr it = meshDS_->elementsIterator(SMDSAbs_Volume); it->more();) {
    volumes.insert(it->next());
  }
  if (volumes.empty()) {
    throw PysmeshError("Mesher.split_volumes: the mesh has no volume elements to split.");
  }
  const Counts before = counts_of(*meshDS_);

  SMESH_MeshEditor::TFacetOfElem facets;
  if (splits_into_prisms(method)) {
    // Which facet of each hexahedron becomes two triangles. Upstream picks it per element
    // from the normal, so a caller says which way to cut rather than naming facets.
    TIDSortedElemSet hexahedra = volumes;
    editor.GetHexaFacetsToSplit(hexahedra, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(nx, ny, nz)),
                                facets);
  } else {
    // A negative facet id is upstream's own "split into tetrahedra" marker.
    for (const SMDS_MeshElement* volume : volumes) {
      facets.insert(std::make_pair(volume, -1));
    }
  }
  editor.SplitVolumes(facets, method);
  publish(*meshDS_);

  py::dict out = report(before, counts_of(*meshDS_), 0);
  return out;
}

// ---- Coincidence and merging ------------------------------------------------------------ //

py::list Mesher::find_coincident_nodes(double tolerance,
                                       const std::vector<std::int64_t>& nodes,
                                       bool separate_corners_and_medium) const {
  ensure_open();
  if (!(tolerance >= 0.0)) {
    // The positive form, so a NaN tolerance is caught here rather than reaching the search
    // as an undefined comparison.
    throw PysmeshError("Mesher.find_coincident_nodes: the tolerance must be >= 0 (got " +
                       std::to_string(tolerance) + ").");
  }
  TIDSortedNodeSet chosen;
  for (const std::int64_t id : nodes) {
    chosen.insert(node_of(*meshDS_, id, "Mesher.find_coincident_nodes"));
  }

  SMESH_MeshEditor editor(mesh_);
  SMESH_MeshEditor::TListOfListOfNodes found;
  editor.FindCoincidentNodes(chosen, tolerance, found, separate_corners_and_medium);

  py::list out;
  for (const std::list<const SMDS_MeshNode*>& group : found) {
    std::vector<std::int64_t> ids;
    ids.reserve(group.size());
    for (const SMDS_MeshNode* node : group) {
      ids.push_back(static_cast<std::int64_t>(node->GetID()));
    }
    out.append(vector_to_array(ids));
  }
  return out;
}

py::dict Mesher::merge_node_groups(const py::list& groups, bool avoid_making_holes) {
  ensure_open();
  SMESH_MeshEditor::TListOfListOfNodes to_merge;
  for (const py::handle& entry : groups) {
    std::list<const SMDS_MeshNode*> group;
    for (const std::int64_t id : entry.cast<std::vector<std::int64_t>>()) {
      group.push_back(node_of(*meshDS_, id, "Mesher.merge_node_groups"));
    }
    if (group.size() < 2) {
      throw PysmeshError("Mesher.merge_node_groups: every group must name at least two "
                         "nodes; the first of each survives and the rest are replaced.");
    }
    to_merge.push_back(group);
  }

  const Counts before = counts_of(*meshDS_);
  const std::int64_t merged = static_cast<std::int64_t>(to_merge.size());
  SMESH_MeshEditor editor(mesh_);
  editor.MergeNodes(to_merge, avoid_making_holes);
  publish(*meshDS_);
  return report(before, counts_of(*meshDS_), merged);
}

py::dict Mesher::merge_nodes(double tolerance) {
  ensure_open();
  if (!(tolerance >= 0.0)) {
    throw PysmeshError("Mesher.merge_nodes: the tolerance must be >= 0 (got " +
                       std::to_string(tolerance) + ").");
  }

  SMESH_MeshEditor editor(mesh_);
  TIDSortedNodeSet whole_mesh;  // empty means "search the whole mesh"
  SMESH_MeshEditor::TListOfListOfNodes coincident;
  editor.FindCoincidentNodes(whole_mesh, tolerance, coincident,
                             /*theSeparateCornersAndMedium=*/false);

  const Counts before = counts_of(*meshDS_);
  const std::int64_t merged = static_cast<std::int64_t>(coincident.size());
  editor.MergeNodes(coincident);
  publish(*meshDS_);
  return report(before, counts_of(*meshDS_), merged);
}

py::list Mesher::find_equal_elements(const std::vector<std::int64_t>& elements) const {
  ensure_open();
  TIDSortedElemSet chosen = element_set(*meshDS_, elements, "Mesher.find_equal_elements");
  SMESH_MeshEditor editor(mesh_);
  SMESH_MeshEditor::TListOfListOfElementsID found;
  editor.FindEqualElements(chosen, found);

  py::list out;
  for (const std::list<smIdType>& group : found) {
    std::vector<std::int64_t> ids;
    ids.reserve(group.size());
    for (const smIdType id : group) {
      ids.push_back(static_cast<std::int64_t>(id));
    }
    out.append(vector_to_array(ids));
  }
  return out;
}

py::dict Mesher::merge_equal_elements() {
  ensure_open();
  const Counts before = counts_of(*meshDS_);
  SMESH_MeshEditor editor(mesh_);
  editor.MergeEqualElements();
  publish(*meshDS_);
  const Counts after = counts_of(*meshDS_);
  const std::int64_t removed = (before.edges + before.faces + before.volumes) -
                               (after.edges + after.faces + after.volumes);
  return report(before, after, removed);
}

// ---- Smoothing ------------------------------------------------------------------------- //

py::dict Mesher::smooth(int method, int iterations, double target_aspect_ratio,
                        bool in_uv_space, const std::vector<std::int64_t>& elements,
                        const std::vector<std::int64_t>& fixed_nodes) {
  ensure_open();
  if (method != SMESH_MeshEditor::LAPLACIAN && method != SMESH_MeshEditor::CENTROIDAL) {
    throw PysmeshError("Mesher.smooth: unknown smoothing method " + std::to_string(method) +
                       ".");
  }
  if (iterations < 1) {
    throw PysmeshError("Mesher.smooth: iterations must be >= 1 (got " +
                       std::to_string(iterations) + ").");
  }
  if (!(target_aspect_ratio >= 1.0)) {
    // Aspect ratio is normalised so that a regular element is 1, so nothing below it is
    // reachable and asking for it would only mean "run every iteration".
    throw PysmeshError("Mesher.smooth: the target aspect ratio must be >= 1 (got " +
                       std::to_string(target_aspect_ratio) + ").");
  }

  TIDSortedElemSet chosen = element_set(*meshDS_, elements, "Mesher.smooth");
  std::set<const SMDS_MeshNode*> fixed;
  for (const std::int64_t id : fixed_nodes) {
    fixed.insert(node_of(*meshDS_, id, "Mesher.smooth"));
  }

  const Counts before = counts_of(*meshDS_);
  SMESH_MeshEditor editor(mesh_);
  editor.Smooth(chosen, fixed, static_cast<SMESH_MeshEditor::SmoothMethod>(method),
                iterations, target_aspect_ratio, in_uv_space);
  publish(*meshDS_);
  return report(before, counts_of(*meshDS_), 0);
}

// ---- Orientation ----------------------------------------------------------------------- //

py::dict Mesher::reorient(const std::vector<std::int64_t>& elements) {
  ensure_open();
  if (elements.empty()) {
    throw PysmeshError("Mesher.reorient: name the elements to reverse.");
  }
  std::int64_t reoriented = 0;
  SMESH_MeshEditor editor(mesh_);
  for (const std::int64_t id : elements) {
    if (editor.Reorient(element_of(*meshDS_, id, "Mesher.reorient"))) {
      ++reoriented;
    }
  }
  publish(*meshDS_);

  py::dict out;
  out["reoriented"] = reoriented;
  return out;
}

py::dict Mesher::reorient_2d(const std::vector<double>& direction,
                             const std::vector<std::int64_t>& faces,
                             const std::vector<std::int64_t>& reference_faces,
                             bool allow_non_manifold) {
  ensure_open();
  require_triple(direction, "Mesher.reorient_2d: direction");
  TIDSortedElemSet chosen =
      element_set_of_family(*meshDS_, faces, SMDSAbs_Face, "Mesher.reorient_2d");
  TIDSortedElemSet reference =
      element_set_of_family(*meshDS_, reference_faces, SMDSAbs_Face, "Mesher.reorient_2d");
  if (chosen.empty()) {
    for (SMDS_ElemIteratorPtr it = meshDS_->elementsIterator(SMDSAbs_Face); it->more();) {
      chosen.insert(it->next());
    }
  }

  SMESH_MeshEditor editor(mesh_);
  const int reoriented =
      editor.Reorient2D(chosen, gp_Vec(direction[0], direction[1], direction[2]), reference,
                        allow_non_manifold);
  publish(*meshDS_);

  py::dict out;
  out["reoriented"] = static_cast<std::int64_t>(reoriented);
  return out;
}

py::dict Mesher::reorient_2d_by_3d(const std::vector<std::int64_t>& faces,
                                   const std::vector<std::int64_t>& volumes,
                                   bool outside_normal) {
  ensure_open();
  TIDSortedElemSet chosen =
      element_set_of_family(*meshDS_, faces, SMDSAbs_Face, "Mesher.reorient_2d_by_3d");
  TIDSortedElemSet cells =
      element_set_of_family(*meshDS_, volumes, SMDSAbs_Volume, "Mesher.reorient_2d_by_3d");
  if (chosen.empty()) {
    for (SMDS_ElemIteratorPtr it = meshDS_->elementsIterator(SMDSAbs_Face); it->more();) {
      chosen.insert(it->next());
    }
  }
  if (cells.empty()) {
    for (SMDS_ElemIteratorPtr it = meshDS_->elementsIterator(SMDSAbs_Volume); it->more();) {
      cells.insert(it->next());
    }
  }
  if (cells.empty()) {
    throw PysmeshError("Mesher.reorient_2d_by_3d: the mesh has no volume cells, so there is "
                       "nothing to take the orientation from. Use reorient_2d instead.");
  }

  SMESH_MeshEditor editor(mesh_);
  const int reoriented = editor.Reorient2DBy3D(chosen, cells, outside_normal);
  publish(*meshDS_);

  py::dict out;
  out["reoriented"] = static_cast<std::int64_t>(reoriented);
  return out;
}

// ---- Face splitting and fusing ---------------------------------------------------------- //

py::dict Mesher::quad_to_tri(const std::vector<std::int64_t>& elements,
                             const std::string& criterion, const py::dict& criterion_params,
                             bool diagonal_13) {
  ensure_open();
  TIDSortedElemSet chosen =
      element_set_of_family(*meshDS_, elements, SMDSAbs_Face, "Mesher.quad_to_tri");
  if (chosen.empty()) {
    for (SMDS_ElemIteratorPtr it = meshDS_->elementsIterator(SMDSAbs_Face); it->more();) {
      chosen.insert(it->next());
    }
  }

  const Counts before = counts_of(*meshDS_);
  SMESH_MeshEditor editor(mesh_);
  bool done = false;
  if (criterion.empty()) {
    // The fixed-diagonal form: every quadrangle is cut the same way, which is what a caller
    // wants when the two halves have to line up with something else.
    done = editor.QuadToTri(chosen, diagonal_13);
  } else {
    done = editor.QuadToTri(
        chosen, criterion_of(criterion, criterion_params, this, "Mesher.quad_to_tri"));
  }
  publish(*meshDS_);
  if (!done) {
    throw PysmeshError("Mesher.quad_to_tri: SMESH could not split the faces given.");
  }
  return report(before, counts_of(*meshDS_), 0);
}

py::dict Mesher::tri_to_quad(const std::vector<std::int64_t>& elements,
                             const std::string& criterion, const py::dict& criterion_params,
                             double max_angle) {
  ensure_open();
  if (!(max_angle >= 0.0)) {
    throw PysmeshError("Mesher.tri_to_quad: the maximum angle must be >= 0 (got " +
                       std::to_string(max_angle) + ").");
  }
  TIDSortedElemSet chosen =
      element_set_of_family(*meshDS_, elements, SMDSAbs_Face, "Mesher.tri_to_quad");
  if (chosen.empty()) {
    for (SMDS_ElemIteratorPtr it = meshDS_->elementsIterator(SMDSAbs_Face); it->more();) {
      chosen.insert(it->next());
    }
  }

  const Counts before = counts_of(*meshDS_);
  SMESH_MeshEditor editor(mesh_);
  const bool done = editor.TriToQuad(
      chosen, criterion_of(criterion, criterion_params, this, "Mesher.tri_to_quad"),
      max_angle);
  publish(*meshDS_);
  if (!done) {
    throw PysmeshError("Mesher.tri_to_quad: SMESH could not fuse the faces given.");
  }
  return report(before, counts_of(*meshDS_), 0);
}

// ---- Duplication ------------------------------------------------------------------------ //

py::dict Mesher::double_elements(const std::vector<std::int64_t>& elements) {
  ensure_open();
  if (elements.empty()) {
    throw PysmeshError("Mesher.double_elements: name the elements to duplicate. Doubling the "
                       "whole mesh is never what an internal wall means.");
  }
  TIDSortedElemSet chosen = element_set(*meshDS_, elements, "Mesher.double_elements");

  const Counts before = counts_of(*meshDS_);
  SMESH_MeshEditor editor(mesh_);
  editor.DoubleElements(chosen);
  publish(*meshDS_);
  return report(before, counts_of(*meshDS_), 0);
}

// ---- Sweeps ----------------------------------------------------------------------------- //

py::dict Mesher::extrusion_sweep(const std::vector<std::int64_t>& elements,
                                 const std::vector<double>& step, int steps,
                                 bool make_boundary, double tolerance) {
  ensure_open();
  require_triple(step, "Mesher.extrusion_sweep: step");
  if (steps < 1) {
    throw PysmeshError("Mesher.extrusion_sweep: steps must be >= 1 (got " +
                       std::to_string(steps) + ").");
  }
  if (step[0] == 0.0 && step[1] == 0.0 && step[2] == 0.0) {
    throw PysmeshError("Mesher.extrusion_sweep: the step must not be the zero vector.");
  }
  if (elements.empty()) {
    throw PysmeshError("Mesher.extrusion_sweep: name the elements to sweep.");
  }

  // Upstream takes two sets and sorts out which is which itself; elements go first.
  TIDSortedElemSet sets[2];
  sets[0] = element_set(*meshDS_, elements, "Mesher.extrusion_sweep");

  const Counts before = counts_of(*meshDS_);
  SMESH_MeshEditor editor(mesh_);
  SMESH_MeshEditor::TTElemOfElemListMap history;
  const int flags = make_boundary ? SMESH_MeshEditor::EXTRUSION_FLAG_BOUNDARY : 0;
  editor.ExtrusionSweep(sets, gp_Vec(step[0], step[1], step[2]), steps, history, flags,
                        tolerance);
  publish(*meshDS_);
  return report(before, counts_of(*meshDS_), 0);
}

py::dict Mesher::rotation_sweep(const std::vector<std::int64_t>& elements,
                                const std::vector<double>& origin,
                                const std::vector<double>& direction, double angle, int steps,
                                double tolerance, bool make_walls) {
  ensure_open();
  require_triple(origin, "Mesher.rotation_sweep: axis_origin");
  require_triple(direction, "Mesher.rotation_sweep: axis_direction");
  if (steps < 1) {
    throw PysmeshError("Mesher.rotation_sweep: steps must be >= 1 (got " +
                       std::to_string(steps) + ").");
  }
  if (direction[0] == 0.0 && direction[1] == 0.0 && direction[2] == 0.0) {
    throw PysmeshError("Mesher.rotation_sweep: the axis direction must not be the zero "
                       "vector.");
  }
  if (elements.empty()) {
    throw PysmeshError("Mesher.rotation_sweep: name the elements to sweep.");
  }

  TIDSortedElemSet sets[2];
  sets[0] = element_set(*meshDS_, elements, "Mesher.rotation_sweep");

  const Counts before = counts_of(*meshDS_);
  SMESH_MeshEditor editor(mesh_);
  editor.RotationSweep(sets,
                       gp_Ax1(gp_Pnt(origin[0], origin[1], origin[2]),
                              gp_Dir(direction[0], direction[1], direction[2])),
                       angle, steps, tolerance, /*theMakeGroups=*/false, make_walls);
  publish(*meshDS_);
  return report(before, counts_of(*meshDS_), 0);
}

// ---- Surface offset --------------------------------------------------------------------- //

py::dict Mesher::offset(double value, const std::vector<std::int64_t>& elements,
                        bool copy_elements, bool fix_self_intersection) {
  ensure_open();
  TIDSortedElemSet chosen =
      element_set_of_family(*meshDS_, elements, SMDSAbs_Face, "Mesher.offset");

  const Counts before = counts_of(*meshDS_);
  SMESH_MeshEditor editor(mesh_);
  SMESH_MeshEditor::PGroupIDs made;
  try {
    // The target is this mesh, always: upstream dereferences the target without checking it,
    // so the null that would naturally mean "in place" is a crash rather than a mode.
    made = editor.Offset(chosen, value, mesh_, /*theMakeGroups=*/false, copy_elements,
                         fix_self_intersection);
  } catch (const PysmeshError&) {
    throw;
  } catch (const std::exception& failure) {
    // The offset refuses a mesh it cannot handle by throwing, and it tests the *whole* source
    // mesh rather than the faces it was given.
    throw PysmeshError(std::string("Mesher.offset: ") + failure.what());
  }
  publish(*meshDS_);
  if (!made) {
    throw PysmeshError("Mesher.offset: the offset produced no elements. The source faces must "
                       "be linear triangles, and the offset distance must not collapse them.");
  }
  return report(before, counts_of(*meshDS_), 0);
}

// ---- Sewing ----------------------------------------------------------------------------- //

py::dict Mesher::sew_free_border(const std::vector<std::int64_t>& border,
                                 const std::vector<std::int64_t>& side,
                                 bool side_is_free_border, bool create_polygons,
                                 bool create_polyhedra) {
  ensure_open();
  if (border.size() != 3) {
    throw PysmeshError("Mesher.sew_free_border: the border must be named by three node ids — "
                       "its first node, the node next to it, and its last node.");
  }
  if (side.size() != 2 && side.size() != 3) {
    throw PysmeshError("Mesher.sew_free_border: the side must be named by two or three node "
                       "ids.");
  }

  const char* owner = "Mesher.sew_free_border";
  const SMDS_MeshNode* b1 = node_of(*meshDS_, border[0], owner);
  const SMDS_MeshNode* b2 = node_of(*meshDS_, border[1], owner);
  const SMDS_MeshNode* b3 = node_of(*meshDS_, border[2], owner);
  const SMDS_MeshNode* s1 = node_of(*meshDS_, side[0], owner);
  const SMDS_MeshNode* s2 = node_of(*meshDS_, side[1], owner);
  const SMDS_MeshNode* s3 =
      side.size() == 3 ? node_of(*meshDS_, side[2], owner) : nullptr;

  const Counts before = counts_of(*meshDS_);
  SMESH_MeshEditor editor(mesh_);
  const int code = editor.SewFreeBorder(b1, b2, b3, s1, s2, s3, side_is_free_border,
                                        create_polygons, create_polyhedra);
  publish(*meshDS_);
  if (code != SMESH_MeshEditor::SEW_OK) {
    throw PysmeshError(std::string(owner) + ": " + sew_error_text(code) + ".");
  }
  return report(before, counts_of(*meshDS_), 0);
}

py::dict Mesher::sew_side_elements(const std::vector<std::int64_t>& side1,
                                   const std::vector<std::int64_t>& side2,
                                   const std::vector<std::int64_t>& first_nodes,
                                   const std::vector<std::int64_t>& second_nodes) {
  ensure_open();
  const char* owner = "Mesher.sew_side_elements";
  if (side1.empty() || side2.empty()) {
    throw PysmeshError(std::string(owner) + ": both sides must name their elements.");
  }
  if (first_nodes.size() != 2 || second_nodes.size() != 2) {
    throw PysmeshError(std::string(owner) +
                       ": first_nodes and second_nodes must each be one node from side 1 and "
                       "its counterpart on side 2.");
  }

  TIDSortedElemSet set1 = element_set(*meshDS_, side1, owner);
  TIDSortedElemSet set2 = element_set(*meshDS_, side2, owner);
  const SMDS_MeshNode* f1 = node_of(*meshDS_, first_nodes[0], owner);
  const SMDS_MeshNode* f2 = node_of(*meshDS_, first_nodes[1], owner);
  const SMDS_MeshNode* g1 = node_of(*meshDS_, second_nodes[0], owner);
  const SMDS_MeshNode* g2 = node_of(*meshDS_, second_nodes[1], owner);

  const Counts before = counts_of(*meshDS_);
  SMESH_MeshEditor editor(mesh_);
  const int code = editor.SewSideElements(set1, set2, f1, f2, g1, g2);
  publish(*meshDS_);
  if (code != SMESH_MeshEditor::SEW_OK) {
    throw PysmeshError(std::string(owner) + ": " + sew_error_text(code) + ".");
  }
  return report(before, counts_of(*meshDS_), 0);
}

}  // namespace mesher
}  // namespace pysmesh
