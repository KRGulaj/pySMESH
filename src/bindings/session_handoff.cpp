// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-06

// pySMESH binding — Session: the export to a mesher, and the id-to-ordinal bijection.
//
// The session is the CAD authority; a mesher is a consumer of it. The boundary between them
// is crossed once, at the meshing handoff, on a shape nobody is editing — not per operation.
// What crosses is BREP bytes plus one thing the bytes cannot carry: which entity id each
// sub-shape of them is.
//
// The pairing is positional and never geometric. Matching by centroid is the obvious
// shortcut and it is wrong by construction, not merely imprecise: a pipe's inner and outer
// cylindrical walls have the same centroid — measured at 6e-17 apart on a plain two-cylinder
// pipe — so a centroid-keyed map collides on one of the most ordinary features in CAD, and
// collides silently, mapping two different faces to one another's names. Ordinals cannot
// collide, and they are reproducible: writing a 28 255-entity assembly and reading it back
// preserved every ordinal of every kind, with the traversal itself verified to repeat.
//
// What can still go wrong is the session's own state, and that is what this file checks. The
// id-to-shape relation is not a bijection in general:
//
//   * a same-domain merge leaves SEVERAL live ids on ONE face;
//   * a split leaves ONE live id on SEVERAL faces.
//
// Both are correct session states and both make "this id is that tag" ambiguous. The export
// refuses them, naming the ids, rather than handing over a map that quietly drops some of
// them — which is the failure a handoff is least able to detect downstream.
//
// See session/session.hpp for the file split.

#include "session/session.hpp"

namespace pysmesh {
namespace session {

namespace {

// The label of one exported sub-shape, and the diagnostics for the two ways it can fail.
struct KindManifest {
  std::vector<EntityId> ids;        // one per ordinal, in traversal order
  std::vector<EntityId> ambiguous;  // ids sharing a shape with another id (a merge)
  std::vector<EntityId> split;      // ids denoting more than one exported shape (a split)
  std::vector<int> unlabelled;      // ordinals of shapes carrying no live id at all
};

}  // namespace

py::dict Session::export_handoff() const {
  const TopoDS_Shape root = state_.root;

  py::dict out;
  std::vector<EntityId> ambiguous;
  std::vector<EntityId> split;
  std::vector<std::string> unlabelled;

  for (TopAbs_ShapeEnum kind : kEntityKinds) {
    ShapeSet shapes;
    TopExp::MapShapes(root, kind, shapes);

    KindManifest m;
    m.ids.reserve(static_cast<std::size_t>(shapes.Extent()));

    // How many exported shapes each id lands on. An id that lands on two is a split, and it
    // is found by counting rather than by asking the registry, so the check reads the same
    // relation the consumer will.
    std::unordered_map<EntityId, int> hits;

    for (int i = 1; i <= shapes.Extent(); ++i) {
      const auto it = state_.registry->by_shape.find(shapes.FindKey(i));
      if (it == state_.registry->by_shape.end() || it->second.empty()) {
        m.unlabelled.push_back(i);
        m.ids.push_back(0);
        continue;
      }
      // by_shape holds every live id on this shape. More than one is a merge, and the
      // handoff cannot choose between them: both names are alive and both mean this face.
      if (it->second.size() > 1) {
        for (EntityId id : it->second) {
          m.ambiguous.push_back(id);
        }
      }
      const EntityId id = it->second.front();
      m.ids.push_back(id);
      ++hits[id];
    }

    for (const auto& entry : hits) {
      if (entry.second > 1) {
        m.split.push_back(entry.first);
      }
    }

    for (EntityId id : m.ambiguous) {
      ambiguous.push_back(id);
    }
    for (EntityId id : m.split) {
      split.push_back(id);
    }
    for (int ordinal : m.unlabelled) {
      unlabelled.push_back(std::string(kind_name(kind)) + " #" + std::to_string(ordinal));
    }

    const std::string key = std::string(kind_name(kind)) + "_id";
    out[py::str(key)] = ids_array(m.ids);
  }

  auto tidy = [](std::vector<EntityId>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  };
  tidy(ambiguous);
  tidy(split);

  if (!ambiguous.empty() || !split.empty()) {
    std::ostringstream detail;
    if (!ambiguous.empty()) {
      detail << ambiguous.size()
             << " id(s) share a sub-shape with another id, which a same-domain merge "
                "produces: the merged entity is denoted by all of them and the handoff "
                "cannot choose one. ";
    }
    if (!split.empty()) {
      detail << split.size()
             << " id(s) denote more than one sub-shape, which a split produces: the "
                "entity is no longer one thing to name. ";
    }
    detail << "Resolve the ambiguity before handing off — a merge is settled by exporting "
              "after the ids the caller no longer needs have been dropped, a split by "
              "treating the pieces as the new entities they are.";
    std::vector<EntityId> blamed = ambiguous;
    blamed.insert(blamed.end(), split.begin(), split.end());
    throw PysmeshError(
        "Session.export_handoff: the entity id to sub-shape map is not a bijection, so the "
        "handoff would silently mis-name entities.",
        detail.str(), ids_as_int(blamed));
  }

  if (!unlabelled.empty()) {
    std::ostringstream detail;
    detail << "First: " << unlabelled.front() << ". A sub-shape of the root with no live id "
           << "means the registry and the root have diverged; this is a bug, not a caller "
           << "error.";
    throw PysmeshError("Session.export_handoff: " + std::to_string(unlabelled.size()) +
                           " sub-shape(s) of the session root carry no entity id.",
                       detail.str());
  }

  // Written last, so a session that fails the bijection check pays nothing for the bytes.
  std::ostringstream stream;
  {
    py::gil_scoped_release release;
    try {
      BRepTools::Write(root, stream);
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.export_handoff: BREP write failed: ") +
                         e.what());
    }
  }

  out["brep"] = py::bytes(stream.str());
  return out;
}

}  // namespace session
}  // namespace pysmesh
