// pySMESH binding — named element and node groups.
//
// A group is a named set of mesh entities that the *mesher* maintains, rather than one a
// consumer re-derives after the fact. That is the whole of its value: SMESH's own editor
// calls AddToSameGroups and ReplaceElemInGroups as it works, so a group defined on a coarse
// mesh still names the right cells after the mesh has been converted to second order, split
// or merged. Re-deriving membership from geometry after each of those steps is the thing this
// replaces, and it is the step that goes wrong.
//
// Three kinds exist upstream and all three are bound, because they differ in what maintains
// the membership and a caller has to know which one it holds:
//
//   * **explicit** — an id list, carried through editing by SMESH itself;
//   * **on a sub-shape** — everything the mesher bound to that face, edge or solid, so it
//     follows a re-compute rather than a snapshot of one;
//   * **on a filter** — everything a predicate accepts, re-evaluated when the mesh changes.
//
// Only the first can be edited by hand; the other two are defined by their source and say so
// rather than silently ignoring an add.
//
// Names are the key here, and upstream's are not: SMESH allows two groups of one name and
// addresses them by an integer id it also uses for persistence. A duplicate name is refused
// at creation so that every later call means exactly one group.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <SMDS_ElemIterator.hxx>
#include <SMDS_MeshElement.hxx>
#include <SMESHDS_Group.hxx>
#include <SMESHDS_GroupBase.hxx>
#include <SMESHDS_GroupOnFilter.hxx>
#include <SMESHDS_GroupOnGeom.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_Group.hxx>
#include <SMESH_Mesh.hxx>

namespace pysmesh {
namespace mesher {
namespace {

// What maintains a group's membership. Reported alongside the members because an add to a
// group defined by a shape or a filter is refused, and a caller needs to know before trying.
constexpr int kSourceExplicit = 0;
constexpr int kSourceShape = 1;
constexpr int kSourceFilter = 2;

int source_of(const SMESHDS_GroupBase* group) {
  if (dynamic_cast<const SMESHDS_GroupOnGeom*>(group) != nullptr) {
    return kSourceShape;
  }
  if (dynamic_cast<const SMESHDS_GroupOnFilter*>(group) != nullptr) {
    return kSourceFilter;
  }
  return kSourceExplicit;
}

SMDSAbs_ElementType family_of(int code) {
  if (code < 0 || code > static_cast<int>(SMDSAbs_Ball)) {
    throw PysmeshError("Unknown element family " + std::to_string(code) +
                       " (expected one of ALL, NODE, EDGE, FACE, VOLUME, ELEM_0D, BALL).");
  }
  return static_cast<SMDSAbs_ElementType>(code);
}

}  // namespace

SMESHDS_GroupBase* Mesher::group_ds(const std::string& name) const {
  ensure_open();
  for (SMESHDS_GroupBase* group : meshDS().GetGroups()) {
    if (group != nullptr && name == group->GetStoreName()) {
      return group;
    }
  }
  return nullptr;
}

namespace {

// Create the SMESH_Group and give it its stored name, refusing a name already in use. The
// SMESH_Mesh owns the result and frees it; nothing here takes ownership.
SMESH_Group& create_group(Mesher& mesher, const std::string& name, int family,
                          const TopoDS_Shape& shape,
                          const SMESH::Controls::PredicatePtr& predicate) {
  if (name.empty()) {
    throw PysmeshError("Mesher.add_group: a group needs a name.");
  }
  if (mesher.group_ds(name) != nullptr) {
    throw PysmeshError("Mesher.add_group: a group named '" + name +
                       "' already exists. Names address a group here, so they have to be "
                       "unique.");
  }
  SMESH_Group* group =
      mesher.smesh().AddGroup(family_of(family), name.c_str(), -1, shape, predicate);
  if (group == nullptr || group->GetGroupDS() == nullptr) {
    throw PysmeshError("Mesher.add_group: SMESH refused to create the group '" + name + "'.");
  }
  group->GetGroupDS()->SetStoreName(name.c_str());
  return *group;
}

SMESHDS_Group& explicit_group(const Mesher& mesher, const std::string& name,
                              const char* operation) {
  SMESHDS_GroupBase* group = mesher.group_ds(name);
  if (group == nullptr) {
    throw PysmeshError(std::string("Mesher.") + operation + ": the mesh has no group named '" +
                       name + "'.");
  }
  SMESHDS_Group* explicit_ds = dynamic_cast<SMESHDS_Group*>(group);
  if (explicit_ds == nullptr) {
    throw PysmeshError(
        std::string("Mesher.") + operation + ": the group '" + name +
        "' is defined by " +
        (source_of(group) == kSourceShape ? "a sub-shape" : "a filter") +
        ", so its membership follows that source and cannot be edited by hand.");
  }
  return *explicit_ds;
}

}  // namespace

void Mesher::add_group(const std::string& name, int family,
                       const std::vector<std::int64_t>& ids) {
  ensure_open();
  SMESH_Group& group = create_group(*this, name, family, TopoDS_Shape(),
                                    SMESH::Controls::PredicatePtr());
  SMESHDS_Group* group_ds = dynamic_cast<SMESHDS_Group*>(group.GetGroupDS());
  if (group_ds == nullptr) {
    throw PysmeshError("Mesher.add_group: the new group '" + name + "' is not an id list.");
  }
  for (const std::int64_t id : ids) {
    if (!group_ds->Add(static_cast<smIdType>(id))) {
      throw PysmeshError("Mesher.add_group: '" + name + "' cannot hold " + std::to_string(id) +
                         " — the mesh has no entity of that id in this family.");
    }
  }
}

void Mesher::add_group_on_shape(const std::string& name, int family, const std::string& kind,
                                int ordinal) {
  ensure_open();
  const TopoDS_Shape& target = sub_shape(kind, ordinal);  // validates kind and ordinal
  create_group(*this, name, family, target, SMESH::Controls::PredicatePtr());
}

void Mesher::add_group_on_filter(const std::string& name, int family,
                                 const std::string& predicate, const py::dict& params) {
  ensure_open();
  SMESH::Controls::PredicatePtr built = build_predicate(predicate, params, this);
  if (!built) {
    throw PysmeshError("Mesher.add_group_on_filter: unknown predicate '" + predicate + "'.");
  }
  create_group(*this, name, family, TopoDS_Shape(), built);
}

void Mesher::remove_group(const std::string& name) {
  ensure_open();
  SMESHDS_GroupBase* group = group_ds(name);
  if (group == nullptr) {
    throw PysmeshError("Mesher.remove_group: the mesh has no group named '" + name + "'.");
  }
  if (!mesh_->RemoveGroup(group->GetID())) {
    throw PysmeshError("Mesher.remove_group: SMESH refused to remove the group '" + name +
                       "'.");
  }
}

void Mesher::edit_group(const std::string& name, const std::vector<std::int64_t>& ids,
                        bool add) {
  ensure_open();
  SMESHDS_Group& group = explicit_group(*this, name, add ? "add_to_group"
                                                         : "remove_from_group");
  for (const std::int64_t id : ids) {
    const bool ok = add ? group.Add(static_cast<smIdType>(id))
                        : group.Remove(static_cast<smIdType>(id));
    if (!ok) {
      throw PysmeshError(
          std::string("Mesher.") + (add ? "add_to_group" : "remove_from_group") +
          ": the group '" + name + "' " + (add ? "cannot hold " : "does not hold ") +
          std::to_string(id) + ".");
    }
  }
}

py::list harvest_groups(const SMESHDS_Mesh& ds) {
  py::list out;
  for (const SMESHDS_GroupBase* group : ds.GetGroups()) {
    if (group == nullptr) {
      continue;
    }
    // Extent() is a count on an explicit group and a full re-evaluation on a filtered one,
    // so it is read once and only as a reservation hint.
    std::vector<std::int64_t> ids;
    ids.reserve(static_cast<std::size_t>(group->Extent()));
    for (SMDS_ElemIteratorPtr it = group->GetElements(); it->more();) {
      ids.push_back(static_cast<std::int64_t>(it->next()->GetID()));
    }
    out.append(py::make_tuple(std::string(group->GetStoreName()),
                              static_cast<int>(group->GetType()), source_of(group),
                              vector_to_array(ids)));
  }
  return out;
}

py::list Mesher::groups() const { return harvest_groups(meshDS()); }

}  // namespace mesher
}  // namespace pysmesh
