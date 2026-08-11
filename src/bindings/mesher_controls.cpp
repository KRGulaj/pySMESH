// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — the mesh quality controls: numerical functors, predicates, filter algebra.
//
// SMESH's controls are the part of this library with no array-side equivalent anywhere in the
// consuming stack, and the reason is dimensional: every 3-D measure here — a cell volume, a
// 3-D aspect ratio, a badly oriented or over-constrained volume, a bare border — has to see
// the whole in-core element graph with its reverse connectivity, which is exactly what a
// streamed surface-triangle pipeline does not have. These are also the numbers that decide
// whether a volume mesh will run in a solver at all.
//
// Two shapes in the upstream API decide how this file is written, both measured rather than
// assumed:
//
//   * **A concrete functor hides NumericalFunctor::GetValue(long).** Most declare
//     `GetValue(const TSequenceOfXYZ&)`, which hides the id-taking base overload by name. So
//     every value is read through a `NumericalFunctor&`. Passing a point sequence instead —
//     the tempting fix for the compile error — computes something different, because several
//     controls take a fast path in the id-taking overload.
//   * **Predicate is a *virtual* base of every concrete predicate**, so downcasting from a
//     PredicatePtr is ill-formed. Every predicate here is therefore configured through its
//     concrete type first and only then handed to the shared_ptr that owns it.
//
// A control is named by a string and parameterised by a dict, the same boundary the algorithm
// catalogue uses and for the same reason: the typed surface is the frozen dataclass on the
// Python side, and `Params::done()` refuses any field a branch here did not read, so the two
// cannot drift apart without failing on the first call.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <SMDS_ElemIterator.hxx>
#include <SMDS_Mesh.hxx>
#include <SMDS_MeshElement.hxx>
#include <SMDS_MeshNode.hxx>
#include <SMESHDS_GroupBase.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_ControlsDef.hxx>

namespace pysmesh {
namespace mesher {
namespace {

namespace ctl = SMESH::Controls;

const Mesher& with_geometry(const Mesher* owner, const std::string& name) {
  if (owner == nullptr) {
    throw PysmeshError("The control '" + name +
                       "' reads the geometry the mesh was built on, so it can only be "
                       "evaluated on a mesher. A mesh handed in as arrays carries no shape.");
  }
  return *owner;
}

// Walk the elements a control applies to. SMDS keeps nodes in their own container, so a
// control whose type is Node — a free-node test, a node connectivity count — needs the node
// iterator and not the element one.
template <class Visit>
void walk(const SMDS_Mesh& mesh, SMDSAbs_ElementType type, Visit visit) {
  if (type == SMDSAbs_Node) {
    for (SMDS_NodeIteratorPtr it = mesh.nodesIterator(); it->more();) {
      visit(static_cast<const SMDS_MeshElement*>(it->next()));
    }
    return;
  }
  for (SMDS_ElemIteratorPtr it = mesh.elementsIterator(type); it->more();) {
    visit(it->next());
  }
}

}  // namespace

// ---- The numerical controls ------------------------------------------------------------ //

ctl::NumericalFunctorPtr build_functor(const std::string& name, const py::dict& values,
                                       const Mesher* owner) {
  Params p(name.c_str(), values);
  ctl::NumericalFunctorPtr functor;

  if (name == "Volume") {
    functor.reset(new ctl::Volume());
  } else if (name == "Area") {
    functor.reset(new ctl::Area());
  } else if (name == "Length") {
    functor.reset(new ctl::Length());
  } else if (name == "AspectRatio") {
    functor.reset(new ctl::AspectRatio());
  } else if (name == "AspectRatio3D") {
    functor.reset(new ctl::AspectRatio3D());
  } else if (name == "Warping") {
    functor.reset(new ctl::Warping());
  } else if (name == "Taper") {
    functor.reset(new ctl::Taper());
  } else if (name == "Skew") {
    functor.reset(new ctl::Skew());
  } else if (name == "MinimumAngle") {
    functor.reset(new ctl::MinimumAngle());
  } else if (name == "Length2D") {
    functor.reset(new ctl::Length2D());
  } else if (name == "Length3D") {
    functor.reset(new ctl::Length3D());
  } else if (name == "Deflection2D") {
    // Reads the CAD surface each face lies on, through the mesh's own shape index.
    with_geometry(owner, name);
    functor.reset(new ctl::Deflection2D());
  } else if (name == "MaxElementLength2D") {
    functor.reset(new ctl::MaxElementLength2D());
  } else if (name == "MaxElementLength3D") {
    functor.reset(new ctl::MaxElementLength3D());
  } else if (name == "MultiConnection") {
    functor.reset(new ctl::MultiConnection());
  } else if (name == "MultiConnection2D") {
    functor.reset(new ctl::MultiConnection2D());
  } else if (name == "NodeConnectivityNumber") {
    functor.reset(new ctl::NodeConnectivityNumber());
  } else {
    return ctl::NumericalFunctorPtr();
  }

  p.done();
  return functor;
}

// ---- The predicates and the filter algebra ---------------------------------------------- //

ctl::PredicatePtr build_predicate(const std::string& name, const py::dict& values,
                                  const Mesher* owner) {
  Params p(name.c_str(), values);
  ctl::PredicatePtr predicate;

  // Every branch below builds the concrete type, configures it, and only then hands it to
  // the shared_ptr — Predicate being a virtual base makes the other order ill-formed.
  if (name == "FreeEdges") {
    predicate.reset(new ctl::FreeEdges());
  } else if (name == "FreeBorders") {
    predicate.reset(new ctl::FreeBorders());
  } else if (name == "FreeNodes") {
    predicate.reset(new ctl::FreeNodes());
  } else if (name == "FreeFaces") {
    predicate.reset(new ctl::FreeFaces());
  } else if (name == "BadOrientedVolume") {
    predicate.reset(new ctl::BadOrientedVolume());
  } else if (name == "BareBorderFace") {
    predicate.reset(new ctl::BareBorderFace());
  } else if (name == "BareBorderVolume") {
    predicate.reset(new ctl::BareBorderVolume());
  } else if (name == "OverConstrainedFace") {
    predicate.reset(new ctl::OverConstrainedFace());
  } else if (name == "OverConstrainedVolume") {
    predicate.reset(new ctl::OverConstrainedVolume());
  } else if (name == "CoincidentNodes") {
    ctl::CoincidentNodes* concrete = new ctl::CoincidentNodes();
    predicate.reset(concrete);
    concrete->SetTolerance(p.number("tolerance"));
  } else if (name == "CoincidentElements") {
    const SMDSAbs_ElementType family = family_of(p.integer("element_family"));
    switch (family) {
      case SMDSAbs_Edge:
        predicate.reset(new ctl::CoincidentElements1D());
        break;
      case SMDSAbs_Face:
        predicate.reset(new ctl::CoincidentElements2D());
        break;
      case SMDSAbs_Volume:
        predicate.reset(new ctl::CoincidentElements3D());
        break;
      default:
        throw PysmeshError("CoincidentElements is defined for the EDGE, FACE and VOLUME "
                           "families only.");
    }
  } else if (name == "ManifoldPart") {
    ctl::ManifoldPart* concrete = new ctl::ManifoldPart();
    predicate.reset(concrete);
    concrete->SetAngleTolerance(p.number("angle_tolerance"));
    concrete->SetIsOnlyManifold(p.flag("only_manifold"));
    concrete->SetStartElem(static_cast<long>(p.integer("start_element")));
  } else if (name == "RangeOfIds") {
    // Built from real ids rather than from a per-type ordinal, because SMDS element ids are
    // one global sequence shared by edges, faces and volumes: a range written as "the first
    // five" would select faces on a mesh whose volumes follow them.
    ctl::RangeOfIds* concrete = new ctl::RangeOfIds();
    predicate.reset(concrete);
    concrete->SetType(family_of(p.integer("element_family")));
    for (const std::int64_t id : p.ids("ids")) {
      concrete->AddToRange(static_cast<long>(id));
    }
  } else if (name == "ElementsOnShape") {
    const Mesher& mesher = with_geometry(owner, name);
    ctl::ElementsOnShape* concrete = new ctl::ElementsOnShape();
    predicate.reset(concrete);
    concrete->SetTolerance(p.number("tolerance"));
    concrete->SetAllNodes(p.flag("all_nodes"));
    const std::pair<std::string, int> on = p.subshape("on");
    const SMDSAbs_ElementType family = family_of(p.integer("element_family"));
    concrete->SetShape(on.second > 0 ? mesher.sub_shape(on.first, on.second)
                                     : mesher.shape_data().shape,
                       family);
  } else if (name == "BelongToMeshGroup") {
    const Mesher& mesher = with_geometry(owner, name);
    const std::string group_name = p.text("group_name");
    SMESHDS_GroupBase* group = mesher.group_ds(group_name);
    if (group == nullptr) {
      throw PysmeshError("BelongToMeshGroup: the mesh has no group named '" + group_name +
                         "'.");
    }
    ctl::BelongToMeshGroup* concrete = new ctl::BelongToMeshGroup();
    predicate.reset(concrete);
    concrete->SetGroup(group);
    // The name as well as the pointer: SetMesh drops the group whenever it is handed a
    // different mesh, and the name is the only way it can find the group again.
    concrete->SetStoreName(group_name);
  } else if (name == "LogicalNOT") {
    ctl::LogicalNOT* concrete = new ctl::LogicalNOT();
    predicate.reset(concrete);
    const py::dict spec = p.nested("predicate");
    concrete->SetPredicate(
        build_predicate(spec["name"].cast<std::string>(), spec["params"].cast<py::dict>(),
                        owner));
  } else if (name == "LogicalAND" || name == "LogicalOR") {
    ctl::LogicalBinary* concrete = nullptr;
    if (name == "LogicalAND") {
      concrete = new ctl::LogicalAND();
    } else {
      concrete = new ctl::LogicalOR();
    }
    predicate.reset(concrete);
    const py::dict first = p.nested("predicate1");
    const py::dict second = p.nested("predicate2");
    concrete->SetPredicate1(
        build_predicate(first["name"].cast<std::string>(), first["params"].cast<py::dict>(),
                        owner));
    concrete->SetPredicate2(
        build_predicate(second["name"].cast<std::string>(), second["params"].cast<py::dict>(),
                        owner));
  } else if (name == "LessThan" || name == "MoreThan" || name == "EqualTo") {
    ctl::Comparator* concrete = nullptr;
    ctl::EqualTo* equal = nullptr;
    if (name == "LessThan") {
      concrete = new ctl::LessThan();
    } else if (name == "MoreThan") {
      concrete = new ctl::MoreThan();
    } else {
      equal = new ctl::EqualTo();
      concrete = equal;
    }
    predicate.reset(concrete);
    const py::dict spec = p.nested("control");
    const std::string control = spec["name"].cast<std::string>();
    ctl::NumericalFunctorPtr functor =
        build_functor(control, spec["params"].cast<py::dict>(), owner);
    if (!functor) {
      throw PysmeshError("Unknown quality control '" + control + "' to compare against.");
    }
    concrete->SetNumFunctor(functor);
    concrete->SetMargin(p.number("margin"));
    if (equal != nullptr) {
      equal->SetTolerance(p.number("tolerance"));
    }
  } else {
    return ctl::PredicatePtr();
  }

  p.done();
  return predicate;
}

// ---- Evaluation ------------------------------------------------------------------------- //

py::dict evaluate_quality(const SMDS_Mesh& mesh, const Mesher* owner, const std::string& name,
                          const py::dict& params) {
  ctl::NumericalFunctorPtr functor = build_functor(name, params, owner);
  if (!functor) {
    throw PysmeshError("Unknown quality control '" + name + "'.");
  }
  functor->SetMesh(&mesh);

  std::vector<std::int64_t> ids;
  std::vector<double> values;
  std::int64_t skipped = 0;
  // A control that does not apply to an element is skipped rather than given a made-up
  // number: a warping is undefined on a triangle, an aspect ratio on a polygon, and a zero
  // there would read as a perfect element.
  walk(mesh, functor->GetType(), [&](const SMDS_MeshElement* element) {
    if (!functor->IsApplicable(element)) {
      ++skipped;
      return;
    }
    ids.push_back(static_cast<std::int64_t>(element->GetID()));
    values.push_back(functor->GetValue(static_cast<long>(element->GetID())));
  });

  py::dict out;
  out["control"] = name;
  out["family"] = static_cast<int>(functor->GetType());
  out["element_ids"] = vector_to_array(ids);
  out["values"] = vector_to_array(values);
  out["skipped"] = skipped;
  return out;
}

py::dict evaluate_selection(const SMDS_Mesh& mesh, const Mesher* owner,
                            const std::string& name, const py::dict& params) {
  ctl::PredicatePtr predicate = build_predicate(name, params, owner);
  if (!predicate) {
    throw PysmeshError("Unknown predicate '" + name + "'.");
  }
  predicate->SetMesh(&mesh);

  const SMDSAbs_ElementType family = predicate->GetType();
  std::vector<std::int64_t> ids;
  walk(mesh, family, [&](const SMDS_MeshElement* element) {
    if (predicate->IsSatisfy(static_cast<long>(element->GetID()))) {
      ids.push_back(static_cast<std::int64_t>(element->GetID()));
    }
  });

  py::dict out;
  out["predicate"] = name;
  out["family"] = static_cast<int>(family);
  out["ids"] = vector_to_array(ids);
  return out;
}

py::dict Mesher::quality(const std::string& name, const py::dict& params) const {
  return evaluate_quality(meshDS(), this, name, params);
}

py::dict Mesher::select(const std::string& name, const py::dict& params) const {
  return evaluate_selection(meshDS(), this, name, params);
}

py::dict mesh_quality(const py::dict& mesh, const std::string& name, const py::dict& params) {
  ScratchMesh scratch;
  rebuild_mesh(scratch.ds(), mesh);
  return evaluate_quality(scratch.ds(), nullptr, name, params);
}

py::dict mesh_select(const py::dict& mesh, const std::string& name, const py::dict& params) {
  ScratchMesh scratch;
  rebuild_mesh(scratch.ds(), mesh);
  return evaluate_selection(scratch.ds(), nullptr, name, params);
}

}  // namespace mesher
}  // namespace pysmesh
