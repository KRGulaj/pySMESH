// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-06

// pySMESH binding — Session: transforms and copy.
//
// Two paths, and which one an operation takes is decided by the transform itself rather than
// by its name. A direct isometry — a translation, a rotation, a mirror about an axis — is a
// change of Location alone: the TShape survives, so identity is structural and the
// relocation path asserts that structure per sub-shape. Anything else — a plane mirror, a
// scaling — is a genuine rebuild, and identity is carried by the transform's own history.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include <GCPnts_AbscissaPoint.hxx>
#include <GeomAdaptor_Curve.hxx>
#include <GeomFill_CorrectedFrenet.hxx>
#include <GeomFill_CurveAndTrihedron.hxx>

#include <algorithm>
#include <cctype>
#include <limits>

namespace pysmesh {
namespace session {

// ---- transforms ------------------------------------------------------------------- //

py::dict Session::translate(double dx, double dy, double dz,
                     const std::optional<std::vector<EntityId>>& entity_ids) {
  finite_arg("translate", "offset", dx, dy, dz);
  OpGuard guard(in_op_);
  gp_Trsf t;
  t.SetTranslation(gp_Vec(dx, dy, dz));
  return apply_trsf(t, entity_ids, "translate");
}

py::dict Session::rotate(double ox, double oy, double oz, double ax, double ay, double az,
                  double angle_rad, const std::optional<std::vector<EntityId>>& entity_ids) {
  finite_arg("rotate", "origin", ox, oy, oz);
  finite_arg("rotate", "axis", ax, ay, az);
  finite_arg("rotate", "angle_rad", angle_rad);
  OpGuard guard(in_op_);
  const gp_Vec axis(ax, ay, az);
  if (axis.Magnitude() <= 0.0) {
    throw PysmeshError("Session.rotate: axis must be a non-zero vector.");
  }
  gp_Trsf t;
  t.SetRotation(gp_Ax1(gp_Pnt(ox, oy, oz), gp_Dir(axis)), angle_rad);
  return apply_trsf(t, entity_ids, "rotate");
}

// Reflection in the plane through `point` with the given normal. A plane mirror has
// determinant -1, so OCCT rebuilds rather than relocating; ids are carried by the
// transform's own history and all of them survive.
py::dict Session::mirror(double px, double py_, double pz, double nx, double ny, double nz,
                  const std::optional<std::vector<EntityId>>& entity_ids) {
  finite_arg("mirror", "point", px, py_, pz);
  finite_arg("mirror", "normal", nx, ny, nz);
  OpGuard guard(in_op_);
  const gp_Dir normal = direction_of("mirror", "normal", nx, ny, nz);
  gp_Trsf t;
  t.SetMirror(gp_Ax2(gp_Pnt(px, py_, pz), normal));
  return apply_trsf(t, entity_ids, "mirror");
}

// Scale about `centre`. A uniform factor stays a gp_Trsf, which keeps analytic surfaces
// analytic; an anisotropic one needs gp_GTrsf, and OCCT then re-approximates every
// non-planar surface as a B-spline.
py::dict Session::scale(double sx, double sy, double sz, double cx, double cy, double cz,
                 const std::optional<std::vector<EntityId>>& entity_ids) {
  finite_arg("scale", "factors", sx, sy, sz);
  finite_arg("scale", "centre", cx, cy, cz);
  OpGuard guard(in_op_);
  require_positive("scale x", sx);
  require_positive("scale y", sy);
  require_positive("scale z", sz);
  const gp_Pnt centre(cx, cy, cz);
  if (sx == sy && sy == sz) {
    if (std::abs(sx - 1.0) <= 1e-15) {
      throw PysmeshError(
          "Session.scale: a uniform factor of 1 is a no-op; it would still consume an "
          "operation index. Skip the call instead.");
    }
    gp_Trsf t;
    t.SetScale(centre, sx);
    return apply_trsf(t, entity_ids, "scale");
  }
  gp_GTrsf g;
  g.SetVectorialPart(gp_Mat(sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, sz));
  g.SetTranslationPart(gp_XYZ((1.0 - sx) * cx, (1.0 - sy) * cy, (1.0 - sz) * cz));
  return rebuild_moved(entity_ids, "scale",
                       [&g](const TopoDS_Shape& input, TopoDS_Shape& out,
                            Handle(BRepTools_History) & hist) {
                         BRepBuilderAPI_GTransform mk(g);
                         mk.Perform(input, /*Copy=*/true);
                         if (mk.IsDone()) {
                           out = mk.Shape();
                           hist = history_of(input, mk);
                         }
                       });
}

namespace {

// The relative tolerance the path's length is integrated to, as point_at_length does it.
constexpr double kPathLengthRelTol = 1e-10;

// The frame GEOM's PositionAlongPath places a shape by, at parameter u of the path curve.
// GeomFill_CurveAndTrihedron over GeomFill_CorrectedFrenet gives the matrix of columns
// (normal, binormal, tangent) and the curve point: the frame's origin is the point, its z
// axis the tangent and its x axis the normal.
gp_Ax3 path_frame(GeomFill_CurveAndTrihedron& law, double u) {
  gp_Mat m;
  gp_Vec v;
  if (!law.D0(u, m, v)) {
    throw Standard_Failure("GeomFill_CurveAndTrihedron::D0 found no frame");
  }
  return gp_Ax3(gp_Pnt(v.XYZ()), gp_Dir(m.Column(3)), gp_Dir(m.Column(1)));
}

}  // namespace

// GEOM's PositionAlongPath (GEOMImpl_PositionDriver) takes the frame of BRepFill_Edge3DLaw
// over GeomFill_CorrectedFrenet and finds the length to 1e-4. One edge needs no law per
// edge of a wire, so the same GeomFill_CurveAndTrihedron runs on the edge's curve, reversed
// on a REVERSED edge as Edge3DLaw does, and the length is integrated to 1e-10 of itself.
// The start frame is the frame at the path's first vertex, not GEOM's section placement:
// the move does not depend on where the moved bodies are.
py::dict Session::position_along_path(EntityId path_edge, double distance,
                                      const std::vector<EntityId>& entity_ids, bool reverse) {
  static constexpr const char* kOp = "position_along_path";
  finite_arg(kOp, "distance", distance);
  OpGuard guard(in_op_);
  const std::string path_text = "path edge " + std::to_string(path_edge);
  if (entity_ids.empty()) {
    throw PysmeshError("Session.position_along_path: no entity is named; there is nothing to "
                       "move along " + path_text + ".",
                       "", {static_cast<int>(path_edge)});
  }
  const TopoDS_Edge edge = sole_edge(kOp, path_edge);
  if (!(distance >= 0.0 && distance <= 1.0)) {
    throw PysmeshError("Session.position_along_path: distance " + number_text(distance) +
                           " along " + path_text +
                           " is outside [0, 1]; it is a fraction of the path's length.",
                       "", {static_cast<int>(path_edge)});
  }
  // The path cannot move with the bodies it places.
  const ShapeKeyed<TopoDS_Shape> owners = body_of_subshape();
  const TopoDS_Shape path_body = sole_owner_body(owners, {edge});
  for (EntityId id : entity_ids) {
    const EntityRecord& rec = require_alive(kOp, id);
    if (!sole_owner_body(owners, rec.shapes).IsSame(path_body)) {
      continue;
    }
    std::string owner;
    const auto body_ids = state_.registry->by_shape.find(path_body);
    if (path_body.IsSame(edge)) {
      owner = "is itself a free edge body";
    } else if (body_ids != state_.registry->by_shape.end()) {
      std::string kind = kind_name(path_body.ShapeType());
      std::transform(kind.begin(), kind.end(), kind.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      owner = "belongs to " + kind + " " + std::to_string(body_ids->second.front());
    } else {
      owner = "belongs to a " + std::string(TopAbs::ShapeTypeToString(path_body.ShapeType())) +
              " body";
    }
    throw PysmeshError("Session.position_along_path: " + path_text + " " + owner +
                           ", which entity " + std::to_string(id) +
                           " moves; a path cannot move with the bodies it places.",
                       "", {static_cast<int>(path_edge), static_cast<int>(id)});
  }

  const double fraction = reverse ? 1.0 - distance : distance;
  double first = 0.0;
  double last = 0.0;
  Handle(Geom_Curve) curve = BRep_Tool::Curve(edge, first, last);
  if (curve.IsNull()) {
    throw PysmeshError("Session.position_along_path: " + path_text +
                           " has no 3D curve (it is degenerate); it has no length to move "
                           "along.",
                       "", {static_cast<int>(path_edge)});
  }
  if (edge.Orientation() == TopAbs_REVERSED) {
    Handle(Geom_TrimmedCurve) reversed = new Geom_TrimmedCurve(curve, first, last);
    reversed->Reverse();
    curve = reversed;
    first = curve->FirstParameter();
    last = curve->LastParameter();
  }
  gp_Trsf move;
  double total = 0.0;
  const char* stage = "measuring the path's length failed";
  try {
    Handle(GeomAdaptor_Curve) adaptor = new GeomAdaptor_Curve(curve, first, last);
    const double tol = std::max(kPathLengthRelTol * GCPnts_AbscissaPoint::Length(*adaptor),
                                std::numeric_limits<double>::min());
    total = GCPnts_AbscissaPoint::Length(*adaptor, tol);
    if (total > Precision::Confusion()) {
      double u = first;
      if (fraction == 1.0) {
        u = last;
      } else if (fraction > 0.0) {
        stage = "finding the parameter at the fraction failed";
        const GCPnts_AbscissaPoint ap(tol, *adaptor, fraction * total, first);
        if (!ap.IsDone()) {
          throw Standard_Failure("GCPnts_AbscissaPoint::IsDone() is false");
        }
        u = ap.Parameter();
      }
      stage = "building the corrected Frenet frames failed";
      GeomFill_CurveAndTrihedron law(new GeomFill_CorrectedFrenet());
      law.SetCurve(adaptor);
      move.SetDisplacement(path_frame(law, first), path_frame(law, u));
    }
  } catch (const std::exception& e) {
    throw PysmeshError("Session.position_along_path: " + path_text + ": " + stage + ": " +
                           e.what(),
                       "", {static_cast<int>(path_edge)});
  }
  if (!(total > Precision::Confusion())) {
    throw PysmeshError("Session.position_along_path: " + path_text + " has length " +
                           number_text(total) + "; it has no length to move along.",
                       "", {static_cast<int>(path_edge)});
  }
  // A6: a move onto the start frame changes nothing.
  gp_XYZ axis;
  double angle = 0.0;
  move.GetRotation(axis, angle);
  if (move.TranslationPart().Modulus() <= Precision::Confusion() &&
      std::abs(angle) <= Precision::Angular()) {
    throw PysmeshError("Session.position_along_path: at fraction " + number_text(fraction) +
                           " of " + path_text +
                           " the frame is the start frame; the move would change nothing.",
                       "", {static_cast<int>(path_edge)});
  }
  return apply_trsf(move, std::optional<std::vector<EntityId>>(entity_ids), kOp);
}

// Duplicate the bodies owning the named entities. Deliberately committed with no history:
// BRepBuilderAPI_Copy reports the duplicate as "modified from" the original, which would
// move the original's id onto the copy. The originals keep their ids because they are
// still in the model; every entity of every copy is a new identity.
py::dict Session::copy(const std::vector<EntityId>& entity_ids) {
  OpGuard guard(in_op_);
  const std::vector<TopoDS_Shape> sources = owner_bodies("copy", entity_ids);
  std::vector<TopoDS_Shape> copies;
  {
    py::gil_scoped_release release;
    for (const TopoDS_Shape& body : sources) {
      BRepBuilderAPI_Copy mk;
      mk.Perform(body);
      copies.push_back(mk.Shape());
    }
  }
  for (const TopoDS_Shape& c : copies) {
    if (c.IsNull()) {
      throw PysmeshError("Session.copy: BRepBuilderAPI_Copy produced a null shape.");
    }
  }
  const TopoDS_Shape added = make_root(copies);
  return commit(concat(root_bodies(state_.root), added), Handle(BRepTools_History)(),
                "copy", added);
}

// A rigid transform applied with Copy = false changes only a shape's Location: the TShape,
// and with it every sub-shape identity, is preserved. Such an operation therefore needs no
// history at all and every id survives trivially — which is a strictly better property
// than a modelling kernel that renumbers on a move.
//
// It does need a remap, because two shapes are "the same" only when their TShape AND their
// Location match, and a relocation changes the Location of every sub-shape it touches. The
// remap pairs old and new sub-shapes positionally (the topology is untouched, so the
// traversal orders agree) and asserts TShape identity on every pair, so a refactor that
// silently turned this into a copying transform would fail here rather than quietly
// orphaning every id in the model.
// Which bodies a scoped operation acts on: all of them, or the ones owning the named
// entities. Shared by the transforms and by the healing family.
std::vector<TopoDS_Shape> Session::scoped_bodies(
      const std::optional<std::vector<EntityId>>& entity_ids, const char* op_name) const {
  const std::vector<TopoDS_Shape> bodies = root_bodies(state_.root);
  if (!entity_ids.has_value()) {
    return bodies;
  }
  if (entity_ids->empty()) {
    throw PysmeshError(std::string("Session.") + op_name +
                       ": entity_ids was given but is empty; pass None to move the "
                       "whole model.");
  }
  const ShapeKeyed<TopoDS_Shape> owners = body_of_subshape();
  std::vector<TopoDS_Shape> moving;
  for (EntityId id : *entity_ids) {
    const EntityRecord& rec = require_alive(op_name, id);
    append_unique(moving, sole_owner_body(owners, rec.shapes));
  }
  // A body sharing sub-shapes with a body that stays put cannot be moved on its own
  // without tearing the model, so that is refused rather than silently split.
  guard_shared_subshapes(bodies, moving, op_name);
  return moving;
}

// A transform OCCT can express as a change of Location alone takes the relocation path,
// where identity is structural; anything else — a plane mirror, a scaling — is a genuine
// rebuild and takes the history path. The two are the same test: BRepBuilderAPI_Transform
// sets a new Location only for a direct isometry, and TopLoc_Datum3D refuses to be built
// from anything else at all.
py::dict Session::apply_trsf(const gp_Trsf& trsf,
                      const std::optional<std::vector<EntityId>>& entity_ids,
                      const char* op_name) {
  if (is_location_only(trsf)) {
    return relocate(trsf, entity_ids, op_name);
  }
  return rebuild_moved(entity_ids, op_name,
                       [&trsf](const TopoDS_Shape& input, TopoDS_Shape& out,
                               Handle(BRepTools_History) & hist) {
                         BRepBuilderAPI_Transform mk(trsf);
                         mk.Perform(input, /*theCopyGeom=*/true);
                         if (mk.IsDone()) {
                           out = mk.Shape();
                           hist = history_of(input, mk);
                         }
                       });
}

// The rebuild half of the transform pair. The moving bodies are handed to OCCT as one
// compound so a single history covers them all; every sub-shape maps one-to-one through
// Modified, so every entity id survives — it just survives by history rather than by
// TShape identity.
py::dict Session::rebuild_moved(const std::optional<std::vector<EntityId>>& entity_ids,
                         const char* op_name,
                         const std::function<void(const TopoDS_Shape&, TopoDS_Shape&,
                                                  Handle(BRepTools_History)&)>& run) {
  const std::vector<TopoDS_Shape> moving = scoped_bodies(entity_ids, op_name);
  const std::vector<TopoDS_Shape> staying = bodies_excluding(moving);
  const TopoDS_Shape input = make_root(moving);

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  {
    py::gil_scoped_release release;
    try {
      run(input, result, hist);
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.") + op_name + ": OCCT's transform threw: " +
                         e.what());
    }
  }
  if (result.IsNull()) {
    throw PysmeshError(std::string("Session.") + op_name +
                       ": OCCT could not rebuild the shape under this transform.");
  }
  return commit(concat(staying, result), hist, op_name, result);
}

py::dict Session::relocate(const gp_Trsf& trsf,
                    const std::optional<std::vector<EntityId>>& entity_ids,
                    const char* op_name) {
  const std::vector<TopoDS_Shape> bodies = root_bodies(state_.root);
  const std::vector<TopoDS_Shape> moving = scoped_bodies(entity_ids, op_name);

  const TopLoc_Location move(trsf);
  std::vector<TopoDS_Shape> new_bodies;
  ShapeKeyed<TopoDS_Shape> remap;

  for (const TopoDS_Shape& body : bodies) {
    bool is_moving = false;
    for (const TopoDS_Shape& m : moving) {
      if (body.IsSame(m)) {
        is_moving = true;
        break;
      }
    }
    if (!is_moving) {
      new_bodies.push_back(body);
      continue;
    }
    const TopoDS_Shape moved = body.Moved(move);
    new_bodies.push_back(moved);

    const std::vector<TopoDS_Shape> before = registered_subshapes(body);
    const std::vector<TopoDS_Shape> after = registered_subshapes(moved);
    if (before.size() != after.size()) {
      throw PysmeshError(std::string("Session.") + op_name +
                         ": the transform changed the body's topology; a rigid transform "
                         "must be a location-only change.");
    }
    for (std::size_t i = 0; i < before.size(); ++i) {
      if (!before[i].IsPartner(after[i])) {
        throw PysmeshError(
            std::string("Session.") + op_name +
            ": the transform did not preserve sub-shape identity (TShape pointers "
            "differ). A rigid transform must not copy the geometry.");
      }
      remap.emplace(before[i], after[i]);
    }
  }

  const std::int64_t op_index = next_op_;
  auto next = std::make_shared<RegistryState>();
  const RegistryState& prev = *state_.registry;
  Delta delta;

  for (const auto& [id, rec] : prev.alive) {
    EntityRecord out;
    out.kind = rec.kind;
    bool changed = false;
    for (const TopoDS_Shape& s : rec.shapes) {
      const auto it = remap.find(s);
      if (it == remap.end()) {
        out.shapes.push_back(s);
      } else {
        out.shapes.push_back(it->second);
        changed = true;
      }
    }
    for (const TopoDS_Shape& s : out.shapes) {
      next->by_shape[s].push_back(id);
    }
    if (changed) {
      delta.modified.push_back(id);
    }
    if (out.shapes.size() > 1) {
      delta.split.push_back(id);
    }
    next->alive.emplace(id, std::move(out));
  }
  for (auto& [shape, ids] : next->by_shape) {
    std::sort(ids.begin(), ids.end());
    if (ids.size() > 1) {
      for (EntityId id : ids) {
        delta.merged.push_back(id);
      }
    }
  }

  state_.registry = std::move(next);
  state_.root = make_root(new_bodies);
  state_.op_index = op_index;
  ++next_op_;
  finalise(delta);
  return delta_dict(delta, op_index, op_name);
}

void Session::guard_shared_subshapes(const std::vector<TopoDS_Shape>& all,
                              const std::vector<TopoDS_Shape>& moving,
                              const char* op_name) const {
  ShapeSet moving_subs;
  for (const TopoDS_Shape& m : moving) {
    for (const TopoDS_Shape& s : registered_subshapes(m)) {
      moving_subs.Add(s);
    }
  }
  for (const TopoDS_Shape& body : all) {
    bool is_moving = false;
    for (const TopoDS_Shape& m : moving) {
      if (body.IsSame(m)) {
        is_moving = true;
        break;
      }
    }
    if (is_moving) {
      continue;
    }
    for (const TopoDS_Shape& s : registered_subshapes(body)) {
      if (moving_subs.Contains(s)) {
        throw PysmeshError(
            std::string("Session.") + op_name +
            ": the selected bodies share sub-shapes with bodies that stay put; moving "
            "them apart would tear the model. Move them together, or separate them "
            "first.");
      }
    }
  }
}

}  // namespace session
}  // namespace pysmesh
