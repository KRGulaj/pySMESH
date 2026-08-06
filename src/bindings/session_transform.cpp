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

namespace pysmesh {
namespace session {

// ---- transforms ------------------------------------------------------------------- //

py::dict Session::translate(double dx, double dy, double dz,
                     const std::optional<std::vector<EntityId>>& entity_ids) {
  OpGuard guard(in_op_);
  gp_Trsf t;
  t.SetTranslation(gp_Vec(dx, dy, dz));
  return apply_trsf(t, entity_ids, "translate");
}

py::dict Session::rotate(double ox, double oy, double oz, double ax, double ay, double az,
                  double angle_rad, const std::optional<std::vector<EntityId>>& entity_ids) {
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
// Which bodies an entity-scoped transform acts on: all of them, or the ones owning the
// named entities.
std::vector<TopoDS_Shape> Session::moving_bodies(
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
  const std::vector<TopoDS_Shape> moving = moving_bodies(entity_ids, op_name);
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
  const std::vector<TopoDS_Shape> moving = moving_bodies(entity_ids, op_name);

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
