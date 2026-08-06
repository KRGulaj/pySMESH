// pySMESH binding — Session: state, identity and the queries over them.
//
// The identity-carrying core lives here: commit() replaces the root shape, carry_registry()
// walks an operation's history to decide which EntityIds survive it, and the snapshot stack
// retains whole states at handle cost. The other translation units build shapes; this one
// decides what those shapes are *called*.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

namespace pysmesh {
namespace session {

Session::Session(bool validate) : validate_(validate) {
  state_.root = make_root({});
  state_.registry = std::make_shared<RegistryState>();
  state_.op_index = 0;
}

// ---- snapshot / restore ----------------------------------------------------------- //

// Guarded like an operation, not like a query. These mutate the session, and an operation
// that has released the GIL is still in flight: a restore landing in that window would
// swap the state out from under it. A read-only query needs no guard, because the state a
// running operation publishes is only ever written while the GIL is held.
std::int64_t Session::snapshot() {
  OpGuard guard(in_op_);
  snapshots_.push_back(state_);
  return static_cast<std::int64_t>(snapshots_.size()) - 1;
}

void Session::restore(std::int64_t mark) {
  OpGuard guard(in_op_);
  if (mark < 0 || static_cast<std::size_t>(mark) >= snapshots_.size()) {
    throw PysmeshError("Session.restore: unknown snapshot mark " + std::to_string(mark) +
                       ".");
  }
  const std::optional<SessionState>& snap = snapshots_[static_cast<std::size_t>(mark)];
  if (!snap.has_value()) {
    throw PysmeshError("Session.restore: snapshot mark " + std::to_string(mark) +
                       " was discarded.");
  }
  state_ = *snap;
}

void Session::discard_snapshot(std::int64_t mark) {
  OpGuard guard(in_op_);
  if (mark < 0 || static_cast<std::size_t>(mark) >= snapshots_.size()) {
    throw PysmeshError("Session.discard_snapshot: unknown snapshot mark " +
                       std::to_string(mark) + ".");
  }
  snapshots_[static_cast<std::size_t>(mark)].reset();
}

std::int64_t Session::snapshot_count() const {
  std::int64_t n = 0;
  for (const std::optional<SessionState>& s : snapshots_) {
    if (s.has_value()) {
      ++n;
    }
  }
  return n;
}

// ---- queries ---------------------------------------------------------------------- //

py::array_t<std::int64_t> Session::entities(const std::string& kind) const {
  const TopAbs_ShapeEnum k = kind_from_name(kind);
  std::vector<EntityId> ids;
  for (const auto& [id, rec] : state_.registry->alive) {
    if (rec.kind == k) {
      ids.push_back(id);
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids_array(ids);
}

std::string Session::entity_kind(EntityId id) const {
  return kind_name(require_alive("entity_kind", id).kind);
}

// "alive" or "dead". An id that was never issued is a caller error, not a state: it
// raises, which is how a positional ordinal from the stateless API fails loudly here
// instead of silently denoting somebody else's entity.
std::string Session::entity_state(EntityId id) const {
  require_issued("entity_state", id);
  return state_.registry->alive.count(id) != 0 ? "alive" : "dead";
}

std::int64_t Session::shape_count(EntityId id) const {
  return static_cast<std::int64_t>(require_alive("shape_count", id).shapes.size());
}

// Bulk geometry of every alive entity of one kind, as parallel arrays. Bulk rather than
// per-entity objects because the calling convention has to stay vectorised for a model
// with tens of thousands of faces.
py::dict Session::entity_table(const std::string& kind) const {
  const TopAbs_ShapeEnum k = kind_from_name(kind);
  std::vector<EntityId> ids;
  for (const auto& [id, rec] : state_.registry->alive) {
    if (rec.kind == k) {
      ids.push_back(id);
    }
  }
  std::sort(ids.begin(), ids.end());

  const auto n = static_cast<py::ssize_t>(ids.size());
  py::array_t<double> measure(n);
  py::array_t<double> centroid({n, static_cast<py::ssize_t>(3)});
  py::array_t<double> bbox({n, static_cast<py::ssize_t>(6)});
  py::array_t<std::int64_t> shapes(n);

  double* mp = measure.mutable_data();
  double* cp = centroid.mutable_data();
  double* bp = bbox.mutable_data();
  std::int64_t* sp = shapes.mutable_data();

  for (py::ssize_t i = 0; i < n; ++i) {
    const EntityRecord& rec = state_.registry->alive.at(ids[static_cast<std::size_t>(i)]);
    sp[i] = static_cast<std::int64_t>(rec.shapes.size());
    // A split entity is measured over all of its shapes; its centroid and bounding box
    // cover them all. That keeps a split honest in the ground-truth sense: the entity is
    // everything it now denotes, not an arbitrary one of the pieces.
    double total = 0.0;
    double wsum = 0.0;
    Bnd_Box box;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (const TopoDS_Shape& s : rec.shapes) {
      const double m = measure_of(s);
      const std::array<double, 3> c = centroid_of(s);
      // A vertex (and a degenerate edge) has zero measure, so weight it as one instead:
      // an unweighted mean of the pieces is the only meaningful centroid there.
      const double w = (m > 0.0) ? m : 1.0;
      total += m;
      wsum += w;
      cx += w * c[0];
      cy += w * c[1];
      cz += w * c[2];
      BRepBndLib::Add(s, box);
    }
    mp[i] = total;
    cp[3 * i + 0] = cx / wsum;
    cp[3 * i + 1] = cy / wsum;
    cp[3 * i + 2] = cz / wsum;
    box.Get(bp[6 * i + 0], bp[6 * i + 1], bp[6 * i + 2], bp[6 * i + 3], bp[6 * i + 4],
            bp[6 * i + 5]);
  }

  py::dict out;
  out["ids"] = ids_array(ids);
  out["measure"] = measure;
  out["centroid"] = centroid;
  out["bbox"] = bbox;
  out["shape_count"] = shapes;
  return out;
}

py::bytes Session::brep() const {
  std::ostringstream stream;
  try {
    BRepTools::Write(state_.root, stream);
  } catch (const std::exception& e) {
    throw PysmeshError(std::string("Session.brep: BREP write failed: ") + e.what());
  }
  return py::bytes(stream.str());
}

// ---- names ------------------------------------------------------------------------ //

// The persistent name of an entity: the operation that issued its id, how that operation
// produced it, and its rank among that operation's issued entities of that role. Purely
// provenance — no geometric fingerprint is involved at any point, because fingerprint
// matching mis-identifies entities under exactly the edits that make naming necessary.
py::dict Session::name_of(EntityId id) const {
  require_alive("name_of", id);
  const Origin& o = origins_.at(id);
  py::dict out;
  out["op_index"] = o.op_index;
  out["role"] = static_cast<int>(o.role);
  out["ordinal"] = o.ordinal;
  return out;
}

py::dict Session::origin(EntityId id) const {
  require_issued("origin", id);
  const Origin& o = origins_.at(id);
  py::dict out;
  out["op_index"] = o.op_index;
  out["role"] = static_cast<int>(o.role);
  out["ordinal"] = o.ordinal;
  out["sources"] = ids_array(o.sources);
  return out;
}

// Resolve a name against the CURRENT state. "lost" is a legitimate, loud answer and the
// caller must handle it; guessing a replacement is never done.
py::dict Session::resolve(std::int64_t op_index, int role, int ordinal) const {
  const auto key = std::make_tuple(op_index, role, ordinal);
  const auto it = name_index_.find(key);
  if (it == name_index_.end()) {
    throw PysmeshError("Session.resolve: no entity was ever named (op_index=" +
                       std::to_string(op_index) + ", role=" + std::to_string(role) +
                       ", ordinal=" + std::to_string(ordinal) + ").");
  }
  const EntityId id = it->second;
  const auto alive = state_.registry->alive.find(id);

  py::dict out;
  if (alive == state_.registry->alive.end()) {
    out["status"] = "lost";
    out["ids"] = ids_array({});
    out["shape_count"] = static_cast<std::int64_t>(0);
    return out;
  }
  const std::int64_t n = static_cast<std::int64_t>(alive->second.shapes.size());
  out["status"] = (n == 1) ? "resolved" : "ambiguous";
  out["ids"] = ids_array({id});
  out["shape_count"] = n;
  return out;
}

std::int64_t Session::entity_count() const {
  return static_cast<std::int64_t>(state_.registry->alive.size());
}

void Session::require_positive(const char* name, double v) {
  if (!(v > 0.0)) {
    throw PysmeshError(std::string("Session: ") + name + " must be > 0 (got " +
                       std::to_string(v) + ").");
  }
}

void Session::require_non_negative(const char* name, double v) {
  if (!(v >= 0.0)) {
    throw PysmeshError(std::string("Session: ") + name + " must be >= 0 (got " +
                       std::to_string(v) + ").");
  }
}

// A partial primitive sweeps through angle_rad about its axis; the full solid is 2*pi.
// OCCT clamps silently outside that band, which would hand back a shape the caller did
// not ask for, so it is refused here instead.
void Session::require_sweep_angle(const char* op, double angle_rad) {
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  if (!(angle_rad > 0.0) || angle_rad > kTwoPi + 1e-12) {
    throw PysmeshError(std::string("Session.") + op +
                       ": angle_rad must be in (0, 2*pi] (got " +
                       std::to_string(angle_rad) + ").");
  }
}

// Entity ids for PysmeshError's offending-id channel. That field is int-typed by v1's
// contract, which is wide enough for every id a session can realistically issue.
std::vector<int> Session::ids_as_int(const std::vector<EntityId>& ids) {
  std::vector<int> out;
  out.reserve(ids.size());
  for (EntityId id : ids) {
    out.push_back(static_cast<int>(id));
  }
  return out;
}

// A shape OCCT's sweeps raise the dimension of. A solid has no higher dimension to sweep
// into, and a compound sweeps element-wise, which would silently produce a shape the
// caller did not ask for.
void Session::require_sweepable(const char* op, const TopoDS_Shape& body) {
  switch (body.ShapeType()) {
    case TopAbs_VERTEX:
    case TopAbs_EDGE:
    case TopAbs_WIRE:
    case TopAbs_FACE:
    case TopAbs_SHELL:
      return;
    default:
      throw PysmeshError(std::string("Session.") + op + ": the named entities belong to a " +
                         std::string(TopAbs::ShapeTypeToString(body.ShapeType())) +
                         " body; a sweep needs a vertex, edge, wire, face or shell "
                         "profile.");
  }
}

void Session::require_issued(const char* op, EntityId id) const {
  if (id < 1 || id >= next_id_) {
    throw PysmeshError(std::string("Session.") + op + ": " + std::to_string(id) +
                       " is not an EntityId this session ever issued (issued 1.." +
                       std::to_string(next_id_ - 1) +
                       "). Session ids are not the positional ordinals the stateless "
                       "API returns.");
  }
}

const EntityRecord& Session::require_alive(const char* op, EntityId id) const {
  require_issued(op, id);
  const auto it = state_.registry->alive.find(id);
  if (it == state_.registry->alive.end()) {
    throw PysmeshError(std::string("Session.") + op + ": entity " + std::to_string(id) +
                       " is dead (it was removed by an earlier operation).");
  }
  return it->second;
}

// ---- root bookkeeping ------------------------------------------------------------- //

std::vector<TopoDS_Shape> Session::bodies_excluding(
    const std::vector<TopoDS_Shape>& drop) const {
  std::vector<TopoDS_Shape> out;
  for (const TopoDS_Shape& body : root_bodies(state_.root)) {
    bool skip = false;
    for (const TopoDS_Shape& d : drop) {
      if (body.IsSame(d)) {
        skip = true;
        break;
      }
    }
    if (!skip) {
      out.push_back(body);
    }
  }
  return out;
}

std::vector<TopoDS_Shape> Session::concat(const std::vector<TopoDS_Shape>& keep,
                                          const TopoDS_Shape& added) {
  std::vector<TopoDS_Shape> out = keep;
  explode_into(added, out);
  return out;
}

// sub-shape -> owning body, for every body of the current root. Built once per call so
// that resolving many entities to their bodies stays linear rather than quadratic.
ShapeKeyed<TopoDS_Shape> Session::body_of_subshape() const {
  ShapeKeyed<TopoDS_Shape> out;
  for (const TopoDS_Shape& body : root_bodies(state_.root)) {
    for (const TopoDS_Shape& s : registered_subshapes(body)) {
      out.emplace(s, body);
    }
  }
  return out;
}

// The single body that owns every one of the given sub-shapes. Fails loud when the
// selection straddles two bodies, because one operation builds one shape and the caller's
// intent is genuinely ambiguous in that case.
TopoDS_Shape Session::sole_owner_body(const ShapeKeyed<TopoDS_Shape>& owners,
                                      const std::vector<TopoDS_Shape>& subs) {
  TopoDS_Shape owner;
  for (const TopoDS_Shape& sub : subs) {
    const auto it = owners.find(sub);
    if (it == owners.end()) {
      throw PysmeshError(
          "Session: a named sub-entity does not belong to any body of the session root.");
    }
    if (owner.IsNull()) {
      owner = it->second;
    } else if (!owner.IsSame(it->second)) {
      throw PysmeshError(
          "Session: the named entities belong to different bodies. One operation acts on "
          "one body; split the call.");
    }
  }
  return owner;
}

// Every distinct body of the current root that owns at least one shape of the named
// entities, in first-appearance order so the outcome is reproducible.
std::vector<TopoDS_Shape> Session::owner_bodies(const char* op,
                                         const std::vector<EntityId>& ids) const {
  if (ids.empty()) {
    throw PysmeshError(std::string("Session.") + op +
                       ": at least one entity must be named.");
  }
  const ShapeKeyed<TopoDS_Shape> owners = body_of_subshape();
  std::vector<TopoDS_Shape> out;
  for (EntityId id : ids) {
    const EntityRecord& rec = require_alive(op, id);
    for (const TopoDS_Shape& s : rec.shapes) {
      const auto it = owners.find(s);
      if (it == owners.end()) {
        throw PysmeshError(std::string("Session.") + op + ": entity " +
                           std::to_string(id) +
                           " does not belong to any body of the session root.");
      }
      append_unique(out, it->second);
    }
  }
  return out;
}

// The single body that owns every named entity. One operation builds one shape, so a
// selection straddling two bodies is a genuinely ambiguous request and is refused.
TopoDS_Shape Session::sole_body(const char* op, const std::vector<EntityId>& ids) const {
  const std::vector<TopoDS_Shape> bodies = owner_bodies(op, ids);
  if (bodies.size() != 1) {
    throw PysmeshError(std::string("Session.") + op + ": the named entities belong to " +
                       std::to_string(bodies.size()) +
                       " different bodies. One operation acts on one body; split the "
                       "call.");
  }
  return bodies.front();
}

// The edges the named entities denote. A split entity is rejected rather than silently
// contributing several edges, because the caller's intent is then ambiguous. `kept`, when
// given, receives the id of each returned edge, so a per-edge diagnostic can name it.
std::vector<TopoDS_Shape> Session::edges_of(const char* op, const std::vector<EntityId>& ids,
                                     std::vector<EntityId>* kept) const {
  if (ids.empty()) {
    throw PysmeshError(std::string("Session.") + op +
                       ": edge_ids must name at least one edge.");
  }
  std::vector<TopoDS_Shape> out;
  for (EntityId id : ids) {
    const EntityRecord& rec = require_alive(op, id);
    if (rec.kind != TopAbs_EDGE) {
      throw PysmeshError(std::string("Session.") + op + ": entity " + std::to_string(id) +
                         " is a " + kind_name(rec.kind) + ", not an EDGE.");
    }
    if (rec.shapes.size() != 1) {
      throw PysmeshError(std::string("Session.") + op + ": entity " + std::to_string(id) +
                         " denotes " + std::to_string(rec.shapes.size()) +
                         " edges (it was split); name one of them instead.");
    }
    const std::size_t before = out.size();
    append_unique(out, rec.shapes.front());
    if (kept != nullptr && out.size() != before) {
      kept->push_back(id);
    }
  }
  return out;
}

// Construction geometry: a body that is a loose edge or a wire. Building a wire or a face
// out of an edge that belongs to a solid would consume the solid, so that is refused.
void Session::require_curve_body(const char* op, const TopoDS_Shape& body) {
  if (body.ShapeType() != TopAbs_EDGE && body.ShapeType() != TopAbs_WIRE) {
    throw PysmeshError(
        std::string("Session.") + op + ": the named edges belong to a " +
        std::string(TopAbs::ShapeTypeToString(body.ShapeType())) +
        " body. This operation consumes its inputs, so it accepts only construction "
        "geometry (loose edges and wires).");
  }
}

// A wire over the named edges. When they are exactly one existing wire body that wire is
// used as-is, which keeps every edge id: BRepBuilderAPI_MakeWire rebuilds any edge whose
// end vertex is merely coincident with the wire's rather than shared, and a rebuilt edge
// is a new entity.
TopoDS_Wire Session::wire_over(const char* op, const std::vector<TopoDS_Shape>& edges,
                               const std::vector<TopoDS_Shape>& owners) {
  if (owners.size() == 1 && owners.front().ShapeType() == TopAbs_WIRE) {
    ShapeSet in_wire;
    TopExp::MapShapes(owners.front(), TopAbs_EDGE, in_wire);
    if (in_wire.Extent() == static_cast<int>(edges.size())) {
      return TopoDS::Wire(owners.front());
    }
  }
  BRepBuilderAPI_MakeWire mk;
  NCollection_List<TopoDS_Shape> list;
  for (const TopoDS_Shape& e : edges) {
    list.Append(e);
  }
  mk.Add(list);
  if (!mk.IsDone()) {
    throw PysmeshError(std::string("Session.") + op +
                       ": the named edges do not form a connected wire.");
  }
  return mk.Wire();
}

// A wire over a whole body, for the operations that sweep along or across one.
TopoDS_Wire Session::wire_of_body(const char* op, const char* argname,
                                  const TopoDS_Shape& body) {
  if (body.ShapeType() == TopAbs_WIRE) {
    return TopoDS::Wire(body);
  }
  if (body.ShapeType() == TopAbs_EDGE) {
    return BRepBuilderAPI_MakeWire(TopoDS::Edge(body)).Wire();
  }
  throw PysmeshError(std::string("Session.") + op + ": " + argname + " names a " +
                     std::string(TopAbs::ShapeTypeToString(body.ShapeType())) +
                     " body; a wire or a single edge is required.");
}

py::dict Session::add_bodies(const TopoDS_Shape& added, const char* op_name) {
  return commit(concat(root_bodies(state_.root), added), Handle(BRepTools_History)(),
                op_name, added);
}

// ---- the identity-carrying core --------------------------------------------------- //

// Replace the root and carry the id registry across the change, using `hist` (which may be
// null, meaning "nothing was rewritten": every surviving sub-shape keeps its id because it
// is literally the same shape).
//
//   * an entity Modified to exactly one output keeps its id;
//   * Modified to several -> the id survives on all of them (a split);
//   * several entities modified onto one output -> all their ids survive on it (a merge);
//   * an output with no input correspondence -> a new id;
//   * removed, or modified onto something absent from the result -> the id dies, and is
//     never reused.
//
// `built` is the shape this operation actually produced. Only that is validity-checked:
// the bodies it left alone were checked when they were built, and re-checking the whole
// root every time would make a long session quadratic in the model size for no new
// information.
py::dict Session::commit(const std::vector<TopoDS_Shape>& bodies,
                  const Handle(BRepTools_History) & history, const char* op_name,
                  const TopoDS_Shape& built) {
  Handle(BRepTools_History) hist = history;
  if (tear_next_history_) {
    hist.Nullify();
    tear_next_history_ = false;
  }

  if (validate_ && !built.IsNull()) {
    bool valid = true;
    {
      py::gil_scoped_release release;
      valid = BRepCheck_Analyzer(built).IsValid();
    }
    if (!valid) {
      throw PysmeshError(std::string("Session.") + op_name +
                         ": the operation produced an invalid shape "
                         "(BRepCheck_Analyzer reported errors); the session is unchanged.");
    }
  }
  const TopoDS_Shape new_root = make_root(bodies);

  const std::int64_t op_index = next_op_;
  const Delta delta = carry_registry(new_root, hist, op_index);

  state_.root = new_root;
  state_.op_index = op_index;
  ++next_op_;
  return delta_dict(delta, op_index, op_name);
}

Delta Session::carry_registry(const TopoDS_Shape& new_root,
                              const Handle(BRepTools_History) & hist, std::int64_t op_index) {
  const std::vector<TopoDS_Shape> new_order = registered_subshapes(new_root);
  ShapeKeyed<TopAbs_ShapeEnum> members;
  members.reserve(new_order.size());
  for (const TopoDS_Shape& s : new_order) {
    members.emplace(s, s.ShapeType());
  }

  auto next = std::make_shared<RegistryState>();
  const RegistryState& prev = *state_.registry;
  Delta delta;

  // Which input entities each new shape was Generated from. Built forward, once, so the
  // per-output lookup in the second pass is O(1) rather than a scan of the old registry.
  ShapeKeyed<std::vector<EntityId>> generated_from;

  // Pass 1 — carry existing ids forward. Ascending id order keeps the outcome independent
  // of hash-table iteration order.
  std::vector<EntityId> old_ids;
  old_ids.reserve(prev.alive.size());
  for (const auto& [id, rec] : prev.alive) {
    old_ids.push_back(id);
  }
  std::sort(old_ids.begin(), old_ids.end());

  for (EntityId id : old_ids) {
    const EntityRecord& rec = prev.alive.at(id);

    // An entity every one of whose shapes is still in the new model did not move, whatever
    // the history says about it. An algorithm that keeps its arguments — an additive
    // section, a splitter that keeps its tools — still reports their sub-shapes as
    // Modified, because it split them inside its own result; following that would migrate
    // a live operand's id onto a fragment of the answer and re-issue an id for the operand
    // itself. The shape being present in the model is the stronger evidence.
    bool intact = true;
    for (const TopoDS_Shape& s : rec.shapes) {
      if (members.count(s) == 0) {
        intact = false;
        break;
      }
    }

    std::vector<TopoDS_Shape> successors;
    for (const TopoDS_Shape& s : rec.shapes) {
      if (!hist.IsNull()) {
        // Generated relations are collected either way: they name the operation's new
        // entities against the inputs they came from, which is true whether or not the
        // input survived.
        for (const TopoDS_Shape& g : hist->Generated(s)) {
          generated_from[g].push_back(id);
        }
        if (intact) {
          // The operand stays where it is, but what the algorithm derived from it is
          // still derived from it. Recording the Modified targets here is what gives an
          // additive operation's new entities a source: the boolean family reports almost
          // everything as Modified and hardly ever as Generated, so without this a
          // section's curves would have no provenance at all. Targets that no id carries
          // onto are exactly the new entities, and pass 2 picks them up there.
          for (const TopoDS_Shape& m : hist->Modified(s)) {
            generated_from[m].push_back(id);
          }
        } else {
          if (hist->IsRemoved(s)) {
            continue;
          }
          const NCollection_List<TopoDS_Shape>& mod = hist->Modified(s);
          if (!mod.IsEmpty()) {
            for (const TopoDS_Shape& m : mod) {
              append_unique(successors, m);
            }
            continue;
          }
        }
      }
      append_unique(successors, s);
    }

    // A successor absent from the result is not a survivor. This is where an entity that
    // a boolean consumed without recording a removal is caught: it dies here rather than
    // being carried onto a shape that is no longer part of the model.
    std::vector<TopoDS_Shape> live;
    for (const TopoDS_Shape& s : successors) {
      if (members.count(s) != 0) {
        append_unique(live, s);
      }
    }

    // An operation that reports a sub-shape as modified, but leaves the original in the
    // model, has not touched that entity: the modified copies live inside the algorithm's
    // own result, which this operation did not adopt. A boolean whose arguments survive
    // the call — an additive section, a splitter that keeps its tools — is exactly that
    // case, and without this the whole untouched operand would be declared deleted.
    if (live.empty()) {
      for (const TopoDS_Shape& s : rec.shapes) {
        if (members.count(s) != 0) {
          append_unique(live, s);
        }
      }
    }

    if (live.empty()) {
      delta.deleted.push_back(id);
      continue;
    }
    EntityRecord out;
    out.kind = rec.kind;
    out.shapes = live;
    for (const TopoDS_Shape& s : live) {
      next->by_shape[s].push_back(id);
    }
    next->alive.emplace(id, std::move(out));

    const bool same = live.size() == rec.shapes.size() &&
                      std::equal(live.begin(), live.end(), rec.shapes.begin(),
                                 [](const TopoDS_Shape& a, const TopoDS_Shape& b) {
                                   return a.IsSame(b);
                                 });
    if (!same) {
      delta.modified.push_back(id);
    }
    if (live.size() > 1) {
      delta.split.push_back(id);
    }
  }

  // Pass 2 — issue ids for outputs nothing carried onto, in the deterministic sub-shape
  // order, so a name's ordinal is reproducible.
  int constructed_rank = 0;
  int generated_rank = 0;
  for (const TopoDS_Shape& s : new_order) {
    if (next->by_shape.count(s) != 0) {
      continue;
    }
    const EntityId id = next_id_++;
    EntityRecord rec;
    rec.kind = s.ShapeType();
    rec.shapes = {s};
    next->by_shape[s].push_back(id);
    next->alive.emplace(id, std::move(rec));

    Origin o;
    o.op_index = op_index;
    const auto gen = generated_from.find(s);
    if (gen != generated_from.end() && !gen->second.empty()) {
      o.role = Role::Generated;
      o.ordinal = generated_rank++;
      o.sources = gen->second;
      std::sort(o.sources.begin(), o.sources.end());
      o.sources.erase(std::unique(o.sources.begin(), o.sources.end()), o.sources.end());
    } else {
      o.role = Role::Constructed;
      o.ordinal = constructed_rank++;
    }
    name_index_.emplace(std::make_tuple(o.op_index, static_cast<int>(o.role), o.ordinal),
                        id);
    origins_.emplace(id, std::move(o));
    delta.created.push_back(id);
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
  finalise(delta);
  return delta;
}

void Session::finalise(Delta& d) {
  auto tidy = [](std::vector<EntityId>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  };
  tidy(d.created);
  tidy(d.deleted);
  tidy(d.modified);
  tidy(d.split);
  tidy(d.merged);
}

py::dict Session::delta_dict(const Delta& d, std::int64_t op_index, const char* op_name) {
  py::dict out;
  out["op_index"] = op_index;
  out["op"] = py::str(op_name);
  out["created"] = ids_array(d.created);
  out["deleted"] = ids_array(d.deleted);
  out["modified"] = ids_array(d.modified);
  out["split"] = ids_array(d.split);
  out["merged"] = ids_array(d.merged);
  return out;
}

}  // namespace session
}  // namespace pysmesh
