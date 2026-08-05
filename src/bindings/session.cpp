// pySMESH binding — Session: the stateful, history-carrying modelling context.
//
// The rest of pysmesh is a stateless BREP-in/BREP-out service: every entry point reads
// bytes, runs one OCCT algorithm, and writes bytes back. That shape is right for one-shot
// work and wrong for interactive modelling, because three things need a shape that survives
// between calls:
//
//   * O(1) undo — a snapshot must BE the shape, not a serialisation of it;
//   * persistent naming — BRepTools_History relates one call's inputs to that call's
//     outputs, and a serialise/deserialise boundary destroys the TShape identity the
//     history is expressed in, so histories cannot be composed across it;
//   * incremental tessellation — BRepMesh caches the triangulation on the TopoDS_Face, and
//     a fresh BRepTools::Read produces faces with no triangulation at all.
//
// A Session owns one live root shape plus an EntityId registry that is carried across every
// operation by that operation's history. Ids are monotonic and are never reused, so a stale
// reference always resolves to *dead* — never to a different entity. That last property is
// the whole point: a dense positional ordinal (what the stateless API returns) always
// resolves to *something*, which is the failure mode persistent naming exists to remove.
//
// Threading: a Session is NOT thread-safe. One session per thread. Sessions are independent
// and several may coexist in one process. Long operations release the GIL, so concurrent use
// from two threads would be a genuine data race; an operation entered while another is in
// flight on the same session raises instead (see OpGuard).
//
// This file is the low-level half. src/pysmesh/session.py wraps it in frozen dataclasses and
// distinct id types, which is where the public surface is documented.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBndLib.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepTools.hxx>
#include <BRepTools_History.hxx>
#include <BRep_Builder.hxx>
#include <Bnd_Box.hxx>
#include <GProp_GProps.hxx>
#include <NCollection_IndexedDataMap.hxx>
#include <NCollection_IndexedMap.hxx>
#include <NCollection_List.hxx>
#include <Standard_Handle.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>
#include <gp_Vec.hxx>

#include "common.hpp"

namespace pysmesh {
namespace {

// ---- Types ------------------------------------------------------------------------- //

using EntityId = std::int64_t;

// TopTools_ShapeMapHasher provides BOTH operator()(shape) -> size_t and
// operator()(shape, shape) -> bool, so it serves as hash AND equality for std containers.
// Equality is IsSame: same TShape and same Location, orientation ignored. That is the right
// notion of entity sameness here — an entity that is only re-oriented is the same entity,
// and one that is relocated is not (see relocate(), which handles that case explicitly).
using ShapeSet = NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher>;
template <typename V>
using ShapeKeyed = std::unordered_map<TopoDS_Shape, V, TopTools_ShapeMapHasher,
                                      TopTools_ShapeMapHasher>;

// How an operation produced an entity whose id it issued.
//   Constructed — no input correspondence at all (a primitive, an import).
//   Generated   — the operation's history relates it to one or more input entities.
// An entity that an operation *modified* keeps its existing id and therefore keeps its
// existing origin; modification never mints a name.
enum class Role : int { Constructed = 0, Generated = 1 };

// The four shape kinds that carry ids. This set is not a choice: BRepTools_History records
// relations for exactly VERTEX, EDGE, FACE and SOLID (BRepTools_History::IsSupportedType),
// so an id on any other kind could not be carried across an operation and would silently
// die at the first boolean.
constexpr TopAbs_ShapeEnum kEntityKinds[] = {TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE,
                                             TopAbs_VERTEX};

const char* kind_name(TopAbs_ShapeEnum k) {
  switch (k) {
    case TopAbs_SOLID:
      return "SOLID";
    case TopAbs_FACE:
      return "FACE";
    case TopAbs_EDGE:
      return "EDGE";
    case TopAbs_VERTEX:
      return "VERTEX";
    default:
      return "OTHER";
  }
}

TopAbs_ShapeEnum kind_from_name(const std::string& name) {
  if (name == "SOLID") return TopAbs_SOLID;
  if (name == "FACE") return TopAbs_FACE;
  if (name == "EDGE") return TopAbs_EDGE;
  if (name == "VERTEX") return TopAbs_VERTEX;
  throw PysmeshError("Unknown entity kind '" + name +
                     "' (expected SOLID, FACE, EDGE or VERTEX).");
}

// One live entity. `shapes` holds one shape normally, and several after a split — an
// operation that modified one input into several outputs keeps the id on all of them,
// because dropping it would lose the caller's reference and picking one arbitrarily would
// be a guess. A name over such an entity resolves as Ambiguous, never silently.
struct EntityRecord {
  std::vector<TopoDS_Shape> shapes;
  TopAbs_ShapeEnum kind = TopAbs_SHAPE;
};

// The id registry at one point in the session's history. Immutable once published: every
// operation builds a fresh RegistryState and swaps it in, which is what makes a snapshot a
// shared_ptr copy rather than a deep copy of the map.
struct RegistryState {
  std::unordered_map<EntityId, EntityRecord> alive;
  ShapeKeyed<std::vector<EntityId>> by_shape;
};
using RegistryPtr = std::shared_ptr<const RegistryState>;

// Everything a snapshot captures. All members are handle-sized: a TopoDS_Shape is a TShape
// handle plus a Location plus an Orientation, and the registry is a shared_ptr. Copying this
// struct is therefore O(1) in the size of the model, which is what makes snapshot/restore
// O(1) at any depth.
struct SessionState {
  TopoDS_Shape root;
  RegistryPtr registry;
  std::int64_t op_index = 0;
};

// Provenance of one issued id. Append-only and session-global: it survives a restore, so a
// name minted on a branch that was later abandoned still resolves — to Lost, which is the
// honest answer, rather than to whatever now occupies that position.
struct Origin {
  std::int64_t op_index = 0;
  Role role = Role::Constructed;
  int ordinal = 0;
  std::vector<EntityId> sources;  // input entities this one was generated from
};

// What one operation did to the id space.
struct Delta {
  std::vector<EntityId> created;
  std::vector<EntityId> deleted;
  std::vector<EntityId> modified;  // survived, but denotes different shape(s) than before
  std::vector<EntityId> split;     // survived, now denotes more than one shape
  std::vector<EntityId> merged;    // survived onto a shape that other ids also denote
};

// ---- Small helpers ------------------------------------------------------------------ //

py::array_t<std::int64_t> ids_array(const std::vector<EntityId>& ids) {
  py::array_t<std::int64_t> out(static_cast<py::ssize_t>(ids.size()));
  std::copy(ids.begin(), ids.end(), out.mutable_data());
  return out;
}

// Flatten a shape into leaf children, descending through nested compounds. The session root
// is kept as a flat compound of bodies so that "which body did this operation consume" is
// answerable without walking a tree.
void explode_into(const TopoDS_Shape& s, std::vector<TopoDS_Shape>& out) {
  if (s.IsNull()) {
    return;
  }
  if (s.ShapeType() == TopAbs_COMPOUND) {
    for (TopoDS_Iterator it(s); it.More(); it.Next()) {
      explode_into(it.Value(), out);
    }
    return;
  }
  out.push_back(s);
}

// Build a fresh compound from the given bodies. Always a new TShape: an operation never
// mutates a compound another state still points at, which is the invariant that makes an
// O(1) snapshot sound.
TopoDS_Shape make_root(const std::vector<TopoDS_Shape>& bodies) {
  TopoDS_Compound c;
  BRep_Builder b;
  b.MakeCompound(c);
  for (const TopoDS_Shape& s : bodies) {
    b.Add(c, s);
  }
  return c;
}

std::vector<TopoDS_Shape> root_bodies(const TopoDS_Shape& root) {
  std::vector<TopoDS_Shape> out;
  explode_into(root, out);
  return out;
}

// Every registered-kind sub-shape of `shape`, in a deterministic traversal order: solids,
// then faces, then edges, then vertices, each in TopExp::MapShapes order. Determinism here
// is load-bearing — it is what makes the ordinal in a name reproducible.
std::vector<TopoDS_Shape> registered_subshapes(const TopoDS_Shape& shape) {
  std::vector<TopoDS_Shape> out;
  if (shape.IsNull()) {
    return out;
  }
  for (TopAbs_ShapeEnum k : kEntityKinds) {
    ShapeSet m;
    TopExp::MapShapes(shape, k, m);
    for (int i = 1; i <= m.Extent(); ++i) {
      out.push_back(m.FindKey(i));
    }
  }
  return out;
}

void append_unique(std::vector<TopoDS_Shape>& dst, const TopoDS_Shape& s) {
  for (const TopoDS_Shape& e : dst) {
    if (e.IsSame(s)) {
      return;
    }
  }
  dst.push_back(s);
}

double measure_of(const TopoDS_Shape& s) {
  GProp_GProps props;
  switch (s.ShapeType()) {
    case TopAbs_SOLID:
      BRepGProp::VolumeProperties(s, props);
      return props.Mass();
    case TopAbs_FACE:
      BRepGProp::SurfaceProperties(s, props);
      return props.Mass();
    case TopAbs_EDGE:
      BRepGProp::LinearProperties(s, props);
      return props.Mass();
    default:
      return 0.0;
  }
}

// Centroid of a shape. For a vertex this is the point itself; GProp's centre of mass is
// undefined for a zero-measure shape, so the vertex case is handled from the bounding box,
// which is exact for a point.
std::array<double, 3> centroid_of(const TopoDS_Shape& s) {
  if (s.ShapeType() == TopAbs_VERTEX) {
    Bnd_Box box;
    BRepBndLib::Add(s, box);
    double a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
    box.Get(a, b, c, d, e, f);
    return {0.5 * (a + d), 0.5 * (b + e), 0.5 * (c + f)};
  }
  GProp_GProps props;
  switch (s.ShapeType()) {
    case TopAbs_SOLID:
      BRepGProp::VolumeProperties(s, props);
      break;
    case TopAbs_FACE:
      BRepGProp::SurfaceProperties(s, props);
      break;
    default:
      BRepGProp::LinearProperties(s, props);
      break;
  }
  const gp_Pnt p = props.CentreOfMass();
  return {p.X(), p.Y(), p.Z()};
}

// Re-entrancy guard. Operations release the GIL, so two threads driving one Session would
// race on the registry. This turns that race into a loud, immediate error at the second
// entry instead of a corrupt registry or a crash. It deliberately guards concurrency rather
// than thread affinity: handing a session to a worker thread is legitimate, using it from
// two at once is not.
class OpGuard {
 public:
  explicit OpGuard(std::atomic<bool>& flag) : flag_(flag) {
    bool expected = false;
    if (!flag_.compare_exchange_strong(expected, true)) {
      throw PysmeshError(
          "Session is already executing an operation. A Session is not thread-safe: "
          "use one Session per thread.");
    }
  }
  ~OpGuard() { flag_.store(false); }
  OpGuard(const OpGuard&) = delete;
  OpGuard& operator=(const OpGuard&) = delete;

 private:
  std::atomic<bool>& flag_;
};

// ---- Session ------------------------------------------------------------------------ //

class Session {
 public:
  explicit Session(bool validate) : validate_(validate) {
    state_.root = make_root({});
    state_.registry = std::make_shared<RegistryState>();
    state_.op_index = 0;
  }

  // ---- construction operations ------------------------------------------------------ //

  py::dict add_brep(const py::bytes& data) {
    OpGuard guard(in_op_);
    const std::string buffer = data;
    TopoDS_Shape imported;
    {
      py::gil_scoped_release release;
      std::istringstream stream(buffer);
      BRep_Builder builder;
      try {
        BRepTools::Read(imported, stream, builder);
      } catch (const std::exception& e) {
        py::gil_scoped_acquire acquire;
        throw PysmeshError(std::string("Session.add_brep: BREP read failed: ") + e.what());
      }
    }
    if (imported.IsNull()) {
      throw PysmeshError(
          "Session.add_brep: BREP read produced a null shape (empty or malformed data).");
    }
    return add_bodies(imported, "add_brep");
  }

  py::dict add_box(double dx, double dy, double dz, double ox, double oy, double oz) {
    OpGuard guard(in_op_);
    require_positive("dx", dx);
    require_positive("dy", dy);
    require_positive("dz", dz);
    TopoDS_Shape solid;
    {
      py::gil_scoped_release release;
      solid = BRepPrimAPI_MakeBox(gp_Pnt(ox, oy, oz), dx, dy, dz).Shape();
    }
    return add_bodies(solid, "add_box");
  }

  py::dict add_cylinder(double radius, double height, double ox, double oy, double oz,
                        double ax, double ay, double az) {
    OpGuard guard(in_op_);
    require_positive("radius", radius);
    require_positive("height", height);
    const gp_Vec axis(ax, ay, az);
    if (axis.Magnitude() <= 0.0) {
      throw PysmeshError("Session.add_cylinder: axis must be a non-zero vector.");
    }
    TopoDS_Shape solid;
    {
      py::gil_scoped_release release;
      const gp_Ax2 frame(gp_Pnt(ox, oy, oz), gp_Dir(axis));
      solid = BRepPrimAPI_MakeCylinder(frame, radius, height).Shape();
    }
    return add_bodies(solid, "add_cylinder");
  }

  // ---- modelling operations --------------------------------------------------------- //

  py::dict fuse(const std::vector<EntityId>& targets, const std::vector<EntityId>& tools,
                double fuzzy, bool parallel) {
    OpGuard guard(in_op_);
    if (targets.empty()) {
      throw PysmeshError("Session.fuse: targets must name at least one solid.");
    }
    if (tools.empty()) {
      throw PysmeshError("Session.fuse: tools must name at least one solid.");
    }
    if (fuzzy < 0.0) {
      throw PysmeshError("Session.fuse: fuzzy must be >= 0 (got " + std::to_string(fuzzy) +
                         ").");
    }

    NCollection_List<TopoDS_Shape> args = solids_of("fuse", "targets", targets);
    NCollection_List<TopoDS_Shape> tls = solids_of("fuse", "tools", tools);

    // Bodies the boolean does not consume pass straight through and keep their identity.
    std::vector<TopoDS_Shape> consumed;
    for (NCollection_List<TopoDS_Shape>::Iterator it(args); it.More(); it.Next()) {
      append_unique(consumed, it.Value());
    }
    for (NCollection_List<TopoDS_Shape>::Iterator it(tls); it.More(); it.Next()) {
      append_unique(consumed, it.Value());
    }
    const std::vector<TopoDS_Shape> survivors = bodies_excluding(consumed);

    TopoDS_Shape result;
    Handle(BRepTools_History) hist;
    std::string errors;
    {
      py::gil_scoped_release release;
      BRepAlgoAPI_Fuse op;
      op.SetArguments(args);
      op.SetTools(tls);
      // The history IS the naming substrate, not a diagnostic: without it every id in the
      // consumed bodies would die at this operation.
      op.SetToFillHistory(true);
      op.SetRunParallel(parallel);
      if (fuzzy > 0.0) {
        op.SetFuzzyValue(fuzzy);
      }
      try {
        op.Build();
      } catch (const std::exception& e) {
        py::gil_scoped_acquire acquire;
        throw PysmeshError(std::string("Session.fuse: BRepAlgoAPI_Fuse::Build failed: ") +
                           e.what());
      }
      if (!op.IsDone() || op.HasErrors()) {
        std::ostringstream s;
        op.DumpErrors(s);
        errors = s.str();
      } else {
        result = op.Shape();
        hist = op.History();
      }
    }
    if (!errors.empty() || result.IsNull()) {
      throw PysmeshError("Session.fuse: the boolean failed; no partial result is returned.",
                         errors, {});
    }
    return commit(concat(survivors, result), hist, "fuse", result);
  }

  py::dict fillet(const std::vector<EntityId>& edge_ids, double radius) {
    OpGuard guard(in_op_);
    if (edge_ids.empty()) {
      throw PysmeshError("Session.fillet: edge_ids must name at least one edge.");
    }
    require_positive("radius", radius);

    std::vector<TopoDS_Shape> edges;
    for (EntityId id : edge_ids) {
      const EntityRecord& rec = require_alive("fillet", id);
      if (rec.kind != TopAbs_EDGE) {
        throw PysmeshError("Session.fillet: entity " + std::to_string(id) + " is a " +
                           kind_name(rec.kind) + ", not an EDGE.");
      }
      if (rec.shapes.size() != 1) {
        throw PysmeshError("Session.fillet: entity " + std::to_string(id) +
                           " denotes " + std::to_string(rec.shapes.size()) +
                           " edges (it was split); name one of them instead.");
      }
      edges.push_back(rec.shapes.front());
    }

    // OCCT's fillet takes edges and derives the owning solid itself, so the caller never has
    // to co-select the solid or a reference face per edge. Every named edge must belong to
    // the same body, because one fillet operation builds one shape.
    const TopoDS_Shape owner = sole_owner_body(body_of_subshape(), edges);
    const std::vector<TopoDS_Shape> survivors = bodies_excluding({owner});

    TopoDS_Shape result;
    Handle(BRepTools_History) hist;
    {
      py::gil_scoped_release release;
      BRepFilletAPI_MakeFillet mk(owner);
      for (const TopoDS_Shape& e : edges) {
        mk.Add(radius, TopoDS::Edge(e));
      }
      try {
        mk.Build();
      } catch (const std::exception& e) {
        py::gil_scoped_acquire acquire;
        throw PysmeshError(
            std::string("Session.fillet: BRepFilletAPI_MakeFillet::Build failed: ") +
            e.what());
      }
      if (mk.IsDone()) {
        result = mk.Shape();
        NCollection_List<TopoDS_Shape> in;
        in.Append(owner);
        hist = new BRepTools_History(in, mk);
      }
    }
    if (result.IsNull()) {
      std::vector<int> bad;
      bad.reserve(edge_ids.size());
      for (EntityId id : edge_ids) {
        bad.push_back(static_cast<int>(id));
      }
      throw PysmeshError(
          "Session.fillet: OCCT could not build a fillet of radius " +
              std::to_string(radius) + " on the named edges.",
          "BRepFilletAPI_MakeFillet::IsDone() is false. The radius is most likely larger "
          "than the local geometry admits.",
          bad);
    }
    return commit(concat(survivors, result), hist, "fillet", result);
  }

  // ---- transforms ------------------------------------------------------------------- //

  py::dict translate(double dx, double dy, double dz,
                     const std::optional<std::vector<EntityId>>& entity_ids) {
    OpGuard guard(in_op_);
    gp_Trsf t;
    t.SetTranslation(gp_Vec(dx, dy, dz));
    return relocate(t, entity_ids, "translate");
  }

  py::dict rotate(double ox, double oy, double oz, double ax, double ay, double az,
                  double angle_rad, const std::optional<std::vector<EntityId>>& entity_ids) {
    OpGuard guard(in_op_);
    const gp_Vec axis(ax, ay, az);
    if (axis.Magnitude() <= 0.0) {
      throw PysmeshError("Session.rotate: axis must be a non-zero vector.");
    }
    gp_Trsf t;
    t.SetRotation(gp_Ax1(gp_Pnt(ox, oy, oz), gp_Dir(axis)), angle_rad);
    return relocate(t, entity_ids, "rotate");
  }

  // ---- snapshot / restore ----------------------------------------------------------- //

  // Guarded like an operation, not like a query. These mutate the session, and an operation
  // that has released the GIL is still in flight: a restore landing in that window would
  // swap the state out from under it. A read-only query needs no guard, because the state a
  // running operation publishes is only ever written while the GIL is held.
  std::int64_t snapshot() {
    OpGuard guard(in_op_);
    snapshots_.push_back(state_);
    return static_cast<std::int64_t>(snapshots_.size()) - 1;
  }

  void restore(std::int64_t mark) {
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

  void discard_snapshot(std::int64_t mark) {
    OpGuard guard(in_op_);
    if (mark < 0 || static_cast<std::size_t>(mark) >= snapshots_.size()) {
      throw PysmeshError("Session.discard_snapshot: unknown snapshot mark " +
                         std::to_string(mark) + ".");
    }
    snapshots_[static_cast<std::size_t>(mark)].reset();
  }

  std::int64_t snapshot_count() const {
    std::int64_t n = 0;
    for (const std::optional<SessionState>& s : snapshots_) {
      if (s.has_value()) {
        ++n;
      }
    }
    return n;
  }

  // ---- queries ---------------------------------------------------------------------- //

  py::array_t<std::int64_t> entities(const std::string& kind) const {
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

  std::string entity_kind(EntityId id) const {
    return kind_name(require_alive("entity_kind", id).kind);
  }

  // "alive" or "dead". An id that was never issued is a caller error, not a state: it
  // raises, which is how a positional ordinal from the stateless API fails loudly here
  // instead of silently denoting somebody else's entity.
  std::string entity_state(EntityId id) const {
    require_issued("entity_state", id);
    return state_.registry->alive.count(id) != 0 ? "alive" : "dead";
  }

  std::int64_t shape_count(EntityId id) const {
    return static_cast<std::int64_t>(require_alive("shape_count", id).shapes.size());
  }

  // Bulk geometry of every alive entity of one kind, as parallel arrays. Bulk rather than
  // per-entity objects because the calling convention has to stay vectorised for a model
  // with tens of thousands of faces.
  py::dict entity_table(const std::string& kind) const {
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

  py::bytes brep() const {
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
  py::dict name_of(EntityId id) const {
    require_alive("name_of", id);
    const Origin& o = origins_.at(id);
    py::dict out;
    out["op_index"] = o.op_index;
    out["role"] = static_cast<int>(o.role);
    out["ordinal"] = o.ordinal;
    return out;
  }

  py::dict origin(EntityId id) const {
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
  py::dict resolve(std::int64_t op_index, int role, int ordinal) const {
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

  // ---- introspection ---------------------------------------------------------------- //

  std::int64_t op_count() const { return next_op_ - 1; }
  std::int64_t state_op_index() const { return state_.op_index; }
  std::int64_t issued_id_count() const { return next_id_ - 1; }
  std::int64_t entity_count() const {
    return static_cast<std::int64_t>(state_.registry->alive.size());
  }

  // Test hook. Falsification of the naming suite is mandatory: a history test that has never
  // been shown to fail is a claim, not a check. This drops the next operation's history so
  // the ids in the consumed bodies die instead of being carried forward, which is precisely
  // the "one delta dropped" tear the identity gate must detect.
  void debug_tear_next_history() { tear_next_history_ = true; }

 private:
  // ---- validation helpers ----------------------------------------------------------- //

  static void require_positive(const char* name, double v) {
    if (!(v > 0.0)) {
      throw PysmeshError(std::string("Session: ") + name + " must be > 0 (got " +
                         std::to_string(v) + ").");
    }
  }

  void require_issued(const char* op, EntityId id) const {
    if (id < 1 || id >= next_id_) {
      throw PysmeshError(std::string("Session.") + op + ": " + std::to_string(id) +
                         " is not an EntityId this session ever issued (issued 1.." +
                         std::to_string(next_id_ - 1) +
                         "). Session ids are not the positional ordinals the stateless "
                         "API returns.");
    }
  }

  const EntityRecord& require_alive(const char* op, EntityId id) const {
    require_issued(op, id);
    const auto it = state_.registry->alive.find(id);
    if (it == state_.registry->alive.end()) {
      throw PysmeshError(std::string("Session.") + op + ": entity " + std::to_string(id) +
                         " is dead (it was removed by an earlier operation).");
    }
    return it->second;
  }

  NCollection_List<TopoDS_Shape> solids_of(const char* op, const char* argname,
                                           const std::vector<EntityId>& ids) const {
    NCollection_List<TopoDS_Shape> out;
    for (EntityId id : ids) {
      const EntityRecord& rec = require_alive(op, id);
      if (rec.kind != TopAbs_SOLID) {
        throw PysmeshError(std::string("Session.") + op + ": " + argname + " entity " +
                           std::to_string(id) + " is a " + kind_name(rec.kind) +
                           ", not a SOLID.");
      }
      for (const TopoDS_Shape& s : rec.shapes) {
        out.Append(s);
      }
    }
    return out;
  }

  // ---- root bookkeeping ------------------------------------------------------------- //

  std::vector<TopoDS_Shape> bodies_excluding(const std::vector<TopoDS_Shape>& drop) const {
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

  static std::vector<TopoDS_Shape> concat(const std::vector<TopoDS_Shape>& keep,
                                          const TopoDS_Shape& added) {
    std::vector<TopoDS_Shape> out = keep;
    explode_into(added, out);
    return out;
  }

  // sub-shape -> owning body, for every body of the current root. Built once per call so
  // that resolving many entities to their bodies stays linear rather than quadratic.
  ShapeKeyed<TopoDS_Shape> body_of_subshape() const {
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
  static TopoDS_Shape sole_owner_body(const ShapeKeyed<TopoDS_Shape>& owners,
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

  py::dict add_bodies(const TopoDS_Shape& added, const char* op_name) {
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
  py::dict commit(const std::vector<TopoDS_Shape>& bodies,
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

  Delta carry_registry(const TopoDS_Shape& new_root, const Handle(BRepTools_History) & hist,
                       std::int64_t op_index) {
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
      std::vector<TopoDS_Shape> successors;
      for (const TopoDS_Shape& s : rec.shapes) {
        if (!hist.IsNull()) {
          for (const TopoDS_Shape& g : hist->Generated(s)) {
            generated_from[g].push_back(id);
          }
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
  py::dict relocate(const gp_Trsf& trsf,
                    const std::optional<std::vector<EntityId>>& entity_ids,
                    const char* op_name) {
    const std::vector<TopoDS_Shape> bodies = root_bodies(state_.root);

    // Which bodies move: all of them, or the ones owning the named entities.
    std::vector<TopoDS_Shape> moving;
    if (entity_ids.has_value()) {
      if (entity_ids->empty()) {
        throw PysmeshError(std::string("Session.") + op_name +
                           ": entity_ids was given but is empty; pass None to move the "
                           "whole model.");
      }
      const ShapeKeyed<TopoDS_Shape> owners = body_of_subshape();
      for (EntityId id : *entity_ids) {
        const EntityRecord& rec = require_alive(op_name, id);
        append_unique(moving, sole_owner_body(owners, rec.shapes));
      }
      // A body sharing sub-shapes with a body that stays put cannot be moved on its own
      // without tearing the model, so that is refused rather than silently split.
      guard_shared_subshapes(bodies, moving, op_name);
    } else {
      moving = bodies;
    }

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

  void guard_shared_subshapes(const std::vector<TopoDS_Shape>& all,
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

  static void finalise(Delta& d) {
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

  static py::dict delta_dict(const Delta& d, std::int64_t op_index, const char* op_name) {
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

  // ---- state ------------------------------------------------------------------------ //

  bool validate_;

  // INVARIANT (load-bearing, and the reason read-only queries need no lock): state_ is only
  // ever written while the GIL is held. An operation releases the GIL exclusively around the
  // OCCT algorithm, which works on local shapes; the registry is carried and published
  // afterwards, under the GIL. A refactor that moved registry work into a released-GIL block
  // would have to re-examine every query on this class.
  SessionState state_;

  // Session-global and monotonic. Deliberately NOT part of SessionState, so restoring an
  // earlier state does not rewind them: if it did, a later operation would re-issue an id
  // (or an op index) that the abandoned branch already used, and a reference held from that
  // branch would resolve to a different entity — the exact failure ids exist to prevent.
  EntityId next_id_ = 1;
  std::int64_t next_op_ = 1;

  // Append-only provenance. Survives restore, so a name minted on an abandoned branch still
  // resolves — to lost.
  std::unordered_map<EntityId, Origin> origins_;
  std::map<std::tuple<std::int64_t, int, int>, EntityId> name_index_;

  std::vector<std::optional<SessionState>> snapshots_;
  std::atomic<bool> in_op_{false};
  bool tear_next_history_ = false;
};

}  // namespace

void bind_session(py::module_& m) {
  py::class_<Session>(m, "Session")
      .def(py::init<bool>(), py::arg("validate"))
      .def("add_brep", &Session::add_brep, py::arg("data"))
      .def("add_box", &Session::add_box, py::arg("dx"), py::arg("dy"), py::arg("dz"),
           py::arg("ox"), py::arg("oy"), py::arg("oz"))
      .def("add_cylinder", &Session::add_cylinder, py::arg("radius"), py::arg("height"),
           py::arg("ox"), py::arg("oy"), py::arg("oz"), py::arg("ax"), py::arg("ay"),
           py::arg("az"))
      .def("fuse", &Session::fuse, py::arg("targets"), py::arg("tools"), py::arg("fuzzy"),
           py::arg("parallel"))
      .def("fillet", &Session::fillet, py::arg("edge_ids"), py::arg("radius"))
      .def("translate", &Session::translate, py::arg("dx"), py::arg("dy"), py::arg("dz"),
           py::arg("entity_ids"))
      .def("rotate", &Session::rotate, py::arg("ox"), py::arg("oy"), py::arg("oz"),
           py::arg("ax"), py::arg("ay"), py::arg("az"), py::arg("angle_rad"),
           py::arg("entity_ids"))
      .def("snapshot", &Session::snapshot)
      .def("restore", &Session::restore, py::arg("mark"))
      .def("discard_snapshot", &Session::discard_snapshot, py::arg("mark"))
      .def("snapshot_count", &Session::snapshot_count)
      .def("entities", &Session::entities, py::arg("kind"))
      .def("entity_kind", &Session::entity_kind, py::arg("entity_id"))
      .def("entity_state", &Session::entity_state, py::arg("entity_id"))
      .def("shape_count", &Session::shape_count, py::arg("entity_id"))
      .def("entity_table", &Session::entity_table, py::arg("kind"))
      .def("brep", &Session::brep)
      .def("name_of", &Session::name_of, py::arg("entity_id"))
      .def("origin", &Session::origin, py::arg("entity_id"))
      .def("resolve", &Session::resolve, py::arg("op_index"), py::arg("role"),
           py::arg("ordinal"))
      .def("op_count", &Session::op_count)
      .def("state_op_index", &Session::state_op_index)
      .def("issued_id_count", &Session::issued_id_count)
      .def("entity_count", &Session::entity_count)
      .def("_debug_tear_next_history", &Session::debug_tear_next_history,
           "Test hook: drop the NEXT operation's history so its input ids die instead of "
           "being carried forward. Exists so the identity suite can be shown to fail.");
}

}  // namespace pysmesh
