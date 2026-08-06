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
// This header is the declaration. The implementation is split by area, because one file
// carrying every operation stops being navigable long before it stops compiling:
//
//   session_core.cpp       — the id registry and its carry across an operation, snapshots,
//                            state queries, names, and the resolution helpers the other
//                            units share;
//   session_construct.cpp  — primitives, curve and surface construction, sweeps;
//   session_boolean.cpp    — the boolean family, fillet and chamfer;
//   session_transform.cpp  — the relocation and rebuild transform paths, and copy;
//   session_heal.cpp       — healing, sewing, defeaturing, imprinting and removal;
//   session_query.cpp      — the geometric query surface over the live shape;
//   session_tessellate.cpp — the render mesh, and the incremental delta over it;
//   session_handoff.cpp    — the export to a mesher, and the id-to-ordinal bijection;
//   session_progress.cpp   — the progress/cancellation driver (declared in progress.hpp,
//                            which is its own header because it is not part of this class);
//   session_bind.cpp       — the pybind11 surface.
//
// What is defined here rather than in a translation unit is defined here for one of three
// reasons: it is a template, a one-line accessor, or a data member.
//
// src/pysmesh/session.py wraps all of it in frozen dataclasses and distinct id types, which
// is where the public surface is documented.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <BOPAlgo_GlueEnum.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_BuilderAlgo.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Defeaturing.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepAlgoAPI_Section.hxx>
#include <BRepAlgoAPI_Splitter.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeSolid.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepFilletAPI_MakeChamfer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepOffsetAPI_MakeFilling.hxx>
#include <BRepOffsetAPI_MakePipe.hxx>
#include <BRepOffsetAPI_MakePipeShell.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCone.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakeTorus.hxx>
#include <BRepPrimAPI_MakeWedge.hxx>
#include <BRepTools.hxx>
#include <BRepTools_History.hxx>
#include <BRepTools_ReShape.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GC_MakeArcOfCircle.hxx>
#include <GProp_GProps.hxx>
#include <GeomAPI_PointsToBSpline.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <GeomAbs_CurveType.hxx>
#include <GeomAbs_Shape.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <GeomLProp_SLProps.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_Circle.hxx>
#include <Geom_Surface.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <HelixBRep_BuilderHelix.hxx>
#include <IMeshTools_Parameters.hxx>
#include <NCollection_Array1.hxx>
#include <NCollection_IndexedDataMap.hxx>
#include <NCollection_IndexedMap.hxx>
#include <NCollection_List.hxx>
#include <Poly_Polygon3D.hxx>
#include <Poly_PolygonOnTriangulation.hxx>
#include <Poly_Triangulation.hxx>
#include <Precision.hxx>
#include <ShapeBuild_ReShape.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeUpgrade_RemoveInternalWires.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <Standard_Handle.hxx>
#include <TopAbs.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopAbs_State.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Ax3.hxx>
#include <gp_Circ.hxx>
#include <gp_Dir.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Mat.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Trsf.hxx>
#include <gp_Vec.hxx>
#include <gp_XYZ.hxx>

#include "common.hpp"
#include "progress.hpp"

namespace pysmesh {
namespace session {


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

inline const char* kind_name(TopAbs_ShapeEnum k) {
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

inline TopAbs_ShapeEnum kind_from_name(const std::string& name) {
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

// What an operation does about a result BRepCheck_Analyzer rejects.
//
// Strict is the rule for every operation that assumes valid input: an invalid result is a
// failure, the session is left untouched, and the caller is told. The healing family is the
// deliberate exception — its input is invalid by assumption, so refusing to commit a shape
// that is *less* invalid than before would make the operations useless on exactly the shapes
// they exist for. Those report the verdict on the delta instead of raising, and the caller
// decides whether the improvement was enough.
enum class Validation : int { Strict = 0, Report = 1 };

// What one face contributed to the previous render mesh: which triangulation it carried and
// where that triangulation sat.
//
// The two are separate on purpose, because they answer different questions and an operation
// can change one without the other. A rebuilt face gets a new Poly_Triangulation. A face that
// was merely *relocated* keeps its triangulation — OCCT stores the nodes in the face's own
// frame — but every node the render mesh emits for it moves. A consumer holding derived data
// needs the second; a consumer that could reuse its buffers under a rigid motion needs to
// tell the two apart. Reporting only one would either force a full rebuild or leave a stale
// render, and those are exactly the two failure directions the render-mesh contract names.
//
// `face` is held rather than dropped: it keeps the TShape the record is keyed by alive, so a
// freed TShape's address can never be reused by a later face and read as "unchanged".
// `triangulation` is held for the same reason.
struct EmittedFace {
  TopoDS_Shape face;
  Handle(Poly_Triangulation) triangulation;
  TopLoc_Location location;
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

  // The BRepCheck_Analyzer verdict on the shape this operation built, when one was taken.
  // Empty means the operation built nothing to check, or the session runs unvalidated.
  std::optional<bool> valid;
};

// ---- Small helpers ------------------------------------------------------------------ //

inline py::array_t<std::int64_t> ids_array(const std::vector<EntityId>& ids) {
  py::array_t<std::int64_t> out(static_cast<py::ssize_t>(ids.size()));
  std::copy(ids.begin(), ids.end(), out.mutable_data());
  return out;
}

// Flatten a shape into leaf children, descending through nested compounds. The session root
// is kept as a flat compound of bodies so that "which body did this operation consume" is
// answerable without walking a tree.
inline void explode_into(const TopoDS_Shape& s, std::vector<TopoDS_Shape>& out) {
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
inline TopoDS_Shape make_root(const std::vector<TopoDS_Shape>& bodies) {
  TopoDS_Compound c;
  BRep_Builder b;
  b.MakeCompound(c);
  for (const TopoDS_Shape& s : bodies) {
    b.Add(c, s);
  }
  return c;
}

inline std::vector<TopoDS_Shape> root_bodies(const TopoDS_Shape& root) {
  std::vector<TopoDS_Shape> out;
  explode_into(root, out);
  return out;
}

// Every registered-kind sub-shape of `shape`, in a deterministic traversal order: solids,
// then faces, then edges, then vertices, each in TopExp::MapShapes order. Determinism here
// is load-bearing — it is what makes the ordinal in a name reproducible.
inline std::vector<TopoDS_Shape> registered_subshapes(const TopoDS_Shape& shape) {
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

inline void append_unique(std::vector<TopoDS_Shape>& dst, const TopoDS_Shape& s) {
  for (const TopoDS_Shape& e : dst) {
    if (e.IsSame(s)) {
      return;
    }
  }
  dst.push_back(s);
}

inline double measure_of(const TopoDS_Shape& s) {
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
inline std::array<double, 3> centroid_of(const TopoDS_Shape& s) {
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

// A caller-supplied point list: (N, 3) float64, C-contiguous. Forcecast so a list of tuples
// or a float32 array is accepted without the caller having to convert.
using PointArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

inline std::vector<gp_Pnt> points_of(const char* op, const char* argname,
                                     const PointArray& a, py::ssize_t min_count) {
  if (a.ndim() != 2 || a.shape(1) != 3) {
    throw PysmeshError(std::string("Session.") + op + ": " + argname +
                       " must be an (N, 3) array of points.");
  }
  if (a.shape(0) < min_count) {
    throw PysmeshError(std::string("Session.") + op + ": " + argname + " needs at least " +
                       std::to_string(min_count) + " points (got " +
                       std::to_string(a.shape(0)) + ").");
  }
  const double* p = a.data();
  std::vector<gp_Pnt> out;
  out.reserve(static_cast<std::size_t>(a.shape(0)));
  for (py::ssize_t i = 0; i < a.shape(0); ++i) {
    out.emplace_back(p[3 * i + 0], p[3 * i + 1], p[3 * i + 2]);
  }
  return out;
}

// A direction from raw components, rejecting the zero vector loudly. gp_Dir's own
// constructor raises Standard_ConstructionError, which would surface as an opaque OCCT
// exception rather than a message naming the argument.
inline gp_Dir direction_of(const char* op, const char* argname, double x, double y, double z) {
  const gp_Vec v(x, y, z);
  if (v.Magnitude() <= 0.0) {
    throw PysmeshError(std::string("Session.") + op + ": " + argname +
                       " must be a non-zero vector.");
  }
  return gp_Dir(v);
}

inline gp_Ax2 frame_of(const char* op, double ox, double oy, double oz, double ax,
                       double ay, double az) {
  return gp_Ax2(gp_Pnt(ox, oy, oz), direction_of(op, "axis", ax, ay, az));
}

// Canonical name of a face's underlying geometry. The spelling matches the stateless API's
// FaceInfo.surface_type exactly, so a consumer that switches from one to the other does not
// have to re-learn the vocabulary.
inline const char* surface_type_name(GeomAbs_SurfaceType t) {
  switch (t) {
    case GeomAbs_Plane:
      return "Plane";
    case GeomAbs_Cylinder:
      return "Cylinder";
    case GeomAbs_Cone:
      return "Cone";
    case GeomAbs_Sphere:
      return "Sphere";
    case GeomAbs_Torus:
      return "Torus";
    case GeomAbs_BezierSurface:
      return "Bezier";
    case GeomAbs_BSplineSurface:
      return "BSpline";
    case GeomAbs_SurfaceOfRevolution:
      return "Revolution";
    case GeomAbs_SurfaceOfExtrusion:
      return "Extrusion";
    case GeomAbs_OffsetSurface:
      return "Offset";
    case GeomAbs_OtherSurface:
      return "Other";
  }
  return "Other";
}

inline const char* curve_type_name(GeomAbs_CurveType t) {
  switch (t) {
    case GeomAbs_Line:
      return "Line";
    case GeomAbs_Circle:
      return "Circle";
    case GeomAbs_Ellipse:
      return "Ellipse";
    case GeomAbs_Hyperbola:
      return "Hyperbola";
    case GeomAbs_Parabola:
      return "Parabola";
    case GeomAbs_BezierCurve:
      return "Bezier";
    case GeomAbs_BSplineCurve:
      return "BSpline";
    case GeomAbs_OffsetCurve:
      return "Offset";
    case GeomAbs_OtherCurve:
      return "Other";
  }
  return "Other";
}

// The larger of the two principal curvatures in magnitude.
//
// This is not a convenience wrapper. MaxCurvature() and MinCurvature() return the SIGNED
// principal curvatures ordered by value, not by magnitude: on an outward-normal cylinder of
// radius R they are (0, -1/R), so MaxCurvature() alone reads 0 and a curvature map keyed on
// it reports every cylinder as flat. They are also non-const and mutate cached derivative
// state, so each must be read exactly once into a local — reading either twice in one
// expression does not give the same answer as reading it once.
inline double peak_curvature(BRepLProp_SLProps& props) {
  const double k1 = props.MaxCurvature();
  const double k2 = props.MinCurvature();
  return std::max(std::abs(k1), std::abs(k2));
}

// A rigid transform applied with Copy = false only changes a shape's Location, which is what
// makes the id carry structural rather than history-borne. OCCT admits that shortcut for a
// *direct isometry* alone: BRepBuilderAPI_Transform rebuilds the geometry for anything else,
// and TopLoc_Datum3D refuses to be constructed from it at all. The two conditions are the
// same one, so this predicate decides which of the two transform paths an operation takes.
inline bool is_location_only(const gp_Trsf& t) {
  return !t.IsNegative() && std::abs(std::abs(t.ScaleFactor()) - 1.0) <= 1e-12;
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
  explicit Session(bool validate);

  // ---- construction operations ------------------------------------------------------ //

  // ---- the long-operation contract --------------------------------------------------- //
  //
  // Every operation below that can exceed a few hundred milliseconds ends with the same two
  // arguments: `progress`, a callable taking the fraction done, and `cancel`, a predicate
  // asked whether to stop. Either may be None. Both are consulted from a helper thread while
  // the operation runs with the GIL released — see session/progress.hpp for why they cannot
  // be called from OCCT's own hooks.
  //
  // A cancelled operation raises CancelledError and commits nothing. That is decided by the
  // driver's own flag, never by the algorithm's reporting: a cancelled ShapeFix_Shape hands
  // back a non-null shape carrying a fraction of the model's faces, and committing it would
  // be exactly the partial result the contract forbids.
  //
  // Two repair operations deliberately have no such arguments — unify_same_domain and
  // remove_internal_wires — because OCCT 8.0 gives ShapeUpgrade_UnifySameDomain::Build and
  // ShapeUpgrade_RemoveInternalWires::Perform no Message_ProgressRange to hand them. They
  // take neither rather than accepting one and ignoring it.

  py::dict add_brep(const py::bytes& data, const py::object& progress,
                    const py::object& cancel);

  py::dict add_box(double dx, double dy, double dz, double ox, double oy, double oz);

  py::dict add_cylinder(double radius, double height, double ox, double oy, double oz,
                        double ax, double ay, double az);

  py::dict add_cone(double radius1, double radius2, double height, double ox, double oy,
                    double oz, double ax, double ay, double az, double angle_rad);

  py::dict add_sphere(double radius, double cx, double cy, double cz, double ax, double ay,
                      double az, double angle_rad);

  py::dict add_torus(double radius1, double radius2, double ox, double oy, double oz,
                     double ax, double ay, double az, double angle_rad);

  // A STEP right angular wedge: a box of dx * dy * dz whose face at y = dy is narrowed to
  // ltx along x. ltx == dx is a plain box; ltx == 0 is a wedge with a knife edge.
  py::dict add_wedge(double dx, double dy, double dz, double ltx, double ox, double oy,
                     double oz, double ax, double ay, double az);

  // ---- construction geometry -------------------------------------------------------- //
  //
  // These add curve and surface bodies to the root. The registry tracks SOLID/FACE/EDGE/
  // VERTEX only (that set is fixed by BRepTools_History), so a WIRE body carries no id of
  // its own: it is named through the ids of its edges, and every operation that consumes a
  // profile resolves the named entities to the single body that owns them.

  py::dict add_line(double x1, double y1, double z1, double x2, double y2, double z2);

  // Three-point arc: through p1, ending at p3, passing through p2.
  py::dict add_arc(double x1, double y1, double z1, double x2, double y2, double z2,
                   double x3, double y3, double z3);

  py::dict add_circle(double cx, double cy, double cz, double nx, double ny, double nz,
                      double radius);

  // A polyline through the given points. Unlike make_wire this shares one vertex between
  // consecutive segments by construction, so no edge is ever rebuilt to connect it.
  py::dict add_polyline(const PointArray& points, bool closed);

  // A B-spline approximating the given points to within tol. This is the "spline through
  // points" construction; add_bspline takes control points instead.
  py::dict add_spline(const PointArray& points, int degree_min, int degree_max, double tol);

  // A clamped, uniformly-knotted B-spline over the given control points. The degree is
  // clamped to len(poles) - 1, because a higher degree has no valid knot vector.
  py::dict add_bspline(const PointArray& poles, int degree);

  // A helical wire. TKHelix's HelixBRep_BuilderHelix is the only helix facade OCCT 8.0 has;
  // there is no BRepPrimAPI-style one. It approximates, so the result is a B-spline wire
  // whose deviation from the exact helix is bounded by tol.
  py::dict add_helix(double cx, double cy, double cz, double ax, double ay, double az,
                     double diameter, double pitch, double turns, double tol);

  // A planar rectangular face, dx by dy in the frame's own x/y directions.
  py::dict add_rectangle(double ox, double oy, double oz, double nx, double ny, double nz,
                         double dx, double dy);

  // Join loose edges and wires into one wire, consuming them.
  py::dict make_wire(const std::vector<EntityId>& edge_ids);

  // A planar face bounded by the named edges, consuming them.
  py::dict make_face(const std::vector<EntityId>& edge_ids);

  // A surface filling the named boundary edges, consuming them. Unlike make_face this
  // handles a non-planar boundary; the surface is an approximation, so the result's edges
  // are new geometry and the boundary edges' ids die.
  py::dict make_filling(const std::vector<EntityId>& edge_ids, const py::object& progress,
                        const py::object& cancel);

  // ---- sweeps ----------------------------------------------------------------------- //
  //
  // Each sweep consumes the profile body and raises its dimension: an edge sweeps to a
  // face, a wire to a shell, a face to a solid. The profile survives inside the result as
  // the sweep's first shape, so its entity ids carry through structurally, and the walls
  // the sweep generates are named against the profile edges they came from.

  py::dict extrude(const std::vector<EntityId>& entity_ids, double vx, double vy,
                   double vz);

  py::dict revolve(const std::vector<EntityId>& entity_ids, double ox, double oy, double oz,
                   double ax, double ay, double az, double angle_rad);

  py::dict pipe(const std::vector<EntityId>& spine_ids,
                const std::vector<EntityId>& profile_ids, const py::object& progress,
                const py::object& cancel);

  // The general sweep. Unlike pipe it exposes the frame law (Frenet vs corrected Frenet)
  // and can close the shell into a solid, which is what a swept CFD body normally needs.
  py::dict pipe_shell(const std::vector<EntityId>& spine_ids,
                      const std::vector<EntityId>& profile_ids, bool frenet, bool solid,
                      const py::object& progress, const py::object& cancel);

  // Loft through an ordered list of section wires, consuming all of them.
  py::dict thru_sections(const std::vector<std::vector<EntityId>>& sections, bool solid,
                         bool ruled, const py::object& progress, const py::object& cancel);

  // ---- modelling operations --------------------------------------------------------- //

  py::dict fuse(const std::vector<EntityId>& targets, const std::vector<EntityId>& tools,
                double fuzzy, bool parallel, const py::object& progress,
                const py::object& cancel);

  py::dict cut(const std::vector<EntityId>& targets, const std::vector<EntityId>& tools,
               double fuzzy, bool parallel, const py::object& progress,
               const py::object& cancel);

  py::dict common(const std::vector<EntityId>& targets, const std::vector<EntityId>& tools,
                  double fuzzy, bool parallel, const py::object& progress,
                  const py::object& cancel);

  // The section curves of targets against tools. Additive: the result of a section is the
  // intersection geometry alone, so both operand groups stay in the model and only the
  // section's vertices and edges are added.
  py::dict section(const std::vector<EntityId>& targets, const std::vector<EntityId>& tools,
                   double fuzzy, bool parallel, const py::object& progress,
                   const py::object& cancel);

  // Split the targets by the tools. The tools are not consumed: OCCT's Splitter excludes
  // their split parts from the result, and a tool the caller still holds ids for must not
  // disappear from the model as a side effect.
  py::dict split(const std::vector<EntityId>& targets, const std::vector<EntityId>& tools,
                 double fuzzy, bool parallel, const py::object& progress,
                 const py::object& cancel);

  // The general fuse: every operand is split by every other and the result keeps all the
  // pieces. This is the operation a conformal multi-body CFD domain is built with.
  py::dict fragment(const std::vector<EntityId>& entity_ids, double fuzzy, bool parallel,
                    const py::object& progress, const py::object& cancel);

  // ---- fillet and chamfer ----------------------------------------------------------- //

  // radius_end, when given, makes the radius evolve linearly along each named edge from
  // radius to radius_end (OCCT's two-radius Add).
  py::dict fillet(const std::vector<EntityId>& edge_ids, double radius,
                  const std::optional<double>& radius_end, const py::object& progress,
                  const py::object& cancel);

  // distance_end + face_id give OCCT's two-distance chamfer, where the first distance is
  // measured on the named reference face. That is the only form in OCCT 8.0 that takes a
  // face at all — there is no (distance, edge, face) overload.
  py::dict chamfer(const std::vector<EntityId>& edge_ids, double distance,
                   const std::optional<double>& distance_end,
                   const std::optional<EntityId>& face_id, const py::object& progress,
                   const py::object& cancel);

  // ---- transforms ------------------------------------------------------------------- //

  py::dict translate(double dx, double dy, double dz,
                     const std::optional<std::vector<EntityId>>& entity_ids);

  py::dict rotate(double ox, double oy, double oz, double ax, double ay, double az,
                  double angle_rad, const std::optional<std::vector<EntityId>>& entity_ids);

  // Reflection in the plane through `point` with the given normal. A plane mirror has
  // determinant -1, so OCCT rebuilds rather than relocating; ids are carried by the
  // transform's own history and all of them survive.
  py::dict mirror(double px, double py_, double pz, double nx, double ny, double nz,
                  const std::optional<std::vector<EntityId>>& entity_ids);

  // Scale about `centre`. A uniform factor stays a gp_Trsf, which keeps analytic surfaces
  // analytic; an anisotropic one needs gp_GTrsf, and OCCT then re-approximates every
  // non-planar surface as a B-spline.
  py::dict scale(double sx, double sy, double sz, double cx, double cy, double cz,
                 const std::optional<std::vector<EntityId>>& entity_ids);

  // Duplicate the bodies owning the named entities. Deliberately committed with no history:
  // BRepBuilderAPI_Copy reports the duplicate as "modified from" the original, which would
  // move the original's id onto the copy. The originals keep their ids because they are
  // still in the model; every entity of every copy is a new identity.
  py::dict copy(const std::vector<EntityId>& entity_ids);

  // ---- healing, defeaturing, imprinting, removal -------------------------------------- //
  //
  // These are the operations whose input is allowed to be broken, and that shapes their
  // contract: the healing three report the BRepCheck_Analyzer verdict on their result
  // instead of refusing to commit it, because a shape that is less invalid than before is
  // progress the caller must be allowed to keep. See Validation.
  //
  // All of them are scoped. ShapeFix_Shape and friends work on whatever shape they are
  // given, so restricting a heal to chosen bodies costs nothing and buys the property a
  // global healing pass cannot offer: every entity outside the scope is left byte-identical,
  // not merely unchanged-looking.

  py::dict heal(const std::optional<std::vector<EntityId>>& entity_ids, double precision,
                double min_tolerance, double max_tolerance, const py::object& progress,
                const py::object& cancel);

  // Sew the named bodies into shells, optionally closing a watertight shell into a solid.
  // This is the "faces to solid" path, and the repair for a model whose faces coincide at
  // their boundaries without sharing edges.
  py::dict sew(const std::vector<EntityId>& entity_ids, double tolerance, bool make_solid,
               bool non_manifold, const py::object& progress, const py::object& cancel);

  // Drop internal wires (holes) smaller than min_area, and optionally the faces they leave
  // behind. This is small-feature defeaturing on the face level, where `defeature` is the
  // solid-level operation.
  py::dict remove_internal_wires(const std::optional<std::vector<EntityId>>& entity_ids,
                                 double min_area, bool remove_faces);

  // Remove the features the named faces belong to, closing the gaps with the surrounding
  // geometry. OCCT removes a *complete* feature: naming part of one leaves the shape
  // untouched while still reporting success, so this verifies every named face actually went
  // away and fails loud naming the ones that did not.
  py::dict defeature(const std::vector<EntityId>& face_ids, bool parallel,
                     const py::object& progress, const py::object& cancel);

  // Imprint the tools onto the targets: the targets are split where the tools meet them, so
  // the interface exists as real topology on both sides. Unlike `split` the tools may be of
  // any dimension, and like `split` they are not consumed.
  py::dict imprint(const std::vector<EntityId>& targets, const std::vector<EntityId>& tools,
                   double fuzzy, bool parallel, int glue, const py::object& progress,
                   const py::object& cancel);

  // Drop the bodies owning the named entities from the model. Every id inside them dies and
  // is never reused; an id on a sub-shape a surviving body also owns stays alive, because
  // that shape is still in the model.
  py::dict remove(const std::vector<EntityId>& entity_ids);

  // Merge faces and edges that lie on one underlying surface or curve. This is the
  // duplicate-removal pass after a boolean, and the stateless API's unify_same_domain with
  // the session's identity carried across it.
  py::dict unify_same_domain(const std::optional<std::vector<EntityId>>& entity_ids,
                             bool unify_faces, bool unify_edges, bool concat_bsplines,
                             double linear_tol, double angular_tol_rad);

  // ---- snapshot / restore ----------------------------------------------------------- //

  // Guarded like an operation, not like a query. These mutate the session, and an operation
  // that has released the GIL is still in flight: a restore landing in that window would
  // swap the state out from under it. A read-only query needs no guard, because the state a
  // running operation publishes is only ever written while the GIL is held.
  std::int64_t snapshot();

  void restore(std::int64_t mark);

  void discard_snapshot(std::int64_t mark);

  std::int64_t snapshot_count() const;

  // ---- queries ---------------------------------------------------------------------- //

  py::array_t<std::int64_t> entities(const std::string& kind) const;

  std::string entity_kind(EntityId id) const;

  // "alive" or "dead". An id that was never issued is a caller error, not a state: it
  // raises, which is how a positional ordinal from the stateless API fails loudly here
  // instead of silently denoting somebody else's entity.
  std::string entity_state(EntityId id) const;

  std::int64_t shape_count(EntityId id) const;

  // Bulk geometry of every alive entity of one kind, as parallel arrays. Bulk rather than
  // per-entity objects because the calling convention has to stay vectorised for a model
  // with tens of thousands of faces.
  py::dict entity_table(const std::string& kind) const;

  py::bytes brep() const;

  // ---- geometric queries -------------------------------------------------------------- //
  //
  // Read-only, so unlike an operation they take no OpGuard: the invariant that state_ is
  // only ever written while the GIL is held makes a query safe without one. A query that
  // releases the GIL copies the shapes it needs first, so it reads handles it owns rather
  // than the session's live state.

  // The underlying geometry type of every live entity of one kind: a surface type for faces,
  // a curve type for edges. Bulk, because a caller classifying a model wants every answer.
  py::dict entity_types(const std::string& kind) const;

  // Bounding boxes alone, for every live entity of one kind. Deliberately separate from
  // entity_table: BRepBndLib is orders of magnitude cheaper than BRepGProp's mass
  // properties, and a caller culling or spatially indexing a model needs only the box.
  py::dict bounding_boxes(const std::string& kind) const;

  // Measure and centre of mass of the named entities. Each is computed by its own kind —
  // volume for a solid, area for a face, length for an edge — never by walking a parent:
  // BRepGProp::LinearProperties on a SOLID visits every edge once per owning face and
  // silently doubles the answer.
  py::dict mass_properties(const std::vector<EntityId>& entity_ids) const;

  // (N, 4) umin, umax, vmin, vmax for the named faces.
  py::array_t<double> face_parameter_bounds(const std::vector<EntityId>& face_ids) const;

  // (N, 2) first, last parameter for the named edges.
  py::array_t<double> edge_parameter_bounds(const std::vector<EntityId>& edge_ids) const;

  // Pairs relating every live entity of one kind to the entities of another kind it touches.
  // Which direction the relation runs is decided by the two kinds: towards a lower dimension
  // it is the boundary (a face's edges), towards a higher one it is the ancestors (an edge's
  // faces). One method rather than two, because they are one relation read from either end.
  py::dict adjacency(const std::string& kind, const std::string& other_kind) const;

  // Positions and outward normals of a face at the given parameters. The normal is flipped
  // for a REVERSED face, so it points out of the body rather than along the surface's own
  // parametrisation — which is the direction every consumer of a normal actually means.
  py::dict surface_at(EntityId face_id, const PointArray& uv) const;

  // Peak absolute curvature of each named face, over an n x n grid of its parameter domain.
  //
  // The grid is the point of the operation. Sampling one point at a face's parametric centre
  // — which is what the API this replaces does — is exact only for a face of constant
  // curvature and arbitrarily wrong otherwise: on a cone tapering 4 to 1 the centre sample
  // reads 0.358 against a true peak of 0.894. Samples land at cell centres, so no sample
  // sits on a seam or a pole, and a sample outside the face's own trimming is discarded
  // rather than reporting a curvature the face does not have.
  py::dict curvature(const std::vector<EntityId>& face_ids, int samples) const;

  // Closest point on a face's underlying surface to each of the given points.
  py::dict project_on_face(EntityId face_id, const PointArray& points) const;

  // Live entities of one kind whose bounding box meets the given box. `strict` asks for
  // containment (the entity's box inside the query box) rather than overlap.
  py::array_t<std::int64_t> entities_in_box(const std::string& kind, double xmin, double ymin,
                                            double zmin, double xmax, double ymax,
                                            double zmax, bool strict) const;

  // (S, P) mask: whether each point is strictly inside each named solid. Strictly inside
  // only — a point within tol of the boundary is ON, and reads False, which is the right
  // contract for seeding a volume.
  py::array_t<bool> contains(const std::vector<EntityId>& solid_ids, const PointArray& points,
                             double tol) const;

  // ---- the render mesh ---------------------------------------------------------------- //

  // Triangles, edge polylines and vertex points of the live shape, all indexed into one node
  // array and all labelled with session entity ids.
  //
  // Three things make this different from the stateless tessellator, and each is the reason
  // the session exists:
  //
  //   * BRepMesh caches its triangulation on the TopoDS_Face. A session keeps its faces
  //     alive across operations, so a face no operation touched is not re-triangulated —
  //     the work is proportional to what changed, not to the model.
  //   * The output says *what* changed, which is the half a consumer cannot recover for
  //     itself: diffing the arrays costs more than the tessellation saved.
  //   * Edges and vertices come from the same call and index into the same nodes. Above the
  //     size at which triangles stop being shippable, the polylines and the points are the
  //     whole picking and wireframe substrate, and a face-only tessellation cannot serve it.
  //
  // Not an operation: it issues no ids, advances no counter and changes no topology. It is
  // still guarded and still non-const, because it writes the triangulation onto shapes that
  // retained states share and it does that with the GIL released.
  py::dict tessellate(double deflection, double angle_rad, bool relative, bool parallel,
                      bool incremental, const py::object& progress,
                      const py::object& cancel);

  // ---- the meshing handoff ------------------------------------------------------------ //

  // The live shape as BREP, plus the entity id at every ordinal of the traversal a reader of
  // those bytes reproduces — the map a mesher's own tags have to be paired against.
  //
  // The pairing is positional, never geometric. Matching by centroid is the obvious shortcut
  // and it is wrong by construction: a pipe's inner and outer walls have the same centroid
  // to within 6e-17, so any centroid-keyed map collides on the most ordinary CAD feature
  // there is. Ordinals cannot collide, and the order is reproducible — measured, on a
  // 28 255-entity assembly written and read back with every ordinal preserved.
  //
  // The map must be a bijection, and this verifies it rather than assuming it. Two session
  // states break it, both reachable and both legitimate: a merge leaves several live ids on
  // one shape, and a split leaves one live id on several. Either makes "this id is that tag"
  // ambiguous, so the export fails loud naming the ids rather than handing over a map that
  // silently loses some of them.
  py::dict export_handoff() const;

  // ---- names ------------------------------------------------------------------------ //

  // The persistent name of an entity: the operation that issued its id, how that operation
  // produced it, and its rank among that operation's issued entities of that role. Purely
  // provenance — no geometric fingerprint is involved at any point, because fingerprint
  // matching mis-identifies entities under exactly the edits that make naming necessary.
  py::dict name_of(EntityId id) const;

  py::dict origin(EntityId id) const;

  // Resolve a name against the CURRENT state. "lost" is a legitimate, loud answer and the
  // caller must handle it; guessing a replacement is never done.
  py::dict resolve(std::int64_t op_index, int role, int ordinal) const;

  // ---- introspection ---------------------------------------------------------------- //

  std::int64_t op_count() const { return next_op_ - 1; }
  std::int64_t state_op_index() const { return state_.op_index; }
  std::int64_t issued_id_count() const { return next_id_ - 1; }
  std::int64_t entity_count() const;

  // Test hook. Falsification of the naming suite is mandatory: a history test that has never
  // been shown to fail is a claim, not a check. This drops the next operation's history so
  // the ids in the consumed bodies die instead of being carried forward, which is precisely
  // the "one delta dropped" tear the identity gate must detect.
  void debug_tear_next_history() { tear_next_history_ = true; }

 private:
  // ---- validation helpers ----------------------------------------------------------- //

  static void require_positive(const char* name, double v);

  static void require_non_negative(const char* name, double v);

  // A partial primitive sweeps through angle_rad about its axis; the full solid is 2*pi.
  // OCCT clamps silently outside that band, which would hand back a shape the caller did
  // not ask for, so it is refused here instead.
  static void require_sweep_angle(const char* op, double angle_rad);

  // Run one OCCT construction with the GIL released, and convert whatever OCCT throws into
  // a typed PysmeshError naming the operation. OCCT signals an invalid parameter
  // combination with Standard_ConstructionError, which derives from std::exception and
  // would otherwise reach the caller as a bare RuntimeError carrying OCCT's wording and no
  // indication of which call produced it.
  template <typename Fn>
  static TopoDS_Shape build_shape(const char* op, Fn&& fn) {
    TopoDS_Shape out;
    {
      py::gil_scoped_release release;
      try {
        out = fn();
      } catch (const std::exception& e) {
        py::gil_scoped_acquire acquire;
        throw PysmeshError(std::string("Session.") + op + ": OCCT rejected the request: " +
                           e.what());
      }
    }
    if (out.IsNull()) {
      throw PysmeshError(std::string("Session.") + op +
                         ": OCCT produced a null shape from these parameters.");
    }
    return out;
  }

  // Same contract as build_shape, but for a construction that reports its own failure
  // through IsDone: a null result means "OCCT declined", and the caller supplies the
  // wording that explains what the caller did wrong.
  template <typename Fn>
  static TopoDS_Shape try_build(const char* op, Fn&& fn) {
    TopoDS_Shape out;
    {
      py::gil_scoped_release release;
      try {
        out = fn();
      } catch (const std::exception& e) {
        py::gil_scoped_acquire acquire;
        throw PysmeshError(std::string("Session.") + op + ": OCCT rejected the request: " +
                           e.what());
      }
    }
    return out;
  }

  // Entity ids for PysmeshError's offending-id channel. That field is int-typed by v1's
  // contract, which is wide enough for every id a session can realistically issue.
  static std::vector<int> ids_as_int(const std::vector<EntityId>& ids);

  // Validate the two caller-supplied hooks and pair them with the poll interval.
  //
  // The interval is fixed rather than exposed. 25 ms is short enough that a cancel is seen
  // far inside the half-second the contract allows and that a progress bar updates at 40 Hz,
  // and long enough that the GIL round trip is nothing beside the operation — a boolean that
  // advances its position 291 303 times is polled a few hundred times instead. A knob here
  // would be a number every caller leaves alone.
  static ProgressHooks hooks_of(const char* op, const py::object& progress,
                                const py::object& cancel);

  // Normalise any BRepBuilderAPI_MakeShape-derived algorithm's history into the same
  // Handle(BRepTools_History) the booleans produce, so one carry routine serves them all.
  template <typename Algo>
  static Handle(BRepTools_History) history_of(const TopoDS_Shape& argument, Algo& algo) {
    NCollection_List<TopoDS_Shape> args;
    args.Append(argument);
    return new BRepTools_History(args, algo);
  }

  // A shape OCCT's sweeps raise the dimension of. A solid has no higher dimension to sweep
  // into, and a compound sweeps element-wise, which would silently produce a shape the
  // caller did not ask for.
  static void require_sweepable(const char* op, const TopoDS_Shape& body);

  void require_issued(const char* op, EntityId id) const;

  const EntityRecord& require_alive(const char* op, EntityId id) const;

  static void require_fuzzy(const char* op, double fuzzy);

  static void require_operands(const char* op, const std::vector<EntityId>& targets,
                               const std::vector<EntityId>& tools, double fuzzy);

  // The root bodies owning the solids two operand lists name. Resolved through the
  // sub-shape -> body map rather than by comparing the solids themselves, so a solid nested
  // inside a compound body still removes the right body from the model.
  std::vector<TopoDS_Shape> bodies_of(const std::vector<EntityId>& a,
                                      const std::vector<EntityId>& b) const;

  // One driver for the whole BOP family. Every BRepAlgoAPI_BuilderAlgo descendant shares
  // the arguments, options, history and error channels, so only the operand assignment (on
  // the concrete type, because SetTools is not virtual) and whether the operation replaces
  // or extends the model differ. `op` must already carry its arguments and tools.
  py::dict run_bop(const char* op_name, BRepAlgoAPI_BuilderAlgo& op,
                   const std::vector<TopoDS_Shape>& consumed, bool additive, double fuzzy,
                   bool parallel, const ProgressHooks& hooks);

  // The edges of every contour OCCT could not build a fillet on. This is the diagnostic
  // that turns "the fillet failed" into "the fillet failed on these edges"; when the
  // builder cannot report it, the caller is blamed for every edge it named instead.
  static std::vector<TopoDS_Shape> faulty_edges(BRepFilletAPI_MakeFillet& mk);

  // Entity ids for the named edges that OCCT blamed, or all of them when it blamed none.
  static std::vector<int> blamed_ids(const std::vector<EntityId>& edge_ids,
                                     const std::vector<TopoDS_Shape>& edges,
                                     const std::vector<TopoDS_Shape>& faulty);

  NCollection_List<TopoDS_Shape> solids_of(const char* op, const char* argname,
                                           const std::vector<EntityId>& ids) const;

  // ---- healing helpers -------------------------------------------------------------- //

  // The rework counterpart of rebuild_moved: hand the scoped bodies to a repair algorithm as
  // one compound, take back a shape and a history, and commit the result reporting its
  // validity rather than refusing it. The bodies outside the scope are never passed to the
  // algorithm at all, which is what makes "everything out of scope stays byte-identical" a
  // property of the construction rather than a hope about the algorithm.
  // The `run` callback takes the progress range so that a repair which accepts one can pass
  // it on; the two that cannot ignore the argument, which is honest — OCCT gives them no
  // range to take.
  py::dict rework(const std::optional<std::vector<EntityId>>& entity_ids, const char* op_name,
                  const ProgressHooks& hooks,
                  const std::function<void(const TopoDS_Shape&, const Message_ProgressRange&,
                                           TopoDS_Shape&, Handle(BRepTools_History)&)>& run);

  // The faces the named entities denote, for the operations that rework a body around them.
  // A split entity is rejected rather than silently contributing several faces.
  std::vector<TopoDS_Shape> faces_of(const char* op, const std::vector<EntityId>& ids) const;

  // The single face a query names.
  TopoDS_Face sole_face(const char* op, EntityId id) const;

  // Bodies for a boolean-family operand list, whatever dimension they are. The boolean
  // family proper takes SOLID ids (solids_of); imprinting deliberately does not, because a
  // face or a wire is a legitimate imprinting tool.
  std::vector<TopoDS_Shape> operand_bodies(const char* op, const char* argname,
                                           const std::vector<EntityId>& ids) const;

  // ---- query helpers ---------------------------------------------------------------- //

  // Live entity ids of one kind, ascending. The order every bulk query's rows are in.
  std::vector<EntityId> ids_of_kind(TopAbs_ShapeEnum kind) const;

  // The id that labels a sub-shape of the current root. A merge leaves several ids on one
  // shape; the lowest is the label, and the others stay alive and still resolve — they just
  // do not appear in a per-shape array, which has one row to give. A shape in the root with
  // no id at all is a torn registry, so it raises rather than being skipped.
  EntityId label_of(const char* op, const TopoDS_Shape& s) const;

  // ---- root bookkeeping ------------------------------------------------------------- //

  std::vector<TopoDS_Shape> bodies_excluding(const std::vector<TopoDS_Shape>& drop) const;

  static std::vector<TopoDS_Shape> concat(const std::vector<TopoDS_Shape>& keep,
                                          const TopoDS_Shape& added);

  // sub-shape -> owning body, for every body of the current root. Built once per call so
  // that resolving many entities to their bodies stays linear rather than quadratic.
  ShapeKeyed<TopoDS_Shape> body_of_subshape() const;

  // The single body that owns every one of the given sub-shapes. Fails loud when the
  // selection straddles two bodies, because one operation builds one shape and the caller's
  // intent is genuinely ambiguous in that case.
  static TopoDS_Shape sole_owner_body(const ShapeKeyed<TopoDS_Shape>& owners,
                                      const std::vector<TopoDS_Shape>& subs);

  // Every distinct body of the current root that owns at least one shape of the named
  // entities, in first-appearance order so the outcome is reproducible.
  std::vector<TopoDS_Shape> owner_bodies(const char* op,
                                         const std::vector<EntityId>& ids) const;

  // The single body that owns every named entity. One operation builds one shape, so a
  // selection straddling two bodies is a genuinely ambiguous request and is refused.
  TopoDS_Shape sole_body(const char* op, const std::vector<EntityId>& ids) const;

  // The edges the named entities denote. A split entity is rejected rather than silently
  // contributing several edges, because the caller's intent is then ambiguous. `kept`, when
  // given, receives the id of each returned edge, so a per-edge diagnostic can name it.
  std::vector<TopoDS_Shape> edges_of(const char* op, const std::vector<EntityId>& ids,
                                     std::vector<EntityId>* kept = nullptr) const;

  // Construction geometry: a body that is a loose edge or a wire. Building a wire or a face
  // out of an edge that belongs to a solid would consume the solid, so that is refused.
  static void require_curve_body(const char* op, const TopoDS_Shape& body);

  // A wire over the named edges. When they are exactly one existing wire body that wire is
  // used as-is, which keeps every edge id: BRepBuilderAPI_MakeWire rebuilds any edge whose
  // end vertex is merely coincident with the wire's rather than shared, and a rebuilt edge
  // is a new entity.
  static TopoDS_Wire wire_over(const char* op, const std::vector<TopoDS_Shape>& edges,
                               const std::vector<TopoDS_Shape>& owners);

  // A wire over a whole body, for the operations that sweep along or across one.
  static TopoDS_Wire wire_of_body(const char* op, const char* argname,
                                  const TopoDS_Shape& body);

  py::dict add_bodies(const TopoDS_Shape& added, const char* op_name);

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
                  const TopoDS_Shape& built, Validation mode = Validation::Strict);

  Delta carry_registry(const TopoDS_Shape& new_root, const Handle(BRepTools_History) & hist,
                       std::int64_t op_index);

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
  // entities. Shared by the transforms and by the healing family, because "move these" and
  // "repair these" select their subject the same way — and carry the same hazard, that a
  // body sharing sub-shapes with one left alone cannot be reworked on its own without
  // tearing the model.
  std::vector<TopoDS_Shape> scoped_bodies(
      const std::optional<std::vector<EntityId>>& entity_ids, const char* op_name) const;

  // A transform OCCT can express as a change of Location alone takes the relocation path,
  // where identity is structural; anything else — a plane mirror, a scaling — is a genuine
  // rebuild and takes the history path. The two are the same test: BRepBuilderAPI_Transform
  // sets a new Location only for a direct isometry, and TopLoc_Datum3D refuses to be built
  // from anything else at all.
  py::dict apply_trsf(const gp_Trsf& trsf,
                      const std::optional<std::vector<EntityId>>& entity_ids,
                      const char* op_name);

  // The rebuild half of the transform pair. The moving bodies are handed to OCCT as one
  // compound so a single history covers them all; every sub-shape maps one-to-one through
  // Modified, so every entity id survives — it just survives by history rather than by
  // TShape identity.
  py::dict rebuild_moved(const std::optional<std::vector<EntityId>>& entity_ids,
                         const char* op_name,
                         const std::function<void(const TopoDS_Shape&, TopoDS_Shape&,
                                                  Handle(BRepTools_History)&)>& run);

  py::dict relocate(const gp_Trsf& trsf,
                    const std::optional<std::vector<EntityId>>& entity_ids,
                    const char* op_name);

  void guard_shared_subshapes(const std::vector<TopoDS_Shape>& all,
                              const std::vector<TopoDS_Shape>& moving,
                              const char* op_name) const;

  static void finalise(Delta& d);

  static py::dict delta_dict(const Delta& d, std::int64_t op_index, const char* op_name);

  // A (N, 2) parameter-pair argument, forcecast so a list of tuples is accepted.
  static std::vector<std::pair<double, double>> pairs_of(const char* op, const char* argname,
                                                         const PointArray& a);

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

  // What the previous tessellate() emitted, keyed by TShape address. Not part of
  // SessionState: it describes the last render mesh handed out, which a restore does not
  // undo. Restoring an earlier root simply means the faces of that root are compared against
  // whatever they last emitted, which is the right answer — their triangulations are still
  // whatever they were.
  std::unordered_map<const void*, EmittedFace> emitted_;
};

}  // namespace session
}  // namespace pysmesh
