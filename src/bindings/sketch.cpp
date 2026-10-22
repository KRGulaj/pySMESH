// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-22

// pySMESH binding — sketch_solve: the 2-D sketch constraint solver, FreeCAD's PlaneGCS over
// flat arrays.
//
// A sketch arrives as arrays; docs/documentation/concepts/sketch.md is the contract:
//   * params (N,) float64: every scalar of the sketch, and the initial guess;
//   * entities (E, 8) int64: one row per entity, column 0 its kind, the other columns
//     parameter indices or entity rows, unused columns -1;
//   * constraints (C, 5) int64: one row per constraint, column 0 its kind, the other columns
//     entity rows (and, for an internal coincidence, a role and an index), unused -1;
//   * values (C,) float64: each dimensional constraint's dimension, NaN for the others;
//   * fixed (N,) bool: parameters the solver must not move;
//   * spline_offsets (S+1,) and spline_indices: the B-spline slots, CSR style.
//
// Each constraint row is added as SALOME Shaper's PlaneGCSSolver adapter maps it (Shaper
// 59c5672c2b67ca007cadb522e198a9cf38434886, V9_15_0-35-g59c5672c2, src/SketchSolver, read
// for its behaviour; no Shaper code is copied), with the
// GCS::System calls listed in the contract page. The system is diagnosed and solved as
// Shaper's PlaneGCSSolver_Solver::solve does: DogLeg, then Levenberg-Marquardt and BFGS
// when the previous one fails, each after its own diagnosis.
//
// Determinism. PlaneGCS sorts and keys its parameters by address (the parameter sets of
// SubSystem::initialize, the reduction maps of System::initSolution). Every scalar a
// constraint points at lives in one std::vector<double> that is sized once: the params
// first, then the values, then the auxiliary scalars. Address order is then index order, the
// same in every run. Containers keyed by constraint address are read only where the order
// does not change a value (the redundant set) or where PlaneGCS breaks a tie by tag (the
// conflict map), and every constraint row has its own tag: row + 1. The rules that tie an
// arc's points to its angles have tag 0, so PlaneGCS never names them as conflicting. No
// thread runs: patches/planegcs/GCS_diagnose_serial.patch keeps diagnose's two QR
// decompositions on the calling thread, Eigen runs on one thread, and PlaneGCS has no other
// state that outlives a System than its logging, which is off.

#include "common.hpp"

#include <GCS.h>

#include <Geom2dAPI_ProjectPointOnCurve.hxx>
#include <Geom2d_BSplineCurve.hxx>
#include <NCollection_Array1.hxx>
#include <Standard_Failure.hxx>
#include <gp_Pnt2d.hxx>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace pysmesh {
namespace {

constexpr std::size_t kEntityColumns = 8;
constexpr std::size_t kConstraintColumns = 5;
constexpr double kPi = 3.141592653589793238462643383279502884;

// The public kind codes (pysmesh.sketch.SketchEntityKind, SketchConstraintKind and
// SketchInternalRole). A value never changes once released.
enum EntityKind : std::int64_t {
  kPoint = 1,
  kLine = 2,
  kCircle = 3,
  kArc = 4,
  kEllipse = 5,
  kEllipticArc = 6,
  kBSpline = 7,
};

enum ConstraintKind : std::int64_t {
  kCoincident = 1,
  kTangent = 2,
  kParallel = 3,
  kPerpendicular = 4,
  kHorizontal = 5,
  kVertical = 6,
  kEqual = 7,
  kCollinear = 8,
  kMidpoint = 9,
  kRigid = 10,
  kLength = 11,
  kAngle = 12,
  kRadius = 13,
  kInternalCoincident = 14,
  kDistance = 15,
  kSymmetric = 16,
};

enum InternalRole : std::int64_t {
  kMajorPositive = 1,
  kMajorNegative = 2,
  kMinorPositive = 3,
  kMinorNegative = 4,
  kFocus2 = 5,
  kPole = 6,
  kKnot = 7,
};

const char* entity_name(std::int64_t kind) {
  switch (kind) {
    case kPoint:
      return "POINT";
    case kLine:
      return "LINE";
    case kCircle:
      return "CIRCLE";
    case kArc:
      return "ARC";
    case kEllipse:
      return "ELLIPSE";
    case kEllipticArc:
      return "ELLIPTIC_ARC";
    case kBSpline:
      return "BSPLINE";
    default:
      return "an unknown kind";
  }
}

const char* constraint_name(std::int64_t kind) {
  switch (kind) {
    case kCoincident:
      return "COINCIDENT";
    case kTangent:
      return "TANGENT";
    case kParallel:
      return "PARALLEL";
    case kPerpendicular:
      return "PERPENDICULAR";
    case kHorizontal:
      return "HORIZONTAL";
    case kVertical:
      return "VERTICAL";
    case kEqual:
      return "EQUAL";
    case kCollinear:
      return "COLLINEAR";
    case kMidpoint:
      return "MIDPOINT";
    case kRigid:
      return "RIGID";
    case kLength:
      return "LENGTH";
    case kAngle:
      return "ANGLE";
    case kRadius:
      return "RADIUS";
    case kInternalCoincident:
      return "INTERNAL_COINCIDENT";
    case kDistance:
      return "DISTANCE";
    case kSymmetric:
      return "SYMMETRIC";
    default:
      return "an unknown kind";
  }
}

bool is_dimensional(std::int64_t kind) {
  return kind == kLength || kind == kAngle || kind == kRadius || kind == kDistance;
}

[[noreturn]] void refuse(const std::string& text) {
  throw PysmeshError("sketch.solve: " + text);
}

// ---- The input arrays ------------------------------------------------------------------ //

// The caller's arrays, copied while the GIL is held.
struct Tables {
  std::vector<double> params;
  std::vector<std::int64_t> entities;
  std::vector<std::int64_t> constraints;
  std::vector<double> values;
  std::vector<std::uint8_t> fixed;
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> indices;

  std::size_t n() const { return params.size(); }
  std::size_t e() const { return entities.size() / kEntityColumns; }
  std::size_t c() const { return constraints.size() / kConstraintColumns; }
  std::int64_t ent(std::size_t row, std::size_t col) const {
    return entities[row * kEntityColumns + col];
  }
  std::int64_t con(std::size_t row, std::size_t col) const {
    return constraints[row * kConstraintColumns + col];
  }
};

std::string dtype_text(const py::array& a) {
  return py::str(static_cast<py::object>(a.dtype())).cast<std::string>();
}

std::string shape_text(const py::array& a) {
  std::string out = "(";
  for (py::ssize_t i = 0; i < a.ndim(); ++i) {
    out += (i == 0 ? "" : ", ") + std::to_string(a.shape(i));
  }
  return out + (a.ndim() == 1 ? ",)" : ")");
}

// `obj` as an array of the dtype kind `kind` and item size `itemsize`, named `name`.
py::array typed_array(const py::object& obj, const char* name, char kind, py::ssize_t itemsize,
                      const char* dtype) {
  if (!py::isinstance<py::array>(obj)) {
    refuse(std::string(name) + " must be a NumPy array of " + dtype + " (got " +
           py::str(static_cast<py::object>(py::type::of(obj))).cast<std::string>() + ").");
  }
  py::array a = obj.cast<py::array>();
  if (a.dtype().kind() != kind || a.itemsize() != itemsize) {
    refuse(std::string(name) + " must be a NumPy array of " + dtype + " (got dtype " +
           dtype_text(a) + ").");
  }
  return a;
}

template <typename T>
std::vector<T> copy_of(const py::array& a) {
  const auto c = a.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
  return std::vector<T>(c.data(), c.data() + c.size());
}

// The 1-D float64 array `name`, of `length` values unless `length` is -1.
std::vector<double> read_f64(const py::object& obj, const char* name, py::ssize_t length) {
  const py::array a = typed_array(obj, name, 'f', 8, "float64");
  if (a.ndim() != 1 || (length >= 0 && a.shape(0) != length)) {
    refuse(std::string(name) + " must have shape (" +
           (length >= 0 ? std::to_string(length) : std::string("N")) + ",) (got " +
           shape_text(a) + ").");
  }
  return copy_of<double>(a);
}

// The int64 array `name`: 1-D when `columns` is 0, else (rows, columns).
std::vector<std::int64_t> read_i64(const py::object& obj, const char* name,
                                   std::size_t columns) {
  const py::array a = typed_array(obj, name, 'i', 8, "int64");
  const bool ok = columns == 0 ? a.ndim() == 1
                               : a.ndim() == 2 &&
                                     a.shape(1) == static_cast<py::ssize_t>(columns);
  if (!ok) {
    refuse(std::string(name) + " must have shape " +
           (columns == 0 ? std::string("(K,)") : "(rows, " + std::to_string(columns) + ")") +
           " (got " + shape_text(a) + ").");
  }
  return copy_of<std::int64_t>(a);
}

// ---- Geometry -------------------------------------------------------------------------- //

// One entity row's PlaneGCS geometry; only the member of the row's kind is set.
struct Geometry {
  std::int64_t kind = 0;
  GCS::Point point;
  GCS::Line line;
  GCS::Circle circle;
  GCS::Arc arc;
  GCS::Ellipse ellipse;
  GCS::ArcOfEllipse elliptic_arc;
  GCS::BSpline bspline;

  GCS::Curve* curve() {
    switch (kind) {
      case kLine:
        return &line;
      case kCircle:
        return &circle;
      case kArc:
        return &arc;
      case kEllipse:
        return &ellipse;
      case kEllipticArc:
        return &elliptic_arc;
      case kBSpline:
        return &bspline;
      default:
        return nullptr;
    }
  }
};

// One B-spline row's slot, resolved.
struct Spline {
  std::vector<std::int64_t> poles;    // point rows
  std::vector<std::int64_t> weights;  // parameter indices
  std::vector<std::int64_t> knots;    // parameter indices
  std::vector<int> mults;
  int degree = 0;
  bool periodic = false;
  Handle(Geom2d_BSplineCurve) occt;  // the same curve, for the projection of a point
};

// How a constraint row becomes PlaneGCS constraints; see Sketch::add_row.
enum class Map {
  kPointPoint,
  kPointLine,
  kPointCircle,
  kPointArc,
  kPointEllipse,
  kPointBSpline,
  kLineCircle,
  kCircleCircle,
  kLineEllipse,
  kAngleViaPoint,
  kLineLine,
  kCenterOnLine,
  kDirect,
  kEqualLength,
  kEqualRadius,
  kEqualRadii,
  kCollinear,
  kMidpointLine,
  kMidpointArc,
  kRigid,
  kInternalEllipse,
  kInternalPole,
  kInternalKnot,
  kPointPointDistance,
  kPointLineDistance,
};

// A constraint row, checked: its operands in a fixed order, its mapping and its auxiliary
// scalars.
struct Plan {
  std::int64_t kind = 0;
  Map map = Map::kDirect;
  std::int64_t a = -1;
  std::int64_t b = -1;
  std::int64_t p = -1;
  int end_a = -1;  // a B-spline operand replaced by its end segment: 0 first, 1 last
  int end_b = -1;
  std::int64_t role = -1;
  std::int64_t index = -1;
  bool on_first = false;  // COLLINEAR: line b's points that go on line a
  bool on_second = false;
  std::vector<std::size_t> rigid;  // RIGID: the scalars it holds
  std::size_t aux = 0;             // the first auxiliary scalar
  std::size_t aux_count = 0;
};

// What sketch_solve gives back.
struct Outcome {
  std::vector<double> params;
  std::string status;
  int dof = 0;
  std::vector<std::int64_t> conflicting;
  std::vector<std::int64_t> redundant;
  std::vector<std::int64_t> partially_redundant;
};

class Sketch {
 public:
  Sketch(const Tables& t, int max_iterations, double tolerance)
      : t_(t), max_iterations_(max_iterations), tolerance_(tolerance) {}

  Outcome solve();

 private:
  // checks
  std::int64_t param_at(std::size_t row, std::size_t col, const char* what);
  std::int64_t entity_at(std::size_t row, std::size_t col, const char* what,
                         std::int64_t want) const;
  void unused_entity_columns(std::size_t row, std::size_t first) const;
  void check_entities();
  void check_splines();
  void check_constraints();
  Plan plan_row(std::size_t row) const;
  void unused_constraint_columns(std::size_t row, std::size_t first) const;
  std::int64_t operand(std::size_t row, std::size_t col, const char* what) const;
  std::int64_t optional_point(std::size_t row, std::size_t col) const;
  bool same_point(std::int64_t a, std::int64_t b) const;
  int spline_end(std::int64_t spline, std::int64_t point) const;
  std::string row_text(std::size_t row) const;
  std::string entity_text(std::int64_t row) const;
  void plan_curves(Plan& plan, std::size_t row, bool tangent) const;
  void plan_rigid(Plan& plan, std::size_t row) const;
  double value(std::size_t row) const { return t_.values[row]; }

  // build and solve
  void build_geometry();
  void add_entity_rules();
  void add_row(std::size_t row, Plan& plan);
  GCS::Curve& operand_curve(std::int64_t row, int end);
  double nearest_parameter(std::int64_t spline_row, std::int64_t point_row) const;
  double* s(std::size_t i) { return &store_[i]; }
  double* s(std::int64_t i) { return &store_[static_cast<std::size_t>(i)]; }
  GCS::Point& pt(std::int64_t row) { return geo_[static_cast<std::size_t>(row)].point; }
  Outcome unsolved_outcome(const std::string& status, int dof) const;
  std::vector<std::int64_t> rows_of(const GCS::VEC_I& tags) const;

  const Tables& t_;
  const int max_iterations_;
  const double tolerance_;

  std::vector<std::uint8_t> role_;  // per parameter: 0 unused, 1 moves, 2 knot
  std::vector<Spline> splines_;     // per entity row (only B-spline rows are set)
  std::vector<Plan> plans_;         // per constraint row
  std::vector<std::size_t> aux_unknowns_;
  std::vector<double> store_;
  std::vector<Geometry> geo_;
  std::deque<GCS::Line> segments_;  // B-spline end segments, as Shaper substitutes them
  GCS::System sys_;
  std::size_t added_ = 0;  // PlaneGCS constraints added
};

std::string Sketch::entity_text(std::int64_t row) const {
  const std::int64_t kind = t_.ent(static_cast<std::size_t>(row), 0);
  const bool known = kind >= kPoint && kind <= kBSpline;
  return "entity row " + std::to_string(row) + " (" +
         (known ? std::string(entity_name(kind)) : "kind code " + std::to_string(kind)) + ")";
}

std::string Sketch::row_text(std::size_t row) const {
  return "constraint row " + std::to_string(row) + " (" + constraint_name(t_.con(row, 0)) +
         ")";
}

// The parameter index in column `col` of entity row `row`, in range, marked as one that
// moves. The knots are marked last (check_splines), so a knot that shares a parameter with
// one of these is found there.
std::int64_t Sketch::param_at(std::size_t row, std::size_t col, const char* what) {
  const std::int64_t i = t_.ent(row, col);
  if (i < 0 || static_cast<std::size_t>(i) >= t_.n()) {
    refuse(entity_text(static_cast<std::int64_t>(row)) + ": column " + std::to_string(col) +
           " (" + what + ") holds parameter index " + std::to_string(i) +
           ", outside params of length " + std::to_string(t_.n()) + ".");
  }
  role_[static_cast<std::size_t>(i)] = 1;
  return i;
}

// The entity row in column `col` of entity row `row`, in range and of kind `want`.
std::int64_t Sketch::entity_at(std::size_t row, std::size_t col, const char* what,
                               std::int64_t want) const {
  const std::int64_t r = t_.ent(row, col);
  if (r < 0 || static_cast<std::size_t>(r) >= t_.e()) {
    refuse(entity_text(static_cast<std::int64_t>(row)) + ": column " + std::to_string(col) +
           " (" + what + ") holds entity row " + std::to_string(r) + ", outside the " +
           std::to_string(t_.e()) + " entity rows.");
  }
  if (t_.ent(static_cast<std::size_t>(r), 0) != want) {
    refuse(entity_text(static_cast<std::int64_t>(row)) + ": column " + std::to_string(col) +
           " (" + what + ") names " + entity_text(r) + ", which is not a " +
           entity_name(want) + ".");
  }
  return r;
}

void Sketch::unused_entity_columns(std::size_t row, std::size_t first) const {
  for (std::size_t col = first; col < kEntityColumns; ++col) {
    if (t_.ent(row, col) != -1) {
      refuse(entity_text(static_cast<std::int64_t>(row)) + ": column " + std::to_string(col) +
             " holds " + std::to_string(t_.ent(row, col)) + "; a " +
             entity_name(t_.ent(row, 0)) + " row uses columns 1 to " +
             std::to_string(first - 1) + " and holds -1 after them.");
    }
  }
}

bool Sketch::same_point(std::int64_t a, std::int64_t b) const {
  if (a == b) {
    return true;
  }
  const std::size_t i = static_cast<std::size_t>(a);
  const std::size_t j = static_cast<std::size_t>(b);
  return t_.ent(i, 1) == t_.ent(j, 1) && t_.ent(i, 2) == t_.ent(j, 2);
}

void Sketch::check_entities() {
  const std::size_t e = t_.e();
  role_.assign(t_.n(), 0);
  splines_.assign(e, Spline());
  const auto x_of = [&](std::int64_t point) {
    return t_.params[static_cast<std::size_t>(t_.ent(static_cast<std::size_t>(point), 1))];
  };
  const auto y_of = [&](std::int64_t point) {
    return t_.params[static_cast<std::size_t>(t_.ent(static_cast<std::size_t>(point), 2))];
  };
  // Points first: the other kinds read their coordinates.
  for (std::size_t row = 0; row < e; ++row) {
    if (t_.ent(row, 0) == kPoint) {
      param_at(row, 1, "x");
      param_at(row, 2, "y");
      unused_entity_columns(row, 3);
    }
  }
  for (std::size_t row = 0; row < e; ++row) {
    const std::int64_t kind = t_.ent(row, 0);
    const std::int64_t self = static_cast<std::int64_t>(row);
    switch (kind) {
      case kPoint:
        break;
      case kLine: {
        const std::int64_t p1 = entity_at(row, 1, "start point", kPoint);
        const std::int64_t p2 = entity_at(row, 2, "end point", kPoint);
        unused_entity_columns(row, 3);
        if (x_of(p1) == x_of(p2) && y_of(p1) == y_of(p2)) {
          refuse(entity_text(self) + ": its points (entity rows " + std::to_string(p1) +
                 " and " + std::to_string(p2) + ") are both at (" + number_text(x_of(p1)) +
                 ", " + number_text(y_of(p1)) +
                 ") in the initial guess; a line of zero length has no direction.");
        }
        break;
      }
      case kCircle:
      case kArc: {
        entity_at(row, 1, "center", kPoint);
        std::size_t radius_col = 2;
        if (kind == kArc) {
          const std::int64_t c = t_.ent(row, 1);
          const std::int64_t a = entity_at(row, 2, "start point", kPoint);
          const std::int64_t b = entity_at(row, 3, "end point", kPoint);
          if (same_point(a, b) || same_point(c, a) || same_point(c, b)) {
            refuse(entity_text(self) + ": its center, start and end (entity rows " +
                   std::to_string(c) + ", " + std::to_string(a) + ", " + std::to_string(b) +
                   ") must be three different points.");
          }
          radius_col = 4;
          param_at(row, 5, "start angle");
          param_at(row, 6, "end angle");
          unused_entity_columns(row, 7);
        } else {
          unused_entity_columns(row, 3);
        }
        const std::int64_t r = param_at(row, radius_col, "radius");
        if (!(t_.params[static_cast<std::size_t>(r)] > 0.0)) {
          refuse(entity_text(self) + ": its radius (parameter " + std::to_string(r) +
                 ") is " + number_text(t_.params[static_cast<std::size_t>(r)]) +
                 " in the initial guess; a radius is above 0.");
        }
        break;
      }
      case kEllipse:
      case kEllipticArc: {
        const std::int64_t c = entity_at(row, 1, "center", kPoint);
        const std::int64_t f = entity_at(row, 2, "focus", kPoint);
        std::size_t radius_col = 3;
        if (kind == kEllipticArc) {
          const std::int64_t a = entity_at(row, 3, "start point", kPoint);
          const std::int64_t b = entity_at(row, 4, "end point", kPoint);
          if (same_point(a, b)) {
            refuse(entity_text(self) + ": its start and end (entity rows " +
                   std::to_string(a) + ", " + std::to_string(b) +
                   ") must be two different points.");
          }
          radius_col = 5;
          param_at(row, 6, "start angle");
          param_at(row, 7, "end angle");
        } else {
          unused_entity_columns(row, 4);
        }
        if (x_of(c) == x_of(f) && y_of(c) == y_of(f)) {
          refuse(entity_text(self) + ": its focus (entity row " + std::to_string(f) +
                 ") is at its center (entity row " + std::to_string(c) +
                 ") in the initial guess; the focus gives the major axis, so it lies off "
                 "the center.");
        }
        const std::int64_t r = param_at(row, radius_col, "minor radius");
        if (!(t_.params[static_cast<std::size_t>(r)] > 0.0)) {
          refuse(entity_text(self) + ": its minor radius (parameter " + std::to_string(r) +
                 ") is " + number_text(t_.params[static_cast<std::size_t>(r)]) +
                 " in the initial guess; a radius is above 0.");
        }
        break;
      }
      case kBSpline:
        unused_entity_columns(row, 5);
        break;
      default:
        refuse("entity row " + std::to_string(row) + " has kind code " + std::to_string(kind) +
               ", which is no entity kind (1 POINT, 2 LINE, 3 CIRCLE, 4 ARC, 5 ELLIPSE, 6 "
               "ELLIPTIC_ARC, 7 BSPLINE).");
    }
  }
  check_splines();
}

void Sketch::check_splines() {
  const std::size_t e = t_.e();
  std::vector<std::size_t> rows;
  for (std::size_t row = 0; row < e; ++row) {
    if (t_.ent(row, 0) == kBSpline) {
      rows.push_back(row);
    }
  }
  const std::vector<std::int64_t>& off = t_.offsets;
  const std::vector<std::int64_t>& idx = t_.indices;
  if (off.empty()) {
    if (!rows.empty()) {
      refuse(entity_text(static_cast<std::int64_t>(rows.front())) +
             " is a B-spline, and spline_offsets is not given; each B-spline names its "
             "slot of spline_offsets and spline_indices.");
    }
    if (!idx.empty()) {
      refuse("spline_indices holds " + std::to_string(idx.size()) +
             " values, and spline_offsets is not given.");
    }
    return;
  }
  if (off.front() != 0 || static_cast<std::size_t>(off.back()) != idx.size()) {
    refuse("spline_offsets must start at 0 and end at the length of spline_indices (" +
           std::to_string(idx.size()) + "); it runs from " + std::to_string(off.front()) +
           " to " + std::to_string(off.back()) + ".");
  }
  for (std::size_t k = 1; k < off.size(); ++k) {
    if (off[k] < off[k - 1]) {
      refuse("spline_offsets must not decrease; slot " + std::to_string(k - 1) +
             " runs from " + std::to_string(off[k - 1]) + " to " + std::to_string(off[k]) +
             ".");
    }
  }
  const std::size_t slots = off.size() - 1;
  std::vector<std::int64_t> owner(slots, -1);
  for (std::size_t row : rows) {
    const std::int64_t self = static_cast<std::int64_t>(row);
    const std::int64_t slot = t_.ent(row, 1);
    const std::int64_t degree = t_.ent(row, 2);
    const std::int64_t periodic = t_.ent(row, 3);
    const std::int64_t n = t_.ent(row, 4);
    if (slot < 0 || static_cast<std::size_t>(slot) >= slots) {
      refuse(entity_text(self) + ": column 1 (slot) holds " + std::to_string(slot) +
             ", outside the " + std::to_string(slots) + " slots of spline_offsets.");
    }
    if (owner[static_cast<std::size_t>(slot)] >= 0) {
      refuse(entity_text(self) + " and " + entity_text(owner[static_cast<std::size_t>(slot)]) +
             " name the same slot " + std::to_string(slot) + "; each B-spline has its own.");
    }
    owner[static_cast<std::size_t>(slot)] = self;
    if (degree < 1 || degree > 25) {
      refuse(entity_text(self) + ": column 2 (degree) holds " + std::to_string(degree) +
             "; a degree is 1 to 25.");
    }
    if (periodic != 0 && periodic != 1) {
      refuse(entity_text(self) + ": column 3 (periodic) holds " + std::to_string(periodic) +
             "; it is 0 or 1.");
    }
    const std::int64_t begin = off[static_cast<std::size_t>(slot)];
    const std::int64_t length = off[static_cast<std::size_t>(slot) + 1] - begin;
    if (n < 2 || length <= 2 * n || (length - 2 * n) % 2 != 0) {
      refuse(entity_text(self) + ": its slot " + std::to_string(slot) + " holds " +
             std::to_string(length) + " values for " + std::to_string(n) +
             " poles; a slot holds the poles, as many weights, then 2 or more knots and "
             "as many multiplicities.");
    }
    const std::int64_t m = (length - 2 * n) / 2;
    Spline& sp = splines_[row];
    sp.degree = static_cast<int>(degree);
    sp.periodic = periodic == 1;
    const auto at = [&](std::int64_t k) { return idx[static_cast<std::size_t>(begin + k)]; };
    for (std::int64_t k = 0; k < n; ++k) {
      const std::int64_t r = at(k);
      if (r < 0 || static_cast<std::size_t>(r) >= t_.e() ||
          t_.ent(static_cast<std::size_t>(r), 0) != kPoint) {
        refuse(entity_text(self) + ": pole " + std::to_string(k) + " is entity row " +
               std::to_string(r) + ", which is not a POINT row.");
      }
      sp.poles.push_back(r);
    }
    // The weights move (role 1); the knots do not (role 2). The B-splines come after
    // every other slot is marked, so a weight or knot that shares a parameter with a
    // slot of the other role is found here.
    const auto param_of = [&](std::int64_t k, const char* what, std::uint8_t role) {
      const std::int64_t i = at(k);
      if (i < 0 || static_cast<std::size_t>(i) >= t_.n()) {
        refuse(entity_text(self) + ": its " + what + " is parameter index " +
               std::to_string(i) + ", outside params of length " + std::to_string(t_.n()) +
               ".");
      }
      std::uint8_t& r = role_[static_cast<std::size_t>(i)];
      if (r != 0 && r != role) {
        refuse(entity_text(self) + " uses parameter " + std::to_string(i) + " as " + what +
               ", and another slot uses it as " +
               (role == 2 ? "a value that moves" : "a knot") +
               ": a knot does not move, so no other value can share its parameter.");
      }
      r = role;
      return i;
    };
    for (std::int64_t k = 0; k < n; ++k) {
      const std::string what = "weight " + std::to_string(k);
      const std::int64_t i = param_of(n + k, what.c_str(), 1);
      if (!(t_.params[static_cast<std::size_t>(i)] > 0.0)) {
        refuse(entity_text(self) + ": weight " + std::to_string(k) + " (parameter " +
               std::to_string(i) + ") is " +
               number_text(t_.params[static_cast<std::size_t>(i)]) +
               " in the initial guess; a weight is above 0.");
      }
      sp.weights.push_back(i);
    }
    for (std::int64_t k = 0; k < m; ++k) {
      const std::string what = "knot " + std::to_string(k);
      sp.knots.push_back(param_of(2 * n + k, what.c_str(), 2));
    }
    for (std::int64_t k = 0; k < m; ++k) {
      const std::int64_t mult = at(2 * n + m + k);
      if (mult < 1 || mult > degree + 1) {
        refuse(entity_text(self) + ": the multiplicity of knot " + std::to_string(k) +
               " is " + std::to_string(mult) + "; a multiplicity is 1 to the degree + 1 (" +
               std::to_string(degree + 1) + ").");
      }
      sp.mults.push_back(static_cast<int>(mult));
    }
    // OCCT checks the rest of the definition: knots that increase, and the count of poles
    // the degree, the multiplicities and the periodicity ask for.
    NCollection_Array1<gp_Pnt2d> poles(1, static_cast<int>(n));
    NCollection_Array1<double> weights(1, static_cast<int>(n));
    NCollection_Array1<double> knots(1, static_cast<int>(m));
    NCollection_Array1<int> mults(1, static_cast<int>(m));
    for (int k = 0; k < static_cast<int>(n); ++k) {
      const std::int64_t point = sp.poles[static_cast<std::size_t>(k)];
      poles.SetValue(k + 1,
                     gp_Pnt2d(t_.params[static_cast<std::size_t>(
                                  t_.ent(static_cast<std::size_t>(point), 1))],
                              t_.params[static_cast<std::size_t>(
                                  t_.ent(static_cast<std::size_t>(point), 2))]));
      weights.SetValue(k + 1, t_.params[static_cast<std::size_t>(sp.weights[k])]);
    }
    for (int k = 0; k < static_cast<int>(m); ++k) {
      knots.SetValue(k + 1, t_.params[static_cast<std::size_t>(sp.knots[k])]);
      mults.SetValue(k + 1, sp.mults[static_cast<std::size_t>(k)]);
    }
    try {
      sp.occt = new Geom2d_BSplineCurve(poles, weights, knots, mults, sp.degree, sp.periodic);
    } catch (const Standard_Failure& f) {
      refuse(entity_text(self) + ": OCCT refuses its definition (" + std::to_string(n) +
             " poles, degree " + std::to_string(degree) + ", " + std::to_string(m) +
             " knots, " + (sp.periodic ? "periodic" : "not periodic") + "): " +
             f.ExceptionType() + ": " + f.what() + ".");
    }
  }
  for (std::size_t k = 0; k < slots; ++k) {
    if (owner[k] < 0) {
      refuse("slot " + std::to_string(k) + " of spline_offsets is named by no B-spline row.");
    }
  }
}

// ---- Constraint rows ------------------------------------------------------------------- //

void Sketch::unused_constraint_columns(std::size_t row, std::size_t first) const {
  for (std::size_t col = first; col < kConstraintColumns; ++col) {
    if (t_.con(row, col) != -1) {
      refuse(row_text(row) + ": column " + std::to_string(col) + " holds " +
             std::to_string(t_.con(row, col)) + "; this kind uses columns 1 to " +
             std::to_string(first - 1) + " and holds -1 after them.");
    }
  }
}

// The entity row in column `col` of constraint row `row`, in range.
std::int64_t Sketch::operand(std::size_t row, std::size_t col, const char* what) const {
  const std::int64_t r = t_.con(row, col);
  if (r < 0 || static_cast<std::size_t>(r) >= t_.e()) {
    refuse(row_text(row) + ": column " + std::to_string(col) + " (" + what +
           ") holds entity row " + std::to_string(r) + ", outside the " +
           std::to_string(t_.e()) + " entity rows.");
  }
  return r;
}

// The optional point in column `col`: -1, or a POINT row.
std::int64_t Sketch::optional_point(std::size_t row, std::size_t col) const {
  const std::int64_t r = t_.con(row, col);
  if (r == -1) {
    return -1;
  }
  const std::int64_t p = operand(row, col, "point");
  if (t_.ent(static_cast<std::size_t>(p), 0) != kPoint) {
    refuse(row_text(row) + ": column " + std::to_string(col) + " (point) names " +
           entity_text(p) + ", which is not a POINT.");
  }
  return p;
}

// 0 when `point` is the first pole of the B-spline row `spline`, 1 when it is the last,
// -1 otherwise (a point row on the same two parameters counts as the pole).
int Sketch::spline_end(std::int64_t spline, std::int64_t point) const {
  const Spline& sp = splines_[static_cast<std::size_t>(spline)];
  if (same_point(point, sp.poles.front())) {
    return 0;
  }
  if (same_point(point, sp.poles.back())) {
    return 1;
  }
  return -1;
}

bool is_round(std::int64_t kind) { return kind == kCircle || kind == kArc; }
bool is_oval(std::int64_t kind) { return kind == kEllipse || kind == kEllipticArc; }

// TANGENT and PERPENDICULAR: the operands in columns 1 and 2, the optional point in
// column 3, and the mapping (Shaper's SketchSolver_ConstraintTangent::rebuild and
// SketchSolver_ConstraintPerpendicular::rebuild).
void Sketch::plan_curves(Plan& plan, std::size_t row, bool tangent) const {
  plan.a = operand(row, 1, "first curve");
  plan.b = operand(row, 2, "second curve");
  plan.p = optional_point(row, 3);
  unused_constraint_columns(row, 4);
  std::int64_t ka = t_.ent(static_cast<std::size_t>(plan.a), 0);
  std::int64_t kb = t_.ent(static_cast<std::size_t>(plan.b), 0);
  const std::string names = entity_text(plan.a) + " and " + entity_text(plan.b);
  if (plan.a == plan.b) {
    refuse(row_text(row) + " names " + entity_text(plan.a) + " twice.");
  }
  if (ka == kPoint || kb == kPoint) {
    refuse(row_text(row) + " names " + names + "; both operands are curves.");
  }
  if (tangent && ka == kLine && kb == kLine) {
    refuse(row_text(row) + " names " + names +
           "; a tangent needs a circle, arc, ellipse, elliptic arc or B-spline (two lines "
           "take PARALLEL or COLLINEAR).");
  }
  if (!tangent && is_round(ka) && is_round(kb)) {
    refuse(row_text(row) + " names " + names +
           "; Shaper's perpendicular takes two lines, a line and a circle or arc, or a "
           "pair with an ellipse, elliptic arc or B-spline, not two circles or arcs.");
  }
  // A B-spline operand is its end segment at a shared end pole (Shaper's substitution):
  // PlaneGCS gives a B-spline's direction by point only there.
  for (int side = 0; side < 2; ++side) {
    const std::int64_t e = side == 0 ? plan.a : plan.b;
    if (t_.ent(static_cast<std::size_t>(e), 0) != kBSpline) {
      continue;
    }
    const Spline& sp = splines_[static_cast<std::size_t>(e)];
    const bool clamped = !sp.periodic && sp.mults.front() > sp.degree &&
                         sp.mults.back() > sp.degree;
    const int end = plan.p >= 0 ? spline_end(e, plan.p) : -1;
    if (!clamped || end < 0) {
      const std::string at =
          plan.p >= 0 ? " at " + entity_text(plan.p) : std::string(" with no point");
      refuse(row_text(row) + " names " + entity_text(e) + at +
             "; PlaneGCS gives a B-spline's direction only at the first or the last pole of "
             "a B-spline that is clamped there (not periodic, end multiplicities of the "
             "degree + 1), so the point is that pole.");
    }
    (side == 0 ? plan.end_a : plan.end_b) = end;
    (side == 0 ? ka : kb) = kLine;
  }
  const bool line_round = (ka == kLine && is_round(kb)) || (is_round(ka) && kb == kLine);
  const bool line_oval = (ka == kLine && is_oval(kb)) || (is_oval(ka) && kb == kLine);
  const bool full_circle = ka == kCircle || kb == kCircle;
  if (plan.end_a >= 0 || plan.end_b >= 0) {
    // Shaper's B-spline case: the curves at the shared point.
    plan.map = Map::kAngleViaPoint;
  } else if (!tangent && ka == kLine && kb == kLine) {
    // Shaper's createConstraintPerpendicular: two lines, whatever point they share.
    plan.map = Map::kLineLine;
  } else if (!tangent && line_round) {
    plan.map = Map::kCenterOnLine;
  } else if (tangent && line_round && (plan.p < 0 || full_circle)) {
    // Shaper's createArcLineTangency: the distance of the center, unless an arc and the
    // line share a point; a full circle takes the distance whatever point it shares.
    plan.map = Map::kLineCircle;
  } else if (tangent && is_round(ka) && is_round(kb) && plan.p < 0) {
    plan.map = Map::kCircleCircle;
  } else if (tangent && line_oval && plan.p < 0) {
    plan.map = Map::kLineEllipse;
  } else if (plan.p >= 0) {
    plan.map = Map::kAngleViaPoint;
  } else {
    refuse(row_text(row) + " names " + names +
           " and no point; this pair needs the point where the curves meet in column 3 (a "
           "point the caller puts on both curves, as Shaper's own construction point).");
  }
  if (plan.map == Map::kAngleViaPoint) {
    plan.aux_count = 1;  // the angle: 0 or pi (tangent), pi/2 or -pi/2 (perpendicular)
  }
}

// RIGID: Shaper's SketchSolver_ConstraintFixed, one equality per parameter of the entity
// (Shaper's toParameters list), each to a copy of its initial value. Parameters fixed by
// the mask are left out: they cannot move.
void Sketch::plan_rigid(Plan& plan, std::size_t row) const {
  plan.a = operand(row, 1, "entity");
  unused_constraint_columns(row, 2);
  const std::size_t e = static_cast<std::size_t>(plan.a);
  std::vector<std::int64_t> params;
  const auto point = [&](std::int64_t p) {
    params.push_back(t_.ent(static_cast<std::size_t>(p), 1));
    params.push_back(t_.ent(static_cast<std::size_t>(p), 2));
  };
  switch (t_.ent(e, 0)) {
    case kPoint:
      point(plan.a);
      break;
    case kLine:
      point(t_.ent(e, 1));
      point(t_.ent(e, 2));
      break;
    case kCircle:
      point(t_.ent(e, 1));
      params.push_back(t_.ent(e, 2));
      break;
    case kArc:
      point(t_.ent(e, 1));
      params.push_back(t_.ent(e, 4));
      params.push_back(t_.ent(e, 5));
      params.push_back(t_.ent(e, 6));
      break;
    case kEllipse:
      point(t_.ent(e, 1));
      point(t_.ent(e, 2));
      params.push_back(t_.ent(e, 3));
      break;
    case kEllipticArc:
      point(t_.ent(e, 1));
      point(t_.ent(e, 2));
      params.push_back(t_.ent(e, 5));
      params.push_back(t_.ent(e, 6));
      params.push_back(t_.ent(e, 7));
      break;
    case kBSpline:
      for (std::int64_t pole : splines_[e].poles) {
        point(pole);
      }
      break;
    default:
      break;
  }
  for (std::int64_t i : params) {
    const std::size_t k = static_cast<std::size_t>(i);
    const bool listed =
        std::find(plan.rigid.begin(), plan.rigid.end(), k) != plan.rigid.end();
    if (t_.fixed[k] == 0 && !listed) {
      plan.rigid.push_back(k);
    }
  }
  plan.aux_count = plan.rigid.size();
}

Plan Sketch::plan_row(std::size_t row) const {
  Plan plan;
  plan.kind = t_.con(row, 0);
  const auto kind_of = [&](std::int64_t e) { return t_.ent(static_cast<std::size_t>(e), 0); };
  const auto need = [&](std::int64_t e, std::size_t col, std::initializer_list<std::int64_t> kinds,
                        const char* what) {
    for (std::int64_t k : kinds) {
      if (kind_of(e) == k) {
        return;
      }
    }
    refuse(row_text(row) + ": column " + std::to_string(col) + " names " + entity_text(e) +
           "; it takes " + what + ".");
  };
  const auto two = [&](const char* first, const char* second) {
    plan.a = operand(row, 1, first);
    plan.b = operand(row, 2, second);
    if (plan.a == plan.b) {
      refuse(row_text(row) + " names " + entity_text(plan.a) + " twice.");
    }
  };
  // The value of a dimensional row; NaN in the others.
  const double v = value(row);
  if (is_dimensional(plan.kind)) {
    if (!std::isfinite(v)) {
      refuse(row_text(row) + " is dimensional, and values[" + std::to_string(row) +
             "] is " + number_text(v) + "; a dimension is a finite number.");
    }
    if ((plan.kind == kLength || plan.kind == kRadius) && !(v > 0.0)) {
      refuse(row_text(row) + ": values[" + std::to_string(row) + "] is " + number_text(v) +
             "; a " + (plan.kind == kLength ? "length" : "radius") + " is above 0.");
    }
    if (plan.kind == kDistance && v < 0.0) {
      refuse(row_text(row) + ": values[" + std::to_string(row) + "] is " + number_text(v) +
             "; a distance is 0 or above.");
    }
  } else if (plan.kind >= kCoincident && plan.kind <= kSymmetric && !std::isnan(v)) {
    refuse(row_text(row) + " is not dimensional, and values[" + std::to_string(row) +
           "] is " + number_text(v) + "; it holds NaN.");
  }
  switch (plan.kind) {
    case kCoincident: {
      two("first entity", "second entity");
      unused_constraint_columns(row, 3);
      const std::string named = entity_text(plan.a) + " and " + entity_text(plan.b);
      if (kind_of(plan.a) != kPoint) {
        std::swap(plan.a, plan.b);
      }
      if (kind_of(plan.a) != kPoint) {
        refuse(row_text(row) + " names " + named + "; one of them is a point.");
      }
      switch (kind_of(plan.b)) {
        case kPoint:
          if (same_point(plan.a, plan.b)) {
            refuse(row_text(row) + " names " + entity_text(plan.a) + " and " +
                   entity_text(plan.b) + ", one point (the same two parameters).");
          }
          plan.map = Map::kPointPoint;
          break;
        case kLine:
          plan.map = Map::kPointLine;
          break;
        case kCircle:
          plan.map = Map::kPointCircle;
          break;
        case kArc:
          plan.map = Map::kPointArc;
          break;
        case kEllipse:
        case kEllipticArc:
          plan.map = Map::kPointEllipse;
          break;
        default:
          plan.map = Map::kPointBSpline;
          plan.aux_count = 1;  // the point's parameter on the B-spline
          break;
      }
      break;
    }
    case kTangent:
      plan_curves(plan, row, true);
      break;
    case kPerpendicular:
      plan_curves(plan, row, false);
      break;
    case kParallel:
    case kAngle:
      two("first line", "second line");
      unused_constraint_columns(row, 3);
      need(plan.a, 1, {kLine}, "a LINE");
      need(plan.b, 2, {kLine}, "a LINE");
      break;
    case kHorizontal:
    case kVertical:
    case kLength:
      plan.a = operand(row, 1, "line");
      unused_constraint_columns(row, 2);
      need(plan.a, 1, {kLine}, "a LINE");
      break;
    case kRadius:
      plan.a = operand(row, 1, "circle or arc");
      unused_constraint_columns(row, 2);
      need(plan.a, 1, {kCircle, kArc}, "a CIRCLE or an ARC");
      break;
    case kEqual: {
      two("first entity", "second entity");
      unused_constraint_columns(row, 3);
      const std::int64_t ka = kind_of(plan.a);
      const std::int64_t kb = kind_of(plan.b);
      if (ka == kLine && kb == kLine) {
        plan.map = Map::kEqualLength;
      } else if (is_round(ka) && is_round(kb)) {
        plan.map = Map::kEqualRadius;
      } else if (is_oval(ka) && is_oval(kb)) {
        plan.map = Map::kEqualRadii;
      } else {
        refuse(row_text(row) + " names " + entity_text(plan.a) + " and " +
               entity_text(plan.b) +
               "; equal takes two lines (lengths), two circles or arcs (radii) or two "
               "ellipses or elliptic arcs (radii), as Shaper's.");
      }
      break;
    }
    case kCollinear: {
      two("first line", "second line");
      unused_constraint_columns(row, 3);
      need(plan.a, 1, {kLine}, "a LINE");
      need(plan.b, 2, {kLine}, "a LINE");
      plan.map = Map::kCollinear;
      const std::size_t a = static_cast<std::size_t>(plan.a);
      const std::size_t b = static_cast<std::size_t>(plan.b);
      const auto on_a = [&](std::int64_t p) {
        return same_point(p, t_.ent(a, 1)) || same_point(p, t_.ent(a, 2));
      };
      // A point of line b that is already a point of line a lies on it (Shaper's
      // SketchSolver_ConstraintCollinear skips the points that are on the other line).
      plan.on_first = !on_a(t_.ent(b, 1));
      plan.on_second = !on_a(t_.ent(b, 2));
      if (!plan.on_first && !plan.on_second) {
        refuse(row_text(row) + " names " + entity_text(plan.a) + " and " +
               entity_text(plan.b) + ", which share both points: they are one line.");
      }
      break;
    }
    case kMidpoint: {
      plan.a = operand(row, 1, "point");
      plan.b = operand(row, 2, "line or arc");
      unused_constraint_columns(row, 3);
      need(plan.a, 1, {kPoint}, "a POINT");
      need(plan.b, 2, {kLine, kArc, kEllipticArc}, "a LINE, an ARC or an ELLIPTIC_ARC");
      if (kind_of(plan.b) == kLine) {
        plan.map = Map::kMidpointLine;
      } else {
        const std::size_t b = static_cast<std::size_t>(plan.b);
        const std::size_t a0 = static_cast<std::size_t>(t_.ent(b, kind_of(plan.b) == kArc ? 5 : 6));
        const std::size_t a1 = static_cast<std::size_t>(t_.ent(b, kind_of(plan.b) == kArc ? 6 : 7));
        if (!(t_.params[a1] > t_.params[a0])) {
          refuse(row_text(row) + " names " + entity_text(plan.b) + ", whose end angle " +
                 number_text(t_.params[a1]) + " is not above its start angle " +
                 number_text(t_.params[a0]) +
                 " in the initial guess; an arc runs counter-clockwise from its start angle "
                 "to a larger end angle, and its midpoint is at their mean.");
        }
        plan.map = Map::kMidpointArc;
        plan.aux_count = 2;  // the midpoint's angle and the half sweep, as Shaper's
      }
      break;
    }
    case kRigid:
      plan_rigid(plan, row);
      plan.map = Map::kRigid;
      break;
    case kInternalCoincident: {
      plan.a = operand(row, 1, "curve");
      plan.b = operand(row, 2, "point");
      need(plan.b, 2, {kPoint}, "a POINT");
      plan.role = t_.con(row, 3);
      plan.index = t_.con(row, 4);
      const std::int64_t ka = kind_of(plan.a);
      if (is_oval(ka)) {
        if (plan.role < kMajorPositive || plan.role > kFocus2 || plan.index != -1) {
          refuse(row_text(row) + " names " + entity_text(plan.a) + " with role " +
                 std::to_string(plan.role) + " and index " + std::to_string(plan.index) +
                 "; an ellipse's roles are 1 MAJOR_POSITIVE, 2 MAJOR_NEGATIVE, 3 "
                 "MINOR_POSITIVE, 4 MINOR_NEGATIVE and 5 FOCUS2, with index -1.");
        }
        plan.map = Map::kInternalEllipse;
      } else if (ka == kBSpline) {
        const Spline& sp = splines_[static_cast<std::size_t>(plan.a)];
        const std::int64_t count = plan.role == kPole
                                       ? static_cast<std::int64_t>(sp.poles.size())
                                       : static_cast<std::int64_t>(sp.knots.size());
        if ((plan.role != kPole && plan.role != kKnot) || plan.index < 0 ||
            plan.index >= count) {
          refuse(row_text(row) + " names " + entity_text(plan.a) + " with role " +
                 std::to_string(plan.role) + " and index " + std::to_string(plan.index) +
                 "; a B-spline's roles are 6 POLE (index 0 to " +
                 std::to_string(sp.poles.size() - 1) + ") and 7 KNOT (index 0 to " +
                 std::to_string(sp.knots.size() - 1) + ").");
        }
        plan.map = plan.role == kPole ? Map::kInternalPole : Map::kInternalKnot;
      } else {
        refuse(row_text(row) + ": column 1 names " + entity_text(plan.a) +
               "; an internal coincidence takes an ELLIPSE, an ELLIPTIC_ARC or a BSPLINE.");
      }
      break;
    }
    case kDistance: {
      two("first entity", "second entity");
      unused_constraint_columns(row, 3);
      const std::string named = entity_text(plan.a) + " and " + entity_text(plan.b);
      if (kind_of(plan.a) != kPoint) {
        std::swap(plan.a, plan.b);
      }
      const std::int64_t kb = kind_of(plan.b);
      if (kind_of(plan.a) != kPoint || (kb != kPoint && kb != kLine)) {
        refuse(row_text(row) + " names " + named +
               "; a distance takes two points or a point and a line.");
      }
      if (kb == kPoint && same_point(plan.a, plan.b)) {
        refuse(row_text(row) + " names " + entity_text(plan.a) + " and " +
               entity_text(plan.b) + ", one point (the same two parameters).");
      }
      // Shaper's SketchSolver_ConstraintDistance: a zero distance is a coincidence.
      if (v == 0.0) {
        plan.map = kb == kPoint ? Map::kPointPoint : Map::kPointLine;
      } else {
        plan.map = kb == kPoint ? Map::kPointPointDistance : Map::kPointLineDistance;
      }
      break;
    }
    case kSymmetric: {
      two("first point", "second point");
      plan.p = operand(row, 3, "line");
      unused_constraint_columns(row, 4);
      need(plan.a, 1, {kPoint}, "a POINT");
      need(plan.b, 2, {kPoint}, "a POINT");
      need(plan.p, 3, {kLine}, "a LINE");
      if (same_point(plan.a, plan.b)) {
        refuse(row_text(row) + " names " + entity_text(plan.a) + " and " +
               entity_text(plan.b) + ", one point (the same two parameters).");
      }
      break;
    }
    default:
      refuse("constraint row " + std::to_string(row) + " has kind code " +
             std::to_string(plan.kind) +
             ", which is no constraint kind (1 COINCIDENT to 16 SYMMETRIC).");
  }
  return plan;
}

void Sketch::check_constraints() {
  plans_.clear();
  plans_.reserve(t_.c());
  for (std::size_t row = 0; row < t_.c(); ++row) {
    plans_.push_back(plan_row(row));
  }
}

// ---- Building the PlaneGCS system ------------------------------------------------------ //

void Sketch::build_geometry() {
  const std::size_t e = t_.e();
  geo_.assign(e, Geometry());
  for (std::size_t row = 0; row < e; ++row) {
    geo_[row].kind = t_.ent(row, 0);
    if (geo_[row].kind == kPoint) {
      geo_[row].point = GCS::Point(s(t_.ent(row, 1)), s(t_.ent(row, 2)));
    }
  }
  for (std::size_t row = 0; row < e; ++row) {
    Geometry& g = geo_[row];
    switch (g.kind) {
      case kLine:
        g.line.p1 = pt(t_.ent(row, 1));
        g.line.p2 = pt(t_.ent(row, 2));
        break;
      case kCircle:
        g.circle.center = pt(t_.ent(row, 1));
        g.circle.rad = s(t_.ent(row, 2));
        break;
      case kArc:
        g.arc.center = pt(t_.ent(row, 1));
        g.arc.start = pt(t_.ent(row, 2));
        g.arc.end = pt(t_.ent(row, 3));
        g.arc.rad = s(t_.ent(row, 4));
        g.arc.startAngle = s(t_.ent(row, 5));
        g.arc.endAngle = s(t_.ent(row, 6));
        break;
      case kEllipse:
        g.ellipse.center = pt(t_.ent(row, 1));
        g.ellipse.focus1 = pt(t_.ent(row, 2));
        g.ellipse.radmin = s(t_.ent(row, 3));
        break;
      case kEllipticArc:
        g.elliptic_arc.center = pt(t_.ent(row, 1));
        g.elliptic_arc.focus1 = pt(t_.ent(row, 2));
        g.elliptic_arc.start = pt(t_.ent(row, 3));
        g.elliptic_arc.end = pt(t_.ent(row, 4));
        g.elliptic_arc.radmin = s(t_.ent(row, 5));
        g.elliptic_arc.startAngle = s(t_.ent(row, 6));
        g.elliptic_arc.endAngle = s(t_.ent(row, 7));
        break;
      case kBSpline: {
        // Shaper's createBSpline: start and end are the first and the last pole.
        const Spline& sp = splines_[row];
        for (std::int64_t pole : sp.poles) {
          g.bspline.poles.push_back(pt(pole));
        }
        for (std::int64_t w : sp.weights) {
          g.bspline.weights.push_back(s(w));
        }
        for (std::int64_t k : sp.knots) {
          g.bspline.knots.push_back(s(k));
        }
        g.bspline.mult = sp.mults;
        g.bspline.degree = sp.degree;
        g.bspline.periodic = sp.periodic;
        g.bspline.start = g.bspline.poles.front();
        g.bspline.end = g.bspline.poles.back();
        g.bspline.setupFlattenedKnots();
        break;
      }
      default:
        break;
    }
  }
}

// The rules that tie an arc's points to its radius and angles, tag 0: Shaper's
// createArcConstraints (two P2PDistance, two P2PAngle) and createEllipticArcConstraints
// (four CurveValue, as System::addConstraintArcOfEllipseRules adds them).
void Sketch::add_entity_rules() {
  for (Geometry& g : geo_) {
    if (g.kind == kArc) {
      sys_.addConstraintP2PDistance(g.arc.center, g.arc.start, g.arc.rad, 0, true);
      sys_.addConstraintP2PDistance(g.arc.center, g.arc.end, g.arc.rad, 0, true);
      sys_.addConstraintP2PAngle(g.arc.center, g.arc.start, g.arc.startAngle, 0.0, 0, true);
      sys_.addConstraintP2PAngle(g.arc.center, g.arc.end, g.arc.endAngle, 0.0, 0, true);
      added_ += 4;
    } else if (g.kind == kEllipticArc) {
      sys_.addConstraintArcOfEllipseRules(g.elliptic_arc, 0, true);
      added_ += 4;
    }
  }
}

// The curve of an operand: the entity's own, or the end segment of a B-spline (its first
// two or last two poles).
GCS::Curve& Sketch::operand_curve(std::int64_t row, int end) {
  Geometry& g = geo_[static_cast<std::size_t>(row)];
  if (end < 0) {
    return *g.curve();
  }
  const GCS::VEC_P& poles = g.bspline.poles;
  GCS::Line segment;
  segment.p1 = end == 0 ? poles[0] : poles[poles.size() - 2];
  segment.p2 = end == 0 ? poles[1] : poles[poles.size() - 1];
  segments_.push_back(segment);
  return segments_.back();
}

// The parameter of the point of B-spline `spline_row` nearest to point `point_row` in the
// initial guess: the projection on the curve, or an end when it is nearer (Shaper's
// GeomAPI_BSpline2d::parameter with no distance limit).
double Sketch::nearest_parameter(std::int64_t spline_row, std::int64_t point_row) const {
  const Handle(Geom2d_BSplineCurve)& curve = splines_[static_cast<std::size_t>(spline_row)].occt;
  const std::size_t p = static_cast<std::size_t>(point_row);
  const gp_Pnt2d point(t_.params[static_cast<std::size_t>(t_.ent(p, 1))],
                       t_.params[static_cast<std::size_t>(t_.ent(p, 2))]);
  double best = curve->FirstParameter();
  double best_distance = point.Distance(curve->Value(best));
  const double last = curve->LastParameter();
  if (point.Distance(curve->Value(last)) < best_distance) {
    best = last;
    best_distance = point.Distance(curve->Value(last));
  }
  Geom2dAPI_ProjectPointOnCurve projection(point, curve);
  if (projection.NbPoints() > 0 && projection.LowerDistance() < best_distance) {
    best = projection.LowerDistanceParameter();
  }
  return best;
}

void Sketch::add_row(std::size_t row, Plan& plan) {
  const int tag = static_cast<int>(row) + 1;
  double* value_ptr = s(t_.n() + row);
  GCS::Point& pa = geo_[static_cast<std::size_t>(plan.a)].point;
  const auto count = [&](std::size_t k) { added_ += k; };
  switch (plan.kind) {
    case kCoincident:
    case kDistance: {
      Geometry& b = geo_[static_cast<std::size_t>(plan.b)];
      switch (plan.map) {
        case Map::kPointPoint:
          sys_.addConstraintP2PCoincident(pa, b.point, tag, true);
          count(2);
          break;
        case Map::kPointLine:
          sys_.addConstraintPointOnLine(pa, b.line, tag, true);
          count(1);
          break;
        case Map::kPointCircle:
          sys_.addConstraintPointOnCircle(pa, b.circle, tag, true);
          count(1);
          break;
        case Map::kPointArc:
          sys_.addConstraintPointOnArc(pa, b.arc, tag, true);
          count(1);
          break;
        case Map::kPointEllipse:
          sys_.addConstraintPointOnEllipse(
              pa, b.kind == kEllipse ? b.ellipse : static_cast<GCS::Ellipse&>(b.elliptic_arc),
              tag, true);
          count(1);
          break;
        case Map::kPointBSpline:
          *s(plan.aux) = nearest_parameter(plan.b, plan.a);
          aux_unknowns_.push_back(plan.aux);
          sys_.addConstraintPointOnBSpline(pa, b.bspline, s(plan.aux), tag, true);
          count(2);
          break;
        case Map::kPointPointDistance:
          sys_.addConstraintP2PDistance(pa, b.point, value_ptr, tag, true);
          count(1);
          break;
        case Map::kPointLineDistance:
          sys_.addConstraintP2LDistance(pa, b.line, value_ptr, tag, true);
          count(1);
          break;
        default:
          break;
      }
      break;
    }
    case kTangent:
    case kPerpendicular: {
      Geometry& ga = geo_[static_cast<std::size_t>(plan.a)];
      Geometry& gb = geo_[static_cast<std::size_t>(plan.b)];
      // The line and the round operand, in that order, for the mappings that take them.
      Geometry& line = ga.kind == kLine ? ga : gb;
      Geometry& other = ga.kind == kLine ? gb : ga;
      switch (plan.map) {
        case Map::kLineLine:
          sys_.addConstraintPerpendicular(ga.line, gb.line, tag, true);
          count(1);
          break;
        case Map::kCenterOnLine:
          sys_.addConstraintPointOnLine(other.kind == kArc ? other.arc.center
                                                           : other.circle.center,
                                        line.line, tag, true);
          count(1);
          break;
        case Map::kLineCircle:
          if (other.kind == kArc) {
            sys_.addConstraintTangent(line.line, other.arc, tag, true);
          } else {
            sys_.addConstraintTangent(line.line, other.circle, tag, true);
          }
          count(1);
          break;
        case Map::kCircleCircle: {
          // System::addConstraintTangent(Circle, Circle): internal when the centers are
          // nearer than either radius in the initial guess (Shaper's
          // isArcArcTangencyInternal).
          GCS::Circle& c1 = ga.kind == kArc ? ga.arc : ga.circle;
          GCS::Circle& c2 = gb.kind == kArc ? gb.arc : gb.circle;
          const double dx = *c2.center.x - *c1.center.x;
          const double dy = *c2.center.y - *c1.center.y;
          const double d = std::sqrt(dx * dx + dy * dy);
          sys_.addConstraintTangentCircumf(c1.center, c2.center, c1.rad, c2.rad,
                                           d < *c1.rad || d < *c2.rad, tag, true);
          count(1);
          break;
        }
        case Map::kLineEllipse:
          sys_.addConstraintTangent(line.line,
                                    other.kind == kEllipse
                                        ? other.ellipse
                                        : static_cast<GCS::Ellipse&>(other.elliptic_arc),
                                    tag, true);
          count(1);
          break;
        case Map::kAngleViaPoint: {
          GCS::Curve& c1 = operand_curve(plan.a, plan.end_a);
          GCS::Curve& c2 = operand_curve(plan.b, plan.end_b);
          GCS::Point& p = pt(plan.p);
          // Shaper's adjustAngleBetweenCurves: the angle the curves make at the point in
          // the initial guess, brought to [-pi, pi], decides between the two values.
          double angle = sys_.calculateAngleViaPoint(c1, c2, p);
          if (angle > kPi) {
            angle -= 2.0 * kPi;
          }
          if (angle < -kPi) {
            angle += 2.0 * kPi;
          }
          if (plan.kind == kTangent) {
            *s(plan.aux) = std::abs(angle) <= kPi / 2.0 ? 0.0 : kPi;
          } else {
            *s(plan.aux) = angle >= 0.0 ? kPi / 2.0 : -kPi / 2.0;
          }
          sys_.addConstraintAngleViaPoint(c1, c2, p, s(plan.aux), tag, true);
          count(1);
          break;
        }
        default:
          break;
      }
      break;
    }
    case kParallel:
      sys_.addConstraintParallel(geo_[static_cast<std::size_t>(plan.a)].line,
                                 geo_[static_cast<std::size_t>(plan.b)].line, tag, true);
      count(1);
      break;
    case kHorizontal:
      sys_.addConstraintHorizontal(geo_[static_cast<std::size_t>(plan.a)].line, tag, true);
      count(1);
      break;
    case kVertical:
      sys_.addConstraintVertical(geo_[static_cast<std::size_t>(plan.a)].line, tag, true);
      count(1);
      break;
    case kEqual: {
      Geometry& ga = geo_[static_cast<std::size_t>(plan.a)];
      Geometry& gb = geo_[static_cast<std::size_t>(plan.b)];
      if (plan.map == Map::kEqualLength) {
        sys_.addConstraintEqualLength(ga.line, gb.line, tag, true);
        count(1);
      } else if (plan.map == Map::kEqualRadius) {
        GCS::Circle& c1 = ga.kind == kArc ? ga.arc : ga.circle;
        GCS::Circle& c2 = gb.kind == kArc ? gb.arc : gb.circle;
        sys_.addConstraintEqual(c1.rad, c2.rad, tag, true);
        count(1);
      } else {
        GCS::Ellipse& e1 = ga.kind == kEllipse ? ga.ellipse : ga.elliptic_arc;
        GCS::Ellipse& e2 = gb.kind == kEllipse ? gb.ellipse : gb.elliptic_arc;
        sys_.addConstraintEqualRadii(e1, e2, tag, true);
        count(2);
      }
      break;
    }
    case kCollinear: {
      GCS::Line& la = geo_[static_cast<std::size_t>(plan.a)].line;
      GCS::Line& lb = geo_[static_cast<std::size_t>(plan.b)].line;
      if (plan.on_first) {
        sys_.addConstraintPointOnLine(lb.p1, la, tag, true);
        count(1);
      }
      if (plan.on_second) {
        sys_.addConstraintPointOnLine(lb.p2, la, tag, true);
        count(1);
      }
      break;
    }
    case kMidpoint: {
      Geometry& gb = geo_[static_cast<std::size_t>(plan.b)];
      if (plan.map == Map::kMidpointLine) {
        // Shaper's createConstraintMiddlePoint for a line.
        sys_.addConstraintPointOnLine(pa, gb.line, tag, true);
        sys_.addConstraintPointOnPerpBisector(pa, gb.line, tag, true);
        count(2);
      } else {
        // Shaper's createConstraintMiddlePointOnArc: the point at angle u on the arc, with
        // u - start = diff = end - u.
        GCS::Curve& arc = *gb.curve();
        double* a0 = gb.kind == kArc ? gb.arc.startAngle : gb.elliptic_arc.startAngle;
        double* a1 = gb.kind == kArc ? gb.arc.endAngle : gb.elliptic_arc.endAngle;
        double* u = s(plan.aux);
        double* diff = s(plan.aux + 1);
        *u = (*a0 + *a1) * 0.5;
        *diff = (*a1 - *a0) * 0.5;
        aux_unknowns_.push_back(plan.aux);
        aux_unknowns_.push_back(plan.aux + 1);
        sys_.addConstraintCurveValue(pa, arc, u, tag, true);
        sys_.addConstraintDifference(a0, u, diff, tag, true);
        sys_.addConstraintDifference(u, a1, diff, tag, true);
        count(4);
      }
      break;
    }
    case kRigid:
      for (std::size_t k = 0; k < plan.rigid.size(); ++k) {
        *s(plan.aux + k) = store_[plan.rigid[k]];
        sys_.addConstraintEqual(s(plan.aux + k), s(plan.rigid[k]), tag, true);
        count(1);
      }
      break;
    case kLength: {
      GCS::Line& l = geo_[static_cast<std::size_t>(plan.a)].line;
      sys_.addConstraintP2PDistance(l.p1, l.p2, value_ptr, tag, true);
      count(1);
      break;
    }
    case kAngle:
      sys_.addConstraintL2LAngle(geo_[static_cast<std::size_t>(plan.a)].line,
                                 geo_[static_cast<std::size_t>(plan.b)].line, value_ptr, tag,
                                 true);
      count(1);
      break;
    case kRadius: {
      Geometry& g = geo_[static_cast<std::size_t>(plan.a)];
      if (g.kind == kArc) {
        sys_.addConstraintArcRadius(g.arc, value_ptr, tag, true);
      } else {
        sys_.addConstraintCircleRadius(g.circle, value_ptr, tag, true);
      }
      count(1);
      break;
    }
    case kInternalCoincident: {
      Geometry& g = geo_[static_cast<std::size_t>(plan.a)];
      GCS::Point& p = pt(plan.b);
      if (plan.map == Map::kInternalEllipse) {
        GCS::Ellipse& e = g.kind == kEllipse ? g.ellipse : g.elliptic_arc;
        if (plan.role == kFocus2) {
          sys_.addConstraintInternalAlignmentEllipseFocus2(e, p, tag, true);
        } else {
          // Shaper's createEllipseConstraints: the x and the y alignment of one point.
          const GCS::InternalAlignmentType x =
              plan.role == kMajorPositive   ? GCS::EllipsePositiveMajorX
              : plan.role == kMajorNegative ? GCS::EllipseNegativeMajorX
              : plan.role == kMinorPositive ? GCS::EllipsePositiveMinorX
                                            : GCS::EllipseNegativeMinorX;
          const GCS::InternalAlignmentType y =
              static_cast<GCS::InternalAlignmentType>(static_cast<int>(x) + 1);
          sys_.addConstraintInternalAlignmentPoint2Ellipse(e, p, x, tag, true);
          sys_.addConstraintInternalAlignmentPoint2Ellipse(e, p, y, tag, true);
        }
        count(2);
      } else if (plan.map == Map::kInternalPole) {
        GCS::Point& pole = g.bspline.poles[static_cast<std::size_t>(plan.index)];
        sys_.addConstraintEqual(pole.x, p.x, tag, true,
                                GCS::Constraint::Alignment::InternalAlignment);
        sys_.addConstraintEqual(pole.y, p.y, tag, true,
                                GCS::Constraint::Alignment::InternalAlignment);
        count(2);
      } else {
        sys_.addConstraintInternalAlignmentKnotPoint(
            g.bspline, p, static_cast<unsigned int>(plan.index), tag, true);
        count(1);
      }
      break;
    }
    case kSymmetric:
      sys_.addConstraintP2PSymmetric(pa, pt(plan.b), geo_[static_cast<std::size_t>(plan.p)].line,
                                     tag, true);
      count(2);
      break;
    default:
      break;
  }
}

std::vector<std::int64_t> Sketch::rows_of(const GCS::VEC_I& tags) const {
  std::vector<std::int64_t> rows;
  for (int tag : tags) {
    if (tag > 0) {
      rows.push_back(tag - 1);
    }
  }
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  return rows;
}

Outcome Sketch::unsolved_outcome(const std::string& status, int dof) const {
  Outcome out;
  out.params = t_.params;
  out.status = status;
  out.dof = dof;
  return out;
}

Outcome Sketch::solve() {
  check_entities();
  check_constraints();
  // One buffer of scalars, sized once: params, values, auxiliary scalars.
  std::size_t aux = t_.n() + t_.c();
  for (Plan& plan : plans_) {
    plan.aux = aux;
    aux += plan.aux_count;
  }
  store_.assign(aux, 0.0);
  std::copy(t_.params.begin(), t_.params.end(), store_.begin());
  std::copy(t_.values.begin(), t_.values.end(),
            store_.begin() + static_cast<std::ptrdiff_t>(t_.n()));
  build_geometry();

  // PlaneGCS's defaults, with the caller's limits: one iteration limit and one tolerance
  // for each algorithm (DogLeg's largest residual, Levenberg-Marquardt's residual norm,
  // BFGS's step), in the solve and in the diagnosis's solve of the redundant constraints.
  sys_.maxIter = max_iterations_;
  sys_.maxIterRedundant = max_iterations_;
  sys_.sketchSizeMultiplier = false;
  sys_.sketchSizeMultiplierRedundant = false;
  sys_.convergence = tolerance_;
  sys_.convergenceRedundant = tolerance_;
  sys_.DL_tolf = tolerance_;
  sys_.DL_tolfRedundant = tolerance_;
  sys_.LM_eps = tolerance_;
  sys_.LM_epsRedundant = tolerance_;
  sys_.debugMode = GCS::NoDebug;

  add_entity_rules();
  for (std::size_t row = 0; row < plans_.size(); ++row) {
    add_row(row, plans_[row]);
  }

  // The unknowns, ascending: the parameters that move and are not fixed, then the
  // auxiliary unknowns.
  GCS::VEC_pD unknowns;
  for (std::size_t i = 0; i < t_.n(); ++i) {
    if (role_[i] == 1 && t_.fixed[i] == 0) {
      unknowns.push_back(s(i));
    }
  }
  std::sort(aux_unknowns_.begin(), aux_unknowns_.end());
  for (std::size_t i : aux_unknowns_) {
    unknowns.push_back(s(i));
  }

  if (added_ == 0) {
    const int dof = static_cast<int>(unknowns.size());
    return unsolved_outcome(dof == 0 ? "SOLVED" : "UNDER_CONSTRAINED", dof);
  }
  if (unknowns.empty()) {
    // Nothing moves: each row holds or conflicts as it stands (PlaneGCS diagnoses nothing
    // without an unknown).
    Outcome out = unsolved_outcome("SOLVED", 0);
    for (std::size_t row = 0; row < plans_.size(); ++row) {
      const double err = sys_.calculateConstraintErrorByTag(static_cast<int>(row) + 1);
      if (std::isfinite(err) && std::abs(err) > tolerance_) {
        out.conflicting.push_back(static_cast<std::int64_t>(row));
      }
    }
    const double rules = sys_.calculateConstraintErrorByTag(0);
    if (!out.conflicting.empty()) {
      out.status = "OVER_CONSTRAINED";
    } else if (std::isfinite(rules) && std::abs(rules) > tolerance_) {
      out.status = "FAILED";
    }
    return out;
  }

  // Shaper's PlaneGCSSolver_Solver::solve: diagnose and solve with DogLeg; when it fails,
  // diagnose and solve with Levenberg-Marquardt; when that fails, with BFGS.
  sys_.declareUnknowns(unknowns);
  int dof = sys_.diagnose(GCS::DogLeg);
  int result = sys_.solve(unknowns, true, GCS::DogLeg);
  if (result == GCS::Failed) {
    dof = sys_.diagnose(GCS::LevenbergMarquardt);
    result = sys_.solve(unknowns, true, GCS::LevenbergMarquardt);
    if (result == GCS::Failed) {
      dof = sys_.diagnose(GCS::BFGS);
      result = sys_.solve(unknowns, true, GCS::BFGS);
    }
  }
  GCS::VEC_I tags;
  Outcome out;
  sys_.getConflicting(tags);
  out.conflicting = rows_of(tags);
  sys_.getRedundant(tags);
  out.redundant = rows_of(tags);
  sys_.getPartiallyRedundant(tags);
  out.partially_redundant = rows_of(tags);
  out.dof = dof;
  if (!out.conflicting.empty()) {
    out.status = "OVER_CONSTRAINED";
    out.params = t_.params;
  } else if (result == GCS::Success) {
    sys_.applySolution();
    out.status = dof == 0 ? "SOLVED" : "UNDER_CONSTRAINED";
    out.params.assign(store_.begin(), store_.begin() + static_cast<std::ptrdiff_t>(t_.n()));
  } else {
    out.status = "FAILED";
    out.params = t_.params;
  }
  return out;
}

py::array_t<std::int64_t> int_array(const std::vector<std::int64_t>& v) {
  py::array_t<std::int64_t> a(static_cast<py::ssize_t>(v.size()));
  std::copy(v.begin(), v.end(), a.mutable_data());
  return a;
}

py::dict sketch_solve(const py::object& params_obj, const py::object& entities_obj,
                      const py::object& constraints_obj, const py::object& values_obj,
                      const py::object& fixed_obj, const py::object& offsets_obj,
                      const py::object& indices_obj, int max_iterations, double tolerance) {
  if (max_iterations < 1) {
    refuse("max_iterations is " + std::to_string(max_iterations) + "; it is 1 or more.");
  }
  if (!std::isfinite(tolerance) || !(tolerance > 0.0)) {
    refuse("tolerance is " + number_text(tolerance) + "; it is a finite number above 0.");
  }
  Tables t;
  t.params = read_f64(params_obj, "params", -1);
  for (std::size_t i = 0; i < t.params.size(); ++i) {
    if (!std::isfinite(t.params[i])) {
      refuse("params[" + std::to_string(i) + "] is " + number_text(t.params[i]) +
             "; every parameter of the initial guess is finite.");
    }
  }
  t.entities = read_i64(entities_obj, "entities", kEntityColumns);
  t.constraints = read_i64(constraints_obj, "constraints", kConstraintColumns);
  t.values = read_f64(values_obj, "values", static_cast<py::ssize_t>(t.c()));
  if (fixed_obj.is_none()) {
    t.fixed.assign(t.n(), 0);
  } else {
    const py::array a = typed_array(fixed_obj, "fixed", 'b', 1, "bool");
    if (a.ndim() != 1 || a.shape(0) != static_cast<py::ssize_t>(t.n())) {
      refuse("fixed must have shape (" + std::to_string(t.n()) + ",), one flag per "
             "parameter (got " + shape_text(a) + ").");
    }
    const std::vector<bool> flags = copy_of<bool>(a);
    t.fixed.assign(flags.begin(), flags.end());
  }
  if (offsets_obj.is_none() != indices_obj.is_none()) {
    refuse("spline_offsets and spline_indices come together; one of them is not given.");
  }
  if (!offsets_obj.is_none()) {
    t.offsets = read_i64(offsets_obj, "spline_offsets", 0);
    t.indices = read_i64(indices_obj, "spline_indices", 0);
    if (t.offsets.empty()) {
      refuse("spline_offsets is empty; it holds the number of slots + 1 offsets, [0] for "
             "none.");
    }
  }
  Outcome out;
  {
    py::gil_scoped_release release;
    try {
      Sketch sketch(t, max_iterations, tolerance);
      out = sketch.solve();
    } catch (const PysmeshError&) {
      throw;
    } catch (const Standard_Failure& f) {
      throw PysmeshError(std::string("sketch.solve: OCCT failed while the sketch was read: ") +
                         f.ExceptionType() + ": " + f.what() + ".");
    } catch (const std::exception& e) {
      throw PysmeshError(std::string("sketch.solve: PlaneGCS failed: ") + e.what() + ".");
    }
  }
  py::array_t<double> solved(static_cast<py::ssize_t>(out.params.size()));
  std::copy(out.params.begin(), out.params.end(), solved.mutable_data());
  py::dict d;
  d["params"] = solved;
  d["status"] = out.status;
  d["dof"] = out.dof;
  d["conflicting"] = int_array(out.conflicting);
  d["redundant"] = int_array(out.redundant);
  d["partially_redundant"] = int_array(out.partially_redundant);
  return d;
}

}  // namespace

void bind_sketch(py::module_& m) {
  m.def("sketch_solve", &sketch_solve, py::arg("params"), py::arg("entities"),
        py::arg("constraints"), py::arg("values"), py::arg("fixed"),
        py::arg("spline_offsets"), py::arg("spline_indices"), py::arg("max_iterations"),
        py::arg("tolerance"),
        "Solve a 2-D sketch given as flat arrays with PlaneGCS (see pysmesh.sketch.solve).");
}

}  // namespace pysmesh
