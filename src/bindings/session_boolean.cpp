// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-06

// pySMESH binding — Session: the boolean family, fillet and chamfer.
//
// Every BRepAlgoAPI_BuilderAlgo descendant shares its arguments, options, history and error
// channels, so one driver (run_bop) serves all six booleans and the three splits that give
// each piece its own id; only the operand assignment and whether the operation replaces or
// extends the model differ.
//
// Fillet and chamfer sit here rather than with construction because they are the same kind
// of operation: a local rework of an existing body, driven by named edges, whose history is
// the only thing keeping the surrounding entity ids alive.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include <BRepFilletAPI_MakeFillet2d.hxx>
#include <ChFi2d_ConstructionError.hxx>

#include <ElSLib.hxx>

#include "shape_checks.hpp"

namespace pysmesh {
namespace session {
namespace {

// The tools of a boolean, for the two OCCT types that have them. A general fuse has none.
std::vector<TopoDS_Shape> tools_of(BRepAlgoAPI_BuilderAlgo& op) {
  std::vector<TopoDS_Shape> out;
  const NCollection_List<TopoDS_Shape>* tools = nullptr;
  if (const auto* bop = dynamic_cast<const BRepAlgoAPI_BooleanOperation*>(&op)) {
    tools = &bop->Tools();
  } else if (const auto* splitter = dynamic_cast<const BRepAlgoAPI_Splitter*>(&op)) {
    tools = &splitter->Tools();
  }
  if (tools != nullptr) {
    for (const TopoDS_Shape& s : *tools) {
      out.push_back(s);
    }
  }
  return out;
}

// Why a result with no solid is wrong, or an empty string when it may be right.
//
// It runs only when the result holds no solid although the targets hold one, so a normal
// result pays nothing. A fuse, a split, a fragment and an imprint never remove a solid, so
// for them no solid is always wrong. A common and a cut can be empty, so they are refused
// only on a witness: a point proven inside both operands of a common, or inside a target
// and outside every tool of a cut, each by more than the tolerance (shape_checks.hpp). The
// tolerance is the operands' largest one plus the fuzzy value, because a fuzzy boolean may
// treat two features closer than that as one.
std::string empty_result_refusal(const std::string& op, BRepAlgoAPI_BuilderAlgo& builder,
                                 const TopoDS_Shape& result, double fuzzy) {
  if (op == "section" || !shape_checks::solids_of({result}).empty()) {
    return std::string();
  }
  std::vector<TopoDS_Shape> targets;
  for (const TopoDS_Shape& s : builder.Arguments()) {
    targets.push_back(s);
  }
  const std::size_t solids = shape_checks::solids_of(targets).size();
  if (solids == 0) {
    return std::string();
  }
  const std::string head = "Session." + op + ": OCCT returned no solid ";
  if (op == "fuse" || op == "split" || op == "fragment" || op == "imprint") {
    return head + "for " + std::to_string(solids) +
           " solid target(s); this operation never removes a solid.";
  }
  const std::vector<TopoDS_Shape> tools = tools_of(builder);
  std::vector<TopoDS_Shape> operands = targets;
  operands.insert(operands.end(), tools.begin(), tools.end());
  const double tol = shape_checks::max_tolerance(operands) + fuzzy;
  const double depth = 4.0 * tol;
  char tol_text[32];
  std::snprintf(tol_text, sizeof(tol_text), "%.3g", tol);
  if (op == "common") {
    if (const std::optional<gp_Pnt> p =
            shape_checks::point_inside_both(targets, tools, depth, tol)) {
      return head + "although the operands overlap: the point " + shape_checks::point_text(*p) +
             " lies inside a target and inside a tool, deeper than " + tol_text +
             " from both boundaries.";
    }
  } else if (op == "cut") {
    if (const std::optional<gp_Pnt> p =
            shape_checks::point_inside_first_outside_second(targets, tools, depth, tol)) {
      return head + "although the targets reach outside the tools: the point " +
             shape_checks::point_text(*p) + " lies inside a target and outside every tool, " +
             "farther than " + tol_text + " from both boundaries.";
    }
  }
  return std::string();
}

// Why a result solid that is not watertight is refused, with the details that name its
// free edges; an empty message when every solid of the result is closed.
//
// Each solid is checked on its own, so an edge shared by two solids of one result (two
// fragments meeting along it) still needs two faces in each. If the operands already had
// free edges, the details say so: then the leak may have come in with the input.
std::pair<std::string, std::string> leaky_solid_refusal(const std::string& op,
                                                        BRepAlgoAPI_BuilderAlgo& builder,
                                                        const TopoDS_Shape& result) {
  std::vector<TopoDS_Shape> leaks;
  for (const TopoDS_Shape& solid : shape_checks::solids_of({result})) {
    for (const TopoDS_Shape& e : shape_checks::free_boundary_edges(solid)) {
      leaks.push_back(e);
    }
  }
  if (leaks.empty()) {
    return {};
  }
  std::string message = "Session." + op + ": the result has " +
                        std::to_string(leaks.size()) +
                        " free boundary edge(s), each bordered by one face only, so a solid "
                        "of it is not watertight. The session is unchanged.";
  std::string details;
  for (std::size_t i = 0; i < leaks.size(); ++i) {
    details += "Free edge " + std::to_string(i + 1) + ": " + shape_checks::edge_text(leaks[i]) +
               ". ";
  }
  std::vector<TopoDS_Shape> operands;
  for (const TopoDS_Shape& s : builder.Arguments()) {
    operands.push_back(s);
  }
  for (const TopoDS_Shape& s : tools_of(builder)) {
    operands.push_back(s);
  }
  std::size_t inherited = 0;
  for (const TopoDS_Shape& solid : shape_checks::solids_of(operands)) {
    inherited += shape_checks::free_boundary_edges(solid).size();
  }
  details += inherited == 0
                 ? std::string("Every operand solid was watertight, so OCCT made the leak.")
                 : "The operand solids already had " + std::to_string(inherited) +
                       " free boundary edge(s); heal or sew them first.";
  return {message, details};
}

// A number as the refusal texts of the splitting operations print it: six significant
// digits, no trailing zeros.
std::string num_text(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return buf;
}

// "solid 4", or "solids 4, 7 and 9", for a refusal text.
std::string ids_text(const char* kind, const char* kinds, const std::vector<EntityId>& ids) {
  std::string out = ids.size() == 1 ? kind : kinds;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    out += (i == 0 ? " " : (i + 1 == ids.size() ? " and " : ", ")) + std::to_string(ids[i]);
  }
  return out;
}

// The lowest and the highest signed distance (p - origin) . normal over the material of a
// shape, read as the z range of its exact box in the plane's own frame, where the plane is
// z = 0 and the normal is +z. For analytic geometry the box is the geometry's own; a
// B-spline adds at most 1e-7 (shape_checks::exact_box).
std::pair<double, double> signed_range(const TopoDS_Shape& s, const gp_Ax3& plane) {
  gp_Trsf to_plane;
  to_plane.SetTransformation(plane);
  const Bnd_Box box = shape_checks::exact_box(s.Moved(TopLoc_Location(to_plane)));
  double x0, y0, z0, x1, y1, z1;
  box.Get(x0, y0, z0, x1, y1, z1);
  return {z0, z1};
}

// The solids of the splitter's result that a target solid became: its Modified images that
// are solids of the result, or the solid itself when the splitter left it whole.
std::vector<TopoDS_Shape> pieces_of(const TopoDS_Shape& solid, const ShapeSet& result_solids,
                                    const Handle(BRepTools_History)& hist) {
  std::vector<TopoDS_Shape> out;
  for (const TopoDS_Shape& m : hist->Modified(solid)) {
    if (m.ShapeType() == TopAbs_SOLID && result_solids.Contains(m)) {
      append_unique(out, m);
    }
  }
  if (out.empty() && result_solids.Contains(solid)) {
    out.push_back(solid);
  }
  return out;
}

// The non-empty lines of OCCT's warning dump.
std::vector<std::string> warning_lines(const std::string& dump) {
  std::vector<std::string> out;
  std::istringstream in(dump);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      out.push_back(line);
    }
  }
  return out;
}

}  // namespace

// ---- modelling operations --------------------------------------------------------- //

py::dict Session::fuse(const std::vector<EntityId>& targets, const std::vector<EntityId>& tools,
                double fuzzy, bool parallel, const py::object& progress,
                const py::object& cancel) {
  OpGuard guard(in_op_);
  require_operands("fuse", targets, tools, fuzzy);
  BRepAlgoAPI_Fuse op;
  op.SetArguments(solids_of("fuse", "targets", targets));
  op.SetTools(solids_of("fuse", "tools", tools));
  return run_bop("fuse", op, bodies_of(targets, tools), /*additive=*/false, fuzzy, parallel,
                 hooks_of("fuse", progress, cancel));
}

py::dict Session::cut(const std::vector<EntityId>& targets, const std::vector<EntityId>& tools,
               double fuzzy, bool parallel, const py::object& progress,
               const py::object& cancel) {
  OpGuard guard(in_op_);
  require_operands("cut", targets, tools, fuzzy);
  BRepAlgoAPI_Cut op;
  op.SetArguments(solids_of("cut", "targets", targets));
  op.SetTools(solids_of("cut", "tools", tools));
  return run_bop("cut", op, bodies_of(targets, tools), /*additive=*/false, fuzzy, parallel,
                 hooks_of("cut", progress, cancel));
}

py::dict Session::common(const std::vector<EntityId>& targets,
                         const std::vector<EntityId>& tools, double fuzzy, bool parallel,
                         const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  require_operands("common", targets, tools, fuzzy);
  BRepAlgoAPI_Common op;
  op.SetArguments(solids_of("common", "targets", targets));
  op.SetTools(solids_of("common", "tools", tools));
  return run_bop("common", op, bodies_of(targets, tools), /*additive=*/false, fuzzy, parallel,
                 hooks_of("common", progress, cancel));
}

// The section curves of targets against tools. Additive: the result of a section is the
// intersection geometry alone, so both operand groups stay in the model and only the
// section's vertices and edges are added.
py::dict Session::section(const std::vector<EntityId>& targets,
                          const std::vector<EntityId>& tools, double fuzzy, bool parallel,
                          const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  require_operands("section", targets, tools, fuzzy);
  BRepAlgoAPI_Section op;
  op.SetArguments(solids_of("section", "targets", targets));
  op.SetTools(solids_of("section", "tools", tools));
  return run_bop("section", op, {}, /*additive=*/true, fuzzy, parallel,
                 hooks_of("section", progress, cancel));
}

// Split the targets by the tools. The tools are not consumed: OCCT's Splitter excludes
// their split parts from the result, and a tool the caller still holds ids for must not
// disappear from the model as a side effect.
py::dict Session::split(const std::vector<EntityId>& targets,
                        const std::vector<EntityId>& tools, double fuzzy, bool parallel,
                        const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  require_operands("split", targets, tools, fuzzy);
  BRepAlgoAPI_Splitter op;
  op.SetArguments(solids_of("split", "targets", targets));
  op.SetTools(solids_of("split", "tools", tools));
  return run_bop("split", op, bodies_of(targets, {}), /*additive=*/false, fuzzy, parallel,
                 hooks_of("split", progress, cancel));
}

// The general fuse: every operand is split by every other and the result keeps all the
// pieces. This is the operation a conformal multi-body CFD domain is built with.
py::dict Session::fragment(const std::vector<EntityId>& entity_ids, double fuzzy,
                           bool parallel, const py::object& progress,
                           const py::object& cancel) {
  OpGuard guard(in_op_);
  if (entity_ids.size() < 2) {
    throw PysmeshError("Session.fragment: at least two solids are required (got " +
                       std::to_string(entity_ids.size()) + ").");
  }
  require_fuzzy("fragment", fuzzy);
  BRepAlgoAPI_BuilderAlgo op;
  op.SetArguments(solids_of("fragment", "entity_ids", entity_ids));
  return run_bop("fragment", op, bodies_of(entity_ids, {}), /*additive=*/false, fuzzy,
                 parallel, hooks_of("fragment", progress, cancel));
}

// ---- splitting with the piece rule ------------------------------------------------- //
//
// All three run BRepAlgoAPI_Splitter through run_bop with parallel = false: a split must not
// depend on the thread schedule (a fragment's parallel and serial results agree in measure,
// but nothing promises the same sub-shape order). run_bop's PieceSplit step checks each target
// against the answer and commits under PieceRule::new_piece_ids(TopAbs_SOLID).

namespace {

// The plane of a plane operation, refusing a zero normal with its value.
gp_Ax3 plane_of(const char* op, double ox, double oy, double oz, double nx, double ny,
                double nz) {
  finite_arg(op, "origin", ox, oy, oz);
  finite_arg(op, "normal", nx, ny, nz);
  if (!(gp_Vec(nx, ny, nz).Magnitude() > 0.0)) {
    throw PysmeshError(std::string("Session.") + op + ": the plane normal (" + num_text(nx) +
                       ", " + num_text(ny) + ", " + num_text(nz) +
                       ") has zero length, so it defines no plane.");
  }
  return gp_Ax3(gp_Pnt(ox, oy, oz), gp_Dir(nx, ny, nz));
}

}  // namespace

py::dict Session::split_by_faces(const std::vector<EntityId>& targets,
                                 const std::vector<EntityId>& tools, double fuzzy,
                                 const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  if (targets.empty()) {
    throw PysmeshError("Session.split_by_faces: no target solid is named; nothing to split.");
  }
  if (tools.empty()) {
    throw PysmeshError(
        "Session.split_by_faces: no tool face is named; nothing to split with.");
  }
  require_fuzzy("split_by_faces", fuzzy);
  const NCollection_List<TopoDS_Shape> arguments =
      solids_of("split_by_faces", "targets", targets);
  const std::vector<TopoDS_Shape> consumed = bodies_of(targets, {});

  // The tool faces, every shape of each id. A face of a target's own body is refused: it
  // lies on the boundary it would cut, so it divides nothing and makes the boolean's input
  // self-interfering.
  const ShapeKeyed<TopoDS_Shape> owners = body_of_subshape();
  NCollection_List<TopoDS_Shape> faces;
  for (EntityId id : tools) {
    const EntityRecord& rec = require_alive("split_by_faces", id);
    if (rec.kind != TopAbs_FACE) {
      throw PysmeshError("Session.split_by_faces: tool " + std::to_string(id) + " is a " +
                             kind_name(rec.kind) + ", not a FACE.",
                         "", ids_as_int({id}));
    }
    for (const TopoDS_Shape& f : rec.shapes) {
      const auto owner = owners.find(f);
      for (const TopoDS_Shape& body : consumed) {
        if (owner != owners.end() && owner->second.IsSame(body)) {
          throw PysmeshError("Session.split_by_faces: face " + std::to_string(id) +
                                 " is a face of a target solid's own body; a solid is not "
                                 "split by its own boundary.",
                             "", ids_as_int({id}));
        }
      }
      faces.Append(f);
    }
  }

  BRepAlgoAPI_Splitter op;
  op.SetArguments(arguments);
  op.SetTools(faces);
  PieceSplit pieces;
  pieces.refine = [&](const TopoDS_Shape& result, const Handle(BRepTools_History)& hist) {
    ShapeSet result_solids;
    TopExp::MapShapes(result, TopAbs_SOLID, result_solids);
    bool any_cut = false;
    for (EntityId id : targets) {
      std::size_t count = 0;
      for (const TopoDS_Shape& s : state_.registry->alive.at(id).shapes) {
        count += pieces_of(s, result_solids, hist).size();
      }
      any_cut = any_cut || count > 1;
    }
    if (!any_cut) {
      throw PysmeshError("Session.split_by_faces: " + ids_text("face", "faces", tools) +
                             " divide none of the " + ids_text("solid", "solids", targets) +
                             "; nothing to split.",
                         "Each tool face either misses every target solid or meets one "
                         "without cutting it through.",
                         ids_as_int(tools));
    }
    return result;
  };
  return run_bop("split_by_faces", op, consumed, /*additive=*/false, fuzzy,
                 /*parallel=*/false, hooks_of("split_by_faces", progress, cancel), &pieces);
}

TopoDS_Face Session::plane_tool(const char* op, const std::vector<EntityId>& targets,
                                const gp_Ax3& plane, double fuzzy, int drop_side) const {
  Bnd_Box all;
  std::string refusals;
  std::vector<EntityId> offending;
  for (EntityId id : targets) {
    const EntityRecord& rec = state_.registry->alive.at(id);
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (const TopoDS_Shape& s : rec.shapes) {
      const std::pair<double, double> range = signed_range(s, plane);
      lo = std::min(lo, range.first);
      hi = std::max(hi, range.second);
      all.Add(shape_checks::exact_box(s));
    }
    // A solid that only touches the plane, within its own tolerance plus fuzzy, lies on one
    // side of it: the splitter would leave it whole.
    const double tol =
        std::max(shape_checks::max_tolerance(rec.shapes), Precision::Confusion()) + fuzzy;
    const std::string range = " (signed distance " + num_text(lo) + " to " + num_text(hi) + ")";
    std::string reason;
    if (drop_side == 0) {
      if (lo >= -tol || hi <= tol) {
        reason = "lies entirely on one side of the plane" + range + "; nothing to split";
      }
    } else {
      const bool all_dropped = drop_side < 0 ? hi <= tol : lo >= -tol;
      const bool all_kept = drop_side < 0 ? lo >= -tol : hi <= tol;
      if (all_dropped) {
        reason = "lies entirely on the side of the plane that is removed" + range +
                 "; nothing would be kept";
      } else if (all_kept) {
        reason = "lies entirely on the side of the plane that is kept" + range +
                 "; nothing to remove";
      }
    }
    if (!reason.empty()) {
      refusals += (refusals.empty() ? "" : "; ") + std::string("solid ") +
                  std::to_string(id) + " " + reason;
      offending.push_back(id);
    }
  }
  if (!offending.empty()) {
    throw PysmeshError(std::string("Session.") + op + ": " + refusals + ".",
                       "Signed distances are measured from the plane along its normal.",
                       ids_as_int(offending));
  }
  // A square on the plane, centred on the projection of the targets' box centre, with a
  // half-side equal to the box diagonal. Every point of the box projects to within half a
  // diagonal of that centre, so the face covers the targets' cross-section twice over.
  double x0, y0, z0, x1, y1, z1;
  all.Get(x0, y0, z0, x1, y1, z1);
  const gp_Pnt centre(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1));
  const double half = gp_Pnt(x0, y0, z0).Distance(gp_Pnt(x1, y1, z1));
  const gp_Pln pln(plane);
  double u = 0.0;
  double v = 0.0;
  ElSLib::Parameters(pln, centre, u, v);
  BRepBuilderAPI_MakeFace mk(pln, u - half, u + half, v - half, v + half);
  if (!mk.IsDone()) {
    throw PysmeshError(std::string("Session.") + op +
                       ": OCCT could not build the planar tool face (half-side " +
                       num_text(half) + ").");
  }
  return mk.Face();
}

py::dict Session::split_by_plane(const std::vector<EntityId>& targets, double ox, double oy,
                                 double oz, double nx, double ny, double nz, double fuzzy,
                                 const py::object& progress, const py::object& cancel) {
  const gp_Ax3 plane = plane_of("split_by_plane", ox, oy, oz, nx, ny, nz);
  OpGuard guard(in_op_);
  if (targets.empty()) {
    throw PysmeshError("Session.split_by_plane: no target solid is named; nothing to split.");
  }
  require_fuzzy("split_by_plane", fuzzy);
  const NCollection_List<TopoDS_Shape> arguments =
      solids_of("split_by_plane", "targets", targets);
  NCollection_List<TopoDS_Shape> tool;
  tool.Append(plane_tool("split_by_plane", targets, plane, fuzzy, /*drop_side=*/0));

  BRepAlgoAPI_Splitter op;
  op.SetArguments(arguments);
  op.SetTools(tool);
  PieceSplit pieces;
  pieces.refine = [&](const TopoDS_Shape& result, const Handle(BRepTools_History)& hist) {
    ShapeSet result_solids;
    TopExp::MapShapes(result, TopAbs_SOLID, result_solids);
    std::vector<EntityId> whole;
    for (EntityId id : targets) {
      std::size_t count = 0;
      for (const TopoDS_Shape& s : state_.registry->alive.at(id).shapes) {
        count += pieces_of(s, result_solids, hist).size();
      }
      if (count < 2) {
        whole.push_back(id);
      }
    }
    if (!whole.empty()) {
      throw PysmeshError("Session.split_by_plane: OCCT left " +
                             ids_text("solid", "solids", whole) +
                             " in one piece, although the plane crosses it; nothing is "
                             "committed.",
                         "", ids_as_int(whole));
    }
    return result;
  };
  return run_bop("split_by_plane", op, bodies_of(targets, {}), /*additive=*/false, fuzzy,
                 /*parallel=*/false, hooks_of("split_by_plane", progress, cancel), &pieces);
}

py::dict Session::keep_half_space(const std::vector<EntityId>& targets, double ox, double oy,
                                  double oz, double nx, double ny, double nz,
                                  bool keep_positive, double fuzzy,
                                  const py::object& progress, const py::object& cancel) {
  const gp_Ax3 plane = plane_of("keep_half_space", ox, oy, oz, nx, ny, nz);
  OpGuard guard(in_op_);
  if (targets.empty()) {
    throw PysmeshError(
        "Session.keep_half_space: no target solid is named; nothing to trim.");
  }
  require_fuzzy("keep_half_space", fuzzy);
  const NCollection_List<TopoDS_Shape> arguments =
      solids_of("keep_half_space", "targets", targets);
  const int drop_side = keep_positive ? -1 : 1;
  NCollection_List<TopoDS_Shape> tool;
  tool.Append(plane_tool("keep_half_space", targets, plane, fuzzy, drop_side));

  BRepAlgoAPI_Splitter op;
  op.SetArguments(arguments);
  op.SetTools(tool);
  PieceSplit pieces;
  pieces.refine = [&](const TopoDS_Shape& result, const Handle(BRepTools_History)& hist) {
    ShapeSet result_solids;
    TopExp::MapShapes(result, TopAbs_SOLID, result_solids);
    // Each piece lies on one side of the plane, so the middle of its signed-distance range
    // has the sign of that side.
    ShapeSet dropped;
    try {
      for (int i = 1; i <= result_solids.Extent(); ++i) {
        const std::pair<double, double> range = signed_range(result_solids(i), plane);
        const double middle = 0.5 * (range.first + range.second);
        if ((middle > 0.0 ? 1 : -1) == drop_side) {
          dropped.Add(result_solids(i));
        }
      }
    } catch (const std::exception& e) {
      throw PysmeshError(std::string("Session.keep_half_space: measuring the pieces failed: ") +
                         e.what());
    }
    std::vector<EntityId> unsplit;
    for (EntityId id : targets) {
      std::size_t kept = 0;
      std::size_t removed = 0;
      for (const TopoDS_Shape& s : state_.registry->alive.at(id).shapes) {
        for (const TopoDS_Shape& p : pieces_of(s, result_solids, hist)) {
          ++(dropped.Contains(p) ? removed : kept);
        }
      }
      if (kept == 0 || removed == 0) {
        unsplit.push_back(id);
      }
    }
    if (!unsplit.empty()) {
      throw PysmeshError("Session.keep_half_space: OCCT did not divide " +
                             ids_text("solid", "solids", unsplit) +
                             " at the plane, although the plane crosses it; nothing is "
                             "committed.",
                         "", ids_as_int(unsplit));
    }
    // The pieces kept, in the order of the splitter's result.
    std::vector<TopoDS_Shape> leaves;
    explode_into(result, leaves);
    TopoDS_Compound kept;
    BRep_Builder builder;
    builder.MakeCompound(kept);
    for (const TopoDS_Shape& leaf : leaves) {
      if (!dropped.Contains(leaf)) {
        builder.Add(kept, leaf);
      }
    }
    return TopoDS_Shape(kept);
  };
  return run_bop("keep_half_space", op, bodies_of(targets, {}), /*additive=*/false, fuzzy,
                 /*parallel=*/false, hooks_of("keep_half_space", progress, cancel), &pieces);
}

// ---- fillet and chamfer ----------------------------------------------------------- //

// radius_end, when given, makes the radius evolve linearly along each named edge from
// radius to radius_end (OCCT's two-radius Add).
py::dict Session::fillet(const std::vector<EntityId>& edge_ids, double radius,
                  const std::optional<double>& radius_end, const py::object& progress,
                  const py::object& cancel) {
  finite_arg("fillet", "radius", radius);
  if (radius_end.has_value()) {
    finite_arg("fillet", "radius_end", *radius_end);
  }
  OpGuard guard(in_op_);
  require_positive("radius", radius);
  if (radius_end.has_value()) {
    require_positive("radius_end", *radius_end);
  }
  std::vector<EntityId> kept;
  const std::vector<TopoDS_Shape> edges = edges_of("fillet", edge_ids, &kept);

  // OCCT's fillet takes edges and derives the owning solid itself, so the caller never has
  // to co-select the solid or a reference face per edge. Every named edge must belong to
  // the same body, because one fillet operation builds one shape.
  const TopoDS_Shape owner = sole_owner_body(body_of_subshape(), edges);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({owner});

  ProgressDriver driver("fillet", hooks_of("fillet", progress, cancel));
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::vector<TopoDS_Shape> faulty;
  {
    py::gil_scoped_release release;
    // Every OCCT call runs inside the try, the history query included (report A2).
    const char* stage = "BRepFilletAPI_MakeFillet::Add failed";
    try {
      BRepFilletAPI_MakeFillet mk(owner);
      for (const TopoDS_Shape& e : edges) {
        if (radius_end.has_value()) {
          mk.Add(radius, *radius_end, TopoDS::Edge(e));
        } else {
          mk.Add(radius, TopoDS::Edge(e));
        }
      }
      stage = "BRepFilletAPI_MakeFillet::Build failed";
      mk.Build(driver.range());
      if (mk.IsDone()) {
        stage = "reading the history of BRepFilletAPI_MakeFillet failed";
        result = mk.Shape();
        hist = history_of(owner, mk);
      } else {
        faulty = faulty_edges(mk);
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.fillet: ") + stage + ": " + e.what());
    }
  }
  driver.finish();
  // Before the failure path, not after it: an operation the caller stopped has no faulty
  // edges to blame, and reporting the radius as unbuildable would be a false diagnostic.
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("fillet");
  }
  if (result.IsNull()) {
    throw PysmeshError(
        "Session.fillet: OCCT could not build a fillet of radius " +
            std::to_string(radius) + " on the named edges.",
        "BRepFilletAPI_MakeFillet::IsDone() is false. The radius is most likely larger "
        "than the local geometry admits. No partial result is returned.",
        blamed_ids(kept, edges, faulty));
  }
  return commit(concat(survivors, result), hist, "fillet", result);
}

// distance_end + face_id give OCCT's two-distance chamfer, where the first distance is
// measured on the named reference face. That is the only form in OCCT 8.0 that takes a
// face at all — there is no (distance, edge, face) overload.
py::dict Session::chamfer(const std::vector<EntityId>& edge_ids, double distance,
                   const std::optional<double>& distance_end,
                   const std::optional<EntityId>& face_id, const py::object& progress,
                   const py::object& cancel) {
  finite_arg("chamfer", "distance", distance);
  if (distance_end.has_value()) {
    finite_arg("chamfer", "distance_end", *distance_end);
  }
  OpGuard guard(in_op_);
  require_positive("distance", distance);
  if (distance_end.has_value()) {
    require_positive("distance_end", *distance_end);
  }
  if (face_id.has_value() != distance_end.has_value()) {
    throw PysmeshError(
        "Session.chamfer: face_id and distance_end must be given together. OCCT 8.0's "
        "only face-aware chamfer is the two-distance form.");
  }
  const std::vector<TopoDS_Shape> edges = edges_of("chamfer", edge_ids);

  TopoDS_Face reference;
  if (face_id.has_value()) {
    const EntityRecord& rec = require_alive("chamfer", *face_id);
    if (rec.kind != TopAbs_FACE) {
      throw PysmeshError("Session.chamfer: entity " + std::to_string(*face_id) +
                         " is a " + kind_name(rec.kind) + ", not a FACE.");
    }
    if (rec.shapes.size() != 1) {
      throw PysmeshError("Session.chamfer: face " + std::to_string(*face_id) +
                         " was split and denotes several faces; name one of them.");
    }
    reference = TopoDS::Face(rec.shapes.front());
  }

  std::vector<TopoDS_Shape> selection = edges;
  if (!reference.IsNull()) {
    selection.push_back(reference);
  }
  const TopoDS_Shape owner = sole_owner_body(body_of_subshape(), selection);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({owner});

  ProgressDriver driver("chamfer", hooks_of("chamfer", progress, cancel));
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::vector<TopoDS_Shape> faulty;
  {
    py::gil_scoped_release release;
    const char* stage = "BRepFilletAPI_MakeChamfer::Add failed";
    try {
      BRepFilletAPI_MakeChamfer mk(owner);
      for (const TopoDS_Shape& e : edges) {
        if (reference.IsNull()) {
          mk.Add(distance, TopoDS::Edge(e));
        } else {
          mk.Add(distance, *distance_end, TopoDS::Edge(e), reference);
        }
      }
      stage = "BRepFilletAPI_MakeChamfer::Build failed";
      mk.Build(driver.range());
      if (mk.IsDone()) {
        stage = "reading the history of BRepFilletAPI_MakeChamfer failed";
        result = mk.Shape();
        hist = history_of(owner, mk);
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.chamfer: ") + stage + ": " + e.what());
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("chamfer");
  }
  if (result.IsNull()) {
    throw PysmeshError(
        "Session.chamfer: OCCT could not build a chamfer of distance " +
            std::to_string(distance) + " on the named edges.",
        "BRepFilletAPI_MakeChamfer::IsDone() is false. The distance is most likely "
        "larger than the adjacent faces admit. No partial result is returned.",
        ids_as_int(edge_ids));
  }
  return commit(concat(survivors, result), hist, "chamfer", result);
}

namespace {

// Why ChFi2d_Builder stopped at a vertex, from the statuses ChFi2d_ConstructionError names.
std::string fillet_2d_reason(ChFi2d_ConstructionError status) {
  switch (status) {
    case ChFi2d_NotPlanar:
      return "the face's surface is not a Geom_Plane";
    case ChFi2d_ComputationError:
      return "no arc of that radius fits between the two edges that meet there";
    case ChFi2d_ConnexionError:
      return "the vertex does not join two edges of the face";
    case ChFi2d_TangencyError:
      return "the two edges that meet there are tangent";
    case ChFi2d_NotAuthorized:
      return "an edge that meets there is neither a line nor a circle arc";
    default:
      return "OCCT's ChFi2d_Builder ended with status " + std::to_string(static_cast<int>(status));
  }
}

}  // namespace

// BRepFilletAPI_MakeFillet2d rebuilds the face from one new wire on the same plane
// (ChFi2d_Builder::BuildNewWire): untouched edges and vertices keep their TShapes, the two
// edges at a rounded corner are rebuilt shorter, and the arc is new. Its Modified answers
// edges only, so the history is written here: the face and each trimmed edge are Modified,
// the arc and its two new end vertices are Generated from the corner vertex, and the corner
// vertex, and an edge the arc used up entirely, are Removed.
py::dict Session::fillet_2d(EntityId face_id, const std::vector<EntityId>& vertex_ids,
                            double radius) {
  static constexpr const char* kOp = "fillet_2d";
  finite_arg(kOp, "radius", radius);
  OpGuard guard(in_op_);
  const std::string face_text = "face " + std::to_string(face_id);
  if (vertex_ids.empty()) {
    throw PysmeshError("Session.fillet_2d: no vertex of " + face_text +
                           " is named; there is no corner to round.",
                       "", {static_cast<int>(face_id)});
  }
  if (!(radius > 0.0)) {
    throw PysmeshError("Session.fillet_2d: the radius " + number_text(radius) +
                           " of the fillets on " + face_text + " is not positive.",
                       "", {static_cast<int>(face_id)});
  }
  const TopoDS_Face face = sole_face(kOp, face_id);
  const TopoDS_Shape body = sole_owner_body(body_of_subshape(), {face});
  if (!body.IsSame(face)) {
    // A face of a solid bounds its material; a face of a shell shares its edges.
    for (TopExp_Explorer solid(body, TopAbs_SOLID); solid.More(); solid.Next()) {
      for (TopExp_Explorer f(solid.Current(), TopAbs_FACE); f.More(); f.Next()) {
        if (f.Current().IsSame(face)) {
          const auto ids = state_.registry->by_shape.find(solid.Current());
          const std::string solid_id = ids == state_.registry->by_shape.end()
                                           ? std::string("of the model")
                                           : std::to_string(ids->second.front());
          throw PysmeshError("Session.fillet_2d: " + face_text + " bounds solid " +
                                 solid_id +
                                 "; a 2D fillet rounds the corners of a free face only.",
                             "", {static_cast<int>(face_id)});
        }
      }
    }
    throw PysmeshError("Session.fillet_2d: " + face_text + " belongs to a " +
                           std::string(TopAbs::ShapeTypeToString(body.ShapeType())) +
                           " body; a 2D fillet rounds the corners of a free face only.",
                       "", {static_cast<int>(face_id)});
  }
  const BRepAdaptor_Surface surface(face, false);
  if (surface.GetType() != GeomAbs_Plane) {
    throw PysmeshError("Session.fillet_2d: " + face_text + " is a " +
                           std::string(surface_type_name(surface.GetType())) +
                           " face; a 2D fillet rounds the corners of a planar face.",
                       "", {static_cast<int>(face_id)});
  }
  int wires = 0;
  for (TopExp_Explorer w(face, TopAbs_WIRE); w.More(); w.Next()) {
    ++wires;
  }
  if (wires != 1) {
    throw PysmeshError("Session.fillet_2d: " + face_text + " is bounded by " +
                           std::to_string(wires) +
                           " wires; OCCT's 2D fillet rebuilds the face from one wire, so it "
                           "rounds the corners of a face without holes only.",
                       "", {static_cast<int>(face_id)});
  }
  ShapeSet face_vertices;
  TopExp::MapShapes(face, TopAbs_VERTEX, face_vertices);
  std::vector<TopoDS_Vertex> corners;
  for (EntityId id : vertex_ids) {
    const EntityRecord& rec = require_alive(kOp, id);
    if (rec.kind != TopAbs_VERTEX) {
      throw PysmeshError("Session.fillet_2d: entity " + std::to_string(id) + " is a " +
                             kind_name(rec.kind) + ", not a VERTEX.",
                         "", {static_cast<int>(id)});
    }
    if (rec.shapes.size() != 1 || !face_vertices.Contains(rec.shapes.front())) {
      throw PysmeshError("Session.fillet_2d: vertex " + std::to_string(id) +
                             " is not a vertex of " + face_text + ".",
                         "", {static_cast<int>(face_id), static_cast<int>(id)});
    }
    const TopoDS_Vertex v = TopoDS::Vertex(face_vertices.FindKey(
        face_vertices.FindIndex(rec.shapes.front())));
    for (const TopoDS_Vertex& c : corners) {
      if (c.IsSame(v)) {
        throw PysmeshError("Session.fillet_2d: vertex " + std::to_string(id) +
                               " of " + face_text + " is named twice.",
                           "", {static_cast<int>(face_id), static_cast<int>(id)});
      }
    }
    corners.push_back(v);
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({body});

  TopoDS_Face result;
  std::vector<TopoDS_Edge> arcs;
  std::size_t stopped_at = corners.size();
  ChFi2d_ConstructionError status = ChFi2d_IsDone;
  Handle(BRepTools_History) hist = new BRepTools_History;
  const char* stage = "BRepFilletAPI_MakeFillet2d::Init failed";
  try {
    BRepFilletAPI_MakeFillet2d mk(face);
    for (std::size_t i = 0; i < corners.size() && stopped_at == corners.size(); ++i) {
      stage = "BRepFilletAPI_MakeFillet2d::AddFillet failed";
      const TopoDS_Edge arc = mk.AddFillet(corners[i], radius);
      if (mk.Status() != ChFi2d_IsDone) {
        status = mk.Status();
        stopped_at = i;
      }
      arcs.push_back(arc);
    }
    if (stopped_at == corners.size()) {
      stage = "BRepFilletAPI_MakeFillet2d::Build failed";
      mk.Build();
      if (mk.IsDone()) {
        stage = "writing the history of the 2D fillet failed";
        result = TopoDS::Face(mk.Shape());
        // ChFi2d_Builder builds the new face FORWARD on the plane of the face as it reads
        // it FORWARD; the face's own orientation gives back its outward normal.
        result.Orientation(face.Orientation());
        ShapeSet out_edges;
        ShapeSet out_vertices;
        TopExp::MapShapes(result, TopAbs_EDGE, out_edges);
        TopExp::MapShapes(result, TopAbs_VERTEX, out_vertices);
        hist->AddModified(face, result);
        for (TopExp_Explorer ex(face, TopAbs_EDGE); ex.More(); ex.Next()) {
          const TopoDS_Edge& e = TopoDS::Edge(ex.Current());
          const TopoDS_Edge now = mk.HasDescendant(e) ? mk.DescendantEdge(e) : e;
          if (!out_edges.Contains(now)) {
            hist->Remove(e);
          } else if (!now.IsSame(e)) {
            hist->AddModified(e, now);
          }
        }
        for (int i = 1; i <= face_vertices.Extent(); ++i) {
          if (!out_vertices.Contains(face_vertices(i))) {
            hist->Remove(face_vertices(i));
          }
        }
        for (std::size_t i = 0; i < corners.size(); ++i) {
          hist->AddGenerated(corners[i], arcs[i]);
          for (TopExp_Explorer ex(arcs[i], TopAbs_VERTEX); ex.More(); ex.Next()) {
            if (!face_vertices.Contains(ex.Current())) {
              hist->AddGenerated(corners[i], ex.Current());
            }
          }
        }
      }
    }
  } catch (const std::exception& e) {
    throw PysmeshError(std::string("Session.fillet_2d: ") + stage + ": " + e.what(), "",
                       {static_cast<int>(face_id)});
  }
  if (stopped_at != corners.size()) {
    throw PysmeshError("Session.fillet_2d: at vertex " +
                           std::to_string(vertex_ids[stopped_at]) + " of " + face_text +
                           " no fillet of radius " + number_text(radius) + " was built: " +
                           fillet_2d_reason(status) + ".",
                       "", {static_cast<int>(face_id),
                            static_cast<int>(vertex_ids[stopped_at])});
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.fillet_2d: OCCT built no face for the fillets of radius " +
                           number_text(radius) + " on " + face_text + ".",
                       "BRepFilletAPI_MakeFillet2d::IsDone() is false.",
                       {static_cast<int>(face_id)});
  }
  return commit(concat(survivors, result), hist, kOp, result);
}

// The fuzzy value is the one tolerance a caller supplies to a boolean, and it is checked
// rather than forwarded.
//
// `!(fuzzy >= 0.0)` rather than `fuzzy < 0.0` is the point of the rewrite: every comparison
// against NaN is false, so the negated form let NaN straight through to SetFuzzyValue, where
// it becomes a tolerance nothing compares equal to and the boolean's behaviour is undefined.
// Infinity is refused for the same reason — it is not a distance.
void Session::require_fuzzy(const char* op, double fuzzy) {
  if (!(fuzzy >= 0.0) || !std::isfinite(fuzzy)) {
    throw PysmeshError(std::string("Session.") + op +
                       ": fuzzy must be a finite value >= 0 (got " + std::to_string(fuzzy) +
                       ").");
  }
}

void Session::require_operands(const char* op, const std::vector<EntityId>& targets,
                               const std::vector<EntityId>& tools, double fuzzy) {
  if (targets.empty()) {
    throw PysmeshError(std::string("Session.") + op +
                       ": targets must name at least one solid.");
  }
  if (tools.empty()) {
    throw PysmeshError(std::string("Session.") + op +
                       ": tools must name at least one solid.");
  }
  require_fuzzy(op, fuzzy);
}

// The root bodies owning the solids two operand lists name. Resolved through the
// sub-shape -> body map rather than by comparing the solids themselves, so a solid nested
// inside a compound body still removes the right body from the model.
std::vector<TopoDS_Shape> Session::bodies_of(const std::vector<EntityId>& a,
                                      const std::vector<EntityId>& b) const {
  const ShapeKeyed<TopoDS_Shape> owners = body_of_subshape();
  std::vector<TopoDS_Shape> out;
  for (const std::vector<EntityId>* list : {&a, &b}) {
    for (EntityId id : *list) {
      const EntityRecord& rec = require_alive("boolean", id);
      for (const TopoDS_Shape& s : rec.shapes) {
        const auto it = owners.find(s);
        if (it != owners.end()) {
          append_unique(out, it->second);
        }
      }
    }
  }
  return out;
}

// One driver for the whole BOP family. Every BRepAlgoAPI_BuilderAlgo descendant shares
// the arguments, options, history and error channels, so only the operand assignment (on
// the concrete type, because SetTools is not virtual) and whether the operation replaces
// or extends the model differ. `op` must already carry its arguments and tools.
py::dict Session::run_bop(const char* op_name, BRepAlgoAPI_BuilderAlgo& op,
                   const std::vector<TopoDS_Shape>& consumed, bool additive, double fuzzy,
                   bool parallel, const ProgressHooks& hooks, const PieceSplit* pieces) {
  // Bodies the boolean does not consume pass straight through and keep their identity.
  const std::vector<TopoDS_Shape> survivors =
      additive ? root_bodies(state_.root) : bodies_excluding(consumed);

  // Constructed under the GIL, before it is released, and destroyed after it is back: the
  // driver holds Python references and starts the thread that touches them.
  ProgressDriver driver(op_name, hooks);
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::string errors;
  std::string warnings;
  std::string refusal;
  std::pair<std::string, std::string> leak;
  {
    py::gil_scoped_release release;
    // The history IS the naming substrate, not a diagnostic: without it every id in the
    // consumed bodies would die at this operation.
    op.SetToFillHistory(true);
    // Not an option, and not a performance knob. BOPAlgo's default is destructive: it
    // updates the argument TShapes in place. A session's whole snapshot contract rests on
    // an operation never mutating a shape an earlier state still points at, so every
    // boolean runs non-destructively and OCCT copies what it needs to change. It is
    // deliberately not a parameter: exposing it would let a caller switch off the property
    // every retained snapshot depends on.
    op.SetNonDestructive(true);
    op.SetRunParallel(parallel);
    if (fuzzy > 0.0) {
      op.SetFuzzyValue(fuzzy);
    }
    try {
      op.Build(driver.range());
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.") + op_name + ": OCCT's boolean threw: " +
                         e.what());
    }
    if (!op.IsDone() || op.HasErrors()) {
      std::ostringstream s;
      op.DumpErrors(s);
      errors = s.str();
    } else {
      result = op.Shape();
      hist = op.History();
      std::ostringstream w;
      op.DumpWarnings(w);
      warnings = w.str();
      try {
        refusal = empty_result_refusal(op_name, op, result, fuzzy);
        leak = leaky_solid_refusal(op_name, op, result);
      } catch (const std::exception& e) {
        py::gil_scoped_acquire acquire;
        throw PysmeshError(std::string("Session.") + op_name +
                           ": the check of OCCT's result threw: " + e.what());
      }
    }
  }
  driver.finish();
  // A cancelled boolean records its own BOPAlgo_AlertUserBreak, so it lands in the failure
  // path above with an error string. It is caught here first, because "the caller stopped
  // it" and "the geometry defeated it" need different handling and the message text is not
  // a safe way to tell them apart.
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled(op_name);
  }
  if (!errors.empty() || result.IsNull()) {
    throw PysmeshError(std::string("Session.") + op_name +
                           ": the boolean failed; no partial result is returned.",
                       errors, {});
  }
  const std::vector<std::string> warning_list = warning_lines(warnings);
  if (!refusal.empty()) {
    std::string details =
        "BRepAlgoAPI reported no error. Faces that nearly coincide with an edge, a seam or a "
        "pole of the other operand (3e-7 to 1e-4 apart) are the known trigger; build the "
        "operands so that those features coincide exactly. Raising fuzzy is not monotone: a "
        "value that fails can work with a larger and with a smaller fuzzy value. OCCT "
        "warnings:";
    if (warning_list.empty()) {
      details += " none.";
    }
    for (const std::string& w : warning_list) {
      details += " " + w;
    }
    throw PysmeshError(refusal + " The session is unchanged.", details, {});
  }
  if (!leak.first.empty()) {
    throw PysmeshError(leak.first, leak.second, {});
  }
  if (pieces != nullptr) {
    const TopoDS_Shape kept = pieces->refine(result, hist);
    return commit(concat(survivors, kept), hist, op_name, kept, Validation::Strict,
                  warning_list, PieceRule::new_piece_ids(TopAbs_SOLID));
  }
  return commit(concat(survivors, result), hist, op_name, result, Validation::Strict,
                warning_list);
}

// The edges of every contour OCCT could not build a fillet on. This is the diagnostic
// that turns "the fillet failed" into "the fillet failed on these edges"; when the
// builder cannot report it, the caller is blamed for every edge it named instead.
std::vector<TopoDS_Shape> Session::faulty_edges(BRepFilletAPI_MakeFillet& mk) {
  std::vector<TopoDS_Shape> out;
  try {
    for (int i = 1; i <= mk.NbFaultyContours(); ++i) {
      const int contour = mk.FaultyContour(i);
      for (int j = 1; j <= mk.NbEdges(contour); ++j) {
        out.push_back(mk.Edge(contour, j));
      }
    }
  } catch (const std::exception&) {
    out.clear();
  }
  return out;
}

// Entity ids for the named edges that OCCT blamed, or all of them when it blamed none.
std::vector<int> Session::blamed_ids(const std::vector<EntityId>& edge_ids,
                                     const std::vector<TopoDS_Shape>& edges,
                                     const std::vector<TopoDS_Shape>& faulty) {
  if (faulty.empty()) {
    return ids_as_int(edge_ids);
  }
  std::vector<int> out;
  for (std::size_t i = 0; i < edges.size() && i < edge_ids.size(); ++i) {
    for (const TopoDS_Shape& f : faulty) {
      if (edges[i].IsSame(f)) {
        out.push_back(static_cast<int>(edge_ids[i]));
        break;
      }
    }
  }
  return out.empty() ? ids_as_int(edge_ids) : out;
}

NCollection_List<TopoDS_Shape> Session::solids_of(const char* op, const char* argname,
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

}  // namespace session
}  // namespace pysmesh
