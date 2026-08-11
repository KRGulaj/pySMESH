// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-06

// pySMESH binding — Session: healing, sewing, defeaturing, imprinting and removal.
//
// These are the operations whose input is allowed to be broken, and that fact shapes the
// whole file. Three consequences run through it:
//
//   * They report validity rather than demanding it. Every other operation refuses to commit
//     a shape BRepCheck_Analyzer rejects, which is right when the input was valid and the
//     result should be too. Here the input is invalid by assumption, so refusing a result
//     that is *less* invalid than before would make the operation useless on exactly the
//     shapes it exists for. The verdict travels on the delta instead.
//   * They are scoped. ShapeFix and friends repair whatever shape they are handed, so
//     restricting a heal to chosen bodies costs nothing — and buys the property a global
//     healing pass cannot offer: bodies outside the scope are never passed to the algorithm,
//     so every entity in them stays byte-identical rather than merely looking unchanged.
//   * Their history arrives through a BRepTools_ReShape context rather than from the
//     algorithm directly. ShapeFix_Shape, BRepBuilderAPI_Sewing and
//     ShapeUpgrade_RemoveInternalWires all record their substitutions in one, and
//     BRepTools_ReShape::History() converts it into the same Handle(BRepTools_History) the
//     booleans produce — so one carry routine serves every operation in the session.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

namespace pysmesh {
namespace session {

namespace {

// A ReShape context's history, or a null handle when the algorithm never allocated one.
// An empty history is the normal outcome of a successful repair, not a failure: many of
// ShapeFix's fixes are re-orientations and tolerance updates, which leave the TShape alone,
// so the shape is literally the same one and the registry carries it on identity.
Handle(BRepTools_History) history_of_context(const Handle(BRepTools_ReShape) & context) {
  if (context.IsNull()) {
    return Handle(BRepTools_History)();
  }
  return context->History();
}

}  // namespace

// ---- healing ------------------------------------------------------------------------ //

py::dict Session::heal(const std::optional<std::vector<EntityId>>& entity_ids,
                       double precision, double min_tolerance, double max_tolerance,
                       const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  require_positive("precision", precision);
  require_positive("min_tolerance", min_tolerance);
  require_positive("max_tolerance", max_tolerance);
  if (max_tolerance < min_tolerance) {
    throw PysmeshError("Session.heal: max_tolerance (" + std::to_string(max_tolerance) +
                       ") must be >= min_tolerance (" + std::to_string(min_tolerance) + ").");
  }
  return rework(entity_ids, "heal", hooks_of("heal", progress, cancel),
                [precision, min_tolerance, max_tolerance](
                    const TopoDS_Shape& input, const Message_ProgressRange& range,
                    TopoDS_Shape& out, Handle(BRepTools_History) & hist) {
                  ShapeFix_Shape fixer;
                  // The context is supplied rather than left to the algorithm, because it is
                  // the only channel through which the repair's history reaches the registry.
                  Handle(ShapeBuild_ReShape) context = new ShapeBuild_ReShape;
                  fixer.SetContext(context);
                  fixer.Init(input);
                  fixer.SetPrecision(precision);
                  fixer.SetMinTolerance(min_tolerance);
                  fixer.SetMaxTolerance(max_tolerance);
                  fixer.Perform(range);
                  out = fixer.Shape();
                  hist = history_of_context(context);
                });
}

py::dict Session::sew(const std::vector<EntityId>& entity_ids, double tolerance,
                      bool make_solid, bool non_manifold, const py::object& progress,
                      const py::object& cancel) {
  OpGuard guard(in_op_);
  require_positive("tolerance", tolerance);
  if (entity_ids.empty()) {
    throw PysmeshError("Session.sew: at least one entity must be named.");
  }
  return rework(entity_ids, "sew", hooks_of("sew", progress, cancel),
                [tolerance, make_solid, non_manifold](const TopoDS_Shape& input,
                                                      const Message_ProgressRange& range,
                                                      TopoDS_Shape& out,
                                                      Handle(BRepTools_History) & hist) {
                  BRepBuilderAPI_Sewing sewing(tolerance, /*sewing=*/true,
                                               /*analysis=*/true, /*cutting=*/true,
                                               non_manifold);
                  // Add, not Load: Load names a *context* shape, while Add names the shapes
                  // that are actually sewed. The scoped bodies go in one at a time.
                  for (TopoDS_Iterator it(input); it.More(); it.Next()) {
                    sewing.Add(it.Value());
                  }
                  sewing.Perform(range);
                  out = sewing.SewedShape();
                  hist = history_of_context(sewing.GetContext());
                  if (!make_solid || out.IsNull()) {
                    return;
                  }
                  // Only a closed shell bounds a volume. An open one is left as a shell:
                  // wrapping it would produce a shape whose interior is undefined, and the
                  // validity check would then fail for a reason that hides the real one —
                  // that the faces did not sew into a watertight surface.
                  BRepBuilderAPI_MakeSolid mk;
                  int shells = 0;
                  for (TopExp_Explorer ex(out, TopAbs_SHELL); ex.More(); ex.Next()) {
                    if (!BRep_Tool::IsClosed(ex.Current())) {
                      return;
                    }
                    mk.Add(TopoDS::Shell(ex.Current()));
                    ++shells;
                  }
                  if (shells > 0 && mk.IsDone()) {
                    out = mk.Solid();
                  }
                });
}

py::dict Session::remove_internal_wires(
    const std::optional<std::vector<EntityId>>& entity_ids, double min_area,
    bool remove_faces) {
  OpGuard guard(in_op_);
  require_positive("min_area", min_area);
  // No hooks: ShapeUpgrade_RemoveInternalWires::Perform takes no Message_ProgressRange in
  // OCCT 8.0, so there is nothing to drive and the range argument is ignored below.
  return rework(entity_ids, "remove_internal_wires", ProgressHooks{},
                [min_area, remove_faces](const TopoDS_Shape& input,
                                         const Message_ProgressRange&, TopoDS_Shape& out,
                                         Handle(BRepTools_History) & hist) {
                  ShapeUpgrade_RemoveInternalWires remover(input);
                  remover.MinArea() = min_area;
                  remover.RemoveFaceMode() = remove_faces;
                  remover.Perform();
                  out = remover.GetResult();
                  hist = history_of_context(remover.Context());
                });
}

py::dict Session::unify_same_domain(const std::optional<std::vector<EntityId>>& entity_ids,
                                    bool unify_faces, bool unify_edges, bool concat_bsplines,
                                    double linear_tol, double angular_tol_rad) {
  OpGuard guard(in_op_);
  if (!unify_faces && !unify_edges) {
    throw PysmeshError(
        "Session.unify_same_domain: at least one of unify_faces and unify_edges must be "
        "true; with both off the operation has nothing to do.");
  }
  require_positive("linear_tol", linear_tol);
  // Zero is meaningful here and is the stateless API's default: OCCT clamps anything below
  // Precision::Angular() up to it, so 0 asks for the tightest angle the kernel admits.
  require_non_negative("angular_tol_rad", angular_tol_rad);
  // No hooks: ShapeUpgrade_UnifySameDomain::Build takes no Message_ProgressRange in OCCT 8.0.
  return rework(entity_ids, "unify_same_domain", ProgressHooks{},
                [unify_faces, unify_edges, concat_bsplines, linear_tol, angular_tol_rad](
                    const TopoDS_Shape& input, const Message_ProgressRange&,
                    TopoDS_Shape& out, Handle(BRepTools_History) & hist) {
                  ShapeUpgrade_UnifySameDomain unify(input, unify_faces, unify_edges,
                                                     concat_bsplines);
                  unify.SetLinearTolerance(linear_tol);
                  unify.SetAngularTolerance(angular_tol_rad);
                  unify.Build();
                  out = unify.Shape();
                  hist = unify.History();
                });
}

// ---- defeaturing --------------------------------------------------------------------- //

py::dict Session::defeature(const std::vector<EntityId>& face_ids, bool parallel,
                            const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  const std::vector<TopoDS_Shape> faces = faces_of("defeature", face_ids);
  const TopoDS_Shape owner = sole_owner_body(body_of_subshape(), faces);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({owner});

  ProgressDriver driver("defeature", hooks_of("defeature", progress, cancel));
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::string diagnostics;
  std::vector<std::size_t> kept;
  {
    py::gil_scoped_release release;
    BRepAlgoAPI_Defeaturing op;
    op.SetShape(owner);
    NCollection_List<TopoDS_Shape> to_remove;
    for (const TopoDS_Shape& f : faces) {
      to_remove.Append(f);
    }
    op.AddFacesToRemove(to_remove);
    op.SetToFillHistory(true);
    op.SetRunParallel(parallel);
    try {
      op.Build(driver.range());
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.defeature: OCCT's defeaturing threw: ") +
                         e.what());
    }
    std::ostringstream s;
    op.DumpErrors(s);
    op.DumpWarnings(s);
    diagnostics = s.str();
    if (op.IsDone() && !op.HasErrors()) {
      // The post-condition that makes the operation trustworthy. Handed an incomplete
      // feature — a blind hole's wall without the flat face capping it — OCCT reports the
      // refusal as a *warning*, leaves IsDone() true and HasErrors() false, and returns the
      // input unchanged. Believing IsDone() would commit a no-op as a success and tell the
      // caller their feature is gone when it is still there.
      for (std::size_t i = 0; i < faces.size(); ++i) {
        if (!op.IsDeleted(faces[i])) {
          kept.push_back(i);
        }
      }
      if (kept.empty()) {
        result = op.Shape();
        hist = op.History();
      }
    }
  }
  driver.finish();
  // Ahead of the post-condition check: a cancelled defeaturing has removed nothing, and
  // blaming the caller's faces for still being present would be a false diagnostic.
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled("defeature");
  }
  if (!kept.empty()) {
    std::vector<EntityId> blamed;
    for (std::size_t i : kept) {
      blamed.push_back(face_ids[i]);
    }
    throw PysmeshError(
        "Session.defeature: OCCT removed no feature for " + std::to_string(kept.size()) +
            " of the " + std::to_string(faces.size()) + " named faces.",
        diagnostics.empty()
            ? std::string("OCCT reported success but the faces are still present. "
                          "Defeaturing removes a complete feature: name every face of it, "
                          "including the flats that cap a blind hole.")
            : diagnostics,
        ids_as_int(blamed));
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.defeature: the defeaturing failed; no partial result is "
                       "returned.",
                       diagnostics, ids_as_int(face_ids));
  }
  return commit(concat(survivors, result), hist, "defeature", result);
}

// ---- imprinting ---------------------------------------------------------------------- //

py::dict Session::imprint(const std::vector<EntityId>& targets,
                          const std::vector<EntityId>& tools, double fuzzy, bool parallel,
                          int glue, const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  if (targets.empty()) {
    throw PysmeshError("Session.imprint: targets must name at least one entity.");
  }
  if (tools.empty()) {
    throw PysmeshError("Session.imprint: tools must name at least one entity.");
  }
  require_fuzzy("imprint", fuzzy);
  if (glue < 0 || glue > 2) {
    throw PysmeshError("Session.imprint: glue must be 0 (off), 1 (partial coincidence) or "
                       "2 (full coincidence); got " +
                       std::to_string(glue) + ".");
  }
  const std::vector<TopoDS_Shape> target_bodies =
      operand_bodies("imprint", "targets", targets);
  const std::vector<TopoDS_Shape> tool_bodies = operand_bodies("imprint", "tools", tools);
  for (const TopoDS_Shape& t : tool_bodies) {
    for (const TopoDS_Shape& a : target_bodies) {
      if (a.IsSame(t)) {
        throw PysmeshError(
            "Session.imprint: a body appears in both targets and tools. Imprinting a body "
            "onto itself is not a request OCCT can answer.");
      }
    }
  }

  BRepAlgoAPI_Splitter op;
  NCollection_List<TopoDS_Shape> args, tls;
  for (const TopoDS_Shape& s : target_bodies) {
    args.Append(s);
  }
  for (const TopoDS_Shape& s : tool_bodies) {
    tls.Append(s);
  }
  op.SetArguments(args);
  op.SetTools(tls);
  // Gluing skips the intersection step for operands the caller declares only touch. It is a
  // large speed-up on an assembly of coincident-faced parts and silently wrong on operands
  // that genuinely interpenetrate, so it stays off unless asked for.
  op.SetGlue(static_cast<BOPAlgo_GlueEnum>(glue));
  // The tools are not consumed: an imprint exists to put the interface into the target, and
  // a tool the caller still holds ids for must not vanish as a side effect.
  return run_bop("imprint", op, target_bodies, /*additive=*/false, fuzzy, parallel,
                 hooks_of("imprint", progress, cancel));
}

// ---- removal ------------------------------------------------------------------------- //

py::dict Session::remove(const std::vector<EntityId>& entity_ids) {
  OpGuard guard(in_op_);
  const std::vector<TopoDS_Shape> doomed = owner_bodies("remove", entity_ids);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(doomed);
  if (survivors.size() == root_bodies(state_.root).size()) {
    throw PysmeshError("Session.remove: the named entities belong to no body of the session "
                       "root, so nothing would be removed.");
  }
  // Committed with no history and nothing built. Every id whose shapes have left the model
  // dies; an id on a sub-shape a surviving body also owns stays alive, because that shape is
  // still there — which is the intact rule doing exactly what it exists for.
  return commit(survivors, Handle(BRepTools_History)(), "remove", TopoDS_Shape());
}

// ---- the scoped-rework driver -------------------------------------------------------- //

py::dict Session::rework(const std::optional<std::vector<EntityId>>& entity_ids,
                         const char* op_name, const ProgressHooks& hooks,
                         const std::function<void(const TopoDS_Shape&,
                                                  const Message_ProgressRange&,
                                                  TopoDS_Shape&,
                                                  Handle(BRepTools_History)&)>& run) {
  const std::vector<TopoDS_Shape> scope = scoped_bodies(entity_ids, op_name);
  const std::vector<TopoDS_Shape> untouched = bodies_excluding(scope);
  const TopoDS_Shape input = make_root(scope);

  ProgressDriver driver(op_name, hooks);
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  {
    py::gil_scoped_release release;
    try {
      run(input, driver.range(), result, hist);
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.") + op_name + ": OCCT's repair threw: " +
                         e.what());
    }
  }
  driver.finish();
  // This check is what stops the repair family committing a partial model, and it has to be
  // the driver's own flag rather than anything the algorithm reports. A cancelled
  // ShapeFix_Shape returns Perform() == false and a shape that is NOT null — measured at 436
  // of an assembly's 5606 faces. The null test below would pass it straight through, and
  // committing it would delete every entity the repair had not reached yet.
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled(op_name);
  }
  if (result.IsNull()) {
    throw PysmeshError(std::string("Session.") + op_name +
                       ": OCCT produced no shape; the session is unchanged.");
  }
  return commit(concat(untouched, result), hist, op_name, result, Validation::Report);
}

}  // namespace session
}  // namespace pysmesh
