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

#include <iomanip>
#include <numeric>

#include <BOPTools_AlgoTools3D.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <IntTools_Context.hxx>

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

// ---- closing sewn shells into solids ----------------------------------------------- //

// A closed shell of a sewing result, oriented, and the solid it bounds on its own.
struct ClosedShell {
  TopoDS_Shell shell;
  TopoDS_Solid solid;
  double volume = 0.0;
  double area = 0.0;
  bool exact = true;  // false: volume and area are the sign check's (enclosed_volume)
  Bnd_Box box;
};

// A closed shell that cannot become a solid, why, and the other shell it was placed against
// when that is the reason (null otherwise). The reason names that shell as "{other}", which
// the caller replaces with its face ids: only the caller holds the registry.
struct ShellRefusal {
  TopoDS_Shell shell;
  std::string reason;
  TopoDS_Shell other;
};

// What closing the shells of a sewing result gives: the shape to commit, or a refusal.
struct Closure {
  TopoDS_Shape result;
  std::optional<ShellRefusal> refusal;
};

std::string six_digits(double v) {
  std::ostringstream s;
  s << std::setprecision(6) << v;
  return s.str();
}

TopoDS_Solid solid_of(const TopoDS_Shell& shell) {
  BRepBuilderAPI_MakeSolid mk(shell);
  return mk.IsDone() ? mk.Solid() : TopoDS_Solid();
}

// Orient one closed shell so that the solid it bounds has its matter inside, then measure
// that solid. Returns the reason it cannot be committed, or nothing.
//
// The orientation comes from BRepClass3d_SolidClassifier: the point at infinity is outside
// every solid, so if it classifies IN, the shell bounds its complement and is reversed.
// The shell is reversed rather than the solid, so the solid stays FORWARD, as every other
// solid in the model is. BRepLib::OrientClosedSolid does the same classification, but its
// header does not say which of the two it flips, and OCCT's source is not in this tree.
//
// The volume is then the check the classifier cannot give (enclosed_volume). Before 4.2.2 a
// sewn tube was committed at -37.165 and a pillow of two coincident rectangles at 0, both
// reported valid, because BRepCheck_Analyzer accepts either orientation. A solid is committed
// only when it encloses more than eps x its area, eps = Precision::Confusion().
std::optional<std::string> orient_and_measure(ClosedShell& c) {
  const double eps = Precision::Confusion();
  c.solid = solid_of(c.shell);
  if (c.solid.IsNull()) {
    return std::string("BRepBuilderAPI_MakeSolid could not build a solid from it");
  }
  BRepClass3d_SolidClassifier where(c.solid);
  where.PerformInfinitePoint(eps);
  if (where.State() == TopAbs_IN) {
    c.shell.Reverse();
    c.solid = solid_of(c.shell);
  } else if (where.State() != TopAbs_OUT) {
    return std::string("the point at infinity classifies neither inside nor outside it (") +
           (where.State() == TopAbs_ON ? "ON" : "UNKNOWN") +
           "), so it has no side to call its inside";
  }

  BRepBndLib::Add(c.shell, c.box);
  const EnclosedVolume enclosed = enclosed_volume(c.solid);
  c.volume = enclosed.volume;
  c.area = enclosed.area;
  c.exact = enclosed.exact;
  const double tol = enclosed.tolerance;
  if (c.volume > tol) {
    return std::nullopt;
  }
  if (c.volume >= -tol) {
    return "it encloses a volume of " + six_digits(c.volume) +
           ", within Precision::Confusion() x its area (" + six_digits(tol) +
           "): its faces bound no interior, so it has no inside to orient by";
  }
  return "the point at infinity classifies outside it, yet the solid it bounds has volume " +
         six_digits(c.volume) +
         ": the classifier and the volume disagree about its inside. A shell that crosses "
         "itself does this";
}

// Replace the sign check's volume and area of a closed shell by the integrated ones.
void make_exact(ClosedShell& c) {
  if (c.exact) {
    return;
  }
  const EnclosedVolume enclosed = enclosed_volume(c.solid, /*precise=*/true);
  c.volume = enclosed.volume;
  c.area = enclosed.area;
  c.exact = true;
}

// Keep every shape of a sewing result that is not a shell: a face that sewed to nothing,
// a free edge. Before 4.2.2 the solid replaced the whole result, and those were deleted.
void keep_all_but_shells(const TopoDS_Shape& s, const BRep_Builder& b, TopoDS_Compound& into) {
  if (s.ShapeType() == TopAbs_SHELL) {
    return;
  }
  if (s.ShapeType() == TopAbs_COMPOUND) {
    for (TopoDS_Iterator it(s); it.More(); it.Next()) {
      keep_all_but_shells(it.Value(), b, into);
    }
    return;
  }
  b.Add(into, s);
}

// Where one closed shell lies against another.
enum class Placement { kOutside, kInside, kUndecided };

// Where shell j lies against shell i, whose solid `where` classifies.
//
// j is a cavity of i only if no point of j is outside i, so any point of j outside i settles
// that it is not: it touches i from outside, crosses it, or lies apart from it. The tests go
// from cheap to dear, and each runs only when the one before settled nothing:
//
//   1. j's vertices. One outside settles it. Parts of an assembly that touch are settled here.
//   2. The distance between the two surfaces. If they are apart, every point of j is on one
//      side of i, and the vertices said which: inside.
//   3. They meet. A point inside each face of j, as the defeature probe places one. One
//      outside settles it. With none outside, j touches i from inside or crosses it where no
//      point fell, and the points cannot tell which, so the placement is undecided.
Placement place(const ClosedShell& j, const ClosedShell& i,
                BRepClass3d_SolidClassifier& where,
                const Handle(IntTools_Context) & context) {
  const double eps = Precision::Confusion();
  bool inside = false;
  ShapeSet vertices;
  TopExp::MapShapes(j.shell, TopAbs_VERTEX, vertices);
  for (int k = 1; k <= vertices.Extent(); ++k) {
    where.Perform(BRep_Tool::Pnt(TopoDS::Vertex(vertices.FindKey(k))), eps);
    if (where.State() == TopAbs_OUT) {
      return Placement::kOutside;
    }
    inside = inside || where.State() == TopAbs_IN;
  }
  BRepExtrema_DistShapeShape gap(i.shell, j.shell);
  if (gap.IsDone() && gap.Value() > eps) {
    return inside ? Placement::kInside : Placement::kUndecided;
  }
  for (TopExp_Explorer ex(j.shell, TopAbs_FACE); ex.More(); ex.Next()) {
    gp_Pnt p;
    gp_Pnt2d uv;
    if (BOPTools_AlgoTools3D::PointInFace(TopoDS::Face(ex.Current()), p, uv, context) != 0) {
      continue;
    }
    where.Perform(p, eps);
    if (where.State() == TopAbs_OUT) {
      return Placement::kOutside;
    }
  }
  return Placement::kUndecided;
}

// Close every closed shell of a sewing result into a solid whose interior is on the inside.
//
// Each shell is oriented and measured on its own (orient_and_measure). Then the shells are
// placed against each other (place). Shell j is a cavity candidate of shell i only when
// their surfaces are apart and a point of j lies inside i. Shells that touch from outside
// or cross are separate solids, which may overlap as two bodies of the model may. A shell
// that meets another with no point outside it is refused. A shell's depth is the number of
// shells it lies inside:
//
//   * even depth: it is the outer shell of its own solid, so disjoint shells are separate
//     solids, and so is an island inside a cavity;
//   * odd depth: it is a cavity of the shell it lies directly inside, the one of depth one
//     less, reversed so that the matter of that solid is outside it.
//
// This is what the geometry says. Before 4.2.2 every closed shell went into one
// BRepBuilderAPI_MakeSolid, so two disjoint tubes became one solid with two outer shells.
// A solid is committed only when its outer volume less its cavities' is still above
// eps x its area.
Closure close_into_solids(const TopoDS_Shape& sewn, const std::vector<TopoDS_Shell>& shells) {
  const double eps = Precision::Confusion();
  const std::size_t n = shells.size();
  std::vector<ClosedShell> closed(n);
  Handle(IntTools_Context) context = new IntTools_Context;
  for (std::size_t i = 0; i < n; ++i) {
    closed[i].shell = shells[i];
    if (std::optional<std::string> why = orient_and_measure(closed[i])) {
      return {TopoDS_Shape(), ShellRefusal{shells[i], *why, TopoDS_Shell()}};
    }
  }

  std::vector<std::vector<std::size_t>> containers(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::optional<BRepClass3d_SolidClassifier> where;
    for (std::size_t j = 0; j < n; ++j) {
      if (j == i || closed[i].box.IsOut(closed[j].box)) {
        continue;
      }
      if (!where.has_value()) {
        where.emplace(closed[i].solid);
      }
      const Placement placement = place(closed[j], closed[i], *where, context);
      if (placement == Placement::kInside) {
        containers[j].push_back(i);
      } else if (placement == Placement::kUndecided) {
        return {TopoDS_Shape(),
                ShellRefusal{shells[j],
                             "it meets the closed shell of faces [{other}], and no point of "
                             "it lies outside that shell: whether it is a cavity of that "
                             "shell or crosses it is undecided",
                             shells[i]}};
      }
    }
  }

  std::vector<std::vector<std::size_t>> cavities(n);
  for (std::size_t j = 0; j < n; ++j) {
    const std::size_t depth = containers[j].size();
    if (depth % 2 == 0) {
      continue;
    }
    const auto parent =
        std::find_if(containers[j].begin(), containers[j].end(),
                     [&](std::size_t i) { return containers[i].size() + 1 == depth; });
    if (parent == containers[j].end()) {
      return {TopoDS_Shape(),
              ShellRefusal{shells[j],
                           "it lies inside " + std::to_string(depth) +
                               " other closed shells, but directly inside none of them: the "
                               "shells cross each other, so which of them it is a cavity of "
                               "is undecided",
                           TopoDS_Shell()}};
    }
    cavities[*parent].push_back(j);
  }

  std::vector<TopoDS_Shape> solids;
  for (std::size_t i = 0; i < n; ++i) {
    if (containers[i].size() % 2 != 0) {
      continue;
    }
    // A solid with cavities is decided on its volume less theirs, which the sign check
    // cannot bound, so those shells are integrated.
    if (!cavities[i].empty()) {
      for (std::size_t k : cavities[i]) {
        make_exact(closed[k]);
      }
      make_exact(closed[i]);
    }
    BRepBuilderAPI_MakeSolid mk(closed[i].shell);
    double volume = closed[i].volume;
    double area = closed[i].area;
    for (std::size_t j : cavities[i]) {
      mk.Add(TopoDS::Shell(closed[j].shell.Reversed()));
      volume -= closed[j].volume;
      area += closed[j].area;
    }
    if (!mk.IsDone() || volume <= eps * area) {
      return {TopoDS_Shape(),
              ShellRefusal{shells[i],
                           "less the cavities inside it, it encloses a volume of " +
                               six_digits(volume) + ", within Precision::Confusion() x its "
                               "area (" + six_digits(eps * area) + ")",
                           TopoDS_Shell()}};
    }
    solids.push_back(mk.Solid());
  }

  BRep_Builder b;
  TopoDS_Compound rest;
  b.MakeCompound(rest);
  keep_all_but_shells(sewn, b, rest);
  if (solids.size() == 1 && !TopoDS_Iterator(rest).More()) {
    return {solids.front(), std::nullopt};
  }
  TopoDS_Compound all;
  b.MakeCompound(all);
  for (const TopoDS_Shape& s : solids) {
    b.Add(all, s);
  }
  for (TopoDS_Iterator it(rest); it.More(); it.Next()) {
    b.Add(all, it.Value());
  }
  return {all, std::nullopt};
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
                [this, tolerance, make_solid, non_manifold](
                    const TopoDS_Shape& input, const Message_ProgressRange& range,
                    TopoDS_Shape& out, Handle(BRepTools_History) & hist) {
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
                  // that the faces did not sew into a watertight surface. One open shell
                  // leaves every shell open, as it always has.
                  std::vector<TopoDS_Shell> shells;
                  for (TopExp_Explorer ex(out, TopAbs_SHELL); ex.More(); ex.Next()) {
                    if (!BRep_Tool::IsClosed(ex.Current())) {
                      return;
                    }
                    shells.push_back(TopoDS::Shell(ex.Current()));
                  }
                  if (shells.empty()) {
                    return;
                  }
                  const Closure closure = close_into_solids(out, shells);
                  if (!closure.refusal.has_value()) {
                    out = closure.result;
                    return;
                  }
                  // A shell's faces are named by the ids of the faces they were sewn from.
                  // Sewing rebuilds a face whose edges it replaced, so a face is matched
                  // either as itself or through the history.
                  const auto ids_of_shell = [&](const TopoDS_Shell& shell) {
                    ShapeSet in_shell;
                    TopExp::MapShapes(shell, TopAbs_FACE, in_shell);
                    std::vector<EntityId> ids;
                    for (TopExp_Explorer ex(input, TopAbs_FACE); ex.More(); ex.Next()) {
                      bool hit = in_shell.Contains(ex.Current());
                      if (!hist.IsNull()) {
                        for (const TopoDS_Shape& m : hist->Modified(ex.Current())) {
                          hit = hit || in_shell.Contains(m);
                        }
                      }
                      if (hit) {
                        ids_on(ex.Current(), ids);
                      }
                    }
                    std::sort(ids.begin(), ids.end());
                    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
                    return ids;
                  };
                  const auto listed = [](const std::vector<EntityId>& ids) {
                    std::string text;
                    for (EntityId id : ids) {
                      text += (text.empty() ? "" : ", ") + std::to_string(id);
                    }
                    return text;
                  };
                  const ShellRefusal& refusal = *closure.refusal;
                  const std::vector<EntityId> faces = ids_of_shell(refusal.shell);
                  std::string reason = refusal.reason;
                  const std::string token = "{other}";
                  const std::size_t at = reason.find(token);
                  if (at != std::string::npos) {
                    reason.replace(at, token.size(), listed(ids_of_shell(refusal.other)));
                  }
                  throw PysmeshError(
                      "Session.sew: make_solid cannot close the shell of faces [" +
                          listed(faces) + "] into a solid: " + reason +
                          ". Nothing is committed; the session is unchanged.",
                      "A solid is committed only with its interior on the inside and a "
                      "volume above Precision::Confusion() x its area. A closed shell is "
                      "a cavity only when its surface is apart from the shell it lies in.",
                      ids_as_int(faces));
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

namespace {

using EdgeFaces = NCollection_IndexedDataMap<TopoDS_Shape, NCollection_List<TopoDS_Shape>,
                                             TopTools_ShapeMapHasher>;

double edge_length_of(const TopoDS_Shape& s) {
  double total = 0.0;
  ShapeSet edges;
  TopExp::MapShapes(s, TopAbs_EDGE, edges);
  for (int i = 1; i <= edges.Extent(); ++i) {
    const TopoDS_Edge& e = TopoDS::Edge(edges.FindKey(i));
    if (BRep_Tool::Degenerated(e)) {
      continue;
    }
    GProp_GProps length;
    BRepGProp::LinearProperties(e, length);
    total += length.Mass();
  }
  return total;
}

TopoDS_Compound compound_of(const std::vector<TopoDS_Shape>& shapes) {
  TopoDS_Compound c;
  BRep_Builder b;
  b.MakeCompound(c);
  for (const TopoDS_Shape& s : shapes) {
    b.Add(c, s);
  }
  return c;
}

// What one removal did to its body, read off the faces it changed.
//
// Both measures are differences over the faces that differ between the input and the
// result. Every other face is the same face in both — the same shape, or the same face
// issued again (see changed_faces()) — so it contributes the same to both, and it is left
// out rather than integrated twice and subtracted. What remains is the patch the removal
// replaced: the named faces, and the neighbours OCCT extended over them.
//
// The tolerances come from Precision::Confusion(), eps, the distance below which OCCT treats
// two points as one: a replacement patch that nowhere lies farther than eps from the patch
// it replaced is, to the kernel, the same surface.
//
//   * Volume. The volume between two such patches is at most eps x the larger of their
//     areas. The input's patch is the named faces, of area A; the result's is A + dA,
//     because the neighbours are carried over otherwise. tol_volume = eps x max(A, A + dA).
//   * Area. A removal that changes nothing only re-trims surfaces that were already there,
//     and re-trimming along an edge moved by at most eps changes the area of each of the two
//     faces it bounds by at most eps x its length. tol_area = 2 x eps x the length of the
//     named faces' edges.
//
// A removal is judged to have changed nothing only inside BOTH tolerances. A real feature
// leaves one of them: a hole or a pocket adds volume, a boss removes it, a fillet changes
// both.
//
// The tolerances are taken over the patch, not over the body, because a feature is local.
// eps x the body's whole area bounds what moving the WHOLE boundary by eps can do, and a
// small real feature moves far less volume than that. Measured on the production STEP
// assembly, a whole-body tolerance refused 14 of 48 single-face removals, among them a strip
// of area 2.5e-5 whose removal moved it by 3.1e-5 on average — 310 x eps — on a body whose
// whole-area tolerance was three times the change. Over the patch its tolerance is 2.5e-12,
// and the change is 310 times that.
struct Removal {
  double d_volume = 0.0;
  double d_area = 0.0;
  double named_area = 0.0;
  double named_edge_length = 0.0;
  double tol_volume = 0.0;
  double tol_area = 0.0;

  bool unchanged() const {
    return std::abs(d_volume) <= tol_volume && std::abs(d_area) <= tol_area;
  }
};

// The faces a removal changed: `gone` from the input, `made` in the result.
struct ChangedFaces {
  std::vector<TopoDS_Shape> gone;
  std::vector<TopoDS_Shape> made;
};

double area_of(const TopoDS_Shape& s) {
  GProp_GProps props;
  BRepGProp::SurfaceProperties(s, props);
  return props.Mass();
}

// The faces of the input the result does not have, and the reverse, less every pair that is
// one face issued twice.
//
// OCCT's defeaturing rebuilds the solid, and most faces come back as new shapes that are the
// faces they were. Measured on a production body of 49 faces, 29 left the input and 28
// arrived in the result for the removal of one small cylinder. Kept, each such pair costs
// two integrations and contributes nothing but their round-off. A pair is dropped when both
// faces lie on one surface — the same Geom_Surface, location and orientation — and their
// areas agree within 2 x eps x the removed face's perimeter: by the tolerance below, that is
// a face whose trimming moved less than eps, which is a face that did not change. A named
// face is never paired: it is the patch being removed.
ChangedFaces changed_faces(const TopoDS_Shape& owner, const TopoDS_Shape& result,
                           const std::vector<TopoDS_Shape>& named) {
  ShapeSet in_owner, in_result, named_set;
  TopExp::MapShapes(owner, TopAbs_FACE, in_owner);
  TopExp::MapShapes(result, TopAbs_FACE, in_result);
  for (const TopoDS_Shape& f : named) {
    named_set.Add(f);
  }
  ChangedFaces c;
  ShapeSet seen;
  for (TopExp_Explorer ex(result, TopAbs_FACE); ex.More(); ex.Next()) {
    if (!in_owner.Contains(ex.Current()) && !seen.Contains(ex.Current())) {
      seen.Add(ex.Current());
      c.made.push_back(ex.Current());
    }
  }
  std::vector<double> made_area(c.made.size(), -1.0);
  std::vector<bool> paired(c.made.size(), false);
  seen.Clear();
  for (TopExp_Explorer ex(owner, TopAbs_FACE); ex.More(); ex.Next()) {
    const TopoDS_Shape& g = ex.Current();
    if (in_result.Contains(g) || seen.Contains(g)) {
      continue;
    }
    seen.Add(g);
    if (named_set.Contains(g)) {
      c.gone.push_back(g);
      continue;
    }
    TopLoc_Location g_loc;
    const Handle(Geom_Surface) & g_surf = BRep_Tool::Surface(TopoDS::Face(g), g_loc);
    const double g_area = area_of(g);
    const double same = 2.0 * Precision::Confusion() * edge_length_of(g);
    bool twin = false;
    for (std::size_t j = 0; j < c.made.size() && !twin; ++j) {
      if (paired[j] || c.made[j].Orientation() != g.Orientation()) {
        continue;
      }
      TopLoc_Location m_loc;
      const Handle(Geom_Surface) & m_surf =
          BRep_Tool::Surface(TopoDS::Face(c.made[j]), m_loc);
      if (m_surf != g_surf || !m_loc.IsEqual(g_loc)) {
        continue;
      }
      if (made_area[j] < 0.0) {
        made_area[j] = area_of(c.made[j]);
      }
      if (std::abs(made_area[j] - g_area) <= same) {
        paired[j] = true;
        twin = true;
      }
    }
    if (!twin) {
      c.gone.push_back(g);
    }
  }
  std::vector<TopoDS_Shape> made;
  for (std::size_t j = 0; j < c.made.size(); ++j) {
    if (!paired[j]) {
      made.push_back(c.made[j]);
    }
  }
  c.made = std::move(made);
  return c;
}

// Measure the removal that turned `owner` into `result` by deleting `named`.
//
// Both integrals use BRepGProp's adaptive rule. Its default rule integrates each face with a
// fixed number of Gauss points, which is exact enough on an analytic face and not on a
// free-form one: on a tube swept along a spline, whose B-spline wall is split in two on one
// surface and whose half-wall "removal" is a no-op, the fixed rule measures a volume change
// of 3.0e-6, well past the 1.9e-6 tolerance, where the adaptive rule measures 1.3e-8.
//
// The volume change is integrated in ONE call, over the result's changed faces together with
// the input's changed faces reversed. Those bound exactly the region between the two
// patches, a closed surface, so its volume is the change and does not depend on the point it
// is taken about. Two separate calls would not do: the volume of an open patch does depend
// on that point, and VolumeProperties picks its own from each shape it is given. Measured on
// an imprinted box, the removed and the new patch over one 13 x 11 face came out 326.857
// and 343.2 in two calls, for a change that is zero.
//
// Each integral is asked for a precision set by the tolerance it serves:
//
//   * Area. The rule refines each face until two steps agree to Eps relative, so over the
//     changed faces, of area A_c, a tenth of tol_area asks for Eps = 0.1 x tol_area / A_c.
//   * Volume. A face contributes (1/3) x the integral of (x - p).n over it, p the point
//     VolumeProperties takes the volume about. OCCT does not document p, and its source is
//     not in this tree; it is inferred from what VolumeProperties returns. A lone planar face
//     measures exactly 0, so p lies in that face's own plane, which means p is taken from the
//     shape and not fixed at the origin. Taken from the shape, p lies inside its bounding
//     box, so |x - p| is at most D, the box's diagonal. Eps x D x A_c / 3 then bounds the
//     error, and a tenth of eps x A asks for Eps = 0.3 x eps x A / (D x A_c).
//
// Both sides of the area difference are integrated with the same Eps, because the rule's
// error is reproducible where it is not what the rule reports. On the tube, its whole-body
// area wanders between 42.529 and 42.541 as Eps goes from 1e-6 to 1e-12, while it reports
// reaching 1e-12, and yet at any one Eps the no-op's two sides agree to 2e-9. The check
// compares a difference, so the difference is what has to be exact.
Removal measure_removal(const TopoDS_Shape& owner, const TopoDS_Shape& result,
                        const std::vector<TopoDS_Shape>& named) {
  const double eps = Precision::Confusion();
  Removal r;
  const TopoDS_Compound named_faces = compound_of(named);
  r.named_area = area_of(named_faces);
  r.named_edge_length = edge_length_of(named_faces);
  r.tol_area = 2.0 * eps * r.named_edge_length;

  const ChangedFaces changed = changed_faces(owner, result, named);
  const TopoDS_Compound gone = compound_of(changed.gone);
  const TopoDS_Compound made = compound_of(changed.made);
  std::vector<TopoDS_Shape> boundary = changed.made;
  for (const TopoDS_Shape& g : changed.gone) {
    boundary.push_back(g.Reversed());
  }
  const TopoDS_Compound between = compound_of(boundary);

  // Guarded against a zero, not floored at a size: a floor would be a unit.
  const double changed_area = std::max(area_of(between), eps);
  Bnd_Box box;
  BRepBndLib::Add(between, box);
  const double diagonal = std::max(std::sqrt(box.SquareExtent()), eps);
  const double area_eps = std::min(0.1 * r.tol_area / changed_area, kAdaptiveEpsCap);
  const double volume_eps =
      std::min(0.3 * eps * r.named_area / (diagonal * changed_area), kAdaptiveEpsCap);

  GProp_GProps area_gone, area_made;
  BRepGProp::SurfaceProperties(gone, area_gone, area_eps);
  BRepGProp::SurfaceProperties(made, area_made, area_eps);
  r.d_area = area_made.Mass() - area_gone.Mass();
  GProp_GProps volume;
  BRepGProp::VolumeProperties(between, volume, volume_eps);
  r.d_volume = volume.Mass();
  r.tol_volume = eps * std::max(r.named_area, r.named_area + r.d_area);
  return r;
}

// The named faces grouped into features: sets connected through shared edges, as indices
// into `faces`, each ascending, ordered by their first face.
//
// This is the grouping BOPAlgo_RemoveFeatures sorts its input into before it removes the
// features one at a time, so a set here is what OCCT removes as one unit — the two
// half-walls of a hole are one feature, two separate holes are two.
std::vector<std::vector<std::size_t>> feature_blocks(const std::vector<TopoDS_Shape>& faces) {
  std::vector<std::size_t> parent(faces.size());
  std::iota(parent.begin(), parent.end(), std::size_t{0});
  const auto root = [&parent](std::size_t i) {
    while (parent[i] != i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  ShapeKeyed<std::size_t> first_owner;
  for (std::size_t i = 0; i < faces.size(); ++i) {
    for (TopExp_Explorer ex(faces[i], TopAbs_EDGE); ex.More(); ex.Next()) {
      const auto [it, inserted] = first_owner.emplace(ex.Current(), i);
      if (!inserted) {
        const std::size_t a = root(i);
        const std::size_t b = root(it->second);
        parent[std::max(a, b)] = std::min(a, b);
      }
    }
  }
  std::vector<std::vector<std::size_t>> blocks;
  std::vector<std::size_t> slot(faces.size(), faces.size());
  for (std::size_t i = 0; i < faces.size(); ++i) {
    const std::size_t r = root(i);
    if (slot[r] == faces.size()) {
      slot[r] = blocks.size();
      blocks.emplace_back();
    }
    blocks[slot[r]].push_back(i);
  }
  return blocks;
}

// Whether a feature's surface still bounds the defeatured body, and through which face.
struct Probe {
  // True when the feature has to be removed on its own to be judged: a point inside one of
  // its faces lies on a face OCCT extended, or no point inside a face could be placed.
  bool candidate = false;
  // The face of the input body whose extension the point lies on, or null.
  TopoDS_Shape absorber;
};

// Look for the one way a removal can leave a feature's surface where it was: OCCT fills a
// feature by extending the faces adjacent to it, and a neighbour lying on the same surface
// extends straight back over the feature. So a point inside each named face is taken, and
// the neighbours' extended images are asked whether they pass through it.
//
// A real removal takes the named face's interior off the boundary — into the material for
// a hole, out of it for a boss — and the extended neighbours meet it along its edges at
// most, so the point, being inside the face, lies off them by a distance of the feature's
// own size. The test is a filter, not the verdict: a candidate is only refused once
// removing it on its own has been measured to change nothing. A face that no interior
// point can be placed on is made a candidate for the same reason: being wrong in that
// direction costs one more removal, and being wrong in the other would commit a no-op.
Probe probe_feature(const std::vector<TopoDS_Shape>& block, const EdgeFaces& edge_faces,
                    const Handle(BRepTools_History) & hist,
                    const Handle(IntTools_Context) & context) {
  ShapeSet own;
  for (const TopoDS_Shape& f : block) {
    own.Add(f);
  }
  Probe out;
  for (const TopoDS_Shape& f : block) {
    const TopoDS_Face& face = TopoDS::Face(f);
    gp_Pnt p;
    gp_Pnt2d uv;
    if (hist.IsNull() || BOPTools_AlgoTools3D::PointInFace(face, p, uv, context) != 0) {
      out.candidate = true;
      continue;
    }
    const TopoDS_Vertex point = BRepBuilderAPI_MakeVertex(p);
    for (TopExp_Explorer ex(face, TopAbs_EDGE); ex.More(); ex.Next()) {
      const NCollection_List<TopoDS_Shape>* owners = edge_faces.Seek(ex.Current());
      if (owners == nullptr) {
        continue;
      }
      for (const TopoDS_Shape& neighbour : *owners) {
        if (own.Contains(neighbour)) {
          continue;
        }
        for (const TopoDS_Shape& image : hist->Modified(neighbour)) {
          if (image.ShapeType() != TopAbs_FACE) {
            continue;
          }
          BRepExtrema_DistShapeShape d(point, image);
          const double on = Precision::Confusion() +
                            std::max(BRep_Tool::Tolerance(face),
                                     BRep_Tool::Tolerance(TopoDS::Face(image)));
          if (d.IsDone() && d.Value() <= on) {
            out.candidate = true;
            if (out.absorber.IsNull()) {
              out.absorber = neighbour;
            }
          }
        }
      }
    }
  }
  return out;
}

// Remove one feature on its own. Null when OCCT declines it or keeps any of its faces.
TopoDS_Shape remove_alone(const TopoDS_Shape& owner, const std::vector<TopoDS_Shape>& block,
                          bool parallel, const Message_ProgressRange& range) {
  BRepAlgoAPI_Defeaturing op;
  op.SetShape(owner);
  NCollection_List<TopoDS_Shape> to_remove;
  for (const TopoDS_Shape& f : block) {
    to_remove.Append(f);
  }
  op.AddFacesToRemove(to_remove);
  op.SetToFillHistory(true);
  op.SetRunParallel(parallel);
  op.Build(range);
  if (!op.IsDone() || op.HasErrors()) {
    return TopoDS_Shape();
  }
  for (const TopoDS_Shape& f : block) {
    if (!op.IsDeleted(f)) {
      return TopoDS_Shape();
    }
  }
  return op.Shape();
}

// A named feature whose removal changed nothing.
struct IdleFeature {
  std::vector<std::size_t> faces;  // indices into the named faces
  TopoDS_Shape absorber;           // the input face extended over it, or null
  Removal removal;                 // what removing it did
  bool declined_alone = false;     // OCCT removes nothing for it on its own
};

std::vector<TopoDS_Shape> pick(const std::vector<TopoDS_Shape>& faces,
                               const std::vector<std::size_t>& idx) {
  std::vector<TopoDS_Shape> out;
  for (std::size_t i : idx) {
    out.push_back(faces[i]);
  }
  return out;
}

// The named features whose removal changed nothing. Empty when every one changed the body.
//
// Judged per feature, because one call can name a real feature and a no-op together, and
// the real one's change would hide the other's. The verdict is always measure_removal(),
// applied to a removal that isolates the feature:
//
//   * If the whole removal changed nothing, no named feature changed anything, and all of
//     them are returned. With one feature — the common case — this is the only test, and it
//     costs one measurement of the faces the removal changed.
//   * Otherwise, with several features, each one probe_feature() cannot clear is removed
//     from the input on its own and measured. A legitimate call clears every probe and pays
//     no extra removal; a candidate pays one.
std::vector<IdleFeature> idle_features(const TopoDS_Shape& owner,
                                       const std::vector<TopoDS_Shape>& faces,
                                       const std::vector<std::vector<std::size_t>>& blocks,
                                       const TopoDS_Shape& result,
                                       const Handle(BRepTools_History) & hist,
                                       bool parallel, const Message_ProgressRange& range) {
  const Removal whole = measure_removal(owner, result, faces);
  if (blocks.size() == 1 && !whole.unchanged()) {
    return {};
  }
  EdgeFaces edge_faces;
  TopExp::MapShapesAndAncestors(owner, TopAbs_EDGE, TopAbs_FACE, edge_faces);
  Handle(IntTools_Context) context = new IntTools_Context;

  std::vector<IdleFeature> out;
  if (whole.unchanged()) {
    for (const std::vector<std::size_t>& block : blocks) {
      const Probe probe = probe_feature(pick(faces, block), edge_faces, hist, context);
      out.push_back({block, probe.absorber, whole, false});
    }
    return out;
  }
  std::vector<std::pair<std::size_t, Probe>> candidates;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    Probe probe = probe_feature(pick(faces, blocks[b]), edge_faces, hist, context);
    if (probe.candidate) {
      candidates.emplace_back(b, probe);
    }
  }
  Message_ProgressScope each(range, nullptr,
                             static_cast<double>(std::max<std::size_t>(candidates.size(), 1)));
  for (const auto& [b, probe] : candidates) {
    if (!each.More()) {
      break;  // cancelled: the caller raises that before it reads anything returned here
    }
    const std::vector<TopoDS_Shape> block = pick(faces, blocks[b]);
    const TopoDS_Shape alone = remove_alone(owner, block, parallel, each.Next());
    if (alone.IsNull()) {
      out.push_back({blocks[b], probe.absorber, Removal(), true});
      continue;
    }
    const Removal removal = measure_removal(owner, alone, block);
    if (removal.unchanged()) {
      out.push_back({blocks[b], probe.absorber, removal, false});
    }
  }
  return out;
}

std::string format_g(double v, int digits) {
  std::ostringstream s;
  s << std::setprecision(digits) << v;
  return s.str();
}

std::string face_list(const std::vector<EntityId>& ids) {
  std::string out = ids.size() == 1 ? "face " : "faces ";
  for (std::size_t i = 0; i < ids.size(); ++i) {
    out += (i == 0 ? "" : ", ") + std::to_string(ids[i]);
  }
  return out;
}

}  // namespace

py::dict Session::defeature(const std::vector<EntityId>& face_ids, bool parallel,
                            const py::object& progress, const py::object& cancel) {
  OpGuard guard(in_op_);
  const std::vector<TopoDS_Shape> faces = faces_of("defeature", face_ids);
  const TopoDS_Shape owner = sole_owner_body(body_of_subshape(), faces);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({owner});
  const std::vector<std::vector<std::size_t>> blocks = feature_blocks(faces);

  ProgressDriver driver("defeature", hooks_of("defeature", progress, cancel));
  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::string diagnostics;
  std::vector<std::size_t> kept;
  std::vector<IdleFeature> idle;
  {
    py::gil_scoped_release release;
    // One step for one feature, so its progress is the removal's own. Two for several: the
    // removal, then the removals that isolate a feature the probe could not clear.
    Message_ProgressScope steps(driver.range(), nullptr, blocks.size() > 1 ? 2.0 : 1.0);
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
      op.Build(steps.Next());
      std::ostringstream s;
      op.DumpErrors(s);
      op.DumpWarnings(s);
      diagnostics = s.str();
      if (op.IsDone() && !op.HasErrors()) {
        // The first post-condition. Handed an incomplete feature — a blind hole's wall
        // without the flat face capping it — OCCT reports the refusal as a *warning*,
        // leaves IsDone() true and HasErrors() false, and returns the input unchanged.
        // Believing IsDone() would commit a no-op as a success and tell the caller their
        // feature is gone when it is still there.
        for (std::size_t i = 0; i < faces.size(); ++i) {
          if (!op.IsDeleted(faces[i])) {
            kept.push_back(i);
          }
        }
        if (kept.empty()) {
          result = op.Shape();
          hist = op.History();
          // The second, and the one that reads the body rather than the faces' ids. A face
          // whose surface continues into a neighbour is "removed" by extending that
          // neighbour straight back over it: IsDeleted() is true, and the body is the one
          // that went in. Measured on a plate whose hole wall is two half-cylinders: naming
          // one of them deleted it, grew the other to the full wall, and changed the volume
          // by 0 of 415.43.
          idle = idle_features(owner, faces, blocks, result, hist, parallel,
                               blocks.size() > 1 ? steps.Next() : Message_ProgressRange());
        }
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.defeature: OCCT's defeaturing threw: ") +
                         e.what());
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
  if (!idle.empty()) {
    // What the body measures, quoted in the message so the caller sees what was left
    // unchanged. Only a refusal pays for it, and GProp's fixed rule is precise enough for a
    // number printed to six digits.
    GProp_GProps body_volume, body_area;
    BRepGProp::VolumeProperties(owner, body_volume);
    BRepGProp::SurfaceProperties(owner, body_area);
    std::vector<std::size_t> blamed_idx;
    std::string message = "Session.defeature: ";
    std::string details;
    for (std::size_t k = 0; k < idle.size(); ++k) {
      const IdleFeature& f = idle[k];
      std::vector<EntityId> named;
      for (std::size_t i : f.faces) {
        named.push_back(face_ids[i]);
        blamed_idx.push_back(i);
      }
      std::vector<EntityId> absorber_ids;
      if (!f.absorber.IsNull()) {
        ids_on(f.absorber, absorber_ids);
      }
      const bool plural = named.size() > 1;
      message += k == 0 ? "removing " : " Removing ";
      if (f.declined_alone) {
        message += face_list(named) + " on " + (plural ? "their" : "its") +
                   " own removes nothing: OCCT declines it";
      } else {
        message += face_list(named) + " left the body's volume " +
                   format_g(body_volume.Mass(), 6) + " and area " +
                   format_g(body_area.Mass(), 6) + " unchanged";
        details += face_list(named) + ": the removal changed the volume by " +
                   format_g(f.removal.d_volume, 3) + " (tolerance " +
                   format_g(f.removal.tol_volume, 3) + ") and the area by " +
                   format_g(f.removal.d_area, 3) + " (tolerance " +
                   format_g(f.removal.tol_area, 3) + "). ";
      }
      if (!absorber_ids.empty()) {
        // A merge can leave several ids on the face; the lowest is its label.
        const EntityId label = *std::min_element(absorber_ids.begin(), absorber_ids.end());
        message += std::string("; ") + (plural ? "their" : "its") +
                   " surface continues into face " + std::to_string(label) +
                   ", so OCCT extended that face over " + (plural ? "them" : "it");
      }
      message += ".";
    }
    message +=
        " Name every face of the feature. A face that only splits one surface with its "
        "neighbour is not a feature: unify_same_domain merges it.";
    details +=
        "A removal is refused when it changes the volume by no more than "
        "Precision::Confusion() x the area of the patch it replaced, and the area by no more "
        "than 2 x Precision::Confusion() x the length of the named faces' edges: that is all "
        "moving the patch by the confusion distance can change. The session is unchanged.";
    std::sort(blamed_idx.begin(), blamed_idx.end());
    std::vector<EntityId> blamed;
    for (std::size_t i : blamed_idx) {
      blamed.push_back(face_ids[i]);
    }
    throw PysmeshError(message, details, ids_as_int(blamed));
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
    } catch (const PysmeshError&) {
      // A refusal the repair raised itself already names what it refused and carries its
      // face ids, so it leaves as it is rather than as an OCCT failure.
      py::gil_scoped_acquire acquire;
      throw;
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
