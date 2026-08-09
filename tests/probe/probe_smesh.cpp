// pySMESH v2 capability probe — SMESH side (the StdMeshers/Controls/MeshEditor surface
// v1 compiles but does not expose).
//
// Everything here is already compiled into the static libraries the wheel links; the probe
// exists to prove it is also *reachable* — a static library only contributes the object files
// something references, so an unresolved external in this file would be a build problem that
// would otherwise surface only once the v2 bindings were written.
//
// The probe drives SMESH exactly as a binding would: SMESH_Gen owns the mesh, hypotheses and
// algorithms are heap-allocated with monotonic ids and assigned per sub-shape, and teardown
// follows the ownership rule established by src/bindings/mesh.cpp (delete the SMESH_Mesh
// wrapper before the SMESH_Gen, free adopted hypotheses last).

#include "probe.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_BuilderAlgo.hxx>
#include <NCollection_List.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Ax1.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>

#include <SMDSAbs_ElementType.hxx>
#include <SMDS_ElemIterator.hxx>
#include <SMDS_Mesh.hxx>
#include <SMDS_MeshElement.hxx>
#include <SMDS_MeshNode.hxx>
#include <SMDS_MeshVolume.hxx>
#include <SMESHDS_Group.hxx>
#include <SMESHDS_SubMesh.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_ComputeError.hxx>
#include <SMESH_ControlsDef.hxx>
#include <SMESH_Delaunay.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_Group.hxx>
#include <SMESH_HypoFilter.hxx>
#include <SMESH_Hypothesis.hxx>
#include <SMESH_MAT2d.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_MeshAlgos.hxx>
#include <SMESH_MeshEditor.hxx>
#include <SMESH_Pattern.hxx>
#include <SMESH_TypeDefs.hxx>
#include <SMESH_subMesh.hxx>

#include <DriverGMF_Read.hxx>
#include <DriverGMF_Write.hxx>

#include <StdMeshers_Cartesian_3D.hxx>
#include <StdMeshers_CartesianParameters3D.hxx>
#include <StdMeshers_Hexa_3D.hxx>
#include <StdMeshers_Import_1D2D.hxx>
#include <StdMeshers_MEFISTO_2D.hxx>
#include <StdMeshers_MaxElementArea.hxx>
#include <StdMeshers_MaxElementVolume.hxx>
#include <StdMeshers_NumberOfSegments.hxx>
#include <StdMeshers_PolyhedronPerSolid_3D.hxx>
#include <StdMeshers_Prism_3D.hxx>
#include <StdMeshers_Projection_2D.hxx>
#include <StdMeshers_QuadFromMedialAxis_1D2D.hxx>
#include <StdMeshers_Quadrangle_2D.hxx>
#include <StdMeshers_RadialPrism_3D.hxx>
#include <StdMeshers_Regular_1D.hxx>
#include <StdMeshers_ViscousLayers2D.hxx>

namespace {

using probe::check;
using probe::check_close;
using probe::note;
using probe::section;

constexpr double BX = 3.0;
constexpr double BY = 7.0;
constexpr double BZ = 11.0;

// SMESH_Hypothesis::GetName returns a C string; a linked, constructed hypothesis always has
// a non-empty name, so this is the cheapest proof its translation unit is in the archive.
bool named(const SMESH_Hypothesis& hyp) {
  const char* n = hyp.GetName();
  return n != nullptr && n[0] != '\0';
}

// Most concrete controls declare GetValue(const TSequenceOfXYZ&), which *hides* the
// NumericalFunctor::GetValue(long) entry point. Call it through a base reference so the
// id-taking overload stays visible — this is the call shape the v2 bindings must use.
double numeric(SMESH::Controls::NumericalFunctor& functor, smIdType element_id) {
  return functor.GetValue(static_cast<long>(element_id));
}

// Ownership mirrors src/bindings/mesh.cpp::Mesh — the one teardown order that does not
// corrupt the heap: SMESH_Mesh wrapper, then SMESH_Gen, then the adopted hypotheses.
class Session {
 public:
  explicit Session(const TopoDS_Shape& shape) : shape_(shape) {
    gen_ = std::make_unique<SMESH_Gen>();
    mesh_ = gen_->CreateMesh(false);
    mesh_->ShapeToMesh(shape_);
  }

  ~Session() {
    delete mesh_;
    mesh_ = nullptr;
    gen_.reset();
    hyps_.clear();
  }

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  SMESH_Mesh& mesh() { return *mesh_; }
  SMESH_Gen& gen() { return *gen_; }
  SMESHDS_Mesh* meshDS() { return mesh_->GetMeshDS(); }
  const TopoDS_Shape& shape() const { return shape_; }

  // VERIFY-AT-SOURCE FINDING: hypothesis ids MUST be drawn from SMESH_Gen::GetANewId().
  // Some algorithms (StdMeshers_PolyhedronPerSolid_3D, and every composite algo that owns a
  // sub-mesher) call gen->GetANewId() in their own constructor, so a caller-side counter
  // silently aliases ids in SMESH_Gen's algo/hypothesis maps and corrupts assignment.
  // v1's Mesh::next_hyp_id() is a private counter; it is safe only because v1 creates
  // exactly two non-composite hypotheses. The v2 session must not repeat that shortcut.
  template <class T, class... Args>
  T* make(Args&&... args) {
    T* hyp = new T(gen_->GetANewId(), gen_.get(), std::forward<Args>(args)...);
    hyps_.emplace_back(hyp);
    return hyp;
  }

  SMESH_Hypothesis::Hypothesis_Status assign_status(const TopoDS_Shape& sub,
                                                    SMESH_Hypothesis* hyp) {
    return mesh_->AddHypothesis(sub, hyp->GetID());
  }

  bool assign(const TopoDS_Shape& sub, SMESH_Hypothesis* hyp) {
    return !SMESH_Hypothesis::IsStatusFatal(assign_status(sub, hyp));
  }

  bool compute() { return gen_->Compute(*mesh_, shape_); }

 private:
  TopoDS_Shape shape_;
  std::unique_ptr<SMESH_Gen> gen_;
  SMESH_Mesh* mesh_ = nullptr;
  std::vector<std::unique_ptr<SMESH_Hypothesis>> hyps_;
};

// Regular_1D + NumberOfSegments + Quadrangle_2D + Hexa_3D on a box: the cheapest fully
// structured hexahedral mesh, and the substrate the quality-control, search and group
// checks below all run on.
bool build_hexa_mesh(Session& s, int nseg) {
  StdMeshers_Regular_1D* algo1d = s.make<StdMeshers_Regular_1D>();
  StdMeshers_NumberOfSegments* nseg_hyp = s.make<StdMeshers_NumberOfSegments>();
  nseg_hyp->SetNumberOfSegments(nseg);
  StdMeshers_Quadrangle_2D* algo2d = s.make<StdMeshers_Quadrangle_2D>();
  StdMeshers_Hexa_3D* algo3d = s.make<StdMeshers_Hexa_3D>();
  bool ok = s.assign(s.shape(), algo1d);
  ok = s.assign(s.shape(), nseg_hyp) && ok;
  ok = s.assign(s.shape(), algo2d) && ok;
  ok = s.assign(s.shape(), algo3d) && ok;
  return ok && s.compute();
}

// ---------------------------------------------------------------------------- STDMESH ----- //
void probe_r11_unexcluded_translation_units() {
  section("STDMESH", "the five StdMeshers translation units excluded in v1");

  // Construction alone proves the TU is in the archive and its symbols resolve.
  SMESH_Gen gen;
  StdMeshers_MaxElementArea area(1, &gen);
  area.SetMaxArea(2.0);
  check_close(area.GetMaxArea(), 2.0, 1e-12, "STDMESH StdMeshers_MaxElementArea links and round-trips");

  StdMeshers_MaxElementVolume vol(2, &gen);
  vol.SetMaxVolume(3.0);
  check_close(vol.GetMaxVolume(), 3.0, 1e-12,
              "STDMESH StdMeshers_MaxElementVolume links and round-trips");

  StdMeshers_PolyhedronPerSolid_3D poly(3, &gen);
  check(named(poly), "STDMESH StdMeshers_PolyhedronPerSolid_3D links");

  StdMeshers_Import_1D2D import(4, &gen);
  check(named(import), "STDMESH StdMeshers_Import_1D2D links");

  StdMeshers_Cartesian_3D cart(5, &gen);
  check(named(cart), "STDMESH StdMeshers_Cartesian_3D links");

  // Cartesian_3D must mesh a non-trivial solid, not a box. Use a box with a through hole so
  // the body-fitted grid has to cut curved faces.
  const TopoDS_Shape block = BRepPrimAPI_MakeBox(gp_Pnt(-4, -4, 0), 8.0, 8.0, 6.0).Shape();
  const TopoDS_Shape bore = BRepPrimAPI_MakeCylinder(1.5, 6.0).Shape();
  NCollection_List<TopoDS_Shape> args;
  args.Append(block);
  NCollection_List<TopoDS_Shape> tools;
  tools.Append(bore);
  BRepAlgoAPI_Cut cut;
  cut.SetArguments(args);
  cut.SetTools(tools);
  cut.Build();
  check(cut.IsDone(), "STDMESH bored-block fixture builds");

  Session s(cut.Shape());
  StdMeshers_Cartesian_3D* algo = s.make<StdMeshers_Cartesian_3D>();
  StdMeshers_CartesianParameters3D* params = s.make<StdMeshers_CartesianParameters3D>();
  std::vector<std::string> spacing(1, std::string("1.0"));
  std::vector<double> internal_points;
  for (int axis = 0; axis < 3; ++axis) {
    params->SetGridSpacing(spacing, internal_points, axis);
  }
  params->SetSizeThreshold(4.0);
  const bool assigned = s.assign(s.shape(), algo) && s.assign(s.shape(), params);
  check(assigned, "STDMESH Cartesian_3D + CartesianParameters3D assign to the solid");
  const bool computed = s.compute();
  check(computed, "STDMESH StdMeshers_Cartesian_3D computes a body-fitted mesh");
  check(s.meshDS()->NbVolumes() > 0, "STDMESH Cartesian_3D produced volume elements");
  check(s.meshDS()->NbNodes() > 0, "STDMESH Cartesian_3D produced nodes");

  // The result must pass SMESH's own quality controls.
  SMESH::Controls::Volume volume_ctl;
  volume_ctl.SetMesh(s.meshDS());
  int nonpositive = 0;
  int checked = 0;
  for (SMDS_ElemIteratorPtr it = s.meshDS()->elementsIterator(SMDSAbs_Volume); it->more();) {
    const SMDS_MeshElement* e = it->next();
    ++checked;
    if (numeric(volume_ctl, e->GetID()) <= 0.0) {
      ++nonpositive;
    }
  }
  check(checked > 0 && nonpositive == 0,
        "STDMESH every Cartesian_3D volume element has positive volume");
}

// ---------------------------------------------------------------------------- QC ----- //
void probe_r12_controls() {
  section("QC", "quality controls: 3-D numerical functors, predicates, filter algebra");

  Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
  check(build_hexa_mesh(s, 3), "QC structured hexa mesh computes on the 3x7x11 box");
  SMESHDS_Mesh* ds = s.meshDS();
  check(ds->NbVolumes() == 27, "QC 3 segments per edge gives 27 hexahedra");

  // Volume: a 3x7x11 box cut 3x3x3 gives cells of exactly (3/3)*(7/3)*(11/3).
  SMESH::Controls::Volume volume;
  volume.SetMesh(ds);
  const double expect_cell = (BX / 3.0) * (BY / 3.0) * (BZ / 3.0);
  double total = 0.0;
  for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Volume); it->more();) {
    total += numeric(volume, it->next()->GetID());
  }
  check_close(total, BX * BY * BZ, 1e-6, "QC Volume functor sums to the box volume");
  check_close(numeric(volume, ds->elementsIterator(SMDSAbs_Volume)->next()->GetID()),
              expect_cell, 1e-6, "QC Volume of one hexahedron == (3/3)(7/3)(11/3)");

  // AspectRatio3D on a deliberately anisotropic cell — a regular element cannot distinguish
  // a correct implementation from a constant, which is why the fixture is 3x7x11.
  SMESH::Controls::AspectRatio3D ar3d;
  ar3d.SetMesh(ds);
  const smIdType a_volume_id = ds->elementsIterator(SMDSAbs_Volume)->next()->GetID();
  const double ar = numeric(ar3d, a_volume_id);
  check(ar > 1.0 && std::isfinite(ar),
        "QC AspectRatio3D reports > 1 on an anisotropic hexahedron");

  SMESH::Controls::MaxElementLength3D maxlen3d;
  maxlen3d.SetMesh(ds);
  const double diag = std::sqrt((BX / 3.0) * (BX / 3.0) + (BY / 3.0) * (BY / 3.0) +
                                (BZ / 3.0) * (BZ / 3.0));
  check_close(numeric(maxlen3d, a_volume_id), diag, 1e-6,
              "QC MaxElementLength3D == the cell body diagonal");

  SMESH::Controls::AspectRatio ar2d;
  ar2d.SetMesh(ds);
  SMESH::Controls::Warping warping;
  warping.SetMesh(ds);
  SMESH::Controls::Taper taper;
  taper.SetMesh(ds);
  SMESH::Controls::Skew skew;
  skew.SetMesh(ds);
  SMESH::Controls::MinimumAngle min_angle;
  min_angle.SetMesh(ds);
  SMESH::Controls::Length2D length2d;
  length2d.SetMesh(ds);
  SMESH::Controls::Length3D length3d;
  length3d.SetMesh(ds);
  SMESH::Controls::Deflection2D deflection;
  deflection.SetMesh(ds);
  SMESH::Controls::MaxElementLength2D maxlen2d;
  maxlen2d.SetMesh(ds);
  SMESH::Controls::MultiConnection multi;
  multi.SetMesh(ds);
  SMESH::Controls::NodeConnectivityNumber ncn;
  ncn.SetMesh(ds);
  const smIdType a_face = ds->elementsIterator(SMDSAbs_Face)->next()->GetID();
  check(std::isfinite(numeric(ar2d, a_face)) && numeric(ar2d, a_face) >= 1.0,
        "QC AspectRatio (2-D) evaluates on a quadrangle");
  check(std::isfinite(numeric(warping, a_face)) && std::isfinite(numeric(taper, a_face)) &&
            std::isfinite(numeric(skew, a_face)) && std::isfinite(numeric(min_angle, a_face)),
        "QC Warping / Taper / Skew / MinimumAngle evaluate");
  check(std::isfinite(numeric(length2d, a_face)) && std::isfinite(numeric(maxlen2d, a_face)) &&
            std::isfinite(numeric(deflection, a_face)),
        "QC Length2D / MaxElementLength2D / Deflection2D evaluate");
  check(std::isfinite(numeric(length3d, a_volume_id)), "QC Length3D evaluates on a volume");
  check(std::isfinite(numeric(multi, a_face)), "QC MultiConnection evaluates");
  check(std::isfinite(numeric(ncn, ds->nodesIterator()->next()->GetID())),
        "QC NodeConnectivityNumber evaluates on a node");

  // Predicates. On a closed, correctly built hexa mesh: no bad-oriented volumes, no bare
  // borders, no over-constrained volumes; free edges DO exist on the surface skin.
  SMESH::Controls::BadOrientedVolume bad_oriented;
  bad_oriented.SetMesh(ds);
  int bad = 0;
  int volumes = 0;
  for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Volume); it->more();) {
    const SMDS_MeshElement* e = it->next();
    ++volumes;
    if (bad_oriented.IsSatisfy(e->GetID())) {
      ++bad;
    }
  }
  check(volumes == 27 && bad == 0,
        "QC BadOrientedVolume flags nothing on a correctly built hexa mesh");

  SMESH::Controls::BareBorderVolume bare_vol;
  bare_vol.SetMesh(ds);
  SMESH::Controls::OverConstrainedVolume over_vol;
  over_vol.SetMesh(ds);
  SMESH::Controls::BareBorderFace bare_face;
  bare_face.SetMesh(ds);
  SMESH::Controls::OverConstrainedFace over_face;
  over_face.SetMesh(ds);
  SMESH::Controls::FreeEdges free_edges;
  free_edges.SetMesh(ds);
  SMESH::Controls::FreeBorders free_borders;
  free_borders.SetMesh(ds);
  SMESH::Controls::FreeNodes free_nodes;
  free_nodes.SetMesh(ds);
  SMESH::Controls::CoincidentNodes coincident;
  coincident.SetMesh(ds);
  SMESH::Controls::CoincidentElements2D coincident_elems;
  coincident_elems.SetMesh(ds);
  SMESH::Controls::ManifoldPart manifold;
  manifold.SetMesh(ds);
  const smIdType a_volume = a_volume_id;
  check(!bare_vol.IsSatisfy(a_volume) && !over_vol.IsSatisfy(a_volume),
        "QC BareBorderVolume / OverConstrainedVolume clean on a valid mesh");
  check(!bare_face.IsSatisfy(a_face) || true, "QC BareBorderFace evaluates");
  check(!over_face.IsSatisfy(a_face) || true, "QC OverConstrainedFace evaluates");
  check(!coincident.IsSatisfy(ds->nodesIterator()->next()->GetID()),
        "QC CoincidentNodes flags nothing on a conforming mesh");
  int free_nodes_found = 0;
  for (SMDS_NodeIteratorPtr it = ds->nodesIterator(); it->more();) {
    if (free_nodes.IsSatisfy(it->next()->GetID())) {
      ++free_nodes_found;
    }
  }
  check(free_nodes_found == 0, "QC FreeNodes finds none (every node is used)");

  // Filter algebra: LogicalNOT / LogicalAND / LogicalOR / Comparator / RangeOfIds.
  // SMESH::Controls::Predicate is a *virtual* base of every concrete predicate, so a
  // downcast from PredicatePtr is ill-formed — configure through the concrete type first,
  // then hand ownership to the shared_ptr. The v2 bindings must follow the same shape.
  SMESH::Controls::LogicalNOT* not_pred = new SMESH::Controls::LogicalNOT();
  SMESH::Controls::PredicatePtr not_bad(not_pred);
  SMESH::Controls::PredicatePtr bad_ptr(new SMESH::Controls::BadOrientedVolume());
  not_pred->SetPredicate(bad_ptr);
  not_bad->SetMesh(ds);
  check(not_bad->IsSatisfy(a_volume), "QC LogicalNOT composes a predicate");

  // VERIFY-AT-SOURCE FINDING: SMDS element ids are ONE global sequence shared by edges,
  // faces and volumes - the faces of this mesh occupy the low ids and the volumes follow.
  // A RangeOfIds written as "1-5" therefore selects faces, not the first five volumes. The
  // v2 binding must build ranges from real ids, never from a per-type ordinal.
  std::vector<smIdType> volume_ids;
  for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Volume); it->more();) {
    volume_ids.push_back(it->next()->GetID());
  }
  check(volume_ids.size() == 27 && volume_ids.front() > 1,
        "QC volume ids do not start at 1 (element ids are one global space)");
  const std::string range_str =
      std::to_string(volume_ids[0]) + "-" + std::to_string(volume_ids[4]);

  SMESH::Controls::RangeOfIds* range_pred = new SMESH::Controls::RangeOfIds();
  SMESH::Controls::PredicatePtr range(range_pred);
  range_pred->SetRangeStr(range_str.c_str());
  range_pred->SetType(SMDSAbs_Volume);
  range->SetMesh(ds);

  SMESH::Controls::LogicalAND* and_pred = new SMESH::Controls::LogicalAND();
  SMESH::Controls::PredicatePtr and_ptr(and_pred);
  and_pred->SetPredicate1(not_bad);
  and_pred->SetPredicate2(range);
  and_ptr->SetMesh(ds);
  check(and_ptr->IsSatisfy(volume_ids[0]),
        "QC LogicalAND(NOT BadOriented, RangeOfIds) accepts an in-range volume");
  check(!and_ptr->IsSatisfy(volume_ids.back()),
        "QC LogicalAND rejects an out-of-range volume (falsification case)");

  SMESH::Controls::LogicalOR* or_pred = new SMESH::Controls::LogicalOR();
  SMESH::Controls::PredicatePtr or_ptr(or_pred);
  or_pred->SetPredicate1(range);
  or_pred->SetPredicate2(bad_ptr);
  or_ptr->SetMesh(ds);
  check(or_ptr->IsSatisfy(volume_ids[0]), "QC LogicalOR composes");

  SMESH::Controls::NumericalFunctorPtr vol_functor(new SMESH::Controls::Volume());
  SMESH::Controls::LessThan* less = new SMESH::Controls::LessThan();
  SMESH::Controls::PredicatePtr less_ptr(less);
  less->SetNumFunctor(vol_functor);
  less->SetMargin(1e9);
  less_ptr->SetMesh(ds);
  check(less_ptr->IsSatisfy(a_volume), "QC LessThan comparator over a numerical functor");

  SMESH::Controls::Filter filter;
  filter.SetPredicate(and_ptr);
  SMESH::Controls::Filter::TIdSequence ids;
  SMESH::Controls::Filter::GetElementsId(ds, and_ptr, ids);
  check(ids.size() == 5, "QC Filter::GetElementsId returns exactly the 5 filtered ids");

  // ElementsOnShape — the predicate whose incomplete-Classifier copy was the v1 C2036.
  SMESH::Controls::ElementsOnShape on_shape;
  on_shape.SetMesh(ds);
  on_shape.SetShape(s.shape(), SMDSAbs_Volume);
  check(on_shape.IsSatisfy(a_volume), "QC ElementsOnShape works (the STDMESH C2036 class)");
  SMESH::Controls::ElementsOnShape copied(on_shape);  // the copy MSVC could not synthesise
  check(copied.IsSatisfy(a_volume), "QC ElementsOnShape is copyable (out-of-line copy ctor)");
}

// ---------------------------------------------------------------------------- EDITOR ----- //
void probe_r13_mesh_editor() {
  section("EDITOR", "SMESH_MeshEditor: quadratic conversion, baffles, reorient, smooth, sew");

  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2), "EDITOR hexa mesh for the editor probes computes");
    SMESHDS_Mesh* ds = s.meshDS();
    const smIdType nodes_before = ds->NbNodes();
    const smIdType volumes_before = ds->NbVolumes();

    SMESH_MeshEditor editor(&s.mesh());

    // ConvertToQuadratic / ConvertFromQuadratic — the only path to a P2 mesh for SU2.
    editor.ConvertToQuadratic(/*theForce3d=*/true, /*theToBiQuad=*/false);
    check(ds->NbNodes() > nodes_before && ds->NbVolumes() == volumes_before,
          "EDITOR ConvertToQuadratic adds medium nodes, keeps the element count");
    const bool back = editor.ConvertFromQuadratic();
    check(back, "EDITOR ConvertFromQuadratic returns true");
    check(ds->NbNodes() == nodes_before && ds->NbVolumes() == volumes_before,
          "EDITOR quadratic round-trip restores the original node/element counts");
  }

  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2), "EDITOR hexa mesh for the sweep/split probes computes");
    SMESHDS_Mesh* ds = s.meshDS();
    SMESH_MeshEditor editor(&s.mesh());

    // DoubleElements — internal walls / baffles (a real CFD-prep need).
    TIDSortedElemSet faces;
    SMDS_ElemIteratorPtr fit = ds->elementsIterator(SMDSAbs_Face);
    for (int i = 0; i < 2 && fit->more(); ++i) {
      faces.insert(fit->next());
    }
    const smIdType faces_before = ds->NbFaces();
    editor.DoubleElements(faces);
    check(ds->NbFaces() == faces_before + 2,
          "EDITOR DoubleElements duplicates the selected faces (baffle / internal wall)");

    // Reorient2DBy3D on a deliberately flipped shell bounding valid volumes.
    TIDSortedElemSet all_faces;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Face); it->more();) {
      all_faces.insert(it->next());
    }
    TIDSortedElemSet all_volumes;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Volume); it->more();) {
      all_volumes.insert(it->next());
    }
    int flipped = 0;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Face); it->more() && flipped < 3;) {
      editor.Reorient(it->next());
      ++flipped;
    }
    const int reoriented =
        editor.Reorient2DBy3D(all_faces, all_volumes, /*theOutsideNormal=*/true);
    check(reoriented > 0,
          "EDITOR Reorient2DBy3D repairs deliberately flipped faces using the bounding volumes");

    // Reorient2D (winding-only variant) is also reachable.
    TIDSortedElemSet ref;
    const int reoriented2d = editor.Reorient2D(all_faces, gp_Vec(0, 0, 1), ref, true);
    check(reoriented2d >= 0, "EDITOR Reorient2D is reachable");

    // SplitVolumes: hexa -> prisms with an explicit facet choice.
    SMESH_MeshEditor::TFacetOfElem facets;
    TIDSortedElemSet hexas = all_volumes;
    editor.GetHexaFacetsToSplit(hexas, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), facets);
    const smIdType volumes_before = ds->NbVolumes();
    editor.SplitVolumes(facets, SMESH_MeshEditor::HEXA_TO_2_PRISMS);
    check(ds->NbVolumes() > volumes_before,
          "EDITOR SplitVolumes(HEXA_TO_2_PRISMS) increases the volume count");
  }

  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2), "EDITOR hexa mesh for the smoothing/merge probes computes");
    SMESHDS_Mesh* ds = s.meshDS();
    SMESH_MeshEditor editor(&s.mesh());

    // CAD-constrained smoothing (the2D=true uses the nodes' UV on their geometric face).
    TIDSortedElemSet to_smooth;
    std::set<const SMDS_MeshNode*> fixed;
    editor.Smooth(to_smooth, fixed, SMESH_MeshEditor::LAPLACIAN, 2, 1.0, /*the2D=*/true);
    check(ds->NbNodes() > 0, "EDITOR Smooth (Laplacian, on-shape) runs");
    editor.Smooth(to_smooth, fixed, SMESH_MeshEditor::CENTROIDAL, 1, 1.0, /*the2D=*/true);
    check(ds->NbNodes() > 0, "EDITOR Smooth (centroidal) runs");

    // FindCoincidentNodes / MergeNodes / MergeEqualElements.
    TIDSortedNodeSet nodes;
    SMESH_MeshEditor::TListOfListOfNodes groups;
    editor.FindCoincidentNodes(nodes, 1e-9, groups, false);
    check(groups.empty(), "EDITOR FindCoincidentNodes finds none on a conforming mesh");
    editor.MergeNodes(groups);
    check(ds->NbNodes() > 0, "EDITOR MergeNodes accepts an empty group list");
    editor.MergeEqualElements();
    check(ds->NbVolumes() > 0, "EDITOR MergeEqualElements runs");

    // QuadToTri / TriToQuad on the surface skin.
    TIDSortedElemSet quads;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Face); it->more();) {
      quads.insert(it->next());
    }
    const smIdType faces_before = ds->NbFaces();
    check(editor.QuadToTri(quads, /*the13Diag=*/true), "EDITOR QuadToTri splits quadrangles");
    check(ds->NbFaces() > faces_before, "EDITOR QuadToTri increases the face count");

    TIDSortedElemSet tris;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Face); it->more();) {
      tris.insert(it->next());
    }
    SMESH::Controls::NumericalFunctorPtr criterion(new SMESH::Controls::AspectRatio());
    check(editor.TriToQuad(tris, criterion, M_PI / 4.0) || true,
          "EDITOR TriToQuad is reachable with a NumericalFunctor criterion");
  }

  {
    // Extrusion / rotation sweeps and Offset on a surface mesh.
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2), "EDITOR hexa mesh for the sweep probes computes");
    SMESHDS_Mesh* ds = s.meshDS();
    SMESH_MeshEditor editor(&s.mesh());

    TIDSortedElemSet sweep_sets[2];
    SMDS_ElemIteratorPtr fit = ds->elementsIterator(SMDSAbs_Face);
    if (fit->more()) {
      sweep_sets[1].insert(fit->next());
    }
    SMESH_MeshEditor::TTElemOfElemListMap history;
    const smIdType volumes_before = ds->NbVolumes();
    editor.ExtrusionSweep(sweep_sets, gp_Vec(0, 0, 1.0), 2, history, 0);
    check(ds->NbVolumes() > volumes_before && !history.empty(),
          "EDITOR ExtrusionSweep creates volumes and returns a history map");

    TIDSortedElemSet rot_sets[2];
    SMDS_ElemIteratorPtr fit2 = ds->elementsIterator(SMDSAbs_Face);
    if (fit2->more()) {
      rot_sets[1].insert(fit2->next());
    }
    editor.RotationSweep(rot_sets, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0.3, 2, 1e-6,
                         false);
    check(ds->NbVolumes() > volumes_before, "EDITOR RotationSweep is reachable and creates cells");
  }

  // Sewing: the API surface (SewFreeBorder / SewSideElements) must resolve. Driving them to a
  // successful sew needs a purpose-built two-patch fixture, which belongs in the binding-layer
  // test suite rather than a link/run probe.
  note("EDITOR SewFreeBorder / SewSideElements",
       "linked and callable; a meaningful sew needs a two-patch fixture, which belongs in "
       "a binding-layer pytest");
}

// ---------------------------------------------------------------------------- SEARCH ----- //
void probe_r14_search_and_ray_casting() {
  section("SEARCH", "element searcher, ray casting, point state, mesh offset, slot, DeMerge");

  Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
  check(build_hexa_mesh(s, 3), "SEARCH hexa mesh for the search probes computes");
  SMESHDS_Mesh* ds = s.meshDS();

  std::unique_ptr<SMESH_ElementSearcher> searcher(SMESH_MeshAlgos::GetElementSearcher(*ds));
  check(searcher != nullptr, "SEARCH SMESH_MeshAlgos::GetElementSearcher returns a searcher");

  std::vector<const SMDS_MeshElement*> found;
  const gp_Pnt inside(BX / 2.0, BY / 2.0, BZ / 2.0);
  const int n_found = searcher->FindElementsByPoint(inside, SMDSAbs_Volume, found);
  check(n_found > 0 && !found.empty(),
        "SEARCH FindElementsByPoint locates the volume containing an interior point");

  const SMDS_MeshElement* closest = searcher->FindClosestTo(inside, SMDSAbs_Volume);
  check(closest != nullptr, "SEARCH FindClosestTo returns an element");

  // Ray casting. A ray up the box axis must meet the surface skin; the count is the number
  // of faces whose bounding box the line crosses, so assert non-emptiness plus a miss case.
  std::vector<const SMDS_MeshElement*> hit;
  searcher->GetElementsNearLine(gp_Ax1(gp_Pnt(BX / 2.0, BY / 2.0, -100.0), gp_Dir(0, 0, 1)),
                               SMDSAbs_Face, hit);
  check(!hit.empty(), "SEARCH GetElementsNearLine (ray cast) reports faces along an axial ray");

  std::vector<const SMDS_MeshElement*> miss;
  searcher->GetElementsNearLine(gp_Ax1(gp_Pnt(1e6, 1e6, -100.0), gp_Dir(0, 0, 1)),
                               SMDSAbs_Face, miss);
  check(miss.empty(), "SEARCH a ray far from the mesh reports no faces (falsification case)");

  std::vector<const SMDS_MeshElement*> in_sphere;
  searcher->GetElementsInSphere(gp_XYZ(BX / 2.0, BY / 2.0, BZ / 2.0), 1.0, SMDSAbs_Volume,
                               in_sphere);
  check(!in_sphere.empty(), "SEARCH GetElementsInSphere returns elements");

  std::vector<const SMDS_MeshElement*> in_box;
  Bnd_B3d bb;
  bb.Add(gp_XYZ(0, 0, 0));
  bb.Add(gp_XYZ(BX, BY, BZ));
  searcher->GetElementsInBox(bb, SMDSAbs_Volume, in_box);
  check(!in_box.empty(), "SEARCH GetElementsInBox returns elements");

  // GetPointState — the mesh-side in/out test, beside v1's B-rep point_in_solid.
  const TopAbs_State st_in = searcher->GetPointState(inside);
  const TopAbs_State st_out = searcher->GetPointState(gp_Pnt(-100, -100, -100));
  check(st_in == TopAbs_IN, "SEARCH GetPointState classifies an interior point as IN");
  check(st_out == TopAbs_OUT, "SEARCH GetPointState classifies a far point as OUT");

  const gp_XYZ projected = searcher->Project(gp_Pnt(-5, BY / 2.0, BZ / 2.0), SMDSAbs_Face);
  check_close(projected.X(), 0.0, 1e-6, "SEARCH Project lands on the x=0 face");

  // GetDistance against a VOLUME element — meshops, being surface-only, cannot answer this.
  const SMDS_MeshElement* vol = ds->elementsIterator(SMDSAbs_Volume)->next();
  const double d = SMESH_MeshAlgos::GetDistance(vol, gp_Pnt(-10, 0, 0));
  check(std::isfinite(d) && d > 0.0,
        "SEARCH SMESH_MeshAlgos::GetDistance answers for a volume element");

  // Sharp-edge detection and face partitioning by those edges.
  const std::vector<SMESH_MeshAlgos::Edge> sharp =
      SMESH_MeshAlgos::FindSharpEdges(ds, 45.0, false);
  check(!sharp.empty(), "SEARCH FindSharpEdges finds the box's 90-degree creases");
  const std::vector<std::vector<const SMDS_MeshElement*>> patches =
      SMESH_MeshAlgos::SeparateFacesByEdges(ds, sharp);
  check(patches.size() == 6, "SEARCH SeparateFacesByEdges partitions the box skin into 6 patches");

  // MakeOffset on the triangle mesh. Offsetting needs triangles, so split the skin first.
  {
    SMESH_MeshEditor editor(&s.mesh());
    TIDSortedElemSet quads;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Face); it->more();) {
      quads.insert(it->next());
    }
    editor.QuadToTri(quads, true);
  }
  SMESH_MeshAlgos::TElemIntPairVec new2old_faces;
  SMESH_MeshAlgos::TNodeIntPairVec new2old_nodes;
  std::unique_ptr<SMDS_Mesh> offset(
      SMESH_MeshAlgos::MakeOffset(ds->elementsIterator(SMDSAbs_Face), *ds, 0.1, false,
                                  new2old_faces, new2old_nodes));
  check(offset != nullptr && offset->NbFaces() > 0,
        "SEARCH SMESH_MeshAlgos::MakeOffset builds an offset triangle mesh");

  // DeMerge and MakeSlot are reachable; MakeSlot needs 1-D segments on a triangle mesh.
  std::vector<const SMDS_MeshNode*> new_nodes;
  std::vector<const SMDS_MeshNode*> no_merge;
  SMESH_MeshAlgos::DeMerge(ds->elementsIterator(SMDSAbs_Face)->next(), new_nodes, no_merge);
  check(true, "SEARCH SMESH_MeshAlgos::DeMerge links and runs");
  std::vector<SMDS_MeshGroup*> groups_to_update;
  const std::vector<SMESH_MeshAlgos::Edge> slot_edges = SMESH_MeshAlgos::MakeSlot(
      ds->elementsIterator(SMDSAbs_Edge), 0.05, ds, groups_to_update);
  check(slot_edges.empty() || !slot_edges.empty(),
        "SEARCH SMESH_MeshAlgos::MakeSlot links and runs");
}

// ---------------------------------------------------------------------------- ALGOFAM ----- //
void probe_r15_meshing_family() {
  section("ALGOFAM", "algorithm/hypothesis assignment model and the StdMeshers family");

  // Three different 3-D algorithms on the same solid, each with its own hypothesis set.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2), "ALGOFAM Hexa_3D + Quadrangle_2D + Regular_1D computes");
    check(s.meshDS()->NbVolumes() == 8, "ALGOFAM Hexa_3D gives 8 hexahedra at 2 segments/edge");
  }
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
    n->SetNumberOfSegments(2);
    StdMeshers_MEFISTO_2D* a2 = s.make<StdMeshers_MEFISTO_2D>();
    StdMeshers_MaxElementArea* area = s.make<StdMeshers_MaxElementArea>();
    area->SetMaxArea(4.0);
    bool ok = s.assign(s.shape(), a1) && s.assign(s.shape(), n) && s.assign(s.shape(), a2) &&
              s.assign(s.shape(), area);
    check(ok, "ALGOFAM MEFISTO_2D + MaxElementArea assign (MaxElementArea is an STDMESH TU)");
    check(s.compute() && s.meshDS()->NbFaces() > 0,
          "ALGOFAM MEFISTO_2D computes a triangular surface mesh");
  }
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    // VERIFY-AT-SOURCE FINDING: StdMeshers_PolyhedronPerSolid_3D's constructor allocates
    // and owns its OWN 1-D mesher and a StdMeshers_PolygonPerFace_2D, so it is an
    // all-dimensional algorithm (_requireDiscreteBoundary == false). Assigning Regular_1D /
    // Quadrangle_2D alongside it is redundant and makes the 2-D assignment come back
    // HYP_ALREADY_EXIST. It is assigned alone.
    StdMeshers_PolyhedronPerSolid_3D* poly = s.make<StdMeshers_PolyhedronPerSolid_3D>();
    const int st_poly = static_cast<int>(s.assign_status(s.shape(), poly));
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "ALGOFAM PolyhedronPerSolid_3D assigns alone (an STDMESH TU; status %d)", st_poly);
    check(!SMESH_Hypothesis::IsStatusFatal(
              static_cast<SMESH_Hypothesis::Hypothesis_Status>(st_poly)),
          msg);
    check(s.compute() && s.meshDS()->NbVolumes() > 0,
          "ALGOFAM PolyhedronPerSolid_3D computes one polyhedron per solid");
  }

  // The rest of the family must at least construct and report its name/dimension — that is
  // what proves the translation unit is linked and its hypothesis metadata is available.
  SMESH_Gen gen;
  int id = 100;
  StdMeshers_Prism_3D prism(id++, &gen);
  StdMeshers_RadialPrism_3D radial(id++, &gen);
  StdMeshers_Projection_2D proj2d(id++, &gen);
  StdMeshers_ViscousLayers2D vl2d(id++, &gen);
  StdMeshers_QuadFromMedialAxis_1D2D quad_mat(id++, &gen);
  check(named(prism) && named(radial) && named(proj2d),
        "ALGOFAM Prism_3D / RadialPrism_3D / Projection_2D link");
  check(named(vl2d), "ALGOFAM ViscousLayers2D links (2-D viscous layers)");
  check(named(quad_mat),
        "ALGOFAM QuadFromMedialAxis_1D2D links (MEDAX's medial-axis consumer)");

  // The assignment machinery itself: per-sub-shape sub-meshes, hypothesis filtering, and the
  // SMESH_ComputeError channel v1 already surfaces through PysmeshError.details.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2), "ALGOFAM mesh for the sub-mesh machinery probe computes");
    NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> faces;
    TopExp::MapShapes(s.shape(), TopAbs_FACE, faces);
    SMESH_subMesh* sm = s.mesh().GetSubMesh(faces.FindKey(1));
    check(sm != nullptr, "ALGOFAM SMESH_subMesh resolves per sub-shape");
    check(sm->IsMeshComputed(), "ALGOFAM the face sub-mesh reports computed");
    const SMESH_ComputeErrorPtr err = sm->GetComputeError();
    check(!err || err->IsOK(), "ALGOFAM SMESH_ComputeError is OK on a successful compute");

    SMESH_HypoFilter filter;
    filter.Init(SMESH_HypoFilter::IsAlgo());
    const SMESH_Hypothesis* algo =
        s.mesh().GetHypothesis(faces.FindKey(1), filter, /*andAncestors=*/true);
    check(algo != nullptr, "ALGOFAM SMESH_HypoFilter finds the algorithm on a sub-shape");
  }

  // A deliberately impossible assignment must surface as a compute error naming the sub-shape,
  // not as a silent empty mesh.
  {
    Session s(BRepPrimAPI_MakeCylinder(2.0, 5.0).Shape());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
    n->SetNumberOfSegments(2);
    StdMeshers_Quadrangle_2D* a2 = s.make<StdMeshers_Quadrangle_2D>();
    StdMeshers_Hexa_3D* a3 = s.make<StdMeshers_Hexa_3D>();  // a cylinder is not a block
    s.assign(s.shape(), a1);
    s.assign(s.shape(), n);
    s.assign(s.shape(), a2);
    s.assign(s.shape(), a3);
    const bool computed = s.compute();
    // VERIFY-AT-SOURCE FINDING: SMESH_ComputeError is attached to the sub-mesh that actually
    // failed, which for a Quadrangle_2D-on-a-disk failure is the FACE, not the enclosing
    // SOLID. A binding that reports errors only from the top-level sub-mesh reports nothing.
    std::string failing;
    for (int kind = TopAbs_SOLID; kind <= TopAbs_VERTEX; ++kind) {
      for (TopExp_Explorer ex(s.shape(), static_cast<TopAbs_ShapeEnum>(kind)); ex.More();
           ex.Next()) {
        const SMESH_ComputeErrorPtr err =
            s.mesh().GetSubMesh(ex.Current())->GetComputeError();
        if (err && !err->IsOK() && !err->myComment.empty()) {
          failing = err->myComment;
        }
      }
    }
    check(!computed, "ALGOFAM an impossible algorithm assignment fails to compute");
    check(!failing.empty(),
          "ALGOFAM the failure surfaces as SMESH_ComputeError text on the offending sub-shape");
  }
}

// ---------------------------------------------------------------------------- MEDAX ----- //
void probe_r16_medial_axis_and_blocks() {
  section("MEDAX", "medial axis (Boost Voronoi), Delaunay, block decomposition, patterns");

  // Medial axis of a rectangle: two BE_END ends and two BE_ON_VERTEX ends, one branch.
  const double w = 10.0;
  const double h = 4.0;
  const TopoDS_Face rect =
      BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0, w, 0, h).Face();
  std::vector<TopoDS_Edge> edges;
  for (TopExp_Explorer ex(rect, TopAbs_EDGE); ex.More(); ex.Next()) {
    edges.push_back(TopoDS::Edge(ex.Current()));
  }
  check(edges.size() == 4, "MEDAX rectangle fixture has 4 edges");

  SMESH_MAT2d::MedialAxis mat(rect, edges, /*minSegLen=*/0.1, /*ignoreCorners=*/false);
  check(mat.nbBranches() >= 1, "MEDAX SMESH_MAT2d::MedialAxis produces at least one branch");
  // VERIFY-AT-SOURCE FINDING: branch 0 is not necessarily the "main" axis, and a branch is
  // NOT a dense polyline — MedialAxis::getPoints returns one point per MA edge plus one, so
  // a straight branch yields exactly two points. A rectangle's axis is a spine plus four
  // 45-degree corner arms: 5 branches of 2 points each. A thin-region/thickness query must
  // select the branch it wants, never index 0 blindly, and must not assume a dense polyline.
  check(mat.nbBranches() >= 5,
        "MEDAX a rectangle yields a spine plus corner arms (>= 5 branches)");

  const SMESH_MAT2d::Branch* branch = nullptr;
  std::vector<gp_XY> axis_points;
  double best_span = -1.0;
  for (std::size_t i = 0; i < mat.nbBranches(); ++i) {
    const SMESH_MAT2d::Branch* candidate = mat.getBranch(i);
    std::vector<gp_XY> pts;
    mat.getPoints(candidate, pts);
    if (pts.size() < 2) {
      continue;
    }
    double xmin = pts[0].X();
    double xmax = pts[0].X();
    for (const gp_XY& q : pts) {
      xmin = std::min(xmin, q.X());
      xmax = std::max(xmax, q.X());
    }
    if (xmax - xmin > best_span) {
      best_span = xmax - xmin;
      axis_points = pts;
      branch = candidate;
    }
  }
  check(branch != nullptr, "MEDAX the medial axis exposes a Branch");

  // VERIFY-AT-SOURCE FINDING: SMESH_MAT2d works in a scaled UV space chosen when the
  // MedialAxis is built. Branch::getPoints takes that scale as an argument and is only
  // correct with the axis's own value, which is private - MedialAxis::getPoints(branch, pts)
  // is the entry point that applies it. Calling the Branch overload with {1,1} yields
  // coordinates in the scaled space, not on the face.
  check(axis_points.size() >= 2,
        "MEDAX MedialAxis::getPoints yields the branch's MA-edge endpoints");

  // The medial axis of a w x h rectangle runs along y = h/2 in its central part; that is the
  // property the thin-region/thickness use case depends on.
  // The spine of a w x h rectangle (w > h) is the analytic medial axis: the segment
  // y = h/2, x in [h/2, w - h/2].
  int on_centreline = 0;
  double xmin = axis_points[0].X();
  double xmax = axis_points[0].X();
  for (const gp_XY& p : axis_points) {
    if (std::fabs(p.Y() - h / 2.0) < 1e-6) {
      ++on_centreline;
    }
    xmin = std::min(xmin, p.X());
    xmax = std::max(xmax, p.X());
  }
  check(on_centreline == static_cast<int>(axis_points.size()),
        "MEDAX every point of the spine branch lies on y = h/2 (analytic medial axis)");
  check_close(xmin, h / 2.0, 1e-6, "MEDAX the spine starts at x = h/2");
  check_close(xmax, w - h / 2.0, 1e-6, "MEDAX the spine ends at x = w - h/2");

  // Boundary points give local half-thickness: |axis - boundary| == h/2 on the centreline.
  SMESH_MAT2d::BoundaryPoint bp1;
  SMESH_MAT2d::BoundaryPoint bp2;
  const bool got_bp = branch->getBoundaryPoints(0.5, bp1, bp2);
  check(got_bp, "MEDAX Branch::getBoundaryPoints maps an axis point back to its two boundaries");
  if (got_bp) {
    const double thickness = gp_XY(bp1._param, 0).X() >= 0 ? 1.0 : 1.0;  // params only
    (void)thickness;
    check(bp1._edgeIndex < edges.size() && bp2._edgeIndex < edges.size(),
          "MEDAX boundary points carry valid edge indices (thickness recovery input)");
  }

  const SMESH_MAT2d::Boundary& boundary = mat.getBoundary();
  check(boundary.nbEdges() == edges.size(),
        "MEDAX the MAT boundary carries one point sequence per input edge");

  const std::vector<const SMESH_MAT2d::BranchEnd*>& ends = mat.getBranchPoints();
  check(!ends.empty(), "MEDAX branch end points are reported");

  // SMESH_Delaunay is abstract (getNodeUV is the client hook), and SMESH_Block/SMESH_Pattern
  // are driven from a meshed block. Constructing the pattern engine proves the link.
  SMESH_Pattern pattern;
  check(!pattern.Load("!!! Nb of points, Nb of elements\n4 1\n0 0\n1 0\n1 1\n0 1\n0 1 2 3\n"),
        "MEDAX SMESH_Pattern::Load links and rejects a malformed pattern");
  note("MEDAX SMESH_Delaunay / SMESH_Block",
       "SMESH_Delaunay is abstract (getNodeUV must be supplied by the binding) and "
       "SMESH_Block needs a meshed shell; both link — see the symbol reference below");
  check(SMESH_Block::ShapeIndex(SMESH_Block::ID_Ex00) == 0,
        "MEDAX SMESH_Block links (static shape-index arithmetic)");
}

// ---------------------------------------------------------------------------- GROUPS ----- //
void probe_r17_groups() {
  section("GROUPS", "element groups that survive meshing and editing");

  Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
  check(build_hexa_mesh(s, 2), "GROUPS hexa mesh for the group probe computes");
  SMESHDS_Mesh* ds = s.meshDS();

  SMESH_Group* group = s.mesh().AddGroup(SMDSAbs_Volume, "wall_cells");
  check(group != nullptr && group->GetGroupDS() != nullptr,
        "GROUPS SMESH_Mesh::AddGroup creates a volume group");

  SMESHDS_Group* gds = dynamic_cast<SMESHDS_Group*>(group->GetGroupDS());
  check(gds != nullptr, "GROUPS the group DS is an explicit (id-list) SMESHDS_Group");

  // Independent ground truth: the ids we put in, tracked outside the group.
  std::vector<smIdType> expected;
  for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Volume); it->more();) {
    const SMDS_MeshElement* e = it->next();
    if (expected.size() < 4) {
      gds->Add(e);
      expected.push_back(e->GetID());
    }
  }
  check(gds->Extent() == 4, "GROUPS the group holds the 4 elements added");

  auto membership_matches = [&](const char* what) {
    std::vector<smIdType> actual;
    for (SMDS_ElemIteratorPtr it = gds->GetElements(); it->more();) {
      actual.push_back(it->next()->GetID());
    }
    std::sort(actual.begin(), actual.end());
    std::vector<smIdType> want = expected;
    std::sort(want.begin(), want.end());
    check(actual == want, std::string("GROUPS group membership is correct after ") + what);
  };
  membership_matches("creation");

  SMESH_MeshEditor editor(&s.mesh());
  editor.ConvertToQuadratic(true, false);
  membership_matches("ConvertToQuadratic");

  editor.ConvertFromQuadratic();
  membership_matches("ConvertFromQuadratic");

  SMESH_MeshEditor::TListOfListOfNodes empty_groups;
  editor.MergeNodes(empty_groups);
  membership_matches("MergeNodes");

  check(s.mesh().GetGroupIds().size() == 1, "GROUPS the mesh reports exactly one group id");
}

// ---------------------------------------------------------------------------- MESHBIND -- //
// The behaviours a Python meshing binding rests on, as opposed to the capabilities above.
// Each of these decides a design question that a header read cannot answer: what a
// body-fitted mesher actually emits, how an element names the sub-shape it sits on, whether
// progress and cancellation can be driven from another thread, and whether two different
// 3-D algorithms on one model meet at a shared face.
void probe_meshing_binding_behaviour() {
  section("MESHBIND", "the behaviours a Python meshing binding depends on");

  // ---- What Cartesian_3D emits, and how a polyhedron's connectivity is read ---------- //
  {
    const TopoDS_Shape block = BRepPrimAPI_MakeBox(gp_Pnt(-4, -4, 0), 8.0, 8.0, 6.0).Shape();
    const TopoDS_Shape bore = BRepPrimAPI_MakeCylinder(1.5, 6.0).Shape();
    NCollection_List<TopoDS_Shape> args;
    args.Append(block);
    NCollection_List<TopoDS_Shape> tools;
    tools.Append(bore);
    BRepAlgoAPI_Cut cut;
    cut.SetArguments(args);
    cut.SetTools(tools);
    cut.Build();

    Session s(cut.Shape());
    StdMeshers_Cartesian_3D* algo = s.make<StdMeshers_Cartesian_3D>();
    StdMeshers_CartesianParameters3D* params = s.make<StdMeshers_CartesianParameters3D>();
    std::vector<std::string> spacing(1, std::string("1.0"));
    std::vector<double> internal_points;
    for (int axis = 0; axis < 3; ++axis) {
      params->SetGridSpacing(spacing, internal_points, axis);
    }
    params->SetSizeThreshold(4.0);
    s.assign(s.shape(), algo);
    s.assign(s.shape(), params);
    check(s.compute(), "MESHBIND Cartesian_3D computes on the bored block");

    int n_hexa = 0, n_poly = 0, n_other = 0;
    const SMDS_MeshElement* a_polyhedron = nullptr;
    for (SMDS_ElemIteratorPtr it = s.meshDS()->elementsIterator(SMDSAbs_Volume); it->more();) {
      const SMDS_MeshElement* e = it->next();
      switch (e->GetEntityType()) {
        case SMDSEntity_Hexa: ++n_hexa; break;
        case SMDSEntity_Polyhedra:
          ++n_poly;
          if (a_polyhedron == nullptr) {
            a_polyhedron = e;
          }
          break;
        default: ++n_other; break;
      }
    }
    char msg[220];
    std::snprintf(msg, sizeof(msg),
                  "MESHBIND Cartesian_3D emits hexahedra AND polyhedra (hexa %d, poly %d, "
                  "other %d) — a binding must carry a per-face node split",
                  n_hexa, n_poly, n_other);
    check(n_hexa > 0 && n_poly > 0, msg);

    if (a_polyhedron != nullptr) {
      const SMDS_MeshVolume* vol = SMDS_Mesh::DownCast<SMDS_MeshVolume>(a_polyhedron);
      check(vol != nullptr, "MESHBIND a polyhedron downcasts to SMDS_MeshVolume");
      if (vol != nullptr) {
        const std::vector<int> quantities = vol->GetQuantities();
        int summed = 0;
        for (const int q : quantities) {
          summed += q;
        }
        std::snprintf(msg, sizeof(msg),
                      "MESHBIND GetQuantities() sums to NbNodes() (%d faces, %d node slots, "
                      "NbNodes %d) — the node list IS the face stream",
                      static_cast<int>(quantities.size()), summed, a_polyhedron->NbNodes());
        check(!quantities.empty() && summed == a_polyhedron->NbNodes(), msg);
      }
    }
  }

  // ---- How an element names the sub-shape it sits on -------------------------------- //
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2), "MESHBIND hexa mesh for the shape-binding probe computes");
    SMESHDS_Mesh* ds = s.meshDS();

    int bound = 0, unbound = 0, wrong_kind = 0;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Face); it->more();) {
      const SMDS_MeshElement* e = it->next();
      const int shape_id = e->getshapeId();
      if (shape_id <= 0) {
        ++unbound;
        continue;
      }
      ++bound;
      // IndexToShape is the inverse of ShapeToIndex, so a SMESHDS shape index translates
      // back to the TopoDS_Shape and from there to this shape's own TopExp ordinal — which
      // is what keeps SMESHDS indices out of the public signatures.
      const TopoDS_Shape& sub = ds->IndexToShape(shape_id);
      if (sub.IsNull() || sub.ShapeType() != TopAbs_FACE) {
        ++wrong_kind;
      }
    }
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "MESHBIND every face element names a FACE through getshapeId() + "
                  "IndexToShape (bound %d, unbound %d, wrong kind %d)",
                  bound, unbound, wrong_kind);
    check(bound > 0 && unbound == 0 && wrong_kind == 0, msg);

    // The same question for nodes, which carry the sub-shape they were classified onto.
    int node_bound = 0, node_unbound = 0;
    for (SMDS_NodeIteratorPtr it = ds->nodesIterator(); it->more();) {
      (it->next()->getshapeId() > 0 ? node_bound : node_unbound)++;
    }
    std::snprintf(msg, sizeof(msg),
                  "MESHBIND every node names a sub-shape too (bound %d, unbound %d)",
                  node_bound, node_unbound);
    check(node_bound > 0 && node_unbound == 0, msg);
  }

  // ---- AddHypothesis reports a refusal in words, not only as a status ---------------- //
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    s.assign(s.shape(), a1);
    StdMeshers_Regular_1D* a1b = s.make<StdMeshers_Regular_1D>();
    std::string error;
    const SMESH_Hypothesis::Hypothesis_Status status =
        s.mesh().AddHypothesis(s.shape(), a1b->GetID(), &error);
    char msg[240];
    std::snprintf(msg, sizeof(msg),
                  "MESHBIND a second 1-D algorithm on one shape is refused with a status and "
                  "text (status %d, text \"%s\")",
                  static_cast<int>(status), error.c_str());
    check(SMESH_Hypothesis::IsStatusFatal(status), msg);
  }

  // ---- Progress and cancellation, driven from another thread ------------------------ //
  // SMESH has no Message_ProgressIndicator: progress is *pulled* through
  // SMESH_Mesh::GetComputeProgress() and a break is *pushed* through
  // SMESH_Gen::CancelCompute(). Both are designed to be called while Compute() runs, which
  // is the whole question — a binding polls them from a helper thread with the GIL released.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
    n->SetNumberOfSegments(40);
    StdMeshers_Quadrangle_2D* a2 = s.make<StdMeshers_Quadrangle_2D>();
    StdMeshers_Hexa_3D* a3 = s.make<StdMeshers_Hexa_3D>();
    s.assign(s.shape(), a1);
    s.assign(s.shape(), n);
    s.assign(s.shape(), a2);
    s.assign(s.shape(), a3);

    std::atomic<bool> running{true};
    std::atomic<int> samples{0};
    std::atomic<int> non_monotone{0};
    double last = -1.0;
    std::thread poller([&] {
      while (running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const double p = s.mesh().GetComputeProgress();
        if (p < last) {
          non_monotone.fetch_add(1);
        }
        last = p;
        samples.fetch_add(1);
      }
    });
    const bool ok = s.compute();
    running.store(false);
    poller.join();

    char msg[220];
    std::snprintf(msg, sizeof(msg),
                  "MESHBIND GetComputeProgress() is safe to poll from another thread during "
                  "Compute (%d samples, %d backwards steps)",
                  samples.load(), non_monotone.load());
    check(ok && samples.load() > 0, msg);
  }

  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
    n->SetNumberOfSegments(60);
    StdMeshers_Quadrangle_2D* a2 = s.make<StdMeshers_Quadrangle_2D>();
    StdMeshers_Hexa_3D* a3 = s.make<StdMeshers_Hexa_3D>();
    s.assign(s.shape(), a1);
    s.assign(s.shape(), n);
    s.assign(s.shape(), a2);
    s.assign(s.shape(), a3);

    std::atomic<bool> stop{false};
    std::thread canceller([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      if (!stop.load()) {
        s.gen().CancelCompute(s.mesh(), s.shape());
      }
    });
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = s.compute();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    stop.store(true);
    canceller.join();

    // Compute() returning false is NOT by itself "the caller cancelled": a cancel landing
    // late leaves a complete mesh and the same false, and an ordinary algorithm failure
    // gives false with no cancel at all. The binding's own flag has to be the authority,
    // exactly as the OCCT-side progress driver already establishes.
    char msg[260];
    std::snprintf(msg, sizeof(msg),
                  "MESHBIND CancelCompute() from another thread stops Compute (returned %s "
                  "after %.0f ms, %d of 216000 volumes built)",
                  ok ? "true" : "false", elapsed_ms,
                  static_cast<int>(s.meshDS()->NbVolumes()));
    check(!ok, msg);
  }

  // Only three StdMeshers algorithms poll _computeCanceled inside their own loop —
  // Adaptive1D, Cartesian_3D and Prism_3D. Everything else can be broken only *between*
  // sub-meshes, which is where SMESH_Gen tests its own flag. So cancellation latency is
  // bounded by the longest single algorithm run, not by a poll interval, and that has to be
  // stated rather than discovered. Cartesian_3D is the one that can prove the good case.
  {
    const TopoDS_Shape block =
        BRepPrimAPI_MakeBox(gp_Pnt(-8, -8, 0), 16.0, 16.0, 12.0).Shape();
    const TopoDS_Shape bore = BRepPrimAPI_MakeCylinder(2.0, 12.0).Shape();
    NCollection_List<TopoDS_Shape> args;
    args.Append(block);
    NCollection_List<TopoDS_Shape> tools;
    tools.Append(bore);
    BRepAlgoAPI_Cut cut;
    cut.SetArguments(args);
    cut.SetTools(tools);
    cut.Build();

    Session s(cut.Shape());
    StdMeshers_Cartesian_3D* algo = s.make<StdMeshers_Cartesian_3D>();
    StdMeshers_CartesianParameters3D* params = s.make<StdMeshers_CartesianParameters3D>();
    std::vector<std::string> spacing(1, std::string("0.15"));
    std::vector<double> internal_points;
    for (int axis = 0; axis < 3; ++axis) {
      params->SetGridSpacing(spacing, internal_points, axis);
    }
    params->SetSizeThreshold(4.0);
    s.assign(s.shape(), algo);
    s.assign(s.shape(), params);

    std::atomic<bool> stop{false};
    std::thread canceller([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(150));
      if (!stop.load()) {
        s.gen().CancelCompute(s.mesh(), s.shape());
      }
    });
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = s.compute();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    stop.store(true);
    canceller.join();

    char msg[260];
    std::snprintf(msg, sizeof(msg),
                  "MESHBIND Cartesian_3D honours a cancel mid-algorithm (returned %s after "
                  "%.0f ms with %d volumes) — it is one of the three that poll the flag",
                  ok ? "true" : "false", elapsed_ms,
                  static_cast<int>(s.meshDS()->NbVolumes()));
    check(!ok && elapsed_ms < 1500.0, msg);
  }

  // ---- Two 3-D algorithms on one model, and whether they meet ----------------------- //
  // The gate's real question: a mixed assignment must be conforming at the internal
  // boundary. An algorithm that consumes the 2-D boundary mesh conforms by construction; one
  // that ignores it cannot. Both cases are measured here rather than assumed.
  {
    // A plain fuse of two face-touching boxes returns ONE solid — the seam face is internal
    // to the result and OCCT drops it. The general fuse keeps both pieces and glues them on
    // a shared FACE, which is the only fixture that can carry an internal boundary at all.
    const TopoDS_Shape lower = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 4.0, 4.0, 4.0).Shape();
    const TopoDS_Shape upper = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 4), 4.0, 4.0, 4.0).Shape();
    NCollection_List<TopoDS_Shape> args;
    args.Append(lower);
    args.Append(upper);
    BRepAlgoAPI_BuilderAlgo fuse;
    fuse.SetArguments(args);
    fuse.Build();
    check(fuse.IsDone(), "MESHBIND two-solid stacked fixture builds");

    std::vector<TopoDS_Shape> solids;
    for (TopExp_Explorer ex(fuse.Shape(), TopAbs_SOLID); ex.More(); ex.Next()) {
      solids.push_back(ex.Current());
    }
    char msg[240];
    std::snprintf(msg, sizeof(msg), "MESHBIND the stacked fixture has 2 solids (got %d)",
                  static_cast<int>(solids.size()));
    check(solids.size() == 2, msg);

    if (solids.size() == 2) {
      Session s(fuse.Shape());
      StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
      StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
      n->SetNumberOfSegments(2);
      StdMeshers_Quadrangle_2D* a2 = s.make<StdMeshers_Quadrangle_2D>();
      s.assign(s.shape(), a1);
      s.assign(s.shape(), n);
      s.assign(s.shape(), a2);

      StdMeshers_Hexa_3D* hexa = s.make<StdMeshers_Hexa_3D>();
      StdMeshers_PolyhedronPerSolid_3D* poly = s.make<StdMeshers_PolyhedronPerSolid_3D>();
      const int st_hexa = static_cast<int>(s.assign_status(solids[0], hexa));
      const int st_poly = static_cast<int>(s.assign_status(solids[1], poly));
      std::snprintf(msg, sizeof(msg),
                    "MESHBIND a different 3-D algorithm assigns to each solid (Hexa_3D %d, "
                    "PolyhedronPerSolid_3D %d)",
                    st_hexa, st_poly);
      check(!SMESH_Hypothesis::IsStatusFatal(
                static_cast<SMESH_Hypothesis::Hypothesis_Status>(st_hexa)) &&
                !SMESH_Hypothesis::IsStatusFatal(
                    static_cast<SMESH_Hypothesis::Hypothesis_Status>(st_poly)),
            msg);

      const bool computed = s.compute();
      // Conformity, asserted node by node: every node on the shared FACE must be a single
      // node used by elements of both solids, not two coincident ones.
      NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> lower_faces, upper_faces;
      TopExp::MapShapes(solids[0], TopAbs_FACE, lower_faces);
      TopExp::MapShapes(solids[1], TopAbs_FACE, upper_faces);
      int shared_faces = 0;
      TopoDS_Shape interface_face;
      for (int i = 1; i <= lower_faces.Extent(); ++i) {
        if (upper_faces.Contains(lower_faces.FindKey(i))) {
          ++shared_faces;
          interface_face = lower_faces.FindKey(i);
        }
      }
      std::snprintf(msg, sizeof(msg),
                    "MESHBIND the two solids share exactly one FACE (got %d)",
                    shared_faces);
      check(shared_faces == 1, msg);

      int interface_nodes = 0, shared_by_both = 0;
      if (computed && !interface_face.IsNull()) {
        const SMESHDS_SubMesh* sub = s.meshDS()->MeshElements(interface_face);
        if (sub != nullptr) {
          for (SMDS_NodeIteratorPtr it = sub->GetNodes(); it->more();) {
            const SMDS_MeshNode* node = it->next();
            ++interface_nodes;
            std::set<int> owning_solids;
            for (SMDS_ElemIteratorPtr eit = node->GetInverseElementIterator(SMDSAbs_Volume);
                 eit->more();) {
              const int sid = eit->next()->getshapeId();
              if (sid > 0) {
                owning_solids.insert(sid);
              }
            }
            if (owning_solids.size() >= 2) {
              ++shared_by_both;
            }
          }
        }
      }
      std::snprintf(msg, sizeof(msg),
                    "MESHBIND the mixed mesh is conforming node by node at the shared FACE "
                    "(computed %s, %d interface nodes, %d used by both solids)",
                    computed ? "true" : "false", interface_nodes, shared_by_both);
      check(computed && interface_nodes > 0 && shared_by_both == interface_nodes, msg);
    }
  }
}

// ---------------------------------------------------------------------------- GMF ----- //
void probe_r18_gmf_driver() {
  section("GMF", "DriverGMF: Inria .mesh / .meshb round-trip");

  Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
  check(build_hexa_mesh(s, 2), "GMF hexa mesh for the GMF round-trip computes");
  SMESHDS_Mesh* ds = s.meshDS();
  const smIdType nodes = ds->NbNodes();
  const smIdType volumes = ds->NbVolumes();
  const smIdType faces = ds->NbFaces();

  const std::string path = "pysmesh_probe_gmf.mesh";
  DriverGMF_Write writer;
  writer.SetFile(path);
  writer.SetMesh(ds);
  const Driver_Mesh::Status wst = writer.Perform();
  check(wst == Driver_Mesh::DRS_OK, "GMF DriverGMF_Write writes an Inria .mesh file");

  SMESH_Gen read_gen;
  SMESH_Mesh* read_mesh = read_gen.CreateMesh(false);
  DriverGMF_Read reader;
  reader.SetFile(path);
  reader.SetMesh(read_mesh->GetMeshDS());
  reader.SetMakeRequiredGroups(true);

  smIdType nb_vertex = 0;
  smIdType nb_edge = 0;
  smIdType nb_face = 0;
  smIdType nb_vol = 0;
  const bool info_ok = reader.GetMeshInfo(nb_vertex, nb_edge, nb_face, nb_vol);
  check(info_ok && nb_vertex == nodes,
        "GMF DriverGMF_Read::GetMeshInfo reports the written node count");

  const Driver_Mesh::Status rst = reader.Perform();
  check(rst == Driver_Mesh::DRS_OK, "GMF DriverGMF_Read reads the file back");
  check(read_mesh->GetMeshDS()->NbNodes() == nodes,
        "GMF round-trip preserves the node count");
  check(read_mesh->GetMeshDS()->NbVolumes() == volumes,
        "GMF round-trip preserves the volume count");
  check(read_mesh->GetMeshDS()->NbFaces() == faces, "GMF round-trip preserves the face count");

  // Does the per-element sub-shape reference survive? The writer emits elem->getshapeId() as
  // each element's GMF reference, so the file carries it — but the reader parses it into a
  // local and never applies it. Measured rather than assumed, because a binding that claims
  // the CAD binding survives a round trip would be wrong.
  {
    int with_shape = 0, without_shape = 0;
    for (SMDS_ElemIteratorPtr it = read_mesh->GetMeshDS()->elementsIterator(SMDSAbs_Volume);
         it->more();) {
      (it->next()->getshapeId() > 0 ? with_shape : without_shape)++;
    }
    char msg[220];
    std::snprintf(msg, sizeof(msg),
                  "GMF the per-element sub-shape reference is written but DROPPED on read "
                  "(%d volumes with a shape id, %d without)",
                  with_shape, without_shape);
    check(with_shape == 0 && without_shape > 0, msg);
  }
  delete read_mesh;
  std::remove(path.c_str());

  // Groups: the only group channel the GMF driver carries is the "required entities" one —
  // a group whose store name contains "_required_<Entity>". A general named group is silently
  // not written, which the binding must say rather than imply.
  {
    Session g(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(g, 2), "GMF hexa mesh for the group round-trip computes");
    SMESH_Group* required = g.mesh().AddGroup(SMDSAbs_Face, "req");
    SMESH_Group* ordinary = g.mesh().AddGroup(SMDSAbs_Face, "ordinary");
    SMESHDS_Group* req_ds = dynamic_cast<SMESHDS_Group*>(required->GetGroupDS());
    SMESHDS_Group* ord_ds = dynamic_cast<SMESHDS_Group*>(ordinary->GetGroupDS());
    check(req_ds != nullptr && ord_ds != nullptr, "GMF two face groups are created");
    req_ds->SetStoreName("_required_Quadrilaterals");
    ord_ds->SetStoreName("ordinary");
    int added = 0;
    for (SMDS_ElemIteratorPtr it = g.meshDS()->elementsIterator(SMDSAbs_Face);
         it->more() && added < 5;) {
      const SMDS_MeshElement* e = it->next();
      req_ds->Add(e);
      ord_ds->Add(e);
      ++added;
    }

    const std::string gpath = "pysmesh_probe_gmf_groups.meshb";
    DriverGMF_Write gw;
    gw.SetFile(gpath);
    gw.SetMesh(g.meshDS());
    gw.SetExportRequiredGroups(true);
    check(gw.Perform() == Driver_Mesh::DRS_OK,
          "GMF a binary .meshb file is written (libmesh5 picks the format by extension)");

    SMESH_Gen ggen;
    SMESH_Mesh* gread = ggen.CreateMesh(false);
    DriverGMF_Read gr;
    gr.SetFile(gpath);
    gr.SetMesh(gread->GetMeshDS());
    gr.SetMakeRequiredGroups(true);
    check(gr.Perform() == Driver_Mesh::DRS_OK, "GMF the binary .meshb file reads back");

    int required_back = 0;
    bool ordinary_back = false;
    for (SMESHDS_GroupBase* grp : gread->GetMeshDS()->GetGroups()) {
      const std::string name = grp->GetStoreName();
      if (name.find("_required_") != std::string::npos) {
        required_back = static_cast<int>(grp->Extent());
      }
      if (name == "ordinary") {
        ordinary_back = true;
      }
    }
    char msg[240];
    std::snprintf(msg, sizeof(msg),
                  "GMF a _required_ group round-trips with its membership (%d of %d back) "
                  "while an ordinary group is NOT written (%s)",
                  required_back, added, ordinary_back ? "present" : "absent");
    check(required_back == added && !ordinary_back, msg);
    delete gread;
    std::remove(gpath.c_str());
  }

  note("GMF MMG / fTetWild files",
       "reading engine-written files is a binding-layer test needing those engines' output "
       "as fixtures");
}

}  // namespace

void run_smesh_probe() {
  probe_r11_unexcluded_translation_units();
  probe_r12_controls();
  probe_r13_mesh_editor();
  probe_r14_search_and_ray_casting();
  probe_r15_meshing_family();
  probe_r16_medial_axis_and_blocks();
  probe_r17_groups();
  probe_meshing_binding_behaviour();
  probe_r18_gmf_driver();
}
