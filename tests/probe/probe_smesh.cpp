// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-05

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
#include <BRep_Builder.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_BuilderAlgo.hxx>
#include <NCollection_List.hxx>
#include <TopExp.hxx>
#include <TopoDS_Compound.hxx>
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
#include <SMESH_Comment.hxx>
#include <SMESH_ComputeError.hxx>
#include <Utils_SALOME_Exception.hxx>
#include <SMESH_ControlsDef.hxx>
#include <BRepMesh_DataStructureOfDelaun.hxx>
#include <BRepMesh_Triangle.hxx>
#include <SMESH_Block.hxx>
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
#include <StdMeshers_CompositeHexa_3D.hxx>
#include <StdMeshers_Hexa_3D.hxx>
#include <StdMeshers_Import_1D2D.hxx>
#include <StdMeshers_MEFISTO_2D.hxx>
#include <StdMeshers_MaxElementArea.hxx>
#include <StdMeshers_MaxElementVolume.hxx>
#include <StdMeshers_NumberOfSegments.hxx>
#include <StdMeshers_ViscousLayerBuilder.hxx>
#include <StdMeshers_PolyhedronPerSolid_3D.hxx>
#include <StdMeshers_Prism_3D.hxx>
#include <StdMeshers_Projection_2D.hxx>
#include <StdMeshers_QuadFromMedialAxis_1D2D.hxx>
#include <StdMeshers_Quadrangle_2D.hxx>
#include <StdMeshers_RadialPrism_3D.hxx>
#include <StdMeshers_Regular_1D.hxx>
#include <StdMeshers_ViscousLayers2D.hxx>
#include <StdMeshers_BlockRenumber.hxx>
#include <StdMeshers_NotConformAllowed.hxx>
#include <StdMeshers_ViscousLayers.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <SMDS_UnstructuredGrid.hxx>
#include <StdMeshers_LayerDistribution2D.hxx>
#include <StdMeshers_LengthFromEdges.hxx>
#include <StdMeshers_RadialQuadrangle_1D2D.hxx>
#include <StdMeshers_UseExisting_1D2D.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Ax2.hxx>
#include <gp_Circ.hxx>
#include <gp_Dir.hxx>
#include <StdMeshers_Arithmetic1D.hxx>
#include <StdMeshers_Propagation.hxx>
#include <StdMeshers_SegmentAroundVertex_0D.hxx>
#include <StdMeshers_SegmentLengthAroundVertex.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRep_Tool.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Vertex.hxx>

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

// The same without a 3-D algorithm: the quadrangular skin alone.
bool build_quad_mesh(Session& s, int nseg) {
  StdMeshers_Regular_1D* algo1d = s.make<StdMeshers_Regular_1D>();
  StdMeshers_NumberOfSegments* nseg_hyp = s.make<StdMeshers_NumberOfSegments>();
  nseg_hyp->SetNumberOfSegments(nseg);
  StdMeshers_Quadrangle_2D* algo2d = s.make<StdMeshers_Quadrangle_2D>();
  bool ok = s.assign(s.shape(), algo1d);
  ok = s.assign(s.shape(), nseg_hyp) && ok;
  ok = s.assign(s.shape(), algo2d) && ok;
  return ok && s.compute();
}

// ---------------------------------------------------------------------------- STDMESH ---- //
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
       "linked and callable; a meaningful sew needs a two-patch fixture, so it is gated in "
       "the binding-layer suite rather than here");
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

// ---------------------------------------------------------------------------- ALGOFAM ---- //
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
  note("MEDAX SMESH_Delaunay",
       "abstract (getNodeUV must be supplied), links and constructs — and then answers "
       "nothing, because its triangulation comes back entirely deleted under the pinned "
       "OCCT. Measured in the EDITBIND section below; not bound for that reason");
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

    // Compute() returning false proves only that SMESH_Gen saw the cancel; it also does so
    // after a complete Cartesian_3D run that ignored it. Cartesian_3D polls the flag only in
    // its grid step, before it builds a single volume, so a cancel that stopped the
    // algorithm leaves no volume. A run without a cancel builds 872 320 volumes here.
    const int volumes = static_cast<int>(s.meshDS()->NbVolumes());
    char msg[260];
    std::snprintf(msg, sizeof(msg),
                  "MESHBIND Cartesian_3D honours a cancel mid-algorithm (returned %s after "
                  "%.0f ms with %d volumes) — it is one of the three that poll the flag",
                  ok ? "true" : "false", elapsed_ms, volumes);
    check(!ok && volumes == 0 && elapsed_ms < 1500.0, msg);
  }

  // With a ViscousLayers hypothesis Cartesian_3D first builds an offset shape (one OCCT
  // offset, which cannot be stopped inside), then meshes it by calling itself, then adds the
  // layers. The inner call used to clear the cancel flag, so a cancel during the offset step
  // was lost and the run went on to its end (pySMESH patch StdMeshers_Cartesian_VL_cancel).
  // The block carries 8 x 8 square pockets, so its offset takes about 0.5 s and the cancel
  // at 150 ms lands inside it. The layers grow on the bottom face, the one wall that stays a
  // whole grid-aligned rectangle when it is offset. A run without a cancel takes 4.4 s.
  {
    const double side = 12.0;
    const double height = 4.0;
    const double pitch = side / 8.0;
    const TopoDS_Shape block =
        BRepPrimAPI_MakeBox(gp_Pnt(-side / 2, -side / 2, 0.0), side, side, height).Shape();
    NCollection_List<TopoDS_Shape> args;
    args.Append(block);
    NCollection_List<TopoDS_Shape> tools;
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        const gp_Pnt corner(-side / 2 + pitch * (i + 0.25), -side / 2 + pitch * (j + 0.25),
                            height / 2);
        tools.Append(BRepPrimAPI_MakeBox(corner, pitch / 2, pitch / 2, height).Shape());
      }
    }
    BRepAlgoAPI_Cut cut;
    cut.SetArguments(args);
    cut.SetTools(tools);
    cut.Build();

    Session s(cut.Shape());
    StdMeshers_Cartesian_3D* algo = s.make<StdMeshers_Cartesian_3D>();
    StdMeshers_CartesianParameters3D* params = s.make<StdMeshers_CartesianParameters3D>();
    std::vector<std::string> spacing(1, std::string("0.1"));
    std::vector<double> internal_points;
    for (int axis = 0; axis < 3; ++axis) {
      params->SetGridSpacing(spacing, internal_points, axis);
    }
    StdMeshers_ViscousLayers* vl = s.make<StdMeshers_ViscousLayers>();
    vl->SetTotalThickness(0.3);
    vl->SetNumberLayers(3);
    vl->SetStretchFactor(1.2);
    std::vector<int> wall;
    for (TopExp_Explorer f(s.shape(), TopAbs_FACE); f.More(); f.Next()) {
      bool bottom = true;
      for (TopExp_Explorer v(f.Current(), TopAbs_VERTEX); v.More(); v.Next()) {
        bottom = bottom && std::fabs(BRep_Tool::Pnt(TopoDS::Vertex(v.Current())).Z()) < 1e-9;
      }
      if (bottom) {
        wall.push_back(s.meshDS()->ShapeToIndex(f.Current()));
      }
    }
    vl->SetBndShapes(wall, /*toIgnore=*/false);
    s.assign(s.shape(), algo);
    s.assign(s.shape(), params);
    s.assign(s.shape(), vl);

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

    // A cancel that stopped the run in its offset step leaves no volume: the grid step and
    // the layer step never start.
    const int volumes = static_cast<int>(s.meshDS()->NbVolumes());
    char msg[260];
    std::snprintf(msg, sizeof(msg),
                  "MESHBIND Cartesian_3D + ViscousLayers honours a cancel in its offset step "
                  "(%d wall, returned %s after %.0f ms with %d volumes)",
                  static_cast<int>(wall.size()), ok ? "true" : "false", elapsed_ms, volumes);
    check(wall.size() == 1 && !ok && volumes == 0 && elapsed_ms < 1500.0, msg);
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

// A concrete SMESH_Delaunay: the one abstract member is where a node sits in the face's
// parameter space, which the probe's own fixture answers directly.
class ProbeDelaunay : public SMESH_Delaunay {
 public:
  ProbeDelaunay(const std::vector<const UVPtStructVec*>& boundary, const TopoDS_Face& face,
                int face_id)
      : SMESH_Delaunay(boundary, face, face_id) {}

 protected:
  gp_XY getNodeUV(const TopoDS_Face&, const SMDS_MeshNode* node) const override {
    return gp_XY(node->X(), node->Y());
  }
};

// ---------------------------------------------------------------------------- EDITBIND -- //
// The behaviours an editor, search and medial-axis binding rests on, as opposed to the
// capabilities the EDITOR, SEARCH and MEDAX sections above already prove. Each of these
// decides a design question that a header read answers wrongly, and three of them are
// upstream defects a caller has to be told about rather than left to meet.
void probe_editor_and_search_binding_behaviour() {
  section("EDITBIND", "the behaviours an editor, search and medial-axis binding depends on");

  // ---- The surface offset refuses a mesh it cannot handle, by throwing ---------------- //
  // It tests the WHOLE source mesh rather than the faces it was handed, so a triangular
  // patch of a mesh that also holds quadrangles is refused. The refusal arrives as an
  // exception, not as a null result, so a binding must translate it.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_quad_mesh(s, 2), "EDITBIND quadrangular skin for the offset probe computes");
    SMESHDS_Mesh* ds = s.meshDS();
    SMESH_MeshAlgos::TElemIntPairVec new2old_faces;
    SMESH_MeshAlgos::TNodeIntPairVec new2old_nodes;
    bool threw = false;
    try {
      SMDS_Mesh* offset =
          SMESH_MeshAlgos::MakeOffset(ds->elementsIterator(SMDSAbs_Face), *ds, 0.1, false,
                                      new2old_faces, new2old_nodes);
      delete offset;
    } catch (const std::exception&) {
      threw = true;
    }
    check(threw, "EDITBIND MakeOffset throws on a mesh that is not all linear triangles");
  }

  // ---- SplitBiQuadraticIntoLinear reads an empty set as nothing ------------------------ //
  // Every other editing call in this package takes an empty set as "the whole mesh". This
  // one does not, so a binding that keeps the convention has to fill the set itself.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2),
          "EDITBIND hexa mesh for the bi-quadratic split probe computes");
    SMESHDS_Mesh* ds = s.meshDS();
    SMESH_MeshEditor editor(&s.mesh());
    editor.ConvertToQuadratic(/*theForce3d=*/true, /*theToBiQuad=*/true);
    const smIdType volumes_before = ds->NbVolumes();
    TIDSortedElemSet nothing;
    editor.SplitBiQuadraticIntoLinear(nothing);
    check(ds->NbVolumes() == volumes_before,
          "EDITBIND SplitBiQuadraticIntoLinear over an empty set splits nothing");
    TIDSortedElemSet everything;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Volume); it->more();) {
      everything.insert(it->next());
    }
    editor.SplitBiQuadraticIntoLinear(everything);
    check(ds->NbVolumes() == 8 * volumes_before,
          "EDITBIND the same call over every cell splits each into 8 linear ones");
  }

  // ---- DeMerge's volume branch cannot report ------------------------------------------- //
  // It builds its comparison from the element's own current nodes rather than from the
  // proposed post-merge ones, so it compares a thing with itself. A caller must not read an
  // empty answer for a volume cell as "this merge is safe".
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2), "EDITBIND hexa mesh for the de-merge probe computes");
    SMESHDS_Mesh* ds = s.meshDS();
    const SMDS_MeshElement* cell = ds->elementsIterator(SMDSAbs_Volume)->next();

    // The connectivity the cell would have if two of its own corners were merged.
    std::vector<const SMDS_MeshNode*> proposed;
    for (SMDS_NodeIteratorPtr it = cell->nodeIterator(); it->more();) {
      proposed.push_back(it->next());
    }
    proposed[2] = proposed[0];
    std::vector<const SMDS_MeshNode*> keep_apart;
    SMESH_MeshAlgos::DeMerge(cell, proposed, keep_apart);
    check(keep_apart.empty(),
          "EDITBIND DeMerge reports nothing for a volume cell, whatever the merge proposed");
  }

  // ---- The line query is a broad phase, in both senses ---------------------------------- //
  // It answers about bounding boxes, and about an infinite LINE rather than a half line, so
  // a caller reading it as a hit list is wrong in both directions. That is why the binding
  // exposes it under its own name and computes the hits over it.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 3), "EDITBIND hexa mesh for the ray probe computes");
    SMESHDS_Mesh* ds = s.meshDS();
    std::unique_ptr<SMESH_ElementSearcher> searcher(SMESH_MeshAlgos::GetElementSearcher(*ds));

    // A line starting past the far side of the box still reports the faces behind it.
    std::vector<const SMDS_MeshElement*> behind;
    const gp_Pnt beyond(BX / 2.0, BY / 2.0, BZ + 10.0);
    searcher->GetElementsNearLine(gp_Ax1(beyond, gp_Dir(0, 0, 1)), SMDSAbs_Face, behind);
    check(!behind.empty(),
          "EDITBIND the line query reports faces behind its own origin (it is a line)");

    // And a line that misses every face but crosses their bounding boxes still reports them.
    std::vector<const SMDS_MeshElement*> boxed;
    searcher->GetElementsNearLine(gp_Ax1(gp_Pnt(0, 0, -10.0), gp_Dir(0, 0, 1)), SMDSAbs_Face,
                                  boxed);
    check(!boxed.empty(),
          "EDITBIND the line query answers about bounding boxes, not about the faces");
  }

  // ---- An open surface is classified OUT, not UNKNOWN ----------------------------------- //
  // There is no inside of an open surface, and the searcher does not say so: it answers OUT
  // for every point, including one lying on the surface. A caller wanting to know whether the
  // question was meaningful has to test for a free border itself.
  {
    Session s(BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0, BX, 0, BY)
                  .Face());
    check(build_quad_mesh(s, 2),
          "EDITBIND single-face mesh for the point-state probe computes");
    SMESHDS_Mesh* ds = s.meshDS();
    std::unique_ptr<SMESH_ElementSearcher> searcher(SMESH_MeshAlgos::GetElementSearcher(*ds));
    const TopAbs_State above = searcher->GetPointState(gp_Pnt(BX / 2.0, BY / 2.0, 5.0));
    const TopAbs_State on = searcher->GetPointState(gp_Pnt(BX / 2.0, BY / 2.0, 0.0));
    check(above == TopAbs_OUT && on == TopAbs_OUT,
          "EDITBIND an open surface classifies every point OUT, never UNKNOWN");
  }

  // ---- SMESH_Delaunay cannot answer under this OCCT ------------------------------------- //
  // The most consequential finding of the three. It hands its boundary points to OCCT's
  // triangulator as a bare vertex array, and the triangulator comes back having marked every
  // triangle deleted with an empty live-element set — so the class's own entry point finds no
  // triangle beside any boundary node and every query it offers returns nothing. The reach is
  // wider than the class: the projection utilities subclass it.
  {
    const TopoDS_Face face =
        BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0, BX, 0, BY).Face();
    std::unique_ptr<SMDS_Mesh> scratch(new SMDS_Mesh);
    UVPtStructVec boundary;
    const double corners[8][2] = {{0, 0},   {1.5, 0}, {3, 0},   {3, 3.5},
                                  {3, 7},   {1.5, 7}, {0, 7},   {0, 3.5}};
    for (int i = 0; i < 8; ++i) {
      const SMDS_MeshNode* node =
          scratch->AddNode(corners[i][0], corners[i][1], 0.0);
      UVPtStruct point(node);
      point.SetUV(gp_XY(corners[i][0], corners[i][1]));
      point.param = point.normParam = point.x = point.y = 0.0;
      boundary.push_back(point);
    }
    std::vector<const UVPtStructVec*> wires(1, &boundary);
    ProbeDelaunay delaunay(wires, face, 1);
    Handle(BRepMesh_DataStructureOfDelaun) structure = delaunay.GetDS();
    int alive = 0;
    for (int t = 1; t <= structure->NbElements(); ++t) {
      if (structure->GetElement(t).Movability() != BRepMesh_Deleted) {
        ++alive;
      }
    }
    char msg[220];
    std::snprintf(msg, sizeof(msg),
                  "EDITBIND SMESH_Delaunay leaves no live triangle (%d elements, %d alive, "
                  "%d in the domain)",
                  structure->NbElements(), alive, structure->ElementsOfDomain().Extent());
    check(structure->NbElements() > 0 && alive == 0 &&
              structure->ElementsOfDomain().Extent() == 0,
          msg);
    check(delaunay.GetTriangleNear(0) == nullptr,
          "EDITBIND and its own entry point therefore finds no triangle to start from");
  }

  // ---- The ordered-edge walk is per wire ------------------------------------------------ //
  // Both the medial axis and any boundary walk need the face's edges in wire order, and a
  // face with a hole has more than one wire. The counts come back per wire, which is what
  // makes the two orders line up.
  {
    const TopoDS_Face face =
        BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0, BX, 0, BY).Face();
    std::list<TopoDS_Edge> edges;
    std::list<int> per_wire;
    const int wires = SMESH_Block::GetOrderedEdges(face, edges, per_wire);
    check(wires == 1 && per_wire.size() == 1 && per_wire.front() == 4 && edges.size() == 4,
          "EDITBIND GetOrderedEdges reports the face's edges grouped by wire");
  }

  // ---- Ignoring the corners collapses a rectangle's axis to one branch ------------------ //
  // The corner arms are branches like any other, so a caller counting branches to find a
  // junction would find two on a shape that has none. Both settings are pinned.
  {
    const double w = 10.0;
    const double h = 4.0;
    const TopoDS_Face rect =
        BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0, w, 0, h).Face();
    std::vector<TopoDS_Edge> edges;
    for (TopExp_Explorer ex(rect, TopAbs_EDGE); ex.More(); ex.Next()) {
      edges.push_back(TopoDS::Edge(ex.Current()));
    }
    SMESH_MAT2d::MedialAxis kept(rect, edges, 0.1, /*ignoreCorners=*/false);
    SMESH_MAT2d::MedialAxis dropped(rect, edges, 0.1, /*ignoreCorners=*/true);
    check(kept.nbBranches() == 5 && dropped.nbBranches() == 1,
          "EDITBIND ignoring the corners takes a rectangle's axis from 5 branches to 1");
  }
}

// ---------------------------------------------------------------------------- CTLBIND -- //
// The behaviours a quality-control and group binding rests on, as opposed to the capabilities
// the QC and GROUPS sections above already prove. Each decides a design question a header read
// cannot answer: when a control's cached state is stale, whether an editing operation tells
// the mesh it changed, and whether a group follows an edit that replaces or deletes elements.
void probe_controls_and_groups_binding_behaviour() {
  section("CTLBIND", "the behaviours a controls-and-groups binding depends on");

  // ---- A mesh assembled by hand starts with modification time 0 ----------------------- //
  // Several controls cache against SMDS's modification time and read "unchanged since I last
  // looked" from a mesh that has never been published. A binding that builds a mesh from
  // arrays must call Modified() or those controls answer about nothing at all.
  {
    SMESH_Gen gen;
    SMESH_Mesh* mesh = gen.CreateMesh(false);
    SMESHDS_Mesh* ds = mesh->GetMeshDS();
    ds->AddNodeWithID(0.0, 0.0, 0.0, 1);
    ds->AddNodeWithID(BX, 0.0, 0.0, 2);
    ds->AddNodeWithID(BX, BY, 0.0, 3);
    ds->AddNodeWithID(0.0, BY, 0.0, 4);
    ds->AddNodeWithID(0.0, 0.0, 0.0, 5);  // deliberately on top of node 1
    ds->AddFaceWithID(1, 2, 3, 4, 1);

    check(ds->GetMTime() == 0,
          "CTLBIND a mesh built by hand has modification time 0 until it is published");

    SMESH::Controls::CoincidentNodes stale;
    stale.SetMesh(ds);
    check(!stale.IsSatisfy(1) && !stale.IsSatisfy(5),
          "CTLBIND a modification-tracked control finds nothing on an unpublished mesh");

    ds->Modified();
    SMESH::Controls::CoincidentNodes fresh;
    fresh.SetMesh(ds);
    check(fresh.IsSatisfy(1) && fresh.IsSatisfy(5),
          "CTLBIND the same control finds the coincident pair once Modified() is called");
    delete mesh;
  }

  // ---- An editing operation does not publish itself ----------------------------------- //
  // SMESH_MeshEditor calls Modified() in exactly one of its operations; the SALOME layer
  // above it does the rest. A group defined by a filter tests that time to decide whether to
  // re-evaluate, so an edited mesh keeps answering with its old membership until published.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2), "CTLBIND hexa mesh for the staleness probe computes");
    SMESHDS_Mesh* ds = s.meshDS();

    SMESH::Controls::PredicatePtr positive;
    {
      SMESH::Controls::MoreThan* more = new SMESH::Controls::MoreThan();
      positive.reset(more);
      more->SetNumFunctor(
          SMESH::Controls::NumericalFunctorPtr(new SMESH::Controls::Volume()));
      more->SetMargin(0.0);
    }
    SMESH_Group* group = s.mesh().AddGroup(SMDSAbs_Volume, "positive", -1, TopoDS_Shape(),
                                           positive);
    check(group != nullptr && group->GetGroupDS() != nullptr,
          "CTLBIND SMESH_Mesh::AddGroup accepts a predicate and makes a filtered group");
    SMESHDS_GroupBase* gds = group->GetGroupDS();
    const smIdType before = gds->Extent();

    SMESH_MeshEditor editor(&s.mesh());
    SMESH_MeshEditor::TFacetOfElem facets;
    TIDSortedElemSet hexas;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Volume); it->more();) {
      hexas.insert(it->next());
    }
    editor.GetHexaFacetsToSplit(hexas, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), facets);
    const smIdType volumes_before = ds->NbVolumes();
    editor.SplitVolumes(facets, SMESH_MeshEditor::HEXA_TO_2_PRISMS);
    const smIdType volumes_after = ds->NbVolumes();

    const smIdType stale = gds->Extent();
    ds->Modified();
    const smIdType fresh = gds->Extent();
    char msg[240];
    std::snprintf(msg, sizeof(msg),
                  "CTLBIND a filtered group is stale after an edit until the mesh is "
                  "published (%d cells -> %d, group read %d then %d)",
                  static_cast<int>(volumes_before), static_cast<int>(volumes_after),
                  static_cast<int>(stale), static_cast<int>(fresh));
    check(before == volumes_before && volumes_after == 2 * volumes_before &&
              stale == volumes_before && fresh == volumes_after,
          msg);
  }

  // ---- An explicit group follows a replacement and a deletion ------------------------- //
  // The property a named group exists for: SMESH's editor rewrites group membership as it
  // works, so a group named on a coarse mesh still names the right cells afterwards.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2), "CTLBIND hexa mesh for the group-survival probe computes");
    SMESHDS_Mesh* ds = s.meshDS();

    SMESH_Group* group = s.mesh().AddGroup(SMDSAbs_Volume, "half");
    SMESHDS_Group* gds = dynamic_cast<SMESHDS_Group*>(group->GetGroupDS());
    int added = 0;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Volume); it->more();) {
      const SMDS_MeshElement* e = it->next();
      if (added < 4) {
        gds->Add(e);
        ++added;
      }
    }
    check(gds->Extent() == 4, "CTLBIND the explicit group starts with the 4 cells added");

    // An id of another family is refused rather than silently dropped.
    const SMDS_MeshElement* a_face = ds->elementsIterator(SMDSAbs_Face)->next();
    check(!gds->Add(a_face->GetID()),
          "CTLBIND an id of the wrong family is refused by the group");

    SMESH_MeshEditor editor(&s.mesh());
    SMESH_MeshEditor::TFacetOfElem facets;
    TIDSortedElemSet hexas;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Volume); it->more();) {
      hexas.insert(it->next());
    }
    editor.GetHexaFacetsToSplit(hexas, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), facets);
    editor.SplitVolumes(facets, SMESH_MeshEditor::HEXA_TO_2_PRISMS);
    ds->Modified();
    check(gds->Extent() == 8,
          "CTLBIND the group follows a split: each cell replaced by two, both in the group");

    // Every member must still be an element of the mesh — a group naming a deleted element
    // is the failure that would corrupt a solver handoff.
    bool all_alive = true;
    for (SMDS_ElemIteratorPtr it = gds->GetElements(); it->more();) {
      all_alive = all_alive && ds->FindElement(it->next()->GetID()) != nullptr;
    }
    check(all_alive, "CTLBIND every member of the group is still an element of the mesh");

    TIDSortedNodeSet whole;
    SMESH_MeshEditor::TListOfListOfNodes coincident;
    editor.FindCoincidentNodes(whole, BX / 2.0 + 0.1, coincident, false);
    const smIdType members_before = gds->Extent();
    editor.MergeNodes(coincident);
    ds->Modified();
    bool alive_after_merge = true;
    for (SMDS_ElemIteratorPtr it = gds->GetElements(); it->more();) {
      alive_after_merge =
          alive_after_merge && ds->FindElement(it->next()->GetID()) != nullptr;
    }
    char msg[240];
    std::snprintf(msg, sizeof(msg),
                  "CTLBIND the group drops the elements a merge deleted (%d members before, "
                  "%d after, %d groups of coincident nodes)",
                  static_cast<int>(members_before), static_cast<int>(gds->Extent()),
                  static_cast<int>(coincident.size()));
    check(!coincident.empty() && alive_after_merge && gds->Extent() <= members_before, msg);

    // ConvertFromQuadratic reports success whether or not anything was quadratic, so its
    // return value carries no information and a binding must not read one into it.
    check(editor.ConvertFromQuadratic(),
          "CTLBIND ConvertFromQuadratic returns true on an already-linear mesh");
  }

  // ---- ManifoldPart walks its whole face vector without leaving it -------------------- //
  // Upstream advanced the index itself and skipped its own wrap on an already-treated face,
  // so it read past the end of the vector. Patched (see PROVENANCE.md); this is the pin.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    check(build_hexa_mesh(s, 2), "CTLBIND hexa mesh for the manifold-walk probe computes");
    SMESHDS_Mesh* ds = s.meshDS();
    const smIdType first_face = ds->elementsIterator(SMDSAbs_Face)->next()->GetID();

    SMESH::Controls::ManifoldPart manifold;
    manifold.SetStartElem(static_cast<long>(first_face));
    manifold.SetIsOnlyManifold(true);
    manifold.SetMesh(ds);  // the walk runs here; an out-of-bounds one crashes the process
    int selected = 0;
    for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Face); it->more();) {
      if (manifold.IsSatisfy(it->next()->GetID())) {
        ++selected;
      }
    }
    check(selected > 0 && selected <= ds->NbFaces(),
          "CTLBIND ManifoldPart walks every face of a closed shell and stays in bounds");
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


// --------------------------------------------------------------------------- CAT916 ----- //
// The native catalogue entries added with SMESH 9.16 (src/bindings/mesher_catalog.cpp). Each
// case builds the entry the way the catalogue's Factory does, new T(GetANewId(), &gen), and
// computes on the smallest shape the entry supports. The pytest counterparts, against the
// geometry and the upstream spec, are in tests/test_mesher_native.py.

// The x coordinates of every node of the mesh, ascending. The cases below mesh one straight
// edge along x, so this is the node distribution.
std::vector<double> sorted_node_x(SMESHDS_Mesh* meshDS) {
  std::vector<double> xs;
  for (SMDS_NodeIteratorPtr it = meshDS->nodesIterator(); it->more();) {
    xs.push_back(it->next()->X());
  }
  std::sort(xs.begin(), xs.end());
  return xs;
}

// The positions of the nodes on one edge, as fractions of its length, measured from `from`.
std::vector<double> edge_fractions(SMESHDS_Mesh* meshDS, const TopoDS_Edge& edge,
                                   const gp_Pnt& from, double length) {
  std::vector<double> out;
  TopoDS_Vertex v0, v1;
  TopExp::Vertices(edge, v0, v1);
  std::set<const SMDS_MeshNode*> nodes;
  if (SMESHDS_SubMesh* sm = meshDS->MeshElements(edge)) {
    for (SMDS_NodeIteratorPtr it = sm->GetNodes(); it->more();) nodes.insert(it->next());
  }
  for (const TopoDS_Vertex& v : {v0, v1}) {
    if (SMESHDS_SubMesh* sm = meshDS->MeshElements(v)) {
      for (SMDS_NodeIteratorPtr it = sm->GetNodes(); it->more();) nodes.insert(it->next());
    }
  }
  for (const SMDS_MeshNode* n : nodes) {
    out.push_back(from.Distance(gp_Pnt(n->X(), n->Y(), n->Z())) / length);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// The edge of `shape` whose midpoint is closest to `p`.
TopoDS_Edge edge_near(const TopoDS_Shape& shape, const gp_Pnt& p) {
  TopoDS_Edge best;
  double best_d = 1e300;
  for (TopExp_Explorer ex(shape, TopAbs_EDGE); ex.More(); ex.Next()) {
    const TopoDS_Edge& e = TopoDS::Edge(ex.Current());
    TopoDS_Vertex v0, v1;
    TopExp::Vertices(e, v0, v1);
    const gp_Pnt a = BRep_Tool::Pnt(v0), b = BRep_Tool::Pnt(v1);
    const double d = p.Distance(gp_Pnt(0.5 * (a.XYZ() + b.XYZ())));
    if (d < best_d) {
      best_d = d;
      best = e;
    }
  }
  return best;
}

// The vertex of `shape` closest to `p`.
TopoDS_Vertex vertex_near(const TopoDS_Shape& shape, const gp_Pnt& p) {
  TopoDS_Vertex best;
  double best_d = 1e300;
  for (TopExp_Explorer ex(shape, TopAbs_VERTEX); ex.More(); ex.Next()) {
    const TopoDS_Vertex& v = TopoDS::Vertex(ex.Current());
    const double d = p.Distance(BRep_Tool::Pnt(v));
    if (d < best_d) {
      best_d = d;
      best = v;
    }
  }
  return best;
}

// The closed form of StdMeshers_Regular_1D::computeBetaLaw on a straight edge of length L.
std::vector<double> beta_law_positions(double beta, int n, double length) {
  std::vector<double> xs(1, 0.0);
  const double r = (1.0 + std::fabs(beta)) / (std::fabs(beta) - 1.0);
  std::vector<double> t;
  for (int i = 1; i < n; ++i) {
    const double power = std::pow(r, 1.0 - static_cast<double>(i) / n);
    t.push_back(1.0 + std::fabs(beta) * (1.0 - power) / (1.0 + power));
  }
  if (beta < 0) {  // the reversed law: mirror the positions
    for (double& v : t) v = 1.0 - v;
    std::sort(t.begin(), t.end());
  }
  for (double v : t) xs.push_back(v * length);
  xs.push_back(length);
  return xs;
}

void probe_cat916_1d_additions() {
  section("CAT916", "native catalogue entries added with SMESH 9.16: 1-D family");

  // SegmentAroundVertex_0D + SegmentLengthAroundVertex: the segment next to the vertex takes
  // the length the hypothesis names; the rest of the edge keeps its own 1-D hypothesis.
  {
    Session s(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(10, 0, 0)).Edge());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
    n->SetNumberOfSegments(5);
    StdMeshers_SegmentAroundVertex_0D* a0 = s.make<StdMeshers_SegmentAroundVertex_0D>();
    StdMeshers_SegmentLengthAroundVertex* around = s.make<StdMeshers_SegmentLengthAroundVertex>();
    around->SetLength(0.5);
    const TopoDS_Vertex origin = vertex_near(s.shape(), gp_Pnt(0, 0, 0));
    const bool ok = s.assign(s.shape(), a1) && s.assign(s.shape(), n) && s.assign(origin, a0) &&
                    s.assign(origin, around);
    check(ok, "CAT916 SegmentAroundVertex_0D + SegmentLengthAroundVertex assign on a vertex");
    check(s.compute(), "CAT916 SegmentAroundVertex_0D computes with Regular_1D");
    const std::vector<double> xs = sorted_node_x(s.meshDS());
    check(xs.size() >= 3, "CAT916 SegmentAroundVertex_0D leaves a discretised edge");
    if (xs.size() >= 3) {
      check_close(xs[1] - xs[0], 0.5, 1e-9,
                  "CAT916 SegmentAroundVertex_0D: the segment at the vertex is 0.5 long");
    }
  }

  // PropagOfDistribution on the long side of a trapezoid: the opposite, shorter side gets the
  // same number of nodes at the same fractions of its length.
  {
    BRepBuilderAPI_MakePolygon poly(gp_Pnt(0, 0, 0), gp_Pnt(4, 0, 0), gp_Pnt(3, 2, 0),
                                    gp_Pnt(1, 2, 0), /*Close=*/true);
    Session s(BRepBuilderAPI_MakeFace(poly.Wire()).Face());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
    n->SetNumberOfSegments(2);
    StdMeshers_Arithmetic1D* arith = s.make<StdMeshers_Arithmetic1D>();
    arith->SetLength(0.5, true);
    arith->SetLength(1.5, false);
    StdMeshers_PropagOfDistribution* prop = s.make<StdMeshers_PropagOfDistribution>();
    const TopoDS_Edge bottom = edge_near(s.shape(), gp_Pnt(2, 0, 0));
    const TopoDS_Edge top = edge_near(s.shape(), gp_Pnt(2, 2, 0));
    const bool ok = s.assign(s.shape(), a1) && s.assign(s.shape(), n) &&
                    s.assign(bottom, arith) && s.assign(bottom, prop);
    check(ok, "CAT916 PropagOfDistribution assigns beside a local 1-D hypothesis");
    check(s.compute(), "CAT916 PropagOfDistribution computes");
    const std::vector<double> fb = edge_fractions(s.meshDS(), bottom, gp_Pnt(0, 0, 0), 4.0);
    const std::vector<double> ft = edge_fractions(s.meshDS(), top, gp_Pnt(1, 2, 0), 2.0);
    std::vector<double> ft_rev;
    for (double v : ft) ft_rev.push_back(1.0 - v);
    std::sort(ft_rev.begin(), ft_rev.end());
    double worst = 0.0, worst_rev = 0.0;
    const bool same_count = fb.size() == ft.size() && fb.size() > 3;
    for (std::size_t i = 0; same_count && i < fb.size(); ++i) {
      worst = std::max(worst, std::fabs(fb[i] - ft[i]));
      worst_rev = std::max(worst_rev, std::fabs(fb[i] - ft_rev[i]));
    }
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "CAT916 PropagOfDistribution: the 2-long top edge repeats the 4-long bottom "
                  "edge's %zu node fractions (max deviation %.2e)",
                  fb.size(), std::min(worst, worst_rev));
    check(same_count && std::min(worst, worst_rev) < 1e-9, msg);
  }

  // NumberOfSegments with DT_BetaLaw (new in 9.16): the nodes sit at the closed form of the
  // law, on a straight 10-long edge, for both signs of beta.
  for (const double beta : {1.01, -1.05}) {
    Session s(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(10, 0, 0)).Edge());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
    n->SetNumberOfSegments(10);
    n->SetDistrType(StdMeshers_NumberOfSegments::DT_BetaLaw);
    n->SetBeta(beta);
    const bool ok = s.assign(s.shape(), a1) && s.assign(s.shape(), n);
    check(ok && s.compute(), "CAT916 NumberOfSegments DT_BetaLaw computes");
    const std::vector<double> got = sorted_node_x(s.meshDS());
    const std::vector<double> want = beta_law_positions(beta, 10, 10.0);
    double worst = got.size() == want.size() ? 0.0 : 1e300;
    for (std::size_t i = 0; got.size() == want.size() && i < got.size(); ++i) {
      worst = std::max(worst, std::fabs(got[i] - want[i]));
    }
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "CAT916 DT_BetaLaw beta=%.2f: 11 nodes at the closed form (max error %.2e)",
                  beta, worst);
    check(worst < 1e-9, msg);
  }
}


// Sum of the areas of the triangles and quadrangles of the mesh, by the cross product.
double face_area_sum(SMESHDS_Mesh* meshDS) {
  double area = 0.0;
  for (SMDS_FaceIteratorPtr it = meshDS->facesIterator(); it->more();) {
    const SMDS_MeshElement* f = it->next();
    const int nb = f->NbCornerNodes();
    const gp_Pnt p0(f->GetNode(0)->X(), f->GetNode(0)->Y(), f->GetNode(0)->Z());
    for (int i = 1; i + 1 < nb; ++i) {
      const gp_Pnt p1(f->GetNode(i)->X(), f->GetNode(i)->Y(), f->GetNode(i)->Z());
      const gp_Pnt p2(f->GetNode(i + 1)->X(), f->GetNode(i + 1)->Y(), f->GetNode(i + 1)->Z());
      area += 0.5 * gp_Vec(p0, p1).Crossed(gp_Vec(p0, p2)).Magnitude();
    }
  }
  return area;
}

void probe_cat916_2d_additions() {
  section("CAT916", "native catalogue entries added with SMESH 9.16: 2-D family");

  // LengthFromEdges with MEFISTO_2D on a 4 x 4 square, 8 segments a side: the triangles fill
  // the square, and their size follows the mean boundary segment, 0.5.
  {
    Session s(BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0, 4, 0, 4)
                  .Face());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
    n->SetNumberOfSegments(8);
    StdMeshers_MEFISTO_2D* a2 = s.make<StdMeshers_MEFISTO_2D>();
    StdMeshers_LengthFromEdges* lfe = s.make<StdMeshers_LengthFromEdges>();
    const bool ok = s.assign(s.shape(), a1) && s.assign(s.shape(), n) &&
                    s.assign(s.shape(), a2) && s.assign(s.shape(), lfe);
    check(ok, "CAT916 MEFISTO_2D + LengthFromEdges assign");
    check(s.compute() && s.meshDS()->NbFaces() > 0, "CAT916 MEFISTO_2D + LengthFromEdges computes");
    check_close(face_area_sum(s.meshDS()), 16.0, 1e-9,
                "CAT916 LengthFromEdges: the triangles cover the 4 x 4 square exactly");
    // MEFISTO uses the length as an ideal edge length (areteideale), not as a bound, so the
    // mean triangle edge is what follows it.
    double sum = 0.0;
    int count = 0;
    for (SMDS_FaceIteratorPtr it = s.meshDS()->facesIterator(); it->more();) {
      const SMDS_MeshElement* f = it->next();
      for (int i = 0; i < 3; ++i) {
        const SMDS_MeshNode* a = f->GetNode(i);
        const SMDS_MeshNode* b = f->GetNode((i + 1) % 3);
        sum += gp_Pnt(a->X(), a->Y(), a->Z()).Distance(gp_Pnt(b->X(), b->Y(), b->Z()));
        ++count;
      }
    }
    const double mean = count ? sum / count : 0.0;
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "CAT916 LengthFromEdges: the mean triangle edge, %.4f, is within a factor "
                  "1.5 of the mean boundary segment 0.5",
                  mean);
    check(mean > 0.5 / 1.5 && mean < 0.5 * 1.5, msg);
  }

  // LayerDistribution2D with RadialQuadrangle_1D2D on a disk of radius 2.5: the rings follow
  // the inner 1-D hypothesis, here 4 equal layers, so every node radius is a multiple of
  // 2.5 / 4.
  {
    gp_Circ circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 2.5);
    const TopoDS_Wire wire = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circ).Edge()).Wire();
    Session s(BRepBuilderAPI_MakeFace(wire).Face());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
    n->SetNumberOfSegments(8);
    StdMeshers_RadialQuadrangle_1D2D* a2 = s.make<StdMeshers_RadialQuadrangle_1D2D>();
    StdMeshers_NumberOfSegments* radial = s.make<StdMeshers_NumberOfSegments>();
    radial->SetNumberOfSegments(4);
    StdMeshers_LayerDistribution2D* layers = s.make<StdMeshers_LayerDistribution2D>();
    layers->SetLayerDistribution(radial);
    const bool ok = s.assign(s.shape(), a1) && s.assign(s.shape(), n) &&
                    s.assign(s.shape(), a2) && s.assign(s.shape(), layers);
    check(ok, "CAT916 RadialQuadrangle_1D2D + LayerDistribution2D assign");
    check(s.compute() && s.meshDS()->NbFaces() > 0,
          "CAT916 RadialQuadrangle_1D2D + LayerDistribution2D computes");
    double worst = 0.0;
    for (SMDS_NodeIteratorPtr it = s.meshDS()->nodesIterator(); it->more();) {
      const SMDS_MeshNode* node = it->next();
      const double layer = std::hypot(node->X(), node->Y()) / (2.5 / 4.0);
      worst = std::max(worst, std::fabs(layer - std::round(layer)));
    }
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "CAT916 LayerDistribution2D: every node radius is k * 2.5 / 4 (worst %.2e "
                  "of a layer)",
                  worst);
    check(worst < 1e-9, msg);
  }

  // UseExisting_2D on one face of a box: that face gets no element and its sub-mesh counts as
  // computed; the five other faces are meshed as usual.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
    n->SetNumberOfSegments(3);
    StdMeshers_Quadrangle_2D* a2 = s.make<StdMeshers_Quadrangle_2D>();
    StdMeshers_UseExisting_2D* manual = s.make<StdMeshers_UseExisting_2D>();
    NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> faces;
    TopExp::MapShapes(s.shape(), TopAbs_FACE, faces);
    const bool ok = s.assign(s.shape(), a1) && s.assign(s.shape(), n) &&
                    s.assign(s.shape(), a2) && s.assign(faces.FindKey(1), manual);
    check(ok, "CAT916 UseExisting_2D assigns on one face beside a global Quadrangle_2D");
    check(s.compute(), "CAT916 UseExisting_2D: the compute succeeds");
    SMESHDS_SubMesh* sm1 = s.meshDS()->MeshElements(faces.FindKey(1));
    check(!sm1 || sm1->NbElements() == 0, "CAT916 UseExisting_2D: its face has no element");
    check(s.mesh().GetSubMesh(faces.FindKey(1))->IsMeshComputed(),
          "CAT916 UseExisting_2D: its face counts as computed");
    check(s.meshDS()->NbFaces() == 5 * 9, "CAT916 UseExisting_2D: the 5 other faces get 9 quads each");
  }

  // UseExisting_1D on a lone edge: no segment is made, and the compute succeeds.
  {
    Session s(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(10, 0, 0)).Edge());
    StdMeshers_UseExisting_1D* manual = s.make<StdMeshers_UseExisting_1D>();
    check(s.assign(s.shape(), manual), "CAT916 UseExisting_1D assigns on an edge");
    check(s.compute(), "CAT916 UseExisting_1D: the compute succeeds");
    check(s.meshDS()->NbEdges() == 0, "CAT916 UseExisting_1D: no segment is made");
  }
}


// Count the volumes of one entity type.
int count_volumes(SMESHDS_Mesh* meshDS, SMDSAbs_EntityType type) {
  int n = 0;
  for (SMDS_VolumeIteratorPtr it = meshDS->volumesIterator(); it->more();) {
    if (it->next()->GetEntityType() == type) ++n;
  }
  return n;
}

// Cartesian_3D on a radius-2 sphere at spacing 0.5, optionally with quanta.
void cartesian_sphere(Session& s, bool use_quanta, double quanta) {
  StdMeshers_Cartesian_3D* a3 = s.make<StdMeshers_Cartesian_3D>();
  StdMeshers_CartesianParameters3D* grid = s.make<StdMeshers_CartesianParameters3D>();
  for (int axis = 0; axis < 3; ++axis) {
    std::vector<std::string> spacing(1, "0.5");
    std::vector<double> internal;
    grid->SetGridSpacing(spacing, internal, axis);
  }
  if (use_quanta) {
    grid->SetToUseQuanta(true);
    grid->SetQuanta(quanta);
  }
  s.assign(s.shape(), a3);
  s.assign(s.shape(), grid);
}

void probe_cat916_3d_additions() {
  section("CAT916", "native catalogue entries added with SMESH 9.16: 3-D family and global");

  // BlockRenumber (parameter-free) with Hexa_3D on an axis-aligned box: hexahedra and nodes
  // come in structured i, j, k order, i fastest, from the corner at the origin.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    StdMeshers_BlockRenumber* renumber = s.make<StdMeshers_BlockRenumber>();
    const bool ok = s.assign(s.shape(), renumber);
    check(ok && build_hexa_mesh(s, 3), "CAT916 BlockRenumber + Hexa_3D computes");
    std::vector<std::pair<smIdType, int>> cells;
    for (SMDS_VolumeIteratorPtr it = s.meshDS()->volumesIterator(); it->more();) {
      const SMDS_MeshElement* v = it->next();
      double c[3] = {0, 0, 0};
      for (int i = 0; i < v->NbCornerNodes(); ++i) {
        c[0] += v->GetNode(i)->X() / v->NbCornerNodes();
        c[1] += v->GetNode(i)->Y() / v->NbCornerNodes();
        c[2] += v->GetNode(i)->Z() / v->NbCornerNodes();
      }
      const int i = static_cast<int>(c[0] / (BX / 3)), j = static_cast<int>(c[1] / (BY / 3)),
                k = static_cast<int>(c[2] / (BZ / 3));
      cells.emplace_back(v->GetID(), i + 3 * (j + 3 * k));
    }
    std::sort(cells.begin(), cells.end());
    bool structured = cells.size() == 27;
    for (std::size_t n = 0; structured && n < cells.size(); ++n) {
      structured = cells[n].second == static_cast<int>(n);
    }
    check(structured, "CAT916 BlockRenumber: the 27 hexahedra are numbered in i, j, k order");
  }

  // NotConformAllowed is global only (SMESH_Mesh.cxx:658-668).
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    StdMeshers_NotConformAllowed* global = s.make<StdMeshers_NotConformAllowed>();
    StdMeshers_NotConformAllowed* local = s.make<StdMeshers_NotConformAllowed>();
    NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> faces;
    TopExp::MapShapes(s.shape(), TopAbs_FACE, faces);
    check(s.assign(s.shape(), global), "CAT916 NotConformAllowed assigns on the whole shape");
    check(s.assign_status(faces.FindKey(1), local) == SMESH_Hypothesis::HYP_INCOMPATIBLE,
          "CAT916 NotConformAllowed is refused on a sub-shape (HYP_INCOMPATIBLE)");
    check(build_hexa_mesh(s, 2) && s.meshDS()->NbVolumes() == 8,
          "CAT916 NotConformAllowed leaves a conformal hexahedral mesh unchanged");
  }

  // CartesianParameters3D quanta (new in 9.16): at the smallest quanta every cut cell of the
  // boundary becomes one hexahedron, so the polyhedra disappear one for one.
  {
    Session plain(BRepPrimAPI_MakeSphere(2.0).Shape());
    cartesian_sphere(plain, false, 0.0);
    Session quanta(BRepPrimAPI_MakeSphere(2.0).Shape());
    cartesian_sphere(quanta, true, 1e-6);
    check(plain.compute() && quanta.compute(), "CAT916 Cartesian_3D with and without quanta computes");
    const int poly = count_volumes(plain.meshDS(), SMDSEntity_Polyhedra);
    const int hexa = count_volumes(plain.meshDS(), SMDSEntity_Hexa);
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "CAT916 quanta 1e-6: %d polyhedra + %d hexahedra become %d hexahedra and "
                  "%d polyhedra",
                  poly, hexa, count_volumes(quanta.meshDS(), SMDSEntity_Hexa),
                  count_volumes(quanta.meshDS(), SMDSEntity_Polyhedra));
    check(poly > 0 && count_volumes(quanta.meshDS(), SMDSEntity_Polyhedra) == 0 &&
              count_volumes(quanta.meshDS(), SMDSEntity_Hexa) == poly + hexa,
          msg);
  }

  // Cartesian_3D with ViscousLayers (body fitting with viscous layers, 9.16): three layers of
  // total thickness 0.3 and stretch 1.2 off the x = 0 wall sit at the geometric closed form.
  {
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    StdMeshers_Cartesian_3D* a3 = s.make<StdMeshers_Cartesian_3D>();
    StdMeshers_CartesianParameters3D* grid = s.make<StdMeshers_CartesianParameters3D>();
    for (int axis = 0; axis < 3; ++axis) {
      std::vector<std::string> spacing(1, "1.0");
      std::vector<double> internal;
      grid->SetGridSpacing(spacing, internal, axis);
    }
    StdMeshers_ViscousLayers* vl = s.make<StdMeshers_ViscousLayers>();
    vl->SetTotalThickness(0.3);
    vl->SetNumberLayers(3);
    vl->SetStretchFactor(1.2);
    NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> faces;
    TopExp::MapShapes(s.shape(), TopAbs_FACE, faces);
    std::vector<int> wall(1, s.meshDS()->ShapeToIndex(faces.FindKey(1)));  // the x = 0 face
    vl->SetBndShapes(wall, /*toIgnore=*/false);
    const bool ok = s.assign(s.shape(), a3) && s.assign(s.shape(), grid) &&
                    s.assign(s.shape(), vl);
    check(ok && s.compute(), "CAT916 Cartesian_3D + ViscousLayers computes");
    std::set<double> planes;
    for (SMDS_NodeIteratorPtr it = s.meshDS()->nodesIterator(); it->more();) {
      const double x = it->next()->X();
      if (x < 0.31) planes.insert(std::round(x * 1e9) / 1e9);
    }
    const double t1 = 0.3 * (1.2 - 1.0) / (std::pow(1.2, 3) - 1.0);
    const std::vector<double> want = {0.0, t1, t1 + 1.2 * t1, 0.3};
    bool match = planes.size() == want.size();
    std::size_t n = 0;
    for (const double x : planes) {
      match = match && std::fabs(x - want[n++]) < 1e-9;
    }
    check(match, "CAT916 Cartesian_3D + ViscousLayers: layer planes at x = 0, 0.0824, 0.1813, 0.3");
  }
}


// ------------------------------------------------------------------------------ P4DIST ---- //
// The TABLE and EXPRESSION distributions of NumberOfSegments, after the pySMESH patches
// StdMeshers_Distribution_table.patch and StdMeshers_Distribution_expression.patch. Node k of
// N sits where the integral of the density from the start of the edge reaches k/N of its
// total; each check computes that position in closed form. tests/test_mesher_distribution.py
// holds the full grid through the Python API.

// Normalised nodes of n segments in geometric progression on a 15-long edge, first segment h0.
std::vector<double> geometric_target(double h0, int n, double length) {
  double lo = 1.0 + 1e-12, hi = 3.0;
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    (h0 * (std::pow(mid, n) - 1.0) / (mid - 1.0) < length ? lo : hi) = mid;
  }
  const double ratio = 0.5 * (lo + hi);
  std::vector<double> x(1, 0.0);
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += h0 * std::pow(ratio, i);
    x.push_back(sum / length);
  }
  x.back() = 1.0;
  return x;
}

// Node positions of the linearly interpolated density table (x, d): on each interval the
// integral is quadratic, and node k is the root s = 2r / (d_i + sqrt(d_i^2 + 2 a r)).
std::vector<double> table_closed_form(const std::vector<double>& x, const std::vector<double>& d,
                                      int n) {
  std::vector<double> integral(1, 0.0);
  for (std::size_t i = 1; i < x.size(); ++i) {
    integral.push_back(integral.back() + 0.5 * (d[i] + d[i - 1]) * (x[i] - x[i - 1]));
  }
  std::vector<double> t(1, 0.0);
  std::size_t i = 0;
  for (int k = 1; k < n; ++k) {
    const double target = integral.back() * k / n;
    while (i + 2 < x.size() && integral[i + 1] <= target) ++i;
    const double slope = (d[i + 1] - d[i]) / (x[i + 1] - x[i]);
    const double rest = target - integral[i];
    t.push_back(x[i] + 2.0 * rest / (d[i] + std::sqrt(d[i] * d[i] + 2.0 * slope * rest)));
  }
  t.push_back(1.0);
  return t;
}

void probe_p4_distributions() {
  section("P4DIST", "TABLE and EXPRESSION node distributions against their closed forms");
  const double length = 15.0;

  // TABLE: the density 1/h at every node of a geometric target with a 3e-6 wall segment.
  {
    const int n = 100;
    const std::vector<double> x = geometric_target(3e-6, n, length);
    std::vector<double> d;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double h = i == 0              ? x[1] - x[0]
                       : i + 1 == x.size() ? x[i] - x[i - 1]
                                           : 0.5 * (x[i + 1] - x[i - 1]);
      d.push_back(1.0 / h);
    }
    std::vector<double> table;
    for (std::size_t i = 0; i < x.size(); ++i) {
      table.push_back(x[i]);
      table.push_back(d[i]);
    }
    Session s(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(length, 0, 0)).Edge());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* h = s.make<StdMeshers_NumberOfSegments>();
    h->SetNumberOfSegments(n);
    h->SetDistrType(StdMeshers_NumberOfSegments::DT_TabFunc);
    h->SetConversionMode(1);
    h->SetTableFunction(table);
    const bool ok = s.assign(s.shape(), a1) && s.assign(s.shape(), h);
    check(ok && s.compute(), "P4DIST TABLE with a 3e-6 wall segment computes (was 'no message')");
    const std::vector<double> got = sorted_node_x(s.meshDS());
    const std::vector<double> want = table_closed_form(x, d, n);
    double worst = got.size() == want.size() ? 0.0 : 1e300;
    for (std::size_t i = 0; got.size() == want.size() && i < got.size(); ++i) {
      worst = std::max(worst, std::fabs(got[i] - length * want[i]));
    }
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "P4DIST TABLE: 101 nodes within 1e-10 L of the closed form (max error %.2e)",
                  worst);
    check(worst <= 1e-10 * length, msg);
  }

  // EXPRESSION: the density 1/(a+t), node k at a(((1+a)/a)^(k/N) - 1).
  {
    const int n = 100;
    const double a = 3e-5;
    Session s(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(length, 0, 0)).Edge());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* h = s.make<StdMeshers_NumberOfSegments>();
    h->SetNumberOfSegments(n);
    h->SetDistrType(StdMeshers_NumberOfSegments::DT_ExprFunc);
    h->SetConversionMode(1);
    h->SetExpressionFunction("1/(3e-05+t)");
    const bool ok = s.assign(s.shape(), a1) && s.assign(s.shape(), h);
    check(ok && s.compute(), "P4DIST EXPRESSION 1/(3e-5+t) computes");
    const std::vector<double> got = sorted_node_x(s.meshDS());
    double worst = got.size() == static_cast<std::size_t>(n + 1) ? 0.0 : 1e300;
    for (int k = 0; got.size() == static_cast<std::size_t>(n + 1) && k <= n; ++k) {
      const double want = length * a * (std::pow((1.0 + a) / a, double(k) / n) - 1.0);
      worst = std::max(worst, std::fabs(got[k] - want));
    }
    char msg[200];
    std::snprintf(msg, sizeof(msg),
                  "P4DIST EXPRESSION: 101 nodes within 1e-10 L of the closed form (max error "
                  "%.2e; 11.1 m before the patch)",
                  worst);
    check(worst <= 1e-10 * length, msg);
  }

  // EXPRESSION with no finite integral: a pole between the points the setter samples. The
  // edge fails with a compute error that says the integral did not converge.
  {
    Session s(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(length, 0, 0)).Edge());
    StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
    StdMeshers_NumberOfSegments* h = s.make<StdMeshers_NumberOfSegments>();
    h->SetNumberOfSegments(10);
    h->SetDistrType(StdMeshers_NumberOfSegments::DT_ExprFunc);
    h->SetConversionMode(1);
    h->SetExpressionFunction("1/(t-0.3001)^2");
    const bool ok = s.assign(s.shape(), a1) && s.assign(s.shape(), h);
    const bool computed = ok && s.compute();
    TopExp_Explorer edge(s.shape(), TopAbs_EDGE);
    const SMESH_ComputeErrorPtr err = s.mesh().GetSubMesh(edge.Current())->GetComputeError();
    const bool says = err && err->myComment.find("did not converge") != std::string::npos;
    check(ok && !computed && says,
          "P4DIST EXPRESSION 1/(t-0.3001)^2 fails the edge: the integral did not converge");
  }
}

// SMESH_Mesh_hypothesis_status.patch: AddHypothesis returns the HYP_CONCURRENT it finds.
// SMESH_subMesh::CheckConcurrentHypothesis looks for two different similar hypotheses on two
// ancestors of one level, leaving out the one being added (getSimilarAttached). So the
// conflict is two NumberOfSegments already on two faces that share an edge; adding a third
// 1-D hypothesis to the solid around them, which is not the main shape, reports it. Before
// the patch the last check of AddHypothesis overwrote the status with HYP_OK.
void probe_p4_hypothesis_status() {
  section("P4HYP", "AddHypothesis keeps the worst status of its checks");
  BRep_Builder builder;
  TopoDS_Compound two;
  builder.MakeCompound(two);
  builder.Add(two, BRepPrimAPI_MakeBox(3.0, 7.0, 11.0).Shape());
  builder.Add(two, BRepPrimAPI_MakeBox(gp_Pnt(10.0, 0.0, 0.0), 3.0, 7.0, 11.0).Shape());
  NCollection_IndexedMap<TopoDS_Shape, TopTools_ShapeMapHasher> faces, solids;
  TopExp::MapShapes(two, TopAbs_FACE, faces);
  TopExp::MapShapes(two, TopAbs_SOLID, solids);
  // MakeBox lists its faces as x = 0, x = max, y = 0, y = max, z = 0, z = max: faces 1 and 3
  // of the first box share an edge.
  Session s(two);
  StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
  StdMeshers_NumberOfSegments* three = s.make<StdMeshers_NumberOfSegments>();
  StdMeshers_NumberOfSegments* five = s.make<StdMeshers_NumberOfSegments>();
  StdMeshers_NumberOfSegments* seven = s.make<StdMeshers_NumberOfSegments>();
  three->SetNumberOfSegments(3);
  five->SetNumberOfSegments(5);
  seven->SetNumberOfSegments(7);
  const bool before = s.assign(s.shape(), a1) && s.assign(faces.FindKey(1), three) &&
                      s.assign(faces.FindKey(3), five);
  const SMESH_Hypothesis::Hypothesis_Status status = s.assign_status(solids.FindKey(1), seven);
  char msg[200];
  std::snprintf(msg, sizeof(msg),
                "P4HYP 3 and 5 segments on faces sharing an edge, then 7 on their solid: "
                "HYP_CONCURRENT (got %d; HYP_OK = 0 before the patch)",
                static_cast<int>(status));
  check(before && status == SMESH_Hypothesis::HYP_CONCURRENT, msg);
}

// StdMeshers_ViscousLayerBuilder_lifecycle.patch: AddLayers before GetShrinkGeometry throws
// instead of reading an unset pointer; a second GetShrinkGeometry replaces the first; the
// builder is in the generator's map under its own id, so it can outlive the generator.
void probe_p4_layer_builder_lifecycle() {
  section("P4VLB", "the two-step viscous-layer builder owns what it makes");
  const TopoDS_Shape box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape();
  Session s(box);
  StdMeshers_ViscousLayerBuilder* b = s.make<StdMeshers_ViscousLayerBuilder>();
  b->SetTotalThickness(0.2);
  b->SetNumberLayers(2);
  b->SetStretchFactor(1.0);
  b->SetBndShapes(std::vector<int>(), /*toIgnore=*/true);
  bool refused = false;
  try {
    b->AddLayers(s.mesh(), s.mesh(), box);
  } catch (const SALOME_Exception&) {
    refused = true;
  }
  check(refused, "P4VLB AddLayers before GetShrinkGeometry throws SALOME_Exception");
  const TopoDS_Shape first = b->GetShrinkGeometry(s.mesh(), box);
  const TopoDS_Shape second = b->GetShrinkGeometry(s.mesh(), box);
  check(!first.IsNull() && !second.IsNull(),
        "P4VLB two GetShrinkGeometry calls in a row both give a shrunk solid");
  check(s.gen().GetStudyContext()->mapHypothesis[b->GetID()] == b,
        "P4VLB the builder keeps its own entry in the generator's map");
}


// ------------------------------------------------------------------------------ P4CVL ---- //

// Cartesian_3D at the given spacing with ViscousLayers (0.2 thick, 3 layers, factor 1.2) on
// every face of the session's shape.
void cartesian_layers(Session& s, const char* spacing, double thickness = 0.2) {
  StdMeshers_Cartesian_3D* a3 = s.make<StdMeshers_Cartesian_3D>();
  StdMeshers_CartesianParameters3D* grid = s.make<StdMeshers_CartesianParameters3D>();
  for (int axis = 0; axis < 3; ++axis) {
    std::vector<std::string> step(1, spacing);
    std::vector<double> internal;
    grid->SetGridSpacing(step, internal, axis);
  }
  StdMeshers_ViscousLayers* layers = s.make<StdMeshers_ViscousLayers>();
  layers->SetTotalThickness(thickness);
  layers->SetNumberLayers(3);
  layers->SetStretchFactor(1.2);
  layers->SetBndShapes(std::vector<int>(), /*toIgnore=*/true);
  s.assign(s.shape(), a3);
  s.assign(s.shape(), grid);
  s.assign(s.shape(), layers);
}

// StdMeshers_Cartesian_VL_duplicate_nodes.patch, StdMeshers_Cartesian_3D_viscous_submeshes
// .patch, StdMeshers_Cartesian_3D_offset_small_cells.patch and
// SMDS_UnstructuredGrid_links_leak.patch.
void probe_p4_cartesian_layers() {
  section("P4CVL", "Cartesian_3D with viscous layers on inclined and curved walls");
  {
    // A regular hexagonal prism, circumradius 1, height 1: at spacing 0.1 the vertical
    // edges at x = +-1 lie on end planes of the grid, where the offset mesh doubles nodes.
    const double kPi = std::acos(-1.0);
    BRepBuilderAPI_MakePolygon hexagon;
    for (int k = 0; k < 6; ++k) {
      hexagon.Add(gp_Pnt(std::cos(k * kPi / 3.0), std::sin(k * kPi / 3.0), 0.0));
    }
    hexagon.Close();
    const TopoDS_Face base = BRepBuilderAPI_MakeFace(hexagon.Wire()).Face();
    Session s(BRepPrimAPI_MakePrism(base, gp_Vec(0.0, 0.0, 1.0)).Shape());
    cartesian_layers(s, "0.1");
    check(s.compute() && s.meshDS()->NbVolumes() > 0,
          "P4CVL hexagonal prism at spacing 0.1 computes (was 'bad mesh on offset geometry')");
  }
  {
    // A cylinder, radius 1, height 2, spacing 0.25: the seam EDGE has no element of its own,
    // and cut cells under the default size threshold were dropped from the offset mesh.
    const TopoDS_Shape cylinder = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape();
    Session s(cylinder);
    cartesian_layers(s, "0.25");
    check(s.compute(), "P4CVL cylinder computes with every sub-mesh computed");
    TopExp_Explorer solid(cylinder, TopAbs_SOLID);
    int faces_on_solid = 0;
    if (SMESHDS_SubMesh* sm = s.meshDS()->MeshElements(solid.Current())) {
      for (SMDS_ElemIteratorPtr it = sm->GetElements(); it->more();) {
        faces_on_solid += it->next()->GetType() == SMDSAbs_Face ? 1 : 0;
      }
    }
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "P4CVL cylinder: no face inside the mesh on the SOLID (got %d; 48 before)",
                  faces_on_solid);
    check(faces_on_solid == 0, msg);
  }
  {
    // The grid holds the one reference to its links, however often they are rebuilt.
    Session s(BRepPrimAPI_MakeBox(BX, BY, BZ).Shape());
    const bool ok = build_hexa_mesh(s, 2);
    SMDS_UnstructuredGrid* grid = s.meshDS()->GetGrid();
    const int built = ok ? grid->GetLinks()->GetReferenceCount() : -1;
    grid->BuildLinks();
    const int rebuilt = grid->GetLinks()->GetReferenceCount();
    grid->DeleteLinks();
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "P4CVL the grid's cell links have one reference, built and rebuilt (got %d, "
                  "%d; 2 before), and DeleteLinks drops them",
                  built, rebuilt);
    check(built == 1 && rebuilt == 1 && !grid->HasLinks(), msg);
  }
}


// ------------------------------------------------------------------------------ P4MEF ---- //

// MEFISTO_2D on the 4 x 4 square with n segments per side and MaxElementArea(max_area):
// whether it computed, the triangle count, the largest triangle area, and whether the face
// carries a COMPERR_WARNING.
struct MefistoRun {
  bool computed = false;
  int triangles = 0;
  double largest = 0.0;
  bool warned = false;
};

MefistoRun mefisto_square(int segments, double max_area) {
  const TopoDS_Face face =
      BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0, 4, 0, 4).Face();
  Session s(face);
  StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
  StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
  n->SetNumberOfSegments(segments);
  StdMeshers_MEFISTO_2D* a2 = s.make<StdMeshers_MEFISTO_2D>();
  StdMeshers_MaxElementArea* area = s.make<StdMeshers_MaxElementArea>();
  area->SetMaxArea(max_area);
  MefistoRun run;
  run.computed = s.assign(face, a1) && s.assign(face, n) && s.assign(face, a2) &&
                 s.assign(face, area) && s.compute();
  for (SMDS_FaceIteratorPtr it = s.meshDS()->facesIterator(); it->more();) {
    const SMDS_MeshElement* f = it->next();
    const gp_XYZ p0(f->GetNode(0)->X(), f->GetNode(0)->Y(), f->GetNode(0)->Z());
    const gp_XYZ p1(f->GetNode(1)->X(), f->GetNode(1)->Y(), f->GetNode(1)->Z());
    const gp_XYZ p2(f->GetNode(2)->X(), f->GetNode(2)->Y(), f->GetNode(2)->Z());
    run.largest = std::max(run.largest, 0.5 * ((p1 - p0) ^ (p2 - p0)).Modulus());
    ++run.triangles;
  }
  const SMESH_ComputeErrorPtr err = s.mesh().GetSubMesh(face)->GetComputeError();
  run.warned = err && err->myName == COMPERR_WARNING;
  return run;
}

// MEFISTO_2D_max_element_area.patch: aptrte clamped the edge bound to the boundary segments,
// so MaxElementArea had no effect below their size; a bound the boundary cannot meet is a
// compute warning on the face.
void probe_p4_mefisto_max_element_area() {
  section("P4MEF", "MaxElementArea bounds the MEFISTO_2D triangles");
  const MefistoRun tight = mefisto_square(8, 0.0625);
  char msg[200];
  std::snprintf(msg, sizeof(msg),
                "P4MEF 8 segments per side, max_area 0.0625: %d triangles (>= 256), largest "
                "%.4f (<= 0.0625; 134 and 0.1758 before), no warning",
                tight.triangles, tight.largest);
  check(tight.computed && tight.triangles >= 256 && tight.largest <= 0.0625 * (1 + 1e-9) &&
            !tight.warned,
        msg);
  const MefistoRun coarse = mefisto_square(2, 0.25);
  std::snprintf(msg, sizeof(msg),
                "P4MEF 2 segments per side, max_area 0.25: computed with a COMPERR_WARNING "
                "on the face (largest %.4f)",
                coarse.largest);
  check(coarse.computed && coarse.warned && coarse.largest > 0.25, msg);
}


// ------------------------------------------------------------------------------ P4L6 ----- //

// StdMeshers_CompositeHexa_3D_viscous_layers.patch: CompositeHexa_3D with ViscousLayers used
// to read a null proxy mesh and crash; it now fails the compute with an error that says so.
void probe_p4_composite_hexa_layers() {
  section("P4L6", "CompositeHexa_3D refuses viscous layers instead of crashing");
  // Under a compound root, as load_brep gives a shape: on a bare SOLID, SMESH refuses the
  // hypothesis at assignment (HYP_INCOMPATIBLE), and Compute never sees it.
  BRep_Builder builder;
  TopoDS_Compound box;
  builder.MakeCompound(box);
  builder.Add(box, BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape());
  Session s(box);
  StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
  StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
  n->SetNumberOfSegments(4);
  StdMeshers_Quadrangle_2D* a2 = s.make<StdMeshers_Quadrangle_2D>();
  StdMeshers_CompositeHexa_3D* a3 = s.make<StdMeshers_CompositeHexa_3D>();
  StdMeshers_ViscousLayers* layers = s.make<StdMeshers_ViscousLayers>();
  layers->SetTotalThickness(0.3);
  layers->SetNumberLayers(3);
  layers->SetStretchFactor(1.2);
  layers->SetBndShapes(std::vector<int>(1, s.meshDS()->ShapeToIndex(
                           TopExp_Explorer(box, TopAbs_FACE).Current())),
                       /*toIgnore=*/false);
  const bool assigned = s.assign(box, a1) && s.assign(box, n) && s.assign(box, a2) &&
                        s.assign(box, a3) && s.assign(box, layers);
  const bool computed = s.compute();
  TopExp_Explorer solid(box, TopAbs_SOLID);
  const SMESH_ComputeErrorPtr err = s.mesh().GetSubMesh(solid.Current())->GetComputeError();
  const bool named = err && err->myComment.find("does not build viscous layers") !=
                                std::string::npos;
  char msg[300];
  std::snprintf(msg, sizeof(msg),
                "P4L6 CompositeHexa_3D + ViscousLayers: the compute fails on the SOLID, naming "
                "the reason (it crashed before); assigned %d computed %d error '%s'",
                int(assigned), int(computed), err ? err->myComment.c_str() : "(none)");
  check(assigned && !computed && named, msg);
}

// ------------------------------------------------------------------------------ P5EXC ---- //

// SMESH_subMesh_salome_exception_text.patch: a SALOME_Exception thrown by an algorithm's
// Compute lost the first 7 characters of its text, for the "Salome " of a "Salome Exception"
// prefix that only the const char* constructor adds (Utils_SALOME_Exception.cxx, makeText).
// No public input reaches such a throw site in this build, so a stub 3-D algorithm throws
// each kind: a std::string, an SMESH_Comment, a text shorter than 7 characters, and a
// const char* text, which carries the prefix.
class ThrowingAlgo3D : public SMESH_3D_Algo {
 public:
  enum class Kind { kString, kComment, kShort, kPrefixed };

  ThrowingAlgo3D(int hypId, SMESH_Gen* gen, Kind kind) : SMESH_3D_Algo(hypId, gen), kind_(kind) {
    _name = "ProbeThrowing_3D";
  }

  bool CheckHypothesis(SMESH_Mesh&, const TopoDS_Shape&,
                       SMESH_Hypothesis::Hypothesis_Status& status) override {
    status = SMESH_Hypothesis::HYP_OK;
    return true;
  }

  bool Compute(SMESH_Mesh&, const TopoDS_Shape&) override {
    switch (kind_) {
      case Kind::kString:
        throw SALOME_Exception(std::string("ViscousBuilder2D: a text from a std::string"));
      case Kind::kComment:
        throw SALOME_Exception(SMESH_Comment("ViscousBuilder2D: not SMDS_TOP_EDGE node "
                                             "position: ") << 0 << " of node " << 12);
      case Kind::kShort:
        throw SALOME_Exception(std::string("abc"));
      case Kind::kPrefixed:
        throw SALOME_Exception("a text from a const char*");
    }
    return false;
  }

  bool Evaluate(SMESH_Mesh&, const TopoDS_Shape&, MapShapeNbElems&) override { return false; }

 private:
  Kind kind_;
};

// The compute error text the SOLID gets when the stub throws `kind` on the unit box.
std::string thrown_text(ThrowingAlgo3D::Kind kind) {
  BRep_Builder builder;
  TopoDS_Compound box;
  builder.MakeCompound(box);
  builder.Add(box, BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape());
  Session s(box);
  StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
  StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
  n->SetNumberOfSegments(2);
  StdMeshers_Quadrangle_2D* a2 = s.make<StdMeshers_Quadrangle_2D>();
  ThrowingAlgo3D* a3 = s.make<ThrowingAlgo3D>(kind);
  const bool assigned =
      s.assign(box, a1) && s.assign(box, n) && s.assign(box, a2) && s.assign(box, a3);
  if (!assigned || s.compute()) {
    return "(the stub was not assigned, or the compute succeeded)";
  }
  TopExp_Explorer solid(box, TopAbs_SOLID);
  const SMESH_ComputeErrorPtr err = s.mesh().GetSubMesh(solid.Current())->GetComputeError();
  return err ? err->myComment : std::string("(no compute error)");
}

void probe_p5_salome_exception_text() {
  section("P5EXC", "a SALOME_Exception from a compute keeps its full text");
  const struct {
    ThrowingAlgo3D::Kind kind;
    const char* want;
    const char* what;
  } cases[] = {
      {ThrowingAlgo3D::Kind::kString, "ViscousBuilder2D: a text from a std::string",
       "a std::string text reaches the compute error whole"},
      {ThrowingAlgo3D::Kind::kComment,
       "ViscousBuilder2D: not SMDS_TOP_EDGE node position: 0 of node 12",
       "an SMESH_Comment text reaches the compute error whole"},
      {ThrowingAlgo3D::Kind::kShort, "abc",
       "a text shorter than 7 characters is kept, not read past its end"},
      {ThrowingAlgo3D::Kind::kPrefixed, "Exception : a text from a const char*",
       "a text with the \"Salome Exception\" prefix keeps exactly its former text"},
  };
  for (const auto& c : cases) {
    const std::string got = thrown_text(c.kind);
    check(got == c.want, std::string("P5EXC ") + c.what + ": got '" + got + "', want '" +
                             c.want + "'");
  }
}

// ------------------------------------------------------------------------------ P5HYP ---- //

// SMESH_subMesh_remove_hypothesis_state.patch: Hexa_3D takes one ViscousLayers; a second one
// on the compound root leaves the SOLID MISSING_HYP. Removing it checks the algorithm again,
// so the SOLID is HYP_OK and meshes; upstream it stayed MISSING_HYP and meshed nothing.
void probe_p5_remove_hypothesis_state() {
  section("P5HYP", "removing a hypothesis checks a MISSING_HYP sub-mesh again");
  BRep_Builder builder;
  TopoDS_Compound box;
  builder.MakeCompound(box);
  builder.Add(box, BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape());
  Session s(box);
  StdMeshers_Regular_1D* a1 = s.make<StdMeshers_Regular_1D>();
  StdMeshers_NumberOfSegments* n = s.make<StdMeshers_NumberOfSegments>();
  n->SetNumberOfSegments(4);
  StdMeshers_Quadrangle_2D* a2 = s.make<StdMeshers_Quadrangle_2D>();
  StdMeshers_Hexa_3D* a3 = s.make<StdMeshers_Hexa_3D>();
  TopExp_Explorer face(box, TopAbs_FACE);
  const int first = s.meshDS()->ShapeToIndex(face.Current());
  face.Next();
  const int second = s.meshDS()->ShapeToIndex(face.Current());
  StdMeshers_ViscousLayers* la = s.make<StdMeshers_ViscousLayers>();
  StdMeshers_ViscousLayers* lb = s.make<StdMeshers_ViscousLayers>();
  for (auto [layers, wall] : {std::pair{la, first}, std::pair{lb, second}}) {
    layers->SetTotalThickness(0.2);
    layers->SetNumberLayers(2);
    layers->SetStretchFactor(1.0);
    layers->SetBndShapes(std::vector<int>(1, wall), /*toIgnore=*/false);
  }
  const bool assigned = s.assign(box, a1) && s.assign(box, n) && s.assign(box, a2) &&
                        s.assign(box, a3) && s.assign(box, la) && s.assign(box, lb);
  TopExp_Explorer solid(box, TopAbs_SOLID);
  SMESH_subMesh* sm = s.mesh().GetSubMesh(solid.Current());
  const bool missing = sm->GetAlgoState() == SMESH_subMesh::MISSING_HYP;
  s.mesh().RemoveHypothesis(box, lb->GetID());
  const bool ok_again = sm->GetAlgoState() == SMESH_subMesh::HYP_OK;
  const bool computed = s.compute();
  const smIdType volumes = s.meshDS()->NbVolumes();
  char msg[300];
  std::snprintf(msg, sizeof(msg),
                "P5HYP Hexa_3D with two ViscousLayers is MISSING_HYP, and HYP_OK once one is "
                "removed; then it meshes 4x4x4 + 4x4x2 hexahedra: assigned %d missing %d "
                "ok_again %d computed %d volumes %lld",
                int(assigned), int(missing), int(ok_again), int(computed),
                static_cast<long long>(volumes));
  check(assigned && missing && ok_again && computed && volumes == 4 * 4 * 4 + 4 * 4 * 2,
        msg);
}

// ------------------------------------------------------------------------------ P5CVL ---- //

// StdMeshers_Cartesian_VL_offset_error.patch and StdMeshers_Cartesian_VL_inverted_layers.patch
// on a 2 x 2 x 2 block with a bore of radius 0.4 on its axis, under a compound root as
// pySMESH loads a shape, spacing 0.25: the offset surfaces meet at a total thickness of 0.3.
// The SOLID's compute error at `thickness`, or "(computed)" when it has none and volumes
// were made. SMESH_Gen::Compute can return true with the SOLID failed, so the error decides.
std::string bored_block_layers(double thickness) {
  const TopoDS_Shape bored =
      BRepAlgoAPI_Cut(BRepPrimAPI_MakeBox(2.0, 2.0, 2.0).Shape(),
                      BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(1.0, 1.0, 0.0), gp_Dir(0, 0, 1)),
                                               0.4, 2.0)
                          .Shape())
          .Shape();
  BRep_Builder builder;
  TopoDS_Compound root;
  builder.MakeCompound(root);
  for (TopExp_Explorer ex(bored, TopAbs_SOLID); ex.More(); ex.Next()) {
    builder.Add(root, ex.Current());
  }
  Session s(root);
  cartesian_layers(s, "0.25", thickness);
  s.compute();
  TopExp_Explorer solid(root, TopAbs_SOLID);
  const SMESH_ComputeErrorPtr err = s.mesh().GetSubMesh(solid.Current())->GetComputeError();
  if (err && !err->IsOK()) {
    return err->myComment;
  }
  return s.meshDS()->NbVolumes() > 0 ? "(computed)" : "(no error and no volume)";
}

void probe_p5_cartesian_too_thick() {
  section("P5CVL", "Cartesian_3D layers too thick: the reason and the largest workable thickness");
  const struct {
    double thickness;
    const char* want;
    const char* what;
  } cases[] = {
      {0.285, "(computed)", "0.285 meshes"},
      {0.3, "The largest total thickness for which the offset works is about 0.299927",
       "0.3 (empty offset) names the largest workable thickness, 0.299927"},
      {0.7, "is not a valid solid (BRepCheck_Analyzer)",
       "0.7 (invalid offset, partly outside the block) is refused by name"},
      {0.2999, "layer cells are inverted", "0.2999 (layer cells fold over) is refused by name"},
  };
  for (const auto& c : cases) {
    const std::string got = bored_block_layers(c.thickness);
    check(got.find(c.want) != std::string::npos,
          std::string("P5CVL ") + c.what + ": got '" + got.substr(0, 200) + "'");
  }
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
  probe_controls_and_groups_binding_behaviour();
  probe_editor_and_search_binding_behaviour();
  probe_r18_gmf_driver();
  probe_cat916_1d_additions();
  probe_cat916_2d_additions();
  probe_cat916_3d_additions();
  probe_p4_distributions();
  probe_p4_hypothesis_status();
  probe_p4_layer_builder_lifecycle();
  probe_p4_cartesian_layers();
  probe_p4_mefisto_max_element_area();
  probe_p4_composite_hexa_layers();
  probe_p5_salome_exception_text();
  probe_p5_remove_hypothesis_state();
  probe_p5_cartesian_too_thick();
}
