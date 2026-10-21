// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH binding — Session: locating entities with GEOM's classifiers.
//
// Three read-only queries, each the session form of one GEOM operation:
//
//   entities_on — GEOM's GetShapesOnPlane / Cylinder / Sphere / Box / Shape, through
//                 GEOMAlgo_FinderShapeOn2 with the classifier settings GEOM gives it;
//   in_place    — GEOM's GetInPlace, through GEOMAlgo_GetInPlace, against old sub-shapes
//                 named by handoff ordinals in BREP bytes or by ids live in a snapshot;
//   find_same   — GEOM's GetSameIDs. Its rule is not in GEOMAlgo: isSameEdge, isSameFace and
//                 isSameSolid are copied below from GEOMImpl_IShapesOperations.cxx.
//
// None of them writes to a shape of the model. That needs care in one place: the finder
// samples a face or an edge at the nodes of its triangulation, and meshes any face that has
// none (GEOMAlgo_FinderShapeOn2::InnerPoints). On the live model it would write a
// triangulation into the session's faces, which changes the bytes brep() returns, and its
// answer would depend on whatever triangulation an earlier tessellate() left there. So
// entities_on classifies a private copy of the model, meshed as GEOM meshes it, and maps the
// copy's sub-shapes back to the model's.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include <GEOMAlgo_Clsf.hxx>
#include <GEOMAlgo_ClsfBox.hxx>
#include <GEOMAlgo_ClsfSolid.hxx>
#include <GEOMAlgo_ClsfSurf.hxx>
#include <GEOMAlgo_FinderShapeOn2.hxx>
#include <GEOMAlgo_GetInPlace.hxx>
#include <GEOMAlgo_State.hxx>
#include <GeomLib_Tool.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_Plane.hxx>
#include <Geom_SphericalSurface.hxx>
#include <IntTools_Context.hxx>
#include <Message_ProgressScope.hxx>
#include <Prs3d.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_MapOfOrientedShape.hxx>
#include <TopTools_MapOfShape.hxx>
#include <gp.hxx>

namespace pysmesh {
namespace session {

namespace {

// ---- GEOM's GetSame rule -------------------------------------------------------------- //
//
// Copied from SALOME GEOM, tag V9_16_0 (7d0a18e1554c25bca2c2d9882e0621ea3f7f9c81),
// src/GEOMImpl/GEOMImpl_IShapesOperations.cxx: MAX_TOLERANCE, isSameEdge, isSameFace and
// isSameSolid, unchanged but for the namespace they live in and the comment lines below
// each banner. Copyright (C) 2007-2026 CEA, EDF, OPEN CASCADE; Copyright (C) 2003-2007
// OPEN CASCADE, EADS/CCR, LIP6, CEA/DEN, CEDRAT, EDF R&D, LEG, PRINCIPIA R&D, BUREAU VERITAS.
// GNU Lesser General Public License version 2.1 or later; used here under LGPL-2.1.
// Recorded in PROVENANCE.md.

  const double MAX_TOLERANCE = 1.e-7;

//=======================================================================
//function : isSameEdge
//purpose  : Returns True if two edges coincide
//=======================================================================
static bool isSameEdge(const TopoDS_Edge& theEdge1, const TopoDS_Edge& theEdge2)
{
  TopoDS_Vertex V11, V12, V21, V22;
  TopExp::Vertices(theEdge1, V11, V12);
  TopExp::Vertices(theEdge2, V21, V22);
  gp_Pnt P11 = BRep_Tool::Pnt(V11);
  gp_Pnt P12 = BRep_Tool::Pnt(V12);
  gp_Pnt P21 = BRep_Tool::Pnt(V21);
  gp_Pnt P22 = BRep_Tool::Pnt(V22);
  bool coincide = false;

  //Check that ends of edges coincide
  if(P11.Distance(P21) <= MAX_TOLERANCE) {
    if(P12.Distance(P22) <= MAX_TOLERANCE) coincide =  true;
  }
  else if(P11.Distance(P22) <= MAX_TOLERANCE) {
    if(P12.Distance(P21) <= MAX_TOLERANCE) coincide = true;
  }

  if(!coincide) return false;

  if (BRep_Tool::Degenerated(theEdge1))
    if (BRep_Tool::Degenerated(theEdge2)) return true;
    else return false;
  else
    if (BRep_Tool::Degenerated(theEdge2)) return false;

  double U11, U12, U21, U22;
  Handle(Geom_Curve) C1 = BRep_Tool::Curve(theEdge1, U11, U12);
  Handle(Geom_Curve) C2 = BRep_Tool::Curve(theEdge2, U21, U22);

  //Check that both edges has the same geometry
  double range = U12-U11;
  double U = U11+ range/3.0;
  gp_Pnt P1 = C1->Value(U);     //Compute a point on one third of the edge's length
  U = U11+range*2.0/3.0;
  gp_Pnt P2 = C1->Value(U);     //Compute a point on two thirds of the edge's length

  C2 = new Geom_TrimmedCurve(C2, U21, U22);
  // vsr 04/10/2018: 0023312 - As curve may be periodic, its parameters may be normalized
  // so, we re-take them from trimmed curve
  U21 = C2->FirstParameter();
  U22 = C2->LastParameter();

  if(!GeomLib_Tool::Parameter(C2, P1, MAX_TOLERANCE, U) ||  U < U21 || U > U22)
    return false;

  if(P1.Distance(C2->Value(U)) > MAX_TOLERANCE) return false;

  if(!GeomLib_Tool::Parameter(C2, P2, MAX_TOLERANCE, U) || U < U21 || U > U22)
    return false;

  if(P2.Distance(C2->Value(U)) > MAX_TOLERANCE) return false;

  return true;
}

//=======================================================================
//function : isSameFace
//purpose  : Returns True if two faces coincide
//=======================================================================
static bool isSameFace(const TopoDS_Face& theFace1, const TopoDS_Face& theFace2)
{
  TopExp_Explorer E(theFace1, TopAbs_EDGE);
  TopTools_ListOfShape LS1, LS2;
  for(; E.More(); E.Next()) LS1.Append(E.Current());

  E.Init(theFace2, TopAbs_EDGE);
  for(; E.More(); E.Next()) LS2.Append(E.Current());

  //Compare the number of edges in the faces
  if(LS1.Extent() != LS2.Extent()) return false;

  double aMin = RealFirst(), aMax = RealLast();
  double xminB1=aMax, yminB1=aMax, zminB1=aMax, xminB2=aMax, yminB2=aMax, zminB2=aMax;
  double xmaxB1=aMin, ymaxB1=aMin, zmaxB1=aMin, xmaxB2=aMin, ymaxB2=aMin, zmaxB2=aMin;

  for(E.Init(theFace1, TopAbs_VERTEX); E.More(); E.Next()) {
    gp_Pnt P = BRep_Tool::Pnt(TopoDS::Vertex(E.Current()));
    if(P.X() < xminB1) xminB1 = P.X();
    if(P.Y() < yminB1) yminB1 = P.Y();
    if(P.Z() < zminB1) zminB1 = P.Z();
    if(P.X() > xmaxB1) xmaxB1 = P.X();
    if(P.Y() > ymaxB1) ymaxB1 = P.Y();
    if(P.Z() > zmaxB1) zmaxB1 = P.Z();
  }

  for(E.Init(theFace2, TopAbs_VERTEX); E.More(); E.Next()) {
    gp_Pnt P = BRep_Tool::Pnt(TopoDS::Vertex(E.Current()));
    if(P.X() < xminB2) xminB2 = P.X();
    if(P.Y() < yminB2) yminB2 = P.Y();
    if(P.Z() < zminB2) zminB2 = P.Z();
    if(P.X() > xmaxB2) xmaxB2 = P.X();
    if(P.Y() > ymaxB2) ymaxB2 = P.Y();
    if(P.Z() > zmaxB2) zmaxB2 = P.Z();
  }

  //Compare the bounding boxes of both faces
  if(gp_Pnt(xminB1, yminB1, zminB1).Distance(gp_Pnt(xminB2, yminB2, zminB2)) > MAX_TOLERANCE)
    return false;

  if(gp_Pnt(xmaxB1, ymaxB1, zmaxB1).Distance(gp_Pnt(xmaxB2, ymaxB2, zmaxB2)) > MAX_TOLERANCE)
    return false;

  Handle(Geom_Surface) S1 = BRep_Tool::Surface(theFace1);
  Handle(Geom_Surface) S2 = BRep_Tool::Surface(theFace2);

  //Check if there a coincidence of two surfaces at least in two points
  double U11, U12, V11, V12, U21, U22, V21, V22;
  BRepTools::UVBounds(theFace1, U11, U12, V11, V12);
  BRepTools::UVBounds(theFace2, U21, U22, V21, V22);

  double rangeU = U12-U11;
  double rangeV = V12-V11;
  double U = U11 + rangeU/3.0;
  double V = V11 + rangeV/3.0;
  gp_Pnt P1 = S1->Value(U, V);
  U = U11+rangeU*2.0/3.0;
  V = V11+rangeV*2.0/3.0;
  gp_Pnt P2 = S1->Value(U, V);

  if (!GeomLib_Tool::Parameters(S2, P1, MAX_TOLERANCE, U, V) || U < U21 || U > U22 || V < V21 || V > V22)
    return false;

  if (P1.Distance(S2->Value(U,V)) > MAX_TOLERANCE) return false;

  if (!GeomLib_Tool::Parameters(S2, P2, MAX_TOLERANCE, U, V) || U < U21 || U > U22 || V < V21 || V > V22)
    return false;

  if (P2.Distance(S2->Value(U, V)) > MAX_TOLERANCE) return false;

  //Check that each edge of the Face1 has a counterpart in the Face2
  TopTools_MapOfOrientedShape aMap;
  TopTools_ListIteratorOfListOfShape LSI1(LS1);
  for(; LSI1.More(); LSI1.Next()) {
    TopoDS_Edge E = TopoDS::Edge(LSI1.Value());
    bool isFound = false;
    TopTools_ListIteratorOfListOfShape LSI2(LS2);
    for(; LSI2.More(); LSI2.Next()) {
      TopoDS_Shape aValue = LSI2.Value();
      if(aMap.Contains(aValue)) continue; //To avoid checking already found edge several times
      if(isSameEdge(E, TopoDS::Edge(aValue))) {
        aMap.Add(aValue);
        isFound = true;
        break;
      }
    }
    if(!isFound) return false;
  }

  return true;
}

//=======================================================================
//function : isSameSolid
//purpose  : Returns True if two solids coincide
//=======================================================================
bool isSameSolid(const TopoDS_Solid& theSolid1, const TopoDS_Solid& theSolid2)
{
  TopExp_Explorer E(theSolid1, TopAbs_FACE);
  TopTools_ListOfShape LS1, LS2;
  for(; E.More(); E.Next()) LS1.Append(E.Current());
  E.Init(theSolid2, TopAbs_FACE);
  for(; E.More(); E.Next()) LS2.Append(E.Current());

  if(LS1.Extent() != LS2.Extent()) return false;

  double aMin = RealFirst(), aMax = RealLast();
  double xminB1=aMax, yminB1=aMax, zminB1=aMax, xminB2=aMax, yminB2=aMax, zminB2=aMax;
  double xmaxB1=aMin, ymaxB1=aMin, zmaxB1=aMin, xmaxB2=aMin, ymaxB2=aMin, zmaxB2=aMin;

  for(E.Init(theSolid1, TopAbs_VERTEX); E.More(); E.Next()) {
    gp_Pnt P = BRep_Tool::Pnt(TopoDS::Vertex(E.Current()));
    if(P.X() < xminB1) xminB1 = P.X();
    if(P.Y() < yminB1) yminB1 = P.Y();
    if(P.Z() < zminB1) zminB1 = P.Z();
    if(P.X() > xmaxB1) xmaxB1 = P.X();
    if(P.Y() > ymaxB1) ymaxB1 = P.Y();
    if(P.Z() > zmaxB1) zmaxB1 = P.Z();
  }

  for(E.Init(theSolid2, TopAbs_VERTEX); E.More(); E.Next()) {
    gp_Pnt P = BRep_Tool::Pnt(TopoDS::Vertex(E.Current()));
    if(P.X() < xminB2) xminB2 = P.X();
    if(P.Y() < yminB2) yminB2 = P.Y();
    if(P.Z() < zminB2) zminB2 = P.Z();
    if(P.X() > xmaxB2) xmaxB2 = P.X();
    if(P.Y() > ymaxB2) ymaxB2 = P.Y();
    if(P.Z() > zmaxB2) zmaxB2 = P.Z();
  }

  //Compare the bounding boxes of both solids
  if(gp_Pnt(xminB1, yminB1, zminB1).Distance(gp_Pnt(xminB2, yminB2, zminB2)) > MAX_TOLERANCE)
    return false;

  if(gp_Pnt(xmaxB1, ymaxB1, zmaxB1).Distance(gp_Pnt(xmaxB2, ymaxB2, zmaxB2)) > MAX_TOLERANCE)
    return false;

  //Check that each face of the Solid1 has a counterpart in the Solid2
  TopTools_MapOfOrientedShape aMap;
  TopTools_ListIteratorOfListOfShape LSI1(LS1);
  for(; LSI1.More(); LSI1.Next()) {
    TopoDS_Face F = TopoDS::Face(LSI1.Value());
    bool isFound = false;
    TopTools_ListIteratorOfListOfShape LSI2(LS2);
    for(; LSI2.More(); LSI2.Next()) {
      if(aMap.Contains(LSI2.Value())) continue; //To avoid checking already found faces several times
      if(isSameFace(F, TopoDS::Face(LSI2.Value()))) {
        aMap.Add(LSI2.Value());
        isFound = true;
        break;
      }
    }
    if(!isFound) return false;
  }

  return true;
}

// ---- end of the copy from GEOM ---------------------------------------------------------- //

// GEOM's GetSameIDs test for one candidate of the kind of `what` (the vertex case is its
// inline distance test).
bool same_as(const TopoDS_Shape& what, const TopoDS_Shape& candidate) {
  switch (what.ShapeType()) {
    case TopAbs_VERTEX:
      return BRep_Tool::Pnt(TopoDS::Vertex(what))
                 .Distance(BRep_Tool::Pnt(TopoDS::Vertex(candidate))) <= MAX_TOLERANCE;
    case TopAbs_EDGE:
      return isSameEdge(TopoDS::Edge(what), TopoDS::Edge(candidate));
    case TopAbs_FACE:
      return isSameFace(TopoDS::Face(what), TopoDS::Face(candidate));
    case TopAbs_SOLID:
      return isSameSolid(TopoDS::Solid(what), TopoDS::Solid(candidate));
    default:
      return false;
  }
}

// ---- entities_on ------------------------------------------------------------------------ //

// GEOM's ceiling on the classification tolerance of GetShapesOnBox (its fixed value) and of
// GetShapesOnShape (its bound on the largest vertex tolerance).
constexpr double kGeomBoxTolerance = 1.0e-4;

// GEOMUtils::CheckTriangulation's mesh, which GEOM builds before every non-vertex search:
// GEOMAlgo_AlgoTools::MeshShape with the default deflection 0.001, relative to the shape's
// box, and the angle 0.349066 rad. mesh_as_geom below takes the same steps.
constexpr double kGeomMeshDeflection = 0.001;
constexpr double kGeomMeshAngle = 0.349066;

// The shares of entities_on's progress range: the copy, the mesh and the finder. They are
// the split measured on a FACE search of a model of 117 solids and 5606 faces: copy 0.17 s,
// mesh 1.5 s, finder 0.7 s.
constexpr double kCopyShare = 1.0;
constexpr double kMeshShare = 6.0;
constexpr double kFindShare = 3.0;

// The finder runs on this many chunks of the sub-shapes of the searched kind, in TopExp
// order, so that a cancel stops it between two chunks: a FACE search of the same model
// against a solid region, state "out", spends 12 s in the finder. GEOMAlgo_FinderShapeOn2
// decides each sub-shape from its own vertices, edges and inner points (ProcessVertices to
// ProcessSolids), so the chunks give the answer of one run over the whole copy. Each chunk
// gets a fresh classifier, as a run does: GEOMAlgo_ClsfSolid::CheckData builds its
// BRepClass3d_SolidClassifier in every Perform, so the count bounds that cost. All chunks
// share one IntTools_Context.
constexpr int kFinderChunks = 32;

GEOMAlgo_State state_of(const std::string& state) {
  if (state == "on") return GEOMAlgo_ST_ON;
  if (state == "in") return GEOMAlgo_ST_IN;
  if (state == "out") return GEOMAlgo_ST_OUT;
  if (state == "on_in") return GEOMAlgo_ST_ONIN;
  if (state == "on_out") return GEOMAlgo_ST_ONOUT;
  throw PysmeshError("Session.entities_on: the state '" + state +
                     "' is not one of on, in, out, on_in, on_out.");
}

// The meaning of GEOMAlgo_FinderShapeOn2's error codes, from the list at the end of
// GEOMAlgo_FinderShapeOn2.cxx.
std::string finder_error_text(int code) {
  switch (code) {
    case 10: return "no classifier";
    case 11: return "no shape to search";
    case 12: return "a sub-shape kind the finder does not take";
    case 13: return "a state the finder does not take";
    case 15: return "a surface type the classifier does not take";
    case 20: return "an edge has no triangulation and no 3-D polygon to sample";
    case 30: return "a triangulation link gives no line";
    case 40: return "a sample point cannot be classified";
    case 41: return "the region is not valid for its classifier";
    case 42: return "the face hatching failed";
    default: return "an undocumented error";
  }
}

// The largest vertex tolerance of a shape, as GEOM's GetShapesOn* compute it; empty when the
// shape has no vertex.
std::optional<double> max_vertex_tolerance(const TopoDS_Shape& shape) {
  std::optional<double> out;
  for (TopExp_Explorer v(shape, TopAbs_VERTEX); v.More(); v.Next()) {
    const double t = BRep_Tool::Tolerance(TopoDS::Vertex(v.Current()));
    if (!out || t > *out) {
      out = t;
    }
  }
  return out;
}

// The steps of GEOMAlgo_AlgoTools::MeshShape, with GEOMAlgo_AlgoTools_CanBeMeshed, for the
// arguments GEOMUtils::CheckTriangulation gives it: the deflection 0.001 relative to the
// shape's box, the angle 0.349066 rad, theForced false, doPostCheck false. Adapted from
// SALOME common_geometry_lib, tag V9_16_0 (e7b5227096fef835a448004ee06cf5938850b1c4),
// src/GEOMAlgo/GEOMAlgo_AlgoTools.cxx. Copyright (C) 2007-2026 CEA, EDF, OPEN CASCADE;
// Copyright (C) 2003-2007 OPEN CASCADE, EADS/CCR, LIP6, CEA/DEN, CEDRAT, EDF R&D, LEG,
// PRINCIPIA R&D, BUREAU VERITAS. GNU Lesser General Public License version 2.1 or later;
// used here under LGPL-2.1. Recorded in PROVENANCE.md.
//
// One change: the mesher gets a progress range, so a cancel stops it. GEOM constructs
// BRepMesh_IncrementalMesh(shape, deflection, false, angle), which meshes at once and takes
// no range. The four parameters set here are the ones that constructor sets, and the rest
// keep their defaults in both. Returns false where MeshShape does: the shape has no face and
// no edge, or its box is void.
bool mesh_as_geom(const TopoDS_Shape& shape, const Message_ProgressRange& range) {
  // GEOMAlgo_AlgoTools_CanBeMeshed with theCheckMesh true.
  bool meshed = true;
  TopLoc_Location location;
  TopExp_Explorer ex(shape, TopAbs_FACE);
  if (ex.More()) {
    for (; ex.More() && meshed; ex.Next()) {
      meshed = !BRep_Tool::Triangulation(TopoDS::Face(ex.Current()), location).IsNull();
    }
  } else {
    ex.Init(shape, TopAbs_EDGE);
    if (!ex.More()) {
      return false;
    }
    for (; ex.More() && meshed; ex.Next()) {
      meshed = !BRep_Tool::Polygon3D(TopoDS::Edge(ex.Current()), location).IsNull();
    }
  }
  if (meshed) {
    return true;
  }
  Bnd_Box box;
  BRepBndLib::Add(shape, box);
  if (box.IsVoid()) {
    return false;
  }
  const double deflection =
      Prs3d::GetDeflection(box, kGeomMeshDeflection, kGeomMeshDeflection);
  BRepTools::Clean(shape);
  BRepMesh_IncrementalMesh mesher;
  mesher.SetShape(shape);
  IMeshTools_Parameters& params = mesher.ChangeParameters();
  params.Deflection = deflection;
  params.Angle = kGeomMeshAngle;
  params.Relative = false;
  params.InParallel = false;
  mesher.Perform(range);
  return true;
}

// ---- in_place --------------------------------------------------------------------------- //

// One GEOMAlgo_GetInPlace run of one entry's old shapes in `where`, as GEOM's GetInPlace
// makes it: the two-argument constructor computes GEOM's tolerances and performs the
// search. Returns the found sub-shapes of `where`, which GEOM keeps only when `where`
// indexes them.
//
// The old shapes go in as one compound, the form GEOM passes a group in. For a compound
// argument GEOMAlgo_GetInPlace::FillImgComplex adds, for each old shape that holds no shape
// of `where`, the shape of `where` that holds it ("find the whole from parts"). A bare face
// gets only the shapes of `where` that lie in it, so a face that a merge made part of a
// larger one would be found nowhere.
std::vector<TopoDS_Shape> found_in_place(const TopoDS_Shape& where,
                                         const ShapeSet& where_index,
                                         const std::vector<TopoDS_Shape>& olds, int& error) {
  TopoDS_Compound what;
  BRep_Builder builder;
  builder.MakeCompound(what);
  for (const TopoDS_Shape& s : olds) {
    builder.Add(what, s);
  }
  GEOMAlgo_GetInPlace gip(where, what);
  error = gip.ErrorStatus();
  std::vector<TopoDS_Shape> out;
  if (error != 0) {
    return out;
  }
  const TopoDS_Shape& result = gip.Result();
  if (result.IsNull()) {
    return out;
  }
  for (TopoDS_Iterator it(result); it.More(); it.Next()) {
    if (where_index.Contains(it.Value())) {
      append_unique(out, it.Value());
    }
  }
  return out;
}

}  // namespace

// ---- entities_on ------------------------------------------------------------------------ //

py::array_t<std::int64_t> Session::entities_on(const std::string& kind,
                                               const std::string& region,
                                               const std::vector<double>& values,
                                               const std::optional<EntityId>& solid_id,
                                               const std::string& state,
                                               const std::optional<double>& tolerance,
                                               const py::object& progress,
                                               const py::object& cancel) const {
  const char* op = "entities_on";
  const TopAbs_ShapeEnum k = kind_from_name(kind);
  const GEOMAlgo_State wanted = state_of(state);
  const ProgressHooks hooks = hooks_of(op, progress, cancel);
  require_finite(std::string("Session.") + op, "the region", values.data(), values.size(), 1);
  if (tolerance.has_value()) {
    finite_arg(op, "tolerance", *tolerance);
    if (!(*tolerance > 0.0)) {
      throw PysmeshError("Session.entities_on: the tolerance " + number_text(*tolerance) +
                         " is not above zero; no point is on a region at that tolerance.");
    }
  }
  const auto need = [&](std::size_t n) {
    if (values.size() != n) {
      throw PysmeshError("Session.entities_on: a " + region + " region takes " +
                         std::to_string(n) + " numbers (got " +
                         std::to_string(values.size()) + ").");
    }
  };

  const TopoDS_Shape root = state_.root;
  Handle(Geom_Surface) surface;
  TopoDS_Shape box;
  TopoDS_Shape region_solid;
  if (region == "plane") {
    need(6);
    const gp_Dir normal = direction_of(op, "the plane normal", values[3], values[4], values[5]);
    surface = new Geom_Plane(gp_Pnt(values[0], values[1], values[2]), normal);
  } else if (region == "cylinder") {
    need(7);
    const gp_Dir axis = direction_of(op, "the cylinder direction", values[3], values[4],
                                     values[5]);
    if (!(values[6] > 0.0)) {
      throw PysmeshError("Session.entities_on: the cylinder radius " + number_text(values[6]) +
                         " is not above zero.");
    }
    surface = new Geom_CylindricalSurface(
        gp_Ax3(gp_Pnt(values[0], values[1], values[2]), axis), values[6]);
  } else if (region == "sphere") {
    need(4);
    if (!(values[3] > 0.0)) {
      throw PysmeshError("Session.entities_on: the sphere radius " + number_text(values[3]) +
                         " is not above zero.");
    }
    // GEOM's GetShapesOnSphere frame: the centre and the global z axis.
    surface = new Geom_SphericalSurface(gp_Ax3(gp_Pnt(values[0], values[1], values[2]), gp::DZ()),
                                        values[3]);
  } else if (region == "box") {
    need(6);
    for (int i = 0; i < 3; ++i) {
      if (!(values[3 + i] > values[i])) {
        throw PysmeshError("Session.entities_on: the box extends " +
                           number_text(values[3 + i] - values[i]) + " along " +
                           std::string(1, static_cast<char>('x' + i)) +
                           " (from " + number_text(values[i]) + " to " +
                           number_text(values[3 + i]) +
                           "); a box region has a positive extent along every axis.");
      }
    }
    box = BRepPrimAPI_MakeBox(gp_Pnt(values[0], values[1], values[2]),
                              gp_Pnt(values[3], values[4], values[5]))
              .Shape();
  } else if (region == "solid") {
    need(0);
    if (!solid_id.has_value()) {
      throw PysmeshError("Session.entities_on: a solid region names no solid.");
    }
    const EntityRecord& rec = require_alive(op, *solid_id);
    if (rec.kind != TopAbs_SOLID) {
      throw PysmeshError("Session.entities_on: the region entity " +
                         std::to_string(*solid_id) + " is a " + kind_name(rec.kind) +
                         ", not a SOLID.");
    }
    if (rec.shapes.size() != 1) {
      throw PysmeshError("Session.entities_on: the region solid " + std::to_string(*solid_id) +
                         " denotes " + std::to_string(rec.shapes.size()) +
                         " solids (it was split); a region is one solid.");
    }
    // The root's own copy, for its orientation: a reversed solid classifies inside out.
    ShapeSet solids;
    TopExp::MapShapes(root, TopAbs_SOLID, solids);
    const int index = solids.FindIndex(rec.shapes.front());
    if (index == 0) {
      throw PysmeshError("Session.entities_on: solid " + std::to_string(*solid_id) +
                         " is alive in the registry but is not a solid of the session root. "
                         "The registry and the root have diverged; this is a bug, not a "
                         "caller error.");
    }
    region_solid = solids.FindKey(index);
  } else {
    throw PysmeshError("Session.entities_on: the region type '" + region +
                       "' is not one of plane, cylinder, sphere, box, solid.");
  }

  const std::vector<EntityId> ids = ids_of_kind(k);
  if (ids.empty()) {
    return ids_array({});
  }

  // GEOM's tolerance for each classifier, over the shape it searches: the whole model here.
  double tol = 0.0;
  if (tolerance.has_value()) {
    tol = *tolerance;
  } else if (!box.IsNull()) {
    tol = kGeomBoxTolerance;
  } else {
    const std::optional<double> largest = max_vertex_tolerance(root);
    if (!largest) {
      throw PysmeshError("Session.entities_on: the model has no vertex, so GEOM's tolerance "
                         "for this region, the largest vertex tolerance of the model, is "
                         "not defined.");
    }
    tol = region_solid.IsNull() ? *largest
                                : std::min(std::max(Precision::Confusion(), *largest),
                                           kGeomBoxTolerance);
  }

  ProgressDriver driver(op, hooks);
  ShapeSet found;
  int error = 0;
  bool unmeshed = false;
  {
    py::gil_scoped_release release;
    try {
      Message_ProgressScope scope(driver.range(), "entities_on",
                                  kCopyShare + kMeshShare + kFindShare);
      // A copy with its own geometry and no triangulation, so that meshing it writes nothing
      // into the model and the sample points do not depend on an earlier tessellate().
      BRepBuilderAPI_Copy copier(root, /*copyGeom=*/true, /*copyMesh=*/false);
      const TopoDS_Shape copy = copier.Shape();
      ShapeKeyed<TopoDS_Shape> original_of;
      ShapeSet originals;
      TopExp::MapShapes(root, k, originals);
      for (int i = 1; i <= originals.Extent(); ++i) {
        original_of.emplace(copier.ModifiedShape(originals.FindKey(i)), originals.FindKey(i));
      }
      scope.Next(kCopyShare);
      if (k != TopAbs_VERTEX) {
        unmeshed = !mesh_as_geom(copy, scope.Next(kMeshShare));
      } else {
        scope.Next(kMeshShare);
      }
      Message_ProgressRange find_range = scope.Next(kFindShare);
      // A cancelled mesh leaves faces without sample points: the finder does not run then.
      if (!unmeshed && scope.More()) {
        Handle(GEOMAlgo_Clsf) clsf;
        if (!surface.IsNull()) {
          Handle(GEOMAlgo_ClsfSurf) on_surface = new GEOMAlgo_ClsfSurf;
          on_surface->SetSurface(surface);
          clsf = on_surface;
        } else if (!box.IsNull()) {
          Handle(GEOMAlgo_ClsfBox) in_box = new GEOMAlgo_ClsfBox;
          in_box->SetBox(box);
          clsf = in_box;
        } else {
          Handle(GEOMAlgo_ClsfSolid) in_solid = new GEOMAlgo_ClsfSolid;
          in_solid->SetShape(copier.ModifiedShape(region_solid));
          clsf = in_solid;
        }
        const Handle(IntTools_Context) context = new IntTools_Context;
        ShapeSet subs;
        TopExp::MapShapes(copy, k, subs);
        const int size = (subs.Extent() + kFinderChunks - 1) / kFinderChunks;
        const int chunks = size == 0 ? 0 : (subs.Extent() + size - 1) / size;
        Message_ProgressScope finding(find_range, "entities_on",
                                      static_cast<double>(std::max(chunks, 1)));
        BRep_Builder builder;
        for (int c = 0; c < chunks && error == 0 && finding.More(); ++c, finding.Next()) {
          TopoDS_Compound part;
          builder.MakeCompound(part);
          const int last = std::min(subs.Extent(), (c + 1) * size);
          for (int i = c * size + 1; i <= last; ++i) {
            builder.Add(part, subs.FindKey(i));
          }
          GEOMAlgo_FinderShapeOn2 finder;
          finder.SetContext(context);
          finder.SetClsf(clsf);
          if (!surface.IsNull()) {
            // GEOM's getShapesOnSurfaceIDs: at least 3 inner points on a face that has none
            // of its own, at most 100 per edge or face.
            finder.SetNbPntsMin(3);
            finder.SetNbPntsMax(100);
          }
          finder.SetShape(part);
          finder.SetTolerance(tol);
          finder.SetShapeType(k);
          finder.SetState(wanted);
          finder.Perform();
          error = finder.ErrorStatus();
          if (error == 0) {
            for (const TopoDS_Shape& s : finder.Shapes()) {
              const auto it = original_of.find(s);
              if (it == original_of.end()) {
                error = -1;
                break;
              }
              found.Add(it->second);
            }
          }
        }
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.entities_on: OCCT threw: ") + e.what());
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled(op);
  }
  if (unmeshed) {
    throw PysmeshError("Session.entities_on: the model has no face and no edge to mesh, or "
                       "its box is void, so GEOM's triangulation of it fails (GEOM: \"Cannot "
                       "build triangulation on the shape\").");
  }
  if (error == -1) {
    throw PysmeshError("Session.entities_on: the finder returned a sub-shape that is not "
                       "the copy of one in the model. This is a bug, not a caller error.");
  }
  if (error != 0) {
    throw PysmeshError("Session.entities_on: GEOMAlgo_FinderShapeOn2 failed with code " +
                       std::to_string(error) + " (" + finder_error_text(error) + ") on the " +
                       region + " region, at the tolerance " + number_text(tol) + ".");
  }

  // An id is in the state when every shape it denotes is: a split id whose pieces lie on
  // both sides of a plane is neither in nor out.
  std::vector<EntityId> hits;
  for (EntityId id : ids) {
    const EntityRecord& rec = state_.registry->alive.at(id);
    bool all = true;
    for (const TopoDS_Shape& s : rec.shapes) {
      all = all && found.Contains(s);
    }
    if (all) {
      hits.push_back(id);
    }
  }
  return ids_array(hits);
}

// ---- in_place --------------------------------------------------------------------------- //

namespace {

// The answers of every GetInPlace run, one id array per old entry, ascending.
struct InPlaceRun {
  std::vector<std::vector<TopoDS_Shape>> found;
  std::size_t failed_entry = 0;
  int error = 0;
};

}  // namespace

static InPlaceRun run_in_place(const TopoDS_Shape& where,
                               const std::vector<std::vector<TopoDS_Shape>>& whats,
                               ProgressDriver& driver) {
  InPlaceRun run;
  run.found.resize(whats.size());
  ShapeSet where_index;
  TopExp::MapShapes(where, where_index);
  Message_ProgressScope scope(driver.range(), "in_place", static_cast<double>(whats.size()));
  for (std::size_t i = 0; i < whats.size() && scope.More(); ++i, scope.Next()) {
    int error = 0;
    run.found[i] = found_in_place(where, where_index, whats[i], error);
    if (error != 0) {
      run.error = error;
      run.failed_entry = i;
      return run;
    }
  }
  return run;
}

py::list Session::in_place_brep(const std::string& kind, const py::bytes& data,
                                const std::vector<std::int64_t>& ordinals,
                                const py::object& progress, const py::object& cancel) const {
  const char* op = "in_place";
  const TopAbs_ShapeEnum k = kind_from_name(kind);
  const ProgressHooks hooks = hooks_of(op, progress, cancel);
  if (ordinals.empty()) {
    throw PysmeshError("Session.in_place: no old " + kind + " is named; there is nothing "
                       "to find.");
  }
  TopoDS_Shape old;
  {
    const std::string bytes = data;
    std::istringstream stream(bytes);
    try {
      BRep_Builder builder;
      BRepTools::Read(old, stream, builder);
    } catch (const std::exception& e) {
      throw PysmeshError(std::string("Session.in_place: the source bytes are not a BREP OCCT "
                                     "can read: ") + e.what());
    }
  }
  if (old.IsNull()) {
    throw PysmeshError("Session.in_place: the source bytes are not a BREP OCCT can read.");
  }
  ShapeSet order;
  handoff_order(old, k, order);
  std::vector<std::vector<TopoDS_Shape>> whats;
  for (std::int64_t o : ordinals) {
    if (o < 0 || o >= order.Extent()) {
      throw PysmeshError("Session.in_place: the ordinal " + std::to_string(o) +
                         " is out of range: the source holds " +
                         std::to_string(order.Extent()) + " " + kind + "(s), ordinals 0.." +
                         std::to_string(order.Extent() - 1) + ".");
    }
    whats.push_back({order.FindKey(static_cast<int>(o) + 1)});
  }
  return in_place_answer(op, kind, whats, hooks, [&](std::size_t i) {
    return "the source " + kind + " at ordinal " + std::to_string(ordinals[i]);
  });
}

py::list Session::in_place_snapshot(const std::string& kind, std::int64_t mark,
                                    const std::vector<EntityId>& ids,
                                    const py::object& progress,
                                    const py::object& cancel) const {
  const char* op = "in_place";
  const TopAbs_ShapeEnum k = kind_from_name(kind);
  const ProgressHooks hooks = hooks_of(op, progress, cancel);
  if (mark < 0 || static_cast<std::size_t>(mark) >= snapshots_.size()) {
    throw PysmeshError("Session.in_place: the snapshot mark " + std::to_string(mark) +
                       " is unknown.");
  }
  const std::optional<SessionState>& snap = snapshots_[static_cast<std::size_t>(mark)];
  if (!snap.has_value()) {
    throw PysmeshError("Session.in_place: the snapshot mark " + std::to_string(mark) +
                       " was discarded.");
  }
  if (ids.empty()) {
    throw PysmeshError("Session.in_place: no old " + kind + " is named; there is nothing "
                       "to find.");
  }
  std::vector<std::vector<TopoDS_Shape>> whats;
  for (EntityId id : ids) {
    const auto it = snap->registry->alive.find(id);
    if (it == snap->registry->alive.end()) {
      throw PysmeshError("Session.in_place: entity " + std::to_string(id) +
                         " is not live in the snapshot " + std::to_string(mark) + ".");
    }
    if (it->second.kind != k) {
      throw PysmeshError("Session.in_place: entity " + std::to_string(id) + " is a " +
                         kind_name(it->second.kind) + " in the snapshot " +
                         std::to_string(mark) + ", not a " + kind + ".");
    }
    whats.push_back(it->second.shapes);
  }
  return in_place_answer(op, kind, whats, hooks, [&](std::size_t i) {
    return "entity " + std::to_string(ids[i]) + " of the snapshot " + std::to_string(mark);
  });
}

py::list Session::in_place_answer(const char* op, const std::string& kind,
                                  const std::vector<std::vector<TopoDS_Shape>>& whats,
                                  const ProgressHooks& hooks,
                                  const std::function<std::string(std::size_t)>& entry) const {
  const TopAbs_ShapeEnum k = kind_from_name(kind);
  const TopoDS_Shape where = state_.root;
  ProgressDriver driver(op, hooks);
  InPlaceRun run;
  {
    py::gil_scoped_release release;
    try {
      run = run_in_place(where, whats, driver);
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.") + op + ": OCCT threw: " + e.what());
    }
  }
  driver.finish();
  if (driver.cancelled()) {
    ProgressDriver::raise_cancelled(op);
  }
  if (run.error != 0) {
    throw PysmeshError(std::string("Session.") + op +
                       ": GEOMAlgo_GetInPlace failed with code " + std::to_string(run.error) +
                       " on " + entry(run.failed_entry) + ".");
  }
  py::list out;
  for (const std::vector<TopoDS_Shape>& shapes : run.found) {
    std::vector<EntityId> found;
    for (const TopoDS_Shape& s : shapes) {
      // GetInPlace's images of a simple shape are of its kind (GEOMAlgo_GetInPlace::
      // FillImgSimple keeps only those).
      if (s.ShapeType() != k) {
        throw PysmeshError(std::string("Session.") + op + ": GEOMAlgo_GetInPlace returned a " +
                           TopAbs::ShapeTypeToString(s.ShapeType()) + " for a " + kind +
                           ". This is a bug, not a caller error.");
      }
      ids_on(s, found);
    }
    std::sort(found.begin(), found.end());
    found.erase(std::unique(found.begin(), found.end()), found.end());
    out.append(ids_array(found));
  }
  return out;
}

// ---- find_same -------------------------------------------------------------------------- //

py::array_t<std::int64_t> Session::find_same(EntityId entity_id,
                                             const std::vector<EntityId>& within) const {
  const char* op = "find_same";
  const EntityRecord& rec = require_alive(op, entity_id);
  if (rec.shapes.size() != 1) {
    throw PysmeshError("Session.find_same: entity " + std::to_string(entity_id) +
                       " denotes " + std::to_string(rec.shapes.size()) + " " +
                       kind_name(rec.kind) + "s (it was split); GEOM's rule compares one.");
  }
  const TopoDS_Shape what = rec.shapes.front();
  const std::vector<TopoDS_Shape> bodies = owner_bodies(op, within);

  std::vector<TopoDS_Shape> same;
  {
    py::gil_scoped_release release;
    try {
      // GetSameIDs explores the shape once, skipping a sub-shape it met already.
      TopTools_MapOfShape seen;
      for (const TopoDS_Shape& body : bodies) {
        for (TopExp_Explorer e(body, rec.kind); e.More(); e.Next()) {
          if (seen.Add(e.Current()) && same_as(what, e.Current())) {
            same.push_back(e.Current());
          }
        }
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.find_same: OCCT threw: ") + e.what());
    }
  }
  std::vector<EntityId> ids;
  for (const TopoDS_Shape& s : same) {
    ids_on(s, ids);
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids_array(ids);
}

}  // namespace session
}  // namespace pysmesh
