// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH binding — Session: GEOM algorithms copied into pySMESH (curves and surfaces).
//
//   fillet_1d          GEOM's MakeFillet1D (GEOMImpl_Fillet1dDriver with GEOMImpl_Fillet1d);
//   smoothing_surface  GEOM's MakeSmoothingSurface (AdvancedEngine_SmoothingSurfaceDriver).
//
// The GEOM code that these need lives in GEOM's driver layer, which pySMESH does not vendor
// (it is built on OCAF documents and GEOM_Function). The algorithm classes are copied whole
// (geom_fillet1d.cpp, geom_edge_from_wire.cpp); the driver logic is copied here, without
// the driver shell, and records the history the id registry carries ids across. Every
// copied block names its origin. All are SALOME GEOM tag V9_16_0
// (7d0a18e1554c25bca2c2d9882e0621ea3f7f9c81), Copyright (C) 2007-2026 CEA, EDF, OPEN
// CASCADE (the smoothing surface: 2013-2026), LGPL-2.1 or later, used here under LGPL-2.1,
// and recorded in PROVENANCE.md.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

#include "geom_edge_from_wire.hpp"
#include "geom_fillet1d.hpp"

#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepLib_FindSurface.hxx>
#include <GeomLib.hxx>
#include <Geom_BSplineSurface.hxx>
#include <GeomPlate_BuildAveragePlane.hxx>
#include <GeomPlate_BuildPlateSurface.hxx>
#include <GeomPlate_MakeApprox.hxx>
#include <GeomPlate_PointConstraint.hxx>
#include <GeomPlate_Surface.hxx>
#include <Geom_Plane.hxx>
#include <ShapeAnalysis_Edge.hxx>
#include <ShapeExtend_WireData.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeFix_ShapeTolerance.hxx>
#include <ShapeFix_Wire.hxx>
#include <StdFail_NotDone.hxx>
#include <TColgp_HArray1OfPnt.hxx>
#include <TopTools_DataMapOfShapeShape.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_MapOfShape.hxx>
#include <gp.hxx>

#include <set>

namespace pysmesh {
namespace session {

namespace {

// The ids, ascending and each once, as "1, 2, 3".
std::string id_text(std::vector<EntityId> ids) {
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  std::string out;
  for (EntityId id : ids) {
    out += (out.empty() ? "" : ", ") + std::to_string(id);
  }
  return out;
}

// ---- fillet_1d: the driver logic of GEOMImpl_Fillet1dDriver.cxx ---------------------- //

// anotherVertex, takePlane and addEdgeRelation are GEOMImpl_Fillet1dDriver.cxx's static
// functions, copied unchanged.

//=======================================================================
//function : anotherVertex
//purpose  : local function to get vertex from edge
//=======================================================================
static TopoDS_Vertex anotherVertex( const TopoDS_Edge& theE,
                                    const TopoDS_Vertex& theV )
{
  // here is an assumption that edge has different vertices
  TopoDS_Vertex aV;
  TopExp_Explorer anExp( theE, TopAbs_VERTEX );
  for ( ; anExp.More(); anExp.Next() )
  {
    if ( BRepTools::Compare(theV,TopoDS::Vertex(anExp.Current())) /*theV.IsSame(anExp.Current())*/ )
      continue;
    aV = TopoDS::Vertex( anExp.Current() );
    break;
  }
  return aV;
}

//=======================================================================
//function : takePlane
//purpose  : local function returns plane of given edges
//=======================================================================
static Standard_Boolean takePlane( const TopoDS_Edge& theE1,
                                   const TopoDS_Edge& theE2,
                                   const TopoDS_Vertex& theV,
                                   gp_Pln& thePlane )
{
  TopoDS_Vertex aV12 = anotherVertex( theE1, theV );
  TopoDS_Vertex aV22 = anotherVertex( theE2, theV );
  // check can closed wire be created by two initial edges
  if ( aV12.IsNull()  || aV22.IsNull() || aV12.IsSame( aV22 ) )
    return false;

  // create plane by 3 points
  gp_XYZ aXYZ = BRep_Tool::Pnt( theV ).XYZ();
  gp_XYZ aXYZ1 = BRep_Tool::Pnt( aV12 ).XYZ();
  gp_XYZ aXYZ2 = BRep_Tool::Pnt( aV22 ).XYZ();
  try {
    gp_Dir aDir1( aXYZ - aXYZ1 );
    gp_Dir aDir2( aXYZ2 - aXYZ );
    Standard_Real anAngle = aDir1.Angle(aDir2);
    if ( fabs(anAngle) <= gp::Resolution() ||
         fabs(anAngle - M_PI) <= gp::Resolution() )
      return false;
    thePlane = gp_Pln( gp_Pnt(aXYZ), aDir1^ aDir2);
  }
  catch (Standard_Failure&) {
    return false;
  }
  return true;
}

//=======================================================================
//function : addEdgeRelation
//purpose  : local function to remember relation between initial and modified edge
//=======================================================================
static void addEdgeRelation(TopTools_DataMapOfShapeShape& theMap,
                            const TopoDS_Edge& theInitE,
                            const TopoDS_Edge& theResE)
{
  if ( theMap.IsBound( theInitE ) )
    theMap.ChangeFind( theInitE ) = theResE;
  else
    theMap.Bind( theInitE, theResE );
}

// Why a vertex of a wire is not a corner a 1-D fillet rounds, or empty when it is one. A
// corner joins exactly two edges of the wire, which GEOM's takePlane takes a plane from
// (their far ends and the vertex are not on one line), and which do not continue each other
// with C1 continuity there (GEOM's AreEdgesC1). GEOM's driver skips the first two kinds of
// vertex; the third is the join of a tangent arc, or of collinear edges, where no arc fits.
std::string not_a_corner(const TopTools_IndexedDataMapOfShapeListOfShape& vertex_edges,
                         const TopoDS_Vertex& v) {
  const int index = vertex_edges.FindIndex(v);
  if (index == 0) {
    return "it is not a vertex of the wire";
  }
  const TopTools_ListOfShape& edges = vertex_edges.FindFromIndex(index);
  if (edges.Extent() != 2) {
    return "it ends the open wire; it joins " + std::to_string(edges.Extent()) +
           " edge(s), not two";
  }
  const TopoDS_Edge e1 = TopoDS::Edge(edges.First());
  const TopoDS_Edge e2 = TopoDS::Edge(edges.Last());
  if (e1.IsSame(e2)) {
    return "it joins the two ends of one closed edge";
  }
  gp_Pln plane;
  if (!takePlane(e1, e2, v, plane)) {
    return "the two edges that meet there lie on one line through it";
  }
  if (geomcopy::AreEdgesC1(e1, e2)) {
    return "the two edges that meet there continue each other with a common tangent";
  }
  return std::string();
}

// One fillet pass over a wire: GEOMImpl_Fillet1dDriver::MakeFillet.
struct FilletPass {
  TopoDS_Wire wire;                 // the result; null when a corner failed
  Handle(BRepTools_History) hist;   // from the pass's input wire to `wire`
  int failed = -1;                  // the index of the corner that failed
  std::string reason;               // why it failed
};

// The edges of a ShapeExtend_WireData, by position.
std::vector<TopoDS_Edge> wire_data_edges(const Handle(ShapeExtend_WireData)& data) {
  std::vector<TopoDS_Edge> out;
  for (int i = 1; i <= data->NbEdges(); ++i) {
    out.push_back(data->Edge(i));
  }
  return out;
}

// The history of ShapeFix_Wire::FixConnected, read from its wire data by position.
// FixConnected never adds, removes or moves an edge: it replaces the edges at a junction by
// copies whose end vertex is the junction's new vertex (ShapeBuild_Edge::CopyReplaceVertices)
// and writes them back at the same position (ShapeExtend_WireData::Set). So the edge at a
// position after the call is the image of the edge at that position before it, and the end
// vertices at a junction after the call are the images of those before it. (GEOM's call
// gives the fixer no context, and a context changes FixConnected's flow, so the history is
// read this way rather than from a context.)
Handle(BRepTools_History) connect_history(const std::vector<TopoDS_Edge>& before,
                                          const std::vector<TopoDS_Edge>& after) {
  Handle(BRepTools_History) hist = new BRepTools_History;
  ShapeAnalysis_Edge sae;
  const std::size_t n = before.size();
  for (std::size_t i = 0; i < n; ++i) {
    if (!after[i].IsSame(before[i])) {
      hist->AddModified(before[i], after[i]);
    }
    const TopoDS_Vertex ends_before[2] = {sae.FirstVertex(before[i]), sae.LastVertex(before[i])};
    const TopoDS_Vertex ends_after[2] = {sae.FirstVertex(after[i]), sae.LastVertex(after[i])};
    for (int k = 0; k < 2; ++k) {
      if (!ends_after[k].IsSame(ends_before[k]) && !hist->IsRemoved(ends_before[k])) {
        bool listed = false;
        for (const TopoDS_Shape& m : hist->Modified(ends_before[k])) {
          listed = listed || m.IsSame(ends_after[k]);
        }
        if (!listed) {
          hist->AddModified(ends_before[k], ends_after[k]);
        }
      }
    }
  }
  return hist;
}

// The vertex of `edge` at the parameter `t` of its curve, or a null vertex.
TopoDS_Vertex vertex_at_parameter(const TopoDS_Edge& edge, double t) {
  for (TopExp_Explorer v(edge, TopAbs_VERTEX); v.More(); v.Next()) {
    if (BRep_Tool::Parameter(TopoDS::Vertex(v.Current()), edge) == t) {
      return TopoDS::Vertex(v.Current());
    }
  }
  return TopoDS_Vertex();
}

// GEOMImpl_Fillet1dDriver::MakeFillet, copied, with three changes:
//  * `theCorners` holds corners only (not_a_corner, checked by the caller), so the loop's
//    skips of vertices that are not corners are gone;
//  * a corner where no arc is built ends the pass in both passes, and the caller refuses it
//    or takes GEOM's fallback (GEOM skips such a corner in its final pass);
//  * the history is written. Each arc and its two vertices are generated from the corner
//    vertex, which is removed; an edge the pass trimmed is modified into its last image,
//    and one an arc used up is removed. GEOMImpl_Fillet1d::Result builds a trimmed edge on
//    the edge's own curve between the arc's end and the edge's other end, with new
//    vertices at both, so the vertex at that other end is modified into the trimmed
//    edge's vertex at the same curve parameter. The wire fix after the loop adds its own
//    history (connect_history), and GEOMUtils::FixShapeTolerance's ShapeFix_Shape its
//    context's.
FilletPass make_fillet(const TopoDS_Wire& theWire, const std::vector<TopoDS_Vertex>& theCorners,
                       const Standard_Real theRadius) {
  FilletPass out;
  out.hist = new BRepTools_History;

  //INFO: this algorithm implemented in assumption that user can select both
  //  vertices of some edges to make fillet. In this case we should remember
  //  already modified initial edges to take care in next fillet step
  TopTools_DataMapOfShapeShape anEdgeToEdgeMap;

  //iterates on vertices, and make fillet on each couple of edges
  //collect result fillet edges in list
  TopTools_ListOfShape aListOfNewEdge;
  // remember relation between initial and modified map
  TopTools_IndexedDataMapOfShapeListOfShape aMapVToEdges;
  TopExp::MapShapesAndAncestors( theWire, TopAbs_VERTEX, TopAbs_EDGE, aMapVToEdges );
  for (std::size_t ci = 0; ci < theCorners.size(); ++ci) {
    const TopoDS_Vertex& aV = theCorners[ci];
    const TopTools_ListOfShape& aVertexEdges = aMapVToEdges.FindFromKey( aV );
    TopoDS_Edge anEdge1 = TopoDS::Edge( aVertexEdges.First() );
    TopoDS_Edge anEdge2 = TopoDS::Edge( aVertexEdges.Last() );
    // check if initial edges already modified in previous fillet operation
    if ( anEdgeToEdgeMap.IsBound( anEdge1 ) ) anEdge1 = TopoDS::Edge(anEdgeToEdgeMap.Find( anEdge1 ));
    if ( anEdgeToEdgeMap.IsBound( anEdge2 ) ) anEdge2 = TopoDS::Edge(anEdgeToEdgeMap.Find( anEdge2 ));
    if ( anEdge1.IsNull() || anEdge2.IsNull() || anEdge1.IsSame( anEdge2 ) ) {
      out.failed = static_cast<int>(ci);
      out.reason = "an edge that meets there is used up by the arc at another corner";
      return out;
    }

    // create plane on 2 edges
    gp_Pln aPlane;
    if ( !takePlane(anEdge1, anEdge2, aV, aPlane) ) {
      out.failed = static_cast<int>(ci);
      out.reason = "the two edges that meet there, as the arcs at other corners trimmed "
                   "them, lie on one line through it";
      return out;
    }

    geomcopy::GEOMImpl_Fillet1d aFilletAlgo (anEdge1, anEdge2, aPlane);
    if (!aFilletAlgo.Perform(theRadius)) {
      out.failed = static_cast<int>(ci);
      out.reason = "no arc of that radius fits between the two edges that meet there";
      return out;
    }

    // take fillet result in given vertex
    TopoDS_Edge aModifE1, aModifE2;
    TopoDS_Edge aNewE = aFilletAlgo.Result(BRep_Tool::Pnt(aV), aModifE1, aModifE2);
    if (aNewE.IsNull()) {
      out.failed = static_cast<int>(ci);
      out.reason = "no arc of that radius fits between the two edges that meet there";
      return out;
    }

    // add  new created edges and take modified edges
    aListOfNewEdge.Append(aNewE);
    out.hist->AddGenerated(aV, aNewE);
    for (TopExp_Explorer av(aNewE, TopAbs_VERTEX); av.More(); av.Next()) {
      out.hist->AddGenerated(aV, av.Current());
    }

    // The vertex at the untrimmed end of a trimmed edge.
    const auto carry_far_end = [&](const TopoDS_Edge& before, const TopoDS_Edge& after) {
      if (after.IsNull() || after.IsSame(before)) {
        return;
      }
      for (TopExp_Explorer v(before, TopAbs_VERTEX); v.More(); v.Next()) {
        const TopoDS_Vertex& old = TopoDS::Vertex(v.Current());
        const TopoDS_Vertex now = vertex_at_parameter(after, BRep_Tool::Parameter(old, before));
        if (!now.IsNull() && !now.IsSame(old)) {
          out.hist->AddModified(old, now);
        }
      }
    };
    carry_far_end(anEdge1, aModifE1);
    carry_far_end(anEdge2, aModifE2);

    // check if wire edges modified,
    // if yes, then map to original edges (from vertex-edges list), because edges can be modified before
    if (aModifE1.IsNull() || !anEdge1.IsSame( aModifE1 ))
      addEdgeRelation( anEdgeToEdgeMap, TopoDS::Edge(aVertexEdges.First()), aModifE1 );
    if (aModifE2.IsNull() || !anEdge2.IsSame( aModifE2 ))
      addEdgeRelation( anEdgeToEdgeMap, TopoDS::Edge(aVertexEdges.Last()), aModifE2 );
  }

  // create new wire instead of original
  Standard_Real aTol;
  Standard_Real aVertMaxTol = -RealLast();
  for (TopExp_Explorer anExp (theWire, TopAbs_EDGE); anExp.More(); anExp.Next()) {
    TopoDS_Shape anEdge = anExp.Current();
    if (!anEdgeToEdgeMap.IsBound(anEdge))
      aListOfNewEdge.Append(anEdge);
    else if (!anEdgeToEdgeMap.Find(anEdge).IsNull())
      aListOfNewEdge.Append(anEdgeToEdgeMap.Find(anEdge));

    // calculate maximum vertex tolerance of the initial wire
    // to be used for resulting wire fixing (some gaps are possible)
    for (TopExp_Explorer anExV (anEdge, TopAbs_VERTEX); anExV.More(); anExV.Next()) {
      TopoDS_Vertex aVert = TopoDS::Vertex(anExV.Current());
      aTol = BRep_Tool::Tolerance(aVert);
      if (aTol > aVertMaxTol)
        aVertMaxTol = aTol;
    }
  }

  // The edges of the pass: each one's last image, or removed. A corner is not marked
  // removed: BRepTools_History::Merge drops the generated images of a shape the later
  // history removes, and the arc is generated from it. The corner leaves the wire, so its
  // id dies with the commit, as an entity no longer in the model does.
  for (TopExp_Explorer anExp (theWire, TopAbs_EDGE); anExp.More(); anExp.Next()) {
    const TopoDS_Shape& anEdge = anExp.Current();
    if (anEdgeToEdgeMap.IsBound(anEdge)) {
      const TopoDS_Shape& image = anEdgeToEdgeMap.Find(anEdge);
      if (image.IsNull()) {
        out.hist->Remove(anEdge);
      } else {
        out.hist->AddModified(anEdge, image);
      }
    }
  }

  // Fix for Mantis issue 0023411: BEGIN
  BRep_Builder B;
  TopoDS_Wire aResWire;
  B.MakeWire(aResWire);
  TopTools_ListIteratorOfListOfShape anItNewEdge (aListOfNewEdge);
  for (; anItNewEdge.More(); anItNewEdge.Next()) {
    B.Add(aResWire, TopoDS::Edge(anItNewEdge.Value()));
  }
  Handle(ShapeFix_Wire) aFW = new ShapeFix_Wire;
  aFW->Load(aResWire);
  aFW->FixReorder();
  aFW->ClosedWireMode() = theWire.Closed();
  // Fix for Mantis issue 0023411
  // We forced to do this because of fillet 1d algorithm feature
  // (see function DivideEdge in file GEOMImpl_Fillet1d.cxx):
  // a distance (gap) from new arc end to a vertex of original wire
  // can reach (aVertexTolerance + Precision::Confusion()),
  // so a distance between two adjacent arcs will be covered by value below
  const std::vector<TopoDS_Edge> reordered = wire_data_edges(aFW->WireData());
  aFW->FixConnected(aVertMaxTol*2.0 + Precision::Confusion()*3.0);
  const std::vector<TopoDS_Edge> connected = wire_data_edges(aFW->WireData());
  out.hist->Merge(connect_history(reordered, connected));
  TopoDS_Wire theResult = aFW->WireAPIMake();
  // WireAPIMake joins the fixed edges with BRepBuilderAPI_MakeWire, which would copy an
  // edge whose vertex is only coincident with the wire's. After FixConnected every
  // junction shares its vertex, so each edge must come through as it is.
  ShapeSet joined;
  if (!theResult.IsNull()) {
    TopExp::MapShapes(theResult, TopAbs_EDGE, joined);
  }
  bool same_edges = !theResult.IsNull() && joined.Extent() == static_cast<int>(connected.size());
  for (const TopoDS_Edge& e : connected) {
    same_edges = same_edges && joined.Contains(e);
  }
  if (!same_edges) {
    out.failed = -2;
    out.reason = "joining the fixed edges into one wire rebuilt an edge (ShapeFix_Wire::"
                 "WireAPIMake)";
    return out;
  }
  // GEOMUtils::FixShapeTolerance(theResult, TopAbs_VERTEX, Precision::Confusion()): its
  // steps, with the history of its ShapeFix_Shape read from the fixer's context.
  TopoDS_Shape aFixed = theResult;
  ShapeFix_ShapeTolerance aSft;
  aSft.LimitTolerance( aFixed, Precision::Confusion(), Precision::Confusion(), TopAbs_VERTEX );
  Handle(ShapeFix_Shape) aSfs = new ShapeFix_Shape( aFixed );
  aSfs->Perform();
  aFixed = aSfs->Shape();
  out.hist->Merge(aSfs->Context()->History());

  // In OCCT 7.0.0 and earlier this gap was successfully covered by
  // implementation of BRepBuilderAPI_MakeWire::Add(TopTools_ListOfShape)
  // (in the case described in Mantis issue 0023411)
  // Fix for Mantis issue 0023411: END

  if (aFixed.IsNull() || aFixed.ShapeType() != TopAbs_WIRE) {
    out.failed = -2;
    out.reason = "the tolerance fix of the filleted wire gave no wire";
    return out;
  }
  out.wire = TopoDS::Wire(aFixed);
  return out;
}

// GEOMImpl_HealingDriver::FuseCollinearEdges as GEOMImpl_Fillet1dDriver calls it, with no
// vertex list, so that every vertex where two edges continue each other with C1 continuity
// goes. Copied with two changes:
//  * GEOM's copy of the wire at its start is not made: the wire it gets here is fillet_1d's
//    private copy, which nothing else holds (GEOM copies to leave the caller's wire as it
//    is);
//  * the history is written: each edge of a chain is modified into the chain's one edge (a
//    merge: every id of the chain survives on it), the inner vertices of a chain are
//    removed, and the chain's end vertices follow MakeEdgeFromWire's fixer to the vertices
//    of the new edge.
// GEOM raises on a wire it cannot treat; `failure` receives the reason instead.
TopoDS_Wire fuse_collinear_edges(const TopoDS_Wire& theWire, Handle(BRepTools_History)& hist,
                                 std::string& failure) {
  hist = new BRepTools_History;

  // Tolerances
  Standard_Real AngTol = Precision::Angular();
  Standard_Real LinTol = Precision::Confusion();
  Standard_Real tol;
  for (TopExp_Explorer ExV (theWire, TopAbs_VERTEX); ExV.More(); ExV.Next()) {
    TopoDS_Vertex Vertex = TopoDS::Vertex(ExV.Current());
    tol = BRep_Tool::Tolerance(Vertex);
    if (tol > LinTol)
      LinTol = tol;
  }

  // 3. Collect vertices of the wire, same or equal to the given vertices
  const bool removeAll = true;
  TopTools_MapOfShape aMapToRemove;

  // Get the ordered list of edges.
  TopTools_ListOfShape   anEdges;
  TopTools_ListOfShape   aCurVertices;
  BRepTools_WireExplorer aWExp (theWire);

  for (; aWExp.More(); aWExp.Next()) {
    anEdges.Append(aWExp.Current());
    aCurVertices.Append(aWExp.CurrentVertex());
  }

  if (anEdges.IsEmpty()) {
    failure = "the wire has no edge to explore";
    return TopoDS_Wire();
  }

  // Treat the case if the wire is closed and first and last edges are C1.
  Standard_Boolean isShift = Standard_False;

  if (BRep_Tool::IsClosed(theWire)) {
    // Wire is closed. Check if there are more than 2 edges in the wire.
    if (!anEdges.First().IsSame(anEdges.Last())) {
      isShift = Standard_True;
    }
  }

  if (isShift) {
    // Put first edge to the end of the list while the chain break is reached.
    TopoDS_Shape aFirstEdge = anEdges.First();

    while (isShift) {
      isShift = Standard_False;

      // Check if the first vertex should be kept
      if (aMapToRemove.Contains(aCurVertices.First()) || removeAll) {
        // Check if first and last edges are C1.
        TopoDS_Edge anEdge1 = TopoDS::Edge(anEdges.Last());
        TopoDS_Edge anEdge2 = TopoDS::Edge(anEdges.First());

        if (geomcopy::AreEdgesC1(anEdge1, anEdge2)) {
          // Make the first edge last.
          anEdges.Append(anEdge2);
          anEdges.RemoveFirst();
          aCurVertices.Append(aCurVertices.First());
          aCurVertices.RemoveFirst();

          // Check if we reached the first edge again.
          // Break the loop in this case.
          isShift = !aFirstEdge.IsSame(anEdges.First());
        }
      }
    }
  }

  // One chain of more than one edge becomes one edge, with its history.
  const auto fuse_chain = [&](const TopTools_ListOfShape& currChain) -> TopoDS_Edge {
    // make wire from the list of edges
    BRep_Builder B;
    TopoDS_Wire aCurrWire;
    B.MakeWire(aCurrWire);
    TopTools_ListIteratorOfListOfShape itEdges (currChain);
    for (; itEdges.More(); itEdges.Next()) {
      TopoDS_Shape aValue = itEdges.Value();
      B.Add(aCurrWire, TopoDS::Edge(aValue));
    }

    // make edge from the wire
    Handle(BRepTools_History) fixed;
    TopoDS_Edge anEdge = geomcopy::MakeEdgeFromWire(aCurrWire, LinTol, AngTol, &fixed);
    if (anEdge.IsNull()) {
      return anEdge;
    }

    // The history of the chain: its edges merge into the new one; a vertex that bounds
    // only chain edges is inner to it and goes, unless it is an end of the new edge.
    ShapeSet new_vertices;
    TopExp::MapShapes(anEdge, TopAbs_VERTEX, new_vertices);
    TopTools_IndexedDataMapOfShapeListOfShape chain_vertex_edges;
    TopExp::MapShapesAndAncestors(aCurrWire, TopAbs_VERTEX, TopAbs_EDGE, chain_vertex_edges);
    for (itEdges.Initialize(currChain); itEdges.More(); itEdges.Next()) {
      hist->AddModified(itEdges.Value(), anEdge);
    }
    for (int i = 1; i <= chain_vertex_edges.Extent(); ++i) {
      const TopoDS_Shape& v = chain_vertex_edges.FindKey(i);
      std::vector<TopoDS_Shape> images;
      if (!fixed.IsNull()) {
        for (const TopoDS_Shape& m : fixed->Modified(v)) {
          images.push_back(m);
        }
      }
      if (images.empty() && (fixed.IsNull() || !fixed->IsRemoved(v))) {
        images.push_back(v);
      }
      bool kept = false;
      for (const TopoDS_Shape& m : images) {
        if (new_vertices.Contains(m)) {
          kept = true;
          if (!m.IsSame(v)) {
            hist->AddModified(v, m);
          }
        }
      }
      if (!kept) {
        hist->Remove(v);
      }
    }
    return anEdge;
  };

  TopTools_ListOfShape finalList, currChain;
  TopTools_ListIteratorOfListOfShape anEIter(anEdges);
  TopTools_ListIteratorOfListOfShape aVIter(aCurVertices);
  TopoDS_Edge                        prevEdge = TopoDS::Edge(anEIter.Value());

  currChain.Append(prevEdge);
  anEIter.Next();
  aVIter.Next();

  for (; anEIter.More(); anEIter.Next(), aVIter.Next()) {
    TopoDS_Edge         anEdge     = TopoDS::Edge(anEIter.Value());
    const TopoDS_Shape &aCurVertex = aVIter.Value();

    bool continueChain = false;
    if (aMapToRemove.Contains(aCurVertex) || removeAll) {
      // if C1 -> continue chain
      if (geomcopy::AreEdgesC1(prevEdge, anEdge)) {
        continueChain = true;
      }
    }

    if (!continueChain) {
      if (currChain.Extent() == 1) {
        // add one edge to the final list
        finalList.Append(currChain.First());
      }
      else {
        TopoDS_Edge aFused = fuse_chain(currChain);
        if (aFused.IsNull()) {
          failure = "GEOM's MakeEdgeFromWire built no edge over collinear edges";
          return TopoDS_Wire();
        }
        // add this new edge to the final list
        finalList.Append(aFused);
      }
      currChain.Clear();
    }

    // add one edge to the chain
    currChain.Append(anEdge);
    prevEdge = anEdge;
  }

  if (currChain.Extent() == 1) {
    // add one edge to the final list
    finalList.Append(currChain.First());
  }
  else {
    TopoDS_Edge aFused = fuse_chain(currChain);
    if (aFused.IsNull()) {
      failure = "GEOM's MakeEdgeFromWire built no edge over collinear edges";
      return TopoDS_Wire();
    }
    // add this new edge to the final list
    finalList.Append(aFused);
  }

  BRep_Builder B;
  TopoDS_Wire aFinalWire;
  B.MakeWire(aFinalWire);
  TopTools_ListIteratorOfListOfShape itEdges (finalList);
  for (; itEdges.More(); itEdges.Next()) {
    TopoDS_Shape aValue = itEdges.Value();
    B.Add(aFinalWire, TopoDS::Edge(aValue));
  }

  if (!BRepCheck_Analyzer(aFinalWire, Standard_True).IsValid()) {
    failure = "the wire with its collinear edges joined is not valid";
    return TopoDS_Wire();
  }
  return aFinalWire;
}

}  // namespace

// ---- fillet_1d ---------------------------------------------------------------------- //

py::dict Session::fillet_1d(const std::vector<EntityId>& edge_ids,
                            const std::vector<EntityId>& vertex_ids, double radius,
                            bool ignore_secant_vertices) {
  const char* op = "fillet_1d";
  finite_arg(op, "radius", radius);
  OpGuard guard(in_op_);
  const std::vector<TopoDS_Shape> edges = edges_of(op, edge_ids);
  const TopoDS_Shape body = sole_body(op, edge_ids);
  const std::string named = id_text(edge_ids);
  if (body.ShapeType() != TopAbs_WIRE) {
    throw PysmeshError("Session.fillet_1d: the edges " + named + " belong to a " +
                           std::string(TopAbs::ShapeTypeToString(body.ShapeType())) +
                           " body; a 1-D fillet rounds the corners of a free wire body.",
                       "", ids_as_int(edge_ids));
  }
  ShapeSet wire_edges;
  TopExp::MapShapes(body, TopAbs_EDGE, wire_edges);
  if (wire_edges.Extent() != static_cast<int>(edges.size())) {
    std::vector<EntityId> all;
    for (int i = 1; i <= wire_edges.Extent(); ++i) {
      ids_on(wire_edges(i), all);
    }
    throw PysmeshError("Session.fillet_1d: the edges " + named + " are " +
                           std::to_string(edges.size()) + " of the " +
                           std::to_string(wire_edges.Extent()) +
                           " edges of their wire; a 1-D fillet rounds one whole wire, and "
                           "this wire's edges are " + id_text(all) + ".",
                       "", ids_as_int(edge_ids));
  }
  const std::string wire_text = "the wire of edges " + named;
  if (!(radius >= Precision::Confusion())) {
    throw PysmeshError("Session.fillet_1d: the radius " + number_text(radius) +
                           " of the fillets on " + wire_text +
                           " is below Precision::Confusion() (1e-07).",
                       "", ids_as_int(edge_ids));
  }

  // The corners: the named vertices, or every corner in the order GEOM's driver collects
  // the wire's vertices (TopExp_Explorer, each once).
  TopTools_IndexedDataMapOfShapeListOfShape vertex_edges;
  TopExp::MapShapesAndAncestors(body, TopAbs_VERTEX, TopAbs_EDGE, vertex_edges);
  std::vector<TopoDS_Vertex> corners;
  std::vector<EntityId> corner_ids;
  if (vertex_ids.empty()) {
    TopTools_MapOfShape seen;
    for (TopExp_Explorer v(body, TopAbs_VERTEX); v.More(); v.Next()) {
      const TopoDS_Vertex vertex = TopoDS::Vertex(v.Current());
      if (seen.Add(vertex) && not_a_corner(vertex_edges, vertex).empty()) {
        corners.push_back(vertex);
        corner_ids.push_back(label_of(op, vertex));
      }
    }
    if (corners.empty()) {
      throw PysmeshError("Session.fillet_1d: " + wire_text +
                             " has no corner: at each vertex the wire ends, or its two edges "
                             "lie on one line or continue each other with a common tangent.",
                         "", ids_as_int(edge_ids));
    }
  } else {
    for (EntityId id : vertex_ids) {
      const EntityRecord& rec = require_alive(op, id);
      if (rec.kind != TopAbs_VERTEX) {
        throw PysmeshError("Session.fillet_1d: entity " + std::to_string(id) + " is a " +
                               kind_name(rec.kind) + ", not a VERTEX.",
                           "", {static_cast<int>(id)});
      }
      const int index = rec.shapes.size() == 1 ? vertex_edges.FindIndex(rec.shapes.front()) : 0;
      if (index == 0) {
        throw PysmeshError("Session.fillet_1d: vertex " + std::to_string(id) +
                               " is not a vertex of " + wire_text + ".",
                           "", {static_cast<int>(id)});
      }
      const TopoDS_Vertex vertex = TopoDS::Vertex(vertex_edges.FindKey(index));
      for (const TopoDS_Vertex& c : corners) {
        if (c.IsSame(vertex)) {
          throw PysmeshError("Session.fillet_1d: vertex " + std::to_string(id) + " of " +
                                 wire_text + " is named twice.",
                             "", {static_cast<int>(id)});
        }
      }
      const std::string why = not_a_corner(vertex_edges, vertex);
      if (!why.empty()) {
        throw PysmeshError("Session.fillet_1d: vertex " + std::to_string(id) + " of " +
                               wire_text + " is not a corner: " + why + ".",
                           "", {static_cast<int>(id)});
      }
      corners.push_back(vertex);
      corner_ids.push_back(id);
    }
  }
  {
    BRepLib_FindSurface plane(body, -1.0, /*OnlyPlane=*/Standard_True);
    if (!plane.Found()) {
      throw PysmeshError("Session.fillet_1d: " + wire_text +
                             " is not planar within the tolerance of its edges "
                             "(BRepLib_FindSurface finds no plane through it).",
                         "", ids_as_int(edge_ids));
    }
  }
  const std::vector<TopoDS_Shape> untouched = bodies_excluding({body});

  TopoDS_Wire result;
  Handle(BRepTools_History) hist;
  int failed = -1;
  std::string reason;
  std::string fuse_failure;
  bool joined = false;  // the fallback joined collinear edges
  {
    py::gil_scoped_release release;
    try {
      // The private copy every step works on: ShapeFix_ShapeTolerance and ShapeFix_Shape
      // edit vertices and edges in place, and the session's wire is shared by its
      // snapshots.
      BRepBuilderAPI_Copy copier(body, /*copyGeom=*/Standard_True, /*copyMesh=*/Standard_False);
      const TopoDS_Wire wire = TopoDS::Wire(copier.Shape());
      hist = history_of(body, copier);
      std::vector<TopoDS_Vertex> copied;
      for (const TopoDS_Vertex& c : corners) {
        copied.push_back(TopoDS::Vertex(copier.ModifiedShape(c)));
      }

      // GEOMImpl_Fillet1dDriver::Execute: at first the fillet on the wire as it is; when a
      // corner fails and secant vertices are ignored, the collinear edges are joined to
      // allow a bigger radius, the corners are found again, and the fillet is repeated.
      FilletPass pass = make_fillet(wire, copied, radius);
      if (pass.failed >= 0 && ignore_secant_vertices) {
        // 1. Fuse
        Handle(BRepTools_History) fused_hist;
        const TopoDS_Wire aWireNew = fuse_collinear_edges(wire, fused_hist, fuse_failure);
        if (fuse_failure.empty()) {
          ShapeSet before;
          ShapeSet after;
          TopExp::MapShapes(wire, TopAbs_EDGE, before);
          TopExp::MapShapes(aWireNew, TopAbs_EDGE, after);
          joined = after.Extent() < before.Extent();
          // 2. Rebuild the list of vertices (by coincidence)
          Standard_Real tol, tolMax = Precision::Confusion();
          for (TopExp_Explorer ExV (aWireNew, TopAbs_VERTEX); ExV.More(); ExV.Next()) {
            TopoDS_Vertex Vertex = TopoDS::Vertex(ExV.Current());
            tol = BRep_Tool::Tolerance(Vertex);
            if (tol > tolMax)
              tolMax = tol;
          }
          std::vector<TopoDS_Vertex> aVertexListNew;
          TopTools_IndexedMapOfShape anIndicesNew;
          TopExp::MapShapes(aWireNew, anIndicesNew);
          for (std::size_t ci = 0; ci < copied.size(); ++ci) {
            const TopoDS_Vertex& aV = copied[ci];
            if (anIndicesNew.Contains(aV)) {
              aVertexListNew.push_back(aV);
              continue;
            }
            // try to find by coords in the new wire
            gp_Pnt aP = BRep_Tool::Pnt(aV);

            bool isFound = false;
            TopTools_MapOfShape mapShape;
            TopExp_Explorer exp (aWireNew, TopAbs_VERTEX);
            for (; exp.More() && !isFound; exp.Next()) {
              if (mapShape.Add(exp.Current())) {
                TopoDS_Vertex aVi = TopoDS::Vertex(exp.Current());
                gp_Pnt aPi = BRep_Tool::Pnt(aVi);
                if (aPi.Distance(aP) < tolMax) {
                  aVertexListNew.push_back(aVi);
                  isFound = true;
                }
              }
            }
            if (!isFound) {
              pass.failed = static_cast<int>(ci);
              pass.reason = "the corner is not in the wire once its collinear edges are "
                            "joined";
              break;
            }
          }
          // 3. Repeat the fillet algorithm
          if (aVertexListNew.size() == copied.size()) {
            hist->Merge(fused_hist);
            pass = make_fillet(aWireNew, aVertexListNew, radius);
          }
        }
      }
      failed = pass.failed;
      reason = pass.reason;
      if (failed == -1 && fuse_failure.empty()) {
        hist->Merge(pass.hist);
        // GEOMImpl_Fillet1dDriver::Execute's last check: GEOMUtils::CheckShape(result, true),
        // and when it fails GEOMUtils::FixShapeTolerance(result) (every kind, 1e-7), whose
        // ShapeFix_Shape history is read from its context.
        TopoDS_Shape aResult = pass.wire;
        if (!BRepCheck_Analyzer(aResult, Standard_True).IsValid()) {
          ShapeFix_ShapeTolerance aSft;
          aSft.LimitTolerance(aResult, Precision::Confusion(), Precision::Confusion(),
                              TopAbs_SHAPE);
          Handle(ShapeFix_Shape) aSfs = new ShapeFix_Shape(aResult);
          aSfs->Perform();
          aResult = aSfs->Shape();
          hist->Merge(aSfs->Context()->History());
        }
        if (!aResult.IsNull() && aResult.ShapeType() == TopAbs_WIRE &&
            BRepCheck_Analyzer(aResult, Standard_True).IsValid()) {
          result = TopoDS::Wire(aResult);
        }
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.fillet_1d: OCCT threw on ") + wire_text + ": " +
                             e.what(),
                         "", ids_as_int(edge_ids));
    }
  }
  if (!fuse_failure.empty()) {
    throw PysmeshError("Session.fillet_1d: a fillet of radius " + number_text(radius) +
                           " does not fit on " + wire_text +
                           " as it is, and joining its collinear edges failed: " +
                           fuse_failure + ".",
                       "", ids_as_int(edge_ids));
  }
  if (failed >= 0) {
    const EntityId at = corner_ids[static_cast<std::size_t>(failed)];
    throw PysmeshError("Session.fillet_1d: at vertex " + std::to_string(at) + " of " +
                           wire_text + " no fillet of radius " + number_text(radius) +
                           " was built: " + reason +
                           (joined ? " (also with the wire's collinear edges joined)." : "."),
                       "", {static_cast<int>(at)});
  }
  if (failed == -2 || result.IsNull()) {
    throw PysmeshError("Session.fillet_1d: the fillets of radius " + number_text(radius) +
                           " on " + wire_text + " gave no valid wire" +
                           (reason.empty() ? std::string() : ": " + reason) + ".",
                       "", ids_as_int(edge_ids));
  }
  return commit(concat(untouched, result), kind_preserving(hist, body), op, result);
}

// ---- smoothing_surface -------------------------------------------------------------- //

namespace {

// AdvancedEngine_SmoothingSurfaceDriver::MakeSmoothingSurfaceUnClosed (V9_16_0,
// src/AdvancedEngine/AdvancedEngine_SmoothingSurfaceDriver.cxx; Copyright (C) 2013-2026
// CEA, EDF, OPEN CASCADE), copied: the OCCT >= 7.7 branch of its adaptor block, a free
// function instead of a const member, and its commented-out debug lines dropped.
TopoDS_Shape MakeSmoothingSurfaceUnClosed
                          (const Handle(TColgp_HArray1OfPnt) &theListOfPoints,
                           const Standard_Integer            theNbMax,
                           const Standard_Integer            theDegMax,
                           const Standard_Real               theDMax)
{
  TopoDS_Face aInitShape;

  // Create an average Plane
  GeomPlate_BuildAveragePlane gpbap(theListOfPoints,theListOfPoints->Length(),Precision::Confusion(),1,1);
  Handle(Geom_Plane) plane(gpbap.Plane());
  Standard_Real Umin, Umax, Vmin, Vmax;
  gpbap.MinMaxBox(Umin,Umax,Vmin,Vmax);
  BRepBuilderAPI_MakeFace mf(plane,Umin,Umax,Vmin,Vmax,Precision::Confusion());
  aInitShape =  mf.Face();

  GeomPlate_BuildPlateSurface aBuilder(3,10);
  // ** Initialization of surface
  Handle(BRepAdaptor_Surface) HSI = new BRepAdaptor_Surface();
  HSI->Initialize(aInitShape);
  aBuilder.LoadInitSurface( BRep_Tool::Surface(HSI->Face()) );

  Standard_Integer j, j1, j2;
  j1 = theListOfPoints->Lower();
  j2 = theListOfPoints->Upper();
  for (j=j1; j<=j2 ; j++)
  {
    gp_Pnt aPnt = theListOfPoints->Value(j);
    Handle(GeomPlate_PointConstraint) PCont = new GeomPlate_PointConstraint(aPnt,0);
    aBuilder.Add(PCont);
  }
  aBuilder.Perform();

  // A ce niveau : surface algo
  Handle(GeomPlate_Surface) gpPlate = aBuilder.Surface();

  Standard_Integer nbcarreau = (theNbMax > 0 ? theNbMax : 2);
  Standard_Integer degmax = (theDegMax > 0 ? theDegMax : 8);
  Standard_Real seuil = theDMax;

  if (seuil <= 0.) {
    seuil = Max(0.0001,10*aBuilder.G0Error());
  }

  GeomPlate_MakeApprox Mapp(gpPlate,0.0001,nbcarreau,degmax,seuil);

  Handle (Geom_Surface) Surf (Mapp.Surface());

  aBuilder.Surface()->Bounds( Umin, Umax, Vmin, Vmax);

  BRepBuilderAPI_MakeFace MF(Surf,Umin, Umax, Vmin, Vmax, Precision::Confusion());
  TopoDS_Shape aShape = MF.Shape();

  return aShape;
}

}  // namespace

py::dict Session::smoothing_surface(const PointArray& points, int max_points, int max_degree,
                                    double max_distance) {
  const char* op = "smoothing_surface";
  finite_arg(op, "max_distance", max_distance);
  OpGuard guard(in_op_);
  const std::vector<gp_Pnt> pts = points_of(op, "points", points, 3);
  if (max_points < 1) {
    throw PysmeshError("Session.smoothing_surface: max_points " + std::to_string(max_points) +
                       " is below 1; the surface needs at least one Bezier piece.");
  }
  if (max_degree < 1) {
    throw PysmeshError("Session.smoothing_surface: max_degree " + std::to_string(max_degree) +
                       " is below 1.");
  }
  if (max_distance < 0.0) {
    throw PysmeshError("Session.smoothing_surface: max_distance " + number_text(max_distance) +
                       " is negative; a distance is zero or above.");
  }
  {
    // Two point constraints at one point make GeomPlate's linear system singular, and
    // OCCT 8.0.1 then fails with an access violation inside the plate solve (measured; GEOM
    // takes distinct vertices, so it never sees two). Points closer than
    // Precision::Confusion() are one point to OCCT, so they are refused. A sweep along x
    // finds every such pair.
    std::vector<std::size_t> order(pts.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&pts](std::size_t a, std::size_t b) {
      return pts[a].X() < pts[b].X() || (pts[a].X() == pts[b].X() && a < b);
    });
    const double tol = Precision::Confusion();
    for (std::size_t k = 0; k < order.size(); ++k) {
      for (std::size_t m = k + 1; m < order.size() && pts[order[m]].X() - pts[order[k]].X() <= tol;
           ++m) {
        if (pts[order[k]].Distance(pts[order[m]]) <= tol) {
          const std::size_t a = std::min(order[k], order[m]);
          const std::size_t b = std::max(order[k], order[m]);
          throw PysmeshError("Session.smoothing_surface: points " + std::to_string(a) + " and " +
                             std::to_string(b) + " (0-based rows) coincide within " +
                             number_text(tol) + "; each point constrains the surface once.");
        }
      }
    }
  }
  Handle(TColgp_HArray1OfPnt) array = new TColgp_HArray1OfPnt(1, static_cast<int>(pts.size()));
  for (std::size_t i = 0; i < pts.size(); ++i) {
    array->SetValue(static_cast<int>(i) + 1, pts[i]);
  }
  {
    // The average plane GeomPlate_BuildAveragePlane takes needs two directions of spread:
    // GeomLib::AxeOfInertia, at the tolerance the plane is built with, calls points that
    // lie on one line (or at one point) singular, and the plane through them is then any
    // plane through that line.
    gp_Ax2 axis;
    bool singular = false;
    GeomLib::AxeOfInertia(array->Array1(), axis, singular, Precision::Confusion());
    if (singular) {
      throw PysmeshError("Session.smoothing_surface: the " + std::to_string(pts.size()) +
                         " points lie on one line or at one point (GeomLib::AxeOfInertia "
                         "calls them singular); no surface through them is defined.");
    }
  }
  TopoDS_Shape face;
  std::string failure;
  {
    py::gil_scoped_release release;
    try {
      face = MakeSmoothingSurfaceUnClosed(array, max_points, max_degree, max_distance);
    } catch (const std::exception& e) {
      failure = e.what();
    }
  }
  if (!failure.empty()) {
    throw PysmeshError("Session.smoothing_surface: OCCT failed on the " +
                       std::to_string(pts.size()) + " points (max_points " +
                       std::to_string(max_points) + ", max_degree " +
                       std::to_string(max_degree) + ", max_distance " +
                       number_text(max_distance) + "): " + failure);
  }
  if (face.IsNull() || face.ShapeType() != TopAbs_FACE) {
    throw PysmeshError("Session.smoothing_surface: GeomPlate built no face over the " +
                       std::to_string(pts.size()) + " points.");
  }
  return add_bodies(face, op);
}

}  // namespace session
}  // namespace pysmesh
