// Copyright (C) 2007-2026  CEA, EDF, OPEN CASCADE
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
//
// See http://www.salome-platform.org/ or email : webmaster.salome@opencascade.com
//

// File      : NETGENPlugin_NETGEN_2D_ONLY.cxx
// Author    : Edward AGAPOV (OCC)
// Project   : SALOME
//
#include "NETGENPlugin_NETGEN_2D_ONLY.hxx"
#include "NETGENPlugin_Hypothesis_2D.hxx"

#include <SMDS_MeshElement.hxx>
#include <SMDS_MeshNode.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_Comment.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_MesherHelper.hxx>
#include <SMESH_subMesh.hxx>
#include <StdMeshers_FaceSide.hxx>
#include <StdMeshers_LengthFromEdges.hxx>
#include <StdMeshers_MaxElementArea.hxx>
#include <StdMeshers_QuadranglePreference.hxx>
#include <StdMeshers_ViscousLayers2D.hxx>

#include <Precision.hxx>
#include <Standard_ErrorHandler.hxx>
#include <Standard_Failure.hxx>

#include <utilities.h>

#include <GEOMUtils.hxx>

#include <list>
#include <vector>
#include <limits>

/*
  Netgen include files
*/
namespace nglib {
#include <nglib.h>
}
#ifndef OCCGEOMETRY
#define OCCGEOMETRY
#endif
#include <occgeom.hpp>
#include <meshing.hpp>

#include "SMESH_Octree.hxx"

//// Used for node projection in curve
#include <BRepAdaptor_Curve.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <BndLib_Add3dCurve.hxx>
#include <GCPnts_TangentialDeflection.hxx>
#include <ShapeAnalysis_Curve.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Vertex.hxx>

//#include <meshtype.hpp>
namespace netgen {
  NETGENPLUGIN_DLL_HEADER
  extern MeshingParameters mparam;
#ifdef NETGEN_V5
  extern void OCCSetLocalMeshSize(OCCGeometry & geom, Mesh & mesh);
#endif
}

using namespace std;
using namespace netgen;
using namespace nglib;

namespace // copied class from StdMesher_Importe_1D   
{
  /*!
  * \brief Compute point position on a curve. Use octree to fast reject far points
  */
  class CurveProjector : public SMESH_Octree
  {
  public:
    CurveProjector( const TopoDS_Edge& edge, double enlarge );

    bool IsOnCurve( const gp_XYZ& point, double & distance2, double & u );

    bool IsOut( const gp_XYZ& point ) const { return getBox()->IsOut( point ); }

  protected:
    CurveProjector() {}
    SMESH_Octree* newChild() const { return new CurveProjector; }
    void          buildChildrenData();
    Bnd_B3d*      buildRootBox();

  private:
    struct CurveSegment : public Bnd_B3d
    {
      double _chord, _chord2, _length2;
      gp_Pnt _pFirst, _pLast;
      gp_Lin _line;
      Handle(Geom_Curve) _curve;

      CurveSegment() {}
      void Init( const gp_Pnt& pf, const gp_Pnt& pl,
                 double uf, double ul, double tol, Handle(Geom_Curve)& curve );
      bool IsOn( const gp_XYZ& point, double & distance2, double & u );
      bool IsInContact( const Bnd_B3d& bb );
    };
    std::vector< CurveSegment > _segments;
  };

    //===============================================================================
  /*!
   * \brief Create an octree of curve segments
   */
  //================================================================================

  CurveProjector::CurveProjector( const TopoDS_Edge& edge, double enlarge )
    :SMESH_Octree( 0 )
  {
    double f,l;
    Handle(Geom_Curve) curve = BRep_Tool::Curve( edge, f, l );
    double curDeflect = 0.3; // Curvature deflection
    double angDeflect = 1e+100; // Angular deflection - don't control chordal error
    GCPnts_TangentialDeflection div( BRepAdaptor_Curve( edge ), angDeflect, curDeflect );
    _segments.resize( div.NbPoints() - 1 );
    for ( int i = 1; i < div.NbPoints(); ++i )
      try {
        _segments[ i - 1 ].Init( div.Value( i ),     div.Value( i+1 ),
                                 div.Parameter( i ), div.Parameter( i+1 ),
                                 enlarge, curve );
      }
      catch ( Standard_Failure& ) {
        _segments.resize( _segments.size() - 1 );
        --i;
      }
    if ( _segments.size() < 3 )
      myIsLeaf = true;

    compute();

    if ( _segments.size() == 1 )
      myBox->Enlarge( enlarge );
  }

  //================================================================================
  /*!
   * \brief Return the maximal box
   */
  //================================================================================

  Bnd_B3d* CurveProjector::buildRootBox()
  {
    Bnd_B3d* box = new Bnd_B3d;
    for ( size_t i = 0; i < _segments.size(); ++i )
      box->Add( _segments[i] );
    return box;
  }

  //================================================================================
  /*!
   * \brief Redistribute segments among children
   */
  //================================================================================

  void CurveProjector::buildChildrenData()
  {
    bool allIn = true;
    for ( size_t i = 0; i < _segments.size(); ++i )
    {
      for (int j = 0; j < 8; j++)
      {
        if ( _segments[i].IsInContact( *myChildren[j]->getBox() ))
          ((CurveProjector*)myChildren[j])->_segments.push_back( _segments[i]);
        else
          allIn = false;
      }
    }
    if ( allIn && _segments.size() < 3 )
    {
      myIsLeaf = true;
      for (int j = 0; j < 8; j++)
        static_cast<CurveProjector*>( myChildren[j])->myIsLeaf = true;
    }
    else
    {
      SMESHUtils::FreeVector( _segments ); // = _segments.clear() + free memory

      for (int j = 0; j < 8; j++)
      {
        CurveProjector* child = static_cast<CurveProjector*>( myChildren[j]);
        if ( child->_segments.size() < 3 )
          child->myIsLeaf = true;
      }
    }
  }

  //================================================================================
  /*!
   * \brief Return true if a point is close to the curve
   *  \param [in] point - the point
   *  \param [out] distance2 - distance to the curve
   *  \param [out] u - parameter on the curve
   *  \return bool - is the point is close to the curve
   */
  //================================================================================

  bool CurveProjector::IsOnCurve( const gp_XYZ& point, double & distance2, double & u )
  {
    if ( getBox()->IsOut( point ))
      return false;

    bool ok = false;
    double dist2, param;
    distance2 = Precision::Infinite();

    if ( isLeaf() )
    {
      for ( size_t i = 0; i < _segments.size(); ++i )
        if ( !_segments[i].IsOut( point ) &&
             _segments[i].IsOn( point, dist2, param ) &&
             dist2 < distance2 )
        {
          distance2 = dist2;
          u         = param;
          ok        = true;
        }
      return ok;
    }
    else
    {
      for (int i = 0; i < 8; i++)
        if (((CurveProjector*) myChildren[i])->IsOnCurve( point, dist2, param ) &&
            dist2 < distance2 )
        {
          distance2 = dist2;
          u         = param;
          ok        = true;
        }
    }
    return ok;
  }

  //================================================================================
  /*!
   * \brief Initialize
   */
  //================================================================================

  void CurveProjector::CurveSegment::Init(const gp_Pnt&       pf,
                                          const gp_Pnt&       pl,
                                          const double        uf,
                                          const double        ul,
                                          const double        tol,
                                          Handle(Geom_Curve)& curve )
  {
    _pFirst  = pf;
    _pLast   = pl;
    _curve   = curve;
    _length2 = pf.SquareDistance( pl );
    _line.SetLocation( pf );
    _line.SetDirection( gp_Vec( pf, pl ));
    _chord2  = Max( _line.     SquareDistance( curve->Value( uf + 0.25 * ( ul - uf ))),
                    Max( _line.SquareDistance( curve->Value( uf + 0.5  * ( ul - uf ))),
                         _line.SquareDistance( curve->Value( uf + 0.75 * ( ul - uf )))));
    _chord2 *= ( 1.05 * 1.05 ); // +5%
    _chord2  = Max( tol, _chord2 );
    _chord   = Sqrt( _chord2 );

    Bnd_Box bb;
    BndLib_Add3dCurve::Add( GeomAdaptor_Curve( curve, uf, ul ), tol, bb );
    Add( bb.CornerMin() );
    Add( bb.CornerMax() );
  }

  //================================================================================
  /*!
   * \brief Return true if a point is close to the curve segment
   *  \param [in] point - the point
   *  \param [out] distance2 - distance to the curve
   *  \param [out] u - parameter on the curve
   *  \return bool - is the point is close to the curve segment
   */
  //================================================================================

  bool CurveProjector::CurveSegment::IsOn( const gp_XYZ& point, double & distance2, double & u )
  {
    distance2 = _line.SquareDistance( point );
    if ( distance2 > _chord2 )
      return false;

    // check if the point projection falls into the segment range
    {
      gp_Vec edge( _pFirst, _pLast );
      gp_Vec n1p ( _pFirst, point  );
      u = ( edge * n1p ) / _length2; // param [0,1] on the edge
      if ( u < 0. )
      {
        if ( _pFirst.SquareDistance( point ) > _chord2 )
          return false;
      }
      else if ( u > 1. )
      {
        if ( _pLast.SquareDistance( point ) > _chord2 )
          return false;
      }
    }
    gp_Pnt proj;
    distance2 = ShapeAnalysis_Curve().Project( _curve, point, Precision::Confusion(),
                                               proj, u, false );
    distance2 *= distance2;
    return true;
  }

  //================================================================================
  /*!
   * \brief Check if the segment is in contact with a box
   */
  //================================================================================

  bool CurveProjector::CurveSegment::IsInContact( const Bnd_B3d& bb )
  {
    if ( bb.IsOut( _line.Position(), /*isRay=*/true, _chord ))
      return false;

    gp_Ax1 axRev = _line.Position().Reversed();
    axRev.SetLocation( _pLast );
    return !bb.IsOut( axRev, /*isRay=*/true, _chord );
  }

}

//=============================================================================
/*!
 *
 */
//=============================================================================

NETGENPlugin_NETGEN_2D_ONLY::NETGENPlugin_NETGEN_2D_ONLY(int        hypId,
                                                         SMESH_Gen* gen)
  : SMESH_2D_Algo(hypId, gen)
{
  _name = "NETGEN_2D_ONLY";

  _shapeType = (1 << TopAbs_FACE);// 1 bit /shape type
  _onlyUnaryInput = false; // treat all FACEs at once

  _compatibleHypothesis.push_back("MaxElementArea");
  _compatibleHypothesis.push_back("LengthFromEdges");
  _compatibleHypothesis.push_back("QuadranglePreference");
  _compatibleHypothesis.push_back("NETGEN_Parameters_2D");
  _compatibleHypothesis.push_back("ViscousLayers2D");

  _hypMaxElementArea       = 0;
  _hypLengthFromEdges      = 0;
  _hypQuadranglePreference = 0;
  _hypParameters           = 0;
}

//=============================================================================
/*!
 *
 */
//=============================================================================

NETGENPlugin_NETGEN_2D_ONLY::~NETGENPlugin_NETGEN_2D_ONLY()
{
  //MESSAGE("NETGENPlugin_NETGEN_2D_ONLY::~NETGENPlugin_NETGEN_2D_ONLY");
}

//=============================================================================
/*!
 *
 */
//=============================================================================

bool NETGENPlugin_NETGEN_2D_ONLY::CheckHypothesis (SMESH_Mesh&         aMesh,
                                                   const TopoDS_Shape& aShape,
                                                   Hypothesis_Status&  aStatus)
{
  _hypMaxElementArea = 0;
  _hypLengthFromEdges = 0;
  _hypQuadranglePreference = 0;
  _hypParameters = 0;
  _progressByTic = -1;

  const list<const SMESHDS_Hypothesis*>& hyps = GetUsedHypothesis(aMesh, aShape, false);

  if (hyps.empty())
  {
    aStatus = HYP_OK; //SMESH_Hypothesis::HYP_MISSING;
    return true;  // (PAL13464) can work with no hypothesis, LengthFromEdges is default one
  }

  aStatus = HYP_MISSING;

  bool hasVL = false;
  list<const SMESHDS_Hypothesis*>::const_iterator ith;
  for (ith = hyps.begin(); ith != hyps.end(); ++ith )
  {
    const SMESHDS_Hypothesis* hyp = (*ith);

    string hypName = hyp->GetName();

    if      ( hypName == "MaxElementArea")
      _hypMaxElementArea = static_cast<const StdMeshers_MaxElementArea*> (hyp);
    else if ( hypName == "LengthFromEdges" )
      _hypLengthFromEdges = static_cast<const StdMeshers_LengthFromEdges*> (hyp);
    else if ( hypName == "QuadranglePreference" )
      _hypQuadranglePreference = static_cast<const StdMeshers_QuadranglePreference*>(hyp);
    else if ( hypName == "NETGEN_Parameters_2D" )
      _hypParameters = static_cast<const NETGENPlugin_Hypothesis_2D*>(hyp);
    else if ( hypName == StdMeshers_ViscousLayers2D::GetHypType() )
      hasVL = true;
    else {
      aStatus = HYP_INCOMPATIBLE;
      return false;
    }
  }

  int nbHyps = bool(_hypMaxElementArea) + bool(_hypLengthFromEdges) + bool(_hypParameters );
  if ( nbHyps > 1 )
    aStatus = HYP_CONCURRENT;
  else if ( hasVL )
    error( StdMeshers_ViscousLayers2D::CheckHypothesis( aMesh, aShape, aStatus ));
  else
    aStatus = HYP_OK;

  if ( aStatus == HYP_OK && _hypParameters && _hypQuadranglePreference )
  {
    aStatus = HYP_INCOMPAT_HYPS;
    return error(SMESH_Comment("\"") << _hypQuadranglePreference->GetName()
                 << "\" and \"" << _hypParameters->GetName()
                 << "\" are incompatible hypotheses");
  }

  return ( aStatus == HYP_OK );
}

// namespace
// {
//   void limitSize( netgen::Mesh* ngMesh,
//                   const double  maxh )
//   {
//     // get bnd box
//     netgen::Point3d pmin, pmax;
//     ngMesh->GetBox( pmin, pmax, 0 );
//     const double dx = pmax.X() - pmin.X();
//     const double dy = pmax.Y() - pmin.Y();
//     const double dz = pmax.Z() - pmin.Z();

//     const int nbX = Max( 2, int( dx / maxh * 3 ));
//     const int nbY = Max( 2, int( dy / maxh * 3 ));
//     const int nbZ = Max( 2, int( dz / maxh * 3 ));

//     if ( ! & ngMesh->LocalHFunction() )
//       ngMesh->SetLocalH( pmin, pmax, 0.1 );

//     netgen::Point3d p;
//     for ( int i = 0; i <= nbX; ++i )
//     {
//       p.X() = pmin.X() +  i * dx / nbX;
//       for ( int j = 0; j <= nbY; ++j )
//       {
//         p.Y() = pmin.Y() +  j * dy / nbY;
//         for ( int k = 0; k <= nbZ; ++k )
//         {
//           p.Z() = pmin.Z() +  k * dz / nbZ;
//           ngMesh->RestrictLocalH( p, maxh );
//         }
//       }
//     }
//   }
// }



/**
 * @brief MapSegmentsToEdges. 
 * @remark To feed 1D segments not associated to any geometry we need:
 *          1) For each face, check all segments that are in the face (use ShapeAnalysis_Surface class)
 *          2) Check to which edge the segment below to, use the copied [from StdMesher_Importe_1D] CurveProjector class 
 *          3) Create new netgen segments with the (u,v) parameters obtained from the ShapeAnalysis_Surface projector
 *          4) also define the 'param' value of the nodes relative to the edges obtained from CurveProjector
 *          5) Add the new netgen segments IN ORDER into the netgen::mesh data structure to form a closed chain
 *          6) Beware with the occ::edge orientation
 * 
 * @param aMesh Mesh file (containing 1D elements)
 * @param aShape Shape file (BREP or STEP format)
 * @param ngLib netgenlib library wrapper
 * @param nodeVec vector of nodes used internally to feed smesh aMesh after computation
 * @param premeshedNodes map of prmeshed nodes and the smesh nodeID associate to it
 * @param newNetgenCoordinates map of 3D coordinate of new points created by netgen
 * @param newNetgenElements map of triangular or quadrangular elements ID and the nodes defining the 2D element
 * @return false
 */
bool NETGENPlugin_NETGEN_2D_ONLY::MapSegmentsToEdges(SMESH_Mesh& aMesh, const TopoDS_Shape& aShape, NETGENPlugin_NetgenLibWrapper &ngLib,
                                                      vector< const SMDS_MeshNode* >& nodeVec, std::map<int,const SMDS_MeshNode*>& premeshedNodes, 
                                                      std::map<int,std::vector<double>>& newNetgenCoordinates, std::map<int,std::vector<smIdType>>& newNetgenElements )
{  
  SMESHDS_Mesh * meshDS = aMesh.GetMeshDS();
  SMESH_MesherHelper helper(aMesh);
  helper.SetElementsOnShape( true );
  const int numberOfPremeshedNodes = aMesh.NbNodes();
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes( aShape, TopAbs_FACE, faces );
  int err = 0;
  for ( int i = 1; i <= faces.Size(); ++i )
  {
    int numOfEdges = 0;
    int totalEdgeLenght = 0;

    TopoDS_Face face = TopoDS::Face(faces( i ) );
    Bnd_Box FaceBox;
    GEOMUtils::PreciseBoundingBox( face, FaceBox );
    if ( face.Orientation() != TopAbs_FORWARD && face.Orientation() != TopAbs_REVERSED )
      face.Orientation( TopAbs_FORWARD );

    // Set the occgeom to be meshed and some ngMesh parameteres
    netgen::OCCGeometry occgeom;
    netgen::Mesh * ngMesh =  (netgen::Mesh*) ngLib._ngMesh;
    ngMesh->DeleteMesh();

    occgeom.shape = face;
    occgeom.fmap.Add( face );
    occgeom.CalcBoundingBox();
    occgeom.facemeshstatus.SetSize(1);
    occgeom.facemeshstatus = 0;
    occgeom.face_maxh_modified.SetSize(1);
    occgeom.face_maxh_modified = 0;
    occgeom.face_maxh.SetSize(1);
    occgeom.face_maxh = netgen::mparam.maxh;

    // Set the face descriptor
    const int solidID = 0, faceID = 1; /*always 1 because faces are meshed one by one*/
    if ( ngMesh->GetNFD() < 1 )
      ngMesh->AddFaceDescriptor( netgen::FaceDescriptor( faceID, solidID, solidID, 0 ));

    ngMesh->SetGlobalH ( netgen::mparam.maxh );
    ngMesh->SetMinimalH( netgen::mparam.minh );
    Box<3> bb = occgeom.GetBoundingBox();
    bb.Increase (bb.Diam()/10);
    ngMesh->SetLocalH (bb.PMin(), bb.PMax(), netgen::mparam.grading);  
    // end set the occgeom to be meshed and some ngMesh parameteres

    Handle(ShapeAnalysis_Surface) sprojector = new ShapeAnalysis_Surface( BRep_Tool::Surface( face ));
    double tol = BRep_Tool::MaxTolerance( face, TopAbs_FACE );
    gp_Pnt surfPnt(0,0,0);
  
    map<const SMDS_MeshNode*, int > node2ngID;
    map<int, const SMDS_MeshNode* > ng2smesh;
    // Figure out which edge is this onde in!
    TopTools_IndexedMapOfShape edges;
    TopExp::MapShapes( face, TopAbs_EDGE, edges );
    TopoDS_Edge meshingEdge;
    // Check wich nodes are in this face!
    SMDS_ElemIteratorPtr iteratorElem = meshDS->elementsIterator(SMDSAbs_Edge);
    std::unique_ptr<CurveProjector> myCurveProjector;
    while ( iteratorElem->more() ) // loop on elements on a geom face
    {
      // check mesh face
      const SMDS_MeshElement* elem = iteratorElem->next();
      const SMDS_MeshNode* node0 = elem->GetNode( 0 );
      const SMDS_MeshNode* node1 = elem->GetNode( 1 );
      SMESH_NodeXYZ nXYZ0( node0 );
      SMESH_NodeXYZ nXYZ1( node1 );
      double segmentLength = ( nXYZ0 - nXYZ1 ).Modulus();
      nXYZ0 += nXYZ1;
      nXYZ0 /= 2.0;
      if( FaceBox.IsOut( nXYZ0 ) )
        continue;      

      gp_XY uv = sprojector->ValueOfUV( nXYZ0, tol ).XY();
      surfPnt = sprojector->Value( uv );
      double dist = surfPnt.Distance( nXYZ0 );
      
      // used for the curve projector of edges
      double geomTol = Precision::Confusion();
      if ( dist < tol /*element is on face*/ )
      {
        numOfEdges++;
        totalEdgeLenght += segmentLength;
        int occEdgeIdFound = -1;
        for ( int edgeId = 1; edgeId <= edges.Size(); ++edgeId ) /*find in which edge the node is placed*/
        {
          meshingEdge = TopoDS::Edge(edges( edgeId ));
          myCurveProjector = std::unique_ptr<CurveProjector>( new CurveProjector(meshingEdge, geomTol) );
          if ( myCurveProjector->IsOut( nXYZ0 ) /*keep searching*/)
            continue;
          else
          {
            occEdgeIdFound = edgeId;
            break;
          }
        }

        netgen::Segment seg;
        for ( size_t nodeId = 0; nodeId < 2; nodeId++)
        {
          const SMDS_MeshNode* node = elem->GetNode( nodeId );
          int ngId = ngMesh->GetNP() + 1;
          ngId = node2ngID.insert( make_pair( node, ngId )).first->second;
          if ( ngId > ngMesh->GetNP() /* mean it is a new node to be add to the mesh*/)
          {
            // Restric size of mesh based on the edge dimension
            {
              SMESH_NodeXYZ nXYZ( node );
              netgen::Point3d pi(nXYZ.X(), nXYZ.Y(), nXYZ.Z());
              ngMesh->RestrictLocalH( pi, segmentLength );
            }
              
            netgen::MeshPoint mp( netgen::Point<3> (node->X(), node->Y(), node->Z()) );
            ngMesh->AddPoint ( mp, 1, netgen::EDGEPOINT );
            ng2smesh.insert( make_pair(ngId, node) );
            premeshedNodes.insert( make_pair( (int)node->GetID(), node ) );
          } 
          seg[nodeId] = ngId;                // ng node id
          SMESH_NodeXYZ nXYZ( node );
          gp_XY uv    = sprojector->ValueOfUV( nXYZ, tol ).XY();

          // Compute the param (relative distance) of the node in relation the edge
          // fundamental for netgen working properly
          double dist2, param;
          if ( myCurveProjector->IsOnCurve( nXYZ, dist2, param ) )
          {
            seg.epgeominfo[ nodeId ].dist = param;
            seg.epgeominfo[ nodeId ].u    = uv.X();
            seg.epgeominfo[ nodeId ].v    = uv.Y();   
            seg.epgeominfo[ nodeId ].edgenr = occEdgeIdFound;
          }       
        }

        if ( meshingEdge.Orientation() != TopAbs_FORWARD )
        {
          swap(seg[0], seg[1]);
          swap(seg.epgeominfo[0], seg.epgeominfo[1] );
        }
        
        seg.edgenr = ngMesh->GetNSeg() + 1; // netgen segment id
        seg.si     = faceID;
        ngMesh->AddSegment( seg );
      } // end if edge is on the face    
    } // end iteration on elements
    
    // set parameters from _hypLengthFromEdges if needed
    if ( !_hypParameters && _hypLengthFromEdges && numOfEdges )
    {
      netgen::mparam.maxh = totalEdgeLenght / numOfEdges;
      if ( netgen::mparam.maxh < DBL_MIN )
        netgen::mparam.maxh = occgeom.GetBoundingBox().Diam();

      occgeom.face_maxh = netgen::mparam.maxh;
      ngMesh->SetGlobalH ( netgen::mparam.maxh );
    }    
    // end set parameters

    ngMesh->CalcSurfacesOfNode();
    const int startWith = MESHCONST_MESHSURFACE;
    const int endWith   = MESHCONST_OPTSURFACE;
    err = ngLib.GenerateMesh(occgeom, startWith, endWith, ngMesh);

    // Ng_Mesh * ngMeshptr = (Ng_Mesh*) ngLib._ngMesh;
    // int NetgenNodes = Ng_GetNP(ngMeshptr);
    // int NetgenSeg_2D = Ng_GetNSeg_2D(ngMeshptr);
    // int NetgenFaces = Ng_GetNSE(ngMeshptr);
    // std::cout << "\n";
    // std::cout << "Number of nodes, seg, faces, vols: " << NetgenNodes << ", " << NetgenSeg_2D << ", " << NetgenFaces << "\n";
    // std::cout << "err from netgen computation: " << err << "\n";

    // ----------------------------------------------------
    // Fill the SMESHDS with the generated nodes and faces
    // ----------------------------------------------------
    nodeVec.clear();
    FillNodesAndElements( aMesh, helper, ngMesh, nodeVec, ng2smesh, newNetgenCoordinates, newNetgenElements, numberOfPremeshedNodes );
  } // Face iteration

  return (bool) err;
}

std::tuple<bool,bool> NETGENPlugin_NETGEN_2D_ONLY::SetParameteres( SMESH_Mesh& aMesh, const TopoDS_Shape& aShape, 
                                                                  NETGENPlugin_Mesher& aMesher, netgen::Mesh * ngMeshes,
                                                                  netgen::OCCGeometry& occgeoComm, bool isSubMeshSupported )
{
  SMESHDS_Mesh* meshDS = aMesh.GetMeshDS();
    
  aMesher.SetParameters( _hypParameters ); // _hypParameters -> netgen::mparam
  if ( _hypMaxElementArea )
  {
    netgen::mparam.maxh = sqrt( 2. * _hypMaxElementArea->GetMaxArea() / sqrt(3.0) );
  }
  if ( _hypQuadranglePreference )
    netgen::mparam.quad = true;

  // local size is common for all FACEs in aShape?
  const bool isCommonLocalSize = ( !_hypLengthFromEdges && !_hypMaxElementArea && netgen::mparam.uselocalh );
  const bool isDefaultHyp = ( !_hypLengthFromEdges && !_hypMaxElementArea && !_hypParameters );

  if ( isCommonLocalSize ) // compute common local size in ngMeshes[0]
  {
    aMesher.PrepareOCCgeometry( occgeoComm, aShape, aMesh );//, meshedSM );

    // local size set at MESHCONST_ANALYSE step depends on
    // minh, face_maxh, grading and curvaturesafety; find minh if not set by the user
    if ( !_hypParameters || netgen::mparam.minh < DBL_MIN )
    {
      if ( !_hypParameters )
        netgen::mparam.maxh = occgeoComm.GetBoundingBox().Diam() / 3.;
      netgen::mparam.minh = aMesher.GetDefaultMinSize( aShape, netgen::mparam.maxh );
    }
    // set local size depending on curvature and NOT closeness of EDGEs
#ifdef NETGEN_V6
    const double factor = 2; //netgen::occparam.resthcloseedgefac;
#else
    const double factor = netgen::occparam.resthcloseedgefac;
    netgen::occparam.resthcloseedgeenable = false;
    netgen::occparam.resthcloseedgefac = 1.0 + netgen::mparam.grading;
#endif
    occgeoComm.face_maxh = netgen::mparam.maxh;
#ifdef NETGEN_V6
    netgen::OCCParameters occparam;
    netgen::OCCSetLocalMeshSize( occgeoComm, *ngMeshes, netgen::mparam, occparam );
#else
    netgen::OCCSetLocalMeshSize( occgeoComm, *ngMeshes );
#endif
    occgeoComm.emap.Clear();
    occgeoComm.vmap.Clear();

    if ( isSubMeshSupported )
    {
      // set local size according to size of existing segments
      TopTools_IndexedMapOfShape edgeMap;
      TopExp::MapShapes( aMesh.GetShapeToMesh(), TopAbs_EDGE, edgeMap );
      for ( int iE = 1; iE <= edgeMap.Extent(); ++iE )
      {
        const TopoDS_Shape& edge = edgeMap( iE );
        if ( SMESH_Algo::isDegenerated( TopoDS::Edge( edge )))
          continue;
        SMESHDS_SubMesh* smDS = meshDS->MeshElements( edge );
        if ( !smDS ) continue;
        SMDS_ElemIteratorPtr segIt = smDS->GetElements();
        while ( segIt->more() )
        {
          const SMDS_MeshElement* seg = segIt->next();
          SMESH_TNodeXYZ n1 = seg->GetNode(0);
          SMESH_TNodeXYZ n2 = seg->GetNode(1);
          gp_XYZ p = 0.5 * ( n1 + n2 );
          netgen::Point3d pi(p.X(), p.Y(), p.Z());
          ngMeshes->RestrictLocalH( pi, factor * ( n1 - n2 ).Modulus() );
        }
      }
    }
    else
    {
      SMDS_ElemIteratorPtr iteratorElem = meshDS->elementsIterator(SMDSAbs_Edge);
      while ( iteratorElem->more() ) // loop on elements on a geom face
      {
        const SMDS_MeshElement* elem = iteratorElem->next();      
        const SMDS_MeshNode* node0 = elem->GetNode( 0 );
        const SMDS_MeshNode* node1 = elem->GetNode( 1 );
        SMESH_NodeXYZ nXYZ0( node0 );
        SMESH_NodeXYZ nXYZ1( node1 );
        double segmentLength = ( nXYZ0 - nXYZ1 ).Modulus();
        gp_XYZ p = 0.5 * ( nXYZ0 + nXYZ1 );
        netgen::Point3d pi(p.X(), p.Y(), p.Z());
        ngMeshes->RestrictLocalH( pi, factor * segmentLength );
      }
    }

    // set local size defined on shapes
    aMesher.SetLocalSize( occgeoComm, *ngMeshes );
    aMesher.SetLocalSizeForChordalError( occgeoComm, *ngMeshes );
    try {
      ngMeshes->LoadLocalMeshSize( mparam.meshsizefilename );
    } catch (NgException & ex) {
      throw error( COMPERR_BAD_PARMETERS, ex.What() );
    }
  }

  return std::make_tuple( isCommonLocalSize, isDefaultHyp );
}

bool NETGENPlugin_NETGEN_2D_ONLY::ComputeMaxhOfFace( TopoDS_Face& Face, NETGENPlugin_Mesher& aMesher, TSideVector& wires, 
                                                      netgen::OCCGeometry& occgeoComm, bool isDefaultHyp, bool isCommonLocalSize )
{
  size_t nbWires = wires.size();
  if ( !_hypParameters )
  {
    double edgeLength = 0;
    if (_hypLengthFromEdges )
    {
      // compute edgeLength as an average segment length
      smIdType nbSegments = 0;
      for ( size_t iW = 0; iW < nbWires; ++iW )
      {
        edgeLength += wires[ iW ]->Length();
        nbSegments += wires[ iW ]->NbSegments();
      }
      if ( nbSegments )
        edgeLength /= double( nbSegments );
      netgen::mparam.maxh = edgeLength;
    }
    else if ( isDefaultHyp )
    {
      // set edgeLength by a longest segment
      double maxSeg2 = 0;
      for ( size_t iW = 0; iW < nbWires; ++iW )
      {
        const UVPtStructVec& points = wires[ iW ]->GetUVPtStruct();
        if ( points.empty() )
          return error( COMPERR_BAD_INPUT_MESH );
        gp_Pnt pPrev = SMESH_TNodeXYZ( points[0].node );
        for ( size_t i = 1; i < points.size(); ++i )
        {
          gp_Pnt p = SMESH_TNodeXYZ( points[i].node );
          maxSeg2 = Max( maxSeg2, p.SquareDistance( pPrev ));
          pPrev = p;
        }
      }
      edgeLength = sqrt( maxSeg2 ) * 1.05;
      netgen::mparam.maxh = edgeLength;
    }
    if ( netgen::mparam.maxh < DBL_MIN )
      netgen::mparam.maxh = occgeoComm.GetBoundingBox().Diam();

    if ( !isCommonLocalSize )
    {
      netgen::mparam.minh = aMesher.GetDefaultMinSize( Face, netgen::mparam.maxh );
    }
  }
  return true;
}

void NETGENPlugin_NETGEN_2D_ONLY::FillNodesAndElements( SMESH_Mesh& aMesh, SMESH_MesherHelper& helper, netgen::Mesh * ngMesh, 
                                                        vector< const SMDS_MeshNode* >& nodeVec, map<int, const SMDS_MeshNode* >& ng2smesh, 
                                                        std::map<int,std::vector<double>>& newNetgenCoordinates, 
                                                        std::map<int,std::vector<smIdType>>& newNetgenElements, const int numberOfPremeshedNodes )
{
  SMESHDS_Mesh* meshDS = aMesh.GetMeshDS();

  int nbNodes = ngMesh->GetNP();
  int nbFaces = ngMesh->GetNSE();
  nodeVec.resize( nbNodes + 1, 0 );
  // map to index local node numeration to global numeration used by Remote mesher to write the final global resulting mesh
  std::map<smIdType,smIdType> local2globalMap;
  
  smIdType myNewCoordinateCounter = newNetgenCoordinates.size() > 0 ? newNetgenCoordinates.rbegin()->first + 1: numberOfPremeshedNodes+1;
  int myNewFaceCounter = newNetgenElements.size() > 0 ? newNetgenElements.rbegin()->first + 1 : 1;
  // add nodes
  for ( int ngID = 1; ngID <= nbNodes; ++ngID )
  {
    const MeshPoint& ngPoint = ngMesh->Point( ngID );
    // Check if ngPoint is not already present because was in the premeshed mesh boundary
    if ( ng2smesh.count( ngID ) == 0 )
    {
      std::vector<double> netgenCoordinates = {ngPoint(0), ngPoint(1), ngPoint(2)};
      SMDS_MeshNode * node = meshDS->AddNode(ngPoint(0), ngPoint(1), ngPoint(2));
      nodeVec[ ngID ] = node;
      newNetgenCoordinates.insert( make_pair( myNewCoordinateCounter, std::move(netgenCoordinates)) );
      local2globalMap.insert( std::make_pair( node->GetID(), myNewCoordinateCounter ) );
      myNewCoordinateCounter++;
    }
    else
    {
      nodeVec[ ngID ] = ng2smesh[ ngID ];
      local2globalMap.insert( std::make_pair( nodeVec[ ngID ]->GetID(), nodeVec[ ngID ]->GetID() ) );
    }
  }
  // create faces
  int i,j;
  vector<const SMDS_MeshNode*> nodes;
  for ( i = 1; i <= nbFaces ; ++i )
  {
    const Element2d& elem = ngMesh->SurfaceElement(i);
    nodes.resize( elem.GetNP() );
    for (j=1; j <= elem.GetNP(); ++j)
    {
      int pind = elem.PNum(j);
      if ( pind < 1 )
        break;
      nodes[ j-1 ] = nodeVec[ pind ];
    }
    if ( j > elem.GetNP() )
    {
      std::vector<smIdType> netgenCoordinates = { local2globalMap[nodes[0]->GetID()], local2globalMap[nodes[1]->GetID()], local2globalMap[nodes[2]->GetID()] };
      newNetgenElements.insert( std::make_pair( myNewFaceCounter, std::move( netgenCoordinates ) ) );
      helper.AddFace(nodes[0],nodes[1],nodes[2]); 
      myNewFaceCounter++;    
    }
  }
}

void NETGENPlugin_NETGEN_2D_ONLY::FillNodesAndElements( SMESH_Mesh& aMesh, SMESH_MesherHelper& helper, netgen::Mesh * ngMesh, vector< const SMDS_MeshNode* >& nodeVec, int faceId )
{
  SMESHDS_Mesh* meshDS = aMesh.GetMeshDS();

  int nbNodes = ngMesh->GetNP();
  int nbFaces = ngMesh->GetNSE();

  int nbInputNodes = (int) nodeVec.size()-1;
  nodeVec.resize( nbNodes+1, 0 );

  // add nodes
  for ( int ngID = nbInputNodes + 1; ngID <= nbNodes; ++ngID )
  {
    const MeshPoint& ngPoint = ngMesh->Point( ngID );
    SMDS_MeshNode * node = meshDS->AddNode(ngPoint(0), ngPoint(1), ngPoint(2));
    nodeVec[ ngID ] = node;
  }

  // create faces
  int i,j;
  vector<const SMDS_MeshNode*> nodes;
  for ( i = 1; i <= nbFaces ; ++i )
  {
    const Element2d& elem = ngMesh->SurfaceElement(i);
    nodes.resize( elem.GetNP() );
    for (j=1; j <= elem.GetNP(); ++j)
    {
      int pind = elem.PNum(j);
      if ( pind < 1 )
        break;
      nodes[ j-1 ] = nodeVec[ pind ];
      if ( nodes[ j-1 ]->GetPosition()->GetTypeOfPosition() == SMDS_TOP_3DSPACE )
      {
        const PointGeomInfo& pgi = elem.GeomInfoPi(j);
        meshDS->SetNodeOnFace( nodes[ j-1 ], faceId, pgi.u, pgi.v);
      }
    }
    if ( j > elem.GetNP() )
    {
      if ( elem.GetType() == TRIG )
        helper.AddFace(nodes[0],nodes[1],nodes[2]);
      else
        helper.AddFace(nodes[0],nodes[1],nodes[2],nodes[3]);
    }
  }
}
  

//=============================================================================
/*!
 *Here we are going to use the NETGEN mesher
 */
//=============================================================================

bool NETGENPlugin_NETGEN_2D_ONLY::Compute(SMESH_Mesh&         aMesh,
                                          const TopoDS_Shape& aShape)
{
  netgen::multithread.terminate = 0;
  //netgen::multithread.task = "Surface meshing";

  SMESHDS_Mesh* meshDS = aMesh.GetMeshDS();
  SMESH_MesherHelper helper(aMesh);
  helper.SetElementsOnShape( true );

  NETGENPlugin_NetgenLibWrapper ngLib;
  ngLib._isComputeOk = false;

  netgen::Mesh   ngMeshNoLocSize;
  netgen::Mesh * ngMeshes[2] = { (netgen::Mesh*) ngLib._ngMesh,  & ngMeshNoLocSize };

  netgen::OCCGeometry occgeoComm;

  // min / max sizes are set as follows:
  // if ( _hypParameters )
  //    min and max are defined by the user
  // else if ( _hypLengthFromEdges )
  //    min = aMesher.GetDefaultMinSize()
  //    max = average segment len of a FACE
  // else if ( _hypMaxElementArea )
  //    min = aMesher.GetDefaultMinSize()
  //    max = f( _hypMaxElementArea )
  // else
  //    min = aMesher.GetDefaultMinSize()
  //    max = max segment len of a FACE

  NETGENPlugin_Mesher aMesher( &aMesh, aShape, /*isVolume=*/false);
  auto options = SetParameteres( aMesh, aShape, aMesher, ngMeshes[0], occgeoComm );
  const bool isCommonLocalSize  = std::get<0>( options );
  const bool isDefaultHyp       = std::get<1>( options );
  const bool toOptimize = _hypParameters ? _hypParameters->GetOptimize() : true;

  netgen::mparam.uselocalh = toOptimize; // restore as it is used at surface optimization

  // ==================
  // Loop on all FACEs
  // ==================

  vector< const SMDS_MeshNode* > nodeVec;

  TopExp_Explorer fExp( aShape, TopAbs_FACE );
  for ( int iF = 0; fExp.More(); fExp.Next(), ++iF )
  {
    TopoDS_Face F = TopoDS::Face( fExp.Current() /*.Oriented( TopAbs_FORWARD )*/);
    int    faceID = meshDS->ShapeToIndex( F );
    SMESH_ComputeErrorPtr& faceErr = aMesh.GetSubMesh( F )->GetComputeError();

    _quadraticMesh = helper.IsQuadraticSubMesh( F );
    const bool ignoreMediumNodes = _quadraticMesh;

    // build viscous layers if required
    if ( F.Orientation() != TopAbs_FORWARD &&
         F.Orientation() != TopAbs_REVERSED )
      F.Orientation( TopAbs_FORWARD ); // avoid pb with TopAbs_INTERNAL
    SMESH_ProxyMesh::Ptr proxyMesh = StdMeshers_ViscousLayers2D::Compute( aMesh, F );
    if ( !proxyMesh )
      continue;

    // ------------------------
    // get all EDGEs of a FACE
    // ------------------------
    TSideVector wires =
      StdMeshers_FaceSide::GetFaceWires( F, aMesh, ignoreMediumNodes, faceErr, &helper, proxyMesh );
    if ( faceErr && !faceErr->IsOK() )
      continue;
    size_t nbWires = wires.size();
    if ( nbWires == 0 )
    {
      faceErr.reset
        ( new SMESH_ComputeError
          ( COMPERR_ALGO_FAILED, "Problem in StdMeshers_FaceSide::GetFaceWires()" ));
      continue;
    }
    if ( wires[0]->NbSegments() < 3 ) // ex: a circle with 2 segments
    {
      faceErr.reset
        ( new SMESH_ComputeError
          ( COMPERR_BAD_INPUT_MESH, SMESH_Comment("Too few segments: ")<<wires[0]->NbSegments()) );
      continue;
    }

    // ----------------------
    // compute maxh of a FACE
    // ----------------------

    bool setMaxh = ComputeMaxhOfFace( F, aMesher, wires, occgeoComm, isDefaultHyp, isCommonLocalSize );
    if (!setMaxh)
      return setMaxh;
      
    // prepare occgeom
    netgen::OCCGeometry occgeom;
    occgeom.shape = F;
    occgeom.fmap.Add( F );
    occgeom.CalcBoundingBox();
    occgeom.facemeshstatus.SetSize(1);
    occgeom.facemeshstatus = 0;
    occgeom.face_maxh_modified.SetSize(1);
    occgeom.face_maxh_modified = 0;
    occgeom.face_maxh.SetSize(1);
    occgeom.face_maxh = netgen::mparam.maxh;

    // -------------------------
    // Fill netgen mesh
    // -------------------------

    // MESHCONST_ANALYSE step may lead to a failure, so we make an attempt
    // w/o MESHCONST_ANALYSE at the second loop
    int err = 0;
    enum { LOC_SIZE, NO_LOC_SIZE };
    int iLoop = isCommonLocalSize ? 0 : 1;
    for ( ; iLoop < 2; iLoop++ )
    {
      //bool isMESHCONST_ANALYSE = false;
      InitComputeError();

      netgen::Mesh * ngMesh = ngMeshes[ iLoop ];
      ngMesh->DeleteMesh();

      if ( iLoop == NO_LOC_SIZE )
      {
        ngMesh->SetGlobalH ( mparam.maxh );
        ngMesh->SetMinimalH( mparam.minh );
        Box<3> bb = occgeom.GetBoundingBox();
        bb.Increase (bb.Diam()/10);
        ngMesh->SetLocalH (bb.PMin(), bb.PMax(), mparam.grading);
        aMesher.SetLocalSize( occgeom, *ngMesh );
        aMesher.SetLocalSizeForChordalError( occgeoComm, *ngMesh );
        try {
          ngMesh->LoadLocalMeshSize( mparam.meshsizefilename );
        } catch (NgException & ex) {
          return error( COMPERR_BAD_PARMETERS, ex.What() );
        }
      }

      nodeVec.clear();
      faceErr = aMesher.AddSegmentsToMesh( *ngMesh, occgeom, wires, helper, nodeVec,
                                           /*overrideMinH=*/!_hypParameters);
      if ( faceErr && !faceErr->IsOK() )
        break;

      //if ( !isCommonLocalSize )
      //limitSize( ngMesh, mparam.maxh * 0.8);

      // -------------------------
      // Generate surface mesh
      // -------------------------

      const int startWith = MESHCONST_MESHSURFACE;
      const int endWith   = toOptimize ? MESHCONST_OPTSURFACE : MESHCONST_MESHSURFACE;

      SMESH_Comment str;
      try {
        OCC_CATCH_SIGNALS;

        err = ngLib.GenerateMesh(occgeom, startWith, endWith, ngMesh);

        if ( netgen::multithread.terminate )
          return false;
        if ( err )
          str << "Error in netgen::OCCGenerateMesh() at " << netgen::multithread.task;
      }
      catch (Standard_Failure& ex)
      {
        err = 1;
        str << "Exception in  netgen::OCCGenerateMesh()"
            << " at " << netgen::multithread.task
            << ": " << ex.DynamicType()->Name();
        if ( ex.GetMessageString() && strlen( ex.GetMessageString() ))
          str << ": " << ex.GetMessageString();
      }
      catch (...) {
        err = 1;
        str << "Exception in  netgen::OCCGenerateMesh()"
            << " at " << netgen::multithread.task;
      }
      if ( err )
      {
        if ( aMesher.FixFaceMesh( occgeom, *ngMesh, 1 ))
          break;
        if ( iLoop == LOC_SIZE )
        {
          netgen::mparam.minh = netgen::mparam.maxh;
          netgen::mparam.maxh = 0;
          for ( size_t iW = 0; iW < wires.size(); ++iW )
          {
            StdMeshers_FaceSidePtr wire = wires[ iW ];
            const vector<UVPtStruct>& uvPtVec = wire->GetUVPtStruct();
            for ( size_t iP = 1; iP < uvPtVec.size(); ++iP )
            {
              SMESH_TNodeXYZ   p( uvPtVec[ iP ].node );
              netgen::Point3d np( p.X(),p.Y(),p.Z());
              double segLen = p.Distance( uvPtVec[ iP-1 ].node );
              double   size = ngMesh->GetH( np );
              netgen::mparam.minh = Min( netgen::mparam.minh, size );
              netgen::mparam.maxh = Max( netgen::mparam.maxh, segLen );
            }
          }
          //cerr << "min " << netgen::mparam.minh << " max " << netgen::mparam.maxh << endl;
          netgen::mparam.minh *= 0.9;
          netgen::mparam.maxh *= 1.1;
          continue;
        }
        else
        {
          faceErr.reset( new SMESH_ComputeError( COMPERR_ALGO_FAILED, str ));
        }
      }

      // ----------------------------------------------------
      // Fill the SMESHDS with the generated nodes and faces
      // ----------------------------------------------------
      FillNodesAndElements( aMesh, helper, ngMesh, nodeVec, faceID );      

      break;
    } // two attempts
  } // loop on FACEs

  return true;
}

void NETGENPlugin_NETGEN_2D_ONLY::CancelCompute()
{
  SMESH_Algo::CancelCompute();
  netgen::multithread.terminate = 1;
}

//================================================================================
/*!
 * \brief Return progress of Compute() [0.,1]
 */
//================================================================================

double NETGENPlugin_NETGEN_2D_ONLY::GetProgress() const
{
  return -1;
  // const char* task1 = "Surface meshing";
  // //const char* task2 = "Optimizing surface";
  // double& progress = const_cast<NETGENPlugin_NETGEN_2D_ONLY*>( this )->_progress;
  // if ( _progressByTic < 0. &&
  //      strncmp( netgen::multithread.task, task1, 3 ) == 0 )
  // {
  //   progress = Min( 0.25, SMESH_Algo::GetProgressByTic() ); // [0, 0.25]
  // }
  // else //if ( strncmp( netgen::multithread.task, task2, 3 ) == 0)
  // {
  //   if ( _progressByTic < 0 )
  //   {
  //     NETGENPlugin_NETGEN_2D_ONLY* me = (NETGENPlugin_NETGEN_2D_ONLY*) this;
  //     me->_progressByTic = 0.25 / (_progressTic+1);
  //   }
  //   const_cast<NETGENPlugin_NETGEN_2D_ONLY*>( this )->_progressTic++;
  //   progress = Max( progress, _progressByTic * _progressTic );
  // }
  // //cout << netgen::multithread.task << " " << _progressTic << endl;
  // return Min( progress, 0.99 );
}

//=============================================================================
/*!
 *
 */
//=============================================================================

bool NETGENPlugin_NETGEN_2D_ONLY::Evaluate(SMESH_Mesh& aMesh,
                                           const TopoDS_Shape& aShape,
                                           MapShapeNbElems& aResMap)
{
  TopoDS_Face F = TopoDS::Face(aShape);
  if(F.IsNull())
    return false;

  // collect info from edges
  smIdType nb0d = 0, nb1d = 0;
  bool IsQuadratic = false;
  bool IsFirst = true;
  double fullLen = 0.0;
  TopTools_MapOfShape tmpMap;
  for (TopExp_Explorer exp(F, TopAbs_EDGE); exp.More(); exp.Next()) {
    TopoDS_Edge E = TopoDS::Edge(exp.Current());
    if( tmpMap.Contains(E) )
      continue;
    tmpMap.Add(E);
    SMESH_subMesh *aSubMesh = aMesh.GetSubMesh(exp.Current());
    MapShapeNbElemsItr anIt = aResMap.find(aSubMesh);
    if( anIt==aResMap.end() ) {
      SMESH_subMesh *sm = aMesh.GetSubMesh(F);
      SMESH_ComputeErrorPtr& smError = sm->GetComputeError();
      smError.reset( new SMESH_ComputeError(COMPERR_ALGO_FAILED,"Submesh can not be evaluated",this));
      return false;
    }
    std::vector<smIdType> aVec = (*anIt).second;
    nb0d += aVec[SMDSEntity_Node];
    nb1d += std::max(aVec[SMDSEntity_Edge],aVec[SMDSEntity_Quad_Edge]);
    double aLen = SMESH_Algo::EdgeLength(E);
    fullLen += aLen;
    if(IsFirst) {
      IsQuadratic = (aVec[SMDSEntity_Quad_Edge] > aVec[SMDSEntity_Edge]);
      IsFirst = false;
    }
  }
  tmpMap.Clear();

  // compute edge length
  double ELen = 0;
  if (( _hypLengthFromEdges ) || ( !_hypLengthFromEdges && !_hypMaxElementArea )) {
    if ( nb1d > 0 )
      ELen = fullLen / double( nb1d );
  }
  if ( _hypMaxElementArea ) {
    double maxArea = _hypMaxElementArea->GetMaxArea();
    ELen = sqrt(2. * maxArea/sqrt(3.0));
  }
  GProp_GProps G;
  BRepGProp::SurfaceProperties(F,G);
  double anArea = G.Mass();

  const int hugeNb = numeric_limits<int>::max()/10;
  if ( anArea / hugeNb > ELen*ELen )
  {
    SMESH_subMesh *sm = aMesh.GetSubMesh(F);
    SMESH_ComputeErrorPtr& smError = sm->GetComputeError();
    smError.reset( new SMESH_ComputeError(COMPERR_ALGO_FAILED,"Submesh can not be evaluated.\nToo small element length",this));
    return false;
  }
  smIdType nbFaces = (smIdType) ( anArea / ( ELen*ELen*sqrt(3.) / 4 ) );
  smIdType nbNodes = (smIdType) ( ( nbFaces*3 - (nb1d-1)*2 ) / 6 + 1 );
  std::vector<smIdType> aVec(SMDSEntity_Last);
  for(smIdType i=SMDSEntity_Node; i<SMDSEntity_Last; i++) aVec[i]=0;
  if( IsQuadratic ) {
    aVec[SMDSEntity_Node] = nbNodes;
    aVec[SMDSEntity_Quad_Triangle] = nbFaces;
  }
  else {
    aVec[SMDSEntity_Node] = nbNodes;
    aVec[SMDSEntity_Triangle] = nbFaces;
  }
  SMESH_subMesh *sm = aMesh.GetSubMesh(F);
  aResMap.insert(std::make_pair(sm,aVec));

  return true;
}
