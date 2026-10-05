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

// NETGENPlugin : C++ implementation
// File   : NETGENPlugin_NETGEN_2D3D_SA.cxx
// Author : Cesar Conopoima (OCC)
// Date   : 01/11/2023
// Project   : SALOME
//=============================================================================
//

#include "NETGENPlugin_DriverParam.hxx"
#include "NETGENPlugin_Hypothesis.hxx"
#include "NETGENPlugin_SimpleHypothesis_3D.hxx"

#include "NETGENPlugin_NETGEN_2D.hxx"
#include "NETGENPlugin_NETGEN_1D2D3D_SA.hxx"

#include <SMESHDS_Mesh.hxx>
#include <SMESH_ControlsDef.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_MesherHelper.hxx>
#include <SMESH_DriverShape.hxx>
#include <SMESH_DriverMesh.hxx>
#include <utilities.h>

#include <list>

namespace nglib {
#include <nglib.h>
}
#include <meshing.hpp>

using namespace nglib;

//=============================================================================
/*!
 *  
 */
//=============================================================================

NETGENPlugin_NETGEN_1D2D3D_SA::NETGENPlugin_NETGEN_1D2D3D_SA()
  : NETGENPlugin_NETGEN_2D3D(0, new SMESH_Gen())
{
  _name = "NETGEN_1D2D3D_SA";  
}

//=============================================================================
/*!
 *  
 */
//=============================================================================

NETGENPlugin_NETGEN_1D2D3D_SA::~NETGENPlugin_NETGEN_1D2D3D_SA()
{
}

/**
 * @brief Check presence and content of orientation file. Implemented for completness and future reference.
 *
 * @param element_orientation_file Binary file containing the orientation of surface elemnts
 * @return true, false
 */
bool NETGENPlugin_NETGEN_1D2D3D_SA::checkOrientationFile( const std::string element_orientation_file )
{
  if(element_orientation_file.empty()){
      MESSAGE("No element orientation file");
      return true;
    } else {
      MESSAGE("Reading from elements from file: " << element_orientation_file);
      // By construction the orientation file written by Remote version has a zero written to mark no need of orientation in 2D meshing
      int nbElement;
      std::ifstream df(element_orientation_file, ios::binary|ios::in);
      df.read((char*)&nbElement, sizeof(int));
      df.close();
      return (nbElement == 0);
    }
}

/**
 * @brief fill plugin hypothesis from the netgen_params structure
 *
 * @param aParams the structure
 */
void NETGENPlugin_NETGEN_1D2D3D_SA::fillHyp(netgen_params aParams)
{  
 if( aParams.myType == hypoType::Hypo )
  {
    NETGENPlugin_Hypothesis * hypParameters = new NETGENPlugin_Hypothesis(0, GetGen());

    hypParameters->SetMaxSize(aParams.maxh);
    hypParameters->SetMinSize(aParams.minh);
    hypParameters->SetNbSegPerEdge(aParams.segmentsperedge);
    hypParameters->SetGrowthRate(aParams.grading);
    hypParameters->SetNbSegPerRadius(aParams.curvaturesafety);
    hypParameters->SetSecondOrder(aParams.secondorder);
    hypParameters->SetQuadAllowed(aParams.quad);
    hypParameters->SetOptimize(aParams.optimize);
    hypParameters->SetFineness((NETGENPlugin_Hypothesis::Fineness)aParams.fineness);
    hypParameters->SetSurfaceCurvature(aParams.uselocalh);
    hypParameters->SetFuseEdges(aParams.merge_solids);
    hypParameters->SetChordalErrorEnabled(aParams.chordalError);
    if(aParams.optimize){
      hypParameters->SetNbSurfOptSteps(aParams.optsteps2d);
      hypParameters->SetNbVolOptSteps(aParams.optsteps3d);
    }
    hypParameters->SetElemSizeWeight(aParams.elsizeweight);
    hypParameters->SetWorstElemMeasure(aParams.opterrpow);
    hypParameters->SetUseDelauney(aParams.delaunay);
    hypParameters->SetCheckOverlapping(aParams.checkoverlap);
    hypParameters->SetCheckChartBoundary(aParams.checkchartboundary);
    hypParameters->SetMeshSizeFile(aParams.meshsizefilename);

    _hypothesis = dynamic_cast< const NETGENPlugin_Hypothesis *> (hypParameters);
  }
  else if ( aParams.myType == hypoType::Simple2D )
  {
    NETGENPlugin_SimpleHypothesis_2D * hypParameters = new NETGENPlugin_SimpleHypothesis_2D(0, GetGen());
    
    // mandatory to fill in this branch case!
    // Number of segments   (int)
    // localLenght          (double)
    // maxElement area      (double)
    // GetAllowQuadrangles  (bool)
    hypParameters->SetNumberOfSegments( aParams.numberOfSegments );
    if ( !aParams.numberOfSegments )
      hypParameters->SetLocalLength( aParams.localLength );      
    hypParameters->SetMaxElementArea( aParams.maxElementArea );
    hypParameters->SetAllowQuadrangles( aParams.allowQuadrangles );
    _hypothesis = dynamic_cast< const NETGENPlugin_SimpleHypothesis_2D *> (hypParameters);
  }
  else if ( aParams.myType == hypoType::Simple3D )
  {
    NETGENPlugin_SimpleHypothesis_3D * hypParameters = new NETGENPlugin_SimpleHypothesis_3D(0, GetGen());
    
    // mandatory to fill in this branch case!
    // Number of segments   (int)
    // localLenght          (double)
    // maxElement area      (double)
    // maxElement volume    (double)
    // GetAllowQuadrangles  (bool)
    hypParameters->SetNumberOfSegments( aParams.numberOfSegments );
    if ( !aParams.numberOfSegments )
      hypParameters->SetLocalLength( aParams.localLength );      
    hypParameters->SetMaxElementArea( aParams.maxElementArea );
    hypParameters->SetMaxElementVolume( aParams.maxElementVol );
    hypParameters->SetAllowQuadrangles( aParams.allowQuadrangles );

    _hypothesis = dynamic_cast< const NETGENPlugin_SimpleHypothesis_3D *> (hypParameters);
  }
}

/**
 * @brief Write a binary file containing information on the elements/nodes
 *        created by the netgen mesher
 *
 * @param nodeVec mapping between the mesh id and the netgen structure id
 * @param ngLib Wrapper on netgen library
 * @param new_element_file Name of the output file
 * @param NumOfPremeshedNodes Number of nodes in the netgen structure
 * @return true if there are some error
 */
bool NETGENPlugin_NETGEN_1D2D3D_SA::FillNewElementFile( std::vector< const SMDS_MeshNode* > &nodeVec, NETGENPlugin_NetgenLibWrapper &ngLib, 
                                                        std::string new_element_file, const NETGENPlugin_Mesher::DIM dim )
{

  // Particularities: As submeshing is not supported nodeVect is empty and NumberOfPremeshedNodes is also zero
  Ng_Mesh* NetgenMesh  = ngLib.ngMesh();
  int NetgenNodes       = Ng_GetNP(NetgenMesh);
  int NetgenSeg2D       = Ng_GetNSeg_2D( NetgenMesh );
  int NetgenFaces       = Ng_GetNSE(NetgenMesh);
  int NetgenVols        = Ng_GetNE(NetgenMesh);
  int segmentId;
  bool isOK = ( NetgenNodes > 0 && NetgenSeg2D > 0 );
  if ( isOK && !new_element_file.empty() )
  {
    MESSAGE("Writting new elements")
    std::ofstream df(new_element_file, ios::out|ios::binary);

    double NetgenPoint[3];
    int    NetgenSegment[2];
    int    NetgenSurface[8];
    int    NetgenVolumens[10];

    // Writing nodevec (correspondence netgen numbering mesh numbering)
    // Number of nodes
    const int NumOfPremeshedNodes = nodeVec.size();
    df.write((char*) &NumOfPremeshedNodes, sizeof(int));
    df.write((char*) &NetgenNodes, sizeof(int));

    for (int nodeIndex = 1 ; nodeIndex <= NumOfPremeshedNodes; ++nodeIndex )
    {
      //Id of the point
      int id = nodeVec.at(nodeIndex)->GetID();
      df.write((char*) &id, sizeof(int));
    }

    // Writing all new points
    for (int nodeIndex = NumOfPremeshedNodes + 1; nodeIndex <= NetgenNodes; ++nodeIndex )
    {
      Ng_GetPoint( NetgenMesh, nodeIndex, NetgenPoint );
      // Coordinates of the point
      df.write((char *) &NetgenPoint, sizeof(double)*3);
    }

    if ( dim >= NETGENPlugin_Mesher::D1 )
    {
      // create segments at boundaries.
      df.write((char*) &NetgenSeg2D, sizeof(int));
      for ( int elemIndex = 1; elemIndex <= NetgenSeg2D; ++elemIndex )
      {
        Ng_GetSegment_2D( NetgenMesh, elemIndex, NetgenSegment, &segmentId );
        df.write((char*) &NetgenSegment, sizeof(int) * 2 );
      }
    }
    if ( dim >= NETGENPlugin_Mesher::D2 ) 
    {
      // create surface elements.
      df.write((char*) &NetgenFaces, sizeof(int));
      for ( int elemIndex = 1; elemIndex <= NetgenFaces; ++elemIndex )
      {
        nglib::Ng_Surface_Element_Type elemType = Ng_GetSurfaceElement( NetgenMesh, elemIndex, NetgenSurface );
        switch (elemType)
        {
          case nglib::NG_TRIG:
          {
            df.write((char*) &NetgenSurface, sizeof(int) * 3 );
            break;
          }            
          case nglib::NG_QUAD:
          {
            df.write((char*) &NetgenSurface, sizeof(int) * 4 );
            break;
          }            
          case nglib::NG_TRIG6:
          {
            df.write((char*) &NetgenSurface, sizeof(int) * 6 );
            break;
          }            
          case nglib::NG_QUAD8:
          {
            df.write((char*) &NetgenSurface, sizeof(int) * 8 );
            break;
          }
          default:
          { break; }         
        }
      }
    }
    if ( dim >= NETGENPlugin_Mesher::D3 ) 
    {
      // create volume elements.
      df.write((char*) &NetgenVols, sizeof(int));
      for ( int elemIndex = 1; elemIndex <= NetgenVols; ++elemIndex )
      {
        nglib::Ng_Volume_Element_Type elemType =  Ng_GetVolumeElement( NetgenMesh, elemIndex, NetgenVolumens );
        switch (elemType)
        {        
          case nglib::NG_TET:
          {
            df.write((char*) &NetgenVolumens, sizeof(int) * 4 );
            break;
          } 
          case nglib::NG_PYRAMID:
          {
            df.write((char*) &NetgenVolumens, sizeof(int) * 5 );
            break;
          }
          case nglib::NG_PRISM:
          {
            df.write((char*) &NetgenVolumens, sizeof(int) * 6 );
            break;
          }
          case nglib::NG_TET10:
          {
            df.write((char*) &NetgenVolumens, sizeof(int) * 10 );
            break;
          }
          default:
          { break; }
        }
      }
    }          
    df.close();
  }
  return false;
}

/**
 * @brief Compute the mesh based on the 
 *
 * @param aMesh the read Mesh
 * @param aShape the loaded shape
 * @param new_element_file output file containing info the elements created by the mesher
 * @param output_mesh whether or not write the created elements into the mesh
 * @param dim the dimension to be meshed.
 * @return negation of mesh fail: true, false
 */
bool NETGENPlugin_NETGEN_1D2D3D_SA::Compute(SMESH_Mesh& aMesh, TopoDS_Shape &aShape, std::string new_element_file, bool output_mesh, NETGENPlugin_Mesher::DIM dim )
{
  //
  netgen::multithread.terminate = 0;
  NETGENPlugin_Mesher mesher(&aMesh, aShape, /*is3D = */ false );
  mesher.SetParameters(dynamic_cast<const NETGENPlugin_Hypothesis*>(_hypothesis));
  if ( dim == NETGENPlugin_Mesher::D3 )
    mesher.SetParameters(dynamic_cast<const NETGENPlugin_SimpleHypothesis_3D*>(_hypothesis));
  else
    mesher.SetParameters(dynamic_cast<const NETGENPlugin_SimpleHypothesis_2D*>(_hypothesis));
  NETGENPlugin_NetgenLibWrapper ngLib;
  vector< const SMDS_MeshNode* > nodeVec;
  bool err = mesher.Compute( ngLib, nodeVec, output_mesh, dim );  
  FillNewElementFile( nodeVec, ngLib, new_element_file, dim );
  return err;
}

/**
 * @brief Running the mesher on the given files
 *
 * @param input_mesh_file Mesh file (containing 2D elements)
 * @param shape_file Shape file (BREP or STEP format)
 * @param hypo_file Ascii file containing the netgen parameters
 * @param element_orientation_file Binary file containing the orientation of surface elemnts
 * @param new_element_file output file containing info the elements created by the mesher
 * @param output_mesh_file output mesh file (if empty it will not be created)
 * @return negation of mesh fail: true, false
 */
int NETGENPlugin_NETGEN_1D2D3D_SA::run(const std::string input_mesh_file,
                                      const std::string shape_file,
                                      const std::string hypo_file,
                                      const std::string element_orientation_file,
                                      const std::string new_element_file,
                                      const std::string output_mesh_file, 
                                      const NETGENPlugin_Mesher::DIM dim )
  {

  // _element_orientation_file = element_orientation_file;

  std::unique_ptr<SMESH_Mesh> myMesh(_gen->CreateMesh(false));

  // Importing mesh
  SMESH_DriverMesh::importMesh(input_mesh_file, *myMesh);
  // Importing shape
  TopoDS_Shape myShape;
  SMESH_DriverShape::importShape(shape_file, myShape);
  // Importing hypothesis
  netgen_params myParams;  
  importNetgenParams(hypo_file, myParams);
  fillHyp(myParams);
  int ret = 1;

  if ( checkOrientationFile(element_orientation_file) )
  {
    ret = Compute( *myMesh, myShape, new_element_file, !output_mesh_file.empty(), dim );
    if(ret){
      std::cerr << "Meshing failed" << std::endl;
      return ret;
    }

    if(!output_mesh_file.empty()){
      std::string meshName = "MESH";
      SMESH_DriverMesh::exportMesh(output_mesh_file, *myMesh, meshName);
    }
  }
  else
    std::cerr << "For NETGENPlugin_NETGEN_1D2D3D_SA, orientation file should be market with 0 or be empty!" << std::endl;
    

  return ret;
}
