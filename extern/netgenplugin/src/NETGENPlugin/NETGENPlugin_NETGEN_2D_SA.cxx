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
// File   : NETGENPlugin_NETGEN_2D_SA.cxx
// Author : Cesar Conopoima (OCC)
// Date   : 23/10/2023
// Project   : SALOME
//=============================================================================
//
#include "NETGENPlugin_NETGEN_2D_SA.hxx"
#include "NETGENPlugin_Hypothesis_2D.hxx"
#include "NETGENPlugin_SimpleHypothesis_2D.hxx"
#include "NETGENPlugin_Mesher.hxx"

#include <SMESHDS_Mesh.hxx>
#include <SMESH_ControlsDef.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_MesherHelper.hxx>
#include <SMESH_DriverShape.hxx>
#include <SMESH_DriverMesh.hxx>
#include <StdMeshers_LengthFromEdges.hxx>
#include <StdMeshers_MaxElementArea.hxx>
#include <utilities.h>

#ifdef WIN32
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif

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

NETGENPlugin_NETGEN_2D_SA::NETGENPlugin_NETGEN_2D_SA()
  : NETGENPlugin_NETGEN_2D_ONLY(0, new SMESH_Gen())
{
  _name = "NETGEN_2D_SA";  
}

//=============================================================================
/*!
 *  
 */
//=============================================================================

NETGENPlugin_NETGEN_2D_SA::~NETGENPlugin_NETGEN_2D_SA()
{
}

/**
 * @brief Check presence and content of orientation file. Implemented for completness and future reference.
 *
 * @param element_orientation_file Binary file containing the orientation of surface elemnts
 * @return true, false
 */
bool NETGENPlugin_NETGEN_2D_SA::checkOrientationFile( const std::string element_orientation_file )
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
void NETGENPlugin_NETGEN_2D_SA::fillHyp(const std::string param_file, netgen_params aParams)
{  

  if( aParams.has_netgen_param && aParams.myType == hypoType::Hypo )
  {
    NETGENPlugin_Hypothesis_2D * hyp = new NETGENPlugin_Hypothesis_2D(0, GetGen());
    hyp->SetMaxSize(aParams.maxh);
    hyp->SetMinSize(aParams.minh);
    hyp->SetGrowthRate(aParams.grading);
    hyp->SetQuadAllowed(aParams.quad);
    hyp->SetFineness((NETGENPlugin_Hypothesis::Fineness)aParams.fineness);
    hyp->SetChordalErrorEnabled(aParams.chordalError);
    if(aParams.optimize){
      hyp->SetNbSurfOptSteps(aParams.optsteps2d);
    }
    _hypParameters = dynamic_cast< const NETGENPlugin_Hypothesis_2D *> (hyp);
  }
  else 
  {
    NETGENPlugin_Hypothesis * hyp = new NETGENPlugin_Hypothesis(0, GetGen());
    if (param_file.find("lenghtfromedge") != std::string::npos )
    {
      _hypLengthFromEdges = dynamic_cast<const StdMeshers_LengthFromEdges*> (hyp);
    }
    else if ( param_file.find("maxarea") != std::string::npos )
    {
      StdMeshers_MaxElementArea * hyp = new StdMeshers_MaxElementArea(0, GetGen());

      std::ifstream myfile(param_file);
      std::string line;
      double maxArea;
      std::getline(myfile, line);
      bool hashypothesis = std::stoi(line);
      if ( hashypothesis )
      {
        std::getline(myfile, line);
        maxArea = std::stod(line);
        hyp->SetMaxArea( maxArea );
        _hypMaxElementArea = static_cast<const StdMeshers_MaxElementArea*> (hyp);
      }
      
      myfile.close();
    }
  }  
}


/**
 * @brief Write a binary file containing information on the elements/nodes
 *        created by the mesher
 *
 * @param premeshedNodes map of the premeshed nodes of 1D elements
 * @param newNetgenCoordinates map of the coordinate of new netgen points created in the mesh
 * @param newNetgenElements map of the element triangulation
 * @return true if there are some error
 */
bool NETGENPlugin_NETGEN_2D_SA::fillNewElementFile( std::string new_element_file,
                                                    const int numberOfGlobalPremeshedNodes,
                                                    std::map<int,const SMDS_MeshNode*>& premeshedNodes, 
                                                    std::map<int,std::vector<double>>& newNetgenCoordinates,
                                                    std::map<int,std::vector<smIdType>>& newNetgenElements )
{
  MESSAGE("Writting new elements")

  int NetgenNbOfNodes     = premeshedNodes.size();
  int NetgenNbOfNodesNew  = numberOfGlobalPremeshedNodes + newNetgenCoordinates.size();
  int NetgenNbOfTriangles = newNetgenElements.size();
  bool isOK = ( NetgenNbOfTriangles > 0 );
  if ( isOK && !new_element_file.empty() )
  {
    int    NetgenElement[3];
    std::ofstream df(new_element_file, ios::out|ios::binary);
    // Writing nodevec (correspondence netgen numbering mesh numbering)
    // Number of nodes
    df.write((char*) &numberOfGlobalPremeshedNodes, sizeof(int));
    df.write((char*) &NetgenNbOfNodes, sizeof(int));
    df.write((char*) &NetgenNbOfNodesNew, sizeof(int));
    for (auto k : premeshedNodes )
      df.write((char*) &k.first, sizeof(int));      
        
    // Writing info on new points
    for (auto k : newNetgenCoordinates )
      df.write((char*) k.second.data(), sizeof(double)*3 ); 
      
    // create triangles
    df.write((char*) &NetgenNbOfTriangles, sizeof(int));
    for ( int elemIndex = 1; elemIndex <= NetgenNbOfTriangles; ++elemIndex )
    {
      int nodes = newNetgenElements[ elemIndex ].size();
      for (int i = 0; i < nodes; i++)
        NetgenElement[ i ] = (int) newNetgenElements[ elemIndex ][ i ];
      
      df.write((char*) &NetgenElement, sizeof(int)* nodes );
    }

    df.close();
  }
  return false;
}

/**
 * @brief Compute the mesh based on the 
 *
 * @param input_mesh_file Mesh file (containing 2D elements)
 * @param shape_file Shape file (BREP or STEP format)
 * @param hypo_file Ascii file containing the netgen parameters
 * @param element_orientation_file Binary file containing the orientation of surface elemnts
 * @param new_element_file output file containing info the elements created by the mesher
 * @param output_mesh_file output mesh file (if empty it will not be created)
 * @return true, false
 */
bool NETGENPlugin_NETGEN_2D_SA::Compute( SMESH_Mesh& aMesh, TopoDS_Shape &aShape, std::string new_element_file )
{
  // Nodes on edge are double because each face is treat one by one, so seam edges are iterated twice 
  vector< const SMDS_MeshNode* > nodeVec;
  NETGENPlugin_Mesher aMesher( &aMesh, aShape, false /*isVolume=*/ );
  NETGENPlugin_NetgenLibWrapper ngLib;
  ngLib._isComputeOk = false;

  fs::path netgen_log_file = fs::path(new_element_file).remove_filename() / fs::path("NETGEN.out");
  MESSAGE("netgen ouput"<<netgen_log_file.string());
  ngLib.setOutputFile(netgen_log_file.string());
  
  netgen::OCCGeometry occgeoComm;
  NETGENPlugin_NETGEN_2D_ONLY::SetParameteres( aMesh, aShape, aMesher, ngLib._ngMesh, occgeoComm, false /*submesh is not supported*/ );

  std::map<int,const SMDS_MeshNode*> premeshedNodes;
  std::map<int,std::vector<double>> newNetgenCoordinates;
  std::map<int,std::vector<smIdType>> newNetgenElements;
  const int numberOfTotalPremeshedNodes = aMesh.NbNodes();
  bool compute = NETGENPlugin_NETGEN_2D_ONLY::MapSegmentsToEdges( aMesh, aShape, ngLib, nodeVec,
                                                                  premeshedNodes, newNetgenCoordinates, 
                                                                  newNetgenElements );
  
  compute = fillNewElementFile(new_element_file, 
                                numberOfTotalPremeshedNodes, 
                                premeshedNodes, 
                                newNetgenCoordinates, 
                                newNetgenElements);
  return compute;
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
 * @return int
 */
int NETGENPlugin_NETGEN_2D_SA::run(const std::string input_mesh_file,
                                    const std::string shape_file,
                                    const std::string hypo_file,
                                    const std::string element_orientation_file,
                                    const std::string new_element_file,
                                    const std::string output_mesh_file)
{

  std::unique_ptr<SMESH_Mesh> myMesh(_gen->CreateMesh(false));
  
  SMESH_DriverMesh::importMesh(input_mesh_file, *myMesh);
  // Importing shape
  TopoDS_Shape myShape;
  SMESH_DriverShape::importShape(shape_file, myShape);
  // Importing hypothesis
  netgen_params myParams;  
  importNetgenParams(hypo_file, myParams);
  fillHyp(hypo_file,myParams);  
  MESSAGE("Meshing with netgen2d");
  int ret = 1;
  if ( checkOrientationFile(element_orientation_file) )
  {
    ret = (int) Compute( *myMesh, myShape, new_element_file );

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
    std::cerr << "For NETGENPlugin_NETGEN_2D_SA orientation file should be market with 0 or be empty!" << std::endl;

  return ret;
}
