// Copyright (C) 2007-2026  CEA, EDF, OPEN CASCADE
//
// Copyright (C) 2003-2007  OPEN CASCADE, EADS/CCR, LIP6, CEA/DEN,
// CEDRAT, EDF R&D, LEG, PRINCIPIA R&D, BUREAU VERITAS
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

//=============================================================================
// File      : NETGENPlugin_NETGEN_3D_SA.cxx
// Created   : lundi 19 Septembre 2022
// Author    : Yoann AUDOUIN (CEA)
// Project   : SALOME
//=============================================================================
//
//
#include "NETGENPlugin_NETGEN_3D_SA.hxx"

#include "NETGENPlugin_DriverParam.hxx"
#include "NETGENPlugin_Hypothesis.hxx"
#include "StdMeshers_MaxElementVolume.hxx"

#include <SMESH_Gen.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_MesherHelper.hxx>
#include <SMESH_DriverShape.hxx>
#include <SMESH_DriverMesh.hxx>
#include <SMESHDS_Mesh.hxx>


#ifdef WIN32
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif

/*
  Netgen include files
*/

#ifndef OCCGEOMETRY
#define OCCGEOMETRY
#endif
#include <occgeom.hpp>

#ifdef NETGEN_V5
#include <ngexception.hpp>
#endif
#ifdef NETGEN_V6
#include <core/exception.hpp>
#endif

namespace nglib {
#include <nglib.h>
}
namespace netgen {

  NETGENPLUGIN_DLL_HEADER
  extern MeshingParameters mparam;

  NETGENPLUGIN_DLL_HEADER
  extern volatile multithreadt multithread;
}
using namespace nglib;

//=============================================================================
/*!
 * Constructor
 */
//=============================================================================

NETGENPlugin_NETGEN_3D_SA::NETGENPlugin_NETGEN_3D_SA()
  :  NETGENPlugin_NETGEN_3D(0, new SMESH_Gen())
{
  _name = "NETGEN_3D_SA";
}

//=============================================================================
/*!
 * Destructor
 */
//=============================================================================

NETGENPlugin_NETGEN_3D_SA::~NETGENPlugin_NETGEN_3D_SA()
{
  if(_gen)
    delete _gen;
}

/**
 * @brief fill plugin hypothesis from the netgen_params structure
 *
 * @param aParams the structure
 * @param gen SMESH_Gen associate with the SA
 */
void NETGENPlugin_NETGEN_3D_SA::fillHyp(netgen_params aParams)
{
  if(aParams.has_netgen_param){
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

    _hypParameters = dynamic_cast< const NETGENPlugin_Hypothesis *> (hypParameters);
  }
  if(aParams.has_maxelementvolume_hyp){
    _hypMaxElementVolume = new StdMeshers_MaxElementVolume(1, GetGen());
    _maxElementVolume = aParams.maxElementVolume;
  }
  // TODO: Handle viscous layer
}

/**
 * @brief Write a binary file containing information on the elements/nodes
 *        created by the mesher
 *
 * @param nodeVec mapping between the mesh id and the netgen structure id
 * @param ngLib Wrapper on netgen library
 * @param new_element_file Name of the output file
 * @param Netgen_NbOfNodes Number of nodes in the netgen structure
 * @return true if there are some error
 */
bool NETGENPlugin_NETGEN_3D_SA::computeFillNewElementFile(
    std::vector< const SMDS_MeshNode* > &nodeVec,
    NETGENPlugin_NetgenLibWrapper &ngLib,
    std::string new_element_file,
    int &Netgen_NbOfNodes
)
{
  MESSAGE("Writting new elements")
  Ng_Mesh* Netgen_mesh = ngLib.ngMesh();

  int Netgen_NbOfNodesNew = Ng_GetNP(Netgen_mesh);
  int Netgen_NbOfTetra    = Ng_GetNE(Netgen_mesh);
  bool isOK = ( Netgen_NbOfTetra > 0 );
  if ( isOK && !new_element_file.empty() )
  {
    std::ofstream df(new_element_file, ios::out|ios::binary);

    double Netgen_point[3];
    int    Netgen_tetrahedron[4];

    // Writing nodevec (correspondence netgen numbering mesh numbering)
    // Number of nodes
    df.write((char*) &Netgen_NbOfNodes, sizeof(int));
    df.write((char*) &Netgen_NbOfNodesNew, sizeof(int));
    for (int nodeIndex = 1 ; nodeIndex <= Netgen_NbOfNodes; ++nodeIndex )
    {
      //Id of the point
      int id = nodeVec.at(nodeIndex)->GetID();
      df.write((char*) &id, sizeof(int));
    }

    // Writing info on new points
    for (int nodeIndex = Netgen_NbOfNodes +1 ; nodeIndex <= Netgen_NbOfNodesNew; ++nodeIndex )
    {
      Ng_GetPoint(Netgen_mesh, nodeIndex, Netgen_point );
      // Coordinates of the point
      df.write((char *) &Netgen_point, sizeof(double)*3);
    }

    // create tetrahedrons
    df.write((char*) &Netgen_NbOfTetra, sizeof(int));
    for ( int elemIndex = 1; elemIndex <= Netgen_NbOfTetra; ++elemIndex )
    {
      Ng_GetVolumeElement(Netgen_mesh, elemIndex, Netgen_tetrahedron);
      df.write((char*) &Netgen_tetrahedron, sizeof(int)*4);
    }
  }
  return false;
}

/**
 * @brief Compute mesh associated to shape
 *
 * @param aShape the shape
 * @param aMesh the mesh
 * @param aParams netgen_params structure
 * @param new_element_file Name of the file containing new element
 * @param output_mesh Name of the output mesh (if empty it will not be written)
 * @return true if there are some error
 */
bool NETGENPlugin_NETGEN_3D_SA::Compute(
        TopoDS_Shape &aShape,
        SMESH_Mesh& aMesh,
        netgen_params& aParams,
        std::string new_element_file,
        bool output_mesh)
{
  MESSAGE("Compute");
  // vector of nodes in which node index == netgen ID
  vector< const SMDS_MeshNode* > nodeVec;
  NETGENPlugin_NetgenLibWrapper ngLib;
  SMESH_MesherHelper helper(aMesh);
  int startWith = netgen::MESHCONST_MESHVOLUME;
  int endWith   = netgen::MESHCONST_OPTVOLUME;
  int Netgen_NbOfNodes=0;

  // Changing netgen log_file putting it next to new_element_file
  fs::path netgen_log_file = fs::path(new_element_file).remove_filename() / fs::path("NETGEN.out");
  MESSAGE("netgen ouput"<<netgen_log_file.string());

  ngLib.setOutputFile(netgen_log_file.string());

  NETGENPlugin_NETGEN_3D::computeFillNgMesh(aMesh, aShape, nodeVec, ngLib, helper, Netgen_NbOfNodes);

  netgen::OCCGeometry occgeo;
  NETGENPlugin_NETGEN_3D::computePrepareParam(aMesh, ngLib, occgeo, helper, endWith);

  NETGENPlugin_NETGEN_3D::computeRunMesher(occgeo, nodeVec, ngLib._ngMesh, ngLib, startWith, endWith);

  computeFillNewElementFile(nodeVec, ngLib, new_element_file, Netgen_NbOfNodes);

  if(output_mesh)
    NETGENPlugin_NETGEN_3D::computeFillMesh(nodeVec, ngLib, helper, Netgen_NbOfNodes);

  return false;
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
int NETGENPlugin_NETGEN_3D_SA::run(const std::string input_mesh_file,
          const std::string shape_file,
          const std::string hypo_file,
          const std::string element_orientation_file,
          const std::string new_element_file,
          const std::string output_mesh_file)
{

  _element_orientation_file = element_orientation_file;

  std::unique_ptr<SMESH_Mesh> myMesh(_gen->CreateMesh(false));

  SMESH_DriverMesh::importMesh(input_mesh_file, *myMesh);

  // Importing shape
  TopoDS_Shape myShape;
  SMESH_DriverShape::importShape(shape_file, myShape);

  // Importing hypothesis
  netgen_params myParams;

  importNetgenParams(hypo_file, myParams);
  fillHyp(myParams);
  MESSAGE("Meshing with netgen3d");
  int ret = Compute(myShape, *myMesh, myParams,
                      new_element_file,
                      !output_mesh_file.empty());


  if(ret){
    std::cerr << "Meshing failed" << std::endl;
    return ret;
  }

  if(!output_mesh_file.empty()){
    std::string meshName = "MESH";
    SMESH_DriverMesh::exportMesh(output_mesh_file, *myMesh, meshName);
  }

  return ret;
}

/**
 * @brief Compute the list of already meshed Surface elements and info
 *        on their orientation and if they are internal
 *
 * @param aMesh Global Mesh
 * @param aShape Shape associated to the mesh
 * @param proxyMesh pointer to mesh used fo find the elements
 * @param internals information on internal sub shapes
 * @param helper helper associated to the mesh
 * @param listElements map of surface element associated with
 *                     their orientation and internal status
 * @return true if their was some error
 */
bool NETGENPlugin_NETGEN_3D_SA::getSurfaceElements(
    SMESH_Mesh&         aMesh,
    const TopoDS_Shape& aShape,
    SMESH_ProxyMesh::Ptr proxyMesh,
    NETGENPlugin_Internals &internals,
    SMESH_MesherHelper &helper,
    std::map<const SMDS_MeshElement*, tuple<bool, bool>, TIDCompare>& listElements
    )
{
  // To remove compilation warnings
  (void) aShape;
  (void) proxyMesh;
  (void) internals;
  (void) helper;
  SMESHDS_Mesh* meshDS = aMesh.GetMeshDS();

  // Get list of elements + their orientation from element_orientation file
  std::map<vtkIdType, bool> elemOrientation;
  {
    // Setting all element orientation to false if there no element orientation file
    if(_element_orientation_file.empty()){
      MESSAGE("No element orientation file");

      SMDS_ElemIteratorPtr iteratorElem = meshDS->elementsIterator(SMDSAbs_Face);
      while ( iteratorElem->more() ) // loop on elements on a geom face
        {
          // check mesh face
          const SMDS_MeshElement* elem = iteratorElem->next();
          elemOrientation[elem->GetID()] = false;
        }
    } else {
      MESSAGE("Reading from elements from file: " << _element_orientation_file);
      std::ifstream df(_element_orientation_file, ios::binary|ios::in);
      int nbElement;
      bool orient;

      // Warning of the use of vtkIdType (I had issue when run_mesher was compiled with internal vtk) and salome not
      // Sizeof was the same but how he othered the type was different
      // Maybe using another type (uint64_t) instead would be better
      vtkIdType id;
      df.read((char*)&nbElement, sizeof(int));

      for(int ielem=0;ielem<nbElement;++ielem){
        df.read((char*) &id, sizeof(vtkIdType));
        df.read((char*) &orient, sizeof(bool));
        elemOrientation[id] = orient;
      }
    }
  }

  // Adding elements from Mesh
  SMDS_ElemIteratorPtr iteratorElem = meshDS->elementsIterator(SMDSAbs_Face);
  bool isRev;

  bool isIn;

  while ( iteratorElem->more() ) // loop on elements on a geom face
  {
    // check mesh face
    const SMDS_MeshElement* elem = iteratorElem->next();
    if ( !elem ){
      return error( COMPERR_BAD_INPUT_MESH, "Null element encounters");
    }
    if ( elem->NbCornerNodes() != 3 ){
      return error( COMPERR_BAD_INPUT_MESH, "Not triangle element encounters");
    }
    // Keeping only element that are in the element orientation file
    isIn = elemOrientation.count(elem->GetID())==1;
    if(!isIn)
      continue;
    // Get orientation
    // Netgen requires that all the triangle point outside
    isRev = elemOrientation[elem->GetID()];
    listElements[elem] = tuple<bool, bool>(isRev, false);
  }

  return false;
}
