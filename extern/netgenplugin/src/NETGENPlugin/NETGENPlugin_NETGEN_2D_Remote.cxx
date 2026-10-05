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
// File      : NETGENPlugin_NETGEN_2D_Remote.cxx
// Created   : mardi 12 Decembre 2023
// Author    : Cesar Conopoima (OCC)
// Project   : SALOME
//=============================================================================
//
//
#include "NETGENPlugin_NETGEN_2D_Remote.hxx"

#include "NETGENPlugin_DriverParam.hxx"
#include "NETGENPlugin_Hypothesis.hxx"

#include "Utils_SALOME_Exception.hxx"

#include <SMESH_Gen.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESH_ParallelMesh.hxx>
#include <SMESH_MesherHelper.hxx>
#include <SMESH_DriverShape.hxx>
#include <SMESH_DriverMesh.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_MeshLocker.hxx>

#include <QString>
#include <QProcess>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
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

NETGENPlugin_NETGEN_2D_Remote::NETGENPlugin_NETGEN_2D_Remote(int hypId, SMESH_Gen * gen)
  : NETGENPlugin_NETGEN_2D_ONLY(hypId, gen)
{
  _name = "NETGEN_2D_Remote";
}

//=============================================================================
/*!
 * Destructor
 */
//=============================================================================

NETGENPlugin_NETGEN_2D_Remote::~NETGENPlugin_NETGEN_2D_Remote()
{
}

/**
 * @brief Fill the structure netgen_param with the information from the hypothesis
 *
 * @param hyp the hypothesis
 * @param aParams the netgen_param structure
 */
void NETGENPlugin_NETGEN_2D_Remote::fillParameters(const NETGENPlugin_Hypothesis* hyp, netgen_params &aParams)
{
  aParams.myType             = hypoType::Hypo;
  aParams.maxh               = hyp->GetMaxSize();
  aParams.minh               = hyp->GetMinSize();
  aParams.segmentsperedge    = hyp->GetNbSegPerEdge();
  aParams.grading            = hyp->GetGrowthRate();
  aParams.curvaturesafety    = hyp->GetNbSegPerRadius();
  aParams.secondorder        = hyp->GetSecondOrder() ? 1 : 0;
  aParams.quad               = hyp->GetQuadAllowed() ? 1 : 0;
  aParams.optimize           = hyp->GetOptimize();
  aParams.fineness           = hyp->GetFineness();
  aParams.uselocalh          = hyp->GetSurfaceCurvature();
  aParams.merge_solids       = hyp->GetFuseEdges();
  aParams.chordalError       = hyp->GetChordalErrorEnabled() ? hyp->GetChordalError() : -1.;
  aParams.optsteps2d         = aParams.optimize ? hyp->GetNbSurfOptSteps() : 0;
  aParams.optsteps3d         = aParams.optimize ? hyp->GetNbVolOptSteps()  : 0;
  aParams.elsizeweight       = hyp->GetElemSizeWeight();
  aParams.opterrpow          = hyp->GetWorstElemMeasure();
  aParams.delaunay           = hyp->GetUseDelauney();
  aParams.checkoverlap       = hyp->GetCheckOverlapping();
  aParams.checkchartboundary = hyp->GetCheckChartBoundary();
#ifdef NETGEN_V6
  // std::string
  aParams.meshsizefilename = hyp->GetMeshSizeFile();
  aParams.closeedgefac = 2;
  aParams.nbThreads = hyp->GetNbThreads();
#else
  // const char*
  aParams.meshsizefilename = hyp->GetMeshSizeFile();
  aParams.closeedgefac = 0;
  aParams.nbThreads = 0;
#endif
}

/**
 * @brief write in a binary file the orientation for each surface element of the mesh
 *
 * @param aMesh The mesh
 * @param aShape the shape associated to the mesh
 * @param output_file name of the binary file
 */
void NETGENPlugin_NETGEN_2D_Remote::exportElementOrientation(const std::string output_file)
{
  std::ofstream df(output_file, ios::out|ios::binary);
  int size=0;
  df.write((char*)&size, sizeof(int));
  df.close();
}

/**
 * @brief Compute mesh associate to shape
 *
 * @param aMesh The mesh
 * @param aShape The shape
 * @return true fi there are some error
 */
bool NETGENPlugin_NETGEN_2D_Remote::Compute(SMESH_Mesh&         aMesh,
                                           const TopoDS_Shape& aShape)
{
  {
    SMESH_MeshLocker myLocker(&aMesh);
    SMESH_Hypothesis::Hypothesis_Status hypStatus;
    NETGENPlugin_NETGEN_2D_ONLY::CheckHypothesis(aMesh, aShape, hypStatus);
  }
  SMESH_ParallelMesh& aParMesh = dynamic_cast<SMESH_ParallelMesh&>(aMesh);

  // Temporary folder for run
#ifdef WIN32
  fs::path tmp_folder = aParMesh.GetTmpFolder() / fs::path("Volume-%%%%-%%%%");
#else
  fs::path tmp_folder = aParMesh.GetTmpFolder() / fs::unique_path(fs::path("Volume-%%%%-%%%%"));
#endif
  fs::create_directories(tmp_folder);
  // Using MESH2D generated after all triangles where created.
  fs::path mesh_file=aParMesh.GetTmpFolder() / fs::path("Mesh1D.med"); // read the premeshed elements from 2D version
  fs::path element_orientation_file=tmp_folder / fs::path("element_orientation.dat");
  fs::path new_element_file=tmp_folder / fs::path("new_elements.dat");
  // Not used kept for debug
  //fs::path output_mesh_file=tmp_folder / fs::path("output_mesh.med");
  fs::path shape_file=tmp_folder / fs::path("shape.brep");
  fs::path param_file=tmp_folder / fs::path("netgen_lenghtfromedge.txt"); /*becuase name contain 'lenghtfromedge' set length of 2D from premeshed 1D elements*/
  fs::path log_file=tmp_folder / fs::path("run.log");
  fs::path cmd_file=tmp_folder / fs::path("cmd.txt");
  std::string mesh_name = "MESH";

  {
    SMESH_MeshLocker myLocker(&aMesh);
    //Writing Shape
    SMESH_DriverShape::exportShape(shape_file.string(), aShape);

    //Writing hypo
    // netgen_params aParams;
    // fillParameters(_hypParameters, aParams);
    // exportNetgenParams(param_file.string(), aParams);
    {
      // Simply write the file with the proper name
      std::ofstream myfile(param_file.string());
      myfile << 1 << std::endl;
      myfile.close();
    }
      
    // Exporting element orientation
    exportElementOrientation(element_orientation_file.string());
  }

  // Calling run_mesher
  // Path to mesher script
  fs::path mesher_launcher = fs::path(std::getenv("SMESH_ROOT_DIR"))/
       fs::path("bin")/
       fs::path("salome")/
       fs::path("mesher_launcher.py");


  std::string s_program="python3";
  std::list<std::string> params;
  params.push_back(mesher_launcher.string());
  params.push_back("NETGEN2D");
  params.push_back(mesh_file.string());
  params.push_back(shape_file.string());
  params.push_back(param_file.string());
  params.push_back("--elem-orient-file=" + element_orientation_file.string());
  params.push_back("--new-element-file=" + new_element_file.string());
  // params.push_back("--output-mesh-file=" + output_mesh_file.string());

  // Parallelism method parameters
  int method = aParMesh.GetParallelismMethod();
  if(method == ParallelismMethod::MultiThread){
    params.push_back("--method=local");
  } else if (method == ParallelismMethod::MultiNode){
    params.push_back("--method=cluster");
    params.push_back("--resource="+aParMesh.GetResource());
    params.push_back("--wc-key="+aParMesh.GetWcKey());
    params.push_back("--nb-proc=1");
    params.push_back("--nb-proc-per-node="+to_string(aParMesh.GetNbProcPerNode()));
    params.push_back("--nb-node="+to_string(aParMesh.GetNbNode()));
    params.push_back("--walltime="+aParMesh.GetWalltime());
  } else {
    throw SALOME_Exception("Unknown parallelism method "+method);
  }
  std::string cmd = "";
  cmd += s_program;
  for(auto arg: params){
    cmd += " " + arg;
  }
  MESSAGE("Running command: ");
  MESSAGE(cmd);
  // Writing command in cmd.log
  {
    std::ofstream flog(cmd_file.string());
    flog << cmd << endl;
  }

  // Building arguments for QProcess
  QString program = QString::fromStdString(s_program);
  QStringList arguments;
  for(auto arg : params){
    arguments << arg.c_str();
  }

  QString out_file = log_file.string().c_str();
  QProcess myProcess;
  // myProcess.setProcessChannelMode(QProcess::MergedChannels);
  myProcess.setProcessChannelMode(QProcess::ForwardedChannels);
  myProcess.setStandardOutputFile(out_file);

  myProcess.start(program, arguments);
  // Waiting for process to finish (argument -1 make it wait until the end of
  // the process otherwise it just waits 30 seconds)
  bool finished = myProcess.waitForFinished(-1);
  int ret = myProcess.exitCode();
  if(ret != 0 || !finished){
    // Run crahed
    std::string msg = "Issue with mesh_launcher: \n";
    msg += "See log for more details: " + log_file.string() + "\n";
    msg += cmd + "\n";
    throw SALOME_Exception(msg);
  }

  {
    SMESH_MeshLocker myLocker(&aMesh);
    std::ifstream df(new_element_file.string(), ios::binary);

    int totalPremeshedNodes;
    int NetgenNbOfNodes;
    int NetgenNbOfNodesNew;
    int NetgenNbOfTriangles;
    double NetgenPoint[3];
    int    NetgenTriangle[3];
    int nodeID;

    SMESH_MesherHelper helper(aMesh);
    // This function is mandatory for setElementsOnShape to work
    helper.IsQuadraticSubMesh(aShape);
    helper.SetElementsOnShape( true );

    df.read((char*) &totalPremeshedNodes, sizeof(int));
    // Number of nodes in intial mesh
    df.read((char*) &NetgenNbOfNodes, sizeof(int));
    // Number of nodes added by netgen
    df.read((char*) &NetgenNbOfNodesNew, sizeof(int));

    // Filling nodevec (correspondence netgen numbering mesh numbering)
    vector< const SMDS_MeshNode* > nodeVec ( NetgenNbOfNodesNew + 2 );
    SMESHDS_Mesh * meshDS = helper.GetMeshDS();
    for (int nodeIndex = 1; nodeIndex <= NetgenNbOfNodes; ++nodeIndex )
    {
      //Id of the point
      df.read((char*) &nodeID, sizeof(int));
      nodeVec.at(nodeID) = meshDS->FindNode(nodeID);
    }

    // Add new points and update nodeVec
    for (int nodeIndex = totalPremeshedNodes + 1; nodeIndex <= NetgenNbOfNodesNew; ++nodeIndex )
    {
      df.read((char *) &NetgenPoint, sizeof(double)*3);
      nodeVec.at(nodeIndex) = helper.AddNode(NetgenPoint[0], NetgenPoint[1], NetgenPoint[2]);
    }

    // Add triangles
    df.read((char*) &NetgenNbOfTriangles, sizeof(int));
    for ( int elemIndex = 1; elemIndex <= NetgenNbOfTriangles; ++elemIndex )
    {
      df.read((char*) &NetgenTriangle, sizeof(int)*3);
      if ( nodeVec.at( NetgenTriangle[0] ) && nodeVec.at( NetgenTriangle[1] ) && nodeVec.at( NetgenTriangle[2] ) )
        helper.AddFace(nodeVec.at( NetgenTriangle[0] ), nodeVec.at( NetgenTriangle[1] ), nodeVec.at( NetgenTriangle[2] ) );            
    }
  }

  return true;
}

/**
 * @brief Assign submeshes to compute
 *
 * @param aSubMesh submesh to add
 */
void NETGENPlugin_NETGEN_2D_Remote::setSubMeshesToCompute(SMESH_subMesh * aSubMesh)
{
  SMESH_MeshLocker myLocker(aSubMesh->GetFather());
  SMESH_Algo::setSubMeshesToCompute(aSubMesh);
}
