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

//  File   : NETGENPlugin_DriverParam.hxx
//  Author : Yoann AUDOUIN, EDF
//  Module : NETGEN
//
#include "NETGENPlugin_DriverParam.hxx"


#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

/**
 * @brief Print content of a netgen_params
 *
 * @param aParams The object to display
 */
void printNetgenParams(netgen_params& aParams){
  if ( aParams.myType == Hypo )
  {
    std::cout << "has_netgen_param: " << aParams.has_netgen_param << std::endl;
    std::cout << "maxh: " << aParams.maxh << std::endl;
    std::cout << "minh: " << aParams.minh << std::endl;
    std::cout << "segmentsperedge: " << aParams.segmentsperedge << std::endl;
    std::cout << "grading: " << aParams.grading << std::endl;
    std::cout << "curvaturesafety: " << aParams.curvaturesafety << std::endl;
    std::cout << "secondorder: " << aParams.secondorder << std::endl;
    std::cout << "quad: " << aParams.quad << std::endl;
    std::cout << "optimize: " << aParams.optimize << std::endl;
    std::cout << "fineness: " << aParams.fineness << std::endl;
    std::cout << "uselocalh: " << aParams.uselocalh << std::endl;
    std::cout << "merge_solids: " << aParams.merge_solids << std::endl;
    std::cout << "chordalError: " << aParams.chordalError << std::endl;
    std::cout << "optsteps2d: " << aParams.optsteps2d << std::endl;
    std::cout << "optsteps3d: " << aParams.optsteps3d << std::endl;
    std::cout << "elsizeweight: " << aParams.elsizeweight << std::endl;
    std::cout << "opterrpow: " << aParams.opterrpow << std::endl;
    std::cout << "delaunay: " << aParams.delaunay << std::endl;
    std::cout << "checkoverlap: " << aParams.checkoverlap << std::endl;
    std::cout << "checkchartboundary: " << aParams.checkchartboundary << std::endl;
    std::cout << "closeedgefac: " << aParams.closeedgefac << std::endl;
    std::cout << "nbThreadMesher: " << aParams.nbThreads << std::endl;
    std::cout << "has_local_size: " << aParams.has_local_size << std::endl;
    std::cout << "meshsizefilename: " << aParams.meshsizefilename << std::endl;
    std::cout << "has_maxelementvolume_hyp: " << aParams.has_maxelementvolume_hyp << std::endl;
    std::cout << "maxElementVolume: " << aParams.maxElementVolume << std::endl;
    std::cout << "has_LengthFromEdges_hyp: " << aParams.has_LengthFromEdges_hyp << std::endl;
  }
}

void importDefaultNetgenParams(const std::string param_file, netgen_params& aParams)
{
  std::ifstream myfile(param_file);
  std::string line;
  // set the default type!
  aParams.myType = Hypo;

  std::getline(myfile, line);
  aParams.has_netgen_param = std::stoi(line);
  std::getline(myfile, line);
  aParams.maxh = std::stod(line);
  std::getline(myfile, line);
  aParams.minh = std::stod(line);
  std::getline(myfile, line);
  aParams.segmentsperedge = std::stod(line);
  std::getline(myfile, line);
  aParams.grading = std::stod(line);
  std::getline(myfile, line);
  aParams.curvaturesafety = std::stod(line);
  std::getline(myfile, line);
  aParams.secondorder = std::stoi(line);
  std::getline(myfile, line);
  aParams.quad = std::stoi(line);
  std::getline(myfile, line);
  aParams.optimize = std::stoi(line);
  std::getline(myfile, line);
  aParams.fineness = std::stoi(line);
  std::getline(myfile, line);
  aParams.uselocalh = std::stoi(line);
  std::getline(myfile, line);
  aParams.merge_solids = std::stoi(line);
  std::getline(myfile, line);
  aParams.chordalError = std::stod(line);
  std::getline(myfile, line);
  aParams.optsteps2d = std::stoi(line);
  std::getline(myfile, line);
  aParams.optsteps3d = std::stoi(line);
  std::getline(myfile, line);
  aParams.elsizeweight = std::stod(line);
  std::getline(myfile, line);
  aParams.opterrpow = std::stoi(line);
  std::getline(myfile, line);
  aParams.delaunay = std::stoi(line);
  std::getline(myfile, line);
  aParams.checkoverlap = std::stoi(line);
  std::getline(myfile, line);
  aParams.checkchartboundary = std::stoi(line);
  std::getline(myfile, line);
  aParams.closeedgefac = std::stoi(line);
  std::getline(myfile, line);
  aParams.nbThreads = std::stoi(line);
  std::getline(myfile, line);
  aParams.has_local_size = std::stoi(line);
  std::getline(myfile, line);
  aParams.meshsizefilename = line;
  std::getline(myfile, line);
  aParams.has_maxelementvolume_hyp = std::stoi(line);
  std::getline(myfile, line);
  aParams.maxElementVolume = std::stod(line);
  std::getline(myfile, line);
  aParams.maxElementVolume = std::stoi(line);
  myfile.close();
}

void importSimple2D3DNetgenParams(const std::string param_file, netgen_params& aParams, bool is3D )
{
  std::ifstream myfile(param_file);
  std::string line;

  aParams.myType = !is3D ? Simple2D : Simple3D;
  std::getline(myfile, line);
  aParams.has_netgen_param = std::stoi(line); // 1
  std::getline(myfile, line);
  aParams.numberOfSegments = std::stoi(line); // segments      (int)
  std::getline(myfile, line);
  aParams.localLength = std::stod(line);      // localLenght  (double)
  std::getline(myfile, line);
  aParams.maxElementArea = std::stod(line);   // max area     (double)
  if ( is3D )
  {
    std::getline(myfile, line);
    aParams.maxElementVol = std::stod(line); // max volume    (double)
  }    
  std::getline(myfile, line);
  aParams.allowQuadrangles = std::stoi(line); // int

  myfile.close();
};

/**
 * @brief Import a param_file into a netgen_params structure
 *
 * @param param_file Name of the file
 * @param aParams Structure to fill
 */
void importNetgenParams(const std::string param_file, netgen_params& aParams){
  
  if ( param_file.find("simple2D") != std::string::npos || param_file.find("simple3D") != std::string::npos /*support simple 2D && 3D*/ )
  {
    importSimple2D3DNetgenParams( param_file, aParams, bool(param_file.find("simple3D") != std::string::npos) );
  }
  else if ( param_file.find("maxarea") == std::string::npos && param_file.find("lenghtfromedge") == std::string::npos /*hypo file for 2D SA*/)
  {
    importDefaultNetgenParams( param_file, aParams );
  }
  else
  {
    aParams.has_netgen_param = false;
  }
};

/**
 * @brief Writes the content of a netgen_param into a file
 *
 * @param param_file the file
 * @param aParams the object
 */
void exportNetgenParams(const std::string param_file, netgen_params& aParams){
  if ( aParams.myType == Hypo ){
    std::ofstream myfile(param_file);
    myfile << aParams.has_netgen_param << std::endl;
    myfile << aParams.maxh << std::endl;
    myfile << aParams.minh << std::endl;
    myfile << aParams.segmentsperedge << std::endl;
    myfile << aParams.grading << std::endl;
    myfile << aParams.curvaturesafety << std::endl;
    myfile << aParams.secondorder << std::endl;
    myfile << aParams.quad << std::endl;
    myfile << aParams.optimize << std::endl;
    myfile << aParams.fineness << std::endl;
    myfile << aParams.uselocalh << std::endl;
    myfile << aParams.merge_solids << std::endl;
    myfile << aParams.chordalError << std::endl;
    myfile << aParams.optsteps2d << std::endl;
    myfile << aParams.optsteps3d << std::endl;
    myfile << aParams.elsizeweight << std::endl;
    myfile << aParams.opterrpow << std::endl;
    myfile << aParams.delaunay << std::endl;
    myfile << aParams.checkoverlap << std::endl;
    myfile << aParams.checkchartboundary << std::endl;
    myfile << aParams.closeedgefac << std::endl;
    myfile << aParams.nbThreads << std::endl;
    myfile << aParams.has_local_size << std::endl;
    myfile << aParams.meshsizefilename << std::endl;
    myfile << aParams.has_maxelementvolume_hyp << std::endl;
    myfile << aParams.maxElementVolume << std::endl;
    myfile << aParams.has_LengthFromEdges_hyp << std::endl;
  }
  else if ( aParams.myType == Simple2D )
  {
    // TODO: Export the 2D && 3D simple versions
  }
};
