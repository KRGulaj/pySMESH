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
// File      : NETGENPlugin_NETGEN_2D_SA.hxx
// Author    : Cesar Conopoima (OCC)
// Date      : 23/10/2023
// Project   : SALOME
//=============================================================================
//
#ifndef _NETGENPlugin_NETGEN_1D2D3D_SA_HXX_
#define _NETGENPlugin_NETGEN_1D2D3D_SA_HXX_

#include "NETGENPlugin_Defs.hxx"
#include "NETGENPlugin_Mesher.hxx"
#include "NETGENPlugin_NETGEN_2D3D.hxx"
#include "SMESH_Algo.hxx"
#include "SMESH_Mesh.hxx"

class NETGENPlugin_Mesher;

class NETGENPLUGIN_EXPORT NETGENPlugin_NETGEN_1D2D3D_SA: public NETGENPlugin_NETGEN_2D3D
{
public:
  NETGENPlugin_NETGEN_1D2D3D_SA();
  virtual ~NETGENPlugin_NETGEN_1D2D3D_SA();

  bool Compute(SMESH_Mesh& aMesh, TopoDS_Shape &aShape, std::string new_element_file, bool output_mesh, NETGENPlugin_Mesher::DIM dim );
  
  int run(const std::string input_mesh_file,
          const std::string shape_file,
          const std::string hypo_file,
          const std::string element_orientation_file,
          const std::string new_element_file,
          const std::string output_mesh_file,
          const NETGENPlugin_Mesher::DIM dim );

private:
  
  bool checkOrientationFile( const std::string element_orientation_file );
  void fillHyp(netgen_params aParams);
  bool FillNewElementFile( std::vector< const SMDS_MeshNode* > &nodeVec, NETGENPlugin_NetgenLibWrapper &ngLib,
                            std::string new_element_file, const NETGENPlugin_Mesher::DIM dim );

};

#endif
