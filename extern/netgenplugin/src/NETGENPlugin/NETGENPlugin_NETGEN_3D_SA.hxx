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
// File      : NETGENPlugin_NETGEN_3D_SA.hxx
// Created   : lundi 19 Septembre 2022
// Author    : Yoann AUDOUIN (EDF)
// Project   : SALOME
//=============================================================================
//
#ifndef _NETGENPlugin_NETGEN_3D_SA_HXX_
#define _NETGENPlugin_NETGEN_3D_SA_HXX_

#include "NETGENPlugin_NETGEN_3D.hxx"

#include <vector>
#include <map>

class NETGENPlugin_NetgenLibWrapper;
class netgen_params;
class SMDS_MeshNode;

using namespace std;

class NETGENPLUGIN_EXPORT NETGENPlugin_NETGEN_3D_SA: public NETGENPlugin_NETGEN_3D
{
 public:
  NETGENPlugin_NETGEN_3D_SA();
  virtual ~NETGENPlugin_NETGEN_3D_SA();

  void fillHyp(netgen_params aParams);
  bool Compute(TopoDS_Shape &aShape, SMESH_Mesh& aMesh, netgen_params& aParams,
                       std::string new_element_file, bool output_mesh);

  int run(const std::string input_mesh_file,
          const std::string shape_file,
          const std::string hypo_file,
          const std::string element_orientation_file,
          const std::string new_element_file,
          const std::string output_mesh_file);

 protected:

  bool computeFillNewElementFile(
    std::vector< const SMDS_MeshNode* > &nodeVec,
    NETGENPlugin_NetgenLibWrapper &ngLib,
    std::string new_element_file,
    int &Netgen_NbOfNodes);

  bool getSurfaceElements(
    SMESH_Mesh&         aMesh,
    const TopoDS_Shape& aShape,
    SMESH_ProxyMesh::Ptr proxyMesh,
    NETGENPlugin_Internals &internals,
    SMESH_MesherHelper &helper,
    std::map<const SMDS_MeshElement*, tuple<bool, bool>, TIDCompare>& listElements
    ) override;

   std::string _element_orientation_file="";
};

#endif
