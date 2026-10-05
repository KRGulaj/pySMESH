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
// File      : NETGENPlugin_NETGEN_2D_Remote.hxx
// Created   : mardi 12 Decembre 2023
// Author    : Cesar Conopoima (OCC)
// Project   : SALOME
//=============================================================================
//
#ifndef _NETGENPlugin_NETGEN_2D_REMOTE_HXX_
#define _NETGENPlugin_NETGEN_2D_REMOTE_HXX_

#include "NETGENPlugin_NETGEN_2D_ONLY.hxx"

#include <vector>
#include <map>

class StdMeshers_ViscousLayers;
class StdMeshers_MaxElementVolume;
class NETGENPlugin_Hypothesis;
class NETGENPlugin_NetgenLibWrapper;
class netgen_params;
class SMDS_MeshNode;

using namespace std;

class NETGENPLUGIN_EXPORT NETGENPlugin_NETGEN_2D_Remote: public NETGENPlugin_NETGEN_2D_ONLY
{
 public:
  NETGENPlugin_NETGEN_2D_Remote(int hypId, SMESH_Gen* gen);
  virtual ~NETGENPlugin_NETGEN_2D_Remote();

  // Function whould not be used with remote Computing
  bool CheckHypothesis (SMESH_Mesh&         aMesh,
                        const TopoDS_Shape& aShape,
                        Hypothesis_Status&  aStatus) override {(void)aMesh;(void)aShape;aStatus = HYP_OK;return true;};

  bool Compute(SMESH_Mesh&         aMesh,
              const TopoDS_Shape& aShape) override;

  void setSubMeshesToCompute(SMESH_subMesh * aSubMesh) override;


 private:
 void exportElementOrientation(const std::string output_file);
                                
  void fillParameters(const NETGENPlugin_Hypothesis* hyp,
                      netgen_params &aParams);


};

#endif
