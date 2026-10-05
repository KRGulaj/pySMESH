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

#ifndef _NETGENPLUGIN_DRIVERPARAM_HXX_
#define _NETGENPLUGIN_DRIVERPARAM_HXX_

#include <string>

enum hypoType{
  Hypo = 1, /*the default hypo common for 1D2D,1D2D3D && 3D*/
  Simple2D,
  Simple3D
};

struct netgen_params{
  // Params from NETGENPlugin_Mesher
  // True if _hypParameters is not null
  bool has_netgen_param=true;
  hypoType myType;
  double maxh;
  double minh;
  double segmentsperedge;
  double grading;
  double curvaturesafety;
  int secondorder;
  int quad;
  bool optimize;
  int fineness;
  bool uselocalh;
  bool merge_solids;
  double chordalError;
  int optsteps2d;
  int optsteps3d;
  double elsizeweight;
  int opterrpow;
  bool delaunay;
  bool checkoverlap;
  bool checkchartboundary;
  int closeedgefac;

  // Number of threads for the mesher
  int nbThreads;

  // True if we have a mesh size file or local size info
  bool has_local_size = false;
  std::string meshsizefilename;

  // Params from NETGEN3D
  // True if _hypMaxElementVolume is not null
  bool has_maxelementvolume_hyp=false;
  double maxElementVolume=0.0;

  // Params from NETGEN2D
  bool has_LengthFromEdges_hyp=false;

  /////////////////////////////////////
  // Quantities proper of Simple2D
  int numberOfSegments; // maybe reuse segmentsperedge (?)
  double localLength;
  double maxElementArea;
  bool allowQuadrangles;
  //// Quantities proper of Simple3D
  double maxElementVol;
};

void printNetgenParams(netgen_params& aParams);

void importNetgenParams(const std::string param_file, netgen_params& aParams);
void importDefaultNetgenParams(const std::string param_file, netgen_params& aParams);
void importSimple2D3DNetgenParams(const std::string param_file, netgen_params& aParams, bool is3D );
void exportNetgenParams(const std::string param_file, netgen_params& aParams);
// TODO symple param to be used with netgen1d2d


#endif
