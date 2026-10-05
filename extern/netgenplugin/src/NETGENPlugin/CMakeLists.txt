# Copyright (C) 2012-2026  CEA, EDF, OPEN CASCADE
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
#
# See http://www.salome-platform.org/ or email : webmaster.salome@opencascade.com
#

INCLUDE(UseQtExt)

# --- options ---
# additional include directories
INCLUDE_DIRECTORIES(
  ${QT_INCLUDES}
  ${SALOMEBOOTSTRAP_INCLUDE_DIRS}
  ${KERNEL_INCLUDE_DIRS}
  ${GUI_INCLUDE_DIRS}
  ${GEOM_INCLUDE_DIRS}
  ${OpenCASCADE_INCLUDE_DIR}
  ${SMESH_INCLUDE_DIRS}
  ${MEDCOUPLING_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OMNIORB_INCLUDE_DIR}
  ${PROJECT_BINARY_DIR}/idl
)

#Avoid compilation warnings from netgen headers
INCLUDE_DIRECTORIES( SYSTEM  ${NETGEN_INCLUDE_DIRS} ${NETGEN_ZLIB_INCLUDE_DIRS} )

# additional preprocessor / compiler flags
ADD_DEFINITIONS(
  ${QT_DEFINITIONS}
  ${OMNIORB_DEFINITIONS}
  ${OpenCASCADE_DEFINITIONS}
  ${BOOST_DEFINITIONS}
  ${NETGEN_DEFINITIONS}
)

# libraries to link to
SET(_link_LIBRARIES
  ${NETGEN_LIBRARIES}
  ${OpenCASCADE_FoundationClasses_LIBRARIES}
  ${OpenCASCADE_ModelingData_LIBRARIES}
  ${OpenCASCADE_ModelingAlgorithms_LIBRARIES}
  ${OpenCASCADE_DataExchange_LIBRARIES}
  ${OpenCASCADE_Visualization_LIBRARIES}
  ${OpenCASCADE_ApplicationFramework_LIBRARIES}
  ${GEOM_GEOMbasic}
  ${SMESH_SMESHimpl}
  ${SMESH_SMESHEngine}
  ${SMESH_SMESHUtils}
  ${SMESH_StdMeshersEngine}
  ${SMESH_StdMeshers}
  ${SMESH_SMESHDS}
  ${SMESH_SMDS}
  ${SMESH_SMESHControls}
  ${KERNEL_SalomeGenericObj}
  ${KERNEL_SalomeNS}
  ${SALOMEBOOTSTRAP_SALOMELocalTrace}
  ${KERNEL_OpUtil}
  VTK::CommonCore
  VTK::CommonDataModel
  SalomeIDLNETGENPLUGIN
  Qt5::Core
)

# --- headers ---

# header files
SET(NETGENEngine_HEADERS
  NETGENPlugin_NETGEN_3D.hxx
  NETGENPlugin_NETGEN_3D_i.hxx
  NETGENPlugin_NETGEN_2D.hxx
  NETGENPlugin_NETGEN_2D_i.hxx
  NETGENPlugin_NETGEN_2D3D.hxx
  NETGENPlugin_NETGEN_2D3D_i.hxx
  NETGENPlugin_NETGEN_2D_ONLY.hxx
  NETGENPlugin_NETGEN_2D_ONLY_i.hxx
  NETGENPlugin_Hypothesis.hxx
  NETGENPlugin_Hypothesis_i.hxx
  NETGENPlugin_Hypothesis_2D.hxx
  NETGENPlugin_Hypothesis_2D_i.hxx
  NETGENPlugin_Hypothesis_3D_i.hxx
  NETGENPlugin_Hypothesis_2D_ONLY_i.hxx
  NETGENPlugin_SimpleHypothesis_2D.hxx
  NETGENPlugin_SimpleHypothesis_3D.hxx
  NETGENPlugin_SimpleHypothesis_2D_i.hxx
  NETGENPlugin_SimpleHypothesis_3D_i.hxx
  NETGENPlugin_Mesher.hxx
  NETGENPlugin_Remesher_2D.hxx
  NETGENPlugin_Defs.hxx
  NETGENPlugin_DriverParam.hxx
  NETGENPlugin_NETGEN_3D_SA.hxx
  NETGENPlugin_NETGEN_2D_SA.hxx
  NETGENPlugin_NETGEN_1D2D3D_SA.hxx
  NETGENPlugin_NETGEN_3D_Remote.hxx
  NETGENPlugin_NETGEN_3D_Remote_i.hxx
  NETGENPlugin_NETGEN_2D_Remote.hxx
  NETGENPlugin_NETGEN_2D_Remote_i.hxx
)

# --- sources ---

# sources / static
SET(NETGENEngine_SOURCES
  NETGENPlugin_NETGEN_3D.cxx
  NETGENPlugin_NETGEN_3D_i.cxx
  NETGENPlugin_Mesher.cxx
  NETGENPlugin_NETGEN_2D.cxx
  NETGENPlugin_NETGEN_2D_i.cxx
  NETGENPlugin_NETGEN_2D3D.cxx
  NETGENPlugin_NETGEN_2D3D_i.cxx
  NETGENPlugin_NETGEN_2D_ONLY.cxx
  NETGENPlugin_NETGEN_2D_ONLY_i.cxx
  NETGENPlugin_Hypothesis.cxx
  NETGENPlugin_Hypothesis_i.cxx
  NETGENPlugin_Hypothesis_2D.cxx
  NETGENPlugin_Hypothesis_2D_i.cxx
  NETGENPlugin_Hypothesis_3D_i.cxx
  NETGENPlugin_Hypothesis_2D_ONLY_i.cxx
  NETGENPlugin_SimpleHypothesis_2D.cxx
  NETGENPlugin_SimpleHypothesis_3D.cxx
  NETGENPlugin_SimpleHypothesis_2D_i.cxx
  NETGENPlugin_SimpleHypothesis_3D_i.cxx
  NETGENPlugin_Remesher_2D.cxx
  NETGENPlugin_i.cxx
  NETGENPlugin_DriverParam.cxx
  NETGENPlugin_NETGEN_3D_SA.cxx
  NETGENPlugin_NETGEN_2D_SA.cxx
  NETGENPlugin_NETGEN_1D2D3D_SA.cxx
  NETGENPlugin_NETGEN_3D_Remote.cxx
  NETGENPlugin_NETGEN_3D_Remote_i.cxx
  NETGENPlugin_NETGEN_2D_Remote.cxx
  NETGENPlugin_NETGEN_2D_Remote_i.cxx
)

SET(NetgenRunner_SOURCES
  NETGENPlugin_Runner_main.cxx
)

# --- scripts ---

# scripts / static
SET(_bin_SCRIPTS
  __init__.py
  NETGENPluginBuilder.py
)

# --- rules ---

ADD_LIBRARY(NETGENEngine ${NETGENEngine_SOURCES})
TARGET_LINK_LIBRARIES(NETGENEngine ${_link_LIBRARIES} )
INSTALL(TARGETS NETGENEngine EXPORT ${PROJECT_NAME}TargetGroup DESTINATION ${SALOME_INSTALL_LIBS})

ADD_EXECUTABLE(NETGENPlugin_Runner ${NetgenRunner_SOURCES})
TARGET_LINK_LIBRARIES(NETGENPlugin_Runner ${_link_LIBRARIES} NETGENEngine )
INSTALL(TARGETS NETGENPlugin_Runner EXPORT ${PROJECT_NAME}TargetGroup DESTINATION ${SALOME_INSTALL_BINS})

INSTALL(FILES ${NETGENEngine_HEADERS} DESTINATION ${SALOME_INSTALL_HEADERS})

SALOME_INSTALL_SCRIPTS("${_bin_SCRIPTS}" ${SALOME_INSTALL_PYTHON}/salome/NETGENPlugin)
