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

include(${SWIG_USE_FILE})

INCLUDE_DIRECTORIES(
  ${PTHREAD_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../Basics
)

SET(KERNELBasics_SOURCES
  KernelBasis.cxx
  Monitoring.cxx
)

ADD_LIBRARY(KERNELBasics ${KERNELBasics_SOURCES})
INSTALL(TARGETS KERNELBasics EXPORT ${PROJECT_NAME}TargetGroup DESTINATION ${BOOTSTRAP_INSTALL_LIBS})

TARGET_LINK_LIBRARIES(KERNELBasics SALOMELog SALOMEBasics ${PLATFORM_LIBS} ${PTHREAD_LIBRARIES})

FILE(GLOB KERNELBasics_HEADERS_HXX "${CMAKE_CURRENT_SOURCE_DIR}/*.hxx")
INSTALL(FILES ${KERNELBasics_HEADERS_HXX} DESTINATION ${BOOTSTRAP_INSTALL_HEADERS})

SET(KernelBasis_HEADERS KernelBasis.hxx KernelBasis.i)
SET(KernelBasis_SOURCES ${KernelBasis_HEADERS})
SET_SOURCE_FILES_PROPERTIES(KernelBasis.i PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(KernelBasis.i PROPERTIES SWIG_FLAGS "-py3")
SET_SOURCE_FILES_PROPERTIES(KernelBasis_wrap.cpp PROPERTIES COMPILE_FLAGS "-DHAVE_CONFIG_H")
SET(_swig_SCRIPTS ${CMAKE_CURRENT_BINARY_DIR}/KernelBasis.py )
IF(${CMAKE_VERSION} VERSION_LESS "3.8.0")
  SWIG_ADD_MODULE(KernelBasis python ${KernelBasis_SOURCES})
ELSE()
  SWIG_ADD_LIBRARY(KernelBasis LANGUAGE python SOURCES ${KernelBasis_SOURCES})
ENDIF()
IF(${MACHINE} STREQUAL WINDOWS)
  SET_TARGET_PROPERTIES(_KernelBasis PROPERTIES DEBUG_OUTPUT_NAME _KernelBasis_d)
ENDIF(${MACHINE} STREQUAL WINDOWS)
#set_target_properties(_KernelBasis PROPERTIES SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)

SWIG_LINK_LIBRARIES(KernelBasis ${PYTHON_LIBRARIES} ${PLATFORM_LIBS} KERNELBasics SALOMEBasics)
install(TARGETS _KernelBasis DESTINATION ${BOOTSTRAP_INSTALL_LIBS})
install(FILES ${KernelBasis_HEADERS} DESTINATION ${BOOTSTRAP_INSTALL_HEADERS})
SALOME_INSTALL_SCRIPTS("${_swig_SCRIPTS}" ${BOOTSTRAP_INSTALL_LIBS}
                        EXTRA_DPYS "${SWIG_MODULE_KernelBasis_REAL_NAME}")
