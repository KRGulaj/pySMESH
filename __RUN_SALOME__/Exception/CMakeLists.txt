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
INCLUDE_DIRECTORIES(
  ${PTHREAD_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../SALOMELocalTrace
  ${CMAKE_CURRENT_SOURCE_DIR}/../Basics
)

SET(HEADERS_HXX
  Utils_SALOME_Exception.hxx
)
SET(SALOMEException_SOURCES
  Utils_SALOME_Exception.cxx
)

ADD_DEFINITIONS(-DBOOTSTRAP_UTILS_EXPORTS)

ADD_LIBRARY(SALOMEException ${SALOMEException_SOURCES})
TARGET_LINK_LIBRARIES(SALOMEException SALOMELocalTrace ${PTHREAD_LIBRARIES})
INSTALL(TARGETS SALOMEException EXPORT ${PROJECT_NAME}TargetGroup DESTINATION ${BOOTSTRAP_INSTALL_LIBS})
INSTALL(FILES ${HEADERS_HXX} DESTINATION ${BOOTSTRAP_INSTALL_HEADERS})
