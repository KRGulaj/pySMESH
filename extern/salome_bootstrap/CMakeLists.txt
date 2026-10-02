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
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(SalomeBootstrap)
# upper case
STRING(TOUPPER ${PROJECT_NAME} PROJECT_NAME_UC)

# Common CMake macros
# ===================
SET(CONFIGURATION_ROOT_DIR $ENV{CONFIGURATION_ROOT_DIR} CACHE PATH "Path to the Salome CMake configuration files")
IF(EXISTS ${CONFIGURATION_ROOT_DIR})
  LIST(APPEND CMAKE_MODULE_PATH "${CONFIGURATION_ROOT_DIR}/cmake")
  INCLUDE(SalomeMacros NO_POLICY_SCOPE)
ELSE()
  MESSAGE(FATAL_ERROR "We absolutely need the Salome CMake configuration files, please define CONFIGURATION_ROOT_DIR !")
ENDIF()

SALOME_SETUP_VERSION(9.16.0)
MESSAGE(STATUS "Building ${PROJECT_NAME_UC} ${${PROJECT_NAME_UC}_VERSION} from \"${${PROJECT_NAME_UC}_GIT_SHA1}\"")

# Platform setup
# ==============
INCLUDE(SalomeSetupPlatform)
SET(BUILD_SHARED_LIBS TRUE)

# User options
# ============
OPTION(SALOME_BUILD_TESTS "Build SALOME tests" ON)
OPTION(WRAP_PYTHON "Build SALOME wrap python" ON)

# Advanced:
OPTION(SALOME_USE_64BIT_IDS "Size of IDs to refer cells and nodes. 32 bits when OFF (default), 64 bits when ON." OFF)

# Required prerequisites
# ======================
FIND_PACKAGE(SalomePythonInterp REQUIRED)
FIND_PACKAGE(SalomePThread REQUIRED)
FIND_PACKAGE(SalomeCppUnit)

IF(WRAP_PYTHON)
  FIND_PACKAGE(SalomePythonLibs REQUIRED)
  FIND_PACKAGE(SalomeSWIG REQUIRED)
ENDIF()

# Directories
#
# Directories have to be given after prerequisites (to be able to use
# Python version string for example).
# ===========
SET(SALOME_INSTALL_BINS bin/salome CACHE PATH "Install path: SALOME binaries")
SET(SALOME_INSTALL_LIBS lib/salome CACHE PATH "Install path: SALOME libs")
SET(SALOME_INSTALL_HEADERS include/salome CACHE PATH "Install path: SALOME headers")
SET(SALOME_INSTALL_CMAKE salome_adm/cmake_files CACHE PATH "Install path: SALOME CMake files")
SET(SALOME_EXT_INSTALL __SALOME_EXT__ CACHE PATH "Install path: SALOME extension install directory")

SET( ABS_APPLI_PATH __RUN_SALOME__ CACHE PATH "Install path: Absolute appli path")

SET( RUN_SALOME_SCRIPTS ${ABS_APPLI_PATH}/${SALOME_INSTALL_BINS} CACHE PATH
   "Install path: SALOME Launching scripts")
SET( BOOTSTRAP_INSTALL_LIBS ${ABS_APPLI_PATH}/${SALOME_INSTALL_LIBS} CACHE PATH
   "Install path: SALOME lib tools")
SET( BOOTSTRAP_INSTALL_HEADERS ${ABS_APPLI_PATH}/${SALOME_INSTALL_HEADERS} CACHE PATH
   "Install path: SALOME headers")
SET( BOOTSTRAP_INSTALL_CMAKE ${ABS_APPLI_PATH}/${SALOME_INSTALL_CMAKE} CACHE PATH
   "Install path: SALOME cmake files")
SET( SALOME_BOOTSTRAP __SALOME_BOOTSTRAP__ )

# For salome test
SET(BOOTSTRAP_TEST_DIR ${SALOME_EXT_INSTALL}/${SALOME_INSTALL_BINS}/test/BOOTSTRAP)
SET(BOOTSTRAP_TEST_LIB ${SALOME_EXT_INSTALL}/${SALOME_INSTALL_BINS}/test/BOOTSTRAP/lib)

SALOME_PACKAGE_REPORT_AND_CHECK()

# Sources
# ========
ADD_SUBDIRECTORY(__RUN_SALOME__)
ADD_SUBDIRECTORY(__SALOME_BOOTSTRAP__)

# Configuration export
# ====================
INCLUDE(CMakePackageConfigHelpers)

# Add all targets to the build-tree export set
SET(_${PROJECT_NAME}_exposed_targets
  SALOMELog SALOMEBasics SALOMELocalTrace SALOMEException)

# WRAP Python
IF(WRAP_PYTHON)
  LIST(APPEND _${PROJECT_NAME}_exposed_targets
  KERNELBasics)
ENDIF()

# Test targets:
IF(SALOME_BUILD_TESTS)
  LIST(APPEND _${PROJECT_NAME}_exposed_targets
  SALOMELocalTraceTest)
ENDIF()

EXPORT(TARGETS ${_${PROJECT_NAME}_exposed_targets}
  FILE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake)

SET(CPPUNIT_ROOT_DIR "${CPPUNIT_ROOT_DIR}")
SET(PTHREAD_ROOT_DIR "${PTHREAD_ROOT_DIR}")
SET(PYTHON_ROOT_DIR "${PYTHON_ROOT_DIR}")

SALOME_CONFIGURE_PREPARE(CppUnit PThread Python)

CONFIGURE_PACKAGE_CONFIG_FILE(${PROJECT_NAME}Config.cmake.in
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION "${BOOTSTRAP_INSTALL_CMAKE}"
    PATH_VARS BOOTSTRAP_INSTALL_CMAKE CMAKE_INSTALL_PREFIX
       CPPUNIT_ROOT_DIR PTHREAD_ROOT_DIR PYTHON_ROOT_DIR)

WRITE_BASIC_PACKAGE_VERSION_FILE(${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${${PROJECT_NAME_UC}_VERSION}
    COMPATIBILITY AnyNewerVersion)

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/ext_mng/env)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/ext_mng/metadata)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/ext_mng/control_files)

# Install the CMake configuration files:
INSTALL(FILES
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION "${BOOTSTRAP_INSTALL_CMAKE}")

# Install the export set for use with the install-tree
INSTALL(EXPORT ${PROJECT_NAME}TargetGroup DESTINATION "${BOOTSTRAP_INSTALL_CMAKE}"
        FILE ${PROJECT_NAME}Targets.cmake)

# Install salome launcher
INSTALL(PROGRAMS salome 
        DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${PROJECT_BINARY_DIR}/ext_mng DESTINATION ${CMAKE_INSTALL_PREFIX})
