# pySMESH — static SMESH/KERNEL/GEOM linked into a pybind11 extension (_core).
#
# Layout:
#   extern/{smesh,kernel,geom,mefisto2,pthread}  pristine vendored upstream (never edited)
#   prepare.py                                   stages + patches -> staged/
#   cmake/{Kernel,Geom,SMESH}/CMakeLists.txt     per-module static-lib definitions
#   src/bindings/                                pybind11 module
#
# OCCT + VTK + Boost are dynamic (conda-forge is shared). VTK is resolved at runtime from the
# host env and MUST NOT be bundled into the wheel; OCCT + Boost DLLs are bundled at repair.

cmake_minimum_required(VERSION 3.27)
project(pysmesh LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --------------------------------------------------------------------------- #
# Staged sources (produced by prepare.py)
# --------------------------------------------------------------------------- #
set(STAGED_DIR ${CMAKE_SOURCE_DIR}/staged)
if(NOT EXISTS ${STAGED_DIR}/.prepared)
    message(FATAL_ERROR
        "staged/ tree not found or incomplete. Run:  python prepare.py")
endif()
set(KERNEL_SRC_DIR ${STAGED_DIR}/src/Kernel/src)
set(GEOM_SRC_DIR   ${STAGED_DIR}/src/Geom/src)
set(SMESH_SRC_DIR  ${STAGED_DIR}/src/SMESH/src)
set(PTHREAD_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/extern/pthread)

# --------------------------------------------------------------------------- #
# Dependencies (all resolved from the conda env via CMAKE_PREFIX_PATH)
# --------------------------------------------------------------------------- #
find_package(OpenCASCADE REQUIRED)
find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel FiltersVerdict)
find_package(Boost REQUIRED COMPONENTS filesystem thread serialization regex)
find_package(Threads REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Normalize the Boost link list to imported targets (robust across FindBoost/BoostConfig).
set(Boost_LIBRARIES Boost::filesystem Boost::thread Boost::serialization Boost::regex)

include_directories(SYSTEM ${OpenCASCADE_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
link_directories(${OpenCASCADE_LIBRARY_DIR})

# --------------------------------------------------------------------------- #
# Generate smIdType.hxx  (SALOME_USE_64BIT_IDS ON -> smIdType = int64 on win_amd64)
# --------------------------------------------------------------------------- #
set(SALOME_USE_64BIT_IDS ON)
configure_file(${KERNEL_SRC_DIR}/Basics/smIdType.hxx.in
               ${KERNEL_SRC_DIR}/Basics/smIdType.hxx @ONLY)

# --------------------------------------------------------------------------- #
# Global compile definitions
#   * OCCT/SALOME Windows legacy defines (from looooo/SMESH root CMakeLists).
#   * SALOME_USE_64BIT_IDS + _OCC64 for 64-bit ids/pointers.
#   * Every SMESH module's *_EXPORTS macro: forces the export headers onto the
#     __declspec(dllexport) branch in every TU so the STATIC libraries can reference each
#     other's symbols without dllimport thunks (replaces plan.md's patch 105).
# --------------------------------------------------------------------------- #
add_compile_definitions(
    NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS
    WNT WIN32 _WINDOWS CSFDB
    SALOME_LIGHT   # no-CORBA build: disables omniORB/CORBA branches in KERNEL (OpUtil, ...)
    SALOME_USE_64BIT_IDS _OCC64
    BOOST_ALL_NO_LIB BOOST_ALL_DYN_LINK
    # --- KERNEL / Geom ---
    BASICS_EXPORTS SALOMEBasics_EXPORTS SALOMELOCALTRACE_EXPORTS SALOMELocalTrace_EXPORTS
    UTILS_EXPORTS OpUtil_EXPORTS
    # --- SMESH modules ---
    SMDS_EXPORTS SMESHDS_EXPORTS SMESHUtils_EXPORTS
    SMESHCONTROLS_EXPORTS SMESHControls_EXPORTS
    SMESH_EXPORTS SMESHimpl_EXPORTS
    STDMESHERS_EXPORTS StdMeshers_EXPORTS
    MEFISTO2D_EXPORTS
    MESHDRIVER_EXPORTS MeshDriver_EXPORTS
    MESHDRIVERDAT_EXPORTS MeshDriverDAT_EXPORTS
    MESHDriverGMF_EXPORTS MeshDriverGMF_EXPORTS
    MESHDRIVERSTL_EXPORTS MeshDriverSTL_EXPORTS
    MESHDRIVERUNV_EXPORTS MeshDriverUNV_EXPORTS
    PLUGINUTILS_EXPORTS PluginUtils_EXPORTS)

if(MSVC)
    add_compile_options(/EHsc /bigobj /MP /utf-8 /W1
        /wd4091 /wd4101 /wd4138 /wd4244 /wd4267 /wd4273 /wd4290 /wd4311 /wd4477
        /wd4503 /wd4715 /wd4805 /wd4806 /wd4996)
endif()

# Cross-module include closure (avoids header path whack-a-mole for the monolithic build).
include_directories(
    ${KERNEL_SRC_DIR}/Basics ${KERNEL_SRC_DIR}/SALOMELocalTrace ${KERNEL_SRC_DIR}/Utils
    ${GEOM_SRC_DIR}/GEOMUtils
    ${SMESH_SRC_DIR}/SMDS ${SMESH_SRC_DIR}/SMESHDS ${SMESH_SRC_DIR}/SMESH
    ${SMESH_SRC_DIR}/SMESHUtils ${SMESH_SRC_DIR}/Controls
    ${SMESH_SRC_DIR}/Driver ${SMESH_SRC_DIR}/DriverDAT ${SMESH_SRC_DIR}/DriverGMF
    ${SMESH_SRC_DIR}/DriverSTL ${SMESH_SRC_DIR}/DriverUNV ${SMESH_SRC_DIR}/MEFISTO2
    ${SMESH_SRC_DIR}/StdMeshers)
if(MSVC)
    include_directories(${PTHREAD_INCLUDE_DIR})
endif()

# --------------------------------------------------------------------------- #
# Static libraries
# --------------------------------------------------------------------------- #
include(${CMAKE_SOURCE_DIR}/cmake/Kernel/CMakeLists.txt)
include(${CMAKE_SOURCE_DIR}/cmake/Geom/CMakeLists.txt)
include(${CMAKE_SOURCE_DIR}/cmake/SMESH/CMakeLists.txt)

# --------------------------------------------------------------------------- #
# Python extension module
# --------------------------------------------------------------------------- #
pybind11_add_module(_core
    src/bindings/module.cpp
    src/bindings/shape.cpp
    src/bindings/mesh.cpp
    src/bindings/viscous.cpp
    src/bindings/unify.cpp
    src/bindings/tessellate.cpp
    src/bindings/offset.cpp
    src/bindings/distance.cpp
    src/bindings/classify.cpp)
target_include_directories(_core PRIVATE ${CMAKE_SOURCE_DIR}/src/bindings)
# TKShHealing (ShapeUpgrade_UnifySameDomain), TKOffset (BRepOffsetAPI_*), and TKTopAlgo
# (BRepClass3d_SolidClassifier) are pulled in transitively via the SMESH static libs, but the
# bindings call them directly, so link them explicitly rather than lean on a transitive edge.
# OCCT toolkits are dynamic (bundled at wheel-repair time), not in SMESH_*.
target_link_libraries(_core PRIVATE
    ${SMESH_STATIC_LIBS} Geom Kernel
    VTK::CommonCore VTK::CommonDataModel VTK::FiltersVerdict
    TKShHealing
    TKMesh
    TKOffset
    TKTopAlgo
    ${Boost_LIBRARIES})

# --------------------------------------------------------------------------- #
# _build_info.py — dependency versions + provenance. VTK_VERSION is the ONLY value
# checked at import (see src/pysmesh/__init__.py); the rest are informational.
# --------------------------------------------------------------------------- #
if(NOT DEFINED PYSMESH_VTK_VERSION)
    set(PYSMESH_VTK_VERSION "${VTK_VERSION}")
endif()
if(NOT DEFINED PYSMESH_OCCT_VERSION)
    set(PYSMESH_OCCT_VERSION "${OpenCASCADE_VERSION}")
endif()
if(NOT DEFINED PYSMESH_BOOST_VERSION)
    set(PYSMESH_BOOST_VERSION "${Boost_VERSION}")
endif()
if(NOT DEFINED PYSMESH_GIT_SHA)
    # Best-effort short commit for provenance; "unknown" outside a git checkout.
    set(PYSMESH_GIT_SHA "unknown")
    find_package(Git QUIET)
    if(Git_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE _pysmesh_git_sha
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_pysmesh_git_sha)
            set(PYSMESH_GIT_SHA "${_pysmesh_git_sha}")
        endif()
    endif()
endif()
string(TIMESTAMP PYSMESH_BUILD_DATE "%Y-%m-%dT%H:%M:%SZ" UTC)
if(NOT DEFINED PYSMESH_WITH_NETGEN)
    set(PYSMESH_WITH_NETGEN "False")
endif()

# Local-dev / in-tree layout: the pybind extension and the generated _build_info.py must
# sit INSIDE the pysmesh package dir for the package's relative imports (`from ._core
# import ...`) to resolve. Generate _build_info.py there and copy _core.pyd there after
# each build. Both are git-ignored. (B4 reconciles this with the wheel packaging path.)
set(PKG_DIR ${CMAKE_SOURCE_DIR}/src/pysmesh)
configure_file(${PKG_DIR}/_build_info.py.in ${PKG_DIR}/_build_info.py @ONLY)
add_custom_command(TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:_core> ${PKG_DIR}/$<TARGET_FILE_NAME:_core>
    COMMENT "Copying _core into src/pysmesh for in-place import")

install(TARGETS _core DESTINATION pysmesh)
# The generated _build_info.py is git-ignored, so scikit-build-core's `packages` collection
# skips it; install it explicitly (CMake install is not filtered by .gitignore) so the wheel
# carries it. Without this, `import pysmesh` fails at `from . import _build_info`.
install(FILES ${PKG_DIR}/_build_info.py DESTINATION pysmesh)

# --------------------------------------------------------------------------- #
# Fixture generator (dev-only, OFF by default). Emits tests/fixtures/*.brep via OCCT.
# Not part of the wheel and not run by CI — the committed *.brep files are the fixtures.
# --------------------------------------------------------------------------- #
option(PYSMESH_BUILD_FIXTURE_GEN "Build the OCCT BREP fixture generator" OFF)
if(PYSMESH_BUILD_FIXTURE_GEN)
    add_executable(generate_fixtures tests/fixtures/generate_fixtures.cpp)
    # TKBO: BRepAlgoAPI_Fuse for the split_box fixture (two coplanar-topped half-boxes fused).
    target_link_libraries(generate_fixtures PRIVATE
        TKernel TKMath TKBRep TKG2d TKG3d TKGeomBase TKGeomAlgo TKTopAlgo TKPrim TKMesh TKBO)
endif()
