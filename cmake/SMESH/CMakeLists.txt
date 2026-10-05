# SMESH static libraries for pySMESH.
#
# Adapted from looooo/SMESH/cmake/SMESH/CMakeLists.txt. Differences:
#   * STATIC libraries (looooo builds SHARED) — everything is linked into _core.pyd.
#   * per-target ``-D<MODULE>_EXPORTS`` COMPILE_FLAGS are dropped; the root CMakeLists
#     instead defines *every* module's ``*_EXPORTS`` macro globally so the SMESH export
#     headers always pick the ``__declspec(dllexport)`` branch (never dllimport), which is
#     what lets the static libraries cross-reference each other's symbols.
#   * MeshVSLink / MED / CGNS drivers are not built. NETGEN and NETGENPlugin are built
#     outside this file, by cmake/Netgen and cmake/NETGENPlugin.
#   * install() rules dropped (we ship a Python extension, not a C++ SDK).
# ${SMESH_SRC_DIR} is set by the root CMakeLists to staged/src/SMESH/src. The per-target
# OCCT toolkit lists (TK*) are copied verbatim from looooo — they are the load-bearing
# record of which OCCT toolkits each module needs.

# --- SMDS : mesh data structure over VTK ---
file(GLOB SMDS_SRC ${SMESH_SRC_DIR}/SMDS/*.cxx)
add_library(SMDS STATIC ${SMDS_SRC})
target_link_libraries(SMDS PUBLIC Kernel VTK::CommonCore VTK::CommonDataModel)
target_include_directories(SMDS PUBLIC ${SMESH_SRC_DIR}/SMDS)

# --- SMESHUtils : geometry/mesh helpers ---
file(GLOB SMESHUtils_SRC ${SMESH_SRC_DIR}/SMESHUtils/*.cxx)
add_library(SMESHUtils STATIC ${SMESHUtils_SRC})
target_link_libraries(SMESHUtils PUBLIC
    SMDS Geom
    TKShHealing TKPrim TKernel TKBRep TKG2d TKG3d TKGeomBase TKGeomAlgo TKTopAlgo TKMesh
    ${Boost_LIBRARIES})
target_include_directories(SMESHUtils PUBLIC ${SMESH_SRC_DIR}/SMESHUtils)

# --- Controls : quality controls (needs VTK verdict) ---
file(GLOB Controls_SRC ${SMESH_SRC_DIR}/Controls/*.cxx)
add_library(Controls STATIC ${Controls_SRC})
target_link_libraries(Controls PUBLIC
    Kernel Geom SMDS SMESHDS SMESHUtils
    TKernel TKBRep TKG3d TKTopAlgo TKGeomBase TKGeomAlgo VTK::FiltersVerdict)
target_include_directories(Controls PUBLIC ${SMESH_SRC_DIR}/Controls)

# --- SMESHDS : mesh data structure bound to a shape ---
file(GLOB SMESHDS_SRC ${SMESH_SRC_DIR}/SMESHDS/*.cxx)
add_library(SMESHDS STATIC ${SMESHDS_SRC})
target_link_libraries(SMESHDS PUBLIC SMDS Kernel TKBRep)
target_include_directories(SMESHDS PUBLIC ${SMESH_SRC_DIR}/SMESHDS)

# --- Driver + concrete non-MED drivers ---
file(GLOB Driver_SRC ${SMESH_SRC_DIR}/Driver/*.cxx)
add_library(Driver STATIC ${Driver_SRC})
target_link_libraries(Driver PUBLIC SMESHUtils TKernel TKTopAlgo)
target_include_directories(Driver PUBLIC ${SMESH_SRC_DIR}/Driver)

file(GLOB DriverDAT_SRC ${SMESH_SRC_DIR}/DriverDAT/*.cxx)
add_library(DriverDAT STATIC ${DriverDAT_SRC})
target_link_libraries(DriverDAT PUBLIC Driver SMDS)
target_include_directories(DriverDAT PUBLIC ${SMESH_SRC_DIR}/DriverDAT)

file(GLOB DriverGMF_SRC ${SMESH_SRC_DIR}/DriverGMF/*.cxx ${SMESH_SRC_DIR}/DriverGMF/libmesh5.c)
add_library(DriverGMF STATIC ${DriverGMF_SRC})
target_link_libraries(DriverGMF PUBLIC SMESHUtils SMESHDS Driver)
target_include_directories(DriverGMF PUBLIC ${SMESH_SRC_DIR}/DriverGMF)

file(GLOB DriverSTL_SRC ${SMESH_SRC_DIR}/DriverSTL/*.cxx)
add_library(DriverSTL STATIC ${DriverSTL_SRC})
# OCCT 8.0 renamed TKSTL -> TKDESTL (DataExchange reorg).
target_link_libraries(DriverSTL PUBLIC Driver SMDS SMESHUtils TKernel TKDESTL TKTopAlgo TKMesh)
target_include_directories(DriverSTL PUBLIC ${SMESH_SRC_DIR}/DriverSTL)

file(GLOB DriverUNV_SRC ${SMESH_SRC_DIR}/DriverUNV/*.cxx)
add_library(DriverUNV STATIC ${DriverUNV_SRC})
target_link_libraries(DriverUNV PUBLIC SMDS SMESHDS Driver)
target_include_directories(DriverUNV PUBLIC ${SMESH_SRC_DIR}/DriverUNV)

# --- MEFISTO2 : 2D triangulator. SALOME removed it after V9_9_0; aptrte.cxx and the f2c
#     trte.c come from extern/mefisto2 (carried forward, see PROVENANCE.md), and
#     prepare.py stages them where V9_9_0 kept them. StdMeshers_MEFISTO_2D.cxx is staged
#     into StdMeshers/ and compiled by the StdMeshers glob below. ---
file(GLOB MEFISTO2_SRC ${SMESH_SRC_DIR}/MEFISTO2/aptrte.cxx ${SMESH_SRC_DIR}/MEFISTO2/trte.c)
add_library(MEFISTO2 STATIC ${MEFISTO2_SRC})
target_link_libraries(MEFISTO2 PUBLIC Kernel)
target_include_directories(MEFISTO2 PUBLIC ${SMESH_SRC_DIR}/MEFISTO2)

# --- SMESH core (MG_ADAPT.cxx excluded: needs MeshGems) ---
# New in V9_16_0 and compiled: SMESH_SequentialMesh (what SMESH_Gen::CreateMesh now
# returns), SMESH_ParallelMesh (its compute throws on WIN32 upstream; never bound),
# SMESH_MeshLocker, SMESH_DriverShape (STEP/BREP shape files: TKDESTEP, TKXSBase) and
# SMESH_DriverMesh (MED files; its bodies are guarded by WITH_MED in SMESH_Mesh.patch).
file(GLOB SMESH_SRC ${SMESH_SRC_DIR}/SMESH/*.cxx)
list(REMOVE_ITEM SMESH_SRC "${SMESH_SRC_DIR}/SMESH/MG_ADAPT.cxx")
add_library(SMESH STATIC ${SMESH_SRC})
target_link_libraries(SMESH PUBLIC
    Kernel Geom Controls SMESHDS Driver DriverDAT DriverSTL DriverUNV DriverGMF
    TKShHealing TKPrim TKG2d TKCDF TKDESTEP TKXSBase
    ${Boost_LIBRARIES})
target_include_directories(SMESH PUBLIC ${SMESH_SRC_DIR}/SMESH)

# --- StdMeshers : standard hypotheses/algorithms, incl. ViscousLayers ---
# Complete: a plain file(GLOB) with no exclusion. v1 excluded Cartesian_3D, Import_1D2D,
# MaxElementVolume, MaxElementArea and PolyhedronPerSolid_3D for MSVC C2036 (the implicit
# copy of SMESH::Controls::ElementsOnShape against an incomplete nested Classifier). Since
# V9_16_0 Classifier is a complete type in SMESH_ControlsClassifier.hxx, so they compile
# unmodified. The glob also takes the 9.16 Cartesian_3D_Grid, _Hexahedron and _VL files,
# StdMeshers_ViscousLayerBuilder, and the carried-forward StdMeshers_MEFISTO_2D.
file(GLOB StdMeshers_SRC ${SMESH_SRC_DIR}/StdMeshers/*.cxx)
add_library(StdMeshers STATIC ${StdMeshers_SRC})
target_link_libraries(StdMeshers PUBLIC SMESH MEFISTO2 TKOffset TKExpress ${Boost_LIBRARIES})
target_include_directories(StdMeshers PUBLIC ${SMESH_SRC_DIR}/StdMeshers)

# Aggregate list consumed by the root CMakeLists when linking _core.
set(SMESH_STATIC_LIBS
    StdMeshers SMESH Controls SMESHDS SMESHUtils SMDS
    Driver DriverDAT DriverGMF DriverSTL DriverUNV MEFISTO2)
