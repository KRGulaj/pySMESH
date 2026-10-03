# SALOME GEOM minimal slice (static) — only GEOMUtils.cxx is compiled, matching
# looooo/SMESH/cmake/Geom/CMakeLists.txt. At V9_16_0 four SMESH files include
# <GEOMUtils.hxx> (SMESH_Mesh.cxx, Controls/SMESH_Controls.cxx,
# Controls/SMESH_ControlsClassifier.cxx, StdMeshers_Cartesian_3D_Grid.hxx), so this is a
# hard dependency of SMESH core. The three GEOMAlgo_AlgoTools functions that GEOMUtils.cxx
# calls since V9_16_0 are compiled into it by patches/geom/GEOMUtils_GEOMAlgo.patch; they use
# TKMesh (BRepMesh_IncrementalMesh), TKV3d (Prs3d) and TKTopAlgo
# (BRepLib_CheckCurveOnSurface), already listed below. ${GEOM_SRC_DIR} is set by the root
# CMakeLists to staged/src/Geom/src.

add_library(Geom STATIC ${GEOM_SRC_DIR}/GEOMUtils/GEOMUtils.cxx)
target_link_libraries(Geom PUBLIC
    Kernel
    TKBRep TKShHealing TKMesh TKGeomAlgo TKTopAlgo TKG2d TKG3d TKV3d TKGeomBase TKBO)
target_include_directories(Geom PUBLIC ${GEOM_SRC_DIR}/GEOMUtils)
