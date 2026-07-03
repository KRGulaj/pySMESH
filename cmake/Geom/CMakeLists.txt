# SALOME GEOM minimal slice (static) — only GEOMUtils.cxx is compiled, matching
# looooo/SMESH/cmake/Geom/CMakeLists.txt. Three SMESH core/algorithm files
# (SMESH_Mesh.cxx, Controls/SMESH_Controls.cxx, StdMeshers_Cartesian_3D.cxx) #include
# <GEOMUtils.hxx>, so this is a hard dependency of SMESH core. ${GEOM_SRC_DIR} is set by the
# root CMakeLists to staged/src/Geom/src.

add_library(Geom STATIC ${GEOM_SRC_DIR}/GEOMUtils/GEOMUtils.cxx)
target_link_libraries(Geom PUBLIC
    Kernel
    TKBRep TKShHealing TKMesh TKGeomAlgo TKTopAlgo TKG2d TKG3d TKV3d TKGeomBase TKBO)
target_include_directories(Geom PUBLIC ${GEOM_SRC_DIR}/GEOMUtils)
