# SALOME NETGENPlugin V9_16_0 static library for pySMESH: the NETGEN algorithms and their
# hypotheses, without the CORBA servants (*_i.cxx), the GUI, the remote and stand-alone
# variants (*_Remote, *_SA, Runner_main) and DriverParam.
#
# ${NETGENPLUGIN_SRC_DIR} is the staged, patched tree (prepare.py: staged/src/NETGENPlugin).
# The plugin includes netgen's own headers (occgeom.hpp, meshing.hpp, nglib.h), so it
# compiles with exactly PYSMESH_NETGEN_DEFINITIONS (cmake/Netgen/CMakeLists.txt), the list
# netgen itself is compiled with.
set(NETGENPLUGIN_SRC_DIR ${STAGED_DIR}/src/NETGENPlugin/src/NETGENPlugin)

set(NETGENPLUGIN_SRCS
    NETGENPlugin_Mesher.cxx
    NETGENPlugin_NETGEN_3D.cxx
    NETGENPlugin_NETGEN_2D.cxx
    NETGENPlugin_NETGEN_2D3D.cxx
    NETGENPlugin_NETGEN_2D_ONLY.cxx
    NETGENPlugin_Hypothesis.cxx
    NETGENPlugin_Hypothesis_2D.cxx
    NETGENPlugin_SimpleHypothesis_2D.cxx
    NETGENPlugin_SimpleHypothesis_3D.cxx
    NETGENPlugin_Remesher_2D.cxx)
list(TRANSFORM NETGENPLUGIN_SRCS PREPEND ${NETGENPLUGIN_SRC_DIR}/)

add_library(NETGENPlugin STATIC ${NETGENPLUGIN_SRCS})
target_compile_definitions(NETGENPlugin PRIVATE ${PYSMESH_NETGEN_DEFINITIONS})
target_compile_options(NETGENPlugin PRIVATE ${PYSMESH_NETGEN_OPTIONS})
target_include_directories(NETGENPlugin PRIVATE ${PYSMESH_NETGEN_INCLUDE_DIRS})
target_include_directories(NETGENPlugin PUBLIC ${NETGENPLUGIN_SRC_DIR})
target_link_libraries(NETGENPlugin PUBLIC
    ${SMESH_STATIC_LIBS} Geom Kernel nglib netgen ngcore
    TKernel TKMath TKG2d TKG3d TKGeomBase TKBRep TKGeomAlgo TKTopAlgo TKMesh TKShHealing)

set(NETGENPLUGIN_STATIC_LIBS NETGENPlugin)
