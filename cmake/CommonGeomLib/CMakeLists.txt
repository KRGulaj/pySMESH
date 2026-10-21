# SALOME common_geometry_lib V9_16_0 (static): the GEOMAlgo and BlockFix packages, one
# library. Upstream builds them as two shared libraries (src/GEOMAlgo/CMakeLists.txt and
# src/BlockFix/CMakeLists.txt); the source lists below are theirs, unchanged. GeomAnaTool
# is not vendored. prepare.py stages both directories at staged/src/CommonGeomLib/src and
# applies the OCCT 8.0.1 port in patches/common_geometry_lib/.
#
# The OCCT toolkits are those of the OCCT packages whose headers the sources include, each
# package mapped to its toolkit by OCCT's own src/*/TK*/PACKAGES.cmake:
#   TKernel   Standard, NCollection, TColStd, Precision
#   TKMath    gp, Bnd, ElCLib, ElSLib, GeomAbs, Poly, TopLoc
#   TKG2d     Geom2d, Geom2dAdaptor
#   TKG3d     Geom, GeomAdaptor, TopAbs
#   TKGeomBase  GC, GProp, GeomLib, GeomConvert, Geom2dConvert
#   TKGeomAlgo  GeomAPI, Geom2dHatch, HatchGen, IntSurf, IntPatch, LocalAnalysis
#   TKBRep    BRep, BRepAdaptor, BRepTools, TopExp, TopTools, TopoDS
#   TKTopAlgo BRepBndLib, BRepBuilderAPI, BRepCheck, BRepClass, BRepClass3d, BRepGProp,
#             BRepLib, BRepTopAdaptor
#   TKBO      BOPAlgo, BOPDS, BOPTools, IntTools
#   TKShHealing ShapeAnalysis, ShapeBuild, ShapeCustom, ShapeExtend, ShapeFix, ShapeUpgrade
#   TKOffset  BRepOffsetAPI (BlockFix's BRepOffsetAPI_MakeFilling)
#   TKMesh    BRepMesh
#   TKV3d     Prs3d (GEOMAlgo_AlgoTools::MeshShape)
# Every one is already in _core's closure. Kernel brings SALOMELocalTrace, whose
# utilities.h the trace macros of both packages use.

set(COMMONGEOMLIB_SRC_DIR ${STAGED_DIR}/src/CommonGeomLib/src)

set(CommonGeomLib_SRCS
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_Algo.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_AlgoTools.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_AlgoTools_1.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_BndSphere.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_BndSphereTree.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_BoxBndTree.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_BuilderShape.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_Clsf.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_ClsfBox.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_ClsfQuad.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_ClsfSolid.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_ClsfSurf.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_CoupleOfShapes.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_FinderShapeOn2.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_Extractor.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_GetInPlace.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_GetInPlace_1.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_GetInPlace_2.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_GetInPlace_3.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_GlueAnalyser.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_GlueDetector.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_Gluer.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_Gluer2.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_Gluer2_1.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_Gluer2_2.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_Gluer2_3.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_GluerAlgo.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_HAlgo.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_PassKey.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_PassKeyMapHasher.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_PassKeyShape.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_PassKeyShapeMapHasher.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_RemoverWebs.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_ShapeAlgo.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_ShapeInfo.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_ShapeInfoFiller.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_ShapeInfoFiller_1.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_ShapeSolid.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_ShellSolid.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_SolidSolid.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_Splitter.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_StateCollector.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_SurfaceTools.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_VertexSolid.cxx
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo/GEOMAlgo_WireSolid.cxx
    ${COMMONGEOMLIB_SRC_DIR}/BlockFix/BlockFix_BlockFixAPI.cxx
    ${COMMONGEOMLIB_SRC_DIR}/BlockFix/BlockFix_CheckTool.cxx
    ${COMMONGEOMLIB_SRC_DIR}/BlockFix/BlockFix.cxx
    ${COMMONGEOMLIB_SRC_DIR}/BlockFix/BlockFix_PeriodicSurfaceModifier.cxx
    ${COMMONGEOMLIB_SRC_DIR}/BlockFix/BlockFix_SphereSpaceModifier.cxx
    ${COMMONGEOMLIB_SRC_DIR}/BlockFix/BlockFix_UnionEdges.cxx
    ${COMMONGEOMLIB_SRC_DIR}/BlockFix/BlockFix_UnionFaces.cxx)

add_library(CommonGeomLib STATIC ${CommonGeomLib_SRCS})
target_include_directories(CommonGeomLib PUBLIC
    ${COMMONGEOMLIB_SRC_DIR}/GEOMAlgo
    ${COMMONGEOMLIB_SRC_DIR}/BlockFix)
target_link_libraries(CommonGeomLib PUBLIC
    Kernel
    TKernel TKMath TKG2d TKG3d TKGeomBase TKGeomAlgo TKBRep TKTopAlgo TKBO TKShHealing
    TKOffset TKMesh TKV3d)
