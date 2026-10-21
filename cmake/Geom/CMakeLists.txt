# SALOME GEOM slice (static): GEOMUtils.cxx and the ShHealOper package.
#
# GEOMUtils.cxx matches looooo/SMESH/cmake/Geom/CMakeLists.txt. At V9_16_0 four SMESH files
# include <GEOMUtils.hxx> (SMESH_Mesh.cxx, Controls/SMESH_Controls.cxx,
# Controls/SMESH_ControlsClassifier.cxx, StdMeshers_Cartesian_3D_Grid.hxx), so this is a
# hard dependency of SMESH core. The three GEOMAlgo_AlgoTools functions that GEOMUtils.cxx
# calls since V9_16_0 are compiled into it by patches/geom/GEOMUtils_GEOMAlgo.patch; they use
# TKMesh (BRepMesh_IncrementalMesh), TKV3d (Prs3d) and TKTopAlgo
# (BRepLib_CheckCurveOnSurface), already listed below. ${GEOM_SRC_DIR} is set by the root
# CMakeLists to staged/src/Geom/src.
#
# ShHealOper is GEOM's shape-healing package. The source list is that of upstream
# src/ShHealOper/CMakeLists.txt, unchanged; prepare.py applies the OCCT 8.0.1 port in
# patches/geom/ShHealOper_*.patch. It adds two link dependencies:
#   CommonGeomLib  ShHealOper_ShapeProcess calls GEOMAlgo_AlgoTools::FixCurveOnSurface-
#                  Tolerances, as upstream links CommonGeomLib's GEOMAlgo;
#   TKCAF          ShHealOper_ShapeProcess::Perform copies its input with
#                  TNaming_CopyShape::CopyTool, a static function of TKCAF's TNaming package.
#                  It reads no OCAF document. _core links TKCAF already (STEP names).
# The other toolkits it calls are the ones below: ShapeProcess, ShapeProcessAPI and the
# Shape* packages are TKShHealing; GeomPlate and GeomAPI are TKGeomAlgo; GCPnts is
# TKGeomBase.

set(ShHealOper_SRCS
    ${GEOM_SRC_DIR}/ShHealOper/ShHealOper_CloseContour.cxx
    ${GEOM_SRC_DIR}/ShHealOper/ShHealOper_EdgeDivide.cxx
    ${GEOM_SRC_DIR}/ShHealOper/ShHealOper_FillHoles.cxx
    ${GEOM_SRC_DIR}/ShHealOper/ShHealOper_RemoveFace.cxx
    ${GEOM_SRC_DIR}/ShHealOper/ShHealOper_RemoveInternalWires.cxx
    ${GEOM_SRC_DIR}/ShHealOper/ShHealOper_Sewing.cxx
    ${GEOM_SRC_DIR}/ShHealOper/ShHealOper_ShapeProcess.cxx
    ${GEOM_SRC_DIR}/ShHealOper/ShHealOper_SplitCurve2d.cxx
    ${GEOM_SRC_DIR}/ShHealOper/ShHealOper_SplitCurve3d.cxx
    ${GEOM_SRC_DIR}/ShHealOper/ShHealOper_ChangeOrientation.cxx
    ${GEOM_SRC_DIR}/ShHealOper/ShHealOper_Tool.cxx)

add_library(Geom STATIC ${GEOM_SRC_DIR}/GEOMUtils/GEOMUtils.cxx ${ShHealOper_SRCS})
target_link_libraries(Geom PUBLIC
    Kernel CommonGeomLib
    TKBRep TKShHealing TKMesh TKGeomAlgo TKTopAlgo TKG2d TKG3d TKV3d TKGeomBase TKBO TKCAF)
target_include_directories(Geom PUBLIC
    ${GEOM_SRC_DIR}/GEOMUtils
    ${GEOM_SRC_DIR}/ShHealOper)
