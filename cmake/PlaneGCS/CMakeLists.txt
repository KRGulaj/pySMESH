# PlaneGCS (static): the 2-D geometric constraint solver of FreeCAD 1.1.4
# (src/Mod/Sketcher/App/planegcs, LGPL-2.0-or-later), with Eigen 3.4.1 (header-only,
# extern/eigen, MPL-2.0).
#
# prepare.py stages the sources at staged/src/PlaneGCS/Mod/Sketcher/App/planegcs, FreeCAD's
# own depth, and stages the shim SketcherGlobal.h two levels up, where the sources'
# "../../SketcherGlobal.h" leads. The other FreeCAD-only headers are shims in this
# directory: FCConfig.h, Base/Console.h and boost_graph_adjacency_list.hpp. The patches in
# patches/planegcs/ run the solver's work on the calling thread.
#
# Three settings are this target's own:
#   * C++20. Constraints.cpp and GCS.cpp include <numbers> (std::numbers::pi). The rest of
#     pySMESH stays at C++17; the PlaneGCS headers need no C++20 feature.
#   * EIGEN_MPL2_ONLY, PUBLIC, so that every translation unit that includes a PlaneGCS
#     header compiles Eigen in MPL2-only mode: an include of LGPL Eigen code is a compile
#     error, and no GPL code can enter _core.pyd.
#   * extern/eigen first on the include path (BEFORE). The build environment has Eigen 5
#     under Library/include/eigen3, which no target puts on the include path; extern/eigen
#     comes first all the same, so <Eigen/...> resolves to Eigen 3.4.1 only.
# Boost.Graph and Boost.Math are header-only and come from the build environment's Boost,
# as every other Boost header does.

set(PLANEGCS_SRC_DIR ${STAGED_DIR}/src/PlaneGCS/Mod/Sketcher/App/planegcs)

set(PlaneGCS_SRCS
    ${PLANEGCS_SRC_DIR}/Constraints.cpp
    ${PLANEGCS_SRC_DIR}/GCS.cpp
    ${PLANEGCS_SRC_DIR}/Geo.cpp
    ${PLANEGCS_SRC_DIR}/SubSystem.cpp
    ${PLANEGCS_SRC_DIR}/qp_eq.cpp)

add_library(PlaneGCS STATIC ${PlaneGCS_SRCS})
set_target_properties(PlaneGCS PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_compile_definitions(PlaneGCS PUBLIC EIGEN_MPL2_ONLY)
target_include_directories(PlaneGCS BEFORE PUBLIC
    ${CMAKE_SOURCE_DIR}/extern/eigen
    ${PLANEGCS_SRC_DIR})
target_include_directories(PlaneGCS PRIVATE ${CMAKE_SOURCE_DIR}/cmake/PlaneGCS)
