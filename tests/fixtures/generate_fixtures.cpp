// Deterministic BREP fixture generator for pySMESH tests.
//
// Not built by default and never run by CI — the committed *.brep outputs ARE the
// fixtures. Rebuild them with:
//
//   cmake -S . -B build -DPYSMESH_BUILD_FIXTURE_GEN=ON  ...(usual flags)
//   cmake --build build --target generate_fixtures
//   ./build/generate_fixtures tests/fixtures
//
// Emits:
//   box.brep       axis-aligned cube, edge length BOX_EDGE, min corner at the origin.
//                  6 faces, total surface area 6 * BOX_EDGE^2 (asserted in test_shape.py).
//   cylinder.brep  radius CYL_RADIUS, height CYL_HEIGHT, axis +Z from the origin.

#include <cstdio>
#include <string>

#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepTools.hxx>
#include <TopoDS_Shape.hxx>

namespace {
constexpr double BOX_EDGE = 2.0;
constexpr double CYL_RADIUS = 1.0;
constexpr double CYL_HEIGHT = 3.0;
}  // namespace

int main(int argc, char** argv) {
  const std::string out_dir = (argc > 1) ? argv[1] : ".";

  const TopoDS_Shape box = BRepPrimAPI_MakeBox(BOX_EDGE, BOX_EDGE, BOX_EDGE).Shape();
  if (!BRepTools::Write(box, (out_dir + "/box.brep").c_str())) {
    std::fprintf(stderr, "failed to write box.brep\n");
    return 1;
  }

  const TopoDS_Shape cyl = BRepPrimAPI_MakeCylinder(CYL_RADIUS, CYL_HEIGHT).Shape();
  if (!BRepTools::Write(cyl, (out_dir + "/cylinder.brep").c_str())) {
    std::fprintf(stderr, "failed to write cylinder.brep\n");
    return 1;
  }

  std::printf("wrote box.brep (edge=%.3f) and cylinder.brep (r=%.3f h=%.3f) to %s\n",
              BOX_EDGE, CYL_RADIUS, CYL_HEIGHT, out_dir.c_str());
  return 0;
}
