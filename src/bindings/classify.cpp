// pySMESH binding — point/solid classification: point_in_solid.
//
// Wraps OCCT's BRepClass3d_SolidClassifier (TKTopAlgo): an exact ray-casting point-in-solid
// test against the analytic B-rep (not a tessellation). Unblocks the report's §5.1 internal
// flow-volume extraction — after Gmsh caps/sews the surfaces and makeSolids yields candidate
// solids, the seed point picks the enclosing solid by an exact inside-test done here.
//
// OCCT-only (no SMESH); shares just the BREP bytes bridge, the NumPy (N,3) helper, and
// PysmeshError with the rest of pysmesh. The classifier is loaded once with the solid and
// then Perform()ed per point (the load builds the internal spatial structure, so per-point
// cost is a single classification, not a rebuild).

#include <cstdint>
#include <sstream>
#include <string>

#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopAbs_State.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Pnt.hxx>

#include "common.hpp"

namespace pysmesh {
namespace {

// Read a BREP shape from in-memory bytes (mirrors shape.cpp::load_brep / unify.cpp /
// distance.cpp, which are file-local there). Raises on parse failure or a null result.
TopoDS_Shape read_brep(const py::bytes& data) {
  const std::string buffer = data;
  std::istringstream stream(buffer);
  TopoDS_Shape shape;
  BRep_Builder builder;
  try {
    BRepTools::Read(shape, stream, builder);
  } catch (const std::exception& e) {
    throw PysmeshError(std::string("BREP read failed: ") + e.what());
  }
  if (shape.IsNull()) {
    throw PysmeshError("BREP read produced a null shape (empty or malformed data)");
  }
  return shape;
}

py::array_t<bool> point_in_solid(const py::bytes& brep, const py::object& points_obj,
                                 double tol) {
  if (!(tol > 0.0)) {
    throw PysmeshError("point_in_solid: tol must be > 0 (got " + std::to_string(tol) + ").");
  }
  const TopoDS_Shape shape = read_brep(brep);

  // BRepClass3d_SolidClassifier requires a solid: the inside-test is only defined against a
  // closed volume. Fail loud on a bare shell/face/wire (e.g. pre-makeSolids input) rather
  // than returning meaningless states.
  TopTools_IndexedMapOfShape solids;
  TopExp::MapShapes(shape, TopAbs_SOLID, solids);
  if (solids.Extent() < 1) {
    throw PysmeshError(
        "point_in_solid: the shape contains no TopAbs_SOLID (build a solid via sew + "
        "makeSolids before classifying). Open shells/faces have no defined interior.");
  }

  Array2d points = as_2d_f64(points_obj, "points", 3);
  const py::ssize_t n = points.shape(0);
  py::array_t<bool> out(n);

  const double* pts = points.data();
  bool* mask = out.mutable_data();
  {
    py::gil_scoped_release release;
    BRepClass3d_SolidClassifier classifier(shape);  // load once (builds the spatial data)
    for (py::ssize_t i = 0; i < n; ++i) {
      const gp_Pnt p(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
      classifier.Perform(p, tol);
      // Strictly inside only: ON (within tol of the boundary) and OUT are False, which is the
      // right contract for seed selection (the seed must be in the interior of the volume).
      mask[i] = classifier.State() == TopAbs_IN;
    }
  }
  return out;
}

}  // namespace

void bind_classify(py::module_& m) {
  m.def("point_in_solid", &point_in_solid, py::arg("brep"), py::arg("points"), py::arg("tol"),
        "Exact point-in-solid test (OCCT BRepClass3d_SolidClassifier). Returns a (N,) bool "
        "mask, True iff the point is strictly inside the solid (TopAbs_IN); points within "
        "'tol' of the boundary (ON) and outside points are False. Raises if the shape has no "
        "solid. Low-level: the pysmesh.point_in_solid wrapper validates tol and forwards.");
}

}  // namespace pysmesh
