// pySMESH binding — Session: the pybind11 surface.
//
// Positional, un-defaulted and untyped by design: src/pysmesh/session.py is the public API,
// and it owns the keyword names, the defaults and the frozen dataclasses. Keeping this layer
// dumb means there is one place a signature can drift, not two.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

namespace pysmesh {

using session::Session;

void bind_session(py::module_& m) {
  py::class_<Session>(m, "Session")
      .def(py::init<bool>(), py::arg("validate"))
      .def("add_brep", &Session::add_brep, py::arg("data"), py::arg("progress"),
           py::arg("cancel"))
      .def("add_box", &Session::add_box, py::arg("dx"), py::arg("dy"), py::arg("dz"),
           py::arg("ox"), py::arg("oy"), py::arg("oz"))
      .def("add_cylinder", &Session::add_cylinder, py::arg("radius"), py::arg("height"),
           py::arg("ox"), py::arg("oy"), py::arg("oz"), py::arg("ax"), py::arg("ay"),
           py::arg("az"))
      .def("add_cone", &Session::add_cone, py::arg("radius1"), py::arg("radius2"),
           py::arg("height"), py::arg("ox"), py::arg("oy"), py::arg("oz"), py::arg("ax"),
           py::arg("ay"), py::arg("az"), py::arg("angle_rad"))
      .def("add_sphere", &Session::add_sphere, py::arg("radius"), py::arg("cx"),
           py::arg("cy"), py::arg("cz"), py::arg("ax"), py::arg("ay"), py::arg("az"),
           py::arg("angle_rad"))
      .def("add_torus", &Session::add_torus, py::arg("radius1"), py::arg("radius2"),
           py::arg("ox"), py::arg("oy"), py::arg("oz"), py::arg("ax"), py::arg("ay"),
           py::arg("az"), py::arg("angle_rad"))
      .def("add_wedge", &Session::add_wedge, py::arg("dx"), py::arg("dy"), py::arg("dz"),
           py::arg("ltx"), py::arg("ox"), py::arg("oy"), py::arg("oz"), py::arg("ax"),
           py::arg("ay"), py::arg("az"))
      .def("add_vertex", &Session::add_vertex, py::arg("x"), py::arg("y"), py::arg("z"))
      .def("add_line", &Session::add_line, py::arg("x1"), py::arg("y1"), py::arg("z1"),
           py::arg("x2"), py::arg("y2"), py::arg("z2"))
      .def("add_arc", &Session::add_arc, py::arg("x1"), py::arg("y1"), py::arg("z1"),
           py::arg("x2"), py::arg("y2"), py::arg("z2"), py::arg("x3"), py::arg("y3"),
           py::arg("z3"))
      .def("add_circle", &Session::add_circle, py::arg("cx"), py::arg("cy"), py::arg("cz"),
           py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("radius"))
      .def("add_ellipse", &Session::add_ellipse, py::arg("cx"), py::arg("cy"), py::arg("cz"),
           py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("rx"), py::arg("ry"),
           py::arg("x_dir"))
      .def("add_polyline", &Session::add_polyline, py::arg("points"), py::arg("closed"))
      .def("add_spline", &Session::add_spline, py::arg("points"), py::arg("degree_min"),
           py::arg("degree_max"), py::arg("tol"))
      .def("add_bspline", &Session::add_bspline, py::arg("poles"), py::arg("degree"))
      .def("add_helix", &Session::add_helix, py::arg("cx"), py::arg("cy"), py::arg("cz"),
           py::arg("ax"), py::arg("ay"), py::arg("az"), py::arg("diameter"),
           py::arg("pitch"), py::arg("turns"), py::arg("tol"))
      .def("add_rectangle", &Session::add_rectangle, py::arg("ox"), py::arg("oy"),
           py::arg("oz"), py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("dx"),
           py::arg("dy"))
      .def("make_wire", &Session::make_wire, py::arg("edge_ids"))
      .def("make_face", &Session::make_face, py::arg("edge_ids"))
      .def("make_filling", &Session::make_filling, py::arg("edge_ids"), py::arg("progress"),
           py::arg("cancel"))
      .def("extrude", &Session::extrude, py::arg("entity_ids"), py::arg("vx"), py::arg("vy"),
           py::arg("vz"))
      .def("revolve", &Session::revolve, py::arg("entity_ids"), py::arg("ox"), py::arg("oy"),
           py::arg("oz"), py::arg("ax"), py::arg("ay"), py::arg("az"), py::arg("angle_rad"))
      .def("pipe", &Session::pipe, py::arg("spine_ids"), py::arg("profile_ids"),
           py::arg("progress"), py::arg("cancel"))
      .def("pipe_shell", &Session::pipe_shell, py::arg("spine_ids"), py::arg("profile_ids"),
           py::arg("frenet"), py::arg("solid"), py::arg("progress"), py::arg("cancel"))
      .def("thru_sections", &Session::thru_sections, py::arg("sections"), py::arg("solid"),
           py::arg("ruled"), py::arg("progress"),
           py::arg("cancel"))
      .def("fuse", &Session::fuse, py::arg("targets"), py::arg("tools"), py::arg("fuzzy"),
           py::arg("parallel"), py::arg("progress"),
           py::arg("cancel"))
      .def("cut", &Session::cut, py::arg("targets"), py::arg("tools"), py::arg("fuzzy"),
           py::arg("parallel"), py::arg("progress"),
           py::arg("cancel"))
      .def("common", &Session::common, py::arg("targets"), py::arg("tools"),
           py::arg("fuzzy"), py::arg("parallel"), py::arg("progress"),
           py::arg("cancel"))
      .def("section", &Session::section, py::arg("targets"), py::arg("tools"),
           py::arg("fuzzy"), py::arg("parallel"), py::arg("progress"),
           py::arg("cancel"))
      .def("split", &Session::split, py::arg("targets"), py::arg("tools"), py::arg("fuzzy"),
           py::arg("parallel"), py::arg("progress"),
           py::arg("cancel"))
      .def("fragment", &Session::fragment, py::arg("entity_ids"), py::arg("fuzzy"),
           py::arg("parallel"), py::arg("progress"),
           py::arg("cancel"))
      .def("fillet", &Session::fillet, py::arg("edge_ids"), py::arg("radius"),
           py::arg("radius_end"), py::arg("progress"),
           py::arg("cancel"))
      .def("chamfer", &Session::chamfer, py::arg("edge_ids"), py::arg("distance"),
           py::arg("distance_end"), py::arg("face_id"), py::arg("progress"),
           py::arg("cancel"))
      .def("translate", &Session::translate, py::arg("dx"), py::arg("dy"), py::arg("dz"),
           py::arg("entity_ids"))
      .def("rotate", &Session::rotate, py::arg("ox"), py::arg("oy"), py::arg("oz"),
           py::arg("ax"), py::arg("ay"), py::arg("az"), py::arg("angle_rad"),
           py::arg("entity_ids"))
      .def("mirror", &Session::mirror, py::arg("px"), py::arg("py"), py::arg("pz"),
           py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("entity_ids"))
      .def("scale", &Session::scale, py::arg("sx"), py::arg("sy"), py::arg("sz"),
           py::arg("cx"), py::arg("cy"), py::arg("cz"), py::arg("entity_ids"))
      .def("copy", &Session::copy, py::arg("entity_ids"))
      .def("heal", &Session::heal, py::arg("entity_ids"), py::arg("precision"),
           py::arg("min_tolerance"), py::arg("max_tolerance"), py::arg("progress"),
           py::arg("cancel"))
      .def("sew", &Session::sew, py::arg("entity_ids"), py::arg("tolerance"),
           py::arg("make_solid"), py::arg("non_manifold"), py::arg("progress"),
           py::arg("cancel"))
      .def("remove_internal_wires", &Session::remove_internal_wires, py::arg("entity_ids"),
           py::arg("min_area"), py::arg("remove_faces"))
      .def("defeature", &Session::defeature, py::arg("face_ids"), py::arg("parallel"),
           py::arg("progress"), py::arg("cancel"))
      .def("imprint", &Session::imprint, py::arg("targets"), py::arg("tools"),
           py::arg("fuzzy"), py::arg("parallel"), py::arg("glue"), py::arg("progress"),
           py::arg("cancel"))
      .def("remove", &Session::remove, py::arg("entity_ids"))
      .def("unify_same_domain", &Session::unify_same_domain, py::arg("entity_ids"),
           py::arg("unify_faces"), py::arg("unify_edges"), py::arg("concat_bsplines"),
           py::arg("linear_tol"), py::arg("angular_tol_rad"))
      .def("entity_types", &Session::entity_types, py::arg("kind"))
      .def("bounding_boxes", &Session::bounding_boxes, py::arg("kind"))
      .def("mass_properties", &Session::mass_properties, py::arg("entity_ids"))
      .def("face_parameter_bounds", &Session::face_parameter_bounds, py::arg("face_ids"))
      .def("edge_parameter_bounds", &Session::edge_parameter_bounds, py::arg("edge_ids"))
      .def("adjacency", &Session::adjacency, py::arg("kind"), py::arg("other_kind"))
      .def("surface_at", &Session::surface_at, py::arg("face_id"), py::arg("uv"))
      .def("curvature", &Session::curvature, py::arg("face_ids"), py::arg("samples"))
      .def("project_on_face", &Session::project_on_face, py::arg("face_id"),
           py::arg("points"))
      .def("entities_in_box", &Session::entities_in_box, py::arg("kind"), py::arg("xmin"),
           py::arg("ymin"), py::arg("zmin"), py::arg("xmax"), py::arg("ymax"),
           py::arg("zmax"), py::arg("strict"))
      .def("contains", &Session::contains, py::arg("solid_ids"), py::arg("points"),
           py::arg("tol"))
      .def("tessellate", &Session::tessellate, py::arg("deflection"), py::arg("angle_rad"),
           py::arg("relative"), py::arg("parallel"), py::arg("incremental"),
           py::arg("progress"), py::arg("cancel"))
      .def("snapshot", &Session::snapshot)
      .def("restore", &Session::restore, py::arg("mark"))
      .def("discard_snapshot", &Session::discard_snapshot, py::arg("mark"))
      .def("snapshot_count", &Session::snapshot_count)
      .def("entities", &Session::entities, py::arg("kind"))
      .def("entity_kind", &Session::entity_kind, py::arg("entity_id"))
      .def("entity_state", &Session::entity_state, py::arg("entity_id"))
      .def("shape_count", &Session::shape_count, py::arg("entity_id"))
      .def("entity_table", &Session::entity_table, py::arg("kind"))
      .def("brep", &Session::brep)
      .def("export_handoff", &Session::export_handoff)
      .def("name_of", &Session::name_of, py::arg("entity_id"))
      .def("origin", &Session::origin, py::arg("entity_id"))
      .def("resolve", &Session::resolve, py::arg("op_index"), py::arg("role"),
           py::arg("ordinal"))
      .def("op_count", &Session::op_count)
      .def("state_op_index", &Session::state_op_index)
      .def("issued_id_count", &Session::issued_id_count)
      .def("entity_count", &Session::entity_count)
      .def("_debug_tear_next_history", &Session::debug_tear_next_history,
           "Test hook: drop the NEXT operation's history so its input ids die instead of "
           "being carried forward. Exists so the identity suite can be shown to fail.");
}


}  // namespace pysmesh
