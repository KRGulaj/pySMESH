// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-06

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
      .def("add_brep", &Session::add_brep, py::arg("data"), py::arg("inside_out"),
           py::arg("progress"),
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
      .def("extract_edges", &Session::extract_edges, py::arg("edge_ids"))
      .def("make_face", &Session::make_face, py::arg("edge_ids"))
      .def("make_filling", &Session::make_filling, py::arg("edge_ids"),
           py::arg("support_faces"), py::arg("progress"), py::arg("cancel"))
      .def("extend_face", &Session::extend_face, py::arg("face_id"), py::arg("u_min"),
           py::arg("u_max"), py::arg("v_min"), py::arg("v_max"))
      .def("extend_edge", &Session::extend_edge, py::arg("edge_id"), py::arg("length"),
           py::arg("at_end"))
      .def("extrude", &Session::extrude, py::arg("entity_ids"), py::arg("vx"), py::arg("vy"),
           py::arg("vz"), py::arg("draft_angle"), py::arg("symmetric"))
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
      .def("split_by_faces", &Session::split_by_faces, py::arg("targets"), py::arg("tools"),
           py::arg("fuzzy"), py::arg("progress"), py::arg("cancel"))
      .def("split_by_plane", &Session::split_by_plane, py::arg("targets"), py::arg("ox"),
           py::arg("oy"), py::arg("oz"), py::arg("nx"), py::arg("ny"), py::arg("nz"),
           py::arg("fuzzy"), py::arg("progress"), py::arg("cancel"))
      .def("keep_half_space", &Session::keep_half_space, py::arg("targets"), py::arg("ox"),
           py::arg("oy"), py::arg("oz"), py::arg("nx"), py::arg("ny"), py::arg("nz"),
           py::arg("keep_positive"), py::arg("fuzzy"), py::arg("progress"),
           py::arg("cancel"))
      .def("fillet", &Session::fillet, py::arg("edge_ids"), py::arg("radius"),
           py::arg("radius_end"), py::arg("progress"),
           py::arg("cancel"))
      .def("chamfer", &Session::chamfer, py::arg("edge_ids"), py::arg("distance"),
           py::arg("distance_end"), py::arg("face_id"), py::arg("progress"),
           py::arg("cancel"))
      .def("fillet_2d", &Session::fillet_2d, py::arg("face_id"), py::arg("vertex_ids"),
           py::arg("radius"))
      .def("make_thick_solid", &Session::make_thick_solid, py::arg("face_ids"),
           py::arg("thickness"), py::arg("tol"), py::arg("progress"), py::arg("cancel"))
      .def("offset", &Session::offset, py::arg("entity_ids"), py::arg("distance"),
           py::arg("tol"), py::arg("progress"), py::arg("cancel"))
      .def("offset_wire", &Session::offset_wire, py::arg("edge_ids"), py::arg("distance"),
           py::arg("intersection"))
      .def("translate", &Session::translate, py::arg("dx"), py::arg("dy"), py::arg("dz"),
           py::arg("entity_ids"))
      .def("rotate", &Session::rotate, py::arg("ox"), py::arg("oy"), py::arg("oz"),
           py::arg("ax"), py::arg("ay"), py::arg("az"), py::arg("angle_rad"),
           py::arg("entity_ids"))
      .def("mirror", &Session::mirror, py::arg("px"), py::arg("py"), py::arg("pz"),
           py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("entity_ids"))
      .def("scale", &Session::scale, py::arg("sx"), py::arg("sy"), py::arg("sz"),
           py::arg("cx"), py::arg("cy"), py::arg("cz"), py::arg("entity_ids"))
      .def("position_along_path", &Session::position_along_path, py::arg("path_edge"),
           py::arg("distance"), py::arg("entity_ids"), py::arg("reverse"))
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
      .def("limit_tolerance", &Session::limit_tolerance, py::arg("entity_ids"),
           py::arg("tolerance"))
      .def("change_orientation", &Session::change_orientation, py::arg("face_ids"))
      .def("unify_same_domain", &Session::unify_same_domain, py::arg("entity_ids"),
           py::arg("unify_faces"), py::arg("unify_edges"), py::arg("concat_bsplines"),
           py::arg("linear_tol"), py::arg("angular_tol_rad"))
      .def("glue", &Session::glue, py::arg("entity_ids"), py::arg("tolerance"),
           py::arg("kind"), py::arg("keep_non_solids"), py::arg("progress"),
           py::arg("cancel"))
      .def("divide_edge", &Session::divide_edge, py::arg("edge_id"), py::arg("positions"),
           py::arg("by"))
      .def("close_contour", &Session::close_contour, py::arg("edge_ids"),
           py::arg("common_vertex"))
      .def("fill_holes", &Session::fill_holes, py::arg("entity_ids"), py::arg("edge_ids"))
      .def("suppress_faces", &Session::suppress_faces, py::arg("face_ids"))
      .def("shape_process", &Session::shape_process, py::arg("entity_ids"),
           py::arg("operators"), py::arg("parameters"), py::arg("progress"),
           py::arg("cancel"))
      .def("fillet_1d", &Session::fillet_1d, py::arg("edge_ids"), py::arg("vertex_ids"),
           py::arg("radius"), py::arg("ignore_secant_vertices"))
      .def("smoothing_surface", &Session::smoothing_surface, py::arg("points"),
           py::arg("max_points"), py::arg("max_degree"), py::arg("max_distance"))
      .def("make_quad", &Session::make_quad, py::arg("edge_ids"))
      .def("make_quad_from_vertices", &Session::make_quad_from_vertices,
           py::arg("vertex_ids"))
      .def("make_hexa", &Session::make_hexa, py::arg("face_ids"))
      .def("check_blocks", &Session::check_blocks, py::arg("entity_ids"),
           py::arg("angular_tolerance"), py::arg("progress"), py::arg("cancel"))
      .def("propagation_chain", &Session::propagation_chain, py::arg("edge_id"))
      .def("opposite_face", &Session::opposite_face, py::arg("block_id"), py::arg("face_id"))
      .def("face_by_edges", &Session::face_by_edges, py::arg("edge_a"), py::arg("edge_b"))
      .def("find_block_near", &Session::find_block_near, py::arg("px"), py::arg("py"),
           py::arg("pz"))
      .def("multi_transform", &Session::multi_transform, py::arg("block_id"),
           py::arg("vectors"), py::arg("counts"))
      .def("entity_types", &Session::entity_types, py::arg("kind"))
      .def("surface_parameters", &Session::surface_parameters, py::arg("face_ids"))
      .def("bounding_boxes", &Session::bounding_boxes, py::arg("kind"))
      .def("mass_properties", &Session::mass_properties, py::arg("entity_ids"),
           py::arg("precision"))
      .def("face_parameter_bounds", &Session::face_parameter_bounds, py::arg("face_ids"))
      .def("edge_parameter_bounds", &Session::edge_parameter_bounds, py::arg("edge_ids"))
      .def("adjacency", &Session::adjacency, py::arg("kind"), py::arg("other_kind"))
      .def("face_wires", &Session::face_wires, py::arg("face_ids"))
      .def("surface_at", &Session::surface_at, py::arg("face_id"), py::arg("uv"))
      .def("distance", &Session::distance, py::arg("entity_id_a"), py::arg("entity_id_b"))
      .def("curve_at", &Session::curve_at, py::arg("edge_id"), py::arg("t"))
      .def("curve_geometry", &Session::curve_geometry, py::arg("edge_ids"))
      .def("curvature", &Session::curvature, py::arg("face_ids"), py::arg("samples"))
      .def("project_on_face", &Session::project_on_face, py::arg("face_id"),
           py::arg("points"))
      .def("face_frame", &Session::face_frame, py::arg("face_id"))
      .def("tangent_frame", &Session::tangent_frame, py::arg("face_id"), py::arg("u"),
           py::arg("v"))
      .def("face_face_axis", &Session::face_face_axis, py::arg("face_a"), py::arg("face_b"))
      .def("point_at_length", &Session::point_at_length, py::arg("edge_id"),
           py::arg("length"), py::arg("from_start"))
      .def("entities_on", &Session::entities_on, py::arg("kind"), py::arg("region"),
           py::arg("values"), py::arg("solid_id"), py::arg("state"), py::arg("tolerance"),
           py::arg("progress"), py::arg("cancel"))
      .def("in_place_brep", &Session::in_place_brep, py::arg("kind"), py::arg("data"),
           py::arg("ordinals"), py::arg("progress"), py::arg("cancel"))
      .def("in_place_snapshot", &Session::in_place_snapshot, py::arg("kind"), py::arg("mark"),
           py::arg("ids"), py::arg("progress"), py::arg("cancel"))
      .def("find_same", &Session::find_same, py::arg("entity_id"), py::arg("within"))
      .def("filter_mask", &Session::filter_mask, py::arg("kind"), py::arg("filter"),
           py::arg("entity_ids"), py::arg("numbers"), py::arg("flag"), py::arg("progress"),
           py::arg("cancel"))
      .def("size_measures", &Session::size_measures, py::arg("kind"), py::arg("progress"),
           py::arg("cancel"))
      .def("check_boolean", &Session::check_boolean, py::arg("operation"),
           py::arg("arguments"), py::arg("tools"), py::arg("progress"), py::arg("cancel"))
      .def("check_shape", &Session::check_shape, py::arg("entity_ids"), py::arg("progress"),
           py::arg("cancel"))
      .def("small_edges", &Session::small_edges, py::arg("threshold"))
      .def("is_closed_shell", &Session::is_closed_shell, py::arg("entity_ids"))
      .def("tolerances", &Session::tolerances, py::arg("entity_ids"))
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
      .def("entities", &Session::entities, py::arg("kind"), py::arg("distinct") = false)
      .def("alias_groups", &Session::alias_groups, py::arg("kind"))
      .def("ordinal_ids", &Session::ordinal_ids, py::arg("kind"))
      .def("entity_kind", &Session::entity_kind, py::arg("entity_id"))
      .def("entity_state", &Session::entity_state, py::arg("entity_id"))
      .def("shape_count", &Session::shape_count, py::arg("entity_id"))
      .def("entity_table", &Session::entity_table, py::arg("kind"))
      .def("brep", &Session::brep)
      .def("export_handoff", &Session::export_handoff, py::arg("allow_aliases") = false)
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
