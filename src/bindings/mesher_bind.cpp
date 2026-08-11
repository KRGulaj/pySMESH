// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — Mesher: the pybind11 surface.
//
// Positional, un-defaulted and untyped by design: src/pysmesh/mesher/ is the public API and
// it owns the keyword names, the defaults and the frozen dataclasses. Keeping this layer
// dumb means there is one place a signature can drift, not two.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

namespace pysmesh {

using mesher::Mesher;

void bind_mesher(py::module_& m) {
  py::class_<Mesher>(m, "Mesher")
      .def(py::init<const py::object&>(), py::arg("shape"))
      .def("assign", &Mesher::assign, py::arg("name"), py::arg("params"), py::arg("kind"),
           py::arg("ordinal"))
      .def("unassign", &Mesher::unassign, py::arg("name"), py::arg("kind"), py::arg("ordinal"))
      .def("assignments", &Mesher::assignments)
      .def("compute", &Mesher::compute, py::arg("progress"), py::arg("cancel"))
      .def("mesh_arrays", &Mesher::mesh_arrays)
      .def("groups", &Mesher::groups)
      .def("quality", &Mesher::quality, py::arg("name"), py::arg("params"))
      .def("select", &Mesher::select, py::arg("name"), py::arg("params"))
      .def("add_group", &Mesher::add_group, py::arg("name"), py::arg("family"), py::arg("ids"))
      .def("add_group_on_shape", &Mesher::add_group_on_shape, py::arg("name"),
           py::arg("family"), py::arg("kind"), py::arg("ordinal"))
      .def("add_group_on_filter", &Mesher::add_group_on_filter, py::arg("name"),
           py::arg("family"), py::arg("predicate"), py::arg("params"))
      .def("remove_group", &Mesher::remove_group, py::arg("name"))
      .def("edit_group", &Mesher::edit_group, py::arg("name"), py::arg("ids"), py::arg("add"))
      .def("convert_to_quadratic", &Mesher::convert_to_quadratic, py::arg("force_3d"),
           py::arg("bi_quadratic"))
      .def("convert_from_quadratic", &Mesher::convert_from_quadratic)
      .def("split_volumes", &Mesher::split_volumes, py::arg("method"), py::arg("nx"),
           py::arg("ny"), py::arg("nz"))
      .def("split_quadratic_into_linear", &Mesher::split_quadratic_into_linear,
           py::arg("elements"))
      .def("merge_nodes", &Mesher::merge_nodes, py::arg("tolerance"))
      .def("find_coincident_nodes", &Mesher::find_coincident_nodes, py::arg("tolerance"),
           py::arg("nodes"), py::arg("separate_corners_and_medium"))
      .def("merge_node_groups", &Mesher::merge_node_groups, py::arg("groups"),
           py::arg("avoid_making_holes"))
      .def("find_equal_elements", &Mesher::find_equal_elements, py::arg("elements"))
      .def("merge_equal_elements", &Mesher::merge_equal_elements)
      .def("smooth", &Mesher::smooth, py::arg("method"), py::arg("iterations"),
           py::arg("target_aspect_ratio"), py::arg("in_uv_space"), py::arg("elements"),
           py::arg("fixed_nodes"))
      .def("reorient", &Mesher::reorient, py::arg("elements"))
      .def("reorient_2d", &Mesher::reorient_2d, py::arg("direction"), py::arg("faces"),
           py::arg("reference_faces"), py::arg("allow_non_manifold"))
      .def("reorient_2d_by_3d", &Mesher::reorient_2d_by_3d, py::arg("faces"),
           py::arg("volumes"), py::arg("outside_normal"))
      .def("quad_to_tri", &Mesher::quad_to_tri, py::arg("elements"), py::arg("criterion"),
           py::arg("criterion_params"), py::arg("diagonal_13"))
      .def("tri_to_quad", &Mesher::tri_to_quad, py::arg("elements"), py::arg("criterion"),
           py::arg("criterion_params"), py::arg("max_angle"))
      .def("double_elements", &Mesher::double_elements, py::arg("elements"))
      .def("extrusion_sweep", &Mesher::extrusion_sweep, py::arg("elements"), py::arg("step"),
           py::arg("steps"), py::arg("make_boundary"), py::arg("tolerance"))
      .def("rotation_sweep", &Mesher::rotation_sweep, py::arg("elements"), py::arg("origin"),
           py::arg("direction"), py::arg("angle"), py::arg("steps"), py::arg("tolerance"),
           py::arg("make_walls"))
      .def("offset", &Mesher::offset, py::arg("value"), py::arg("elements"),
           py::arg("copy_elements"), py::arg("fix_self_intersection"))
      .def("sew_free_border", &Mesher::sew_free_border, py::arg("border"), py::arg("side"),
           py::arg("side_is_free_border"), py::arg("create_polygons"),
           py::arg("create_polyhedra"))
      .def("sew_side_elements", &Mesher::sew_side_elements, py::arg("side1"), py::arg("side2"),
           py::arg("first_nodes"), py::arg("second_nodes"))
      .def("find_elements_by_point", &Mesher::find_elements_by_point, py::arg("points"),
           py::arg("family"))
      .def("find_closest", &Mesher::find_closest, py::arg("points"), py::arg("family"))
      .def("closest_distance", &Mesher::closest_distance, py::arg("points"), py::arg("family"))
      .def("project_points", &Mesher::project_points, py::arg("points"), py::arg("family"))
      .def("point_state", &Mesher::point_state, py::arg("points"))
      .def("elements_in_sphere", &Mesher::elements_in_sphere, py::arg("centre"),
           py::arg("radius"), py::arg("family"))
      .def("elements_in_box", &Mesher::elements_in_box, py::arg("minimum"), py::arg("maximum"),
           py::arg("family"))
      .def("elements_near_line", &Mesher::elements_near_line, py::arg("origin"),
           py::arg("direction"), py::arg("family"))
      .def("ray_hits", &Mesher::ray_hits, py::arg("origin"), py::arg("direction"),
           py::arg("tolerance"))
      .def("sharp_edges", &Mesher::sharp_edges, py::arg("angle"), py::arg("add_existing"))
      .def("separate_faces_by_edges", &Mesher::separate_faces_by_edges, py::arg("node1"),
           py::arg("node2"), py::arg("medium"))
      .def("de_merge", &Mesher::de_merge, py::arg("element"), py::arg("groups"))
      .def("make_slot", &Mesher::make_slot, py::arg("width"), py::arg("segments"))
      .def("pattern_from_face", &Mesher::pattern_from_face, py::arg("face"),
           py::arg("project"))
      .def("apply_pattern_to_face", &Mesher::apply_pattern_to_face, py::arg("text"),
           py::arg("face"), py::arg("vertex"), py::arg("reverse"), py::arg("create_polygons"))
      .def("apply_pattern_to_block", &Mesher::apply_pattern_to_block, py::arg("text"),
           py::arg("solid"), py::arg("vertex000"), py::arg("vertex001"),
           py::arg("create_polyhedra"))
      .def("release", &Mesher::release)
      .def("is_open", &Mesher::is_open);

  m.def("medial_axis", &mesher::medial_axis, py::arg("shape"), py::arg("face"),
        py::arg("min_segment_length"), py::arg("ignore_corners"), py::arg("samples"));
  m.def("block_shapes", &mesher::block_shapes, py::arg("shape"), py::arg("solid"),
        py::arg("vertex000"), py::arg("vertex001"));
  m.def("block_points", &mesher::block_points, py::arg("shape"), py::arg("solid"),
        py::arg("vertex000"), py::arg("vertex001"), py::arg("parameters"));
  m.def("block_parameters", &mesher::block_parameters, py::arg("shape"), py::arg("solid"),
        py::arg("vertex000"), py::arg("vertex001"), py::arg("points"), py::arg("tolerance"));

  m.def("read_gmf", &mesher::read_gmf, py::arg("path"));
  m.def("write_gmf", &mesher::write_gmf, py::arg("path"), py::arg("mesh"), py::arg("groups"));
  m.def("mesh_quality", &mesher::mesh_quality, py::arg("mesh"), py::arg("name"),
        py::arg("params"));
  m.def("mesh_select", &mesher::mesh_select, py::arg("mesh"), py::arg("name"),
        py::arg("params"));
}

}  // namespace pysmesh
