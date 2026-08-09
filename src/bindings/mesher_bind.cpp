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
      .def("release", &Mesher::release)
      .def("is_open", &Mesher::is_open);

  m.def("read_gmf", &mesher::read_gmf, py::arg("path"));
  m.def("write_gmf", &mesher::write_gmf, py::arg("path"), py::arg("mesh"), py::arg("groups"));
}

}  // namespace pysmesh
