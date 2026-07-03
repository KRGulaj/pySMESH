// pySMESH _core extension module: geometry query (Shape), surface-mesh injection (Mesh),
// viscous prism layers (compute_viscous_layers), and same-domain healing (unify_same_domain).
//
// Aggregates the per-file binders (bind_shape, bind_mesh, bind_viscous, bind_unify) and
// installs the typed exception (PysmeshError) with its .details / .face_ids attributes.

#include <exception>

#include "common.hpp"

namespace pysmesh {

void bind_shape(py::module_& m);
void bind_mesh(py::module_& m);
void bind_viscous(py::module_& m);
void bind_unify(py::module_& m);

namespace {

// Borrowed handle to the Python exception type. The owning reference lives on the module
// object (via m.add_object below), so this handle stays valid for the module's lifetime.
// A borrowed py::handle (not a py::object) avoids a Py_DECREF at interpreter shutdown,
// which pybind11 warns against for global storage.
py::handle g_error_type;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

}  // namespace

void register_error_type(py::module_& m) {
  py::object error_type = py::reinterpret_steal<py::object>(
      PyErr_NewException("pysmesh._core.PysmeshError", PyExc_RuntimeError, nullptr));
  m.add_object("PysmeshError", error_type);  // module now owns a reference
  g_error_type = error_type;                  // borrowed handle for the translator

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const PysmeshError& e) {
      py::object exc = g_error_type(py::str(e.what()));
      exc.attr("details") = py::str(e.details);
      exc.attr("face_ids") = py::cast(e.face_ids);
      PyErr_SetObject(g_error_type.ptr(), exc.ptr());
    }
  });
}

}  // namespace pysmesh

PYBIND11_MODULE(_core, m) {
  m.doc() = "pySMESH native core: SMESH ViscousLayers bindings (Tier-1).";
  pysmesh::register_error_type(m);
  pysmesh::bind_shape(m);
  pysmesh::bind_mesh(m);
  pysmesh::bind_viscous(m);
  pysmesh::bind_unify(m);
}
