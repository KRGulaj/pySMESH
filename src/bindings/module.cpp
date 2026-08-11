// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-07-03

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
void bind_tessellate(py::module_& m);
void bind_offset(py::module_& m);
void bind_distance(py::module_& m);
void bind_classify(py::module_& m);
void bind_step_xde(py::module_& m);
void bind_session(py::module_& m);

namespace {

// Borrowed handle to the Python exception type. The owning reference lives on the module
// object (via m.add_object below), so this handle stays valid for the module's lifetime.
// A borrowed py::handle (not a py::object) avoids a Py_DECREF at interpreter shutdown,
// which pybind11 warns against for global storage.
py::handle g_error_type;      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
py::handle g_cancelled_type;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// Raise `type` carrying the message and PysmeshError's two attributes.
void raise_as(py::handle type, const PysmeshError& e) {
  py::object exc = py::reinterpret_borrow<py::object>(type)(py::str(e.what()));
  exc.attr("details") = py::str(e.details);
  exc.attr("face_ids") = py::cast(e.face_ids);
  PyErr_SetObject(type.ptr(), exc.ptr());
}

}  // namespace

void register_error_type(py::module_& m) {
  py::object error_type = py::reinterpret_steal<py::object>(
      PyErr_NewException("pysmesh._core.PysmeshError", PyExc_RuntimeError, nullptr));
  m.add_object("PysmeshError", error_type);  // module now owns a reference
  g_error_type = error_type;                  // borrowed handle for the translator

  // Cancellation derives from PysmeshError so it is a refinement of the existing contract
  // rather than a second one: code that only cares that the operation did not happen keeps
  // catching PysmeshError, and code that must tell "the user stopped it" from "it failed"
  // catches this instead.
  py::object cancelled_type = py::reinterpret_steal<py::object>(PyErr_NewException(
      "pysmesh._core.PysmeshCancelled", error_type.ptr(), nullptr));
  m.add_object("PysmeshCancelled", cancelled_type);
  g_cancelled_type = cancelled_type;

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const CancelledError& e) {
      // Caught before PysmeshError: it derives from it, so the order is what decides which
      // Python type the caller sees.
      raise_as(g_cancelled_type, e);
    } catch (const PysmeshError& e) {
      raise_as(g_error_type, e);
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
  pysmesh::bind_tessellate(m);
  pysmesh::bind_offset(m);
  pysmesh::bind_distance(m);
  pysmesh::bind_classify(m);
  pysmesh::bind_step_xde(m);
  pysmesh::bind_session(m);
}
