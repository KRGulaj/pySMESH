// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH shim for FreeCAD's src/Base/Console.h, which PlaneGCS's GCS.cpp includes.
//
// GCS.cpp writes its solver traces through Base::Console().log(), and one warning through
// Base::Console().warning() that is compiled only for an Eigen older than 3.2.2. FreeCAD
// routes these to its report view. pySMESH writes nothing to the standard streams (the
// SALOME traces stay silent unless SALOME_VERBOSE is set), so every message is discarded
// here. The calls keep their printf form: a format and its arguments.

#ifndef PYSMESH_PLANEGCS_BASE_CONSOLE_H
#define PYSMESH_PLANEGCS_BASE_CONSOLE_H

namespace Base {

class ConsoleSingleton {
 public:
  template <typename... Args>
  void log(const char* /*format*/, Args&&... /*args*/) {}

  template <typename... Args>
  void warning(const char* /*format*/, Args&&... /*args*/) {}
};

inline ConsoleSingleton& Console() {
  static ConsoleSingleton console;
  return console;
}

}  // namespace Base

#endif  // PYSMESH_PLANEGCS_BASE_CONSOLE_H
