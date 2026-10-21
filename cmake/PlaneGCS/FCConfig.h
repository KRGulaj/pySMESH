// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH shim for FreeCAD's src/FCConfig.h, which PlaneGCS's GCS.cpp includes.
//
// FreeCAD's header sets platform and library switches for the whole of FreeCAD. GCS.cpp
// reads none of them: it includes the header and uses no macro that the header defines. So
// the shim defines nothing.

#ifndef PYSMESH_PLANEGCS_FCCONFIG_H
#define PYSMESH_PLANEGCS_FCCONFIG_H

#endif  // PYSMESH_PLANEGCS_FCCONFIG_H
