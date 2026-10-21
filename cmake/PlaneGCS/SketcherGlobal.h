// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH shim for FreeCAD's src/Mod/Sketcher/SketcherGlobal.h, which the PlaneGCS headers
// include as "../../SketcherGlobal.h". prepare.py stages this file at that path.
//
// FreeCAD's header defines the DLL export macro of its Sketcher module. pySMESH compiles
// PlaneGCS into a static library that is linked into _core.pyd, so nothing is exported or
// imported and the macro is empty.

#ifndef SKETCHER_GLOBAL_H
#define SKETCHER_GLOBAL_H

#ifndef SketcherExport
#define SketcherExport
#endif

#endif  // SKETCHER_GLOBAL_H
