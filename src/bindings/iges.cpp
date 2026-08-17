// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-17

// pySMESH binding — IGES import/export: read_iges, write_iges.
//
// The IGES counterpart of step_xde.cpp. Both formats carry a declared length unit in their
// header, and both are useless to a CFD host that receives the geometry without it: a 2 mm
// part read as 2 m is the 1000x defect. read_step_xde solved that for STEP by returning the
// geometry in the file's NATIVE unit plus a metres-per-unit factor. These two do the same for
// IGES, on both sides of the boundary.
//
//   read_iges(path) -> {brep, length_unit, unit_name}
//     brep         : the transferred geometry as BREP bytes, in the IGES file's NATIVE length
//                    unit (OCCT's normalisation to the cascade unit is reversed).
//     length_unit  : metres per model unit (mm -> 0.001, m -> 1.0, inch -> 0.0254). Multiply
//                    BREP coordinates by this to reach SI metres.
//     unit_name    : the IGES unit name the file declares ("MM", "M", "INCH", ...). Feed it
//                    straight back to write_iges to re-export in the same unit.
//
//   write_iges(brep, unit, brep_mode) -> bytes
//     `unit` is the unit the BREP coordinates ARE IN — an argument, never a global. The
//     header declares exactly that unit and the coordinates are written verbatim; the file
//     cannot come out mislabelled the way a unit read from Interface_Static can.
//
// Unit mechanics, verified against OCCT 8.0 sources (not assumed):
//   IGESData_GlobalSection::UnitValue() == UnitFlagValue(flag) / CascadeUnit, i.e. cascade
//   units per file unit. UnitFlagValue is the unit's size in MILLIMETRES (IGESData_BasicEditor
//   §UnitFlagValue). The read side multiplies file coordinates by UnitValue()
//   (IGESToBRep_CurveAndSurface::SetModel) and the write side divides by it
//   (BRepToIGES_BREntity::GetUnit, applied as 1./GetUnit()).
//     - read_iges therefore scales the transferred shape by 1/UnitValue() to return native
//       coordinates, and reports UnitFlagValue(flag) * 1e-3 as length_unit.
//     - write_iges sets the writer model's CascadeUnit to UnitFlagValue(flag), which makes
//       UnitValue() exactly 1: no rescale on the way out.
//   Both paths read the factor off the model in hand, so neither depends on the ambient
//   "xstep.cascade.unit" setting and neither mutates it.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <string>

#include <BRepBuilderAPI_Transform.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <IGESControl_Reader.hxx>
#include <IGESControl_Writer.hxx>
#include <IGESData_BasicEditor.hxx>
#include <IGESData_GlobalSection.hxx>
#include <IGESData_IGESModel.hxx>
#include <TCollection_HAsciiString.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>

#include "common.hpp"

namespace pysmesh {
namespace {

// The ten IGES length units OCCT names, for error messages. Flag 3 ("unit named in the global
// section") is not among them: it carries no value of its own.
constexpr const char* kUnitNames = "MM, CM, M, KM, UM, INCH (IN), MIL, UIN, FT, MI";

// ASCII upper-case. IGES unit names are matched by exact strcmp inside OCCT, so the binding
// normalises the caller's spelling once, here.
std::string upper(const std::string& s) {
  std::string out = s;
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return out;
}

// Read a BREP shape from in-memory bytes (mirrors the file-local helper in the other TUs).
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

// Which IGES BRep write mode a shape can be represented in, following SALOME's
// IGESPlugin_ExportDriver::KindOfBRep (Mantis 0021350):
//   0  only wires/edges/vertices  -> needs the Faces mode (IGES 5.1)
//   1  only shells/solids/compsolids
//   2  faces or anything neutral  -> either mode
//  -1  a compound mixing 0 and 1  -> no single mode carries all of it
int kind_of_brep(const TopoDS_Shape& shape) {
  switch (shape.ShapeType()) {
    case TopAbs_COMPOUND: {
      bool has_simple = false;
      bool has_complex = false;
      for (TopoDS_Iterator it(shape, Standard_True, Standard_True); it.More(); it.Next()) {
        const int sub = kind_of_brep(it.Value());
        if (sub == 0) {
          has_simple = true;
        } else if (sub == 1) {
          has_complex = true;
        } else if (sub == -1) {
          return -1;
        }
      }
      if (has_simple && has_complex) {
        return -1;
      }
      if (has_simple) {
        return 0;
      }
      return has_complex ? 1 : 2;
    }
    case TopAbs_COMPSOLID:
    case TopAbs_SOLID:
    case TopAbs_SHELL:
      return 1;
    case TopAbs_WIRE:
    case TopAbs_EDGE:
    case TopAbs_VERTEX:
      return 0;
    default:
      return 2;
  }
}

py::dict read_iges(const std::string& path) {
  std::string brep_bytes;
  double length_unit = 0.0;
  std::string unit_name;

  {
    py::gil_scoped_release release;

    IGESControl_Reader reader;
    const IFSelect_ReturnStatus status = reader.ReadFile(path.c_str());
    if (status != IFSelect_RetDone) {
      throw PysmeshError("read_iges: IGES parse failed (IFSelect status " +
                          std::to_string(static_cast<int>(status)) + "; '" + path +
                          "' is missing or is not a valid IGES file).");
    }

    const occ::handle<IGESData_IGESModel> model = reader.IGESModel();
    if (model.IsNull()) {
      throw PysmeshError("read_iges: the file loaded but carries no IGES model.");
    }
    const IGESData_GlobalSection& gs = model->GlobalSection();

    // The declared unit. OCCT resolves a global-section unit flag of 0 or 3 ("name given in
    // the header") to a real flag at read time whenever the name is one IGES defines, so a
    // flag that is still 3 — or outside 1..11 — means the file names a unit this format has
    // no value for. Refuse it: length_unit is the whole point of this binding, and a guessed
    // one reintroduces exactly the defect it exists to prevent.
    //
    // Both numbers are taken from the header in one snapshot, before the transfer, so the
    // factor reported to the caller and the factor used to undo OCCT's scaling can never come
    // from two different states.
    const int flag = gs.UnitFlag();
    const double mm_per_unit = IGESData_BasicEditor::UnitFlagValue(flag);
    const double unit_value = gs.UnitValue();  // cascade units per file unit
    if (flag == 3 || mm_per_unit <= 0.0 || !(unit_value > 0.0)) {
      const occ::handle<TCollection_HAsciiString> raw_name = gs.UnitName();
      const std::string named = raw_name.IsNull() ? std::string("<none>")
                                                  : std::string(raw_name->ToCString());
      throw PysmeshError("read_iges: the file declares a length unit this format has no value "
                          "for (global section unit flag " + std::to_string(flag) +
                          ", unit name '" + named + "'). Supported units: " + kUnitNames +
                          ". Refusing to guess: the declared unit is the whole contract of "
                          "this call.");
    }
    length_unit = mm_per_unit * 1.0e-3;
    unit_name = IGESData_BasicEditor::UnitFlagName(flag);

    reader.ClearShapes();
    const int nb_roots = reader.TransferRoots();
    if (nb_roots < 1 || reader.NbShapes() < 1) {
      throw PysmeshError("read_iges: no root entity transferred to a shape (the file holds "
                          "no transferable geometry).");
    }
    const TopoDS_Shape transferred = reader.OneShape();
    if (transferred.IsNull()) {
      throw PysmeshError("read_iges: transferred shape is null.");
    }

    // OCCT hands the shape back in cascade units. unit_value is cascade units per file unit,
    // so its reciprocal returns the geometry to the file's native unit. Identity whenever the
    // two coincide (an mm file with the default mm cascade unit).
    const double to_native = 1.0 / unit_value;
    TopoDS_Shape native = transferred;
    if (std::abs(to_native - 1.0) > 1.0e-12) {
      gp_Trsf trsf;
      trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), to_native);
      native = BRepBuilderAPI_Transform(transferred, trsf, Standard_True).Shape();
    }

    std::ostringstream out;
    BRepTools::Write(native, out);
    brep_bytes = out.str();
  }

  py::dict result;
  result["brep"] = py::bytes(brep_bytes);
  result["length_unit"] = length_unit;
  result["unit_name"] = unit_name;
  return result;
}

py::bytes write_iges(const py::bytes& brep, const std::string& unit, bool brep_mode) {
  const TopoDS_Shape shape = read_brep(brep);

  // Resolve the caller's unit BEFORE any OCCT state is built, so a bad name costs nothing.
  const std::string name = upper(unit);
  const int flag = name.empty() ? 0 : IGESData_BasicEditor::UnitNameFlag(name.c_str());
  if (flag < 1) {
    throw PysmeshError("write_iges: '" + unit + "' is not an IGES length unit. Use one of: " +
                        kUnitNames + ".");
  }
  const double mm_per_unit = IGESData_BasicEditor::UnitFlagValue(flag);

  if (brep_mode) {
    const int kind = kind_of_brep(shape);
    if (kind == -1) {
      throw PysmeshError("write_iges: the shape is a compound mixing solids/shells with "
                          "standalone wires, edges or vertices. IGES BRep mode (5.3) cannot "
                          "carry both, and either choice would silently drop half the shape. "
                          "Export the two parts separately.");
    }
    if (kind == 0) {
      throw PysmeshError("write_iges: the shape holds only wires, edges or vertices, which "
                          "IGES BRep mode (5.3) cannot represent. Pass brep_mode=False to "
                          "write them as IGES 5.1 entities.");
    }
  }

  std::string iges_bytes;
  {
    py::gil_scoped_release release;

    IGESControl_Writer writer(name.c_str(), brep_mode ? 1 : 0);

    // Pin the model's cascade unit to the unit the caller declared. GlobalSection::UnitValue()
    // is UnitFlagValue(flag) / CascadeUnit and the write path divides coordinates by it, so
    // this makes the factor exactly 1: the BREP's numbers reach the file unchanged and the
    // header declares the unit they are actually in. The writer's global section is otherwise
    // all-zero at this point (fresh IGESData_GlobalSection), so nothing already scaled by the
    // constructor's ApplyUnit() needs undoing.
    writer.Model()->ChangeGlobalSection().SetCascadeUnit(mm_per_unit);

    if (!writer.AddShape(shape)) {
      throw PysmeshError("write_iges: the shape could not be translated to IGES entities "
                          "(brep_mode=" + std::string(brep_mode ? "True" : "False") + ").");
    }
    writer.ComputeModel();

    std::ostringstream out;
    if (!writer.Write(out)) {
      throw PysmeshError("write_iges: IGES write failed.");
    }
    iges_bytes = out.str();
  }

  return py::bytes(iges_bytes);
}

}  // namespace

void bind_iges(py::module_& m) {
  m.def("read_iges", &read_iges, py::arg("path"),
        "Import an IGES file via OCCT's IGESControl_Reader. Returns a dict with 'brep' (BREP "
        "bytes in the file's native length unit), 'length_unit' (metres per model unit) and "
        "'unit_name' (the IGES unit the header declares).");
  m.def("write_iges", &write_iges, py::arg("brep"), py::arg("unit"), py::arg("brep_mode"),
        "Export a BREP to IGES bytes. 'unit' is the IGES unit name the BREP coordinates are "
        "already in; the header declares it and the coordinates are written unchanged. "
        "'brep_mode' selects IGES 5.3 BRep entities (True) or 5.1 face entities (False).");
}

}  // namespace pysmesh
