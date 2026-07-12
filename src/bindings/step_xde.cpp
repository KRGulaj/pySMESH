// pySMESH binding — STEP XDE import/export: read_step_xde, write_step_xde.
//
// The CFD headliner (report §6.1/§6.2): unlike Gmsh's importShapes, which discards STEP
// product names, colours, and the file's length unit, read_step_xde carries all three across
// the boundary so downstream tooling receives pre-tagged geometry.
//
//   read_step_xde(data_or_path) -> {brep, face_labels, solid_labels, length_unit}
//     brep         : the transferred geometry as BREP bytes, in the STEP file's NATIVE length
//                    unit (no silent rescale — OCCT's default mm scaling is reversed).
//     length_unit  : metres per model unit (mm -> 0.001, m -> 1.0, inch -> 0.0254). Multiply
//                    BREP coordinates by this to reach SI metres — the fix for the
//                    mm-imported-as-metres bug.
//     face_labels  : [{face_id, name, color}] for faces carrying a name or colour.
//     solid_labels : [{solid_id, name, color}] for solids carrying a name or colour.
//                    ids are the 1-based TopExp ordinals of the returned brep, so load_brep()
//                    on `brep` reproduces exactly these Shape.faces()/.solids() ids.
//
//   write_step_xde(brep, face_names, face_colors, name) -> bytes
//     The near-free XDE round-trip (report §6.6): tag a BREP with a product name and per-face
//     names/colours and emit STEP bytes via STEPCAFControl_Writer.
//
// OCAF lifetime: the TDocStd_Document is created via the XCAFApp_Application singleton, kept
// alive only for the read, and every label/colour/name is copied into plain C++ values before
// the document is closed — no OCAF handle ever crosses the Python boundary.

#include <array>
#include <cmath>
#include <cstdint>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

#include <BRepBuilderAPI_Transform.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <NCollection_Sequence.hxx>
#include <Quantity_Color.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPControl_StepModelType.hxx>
#include <TCollection_AsciiString.hxx>
#include <TCollection_ExtendedString.hxx>
#include <TDF_Label.hxx>
#include <TDF_LabelSequence.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS_Shape.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_ColorTool.hxx>
#include <XCAFDoc_ColorType.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>

#include "common.hpp"

namespace pysmesh {
namespace {

// A name/colour record keyed to a 1-based TopExp ordinal of the returned brep.
struct EntityLabel {
  int id;
  std::string name;         // empty if the entity has no name
  bool has_color = false;   // false -> no colour assigned
  std::array<double, 3> rgb{};
};

// UTF-16 XDE string -> std::string. STEP product names are near-universally ASCII; any
// non-ASCII codepoint folds to '?' (documented). Keeps the boundary dependency-free.
std::string to_ascii(const TCollection_ExtendedString& ext) {
  const TCollection_AsciiString ascii(ext, '?');
  return std::string(ascii.ToCString());
}

// Metres per one unit of the given STEP length-unit name (case-insensitive). OCCT's FileUnits
// reports the file's declared unit; this maps it to the SI factor flux multiplies coordinates
// by. Returns 0.0 for an unrecognised unit (caller falls back to millimetres).
double metres_per_unit(const TCollection_AsciiString& raw) {
  TCollection_AsciiString n = raw;
  n.UpperCase();
  const std::string s(n.ToCString());
  if (s == "MM" || s == "MILLIMETRE" || s == "MILLIMETER") return 1.0e-3;
  if (s == "M" || s == "METRE" || s == "METER") return 1.0;
  if (s == "CM" || s == "CENTIMETRE" || s == "CENTIMETER") return 1.0e-2;
  if (s == "DM" || s == "DECIMETRE" || s == "DECIMETER") return 1.0e-1;
  if (s == "KM" || s == "KILOMETRE" || s == "KILOMETER") return 1.0e3;
  if (s == "UM" || s == "MICROMETRE" || s == "MICROMETER" || s == "MICRON") return 1.0e-6;
  if (s == "NM" || s == "NANOMETRE" || s == "NANOMETER") return 1.0e-9;
  if (s == "INCH" || s == "IN") return 0.0254;
  if (s == "FT" || s == "FOOT" || s == "FEET") return 0.3048;
  if (s == "YD" || s == "YARD") return 0.9144;
  if (s == "MIL" || s == "THOU") return 2.54e-5;
  return 0.0;  // unknown -> caller keeps OCCT's millimetre normalisation
}

// Name attached to a shape's XDE label, or "" if none.
std::string name_of(const occ::handle<XCAFDoc_ShapeTool>& st, const TopoDS_Shape& s) {
  const TDF_Label lab = st->FindShape(s, Standard_False);
  if (lab.IsNull()) {
    return "";
  }
  occ::handle<TDataStd_Name> attr;
  if (lab.FindAttribute(TDataStd_Name::GetID(), attr)) {
    return to_ascii(attr->Get());
  }
  return "";
}

// Colour attached to a shape (surface colour preferred, then generic), if any.
bool color_of(const occ::handle<XCAFDoc_ColorTool>& ct, const TopoDS_Shape& s,
              std::array<double, 3>& rgb) {
  Quantity_Color col;
  if (ct->GetColor(s, XCAFDoc_ColorSurf, col) || ct->GetColor(s, XCAFDoc_ColorGen, col)) {
    rgb = {col.Red(), col.Green(), col.Blue()};
    return true;
  }
  return false;
}

// Read a STEP model into `reader`, from bytes (ReadStream) or a filesystem path (ReadFile).
IFSelect_ReturnStatus read_into(STEPCAFControl_Reader& reader, const py::object& src) {
  if (py::isinstance<py::bytes>(src)) {
    const std::string buffer = src.cast<std::string>();
    std::istringstream stream(buffer);
    return reader.ReadStream("stepdata", stream);
  }
  if (py::isinstance<py::str>(src)) {
    const std::string path = src.cast<std::string>();
    return reader.ReadFile(path.c_str());
  }
  throw PysmeshError("read_step_xde: expected STEP bytes or a path string.");
}

py::dict read_step_xde(const py::object& data_or_path) {
  STEPCAFControl_Reader reader;
  reader.SetNameMode(true);
  reader.SetColorMode(true);
  reader.SetLayerMode(false);

  std::string brep_bytes;
  double length_unit = 0.001;  // metres per returned-model unit (native)
  std::vector<EntityLabel> face_labels;
  std::vector<EntityLabel> solid_labels;

  occ::handle<XCAFApp_Application> app = XCAFApp_Application::GetApplication();
  occ::handle<TDocStd_Document> doc;
  app->NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc);

  {
    py::gil_scoped_release release;

    const IFSelect_ReturnStatus status = read_into(reader, data_or_path);
    if (status != IFSelect_RetDone) {
      throw PysmeshError("read_step_xde: STEP parse failed (IFSelect status " +
                          std::to_string(static_cast<int>(status)) +
                          "; not a valid STEP file/bytes).");
    }
    // The file's declared length unit (before transfer normalises geometry to mm). OCCT's
    // FileUnits reports the unit *name*; metres_per_unit maps it to the SI factor. Read it
    // before Transfer, while the model still carries the original unit context.
    double metres = 1.0e-3;  // default: assume millimetres
    NCollection_Sequence<TCollection_AsciiString> len_units, ang_units, solid_units;
    reader.ChangeReader().FileUnits(len_units, ang_units, solid_units);
    if (len_units.Length() >= 1) {
      const double m = metres_per_unit(len_units.First());
      if (m > 0.0) {
        metres = m;
      }
    }
    length_unit = metres;              // metres per native model unit
    const double f_mm = metres * 1.0e3;  // millimetres per native unit (OCCT normalises to mm)

    if (!reader.Transfer(doc)) {
      throw PysmeshError("read_step_xde: STEP->XDE transfer produced no shape.");
    }

    occ::handle<XCAFDoc_ShapeTool> st = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
    occ::handle<XCAFDoc_ColorTool> ct = XCAFDoc_DocumentTool::ColorTool(doc->Main());

    TDF_LabelSequence free_labels;
    st->GetFreeShapes(free_labels);
    if (free_labels.Length() < 1) {
      throw PysmeshError("read_step_xde: STEP file contains no free shapes.");
    }
    const TopoDS_Shape orig_whole = XCAFDoc_ShapeTool::GetOneShape(free_labels);
    if (orig_whole.IsNull()) {
      throw PysmeshError("read_step_xde: transferred shape is null.");
    }

    // Reverse OCCT's implicit scale-to-mm so the returned geometry is in the file's NATIVE
    // unit (1 native unit = f_mm mm). Identity when the file is already in mm (f_mm == 1).
    const double to_native = (f_mm != 0.0) ? (1.0 / f_mm) : 1.0;
    const bool scaled = std::abs(to_native - 1.0) > 1.0e-12;
    BRepBuilderAPI_Transform xf(gp_Trsf{});
    TopoDS_Shape ret_whole = orig_whole;
    if (scaled) {
      gp_Trsf trsf;
      trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), to_native);
      xf = BRepBuilderAPI_Transform(orig_whole, trsf, Standard_True);
      ret_whole = xf.Shape();
    }

    // Ordinal maps of the RETURNED geometry: these ids are what load_brep(brep) reproduces.
    TopTools_IndexedMapOfShape ret_solids;
    TopTools_IndexedMapOfShape ret_faces;
    TopExp::MapShapes(ret_whole, TopAbs_SOLID, ret_solids);
    TopExp::MapShapes(ret_whole, TopAbs_FACE, ret_faces);

    // Names/colours live on the ORIGINAL (in-document) shapes; translate each to its returned
    // ordinal through ModifiedShape (identity when unscaled), so ids stay correct after scale.
    TopTools_IndexedMapOfShape orig_solids;
    TopTools_IndexedMapOfShape orig_faces;
    TopExp::MapShapes(orig_whole, TopAbs_SOLID, orig_solids);
    TopExp::MapShapes(orig_whole, TopAbs_FACE, orig_faces);

    auto collect = [&](const TopTools_IndexedMapOfShape& orig_map,
                       const TopTools_IndexedMapOfShape& ret_map,
                       std::vector<EntityLabel>& out) {
      for (int i = 1; i <= orig_map.Extent(); ++i) {
        const TopoDS_Shape& orig = orig_map.FindKey(i);
        const TopoDS_Shape ret = scaled ? xf.ModifiedShape(orig) : orig;
        const int id = ret_map.FindIndex(ret);
        if (id < 1) {
          continue;  // sub-shape not present in the returned map (should not happen)
        }
        EntityLabel entry;
        entry.id = id;
        entry.name = name_of(st, orig);
        entry.has_color = color_of(ct, orig, entry.rgb);
        if (!entry.name.empty() || entry.has_color) {
          out.push_back(entry);
        }
      }
    };
    collect(orig_solids, ret_solids, solid_labels);
    collect(orig_faces, ret_faces, face_labels);

    std::ostringstream out;
    BRepTools::Write(ret_whole, out);
    brep_bytes = out.str();
  }

  app->Close(doc);

  // Build the Python result (GIL held).
  auto to_entry = [](const EntityLabel& e, const char* id_key) {
    py::dict d;
    d[id_key] = e.id;
    d["name"] = e.name;
    if (e.has_color) {
      d["color"] = py::make_tuple(e.rgb[0], e.rgb[1], e.rgb[2]);
    } else {
      d["color"] = py::none();
    }
    return d;
  };

  py::list face_out;
  for (const EntityLabel& e : face_labels) {
    face_out.append(to_entry(e, "face_id"));
  }
  py::list solid_out;
  for (const EntityLabel& e : solid_labels) {
    solid_out.append(to_entry(e, "solid_id"));
  }

  py::dict result;
  result["brep"] = py::bytes(brep_bytes);
  result["length_unit"] = length_unit;
  result["face_labels"] = face_out;
  result["solid_labels"] = solid_out;
  return result;
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

py::bytes write_step_xde(const py::bytes& brep, const py::dict& face_names,
                         const py::dict& face_colors, const std::string& name) {
  const TopoDS_Shape shape = read_brep(brep);

  // Materialise the Python maps into C++ before releasing the GIL.
  std::vector<std::pair<int, std::string>> names;
  for (const auto& item : face_names) {
    names.emplace_back(item.first.cast<int>(), item.second.cast<std::string>());
  }
  std::vector<std::pair<int, std::array<double, 3>>> colors;
  for (const auto& item : face_colors) {
    const auto rgb = item.second.cast<std::array<double, 3>>();
    colors.emplace_back(item.first.cast<int>(), rgb);
  }

  std::string step_bytes;
  {
    py::gil_scoped_release release;

    TopTools_IndexedMapOfShape faces;
    TopExp::MapShapes(shape, TopAbs_FACE, faces);

    occ::handle<XCAFApp_Application> app = XCAFApp_Application::GetApplication();
    occ::handle<TDocStd_Document> doc;
    app->NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc);
    occ::handle<XCAFDoc_ShapeTool> st = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
    occ::handle<XCAFDoc_ColorTool> ct = XCAFDoc_DocumentTool::ColorTool(doc->Main());

    const TDF_Label shape_lab = st->AddShape(shape, Standard_False);
    if (!name.empty()) {
      TDataStd_Name::Set(shape_lab, TCollection_ExtendedString(name.c_str()));
    }

    auto face_by_id = [&](int id) -> TopoDS_Shape {
      if (id < 1 || id > faces.Extent()) {
        throw PysmeshError("write_step_xde: face id " + std::to_string(id) +
                            " out of range (shape has " + std::to_string(faces.Extent()) +
                            " faces).");
      }
      return faces.FindKey(id);
    };
    for (const auto& [id, fname] : names) {
      const TopoDS_Shape f = face_by_id(id);
      const TDF_Label flab = st->AddSubShape(shape_lab, f);
      if (!flab.IsNull()) {
        TDataStd_Name::Set(flab, TCollection_ExtendedString(fname.c_str()));
      }
    }
    for (const auto& [id, rgb] : colors) {
      const TopoDS_Shape f = face_by_id(id);
      ct->SetColor(f, Quantity_Color(rgb[0], rgb[1], rgb[2], Quantity_TOC_RGB),
                   XCAFDoc_ColorSurf);
    }

    STEPCAFControl_Writer writer;
    writer.SetColorMode(true);
    writer.SetNameMode(true);
    if (!writer.Transfer(doc, STEPControl_AsIs)) {
      throw PysmeshError("write_step_xde: STEP transfer failed.");
    }
    std::ostringstream out;
    if (writer.WriteStream(out) != IFSelect_RetDone) {
      throw PysmeshError("write_step_xde: STEP write failed.");
    }
    step_bytes = out.str();
    app->Close(doc);
  }

  return py::bytes(step_bytes);
}

}  // namespace

void bind_step_xde(py::module_& m) {
  m.def("read_step_xde", &read_step_xde, py::arg("data_or_path"),
        "Import a STEP file (bytes or path) via OCCT XDE. Returns a dict with 'brep' (BREP "
        "bytes in the file's native length unit), 'length_unit' (metres per model unit), "
        "'face_labels' and 'solid_labels' (name/colour keyed to the returned brep's 1-based "
        "TopExp ordinals).");
  m.def("write_step_xde", &write_step_xde, py::arg("brep"), py::arg("face_names"),
        py::arg("face_colors"), py::arg("name"),
        "Export a BREP to STEP bytes via OCCT XDE, tagging the product with 'name' and each "
        "face id (1-based TopExp ordinal) with a name and/or RGB colour.");
}

}  // namespace pysmesh
