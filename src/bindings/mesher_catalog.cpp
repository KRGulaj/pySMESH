// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-09

// pySMESH binding — the algorithm and hypothesis catalogue.
//
// One factory, keyed on the name the Python dataclass declares. The typed public surface is
// the dataclass; this layer reads its fields back out of a dict and calls the upstream
// setters. Keeping the pybind boundary untyped means a signature has one place to drift
// rather than two, and `Params::done()` closes the hole that would otherwise open: a field
// added on the Python side with no branch here is refused, not silently dropped.
//
// Every id comes from SMESH_Gen::GetANewId(). This is not a style point. Composite
// algorithms allocate ids from the generator inside their own constructors — the polyhedral
// mesher builds its own 1-D and 2-D sub-meshers that way — so a caller-side counter aliases
// entries in the generator's maps, and the symptom (a later assignment refused as already
// existing) is remote from the cause.
//
// See mesher/mesher.hpp for the file split.

#include "mesher/mesher.hpp"

#include <cmath>
#include <utility>

#include <SMESHDS_Mesh.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_Hypothesis.hxx>
#include <SMESH_Mesh.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>

// --- algorithms ---
// StdMeshers_CompositeHexa_3D.hxx and StdMeshers_CompositeSegment_1D.hxx ship with the SAME
// include guard upstream, so whichever came second was silenced and its class was never
// declared — symmetrically, which is why no include order fixes it. prepare.py gives the
// first header its own guard; see PROVENANCE.md.
#include <StdMeshers_Cartesian_3D.hxx>
#include <StdMeshers_CompositeHexa_3D.hxx>
#include <StdMeshers_CompositeSegment_1D.hxx>
#include <StdMeshers_HexaFromSkin_3D.hxx>
#include <StdMeshers_Hexa_3D.hxx>
#include <StdMeshers_MEFISTO_2D.hxx>
#include <StdMeshers_PolygonPerFace_2D.hxx>
#include <StdMeshers_PolyhedronPerSolid_3D.hxx>
#include <StdMeshers_Prism_3D.hxx>
#include <StdMeshers_Projection_1D.hxx>
#include <StdMeshers_Projection_1D2D.hxx>
#include <StdMeshers_Projection_2D.hxx>
#include <StdMeshers_Projection_3D.hxx>
#include <StdMeshers_QuadFromMedialAxis_1D2D.hxx>
#include <StdMeshers_Quadrangle_2D.hxx>
#include <StdMeshers_RadialPrism_3D.hxx>
#include <StdMeshers_RadialQuadrangle_1D2D.hxx>
#include <StdMeshers_Regular_1D.hxx>
#include <StdMeshers_SegmentAroundVertex_0D.hxx>
#include <StdMeshers_UseExisting_1D2D.hxx>

// --- hypotheses ---
#include <StdMeshers_Adaptive1D.hxx>
#include <StdMeshers_Arithmetic1D.hxx>
#include <StdMeshers_AutomaticLength.hxx>
#include <StdMeshers_BlockRenumber.hxx>
#include <StdMeshers_CartesianParameters3D.hxx>
#include <StdMeshers_Deflection1D.hxx>
#include <StdMeshers_FixedPoints1D.hxx>
#include <StdMeshers_Geometric1D.hxx>
#include <StdMeshers_LayerDistribution.hxx>
#include <StdMeshers_LayerDistribution2D.hxx>
#include <StdMeshers_LengthFromEdges.hxx>
#include <StdMeshers_LocalLength.hxx>
#include <StdMeshers_MaxElementArea.hxx>
#include <StdMeshers_MaxElementVolume.hxx>
#include <StdMeshers_MaxLength.hxx>
#include <StdMeshers_NumberOfLayers.hxx>
#include <StdMeshers_NumberOfLayers2D.hxx>
#include <StdMeshers_NotConformAllowed.hxx>
#include <StdMeshers_NumberOfSegments.hxx>
#include <StdMeshers_ProjectionSource1D.hxx>
#include <StdMeshers_ProjectionSource2D.hxx>
#include <StdMeshers_ProjectionSource3D.hxx>
#include <StdMeshers_Propagation.hxx>
#include <StdMeshers_QuadranglePreference.hxx>
#include <StdMeshers_QuadrangleParams.hxx>
#include <StdMeshers_QuadraticMesh.hxx>
#include <StdMeshers_Reversible1D.hxx>
#include <StdMeshers_SegmentLengthAroundVertex.hxx>
#include <StdMeshers_StartEndLength.hxx>
#include <StdMeshers_ViscousLayers.hxx>
#include <StdMeshers_ViscousLayers2D.hxx>

namespace pysmesh {
namespace mesher {
namespace {

// Allocates every algorithm and hypothesis with an id drawn from the generator, and hands it
// to the Mesher's ownership at once: a setter called next may throw (report N1), and an
// object owned only by a local pointer would then leak. A hypothesis whose setters fail stays
// owned, unassigned, until the Mesher closes, as a refused assignment does.
class Factory {
 public:
  Factory(SMESH_Gen& gen, std::vector<std::unique_ptr<SMESH_Hypothesis>>& owned)
      : gen_(gen), owned_(owned) {}

  template <class T>
  T* make() {
    T* hyp = new T(gen_.GetANewId(), &gen_);
    owned_.emplace_back(hyp);
    return hyp;
  }

 private:
  SMESH_Gen& gen_;
  std::vector<std::unique_ptr<SMESH_Hypothesis>>& owned_;
};

// The text of an upstream exception without SALOME_Exception's "Salome Exception : " lead
// (Utils_SALOME_Exception.cxx, makeText).
std::string upstream_text(const char* what) {
  std::string text = what != nullptr ? what : "";
  const std::string lead = "Salome Exception";
  if (text.compare(0, lead.size(), lead) == 0) {
    const std::size_t colon = text.find(": ");
    text = colon == std::string::npos ? text.substr(lead.size()) : text.substr(colon + 2);
  }
  return text;
}

// The algorithms. All of them are configured entirely through their hypotheses, so none
// takes a parameter of its own — which is why they are a separate, flat table.
SMESH_Hypothesis* make_algorithm(const std::string& name, Factory& f) {
  // 0-D. It meshes nothing itself: it exists so that SegmentLengthAroundVertex on a vertex is
  // taken into account by the 1-D algorithm of the edges that vertex bounds.
  if (name == "SegmentAroundVertex_0D") return f.make<StdMeshers_SegmentAroundVertex_0D>();
  // 1-D
  if (name == "Regular_1D") return f.make<StdMeshers_Regular_1D>();
  if (name == "CompositeSegment_1D") return f.make<StdMeshers_CompositeSegment_1D>();
  if (name == "Projection_1D") return f.make<StdMeshers_Projection_1D>();
  // "Use Edges/Faces to be Created Manually": they create nothing and mark their sub-mesh
  // computed, so elements made by a script stand as that sub-shape's mesh.
  if (name == "UseExisting_1D") return f.make<StdMeshers_UseExisting_1D>();
  // 2-D
  if (name == "UseExisting_2D") return f.make<StdMeshers_UseExisting_2D>();
  if (name == "Quadrangle_2D") return f.make<StdMeshers_Quadrangle_2D>();
  if (name == "MEFISTO_2D") return f.make<StdMeshers_MEFISTO_2D>();
  if (name == "PolygonPerFace_2D") return f.make<StdMeshers_PolygonPerFace_2D>();
  if (name == "Projection_2D") return f.make<StdMeshers_Projection_2D>();
  if (name == "Projection_1D2D") return f.make<StdMeshers_Projection_1D2D>();
  if (name == "QuadFromMedialAxis_1D2D")
    return f.make<StdMeshers_QuadFromMedialAxis_1D2D>();
  if (name == "RadialQuadrangle_1D2D") return f.make<StdMeshers_RadialQuadrangle_1D2D>();
  // 3-D
  if (name == "Cartesian_3D") return f.make<StdMeshers_Cartesian_3D>();
  if (name == "Hexa_3D") return f.make<StdMeshers_Hexa_3D>();
  if (name == "CompositeHexa_3D") return f.make<StdMeshers_CompositeHexa_3D>();
  if (name == "HexaFromSkin_3D") return f.make<StdMeshers_HexaFromSkin_3D>();
  if (name == "Prism_3D") return f.make<StdMeshers_Prism_3D>();
  if (name == "RadialPrism_3D") return f.make<StdMeshers_RadialPrism_3D>();
  if (name == "Projection_3D") return f.make<StdMeshers_Projection_3D>();
  if (name == "PolyhedronPerSolid_3D") return f.make<StdMeshers_PolyhedronPerSolid_3D>();
  return nullptr;
}

// The edges a reversible 1-D distribution runs from their last vertex to their first, sent
// as SMESHDS indices (report C1). StdMeshers_Regular_1D compares each edge's index with this
// list (StdMeshers_Regular_1D.cxx:202-279). Read only when sent, so a caller that never sends
// `reversed_edges` keeps the behaviour it had.
void set_reversed_edges(StdMeshers_Reversible1D* h, Params& p, const Mesher& m) {
  if (!p.has("reversed_edges")) {
    return;
  }
  std::vector<int> indices;
  for (const int ordinal : p.integers("reversed_edges")) {
    indices.push_back(m.meshDS().ShapeToIndex(m.sub_shape("EDGE", ordinal)));
  }
  h->SetReversedEdges(indices);
}

// The 1-D distribution family, plus the two hypotheses that carry no sub-shape reference.
SMESH_Hypothesis* make_1d_hypothesis(const std::string& name, Params& p, Factory& f,
                                     const Mesher& m) {
  if (name == "NumberOfSegments") {
    StdMeshers_NumberOfSegments* h = f.make<StdMeshers_NumberOfSegments>();
    h->SetNumberOfSegments(static_cast<smIdType>(p.integer("count")));
    const int distribution = p.integer("distribution");
    h->SetDistrType(static_cast<StdMeshers_NumberOfSegments::DistrType>(distribution));
    if (distribution == StdMeshers_NumberOfSegments::DT_Scale) {
      h->SetScaleFactor(p.number("scale_factor"));
    } else {
      p.number("scale_factor");  // consumed so the field is not reported as unknown
    }
    if (distribution == StdMeshers_NumberOfSegments::DT_BetaLaw) {
      // SMESH 9.16 places node i at 1 + beta * (1 - r^(1-i/n)) / (1 + r^(1-i/n)) with
      // r = (1 + beta) / (beta - 1) (StdMeshers_Regular_1D::computeBetaLaw). Upstream
      // documents |beta| <= 1 as forbidden for that log, but SetBeta does not check it, and
      // such a value yields NaN positions; so it is refused here. Read for this law only,
      // so a caller that never sends `beta` keeps the behaviour it had.
      const double beta = p.number("beta");
      if (!(std::abs(beta) > 1.0)) {
        throw PysmeshError("NumberOfSegments: the beta law needs |beta| > 1 (got " +
                           std::to_string(beta) + ").");
      }
      h->SetBeta(beta);
    }
    if (distribution == StdMeshers_NumberOfSegments::DT_TabFunc) {
      h->SetConversionMode(p.integer("conversion_mode"));
      h->SetTableFunction(p.numbers("table"));
      p.text("expression");
    } else if (distribution == StdMeshers_NumberOfSegments::DT_ExprFunc) {
      h->SetConversionMode(p.integer("conversion_mode"));
      h->SetExpressionFunction(p.text("expression").c_str());
      p.numbers("table");
    } else {
      p.integer("conversion_mode");
      p.numbers("table");
      p.text("expression");
    }
    set_reversed_edges(h, p, m);
    return h;
  }
  if (name == "Arithmetic1D") {
    StdMeshers_Arithmetic1D* h = f.make<StdMeshers_Arithmetic1D>();
    h->SetLength(p.number("start_length"), true);
    h->SetLength(p.number("end_length"), false);
    set_reversed_edges(h, p, m);
    return h;
  }
  if (name == "StartEndLength") {
    StdMeshers_StartEndLength* h = f.make<StdMeshers_StartEndLength>();
    h->SetLength(p.number("start_length"), true);
    h->SetLength(p.number("end_length"), false);
    set_reversed_edges(h, p, m);
    return h;
  }
  if (name == "Geometric1D") {
    StdMeshers_Geometric1D* h = f.make<StdMeshers_Geometric1D>();
    h->SetStartLength(p.number("start_length"));
    h->SetCommonRatio(p.number("common_ratio"));
    set_reversed_edges(h, p, m);
    return h;
  }
  if (name == "FixedPoints1D") {
    StdMeshers_FixedPoints1D* h = f.make<StdMeshers_FixedPoints1D>();
    std::vector<double> points = p.numbers("points");
    h->SetPoints(points);
    const std::vector<int> counts = p.integers("segment_counts");
    std::vector<smIdType> wide(counts.begin(), counts.end());
    h->SetNbSegments(wide);
    set_reversed_edges(h, p, m);
    return h;
  }
  if (name == "Adaptive1D") {
    StdMeshers_Adaptive1D* h = f.make<StdMeshers_Adaptive1D>();
    h->SetMinSize(p.number("min_size"));
    h->SetMaxSize(p.number("max_size"));
    h->SetDeflection(p.number("deflection"));
    return h;
  }
  if (name == "AutomaticLength") {
    StdMeshers_AutomaticLength* h = f.make<StdMeshers_AutomaticLength>();
    h->SetFineness(p.number("fineness"));
    return h;
  }
  if (name == "Deflection1D") {
    StdMeshers_Deflection1D* h = f.make<StdMeshers_Deflection1D>();
    h->SetDeflection(p.number("deflection"));
    return h;
  }
  if (name == "LocalLength") {
    StdMeshers_LocalLength* h = f.make<StdMeshers_LocalLength>();
    h->SetLength(p.number("length"));
    // Upstream SetPrecision tests the old value, not the new one, so it accepts a negative
    // precision (StdMeshers_LocalLength.cxx: `if (_precision < 0)`). Checked here instead.
    const double precision = p.number("precision");
    if (!(precision >= 0.0)) {
      throw PysmeshError("LocalLength: precision cannot be negative (got " +
                         std::to_string(precision) + ").");
    }
    h->SetPrecision(precision);
    return h;
  }
  if (name == "MaxLength") {
    StdMeshers_MaxLength* h = f.make<StdMeshers_MaxLength>();
    h->SetLength(p.number("length"));
    h->SetUsePreestimatedLength(p.flag("use_preestimated"));
    return h;
  }
  if (name == "SegmentLengthAroundVertex") {
    StdMeshers_SegmentLengthAroundVertex* h = f.make<StdMeshers_SegmentLengthAroundVertex>();
    h->SetLength(p.number("length"));
    return h;
  }
  if (name == "Propagation") return f.make<StdMeshers_Propagation>();
  // Propagates the relative node distribution, not the hypothesis: an opposite edge of a
  // different length gets the same number of nodes at the same fractions of its length.
  if (name == "PropagOfDistribution") return f.make<StdMeshers_PropagOfDistribution>();
  if (name == "QuadraticMesh") return f.make<StdMeshers_QuadraticMesh>();
  return nullptr;
}

// The 2-D and 3-D hypotheses. Those that point at a sub-shape resolve it through the
// Mesher's own per-kind ordinals, and translate to a SMESHDS index only where upstream
// insists on one — a base vertex and the quadrangle corner list are stored as indices.
SMESH_Hypothesis* make_area_hypothesis(const std::string& name, Params& p, Factory& f,
                                       const Mesher& m) {
  if (name == "MaxElementArea") {
    StdMeshers_MaxElementArea* h = f.make<StdMeshers_MaxElementArea>();
    h->SetMaxArea(p.number("max_area"));
    return h;
  }
  if (name == "MaxElementVolume") {
    StdMeshers_MaxElementVolume* h = f.make<StdMeshers_MaxElementVolume>();
    h->SetMaxVolume(p.number("max_volume"));
    return h;
  }
  // The triangle size of MEFISTO_2D taken from the mean length of the boundary segments.
  // Its one parameter, the mode, has the single value 1 upstream, set by the constructor.
  if (name == "LengthFromEdges") return f.make<StdMeshers_LengthFromEdges>();
  if (name == "QuadranglePreference") return f.make<StdMeshers_QuadranglePreference>();
  if (name == "QuadrangleParams") {
    StdMeshers_QuadrangleParams* h = f.make<StdMeshers_QuadrangleParams>();
    h->SetQuadType(static_cast<StdMeshers_QuadType>(p.integer("quad_type")));
    const std::pair<std::string, int> base = p.subshape("base_vertex");
    if (base.second > 0) {
      h->SetTriaVertex(m.meshDS().ShapeToIndex(m.sub_shape(base.first, base.second)));
    }
    std::vector<int> corners;
    for (const int ordinal : p.integers("corner_vertices")) {
      corners.push_back(m.meshDS().ShapeToIndex(m.sub_shape("VERTEX", ordinal)));
    }
    h->SetCorners(corners);
    return h;
  }
  if (name == "NumberOfLayers") {
    StdMeshers_NumberOfLayers* h = f.make<StdMeshers_NumberOfLayers>();
    h->SetNumberOfLayers(p.integer("count"));
    return h;
  }
  if (name == "NumberOfLayers2D") {
    StdMeshers_NumberOfLayers2D* h = f.make<StdMeshers_NumberOfLayers2D>();
    h->SetNumberOfLayers(p.integer("count"));
    return h;
  }
  if (name == "CartesianParameters3D") {
    StdMeshers_CartesianParameters3D* h = f.make<StdMeshers_CartesianParameters3D>();
    const std::vector<double> spacing_from = p.numbers("spacing_from");
    const std::string sx = p.text("spacing_x");
    const std::string sy = p.text("spacing_y");
    const std::string sz = p.text("spacing_z");
    const std::string* per_axis[3] = {&sx, &sy, &sz};
    for (int axis = 0; axis < 3; ++axis) {
      std::vector<std::string> spacing(1, *per_axis[axis]);
      std::vector<double> internal(spacing_from);
      h->SetGridSpacing(spacing, internal, axis);
    }
    h->SetSizeThreshold(p.number("size_threshold"));
    h->SetToAddEdges(p.flag("add_edges"));
    h->SetToCreateFaces(p.flag("create_faces"));
    h->SetToConsiderInternalFaces(p.flag("consider_internal_faces"));
    // New in SMESH 9.16 (9a170f0e1): a boundary polyhedron is replaced by a hexahedron when
    // its volume divided by the volume of the equivalent hexahedron is bigger than `quanta`.
    // Read only when sent, so the existing dataclass, which does not send it, keeps its
    // behaviour. The range is the one SetQuanta accepts (StdMeshers_CartesianParameters3D
    // .cxx:804); it is checked here so that a bad value raises PysmeshError.
    if (p.has("use_quanta")) {
      h->SetToUseQuanta(p.flag("use_quanta"));
      const double quanta = p.number("quanta");
      if (!(quanta >= 1e-6 && quanta <= 1.0)) {
        throw PysmeshError("CartesianParameters3D: quanta must lie in [1e-6, 1] (got " +
                           std::to_string(quanta) + ").");
      }
      h->SetQuanta(quanta);
    }
    return h;
  }
  // Renumbers the hexahedra and nodes of Hexa_3D like a structured i, j, k grid. For a block
  // with edges parallel to the global axes the local axes default to the global ones (SMESH
  // 3d_meshing_hypo.rst). The explicit form names, per block, the solid and its vertices at
  // (0,0,0) and (0,0,1) by entry strings; the Mesher's TCallUp resolves "KIND:ordinal"
  // (report W1.6). `blocks` is read only when sent, so the parameter-free call keeps the
  // behaviour it had.
  if (name == "BlockRenumber") {
    StdMeshers_BlockRenumber* h = f.make<StdMeshers_BlockRenumber>();
    if (p.has("blocks")) {
      std::vector<StdMeshers_BlockCS> blocks;
      for (const std::vector<int>& row : p.integer_rows("blocks")) {
        if (row.size() != 3) {
          throw PysmeshError("BlockRenumber: each block is (solid, vertex_000, vertex_001); "
                             "got " + std::to_string(row.size()) + " ordinals.");
        }
        m.sub_shape("SOLID", row[0]);  // validates each ordinal, naming a bad one
        m.sub_shape("VERTEX", row[1]);
        m.sub_shape("VERTEX", row[2]);
        StdMeshers_BlockCS cs;
        cs._solid = "SOLID:" + std::to_string(row[0]);
        cs._vertex000 = "VERTEX:" + std::to_string(row[1]);
        cs._vertex001 = "VERTEX:" + std::to_string(row[2]);
        blocks.push_back(cs);
      }
      h->SetBlocksOrientation(blocks);
    }
    return h;
  }
  // Lets local algorithms that mesh their own boundary sit side by side on adjacent
  // sub-shapes, which gives a non-conformal mesh. Global only (SMESH_Mesh.cxx:658-668).
  if (name == "NotConformAllowed") return f.make<StdMeshers_NotConformAllowed>();
  return nullptr;
}

// The hypotheses that name another part of the model: a projection source, or the wall set a
// viscous layer grows on.
SMESH_Hypothesis* make_referring_hypothesis(const std::string& name, Params& p, Factory& f,
                                            const Mesher& m) {
  auto vertex_of = [&m](const std::pair<std::string, int>& ref) -> TopoDS_Shape {
    if (ref.second <= 0) {
      return TopoDS_Shape();
    }
    return m.sub_shape(ref.first, ref.second);
  };

  if (name == "ProjectionSource1D") {
    StdMeshers_ProjectionSource1D* h = f.make<StdMeshers_ProjectionSource1D>();
    const std::pair<std::string, int> src = p.subshape("source_edge");
    h->SetSourceEdge(m.sub_shape(src.first, src.second));
    h->SetVertexAssociation(vertex_of(p.subshape("source_vertex")),
                            vertex_of(p.subshape("target_vertex")));
    return h;
  }
  if (name == "ProjectionSource2D") {
    StdMeshers_ProjectionSource2D* h = f.make<StdMeshers_ProjectionSource2D>();
    const std::pair<std::string, int> src = p.subshape("source_face");
    h->SetSourceFace(m.sub_shape(src.first, src.second));
    h->SetVertexAssociation(vertex_of(p.subshape("source_vertex1")),
                            vertex_of(p.subshape("source_vertex2")),
                            vertex_of(p.subshape("target_vertex1")),
                            vertex_of(p.subshape("target_vertex2")));
    return h;
  }
  if (name == "ProjectionSource3D") {
    StdMeshers_ProjectionSource3D* h = f.make<StdMeshers_ProjectionSource3D>();
    const std::pair<std::string, int> src = p.subshape("source_solid");
    h->SetSource3DShape(m.sub_shape(src.first, src.second));
    h->SetVertexAssociation(vertex_of(p.subshape("source_vertex1")),
                            vertex_of(p.subshape("source_vertex2")),
                            vertex_of(p.subshape("target_vertex1")),
                            vertex_of(p.subshape("target_vertex2")));
    return h;
  }
  if (name == "ViscousLayers" || name == "ViscousLayers2D") {
    StdMeshers_ViscousLayers* h = name == "ViscousLayers"
                                      ? f.make<StdMeshers_ViscousLayers>()
                                      : f.make<StdMeshers_ViscousLayers2D>();
    h->SetTotalThickness(p.number("total_thickness"));
    h->SetNumberLayers(p.integer("layer_count"));
    h->SetStretchFactor(p.number("stretch_factor"));
    h->SetMethod(static_cast<StdMeshers_ViscousLayers::ExtrusionMethod>(p.integer("method")));
    h->SetGroupName(p.text("group_name"));
    // SetBndShapes wants SMESHDS indices, so the caller's ordinals are translated here and
    // an ordinal never reaches SMESH raw.
    std::vector<int> indices;
    const bool ignore = p.flag("ignore");
    const std::string kind = name == "ViscousLayers" ? "FACE" : "EDGE";
    for (const int ordinal : p.integers("boundary")) {
      indices.push_back(m.meshDS().ShapeToIndex(m.sub_shape(kind, ordinal)));
    }
    h->SetBndShapes(indices, ignore);
    return h;
  }
  return nullptr;
}

}  // namespace

SMESH_Hypothesis* Mesher::build(const std::string& name, const py::dict& values) {
  ensure_open();
  Factory factory(*gen_, owned_);
  Params p(name.c_str(), values);
  // An upstream setter refuses a bad value with SALOME_Exception, and OCCT with
  // Standard_Failure; both are std::exception. Either reaches the caller as PysmeshError,
  // naming the hypothesis and the parameter read last, which is the one being set. A Python
  // exception raised while a value is read passes through unchanged.
  try {
    SMESH_Hypothesis* hyp = make_algorithm(name, factory);
    if (hyp == nullptr) {
      hyp = make_1d_hypothesis(name, p, factory, *this);
    }
    if (hyp == nullptr) {
      hyp = make_area_hypothesis(name, p, factory, *this);
    }
    if (hyp == nullptr) {
      hyp = make_referring_hypothesis(name, p, factory, *this);
    }
    if (hyp == nullptr) {
      // A layer distribution carries a 1-D hypothesis of its own, so it is built through
      // the same factory recursively rather than through a second, parallel one. The 2-D
      // form, for RadialQuadrangle_1D2D, is the same class under its own name.
      if (name == "LayerDistribution" || name == "LayerDistribution2D") {
        const py::dict spec = p.nested("distribution");
        // A layer distribution spaces the layers between two shells, not the nodes of an
        // edge of the model, so an edge to reverse names nothing there (report C1).
        const py::dict inner_params = spec["params"].cast<py::dict>();
        if (inner_params.contains("reversed_edges") &&
            py::len(inner_params["reversed_edges"]) > 0) {
          throw PysmeshError(name + ": its distribution sets reversed_edges, which a layer "
                             "distribution cannot use: it spaces the layers between two "
                             "shells, not the nodes of an edge. Leave reversed_edges empty.");
        }
        SMESH_Hypothesis* inner = nullptr;
        try {
          inner = build(spec["name"].cast<std::string>(), spec["params"].cast<py::dict>());
        } catch (const PysmeshError& e) {
          throw PysmeshError(name + ": its distribution is refused. " + e.what(), e.details,
                             e.face_ids);
        }
        StdMeshers_LayerDistribution* h =
            name == "LayerDistribution"
                ? factory.make<StdMeshers_LayerDistribution>()
                : factory.make<StdMeshers_LayerDistribution2D>();
        h->SetLayerDistribution(inner);
        hyp = h;
      }
    }
    if (hyp == nullptr) {
      throw PysmeshError("Mesher: unknown algorithm or hypothesis '" + name + "'.");
    }
    // Refuses any field the branch above did not read, so a dataclass and its factory
    // branch cannot drift apart silently.
    p.done();
    return hyp;
  } catch (const PysmeshError&) {
    throw;
  } catch (const py::error_already_set&) {
    throw;
  } catch (const std::exception& e) {
    const std::string key = p.last();
    throw PysmeshError("Mesher: " + name + " refused its parameters" +
                           (key.empty() ? std::string() : " at '" + key + "'") + ": " +
                           upstream_text(e.what()) + ".",
                       "The hypothesis is not assigned.");
  }
}

}  // namespace mesher
}  // namespace pysmesh
