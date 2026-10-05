// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-05

// pySMESH binding — the NETGEN algorithms and hypotheses of NETGENPlugin V9_16_0, and the
// lock that every NETGEN compute holds.
//
// This is the one binding file that includes the plugin's headers, and through them
// netgen's own (occgeom.hpp, meshing.hpp). CMake compiles it with PYSMESH_NETGEN_DEFINITIONS,
// the define set of netgen and of the plugin: another set would give netgen::Mesh another
// layout in this file only, which corrupts memory without a word.
//
// Two facts shape the code:
//
//   * **netgen keeps its state in globals**: mparam, multithread, testout, and the plugin's
//     local-size maps (NETGENPlugin_Mesher.cxx:130-137). pySMESH releases the GIL during a
//     compute, so two threads could run netgen at once. Every compute that runs a NETGEN
//     algorithm holds netgen_mutex() for its whole run (Mesher::compute).
//   * **The fineness preset depends on the call order.** SetFineness applies its preset
//     only when the fineness changes, and SetGrowthRate, SetNbSegPerEdge and
//     SetNbSegPerRadius switch the fineness to UserDefined (NETGENPlugin_Hypothesis.cxx).
//     So the fineness is set first and the explicit values after it, and an explicit value
//     is refused unless the fineness is UserDefined.

#include "mesher/mesher.hpp"

#include <fstream>
#include <mutex>
#include <string>

#include <SMESH_Gen.hxx>
#include <SMESH_Hypothesis.hxx>

#include <NETGENPlugin_Hypothesis.hxx>
#include <NETGENPlugin_Hypothesis_2D.hxx>
#include <NETGENPlugin_NETGEN_2D.hxx>
#include <NETGENPlugin_NETGEN_2D3D.hxx>
#include <NETGENPlugin_NETGEN_2D_ONLY.hxx>
#include <NETGENPlugin_NETGEN_3D.hxx>
#include <NETGENPlugin_SimpleHypothesis_2D.hxx>
#include <NETGENPlugin_SimpleHypothesis_3D.hxx>

namespace pysmesh {
namespace mesher {
namespace {

using Owned = std::vector<std::unique_ptr<SMESH_Hypothesis>>;

// An algorithm or hypothesis with its id drawn from the generator, owned by the Mesher at
// once, as mesher_catalog.cpp's Factory does.
template <class T>
T* adopt(SMESH_Gen& gen, Owned& owned) {
  T* hyp = new T(gen.GetANewId(), &gen);
  owned.emplace_back(hyp);
  return hyp;
}

// A number the caller must give as positive, refused with the hypothesis and field named.
double positive(Params& p, const char* owner, const char* key) {
  const double value = p.number(key);
  if (!(value > 0.0)) {
    throw PysmeshError(std::string(owner) + ": " + key + " must be > 0 (got " +
                       std::to_string(value) + ").");
  }
  return value;
}

// The fields NETGEN_Parameters and NETGEN_Parameters_2D share. `volume` adds the two that
// only the volume mesher reads: the volume optimisation steps and the Delaunay switch.
void set_parameters(NETGENPlugin_Hypothesis* h, const std::string& owner, Params& p,
                    const Mesher& m, bool volume) {
  const char* who = owner.c_str();
  h->SetMaxSize(positive(p, who, "max_size"));
  const double min_size = p.number("min_size");
  if (!(min_size >= 0.0)) {
    throw PysmeshError(owner + ": min_size cannot be negative (got " +
                       std::to_string(min_size) + ").");
  }
  h->SetMinSize(min_size);
  h->SetSecondOrder(p.flag("second_order"));
  h->SetOptimize(p.flag("optimize"));

  const int fineness = p.integer("fineness");
  if (fineness < NETGENPlugin_Hypothesis::VeryCoarse ||
      fineness > NETGENPlugin_Hypothesis::UserDefined) {
    throw PysmeshError(owner + ": unknown fineness " + std::to_string(fineness) + ".");
  }
  h->SetFineness(static_cast<NETGENPlugin_Hypothesis::Fineness>(fineness));
  const char* preset_keys[3] = {"growth_rate", "segments_per_edge", "segments_per_radius"};
  for (const char* key : preset_keys) {
    if (p.has(key) && fineness != NETGENPlugin_Hypothesis::UserDefined) {
      throw PysmeshError(owner + ": " + key + " is set by the fineness preset; give it only "
                         "with Fineness.USER_DEFINED.");
    }
  }
  if (p.has("growth_rate")) {
    h->SetGrowthRate(positive(p, who, "growth_rate"));
  }
  if (p.has("segments_per_edge")) {
    h->SetNbSegPerEdge(positive(p, who, "segments_per_edge"));
  }
  if (p.has("segments_per_radius")) {
    h->SetNbSegPerRadius(positive(p, who, "segments_per_radius"));
  }

  if (p.has("chordal_error")) {
    h->SetChordalErrorEnabled(true);
    h->SetChordalError(positive(p, who, "chordal_error"));
  }

  // A local size names a sub-shape of the meshed shape; the plugin resolves the entry
  // "KIND:ordinal" through the mesh's call-up (NETGENPlugin_local_size_by_subshape.patch).
  for (const py::handle item : p.list("local_sizes")) {
    const py::sequence pair = item.cast<py::sequence>();
    if (py::len(pair) != 2) {
      throw PysmeshError(owner + ": each local size is (sub-shape, size).");
    }
    const py::sequence where = pair[0].cast<py::sequence>();
    if (py::len(where) != 2) {
      throw PysmeshError(owner + ": a local size names its sub-shape as (kind, ordinal).");
    }
    const std::string kind = where[0].cast<std::string>();
    const int ordinal = where[1].cast<int>();
    m.sub_shape(kind, ordinal);  // validates the kind and the ordinal, naming a bad one
    const double size = pair[1].cast<double>();
    require_finite(owner, "local_sizes", size);
    if (!(size > 0.0)) {
      throw PysmeshError(owner + ": the local size on " + kind + " " +
                         std::to_string(ordinal) + " must be > 0 (got " +
                         std::to_string(size) + ").");
    }
    h->SetLocalSizeOnEntry(kind + ":" + std::to_string(ordinal), size);
  }

  const std::string size_file = p.text("mesh_size_file");
  if (!size_file.empty()) {
    if (!std::ifstream(size_file).good()) {
      throw PysmeshError(owner + ": cannot read the mesh size file '" + size_file + "'.");
    }
    h->SetMeshSizeFile(size_file);
  }

  h->SetQuadAllowed(p.flag("quad_allowed"));
  h->SetSurfaceCurvature(p.flag("surface_curvature"));
  h->SetFuseEdges(p.flag("fuse_edges"));
  const int surface_steps = p.integer("surface_optimization_steps");
  if (surface_steps < 0) {
    throw PysmeshError(owner + ": surface_optimization_steps cannot be negative.");
  }
  h->SetNbSurfOptSteps(surface_steps);
  h->SetElemSizeWeight(p.number("element_size_weight"));
  h->SetWorstElemMeasure(p.integer("worst_element_measure"));
  h->SetCheckOverlapping(p.flag("check_overlapping"));
  h->SetCheckChartBoundary(p.flag("check_chart_boundary"));
  if (volume) {
    const int volume_steps = p.integer("volume_optimization_steps");
    if (volume_steps < 0) {
      throw PysmeshError(owner + ": volume_optimization_steps cannot be negative.");
    }
    h->SetNbVolOptSteps(volume_steps);
    h->SetUseDelauney(p.flag("use_delaunay"));
  }
  if (p.has("threads")) {
    const int threads = p.integer("threads");
    if (threads < 1) {
      throw PysmeshError(owner + ": threads must be >= 1 (got " + std::to_string(threads) +
                         ").");
    }
    h->SetNbThreads(threads);
  }
}

// NETGEN_SimpleParameters_2D / _3D: one segment rule, and an area (and volume) bound or
// the size taken from the edges (and faces).
void set_simple(NETGENPlugin_SimpleHypothesis_2D* h, const std::string& owner, Params& p) {
  const char* who = owner.c_str();
  const bool by_count = p.has("number_of_segments");
  const bool by_length = p.has("local_length");
  if (by_count == by_length) {
    throw PysmeshError(owner + ": give exactly one of number_of_segments and local_length.");
  }
  if (by_count) {
    const int count = p.integer("number_of_segments");
    if (count < 1) {
      throw PysmeshError(owner + ": number_of_segments must be >= 1 (got " +
                         std::to_string(count) + ").");
    }
    h->SetNumberOfSegments(count);
  } else {
    h->SetLocalLength(positive(p, who, "local_length"));
  }
  if (p.has("max_element_area")) {
    h->SetMaxElementArea(positive(p, who, "max_element_area"));
  } else {
    h->LengthFromEdges();
  }
  h->SetAllowQuadrangles(p.flag("allow_quadrangles"));
}

}  // namespace

std::mutex& netgen_mutex() {
  static std::mutex mutex;
  return mutex;
}

bool is_netgen_algorithm(const std::string& name) {
  return name == "NETGEN_3D" || name == "NETGEN_2D" || name == "NETGEN_2D3D" ||
         name == "NETGEN_2D_ONLY";
}

SMESH_Hypothesis* make_netgen(const std::string& name, Params& p, SMESH_Gen& gen,
                              Owned& owned, const Mesher& m) {
  // Algorithms. All four take their parameters from the hypotheses beside them.
  if (name == "NETGEN_3D") return adopt<NETGENPlugin_NETGEN_3D>(gen, owned);
  if (name == "NETGEN_2D") return adopt<NETGENPlugin_NETGEN_2D>(gen, owned);
  if (name == "NETGEN_2D3D") return adopt<NETGENPlugin_NETGEN_2D3D>(gen, owned);
  if (name == "NETGEN_2D_ONLY") return adopt<NETGENPlugin_NETGEN_2D_ONLY>(gen, owned);
  // Hypotheses.
  if (name == "NETGEN_Parameters") {
    NETGENPlugin_Hypothesis* h = adopt<NETGENPlugin_Hypothesis>(gen, owned);
    set_parameters(h, name, p, m, /*volume=*/true);
    return h;
  }
  if (name == "NETGEN_Parameters_2D") {
    NETGENPlugin_Hypothesis_2D* h = adopt<NETGENPlugin_Hypothesis_2D>(gen, owned);
    set_parameters(h, name, p, m, /*volume=*/false);
    return h;
  }
  if (name == "NETGEN_SimpleParameters_2D") {
    NETGENPlugin_SimpleHypothesis_2D* h = adopt<NETGENPlugin_SimpleHypothesis_2D>(gen, owned);
    set_simple(h, name, p);
    return h;
  }
  if (name == "NETGEN_SimpleParameters_3D") {
    NETGENPlugin_SimpleHypothesis_3D* h = adopt<NETGENPlugin_SimpleHypothesis_3D>(gen, owned);
    set_simple(h, name, p);
    if (p.has("max_element_volume")) {
      h->SetMaxElementVolume(positive(p, name.c_str(), "max_element_volume"));
    } else {
      h->LengthFromFaces();
    }
    return h;
  }
  return nullptr;
}

}  // namespace mesher
}  // namespace pysmesh
