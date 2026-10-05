// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-05

// pySMESH v2 capability probe — netgen 6.2.2101 and NETGENPlugin V9_16_0, linked
// statically, meshing OCCT 8.0.1 shapes: through the nglib OCC API (section NG3), and through
// SMESH with the NETGEN_2D3D algorithm of the plugin (section NP2).
//
// The checks are the mesh oracles of the netgen tests, on shapes whose answer is known:
//
//   1. Conformity: every face of a tetrahedron belongs to one or two tetrahedra; the faces
//      that belong to one are exactly the surface triangles; every point is a corner of a
//      tetrahedron; no two points coincide.
//   2. Validity: every tetrahedron has a positive volume in netgen's own orientation
//      (meshtype.cpp, Element::Volume: -((p2-p1) x (p3-p1)) . (p4-p1) / 6), which
//      NETGENPlugin keeps when it builds the SMDS tetrahedra. The smallest dihedral angle is
//      printed; the floor of 5 degrees only rules out slivers.
//   3. Measure: the sum of the volumes equals the CAD volume. A box has planar faces, so the
//      sum is exact to 1e-9 relative. A cylinder's boundary points lie on the surface, so the
//      mesh is inscribed: its volume is at most the CAD volume, and the missing volume lies
//      between the lateral facets and the surface. A facet whose edges are at most L long
//      stays within the sagitta L^2 / (8 R) of the surface, so the deficit is at most the
//      lateral area 2 pi R H times L^2 / (8 R) = pi H L^2 / 4, with L the longest surface
//      edge.
//
// nglib reads geometry from a file only, so each shape goes through a BREP file in the
// temp directory, removed after the load. The plugin reads the shape from SMESH.

#include "probe.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepTools.hxx>
#include <TopoDS_Shape.hxx>

#include <SMDS_MeshElement.hxx>
#include <SMDS_MeshNode.hxx>
#include <SMESHDS_Mesh.hxx>
#include <SMESH_Gen.hxx>
#include <SMESH_Mesh.hxx>

#include <NETGENPlugin_Hypothesis.hxx>
#include <NETGENPlugin_NETGEN_2D3D.hxx>

#include <meshing.hpp>

namespace nglib {
#include <nglib.h>
}

namespace {

using probe::check;
using probe::check_close;
using probe::section;

using Point = std::array<double, 3>;
using Face = std::array<int, 3>;

constexpr double kPi = 3.14159265358979323846;

Point sub(const Point& a, const Point& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Point cross(const Point& a, const Point& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(const Point& a) { return std::sqrt(dot(a, a)); }

Face sorted_face(int a, int b, int c) {
  Face f{a, b, c};
  std::sort(f.begin(), f.end());
  return f;
}

// The interior dihedral angle, in degrees, along edge (i, j) between the faces (i, j, k)
// and (i, j, l).
double dihedral(const Point& pi, const Point& pj, const Point& pk, const Point& pl) {
  const Point e = sub(pj, pi);
  const double ee = dot(e, e);
  Point u = sub(pk, pi);
  Point w = sub(pl, pi);
  const double su = dot(u, e) / ee;
  const double sw = dot(w, e) / ee;
  for (int c = 0; c < 3; ++c) {
    u[c] -= su * e[c];
    w[c] -= sw * e[c];
  }
  const double cosine = std::clamp(dot(u, w) / (norm(u) * norm(w)), -1.0, 1.0);
  return std::acos(cosine) * 180.0 / kPi;
}

struct NetgenMesh {
  bool generated = false;
  std::string how;                     // the results of the mesher calls, for the log
  std::vector<Point> points;           // 0-based copy of netgen's 1-based points
  std::vector<std::array<int, 4>> tets;  // 0-based point indices
  std::vector<Face> triangles;         // 0-based point indices
  int non_tets = 0;
  int non_triangles = 0;
};

NetgenMesh mesh_with_nglib(const TopoDS_Shape& shape, const std::string& name, double maxh) {
  NetgenMesh out;
  const std::filesystem::path brep =
      std::filesystem::temp_directory_path() / ("pysmesh_probe_netgen_" + name + ".brep");
  BRepTools::Write(shape, brep.string().c_str());

  nglib::Ng_Init();
  nglib::Ng_OCC_Geometry* geo = nglib::Ng_OCC_Load_BREP(brep.string().c_str());
  std::error_code ignored;
  std::filesystem::remove(brep, ignored);
  if (geo == nullptr) {
    nglib::Ng_Exit();
    return out;
  }
  nglib::Ng_Mesh* mesh = nglib::Ng_NewMesh();
  nglib::Ng_Meshing_Parameters mp;
  mp.maxh = maxh;
  const int r1 = nglib::Ng_OCC_SetLocalMeshSize(geo, mesh, &mp);
  const int r2 = nglib::Ng_OCC_GenerateEdgeMesh(geo, mesh, &mp);
  const int r3 = nglib::Ng_OCC_GenerateSurfaceMesh(geo, mesh, &mp);
  const int r4 = nglib::Ng_GenerateVolumeMesh(mesh, &mp);
  out.generated = r1 == nglib::NG_OK && r2 == nglib::NG_OK && r3 == nglib::NG_OK &&
                  r4 == nglib::NG_OK;
  out.how = "nglib results " + std::to_string(r1) + " " + std::to_string(r2) + " " +
            std::to_string(r3) + " " + std::to_string(r4);

  const int np = nglib::Ng_GetNP(mesh);
  for (int i = 1; i <= np; ++i) {
    Point p{};
    nglib::Ng_GetPoint(mesh, i, p.data());
    out.points.push_back(p);
  }
  int pi[20];
  for (int i = 1; i <= nglib::Ng_GetNE(mesh); ++i) {
    if (nglib::Ng_GetVolumeElement(mesh, i, pi) != nglib::NG_TET) {
      ++out.non_tets;
      continue;
    }
    out.tets.push_back({pi[0] - 1, pi[1] - 1, pi[2] - 1, pi[3] - 1});
  }
  for (int i = 1; i <= nglib::Ng_GetNSE(mesh); ++i) {
    if (nglib::Ng_GetSurfaceElement(mesh, i, pi) != nglib::NG_TRIG) {
      ++out.non_triangles;
      continue;
    }
    out.triangles.push_back({pi[0] - 1, pi[1] - 1, pi[2] - 1});
  }
  nglib::Ng_DeleteMesh(mesh);
  nglib::Ng_OCC_DeleteGeometry(geo);
  nglib::Ng_Exit();
  return out;
}

// The same shape meshed by NETGENPlugin's NETGEN_2D3D inside SMESH, with a
// NETGEN_Parameters hypothesis of the given max size, read back from SMESHDS. The teardown
// order is the one src/bindings/mesh.cpp established: the mesh, the generator, then the
// hypotheses.
NetgenMesh mesh_with_plugin(const TopoDS_Shape& shape, double max_size) {
  NetgenMesh out;
  auto gen = std::make_unique<SMESH_Gen>();
  SMESH_Mesh* mesh = gen->CreateMesh(false);
  mesh->ShapeToMesh(shape);
  std::vector<std::unique_ptr<SMESH_Hypothesis>> owned;
  auto* algo = new NETGENPlugin_NETGEN_2D3D(gen->GetANewId(), gen.get());
  owned.emplace_back(algo);
  auto* hyp = new NETGENPlugin_Hypothesis(gen->GetANewId(), gen.get());
  owned.emplace_back(hyp);
  hyp->SetMaxSize(max_size);
  const SMESH_Hypothesis::Hypothesis_Status s1 = mesh->AddHypothesis(shape, algo->GetID());
  const SMESH_Hypothesis::Hypothesis_Status s2 = mesh->AddHypothesis(shape, hyp->GetID());
  const bool assigned =
      !SMESH_Hypothesis::IsStatusFatal(s1) && !SMESH_Hypothesis::IsStatusFatal(s2);
  out.generated = assigned && gen->Compute(*mesh, shape);
  out.how = "assign status " + std::to_string(int(s1)) + " " + std::to_string(int(s2)) +
            ", compute " + (out.generated ? "true" : "false");

  SMESHDS_Mesh* ds = mesh->GetMeshDS();
  std::map<smIdType, int> row;
  for (SMDS_NodeIteratorPtr it = ds->nodesIterator(); it->more();) {
    const SMDS_MeshNode* n = it->next();
    row[n->GetID()] = static_cast<int>(out.points.size());
    out.points.push_back({n->X(), n->Y(), n->Z()});
  }
  for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Volume); it->more();) {
    const SMDS_MeshElement* e = it->next();
    if (e->GetEntityType() != SMDSEntity_Tetra) {
      ++out.non_tets;
      continue;
    }
    out.tets.push_back({row[e->GetNode(0)->GetID()], row[e->GetNode(1)->GetID()],
                        row[e->GetNode(2)->GetID()], row[e->GetNode(3)->GetID()]});
  }
  for (SMDS_ElemIteratorPtr it = ds->elementsIterator(SMDSAbs_Face); it->more();) {
    const SMDS_MeshElement* e = it->next();
    if (e->GetEntityType() != SMDSEntity_Triangle) {
      ++out.non_triangles;
      continue;
    }
    out.triangles.push_back({row[e->GetNode(0)->GetID()], row[e->GetNode(1)->GetID()],
                             row[e->GetNode(2)->GetID()]});
  }
  delete mesh;
  gen.reset();
  owned.clear();
  return out;
}

// Oracles 1 to 3 on one mesh. `exact_volume` is the CAD volume. `cylinder_height` 0 asks for
// the planar bound (1e-9 relative); a positive height asks for the inscribed-cylinder bound
// of the file comment.
void check_mesh(const NetgenMesh& m, const std::string& id, double exact_volume,
                double cylinder_height) {
  char msg[512];
  std::snprintf(msg, sizeof(msg),
                "%s: meshed (%s; %zu points, %zu tetrahedra, %zu surface triangles)",
                id.c_str(), m.how.c_str(), m.points.size(), m.tets.size(),
                m.triangles.size());
  check(m.generated && !m.tets.empty() && !m.triangles.empty(), msg);
  if (m.tets.empty()) {
    return;
  }
  check(m.non_tets == 0 && m.non_triangles == 0,
        id + ": every volume element is a tetrahedron and every surface element a triangle");

  // 1. Conformity.
  std::map<Face, int> uses;
  std::vector<int> corner_of(m.points.size(), 0);
  for (const auto& t : m.tets) {
    ++uses[sorted_face(t[0], t[1], t[2])];
    ++uses[sorted_face(t[0], t[1], t[3])];
    ++uses[sorted_face(t[0], t[2], t[3])];
    ++uses[sorted_face(t[1], t[2], t[3])];
    for (const int p : t) {
      ++corner_of[static_cast<std::size_t>(p)];
    }
  }
  std::set<Face> boundary;
  int overused = 0;
  for (const auto& entry : uses) {
    if (entry.second == 1) {
      boundary.insert(entry.first);
    } else if (entry.second != 2) {
      ++overused;
    }
  }
  std::set<Face> surface;
  for (const Face& f : m.triangles) {
    surface.insert(sorted_face(f[0], f[1], f[2]));
  }
  std::snprintf(msg, sizeof(msg),
                "%s oracle 1: every tetrahedron face lies on 1 or 2 tetrahedra (%d on more), "
                "and the faces on one are the %zu surface triangles (got %zu)",
                id.c_str(), overused, surface.size(), boundary.size());
  check(overused == 0 && boundary == surface, msg);
  const auto unused = std::count(corner_of.begin(), corner_of.end(), 0);
  std::snprintf(msg, sizeof(msg), "%s oracle 1: every point is a tetrahedron corner (%td free)",
                id.c_str(), unused);
  check(unused == 0, msg);
  Point lo = m.points.front();
  Point hi = m.points.front();
  for (const Point& p : m.points) {
    for (int c = 0; c < 3; ++c) {
      lo[c] = std::min(lo[c], p[c]);
      hi[c] = std::max(hi[c], p[c]);
    }
  }
  const double tol = 1e-9 * norm(sub(hi, lo));
  int duplicates = 0;
  for (std::size_t i = 0; i < m.points.size(); ++i) {
    for (std::size_t j = i + 1; j < m.points.size(); ++j) {
      duplicates += norm(sub(m.points[i], m.points[j])) <= tol ? 1 : 0;
    }
  }
  std::snprintf(msg, sizeof(msg), "%s oracle 1: no two points within %.3g (%d pairs)",
                id.c_str(), tol, duplicates);
  check(duplicates == 0, msg);

  // 2. Validity, and 3. Measure.
  double total = 0.0;
  double smallest_volume = 1e300;
  double smallest_angle = 180.0;
  for (const auto& t : m.tets) {
    const Point& a = m.points[static_cast<std::size_t>(t[0])];
    const Point& b = m.points[static_cast<std::size_t>(t[1])];
    const Point& c = m.points[static_cast<std::size_t>(t[2])];
    const Point& d = m.points[static_cast<std::size_t>(t[3])];
    const double volume = -dot(cross(sub(b, a), sub(c, a)), sub(d, a)) / 6.0;
    total += volume;
    smallest_volume = std::min(smallest_volume, volume);
    smallest_angle = std::min({smallest_angle, dihedral(a, b, c, d), dihedral(a, c, b, d),
                               dihedral(a, d, b, c), dihedral(b, c, a, d),
                               dihedral(b, d, a, c), dihedral(c, d, a, b)});
  }
  std::snprintf(msg, sizeof(msg),
                "%s oracle 2: every tetrahedron has a positive netgen volume (smallest %.3g)",
                id.c_str(), smallest_volume);
  check(smallest_volume > 0.0, msg);
  std::snprintf(msg, sizeof(msg), "%s oracle 2: smallest dihedral angle %.2f deg >= 5 deg",
                id.c_str(), smallest_angle);
  check(smallest_angle >= 5.0, msg);

  if (cylinder_height <= 0.0) {
    check_close(total, exact_volume, 1e-9 * exact_volume,
                id + " oracle 3: the tetrahedra fill the planar-faced solid exactly");
    return;
  }
  double longest = 0.0;
  for (const Face& f : m.triangles) {
    for (int e = 0; e < 3; ++e) {
      const Point& p = m.points[static_cast<std::size_t>(f[e])];
      const Point& q = m.points[static_cast<std::size_t>(f[(e + 1) % 3])];
      longest = std::max(longest, norm(sub(p, q)));
    }
  }
  const double bound = kPi * cylinder_height * longest * longest / 4.0;
  const double deficit = exact_volume - total;
  std::snprintf(msg, sizeof(msg),
                "%s oracle 3: inscribed volume deficit %.6g lies in [0, pi H L^2 / 4 = %.6g] "
                "(L = %.4g, mesh %.10g, CAD %.10g)",
                id.c_str(), deficit, bound, longest, total, exact_volume);
  check(deficit >= -1e-12 * exact_volume && deficit <= bound, msg);
}

}  // namespace

void run_netgen_probe() {
  section("NG3", "netgen 6.2.2101 (static) meshes OCCT 8.0.1 shapes through nglib");
  // netgen prints its progress at importance 3 (msghandler.cpp); the probe keeps its own
  // log readable.
  netgen::printmessage_importance = 0;

  const TopoDS_Shape box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape();
  check_mesh(mesh_with_nglib(box, "box", 6.0), "NG3 box 10 x 20 x 30, maxh 6", 6000.0, 0.0);

  const double radius = 1.0;
  const double height = 3.0;
  const TopoDS_Shape cylinder = BRepPrimAPI_MakeCylinder(radius, height).Shape();
  check_mesh(mesh_with_nglib(cylinder, "cylinder", 0.3), "NG3 cylinder R 1 H 3, maxh 0.3",
             kPi * radius * radius * height, height);

  section("NP2", "NETGENPlugin V9_16_0 (static): NETGEN_2D3D meshes through SMESH");
  check_mesh(mesh_with_plugin(box, 6.0), "NP2 box 10 x 20 x 30, NETGEN_2D3D max size 6",
             6000.0, 0.0);
  check_mesh(mesh_with_plugin(cylinder, 0.3),
             "NP2 cylinder R 1 H 3, NETGEN_2D3D max size 0.3",
             kPi * radius * radius * height, height);
}
