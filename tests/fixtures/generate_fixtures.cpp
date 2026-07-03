// Deterministic fixture generator for pySMESH tests (dev-only; never run by CI).
//
// Emits the committed BREP shapes AND classified surface meshes (as .npy files) that the
// viscous-layer tests inject. Rebuild with:
//
//   cmake -S . -B build -DPYSMESH_BUILD_FIXTURE_GEN=ON  ...(usual flags)
//   cmake --build build --target generate_fixtures
//   ./build/generate_fixtures tests/fixtures
//
// Outputs (in <out_dir>):
//   box.brep       axis-aligned cube, edge BOX_EDGE, min corner at origin (6 faces).
//   cylinder.brep  radius CYL_RADIUS, height CYL_HEIGHT, axis +Z.
//   box_mesh/*.npy structured conformal surface mesh of box.brep with per-node CAD
//                  classification (see write_structured_mesh). Node ids in the .npy are
//                  0-based fixture-local indices; the test maps them to SMESH ids via the
//                  array returned by Mesh.add_nodes. face/edge/vertex ids are pySMESH's
//                  1-based TopExp ordinals (same TopExp::MapShapes filter as ShapeData), so
//                  they line up with Shape.faces()/edges()/vertices().
//
// The structured-grid mesher samples each face uniformly in its (u, v) parametric domain,
// so xyz is derived from S->Value(u, v) and the stored uv is exact. For the axis-aligned
// planar box, uv-uniform sampling is xyz-uniform, so both faces adjacent to a shared edge
// land the same physical points and weld by coordinate. This is NOT general (curved shapes
// need a conformal mesher); it is exact for the box and cylinder-cap style planar geometry.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <BRepBndLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepTools.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <Geom_Curve.hxx>
#include <Geom_Surface.hxx>
#include <Poly_PolygonOnTriangulation.hxx>
#include <Poly_Triangulation.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <TColStd_HArray1OfReal.hxx>
#include <TopAbs.hxx>
#include <TopExp.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Trsf.hxx>

namespace {
constexpr double BOX_EDGE = 2.0;
constexpr double CYL_RADIUS = 1.0;
constexpr double CYL_HEIGHT = 3.0;
constexpr double SPHERE_RADIUS = 1.0;
constexpr int BOX_GRID_N = 4;              // cells per face edge -> 2*N*N tris/face
constexpr double SPHERE_DEFLECTION = 0.08;  // BRepMesh chord tolerance for the sphere

// ---- minimal .npy writer (little-endian, C-order) --------------------------------- //
void write_npy(const std::string& path, const char* descr, const std::vector<int>& dims,
               const void* data, std::size_t nbytes) {
  std::string shape = "(";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    shape += std::to_string(dims[i]);
    if (i + 1 < dims.size() || dims.size() == 1) {
      shape += ", ";
    }
  }
  shape += ")";
  std::string hdr = "{'descr': '" + std::string(descr) +
                    "', 'fortran_order': False, 'shape': " + shape + ", }";
  std::size_t total = 10 + hdr.size() + 1;  // magic(6)+ver(2)+len(2) + header + '\n'
  std::size_t pad = (64 - total % 64) % 64;
  hdr.append(pad, ' ');
  hdr.push_back('\n');
  const std::uint16_t hlen = static_cast<std::uint16_t>(hdr.size());

  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  std::fwrite(&hlen, sizeof(hlen), 1, f);
  std::fwrite(hdr.data(), 1, hdr.size(), f);
  std::fwrite(data, 1, nbytes, f);
  std::fclose(f);
}

void write_f64(const std::string& path, const std::vector<double>& v, int rows, int cols) {
  std::vector<int> dims = cols > 0 ? std::vector<int>{rows, cols} : std::vector<int>{rows};
  write_npy(path, "<f8", dims, v.data(), v.size() * sizeof(double));
}
void write_i64(const std::string& path, const std::vector<std::int64_t>& v, int rows,
               int cols) {
  std::vector<int> dims = cols > 0 ? std::vector<int>{rows, cols} : std::vector<int>{rows};
  write_npy(path, "<i8", dims, v.data(), v.size() * sizeof(std::int64_t));
}

// ---- structured surface mesh with CAD classification ------------------------------ //
struct Classif {
  int kind = -1;  // 0 face, 1 edge, 2 vertex
  int id = 0;     // 1-based TopExp ordinal
  double a = 0.0;  // face: u ; edge: t
  double b = 0.0;  // face: v
};

int find_or_add(std::vector<gp_Pnt>& nodes, const gp_Pnt& p, double tol) {
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].Distance(p) <= tol) {
      return static_cast<int>(i);
    }
  }
  nodes.push_back(p);
  return static_cast<int>(nodes.size() - 1);
}

// Flatten the welded nodes + classification + connectivity into the committed .npy set.
void emit_mesh(const std::string& dir, const std::vector<gp_Pnt>& nodes,
               std::vector<Classif>& cls, const std::vector<std::int64_t>& tris,
               const std::vector<std::int64_t>& tri_face,
               const std::vector<std::int64_t>& segs,
               const std::vector<std::int64_t>& seg_edge, int nf, int ne, int nv) {
  const int P = static_cast<int>(nodes.size());
  cls.resize(P);

  std::vector<double> coords;
  coords.reserve(P * 3);
  for (const gp_Pnt& p : nodes) {
    coords.push_back(p.X());
    coords.push_back(p.Y());
    coords.push_back(p.Z());
  }
  std::vector<std::int64_t> f_nid, f_id, e_nid, e_id, v_nid, v_id;
  std::vector<double> f_uv, e_t;
  for (int g = 0; g < P; ++g) {
    const Classif& c = cls[g];
    if (c.kind == 0) {
      f_nid.push_back(g);
      f_id.push_back(c.id);
      f_uv.push_back(c.a);
      f_uv.push_back(c.b);
    } else if (c.kind == 1) {
      e_nid.push_back(g);
      e_id.push_back(c.id);
      e_t.push_back(c.a);
    } else if (c.kind == 2) {
      v_nid.push_back(g);
      v_id.push_back(c.id);
    }
  }

  write_f64(dir + "/nodes.npy", coords, P, 3);
  write_i64(dir + "/tris.npy", tris, static_cast<int>(tris.size() / 3), 3);
  write_i64(dir + "/tri_face_ids.npy", tri_face, static_cast<int>(tri_face.size()), 0);
  write_i64(dir + "/segments.npy", segs, static_cast<int>(segs.size() / 2), 2);
  write_i64(dir + "/segment_edge_ids.npy", seg_edge, static_cast<int>(seg_edge.size()), 0);
  write_i64(dir + "/face_node_ids.npy", f_nid, static_cast<int>(f_nid.size()), 0);
  write_i64(dir + "/face_ids.npy", f_id, static_cast<int>(f_id.size()), 0);
  write_f64(dir + "/face_uv.npy", f_uv, static_cast<int>(f_nid.size()), 2);
  write_i64(dir + "/edge_node_ids.npy", e_nid, static_cast<int>(e_nid.size()), 0);
  write_i64(dir + "/edge_ids.npy", e_id, static_cast<int>(e_id.size()), 0);
  write_f64(dir + "/edge_t.npy", e_t, static_cast<int>(e_t.size()), 0);
  write_i64(dir + "/vertex_node_ids.npy", v_nid, static_cast<int>(v_nid.size()), 0);
  write_i64(dir + "/vertex_ids.npy", v_id, static_cast<int>(v_id.size()), 0);

  std::printf("  %d nodes, %d tris, faces=%d edges=%d verts=%d (on %d/%d/%d cad entities)\n",
              P, static_cast<int>(tris.size() / 3), static_cast<int>(f_nid.size()),
              static_cast<int>(e_nid.size()), static_cast<int>(v_nid.size()), nf, ne, nv);
}

void write_structured_mesh(const TopoDS_Shape& shape, int n, const std::string& dir) {
  TopTools_IndexedMapOfShape fmap, emap, vmap;
  TopExp::MapShapes(shape, TopAbs_FACE, fmap);
  TopExp::MapShapes(shape, TopAbs_EDGE, emap);
  TopExp::MapShapes(shape, TopAbs_VERTEX, vmap);

  Bnd_Box bb;
  BRepBndLib::Add(shape, bb);
  double xmin, ymin, zmin, xmax, ymax, zmax;
  bb.Get(xmin, ymin, zmin, xmax, ymax, zmax);
  const double diag = gp_Pnt(xmin, ymin, zmin).Distance(gp_Pnt(xmax, ymax, zmax));
  const double tol = 1.0e-7 * diag;

  std::vector<gp_Pnt> nodes;
  std::vector<Classif> cls;
  std::vector<std::int64_t> tris;  // flattened (M,3) global ids
  std::vector<std::int64_t> tri_face;
  std::vector<std::int64_t> segs;  // flattened (S,2) global ids along edges
  std::vector<std::int64_t> seg_edge;

  auto ensure_cls = [&](int g) {
    if (static_cast<int>(cls.size()) <= g) {
      cls.resize(g + 1);
    }
  };

  // Face pass: sample each face on an (n+1)x(n+1) uv grid; triangulate; tentative face cls.
  for (int fi = 1; fi <= fmap.Extent(); ++fi) {
    const TopoDS_Face& face = TopoDS::Face(fmap.FindKey(fi));
    Handle(Geom_Surface) surf = BRep_Tool::Surface(face);
    double umin, umax, vmin, vmax;
    BRepTools::UVBounds(face, umin, umax, vmin, vmax);
    const bool reversed = face.Orientation() == TopAbs_REVERSED;

    std::vector<std::vector<int>> grid(n + 1, std::vector<int>(n + 1, -1));
    for (int i = 0; i <= n; ++i) {
      for (int j = 0; j <= n; ++j) {
        const double u = umin + (umax - umin) * i / n;
        const double v = vmin + (vmax - vmin) * j / n;
        const gp_Pnt p = surf->Value(u, v);
        const int g = find_or_add(nodes, p, tol);
        ensure_cls(g);
        cls[g] = Classif{0, fi, u, v};
        grid[i][j] = g;
      }
    }
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        const int a = grid[i][j], b = grid[i + 1][j], c = grid[i + 1][j + 1],
                  d = grid[i][j + 1];
        if (!reversed) {
          tris.insert(tris.end(), {a, b, c});
          tris.insert(tris.end(), {a, c, d});
        } else {
          tris.insert(tris.end(), {a, c, b});
          tris.insert(tris.end(), {a, d, c});
        }
        tri_face.push_back(fi);
        tri_face.push_back(fi);
      }
    }
  }

  // Edge pass: sample each edge curve uniformly; override classification to edge.
  for (int ei = 1; ei <= emap.Extent(); ++ei) {
    const TopoDS_Edge& edge = TopoDS::Edge(emap.FindKey(ei));
    double t0, t1;
    Handle(Geom_Curve) curve = BRep_Tool::Curve(edge, t0, t1);
    if (curve.IsNull()) {
      continue;
    }
    int prev = -1;
    for (int k = 0; k <= n; ++k) {
      const double t = t0 + (t1 - t0) * k / n;
      const gp_Pnt p = curve->Value(t);
      const int g = find_or_add(nodes, p, tol);  // must already exist for the box
      ensure_cls(g);
      cls[g] = Classif{1, ei, t, 0.0};
      // 1-D segments chain the ordered samples (endpoints are re-tagged to vertices below,
      // but the segment topology on the edge submesh is what viscous layers require).
      if (prev >= 0) {
        segs.push_back(prev);
        segs.push_back(g);
        seg_edge.push_back(ei);
      }
      prev = g;
    }
  }

  // Vertex pass: exact corner points override to vertex classification.
  for (int vi = 1; vi <= vmap.Extent(); ++vi) {
    const TopoDS_Vertex& vtx = TopoDS::Vertex(vmap.FindKey(vi));
    const gp_Pnt p = BRep_Tool::Pnt(vtx);
    const int g = find_or_add(nodes, p, tol);
    ensure_cls(g);
    cls[g] = Classif{2, vi, 0.0, 0.0};
  }

  emit_mesh(dir, nodes, cls, tris, tri_face, segs, seg_edge, fmap.Extent(), emap.Extent(),
           vmap.Extent());
}

// ---- conformal surface mesh via OCCT BRepMesh (for curved shapes) ------------------ //
// BRepMesh gives a watertight per-face triangulation sharing edge discretization, so a
// coordinate weld reconstructs a conformal mesh. UV comes from the triangulation's UVNodes
// (exact); edge nodes/params from Poly_PolygonOnTriangulation; corners from vertex points.
// Unlike the structured grid, this refines curved faces (planar faces stay coarse, which is
// why the box uses the structured mesher instead).
void write_brepmesh_mesh(const TopoDS_Shape& shape, double deflection,
                         const std::string& dir) {
  BRepMesh_IncrementalMesh mesher(shape, deflection, Standard_False, 0.5, Standard_True);
  mesher.Perform();

  TopTools_IndexedMapOfShape fmap, emap, vmap;
  TopExp::MapShapes(shape, TopAbs_FACE, fmap);
  TopExp::MapShapes(shape, TopAbs_EDGE, emap);
  TopExp::MapShapes(shape, TopAbs_VERTEX, vmap);
  TopTools_IndexedDataMapOfShapeListOfShape edgeFaces;
  TopExp::MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edgeFaces);

  Bnd_Box bb;
  BRepBndLib::Add(shape, bb);
  double xmin, ymin, zmin, xmax, ymax, zmax;
  bb.Get(xmin, ymin, zmin, xmax, ymax, zmax);
  const double tol = 1.0e-6 * gp_Pnt(xmin, ymin, zmin).Distance(gp_Pnt(xmax, ymax, zmax));

  std::vector<gp_Pnt> nodes;
  std::vector<Classif> cls;
  std::vector<std::int64_t> tris, tri_face, segs, seg_edge;
  std::vector<Handle(Poly_Triangulation)> faceTri(fmap.Extent() + 1);
  std::vector<TopLoc_Location> faceLoc(fmap.Extent() + 1);
  std::vector<std::vector<int>> faceL2G(fmap.Extent() + 1);
  auto ensure_cls = [&](int g) {
    if (static_cast<int>(cls.size()) <= g) {
      cls.resize(g + 1);
    }
  };

  for (int fi = 1; fi <= fmap.Extent(); ++fi) {
    const TopoDS_Face& face = TopoDS::Face(fmap.FindKey(fi));
    TopLoc_Location loc;
    Handle(Poly_Triangulation) T = BRep_Tool::Triangulation(face, loc);
    if (T.IsNull()) {
      continue;
    }
    const gp_Trsf trsf = loc.Transformation();
    const bool hasUV = T->HasUVNodes();
    const bool rev = face.Orientation() == TopAbs_REVERSED;
    std::vector<int> l2g(T->NbNodes() + 1, -1);
    for (int i = 1; i <= T->NbNodes(); ++i) {
      gp_Pnt p = T->Node(i);
      p.Transform(trsf);
      const int g = find_or_add(nodes, p, tol);
      l2g[i] = g;
      ensure_cls(g);
      double u = 0.0, v = 0.0;
      if (hasUV) {
        const gp_Pnt2d q = T->UVNode(i);
        u = q.X();
        v = q.Y();
      }
      cls[g] = Classif{0, fi, u, v};
    }
    for (int k = 1; k <= T->NbTriangles(); ++k) {
      int a = 0, b = 0, c = 0;
      T->Triangle(k).Get(a, b, c);
      int ga = l2g[a], gb = l2g[b], gc = l2g[c];
      if (rev) {
        std::swap(gb, gc);
      }
      tris.insert(tris.end(), {ga, gb, gc});
      tri_face.push_back(fi);
    }
    faceTri[fi] = T;
    faceLoc[fi] = loc;
    faceL2G[fi] = l2g;
  }

  for (int ei = 1; ei <= emap.Extent(); ++ei) {
    const TopoDS_Edge& edge = TopoDS::Edge(emap.FindKey(ei));
    Handle(Poly_PolygonOnTriangulation) poly;
    int usedFace = 0;
    const TopTools_ListOfShape& adj = edgeFaces.FindFromKey(edge);
    for (TopTools_ListIteratorOfListOfShape it(adj); it.More(); it.Next()) {
      const int fj = fmap.FindIndex(it.Value());
      if (fj < 1 || faceTri[fj].IsNull()) {
        continue;
      }
      Handle(Poly_PolygonOnTriangulation) pp =
          BRep_Tool::PolygonOnTriangulation(edge, faceTri[fj], faceLoc[fj]);
      if (!pp.IsNull()) {
        poly = pp;
        usedFace = fj;
        break;
      }
    }
    if (poly.IsNull()) {
      continue;
    }
    const TColStd_Array1OfInteger& idx = poly->Nodes();
    const Handle(TColStd_HArray1OfReal) par = poly->Parameters();
    int prev = -1;
    for (int kk = idx.Lower(); kk <= idx.Upper(); ++kk) {
      const int g = faceL2G[usedFace][idx.Value(kk)];
      ensure_cls(g);
      cls[g] = Classif{1, ei, par.IsNull() ? 0.0 : par->Value(kk), 0.0};
      if (prev >= 0 && prev != g) {
        segs.push_back(prev);
        segs.push_back(g);
        seg_edge.push_back(ei);
      }
      prev = g;
    }
  }

  for (int vi = 1; vi <= vmap.Extent(); ++vi) {
    const gp_Pnt p = BRep_Tool::Pnt(TopoDS::Vertex(vmap.FindKey(vi)));
    const int g = find_or_add(nodes, p, tol);
    ensure_cls(g);
    cls[g] = Classif{2, vi, 0.0, 0.0};
  }

  emit_mesh(dir, nodes, cls, tris, tri_face, segs, seg_edge, fmap.Extent(), emap.Extent(),
           vmap.Extent());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string out_dir = (argc > 1) ? argv[1] : ".";

  const TopoDS_Shape box = BRepPrimAPI_MakeBox(BOX_EDGE, BOX_EDGE, BOX_EDGE).Shape();
  if (!BRepTools::Write(box, (out_dir + "/box.brep").c_str())) {
    std::fprintf(stderr, "failed to write box.brep\n");
    return 1;
  }
  const TopoDS_Shape cyl = BRepPrimAPI_MakeCylinder(CYL_RADIUS, CYL_HEIGHT).Shape();
  if (!BRepTools::Write(cyl, (out_dir + "/cylinder.brep").c_str())) {
    std::fprintf(stderr, "failed to write cylinder.brep\n");
    return 1;
  }
  const TopoDS_Shape sphere = BRepPrimAPI_MakeSphere(SPHERE_RADIUS).Shape();
  if (!BRepTools::Write(sphere, (out_dir + "/sphere.brep").c_str())) {
    std::fprintf(stderr, "failed to write sphere.brep\n");
    return 1;
  }

  std::printf("box_mesh:\n");
  write_structured_mesh(box, BOX_GRID_N, out_dir + "/box_mesh");
  std::printf("sphere_mesh:\n");
  write_brepmesh_mesh(sphere, SPHERE_DEFLECTION, out_dir + "/sphere_mesh");

  std::printf("wrote box.brep, cylinder.brep, sphere.brep, box_mesh + sphere_mesh to %s\n",
              out_dir.c_str());
  return 0;
}
