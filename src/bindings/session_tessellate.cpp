// pySMESH binding — Session: the render mesh, and the incremental delta over it.
//
// One call produces everything a viewer needs from the live shape: triangles with per-node
// normals, a polyline per edge, a point per vertex, and the set of faces whose contribution
// differs from the previous call. All of it is indexed into one node array and labelled with
// session entity ids, so a picked triangle, the edge beside it and the solid they belong to
// are addressable in the same namespace as everything else.
//
// Three properties of the output are contract, not implementation, because nothing in the
// arrays states them and a consumer that builds a derived structure has to know:
//
//   1. The mesh is UNWELDED across faces. Each face contributes its own node range, which is
//      what gives a hard edge at a face seam and smooth shading inside a curved patch.
//   2. The two coincident nodes a shared edge produces are BITWISE equal. BRepMesh
//      discretises each edge once and both adjacent faces read that one polygon, so the
//      positions agree exactly rather than approximately — measured, not assumed. A consumer
//      welding by exact position therefore gets the seam and nothing else.
//   3. The delta is reported, not merely exploited. Re-tessellating after an operation is
//      O(faces the operation touched), but that property is unusable by a consumer holding a
//      GPU buffer or an out-of-core store unless it is told *which* faces — and recovering
//      that by diffing the arrays costs more than the tessellation saved.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

namespace pysmesh {
namespace session {

namespace {

// One face's contribution to the render arrays, resolved once so that the fill pass is a
// straight walk with no lookups.
struct FaceBlock {
  TopoDS_Face face;
  Handle(Poly_Triangulation) triangulation;
  TopLoc_Location location;
  EntityId id = 0;
  int node_begin = 0;
  int node_end = 0;
  int tri_begin = 0;
  int tri_end = 0;
  bool reversed = false;
};

// One polyline segment: a pair of node indices and the edge they belong to.
struct EdgeSegment {
  int a = 0;
  int b = 0;
  EntityId id = 0;
};

constexpr double kPi = 3.14159265358979323846;

}  // namespace

py::dict Session::tessellate(double deflection, double angle_rad, bool relative, bool parallel,
                             bool incremental) {
  OpGuard guard(in_op_);

  // OCCT throws Standard_NumericError below its own floors rather than clamping, and that
  // exception would reach the caller as an opaque runtime error naming neither the parameter
  // nor the call. Refuse here instead, with the value.
  if (!(deflection >= Precision::Confusion())) {
    throw PysmeshError(
        "Session.tessellate: deflection must be >= Precision::Confusion (1e-7) (got " +
        std::to_string(deflection) + ").");
  }
  if (!(angle_rad >= Precision::Angular()) || !(angle_rad < kPi)) {
    throw PysmeshError(
        "Session.tessellate: angle_rad must be in [Precision::Angular (1e-12), pi) rad "
        "(got " +
        std::to_string(angle_rad) + ").");
  }

  const TopoDS_Shape root = state_.root;

  // Forcing a full re-tessellation means removing the cached triangulations, and
  // BRepTools::Clean is what does that. IMeshTools_Parameters::CleanModel is NOT the
  // mechanism — it governs the mesher's own temporary data model and leaves every existing
  // Poly_Triangulation in place, measured both ways. Dropping the emission cache with
  // it keeps the two consistent: after a full re-mesh every face is genuinely new.
  if (!incremental) {
    emitted_.clear();
  }
  {
    py::gil_scoped_release release;
    if (!incremental) {
      BRepTools::Clean(root);
    }
    IMeshTools_Parameters params;
    params.Deflection = deflection;
    params.Angle = angle_rad;
    params.Relative = relative;
    params.InParallel = parallel;
    // The constructor meshes; calling Perform() again would walk the whole model a second
    // time to discover that there is nothing left to do.
    BRepMesh_IncrementalMesh mesher(root, params);
  }

  // ---- resolve the faces ------------------------------------------------------------- //
  //
  // TopExp::MapShapes order, which is the same deterministic traversal every other bulk
  // query uses, so a caller can line the rows up against entities("FACE") without a sort.

  ShapeSet face_map;
  TopExp::MapShapes(root, TopAbs_FACE, face_map);

  std::vector<FaceBlock> blocks;
  blocks.reserve(static_cast<std::size_t>(face_map.Extent()));
  int total_nodes = 0;
  int total_tris = 0;
  for (int i = 1; i <= face_map.Extent(); ++i) {
    FaceBlock b;
    b.face = TopoDS::Face(face_map.FindKey(i));
    b.triangulation = BRep_Tool::Triangulation(b.face, b.location);
    b.id = label_of("tessellate", b.face);
    b.reversed = b.face.Orientation() == TopAbs_REVERSED;
    b.node_begin = total_nodes;
    b.tri_begin = total_tris;
    // A face the mesher could not triangulate contributes no nodes and no triangles. It is
    // still listed, with an empty range, because a caller indexing by face must find every
    // face — an absent row and an empty one mean different things.
    if (!b.triangulation.IsNull()) {
      total_nodes += b.triangulation->NbNodes();
      total_tris += b.triangulation->NbTriangles();
    }
    b.node_end = total_nodes;
    b.tri_end = total_tris;
    blocks.push_back(std::move(b));
  }

  // ---- the delta --------------------------------------------------------------------- //
  //
  // Two answers, because an operation can move a face without re-triangulating it and the
  // consumer needs to tell those apart. A rebuilt face gets a new Poly_Triangulation; a
  // relocated one keeps its triangulation and moves every node the mesh emits for it.

  std::vector<EntityId> retriangulated;
  std::vector<EntityId> changed;
  std::unordered_map<const void*, EmittedFace> now;
  now.reserve(blocks.size());
  for (const FaceBlock& b : blocks) {
    const void* key = b.face.TShape().get();
    const auto it = emitted_.find(key);
    const Poly_Triangulation* before =
        (it == emitted_.end()) ? nullptr : it->second.triangulation.get();
    if (before != b.triangulation.get()) {
      retriangulated.push_back(b.id);
      changed.push_back(b.id);
    } else if (!b.location.IsEqual(it->second.location)) {
      changed.push_back(b.id);
    }
    now.emplace(key, EmittedFace{b.face, b.triangulation, b.location});
  }
  emitted_ = std::move(now);

  auto tidy = [](std::vector<EntityId>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  };
  tidy(retriangulated);
  tidy(changed);

  // ---- resolve the edge polylines ----------------------------------------------------- //
  //
  // Harvested from what BRepMesh has already computed, so there is no second discretisation
  // to pay for. An edge that bounds a face carries a Poly_PolygonOnTriangulation, whose nodes
  // are indices into that face's own triangulation — so its polyline reuses nodes the mesh
  // already has and costs nothing but the index pairs. An edge that bounds no face carries a
  // Poly_Polygon3D of real points instead, and those points are appended after every face's
  // nodes. The two are disjoint: a face's edge has no Polygon3D and a free edge has no
  // polygon on a triangulation.

  NCollection_IndexedDataMap<TopoDS_Shape, NCollection_List<TopoDS_Shape>,
                             TopTools_ShapeMapHasher>
      faces_of_edge;
  TopExp::MapShapesAndAncestors(root, TopAbs_EDGE, TopAbs_FACE, faces_of_edge);

  ShapeSet edge_map;
  TopExp::MapShapes(root, TopAbs_EDGE, edge_map);

  // Face -> its block, so an edge reaches its owning face's node offset in O(1) rather than
  // by scanning every block.
  ShapeKeyed<std::size_t> block_of_face;
  block_of_face.reserve(blocks.size());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    block_of_face.emplace(blocks[i].face, i);
  }

  std::vector<EdgeSegment> segments;
  std::vector<gp_Pnt> free_nodes;
  for (int i = 1; i <= edge_map.Extent(); ++i) {
    const TopoDS_Edge edge = TopoDS::Edge(edge_map.FindKey(i));
    const EntityId id = label_of("tessellate", edge);

    // The first owning face that carries a polygon for this edge wins. Both adjacent faces
    // read the same edge discretisation, so the choice does not change the geometry — only
    // which node range the indices point into.
    bool done = false;
    if (faces_of_edge.Contains(edge)) {
      for (const TopoDS_Shape& f : faces_of_edge.FindFromKey(edge)) {
        const auto at = block_of_face.find(f);
        if (at == block_of_face.end()) {
          continue;
        }
        const FaceBlock& b = blocks[at->second];
        if (b.triangulation.IsNull()) {
          continue;
        }
        const Handle(Poly_PolygonOnTriangulation) poly =
            BRep_Tool::PolygonOnTriangulation(edge, b.triangulation, b.location);
        if (poly.IsNull() || poly->NbNodes() < 2) {
          continue;
        }
        for (int k = 1; k < poly->NbNodes(); ++k) {
          EdgeSegment s;
          s.a = b.node_begin + poly->Node(k) - 1;
          s.b = b.node_begin + poly->Node(k + 1) - 1;
          s.id = id;
          segments.push_back(s);
        }
        done = true;
        break;
      }
    }
    if (done) {
      continue;
    }

    TopLoc_Location loc;
    const Handle(Poly_Polygon3D) poly3d = BRep_Tool::Polygon3D(edge, loc);
    if (poly3d.IsNull() || poly3d->NbNodes() < 2) {
      // No discretisation exists for this edge — a degenerate seam, or a face the mesher
      // declined. Omitted rather than faked: a caller checking coverage must be able to see
      // the gap, and a two-node polyline over an invented pair of points would hide it.
      continue;
    }
    const int first = total_nodes + static_cast<int>(free_nodes.size());
    const NCollection_Array1<gp_Pnt>& pts = poly3d->Nodes();
    for (int k = pts.Lower(); k <= pts.Upper(); ++k) {
      gp_Pnt p = pts.Value(k);
      p.Transform(loc.Transformation());
      free_nodes.push_back(p);
    }
    for (int k = 0; k + 1 < poly3d->NbNodes(); ++k) {
      EdgeSegment s;
      s.a = first + k;
      s.b = first + k + 1;
      s.id = id;
      segments.push_back(s);
    }
  }

  // ---- resolve the vertices ------------------------------------------------------------ //

  ShapeSet vertex_map;
  TopExp::MapShapes(root, TopAbs_VERTEX, vertex_map);
  std::vector<EntityId> vertex_ids;
  vertex_ids.reserve(static_cast<std::size_t>(vertex_map.Extent()));
  for (int i = 1; i <= vertex_map.Extent(); ++i) {
    vertex_ids.push_back(label_of("tessellate", vertex_map.FindKey(i)));
  }

  // ---- allocate and fill --------------------------------------------------------------- //

  const auto node_count = static_cast<py::ssize_t>(total_nodes + free_nodes.size());
  const auto tri_count = static_cast<py::ssize_t>(total_tris);
  const auto seg_count = static_cast<py::ssize_t>(segments.size());
  const auto face_count = static_cast<py::ssize_t>(blocks.size());
  const auto vert_count = static_cast<py::ssize_t>(vertex_map.Extent());

  py::array_t<double> nodes({node_count, py::ssize_t{3}});
  py::array_t<double> normals({node_count, py::ssize_t{3}});
  py::array_t<std::int32_t> tris({tri_count, py::ssize_t{3}});
  py::array_t<std::int64_t> tri_face_id(tri_count);
  py::array_t<std::int32_t> edge_lines({seg_count, py::ssize_t{2}});
  py::array_t<std::int64_t> edge_id(seg_count);
  py::array_t<double> vertex_xyz({vert_count, py::ssize_t{3}});
  py::array_t<std::int32_t> face_node_range({face_count, py::ssize_t{2}});
  py::array_t<std::int32_t> face_tri_range({face_count, py::ssize_t{2}});

  double* np_ = nodes.mutable_data();
  double* mp = normals.mutable_data();
  std::int32_t* tp = tris.mutable_data();
  std::int64_t* fp = tri_face_id.mutable_data();
  std::int32_t* ep = edge_lines.mutable_data();
  std::int64_t* eip = edge_id.mutable_data();
  double* vp = vertex_xyz.mutable_data();
  std::int32_t* nrp = face_node_range.mutable_data();
  std::int32_t* trp = face_tri_range.mutable_data();

  {
    py::gil_scoped_release release;
    std::fill(mp, mp + 3 * node_count, 0.0);

    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
      const FaceBlock& b = blocks[bi];
      nrp[2 * bi + 0] = b.node_begin;
      nrp[2 * bi + 1] = b.node_end;
      trp[2 * bi + 0] = b.tri_begin;
      trp[2 * bi + 1] = b.tri_end;
      if (b.triangulation.IsNull()) {
        continue;
      }
      const Poly_Triangulation& t = *b.triangulation;
      const gp_Trsf trsf = b.location.Transformation();

      for (int i = 1; i <= t.NbNodes(); ++i) {
        gp_Pnt p = t.Node(i);
        p.Transform(trsf);
        const int g = b.node_begin + i - 1;
        np_[3 * g + 0] = p.X();
        np_[3 * g + 1] = p.Y();
        np_[3 * g + 2] = p.Z();
      }

      // Normals come from the underlying surface at each node's own UV, not from averaging
      // the triangles: the surface answer is exact where the facet answer is an
      // approximation of it, and it costs almost nothing next to the meshing (measured at
      // 0.2 ms against 36 ms for a 5106-node sphere).
      if (t.HasUVNodes()) {
        TopLoc_Location surf_loc;
        const Handle(Geom_Surface) surf = BRep_Tool::Surface(b.face, surf_loc);
        const gp_Trsf surf_trsf = surf_loc.Transformation();
        GeomLProp_SLProps props(surf, 1, 1.0e-9);
        for (int i = 1; i <= t.NbNodes(); ++i) {
          const gp_Pnt2d uv = t.UVNode(i);
          props.SetParameters(uv.X(), uv.Y());
          if (!props.IsNormalDefined()) {
            // A degeneracy — a cone's apex, a sphere's pole. The zero vector stays, because
            // a fabricated direction there would be indistinguishable from a real one.
            continue;
          }
          gp_Dir n = props.Normal();
          if (b.reversed) {
            n.Reverse();
          }
          n.Transform(surf_trsf);
          const int g = b.node_begin + i - 1;
          mp[3 * g + 0] = n.X();
          mp[3 * g + 1] = n.Y();
          mp[3 * g + 2] = n.Z();
        }
      }

      for (int k = 1; k <= t.NbTriangles(); ++k) {
        int a = 0, bb = 0, c = 0;
        t.Triangle(k).Get(a, bb, c);
        const int ga = b.node_begin + a - 1;
        const int gb = b.node_begin + bb - 1;
        const int gc = b.node_begin + c - 1;
        const std::size_t row = static_cast<std::size_t>(b.tri_begin) +
                                static_cast<std::size_t>(k) - 1;
        tp[3 * row + 0] = static_cast<std::int32_t>(ga);
        // Swapped for a REVERSED face, so the winding agrees with the corrected outward
        // normal and a consumer can cull back faces without a per-face exception.
        tp[3 * row + 1] = static_cast<std::int32_t>(b.reversed ? gc : gb);
        tp[3 * row + 2] = static_cast<std::int32_t>(b.reversed ? gb : gc);
        fp[row] = b.id;
      }
    }

    for (std::size_t i = 0; i < free_nodes.size(); ++i) {
      const std::size_t g = static_cast<std::size_t>(total_nodes) + i;
      np_[3 * g + 0] = free_nodes[i].X();
      np_[3 * g + 1] = free_nodes[i].Y();
      np_[3 * g + 2] = free_nodes[i].Z();
    }

    for (std::size_t i = 0; i < segments.size(); ++i) {
      ep[2 * i + 0] = static_cast<std::int32_t>(segments[i].a);
      ep[2 * i + 1] = static_cast<std::int32_t>(segments[i].b);
      eip[i] = segments[i].id;
    }
  }

  for (int i = 1; i <= vertex_map.Extent(); ++i) {
    const gp_Pnt p = BRep_Tool::Pnt(TopoDS::Vertex(vertex_map.FindKey(i)));
    vp[3 * (i - 1) + 0] = p.X();
    vp[3 * (i - 1) + 1] = p.Y();
    vp[3 * (i - 1) + 2] = p.Z();
  }

  std::vector<EntityId> face_ids;
  face_ids.reserve(blocks.size());
  for (const FaceBlock& b : blocks) {
    face_ids.push_back(b.id);
  }

  py::dict out;
  out["nodes"] = nodes;
  out["normals"] = normals;
  out["tris"] = tris;
  out["tri_face_id"] = tri_face_id;
  out["edge_lines"] = edge_lines;
  out["edge_id"] = edge_id;
  out["vertex_xyz"] = vertex_xyz;
  out["vertex_id"] = ids_array(vertex_ids);
  out["face_id"] = ids_array(face_ids);
  out["face_node_range"] = face_node_range;
  out["face_tri_range"] = face_tri_range;
  out["retriangulated"] = ids_array(retriangulated);
  out["changed"] = ids_array(changed);
  return out;
}

}  // namespace session
}  // namespace pysmesh
