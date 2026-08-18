// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-06

// pySMESH binding — Session: the geometric query surface over the live shape.
//
// Everything a consumer reads from a modelling kernel — types, boxes, mass properties,
// parameter ranges, adjacency, positions, normals, curvature, projections, containment —
// answered against the session's own entity ids rather than positional ordinals.
//
// Two conventions run through the file.
//
// Bulk over per-entity. A query that covers a whole kind returns parallel arrays in
// ascending id order, because a model with tens of thousands of faces cannot afford a Python
// object per answer. A query that names its subjects returns rows in the order they were
// named, so the caller can index its own arrays alongside.
//
// No guard, and no lock. These are const: the invariant that state_ is only ever written
// while the GIL is held is what makes them safe without one. A query long enough to release
// the GIL copies the shapes it needs first, so it works on handles it owns rather than
// reading the session's live state from a released-GIL block.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

namespace pysmesh {
namespace session {

namespace {

// A face's parameter domain, sampled at cell centres.
//
// Cell centres rather than grid corners, and that is not cosmetic: a corner sample lands on
// the seam of a closed surface and on the poles of a sphere, where the curvature is either
// undefined or defined twice. Cell centres never touch the boundary of the domain, so an
// n = 1 grid is exactly the single centre sample this operation exists to improve on.
double cell_centre(double lo, double hi, int i, int n) {
  return lo + (hi - lo) * (static_cast<double>(i) + 0.5) / static_cast<double>(n);
}

// "This surface does not define that parameter." Not 0.0: a filter reading `radius1 < 1.0`
// treats 0.0 as a very small radius and picks up every plane in the model.
constexpr double kUndefined = std::numeric_limits<double>::quiet_NaN();

// Write one gp_Ax3 into the origin / axis / ref_dir rows of a surface-parameter table.
void write_frame(const gp_Ax3& frame, double* origin, double* axis, double* ref_dir) {
  const gp_Pnt& o = frame.Location();
  const gp_Dir& d = frame.Direction();
  const gp_Dir& x = frame.XDirection();
  origin[0] = o.X();
  origin[1] = o.Y();
  origin[2] = o.Z();
  axis[0] = d.X();
  axis[1] = d.Y();
  axis[2] = d.Z();
  ref_dir[0] = x.X();
  ref_dir[1] = x.Y();
  ref_dir[2] = x.Z();
}

}  // namespace

py::dict Session::entity_types(const std::string& kind) const {
  const TopAbs_ShapeEnum k = kind_from_name(kind);
  const std::vector<EntityId> ids = ids_of_kind(k);

  py::list types;
  for (EntityId id : ids) {
    const EntityRecord& rec = state_.registry->alive.at(id);
    // A split entity is typed by its first shape. The pieces of a split share the underlying
    // geometry they were cut from, so the type is the same for all of them.
    const TopoDS_Shape& s = rec.shapes.front();
    switch (k) {
      case TopAbs_FACE:
        types.append(
            py::str(surface_type_name(BRepAdaptor_Surface(TopoDS::Face(s)).GetType())));
        break;
      case TopAbs_EDGE:
        types.append(py::str(curve_type_name(BRepAdaptor_Curve(TopoDS::Edge(s)).GetType())));
        break;
      default:
        types.append(py::str(kind_name(k)));
        break;
    }
  }

  py::dict out;
  out["ids"] = ids_array(ids);
  out["types"] = types;
  return out;
}

py::dict Session::surface_parameters(const std::vector<EntityId>& face_ids) const {
  const auto n = static_cast<py::ssize_t>(face_ids.size());
  const auto three = static_cast<py::ssize_t>(3);
  py::array_t<double> origin({n, three});
  py::array_t<double> axis({n, three});
  py::array_t<double> ref_dir({n, three});
  py::array_t<double> radius1(n);
  py::array_t<double> radius2(n);
  py::array_t<double> half_angle(n);
  py::array_t<bool> reversed(n);

  double* op = origin.mutable_data();
  double* ap = axis.mutable_data();
  double* xp = ref_dir.mutable_data();
  double* r1 = radius1.mutable_data();
  double* r2 = radius2.mutable_data();
  double* ha = half_angle.mutable_data();
  bool* rv = reversed.mutable_data();

  // Every cell starts undefined and only the ones the surface type actually defines are
  // written. The default has to be NaN rather than zero — see kUndefined.
  std::fill(op, op + 3 * n, kUndefined);
  std::fill(ap, ap + 3 * n, kUndefined);
  std::fill(xp, xp + 3 * n, kUndefined);
  std::fill(r1, r1 + n, kUndefined);
  std::fill(r2, r2 + n, kUndefined);
  std::fill(ha, ha + n, kUndefined);

  py::list types;
  const ShapeSet& faces_in_root = root_faces();
  for (py::ssize_t i = 0; i < n; ++i) {
    const TopoDS_Face face = sole_face("surface_parameters",
                                       face_ids[static_cast<std::size_t>(i)], faces_in_root);
    // BRepAdaptor_Surface is the transformed adaptor, so Plane()/Cylinder()/... come back in
    // model coordinates rather than in the underlying Geom_Surface's local frame.
    const BRepAdaptor_Surface surf(face);
    const GeomAbs_SurfaceType type = surf.GetType();
    types.append(py::str(surface_type_name(type)));
    rv[i] = face.Orientation() == TopAbs_REVERSED;

    double* o = op + 3 * i;
    double* a = ap + 3 * i;
    double* x = xp + 3 * i;
    switch (type) {
      case GeomAbs_Plane:
        write_frame(surf.Plane().Position(), o, a, x);
        break;
      case GeomAbs_Cylinder: {
        const gp_Cylinder c = surf.Cylinder();
        write_frame(c.Position(), o, a, x);
        r1[i] = c.Radius();
        break;
      }
      case GeomAbs_Cone: {
        const gp_Cone c = surf.Cone();
        write_frame(c.Position(), o, a, x);
        // RefRadius, not "the radius": a cone has one radius per station along its axis, and
        // this is the one at the frame's origin. SemiAngle is signed — its sign says which
        // way along the axis the cone widens, so it must not be reported as a magnitude.
        r1[i] = c.RefRadius();
        ha[i] = c.SemiAngle();
        break;
      }
      case GeomAbs_Sphere: {
        const gp_Sphere sp = surf.Sphere();
        write_frame(sp.Position(), o, a, x);
        r1[i] = sp.Radius();
        break;
      }
      case GeomAbs_Torus: {
        const gp_Torus t = surf.Torus();
        write_frame(t.Position(), o, a, x);
        r1[i] = t.MajorRadius();
        r2[i] = t.MinorRadius();
        break;
      }
      case GeomAbs_SurfaceOfRevolution: {
        // No radius — the profile curve decides that, and it varies along the axis. The axis
        // itself is well defined, and it is the half a consumer can use.
        const gp_Ax1 ax = surf.AxeOfRevolution();
        const gp_Pnt& loc = ax.Location();
        const gp_Dir& dir = ax.Direction();
        o[0] = loc.X();
        o[1] = loc.Y();
        o[2] = loc.Z();
        a[0] = dir.X();
        a[1] = dir.Y();
        a[2] = dir.Z();
        break;
      }
      case GeomAbs_SurfaceOfExtrusion: {
        // The sweep direction. There is no origin: the basis curve is anywhere along it.
        const gp_Dir dir = surf.Direction();
        a[0] = dir.X();
        a[1] = dir.Y();
        a[2] = dir.Z();
        break;
      }
      default:
        // Bezier, BSpline, Offset, Other: no analytic parameters at all. The row stays NaN.
        break;
    }
  }

  py::dict out;
  out["ids"] = ids_array(face_ids);
  out["types"] = types;
  out["origin"] = origin;
  out["axis"] = axis;
  out["ref_dir"] = ref_dir;
  out["radius1"] = radius1;
  out["radius2"] = radius2;
  out["half_angle"] = half_angle;
  out["reversed"] = reversed;
  return out;
}

py::dict Session::bounding_boxes(const std::string& kind) const {
  const std::vector<EntityId> ids = ids_of_kind(kind_from_name(kind));
  const auto n = static_cast<py::ssize_t>(ids.size());
  py::array_t<double> bbox({n, static_cast<py::ssize_t>(6)});
  double* bp = bbox.mutable_data();

  for (py::ssize_t i = 0; i < n; ++i) {
    const EntityRecord& rec = state_.registry->alive.at(ids[static_cast<std::size_t>(i)]);
    Bnd_Box box;
    for (const TopoDS_Shape& s : rec.shapes) {
      BRepBndLib::Add(s, box);
    }
    box.Get(bp[6 * i + 0], bp[6 * i + 1], bp[6 * i + 2], bp[6 * i + 3], bp[6 * i + 4],
            bp[6 * i + 5]);
  }

  py::dict out;
  out["ids"] = ids_array(ids);
  out["bbox"] = bbox;
  return out;
}

py::dict Session::mass_properties(const std::vector<EntityId>& entity_ids) const {
  const auto n = static_cast<py::ssize_t>(entity_ids.size());
  py::array_t<double> measure(n);
  py::array_t<double> centroid({n, static_cast<py::ssize_t>(3)});
  double* mp = measure.mutable_data();
  double* cp = centroid.mutable_data();

  for (py::ssize_t i = 0; i < n; ++i) {
    const EntityRecord& rec =
        require_alive("mass_properties", entity_ids[static_cast<std::size_t>(i)]);
    // Each shape is measured by its own kind — volume for a solid, area for a face, length
    // for an edge. Never by walking a parent: BRepGProp::LinearProperties on a SOLID visits
    // every edge once per owning face, so a total edge length taken that way is silently
    // doubled.
    double total = 0.0;
    double wsum = 0.0;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (const TopoDS_Shape& s : rec.shapes) {
      const double m = measure_of(s);
      const std::array<double, 3> c = centroid_of(s);
      const double w = (m > 0.0) ? m : 1.0;
      total += m;
      wsum += w;
      cx += w * c[0];
      cy += w * c[1];
      cz += w * c[2];
    }
    mp[i] = total;
    cp[3 * i + 0] = cx / wsum;
    cp[3 * i + 1] = cy / wsum;
    cp[3 * i + 2] = cz / wsum;
  }

  py::dict out;
  out["ids"] = ids_array(entity_ids);
  out["measure"] = measure;
  out["centroid"] = centroid;
  return out;
}

py::array_t<double> Session::face_parameter_bounds(
    const std::vector<EntityId>& face_ids) const {
  const auto n = static_cast<py::ssize_t>(face_ids.size());
  py::array_t<double> out({n, static_cast<py::ssize_t>(4)});
  double* p = out.mutable_data();
  const ShapeSet& faces_in_root = root_faces();
  for (py::ssize_t i = 0; i < n; ++i) {
    const BRepAdaptor_Surface s(sole_face(
        "face_parameter_bounds", face_ids[static_cast<std::size_t>(i)], faces_in_root));
    p[4 * i + 0] = s.FirstUParameter();
    p[4 * i + 1] = s.LastUParameter();
    p[4 * i + 2] = s.FirstVParameter();
    p[4 * i + 3] = s.LastVParameter();
  }
  return out;
}

py::array_t<double> Session::edge_parameter_bounds(
    const std::vector<EntityId>& edge_ids) const {
  const std::vector<TopoDS_Shape> edges = edges_of("edge_parameter_bounds", edge_ids);
  const auto n = static_cast<py::ssize_t>(edges.size());
  py::array_t<double> out({n, static_cast<py::ssize_t>(2)});
  double* p = out.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    const BRepAdaptor_Curve c(TopoDS::Edge(edges[static_cast<std::size_t>(i)]));
    p[2 * i + 0] = c.FirstParameter();
    p[2 * i + 1] = c.LastParameter();
  }
  return out;
}

py::dict Session::adjacency(const std::string& kind, const std::string& other_kind) const {
  const TopAbs_ShapeEnum a = kind_from_name(kind);
  const TopAbs_ShapeEnum b = kind_from_name(other_kind);
  if (a == b) {
    throw PysmeshError("Session.adjacency: the two kinds must differ; " + kind +
                       " has no adjacency relation to itself.");
  }

  // Which way the relation runs is decided by the dimensions, not by an argument. TopAbs
  // orders the kinds by decreasing dimension (SOLID < FACE < EDGE < VERTEX as enum values),
  // so `b > a` means the other kind is the lower-dimensional one and the relation is the
  // boundary; otherwise it is the ancestors.
  const bool towards_boundary = b > a;

  ShapeKeyed<std::vector<EntityId>> ids_of_shape;
  for (const auto& [id, rec] : state_.registry->alive) {
    if (rec.kind != a && rec.kind != b) {
      continue;
    }
    for (const TopoDS_Shape& s : rec.shapes) {
      ids_of_shape[s].push_back(id);
    }
  }

  std::vector<EntityId> left, right;
  const std::vector<EntityId> subjects = ids_of_kind(a);

  if (towards_boundary) {
    for (EntityId id : subjects) {
      const EntityRecord& rec = state_.registry->alive.at(id);
      ShapeSet parts;
      for (const TopoDS_Shape& s : rec.shapes) {
        TopExp::MapShapes(s, b, parts);
      }
      for (int i = 1; i <= parts.Extent(); ++i) {
        const auto it = ids_of_shape.find(parts.FindKey(i));
        if (it == ids_of_shape.end()) {
          continue;
        }
        for (EntityId other : it->second) {
          left.push_back(id);
          right.push_back(other);
        }
      }
    }
  } else {
    NCollection_IndexedDataMap<TopoDS_Shape, NCollection_List<TopoDS_Shape>,
                               TopTools_ShapeMapHasher>
        ancestors;
    TopExp::MapShapesAndAncestors(state_.root, a, b, ancestors);
    for (EntityId id : subjects) {
      const EntityRecord& rec = state_.registry->alive.at(id);
      for (const TopoDS_Shape& s : rec.shapes) {
        if (!ancestors.Contains(s)) {
          continue;
        }
        for (const TopoDS_Shape& up : ancestors.FindFromKey(s)) {
          const auto it = ids_of_shape.find(up);
          if (it == ids_of_shape.end()) {
            continue;
          }
          for (EntityId other : it->second) {
            left.push_back(id);
            right.push_back(other);
          }
        }
      }
    }
  }

  py::dict out;
  out["ids"] = ids_array(left);
  out["related"] = ids_array(right);
  return out;
}

py::dict Session::face_wires(const std::vector<EntityId>& face_ids) const {
  std::vector<EntityId> wire_face;
  std::vector<char> is_outer;  // char, not bool: std::vector<bool> has no writable data()
  std::vector<char> ordered;
  std::vector<std::int32_t> range;  // start, end interleaved
  std::vector<EntityId> edge_ids;

  const ShapeSet& faces_in_root = root_faces();
  for (EntityId face_id : face_ids) {
    const TopoDS_Face face = sole_face("face_wires", face_id, faces_in_root);
    const TopoDS_Wire outer = BRepTools::OuterWire(face);

    std::size_t wires_here = 0;
    std::size_t outers_here = 0;
    for (TopExp_Explorer ex(face, TopAbs_WIRE); ex.More(); ex.Next()) {
      const TopoDS_Wire wire = TopoDS::Wire(ex.Current());
      ++wires_here;

      // The wire's full edge set, orientation-insensitive: this is the count the traversal
      // has to reproduce to be trusted.
      ShapeSet all;
      TopExp::MapShapes(wire, TopAbs_EDGE, all);

      // Connection order. A seam edge belongs to its wire twice, once per orientation, so
      // the traversal visits it twice and the second visit is dropped — an id cannot carry
      // an orientation, and the same id listed twice in one loop reads as a duplicate rather
      // than as a seam.
      std::vector<TopoDS_Shape> walked;
      ShapeSet seen;
      for (BRepTools_WireExplorer we(wire, face); we.More(); we.Next()) {
        const TopoDS_Edge& e = we.Current();
        if (!seen.Contains(e)) {
          seen.Add(e);
          walked.push_back(e);
        }
      }

      const bool complete = walked.size() == static_cast<std::size_t>(all.Extent());
      const auto start = static_cast<std::int32_t>(edge_ids.size());
      if (complete) {
        for (const TopoDS_Shape& e : walked) {
          edge_ids.push_back(label_of("face_wires", e));
        }
      } else {
        // BRepTools_WireExplorer stopped early — the wire has a defect it does not walk
        // past. Emit the map's order instead, so the caller still gets every edge of the
        // loop, and say so on the row rather than shipping a loop with edges missing.
        for (int i = 1; i <= all.Extent(); ++i) {
          edge_ids.push_back(label_of("face_wires", all.FindKey(i)));
        }
      }

      const bool this_is_outer = wire.IsSame(outer);
      outers_here += this_is_outer ? 1u : 0u;
      wire_face.push_back(face_id);
      is_outer.push_back(this_is_outer ? 1 : 0);
      ordered.push_back(complete ? 1 : 0);
      range.push_back(start);
      range.push_back(static_cast<std::int32_t>(edge_ids.size()));
    }

    // A face with wires but no outer one would answer "this face has only holes", which is
    // a wrong answer rather than a partial one. Raise instead: every hole test downstream is
    // built on the outer/inner split, and a silent zero there is unrecoverable.
    if (wires_here > 0 && outers_here == 0) {
      throw PysmeshError("Session.face_wires: OCCT could not identify the outer wire of face "
                         + std::to_string(face_id) + ", which has " +
                         std::to_string(wires_here) +
                         " wires. The face's boundary is malformed; heal it before asking "
                         "which of its loops is the outer one.");
    }
  }

  const auto w = static_cast<py::ssize_t>(wire_face.size());
  py::array_t<bool> outer_arr(w);
  py::array_t<bool> ordered_arr(w);
  py::array_t<std::int32_t> range_arr({w, static_cast<py::ssize_t>(2)});
  std::copy(is_outer.begin(), is_outer.end(), outer_arr.mutable_data());
  std::copy(ordered.begin(), ordered.end(), ordered_arr.mutable_data());
  std::copy(range.begin(), range.end(), range_arr.mutable_data());

  py::dict out;
  out["face_id"] = ids_array(wire_face);
  out["is_outer"] = outer_arr;
  out["ordered"] = ordered_arr;
  out["edge_range"] = range_arr;
  out["edge_id"] = ids_array(edge_ids);
  return out;
}

py::dict Session::surface_at(EntityId face_id, const PointArray& uv) const {
  const TopoDS_Face face = sole_face("surface_at", face_id);
  const std::vector<std::pair<double, double>> params = pairs_of("surface_at", "uv", uv);

  const auto n = static_cast<py::ssize_t>(params.size());
  py::array_t<double> points({n, static_cast<py::ssize_t>(3)});
  py::array_t<double> normals({n, static_cast<py::ssize_t>(3)});
  py::array_t<bool> defined(n);
  double* pp = points.mutable_data();
  double* np = normals.mutable_data();
  bool* dp = defined.mutable_data();

  // The face's orientation, taken once. A REVERSED face's surface normal points into the
  // body, and every consumer of a normal — a boundary condition, a render shade, an offset
  // direction — means the outward one.
  const double sign = (face.Orientation() == TopAbs_REVERSED) ? -1.0 : 1.0;

  {
    py::gil_scoped_release release;
    BRepAdaptor_Surface surf(face);
    for (py::ssize_t i = 0; i < n; ++i) {
      const auto& [u, v] = params[static_cast<std::size_t>(i)];
      BRepLProp_SLProps props(surf, u, v, 1, 1e-7);
      const gp_Pnt p = surf.Value(u, v);
      pp[3 * i + 0] = p.X();
      pp[3 * i + 1] = p.Y();
      pp[3 * i + 2] = p.Z();
      // A normal is undefined where the surface degenerates — a cone's apex, a sphere's
      // pole. Reported rather than faked, because a zero-length normal that looks like a
      // direction is worse than an admitted gap.
      dp[i] = props.IsNormalDefined();
      if (dp[i]) {
        const gp_Dir dir = props.Normal();
        np[3 * i + 0] = sign * dir.X();
        np[3 * i + 1] = sign * dir.Y();
        np[3 * i + 2] = sign * dir.Z();
      } else {
        np[3 * i + 0] = np[3 * i + 1] = np[3 * i + 2] = 0.0;
      }
    }
  }

  py::dict out;
  out["points"] = points;
  out["normals"] = normals;
  out["defined"] = defined;
  return out;
}

py::dict Session::curvature(const std::vector<EntityId>& face_ids, int samples) const {
  if (face_ids.empty()) {
    throw PysmeshError("Session.curvature: face_ids must name at least one face.");
  }
  if (samples < 1) {
    throw PysmeshError("Session.curvature: samples must be >= 1 (got " +
                       std::to_string(samples) + ").");
  }

  // Resolved under the GIL, so the loop below reads shapes this call owns rather than the
  // session's live state.
  std::vector<TopoDS_Face> faces;
  faces.reserve(face_ids.size());
  const ShapeSet& faces_in_root = root_faces();
  for (EntityId id : face_ids) {
    faces.push_back(sole_face("curvature", id, faces_in_root));
  }

  const auto n = static_cast<py::ssize_t>(faces.size());
  py::array_t<double> k_max(n);
  py::array_t<double> uv({n, static_cast<py::ssize_t>(2)});
  py::array_t<double> xyz({n, static_cast<py::ssize_t>(3)});
  py::array_t<std::int64_t> used(n);
  double* kp = k_max.mutable_data();
  double* up = uv.mutable_data();
  double* xp = xyz.mutable_data();
  std::int64_t* np = used.mutable_data();

  {
    py::gil_scoped_release release;
    for (py::ssize_t f = 0; f < n; ++f) {
      const TopoDS_Face& face = faces[static_cast<std::size_t>(f)];
      BRepAdaptor_Surface surf(face);
      BRepTopAdaptor_FClass2d inside(face, Precision::Confusion());
      const double u0 = surf.FirstUParameter(), u1 = surf.LastUParameter();
      const double v0 = surf.FirstVParameter(), v1 = surf.LastVParameter();

      double best = 0.0, bu = 0.0, bv = 0.0;
      std::int64_t count = 0;
      for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < samples; ++j) {
          const double u = cell_centre(u0, u1, i, samples);
          const double v = cell_centre(v0, v1, j, samples);
          // A sample outside the face's own trimming is on the underlying surface but not on
          // the face — the middle of a hole, or the cut-away part of a trimmed patch. Taking
          // it would report a curvature the face does not have.
          if (inside.Perform(gp_Pnt2d(u, v)) == TopAbs_OUT) {
            continue;
          }
          BRepLProp_SLProps props(surf, u, v, 2, 1e-7);
          if (!props.IsCurvatureDefined()) {
            continue;
          }
          ++count;
          const double k = peak_curvature(props);
          if (count == 1 || k > best) {
            best = k;
            bu = u;
            bv = v;
          }
        }
      }

      np[f] = count;
      kp[f] = count > 0 ? best : 0.0;
      up[2 * f + 0] = bu;
      up[2 * f + 1] = bv;
      if (count > 0) {
        const gp_Pnt p = surf.Value(bu, bv);
        xp[3 * f + 0] = p.X();
        xp[3 * f + 1] = p.Y();
        xp[3 * f + 2] = p.Z();
      } else {
        xp[3 * f + 0] = xp[3 * f + 1] = xp[3 * f + 2] = 0.0;
      }
    }
  }

  py::dict out;
  out["ids"] = ids_array(face_ids);
  out["k_max"] = k_max;
  out["uv"] = uv;
  out["xyz"] = xyz;
  out["samples_used"] = used;
  return out;
}

py::dict Session::project_on_face(EntityId face_id, const PointArray& points) const {
  const TopoDS_Face face = sole_face("project_on_face", face_id);
  const std::vector<gp_Pnt> pts = points_of("project_on_face", "points", points, 1);

  const auto n = static_cast<py::ssize_t>(pts.size());
  py::array_t<double> closest({n, static_cast<py::ssize_t>(3)});
  py::array_t<double> uv({n, static_cast<py::ssize_t>(2)});
  py::array_t<double> distance(n);
  double* cp = closest.mutable_data();
  double* up = uv.mutable_data();
  double* dp = distance.mutable_data();

  bool failed = false;
  {
    py::gil_scoped_release release;
    const Handle(Geom_Surface) surface = BRep_Tool::Surface(face);
    BRepAdaptor_Surface surf(face);
    GeomAPI_ProjectPointOnSurf proj;
    proj.Init(surface, surf.FirstUParameter(), surf.LastUParameter(),
              surf.FirstVParameter(), surf.LastVParameter());
    for (py::ssize_t i = 0; i < n; ++i) {
      proj.Perform(pts[static_cast<std::size_t>(i)]);
      if (!proj.IsDone() || proj.NbPoints() < 1) {
        failed = true;
        break;
      }
      const gp_Pnt p = proj.NearestPoint();
      double u = 0.0, v = 0.0;
      proj.LowerDistanceParameters(u, v);
      cp[3 * i + 0] = p.X();
      cp[3 * i + 1] = p.Y();
      cp[3 * i + 2] = p.Z();
      up[2 * i + 0] = u;
      up[2 * i + 1] = v;
      dp[i] = proj.LowerDistance();
    }
  }
  if (failed) {
    throw PysmeshError(
        "Session.project_on_face: OCCT found no projection of a point onto face " +
            std::to_string(face_id) + ".",
        "GeomAPI_ProjectPointOnSurf returned no solution. A point on the surface's axis of "
        "revolution has no unique nearest point.",
        {static_cast<int>(face_id)});
  }

  py::dict out;
  out["points"] = closest;
  out["uv"] = uv;
  out["distance"] = distance;
  return out;
}

py::array_t<std::int64_t> Session::entities_in_box(const std::string& kind, double xmin,
                                                   double ymin, double zmin, double xmax,
                                                   double ymax, double zmax,
                                                   bool strict) const {
  if (xmax < xmin || ymax < ymin || zmax < zmin) {
    throw PysmeshError("Session.entities_in_box: every max must be >= its min.");
  }
  Bnd_Box query;
  query.Update(xmin, ymin, zmin, xmax, ymax, zmax);

  std::vector<EntityId> hits;
  for (EntityId id : ids_of_kind(kind_from_name(kind))) {
    const EntityRecord& rec = state_.registry->alive.at(id);
    Bnd_Box box;
    for (const TopoDS_Shape& s : rec.shapes) {
      BRepBndLib::Add(s, box);
    }
    if (box.IsVoid()) {
      continue;
    }
    if (strict) {
      double a, b, c, d, e, f;
      box.Get(a, b, c, d, e, f);
      if (a >= xmin && b >= ymin && c >= zmin && d <= xmax && e <= ymax && f <= zmax) {
        hits.push_back(id);
      }
    } else if (!query.IsOut(box)) {
      hits.push_back(id);
    }
  }
  return ids_array(hits);
}

py::array_t<bool> Session::contains(const std::vector<EntityId>& solid_ids,
                                    const PointArray& points, double tol) const {
  if (solid_ids.empty()) {
    throw PysmeshError("Session.contains: solid_ids must name at least one solid.");
  }
  require_positive("tol", tol);

  std::vector<TopoDS_Shape> solids;
  for (EntityId id : solid_ids) {
    const EntityRecord& rec = require_alive("contains", id);
    if (rec.kind != TopAbs_SOLID) {
      throw PysmeshError("Session.contains: entity " + std::to_string(id) + " is a " +
                         kind_name(rec.kind) + ", not a SOLID.");
    }
    if (rec.shapes.size() != 1) {
      throw PysmeshError("Session.contains: solid " + std::to_string(id) +
                         " was split and denotes several solids; name one of them.");
    }
    solids.push_back(rec.shapes.front());
  }
  const std::vector<gp_Pnt> pts = points_of("contains", "points", points, 1);

  const auto s = static_cast<py::ssize_t>(solids.size());
  const auto p = static_cast<py::ssize_t>(pts.size());
  py::array_t<bool> out({s, p});
  bool* mask = out.mutable_data();

  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < s; ++i) {
      // Loaded once per solid: the classifier builds its spatial data on construction, so
      // one per point would pay that cost for every query.
      BRepClass3d_SolidClassifier classifier(solids[static_cast<std::size_t>(i)]);
      for (py::ssize_t j = 0; j < p; ++j) {
        classifier.Perform(pts[static_cast<std::size_t>(j)], tol);
        // Strictly inside only. A point within tol of the boundary is ON and reads False,
        // which is what a caller seeding a volume needs.
        mask[i * p + j] = classifier.State() == TopAbs_IN;
      }
    }
  }
  return out;
}

}  // namespace session
}  // namespace pysmesh
