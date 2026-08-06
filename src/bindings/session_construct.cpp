// pySMESH binding — Session: primitives, construction geometry and sweeps.
//
// Everything that adds new geometry to the model rather than reworking what is there. The
// add_* operations consume nothing; make_wire/make_face/make_filling and the sweeps consume
// the bodies they are given, because they replace a profile with what was built from it.
//
// Wires carry no EntityId of their own — BRepTools_History supports four shape kinds and
// WIRE is not among them — so a wire is named through the ids of its edges, and each of
// these operations resolves the entities it is given to the single body that owns them.
//
// See session/session.hpp for the split.

#include "session/session.hpp"

namespace pysmesh {
namespace session {

// ---- construction operations ------------------------------------------------------ //

py::dict Session::add_brep(const py::bytes& data) {
  OpGuard guard(in_op_);
  const std::string buffer = data;
  TopoDS_Shape imported;
  {
    py::gil_scoped_release release;
    std::istringstream stream(buffer);
    BRep_Builder builder;
    try {
      BRepTools::Read(imported, stream, builder);
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.add_brep: BREP read failed: ") + e.what());
    }
  }
  if (imported.IsNull()) {
    throw PysmeshError(
        "Session.add_brep: BREP read produced a null shape (empty or malformed data).");
  }
  return add_bodies(imported, "add_brep");
}

py::dict Session::add_box(double dx, double dy, double dz, double ox, double oy, double oz) {
  OpGuard guard(in_op_);
  require_positive("dx", dx);
  require_positive("dy", dy);
  require_positive("dz", dz);
  const TopoDS_Shape solid = build_shape("add_box", [&] {
    return BRepPrimAPI_MakeBox(gp_Pnt(ox, oy, oz), dx, dy, dz).Shape();
  });
  return add_bodies(solid, "add_box");
}

py::dict Session::add_cylinder(double radius, double height, double ox, double oy, double oz,
                        double ax, double ay, double az) {
  OpGuard guard(in_op_);
  require_positive("radius", radius);
  require_positive("height", height);
  const gp_Vec axis(ax, ay, az);
  if (axis.Magnitude() <= 0.0) {
    throw PysmeshError("Session.add_cylinder: axis must be a non-zero vector.");
  }
  const gp_Ax2 frame(gp_Pnt(ox, oy, oz), gp_Dir(axis));
  const TopoDS_Shape solid = build_shape("add_cylinder", [&] {
    return BRepPrimAPI_MakeCylinder(frame, radius, height).Shape();
  });
  return add_bodies(solid, "add_cylinder");
}

py::dict Session::add_cone(double radius1, double radius2, double height, double ox, double oy,
                    double oz, double ax, double ay, double az, double angle_rad) {
  OpGuard guard(in_op_);
  require_non_negative("radius1", radius1);
  require_non_negative("radius2", radius2);
  if (radius1 <= 0.0 && radius2 <= 0.0) {
    throw PysmeshError("Session.add_cone: at least one radius must be > 0.");
  }
  if (radius1 == radius2) {
    throw PysmeshError(
        "Session.add_cone: OCCT's cone needs two different radii (both are " +
        std::to_string(radius1) + "). Use add_cylinder for a straight tube.");
  }
  require_positive("height", height);
  require_sweep_angle("add_cone", angle_rad);
  const gp_Ax2 frame = frame_of("add_cone", ox, oy, oz, ax, ay, az);
  const TopoDS_Shape solid = build_shape("add_cone", [&] {
    return BRepPrimAPI_MakeCone(frame, radius1, radius2, height, angle_rad).Shape();
  });
  return add_bodies(solid, "add_cone");
}

py::dict Session::add_sphere(double radius, double cx, double cy, double cz, double ax,
                             double ay, double az, double angle_rad) {
  OpGuard guard(in_op_);
  require_positive("radius", radius);
  require_sweep_angle("add_sphere", angle_rad);
  const gp_Ax2 frame = frame_of("add_sphere", cx, cy, cz, ax, ay, az);
  const TopoDS_Shape solid = build_shape("add_sphere", [&] {
    return BRepPrimAPI_MakeSphere(frame, radius, angle_rad).Shape();
  });
  return add_bodies(solid, "add_sphere");
}

py::dict Session::add_torus(double radius1, double radius2, double ox, double oy, double oz,
                     double ax, double ay, double az, double angle_rad) {
  OpGuard guard(in_op_);
  require_positive("radius1", radius1);
  require_positive("radius2", radius2);
  if (radius2 >= radius1) {
    throw PysmeshError("Session.add_torus: radius2 (the tube radius, " +
                       std::to_string(radius2) +
                       ") must be smaller than radius1 (the ring radius, " +
                       std::to_string(radius1) + ") for a self-intersection-free torus.");
  }
  require_sweep_angle("add_torus", angle_rad);
  const gp_Ax2 frame = frame_of("add_torus", ox, oy, oz, ax, ay, az);
  const TopoDS_Shape solid = build_shape("add_torus", [&] {
    return BRepPrimAPI_MakeTorus(frame, radius1, radius2, angle_rad).Shape();
  });
  return add_bodies(solid, "add_torus");
}

// A STEP right angular wedge: a box of dx * dy * dz whose face at y = dy is narrowed to
// ltx along x. ltx == dx is a plain box; ltx == 0 is a wedge with a knife edge.
py::dict Session::add_wedge(double dx, double dy, double dz, double ltx, double ox, double oy,
                     double oz, double ax, double ay, double az) {
  OpGuard guard(in_op_);
  require_positive("dx", dx);
  require_positive("dy", dy);
  require_positive("dz", dz);
  require_non_negative("ltx", ltx);
  const gp_Ax2 frame = frame_of("add_wedge", ox, oy, oz, ax, ay, az);
  const TopoDS_Shape solid = build_shape("add_wedge", [&] {
    return BRepPrimAPI_MakeWedge(frame, dx, dy, dz, ltx).Shape();
  });
  return add_bodies(solid, "add_wedge");
}

// ---- construction geometry -------------------------------------------------------- //
//
// These add curve and surface bodies to the root. The registry tracks SOLID/FACE/EDGE/
// VERTEX only (that set is fixed by BRepTools_History), so a WIRE body carries no id of
// its own: it is named through the ids of its edges, and every operation that consumes a
// profile resolves the named entities to the single body that owns them.

py::dict Session::add_line(double x1, double y1, double z1, double x2, double y2, double z2) {
  OpGuard guard(in_op_);
  const gp_Pnt a(x1, y1, z1);
  const gp_Pnt b(x2, y2, z2);
  if (a.Distance(b) <= 0.0) {
    throw PysmeshError("Session.add_line: the two points are coincident.");
  }
  const TopoDS_Shape edge =
      build_shape("add_line", [&] { return BRepBuilderAPI_MakeEdge(a, b).Edge(); });
  return add_bodies(edge, "add_line");
}

// Three-point arc: through p1, ending at p3, passing through p2.
py::dict Session::add_arc(double x1, double y1, double z1, double x2, double y2, double z2,
                   double x3, double y3, double z3) {
  OpGuard guard(in_op_);
  const TopoDS_Shape edge = try_build("add_arc", [&]() -> TopoDS_Shape {
    GC_MakeArcOfCircle mk(gp_Pnt(x1, y1, z1), gp_Pnt(x2, y2, z2), gp_Pnt(x3, y3, z3));
    if (!mk.IsDone()) {
      return TopoDS_Shape();
    }
    return BRepBuilderAPI_MakeEdge(mk.Value()).Edge();
  });
  if (edge.IsNull()) {
    throw PysmeshError(
        "Session.add_arc: no circular arc passes through the three points (they are "
        "collinear or two of them coincide).");
  }
  return add_bodies(edge, "add_arc");
}

py::dict Session::add_circle(double cx, double cy, double cz, double nx, double ny, double nz,
                      double radius) {
  OpGuard guard(in_op_);
  require_positive("radius", radius);
  const gp_Ax2 frame = frame_of("add_circle", cx, cy, cz, nx, ny, nz);
  const TopoDS_Shape edge = build_shape("add_circle", [&] {
    return BRepBuilderAPI_MakeEdge(new Geom_Circle(gp_Circ(frame, radius))).Edge();
  });
  return add_bodies(edge, "add_circle");
}

// A polyline through the given points. Unlike make_wire this shares one vertex between
// consecutive segments by construction, so no edge is ever rebuilt to connect it.
py::dict Session::add_polyline(const PointArray& points, bool closed) {
  OpGuard guard(in_op_);
  const std::vector<gp_Pnt> pts = points_of("add_polyline", "points", points, 2);
  const TopoDS_Shape wire = try_build("add_polyline", [&]() -> TopoDS_Shape {
    BRepBuilderAPI_MakePolygon poly;
    for (const gp_Pnt& p : pts) {
      poly.Add(p);
    }
    if (closed) {
      poly.Close();
    }
    if (!poly.IsDone()) {
      return TopoDS_Shape();
    }
    return poly.Wire();
  });
  if (wire.IsNull()) {
    throw PysmeshError(
        "Session.add_polyline: OCCT could not build the polygon (consecutive points are "
        "most likely coincident).");
  }
  return add_bodies(wire, "add_polyline");
}

// A B-spline approximating the given points to within tol. This is the "spline through
// points" construction; add_bspline takes control points instead.
py::dict Session::add_spline(const PointArray& points, int degree_min, int degree_max,
                             double tol) {
  OpGuard guard(in_op_);
  const std::vector<gp_Pnt> pts = points_of("add_spline", "points", points, 2);
  require_positive("tol", tol);
  if (degree_min < 1 || degree_max < degree_min) {
    throw PysmeshError("Session.add_spline: need 1 <= degree_min <= degree_max (got " +
                       std::to_string(degree_min) + ", " + std::to_string(degree_max) +
                       ").");
  }
  const TopoDS_Shape edge = try_build("add_spline", [&]() -> TopoDS_Shape {
    NCollection_Array1<gp_Pnt> arr(1, static_cast<int>(pts.size()));
    for (std::size_t i = 0; i < pts.size(); ++i) {
      arr.SetValue(static_cast<int>(i) + 1, pts[i]);
    }
    GeomAPI_PointsToBSpline fit(arr, degree_min, degree_max, GeomAbs_C2, tol);
    if (!fit.IsDone()) {
      return TopoDS_Shape();
    }
    return BRepBuilderAPI_MakeEdge(fit.Curve()).Edge();
  });
  if (edge.IsNull()) {
    throw PysmeshError(
        "Session.add_spline: GeomAPI_PointsToBSpline could not approximate the points "
        "to the requested tolerance.");
  }
  return add_bodies(edge, "add_spline");
}

// A clamped, uniformly-knotted B-spline over the given control points. The degree is
// clamped to len(poles) - 1, because a higher degree has no valid knot vector.
py::dict Session::add_bspline(const PointArray& poles, int degree) {
  OpGuard guard(in_op_);
  const std::vector<gp_Pnt> pts = points_of("add_bspline", "poles", poles, 2);
  if (degree < 1) {
    throw PysmeshError("Session.add_bspline: degree must be >= 1 (got " +
                       std::to_string(degree) + ").");
  }
  const int n = static_cast<int>(pts.size());
  const int p = std::min(degree, n - 1);
  const TopoDS_Shape edge = try_build("add_bspline", [&]() -> TopoDS_Shape {
    NCollection_Array1<gp_Pnt> arr(1, n);
    for (int i = 0; i < n; ++i) {
      arr.SetValue(i + 1, pts[static_cast<std::size_t>(i)]);
    }
    const int nk = n - p + 1;
    NCollection_Array1<double> knots(1, nk);
    NCollection_Array1<int> mults(1, nk);
    for (int i = 1; i <= nk; ++i) {
      knots.SetValue(i, static_cast<double>(i - 1));
      mults.SetValue(i, 1);
    }
    mults.SetValue(1, p + 1);
    mults.SetValue(nk, p + 1);
    return BRepBuilderAPI_MakeEdge(new Geom_BSplineCurve(arr, knots, mults, p)).Edge();
  });
  return add_bodies(edge, "add_bspline");
}

// A helical wire. TKHelix's HelixBRep_BuilderHelix is the only helix facade OCCT 8.0 has;
// there is no BRepPrimAPI-style one. It approximates, so the result is a B-spline wire
// whose deviation from the exact helix is bounded by tol.
py::dict Session::add_helix(double cx, double cy, double cz, double ax, double ay, double az,
                     double diameter, double pitch, double turns, double tol) {
  OpGuard guard(in_op_);
  require_positive("diameter", diameter);
  require_positive("pitch", pitch);
  require_positive("turns", turns);
  require_positive("tol", tol);
  const gp_Dir dir = direction_of("add_helix", "axis", ax, ay, az);
  int status = 0;
  const TopoDS_Shape wire = try_build("add_helix", [&]() -> TopoDS_Shape {
    HelixBRep_BuilderHelix mk;
    NCollection_Array1<double> pitches(1, 1);
    pitches.SetValue(1, pitch);
    NCollection_Array1<double> nb_turns(1, 1);
    nb_turns.SetValue(1, turns);
    mk.SetApproxParameters(tol, 8, GeomAbs_C2);
    mk.SetParameters(gp_Ax3(gp_Pnt(cx, cy, cz), dir), diameter, pitches, nb_turns);
    mk.Perform();
    status = mk.ErrorStatus();
    return status == 0 ? mk.Shape() : TopoDS_Shape();
  });
  if (status != 0 || wire.IsNull()) {
    throw PysmeshError(
        "Session.add_helix: HelixBRep_BuilderHelix failed.",
        "ErrorStatus " + std::to_string(status) +
            " (2 = approximation failed, 10 = radius below tolerance, 11 = pitch below "
            "tolerance, 12 = height below tolerance).",
        {});
  }
  return add_bodies(wire, "add_helix");
}

// A planar rectangular face, dx by dy in the frame's own x/y directions.
py::dict Session::add_rectangle(double ox, double oy, double oz, double nx, double ny,
                                double nz, double dx, double dy) {
  OpGuard guard(in_op_);
  require_positive("dx", dx);
  require_positive("dy", dy);
  const gp_Ax2 frame = frame_of("add_rectangle", ox, oy, oz, nx, ny, nz);
  const TopoDS_Shape face = build_shape("add_rectangle", [&] {
    return BRepBuilderAPI_MakeFace(gp_Pln(gp_Ax3(frame)), 0.0, dx, 0.0, dy).Face();
  });
  return add_bodies(face, "add_rectangle");
}

// Join loose edges and wires into one wire, consuming them.
py::dict Session::make_wire(const std::vector<EntityId>& edge_ids) {
  OpGuard guard(in_op_);
  const std::vector<TopoDS_Shape> edges = edges_of("make_wire", edge_ids);
  const std::vector<TopoDS_Shape> owners = owner_bodies("make_wire", edge_ids);
  for (const TopoDS_Shape& b : owners) {
    require_curve_body("make_wire", b);
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(owners);
  TopoDS_Shape wire;
  {
    py::gil_scoped_release release;
    BRepBuilderAPI_MakeWire mk;
    NCollection_List<TopoDS_Shape> list;
    for (const TopoDS_Shape& e : edges) {
      list.Append(e);
    }
    mk.Add(list);
    if (mk.IsDone()) {
      wire = mk.Wire();
    }
  }
  if (wire.IsNull()) {
    throw PysmeshError(
        "Session.make_wire: the named edges do not form a connected wire.",
        "BRepBuilderAPI_MakeWire reports DisconnectedWire: every edge after the first "
        "must share or geometrically touch a vertex of the wire built so far.",
        ids_as_int(edge_ids));
  }
  return commit(concat(survivors, wire), Handle(BRepTools_History)(), "make_wire", wire);
}

// A planar face bounded by the named edges, consuming them.
py::dict Session::make_face(const std::vector<EntityId>& edge_ids) {
  OpGuard guard(in_op_);
  const std::vector<TopoDS_Shape> edges = edges_of("make_face", edge_ids);
  const std::vector<TopoDS_Shape> owners = owner_bodies("make_face", edge_ids);
  for (const TopoDS_Shape& b : owners) {
    require_curve_body("make_face", b);
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(owners);
  TopoDS_Shape face;
  {
    py::gil_scoped_release release;
    const TopoDS_Wire wire = wire_over("make_face", edges, owners);
    // OnlyPlane: a non-planar boundary is a fail-loud here rather than a silently
    // approximated surface. make_filling is the operation for that case.
    BRepBuilderAPI_MakeFace mk(wire, /*OnlyPlane=*/true);
    if (mk.IsDone()) {
      face = mk.Face();
    }
  }
  if (face.IsNull()) {
    throw PysmeshError(
        "Session.make_face: the named edges do not bound a closed planar face.",
        "BRepBuilderAPI_MakeFace(wire, OnlyPlane=true) failed. Use make_filling for a "
        "non-planar boundary.",
        ids_as_int(edge_ids));
  }
  return commit(concat(survivors, face), Handle(BRepTools_History)(), "make_face", face);
}

// A surface filling the named boundary edges, consuming them. Unlike make_face this
// handles a non-planar boundary; the surface is an approximation, so the result's edges
// are new geometry and the boundary edges' ids die.
py::dict Session::make_filling(const std::vector<EntityId>& edge_ids) {
  OpGuard guard(in_op_);
  const std::vector<TopoDS_Shape> edges = edges_of("make_filling", edge_ids);
  const std::vector<TopoDS_Shape> owners = owner_bodies("make_filling", edge_ids);
  for (const TopoDS_Shape& b : owners) {
    require_curve_body("make_filling", b);
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(owners);
  TopoDS_Shape face;
  {
    py::gil_scoped_release release;
    BRepOffsetAPI_MakeFilling mk;
    for (const TopoDS_Shape& e : edges) {
      mk.Add(TopoDS::Edge(e), GeomAbs_C0);
    }
    try {
      mk.Build();
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(
          std::string("Session.make_filling: BRepOffsetAPI_MakeFilling::Build failed: ") +
          e.what());
    }
    if (mk.IsDone()) {
      face = mk.Shape();
    }
  }
  if (face.IsNull()) {
    throw PysmeshError("Session.make_filling: OCCT could not fill the named boundary.",
                       "BRepOffsetAPI_MakeFilling::IsDone() is false.",
                       ids_as_int(edge_ids));
  }
  return commit(concat(survivors, face), Handle(BRepTools_History)(), "make_filling",
                face);
}

// ---- sweeps ----------------------------------------------------------------------- //
//
// Each sweep consumes the profile body and raises its dimension: an edge sweeps to a
// face, a wire to a shell, a face to a solid. The profile survives inside the result as
// the sweep's first shape, so its entity ids carry through structurally, and the walls
// the sweep generates are named against the profile edges they came from.

py::dict Session::extrude(const std::vector<EntityId>& entity_ids, double vx, double vy,
                   double vz) {
  OpGuard guard(in_op_);
  const gp_Vec vec(vx, vy, vz);
  if (vec.Magnitude() <= 0.0) {
    throw PysmeshError("Session.extrude: the extrusion vector must be non-zero.");
  }
  const TopoDS_Shape profile = sole_body("extrude", entity_ids);
  require_sweepable("extrude", profile);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({profile});

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  {
    py::gil_scoped_release release;
    BRepPrimAPI_MakePrism mk(profile, vec, /*Copy=*/false);
    try {
      mk.Build();
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.extrude: BRepPrimAPI_MakePrism failed: ") +
                         e.what());
    }
    if (mk.IsDone()) {
      result = mk.Shape();
      hist = history_of(profile, mk);
    }
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.extrude: OCCT could not sweep the profile.", "",
                       ids_as_int(entity_ids));
  }
  return commit(concat(survivors, result), hist, "extrude", result);
}

py::dict Session::revolve(const std::vector<EntityId>& entity_ids, double ox, double oy,
                          double oz, double ax, double ay, double az, double angle_rad) {
  OpGuard guard(in_op_);
  require_sweep_angle("revolve", angle_rad);
  const gp_Ax1 axis(gp_Pnt(ox, oy, oz), direction_of("revolve", "axis", ax, ay, az));
  const TopoDS_Shape profile = sole_body("revolve", entity_ids);
  require_sweepable("revolve", profile);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({profile});

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  {
    py::gil_scoped_release release;
    BRepPrimAPI_MakeRevol mk(profile, axis, angle_rad, /*Copy=*/false);
    try {
      mk.Build();
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.revolve: BRepPrimAPI_MakeRevol failed: ") +
                         e.what());
    }
    if (mk.IsDone()) {
      result = mk.Shape();
      hist = history_of(profile, mk);
    }
  }
  if (result.IsNull()) {
    throw PysmeshError(
        "Session.revolve: OCCT could not revolve the profile.",
        "A profile that crosses the axis cannot be revolved without self-intersection.",
        ids_as_int(entity_ids));
  }
  return commit(concat(survivors, result), hist, "revolve", result);
}

py::dict Session::pipe(const std::vector<EntityId>& spine_ids,
                const std::vector<EntityId>& profile_ids) {
  OpGuard guard(in_op_);
  const TopoDS_Shape spine_body = sole_body("pipe", spine_ids);
  const TopoDS_Shape profile = sole_body("pipe", profile_ids);
  if (spine_body.IsSame(profile)) {
    throw PysmeshError("Session.pipe: the spine and the profile are the same body.");
  }
  require_sweepable("pipe", profile);
  const TopoDS_Wire spine = wire_of_body("pipe", "spine_ids", spine_body);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({spine_body, profile});

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  {
    py::gil_scoped_release release;
    BRepOffsetAPI_MakePipe mk(spine, profile);
    try {
      mk.Build();
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(std::string("Session.pipe: BRepOffsetAPI_MakePipe failed: ") +
                         e.what());
    }
    if (mk.IsDone()) {
      result = mk.Shape();
      hist = history_of(profile, mk);
    }
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.pipe: OCCT could not sweep the profile along the spine.",
                       "", ids_as_int(profile_ids));
  }
  return commit(concat(survivors, result), hist, "pipe", result);
}

// The general sweep. Unlike pipe it exposes the frame law (Frenet vs corrected Frenet)
// and can close the shell into a solid, which is what a swept CFD body normally needs.
py::dict Session::pipe_shell(const std::vector<EntityId>& spine_ids,
                      const std::vector<EntityId>& profile_ids, bool frenet, bool solid) {
  OpGuard guard(in_op_);
  const TopoDS_Shape spine_body = sole_body("pipe_shell", spine_ids);
  const TopoDS_Shape profile = sole_body("pipe_shell", profile_ids);
  if (spine_body.IsSame(profile)) {
    throw PysmeshError("Session.pipe_shell: the spine and the profile are the same body.");
  }
  const TopoDS_Wire spine = wire_of_body("pipe_shell", "spine_ids", spine_body);
  const TopoDS_Wire prof = wire_of_body("pipe_shell", "profile_ids", profile);
  const std::vector<TopoDS_Shape> survivors = bodies_excluding({spine_body, profile});

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  std::string detail;
  {
    py::gil_scoped_release release;
    BRepOffsetAPI_MakePipeShell mk(spine);
    mk.SetMode(frenet);
    mk.Add(prof);
    if (!mk.IsReady()) {
      detail = "BRepOffsetAPI_MakePipeShell::IsReady() is false.";
    } else {
      try {
        mk.Build();
      } catch (const std::exception& e) {
        py::gil_scoped_acquire acquire;
        throw PysmeshError(
            std::string("Session.pipe_shell: BRepOffsetAPI_MakePipeShell failed: ") +
            e.what());
      }
      if (mk.IsDone()) {
        if (solid && !mk.MakeSolid()) {
          detail = "MakeSolid() failed: the swept shell is not closed.";
        } else {
          result = mk.Shape();
          hist = history_of(profile, mk);
        }
      }
    }
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.pipe_shell: OCCT could not sweep the profile.", detail,
                       ids_as_int(profile_ids));
  }
  return commit(concat(survivors, result), hist, "pipe_shell", result);
}

// Loft through an ordered list of section wires, consuming all of them.
py::dict Session::thru_sections(const std::vector<std::vector<EntityId>>& sections, bool solid,
                         bool ruled) {
  OpGuard guard(in_op_);
  if (sections.size() < 2) {
    throw PysmeshError("Session.thru_sections: at least two sections are required (got " +
                       std::to_string(sections.size()) + ").");
  }
  std::vector<TopoDS_Shape> bodies;
  std::vector<TopoDS_Wire> wires;
  for (const std::vector<EntityId>& ids : sections) {
    const TopoDS_Shape body = sole_body("thru_sections", ids);
    for (const TopoDS_Shape& seen : bodies) {
      if (seen.IsSame(body)) {
        throw PysmeshError(
            "Session.thru_sections: the same body was named as two different sections.");
      }
    }
    bodies.push_back(body);
    wires.push_back(wire_of_body("thru_sections", "sections", body));
  }
  const std::vector<TopoDS_Shape> survivors = bodies_excluding(bodies);

  TopoDS_Shape result;
  Handle(BRepTools_History) hist;
  {
    py::gil_scoped_release release;
    BRepOffsetAPI_ThruSections mk(solid, ruled);
    for (const TopoDS_Wire& w : wires) {
      mk.AddWire(w);
    }
    try {
      mk.Build();
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw PysmeshError(
          std::string("Session.thru_sections: BRepOffsetAPI_ThruSections failed: ") +
          e.what());
    }
    if (mk.IsDone()) {
      result = mk.Shape();
      NCollection_List<TopoDS_Shape> args;
      for (const TopoDS_Shape& b : bodies) {
        args.Append(b);
      }
      hist = new BRepTools_History(args, mk);
    }
  }
  if (result.IsNull()) {
    throw PysmeshError("Session.thru_sections: OCCT could not loft the sections.",
                       "Sections must all be closed or all be open, and must not "
                       "self-intersect when joined.",
                       {});
  }
  return commit(concat(survivors, result), hist, "thru_sections", result);
}

}  // namespace session
}  // namespace pysmesh
