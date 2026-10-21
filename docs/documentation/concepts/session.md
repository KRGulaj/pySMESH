# Session

`pysmesh.Session` is the stateful CAD modelling context. It owns one live shape and a
persistent entity-id registry, and it is the class to reach for whenever a pipeline builds
or edits geometry interactively rather than running one algorithm on a file. This page
explains why it exists, what it costs, and the operation families it exposes. For the id
scheme itself, see [Entity IDs and ordinals](entity-ids.md).

## Why a stateful session

The rest of pySMESH is a stateless BREP-in, BREP-out service: each entry point reads bytes,
runs one OCCT algorithm, and writes bytes back. That is the right shape for one-shot work.
It is the wrong shape for interactive modelling, because a serialise and deserialise
boundary destroys the shape identity OCCT's own history maps are expressed in. Without that
identity, operation histories cannot be composed, an undo costs a full re-parse, and every
render starts tessellation from scratch.

A `Session` keeps the shape alive in memory across every operation, and carries each
operation's OCCT history forward into its own `EntityId` registry. Two properties follow,
and they are the reason the class exists:

- **Ids are never reused.** A stale reference always resolves to *dead*, never to a
  different entity.
- **Snapshot and restore are O(1).** A shape is a handle triple, and the registry is shared
  immutably, so a snapshot copies handles rather than geometry, at any model size.

## Creating a session

```python
from pysmesh import Session

s = Session()          # validate=True by default
s.add_box(3.0, 7.0, 11.0)
```

`Session(validate=False)` skips the `BRepCheck_Analyzer` pass most operations run on the
shape they just built. Validation costs time per operation, not per model, so leave it on
unless a measured hot path needs it off. Every operation but the healing family raises
rather than committing an invalid result; the healing family reports a verdict on
`HistoryDelta.valid` instead, because its input is invalid by assumption.

## Snapshot and restore

```python
mark = s.snapshot()     # O(1): copies handles, not geometry
s.restore(mark)          # O(1): rewinds the shape and the id registry
s.discard_snapshot(mark)  # release it once no branch needs it
```

`restore` rewinds the live shape and the registry to a retained state, but it does **not**
rewind the operation counter or the id counter. That is deliberate: rewinding either would
let a later operation re-issue an id an abandoned branch already used, which is the one
failure the id scheme exists to prevent. See
[Entity IDs and ordinals](entity-ids.md#why-ids-are-never-reused).

Because a snapshot is a handle copy, its cost is bookkeeping, not geometry: retaining
hundreds of snapshots of a large model measures in kilobytes, not megabytes, and both
`snapshot` and `restore` stay under a millisecond regardless of model size. Discard a mark
once nothing can restore to it, or memory grows with the number of retained states.

## The operation families

`Session` is assembled from operation groups. Each group lives in its own module on the
native side and on the Python side, so a change on either has an obvious counterpart on the
other.

| Family | Covers |
|---|---|
| Primitives | `add_box`, `add_cylinder`, `add_cone`, `add_sphere`, `add_torus`, `add_wedge`, `add_vertex` |
| Curve and surface construction | `add_line`, `add_arc`, `add_circle`, `add_ellipse`, `add_polyline`, `add_spline`, `add_bspline`, `add_helix`, `add_rectangle`, `make_wire`, `extract_edges`, `make_face`, `make_filling`, `extend_face`, `extend_edge` |
| Sweeps | `extrude`, `revolve`, `pipe`, `pipe_shell`, `thru_sections` |
| Booleans with history | `fuse`, `cut`, `common`, `section`, `split`, `fragment` |
| Splits with piece ids | `split_by_faces`, `split_by_plane`, `keep_half_space` |
| Fillet and chamfer | `fillet`, `chamfer`, `fillet_2d` (corners of a free planar face) |
| Offsets | `make_thick_solid`, `offset`, `offset_wire` (a parallel planar wire) |
| Transforms | `translate`, `rotate`, `mirror`, `scale`, `position_along_path`, `copy` |
| Healing | `heal`, `sew`, `remove_internal_wires`, `unify_same_domain`, `defeature`, `imprint`, `remove`, `limit_tolerance`, `change_orientation` |
| Gluing | `glue` (coincident faces or edges, GEOM's gluer) |
| GEOM healing operators | `divide_edge`, `close_contour`, `fill_holes` |
| Tessellation | `tessellate` (the incremental render mesh) |
| Queries | `entity_table`, `entity_types`, `bounding_boxes`, `mass_properties`, `surface_parameters`, `curve_geometry`, `face_wires`, `surface_at`, `curve_at`, `curvature`, `project_on_face`, `distance`, `entities_in_box`, `contains`, `adjacency`, `face_parameter_bounds`, `edge_parameter_bounds` |
| Construction queries | `face_frame`, `tangent_frame`, `face_face_axis`, `point_at_length` |
| Locating entities | `entities_on`, `in_place`, `find_same` |
| Selection rules | `select`, with a `Rule` of filters; `rule_to_dict`, `rule_from_dict` |
| Checks | `check_boolean`, `check_shape`, `small_edges`, `is_closed_shell`, `tolerances` |
| Handoff | `export_handoff`, `brep`, `write_step` |
| Identity and introspection | `entities`, `entity_kind`, `is_alive`, `shape_count`, `name_of`, `origin`, `resolve`, `op_count`, `state_op_index`, `issued_id_count`, `entity_count` |

Every mutating operation returns a `HistoryDelta`: which ids it created, deleted, modified,
split and merged, plus the `BRepCheck_Analyzer` verdict on the shape it built. See
[Entity IDs and ordinals](entity-ids.md#identity-rules-across-an-edit) for what each of
those five arrays means.

### Splits with piece ids

`split_by_faces`, `split_by_plane` and `keep_half_space` cut solids by faces or by a
plane, and give each piece of a cut solid its own id: the sixth rule of
[Entity IDs and ordinals](entity-ids.md#the-sixth-rule-pieces-of-a-cut-solid). The tool
faces of `split_by_faces` stay in the model; the plane of the other two is a planar face
that the operation builds and does not add to the model. A target that the plane does
not cross is refused, with its range of signed distances from the plane. The three run
their boolean on one thread, so they take no `parallel` argument, and two runs on equal
input give equal bytes and equal ids.

### Drafted and symmetric extrusion

`extrude` takes two keyword options. With `symmetric=True` the sweep runs from
`-vector` to `+vector`. With `draft_angle` the walls lean by that angle, in radians:
a positive angle narrows the solid along the sweep, a negative one widens it. A draft
applies to one planar face swept along its normal. OCCT's evolved sweep builds the
drafted solid with exact planes and cones. Its offset rounds a corner with a cylinder
or cone face where the offset moves away from the corner. A narrowing draft whose walls
meet before the end of the sweep is refused, with the height where they meet. With
neither option, `extrude` is the plain sweep, unchanged.

### Filling tangent to faces

`make_filling` takes `support_faces`, a mapping from boundary edges to faces
that contain them. The surface is tangent to each support face along its edge
(G1), as in GEOM's `MakeFaceWithConstraints`. A supported edge belongs to its
face's body, so it is not consumed and keeps its id. The boundary edges must
meet at shared vertices: edges of two bodies that only touch at coincident
vertices are refused, because OCCT leaves that boundary open. With no
`support_faces`, `make_filling` is unchanged.

### Moving along a path

`position_along_path` moves the bodies of the named entities rigidly along a
path edge. It takes the path's frame at its start onto its frame at a fraction of
the path's length, measured by arc length. The frame is OCCT's corrected Frenet
frame, as GEOM's `PositionAlongPath` uses it: origin on the curve, z along the
tangent, x along the normal. On a straight path the move is a translation, on a
circular arc a rotation about the arc's axis. Every id survives, as for
`translate`. A moved body that owns the path edge is refused, because the path
would move with it.

### Gluing

`glue(entities, tolerance)` is GEOM's `MakeGlueFaces`: it merges the faces, edges and
vertices of the named bodies that coincide within `tolerance`. With `kind="edges"` it is
`MakeGlueEdges`, and only coincident edges merge. Every id of a glued group survives on its
one shape, as for a merge. Glue intersects nothing: two faces merge only when they coincide
as a whole. `fragment` intersects the operands and splits faces that overlap in part. A
glue that finds nothing coincident is refused, naming the bodies and the tolerance.

### Dividing an edge

`divide_edge(edge, positions, by="length")` is GEOM's `DivideEdge`. It divides an edge at
fractions of its length, or of its parameter range with `by="parameter"`, in every face
that holds it. The pieces get new ids generated from the edge, which dies: the sixth rule
for edges, see [Entity IDs and ordinals](entity-ids.md#the-sixth-rule-for-edges-pieces-of-a-divided-edge).
The faces keep their ids.

### Closing a contour

`close_contour(edges, common_vertex=False)` is GEOM's `CloseContour`. The edges must be
exactly one open wire: a wire of a face, or a free wire. A gap wider than 1 model unit is
closed by a new straight edge, a narrower one by pulling the two end edges together. With
`common_vertex=True` the two ends become one vertex whose tolerance covers the gap. A new
edge gets a new id generated from the two edges it joins; every other id stays. A free
wire comes out as one closed wire. Like `heal`, the operation reports the validity verdict
instead of refusing an invalid result.

### Filling holes

`fill_holes(entities, edges=None)` is GEOM's `FillHoles`. It fills each free-boundary hole
of the named shells and faces with a GeomPlate face built on the hole's own edges, and puts
the face in the shell, so the shell closes. With `edges`, only the holes those edges bound
are filled. Each new face gets a new id generated from the edges of its hole.

When to use which:

| Need | Use |
|---|---|
| Close an open shell in one step, the cap sharing the shell's edges | `fill_holes` |
| An exact planar cap on a planar loop | `extract_edges` of the loop, then `make_face` |
| A free filling surface on a non-planar loop, tangent to faces if asked | `extract_edges`, then `make_filling` |

`make_face` and `make_filling` make a separate face on copies of the edges, so the body
stays open until a `sew` joins the cap. The surface of `fill_holes` is a GeomPlate
approximation within 1e-4, also for a planar hole: on a disk of radius 2 its area is
12.56630 against the exact 12.56637.

### Booleans: `fuzzy` and `parallel`

Every boolean and `imprint` takes `fuzzy` (extra tolerance for deciding two shapes touch)
and `parallel` (run OCCT's internal steps on several threads). Both default to values that
are correct for clean geometry: `fuzzy=0.0` means "use each shape's own stored tolerance",
which is the right answer for geometry built in the session or imported cleanly. Raise it
only for an import whose faces do not quite meet, and choose it against the measured gap
rather than turning it up for luck: a `fuzzy` value larger than the model's smallest real
feature merges things that are genuinely separate.

`parallel=True` does not change the result, but it changes the memory, not only the speed.
Each OCCT worker thread holds its own working data. Measured on 16 threads with `fragment`
of curved blocks: about 5 MB per input B-spline face in parallel, against about 0.8 MB with
`parallel=False`, 6 to 7 times more. 512 lofted blocks (3 072 faces) peaked at 15.6 GB in
parallel and at 2.2 GB without. Pass `parallel=False` to `fragment` and to the other
booleans when 5 MB times the face count nears the memory you can spare.

## Two queries a feature filter needs

`surface_parameters` reads a face's analytic parameters straight off its surface: a
cylinder's radius, a cone's taper, a torus's two radii. A parameter the surface type does
not define reads `NaN`, never a stand-in `0.0`. `face_wires` splits a face's boundary into
its loops, so an inner loop (a hole) is distinguishable from the outer one.

```python
import numpy as np
from pysmesh import Session
from pysmesh.session import EntityKind

s = Session()
s.add_box(3.0, 7.0, 11.0)

faces = s.entities(EntityKind.FACE)
params = s.surface_parameters(faces)

# Every cylindrical face under 1 mm across: fillets and small bores.
small = params.ids[(np.array(params.types) == "Cylinder") & (params.radius1 < 1.0)]

wires = s.face_wires(faces)
for row in np.flatnonzero(~wires.is_outer):        # one row per hole
    lo, hi = wires.edge_range[row]
    hole_edges = wires.edge_id[lo:hi]
```

## The edge and clearance queries

Three more queries answer about edges and about the space between entities. Each takes
entity ids and returns arrays, so a worker holding one document can answer without
serialising anything out.

`curve_geometry` is the curve-side `surface_parameters`. It reads a line's direction, a
circle's or an ellipse's centre, plane normal and radii. Only `"Line"`, `"Circle"` and
`"Ellipse"` carry parameters; every other curve type reports `analytic` False and a row of
`NaN`. The axis is the curve's own, so two arcs cut from one circle report the same axis.

`curve_at` gives the point and unit tangent at each parameter of one edge. **The tangent
follows increasing parameter and is not flipped for a reversed edge**, which is the opposite
of `surface_at`. The asymmetry is in the geometry: a face belongs to one shell, so REVERSED
is the face's own property and decides its outward normal, while an edge shared by two faces
is FORWARD in one wire and REVERSED in the other. `defined` reads False on a degenerate edge,
where the first derivative is the null vector and any direction would be invented.

`distance` is the minimum distance between two entities of any kind, with the witness point
on each. It is OCCT's exact shape-to-shape extremum, not a bounding-box estimate, so it is
what a clearance check reads; `0.0` means the two entities touch or intersect.

```python
edges = s.entities(EntityKind.EDGE)
curves = s.curve_geometry(edges)

# Every circular edge under 1 mm across: the rims of small bores.
is_circle = np.array(curves.type) == "Circle"
small_rims = curves.edge_id[is_circle & (curves.radius[:, 0] < 1.0)]

first, last = s.edge_parameter_bounds([edges[0]])[0]
sample = s.curve_at(edges[0], np.linspace(first, last, 32))
direction = sample.tangents[sample.defined]

# The clearance between two entities, with the witness point on each.
gap = s.distance(faces[0], faces[2])       # gap.distance, gap.point_a, gap.point_b
```

## Construction queries

Four queries give a place to build new geometry on. `face_frame` is a planar face's
centroid, outward normal and in-plane x-direction. `tangent_frame` is the point, the
outward normal and the unit `dS/du` of a face at each `(u, v)`. `face_face_axis` is the
common line of two planar faces' planes, along the cross product of their outward
normals. `point_at_length` is the point at a length along an edge, from either end.
Each refuses what has no answer, naming the entity and the value: a curved face for a
frame, a parameter outside the face or at a singular point, two parallel planes, a
length off the edge. None of them issues an id.

## Locating entities

Three queries run GEOM's classifiers on the live model. None of them issues an id or
writes to a shape of the model.

- `entities_on(kind, region, state)` lists the live ids of a kind that lie on, in or out
  of a region: a `PlaneRegion`, a `CylinderRegion`, a `SphereRegion`, a `BoxRegion` or a
  solid of the model. It is GEOM's `GetShapesOnPlane`, `GetShapesOnCylinder`,
  `GetShapesOnSphere`, `GetShapesOnBox` and `GetShapesOnShape`, with GEOM's tolerances.
  For a plane, "out" is the side the normal points to.
- `in_place(kind, source, old)` lists the live ids that now stand in place of old
  sub-shapes. `source` is BREP bytes with handoff ordinals in `old`, or a snapshot mark
  with ids live in that snapshot. It is GEOM's `GetInPlace`. It finds a face that an
  operation trimmed, split or merged into a larger one. An empty answer means the old
  sub-shape is gone.
- `find_same(entity, within)` lists the live ids that GEOM's `GetSame` rule calls the
  same as one entity: coincident vertices, edges, faces or solids within 1e-7.

```python
from pysmesh import PlaneRegion

below = s.entities_on(EntityKind.FACE, PlaneRegion((0, 0, 5), (0, 0, 1)), "in")

handoff = s.export_handoff()
mark = s.snapshot()
s.fillet(edges, 0.5)
now = s.in_place(EntityKind.FACE, mark, [face])            # by snapshot ids
same = s.in_place(EntityKind.FACE, handoff.brep, [0, 1])   # by handoff ordinals
```

`entities_on` samples each face at the nodes of a triangulation of a private copy of
the model, meshed as GEOM meshes it. Its answer does not depend on an earlier
`tessellate`, and the model's faces keep the triangulation they had.

## Selection rules

`select(kind, rule)` returns the live ids of a kind that pass every filter of a `Rule`.
The sixteen filters follow SALOME Shaper's `FiltersPlugin`: sizes, horizontal and vertical
faces, entities on a plane, a line or a geometry, on one side of a plane, relative to a
solid, belonging to entities, feature edges, continuous faces, opposite edges, external
faces and topologically connected faces. Each filter has `reverse`. `rule_to_dict` and
`rule_from_dict` give a rule a versioned stored form. See
[Selection rules](selection-rules.md) for the filters and the format.

## Checks

Five read-only checks say whether a model is fit for the next operation.
`check_boolean` runs OCCT's `BOPAlgo_ArgumentAnalyzer` on a boolean's arguments and
tools with every check it offers, and returns one finding per problem: the OCCT check
that failed and the ids on each side. `check_shape` runs `BRepCheck_Analyzer` on the
bodies and returns one finding per invalid sub-shape, with its id, its kind and its
`BRepCheck_Status` names. Both take `progress` and `cancel`; `check_shape` checks one
body at a time, because OCCT gives its analyzer no progress range. `small_edges` lists
the edges shorter than a length, `is_closed_shell` says per entity whether its body is
bounded by closed, consistently oriented shells, and `tolerances` gives the stored
tolerance of each face, edge or vertex. An empty finding list means no problem.

## Reaching a mesher

A `Session` never meshes. `export_handoff()` crosses the boundary once, on a shape nobody
is editing, and returns the BREP bytes plus one id array per entity kind, each in the
traversal order a reader of the bytes reproduces:

```python
import pysmesh
from pysmesh.mesher import Hexa3D, Mesher, NumberOfSegments, Quadrangle2D, Regular1D

handoff = s.export_handoff()

mesher = Mesher(pysmesh.load_brep(handoff.brep))
mesher.assign(Regular1D())
mesher.assign(NumberOfSegments(count=8))
mesher.assign(Quadrangle2D())
mesher.assign(Hexa3D())
mesher.compute()
```

`handoff.face_id[i]` is the `EntityId` of the face a reader of `handoff.brep` enumerates at
position `i`. Pairing that array with a mesh element's sub-shape ordinal (see
[Meshing model](meshing-model.md)) is what carries a mesh cell back to the session entity it
came from. By default `export_handoff` verifies the id-to-sub-shape map is a bijection
before it returns, and raises rather than handing back a map that has quietly lost some of
the caller's names. Three causes make the pairing ambiguous: a same-domain merge leaves
several live ids on one face, a boolean leaves both operands' ids on every sub-shape they
share (every `common`, and a `fuse` with coincident sub-shapes), and a split leaves one live
id on several.

`export_handoff(allow_aliases=True)` returns a many-to-one map instead. Each ordinal
carries its sub-shape's label, the lowest live id that denotes it, so a split id appears at
every ordinal of its pieces. `handoff.face_ids_of[i]` (and the same for solids, edges and
vertices) lists every live id of the sub-shape at ordinal `i`, label first. An id resolves
to every ordinal whose tuple lists it: those are exactly the sub-shapes it denotes, also
where two ids share only part of their sub-shapes, as after a coplanar `fuse`.

`write_step(unit=..., face_names=..., face_colors=...)` writes the live shape to STEP with
face names and face colours keyed by entity id, so they survive the edits that move
ordinals. Names and colours follow one set of rules, which follow the three causes above:

- A split id names (colours) every piece it denotes.
- A face that several ids denote takes the name (the colour) when every id among them that
  gives one gives the same one. An id that gives none does not block another id.
- If the ids of one face give different names (colours), `write_step` raises. The message
  lists each such face with its ids and their values.

A colour is `(r, g, b)`, each component a finite number in `[0, 1]`. Any other colour is
refused, with its face id. OCCT keeps a colour as three 32-bit floats, so 0.9 reads back as
0.8999999761581421.

`read_step_xde` reads the names and the colours back on `face_labels`. `write_step` reads
the ids of each face from the registry, which is the same relation `face_ids_of` reports.

## Non-finite arguments

Every float argument of a session operation must be finite. A NaN or an infinity raises
`PysmeshError` before any OCCT call. The message names the operation, the argument and
the value, for example `Session.extrude: vector must be finite (got (nan, 0, 1))`. The
session is left exactly as it was. The standalone functions (`offset_shape`,
`make_thick_solid`, `point_in_solid`, `tessellate`, `unify_same_domain`,
`compute_viscous_layers`, the `Shape` and `Mesh` methods) and every numeric field of a
mesher hypothesis follow the same rule.

## Thread contract

A `Session` is **not** thread-safe. Use one session per thread. Sessions are independent,
so several may coexist in one process with no cross-talk. Long operations release the GIL,
so entering an operation on a session while another is already in flight on that same
session raises, rather than racing on the registry.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
