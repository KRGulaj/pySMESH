# Mesh Editing

Once `Mesher.compute()` has built a mesh, or a mesh has been filled from arrays (see
[Discrete meshes](../concepts/discrete-meshes.md)), the rest of `Mesher` measures, names,
edits and searches it. This guide covers quality controls, groups, the editor, the search
surface, the medial axis, and viscous boundary layers.

## Quality controls

Two kinds of object answer two different questions.

- A **`Control`** measures one element and returns a number: a cell volume, an aspect
  ratio, a skew angle. `mesher.quality(control)` evaluates it over every element it applies
  to.
- A **`Predicate`** answers yes or no about one element or node: is this volume inverted,
  does this face sit on a bare border. `mesher.select(predicate)` resolves it to the ids
  that satisfy it.

```python
from pysmesh.mesher import AspectRatio3D, BadOrientedVolume

quality = mesher.quality(AspectRatio3D())
quality.element_ids   # (K,) int64
quality.values          # (K,) float64, parallel to element_ids
quality.skipped          # entities of that family the control does not apply to

inverted = mesher.select(BadOrientedVolume())
inverted.ids
```

Every 3-D measure here has no counterpart in a purely array-side toolkit, because a streamed
surface mesh has neither volume cells nor the reverse connectivity a bare-border or
over-constrained check needs.

### Controls

| Control | Measures | Applies to |
|---|---|---|
| `Volume` | Signed volume; negative means inverted | Volumes |
| `Area` | Element area | Faces, including polygons |
| `Length` | Edge length | Edges |
| `AspectRatio` | Normalised aspect ratio, 1 is regular | Faces, not polygons |
| `AspectRatio3D` | Normalised aspect ratio, 1 is regular | Volumes, not polyhedra |
| `Warping` | Departure from planar, degrees | Four-node faces |
| `Warping3D` | Largest `Warping` of a cell's facets, degrees | Volumes with a four-node facet |
| `ScaledJacobian` | Smallest corner determinant of unit edge vectors; 1 is right-angled, negative is inverted | Volumes, not polyhedra |
| `Taper` | Inequality of the four corner triangles, `[0, 1]` | Four-node faces |
| `Skew` | Departure from right angles, degrees | Faces of 3 or 4 nodes |
| `MinimumAngle` | Smallest interior angle, degrees | Faces |
| `Length2D` | Shortest edge | Faces |
| `Length3D` | Shortest edge | Volumes |
| `Deflection2D` | Distance from the CAD surface (needs a live `Mesher`) | Faces |
| `MaxElementLength2D` | Longest side or diagonal | Faces |
| `MaxElementLength3D` | Longest edge or diagonal | Volumes |
| `MultiConnection` | Elements of higher dimension sharing an edge | Edges |
| `MultiConnection2D` | Largest count sharing a face's border | Faces |
| `NodeConnectivityNumber` | Highest-dimension elements using a node | Nodes |

### Predicates

| Predicate | Accepts |
|---|---|
| `FreeEdges` | A face with a border no other face shares |
| `FreeBorders` | A 1-D element bordering one face or none |
| `FreeNodes` | A node no element uses |
| `FreeFaces` | A face bounding one volume or none: the skin of a volume mesh |
| `BadOrientedVolume` | An inverted cell |
| `BareBorderFace` | A face with a border carried by no 1-D element |
| `BareBorderVolume` | A cell with a boundary facet no face element covers |
| `OverConstrainedFace` / `OverConstrainedVolume` | An element with no free degree of freedom left |
| `CoincidentNodes(tolerance)` | A node with another within `tolerance` |
| `CoincidentElements(element_family)` | An element on exactly the same nodes as another |
| `ManifoldPart(...)` | A face reachable from a start element across manifold borders only |
| `RangeOfIds(ids, element_family)` | Membership of an explicit id set |
| `ElementsOnShape(...)` | An element lying on one sub-shape (needs a live `Mesher`) |
| `BelongToGroup(group_name)` | Membership of a named group |

`And`, `Or` and `Not` compose predicates into a tree; `LessThan`, `MoreThan` and `EqualTo`
turn any `Control` into a predicate by comparing it against a margin:

```python
from pysmesh.mesher import AspectRatio3D, BadOrientedVolume, LessThan, Not, Or

poor_or_inverted = Or(
    LessThan(control=AspectRatio3D(), margin=0.0),   # never true; illustrates composition
    BadOrientedVolume(),
)
mesher.select(Not(poor_or_inverted))
```

`quality` and `select` are also free functions, `pysmesh.mesher.quality(mesh, control)` and
`pysmesh.mesher.select(mesh, predicate)`, for a mesh given as `MeshData` arrays with no
mesher behind it. `Deflection2D` and `ElementsOnShape` read the geometry and only work
through a live `Mesher`.

## Groups

A group is a named set of mesh entities the mesher itself maintains, not one a caller
re-derives after every edit. Three kinds exist, and only the first can be edited by hand:

| Source | Maintained by |
|---|---|
| `GroupSource.EXPLICIT` | An id list, carried through editing by SMESH itself |
| `GroupSource.SHAPE` | Everything bound to one sub-shape |
| `GroupSource.FILTER` | Everything a predicate accepts, re-evaluated on read |

```python
from pysmesh.mesher import ElementDimension, SubShape, SubShapeKind

mesher.add_group("wall", ElementDimension.FACE, ids=[101, 102, 103])
mesher.add_group_on_shape("inlet", ElementDimension.FACE, SubShape(SubShapeKind.FACE, 1))
mesher.add_group_on_filter("bad_cells", ElementDimension.VOLUME, BadOrientedVolume())

mesher.add_to_group("wall", [104])
mesher.remove_from_group("wall", [101])

mesher.groups()          # every MeshGroup, membership as of now
mesher.group("wall")      # one by name
mesher.group_names()       # every name
mesher.remove_group("wall")
```

SMESH rewrites explicit-group membership as it edits: a replaced element is replaced in the
group, a deleted one is dropped. That is what makes it safe to name a region before editing
rather than only after.

## The editor

The editor changes a mesh after it has been computed. Every operation reports element
counts either side of itself, because what an edit did is only readable against what was
there before it.

**Element order.** `convert_to_quadratic(force_3d=True, bi_quadratic=False)` converts the
whole mesh to second order in place, keeping every element id. `convert_from_quadratic()`
converts back to first order. `split_quadratic_into_linear(elements=())` splits
bi-quadratic elements into linear ones with no new nodes.

**Volume splitting.** `split_volumes(method, facet_normal)` cuts every volume cell, which is
how a structured hexahedral block reaches a solver that only takes simplices:

```python
from pysmesh.mesher import SplitMethod

mesher.split_volumes(SplitMethod.HEXA_TO_6)
```

With `avoid_over_constrained=True`, `HEXA_TO_5` and `HEXA_TO_6` choose, per cell, a cut
that makes no tetrahedron whose 4 nodes all lie on 2-D elements. Where no standard cut
qualifies, the cell is cut through its barycentre, which adds a node.

**Boundary elements.** `make_boundary_mesh(dimension, elements=(), around_elements=False,
all_elements=False)` creates the missing faces or edges of volumes, or the edges of faces,
and returns the ids it created. By default only the free boundary gets elements: on an
a x b x c grid of hexahedra that is 2(ab + bc + ca) quadrangles. `all_elements=True`
puts one on every facet, shared or free. An element that exists already is not made
again.

```python
from pysmesh.mesher import BoundaryDimension

skin = mesher.make_boundary_mesh(BoundaryDimension.FACES_OF_VOLUMES)
mesher.add_group("skin", ElementDimension.FACE, skin)
```

**Coincidence and merging.** `find_coincident_nodes(tolerance)` answers what would collapse
without changing anything; `merge_node_groups(groups)` and `merge_nodes(tolerance)` do the
collapsing. `find_equal_elements()` and `merge_equal_elements()` do the same for duplicate
elements built on the same nodes.

**Smoothing.** `smooth(method, iterations, target_aspect_ratio, on_shape, elements,
fixed_nodes)` moves the free nodes of a surface mesh to improve element shapes. With
`on_shape=True` (the default, and only meaningful on a shape-backed mesher), nodes move in
the parameter space of the face they sit on, which is what keeps them on a curved CAD
surface rather than drifting off it.

**Orientation.** `reorient(elements)` reverses named elements outright. `reorient_2d(...)`
makes a connected set of faces consistently wound relative to a direction or a set of
reference faces. `reorient_2d_by_3d(faces, volumes, outside_normal)` orients faces from the
volume cells behind them, the operation an imported surface mesh usually needs, since it can
tell inward from outward where winding alone cannot.

**Splitting and fusing faces.** `quad_to_tri(elements, criterion, diagonal_13)` splits
quadrangles into triangles. `tri_to_quad(elements, criterion, max_angle)` fuses neighbouring
triangles into quadrangles.

**Duplication.** `double_elements(elements)` creates a second element on the same nodes as
each named one, the only way to express a zero-thickness internal wall (a baffle) in this
model.

**Sweeps.** `extrusion_sweep(elements, step, steps, make_boundary, tolerance)` and
`rotation_sweep(elements, axis_origin, axis_direction, angle, steps, tolerance,
make_walls)` sweep elements to fill the swept region with cells of one higher dimension.

**Surface offset.** `offset(value, elements, copy_elements, fix_self_intersection)` builds
an offset surface from linear triangles.

**Sewing.** `sew_free_border(...)` joins one free rim of the mesh to another rim or to a
chain of element edges. `sew_side_elements(...)` merges two matching patches node for node.

**Deletion.** `remove_elements(elements, free_nodes=False)` and `remove_nodes(nodes)` both
return a `RemovalReport` naming exactly which ids went, including the ones nobody asked
for: every element a removed node carried, and, with `free_nodes=True`, every node the
removal left carrying nothing.

```python
report = mesher.remove_elements([1001, 1002], free_nodes=True)
report.elements   # every element id that is gone
report.nodes        # every node id the removal orphaned and then took
```

## Search

Every search query takes a batch of points or one ray, because building the octree the
query runs on is the expensive part; asking one point at a time would pay for it repeatedly.

```python
import numpy as np
from pysmesh.mesher import ElementDimension

points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)

hits = mesher.find_elements_by_point(points)                 # ElementsAtPoints
nearest = mesher.find_closest(points, ElementDimension.FACE)  # (N,) int64
distance = mesher.closest_distance(points, ElementDimension.VOLUME)  # ClosestElements
projected = mesher.project_points(points)                      # ProjectedPoints
state = mesher.point_state(points)                              # (N,) PointState, closed surfaces only

in_sphere = mesher.elements_in_sphere(centre=(0.0, 0.0, 0.0), radius=2.0)
in_box = mesher.elements_in_box(minimum=(-1.0, -1.0, -1.0), maximum=(1.0, 1.0, 1.0))
near_line = mesher.elements_near_line(origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0))

hits = mesher.ray_hits(origin=(0.0, 0.0, -5.0), direction=(0.0, 0.0, 1.0))
hits.ids            # faces hit, nearest first
hits.crossings       # distinct positions the surface was actually crossed at

cells = mesher.ray_volumes(origin=(0.0, 0.0, -5.0), direction=(0.0, 0.0, 1.0))
cells.ids            # volume cells crossed, in the order the ray enters them
cells.entry, cells.exit   # distances where it enters and leaves each one
```

`ray_volumes` cuts the ray by the plane of every facet of each cell (Haines' test), which
is exact for cells with planar facets. The origin's own cell has a negative entry, and a
cell behind the origin is not reported.

`closest_distance(points, family=ElementDimension.VOLUME)` is the one query with no
counterpart in a surface-only pipeline: the distance from a point to a **volume cell**.

`sharp_edges` and `separate_faces_by_edges` divide a surface mesh into regions bounded by
its creases; see [Discrete meshes](../concepts/discrete-meshes.md#the-patch-workflow) for
the full workflow, which applies identically whether or not the mesher has a shape.

`merge_obstruction(element, groups)` answers, before a merge runs, which of one element's
nodes the merge must keep apart to leave it valid rather than folded. `make_slot(width,
segments)` cuts a slot of the given width around a chain of 1-D elements lying on a
triangle mesh.

## The medial axis

The medial axis of a face is the set of centres of the maximal circles that fit inside it:
its centreline, and the local wall thickness at every point along it. `medial_axis` computes
it directly from the geometry, so it needs a `Shape`, not a `Mesher`:

```python
from pysmesh import medial_axis

axis = medial_axis(shape, face=1, min_segment_length=0.05)
axis.branches          # one MedialBranch per branch, in construction order
axis.branch_points      # how many points three or more branches meet at

spine = axis.longest    # the longest branch: a thin region's spine
spine.widths              # local width, sampled along the branch
spine.boundary1_edge       # which EDGE ordinal each width sample's boundary point lies on
```

A branch is a polyline with one point per medial-axis edge, plus one. The boundary
discretisation decides how many points that is: SMESH cuts every boundary edge into at least
9 pieces, each shorter than `min_segment_length`. A smaller `min_segment_length` therefore
gives more points. A straight branch is a polyline of collinear points, not its two end
points. For example, the spine of the 10 x 4 rectangle in `tests/test_medial.py` has 121
points at `min_segment_length=0.1`. Branch 0 is not necessarily the spine; `axis.longest`
is the reliable pick for a thin region. Pass `ignore_corners=True` to drop the arms that run
into a boundary's convex corners and keep only the axis proper.

## Viscous boundary layers

Three paths grow prism layers (or, in a face, quadrangle layers) on chosen walls. They share
one stack definition, and they serve different situations.

### The stack

Every path takes the same three numbers: the total thickness `T`, the layer count `N`, and
the stretch factor `f`, the ratio of one layer's thickness to the one before it. The first
layer, at the wall, is

    t1 = T (f - 1) / (f^N - 1)        (t1 = T / N when f = 1)

and layer `k` ends at `t1 (f^k - 1) / (f - 1)` from the wall (`k t1` when `f = 1`), so the
layers add up to `T`. `pysmesh.first_layer_thickness(T, f, N)` returns `t1`. SMESH grows a
stack only for `T > 0`, `N >= 1` and `f >= 1`; every path refuses other values with a
`PysmeshError` when you build the parameters, not later in the compute.

Each layer edge runs from a node of the inner mesh to its nearest point on the wall, and the
layer nodes divide it at the fractions of the closed form. On a plane wall the edge is the
wall normal, so the layer nodes lie on the closed-form planes. On a curved wall they lie on
the closed-form offsets along the normal (radially, on a cylinder).

### Path 1: the hypothesis inside a `Mesher`

Assign `ViscousLayers` (grown from faces of a solid) or `ViscousLayers2D` (grown from edges
of a face) beside the algorithms. `boundary` names the walls by ordinal, or, with
`ignore=True`, the faces or edges without layers. The layer cells land in the group the
hypothesis names:

```python
from pysmesh.mesher import Hexa3D, Mesher, NumberOfSegments, Quadrangle2D, Regular1D, ViscousLayers

mesher = Mesher(shape)
mesher.assign(Regular1D())
mesher.assign(NumberOfSegments(count=3))
mesher.assign(Quadrangle2D())
mesher.assign(Hexa3D())
mesher.assign(
    ViscousLayers(
        total_thickness=0.4,
        layer_count=2,
        stretch_factor=1.2,
        boundary=(1,),           # face ordinals the layers grow on
        group_name="wall_layers",
    )
)
report = mesher.compute()
layer_cells = mesher.group("wall_layers")
```

Only some algorithms build the layers in their compute:

| Hypothesis | Algorithms that build it |
|---|---|
| `ViscousLayers` | `Hexa3D`, `PolyhedronPerSolid3D`, `Cartesian3D` |
| `ViscousLayers2D` | `Quadrangle2D`, `QuadFromMedialAxis1D2D`, `Mefisto2D` |

If a layer hypothesis reaches a solid (a face) that another algorithm meshes, `compute()`
raises before it meshes anything, and names the sub-shape and the algorithm. Without that
check the layers were dropped with no word (`Prism3D`, `RadialQuadrangle1D2D`), or the compute
failed after building some of them (`PolygonPerFace2D`), or the process crashed
(`CompositeHexa3D`). Assign the layers only to the sub-shapes that a building algorithm
meshes.

Several `ViscousLayers` can reach one solid, each with its own face set, to give each face set
its own thickness:

| Algorithm | Several `ViscousLayers` on one solid |
|---|---|
| `PolyhedronPerSolid3D` | Each hypothesis grows its own stack on its own faces. |
| `Hexa3D`, `Cartesian3D` | Not read: each reads one hypothesis per solid. `compute()` refuses a second one. |

SMESH refuses two face sets that share a face, and two face sets with a different
`layer_count` on faces that share an edge. `compute()` raises with SMESH's reason before it
meshes anything, and names the face by its ordinal. To detach one of several hypotheses, give
`Mesher.unassign` an instance equal to it, field for field.

`Cartesian3D` grows its layers another way. It shrinks the shape by `T`, lays its grid in
the shrunk shape, and fills the gap with layer cells. For that inner mesh it keeps every cut
cell that has volume, whatever `CartesianParameters3D.size_threshold` says, because a dropped
cell left a hole in the layers.

### Path 2: the two-step builder

`ViscousLayerBuilder` splits the same work in two, so that you choose the mesher of the inner
volume yourself:

```python
from pysmesh.mesher import Cartesian3D, CartesianParameters3D, Mesher, ViscousLayerBuilder

builder = ViscousLayerBuilder(0.2, 3, 1.2, boundary=(1,), ignore=False, group_name="bl")
with Mesher(shape) as outer:
    shrunk = outer.shrink_geometry(builder)      # the shape offset inward by T
    with Mesher(shrunk) as inner:
        inner.assign(Cartesian3D())
        inner.assign(CartesianParameters3D(spacing_x="0.25", spacing_y="0.25",
                                           spacing_z="0.25", create_faces=True,
                                           add_edges=True))
        inner.compute()
        outer.add_layers(builder, inner)         # inner mesh + layers, into `outer`
    mesh = outer.mesh()
```

Any inner mesher works, provided it meshes the shrunk shape itself. A Cartesian inner mesh
needs `create_faces=True` and `add_edges=True`, because the layers grow from its boundary
faces and edges. On a face, `shrink_geometry` offsets the whole wire and `add_layers` grows
quadrangle rings. `add_layers` refuses a call before `shrink_geometry`, a builder other than
the one that shrank the shape, and an inner mesher on another shape.

### Path 3: `compute_viscous_layers` on a surface mesh

`compute_viscous_layers` is the standalone, lower-level entry point. It grows layers on a
surface mesh that was put onto a `Shape` by hand through the low-level `pysmesh.Mesh` class,
rather than computed by a `Mesher`. Use it for a surface mesh that came from somewhere else
and needs layers before it goes to a solver:

```python
import pysmesh

mesh = pysmesh.Mesh(shape)
# ... inject a classified surface mesh via mesh.add_nodes / mesh.classify_on_face /
#     mesh.add_triangles; see examples/box_bl.py for the full, verified sequence ...

params = pysmesh.VLParams(
    face_ids=tuple(f.id for f in shape.faces()),
    total_thickness=0.1,
    n_layers=5,
    stretch_factor=1.2,
    group_name="BL",
)
result = pysmesh.compute_viscous_layers(mesh, params)

result.prism_connectivity   # (K, 6) int32 row indices, VTK wedge node order
result.node_coords            # (P, 3) float64, every node after the compute
result.failed_face_ids         # wall faces that received no layers
mesh.release()
```

`is_ignore=True` on `VLParams` reads `face_ids` as the faces without layers.

### Limits that remain

- `CompositeHexa3D` builds no layers. With the hypothesis made readable, the layer cells on
  its side faces give their grids more rows than the opposite faces have, and its block grid
  breaks. Use `Hexa3D` for a block with layers.
- `PolygonPerFace2D` builds no layers: after the layer step it finds too few nodes on the
  face wire.
- `ViscousLayers2D` takes no extrusion method; only the 3-D hypothesis has one.
- `Cartesian3D` with layers: a stack too thick for the shape, so that one shrunk surface
  meets another, is not supported. The compute fails on the solid with the reason ("the
  solid offset inward by the total thickness ... is empty ... the layers are too thick for
  the shape"), and leaves no cell. At an edge between two walls with layers, the corner cells have warped faces where
  the grid lines on a wall cross that edge at an angle other than 90 degrees (the caps of a
  hexagonal prism). The mesh there is conforming, but the `Volume` control
  splits each warped cell on its own, so its sum can differ from the shape's volume by about
  1e-4 relative (2.7e-4 on a hexagonal prism with layers on every face).

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
