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
```

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

A branch is not a dense polyline: it carries one point per medial-axis edge plus one, so a
straight branch is exactly two points. Branch 0 is not necessarily the spine; `axis.longest`
is the reliable pick for a thin region. Pass `ignore_corners=True` to drop the arms that run
into a boundary's convex corners and keep only the axis proper.

## Viscous boundary layers

Two different paths grow prism (or, on a face, quadrangle) boundary layers, and they serve
different situations.

**Inside a normal `Mesher.compute()`**, assign the `ViscousLayers` (3-D, grown from named
faces of a solid) or `ViscousLayers2D` (2-D, grown from named edges of a face) hypothesis
alongside the volume or face algorithm. The layer cells land in the group the hypothesis
names:

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

**`compute_viscous_layers`** is the standalone, lower-level entry point: it grows layers on
a surface mesh that was injected onto a `Shape` by hand through the low-level `pysmesh.Mesh`
class, rather than computed by a `Mesher`. This is the path for a surface mesh that came
from somewhere else and needs boundary layers added before it is handed to a solver:

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

Both paths accept the same shape of parameters: `total_thickness`, a layer count, a
`stretch_factor` (the geometric growth ratio between one layer and the next, greater than
1), and which faces or edges the layers grow from. `is_ignore=True` on `VLParams` (or
`ignore=True` on the hypothesis) reads that face or edge list as an exclusion instead: layers
grow on every other one.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
