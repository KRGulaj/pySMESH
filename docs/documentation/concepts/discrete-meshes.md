# Discrete Meshes

A `Mesher` does not need a B-rep. `Mesher()` starts empty and is filled from arrays, which
is how a body that never had CAD geometry behind it reaches the rest of pySMESH: an
imported STL, OBJ or PLY, a shrink-wrap result, the boundary another mesher produced, or a
mesh read back from a file with `read_gmf`. This page explains why that path exists, how to
fill a mesher from arrays, how to divide such a mesh into addressable regions with no
sub-shape ordinals to use, and exactly what such a mesher cannot do.

## Why it exists

Everything else in pySMESH that names a sub-shape does so through a `Shape`'s 1-based
`TopExp` ordinals, which come from a B-rep. A triangle soup has no B-rep. Before the fill
path existed, there was no way to hand such a mesh to a `Mesher` at all: no quality
controls, no editor, no groups, no search. `Mesher.from_arrays` and `Mesher.from_mesh` close
that gap, and once filled, the whole editing and search surface applies to the result.

## Filling a mesher

```python
import numpy as np
from pysmesh import Mesher

points = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    dtype=np.float64,
)
triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

mesher = Mesher.from_arrays(points, triangles)
```

`Mesher.from_arrays` is the one-call form of `Mesher()` followed by `add_nodes` and
`add_elements`. Its `elements` argument holds **0-based row indices** into `node_coords`,
not node ids, because at that point no ids exist yet. That is also how a harvest expresses
connectivity: `MeshData.element_nodes` is row-indexed too, so an array taken from
`Mesher.mesh()` fits straight back in.

`Mesher.add_elements` on an already-populated mesher is different: it takes **node ids**,
what `add_nodes` returned. The two conventions are not interchangeable, and mixing them up
does not raise; it builds elements on the wrong nodes.

`Mesher.from_mesh(mesh_data)` is the way back from arrays to a live mesh, keeping every id
rather than reassigning it. This is the path from `read_gmf`, which reads a file into
arrays and drops the CAD binding, back to a mesher that can be edited, queried and grouped.

## `has_shape`

```python
mesher.has_shape   # False for Mesher() or Mesher.from_arrays(...)
```

`has_shape` is the query for whether a mesher carries sub-shape ordinals at all. It is
`False` for a mesher built with `shape=None`, whether directly, through `from_arrays`, or
through `from_mesh`.

## The patch workflow

Without CAD faces, a discrete mesh has no sub-shapes to assign, group, or select by. The
substitute is a two-step workflow: find the creases, then partition by them.

```python
edges = mesher.sharp_edges(angle=40.0)
patches = mesher.separate_faces_by_edges(edges, name_prefix="patch_")
```

`sharp_edges(angle)` finds every edge of the surface mesh where two faces meet at an angle
of at least `angle` degrees, plus every non-manifold edge. `separate_faces_by_edges` then
partitions the mesh's faces into the regions those edges bound. This is what a viewport
picks and hides on when there is no CAD face to pick instead.

### A patch index is not stable. A patch group is.

Each call to `separate_faces_by_edges` re-derives the partition from scratch. Once faces
have been deleted, the same region can come back under a different index on the next call.
Passing `name_prefix` stores each patch as an explicit group named `f"{name_prefix}{i}"`,
and SMESH maintains that membership itself from then on: a deleted element leaves the group
it was in, and a survivor keeps its place. Read the groups back with `mesher.groups()`.

```python
gone = mesher.remove_elements(patches.at(2), free_nodes=True)
print(gone.elements, gone.nodes)   # exactly what went, including the freed nodes

# "patch_2" is still that same region, even though patches.at(2) may now be a
# different index on the next call to separate_faces_by_edges.
regrouped = {g.name: g.element_ids for g in mesher.groups()}
```

## What a shape-free mesher cannot do

Every operation that resolves a sub-shape ordinal refuses on a mesher with `has_shape`
`False`. Each one names itself in the error rather than failing somewhere further in:

| Operation | Why it needs a shape |
|---|---|
| `Mesher.compute` | Runs the assigned algorithms over the geometry |
| `Mesher.assign` / `Mesher.unassign` | Attaches an algorithm or hypothesis to a sub-shape |
| `Mesher.add_group_on_shape` | Names a group by a sub-shape's ordinal |
| `Mesher.pattern_from_face` / `apply_pattern_to_face` / `apply_pattern_to_block` | Reads or maps a pattern against a face's or block's geometry |
| `Mesher.smooth(on_shape=True)` | Moves nodes in the parameter space of the face each one sits on |
| The `ElementsOnShape` control | Classifies an element by which sub-shape it lies on |
| The `Deflection2D` control | Measures a face's distance from the CAD surface it lies on |

```python
import pysmesh
from pysmesh.mesher import Regular1D

try:
    mesher.assign(Regular1D())
except pysmesh.PysmeshError as exc:
    print(exc)   # "Mesher.assign: ... has no shape"
```

Everything else works identically whether or not the mesher has a shape: the editor, the
search surface (`sharp_edges`, ray casting, point classification, closest element), the
quality controls that do not read the geometry, and groups by id or by filter. This is what
makes the discrete path a first-class citizen rather than a reduced one.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
