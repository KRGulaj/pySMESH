# Getting Started

This guide installs pySMESH and walks through a complete example: build a box, round its
edges, hand the shape to a mesher, assign a structured hexahedral mesh, compute it, and read
the result back. Every step is explained as it happens.

## Install

```bash
pip install pysmesh
```

That is the whole procedure. The wheel is self-contained: SMESH, OCCT, Boost and VTK all
ship inside it, name-mangled so they never collide with anything in your own environment.
NumPy is the only package pip pulls in. pySMESH targets Windows x64, CPython 3.11 to 3.14.

## Step 1: build a shape

`Session` is the stateful CAD modelling class. It owns one live shape and gives every
entity a persistent id across every edit:

```python
from pysmesh import Session

s = Session()
s.add_box(3.0, 7.0, 11.0)
```

`add_box(dx, dy, dz)` adds an axis-aligned box of those extents, with its minimum corner at
the origin by default. The box, its 6 faces, its 12 edges, and its 8 vertices are all newly
issued entities in this session.

## Step 2: edit the shape

Round every edge with a 0.5-unit fillet:

```python
from pysmesh.session import EntityKind

s.fillet(edge_ids=s.entities(EntityKind.EDGE), radius=0.5)
```

`s.entities(EntityKind.EDGE)` lists every live edge id. Filleting an edge kills that edge's
id and adds a new face (the round) plus new edges bounding it. See
[Entity IDs and ordinals](../concepts/entity-ids.md) for exactly what survives an edit and
what does not.

Snapshot the shape before an edit whose outcome you might want to undo. Both operations are
O(1), because a snapshot copies handles, not geometry:

```python
mark = s.snapshot()
# s.restore(mark) rewinds to this point, at any point later
```

## Step 3: hand the shape to a mesher

A `Session` never meshes. `export_handoff()` crosses that boundary once, returning the shape
as BREP bytes plus the session id of every sub-shape in it:

```python
import pysmesh

handoff = s.export_handoff()
shape = pysmesh.load_brep(handoff.brep)
```

`pysmesh.load_brep` reads the bytes back into a `Shape`, whose faces, edges, solids and
vertices are numbered by 1-based positional ordinal, the id space every stateless pySMESH
function uses. See [Session](../concepts/session.md#reaching-a-mesher) for how
`handoff.face_id` and the rest pair those ordinals back to the session's own ids.

## Step 4: assign algorithms and hypotheses

`Mesher` builds a mesh by assigning an algorithm and its hypotheses to the shape, or to one
of its sub-shapes. This box gets a uniform structured hexahedral mesh: 8 segments per edge,
quadrangles on every face, hexahedra filling the solid.

```python
from pysmesh.mesher import Hexa3D, Mesher, NumberOfSegments, Quadrangle2D, Regular1D

mesher = Mesher(shape)
mesher.assign(Regular1D())                 # 1-D: discretise every edge
mesher.assign(NumberOfSegments(count=8))   # 1-D hypothesis: 8 segments per edge
mesher.assign(Quadrangle2D())               # 2-D: mesh every face from its edges
mesher.assign(Hexa3D())                     # 3-D: mesh every solid from its faces
```

Each `assign` call with no `on` argument governs the whole shape, which is what makes this a
uniform mesh. Read [Meshing model](../concepts/meshing-model.md) for the dimension hierarchy
this recipe follows, and for how to override an assignment on one sub-shape only.

## Step 5: compute

```python
report = mesher.compute()

report.nodes     # node count of the whole mesh
report.volumes   # 3-D element count
```

`compute()` returns a `ComputeReport`. If any sub-mesh fails, `compute()` raises
`PysmeshError` instead, naming every failed sub-shape with SMESH's own reason. See
[Meshing model](../concepts/meshing-model.md#reading-compute) for how to read that failure.

## Step 6: read the mesh back

```python
mesh = mesher.mesh()

mesh.node_count       # number of nodes
mesh.element_count    # edges + faces + volumes together
```

`mesh` is a `MeshData`: a `node_coords` array, a compressed element list, and the sub-shape
each node and element is bound to. Nothing here is a VTK object; see
[VTK privacy](../concepts/vtk-privacy.md) for why that is a load-bearing guarantee, not an
implementation detail.

## Step 7: check quality

```python
from pysmesh.mesher import AspectRatio3D

quality = mesher.quality(AspectRatio3D())
quality.values.max()   # the worst cell's aspect ratio; 1.0 is a perfect cube
```

`mesher.quality(control)` evaluates one measure over every element it applies to. See
[Mesh editing](mesh-editing.md) for the rest of the quality, group, editing and search
surface once a mesh exists.

## The whole example in one block

```python
import pysmesh
from pysmesh import Session
from pysmesh.mesher import AspectRatio3D, Hexa3D, Mesher, NumberOfSegments, Quadrangle2D, Regular1D
from pysmesh.session import EntityKind

s = Session()
s.add_box(3.0, 7.0, 11.0)
s.fillet(edge_ids=s.entities(EntityKind.EDGE), radius=0.5)

handoff = s.export_handoff()
shape = pysmesh.load_brep(handoff.brep)

mesher = Mesher(shape)
mesher.assign(Regular1D())
mesher.assign(NumberOfSegments(count=8))
mesher.assign(Quadrangle2D())
mesher.assign(Hexa3D())
report = mesher.compute()

mesh = mesher.mesh()
print(mesh.node_count, mesh.element_count)
print(mesher.quality(AspectRatio3D()).values.max())
```

## Where to go next

- [Entity IDs and ordinals](../concepts/entity-ids.md) explains the id scheme this example
  relies on.
- [Meshing model](../concepts/meshing-model.md) covers every algorithm and hypothesis, and
  how to mesh different parts of a shape differently.
- [Discrete meshes](../concepts/discrete-meshes.md) covers meshing a body with no CAD
  behind it at all.
- [Geometry operations](geometry-operations.md) covers STEP and IGES import and export,
  tessellation, and the standalone OCCT surface.
- [Mesh editing](mesh-editing.md) covers what to do once a mesh exists: quality, groups,
  editing, search, and viscous layers.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
