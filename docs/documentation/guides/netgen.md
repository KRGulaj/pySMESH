# Free Meshing with NETGEN

pySMESH builds netgen 6.2.2101 and SALOME's NETGENPlugin `V9_16_0` into `_core`. They give
free triangles and tetrahedra on any shape, quad-dominant surface meshes, viscous layers
with tetrahedra inside, and a remesher for a triangle surface with no CAD behind it. netgen
is private to the wheel, as OCCT is: no DLL, no file and no console output reach the host.

## The algorithms

| Class | What it meshes | Needs beneath | Hypotheses it reads |
|---|---|---|---|
| `Netgen1D2D3D` | A solid, in one algorithm: segments, triangles, then tetrahedra. | Nothing | `NetgenParameters` or `NetgenSimpleParameters3D`, `ViscousLayers` |
| `Netgen3D` | A solid, by tetrahedra from the mesh of its faces. Where the faces carry quadrangles, pyramids join them to the tetrahedra. | A 2-D algorithm on every face | `NetgenParameters`, `MaxElementVolume`, `ViscousLayers` |
| `Netgen1D2D` | A face, in one algorithm: segments, then triangles (with `quad_allowed`, a quad-dominant mesh). | Nothing | `NetgenParameters2D` or `NetgenSimpleParameters2D`, `ViscousLayers2D` |
| `Netgen2D` | A face, by triangles from the segments of its edges. | A 1-D algorithm on its edges | `MaxElementArea`, `LengthFromEdges`, `QuadranglePreference`, `NetgenParameters2D`, `ViscousLayers2D` |
| `NetgenRemesher2D` | The triangles of a mesher with no shape, meshed again. | A surface mesh from arrays | `NetgenRemesherParameters2D` |

A sub-shape that another algorithm meshes keeps its mesh. `Regular1D` on an edge beside
`Netgen1D2D3D` keeps its segments, and NETGEN builds the faces on its nodes. `Quadrangle2D`
on a face below `Netgen3D` keeps its quadrangles.

## Sizes

`NetgenParameters` (3-D) and `NetgenParameters2D` (2-D) share their sizing fields:

- `max_size` bounds every element edge. `min_size` stops the refinement; 0 sets no limit.
- `fineness` is a preset of the growth rate, the segments per edge and the segments per
  radius of curvature (see `Fineness`). `Fineness.USER_DEFINED` takes the three values from
  `growth_rate`, `segments_per_edge` and `segments_per_radius`. Each of the three fields is
  refused with any other preset.
- `chordal_error` sizes a curved face from its curvature. It is a target: the mean distance
  between the triangles and the face keeps within it, and single triangles can exceed it.
- `local_sizes` gives a size near a vertex, along an edge, on a face or in a solid, as
  `(SubShape, size)` pairs.
- `second_order` makes quadratic elements, with every mid-edge node on the geometry.

```python
from pysmesh import Session, load_brep
from pysmesh.mesher import Fineness, Mesher, Netgen1D2D3D, NetgenParameters

s = Session()
s.add_cylinder(1.0, 3.0)
shape = load_brep(s.brep())

with Mesher(shape) as mesher:
    mesher.assign(Netgen1D2D3D())
    mesher.assign(NetgenParameters(max_size=0.4, fineness=Fineness.MODERATE))
    report = mesher.compute()    # 199 nodes, 336 triangles, 583 tetrahedra
```

A local size refines one sub-shape. The size then grows away from it by at most the growth
rate per unit length:

```python
from pysmesh import SubShape, SubShapeKind

s = Session()
s.add_box(4.0, 4.0, 4.0)
cube = load_brep(s.brep())

with Mesher(cube) as mesher:
    mesher.assign(Netgen1D2D3D())
    mesher.assign(
        NetgenParameters(
            max_size=1.0,
            local_sizes=((SubShape(SubShapeKind.FACE, 1), 0.1),),
        )
    )
    report = mesher.compute()    # 22 354 tetrahedra, fine at face 1, coarse away from it
```

The short forms `NetgenSimpleParameters2D` and `NetgenSimpleParameters3D` give each edge a
segment count or a segment length, and each face an area bound or a size taken from its
edges.

## Layers

`ViscousLayers` beside `Netgen1D2D3D` or `Netgen3D` grows the stack on the chosen faces.
NETGEN then fills the rest of the solid with tetrahedra. Each wall triangle gives one prism
per layer, and the layer nodes lie at the closed-form depths of the stack:

```python
from pysmesh.mesher import ViscousLayers

s = Session()
s.add_box(2.0, 3.0, 4.0)
box = load_brep(s.brep())

with Mesher(box) as mesher:
    mesher.assign(Netgen1D2D3D())
    mesher.assign(NetgenParameters(max_size=0.5))
    mesher.assign(
        ViscousLayers(
            total_thickness=0.3,
            layer_count=3,
            stretch_factor=1.2,
            boundary=(5,),           # the face z = 0
            group_name="layers",
        )
    )
    report = mesher.compute()    # 1 633 cells, 162 of them prisms
    prisms = mesher.group("layers").element_ids
```

`ViscousLayers2D` beside `Netgen1D2D` or `Netgen2D` grows quadrangle layers on the chosen
edges of a face. Several `ViscousLayers` on one solid each grow their own stack. See
[Viscous boundary layers](mesh-editing.md#viscous-boundary-layers) for the stack, the
three paths and the limits.

## The remesher

A mesher with no shape takes one algorithm: `NetgenRemesher2D`, on the whole mesh. NETGEN
reads the triangles as an STL surface (a quadrangle as two triangles). It splits the
surface into charts at its feature edges and meshes each chart again. The result replaces
the mesh:

```python
import numpy as np

from pysmesh.mesher import NetgenRemesher2D, NetgenRemesherParameters2D

xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
tris = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], dtype=np.int64)

with Mesher.from_arrays(xyz, tris) as mesher:
    mesher.assign(NetgenRemesher2D())
    mesher.assign(NetgenRemesherParameters2D(max_size=0.1))
    report = mesher.compute()    # 618 triangles, 78 segments on the six feature edges
```

- Without `NetgenRemesherParameters2D`, the size is the diagonal of the bounding box
  divided by 10.
- `ridge_angle` decides which edges are feature edges: the edges where the normals of the
  two triangles differ by more than that angle. The new mesh keeps the feature edges.
- `fixed_edges` names an EDGE group of the mesher. Each node of the group that lies on a
  feature edge stays a node of the new mesh.
- `make_groups_of_surfaces` gives one face group per chart.
- The remesher refuses a non-manifold surface (three triangles on one edge) with netgen's
  reason. It refuses faces with no area before netgen runs.
- The same input gives the same mesh in every process and on every repeat.

## Threads, cancel and output

- `threads` on the parameters sets how many threads netgen may use. None (the default)
  uses one per hardware thread. The mesh does not depend on it: 1, 4 and 16 threads give
  the same mesh, bit for bit. Two `Mesher` objects can compute with NETGEN on two threads
  at once.
- `compute(cancel=...)` passes the cancel to netgen, which checks it between its steps.
  The longest measured wait is 2.2 s, early in volume meshing. After a cancel the mesh is
  empty, and the same mesher computes again. A cancelled remesh leaves the input mesh, or
  the whole remesh if the cancel came after the remesh replaced it. The details of the
  `PysmeshCancelled` say which.
- A NETGEN compute writes nothing to stdout or stderr. It creates no file and keeps the
  working directory. A failure raises `PysmeshError` with netgen's own reason.

## Limits

- netgen 6.2.2101 fails on a unit sphere at `Fineness.COARSE` with `max_size` 0.866:
  "Problem in Surface mesh generation", and the plugin adds "Intersecting triangles".
  `Fineness.MODERATE` and finer mesh it. A sphere of radius 1.7 meshes at every preset.
- NETGEN makes no hexahedral volume mesh. For hexahedra, use the structured algorithms or
  `Cartesian3D` (see [Meshing model](../concepts/meshing-model.md)).

---
*Author: Kajetan R. Gułaj*
*Date: 2026-10-05*
