# Block Decomposition

A structured hexahedral mesher (`Hexa3D`) meshes blocks. A block is a solid of 6 faces,
and each face is one wire of 4 edges. A compound of blocks that share their common faces
(glued blocks) meshes as one structured grid. This guide builds block models with GEOM's
block operations, checks them, and finds the chains of edges that take one number of
segments. See [Session](../concepts/session.md) for the full list of block operations.

## A pipe from a divided disk

`divided_disk` adds a disk of quadrangles: an inner square or hexagon and the faces between
it and the circle. A sweep of the disk gives a pipe of blocks.

```python
import numpy as np

from pysmesh import EntityKind, Session

s = Session()
s.divided_disk(1.0, 50.0)                      # 5 quadrangles in the plane z = 0
s.extrude(s.entities(EntityKind.FACE), (0.0, 0.0, 5.0))

check = s.check_blocks()
check.is_compound_of_blocks                    # True: 5 glued blocks
```

`ratio` is GEOM's: the distance of the inner square's corners from the centre, in percent
of the radius. Here the corners are at 0.5 from the axis, so each radial edge is 0.5 long.

## Chains of edges

`propagation_chain(edge)` gives the edges opposite to `edge` through quadrangles, and so on
across the model. A structured mesh puts the same number of segments on every edge of one
chain. Give each chain its own count:

```python
from pysmesh import load_brep
from pysmesh.mesher import (Hexa3D, Mesher, NumberOfSegments, Quadrangle2D, Regular1D,
                            SubShape, SubShapeKind)

edges = s.entities(EntityKind.EDGE)
lengths = s.mass_properties(edges).measure
axial_chain = s.propagation_chain(edges[np.isclose(lengths, 5.0)][0])    # 8 edges
radial_chain = s.propagation_chain(edges[np.isclose(lengths, 0.5)][0])   # 8 edges

handoff = s.export_handoff()
mesher = Mesher(load_brep(handoff.brep))
mesher.assign(Regular1D())
mesher.assign(NumberOfSegments(count=4))
for chain, count in ((axial_chain, 10), (radial_chain, 6)):
    for edge in chain:
        ordinal = int(np.flatnonzero(handoff.edge_id == edge)[0]) + 1
        mesher.assign(NumberOfSegments(count=count), on=SubShape(SubShapeKind.EDGE, ordinal))
mesher.assign(Quadrangle2D())
mesher.assign(Hexa3D())
report = mesher.compute()
report.volumes                                 # (4 * 4 + 4 * 4 * 6) * 10 = 1 120
```

The cross-section has 4 x 4 quadrangles in the square and 4 x 6 in each of the 4 outer
faces. The axial chain gives 10 layers.

## A block and an array of blocks

`make_hexa` builds a block on 6 faces, or between 2 opposite faces. `multi_transform`
copies a block along 1 or 2 translation vectors and glues the copies.

```python
t = Session()
t.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, 1.0)
t.add_rectangle((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 2.0, 1.0)
t.make_hexa(t.entities(EntityKind.FACE))       # the block between the two rectangles
(block,) = t.entities(EntityKind.SOLID)

t.multi_transform(block, [(2.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [3, 2])
len(t.entities(EntityKind.SOLID))              # 6 blocks
len(t.entities(EntityKind.FACE))               # 29 faces: the shared ones once
t.check_blocks().is_compound_of_blocks         # True

t.find_block_near((5.0, 1.5, 0.5))             # the block centred at (5, 1.5, 0.5)
```

The original block keeps its ids. The entities of each copy are generated from the
original's entities.

## A T-shape of pipes

`pipe_t_shape` is GEOM's `MakePipeTShape`: a main pipe along x and a branch along z, cut
into blocks. Its groups name the faces and the edge chains a mesh needs.

```python
p = Session()
tee = p.pipe_t_shape(2.0, 0.5, 6.0, 1.0, 0.4, 5.0)
len(p.entities(EntityKind.SOLID))              # 16 blocks
p.check_blocks().is_compound_of_blocks         # True
[(g.name, len(g.ids)) for g in tee.groups]
# JUNCTION_FACE_1 4, JUNCTION_FACE_2 4, JUNCTION_FACE_3 4, THICKNESS 21,
# CIRCULAR_QUARTER_PIPE 6, CIRCULAR_QUARTER_PIPE 8, HALF_LENGTH_MAIN_PIPE 8, FLANGE 8,
# HALF_LENGTH_INCIDENT_PIPE 8, INTERNAL_FACES 16
```

The groups are GEOM's. The junction faces are the three pipe ends. Each edge group is
one propagation chain. `INTERNAL_FACES` are the faces on the inner cylinders.

## What `check_blocks` finds

`check_blocks` is GEOM's `CheckCompoundOfBlocks`. When the verdict is `False`, each finding
names its kind and the ids of the bodies:

| Kind | Meaning |
|---|---|
| `NOT_BLOCK` | A solid that is not a block (6 faces, each a quadrangle), or a body that is not a solid, face or shell. |
| `EXTRA_EDGE` | A solid that becomes a block when its seam or degenerated edges, or its faces on one surface, are merged. `remove_extra_edges` merges them. |
| `NOT_GLUED` | Two blocks that touch along coincident faces that are not one face. `glue` makes them one face. |
| `NOT_CONNECTED` | A block that no chain of blocks connects to the largest connected set. |

---
*Author: Kajetan R. Gułaj*
*Date: 2026-10-21*
