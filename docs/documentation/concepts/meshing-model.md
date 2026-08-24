# Meshing Model

`pysmesh.Mesher` exposes SMESH's own algorithm and hypothesis assignment model. This is the
single most important concept for using the mesher correctly. Get the dimension hierarchy
wrong and `compute()` either refuses outright or, worse, silently meshes less of the model
than intended. This page explains the model, gives the algorithm and hypothesis catalogue
per dimension, and explains how to read what `compute()` returns.

## Algorithms and hypotheses are two different things

SMESH separates *how* a sub-shape is meshed from the numbers that control it.

- An **algorithm** decides how a sub-shape is meshed. It takes no parameters of its own.
- A **hypothesis** supplies a number the algorithm beside it reads: a segment count, a
  maximum element area, a layer thickness. Several hypotheses can sit beside one algorithm.

Both are frozen dataclasses in `pysmesh.mesher`, and `Mesher.assign(item, on=None)` attaches
either kind to a sub-shape. `on=None` assigns to the whole shape, which is the model's
default; naming a sub-shape with `on=SubShape(kind, ordinal)` overrides the default there.

## The dimension hierarchy

A mesh is built bottom-up. A 2-D algorithm normally needs a 1-D layer beneath it, because it
meshes a face's interior from the discretisation already on that face's edges. A 3-D
algorithm normally needs a 2-D layer beneath it for the same reason: it fills a solid's
interior from the mesh already on that solid's faces.

```python
from pysmesh import load_brep, Session
from pysmesh.mesher import Hexa3D, Mesher, NumberOfSegments, Quadrangle2D, Regular1D

session = Session()
session.add_box(3.0, 7.0, 11.0)

mesher = Mesher(load_brep(session.brep()))
mesher.assign(Regular1D())                # 1-D: discretise every edge
mesher.assign(NumberOfSegments(count=8))  # 1-D hypothesis: 8 segments per edge
mesher.assign(Quadrangle2D())              # 2-D: mesh every face from its edges
mesher.assign(Hexa3D())                    # 3-D: mesh every solid from its faces
mesher.compute()
```

**Three algorithms break that rule on purpose**, because they mesh every dimension of their
sub-shape themselves: `Cartesian3D`, `PolyhedronPerSolid3D`, and `Prism3D` (for its lateral
faces and edges; the source face beneath it still needs its own 2-D algorithm). A
lower-dimension algorithm assigned beside one of these three is accepted, not refused, and
then **hidden**: it has no effect where the all-dimensional algorithm governs. SMESH treats
this as a normal state, because refusing it would break the ordinary pattern of setting a
model-wide default and overriding it on one solid. Read `ComputeReport.meshed` after
`compute()` to see which sub-shapes actually received elements from which assignment.

## The catalogue, by dimension

Every entry below is verified against SMESH's own hypothesis compatibility, either from the
native `StdMeshers` source or from a test that computes a real mesh with it.

### 1-D algorithms

| Algorithm | What it does | Hypotheses it reads |
|---|---|---|
| `Regular1D` | Discretises every edge it governs, spaced by whichever 1-D hypothesis applies there. The usual base of any assignment. | `NumberOfSegments`, `Arithmetic1D`, `StartEndLength`, `Geometric1D`, `FixedPoints1D`, `Adaptive1D`, `AutomaticLength`, `Deflection1D`, `LocalLength`, `MaxLength`, `SegmentLengthAroundVertex` (vertex-scoped), `Propagation` (edge-scoped) |
| `CompositeSegment1D` | Discretises a chain of C1-continuous edges as if it were one edge. Useful where an import split one geometric curve into several edges. | The same 1-D hypotheses as `Regular1D`, applied to the whole chain |
| `Projection1D` | Copies an edge's discretisation from another edge. | `ProjectionSource1D` (required) |

### 2-D algorithms

| Algorithm | What it does | Needs beneath | Hypotheses it reads |
|---|---|---|---|
| `Quadrangle2D` | Mapped quadrangle meshing of a face bounded by four logical sides. Refuses a face it cannot read as four sides. | A 1-D algorithm and hypothesis on its edges | `QuadrangleParams` (base vertex, corner vertices, how to resolve mismatched sides), `QuadranglePreference` |
| `Mefisto2D` | Free triangle meshing of a face. | A 1-D algorithm and hypothesis on its edges | `MaxElementArea` (a bound, not a target: it only binds where the boundary would otherwise produce larger elements) |
| `PolygonPerFace2D` | One polygonal element per face, using the edge discretisation directly as its boundary. | A 1-D algorithm and hypothesis on its edges | None |
| `Projection2D` | Copies a face's mesh from another face. This is how a periodic pair is made to match node for node. | A 1-D algorithm and hypothesis on its own edges, matching the source face's edge counts | `ProjectionSource2D` (required) |
| `Projection1D2D` | Projects a face's mesh **and** its boundary discretisation from another face. | Nothing: it supplies its own 1-D layer from the source | `ProjectionSource2D` (required) |
| `QuadFromMedialAxis1D2D` | Quad-dominant meshing of a thin face, built on its medial axis. The only algorithm in the catalogue that reports true progress. | A 1-D algorithm and hypothesis on its edges | None beyond the 1-D layer |
| `RadialQuadrangle1D2D` | Radial quadrangle meshing of a disk or an annulus. | A 1-D algorithm and hypothesis on the boundary edge | `NumberOfLayers2D`, or a 1-D hypothesis applied to the radial direction |

### 3-D algorithms

| Algorithm | What it does | Needs beneath | Hypotheses it reads |
|---|---|---|---|
| `Cartesian3D` | Body-fitted Cartesian volume meshing: a regular grid, cut against the geometry at the boundary. Hexahedra inside, polyhedra at every cut cell. Meshes every dimension itself; hides any lower-dimension algorithm. Its polyhedra cannot be written to Inria `.mesh`. | Nothing | `CartesianParameters3D` |
| `Hexa3D` | Structured hexahedral meshing of a block: a solid bounded by six logical faces. Consumes the 2-D mesh below it. | A conforming quadrangle mesh on its six logical faces | None |
| `CompositeHexa3D` | Structured hexahedral meshing of a solid whose six logical sides are each split into more faces. The counterpart of `Hexa3D` for such an import. | The same conforming quadrangle mesh `Hexa3D` needs, split across more faces | None |
| `HexaFromSkin3D` | Fills a solid with hexahedra derived from an existing all-quadrangle surface mesh. | An existing all-quadrangle mesh on the solid's skin | None |
| `Prism3D` | Extrudes a source face's mesh through a prismatic solid. Meshes the lateral faces and edges itself. | A 1-D and 2-D algorithm on the source face only | None of its own |
| `RadialPrism3D` | An O-grid between an inner and an outer shell: a pipe wall, an annulus. Needs the two shells' meshes to already match, typically via `Projection2D`. | Matching 2-D meshes on the inner and outer shell | `NumberOfLayers` or `LayerDistribution` |
| `Projection3D` | Copies a solid's mesh from another solid. | Nothing beyond the source solid's own mesh | `ProjectionSource3D` (required) |
| `PolyhedronPerSolid3D` | One polyhedral element per solid, from the face mesh bounding it. Meshes every dimension itself; hides a lower-dimension algorithm beside it. Unlike `Cartesian3D`, it does consume an existing boundary mesh where one is present. | Nothing required; uses a boundary mesh if present | None |

### Hypotheses that name another part of the model

`ProjectionSource1D`, `ProjectionSource2D` and `ProjectionSource3D` each carry a `SubShape`
naming the edge, face or solid to copy from, and optional vertex pairs to pin the
correspondence. Without the vertex pairs, the algorithm picks a correspondence itself, which
is fine for a face with one obvious mapping and wrong for a periodic pair where the wrong
choice is a rotated mesh.

### Whole-mesh switches

`QuadraticMesh` assigned anywhere produces second-order elements instead of linear ones. It
changes what the algorithms build, so it is not the same as converting an existing linear
mesh in place with `Mesher.convert_to_quadratic`.

## A verified worked example: an O-grid

This is the recipe a test in this repository computes and checks. It builds a solid between
two concentric shells (a hollow sphere, made by cutting a small sphere out of a bigger one),
free-meshes the outer shell, projects that mesh onto the inner shell so the two match node
for node, then fills the wall radially. `RadialPrism3D` refuses two shells whose meshes do
not already match, which is why the projection step is not optional here.

```python
from pysmesh import load_brep, Session
from pysmesh.mesher import (
    Mefisto2D, MaxElementArea, Mesher, NumberOfLayers, NumberOfSegments,
    Projection2D, ProjectionSource2D, RadialPrism3D, Regular1D, SubShape, SubShapeKind,
)
from pysmesh.session import EntityKind

session = Session()
session.add_sphere(3.0)
outer_solid = list(session.entities(EntityKind.SOLID))
session.add_sphere(2.0)
inner_solid = [e for e in session.entities(EntityKind.SOLID) if e not in outer_solid]
session.cut(outer_solid, inner_solid)

shape = load_brep(session.brep())
areas = sorted((face.area, ordinal) for ordinal, face in enumerate(shape.faces(), 1))
outer = SubShape(SubShapeKind.FACE, areas[-1][1])  # the larger face, by area
inner = SubShape(SubShapeKind.FACE, areas[0][1])   # the smaller face, by area

mesher = Mesher(shape)
mesher.assign(Regular1D())
mesher.assign(NumberOfSegments(count=6))
mesher.assign(Mefisto2D(), on=outer)
mesher.assign(MaxElementArea(max_area=2.0), on=outer)
mesher.assign(Projection2D(), on=inner)
mesher.assign(ProjectionSource2D(source_face=outer), on=inner)
mesher.assign(RadialPrism3D())
mesher.assign(NumberOfLayers(count=4))
report = mesher.compute()
```

## Assigning to a sub-shape

`SubShape(kind, ordinal)` names one sub-shape the same way the stateless geometry API does:
`kind` is a `SubShapeKind` (`SOLID`, `FACE`, `EDGE`, `VERTEX`), and `ordinal` is the 1-based
rank in that kind's traversal, exactly as `Shape.faces()` and the rest number them. See
[Entity IDs and ordinals](entity-ids.md) for what that ordinal is and is not stable across.

```python
from pysmesh.mesher import Hexa3D, SubShape, SubShapeKind

mesher.assign(Hexa3D(), on=SubShape(SubShapeKind.SOLID, 1))
```

## Reading `compute()`

`compute()` returns a `ComputeReport`:

```python
report = mesher.compute()
report.nodes     # node count of the whole mesh
report.edges     # 1-D element count
report.faces     # 2-D element count
report.volumes   # 3-D element count
report.meshed    # one SubMeshCount per sub-shape that received elements
```

`report.meshed` is what tells "meshed by the algorithm I put there" from "meshed by an
enclosing all-dimensional algorithm that hid it". Read it whenever a mixed assignment is in
play.

**A failure names every failed sub-shape.** `compute()` raises `PysmeshError` if any
sub-mesh failed. The message carries SMESH's own reason plus the algorithm that reported it,
for every failed sub-shape, and `.face_ids` carries the ordinals of the failed faces. The
partial mesh is **kept**, not cleared, because how far the assignment got is itself the
diagnostic:

```python
import pysmesh

try:
    mesher.compute()
except pysmesh.PysmeshError as exc:
    print(exc.details)     # SMESH's own error text
    print(exc.face_ids)    # which faces failed
    partial = mesher.mesh()  # whatever was built before the failure
```

Cancellation is different from failure: if `cancel` returns `True`, or `progress` raises,
`compute()` raises `PysmeshCancelled` and the mesh is cleared, so nothing partial survives.

**Progress is exact only at sub-mesh granularity.** The fraction of sub-meshes already done
is real. Inside one running algorithm, SMESH interpolates with a tick counter, so an
algorithm that meshes the whole model in one call (`Cartesian3D`, for instance) reports
values that creep up from near zero and jump to 1.0 at the end. Only
`QuadFromMedialAxis1D2D` reports its own true fraction. Cancellation is not preemptive
either: only `Cartesian3D`, `Prism3D`, and the algorithm driven by `Adaptive1D` poll it
inside their own loop; every other algorithm can be stopped only between sub-meshes.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
