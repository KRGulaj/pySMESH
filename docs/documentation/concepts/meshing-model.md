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

**Five algorithms break that rule on purpose**, because they mesh every dimension of their
sub-shape themselves: `Cartesian3D`, `PolyhedronPerSolid3D`, `Netgen1D2D3D`, `Netgen1D2D`
and `Prism3D` (for its lateral faces and edges; the source face beneath it still needs its
own 2-D algorithm). A lower-dimension algorithm assigned beside one of these five is
accepted, not refused, and then **hidden**: it has no effect where the all-dimensional
algorithm governs. A global `Regular1D` with 3 segments beside a global `Netgen1D2D3D`
leaves NETGEN's own segments on every edge. The two NETGEN algorithms keep the mesh of
an algorithm assigned on the edge or the face itself. SMESH treats
this as a normal state, because refusing it would break the ordinary pattern of setting a
model-wide default and overriding it on one solid. Read `ComputeReport.meshed` after
`compute()` to see which sub-shapes actually received elements from which assignment.

## The catalogue, by dimension

Every entry below is verified against SMESH's own hypothesis compatibility, either from the
native `StdMeshers` source or from a test that computes a real mesh with it.

### 0-D algorithms

| Algorithm | What it does | Hypotheses it reads |
|---|---|---|
| `SegmentAroundVertex0D` | Assigned on a vertex. Meshes nothing itself: it makes the 1-D algorithm of each edge at that vertex give the segment touching it the length of `SegmentLengthAroundVertex`. | `SegmentLengthAroundVertex` (on the same vertex) |

### 1-D algorithms

| Algorithm | What it does | Hypotheses it reads |
|---|---|---|
| `Regular1D` | Discretises every edge it governs, spaced by whichever 1-D hypothesis applies there. The usual base of any assignment. | `NumberOfSegments`, `Arithmetic1D`, `StartEndLength`, `Geometric1D`, `FixedPoints1D`, `Adaptive1D`, `AutomaticLength`, `Deflection1D`, `LocalLength`, `MaxLength`, `SegmentLengthAroundVertex` (vertex-scoped, read only with `SegmentAroundVertex0D` on the vertex), `Propagation` and `PropagOfDistribution` (edge-scoped: the first carries the hypothesis to the opposite edges, the second its node fractions) |
| `CompositeSegment1D` | Discretises a chain of C1-continuous edges as if it were one edge. Useful where an import split one geometric curve into several edges. | The same 1-D hypotheses as `Regular1D`, applied to the whole chain |
| `Projection1D` | Copies an edge's discretisation from another edge. | `ProjectionSource1D` (required) |
| `UseExisting1D` | Takes the segments a script made on the edge, with `Mesher.add_nodes` and `Mesher.add_segments` bound to it by `on`, as the edge's mesh. Creates nothing itself. | None |

### 2-D algorithms

| Algorithm | What it does | Needs beneath | Hypotheses it reads |
|---|---|---|---|
| `Quadrangle2D` | Mapped quadrangle meshing of a face bounded by four logical sides. Refuses a face it cannot read as four sides. | A 1-D algorithm and hypothesis on its edges | `QuadrangleParams` (base vertex, corner vertices, how to resolve mismatched sides), `QuadranglePreference`, `ViscousLayers2D` |
| `Mefisto2D` | Free triangle meshing of a face. | A 1-D algorithm and hypothesis on its edges | `MaxElementArea` (a bound, not a target: it refines the face below the boundary segments where it asks for that, and never coarsens it past the longest one; where the boundary segments are too long for the bound, the compute reports a `ComputeWarning` naming the largest triangle area), `LengthFromEdges` (the mean boundary segment as the target edge length; the default when no 2-D hypothesis applies), `ViscousLayers2D` |
| `PolygonPerFace2D` | One polygonal element per face, using the edge discretisation directly as its boundary. With `ViscousLayers2D`, layer quadrangles on the chosen edges and one polygon for the rest. | A 1-D algorithm and hypothesis on its edges | `ViscousLayers2D` |
| `Projection2D` | Copies a face's mesh from another face. This is how a periodic pair is made to match node for node. | A 1-D algorithm and hypothesis on its own edges, matching the source face's edge counts | `ProjectionSource2D` (required) |
| `Projection1D2D` | Projects a face's mesh **and** its boundary discretisation from another face. | Nothing: it supplies its own 1-D layer from the source | `ProjectionSource2D` (required) |
| `QuadFromMedialAxis1D2D` | Quad-dominant meshing of a thin face, built on its medial axis. The only algorithm in the catalogue that reports true progress. | A 1-D algorithm and hypothesis on its edges | `ViscousLayers2D` |
| `UseExisting2D` | Takes the faces a script made on the face, bound to it by `on`, as the face's mesh. Creates nothing itself. | Nodes and faces made by a script | None |
| `Netgen1D2D` | Free triangle meshing of a face and its edges by NETGEN, in one algorithm. With `quad_allowed`, a quad-dominant mesh. An edge with its own 1-D algorithm keeps that mesh. | Nothing: it meshes the edges itself | `NetgenParameters2D` or `NetgenSimpleParameters2D`, `ViscousLayers2D` |
| `Netgen2D` | Free triangle meshing of a face by NETGEN, from the segments of its edges. | A 1-D algorithm and hypothesis on its edges | One of `MaxElementArea`, `LengthFromEdges` and `NetgenParameters2D` per face (none: the size comes from the boundary segments), `QuadranglePreference` (not with `NetgenParameters2D`), `ViscousLayers2D` |
| `NetgenRemesher2D` | Meshes the triangles of a mesher with no shape again (see [Discrete meshes](discrete-meshes.md)). The only algorithm such a mesher takes. | A surface mesh from arrays | `NetgenRemesherParameters2D` |
| `RadialQuadrangle1D2D` | Radial quadrangle meshing of a disk or an annulus. | A 1-D algorithm and hypothesis on the boundary edge | `NumberOfLayers2D`, `LayerDistribution2D` (a 1-D hypothesis laid along the radius from the curve inward), or a 1-D hypothesis applied to the radial direction |

### Three 2-D limits, measured

The measurements are on SMESH 9.16. The first two are on a planar NACA 0012 cap with a
1.25 m chord and a sharp trailing edge, whose wedge angle is 16.54 degrees.

- `Quadrangle2D` maps the face from four corners. On the cap split into 4 edges (trailing
  edge, mid-upper, leading edge, mid-lower), three of the corners lie on smooth curves, so
  the map shears the cells. The minimum angle is 2.207 degrees with 30 segments on every
  edge, 0.347 degrees with them clustered toward the ends, and 0.006 degrees with 40 aft
  and 30 fore segments under `QUADRANGLE_PREFERENCE`. With 1 or 2 edges it refuses: "Face
  must have 4 sides but not 1" (`StdMeshers_Quadrangle_2D.cxx:1538`).
- `QuadFromMedialAxis1D2D` meshes a ring, or a thin strip with two short ends, like a river
  between its banks (`getSinuousEdges`, `StdMeshers_QuadFromMedialAxis_1D2D.cxx:501`).
  The cap has a cusp at one end and a round nose at the other, so it fails with 1, 2 or 4
  edges and 4 or 8 layers: "Not implemented so far" (`:2205`).

The way out for such a face is a better block topology: for example, a C-shaped strip
along the camber line plus a nose block. Or mesh it with triangles.

The third is `Mefisto2D` on a plain face. Its own quality step (`teamqt`,
`mefisto2/trte.c:4903`) can leave slivers beside a boundary edge. On the 4 x 4 square with
16 segments per side, sized by its boundary, 20 of 494 triangles are below 5 degrees and the
smallest is 0.99 degrees. Run `Mesher.smooth` after `Mefisto2D`. One pass lifts the
smallest angle there to 15.45 degrees (Laplacian) or 24.93 degrees (centroidal), and no
triangle stays below 5 degrees. After 3 passes the smallest angle is 21.35 degrees
(Laplacian) or 29.13 degrees (centroidal).

### 3-D algorithms

| Algorithm | What it does | Needs beneath | Hypotheses it reads |
|---|---|---|---|
| `Cartesian3D` | Body-fitted Cartesian volume meshing: a regular grid, cut against the geometry at the boundary. Hexahedra inside, polyhedra at every cut cell. Meshes every dimension itself; hides any lower-dimension algorithm. Its polyhedra cannot be written to Inria `.mesh`. With `ViscousLayers` it grows prism layers on the chosen faces: it meshes the shape shrunk by the layer thickness and fills the gap with layer cells (see the viscous layer section of the mesh editing guide). | Nothing | `CartesianParameters3D`, `ViscousLayers` |
| `Hexa3D` | Structured hexahedral meshing of a block: a solid bounded by six logical faces. Consumes the 2-D mesh below it. | A conforming quadrangle mesh on its six logical faces | `BlockRenumber` (hexahedra and nodes in i, j, k order; axes global by default, or set per block by two vertices), `ViscousLayers` (one per solid) |
| `CompositeHexa3D` | Structured hexahedral meshing of a solid whose six logical sides are each split into more faces. The counterpart of `Hexa3D` for such an import. | The same conforming quadrangle mesh `Hexa3D` needs, split across more faces | `ViscousLayers` (one per solid) |
| `HexaFromSkin3D` | Fills a solid with hexahedra derived from an existing all-quadrangle surface mesh. | An existing all-quadrangle mesh on the solid's skin | None |
| `Prism3D` | Extrudes a source face's mesh through a prismatic solid. Meshes the lateral faces and edges itself. | A 1-D and 2-D algorithm on the source face only | None of its own; it sweeps the `ViscousLayers2D` of its source face |
| `RadialPrism3D` | An O-grid between an inner and an outer shell: a pipe wall, an annulus. Needs the two shells' meshes to already match, typically via `Projection2D`. | Matching 2-D meshes on the inner and outer shell | `NumberOfLayers` or `LayerDistribution` |
| `Projection3D` | Copies a solid's mesh from another solid. | Nothing beyond the source solid's own mesh | `ProjectionSource3D` (required) |
| `Netgen1D2D3D` | Free tetrahedral meshing of a solid by NETGEN: segments, triangles, then tetrahedra, in one algorithm. A face or an edge with its own algorithm keeps that mesh. | Nothing: it meshes the faces and edges itself | `NetgenParameters` or `NetgenSimpleParameters3D`, `ViscousLayers` (one or several, each with its own face set) |
| `Netgen3D` | Free tetrahedral meshing of a solid by NETGEN, from the mesh of its faces. Pyramids join quadrangle faces to the tetrahedra. | A 2-D algorithm on every face | `NetgenParameters`, `MaxElementVolume`, `ViscousLayers` (one or several, each with its own face set) |
| `PolyhedronPerSolid3D` | One polyhedral element per solid, from the face mesh bounding it. Meshes every dimension itself; hides a lower-dimension algorithm beside it. Unlike `Cartesian3D`, it does consume an existing boundary mesh where one is present. | Nothing required; uses a boundary mesh if present | `ViscousLayers` (one or several, each with its own face set) |

### `Prism3D`: the source face, and a side face with a composite side

`Prism3D` sweeps from a face that is already meshed. Assign the 2-D algorithm on one face
alone, with a `SubShape`, and that face is meshed first and becomes the source. On two unit
boxes stacked and fused, every face a quadrangle, with 3 segments on every edge,
`Quadrangle2D` on the base face alone gives 3 x 3 x 6 = 54 hexahedra of volume 2. On the top
face alone it gives the same mesh.

With a global 2-D algorithm only, a face with more or fewer than four edges marks the
source. If every face reads as a quadrangle, `Prism3D` tries the faces in turn. Up to 4.2.2
it then reported the error of a face it had rejected, even when a later face worked: the
stacked boxes failed with "Wrong source face".

A side face whose bottom or top side has more than one edge has a composite side. This
happens where a cap edge is split under a whole one: the side face between them has 5
edges. `Prism3D` projects the opposite side onto the composite side as a whole, each node
at its share of the side's length. It sweeps through that face only if:

- the edges of the composite side join smoothly, as the parts of a split edge do;
- each split point gets a node, within the vertex tolerance;
- each edge of the composite side gets the number of segments that its own 1-D hypothesis
  gives, so that no hypothesis is dropped. Inside an edge the projection places the nodes,
  as on every edge of the target cap;
- the mesh is linear.

On a regular n-gon prism (circumradius 1, height 1) with one bottom edge split at its
midpoint, 4 segments on every edge and 2 on each half-edge, `Prism3D` sweeps between the
caps for every n from 4 to 8: each top node lies 1 above a bottom node, the cells are the
cap faces times 4, and the cell volumes sum to the prism volume within 1.6e-14 relative.
Up to 5.0.0 it projected onto the first edge of a side only, and refused n = 4, 6 and 8.

If the conditions fail, `Prism3D` tries another face as the source. With 4 segments on each
half-edge under 4 on the whole edge, n = 5 sweeps from a side face, as in 5.0.0; n = 6 and
8 have no face left, and the compute fails and names, for each face, why it is not the
source. So does a box with one bottom edge split and 4 segments on every edge: the
message names the two half-edges, their 8 segments and the 4 below. The way out: split
the opposite cap edge at the same points, so that the side face becomes two
quadrangles, or give the split edges the segments that the opposite side puts on them.

`Prism3D` builds no viscous layers itself, but it sweeps the 2-D layers of its source face.
Assign the 2-D algorithm and `ViscousLayers2D` on the source face alone: at every level of
the sweep the layer lines sit at the closed-form depths, and the levels follow the 1-D
hypothesis of the side edges. On a 2 x 1 x 1.875 block with `Geometric1D(0.125, 2)` on the
side edges, the levels are 0.125, 0.375, 0.875 and 1.875. `ViscousLayers2D` on any other
face of the solid is refused, because `Prism3D` meshes that face itself.

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

`NotConformAllowed` is global only: `assign` refuses it on a sub-shape. It allows a
non-conformal mesh between local algorithms that mesh their own boundary. With the
algorithms of this catalogue, no combination is known in which it changes the mesh.

### NETGEN

The NETGEN algorithms mesh any face or solid freely: triangles, tetrahedra, or a
quad-dominant surface. The sizes, the presets, the layers and the remesher are in the
[NETGEN guide](../guides/netgen.md).

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

A hypothesis on a sub-shape takes priority over one on a shape around it. Two hypotheses of
one kind on two shapes of the same type that share a sub-shape leave that sub-shape
ambiguous: for example 3 segments on one face and 5 on the face next to it, for their
common edge. SMESH reports this (`HYP_CONCURRENT`) when a later assignment makes it check
that edge, and `assign` then raises `PysmeshError`. The details name the edge, the faces
and their hypotheses. The assignment is undone, so `assignments()` is unchanged. Settle it
by assigning a hypothesis on the shared edge itself.

`assign` lets four other SMESH statuses pass. A missing hypothesis (`HYP_MISSING`) is the
normal state while a model is built, and `compute()` names what is still missing. A bad
parameter (`HYP_BAD_PARAMETER`) is refused when the hypothesis is built, or named by
`compute()`. A hidden or hiding algorithm (`HYP_HIDDEN_ALGO`, `HYP_HIDING_ALGO`) is SMESH's
defined priority of an all-dimensional algorithm, and which of the two statuses SMESH
reports depends only on the order of the assignments; `report.meshed` shows which algorithm
meshed each sub-shape.

## Reading `compute()`

`compute()` returns a `ComputeReport`:

```python
report = mesher.compute()
report.nodes     # node count of the whole mesh
report.edges     # 1-D element count
report.faces     # 2-D element count
report.volumes   # 3-D element count
report.meshed    # one SubMeshCount per sub-shape that received elements
report.warnings  # one ComputeWarning per sub-shape meshed with a warning
```

`report.meshed` is what tells "meshed by the algorithm I put there" from "meshed by an
enclosing all-dimensional algorithm that hid it". Read it whenever a mixed assignment is in
play.

**A warning is not a failure.** SMESH marks a sub-mesh computed when its algorithm reports a
warning: the algorithm met a problem, and it meshed the sub-shape anyway. `compute()` then
succeeds, and `report.warnings` lists each such sub-shape with its kind, its ordinal, the
algorithm and SMESH's own words. `Quadrangle2D` asked for `QuadType.REDUCED` on a face whose
opposite sides have different segment counts is an example: it warns that it used the
standard transition, and the mesh is the `QuadType.STANDARD` mesh.

```python
for w in report.warnings:
    print(w.kind, w.ordinal, w.algorithm, w.text)
```

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

**A missing algorithm or hypothesis is named by its state.** A sub-shape without an
algorithm, where an enclosing algorithm needs its mesh, holds the algorithm state
`NO_ALGO`; an algorithm without a hypothesis it needs holds `MISSING_HYP`. Neither is a
compute error in SMESH, so the failure names each such sub-shape with its algorithm and its
state, for example `SOLID 1: Cartesian_3D is missing a hypothesis it needs (algorithm state
MISSING_HYP)`. SMESH reports a compute as done when such a sub-shape is simply left
unmeshed: `Cartesian3D` without `CartesianParameters3D` made no volume, and `Projection2D`
without a source left its face empty. `compute()` raises for those too. A vertex keeps its
node, but `SegmentAroundVertex0D` without `SegmentLengthAroundVertex` does nothing, so it
is named as well. A sub-shape with no algorithm of its own is no error where an enclosing
algorithm meshes it, or where nothing needs its mesh, as for the solid under a surface mesh.

Cancellation is different from failure: if `cancel` returns `True`, or `progress` raises,
`compute()` raises `PysmeshCancelled` and the mesh is cleared, so nothing partial survives.
The remesher of a shape-free mesher is the exception: a cancel leaves the input mesh, or
the whole remesh if it came after the remesh replaced the mesh.

**Progress is exact only at sub-mesh granularity.** The fraction of sub-meshes already done
is real. Inside one running algorithm, SMESH interpolates with a tick counter, so an
algorithm that meshes the whole model in one call (`Cartesian3D`, for instance) reports
values that creep up from near zero and jump to 1.0 at the end. Only
`QuadFromMedialAxis1D2D` reports its own true fraction. Cancellation is not preemptive
either: only `Cartesian3D`, `Prism3D`, and the algorithm driven by `Adaptive1D` poll it
inside their own loop. The NETGEN algorithms pass it to netgen, which checks it between
its steps (up to 2.2 s measured). Every other algorithm can be stopped only between
sub-meshes.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
