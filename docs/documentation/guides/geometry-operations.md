# Geometry Operations

pySMESH exposes a set of standalone Open CASCADE operations that need no `Session` and no
`Mesher`. Each one is headless: it takes BREP bytes and NumPy arrays, runs one OCCT
algorithm, and returns BREP bytes and NumPy arrays. Every result is keyed to the same
1-based ordinals `Shape.faces()` / `.edges()` / `.solids()` use. This guide walks through
each operation with a runnable example. See [Units](../concepts/units.md) for the STEP and
IGES length-unit contract in full, and [Entity IDs and ordinals](../concepts/entity-ids.md)
for what the ordinals mean and when they change.

## STEP: `read_step_xde` / `write_step_xde`

`read_step_xde` imports a STEP file through OCCT's XDE stack, carrying product names,
per-face colours, and the file's length unit across the boundary, none of which a plain
geometry transfer preserves:

```python
import pysmesh

imported = pysmesh.read_step_xde("blade.step")   # bytes or a path both work
imported.length_unit                              # metres per model unit
imported.face_labels                               # EntityLabel(id, name, color) tuples
imported.solid_labels

shape = pysmesh.load_brep(imported.brep)
shape.faces()[imported.face_labels[0].id - 1]      # the labelled face, by its 1-based id
```

`write_step_xde` writes a BREP back out, tagging a product name and per-face names and
colours by their 1-based face id:

```python
data = pysmesh.write_step_xde(
    imported.brep,
    unit=imported.unit_name,
    name="blade",
    face_names={1: "inlet"},
    face_colors={1: (0.0, 1.0, 0.0)},
)
```

`unit` is required and names the unit the coordinates are already in. It labels them and
never rescales them. Passing `imported.unit_name` makes the export a round trip. See
[Units](../concepts/units.md#the-write-contract) for the full contract.

## IGES: `read_iges` / `write_iges`

```python
igs = pysmesh.read_iges("housing.igs")   # a path, not bytes
igs.length_unit    # 0.001 for an MM file, 0.0254 for an INCH file
igs.unit_name       # "MM", "INCH", "M", ...

pysmesh.write_iges(igs.brep, unit=igs.unit_name)   # re-export, unit-exact
```

`write_iges` takes the unit of the coordinates as an explicit argument and declares it in
the header without rescaling, exactly as `write_step_xde` does. Both accept the same ten
names. See [Units](../concepts/units.md) for the full contract, including why `read_iges`
takes a path and not bytes.

## Tessellation: `tessellate`

`tessellate` drives OCCT's `BRepMesh_IncrementalMesh` to build a render-quality
triangulation, with per-vertex normals:

```python
from pysmesh import TessellateParams, tessellate

result = tessellate(brep, TessellateParams(lin_defl=0.05, ang_defl_deg=15.0))
result.nodes         # (N, 3) float64
result.tris           # (M, 3) int32, 0-based row indices into nodes
result.tri_face_id    # (M,) int32, 1-based face id per triangle
result.normals        # (N, 3) float64, per-vertex outward normals
```

Nodes at face boundaries are not welded: each face contributes its own node range, which is
correct for B-rep shading (a hard edge at a face seam, smooth shading inside a curved
patch). `lin_defl` is the chord deflection in model units by default; pass `relative=True`
in `TessellateParams` to read it as a fraction of the shape's bounding-box diagonal instead.

## Offsets: `offset_shape` / `make_thick_solid`

`offset_shape` moves every face of a shell or solid outward or inward by a signed distance:

```python
from pysmesh import OffsetParams, offset_shape

result = offset_shape(brep, OffsetParams(offset=0.5))   # positive: grow outward
result.brep
result.face_map   # (n_faces_in,) int32; face_map[i - 1] is the new id, or -1 if removed
```

`make_thick_solid` hollows a solid: name the faces to remove as openings, and the rest are
offset inward (or outward) to build the wall:

```python
from pysmesh import ThickSolidParams, make_thick_solid

result = make_thick_solid(
    brep, ThickSolidParams(remove_face_ids=(3,), thickness=-0.2)
)
```

A negative `thickness` hollows the solid; a positive one enlarges it. Both raise
`PysmeshError` with the offending new face ids on `.face_ids` when the offset
self-intersects, which happens once `abs(thickness)` exceeds the smallest feature it has to
clear.

## Distance and leaks: `shape_distance` / `free_boundary_edges`

`shape_distance` is the exact minimum distance between two shapes of any kind, with the
witness point on each:

```python
from pysmesh import shape_distance

result = shape_distance(brep_a, brep_b)
result.distance   # 0.0 when the shapes touch or overlap
result.point_a
result.point_b
```

`free_boundary_edges` returns the 1-based ids of every edge bordered by exactly one face,
the naked boundary of an open shell. A watertight solid returns an empty array; a non-empty
result localises a leak:

```python
from pysmesh import free_boundary_edges

leaks = free_boundary_edges(brep)
if leaks.size:
    shape = pysmesh.load_brep(brep)
    leaky_edges = [shape.edges()[i - 1] for i in leaks.tolist()]
```

## Point classification: `point_in_solid`

An exact inside test against a BREP solid, using `BRepClass3d_SolidClassifier`:

```python
import numpy as np
from pysmesh import point_in_solid

points = np.array([[0.5, 0.5, 0.5], [10.0, 10.0, 10.0]], dtype=np.float64)
mask = point_in_solid(brep, points, tol=1e-7)   # (2,) bool
```

A point within `tol` of the boundary counts as *on* it, not inside, so it reads `False`.
That is the right contract for choosing a seed point inside a volume, where a point on the
wall is not in the volume.

## Same-domain healing: `unify_same_domain`

Real B-rep face and edge merging: it deletes the shared seam from the topology, so a
downstream mesher never places nodes along it, unlike a mesher hint that only suppresses the
seam visually.

```python
from pysmesh import UnifyParams, unify_same_domain

result = unify_same_domain(brep, UnifyParams(linear_tol=1e-6, angular_tol_deg=0.5))
result.n_faces_before, result.n_faces_after
result.face_map   # (n_faces_before,) int32, new id per old id, -1 if removed
```

`Session` has the same operation as `Session.unify_same_domain`, which carries the
session's own entity identity across the merge instead of returning a raw id map. Use the
stateless form here when there is no session in play.

## `Shape`: per-entity metadata

`load_brep` returns a `Shape`, the read side every operation above keys its ids to:

```python
shape = pysmesh.load_brep(brep)

shape.solids()    # list[SolidInfo]: id, volume, centroid, bbox
shape.faces()      # list[FaceInfo]: id, area, surface_type, centroid, bbox, uv_bounds
shape.edges()       # list[EdgeInfo]: id, length, bbox, t_bounds
shape.vertices()    # list[VertexInfo]: id, xyz

shape.faces()[0].surface_type   # "Plane" / "Cylinder" / "Cone" / "Sphere" / "Torus" / ...
```

`Shape.face_adjacency()` lists which faces touch which, as `(face_a, face_b, shared_edge)`
triples. `Shape.match_faces(centroids, tol)` pairs a caller-supplied set of centroids to the
nearest face by position, which is the tool for tracking a face across two independently
produced shapes when neither carries the other's ids; it is a nearest-centroid heuristic,
not a proof of identity, and two faces sharing a centroid (a pipe's inner and outer wall, for
instance) cannot be told apart by it.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
