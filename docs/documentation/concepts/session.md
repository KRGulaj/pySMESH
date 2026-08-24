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
| Curve and surface construction | `add_line`, `add_arc`, `add_circle`, `add_ellipse`, `add_polyline`, `add_spline`, `add_bspline`, `add_helix`, `add_rectangle`, `make_wire`, `make_face`, `make_filling` |
| Sweeps | `extrude`, `revolve`, `pipe`, `pipe_shell`, `thru_sections` |
| Booleans with history | `fuse`, `cut`, `common`, `section`, `split`, `fragment` |
| Fillet and chamfer | `fillet`, `chamfer` |
| Transforms | `translate`, `rotate`, `mirror`, `scale`, `copy` |
| Healing | `heal`, `sew`, `remove_internal_wires`, `unify_same_domain`, `defeature`, `imprint`, `remove` |
| Tessellation | `tessellate` (the incremental render mesh) |
| Queries | `entity_table`, `entity_types`, `bounding_boxes`, `mass_properties`, `surface_parameters`, `face_wires`, `surface_at`, `curvature`, `project_on_face`, `entities_in_box`, `contains`, `adjacency`, `face_parameter_bounds`, `edge_parameter_bounds` |
| Handoff | `export_handoff`, `brep` |
| Identity and introspection | `entities`, `entity_kind`, `is_alive`, `shape_count`, `name_of`, `origin`, `resolve`, `op_count`, `state_op_index`, `issued_id_count`, `entity_count` |

Every mutating operation returns a `HistoryDelta`: which ids it created, deleted, modified,
split and merged, plus the `BRepCheck_Analyzer` verdict on the shape it built. See
[Entity IDs and ordinals](entity-ids.md#identity-rules-across-an-edit) for what each of
those five arrays means.

### Booleans: `fuzzy` and `parallel`

Every boolean and `imprint` takes `fuzzy` (extra tolerance for deciding two shapes touch)
and `parallel` (run OCCT's internal steps on several threads). Both default to values that
are correct for clean geometry: `fuzzy=0.0` means "use each shape's own stored tolerance",
which is the right answer for geometry built in the session or imported cleanly. Raise it
only for an import whose faces do not quite meet, and choose it against the measured gap
rather than turning it up for luck: a `fuzzy` value larger than the model's smallest real
feature merges things that are genuinely separate. `parallel=True` is a speed setting only;
the result does not depend on it.

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
came from. `export_handoff` verifies the id-to-sub-shape map is a bijection before it
returns, and raises rather than handing back a map that has quietly lost some of the
caller's names: a same-domain merge leaves several live ids on one face, and a split leaves
one live id on several, and either makes the pairing ambiguous.

## Thread contract

A `Session` is **not** thread-safe. Use one session per thread. Sessions are independent,
so several may coexist in one process with no cross-talk. Long operations release the GIL,
so entering an operation on a session while another is already in flight on that same
session raises, rather than racing on the registry.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
