# API Reference

The concept and guide pages explain how pySMESH's pieces fit together. This section is the
generated reference: every public class and function, with its full Google-style docstring
and typed signature, rendered directly from source by `mkdocstrings`.

349 of pySMESH's 355 public entities carry a complete docstring, and `src/pysmesh/_core.pyi`
gives the native extension's full typed surface, the same file `mypy --strict` checks a
consuming application against. Nothing here is hand-copied: this page and the three that
follow it exist to point `mkdocstrings` at the right modules, not to restate what a
docstring already says. If a reference page and a docstring ever disagree, the docstring in
the source is correct; open an issue against the page.

## Modules

| Module | Covers |
|---|---|
| `pysmesh` | The top-level namespace. Re-exports every public name below, so `import pysmesh` is enough for almost everything. |
| [`pysmesh.session`](session.md) | `Session`, `EntityId`, `EntityKind`, and every dataclass an operation or a query returns. See also [Session](../concepts/session.md) and [Entity IDs and ordinals](../concepts/entity-ids.md). |
| [`pysmesh.mesher`](mesher.md) | `Mesher`, the algorithm and hypothesis catalogue, quality controls and predicates, groups, the editor, the search surface, the medial axis, block decomposition and pattern mapping, and Inria GMF interchange. See also [Meshing model](../concepts/meshing-model.md). |
| [`pysmesh.step`](geometry.md#step) | `read_step_xde`, `write_step_xde`, `StepImport`, `EntityLabel`. |
| [`pysmesh.iges`](geometry.md#iges) | `read_iges`, `write_iges`, `IgesImport`, `IGES_UNITS`. |
| [`pysmesh.tessellate`](geometry.md#tessellation) | `tessellate`, `TessellateParams`, `TessellateResult`. |
| [`pysmesh.offset`](geometry.md#offsets) | `offset_shape`, `make_thick_solid`, and their parameter and result dataclasses. |
| [`pysmesh.distance`](geometry.md#distance-and-leaks) | `shape_distance`, `free_boundary_edges`, `ShapeDistanceResult`. |
| [`pysmesh.classify`](geometry.md#point-classification) | `point_in_solid`. |
| [`pysmesh.unify`](geometry.md#same-domain-healing) | `unify_same_domain` (the stateless form; `Session.unify_same_domain` carries session identity across the same operation). |
| [`pysmesh.viscous`](geometry.md#viscous-layers) | `compute_viscous_layers`, `VLParams`, `VLResult`, `ExtrusionMethod`. |
| [`pysmesh._core`](geometry.md#the-native-extension) | The native extension. `Shape`, `FaceInfo`, `EdgeInfo`, `SolidInfo`, `VertexInfo`, `Mesh`, `load_brep`, `PysmeshError`, `PysmeshCancelled`. Documented from its typed stub, `src/pysmesh/_core.pyi`, since the module itself is a compiled `.pyd` with no Python source to read. |

## Reading a generated entry

Each entry below shows the class or function's full signature, its parameter types, and its
docstring exactly as written in the source: an `Args:` section for parameters, a
`Returns:` section for the result, and a `Raises:` section for every documented failure
mode, including which ids or fields a `PysmeshError` carries in that case. Nothing here
infers behaviour from a name; every docstring was written against the implementation it
documents.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
