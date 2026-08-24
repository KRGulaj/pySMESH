# pySMESH

pySMESH is a standalone Windows Python wheel. It exposes SALOME SMESH's meshing operations
and Open CASCADE's (OCCT) geometry operations through NumPy arrays and BREP bytes. SMESH,
OCCT, Boost and VTK are statically linked or bundled into one extension module, `_core.pyd`.
Installing the wheel adds one pip entry and no other native dependency.

pySMESH is for a CFD or CAD preprocessing pipeline that needs a production meshing and
geometry kernel, without SALOME's GUI, CORBA layer, or KERNEL platform. It does not replace
SALOME. It exposes the parts of SMESH and OCCT such a pipeline needs.

## Who this is for

Use pySMESH if you build or edit CAD geometry in Python, generate volume or surface meshes
from that geometry, or process a mesh that already exists (an imported STL, a shrink-wrap
result, a file read from disk). Correctness of the meshing and geometry math is
non-negotiable: every operation is a thin, verified wrapper over SMESH or OCCT, not a
re-implementation.

## The three areas

pySMESH covers three areas. Each has its own concept page and its own guide in this
documentation.

- **Session.** A stateful CAD modelling session. Build and edit a shape: primitives,
  booleans with history, fillets, chamfers, sweeps, healing, defeaturing, imprinting. Every
  entity keeps a persistent id across every edit. See
  [Session](concepts/session.md) and [Entity IDs and ordinals](concepts/entity-ids.md).
- **Mesher.** SMESH's meshing pipeline. Assign an algorithm and its hypotheses to a
  sub-shape, compute a mesh, then edit, query, and check its quality. A `Mesher` also
  accepts a mesh it did not build: a discrete body with no B-rep goes in as plain arrays.
  See [Meshing model](concepts/meshing-model.md) and
  [Discrete meshes](concepts/discrete-meshes.md).
- **Standalone OCCT geometry.** STEP and IGES import and export, tessellation, offsets,
  distance and leak checks, point-in-solid classification, and geometry queries. See
  [Geometry operations](guides/geometry-operations.md).

## Install

```bash
pip install pysmesh
```

That is the whole procedure. The wheel is self-contained: SMESH, OCCT, Boost and VTK all
ship inside it. NumPy is the only thing pip pulls in.

pySMESH targets Windows x64, CPython 3.11 to 3.14. There is one wheel per interpreter and
none for other platforms. Pip picks the matching wheel and refuses to install on anything
unsupported.

Since version 4.0.0, no VTK, OCCT or Boost installation is required or recognised on the
host. See [VTK privacy](concepts/vtk-privacy.md) for why that guarantee holds, and why it
is load-bearing for anyone integrating pySMESH beside their own VTK.

## Documentation map

| Page | Covers |
|---|---|
| [Entity IDs and ordinals](concepts/entity-ids.md) | The 1-based TopExp ordinal convention. How a `Session` id differs from an ordinal. What survives an edit and what does not. |
| [Session](concepts/session.md) | The stateful CAD modelling session: identity, snapshot and restore, the operation families, the handoff to a `Mesher`. |
| [Meshing model](concepts/meshing-model.md) | SMESH's algorithm and hypothesis assignment model, per dimension. What `compute()` returns and how to read a failure. |
| [Discrete meshes](concepts/discrete-meshes.md) | `Mesher` with no B-rep. Filling from arrays. The patch workflow. What refuses without a shape. |
| [Units](concepts/units.md) | The STEP and IGES unit contract: native coordinates, `length_unit`, and why a file cannot arrive silently rescaled. |
| [VTK privacy](concepts/vtk-privacy.md) | Why no VTK, OCCT or Boost object may cross the Python boundary, and how that is enforced. |
| [Getting started](guides/getting-started.md) | Install, then a complete worked example from a box to a computed mesh. |
| [Geometry operations](guides/geometry-operations.md) | The standalone OCCT surface: STEP/IGES, tessellation, offsets, distance, `Shape`. |
| [Mesh editing](guides/mesh-editing.md) | Quality controls, groups, the editor, search, the medial axis, and viscous layers. |
| [API reference](reference/index.md) | The generated reference: every public class and function, with its docstring and typed signature. |

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
