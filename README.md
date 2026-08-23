# pySMESH

Python bindings to SALOME SMESH and Open CASCADE (OCCT), packaged as one
self-contained Windows wheel. No SALOME platform. No CORBA. No GUI.

pySMESH gives a CFD or CAD preprocessing pipeline direct access to a
production meshing and geometry kernel, through plain NumPy arrays and BREP
bytes. It does not replace SALOME. It exposes the parts of SMESH and OCCT a
pipeline needs, as a normal `pip`-installable module.

Meta:

- **License:** LGPL-2.1-only (see [LICENSE](LICENSE), [NOTICE.md](NOTICE.md))
- **Platform:** Windows x64, CPython 3.13
- **Runtime dependencies:** NumPy. Nothing else. SMESH, OCCT, Boost and VTK all ship
  inside the wheel.

## What it covers

pySMESH covers three areas:

- **`Session`**: a stateful CAD modelling session. Build and edit a shape
  (primitives, booleans, fillets, chamfers, sweeps, healing). Every entity
  keeps a persistent id across every edit. See
  [CAD modelling](#cad-modelling).
- **`Mesher`**: SMESH's full meshing pipeline. Assign algorithms and
  hypotheses per sub-shape, compute a mesh, then edit, query, and check its
  quality. It also accepts a mesh it did not build: a discrete body with no
  B-rep goes in as plain arrays. See
  [Mesh generation and editing](#mesh-generation-and-editing) and
  [Discrete meshes](#discrete-meshes-no-cad).
- **Standalone OCCT geometry operations**: STEP and IGES import/export,
  tessellation, offsets, distance and leak checks, point-in-solid
  classification, and geometry queries. See
  [Geometry operations](#geometry-operations).

Two entry points serve one-shot work outside a session. `compute_viscous_layers`
wraps `StdMeshers_ViscousLayers` for 3-D boundary-layer prism meshing.
`unify_same_domain` wraps `ShapeUpgrade_UnifySameDomain` for B-rep face
merging that removes STEP import seams.

## Install

```bash
pip install pysmesh
```

That is the whole procedure. The wheel is self-contained: SMESH, OCCT, Boost
and VTK all ship inside it. NumPy is the only thing pip pulls in.

**Platform:** Windows x64, CPython 3.13 only. There are no other wheels. Pip
will refuse the wheel on any other interpreter rather than install something
that cannot import.

Wheels are also attached to every
[GitHub Release](https://github.com/KRGulaj/pySMESH/releases), for pinning a
build by exact file.

> **Upgrading from 3.x:** 4.0.0 removes the shared VTK requirement. Earlier
> versions linked the host environment's VTK and refused to import unless it
> was exactly 9.6.2. That constraint is gone. pySMESH now carries its own
> private VTK, so it no longer cares which VTK you have, or whether you have
> one at all. If you install with `--no-deps`, or gate on
> `_build_info.VTK_VERSION`, that check is now obsolete and always passes.

## Quick example

```python
import pysmesh
from pysmesh import Session
from pysmesh.session import EntityKind

s = Session()
s.add_box(3.0, 7.0, 11.0)
s.fillet(edge_ids=s.entities(EntityKind.EDGE), radius=0.5)

handoff = s.export_handoff()  # brep bytes + per-entity id arrays, ready for Mesher

mesher = pysmesh.Mesher(pysmesh.load_brep(handoff.brep))
mesher.assign(pysmesh.mesher.Regular1D())
mesher.assign(pysmesh.mesher.NumberOfSegments(count=8))
mesher.assign(pysmesh.mesher.Quadrangle2D())
mesher.assign(pysmesh.mesher.Hexa3D(), on=pysmesh.mesher.SubShape(
    pysmesh.mesher.SubShapeKind.SOLID, 1
))
mesher.compute()
mesh = mesher.mesh()
```

## CAD modelling

`Session` owns one live shape and gives every entity a persistent id, so
edits, undo, and mesh handoff all stay correct as the shape changes.

```python
from pysmesh import Session
from pysmesh.session import EntityKind

s = Session()
s.add_box(3.0, 7.0, 11.0)
s.fillet(edge_ids=s.entities(EntityKind.EDGE), radius=0.5)
mark = s.snapshot()          # O(1)
s.restore(mark)              # O(1)

handoff = s.export_handoff()  # brep bytes + per-entity id arrays, ready for Mesher
```

`Session` covers primitives, curve and surface construction, sweeps, booleans
with history, fillet and chamfer, transforms, healing, defeaturing,
imprinting, tessellation, and geometric queries. See
`src/pysmesh/session/__init__.py` for the full API.

Two queries answer the questions a feature filter asks. `surface_parameters`
reads a face's analytic parameters off its surface: a radius, a cone's
taper, a torus's two radii. `face_wires` splits a face's boundary into its
loops, so an inner loop (a hole) is distinguishable from the outer one. A
parameter the surface type does not define reads `NaN`, never a stand-in
value.

```python
import numpy as np

faces = s.entities(EntityKind.FACE)
params = s.surface_parameters(faces)

# every cylindrical face under 1 mm across: fillets and small bores
small = params.ids[(np.array(params.types) == "Cylinder") & (params.radius1 < 1.0)]

wires = s.face_wires(faces)
for row in np.flatnonzero(~wires.is_outer):        # one row per hole
    lo, hi = wires.edge_range[row]
    hole_edges = wires.edge_id[lo:hi]
```

## Mesh generation and editing

`Mesher` builds a volume or surface mesh from a shape, using SMESH's own
algorithm and hypothesis model. Assign an algorithm and its hypotheses to a
sub-shape. Different sub-shapes can use different algorithms. Compute the
mesh, then read it back as NumPy arrays.

```python
from pysmesh import load_brep
from pysmesh.mesher import Hexa3D, Mesher, NumberOfSegments, Quadrangle2D, Regular1D
from pysmesh.mesher import SubShape, SubShapeKind

mesher = Mesher(load_brep(handoff.brep))
mesher.assign(Regular1D())
mesher.assign(NumberOfSegments(count=8))
mesher.assign(Quadrangle2D())
mesher.assign(Hexa3D(), on=SubShape(SubShapeKind.SOLID, 1))
report = mesher.compute()
mesh = mesher.mesh()
```

Once a mesh exists:

- **Quality controls** measure and classify cells: aspect ratio, skew,
  orientation, and more.
- **Groups** name sets of elements. A group survives edits, so a wall named
  on a coarse mesh is still the wall after conversion to second order.
- **The editor** smooths, merges coincident nodes, reorients cells, splits
  and fuses faces, converts between linear and quadratic, sews free borders,
  offsets a surface, and deletes elements and nodes.
- **Search** locates elements at a point, casts rays through the mesh, finds
  sharp edges, and classifies a point as inside or outside a closed surface.
- **The medial axis** of a face reports its centreline and local wall
  thickness. A face can also be decomposed into blocks or have a pattern
  mapped onto it.
- **`compute_viscous_layers`** grows prism boundary layers on a classified
  surface mesh. See `examples/box_bl.py` for the end-to-end walkthrough.

See `src/pysmesh/mesher/__init__.py` for the full model and
`src/pysmesh/_core.pyi` for the typed API.

## Discrete meshes (no CAD)

A `Mesher` does not need a shape. `Mesher()` starts empty and is filled from
arrays, which is how a body that never had a B-rep gets in: an imported STL,
OBJ or PLY, a shrink-wrap result, the boundary another mesher produced, or a
mesh read back from a file.

```python
from pysmesh import Mesher

mesher = Mesher.from_arrays(points, triangles)   # (N, 3) float64, (M, 3) row indices

# Divide the surface into patches. Without CAD faces, this is what a viewport picks on.
edges = mesher.sharp_edges(angle=40.0)
patches = mesher.separate_faces_by_edges(edges, name_prefix="patch_")

# Delete one patch. The report names every id that went, including the freed nodes.
gone = mesher.remove_elements(patches.at(2), free_nodes=True)
print(gone.elements, gone.nodes)
```

Three things are worth knowing:

- **Ids are the handle.** Nodes and elements keep their ids for as long as
  they exist, and nothing is ever renumbered. `add_nodes` and
  `add_elements` return the ids they created. `Mesher.from_mesh(mesh_data)`
  rebuilds a live mesh from a harvest and keeps every one of them, which is
  the way back from `read_gmf`.
- **A patch index is not stable. A patch group is.** Each call to
  `separate_faces_by_edges` re-derives the partition, so indices can shift
  once faces have been deleted. Passing `name_prefix` stores each patch as
  a group, and SMESH maintains that membership itself: a deleted element
  leaves the group, survivors keep their place.
- **What such a mesher cannot do.** Anything that resolves a sub-shape
  ordinal: `compute`, `assign`/`unassign`, `add_group_on_shape`, the
  pattern mapping, `smooth(in_uv_space=True)`, and the `ElementsOnShape` and
  `Deflection2D` controls. Each refuses by name. Everything else, the
  editor, search, quality controls, groups by id or by filter, behaves
  identically. Check with `mesher.has_shape`.

## Geometry operations

Standalone Open CASCADE operations. All are headless (no VTK, no SMESH),
take and return BREP bytes and NumPy arrays, and key every result to the
same 1-based ordinals `Shape.faces()` / `.edges()` / `.solids()` use.

- **`read_step_xde` / `write_step_xde`**: STEP import/export through OCCT's
  XDE stack, preserving product names, per-face colours, and the file's
  length unit.
- **`read_iges` / `write_iges`**: IGES import/export. The reader returns
  the geometry in the file's native unit plus `length_unit` (metres per
  unit), exactly as `read_step_xde` does. The writer takes the unit of the
  coordinates as an argument and declares it in the header without
  rescaling. Neither side reads or writes OCCT's global
  `Interface_Static` unit, so a file cannot arrive silently normalised or
  leave silently mislabelled.
- **`tessellate`**: render-ready triangulation with per-vertex normals.
- **`offset_shape` / `make_thick_solid`**: B-rep offset and hollowed
  thick-solid operations.
- **`shape_distance` / `free_boundary_edges`**: exact minimum distance
  between two shapes, and the naked edges that localise a hole in an open
  shell.
- **`point_in_solid`**: exact inside test against a solid.
- **`unify_same_domain`**: real B-rep face merging that deletes the shared
  seam from the topology, instead of leaving a mesher hint that still
  forces nodes along it.
- **`Shape`**: per-entity metadata. Surface type, face adjacency, solids,
  and centroid-based face matching.

```python
import pysmesh

imp = pysmesh.read_step_xde("blade.step")
shape = pysmesh.load_brep(imp.brep)
shape.faces()[0].surface_type   # "Plane" / "Cylinder" / "Cone" / "Sphere" / "Torus" / ...

mask = pysmesh.point_in_solid(imp.brep, points, tol=1e-7)

# IGES carries the same unit contract. Coordinates stay native; the factor comes back with
# them, and a re-export declares the unit it was handed.
igs = pysmesh.read_iges("housing.igs")
igs.length_unit                              # 0.001 for an MM file, 0.0254 for an INCH file
pysmesh.write_iges(igs.brep, unit=igs.unit_name)
```

`read_iges` takes a path, not bytes. OCCT ships no IGES stream reader
(`IGESSelect_WorkLibrary` does not override `IFSelect_WorkLibrary::ReadStream`).
Reading one also makes OCCT print `Total number of loaded entities N.` to
stdout. That is an unconditional info-level message inside `IGESFile_Read`,
and OCCT gives no switch to silence it.

See `src/pysmesh/_core.pyi` for the full typed API. `mypy --strict`
type-checks against it.

## Packaging model

SMESH and OCCT are mature, production-grade CAD and meshing libraries, but
neither ships a standalone Python wrapper. Both are normally reached only
through the full SALOME platform, wrapped through CORBA/SWIG. That pulls in
the entire SALOME GUI and KERNEL stack. pySMESH strips both down to the
static library set a CFD preprocessing pipeline needs, and exposes them as
a plain, pip-installable module.

Doing that also solves a packaging problem. SMESH pulls in OCCT, Boost and
VTK as dependencies. Installing those directly into a host application's
environment can trigger a dependency solver cascade that downgrades
unrelated packages: Qt bindings, MKL, and more. pySMESH's build makes that
impossible by construction.

- **SMESH and KERNEL are statically linked** into a single `_core.pyd`.
- **OCCT, Boost and VTK are private** to that binary. Their DLLs are
  **bundled into the wheel** and name-mangled, so they never appear in the
  host environment and cannot collide with the host's own copies.

Net effect on the host environment: installing the wheel adds **one** pip
entry (`pysmesh`), plus NumPy. It constrains nothing else. Your application
is free to use any VTK it likes, including a different version, because
pySMESH never touches it.

That privacy rests on one property, which is worth stating plainly because
it is the thing the design protects: **no VTK object crosses the Python
boundary.** Every result leaves as a NumPy array or BREP bytes. A
`vtkUnstructuredGrid` built inside `_core` would be an instance of a class
from the private, name-mangled copy, and therefore a different C++ type from
the host's, even at an identical version string.
[tests/test_vtk_privacy.py](tests/test_vtk_privacy.py) fails the build if a
binding ever exports one.

> **Binary size:** the 4.0.0 wheel is **41 MB**, holding 75 bundled DLLs.
> OCCT is the largest share at 18.7 MB; private VTK costs 15.7 MB. That is
> the deliberate trade for zero native footprint in the host environment.
> `_core` links only three VTK components (`CommonCore`, `CommonDataModel`,
> `FiltersVerdict`), so the bundle carries no rendering, IO, or
> Python-wrapper modules: 17 VTK DLLs out of the 187 MB a full VTK install
> would put in your environment. CI reports the breakdown on every build and
> fails if the wheel would exceed PyPI's 100 MB limit.

## Build from source

Requires MSVC v143, GNU `patch`, and a conda-forge build environment. All of
VTK, OCCT and Boost are build-time only, and end up inside the wheel:

```bash
conda env create -f ci/environment.yml
conda activate <the env name in ci/environment.yml>

python prepare.py                                # stage extern/ -> staged/ and apply patches
pip wheel . --no-build-isolation --no-deps -w dist
# CI additionally repairs the wheel with delvewheel, bundling the whole native closure
# (OCCT + Boost + VTK) and name-mangling every DLL.
```

For local development (run tests against a freshly built extension without a
wheel):

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=<env>/Library -DPython_EXECUTABLE=<env>/python.exe
cmake --build build --target _core               # copies _core + _build_info into src/pysmesh
pytest tests/ -q
python examples/box_bl.py
```

### Capability probe

`tests/probe` is a build-verification target: it constructs and runs every
OCCT class and SMESH capability pySMESH depends on, against the exact link
set `_core` uses. Run it after any change to the OCCT toolkit list, the
patch series, or the `StdMeshers` source set. It turns a missing toolkit, a
dead-stripped SMESH object, or an un-built translation unit into a named
failure instead of a surprise mid-binding.

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=<env>/Library -DPython_EXECUTABLE=<env>/python.exe \
      -DPYSMESH_BUILD_V2_PROBE=ON
cmake --build build --target v2_probe
./build/v2_probe.exe                             # exit 0 == every probed capability is usable
```

## Design principles

- **Narrow API.** Every exported function exists to serve a concrete CAD or
  meshing pipeline need. No SWIG, no `smeshBuilder` emulation, no MED/CGNS
  I/O. Data crosses the boundary as NumPy arrays and BREP bytes.
- **Fail loud.** Every failure is a typed `pysmesh.PysmeshError` carrying
  the underlying SMESH/OCCT message and, where applicable, the offending
  ids. Never a silent best-effort fallback.

## Provenance and licensing

Every vendored source and patch is traced in [PROVENANCE.md](PROVENANCE.md).
The third-party component table is in [NOTICE.md](NOTICE.md).
