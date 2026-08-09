# pySMESH

Standalone Python bindings to SALOME SMESH and Open CASCADE (OCCT), packaged as one
self-contained `cp313-win_amd64` wheel. No SALOME platform, no CORBA, no GUI.

pySMESH covers three areas:

- **`Session`** — a stateful CAD modelling session. Build and edit a shape (primitives,
  booleans, fillets, chamfers, sweeps, healing), with persistent entity identity across
  every edit. See [CAD modelling](#cad-modelling).
- **`Mesher`** — SMESH's full meshing pipeline. Assign algorithms and hypotheses per
  sub-shape, compute a mesh, then edit, query, and check its quality. See
  [Mesh generation and editing](#mesh-generation-and-editing).
- **Standalone OCCT geometry operations** — STEP import/export, tessellation, offsets,
  distance and leak checks, point-in-solid classification, and geometry queries. See
  [Geometry operations](#geometry-operations).

Two entry points serve one-shot work outside a session: `compute_viscous_layers`
(`StdMeshers_ViscousLayers`, 3-D boundary-layer prism meshing) and `unify_same_domain`
(`ShapeUpgrade_UnifySameDomain`, B-rep face merging that removes STEP import seams).

Meta:

- **License:** LGPL-2.1-only (see [LICENSE](LICENSE), [NOTICE.md](NOTICE.md))
- **Platform:** Windows x64, CPython 3.13
- **Runtime dependency shared with the host:** **VTK 9.6.2** (exact, checked at import)

## Why this exists

SMESH and OCCT are mature, production-grade CAD and meshing libraries, but neither ships
a standalone Python wrapper. Both are normally reached only through the full SALOME
platform, wrapped through CORBA/SWIG, pulling in the entire SALOME GUI and KERNEL stack.
pySMESH strips both down to the static library set a CFD preprocessing pipeline needs,
and exposes them as a plain, pip-installable module.

Doing that also solves a packaging problem. SMESH pulls in OCCT and Boost as
dependencies, and installing `occt`/`boost` directly into a host application's
environment can trigger a dependency solver cascade that downgrades unrelated packages
(VTK, Qt bindings, MKL, and so on). pySMESH's build makes that impossible by
construction:

- **SMESH and KERNEL are statically linked** into a single `_core.pyd`.
- **OCCT and Boost are private** to that binary. Their DLLs are **bundled into the
  wheel**, so they never appear in the host environment.
- **VTK is the one shared dependency** (SMESH's data structure is built on
  `vtkUnstructuredGrid`). It is linked **dynamically against the host's own VTK**, and
  its version is **hard-checked at import**. A mismatch raises `ImportError` instead of
  risking a silent ABI crash. This is the only version coupling a consuming application
  needs to track.

Net effect on the host environment: installing the wheel adds **one** pip entry
(`pysmesh`) and nothing else. No `occt`, no `boost`, no VTK downgrade.

> **Binary size:** `_core.pyd` is a few MB, and the bundled OCCT/Boost DLLs add tens of
> MB. That is the deliberate trade for zero OCCT/Boost footprint in the host
> environment, and it is still smaller than shipping OCCT and its transitive DLLs
> separately.

## Install

Download the wheel from the
[GitHub Releases page](https://github.com/KRGulaj/pySMESH/releases), then install it
alongside its one shared dependency:

```bash
# 1. VTK 9.6.2 must be present in the target environment (exact version, checked at import)
pip install "vtk==9.6.2"
# or, if you manage the environment with conda:
# conda install -c conda-forge vtk==9.6.2

# 2. Install the downloaded wheel (OCCT and Boost are bundled, no other deps needed)
pip install pysmesh-3.0.0-cp313-win_amd64.whl
```

**Platform:** Windows x64, CPython 3.13 only. There are no other wheels.

**VTK coupling:** pySMESH links against the host's VTK at runtime. `import pysmesh`
checks the exact version and raises `ImportError` on a mismatch, rather than risking a
silent ABI crash. If your project pins a different VTK version, pySMESH cannot be used
in the same environment.

## CAD modelling

`Session` owns one live shape and gives every entity a persistent id, so edits, undo,
and mesh handoff all stay correct as the shape changes.

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

`Session` covers primitives, curve and surface construction, sweeps, booleans with
history, fillet and chamfer, transforms, healing, defeaturing, imprinting, tessellation,
and geometric queries. See `src/pysmesh/session/__init__.py` for the full API.

## Mesh generation and editing

`Mesher` builds a volume or surface mesh from a shape, using SMESH's own algorithm and
hypothesis model. Assign an algorithm and its hypotheses to a sub-shape. Different
sub-shapes can use different algorithms. Compute the mesh, then read it back as NumPy
arrays.

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

- **Quality controls** measure and classify cells: aspect ratio, skew, orientation, and
  more.
- **Groups** name sets of elements. A group survives edits, so a wall named on a coarse
  mesh is still the wall after conversion to second order.
- **The editor** smooths, merges coincident nodes, reorients cells, splits and fuses
  faces, converts between linear and quadratic, sews free borders, and offsets a
  surface.
- **Search** locates elements at a point, casts rays through the mesh, finds sharp
  edges, and classifies a point as inside or outside a closed surface.
- **The medial axis** of a face reports its centreline and local wall thickness. A face
  can also be decomposed into blocks or have a pattern mapped onto it.
- **`compute_viscous_layers`** grows prism boundary layers on a classified surface mesh.
  See `examples/box_bl.py` for the end-to-end walkthrough.

See `src/pysmesh/mesher/__init__.py` for the full model and `src/pysmesh/_core.pyi` for
the typed API.

## Geometry operations

Standalone Open CASCADE operations. All are headless (no VTK, no SMESH), take and
return BREP bytes and NumPy arrays, and key every result to the same 1-based ordinals
`Shape.faces()` / `.edges()` / `.solids()` use.

- **`read_step_xde` / `write_step_xde`** — STEP import/export through OCCT's XDE stack,
  preserving product names, per-face colours, and the file's length unit.
- **`tessellate`** — render-ready triangulation with per-vertex normals.
- **`offset_shape` / `make_thick_solid`** — B-rep offset and hollowed thick-solid
  operations.
- **`shape_distance` / `free_boundary_edges`** — exact minimum distance between two
  shapes, and the naked edges that localise a hole in an open shell.
- **`point_in_solid`** — exact inside test against a solid.
- **`unify_same_domain`** — real B-rep face merging that deletes the shared seam from
  the topology, rather than leaving a mesher hint that still forces nodes along it.
- **`Shape`** — per-entity metadata: surface type, face adjacency, solids, and
  centroid-based face matching.

```python
import pysmesh

imp = pysmesh.read_step_xde("blade.step")
shape = pysmesh.load_brep(imp.brep)
shape.faces()[0].surface_type   # "Plane" / "Cylinder" / "Cone" / "Sphere" / "Torus" / ...

mask = pysmesh.point_in_solid(imp.brep, points, tol=1e-7)
```

See `src/pysmesh/_core.pyi` for the full typed API. `mypy --strict` type-checks against
it.

## Build from source

Requires MSVC v143, GNU `patch`, and a conda-forge build environment (VTK **pinned** to
the host application's version; OCCT/Boost free to resolve):

```bash
conda env create -f ci/environment.yml
conda activate <the env name in ci/environment.yml>

python prepare.py                                # stage extern/ -> staged/ and apply patches
pip wheel . --no-build-isolation --no-deps -w dist
# CI additionally repairs the wheel with delvewheel to bundle OCCT/Boost and EXCLUDE vtk*.dll
```

For local development (run tests against a freshly built extension without a wheel):

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=<env>/Library -DPython_EXECUTABLE=<env>/python.exe
cmake --build build --target _core               # copies _core + _build_info into src/pysmesh
pytest tests/ -q
python examples/box_bl.py
```

### Capability probe

`tests/probe` is a build-verification target: it constructs and runs every OCCT class
and SMESH capability pySMESH depends on, against the exact link set `_core` uses. Run it
after any change to the OCCT toolkit list, the patch series, or the `StdMeshers` source
set. It turns a missing toolkit, a dead-stripped SMESH object, or an un-built
translation unit into a named failure instead of a surprise mid-binding.

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=<env>/Library -DPython_EXECUTABLE=<env>/python.exe \
      -DPYSMESH_BUILD_V2_PROBE=ON
cmake --build build --target v2_probe
./build/v2_probe.exe                             # exit 0 == every probed capability is usable
```

## Design principles

- **Narrow API.** Every exported function exists to serve a concrete CAD or meshing
  pipeline need. No SWIG, no `smeshBuilder` emulation, no MED/CGNS I/O. Data crosses the
  boundary as NumPy arrays and BREP bytes.
- **Fail loud.** Every failure is a typed `pysmesh.PysmeshError` carrying the underlying
  SMESH/OCCT message and, where applicable, the offending ids. Never a silent
  best-effort fallback.

## Provenance & licensing

Every vendored source and patch is traced in [PROVENANCE.md](PROVENANCE.md); the
third-party component table is in [NOTICE.md](NOTICE.md).
