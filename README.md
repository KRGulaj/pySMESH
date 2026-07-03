# pySMESH

Standalone Python bindings to SALOME **SMESH**'s `StdMeshers_ViscousLayers` — 3-D
boundary-layer (prism) meshing — packaged as a self-contained `cp313-win_amd64` wheel.

- **License:** LGPL-2.1-only (see [LICENSE](LICENSE), [NOTICE.md](NOTICE.md))
- **Platform:** Windows x64, CPython 3.13
- **Runtime dependency shared with the host:** **VTK 9.6.2** (exact, checked at import)

## Why this exists

SALOME SMESH's `StdMeshers_ViscousLayers` is a mature, production-grade 3-D boundary-layer
(prism) mesher, but it has no standalone Python wrapper: SMESH ships as part of the full
SALOME platform, wrapped through CORBA/SWIG and pulling in the entire SALOME GUI/KERNEL
stack just to reach one meshing algorithm. pySMESH strips SMESH down to the minimum static
library set the viscous-layer algorithm needs and exposes it as a plain, pip-installable
Python module — no SALOME platform, no CORBA, no GUI.

Doing that standalone strip also solves a second, more mundane problem: SMESH pulls in Open
CASCADE (OCCT) and Boost as dependencies, and installing `occt`/`boost` directly into a host
application's environment can trigger a dependency solver cascade that downgrades unrelated
packages (VTK, Qt bindings, MKL, etc.) — a real risk for any app with a carefully pinned
scientific-Python stack. pySMESH's build makes that impossible by construction:

- **SMESH + KERNEL are statically linked** into a single `_core.pyd`.
- **OCCT and Boost are private** to that binary — their DLLs are **bundled into the wheel**,
  so they never appear in the host env and cannot perturb its dependency solve.
- **VTK is the one shared dependency** (SMESH's data structure is built on
  `vtkUnstructuredGrid`). It is linked **dynamically against the host's own VTK** and its
  version is **hard-checked at import** — a mismatch raises `ImportError` instead of risking
  a silent ABI crash. This is the only version coupling a consuming application needs to track.

Net effect on the host env: installing the wheel adds **one** pip entry (`pysmesh`) and
nothing else — no `occt`, no `boost`, no VTK downgrade.

> **Binary size:** `_core.pyd` is a few MB and the bundled OCCT/Boost DLLs add tens of MB.
> That is expected — it is the deliberate trade for zero OCCT/Boost footprint in the host
> environment, and still smaller than shipping OCCT + its transitive DLLs separately.

## Install

```bash
pip install pysmesh-0.1.0-cp313-win_amd64.whl
```

The host environment must already provide **VTK 9.6.2** (the version pySMESH was built
against). `import pysmesh` verifies this and fails loudly otherwise.

## Quickstart

`examples/box_bl.py` is the end-to-end walkthrough: load a BREP solid, inject a classified
surface mesh, and grow five prism layers on every wall.

```python
import numpy as np
import pysmesh

shape = pysmesh.load_brep(open("box.brep", "rb").read())
mesh = pysmesh.Mesh(shape)

node_ids = mesh.add_nodes(nodes)                       # (N,3) float64 -> SMESH ids
mesh.classify_on_face(node_ids[face_nodes], face_id, uv)   # CAD classification
mesh.classify_on_edge(node_ids[edge_nodes], edge_id, t)
mesh.classify_on_vertex(int(node_ids[k]), vertex_id)
mesh.add_segments(node_ids[edge_conn], edge_id)        # 1-D elements (required by VL)
mesh.add_triangles(node_ids[tri_conn], face_id)        # 2-D elements
mesh.validate()

result = pysmesh.compute_viscous_layers(
    mesh,
    pysmesh.VLParams(
        face_ids=tuple(f.id for f in shape.faces()),
        total_thickness=0.1, n_layers=5, stretch_factor=1.2, group_name="BL",
    ),
)
result.prism_connectivity   # (K,6) int32 — VTK wedge order, row-indexed into node_coords
result.node_coords          # (P,3) float64
result.inner_surface_tris   # (S,3) int32 — the shrunk inner surface
result.failed_face_ids      # walls that received no layers
```

See `src/pysmesh/_core.pyi` for the full typed API. `mypy --strict` type-checks against it.

## Build from source

Requires MSVC v143, GNU `patch`, and a conda-forge build environment (VTK **pinned** to the
host application's version; OCCT/Boost free to resolve):

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

## Design principles

- **Narrow API.** No general meshing-API parity with SALOME — every exported function exists
  to serve a concrete boundary-layer-meshing use case. No SWIG, no `smeshBuilder` emulation,
  no MED/CGNS I/O (arrays cross the boundary as NumPy).
- **Fail loud.** Every failure is a typed `pysmesh.PysmeshError` carrying the underlying
  SMESH/OCCT message and, where applicable, the offending face ids — never a silent
  best-effort fallback.

## Provenance & licensing

Every vendored source and patch is traced in [PROVENANCE.md](PROVENANCE.md); the third-party
component table is in [NOTICE.md](NOTICE.md).
