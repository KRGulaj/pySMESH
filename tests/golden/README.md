<!--
SPDX-License-Identifier: LGPL-2.1-only
Copyright (C) 2026 Kajetan R. Gulaj
-->

# Upgrade baseline (Phase 0)

This folder holds the baseline that the dependency upgrade must reproduce: OCCT 8.0.0 →
8.0.1 built from source, then SMESH `V9_9_0` → `V9_16_0`. The baseline is taken on 4.2.2,
before any upgrade. Every later phase captures again and compares. A phase merges only when
nothing outside the `defect` group differs, or when each difference is explained.

pytest does not collect this folder. No file here is named `test_*.py`.

## Files

| File | What it is |
|---|---|
| `pytest_4.2.2.json` | Every test node id with its outcome and duration. 1 274 tests: 1 272 passed, 2 skipped, 0 failed (256 s, slow tests included). |
| `baseline_4.2.2.json` | 89 golden probes: values, not assertions. |
| `pytest_summary.py` | Reduces a JUnit XML file to the table above, and compares two tables. |
| `capture.py` | Runs the probes and writes a golden file. |
| `compare.py` | Compares two golden files, grouped and with tolerances. |

## Probe groups

* **`geometry`** (OCCT): primitives; `fuse`, `cut`, `common`, `fragment`, `split` and
  `section`; fillet and chamfer; lofts; sweeps; offsets; `sew`, `heal`, `unify` and
  `defeature`; tessellation; bbox, distance and `contains`; STEP and IGES round trips; the
  production STEP assembly.
* **`mesh`** (SMESH): every 1-D distribution; every 2-D and 3-D algorithm with a working
  recipe; viscous layers, both through the hypothesis and through
  `compute_viscous_layers`; mesh editing; the medial axis.
* **`defect`**: the reproductions in `docs/reports/defect_sweep_4.2.2.md`. These are
  expected to change in Phase 4. They are reported apart from the other groups and do not
  block.

Topology is counted on the BREP. Volumes and areas use the adaptive rule (1e-9) on a fresh
session read from the BREP. A shape that carries several ids after a boolean is then
measured once. The alias counts are recorded beside the measures (`solid_ids`,
`face_ids`).

Each probe runs in its own process. A native crash is recorded as the probe's value
(`crash: exit code 0x…`), not as the end of the capture. A probe that raises records
`{"error": …}`. A changed error is a changed behaviour.

## Determinism

Two captures on one machine match with `--rtol 0 --atol 0`: every value is bit-identical.
`compare.py` defaults to `--rtol 1e-9 --atol 1e-12`. Those tolerances admit the round-off
that a dependency upgrade may move, and nothing larger.

## Commands

Run every command from the repository root, through `conda run`. Calling the env's
`python.exe` by path skips the activation that puts `Library\bin` on `PATH`. An env can
hold an installed pysmesh in `site-packages` (`flux-pysmesh-build` holds a stale 0.2.0).
Set `PYTHONPATH` to `src` for pytest, so that the tree under test is imported.

Since Phase 1 the in-tree `_core.pyd` links our own OCCT 8.0.1 build
(`ci/build_occt.py`), which lives outside the env. Set `PYSMESH_OCCT_BIN` to that build's
`bin` directory. `tests/conftest.py` and `capture.py` add it with `os.add_dll_directory`,
because Python does not search `PATH` for an extension's DLLs. The README "Build from
source" section builds `_core` against it. `<env>` is the build env (Phase 1 used
`pysmesh-p1`). `<deps>` is the directory that holds the OCCT build.

```bash
export PYSMESH_OCCT_BIN=<deps>/occt-8.0.1/install/bin

# test suite
PYTHONPATH=src conda run -n <env> python -m pytest tests -p no:cacheprovider \
    --junitxml=junit.xml
conda run -n <env> python tests/golden/pytest_summary.py summarize junit.xml pytest_new.json
conda run -n <env> python tests/golden/pytest_summary.py compare \
    tests/golden/pytest_4.2.2.json pytest_new.json

# golden values (about 100 s)
conda run -n <env> python tests/golden/capture.py golden_new.json
conda run -n <env> python tests/golden/compare.py \
    tests/golden/baseline_4.2.2.json golden_new.json
```

Both compare commands exit with 0 on a match, and with 1 on a blocking difference.

## Recorded at 4.2.2

Build `4e5a4e1b7` (`_build_info.GIT_SHA`), captured on branch head `885d3c2be`: SALOME
`V9_9_0`, OCCT 8.0.0, VTK 9.6.2, Boost 1.90.0, Python 3.13.14, NumPy 2.5.0, Windows 11.

The capture recorded these errors, and they are part of the baseline:

| Probe | Recorded value | Cause |
|---|---|---|
| `1d/adaptive_circle` | `crash: exit code 0xC0000005` | Report A9. `Adaptive1D` on any shape with faces dereferences `NULL`. |
| `3d/hexa_from_skin_refuses_geometry` | "Algorithm can't work with geometrical shapes" | SMESH behaviour: `HexaFromSkin_3D` meshes from a skin mesh, not from a shape. |
| `S1/table_15m_first_cell_3e-06` | `details`: "EDGE 1: no message" | Report S1. |
| `O1/ruled_loft_4_circles` | `RuntimeError: …FindFromKey` | Report O1, A2 and A3. |
| `B1/prism_unequal_edge_counts` | `details`: `Standard_NotImplemented: Adaptor3d_Surface::EvalD0` | Report B1. |
| `A1/reduced_quad_warning` | warning raised as a failure | Report A1. |
| `A6/missing_hypothesis_message` | "0 sub-shape(s)" | Report A6. |
| `O4/bowed_end_section` | "invalid shape" | Report O4 and A7. |
| `C2/export_handoff_after_common` | not a bijection | Report C2. |

## Explained differences since SALOME V9_16_0

Since the SALOME stack moved from `V9_9_0` to `V9_16_0`, a capture differs from the
4.2.2 baseline in these probes. Each difference is explained; none is a port error.

| Probe | 4.2.2 | Since `V9_16_0` | Cause |
|---|---|---|---|
| `1d/adaptive_circle` | `crash: exit code 0xC0000005` | 28 segments | The crash (report B2) came from a vendored patch, which the upgrade dropped. The values equal SMESH 9.9 with the one-line fix. |
| `2d/quad_from_medial_axis_strip` | min angle 90 - 8e-13 deg | min angle 90 - 4.5e-6 deg | SMESH `27c8af8c6` discretises straight boundary edges for the medial axis, whose points are rounded to an integer Voronoi grid. |
| `B1/prism_unequal_edge_counts` (defect) | `Adaptor3d_Surface::EvalD0` | "Composite 'horizontal' edges are not supported" | The OCCT 8 `EvalD0` override of `TSideFace` removes the B1 exception; the case now meets an older SMESH refusal. |

`V9_16_0` also stopped `Cartesian_3D` from seeing a cancel during its run (SMESH
`9f7d4a55e`). `patches/smesh/StdMeshers_Cartesian_3D_cancel.patch` restores the `V9_9_0`
behaviour. That patch changes no probe value: a run without a cancel builds the same mesh.
