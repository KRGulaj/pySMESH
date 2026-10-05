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
* **`netgen`** (netgen 6.2.2101 through NETGENPlugin, since Phase 5): each NETGEN
  algorithm on the primitives, with the counts, the volume error against the exact volume
  and the smallest angles. See the Phase 5 section below.
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

## Explained differences since the Adaptive1D patches

`patches/smesh/StdMeshers_Adaptive1D_deflection.patch` makes `Adaptive1D` hold its
documented deflection. One probe changes:

| Probe | Before the patch | With the patch | Cause |
|---|---|---|---|
| `1d/adaptive_circle` | 28 segments, chords 0.332348 to 0.352196, sum 9.404985 | 29 segments, chords 0.307501 to 0.331708, sum 9.406311 | On the radius-1.5 rim the longest chord had the sagitta 0.0103727, 1.037 times the deflection 0.01. Now the largest sagitta is 0.009198 (0.920 times), every chord lies in [0.05, 1.0], and neighbours differ by at most 1.058 times. |

`patches/smesh/StdMeshers_Adaptive1D_bounds.patch` (the size bounds and the factor 2 on every
edge) and `patches/smesh/StdMeshers_Cartesian_VL_cancel.patch` (a cancel during the viscous
layer steps of `Cartesian_3D`) change no probe value.

## Explained differences since the distribution patches (Phase 4)

`patches/smesh/StdMeshers_Distribution_table.patch` (report S1) and
`patches/smesh/StdMeshers_Distribution_expression.patch` (report S2) make the TABLE and
EXPRESSION distributions place node k of N where the integral of the density, from the start of
the edge, reaches k/N of its total. Two mesh probes change. Both edges are 11 long, with N = 5.

| Probe | Before the patches | With the patches | Closed form |
|---|---|---|---|
| `1d/table_5` | first 3.368683, last 1.538424 | first 3.368484, last 1.538382 | density 1 + 2t, F(t) = t + t^2; t_k = (-1 + sqrt(1 + 8k/5)) / 2 |
| `1d/expression_5` | first 2.868500, last 1.573664 | first 2.868324, last 1.573894 | density 1 + t^2, F(t) = t + t^3/3; t_k is the real root of t^3 + 3t - 4k/5 = 0 |

The new values equal the closed forms to 3.8e-14. The old values missed them by up to 2.3e-4,
because the bisection stopped at an absolute 1e-4 of the edge (S1) and the expression was
integrated by one 20-point Gauss rule (S2). The `defect` probes `S1/table_15m_first_cell_*` and
`S2/expression_15m_1_over_0.0003_plus_t` now give the exact wall cells (3.1085 um, 306.56 um,
380.26 um) instead of "no message" and cells up to 50 % off. No other probe changes, and no
Cartesian grid changes: Cartesian spacing functions call `FunctionExpr::value` only.

## Explained differences since the validity guards (Phase 4, group 2)

The guards of report §6 (V1, V2, V3, V5), the sign check of P2, the refusal of A8 and the
statuses of A7 change no `geometry` or `mesh` probe: every committed result they accept is
bit for bit. Four `defect` probes change, each to a refusal or to a named reason:

| Probe | Before | Now | Issue |
|---|---|---|---|
| `V1/sphere_near_coincident_grid` | 40 empty cells | 40 cells raise "OCCT returned no solid although the operands overlap" | V1 |
| `V2/wing_cut_free_edges` | a solid with 2 free edges | raises "the result has 2 free boundary edge(s)", naming both (lengths 1.5 and 0.866012) | V2 |
| `V3/inside_out_box_import` | imported at volume -1 | raises "the BREP holds 1 solid(s) that are inside out: solid 1 (volume -1)" | V3 |
| `O4/bowed_end_section` | "invalid shape", no details | the same refusal, details "SHELL 1 of the result: BRepCheck_NotClosed on SOLID 1" | A7 |

## Explained differences since the mesher reporting fixes (Phase 4, group 3)

The reporting fixes of report §2 (A1, A6), §17.2 (M1), N1 and §18.1 (R1) change no
`geometry` or `mesh` probe. Two `defect` probes change:

| Probe | Before | Now | Issue |
|---|---|---|---|
| `A1/reduced_quad_warning` | raises "meshing failed on 1 sub-shape(s)" with SMESH's warning text | meshed: 461 nodes, 398 quadrangles, 24 triangles, 100 edges; one warning on `ComputeReport.warnings` | A1 |
| `A6/missing_hypothesis_message` | "failed on 0 sub-shape(s)", "SMESH reported no per-sub-shape error text" | "failed on 12 sub-shape(s)", each of EDGE 1-12 named "Regular_1D is missing a hypothesis it needs (algorithm state MISSING_HYP)" | A6 |

The A1 oracle is the STANDARD mesh of the same face: the REDUCED request falls back to it,
and SMESH says so. The new counts equal the STANDARD counts, and `kept_faces` (422) is the
face count the reference kept when it raised. Node coordinates and connectivity equal the
STANDARD mesh, with one limit. Two builds of this four-spline NACA face in one process can
differ by 4.2e-17 in node coordinates, for STANDARD alone too (report §18.1, M2), so the
equality is bit for bit only between meshes of one build of the face
(`docs/reports/agents/phase4_fixes/g3/a1_golden_oracle.txt`, gitignored).

## Explained differences since the construction fixes (Phase 4, group 4)

The construction fixes of report §2 (A2, A3), §4 (C3), §6 (V4), §9 (O4) and §18.1 (M2) change
no `geometry` or `mesh` probe. A loft now runs on copies of its sections, and its results are
byte for byte those of the reference. One `defect` probe changes:

| Probe | Before | Now | Issue |
|---|---|---|---|
| `O4/bowed_end_section` | "invalid shape", details "SHELL 1 of the result: BRepCheck_NotClosed on SOLID 1" | "the solid loft has no valid cap: section 1 of 3 (first) is not planar: its points spread 1.02355e-05 across the plane that fits them best, and OCCT could not close the solid with a planar face on it" | O4 |

The probe bows the first section by w sin(pi x) with w = 1e-5. That bow spreads the section's
points exactly w across the plane z = const, so the reported 1.02355e-05 is w within 2.4 %:
the least-squares plane of the samples tilts a little from z = const.

## Explained differences since the geometry query fixes (Phase 4, group 5)

The query fixes of report §5 (D1, D2, D3) and §4 (C5) change three `geometry` probes and
two `defect` probes. C5 adds an argument and a query; with the default `distinct=False`,
no result changes. The two OCCT patches of finding F2 change no probe: the old and the
patched OCCT, on one commit, gave a golden capture with 0 differences at rtol 0.

| Probe | Before | Now | Issue |
|---|---|---|---|
| `exchange/production_step.volume_fixed_rule` | 3.3671561379855857 | 3.3684720794880767 | D3 |
| `query/bbox_and_distance.bbox0_xmin` | -1e-07 | 0.0 | D1 |
| `query/bbox_and_distance.bbox0_xmax` | 3.0000001 | 3.0 | D1 |
| `query/bbox_and_distance.bbox1_xmin` | 4.9999999 | 5.0 | D1 |
| `D1_D2/bbox_padding_and_spline.line_xmin` | -1e-07 | 0.0 | D1 |
| `D1_D2/bbox_padding_and_spline.spline_xmin` | 4.99718766183329 | 4.999912477096879 | D2 |
| `D3/fixed_rule_wing_volume.fixed` | 0.10610073921072888 | 0.13290856218656688 | D3 |

- `production_step` sums the `Shape.solids()` volumes. D3 makes them adaptive at the
  default relative precision 1e-6; the key keeps its old name. `mass_properties` at 1e-9
  gives 3.368472869988658, within 2.3e-7 relative of the new value. The fixed rule was
  3.9e-4 low.
- `bbox_and_distance` bounds a 3 x 7 x 11 box at the origin and a unit sphere centred at
  x = 6. The boxes no longer carry the shape tolerance 1e-7: they are the extents of the
  geometry, 0 and 3 for the box, 5 for the sphere.
- `bbox_padding_and_spline`: the line's box is its end points. The spline's new minimum is
  the minimum of 200 001 curve samples, 4.99991258, less OCCT's pad of 1e-7 for a B-spline
  (`Precision::Confusion()`). The old minimum was the control polygon, 2.72 mm outside.
- `fixed_rule_wing_volume.fixed` calls `mass_properties` without a precision, which is now
  the adaptive rule at 1e-6. The section integral of the wing is 0.132908562138116; the new
  value is 3.6e-10 relative above it.

## Explained differences since the Prism3D and limits group (Phase 4, group 6)

The two-level `entities_in_box` (amendment 5), the Prism_3D patch of report §3 B1 and the
documentation of §8 S3-S6 change no `geometry` or `mesh` probe. One `defect` probe changes:

| Probe | Before | Now | Issue |
|---|---|---|---|
| `B1/prism_unequal_edge_counts` | raises, "Composite 'horizontal' edges are not supported" | meshed: 225 nodes, 128 hexahedra, 160 quadrangles, 68 edges; cell volumes sum to 2.377641290737884 | B1 |

The probe is the n = 5 prism of report §3 B1. Prism_3D used to reject the 5-edge side face
as the source, mesh the solid from another face and still report the rejected face's error.
The patch clears that error, refuses a side face with a composite horizontal side, and the
search goes on to a face it can sweep from. The oracle is the volume: the regular pentagon
of circumradius 1 has the area 5/2 sin(72 degrees) = 2.3776412907378837, and the prism is
1 high, so the cell volumes match it to 1.9e-16 relative. The smallest corner angle is 54
degrees, and every cell has a positive volume.

## Explained differences since the ids and labelled export group (Phase 4, group 8)

Group 7 changes no probe. Group 8 (report §4 A5, C2, C4) changes one `geometry` probe and
one `defect` probe:

| Probe | Before | Now | Issue |
|---|---|---|---|
| `exchange/step_round_trip.face_labels` | 0 | 1 | C4 |
| `C2/export_handoff_after_common.details` | names only a same-domain merge; lists no ids | names a boolean as a cause and `allow_aliases=True` as the way out; ends "The ids, by kind: SOLID 1, 28." | A5, C2 |

- `step_round_trip` writes a box with `face_names={1: "inlet"}` and reads it back. OCCT
  writes and reads a face name only with `write/read.stepcaf.subshapes.name` on, and both
  are off by default, so the name was lost on the way out. The reader also looked up names
  on top-level labels only. Now the name is the name of the face's `ADVANCED_FACE` entity,
  and it reads back on face 1. The count of 1 is the one face named.
- `export_handoff_after_common` is the overlapping `common` of report §4 C2. Its two solid
  ids (1 and 28) both denote the result. The refusal keeps its message, `not a bijection`.
  Its details now list the ids by kind (A5) and name the boolean cause and the
  `allow_aliases=True` way out (C2).

## Explained differences since the viscous layer and MEFISTO group (Phase 4, group 10)

Group 9 changes no probe. Group 10 (brief §6 L1-L7, amendment 8 for C2, amendment 9 for E6,
amendments 10-12) changes no `geometry` probe, 5 `mesh` probes and 1 `defect` probe. The
Cartesian layer patches, the leak patches, the layer refusals, the offset error of amendment
11 and the refusal of a concurrent assignment (amendment 10) change no probe: no probe builds
Cartesian layers, assigns layers to an algorithm that does not build them, or makes a
concurrent assignment.

| Probe | Before | Now | Issue |
|---|---|---|---|
| `2d/mefisto_box` | 440 triangles, 222 nodes, smallest angle 0.187 degrees, largest aspect ratio 314 | 1136 triangles, 570 nodes, smallest angle 9.61 degrees, largest aspect ratio 4.01 | E6 |
| `edit/smooth_merge_quad_to_tri` | the 440-triangle mesh; smallest angle 0.187 before and 18.43 after smoothing | the 1136-triangle mesh; 9.61 before and 20.46 after | E6 |
| `2d/projection_opposite_faces`, `2d/projection_1d2d_opposite_faces` | counts as now; aspect sum 364.71, angle sum 4051.70 | aspect sum 357.52, angle sum 4050.75 | E6 |
| `3d/prism_extruded_triangle` | counts as now; smallest cell volume 1.068, smallest angle 14.36 degrees | smallest cell volume 1.254, smallest angle 16.23 degrees | E6 |
| `C2/export_handoff_after_common.details` | ends "... and Handoff.aliases maps every other id to it." | names `Handoff.face_ids_of` (and the same for the other kinds) as the list of every live id of each sub-shape | C2, E5 |

- All four E6 probes mesh with `MaxElementArea`. MEFISTO turns the area `A` into the edge
  bound 1.5 sqrt(2 A / sqrt 3) and used to clamp it: a bound up to 2.05 times the longest
  boundary edge was kept, so the interior grew coarser than the boundary. Now the bound is
  capped at the longest boundary edge (`MEFISTO_2D_max_element_area.patch`). Measured in
  `g10/e6_s6_*.txt` and `g10/e6_golden.txt` of the phase record.
- `mefisto_box` is the 3 x 7 x 11 box with 1 m boundary edges and `MaxElementArea(1.0)`.
  The bound of 1.61 m was kept, and the triangles grew to an area of 1.77 with 0.187 degree
  slivers. Capped at 1 m, the mesh is the one the boundary alone gives (1136 triangles, the
  same as with `MaxElementArea(4.0)` or no 2-D hypothesis), and every triangle has an area of
  at most 0.573. The smoothing probe meshes the same box, so its angles follow.
- The projection probes mesh the box with 3 segments per edge and `MaxElementArea(3.0)`. On
  the two 3 x 7 faces the bound of 2.79 is now capped at their longest edge, 2.33, and the
  largest triangle goes from 2.68 to 2.30; the counts stay. On the other four faces the 3.67
  boundary edges are too long for the bound (largest triangles 5.26 and 4.12, as before),
  which the compute now reports as 4 warnings; their meshes do not change.
- The prism probe meshes its 6 x 5 triangle base with `MaxElementArea(3.0)`: the bound of
  2.79 is capped at the longest base edge, 2.60. The base keeps its 11 triangles, the largest
  goes from 2.147 to 2.153, both within 3, and the cells are better shaped.
- `export_handoff_after_common` refuses as before (`not a bijection`). Amendment 8 replaced
  `Handoff.aliases` with per-ordinal id lists, and the details name them.

## Explained differences since the Prism_3D composite side group (Phase 5, group 4)

Phase 5 groups 1 to 3 change no probe. Group 4 (PR1) changes one `defect` probe. The
`geometry` and `mesh` groups are equal to the reference bit for bit (`--rtol 0 --atol 0`).

| Probe | Before | Now | Issue |
|---|---|---|---|
| `B1/prism_unequal_edge_counts` | swept from a side face: 225 nodes, 128 hexahedra, 160 quadrangles, 68 edges | swept from the pentagon cap: 145 nodes, 64 hexahedra and 16 pentahedra, 112 quadrangles and 8 triangles, 60 edges; cell volumes sum to 2.377641290737884, as before | PR1 |

The probe is the n = 5 prism of report §3 B1, with 2 segments on each half-edge. Up to
5.0.0, Prism_3D refused the pentagon cap as the source, because the 5-edge side face has a
composite horizontal side, and swept from a side face. That sweep gave 8 segments to two
edges whose hypothesis gives 4. PR1 projects the whole side onto the composite side. The
hypotheses of the two half-edges give the 2 segments that the projection puts on each, so
Prism_3D now sweeps between the caps. The oracle is the structure of a straight prism:

- the pentagon cap has 29 nodes, and 5 levels give 145 nodes;
- its 16 quadrangles and 4 triangles give 64 hexahedra and 16 pentahedra over 4 layers;
- the walls take 20 segments x 4 layers = 80 quadrangles, and the two caps 32 more;
- the cap boundaries take 2 x 20 segments, and the 5 vertical edges 4 each: 60 edges;
- the cell volumes sum to the prism volume 5/2 sin(72 degrees) = 2.3776412907378837
  (1.9e-16 relative), and every cell has a positive volume.

## The NETGEN probes (Phase 5, NETGEN item, group 2)

Phase 5 links netgen 6.2.2101 and NETGENPlugin V9_16_0 into `_core`. The capture records
`with_netgen: True` in `meta` (False before), and a new group, `netgen`, of 12 probes. No
`geometry`, `mesh` or `defect` value changes: the capture equals the Phase 4 reference at
`--rtol 0 --atol 0`. `compare.py` prints a probe that the baseline lacks with its values,
and it does not block: it has no baseline value to differ from. Before Phase 5 the script
printed only four named groups.

Each probe runs `_mesh_stats` (counts by type, the quality controls) and adds two values:

* `volume_rel_error`: |sum of the cell volumes - exact volume| / exact volume, for a solid
  primitive. A planar-faced solid gives 0 to round-off. A curved one gives the deficit of its
  chords, which grows with `max_size` over the radius.
* `min_dihedral_deg`: the smallest dihedral angle of the tetrahedra, linear or quadratic
  (corners only); `null` for a surface mesh.

The shapes are the primitives of the `geometry` group: the 3 x 7 x 11 box, the cylinder of
radius 1.5 and height 4, the sphere of radius 1.7 and the torus (3, 0.8).

| Probe | Nodes | Faces | Volumes | `volume_rel_error` | Smallest face angle (deg) | `min_dihedral_deg` |
|---|---|---|---|---|---|---|
| `1d2d3d_box` (max 2.0) | 80 | 156 | 166 | 0 | 28.9 | 24.8 |
| `1d2d3d_cylinder` (max 0.6) | 192 | 314 | 576 | 2.09e-2 | 39.6 | 19.7 |
| `1d2d3d_sphere` (max 0.5) | 212 | 340 | 639 | 3.35e-2 | 42.8 | 15.9 |
| `1d2d3d_torus` (max 0.4) | 950 | 1688 | 2633 | 3.31e-2 | 35.3 | 3.2 |
| `1d2d_then_3d_partitioned_box` (2 x 1 x 1 in two solids, max 0.3) | 139 | 242 | 431 | 1.1e-16 | 33.4 | 22.7 |
| `regular_2d_3d_cylinder` (LocalLength 0.6) | 178 | 330 | 454 | 2.10e-2 | 37.4 | 16.6 |
| `quadrangle_faces_3d_box` (4 segments, quadrangles, pyramids) | 154 | 96 | 455 | 0 | 90.0 | 7.9 |
| `1d2d_sphere_surface` (max 0.4) | 260 | 516 | 0 | - | 40.6 | - |
| `simple_3d_box` (4 segments per edge) | 173 | 216 | 606 | 0 | 11.1 | 22.3 |
| `second_order_sphere` (max 0.6) | 649 | 230 | 289 | - | 41.8 | 12.1 |
| `local_size_on_face_box` (0.5 on face 1, max 2.0) | 921 | 1132 | 3561 | 0 | 23.7 | 17.0 |
| `chordal_error_cylinder_surface` (0.01, max 0.6) | 617 | 1230 | 0 | - | 36.8 | - |

- netgen gives one mesh at any thread count and on repeat (`tests/test_mesher_netgen_threads.py`),
  so these values do not depend on the machine's thread count.
- `simple_3d_box` gives every edge 4 segments, so the 3 m and the 11 m edges make long thin
  triangles: 11.1 degrees.
- `quadrangle_faces_3d_box` keeps the 96 quadrangles of `Quadrangle2D`. NETGEN_3D puts a
  pyramid on each and fills the rest with tetrahedra.
- `second_order_sphere` and the two surface probes have no `volume_rel_error`: a quadratic
  cell's volume and a surface mesh have no exact counterpart here.
