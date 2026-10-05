# Changelog

## 5.0.0

5.0.0 moves both native kernels to new releases and fixes every pySMESH defect found in a full
sweep of 4.2.2. The public API only grows: no name is removed. The major number marks
changed results and new refusals. Code that ran on 4.2.2 can now get a different mesh,
a different measure, or an exception where 4.2.2 returned a bad result in silence.

### Native kernels

- **SALOME SMESH, KERNEL and GEOMUtils `V9_9_0` → `V9_16_0`.** Four years of upstream
  algorithm work, including the rewritten Cartesian mesher. KERNEL's basics now come from
  `salome_bootstrap` `V9_16_0`. `Mefisto2D` stays public: upstream deleted it in 2022, so
  its `V9_9_0` sources are carried forward unchanged.
- **OCCT 8.0.0 → 8.0.1, built from source** by `ci/build_occt.py`, with only the toolkits
  that `_core` needs. Three OCCT patches:
  - the ruled loft history through four or more closed single-edge sections;
  - a native crash in chamfer;
  - a native crash in pipe.
- **SALOME patches:** 36 on SMESH, KERNEL and GEOM. Among the fixes: the TABLE and
  EXPRESSION distributions, the `Adaptive1D` crash, its deflection and size bounds, and the
  Cartesian cancel. `PROVENANCE.md` lists every patch.

### Wheel

| | 4.2.2 | 5.0.0 |
|---|---|---|
| Wheel (cp313) | 43 345 198 bytes (41.3 MiB) | 39 991 806 bytes (38.1 MiB) |
| Bundled DLLs | 75 | 52 |

NumPy is still the only runtime dependency. OCCT, Boost and VTK stay private to the wheel.

### Changed results

If your code compares against stored values from 4.x, expect these to move.

- **TABLE and EXPRESSION distributions** place their nodes at the positions the function
  defines. 4.x used an absolute bisection tolerance and a single 20-point Gauss rule, so
  steep tables and expressions were off.
- **`Adaptive1D`** no longer crashes on a shape with faces. Every edge now meets its
  deflection, `[min_size, max_size]` and the factor 2 between neighbours. Segment counts
  change only on the edges that broke a rule before.
- **Measures** (the default volume, area and length) are adaptive at relative precision
  1e-6, as `Session.mass_properties`. 4.x gave a lofted wing's volume 20 % low.
- **Bounding boxes** are the geometry's box. They are no longer grown by the shape
  tolerance, and a B-spline box bounds the curve, not its poles.
- **`Mefisto2D`** honours `MaxElementArea`. In 4.x the bound had no effect.
- **The medial axis** samples straight edges with at least 10 points (SMESH 9.16).
- **A SMESH warning no longer fails the compute.** It is reported in
  `ComputeReport.warnings`.

### New refusals

Each one raises `PysmeshError` with the reason and the offending ids. In 4.x each case
returned a result that was wrong, or crashed the process.

- A boolean whose result is empty or not watertight.
- An inside-out solid on import. Pass `inside_out="reverse"` to `load_brep` or
  `Session.add_brep` to repair it instead.
- `sew(make_solid=True)` when no solid can be made.
- A swept or lofted solid that crosses itself, including sections that meet at one point.
- A solid loft with a non-planar end section, and a loft with a slit inside.
- A NaN or infinite argument to a geometry operation. Before, it reached OCCT and could
  crash the process.
- An assignment that makes the model ambiguous (SMESH's `HYP_CONCURRENT`). `assign` raises
  and undoes the assignment.
- A hypothesis value that SMESH refuses.
- Viscous layers under an algorithm that does not build them. `Hexa3D`,
  `PolyhedronPerSolid3D` and `Cartesian3D` build 3-D layers. `CompositeHexa3D` with layers is
  refused.
- A layer stack with `total_thickness <= 0`, `layer_count < 1` or `stretch_factor < 1`, at
  construction. `stretch_factor = 1` (equal layers) is now accepted.
- `Prism3D` on a composite horizontal side. The message names the way out. 4.x threw
  `Standard_NotImplemented: Adaptor3d_Surface::EvalD0` there.

### New API

- **Algorithms and hypotheses** from SMESH 9.16: `SegmentAroundVertex0D`, `UseExisting1D`,
  `UseExisting2D`, `PropagOfDistribution`, `LayerDistribution2D`, `LengthFromEdges`,
  `BlockRenumber`, `NotConformAllowed`.
- **New fields:**
  - `reversed_edges` on `NumberOfSegments`, `Arithmetic1D`, `StartEndLength`, `Geometric1D`
    and `FixedPoints1D`;
  - `Distribution.BETA_LAW` with `beta`;
  - `enforced_vertices` and `enforced_points` on `QuadrangleParams`;
  - explicit grid coordinates, `fixed_point`, `axis_directions`, `use_quanta` / `quanta` and
    `threshold_for_internal_faces` on `CartesianParameters3D`.
- **Viscous layers:** `ViscousLayerBuilder`, with `Mesher.shrink_geometry` and
  `Mesher.add_layers` (shrink the geometry, mesh it, then add the layers), and
  `first_layer_thickness(total_thickness, stretch_factor, layer_count)`.
- **Mesh operations and controls:** `Mesher.make_boundary_mesh` with `BoundaryDimension`,
  `Mesher.ray_volumes` returning `RayVolumeHits`, `split_volumes(avoid_over_constrained=...)`,
  `add_nodes` / `add_elements` bound to a sub-shape (`on=`, `parameters=`), and the
  `Warping3D` and `ScaledJacobian` quality controls.
- **Session:**
  - `entities(kind, distinct=True)` and `alias_groups(kind)`;
  - `export_handoff(allow_aliases=True)`, with `Handoff.solid_ids_of`, `face_ids_of`,
    `edge_ids_of` and `vertex_ids_of`;
  - `Session.write_step`, with face names keyed by session id;
  - a closed loft: name the first section again as the last;
  - `HistoryDelta.warnings`.
- **Reporting:** `ComputeReport.warnings` (`ComputeWarning`: kind, ordinal, algorithm,
  text).
- **Misc:** `pysmesh.__version__`. `read_iges` takes the file content as bytes, or a path.

### Migration

- `read_iges(path=...)` is now `read_iges(data_or_path=...)`. A positional call is
  unchanged.
- If you caught the exception that a SMESH warning raised, read `ComputeReport.warnings`
  instead.
- If you import solids of unknown orientation, pass `inside_out="reverse"`.
- If you assign hypotheses on overlapping sub-shapes, assign on the shared sub-shape first:
  it takes priority over those around it.
- Re-baseline stored meshes and measures that use TABLE or EXPRESSION distributions,
  `Adaptive1D`, `Mefisto2D` with `MaxElementArea`, bounding boxes, or default measures.

### Known limits

- `Prism3D` on a composite horizontal side is refused, not meshed.
- Viscous layers with `CompositeHexa3D` are refused.
- Cartesian viscous layers thicker than the shape allows are refused with the reason.
- OCCT pads B-spline and Bezier boxes by 1e-7. A strict `entities_in_box` query at a
  spline's exact extent misses by that amount.
- There is no free tetrahedral volume mesher. NETGEN is planned.

## Earlier versions

Releases before 5.0.0 have no detailed entries. This is what each major line added. Every
wheel is on the [GitHub Releases](https://github.com/KRGulaj/pySMESH/releases) page. PyPI
carries 4.0.0 and later.

| Line | First release | What it added |
|---|---|---|
| 4.x | 4.0.0, 2026-08-24 | VTK is bundled privately in the wheel, as OCCT and Boost already were. The host no longer needs VTK 9.6.2, and the import-time VTK check is gone. One wheel per interpreter, CPython 3.11 to 3.14. |
| 3.x | 3.0.0, 2026-08-09 | `Mesher`: SMESH's full meshing pipeline. Algorithms and hypotheses assigned per sub-shape, mesh editing, search, quality controls and the medial axis. |
| 2.x | 2.0.0, 2026-08-09 | `Session`: stateful OCCT CAD modelling with persistent entity ids. Primitives, booleans with history, fillets, chamfers, transforms, healing and tessellation of the live shape. 2.1 to 2.3 shipped beside 3.1 to 3.3, with the same CAD and IGES additions. |
| 1.x | 1.0.0, 2026-07-12 | One SMESH algorithm, `compute_viscous_layers` (`StdMeshers_ViscousLayers`, 3-D boundary-layer prisms). With it: OCCT `unify_same_domain` healing and standalone OCCT geometry operations (STEP, tessellation, offsets, distance and classification). |
