# Geometry Functions Reference

Generated from pySMESH's standalone OCCT modules and the native `_core` extension. See
[Geometry operations](../guides/geometry-operations.md) for a walked-through guide to this
surface, and [Units](../concepts/units.md) for the STEP and IGES length-unit contract.

## STEP

::: pysmesh.step
    options:
      show_root_heading: false
      heading_level: 3

## IGES

::: pysmesh.iges
    options:
      show_root_heading: false
      heading_level: 3

## Tessellation

::: pysmesh.tessellate
    options:
      show_root_heading: false
      heading_level: 3

## Offsets

::: pysmesh.offset
    options:
      show_root_heading: false
      heading_level: 3

## Distance and leaks

::: pysmesh.distance
    options:
      show_root_heading: false
      heading_level: 3

## Point classification

::: pysmesh.classify
    options:
      show_root_heading: false
      heading_level: 3

## Same-domain healing

::: pysmesh.unify
    options:
      show_root_heading: false
      heading_level: 3

## Viscous layers

::: pysmesh.viscous
    options:
      show_root_heading: false
      heading_level: 3

## The native extension

`Shape`, `FaceInfo`, `EdgeInfo`, `SolidInfo`, `VertexInfo`, `Mesh`, `load_brep`,
`PysmeshError` and `PysmeshCancelled` are implemented in the compiled `_core` extension and
typed in `src/pysmesh/_core.pyi`. All are re-exported from the top-level `pysmesh` package.

::: pysmesh._core
    options:
      show_root_heading: false
      heading_level: 3
      filters:
        - "^Shape$"
        - "^FaceInfo$"
        - "^EdgeInfo$"
        - "^SolidInfo$"
        - "^VertexInfo$"
        - "^Mesh$"
        - "^MeshStats$"
        - "^load_brep$"
        - "^PysmeshError$"
        - "^PysmeshCancelled$"

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
