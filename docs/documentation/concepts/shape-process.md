# Shape Processing

`Session.shape_process(entities, operators, parameters)` runs OCCT's ShapeProcess operators
in order on the bodies of the named entities. It is GEOM's `ShapeProcess`: the steps of
`ShHealOper_ShapeProcess::Perform` (GEOM V9_16_0), with the copy map kept so that every id
is carried. This page lists every operator of OCCT 8.0.1 and every parameter that its
function reads (`ShapeProcess_OperLibrary.cxx`).

## Every parameter is given

`parameters` maps `"Operator.Name"` to a value, for example
`{"SplitAngle.Angle": 1.5707963267948966, "SplitAngle.MaxTolerance": 1e-7}`. The rules:

- Give every parameter of every named operator, and no other. A missing, unknown or
  wrongly typed parameter is refused, and the message names it.
- A **real** is a `float` or an `int`, finite. An **integer** is an `int`. A **bool** is a
  `bool`. A **continuity** is one of `"C0"`, `"G1"`, `"C1"`, `"G2"`, `"C2"`, `"C3"`, `"CN"`.
- There are no defaults. OCCT has a default for every parameter inside its operator
  functions, and GEOM's `ShHealing` resource file has values for many; `shape_process`
  uses neither.
- A parameter that an operator reads only under a condition is required too.
  SetTolerance reads `Value` and `Ratio` only when `Mode > 0`, and calls
  `BRepLib::EncodeRegularity` only when `Regularity` is set; all four are required, so
  `EncodeRegularity` always runs.

The sequence reads no resource file. The `CSF_ShHealingDefaults` and
`CSF_ShHealingUserDefaults` files that GEOM's resource manager reads change nothing, and a
real is passed with all its digits (GEOM's `SetParameter` keeps 6 significant digits).

## GEOM's values are a reference, not defaults

**Warning.** GEOM's values are absolute lengths in model units. `DropSmallEdges.Tolerance3d`
0.05 and `FixFaceSize.Tolerance` 0.05 are plausible for a model in millimetres, and 1000
times too large for a model in metres: they would drop real edges and faces. Choose every
value for the model.

The column "GEOM" gives the value in GEOM's `resources/ShHealing` (GEOM V9_16_0). A dash
means that GEOM's file sets no value, and OCCT's operator then uses its own. GEOM's default
sequence is FixShape, FixFaceSize, DropSmallEdges, SameParameter. One of GEOM's entries is
not in the tables: `SplitAngle.SegmentSurfaceMode`, which OCCT 8.0.1's SplitAngle does not
read. GEOM documents `ToBezier.EdgeMode` as 0, 1 or 2; OCCT 8.0.1 reads it as a bool.

## The operators

### DirectFaces

No parameter.

### SameParameter

| Parameter | Type | GEOM |
|---|---|---|
| `SameParameter.Force` | bool | 0 |
| `SameParameter.Tolerance3d` | real | 1e-7 |

### SetTolerance

| Parameter | Type | GEOM |
|---|---|---|
| `SetTolerance.Mode` | integer | - |
| `SetTolerance.Value` | real | - |
| `SetTolerance.Ratio` | real | - |
| `SetTolerance.Regularity` | real | - |

### SplitAngle

| Parameter | Type | GEOM |
|---|---|---|
| `SplitAngle.Angle` | real | 3.1415928 |
| `SplitAngle.MaxTolerance` | real | 0.1 |

### BSplineRestriction

| Parameter | Type | GEOM |
|---|---|---|
| `BSplineRestriction.SurfaceMode` | bool | 1 |
| `BSplineRestriction.Curve3dMode` | bool | 1 |
| `BSplineRestriction.Curve2dMode` | bool | 1 |
| `BSplineRestriction.Tolerance3d` | real | 1e-3 |
| `BSplineRestriction.Tolerance2d` | real | 1e-5 |
| `BSplineRestriction.Continuity3d` | continuity | C1 |
| `BSplineRestriction.Continuity2d` | continuity | C1 |
| `BSplineRestriction.RequiredDegree` | integer | 15 |
| `BSplineRestriction.RequiredNbSegments` | integer | 100 |
| `BSplineRestriction.PreferDegree` | bool | 1 |
| `BSplineRestriction.RationalToPolynomial` | bool | 1 |
| `BSplineRestriction.MaxDegree` | integer | 15 |
| `BSplineRestriction.MaxNbSegments` | integer | 10000 |
| `BSplineRestriction.OffsetSurfaceMode` | bool | 1 |
| `BSplineRestriction.OffsetCurve3dMode` | bool | 1 |
| `BSplineRestriction.OffsetCurve2dMode` | bool | 1 |
| `BSplineRestriction.LinearExtrusionMode` | bool | 1 |
| `BSplineRestriction.RevolutionMode` | bool | 1 |
| `BSplineRestriction.SegmentSurfaceMode` | bool | 1 |
| `BSplineRestriction.ConvCurve3dMode` | bool | 0 |
| `BSplineRestriction.ConvCurve2dMode` | bool | 0 |
| `BSplineRestriction.BezierMode` | bool | 0 |
| `BSplineRestriction.PlaneMode` | bool | 0 |
| `BSplineRestriction.ConicalSurfMode` | bool | - |
| `BSplineRestriction.CylindricalSurfMode` | bool | - |
| `BSplineRestriction.ToroidalSurfMode` | bool | - |
| `BSplineRestriction.SphericalSurfMode` | bool | - |

### ElementaryToRevolution

No parameter.

### SweptToElementary

No parameter.

### SurfaceToBSpline

| Parameter | Type | GEOM |
|---|---|---|
| `SurfaceToBSpline.LinearExtrusionMode` | bool | - |
| `SurfaceToBSpline.RevolutionMode` | bool | - |
| `SurfaceToBSpline.OffsetMode` | bool | - |

### ToBezier

| Parameter | Type | GEOM |
|---|---|---|
| `ToBezier.Curve3dMode` | bool | 1 |
| `ToBezier.Curve2dMode` | bool | 1 |
| `ToBezier.SurfaceMode` | bool | 0 |
| `ToBezier.Line3dMode` | bool | 1 |
| `ToBezier.Circle3dMode` | bool | 1 |
| `ToBezier.Conic3dMode` | bool | 1 |
| `ToBezier.SegmentSurfaceMode` | bool | 1 |
| `ToBezier.PlaneMode` | bool | 1 |
| `ToBezier.RevolutionMode` | bool | 1 |
| `ToBezier.ExtrusionMode` | bool | 1 |
| `ToBezier.BSplineMode` | bool | 1 |
| `ToBezier.MaxTolerance` | real | 0.1 |
| `ToBezier.MinCurveLength` | real | 0.05 |
| `ToBezier.EdgeMode` | bool | 0 |

### SplitContinuity

| Parameter | Type | GEOM |
|---|---|---|
| `SplitContinuity.Tolerance3d` | real | 1e-3 |
| `SplitContinuity.Tolerance2d` | real | - |
| `SplitContinuity.CurveContinuity` | continuity | C1 |
| `SplitContinuity.SurfaceContinuity` | continuity | C0 |
| `SplitContinuity.Curve2dContinuity` | continuity | - |
| `SplitContinuity.MaxTolerance` | real | - |

### SplitClosedFaces

| Parameter | Type | GEOM |
|---|---|---|
| `SplitClosedFaces.CloseTolerance` | real | 0.05 |
| `SplitClosedFaces.MaxTolerance` | real | 0.1 |
| `SplitClosedFaces.NbSplitPoints` | integer | 1 |
| `SplitClosedFaces.SegmentSurfaceMode` | bool | 1 |

### FixWireGaps

| Parameter | Type | GEOM |
|---|---|---|
| `FixWireGaps.Tolerance3d` | real | - |

### FixFaceSize

| Parameter | Type | GEOM |
|---|---|---|
| `FixFaceSize.Tolerance` | real | 0.05 |

### DropSmallSolids

| Parameter | Type | GEOM |
|---|---|---|
| `DropSmallSolids.FixMode` | integer | - |
| `DropSmallSolids.VolumeThreshold` | real | 1000 |
| `DropSmallSolids.WidthFactorThreshold` | real | 1 |
| `DropSmallSolids.MergeSolids` | bool | 1 |

### DropSmallEdges

| Parameter | Type | GEOM |
|---|---|---|
| `DropSmallEdges.Tolerance3d` | real | 0.05 |

### FixShape

| Parameter | Type | GEOM |
|---|---|---|
| `FixShape.Tolerance3d` | real | 1e-7 |
| `FixShape.MinTolerance3d` | real | 1e-7 |
| `FixShape.MaxTolerance3d` | real | 1 |
| `FixShape.FixFreeShellMode` | integer | -1 |
| `FixShape.FixFreeFaceMode` | integer | -1 |
| `FixShape.FixFreeWireMode` | integer | -1 |
| `FixShape.FixSameParameterMode` | integer | -1 |
| `FixShape.FixSolidMode` | integer | -1 |
| `FixShape.FixVertexPositionMode` | integer | - |
| `FixShape.FixVertexToleranceMode` | integer | - |
| `FixShape.FixShellMode` | integer | -1 |
| `FixShape.FixShellOrientationMode` | integer | - |
| `FixShape.CreateOpenSolidMode` | bool | 0 |
| `FixShape.FixFaceMode` | integer | -1 |
| `FixShape.FixFaceOrientationMode` | integer | - |
| `FixShape.FixWireMode` | integer | -1 |
| `FixShape.FixOrientationMode` | integer | -1 |
| `FixShape.FixAddNaturalBoundMode` | integer | - |
| `FixShape.FixMissingSeamMode` | integer | -1 |
| `FixShape.FixSmallAreaWireMode` | integer | -1 |
| `FixShape.RemoveSmallAreaFaceMode` | integer | - |
| `FixShape.FixIntersectingWiresMode` | integer | -1 |
| `FixShape.FixLoopWiresMode` | integer | -1 |
| `FixShape.FixSplitFaceMode` | integer | - |
| `FixShape.ModifyTopologyMode` | bool | 0 |
| `FixShape.ModifyGeometryMode` | bool | 1 |
| `FixShape.ClosedWireMode` | bool | 1 |
| `FixShape.PreferencePCurveMode` | bool | 1 |
| `FixShape.FixReorderMode` | integer | -1 |
| `FixShape.FixSmallMode` | integer | -1 |
| `FixShape.FixConnectedMode` | integer | -1 |
| `FixShape.FixEdgeCurvesMode` | integer | -1 |
| `FixShape.FixDegeneratedMode` | integer | -1 |
| `FixShape.FixLackingMode` | integer | -1 |
| `FixShape.FixSelfIntersectionMode` | integer | -1 |
| `FixShape.RemoveLoopMode` | integer | - |
| `FixShape.FixReversed2dMode` | integer | -1 |
| `FixShape.FixRemovePCurveMode` | integer | -1 |
| `FixShape.FixRemoveCurve3dMode` | integer | -1 |
| `FixShape.FixAddPCurveMode` | integer | -1 |
| `FixShape.FixAddCurve3dMode` | integer | -1 |
| `FixShape.FixShiftedMode` | integer | -1 |
| `FixShape.FixSeamMode` | integer | -1 |
| `FixShape.FixEdgeSameParameterMode` | integer | 0 |
| `FixShape.FixNotchedEdgesMode` | integer | - |
| `FixShape.FixTailMode` | integer | - |
| `FixShape.MaxTailAngle` | real | - |
| `FixShape.MaxTailWidth` | real | - |
| `FixShape.FixSelfIntersectingEdgeMode` | integer | -1 |
| `FixShape.FixIntersectingEdgesMode` | integer | -1 |
| `FixShape.FixNonAdjacentIntersectingEdgesMode` | integer | -1 |

### SplitClosedEdges

| Parameter | Type | GEOM |
|---|---|---|
| `SplitClosedEdges.NbSplitPoints` | integer | - |

### SplitCommonVertex

No parameter.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-10-21*
