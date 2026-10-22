# Sketch arrays

`pysmesh.sketch.solve` solves a 2-D sketch: points, lines, circles, arcs, ellipses, elliptic
arcs and B-splines under geometric and dimensional constraints. The solver is FreeCAD's
PlaneGCS, linked into the native core. It is a free function with no state, and it touches no
`Session`.

A sketch is a set of flat NumPy arrays. A row's index is its id: an entity's id is its row of
`entities`, and a constraint's id is its row of `constraints`. Every row that the solution
names maps straight back to the caller's own lists.

This page is the contract of the arrays. A kind code never changes its value once released.

## The arrays

| Array | dtype, shape | Holds |
|---|---|---|
| `params` | float64, (N,) | Every scalar of the sketch: point coordinates, radii, angles, B-spline weights and knots. It is also the initial guess. Every value is finite. |
| `entities` | int64, (E, 8) | One row per entity. Column 0 is the kind, the other columns are parameter indices or entity rows. Unused columns hold -1. |
| `constraints` | int64, (C, 5) | One row per constraint. Column 0 is the kind, the other columns are entity rows (an internal coincidence also holds a role and an index). Unused columns hold -1. |
| `values` | float64, (C,) | The dimension of each dimensional row (`LENGTH`, `ANGLE`, `RADIUS`, `DISTANCE`); NaN in the other rows. |
| `fixed` | bool, (N,), optional | The parameters the solver does not move. |
| `spline_offsets` | int64, (S+1,), optional | The B-spline slots: slot `k` is `spline_indices[spline_offsets[k]:spline_offsets[k+1]]`. |
| `spline_indices` | int64, (L,), optional | The contents of the slots. |

`ENTITY_COLUMNS` (8) and `CONSTRAINT_COLUMNS` (5) are the two table widths.

## Units and signs

- Lengths are in model units. Angles are in radians.
- An arc runs counter-clockwise from its start angle to its end angle. To draw a clockwise
  arc, swap its start and its end.
- An `ANGLE` dimension is the angle from the first line's direction (start to end) to the
  second line's direction, counter-clockwise positive.
- An ellipse's focus lies on its positive major axis. The positive minor axis end is a quarter
  turn counter-clockwise from the positive major axis end.
- An elliptic arc's angles are the ellipse's own parameter `u`, measured from the major axis:
  the point at `u` is `c + a cos(u) e1 + b sin(u) e2`.

## Entities

`SketchEntityKind` gives column 0. The other columns, in order:

| Code | Kind | Columns 1 to 7 |
|---|---|---|
| 1 | `POINT` | x, y (parameters) |
| 2 | `LINE` | start, end (points) |
| 3 | `CIRCLE` | center (point), radius (parameter) |
| 4 | `ARC` | center, start, end (points), radius, start angle, end angle (parameters) |
| 5 | `ELLIPSE` | center, focus (points), minor radius (parameter) |
| 6 | `ELLIPTIC_ARC` | center, focus, start, end (points), minor radius, start angle, end angle (parameters) |
| 7 | `BSPLINE` | slot, degree (1 to 25), periodic (0 or 1), pole count |

A "point" column holds the row of a `POINT` entity. Two entities that name one point row share
that point. Two point rows on the same two parameters are also one point.

An arc and an elliptic arc carry rules that tie their end points to their angles. The arc
takes Shaper's four (`PlaneGCSSolver_Storage`, `createArcConstraints`): the center-to-start
and center-to-end distances equal the radius (`ConstraintP2PDistance`), and the directions of
the start and the end from the center equal the two angles (`ConstraintP2PAngle`). The
elliptic arc takes `System::addConstraintArcOfEllipseRules`, as Shaper's
`createEllipticArcConstraints` does. These rules have tag 0, so the diagnosis never names them.

### B-spline slots

A B-spline row names its slot in column 1. A slot of a B-spline with `n` poles and `m` knots
holds `2n + 2m` values, in this order:

1. `n` pole rows (points);
2. `n` weight parameters;
3. `m` knot parameters, increasing;
4. `m` knot multiplicities (integers).

Every slot belongs to exactly one B-spline row. OCCT checks the definition
(`Geom2d_BSplineCurve`), so the counts must agree with the degree, the multiplicities and the
periodicity. The start and the end of a B-spline are its first and its last pole, as in
Shaper's `createBSpline`. Knots never move. The weights move unless `fixed` holds them; the
builder fixes them unless a `BSpline` is `rational`.

Example: a cubic Bezier curve on the points in rows 0 to 3, whose parameters 8 to 11 hold the
weights and 12 to 13 the knots 0 and 1:

```python
entities[4] = [7, 0, 3, 0, 4, -1, -1, -1]            # BSPLINE, slot 0, degree 3, open, 4 poles
spline_offsets = [0, 12]
spline_indices = [0, 1, 2, 3,  8, 9, 10, 11,  12, 13,  4, 4]
```

## Constraints

`SketchConstraintKind` gives column 0. Each row becomes PlaneGCS equations as SALOME Shaper's
PlaneGCS adapter maps it (`src/SketchSolver`, Shaper commit `59c5672c2`, `V9_15_0-35`). The
table lists the `GCS::System` calls.

| Code | Kind | Columns | `GCS::System` calls |
|---|---|---|---|
| 1 | `COINCIDENT` | two points, or a point and a curve, either order | two points: `addConstraintP2PCoincident`. A line: `addConstraintPointOnLine`. A circle: `addConstraintPointOnCircle`. An arc: `addConstraintPointOnArc` (its full circle). An ellipse or an elliptic arc: `addConstraintPointOnEllipse` (the full ellipse). A B-spline: `addConstraintPointOnBSpline`, with the point's parameter on the curve as an extra unknown. |
| 2 | `TANGENT` | two curves; optional point | See [Tangent and perpendicular](#tangent-and-perpendicular). |
| 3 | `PARALLEL` | two lines | `addConstraintParallel` |
| 4 | `PERPENDICULAR` | two curves; optional point | See [Tangent and perpendicular](#tangent-and-perpendicular). |
| 5 | `HORIZONTAL` | a line | `addConstraintHorizontal` |
| 6 | `VERTICAL` | a line | `addConstraintVertical` |
| 7 | `EQUAL` | two lines, two circles or arcs, or two ellipses or elliptic arcs | lines: `addConstraintEqualLength`. Circles and arcs: `addConstraintEqual` of the radii. Ellipses: `addConstraintEqualRadii`. |
| 8 | `COLLINEAR` | two lines | `addConstraintPointOnLine` of each point of the second line that is not a point of the first. |
| 9 | `MIDPOINT` | a point, then a line, an arc or an elliptic arc | A line: `addConstraintPointOnLine` and `addConstraintPointOnPerpBisector`. An arc: `addConstraintCurveValue` at an extra unknown angle `u`, with `addConstraintDifference` twice, so that `u - start = end - u`. The arc's end angle is above its start angle in the initial guess. |
| 10 | `RIGID` | an entity | `addConstraintEqual` of each of its parameters to a copy of its initial value (Shaper's `SketchSolver_ConstraintFixed`). Parameters that `fixed` holds are left out. |
| 11 | `LENGTH` | a line; a length above 0 | `addConstraintP2PDistance` of its two points |
| 12 | `ANGLE` | two lines; an angle | `addConstraintL2LAngle` |
| 13 | `RADIUS` | a circle or an arc; a radius above 0 | `addConstraintCircleRadius`, `addConstraintArcRadius` |
| 14 | `INTERNAL_COINCIDENT` | a curve, a point, a role, an index | See [Internal coincidence](#internal-coincidence). |
| 15 | `DISTANCE` | two points, or a point and a line, either order; a distance of 0 or above | `addConstraintP2PDistance`, `addConstraintP2LDistance` (unsigned). A distance of 0 is a coincidence, as in Shaper's `SketchSolver_ConstraintDistance`. |
| 16 | `SYMMETRIC` | two points, then a line | `addConstraintP2PSymmetric` |

A row that names one entity twice is refused. A row whose operand kinds do not fit its kind is
refused, naming the row and the kinds.

### Tangent and perpendicular

Column 3 holds a point, or -1. The point is where the curves meet. The caller puts it on both
curves: it is an end point they share, or a point that `COINCIDENT` rows put on both. Shaper
creates such a point itself; here it is a row of the sketch, so its id is visible.

| Pair | Point | `GCS::System` call |
|---|---|---|
| tangent: line and circle | ignored | `addConstraintTangent(Line, Circle)` |
| tangent: line and arc | none | `addConstraintTangent(Line, Arc)` |
| tangent: two circles or arcs | none | `addConstraintTangentCircumf`, inner when the centers are nearer than either radius in the initial guess |
| tangent: line and ellipse or elliptic arc | none | `addConstraintTangent(Line, Ellipse)` |
| tangent: any other pair, or a point given | required | `addConstraintAngleViaPoint`, angle 0 or pi, chosen by the angle the curves make at the point in the initial guess |
| perpendicular: two lines | ignored | `addConstraintPerpendicular` |
| perpendicular: line and circle or arc | ignored | `addConstraintPointOnLine` of the center |
| perpendicular: a pair with an ellipse, an elliptic arc or a B-spline | required | `addConstraintAngleViaPoint`, angle pi/2 or -pi/2, chosen by the initial guess |

A B-spline takes a tangent or a perpendicular only at its first or its last pole, and only when
it is clamped there (not periodic, end multiplicities of the degree + 1). The point is that
pole. The B-spline is then its end segment, from the first to the second pole or from the last
but one to the last, as Shaper substitutes it. PlaneGCS gives a B-spline's direction by point
only there.

### Internal coincidence

Column 1 is the curve, column 2 the point, column 3 a `SketchInternalRole`, column 4 an index.

| Code | Role | Curve | Index | `GCS::System` calls |
|---|---|---|---|---|
| 1 | `MAJOR_POSITIVE` | ellipse, elliptic arc | -1 | `addConstraintInternalAlignmentPoint2Ellipse`, x and y |
| 2 | `MAJOR_NEGATIVE` | ellipse, elliptic arc | -1 | the same |
| 3 | `MINOR_POSITIVE` | ellipse, elliptic arc | -1 | the same |
| 4 | `MINOR_NEGATIVE` | ellipse, elliptic arc | -1 | the same |
| 5 | `FOCUS2` | ellipse, elliptic arc | -1 | `addConstraintInternalAlignmentEllipseFocus2` |
| 6 | `POLE` | B-spline | the pole | `addConstraintEqual` of x and y, marked as an internal alignment |
| 7 | `KNOT` | B-spline | the knot | `addConstraintInternalAlignmentKnotPoint` |

PlaneGCS never names an internal alignment as conflicting.

## What moves

The unknowns are the parameters that an entity moves and that `fixed` does not hold: point
coordinates, radii, arc and elliptic-arc angles, minor radii, and B-spline weights. Some rows
add their own unknowns: the parameter of a point on a B-spline, and the angle and half sweep of
an arc's midpoint. A parameter that no entity uses keeps its value, and so does every knot.

## How it solves

The solve follows Shaper's `PlaneGCSSolver_Solver::solve`:

1. Diagnose and solve with DogLeg.
2. If DogLeg fails, diagnose and solve with Levenberg-Marquardt.
3. If that fails, diagnose and solve with BFGS.

The settings are PlaneGCS's own defaults, with two from the call. `max_iterations` (100) is
the iteration limit of each run. `tolerance` (1e-10) is DogLeg's largest residual,
Levenberg-Marquardt's residual norm and BFGS's step at which a run stops as solved. The
geometry it returns is within that order of the exact one.

The result depends on `params` as the initial guess. An under-constrained sketch keeps the
solution nearest the guess that the solver reaches, and a choice between mirror solutions
follows the guess. A caller that replays a sketch passes the same initial guess.

Identical arrays give bit-identical results, in fresh processes and after any number of other
solves. PlaneGCS orders its parameters by memory address; every scalar lives in one buffer,
in index order, so that order is fixed. Each constraint row has its own tag (row + 1), and
PlaneGCS breaks its diagnosis ties by tag. The solve runs on the calling thread, with the GIL
released.

## The solution

`SketchSolution` holds:

- `params`: a new array. It holds the solved parameters when the status is `SOLVED` or
  `UNDER_CONSTRAINED`, and a copy of the initial guess otherwise.
- `status`: a `SketchStatus`.
- `dof`: PlaneGCS's count of the degrees of freedom left: the unknowns less the rank of the
  constraint Jacobian. When the sketch is over-constrained with a Jacobian of full rank, it is
  the unknowns less the non-redundant equations, which is negative.
- `conflicting`, `redundant`, `partially_redundant`: sorted constraint rows from PlaneGCS's
  diagnosis.

| Status | When |
|---|---|
| `SOLVED` | The solve reached the tolerance and `dof` is 0. |
| `UNDER_CONSTRAINED` | The solve reached the tolerance and `dof` is above 0. |
| `OVER_CONSTRAINED` | The diagnosis names conflicting rows. Nothing is applied. |
| `FAILED` | No conflict, and no run reached the tolerance (PlaneGCS's `Failed`, or BFGS's `Converged`: the residual is not zero). Nothing is applied. |

Redundant rows do not change the status, as in Shaper: the other rows imply them, and the
sketch solves. PlaneGCS picks which row of a dependent group it names: the most frequent one,
then the one with fewer equations, then the latest row. A row is partially redundant when some
of its PlaneGCS equations are redundant and some are not.

A sketch without unknowns is not diagnosed: each row holds or conflicts as it stands, and a
row whose residual exceeds the tolerance is conflicting.

## Differences from Shaper

- A BFGS result that does not reach zero residual (`Converged`) is `FAILED`. Shaper applies
  it.
- A point on a B-spline uses PlaneGCS's own `addConstraintPointOnBSpline`. Shaper's two
  `ConstraintCurveValue` equations need Shaper's own B-spline class, which pySMESH does not
  compile.
- A tangent or a perpendicular with a B-spline is accepted only at an end pole (see above).
- Equal lines use `addConstraintEqualLength`, one equation. Shaper uses two distances to an
  extra common length; the geometry is the same.

## Building the arrays

`SketchBuilder` writes the arrays from frozen entity descriptions (`Point`, `Line`, `Circle`,
`Arc`, `Ellipse`, `EllipticArc`, `BSpline`). Each method returns the row it appends. `solve`
does not need the builder.

```python
import pysmesh.sketch as sk

b = sk.SketchBuilder()
p = [b.add(sk.Point(x, y)) for x, y in ((0, 0), (2.9, 0.1), (3.1, 2), (0, 1.9))]
lines = [b.add(sk.Line(p[k], p[(k + 1) % 4])) for k in range(4)]
b.fix(p[0])
b.horizontal(lines[0])
b.horizontal(lines[2])
b.vertical(lines[1])
b.vertical(lines[3])
b.length(lines[0], 3.0)
b.length(lines[1], 2.0)
arrays = b.build()
solution = arrays.solve()
solution.status                        # SketchStatus.SOLVED
arrays.point(p[2], solution.params)    # (3.0, 2.0)
```

A solved sketch becomes CAD through the session's `add_line`, `add_arc`, `add_bspline`,
`make_wire`, `make_face`, `fillet_1d` and `offset_wire`.

## Refusals

Every refusal is a `PysmeshError` whose message starts with `sketch.solve:`. It names the
entity or constraint row (or the array) at fault and the offending value. The checks are:

- the dtypes and shapes of the arrays, and the lengths of `values` and `fixed`;
- every index and entity row in range, and every operand of the kind its row takes;
- a finite initial guess, a line of non-zero length, radii above 0, a focus off its center;
- the B-spline slots, and OCCT's check of each definition;
- a finite dimension on a dimensional row, NaN on the others, and the dimension's range.
