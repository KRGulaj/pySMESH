# Units

STEP and IGES files each declare a length unit in their header. On the **read** side,
pySMESH treats that unit as data the caller reads explicitly, never as ambient state the
library resolves on its own. This page states that contract precisely, because the
alternative, a reader that silently normalises, is where the classic "millimetre part
imported as a metre part" defect comes from. Both formats carry the same contract on the
write side too: the caller declares the unit and the coordinates are never rescaled.

## The read contract

- **Coordinates stay in the file's native unit.** `read_step_xde` and `read_iges` never
  rescale a coordinate. A file whose header says millimetres comes back with millimetre
  numbers, not metres.
- **`length_unit` comes back with the geometry.** Both readers report `length_unit`: the
  number of metres per model unit. Multiply a BREP coordinate by `length_unit` to reach SI
  metres.
- **Neither reader touches OCCT's global `Interface_Static` unit setting.** That global is a
  process-wide default many OCCT-based tools read implicitly. pySMESH never sets or reads
  it for a read, so a file cannot arrive silently normalised to some other tool's default.

```python
import pysmesh

imported = pysmesh.read_step_xde("blade.step")
imported.length_unit   # 0.001 for a millimetre file, 1.0 for a metre file

# Native-unit coordinates times length_unit is the physical size in SI metres.
shape = pysmesh.load_brep(imported.brep)
physical_bbox = shape.faces()[0].bbox * imported.length_unit
```

Internally, OCCT's STEP transfer normalises every shape to millimetres regardless of the
file's declared unit. `read_step_xde` reverses that scale before returning the shape, so the
coordinates a caller sees are numerically the file's own native-unit values, identity when
the file already declared millimetres. This reversal is why the contract above holds; it is
not that OCCT leaves the numbers untouched end to end.

## The write contract

Both writers take the unit of the caller's coordinates as a required `unit` argument and
declare it in the header without rescaling:

```python
pysmesh.write_iges(brep, unit="M")            # declares the header unit; coordinates unchanged
pysmesh.write_step_xde(brep, unit="M")        # same contract, same vocabulary
```

`unit` names the unit the coordinates **are already in**. It labels them; it does not
convert them. Passing `"M"` for a model whose numbers are millimetres produces a file that
claims to be 1000 times larger, so pass what the geometry actually is.

Both accept the same ten names, the keys of `pysmesh.IGES_UNITS`: `UIN`, `UM`, `MIL`, `MM`,
`CM`, `INCH`, `FT`, `M`, `KM`, `MI`. Matching is case-insensitive. An unrecognised name
raises `PysmeshError` listing the accepted set, rather than falling back to a default.

Because a reader reports `unit_name` and a writer accepts it, a re-export is a round trip
rather than a conversion:

```python
imported = pysmesh.read_step_xde("part.step")
pysmesh.write_step_xde(imported.brep, unit=imported.unit_name)   # unit-exact
```

### A note on precision at extreme unit ratios

OCCT normalises STEP geometry to millimetres internally. A round trip through a unit far
from a millimetre therefore divides and re-multiplies by a large factor, and the low digits
do not survive. `MM`, `CM`, `M`, `INCH`, `FT`, `KM` and `MI` round-trip exactly. `UM`,
`MIL` and `UIN` keep their declared unit exactly but drift in the last digits, by well under
a part in 100. Those three are surface-finish units, not geometry units, so this rarely
matters in practice. It is stated here because it is measurable.

### This changed in 4.0.0

Before 4.0.0, `write_step_xde` took no `unit` argument. Every file it produced inherited
OCCT's process-wide default of millimetres, whatever the coordinates were. A 2 metre part
read in and written back out came back as a 2 millimetre part, with nothing in the API to
warn the caller or to prevent it. `unit` is required now precisely so that failure mode
cannot recur.

## IGES specifics

`read_iges` reports the same pair, `length_unit` and the file's own `unit_name`:

```python
igs = pysmesh.read_iges("housing.igs")
igs.length_unit    # 0.001 for an MM file, 0.0254 for an INCH file
igs.unit_name       # "MM", "INCH", "M", ...: one of the keys of IGES_UNITS

pysmesh.write_iges(igs.brep, unit=igs.unit_name)   # re-export, unit-exact
```

`IGES_UNITS` is the full table of the ten length units an IGES global section can declare,
each given as metres per unit:

| Unit name | Metres per unit |
|---|---|
| `UIN` | 2.54e-8 |
| `UM` | 1.0e-6 |
| `MIL` | 2.54e-5 |
| `MM` | 1.0e-3 |
| `CM` | 1.0e-2 |
| `INCH` | 0.0254 |
| `FT` | 0.3048 |
| `M` | 1.0 |
| `KM` | 1.0e3 |
| `MI` | 1609.344 |

`write_iges` also accepts `"IN"` as an alias for inches on input, though `unit_name` on a
read result is always one of the ten names above.

## Why `read_iges` takes a path, not bytes

`read_step_xde` accepts either raw bytes or a path. `read_iges` accepts only a path:

```python
import pathlib

import pysmesh

pysmesh.read_iges("housing.igs")                       # a str
pysmesh.read_iges(pathlib.Path("housing.igs"))          # or an os.PathLike
```

OCCT ships no IGES stream reader: `IGESSelect_WorkLibrary` does not override
`IFSelect_WorkLibrary::ReadStream`. There is no in-memory entry point to wrap, so the reader
takes a filesystem path.

Reading an IGES file also makes OCCT print one line to standard output,
`Total number of loaded entities N.`. That is an unconditional info-level message inside
`IGESFile_Read`, and OCCT gives no switch to silence it. `write_iges` has no such side
effect and returns bytes, matching `write_step_xde`.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
