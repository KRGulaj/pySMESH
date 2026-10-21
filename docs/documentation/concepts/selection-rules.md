# Selection Rules

A selection rule names entities by what they are, not by their ids: "the faces of this
body with an area below 2", "the edges where two faces meet at more than 5 degrees". A
`Rule` holds filters, and `Session.select(kind, rule)` returns the live ids of one kind that
pass every filter. The filters follow the filters of SALOME Shaper's `FiltersPlugin`. Each
filter's docstring gives its test and every difference from Shaper.

This page states the stored form of a rule as a contract. A stored rule that this version
writes is read back to an equal rule by every later version that reads the same format
number.

## Selecting

```python
from pysmesh import EdgeSize, EntityKind, FeatureEdges, Rule, Session

s = Session()
s.add_box(3.0, 7.0, 11.0)
short_sharp = s.select(EntityKind.EDGE, Rule((FeatureEdges(5.0), EdgeSize("lt", 5.0))))
```

- The filters combine with AND, in order. `reverse=True` negates one filter alone.
- A filter that does not apply to the kind is refused, naming the filter and the kind.
- A filter that names a dead id is refused, naming the id.
- An id that denotes several shapes (a split id) passes a filter when the filter holds for
  the whole it denotes. A size filter compares the total measure. Every other filter must
  hold for each of the id's shapes.
- `select` is a query. It issues no id and changes nothing.

## The filters

| Type | Fields | Kinds | Shaper filter |
|---|---|---|---|
| `BelongsTo` | `entities` | SOLID, FACE, EDGE, VERTEX | `BelongsTo` |
| `ContinuousFaces` | `faces`, `angle_deg` | FACE | `ContinuousFaces` |
| `EdgeSize` | `comparator`, `value`, `value_max` | EDGE | `EdgeSize` |
| `ExternalFaces` | (none) | FACE | `ExternalFaces` |
| `FaceSize` | `comparator`, `value`, `value_max` | FACE | `FaceSize` |
| `FeatureEdges` | `angle_deg` | EDGE | `FeatureEdges` |
| `HorizontalFace` | (none) | FACE | `HorizontalFaces` |
| `OnGeometry` | `entities` | FACE, EDGE | `OnGeometry` |
| `OnLine` | `edges` | EDGE, VERTEX | `OnLine` |
| `OnPlane` | `faces` | FACE, EDGE, VERTEX | `OnPlane` |
| `OnPlaneSide` | `face` | SOLID, FACE, EDGE, VERTEX | `OnPlaneSide` |
| `OppositeToEdge` | `edge` | EDGE | `OppositeToEdge` |
| `RelativeToSolid` | `solid`, `location` | FACE, EDGE, VERTEX | `RelativeToSolid` |
| `TopoConnectedFaces` | `entity`, `propagation` | FACE | `TopoConnectedFaces` |
| `VerticalFace` | (none) | FACE | `VerticalFaces` |
| `VolumeSize` | `comparator`, `value`, `value_max` | SOLID | `VolumeSize` |

Every filter also has the field `reverse`. Shaper's `OnShapeName` is not ported: a session
has no shape names.

`comparator` is one of `"lt"`, `"le"`, `"gt"`, `"ge"`, `"between"` and
`"strictly_between"` (Shaper's `inf`, `infEq`, `sup`, `supEq`, `isBetween` and
`isStrictlyBetween`). `location` is one of `"in"`, `"out"`, `"on"`, `"not_on"`,
`"not_out"` and `"not_in"`. Angles are in degrees, as in Shaper.

## Stored form, format 1

`rule_to_dict(rule)` returns a JSON-compatible dict, and `rule_from_dict(data)` reads it
back:

```json
{
  "format": 1,
  "filters": [
    {"type": "FeatureEdges", "angle_deg": 5.0, "reverse": false},
    {"type": "EdgeSize", "comparator": "lt", "value": 5.0, "value_max": 0.0,
     "reverse": false},
    {"type": "BelongsTo", "entities": [1], "reverse": true}
  ]
}
```

The contract:

1. The top level has exactly two keys: `"format"`, the integer 1, and `"filters"`, a list.
2. Each entry of `"filters"` is an object with `"type"`, the filter's class name from the
   table above, and one key per field of that filter, `"reverse"` included. The order of
   the entries is the order of the filters.
3. Values are JSON types:
   - `entities`, `faces` and `edges` are lists of integer entity ids, at least one;
   - `face`, `edge`, `solid` and `entity` are integer entity ids;
   - `angle_deg`, `value` and `value_max` are finite numbers;
   - `comparator` and `location` are strings from the lists above;
   - `reverse` and `propagation` are booleans.
4. `value_max` is present in every size filter. Only `"between"` and `"strictly_between"`
   read it, as in Shaper.
5. Entity ids are the ids of the session the rule was written for. A stored rule names
   entities only through those ids.

`rule_from_dict` refuses:

- a format other than 1;
- a missing or an extra key, at the top level or in a filter, and lists them;
- an unknown filter type;
- a value of the wrong type, a negative angle or size, an empty id list, and a
  `"between"` or `"strictly_between"` filter whose `value` is above `value_max`.

A later version that adds a filter type or a field, or changes the layout, writes a new
format number.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-10-21*
