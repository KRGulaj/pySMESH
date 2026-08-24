# Entity IDs and Ordinals

pySMESH keys every geometry result to an integer id. Two different id schemes exist, and
they answer two different questions. Mixing them up is silent and wrong: a value that is
a valid id in one scheme is often a valid, but different, id in the other. This page
explains both schemes, when each applies, and what happens to an id across an edit.

## The two id spaces

| Space | Produced by | Meaning |
|---|---|---|
| Positional ordinal | `load_brep`, `Shape.faces()` / `.edges()` / `.solids()` / `.vertices()`, and every stateless free function (`read_step_xde`, `read_iges`, `tessellate`, `unify_same_domain`, and so on) | The 1-based rank of a sub-shape in a per-kind `TopExp` traversal. Changes whenever the topology changes. |
| `EntityId` | `pysmesh.Session` | A session-issued identity. Monotonic. Never reused. |

Both are plain integers at the storage level, so nothing at the Python level stops a caller
from passing one where the other belongs, except one guard: `EntityId` is declared as a
distinct `typing.NewType` over `int`. Passing a raw ordinal into a `Session` method is
caught by `mypy --strict`. At run time, an id the session never issued raises
`PysmeshError` rather than silently denoting another entity.

## Positional ordinals

The stateless API has no memory between calls. Each function reads BREP bytes, runs one
OCCT algorithm, and writes bytes back. There is no session to hold identity, so a result is
keyed to a face, edge, solid or vertex by its **1-based rank** in the order OCCT's `TopExp`
explorer visits that shape.

```python
import pysmesh

shape = pysmesh.load_brep(brep_bytes)
face = shape.faces()[0]        # rank 1 in the FACE traversal
face.id                        # 1
```

This convention is pervasive. `Shape.face_distance`, `tessellate`'s `tri_face_id`,
`read_step_xde`'s `face_labels`, `unify_same_domain`'s `face_map`, and every quality control
that reports a sub-shape all key their results to this same 1-based ordinal.

**An ordinal is silently wrong when the topology changes.** Cut a hole in a box and the box
still has faces numbered 1 through 6, but they are not the same six faces as before: OCCT
re-traversed the shape and the ranks were reassigned. An ordinal held across an edit does
not raise. It resolves to whatever now occupies that rank, which is rarely what the caller
meant. This is the failure mode the `Session` id scheme exists to remove.

## `Session`'s `EntityId`

A `Session` owns one live shape and an `EntityId` registry carried across every operation
by that operation's OCCT history. `EntityId` values are monotonic and session-scoped: two
sessions in one process issue ids from independent counters, so an id from one session
means nothing to another.

```python
from pysmesh import Session
from pysmesh.session import EntityKind

s = Session()
s.add_box(3.0, 7.0, 11.0)
edges = s.entities(EntityKind.EDGE)   # (N,) int64 of live EntityId values
```

## Identity rules across an edit

Every operation on a `Session` follows the same five rules, stated on `HistoryDelta`:

- An entity modified to exactly **one** output keeps its id.
- An entity modified to **several** outputs keeps its id on all of them. Its name then
  resolves as `ResolutionStatus.AMBIGUOUS` rather than picking one.
- Several entities **merged** onto one output: all of their ids survive on it.
- An output with **no input correspondence** gets a new id.
- A **removed** entity's id is marked dead and is never reused.

A fillet is a concrete case of the first and fourth rules at once. Filleting an edge kills
that edge's id and adds a new face (the fillet surface) plus new edges bounding it:

```python
from pysmesh import Session
from pysmesh.session import EntityKind

s = Session()
s.add_box(3.0, 7.0, 11.0)
edge = int(s.entities(EntityKind.EDGE)[0])

s.fillet(edge_ids=[edge], radius=0.5)

s.is_alive(edge)   # False: the filleted edge id died
```

## Persistent names and resolution

`Session.name_of(entity_id)` mints a `Name`: a triple of `(op_index, role, ordinal)`
describing which operation issued the id, how (`NameRole.CONSTRUCTED` for a primitive or an
import, `NameRole.GENERATED` for anything derived from other entities), and its rank among
that operation's issued entities. A `Name` is provenance, not a geometric fingerprint.
pySMESH never matches entities by shape or position, because that is exactly the kind of
match that goes wrong under the edits persistent naming exists to survive.

`Session.resolve(name)` answers what a `Name` denotes **now**, without raising:

```python
face = int(s.entities(EntityKind.FACE)[0])
name = s.name_of(face)

resolution = s.resolve(name)
resolution.status   # RESOLVED, AMBIGUOUS, or LOST
resolution.ids       # the surviving EntityId values; empty when LOST
```

- `ResolutionStatus.RESOLVED`: the name denotes exactly one live entity.
- `ResolutionStatus.AMBIGUOUS`: the entity survives but now denotes more than one shape (it
  was split).
- `ResolutionStatus.LOST`: the entity is dead. This is a legitimate answer. A caller must
  handle it; the session never guesses a replacement.

## Several ways to be told an id is gone

pySMESH reports a dead or unknown id differently depending on which query is used, and the
difference is deliberate, not incidental.

- **Most direct id-based queries raise on a dead id, and on an unissued one.**
  `Session.entity_kind`, `Session.mass_properties`, `Session.shape_count` and
  `Session.name_of` all raise `PysmeshError` for an id the session never issued, and raise
  again for one that has since died.
- **`Session.is_alive` is built to answer a dead id without raising.** It raises only if the
  session never issued the id at all. For an id it did issue, alive or dead, it returns
  `True` or `False`. That is the query to reach for when "is this still here" is an expected
  question, not a bug to catch.
- **`Session.origin` answers for a dead id too**, unlike `name_of`: it explains what an id
  *was*, so it stays useful after the entity is gone. Like `is_alive`, it raises only for an
  id the session never issued in the first place.
- **`Session.resolve` never raises for a dead name.** It answers `ResolutionStatus.LOST`
  instead, for exactly the same reason `is_alive` does not raise: a caller asking "does this
  still exist" is asking a real question, and `LOST` is the honest answer to it.

## Why ids are never reused

`Session.restore` rewinds the live shape and the id registry to a retained state in O(1),
but it does **not** rewind the operation counter or the id counter. If it did, a later
operation on the restored branch would re-issue an id that an abandoned branch had already
used. A reference held from that abandoned branch would then resolve to a different entity:
the one failure the whole scheme exists to prevent.

Ids issued on an abandoned branch simply report dead forever. A stale `EntityId` always
resolves to *dead*, never to *something else*. That is the property a positional ordinal
cannot offer, and the reason `Session` exists on top of the stateless API.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
