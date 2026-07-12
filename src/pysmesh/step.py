"""STEP XDE import/export: read_step_xde and write_step_xde (Tier-2).

Public surface: :class:`EntityLabel`, :class:`StepImport`, :func:`read_step_xde`,
:func:`write_step_xde`. These wrap the low-level ``_core`` XDE queries (which return / accept
raw dicts) in pysmesh's frozen-dataclass + 1-based-id convention.

``read_step_xde`` imports a STEP file via OCCT's XDE stack (``STEPCAFControl_Reader`` +
``XCAFDoc_ShapeTool``/``XCAFDoc_ColorTool``), carrying the STEP product names, per-face
colours, and the file's length unit across the boundary — everything Gmsh's ``importShapes``
discards. The returned geometry is in the file's *native* length unit; ``length_unit`` is the
metres-per-unit factor to convert coordinates to SI. Names/colours are keyed to the returned
BREP's 1-based TopExp ordinals, so ``load_brep(result.brep)`` reproduces exactly those
:meth:`Shape.faces`/:meth:`Shape.solids` ids (or match by centroid via :meth:`Shape.match_faces`).

``write_step_xde`` is the XDE round-trip: tag a BREP with a product name and per-face names /
colours and emit STEP bytes via ``STEPCAFControl_Writer``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Union, cast

from ._core import read_step_xde as _read_step_xde
from ._core import write_step_xde as _write_step_xde

# A STEP source: raw bytes, or a filesystem path (str / os.PathLike).
StepSource = Union[bytes, str, "os.PathLike[str]"]

# An RGB colour in [0, 1] per channel, or None when the entity carries no colour.
Color = Union[tuple[float, float, float], None]


@dataclass(frozen=True)
class EntityLabel:
    """A name and/or colour attached to one imported entity (face or solid).

    Attributes:
        id: 1-based TopExp ordinal of the entity in the returned BREP — the same id
            :meth:`Shape.faces` / :meth:`Shape.solids` yields for ``load_brep(brep)``.
        name: The STEP product/entity name, or ``""`` when the entity is unnamed.
        color: ``(r, g, b)`` in ``[0, 1]``, or ``None`` when the entity has no colour.
    """

    id: int
    name: str
    color: Color


@dataclass(frozen=True)
class StepImport:
    """Result of :func:`read_step_xde`.

    Attributes:
        brep: The transferred geometry as BREP bytes, in the file's native length unit. Load it
            with :func:`load_brep` to obtain a :class:`Shape` whose ids match the labels below.
        length_unit: Metres per model unit (mm → 0.001, m → 1.0, inch → 0.0254). Multiply BREP
            coordinates by this to reach SI metres.
        face_labels: Labels for faces carrying a name or colour, keyed by face id.
        solid_labels: Labels for solids carrying a name or colour, keyed by solid id.
    """

    brep: bytes
    length_unit: float
    face_labels: tuple[EntityLabel, ...]
    solid_labels: tuple[EntityLabel, ...]


def _to_label(entry: Mapping[str, object], id_key: str) -> EntityLabel:
    """Convert one raw ``_core`` label dict into an :class:`EntityLabel`."""
    raw_color = entry["color"]
    color: Color = None if raw_color is None else cast("tuple[float, float, float]", raw_color)
    return EntityLabel(id=int(cast("int", entry[id_key])), name=str(entry["name"]), color=color)


def read_step_xde(data_or_path: StepSource) -> StepImport:
    """Import a STEP file via OCCT XDE, preserving names, colours, and the length unit.

    Args:
        data_or_path: The STEP source — raw bytes, or a filesystem path (``str`` / ``Path``).

    Returns:
        A :class:`StepImport` with the native-unit BREP, the metres-per-unit ``length_unit``,
        and the face/solid labels keyed to the BREP's 1-based TopExp ordinals.

    Raises:
        PysmeshError: On a malformed / non-STEP input, a transfer that yields no shape, or a
            file with no free shapes.
    """
    src: StepSource
    if isinstance(data_or_path, (bytes, bytearray)):
        src = bytes(data_or_path)
    else:
        src = os.fspath(data_or_path)
    raw = _read_step_xde(src)
    return StepImport(
        brep=cast("bytes", raw["brep"]),
        length_unit=float(cast("float", raw["length_unit"])),
        face_labels=tuple(
            _to_label(e, "face_id") for e in cast("list[Mapping[str, object]]", raw["face_labels"])
        ),
        solid_labels=tuple(
            _to_label(e, "solid_id")
            for e in cast("list[Mapping[str, object]]", raw["solid_labels"])
        ),
    )


def write_step_xde(
    brep: bytes,
    *,
    name: str = "",
    face_names: Mapping[int, str] | None = None,
    face_colors: Mapping[int, tuple[float, float, float]] | None = None,
) -> bytes:
    """Export a BREP to STEP bytes via OCCT XDE, tagging the product and its faces.

    Args:
        brep: The shape to export, as BREP bytes.
        name: Product name for the whole shape (omitted when empty).
        face_names: Optional mapping of 1-based face id → name.
        face_colors: Optional mapping of 1-based face id → ``(r, g, b)`` in ``[0, 1]``.

    Returns:
        The STEP file content as bytes.

    Raises:
        PysmeshError: On a malformed BREP, an out-of-range face id, or a STEP write failure.
    """
    return _write_step_xde(brep, dict(face_names or {}), dict(face_colors or {}), name)
