# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-17

"""IGES import/export: read_iges and write_iges (Tier-2).

Public surface: :data:`IGES_UNITS`, :class:`IgesImport`, :func:`read_iges`,
:func:`write_iges`. These wrap the low-level ``_core`` IGES entry points (which return /
accept raw values) in pysmesh's frozen-dataclass convention.

Both directions treat the length unit as data, never as ambient state:

``read_iges`` returns the geometry in the file's *native* unit and reports
``length_unit``, the metres-per-unit factor, exactly as :func:`read_step_xde` does. A reader
that normalises silently to millimetres is where the 1000x "millimetre part imported as a
metre part" defect comes from; this one never rescales behind the caller's back.

``write_iges`` takes the unit of the BREP coordinates as an *argument*. The header declares
that unit and the coordinates are written unchanged, so the file cannot come out labelled in
one unit while holding numbers in another.

OCCT has no IGES stream reader (``IGESSelect_WorkLibrary`` does not override
``IFSelect_WorkLibrary::ReadStream``), so :func:`read_iges` takes a path, not bytes.
:func:`write_iges` returns bytes, matching :func:`write_step_xde`.

Importing an IGES file makes OCCT print one line to stdout ("Total number of loaded
entities N."). It is an unconditional info-level message inside ``IGESFile_Read``; OCCT
exposes no switch for it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Mapping, Union, cast

from ._core import read_iges as _read_iges
from ._core import write_iges as _write_iges

# An IGES source: a filesystem path (str / os.PathLike). Bytes are not accepted — see the
# module docstring.
IgesPath = Union[str, "os.PathLike[str]"]

# The ten length units an IGES global section can declare, as metres per unit. The names are
# OCCT's (``IGESData_BasicEditor::UnitFlagName``) and are what :attr:`IgesImport.unit_name`
# reports and :func:`write_iges` accepts. "IN" is also accepted for inches on input.
IGES_UNITS: Final[Mapping[str, float]] = {
    "UIN": 2.54e-8,
    "UM": 1.0e-6,
    "MIL": 2.54e-5,
    "MM": 1.0e-3,
    "CM": 1.0e-2,
    "INCH": 0.0254,
    "FT": 0.3048,
    "M": 1.0,
    "KM": 1.0e3,
    "MI": 1609.344,
}


@dataclass(frozen=True)
class IgesImport:
    """Result of :func:`read_iges`.

    Attributes:
        brep: The transferred geometry as BREP bytes, in the file's native length unit. Load
            it with :func:`load_brep` to obtain a :class:`Shape`.
        length_unit: Metres per model unit (mm → 0.001, m → 1.0, inch → 0.0254). Multiply
            BREP coordinates by this to reach SI metres.
        unit_name: The IGES unit name the file's global section declares, one of the keys of
            :data:`IGES_UNITS`. Pass it back to :func:`write_iges` to re-export unchanged.
    """

    brep: bytes
    length_unit: float
    unit_name: str


def read_iges(path: IgesPath) -> IgesImport:
    """Import an IGES file via OCCT's IGESControl_Reader, preserving the declared unit.

    Args:
        path: Filesystem path to the ``.igs`` / ``.iges`` file.

    Returns:
        An :class:`IgesImport` with the native-unit BREP, the metres-per-unit
        ``length_unit``, and the ``unit_name`` the header declares.

    Raises:
        PysmeshError: On a missing or malformed file, a file whose global section declares a
            unit IGES has no value for, or a file holding no transferable geometry.
    """
    raw = _read_iges(os.fspath(path))
    return IgesImport(
        brep=cast("bytes", raw["brep"]),
        length_unit=float(cast("float", raw["length_unit"])),
        unit_name=str(raw["unit_name"]),
    )


def write_iges(brep: bytes, *, unit: str, brep_mode: bool = True) -> bytes:
    """Export a BREP to IGES bytes, declaring the unit the coordinates are already in.

    The coordinates are written verbatim: ``unit`` labels them, it does not rescale them.
    Pass the unit the BREP is actually in — ``"M"`` for an SI-metre model, ``imported
    .unit_name`` to re-export what :func:`read_iges` returned.

    Args:
        brep: The shape to export, as BREP bytes.
        unit: The IGES unit name of the BREP's coordinates, case-insensitive. One of the keys
            of :data:`IGES_UNITS` (``"IN"`` is also accepted for inches).
        brep_mode: ``True`` writes IGES 5.3 BRep entities, which carry solids and shells.
            ``False`` writes IGES 5.1 face entities, the only mode that carries standalone
            wires, edges and vertices.

    Returns:
        The IGES file content as bytes.

    Raises:
        PysmeshError: On a malformed BREP, an unknown unit name, a shape ``brep_mode`` cannot
            represent, or an IGES write failure.
    """
    return _write_iges(brep, unit, brep_mode)
