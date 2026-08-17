# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-17

"""Tier-2 IGES tests: read_iges and write_iges.

Fixtures are IGES files from ``tests/fixtures/generate_fixtures.cpp``, written by STOCK OCCT
(plain ``IGESControl_Writer(unit, 1)``), never by pySMESH's own writer:
  box_mm.igs    a cube of native coordinate extent 2.0, header unit MM.
  box_m.igs     the same native extent 2.0, header unit M.
  box_inch.igs  the same native extent 2.0, header unit INCH.

Reference spec (IGES 5.3 global section, parameters 14/15; OCCT's unit table is
``IGESData_BasicEditor::UnitFlagValue``, which gives each unit's size in millimetres):
  read_iges returns geometry in the file's NATIVE unit and reports length_unit as metres per
  model unit (MM -> 0.001, M -> 1.0, INCH -> 0.0254). The three fixtures share one native
  extent, so length_unit is the only thing that distinguishes them — a reader that normalises
  to millimetres reports 0.001 for all three and is wrong on two.

  write_iges declares the unit it is given and writes the coordinates unchanged. The header
  assertions parse the IGES global section directly (fixed 80-column records, section letter
  in column 73) rather than trusting read_iges, so the writer is not tested against its own
  inverse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pysmesh
from pysmesh import (
    IGES_UNITS,
    IgesImport,
    PysmeshError,
    load_brep,
    read_iges,
    write_iges,
)

BOX_EDGE = 2.0

# Every fixture holds this native extent; only the declared unit differs.
UNIT_CASES = [
    ("box_mm.igs", "MM", 1.0e-3),
    ("box_m.igs", "M", 1.0),
    ("box_inch.igs", "INCH", 0.0254),
]


def _extent(brep: bytes) -> np.ndarray:
    """Axis-aligned extent of a BREP's faces, in the BREP's own coordinates."""
    bboxes = np.array([f.bbox for f in load_brep(brep).faces()], dtype=np.float64)
    return bboxes[:, 3:].max(axis=0) - bboxes[:, :3].min(axis=0)


# 0-based positions of the global-section parameters these tests read. IGES numbers them from
# 1: P14 unit flag, P15 unit name, P20 largest coordinate value in the file's units.
UNIT_FLAG = 13
UNIT_NAME = 14
MAX_COORD = 19


def _global_section(data: bytes) -> list[str]:
    """Comma-separated parameters of an IGES global section, read straight from the bytes.

    IGES is a fixed 80-column record format: column 73 carries the section letter and columns
    74-80 the sequence number. The global section's records are the ones marked ``G``.

    The split is naive on purpose — it would break on a Hollerith string containing a comma,
    which none of the strings OCCT writes here does.
    """
    text = data.decode("ascii", "replace")
    body = "".join(
        line[:72] for line in text.splitlines() if len(line) > 72 and line[72] == "G"
    )
    return body.split(",")


# ---------------------------------------------------------------------------
# read_iges — structure & geometry
# ---------------------------------------------------------------------------


def test_read_iges_returns_iges_import(box_mm_iges_path: str) -> None:
    """The wrapper returns an IgesImport dataclass."""
    result = read_iges(box_mm_iges_path)
    assert isinstance(result, IgesImport)


def test_read_iges_brep_loads_to_cube(box_mm_iges_path: str) -> None:
    """The returned BREP is a single 6-face, 1-solid cube."""
    result = read_iges(box_mm_iges_path)
    shape = load_brep(result.brep)
    assert len(shape.faces()) == 6
    assert len(shape.solids()) == 1


def test_read_iges_accepts_path_object(fixtures_dir: Path) -> None:
    """A pathlib.Path is accepted as well as a str."""
    result = read_iges(fixtures_dir / "box_mm.igs")
    assert len(load_brep(result.brep).faces()) == 6


# ---------------------------------------------------------------------------
# read_iges — the declared length unit (the 1000x defect)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "unit_name", "metres"), UNIT_CASES)
def test_read_iges_reports_declared_unit(
    fixtures_dir: Path, name: str, unit_name: str, metres: float
) -> None:
    """Each fixture reports the unit its header declares, as a name and as metres per unit."""
    result = read_iges(fixtures_dir / name)

    assert result.unit_name == unit_name
    assert result.length_unit == pytest.approx(metres)


@pytest.mark.parametrize("name", [case[0] for case in UNIT_CASES])
def test_read_iges_geometry_stays_in_native_unit(fixtures_dir: Path, name: str) -> None:
    """No silent rescale: every fixture comes back at its native extent, whatever the unit."""
    result = read_iges(fixtures_dir / name)

    extent = _extent(result.brep)

    assert extent == pytest.approx([BOX_EDGE, BOX_EDGE, BOX_EDGE], abs=1e-5)


@pytest.mark.parametrize(
    ("name", "metres"), [(case[0], case[2]) for case in UNIT_CASES]
)
def test_read_iges_physical_size_via_length_unit(
    fixtures_dir: Path, name: str, metres: float
) -> None:
    """Native extent * length_unit recovers the true physical size in metres."""
    result = read_iges(fixtures_dir / name)

    physical = _extent(result.brep)[0] * result.length_unit

    assert physical == pytest.approx(BOX_EDGE * metres, rel=1e-6)


def test_read_iges_units_differ_between_fixtures(
    box_mm_iges_path: str, box_m_iges_path: str, box_inch_iges_path: str
) -> None:
    """The three fixtures are told apart only by length_unit — a mm-normalising reader is not."""
    units = {
        read_iges(p).length_unit
        for p in (box_mm_iges_path, box_m_iges_path, box_inch_iges_path)
    }
    assert len(units) == 3


def test_read_iges_unit_name_is_a_known_iges_unit(box_inch_iges_path: str) -> None:
    """unit_name is a key of IGES_UNITS and agrees with the reported length_unit."""
    result = read_iges(box_inch_iges_path)

    assert result.unit_name in IGES_UNITS
    assert IGES_UNITS[result.unit_name] == pytest.approx(result.length_unit)


# ---------------------------------------------------------------------------
# read_iges — validation
# ---------------------------------------------------------------------------


def test_read_iges_missing_file_raises(tmp_path: Path) -> None:
    """A path that does not exist is rejected."""
    with pytest.raises(PysmeshError):
        read_iges(tmp_path / "absent.igs")


def test_read_iges_malformed_raises(tmp_path: Path) -> None:
    """A file that is not IGES is rejected."""
    bad = tmp_path / "bad.igs"
    bad.write_bytes(b"not an iges file")

    with pytest.raises(PysmeshError):
        read_iges(bad)


def test_read_iges_undeclarable_unit_raises(box_mm_iges_path: str, tmp_path: Path) -> None:
    """A unit the format has no value for is refused, not silently taken as millimetres.

    IGES unit flag 3 means "the unit is the name in parameter 15". OCCT resolves that to a
    real flag when the name is one it knows, so a flag that survives as 3 names a unit nobody
    can size — and ``IGESData_BasicEditor::UnitFlagValue(3)`` returns 1.0, i.e. millimetres.
    Trusting that value is the mislabelling this binding exists to prevent.

    The tamper is length-preserving (``,2,2HMM,`` -> ``,3,2HZZ,``) so the file keeps its
    fixed 80-column record packing and stays parseable.
    """
    original = Path(box_mm_iges_path).read_bytes()
    assert original.count(b",2,2HMM,") == 1
    tampered = tmp_path / "unknown_unit.igs"
    tampered.write_bytes(original.replace(b",2,2HMM,", b",3,2HZZ,"))

    with pytest.raises(PysmeshError, match="unit"):
        read_iges(tampered)


# ---------------------------------------------------------------------------
# write_iges — the header declares the unit it was given
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("unit_name", "flag"), [("MM", "2"), ("M", "6"), ("INCH", "1")])
def test_write_iges_header_declares_requested_unit(
    box_brep: bytes, unit_name: str, flag: str
) -> None:
    """Global section parameters 14/15 carry the requested unit flag and name."""
    data = write_iges(box_brep, unit=unit_name)

    params = _global_section(data)

    assert params[UNIT_FLAG] == flag
    assert params[UNIT_NAME] == f"{len(unit_name)}H{unit_name}"


@pytest.mark.parametrize("unit_name", ["MM", "M", "INCH"])
def test_write_iges_does_not_rescale_coordinates(box_brep: bytes, unit_name: str) -> None:
    """The declared unit labels the coordinates; it never scales them.

    Global section parameter 20 is the largest coordinate in the file's units. The source BREP
    spans BOX_EDGE, so every export must declare BOX_EDGE — a writer that converted from an
    assumed millimetre model would write 0.002 for "M" and 0.0787 for "INCH".
    """
    data = write_iges(box_brep, unit=unit_name)

    params = _global_section(data)

    assert float(params[MAX_COORD]) == pytest.approx(BOX_EDGE, rel=1e-6)


def test_write_iges_unit_is_the_only_difference(box_brep: bytes) -> None:
    """Exports differing only in declared unit hold the same number of entities."""
    mm = _global_section(write_iges(box_brep, unit="MM"))
    metre = _global_section(write_iges(box_brep, unit="M"))

    assert len(mm) == len(metre)
    assert mm[UNIT_FLAG] != metre[UNIT_FLAG]


def test_write_iges_unit_name_is_case_insensitive(box_brep: bytes) -> None:
    """A lower-case unit name resolves to the same flag as its upper-case spelling."""
    lower = _global_section(write_iges(box_brep, unit="inch"))
    upper = _global_section(write_iges(box_brep, unit="INCH"))

    assert lower[UNIT_FLAG] == upper[UNIT_FLAG]


def test_write_iges_returns_bytes(box_brep: bytes) -> None:
    """The writer returns IGES file content as bytes with a start section."""
    data = write_iges(box_brep, unit="MM")

    assert isinstance(data, bytes)
    assert any(len(line) > 72 and line[72] == "S" for line in data.decode().splitlines())


# ---------------------------------------------------------------------------
# write_iges — round trip through read_iges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("unit_name", "metres"), [("MM", 1.0e-3), ("M", 1.0), ("INCH", 0.0254)])
def test_write_iges_roundtrip_preserves_size_and_unit(
    box_brep: bytes, tmp_path: Path, unit_name: str, metres: float
) -> None:
    """Write then read returns the same coordinates and the unit that was declared."""
    path = tmp_path / f"box_{unit_name}.igs"
    path.write_bytes(write_iges(box_brep, unit=unit_name))

    result = read_iges(path)

    assert result.unit_name == unit_name
    assert result.length_unit == pytest.approx(metres)
    assert _extent(result.brep) == pytest.approx(_extent(box_brep), abs=1e-5)


def test_write_iges_roundtrip_from_imported_unit_name(
    box_inch_iges_path: str, tmp_path: Path
) -> None:
    """unit_name feeds straight back into write_iges: a re-export is unit-exact."""
    imported = read_iges(box_inch_iges_path)
    path = tmp_path / "again.igs"
    path.write_bytes(write_iges(imported.brep, unit=imported.unit_name))

    again = read_iges(path)

    assert again.unit_name == imported.unit_name
    assert again.length_unit == pytest.approx(imported.length_unit)
    assert _extent(again.brep) == pytest.approx(_extent(imported.brep), abs=1e-5)


def test_write_iges_faces_mode_roundtrips(box_brep: bytes, tmp_path: Path) -> None:
    """IGES 5.1 face mode also round-trips the geometry at its native size."""
    path = tmp_path / "faces.igs"
    path.write_bytes(write_iges(box_brep, unit="MM", brep_mode=False))

    result = read_iges(path)

    assert len(load_brep(result.brep).faces()) == 6
    assert _extent(result.brep) == pytest.approx([BOX_EDGE] * 3, abs=1e-5)


def test_write_iges_faces_mode_drops_the_solid(box_brep: bytes, tmp_path: Path) -> None:
    """The documented cost of IGES 5.1: faces survive, the solid does not.

    Pinned so the difference between the two modes stays a stated trade-off rather than a
    surprise: a caller who needs the solid must use brep_mode=True.
    """
    path = tmp_path / "faces_only.igs"
    path.write_bytes(write_iges(box_brep, unit="MM", brep_mode=False))

    assert load_brep(read_iges(path).brep).solids() == []


def test_write_iges_brep_mode_keeps_the_solid(box_brep: bytes, tmp_path: Path) -> None:
    """BRep mode (5.3) carries the solid across; face mode (5.1) cannot."""
    brep_path = tmp_path / "solid.igs"
    brep_path.write_bytes(write_iges(box_brep, unit="MM", brep_mode=True))

    assert len(load_brep(read_iges(brep_path).brep).solids()) == 1


# ---------------------------------------------------------------------------
# write_iges — validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit_name", ["", "MILLIMETRE", "metres", "0.001", "FURLONG"])
def test_write_iges_unknown_unit_raises(box_brep: bytes, unit_name: str) -> None:
    """A name IGES does not define is refused rather than silently defaulted."""
    with pytest.raises(PysmeshError):
        write_iges(box_brep, unit=unit_name)


def test_write_iges_malformed_brep_raises() -> None:
    """Garbage BREP bytes are rejected."""
    with pytest.raises(PysmeshError):
        write_iges(b"not a brep", unit="MM")


def test_write_iges_empty_shape_raises() -> None:
    """A shape with no geometry produces no IGES entity, and says so."""
    empty = pysmesh.Session().brep()

    with pytest.raises(PysmeshError):
        write_iges(empty, unit="MM")


def test_write_iges_brep_mode_refuses_wire_only_shape() -> None:
    """A wire-only shape cannot be written as IGES 5.3, and says so instead of losing it."""
    session = pysmesh.Session()
    session.add_polyline([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0)])

    with pytest.raises(PysmeshError):
        write_iges(session.brep(), unit="MM", brep_mode=True)


# ---------------------------------------------------------------------------
# Unit table & public namespace
# ---------------------------------------------------------------------------


def test_iges_units_table_matches_the_iges_definitions() -> None:
    """IGES_UNITS holds the ten IGES length units at their defined size in metres."""
    assert IGES_UNITS["MM"] == pytest.approx(1.0e-3)
    assert IGES_UNITS["M"] == pytest.approx(1.0)
    assert IGES_UNITS["INCH"] == pytest.approx(0.0254)
    assert IGES_UNITS["FT"] == pytest.approx(12 * 0.0254)
    assert IGES_UNITS["MI"] == pytest.approx(5280 * 12 * 0.0254)
    assert IGES_UNITS["MIL"] == pytest.approx(0.0254 / 1000.0)
    assert IGES_UNITS["UIN"] == pytest.approx(0.0254 / 1_000_000.0)
    assert IGES_UNITS["UM"] == pytest.approx(1.0e-6)
    assert IGES_UNITS["CM"] == pytest.approx(1.0e-2)
    assert IGES_UNITS["KM"] == pytest.approx(1.0e3)


@pytest.mark.parametrize("unit_name", sorted(IGES_UNITS))
def test_write_iges_accepts_every_unit_in_the_table(box_brep: bytes, unit_name: str) -> None:
    """Every name in IGES_UNITS is a name write_iges accepts."""
    data = write_iges(box_brep, unit=unit_name)

    params = _global_section(data)

    assert params[UNIT_NAME] == f"{len(unit_name)}H{unit_name}"


def test_public_namespace_exports() -> None:
    """read_iges / write_iges / IgesImport / IGES_UNITS are importable from pysmesh."""
    assert pysmesh.read_iges is read_iges
    assert pysmesh.write_iges is write_iges
    assert hasattr(pysmesh, "IgesImport")
    assert hasattr(pysmesh, "IGES_UNITS")
