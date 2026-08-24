# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-07-12

"""Tier-2 STEP XDE tests: read_step_xde, write_step_xde, and Shape.solids (A4/B1).

Fixtures are labelled STEP files from ``tests/fixtures/generate_fixtures.cpp``:
  named_box_mm.step  a 2 mm cube declared in millimetres; product name ``blade_solid``; face 1
                     (1-based TopExp ordinal) coloured pure red (1, 0, 0).
  named_box_m.step   a 2 m cube declared in metres — same native coordinate extent (2.0) as the
                     mm fixture, so the two differ only in the declared length unit.

Reference spec:
  read_step_xde returns geometry in the file's NATIVE unit; length_unit is metres per model unit
  (mm → 0.001, m → 1.0). Names/colours are keyed to the returned BREP's 1-based TopExp ordinals,
  so load_brep(brep).faces()/.solids() reproduce those ids.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import pysmesh
from pysmesh import (
    EntityLabel,
    PysmeshError,
    StepImport,
    load_brep,
    read_step_xde,
    write_step_xde,
)

BOX_EDGE = 2.0


# ---------------------------------------------------------------------------
# read_step_xde — structure & geometry
# ---------------------------------------------------------------------------


def test_read_step_returns_step_import(named_box_mm_step_path: str) -> None:
    """The wrapper returns a StepImport dataclass."""
    result = read_step_xde(named_box_mm_step_path)
    assert isinstance(result, StepImport)


def test_read_step_brep_loads_to_cube(named_box_mm_step_path: str) -> None:
    """The returned BREP is a single 6-face, 1-solid cube."""
    result = read_step_xde(named_box_mm_step_path)
    shape = load_brep(result.brep)
    assert len(shape.faces()) == 6
    assert len(shape.solids()) == 1


def test_read_step_native_extent_is_box_edge(named_box_mm_step_path: str) -> None:
    """Native coordinates: the cube spans BOX_EDGE per axis (no silent rescale to metres)."""
    result = read_step_xde(named_box_mm_step_path)
    shape = load_brep(result.brep)
    bboxes = np.array([f.bbox for f in shape.faces()])
    extent = bboxes[:, 3:].max(axis=0) - bboxes[:, :3].min(axis=0)
    assert extent == pytest.approx([BOX_EDGE, BOX_EDGE, BOX_EDGE], abs=1e-5)


# ---------------------------------------------------------------------------
# read_step_xde — length unit (the mm-imported-as-metres fix)
# ---------------------------------------------------------------------------


def test_read_step_mm_length_unit_is_milli(named_box_mm_step_path: str) -> None:
    """A millimetre STEP file reports length_unit 0.001 (metres per unit)."""
    result = read_step_xde(named_box_mm_step_path)
    assert result.length_unit == pytest.approx(1.0e-3)


def test_read_step_m_length_unit_is_one(named_box_m_step_path: str) -> None:
    """A metre STEP file reports length_unit 1.0."""
    result = read_step_xde(named_box_m_step_path)
    assert result.length_unit == pytest.approx(1.0)


def test_read_step_physical_size_via_length_unit(
    named_box_mm_step_path: str, named_box_m_step_path: str
) -> None:
    """Native extent * length_unit recovers the true physical size: 0.002 m vs 2 m."""
    mm = read_step_xde(named_box_mm_step_path)
    m = read_step_xde(named_box_m_step_path)
    assert BOX_EDGE * mm.length_unit == pytest.approx(2.0e-3)
    assert BOX_EDGE * m.length_unit == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# read_step_xde — names & colours keyed to ordinals
# ---------------------------------------------------------------------------


def test_read_step_solid_name(named_box_mm_step_path: str) -> None:
    """The product name lands on solid id 1."""
    result = read_step_xde(named_box_mm_step_path)
    assert len(result.solid_labels) == 1
    label = result.solid_labels[0]
    assert isinstance(label, EntityLabel)
    assert label.id == 1
    assert label.name == "blade_solid"


def test_read_step_face_color(named_box_mm_step_path: str) -> None:
    """Face 1 carries the pure-red surface colour."""
    result = read_step_xde(named_box_mm_step_path)
    reds = [f for f in result.face_labels if f.color is not None]
    assert len(reds) == 1
    label = reds[0]
    assert label.id == 1
    assert label.color == pytest.approx((1.0, 0.0, 0.0))


def test_read_step_labels_only_for_tagged_entities(named_box_mm_step_path: str) -> None:
    """Only tagged entities appear: one coloured face, no spurious labels on the other five."""
    result = read_step_xde(named_box_mm_step_path)
    assert len(result.face_labels) == 1


def test_read_step_color_ordinal_matches_loaded_shape(named_box_mm_step_path: str) -> None:
    """The coloured face id is a valid 1-based ordinal of the loaded shape."""
    result = read_step_xde(named_box_mm_step_path)
    shape = load_brep(result.brep)
    valid_ids = {f.id for f in shape.faces()}
    assert result.face_labels[0].id in valid_ids


# ---------------------------------------------------------------------------
# read_step_xde — input modes & validation
# ---------------------------------------------------------------------------


def test_read_step_accepts_bytes(named_box_mm_step_bytes: bytes) -> None:
    """Raw STEP bytes are accepted and yield the same solid name as the path input."""
    result = read_step_xde(named_box_mm_step_bytes)
    assert result.solid_labels[0].name == "blade_solid"


def test_read_step_bytes_and_path_agree(
    named_box_mm_step_path: str, named_box_mm_step_bytes: bytes
) -> None:
    """Bytes and path inputs produce the same length unit and label counts."""
    from_path = read_step_xde(named_box_mm_step_path)
    from_bytes = read_step_xde(named_box_mm_step_bytes)
    assert from_path.length_unit == from_bytes.length_unit
    assert len(from_path.face_labels) == len(from_bytes.face_labels)


def test_read_step_malformed_raises() -> None:
    """Garbage bytes are rejected."""
    with pytest.raises(PysmeshError):
        read_step_xde(b"not a step file")


# ---------------------------------------------------------------------------
# write_step_xde — round-trip
# ---------------------------------------------------------------------------


def test_write_step_roundtrip_name_and_color(box_brep: bytes) -> None:
    """Names and colours written to STEP survive a read_step_xde round-trip."""
    step_bytes = write_step_xde(
        box_brep,
        unit="MM",
        name="wing",
        face_names={2: "inlet"},
        face_colors={1: (0.0, 1.0, 0.0)},
    )
    result = read_step_xde(step_bytes)

    assert result.solid_labels[0].name == "wing"
    greens = [f for f in result.face_labels if f.color is not None]
    assert greens[0].id == 1
    assert greens[0].color == pytest.approx((0.0, 1.0, 0.0))


def test_write_step_returns_bytes(box_brep: bytes) -> None:
    """The writer returns STEP file content as bytes beginning with the ISO-10303 header."""
    step_bytes = write_step_xde(box_brep, unit="MM", name="part")
    assert isinstance(step_bytes, bytes)
    assert step_bytes.startswith(b"ISO-10303-21;")


def test_write_step_bad_face_id_raises(box_brep: bytes) -> None:
    """An out-of-range face id is rejected."""
    with pytest.raises(PysmeshError):
        write_step_xde(box_brep, unit="MM", face_colors={999: (1.0, 0.0, 0.0)})


# ---------------------------------------------------------------------------
# Shape.solids (A4 id home for solid-level names)
# ---------------------------------------------------------------------------


def test_shape_solids_box_single_solid(box_brep: bytes) -> None:
    """The box is one solid with id 1 and the analytical volume BOX_EDGE**3."""
    shape = load_brep(box_brep)
    solids = shape.solids()
    assert len(solids) == 1
    assert solids[0].id == 1
    assert solids[0].volume == pytest.approx(BOX_EDGE**3)


def test_shape_solids_centroid_is_box_center(box_brep: bytes) -> None:
    """The solid centroid is the box centre."""
    shape = load_brep(box_brep)
    centroid = np.asarray(shape.solids()[0].centroid)
    assert centroid == pytest.approx([1.0, 1.0, 1.0])


def test_shape_solids_open_shell_is_empty(open_box_shell_brep: bytes) -> None:
    """An open shell contains no solids."""
    shape = load_brep(open_box_shell_brep)
    assert shape.solids() == []


# ---------------------------------------------------------------------------
# Public namespace
# ---------------------------------------------------------------------------


def test_public_namespace_exports() -> None:
    """read_step_xde / write_step_xde / StepImport / EntityLabel are importable from pysmesh."""
    assert pysmesh.read_step_xde is read_step_xde
    assert pysmesh.write_step_xde is write_step_xde
    assert hasattr(pysmesh, "StepImport")
    assert hasattr(pysmesh, "EntityLabel")
    assert hasattr(pysmesh, "SolidInfo")


# ---------------------------------------------------------------------------
# write_step_xde: the declared length unit (4.0.0)
#
# Mirrors the IGES unit contract in test_iges.py. Before 4.0.0 write_step_xde
# took no unit and emitted OCCT's global default of millimetres whatever the
# coordinates were, so a metre model round-tripped 1000x smaller with nothing
# to warn the caller. These tests fail on that behaviour.
# ---------------------------------------------------------------------------


def _solid_extent(brep: bytes) -> np.ndarray:
    """Bounding-box extent of the first solid, in the BREP's own coordinates."""
    session = pysmesh.Session()
    session.add_brep(brep)
    bbox = session.bounding_boxes(pysmesh.EntityKind.SOLID).bbox[0]
    return np.asarray(bbox[3:] - bbox[:3], dtype=float)


@pytest.mark.parametrize("unit_name", ["MM", "CM", "M", "INCH", "FT", "KM"])
def test_write_step_does_not_rescale_coordinates(box_brep: bytes, unit_name: str) -> None:
    """``unit`` labels the coordinates; it never scales them."""
    before = _solid_extent(box_brep)

    after = _solid_extent(read_step_xde(write_step_xde(box_brep, unit=unit_name)).brep)

    assert np.allclose(after, before, rtol=1e-9)


@pytest.mark.parametrize(
    ("unit_name", "expected_factor"),
    [("MM", 1.0e-3), ("CM", 1.0e-2), ("M", 1.0), ("INCH", 0.0254), ("FT", 0.3048)],
)
def test_write_step_header_declares_requested_unit(
    box_brep: bytes, unit_name: str, expected_factor: float
) -> None:
    """A file written in ``unit`` reads back reporting that unit's factor."""
    result = read_step_xde(write_step_xde(box_brep, unit=unit_name))

    assert result.length_unit == pytest.approx(expected_factor, rel=1e-12)
    assert result.unit_name == unit_name


def test_write_step_unit_is_case_insensitive(box_brep: bytes) -> None:
    """Unit names are matched case-insensitively, as in write_iges."""
    lower = read_step_xde(write_step_xde(box_brep, unit="inch"))
    upper = read_step_xde(write_step_xde(box_brep, unit="INCH"))

    assert lower.length_unit == upper.length_unit == pytest.approx(0.0254, rel=1e-12)


def test_write_step_unknown_unit_raises(box_brep: bytes) -> None:
    """An unrecognised unit fails loudly rather than falling back to millimetres."""
    with pytest.raises(PysmeshError, match="unit"):
        write_step_xde(box_brep, unit="FURLONG")


@pytest.mark.parametrize("fixture_name", ["named_box_mm.step", "named_box_m.step"])
def test_write_step_roundtrip_preserves_physical_size(
    fixtures_dir: Path, fixture_name: str
) -> None:
    """Read a STEP file, re-export it in its own unit, and the physical size survives.

    This is the regression test for the 4.0.0 fix. On the metre fixture the pre-4.0.0
    writer produced a file 1000x smaller, because it declared millimetres regardless.
    """
    original = read_step_xde(str(fixtures_dir / fixture_name))
    physical_before = _solid_extent(original.brep) * original.length_unit

    reread = read_step_xde(write_step_xde(original.brep, unit=original.unit_name))
    physical_after = _solid_extent(reread.brep) * reread.length_unit

    assert reread.unit_name == original.unit_name
    assert reread.length_unit == pytest.approx(original.length_unit, rel=1e-12)
    assert np.allclose(physical_after, physical_before, rtol=1e-6)


def test_read_step_unit_name_matches_length_unit(named_box_m_step_path: str) -> None:
    """``unit_name`` and ``length_unit`` describe the same unit."""
    from pysmesh import IGES_UNITS

    result = read_step_xde(named_box_m_step_path)

    assert result.unit_name == "M"
    assert IGES_UNITS[result.unit_name] == pytest.approx(result.length_unit, rel=1e-12)


def test_write_step_unit_does_not_leak_into_iges(box_brep: bytes) -> None:
    """Writing STEP in one unit must not disturb the IGES writer's unit.

    Both bindings set their unit on their own model rather than on OCCT's global
    ``Interface_Static``, so neither can affect the other inside one process.
    """
    from pysmesh import read_iges, write_iges

    write_step_xde(box_brep, unit="KM")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "after_step.igs"
        path.write_bytes(write_iges(box_brep, unit="MM"))
        result = read_iges(str(path))

    assert result.unit_name == "MM"
    assert result.length_unit == pytest.approx(1.0e-3, rel=1e-12)


@pytest.mark.parametrize("unit_name", ["UM", "MIL", "UIN"])
def test_write_step_extreme_ratio_units_keep_the_unit_but_lose_digits(
    box_brep: bytes, unit_name: str
) -> None:
    """The unit survives on extreme-ratio units; the coordinates lose a little precision.

    A microinch is 2.54e-8 m. OCCT normalises STEP geometry to millimetres internally, so a
    round trip through such a unit divides and re-multiplies by ~1e-5 and the low digits do
    not survive. The declared unit is still exact, and the drift is well under a part in 100.
    Practical CAD units (MM, CM, M, INCH, FT, KM) round-trip exactly; those are covered by
    ``test_write_step_does_not_rescale_coordinates``.
    """
    before = _solid_extent(box_brep)

    result = read_step_xde(write_step_xde(box_brep, unit=unit_name))

    assert result.unit_name == unit_name
    assert np.allclose(_solid_extent(result.brep), before, rtol=1e-2)
