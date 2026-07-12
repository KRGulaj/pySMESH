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
    step_bytes = write_step_xde(box_brep, name="part")
    assert isinstance(step_bytes, bytes)
    assert step_bytes.startswith(b"ISO-10303-21;")


def test_write_step_bad_face_id_raises(box_brep: bytes) -> None:
    """An out-of-range face id is rejected."""
    with pytest.raises(PysmeshError):
        write_step_xde(box_brep, face_colors={999: (1.0, 0.0, 0.0)})


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
