# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-07-12

"""Tests for pysmesh.make_thick_solid and pysmesh.offset_shape (A2 / D1).

Covers: output shape, dtype, face_map invariants, id alignment with Shape.faces(),
analytical geometry (enlarged-box dimensions, hollowed-box wall count), parameter
validation, BrepCheck integration, and public namespace exports.

Reference specs:
  make_thick_solid: BRepOffsetAPI_MakeThickSolid::MakeThickSolidByJoin — returns a
    hollowed/thickened solid; face_map[i-1] = new 1-based id for original face i, -1
    if the face was removed (opened).
  offset_shape: BRepOffsetAPI_MakeOffsetShape::PerformByJoin (BRepOffset_Skin,
    GeomAbs_Intersection) — uniformly offsets all faces by a signed distance; no faces
    are deleted for a convex solid in Skin mode.
"""

from __future__ import annotations

import pytest

import pysmesh
from pysmesh import (
    PysmeshError,
    Shape,
    load_brep,
    make_thick_solid,
    offset_shape,
)
from pysmesh.offset import (
    OffsetParams,
    OffsetResult,
    ThickSolidParams,
    ThickSolidResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _thick(brep: bytes, remove_ids: tuple[int, ...],
           thickness: float = -0.05, tol: float = 1e-7) -> ThickSolidResult:
    return make_thick_solid(brep, ThickSolidParams(
        remove_face_ids=remove_ids, thickness=thickness, tol=tol
    ))


def _offset(brep: bytes, offset: float = 0.1, tol: float = 1e-7) -> OffsetResult:
    return offset_shape(brep, OffsetParams(offset=offset, tol=tol))


# ---------------------------------------------------------------------------
# make_thick_solid — output shape / type
# ---------------------------------------------------------------------------


def test_make_thick_solid_result_types(box_brep: bytes) -> None:
    """ThickSolidResult.brep is bytes; face_map is int32 ndarray."""
    import numpy as np

    r = _thick(box_brep, (1,))
    assert isinstance(r.brep, bytes)
    assert r.face_map.dtype == np.int32


def test_make_thick_solid_face_map_length(box_brep: bytes) -> None:
    """face_map length equals the number of original faces."""
    shape = load_brep(box_brep)
    n_original = len(shape.faces())
    r = _thick(box_brep, (1,))
    assert r.face_map.shape == (n_original,)


def test_make_thick_solid_removed_face_maps_to_minus_one(box_brep: bytes) -> None:
    """The removed face id maps to -1 in face_map."""
    remove_id = 3
    r = _thick(box_brep, (remove_id,))
    assert r.face_map[remove_id - 1] == -1


def test_make_thick_solid_multiple_removed_faces_all_minus_one(box_brep: bytes) -> None:
    """All explicitly removed face ids map to -1."""
    # Remove two non-adjacent faces from the 6-face box.
    remove_ids = (1, 4)
    r = _thick(box_brep, remove_ids, thickness=-0.05)
    for fid in remove_ids:
        assert r.face_map[fid - 1] == -1


def test_make_thick_solid_kept_faces_map_to_positive_ids(box_brep: bytes) -> None:
    """All non-removed original faces map to positive (valid) new face ids."""
    shape = load_brep(box_brep)
    n = len(shape.faces())
    remove_id = 2
    r = _thick(box_brep, (remove_id,))
    for i in range(1, n + 1):
        if i == remove_id:
            continue
        assert r.face_map[i - 1] > 0, f"face {i} unexpectedly removed"


def test_make_thick_solid_result_brep_is_loadable(box_brep: bytes) -> None:
    """The result BREP bytes can be loaded back as a Shape without error."""
    r = _thick(box_brep, (1,))
    shape = load_brep(r.brep)
    assert len(shape.faces()) > 0


def test_make_thick_solid_result_has_more_faces_than_input(box_brep: bytes) -> None:
    """Hollowing adds inner-shell faces; result has strictly more faces than input."""
    shape_in = load_brep(box_brep)
    n_in = len(shape_in.faces())
    r = _thick(box_brep, (1,))
    shape_out = load_brep(r.brep)
    n_out = len(shape_out.faces())
    # inner shell contributes (n_in - 1) faces + connecting faces at the opening rim
    assert n_out > n_in


def test_make_thick_solid_face_map_new_ids_in_result_range(box_brep: bytes) -> None:
    """Non-negative face_map entries are valid 1-based ids in the result shape."""
    remove_id = 1
    r = _thick(box_brep, (remove_id,))
    shape_out = load_brep(r.brep)
    n_new = len(shape_out.faces())
    for v in r.face_map:
        assert v == -1 or (1 <= v <= n_new), f"face_map value {v} out of range [1, {n_new}]"


def test_make_thick_solid_face_map_kept_ids_unique(box_brep: bytes) -> None:
    """Non-(-1) face_map entries are distinct (no two original faces share a new id)."""
    r = _thick(box_brep, (1,))
    positive = [v for v in r.face_map if v != -1]
    assert len(positive) == len(set(positive))


# ---------------------------------------------------------------------------
# make_thick_solid — parameter validation
# ---------------------------------------------------------------------------


def test_make_thick_solid_empty_remove_ids_raises(box_brep: bytes) -> None:
    """PysmeshError when remove_face_ids is empty (Python layer)."""
    with pytest.raises(PysmeshError):
        ThickSolidParams(remove_face_ids=(), thickness=-0.1)


def test_make_thick_solid_zero_thickness_raises(box_brep: bytes) -> None:
    """PysmeshError when thickness is zero (Python layer)."""
    with pytest.raises(PysmeshError):
        ThickSolidParams(remove_face_ids=(1,), thickness=0.0)


def test_make_thick_solid_nonpositive_tol_raises(box_brep: bytes) -> None:
    """PysmeshError when tol <= 0 (Python layer)."""
    with pytest.raises(PysmeshError):
        ThickSolidParams(remove_face_ids=(1,), thickness=-0.05, tol=0.0)


def test_make_thick_solid_invalid_face_id_raises(box_brep: bytes) -> None:
    """PysmeshError when a remove_face_id is out of range (C++ layer)."""
    with pytest.raises(PysmeshError):
        make_thick_solid(box_brep, ThickSolidParams(remove_face_ids=(99,), thickness=-0.05))


def test_make_thick_solid_malformed_brep_raises() -> None:
    """PysmeshError on garbage BREP bytes."""
    with pytest.raises(PysmeshError):
        make_thick_solid(b"not-a-brep", ThickSolidParams(remove_face_ids=(1,), thickness=-0.05))


# ---------------------------------------------------------------------------
# offset_shape — output shape / type
# ---------------------------------------------------------------------------


def test_offset_shape_result_types(box_brep: bytes) -> None:
    """OffsetResult.brep is bytes; face_map is int32 ndarray."""
    import numpy as np

    r = _offset(box_brep, offset=0.1)
    assert isinstance(r.brep, bytes)
    assert r.face_map.dtype == np.int32


def test_offset_shape_face_map_length(box_brep: bytes) -> None:
    """face_map length equals number of input faces."""
    shape = load_brep(box_brep)
    n = len(shape.faces())
    r = _offset(box_brep, offset=0.1)
    assert r.face_map.shape == (n,)


def test_offset_shape_outward_no_deleted_faces(box_brep: bytes) -> None:
    """For a convex solid offset outward (Skin mode), no faces are deleted (-1)."""
    r = _offset(box_brep, offset=0.1)
    assert all(v > 0 for v in r.face_map), "unexpected -1 entry in offset_shape face_map"


def test_offset_shape_inward_no_deleted_faces(box_brep: bytes) -> None:
    """For a small inward offset that does not self-intersect, no faces are deleted."""
    r = _offset(box_brep, offset=-0.1)
    assert all(v > 0 for v in r.face_map)


def test_offset_shape_result_brep_is_loadable(box_brep: bytes) -> None:
    """Result BREP can be loaded back as a Shape."""
    r = _offset(box_brep, offset=0.1)
    shape = load_brep(r.brep)
    assert len(shape.faces()) > 0


def test_offset_shape_face_map_new_ids_in_range(box_brep: bytes) -> None:
    """All face_map values are valid 1-based ids in the result shape."""
    r = _offset(box_brep, offset=0.1)
    shape_out = load_brep(r.brep)
    n_new = len(shape_out.faces())
    for v in r.face_map:
        assert v == -1 or (1 <= v <= n_new)


def test_offset_shape_face_map_outward_ids_unique(box_brep: bytes) -> None:
    """face_map entries are distinct for outward offset of a box (one-to-one)."""
    r = _offset(box_brep, offset=0.1)
    positive = [v for v in r.face_map if v != -1]
    assert len(positive) == len(set(positive))


def test_offset_shape_box_outward_same_face_count(box_brep: bytes) -> None:
    """Offsetting a convex box outward (Skin + Intersection join) preserves face count."""
    shape_in = load_brep(box_brep)
    n_in = len(shape_in.faces())
    r = _offset(box_brep, offset=0.1)
    shape_out = load_brep(r.brep)
    n_out = len(shape_out.faces())
    # GeomAbs_Intersection join does not add fillet faces for flat-faced solids;
    # expect same face count.
    assert n_out == n_in


def test_offset_shape_cylinder_outward(cylinder_brep: bytes) -> None:
    """offset_shape succeeds on a curved solid (cylinder)."""
    r = _offset(cylinder_brep, offset=0.05)
    shape_out = load_brep(r.brep)
    assert len(shape_out.faces()) > 0
    assert r.face_map.shape[0] == len(load_brep(cylinder_brep).faces())


# ---------------------------------------------------------------------------
# offset_shape — parameter validation
# ---------------------------------------------------------------------------


def test_offset_shape_zero_offset_raises(box_brep: bytes) -> None:
    """PysmeshError when offset is zero (Python layer)."""
    with pytest.raises(PysmeshError):
        OffsetParams(offset=0.0)


def test_offset_shape_nonpositive_tol_raises(box_brep: bytes) -> None:
    """PysmeshError when tol <= 0 (Python layer)."""
    with pytest.raises(PysmeshError):
        OffsetParams(offset=0.1, tol=-1e-7)


def test_offset_shape_malformed_brep_raises() -> None:
    """PysmeshError on garbage BREP bytes."""
    with pytest.raises(PysmeshError):
        offset_shape(b"garbage", OffsetParams(offset=0.1))


# ---------------------------------------------------------------------------
# What the result has to be (4.2.0)
#
# These are the statements Session has carried since 4.1.2, on the module that
# reaches the same two OCCT calls from the other side. Before 4.2.0 this module
# had BRepOffsetAPI's own IsDone(), a null check and BRepCheck_Analyzer, and
# committed every case below. A face is named here by its 1-based ordinal in
# the input's face map, the same ordinal remove_face_ids is given in.
#
# The cylinder fixture is radius 1 by height 3, the sphere radius 1, and the
# box 2 x 2 x 2.
# ---------------------------------------------------------------------------


def _volume(brep: bytes) -> float:
    """Total volume of the solids in a BREP, through a throwaway session."""
    from pysmesh import Session

    s = Session()
    s.add_brep(brep)
    ids = [int(i) for i in s.entity_types("SOLID").ids]
    return float(sum(s.mass_properties(ids).measure)) if ids else float("nan")


@pytest.mark.parametrize("offset", [-1.0, -1.01, -1.5, -3.0])
def test_offset_shape_past_a_cylinder_radius_raises(cylinder_brep: bytes,
                                                    offset: float) -> None:
    """Past its radius OCCT rebuilds the wall mirrored through its own axis.

    Measured on a cylinder of radius 2 and height 7 -- the same surface, scaled --
    a shrink of 2.01 committed 0.0009361946107697187, which is exactly the
    r = 0.01 cylinder. The result is a valid solid of positive volume with the
    right three faces, so only a statement made before OCCT runs can see it.
    """
    with pytest.raises(PysmeshError, match="do not survive it") as excinfo:
        _offset(cylinder_brep, offset)

    assert "cylinder of radius" in str(excinfo.value)
    assert list(excinfo.value.face_ids) == [1]


def test_offset_shape_up_to_a_cylinder_radius_still_commits(cylinder_brep: bytes) -> None:
    """The regime the rule must leave alone: a sliver of a cylinder is still one."""
    import math

    result = _offset(cylinder_brep, -0.9)

    assert _volume(result.brep) == pytest.approx(math.pi * 0.1**2 * (3.0 - 1.8))


@pytest.mark.parametrize("offset", [-1.0, -1.5])
def test_offset_shape_past_a_sphere_radius_raises(sphere_brep: bytes, offset: float) -> None:
    """A sphere shrunk by its own radius has nothing left to be."""
    with pytest.raises(PysmeshError, match="do not survive it") as excinfo:
        _offset(sphere_brep, offset)

    assert "sphere of radius" in str(excinfo.value)


@pytest.mark.parametrize("thickness", [-1.05, -1.5])
def test_make_thick_solid_past_a_cylinder_radius_raises(cylinder_brep: bytes,
                                                        thickness: float) -> None:
    """Opened at a planar cap, the curved wall is what the thickness is spent on.

    This is the case the opening's surface type used to decide: opened at the
    curved wall OCCT declines, and opened at a cap it committed a wall thicker
    than the body -- 87.81 against a solid of 87.96, measured on radius 2.
    """
    with pytest.raises(PysmeshError, match="do not survive it") as excinfo:
        _thick(cylinder_brep, (2,), thickness)

    assert "cylinder of radius" in str(excinfo.value)
    assert list(excinfo.value.face_ids) == [1]


def test_make_thick_solid_does_not_judge_the_faces_it_opens(cylinder_brep: bytes) -> None:
    """An opened face is removed, not offset, so its radius is not spent.

    Opening the curved wall leaves two caps with no wall between them, which OCCT
    declines on its own. The point under test is which refusal arrives: the radius
    rule must not be the one speaking, because the face carrying the radius is the
    one being removed.
    """
    with pytest.raises(PysmeshError) as excinfo:
        _thick(cylinder_brep, (1,), -1.05)

    assert "do not survive it" not in str(excinfo.value)
    assert "IsDone() is false" in str(excinfo.value)


@pytest.mark.parametrize("thickness", [-1.0, -2.0, -5.0])
def test_make_thick_solid_refuses_a_result_that_is_the_input(box_brep: bytes,
                                                             thickness: float) -> None:
    """Past what the box can carry, the inner shell collapses and the input comes back.

    MakeThickSolidByJoin reports IsDone(), BRepCheck_Analyzer accepts the shape,
    and the shape is the 2 x 2 x 2 box itself: volume 8.0 against an input of 8.0,
    with the opened face re-issued under a new ordinal. Half the box's smallest
    extent is 1.0, which is where it starts.
    """
    with pytest.raises(PysmeshError, match="is not a hollowed solid") as excinfo:
        _thick(box_brep, (1,), thickness)

    assert "no cavity was cut" in str(excinfo.value)
    assert list(excinfo.value.face_ids) == [1]


def test_make_thick_solid_refuses_a_plate_that_came_back_inside_out(box_brep: bytes) -> None:
    """Every face but one opened, thickened outward: the plate has negative volume."""
    with pytest.raises(PysmeshError, match="is not a hollowed solid") as excinfo:
        _thick(box_brep, (1, 2, 3, 4, 5), 0.5)

    assert "turned inside out" in str(excinfo.value)


def test_make_thick_solid_refuses_an_opening_of_the_whole_boundary(box_brep: bytes) -> None:
    """With every face opened there is no wall to build, and the input comes back."""
    with pytest.raises(PysmeshError, match="is not a hollowed solid"):
        _thick(box_brep, (1, 2, 3, 4, 5, 6), -0.3)


def test_make_thick_solid_still_hollows_what_the_box_can_carry(box_brep: bytes) -> None:
    """The thin wall the statements must not touch: 2 x 2 x 2 walled by 0.3."""
    result = _thick(box_brep, (1,), -0.3)

    cavity = (2.0 - 0.6) * (2.0 - 0.6) * (2.0 - 0.3)
    assert _volume(result.brep) == pytest.approx(8.0 - cavity)


def test_offset_shape_still_grows_and_shrinks_the_box(box_brep: bytes) -> None:
    """Both signs of the volume statement, on distances the box carries."""
    assert _volume(_offset(box_brep, 0.5).brep) == pytest.approx(3.0**3)
    assert _volume(_offset(box_brep, -0.4).brep) == pytest.approx(1.2**3)


# ---------------------------------------------------------------------------
# Public namespace
# ---------------------------------------------------------------------------


def test_public_namespace_exports() -> None:
    """All A2 types and functions are importable from pysmesh directly."""
    assert hasattr(pysmesh, "make_thick_solid")
    assert hasattr(pysmesh, "offset_shape")
    assert hasattr(pysmesh, "ThickSolidParams")
    assert hasattr(pysmesh, "ThickSolidResult")
    assert hasattr(pysmesh, "OffsetParams")
    assert hasattr(pysmesh, "OffsetResult")
    assert pysmesh.make_thick_solid is make_thick_solid
    assert pysmesh.offset_shape is offset_shape


def test_thick_solid_params_defaults() -> None:
    """ThickSolidParams default tol is 1e-7."""
    p = ThickSolidParams(remove_face_ids=(1,), thickness=-0.1)
    assert p.tol == pytest.approx(1e-7)


def test_offset_params_defaults() -> None:
    """OffsetParams default tol is 1e-7."""
    p = OffsetParams(offset=0.1)
    assert p.tol == pytest.approx(1e-7)


def test_thick_solid_result_is_frozen() -> None:
    """ThickSolidResult is a frozen dataclass (immutable)."""
    import numpy as np

    r = ThickSolidResult(brep=b"x", face_map=np.array([1], dtype=np.int32))
    with pytest.raises((AttributeError, TypeError)):
        r.brep = b"y"  # type: ignore[misc]


def test_offset_result_is_frozen() -> None:
    """OffsetResult is a frozen dataclass (immutable)."""
    import numpy as np

    r = OffsetResult(brep=b"x", face_map=np.array([1], dtype=np.int32))
    with pytest.raises((AttributeError, TypeError)):
        r.brep = b"y"  # type: ignore[misc]
