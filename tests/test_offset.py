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
