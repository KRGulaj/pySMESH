"""Same-domain healing tests (Tier-2 / C1).

Heals the committed ``split_box`` fixture — two cubes fused along a shared plane, whose four
coplanar seam pairs make a plain rectangular block carry 10 faces instead of 6 — and checks
that ``unify_same_domain`` merges the coplanar pairs, removes the seam edges, preserves the
geometry (total surface area is invariant under same-domain merging), and composes ids
correctly through ``face_map`` / ``edge_map``.

The healed block is a rectangular solid, so its canonical topology is exactly 6 faces and 12
edges — the target the seams should collapse to.
"""

from __future__ import annotations

import numpy as np
import pytest

from pysmesh import PysmeshError, UnifyParams, UnifyResult, load_brep, unify_same_domain

CANONICAL_BOX_FACES = 6
CANONICAL_BOX_EDGES = 12


def _total_area(brep: bytes) -> float:
    """Sum of every face area — invariant under same-domain merging (union of coplanar
    rectangles has the same area as the merged rectangle)."""
    return float(sum(f.area for f in load_brep(brep).faces()))


def test_unify_split_box_reduces_face_count_to_canonical_six(split_box_brep: bytes) -> None:
    result = unify_same_domain(split_box_brep)

    assert result.n_faces_before > CANONICAL_BOX_FACES
    assert result.n_faces_after == CANONICAL_BOX_FACES


def test_unify_split_box_removes_seam_edges(split_box_brep: bytes) -> None:
    result = unify_same_domain(split_box_brep)

    assert result.n_edges_after == CANONICAL_BOX_EDGES
    assert result.n_edges_after < result.n_edges_before
    # The seam edges bounding the merged face pairs are deleted, not remapped.
    assert bool((result.edge_map == -1).any())


def test_unify_split_box_merges_coplanar_faces_many_to_one(split_box_brep: bytes) -> None:
    result = unify_same_domain(split_box_brep)

    # Faces are merged (Modified), never removed: no -1, and every original resolves.
    assert not bool((result.face_map == -1).any())
    survivors = np.unique(result.face_map)
    # Distinct survivors == the healed face count, with strictly fewer survivors than
    # originals => a genuine many-to-one merge.
    assert survivors.size == result.n_faces_after
    assert survivors.size < result.face_map.size


def test_unify_split_box_perpendicular_faces_not_over_merged(split_box_brep: bytes) -> None:
    result = unify_same_domain(split_box_brep)

    # Every surviving face id occurs; two originals that end on different survivors prove
    # non-coplanar faces stay separate (no over-merge). A canonical box has 6 distinct faces.
    assert np.unique(result.face_map).size == CANONICAL_BOX_FACES


def test_unify_preserves_total_surface_area(split_box_brep: bytes) -> None:
    before = _total_area(split_box_brep)

    result = unify_same_domain(split_box_brep)
    after = _total_area(result.brep)

    assert after == pytest.approx(before, rel=1e-9)


def test_unify_result_brep_reloads_with_healed_face_count(split_box_brep: bytes) -> None:
    result = unify_same_domain(split_box_brep)

    reloaded = load_brep(result.brep)

    assert len(reloaded.faces()) == result.n_faces_after == CANONICAL_BOX_FACES


def test_unify_default_params_matches_explicit(split_box_brep: bytes) -> None:
    default = unify_same_domain(split_box_brep)
    explicit = unify_same_domain(split_box_brep, UnifyParams())

    assert default.n_faces_after == explicit.n_faces_after
    assert np.array_equal(default.face_map, explicit.face_map)


def test_unify_faces_disabled_leaves_face_count_unchanged(split_box_brep: bytes) -> None:
    result = unify_same_domain(
        split_box_brep, UnifyParams(unify_faces=False, unify_edges=True)
    )

    assert result.n_faces_after == result.n_faces_before


def test_unify_face_map_is_int32_and_sized_to_before(split_box_brep: bytes) -> None:
    result = unify_same_domain(split_box_brep)

    assert isinstance(result, UnifyResult)
    assert result.face_map.dtype == np.int32
    assert result.edge_map.dtype == np.int32
    assert result.face_map.size == result.n_faces_before
    assert result.edge_map.size == result.n_edges_before


def test_unify_malformed_brep_raises(split_box_brep: bytes) -> None:
    with pytest.raises(PysmeshError):
        unify_same_domain(b"this is not a BREP file")


@pytest.mark.parametrize("bad_tol", [0.0, -1.0e-6])
def test_unify_params_nonpositive_linear_tol_raises(bad_tol: float) -> None:
    with pytest.raises(PysmeshError):
        UnifyParams(linear_tol=bad_tol)


def test_unify_params_negative_angular_raises() -> None:
    with pytest.raises(PysmeshError):
        UnifyParams(angular_tol_deg=-1.0)


def test_unify_params_both_disabled_raises() -> None:
    with pytest.raises(PysmeshError):
        UnifyParams(unify_faces=False, unify_edges=False)
