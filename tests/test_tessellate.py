"""Tests for pysmesh.tessellate (D1).

Covers: output shapes and dtypes, face-id alignment with Shape.faces(), triangle index
validity, normal unit-length and outward-direction invariants (analytical: sphere, box),
winding consistency, relative vs. absolute deflection, parameter validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pysmesh
from pysmesh import PysmeshError, load_brep, tessellate
from pysmesh.tessellate import TessellateParams, TessellateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tess(brep: bytes, lin_defl: float = 0.05, ang_defl_deg: float = 20.0,
          relative: bool = False) -> TessellateResult:
    return tessellate(brep, TessellateParams(lin_defl=lin_defl, ang_defl_deg=ang_defl_deg,
                                             relative=relative))


# ---------------------------------------------------------------------------
# Output shape and dtype invariants
# ---------------------------------------------------------------------------


def test_tessellate_box_output_shapes(box_brep: bytes) -> None:
    """nodes (N,3), tris (M,3), tri_face_id (M,), normals (N,3) with consistent N and M."""
    r = _tess(box_brep)

    n = r.nodes.shape[0]
    m = r.tris.shape[0]
    assert r.nodes.ndim == 2 and r.nodes.shape[1] == 3
    assert r.tris.ndim == 2 and r.tris.shape[1] == 3
    assert r.tri_face_id.ndim == 1 and r.tri_face_id.shape[0] == m
    assert r.normals.ndim == 2 and r.normals.shape == (n, 3)


def test_tessellate_box_output_dtypes(box_brep: bytes) -> None:
    """nodes and normals are float64; tris and tri_face_id are int32."""
    r = _tess(box_brep)

    assert r.nodes.dtype == np.float64
    assert r.normals.dtype == np.float64
    assert r.tris.dtype == np.int32
    assert r.tri_face_id.dtype == np.int32


def test_tessellate_box_nonempty(box_brep: bytes) -> None:
    """A unit box with any positive deflection yields at least 12 triangles (2 per face)."""
    r = _tess(box_brep)

    assert r.nodes.shape[0] > 0
    assert r.tris.shape[0] >= 12


# ---------------------------------------------------------------------------
# Face-id alignment with Shape.faces()
# ---------------------------------------------------------------------------


def test_tessellate_box_face_ids_match_shape(box_brep: bytes) -> None:
    """tri_face_id values are exactly the 1-based ids returned by Shape.faces()."""
    shape = load_brep(box_brep)
    valid_ids = {fi.id for fi in shape.faces()}

    r = _tess(box_brep)

    assert set(r.tri_face_id.tolist()).issubset(valid_ids)


def test_tessellate_box_all_six_faces_covered(box_brep: bytes) -> None:
    """Every face of the box appears in at least one triangle."""
    shape = load_brep(box_brep)
    n_faces = len(shape.faces())

    r = _tess(box_brep)

    assert set(r.tri_face_id.tolist()) == set(range(1, n_faces + 1))


def test_tessellate_sphere_face_id_range(sphere_brep: bytes) -> None:
    """All tri_face_id values are in [1, n_faces] for the sphere."""
    shape = load_brep(sphere_brep)
    n_faces = len(shape.faces())

    r = _tess(sphere_brep, lin_defl=0.05)

    assert int(r.tri_face_id.min()) >= 1
    assert int(r.tri_face_id.max()) <= n_faces


# ---------------------------------------------------------------------------
# Triangle index validity
# ---------------------------------------------------------------------------


def test_tessellate_box_tris_index_range(box_brep: bytes) -> None:
    """All triangle vertex indices are in [0, N_nodes). No out-of-bounds reference."""
    r = _tess(box_brep)

    n = r.nodes.shape[0]
    assert int(r.tris.min()) >= 0
    assert int(r.tris.max()) < n


def test_tessellate_sphere_tris_index_range(sphere_brep: bytes) -> None:
    """Same index-range check for a curved shape."""
    r = _tess(sphere_brep, lin_defl=0.05)

    n = r.nodes.shape[0]
    assert int(r.tris.min()) >= 0
    assert int(r.tris.max()) < n


# ---------------------------------------------------------------------------
# Normal invariants
# ---------------------------------------------------------------------------


def test_tessellate_box_normals_unit_length(box_brep: bytes) -> None:
    """Every non-zero normal has unit length to within float64 precision."""
    r = _tess(box_brep)

    lengths = np.linalg.norm(r.normals, axis=1)
    nonzero = lengths > 0.5  # degenerate surface points carry the zero vector
    assert nonzero.sum() > 0, "Expected at least some non-zero normals"
    np.testing.assert_allclose(lengths[nonzero], 1.0, atol=1e-9)


def test_tessellate_sphere_normals_radial(sphere_brep: bytes) -> None:
    """Sphere normals point radially outward: dot(node_xyz, normal) == |node_xyz| ≈ 1.

    BRepPrimAPI_MakeSphere(1.0) centers the sphere at the origin, so each node lies at
    distance ≈ 1 from the origin and its outward normal equals node_xyz / |node_xyz|.
    Hence dot(node_xyz, normal) ≈ 1.0 for all non-degenerate nodes.
    Cite: unit sphere surface — every surface normal is the outward unit radial direction.
    """
    r = _tess(sphere_brep, lin_defl=0.02, ang_defl_deg=10.0)

    lengths = np.linalg.norm(r.normals, axis=1)
    nonzero = lengths > 0.5
    assert nonzero.sum() > 0

    dots = np.einsum("ij,ij->i", r.nodes[nonzero], r.normals[nonzero])
    np.testing.assert_allclose(dots, 1.0, atol=5e-3)


def test_tessellate_box_normals_axis_aligned(box_brep: bytes) -> None:
    """Planar box faces carry normals that are axis-aligned (each component ∈ {-1, 0, 1}).

    For a rectangular face the underlying surface is a plane; GeomLProp_SLProps returns the
    plane normal at every UV node identically. Each node's normal should have exactly one
    component of magnitude ≈ 1 and the other two ≈ 0.
    """
    r = _tess(box_brep)

    lengths = np.linalg.norm(r.normals, axis=1)
    nonzero = lengths > 0.5
    n_nz = r.normals[nonzero]

    # Each normal has one dominant component and two near-zero components.
    sorted_abs = np.sort(np.abs(n_nz), axis=1)  # ascending
    np.testing.assert_allclose(sorted_abs[:, 0], 0.0, atol=1e-9)  # smallest ≈ 0
    np.testing.assert_allclose(sorted_abs[:, 1], 0.0, atol=1e-9)  # middle ≈ 0
    np.testing.assert_allclose(sorted_abs[:, 2], 1.0, atol=1e-9)  # largest ≈ 1


def test_tessellate_cylinder_flat_cap_normals_axis_aligned(cylinder_brep: bytes) -> None:
    """Cylinder flat cap faces have axis-aligned normals; the wall face has radial normals.

    The flat caps are planes → normals constant ≈ ±axis. The curved wall → normals radial in
    the XY plane (assuming BRepPrimAPI_MakeCylinder axis along Z). We validate that every
    non-zero normal from a cap face has its largest component along Z.
    """
    shape = load_brep(cylinder_brep)
    face_areas = {fi.id: fi.area for fi in shape.faces()}

    r = _tess(cylinder_brep, lin_defl=0.02)

    # The two smallest-area faces are the flat caps (circles), the largest is the wall.
    sorted_by_area = sorted(face_areas.items(), key=lambda kv: kv[1])
    cap_ids = {sorted_by_area[0][0], sorted_by_area[1][0]}

    cap_mask = np.isin(r.tri_face_id, list(cap_ids))
    cap_node_indices = np.unique(r.tris[cap_mask])
    cap_normals = r.normals[cap_node_indices]

    lengths = np.linalg.norm(cap_normals, axis=1)
    nonzero = lengths > 0.5
    if nonzero.sum() == 0:
        pytest.skip("No non-degenerate cap normals found")

    # For a Z-aligned cylinder cap, |n_z| should be the dominant component.
    abs_n = np.abs(cap_normals[nonzero])
    z_dominant = abs_n[:, 2] > 0.9
    assert z_dominant.all(), (
        f"Expected all cap normals z-dominant; got min |n_z| = {abs_n[:, 2].min():.4f}"
    )


# ---------------------------------------------------------------------------
# Winding consistency
# ---------------------------------------------------------------------------


def test_tessellate_box_winding_consistent_with_normals(box_brep: bytes) -> None:
    """Cross product of each triangle's edges agrees with the face normal sign.

    For each triangle, the cross product (v1-v0) × (v2-v0) should point in the same
    hemisphere as the node normals of that triangle (dot > 0), confirming that the winding
    correction for REVERSED faces is applied correctly.
    """
    r = _tess(box_brep)

    v0 = r.nodes[r.tris[:, 0]]
    v1 = r.nodes[r.tris[:, 1]]
    v2 = r.nodes[r.tris[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # (M, 3) unnormalized

    # Use the per-node normal of vertex 0 as the reference outward direction.
    n0 = r.normals[r.tris[:, 0]]
    # Filter to rows where n0 is non-zero (non-degenerate).
    lengths = np.linalg.norm(n0, axis=1)
    nonzero = lengths > 0.5
    assert nonzero.sum() > 0

    dots = np.einsum("ij,ij->i", face_normals[nonzero], n0[nonzero])
    assert (dots > 0).all(), (
        f"Found {(dots <= 0).sum()} triangles with winding opposite to surface normal"
    )


# ---------------------------------------------------------------------------
# Relative deflection mode
# ---------------------------------------------------------------------------


def test_tessellate_box_relative_mode_valid(box_brep: bytes) -> None:
    """relative=True does not raise and produces a valid mesh."""
    r = _tess(box_brep, lin_defl=0.01, relative=True)

    assert r.nodes.shape[0] > 0
    assert r.tris.shape[0] >= 12
    assert int(r.tris.min()) >= 0
    assert int(r.tris.max()) < r.nodes.shape[0]


def test_tessellate_relative_finer_than_absolute(box_brep: bytes) -> None:
    """relative=True with a tight fraction produces at least as many triangles as absolute."""
    r_abs = _tess(box_brep, lin_defl=0.1, relative=False)
    r_rel = _tess(box_brep, lin_defl=0.001, relative=True)

    assert r_rel.tris.shape[0] >= r_abs.tris.shape[0]


# ---------------------------------------------------------------------------
# Default-params convenience
# ---------------------------------------------------------------------------


def test_tessellate_default_params(box_brep: bytes) -> None:
    """Calling tessellate(brep) with no params uses TessellateParams defaults and succeeds."""
    r = tessellate(box_brep)

    assert r.nodes.shape[0] > 0
    assert r.tris.shape[0] >= 12


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lin_defl", [0.0, -0.1, -1e10])
def test_tessellate_params_invalid_lin_defl_raises(lin_defl: float) -> None:
    """Non-positive lin_defl raises PysmeshError before touching OCCT."""
    with pytest.raises(PysmeshError, match="lin_defl"):
        TessellateParams(lin_defl=lin_defl)


@pytest.mark.parametrize("ang_defl_deg", [0.0, -5.0, 180.0, 270.0])
def test_tessellate_params_invalid_ang_defl_raises(ang_defl_deg: float) -> None:
    """Out-of-range ang_defl_deg raises PysmeshError before touching OCCT."""
    with pytest.raises(PysmeshError, match="ang_defl_deg"):
        TessellateParams(ang_defl_deg=ang_defl_deg)


def test_tessellate_malformed_brep_raises() -> None:
    """Garbage bytes raise PysmeshError, not a segfault or generic exception."""
    with pytest.raises(PysmeshError):
        tessellate(b"not a brep", TessellateParams())


def test_tessellate_params_valid_boundary_values() -> None:
    """Boundary-adjacent valid values do not raise."""
    p = TessellateParams(lin_defl=1e-12, ang_defl_deg=0.001)
    assert p.lin_defl == pytest.approx(1e-12)
    assert p.ang_defl_deg == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Public surface: exported from pysmesh top-level
# ---------------------------------------------------------------------------


def test_tessellate_in_pysmesh_namespace() -> None:
    """tessellate, TessellateParams, TessellateResult are importable from pysmesh."""
    assert hasattr(pysmesh, "tessellate")
    assert hasattr(pysmesh, "TessellateParams")
    assert hasattr(pysmesh, "TessellateResult")
