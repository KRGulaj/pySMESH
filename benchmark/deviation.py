# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-11

"""Achieved chordal deviation of a triangle soup from the exact B-rep surface.

Two independent referees, so no claim rests on either library's own projector:

* analytic   closed-form distance to the exact surface. Used for the sphere and the
             truncated cone, where the surface of revolution reduces to a point-to-segment
             or point-to-circle distance in the (r, z) half-plane. No library involved.
* gmsh       gmsh.model.getClosestPoint, an OCC orthogonal projection. Used for imported
             STEP bodies that have no closed form. Judging pySMESH's triangles with gmsh's
             kernel is deliberate: it cannot flatter pySMESH.

Deviation is sampled on a barycentric grid over each triangle, not at its nodes. BRepMesh
places nodes exactly on the surface, so node error is ~1e-15 and says nothing; the chordal
error lives in the triangle interior.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def barycentric_grid(order: int) -> Arr:
    """Barycentric weights for a triangular grid of the given order.

    order=1 gives the centroid only; order=k gives (k+1)(k+2)/2 points including vertices.
    """
    pts = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            k = order - i - j
            pts.append((i / order, j / order, k / order))
    return np.asarray(pts, dtype=np.float64)


def sample_triangles(tris: Arr, order: int = 6, interior: bool = False) -> Arr:
    """Sample points across every triangle. tris is (M, 3, 3) -> (M * P, 3).

    interior=True drops samples on the triangle's own boundary. Orthogonal projection onto a
    trimmed face has no solution for a point sitting exactly on the face's outer boundary, so
    the interior grid is what a projection-based referee can actually evaluate. The chordal
    error is largest in the interior anyway: the vertices lie exactly on the surface.
    """
    w = barycentric_grid(order)  # (P, 3)
    if interior:
        w = w[(w > 0.0).all(axis=1)]
        if len(w) == 0:
            raise ValueError(f"order {order} leaves no interior samples")
    # (M, P, 3) = sum_v w[:, v] * tris[:, v, :]
    return np.einsum("pv,mvc->mpc", w, tris).reshape(-1, 3)


# --------------------------------------------------------------------------------------- #
# Analytic referees
# --------------------------------------------------------------------------------------- #
def dev_sphere(points: Arr, centre: Arr, radius: float) -> Arr:
    """Exact distance from each point to a sphere of the given centre and radius."""
    return np.abs(np.linalg.norm(points - centre[None, :], axis=1) - radius)


def dev_sphere_grid(points: Arr, pitch: float, radius: float, side: int) -> Arr:
    """Distance to the nearest sphere of a cubic lattice of spheres.

    The lattice is axis-aligned with pitch `pitch`, so the owning sphere of a point is found
    by rounding, not by search.
    """
    idx = np.clip(np.rint(points / pitch), 0, side - 1)
    centres = idx * pitch
    return np.abs(np.linalg.norm(points - centres, axis=1) - radius)


def _point_segment_distance_2d(p: Arr, a: Arr, b: Arr) -> Arr:
    """Distance from points p (N,2) to the segment ab, both 2-vectors."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(p - proj, axis=1)


def dev_truncated_cone(points: Arr, r1: float, r2: float, height: float) -> Arr:
    """Exact distance to the boundary of a truncated cone.

    The cone is a surface of revolution about +z with base radius r1 at z=0 and top radius
    r2 at z=height, so the problem collapses to a 2-D distance in the (r, z) half-plane:
    the lateral face is the segment (r1,0)-(r2,H), and each planar cap is the segment
    (0,0)-(r1,0) or (0,H)-(r2,H). The distance to the solid's boundary is the minimum over
    the three, which is exact and needs no projection.
    """
    r = np.hypot(points[:, 0], points[:, 1])
    p = np.column_stack([r, points[:, 2]])
    lateral = _point_segment_distance_2d(
        p, np.array([r1, 0.0]), np.array([r2, height])
    )
    cap0 = _point_segment_distance_2d(p, np.array([0.0, 0.0]), np.array([r1, 0.0]))
    cap1 = _point_segment_distance_2d(
        p, np.array([0.0, height]), np.array([r2, height])
    )
    return np.minimum(lateral, np.minimum(cap0, cap1))


# --------------------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------------------- #
def summarise(dev: Arr) -> dict[str, float]:
    return {
        "dev_max": float(np.max(dev)),
        "dev_rms": float(np.sqrt(np.mean(dev**2))),
        "dev_p99": float(np.percentile(dev, 99.0)),
        "dev_mean": float(np.mean(dev)),
        "n_samples": int(dev.size),
    }


def canonical(tris: Arr, decimals: int = 9) -> Arr:
    """Order-independent canonical form of a triangle soup, for identity testing."""
    t = np.round(tris, decimals)
    order = np.lexsort((t[:, :, 2], t[:, :, 1], t[:, :, 0]), axis=1)
    t = np.take_along_axis(t, order[:, :, None], axis=1)
    flat = np.ascontiguousarray(t.reshape(len(t), 9))
    keys = [flat[:, i] for i in range(8, -1, -1)]
    return flat[np.lexsort(keys)]


def identical(a: Arr, b: Arr, tol: float = 1e-9) -> tuple[bool, float]:
    """Are two triangle soups the same set of triangles?"""
    if a.shape != b.shape:
        return False, float("inf")
    ca, cb = canonical(a), canonical(b)
    d = float(np.abs(ca - cb).max())
    return d <= tol, d
