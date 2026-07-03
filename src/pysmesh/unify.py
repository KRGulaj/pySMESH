"""Same-domain topology healing (Tier-2).

Public surface: :class:`UnifyParams`, :class:`UnifyResult`, and :func:`unify_same_domain`.
These wrap the low-level ``_core.unify_same_domain`` (which returns raw BREP bytes + NumPy
arrays) in frozen dataclasses and validate parameters up front.

``unify_same_domain`` merges adjacent faces that lie on one underlying surface — and
collinear edges — into a single face/edge, deleting the artificial seams that
over-segmented STEP imports carry. It is a pure OCCT B-rep operation
(``ShapeUpgrade_UnifySameDomain``): the seam face/edge is removed from the topology, so a
downstream mesher never places nodes along it. Both the input and the healed output cross
the boundary as raw BREP bytes, exactly like :func:`load_brep`.

``face_map`` / ``edge_map`` follow pySMESH's 1-based TopExp id convention (the same ids
:meth:`Shape.faces` / :meth:`Shape.edges` return): ``face_map[i - 1]`` is the new face id
that original face id ``i`` survives as, or ``-1`` if it was removed. Merged originals share
one survivor (many-to-one), which is how a consumer composes a pre-heal tag onto the
post-heal geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._core import PysmeshError
from ._core import unify_same_domain as _unify_same_domain

# OCCT ShapeUpgrade_UnifySameDomain defaults (Precision::Confusion for the linear tolerance).
# The angular default is Precision::Angular (~1e-12 rad); passing 0.0 lets OCCT clamp to it.
_DEFAULT_LINEAR_TOL: float = 1.0e-7


@dataclass(frozen=True)
class UnifyParams:
    """Same-domain healing parameters.

    Attributes:
        unify_faces: Merge adjacent faces sharing one underlying surface.
        unify_edges: Merge collinear/cotangent edges sharing one underlying curve.
        concat_bsplines: Concatenate merged B-spline geometry into a single surface/curve
            instead of leaving it trimmed. Off by default (structure-preserving).
        linear_tol: Chord tolerance [model units] for deciding coplanarity (> 0).
        angular_tol_deg: Maximum connection angle [degrees] for merging; two shapes forming
            a sharper angle are left separate. 0.0 uses OCCT's tight default (~1e-12 rad).

    Raises:
        PysmeshError: On any out-of-range parameter.
    """

    unify_faces: bool = True
    unify_edges: bool = True
    concat_bsplines: bool = False
    linear_tol: float = _DEFAULT_LINEAR_TOL
    angular_tol_deg: float = 0.0

    def __post_init__(self) -> None:
        if not self.linear_tol > 0.0:
            raise PysmeshError(
                f"UnifyParams.linear_tol must be > 0 (got {self.linear_tol})."
            )
        if self.angular_tol_deg < 0.0:
            raise PysmeshError(
                f"UnifyParams.angular_tol_deg must be >= 0 (got {self.angular_tol_deg})."
            )
        if not (self.unify_faces or self.unify_edges):
            raise PysmeshError(
                "UnifyParams must enable at least one of unify_faces / unify_edges."
            )


@dataclass(frozen=True)
class UnifyResult:
    """Result of :func:`unify_same_domain`.

    Attributes:
        brep: The healed shape as BREP bytes (re-loadable via :func:`load_brep`).
        n_faces_before: Face count of the input shape.
        n_faces_after: Face count of the healed shape (<= ``n_faces_before``).
        n_edges_before: Edge count of the input shape.
        n_edges_after: Edge count of the healed shape (<= ``n_edges_before``).
        face_map: (n_faces_before,) int32 — new 1-based face id per original face id (row
            ``i`` is original id ``i + 1``), or ``-1`` if the face was removed.
        edge_map: (n_edges_before,) int32 — same mapping for edges.
    """

    brep: bytes
    n_faces_before: int
    n_faces_after: int
    n_edges_before: int
    n_edges_after: int
    face_map: NDArray[np.int32]
    edge_map: NDArray[np.int32]


def unify_same_domain(brep: bytes, params: UnifyParams | None = None) -> UnifyResult:
    """Merge same-domain faces/edges of a BREP shape.

    Args:
        brep: Input shape as BREP bytes (e.g. from :func:`load_brep`'s source, or any OCCT
            ``BRepTools::Write`` output).
        params: Healing parameters. Defaults to :class:`UnifyParams` (faces + edges, OCCT
            default tolerances).

    Returns:
        The healed shape plus before/after counts and the id-composition maps.

    Raises:
        PysmeshError: On a malformed BREP, or if the healed shape fails OCCT's validity
            check (``BRepCheck_Analyzer``).
    """
    p = params if params is not None else UnifyParams()
    raw = _unify_same_domain(
        brep,
        p.unify_faces,
        p.unify_edges,
        p.concat_bsplines,
        p.linear_tol,
        math.radians(p.angular_tol_deg),
    )
    return UnifyResult(
        brep=cast("bytes", raw["brep"]),
        n_faces_before=cast("int", raw["n_faces_before"]),
        n_faces_after=cast("int", raw["n_faces_after"]),
        n_edges_before=cast("int", raw["n_edges_before"]),
        n_edges_after=cast("int", raw["n_edges_after"]),
        face_map=cast("NDArray[np.int32]", raw["face_map"]),
        edge_map=cast("NDArray[np.int32]", raw["edge_map"]),
    )
