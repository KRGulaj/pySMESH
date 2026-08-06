"""pySMESH session — the render mesh, and the incremental delta over it.

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import (
    CancelPredicate,
    ProgressCallback,
    RenderMesh,
    _DEFAULT_DEFLECTION,
    _DEFAULT_MESH_ANGLE_DEG,
)


class _RenderOps(_SessionBase):
    """The render mesh, and the incremental delta over it."""

    __slots__ = ()

    def tessellate(
        self,
        *,
        deflection: float = _DEFAULT_DEFLECTION,
        angle_deg: float = _DEFAULT_MESH_ANGLE_DEG,
        relative: bool = False,
        parallel: bool = True,
        incremental: bool = True,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> RenderMesh:
        """Triangles, edge polylines and vertex points of the live shape, from one call.

        Two things separate this from the stateless :func:`tessellate`, and both come from
        the session owning its shape rather than re-reading it.

        **The work is proportional to what changed.** ``BRepMesh`` caches its triangulation
        on the face itself, and a session keeps its faces alive across operations, so a face
        no operation touched is not re-triangulated. Reading a shape back from bytes produces
        new faces with no triangulation, which is why the stateless entry point re-meshes the
        whole model every time however little of it moved.

        **The result says what changed.** :attr:`RenderMesh.changed` names the faces whose
        nodes differ from the previous call's, and :attr:`RenderMesh.retriangulated` the
        subset that was genuinely re-meshed rather than merely moved. A consumer holding
        anything derived from the arrays cannot use the first property without the second
        piece of information, and recovering it by diffing the arrays costs more than the
        tessellation saved.

        Not an operation: no ids are issued, :meth:`op_count` does not advance and the
        topology is untouched. It does write the triangulation onto the shape, which is a
        cache and is shared with retained states — a snapshot taken before this call and
        restored after it sees the same geometry, and gets the cached mesh for free.

        Args:
            deflection: Largest distance a mesh edge may stray from the true geometry, in
                model units (>= 1e-7). In relative mode, a fraction of each edge's own length.
            angle_deg: Largest turn a single mesh edge may span, in degrees, in (0, 180).
                Smaller values refine tightly curved geometry.
            relative: Interpret ``deflection`` per edge, as a fraction of that edge's length,
                rather than as an absolute distance. Per *edge*, not per model, so growing the
                model does not invalidate what is already meshed.
            parallel: Mesh faces on several threads. The result is the same either way.
            incremental: Keep what is already meshed and re-triangulate only what needs it.
                ``False`` discards every cached triangulation first and re-meshes the whole
                model, which is what a change of quality needs — a request for a **coarser**
                mesh than the one already computed is otherwise ignored, because OCCT will
                not lower the quality of a triangulation it already has.
            progress: Called with the fraction done — a float in ``[0, 1]``, strictly
                increasing — while the operation runs. ``None`` reports nothing.
            cancel: Called with no arguments; return ``True`` to stop the operation.
                It then raises :class:`PysmeshCancelled`, and the session is left
                exactly as it was.

        Returns:
            The render mesh, and the set of faces whose contribution to it changed.

        Raises:
            PysmeshError: On a deflection below 1e-7 or an ``angle_deg`` outside (0, 180),
                or if a sub-shape of the root carries no entity id.
        """
        if not (0.0 < angle_deg < 180.0):
            raise PysmeshError(
                f"Session.tessellate: angle_deg must be in (0, 180) (got {angle_deg})."
            )
        raw = self._s.tessellate(
            deflection,
            math.radians(angle_deg),
            relative,
            parallel,
            incremental,
            progress,
            cancel,
        )
        return RenderMesh(
            nodes=cast("NDArray[np.float64]", raw["nodes"]),
            normals=cast("NDArray[np.float64]", raw["normals"]),
            tris=cast("NDArray[np.int32]", raw["tris"]),
            tri_face_id=cast("NDArray[np.int64]", raw["tri_face_id"]),
            edge_lines=cast("NDArray[np.int32]", raw["edge_lines"]),
            edge_id=cast("NDArray[np.int64]", raw["edge_id"]),
            vertex_xyz=cast("NDArray[np.float64]", raw["vertex_xyz"]),
            vertex_id=cast("NDArray[np.int64]", raw["vertex_id"]),
            face_id=cast("NDArray[np.int64]", raw["face_id"]),
            face_node_range=cast("NDArray[np.int32]", raw["face_node_range"]),
            face_tri_range=cast("NDArray[np.int32]", raw["face_tri_range"]),
            retriangulated=cast("NDArray[np.int64]", raw["retriangulated"]),
            changed=cast("NDArray[np.int64]", raw["changed"]),
        )
