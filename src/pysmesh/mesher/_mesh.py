"""pySMESH mesher — the assignment model, the compute, and the mesh out.

Part of the :mod:`pysmesh.mesher` package. This is the core of the meshing surface: an
algorithm plus its hypotheses attach to a sub-shape, :meth:`_MeshOps.compute` runs whatever
governs each part of the model, and :meth:`_MeshOps.mesh` reads the result back as arrays.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import cast

from ..session import CancelPredicate, ProgressCallback
from ._base import _MesherBase
from ._catalog import Algorithm, Hypothesis
from ._gmf import write_gmf
from ._types import (
    ComputeReport,
    MeshData,
    MeshGroup,
    SubShape,
    SubShapeKind,
    _mesh_data,
    _report,
)


class _MeshOps(_MesherBase):
    """Assignment, compute, and the harvest."""

    __slots__ = ()

    # ---- Assignment ------------------------------------------------------------------- #

    def assign(self, item: Algorithm | Hypothesis, on: SubShape | None = None) -> None:
        """Attach an algorithm or a hypothesis to a sub-shape.

        Args:
            item: The algorithm or hypothesis to attach.
            on: The sub-shape it governs. ``None`` means the whole shape, which is how a
                default for the model is expressed.

        Raises:
            PysmeshError: If SMESH refuses the assignment. The message names the sub-shape
                and why — a second algorithm of the same dimension already there, a
                hypothesis that does not fit the algorithm beside it, a sub-shape whose
                geometry the algorithm cannot read, and so on.
        """
        kind = "" if on is None else on.kind.name
        ordinal = 0 if on is None else on.ordinal
        self._m.assign(item.native_name, item.params(), kind, ordinal)

    def unassign(self, item: Algorithm | Hypothesis, on: SubShape | None = None) -> None:
        """Detach an algorithm or a hypothesis previously attached to a sub-shape.

        Args:
            item: The algorithm or hypothesis to detach. Only its name is used, so an
                equivalent instance works.
            on: The sub-shape it was attached to.

        Raises:
            PysmeshError: If nothing of that name is attached there, or SMESH refuses to
                detach it.
        """
        kind = "" if on is None else on.kind.name
        ordinal = 0 if on is None else on.ordinal
        self._m.unassign(item.native_name, kind, ordinal)

    def assignments(self) -> tuple[tuple[str, SubShape | None], ...]:
        """Everything attached, in the order it was attached.

        Returns:
            One ``(native_name, sub_shape)`` pair per assignment, with ``None`` for the ones
            governing the whole shape.
        """
        out: list[tuple[str, SubShape | None]] = []
        for entry in self._m.assignments():
            name, kind, ordinal = cast("tuple[str, str, int]", entry)
            on = None if not kind else SubShape(SubShapeKind[kind], ordinal)
            out.append((name, on))
        return tuple(out)

    # ---- Compute ---------------------------------------------------------------------- #

    def compute(
        self,
        progress: ProgressCallback | None = None,
        cancel: CancelPredicate | None = None,
    ) -> ComputeReport:
        """Run the assigned algorithms over the shape.

        Progress and cancellation both work, with two limits that are properties of SMESH
        rather than of this binding and are worth knowing before relying on either:

        * **A cancel is not preemptive.** Three algorithms poll it inside their own loop —
          :class:`Cartesian3D`, :class:`Prism3D` and the one driven by :class:`Adaptive1D` —
          and every other one can be stopped only between sub-meshes. So the latency is
          bounded by the longest single algorithm run, not by any poll interval.
        * **Progress is exact only at sub-mesh granularity.** The fraction of the sub-meshes
          already done is real. Within one running algorithm SMESH interpolates with a tick
          counter that advances once per enquiry, so the value there tracks the enquiry, not
          the work. Only :class:`QuadFromMedialAxis1D2D` reports its own true fraction. The
          practical shape of this: an algorithm that meshes the whole model in one call — a
          body-fitted Cartesian run, say — reports values that creep up from near zero and
          then jump to 1.0 at the end. The updates are real and monotone; the *number* they
          carry is not a fraction of the work done.

        Args:
            progress: Called with a float in ``[0, 1]``, strictly increasing, from a helper
                thread while the mesher runs. An exception raised in it cancels the compute
                and is re-raised here with its own type and traceback.
            cancel: Asked whether to stop. It is asked once before anything starts, so a
                flag set beforehand is honoured even by a mesh that finishes quickly.

        Returns:
            What the run produced, and which sub-shapes received elements.

        Raises:
            PysmeshCancelled: If ``cancel`` returned True or ``progress`` raised. The mesh is
                cleared, so nothing partial survives.
            PysmeshError: If any sub-mesh failed. The message names every failed sub-shape
                with SMESH's own reason and the algorithm that reported it, and ``.face_ids``
                carries the ordinals of the failed faces. The partial mesh is **kept** here
                rather than cleared, because how far the assignment got is the diagnostic.
        """
        return _report(self._m.compute(progress, cancel))

    # ---- Results ---------------------------------------------------------------------- #

    def mesh(self) -> MeshData:
        """The mesh as arrays.

        Reading is always allowed. It makes no claim that the mesh is complete —
        :meth:`compute` is the only thing that reports success — so after a caught failure
        this returns however much was built.

        Returns:
            The nodes, the elements, and the sub-shape each of them sits on.
        """
        return _mesh_data(self._m.mesh_arrays())

    def write_gmf(
        self, path: str | os.PathLike[str], groups: Sequence[MeshGroup] = ()
    ) -> None:
        """Write this mesher's mesh to an Inria ``.mesh`` or ``.meshb`` file.

        Args:
            path: File to write.
            groups: Groups to write; see :func:`~pysmesh.write_gmf` for what the format can
                carry.

        Raises:
            PysmeshError: If the mesh holds an element the format cannot represent — a
                body-fitted Cartesian mesh always does — or the file cannot be written.
        """
        write_gmf(path, self.mesh(), groups)
