"""Volume and surface meshing: SMESH's algorithms, hypotheses and assignment model.

This is the meshing side of the CAD/mesher boundary. A :class:`~pysmesh.Session` owns a live
shape and hands one BREP over at the handoff; a :class:`Mesher` takes that shape back as a
plain :class:`~pysmesh.Shape` — positional ordinals, which is what the stateless API already
speaks — and turns it into a mesh.

The point of the package is not any single algorithm. It is SMESH's **assignment model**: an
algorithm plus its hypotheses are attached to a *sub-shape*, and each part of the model is
meshed by whichever assignment governs it. That is what lets one mesh be structured through
an extruded region, body-fitted in the interior, projected onto a periodic face and layered
at the walls — which a single global algorithm cannot express at all::

    mesher = Mesher(load_brep(data))
    mesher.assign(Regular1D())                                  # the whole shape
    mesher.assign(NumberOfSegments(count=4))
    mesher.assign(Quadrangle2D())
    mesher.assign(Hexa3D(), on=SubShape(SubShapeKind.SOLID, 1))  # this solid only
    report = mesher.compute()
    mesh = mesher.mesh()

**Three id spaces meet here and must not be confused.** A mesh entity carries its own mesh
id; the sub-shape it sits on is named by a *positional ordinal* of the meshed shape; and the
session that produced that shape names the same sub-shape by an :data:`~pysmesh.EntityId`.
The middle one is the pivot: pairing :attr:`MeshData.element_ordinal` with the handoff's
per-kind id array carries a mesh cell all the way back to the model it came from.

Thread contract: a :class:`Mesher` is **not** thread-safe. Use one per thread. Meshing
releases the GIL, and the progress and cancel hooks are called from a helper thread.

**Layout.**

===================  =====================================================================
``_types``           the element and sub-shape enums, and what a compute and harvest return
``_catalog``         the algorithm and hypothesis dataclasses
``_gmf``             Inria ``.mesh`` / ``.meshb`` interchange
===================  =====================================================================
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import cast

from .._core import Mesher as _Mesher
from .._core import Shape
from ._catalog import (
    Adaptive1D,
    Algorithm,
    Arithmetic1D,
    AutomaticLength,
    Cartesian3D,
    CartesianParameters3D,
    CompositeHexa3D,
    CompositeSegment1D,
    Deflection1D,
    Distribution,
    FixedPoints1D,
    Geometric1D,
    Hexa3D,
    HexaFromSkin3D,
    Hypothesis,
    LayerDistribution,
    LocalLength,
    MaxElementArea,
    MaxElementVolume,
    MaxLength,
    Mefisto2D,
    NumberOfLayers,
    NumberOfLayers2D,
    NumberOfSegments,
    PolygonPerFace2D,
    PolyhedronPerSolid3D,
    Prism3D,
    Projection1D,
    Projection1D2D,
    Projection2D,
    Projection3D,
    ProjectionSource1D,
    ProjectionSource2D,
    ProjectionSource3D,
    Propagation,
    QuadFromMedialAxis1D2D,
    Quadrangle2D,
    QuadrangleParams,
    QuadranglePreference,
    QuadraticMesh,
    QuadType,
    RadialPrism3D,
    RadialQuadrangle1D2D,
    Regular1D,
    SegmentLengthAroundVertex,
    StartEndLength,
    ViscousLayers,
    ViscousLayers2D,
)
from ._gmf import GmfMesh, gmf_unwritable_types, gmf_writable_group_name, read_gmf, write_gmf
# The hook aliases belong to the session, which defines them; the mesher shares them
# rather than declaring a second pair that means the same thing.
from ..session import CancelPredicate, ProgressCallback
from ._types import (
    GMF_REQUIRED_MARKER,
    GMF_WRITABLE_TYPES,
    ComputeReport,
    ElementDimension,
    ElementType,
    MeshData,
    MeshGroup,
    SubMeshCount,
    SubShape,
    SubShapeKind,
    _groups,
    _mesh_data,
    _report,
)


class Mesher:
    """A shape, the algorithms and hypotheses assigned across it, and the mesh they produce.

    A mesher is built on one :class:`~pysmesh.Shape` and holds a single mesh. Assignments
    accumulate; :meth:`compute` runs them all and :meth:`mesh` reads the result back as
    arrays.

    Assignment is scoped. An assignment with no ``on`` governs the whole shape and is the
    default everywhere; one naming a sub-shape overrides it there. That is the whole of the
    model, and it is what makes a mixed mesh expressible.

    Thread contract: **not thread-safe**. One mesher per thread. :meth:`compute` releases the
    GIL and calls its hooks from a helper thread.

    Example:
        >>> m = Mesher(load_brep(data))                     # doctest: +SKIP
        >>> m.assign(Regular1D())                           # doctest: +SKIP
        >>> m.assign(NumberOfSegments(count=4))             # doctest: +SKIP
        >>> m.assign(Quadrangle2D())                        # doctest: +SKIP
        >>> m.assign(Hexa3D())                              # doctest: +SKIP
        >>> report = m.compute()                            # doctest: +SKIP
    """

    __slots__ = ("_m",)

    _m: _Mesher

    def __init__(self, shape: Shape) -> None:
        """Build a mesher on a shape.

        Args:
            shape: The shape to mesh, from :func:`~pysmesh.load_brep` or another loader.
        """
        self._m = _Mesher(shape)

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

    def groups(self) -> tuple[MeshGroup, ...]:
        """The groups carried on the mesh.

        Algorithms create these themselves where they have something to name — a viscous
        layer stack collects its cells into the group its hypothesis names.

        Returns:
            One entry per group.
        """
        return _groups(cast("Sequence[object]", self._m.groups()))

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

    # ---- Lifetime --------------------------------------------------------------------- #

    def release(self) -> None:
        """Free the underlying mesh. Idempotent; the mesher is unusable afterwards."""
        self._m.release()

    def is_open(self) -> bool:
        """Whether the mesher still holds a mesh.

        Returns:
            False once :meth:`release` has run.
        """
        return bool(self._m.is_open())

    def __enter__(self) -> Mesher:
        """Enter a context that releases the mesh on exit.

        Returns:
            This mesher.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the mesh."""
        self.release()


__all__ = [
    "GMF_REQUIRED_MARKER",
    "GMF_WRITABLE_TYPES",
    "Adaptive1D",
    "Algorithm",
    "Arithmetic1D",
    "AutomaticLength",
    "CancelPredicate",
    "Cartesian3D",
    "CartesianParameters3D",
    "CompositeHexa3D",
    "CompositeSegment1D",
    "ComputeReport",
    "Deflection1D",
    "Distribution",
    "ElementDimension",
    "ElementType",
    "FixedPoints1D",
    "Geometric1D",
    "GmfMesh",
    "Hexa3D",
    "HexaFromSkin3D",
    "Hypothesis",
    "LayerDistribution",
    "LocalLength",
    "MaxElementArea",
    "MaxElementVolume",
    "MaxLength",
    "Mefisto2D",
    "MeshData",
    "MeshGroup",
    "Mesher",
    "NumberOfLayers",
    "NumberOfLayers2D",
    "NumberOfSegments",
    "PolygonPerFace2D",
    "PolyhedronPerSolid3D",
    "Prism3D",
    "ProgressCallback",
    "Projection1D",
    "Projection1D2D",
    "Projection2D",
    "Projection3D",
    "ProjectionSource1D",
    "ProjectionSource2D",
    "ProjectionSource3D",
    "Propagation",
    "QuadFromMedialAxis1D2D",
    "QuadType",
    "QuadrangleParams",
    "QuadranglePreference",
    "Quadrangle2D",
    "QuadraticMesh",
    "RadialPrism3D",
    "RadialQuadrangle1D2D",
    "Regular1D",
    "SegmentLengthAroundVertex",
    "StartEndLength",
    "SubMeshCount",
    "SubShape",
    "SubShapeKind",
    "ViscousLayers",
    "ViscousLayers2D",
    "gmf_unwritable_types",
    "gmf_writable_group_name",
    "read_gmf",
    "write_gmf",
]
