# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-21

"""pySMESH session — GEOM's healing operators (ShHealOper).

Part of the :mod:`pysmesh.session` package. The session's operations are declared on one
class and implemented per area, the same way the native `Session` is one class implemented
across per-area translation units; see the package docstring for the whole surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from .._core import PysmeshError
from ._base import _SessionBase
from ._types import EntityId, HistoryDelta, _delta, _ids


def _fractions(where: str, values: Sequence[float]) -> list[float]:
    """A caller's fractions as floats; refused when one is not a real number."""
    out: list[float] = []
    for v in values:
        if not isinstance(v, (int, float, np.integer, np.floating)) or isinstance(
            v, bool
        ):
            raise PysmeshError(f"{where}: the position {v!r} is not a number.")
        out.append(float(v))
    return out


class _GeomHealOps(_SessionBase):
    """GEOM's healing operators."""

    __slots__ = ()

    def divide_edge(
        self,
        edge: EntityId,
        positions: Sequence[float],
        *,
        by: Literal["length", "parameter"] = "length",
    ) -> HistoryDelta:
        """Divide an edge at fractions of its length or of its parameter range.

        GEOM's ``DivideEdge``: ``ShHealOper_EdgeDivide`` splits the edge's curve, and the
        curve on each face that holds it, at each position, so every face that uses the
        edge holds the pieces in its place. Each position is a fraction measured from the
        end at the start of the edge's curve, the first parameter that
        :meth:`edge_parameter_bounds` reports.

        * ``by="length"``: the fraction of the arc length. The parameter of each point is
          found by OCCT's ``GCPnts_AbscissaPoint`` at 1e-10 of the length, as
          :meth:`point_at_length` finds it. GEOM's own length mode integrates with a
          fixed Gauss rule, which is exact on lines and circles only.
        * ``by="parameter"``: the fraction of the parameter range, as GEOM's
          ``isByParameter``. On a line and a circle the two modes agree.

        **Identity.** The divided edge's id dies. Each piece gets a new id whose role is
        ``GENERATED`` and whose origin is the divided edge, and each new vertex gets a new
        id generated from it too. The faces that hold the edge keep their ids (they are in
        ``modified``), as do every other entity of the body.

        Args:
            edge: An edge entity id that denotes one edge, not degenerate.
            positions: Strictly increasing fractions, each strictly between 0 and 1. At
                least one.
            by: ``"length"`` or ``"parameter"``.

        Returns:
            The delta: the edge in ``deleted``, the pieces and the new vertices in
            ``created``.

        Raises:
            PysmeshError: If the id is dead, is not an edge or was split; if the edge is
                degenerate; if ``positions`` is empty, holds a value that is not in
                ``(0, 1)``, or is not strictly increasing (the message gives the values);
                if ``by`` is not ``"length"`` or ``"parameter"``; if the divider fails or
                gives a number of pieces other than ``len(positions) + 1``; or if the
                result is invalid. Nothing is committed then.
        """
        values = _fractions("Session.divide_edge", positions)
        return _delta(self._s.divide_edge(int(edge), values, by))

    def close_contour(
        self, edges: Sequence[EntityId], *, common_vertex: bool = False
    ) -> HistoryDelta:
        """Close the gap between the two ends of an open wire.

        GEOM's ``CloseContour``: ``ShHealOper_CloseContour`` on the body that owns the
        edges, with GEOM's call ``Perform(wire, isCommonVertex, not isCommonVertex)``.
        The edges must be exactly the edges of one open wire: a wire of a face of the body
        (an open contour, which makes the face invalid) or a free wire.

        **common_vertex.** GEOM: "If True: closure by creation of a common vertex; if
        False: closure by creation of an edge between ends."

        * ``common_vertex=False``: a gap wider than 1 model unit is closed by a new
          straight edge between the two ends: a line in the face's parameter space when
          the wire is a face's, a 3-D line for a free wire. A narrower gap is closed by
          ``ShapeFix_Wire``, which moves the ends of the curves together
          (``FixGaps3d``, ``FixGaps2d``) and merges the two end vertices.
        * ``common_vertex=True``: no edge is added. The two end vertices become one,
          whose tolerance grows to cover the gap (GEOM's documentation: "its tolerance is
          increased to a value of the gap between the ends of the contour").

        The threshold of 1 model unit is ShHealOper's own maximum tolerance
        (``ShHealOper_CloseContour::Init``): an absolute length, which GEOM does not
        expose.

        **Identity.** A new closing edge gets a new id whose role is ``GENERATED`` and
        whose origin is the edges whose ends it joins. ShHealOper reports the closing edge
        in a face as a modification of the edge before the gap; the session takes it out
        of that edge's history, so that edge keeps its id alone. The wire's edges and the
        face keep their ids.

        Like :meth:`heal`, the operation commits what OCCT built whatever
        ``BRepCheck_Analyzer`` says of it: its input is broken by assumption.
        :attr:`HistoryDelta.valid` carries the verdict.

        Args:
            edges: The edges of one open wire, in any order. At least one.
            common_vertex: Close by one common vertex instead of a new edge.

        Returns:
            The delta, with the verdict on ``valid``.

        Raises:
            PysmeshError: If an id is dead, is not an edge or was split; if the edges
                belong to two bodies or are not exactly the edges of one wire of their
                body; if the wire is closed already; or if ShHealOper closes nothing.
        """
        return _delta(self._s.close_contour(_ids(edges), common_vertex))

    def fill_holes(
        self,
        entities: Sequence[EntityId],
        *,
        edges: Sequence[EntityId] | None = None,
    ) -> HistoryDelta:
        """Fill the holes of shells and faces with new faces sewn in along their edges.

        GEOM's ``FillHoles``: ``ShHealOper_FillHoles`` with GEOM's default parameters. It
        finds the free boundaries of the bodies that own ``entities`` (the loops of edges
        that one face holds), builds a ``GeomPlate`` surface through each loop (degree 3,
        5 points on each curve, 12 iterations, 3-D tolerance 1e-4, approximated to degree
        8 with 9 segments at most) and adds a face on it bounded by the loop's own edges.
        The face goes into the shell that holds the loop, which becomes one shell with the
        new face. A loop of one edge is filled only when the edge is a closed curve.

        **Which holes.** With ``edges=None`` every free boundary is filled, and a free face
        has its outer boundary filled too, since that boundary is free as well. With
        ``edges``, only the loops those edges form are filled.

        **Identity.** Each new face gets a new id, generated from the edges of its hole.
        The bodies keep every id: ShHealOper writes the new face's curves into the boundary
        edges, so it works on a copy of the bodies, and every id of them is carried onto
        the copy (they are all in ``modified``).

        **fill_holes, make_face and make_filling.** :meth:`make_face` caps a planar loop
        with an exact plane, and :meth:`make_filling` a non-planar loop with a filling
        surface. Each makes a separate free face on copies of the loop's edges
        (:meth:`extract_edges`), so the body stays open and a :meth:`sew` is needed to
        join the cap. fill_holes builds the face on the loop's own edges and puts it in the
        body's shell, so the shell is closed in one step; its surface is a ``GeomPlate``
        approximation within 1e-4 even for a planar hole. Use make_face where the cap must
        be an exact plane, and fill_holes to close a shell.

        Args:
            entities: Entities whose owning bodies are filled. At least one.
            edges: Free edges of those bodies that bound the holes to fill. ``None`` fills
                every free boundary.

        Returns:
            The delta: the new faces in ``created``.

        Raises:
            PysmeshError: If no entity is named, an id is dead, or the bodies share
                sub-shapes with bodies left out; if a named edge is not an edge of those
                bodies, or is held by more than one face; if there is no hole bounded by
                free edges, or GeomPlate builds no surface (nothing is committed then); or
                if the result is invalid.
        """
        edge_ids = None if edges is None else _ids(edges)
        return _delta(self._s.fill_holes(_ids(entities), edge_ids))

    def suppress_faces(self, faces: Sequence[EntityId]) -> HistoryDelta:
        """Take faces out of their shells and leave the hole open.

        GEOM's ``SuppressFaces``: ``ShHealOper_RemoveFace::Perform(faces)``. What happens,
        read at the source (``ShHealOper_RemoveFace.cxx``) and measured on OCCT 8.0.1:

        * Each named face is removed from its shell. Nothing is extended or built to close
          the hole: the neighbouring faces keep their surfaces and their boundaries. Their
          edges along the hole only lose the curve they had on the removed face's surface
          (they are copied without it, and keep their ids).
        * A shell with two or more faces left stays a shell, now open. A shell with one
          face left becomes that face. A shell with none left disappears.
        * A solid is built again from its shells (``ShapeFix_Solid``). When the removal
          opens its shell, the result is that open shell, not a solid: the solid's id dies,
          and the body becomes a shell body.

        **suppress_faces and defeature.** :meth:`defeature` removes a feature and closes
        the gap: it extends the neighbouring faces until they meet, so a solid stays a
        closed solid (a filleted box becomes the sharp box, a plate with a hole the plain
        plate). suppress_faces closes nothing: the neighbours stay as they are, and a solid
        becomes an open shell with a hole where the faces were. Use defeature to simplify a
        solid; use suppress_faces to cut faces out of a shell, for example before
        :meth:`fill_holes` or a :meth:`sew` with other faces.

        As in :meth:`heal`, the result is committed whatever ``BRepCheck_Analyzer`` says of
        it, and :attr:`HistoryDelta.valid` carries the verdict. An open shell is valid.

        Args:
            faces: Face entity ids, each denoting one face. At least one. They may belong
                to several bodies.

        Returns:
            The delta: the named faces in ``deleted``, with every edge and vertex that only
            they held, and the id of each solid that became a shell.

        Raises:
            PysmeshError: If no face is named; if an id is dead, is not a face or was
                split; if the bodies share sub-shapes with bodies left out; or if
                ShHealOper removes none of the faces or leaves one of them in the result
                (``.face_ids`` names those). Nothing is committed then.
        """
        return _delta(self._s.suppress_faces(_ids(faces)))
