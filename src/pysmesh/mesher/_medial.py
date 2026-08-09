"""pySMESH mesher — the medial axis of a face.

Part of the :mod:`pysmesh.mesher` package.

The **medial axis** of a 2-D region is the set of centres of the maximal circles that fit
inside it — its skeleton. It answers two questions a mesh-preparation workflow keeps asking
and has no other exact way to answer: *where is the centreline of this thin region*, and
*how thick is it here*. SMESH computes it over a Voronoi diagram of the boundary
discretisation, so the answer is a construction rather than a sampled approximation.

Three properties of the result decide how to read it, and none is what a first reading would
assume:

* **A branch is not a dense polyline.** It carries one point per medial-axis edge plus one,
  so a straight branch is exactly two points. The axis of a rectangle is a spine plus four
  corner arms — five branches — not one line.
* **Branch 0 is not the spine.** Branches come out in construction order, so a thickness
  query has to pick the branch it means. :attr:`MedialAxis.longest` is the usual pick.
* **Width comes from the boundary, not from the axis.** Each sample carries the two nearest
  boundary points and their distance, which is the local width where the two boundaries face
  one another — which is what a thin region is.

SMESH's own constrained-Delaunay class is **not** exposed beside it, and the reason is a
measurement rather than a choice: under the pinned OCCT its triangulation comes back with
every triangle marked deleted, so none of its queries can answer. See the module comment in
the native unit for the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .._core import Shape
from .._core import medial_axis as _medial_axis


class BranchEnd(IntEnum):
    """What one end of a medial-axis branch is.

    The integer values are SMESH's own; do not reorder.

    Attributes:
        UNDEF: Not classified.
        ON_VERTEX: The branch runs into a convex corner of the boundary.
        BRANCH_POINT: Three or more branches meet here. A region with a genuine junction —
            an L, a T, a cross — has at least one.
        END: The branch stops at a point equidistant from several boundary segments, with
            nothing beyond it.
    """

    UNDEF = 0
    ON_VERTEX = 1
    BRANCH_POINT = 2
    END = 3


@dataclass(frozen=True)
class MedialBranch:
    """One branch of a medial axis, and the boundary it is equidistant from.

    Attributes:
        uv: (K, 2) float64 — the branch's own points, in the face's parameter space.
        points: (K, 3) float64 — the same points in model space.
        end_types: What each of the branch's two ends is.
        parameters: (S,) float64 — the sampled positions along the branch, from 0 to 1.
        boundary1: (S, 3) float64 — one of the two nearest boundary points at each sample.
        boundary2: (S, 3) float64 — the other.
        boundary1_edge: (S,) int64 — the shape's own EDGE ordinal that ``boundary1`` lies
            on, so a sample can be traced back to the geometry.
        boundary2_edge: (S,) int64 — the same for ``boundary2``.
        widths: (S,) float64 — the distance between the two boundary points. Where the two
            boundaries face one another, this is the local width of the region.
    """

    uv: NDArray[np.float64]
    points: NDArray[np.float64]
    end_types: tuple[BranchEnd, BranchEnd]
    parameters: NDArray[np.float64]
    boundary1: NDArray[np.float64]
    boundary2: NDArray[np.float64]
    boundary1_edge: NDArray[np.int64]
    boundary2_edge: NDArray[np.int64]
    widths: NDArray[np.float64]

    @property
    def length(self) -> float:
        """The branch's own length in model space, summed along its points."""
        if self.points.shape[0] < 2:
            return 0.0
        steps = np.diff(self.points, axis=0)
        return float(np.linalg.norm(steps, axis=1).sum())

    def has_end(self, end: BranchEnd) -> bool:
        """Whether either end of the branch is of the given type.

        Args:
            end: The end type to look for.

        Returns:
            True if one or both ends match.
        """
        return end in self.end_types


@dataclass(frozen=True)
class MedialAxis:
    """The medial axis of one face.

    Attributes:
        face: The 1-based FACE ordinal the axis was built on.
        branches: One entry per branch, in the order SMESH built them.
        branch_points: How many points three or more branches meet at. A rectangle has none;
            an L-shaped region has one.
        boundary_edges: How many edges the axis discretised the face's boundary into. It
            equals the face's own edge count.
    """

    face: int
    branches: tuple[MedialBranch, ...]
    branch_points: int
    boundary_edges: int

    @property
    def longest(self) -> MedialBranch:
        """The longest branch, which for a thin region is its spine.

        Returns:
            The branch of greatest length.

        Raises:
            ValueError: If the axis has no branches.
        """
        if not self.branches:
            raise ValueError("this medial axis has no branches.")
        return max(self.branches, key=lambda branch: branch.length)


def _branch(raw: dict[str, object]) -> MedialBranch:
    """Build a :class:`MedialBranch` from the native entry."""
    ends = cast("tuple[int, int]", raw["end_types"])
    return MedialBranch(
        uv=cast("NDArray[np.float64]", raw["uv"]),
        points=cast("NDArray[np.float64]", raw["points"]),
        end_types=(BranchEnd(ends[0]), BranchEnd(ends[1])),
        parameters=cast("NDArray[np.float64]", raw["parameters"]),
        boundary1=cast("NDArray[np.float64]", raw["boundary1"]),
        boundary2=cast("NDArray[np.float64]", raw["boundary2"]),
        boundary1_edge=cast("NDArray[np.int64]", raw["boundary1_edge"]),
        boundary2_edge=cast("NDArray[np.int64]", raw["boundary2_edge"]),
        widths=cast("NDArray[np.float64]", raw["widths"]),
    )


def medial_axis(
    shape: Shape,
    face: int,
    min_segment_length: float,
    ignore_corners: bool = False,
    samples: int = 21,
) -> MedialAxis:
    """Compute the medial axis of one face of a shape.

    This reads the geometry, not a mesh, so it needs no mesher.

    Args:
        shape: The shape the face belongs to.
        face: The 1-based FACE ordinal.
        min_segment_length: How finely the face's boundary is discretised before the axis is
            built. Smaller gives a more detailed axis and costs more; it should be well below
            the smallest feature that matters.
        ignore_corners: Drop the arms that run into the boundary's convex corners, leaving
            only the axis proper. A rectangle's axis goes from five branches to one.
        samples: How many positions along each branch to measure the boundary at. The
            samples are evenly spaced in the branch's own parameter, including both ends.

    Returns:
        The axis, its branches and their local widths.

    Raises:
        PysmeshError: If the ordinal names no face, if ``min_segment_length`` is not
            positive, or if ``samples`` is below 2.
    """
    raw = _medial_axis(shape, face, min_segment_length, ignore_corners, samples)
    branches = cast("list[dict[str, object]]", raw["branches"])
    return MedialAxis(
        face=cast("int", raw["face"]),
        branches=tuple(_branch(entry) for entry in branches),
        branch_points=cast("int", raw["branch_points"]),
        boundary_edges=cast("int", raw["boundary_edges"]),
    )
