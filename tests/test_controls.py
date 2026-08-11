# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-08-09

"""Gates for the mesh quality controls: the measures, the predicates and the filter algebra.

Three claims are under test.

* **Every measure returns the number its definition says it should**, checked on a
  deliberately irregular element rather than a regular one — a regular element cannot tell a
  correct implementation from a constant. Each value is computed here from the published
  definition (SMESH follows Frey and George, *Maillages, applications aux elements finis*,
  Hermes Science 1999; the tetrahedral aspect ratio follows Verdict, as VTK implements it) on
  coordinates chosen so the answer is unambiguous. Where a measure carries a normalisation,
  the regular element is asserted **as well**, so the pair pins both the shape of the formula
  and its constant.
* **Every predicate is checked against a fixture built to trigger it and one built not to.**
  A predicate that answers yes to everything and one that answers no to everything both pass
  a one-sided test, so neither side is optional.
* **The results are NumPy arrays keyed by mesh id**, which is what makes them joinable with a
  harvest.

Fixtures are hand-built meshes with exact coordinates, so the expected value is arithmetic
rather than a recorded output. Fixture sizing follows the project rule: 3, 7 and 11, never a
unit cube.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    And,
    Area,
    AspectRatio,
    AspectRatio3D,
    BadOrientedVolume,
    BareBorderFace,
    BareBorderVolume,
    BelongToGroup,
    CoincidentElements,
    CoincidentNodes,
    Deflection2D,
    ElementDimension,
    ElementsOnShape,
    ElementType,
    EqualTo,
    FreeBorders,
    FreeEdges,
    FreeFaces,
    FreeNodes,
    Hexa3D,
    Length,
    Length2D,
    Length3D,
    LessThan,
    ManifoldPart,
    MaxElementArea,
    MaxElementLength2D,
    MaxElementLength3D,
    Mefisto2D,
    MeshData,
    Mesher,
    MinimumAngle,
    MoreThan,
    MultiConnection,
    MultiConnection2D,
    NodeConnectivityNumber,
    Not,
    NumberOfSegments,
    Or,
    OverConstrainedFace,
    OverConstrainedVolume,
    PysmeshError,
    Quadrangle2D,
    RangeOfIds,
    Regular1D,
    Session,
    Skew,
    SubShape,
    SubShapeKind,
    Taper,
    Volume,
    Warping,
    quality,
    select,
)

# The three edge vectors of the sheared cell every 3-D gate uses. Deliberately not axis
# aligned and deliberately not equal, so no measure can be right by accident.
EDGE_A: NDArray[np.float64] = np.array([3.0, 0.0, 0.0])
EDGE_B: NDArray[np.float64] = np.array([1.0, 7.0, 0.0])
EDGE_C: NDArray[np.float64] = np.array([2.0, 1.0, 11.0])


# ---- Fixture construction --------------------------------------------------------------- #


def _mesh(
    coords: Sequence[Sequence[float]],
    elements: Sequence[tuple[ElementType, tuple[int, ...]]],
) -> MeshData:
    """Build a mesh from coordinates and connectivity, with ids 1..n in the given order."""
    points = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    offsets: list[int] = [0]
    connectivity: list[int] = []
    types: list[int] = []
    for kind, nodes in elements:
        connectivity.extend(nodes)
        offsets.append(len(connectivity))
        types.append(int(kind))
    n_nodes = points.shape[0]
    n_elements = len(elements)
    return MeshData(
        node_coords=points,
        node_id=np.arange(1, n_nodes + 1, dtype=np.int64),
        node_kind=np.zeros(n_nodes, dtype=np.int8),
        node_ordinal=np.zeros(n_nodes, dtype=np.int32),
        element_offsets=np.array(offsets, dtype=np.int64),
        element_nodes=np.array(connectivity, dtype=np.int32),
        element_type=np.array(types, dtype=np.int8),
        element_id=np.arange(1, n_elements + 1, dtype=np.int64),
        element_kind=np.zeros(n_elements, dtype=np.int8),
        element_ordinal=np.zeros(n_elements, dtype=np.int32),
        face_offsets=np.zeros(n_elements + 1, dtype=np.int64),
        face_sizes=np.zeros(0, dtype=np.int32),
    )


def _sheared_hexa_nodes() -> list[NDArray[np.float64]]:
    """The eight corners of the sheared parallelepiped, in SMESH's own node order.

    SMESH takes a hexahedron's first quad as the one whose winding normal points **out** of
    the cell, so the face spanned by A and B is listed after the one offset along C. Built the
    other way round the cell reads as inverted, which is what the orientation gate uses.
    """
    origin = np.zeros(3, dtype=np.float64)
    lower = [origin, origin + EDGE_A, origin + EDGE_A + EDGE_B, origin + EDGE_B]
    upper = [point + EDGE_C for point in lower]
    return [*upper, *lower]


def _sheared_hexa() -> MeshData:
    """One sheared hexahedron of known volume, with no faces or edges beside it."""
    return _mesh(_sheared_hexa_nodes(), [(ElementType.HEXAHEDRON, tuple(range(8)))])


def _skew_triangle_nodes() -> list[NDArray[np.float64]]:
    """A triangle whose three sides and three angles are all different."""
    return [
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([1.0, 3.0, 0.0]),
    ]


def _skew_quad_nodes() -> list[NDArray[np.float64]]:
    """A planar parallelogram sheared off the axes: skewed, but not warped or tapered."""
    return [
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([5.0, 3.0, 0.0]),
        np.array([1.0, 3.0, 0.0]),
    ]


def _closed_box_shell() -> MeshData:
    """The six faces of the sheared cell as a closed shell, with the cell itself."""
    nodes = _sheared_hexa_nodes()
    faces: list[tuple[ElementType, tuple[int, ...]]] = [
        (ElementType.HEXAHEDRON, tuple(range(8))),
        (ElementType.QUADRANGLE, (0, 1, 2, 3)),
        (ElementType.QUADRANGLE, (4, 5, 6, 7)),
        (ElementType.QUADRANGLE, (0, 1, 5, 4)),
        (ElementType.QUADRANGLE, (1, 2, 6, 5)),
        (ElementType.QUADRANGLE, (2, 3, 7, 6)),
        (ElementType.QUADRANGLE, (3, 0, 4, 7)),
    ]
    return _mesh(nodes, faces)


@pytest.fixture()
def box_mesher() -> Iterator[Mesher]:
    """A structured hexahedral mesh of the 3 x 7 x 11 box, released afterwards."""
    session = Session()
    session.add_box(3.0, 7.0, 11.0)
    with Mesher(ps.load_brep(session.brep())) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=3))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.compute()
        yield mesher


# ---- Independent oracles, from the published definitions --------------------------------- #


def _distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(q - p))


def _triangle_area(
    p: NDArray[np.float64], q: NDArray[np.float64], r: NDArray[np.float64]
) -> float:
    """Area of the triangle three points span."""
    return 0.5 * float(np.linalg.norm(np.cross(q - p, r - p)))


def _triangle_aspect_ratio(points: Sequence[NDArray[np.float64]]) -> float:
    """``sqrt(3)/6 * h * p / S`` — Frey and George's normalised triangle aspect ratio."""
    sides = [
        _distance(points[0], points[1]),
        _distance(points[1], points[2]),
        _distance(points[2], points[0]),
    ]
    return (
        math.sqrt(3.0)
        / 6.0
        * max(sides)
        * (sum(sides) / 2.0)
        / _triangle_area(points[0], points[1], points[2])
    )


def _quad_aspect_ratio(points: Sequence[NDArray[np.float64]]) -> float:
    """``sqrt(1/32) * L * C1 / C2`` — the quadrangle form of the same definition."""
    sides = [_distance(points[i], points[(i + 1) % 4]) for i in range(4)]
    diagonals = [_distance(points[0], points[2]), _distance(points[1], points[3])]
    corners = [
        _triangle_area(points[0], points[1], points[2]),
        _triangle_area(points[0], points[1], points[3]),
        _triangle_area(points[0], points[2], points[3]),
        _triangle_area(points[1], points[2], points[3]),
    ]
    longest = max([*sides, *diagonals])
    root_sum = math.sqrt(sum(side * side for side in sides))
    return math.sqrt(1.0 / 32.0) * longest * root_sum / min(corners)


def _tetra_aspect_ratio(points: Sequence[NDArray[np.float64]]) -> float:
    """``L_max / (2 * sqrt(6) * r_in)`` — Verdict's tetrahedral aspect ratio, as VTK computes
    it, written as ``L_max * A_total / (6 * sqrt(6) * V)`` since ``r_in = 3V/A``.
    """
    longest = max(
        _distance(points[i], points[j]) for i in range(4) for j in range(i)
    )
    area = sum(
        _triangle_area(points[i], points[j], points[k])
        for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    )
    volume = abs(
        float(
            np.linalg.det(
                np.array(
                    [
                        points[1] - points[0],
                        points[2] - points[0],
                        points[3] - points[0],
                    ]
                )
            )
        )
    ) / 6.0
    return longest * area / (6.0 * math.sqrt(6.0) * volume)


def _warping(points: Sequence[NDArray[np.float64]]) -> float:
    """The largest of the four corner departures from planarity, in degrees.

    Each corner takes the two edge midpoints either side of it, builds the normal of the
    triangle they make with the element centroid, and reports the angle whose sine is the
    corner's offset along that normal over half the shorter of the two edges.
    """
    centroid = sum(points) / 4.0

    def corner(
        first: NDArray[np.float64],
        middle: NDArray[np.float64],
        last: NDArray[np.float64],
    ) -> float:
        half = min(_distance(first, middle), _distance(middle, last)) * 0.5
        normal = np.cross(
            (middle + first) / 2.0 - centroid, (last + middle) / 2.0 - centroid
        )
        normal = normal / np.linalg.norm(normal)
        return math.degrees(
            math.asin(abs(float(np.dot(middle - centroid, normal)) / half))
        )

    return max(
        corner(points[i], points[(i + 1) % 4], points[(i + 2) % 4]) for i in range(4)
    )


def _taper(points: Sequence[NDArray[np.float64]]) -> float:
    """The largest relative departure of a corner triangle's area from their mean."""
    corners = [
        _triangle_area(points[3], points[0], points[1]),
        _triangle_area(points[2], points[0], points[1]),
        _triangle_area(points[1], points[2], points[3]),
        _triangle_area(points[2], points[3], points[0]),
    ]
    mean = sum(corners) / 4.0
    return max(abs((corner - mean) / mean) for corner in corners)


def _skew_angle(
    p: NDArray[np.float64], q: NDArray[np.float64], r: NDArray[np.float64]
) -> float:
    """The angle, in radians, between the median from ``q`` and the midline facing it."""
    first = (p + r) / 2.0 - q
    second = (q + p) / 2.0 - (r + q) / 2.0
    cosine = float(np.dot(first, second)) / float(
        np.linalg.norm(first) * np.linalg.norm(second)
    )
    return math.acos(max(-1.0, min(1.0, cosine)))


def _quad_skew(points: Sequence[NDArray[np.float64]]) -> float:
    """The departure from a right angle, in degrees, of the two midline directions."""
    first = (points[2] + points[3]) / 2.0 - (points[0] + points[1]) / 2.0
    second = (points[1] + points[2]) / 2.0 - (points[3] + points[0]) / 2.0
    cosine = float(np.dot(first, second)) / float(
        np.linalg.norm(first) * np.linalg.norm(second)
    )
    return math.degrees(abs(math.pi / 2.0 - math.acos(max(-1.0, min(1.0, cosine)))))


def _minimum_angle(points: Sequence[NDArray[np.float64]]) -> float:
    """The smallest interior angle of a polygon, in degrees."""
    count = len(points)
    angles = []
    for i in range(count):
        before = points[(i - 1) % count] - points[i]
        after = points[(i + 1) % count] - points[i]
        cosine = float(np.dot(before, after)) / float(
            np.linalg.norm(before) * np.linalg.norm(after)
        )
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return min(angles)


def _value_of(result: ps.QualityResult, element_id: int) -> float:
    """The one value the result carries for a given mesh id."""
    row = int(np.flatnonzero(result.element_ids == element_id)[0])
    return float(result.values[row])


# ---- The measures ------------------------------------------------------------------------ #


def test_volume_of_a_sheared_hexahedron_equals_the_edge_vector_determinant() -> None:
    """A parallelepiped's volume is the determinant of its three edge vectors."""
    cell = _sheared_hexa()
    expected = abs(float(np.linalg.det(np.array([EDGE_A, EDGE_B, EDGE_C]))))

    result = quality(cell, Volume())

    assert result.count == 1
    assert result.values[0] == pytest.approx(expected, rel=1e-12)
    assert result.values[0] == pytest.approx(231.0, rel=1e-12)


def test_volume_of_an_inverted_hexahedron_is_negative() -> None:
    """The falsification: the sign is what makes this measure worth running."""
    nodes = _sheared_hexa_nodes()
    inverted = _mesh(nodes, [(ElementType.HEXAHEDRON, (4, 5, 6, 7, 0, 1, 2, 3))])

    result = quality(inverted, Volume())

    assert result.values[0] == pytest.approx(-231.0, rel=1e-12)


def test_volume_of_a_tetrahedron_equals_one_sixth_of_its_triple_product() -> None:
    """A tetrahedron on three edge vectors holds a sixth of their parallelepiped."""
    origin = np.zeros(3, dtype=np.float64)
    points = [origin, origin + EDGE_A, origin + EDGE_B, origin + EDGE_C]
    cell = _mesh(points, [(ElementType.TETRAHEDRON, (0, 2, 1, 3))])
    expected = abs(float(np.linalg.det(np.array([EDGE_A, EDGE_B, EDGE_C])))) / 6.0

    result = quality(cell, Volume())

    assert result.values[0] == pytest.approx(expected, rel=1e-12)


def test_area_of_a_skewed_quadrangle_equals_its_shoelace_area() -> None:
    """A parallelogram's area is the cross product of the two sides that span it."""
    points = _skew_quad_nodes()
    face = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])
    expected = float(np.linalg.norm(np.cross(points[1] - points[0], points[3] - points[0])))

    result = quality(face, Area())

    assert result.values[0] == pytest.approx(expected, rel=1e-12)
    assert result.values[0] == pytest.approx(12.0, rel=1e-12)


def test_length_of_an_edge_equals_the_distance_between_its_nodes() -> None:
    """A 3-4-12 offset, so the expected 13 is exact in binary."""
    edge = _mesh(
        [[1.0, 2.0, 3.0], [4.0, 6.0, 15.0]], [(ElementType.EDGE, (0, 1))]
    )

    result = quality(edge, Length())

    assert result.values[0] == pytest.approx(13.0, rel=1e-12)


def test_aspect_ratio_of_an_equilateral_triangle_is_exactly_one() -> None:
    """The normalisation anchor: without it a wrong constant would pass the skewed case."""
    points = [
        [0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        [3.5, 7.0 * math.sqrt(3.0) / 2.0, 0.0],
    ]
    face = _mesh(points, [(ElementType.TRIANGLE, (0, 1, 2))])

    result = quality(face, AspectRatio())

    assert result.values[0] == pytest.approx(1.0, rel=1e-12)


def test_aspect_ratio_of_a_skewed_triangle_matches_its_published_definition() -> None:
    """A triangle with three different sides, against ``sqrt(3)/6 * h * p / S``."""
    points = _skew_triangle_nodes()
    face = _mesh(points, [(ElementType.TRIANGLE, (0, 1, 2))])

    result = quality(face, AspectRatio())

    assert result.values[0] == pytest.approx(_triangle_aspect_ratio(points), rel=1e-12)
    assert result.values[0] > 1.0


def test_aspect_ratio_of_a_square_is_exactly_one() -> None:
    """The quadrangle branch's normalisation anchor."""
    face = _mesh(
        [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [7.0, 7.0, 0.0], [0.0, 7.0, 0.0]],
        [(ElementType.QUADRANGLE, (0, 1, 2, 3))],
    )

    result = quality(face, AspectRatio())

    assert result.values[0] == pytest.approx(1.0, rel=1e-12)


def test_aspect_ratio_of_a_skewed_quadrangle_matches_its_published_definition() -> None:
    """A sheared parallelogram, against ``sqrt(1/32) * L * C1 / C2``."""
    points = _skew_quad_nodes()
    face = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])

    result = quality(face, AspectRatio())

    assert result.values[0] == pytest.approx(_quad_aspect_ratio(points), rel=1e-12)
    assert result.values[0] > 1.0


def test_aspect_ratio_3d_of_a_regular_tetrahedron_is_one() -> None:
    """The tetrahedral normalisation anchor, on the cell the measure is normalised to."""
    half = math.sqrt(2.0) / 2.0
    points = [
        [1.0, 0.0, -half],
        [-1.0, 0.0, -half],
        [0.0, 1.0, half],
        [0.0, -1.0, half],
    ]
    cell = _mesh(points, [(ElementType.TETRAHEDRON, (0, 1, 2, 3))])

    result = quality(cell, AspectRatio3D())

    assert result.values[0] == pytest.approx(1.0, rel=1e-9)


def test_aspect_ratio_3d_of_a_skewed_tetrahedron_matches_the_verdict_ratio() -> None:
    """The tetrahedral branch goes through VTK, so it is checked against Verdict's formula."""
    origin = np.zeros(3, dtype=np.float64)
    points = [origin, origin + EDGE_A, origin + EDGE_B, origin + EDGE_C]
    cell = _mesh(points, [(ElementType.TETRAHEDRON, (0, 2, 1, 3))])

    result = quality(cell, AspectRatio3D())

    assert result.values[0] == pytest.approx(_tetra_aspect_ratio(points), rel=1e-9)
    assert result.values[0] > 2.0


def test_aspect_ratio_3d_of_a_cube_is_one() -> None:
    """The hexahedral branch's normalisation anchor."""
    cube = [
        [0.0, 0.0, 7.0],
        [7.0, 0.0, 7.0],
        [7.0, 7.0, 7.0],
        [0.0, 7.0, 7.0],
        [0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        [7.0, 7.0, 0.0],
        [0.0, 7.0, 0.0],
    ]
    cell = _mesh(cube, [(ElementType.HEXAHEDRON, tuple(range(8)))])

    result = quality(cell, AspectRatio3D())

    assert result.values[0] == pytest.approx(1.0, rel=1e-9)


def test_aspect_ratio_3d_of_a_hexahedron_worsens_as_the_cell_is_stretched() -> None:
    """A cube reads 1 and every longer box reads strictly worse than the one before it.

    The hexahedral branch is HOMARD's own measure — the worst of the 24 corner tetrahedra
    against a cube's constant — so the gate is its two properties rather than a transcription
    of it: exactly 1 on the cell it is normalised to, and strictly monotone in the
    elongation, in both directions from the cube.
    """
    heights = [3.5, 7.0, 14.0, 28.0]
    values = []
    for height in heights:
        box = [
            [0.0, 0.0, height],
            [7.0, 0.0, height],
            [7.0, 7.0, height],
            [0.0, 7.0, height],
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [7.0, 7.0, 0.0],
            [0.0, 7.0, 0.0],
        ]
        cell = _mesh(box, [(ElementType.HEXAHEDRON, tuple(range(8)))])
        values.append(float(quality(cell, AspectRatio3D()).values[0]))

    flattened, cube, doubled, quadrupled = values
    assert cube == pytest.approx(1.0, rel=1e-9)
    assert flattened > cube
    assert doubled > cube
    assert quadrupled > doubled


def test_warping_of_a_planar_quadrangle_is_zero() -> None:
    """The non-trigger case: a flat face is not warped."""
    face = _mesh(_skew_quad_nodes(), [(ElementType.QUADRANGLE, (0, 1, 2, 3))])

    result = quality(face, Warping())

    assert result.values[0] == 0.0


def test_warping_of_a_folded_quadrangle_matches_its_corner_normal_angle() -> None:
    """One corner lifted out of plane, against the four-corner construction."""
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([4.0, 4.0, 1.0]),
        np.array([0.0, 4.0, 0.0]),
    ]
    face = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])

    result = quality(face, Warping())

    assert result.values[0] == pytest.approx(_warping(points), rel=1e-9)
    assert result.values[0] > 1.0


def test_warping_below_a_tenth_of_a_degree_reads_as_exactly_zero() -> None:
    """The deadband, pinned: a nearly planar face is reported as planar, not as tiny."""
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([4.0, 4.0, 1e-4]),
        np.array([0.0, 4.0, 0.0]),
    ]
    face = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])

    result = quality(face, Warping())

    assert 0.0 < _warping(points) < 0.1
    assert result.values[0] == 0.0


def test_taper_of_a_parallelogram_is_zero() -> None:
    """The non-trigger case: equal corner triangles mean no taper."""
    face = _mesh(_skew_quad_nodes(), [(ElementType.QUADRANGLE, (0, 1, 2, 3))])

    result = quality(face, Taper())

    assert result.values[0] == 0.0


def test_taper_of_a_trapezium_equals_its_relative_corner_area_spread() -> None:
    """A 6-wide base against a 3-wide top, so the four corner triangles differ by a third."""
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([6.0, 0.0, 0.0]),
        np.array([4.0, 3.0, 0.0]),
        np.array([1.0, 3.0, 0.0]),
    ]
    face = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])

    result = quality(face, Taper())

    assert result.values[0] == pytest.approx(_taper(points), rel=1e-12)
    assert result.values[0] == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_skew_of_a_rectangle_is_zero() -> None:
    """The non-trigger case: perpendicular midlines mean no skew."""
    face = _mesh(
        [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [7.0, 3.0, 0.0], [0.0, 3.0, 0.0]],
        [(ElementType.QUADRANGLE, (0, 1, 2, 3))],
    )

    result = quality(face, Skew())

    assert result.values[0] == 0.0


def test_skew_of_a_sheared_quadrangle_equals_its_midline_departure() -> None:
    """A rhombus, against the angle between the two lines joining opposite midpoints."""
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([6.0, 3.0, 0.0]),
        np.array([2.0, 3.0, 0.0]),
    ]
    face = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])

    result = quality(face, Skew())

    assert result.values[0] == pytest.approx(_quad_skew(points), rel=1e-12)
    assert result.values[0] > 30.0


def test_skew_of_a_triangle_uses_its_medians_not_its_interior_angles() -> None:
    """Pinned because the natural expectation is the other one and it reads differently."""
    points = _skew_triangle_nodes()
    face = _mesh(points, [(ElementType.TRIANGLE, (0, 1, 2))])
    expected = max(
        math.degrees(abs(math.pi / 2.0 - _skew_angle(points[i - 1], points[i], points[i - 2])))
        for i in range(3)
    )

    result = quality(face, Skew())

    assert result.values[0] == pytest.approx(expected, rel=1e-12)
    assert result.values[0] != pytest.approx(90.0 - _minimum_angle(points), rel=1e-3)


def test_minimum_angle_of_a_skewed_triangle_equals_its_smallest_interior_angle() -> None:
    """A 45-degree corner, arranged so the answer is exact."""
    points = _skew_triangle_nodes()
    face = _mesh(points, [(ElementType.TRIANGLE, (0, 1, 2))])

    result = quality(face, MinimumAngle())

    assert result.values[0] == pytest.approx(_minimum_angle(points), rel=1e-12)
    assert result.values[0] == pytest.approx(45.0, rel=1e-12)


def test_length2d_of_a_skewed_triangle_equals_its_shortest_side() -> None:
    """The shortest of the three sides, which is neither the first nor the last."""
    points = _skew_triangle_nodes()
    face = _mesh(points, [(ElementType.TRIANGLE, (0, 1, 2))])
    expected = min(
        _distance(points[i], points[(i + 1) % 3]) for i in range(3)
    )

    result = quality(face, Length2D())

    assert result.values[0] == pytest.approx(expected, rel=1e-12)
    assert result.values[0] == pytest.approx(math.sqrt(10.0), rel=1e-12)


def test_length3d_of_a_sheared_hexahedron_equals_its_shortest_edge() -> None:
    """The three edge vectors have lengths 3, sqrt(50) and sqrt(126); the answer is 3."""
    cell = _sheared_hexa()

    result = quality(cell, Length3D())

    assert result.values[0] == pytest.approx(float(np.linalg.norm(EDGE_A)), rel=1e-12)
    assert result.values[0] == pytest.approx(3.0, rel=1e-12)


def test_max_element_length2d_of_a_skewed_quadrangle_equals_its_longest_span() -> None:
    """A parallelogram whose long diagonal beats every side."""
    points = _skew_quad_nodes()
    face = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])
    spans = [_distance(points[i], points[(i + 1) % 4]) for i in range(4)]
    spans += [_distance(points[0], points[2]), _distance(points[1], points[3])]

    result = quality(face, MaxElementLength2D())

    assert result.values[0] == pytest.approx(max(spans), rel=1e-12)


def test_max_element_length3d_of_a_sheared_hexahedron_equals_its_body_diagonal() -> None:
    """The longest span of a parallelepiped is the diagonal along A + B + C."""
    cell = _sheared_hexa()
    expected = float(np.linalg.norm(EDGE_A + EDGE_B + EDGE_C))

    result = quality(cell, MaxElementLength3D())

    assert result.values[0] == pytest.approx(expected, rel=1e-12)
    assert result.values[0] == pytest.approx(math.sqrt(221.0), rel=1e-12)


def test_multi_connection_counts_every_higher_element_sharing_an_edge_element(
    box_mesher: Mesher,
) -> None:
    """Checked against a count taken from the harvest's own connectivity.

    Volumes count as well as faces, which is worth pinning: an edge on the skin of a
    structured hexahedral mesh reads 3 — two surface quadrangles plus the cell behind them —
    and not the 2 the name suggests.
    """
    mesh = box_mesher.mesh()
    higher_rows = [
        i
        for i in range(mesh.element_count)
        if int(mesh.element_type[i])
        in (int(ElementType.QUADRANGLE), int(ElementType.HEXAHEDRON))
    ]
    edge_rows = [
        i
        for i in range(mesh.element_count)
        if int(mesh.element_type[i]) == int(ElementType.EDGE)
    ]
    higher_nodes = [frozenset(int(n) for n in mesh.nodes_of(i)) for i in higher_rows]
    expected = {
        int(mesh.element_id[i]): sum(
            1
            for nodes in higher_nodes
            if set(int(n) for n in mesh.nodes_of(i)) <= nodes
        )
        for i in edge_rows
    }

    result = box_mesher.quality(MultiConnection())

    assert result.family is ElementDimension.EDGE
    assert result.count == len(edge_rows)
    for element_id, count in expected.items():
        assert _value_of(result, element_id) == pytest.approx(float(count))
    assert set(expected.values()) == {3}


def test_multi_connection2d_counts_the_elements_sharing_a_face_border() -> None:
    """Every border of a closed shell is shared by exactly two of its faces."""
    shell = _closed_box_shell()

    result = quality(shell, MultiConnection2D())

    assert result.count == 6
    assert np.all(result.values == 2.0)


def test_node_connectivity_number_counts_the_cells_using_each_node(
    box_mesher: Mesher,
) -> None:
    """Checked against a count taken from the harvest, cell by cell."""
    mesh = box_mesher.mesh()
    counts: dict[int, int] = {int(node): 0 for node in mesh.node_id}
    for row in range(mesh.element_count):
        if int(mesh.element_type[row]) != int(ElementType.HEXAHEDRON):
            continue
        for node_row in set(int(n) for n in mesh.nodes_of(row)):
            counts[int(mesh.node_id[node_row])] += 1

    result = box_mesher.quality(NodeConnectivityNumber())

    assert result.family is ElementDimension.NODE
    for node_id, count in counts.items():
        assert _value_of(result, node_id) == pytest.approx(float(count))
    assert max(counts.values()) == 8


def test_deflection2d_measures_the_gap_between_a_face_and_the_surface_it_lies_on() -> None:
    """The one measure that says whether a surface mesh follows the geometry, not whether
    its elements are well shaped.

    On a cylinder the projection is radial, so a face's deflection is exactly how far its
    centroid sits inside the radius. Both directions are asserted: a face on the curved wall
    is off the surface by that gap, and a face on a flat cap is on the surface exactly.
    """
    radius = 7.0
    session = Session()
    session.add_cylinder(radius, 11.0)
    with Mesher(ps.load_brep(session.brep())) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=12))
        mesher.assign(Mefisto2D())
        mesher.assign(MaxElementArea(max_area=4.0))
        mesher.compute()

        result = mesher.quality(Deflection2D())
        mesh = mesher.mesh()

    rows = {int(mesh.element_id[i]): i for i in range(mesh.element_count)}
    wall: list[tuple[float, float]] = []
    caps: list[float] = []
    for k in range(result.count):
        row = rows[int(result.element_ids[k])]
        centroid = mesh.node_coords[[int(n) for n in mesh.nodes_of(row)]].mean(axis=0)
        gap = radius - math.hypot(float(centroid[0]), float(centroid[1]))
        if int(mesh.element_ordinal[row]) == 1:
            wall.append((float(result.values[k]), gap))
        else:
            caps.append(float(result.values[k]))

    assert wall, "the fixture produced no face on the cylindrical wall"
    assert caps, "the fixture produced no face on a flat cap"
    for reported, gap in wall:
        assert reported == pytest.approx(gap, abs=1e-9)
    assert max(reported for reported, _ in wall) > 0.1
    assert max(caps) == pytest.approx(0.0, abs=1e-9)


def test_a_control_reports_the_elements_it_cannot_measure_as_skipped() -> None:
    """A warping is undefined on a triangle, and a zero there would read as a flat face."""
    nodes = [*_skew_quad_nodes(), np.array([2.0, 6.0, 0.0])]
    mesh = _mesh(
        nodes,
        [
            (ElementType.QUADRANGLE, (0, 1, 2, 3)),
            (ElementType.TRIANGLE, (0, 1, 4)),
            (ElementType.TRIANGLE, (1, 2, 4)),
        ],
    )

    result = quality(mesh, Warping())

    assert result.count == 1
    assert result.skipped == 2
    assert int(result.element_ids[0]) == 1


def test_a_quality_result_is_numpy_arrays_keyed_by_mesh_id(box_mesher: Mesher) -> None:
    """The join key: a value is addressable by the same id the harvest reports."""
    mesh = box_mesher.mesh()

    result = box_mesher.quality(Volume())

    assert isinstance(result.element_ids, np.ndarray)
    assert isinstance(result.values, np.ndarray)
    assert result.element_ids.dtype == np.int64
    assert result.values.dtype == np.float64
    assert result.element_ids.shape == result.values.shape
    volumes = mesh.element_id[mesh.element_type == int(ElementType.HEXAHEDRON)]
    assert set(result.element_ids.tolist()) == set(volumes.tolist())
    assert float(result.values.sum()) == pytest.approx(3.0 * 7.0 * 11.0, rel=1e-9)


# ---- The predicates, both directions ------------------------------------------------------ #


def test_bad_oriented_volume_flags_an_inverted_cell_and_not_a_correct_one() -> None:
    """Both directions, on the same coordinates with the winding reversed."""
    nodes = _sheared_hexa_nodes()
    correct = _mesh(nodes, [(ElementType.HEXAHEDRON, tuple(range(8)))])
    inverted = _mesh(nodes, [(ElementType.HEXAHEDRON, (4, 5, 6, 7, 0, 1, 2, 3))])

    assert select(correct, BadOrientedVolume()).count == 0
    assert select(inverted, BadOrientedVolume()).ids.tolist() == [1]


def test_bare_border_volume_flags_a_cell_whose_skin_is_missing_and_not_a_covered_one() -> None:
    """A volume mesh with no face elements has nothing to carry a boundary condition."""
    bare = _sheared_hexa()
    covered = _closed_box_shell()

    assert select(bare, BareBorderVolume()).ids.tolist() == [1]
    assert select(covered, BareBorderVolume()).count == 0


def test_over_constrained_volume_flags_a_cell_with_every_node_on_the_boundary() -> None:
    """The same two fixtures, the other way round: a fully skinned lone cell is trapped."""
    covered = _closed_box_shell()
    bare = _sheared_hexa()

    assert select(covered, OverConstrainedVolume()).ids.tolist() == [1]
    assert select(bare, OverConstrainedVolume()).count == 0


def test_over_constrained_face_flags_a_face_whose_nodes_all_carry_edge_elements() -> None:
    """A quadrangle with its four borders as edge elements has no free node left."""
    points = _skew_quad_nodes()
    bordered = _mesh(
        points,
        [
            (ElementType.QUADRANGLE, (0, 1, 2, 3)),
            (ElementType.EDGE, (0, 1)),
            (ElementType.EDGE, (1, 2)),
            (ElementType.EDGE, (2, 3)),
            (ElementType.EDGE, (3, 0)),
        ],
    )
    free = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])

    assert select(bordered, OverConstrainedFace()).ids.tolist() == [1]
    assert select(free, OverConstrainedFace()).count == 0


def test_bare_border_face_flags_a_face_whose_border_has_no_edge_element() -> None:
    """The 2-D counterpart, both ways round."""
    points = _skew_quad_nodes()
    free = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])
    bordered = _mesh(
        points,
        [
            (ElementType.QUADRANGLE, (0, 1, 2, 3)),
            (ElementType.EDGE, (0, 1)),
            (ElementType.EDGE, (1, 2)),
            (ElementType.EDGE, (2, 3)),
            (ElementType.EDGE, (3, 0)),
        ],
    )

    assert select(free, BareBorderFace()).ids.tolist() == [1]
    assert select(bordered, BareBorderFace()).count == 0


def test_free_edges_flags_an_open_patch_and_not_a_closed_shell() -> None:
    """A lone face is all border; the six faces of a closed cell share every one of theirs."""
    open_patch = _mesh(_skew_quad_nodes(), [(ElementType.QUADRANGLE, (0, 1, 2, 3))])
    closed = _closed_box_shell()

    assert select(open_patch, FreeEdges()).ids.tolist() == [1]
    assert select(closed, FreeEdges()).count == 0


def test_free_faces_flags_the_skin_of_a_volume_mesh_and_not_an_internal_face() -> None:
    """A face between two cells is not free; the ones on the outside are."""
    interior = _stacked_cells_with_interface()
    skin = _closed_box_shell()

    assert select(interior, FreeFaces()).count == 0
    assert select(skin, FreeFaces()).count == 6


def _stacked_cells_with_interface() -> MeshData:
    """Two hexahedra sharing a facet, with one face element on that facet only."""
    lower = [
        [0.0, 0.0, 3.0],
        [7.0, 0.0, 3.0],
        [7.0, 11.0, 3.0],
        [0.0, 11.0, 3.0],
        [0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        [7.0, 11.0, 0.0],
        [0.0, 11.0, 0.0],
    ]
    upper = [
        [0.0, 0.0, 6.0],
        [7.0, 0.0, 6.0],
        [7.0, 11.0, 6.0],
        [0.0, 11.0, 6.0],
    ]
    return _mesh(
        [*lower, *upper],
        [
            (ElementType.HEXAHEDRON, (0, 1, 2, 3, 4, 5, 6, 7)),
            (ElementType.HEXAHEDRON, (8, 9, 10, 11, 0, 1, 2, 3)),
            (ElementType.QUADRANGLE, (0, 1, 2, 3)),
        ],
    )


def test_free_nodes_flags_an_unused_node_and_not_a_used_one() -> None:
    """A node no element references, which no element-side check would ever notice."""
    points = [*_skew_quad_nodes(), np.array([100.0, 100.0, 100.0])]
    with_orphan = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])
    without = _mesh(_skew_quad_nodes(), [(ElementType.QUADRANGLE, (0, 1, 2, 3))])

    assert select(with_orphan, FreeNodes()).ids.tolist() == [5]
    assert select(without, FreeNodes()).count == 0


def test_free_borders_flags_an_edge_element_bordering_at_most_one_face() -> None:
    """Both directions: the border of a lone face against one inside a closed shell."""
    points = _skew_quad_nodes()
    lone = _mesh(
        points,
        [(ElementType.QUADRANGLE, (0, 1, 2, 3)), (ElementType.EDGE, (0, 1))],
    )
    nodes = _sheared_hexa_nodes()
    shared = _mesh(
        nodes,
        [
            (ElementType.QUADRANGLE, (0, 1, 2, 3)),
            (ElementType.QUADRANGLE, (0, 1, 5, 4)),
            (ElementType.EDGE, (0, 1)),
        ],
    )

    assert select(lone, FreeBorders()).count == 1
    assert select(shared, FreeBorders()).count == 0


def test_coincident_nodes_flags_a_duplicated_position_and_not_a_distinct_one() -> None:
    """Two nodes at the same place, and the same mesh with them moved apart."""
    points = [*_skew_quad_nodes(), np.array([0.0, 0.0, 0.0])]
    duplicated = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])
    apart = [*_skew_quad_nodes(), np.array([0.0, 0.0, 1.0])]
    distinct = _mesh(apart, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])

    assert sorted(select(duplicated, CoincidentNodes(tolerance=1e-6)).ids.tolist()) == [1, 5]
    assert select(distinct, CoincidentNodes(tolerance=1e-6)).count == 0


def test_coincident_elements_flags_a_duplicated_face_and_not_a_distinct_one() -> None:
    """Two faces on exactly the same nodes — a baffle written twice by accident."""
    points = _skew_quad_nodes()
    duplicated = _mesh(
        points,
        [
            (ElementType.QUADRANGLE, (0, 1, 2, 3)),
            (ElementType.QUADRANGLE, (0, 1, 2, 3)),
        ],
    )
    single = _mesh(points, [(ElementType.QUADRANGLE, (0, 1, 2, 3))])
    predicate = CoincidentElements(element_family=ElementDimension.FACE)

    assert select(duplicated, predicate).count == 2
    assert select(single, predicate).count == 0


def test_manifold_part_keeps_a_sound_patch_and_drops_a_degenerate_face() -> None:
    """Both directions: two faces that grow into one manifold region, and a collinear one.

    A face of zero area has no normal to walk across, so the walk stops rather than
    misclassifying it — which is the property that makes this the right tool for pulling one
    sound shell out of a mesh that carries damage.
    """
    sound = _mesh(
        [
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [7.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
            [14.0, 3.0, 0.0],
            [14.0, 0.0, 0.0],
        ],
        [
            (ElementType.QUADRANGLE, (0, 1, 2, 3)),
            (ElementType.QUADRANGLE, (1, 5, 4, 2)),
        ],
    )
    damaged = _mesh(
        [
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [7.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
            [14.0, 0.0, 0.0],
        ],
        [
            (ElementType.QUADRANGLE, (0, 1, 2, 3)),
            (ElementType.TRIANGLE, (0, 1, 4)),
        ],
    )

    assert select(sound, ManifoldPart(start_element=1)).ids.tolist() == [1, 2]
    assert select(damaged, ManifoldPart(start_element=1)).ids.tolist() == [1]


def test_range_of_ids_accepts_exactly_the_ids_it_was_given(box_mesher: Mesher) -> None:
    """Built from real mesh ids, because element ids are one sequence across every family."""
    volumes = box_mesher.quality(Volume()).element_ids
    chosen = [int(i) for i in volumes[:5]]

    inside = box_mesher.select(
        RangeOfIds(ids=tuple(chosen), element_family=ElementDimension.VOLUME)
    )
    outside = box_mesher.select(
        RangeOfIds(ids=(int(volumes[-1]),), element_family=ElementDimension.VOLUME)
    )

    assert sorted(inside.ids.tolist()) == sorted(chosen)
    assert outside.ids.tolist() == [int(volumes[-1])]


def test_elements_on_shape_selects_one_face_of_the_model_and_not_the_others(
    box_mesher: Mesher,
) -> None:
    """Both directions: the elements on face 1, against the ones the harvest binds there."""
    mesh = box_mesher.mesh()
    expected = {
        int(mesh.element_id[row])
        for row in range(mesh.element_count)
        if int(mesh.element_kind[row]) == int(SubShapeKind.FACE)
        and int(mesh.element_ordinal[row]) == 1
    }

    on_first = box_mesher.select(
        ElementsOnShape(
            element_family=ElementDimension.FACE, on=SubShape(SubShapeKind.FACE, 1)
        )
    )
    on_second = box_mesher.select(
        ElementsOnShape(
            element_family=ElementDimension.FACE, on=SubShape(SubShapeKind.FACE, 2)
        )
    )

    assert set(on_first.ids.tolist()) == expected
    assert set(on_second.ids.tolist()).isdisjoint(expected)


def test_a_shape_reading_control_is_refused_on_a_mesh_given_as_arrays() -> None:
    """It would otherwise answer 0 for every face, which reads as a perfect fit."""
    mesh = _closed_box_shell()

    with pytest.raises(PysmeshError, match="reads the geometry"):
        quality(mesh, Deflection2D())


def test_a_group_reading_predicate_is_refused_on_a_mesh_given_as_arrays() -> None:
    """A mesh handed over as arrays carries no groups to test membership of."""
    mesh = _closed_box_shell()

    with pytest.raises(PysmeshError, match="reads the geometry"):
        select(mesh, BelongToGroup(group_name="wall"))


# ---- The filter algebra ------------------------------------------------------------------- #


def test_not_inverts_a_predicate(box_mesher: Mesher) -> None:
    """Every cell is well oriented, so the negation selects all of them."""
    total = box_mesher.quality(Volume()).count

    inverted = box_mesher.select(Not(predicate=BadOrientedVolume()))

    assert box_mesher.select(BadOrientedVolume()).count == 0
    assert inverted.count == total


def test_and_selects_the_intersection_and_or_the_union(box_mesher: Mesher) -> None:
    """Two comparators on the same measure, so the expected sets are arithmetic."""
    values = box_mesher.quality(Volume())
    cell = float(values.values[0])
    big = MoreThan(control=Volume(), margin=cell / 2.0)
    small = LessThan(control=Volume(), margin=cell / 2.0)

    both = box_mesher.select(And(predicate1=big, predicate2=small))
    either = box_mesher.select(Or(predicate1=big, predicate2=small))

    assert both.count == 0
    assert either.count == values.count


def test_and_narrows_rather_than_passing_everything(box_mesher: Mesher) -> None:
    """The falsification: an AND that ignored one side would select all 27 cells."""
    volumes = box_mesher.quality(Volume()).element_ids
    chosen = tuple(int(i) for i in volumes[:4])
    narrowed = box_mesher.select(
        And(
            predicate1=Not(predicate=BadOrientedVolume()),
            predicate2=RangeOfIds(ids=chosen, element_family=ElementDimension.VOLUME),
        )
    )

    assert sorted(narrowed.ids.tolist()) == sorted(chosen)
    assert narrowed.count < len(volumes)


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda cell: LessThan(control=Volume(), margin=cell * 1.5), 27),
        (lambda cell: LessThan(control=Volume(), margin=cell * 0.5), 0),
        (lambda cell: MoreThan(control=Volume(), margin=cell * 0.5), 27),
        (lambda cell: MoreThan(control=Volume(), margin=cell * 1.5), 0),
    ],
)
def test_a_comparator_turns_a_measure_into_a_selection(
    box_mesher: Mesher, factory: object, expected: int
) -> None:
    """Both directions on both comparators: every cell, or none."""
    cell = float(box_mesher.quality(Volume()).values[0])
    predicate = factory(cell)  # type: ignore[operator]  # parametrised factory

    result = box_mesher.select(predicate)

    assert result.count == expected


def test_equal_to_matches_within_its_tolerance_and_not_outside_it(
    box_mesher: Mesher,
) -> None:
    """A uniform mesh: every cell matches its own volume and none matches twice it."""
    cell = float(box_mesher.quality(Volume()).values[0])

    matching = box_mesher.select(EqualTo(control=Volume(), margin=cell, tolerance=1e-6))
    missing = box_mesher.select(
        EqualTo(control=Volume(), margin=cell * 2.0, tolerance=1e-6)
    )

    assert matching.count == 27
    assert missing.count == 0


# ---- The drift check ---------------------------------------------------------------------- #


def test_an_unknown_control_is_refused_by_name() -> None:
    """The catalogue is closed; a name that is not in it fails rather than doing nothing."""
    mesh = _sheared_hexa()

    with pytest.raises(PysmeshError, match="Unknown quality control"):
        mesh_quality = ps._core.mesh_quality  # noqa: SLF001 - the drift check is native
        mesh_quality(
            {
                "node_coords": mesh.node_coords,
                "node_id": mesh.node_id,
                "element_offsets": mesh.element_offsets,
                "element_nodes": mesh.element_nodes,
                "element_type": mesh.element_type,
                "element_id": mesh.element_id,
            },
            "NotAControl",
            {},
        )


def test_an_unknown_parameter_is_refused_by_name() -> None:
    """The drift check itself, shown able to fail: a field with no branch is not dropped."""
    mesh = _sheared_hexa()

    with pytest.raises(PysmeshError, match="does not take the parameter"):
        ps._core.mesh_select(  # noqa: SLF001 - the drift check is native
            {
                "node_coords": mesh.node_coords,
                "node_id": mesh.node_id,
                "element_offsets": mesh.element_offsets,
                "element_nodes": mesh.element_nodes,
                "element_type": mesh.element_type,
                "element_id": mesh.element_id,
            },
            "CoincidentNodes",
            {"tolerance": 1e-6, "unheard_of": 1},
        )


def test_a_polyhedral_mesh_cannot_be_measured_through_the_array_route() -> None:
    """Refused by name: a polyhedron's node count does not determine its shape."""
    session = Session()
    session.add_box(8.0, 8.0, 6.0, origin=(-4.0, -4.0, 0.0))
    block = list(session.entities(ps.EntityKind.SOLID))
    session.add_cylinder(1.5, 6.0)
    bore = [e for e in session.entities(ps.EntityKind.SOLID) if e not in block]
    session.cut(block, bore)
    with Mesher(ps.load_brep(session.brep())) as mesher:
        mesher.assign(ps.Cartesian3D())
        mesher.assign(
            ps.CartesianParameters3D(
                spacing_x="2.0", spacing_y="2.0", spacing_z="2.0", size_threshold=4.0
            )
        )
        mesher.compute()
        mesh = mesher.mesh()
        from_mesher = mesher.quality(Volume())

    assert from_mesher.count > 0
    with pytest.raises(PysmeshError, match="polygon or a polyhedron"):
        quality(mesh, Volume())
