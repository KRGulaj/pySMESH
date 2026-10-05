# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-05

"""NETGEN with viscous layers: prisms on chosen walls, tetrahedra inside.

NETGEN_3D and NETGEN_2D3D build ViscousLayers, NETGEN_2D and NETGEN_2D_ONLY build
ViscousLayers2D (``NETGENPlugin_NETGEN_3D.cxx``, ``NETGENPlugin_Mesher.cxx``): SMESH's
layer builder grows the stack into a proxy mesh, and NETGEN fills the rest. The oracle
of a stack of ``N`` layers of total thickness ``T`` and stretch ``f``: the first layer
is ``t1 = T (f - 1) / (f^N - 1)``, and layer ``k`` ends at
``d_k = t1 (f^k - 1) / (f - 1)`` from the wall. On a flat wall every layer node lies on
one of these planes.

Each mesh is also checked for conformity (every face of a cell is shared by exactly two
cells, or lies on the shape's boundary and belongs to one) and for inverted cells (every
cell has a positive volume by the divergence theorem over its outward-wound faces).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    ElementType,
    EntityKind,
    LocalLength,
    MeshData,
    Mesher,
    Netgen1D2D,
    Netgen1D2D3D,
    Netgen2D,
    Netgen3D,
    NetgenParameters,
    NetgenParameters2D,
    PysmeshError,
    QuadranglePreference,
    Regular1D,
    Session,
    SubShapeKind,
    ViscousLayerBuilder,
    ViscousLayers,
    ViscousLayers2D,
)

_T, _N, _F = 0.3, 3, 1.2
_TOL = 1e-9

# The faces of each linear cell type, as node positions in SMESH's own order, wound so
# the outward normal points away from the cell (the table of tests/test_mesher.py).
_CELL_FACES: dict[int, tuple[tuple[int, ...], ...]] = {
    int(ElementType.TETRAHEDRON): ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
    int(ElementType.PYRAMID): (
        (0, 1, 2, 3),
        (0, 4, 1),
        (1, 4, 2),
        (2, 4, 3),
        (3, 4, 0),
    ),
    int(ElementType.PENTAHEDRON): (
        (0, 1, 2),
        (3, 5, 4),
        (0, 3, 4, 1),
        (1, 4, 5, 2),
        (2, 5, 3, 0),
    ),
    int(ElementType.HEXAHEDRON): (
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ),
}

Recipe = Callable[[Mesher], None]


def _layer_ends() -> list[float]:
    """0 and d_1 .. d_N of the stack (T, N, f)."""
    t1 = _T * (_F - 1.0) / (_F**_N - 1.0)
    return [0.0] + [t1 * (_F**k - 1.0) / (_F - 1.0) for k in range(1, _N + 1)]


def _box() -> ps.Shape:
    """The 2 x 3 x 4 box at the origin."""
    s = Session()
    s.add_box(2.0, 3.0, 4.0)
    return ps.load_brep(s.brep())


def _face_at(shape: ps.Shape, axis: int, value: float) -> int:
    """The ordinal of the planar face at coordinate ``value`` on ``axis``."""
    for i, f in enumerate(shape.faces(), start=1):
        b = np.asarray(f.bbox).ravel()
        if b[axis] == value and b[axis + 3] == value:
            return i
    raise AssertionError(f"no face at {'xyz'[axis]} = {value}")


def _count(md: MeshData, kind: ElementType) -> int:
    """How many elements of one type the mesh holds."""
    return int((md.element_type == int(kind)).sum())


def _on_face(md: MeshData, face: int, kinds: tuple[ElementType, ...]) -> int:
    """How many elements of the given types are bound to one face."""
    hit = np.isin(md.element_type, [int(k) for k in kinds])
    hit &= md.element_kind == int(SubShapeKind.FACE)
    hit &= md.element_ordinal == face
    return int(hit.sum())


def _cells(md: MeshData) -> list[tuple[int, NDArray[np.int64]]]:
    """Every volume cell as (type, node rows)."""
    out = []
    for e in range(md.element_count):
        t = int(md.element_type[e])
        if t in _CELL_FACES:
            out.append((t, np.asarray(md.nodes_of(e), dtype=np.int64)))
    return out


def _flux(
    a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64]
) -> float:
    """The flux of x / 3 through the triangle (a, b, c): its signed volume term."""
    return float(np.dot(a, np.cross(b, c))) / 6.0


def _cell_volume(xyz: NDArray[np.float64], t: int, nodes: NDArray[np.int64]) -> float:
    """A cell's volume by the divergence theorem over its outward-wound faces.

    A quadrangle face counts as its bilinear patch: the flux through it is the mean of
    its two triangle splits. So a warped quadrangle that two cells share adds the same
    amount to one as it takes from the other, and the volumes of a conforming mesh add
    up to the volume its boundary encloses.
    """
    total = 0.0
    for face in _CELL_FACES[t]:
        p = xyz[nodes[list(face)]]
        if len(p) == 3:
            total += _flux(p[0], p[1], p[2])
        else:
            first = _flux(p[0], p[1], p[2]) + _flux(p[0], p[2], p[3])
            second = _flux(p[0], p[1], p[3]) + _flux(p[1], p[2], p[3])
            total += 0.5 * (first + second)
    return total


def _check_cells(
    md: MeshData, on_boundary: Callable[[NDArray[np.float64]], bool]
) -> float:
    """Conformity and orientation of every cell; returns the total volume."""
    xyz = md.node_coords
    use: dict[tuple[int, ...], int] = {}
    total = 0.0
    for t, nodes in _cells(md):
        volume = _cell_volume(xyz, t, nodes)
        assert volume > 0.0
        total += volume
        for face in _CELL_FACES[t]:
            key = tuple(sorted(int(nodes[i]) for i in face))
            use[key] = use.get(key, 0) + 1
    assert set(use.values()) <= {1, 2}
    for key, n in use.items():
        if n == 1:
            assert on_boundary(xyz[list(key)])
    return total


def _on_box(points: NDArray[np.float64]) -> bool:
    """Whether all points lie on one face plane of the 2 x 3 x 4 box."""
    for axis, top in ((0, 2.0), (1, 3.0), (2, 4.0)):
        for value in (0.0, top):
            if np.all(np.abs(points[:, axis] - value) <= _TOL):
                return True
    return False


def _levels(values: NDArray[np.float64]) -> list[float]:
    """The distinct distances from the wall below T + tol, rounded to 9 digits."""
    near = values[values < _T + _TOL]
    return sorted({round(float(v), 9) for v in near})


def _netgen_1d2d3d(m: Mesher) -> None:
    """Netgen1D2D3D at max size 0.5."""
    m.assign(Netgen1D2D3D())
    m.assign(NetgenParameters(max_size=0.5))


def _netgen_1d2d_then_3d(m: Mesher) -> None:
    """Netgen1D2D, then Netgen3D, at max size 0.5."""
    m.assign(Netgen1D2D())
    m.assign(NetgenParameters2D(max_size=0.5))
    m.assign(Netgen3D())
    m.assign(NetgenParameters(max_size=0.5))


def _regular_then_netgen(m: Mesher) -> None:
    """Regular1D at 0.5, Netgen2D, Netgen3D."""
    m.assign(Regular1D())
    m.assign(LocalLength(length=0.5))
    m.assign(Netgen2D())
    m.assign(Netgen3D())


_RECIPES_3D: dict[str, Recipe] = {
    "1d2d3d": _netgen_1d2d3d,
    "1d2d+3d": _netgen_1d2d_then_3d,
    "regular+2d+3d": _regular_then_netgen,
}


@pytest.mark.parametrize("recipe", list(_RECIPES_3D))
def test_netgen_grows_prism_layers_on_a_flat_wall_and_fills_the_box(
    recipe: str,
) -> None:
    """Layers (T 0.3, N 3, f 1.2) on the bottom face z = 0 of a 2 x 3 x 4 box.

    Prisms: N per wall triangle. Every node within T of the wall lies on a plane d_k.
    The cell volumes sum to 24 (planar faces: exact to round-off).
    """
    shape = _box()
    bottom = _face_at(shape, 2, 0.0)

    with Mesher(shape) as mesher:
        _RECIPES_3D[recipe](mesher)
        mesher.assign(
            ViscousLayers(
                total_thickness=_T,
                layer_count=_N,
                stretch_factor=_F,
                boundary=(bottom,),
                group_name="layers",
            )
        )
        mesher.compute()
        md = mesher.mesh()
        group = mesher.group("layers")
    wall = _on_face(md, bottom, (ElementType.TRIANGLE,))
    total = _check_cells(md, _on_box)

    assert wall > 0
    assert _count(md, ElementType.PENTAHEDRON) == _N * wall
    assert len(group.element_ids) == _N * wall
    assert _levels(md.node_coords[:, 2]) == pytest.approx(_layer_ends(), abs=_TOL)
    assert total == pytest.approx(24.0, rel=1e-12)


def test_netgen_grows_prism_layers_on_a_cylinder_wall() -> None:
    """Layers (T 0.1, N 3, f 1.2) on the lateral face of a cylinder r 1, h 3, max 0.4.

    Prisms: N per wall triangle. The cells are conforming and none is inverted. The
    volume is that of the inscribed polyhedron: below pi r^2 h, by at most the chord
    deficit, (L / r)^2 / 4 of it for edges of length L <= 0.6 (1.5 x max_size).
    """
    s = Session()
    s.add_cylinder(1.0, 3.0)
    shape = ps.load_brep(s.brep())
    lateral = next(
        i
        for i, f in enumerate(shape.faces(), start=1)
        if np.ptp(np.asarray(f.bbox).ravel()[[2, 5]]) > 0.0
    )

    with Mesher(shape) as mesher:
        mesher.assign(Netgen1D2D3D())
        mesher.assign(NetgenParameters(max_size=0.4))
        mesher.assign(
            ViscousLayers(
                total_thickness=0.1,
                layer_count=_N,
                stretch_factor=_F,
                boundary=(lateral,),
                group_name="layers",
            )
        )
        mesher.compute()
        md = mesher.mesh()

    def on_cylinder(points: NDArray[np.float64]) -> bool:
        z = points[:, 2]
        rho = np.hypot(points[:, 0], points[:, 1])
        return bool(
            np.all(np.abs(z) <= _TOL)
            or np.all(np.abs(z - 3.0) <= _TOL)
            or np.all(rho <= 1.0 + _TOL)
        )

    wall = _on_face(md, lateral, (ElementType.TRIANGLE,))
    total = _check_cells(md, on_cylinder)
    exact = math.pi * 3.0

    assert _count(md, ElementType.PENTAHEDRON) == _N * wall
    assert total < exact
    assert exact - total <= (0.6 / 1.0) ** 2 / 4.0 * exact


@pytest.mark.parametrize(
    "recipe",
    ["1d2d", "regular+2d_only"],
)
def test_netgen_grows_quadrangle_layers_on_an_edge_of_a_face(recipe: str) -> None:
    """ViscousLayers2D (T 0.3, N 3, f 1.2) on the edge y = 0 of a 2 x 3 rectangle.

    Quadrangles: N per wall segment. Every node within T of the edge lies on a line d_k.
    The signed area of the elements sums to 6, and each element is counter-clockwise.
    """
    s = Session()
    s.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, 3.0)
    face = ps.load_brep(s.brep())
    low = next(
        i
        for i, e in enumerate(face.edges(), start=1)
        if np.all(np.asarray(e.bbox).ravel()[[1, 4]] == 0.0)
    )

    with Mesher(face) as mesher:
        if recipe == "1d2d":
            mesher.assign(Netgen1D2D())
            mesher.assign(NetgenParameters2D(max_size=0.4))
        else:
            mesher.assign(Regular1D())
            mesher.assign(LocalLength(length=0.4))
            mesher.assign(Netgen2D())
        mesher.assign(
            ViscousLayers2D(
                total_thickness=_T,
                layer_count=_N,
                stretch_factor=_F,
                boundary=(low,),
                group_name="layers",
            )
        )
        mesher.compute()
        md = mesher.mesh()
    xyz = md.node_coords
    segments = int(
        (
            (md.element_type == int(ElementType.EDGE))
            & (md.element_kind == int(SubShapeKind.EDGE))
            & (md.element_ordinal == low)
        ).sum()
    )
    areas = []
    for e in range(md.element_count):
        if int(md.element_type[e]) in (
            int(ElementType.TRIANGLE),
            int(ElementType.QUADRANGLE),
        ):
            p = xyz[md.nodes_of(e)]
            areas.append(
                sum(
                    0.5 * float(np.cross(p[i] - p[0], p[i + 1] - p[0])[2])
                    for i in range(1, len(p) - 1)
                )
            )

    assert segments > 0
    assert _count(md, ElementType.QUADRANGLE) == _N * segments
    assert _levels(xyz[:, 1]) == pytest.approx(_layer_ends(), abs=_TOL)
    assert min(areas) > 0.0
    assert sum(areas) == pytest.approx(6.0, rel=1e-12)


@pytest.mark.parametrize("surface", ["quadrangle_preference", "quad_allowed"])
def test_netgen_grows_hexahedral_layers_on_a_quad_dominant_wall(surface: str) -> None:
    """A quad-dominant surface mesh under the layers: hexahedra on the wall quadrangles.

    Hexahedra: N per wall quadrangle, prisms N per wall triangle; pyramids join the
    quadrangles to NETGEN's tetrahedra. Conforming, none inverted, volume 24.
    """
    shape = _box()
    bottom = _face_at(shape, 2, 0.0)

    with Mesher(shape) as mesher:
        if surface == "quadrangle_preference":
            mesher.assign(Regular1D())
            mesher.assign(LocalLength(length=0.5))
            mesher.assign(Netgen2D())
            mesher.assign(QuadranglePreference())
        else:
            mesher.assign(Netgen1D2D())
            mesher.assign(NetgenParameters2D(max_size=0.5, quad_allowed=True))
        mesher.assign(Netgen3D())
        mesher.assign(
            ViscousLayers(
                total_thickness=_T,
                layer_count=_N,
                stretch_factor=_F,
                boundary=(bottom,),
                group_name="layers",
            )
        )
        mesher.compute()
        md = mesher.mesh()
    quads = _on_face(md, bottom, (ElementType.QUADRANGLE,))
    tris = _on_face(md, bottom, (ElementType.TRIANGLE,))
    total = _check_cells(md, _on_box)

    assert quads > 0
    assert _count(md, ElementType.HEXAHEDRON) == _N * quads
    assert _count(md, ElementType.PENTAHEDRON) == _N * tris
    assert _count(md, ElementType.PYRAMID) > 0
    assert _levels(md.node_coords[:, 2]) == pytest.approx(_layer_ends(), abs=_TOL)
    assert total == pytest.approx(24.0, rel=1e-12)


def test_the_layer_builder_wraps_a_netgen_inner_mesh() -> None:
    """ViscousLayerBuilder on the bottom of the 2 x 3 x 4 box, NETGEN inside.

    The shrunk shape is the box above z = T: 2 x 3 x 3.7 = 22.2. Wedges: N per triangle
    of the inner mesh's face z = T. Node planes d_k; conforming with the inner
    tetrahedra; volume 24.
    """
    shape = _box()
    builder = ViscousLayerBuilder(
        total_thickness=_T,
        layer_count=_N,
        stretch_factor=_F,
        boundary=(_face_at(shape, 2, 0.0),),
        ignore=False,
        group_name="layers",
    )

    with Mesher(shape) as outer:
        shrunk = outer.shrink_geometry(builder)
        with Mesher(shrunk) as inner:
            inner.assign(Netgen1D2D3D())
            inner.assign(NetgenParameters(max_size=0.5))
            inner.compute()
            wall = _on_face(
                inner.mesh(), _face_at(shrunk, 2, _T), (ElementType.TRIANGLE,)
            )
            inner_volume = _check_cells(inner.mesh(), lambda p: True)
            outer.add_layers(builder, inner)
        md = outer.mesh()
    total = _check_cells(md, _on_box)

    assert inner_volume == pytest.approx(2.0 * 3.0 * (4.0 - _T), rel=1e-12)
    assert _count(md, ElementType.PENTAHEDRON) == _N * wall
    assert _levels(md.node_coords[:, 2]) == pytest.approx(_layer_ends(), abs=_TOL)
    assert total == pytest.approx(24.0, rel=1e-12)


def test_layers_in_a_narrow_gap_are_cut_with_a_warning() -> None:
    """A 2 x 2 x 0.4 plate with layers (T 0.3) on its top and bottom.

    The two stacks would overlap (0.6 > 0.4); SMESH's builder limits them where they
    meet and reports it. The warning reaches ComputeReport.warnings, and the thickness
    reached is at most half the gap.
    """
    s = Session()
    s.add_box(2.0, 2.0, 0.4)
    plate = ps.load_brep(s.brep())

    with Mesher(plate) as mesher:
        mesher.assign(Netgen1D2D3D())
        mesher.assign(NetgenParameters(max_size=0.3))
        mesher.assign(
            ViscousLayers(
                total_thickness=_T,
                layer_count=_N,
                stretch_factor=_F,
                boundary=(_face_at(plate, 2, 0.0), _face_at(plate, 2, 0.4)),
                group_name="layers",
            )
        )
        report = mesher.compute()
    texts = [w.text for w in report.warnings if w.algorithm == "NETGEN_2D3D"]

    assert len(texts) == 1
    assert "not reached" in texts[0]
    reached = re.search(r"average reached thickness is ([0-9.eE+-]+)", texts[0])
    assert reached is not None
    assert 0.0 < float(reached.group(1)) <= 0.2


def _grooved_block(angle_deg: float) -> ps.Shape:
    """A 2 x 2 x 1 box with a V-groove 0.5 deep along y, of the given opening angle."""
    s = Session()
    s.add_box(2.0, 2.0, 1.0)
    block = s.entities(EntityKind.SOLID).tolist()
    half = 0.5 * math.tan(math.radians(angle_deg) / 2.0)
    before = set(s.entities(EntityKind.EDGE).tolist())
    s.add_polyline(
        [(1.0 - half, -0.1, 1.001), (1.0 + half, -0.1, 1.001), (1.0, -0.1, 0.5)],
        closed=True,
    )
    edges = [e for e in s.entities(EntityKind.EDGE).tolist() if e not in before]
    faces_before = set(s.entities(EntityKind.FACE).tolist())
    s.make_face(edges)
    face = [f for f in s.entities(EntityKind.FACE).tolist() if f not in faces_before]
    solids_before = set(s.entities(EntityKind.SOLID).tolist())
    s.extrude(face, (0.0, 2.2, 0.0))
    cutter = [
        x for x in s.entities(EntityKind.SOLID).tolist() if x not in solids_before
    ]
    s.cut(block, cutter)
    return ps.load_brep(s.brep())


def test_layers_in_a_too_thin_groove_raise_with_smeshs_reason() -> None:
    """Layers on every face of a block with a 5 degree V-groove: no first step fits."""
    with Mesher(_grooved_block(5.0)) as mesher:
        mesher.assign(Netgen1D2D3D())
        mesher.assign(NetgenParameters(max_size=0.2))
        mesher.assign(
            ViscousLayers(
                total_thickness=0.1,
                layer_count=_N,
                stretch_factor=_F,
                boundary=(),
                ignore=True,
                group_name="layers",
            )
        )
        with pytest.raises(PysmeshError) as raised:
            mesher.compute()

    assert "failed at the very first inflation step" in raised.value.details
    assert "NETGEN_2D3D" in raised.value.details


def test_layers_on_a_mesher_with_no_shape_are_refused_before_compute() -> None:
    """NETGEN_3D on a mesh alone takes no layers: the assignment itself is refused."""
    with Mesher() as mesher:
        with pytest.raises(PysmeshError, match="has no shape"):
            mesher.assign(Netgen3D())
        with pytest.raises(PysmeshError, match="has no shape"):
            mesher.assign(
                ViscousLayers(
                    total_thickness=_T,
                    layer_count=_N,
                    stretch_factor=_F,
                    boundary=(1,),
                    group_name="layers",
                )
            )


def test_netgen_grows_two_stacks_on_their_own_faces() -> None:
    """Two ViscousLayers on one solid: T 0.3 on the bottom, T 0.2 (f 1) on face x = 0.

    NETGEN takes several layer hypotheses per solid (NETGENPlugin_NETGEN_3D.cxx,
    CheckHypothesis). Each group holds N prisms per triangle of its own wall, and away
    from the corner each wall keeps its own planes: x = 0.2 k / 3 under the side stack.
    """
    shape = _box()
    bottom, side = _face_at(shape, 2, 0.0), _face_at(shape, 0, 0.0)

    with Mesher(shape) as mesher:
        _netgen_1d2d3d(mesher)
        mesher.assign(
            ViscousLayers(
                total_thickness=_T,
                layer_count=_N,
                stretch_factor=_F,
                boundary=(bottom,),
                group_name="bottom",
            )
        )
        mesher.assign(
            ViscousLayers(
                total_thickness=0.2,
                layer_count=_N,
                stretch_factor=1.0,
                boundary=(side,),
                group_name="side",
            )
        )
        mesher.compute()
        md = mesher.mesh()
        sizes = {g.name: len(g.element_ids) for g in mesher.groups()}
    xyz = md.node_coords
    away = (xyz[:, 2] > _T + _TOL) & (xyz[:, 0] < 0.2 + _TOL)
    total = _check_cells(md, _on_box)

    assert sizes["bottom"] == _N * _on_face(md, bottom, (ElementType.TRIANGLE,))
    assert sizes["side"] == _N * _on_face(md, side, (ElementType.TRIANGLE,))
    assert sorted({round(float(v), 9) for v in xyz[away, 0]}) == pytest.approx(
        [0.0, 0.2 / 3.0, 0.4 / 3.0, 0.2], abs=_TOL
    )
    assert total == pytest.approx(24.0, rel=1e-12)
