"""Viscous-layer end-to-end tests (B3).

Injects the committed classified box surface mesh, grows prism layers, and checks the
result against the analytic geometric-series layer law and prism-orientation invariants.

Layer thickness law (geometric series, ratio ``g``, ``N`` layers, total ``T``):

    h_1 = T * (g - 1) / (g**N - 1)      first-layer height
    h_k = h_1 * g**(k-1)                k-th layer height
    sum_{k=1..N} h_k = T

Reference: standard geometric boundary-layer grading (e.g. Blazek, *CFD Principles and
Applications*, boundary-layer meshing).
"""

from __future__ import annotations

import numpy as np
import pytest

from pysmesh import (
    ExtrusionMethod,
    Mesh,
    PysmeshError,
    Shape,
    VLParams,
    compute_viscous_layers,
    load_brep,
)

T_TOTAL = 0.1
N_LAYERS = 5
GROWTH = 1.2


def _inject(shape: Shape, m: dict[str, np.ndarray]) -> Mesh:
    """Build a fully-injected, classified Mesh from the box surface-mesh arrays."""
    mesh = Mesh(shape)
    sid = mesh.add_nodes(m["nodes"])
    for fid in np.unique(m["face_ids"]):
        sel = m["face_ids"] == fid
        mesh.classify_on_face(sid[m["face_node_ids"][sel]], int(fid), m["face_uv"][sel])
    for eid in np.unique(m["edge_ids"]):
        sel = m["edge_ids"] == eid
        mesh.classify_on_edge(sid[m["edge_node_ids"][sel]], int(eid), m["edge_t"][sel])
    for nl, vid in zip(m["vertex_node_ids"], m["vertex_ids"]):
        mesh.classify_on_vertex(int(sid[nl]), int(vid))
    for eid in np.unique(m["segment_edge_ids"]):
        sel = m["segment_edge_ids"] == eid
        mesh.add_segments(sid[m["segments"][sel]].astype(np.int64), int(eid))
    for fid in np.unique(m["tri_face_ids"]):
        sel = m["tri_face_ids"] == fid
        mesh.add_triangles(sid[m["tris"][sel]].astype(np.int64), int(fid))
    mesh.validate()
    return mesh


def _prism_volumes(prisms: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Signed volume of each VTK wedge via a 3-tetrahedron decomposition."""
    p = coords[prisms]  # (K, 6, 3)

    def tet(a: int, b: int, c: int, d: int) -> np.ndarray:
        va, vb, vc, vd = p[:, a], p[:, b], p[:, c], p[:, d]
        return np.einsum("ij,ij->i", np.cross(vb - va, vc - va), vd - va) / 6.0

    return tet(0, 1, 2, 3) + tet(1, 2, 3, 4) + tet(2, 3, 4, 5)


def _lateral_heights(prisms: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Per-prism layer thickness = length of a lateral edge (cap-0 node -> cap-1 node)."""
    p = coords[prisms]
    return np.linalg.norm(p[:, 0, :] - p[:, 3, :], axis=1)


def test_box_all_walls_prism_count(box_brep: bytes, box_mesh: dict[str, np.ndarray]) -> None:
    shape = load_brep(box_brep)
    mesh = _inject(shape, box_mesh)
    n_wall_tris = box_mesh["tris"].shape[0]

    res = compute_viscous_layers(
        mesh,
        VLParams(
            face_ids=tuple(f.id for f in shape.faces()),
            total_thickness=T_TOTAL,
            n_layers=N_LAYERS,
            stretch_factor=GROWTH,
            group_name="BL",
        ),
    )

    assert res.prism_connectivity.shape == (N_LAYERS * n_wall_tris, 6)
    assert res.failed_face_ids == ()


def test_box_inner_surface_equals_wall_tris(
    box_brep: bytes, box_mesh: dict[str, np.ndarray]
) -> None:
    shape = load_brep(box_brep)
    mesh = _inject(shape, box_mesh)
    n_wall_tris = box_mesh["tris"].shape[0]

    res = compute_viscous_layers(
        mesh,
        VLParams(
            face_ids=tuple(f.id for f in shape.faces()),
            total_thickness=T_TOTAL,
            n_layers=N_LAYERS,
            stretch_factor=GROWTH,
            group_name="BL",
        ),
    )

    # The shrunk inner surface is a copy of every wall triangle, tagged by source face.
    assert res.inner_surface_tris.shape == (n_wall_tris, 3)
    assert res.inner_surface_face_map.shape == (n_wall_tris,)
    assert set(res.inner_surface_face_map.tolist()) == {f.id for f in shape.faces()}


def test_box_first_layer_height_matches_geometric_series(
    box_brep: bytes, box_mesh: dict[str, np.ndarray]
) -> None:
    shape = load_brep(box_brep)
    mesh = _inject(shape, box_mesh)

    res = compute_viscous_layers(
        mesh,
        VLParams(
            face_ids=tuple(f.id for f in shape.faces()),
            total_thickness=T_TOTAL,
            n_layers=N_LAYERS,
            stretch_factor=GROWTH,
            group_name="BL",
        ),
    )

    heights = _lateral_heights(res.prism_connectivity, res.node_coords)
    h1_expected = T_TOTAL * (GROWTH - 1) / (GROWTH**N_LAYERS - 1)
    # The thinnest layer everywhere is the first (undisturbed face-interior columns).
    assert heights.min() == pytest.approx(h1_expected, rel=0.01)


def test_box_growth_ratio_series_present(
    box_brep: bytes, box_mesh: dict[str, np.ndarray]
) -> None:
    shape = load_brep(box_brep)
    mesh = _inject(shape, box_mesh)

    res = compute_viscous_layers(
        mesh,
        VLParams(
            face_ids=tuple(f.id for f in shape.faces()),
            total_thickness=T_TOTAL,
            n_layers=N_LAYERS,
            stretch_factor=GROWTH,
            group_name="BL",
        ),
    )

    heights = _lateral_heights(res.prism_connectivity, res.node_coords)
    h1 = T_TOTAL * (GROWTH - 1) / (GROWTH**N_LAYERS - 1)
    # Every layer of an undisturbed column, h1*g**k, must be realized in the mesh.
    for k in range(N_LAYERS):
        expected = h1 * GROWTH**k
        assert np.any(np.isclose(heights, expected, rtol=0.01)), f"missing layer {k}"


def test_box_zero_inverted_prisms(
    box_brep: bytes, box_mesh: dict[str, np.ndarray]
) -> None:
    shape = load_brep(box_brep)
    mesh = _inject(shape, box_mesh)

    res = compute_viscous_layers(
        mesh,
        VLParams(
            face_ids=tuple(f.id for f in shape.faces()),
            total_thickness=T_TOTAL,
            n_layers=N_LAYERS,
            stretch_factor=GROWTH,
            group_name="BL",
        ),
    )

    vols = _prism_volumes(res.prism_connectivity, res.node_coords)
    # VTK wedge order (normalized in the binding) => strictly positive Jacobian, no tangling.
    assert np.all(vols > 0.0)


def test_box_node_count_is_surface_times_layers(
    box_brep: bytes, box_mesh: dict[str, np.ndarray]
) -> None:
    shape = load_brep(box_brep)
    n_surface_nodes = box_mesh["nodes"].shape[0]
    mesh = _inject(shape, box_mesh)

    res = compute_viscous_layers(
        mesh,
        VLParams(
            face_ids=tuple(f.id for f in shape.faces()),
            total_thickness=T_TOTAL,
            n_layers=N_LAYERS,
            stretch_factor=GROWTH,
            group_name="BL",
        ),
    )

    # All-wall box: every surface node spawns one node per layer.
    assert res.node_coords.shape == (n_surface_nodes * (N_LAYERS + 1), 3)
    assert res.node_ids.shape == (n_surface_nodes * (N_LAYERS + 1),)


def test_is_ignore_complement_equivalence(
    box_brep: bytes, box_mesh: dict[str, np.ndarray]
) -> None:
    shape = load_brep(box_brep)
    all_ids = tuple(f.id for f in shape.faces())
    walls = all_ids[:-1]
    excluded = (all_ids[-1],)

    res_direct = compute_viscous_layers(
        _inject(shape, box_mesh),
        VLParams(face_ids=walls, total_thickness=T_TOTAL, n_layers=N_LAYERS,
                 stretch_factor=GROWTH, group_name="BL"),
    )
    res_ignore = compute_viscous_layers(
        _inject(shape, box_mesh),
        VLParams(face_ids=excluded, is_ignore=True, total_thickness=T_TOTAL,
                 n_layers=N_LAYERS, stretch_factor=GROWTH, group_name="BL"),
    )

    assert res_direct.prism_connectivity.shape == res_ignore.prism_connectivity.shape
    assert set(res_direct.inner_surface_face_map.tolist()) == set(walls)
    assert set(res_ignore.inner_surface_face_map.tolist()) == set(walls)


def test_method_enum_values_are_fixed() -> None:
    # Persisted by SMESH SaveTo/LoadFrom — the order must not drift.
    assert (
        int(ExtrusionMethod.SURF_OFFSET_SMOOTH),
        int(ExtrusionMethod.FACE_OFFSET),
        int(ExtrusionMethod.NODE_OFFSET),
    ) == (0, 1, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_thickness": 0.0},
        {"total_thickness": -1.0},
        {"n_layers": 0},
        {"stretch_factor": 1.0},
        {"group_name": ""},
        {"face_ids": ()},
    ],
)
def test_degenerate_params_raise(kwargs: dict[str, object]) -> None:
    base = dict(
        face_ids=(1,),
        total_thickness=T_TOTAL,
        n_layers=N_LAYERS,
        stretch_factor=GROWTH,
        group_name="BL",
    )
    base.update(kwargs)

    with pytest.raises(PysmeshError):
        VLParams(**base)  # type: ignore[arg-type]


def test_sphere_zero_inverted_prisms(
    sphere_brep: bytes, sphere_mesh: dict[str, np.ndarray]
) -> None:
    # Doubly-curved wall: the real regression for prism tangling/inversion.
    shape = load_brep(sphere_brep)
    mesh = _inject(shape, sphere_mesh)

    res = compute_viscous_layers(
        mesh,
        VLParams(
            face_ids=tuple(f.id for f in shape.faces()),
            total_thickness=0.1,
            n_layers=N_LAYERS,
            stretch_factor=GROWTH,
            group_name="BL",
        ),
    )

    vols = _prism_volumes(res.prism_connectivity, res.node_coords)
    assert vols.size > 0
    assert np.all(vols > 0.0)


def test_sphere_prism_count_and_coverage(
    sphere_brep: bytes, sphere_mesh: dict[str, np.ndarray]
) -> None:
    shape = load_brep(sphere_brep)
    n_face_tris = sphere_mesh["tris"].shape[0]
    mesh = _inject(shape, sphere_mesh)

    res = compute_viscous_layers(
        mesh,
        VLParams(
            face_ids=tuple(f.id for f in shape.faces()),
            total_thickness=0.1,
            n_layers=N_LAYERS,
            stretch_factor=GROWTH,
            group_name="BL",
        ),
    )

    n_inner = res.inner_surface_tris.shape[0]
    # Every inner-surface triangle is capped by exactly n_layers prisms.
    assert res.prism_connectivity.shape[0] == N_LAYERS * n_inner
    # Near-full coverage (a couple of degenerate pole triangles may be dropped).
    assert n_inner >= 0.98 * n_face_tris
    assert res.failed_face_ids == ()


def test_bad_face_id_raises(box_brep: bytes, box_mesh: dict[str, np.ndarray]) -> None:
    shape = load_brep(box_brep)
    mesh = _inject(shape, box_mesh)

    with pytest.raises(PysmeshError, match="999"):
        compute_viscous_layers(
            mesh,
            VLParams(face_ids=(999,), total_thickness=T_TOTAL, n_layers=N_LAYERS,
                     stretch_factor=GROWTH, group_name="BL"),
        )
