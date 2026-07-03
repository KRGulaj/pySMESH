"""Tier-1 surface-mesh injection tests: node/element insertion, classification,
validate(), stats().

Uses the committed box fixture (6 faces). A fully-valid injection puts one triangle on
every face with all nodes classified onto their face, so validate() passes; the negative
tests then remove one classification or one face's elements and assert validate() names
the gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from pysmesh import Mesh, PysmeshError, Shape, load_brep

N_BOX_FACES = 6
NODES_PER_FACE = 3


def _inject_face(mesh: Mesh, face_id: int) -> np.ndarray:
    """Add three distinct nodes on a face, classify them, add one triangle. Returns ids."""
    # Distinct coords per face (geometry is irrelevant to classification, which is by uv).
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    ) + float(face_id)
    node_ids = mesh.add_nodes(coords)
    mesh.classify_on_face(node_ids, face_id, np.zeros((NODES_PER_FACE, 2), np.float64))
    mesh.add_triangles(node_ids.reshape(1, 3), face_id)
    return node_ids


def _build_valid_box_mesh(box_brep: bytes) -> tuple[Shape, Mesh, list[int]]:
    shape = load_brep(box_brep)
    mesh = Mesh(shape)
    face_ids = [f.id for f in shape.faces()]
    for fid in face_ids:
        _inject_face(mesh, fid)
    return shape, mesh, face_ids


def test_add_nodes_returns_unique_int64_ids(box_brep: bytes) -> None:
    shape = load_brep(box_brep)
    mesh = Mesh(shape)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)

    ids = mesh.add_nodes(coords)

    assert ids.dtype == np.int64
    assert ids.shape == (2,)
    assert len(np.unique(ids)) == 2


def test_stats_round_trips_node_and_element_counts(box_brep: bytes) -> None:
    _shape, mesh, face_ids = _build_valid_box_mesh(box_brep)

    stats = mesh.stats()

    assert stats.n_nodes == N_BOX_FACES * NODES_PER_FACE  # 18
    assert stats.n_faces == N_BOX_FACES  # one triangle per face
    assert stats.per_face_element_counts == {fid: 1 for fid in face_ids}


def test_validate_passes_on_fully_classified_mesh(box_brep: bytes) -> None:
    _shape, mesh, _face_ids = _build_valid_box_mesh(box_brep)

    mesh.validate()  # must not raise


def test_validate_names_unclassified_node(box_brep: bytes) -> None:
    _shape, mesh, _face_ids = _build_valid_box_mesh(box_brep)
    stray = mesh.add_nodes(np.array([[9.0, 9.0, 9.0]], dtype=np.float64))  # not classified

    with pytest.raises(PysmeshError, match=r"unclassified node ids.*\b%d\b" % stray[0]):
        mesh.validate()


def test_validate_names_face_without_elements(box_brep: bytes) -> None:
    shape = load_brep(box_brep)
    mesh = Mesh(shape)
    face_ids = [f.id for f in shape.faces()]
    skipped = face_ids[-1]
    for fid in face_ids:
        if fid != skipped:
            _inject_face(mesh, fid)

    with pytest.raises(PysmeshError, match=r"face_ids with no elements.*\b%d\b" % skipped):
        mesh.validate()


def test_add_triangles_bad_face_id_raises(box_brep: bytes) -> None:
    shape = load_brep(box_brep)
    mesh = Mesh(shape)
    node_ids = mesh.add_nodes(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    )

    with pytest.raises(PysmeshError, match="999"):
        mesh.add_triangles(node_ids.reshape(1, 3), 999)


def test_add_triangles_unknown_node_id_raises(box_brep: bytes) -> None:
    shape = load_brep(box_brep)
    mesh = Mesh(shape)

    with pytest.raises(PysmeshError, match=r"node id"):
        mesh.add_triangles(np.array([[1, 2, 999999]], dtype=np.int64), 1)


def test_release_makes_further_ops_raise(box_brep: bytes) -> None:
    shape = load_brep(box_brep)
    mesh = Mesh(shape)
    mesh.release()

    with pytest.raises(PysmeshError, match="released"):
        mesh.add_nodes(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))


def test_context_manager_releases_on_exit(box_brep: bytes) -> None:
    shape = load_brep(box_brep)

    with Mesh(shape) as mesh:
        mesh.add_nodes(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))

    with pytest.raises(PysmeshError, match="released"):
        mesh.stats()
