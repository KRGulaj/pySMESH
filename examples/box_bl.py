"""End-to-end viscous-layer example (also the CI smoke test and README quickstart).

Loads the committed box BREP, injects its classified surface mesh, grows five prism layers
on every wall, and prints the result summary. Run from the repo root::

    python examples/box_bl.py

Requires the ``pysmesh`` package importable (``src`` on ``PYTHONPATH`` for a dev build) and
the matching host VTK (checked at import).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pysmesh

_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _load(name: str) -> np.ndarray:
    return np.load(_FIXTURES / "box_mesh" / f"{name}.npy")


def main() -> None:
    shape = pysmesh.load_brep((_FIXTURES / "box.brep").read_bytes())
    mesh = pysmesh.Mesh(shape)

    node_ids = mesh.add_nodes(_load("nodes"))
    face_id, face_nid, face_uv = _load("face_ids"), _load("face_node_ids"), _load("face_uv")
    edge_id, edge_nid, edge_t = _load("edge_ids"), _load("edge_node_ids"), _load("edge_t")
    seg_edge, segments = _load("segment_edge_ids"), _load("segments")
    tri_face, tris = _load("tri_face_ids"), _load("tris")
    vert_nid, vert_id = _load("vertex_node_ids"), _load("vertex_ids")

    for fid in np.unique(face_id):
        sel = face_id == fid
        mesh.classify_on_face(node_ids[face_nid[sel]], int(fid), face_uv[sel])
    for eid in np.unique(edge_id):
        sel = edge_id == eid
        mesh.classify_on_edge(node_ids[edge_nid[sel]], int(eid), edge_t[sel])
    for local, vid in zip(vert_nid, vert_id):
        mesh.classify_on_vertex(int(node_ids[local]), int(vid))
    for eid in np.unique(seg_edge):
        sel = seg_edge == eid
        mesh.add_segments(node_ids[segments[sel]].astype(np.int64), int(eid))
    for fid in np.unique(tri_face):
        sel = tri_face == fid
        mesh.add_triangles(node_ids[tris[sel]].astype(np.int64), int(fid))
    mesh.validate()

    params = pysmesh.VLParams(
        face_ids=tuple(f.id for f in shape.faces()),
        total_thickness=0.1,
        n_layers=5,
        stretch_factor=1.2,
        group_name="BL",
    )
    result = pysmesh.compute_viscous_layers(mesh, params)

    print(f"prisms          : {result.prism_connectivity.shape[0]}")  # noqa: T201
    print(f"nodes (total)   : {result.node_coords.shape[0]}")  # noqa: T201
    print(f"inner-surf tris : {result.inner_surface_tris.shape[0]}")  # noqa: T201
    print(f"failed faces    : {result.failed_face_ids}")  # noqa: T201
    mesh.release()


if __name__ == "__main__":
    main()
