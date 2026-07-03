"""End-to-end same-domain healing example (also a CI smoke test and README quickstart).

Loads the committed ``split_box`` BREP — two fused cubes whose four coplanar seam pairs give
a plain rectangular block 10 faces instead of 6 — heals it with ``unify_same_domain``, and
prints the before/after summary plus the old->new face-id composition map. Run from the repo
root::

    python examples/split_box_unify.py

Requires the ``pysmesh`` package importable (``src`` on ``PYTHONPATH`` for a dev build) and
the matching host VTK (checked at import).
"""

from __future__ import annotations

from pathlib import Path

import pysmesh

_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main() -> None:
    brep = (_FIXTURES / "split_box.brep").read_bytes()

    result = pysmesh.unify_same_domain(
        brep,
        pysmesh.UnifyParams(linear_tol=1e-6, angular_tol_deg=0.5),
    )

    print(f"faces : {result.n_faces_before} -> {result.n_faces_after}")  # noqa: T201
    print(f"edges : {result.n_edges_before} -> {result.n_edges_after}")  # noqa: T201
    # old 1-based face id -> new id (-1 if removed); merged faces share a survivor.
    face_map = {i + 1: int(result.face_map[i]) for i in range(result.n_faces_before)}
    print(f"face_map : {face_map}")  # noqa: T201

    healed = pysmesh.load_brep(result.brep)
    assert len(healed.faces()) == result.n_faces_after  # round-trips


if __name__ == "__main__":
    main()
