"""B1 exit-gate smoke test: the native _core links and runs end to end.

Exercises the full static-link graph (SALOME KERNEL + Geom + SMDS/SMESHDS/SMESH core +
StdMeshers, against static OCCT/Boost-decorated symbols and dynamic OCCT/VTK): construct an
``SMESH_Mesh`` on a unit box, inject one node, and read the node count back through the
SMESHDS API. Removed once Tier-1 bindings (B2) replace the temporary ``create_test_mesh``
helper.
"""

from __future__ import annotations

import _core


def test_create_test_mesh_single_node_returns_1() -> None:
    # Arrange / Act
    n_nodes = _core.create_test_mesh()

    # Assert
    assert n_nodes == 1
