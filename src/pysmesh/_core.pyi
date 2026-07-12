"""Type stubs for the pySMESH native ``_core`` extension (Tier-1).

flux runs ``mypy --strict`` against these; keep signatures exact.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

class PysmeshError(RuntimeError):
    """All library failures.

    Attributes:
        details: SMESH ``SMESH_ComputeError`` text / OCCT message, when available.
        face_ids: The offending face ids, where applicable (empty otherwise).
    """

    details: str
    face_ids: list[int]

class FaceInfo:
    id: int
    area: float
    @property
    def centroid(self) -> NDArray[np.float64]: ...  # (3,)
    @property
    def bbox(self) -> NDArray[np.float64]: ...  # (6,) xmin,ymin,zmin,xmax,ymax,zmax
    @property
    def uv_bounds(self) -> NDArray[np.float64]: ...  # (4,) umin,umax,vmin,vmax

class EdgeInfo:
    id: int
    length: float
    @property
    def bbox(self) -> NDArray[np.float64]: ...  # (6,)
    @property
    def t_bounds(self) -> NDArray[np.float64]: ...  # (2,) first,last

class VertexInfo:
    id: int
    @property
    def xyz(self) -> NDArray[np.float64]: ...  # (3,)

class Shape:
    def faces(self) -> list[FaceInfo]: ...
    def edges(self) -> list[EdgeInfo]: ...
    def vertices(self) -> list[VertexInfo]: ...
    def face_distance(
        self, face_id: int, points: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

class MeshStats:
    n_nodes: int
    n_faces: int
    @property
    def per_face_element_counts(self) -> dict[int, int]: ...

class Mesh:
    def __init__(self, shape: Shape) -> None: ...
    def add_nodes(self, coords: NDArray[np.float64]) -> NDArray[np.int64]: ...
    def classify_on_face(
        self, node_ids: NDArray[np.int64], face_id: int, uv: NDArray[np.float64]
    ) -> None: ...
    def classify_on_edge(
        self, node_ids: NDArray[np.int64], edge_id: int, t: NDArray[np.float64]
    ) -> None: ...
    def classify_on_vertex(self, node_id: int, vertex_id: int) -> None: ...
    def add_segments(self, conn: NDArray[np.int64], edge_id: int) -> None: ...
    def add_triangles(self, conn: NDArray[np.int64], face_id: int) -> None: ...
    def validate(self) -> None: ...
    def stats(self) -> MeshStats: ...
    def release(self) -> None: ...
    def __enter__(self) -> Mesh: ...
    def __exit__(self, *args: object) -> None: ...

def load_brep(data: bytes) -> Shape: ...
def make_thick_solid(
    brep: bytes,
    remove_face_ids: list[int],
    thickness: float,
    tol: float,
) -> dict[str, object]: ...
def offset_shape(
    brep: bytes,
    offset: float,
    tol: float,
) -> dict[str, object]: ...
def shape_distance(brep_a: bytes, brep_b: bytes) -> dict[str, object]: ...
def free_boundary_edges(brep: bytes) -> NDArray[np.int32]: ...
def tessellate(
    brep: bytes,
    lin_defl: float,
    ang_defl: float,
    relative: bool = ...,
) -> dict[str, object]: ...
def compute_viscous_layers(
    mesh: Mesh,
    face_ids: list[int],
    is_ignore: bool,
    total_thickness: float,
    n_layers: int,
    stretch_factor: float,
    method: int,
    group_name: str,
) -> dict[str, object]: ...
def unify_same_domain(
    brep: bytes,
    unify_faces: bool,
    unify_edges: bool,
    concat_bsplines: bool,
    linear_tol: float,
    angular_tol_rad: float,
) -> dict[str, object]: ...
