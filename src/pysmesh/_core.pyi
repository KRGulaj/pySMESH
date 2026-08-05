"""Type stubs for the pySMESH native ``_core`` extension (Tier-1).

The host application runs ``mypy --strict`` against these; keep signatures exact.
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
    surface_type: str  # Plane/Cylinder/Cone/Sphere/Torus/BSpline/...
    @property
    def centroid(self) -> NDArray[np.float64]: ...  # (3,)
    @property
    def bbox(self) -> NDArray[np.float64]: ...  # (6,) xmin,ymin,zmin,xmax,ymax,zmax
    @property
    def uv_bounds(self) -> NDArray[np.float64]: ...  # (4,) umin,umax,vmin,vmax

class SolidInfo:
    id: int
    volume: float
    @property
    def centroid(self) -> NDArray[np.float64]: ...  # (3,)
    @property
    def bbox(self) -> NDArray[np.float64]: ...  # (6,) xmin,ymin,zmin,xmax,ymax,zmax

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
    def solids(self) -> list[SolidInfo]: ...
    def faces(self) -> list[FaceInfo]: ...
    def edges(self) -> list[EdgeInfo]: ...
    def vertices(self) -> list[VertexInfo]: ...
    def face_distance(
        self, face_id: int, points: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
    def face_adjacency(self) -> list[tuple[int, int, int]]: ...
    def match_faces(
        self, centroids: NDArray[np.float64], tol: float
    ) -> NDArray[np.int32]: ...

class Session:
    """Stateful modelling context. ``pysmesh.session.Session`` is the typed wrapper."""

    def __init__(self, validate: bool) -> None: ...
    def add_brep(self, data: bytes) -> dict[str, object]: ...
    def add_box(
        self, dx: float, dy: float, dz: float, ox: float, oy: float, oz: float
    ) -> dict[str, object]: ...
    def add_cylinder(
        self,
        radius: float,
        height: float,
        ox: float,
        oy: float,
        oz: float,
        ax: float,
        ay: float,
        az: float,
    ) -> dict[str, object]: ...
    def fuse(
        self,
        targets: list[int],
        tools: list[int],
        fuzzy: float,
        parallel: bool,
    ) -> dict[str, object]: ...
    def fillet(self, edge_ids: list[int], radius: float) -> dict[str, object]: ...
    def translate(
        self, dx: float, dy: float, dz: float, entity_ids: list[int] | None
    ) -> dict[str, object]: ...
    def rotate(
        self,
        ox: float,
        oy: float,
        oz: float,
        ax: float,
        ay: float,
        az: float,
        angle_rad: float,
        entity_ids: list[int] | None,
    ) -> dict[str, object]: ...
    def snapshot(self) -> int: ...
    def restore(self, mark: int) -> None: ...
    def discard_snapshot(self, mark: int) -> None: ...
    def snapshot_count(self) -> int: ...
    def entities(self, kind: str) -> NDArray[np.int64]: ...
    def entity_kind(self, entity_id: int) -> str: ...
    def entity_state(self, entity_id: int) -> str: ...
    def shape_count(self, entity_id: int) -> int: ...
    def entity_table(self, kind: str) -> dict[str, object]: ...
    def brep(self) -> bytes: ...
    def name_of(self, entity_id: int) -> dict[str, object]: ...
    def origin(self, entity_id: int) -> dict[str, object]: ...
    def resolve(self, op_index: int, role: int, ordinal: int) -> dict[str, object]: ...
    def op_count(self) -> int: ...
    def state_op_index(self) -> int: ...
    def issued_id_count(self) -> int: ...
    def entity_count(self) -> int: ...
    def _debug_tear_next_history(self) -> None: ...

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
def point_in_solid(
    brep: bytes, points: NDArray[np.float64], tol: float
) -> NDArray[np.bool_]: ...
def read_step_xde(data_or_path: bytes | str) -> dict[str, object]: ...
def write_step_xde(
    brep: bytes,
    face_names: dict[int, str],
    face_colors: dict[int, tuple[float, float, float]],
    name: str,
) -> bytes: ...
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
