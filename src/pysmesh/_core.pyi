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

class PysmeshCancelled(PysmeshError):
    """An operation the caller stopped through its ``cancel`` predicate.

    A subclass of :class:`PysmeshError`, so code that only cares that the operation did not
    happen keeps catching that. The session is left exactly as it was.
    """

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
    def add_brep(
        self, data: bytes, progress: object, cancel: object
    ) -> dict[str, object]: ...
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
    def add_cone(
        self,
        radius1: float,
        radius2: float,
        height: float,
        ox: float,
        oy: float,
        oz: float,
        ax: float,
        ay: float,
        az: float,
        angle_rad: float,
    ) -> dict[str, object]: ...
    def add_sphere(
        self,
        radius: float,
        cx: float,
        cy: float,
        cz: float,
        ax: float,
        ay: float,
        az: float,
        angle_rad: float,
    ) -> dict[str, object]: ...
    def add_torus(
        self,
        radius1: float,
        radius2: float,
        ox: float,
        oy: float,
        oz: float,
        ax: float,
        ay: float,
        az: float,
        angle_rad: float,
    ) -> dict[str, object]: ...
    def add_wedge(
        self,
        dx: float,
        dy: float,
        dz: float,
        ltx: float,
        ox: float,
        oy: float,
        oz: float,
        ax: float,
        ay: float,
        az: float,
    ) -> dict[str, object]: ...
    def add_vertex(self, x: float, y: float, z: float) -> dict[str, object]: ...
    def add_line(
        self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
    ) -> dict[str, object]: ...
    def add_arc(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        x3: float,
        y3: float,
        z3: float,
    ) -> dict[str, object]: ...
    def add_circle(
        self,
        cx: float,
        cy: float,
        cz: float,
        nx: float,
        ny: float,
        nz: float,
        radius: float,
    ) -> dict[str, object]: ...
    def add_ellipse(
        self,
        cx: float,
        cy: float,
        cz: float,
        nx: float,
        ny: float,
        nz: float,
        rx: float,
        ry: float,
        x_dir: tuple[float, float, float] | None,
    ) -> dict[str, object]: ...
    def add_polyline(
        self, points: NDArray[np.float64], closed: bool
    ) -> dict[str, object]: ...
    def add_spline(
        self,
        points: NDArray[np.float64],
        degree_min: int,
        degree_max: int,
        tol: float,
    ) -> dict[str, object]: ...
    def add_bspline(
        self, poles: NDArray[np.float64], degree: int
    ) -> dict[str, object]: ...
    def add_helix(
        self,
        cx: float,
        cy: float,
        cz: float,
        ax: float,
        ay: float,
        az: float,
        diameter: float,
        pitch: float,
        turns: float,
        tol: float,
    ) -> dict[str, object]: ...
    def add_rectangle(
        self,
        ox: float,
        oy: float,
        oz: float,
        nx: float,
        ny: float,
        nz: float,
        dx: float,
        dy: float,
    ) -> dict[str, object]: ...
    def make_wire(self, edge_ids: list[int]) -> dict[str, object]: ...
    def make_face(self, edge_ids: list[int]) -> dict[str, object]: ...
    def make_filling(
        self, edge_ids: list[int], progress: object, cancel: object
    ) -> dict[str, object]: ...
    def extrude(
        self, entity_ids: list[int], vx: float, vy: float, vz: float
    ) -> dict[str, object]: ...
    def revolve(
        self,
        entity_ids: list[int],
        ox: float,
        oy: float,
        oz: float,
        ax: float,
        ay: float,
        az: float,
        angle_rad: float,
    ) -> dict[str, object]: ...
    def pipe(
        self,
        spine_ids: list[int],
        profile_ids: list[int],
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def pipe_shell(
        self,
        spine_ids: list[int],
        profile_ids: list[int],
        frenet: bool,
        solid: bool,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def thru_sections(
        self,
        sections: list[list[int]],
        solid: bool,
        ruled: bool,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def fuse(
        self,
        targets: list[int],
        tools: list[int],
        fuzzy: float,
        parallel: bool,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def cut(
        self,
        targets: list[int],
        tools: list[int],
        fuzzy: float,
        parallel: bool,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def common(
        self,
        targets: list[int],
        tools: list[int],
        fuzzy: float,
        parallel: bool,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def section(
        self,
        targets: list[int],
        tools: list[int],
        fuzzy: float,
        parallel: bool,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def split(
        self,
        targets: list[int],
        tools: list[int],
        fuzzy: float,
        parallel: bool,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def fragment(
        self,
        entity_ids: list[int],
        fuzzy: float,
        parallel: bool,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def fillet(
        self,
        edge_ids: list[int],
        radius: float,
        radius_end: float | None,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def chamfer(
        self,
        edge_ids: list[int],
        distance: float,
        distance_end: float | None,
        face_id: int | None,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
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
    def mirror(
        self,
        px: float,
        py: float,
        pz: float,
        nx: float,
        ny: float,
        nz: float,
        entity_ids: list[int] | None,
    ) -> dict[str, object]: ...
    def scale(
        self,
        sx: float,
        sy: float,
        sz: float,
        cx: float,
        cy: float,
        cz: float,
        entity_ids: list[int] | None,
    ) -> dict[str, object]: ...
    def copy(self, entity_ids: list[int]) -> dict[str, object]: ...
    def heal(
        self,
        entity_ids: list[int] | None,
        precision: float,
        min_tolerance: float,
        max_tolerance: float,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def sew(
        self,
        entity_ids: list[int],
        tolerance: float,
        make_solid: bool,
        non_manifold: bool,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def remove_internal_wires(
        self, entity_ids: list[int] | None, min_area: float, remove_faces: bool
    ) -> dict[str, object]: ...
    def defeature(
        self,
        face_ids: list[int],
        parallel: bool,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def imprint(
        self,
        targets: list[int],
        tools: list[int],
        fuzzy: float,
        parallel: bool,
        glue: int,
        progress: object,
        cancel: object,
    ) -> dict[str, object]: ...
    def remove(self, entity_ids: list[int]) -> dict[str, object]: ...
    def unify_same_domain(
        self,
        entity_ids: list[int] | None,
        unify_faces: bool,
        unify_edges: bool,
        concat_bsplines: bool,
        linear_tol: float,
        angular_tol_rad: float,
    ) -> dict[str, object]: ...
    def entity_types(self, kind: str) -> dict[str, object]: ...
    def bounding_boxes(self, kind: str) -> dict[str, object]: ...
    def mass_properties(self, entity_ids: list[int]) -> dict[str, object]: ...
    def face_parameter_bounds(self, face_ids: list[int]) -> NDArray[np.float64]: ...
    def edge_parameter_bounds(self, edge_ids: list[int]) -> NDArray[np.float64]: ...
    def adjacency(self, kind: str, other_kind: str) -> dict[str, object]: ...
    def surface_at(
        self, face_id: int, uv: NDArray[np.float64]
    ) -> dict[str, object]: ...
    def curvature(
        self, face_ids: list[int], samples: int
    ) -> dict[str, object]: ...
    def project_on_face(
        self, face_id: int, points: NDArray[np.float64]
    ) -> dict[str, object]: ...
    def entities_in_box(
        self,
        kind: str,
        xmin: float,
        ymin: float,
        zmin: float,
        xmax: float,
        ymax: float,
        zmax: float,
        strict: bool,
    ) -> NDArray[np.int64]: ...
    def contains(
        self, solid_ids: list[int], points: NDArray[np.float64], tol: float
    ) -> NDArray[np.bool_]: ...
    def tessellate(
        self,
        deflection: float,
        angle_rad: float,
        relative: bool,
        parallel: bool,
        incremental: bool,
        progress: object,
        cancel: object,
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
    def export_handoff(self) -> dict[str, object]: ...
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
