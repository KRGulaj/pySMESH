"""Gates for named mesh groups, and for the three editing operations one has to survive.

The claim under test is the one that makes a group worth having: **the mesher maintains
membership across editing, so a consumer does not have to re-derive it**. Re-deriving face
membership after each refinement is exactly the step that goes wrong in a geometry-to-solver
handoff, and a group that quietly went stale would be worse than no group at all.

The membership is therefore never checked against the group itself. Each check recomputes it
from the harvest arrays by a geometric rule the group knows nothing about — *the cells whose
centroid lies in the lower half of the box* — and compares. The rule survives all three edits
by construction: converting to second order moves no corner, the split is vertical, and a
merge only removes cells. The falsification is asserted too: the same check against the upper
half must **disagree**, so a check that passed vacuously would be visible.

Three edits are covered because each breaks the correspondence differently — one keeps ids,
one replaces elements, one deletes them.

Fixture sizing follows the project rule: 3 x 7 x 11, never a unit cube.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

import pysmesh as ps
from pysmesh import (
    BadOrientedVolume,
    ElementDimension,
    ElementType,
    GroupSource,
    Hexa3D,
    LessThan,
    MeshData,
    Mesher,
    MoreThan,
    NumberOfSegments,
    PysmeshError,
    Quadrangle2D,
    Regular1D,
    Session,
    SplitMethod,
    SubShape,
    SubShapeKind,
    Volume,
)

BOX_DX: float = 3.0
BOX_DY: float = 7.0
BOX_DZ: float = 11.0
SEGMENTS: int = 2


# ---- Fixtures ---------------------------------------------------------------------------- #


@pytest.fixture()
def box_mesher() -> Iterator[Mesher]:
    """A structured hexahedral mesh of the 3 x 7 x 11 box: 2 segments an edge, 8 cells."""
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    with Mesher(ps.load_brep(session.brep())) as mesher:
        mesher.assign(Regular1D())
        mesher.assign(NumberOfSegments(count=SEGMENTS))
        mesher.assign(Quadrangle2D())
        mesher.assign(Hexa3D())
        mesher.compute()
        yield mesher


def _cell_centroids(mesh: MeshData) -> dict[int, NDArray[np.float64]]:
    """Every volume cell's centroid, keyed by mesh id, taken from the harvest alone.

    A quadratic cell's corner nodes come first in its connectivity, so the first eight rows
    are the corners of a hexahedron whatever its order. Taking the corners rather than every
    node keeps the rule the same before and after a conversion to second order.
    """
    centroids: dict[int, NDArray[np.float64]] = {}
    for row in range(mesh.element_count):
        kind = int(mesh.element_type[row])
        if kind not in (
            int(ElementType.HEXAHEDRON),
            int(ElementType.QUAD_HEXAHEDRON),
            int(ElementType.TRIQUAD_HEXAHEDRON),
            int(ElementType.PENTAHEDRON),
            int(ElementType.TETRAHEDRON),
            int(ElementType.PYRAMID),
        ):
            continue
        corners = {
            int(ElementType.HEXAHEDRON): 8,
            int(ElementType.QUAD_HEXAHEDRON): 8,
            int(ElementType.TRIQUAD_HEXAHEDRON): 8,
            int(ElementType.PENTAHEDRON): 6,
            int(ElementType.TETRAHEDRON): 4,
            int(ElementType.PYRAMID): 5,
        }[kind]
        rows = [int(n) for n in mesh.nodes_of(row)[:corners]]
        centroids[int(mesh.element_id[row])] = mesh.node_coords[rows].mean(axis=0)
    return centroids


def _lower_half(mesh: MeshData) -> set[int]:
    """The independent oracle: the ids of the cells whose centroid is below mid-height."""
    return {
        element_id
        for element_id, centroid in _cell_centroids(mesh).items()
        if float(centroid[2]) < BOX_DZ / 2.0
    }


def _upper_half(mesh: MeshData) -> set[int]:
    """The falsification's oracle: the cells the group is deliberately not made of."""
    return {
        element_id
        for element_id, centroid in _cell_centroids(mesh).items()
        if float(centroid[2]) > BOX_DZ / 2.0
    }


def _members(mesher: Mesher, name: str) -> set[int]:
    """A group's membership as a set of mesh ids."""
    return {int(i) for i in mesher.group(name).element_ids}


# ---- Creating, reading and removing ----------------------------------------------------- #


def test_an_explicit_group_holds_exactly_the_ids_it_was_given(box_mesher: Mesher) -> None:
    """The base case, and the shape everything else is checked against."""
    chosen = sorted(_lower_half(box_mesher.mesh()))

    box_mesher.add_group("floor", ElementDimension.VOLUME, chosen)

    group = box_mesher.group("floor")
    assert group.source is GroupSource.EXPLICIT
    assert group.dimension is ElementDimension.VOLUME
    assert sorted(group.element_ids.tolist()) == chosen
    assert box_mesher.group_names() == ("floor",)


def test_a_second_group_of_the_same_name_is_refused(box_mesher: Mesher) -> None:
    """Names address a group here, so a duplicate would make every later call ambiguous."""
    box_mesher.add_group("floor", ElementDimension.VOLUME, [])

    with pytest.raises(PysmeshError, match="already exists"):
        box_mesher.add_group("floor", ElementDimension.FACE, [])


def test_an_id_of_the_wrong_family_is_refused_naming_it(box_mesher: Mesher) -> None:
    """A face id put into a volume group would otherwise be silently dropped."""
    mesh = box_mesher.mesh()
    a_face = int(mesh.element_id[mesh.element_type == int(ElementType.QUADRANGLE)][0])

    with pytest.raises(PysmeshError, match=str(a_face)):
        box_mesher.add_group("floor", ElementDimension.VOLUME, [a_face])


def test_a_group_can_be_edited_by_hand_in_both_directions(box_mesher: Mesher) -> None:
    """Adding and removing ids, with the membership read back each time."""
    chosen = sorted(_lower_half(box_mesher.mesh()))
    box_mesher.add_group("floor", ElementDimension.VOLUME, chosen[:2])

    box_mesher.add_to_group("floor", chosen[2:])
    grown = _members(box_mesher, "floor")
    box_mesher.remove_from_group("floor", chosen[:1])
    shrunk = _members(box_mesher, "floor")

    assert grown == set(chosen)
    assert shrunk == set(chosen[1:])


def test_removing_a_group_leaves_its_elements_alone(box_mesher: Mesher) -> None:
    """A group names elements; deleting the name must not delete what it named."""
    before = box_mesher.mesh().element_count
    floor = sorted(_lower_half(box_mesher.mesh()))
    box_mesher.add_group("floor", ElementDimension.VOLUME, floor)

    box_mesher.remove_group("floor")

    assert box_mesher.group_names() == ()
    assert box_mesher.mesh().element_count == before


def test_an_unknown_group_name_is_refused_rather_than_answered(box_mesher: Mesher) -> None:
    """Both the read and the delete, so neither invents an empty group."""
    with pytest.raises(PysmeshError, match="no group named"):
        box_mesher.group("nowhere")
    with pytest.raises(PysmeshError, match="no group named"):
        box_mesher.remove_group("nowhere")


# ---- Groups defined by a source --------------------------------------------------------- #


def test_a_group_on_a_sub_shape_holds_what_the_mesher_bound_there(box_mesher: Mesher) -> None:
    """Checked against the harvest's own CAD binding, which the group does not consult."""
    mesh = box_mesher.mesh()
    expected = {
        int(mesh.element_id[row])
        for row in range(mesh.element_count)
        if int(mesh.element_kind[row]) == int(SubShapeKind.FACE)
        and int(mesh.element_ordinal[row]) == 2
    }

    box_mesher.add_group_on_shape(
        "inlet", ElementDimension.FACE, SubShape(SubShapeKind.FACE, 2)
    )

    group = box_mesher.group("inlet")
    assert group.source is GroupSource.SHAPE
    assert set(group.element_ids.tolist()) == expected
    assert expected


def test_a_group_on_a_filter_holds_what_the_same_predicate_selects(
    box_mesher: Mesher,
) -> None:
    """Two routes to one answer: the group and the selection must agree exactly."""
    cell = float(box_mesher.quality(Volume()).values[0])
    predicate = MoreThan(control=Volume(), margin=cell / 2.0)
    expected = set(box_mesher.select(predicate).ids.tolist())

    box_mesher.add_group_on_filter("sound", ElementDimension.VOLUME, predicate)

    group = box_mesher.group("sound")
    assert group.source is GroupSource.FILTER
    assert set(group.element_ids.tolist()) == expected
    assert len(expected) == SEGMENTS**3


def test_a_filtered_group_is_empty_when_its_predicate_accepts_nothing(
    box_mesher: Mesher,
) -> None:
    """The falsification for the one above: a group that always filled would pass it."""
    cell = float(box_mesher.quality(Volume()).values[0])

    box_mesher.add_group_on_filter(
        "impossible", ElementDimension.VOLUME, LessThan(control=Volume(), margin=cell / 2.0)
    )

    assert box_mesher.group("impossible").element_ids.size == 0


@pytest.mark.parametrize("source", ["shape", "filter"])
def test_a_group_defined_by_its_source_refuses_a_hand_edit(
    box_mesher: Mesher, source: str
) -> None:
    """Refused naming the source, rather than accepted and then overwritten."""
    if source == "shape":
        box_mesher.add_group_on_shape(
            "named", ElementDimension.FACE, SubShape(SubShapeKind.FACE, 1)
        )
    else:
        box_mesher.add_group_on_filter(
            "named", ElementDimension.VOLUME, BadOrientedVolume()
        )
    a_cell = int(box_mesher.quality(Volume()).element_ids[0])

    with pytest.raises(PysmeshError, match="cannot be edited by hand"):
        box_mesher.add_to_group("named", [a_cell])


def test_a_filtered_group_re_evaluates_after_the_mesh_changes(box_mesher: Mesher) -> None:
    """The point of a filtered group: it follows the mesh rather than recording one moment.

    Splitting every cell in two leaves twice as many cells, all still of positive volume, and
    the group must say so without being asked to refresh.
    """
    box_mesher.add_group_on_filter(
        "positive", ElementDimension.VOLUME, MoreThan(control=Volume(), margin=0.0)
    )
    before = box_mesher.group("positive").element_ids.size

    report = box_mesher.split_volumes(SplitMethod.HEXA_TO_2_PRISMS)

    assert before == SEGMENTS**3
    assert report.volumes_after == 2 * report.volumes_before
    assert box_mesher.group("positive").element_ids.size == report.volumes_after


# ---- The gate: survival across the three edits ------------------------------------------ #


def test_a_group_survives_conversion_to_second_order_with_its_membership_correct(
    box_mesher: Mesher,
) -> None:
    """Element ids are preserved, so the same cells must still be named."""
    floor = sorted(_lower_half(box_mesher.mesh()))
    box_mesher.add_group("floor", ElementDimension.VOLUME, floor)
    before = _members(box_mesher, "floor")

    box_mesher.convert_to_quadratic()

    mesh = box_mesher.mesh()
    assert mesh.count_of(ElementType.QUAD_HEXAHEDRON) == SEGMENTS**3
    assert mesh.count_of(ElementType.HEXAHEDRON) == 0
    assert _members(box_mesher, "floor") == _lower_half(mesh)
    assert _members(box_mesher, "floor") == before


def test_a_group_survives_a_volume_split_with_its_membership_correct(
    box_mesher: Mesher,
) -> None:
    """Each cell is replaced by two, and the group must follow the replacement."""
    floor = sorted(_lower_half(box_mesher.mesh()))
    box_mesher.add_group("floor", ElementDimension.VOLUME, floor)
    before = _members(box_mesher, "floor")

    report = box_mesher.split_volumes(SplitMethod.HEXA_TO_2_PRISMS)

    mesh = box_mesher.mesh()
    after = _members(box_mesher, "floor")
    assert report.volumes_after == 2 * report.volumes_before
    assert after == _lower_half(mesh)
    assert len(after) == 2 * len(before)


def test_a_group_survives_a_node_merge_and_never_names_a_deleted_element(
    box_mesher: Mesher,
) -> None:
    """A merge deletes the cells that collapse, and a group must drop exactly those.

    A group still naming a deleted element is the failure that would corrupt a solver
    handoff, so it is asserted directly against the ids the mesh still holds.
    """
    floor = sorted(_lower_half(box_mesher.mesh()))
    box_mesher.add_group("floor", ElementDimension.VOLUME, floor)
    before = _members(box_mesher, "floor")

    report = box_mesher.merge_nodes(tolerance=BOX_DX / SEGMENTS + 0.1)

    mesh = box_mesher.mesh()
    alive = {int(i) for i in mesh.element_id}
    after = _members(box_mesher, "floor")
    assert report.groups_merged > 0
    assert report.elements_after < report.elements_before
    assert after <= alive
    assert after < before


def test_a_conforming_mesh_merges_nothing(box_mesher: Mesher) -> None:
    """The control for the merge gate: nothing coincident, nothing removed."""
    before = box_mesher.mesh().element_count

    report = box_mesher.merge_nodes(tolerance=1e-9)

    assert report.groups_merged == 0
    assert report.nodes_after == report.nodes_before
    assert box_mesher.mesh().element_count == before


def test_the_membership_check_can_fail(box_mesher: Mesher) -> None:
    """The falsification. Without it every survival test above could be passing vacuously."""
    mesh = box_mesher.mesh()
    box_mesher.add_group("floor", ElementDimension.VOLUME, sorted(_lower_half(mesh)))

    box_mesher.convert_to_quadratic()

    after = box_mesher.mesh()
    assert _members(box_mesher, "floor") == _lower_half(after)
    assert _members(box_mesher, "floor") != _upper_half(after)
    assert _lower_half(after) and _upper_half(after)


def test_a_group_of_nodes_survives_conversion_to_second_order(box_mesher: Mesher) -> None:
    """The medium nodes are new, so a node group must not silently gain them."""
    mesh = box_mesher.mesh()
    floor_nodes = sorted(
        int(mesh.node_id[row])
        for row in range(mesh.node_count)
        if float(mesh.node_coords[row][2]) == 0.0
    )
    box_mesher.add_group("floor_nodes", ElementDimension.NODE, floor_nodes)

    box_mesher.convert_to_quadratic()

    after = box_mesher.mesh()
    assert after.node_count > mesh.node_count
    assert sorted(box_mesher.group("floor_nodes").element_ids.tolist()) == floor_nodes


# ---- The editing operations in their own right ------------------------------------------ #


def test_conversion_to_second_order_round_trips_back_to_first(box_mesher: Mesher) -> None:
    """Counts and ids either side, so the conversion is proved reversible rather than lossy."""
    before = box_mesher.mesh()
    ids_before = sorted(before.element_id.tolist())

    box_mesher.convert_to_quadratic()
    quadratic = box_mesher.mesh()
    converted = box_mesher.convert_from_quadratic()
    linear = box_mesher.mesh()

    assert quadratic.node_count > before.node_count
    assert quadratic.element_count == before.element_count
    assert converted is True
    assert linear.node_count == before.node_count
    assert sorted(linear.element_id.tolist()) == ids_before


def test_converting_a_linear_mesh_from_quadratic_changes_nothing(
    box_mesher: Mesher,
) -> None:
    """Pinned because the return value looks like an answer and is not one.

    SMESH returns True whether or not it found anything quadratic, so the counts are the only
    way to tell a real conversion from a walk that found nothing. This binding forwards the
    value rather than inventing a better one.
    """
    before = box_mesher.mesh()

    reported = box_mesher.convert_from_quadratic()

    after = box_mesher.mesh()
    assert reported is True
    assert after.node_count == before.node_count
    assert after.element_count == before.element_count


@pytest.mark.parametrize(
    ("method", "factor"),
    [
        (SplitMethod.HEXA_TO_5, 5),
        (SplitMethod.HEXA_TO_6, 6),
        (SplitMethod.HEXA_TO_24, 24),
        (SplitMethod.HEXA_TO_2_PRISMS, 2),
        (SplitMethod.HEXA_TO_4_PRISMS, 4),
    ],
)
def test_each_split_method_produces_the_cell_count_its_name_promises(
    box_mesher: Mesher, method: SplitMethod, factor: int
) -> None:
    """The name is the contract; a method that split differently would be a silent surprise."""
    report = box_mesher.split_volumes(method)

    assert report.volumes_before == SEGMENTS**3
    assert report.volumes_after == factor * report.volumes_before


def test_a_split_keeps_the_volume_it_started_with(box_mesher: Mesher) -> None:
    """The cells change; the material does not. Summed from the control, not from the split."""
    before = float(box_mesher.quality(Volume()).values.sum())

    box_mesher.split_volumes(SplitMethod.HEXA_TO_6)

    after = box_mesher.quality(Volume())
    assert float(after.values.sum()) == pytest.approx(before, rel=1e-9)
    assert float(after.values.min()) > 0.0
    assert before == pytest.approx(BOX_DX * BOX_DY * BOX_DZ, rel=1e-9)


def test_a_split_along_a_zero_normal_is_refused(box_mesher: Mesher) -> None:
    """There is no facet to pick, so the direction has to mean something."""
    with pytest.raises(PysmeshError, match="zero vector"):
        box_mesher.split_volumes(SplitMethod.HEXA_TO_2_PRISMS, facet_normal=(0.0, 0.0, 0.0))


@pytest.mark.parametrize("tolerance", [-1.0, float("nan")])
def test_a_bad_merge_tolerance_is_refused(box_mesher: Mesher, tolerance: float) -> None:
    """Validated in the positive form, so a NaN is caught rather than passed through."""
    with pytest.raises(PysmeshError, match="tolerance"):
        box_mesher.merge_nodes(tolerance=tolerance)


def test_splitting_a_mesh_with_no_volumes_is_refused(box_mesher: Mesher) -> None:
    """A no-op that reported success would read as "the mesh was already simplices"."""
    session = Session()
    session.add_rectangle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), BOX_DX, BOX_DY)
    with Mesher(ps.load_brep(session.brep())) as surface:
        surface.assign(Regular1D())
        surface.assign(NumberOfSegments(count=2))
        surface.assign(Quadrangle2D())
        surface.compute()

        with pytest.raises(PysmeshError, match="no volume elements"):
            surface.split_volumes(SplitMethod.HEXA_TO_6)


def test_every_group_operation_refuses_a_released_mesher() -> None:
    """The lifetime contract, on the surface this stage adds."""
    session = Session()
    session.add_box(BOX_DX, BOX_DY, BOX_DZ)
    mesher = Mesher(ps.load_brep(session.brep()))
    mesher.release()

    with pytest.raises(PysmeshError, match="released"):
        mesher.add_group("floor", ElementDimension.VOLUME, [])
    with pytest.raises(PysmeshError, match="released"):
        mesher.groups()
    with pytest.raises(PysmeshError, match="released"):
        mesher.convert_to_quadratic()
    with pytest.raises(PysmeshError, match="released"):
        mesher.merge_nodes()
    with pytest.raises(PysmeshError, match="released"):
        mesher.quality(Volume())
