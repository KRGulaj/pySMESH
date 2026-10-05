# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-05

"""The NETGEN entries of the native catalogue, and the assignment rules of
NETGEN_2D_ONLY.

The native factory builds the four NETGEN algorithms and their hypotheses by their
NETGENPlugin names. These tests talk to it by those names, as the catalogue's other
native-contract tests do (``test_mesher_family.py``), so they test the factory and
:meth:`~pysmesh.Mesher.assign` and not the typed classes.

NETGEN_2D_ONLY (``Netgen2D``) reads at most one of MaxElementArea, LengthFromEdges and
NETGEN_Parameters_2D on a face, and QuadranglePreference not beside NETGEN_Parameters_2D
(``NETGENPlugin_NETGEN_2D_ONLY.cxx``, ``CheckHypothesis``: ``HYP_CONCURRENT`` and
``HYP_INCOMPAT_HYPS``). The first rule never reaches the plugin: SMESH refuses a second
of the three on one sub-shape (same priority, ``HYP_ALREADY_EXIST``), and it hands an
algorithm at most one non-auxiliary hypothesis per sub-shape (``SMESH_Mesh.cxx``,
``GetHypotheses``: ``mainHypFound``), so the plugin's ``HYP_CONCURRENT`` branch is
unreachable. The auxiliary QuadranglePreference passes SMESH; the binding asks the
NETGEN algorithm, which reports ``HYP_INCOMPAT_HYPS``. Each refusal leaves the model as
it was.
"""

from __future__ import annotations

import pytest

import pysmesh as ps
from pysmesh import Mesher, PysmeshError, Session

# The full parameter set NETGEN_Parameters_2D reads, at the plugin's defaults. The typed
# class NetgenParameters2D sends the same keys.
_PARAMETERS_2D: dict[str, object] = {
    "max_size": 1000.0,
    "min_size": 0.0,
    "fineness": 2,
    "local_sizes": [],
    "second_order": False,
    "optimize": True,
    "quad_allowed": False,
    "surface_curvature": True,
    "fuse_edges": True,
    "surface_optimization_steps": 3,
    "element_size_weight": 0.2,
    "worst_element_measure": 2,
    "check_overlapping": True,
    "check_chart_boundary": True,
}
_PARAMETERS_3D: dict[str, object] = {
    **_PARAMETERS_2D,
    "volume_optimization_steps": 3,
    "use_delaunay": True,
}


def _box() -> ps.Shape:
    """A 2 x 3 x 4 box."""
    s = Session()
    s.add_box(2.0, 3.0, 4.0)
    return ps.load_brep(s.brep())


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("NETGEN_3D", {}),
        ("NETGEN_2D", {}),
        ("NETGEN_2D3D", {}),
        ("NETGEN_2D_ONLY", {}),
        ("NETGEN_Parameters", _PARAMETERS_3D),
        ("NETGEN_Parameters_2D", _PARAMETERS_2D),
        (
            "NETGEN_SimpleParameters_2D",
            {"number_of_segments": 3, "allow_quadrangles": False},
        ),
        (
            "NETGEN_SimpleParameters_3D",
            {"local_length": 0.5, "allow_quadrangles": False},
        ),
    ],
)
def test_the_native_factory_builds_every_netgen_entry_by_its_plugin_name(
    name: str, params: dict[str, object]
) -> None:
    """Each name builds and attaches to the whole shape."""
    with Mesher(_box()) as mesher:
        mesher._m.assign(name, params, "", 0)

        assigned = [entry[0] for entry in mesher.assignments()]

    assert assigned == [name]


def test_a_netgen_parameter_the_factory_does_not_read_is_refused() -> None:
    """NETGEN_Parameters_2D refuses the volume-only fields of NETGEN_Parameters."""
    with Mesher(_box()) as mesher:
        with pytest.raises(PysmeshError, match="does not take the parameter"):
            mesher._m.assign("NETGEN_Parameters_2D", _PARAMETERS_3D, "", 0)


def test_a_preset_field_is_refused_unless_the_fineness_is_user_defined() -> None:
    """growth_rate with the MODERATE preset is refused by the factory as well."""
    params = {**_PARAMETERS_2D, "growth_rate": 0.2}

    with Mesher(_box()) as mesher:
        with pytest.raises(PysmeshError, match="only with Fineness.USER_DEFINED"):
            mesher._m.assign("NETGEN_Parameters_2D", params, "", 0)


_SIZE_PARAMS: dict[str, dict[str, object]] = {
    "MaxElementArea": {"max_area": 0.5},
    "LengthFromEdges": {},
    "NETGEN_Parameters_2D": _PARAMETERS_2D,
}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("MaxElementArea", "LengthFromEdges"),
        ("LengthFromEdges", "NETGEN_Parameters_2D"),
        ("MaxElementArea", "NETGEN_Parameters_2D"),
    ],
)
def test_netgen_2d_only_refuses_a_second_size_hypothesis_on_one_sub_shape(
    first: str, second: str
) -> None:
    """Two of MaxElementArea, LengthFromEdges, NETGEN_Parameters_2D on one sub-shape.

    The three are 2-D hypotheses of the same priority, so SMESH refuses the second on
    the same sub-shape (HYP_ALREADY_EXIST), and the model stays as it was.
    """
    with Mesher(_box()) as mesher:
        mesher._m.assign("Regular_1D", {}, "", 0)
        mesher._m.assign("NETGEN_2D_ONLY", {}, "", 0)
        mesher._m.assign(first, _SIZE_PARAMS[first], "", 0)
        before = mesher.assignments()

        with pytest.raises(PysmeshError, match="already assigned there"):
            mesher._m.assign(second, _SIZE_PARAMS[second], "", 0)
        after = mesher.assignments()

    assert after == before


def test_netgen_2d_only_refuses_quadrangle_preference_beside_its_parameters() -> None:
    """QuadranglePreference with NETGEN_Parameters_2D: HYP_INCOMPAT_HYPS, refused."""
    with Mesher(_box()) as mesher:
        mesher._m.assign("Regular_1D", {}, "", 0)
        mesher._m.assign("NETGEN_2D_ONLY", {}, "", 0)
        mesher._m.assign("NETGEN_Parameters_2D", _PARAMETERS_2D, "", 0)
        before = mesher.assignments()

        with pytest.raises(PysmeshError, match="incompatible"):
            mesher._m.assign("QuadranglePreference", {}, "", 0)
        after = mesher.assignments()

    assert after == before


def test_netgen_2d_only_takes_quadrangle_preference_beside_max_element_area() -> None:
    """The one combination of two that NETGEN_2D_ONLY reads: both stay assigned."""
    with Mesher(_box()) as mesher:
        mesher._m.assign("Regular_1D", {}, "", 0)
        mesher._m.assign("NumberOfSegments", _number_of_segments(4), "", 0)
        mesher._m.assign("NETGEN_2D_ONLY", {}, "", 0)
        mesher._m.assign("MaxElementArea", {"max_area": 0.5}, "", 0)

        mesher._m.assign("QuadranglePreference", {}, "", 0)
        assigned = [entry[0] for entry in mesher.assignments()]

    assert assigned[-2:] == ["MaxElementArea", "QuadranglePreference"]


def _number_of_segments(count: int) -> dict[str, object]:
    """The native parameters of NumberOfSegments with a regular distribution."""
    return dict(ps.NumberOfSegments(count=count).params())
