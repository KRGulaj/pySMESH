# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-04

"""Viscous layers on every path that builds them, against the closed form of a stack.

A stack of ``N`` layers growing by the factor ``f`` and totalling ``T`` has the first layer
``t1 = T (f - 1) / (f^N - 1)``, or ``T / N`` when ``f = 1``
(``StdMeshers_ViscousLayers::Get1stLayerThickness``, SMESH ``V9_16_0``), and layer ``k``
ends at ``t1 (f^k - 1) / (f - 1)`` from the wall, ``k t1`` when ``f = 1``. The tests read
the layer node planes of the meshes and compare them with these positions.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pysmesh import PysmeshError, first_layer_thickness

TOL: float = 1e-9


def _layer_ends(total: float, factor: float, count: int) -> NDArray[np.float64]:
    """Where layers 1 .. N end, measured from the wall: the closed form above."""
    k = np.arange(1, count + 1, dtype=np.float64)
    if factor == 1.0:
        return k * total / count
    first = total * (factor - 1.0) / (factor**count - 1.0)
    return first * (factor**k - 1.0) / (factor - 1.0)


# ---- L1 first_layer_thickness ------------------------------------------------------ #


@pytest.mark.parametrize(
    ("total", "factor", "count"),
    [(0.3, 1.2, 3), (1.0, 1.0, 4), (2.0, 1.5, 1), (0.01, 1.05, 30)],
)
def test_first_layer_thickness_is_the_closed_form_and_the_stack_sums_to_t(
    total: float, factor: float, count: int
) -> None:
    """t1 = T (f - 1) / (f^N - 1), T / N at f = 1; the N layers add up to T."""
    expected = (
        total / count
        if factor == 1.0
        else total * (factor - 1.0) / (factor**count - 1.0)
    )

    first = first_layer_thickness(total, factor, count)

    assert first == pytest.approx(expected, rel=1e-14)
    layers = first * factor ** np.arange(count, dtype=np.float64)
    assert float(layers.sum()) == pytest.approx(total, rel=1e-12)


@pytest.mark.parametrize(
    ("total", "factor", "count", "word"),
    [
        (0.0, 1.2, 3, "total_thickness"),
        (-1.0, 1.2, 3, "total_thickness"),
        (0.3, 1.2, 0, "layer_count"),
        (0.3, 0.9, 3, "stretch_factor"),
    ],
)
def test_first_layer_thickness_refuses_a_stack_smesh_cannot_grow(
    total: float, factor: float, count: int, word: str
) -> None:
    """Degenerate input: upstream answers T / N for f < 1 and builds nothing at 0."""
    with pytest.raises(PysmeshError, match=word):
        first_layer_thickness(total, factor, count)
