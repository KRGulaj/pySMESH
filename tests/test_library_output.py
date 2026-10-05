# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-04

"""Gates for report ``defect_sweep_4.2.2.md`` §2 A4: no call writes to stdout or stderr.

A library call must leave the host's console alone. OCCT sent its transfer banners (the
STEP and IGES writers), the IGES entity count, and "File was not written with this
version of the topology" (a BREP read of data with no version line) to ``std::cout``.
The check is at the file-descriptor level (``capfd``), so C, C++ and Fortran writes
count as well as Python's. OCCT writes through the C runtime's ``FILE`` buffers, which
are flushed only at exit unless asked, so each check flushes them (``fflush(NULL)`` in
the shared UCRT) before it reads the capture. Each call's input is built before the
capture starts, so only the call itself is checked. Since phase 5 the NETGEN algorithms,
a NETGEN failure and a NETGEN cancel are checked too.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from pathlib import Path

import pytest

import pysmesh as ps
from pysmesh import (
    EntityKind,
    Mesher,
    Netgen1D2D,
    Netgen1D2D3D,
    Netgen2D,
    Netgen3D,
    NetgenParameters,
    NetgenParameters2D,
    NumberOfSegments,
    PysmeshError,
    Quadrangle2D,
    Regular1D,
    Session,
)

# The process's C runtime: python.exe and _core.pyd share ucrtbase.dll and its stdout.
_UCRT = ctypes.CDLL("ucrtbase")


def _box_brep() -> bytes:
    """A 3 x 7 x 11 box as BREP bytes."""
    s = Session()
    s.add_box(3.0, 7.0, 11.0)
    return s.brep()


def _written(capfd: pytest.CaptureFixture[str], call: Callable[[], object]) -> str:
    """Everything the call writes to fd 1 and fd 2, C runtime buffers included."""
    _UCRT.fflush(None)
    capfd.readouterr()
    call()
    _UCRT.fflush(None)
    out, err = capfd.readouterr()
    return out + err


def _failed_compute(shape: ps.Shape) -> None:
    """Regular1D with no hypothesis: the compute fails on every edge."""
    with Mesher(shape) as m:
        m.assign(Regular1D())
        m.assign(Quadrangle2D())
        with pytest.raises(PysmeshError):
            m.compute()


def _empty_brep() -> None:
    """load_brep of empty bytes, refused."""
    with pytest.raises(PysmeshError, match="null shape"):
        ps.load_brep(b"")


def _call(name: str, tmp: Path) -> Callable[[], object]:
    """The call to check, its input built now, before the capture starts."""
    brep = _box_brep()
    if name == "write_step_xde":
        return lambda: ps.write_step_xde(brep, unit="MM")
    if name == "read_step_xde":
        step = ps.write_step_xde(brep, unit="MM")
        return lambda: ps.read_step_xde(step)
    if name == "write_iges":
        return lambda: ps.write_iges(brep, unit="MM")
    if name == "read_iges":
        path = tmp / "box.igs"
        path.write_bytes(ps.write_iges(brep, unit="MM"))
        return lambda: ps.read_iges(path)
    if name == "failed_compute":
        shape = ps.load_brep(brep)
        return lambda: _failed_compute(shape)
    if name == "sew":
        s = Session()
        s.add_box(1.0, 1.0, 1.0)
        faces = s.entities(EntityKind.FACE).tolist()
        return lambda: s.sew(faces, tolerance=1e-6)
    return _empty_brep


@pytest.mark.parametrize(
    "name",
    [
        "write_step_xde",
        "read_step_xde",
        "write_iges",
        "read_iges",
        "failed_compute",
        "sew",
        "load_brep_empty",
    ],
)
def test_a_call_writes_nothing_to_stdout_or_stderr(
    capfd: pytest.CaptureFixture[str], tmp_path: Path, name: str
) -> None:
    """The call leaves fd 1 and fd 2 empty (A4)."""
    call = _call(name, tmp_path)

    written = _written(capfd, call)

    assert written == ""


# ---- NETGEN (phase 5): every NETGEN algorithm, a failure and a cancel -------------- #


def _shape_of(kind: str) -> ps.Shape:
    """A unit sphere (netgen retries its surface once), a box or a torus."""
    s = Session()
    if kind == "sphere":
        s.add_sphere(1.0)
    elif kind == "box":
        s.add_box(1.0, 2.0, 3.0)
    else:
        s.add_torus(3.0, 1.0)
    return ps.load_brep(s.brep())


def _netgen_compute(name: str, shape: ps.Shape) -> None:
    """Mesh the shape with one NETGEN recipe; a failure or a cancel is caught."""
    with Mesher(shape) as m:
        if name == "netgen_1d2d3d":
            m.assign(Netgen1D2D3D())
            m.assign(NetgenParameters(max_size=0.4))
        elif name == "netgen_1d2d_and_3d":
            m.assign(Netgen1D2D())
            m.assign(NetgenParameters2D(max_size=0.4))
            m.assign(Netgen3D())
            m.assign(NetgenParameters(max_size=0.4))
        elif name in ("netgen_regular_2d_and_3d", "netgen_failure"):
            m.assign(Regular1D())
            count = 3 if name == "netgen_failure" else 12
            m.assign(NumberOfSegments(count=count))
            m.assign(Netgen2D())
            m.assign(Netgen3D())
        elif name == "netgen_quadrangle_and_3d":
            m.assign(Regular1D())
            m.assign(NumberOfSegments(count=3))
            m.assign(Quadrangle2D())
            m.assign(Netgen3D())
        else:
            m.assign(Netgen1D2D3D())
            m.assign(NetgenParameters(max_size=0.08))
        asked = [0]

        def cancel() -> bool:
            asked[0] += 1
            return name == "netgen_cancel" and asked[0] > 1

        try:
            m.compute(cancel=cancel)
        except PysmeshError:
            if name not in ("netgen_failure", "netgen_cancel"):
                raise


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("netgen_1d2d3d", "sphere"),
        ("netgen_1d2d_and_3d", "sphere"),
        ("netgen_regular_2d_and_3d", "sphere"),
        ("netgen_quadrangle_and_3d", "box"),
        ("netgen_failure", "torus"),
        ("netgen_cancel", "torus"),
    ],
)
def test_a_netgen_compute_writes_nothing_to_stdout_or_stderr(
    capfd: pytest.CaptureFixture[str], name: str, kind: str
) -> None:
    """Every NETGEN algorithm, a NETGEN failure and a cancel leave fd 1 and fd 2 empty.

    netgen wrote its OCC meshing messages ("retry Surface 1" on every sphere, the
    "NOT ALL FACES HAVE BEEN MESHED" block on a failure or a cancel) and its overlap
    details straight to std::cout (patches/netgen/netgen_console_writes.patch), and
    NETGENPlugin pointed std::cout at a log file for the whole process.
    """
    shape = _shape_of(kind)

    written = _written(capfd, lambda: _netgen_compute(name, shape))

    assert written == ""
