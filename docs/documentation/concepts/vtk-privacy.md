# VTK Privacy

Since version 4.0.0, VTK, OCCT and Boost are all private to `_core.pyd`. Their DLLs are
bundled into the wheel and name-mangled by `delvewheel`, so none of them ever appears in the
host environment or can collide with a copy the host loaded for its own use. This page
states the architecture rule that privacy rests on, why it holds, and how it is enforced.
Read this before integrating pySMESH beside your own VTK, and before adding any binding
that touches VTK.

## The rule

**No VTK object may cross the Python boundary.** Every result pySMESH returns leaves as a
NumPy array or as BREP bytes. Nothing else is acceptable, and the rule has no exceptions for
convenience: not a `vtkPoints` handle, not a `vtkUnstructuredGrid`, not a smart pointer
wrapping either.

`CLAUDE.md` states this as load-bearing for the whole project, and `tests/test_vtk_privacy.py`
enforces it as a build gate. It must never be weakened or skipped.

## Why it matters

A bundled, name-mangled VTK copy is safe to carry inside `_core.pyd` for exactly one reason:
nothing shares VTK objects between `_core` and the host process. If that ever stopped being
true, the consequence would not be a clean error.

A `vtkUnstructuredGrid` built inside `_core` is an instance of a class from the *private*
VTK copy `_core` links. The host process, if it has VTK at all, has its own, separate copy.
Those two classes are **different C++ types**, even when both report an identical version
string, because they were compiled from different translation units into different DLLs.
Passing an object of one type where the other is expected is not a type error Python would
catch. It is memory corruption: the vtable, the allocator, and the object layout all belong
to a different binary than the one dereferencing them.

This is why the rule is stated as absolute rather than "usually returns arrays". A single
binding that leaked one VTK object would reintroduce, silently, the exact hazard that
bundling VTK privately was supposed to remove.

## What this replaces

Before 4.0.0, VTK was pySMESH's one shared runtime dependency: `_core` linked the host's
VTK, and the package refused to import unless that VTK was an exact version match. The
stated reason was ABI connectivity, since SMESH's mesh data structure (`SMDS`) is built on a
`vtkUnstructuredGrid` subclass internally. That reasoning assumed two copies of VTK in one
process were a hazard by themselves.

They are not. They are a hazard only if VTK objects actually pass between them, and none do.
`_core` links exactly three VTK components: `CommonCore`, `CommonDataModel`, and
`FiltersVerdict`. Every SMESH-derived result crosses the Python boundary as arrays. The
sharing bought nothing, and it cost every consumer an exact-version constraint on a library
pySMESH only ever used internally. 4.0.0 removed that constraint by bundling VTK privately
instead of resolving it from the host: see the
[4.0.0 release record](https://github.com/KRGulaj/pySMESH) for the full before/after.

## What is enforced, and how

`tests/test_vtk_privacy.py` runs four static guards plus two runtime checks. They are meant
to fail on the commit that introduces a violation, not months later in a consumer's process:

| Check | What it looks for |
|---|---|
| No VTK type in any pybind11 export | Scans every `.cpp` in `src/bindings/` for a `vtk*` type name on a `.def`, `.def_readonly`, `.def_readwrite`, `.def_property*`, or `.attr` line. |
| No VTK type registered with pybind11 | Scans for a `vtk*` type on a `py::class_<...>`, a holder declaration, or an opaque-type declaration. |
| No VTK type in the typed stubs | Scans `src/pysmesh/_core.pyi`, the public API contract, for any `vtk*` name. |
| No package module imports `vtk` | Scans every `.py` under `src/pysmesh/` for an `import vtk` or `from vtk import`. |
| `Session.tessellate` returns plain NumPy | Builds a box, tessellates it, and asserts the result arrays are `numpy.ndarray`, not a VTK type. |
| `Mesher.mesh()` returns plain NumPy | The closest thing to a VTK leak in this codebase: builds a mesh from arrays and asserts the harvested `node_coords` and `element_nodes` are plain `numpy.ndarray` of the expected dtype. |

The four static guards need no built extension. They run in a bare checkout, so a lint-only
CI job can enforce the architecture without compiling anything first.

## Cost, measured

Bundling VTK privately is not free. The 4.0.0 wheel carries the three linked VTK components
and their closure, which CI's `ci/report_wheel_size.py` measures and reports on every build,
failing the build if the wheel would exceed PyPI's 100 MB per-file limit. Because only three
narrow VTK components are linked rather than a full install, the bundled share is a small
fraction of what a complete VTK installation would cost the host environment.

## For a contributor

If a binding needs to return something VTK computed, harvest it into a NumPy array or BREP
bytes inside the binding, and never let a `vtk*` C++ type reach a `py::class_`, a `.def`, a
`.attr`, or the `.pyi` stubs. If you are unsure whether a change is safe, run
`pytest tests/test_vtk_privacy.py` before anything else; its static checks run without a
build.

## For an integrator

If your own application uses VTK, pySMESH's private copy cannot collide with yours. You are
free to use any VTK version, or none. pySMESH never resolves your VTK, never checks its
version, and never hands you an object from its own copy. The two coexist in one process
only because neither one is ever asked to interpret the other's objects.

---
*Author: Kajetan R. Gułaj*
*Date: 2026-08-24*
