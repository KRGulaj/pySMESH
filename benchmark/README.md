<!--
SPDX-License-Identifier: LGPL-2.1-only
Copyright (C) 2026 Kajetan R. Gulaj
-->

# Tessellation benchmark harness

Measures pySMESH against gmsh on surface tessellation at a **matched chordal deviation
tolerance**. The published results and the full methodology are in
[`docs/documentation/benchmarks/gmsh-vs-pysmesh.md`](../docs/documentation/benchmarks/gmsh-vs-pysmesh.md).

Both tools drive the same OCCT routine, `BRepMesh_IncrementalMesh`. This harness exists to
compare how each one dispatches it, at settings where both produce the same mesh quality.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
```

Python 3.13 on Windows x64. pySMESH ships only that wheel, so the harness follows it.

## Geometry

Two kinds, and neither is committed to this repository.

**Synthetic** — generated locally, and geometrically reproducible:

```bash
python corpus_gen.py
```

The geometry is deterministic, but the files are **not** byte-identical between runs: OCCT
writes a creation timestamp into the STEP `FILE_NAME` header, so the SHA-256 of a regenerated
file differs while every coordinate in it is the same. A hash is therefore not a useful check
on these files; the face count and the resulting triangle count are.

This writes `corpus/` (about 14 MB): a ladder of 1, 10, 100, 1000 and 5000 identical spheres,
plus a truncated cone. Face count is exactly the sphere count, and per-face work is constant
across the ladder, which is what isolates dispatch cost from tessellation cost.

**Real** — taken from the gmsh source distribution's own `benchmarks/` directory. Those CAD
models belong to the gmsh project, so they are referenced by SHA-256 rather than
redistributed here. Clone gmsh and point the harness at it:

```bash
git clone https://gitlab.onelab.info/gmsh/gmsh.git
export GMSH_BENCH=/path/to/gmsh/benchmarks      # or pass --gmsh-bench
```

Cases whose files are missing are skipped with a message, so the synthetic suites run without
this step.

The `big` suite takes an arbitrary large assembly supplied by the operator. None ships here:

```bash
python bench_driver.py --suite big --big-assembly /path/to/assembly.step
```

## Run

```bash
python bench_driver.py --suite ladder --reps 5     # face-count scaling
python bench_driver.py --suite sweep  --reps 5     # tolerance sweeps
python bench_driver.py --suite real   --reps 5     # imported STEP bodies

python verify_quality.py      # achieved deviation against closed-form surfaces
python real_deviation.py      # achieved deviation by OCC projection, imported bodies
python gmsh_overhead.py       # splits gmsh's BRepMesh time from its mesh construction
python diag_units.py          # unit-agreement check between the two importers
python analyze.py             # tables + results/summary.csv
```

Raw per-run data is appended to `results/raw_runs.jsonl`. Re-running appends rather than
overwrites, so delete the file first for a clean set.

## Reading the numbers

Results are **specific to one machine**. The parallel figures are a function of how many
logical processors `BRepMesh` is given; the published set used 14. `bench_driver.py` pins
every run to the same processors via `AFFINITY`. Edit that constant to match your machine
before comparing anything.

Every cell reports median, min and max over 5 repetitions, after one discarded warm-up.

## The unit trap

The two importers disagree about length units. gmsh's STEP reader rescales every model to
millimetres; `pysmesh.read_step_xde` preserves the file's declared unit. The same numeric
deflection is therefore a 1000x coarser request on a metre-declared file, which produces a
coarser mesh in less time and a meaningless speed-up.

`bench_worker.native_lin` converts a millimetre tolerance into each model's own unit. Any
benchmark of these two libraries that skips this step is measuring units, not speed.
`diag_units.py` checks for the disagreement directly.

## Files

| file | role |
|---|---|
| `bench_driver.py` | runs the (tool x geometry x tolerance) matrix, one subprocess per cell |
| `bench_worker.py` | one measured cell; holds the unit conversion and the timed regions |
| `corpus_gen.py` | generates the synthetic corpus |
| `deviation.py` | barycentric sampling and closed-form distance to sphere and cone |
| `verify_quality.py` | matched-quality check against closed-form surfaces |
| `real_deviation.py` | matched-quality check on imported bodies, by OCC projection |
| `gmsh_overhead.py` | isolates gmsh's `BRepMesh` time from `storeSTLAsMesh` |
| `gmsh_native_note.py` | records what gmsh's native 2-D mesher costs; excluded from the comparison, see the report |
| `diag_units.py` | importer unit-agreement diagnostic |
| `analyze.py` | aggregates raw runs into the report's tables |

`results/raw_runs_pre_unitfix.jsonl` is a superseded run kept deliberately, so the unit
correction described in the report can be audited rather than taken on trust.

## Attribution

Gmsh is copyright (C) 1997–2026 Christophe Geuzaine and Jean-François Remacle, licensed
GPL-2-or-later. It is installed from PyPI by `requirements.txt` and is **not** redistributed
by this repository. Gmsh's manual asks that work using it cite:

> C. Geuzaine and J.-F. Remacle. *Gmsh: a three-dimensional finite element mesh generator
> with built-in pre- and post-processing facilities.* International Journal for Numerical
> Methods in Engineering, **79**(11), pp. 1309–1331, 2009.

The real CAD corpus belongs to the Gmsh distribution and is referenced by SHA-256, never
copied here. Both libraries tessellate with Open CASCADE Technology, copyright (C) Open
CASCADE SAS, LGPL-2.1 with an exception.

This harness is written by the author of pySMESH and is not endorsed by the Gmsh project.
Full attribution, and the reasoning behind the comparison, are in
[the report](../docs/documentation/benchmarks/gmsh-vs-pysmesh.md).
