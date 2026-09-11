# Surface tessellation: pySMESH against gmsh

How fast is pySMESH's surface tessellation, measured against gmsh at a **matched chordal
deviation tolerance**, and where does any difference come from?

Measured 2026-09-11. Every number on this page comes from the raw data linked in §13.

---

## 1. Quotable summary

> Tessellating gmsh's own benchmark model `Block.stp` (533 faces) to an absolute chordal
> tolerance of 0.05 mm, pySMESH 4.0.0 produces 14,182 triangles in 61 ms and gmsh 4.15.2
> produces 14,174 triangles in 200 ms — a 3.26x wall-clock speed-up at an identical achieved
> maximum chordal deviation of 0.1197 mm. Both call the same OCCT routine, so the difference
> is scheduling and dispatch rather than a better algorithm: with pySMESH restricted to one
> thread the same comparison is 1.70x.

Use that sentence, not a larger one. The larger figures on this page are real, but they are
tied to geometry with many equally sized faces, which is a favourable special case.

That cell was measured three times: twice in separate suites run hours apart, and once more
from a clean checkout with a freshly built virtualenv.

| run | gmsh | pySMESH | speed-up |
|---|---:|---:|---:|
| first | 199.6 ms | 61.2 ms | 3.26x |
| second, separate suite | 191.3 ms | 58.7 ms | 3.26x |
| third, clean checkout | 204.7 ms | 58.8 ms | 3.48x |

All three produced 14,174 triangles for gmsh and 14,182 for pySMESH, every time. The quoted
**3.26x is the lowest of the three**, and is used deliberately for that reason.

---

## 2. The finding in one table

Wall-clock median over 5 repetitions. Same requested tolerance on both sides. "1 thread" is
pySMESH with `parallel=False`.

| geometry | faces | triangles (gmsh / pySMESH) | gmsh | pySMESH 1 thread | pySMESH parallel | speed-up |
|---|---:|---|---:|---:|---:|---:|
| single sphere | 1 | 33,506 / 33,506 | 257 ms | 245 ms | 270 ms | **0.95x** |
| truncated cone | 3 | 458 / 458 | 2.80 ms | 2.29 ms | 3.18 ms | **0.88x** |
| `part.step` | 26 | 972 / 972 | 13.2 ms | 11.3 ms | 7.6 ms | 1.74x |
| `Kurbelwelle.stp` | 58 | 2,490 / 2,490 | 30.2 ms | 18.2 ms | 8.4 ms | 3.61x |
| `1385_Laufrad_1000.stp` | 123 | 47,071 / 47,322 | 252 ms | 219 ms | 178 ms | 1.42x |
| `U_Joint_2.stp` | 133 | 9,602 / 9,576 | 83.8 ms | 54.5 ms | 20.6 ms | 4.06x |
| `Zylkopf.stp` | 137 | 6,230 / 6,260 | 65.9 ms | 38.0 ms | 17.9 ms | 3.68x |
| `917_fusee.stp` | 249 | 13,225 / 13,285 | 140 ms | 96.6 ms | 39.9 ms | 3.52x |
| `Block.stp` | 533 | 14,174 / 14,182 | 200 ms | 117 ms | 61.2 ms | 3.26x |
| large assembly | 5,606 | 865,462 / 869,296 | 5.611 s | 4.667 s | 1.140 s | 4.92x |
| 5,000 spheres | 5,000 | 4,890,000 / 4,890,000 | 13.783 s | 10.928 s | 1.568 s | **8.79x** |

Across all 37 measured cells: parallel speed-up ranges from **0.88x to 9.36x**, median
**3.26x**. Single-threaded it ranges from **0.97x to 1.74x**, median **1.26x**.

There is no single speed-up figure for these two libraries. The number depends almost
entirely on how many faces the model has, and on how evenly sized they are. §7 gives the
whole curve rather than one ratio.

---

## 3. Why this comparison can be exact

Both tools call the same function in the same library. This was verified in source, not
inferred from behaviour.

**gmsh.** `makeSTL` in `src/geo/GModelIO_OCC.cpp:6328` reads three options and constructs
the mesher at line 6339:

```cpp
double lin = CTX::instance()->mesh.stlLinearDeflection;
bool   rel = CTX::instance()->mesh.stlLinearDeflectionRelative;
double ang = CTX::instance()->mesh.stlAngularDeflection;
BRepMesh_IncrementalMesh aMesher(s, lin, rel, ang, true);
```

Note the argument type: `const TopoDS_Face &s`. gmsh meshes **one face per call**.
`OCC_Internals::_makeSTL` at line 6480 drives it with a serial `TopExp_Explorer` loop over
every face.

**pySMESH.** `Session::tessellate` in `src/bindings/session_tessellate.cpp:99` fills the same
parameters and performs **one call over the whole shape**:

```cpp
IMeshTools_Parameters params;
params.Deflection = deflection;
params.Angle      = angle_rad;
params.Relative   = relative;
params.InParallel = parallel;
BRepMesh_IncrementalMesh mesher;
mesher.SetShape(root);
mesher.Perform(driver.range());
```

So the measurand is not "whose tessellator is better". It is "what does it cost to dispatch
the same OCCT tessellator one way or the other". That is a narrower claim, and a defensible
one.

### 3.1 What gmsh's per-face loop is for

`makeSTL` is not a bulk tessellation entry point, and it was never designed as one. gmsh
builds an STL representation **one face at a time because that is what its own consumers
ask for**: the triangulation of a single face for display, for `classifySurfaces`, for STL
export, and as input to its own meshing algorithms. Every one of those paths is a
per-face question, and a whole-shape call would buy gmsh nothing in any of them.

pySMESH exposes tessellation as a bulk operation because its consumer is different: a
viewport that needs every face of the model at once, as fast as possible.

The two libraries therefore answer different questions with the same OCCT routine. The
measurements below say what the per-face design costs **when it is used for a bulk job**.
They do not say that gmsh chose wrongly for the job gmsh is actually doing. A reader
deciding between the two tools should weigh this against their own use: if you need one
face at a time, the per-face path is the right shape and this benchmark does not apply to
you.

The three controls map one to one:

| control | gmsh option | pySMESH argument |
|---|---|---|
| linear deflection | `Mesh.StlLinearDeflection` | `deflection` |
| angular deflection | `Mesh.StlAngularDeflection` | `angle_deg` (converted to rad) |
| relative mode | `Mesh.StlLinearDeflectionRelative` | `relative` |

Both sides used absolute mode (`relative = 0`) throughout. OCCT's relative mode is defined
per edge, as a fraction of that edge's own length, so it is not a single model-wide quantity
and is a poor shared control. Angular deflection was fixed at 0.5 rad on both sides.

---

## 4. The measurand

Control is on chordal deviation. Triangle count is an **output**, never a control.

Recorded per run:

1. wall-clock time;
2. CPU time, summed over the process and all children;
3. peak resident set;
4. triangle count;
5. achieved maximum and RMS chordal deviation from the exact B-rep surface.

Achieved deviation is the check that both sides were asked for the same thing. The rule was
set in advance: if achieved maximum deviation differs by more than 5% at the same requested
tolerance, that cell is void and is not used.

**No cell in the final data set is void.** Achieved deviation agreed to at least six
significant figures everywhere it was measured.

### 4.1 How deviation was measured

Mesh nodes sit exactly on the surface. Node error is about 1e-15 and says nothing. The
chordal error lives in the triangle interior. Each triangle was therefore sampled on a
barycentric grid, and every sample was measured against the exact surface.

Two independent referees were used, so no claim depends on either library's own projector.

**Closed form**, for the sphere and the truncated cone. A surface of revolution reduces to a
point-to-segment distance in the (r, z) half-plane. No library is involved at all.

| geometry | requested | triangles | achieved max dev | achieved RMS dev | agreement |
|---|---:|---:|---:|---:|---|
| sphere | 1.0 | 306 / 306 | 6.006414e-01 both | 1.405315e-01 both | exact |
| sphere | 0.3 | 360 / 360 | 3.225484e-01 both | 1.137272e-01 both | exact |
| sphere | 0.1 | 978 / 978 | 7.482400e-02 both | 4.242826e-02 both | exact |
| sphere | 0.03 | 3,278 / 3,278 | 2.279656e-02 both | 1.288969e-02 both | exact |
| sphere | 0.01 | 10,108 / 10,108 | 2.058166e-02 both | 4.247392e-03 both | exact |
| sphere | 0.003 | 33,506 / 33,506 | 6.201572e-03 both | 1.288930e-03 both | exact |
| cone | 1.0 | 152 / 152 | 4.081204e-01 both | 1.235247e-01 both | exact |
| cone | 0.1 | 458 / 458 | 7.890192e-02 both | 3.270251e-02 both | exact |
| cone | 0.003 | 6,956 / 6,956 | 2.673569e-03 both | 1.165776e-03 both | exact |
| 1,000 spheres | 0.1 | 978,000 / 978,000 | 7.482400e-02 both | 4.241463e-02 / 4.241629e-02 | 0.0039% |

**OCC projection**, for imported bodies with no closed form. Each side's triangles are
projected onto the face that produced them, by that side's own kernel: gmsh via
`gmsh.model.getClosestPoint` (OCC 7.8.1), pySMESH via `Session.project_on_face` (OCCT 8.0.0).
Each kernel judges its own output, which avoids matching face identifiers across two
importers. The two projectors agree exactly on the analytic cases above, so neither flatters
its own side.

| geometry | requested | gmsh max dev | pySMESH max dev | gmsh RMS | pySMESH RMS |
|---|---:|---:|---:|---:|---:|
| `Block.stp` | 0.05 mm | 1.19692e-01 | 1.19692e-01 | 1.14606e-02 | 1.14573e-02 |
| `917_fusee.stp` | 0.1 mm | 6.30535e-01 | 6.30535e-01 | 3.02939e-02 | 3.02031e-02 |
| `Kurbelwelle.stp` | 0.02 mm | 1.43797e-02 | 1.43797e-02 | 5.20248e-03 | 5.20248e-03 |

Samples on a triangle's own boundary were excluded. Orthogonal projection onto a trimmed face
has no solution for a point exactly on the face boundary, on either side.

### 4.2 Are the two meshes the same mesh?

Not always, and this page does not claim they are.

On the sphere the two triangle soups are bit-identical at most tolerances. On the truncated
cone they are not: the node sets are bit-identical (78 nodes, maximum coordinate difference
exactly 0.0), the triangle count is identical, and the achieved deviation is identical, but
144 of 152 triangles connect those nodes differently. OCCT 7.8.1 and OCCT 8.0.0 choose
different diagonals.

The correct statement is therefore: **equal triangle count, equal achieved deviation,
sometimes different connectivity**. That satisfies matched quality. It is not bit-identity.

---

## 5. Controls

| control | value |
|---|---|
| processor | Intel Core Ultra 7 255H (Family 6 Model 197 Stepping 2) |
| logical processors used | LP 0–13, pinned identically for every run |
| excluded | LP 14–15 (LP-E cores), matching this machine's normal worker pool |
| memory | 63.4 GiB |
| OS | Windows 11 26200 |
| repetitions | 5 timed, plus 1 warm-up discarded |
| statistic | median, with min and max in the raw data |
| process isolation | each cell runs in a fresh subprocess |

Each repetition starts from a clean tessellation state. gmsh gets a freshly imported model,
because `GFace::buildSTLTriangulation` caches and returns early on a second call
(`src/geo/GFace.cpp:1558`). pySMESH is called with `incremental=False`, which runs
`BRepTools::Clean` on the shape first and forces a full re-mesh. That clean is inside
pySMESH's timed region and gmsh pays no equivalent, so this control is slightly unfavourable
to pySMESH.

### 5.1 Versions

| component | gmsh side | pySMESH side |
|---|---|---|
| tool | gmsh 4.15.2 (pip wheel) | pySMESH 4.0.0 (PyPI wheel, sha 74cb77e) |
| OCCT | **7.8.1** | **8.0.0** |
| Python | 3.13.12 | 3.13.12 |
| NumPy | 2.5.3 | 2.5.3 |

The OCCT versions differ. §9.1 treats that as a confound and bounds it.

### 5.2 What is inside the timed region

Both sides start from a CAD model already loaded in memory. STEP import is excluded on both
sides, because it is a different operation.

* **gmsh**: `gmsh.model.mesh.importStl()`, then `getElements(2)` and `getNodes()`.
* **pySMESH**: `Session.tessellate(...)`, then the returned NumPy arrays.

Both are Python API calls, and both call overheads are inside the measured region. Neither
side is measured through C++.

`importStl` is gmsh's fastest in-memory route. The alternative, `gmsh.write("out.stl")`, was
measured at 286 ms against `importStl`'s 199 ms on `Block.stp`, so the faster route was given
to gmsh.

---

## 6. Corpus

Real geometry is taken from **gmsh's own published benchmark suite**, so the corpus cannot be
said to favour pySMESH. The synthetic ladder is generated by a short gmsh script and is
reproducible by anyone.

| file | faces | solids | size | declared unit | bbox diag (mm) | sha256 (first 16) |
|---|---:|---:|---:|---|---:|---|
| `spheres_00001.step` | 1 | 1 | 2 KiB | MM | 34.6 | `85cd55aa071b7421` |
| `spheres_00010.step` | 10 | 10 | 22 KiB | MM | 123.7 | `dda438535efc2427` |
| `spheres_00100.step` | 100 | 100 | 208 KiB | MM | 226.5 | `8ff1a8de996190d9` |
| `spheres_01000.step` | 1000 | 1000 | 2141 KiB | MM | 502.3 | `81c90cebb36d81b7` |
| `spheres_05000.step` | 5000 | 5000 | 10959 KiB | MM | 884.7 | `ae1fbb5c463fdfb1` |
| `cone.step` | 3 | 1 | 6 KiB | MM | 69.3 | `0d582895e4481a1a` |
| `part.step` | 26 | 1 | 60 KiB | M | 213.0 | `46ea7d269b424ad7` |
| `wrenchnut.stp` | 36 | 2 | 258 KiB | MM | 433.0 | `dba15ffe7ec4fae9` |
| `Kurbelwelle.stp` | 58 | 1 | 79 KiB | MM | 68.3 | `e72b0f347184fbaa` |
| `1385_Laufrad_1000.stp` | 123 | 1 | 373 KiB | MM | 2169.3 | `e6003effb8d36766` |
| `U_Joint_2.stp` | 133 | 1 | 1176 KiB | INCH | 138.7 | `b607a4886a1820e9` |
| `Zylkopf.stp` | 137 | 1 | 248 KiB | MM | 42.0 | `cd0ced947da2f617` |
| `917_fusee.stp` | 249 | 1 | 726 KiB | M | 472.4 | `10c201b8b406a756` |
| `Block.stp` | 533 | 1 | 908 KiB | MM | 84.2 | `03de4b2e21961e1f` |
| large assembly | 5606 | 117 | 16297 KiB | M | 5853.4 | `5c64e2ae06d21fcb` |

A caveat on the hashes. They are meaningful for the imported bodies, which are fixed files.
They are **not** reproducible for the six synthetic files: OCCT stamps a creation time into
the STEP `FILE_NAME` header, so regenerating the corpus yields identical geometry and a
different hash. The hashes above identify the exact files measured here; a reader
regenerating the corpus should expect matching face and triangle counts, not matching
digests.

The corpus meets the intended spread: an analytic primitive with known exact curvature
(sphere), a tapering surface (truncated cone), an assembly with many small faces
(`Block.stp`, 533 faces), several real imported STEP bodies, and one large assembly.

gmsh completed **every** case, including the 5,606-face assembly. Nothing is reported as a
timeout, and no result rests on one tool failing to finish.

The 5,606-face assembly is a private model and is not distributed. Every other geometry is
either generated by `corpus_gen.py` or available from the gmsh source distribution.

---

## 7. Results

### 7.1 Face-count ladder — the mechanism

N identical spheres on a lattice. Face count is exactly N. Per-face work is constant across
the ladder, so any change in the ratio isolates dispatch from work. Requested deflection
0.1 mm. All four columns produce exactly the same triangle count.

| faces | triangles | gmsh | pySMESH 1 thread | pySMESH parallel | 1-thread ratio | parallel ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 978 | 3.0 ms | 2.4 ms | 2.5 ms | 1.24x | 1.16x |
| 10 | 9,780 | 29.4 ms | 21.9 ms | 6.2 ms | 1.34x | 4.76x |
| 100 | 97,800 | 275 ms | 216 ms | 33.4 ms | 1.27x | 8.26x |
| 1,000 | 978,000 | 2.728 s | 2.177 s | 310 ms | 1.25x | 8.79x |
| 5,000 | 4,890,000 | 13.783 s | 10.928 s | 1.568 s | 1.26x | 8.79x |

Two things are visible.

The single-threaded ratio is flat near 1.26x and does not grow with face count. On this
geometry gmsh's per-face dispatch is therefore **not** the main cost. The 1.26x comes from
the OCCT version difference and from harvest efficiency.

The parallel ratio climbs and then saturates near 8.8x on 14 logical processors. That
saturation, not the algorithm, is what produces every large figure on this page.

### 7.2 Time against achieved deviation

This is the stronger result, because it is a curve rather than a ratio.

**Single sphere, 1 face.** Requested deflection spans 1.0 down to 0.003, a factor of 333.

| requested | triangles | achieved max dev | achieved RMS | gmsh | pySMESH 1 thr | pySMESH par | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 306 | 6.0064e-01 | 1.4053e-01 | 1.30 ms | 1.04 ms | 1.17 ms | 1.11x |
| 0.3 | 360 | 3.2255e-01 | 1.1373e-01 | 1.48 ms | 1.14 ms | 1.44 ms | 1.03x |
| 0.1 | 978 | 7.4824e-02 | 4.2428e-02 | 2.89 ms | 2.45 ms | 2.45 ms | 1.18x |
| 0.03 | 3,278 | 2.2797e-02 | 1.2890e-02 | 9.89 ms | 8.47 ms | 8.79 ms | 1.13x |
| 0.01 | 10,108 | 2.0582e-02 | 4.2474e-03 | 42.7 ms | 36.5 ms | 40.9 ms | 1.04x |
| 0.003 | 33,506 | 6.2016e-03 | 1.2889e-03 | 257 ms | 245 ms | 270 ms | **0.95x** |

**Truncated cone, 3 faces.**

| requested | triangles | achieved max dev | achieved RMS | gmsh | pySMESH 1 thr | pySMESH par | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 152 | 4.0812e-01 | 1.2352e-01 | 1.51 ms | 1.26 ms | 1.67 ms | **0.90x** |
| 0.3 | 214 | 2.0752e-01 | 7.9012e-02 | 2.04 ms | 1.43 ms | 1.76 ms | 1.16x |
| 0.1 | 458 | 7.8902e-02 | 3.2703e-02 | 2.80 ms | 2.29 ms | 3.18 ms | **0.88x** |
| 0.03 | 1,054 | 2.4889e-02 | 1.0563e-02 | 6.56 ms | 6.04 ms | 5.43 ms | 1.21x |
| 0.01 | 2,620 | 8.6629e-03 | 3.7348e-03 | 22.9 ms | 20.4 ms | 18.7 ms | 1.23x |
| 0.003 | 6,956 | 2.6736e-03 | 1.1658e-03 | 94.3 ms | 96.8 ms | 90.7 ms | 1.04x |

**`Block.stp`, 533 faces.**

| requested | triangles | gmsh | pySMESH 1 thr | pySMESH par | 1-thread ratio | parallel ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 10,480 | 174 ms | 102 ms | 59.0 ms | 1.71x | 2.94x |
| 0.2 | 10,708 | 180 ms | 106 ms | 58.5 ms | 1.69x | 3.07x |
| 0.05 | 14,174 | 191 ms | 120 ms | 58.7 ms | 1.60x | 3.26x |
| 0.02 | 21,068 | 223 ms | 145 ms | 64.1 ms | 1.54x | 3.49x |
| 0.005 | 44,117 | 336 ms | 243 ms | 74.8 ms | 1.38x | 4.49x |

Here the single-threaded ratio is 1.38x to 1.71x, clearly above the 1.26x of the sphere
ladder. `Block.stp` has 533 faces of unequal size, so gmsh's per-face dispatch does cost
something on real geometry. The effect is real but small.

**1,000 spheres.**

| requested | triangles | gmsh | pySMESH 1 thr | pySMESH par | 1-thread ratio | parallel ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 306,000 | 1.005 s | 0.753 s | 0.150 s | 1.34x | 6.72x |
| 0.3 | 360,000 | 1.153 s | 0.873 s | 0.160 s | 1.32x | 7.21x |
| 0.1 | 978,000 | 2.702 s | 2.185 s | 0.308 s | 1.24x | 8.78x |
| 0.03 | 3,278,000 | 10.280 s | 8.719 s | 1.098 s | 1.18x | **9.36x** |
| 0.01 | 10,108,000 | 40.855 s | 34.729 s | 4.993 s | 1.18x | 8.18x |

### 7.3 A 5,606-face assembly

A large real assembly, for contrast with the uniform synthetic ladder. gmsh finishes it
comfortably at both tolerances.

| requested | triangles (gmsh / pySMESH) | delta | gmsh | pySMESH 1 thr | pySMESH par | 1-thread | parallel |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1.0 mm | 302,358 / 303,692 | +0.44% | 2.815 s | 2.186 s | 0.973 s | 1.29x | **2.89x** |
| 0.25 mm | 865,462 / 869,296 | +0.44% | 5.611 s | 4.667 s | 1.140 s | 1.20x | **4.92x** |

This assembly parallelises **worse** than the synthetic ladder, despite having more faces
(5,606 against 5,000). Its faces differ greatly in size, and `BRepMesh` distributes whole
faces across threads. One large face therefore sets a floor on wall-clock time that more
threads cannot lower.

Face count alone does not predict the speed-up. Face-size uniformity matters more. This is
the single most important caveat on this page for anyone estimating what they will get on
their own CAD.

### 7.4 Memory

Peak resident set, same runs.

| geometry | triangles | gmsh | pySMESH | ratio |
|---|---:|---:|---:|---:|
| 1,000 spheres | 978,000 | 381 MB | 193 MB | 1.97x less |
| 5,000 spheres | 4,890,000 | 1,693 MB | 715 MB | 2.37x less |
| large assembly @ 0.25 | 869,296 | 492 MB | 360 MB | 1.37x less |

gmsh materialises `MVertex` and `MTriangle` objects; pySMESH returns flat NumPy arrays. The
gap grows with triangle count, as that difference predicts.

### 7.5 Total CPU cost

Wall-clock is not free. On 5,000 spheres:

| | wall | CPU | CPU/wall |
|---|---:|---:|---:|
| gmsh | 13.783 s | 13.797 s | 1.00 |
| pySMESH 1 thread | 10.928 s | 10.703 s | 0.98 |
| pySMESH parallel | 1.568 s | 16.531 s | 10.54 |

pySMESH parallel spends about **20% more total CPU** than gmsh to deliver 8.8x less wall
time. On a loaded machine running many jobs at once, that trade is worse than it looks here.
State it whenever the 8.8x figure is used.

---

## 8. Where gmsh wins or ties

These are not rounded away.

| geometry | requested | gmsh | pySMESH parallel | result |
|---|---:|---:|---:|---|
| truncated cone | 0.1 | 2.80 ms | 3.18 ms | **gmsh 14% faster** |
| truncated cone | 1.0 | 1.51 ms | 1.67 ms | **gmsh 11% faster** |
| single sphere | 0.003 | 257 ms | 270 ms | **gmsh 5% faster** |
| truncated cone | 0.003 | 94.3 ms | 90.7 ms | tie (1.04x) |
| single sphere | 0.01 | 42.7 ms | 40.9 ms | tie (1.04x) |
| single sphere | 0.3 | 1.48 ms | 1.44 ms | tie (1.03x) |

The pattern is consistent and has a clear cause. On one or three faces there is nothing to
parallelise, and pySMESH's thread pool is pure overhead. gmsh wins those cases outright.
Single-threaded, pySMESH is also slower than gmsh on the cone at the finest tolerance
(0.97x).

**If your geometry is a handful of large faces, pySMESH offers you nothing here, and may cost
you a little.** The advantage begins at roughly 10 faces and is worth having from roughly 30.

---

## 9. Confounds

### 9.1 The OCCT versions differ, and this cannot be isolated

gmsh 4.15.2 ships OCC **7.8.1**. pySMESH 4.0.0 bundles OCCT **8.0.0**. Both are statically
linked, and neither can be swapped without rebuilding that project.

This is the weakest point in the comparison, and a reader is right to press on it. Part of
the flat 1.26x single-threaded ratio is a kernel-version difference, and not a pySMESH
property at all.

What bounds it: the 1-face case has no per-face dispatch difference, so its single-threaded
ratio is almost pure kernel-plus-harvest. That ratio is 1.24x on the sphere ladder, and
ranges from 1.05x to 1.30x across the six-point sphere tolerance sweep. On the truncated cone
at the finest tolerance it is 0.97x, that is, gmsh is faster.

So the kernel and harvest difference is **small, tolerance-dependent, and sometimes favours
gmsh**. It cannot account for a figure above about 1.3x. Every larger number on this page
survives even if the whole of that 1.3x is credited to OCCT 8.0.0 rather than to pySMESH.

### 9.2 gmsh's mesh construction overhead — measured, and small

`importStl` does two things per face: `buildSTLTriangulation` (the OCCT work) and
`storeSTLAsMesh` (allocating gmsh's mesh objects). A second `importStl` call hits the STL
cache and re-runs only the conversion, which isolates it.

| geometry | `importStl` total | BRepMesh | conversion | conversion share |
|---|---:|---:|---:|---:|
| 100 spheres | 274 ms | 268 ms | 6 ms | 2.2% |
| 1,000 spheres | 2.688 s | 2.617 s | 70 ms | 2.6% |
| `Block.stp` | 206 ms | 204 ms | 2 ms | 0.8% |
| `917_fusee.stp` | 143 ms | 142 ms | 1 ms | 0.8% |

gmsh's measured time is 97%–99% OCCT tessellation. The comparison is not an artefact of
charging gmsh for bookkeeping.

### 9.3 Both sides are measured through their Python API

Yes, and deliberately. Both overheads are inside the measured region. Neither tool is given a
C++ path the other does not get. For pySMESH the stateless entry point
`pysmesh.tessellate(brep_bytes)` was also measured; it parses BREP bytes inside its own timed
region and is consistently slower than the `Session` path. It is in the raw data and excluded
from headline numbers.

### 9.4 gmsh's native 2-D mesher is a different product and is excluded

A gmsh user will object that `importStl` is gmsh's visualisation path, and that gmsh's real
output is `gmsh.model.mesh.generate(2)`.

That is true, and it is exactly why `generate(2)` is **not** on this page. `generate(2)` is
controlled by element size — `Mesh.MeshSizeMax`, `Mesh.MeshSizeFromCurvature` — and takes no
chordal-deviation tolerance. There is no setting at which it and `BRepMesh` can be asked for
the same thing. Putting the two in one table would be an unmatched comparison.

For scale only, and claiming nothing: `generate(2)` on `Block.stp` at `MeshSizeMax = 2.0` did
not finish within 20 minutes on this machine, and was stopped. It solves a harder problem, a
graded quality-controlled mesh for analysis. It is not slower at the same job. It is doing a
different job.

### 9.5 Parallelism is not free

See §7.5. pySMESH parallel uses about 20% more total CPU.

---

## 10. A unit trap that voided an earlier run of this benchmark

This is recorded because it would silently corrupt any comparison of these two libraries, and
because it nearly corrupted this one.

The two importers disagree about length units. See also [Units](../concepts/units.md).

* gmsh's STEP reader rescales every model to **millimetres** (OCCT's `xstep.cascade.unit`
  default).
* `pysmesh.read_step_xde` preserves the file's **declared** unit, and reports it as
  `length_unit` (metres per model unit).

For a file declaring metres, the same numeric deflection is therefore a **1,000x coarser**
request on pySMESH's side. The first run of the real corpus produced this:

| file | declared unit | gmsh triangles | pySMESH triangles | apparent delta |
|---|---|---:|---:|---:|
| `917_fusee.stp` | M | 13,225 | 1,505 | −88.6% |
| `part.step` | M | 972 | 358 | −63.2% |
| `U_Joint_2.stp` | INCH | 9,602 | 3,726 | −61.2% |

pySMESH looked fast on those three files because it had been asked for a much coarser mesh.
The fix is to express the tolerance in millimetres and convert per model:

```python
lin_native = lin_mm / (imp.length_unit * 1000.0)
```

After the fix the same three files agree to within 0.45%, and `part.step` matches exactly at
972 triangles on both sides. Every number on this page uses the corrected harness. The
contaminated first run is kept as `raw_runs_pre_unitfix.jsonl` so the correction can be
audited rather than taken on trust.

A benchmark of these two libraries that does not handle this is measuring units, not speed.

---

## 11. Limitations

1. **One machine, one OS, one compiler pair.** No claim is made about Linux, about other
   processors, or about different core counts. The parallel figure is a function of 14
   logical processors and will differ elsewhere.
2. **The OCCT version difference cannot be removed** without rebuilding one project against
   the other's kernel. See §9.1.
3. **Achieved deviation was measured directly** on the sphere, the cone, and three imported
   bodies. On the remaining cases, matched quality rests on triangle-count agreement, which
   was within 0.53% everywhere.
4. **The 5,606-face assembly is not publicly available**, so its two rows cannot be
   independently reproduced. Every other geometry can be. The quotable claim in §1
   deliberately uses `Block.stp`.
5. **Angular deflection was fixed at 0.5 rad** and not swept. Both sides received the same
   value.
6. **5 repetitions per cell.** Enough to expose the spread reported in the raw data, not
   enough for a confidence interval.
7. `generate(2)` was stopped at 20 minutes rather than run to completion. No conclusion is
   drawn from that, and it is outside the comparison.

---

## 12. Reproduction

The harness is in
[`benchmark/`](https://github.com/KRGulaj/pySMESH/tree/main/benchmark).

```bash
cd benchmark
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt          # pins gmsh 4.15.2 and pySMESH 4.0.0

python corpus_gen.py                     # synthetic ladder + cone

# real geometry comes from a gmsh source checkout, it is not redistributed here
git clone https://gitlab.onelab.info/gmsh/gmsh.git
export GMSH_BENCH=$PWD/gmsh/benchmarks

python bench_driver.py --suite ladder --reps 5
python bench_driver.py --suite sweep  --reps 5
python bench_driver.py --suite real   --reps 5

python verify_quality.py                 # achieved deviation, closed form
python real_deviation.py                 # achieved deviation, OCC projection
python gmsh_overhead.py                  # gmsh BRepMesh vs conversion split
python analyze.py                        # tables + results/summary.csv
```

`AFFINITY` in `bench_driver.py` pins every run to the same logical processors. Edit it to
match your machine before comparing anything. File hashes for the real corpus are in §6.

---

## 13. Raw data

All of it is in
[`benchmark/results/`](https://github.com/KRGulaj/pySMESH/tree/main/benchmark/results).

| file | contents |
|---|---|
| `raw_runs.jsonl` | every timed run: per-repetition wall and CPU times, triangle count, peak RSS, affinity, native and millimetre tolerance |
| `raw_runs_pre_unitfix.jsonl` | the superseded first run, kept so §10 can be audited |
| `summary.csv` | per-cell medians, min, max, CPU/wall ratio, speed-up |
| `quality.jsonl` | achieved deviation against the closed form |
| `real_quality.jsonl` | achieved deviation by OCC projection on imported bodies |
| `gmsh_overhead.jsonl` | gmsh BRepMesh time against conversion time |
| `corpus.json` | corpus metadata and file hashes |
| `tables.md` | the full per-cell table for all 37 cells |

---

## 14. Conclusion

gmsh and pySMESH do not have competing tessellators. They have the same tessellator, OCCT's
`BRepMesh_IncrementalMesh`, reached two different ways. At a matched chordal tolerance they
produce the same number of triangles and the same achieved deviation.

What differs is dispatch. gmsh calls the mesher once per face, in a serial loop, which leaves
OCCT's own parallel mode nothing to distribute. pySMESH calls it once for the whole shape and
lets OCCT spread the faces over threads.

The consequences, measured:

* **1 to 3 faces**: gmsh ties or wins, by up to 14%.
* **26 to 533 faces**: pySMESH is 1.4x to 4.5x faster.
* **many faces of similar size**: up to 9.36x, saturating near 8.8x on 14 logical processors.
* **a 5,606-face assembly with uneven faces**: 2.89x to 4.92x.
* **single-threaded, everywhere**: 0.97x to 1.74x, median 1.26x.
* **memory**: pySMESH peaks 1.4x to 2.4x lower.
* **total CPU**: pySMESH parallel spends about 20% more.

The defensible headline is a **3.26x speed-up on a 533-face model from gmsh's own benchmark
suite, at an identical achieved maximum chordal deviation of 0.1197 mm**, reproduced in two
independent runs.

---

## 15. Software and attribution

### Gmsh

Gmsh is copyright (C) 1997–2026 Christophe Geuzaine and Jean-François Remacle, and is
distributed under the GNU General Public License, version 2 or later, with an exception for
combination with Netgen, METIS, OpenCASCADE and ParaView. Version 4.15.2 was used here,
installed from the official PyPI wheel. No part of Gmsh is redistributed by this repository.

Gmsh's manual asks that work using it cite the following paper, and this page does so
gladly:

> C. Geuzaine and J.-F. Remacle. *Gmsh: a three-dimensional finite element mesh generator
> with built-in pre- and post-processing facilities.* International Journal for Numerical
> Methods in Engineering, **79**(11), pp. 1309–1331, 2009.

The C++ excerpts in §3 are quoted from the Gmsh sources — `src/geo/GModelIO_OCC.cpp` and
`src/geo/GFace.cpp` — for the purpose of technical analysis, and remain under Gmsh's own
licence and copyright. They are reproduced with file and line references so that any reader
can check them against the original rather than take this page's word for it.

### Geometry

The real corpus is taken from the `benchmarks/step` and `benchmarks/statreport` directories
of the Gmsh source distribution. Those CAD models are third-party samples bundled with Gmsh,
not work of this project, and they are **referenced by SHA-256 rather than redistributed**.
See §12 for how to obtain them.

The synthetic corpus is generated by `benchmark/corpus_gen.py` and belongs to this project.

### Open CASCADE Technology

Both libraries perform their tessellation with OCCT's `BRepMesh_IncrementalMesh`. Open
CASCADE Technology is copyright (C) Open CASCADE SAS and is distributed under the GNU Lesser
General Public License, version 2.1, with an additional exception. Gmsh 4.15.2 links OCC
7.8.1; pySMESH 4.0.0 bundles OCCT 8.0.0.

### Independence

This benchmark was produced by the author of pySMESH. It is not endorsed by, affiliated with,
or reviewed by the Gmsh project. The Gmsh name is used only to identify the software
compared. Everything needed to check the result independently — the harness, the raw
per-run data, the exact versions, and the file hashes of every input — is published with it,
precisely so that the reader does not have to trust the author's impartiality.
