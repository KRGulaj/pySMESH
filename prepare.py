"""Stage and patch vendored SMESH/KERNEL/GEOM sources into a build tree.

pySMESH2 vendors pristine upstream sources as git subtrees/slices under ``extern/``
(``extern/smesh``, ``extern/kernel``, ``extern/geom``, ``extern/mefisto2``). Those trees
are never modified. ``prepare.py`` copies the pieces we compile into ``staged/`` in the
directory layout that SALOME/looooo's patch series expects, then applies the patches:

1. **looooo/SMESH** patch series (``patches/{kernel,geom,smesh}/*.patch``) — Windows/MSVC
   shims, VTK 9.4/9.6 API breaks, MED strip, and the source fixes that make vanilla
   SalomePlatform/smesh ``V9_9_0`` build standalone. looooo's ``external/SMESH`` submodule
   *is* SalomePlatform/smesh ``V9_9_0``, so these apply against our ``extern/smesh`` cleanly.
2. **conda-forge/smesh-feedstock** OCCT-8.0 layer (``patches/occt8/*.patch``) — brings the
   OCCT-7.9-ready tree up to OCCT 8.0.0 (conda's authoritative Windows/OCCT-8.0 recipe).

NETGEN is disabled, so NETGEN/NETGENPlugin sources and the ``0005-occt-8.0-netgen`` patch
are intentionally excluded (see docs/reports/B0.md and PROVENANCE.md).

Idempotent: re-running is a no-op once ``staged/.prepared`` exists unless ``--force`` is
given. ``staged/`` is git-ignored.

Usage:
    python prepare.py [--force]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
EXTERN: Path = ROOT / "extern"
PATCHES: Path = ROOT / "patches"
STAGED: Path = ROOT / "staged"
SENTINEL: Path = STAGED / ".prepared"

# KERNEL compiled slice: only these three dirs are built (cmake/Kernel), and looooo proves
# they are include-closed after the strip-corba patch. Source: cmake/Kernel/CMakeLists.txt.
KERNEL_SLICE_DIRS: tuple[str, ...] = ("Basics", "SALOMELocalTrace", "Utils")

# Ordered patch manifest: (patch path relative to patches/, apply-root relative to STAGED).
# Patches are git-format (``a/``/``b/`` prefixes), applied with ``patch -p1`` under each root:
# looooo patches at ``src/<module>`` and conda's OCCT-8.0 layer at the staged top, exactly
# reproducing looooo/prepare.py and conda's recipe. GNU patch is used (not python-patch)
# because our vendored V9_9_0 *tag* is newer than looooo's V9_9_0 *branch* pin and already
# carries some of these fixes: ``-N`` skips already-applied hunks and ``--fuzz`` absorbs the
# minor context offsets between tag and pin. Genuine conflicts still surface as ``*.rej``.
PATCH_MANIFEST: tuple[tuple[str, str], ...] = (
    # --- KERNEL (looooo) : root staged/src/Kernel ---
    ("kernel/Kernel.patch", "src/Kernel"),
    ("kernel/Kernel_occt781.patch", "src/Kernel"),
    ("kernel/Kernel_mingw_gcc15.patch", "src/Kernel"),
    ("kernel/Kernel_msvc_pthread.patch", "src/Kernel"),
    ("kernel/Kernel_msvc_set_unexpected.patch", "src/Kernel"),
    # --- GEOM OCCT-8.0 fix is applied as a string replace (see _apply_geomutils_occt_fix);
    #     looooo's geom/GEOMUtils.patch is context-fragile across GEOM V9_9_0 tag-vs-pin skew ---
    # --- SMESH (looooo) : root staged/src/SMESH ; order mirrors looooo/prepare.py ---
    ("smesh/mefisto.patch", "src/SMESH"),
    ("smesh/SMESH_ControlPnt.patch", "src/SMESH"),
    ("smesh/SMESH_Controls.patch", "src/SMESH"),
    ("smesh/SMESH_Mesh.patch", "src/SMESH"),
    ("smesh/SMESH_MeshAlgos.patch", "src/SMESH"),
    ("smesh/SMESH_Slot.patch", "src/SMESH"),
    ("smesh/SMESH_SMDS.patch", "src/SMESH"),
    ("smesh/StdMeshers_Adaptive1D.patch", "src/SMESH"),
    ("smesh/StdMeshers_Projection_2D.patch", "src/SMESH"),
    ("smesh/StdMeshers_ViscousLayers.patch", "src/SMESH"),
    ("smesh/SMESH_occt781.patch", "src/SMESH"),
    ("smesh/SMDS_UnstructuredGrid_vtk94.patch", "src/SMESH"),
    # (SMDS_Mesh.cxx vtkPoints alloc fix is applied here as a string replace)
    ("smesh/SMDS_MeshVolume_vtk96.patch", "src/SMESH"),
    ("smesh/SMDS_VtkCellIterator_vtk96.patch", "src/SMESH"),
    ("smesh/SMESH_MeshEditor_vtk96.patch", "src/SMESH"),
    ("smesh/SMESH_File_mingw.patch", "src/SMESH"),
    ("smesh/SMESH_MesherHelper_msvc.patch", "src/SMESH"),
    ("smesh/StdMeshers_Quadrangle_2D_msvc.patch", "src/SMESH"),
    # --- OCCT 8.0 layer (conda) : root staged/ ---
    ("occt8/0003-boost-regex-str-enum.patch", "."),
    ("occt8/0004-occt-8.0-compat.patch", "."),
)

# The looooo VTK-alloc string replace is sequenced right after the vtk94 patch.
_VTK_ALLOC_AFTER: str = "smesh/SMDS_UnstructuredGrid_vtk94.patch"


def _log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def _copytree(src: Path, dst: Path) -> None:
    """Copy ``src`` onto ``dst`` (dst parent created); fail loudly if src is missing."""
    if not src.exists():
        raise FileNotFoundError(f"expected vendored source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _stage_sources() -> None:
    """Copy the compiled slices from extern/ into staged/src/{Kernel,Geom,SMESH}/src."""
    _log("staging KERNEL slice")
    for sub in KERNEL_SLICE_DIRS:
        _copytree(EXTERN / "kernel/src" / sub, STAGED / "src/Kernel/src" / sub)

    _log("staging GEOMUtils slice")
    _copytree(EXTERN / "geom/src/GEOMUtils", STAGED / "src/Geom/src/GEOMUtils")

    _log("staging SMESH src")
    _copytree(EXTERN / "smesh/src", STAGED / "src/SMESH/src")

    _log("staging MEFISTO2 f2c trte.c")
    trte = EXTERN / "mefisto2/trte.c"
    if not trte.exists():
        raise FileNotFoundError(f"expected vendored source missing: {trte}")
    shutil.copyfile(trte, STAGED / "src/SMESH/src/MEFISTO2/trte.c")


def _rej_files() -> set[Path]:
    return set(STAGED.rglob("*.rej"))


def _hunk_tally(patch_stdout: str) -> tuple[int, int]:
    """Sum (failed, total) hunks across all files from GNU patch's summary lines.

    Lines look like ``N out of M hunks FAILED -- saving rejects to file ...``. Files with no
    failures don't emit such a line, so ``total`` here is the count over *failing files only*
    — which is exactly what we need to decide "every hunk of this file failed" (superseded).
    """
    import re

    failed = total = 0
    for m in re.finditer(r"(\d+) out of (\d+) hunks? FAILED", patch_stdout):
        failed += int(m.group(1))
        total += int(m.group(2))
    return failed, total


def _apply(patch_rel: str, root_rel: str) -> None:
    """Apply one patch at the given staged root with GNU patch.

    ``-N`` skips hunks already present in our (newer) tag tree; ``--fuzz=2`` tolerates the
    small context offsets between the V9_9_0 tag and looooo's branch pin. A genuine failure
    is a *new* ``.rej`` file — that raises, and its contents are logged for the fixup.
    """
    patch_path = PATCHES / patch_rel
    if not patch_path.exists():
        raise FileNotFoundError(f"patch not found: {patch_path}")
    root = (STAGED / root_rel).resolve()
    before = _rej_files()
    proc = subprocess.run(
        ["patch", "-p1", "-N", "--fuzz=2", "--no-backup-if-mismatch",
         "-i", str(patch_path)],
        cwd=str(root), capture_output=True, text=True,
    )
    new_rejs = _rej_files() - before
    if not new_rejs:
        status = "applied"
        if "ignored" in proc.stdout or "previously applied" in proc.stdout:
            status = "applied (some hunks already in tag)"
        _log(f"{status}: {patch_rel}")
        return

    # A reject appeared. Distinguish "fully superseded by our newer tag" (every hunk failed —
    # the file already carries an equivalent/newer fix) from a genuine partial conflict.
    failed, total = _hunk_tally(proc.stdout)
    if total > 0 and failed == total:
        for rej in new_rejs:
            rej.unlink()
        _log(f"SKIPPED (superseded by V9_9_0 tag; {failed}/{total} hunks already covered): "
             f"{patch_rel}")
        return

    detail = "\n".join(f"  {r.relative_to(STAGED)}" for r in sorted(new_rejs))
    raise RuntimeError(
        f"PARTIAL conflict applying {patch_rel} at {root_rel} "
        f"({failed}/{total} hunks failed); rejects:\n{detail}\n"
        f"--- patch stdout ---\n{proc.stdout}\n--- patch stderr ---\n{proc.stderr}"
    )


def _apply_smds_mesh_vtk_alloc() -> None:
    """VTK 9: pre-allocate vtkPoints to avoid an InsertPoint crash on Windows.

    ``SetNumberOfPoints`` allocates the array; ``Allocate`` alone only reserves capacity.
    Source: looooo/SMESH/prepare.py :: _apply_smds_mesh_vtk_alloc.
    """
    target = STAGED / "src/SMESH/src/SMDS/SMDS_Mesh.cxx"
    old = "  points->SetNumberOfPoints( 0 );\n  myGrid->SetPoints( points );"
    new = "  points->SetNumberOfPoints( chunkSize );\n  myGrid->SetPoints( points );"
    content = target.read_text(encoding="utf-8", errors="surrogateescape")
    if new in content:
        _log("vtkPoints alloc fix already present")
        return
    if old not in content:
        raise RuntimeError(f"vtkPoints alloc fix: pattern not found in {target}")
    target.write_text(content.replace(old, new), encoding="utf-8",
                      errors="surrogateescape")
    _log("applied vtkPoints alloc fix (SMDS_Mesh.cxx)")


def _replace_once(target: Path, old: str, new: str) -> None:
    """Apply a single exact string replacement; idempotent; raise if the anchor is absent."""
    content = target.read_text(encoding="utf-8", errors="surrogateescape")
    if new in content:
        return
    if old not in content:
        raise RuntimeError(f"fixup anchor not found in {target}: {old!r}")
    target.write_text(content.replace(old, new, 1), encoding="utf-8",
                      errors="surrogateescape")


def _apply_tag_fixups() -> None:
    """V9_9_0-*tag*-specific source deltas not covered by the looooo/conda patch series.

    The looooo/conda patches target looooo's V9_9_0 *branch* pin; our vendored V9_9_0 *tag*
    differs in a few files, needing these OCCT-8.0 / Windows deltas on top (CORBA in OpUtil
    is instead disabled via the SALOME_LIGHT compile definition in CMakeLists.txt):
      * Basics_Utils.cxx : gethostname() needs <winsock2.h> on Windows (ws2_32 already linked).
      * SMESH_TypeDefs.hxx: SMESH_TLink needs a default ctor for OCCT 8.0 NCollection maps.
    """
    _replace_once(
        STAGED / "src/Kernel/src/Basics/Basics_Utils.cxx",
        "#ifndef WIN32\n#include <unistd.h>\n#include <sys/stat.h>\n#include <execinfo.h>\n#endif",
        "#ifndef WIN32\n#include <unistd.h>\n#include <sys/stat.h>\n#include <execinfo.h>\n"
        "#else\n#include <winsock2.h>\n#endif")
    _replace_once(
        STAGED / "src/SMESH/src/SMESHUtils/SMESH_TypeDefs.hxx",
        "struct SMESH_TLink: public NLink\n{",
        "struct SMESH_TLink: public NLink\n{\n"
        "  SMESH_TLink() {} // default ctor required by OCCT 8.0 NCollection maps")
    # OCCT 8.0 calls a map's hasher as a functor (myHasher(key)); SMESH_TLink is used as its
    # own hasher in NCollection_DataMap<SMESH_TLink,int,SMESH_TLink>, so give it the functor
    # interface (mirrors the adjacent SMESH_TLinkHasher) in addition to the old static API.
    _replace_once(
        STAGED / "src/SMESH/src/SMESHUtils/SMESH_TypeDefs.hxx",
        "  static Standard_Boolean IsEqual(const SMESH_TLink& l1, const SMESH_TLink& l2)\n"
        "  {\n"
        "    return ( l1.node1() == l2.node1() && l1.node2() == l2.node2() );\n"
        "  }\n"
        "};",
        "  static Standard_Boolean IsEqual(const SMESH_TLink& l1, const SMESH_TLink& l2)\n"
        "  {\n"
        "    return ( l1.node1() == l2.node1() && l1.node2() == l2.node2() );\n"
        "  }\n"
        "  size_t operator()(const SMESH_TLink& link) const\n"
        "  { return smIdHasher()( link.node1()->GetID() + link.node2()->GetID() ); }\n"
        "  bool operator()(const SMESH_TLink& l1, const SMESH_TLink& l2) const\n"
        "  { return ( l1.node1() == l2.node1() && l1.node2() == l2.node2() ); }\n"
        "};")
    # ElementsOnShape holds std::vector<Classifier> with Classifier only forward-declared in
    # the header; MSVC eagerly instantiates the implicit copy ctor/operator= with an
    # incomplete type (C2036) in every TU that copies the predicate (incl.
    # StdMeshers_ViscousLayers). Make copy operations out-of-line so vector<Classifier> is
    # instantiated only in SMESH_Controls.cxx, where Classifier is fully defined.
    _replace_once(
        STAGED / "src/SMESH/src/Controls/SMESH_ControlsDef.hxx",
        "      ElementsOnShape();\n      ~ElementsOnShape();",
        "      ElementsOnShape();\n"
        "      ElementsOnShape(const ElementsOnShape&);\n"
        "      ElementsOnShape& operator=(const ElementsOnShape&);\n"
        "      ~ElementsOnShape();")
    _replace_once(
        STAGED / "src/SMESH/src/Controls/SMESH_Controls.cxx",
        "ElementsOnShape::~ElementsOnShape()\n{\n  clearClassifiers();\n}",
        "ElementsOnShape::~ElementsOnShape()\n{\n  clearClassifiers();\n}\n\n"
        "// Out-of-line so std::vector<Classifier> instantiates where Classifier is complete.\n"
        "ElementsOnShape::ElementsOnShape(const ElementsOnShape&) = default;\n"
        "ElementsOnShape& ElementsOnShape::operator=(const ElementsOnShape&) = default;")
    _log("applied V9_9_0-tag fixups (winsock, SMESH_TLink ctor+hasher, ElementsOnShape copy)")


def _apply_geomutils_occt_fix() -> None:
    """Drop ``V3d_Coordinate`` (removed in recent OCCT) from GEOMUtils.cxx.

    Equivalent to looooo's ``geom/GEOMUtils.patch`` but applied as a string replace so it is
    robust to the 1-line context skew between GEOM ``V9_9_0`` tag (`b6f0965`, what we vendor)
    and looooo's Geom submodule pin (`71b630d7`). ``ConvertClickToPoint`` is a headless-unused
    GUI helper, but the whole .cxx must still compile, so the removed type must resolve.
    Source of the edits: looooo/SMESH/patch/GEOMUtils.patch.
    """
    target = STAGED / "src/Geom/src/GEOMUtils/GEOMUtils.cxx"
    content = target.read_text(encoding="utf-8", errors="surrogateescape")
    edits = (
        ("#include <V3d_Coordinate.hxx>\n\n", ""),
        ("V3d_Coordinate XEye, YEye, ZEye, XAt, YAt, ZAt;",
         "Standard_Real XEye, YEye, ZEye, XAt, YAt, ZAt;"),
    )
    for old, new in edits:
        if old not in content:
            raise RuntimeError(f"GEOMUtils OCCT fix: pattern not found in {target}: {old!r}")
        content = content.replace(old, new)
    target.write_text(content, encoding="utf-8", errors="surrogateescape")
    _log("applied GEOMUtils OCCT fix (V3d_Coordinate removal)")


def prepare(force: bool = False) -> None:
    if SENTINEL.exists() and not force:
        _log(f"already prepared ({SENTINEL} exists); pass --force to rebuild")
        return
    if STAGED.exists():
        _log("removing existing staged/ tree")
        shutil.rmtree(STAGED)
    STAGED.mkdir(parents=True)

    _stage_sources()
    _apply_geomutils_occt_fix()

    _log("applying patch series")
    for patch_rel, root_rel in PATCH_MANIFEST:
        _apply(patch_rel, root_rel)
        if patch_rel == _VTK_ALLOC_AFTER:
            _apply_smds_mesh_vtk_alloc()

    _apply_tag_fixups()
    SENTINEL.write_text("prepared\n", encoding="utf-8")
    _log(f"done: staged tree ready at {STAGED}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage and patch SMESH sources.")
    parser.add_argument("--force", action="store_true",
                        help="rebuild staged/ even if the sentinel exists")
    args = parser.parse_args()
    try:
        prepare(force=args.force)
    except (FileNotFoundError, RuntimeError) as exc:
        _log(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
