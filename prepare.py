"""Stage and patch vendored SALOME sources into a build tree.

pySMESH vendors pristine upstream sources under ``extern/``. Those trees are never
modified. Every SALOME tree is at tag ``V9_16_0``:

* ``extern/smesh``: SALOME SMESH, a squashed git subtree.
* ``extern/kernel``: SALOME KERNEL, a squashed git subtree. Only ``src/Utils`` is
  compiled.
* ``extern/salome_bootstrap``: SALOME salome_bootstrap, a squashed git subtree. Since
  ``V9_16_0`` it holds the KERNEL basics: ``Basics/``, ``SALOMELocalTrace/`` and the
  ``SALOME_Exception`` class (``Exception/``).
* ``extern/geom/src/GEOMUtils``: one directory of SALOME GEOM, a sparse copy.
* ``extern/mefisto2``: the MEFISTO 2-D triangulator, carried forward verbatim from SMESH
  ``V9_9_0`` (SALOME removed it in 2022), and looooo's f2c translation ``trte.c``.
* ``extern/netgen``: netgen at tag ``v6.2.2101``, a sparse copy of ``libsrc/``,
  ``nglib/`` and the licence.
* ``extern/netgenplugin``: SALOME NETGENPlugin at tag ``V9_16_0``, a sparse copy of
  ``src/NETGENPlugin/`` and the licence.

``extern/zlib`` (zlib 1.3.2, the top-level files of the release) is compiled where it is
and is not staged: no patch touches it.

``prepare.py`` copies the pieces we compile into ``staged/``, in the layout that the
looooo/SMESH patch series expects. Then it applies the patches and the source edits:

1. The **looooo/SMESH** patch series (``patches/{kernel,smesh}/*.patch``): Windows/MSVC
   shims, the MED strip and the MEFISTO f2c wiring.
2. Two **pySMESH** patches for code that is new in ``V9_16_0``:
   ``patches/geom/GEOMUtils_GEOMAlgo.patch`` and
   ``patches/smesh/SMESH_Gen_no_qt.patch``.
3. The **conda-forge/smesh-feedstock** OCCT 8.0 layer (``patches/occt8/*.patch``).
4. The netgen series (``patches/netgen/*.patch``): SALOME's ``netgen62ForSalome.patch``
   verbatim, then the OCCT 8 port from looooo/SMESH and conda-forge, then the pySMESH
   patches and the fixes backported from later netgen releases (each named after its
   upstream commit).
5. The NETGENPlugin series (``patches/netgenplugin/*.patch``): the OCCT 8 port from
   conda-forge, then the pySMESH patches that remove CORBA and SALOMEDS and keep the
   plugin inside the host process's contract.
6. The source edits in this file that no patch carries (see ``_apply_source_edits``).

Every patch is re-ported to its tree and must apply exactly: every hunk at fuzz 0. A
hunk that fails, is already applied, or targets a file that is not staged stops the run.

Idempotent: re-running is a no-op once ``staged/.prepared`` exists unless ``--force`` is
given. ``staged/`` is git-ignored.

Usage:
    python prepare.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

ROOT: Final[Path] = Path(__file__).resolve().parent
EXTERN: Final[Path] = ROOT / "extern"
PATCHES: Final[Path] = ROOT / "patches"
STAGED: Final[Path] = ROOT / "staged"
SENTINEL: Final[Path] = STAGED / ".prepared"

# KERNEL compiled slice: (source under extern/, destination under
# staged/src/Kernel/src). cmake/Kernel/CMakeLists.txt compiles these four directories.
# Since V9_16_0 SALOME keeps Basics/, SALOMELocalTrace/ and Exception/ in
# salome_bootstrap, and Utils/ in KERNEL. The destination keeps the flat layout that the
# KERNEL patches and the CMake files expect.
KERNEL_SLICE: Final[tuple[tuple[str, str], ...]] = (
    ("salome_bootstrap/__RUN_SALOME__/Basics", "Basics"),
    ("salome_bootstrap/__RUN_SALOME__/SALOMELocalTrace", "SALOMELocalTrace"),
    ("salome_bootstrap/__RUN_SALOME__/Exception", "Exception"),
    ("kernel/src/Utils", "Utils"),
)

# MEFISTO carry-forward: (file in extern/mefisto2, directory under
# staged/src/SMESH/src). Each file goes where SMESH V9_9_0 kept it, so mefisto.patch and
# the CMake files find it.
MEFISTO_FILES: Final[tuple[tuple[str, str], ...]] = (
    ("aptrte.cxx", "MEFISTO2"),
    ("aptrte.h", "MEFISTO2"),
    ("Rn.h", "MEFISTO2"),
    ("trte.c", "MEFISTO2"),
    ("StdMeshers_MEFISTO_2D.cxx", "StdMeshers"),
    ("StdMeshers_MEFISTO_2D.hxx", "StdMeshers"),
)

# Ordered patch manifest: (patch path relative to patches/, apply-root relative to
# STAGED). Patches are git-format (``a/``/``b/`` prefixes), applied with ``patch -p1``
# under each root: looooo patches at ``src/<module>`` and conda's OCCT 8.0 layer at the
# staged top, as in looooo/prepare.py and conda's recipe. PROVENANCE.md lists the
# patches that V9_16_0 made obsolete.
PATCH_MANIFEST: Final[tuple[tuple[str, str], ...]] = (
    # --- KERNEL (looooo) : root staged/src/Kernel ---
    ("kernel/Kernel.patch", "src/Kernel"),
    ("kernel/Kernel_mingw_gcc15.patch", "src/Kernel"),
    ("kernel/Kernel_msvc_pthread.patch", "src/Kernel"),
    ("kernel/Kernel_msvc_set_unexpected.patch", "src/Kernel"),
    # --- GEOM (pySMESH) : root staged/src/Geom ---
    ("geom/GEOMUtils_GEOMAlgo.patch", "src/Geom"),
    # --- SMESH (looooo) : root staged/src/SMESH ; order mirrors looooo/prepare.py ---
    ("smesh/mefisto.patch", "src/SMESH"),
    ("smesh/SMESH_Mesh.patch", "src/SMESH"),
    ("smesh/SMESH_Gen_no_qt.patch", "src/SMESH"),
    ("smesh/SMESH_MeshAlgos.patch", "src/SMESH"),
    ("smesh/SMESH_Slot.patch", "src/SMESH"),
    ("smesh/SMESH_SMDS.patch", "src/SMESH"),
    ("smesh/SMESH_occt781.patch", "src/SMESH"),
    ("smesh/SMESH_File_mingw.patch", "src/SMESH"),
    ("smesh/StdMeshers_Quadrangle_2D_msvc.patch", "src/SMESH"),
    # --- SMESH (pySMESH) : root staged/src/SMESH ---
    ("smesh/StdMeshers_Cartesian_3D_cancel.patch", "src/SMESH"),
    ("smesh/StdMeshers_Cartesian_VL_cancel.patch", "src/SMESH"),
    ("smesh/StdMeshers_Adaptive1D_deflection.patch", "src/SMESH"),
    ("smesh/StdMeshers_Adaptive1D_bounds.patch", "src/SMESH"),
    ("smesh/StdMeshers_Adaptive1D_short_edges.patch", "src/SMESH"),
    ("smesh/StdMeshers_Distribution_table.patch", "src/SMESH"),
    ("smesh/StdMeshers_Distribution_expression.patch", "src/SMESH"),
    ("smesh/StdMeshers_Prism_3D_composite_side.patch", "src/SMESH"),
    ("smesh/SMESH_Mesh_hypothesis_status.patch", "src/SMESH"),
    ("smesh/StdMeshers_ViscousLayerBuilder_lifecycle.patch", "src/SMESH"),
    ("smesh/StdMeshers_Cartesian_VL_group_2d.patch", "src/SMESH"),
    ("smesh/StdMeshers_Cartesian_VL_duplicate_nodes.patch", "src/SMESH"),
    ("smesh/StdMeshers_Cartesian_3D_offset_mesh_leak.patch", "src/SMESH"),
    ("smesh/StdMeshers_Cartesian_3D_viscous_submeshes.patch", "src/SMESH"),
    ("smesh/StdMeshers_Cartesian_3D_offset_small_cells.patch", "src/SMESH"),
    ("smesh/StdMeshers_Cartesian_VL_offset_makers_leak.patch", "src/SMESH"),
    ("smesh/SMDS_UnstructuredGrid_links_leak.patch", "src/SMESH"),
    ("smesh/MEFISTO_2D_max_element_area.patch", "src/SMESH"),
    ("smesh/StdMeshers_CompositeHexa_3D_viscous_layers.patch", "src/SMESH"),
    ("smesh/StdMeshers_Cartesian_VL_offset_error.patch", "src/SMESH"),
    # --- OCCT 8.0 layer (conda) : root staged/ ---
    ("occt8/0003-boost-regex-str-enum.patch", "."),
    ("occt8/0004-occt-8.0-compat.patch", "."),
    # --- netgen 6.2.2101 : root staged/src/Netgen, or staged/ for the conda layout ---
    ("netgen/netgen62ForSalome.patch", "src/Netgen"),
    ("netgen/occgenmesh_OCCT76.patch", "src/Netgen"),
    ("netgen/Partition_Loop3d_occt781.patch", "src/Netgen"),
    ("netgen/0004-occt-8.0-netgen-partition.patch", "."),
    ("netgen/0005-occt-8.0-netgen-occ.patch", "."),
    ("netgen/occgeom_save_without_stl.patch", "src/Netgen"),
    ("netgen/netgen_console_writes.patch", "src/Netgen"),
    ("netgen/netgen_no_ngprofile.patch", "src/Netgen"),
    ("netgen/e1d71a78_no_need_to_remove_archive_type_infos.patch", "src/Netgen"),
    # --- NETGENPlugin V9_16_0 : root staged/src/NETGENPlugin (looooo's layout) ---
    ("netgenplugin/NETGENPlugin_occt8.patch", "src/NETGENPlugin"),
    ("netgenplugin/NETGENPlugin_local_size_by_subshape.patch", "src/NETGENPlugin"),
    ("netgenplugin/NETGENPlugin_runtime_containment.patch", "src/NETGENPlugin"),
    ("netgenplugin/NETGENPlugin_edge_local_size_ends.patch", "src/NETGENPlugin"),
    ("netgenplugin/NETGENPlugin_face_maxh_index.patch", "src/NETGENPlugin"),
    ("netgenplugin/NETGENPlugin_curvature_before_read.patch", "src/NETGENPlugin"),
    ("netgenplugin/NETGENPlugin_debug_text_threads.patch", "src/NETGENPlugin"),
    ("netgenplugin/NETGENPlugin_remesher_stl_topology.patch", "src/NETGENPlugin"),
    ("netgenplugin/NETGENPlugin_remesher_no_parameters.patch", "src/NETGENPlugin"),
)

# netgen slice: the directories of extern/netgen that prepare.py copies to
# staged/src/Netgen. That is the layout of looooo/SMESH and conda-forge, whose netgen
# patches name their files under src/Netgen.
NETGEN_SLICE: Final[tuple[str, ...]] = ("libsrc", "nglib")


def _copytree(src: Path, dst: Path) -> None:
    """Copy ``src`` onto ``dst`` (dst parent created); fail loudly if src is missing."""
    if not src.is_dir():
        raise FileNotFoundError(f"expected vendored source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _stage_sources() -> None:
    """Copy the compiled slices from extern/ into staged/src/{Kernel,Geom,SMESH,Netgen}."""
    logger.info("staging KERNEL slice (salome_bootstrap + kernel)")
    for src_rel, dst_rel in KERNEL_SLICE:
        _copytree(EXTERN / src_rel, STAGED / "src/Kernel/src" / dst_rel)

    logger.info("staging GEOMUtils slice")
    _copytree(EXTERN / "geom/src/GEOMUtils", STAGED / "src/Geom/src/GEOMUtils")

    logger.info("staging SMESH src")
    _copytree(EXTERN / "smesh/src", STAGED / "src/SMESH/src")

    logger.info("staging MEFISTO carry-forward (extern/mefisto2)")
    for name, dst_dir in MEFISTO_FILES:
        src = EXTERN / "mefisto2" / name
        if not src.is_file():
            raise FileNotFoundError(f"expected vendored source missing: {src}")
        dst = STAGED / "src/SMESH/src" / dst_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    logger.info("staging netgen slice (extern/netgen)")
    for name in NETGEN_SLICE:
        _copytree(EXTERN / "netgen" / name, STAGED / "src/Netgen" / name)

    logger.info("staging NETGENPlugin sources (extern/netgenplugin)")
    _copytree(
        EXTERN / "netgenplugin/src/NETGENPlugin",
        STAGED / "src/NETGENPlugin/src/NETGENPlugin",
    )


def _apply(patch_rel: str, root_rel: str) -> None:
    """Apply one patch at the given staged root with GNU patch, exactly.

    ``--fuzz=0`` makes every hunk match its full context; an offset is accepted, because
    the context still matched line for line. ``-N`` turns an already-applied hunk into
    an ignored one instead of a reversal. Closed stdin stops patch from asking for a
    file it cannot find. Patch exits non-zero on a failed, ignored or unreachable hunk;
    that raises with the full patch output.

    Args:
        patch_rel: Patch path relative to ``patches/``.
        root_rel: Apply root relative to ``staged/``.

    Raises:
        FileNotFoundError: The patch file does not exist.
        RuntimeError: Any hunk did not apply.
    """
    patch_path = PATCHES / patch_rel
    if not patch_path.is_file():
        raise FileNotFoundError(f"patch not found: {patch_path}")
    root = (STAGED / root_rel).resolve()
    proc = subprocess.run(
        [
            "patch",
            "-p1",
            "-N",
            "--fuzz=0",
            "--no-backup-if-mismatch",
            "-i",
            str(patch_path),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        rejects = "\n".join(
            f"  {r.relative_to(STAGED)}" for r in sorted(STAGED.rglob("*.rej"))
        )
        raise RuntimeError(
            f"patch {patch_rel} at {root_rel} did not apply exactly "
            f"(exit {proc.returncode}); rejects:\n{rejects}\n"
            f"--- patch stdout ---\n{proc.stdout}\n--- patch stderr ---\n{proc.stderr}"
        )
    logger.info("applied: %s", patch_rel)


def _replace_once(target: Path, old: str, new: str) -> None:
    """Apply one exact string replacement; idempotent; raise if the anchor is absent.

    Args:
        target: File to edit in place.
        old: Anchor text, which must occur in the file.
        new: Replacement text.

    Raises:
        RuntimeError: Neither the anchor nor its replacement is in the file.
    """
    content = target.read_text(encoding="utf-8", errors="surrogateescape")
    if new in content:
        return
    if old not in content:
        raise RuntimeError(f"fixup anchor not found in {target}: {old!r}")
    target.write_text(
        content.replace(old, new, 1), encoding="utf-8", errors="surrogateescape"
    )


def _apply_smds_mesh_vtk_alloc() -> None:
    """VTK 9: pre-allocate vtkPoints to avoid an InsertPoint crash on Windows.

    ``SetNumberOfPoints`` allocates the array; ``Allocate`` alone only reserves
    capacity. Both sites (the constructor and ``Clear``) carry the anchor at V9_16_0.
    Source: looooo/SMESH/prepare.py :: _apply_smds_mesh_vtk_alloc.

    Raises:
        RuntimeError: The anchor is not in ``SMDS_Mesh.cxx``.
    """
    target = STAGED / "src/SMESH/src/SMDS/SMDS_Mesh.cxx"
    old = "  points->SetNumberOfPoints( 0 );\n  myGrid->SetPoints( points );"
    new = "  points->SetNumberOfPoints( chunkSize );\n  myGrid->SetPoints( points );"
    content = target.read_text(encoding="utf-8", errors="surrogateescape")
    if old not in content:
        raise RuntimeError(f"vtkPoints alloc fix: pattern not found in {target}")
    target.write_text(
        content.replace(old, new), encoding="utf-8", errors="surrogateescape"
    )
    logger.info("applied vtkPoints alloc fix (SMDS_Mesh.cxx)")


# One-line comment inserted before each EvalD0 override (see _apply_source_edits).
_EVALD0_NOTE: Final[str] = (
    "    // OCCT 8.0: Value is no longer virtual;"
    " EvalD0 is the evaluation entry point.\n"
)


def _apply_source_edits() -> None:
    """Source edits that no upstream patch carries, applied on the V9_16_0 tree.

    Each edit is an exact string replacement; a missing anchor raises. CORBA in OpUtil
    is disabled by the SALOME_LIGHT compile definition in CMakeLists.txt, not here.
    PROVENANCE.md records each edit and the V9_9_0 edits that V9_16_0 made obsolete.
    """
    # Basics_Utils.cxx calls gethostname(), which needs <winsock2.h> on Windows; the
    # global WIN32_LEAN_AND_MEAN keeps <windows.h> from pulling it in (ws2_32 is
    # linked).
    _replace_once(
        STAGED / "src/Kernel/src/Basics/Basics_Utils.cxx",
        "#ifndef WIN32\n#include <unistd.h>\n#include <sys/stat.h>\n"
        "#include <execinfo.h>\n#endif",
        "#ifndef WIN32\n#include <unistd.h>\n#include <sys/stat.h>\n"
        "#include <execinfo.h>\n#else\n#include <winsock2.h>\n#endif",
    )
    # StdMeshers_CompositeHexa_3D.hxx carries the include guard of
    # StdMeshers_CompositeSegment_1D (_SMESH_CompositeSegment_1D_HXX_) verbatim, so
    # whichever of the two is included second is silenced entirely and its class is
    # never declared. Any translation unit that needs both cannot be made to compile by
    # reordering, because the collision is symmetric. Give the header its own guard.
    _replace_once(
        STAGED / "src/SMESH/src/StdMeshers/StdMeshers_CompositeHexa_3D.hxx",
        "#ifndef _SMESH_CompositeSegment_1D_HXX_\n"
        "#define _SMESH_CompositeSegment_1D_HXX_",
        "#ifndef _SMESH_CompositeHexa_3D_HXX_\n#define _SMESH_CompositeHexa_3D_HXX_",
    )
    # StdMeshers_Prism_3D caches three helper algorithms in function-local statics, each
    # constructed against the FIRST SMESH_Gen it ever sees. That is safe in SALOME,
    # which has one process-global generator, and unsafe here: pySMESH gives each Mesher
    # its own generator, and ~SMESH_Gen nullifies the _gen of every hypothesis
    # registered with it, these singletons included. A second Prism_3D compute in the
    # same process then works through a singleton whose generator is gone, and
    # segfaults.
    #
    # Rebuild the singleton whenever the generator differs from the one it was built
    # against. Deleting the stale one is safe in both directions: ~SMESH_Hypothesis is
    # guarded on `_gen` for the dead-generator case, and merely un-registers itself for
    # the live one.
    for kind in ("TQuadrangleAlgo", "TProjction1dAlgo", "TProjction2dAlgo"):
        _replace_once(
            STAGED / "src/SMESH/src/StdMeshers/StdMeshers_Prism_3D.cxx",
            f"      static {kind}* algo = new {kind}( fatherAlgo->GetGen() );",
            f"      static {kind}* algo = 0;\n"
            f"      if ( !algo || algo->GetGen() != fatherAlgo->GetGen() )\n"
            "      {\n"
            "        delete algo; // its SMESH_Gen is gone;"
            " ~SMESH_Hypothesis guards on _gen\n"
            f"        algo = new {kind}( fatherAlgo->GetGen() );\n"
            "      }",
        )
    # OCCT 8.0 made Adaptor3d_Curve::Value and Adaptor3d_Surface::Value NON-virtual
    # inlines that forward to a new virtual EvalD0, whose base implementation raises
    # Standard_NotImplemented. The three 3-D adaptors of Prism_3D still define Value,
    # which now merely *hides* the base one: every call through an Adaptor3d_Curve or
    # Adaptor3d_Surface reference reaches the base EvalD0 and throws. For the two curve
    # adaptors that failed Prism_3D on every solid; for TSideFace it fails every compute
    # that reaches the block approach (SMESH_Block::TFace::Point calls Value through the
    # base, report B1). Overriding EvalD0 to forward to their own Value restores both
    # paths. Adaptor2d_Curve2d::Value is still virtual in 8.0.1, so the 2-D adaptors
    # (TPCurveOnHorFaceAdaptor, the Adaptor2dCurve2d of StdMeshers_FaceSide and
    # GEOMUtils::TrsfCurve2d) need nothing.
    prism_hxx = STAGED / "src/SMESH/src/StdMeshers/StdMeshers_Prism_3D.hxx"
    side_face = (
        "    // redefine Adaptor methods\n"
        "    gp_Pnt Value(const Standard_Real U,const Standard_Real V) const;"
    )
    _replace_once(
        prism_hxx,
        side_face,
        side_face
        + "\n"
        + _EVALD0_NOTE
        + "    gp_Pnt EvalD0(const Standard_Real U, const Standard_Real V) const"
        " override\n"
        "    { return Value(U, V); }",
    )
    curve_evald0 = (
        "    gp_Pnt EvalD0(const Standard_Real U) const override { return Value(U); }"
    )
    vertical = (
        "    TVerticalEdgeAdaptor( const TParam2ColumnMap* columnsMap, "
        "const double parameter );\n"
        "    gp_Pnt Value(const Standard_Real U) const;"
    )
    _replace_once(prism_hxx, vertical, vertical + "\n" + _EVALD0_NOTE + curve_evald0)
    horizontal = (
        "      :mySide(sideFace), myV( isTop ? 1.0 : 0.0 ) {}\n"
        "    gp_Pnt Value(const Standard_Real U) const;"
    )
    _replace_once(
        prism_hxx, horizontal, horizontal + "\n" + _EVALD0_NOTE + curve_evald0
    )
    # ManifoldPart::process() walks its face vector from the requested start element
    # and wraps at the end, but it advances the index itself and the wrap sits AFTER a
    # `continue` that skips an already-treated face. So the moment the last face has
    # already been treated (the ordinary case, since findConnected() treats a whole
    # connected region at once) the index runs past the end and the process reads
    # unallocated memory. With the start element at index 0 the loop also never
    # terminates by its own condition. Both are reachable from a plain selection;
    # measured as an access violation on a three-face fixture. Rewritten as a bounded
    # modulo walk, which is the documented intent: visit every face exactly once,
    # starting at the requested one.
    _replace_once(
        STAGED / "src/SMESH/src/Controls/SMESH_Controls.cxx",
        "  const int aStartIndx = myAllFacePtrIntDMap[aStartFace];\n"
        "  bool isStartTreat = false;\n"
        "  for ( int fi = aStartIndx; !isStartTreat || fi != aStartIndx ; fi++ )\n"
        "  {\n"
        "    if ( fi == aStartIndx )\n"
        "      isStartTreat = true;\n"
        "    // as result next time when fi will be equal to aStartIndx\n"
        "\n"
        "    SMDS_MeshFace* aFacePtr = myAllFacePtr[ fi ];",
        "  const int aStartIndx = myAllFacePtrIntDMap[aStartFace];\n"
        "  const int aNbFaces   = (int) myAllFacePtr.size();\n"
        "  // Visit every face exactly once, starting at aStartIndx and wrapping."
        " Indexing the\n"
        "  // vector modulo its size keeps the walk in bounds whatever the body does.\n"
        "  for ( int fj = 0; fj < aNbFaces; fj++ )\n"
        "  {\n"
        "    const int fi = ( aStartIndx + fj ) % aNbFaces;\n"
        "    SMDS_MeshFace* aFacePtr = myAllFacePtr[ fi ];",
    )
    _replace_once(
        STAGED / "src/SMESH/src/Controls/SMESH_Controls.cxx",
        "    if ( fi == int( myAllFacePtr.size() - 1 ))\n"
        "      fi = 0;\n"
        "  } // end run on vector of faces",
        "  } // end run on vector of faces; the wrap now lives in the loop head",
    )
    logger.info(
        "applied source edits (winsock, CompositeHexa_3D include guard, Prism_3D "
        "per-generator singletons, Prism_3D adaptors EvalD0, ManifoldPart bounded walk)"
    )


def prepare(force: bool = False) -> None:
    """Build ``staged/`` from ``extern/``: stage, patch, edit, then write the sentinel.

    Args:
        force: Rebuild ``staged/`` even if the sentinel exists.
    """
    if SENTINEL.exists() and not force:
        logger.info("already prepared (%s exists); pass --force to rebuild", SENTINEL)
        return
    if STAGED.exists():
        logger.info("removing existing staged/ tree")
        shutil.rmtree(STAGED)
    STAGED.mkdir(parents=True)

    _stage_sources()

    logger.info("applying patch series")
    for patch_rel, root_rel in PATCH_MANIFEST:
        _apply(patch_rel, root_rel)

    _apply_smds_mesh_vtk_alloc()
    _apply_source_edits()
    SENTINEL.write_text("prepared\n", encoding="utf-8")
    logger.info("done: staged tree ready at %s", STAGED)


def main() -> int:
    """Parse the command line and run :func:`prepare`; return the process exit code."""
    logging.basicConfig(level=logging.INFO, format="[prepare] %(message)s")
    parser = argparse.ArgumentParser(description="Stage and patch SMESH sources.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild staged/ even if the sentinel exists",
    )
    args = parser.parse_args()
    try:
        prepare(force=args.force)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("ERROR: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
