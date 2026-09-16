"""Count pySMESH's public entities and fail loudly if any of them lacks a docstring.

``docs/documentation/reference/index.md`` states how many public entities the library has
and how many carry a complete docstring. That sentence was hand-maintained, so it drifted:
nothing recomputed it when the surface grew. This script is the definition it should always
have had, so the number in the page is re-derivable rather than remembered.

**What counts as a public entity.** A name a consumer reaches through the documented API:

1. every public class defined at module level under ``src/pysmesh/``;
2. every public module-level function;
3. every public method and property of any class in those modules.

Point 3 deliberately includes the methods of the *private* mixin bases — ``_QueryOps``,
``_BooleanOps``, ``_MeshOps`` and the rest. ``Session`` and ``Mesher`` are assembled from
them, so ``Session.distance`` is public API even though it is defined on ``_QueryOps``.
That is also exactly the set ``mkdocs.yml``'s ``inherited_members: true`` renders, so the
count and the generated reference describe the same surface.

**What does not count.** Anything whose name starts with an underscore; dataclass fields
and enum members, which are documented in their owner's ``Attributes:`` section rather than
on their own; and ``src/pysmesh/_core.pyi``, which is a typed stub with no docstrings by
design — the reference page cites it separately, as the native extension's typed surface.

Exits non-zero when a public entity has no docstring, naming each one.

Usage:
    python ci/count_documented.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE = _ROOT / "src" / "pysmesh"

_FuncDef = (ast.FunctionDef, ast.AsyncFunctionDef)


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _entities(path: Path) -> list[tuple[str, bool]]:
    """Every public entity in one module, as (qualified name, has docstring)."""
    module = path.relative_to(_PACKAGE).with_suffix("").as_posix().replace("/", ".")
    module = module.removesuffix(".__init__")
    tree = ast.parse(path.read_text(encoding="utf-8"))

    found: list[tuple[str, bool]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if _is_public(node.name):
                found.append((f"{module}.{node.name}", ast.get_docstring(node) is not None))
            # A private mixin contributes no entry of its own, but its public methods are
            # the public API of the class assembled from it.
            for member in node.body:
                if isinstance(member, _FuncDef) and _is_public(member.name):
                    found.append(
                        (
                            f"{module}.{node.name}.{member.name}",
                            ast.get_docstring(member) is not None,
                        )
                    )
        elif isinstance(node, _FuncDef) and _is_public(node.name):
            found.append((f"{module}.{node.name}", ast.get_docstring(node) is not None))
    return found


def main() -> None:
    if not _PACKAGE.is_dir():
        raise SystemExit(f"package not found: {_PACKAGE}")

    found: list[tuple[str, bool]] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        found.extend(_entities(path))

    total = len(found)
    undocumented = sorted(name for name, has_doc in found if not has_doc)
    documented = total - len(undocumented)

    print(f"public entities : {total}")
    print(f"documented      : {documented}")

    if undocumented:
        print(f"undocumented    : {len(undocumented)}")
        for name in undocumented:
            print(f"  {name}")
        raise SystemExit(
            f"{len(undocumented)} public entities carry no docstring; "
            "document them or make them private."
        )

    print("every public entity carries a docstring")


if __name__ == "__main__":
    main()
    sys.exit(0)
