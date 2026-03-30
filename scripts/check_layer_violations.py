"""Enforce import layering in vibespatial.

Layer contract
--------------
Layer 0: cuda/             -- no internal vibespatial deps (standalone)
Layer 1: runtime/          -- may import cuda only
Layer 2: geometry/         -- may import cuda, runtime
Layer 3: kernels/core/     -- may import cuda, runtime, geometry
Layer 4: constructive/, spatial/, predicates/, overlay/,
         kernels/constructive/, kernels/predicates/
                           -- may import layers 0-3 + each other
Layer 5: io/               -- may import layers 0-4
Layer 6: api/              -- may import everything
Layer 7: bench/, testing/  -- may import everything (dev only)

Only TOP-LEVEL imports are checked.  Function-body imports and
``if TYPE_CHECKING:`` blocks are ignored (they do not execute at
import time).

Usage::

    python scripts/check_layer_violations.py
    python scripts/check_layer_violations.py --verbose

Exit code 0 means clean; exit code 1 means violations found.
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "vibespatial"
PACKAGE_NAME = "vibespatial"

# ── Layer definitions ────────────────────────────────────────────────
# Order matters: longest prefix wins during classification.  The tuple
# format is (directory prefix relative to src/vibespatial, layer number).
LAYER_MAP: list[tuple[str, int]] = [
    ("cuda/", 0),
    ("runtime/", 1),
    ("geometry/", 2),
    ("kernels/core/", 3),
    ("kernels/constructive/", 4),
    ("kernels/predicates/", 4),
    ("constructive/", 4),
    ("spatial/", 4),
    ("predicates/", 4),
    ("overlay/", 4),
    ("io/", 5),
    ("api/", 6),
    ("bench/", 7),
    ("testing/", 7),
]

# Same map expressed as module-path prefixes (for resolving import targets).
# Derived from LAYER_MAP: "cuda/" -> "vibespatial.cuda."
MODULE_LAYER_MAP: list[tuple[str, int]] = [
    (f"{PACKAGE_NAME}.{prefix.rstrip('/').replace('/', '.')}.", layer)
    for prefix, layer in LAYER_MAP
]

# Known pre-existing violations that are baselined.  These are tracked
# debt -- the baseline MUST only shrink over time.
#
# Format: (relative path from src/vibespatial, imported module prefix)
_BASELINE_VIOLATIONS: set[tuple[str, str]] = {
    # kernels/core/wkb_decode.py uses io/pylibcudf helpers for device
    # buffer assembly.  Tight coupling tracked as tech-debt.
    ("kernels/core/wkb_decode.py", "vibespatial.io.pylibcudf"),
}


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    imported_module: str
    source_layer: int
    target_layer: int

    def render(self) -> str:
        return (
            f"VIOLATION: {self.file}:{self.line} imports "
            f"{self.imported_module} (layer {self.source_layer} "
            f"\u2192 layer {self.target_layer})"
        )


# ── Helpers ──────────────────────────────────────────────────────────

def _classify_file(rel_path: str) -> int | None:
    """Map a relative file path (from src/vibespatial) to its layer number.

    Returns None for files that are not in any layer (e.g. the package
    root __init__.py).
    """
    # Normalise to forward slashes for cross-platform safety.
    rel_path = rel_path.replace("\\", "/")
    best_layer: int | None = None
    best_prefix_len = 0
    for prefix, layer in LAYER_MAP:
        if rel_path.startswith(prefix) and len(prefix) > best_prefix_len:
            best_layer = layer
            best_prefix_len = len(prefix)
    return best_layer


def _classify_module(module: str) -> int | None:
    """Map a fully-qualified module path to its layer number.

    Returns None if the module is external or not mappable to a layer
    (e.g. ``vibespatial`` itself without a subpackage).
    """
    best_layer: int | None = None
    best_prefix_len = 0
    for prefix, layer in MODULE_LAYER_MAP:
        if module.startswith(prefix) and len(prefix) > best_prefix_len:
            best_layer = layer
            best_prefix_len = len(prefix)
    # Handle exact-match for the subpackage __init__ (e.g. "vibespatial.cuda")
    if best_layer is None:
        for prefix, layer in MODULE_LAYER_MAP:
            mod_prefix = prefix.rstrip(".")
            if module == mod_prefix:
                best_layer = layer
                break
    return best_layer


def _resolve_relative_import(
    module: str | None, level: int, file_rel_path: str,
) -> str | None:
    """Resolve a relative import to an absolute module path.

    ``level`` is the number of leading dots.  ``module`` is the dotted
    name after the dots (may be None for ``from . import X``).
    ``file_rel_path`` is the importing file relative to src/vibespatial.
    """
    # Build the package path for the importing file.
    parts = file_rel_path.replace("\\", "/").split("/")
    # Remove filename to get package parts.
    package_parts = [PACKAGE_NAME] + parts[:-1]

    # Go up ``level`` packages.
    if level > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - level + 1]

    base = ".".join(base_parts)
    if module:
        return f"{base}.{module}"
    return base


def _is_type_checking_guard(node: ast.If) -> bool:
    """Return True if the ``if`` node is ``if TYPE_CHECKING:``."""
    test = node.test
    # Plain ``TYPE_CHECKING``
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    # ``typing.TYPE_CHECKING``
    if (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    ):
        return True
    return False


def _extract_top_level_imports(
    tree: ast.Module, file_rel_path: str,
) -> list[tuple[int, str]]:
    """Extract (line_number, absolute_module) for top-level imports.

    Skips:
    - Imports inside function/method/class bodies
    - Imports inside ``if TYPE_CHECKING:`` blocks
    """
    results: list[tuple[int, str]] = []

    for node in tree.body:
        # Skip TYPE_CHECKING guards.
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            continue

        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((node.lineno, alias.name))

        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import -- resolve to absolute.
                resolved = _resolve_relative_import(
                    node.module, node.level, file_rel_path,
                )
                if resolved:
                    results.append((node.lineno, resolved))
            elif node.module:
                results.append((node.lineno, node.module))

        # Handle try/except blocks at top level (common pattern for
        # optional imports).
        elif isinstance(node, (ast.Try,)):
            for handler_body in [node.body, node.orelse, node.finalbody]:
                for child in handler_body:
                    if isinstance(child, ast.Import):
                        for alias in child.names:
                            results.append((child.lineno, alias.name))
                    elif isinstance(child, ast.ImportFrom):
                        if child.level and child.level > 0:
                            resolved = _resolve_relative_import(
                                child.module, child.level, file_rel_path,
                            )
                            if resolved:
                                results.append((child.lineno, resolved))
                        elif child.module:
                            results.append((child.lineno, child.module))
            for handler in node.handlers:
                for child in handler.body:
                    if isinstance(child, ast.Import):
                        for alias in child.names:
                            results.append((child.lineno, alias.name))
                    elif isinstance(child, ast.ImportFrom):
                        if child.level and child.level > 0:
                            resolved = _resolve_relative_import(
                                child.module, child.level, file_rel_path,
                            )
                            if resolved:
                                results.append((child.lineno, resolved))
                        elif child.module:
                            results.append((child.lineno, child.module))

    return results


def _is_baselined(file_rel: str, module: str) -> bool:
    """Return True if this violation is in the known baseline."""
    for bl_file, bl_prefix in _BASELINE_VIOLATIONS:
        if file_rel == bl_file and module.startswith(bl_prefix):
            return True
    return False


# ── Main scan ────────────────────────────────────────────────────────

def scan_violations(verbose: bool = False) -> list[Violation]:
    """Scan all Python files in src/vibespatial for layer violations."""
    violations: list[Violation] = []

    if not SRC_ROOT.exists():
        print(f"ERROR: source root not found: {SRC_ROOT}", file=sys.stderr)
        return violations

    python_files = sorted(
        p for p in SRC_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    )

    for path in python_files:
        rel_path = str(path.relative_to(SRC_ROOT))
        source_layer = _classify_file(rel_path)
        if source_layer is None:
            # File is in the package root (e.g. __init__.py) -- not in
            # any layer, skip.
            if verbose:
                print(f"  SKIP (no layer): {rel_path}")
            continue

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            if verbose:
                print(f"  SKIP (syntax error): {rel_path}: {exc}")
            continue

        imports = _extract_top_level_imports(tree, rel_path)

        for lineno, module in imports:
            # Only check internal vibespatial imports.
            if not module.startswith(f"{PACKAGE_NAME}."):
                continue

            target_layer = _classify_module(module)
            if target_layer is None:
                # Import of vibespatial itself or an unmapped sub-module.
                if verbose:
                    print(
                        f"  SKIP (unmapped target): {rel_path}:{lineno} "
                        f"-> {module}"
                    )
                continue

            if target_layer > source_layer:
                # Upward import -- potential violation.
                if _is_baselined(rel_path, module):
                    if verbose:
                        print(
                            f"  BASELINED: {rel_path}:{lineno} imports "
                            f"{module} (layer {source_layer} -> "
                            f"layer {target_layer})"
                        )
                    continue

                violations.append(
                    Violation(
                        file=rel_path,
                        line=lineno,
                        imported_module=module,
                        source_layer=source_layer,
                        target_layer=target_layer,
                    )
                )
            elif verbose:
                print(
                    f"  OK: {rel_path}:{lineno} imports {module} "
                    f"(layer {source_layer} -> layer {target_layer})"
                )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check vibespatial import layering violations.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print all import classifications, not just violations.",
    )
    args = parser.parse_args()

    violations = scan_violations(verbose=args.verbose)

    if violations:
        print(f"\nFound {len(violations)} layer violation(s):\n")
        for v in violations:
            print(f"  {v.render()}")
        print(
            "\nLayer contract: lower-numbered layers must not import from "
            "higher-numbered layers at module scope."
        )
        print(
            "Fix: move the import into a function body (lazy import) or "
            "restructure the dependency."
        )
        return 1

    print("No layer violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
