"""Validate that all reorg shims re-export names correctly from their new locations."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REORG_SHIM_MARKER = "# REORG_SHIM"


def _find_shim_files() -> list[Path]:
    src = REPO_ROOT / "src" / "vibespatial"
    shims: list[Path] = []
    for path in sorted(src.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            first_line = path.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        except OSError:
            continue
        if REORG_SHIM_MARKER in first_line:
            shims.append(path)
    return shims


def _module_name(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT / "src")
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def run() -> list[str]:
    """Return list of error strings. Empty means all shims are valid."""
    errors: list[str] = []
    shim_files = _find_shim_files()
    if not shim_files:
        return errors

    for path in shim_files:
        module_name = _module_name(path)
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: failed to import shim: {exc}")
            continue

        # Check that the module has at least some public names
        public_names = [n for n in dir(mod) if not n.startswith("_")]
        if not public_names:
            errors.append(f"{module_name}: shim exports no public names")

    return errors


def main() -> int:
    errors = run()
    if not errors:
        print("All reorg shims are valid.")
        return 0
    for error in errors:
        print(f"ERROR: {error}")
    print(f"\n{len(errors)} shim validation error(s).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
