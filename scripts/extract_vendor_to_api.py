#!/usr/bin/env python3
"""Phase 1: Extract vendor tree into src/vibespatial/api/.

Copies all Python files from src/vibespatial/_vendor/geopandas/ into
src/vibespatial/api/, rewriting internal imports from the vendor namespace
to the new api namespace.

This script is idempotent -- it overwrites the target directory.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = REPO_ROOT / "src" / "vibespatial" / "_vendor" / "geopandas"
API_ROOT = REPO_ROOT / "src" / "vibespatial" / "api"

# Files to skip -- not needed in the extracted copy
SKIP_NAMES = {"__pycache__", "conftest.py"}

# Rename map: vendor filename -> api filename
RENAME_MAP = {
    "array.py": "geometry_array.py",
    "base.py": "geo_base.py",
}

# Import rewriting patterns (applied in order)
REWRITE_RULES = [
    # Fully-qualified vendor imports -> api imports
    (
        re.compile(r"from vibespatial\._vendor\.geopandas\.array\b"),
        "from vibespatial.api.geometry_array",
    ),
    (
        re.compile(r"from vibespatial\._vendor\.geopandas\.base\b"),
        "from vibespatial.api.geo_base",
    ),
    (
        re.compile(r"from vibespatial\._vendor\.geopandas\b"),
        "from vibespatial.api",
    ),
    (
        re.compile(r"import vibespatial\._vendor\.geopandas\.array\b"),
        "import vibespatial.api.geometry_array",
    ),
    (
        re.compile(r"import vibespatial\._vendor\.geopandas\.base\b"),
        "import vibespatial.api.geo_base",
    ),
    (
        re.compile(r"import vibespatial\._vendor\.geopandas\b"),
        "import vibespatial.api",
    ),
    # Relative imports within vendor -> absolute api imports.
    # These must come after the fully-qualified rules.
    # Handle renamed modules first.
    (re.compile(r"from \.array\b"), "from vibespatial.api.geometry_array"),
    (re.compile(r"from \.base\b"), "from vibespatial.api.geo_base"),
    # Then general relative imports.
    (re.compile(r"from \.tools\b"), "from vibespatial.api.tools"),
    (re.compile(r"from \.io\b"), "from vibespatial.api.io"),
    (re.compile(r"from \.datasets\b"), "from vibespatial.api.datasets"),
    (re.compile(r"from \.(\w+)"), r"from vibespatial.api.\1"),
    (re.compile(r"from \.\. import"), "from vibespatial.api import"),
    (re.compile(r"from \.\.\b"), "from vibespatial.api"),
]

# Sub-package relative imports (within io/, tools/, datasets/)
SUBPKG_REWRITE_RULES = {
    "io": [
        (re.compile(r"from \.\.\b"), "from vibespatial.api"),
        (re.compile(r"from \.(\w+)"), r"from vibespatial.api.io.\1"),
    ],
    "tools": [
        (re.compile(r"from \.\.\b"), "from vibespatial.api"),
        (re.compile(r"from \.(\w+)"), r"from vibespatial.api.tools.\1"),
    ],
    "datasets": [
        (re.compile(r"from \.\.\b"), "from vibespatial.api"),
        (re.compile(r"from \.(\w+)"), r"from vibespatial.api.datasets.\1"),
    ],
}


def rewrite_source(text: str, subpkg: str | None = None) -> str:
    """Rewrite import statements in a Python source string."""
    lines = text.split("\n")
    result = []
    for line in lines:
        # Apply sub-package rules first if applicable
        if subpkg and subpkg in SUBPKG_REWRITE_RULES:
            for pattern, replacement in SUBPKG_REWRITE_RULES[subpkg]:
                line = pattern.sub(replacement, line)
        # Then apply general rules
        for pattern, replacement in REWRITE_RULES:
            line = pattern.sub(replacement, line)
        result.append(line)
    return "\n".join(result)


def copy_file(src: Path, dst: Path, subpkg: str | None = None) -> None:
    """Copy a Python file, rewriting imports."""
    if src.suffix == ".py":
        text = src.read_text(encoding="utf-8")
        rewritten = rewrite_source(text, subpkg=subpkg)
        dst.write_text(rewritten, encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def copy_tree(src_dir: Path, dst_dir: Path, subpkg: str | None = None) -> None:
    """Recursively copy a directory, rewriting Python imports."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_dir.iterdir()):
        if item.name in SKIP_NAMES:
            continue
        dst_name = RENAME_MAP.get(item.name, item.name) if subpkg is None else item.name
        dst = dst_dir / dst_name
        if item.is_dir():
            copy_tree(item, dst, subpkg=item.name)
        elif item.is_file():
            copy_file(item, dst, subpkg=subpkg)


def main() -> None:
    # Clean target
    if API_ROOT.exists():
        shutil.rmtree(API_ROOT)

    # Copy and rewrite
    copy_tree(VENDOR_ROOT, API_ROOT)

    # Verify __init__.py exists
    init_path = API_ROOT / "__init__.py"
    if not init_path.exists():
        init_path.write_text("")

    # Report
    py_files = list(API_ROOT.rglob("*.py"))
    print(f"Extracted {len(py_files)} Python files to {API_ROOT}")


if __name__ == "__main__":
    main()
