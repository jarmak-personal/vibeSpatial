from __future__ import annotations

import argparse
import shutil
from pathlib import Path

COPY_TARGETS = (
    ("geopandas/conftest.py", "tests/upstream/geopandas/conftest.py"),
    ("geopandas/tests", "tests/upstream/geopandas/tests"),
    ("geopandas/io/tests", "tests/upstream/geopandas/io/tests"),
    ("geopandas/tools/tests", "tests/upstream/geopandas/tools/tests"),
)

PACKAGE_MARKERS = (
    "tests/__init__.py",
    "tests/upstream/__init__.py",
    "tests/upstream/geopandas/__init__.py",
    "tests/upstream/geopandas/io/__init__.py",
    "tests/upstream/geopandas/tools/__init__.py",
)

IMPORT_REWRITES = {
    "from geopandas.tests.util import": "from tests.upstream.geopandas.tests.util import",
}


def copy_path(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def ensure_packages(repo_root: Path) -> None:
    for relative_path in PACKAGE_MARKERS:
        path = repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


def rewrite_imports(vendor_root: Path) -> None:
    for path in vendor_root.rglob("*.py"):
        text = path.read_text()
        updated = text
        for old, new in IMPORT_REWRITES.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor the GeoPandas test suite into this repo.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Path to the local geopandas repository. Defaults to ../geopandas",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_root = (args.source_root or (repo_root.parent / "geopandas")).resolve()

    for source_relative, destination_relative in COPY_TARGETS:
        source = source_root / source_relative
        destination = repo_root / destination_relative
        if not source.exists():
            raise FileNotFoundError(f"Missing upstream path: {source}")
        copy_path(source, destination)

    ensure_packages(repo_root)
    rewrite_imports(repo_root / "tests/upstream/geopandas")

    print(f"Vendored GeoPandas tests from {source_root}")


if __name__ == "__main__":
    main()
