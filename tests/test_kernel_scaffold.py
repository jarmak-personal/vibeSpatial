from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_kernel_scaffold import (
    KERNEL_INVENTORY_PATH,
    VARIANT_MANIFEST_PATH,
    KernelSpec,
    check_files,
    expected_files,
    write_files,
)


def test_expected_files_include_manifest_inventory_and_stubs(tmp_path: Path) -> None:
    spec = KernelSpec(name="point_bounds")
    files = expected_files(tmp_path, spec)

    assert tmp_path / spec.source_file in files
    assert tmp_path / spec.test_file in files
    assert tmp_path / spec.benchmark_file in files
    assert tmp_path / VARIANT_MANIFEST_PATH in files
    assert tmp_path / KERNEL_INVENTORY_PATH in files


def test_write_is_idempotent(tmp_path: Path) -> None:
    spec = KernelSpec(name="point_bounds")
    files = expected_files(tmp_path, spec)

    first = write_files(files)
    second = write_files(expected_files(tmp_path, spec))

    assert first
    assert second == []


def test_check_detects_missing_file(tmp_path: Path) -> None:
    spec = KernelSpec(name="point_bounds")
    files = expected_files(tmp_path, spec)

    mismatches = check_files(files, tmp_path)
    assert any("is missing" in mismatch for mismatch in mismatches)


def test_manifest_entry_is_stable(tmp_path: Path) -> None:
    spec = KernelSpec(name="point_bounds")
    write_files(expected_files(tmp_path, spec))
    manifest = json.loads((tmp_path / VARIANT_MANIFEST_PATH).read_text(encoding="utf-8"))

    assert manifest == [
        {
            "geom_types": ["point", "polygon"],
            "kernel": "point_bounds",
            "module": "vibespatial.kernels.predicates",
            "tier": 4,
            "variants": ["cpu"],
        }
    ]
