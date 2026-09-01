#!/usr/bin/env python3
"""Inventory SpatialBench WKB structure and byte order with the GPU scanner.

The input must be one exact scale directory. Every geometry column in every
Parquet shard is scanned row-group by row-group; geometry payload bytes never
leave device memory. Only bounded integer aggregates are published as JSON.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from prepare_geoparquet import GEOMETRY_COLUMNS

from vibespatial.io.geoparquet import _read_geoparquet_table_with_pylibcudf
from vibespatial.io.pylibcudf import (
    _pylibcudf_validity_mask,
    _pylibcudf_wkb_offsets,
    _pylibcudf_wkb_payload,
)
from vibespatial.io.wkb_decode_status import WKB_STATUS_REASONS
from vibespatial.kernels.core.wkb_decode import (
    scan_wkb_device_structural_plan,
    summarize_wkb_device_plan,
)

_COUNT_KEYS = (
    "rows",
    "payload_bytes",
    "native_little_endian_rows",
    "native_big_endian_rows",
    "native_mixed_endian_rows",
    "null_rows",
    "declined_rows",
    "part_count",
    "ring_count",
    "coordinate_count",
)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _empty_summary() -> dict[str, Any]:
    return {
        **{key: 0 for key in _COUNT_KEYS},
        "family_counts": {
            family: 0
            for family in (
                "point",
                "linestring",
                "polygon",
                "multipoint",
                "multilinestring",
                "multipolygon",
            )
        },
    }


def _merge_summary(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in _COUNT_KEYS:
        target[key] += int(source[key])
    for family, count in source["family_counts"].items():
        target["family_counts"][family] += int(count)


def _scan_column(path: Path, column_name: str) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    summary = _empty_summary()
    for row_group in range(parquet_file.metadata.num_row_groups):
        table = _read_geoparquet_table_with_pylibcudf(
            path,
            columns=[column_name],
            row_groups=[row_group],
        )
        column = table.columns()[0]
        offsets = _pylibcudf_wkb_offsets(column)
        payload = _pylibcudf_wkb_payload(column)
        validity = _pylibcudf_validity_mask(column)
        plan = scan_wkb_device_structural_plan(
            payload,
            offsets,
            int(column.size()),
            validity_device=validity,
        )
        _merge_summary(summary, summarize_wkb_device_plan(plan))
    return summary


def inventory(input_root: Path) -> dict[str, Any]:
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)

    output: dict[str, Any] = {
        "schema_version": 1,
        "kind": "spatialbench_gpu_wkb_inventory",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(input_root.resolve()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            package: _package_version(distribution)
            for package, distribution in {
                "vibespatial": "vibespatial",
                "cupy": "cupy-cuda12x",
                "pylibcudf": "pylibcudf-cu12",
                "pyarrow": "pyarrow",
            }.items()
        },
        "status_reasons": {
            str(int(status)): reason for status, reason in WKB_STATUS_REASONS.items()
        },
        "tables": {},
        "aggregate": _empty_summary(),
    }

    for table_name, geometry_columns in GEOMETRY_COLUMNS.items():
        source_dir = input_root / table_name
        files = sorted(source_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no Parquet shards found in {source_dir}")
        table_summary: dict[str, Any] = {
            "geometry_columns": list(geometry_columns),
            "files": [],
            "aggregate": _empty_summary(),
        }
        for position, path in enumerate(files, start=1):
            parquet_file = pq.ParquetFile(path)
            file_summary: dict[str, Any] = {
                "path": str(path.relative_to(input_root)),
                "size_bytes": path.stat().st_size,
                "rows": int(parquet_file.metadata.num_rows),
                "row_groups": int(parquet_file.metadata.num_row_groups),
                "schema": str(parquet_file.schema_arrow),
                "columns": {},
            }
            for column_name in geometry_columns:
                column_summary = _scan_column(path, column_name)
                file_summary["columns"][column_name] = column_summary
                _merge_summary(table_summary["aggregate"], column_summary)
                _merge_summary(output["aggregate"], column_summary)
            table_summary["files"].append(file_summary)
            print(
                f"[{table_name} {position}/{len(files)}] {path.name}: "
                f"{file_summary['rows']:,} rows",
                flush=True,
            )
        output["tables"][table_name] = table_summary
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
