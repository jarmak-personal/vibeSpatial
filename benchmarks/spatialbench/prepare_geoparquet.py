#!/usr/bin/env python3
"""Convert legacy SpatialBench WKB Parquet tables to GeoParquet 1.1.

Preparation is intentionally outside benchmark timing. Each source shard maps
to one output shard so query implementations can retain bounded streaming. The
default output uses native GeoArrow. ``--preserve-wkb`` instead performs a
device-native metadata-only transcode and preserves every source WKB byte.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq

import geopandas as gpd

GEOMETRY_COLUMNS = {
    "building": ("b_boundary",),
    "trip": ("t_pickuploc", "t_dropoffloc"),
    "zone": ("z_boundary",),
}


def _convert_file_native_geoarrow(
    source: Path,
    target: Path,
    geometry_columns: tuple[str, ...],
) -> int:
    table = pq.read_table(source)
    frame = table.to_pandas()
    for column in geometry_columns:
        frame[column] = gpd.GeoSeries.from_wkb(frame[column], crs=None)
    spatial = gpd.GeoDataFrame(frame, geometry=geometry_columns[0], crs=None)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp-{os.getpid()}")
    spatial.to_parquet(
        temporary,
        geometry_encoding="geoarrow",
        schema_version="1.1.0",
        compression="zstd",
        index=False,
    )
    output_rows = pq.ParquetFile(temporary).metadata.num_rows
    if output_rows != table.num_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"row-count mismatch for {source}: {table.num_rows} != {output_rows}"
        )
    temporary.replace(target)
    return int(output_rows)


def _convert_file_preserved_wkb(
    source: Path,
    target: Path,
    geometry_columns: tuple[str, ...],
) -> tuple[int, dict[str, object]]:
    from vibespatial.io.geoparquet import (
        transcode_legacy_wkb_parquet_to_geoparquet,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    source_file = pq.ParquetFile(source)
    row_group_size = max(
        (
            int(source_file.metadata.row_group(index).num_rows)
            for index in range(source_file.metadata.num_row_groups)
        ),
        default=None,
    )
    result = transcode_legacy_wkb_parquet_to_geoparquet(
        source,
        target,
        geometry_columns={column: {"crs": None} for column in geometry_columns},
        primary_geometry=geometry_columns[0],
        compression="zstd",
        row_group_size=row_group_size,
    )
    return result.row_count, {
        "source_bytes": result.source_bytes,
        "output_bytes": result.output_bytes,
        "backend": result.backend,
        "schema_validated": result.schema_validated,
        "values_validated": result.values_validated,
        "atomic_publication": result.atomic_publication,
    }


def prepare(
    input_root: Path,
    output_root: Path,
    *,
    force: bool,
    preserve_wkb: bool = False,
) -> dict:
    manifest = {
        "format": "GeoParquet 1.1",
        "geometry_encoding": "WKB" if preserve_wkb else "native GeoArrow",
        "payload_policy": (
            "device-native metadata-only byte-preserving transcode"
            if preserve_wkb
            else "host correctness conversion to native GeoArrow"
        ),
        "source": str(input_root.resolve()),
        "tables": {},
    }
    for table_name, geometry_columns in GEOMETRY_COLUMNS.items():
        source_dir = input_root / table_name
        files = sorted(source_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no Parquet shards found in {source_dir}")
        rows = 0
        source_bytes = 0
        output_bytes = 0
        validation = {
            "schema_validated": True,
            "values_validated": True,
            "atomic_publication": True,
        }
        started = perf_counter()
        for position, source in enumerate(files, start=1):
            target = output_root / table_name / source.name
            if target.exists() and not force:
                output_rows = pq.ParquetFile(target).metadata.num_rows
                file_evidence = {
                    "source_bytes": source.stat().st_size,
                    "output_bytes": target.stat().st_size,
                }
            elif preserve_wkb:
                output_rows, file_evidence = _convert_file_preserved_wkb(
                    source,
                    target,
                    geometry_columns,
                )
            else:
                output_rows = _convert_file_native_geoarrow(
                    source,
                    target,
                    geometry_columns,
                )
                file_evidence = {
                    "source_bytes": source.stat().st_size,
                    "output_bytes": target.stat().st_size,
                }
            rows += int(output_rows)
            source_bytes += int(file_evidence["source_bytes"])
            output_bytes += int(file_evidence["output_bytes"])
            for key in validation:
                if key in file_evidence:
                    validation[key] = validation[key] and bool(file_evidence[key])
            print(
                f"[{table_name} {position}/{len(files)}] {source.name}: "
                f"{output_rows:,} rows",
                flush=True,
            )
        manifest["tables"][table_name] = {
            "files": len(files),
            "rows": rows,
            "geometry_columns": list(geometry_columns),
            "source_bytes": source_bytes,
            "output_bytes": output_bytes,
            **(validation if preserve_wkb else {}),
            "elapsed_seconds": perf_counter() - started,
        }

    # Non-spatial dimensions are safe to reference directly. Record their
    # source locations rather than duplicating bytes in the prepared dataset.
    for table_name in ("customer", "driver", "vehicle"):
        source_dir = input_root / table_name
        target = output_root / table_name
        if not target.exists():
            target.symlink_to(source_dir.resolve(), target_is_directory=True)

    manifest_path = output_root / "geoparquet_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing converted shards after validating the replacement",
    )
    parser.add_argument(
        "--preserve-wkb",
        action="store_true",
        help=(
            "publish WKB GeoParquet with exact source payload bytes using the "
            "device-native metadata-only transcode"
        ),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = prepare(
        args.input_root,
        args.output_root,
        force=args.force,
        preserve_wkb=args.preserve_wkb,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
